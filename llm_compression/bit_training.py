"""Offline count estimation and held-out compression tuning for MCU models."""

import argparse
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import zlib

import numpy as np

from .bit_codec import compress_payload, decompress_payload, test_vectors
from .bit_model import (BOS, FNV_OFFSET, FNV_PRIME, PREFIX_ROWS, ROOT_ROWS, TOTAL,
                        BitModel, layout, save_model, validate_dimensions)
from .corpus import read_messages
from .training_corpus import load_prepared, reservoir, take_bytes


def split_corpus(path, *, text_field="body", validation_fraction=0.1, test_fraction=0.1,
                 split="chronological", seed=42, keep_duplicates=False, limit=None,
                 max_message_bytes=8192):
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    if split not in {"chronological", "random"}:
        raise ValueError("Unknown split strategy")
    if max_message_bytes < 0:
        raise ValueError("Maximum message bytes must be nonnegative; 0 means no limit")
    groups = Counter()
    input_rows = empty = oversized = 0
    for text in read_messages(path, text_field):
        if limit is not None and input_rows >= limit:
            break
        input_rows += 1
        data = text.encode("utf-8")
        if not data:
            empty += 1
        elif max_message_bytes and len(data) > max_message_bytes:
            oversized += 1
        else:
            groups[data] += 1
    unique = list(groups)
    if len(unique) < 10:
        raise ValueError("At least 10 distinct nonempty messages are needed for train/validation/test splits")
    if split == "random":
        random.Random(seed).shuffle(unique)
    validation_size = max(1, int(len(unique) * validation_fraction))
    test_size = max(1, int(len(unique) * test_fraction))
    train_size = len(unique) - validation_size - test_size
    if train_size < 1:
        raise ValueError("No training messages remain after splitting")
    partitions = {"train": unique[:train_size], "validation": unique[train_size:train_size + validation_size], "test": unique[train_size + validation_size:]}
    if keep_duplicates:
        partitions = {name: [text for text in texts for _ in range(groups[text])] for name, texts in partitions.items()}
    report = {"input_rows": input_rows, "unique_messages": len(unique), "empty_skipped": empty,
              "oversized_skipped": oversized, "duplicate_occurrences": sum(groups.values()) - len(groups),
              "duplicates_retained_within_splits": keep_duplicates, "split_strategy": split,
              "chronology_note": "Ordered by first occurrence in the input; identical text is assigned to one split only.",
              "text_field": text_field, "seed": seed,
              "splits": {name: {"messages": len(texts), "utf8_bytes": sum(map(len, texts))} for name, texts in partitions.items()}}
    return partitions, report


def batches(messages, target_bytes):
    batch, size = [], 0
    for message in messages:
        if batch and size + len(message) > target_bytes:
            yield batch
            batch, size = [], 0
        batch.append(message)
        size += len(message)
    if batch:
        yield batch


def inputs(messages):
    lengths = np.array([len(message) for message in messages], dtype=np.int64)
    data = np.frombuffer(b"".join(messages), dtype=np.uint8)
    starts = np.cumsum(lengths) - lengths
    positions = np.arange(len(data), dtype=np.int64) - np.repeat(starts, lengths)
    return data, positions


def context_hashes(data, positions, order):
    value = np.full(len(data), FNV_OFFSET, dtype=np.uint32)
    for lag in range(1, order + 1):
        previous = np.full(len(data), BOS, dtype=np.uint32)
        if lag < len(data):
            previous[lag:] = data[:-lag]
        previous[positions < lag] = BOS
        value = (value ^ previous) * np.uint32(FNV_PRIME)
        yield value


def indices(context, prefix, slots):
    value = context
    value = (value ^ (value >> np.uint32(16))) * np.uint32(0x7FEB352D)
    value = (value ^ (value >> np.uint32(15))) * np.uint32(0x846CA68B)
    bucket = (value ^ (value >> np.uint32(16))).astype(np.uint64) % slots
    # Prefix/row offsets can exceed uint32 for large tables. NumPy array indices
    # follow the host's address space, independently of the 32-bit hash itself.
    return (np.asarray(prefix, dtype=np.int64) - 1) * slots + bucket.astype(np.int64)


def prefixes(data, depth):
    return (data >> (8 - depth)).astype(np.uint32) | np.uint32(1 << depth)


def default_workers():
    """Keep automatic memory use modest and respect the process's CPU affinity."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return min(4, max(1, available))


def count_batch(batch, order, slots):
    """Private integer arrays per task: no shared writable NumPy state."""
    data, positions = inputs(batch)
    root = np.bincount(data, minlength=256).astype(np.uint64).reshape(1, 256)
    histograms = []
    for context in context_hashes(data, positions, order):
        buckets = indices(context, np.uint32(1), slots).astype(np.int64)
        rows = buckets * 256 + data
        histograms.append(np.bincount(rows, minlength=slots * 256).reshape(slots, 256).astype(np.uint64))
    return root, histograms, len(batch), len(data)


def batch_histograms(messages, order, slots, batch_bytes, workers):
    source = iter(batches(messages, batch_bytes))
    if workers == 1:
        for batch in source:
            yield count_batch(batch, order, slots)
        return
    # Executor.map eagerly consumes its iterable on Python 3.10–3.12. Instead,
    # keep at most two batches per worker submitted, including completed results.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="compression-training") as pool:
        pending = deque()
        try:
            for _ in range(2 * workers):
                batch = next(source, None)
                if batch is None:
                    break
                pending.append(pool.submit(count_batch, batch, order, slots))
            while pending:
                yield pending.popleft().result()
                batch = next(source, None)
                if batch is not None:
                    pending.append(pool.submit(count_batch, batch, order, slots))
        finally:
            for future in pending:
                future.cancel()


def train_counts(messages, order, slots, batch_bytes=1048576, progress=None, *, workers=1):
    validate_dimensions(order, slots)
    if type(workers) is not int or workers < 1:
        raise ValueError("Workers must be a positive integer")
    if batch_bytes <= 0:
        raise ValueError("Batch size must be positive")
    # Count whole next-byte histograms per context bucket, then project into the
    # bit-prefix tree once. This is exactly equivalent to eight bitwise passes.
    root_histogram = np.zeros((1, 256), dtype=np.uint64)
    histograms = [np.zeros((slots, 256), dtype=np.uint64) for _ in range(order)]
    message_count = byte_count = 0
    for root, partial, messages_in_batch, bytes_in_batch in batch_histograms(messages, order, slots, batch_bytes, workers):
        root_histogram += root
        for histogram, addition in zip(histograms, partial):
            histogram += addition
        message_count += messages_in_batch
        byte_count += bytes_in_batch
        if progress:
            progress(message_count, byte_count)
    counts = np.zeros((ROOT_ROWS + order * PREFIX_ROWS * slots, 2), dtype=np.uint64)
    for prefix in range(1, 256):
        depth = prefix.bit_length() - 1
        span = 1 << (7 - depth)
        start = (prefix - (1 << depth)) * span * 2
        for level, histogram in enumerate([root_histogram, *histograms]):
            rows = slice(prefix, prefix + 1) if level == 0 else slice(
                ROOT_ROWS + ((level - 1) * PREFIX_ROWS + prefix - 1) * slots,
                ROOT_ROWS + ((level - 1) * PREFIX_ROWS + prefix) * slots)
            counts[rows, 0] = histogram[:, start:start + span].sum(axis=1)
            counts[rows, 1] = histogram[:, start + span:start + 2 * span].sum(axis=1)
    # Preserve count ratios in uint16; inference uses the quantized counts too.
    scales = np.maximum(1, (counts.max(axis=1) + 65534) // 65535)
    quantized = ((counts + scales[:, None] // 2) // scales[:, None]).astype(np.uint16)
    stats = {"rows": len(counts), "occupied_rows": int(np.count_nonzero(counts.sum(axis=1))),
             "rescaled_rows": int(np.count_nonzero(scales > 1)), "count_storage_bits": 16,
             "messages": message_count, "utf8_bytes": byte_count}
    return quantized, stats


def evaluate(messages, counts, order, slots, strength, batch_bytes=1048576):
    nll = 0.0
    byte_count = characters = message_count = 0
    for batch in batches(messages, batch_bytes):
        data, positions = inputs(batch)
        byte_count += len(data)
        characters += sum(len(message.decode("utf-8")) for message in batch)
        message_count += len(batch)
        # Hold one probability vector per bit, then stream over context orders.
        # Retaining every context hash would cost O(order * batch_bytes) RAM.
        probabilities = np.empty((8, len(data)), dtype=np.uint64)
        for depth in range(8):
            prefix = prefixes(data, depth)
            root = counts[prefix].astype(np.uint64)
            probabilities[depth] = ((root[:, 1] + 1) * TOTAL) // (root.sum(axis=1) + 2)
        for level, context in enumerate(context_hashes(data, positions, order)):
            for depth in range(8):
                prefix = prefixes(data, depth)
                pair = counts[ROOT_ROWS + level * PREFIX_ROWS * slots + indices(context, prefix, slots)].astype(np.uint64)
                denominator = pair.sum(axis=1) + strength
                probabilities[depth] = (pair[:, 1] * TOTAL + strength * probabilities[depth] + denominator // 2) // denominator
        for depth in range(8):
            probability = np.clip(probabilities[depth], 1, TOTAL - 1)
            bit = (data >> (7 - depth)) & 1
            actual = np.where(bit, probability, TOTAL - probability)
            nll -= float(np.log2(actual.astype(np.float64) / TOTAL).sum())
    if not byte_count:
        raise ValueError("Cannot evaluate an empty split")
    return {"messages": message_count, "utf8_bytes": byte_count, "unicode_characters": characters,
            "predicted_bits": nll, "bits_per_utf8_byte": nll / byte_count,
            "bits_per_character": nll / characters}


def benchmark(messages, model, limit=1000, seed=42):
    if isinstance(messages, Sequence):
        selected = messages if limit == 0 or len(messages) <= limit else random.Random(seed).sample(messages, limit)
    else:
        selected = reservoir(messages, limit, seed)
    original = payload_bytes = framed_bytes = gzip_bytes = deflate_bytes = fits = count_messages = 0
    started = time.perf_counter()
    for message in selected:
        count_messages += 1
        payload = compress_payload(message, model)
        if decompress_payload(payload, len(message), model) != message:
            raise ValueError("Held-out message round-trip failed")
        count = len(message)
        varint_bytes = max(1, (count.bit_length() + 6) // 7)
        framed = len(payload) + 4 + varint_bytes  # nonce(2), byte count, CRC16(2).
        original += count
        payload_bytes += len(payload)
        framed_bytes += framed
        fits += framed <= 163
        gzip_bytes += len(gzip.compress(message, compresslevel=9, mtime=0))
        compressor = zlib.compressobj(level=9, wbits=-15)
        deflate_bytes += len(compressor.compress(message) + compressor.flush())
    if not original:
        raise ValueError("Cannot benchmark an empty split")
    return {"messages": count_messages, "original_utf8_bytes": original, "arithmetic_payload_bytes": payload_bytes,
            "minimal_framed_bytes": framed_bytes, "independent_gzip_bytes": gzip_bytes,
            "independent_raw_deflate_bytes": deflate_bytes,
            "payload_fraction_of_original": payload_bytes / original,
            "framed_fraction_of_original": framed_bytes / original,
            "fits_163_byte_datagram": fits, "all_round_trips_exact": True,
            "reference_python_encode_decode_seconds": time.perf_counter() - started,
            "framing_note": "Size estimate: nonce(2) + ULEB128 original byte count + CRC16(2) + arithmetic stream. Not the current Hugging Face MeshCore wire profile."}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args):
    slots, model_bytes = layout(args.model_kib, args.order)
    if args.batch_bytes <= 0 or args.benchmark_messages < 0 or args.evaluation_messages < 0 or (args.train_bytes is not None and args.train_bytes <= 0) or (args.limit_messages is not None and args.limit_messages <= 0) or args.max_message_bytes < 0:
        raise ValueError("Batch/limit sizes must be positive; benchmark limit may be zero for all test messages")
    if args.workers < 1:
        raise ValueError("Workers must be a positive integer")
    strengths = sorted(set(int(value) for value in args.strengths.split(",")))
    if not strengths or any(not 1 <= value <= 4096 for value in strengths):
        raise ValueError("Smoothing strengths must be integers between 1 and 4096")
    output = args.output or Path(f"models/mcu-{args.model_kib}k-order{args.order}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prepared = args.input.is_dir() or args.input.name == "manifest.json"
    if prepared:
        if args.keep_duplicates or args.limit_messages is not None:
            raise ValueError("Prepared splits already fix grouping; use --train-bytes for a development limit")
        partitions, corpus_report = load_prepared(args.input, max_message_bytes=args.max_message_bytes)
        input_path = Path(corpus_report["prepared_manifest"])
    else:
        partitions, corpus_report = split_corpus(
            args.input, text_field=args.text_field, validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction, split=args.split, seed=args.seed,
            keep_duplicates=args.keep_duplicates, limit=args.limit_messages, max_message_bytes=args.max_message_bytes,
        )
        input_path = args.input
    print(f"Training split: {len(partitions['train']):,} messages; model budget {args.model_kib} KiB, planned file at most {model_bytes:,} bytes.", file=sys.stderr)
    print(f"Counting with {args.workers} worker thread(s); at most {2 * args.workers if args.workers > 1 else 1} batches in flight.", file=sys.stderr)
    if args.train_bytes:
        print(f"Development training limit: {args.train_bytes:,} UTF-8 bytes (whole messages only).", file=sys.stderr)
    last_progress = started
    def progress(messages, byte_count):
        nonlocal last_progress
        now = time.perf_counter()
        if now - last_progress >= 10:
            print(f"Counted {messages:,} messages / {byte_count / 1048576:,.1f} MiB ({now - started:.1f}s)", file=sys.stderr, flush=True)
            last_progress = now
    training = take_bytes(partitions["train"], args.train_bytes) if args.train_bytes else partitions["train"]
    counts, count_stats = train_counts(training, args.order, slots, args.batch_bytes, progress=progress, workers=args.workers)
    if not count_stats["messages"]:
        raise ValueError("No training messages fit the configured byte limit")
    evaluation = {}
    for index, name in enumerate(("validation", "test")):
        print(f"Preparing {name} evaluation sample (limit {args.evaluation_messages or 'all'})…", file=sys.stderr, flush=True)
        # Reservoir sampling scans once but retains at most the configured limit.
        # For unlimited evaluation the source remains re-iterable on disk.
        evaluation[name] = reservoir(partitions[name], args.evaluation_messages, args.seed + index)
    validation = []
    for strength in strengths:
        result = evaluate(evaluation["validation"], counts, args.order, slots, strength, args.batch_bytes)
        validation.append({"strength": strength, **result})
        print(f"Validation strength {strength}: {result['bits_per_utf8_byte']:.4f} bits/UTF-8 byte", file=sys.stderr)
    best = min(validation, key=lambda result: (result["predicted_bits"], result["strength"]))
    baseline = {
        name: evaluate(evaluation[name], counts[:ROOT_ROWS], 0, 0, 1, args.batch_bytes)
        for name in ("validation", "test")
    }
    model_path = output / "model.mcblm"
    save_model(model_path, order=args.order, slots=slots, strength=best["strength"], counts=counts)
    test = evaluate(evaluation["test"], counts, args.order, slots, best["strength"], args.batch_bytes)
    with BitModel(model_path) as model:
        print(f"Benchmarking arithmetic round trips on up to {args.benchmark_messages or 'all'} test messages…", file=sys.stderr)
        actual = benchmark(evaluation["test"], model, args.benchmark_messages, args.seed)
        vectors = test_vectors(model)
        model_info = model.info()
    if not prepared:
        for name, messages in partitions.items():
            with (output / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
                for message in messages:
                    stream.write(json.dumps({"text": message.decode("utf-8")}, ensure_ascii=False) + "\n")
    report = {"model": model_info, "requested_model_kib": args.model_kib,
              "corpus": {"sha256": file_sha256(input_path), **corpus_report},
              "training_counts": count_stats, "validation_candidates": validation,
              "selected_strength": best["strength"], "test": test, "benchmark": actual,
              "unigram_baseline": baseline,
              "training_byte_limit": args.train_bytes,
              "parallelism": {"backend": "threads" if args.workers > 1 else "serial", "workers": args.workers,
                              "max_in_flight_batches": 2 * args.workers if args.workers > 1 else 1,
                              "batch_bytes": args.batch_bytes,
                              "scope": "training count estimation; input reading, evaluation sampling, validation, and arithmetic benchmarks remain sequential"},
              "evaluation": {"message_limit_per_split": args.evaluation_messages,
                             "sampling": "uniform reservoir without replacement; validation seed=seed, test seed=seed+1; 0 means all",
                             "test_messages": test["messages"], "validation_messages": best["messages"]},
              "training_seconds": time.perf_counter() - started,
              "architecture": "Fixed hashed bit-context statistical language model; offline frequency estimation, uint16 count quantization, validation-selected smoothing. No neural network or autoregressive text-generation API.",
              "deployment": "Binary tables plus Python integer reference and test vectors; a native MCU runtime port and hardware measurements are still required."}
    (output / "test-vectors.json").write_text(json.dumps(vectors, indent=2) + "\n", encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "model_bytes": model_info["model_bytes"], "test_bits_per_utf8_byte": test["bits_per_utf8_byte"],
                      "benchmark_payload_fraction": actual["payload_fraction_of_original"], "benchmark_framed_fraction": actual["framed_fraction_of_original"],
                      "round_trips_exact": True}, indent=2))
    return report


def build_parser(*, sweep=False):
    parser = argparse.ArgumentParser(description="Train a size-bounded integer bit-context language model for MCU text compression.")
    parser.add_argument("input", type=Path, nargs="?", default=Path("data/messages.jsonl"), help="JSONL/CSV corpus, or a prepared Wikipedia directory/manifest.json")
    parser.add_argument("--output", type=Path, help="default: models/mcu-SIZEk-orderN")
    if not sweep:
        parser.add_argument("--model-kib", type=int, default=256, help="maximum binary model size in KiB, including its header; no software upper limit")
        parser.add_argument("--order", type=int, default=4, help="previous-byte context depth; any nonnegative integer, no software upper limit")
    parser.add_argument("--strengths", default="4,16,64,256,1024,4096", help="validation smoothing candidates (comma-separated)")
    parser.add_argument("--text-field", choices=("body", "text"), default="body")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--keep-duplicates", action="store_true", help="keep repeated messages within their assigned split; exact duplicates never cross splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-messages", type=int)
    parser.add_argument("--max-message-bytes", type=int, default=8192, help="skip longer JSONL/CSV messages or require prepared samples to fit; 0 disables this size limit")
    parser.add_argument("--batch-bytes", type=int, default=1048576, help="training/evaluation vectorization batch target")
    parser.add_argument("--workers", type=int, default=default_workers(), help="training worker threads; default up to 4 available CPUs; 1 disables parallel counting")
    parser.add_argument("--train-bytes", type=int, help="development limit on training UTF-8 bytes; default trains on all data")
    parser.add_argument("--evaluation-messages", type=int, default=100000, help="uniform reservoir sample per held-out split; 0 evaluates all; default 100000")
    parser.add_argument("--benchmark-messages", type=int, default=1000, help="real arithmetic round trips on held-out messages; 0 means all")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train(args)
    except (MemoryError, OverflowError) as exc:
        print(f"error: model/context exceeds available memory or this runtime's address space: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
