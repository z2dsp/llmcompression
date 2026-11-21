"""Run with python -m llm_compression --help."""

import argparse
import json
from pathlib import Path
import sys
import time

from .codec import compress, decompress, unpack_archive
from .demo import SAMPLE, size_report
from .model import DEFAULT_CACHE, DEFAULT_MODEL, HuggingFacePredictor, runtime_signature


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face causal language model ID")
    parser.add_argument("--revision", help="model commit or branch (resolved to a commit in the archive)")
    parser.add_argument("--context-tokens", type=_positive, default=256, help="reset prediction context every N tokens (default: 256)")


def _runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="model download cache (default: project .cache/huggingface)")
    parser.add_argument("--offline", action="store_true", help="use cached model files only")
    parser.add_argument("--quiet", action="store_true", help="suppress progress on stderr")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Lossless text compression with a shared language model and arithmetic coding.")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="compress and decompress an English sample; compare sizes with gzip")
    _model_options(demo)
    _runtime_options(demo)
    demo.add_argument("--input", type=Path, help="use a UTF-8 file instead of the built-in sample")
    demo.add_argument("--trace", type=_nonnegative, default=12, metavar="N", help="show probabilities and bit costs of the first N tokens")
    demo.add_argument("--json", action="store_true", help="print a machine-readable report")
    enc = commands.add_parser("compress", help="compress a UTF-8 file into an .llmc archive")
    enc.add_argument("input", type=Path)
    enc.add_argument("output", type=Path)
    _model_options(enc)
    _runtime_options(enc)
    dec = commands.add_parser("decompress", help="restore an .llmc archive using its saved model and settings")
    dec.add_argument("input", type=Path)
    dec.add_argument("output", type=Path)
    _runtime_options(dec)
    info = commands.add_parser("info", help="inspect archive metadata without loading a model")
    info.add_argument("input", type=Path)
    return root


def _progress(stage: str, quiet: bool):
    last_update = time.monotonic()

    def update(done: int, total: int) -> None:
        nonlocal last_update
        now = time.monotonic()
        if not quiet and (done == total or now - last_update >= 2):
            print(f"{stage}: {done}/{total} tokens", file=sys.stderr)
            last_update = now

    return update


def _load(args, metadata: dict | None = None) -> HuggingFacePredictor:
    if not args.quiet:
        print("Loading language model on CPU (float32, one thread)...", file=sys.stderr)
    if metadata is None:
        return HuggingFacePredictor(args.model, args.revision, args.context_tokens, args.cache_dir, args.offline)
    required = {"model_id", "revision", "context_tokens", "runtime", "inference", "vocab_size", "start_token", "cdf_total"}
    if set(metadata) != required:
        raise ValueError("Unsupported predictor metadata")
    if metadata["runtime"] != runtime_signature():
        raise ValueError("Runtime differs from archive; restore the Python, library versions, and platform listed by 'info'")
    if not isinstance(metadata["model_id"], str) or not isinstance(metadata["revision"], str) or type(metadata["context_tokens"]) is not int:
        raise ValueError("Invalid model metadata")
    predictor = HuggingFacePredictor(
        metadata["model_id"], metadata["revision"], metadata["context_tokens"], args.cache_dir, args.offline
    )
    if predictor.metadata != metadata:
        raise ValueError("Loaded predictor differs from archive metadata")
    return predictor


def _print_report(report: dict, trace: list[dict]) -> None:
    print(f"{report['characters']:,} characters, {report['utf8_bytes']:,} UTF-8 bytes, {report['tokens']:,} tokens")
    print(f"{'Representation':<22} {'Bytes':>9} {'Bits/char':>11} {'% original':>12}")
    for name, item in report["sizes"].items():
        bpc = "n/a" if item["bits_per_character"] is None else f"{item['bits_per_character']:.3f}"
        fraction = "n/a" if item["fraction_of_original"] is None else f"{item['fraction_of_original'] * 100:.1f}%"
        print(f"{name:<22} {item['bytes']:>9,} {bpc:>11} {fraction:>12}")
    print(f"Predicted coding cost: {report['predicted_bits']:.2f} bits; payload includes termination and byte padding.")
    print(f"Archive overhead: {report['metadata_bytes']} bytes. Shared model weights are excluded.")
    print(f"Round-trip: exact ({report['compress_seconds']:.2f}s compress, {report['decompress_seconds']:.2f}s decompress)")
    if trace:
        print("\nFirst tokens (probability after integer quantization; bit cost = -log2(p)):")
        print(f"{'Token':<32} {'Probability':>12} {'Bits':>9}")
        for item in trace:
            label = repr(item["token"])
            if len(label) > 31:
                label = label[:28] + "..."
            print(f"{label:<32} {item['probability']:>12.6f} {item['bits']:>9.3f}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "info":
            archive = unpack_archive(args.input.read_bytes())
            print(json.dumps(archive.header, indent=2, sort_keys=True))
            return 0
        if args.command in {"compress", "decompress"} and args.output.exists():
            raise ValueError(f"Output already exists: {args.output}; choose a new output path")
        if args.command == "decompress":
            data = args.input.read_bytes()
            archive = unpack_archive(data)
            predictor = _load(args, archive.header["predictor"])
            restored = decompress(data, predictor, _progress("Decompressing", args.quiet))
            with args.output.open("xb") as output:
                output.write(restored)
            print(f"Restored {len(restored):,} bytes to {args.output}; SHA-256 verified.")
            return 0
        data = args.input.read_bytes() if args.input else SAMPLE.encode("utf-8")
        # Reject non-UTF-8 before spending time loading model weights.
        data.decode("utf-8")
        predictor = _load(args)
        started = time.perf_counter()
        result = compress(data, predictor, getattr(args, "trace", 0), _progress("Compressing", args.quiet))
        compress_seconds = time.perf_counter() - started
        if args.command == "compress":
            with args.output.open("xb") as output:
                output.write(result.archive)
            print(f"Compressed {len(data):,} bytes into {len(result.archive):,} bytes ({result.payload_bytes:,} payload) at {args.output}.")
            return 0
        started = time.perf_counter()
        restored = decompress(result.archive, predictor, _progress("Decompressing", args.quiet))
        if restored != data:
            raise ValueError("Round-trip verification failed")
        report = size_report(data, result)
        report.update({
            "compress_seconds": compress_seconds,
            "decompress_seconds": time.perf_counter() - started,
            "round_trip_exact": True,
            "predictor": predictor.metadata,
            "trace": result.trace,
        })
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_report(report, result.trace)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
