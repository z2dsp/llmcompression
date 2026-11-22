import json
import math
from pathlib import Path
import threading

import numpy as np
import pytest

from llm_compression.bit_codec import compress_payload, decompress_payload, pack_file, unpack_file
from llm_compression.bit_model import (BOS, FNV_OFFSET, FNV_PRIME, HEADER, MASK32,
                                      PREFIX_ROWS, ROOT_ROWS, TOTAL, BitModel,
                                      layout, mix32, save_model)
from llm_compression.bit_training import evaluate, main as train_main, split_corpus, train_counts
from llm_compression import bit_training


@pytest.fixture
def model_path(tmp_path):
    slots, _ = layout(8, 3)
    messages = [b"Hello mesh!", b"The weather is pleasant.", b"Hello world!"] * 10
    counts, _ = train_counts(messages, 3, slots, batch_bytes=29)
    path = tmp_path / "model.mcblm"
    save_model(path, order=3, slots=slots, strength=16, counts=counts)
    return path


@pytest.mark.parametrize("order", [0, 1, 4, 8])
def test_size_budget(order):
    slots, size = layout(64, order)
    assert size <= 64 * 1024
    assert slots > 0 if order else slots == 0


def test_vectorized_training_matches_independent_scalar_counts():
    messages = [b"ab", b"\x00ab", b"b", b"another message"]
    order, slots = 3, 17
    expected = np.zeros((ROOT_ROWS + order * PREFIX_ROWS * slots, 2), dtype=np.uint16)
    for message in messages:
        history = [BOS] * order
        for byte in message:
            context = []
            value = FNV_OFFSET
            for previous in history:
                value = ((value ^ previous) * FNV_PRIME) & MASK32
                context.append(value)
            prefix = 1
            for shift in range(7, -1, -1):
                bit = (byte >> shift) & 1
                expected[prefix, bit] += 1
                for level, value in enumerate(context):
                    row = ROOT_ROWS + (level * PREFIX_ROWS + prefix - 1) * slots + mix32(value) % slots
                    expected[row, bit] += 1
                prefix = (prefix << 1) | bit
            history = [byte] + history[:-1]
    actual, _ = train_counts(messages, order, slots, batch_bytes=4)
    assert np.array_equal(actual, expected)


def test_vectorized_loss_matches_exported_integer_runtime(model_path):
    messages = [b"Hello there", "Café 🌊".encode()]
    with BitModel(model_path) as model:
        counts = np.frombuffer(model_path.read_bytes()[HEADER.size:], dtype="<u2").reshape(-1, 2)
        predicted = evaluate(messages, counts, model.order, model.slots, model.strength, batch_bytes=5)
        bits = 0.0
        for message in messages:
            state = model.state()
            for byte in message:
                prefix = 1
                for shift in range(7, -1, -1):
                    bit = (byte >> shift) & 1
                    q = state.probability_one(prefix)
                    bits -= math.log2((q if bit else TOTAL - q) / TOTAL)
                    prefix = (prefix << 1) | bit
                state.observe_byte(byte)
        assert predicted["predicted_bits"] == pytest.approx(bits, abs=1e-9)


@pytest.mark.parametrize("data", [b"", b"Hello mesh!", bytes(range(256)), b"\x00" * 200, b"\r\n\t  ", "Café 日本語 🌊".encode()])
def test_exact_arithmetic_round_trips(model_path, data):
    with BitModel(model_path) as model:
        payload = compress_payload(data, model)
        assert decompress_payload(payload, len(data), model) == data
        assert unpack_file(pack_file(data, model), model) == data
        assert compress_payload(data, model) == payload


def test_corrupt_model_and_archive_detection(model_path, tmp_path):
    damaged = bytearray(model_path.read_bytes())
    damaged[-1] ^= 1
    other = tmp_path / "bad.mcblm"
    other.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum"):
        BitModel(other)
    with BitModel(model_path) as model:
        archive = bytearray(pack_file(b"Hello", model))
        archive[8] ^= 1
        with pytest.raises(ValueError, match="wrong shared model"):
            unpack_file(bytes(archive), model)
        payload = compress_payload(b"Hello", model)
        with pytest.raises(ValueError, match="noncanonical"):
            decompress_payload(payload + b"\x00", 5, model)
        with pytest.raises(ValueError, match="excessive"):
            decompress_payload(payload, 999, model, max_output_bytes=10)


def test_corpus_splits_group_duplicates_without_leakage(tmp_path):
    path = tmp_path / "messages.jsonl"
    with path.open("w") as output:
        for index in range(100):
            output.write(json.dumps({"body": f"message {index % 30}"}) + "\n")
    splits, report = split_corpus(path, keep_duplicates=True)
    assert report["unique_messages"] == 30
    assert report["duplicate_occurrences"] == 70
    assert sum(map(len, splits.values())) == 100
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["validation"]) & set(splits["test"])
    assert not set(splits["train"]) & set(splits["validation"])
    assert split_corpus(path, split="random", seed=7)[0] == split_corpus(path, split="random", seed=7)[0]


def test_training_batch_size_does_not_change_model():
    messages = [f"message {number}".encode() for number in range(100)]
    small, _ = train_counts(messages, 4, 37, batch_bytes=9)
    large, _ = train_counts(messages, 4, 37, batch_bytes=1000000)
    assert np.array_equal(small, large)


def test_large_counts_quantize_without_overflow():
    counts, stats = train_counts([b"a" * 140000], 0, 0)
    # Every bit of the repeated byte should remain almost certain after rescaling.
    assert counts.dtype == np.uint16
    assert stats["rescaled_rows"] == 8
    score = evaluate([b"a" * 100], counts, 0, 0, 1)
    assert score["bits_per_utf8_byte"] < 0.01


def test_training_command_exports_reproducible_working_model(tmp_path):
    corpus = tmp_path / "messages.jsonl"
    corpus.write_text("".join(json.dumps({"body": f"Hello mesh number {i}!"}) + "\n" for i in range(40)))
    common = [str(corpus), "--model-kib", "8", "--order", "2", "--strengths", "4,16", "--benchmark-messages", "0", "--batch-bytes", "29"]
    first, second = tmp_path / "first", tmp_path / "second"
    assert train_main(common + ["--output", str(first), "--workers", "1"]) == 0
    assert train_main(common + ["--output", str(second), "--workers", "3"]) == 0
    assert (first / "model.mcblm").read_bytes() == (second / "model.mcblm").read_bytes()
    report = json.loads((first / "report.json").read_text())
    assert report["model"]["model_bytes"] <= 8192
    assert report["selected_strength"] in (4, 16)
    assert report["benchmark"]["messages"] == 4
    assert report["benchmark"]["all_round_trips_exact"]
    assert report["parallelism"]["workers"] == 1
    assert json.loads((second / "report.json").read_text())["parallelism"]["workers"] == 3
    assert report["unigram_baseline"]["test"]["utf8_bytes"] == report["test"]["utf8_bytes"]
    with BitModel(first / "model.mcblm") as model:
        for vector in json.loads((first / "test-vectors.json").read_text())["vectors"]:
            assert decompress_payload(bytes.fromhex(vector["payload_hex"]), vector["byte_count"], model).hex() == vector["input_hex"]
    # Existing results cannot be overwritten accidentally.
    assert train_main(common + ["--output", str(first)]) == 1
    invalid = tmp_path / "invalid"
    assert train_main(common + ["--output", str(invalid), "--workers", "0"]) == 1
    assert not invalid.exists()


@pytest.mark.parametrize("order", [0, 4, 8])
@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_counts_match_serial_before_and_after_rescaling(order, workers):
    messages = [b"a" * 4096] * 64 + [bytes(range(256))] * 10 + ["Café 🌊".encode()]
    slots = 17 if order else 0
    expected, expected_stats = train_counts(messages, order, slots, batch_bytes=100000)
    progress = []
    actual, actual_stats = train_counts(iter(messages), order, slots, batch_bytes=8192, workers=workers,
                                       progress=lambda count, size: progress.append((count, size)))
    assert np.array_equal(actual, expected)
    assert actual_stats == expected_stats
    assert actual_stats["rescaled_rows"] > 0
    assert progress == sorted(progress)
    assert progress[-1] == (len(messages), sum(map(len, messages)))


def test_parallel_counting_runs_concurrently_in_distinct_threads(monkeypatch):
    messages = [bytes([value]) * 100 for value in range(12)]
    expected, _ = train_counts(messages, 2, 17, batch_bytes=100)
    original = bit_training.count_batch
    barrier = threading.Barrier(3)
    lock = threading.Lock()
    thread_ids = set()
    calls = 0
    def count(batch, order, slots):
        nonlocal calls
        with lock:
            index = calls
            calls += 1
            thread_ids.add(threading.get_ident())
        if index < 3:
            barrier.wait(timeout=5)
        return original(batch, order, slots)
    monkeypatch.setattr(bit_training, "count_batch", count)
    actual, _ = train_counts(messages, 2, 17, batch_bytes=100, workers=3)
    assert len(thread_ids) == 3
    assert np.array_equal(actual, expected)


def test_parallel_input_prefetch_is_bounded(monkeypatch):
    consumed = 0
    first_progress = []
    def source():
        nonlocal consumed
        for _ in range(10000):
            consumed += 1
            yield b"x" * 100
    def progress(*unused):
        if not first_progress:
            first_progress.append(consumed)
    _, stats = train_counts(source(), 0, 0, batch_bytes=10000, workers=3, progress=progress)
    # Six batches of 100 messages, plus one message of batcher lookahead.
    assert first_progress[0] <= 2 * 3 * 100 + 1
    assert stats["messages"] == 10000


def test_worker_errors_propagate_without_consuming_the_whole_source(monkeypatch):
    consumed = 0
    def source():
        nonlocal consumed
        for _ in range(10000):
            consumed += 1
            yield b"message"
    def fail(*args):
        raise ValueError("worker failed")
    monkeypatch.setattr(bit_training, "count_batch", fail)
    with pytest.raises(ValueError, match="worker failed"):
        train_counts(source(), 1, 17, batch_bytes=7, workers=2)
    assert consumed <= 5


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_worker_count_is_rejected(workers):
    with pytest.raises(ValueError, match="Workers"):
        train_counts([b"test"], 1, 17, workers=workers)
