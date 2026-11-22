import hashlib
import math
import weakref

import numpy as np
import pytest

from llm_compression.bit_codec import compress_payload, decompress_payload
from llm_compression.bit_model import (
    EXTENDED_MAGIC, MAGIC, ROOT_ROWS, PREFIX_ROWS, SCHEME, TOTAL,
    BitModel, decode_model_header, encode_model_prefix, encode_uint, layout, mix32, save_model,
)
from llm_compression import bit_training
from llm_compression.bit_training import evaluate, indices, main as train_main, split_corpus, train_counts
from llm_compression.training_corpus import EXTENDED_CORPUS_MAGIC, RecordWriter, read_records


@pytest.mark.parametrize("budget,order", [(65537, 16), (1048576, 64), (8 * 1024 * 1024, 32), (1 << 64, 1 << 40)])
def test_layout_has_no_fixed_size_or_depth_ceiling(budget, order):
    slots, size = layout(budget, order)
    assert slots >= 1
    assert size <= budget * 1024
    assert budget * 1024 - size < 4 * order * PREFIX_ROWS + 32


@pytest.mark.parametrize("budget,order", [(0, 4), (-1, 4), (True, 4), (2, -1), (2, 1.5), (2, True)])
def test_invalid_dimensions_still_rejected(budget, order):
    with pytest.raises(ValueError):
        layout(budget, order)


def test_insufficient_budget_explains_required_size():
    with pytest.raises(ValueError, match="needs at least"):
        layout(2, 1000)


def test_large_row_offsets_do_not_overflow_uint32():
    contexts = np.array([0, 1, 0xFFFFFFFF], dtype=np.uint32)
    prefix = np.array([255, 128, 2], dtype=np.uint32)
    slots = (1 << 32) + 13
    actual = indices(contexts, prefix, slots)
    expected = [(int(p) - 1) * slots + mix32(int(h)) % slots for h, p in zip(contexts, prefix)]
    assert list(actual) == expected
    assert actual[0] > 0xFFFFFFFF


def test_extended_header_supports_arbitrary_precision_dimensions():
    order, slots = 1 << 65, (1 << 70) + 3
    body_size = 4 * (ROOT_ROWS + order * PREFIX_ROWS * slots)
    prefix = encode_model_prefix(order, slots, 16, body_size)
    magic, fields, checksum, size = decode_model_header(prefix + bytes(32))
    assert magic == EXTENDED_MAGIC
    assert fields == (order, slots, TOTAL, 16, SCHEME, body_size)
    assert checksum == bytes(32)
    assert size == len(prefix) + 32


@pytest.mark.parametrize("order", [9, 32, 257])
def test_deep_model_training_integer_loss_and_round_trips(tmp_path, order):
    messages = [b"hello mesh", "Café 🌊".encode(), b"another message"] * 10
    counts, _ = train_counts(messages, order, 2, batch_bytes=29, workers=2)
    path = tmp_path / "deep.mcblm"
    save_model(path, order=order, slots=2, strength=16, counts=counts)
    assert path.read_bytes().startswith(EXTENDED_MAGIC)
    with BitModel(path) as model:
        assert model.order == order
        assert model.size_bytes == path.stat().st_size
        assert model.info()["reference_mcu_state_budget_bytes"] >= 128 + 10 * order
        score = evaluate(messages, counts, order, 2, 16, batch_bytes=29)
        predicted = 0.0
        for message in messages:
            state = model.state()
            for byte in message:
                prefix = 1
                for shift in range(7, -1, -1):
                    bit = (byte >> shift) & 1
                    q = state.probability_one(prefix)
                    predicted -= math.log2((q if bit else TOTAL - q) / TOTAL)
                    prefix = prefix * 2 + bit
                state.observe_byte(byte)
        assert score["predicted_bits"] == pytest.approx(predicted, abs=1e-8)
        for message in messages[:3]:
            payload = compress_payload(message, model)
            assert decompress_payload(payload, len(message), model) == message


def test_ninth_previous_byte_actually_changes_prediction(tmp_path):
    messages = [b"xABCDEFGH0", b"yABCDEFGH1"] * 400
    counts, _ = train_counts(messages, 9, 257)
    path = tmp_path / "ninth-byte.mcblm"
    save_model(path, order=9, slots=257, strength=4, counts=counts)
    with BitModel(path) as model:
        predictions = []
        for context in (b"xABCDEFGH", b"yABCDEFGH"):
            state = model.state()
            for byte in context:
                state.observe_byte(byte)
            # ASCII 0 and 1 have the same first seven bits and differ in the LSB.
            predictions.append(state.probability_one(128 + (ord("0") >> 1)))
        assert predictions[0] < TOTAL // 4
        assert predictions[1] > TOTAL * 3 // 4


def test_model_larger_than_old_64_mib_limit_exports_and_loads(tmp_path):
    slots, planned_size = layout(65537, 1)
    counts = np.zeros((ROOT_ROWS + PREFIX_ROWS * slots, 2), dtype=np.uint16)
    counts[1] = [3, 1]
    path = tmp_path / "large.mcblm"
    save_model(path, order=1, slots=slots, strength=16, counts=counts)
    assert 64 * 1024 * 1024 < path.stat().st_size <= planned_size
    with BitModel(path) as model:
        original = b"Larger model, unchanged arithmetic coding."
        payload = compress_payload(original, model)
        assert decompress_payload(payload, len(original), model) == original


def test_streamed_export_removes_failed_output_and_preserves_existing(tmp_path):
    counts = np.zeros((ROOT_ROWS + 9 * PREFIX_ROWS * 8, 2), dtype=np.int64)
    counts[-1, -1] = 65536
    path = tmp_path / "invalid.mcblm"
    with pytest.raises(ValueError, match="uint16"):
        save_model(path, order=9, slots=8, strength=16, counts=counts)
    assert not path.exists()
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        save_model(path, order=9, slots=8, strength=16, counts=counts)
    assert path.read_bytes() == b"existing"


def test_extended_checksum_and_noncanonical_header_detection(tmp_path):
    counts, _ = train_counts([b"hello"], 9, 2)
    path = tmp_path / "damaged.mcblm"
    save_model(path, order=9, slots=2, strength=16, counts=counts)
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum"):
        BitModel(path)
    with pytest.raises(ValueError, match="Noncanonical"):
        decode_model_header(EXTENDED_MAGIC + b"\x80\x00" + bytes(64))
    with pytest.raises(ValueError, match="Truncated"):
        decode_model_header(EXTENDED_MAGIC + b"\x80")


def test_cli_accepts_large_context_and_writes_extended_model(tmp_path):
    import json
    corpus = tmp_path / "messages.jsonl"
    corpus.write_text("".join(json.dumps({"text": f"message number {i}"}) + "\n" for i in range(30)))
    output = tmp_path / "trained"
    assert train_main([str(corpus), "--output", str(output), "--model-kib", "128", "--order", "64",
                       "--workers", "2", "--strengths", "16", "--benchmark-messages", "3"]) == 0
    with BitModel(output / "model.mcblm") as model:
        assert model.order == 64
        assert model.magic == EXTENDED_MAGIC
        assert model.size_bytes <= 128 * 1024


@pytest.mark.parametrize("extended", [False, True])
def test_large_records_and_both_corpus_formats_round_trip(tmp_path, extended):
    path = tmp_path / "large.mccorpus"
    message = ("Café 🌊 " * 160000).encode()
    assert len(message) > 1048576
    writer = RecordWriter(path, extended_lengths=extended)
    writer.write(message)
    writer.write(b"second message")
    stats = writer.close()
    assert stats["file_bytes"] == path.stat().st_size
    assert stats["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert list(read_records(path)) == [message, b"second message"]
    with pytest.raises(ValueError, match="excessive"):
        list(read_records(path, max_message_bytes=1048576))


def test_impossible_extended_record_length_is_rejected_before_allocation(tmp_path):
    path = tmp_path / "truncated.mccorpus"
    path.write_bytes(EXTENDED_CORPUS_MAGIC + encode_uint(1 << 80))
    with pytest.raises(ValueError, match="Truncated"):
        list(read_records(path))


def test_evaluation_does_not_retain_hash_arrays_for_every_order(monkeypatch):
    counts, _ = train_counts([b"long context example"], 64, 2)
    original = bit_training.context_hashes
    references = []
    peak = 0
    def observe(*args):
        nonlocal peak
        for context in original(*args):
            references.append(weakref.ref(context))
            peak = max(peak, sum(reference() is not None for reference in references))
            yield context
    monkeypatch.setattr(bit_training, "context_hashes", observe)
    evaluate([b"long context example"], counts, 64, 2, 16)
    assert len(references) == 64
    assert peak <= 3


def test_message_filter_can_be_disabled_for_long_training_records(tmp_path):
    import json
    corpus = tmp_path / "long.jsonl"
    corpus.write_text("".join(json.dumps({"text": "long message " * 1000 + str(i)}) + "\n" for i in range(10)))
    partitions, report = split_corpus(corpus, max_message_bytes=0)
    assert sum(map(len, partitions.values())) == 10
    assert report["oversized_skipped"] == 0
