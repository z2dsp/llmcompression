import hashlib
import io
import json
from pathlib import Path

import pytest

from llm_compression import wikipedia
from llm_compression.bit_training import evaluate, main as train_main, train_counts
from llm_compression.training_corpus import (CORPUS_MAGIC, FileMessages, RecordWriter,
                                             load_prepared, read_records, reservoir)


@pytest.mark.parametrize("size", [4, 5, 17, 160])
@pytest.mark.parametrize("text", ["", "hello world\r\n" * 90, "Café 日本語 🌊" * 90, "a" * 1000, " \n\t" * 80])
def test_chunking_preserves_all_bytes_and_complete_unicode(text, size):
    original = text.encode()
    chunks = list(wikipedia.chunk_utf8(original, size))
    assert b"".join(chunks) == original
    assert all(1 <= len(chunk) <= size for chunk in chunks)
    assert "".join(chunk.decode() for chunk in chunks) == text


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}


@pytest.mark.parametrize("supports_range", [False, True])
def test_download_resumes_or_safely_restarts_and_verifies(tmp_path, monkeypatch, supports_range):
    data = bytes(range(256)) * 200
    entry = {"path": "20231101.en/train-00000-of-00041.parquet", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    destination = tmp_path / entry["path"]
    destination.parent.mkdir()
    partial = destination.with_suffix(".parquet.part")
    partial.write_bytes(data[:1000])
    calls = []
    def get(url, **headers):
        calls.append(headers)
        if supports_range:
            return Response(data[1000:], 206, {"Content-Range": f"bytes 1000-{len(data)-1}/{len(data)}"})
        return Response(data)
    monkeypatch.setattr(wikipedia, "request", get)
    wikipedia.download_file(tmp_path, entry, wikipedia.REVISION)
    assert calls == [{"Range": "bytes=1000-"}]
    assert destination.read_bytes() == data
    assert not partial.exists()
    assert wikipedia.download_file(tmp_path, entry, wikipedia.REVISION)[1] == "verified existing"
    assert len(calls) == 1
    destination.write_bytes(b"x" * len(data))
    with pytest.raises(ValueError, match="fails verification"):
        wikipedia.download_file(tmp_path, entry, wikipedia.REVISION)


def test_bad_download_never_becomes_a_complete_shard(tmp_path, monkeypatch):
    entry = {"path": "bad.parquet", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    monkeypatch.setattr(wikipedia, "request", lambda *args, **kwargs: Response(b"bad"))
    monkeypatch.setattr(wikipedia.time, "sleep", lambda delay: None)
    with pytest.raises(OSError, match="SHA-256"):
        wikipedia.download_file(tmp_path, entry, wikipedia.REVISION, attempts=2)
    assert not (tmp_path / "bad.parquet").exists()
    assert not (tmp_path / "bad.parquet.part").exists()


@pytest.fixture
def prepared(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    raw, output = tmp_path / "raw", tmp_path / "prepared"
    source_path = raw / "20231101.en/train-00000-of-00041.parquet"
    source_path.parent.mkdir(parents=True)
    texts = [f"Article {number}: unique example text 🌊." for number in range(200)]
    texts.extend([texts[0], "日本語 🌊 " * 100, "X" * 600, ""])
    pq.write_table(pa.table({"id": [str(i) for i in range(len(texts))], "text": texts}), source_path)
    entry = {"path": str(source_path.relative_to(raw)), "bytes": source_path.stat().st_size,
             "sha256": wikipedia.sha256_file(source_path)}
    config = {"version": 1, "sample_bytes": 160, "seed": 42, "validation_fraction": 0.1,
              "test_fraction": 0.1, "arrow_batch_articles": 7}
    (output / "parts").mkdir(parents=True)
    stats = wikipedia.prepare_shard(raw, entry, output, config)
    manifest = {"format": "mesh-compression-corpus-v1", "source": {"revision": wikipedia.REVISION},
                "preparation": config, "splits": {name: {**row, "files": [row]} for name, row in stats["splits"].items()}}
    # The global manifest has aggregate statistics and file entries, rather than
    # embedding per-file digest/path in the aggregate.
    for split in manifest["splits"].values():
        for key in ("path", "sha256"):
            split.pop(key)
    wikipedia.write_json(output / "manifest.json", manifest)
    return raw, output, entry, config, texts, stats


def test_preparation_splits_whole_articles_preserves_bytes_and_resumes(prepared):
    raw, output, entry, config, texts, stats = prepared
    expected = {name: [] for name in ("train", "validation", "test")}
    for text in texts:
        data = text.encode()
        name = wikipedia.article_split(data, 42, 0.1, 0.1)
        expected[name].extend(wikipedia.chunk_utf8(data, 160))
    assert stats["articles"] == len(texts)
    assert stats["empty_articles"] == 1
    assert sum(len(text.encode()) for text in texts) == stats["original_utf8_bytes"]
    for name, row in stats["splits"].items():
        path = output / row["path"]
        assert list(read_records(path)) == expected[name]
        assert wikipedia.sha256_file(path) == row["sha256"]
    sources, _ = load_prepared(output)
    assert {name: list(source) for name, source in sources.items()} == expected
    assert {name: list(source) for name, source in sources.items()} == expected  # re-iterable
    mtime = (output / "parts/00000.train.mccorpus").stat().st_mtime_ns
    assert wikipedia.prepare_shard(raw, entry, output, config) == stats
    assert (output / "parts/00000.train.mccorpus").stat().st_mtime_ns == mtime
    with pytest.raises(ValueError, match="configuration changed"):
        wikipedia.prepare_shard(raw, entry, output, {**config, "seed": 43})


def test_whole_article_preparation_uses_unbounded_record_lengths(prepared, tmp_path):
    raw, _, entry, config, texts, _ = prepared
    output = tmp_path / "whole-articles"
    (output / "parts").mkdir(parents=True)
    result = wikipedia.prepare_shard(raw, entry, output, {**config, "sample_bytes": 0})
    expected = {name: [] for name in ("train", "validation", "test")}
    for text in texts:
        data = text.encode()
        if data:
            expected[wikipedia.article_split(data, 42, 0.1, 0.1)].append(data)
    assert sum(row["messages"] for row in result["splits"].values()) == len(texts) - 1
    for name, row in result["splits"].items():
        path = output / row["path"]
        assert path.read_bytes()[:8] == b"MCBCORP2"
        assert list(read_records(path)) == expected[name]
        assert path.stat().st_size == row["file_bytes"]
    manifest = {"format": "mesh-compression-corpus-v1", "source": {"revision": wikipedia.REVISION},
                "preparation": {**config, "sample_bytes": 0},
                "splits": {name: {"files": [row]} for name, row in result["splits"].items()}}
    wikipedia.write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="max-message-bytes"):
        load_prepared(output)
    sources, _ = load_prepared(output, max_message_bytes=0)
    assert {name: list(source) for name, source in sources.items()} == expected


@pytest.mark.parametrize("evaluation_limit,benchmark_limit", [(17, 3), (0, 0)])
def test_training_directly_from_prepared_stream(prepared, tmp_path, evaluation_limit, benchmark_limit):
    _, output, _, _, _, _ = prepared
    destination = tmp_path / "model"
    assert train_main([str(output), "--output", str(destination), "--model-kib", "8", "--order", "2",
                       "--train-bytes", "1000", "--evaluation-messages", str(evaluation_limit), "--strengths", "4,16",
                       "--benchmark-messages", str(benchmark_limit), "--workers", "2", "--batch-bytes", "128"]) == 0
    report = json.loads((destination / "report.json").read_text())
    assert 0 < report["training_counts"]["utf8_bytes"] <= 1000
    for name in ("validation", "test"):
        if evaluation_limit:
            assert report["evaluation"][f"{name}_messages"] <= evaluation_limit
        else:
            assert report["evaluation"][f"{name}_messages"] == report["corpus"]["splits"][name]["messages"]
    assert report["benchmark"]["all_round_trips_exact"]
    assert report["parallelism"]["backend"] == "threads"
    assert report["parallelism"]["workers"] == 2
    assert not (destination / "train.jsonl").exists()  # no full corpus copies
    assert report["corpus"]["source"]["revision"] == wikipedia.REVISION


def test_offline_verification_detects_same_size_corruption(prepared):
    _, output, _, _, _, stats = prepared
    assert wikipedia.verify(output) == 3
    path = output / stats["splits"]["train"]["path"]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="verification failed"):
        wikipedia.verify(output)


def test_records_reject_truncation_and_oversized_lengths(tmp_path):
    path = tmp_path / "messages.mccorpus"
    writer = RecordWriter(path)
    writer.write("Café 🌊".encode())
    row = writer.close()
    row["path"] = path.name
    source = FileMessages(tmp_path, [row])
    assert list(source) == ["Café 🌊".encode()]
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        list(read_records(path, max_message_bytes=1048576))
    path.write_bytes(CORPUS_MAGIC + (2**32-1).to_bytes(4, "little"))
    with pytest.raises(ValueError, match="excessive"):
        list(read_records(path, max_message_bytes=1048576))


def test_reservoir_is_bounded_repeatable_and_covers_the_source():
    sample = reservoir((str(i).encode() for i in range(10000)), 25, 42)
    assert len(sample) == len(set(sample)) == 25
    assert max(map(int, sample)) > 9000
    assert sample == reservoir((str(i).encode() for i in range(10000)), 25, 42)


def test_counting_and_evaluation_work_with_single_pass_generators():
    consumed = 0
    def messages():
        nonlocal consumed
        for i in range(1000):
            consumed += 1
            yield f"message {i} 🌊".encode()
    checkpoints = []
    actual, stats = train_counts(messages(), 2, 17, batch_bytes=1024,
                                 progress=lambda *args: checkpoints.append(consumed))
    assert checkpoints[0] < 100  # no eager list conversion
    assert consumed == stats["messages"] == 1000
    expected, _ = train_counts([f"message {i} 🌊".encode() for i in range(1000)], 2, 17)
    assert (actual == expected).all()
    assert evaluate(messages(), actual, 2, 17, 16) == evaluate(
        [f"message {i} 🌊".encode() for i in range(1000)], actual, 2, 17, 16)
