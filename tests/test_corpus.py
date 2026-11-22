import csv
import json
from pathlib import Path
import sqlite3

import pytest

from llm_compression.corpus import extract, read_messages


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "messages.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE channels (key TEXT PRIMARY KEY, name TEXT);
            CREATE TABLE messages (id INTEGER PRIMARY KEY, type TEXT, text TEXT,
                conversation_key TEXT, sender_name TEXT, sender_timestamp INTEGER,
                received_at INTEGER, outgoing INTEGER);
            INSERT INTO channels VALUES ('secret-not-for-export', '#test');
        """)
        db.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
            (2, "CHAN", "Alice: Hello\nworld 🌊", "secret-not-for-export", "Alice", 2, 20, 0),
            (1, "PRIV", " Private\r\nmessage\x00", "private-key", "Bob", 1, 10, 1),
            (3, "CHAN", "Alice: Hello\nworld 🌊", "secret-not-for-export", "Alice", 3, 30, 0),
            (4, "CHAN", "unprefixed", "secret-not-for-export", None, 4, 40, 0),
        ])
    return path


@pytest.mark.parametrize("file_format", ["jsonl", "csv"])
def test_all_messages_and_exact_originals(database, tmp_path, file_format):
    output = tmp_path / f"messages.{file_format}"
    before = database.read_bytes()
    report = extract(database, output, file_format=file_format)
    assert report["messages"] == 4
    assert report["types"] == {"PRIV": 1, "CHAN": 3}
    assert list(read_messages(output, "text")) == [" Private\r\nmessage\x00", "Alice: Hello\nworld 🌊", "Alice: Hello\nworld 🌊", "unprefixed"]
    assert list(read_messages(output)) == [" Private\r\nmessage\x00", "Hello\nworld 🌊", "Hello\nworld 🌊", "unprefixed"]
    assert report["sender_prefixes_removed_in_body"] == 2
    assert "secret-not-for-export" not in output.read_text()
    assert "private-key" not in output.read_text()
    assert database.read_bytes() == before


def test_extraction_filters_and_no_overwrite(database, tmp_path):
    output = tmp_path / "filtered.jsonl"
    assert extract(database, output, kind="channel", channel="#test", incoming_only=True)["messages"] == 3
    with pytest.raises(FileExistsError):
        extract(database, output)
    assert extract(database, output, kind="private", overwrite=True)["messages"] == 1


def test_wal_transactions_are_included(tmp_path):
    path = tmp_path / "wal.db"
    writer = sqlite3.connect(path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE messages (id INTEGER, type TEXT, text TEXT, received_at INTEGER)")
        writer.execute("INSERT INTO messages VALUES (1, 'CHAN', 'still in WAL', 1)")
        writer.commit()
        assert Path(str(path) + "-wal").stat().st_size > 0
        output = tmp_path / "snapshot.jsonl"
        assert extract(path, output)["messages"] == 1
        assert list(read_messages(output)) == ["still in WAL"]
    finally:
        writer.close()


def test_bad_database_and_output_protection(database, tmp_path):
    with pytest.raises(ValueError, match="replace"):
        extract(database, database, overwrite=True)
    missing = tmp_path / "missing.db"
    with pytest.raises(ValueError, match="not found"):
        extract(missing, tmp_path / "output.jsonl")
    assert not missing.exists()


def test_body_falls_back_to_standard_text_jsonl(tmp_path):
    path = tmp_path / "plain.jsonl"
    path.write_text(json.dumps({"text": "hello\nworld"}) + "\n", encoding="utf-8")
    assert list(read_messages(path)) == ["hello\nworld"]
