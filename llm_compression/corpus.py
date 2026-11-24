"""Read-only extraction of Remote Terminal for MeshCore's decrypted messages."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

FIELDS = ("id", "type", "text", "body", "sender_name", "channel", "sender_timestamp", "received_at", "outgoing")


def message_body(text: str, sender: str | None, kind: str) -> str:
    """Remove only the exact known channel sender envelope, preserving content."""
    if kind == "CHAN" and sender and text.startswith(sender + ":"):
        body = text[len(sender) + 1:]
        return body[1:] if body.startswith(" ") else body
    return text


def extract(database: Path, output: Path, *, file_format="jsonl", kind="all",
            channel=None, incoming_only=False, overwrite=False, exclude_channels=()) -> dict:
    database, output = Path(database).resolve(), Path(output).resolve()
    if not database.is_file():
        raise ValueError(f"Database not found: {database}")
    if output == database or str(output).startswith(str(database) + "-"):
        raise ValueError("Output must not replace the database or its WAL/SHM files")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it")
    if file_format not in {"jsonl", "csv"} or kind not in {"all", "channel", "private"}:
        raise ValueError("Invalid output format or message type")
    output.parent.mkdir(parents=True, exist_ok=True)
    uri = database.as_uri()
    # SQLite rejects non-local URI authorities; preserve UNC as a path instead.
    if database.drive.startswith("\\\\"):
        uri = uri.replace("file://", "file:////", 1)
    connection = sqlite3.connect(uri + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    temporary = None
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")  # Consistent WAL-aware read snapshot.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
        if not {"id", "type", "text", "received_at"} <= columns:
            raise ValueError("No supported Remote Terminal messages table found")
        channel_columns = {row[1] for row in connection.execute("PRAGMA table_info(channels)")}
        can_join = {"key", "name"} <= channel_columns and "conversation_key" in columns
        expressions = []
        for name in ("id", "type", "text", "sender_name", "sender_timestamp", "received_at", "outgoing"):
            fallback = "0" if name == "outgoing" else "NULL"
            expressions.append(f'm."{name}" AS "{name}"' if name in columns else f'{fallback} AS "{name}"')
        expressions.append("c.name AS channel" if can_join else "NULL AS channel")
        query = "SELECT " + ", ".join(expressions) + " FROM messages m"
        if can_join:
            query += " LEFT JOIN channels c ON m.type = 'CHAN' AND m.conversation_key = c.key"
        filters, parameters = [], []
        if kind != "all":
            filters.append("m.type = ?")
            parameters.append("CHAN" if kind == "channel" else "PRIV")
        if channel is not None:
            if not can_join:
                raise ValueError("Database has no channel-name mapping")
            filters.append("c.name = ?")
            parameters.append(channel)
        if exclude_channels:
            if not can_join:
                raise ValueError("Database has no channel-name mapping")
            names = tuple(name.strip().casefold() for name in exclude_channels)
            filters.append("(c.name IS NULL OR lower(c.name) NOT IN (" + ",".join("?" for _ in names) + "))")
            parameters.extend(names)
        if incoming_only:
            if "outgoing" not in columns:
                raise ValueError("Database has no outgoing/incoming marker")
            filters.append("COALESCE(m.outgoing, 0) = 0")
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY m.received_at, m.id"
        types = Counter()
        count = byte_count = stripped = 0
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent,
                                         prefix=".messages-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS) if file_format == "csv" else None
            if writer:
                writer.writeheader()
            for row in connection.execute(query, parameters):
                record = dict(row)
                text = record["text"]
                if not isinstance(text, str):
                    raise ValueError(f"Message {record['id']} contains non-text data")
                record["body"] = message_body(text, record["sender_name"], record["type"])
                record["outgoing"] = bool(record["outgoing"])
                if writer:
                    writer.writerow(record)
                else:
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                encoded = text.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
                count += 1
                byte_count += len(encoded)
                stripped += record["body"] != text
                types[record["type"]] += 1
        if overwrite:
            temporary.replace(output)
        else:
            # Hard-link creation fails atomically if another process created it.
            import os
            os.link(temporary, output)
            temporary.unlink()
        temporary = None
        return {"messages": count, "types": dict(types), "original_text_utf8_bytes": byte_count,
                "sender_prefixes_removed_in_body": stripped, "text_sequence_sha256": digest.hexdigest(),
                "output": str(output), "format": file_format}
    finally:
        connection.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_messages(path: Path, text_field="body"):
    """Yield one exact message per JSONL record or CSV row; no newline splitting."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as stream:
        records = csv.DictReader(stream) if path.suffix.lower() == ".csv" else (json.loads(line) for line in stream if line.strip())
        for number, record in enumerate(records, 1):
            if not isinstance(record, dict):
                raise ValueError(f"Record {number} is not an object")
            text = record.get(text_field, record.get("text") if text_field == "body" else None)
            if not isinstance(text, str):
                raise ValueError(f"Record {number} has no string field {text_field!r}")
            yield text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract all decrypted Remote Terminal MeshCore messages, preserving originals and clean message bodies.")
    parser.add_argument("database", type=Path, nargs="?", default=Path("data/meshcore.db"))
    parser.add_argument("--output", type=Path, default=Path("data/messages.jsonl"))
    parser.add_argument("--format", choices=("jsonl", "csv"), help="default: inferred from output suffix")
    parser.add_argument("--type", choices=("all", "channel", "private"), default="all")
    parser.add_argument("--channel", help="optional channel name filter")
    parser.add_argument("--exclude-channel", action="append", default=[], help="repeatable channel exclusion; preserves private/unmapped messages")
    parser.add_argument("--incoming-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = extract(args.database, args.output, file_format=args.format or ("csv" if args.output.suffix.lower() == ".csv" else "jsonl"),
                         kind=args.type, channel=args.channel, incoming_only=args.incoming_only, overwrite=args.overwrite, exclude_channels=args.exclude_channel)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0
