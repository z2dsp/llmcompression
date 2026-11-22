"""Small length-prefixed records and re-iterable, bounded-memory training sources."""

import hashlib
import json
from pathlib import Path
import random
import struct

from .bit_model import encode_uint

CORPUS_MAGIC = b"MCBCORP1"
EXTENDED_CORPUS_MAGIC = b"MCBCORP2"
LENGTH = struct.Struct("<I")


def read_records(path, *, max_message_bytes=None):
    with Path(path).open("rb", buffering=1048576) as stream:
        remaining = Path(path).stat().st_size - 8
        magic = stream.read(8)
        if magic not in (CORPUS_MAGIC, EXTENDED_CORPUS_MAGIC):
            raise ValueError(f"Unsupported training corpus: {path}")
        while remaining:
            if magic == CORPUS_MAGIC:
                header = stream.read(LENGTH.size)
                remaining -= len(header)
                if len(header) != LENGTH.size:
                    raise ValueError(f"Truncated record length: {path}")
                size = LENGTH.unpack(header)[0]
            else:
                size = shift = 0
                while True:
                    byte = stream.read(1)
                    remaining -= len(byte)
                    if not byte:
                        raise ValueError(f"Truncated record length: {path}")
                    value = byte[0]
                    size |= (value & 127) << shift
                    if value < 128:
                        if shift and value == 0:
                            raise ValueError("Noncanonical training record length")
                        break
                    shift += 7
            if size < 1 or (max_message_bytes is not None and size > max_message_bytes):
                raise ValueError(f"Invalid or excessive training record length: {size}")
            if size > remaining:
                raise ValueError(f"Truncated training record: {path}")
            message = stream.read(size)
            if len(message) != size:
                raise ValueError(f"Truncated training record: {path}")
            remaining -= size
            yield message


class RecordWriter:
    def __init__(self, path, *, extended_lengths=False):
        self.stream = Path(path).open("wb", buffering=1048576)
        self.digest = hashlib.sha256()
        self.messages = self.utf8_bytes = self.characters = 0
        self.extended_lengths = extended_lengths
        self._write(EXTENDED_CORPUS_MAGIC if extended_lengths else CORPUS_MAGIC)

    def _write(self, block):
        self.stream.write(block)
        self.digest.update(block)

    def write(self, message):
        if not message:
            raise ValueError("Training records must be nonempty")
        if not self.extended_lengths and len(message) > 0xFFFFFFFF:
            raise ValueError("This record needs the extended-length corpus format")
        self._write(encode_uint(len(message)) if self.extended_lengths else LENGTH.pack(len(message)))
        self._write(message)
        self.messages += 1
        self.utf8_bytes += len(message)
        self.characters += len(message.decode("utf-8"))

    def close(self):
        file_bytes = self.stream.tell()
        self.stream.close()
        return {"messages": self.messages, "utf8_bytes": self.utf8_bytes,
                "unicode_characters": self.characters,
                "file_bytes": file_bytes,
                "sha256": self.digest.hexdigest()}


class FileMessages:
    def __init__(self, root, entries):
        self.root, self.entries = Path(root).resolve(), entries
        self.paths = []
        for entry in entries:
            path = (self.root / entry["path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Missing or invalid prepared corpus path: {entry['path']}")
            if path.stat().st_size != entry["file_bytes"]:
                raise ValueError(f"Prepared corpus file size mismatch: {path}")
            self.paths.append(path)

    def __len__(self):
        return sum(entry["messages"] for entry in self.entries)

    def __iter__(self):
        for path, entry in zip(self.paths, self.entries):
            messages = byte_count = 0
            for message in read_records(path):
                messages += 1
                byte_count += len(message)
                yield message
            if (messages, byte_count) != (entry["messages"], entry["utf8_bytes"]):
                raise ValueError(f"Prepared corpus record counts mismatch: {path}")


def load_prepared(path, *, max_message_bytes=8192):
    path = Path(path)
    if path.is_dir():
        path /= "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "mesh-compression-corpus-v1":
        raise ValueError("Unsupported prepared corpus manifest")
    sample_bytes = manifest["preparation"]["sample_bytes"]
    if max_message_bytes and (sample_bytes == 0 or sample_bytes > max_message_bytes):
        raise ValueError("Prepared sample size exceeds --max-message-bytes; raise it or use 0 for no limit/whole articles")
    sources = {name: FileMessages(path.parent, manifest["splits"][name]["files"])
               for name in ("train", "validation", "test")}
    if any(len(source) == 0 for source in sources.values()):
        raise ValueError("Prepared corpus must have nonempty training, validation, and test splits")
    report = {"prepared_manifest": str(path.resolve()), "prepared_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source": manifest["source"], "preparation": manifest["preparation"],
              "splits": {name: {key: value for key, value in manifest["splits"][name].items() if key != "files"}
                         for name in sources},
              "split_strategy": "preassigned by whole article content hash; no in-memory regrouping",
              "corpus_storage": "re-iterable binary record files; bounded message batches"}
    return sources, report


def reservoir(messages, limit, seed):
    """Uniform deterministic sample, with O(limit) retained messages."""
    if limit < 0:
        raise ValueError("Sample limit must be nonnegative")
    if limit == 0:
        return messages
    generator = random.Random(seed)
    selected = []
    for index, message in enumerate(messages):
        if index < limit:
            selected.append(message)
        else:
            choice = generator.randrange(index + 1)
            if choice < limit:
                selected[choice] = message
    return selected


def take_bytes(messages, limit):
    """Take whole messages up to a byte limit; never split a training message."""
    total = 0
    for message in messages:
        if total + len(message) > limit:
            break
        total += len(message)
        yield message
