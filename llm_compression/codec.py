"""A versioned archive around a language predictor and an arithmetic stream."""

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Callable

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder
from .model import Predictor

MAGIC = b"LLMC\x01"
MAX_HEADER = 65536
DIGEST_BYTES = 32
Progress = Callable[[int, int], None]


@dataclass
class Archive:
    header: dict
    payload: bytes
    header_bytes: int


@dataclass
class CompressionResult:
    archive: bytes
    token_count: int
    predicted_bits: float
    payload_bytes: int
    trace: list[dict]


def pack_archive(header: dict, payload: bytes) -> bytes:
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(encoded) > MAX_HEADER:
        raise ValueError("Archive metadata is too large")
    body = MAGIC + struct.pack(">I", len(encoded)) + encoded + payload
    return body + hashlib.sha256(body).digest()


def unpack_archive(data: bytes) -> Archive:
    prefix = len(MAGIC) + 4
    if len(data) < prefix + DIGEST_BYTES or not data.startswith(MAGIC):
        raise ValueError("Not a supported LLMC archive")
    size = struct.unpack(">I", data[len(MAGIC):prefix])[0]
    if not 0 < size <= MAX_HEADER or prefix + size + DIGEST_BYTES > len(data):
        raise ValueError("Invalid or truncated archive header")
    if hashlib.sha256(data[:-DIGEST_BYTES]).digest() != data[-DIGEST_BYTES:]:
        raise ValueError("Archive checksum mismatch: file is corrupt or truncated")
    try:
        header = json.loads(data[prefix:prefix + size])
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid archive metadata") from exc
    if not isinstance(header, dict):
        raise ValueError("Archive metadata must be an object")
    for key in ("token_count", "original_bytes", "payload_bytes"):
        if type(header.get(key)) is not int or header[key] < 0:
            raise ValueError(f"Invalid archive field: {key}")
    if not isinstance(header.get("predictor"), dict):
        raise ValueError("Archive is missing predictor metadata")
    digest = header.get("original_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid original text digest")
    payload = data[prefix + size:-DIGEST_BYTES]
    if len(payload) != header["payload_bytes"] or not payload:
        raise ValueError("Invalid arithmetic payload length")
    return Archive(header, payload, len(data) - len(payload))


def compress(
    data: bytes,
    predictor: Predictor,
    trace_limit: int = 0,
    progress: Progress | None = None,
) -> CompressionResult:
    """Compress UTF-8 bytes, checking tokenization preserves every byte."""
    text = data.decode("utf-8")
    tokens = predictor.encode(text)
    if predictor.decode(tokens).encode("utf-8") != data:
        raise ValueError("Tokenizer cannot round-trip this input exactly")
    encoder = ArithmeticEncoder()
    predicted_bits = 0.0
    trace = []
    previous = None
    for position, token in enumerate(tokens):
        cdf = predictor.cdf(previous, position)
        encoder.write(cdf, token)
        probability = (cdf[token + 1] - cdf[token]) / cdf[-1]
        bits = -math.log2(probability)
        predicted_bits += bits
        if position < trace_limit:
            trace.append({
                "position": position,
                "token_id": token,
                "token": predictor.token_label(token),
                "probability": probability,
                "bits": bits,
                "cumulative_bits": predicted_bits,
            })
        previous = token
        if progress:
            progress(position + 1, len(tokens))
    payload = encoder.finish()
    header = {
        "predictor": predictor.metadata,
        "token_count": len(tokens),
        "original_bytes": len(data),
        "original_sha256": hashlib.sha256(data).hexdigest(),
        "payload_bytes": len(payload),
    }
    return CompressionResult(pack_archive(header, payload), len(tokens), predicted_bits, len(payload), trace)


def decompress(data: bytes, predictor: Predictor, progress: Progress | None = None) -> bytes:
    archive = unpack_archive(data)
    if archive.header["predictor"] != predictor.metadata:
        raise ValueError("Predictor does not match the archive's model, revision, settings, or runtime")
    decoder = ArithmeticDecoder(archive.payload)
    tokens = []
    previous = None
    count = archive.header["token_count"]
    for position in range(count):
        cdf = predictor.cdf(previous, position)
        token = decoder.read(cdf)
        tokens.append(token)
        previous = token
        if progress:
            progress(position + 1, count)
    restored = predictor.decode(tokens).encode("utf-8")
    if len(restored) != archive.header["original_bytes"] or hashlib.sha256(restored).hexdigest() != archive.header["original_sha256"]:
        raise ValueError(
            "Decoded text checksum mismatch. Model probabilities differed; "
            "use the original model, runtime versions, and CPU environment."
        )
    return restored
