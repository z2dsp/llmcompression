"""Pre-shared compression profiles and a minimal, single-datagram wire format.

PAYLOAD_TYPE_GRP_DATA already carries the application data type and byte length.
Our payload is nonce(2), token_count(ULEB128), CRC16(2), arithmetic_stream(rest).
There are no model names, timestamps, sender names, or JSON headers on the air.
"""

import binascii
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import secrets
import struct

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder
from .model import DEFAULT_CACHE, HuggingFacePredictor, Predictor, runtime_signature

CHANNEL = "#llmcompression"
CHANNEL_KEY = hashlib.sha256(CHANNEL.encode("ascii")).digest()[:16]
DATA_TYPE = 0xFFFF  # MeshCore's experimental/developer application namespace.
MAX_DATAGRAM = 163  # Conservative limit shared with meshcore 2.3.19.
MAX_TOKENS = 2048
MAX_TEXT_BYTES = 8192
PROFILE_FORMAT = "llmcompression-mesh-v1"
CALIBRATION_TEXT = "The weather is pleasant today.\n"


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _calibration(predictor: Predictor) -> str:
    """Catch numerical mismatches before sending, without a radio handshake."""
    digest = hashlib.sha256()
    previous = None
    for position, token in enumerate(predictor.encode(CALIBRATION_TEXT)):
        cdf = predictor.cdf(previous, position)
        digest.update(struct.pack(f"<{len(cdf)}I", *cdf))
        previous = token
    return digest.hexdigest()


@dataclass(frozen=True)
class SharedProfile:
    predictor: dict
    calibration_sha256: str

    def as_dict(self) -> dict:
        return {
            "format": PROFILE_FORMAT,
            "channel": CHANNEL,
            "data_type": DATA_TYPE,
            "predictor": self.predictor,
            "calibration_sha256": self.calibration_sha256,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(self.as_dict())).hexdigest()

    @classmethod
    def create(cls, predictor: Predictor) -> "SharedProfile":
        return cls(dict(predictor.metadata), _calibration(predictor))

    def save(self, path: Path) -> None:
        with path.open("x", encoding="utf-8") as output:
            json.dump(self.as_dict(), output, indent=2, sort_keys=True)
            output.write("\n")

    @classmethod
    def load(cls, path: Path) -> "SharedProfile":
        if path.stat().st_size > 65536:
            raise ValueError("Shared profile is too large")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {
            "format", "channel", "data_type", "predictor", "calibration_sha256"
        }:
            raise ValueError("Invalid shared profile")
        if document["format"] != PROFILE_FORMAT or document["channel"] != CHANNEL or document["data_type"] != DATA_TYPE:
            raise ValueError("Unsupported shared profile format, channel, or application data type")
        metadata = document["predictor"]
        required = {"model_id", "revision", "context_tokens", "runtime", "inference", "vocab_size", "start_token", "cdf_total"}
        if not isinstance(metadata, dict) or set(metadata) != required:
            raise ValueError("Invalid predictor metadata")
        if not isinstance(metadata["model_id"], str) or not metadata["model_id"]:
            raise ValueError("Profile model ID must be a Hugging Face ID")
        commit = metadata["revision"]
        if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("Profile must pin an immutable model commit")
        if type(metadata["context_tokens"]) is not int or metadata["context_tokens"] <= 0:
            raise ValueError("Invalid profile context size")
        checksum = document["calibration_sha256"]
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid profile calibration digest")
        return cls(metadata, checksum)

    def load_predictor(self, cache_dir: Path = DEFAULT_CACHE, offline: bool = False) -> HuggingFacePredictor:
        if self.predictor["runtime"] != runtime_signature():
            raise ValueError("Runtime differs from the shared profile; use its Python/library versions and CPU environment")
        predictor = HuggingFacePredictor(
            self.predictor["model_id"], self.predictor["revision"],
            self.predictor["context_tokens"], cache_dir, offline,
        )
        if predictor.metadata != self.predictor or _calibration(predictor) != self.calibration_sha256:
            raise ValueError("Model or integer probability tables differ from the shared profile")
        return predictor


def _encode_count(count: int) -> bytes:
    encoded = bytearray()
    while count >= 128:
        encoded.append((count & 0x7F) | 0x80)
        count >>= 7
    encoded.append(count)
    return bytes(encoded)


def _read_count(packet: bytes) -> tuple[int, int]:
    value = 0
    for shift, offset in ((0, 2), (7, 3)):
        if offset >= len(packet):
            break
        byte = packet[offset]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if value > MAX_TOKENS or packet[2:offset + 1] != _encode_count(value):
                raise ValueError("Invalid or excessive token count")
            return value, offset + 1
    raise ValueError("Truncated or excessive token count")


@dataclass(frozen=True)
class MeshMessage:
    packet: bytes
    token_count: int
    text_bytes: int
    arithmetic_bytes: int


def encode_message(text: str, predictor: Predictor, nonce: bytes | None = None) -> MeshMessage:
    original = text.encode("utf-8")
    if len(original) > MAX_TEXT_BYTES:
        raise ValueError(f"Message exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
    tokens = predictor.encode(text)
    if len(tokens) > MAX_TOKENS:
        raise ValueError(f"Message exceeds {MAX_TOKENS} tokens")
    if predictor.decode(tokens) != text:
        raise ValueError("Tokenizer does not preserve this message exactly")
    encoder = ArithmeticEncoder()
    previous = None
    for position, token in enumerate(tokens):
        encoder.write(predictor.cdf(previous, position), token)
        previous = token
    payload = encoder.finish()
    nonce = secrets.token_bytes(2) if nonce is None else nonce
    if not isinstance(nonce, bytes) or len(nonce) != 2:
        raise ValueError("Message nonce must be exactly two bytes")
    prefix = nonce + _encode_count(len(tokens))
    checksum = binascii.crc_hqx(prefix + original, 0xFFFF).to_bytes(2, "little")
    packet = prefix + checksum + payload
    if len(packet) > MAX_DATAGRAM:
        raise ValueError(
            f"Compressed message needs {len(packet)} bytes; one MeshCore datagram allows {MAX_DATAGRAM}. "
            "Shorten the message (automatic fragmentation is not enabled)."
        )
    return MeshMessage(packet, len(tokens), len(original), len(payload))


def decode_message(packet: bytes, predictor: Predictor) -> str:
    if not 6 <= len(packet) <= MAX_DATAGRAM:
        raise ValueError("Invalid compressed datagram length")
    count, offset = _read_count(packet)
    if len(packet) < offset + 3:
        raise ValueError("Truncated compressed datagram")
    checksum = int.from_bytes(packet[offset:offset + 2], "little")
    arithmetic = packet[offset + 2:]
    decoder = ArithmeticDecoder(arithmetic)
    canonical = ArithmeticEncoder()
    tokens = []
    previous = None
    for position in range(count):
        cdf = predictor.cdf(previous, position)
        token = decoder.read(cdf)
        canonical.write(cdf, token)
        tokens.append(token)
        previous = token
    if canonical.finish() != arithmetic:
        raise ValueError("Noncanonical arithmetic stream; truncated, damaged, or mismatched profile")
    text = predictor.decode(tokens)
    original = text.encode("utf-8")
    if len(original) > MAX_TEXT_BYTES or binascii.crc_hqx(packet[:offset] + original, 0xFFFF) != checksum:
        raise ValueError("Message checksum mismatch; damaged data or a different compression profile")
    return text
