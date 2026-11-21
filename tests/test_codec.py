import hashlib

import pytest

from llm_compression.codec import compress, decompress, pack_archive, unpack_archive
from llm_compression.demo import size_report


class BytePredictor:
    """Small adaptive predictor: test the format without a model download."""

    def __init__(self):
        self.metadata = {"model_id": "test-byte-predictor", "context_tokens": 7}

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")

    def token_label(self, token):
        return repr(bytes([token]))

    def cdf(self, previous, position):
        favored = 32 if position % 7 == 0 else (previous + 1) % 256
        cdf = [0]
        for token in range(256):
            cdf.append(cdf[-1] + (1000 if token == favored else 1))
        return cdf


@pytest.mark.parametrize("text", [
    "", "a", "hello world\n", " " * 1000, "\ufeffHello\r\n\tGoodbye\r\n",
    "Café — 日本語 🌊 e\u0301\x00\n", "<|endoftext|> special-looking tokens",
    "Repeated text and punctuation!\n" * 100,
])
def test_byte_exact_round_trip(text):
    data = text.encode("utf-8")
    result = compress(data, BytePredictor(), trace_limit=3)
    # A separate predictor has no access to the original text or encoder state.
    assert decompress(result.archive, BytePredictor()) == data
    parsed = unpack_archive(result.archive)
    assert parsed.header["original_sha256"] == hashlib.sha256(data).hexdigest()
    assert parsed.header["token_count"] == len(data)
    assert len(result.trace) == min(3, len(data))


def test_corruption_and_truncation():
    result = compress(b"hello world", BytePredictor())
    data = bytearray(result.archive)
    data[-33] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        decompress(bytes(data), BytePredictor())
    for cut in [0, 5, 10, len(result.archive) - 1]:
        with pytest.raises(ValueError):
            unpack_archive(result.archive[:cut])


def test_wrong_predictor():
    result = compress(b"hello world", BytePredictor())
    other = BytePredictor()
    other.metadata["context_tokens"] = 8
    with pytest.raises(ValueError, match="Predictor does not match"):
        decompress(result.archive, other)


def test_wrong_text_digest():
    parsed = unpack_archive(compress(b"hello world", BytePredictor()).archive)
    parsed.header["original_sha256"] = "0" * 64
    archive = pack_archive(parsed.header, parsed.payload)
    with pytest.raises(ValueError, match="Decoded text checksum"):
        decompress(archive, BytePredictor())


@pytest.mark.parametrize("key,value", [("token_count", -1), ("token_count", True), ("payload_bytes", 0), ("predictor", []), ("original_sha256", "bad")])
def test_invalid_header_fields(key, value):
    parsed = unpack_archive(compress(b"hello", BytePredictor()).archive)
    parsed.header[key] = value
    with pytest.raises(ValueError):
        unpack_archive(pack_archive(parsed.header, parsed.payload))


def test_invalid_utf8():
    with pytest.raises(UnicodeDecodeError):
        compress(b"\xff", BytePredictor())


def test_lossy_tokenizer_rejected():
    class LossyPredictor(BytePredictor):
        def encode(self, text):
            return super().encode(text.strip())

    with pytest.raises(ValueError, match="Tokenizer"):
        compress(b"hello\n", LossyPredictor())


def test_changed_probabilities_detected():
    class DifferentPredictor(BytePredictor):
        def cdf(self, previous, position):
            return list(range(257))

    result = compress(b"all plain ascii so any decoded byte string is allowed", BytePredictor())
    with pytest.raises((ValueError, UnicodeError)):
        decompress(result.archive, DifferentPredictor())


def test_compression_is_deterministic():
    assert compress(b"hello", BytePredictor()).archive == compress(b"hello", BytePredictor()).archive


def test_empty_report():
    report = size_report(b"", compress(b"", BytePredictor()))
    assert report["sizes"]["llm_archive"]["bits_per_character"] is None
    assert report["predicted_bits"] == 0
