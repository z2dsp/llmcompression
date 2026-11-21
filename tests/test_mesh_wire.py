import json

import pytest

from llm_compression.mesh_wire import (
    CHANNEL, DATA_TYPE, MAX_DATAGRAM, MAX_TEXT_BYTES, MAX_TOKENS,
    SharedProfile, decode_message, encode_message,
)


class UniformPredictor:
    def __init__(self):
        self.metadata = {
            "model_id": "test/uniform", "revision": "0" * 40, "context_tokens": 8,
            "runtime": {}, "inference": "test", "vocab_size": 256,
            "start_token": 0, "cdf_total": 256,
        }

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")

    def cdf(self, previous, position):
        return list(range(257))


@pytest.mark.parametrize("text", ["", "a", "Hello mesh!", "\ufeffCafé 日本語 🌊 e\u0301\x00\r\n", "Line one\r\nLine two\n", "a" * 128])
def test_wire_round_trip(text):
    result = encode_message(text, UniformPredictor())
    assert decode_message(result.packet, UniformPredictor()) == text
    assert len(result.packet) <= MAX_DATAGRAM
    assert len(result.packet) - result.arithmetic_bytes in (5, 6)


def test_uniform_arithmetic_has_a_known_encoding():
    result = encode_message("hello", UniformPredictor(), nonce=b"\x12\x34")
    assert result.packet[:3] == b"\x12\x34\x05"
    assert result.packet[5:] == b"hello\x40"
    assert result.text_bytes == 5
    assert result.arithmetic_bytes == 6


def test_nonce_makes_repeated_text_distinct():
    first = encode_message("hello", UniformPredictor(), nonce=b"\x00\x01")
    second = encode_message("hello", UniformPredictor(), nonce=b"\x00\x02")
    assert first.packet != second.packet
    assert first.packet[5:] == second.packet[5:]


@pytest.mark.parametrize("offset", range(11))
def test_each_corrupt_byte_detected(offset):
    damaged = bytearray(encode_message("hello", UniformPredictor(), nonce=b"\x00\x01").packet)
    damaged[offset] ^= 1
    with pytest.raises(ValueError):
        decode_message(bytes(damaged), UniformPredictor())


def test_truncation_and_extra_bytes_rejected():
    packet = encode_message("hello", UniformPredictor()).packet
    for length in range(len(packet)):
        with pytest.raises(ValueError):
            decode_message(packet[:length], UniformPredictor())
    with pytest.raises(ValueError, match="Noncanonical"):
        decode_message(packet + b"\x00", UniformPredictor())


@pytest.mark.parametrize("count", [b"\x80\x00", b"\xff\x7f", b"\x80\x80\x00"])
def test_invalid_or_excessive_token_counts(count):
    with pytest.raises(ValueError, match="token count"):
        decode_message(b"\x00\x01" + count + b"\x00\x00\x40", UniformPredictor())


def test_limits_checked_before_radio_send():
    with pytest.raises(ValueError, match="datagram"):
        encode_message("x" * 200, UniformPredictor())
    with pytest.raises(ValueError, match="tokens"):
        encode_message("x" * (MAX_TOKENS + 1), UniformPredictor())
    with pytest.raises(ValueError, match="UTF-8"):
        encode_message("x" * (MAX_TEXT_BYTES + 1), UniformPredictor())


def test_profile_round_trip_and_fingerprint(tmp_path):
    profile = SharedProfile.create(UniformPredictor())
    path = tmp_path / "shared.json"
    profile.save(path)
    restored = SharedProfile.load(path)
    assert restored == profile
    assert restored.fingerprint == profile.fingerprint
    assert restored.as_dict()["channel"] == CHANNEL
    assert restored.as_dict()["data_type"] == DATA_TYPE
    with pytest.raises(FileExistsError):
        profile.save(path)


@pytest.mark.parametrize("field,value", [("channel", "#other"), ("data_type", 0), ("format", "unknown"), ("calibration_sha256", "bad")])
def test_incompatible_profile_rejected(tmp_path, field, value):
    document = SharedProfile.create(UniformPredictor()).as_dict()
    document[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        SharedProfile.load(path)


def test_profile_runtime_and_probability_mismatch_rejected(monkeypatch):
    import llm_compression.mesh_wire as wire

    profile = SharedProfile.create(UniformPredictor())
    monkeypatch.setattr(wire, "runtime_signature", lambda: {"different": True})
    with pytest.raises(ValueError, match="Runtime differs"):
        profile.load_predictor()
    monkeypatch.setattr(wire, "runtime_signature", lambda: {})

    class DifferentPredictor(UniformPredictor):
        def cdf(self, previous, position):
            return [0] + [value * 2 for value in range(1, 257)]

    monkeypatch.setattr(wire, "HuggingFacePredictor", lambda *args: DifferentPredictor())
    with pytest.raises(ValueError, match="probability tables differ"):
        profile.load_predictor()


@pytest.mark.integration
def test_actual_shared_profile_if_requested():
    import os
    if os.environ.get("RUN_MODEL_TESTS") != "1":
        pytest.skip("set RUN_MODEL_TESTS=1 to run the real model")
    from llm_compression.demo import SAMPLE
    from llm_compression.model import HuggingFacePredictor

    predictor = HuggingFacePredictor()
    profile = SharedProfile.create(predictor)
    decoder = profile.load_predictor(offline=True)
    packet = encode_message(SAMPLE, predictor)
    assert packet.arithmetic_bytes == 149
    assert len(packet.packet) == 155
    assert decode_message(packet.packet, decoder) == SAMPLE
