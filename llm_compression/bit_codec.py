"""Reference arithmetic compressor for the MCU bit model (stdlib inference)."""

import argparse
from pathlib import Path
import struct
import sys
import zlib

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder
from .bit_model import BitModel, TOTAL

FILE_HEADER = struct.Struct("<8s32sIII")
FILE_MAGIC = b"MCBFILE1"


def compress_payload(data: bytes, model: BitModel) -> bytes:
    encoder = ArithmeticEncoder()
    state = model.state()
    for byte in data:
        prefix = 1
        for shift in range(7, -1, -1):
            bit = (byte >> shift) & 1
            one = state.probability_one(prefix)
            encoder.write((0, TOTAL - one, TOTAL), bit)
            prefix = (prefix << 1) | bit
        state.observe_byte(byte)
    return encoder.finish()


def decompress_payload(payload: bytes, byte_count: int, model: BitModel, *, max_output_bytes=64 * 1024 * 1024) -> bytes:
    if type(byte_count) is not int or not 0 <= byte_count <= max_output_bytes:
        raise ValueError("Invalid or excessive output byte count")
    if not payload:
        raise ValueError("Missing arithmetic stream")
    decoder, canonical = ArithmeticDecoder(payload), ArithmeticEncoder()
    state = model.state()
    result = bytearray()
    for _ in range(byte_count):
        prefix = 1
        for _ in range(8):
            one = state.probability_one(prefix)
            cdf = (0, TOTAL - one, TOTAL)
            bit = decoder.read(cdf)
            canonical.write(cdf, bit)
            prefix = (prefix << 1) | bit
        byte = prefix - 256
        result.append(byte)
        state.observe_byte(byte)
    if canonical.finish() != payload:
        raise ValueError("Truncated, corrupt, or noncanonical arithmetic stream")
    return bytes(result)


def pack_file(data: bytes, model: BitModel) -> bytes:
    if len(data) > 0xFFFFFFFF:
        raise ValueError("Input exceeds the archive format's 32-bit length")
    payload = compress_payload(data, model)
    return FILE_HEADER.pack(FILE_MAGIC, bytes.fromhex(model.sha256), len(data), len(payload), zlib.crc32(data)) + payload


def unpack_file(archive: bytes, model: BitModel, *, max_output_bytes=64 * 1024 * 1024) -> bytes:
    if len(archive) < FILE_HEADER.size:
        raise ValueError("Truncated MCU compression file")
    magic, fingerprint, byte_count, payload_size, checksum = FILE_HEADER.unpack_from(archive)
    if magic != FILE_MAGIC or fingerprint != bytes.fromhex(model.sha256):
        raise ValueError("Unsupported file or wrong shared model")
    if len(archive) != FILE_HEADER.size + payload_size:
        raise ValueError("Arithmetic payload length mismatch")
    data = decompress_payload(archive[FILE_HEADER.size:], byte_count, model, max_output_bytes=max_output_bytes)
    if zlib.crc32(data) != checksum:
        raise ValueError("Decompressed content checksum mismatch")
    return data


def test_vectors(model: BitModel) -> dict:
    vectors = []
    for data in (b"", b"Hello mesh!", b"Line one\r\nLine two\n", "Café 日本語 🌊".encode(), bytes(range(256))):
        payload = compress_payload(data, model)
        if decompress_payload(payload, len(data), model) != data:
            raise ValueError("Model test-vector round trip failed")
        state = model.state()
        probabilities = []
        for byte in data[:2]:
            prefix = 1
            for shift in range(7, -1, -1):
                probabilities.append(state.probability_one(prefix))
                prefix = (prefix << 1) | ((byte >> shift) & 1)
            state.observe_byte(byte)
        vectors.append({"input_hex": data.hex(), "byte_count": len(data), "payload_hex": payload.hex(),
                        "crc32": zlib.crc32(data), "first_bit_probabilities_q15": probabilities})
    return {"format": "MCBITLM1-vectors-v1", "model_format": model.magic.decode("ascii"),
            "model_sha256": model.sha256, "vectors": vectors}


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description="Inspect or losslessly compress/decompress with a trained MCU bit-context model.")
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info")
    info.add_argument("--model", type=Path, required=True)
    for command in ("compress", "decompress"):
        sub = commands.add_parser(command)
        sub.add_argument("input", type=Path)
        sub.add_argument("output", type=Path)
        sub.add_argument("--model", type=Path, required=True)
        if command == "decompress":
            sub.add_argument("--max-output-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        with BitModel(args.model) as model:
            if args.command == "info":
                print(json.dumps(model.info(), indent=2))
                return 0
            if args.output.exists():
                raise FileExistsError(f"Output exists: {args.output}")
            source = args.input.read_bytes()
            result = pack_file(source, model) if args.command == "compress" else unpack_file(source, model, max_output_bytes=args.max_output_bytes)
            with args.output.open("xb") as output:
                output.write(result)
            print(f"{len(source):,} input bytes -> {len(result):,} output bytes; model shared separately.")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
