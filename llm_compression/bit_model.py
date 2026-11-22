"""Fixed-size statistical bit-context language model, with integer inference.

Order zero predicts from the partial current byte. Separate higher-order hash
tables also use previous bytes. Each learned count pair is smoothed toward the
lower-order prediction. Tables stay fixed during inference; collisions are
intentional and deterministic. Training needs NumPy, inference only stdlib.
"""

import hashlib
import mmap
from pathlib import Path
import struct

MAGIC = b"MCBITLM1"
EXTENDED_MAGIC = b"MCBITLM2"
HEADER = struct.Struct("<8s6I32s")
TOTAL = 32768
BOS = 256
MASK32 = 0xFFFFFFFF
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619
ROOT_ROWS = 256
PREFIX_ROWS = 255
SCHEME = 2


def validate_dimensions(order, slots):
    if type(order) is not int or order < 0 or type(slots) is not int or (slots <= 0 if order else slots != 0):
        raise ValueError("Context order must be a nonnegative integer, with positive slots for nonzero order")


def encode_uint(value):
    """Unsigned LEB128, with no fixed integer width in the extended header."""
    if type(value) is not int or value < 0:
        raise ValueError("Header integers must be nonnegative")
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def encode_model_prefix(order, slots, strength, body_size):
    fields = (order, slots, TOTAL, strength, SCHEME, body_size)
    # Preserve existing model identities for the original, MCU-sized settings.
    if order <= 8 and all(0 <= value <= 0xFFFFFFFF for value in fields):
        return struct.pack("<8s6I", MAGIC, *fields)
    return EXTENDED_MAGIC + b"".join(encode_uint(value) for value in fields)


def decode_model_header(data):
    magic = bytes(data[:8])
    if magic == MAGIC:
        if len(data) < HEADER.size:
            raise ValueError("Truncated bit model header")
        _, *fields, digest = HEADER.unpack_from(data)
        return magic, tuple(fields), digest, HEADER.size
    if magic != EXTENDED_MAGIC:
        raise ValueError("Unsupported bit model format")
    offset = 8
    fields = []
    for _ in range(6):
        value = shift = 0
        while True:
            if offset >= len(data):
                raise ValueError("Truncated extended bit model header")
            byte = data[offset]
            offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                if shift and byte == 0:
                    raise ValueError("Noncanonical extended header integer")
                break
            shift += 7
        fields.append(value)
    if offset + 32 > len(data):
        raise ValueError("Truncated bit model checksum")
    return magic, tuple(fields), bytes(data[offset:offset + 32]), offset + 32


def mix32(value: int) -> int:
    value &= MASK32
    value ^= value >> 16
    value = (value * 0x7FEB352D) & MASK32
    value ^= value >> 15
    value = (value * 0x846CA68B) & MASK32
    return value ^ (value >> 16)


def layout(model_kib: int, order: int) -> tuple[int, int]:
    if type(order) is not int or order < 0:
        raise ValueError("Context order must be a nonnegative integer")
    if type(model_kib) is not int or model_kib < 2:
        raise ValueError("Model budget must be at least 2 KiB")
    budget = model_kib * 1024
    if not order:
        return 0, HEADER.size + ROOT_ROWS * 4
    # Header length depends on the dimensions. Binary search finds the largest
    # fitting table without assuming a fixed-width or fixed-size header.
    low, high = 0, (budget - ROOT_ROWS * 4) // (order * PREFIX_ROWS * 4)
    while low < high:
        candidate = (low + high + 1) // 2
        body_size = 4 * (ROOT_ROWS + order * PREFIX_ROWS * candidate)
        size = len(encode_model_prefix(order, candidate, 4096, body_size)) + 32 + body_size
        if size <= budget:
            low = candidate
        else:
            high = candidate - 1
    if low < 1:
        body_size = 4 * (ROOT_ROWS + order * PREFIX_ROWS)
        minimum = body_size + len(encode_model_prefix(order, 1, 4096, body_size)) + 32
        raise ValueError(f"Budget is too small for context order {order}; needs at least {(minimum + 1023) // 1024} KiB")
    body_size = 4 * (ROOT_ROWS + order * PREFIX_ROWS * low)
    return low, body_size + len(encode_model_prefix(order, low, 4096, body_size)) + 32


def save_model(path: Path, *, order: int, slots: int, strength: int, counts) -> None:
    import numpy as np

    validate_dimensions(order, slots)
    if type(strength) is not int or not 1 <= strength <= 4096:
        raise ValueError("Invalid bit model smoothing strength")
    array = np.asarray(counts)
    if array.shape != (ROOT_ROWS + order * PREFIX_ROWS * slots, 2) or array.dtype.kind not in "iu":
        raise ValueError("Expected one uint16 zero/one count pair per table row")
    body_size = array.shape[0] * 4
    prefix = encode_model_prefix(order, slots, strength, body_size)
    digest = hashlib.sha256(prefix)
    path = Path(path)
    created = False
    try:
        with path.open("xb") as output:
            created = True
            output.write(prefix + bytes(32))
            # Stream conversion, checksum, and writing: no model-sized bytes copy.
            for start in range(0, len(array), 16384):
                chunk = array[start:start + 16384]
                if (chunk < 0).any() or (chunk > 65535).any():
                    raise ValueError("Expected uint16 zero/one counts")
                body = chunk.astype("<u2", copy=False).tobytes()
                digest.update(body)
                output.write(body)
            output.seek(len(prefix))
            output.write(digest.digest())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


class BitModel:
    """Memory-mapped read-only tables, without a heap-sized model copy."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mapping = None
        try:
            if self.path.stat().st_size < 8:
                raise ValueError("Truncated bit model")
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            magic, fields, digest, header_size = decode_model_header(self._mapping)
            order, slots, total, strength, scheme, size = fields
            if scheme != SCHEME or total != TOTAL:
                raise ValueError("Unsupported bit model format")
            validate_dimensions(order, slots)
            if not 1 <= strength <= 4096:
                raise ValueError("Invalid bit model smoothing")
            if size != 4 * (ROOT_ROWS + order * PREFIX_ROWS * slots) or len(self._mapping) != header_size + size:
                raise ValueError("Bit model file length does not match its header")
            checksum = hashlib.sha256(self._mapping[:header_size - 32])
            for offset in range(header_size, len(self._mapping), 65536):
                checksum.update(self._mapping[offset:offset + 65536])
            if checksum.digest() != digest:
                raise ValueError("Bit model checksum mismatch")
            self.order, self.slots, self.strength = order, slots, strength
            self.magic, self.header_size = magic, header_size
            complete = hashlib.sha256()
            for offset in range(0, len(self._mapping), 65536):
                complete.update(self._mapping[offset:offset + 65536])
            self.sha256 = complete.hexdigest()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def size_bytes(self) -> int:
        return self.header_size + 4 * (ROOT_ROWS + self.order * PREFIX_ROWS * self.slots)

    def counts(self, row: int) -> tuple[int, int]:
        return struct.unpack_from("<HH", self._mapping, self.header_size + row * 4)

    def state(self) -> "BitState":
        return BitState(self)

    def info(self) -> dict:
        return {
            "format": self.magic.decode("ascii"), "sha256": self.sha256,
            "header_bytes": self.header_size,
            "scheme": SCHEME, "model_bytes": self.size_bytes, "context_order": self.order,
            "slots_per_prefix_per_order": self.slots, "smoothing_strength": self.strength,
            "count_parameters": 2 * (ROOT_ROWS + self.order * PREFIX_ROWS * self.slots),
            "probability_total": TOTAL, "reference_mcu_state_budget_bytes": max(1024, 128 + 10 * self.order),
            "state_budget_note": "Algorithm state allowance; excludes message/radio buffers, stacks, filesystem cache, and the Python interpreter.",
        }


class BitState:
    def __init__(self, model: BitModel):
        self.model = model
        self.history = [BOS] * model.order
        self._update_hashes()

    def _update_hashes(self):
        value = FNV_OFFSET
        self.hashes = []
        self.buckets = []
        for previous in self.history:
            value = ((value ^ previous) * FNV_PRIME) & MASK32
            self.hashes.append(value)
            self.buckets.append(mix32(value) % self.model.slots)

    def observe_byte(self, byte: int):
        if not 0 <= byte <= 255:
            raise ValueError("Byte must be between 0 and 255")
        if self.history:
            self.history = [byte] + self.history[:-1]
            self._update_hashes()

    def probability_one(self, prefix: int) -> int:
        if not 1 <= prefix <= 255:
            raise ValueError("Bit prefix must be between 1 and 255")
        zero, one = self.model.counts(prefix)
        probability = ((one + 1) * TOTAL) // (zero + one + 2)
        for order, bucket in enumerate(self.buckets):
            row = ROOT_ROWS + (order * PREFIX_ROWS + prefix - 1) * self.model.slots + bucket
            zero, one = self.model.counts(row)
            denominator = zero + one + self.model.strength
            numerator = one * TOTAL + self.model.strength * probability
            probability = (numerator + denominator // 2) // denominator
        return min(TOTAL - 1, max(1, probability))
