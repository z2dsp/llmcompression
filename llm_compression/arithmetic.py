"""Integer arithmetic coding, independent of PyTorch and Hugging Face.

Each symbol narrows an inclusive 32-bit interval using an integer cumulative
distribution (CDF). Renormalization emits common leading bits, including the
underflow case where the interval straddles the midpoint. The decoder performs
the inverse operations with the same sequence of CDFs.
"""

from bisect import bisect_right
from collections.abc import Sequence

BITS = 32
FULL = 1 << BITS
HALF = FULL >> 1
QUARTER = HALF >> 1
MASK = FULL - 1
MAX_TOTAL = QUARTER


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.byte = 0
        self.used = 0

    def write(self, bit: int) -> None:
        self.byte = (self.byte << 1) | bit
        self.used += 1
        if self.used == 8:
            self.data.append(self.byte)
            self.byte = self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.data.append(self.byte << (8 - self.used))
        return bytes(self.data)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def read(self) -> int:
        # The arithmetic stream is conceptually followed by zero bits. Archive
        # length and SHA-256 checks detect physical truncation before decoding.
        pos = self.position
        self.position += 1
        if pos >= len(self.data) * 8:
            return 0
        return (self.data[pos // 8] >> (7 - pos % 8)) & 1


def _check(cdf: Sequence[int]) -> None:
    if len(cdf) < 2 or cdf[0] != 0 or not 0 < cdf[-1] <= MAX_TOTAL:
        raise ValueError("CDF must start at zero and have a positive, bounded total")


class ArithmeticEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = MASK
        self.pending = 0
        self.writer = _BitWriter()
        self.finished = False

    def _emit(self, bit: int) -> None:
        self.writer.write(bit)
        for _ in range(self.pending):
            self.writer.write(bit ^ 1)
        self.pending = 0

    def write(self, cdf: Sequence[int], symbol: int) -> None:
        if self.finished:
            raise ValueError("Encoder is already finished")
        _check(cdf)
        if not 0 <= symbol < len(cdf) - 1 or cdf[symbol] >= cdf[symbol + 1]:
            raise ValueError("Symbol must have a positive frequency")
        width = self.high - self.low + 1
        self.high = self.low + width * cdf[symbol + 1] // cdf[-1] - 1
        self.low += width * cdf[symbol] // cdf[-1]
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low *= 2
            self.high = self.high * 2 + 1

    def finish(self) -> bytes:
        if self.finished:
            raise ValueError("Encoder is already finished")
        self.finished = True
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        return self.writer.finish()


class ArithmeticDecoder:
    def __init__(self, data: bytes) -> None:
        self.low = 0
        self.high = MASK
        self.reader = _BitReader(data)
        self.code = 0
        for _ in range(BITS):
            self.code = (self.code << 1) | self.reader.read()

    def read(self, cdf: Sequence[int]) -> int:
        _check(cdf)
        width = self.high - self.low + 1
        value = ((self.code - self.low + 1) * cdf[-1] - 1) // width
        symbol = bisect_right(cdf, value) - 1
        if not 0 <= symbol < len(cdf) - 1:
            raise ValueError("Invalid arithmetic stream")
        self.high = self.low + width * cdf[symbol + 1] // cdf[-1] - 1
        self.low += width * cdf[symbol] // cdf[-1]
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low *= 2
            self.high = self.high * 2 + 1
            self.code = self.code * 2 + self.reader.read()
        return symbol
