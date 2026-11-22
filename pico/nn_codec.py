"""Portable 32-bit arithmetic coder. Python big integers avoid overflow here.

The dense layers use Viper; arithmetic interval products must NOT be moved to
32-bit Viper without implementing wide multiplication/division.
"""
from nn_runtime import advance

HALF = 1 << 31
QUARTER = 1 << 30
MASK = (1 << 32) - 1


class Bits:
    def __init__(self, data=b''):
        self.data = bytearray(data)
        self.position = 0
        self.byte = 0
        self.used = 0

    def write(self, bit):
        self.byte = (self.byte << 1) | bit
        self.used += 1
        if self.used == 8:
            self.data.append(self.byte)
            self.byte = self.used = 0

    def finish(self):
        if self.used:
            self.data.append(self.byte << (8 - self.used))
        return bytes(self.data)

    def read(self):
        pos = self.position
        self.position += 1
        if pos >= len(self.data) * 8:
            return 0
        return (self.data[pos // 8] >> (7 - pos % 8)) & 1


def compress(message, model):
    history = [256] * model.context
    low, high, pending = 0, MASK, 0
    bits = Bits()
    for symbol in message:
        cdf = model.cdf(history)
        width = high - low + 1
        high = low + width * cdf[symbol + 1] // cdf[-1] - 1
        low += width * cdf[symbol] // cdf[-1]
        while True:
            if high < HALF:
                bit = 0
            elif low >= HALF:
                bit = 1
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                pending += 1
                low -= QUARTER
                high -= QUARTER
                low *= 2
                high = high * 2 + 1
                continue
            else:
                break
            bits.write(bit)
            for _ in range(pending):
                bits.write(bit ^ 1)
            pending = 0
            low *= 2
            high = high * 2 + 1
        advance(history, symbol)
    pending += 1
    bit = 0 if low < QUARTER else 1
    bits.write(bit)
    for _ in range(pending):
        bits.write(bit ^ 1)
    return bits.finish()


def decompress(payload, length, model):
    if length < 0:
        raise ValueError('Negative message length')
    history = [256] * model.context
    low, high, code = 0, MASK, 0
    bits = Bits(payload)
    for _ in range(32):
        code = (code << 1) | bits.read()
    result = bytearray()
    for _ in range(length):
        cdf = model.cdf(history)
        width = high - low + 1
        value = ((code - low + 1) * cdf[-1] - 1) // width
        left, right = 0, 256
        while left + 1 < right:
            middle = (left + right) // 2
            if cdf[middle] <= value:
                left = middle
            else:
                right = middle
        symbol = left
        high = low + width * cdf[symbol + 1] // cdf[-1] - 1
        low += width * cdf[symbol] // cdf[-1]
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                code -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                low -= QUARTER
                high -= QUARTER
                code -= QUARTER
            else:
                break
            low *= 2
            high = high * 2 + 1
            code = code * 2 + bits.read()
        result.append(symbol)
        advance(history, symbol)
    return bytes(result)
