"""Integer neural byte predictor for MicroPython, with a CPython reference path.

PNN1 is deliberately distinct from the float64 MCNN codec. Both peers must use
the same PNN file and this integer probability allocation algorithm.
"""
from array import array
import gc
import hashlib
import struct

try:
    import micropython
except ImportError:
    micropython = None

TOTAL = 65536
HEADER = '<8s8I32s'
HEADER_SIZE = 72


def _dot_python(weights, inputs, outputs, width):
    for row in range(len(outputs)):
        value = 0
        offset = row * width
        for col in range(width):
            value += inputs[col] * (weights[offset + col] - 128)
        outputs[row] = value


if micropython is not None:
    @micropython.viper
    def _dot_viper(weights, inputs, outputs, width: int):
        w = ptr8(weights)
        x = ptr32(inputs)
        y = ptr32(outputs)
        count = int(len(outputs))
        for row in range(count):
            acc = int(0)
            offset = row * width
            for col in range(width):
                acc += int(x[col]) * (int(w[offset + col]) - 128)
            y[row] = acc

    @micropython.viper
    def _hidden_viper(values, biases, table, parameters):
        h = ptr32(values)
        b = ptr32(biases)
        lut = ptr16(table)
        p = ptr32(parameters)
        count = int(p[0])
        multiplier = int(p[1])
        shift = int(p[2])
        for i in range(count):
            index = ((int(h[i]) * multiplier) >> shift) + int(b[i])
            if index < -2048:
                index = -2048
            elif index > 2048:
                index = 2048
            value = int(lut[index + 2048])
            if value >= 32768:
                value -= 65536
            h[i] = value

    @micropython.viper
    def _frequencies_viper(logits, table, workspace, parameters):
        l = ptr32(logits)
        lut = ptr16(table)
        w = ptr32(workspace)
        p = ptr32(parameters)
        multiplier = int(p[0])
        shift = int(p[1])
        pre = int(p[2])
        maximum = int(-2147483647)
        for i in range(256):
            value = (((int(l[i]) >> pre) * multiplier) >> shift) + int(p[3 + i])
            l[i] = value
            if value > maximum:
                maximum = value
        mass = int(0)
        for i in range(256):
            distance = maximum - int(l[i])
            score = int(0)
            if distance <= 4096:
                score = int(lut[distance])
            w[i] = score
            mass += score
        allocated = int(0)
        for i in range(256):
            numerator = int(w[i]) * 65280
            frequency = 1 + numerator // mass
            w[i] = frequency
            w[256 + i] = numerator % mass
            w[512 + i] = i
            allocated += frequency
        # Stable insertion sort, descending residual; ties remain in byte order.
        for i in range(1, 256):
            symbol = int(w[512 + i])
            remainder = int(w[256 + symbol])
            j = i
            while j > 0:
                other = int(w[512 + j - 1])
                if int(w[256 + other]) >= remainder:
                    break
                w[512 + j] = other
                j -= 1
            w[512 + j] = symbol
        for i in range(65536 - allocated):
            symbol = int(w[512 + i])
            w[symbol] = int(w[symbol]) + 1
        w[768] = 0
        cumulative = int(0)
        for i in range(256):
            cumulative += int(w[i])
            w[769 + i] = cumulative


class Model:
    def __init__(self, path, backend='viper'):
        if backend not in ('python', 'viper'):
            raise ValueError('backend must be python or viper')
        if backend == 'viper' and micropython is None:
            raise ValueError('Viper requires MicroPython')
        with open(path, 'rb') as stream:
            header = stream.read(HEADER_SIZE)
            if len(header) != HEADER_SIZE:
                raise ValueError('Truncated PNN header')
            fields = struct.unpack(HEADER, header)
            if fields[0] != b'MCPNN001':
                raise ValueError('Unsupported PNN format')
            (self.context, self.embedding, self.hidden, self.mul1, self.shift1,
             self.mul2, self.shift2, self.pre2) = fields[1:9]
            if self.context < 1 or self.embedding < 1 or self.hidden < 1:
                raise ValueError('Invalid PNN dimensions')
            if not (0 <= self.shift1 <= 24 and 0 <= self.shift2 <= 24
                    and 0 <= self.pre2 <= 24 and 0 < self.mul1 < 2**30
                    and 0 < self.mul2 < 2**30):
                raise ValueError('Invalid PNN scaling')
            max1 = self.context * self.embedding * 128 * 128
            max2 = self.hidden * 4096 * 128
            if (max(max1, max2) >= 2**30 or max1 * self.mul1 >= 2**30
                    or ((max2 + (1 << self.pre2) - 1) >> self.pre2) * self.mul2 >= 2**30):
                raise ValueError('PNN scaling exceeds signed Viper accumulator bounds')
            digest = hashlib.sha256(header[:40])

            def read(size):
                block = stream.read(size)
                if len(block) != size:
                    raise ValueError('Truncated PNN weights')
                digest.update(block)
                return block

            self.embeddings = read(257 * self.embedding)
            self.w1 = read(self.context * self.embedding * self.hidden)
            self.b1 = array('i', read(self.hidden * 4))
            self.w2 = read(self.hidden * 256)
            self.b2 = array('i', read(256 * 4))
            self.tanh = array('h', read(4097 * 2))
            self.exp = array('H', read(4097 * 2))
            if stream.read(1) or digest.digest() != fields[-1]:
                raise ValueError('PNN checksum/length mismatch')
        self.backend = backend
        self.dot = _dot_viper if backend == 'viper' else _dot_python
        self.x = array('i', [0] * (self.context * self.embedding))
        self.h = array('i', [0] * self.hidden)
        self.logits = array('i', [0] * 256)
        self.scores = array('i', [0] * 256)
        self.remainders = array('i', [0] * 256)
        self.cumulative = array('i', [0] * 257)
        self.workspace = array('i', [0] * 1025)
        self.viper_cdf = memoryview(self.workspace)[768:]
        self.hidden_parameters = array('i', [self.hidden, self.mul1, self.shift1])
        self.frequency_parameters = array('i', [self.mul2, self.shift2, self.pre2])
        self.frequency_parameters.extend(self.b2)
        self.indices = list(range(256))
        # Explicit tie key: do not depend on MicroPython's sort stability.
        self._remainder_key = lambda i: (self.remainders[i], -i)
        gc.collect()

    def features(self, history):
        if len(history) != self.context:
            raise ValueError('Incorrect context length')
        for pos, symbol in enumerate(history):
            if not 0 <= symbol <= 256:
                raise ValueError('Context symbols must be bytes or BOS=256')
            offset = symbol * self.embedding
            start = pos * self.embedding
            for col in range(self.embedding):
                self.x[start + col] = self.embeddings[offset + col] - 128

    def hidden_activation(self):
        if self.backend == 'viper':
            _hidden_viper(self.h, self.b1, self.tanh, self.hidden_parameters)
            return
        for i in range(self.hidden):
            index = ((self.h[i] * self.mul1) >> self.shift1) + self.b1[i]
            index = min(2048, max(-2048, index)) + 2048
            self.h[i] = self.tanh[index]

    def frequencies(self):
        if self.backend == 'viper':
            _frequencies_viper(self.logits, self.exp, self.workspace, self.frequency_parameters)
            return self.viper_cdf
        maximum = -2**31
        for i in range(256):
            value = (((self.logits[i] >> self.pre2) * self.mul2)
                     >> self.shift2) + self.b2[i]
            self.logits[i] = value
            maximum = max(maximum, value)
        mass = 0
        for i in range(256):
            distance = maximum - self.logits[i]
            score = self.exp[distance] if distance <= 4096 else 0
            self.scores[i] = score
            mass += score
        allocated = 0
        for i in range(256):
            numerator = self.scores[i] * (TOTAL - 256)
            frequency = 1 + numerator // mass
            self.scores[i] = frequency
            self.remainders[i] = numerator % mass
            allocated += frequency
        # Ties must go to the lower byte value on both peers.
        for i in range(256):
            self.indices[i] = i
        self.indices.sort(key=self._remainder_key, reverse=True)
        for i in range(TOTAL - allocated):
            self.scores[self.indices[i]] += 1
        cdf = self.cumulative
        cdf[0] = 0
        for i in range(256):
            cdf[i + 1] = cdf[i] + self.scores[i]
        return cdf

    def cdf(self, history):
        self.features(history)
        self.dot(self.w1, self.x, self.h, len(self.x))
        self.hidden_activation()
        self.dot(self.w2, self.h, self.logits, self.hidden)
        return self.frequencies()


def advance(history, byte):
    if history:
        history.pop(0)
        history.append(byte)
