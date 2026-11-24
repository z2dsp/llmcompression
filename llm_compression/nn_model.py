"""Portable fixed-context neural byte predictor and arithmetic payload codec."""
import hashlib
import math
from pathlib import Path
import struct

import numpy as np

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder

LEGACY_MAGIC = b'MCNNLM01'
MAGIC = b'MCNNLM03'
MODEL_MAGICS = {1: LEGACY_MAGIC, 2: b'MCNNLM02', 3: MAGIC}
HEADER = struct.Struct('<8s5I32s')
TOTAL = 1 << 20


def shapes(context, embedding, hidden, hidden_layers=3):
    if hidden_layers not in MODEL_MAGICS:
        raise ValueError('Expected one, two, or three hidden layers')
    tensors = [(257, embedding), (context * embedding, hidden), (hidden,)]
    tensors += [(hidden, hidden), (hidden,)] * (hidden_layers - 1)
    return tensors + [(hidden, 256), (256,)]


def model_size(context, embedding, hidden, storage, hidden_layers=3):
    tensors = shapes(context, embedding, hidden, hidden_layers)
    count = sum(math.prod(shape) for shape in tensors)
    return HEADER.size + count * (4 if storage == 'float32' else 1) + (4 * len(tensors) if storage == 'int8' else 0)


def layout(budget_kib, context, embedding, storage, hidden_layers=3):
    if budget_kib <= 0 or context < 0 or embedding <= 0 or storage not in ('float32', 'int8'):
        raise ValueError('Budget/embedding must be positive, context nonnegative, storage float32 or int8')
    budget = budget_kib * 1024
    minimum = model_size(context, embedding, 1, storage, hidden_layers)
    if minimum > budget:
        raise ValueError(f'Budget too small: at least {minimum} bytes needed for these settings')
    # The middle dense matrices make size quadratic in width. Integer binary
    # search avoids floating-point rounding even for very large budgets.
    low, high = 1, budget // (4 if storage == 'float32' else 1)
    while low < high:
        middle = (low + high + 1) // 2
        if model_size(context, embedding, middle, storage, hidden_layers) <= budget:
            low = middle
        else:
            high = middle - 1
    return low, model_size(context, embedding, low, storage, hidden_layers)


def export_model(path, arrays, context, embedding, hidden, storage, hidden_layers=3):
    tensors = shapes(context, embedding, hidden, hidden_layers)
    if storage not in ('float32', 'int8') or len(arrays) != len(tensors):
        raise ValueError(f'Expected {len(tensors)} tensors and float32 or int8 storage')
    blocks = []
    for array, shape in zip(arrays, tensors):
        array = np.asarray(array, dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('Invalid neural weights')
        if storage == 'int8':
            scale = np.float32(max(float(np.max(np.abs(array), initial=0)) / 127, 1e-12))
            blocks.append(struct.pack('<f', scale))
            blocks.append(np.clip(np.rint(array / scale), -127, 127).astype('i1').tobytes())
        else:
            blocks.append(array.astype('<f4').tobytes())
    body = b''.join(blocks)
    magic = MODEL_MAGICS[hidden_layers]
    fields = (magic, context, embedding, hidden, int(storage == 'int8'), len(body))
    prefix = struct.pack('<8s5I', *fields)
    digest = hashlib.sha256(prefix + body).digest()
    Path(path).write_bytes(HEADER.pack(*fields, digest) + body)


class NeuralModel:
    def __init__(self, path):
        data = Path(path).read_bytes()
        if len(data) < HEADER.size:
            raise ValueError('Truncated neural model')
        magic, self.context, self.embedding, self.hidden, kind, size, digest = HEADER.unpack_from(data)
        if magic not in MODEL_MAGICS.values() or kind not in (0, 1) or self.embedding < 1 or self.hidden < 1:
            raise ValueError('Unsupported neural model')
        self.hidden_layers = next(depth for depth, value in MODEL_MAGICS.items() if value == magic)
        self.storage = 'int8' if kind else 'float32'
        if len(data) != HEADER.size + size or len(data) != model_size(self.context, self.embedding, self.hidden, self.storage, self.hidden_layers):
            raise ValueError('Neural model length mismatch')
        if hashlib.sha256(data[:HEADER.size - 32] + data[HEADER.size:]).digest() != digest:
            raise ValueError('Neural model checksum mismatch')
        offset, arrays = HEADER.size, []
        for shape in shapes(self.context, self.embedding, self.hidden, self.hidden_layers):
            count = math.prod(shape)
            if kind:
                scale = struct.unpack_from('<f', data, offset)[0]
                offset += 4
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError('Invalid quantization scale')
                array = np.frombuffer(data, dtype='i1', count=count, offset=offset).astype(np.float64) * scale
                offset += count
            else:
                array = np.frombuffer(data, dtype='<f4', count=count, offset=offset).astype(np.float64)
                offset += count * 4
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite neural weights')
            arrays.append(array.reshape(shape))
        self.arrays, self.size_bytes = arrays, len(data)
        self.sha256 = hashlib.sha256(data).hexdigest()

    def logits(self, history):
        """Raw next-byte scores, before softmax and codec quantization."""
        if len(history) != self.context:
            raise ValueError('Incorrect neural context length')
        x = self.arrays[0][np.asarray(history, dtype=np.int64)].reshape(-1)
        for index in range(1, len(self.arrays) - 2, 2):
            x = np.tanh(x @ self.arrays[index] + self.arrays[index + 1])
        return x @ self.arrays[-2] + self.arrays[-1]

    def cdf(self, history):
        logits = self.logits(history)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return probabilities_to_cdf(probabilities)


def probabilities_to_cdf(probabilities):
    # A fixed total and one count per byte preserve support for arbitrary bytes.
    scaled = probabilities * (TOTAL - 256)
    frequencies = np.floor(scaled).astype(np.int64) + 1
    remaining = TOTAL - int(frequencies.sum())
    indices = np.argsort(-(scaled - np.floor(scaled)), kind='stable')[:remaining]
    frequencies[indices] += 1
    return np.concatenate(([0], np.cumsum(frequencies))).tolist()


def advance(history, byte):
    if history:
        history.pop(0)
        history.append(byte)


def compress_payload(message, model):
    coder = ArithmeticEncoder()
    history = [256] * model.context
    for byte in message:
        cdf = model.cdf(history)
        coder.write(cdf, byte)
        advance(history, byte)
    return coder.finish()


def decompress_payload(payload, length, model):
    coder = ArithmeticDecoder(payload)
    history, result = [256] * model.context, bytearray()
    for _ in range(length):
        byte = coder.read(model.cdf(history))
        result.append(byte)
        advance(history, byte)
    return bytes(result)
