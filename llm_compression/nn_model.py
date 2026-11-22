"""Portable fixed-context neural byte predictor and arithmetic payload codec."""
import hashlib
import math
from pathlib import Path
import struct

import numpy as np

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder

MAGIC = b'MCNNLM01'
HEADER = struct.Struct('<8s5I32s')
TOTAL = 1 << 20


def shapes(context, embedding, hidden):
    return [(257, embedding), (context * embedding, hidden), (hidden,), (hidden, 256), (256,)]


def model_size(context, embedding, hidden, storage):
    count = sum(math.prod(shape) for shape in shapes(context, embedding, hidden))
    return HEADER.size + count * (4 if storage == 'float32' else 1) + (20 if storage == 'int8' else 0)


def layout(budget_kib, context, embedding, storage):
    if budget_kib <= 0 or context < 0 or embedding <= 0 or storage not in ('float32', 'int8'):
        raise ValueError('Budget/embedding must be positive, context nonnegative, storage float32 or int8')
    budget = budget_kib * 1024
    minimum = model_size(context, embedding, 1, storage)
    if minimum > budget:
        raise ValueError(f'Budget too small: at least {minimum} bytes needed for these settings')
    step = model_size(context, embedding, 2, storage) - minimum
    hidden = 1 + (budget - minimum) // step
    return hidden, model_size(context, embedding, hidden, storage)


def export_model(path, arrays, context, embedding, hidden, storage):
    if storage not in ('float32', 'int8') or len(arrays) != 5:
        raise ValueError('Expected five tensors and float32 or int8 storage')
    blocks = []
    for array, shape in zip(arrays, shapes(context, embedding, hidden)):
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
    fields = (MAGIC, context, embedding, hidden, int(storage == 'int8'), len(body))
    prefix = struct.pack('<8s5I', *fields)
    digest = hashlib.sha256(prefix + body).digest()
    Path(path).write_bytes(HEADER.pack(*fields, digest) + body)


class NeuralModel:
    def __init__(self, path):
        data = Path(path).read_bytes()
        if len(data) < HEADER.size:
            raise ValueError('Truncated neural model')
        magic, self.context, self.embedding, self.hidden, kind, size, digest = HEADER.unpack_from(data)
        if magic != MAGIC or kind not in (0, 1) or self.embedding < 1 or self.hidden < 1:
            raise ValueError('Unsupported neural model')
        self.storage = 'int8' if kind else 'float32'
        if len(data) != HEADER.size + size or len(data) != model_size(self.context, self.embedding, self.hidden, self.storage):
            raise ValueError('Neural model length mismatch')
        if hashlib.sha256(data[:HEADER.size - 32] + data[HEADER.size:]).digest() != digest:
            raise ValueError('Neural model checksum mismatch')
        offset, arrays = HEADER.size, []
        for shape in shapes(self.context, self.embedding, self.hidden):
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

    def cdf(self, history):
        if len(history) != self.context:
            raise ValueError('Incorrect neural context length')
        embedding, w1, b1, w2, b2 = self.arrays
        x = embedding[np.asarray(history, dtype=np.int64)].reshape(-1)
        logits = np.tanh(x @ w1 + b1) @ w2 + b2
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
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
