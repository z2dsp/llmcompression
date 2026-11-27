"""Size-bounded T9 neural weights with portable NumPy-only FP32 inference."""
import hashlib
import math
from pathlib import Path
import struct

import numpy as np

MAGIC = b'T9NNLM01'
PREFIX = struct.Struct('<8s7I')
HEADER = struct.Struct('<8s7I32s')


def shapes(context, key_context, embedding, hidden, hidden_layers=3):
    if context < 0 or key_context < 1 or embedding < 1 or hidden < 1 or hidden_layers not in (1, 2, 3):
        raise ValueError('Context must be nonnegative; key context, embedding and width positive; layers 1..3')
    return ([(257, embedding), (10, embedding),
             ((context + key_context) * embedding, hidden), (hidden,)]
            + [(hidden, hidden), (hidden,)] * (hidden_layers - 1)
            + [(hidden, 26), (26,)])


def model_size(context, key_context, embedding, hidden, storage, hidden_layers=3):
    if storage not in ('int8', 'float32'):
        raise ValueError('Storage must be int8 or float32')
    tensors = shapes(context, key_context, embedding, hidden, hidden_layers)
    return (HEADER.size + sum(math.prod(shape) for shape in tensors) * (1 if storage == 'int8' else 4)
            + (4 * len(tensors) if storage == 'int8' else 0))


def layout(budget_kib, context, key_context, embedding, storage, hidden_layers=3):
    if not math.isfinite(budget_kib) or budget_kib <= 0:
        raise ValueError('Model budget must be positive and finite')
    budget = int(budget_kib * 1024)
    minimum = model_size(context, key_context, embedding, 1, storage, hidden_layers)
    if minimum > budget:
        raise ValueError(f'Budget too small: at least {minimum} bytes required')
    low, high = 1, budget
    while low < high:
        middle = (low + high + 1) // 2
        if model_size(context, key_context, embedding, middle, storage, hidden_layers) <= budget:
            low = middle
        else:
            high = middle - 1
    return low, model_size(context, key_context, embedding, low, storage, hidden_layers)


def export_model(path, arrays, context, key_context, embedding, hidden, storage, hidden_layers=3):
    tensors = shapes(context, key_context, embedding, hidden, hidden_layers)
    if storage not in ('int8', 'float32') or len(arrays) != len(tensors):
        raise ValueError(f'Expected {len(tensors)} tensors and int8 or float32 storage')
    blocks = []
    for array, shape in zip(arrays, tensors):
        array = np.asarray(array, dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('Invalid T9 weights')
        if storage == 'int8':
            scale = np.float32(max(float(np.max(np.abs(array), initial=0)) / 127, 1e-12))
            blocks.extend([struct.pack('<f', scale),
                           np.clip(np.rint(array / scale), -127, 127).astype('i1').tobytes()])
        else:
            blocks.append(array.astype('<f4').tobytes())
    body = b''.join(blocks)
    fields = (MAGIC, context, key_context, embedding, hidden, hidden_layers, int(storage == 'int8'), len(body))
    prefix = PREFIX.pack(*fields)
    digest = hashlib.sha256(prefix + body).digest()
    Path(path).write_bytes(HEADER.pack(*fields, digest) + body)


class T9Model:
    def __init__(self, path):
        data = Path(path).read_bytes()
        if len(data) < HEADER.size:
            raise ValueError('Truncated T9 model')
        (magic, self.context, self.key_context, self.embedding, self.hidden,
         self.hidden_layers, kind, size, digest) = HEADER.unpack_from(data)
        if magic != MAGIC or kind not in (0, 1):
            raise ValueError('Unsupported T9 model')
        self.storage = 'int8' if kind else 'float32'
        tensors = shapes(self.context, self.key_context, self.embedding, self.hidden, self.hidden_layers)
        if (len(data) != HEADER.size + size
                or len(data) != model_size(self.context, self.key_context, self.embedding,
                                          self.hidden, self.storage, self.hidden_layers)):
            raise ValueError('T9 model length mismatch')
        if hashlib.sha256(data[:PREFIX.size] + data[HEADER.size:]).digest() != digest:
            raise ValueError('T9 model checksum mismatch')
        offset, self.arrays = HEADER.size, []
        for shape in tensors:
            count = math.prod(shape)
            if kind:
                scale = struct.unpack_from('<f', data, offset)[0]
                offset += 4
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError('Invalid T9 quantization scale')
                array = np.frombuffer(data, dtype='i1', count=count, offset=offset).astype(np.float32) * scale
                offset += count
            else:
                array = np.frombuffer(data, dtype='<f4', count=count, offset=offset).astype(np.float32)
                offset += count * 4
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite T9 weights')
            self.arrays.append(array.reshape(shape))
        self.size_bytes = len(data)
        self.sha256 = hashlib.sha256(data).hexdigest()

    def logits(self, histories, keys):
        histories = np.asarray(histories, dtype=np.int64)
        keys = np.asarray(keys, dtype=np.int64)
        if histories.ndim != 2 or histories.shape[1] != self.context or keys.shape != (len(histories), self.key_context):
            raise ValueError('Incorrect T9 input dimensions')
        if np.any((histories < 0) | (histories > 256)) or np.any((keys < 0) | (keys > 9)):
            raise ValueError('Invalid T9 input indices')
        text_features = self.arrays[0][histories].reshape(len(histories), self.context * self.embedding)
        key_features = self.arrays[1][keys].reshape(len(keys), self.key_context * self.embedding)
        features = np.concatenate((text_features, key_features), axis=1)
        for index in range(2, len(self.arrays) - 2, 2):
            features = np.tanh(features @ self.arrays[index] + self.arrays[index + 1])
        return features @ self.arrays[-2] + self.arrays[-1]
