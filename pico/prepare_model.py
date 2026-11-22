#!/usr/bin/env python3
"""Export trained MCNN int8 weights to a deterministic integer Pico model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_compression.corpus import read_messages
from llm_compression.nn_model import NeuralModel, shapes, HEADER


def multiplier(coefficient, maximum, pre_shift=0):
    """Keep every signed Viper intermediate below 2**30, with headroom."""
    for shift in range(24, -1, -1):
        value = round(coefficient * (1 << (shift + pre_shift)))
        if value > 0 and ((maximum + (1 << pre_shift) - 1) >> pre_shift) * value < 1 << 30:
            return value, shift
    raise ValueError('Cannot represent layer scaling safely')


def export(source, destination):
    model = NeuralModel(source)
    if model.storage != 'int8' or model.context < 1:
        raise ValueError('Pico exporter needs int8 weights and a nonzero context')
    data = Path(source).read_bytes()
    offset, arrays, scales = HEADER.size, [], []
    for shape in shapes(model.context, model.embedding, model.hidden):
        count = math.prod(shape)
        scales.append(struct.unpack_from('<f', data, offset)[0])
        offset += 4
        arrays.append(np.frombuffer(data, dtype='i1', count=count, offset=offset).reshape(shape))
        offset += count
    embedding, w1, b1, w2, b2 = arrays
    width = model.context * model.embedding
    max1 = width * 128 * 128
    max2 = model.hidden * 4096 * 128
    if max(max1, max2) >= 1 << 30:
        raise ValueError('Model too large for signed 32-bit accumulators')
    mul1, shift1 = multiplier(scales[0] * scales[1] * 256, max1)
    pre2 = 8
    mul2, shift2 = multiplier(scales[3] * 256 / 4096, max2, pre2)
    unsigned = lambda value: (value.astype(np.int16) + 128).astype('u1').tobytes()
    body = b''.join((
        unsigned(embedding), unsigned(w1.T),
        np.rint(b1.astype(np.float64) * scales[2] * 256).astype('<i4').tobytes(),
        unsigned(w2.T),
        np.rint(b2.astype(np.float64) * scales[4] * 256).astype('<i4').tobytes(),
        np.array([round(math.tanh(i / 256) * 4096) for i in range(-2048, 2049)], dtype='<i2').tobytes(),
        np.array([round(math.exp(-i / 256) * 32768) for i in range(4097)], dtype='<u2').tobytes(),
    ))
    prefix = struct.pack('<8s8I', b'MCPNN001', model.context, model.embedding,
                         model.hidden, mul1, shift1, mul2, shift2, pre2)
    exported = prefix + hashlib.sha256(prefix + body).digest() + body
    Path(destination).write_bytes(exported)
    return {
        'format': 'MCPNN001', 'source': str(source), 'source_sha256': model.sha256,
        'source_bytes': model.size_bytes, 'export_bytes': len(exported),
        'export_sha256': hashlib.sha256(exported).hexdigest(),
        'context': model.context, 'embedding': model.embedding, 'hidden': model.hidden,
        'dense_macs_per_byte': model.hidden * (width + 256),
        'mul1': mul1, 'shift1': shift1, 'mul2': mul2, 'shift2': shift2, 'pre2': pre2,
        'max_abs_dot1': max1, 'max_abs_dot2': max2,
        'max_scaled_dot1': max1 * mul1,
        'max_scaled_dot2': (max2 >> pre2) * mul2,
        'notes': ['Same trained int8 weights; transposed dense matrices and integer activation/probability arithmetic.',
                  'Not wire-compatible with the original float64 MCNN codec. Both peers need this PNN file.',
                  'Model and probability tables are pre-shared. Message length and integrity framing are external.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/nn-cuda-sweep-mc-small/32k-context3/model.mcnn'))
    parser.add_argument('--output', type=Path, default=Path('models/pico-32k-context3'))
    parser.add_argument('--corpus', type=Path, default=Path('data/messages.jsonl'))
    parser.add_argument('--messages', type=int, default=30)
    args = parser.parse_args()
    if args.messages < 1:
        parser.error('--messages must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = export(args.model, args.output / 'model.pnn')
    from nn_runtime import Model, advance
    from nn_codec import compress, decompress
    runtime = Model(args.output / 'model.pnn', backend='python')
    original = NeuralModel(args.model)
    messages = [text.encode('utf-8') for text in read_messages(args.corpus) if text]
    selected = random.Random(42).sample(messages, min(args.messages, len(messages)))
    cases = []
    reference_bits = integer_bits = 0.0
    for message in selected:
        history = [256] * runtime.context
        for byte in message:
            cdf = runtime.cdf(history)
            integer_bits -= math.log2((cdf[byte + 1] - cdf[byte]) / cdf[-1])
            cdf = original.cdf(history)
            reference_bits -= math.log2((cdf[byte + 1] - cdf[byte]) / cdf[-1])
            advance(history, byte)
        payload = compress(message, runtime)
        assert decompress(payload, len(message), runtime) == message
        cases.append({'message_hex': message.hex(), 'payload_hex': payload.hex()})
    # Include binary, BOS, Unicode and empty inputs in cross-platform correctness.
    checks = [b'', bytes(range(256)), 'Pico W: caf\u00e9 \U0001f4fb'.encode(), b'hello mesh']
    vectors = []
    for message in checks:
        payload = compress(message, runtime)
        assert decompress(payload, len(message), runtime) == message
        vectors.append({'message_hex': message.hex(), 'payload_hex': payload.hex()})
    rng = random.Random(431)
    contexts = [[256] * runtime.context] + [
        [rng.randrange(257) for _ in range(runtime.context)] for _ in range(31)]
    cdfs = []
    for history in contexts:
        cdf = runtime.cdf(history)
        packed = struct.pack('<257I', *cdf)
        cdfs.append({'history': history, 'sha256': hashlib.sha256(packed).hexdigest()})
    count_bytes = sum(len(message) for message in selected)
    metadata['host_sample'] = {
        'messages': len(selected), 'original_bytes': count_bytes,
        'original_float64_model_bits_per_byte': reference_bits / count_bytes,
        'integer_model_bits_per_byte': integer_bits / count_bytes,
        'integer_payload_bits_per_byte': 8 * sum(len(bytes.fromhex(c['payload_hex'])) for c in cases) / count_bytes,
        'all_round_trips_exact': True, 'seed': 42,
        'note': 'Sample from source corpus, may overlap training; not a held-out quality measurement.',
    }
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'vectors.json').write_text(json.dumps({'cases': cases, 'checks': vectors, 'cdfs': cdfs}))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
