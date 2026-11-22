"""On-device benchmark; no Wi-Fi or radio transmissions and no boot hooks."""
from array import array
import binascii
import gc
import hashlib
import json
import machine
import os
import struct
import sys
import time

from nn_runtime import Model
from nn_codec import compress, decompress


def timed(call, iterations):
    gc.collect()
    start = time.ticks_us()
    for _ in range(iterations):
        call()
    return time.ticks_diff(time.ticks_us(), start) / iterations


def run(directory='/llm_pico_demo', iterations=30):
    gc.collect()
    before = gc.mem_free()
    model = Model(directory + '/model.pnn')
    loaded_free = gc.mem_free()
    with open(directory + '/vectors.json') as stream:
        vectors = json.load(stream)
    print('# model and vectors loaded')
    for item in vectors['cdfs']:
        values = model.cdf(item['history'])
        digest = hashlib.sha256(struct.pack('<257I', *values)).digest()
        if binascii.hexlify(digest).decode() != item['sha256']:
            raise ValueError('Desktop/Pico CDF mismatch')
    history = [256] * model.context
    print('# desktop CDF checks passed')
    # Same context and same complete probability path for both execution modes.
    model.cdf(history)
    viper_us = timed(lambda: model.cdf(history), iterations)
    from nn_runtime import _dot_python
    fast_dot = model.dot
    model.dot = _dot_python
    model.backend = 'python'
    python_us = timed(lambda: model.cdf(history), min(iterations, 5))
    model.dot = fast_dot
    model.backend = 'viper'
    print('# prediction timings measured')

    def dense1():
        model.dot(model.w1, model.x, model.h, len(model.x))

    def activation():
        # Activation modifies h, so restore a fixed preactivation each time.
        model.h[:] = hidden_raw
        model.hidden_activation()

    def dense2():
        model.dot(model.w2, model.h, model.logits, model.hidden)

    def frequencies():
        model.logits[:] = logits_raw
        model.frequencies()

    model.features(history)
    dense1()
    hidden_raw = array('i', model.h)
    dense1_us = timed(dense1, iterations)
    activation_us = timed(activation, iterations)
    model.h[:] = hidden_raw
    model.hidden_activation()
    dense2()
    logits_raw = array('i', model.logits)
    dense2_us = timed(dense2, iterations)
    frequencies_us = timed(frequencies, iterations)
    feature_us = timed(lambda: model.features(history), iterations)

    for item in vectors['checks']:
        message = binascii.unhexlify(item['message_hex'])
        expected = binascii.unhexlify(item['payload_hex'])
        if compress(message, model) != expected:
            raise ValueError('Desktop/Pico check payload mismatch')
        if decompress(expected, len(message), model) != message:
            raise ValueError('Desktop/Pico check decode mismatch')
    print('# extra codec checks passed')

    encoded_bytes = original_bytes = 0
    encode_us = decode_us = 0
    rows = []
    for item in vectors['cases']:
        message = binascii.unhexlify(item['message_hex'])
        expected = binascii.unhexlify(item['payload_hex'])
        start = time.ticks_us()
        payload = compress(message, model)
        enc = time.ticks_diff(time.ticks_us(), start)
        if payload != expected:
            raise ValueError('Desktop/Pico sample payload mismatch')
        start = time.ticks_us()
        restored = decompress(expected, len(message), model)
        dec = time.ticks_diff(time.ticks_us(), start)
        if restored != message:
            raise ValueError('Desktop/Pico sample decode mismatch')
        original_bytes += len(message)
        encoded_bytes += len(payload)
        encode_us += enc
        decode_us += dec
        rows.append({'original_bytes': len(message), 'payload_bytes': len(payload),
                     'encode_us': enc, 'decode_us': dec})
        if len(rows) % 5 == 0:
            print('# sample messages checked:', len(rows))
    gc.collect()
    fs = os.statvfs(directory)
    freq = machine.freq()
    result = {
        'firmware': str(sys.implementation), 'uname': list(os.uname()),
        'cpu_hz': freq, 'model_bytes': os.stat(directory + '/model.pnn')[6],
        'context': model.context, 'embedding': model.embedding, 'hidden': model.hidden,
        'iterations': iterations,
        'memory': {'free_before_load': before, 'free_after_load': loaded_free,
                   'model_heap_delta': before - loaded_free, 'free_after_benchmark': gc.mem_free(),
                   'allocated_after_benchmark': gc.mem_alloc(), 'free_flash_bytes': fs[0] * fs[3]},
        'prediction': {'viper_us_per_byte': viper_us, 'python_us_per_byte': python_us,
                       'viper_speedup': python_us / viper_us,
                       'estimated_clock_cycles_per_byte': freq * viper_us / 1000000,
                       'stage_us': {'features': feature_us, 'dense1': dense1_us,
                                    'tanh_lut': activation_us, 'dense2': dense2_us,
                                    'softmax_and_cdf': frequencies_us}},
        'sample': {'messages': len(rows), 'original_bytes': original_bytes,
                   'payload_bytes': encoded_bytes, 'bits_per_original_byte': 8 * encoded_bytes / original_bytes,
                   'encode_us': encode_us, 'decode_us': decode_us,
                   'encode_bytes_per_second': original_bytes * 1000000 / encode_us,
                   'decode_bytes_per_second': original_bytes * 1000000 / decode_us,
                   'all_round_trips_exact': True, 'all_payloads_match_desktop': True,
                   'extra_correctness_vectors': len(vectors['checks']),
                   'cdf_contexts_match_desktop': len(vectors['cdfs']), 'rows': rows},
        'notes': ['Wall-clock measurements at the existing CPU frequency; no overclocking or second core.',
                  'Cycle estimate is elapsed time times CPU frequency, not a hardware cycle-counter measurement.',
                  'Model is pre-shared; payload size excludes length, integrity, and radio framing.',
                  'Heap delta includes weights, lookup tables, and persistent inference buffers; modules already imported.',
                  'Stage timings include loop/call overhead and activation/softmax scratch restoration.',
                  'Private messages are not printed in this report.'],
    }
    return result
