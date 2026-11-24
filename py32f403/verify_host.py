#!/usr/bin/env python3
"""Compile the portable C codec on the host and verify desktop/Pico vectors."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def load_library(build):
    build = Path(build).resolve()
    wrapper = build / 'host_wrapper.c'
    wrapper.write_text('#include "nn.h"\nsize_t nn_state_size(void){return sizeof(NNState);}\n'
                       'void nn_set_history(NNState *s,const uint16_t *h){'
                       'for(unsigned i=0;i<NN_CONTEXT;++i)s->history[i]=h[i];}\n')
    library = build / 'nn_host.so'
    subprocess.run(['gcc', '-O3', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=undefined', '-fno-sanitize-recover=all',
                    '-I' + str(HERE), '-I' + str(build), str(HERE / 'nn.c'),
                    str(HERE / 'codec.c'), str(build / 'model.c'), str(wrapper),
                    '-o', str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    lib.nn_state_size.restype = ctypes.c_size_t
    lib.nn_set_history.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16)]
    lib.nn_predict.argtypes = [ctypes.c_void_p]
    lib.nn_predict.restype = ctypes.POINTER(ctypes.c_uint32)
    lib.nn_compress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.nn_compress.restype = ctypes.c_int
    lib.nn_decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.nn_decompress.restype = ctypes.c_int
    return lib


def verify(build, vectors):
    lib = load_library(build)
    state = ctypes.create_string_buffer(lib.nn_state_size())
    output = ctypes.create_string_buffer(4096)
    for case in vectors['cdfs']:
        history = (ctypes.c_uint16 * len(case['history']))(*case['history'])
        lib.nn_set_history(state, history)
        cdf = lib.nn_predict(state)
        if hashlib.sha256(ctypes.string_at(cdf, 257 * 4)).hexdigest() != case['sha256']:
            raise ValueError('C/Python CDF mismatch')
    for case in vectors['checks'] + vectors['cases']:
        message = bytes.fromhex(case['message_hex'])
        expected = bytes.fromhex(case['payload_hex'])
        size = lib.nn_compress(state, message, len(message), output, len(output))
        if size < 0 or output.raw[:size] != expected:
            raise ValueError('C/Python encode mismatch')
        size = lib.nn_decompress(state, expected, len(expected), len(message), output, len(output))
        if size != len(message) or output.raw[:size] != message:
            raise ValueError('C/Python decode mismatch')
    return {'cdf_contexts': len(vectors['cdfs']), 'codec_vectors': len(vectors['checks']) + len(vectors['cases']),
            'state_bytes': lib.nn_state_size(), 'undefined_behavior_sanitizer': True,
            'all_cdfs_and_payloads_match': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=HERE / 'build')
    parser.add_argument('--vectors', type=Path, default=HERE.parent / 'models/pico-32k-context3/vectors.json')
    args = parser.parse_args()
    print(json.dumps(verify(args.build, json.loads(args.vectors.read_text())), indent=2))
