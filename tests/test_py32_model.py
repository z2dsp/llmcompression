"""C/Python interoperability and bounds checks for the RAM-constrained port."""
import ctypes
import shutil

import numpy as np
import pytest

from llm_compression.nn_model import compress_payload, export_model, shapes
from pico.nn_runtime import Model
from pico.prepare_model import export
from py32f403.build import generate_model
from py32f403.verify_host import load_library


@pytest.fixture(params=[1, 2, 3])
def native(tmp_path, request):
    if shutil.which('gcc') is None:
        pytest.skip('Host C compiler unavailable')
    rng = np.random.default_rng(42)
    arrays = [rng.normal(0, 0.2, size=shape).astype(np.float32) for shape in shapes(3, 2, 5, request.param)]
    source, packed = tmp_path / 'model.mcnn', tmp_path / 'model.pnn'
    export_model(source, arrays, 3, 2, 5, 'int8', request.param)
    export(source, packed)
    generate_model(packed, tmp_path)
    return load_library(tmp_path), Model(packed, backend='python'), packed, tmp_path


def test_c_probability_tables_match_python_after_context_changes(native):
    lib, reference, _, _ = native
    state = ctypes.create_string_buffer(lib.nn_state_size())
    rng = np.random.default_rng(19)
    histories = [[256] * 3, [0, 255, 0]] + rng.integers(0, 257, (64, 3)).tolist()
    for history in histories:
        lib.nn_set_history(state, (ctypes.c_uint16 * 3)(*history))
        cdf = lib.nn_predict(state)
        assert list(cdf[:257]) == list(reference.cdf(history))
    assert lib.nn_state_size() < 4096


@pytest.mark.parametrize('message', [b'', b'hello mesh', bytes(range(256)), b'\xff\x00' * 128,
                                     'Native PY32: caf\u00e9 \U0001f4fb'.encode()])
def test_native_arithmetic_stream_matches_reference(native, message):
    lib, reference, _, _ = native
    state = ctypes.create_string_buffer(lib.nn_state_size())
    output = ctypes.create_string_buffer(4096)
    expected = compress_payload(message, reference)
    size = lib.nn_compress(state, message, len(message), output, len(output))
    assert size == len(expected) and output.raw[:size] == expected
    size = lib.nn_decompress(state, expected, len(expected), len(message), output, len(output))
    assert size == len(message) and output.raw[:size] == message
    assert lib.nn_compress(state, message, len(message), output, 0) == -1
    if message:
        assert lib.nn_decompress(state, expected, len(expected), len(message), output, len(message) - 1) == -1


def test_c_model_generator_rejects_corruption(native):
    _, _, packed, output = native
    data = bytearray(packed.read_bytes())
    data[-1] ^= 1
    packed.write_bytes(data)
    with pytest.raises(ValueError, match='checksum'):
        generate_model(packed, output)
