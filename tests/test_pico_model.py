"""Host checks for the same integer codec executed by Viper on the Pico."""
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pytest

from llm_compression.nn_model import compress_payload, decompress_payload, export_model, shapes

PICO = Path(__file__).resolve().parents[1] / 'pico'


def load(name):
    spec = importlib.util.spec_from_file_location(name, PICO / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def implementation(monkeypatch):
    runtime = load('nn_runtime')
    monkeypatch.setitem(sys.modules, 'nn_runtime', runtime)
    return runtime, load('nn_codec'), load('prepare_model')


@pytest.fixture(params=[1, 2, 3])
def model(tmp_path, implementation, request):
    runtime, codec, prepare = implementation
    source, destination = tmp_path / 'model.mcnn', tmp_path / 'model.pnn'
    rng = np.random.default_rng(42)
    arrays = [rng.normal(0, 0.2, size=shape).astype(np.float32) for shape in shapes(3, 2, 5, request.param)]
    export_model(source, arrays, 3, 2, 5, 'int8', request.param)
    metadata = prepare.export(source, destination)
    return runtime.Model(destination, backend='python'), codec, metadata, destination


def test_pico_cdf_is_positive_normalized_and_repeatable(model):
    runtime, _, metadata, _ = model
    rng = random.Random(42)
    for history in [[256] * 3] + [[rng.randrange(257) for _ in range(3)] for _ in range(20)]:
        cdf = list(runtime.cdf(history))
        assert cdf[0] == 0 and cdf[-1] == 65536
        assert all(a < b for a, b in zip(cdf, cdf[1:]))
        assert list(runtime.cdf(history)) == cdf
        runtime.cdf([0, 1, 2])
        assert list(runtime.cdf(history)) == cdf
    assert metadata['max_scaled_dot1'] < 1 << 30
    assert metadata['max_scaled_dot2'] < 1 << 30
    assert metadata['hidden_layers'] == runtime.hidden_layers


@pytest.mark.parametrize('message', [b'', b'hello mesh', bytes(range(256)), b'\x00' * 80,
                                     'Pico: caf\u00e9 \U0001f4fb'.encode()])
def test_pico_codec_matches_project_arithmetic_coder(model, message):
    runtime, codec, _, _ = model
    payload = codec.compress(message, runtime)
    assert payload == compress_payload(message, runtime)
    assert codec.decompress(payload, len(message), runtime) == message
    assert decompress_payload(payload, len(message), runtime) == message


def test_pico_model_checksum_and_truncation(model, implementation):
    _, _, _, path = model
    runtime = implementation[0]
    data = path.read_bytes()
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError, match='Truncated'):
        runtime.Model(path, backend='python')
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match='checksum'):
        runtime.Model(path, backend='python')


def test_pico_rejects_wrong_context_and_unsupported_backend(model, implementation):
    runtime, codec, _, path = model
    with pytest.raises(ValueError, match='context'):
        runtime.cdf([0])
    with pytest.raises(ValueError, match='Context symbols'):
        runtime.cdf([0, 1, 257])
    with pytest.raises(ValueError, match='Negative'):
        codec.decompress(b'', -1, runtime)
    with pytest.raises(ValueError, match='MicroPython'):
        implementation[0].Model(path, backend='viper')
