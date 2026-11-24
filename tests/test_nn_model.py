import json
import math
import numpy as np
import pytest

from llm_compression.nn_model import (NeuralModel, TOTAL, compress_payload, decompress_payload,
                                      export_model, layout, model_size, shapes)
from llm_compression.nn_training import evaluate, main, training_batches
from llm_compression.training_corpus import RecordWriter


@pytest.mark.parametrize('storage', ['float32', 'int8'])
@pytest.mark.parametrize('context', [0, 1, 16])
@pytest.mark.parametrize('hidden_layers', [1, 2, 3])
def test_model_storage_and_coding(tmp_path, storage, context, hidden_layers):
    hidden, size = layout(8, context, 4, storage, hidden_layers)
    assert size <= 8192 < model_size(context, 4, hidden + 1, storage, hidden_layers)
    rng = np.random.default_rng(10)
    arrays = [rng.normal(0, 0.05, shape).astype('float32') for shape in shapes(context, 4, hidden, hidden_layers)]
    path = tmp_path / 'model.mcnn'
    export_model(path, arrays, context, 4, hidden, storage, hidden_layers)
    model = NeuralModel(path)
    assert model.hidden_layers == hidden_layers
    assert path.read_bytes()[:8] == ('MCNNLM0' + str(hidden_layers)).encode()
    assert path.stat().st_size == model.size_bytes == size
    cdf = model.cdf([256] * context)
    assert cdf[0] == 0 and cdf[-1] == TOTAL
    assert all(b > a for a, b in zip(cdf, cdf[1:]))
    message = bytes(range(256)) + 'Hello mesh café'.encode()
    assert decompress_payload(compress_payload(message, model), len(message), model) == message
    path.write_bytes(path.read_bytes()[:-1] + bytes([path.read_bytes()[-1] ^ 1]))
    with pytest.raises(ValueError, match='checksum'):
        NeuralModel(path)


def test_context_boundaries_and_loss():
    x, y = next(training_batches([b'ab', b'cd'], 2, 20))
    assert x.tolist() == [[256, 256], [256, 97], [256, 256], [256, 99]]
    assert y.tolist() == list(b'abcd')
    class Uniform:
        context = 0
        def cdf(self, history):
            return [i * (TOTAL // 256) for i in range(257)]
    assert evaluate([b'abc'], Uniform())['bits_per_utf8_byte'] == 8
    with pytest.raises(ValueError, match='too small'):
        layout(1, 32, 8, 'float32')


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_training_selects_exported_validation_and_roundtrips(tmp_path, prepared, device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA GPU unavailable')
    if prepared:
        source = tmp_path / 'corpus'
        source.mkdir()
        splits = {}
        for name in ('train', 'validation', 'test'):
            writer = RecordWriter(source / f'{name}.mccorpus')
            for i in range(12):
                writer.write(f'{name} hello {i}'.encode())
            stats = writer.close()
            splits[name] = {**stats, 'files': [{'path': f'{name}.mccorpus', **stats}]}
        (source / 'manifest.json').write_text(json.dumps({
            'format': 'mesh-compression-corpus-v1', 'source': {'test': True},
            'preparation': {'sample_bytes': 160}, 'splits': splits}))
    else:
        source = tmp_path / 'messages.jsonl'
        source.write_text(''.join(json.dumps({'body': f'hello hello mesh {i}'}) + '\n' for i in range(60)))
    output = tmp_path / 'nn'
    args = [str(source), '--device', device, '--model-kib', '4', '--context-bytes', '2', '--embedding-dim', '2',
            '--epochs', '2', '--threads', '1', '--batch-size', '64',
            '--evaluation-messages', '3', '--benchmark-messages', '2', '--output', str(output)]
    assert main(args) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['runtime']['training_device'] == device
    assert report['phase_seconds']['training'] == pytest.approx(sum(row['training_seconds'] for row in report['epochs']))
    for row in report['epochs']:
        assert row['training_seconds'] > 0 and row['validation_seconds'] > 0
        assert row['training_bytes_per_second'] == pytest.approx(row['training_bytes'] / row['training_seconds'])
    if device == 'cuda':
        assert report['runtime']['peak_cuda_allocated_bytes'] > 0
    assert report['model']['model_bytes'] <= 4096
    assert report['model']['context_order'] == 2
    assert report['model']['hidden_layers'] == report['settings']['hidden_layers'] == 3
    hidden = report['model']['hidden_dim']
    assert report['model']['parameter_count'] == 257 * 2 + (2 * 2 + 1) * hidden + 2 * (hidden**2 + hidden) + (hidden + 1) * 256
    assert report['validation']['bits_per_utf8_byte'] == min(r['validation']['bits_per_utf8_byte'] for r in report['epochs'])
    assert report['benchmark']['all_round_trips_exact']
    assert report['benchmark']['messages'] == 2
    model = NeuralModel(output / 'model.mcnn')
    for row in json.loads((output / 'test-vectors.json').read_text()):
        data, payload = bytes.fromhex(row['input_hex']), bytes.fromhex(row['payload_hex'])
        assert decompress_payload(payload, len(data), model) == data
    with pytest.raises(SystemExit):
        main(args)


def test_device_selection_and_cpu_only_diagnostic():
    from types import SimpleNamespace
    from llm_compression.nn_training import resolve_device
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                            version=SimpleNamespace(cuda=None))
    assert resolve_device(torch, 'auto') == 'cpu'
    assert resolve_device(torch, 'cpu') == 'cpu'
    with pytest.raises(ValueError, match='CPU-only PyTorch'):
        resolve_device(torch, 'cuda')
    torch.version.cuda = '12.4'
    with pytest.raises(ValueError, match='NVIDIA driver'):
        resolve_device(torch, 'cuda')
    torch.cuda.is_available = lambda: True
    assert resolve_device(torch, 'auto') == 'cuda'
    assert resolve_device(torch, 'cuda') == 'cuda'
