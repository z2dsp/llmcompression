"""Boundary, numerical accuracy, and precision checks for the packed GPU path."""
import json

import numpy as np
import pytest

from llm_compression import nn_training
from llm_compression.nn_cuda import Float32Sum, device_batches


def get_torch(device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA GPU unavailable')
    return torch


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('context', [0, 1, 3, 17, 129])
@pytest.mark.parametrize('chunk_bytes', [1, 16, 39, 1048576])
def test_device_batches_match_every_reference_context_and_boundary(device, context, chunk_bytes):
    torch = get_torch(device)
    messages = [b'', b'a', b'bc', bytes(range(256)), 'Café 🌊'.encode(), b'd', b'efgh']
    reference = list(nn_training.training_batches(messages, context, 13))
    actual = list(device_batches(torch, iter(messages), context, 13,
                                 chunk_bytes=chunk_bytes, device=device))
    assert len(actual) == len(reference)
    for (x, y), (expected_x, expected_y) in zip(actual, reference):
        assert x.dtype == y.dtype == torch.int64
        np.testing.assert_array_equal(x.cpu().numpy(), expected_x)
        np.testing.assert_array_equal(y.cpu().numpy(), expected_y)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_shuffling_is_seeded_and_preserves_context_target_pairs(device):
    torch = get_torch(device)
    def run(seed):
        return list(device_batches(torch, [bytes(range(60))], 3, 13,
                    chunk_bytes=27, device=device,
                    generator=torch.Generator(device=device).manual_seed(seed)))
    a, b, different = run(12), run(12), run(13)
    for (x, y), (again_x, again_y) in zip(a, b):
        assert torch.equal(x, again_x) and torch.equal(y, again_y)
        for history, target in zip(x.cpu().tolist(), y.cpu().tolist()):
            assert history == ([256] * max(0, 3 - target) + list(range(max(0, target - 3), target)))
    assert not torch.equal(a[0][1], different[0][1])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fp32_accumulation_keeps_small_increments_on_long_runs(device):
    torch = get_torch(device)
    accumulator = Float32Sum(torch, device, block_size=16)
    accumulator.add(torch.tensor(1e8, device=device))
    accumulator.item()
    increment = torch.tensor(1.0, device=device)
    for i in range(4096):
        accumulator.add(increment)
        if i == 2000:
            accumulator.item()  # Progress reads must not double count.
    assert accumulator.item() == pytest.approx(1e8 + 4096, abs=8)
    assert accumulator.item() == pytest.approx(1e8 + 4096, abs=8)
    assert accumulator.buffer.dtype == accumulator.total.dtype == torch.float32
    assert accumulator.correction.dtype == torch.float32


def test_cuda_launcher_defaults_and_explicit_cpu_scoring_controls():
    parser = nn_training.build_parser(cuda_defaults=True)
    args = parser.parse_args(['messages.jsonl'])
    assert args.validation_device == 'cuda' and args.skip_cpu_validation
    assert args.test_device == 'auto'
    args = parser.parse_args(['messages.jsonl', '--validation-device', 'cpu',
                              '--no-skip-cpu-validation', '--test-device', 'cpu'])
    assert args.validation_device == 'cpu' and not args.skip_cpu_validation
    assert args.test_device == 'cpu'


def test_default_cuda_training_and_scoring_never_create_fp64_gpu_tensors(tmp_path, monkeypatch):
    torch = get_torch('cuda')
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class ForbidDoubleCUDA(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            leaves, _ = tree_flatten((args, kwargs, result))
            assert not any(isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float64
                           for x in leaves), f'FP64 CUDA operation: {func}'
            return result

    source = tmp_path / 'messages.jsonl'
    source.write_text(''.join(json.dumps({'body': f'hello mesh {i}'}) + '\n' for i in range(30)))
    output = tmp_path / 'trained'
    monkeypatch.setattr(nn_training, 'evaluate', lambda *a, **k: pytest.fail('Unexpected serial CPU scoring'))
    with ForbidDoubleCUDA():
        assert nn_training.main([str(source), '--device', 'cuda', '--output', str(output),
            '--model-kib', '4', '--context-bytes', '3', '--batch-size', '32', '--epochs', '2',
            '--evaluation-messages', '2', '--benchmark-messages', '1', '--cuda-chunk-bytes', '70'],
            cuda_defaults=True) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['cpu_validation_skipped'] and report['validation'] is None
    assert report['runtime']['test_device'] == 'cuda'
    assert report['runtime']['training_precision'] == 'float32'
    assert report['runtime']['validation_precision'] == report['runtime']['test_precision'] == 'float32'
    assert report['model']['model_bytes'] <= 4 * 1024
    assert report['benchmark']['all_round_trips_exact']
