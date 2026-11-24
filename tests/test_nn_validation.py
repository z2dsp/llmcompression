import json

import numpy as np
import pytest

from llm_compression import nn_training
from llm_compression.nn_model import NeuralModel, TOTAL, export_model, layout, shapes


def require_torch(device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA GPU unavailable')
    return torch


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_target_integer_frequencies_match_codec_including_ties(device):
    torch = require_torch(device)
    rng = np.random.default_rng(12)
    uniform = np.ones(256) / 256
    tied = np.zeros(256)
    tied[:11] = 1 / 11
    certain = np.zeros(256)
    certain[-1] = 1
    random = rng.random(256)
    random /= random.sum()
    for probabilities in (uniform, tied, certain, random):
        scaled = probabilities * (TOTAL - 256)
        expected = np.floor(scaled).astype(np.int64) + 1
        remaining = TOTAL - int(expected.sum())
        indices = np.argsort(-(scaled - np.floor(scaled)), kind='stable')[:remaining]
        expected[indices] += 1
        values = torch.tensor(np.tile(probabilities, (256, 1)), device=device)
        targets = torch.arange(256, device=device)
        actual = nn_training.target_frequencies(torch, values, targets).cpu().numpy()
        np.testing.assert_array_equal(actual, expected)
        assert actual.sum() == TOTAL and actual.min() >= 1


@pytest.mark.parametrize('context', [0, 3, 16])
@pytest.mark.parametrize('storage', ['int8', 'float32'])
@pytest.mark.parametrize('hidden_layers', [1, 2, 3])
def test_cuda_validation_scores_exported_weights_close_to_cpu_reference(tmp_path, context, storage, hidden_layers):
    require_torch('cuda')
    hidden, _ = layout(8, context, 4, storage, hidden_layers)
    rng = np.random.default_rng(10)
    arrays = [rng.normal(0, 0.05, shape).astype('float32') for shape in shapes(context, 4, hidden, hidden_layers)]
    path = tmp_path / 'model.mcnn'
    export_model(path, arrays, context, 4, hidden, storage, hidden_layers)
    model = NeuralModel(path)
    messages = [b'', b'ab', b'c', 'Café 🌊'.encode(), bytes(range(256)), b'another message']
    reference = nn_training.evaluate(messages, model)
    actual = nn_training.evaluate_cuda(iter(messages), model, batch_size=17)
    assert actual['messages'] == reference['messages']
    assert actual['utf8_bytes'] == reference['utf8_bytes']
    assert actual['bits_per_utf8_byte'] == pytest.approx(reference['bits_per_utf8_byte'], abs=1e-5)
    with pytest.raises(ValueError, match='empty split'):
        nn_training.evaluate_cuda([b''], model)


@pytest.mark.parametrize('interval,expected_epochs', [(1, [1, 2, 3, 4, 5]), (2, [2, 4, 5]), (20, [5])])
def test_validation_interval_always_checks_final_epoch(tmp_path, interval, expected_epochs):
    require_torch('cpu')
    source = tmp_path / 'messages.jsonl'
    source.write_text(''.join(json.dumps({'body': f'hello {i}'}) + '\n' for i in range(20)))
    output = tmp_path / 'trained'
    assert nn_training.main([str(source), '--output', str(output), '--device', 'cpu',
                            '--model-kib', '4', '--context-bytes', '2', '--epochs', '5',
                            '--validate-every', str(interval), '--evaluation-messages', '2',
                            '--early-stopping-patience', '0',
                            '--benchmark-messages', '1']) == 0
    report = json.loads((output / 'report.json').read_text())
    actual = [row['epoch'] for row in report['epochs'] if row['validation'] is not None]
    assert actual == expected_epochs
    assert report['best_epoch'] in expected_epochs
    assert report['validation'] == report['selection_validation']
    for row in report['epochs']:
        if row['validation'] is None:
            assert row['export_seconds'] == row['validation_seconds'] == 0
    assert report['benchmark']['all_round_trips_exact']
    assert not (output / 'candidate.mcnn').exists()


@pytest.mark.parametrize('skip_cpu_validation', [False, True])
def test_cuda_checkpoint_selection_has_optional_cpu_reference_scores(tmp_path, monkeypatch, skip_cpu_validation):
    require_torch('cuda')
    source = tmp_path / 'messages.jsonl'
    source.write_text(''.join(json.dumps({'body': f'hello mesh {i}'}) + '\n' for i in range(20)))
    output = tmp_path / 'trained'
    original_evaluate = nn_training.evaluate
    cpu_calls = []
    def record_cpu_evaluation(messages, model):
        messages = list(messages)
        cpu_calls.append(messages)
        return original_evaluate(messages, model)
    monkeypatch.setattr(nn_training, 'evaluate', record_cpu_evaluation)
    args = [str(source), '--output', str(output), '--device', 'cuda',
                            '--model-kib', '4', '--context-bytes', '2', '--epochs', '3',
                            '--validate-every', '2', '--validation-device', 'cuda',
                            '--validation-batch-size', '17', '--evaluation-messages', '2',
                            '--benchmark-messages', '1', '--test-device', 'cpu']
    if skip_cpu_validation:
        args.append('--skip-cpu-validation')
    assert nn_training.main(args) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['runtime']['validation_device'] == 'cuda'
    assert report['best_epoch'] in [2, 3]
    assert report['selection_validation'] == report['epochs'][report['best_epoch'] - 1]['validation']
    model = NeuralModel(output / 'model.mcnn')
    partitions, _ = nn_training.split_corpus(source)
    selected = nn_training.reservoir(partitions['validation'], 2, 42)
    selected_test = nn_training.reservoir(partitions['test'], 2, 43)
    assert report['cpu_validation_skipped'] == skip_cpu_validation
    if skip_cpu_validation:
        assert report['validation'] is None
        assert report['phase_seconds']['reference_validation'] == 0
        assert cpu_calls == [selected_test]
    else:
        assert report['phase_seconds']['reference_validation'] > 0
        assert report['validation'] == original_evaluate(selected, model)
        assert cpu_calls == [selected, selected_test]
    assert report['test'] == original_evaluate(selected_test, model)
    assert report['benchmark']['all_round_trips_exact']


@pytest.mark.parametrize('flag', ['--validate-every', '--validation-batch-size'])
def test_invalid_validation_controls_are_rejected(tmp_path, flag, capsys):
    require_torch('cpu')
    with pytest.raises(SystemExit) as error:
        nn_training.main([str(tmp_path / 'missing.jsonl'), flag, '0'])
    assert error.value.code == 2
    assert 'Validation interval and batch size must be positive' in capsys.readouterr().err


def test_skipping_cpu_validation_requires_cuda_selection(tmp_path, capsys):
    require_torch('cpu')
    with pytest.raises(SystemExit) as error:
        nn_training.main([str(tmp_path / 'missing.jsonl'), '--skip-cpu-validation'])
    assert error.value.code == 2
    assert '--skip-cpu-validation requires --validation-device cuda' in capsys.readouterr().err
