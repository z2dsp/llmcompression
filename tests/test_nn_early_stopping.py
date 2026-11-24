import hashlib
import json

import pytest

from llm_compression import nn_training


@pytest.mark.parametrize('losses,patience,delta,expected_stop', [
    ([5, 4, 3, 2], 2, 0, None),
    ([3, 3.1, 3.2], 2, 0, 3),
    ([3, 3, 3], 2, 0, 3),
    ([4, 4.1, 3.8, 4, 4.2], 2, 0, 5),
    ([5, 4.96, 4.92, 4.88, 4.87, 4.86, 4.85], 3, 0.1, 7),
    ([3] * 20, 0, 0, None),
])
def test_patience_plateaus_recovery_and_cumulative_improvement(losses, patience, delta, expected_stop):
    stopping = nn_training.EarlyStopping(patience, delta)
    stopped_at = None
    for index, loss in enumerate(losses, 1):
        if stopping.update(loss):
            stopped_at = index
            break
    assert stopped_at == expected_stop
    assert stopping.checks == (expected_stop or len(losses))


@pytest.mark.parametrize('patience,delta', [(-1, 0), (1.5, 0), (3, -0.1), (3, float('nan')), (3, float('inf'))])
def test_invalid_stopping_settings(patience, delta):
    with pytest.raises(ValueError, match='Early stopping'):
        nn_training.EarlyStopping(patience, delta)


@pytest.mark.parametrize('loss', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_validation_is_never_saved_as_a_checkpoint(loss):
    stopping = nn_training.EarlyStopping()
    with pytest.raises(ValueError, match='Nonfinite validation'):
        stopping.update(loss)
    assert stopping.best_loss is None


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('interval', [1, 2])
def test_training_stops_after_bad_validation_checks_and_keeps_best_model(tmp_path, monkeypatch, device, interval):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA GPU unavailable')
    corpus = tmp_path / 'messages.jsonl'
    corpus.write_text(''.join(json.dumps({'body': f'hello {i}'}) + '\n' for i in range(20)))
    output = tmp_path / 'trained'
    original_evaluate = nn_training.evaluate
    losses, snapshots = [4.0, 3.0, 3.2, 3.4], []
    def controlled_validation(messages, model, *args, **kwargs):
        result = original_evaluate(messages, model)
        if len(snapshots) < len(losses):
            loss = losses[len(snapshots)]
            snapshots.append((output / 'candidate.mcnn').read_bytes())
            result.update(bits_per_utf8_byte=loss, predicted_bits=loss * result['utf8_bytes'])
        return result
    monkeypatch.setattr(nn_training, 'evaluate_cuda' if device == 'cuda' else 'evaluate', controlled_validation)
    assert nn_training.main([
        str(corpus), '--output', str(output), '--device', device, '--validation-device', device,
        '--model-kib', '4', '--context-bytes', '2', '--epochs', '20', '--threads', '1',
        '--early-stopping-patience', '2', '--validate-every', str(interval),
        '--evaluation-messages', '2', '--benchmark-messages', '1',
    ]) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['early_stopping']['stopped_early']
    assert report['early_stopping']['completed_epochs'] == 4 * interval
    assert report['early_stopping']['validation_checks'] == 4
    assert report['early_stopping']['reason'] == 'validation_no_improvement'
    assert report['best_epoch'] == 2 * interval
    assert report['selection_validation']['bits_per_utf8_byte'] == 3.0
    assert (output / 'model.mcnn').read_bytes() == snapshots[1]
    assert report['model']['sha256'] == hashlib.sha256(snapshots[1]).hexdigest()
    assert report['benchmark']['all_round_trips_exact']
    assert report['epochs'][-1]['early_stopping_bad_checks'] == 2
    assert not (output / 'candidate.mcnn').exists()


@pytest.mark.parametrize('patience,delta,losses,expected_epochs,expected_best', [
    (0, 0, [4, 4, 4, 4, 4], 5, 1),
    (2, 0.2, [4, 3.95, 3.9], 3, 3),
])
def test_disable_or_require_significant_progress_without_discarding_best_checkpoint(
        tmp_path, monkeypatch, patience, delta, losses, expected_epochs, expected_best):
    pytest.importorskip('torch')
    corpus = tmp_path / 'messages.jsonl'
    corpus.write_text(''.join(json.dumps({'body': f'hello {i}'}) + '\n' for i in range(20)))
    output = tmp_path / 'trained'
    original_evaluate = nn_training.evaluate
    snapshots = []
    def controlled_validation(messages, model):
        result = original_evaluate(messages, model)
        if len(snapshots) < len(losses):
            loss = losses[len(snapshots)]
            snapshots.append((output / 'candidate.mcnn').read_bytes())
            result.update(bits_per_utf8_byte=loss, predicted_bits=loss * result['utf8_bytes'])
        return result
    monkeypatch.setattr(nn_training, 'evaluate', controlled_validation)
    assert nn_training.main([
        str(corpus), '--output', str(output), '--device', 'cpu', '--epochs', '5',
        '--model-kib', '4', '--context-bytes', '2', '--threads', '1',
        '--early-stopping-patience', str(patience), '--early-stopping-min-delta', str(delta),
        '--evaluation-messages', '2', '--benchmark-messages', '1',
    ]) == 0
    report = json.loads((output / 'report.json').read_text())
    assert len(report['epochs']) == expected_epochs
    assert report['best_epoch'] == expected_best
    assert (output / 'model.mcnn').read_bytes() == snapshots[expected_best - 1]
    assert report['early_stopping']['stopped_early'] == bool(patience)


def test_cli_rejects_invalid_patience_before_reading_input(tmp_path, capsys):
    pytest.importorskip('torch')
    with pytest.raises(SystemExit) as error:
        nn_training.main([str(tmp_path / 'missing.jsonl'), '--early-stopping-patience', '-1'])
    assert error.value.code == 2
    assert 'Early stopping patience must be nonnegative' in capsys.readouterr().err
