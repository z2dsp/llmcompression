import csv
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import pytest

from llm_compression import nn_sweep
from llm_compression.nn_training import build_parser


@contextmanager
def expect_error(capsys, message):
    with pytest.raises(SystemExit) as error:
        yield
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_sweep_forwards_trainer_options_and_requires_cuda(tmp_path):
    corpus = tmp_path / 'messages with spaces.jsonl'
    args = build_parser(sweep=True).parse_args([
        str(corpus), '--storage', 'float32', '--embedding-dim', '3', '--epochs', '2',
        '--keep-duplicates', '--train-bytes', '1024', '--split', 'random', '--seed', '99',
        '--max-message-bytes', '0', '--batch-size', '32', '--threads', '2',
        '--learning-rate', '0.002', '--evaluation-messages', '3', '--benchmark-messages', '1',
        '--validate-every', '3', '--validation-device', 'cuda', '--validation-batch-size', '17',
        '--skip-cpu-validation',
        '--early-stopping-patience', '2', '--early-stopping-min-delta', '0.001',
    ])
    args.sizes_kib, args.contexts, args.resume, args.plot_only = [8], [16], False, False
    directory = tmp_path / 'result with spaces'
    command = nn_sweep.run_command(args, 8, 16, directory)
    assert command[:2] == [sys.executable, '-u']
    assert Path(command[2]).name == 'train_nn_cuda.py'
    parsed = build_parser().parse_args(command[3:])
    assert parsed.device == 'cuda'
    assert parsed.model_kib == 8 and parsed.context_bytes == 16
    assert parsed.input == corpus.resolve() and parsed.output == directory.resolve()
    for key, value in vars(args).items():
        if key not in nn_sweep.SWEEP_OPTIONS:
            assert getattr(parsed, key) == value


def test_no_feasible_configurations_are_reported_without_launching_training(tmp_path, monkeypatch, capsys):
    pytest.importorskip('matplotlib')
    corpus = tmp_path / 'messages.jsonl'
    corpus.write_text('{}\n')
    monkeypatch.setattr(nn_sweep.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Should skip training'))
    output = tmp_path / 'empty-sweep'
    with expect_error(capsys, 'No feasible configurations'):
        nn_sweep.main([str(corpus), '--output', str(output), '--sizes-kib', '1', '--contexts', '0,16'])
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['results'] == []
    assert len(summary['skipped']) == 2


def test_child_failure_leaves_resumable_configuration(tmp_path, monkeypatch, capsys):
    pytest.importorskip('matplotlib')
    corpus = tmp_path / 'messages.jsonl'
    corpus.write_text('{}\n')
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(nn_sweep.subprocess, 'run', fail)
    output = tmp_path / 'failed-sweep'
    args = [str(corpus), '--output', str(output), '--sizes-kib', '4', '--contexts', '4']
    with expect_error(capsys, 'CUDA training exited with status 2'):
        nn_sweep.main(args)
    assert len(calls) == 1
    assert (output / 'sweep.json').is_file()
    assert json.loads((output / 'summary.json').read_text())['results'] == []
    with expect_error(capsys, 'CUDA training exited with status 2'):
        nn_sweep.main(args + ['--resume'])
    assert len(calls) == 2


def test_cuda_grid_resume_integrity_interruption_and_plots(tmp_path, monkeypatch, capsys):
    pytest.importorskip('matplotlib')
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA GPU unavailable')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '1')
    corpus = tmp_path / 'messages.jsonl'
    corpus.write_text(''.join(json.dumps({'body': f'hello mesh {i}'}) + '\n' for i in range(20)))
    output = tmp_path / 'sweep'
    args = [str(corpus), '--output', str(output), '--sizes-kib', '1,2,4', '--contexts', '0,4',
            '--embedding-dim', '2', '--epochs', '1', '--batch-size', '64', '--threads', '1',
            '--early-stopping-patience', '0',
            '--evaluation-messages', '2', '--benchmark-messages', '1']
    assert nn_sweep.main(args) == 0
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary['results']) == 4 and len(summary['skipped']) == 2
    for row in summary['results']:
        assert row['model_bytes'] <= row['requested_model_kib'] * 1024
        assert row['peak_cuda_allocated_bytes'] > 0
        assert row['benchmark_messages'] == 1
    with (output / 'summary.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 4
    assert (output / 'compression.png').read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (output / 'compression.svg').read_text()

    model = output / '2k-context0' / 'model.mcnn'
    mtime = model.stat().st_mtime_ns
    real_subprocess_run = nn_sweep.subprocess.run
    monkeypatch.setattr(nn_sweep.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Should reuse completed runs'))
    assert nn_sweep.main(args + ['--resume']) == 0
    assert model.stat().st_mtime_ns == mtime
    # Existing default sweeps predate these flags and remain resumable.
    manifest = json.loads((output / 'sweep.json').read_text())
    legacy_defaults = {**nn_sweep.VALIDATION_DEFAULTS, **nn_sweep.LEGACY_EARLY_STOPPING_DEFAULTS}
    for key in legacy_defaults:
        manifest.pop(key)
    (output / 'sweep.json').write_text(json.dumps(manifest))
    for row in summary['results']:
        path = Path(row['report'])
        old = json.loads(path.read_text())
        for key in legacy_defaults:
            old['settings'].pop(key)
        old.pop('selection_validation')
        old.pop('early_stopping')
        old.pop('cpu_validation_skipped')
        old['runtime'].pop('validation_device')
        path.write_text(json.dumps(old))
    assert nn_sweep.main(args + ['--resume']) == 0
    assert model.stat().st_mtime_ns == mtime
    with expect_error(capsys, 'Resume configuration/corpus differs'):
        nn_sweep.main(args + ['--resume', '--early-stopping-patience', '3'])
    with expect_error(capsys, 'Output is not empty'):
        nn_sweep.main(args)
    with expect_error(capsys, 'Resume configuration/corpus differs'):
        nn_sweep.main(args + ['--resume', '--seed', '99'])
    corpus.write_text(corpus.read_text() + '\n')
    with expect_error(capsys, 'Resume configuration/corpus differs'):
        nn_sweep.main(args + ['--resume'])
    # Plot-only neither opens the corpus nor starts CUDA training.
    assert nn_sweep.main([str(tmp_path / 'missing.jsonl'), '--output', str(output), '--plot-only']) == 0
    corpus.write_text(corpus.read_text()[:-1])

    report_path = output / '2k-context0' / 'report.json'
    original_report = report_path.read_bytes()
    changed = json.loads(original_report)
    changed['settings']['epochs'] = 99
    report_path.write_text(json.dumps(changed))
    with expect_error(capsys, 'Completed run does not match configuration'):
        nn_sweep.main(args + ['--resume'])
    report_path.write_bytes(original_report)

    original_model = model.read_bytes()
    damaged = bytearray(original_model)
    damaged[-1] ^= 1
    model.write_bytes(damaged)
    with expect_error(capsys, 'checksum'):
        nn_sweep.main(args + ['--resume'])
    model.write_bytes(original_model)

    incomplete = output / '4k-context4'
    (incomplete / 'report.json').unlink()
    monkeypatch.setattr(nn_sweep.subprocess, 'run', real_subprocess_run)
    assert nn_sweep.main(args + ['--resume']) == 0
    preserved = list(output.glob('4k-context4.incomplete-*'))
    assert len(preserved) == 1 and (preserved[0] / 'model.mcnn').is_file()
    assert model.read_bytes() == original_model


def test_cuda_sweep_without_cpu_validation_plots_and_resumes(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA GPU unavailable')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '1')
    source = tmp_path / 'messages.jsonl'
    source.write_text(''.join(json.dumps({'body': f'hello mesh {i}'}) + '\n' for i in range(20)))
    output = tmp_path / 'sweep'
    args = [str(source), '--output', str(output), '--sizes-kib', '4', '--contexts', '2',
            '--epochs', '1', '--threads', '1', '--evaluation-messages', '2', '--benchmark-messages', '1',
            '--validation-device', 'cuda', '--skip-cpu-validation']
    assert nn_sweep.main(args) == 0
    row = json.loads((output / 'summary.json').read_text())['results'][0]
    report = json.loads(Path(row['report']).read_text())
    assert report['validation'] is None and report['cpu_validation_skipped']
    assert row['validation_score_device'] == 'cuda'
    assert row['validation_bits_per_utf8_byte'] == report['selection_validation']['bits_per_utf8_byte']
    assert 'CUDA validation estimate' in (output / 'compression.svg').read_text()
    assert report['benchmark']['all_round_trips_exact']
    monkeypatch.setattr(nn_sweep.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Should reuse completed run'))
    assert nn_sweep.main(args + ['--resume']) == 0
