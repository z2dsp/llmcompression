import json
from pathlib import Path
import pytest

from llm_compression.bit_sweep import main

pytest.importorskip('matplotlib')


def test_sweep_round_trips_resume_and_plot(tmp_path):
    corpus = tmp_path / 'messages.jsonl'
    corpus.write_text(''.join(json.dumps({'body': f'Radio check message {i}: hello mesh {i % 7}!'}) + '\n'
                              for i in range(60)), encoding='utf-8')
    output = tmp_path / 'sweep'
    args = [str(corpus), '--output', str(output), '--sizes-kib', '3,8', '--orders', '0,1,4',
            '--strengths', '16', '--workers', '1', '--evaluation-messages', '5', '--benchmark-messages', '3']
    assert main(args) == 0
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary['results']) == 5
    assert len(summary['skipped']) == 1
    for row in summary['results']:
        report = json.loads(Path(row['report']).read_text())
        assert report['benchmark']['all_round_trips_exact']
        assert row['model_bytes'] <= row['requested_model_kib'] * 1024
        assert row['test_messages'] == 5
    assert (output / 'compression.png').read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (output / 'compression.svg').read_text()
    assert len((output / 'summary.csv').read_text().splitlines()) == 6
    model = output / '3k-order0' / 'model.mcblm'
    original_mtime = model.stat().st_mtime_ns
    assert main(args + ['--resume']) == 0
    assert model.stat().st_mtime_ns == original_mtime
    assert main(['--output', str(output), '--plot-only']) == 0
    with pytest.raises(SystemExit):
        main(args + ['--resume', '--seed', '99'])
    with pytest.raises(SystemExit):
        main(args)
    # An interrupted run is preserved and rebuilt, without touching completed runs.
    incomplete = output / '8k-order4'
    (incomplete / 'report.json').unlink()
    assert main(args + ['--resume']) == 0
    assert len(list(output.glob('8k-order4.incomplete-*'))) == 1
    assert model.stat().st_mtime_ns == original_mtime
