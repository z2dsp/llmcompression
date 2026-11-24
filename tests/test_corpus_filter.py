import csv
import json
from pathlib import Path
import pytest
from llm_compression.corpus_filter import filter_channels


@pytest.mark.parametrize('suffix', ['.jsonl', '.csv'])
def test_channel_filter_preserves_remaining_records_and_backup(tmp_path, suffix):
    path = tmp_path / ('messages' + suffix)
    rows = [{'id': i, 'channel': ch, 'body': 'Café\n  exact \x00text', 'type': 'CHAN' if ch else 'PRIV'}
            for i, ch in enumerate(['#test', '#alert', '#bot', '#nrl', '#TEST', None, '#general', '#testing'])]
    if suffix == '.csv':
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    else:
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),encoding='utf-8')
    original = path.read_bytes()
    report = filter_channels(path)
    assert report['removed_messages'] == 5
    assert report['retained_messages'] == 3
    assert Path(report['backup']).read_bytes() == original
    with path.open(encoding='utf-8', newline='') as f:
        remaining = list(csv.DictReader(f)) if suffix == '.csv' else [json.loads(line) for line in f]
    assert [int(row['id']) for row in remaining] == [5, 6, 7]
    assert all(row['body'] == rows[0]['body'] for row in remaining)
    after = path.read_bytes()
    assert filter_channels(path)['removed_messages'] == 0
    assert path.read_bytes() == after


def test_filter_rejects_missing_metadata_without_modifying_input(tmp_path):
    path = tmp_path / 'messages.jsonl'
    original = b'{"text":"hello"}\n'
    path.write_bytes(original)
    with pytest.raises(ValueError, match='channel'):
        filter_channels(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.bak'))
