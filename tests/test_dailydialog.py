import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from llm_compression import dailydialog as dd
from llm_compression.corpus import read_messages


def make_archive(path, dialogues=None):
    if dialogues is None:
        dialogues = [
            {'dataset': 'dailydialog', 'data_split': split, 'dialogue_id': f'dialogue-{i}',
             'turns': [{'utt_idx': 0, 'utterance': '  Café\nhello!  '},
                       {'utt_idx': 1, 'utterance': ''}]}
            for i, split in enumerate(dd.SPLITS)
        ]
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('data/dialogues.json', json.dumps(dialogues))
        archive.writestr('data/ontology.json', '{}')
    return path


def test_all_turns_preserved_including_repeated_text_and_empty_strings(tmp_path):
    rows = list(dd.iter_utterances(make_archive(tmp_path / 'data.zip')))
    assert len(rows) == 6
    assert len({row['source_id'] for row in rows}) == 6
    assert [row['source_split'] for row in rows] == [s for s in dd.SPLITS for _ in range(2)]
    assert [row['body'] for row in rows] == ['  Café\nhello!  ', ''] * 3


@pytest.mark.parametrize('ending', [b'', b'\r\n', b'\n\n'])
def test_atomic_append_preserves_original_and_rerun_is_unchanged(tmp_path, ending):
    corpus = tmp_path / 'messages.jsonl'
    original = b'{ "body": "original", "private": true }' + ending
    corpus.write_bytes(original)
    corpus.chmod(0o640)
    rows = list(dd.iter_utterances(make_archive(tmp_path / 'data.zip')))
    report = dd.append_utterances(corpus, rows)
    assert report['added_utterances'] == 6
    assert report['source_utterances_by_split'] == dict.fromkeys(dd.SPLITS, 2)
    assert report['original_records'] == 1
    assert report['total_records'] == 7
    assert Path(report['backup']).read_bytes() == original
    assert corpus.read_bytes().startswith(original)
    assert corpus.stat().st_mode & 0o777 == 0o640
    assert [json.loads(line)['body'] for line in corpus.read_text().splitlines() if line] == (
        ['original'] + [row['body'] for row in rows])
    assert list(read_messages(corpus)) == ['original'] + [row['body'] for row in rows]
    after, mtime = corpus.read_bytes(), corpus.stat().st_mtime_ns
    repeated = dd.append_utterances(corpus, rows)
    assert repeated['added_utterances'] == 0
    assert repeated['already_present'] == 6
    assert repeated['backup'] is None
    assert corpus.read_bytes() == after
    assert corpus.stat().st_mtime_ns == mtime
    assert len(list(tmp_path.glob('*.bak'))) == 1


def test_invalid_incoming_data_does_not_replace_existing_corpus(tmp_path):
    corpus = tmp_path / 'messages.jsonl'
    original = b'{"body":"keep"}\n'
    corpus.write_bytes(original)
    rows = list(dd.iter_utterances(make_archive(tmp_path / 'data.zip')))
    with pytest.raises(ValueError, match='Duplicate input'):
        dd.append_utterances(corpus, rows + rows[:1])
    assert corpus.read_bytes() == original
    assert not list(tmp_path.glob('*.bak'))
    assert not list(tmp_path.glob('*.tmp'))


def test_conflicting_existing_source_id_is_rejected(tmp_path):
    corpus = tmp_path / 'messages.jsonl'
    rows = list(dd.iter_utterances(make_archive(tmp_path / 'data.zip')))
    dd.append_utterances(corpus, rows)
    original = corpus.read_bytes()
    with pytest.raises(ValueError, match='body differs'):
        dd.append_utterances(corpus, [dict(rows[0], body='changed')])
    assert corpus.read_bytes() == original
    assert not list(tmp_path.glob('*.bak'))


def test_malformed_existing_corpus_is_not_overwritten(tmp_path):
    corpus = tmp_path / 'messages.jsonl'
    corpus.write_bytes(b'not json\n')
    with pytest.raises(ValueError):
        dd.append_utterances(corpus, [])
    assert corpus.read_bytes() == b'not json\n'


def test_malformed_later_dialogue_does_not_commit_partial_import(tmp_path):
    archive = make_archive(tmp_path / 'data.zip')
    with zipfile.ZipFile(archive) as source:
        dialogues = json.loads(source.read('data/dialogues.json'))
    dialogues[-1]['turns'][0]['utterance'] = None
    make_archive(archive, dialogues)
    corpus = tmp_path / 'messages.jsonl'
    with pytest.raises(ValueError, match='string utterance'):
        dd.append_utterances(corpus, dd.iter_utterances(archive))
    assert not corpus.exists()
    assert not list(tmp_path.glob('*.tmp'))


def test_download_verifies_archive_and_reuses_cache_offline(tmp_path, monkeypatch):
    content = make_archive(tmp_path / 'fixture.zip').read_bytes()
    monkeypatch.setattr(dd, 'ARCHIVE_BYTES', len(content))
    monkeypatch.setattr(dd, 'ARCHIVE_SHA256', hashlib.sha256(content).hexdigest())
    requests = []

    def download(request, timeout):
        requests.append(request.full_url)
        return io.BytesIO(content)

    monkeypatch.setattr(dd, 'urlopen', download)
    cache = tmp_path / 'cache'
    archive = dd.download_archive(cache)
    assert archive.read_bytes() == content
    assert requests == [dd.ARCHIVE_URL]
    assert dd.download_archive(cache, offline=True) == archive
    assert len(requests) == 1
    assert json.loads((cache / 'source.json').read_text())['revision'] == dd.REVISION
    archive.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='fails verification'):
        dd.download_archive(cache, offline=True)


def test_bad_download_is_cleaned_up(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, 'urlopen', lambda *a, **k: io.BytesIO(b'bad archive'))
    with pytest.raises(ValueError, match='mismatch'):
        dd.download_archive(tmp_path)
    assert not (tmp_path / 'raw' / 'data.zip').exists()
    assert not list((tmp_path / 'raw').iterdir())
    with pytest.raises(FileNotFoundError, match='not cached'):
        dd.download_archive(tmp_path, offline=True)


def test_cli_reports_and_rejects_output_collisions(tmp_path, monkeypatch, capsys):
    archive = make_archive(tmp_path / 'fixture.zip')
    monkeypatch.setattr(dd, 'download_archive', lambda *a, **kw: archive)
    corpus, cache = tmp_path / 'messages.jsonl', tmp_path / 'cache'
    assert dd.main(['--corpus', str(corpus), '--cache-dir', str(cache)]) == 0
    report = json.loads((cache / 'append-report.json').read_text())
    assert report['added_utterances'] == 6
    assert json.loads(capsys.readouterr().out) == report
    with pytest.raises(SystemExit) as exc:
        dd.main(['--corpus', str(corpus), '--report', str(corpus)])
    assert exc.value.code == 2
