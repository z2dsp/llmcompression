"""Download DailyDialog and append every utterance to a training JSONL corpus."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
import uuid
import zipfile

REPOSITORY = 'ConvLab/dailydialog'
REVISION = '745c1796cfe209b469394567f496815d2bc495d2'
SOURCE_URL = 'https://huggingface.co/datasets/' + REPOSITORY
ARCHIVE_URL = f'{SOURCE_URL}/resolve/{REVISION}/data.zip?download=true'
ARCHIVE_SHA256 = 'b1f38f3bf5431f9384107011a943f1a3446027721292e3a248a894b7766fa9d3'
ARCHIVE_BYTES = 3738236
SPLITS = ('train', 'validation', 'test')
LICENSE = 'CC-BY-NC-SA-4.0'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(report, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download_archive(cache_dir, *, offline=False):
    """Use a pinned, checksum-verified archive; execute no remote dataset code."""
    cache_dir = Path(cache_dir)
    destination = cache_dir / 'raw' / 'data.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        if offline:
            raise FileNotFoundError(f'Archive not cached: {destination}; run once without --offline')
        request = Request(ARCHIVE_URL, headers={'User-Agent': 'llm-compression-demo/0.1 (DailyDialog)'})
        temporary = None
        try:
            with urlopen(request, timeout=90) as source, tempfile.NamedTemporaryFile(
                    dir=destination.parent, prefix='.data-', suffix='.zip.part', delete=False) as target:
                temporary = Path(target.name)
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    target.write(block)
                    if target.tell() > ARCHIVE_BYTES:
                        raise ValueError('DailyDialog download exceeds the expected size')
            if temporary.stat().st_size != ARCHIVE_BYTES or sha256_file(temporary) != ARCHIVE_SHA256:
                raise ValueError('DailyDialog download size or SHA-256 mismatch')
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if destination.stat().st_size != ARCHIVE_BYTES or sha256_file(destination) != ARCHIVE_SHA256:
        raise ValueError(f'Cached DailyDialog archive fails verification: {destination}')
    write_report(cache_dir / 'source.json', {
        'repository': REPOSITORY, 'revision': REVISION, 'url': SOURCE_URL,
        'archive_url': ARCHIVE_URL, 'archive_bytes': ARCHIVE_BYTES, 'archive_sha256': ARCHIVE_SHA256,
        'license': LICENSE, 'license_url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
        'dataset_card': f'{SOURCE_URL}/blob/{REVISION}/README.md',
        'paper': 'https://arxiv.org/abs/1710.03957',
    })
    return destination


def iter_utterances(archive_path):
    """Preserve every turn occurrence and its exact text, across all splits."""
    with zipfile.ZipFile(archive_path) as archive:
        # Select only dialogue data; ontology.json is not a dialogue corpus.
        with archive.open('data/dialogues.json') as stream:
            dialogues = json.load(stream)
    if not isinstance(dialogues, list):
        raise ValueError('Expected a list of DailyDialog dialogues')
    seen = set()
    for dialogue in dialogues:
        if not isinstance(dialogue, dict) or dialogue.get('dataset') != 'dailydialog':
            raise ValueError('Invalid DailyDialog dialogue record')
        split, dialogue_id = dialogue.get('data_split'), dialogue.get('dialogue_id')
        if split not in SPLITS or not isinstance(dialogue_id, str) or not dialogue_id:
            raise ValueError('Invalid DailyDialog split or dialogue ID')
        turns = dialogue.get('turns')
        if not isinstance(turns, list):
            raise ValueError(f'Dialogue has no turns list: {dialogue_id}')
        for turn in turns:
            if not isinstance(turn, dict) or not isinstance(turn.get('utterance'), str):
                raise ValueError(f'Turn has no string utterance: {dialogue_id}')
            index = turn.get('utt_idx')
            if type(index) is not int or index < 0:
                raise ValueError(f'Invalid utterance index: {dialogue_id}')
            source_id = f'{dialogue_id}:{index}'
            if source_id in seen:
                raise ValueError(f'Duplicate source utterance ID: {source_id}')
            seen.add(source_id)
            yield {'body': turn['utterance'], 'source': REPOSITORY, 'source_id': source_id,
                   'source_split': split, 'dialogue_id': dialogue_id, 'utterance_index': index}


def append_utterances(corpus_path, records):
    """Stage the whole update, back up the original, and replace it atomically.

    Text duplicates with different source IDs are retained. Only turns already
    imported from this dataset are skipped, and conflicting content is rejected.
    Existing corpus bytes are preserved, including line endings and whitespace.
    """
    corpus = Path(corpus_path).resolve()
    corpus.parent.mkdir(parents=True, exist_ok=True)
    existed = corpus.exists()
    original_digest = hashlib.sha256()
    existing, source_ids = {}, set()
    original_rows = added = skipped = body_bytes = 0
    source_counts, added_counts = Counter(), Counter()
    backup = temporary = None
    last_byte = b''
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=corpus.parent, prefix='.dailydialog-',
                                         suffix='.jsonl.tmp', delete=False) as target:
            temporary = Path(target.name)
            if existed:
                with corpus.open('rb') as source:
                    for line_number, line in enumerate(source, 1):
                        original_digest.update(line)
                        target.write(line)
                        last_byte = line[-1:]
                        if not line.strip():
                            continue
                        row = json.loads(line.decode('utf-8'))
                        if not isinstance(row, dict):
                            raise ValueError(f'Corpus line {line_number} is not a JSON object')
                        original_rows += 1
                        if row.get('source') == REPOSITORY:
                            key, body = row.get('source_id'), row.get('body')
                            if not isinstance(key, str) or not isinstance(body, str):
                                raise ValueError(f'DailyDialog metadata missing on corpus line {line_number}')
                            if key in existing and existing[key] != body:
                                raise ValueError(f'Conflicting existing DailyDialog source ID: {key}')
                            existing[key] = body
            for record in records:
                if (not isinstance(record, dict) or record.get('source') != REPOSITORY
                        or not isinstance(record.get('source_id'), str)
                        or not isinstance(record.get('body'), str) or record.get('source_split') not in SPLITS):
                    raise ValueError('Invalid utterance record')
                key, body, split = record['source_id'], record['body'], record['source_split']
                if key in source_ids:
                    raise ValueError(f'Duplicate input source ID: {key}')
                source_ids.add(key)
                source_counts[split] += 1
                if key in existing:
                    if existing[key] != body:
                        raise ValueError(f'Existing DailyDialog body differs for source ID: {key}')
                    skipped += 1
                    continue
                if not added and last_byte and last_byte != b'\n':
                    target.write(b'\n')
                target.write((json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8'))
                added += 1
                added_counts[split] += 1
                body_bytes += len(body.encode('utf-8'))
            target.flush()
            os.fsync(target.fileno())
        before = original_digest.hexdigest() if existed else None
        if (existed and (not corpus.exists() or sha256_file(corpus) != before)) or (not existed and corpus.exists()):
            raise ValueError('Corpus changed during processing; original was not replaced')
        if added:
            if existed:
                backup = corpus.with_name(corpus.name + '.before-dailydialog-' + uuid.uuid4().hex[:12] + '.bak')
                # Exclusive creation protects an existing file even on an ID collision.
                with corpus.open('rb') as source, backup.open('xb') as output:
                    shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                shutil.copystat(corpus, backup)
                if sha256_file(backup) != before or sha256_file(corpus) != before:
                    raise ValueError('Corpus changed during backup; original was not replaced')
                # Preserve permissions and extended ACLs while retaining the
                # modification time of the newly written corpus.
                written = temporary.stat()
                shutil.copystat(corpus, temporary)
                os.utime(temporary, ns=(written.st_atime_ns, written.st_mtime_ns))
            temporary.replace(corpus)
        return {'corpus': str(corpus), 'repository': REPOSITORY, 'revision': REVISION,
                'license': LICENSE, 'source_utterances': sum(source_counts.values()),
                'source_utterances_by_split': dict(source_counts), 'added_utterances': added,
                'added_by_split': dict(added_counts), 'already_present': skipped,
                'added_body_utf8_bytes': body_bytes, 'original_records': original_rows,
                'total_records': original_rows + added, 'before_sha256': before,
                'after_sha256': sha256_file(corpus) if corpus.exists() else None,
                'backup': str(backup) if backup is not None else None}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=Path('data/combined-messages.jsonl'))
    parser.add_argument('--cache-dir', type=Path, default=Path('data/dailydialog'))
    parser.add_argument('--offline', action='store_true', help='use the verified cached archive without downloading')
    parser.add_argument('--report', type=Path, help='default: CACHE_DIR/append-report.json')
    args = parser.parse_args(argv)
    report_path = args.report or args.cache_dir / 'append-report.json'
    paths = [args.corpus, report_path, args.cache_dir / 'source.json',
             args.cache_dir / 'raw' / 'data.zip']
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error('Corpus, report, source metadata, and archive paths must differ')
    try:
        archive = download_archive(args.cache_dir, offline=args.offline)
        report = append_utterances(args.corpus, iter_utterances(archive))
        write_report(report_path, report)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(2, f'DailyDialog import failed: {exc}\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
