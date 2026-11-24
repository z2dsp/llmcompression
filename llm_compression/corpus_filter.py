"""Atomically remove named channels from existing MeshCore corpus exports."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import uuid

DEFAULT_EXCLUSIONS = ('#test', '#alert', '#bot', '#nrl')


def channel_key(value):
    return value.strip().casefold().removeprefix('#') if isinstance(value, str) else None


def filter_channels(path, excluded=DEFAULT_EXCLUSIONS):
    path = Path(path).resolve()
    keys = {channel_key(value) for value in excluded}
    if not keys or None in keys or '' in keys:
        raise ValueError('Provide nonempty channel names')
    if path.suffix.lower() not in ('.jsonl', '.csv'):
        raise ValueError('Expected a JSONL or CSV corpus export')
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    removed, retained, original = Counter(), 0, 0
    temporary = None
    try:
        with path.open(encoding='utf-8', newline='') as source, tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', newline='', dir=path.parent,
                prefix='.filtered-messages-', suffix='.tmp', delete=False) as target:
            temporary = Path(target.name)
            if path.suffix.lower() == '.csv':
                reader = csv.DictReader(source)
                if not reader.fieldnames or 'channel' not in reader.fieldnames:
                    raise ValueError('Corpus CSV has no channel column')
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()
                rows = ((row, None) for row in reader)
            else:
                rows = ((json.loads(line), line) for line in source if line.strip())
            for row, raw in rows:
                if not isinstance(row, dict) or 'channel' not in row:
                    raise ValueError('Corpus record has no channel metadata')
                original += 1
                if channel_key(row['channel']) in keys:
                    removed[row['channel']] += 1
                    continue
                retained += 1
                if raw is not None:
                    target.write(raw)
                else:
                    writer.writerow(row)
        backup = None
        if removed:
            if hashlib.sha256(path.read_bytes()).hexdigest() != before:
                raise ValueError('Corpus changed during filtering; original was not replaced')
            backup = path.with_name(path.name + '.before-channel-filter-' + uuid.uuid4().hex[:12] + '.bak')
            with path.open('rb') as source, backup.open('xb') as target:
                shutil.copyfileobj(source, target)
            temporary.replace(path)
            temporary = None
        return {'path': str(path), 'original_messages': original, 'retained_messages': retained,
                'removed_messages': sum(removed.values()), 'removed_by_channel': dict(removed),
                'excluded_channels': list(excluded), 'before_sha256': before,
                'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'backup': str(backup) if backup else None}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path,
                        default=[Path('data/messages.jsonl'), Path('data/messages.csv')])
    parser.add_argument('--exclude-channel', action='append', help='repeatable; defaults to #test, #alert, #bot, #nrl')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    try:
        if any(not path.is_file() for path in args.paths):
            raise ValueError('Every corpus path must exist before filtering')
        results = [filter_channels(path, args.exclude_channel or DEFAULT_EXCLUSIONS) for path in args.paths]
        text = json.dumps(results, indent=2) + '\n'
        if args.report:
            args.report.write_text(text, encoding='utf-8')
        print(text, end='')
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Corpus filtering failed: {exc}\n')
    return 0
