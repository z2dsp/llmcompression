import json
from pathlib import Path
import pytest
pytest.importorskip('duckdb')
pytest.importorskip('mistune')
import duckdb
from llm_compression.reddit_corpus import build


def fixture_corpus(tmp_path):
    root=tmp_path/'reddit'; root.mkdir()
    con=duckdb.connect()
    con.execute('CREATE TABLE comments(body VARCHAR,id VARCHAR,subreddit_name_prefixed VARCHAR)')
    for shard in range(2):
        con.execute('DELETE FROM comments')
        con.executemany('INSERT INTO comments VALUES (?,?,?)',[
            (f'body {shard}-{i}\nCafé',f'{shard}-{i}',f'r/test{shard}') for i in range(80)] +
            [(v,f'bad-{shard}-{i}','r/test') for i,v in enumerate(['','   ','[deleted]','[removed]',None,' \t[DELETED]\n','\n[Removed]\t'])])
        con.execute('COPY comments TO ? (FORMAT PARQUET)',[str(root/f'{shard}.parquet')])
    con.close()
    base=tmp_path/'base.jsonl'
    records=[{'body':'existing message','extra':{'keep':True}}, {'body':'second','source':'old'}]
    base.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    return root,base,records


@pytest.mark.parametrize('stage', [False, True])
def test_uniform_sample_shuffle_preserves_base_and_is_repeatable(tmp_path, monkeypatch, stage):
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path))
    root,base,original=fixture_corpus(tmp_path)
    before=base.read_bytes()
    manifests=[]
    for n in range(2):
        sample=tmp_path/f'sample{n}.jsonl'; output=tmp_path/f'out{n}.jsonl'
        build(root,base,sample,output,count=37,seed=42,memory_limit='128MB',local_stage=stage)
        selected=[json.loads(line) for line in sample.read_text(encoding='utf-8').splitlines()]
        combined=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
        assert len(selected)==37 and len(combined)==39
        assert len({r['source_id'] for r in selected})==37
        assert all(r['body'].startswith('body ') for r in selected)
        assert {r['subreddit'] for r in selected}=={'r/test0','r/test1'}
        assert all(r in combined for r in original+selected)
        assert base.read_bytes()==before
        manifests.append(json.loads(output.with_suffix('.manifest.json').read_text()))
    assert manifests[0]['sample']['sha256']==manifests[1]['sample']['sha256']
    assert manifests[0]['output']['sha256']==manifests[1]['output']['sha256']
    with pytest.raises(ValueError,match='new'):
        build(root,base,sample,output,count=37)


def test_short_dataset_does_not_publish_partial_outputs(tmp_path):
    root,base,_=fixture_corpus(tmp_path)
    sample,output=tmp_path/'sample.jsonl',tmp_path/'out.jsonl'
    with pytest.raises(ValueError,match='eligible'):
        build(root,base,sample,output,count=1000,memory_limit='128MB')
    assert not sample.exists() and not output.exists()

@pytest.mark.parametrize('factor', [1.0,4.0])
def test_fast_mode_filters_and_returns_exact_count(tmp_path, factor):
    root,base,original=fixture_corpus(tmp_path)
    sample,output=tmp_path/'sample.jsonl',tmp_path/'out.jsonl'
    report=build(root,base,sample,output,count=37,method='fast',candidate_factor=factor,memory_limit='128MB')
    records=[json.loads(line) for line in sample.read_text(encoding='utf-8').splitlines()]
    assert len(records)==37 and len({r['source_id'] for r in records})==37
    assert all(r['body'].startswith('body ') for r in records)
    assert report['sampling_method']=='fast'
    assert report['candidate_factor']==factor
    assert 'clustered' in report['sampling']
    assert report['sampling_attempts'][-1]['eligible_sample_rows']==37
    assert all(0<a['block_percent']<=100 for a in report['sampling_attempts'])
    combined=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    assert len(combined)==39 and all(r in combined for r in original+records)


def test_invalid_candidate_factor_rejected_before_scanning(tmp_path):
    root,base,_=fixture_corpus(tmp_path)
    for factor in (0.0,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='candidate factor'):
            build(root,base,tmp_path/'s.jsonl',tmp_path/'o.jsonl',method='fast',candidate_factor=factor)


def test_sample_is_cleaned_before_writing_both_outputs(tmp_path):
    root = tmp_path / 'reddit'
    root.mkdir()
    connection = duckdb.connect()
    connection.execute('CREATE TABLE comments(body VARCHAR,id VARCHAR,subreddit_name_prefixed VARCHAR)')
    connection.executemany('INSERT INTO comments VALUES (?,?,?)', [
        ('**Hello** [friend](https://example.com)\n\n\n_next line_', 'formatted', 'r/test'),
        ('https://example.com', 'url-only', 'r/test'),
    ])
    connection.execute('COPY comments TO ? (FORMAT PARQUET)', [str(root / 'data.parquet')])
    connection.close()
    base = tmp_path / 'base.jsonl'
    original = {'body': '**existing** https://keep.example\n\ntext', 'extra': {'keep': True}}
    base.write_text(json.dumps(original) + '\n', encoding='utf-8')
    before = base.read_bytes()
    sample, output = tmp_path / 'sample.jsonl', tmp_path / 'combined.jsonl'
    report = build(root, base, sample, output, count=2, memory_limit='128MB')
    rows = [json.loads(line) for line in sample.read_text().splitlines()]
    assert {row['source_id']: row['body'] for row in rows} == {
        'formatted': 'Hello friend\nnext line', 'url-only': ''}
    combined = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(combined) == 3 and all(row in combined for row in [original, *rows])
    assert base.read_bytes() == before
    assert report['preprocessing']['version'] == 'reddit-plain-text-v1'
    assert report['preprocessing']['changed_comments'] == 2
    assert report['preprocessing']['empty_after_preprocessing'] == 1
