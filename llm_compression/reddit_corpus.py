"""Sample Reddit comments, strip formatting/URLs, and build a shuffled JSONL corpus."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
import hashlib
from html.parser import HTMLParser
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
import time
import uuid


class _PlainText(HTMLParser):
    """Collect rendered text only; discard HTML tags, attributes, and comments."""

    BLOCKS = {'p', 'div', 'section', 'article', 'header', 'footer', 'aside',
              'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li',
              'blockquote', 'pre', 'table', 'thead', 'tbody', 'tr', 'hr', 'br'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        if self.hidden:
            return
        if tag in self.BLOCKS:
            self.parts.append('\n')
        elif tag in ('td', 'th'):
            self.parts.append(' ')
        elif tag == 'img':
            self.parts.append(dict(attrs).get('alt') or '')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in self.BLOCKS:
            self.parts.append('\n')
        elif not self.hidden and tag in ('td', 'th'):
            self.parts.append(' ')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


@lru_cache(maxsize=1)
def _markdown_parser():
    import mistune
    return mistune.create_markdown(escape=False,
        plugins=['strikethrough', 'table', 'task_lists', 'superscript'])


_URL = re.compile(r"(?i)(?<![\w@])(?:(?:[a-z][a-z0-9+.-]*://)|www\.)[^\s<>]+")


def _remove_url(match):
    # Leave sentence punctuation and any enclosing brackets outside the URL.
    url, suffix = match.group(), ''
    pairs = {')': '(', ']': '[', '}': '{'}
    while url:
        last = url[-1]
        if last in '.,!?;:' or (last in pairs and url.count(last) > url.count(pairs[last])):
            suffix = last + suffix
            url = url[:-1]
        else:
            break
    return suffix


def clean_comment(body):
    """Strip Reddit Markdown/HTML and URLs, retaining readable plain text.

    Link labels, image descriptions, code contents, and Unicode are retained.
    Markdown markers are parsed rather than deleting literal word underscores
    or arithmetic asterisks. Empty lines, including whitespace-only ones, collapse
    to one newline. A URL/formatting-only comment may become an empty string.
    """
    # Reddit's spoiler and superscript syntax extends standard Markdown.
    body = re.sub(r'>!(.*?)!<', r'\1', body, flags=re.DOTALL)
    body = re.sub(r'(?<!\\)\^\(([^()\n]*)\)', r'\1', body)
    body = re.sub(r'(?<![\w\\])\^([^\s^()]+)', r'\1', body)
    parser = _PlainText()
    parser.feed(_markdown_parser()(body))
    parser.close()
    text = _URL.sub(_remove_url, ''.join(parser.parts))
    text = '\n'.join(re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.split('\n'))
    return re.sub(r'\n+', '\n', text).strip()


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def query(connection, sql, parameters, label):
    print(label,flush=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result=executor.submit(connection.execute,sql,parameters)
        while True:
            try: return result.result(timeout=30)
            except KeyboardInterrupt:
                connection.interrupt()
                raise
            except TimeoutError:
                progress=connection.query_progress()
                print(f'{label}: {progress:.1f}%' if progress>=0 else f'{label}: still scanning',flush=True)



ELIGIBLE = r"""body IS NOT NULL AND regexp_matches(body, '\S')
    AND lower(regexp_replace(body, '^\s+|\s+$', '', 'g')) NOT IN ('[deleted]','[removed]')"""


def select_comments(connection, files, count, seed, total, method, candidate_factor):
    attempts=[]
    percent=min(100.0,100.0*count*candidate_factor/max(total,1))
    while True:
        exact=method=='exact' or percent>=100
        if exact:
            source='SELECT body,id,subreddit_name_prefixed,filename FROM read_parquet(?,filename=true)'
            cte=''
        else:
            cte=f'''WITH candidates AS MATERIALIZED (
                SELECT body,id,subreddit_name_prefixed,filename FROM read_parquet(?,filename=true)
                USING SAMPLE system({percent:.12f} PERCENT) REPEATABLE({seed}))'''
            source='SELECT body,id,subreddit_name_prefixed,filename FROM candidates'
        sql=f'''CREATE TEMP TABLE selected AS {cte}
            SELECT body,id,subreddit_name_prefixed,filename FROM (
                SELECT * FROM ({source}) raw WHERE {ELIGIBLE}
            ) eligible USING SAMPLE reservoir({count} ROWS) REPEATABLE({seed})'''
        query(connection,sql,[[str(path) for path in files]],
              'Sampling all eligible rows' if exact else f'Fast preselection: {percent:.4f}% of blocks')
        selected=connection.execute('SELECT count(*) FROM selected').fetchone()[0]
        attempts.append({'block_percent':100.0 if exact else percent,'eligible_sample_rows':selected,
                         'full_scan_fallback':method=='fast' and exact})
        if selected>=count or exact: return attempts
        connection.execute('DROP TABLE selected')
        print(f'Only {selected:,} eligible candidates; doubling block coverage',flush=True)
        percent=min(100.0,percent*2)


def build(dataset, base, sample, output, count=1_000_000, seed=42, memory_limit='4GB', local_stage=False, method='exact', candidate_factor=4.0):
    import duckdb
    _markdown_parser()  # Check optional dependencies before scanning the dataset.
    dataset,base,sample,output=map(lambda p:Path(p).resolve(),(dataset,base,sample,output))
    if count<=0 or not 0<=seed<2**32: raise ValueError('Count must be positive; seed must fit uint32')
    if method not in ('exact','fast') or not math.isfinite(candidate_factor) or candidate_factor<1:
        raise ValueError('Method must be exact/fast; candidate factor must be finite and at least 1')
    report_path=output.with_suffix('.manifest.json')
    if len({sample,output,report_path,base})!=4 or any(p.exists() for p in (sample,output,report_path)):
        raise ValueError('Choose distinct, new sample/output/report paths; existing files are never overwritten')
    files=sorted(dataset.rglob('*.parquet'))
    if not files: raise ValueError('No Reddit Parquet files found')
    before=sha256(base); base_count=0
    with base.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip(): continue
            row=json.loads(line)
            if not isinstance(row,dict) or not isinstance(row.get('body',row.get('text')),str):
                raise ValueError('Existing corpus has a record without a string body/text')
            base_count+=1
    snapshots=[{'path':str(p.relative_to(dataset)),'bytes':p.stat().st_size} for p in files]
    for path in (sample,output): path.parent.mkdir(parents=True,exist_ok=True)
    run=uuid.uuid4().hex[:12]
    sample_tmp=sample.with_name(sample.name+'.'+run+'.partial')
    output_tmp=output.with_name(output.name+'.'+run+'.partial')
    scratch=output.parent/'.reddit-sampling-tmp'/run
    scratch.mkdir(parents=True,exist_ok=True)
    staging=None; scan_files=files; stage_root=dataset
    connection=duckdb.connect(); started=time.perf_counter()
    try:
        if local_stage:
            required=sum(item['bytes'] for item in snapshots)
            parent=Path(os.environ.get('LOCALAPPDATA',tempfile.gettempdir()))/'llm-compression/reddit-staging'
            parent.mkdir(parents=True,exist_ok=True)
            if shutil.disk_usage(parent).free<required+5*1024**3:
                raise ValueError('Insufficient local space for source snapshot plus 5 GiB reserve')
            staging=tempfile.TemporaryDirectory(prefix='snapshot-',dir=parent)
            stage_root=Path(staging.name).resolve()
            if stage_root.parent!=parent.resolve(): raise ValueError('Unexpected staging directory')
            def copy_shard(pair):
                source,item=pair
                destination=stage_root/source.relative_to(dataset)
                destination.parent.mkdir(parents=True,exist_ok=True)
                with source.open('rb') as inp,destination.open('wb',buffering=8*1024*1024) as out:
                    shutil.copyfileobj(inp,out,length=8*1024*1024)
                if destination.stat().st_size!=item['bytes']: raise ValueError('Source shard changed during staging')
                return destination,item['bytes']
            copied=0; last=time.monotonic(); scan_files=[]
            with ThreadPoolExecutor(max_workers=4) as pool:
                for destination,size in pool.map(copy_shard,zip(files,snapshots)):
                    scan_files.append(destination); copied+=size
                    if time.monotonic()-last>=10:
                        print(f'Staged {copied/1e9:.2f}/{required/1e9:.2f} GB',flush=True); last=time.monotonic()
        connection.execute('SET threads=1')
        connection.execute('SET memory_limit=?',[memory_limit])
        connection.execute('SET temp_directory=?',[str(scratch)])
        connection.execute('SET enable_progress_bar=true')
        connection.execute('SET enable_progress_bar_print=false')
        metadata=query(connection,'SELECT file_name,num_rows FROM parquet_file_metadata(?)',
                       [[str(p) for p in scan_files]],'Inspecting source shard metadata').fetchall()
        rows_by_file={str(Path(name)):rows for name,rows in metadata}
        for snapshot,path in zip(snapshots,scan_files): snapshot['rows']=rows_by_file[str(path)]
        total=sum(rows for _,rows in metadata)
        print(f'{len(files)} shards; {total:,} source rows; {base_count:,} existing records',flush=True)
        sampling_attempts=select_comments(connection,scan_files,count,seed,total,method,candidate_factor)
        selected,unique_ids=connection.execute('SELECT count(*),count(DISTINCT id) FROM selected').fetchone()
        if selected!=count: raise ValueError(f'Only {selected:,} eligible comments found; requested {count:,}')
        subreddits,shards=Counter(),Counter()
        preprocessing=Counter()
        cursor=connection.execute('SELECT body,id,subreddit_name_prefixed,filename FROM selected')
        with sample_tmp.open('w',encoding='utf-8',newline='\n') as stream:
            while True:
                batch=cursor.fetchmany(4096)
                if not batch: break
                for body,identifier,subreddit,filename in batch:
                    shard=str(Path(filename).relative_to(stage_root))
                    cleaned=clean_comment(body)
                    preprocessing['changed_comments'] += cleaned != body
                    preprocessing['empty_after_preprocessing'] += not cleaned
                    preprocessing['original_utf8_bytes'] += len(body.encode('utf-8'))
                    preprocessing['cleaned_utf8_bytes'] += len(cleaned.encode('utf-8'))
                    row={'body':cleaned,'source':'REDDIT_comments','source_id':identifier,
                         'subreddit':subreddit,'source_file':shard}
                    stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                    subreddits[subreddit or '']=subreddits.get(subreddit or '',0)+1
                    shards[shard]+=1
        # Sort original JSONL lines, preserving every record and its fields.
        connection.execute('SELECT setseed(?)',[seed/2**32])
        cursor=query(connection,'''SELECT line FROM read_csv(?,header=false,
            columns={'line':'VARCHAR'},delim='\x1f',quote='',escape='',strict_mode=false,
            max_line_size=16777216) WHERE line IS NOT NULL ORDER BY random()''',
            [[str(base),str(sample_tmp)]],'Shuffling existing corpus and sampled comments')
        combined=0
        with output_tmp.open('w',encoding='utf-8',newline='\n') as stream:
            while True:
                batch=cursor.fetchmany(4096)
                if not batch: break
                for (line,) in batch:
                    stream.write(line+'\n'); combined+=1
        if combined!=base_count+count: raise ValueError('Combined record count mismatch')
        if sha256(base)!=before: raise ValueError('Original corpus changed while sampling; no outputs published')
        for p,snapshot in zip(files,snapshots):
            if p.stat().st_size!=snapshot['bytes']: raise ValueError('Reddit source changed during sampling')
        report={'format':'reddit-combined-corpus-v1','seed':seed,'sampled_comments':selected,
                'distinct_nonnull_reddit_ids':unique_ids,'source_rows':total,'source_files':snapshots,
                'sampling_method':method,'candidate_factor':candidate_factor,'sampling_attempts':sampling_attempts,
                'sampling':('Uniform reservoir without replacement over all eligible rows; one DuckDB thread.' if method=='exact' else
                            'Random system-block preselection, eligibility filtering, then reservoir sampling; clustered approximate global distribution.'),
                'eligibility':'Raw non-null body containing non-whitespace, excluding case-insensitive trimmed [deleted]/[removed], before preprocessing.',
                'preprocessing':{'version':'reddit-plain-text-v1', **dict(preprocessing),
                                 'rules':'Strip Markdown/HTML formatting and URLs; keep visible link labels, image descriptions and code; collapse repeated newlines and horizontal whitespace; trim surrounding whitespace.',
                                 'empty_policy':'Retain selected comments that become empty, preserving the requested sample count.'},
                'combined_order':'Seeded random shuffle; original JSON records and fields preserved.',
                'base':{'path':str(base),'records':base_count,'sha256':before},
                'sample':{'path':str(sample),'sha256':sha256(sample_tmp),'bytes':sample_tmp.stat().st_size},
                'output':{'path':str(output),'records':combined,'sha256':sha256(output_tmp),'bytes':output_tmp.stat().st_size},
                'sample_by_subreddit':dict(subreddits),'sample_by_shard':dict(shards),
                'duckdb_version':duckdb.__version__,'elapsed_seconds':time.perf_counter()-started,
                'notes':['No text deduplication; distinct comments can have identical bodies.',
                         'Source snapshot records filenames, sizes, and row counts; source Parquet bytes are not rehashed.']}
        # Atomic publication without replacing any preexisting user files.
        os.link(sample_tmp,sample); os.link(output_tmp,output)
        with report_path.open('x',encoding='utf-8') as stream: stream.write(json.dumps(report,indent=2)+'\n')
        print(f'Created {output}: {combined:,} records ({base_count:,} existing + {selected:,} Reddit)',flush=True)
        print(f'Sample: {sample}\nManifest: {report_path}',flush=True)
        return report
    finally:
        connection.close()
        if staging is not None:
            staging.cleanup()
        sample_tmp.unlink(missing_ok=True); output_tmp.unlink(missing_ok=True)
        # DuckDB removes spill files on close; remove only our empty scratch directories.
        try: scratch.rmdir(); scratch.parent.rmdir()
        except OSError: pass


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('data/REDDIT_comments'))
    parser.add_argument('--base',type=Path,default=Path('data/combined-messages.jsonl'))
    parser.add_argument('--sample',type=Path,default=Path('data/reddit-comments-1m.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('data/combined-messages-reddit-1m.jsonl'))
    parser.add_argument('--count',type=int,default=1_000_000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--local-stage',action='store_true',help='stage the source snapshot locally to accelerate network-share reads')
    parser.add_argument('--method',choices=('exact','fast'),default='exact',help='exact scans/filters every row; fast preselects random blocks')
    parser.add_argument('--candidate-factor',type=float,default=4.0,help='fast-mode initial raw candidate target / requested count (default 4); coverage doubles if insufficient')
    parser.add_argument('--memory-limit',default='4GB')
    args=parser.parse_args(argv)
    try:
        import duckdb
        build(args.dataset,args.base,args.sample,args.output,args.count,args.seed,args.memory_limit,args.local_stage,args.method,args.candidate_factor)
    except (OSError,ValueError,ImportError) as exc: parser.exit(2,f'Reddit sampling failed: {exc}\n')
    except duckdb.Error as exc:
        parser.exit(2,f'Reddit sampling failed: {exc}\n')
    except KeyboardInterrupt:
        parser.exit(130,'Reddit sampling interrupted; incomplete output files were not published.\n')
    return 0
