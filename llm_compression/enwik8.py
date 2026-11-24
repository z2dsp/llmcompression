"""Download verified enwik8 and benchmark lossless codecs on identical bytes."""
import argparse
import bz2
import csv
import gzip
import hashlib
import json
import lzma
from pathlib import Path
import platform
import struct
import time
import urllib.request
import zipfile

URL = 'https://mattmahoney.net/dc/enwik8.zip'
SIZE = 100_000_000
MD5 = 'a1fa5ffddb56f4953e226637dabbb36a'
SHA1 = '57b8363b814821dc9d47aa4d41f58733519076b2'
HEADER = struct.Struct('<8s32s32sQI')
BLOCK = struct.Struct('<II')


def digest_file(path, limit=0):
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        while not limit or size < limit:
            chunk = stream.read(min(1048576, limit - size) if limit else 1048576)
            if not chunk:
                break
            digest.update(chunk); size += len(chunk)
    return size, digest.hexdigest()


def verify_dataset(path):
    md5, sha1, size = hashlib.md5(), hashlib.sha1(), 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            md5.update(chunk); sha1.update(chunk); size += len(chunk)
    if (size, md5.hexdigest(), sha1.hexdigest()) != (SIZE, MD5, SHA1):
        raise ValueError('enwik8 size or published checksum mismatch')


def download(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'enwik8'
    if output.exists():
        verify_dataset(output)
        return output
    archive = directory / 'enwik8.zip'
    if not archive.exists():
        partial = archive.with_suffix('.zip.partial')
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open('wb') as target:
            count, last = 0, time.monotonic()
            while True:
                chunk = response.read(1048576)
                if not chunk: break
                target.write(chunk); count += len(chunk)
                if time.monotonic() - last > 10:
                    print(f'Downloaded {count:,} bytes', flush=True); last = time.monotonic()
        partial.replace(archive)
    partial = directory / 'enwik8.extracting'
    try:
        with zipfile.ZipFile(archive) as zipped:
            info = zipped.getinfo('enwik8')
            if info.file_size != SIZE:
                raise ValueError('Unexpected enwik8 ZIP member size')
            with zipped.open(info) as source, partial.open('wb') as target:
                for chunk in iter(lambda: source.read(1048576), b''): target.write(chunk)
        verify_dataset(partial)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    (directory / 'source.json').write_text(json.dumps({'url': URL, 'description': 'First 100,000,000 bytes of the 2006 English Wikipedia XML dump; no normalization.',
        'size': SIZE, 'md5': MD5, 'sha1': SHA1, 'reference': 'https://mattmahoney.net/dc/textdata.html'}, indent=2)+'\n')
    return output


def chunks(path, limit=0, block_size=1048576):
    used = 0
    with Path(path).open('rb') as stream:
        while not limit or used < limit:
            data = stream.read(min(block_size, limit-used) if limit else block_size)
            if not data: break
            used += len(data)
            yield data


def load_model(path):
    path = Path(path)
    with path.open('rb') as stream: magic = stream.read(8)
    if magic in (b'MCBITLM1', b'MCBITLM2'):
        from .bit_model import BitModel
        from .bit_codec import compress_payload, decompress_payload
        model = BitModel(path)
        return model, compress_payload, decompress_payload, model.close if hasattr(model, 'close') else lambda: None
    if magic in (b'MCNNLM01', b'MCNNLM02', b'MCNNLM03'):
        from .nn_model import NeuralModel, compress_payload, decompress_payload
        return NeuralModel(path), compress_payload, decompress_payload, lambda: None
    if magic in (b'MCPNN001', b'MCPNN002', b'MCPNN003'):
        from pico.nn_runtime import Model
        from .nn_model import compress_payload, decompress_payload
        return Model(path, backend='python'), compress_payload, decompress_payload, lambda: None
    raise ValueError(f'Unsupported model format: {path}')


def benchmark_standard(source, destination, algorithm, limit):
    started = time.perf_counter()
    if algorithm == 'gzip':
        with destination.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=9, mtime=0) as target:
            for data in chunks(source, limit): target.write(data)
        opener = gzip.open
    else:
        codec = bz2.BZ2Compressor(9) if algorithm == 'bzip2' else lzma.LZMACompressor(format=lzma.FORMAT_XZ, preset=9)
        with destination.open('wb') as target:
            for data in chunks(source, limit): target.write(codec.compress(data))
            target.write(codec.flush())
        opener = bz2.open if algorithm == 'bzip2' else lzma.open
    encode_seconds = time.perf_counter() - started
    started, restored, digest = time.perf_counter(), 0, hashlib.sha256()
    with opener(destination, 'rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            restored += len(data); digest.update(data)
    return encode_seconds, time.perf_counter()-started, restored, digest.hexdigest(), destination.stat().st_size


def benchmark_model(source, destination, path, limit, block_size, original_size, original_hash):
    model, compress, decompress, close = load_model(path)
    model_hash = digest_file(path)[1]
    started, payload_bytes = time.perf_counter(), 0
    try:
        with destination.open('wb') as target:
            target.write(HEADER.pack(b'ENWKBM01', bytes.fromhex(model_hash), bytes.fromhex(original_hash), original_size, block_size))
            completed = 0
            for data in chunks(source, limit, block_size):
                payload = compress(data, model)
                target.write(BLOCK.pack(len(data), len(payload))); target.write(payload)
                payload_bytes += len(payload); completed += len(data)
                print(f'{Path(path).name}: encoded {completed:,}/{original_size:,} bytes', flush=True)
        encode_seconds = time.perf_counter()-started
        started, restored, digest = time.perf_counter(), 0, hashlib.sha256()
        with destination.open('rb') as stream:
            magic, stored_model, stored_source, length, stored_block = HEADER.unpack(stream.read(HEADER.size))
            if magic != b'ENWKBM01' or stored_model.hex() != model_hash or length != original_size or stored_source.hex() != original_hash or stored_block != block_size:
                raise ValueError('Model archive metadata mismatch')
            while restored < length:
                header = stream.read(BLOCK.size)
                if len(header) != BLOCK.size: raise ValueError('Truncated block header')
                count, size = BLOCK.unpack(header)
                if not 0 < count <= min(block_size, length-restored) or size == 0: raise ValueError('Invalid block length')
                payload = stream.read(size)
                if len(payload) != size: raise ValueError('Truncated model payload')
                data = decompress(payload, count, model)
                digest.update(data); restored += len(data)
                print(f'{Path(path).name}: verified {restored:,}/{original_size:,} bytes', flush=True)
            if stream.read(1): raise ValueError('Trailing model archive data')
        return encode_seconds, time.perf_counter()-started, restored, digest.hexdigest(), payload_bytes
    finally:
        close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='custom input; default downloads/verifies canonical enwik8')
    parser.add_argument('--data-dir', type=Path, default=Path('data/enwik8'))
    parser.add_argument('--output', type=Path, default=Path('data/enwik8/benchmark'))
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--model', type=Path, action='append', default=[], help='repeatable .mcnn, .mcblm, or .pnn model; includes actual arithmetic coding and decoding')
    parser.add_argument('--algorithms', default='gzip,bzip2,xz', help='comma-separated conventional codecs')
    parser.add_argument('--limit-bytes', type=int, default=0, help='benchmark only this prefix; 0 means entire input')
    parser.add_argument('--model-block-bytes', type=int, default=262144, help='independent model coding blocks; bounds memory and prints progress')
    args = parser.parse_args(argv)
    try:
        if args.limit_bytes < 0 or not 0 < args.model_block_bytes <= 0xffffffff:
            raise ValueError('Invalid byte limit or model block size')
        algorithms = list(dict.fromkeys(args.algorithms.split(','))) if args.algorithms else []
        if any(a not in ('gzip','bzip2','xz') for a in algorithms): raise ValueError('Choose gzip,bzip2,xz')
        if not algorithms and not args.model and not args.download_only: raise ValueError('No codecs selected')
        for model in args.model:
            if not model.is_file(): raise ValueError(f'Missing model: {model}')
        source = args.input or download(args.data_dir)
        if args.download_only:
            print(f'Dataset ready: {source} ({source.stat().st_size:,} bytes)'); return 0
        if args.output.exists() and any(args.output.iterdir()): raise ValueError('Output is not empty; choose a new --output')
        size, digest = digest_file(source, args.limit_bytes)
        if not size: raise ValueError('Input must be nonempty')
        args.output.mkdir(parents=True, exist_ok=True)
        report = {'input': str(source.resolve()), 'input_file_bytes': source.stat().st_size, 'benchmarked_bytes': size,
                  'sha256': digest, 'prefix_only': size < source.stat().st_size,
                  'model_block_bytes': args.model_block_bytes, 'python': platform.python_version(), 'results': [],
                  'notes': ['All rows use identical original bytes and are verified by decompressed length and SHA-256.',
                            'Standard codecs are one continuous stream at level/preset 9, including native container overhead.',
                            'Model archives reset context per block and include an 84-byte header plus 8 bytes per block; no radio framing.',
                            'Shared-model sizes exclude model storage; self-contained totals add the supplied model file, not decoder/runtime code.',
                            'Models are not trained here. Record training overlap separately; enwik8 is Wikipedia XML, not cleaned article text.']}
        jobs = [(a,None) for a in algorithms]+[(f'model-{i+1}', p) for i,p in enumerate(args.model)]
        for name, model in jobs:
            destination = args.output / (name + ('.bin' if model else {'gzip':'.gz','bzip2':'.bz2','xz':'.xz'}[name]))
            print(f'Running {name} on {size:,} bytes' + (f' with {model}' if model else ''), flush=True)
            result = (benchmark_model(source,destination,model,args.limit_bytes,args.model_block_bytes,size,digest)
                      if model else benchmark_standard(source,destination,name,args.limit_bytes))
            enc, dec, restored, restored_hash, payload_size = result
            if (restored, restored_hash) != (size,digest): raise ValueError(f'{name}: round-trip mismatch')
            compressed = destination.stat().st_size
            model_bytes = model.stat().st_size if model else 0
            row = {'algorithm':name,'model':str(model) if model else None,'model_sha256':digest_file(model)[1] if model else None,
                   'compressed_bytes':compressed,'arithmetic_payload_bytes':payload_size if model else None,
                   'percent_of_original':100*compressed/size,'bits_per_byte':8*compressed/size,
                   'model_bytes':model_bytes,'bytes_including_model':compressed+model_bytes,
                   'encode_seconds':enc,'decode_seconds':dec,'round_trip_exact':True,'artifact':str(destination)}
            report['results'].append(row)
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            with (args.output/'report.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader(); writer.writerows(report['results'])
            print(f'{name}: {compressed:,} bytes, {8*compressed/size:.4f} bits/byte, verified',flush=True)
        print(f'Report: {args.output / "report.json"}')
    except (OSError,ValueError,ImportError,zipfile.BadZipFile,KeyError) as exc:
        parser.exit(2,f'Benchmark failed: {exc}\n')
    return 0
