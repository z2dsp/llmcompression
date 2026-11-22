"""Reproducible, resumable Wikipedia download and bounded-memory preparation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .training_corpus import RecordWriter

REPO = "wikimedia/wikipedia"
SUBSET = "20231101.en"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
USER_AGENT = "llm-compression-demo/0.1 (Wikipedia training corpus)"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def request(url, **headers):
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT, **headers}), timeout=90)


def get_download_manifest(directory, revision=REVISION):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Revision must be an immutable 40-character commit hash")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "download-manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest["repository"], manifest["subset"], manifest["revision"]) != (REPO, SUBSET, revision):
            raise ValueError("Existing download manifest belongs to a different dataset revision")
        return manifest
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{SUBSET}"
    with request(url) as response:
        entries = json.load(response)
    files = []
    for entry in entries:
        if entry["type"] == "file" and re.fullmatch(rf"{re.escape(SUBSET)}/train-\d{{5}}-of-00041\.parquet", entry["path"]):
            files.append({"path": entry["path"], "bytes": entry["size"], "sha256": entry["lfs"]["oid"]})
    files.sort(key=lambda entry: entry["path"])
    if len(files) != 41 or any(not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) for entry in files):
        raise ValueError("Expected all 41 English Wikipedia shards and their LFS SHA-256 checksums")
    manifest = {"format": "wikipedia-download-v1", "repository": REPO, "subset": SUBSET,
                "revision": revision, "source": f"https://huggingface.co/datasets/{REPO}",
                "license_source": f"https://huggingface.co/datasets/{REPO}/blob/{revision}/README.md",
                "files": files, "total_bytes": sum(entry["bytes"] for entry in files)}
    write_json(path, manifest)
    return manifest


def download_file(directory, entry, revision, attempts=8):
    destination = Path(directory) / entry["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != entry["bytes"] or sha256_file(destination) != entry["sha256"]:
            raise ValueError(f"Existing file fails verification: {destination}; move it aside before retrying")
        return destination, "verified existing"
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{quote(entry['path'])}?download=true"
    for attempt in range(attempts):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset > entry["bytes"]:
                raise ValueError(f"Partial file is larger than expected: {temporary}")
            if offset < entry["bytes"]:
                with request(url, **({"Range": f"bytes={offset}-"} if offset else {})) as response:
                    status = response.status
                    if status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError("Server returned the wrong byte range")
                    elif status == 200:
                        offset = 0  # Range ignored: restart, do not append a second copy.
                    else:
                        raise ValueError(f"Unexpected HTTP status {status}")
                    with temporary.open("ab" if offset else "wb") as output:
                        while block := response.read(4 * 1024 * 1024):
                            output.write(block)
                            if output.tell() > entry["bytes"]:
                                raise ValueError("Download exceeds expected file size")
            if temporary.stat().st_size != entry["bytes"]:
                raise OSError("Incomplete download")
            if sha256_file(temporary) != entry["sha256"]:
                temporary.unlink()
                raise OSError("Download SHA-256 mismatch; retrying from the start")
            temporary.replace(destination)
            return destination, "downloaded and verified"
        except (OSError, HTTPError, URLError) as exc:
            if attempt + 1 == attempts:
                raise OSError(f"Failed downloading {entry['path']} after {attempts} attempts: {exc}") from exc
            print(f"Retry {attempt + 1}/{attempts}: {entry['path']}: {exc}", file=sys.stderr, flush=True)
            time.sleep(min(30, 2 ** attempt))


def download(directory, *, revision=REVISION, workers=4):
    if not 1 <= workers <= 16:
        raise ValueError("Download workers must be between 1 and 16")
    manifest = get_download_manifest(directory, revision)
    print(f"Downloading all {len(manifest['files'])} English Wikipedia shards: {manifest['total_bytes']:,} bytes; revision {revision}", flush=True)
    card = Path(directory) / "README.source.md"
    if not card.exists():
        with request(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/README.md") as response:
            text = response.read().decode("utf-8")
        card.write_text(text, encoding="utf-8")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_file, directory, entry, revision): entry for entry in manifest["files"]}
        completed = 0
        for future in as_completed(futures):
            path, status = future.result()
            completed += 1
            print(f"[{completed}/{len(futures)}] {path.name}: {status} ({time.perf_counter() - started:.1f}s)", flush=True)
    write_json(Path(directory) / "download-complete.json", {"revision": revision, "files_verified": len(manifest["files"]),
                                                            "bytes_verified": manifest["total_bytes"]})
    return manifest


def article_split(data, seed, validation_fraction, test_fraction):
    digest = hashlib.sha256(b"WIKI-SPLIT-v1\0" + str(seed).encode("ascii") + b"\0" + data).digest()
    value = int.from_bytes(digest[:8], "big")
    # Integer thresholds, rather than a rounded floating-point hash fraction.
    if value < int(test_fraction * (1 << 64)):
        return "test"
    if value < int((test_fraction + validation_fraction) * (1 << 64)):
        return "validation"
    return "train"


def chunk_utf8(data, size):
    """Lossless UTF-8 chunks, preferring a whitespace boundary in the latter half."""
    if size == 0:
        if data:
            yield data
        return
    if size < 4:
        raise ValueError("UTF-8 sample size must be at least 4 bytes, or 0 for whole articles")
    start = 0
    while start < len(data):
        end = min(start + size, len(data))
        if end < len(data):
            while data[end] & 0xC0 == 0x80:
                end -= 1
            boundary = max(data.rfind(b" ", start + size // 2, end),
                           data.rfind(b"\n", start + size // 2, end))
            if boundary >= start:
                end = boundary + 1
        if end <= start:
            raise ValueError("Sample size must fit a complete UTF-8 character")
        yield data[start:end]
        start = end


def prepare_shard(raw_directory, entry, output, config):
    import pyarrow.parquet as pq

    output = Path(output)
    identifier = Path(entry["path"]).stem.split("-")[1]
    report_path = output / "parts" / f"{identifier}.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous["source_sha256"] != entry["sha256"] or previous["preparation"] != config:
            raise ValueError("Prepared shard configuration changed; choose a new output directory")
        for split in previous["splits"].values():
            path = output / split["path"]
            if not path.exists() or path.stat().st_size != split["file_bytes"]:
                raise ValueError(f"Incomplete prepared shard: {path}; move its checkpoint aside to rebuild")
        return previous
    source = Path(raw_directory) / entry["path"]
    if source.stat().st_size != entry["bytes"] or sha256_file(source) != entry["sha256"]:
        raise ValueError(f"Source shard failed SHA-256 verification: {source}")
    writers, temporary = {}, {}
    for name in ("train", "validation", "test"):
        temporary[name] = output / "parts" / f"{identifier}.{name}.mccorpus.tmp"
        writers[name] = RecordWriter(temporary[name], extended_lengths=config["sample_bytes"] == 0 or config["sample_bytes"] > 0xFFFFFFFF)
    articles = {name: 0 for name in writers}
    original_bytes = empty = 0
    completed = False
    try:
        with pq.ParquetFile(source) as parquet:
            for batch in parquet.iter_batches(batch_size=config["arrow_batch_articles"], columns=["text"], use_threads=False):
                for text in batch.column(0).to_pylist():
                    if not isinstance(text, str):
                        raise ValueError("Wikipedia article has a non-string text field")
                    data = text.encode("utf-8")
                    name = article_split(data, config["seed"], config["validation_fraction"], config["test_fraction"])
                    articles[name] += 1
                    original_bytes += len(data)
                    empty += not data
                    for message in chunk_utf8(data, config["sample_bytes"]):
                        writers[name].write(message)
        stats = {}
        for name, writer in writers.items():
            stats[name] = writer.close()
            final = temporary[name].with_suffix("")
            temporary[name].replace(final)
            stats[name].update({"path": str(final.relative_to(output)), "articles": articles[name]})
        if sum(row["utf8_bytes"] for row in stats.values()) != original_bytes:
            raise ValueError("Preparation lost article bytes")
        result = {"source_path": entry["path"], "source_sha256": entry["sha256"], "preparation": config,
                  "articles": sum(articles.values()), "empty_articles": empty, "original_utf8_bytes": original_bytes,
                  "splits": stats}
        write_json(report_path, result)  # Checkpoint is committed after all three files.
        completed = True
        return result
    finally:
        if not completed:
            for writer in writers.values():
                writer.stream.close()
            for path in temporary.values():
                path.unlink(missing_ok=True)


def prepare(raw_directory, output, *, sample_bytes=160, seed=42, validation_fraction=0.1,
            test_fraction=0.1, workers=2, arrow_batch_articles=128, wait_for_download=False):
    if importlib.util.find_spec("pyarrow") is None:
        raise ValueError("Wikipedia preparation requires PyArrow: python -m pip install pyarrow==23.0.1")
    if (sample_bytes != 0 and sample_bytes < 4) or not 1 <= workers <= 8 or arrow_batch_articles < 1:
        raise ValueError("Sample size must be at least 4 bytes or 0 for whole articles; workers 1–8; Arrow batch size positive")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    raw_directory, output = Path(raw_directory), Path(output)
    source = json.loads((raw_directory / "download-manifest.json").read_text())
    if source.get("format") != "wikipedia-download-v1" or len(source["files"]) != 41:
        raise ValueError("Expected a complete 41-shard download manifest")
    output.mkdir(parents=True, exist_ok=True)
    (output / "parts").mkdir(exist_ok=True)
    config = {"version": 1, "sample_bytes": sample_bytes, "seed": seed,
              "validation_fraction": validation_fraction, "test_fraction": test_fraction,
              "arrow_batch_articles": arrow_batch_articles,
              "split_method": "SHA-256 of seed and whole article UTF-8 text, before lossless chunking; identical articles share a split"}
    plan = {"revision": source["revision"], "preparation": config}
    plan_path = output / "preparation.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Output belongs to another preparation configuration; choose a new directory")
    write_json(plan_path, plan)
    pending = list(source["files"])
    results = []
    started = last_progress = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active = {}
        while pending or active:
            ready = [entry for entry in pending if (raw_directory / entry["path"]).is_file()]
            for entry in ready[:max(0, workers - len(active))]:
                pending.remove(entry)
                active[pool.submit(prepare_shard, raw_directory, entry, output, config)] = entry
            if not active and pending and not wait_for_download:
                raise ValueError("Some raw shards are missing; finish download or use --wait-for-download")
            if not active:
                if time.monotonic() - started > 14400:
                    raise TimeoutError("Timed out waiting for missing download shards after four hours; rerun to resume")
                time.sleep(5)
            else:
                done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    entry = active.pop(future)
                    result = future.result()
                    results.append(result)
                    print(f"[{len(results)}/41] prepared {Path(entry['path']).name}: {result['articles']:,} articles, {result['original_utf8_bytes']:,} UTF-8 bytes", flush=True)
            if time.monotonic() - last_progress >= 30:
                print(f"Preparation: {len(results)}/41 finished; {len(active)} active; {len(pending)} awaiting processing/download", flush=True)
                last_progress = time.monotonic()
    results.sort(key=lambda result: result["source_path"])
    splits = {}
    for name in ("train", "validation", "test"):
        files = [result["splits"][name] for result in results]
        splits[name] = {key: sum(row[key] for row in files)
                        for key in ("messages", "articles", "utf8_bytes", "unicode_characters", "file_bytes")}
        splits[name]["files"] = files
    manifest = {"format": "mesh-compression-corpus-v1", "source": source,
                "preparation": config, "articles": sum(row["articles"] for row in results),
                "empty_articles": sum(row["empty_articles"] for row in results),
                "original_utf8_bytes": sum(row["original_utf8_bytes"] for row in results), "splits": splits}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "articles": manifest["articles"],
                      "utf8_bytes": manifest["original_utf8_bytes"],
                      "samples": sum(row["messages"] for row in splits.values()),
                      "elapsed_seconds": time.monotonic() - started}, indent=2), flush=True)
    return manifest


def verify(directory, *, workers=2):
    """Verify saved raw download or prepared corpus checksums, entirely offline."""
    if not 1 <= workers <= 16:
        raise ValueError("Verification workers must be between 1 and 16")
    directory = Path(directory)
    path = directory / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("format") != "mesh-compression-corpus-v1":
            raise ValueError("Unsupported prepared corpus manifest")
        entries = [entry for split in manifest["splits"].values() for entry in split["files"]]
    else:
        manifest = json.loads((directory / "download-manifest.json").read_text())
        if manifest.get("format") != "wikipedia-download-v1":
            raise ValueError("Unsupported download manifest")
        entries = manifest["files"]
    def check(entry):
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Corpus manifest path escapes its directory")
        if path.stat().st_size != entry.get("file_bytes", entry.get("bytes")) or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"SHA-256/size verification failed: {path}")
        return path
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check, entry) for entry in entries]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"[{index}/{len(entries)}] verified {future.result().name}", flush=True)
    return len(entries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="download and SHA-256 verify all English snapshot shards; resumable")
    get.add_argument("--directory", type=Path, default=Path("data/wikipedia/raw"))
    get.add_argument("--revision", default=REVISION)
    get.add_argument("--workers", type=int, default=4)
    prep = commands.add_parser("prepare", help="prepare all 41 shards as lossless, article-split short-message records; resumable")
    prep.add_argument("--directory", type=Path, default=Path("data/wikipedia/raw"))
    prep.add_argument("--output", type=Path, default=Path("data/wikipedia/prepared"))
    prep.add_argument("--sample-bytes", type=int, default=160, help="maximum UTF-8 sample bytes, at least 4; 0 keeps whole articles; no software upper limit")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--validation-fraction", type=float, default=0.1)
    prep.add_argument("--test-fraction", type=float, default=0.1)
    prep.add_argument("--workers", type=int, default=2)
    prep.add_argument("--arrow-batch-articles", type=int, default=128)
    prep.add_argument("--wait-for-download", action="store_true", help="start processing finished shards while download continues")
    check = commands.add_parser("verify", help="offline SHA-256/size verification of raw or prepared files")
    check.add_argument("directory", type=Path)
    check.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            download(args.directory, revision=args.revision, workers=args.workers)
        elif args.command == "prepare":
            prepare(args.directory, args.output, sample_bytes=args.sample_bytes, seed=args.seed,
                    validation_fraction=args.validation_fraction, test_fraction=args.test_fraction,
                    workers=args.workers, arrow_batch_articles=args.arrow_batch_articles,
                    wait_for_download=args.wait_for_download)
        else:
            verify(args.directory, workers=args.workers)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
