# Train a MeshCore compression model

This workflow extracts the local Remote Terminal database or prepares English
Wikipedia, then trains a fixed,
size-bounded **statistical language model** for arithmetic compression. It learns
bit probabilities from message bytes. It is not a neural network: there are no
tokenizer, embeddings, floating-point inference, or KV cache. This keeps the model
and runtime state practical for an MCU implementation. Training and the reference
encoder/decoder are Python; native MCU firmware is not included.

## Extract messages

From the project directory:

```bash
# Standard library only; includes channel and private messages by default.
python extract_messages.py data/meshcore.db --output data/messages.jsonl
# Optional spreadsheet-friendly export (quoted multiline fields).
python extract_messages.py data/meshcore.db --output data/messages.csv
# Optional filters; choose a different output name.
python extract_messages.py --type channel --channel '#llmcompression' \
  --incoming-only --output data/channel-messages.jsonl
```

The database is opened read-only in a consistent, WAL-aware transaction. Keep the
database and its `-wal`/`-shm` companions together when copying an active database.
The extractor reads `messages` and channel names; it does not export channel keys,
contact keys, or application settings. Output creation is atomic and existing
files require `--overwrite`.

Each JSONL object / CSV row contains `id`, `type`, `text`, `body`, `sender_name`,
`channel`, `sender_timestamp`, `received_at`, and `outgoing`. `text` preserves the
original database string exactly, including Unicode, whitespace, and newlines.
`body` removes a channel's exact known `sender_name:` prefix and at most one
following space. It leaves private-message text untouched. Rows are ordered by
`received_at, id`; repeated messages are included in the export.

The supplied database yielded **72,340 messages**: 70,916 channel and 1,424 private,
containing 3,905,126 original UTF-8 text bytes. Both exports are in `data/`.

Exports and trained artifacts contain or derive from private conversations.
`data/` and `models/` are Git-ignored and excluded from the shareable demo ZIP.
Share a trained model deliberately: learned frequency tables are not a privacy
guarantee. To exclude private conversations, extract with `--type channel` before
training.

## Add DailyDialog utterances to the combined corpus

From the project directory, run this standard-library-only importer:

```bash
python append_dailydialog.py
# Reuse the downloaded archive without network access:
python append_dailydialog.py --offline
# Optional separate destination:
python append_dailydialog.py --corpus data/another-combined.jsonl
```

It downloads the pinned [ConvLab/DailyDialog dataset](https://huggingface.co/datasets/ConvLab/dailydialog)
(about 3.7 MB), verifies its SHA-256, and reads the dialogue JSON directly from
`data/dailydialog/raw/data.zip`. Every utterance becomes a JSONL record with
`body`, `source`, `source_id`, `source_split`, `dialogue_id`, and `utterance_index`.
The trainers already read `body` by default; no conversion is needed. Text is
preserved exactly as supplied by ConvLab, including repeated utterances.

All **102,979 utterances** are included: 87,170 train, 8,069 validation, and 7,740
test. Stable source IDs prevent duplicate appends on reruns. A changed body for an
existing ID aborts the import. Existing corpus bytes are preserved, with a final
newline added if needed before the new records. Before replacing an existing
corpus, the script creates a sibling `*.before-dailydialog-<id>.bak` backup. It
stages the complete result before replacing the corpus atomically.

The default destination is `data/combined-messages.jsonl`. `--cache-dir` changes
where the archive and provenance metadata are stored; `--report` overrides the
default `data/dailydialog/append-report.json` import report. No extra packages or
remote dataset code are required. Downloads, imported data, reports, and backups
remain under Git-ignored `data/` by default.

For combined-source neural training, use a fresh output directory and a random
split so the file's source ordering does not determine the held-out sets:

```bash
.venv-cuda/bin/python train_nn_cuda.py data/combined-messages.jsonl \
  --model-kib 1024 --context 3 --split random \
  --output models/nn-combined-dailydialog-1024k
```

The trainer does not use `source_split`: importing all three DailyDialog splits
means its original test/validation sets are no longer untouched evaluation data.
Random splitting also allows related turns from one dialogue into different
splits; use an independently held-out conversation corpus for final evaluation.

DailyDialog is by Yanran Li et al. (IJCNLP 2017), with this version transformed by
Qi Zhu for ConvLab. The dataset card specifies **CC BY-NC-SA 4.0** (attribution,
noncommercial, share-alike). Source, immutable revision, license link, and paper
are recorded in `data/dailydialog/source.json`.

## Train on a desktop

Training requires NumPy; it does not use PyTorch or download a model:

```bash
# Use the project's existing .venv, or make a lightweight training environment:
python -m venv .venv-mcu
source .venv-mcu/bin/activate
python -m pip install numpy==2.1.3

python train_compression_model.py data/messages.jsonl \
  --model-kib 256 --order 4 --output models/my-256k
# Tiny byte-frequency baseline, only 1,088 model bytes:
python train_compression_model.py data/messages.jsonl \
  --model-kib 2 --order 0 --strengths 1 --output models/my-unigram
```

`--model-kib` is a maximum for the entire exported binary, including its header;
it is not a RAM claim. Budgets of at least 2 KiB are accepted when the selected
order fits, with no software upper limit. `--order` accepts any finite nonnegative
integer; there is no fixed maximum depth. `--order 0` uses only a byte-frequency
model and occupies 1,088 bytes regardless of a larger budget. Positive orders add
preceding-byte contexts; order 4 is the default. Bigger tables reduce hash collisions, but more context is not
automatically better at every budget. Compare validation loss when choosing a
size/order, and keep a fresh test set for a final comparison after extensive tuning.

For example, request a 128 MiB model with 64 preceding bytes of context:

```bash
python train_compression_model.py data/messages.jsonl \
  --model-kib 131072 --order 64 --workers 4 --output models/my-128m-order64
```

Available RAM, disk space, and the runtime's address space still limit feasible
settings. Training uses dense integer histograms and temporary arrays, consuming
several times the exported model size, with additional tables per counting worker
and queued result. Reduce `--workers` and `--batch-bytes` for large models.
Each context level needs at least one bucket; an insufficient budget reports the
minimum required size. Counting, validation, and inference work per byte grow
with context depth. Validation streams context hashes one level at a time instead
of retaining an array for every level. The hashes remain 32-bit, so more than
`2**32` slots per subtable cannot add reachable buckets.

Useful options:

| Option | Effect |
| --- | --- |
| `--text-field body` | Default: train message content without known sender envelopes. `text` uses original stored text. |
| `--split chronological` | Default: order unique messages by their first occurrence in the input. |
| `--split random --seed 42` | Reproducible grouping and shuffling of unique messages. |
| `--keep-duplicates` | Weight repeated texts by occurrence count, within their assigned split only. |
| `--validation-fraction 0.1 --test-fraction 0.1` | Default 80/10/10 split by unique text groups. |
| `--strengths 4,16,64,256,1024,4096` | Default smoothing candidates; choose using validation coding loss. |
| `--limit-messages 10000` | Faster development run on the first input rows. |
| `--max-message-bytes 8192` | Skip longer messages, reporting their count; `0` disables the length filter. Empty messages are always reported/skipped. |
| `--batch-bytes 1048576` | Vectorization target; messages are never split between batches. |
| `--workers 4` | Parallel counting threads; defaults to up to four available CPUs. Use `1` for serial counting. |
| `--train-bytes 67108864` | Optional development cap on training bytes; whole messages only. Default trains on the entire training split. |
| `--evaluation-messages 100000` | Uniform reservoir sample from each held-out split; `0` evaluates the complete splits. |
| `--benchmark-messages 1000` | Exact arithmetic round trips on a seeded sample of the evaluated test messages; `0` benchmarks all evaluated test messages. |

For JSONL/CSV input, training first groups identical UTF-8 bodies in desktop RAM.
By default it keeps one occurrence,
so retransmissions and repeated bot text cannot dominate the fit or cross splits.
Chronological splitting refers to **first occurrence**, not a strict timestamp
cutoff: later copies stay with the first occurrence even with `--keep-duplicates`.
Near duplicates, quotes, and related conversations can still span splits. There
is no case folding, punctuation removal, Unicode normalization, or vocabulary
restriction. Sender/channel metadata is not part of the training input.

Training estimates zero/one frequencies for each bit context, scales large pairs
into uint16 without changing their ratio substantially, and chooses a smoothing
strength using validation negative log likelihood. Loss uses the exact quantized
integer probabilities used by the decoder. There are no generation-oriented
objectives, generation sampling, or text-generation API, and no gradient epochs. Inference
never changes learned counts; context resets at each message. Selecting a large
finite `--order` does not carry history across messages or automatically expand
the context during inference.

Each output directory contains:

- `model.mcblm`: portable little-endian binary probability tables.
- `report.json`: corpus counts/hash, validation candidates, test coding loss,
  unigram baseline, actual coding sizes, and desktop timings.
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: exact selected message bodies
  for JSONL/CSV input. Prepared Wikipedia input is referenced in the report,
  without copying the large corpus into each model directory.
- `test-vectors.json`: artificial public inputs, exact payload bytes, CRC32, and
  initial integer probabilities for checking an independent decoder.

Existing nonempty output directories are refused. Same input/order/budget/settings
produce identical model bytes; report timings naturally vary. The input SHA-256
in the report identifies the corpus snapshot.

## Download and prepare the full English Wikipedia

The downloader uses the cleaned **2023-11-01 English Wikipedia snapshot**, not
all languages or revision histories. It contains 41 Parquet shards totaling
**11,630,929,031 bytes**. The source is
[Wikimedia's Hugging Face dataset](https://huggingface.co/datasets/wikimedia/wikipedia),
pinned to commit `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. The full English file
list is [here](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en).
This snapshot already strips Wikipedia markup and unwanted sections. It remains
encyclopedia text, with a different distribution from short radio conversations.

```bash
# In the existing project environment; only preparation needs PyArrow.
python -m pip install pyarrow==23.0.1

# Downloads every English shard, with four concurrent connections.
python prepare_wikipedia.py download --workers 4

# Produces bounded-size samples and train/validation/test files.
python prepare_wikipedia.py prepare --workers 2 --sample-bytes 160
```

Paths default to `data/wikipedia/raw/` and `data/wikipedia/prepared/`. Run the same
commands to resume. Downloads use partial files, HTTP ranges, expected sizes, and
the repository's LFS SHA-256 checksums. Complete raw files are renamed into place
only after verification. Preparation checkpoints each shard after all its split
files are complete. Changing the preparation configuration requires a new output
directory. Run one downloader and one preparer per destination at a time.

To overlap downloading and preparation, start the download in one terminal and
run this in a second terminal:

```bash
python prepare_wikipedia.py prepare --wait-for-download --workers 2
```

The preparer waits for verified complete raw files. The final `manifest.json`
is written only after **all 41 shards** are processed. The raw Parquet files are
retained, including article IDs, titles, URLs, and original cleaned text. The
dataset card/license information is retained in `raw/README.source.md`; its
source URL and revision are recorded in the manifests.

Preparation assigns an entire article to a split by a seeded SHA-256 hash of
its complete UTF-8 text, **before chunking**. Identical full articles necessarily
share a split even if their IDs differ. The default proportions are approximately
80/10/10, not exact row counts. Distinct articles can still contain identical or
related passages; this is article-level isolation, not global passage deduplication.
All nonempty articles contribute their text; empty articles are counted.

Chunks contain at most 160 UTF-8 bytes by default. Boundaries prefer a space or
newline in the latter half of a chunk, and never split a Unicode character.
Whitespace and punctuation are preserved. Concatenating an article's chunks
reconstructs its exact source text bytes. No large article is silently discarded
by the trainer's default 8,192-byte message limit.

To train on longer contexts, prepare larger chunks using `--sample-bytes N`, or
use `--sample-bytes 0` to keep whole articles. There is no software upper limit
on chunk size. Use a fresh output directory when changing this setting:

```bash
python prepare_wikipedia.py prepare --workers 2 --sample-bytes 0 \
  --output data/wikipedia/articles
python train_compression_model.py data/wikipedia/articles \
  --model-kib 131072 --order 64 --max-message-bytes 0 \
  --workers 4 --output models/wikipedia-128m-order64
```

The existing prepared corpus still contains 160-byte samples. Increasing
`--order` alone does not join those samples. Whole-article training requires
`--max-message-bytes 0`; batches may exceed `--batch-bytes` to hold a whole article.
Evaluation's default 100,000-message reservoir can also use substantial RAM with
whole articles: reduce `--evaluation-messages`, or use `0` to score directly from
disk. Reduce `--benchmark-messages` when testing slow, deep-context models.

Prepared files are compact `.mccorpus` records: magic `MCBCORP1`, followed by
repeated little-endian uint32 byte lengths and UTF-8 message bytes. Whole-article
preparation (or a chunk limit above `2**32-1`) uses `MCBCORP2`, with canonical
unsigned LEB128 lengths instead. The reader supports both formats and has no
default record-size ceiling; records must fit the available memory. They need no
PyArrow at training time. Each source shard yields one file per split under
`prepared/parts/`; the manifest records sizes, message/article/character counts,
and SHA-256 for every file. `training_corpus.read_records()` exposes the records
as a Python iterator. Keep the manifest and its `parts/` directory together.

The prepared text occupies additional disk space beyond the 11.63 GB download;
it is not compressed. Memory use depends on the Arrow batch, worker count, and
largest article/page buffers, rather than the entire corpus. Preparation reads
only the text column using
[PyArrow record batches](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html).
`--arrow-batch-articles 128` is the default; reduce workers/batch size if needed.

The completed local corpus contains **6,407,814 articles** and
**19,567,594,259 UTF-8 text bytes**, represented as **127,959,814 samples**.
The prepared files occupy 20,079,434,499 bytes including record lengths/headers;
raw plus prepared files occupy about **31.7 GB**. No articles were empty.

| Split | Articles | Samples | UTF-8 bytes |
| --- | ---: | ---: | ---: |
| Train | 5,126,212 | 102,355,386 | 15,652,147,910 |
| Validation | 641,030 | 12,782,925 | 1,954,697,833 |
| Test | 640,572 | 12,821,503 | 1,960,748,516 |

## Train from the prepared Wikipedia corpus

```bash
# Full training split, 256 KiB maximum model, no source-corpus RAM copy.
python train_compression_model.py data/wikipedia/prepared \
  --model-kib 256 --order 4 --workers 4 --output models/my-wikipedia-256k

# Optional smaller development run on 64 MiB of training text.
python train_compression_model.py data/wikipedia/prepared \
  --train-bytes 67108864 --evaluation-messages 10000 \
  --model-kib 64 --output models/wikipedia-development

# Optional offline integrity audit after copying raw/prepared files.
python prepare_wikipedia.py verify data/wikipedia/raw
python prepare_wikipedia.py verify data/wikipedia/prepared
```

Input can be the prepared directory or its `manifest.json`. The article splits
are used unchanged; split fractions and grouping options for JSONL/CSV are not
applied. `--keep-duplicates` and `--limit-messages` are refused for prepared input;
use `--train-bytes` for development. `--seed` controls evaluation sampling; the
preparation seed and article split remain recorded in the corpus manifest.

Training streams approximately 1 MiB of message bytes per batch. Worker threads
count independent batches into private uint64 next-byte histograms; the main
thread merges those integer counts. There are no shared writable worker arrays.
The combined histograms are projected into bit-prefix counts and quantized once,
after all training batches finish. Changing the worker count preserves model
bytes, including when large counts require rescaling.

`--workers` defaults to the smaller of four and the available CPU count, respecting
process CPU affinity where supported. Use `--workers 8` to request eight threads,
or `--workers 1` to disable parallel counting. The queue holds at most twice the
worker count in submitted batches, including completed results. It never submits
the entire corpus at once. RAM use grows with worker count, batch size, largest
message, and model size, rather than corpus size. The NumPy integer-array work
can execute concurrently in these threads
([NumPy thread safety](https://numpy.org/doc/2.1/reference/thread_safety.html)).

Input reading/grouping, evaluation sampling, validation scoring, and arithmetic
round-trip benchmarks remain sequential. The `parallelism` section of new
`report.json` files records worker count, batch size, queue limit, and this scope.

A local counting benchmark on the same first 256 MiB of prepared Wikipedia,
using order 4, a 256 KiB model budget, and 1 MiB batches, measured:

| Threads | Counting seconds | Peak process RSS |
| ---: | ---: | ---: |
| 1 | 10.21 | 93 MiB |
| 2 | 6.49 | 152 MiB |
| 4 | 4.73 | 260 MiB |
| 8 | 4.39 | 470 MiB |

All four runs produced identical quantized tables and training statistics. Four
threads were about 2.2 times faster in this benchmark; eight offered a smaller
additional gain while using more RAM. These are single local measurements of
counting, not promises for other hardware or the entire training/evaluation
command. Details are in `models/threading-benchmark.json`.

By default all training messages are counted. Validation and test each use a
deterministic, uniform reservoir sample of up to 100,000 messages, selected by
scanning the respective complete split once. That retains about 16 MB per split
at a 160-byte maximum, plus Python object overhead. `--evaluation-messages 0`
keeps evaluation on disk and evaluates all held-out messages; it takes longer.
The default actual arithmetic benchmark round-trips 1,000 evaluated test messages.
Reports distinguish full source split sizes, training bytes actually used, and
evaluation sample sizes. There is no default training byte cap.

Compare the trained model on held-out MeshCore messages before choosing it for
radio use. More encyclopedia data may improve general English estimates without
improving callsigns, abbreviations, or short-message compression. MCU binary model
size and runtime format remain controlled by `--model-kib` and `--order`.

### Completed full-corpus training run

`models/wikipedia-256k/model.mcblm` was trained on **every one of the 102,355,386
training records**, totaling **15,652,147,910 bytes**, without a development cap.
The order-4 model is **258,128 bytes** and selected smoothing strength 4096.
Validation and test each sampled 100,000 messages uniformly from their complete
held-out splits. Test loss is **4.696 bits per UTF-8 byte**, compared with 4.800
for its unigram baseline.

This recorded run used the original serial counter, before the `--workers` option
was added. The full training/evaluation command took about **597 seconds**, with
**233 MiB peak process RSS** on this Linux desktop. These are training measurements, not
MCU runtime memory or speed claims. `runtime.json` records the measurement;
`report.json` records the corpus provenance, exact training counts, evaluation
sample sizes, and coding results. All 123 prepared files were independently
SHA-256 verified after writing, and Parquet row counts matched every shard's
preparation report.

An independent 1,000-message Wikipedia benchmark encoded 153,376 original bytes
into **90,427 arithmetic payload bytes (58.96%)**. Estimated framing raises this
to 62.85%; it still excludes the shared model and radio overhead. All 1,000
round trips matched exactly. Two samples exceeded the benchmark's estimated
163-byte datagram limit, so sample length does not guarantee packet fit.

On the existing, separate MeshCore test set, this Wikipedia-trained model was
**worse** than the same-sized model trained on radio messages:

| Training source | MeshCore test bits / UTF-8 byte | MeshCore arithmetic payload / original |
| --- | ---: | ---: |
| MeshCore messages | 5.090 | 64.54% |
| Full English Wikipedia | 6.013 | 75.94% |

Both payload comparisons use the same 1,000 held-out MeshCore messages (51,404
original bytes), and both decode exactly. The Wikipedia model's loss covers all
5,305 held-out MeshCore messages. These messages were used only for this
comparison, not to fit or select the Wikipedia model. Details are in
`models/wikipedia-256k/meshcore-evaluation.json`. Retain the MeshCore-trained model
as the stronger measured choice for this traffic; corpus size alone did not
improve compression in this comparison.

```bash
python mcu_compress.py info --model models/wikipedia-256k/model.mcblm
```

## Results on this database

The local models below use 42,441 distinct training messages, 5,305 validation
messages, and 5,305 test messages. The three context models use order 4 and selected
smoothing strength 4096 on validation data; the unigram uses order 0. Test loss
covers all 5,305 held-out messages. Actual byte
sizes cover the same seeded sample of 1,000 messages (51,404 original bytes), each
encoded and decoded independently with a fresh context.

| Local output | Model bytes | Test bits / UTF-8 byte | Arithmetic payload / original | Estimated framed / original |
| --- | ---: | ---: | ---: | ---: |
| `models/mcu-unigram/` | 1,088 | 5.503 | 69.60% | 79.34% |
| `models/mcu-64k/` | 62,288 | 5.360 | 67.89% | 77.63% |
| `models/mcu-256k/` | 258,128 | 5.090 | 64.54% | 74.27% |
| `models/mcu-1024k/` | 1,045,568 | 4.629 | 58.78% | 68.51% |

All 4,000 benchmark round trips matched exactly. Independent gzip and raw DEFLATE on the 1,000-message
sample total 67,903 and 49,903 bytes respectively (132.10% and 97.08% of original).
These comparisons do not charge any codec for its implementation or pre-shared
tables, and do not compare against a continuous stream with shared DEFLATE state.
Short or unusual messages can expand; these are aggregate ratios, not a guarantee
for individual messages or another channel.

The framed column is explicitly a **size estimate**: two nonce bytes, a ULEB128
original byte count, two CRC16 bytes, plus the arithmetic stream. It excludes
MeshCore/radio headers, encryption overhead, region data, and the shared model.
This new codec is **not wired into the current MeshCore GUI/CLI or its Hugging Face
wire profile**. Integrating it requires an agreed new profile and packet format
on both ends. The benchmark does not transmit radio messages.

## Try the trained model

Inference and file coding use only the Python standard library:

```bash
python mcu_compress.py info --model models/mcu-256k/model.mcblm
python mcu_compress.py compress examples/english.txt /tmp/english.mcb \
  --model models/mcu-256k/model.mcblm
python mcu_compress.py decompress /tmp/english.mcb /tmp/english-restored.txt \
  --model models/mcu-256k/model.mcblm
cmp examples/english.txt /tmp/english-restored.txt
```

The file wrapper has a 52-byte header: magic `MCBFILE1`, model SHA-256 (32 bytes),
original byte count (uint32), payload byte count (uint32), and original CRC32
(uint32), all integers little-endian. It checks the model, output-size bound,
canonical arithmetic stream, and reconstructed CRC before writing output. The
default decompression bound is 64 MiB (`--max-output-bytes` can change it).

`compress_payload()` and `decompress_payload()` in `bit_codec.py` expose the bare
stream. They need a pre-shared model and the exact original **byte** count; no end
marker is encoded. The bare stream has no content checksum. A transport must
provide integrity checking and original length, and agree on the model/profile
out of band. The model, vocabulary, training settings, and full model fingerprint
do not need to accompany every radio message.

## MCU memory and porting

The 1,088-byte unigram is a useful first port: its compression is close to the
64 KiB model on this corpus and it needs only eight table lookups per input byte.
The 64 KiB / 256 KiB options provide context prediction for an STM32F4-class native
implementation. Place tables in readable flash/external memory if supported by
the particular part and board; they never need to be modified or copied entirely
into SRAM. For example, the STM32F405/407 family has up to 1 MiB flash and
192 + 4 KiB SRAM, so firmware and radio buffers must be budgeted alongside the
model ([ST datasheet](https://www.st.com/resource/en/datasheet/stm32f405rg.pdf)).

An ESP32-S3 board with PSRAM offers room to test larger tables; support and usable
memory depend on the module, mapping, and ESP-IDF configuration
([Espressif external RAM guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-guides/external-ram.html)).
A 1 MiB table does not imply that a device has 1 MiB of free internal RAM.

The native algorithm needs only the last `order` byte values (plus BOS sentinels),
context buckets, bit prefix, and arithmetic coder state. `info` gives an
**algorithm-state allowance of `max(1024, 128 + 10 * order)` bytes**, not a measured
allocation. This allows two bytes per history value (including BOS), four per
hash, four per bucket, and a fixed allowance. It excludes stacks, code,
message/radio buffers, external-memory cache, filesystem, and Python. The desktop
implementation uses `mmap`; that mechanism and CPython's object sizes do not
describe MCU memory use. A firmware port should access the table directly and
stream the arithmetic input/output. Training needs desktop RAM, not MCU RAM.

With order 4 each input byte requires 40 count-pair lookups plus four context hash
updates. Integer division and random table reads can dominate latency on an MCU,
especially over external SPI memory. This repository has no hardware speed, power,
or peak-memory measurements. Port and measure before selecting a production size.

### Binary model formats (scheme 2)

Existing models use a 64-byte header, equivalent to Python
`struct.Struct('<8s6I32s')`:

| Offset | Field |
| ---: | --- |
| 0 | Eight bytes `MCBITLM1` |
| 8 | Context order, uint32 |
| 12 | Slots per bit prefix per context order, uint32 |
| 16 | Probability total, uint32, always 32768 |
| 20 | Smoothing strength, uint32, 1–4096 |
| 24 | Scheme, uint32, currently 2 |
| 28 | Table body length in bytes, uint32 |
| 32 | SHA-256 of header bytes 0–31 followed by the table body |
| 64 | Count pairs, two little-endian uint16 values per row: zero count, one count |

The writer preserves this format and exact model bytes when order is at most 8
and all six integer fields fit uint32. Other settings use **`MCBITLM2`**: eight
magic bytes, then six canonical unsigned LEB128 integers in the same order
(context order, slots, probability total, smoothing strength, scheme, body length),
then a 32-byte SHA-256 digest and the unchanged count-pair body. The digest covers
the encoded prefix before the digest followed by the body. These variable-length
integers remove the old 4 GiB body-length and fixed-width dimension limits.
The Python reader supports both formats; a firmware port must implement the
extended header to load new-format models. Use `model.header_size` to locate the
body rather than assuming 64 bytes.

For header size `H`, order `O`, and slots `S`, model size is
`H + 4 * (256 + O * 255 * S)` bytes.
Rows 1–255 are the unconditional bit-prefix tree; row 0 is reserved. Each context
order has a separate `255 * S` row table. Prefixes have separate subtables to
prevent collisions between different positions/branches of a byte. Previous-byte
contexts within a subtable share hashed buckets. No raw message dictionary is
stored.

### Integer prediction and coding contract

Follow `bit_model.py` and `arithmetic.py` exactly, then compare `test-vectors.json`:

1. Reset each message's byte history to `O` copies of BOS=256. Keep the newest
   observed byte first. Before a byte, start `h=2166136261`; for each history value,
   compute `h=((h XOR value)*16777619) mod 2**32`. The successive `h` values are the
   contexts of orders 1 through O.
2. Mix each context as in `mix32`: XOR right-shift 16, multiply `0x7feb352d`;
   XOR right-shift 15, multiply `0x846ca68b`; XOR right-shift 16. Each multiplication
   wraps to uint32. The bucket is the result modulo S.
3. Process each byte most-significant bit first, starting prefix `p=1`. Fetch root
   counts `(z,n)` at row p. Start `q=((n+1)*32768)//(z+n+2)`.
4. For each level `l=0..O-1`, fetch counts at
   `row=256+(l*255+p-1)*S+bucket[l]`. Set `d=z+n+strength` and
   `q=(n*32768+strength*q+d//2)//d`. Clamp q to `[1,32767]` **after all levels**.
5. Code bit 0/1 using cumulative frequencies `(0,32768-q,32768)`. Update
   `p=2*p+bit`. After eight bits, prepend the recovered byte to history and drop
   the oldest value. Recompute buckets for the next byte. Never adapt the counts.
6. Use the existing inclusive 32-bit arithmetic interval, E1/E2/E3 renormalization,
   termination, MSB-first bit packing, and zero padding exactly. Width can equal
   `2**32`; interval products/divisions need uint64 intermediates, including
   decoder initialization and symbol selection. Unsigned 32-bit intermediates
   suffice for the probability formula with the specified limits. Include the
   model/header digest checks when installing a shared model, and length/content
   checks in the chosen packet format.

Large tables can require 64-bit row indices and byte offsets even though hashes
and arithmetic intervals remain 32-bit. Deeper contexts also increase runtime
state and table lookups; removing the configuration limits does not make a large
model fit an MCU's memory or execution-time budget.

This uses bitwise statistical context prediction, related to work such as
[Fenwick, *PPM compression without escapes* (2011)](https://www.cs.auckland.ac.nz/~peter-f/FTPfiles/2011%20PPM%20noESC.pdf),
with fixed offline-trained hashed tables and hierarchical smoothing rather than
an adaptive PPM implementation. The shared prediction-to-arithmetic-coding idea
is the same as in the larger Hugging Face demo.

## Sweep model size and context depth

Install the plotting dependency, then run a grid using the same corpus split,
seed, evaluation samples, and training settings for every configuration:

```bash
python -m pip install -e ".[sweep]"
python sweep_compression_models.py data/messages.jsonl \
  --sizes-kib 16,64,256,1024 --orders 0,1,2,4,8 \
  --output models/mcu-sweep
```

This trains 20 models sequentially, with the trainer's usual worker threads per
model. All ordinary corpus, smoothing, sampling, and worker options are supported;
use `--sizes-kib` and `--orders` in place of the single-run `--model-kib` and
`--order`. Budgets are KiB; context orders count preceding UTF-8 bytes. Infeasible
budget/order pairs are skipped and recorded. Order zero is the same 1,088-byte
unigram model at every budget, so its size plot has one point.

For a quick exploratory run on the prepared Wikipedia corpus:

```bash
python sweep_compression_models.py data/wikipedia/prepared \
  --sizes-kib 16,64,256 --orders 0,1,2,4,8 \
  --train-bytes 1048576 --evaluation-messages 1000 --benchmark-messages 100 \
  --output models/wikipedia-sweep-quick
```

Each grid cell keeps the usual model, report, and test vectors in a directory
such as `256k-order4/`. The sweep also writes:

- `summary.csv` and `summary.json`: actual model size, chosen smoothing,
  validation/test bits per UTF-8 byte, payload/framed percentages, and training time.
- `compression.png` and `compression.svg`: four plots showing loss against actual
  model size and context depth, plus actual payload and estimated framed sizes.
- `sweep.json`: settings and corpus hash used to validate resumption.

Lower values mean better compression. Payload percentages use exact arithmetic
round trips on the configured benchmark sample; loss uses the held-out evaluation
sample. Framing uses the trainer's nonce/length/CRC estimate. Sizes exclude the
shared model and radio overhead. Training times describe the desktop reference,
not MCU performance. Use validation loss to select parameters; after exploring
many test curves, evaluate the chosen configuration on fresh traffic.

Repeat the same command with `--resume` to reuse completed, verified models.
An interrupted cell is moved to an `.incomplete-*` directory and retrained; its
existing files are preserved. Changing the corpus or settings requires a new
output directory. To redraw existing results without training or reading a corpus:

```bash
python sweep_compression_models.py --output models/mcu-sweep --plot-only
```

The installed equivalent command is `mcu-sweep`.

## Train neural models for comparison

`train_nn_compression_model.py` trains a byte-level neural network with an exact,
finite context window. It embeds each of the preceding bytes (plus BOS=256),
flattens those position-specific embeddings, applies three equal-width tanh hidden
layers, and predicts a 256-way next-byte distribution. It has no recurrent state
or tokenizer. The largest hidden width that fits `--model-kib` is chosen using
integer arithmetic, counting the whole portable binary: header, embeddings,
all four dense matrices, biases, and per-tensor quantization scales. The middle
matrices grow quadratically with width; all hidden layers are narrowed to stay
within the same budget. Context zero remains an unconditional neural baseline.

Three hidden layers are the default for all neural training entry points and
CUDA sweeps. Use `--hidden-layers 1` or `--hidden-layers 2` to reproduce an earlier
architecture. New three-layer files use `MCNNLM03`; the loader still reads
existing `MCNNLM01` and `MCNNLM02` files without changing their predictions.
Reports record `hidden_layers` and the per-layer `hidden_dim`. Select a fresh
output directory for the new architecture; resume an existing sweep with its
original options plus the matching `--hidden-layers 1` or `--hidden-layers 2`.

The Pico exporter and the Python/C integer runtimes also support all three
depths: `MCPNN001`, `MCPNN002`, and `MCPNN003` carry the respective matrices,
biases, and fixed-point scaling. Two- and three-layer inference alternate two
activation buffers; layer three reuses layer one's RAM. Both peers need the
same model and a runtime that supports its format. The training budget constrains `.mcnn` size;
the MCU `.pnn` conversion additionally stores lookup tables and wider biases,
so its file size is larger. Firmware RAM/flash limits are checked separately.
Changing depth does not establish better compression; compare held-out loss at
the same model budget.

Use Python with NumPy and PyTorch installed. On this Windows machine, `python`
resolves to Inkscape's bundled runtime; use the separate Python 3.11 executable:

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $py -m pip install numpy==2.1.3 torch==2.5.1

# Radio-message model: same default grouping/split/seed as the statistical trainer.
& $py train_nn_compression_model.py data/messages.jsonl --model-kib 256 --context-bytes 4 --epochs 10 --output models/nn-mesh-256k-c4

# Wikipedia model: consumes the existing prepared splits without loading all text.
& $py train_nn_compression_model.py data/wikipedia/prepared --model-kib 256 --context-bytes 4 --epochs 10 --output models/nn-wikipedia-256k-c4
```

On Linux use `python` in your project virtual environment in place of `& $py`.
For an initial Wikipedia trial, add `--train-bytes 1048576
--evaluation-messages 1000 --benchmark-messages 100`. Remove the training cap for
a full-corpus run: neural training makes an entire pass per epoch and can take
substantially longer than counting statistical tables. Evaluation samples scan
all held-out records once; `--evaluation-messages 0` scores the full splits.

Important controls:

- `--context-bytes N` (alias `--order`): preceding UTF-8 bytes, reset per record.
- `--model-kib N`: maximum complete exported file size in KiB, not a RAM limit.
- `--hidden-layers {1,2,3}`: number of equal-width tanh hidden layers; default 3.
- `--storage int8` (default) or `float32`: export representation. Training uses
  float32; int8 uses one scale per tensor and validation selects epochs after
  quantization. Embedding dimension defaults to 8; use `--embedding-dim` to vary it.
- `--epochs`, `--learning-rate`, `--batch-size`, and `--threads`: training controls.
- `--early-stopping-patience N` (alias `--patience`): stop after N validation
  checks without improvement; enabled by default with 3 checks. Use `0` to disable.
- `--early-stopping-min-delta D`: minimum significant validation loss decrease in
  bits/byte; default `0` accepts any strict decrease. The best checkpoint still
  includes smaller improvements.
- `--device cuda`: require GPU training; the ordinary trainer's default `auto`
  selects CUDA when available. The ordinary trainer defaults to CPU validation;
  `train_nn_cuda.py` and `sweep_nn_cuda.py` default to FP32 CUDA validation and
  skip the final CPU reference validation pass.
- `--skip-cpu-validation` / `--no-skip-cpu-validation`: omit/restore the final CPU
  validation pass after GPU selection. Skipping requires CUDA validation.
- `--test-device auto|cpu|cuda`: test scoring; `auto` follows the validation device.
  CPU arithmetic round-trip checks still run on the benchmark sample.
- `--cuda-chunk-bytes N`: compact input chunk size (default 1 MiB), aligned to
  whole optimizer batches. Context tensors are assembled on the GPU per batch.
- `--validate-every N`: validate and export every N epochs (default 1), always
  including the final epoch. Only validated epochs compete for the best model.
- `--train-bytes`: cap whole training messages per epoch; the same prefix is used
  each epoch. Batches shuffle their byte examples; message order is otherwise fixed.
- `--evaluation-messages` (default 1000), `--benchmark-messages` (default 100),
  `--seed`, `--split`, `--text-field`, and split fractions: match these to the
  statistical runs for a fair comparison. JSONL/CSV also supports `--limit-messages`
  and `--keep-duplicates`, with the same grouping logic as the statistical trainer.

Outputs are `model.mcnn`, `report.json`, `epochs.json`, and public byte-level
`test-vectors.json`. The selected checkpoint minimizes validation bits per byte
among the epochs that are checked, using exported weights and the codec's integer
frequency allocation. Optional GPU selection can differ slightly from CPU
reference scoring, as described below. Test
loss is evaluated only after epoch selection. Reports include actual model bytes,
parameter count, architecture dimensions, corpus identity, per-epoch validation,
test bits per UTF-8 byte, and real arithmetic payload/framing percentages. Every
benchmark message is compressed, decoded, and checked for byte-for-byte equality.
The benchmark uses the same sampling and framing convention as the statistical
trainer; match `--benchmark-messages` explicitly (its statistical default is 1000).

To compare a grid in PowerShell, choose a fresh output prefix for each experiment:

```powershell
foreach ($size in 16,64,256,1024) {
    foreach ($context in 1,2,4,8,16) {
        & $py train_nn_compression_model.py data/messages.jsonl --model-kib $size --context-bytes $context --epochs 10 --output "models/nn-mesh-${size}k-c${context}"
        if ($LASTEXITCODE -ne 0) { throw "Neural training failed" }
    }
}
```

The `.mcnn` format differs from `.mcblm`; the existing `mcu_compress.py` command
cannot load it. `llm_compression.nn_model` provides `NeuralModel`,
`compress_payload`, and `decompress_payload` for reference use. Payload decoding
needs the original byte length. This is an offline training and inference
reference, not native MCU firmware. Stored int8 weights are expanded to float64
for CPU inference; training/optimizer memory and inference RAM exceed the stored
file size. Cross-platform floating-point probability agreement is not guaranteed;
use matching runtimes and the test vectors when developing a native decoder.

### Generate a continuation

Use a trained `.mcnn` model for greedy generation or temperature sampling:

```bash
.venv/bin/python generate_nn.py models/my-neural-model/model.mcnn \
  "Hello from the mesh" --bytes 200

# Sample alternatives to the most likely byte; the seed makes the run repeatable.
.venv/bin/python generate_nn.py models/my-neural-model/model.mcnn \
  "Hello from the mesh" --bytes 200 --temperature 0.8 --seed 42

# Save the prompt plus 200 new bytes exactly, without UTF-8 display replacement.
.venv/bin/python generate_nn.py models/my-neural-model/model.mcnn \
  "Hello from the mesh" --bytes 200 --output generated.txt
```

The initial text context is a quoted positional argument. Omit it (or pass an
empty string) to start with BOS padding. `--bytes` counts newly generated bytes,
excluding the initial context; its default is 256. `--continuation-only` omits the
prompt from the output while still using it to initialize the model's history.
The model uses only its configured number of preceding UTF-8 bytes and fills
missing context with BOS, matching compression. Longer prompts are printed in
full but only their final context window affects predictions.

`--temperature 0` (the default) uses the maximum raw logit before codec
probability quantization, with the lower byte value breaking exact ties.
Positive temperatures sample from `softmax(logits / temperature)`: values below
1 concentrate probability on likely bytes, 1 uses the model's distribution,
and values above 1 spread probability more evenly. Sampling allows alternatives
to the greedy choice, helping escape deterministic loops; repetitions can still
occur. `--seed N` reproduces sampling choices for the same model, context,
temperature, and runtime. Omit it for a fresh random seed. Temperature must be
finite and nonnegative, and seeds must be nonnegative integers.

Each chosen byte is fed back into the context. There is no repetition penalty,
UTF-8 filter, or end-of-text token; it generates the requested number of new
bytes. All one-, two-, and three-hidden-layer `.mcnn` files are supported, with
either int8 or float32 storage; inference needs only NumPy on CPU.

Terminal output is streamed with incremental UTF-8 decoding, so multibyte
characters are displayed correctly. Invalid sequences or an incomplete final
character appear as replacement characters for display only. `--output PATH`
preserves the exact bytes, refuses to overwrite existing files, and sends no
text to stdout. No newline is appended automatically. The installed equivalent
is `nn-generate`; `python -m llm_compression.nn_generation` also works.

### Visualize neural activations

The basic desktop viewer loads trained `.mcnn` models with one, two, or three
hidden layers. It uses NumPy and Tkinter on CPU; PyTorch and CUDA are not needed.
The exported int8 weights are dequantized using the same reference loader as the
compression codec. This viewer inspects `.mcnn` files, not the MCU `.pnn` format
or Hugging Face checkpoints.

```bash
# Open the GUI and choose a trained model.
.venv/bin/python visualize_nn.py

# Open a particular model and start partway through a message.
.venv/bin/python visualize_nn.py models/my-neural-model/model.mcnn \
  --text "Hello from the mesh!" --position 5
```

The installed command is `nn-visualize` (reinstall the editable package to add
new entry points). `python -m llm_compression.nn_visualizer` also works. A desktop
display is required. If Tkinter is missing on Debian/Ubuntu, install `python3-tk`.

- Browse/Load selects the model; the header shows its size and architecture.
- Editing the message updates the view automatically. Previous/Next, the byte
  slider, and the position box choose the next byte to predict. Position zero
  uses BOS padding; the final position predicts beyond the input. Play animates
  the byte positions, with a configurable delay. UTF-8 characters can span
  several steps: the predictor operates on bytes rather than characters.
- Blue is a negative activation, amber is positive, and dark is near zero.
  Hidden tanh layers always use the fixed -1 to +1 scale. Embedding colors are
  normalized to their largest absolute value in each frame; output green is
  normalized to the largest next-byte probability in that frame.
- Hover over a tile for its exact activation and pre-tanh value, embedding
  coordinate, or byte probability. Large layers group consecutive units up to
  the selected tile limit; a group shows its strongest absolute activation,
  with its sign retained. Hover reveals the range, peak unit, mean, and extrema.
  Grouping affects only drawing; every neuron is evaluated.
- The sidebar lists the most likely next bytes and highlights the actual next
  byte. Its probabilities and bit cost use the codec's integer frequency table;
  output hover also shows the original softmax probability and logit. Predictions
  use preceding input only; the actual next byte is used only for comparison.
- The bold cost line sums the coding cost of the message prefix, including the
  selected byte, and shows bits and equivalent fractional bytes. Position zero
  includes the first byte; the final position shows the complete message cost.
  The selected byte's individual cost is still shown above it. The cumulative
  cost is the sum of `-log2(codec probability)` for the included bytes.
- "Payload if ended here" is the actual arithmetic payload length for that
  prefix, including termination and byte padding. It excludes framing and the
  shared model. Empty input has zero probability cost, but its terminated
  arithmetic payload still occupies one byte. Prefix totals are cached for
  playback and navigation and reset whenever the model or message changes.

The viewer does not save entered messages or change model files. Loading and
inference run in a worker thread so the window remains responsive. The network
arrows show layer order; they do not represent individual synaptic weights.

### Pico W MicroPython inference

The [Pico demo](pico/README.md) implements a separate integer inference format
using trained `.mcnn` weights, lookup-table activations, Viper kernels, and a
lossless arithmetic codec. A 32 KiB/context-3 source model was deployed and
benchmarked on the connected Pico W: about 27 ms per byte prediction and
37 original bytes/s encoding at 125 MHz. Its exported model plus tables occupies
50,212 flash bytes, with a measured 63,280-byte inference heap delta.
The Pico payloads agree exactly with its desktop integer reference, but are not
compatible with the original float64 `.mcnn` runtime.

### PY32F403 native inference

The [PY32F403 firmware](py32f403/README.md) runs the same integer model and
arithmetic payload format in native C on a PY32F403V1DT6. All model tensors and
lookup tables remain in flash; NN working state is 2,836 bytes. The complete
application reserves 14,888 bytes of its 64 KiB SRAM, including an 8 KiB stack
budget. At nominal 144 MHz it measured 3.20 ms per prediction and about 333
original bytes/s encoding/decoding. Payloads match the Pico and desktop integer
codec exactly. Python tools build, back up, program, verify, and benchmark the
device through ST-Link and the Pico W UART bridge on GPIO0/GPIO1.

The same target was also flashed with the 64 KiB/context-3 model: prediction
measured 5.47 ms/byte, encoding/decoding about 188 original bytes/s, and the same
30-message sample improved from 3.23 to 2.94 payload bits/byte. It reserves
15,120 RAM bytes including the 8 KiB stack budget, with 3,068 bytes of NN working
state. Firmware size is 90,672 flash bytes. The 64 KiB model is currently on the
device; [reproduction commands and comparison](py32f403/README.md#reproduce-the-64-kibcontext-3-run)
preserve the original 32 KiB build and measurements.

### CUDA training on Linux (this machine)

Use the dedicated `.venv-cuda` environment with Python 3.12.3, NumPy 2.1.3,
and the official PyTorch 2.5.1 CUDA 12.4 build. The existing `.venv` retains its
CPU build for the original compression demo and its runtime-sensitive archives.
The shared `.cache/cuda-runtime` contains a Windows Python 3.11 build; the
launcher ignores it on Linux. `install_nn_cuda.py` is a Windows-only helper.

To recreate the Linux training environment from the project directory:

```bash
.venv/bin/python -m venv .venv-cuda
.venv-cuda/bin/python -m pip install torch==2.5.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
.venv-cuda/bin/python -m pip install numpy==2.1.3 pytest==8.3.5

# Check both the installed build and access to the GPU.
.venv-cuda/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'

# Train on radio messages; choose a fresh output directory for each run.
.venv-cuda/bin/python train_nn_cuda.py data/messages.jsonl \
  --context-bytes 16 --model-kib 256 --epochs 10 \
  --output models/nn-linux-cuda-256k-c16

# Optional initial Wikipedia trial using the existing prepared corpus.
.venv-cuda/bin/python train_nn_cuda.py data/wikipedia/prepared \
  --context-bytes 16 --model-kib 256 --epochs 1 --train-bytes 1048576 \
  --evaluation-messages 100 --benchmark-messages 10 \
  --output models/nn-linux-wikipedia-trial
```

You can instead run `source .venv-cuda/bin/activate`, then use `python` for those
commands. Prepared `.mccorpus` input needs no PyArrow during training. This machine
has an NVIDIA GeForce RTX 3070 with 8 GiB VRAM and driver 580.178.04; the PyTorch
wheel supplies its CUDA runtime libraries. These scripts do not compile CUDA
code or require a separate CUDA toolkit installation. The environment and its
package cache are ignored by Git.

The CUDA launcher defaults to FP32 GPU training, exported-model validation, and
test scoring, skipping final CPU reference validation. Exact arithmetic round
trips still use a sampled CPU reference check. Reports record the actual training
device, GPU name, precision, CUDA version, and peak allocated GPU memory. The CUDA launcher requests CUDA explicitly and
reports an error if it is unavailable, rather than silently using CPU.

Setup verification passed all 15 neural-model and launcher tests, including
training on CUDA with both message and prepared-corpus input. A separate
one-epoch launcher run trained on 8,162 local message bytes, exported a
261,882-byte int8 model, and decoded all three benchmark messages exactly.
Its runtime report confirms CUDA 12.4 on the RTX 3070 and 73,441,792 peak allocated
GPU bytes. The model/report are in `models/nn-linux-cuda-smoke-6m78_mfq/`;
this short run verifies the environment, not final compression quality.
An installed-package snapshot is saved in `.cache/nn-cuda-installed.txt`.

### FP32 GPU training and scoring

The CUDA launcher and CUDA sweeps now use FP32 for gradient training, exported
model validation, test scoring, softmax, and metric reductions. No FP64 tensors
or arithmetic are used on CUDA. TF32 remains disabled. Exported model size,
int8/float32 storage options, hidden-layer count, and codec format are unchanged.

```bash
.venv-cuda/bin/python train_nn_cuda.py data/combined-messages.jsonl \
  --model-kib 1024 --context-bytes 3 --split random --epochs 30 \
  --output models/nn-combined-fp32-1024k-c3
```

CUDA batches upload compact bytes and per-byte message positions in bounded
chunks, then build preceding-byte contexts and shuffle batch rows on the GPU.
History survives chunk boundaries but resets at every message/record boundary.
Prepared Wikipedia input continues to stream; the whole corpus and all expanded
contexts need not fit in GPU memory. A fixed-size FP32 reduction buffer with
compensated block sums avoids the cumulative drift of a single FP32 loss scalar
on long training runs. CUDA shuffling is seeded, but need not reproduce model
hashes from older CPU-side shuffling. Batch size and training data order are
otherwise preserved. Increase `--batch-size` explicitly for more GPU work per
step, and compare validation quality because that changes optimization.

GPU scoring uses **exported, quantized/dequantized weights** and the codec's
integer frequency allocation, rather than the training loss. FP32 rounding can
change individual frequencies and checkpoint selection slightly. GPU scores are
estimates for the existing CPU decoder. CPU validation can be restored with
`--no-skip-cpu-validation`, and `--test-device cpu` restores CPU test scoring.
For all CPU scoring, use `--validation-device cpu --no-skip-cpu-validation
--test-device cpu`. The generic `train_nn_compression_model.py` keeps its CPU
validation default. Exact CPU compression/decompression checks remain limited
to `--benchmark-messages` (default 100); they and the published test vectors
still use the original NumPy float64 reference codec. No MCU runtime change is
required.

A local RTX 3070 comparison used a 1 MiB int8 export budget, three hidden layers,
context 3, batch size 1024, and 1,048,541 training bytes. Median training time for
the two warmed epochs changed from **1.579 s to 1.541 s**. The main improvement
was scoring the same exported model on 1,000 held-out messages (64,478 bytes):

| Scoring implementation | Median seconds | Bits/UTF-8 byte |
| --- | ---: | ---: |
| Previous CUDA FP64 | 0.5622 | 2.636799576 |
| Current CUDA FP32 | 0.02766 | 2.636799325 |
| CPU codec reference (one timing) | 8.961 | 2.636799576 |

FP32 GPU scoring was about **20 times faster** than FP64 GPU scoring in this
sample, with a difference of about **0.00000025 bits/byte** from the CPU
reference. This is a short local benchmark, not a convergence study or a speed
guarantee for every configuration. Details are saved in
`models/nn-fp32-performance/benchmark.json`. A fused Adam experiment was slower
for this workload, so the existing Adam implementation is retained.

Reports include training/validation/test precision, test scoring device, input
pipeline version, phase timings, and per-epoch throughput. New sweeps use format
`nn-cuda-sweep-v2-fp32`, preventing old FP64 scoring results from being silently
mixed into an FP32 run. Start a fresh output directory for a new sweep; old
results remain available through `--plot-only`.

### Earlier CUDA training throughput measurements

The earlier trainer packed message bytes into bounded batches and constructs preceding-byte
contexts with NumPy array operations. It preserves message boundaries, BOS padding,
batch boundaries, and seeded shuffling. It no longer creates a Python history list
for each byte. Input transfers use nonblocking CUDA copies with distinct source
arrays. The training loss accumulates on the GPU instead of reading a scalar and
testing a CUDA boolean on the CPU every step. Loss is checked for finiteness at
progress updates (about every ten seconds) and at each epoch's end, before export.
The optimizer, float32 training arithmetic, and compression reference are unchanged.

A local RTX 3070 comparison used a 256 KiB int8 export budget, context 16,
batch size 1024, two epochs, and the same first 1 MiB of whole training messages.
Both runs were profiled. The second epoch measured:

| Implementation | Training seconds | Training bytes/second |
| --- | ---: | ---: |
| Original byte-list loop | 4.55 | about 231,000 |
| Vectorized batching and fewer synchronizations | 1.53 | about 687,000 |

This is about **3 times faster** for that measurement. Both runs produced the
exact same model file, validation/test losses, and benchmark payloads. Timings are
single local observations and exclude CPU evaluation, not a promise for every
model or corpus. Details are in `models/nn-training-benchmark.json`; the before
and after reports are in `models/nn-performance-before/` and
`models/nn-performance-packed/`.

For more GPU work per optimizer step, try a larger batch explicitly:

```bash
OPENBLAS_NUM_THREADS=1 .venv-cuda/bin/python train_nn_cuda.py data/messages.jsonl \
  --context-bytes 16 --model-kib 256 --batch-size 8192 --epochs 10 \
  --output models/nn-cuda-256k-c16-b8192
```

An additional unprofiled run at batch size 8192 measured 0.37 seconds for its
second 1 MiB epoch (about 2.8 million bytes/second). A larger batch consumes more
memory and takes fewer optimizer steps per epoch, changing convergence; the
short trial had worse validation loss. Compare validation results and tune
epochs/learning rate before choosing it for compression. The default remains
1024. The same `--batch-size` option works with `sweep_nn_cuda.py`.

Progress and epoch summaries now print training throughput. `epochs.json` and
`report.json` record per-epoch training, export, and validation times plus training
bytes/second. `report.json` also has a `phase_seconds` breakdown for training,
export, validation, test scoring, and the coding benchmark. The existing outer
`training_seconds` field remains the combined elapsed run time. Sweep summaries
include the training phase time and throughput when available; older reports
without these measurements can still be resumed.

Those earlier runs used CPU validation, test scoring, and arithmetic coding.
Current CUDA launchers use FP32 GPU validation/test scoring by default, while
arithmetic coding remains a sampled NumPy CPU reference check. A small network can also be limited by
kernel launch overhead. Judge improvements by training bytes/second and total
run time, as well as GPU utilization.

### Stop training when validation stops improving

Neural training now uses validation-based early stopping by default. `--epochs`
is an upper bound. If the exported model's validation coding loss fails to improve
for **three consecutive validation checks**, training ends and keeps the model
with the lowest measured validation loss. This limits wasted training when
validation worsens as the network fits its training data, and also stops plateaus.
The test set is not used to make this decision.

```bash
.venv-cuda/bin/python train_nn_cuda.py data/messages.jsonl \
  --model-kib 256 --context-bytes 16 --epochs 100 \
  --validation-device cuda --early-stopping-patience 3 \
  --early-stopping-min-delta 0.001 --output models/nn-cuda-early-stop
```

Patience counts validation checks, not epochs. With `--validate-every 5`, three
unsuccessful checks can mean 15 additional training epochs. Skipped validation
epochs do not increment or reset patience. Monitoring uses the configured CPU
or GPU validation score, including its existing rounding caveat. Earlier checks
catch overfitting sooner; a wider interval reduces evaluation work.

The optional minimum delta ignores improvements too small to count as meaningful
progress. A decrease must exceed that value to reset patience, measured from the
last significant improvement. Several small decreases can accumulate enough to
reset patience. Checkpoint selection remains independent: every strict new loss
minimum is saved, even when it is smaller than the stopping threshold. The best
checkpoint can therefore be newer than the last patience reset.

Use `--early-stopping-patience 0` to train up to `--epochs` without this stopping
rule. Negative patience and negative/nonfinite minimum deltas are rejected.
After stopping, the configured final CPU validation (unless skipped), test
evaluation, and exact compression round trips run on the retained best model. `report.json` includes
an `early_stopping` section with the completed epoch count, stopping reason,
patience, threshold, and number of validation checks. `epochs.json` records the
count of checks without significant improvement; skipped epochs store `null`.

These options also work with `sweep_nn_cuda.py`; each grid point stops
independently and the sweep proceeds to the next point. Summaries record actual
completed epochs and whether stopping occurred. To resume an existing sweep
created before early stopping was added, supply `--early-stopping-patience 0`
with its original command. Enabling it changes the training configuration and
requires a fresh output directory; completed old models are preserved.

The implementation passed 96 focused neural/CUDA/sweep tests. A separate real
CUDA run, capped at 50 epochs and using a small 1 KiB training prefix, stopped at
epoch 26 after validation worsened for three checks. It retained epoch 23 and
passed all final exact round trips. Its report and epoch history are in
`models/nn-early-stopping-demo-lybf_r4t/`; this small run verifies stopping behavior,
not final compression quality.

### Validate less often or on the GPU

Validation measures compression on held-out messages and chooses a checkpoint;
it is not needed for gradient updates. It need not run every epoch. For example:

```bash
.venv-cuda/bin/python train_nn_cuda.py data/messages.jsonl \
  --model-kib 256 --context-bytes 16 --epochs 20 \
  --validate-every 5 --validation-device cuda --validation-batch-size 4096 \
  --output models/nn-cuda-validation-c16
```

This validates at epochs 5, 10, 15, and 20. The final epoch is always checked,
even when it is not a multiple of the interval or the interval exceeds the total
epoch count. Skipped epochs do not export a candidate and have `validation: null`
in `epochs.json`. They cannot be selected as best, so a wider interval can miss
a better intermediate model. Validation runs every epoch by default. The CUDA
launcher defaults to GPU validation; the generic trainer defaults to CPU.

`--validation-device cuda` batches byte contexts and scores the **exported**
model, including int8 quantization, using FP32 dequantized weights on the GPU.
It uses the codec's integer frequency allocation and byte-value tie breaking;
it does not substitute the training network's softmax loss. CPU/GPU precision
and matrix calculations can round differently, as explained in
[PyTorch's numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
GPU validation is therefore a checkpoint-selection estimate, with possible tiny
differences from the reference coding loss. Reduce `--validation-batch-size` if
the GPU evaluation needs too much memory; this is independent of training batch
size. Explicit CUDA validation fails if no CUDA device is available.

An earlier FP64 GPU comparison scored the same 1,000 held-out messages
(50,072 bytes) with a 256 KiB/context-16 exported model. CPU reference scoring
took **3.70 seconds**, versus **0.158 seconds on CUDA**, about **23 times faster**.
The scores differed by about `6e-14` bits/byte in that sample. This is a single
local measurement with `OPENBLAS_NUM_THREADS=1`; rounding agreement and speed
are not guaranteed for other models or hardware. Details are recorded in
`models/nn-validation-benchmark.json`.

The generic trainer performs one CPU reference validation pass after GPU
selection unless `--skip-cpu-validation` is set. CUDA launchers skip it by default;
use `--no-skip-cpu-validation` to restore it. Test scoring follows the validation
device unless `--test-device` overrides it. Exact compression round trips remain
on CPU. With CPU selection, the existing validation result is reused.

To skip that extra CPU validation pass, add `--skip-cpu-validation` together with
`--validation-device cuda`. For the prepared Wikipedia corpus, for example:

```bash
OPENBLAS_NUM_THREADS=1 .venv-cuda/bin/python train_nn_cuda.py data/wikipedia/prepared \
  --model-kib 256 --context-bytes 16 --epochs 100 \
  --validation-device cuda --skip-cpu-validation \
  --output models/nn-wikipedia-cuda-no-cpu-validation
```

GPU validation and early stopping continue normally. Only the final reference
validation pass is omitted; test scoring and exact compression round trips
still run. The flag requires CUDA validation and is enabled by default in
`train_nn_cuda.py` and `sweep_nn_cuda.py`. See the FP32 workflow above for current
sweep compatibility and CPU reference options.

`report.json` stores the CPU reference score in `validation` and the score used
to choose the checkpoint in `selection_validation`. Per-epoch scores use the
selected validation device; `runtime.validation_device` records that device.
`phase_seconds.reference_validation` measures the additional final CPU check.
When skipped, `validation` is `null`, `cpu_validation_skipped` is `true`, and the
reference validation time is zero; the CUDA score remains in `selection_validation`.
Sweep plots and `validation_bits_per_utf8_byte` in CSV summaries use the CPU
reference when available, otherwise the CUDA selection estimate. The
`validation_score_device` column identifies the metric's source, and plots with
CUDA scores label them as estimates. Selection score/device are also recorded.

Both validation controls work with `sweep_nn_cuda.py`. Start a new output
directory when moving an older sweep to the FP32 backend or changing validation
settings. New FP32 sweeps support `--resume` with matching settings. `--evaluation-messages` can also reduce scoring time
by using a smaller held-out sample, at the cost of a noisier model comparison.

### Sweep CUDA neural model sizes and contexts

`sweep_nn_cuda.py` runs a grid through `train_nn_cuda.py`, using the same Python
as the sweep. Run it with `.venv-cuda/bin/python` on Linux, or the Windows
Python 3.11 executable described below. The plotting dependency is installed in
this machine's CUDA environment; to recreate it:

```bash
.venv-cuda/bin/python -m pip install 'matplotlib>=3.7,<4'
.venv-cuda/bin/python sweep_nn_cuda.py data/messages.jsonl \
  --sizes-kib 16,64,256,1024 --contexts 0,1,2,4,8,16 \
  --epochs 10 --output models/nn-cuda-sweep
```

These are also the default grid values: 24 runs, each with up to 10 epochs and
early stopping after three validation checks without improvement.
`--sizes-kib` gives complete exported-model budgets in KiB. `--contexts` (alias
`--orders`) gives preceding-byte depths, reset for each message or prepared
record. Context zero is a learned neural baseline whose width still changes
with the size budget. Pairs that cannot fit even one hidden unit are recorded
as skipped. CUDA is required for training; the sweep does not fall back to CPU.

The sweep accepts the neural trainer's other training/corpus options, including
`--storage`, `--embedding-dim`, `--hidden-layers`, `--batch-size`, `--learning-rate`, `--threads`,
`--seed`, split controls, `--limit-messages`, and evaluation/benchmark limits.
All cells use the same split, seeds, samples, and training settings. Each cell
runs sequentially in its own process, releasing its GPU allocations when it
finishes. Validation can run on CPU or CUDA; final reference validation (unless
skipped), test scoring, and exact arithmetic round trips run on CPU.

For a smaller Wikipedia trial:

```bash
.venv-cuda/bin/python sweep_nn_cuda.py data/wikipedia/prepared \
  --sizes-kib 16,64,256 --contexts 1,4,16 --epochs 1 \
  --train-bytes 1048576 --evaluation-messages 100 --benchmark-messages 10 \
  --output models/nn-wikipedia-sweep-trial
```

Remove `--train-bytes` to train on the whole training split each epoch. Even
limited prepared-corpus trials scan the held-out splits to sample messages.
On Linux, prefix the command with `OPENBLAS_NUM_THREADS=1` if the small,
per-byte CPU inference calls spend too much time in multithreaded BLAS.

Each cell has a directory such as `256k-context16/`, containing the usual
`model.mcnn`, `report.json`, `epochs.json`, and `test-vectors.json`. The sweep
also writes:

- `summary.csv` and `summary.json`: actual size, architecture, best epoch,
  validation/test bits per byte, payload/framed percentages, time, and GPU memory.
- `compression.png` and `compression.svg`: validation loss against size/context,
  plus actual test payload and estimated framed percentages against context.
- `sweep.json`: corpus SHA-256 and settings checked before resuming.

Summaries are updated after each successful cell. Add `--resume` to the same
command to reuse completed runs after checking their model checksums, corpus
identity, and trainer settings. An incomplete cell without a final report is
preserved in an `.incomplete-*` directory and retrained. Training failures stop
the sweep; completed cells remain reusable. Use a fresh output directory if
you change settings or corpus, and run only one sweep per output directory.
To redraw existing plots without CUDA, training, or the original corpus:

```bash
.venv-cuda/bin/python sweep_nn_cuda.py --output models/nn-cuda-sweep --plot-only
```

Choose parameters using validation loss. The test comparisons are exploratory;
use fresh traffic for a final measurement. Percentages exclude the shared model
and radio overhead; framed sizes use the trainer's existing nonce/length/CRC
estimate.

### CUDA training on Windows (the RTX 3070 computer)

Use Windows Python 3.11 and the dedicated CUDA launcher:

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $py train_nn_cuda.py data/messages.jsonl --context-bytes 16 --model-kib 256 --epochs 10 --output models/nn-mesh-gpu-256k-c16
& $py train_nn_cuda.py data/wikipedia/prepared --context-bytes 16 --model-kib 256 --epochs 10 --output models/nn-wikipedia-gpu-256k-c16
```

The launcher loads the GPU runtime from
`%LOCALAPPDATA%/llm-compression/cuda-2.5.1-cu124-py311`. Windows denied loading
DLLs from this network share, so runtime libraries must be local. Large DLLs
use Windows LZX compression to fit on the crowded system drive. The verified
wheel download stays under `.cache/cuda-wheels` on the share; system PyTorch
is left unchanged. To recreate the installation:

```powershell
& $py install_nn_cuda.py
```

The installer obtains the official Windows Python 3.11 CUDA 12.4 wheel, supports
partial-download resumption, verifies its published SHA-256, and extracts the
runtime without C++ headers/static libraries. Existing Python supplies NumPy
and PyTorch's Python dependencies. The [official PyTorch installation table](https://pytorch.org/get-started/previous-versions/)
lists CUDA 12.4 wheels for the project's pinned PyTorch 2.5.1; this machine's
NVIDIA driver 552.44 reports CUDA 12.4 support.

The ordinary neural trainer defaults to `--device auto`, selecting CUDA only
when its Python environment supports it. The system Python's `+cpu` PyTorch
build cannot use the GPU; use the launcher for this dedicated GPU runtime.
`python` on this machine resolves to Inkscape's bundled Python, so use `$py`
as shown above. Linux continues to use its own `.venv-cuda/bin/python`.

Training runs on the selected device. The CUDA launcher now defaults to FP32
CUDA checkpoint validation and test scoring, skipping final CPU validation.
`--no-skip-cpu-validation` and `--test-device cpu` restore CPU reference scoring.
Arithmetic round trips use the NumPy CPU reference. Low GPU utilization during
those sampled CPU coding checks is expected. Reports record the training device, GPU name, CUDA version, and peak
allocated GPU memory. `--device cpu` explicitly selects CPU training.

## Filtered MeshCore corpus

The current `data/messages.jsonl` and `data/messages.csv` exclude all messages
from `#test`, `#alert`, `#bot`, and `#nrl`. Both now contain **48,552 messages**,
down from 72,340. Removed counts: `#test` 19,068; `#alert` 2,698; `#bot` 677;
`#nrl` 1,345 (23,788 total). Other records retain their original values and order.
The source database was not changed. Original exports are preserved as
`data/messages.*.before-channel-filter-*.bak`; `data/channel-filter-report.json`
records the exact paths, counts, and before/after SHA-256 values.

Repeat the same filtering on future existing exports with:

```bash
python filter_message_corpus.py --report data/channel-filter-report.json
```

To produce a filtered export directly from the database:

```bash
python extract_messages.py data/meshcore.db --output data/messages.jsonl --overwrite \
  --exclude-channel '#test' --exclude-channel '#alert' \
  --exclude-channel '#bot' --exclude-channel '#nrl'
```

Use `--output data/messages.csv` for CSV. Exclusions do not remove private or
unmapped messages. Existing trained models and their saved train/validation/test
splits still describe the old corpus; train into fresh output directories to
measure the filtered corpus. Earlier recorded results in this document remain
historical measurements, not results for this filtered snapshot.


On the Windows RTX 3070 machine, the CUDA launcher also works with the default
`python` command, which points to Inkscape Python 3.10. When the local CUDA runtime
is present, the launcher automatically relaunches the installed Python 3.11,
preserving all arguments and the process exit status:

```powershell
python train_nn_cuda.py data/messages.jsonl --model-kib 64 --context-bytes 3 --epochs 10 --validation-device cuda --output models/nn-mesh-cuda-new
```

The existing local CUDA 12.4 runtime was checked successfully with GPU training,
GPU checkpoint validation, CPU reference scoring, and exact arithmetic round trips.
No CUDA reinstall is needed for that runtime. Use a new output directory per run.

### Refreshed export including wardriving exclusion

The latest database re-export additionally excludes `#wardriving` (1,960
messages). Current `data/messages.jsonl` and `data/messages.csv` each contain
**46,592 messages**: 45,168 channel and 1,424 private messages. The five excluded
channels account for 25,748 of the source database's 72,340 messages.
`data/remote-term-export-report.json` records exclusion counts, export checksums,
and backup paths. The previous four-channel-filtered exports are preserved as
`data/messages.*.before-wardriving-export-*.bak`.

To repeat this export, add `--exclude-channel '#wardriving'` to the database
extraction command above, retaining all four earlier exclusions. The source
database remains unchanged; the existing four-channel filter audit is historical.

The Windows sweep launcher now also switches from Inkscape Python to the installed
Python 3.11 before loading NumPy or plotting libraries. A working example is:

```powershell
python sweep_nn_cuda.py data/messages.jsonl --sizes-kib 16,64,256 --contexts 1,3,8 --epochs 10 --validation-device cuda --output models/nn-mesh-sweep-new
```

The NumPy 2.1.3 environment needs compatible plotting binaries. This machine was
repaired with Matplotlib 3.9.4 and ContourPy 1.3.2. The `sweep` optional dependency
now requires Matplotlib >=3.9. On Windows the launcher keeps Matplotlib's font
cache under `.cache/matplotlib` on the project share.

## Compare codecs on enwik8

`benchmark_enwik8.py` downloads the original **100,000,000-byte enwik8** XML
corpus from [Matt Mahoney's test-data page](https://mattmahoney.net/dc/textdata.html).
It preserves every input byte and checks the published MD5 and SHA-1 before
using the dataset. Files stay under the ignored `data/enwik8/` directory.

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $py benchmark_enwik8.py --download-only

# Full-file gzip level 9, bzip2 level 9, and xz preset 9 comparison.
& $py benchmark_enwik8.py --output data/enwik8/full-comparison-new

# Add one or more project models; entire enwik8 is used unless a limit is given.
$env:OPENBLAS_NUM_THREADS = '1'
& $py benchmark_enwik8.py --model models/nn-mesh-cuda-new-64-4/model.mcnn --output data/enwik8/full-neural-comparison

# Faster exploratory comparison, explicitly labelled as a prefix in the report.
& $py benchmark_enwik8.py --limit-bytes 65536 --model models/nn-mesh-cuda-new-64-4/model.mcnn --output data/enwik8/prefix-comparison-new
```

On Linux use your project's `python` instead of `& $py` and set
`OPENBLAS_NUM_THREADS=1` before invoking Python for the neural reference codec.
The conventional codecs need only the standard library. Model loading uses the
project's NumPy dependency. `--model` is repeatable and accepts `.mcnn`, `.mcblm`,
and `.pnn` files (detected by their magic, not their filename). Models must already
be trained; the benchmark never trains on enwik8. A cleaned-Wikipedia-trained
model is not necessarily held out from this older XML corpus; record provenance
when interpreting compression quality.

Each algorithm writes a real compressed artifact and is decompressed again to
verify its original byte count and SHA-256. `report.json`, `report.csv`, and console
output show compressed bytes, bits per original byte, percentage of original,
and encoding/decoding times. Reports distinguish shared-model storage from the
total including the supplied model file. Decoder/runtime program storage is not
counted for either model or conventional codecs. Use a fresh output directory
for each run; nonempty outputs are refused, and partial reports remain available
if a later algorithm fails.

Conventional codecs use one continuous stream with their normal container.
Model codecs use independently reset blocks, default 262,144 bytes, to bound
memory and report progress. `--model-block-bytes` adjusts that boundary. Their
benchmark container adds an 84-byte header (magic, model SHA-256, original SHA-256,
original byte count, block size) and 8 bytes per block (original/compressed sizes).
`arithmetic_payload_bytes` excludes this container overhead. This is a desktop
benchmark container, not radio framing. Model arithmetic coding and decoding use
the existing exact CPU reference; GPU training hardware does not accelerate this
path. Full neural/Pico reference runs can be substantially slower than standard
compressors. Prefix results are explicitly marked and must not be presented as
full-enwik8 scores.

## Add a random Reddit comment sample

`sample_reddit_corpus.py` samples **1,000,000 eligible comments** without replacement
from all Parquet files under `data/REDDIT_comments`, then combines them with
`data/combined-messages.jsonl` into a new shuffled corpus. It excludes null,
whitespace-only, `[deleted]`, and `[removed]` raw bodies. Before writing each
sampled Reddit comment, it removes Markdown/HTML formatting and URLs, retains
visible link labels, image descriptions and code contents, collapses repeated
newlines to one, normalizes horizontal whitespace, and trims surrounding
whitespace. Literal underscores inside words and arithmetic asterisks are kept.
Comments are sampled uniformly across eligible raw rows without text
deduplication, so larger subreddits contribute proportionally more comments.
Source record IDs, subreddit names, and shard filenames remain as metadata.
The base corpus is copied through without preprocessing. The manifest records
cleanup counts and byte totals. Selected URL/formatting-only comments may become
empty strings; these records are retained to preserve the requested sample count.
The trainer already skips empty message bodies.

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $py -m pip install 'duckdb>=1.4,<3' 'mistune>=3,<4'
& $py sample_reddit_corpus.py
```

Default outputs are `data/reddit-comments-1m.jsonl`,
`data/combined-messages-reddit-1m.jsonl`, and
`data/combined-messages-reddit-1m.manifest.json`. Existing inputs and outputs are
never overwritten. The manifest records source shard sizes/row counts, seed,
selected IDs count, per-subreddit/shard sample counts, corpus counts, and SHA-256
hashes for the original, sample, and combined JSONL files.

`--count`, `--seed`, `--base`, `--sample`, `--output`, and `--memory-limit` are
configurable. The default seed is 42, the default memory budget is 4GB, and
DuckDB uses one thread for repeatable reservoir selection and output shuffling.
The shuffle distributes the original sources and Reddit across the new corpus;
original JSON records retain all fields. The complete Parquet body columns must
be scanned to exclude placeholders before uniform sampling. Temporary query
spill files stay alongside the output, not on the system drive.

### Run the Reddit export on the share host

Run from the project's **local filesystem directory on the server**, rather
than through an SMB/network mount. Create a small environment for the one required
library; CUDA, PyTorch, NumPy, and PyArrow are not required:

```bash
python3 -m venv .venv-reddit
.venv-reddit/bin/python -m pip install duckdb==2.0.1 'mistune>=3,<4'
.venv-reddit/bin/python -u sample_reddit_corpus.py --count 1000000 --seed 42 --memory-limit 4GB
```

Do not add `--local-stage` when the dataset is already on the host's local disk;
that option is only for accelerating reads from a remote share. The script takes
a snapshot of the Parquet filenames available at startup; files downloaded later
are not part of that run. It scans all eligible comments in that snapshot before
selecting one million. `--memory-limit` controls DuckDB's working-memory budget,
not a hard limit on total process RAM; adjust it for the host if needed.

The original corpus remains unchanged. Default new files:

- `data/reddit-comments-1m.jsonl`: the million sampled Reddit records.
- `data/combined-messages-reddit-1m.jsonl`: original records plus the sample,
  shuffled together with the fixed seed.
- `data/combined-messages-reddit-1m.manifest.json`: counts, source snapshot,
  sampling settings, per-shard/subreddit distribution, and output hashes.

Existing outputs are refused; choose new `--sample` and `--output` paths for a
second completed run. Ctrl+C interrupts an active query; incomplete corpus files
are not published. The tests use tiny synthetic Parquet files; no full Reddit
export is needed to run them.

### Faster Reddit sampling

The original method remains the default (`--method exact`). On a CPU-limited
host, use random-block preselection to avoid running text filters and a large
reservoir over every source row:

```bash
.venv-reddit/bin/python -u sample_reddit_corpus.py --method fast --count 1000000 --seed 42
```

Fast mode targets a raw candidate pool four times the requested count by default.
It samples random DuckDB system blocks before checking bodies, then runs the same
eligibility filter and final reservoir on that smaller pool. If too few eligible
comments remain, it doubles block coverage and retries; at 100% it falls back to
the exact full scan. It still returns exactly the requested number without
replacement when enough eligible rows exist, preserves comment bodies, removes
empty/deleted/removed placeholders, and shuffles the combined corpus as before.

The statistical compromise is **clustered approximate sampling**: nearby comments
can enter the candidate pool together. This is not an exact globally uniform
sample of eligible comments, and individual shard/subreddit representation can
vary more. The manifest records `sampling_method`, `candidate_factor`, and every
attempt's block coverage, including any full-scan fallback. A fixed seed and one
thread keep runs repeatable for an unchanged snapshot and DuckDB version.

Use `--candidate-factor 8` if placeholders dominate and the default causes retries.
This processes a larger candidate pool. Fast mode mainly reduces CPU/filtering and
reservoir work; reading Parquet metadata and some source data is still required,
so storage performance can limit the speedup. Do not add `--local-stage` on the
share host, where data is already local. Existing outputs remain protected; use
new `--sample` and `--output` paths when keeping results from an earlier run.

A small synthetic one-million-row test selected 5,000 eligible rows in 0.145 s
with fast mode versus 1.356 s with exact mode on the development machine. This is
an implementation smoke benchmark, not a throughput prediction for Reddit.

## Train on one corpus and verify on another

Both trainers and both sweeps accept `--verification-input` (alias
`--verification-corpus`). The positional input supplies training messages; all
validation, early stopping/smoothing selection, test scoring, and arithmetic
round-trip benchmarks use only the separate verification corpus.

For example, run on the host from the project directory:

```bash
.venv-cuda/bin/python train_nn_cuda.py data/reddit-comments-1m.jsonl \
  --verification-input data/messages.jsonl \
  --model-kib 1024 --context-bytes 8 --epochs 20 \
  --validation-device cuda --output models/nn-reddit-mesh-verification

.venv-cuda/bin/python sweep_nn_cuda.py data/reddit-comments-1m.jsonl \
  --verification-input data/messages.jsonl \
  --sizes-kib 64,256,1024 --contexts 3,4,8 \
  --validation-device cuda --output models/nn-reddit-mesh-verification-sweep

python train_compression_model.py data/reddit-comments-1m.jsonl \
  --verification-input data/messages.jsonl --model-kib 256 --order 4 \
  --output models/stat-reddit-mesh-verification
```

On Windows the corresponding CUDA launchers work through the default `python`
command. The verification file may be JSONL or CSV and uses the same `--text-field`
as training. It needs at least two distinct eligible messages. The default equal
validation/test fractions split **all** verification text groups 50/50; custom
fractions are normalized relative weights in this mode. `--split chronological`
uses first occurrence order; `--split random --seed 42` shuffles groups. Identical
verification texts never cross validation/test sets. `--evaluation-messages` and
`--benchmark-messages` still limit the scoring samples; use 0 to score all.

For a JSONL/CSV training input, all eligible messages are training candidates;
there is no additional 80/10/10 training-source split. Exact matches to any
eligible verification text are removed from training, including repeats. The
length/empty-message filters still apply. Training messages are streamed, with
SHA-256 keys used for duplicate suppression by default. For a very large corpus,
`--keep-duplicates` avoids the growing duplicate-key set and retains repeated
training rows. This trades lower memory use for frequency weighting. A preflight
pass counts eligible training messages before training begins.

Prepared Wikipedia input continues to use its original training partition;
its validation/test partitions are not used when a separate verification input
is specified. Matching verification texts are excluded there as well. Original
prepared splitting/grouping constraints still apply.

Reports record both input identities, verification SHA-256, split counts, and
training overlap/filter counts under `corpus`. Both sweep manifests also hash the
verification file and refuse resumption if its contents or settings change.
The option is shared by `sweep_compression_models.py` too. When omitted, the
existing single-corpus split and evaluation workflow is unchanged. Use fresh
output directories when switching to independent verification.
