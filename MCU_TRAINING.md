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
flattens those position-specific embeddings, applies a tanh hidden layer, and
predicts a 256-way next-byte distribution. It has no recurrent state or tokenizer.
The hidden width is automatically chosen to fit `--model-kib`, counting the whole
portable binary: header, embeddings, both dense layers, biases, and quantization
scales. Context zero is supported as an unconditional neural baseline.

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
- `--storage int8` (default) or `float32`: export representation. Training uses
  float32; int8 uses one scale per tensor and validation selects epochs after
  quantization. Embedding dimension defaults to 8; use `--embedding-dim` to vary it.
- `--epochs`, `--learning-rate`, `--batch-size`, and `--threads`: training controls.
- `--early-stopping-patience N` (alias `--patience`): stop after N validation
  checks without improvement; enabled by default with 3 checks. Use `0` to disable.
- `--early-stopping-min-delta D`: minimum significant validation loss decrease in
  bits/byte; default `0` accepts any strict decrease. The best checkpoint still
  includes smaller improvements.
- `--device cuda`: require GPU training; the default `auto` selects CUDA when
  available, otherwise CPU. Validation defaults to the NumPy CPU reference;
  `--validation-device cuda` enables batched GPU checkpoint selection. Final
  test/coding checks use the CPU reference.
- `--skip-cpu-validation`: omit the final CPU validation pass after GPU selection;
  requires `--validation-device cuda`. CPU test scoring and coding checks still run.
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

### Pico W MicroPython inference

The [Pico demo](pico/README.md) implements a separate integer inference format
using trained `.mcnn` weights, lookup-table activations, Viper kernels, and a
lossless arithmetic codec. A 32 KiB/context-3 source model was deployed and
benchmarked on the connected Pico W: about 27 ms per byte prediction and
37 original bytes/s encoding at 125 MHz. Its exported model plus tables occupies
50,212 flash bytes, with a measured 63,280-byte inference heap delta.
The Pico payloads agree exactly with its desktop integer reference, but are not
compatible with the original float64 `.mcnn` runtime.

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

Training uses CUDA; exported-model validation defaults to the NumPy CPU reference,
with optional batched GPU validation described below. Final reference validation,
test scoring, and arithmetic round trips use CPU. Low GPU utilization during
those CPU stages is expected. Reports record the actual training device, GPU name, CUDA version,
and peak allocated GPU memory. The CUDA launcher requests CUDA explicitly and
reports an error if it is unavailable, rather than silently using CPU.

Setup verification passed all 15 neural-model and launcher tests, including
training on CUDA with both message and prepared-corpus input. A separate
one-epoch launcher run trained on 8,162 local message bytes, exported a
261,882-byte int8 model, and decoded all three benchmark messages exactly.
Its runtime report confirms CUDA 12.4 on the RTX 3070 and 73,441,792 peak allocated
GPU bytes. The model/report are in `models/nn-linux-cuda-smoke-6m78_mfq/`;
this short run verifies the environment, not final compression quality.
An installed-package snapshot is saved in `.cache/nn-cuda-installed.txt`.

### CUDA training throughput

The trainer packs message bytes into bounded batches and constructs preceding-byte
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

Default validation, final test scoring, and arithmetic coding use the NumPy CPU
reference and can leave the GPU idle. A small network can also be limited by
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
a better intermediate model. Defaults remain validation every epoch on CPU.

`--validation-device cuda` batches byte contexts and scores the **exported**
model, including int8 quantization, using float64 dequantized weights on the GPU.
It uses the codec's integer frequency allocation and byte-value tie breaking;
it does not substitute the training network's softmax loss. Float64 CPU/GPU
matrix calculations can still round differently, as explained in
[PyTorch's numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
GPU validation is therefore a checkpoint-selection estimate, with possible tiny
differences from the reference coding loss. Reduce `--validation-batch-size` if
the GPU evaluation needs too much memory; this is independent of training batch
size. Explicit CUDA validation fails if no CUDA device is available.

A warmed local RTX 3070 comparison scored the same 1,000 held-out messages
(50,072 bytes) with a 256 KiB/context-16 exported model. CPU reference scoring
took **3.70 seconds**, versus **0.158 seconds on CUDA**, about **23 times faster**.
The scores differed by about `6e-14` bits/byte in that sample. This is a single
local measurement with `OPENBLAS_NUM_THREADS=1`; rounding agreement and speed
are not guaranteed for other models or hardware. Details are recorded in
`models/nn-validation-benchmark.json`.

By default after GPU selection, the chosen model gets **one CPU reference validation pass**.
Final test scoring and exact compression round trips also remain on CPU. These
checks can still take time, but they run once per training run instead of every
epoch. With CPU selection, the previously computed validation result is reused.

To skip that extra CPU validation pass, add `--skip-cpu-validation` together with
`--validation-device cuda`. For the prepared Wikipedia corpus, for example:

```bash
OPENBLAS_NUM_THREADS=1 .venv-cuda/bin/python train_nn_cuda.py data/wikipedia/prepared \
  --model-kib 256 --context-bytes 16 --epochs 100 \
  --validation-device cuda --skip-cpu-validation \
  --output models/nn-wikipedia-cuda-no-cpu-validation
```

GPU validation and early stopping continue normally. Only the final reference
validation pass is omitted; CPU test scoring and exact compression round trips
still run. The flag requires CUDA validation, and is also accepted by
`sweep_nn_cuda.py`. It defaults to disabled, so older sweeps remain resumable
with their original settings.

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

Both validation controls work with `sweep_nn_cuda.py`. Existing default CPU
sweeps can still be resumed; changing the validation interval or device requires
a fresh output directory. `--evaluation-messages` can also reduce scoring time
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
`--storage`, `--embedding-dim`, `--batch-size`, `--learning-rate`, `--threads`,
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

### CUDA training on Windows (the other computer)

The neural trainer defaults to `--device auto`: CUDA is selected when PyTorch
can access a GPU, otherwise CPU. `--device cuda` requires a working CUDA runtime
and fails with an installation/driver diagnostic if it is unavailable.

Use the same Python for installation and training. In PowerShell, with the
Python 3.11 installation used in this project:

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $py -m pip install --upgrade torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
& $py -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
& $py train_nn_compression_model.py data/messages.jsonl --device cuda --context-bytes 16 --model-kib 256 --epochs 10 --output models/nn-mesh-cuda-256k-c16
```

The official [PyTorch installation table](https://pytorch.org/get-started/previous-versions/)
lists CUDA 12.4 wheels for the project's pinned PyTorch 2.5.1. The Windows machine's
NVIDIA driver 552.44 reports CUDA 12.4 support. A `+cpu` PyTorch build cannot use
the RTX 3070 even when `nvidia-smi` detects it. `python` on that machine points
to Inkscape's Python; use the explicit `$py` above.

Training runs on the selected device. Checkpoint validation can use CUDA with
`--validation-device cuda`. Final validation (unless skipped), test scoring, and arithmetic round
trips use the NumPy CPU reference so final compression metrics describe that
decoder. Low GPU utilization during those CPU stages is expected. Reports record the actual training device,
GPU name, CUDA version, and peak allocated GPU memory. Set `--device cpu` to
force CPU training.

On the Windows machine C: had insufficient free space for the CUDA wheel plus its
installed libraries. The CUDA build is therefore installed under the ignored
`.cache/cuda-runtime` directory on the project share. Use this launcher with
Windows Python 3.11 to select that runtime for this process only:

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $py train_nn_cuda.py data/messages.jsonl --context-bytes 16 --model-kib 256 --epochs 10 --output models/nn-mesh-gpu-256k-c16
& $py train_nn_cuda.py data/wikipedia/prepared --context-bytes 16 --model-kib 256 --epochs 10 --output models/nn-wikipedia-gpu-256k-c16
```

The CUDA launcher accepts the same options as the regular neural trainer and
selects CUDA explicitly. The ordinary Python installation retains its CPU build;
use the launcher to load the project-local GPU build. No shell activation is
required. The cached runtime is machine-specific and is not committed to Git.
The project-local CUDA runtime can be recreated with:

```powershell
& $py install_nn_cuda.py
```

This helper downloads the Windows Python 3.11 PyTorch 2.5.1 CUDA 12.4 wheel from
the official PyTorch index, verifies its published SHA-256, and extracts it into
`.cache/cuda-runtime`. It uses large buffered reads/writes for the network share
and leaves the system PyTorch installation unchanged. The project's existing
Python installation supplies NumPy and PyTorch's Python dependencies.
