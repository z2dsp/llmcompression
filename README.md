# Language models as lossless text compressors

A runnable Python demonstration inspired by [3Blue1Brown's video](https://www.youtube.com/watch?v=qbsb9DpdZ6s): use a language model's next-token probabilities to drive an **arithmetic coder**, then recover the original UTF-8 file exactly.

The default is the base [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) model, with about 135 million parameters, comfortably below 1B. Its weights are about 269 MB to download and are cached under this project's `.cache/huggingface/`. Inference runs locally on CPU. There is no training or hosted inference API.

For the desktop MeshCore messaging app over USB or TCP, including shared profiles and flood region scopes, see [MESHCORE.md](MESHCORE.md). Launch it with `.venv/bin/python -m llm_compression.meshcore_app` after installing the `[mesh]` extra.

## Run it

For a fresh checkout or an extracted copy of `llm-compression-demo.zip`, follow the setup instructions below. The ZIP includes the Python source, example input, tests, and dependency definitions. Dependencies and model weights download during setup and the first run; they are not bundled in the ZIP.

If the project virtual environment and default model are already installed:

```bash
.venv/bin/python -m llm_compression demo --offline
```

This compresses an original English sample, decompresses it, verifies every byte, compares sizes with gzip, and prints the probability and bit cost of the first few tokens. The demo includes model loading time in neither compression nor decompression timing.

To set up a fresh environment, use **Python 3.10–3.12** (tested with 3.12.3; the pinned PyTorch version does not support Python 3.14):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU wheels avoid downloading CUDA libraries on Linux.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m llm_compression demo
```

On macOS, install `torch==2.5.1` from the normal PyPI index instead. The first run needs network access to Hugging Face. Afterward, `--offline` uses only the cache. Use `--cache-dir PATH` to choose another cache location.

## Compress your own text

```bash
python -m llm_compression compress examples/english.txt example.llmc --offline
python -m llm_compression info example.llmc
python -m llm_compression decompress example.llmc restored.txt --offline
```

Decompression automatically loads the model revision and prediction settings saved in the archive, then verifies the original length and SHA-256. File input and output use binary mode, preserving UTF-8, CRLF, tabs, and trailing newlines. Invalid UTF-8 input is rejected. Output files must not already exist.

To compare compression on another file or get a JSON report:

```bash
python -m llm_compression demo --input examples/english.txt --trace 30 --offline
python -m llm_compression demo --json --offline > demo-results.json
```

The installed `llm-compress` command supports the same arguments.

## Swap the model

Pass a Hugging Face causal language model ID:

```bash
python -m llm_compression demo --model HuggingFaceTB/SmolLM2-360M
python -m llm_compression compress examples/english.txt bigger.llmc \
    --model HuggingFaceTB/SmolLM2-360M --context-tokens 512
```

The 360M alternative is also below 1B parameters. `--revision COMMIT_OR_BRANCH` selects a particular checkpoint. The default 135M model is pinned to commit `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; other IDs resolve `main` to a commit, which is recorded in the archive. Both model and tokenizer load from that same commit.

The adapter uses `AutoModelForCausalLM` and `AutoTokenizer`. Models must be supported by the pinned Transformers version, have safetensors weights, run with CPU float32/eager attention, and use a tokenizer that preserves the input exactly. A tokenizer that normalizes or drops input is rejected rather than silently losing text. Custom remote model code is disabled. The default model is the one verified end to end; other checkpoints can differ in compatibility, memory, and compression quality.

For a different prediction backend, implement the small `Predictor` protocol in `llm_compression/model.py`. The arithmetic coder and archive format do not depend on Hugging Face.

## What is happening?

1. Tokenize the text with no automatically added special tokens. Check that decoding those tokens reproduces the exact input.
2. Feed a shared beginning-of-sequence token to the model. Its logits give a distribution for the first input token.
3. Convert that distribution to integer frequencies totaling `2**24`. Reserve at least one count per token so even an extremely unlikely token can be encoded.
4. Narrow the arithmetic coder's interval according to the actual token's cumulative probability range. Emit bits as leading bits of the interval become fixed.
5. Feed the actual token back to the model to predict the next token, and continue.
6. During decoding, compute the same distribution and find which token's interval contains the encoded value. Feed that recovered token back into the model. The decoder never needs the original text.

A token with probability `p` contributes approximately `-log2(p)` bits. A predictable token can cost less than one bit because arithmetic coding encodes the whole sequence together. The report's predicted coding cost uses the actual quantized probabilities; the payload also includes arithmetic termination and byte padding. Token labels such as `Ġthe` come from the tokenizer: `Ġ` represents a preceding space in this tokenizer.

Context restarts every 256 tokens by default. Each block begins with the same start token, and the model's KV cache is reset. This bounds memory and supports files longer than the model's context window. A larger `--context-tokens` can improve prediction but uses more memory; it must fit the model's position limit. The arithmetic stream continues across those model context resets.

The archive contains a magic/version prefix, a length-prefixed JSON header, the arithmetic payload, and a SHA-256 digest of the archive body. The header stores the model ID and immutable revision, tokenizer/model vocabulary settings, runtime versions, context size, original byte length and SHA-256, payload length, and token count. The token count tells the decoder when to stop; no end-of-text token is appended to the input.

## Interpret the comparison

The demonstration reports these separately:

| Representation | Includes |
| --- | --- |
| `original_utf8` | Original file bytes |
| `gzip` | A complete gzip file at compression level 9 |
| `llm_payload` | Arithmetic-coded tokens with termination and byte padding |
| `llm_archive` | Payload plus model/runtime metadata, length fields, and checksums |

All sizes exclude the downloaded model weights. Sender and receiver must already share the model. Counting a roughly 269 MB checkpoint against a small text file would make this a very poor compressor. Metadata can also dominate short messages; check the full archive size when comparing with gzip. Bits per character use Unicode code points, and the size ratio uses original UTF-8 bytes.

Compression quality depends on the text and model. This demo does not promise a particular bits-per-character result or that it beats gzip. It also makes hundreds of model calls for a small file, so it is much slower than conventional compression.

## Reproducibility and limits

Arithmetic decoding requires exactly matching integer probability tables. Both directions therefore use the same one-token inference path, CPU float32, eager attention, one PyTorch thread, and deterministic algorithms. Compression intentionally avoids batched teacher forcing, which can change floating-point rounding.

Archives record and check Python, PyTorch (including the build suffix), Transformers, Tokenizers, NumPy, OS, architecture, and PyTorch's CPU capability. These checks reduce accidental mismatches; they do **not** guarantee portability between different CPUs or builds. Keep the original environment and cached model for reliable decoding. A final SHA-256 check detects an incorrect reconstruction before the CLI writes the restored file. This is an educational format, rather than a long-term archival format or a hardened decoder for hostile files.

## Train a small model on MeshCore messages

See [MCU_TRAINING.md](MCU_TRAINING.md) for database extraction, configurable
statistical language model training, measured compression results, and an MCU
porting guide. The integer model is a separate compression backend, with no
PyTorch requirement at inference. It is not yet integrated into the MeshCore GUI.

```bash
python extract_messages.py data/meshcore.db --output data/messages.jsonl
python train_compression_model.py data/messages.jsonl \
  --model-kib 256 --output models/my-256k
python mcu_compress.py info --model models/my-256k/model.mcblm
```

The extraction script needs only Python; training needs NumPy. Local data and
trained models are excluded from Git and the demo ZIP.

The statistical trainer accepts any finite `--order` and has no software upper
limit on `--model-kib`; available memory and storage determine practical sizes.
For example, `--model-kib 131072 --order 64` requests a 128 MiB budget and a
64-byte context. History resets for each message.

For the full English Wikipedia training corpus, use
`prepare_wikipedia.py download` and `prepare_wikipedia.py prepare`, then pass
`data/wikipedia/prepared` to the trainer. Downloads and preparation are resumable;
prepared-corpus training streams batches without loading the corpus into RAM.
Training counts batches with up to four worker threads by default; use
`--workers N` to configure it or `--workers 1` for serial counting.
See the [Wikipedia workflow](MCU_TRAINING.md#download-and-prepare-the-full-english-wikipedia)
for the additional PyArrow dependency and commands.
For longer Wikipedia contexts, prepare a fresh corpus with larger
`--sample-bytes`, or `--sample-bytes 0` for whole articles and train with
`--max-message-bytes 0`. The existing prepared corpus uses 160-byte samples.

To add conversational training data to your combined messages, run
`python append_dailydialog.py`. It downloads the pinned
[ConvLab/DailyDialog dataset](https://huggingface.co/datasets/ConvLab/dailydialog)
and adds all 102,979 utterances to `data/combined-messages.jsonl` using the `body`
field. Reruns skip already imported source IDs, and an existing corpus is backed
up before replacement. This importer uses only Python's standard library.
DailyDialog is licensed CC BY-NC-SA 4.0. See the
[DailyDialog workflow](MCU_TRAINING.md#add-dailydialog-utterances-to-the-combined-corpus)
for options and training split guidance.

## Tests and source

```bash
python -m pytest -q
# Include the real 135M model test; downloads weights if missing:
RUN_MODEL_TESTS=1 python -m pytest -q
# Entirely offline once the default model is cached:
HF_HUB_OFFLINE=1 RUN_MODEL_TESTS=1 python -m pytest -q
```

Tests cover changing arithmetic distributions, rare symbols, coding length versus information content, probability quantization, exact UTF-8 round trips, empty input, context resets, repeatability, incorrect settings, corruption, and CLI error handling. The opt-in model test uses separate encoder and decoder model instances.

- `llm_compression/arithmetic.py`: pure Python integer arithmetic coder.
- `llm_compression/model.py`: Hugging Face adapter and deterministic probability quantization.
- `llm_compression/codec.py`: compression, decompression, checksums, and archive parsing.
- `llm_compression/cli.py`: runnable commands.
- `llm_compression/demo.py`: original English sample and size comparison.

For the broader connection between prediction and compression, see [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668).

Compare MCU model budgets and context depths with `sweep_compression_models.py`; see the [sweep workflow](MCU_TRAINING.md#sweep-model-size-and-context-depth) for commands and generated compression plots.

Train bounded-storage neural byte models on the same datasets with `train_nn_compression_model.py`; see [neural comparison training](MCU_TRAINING.md#train-neural-models-for-comparison).

Use `train_nn_cuda.py` for CUDA training and `sweep_nn_cuda.py` for model-size/context
sweeps. Both now default to FP32 GPU training, validation, and test scoring, with
no FP64 CUDA arithmetic. See the [FP32 GPU workflow](MCU_TRAINING.md#fp32-gpu-training-and-scoring)
for commands, CPU reference options, and measured performance.

Generate a text continuation with
`.venv/bin/python generate_nn.py MODEL.mcnn "Initial context" --bytes 200`.
Add `--temperature 0.8 --seed 42` for reproducible sampling instead of the default
greedy choice; see [text generation](MCU_TRAINING.md#generate-a-continuation).

Explore neural model activations with `.venv/bin/python visualize_nn.py` and
choose a trained `model.mcnn` file. Type a message, step through UTF-8 bytes, or
press Play to see the embeddings, hidden neurons, and next-byte probabilities
change. See the [activation viewer](MCU_TRAINING.md#visualize-neural-activations).

For integer inference and lossless compression on a Raspberry Pi Pico W, see the
[Pico MicroPython demo](pico/README.md), including Viper kernels and measured
[benchmark results](pico/benchmark-results.json).

The [PY32F403 native firmware](py32f403/README.md) runs the same integer codec
with flash-resident weights, a 2.8 KiB NN working state, and ST-Link/UART build
and benchmark tools for a 64 KiB RAM MCU.
Compare model codecs with gzip, bzip2, and xz using `benchmark_enwik8.py`; see the [enwik8 benchmark workflow](MCU_TRAINING.md#compare-codecs-on-enwik8).
