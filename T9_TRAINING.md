# Small neural models for T9 prediction

The T9 scripts train a separate model for choosing text from ambiguous phone
keys. The existing compression scripts and models keep their own formats.

## Train

Use the project's Python environment with NumPy and PyTorch installed:

```bash
python train_t9_model.py corpus.txt --output models/t9-1m --model-kib 1024
```

Input can also be a message JSONL/CSV file (the same body/text fields as the
compression trainers), or a prepared Wikipedia directory/manifest:

```bash
python train_t9_model.py combined.jsonl --output models/t9-messages --model-kib 1024
python train_t9_cuda.py data/wikipedia/prepared --output models/t9-10m --model-kib 10240
```

`train_t9_model.py` automatically selects CUDA when available, otherwise CPU.
`train_t9_cuda.py` requires CUDA and supports the same cached Windows Python
3.11 runtime as `train_nn_cuda.py` (install with `install_nn_cuda.py` if needed).
Validation and test use the deployed NumPy CPU predictor for both launchers.

`--model-kib` is a hard budget for the complete exported file: 1024 is 1 MiB
and 10240 is 10 MiB. The trainer chooses the largest equal hidden-layer width
that fits the budget, including the header, weights, biases, and quantization
scales. Defaults are int8 storage, three tanh hidden layers, 8-dimensional
embeddings, 32 preceding text bytes, and 16 current/upcoming word keys.
`--storage float32` is also available.

Useful settings:

```bash
python train_t9_model.py combined.jsonl \
  --output models/t9-custom --model-kib 2048 \
  --context-bytes 64 --key-context 24 --embedding-dim 8 \
  --epochs 30 --batch-size 1024 --learning-rate 0.001 \
  --beam-width 8 --selection-metric word_accuracy \
  --split random --seed 42 --evaluation-messages 1000 --patience 3
```

Use one line on PowerShell, or its backtick line continuation. Run each script
with `--help` for all options. Installed entry points are `t9-train`,
`text-to-t9`, and `t9-predict`.

The output directory must be empty or absent. It will contain:

- `model.t9nn`: the best exported checkpoint.
- `epochs.json`: training loss and scheduled validation accuracy.
- `report.json`: architecture, file size, corpus splits, runtime, best epoch,
  and held-out validation/test accuracy.

## Convert a text file and predict it back

Keep an independent text file to measure performance on unseen text:

```bash
python text_to_t9.py heldout.txt heldout.t9
python predict_t9.py models/t9-1m/model.t9nn heldout.t9 predicted.txt \
  --reference heldout.txt --report accuracy.json --beam-width 8
```

Conversion and prediction accept UTF-8 files. Output files must be new.
The original reference is optional; omit `--reference` and `--report` to
decode without scoring. A mismatched reference/keystroke pair is rejected.

The decoder receives only keystrokes. It feeds back its own guessed text,
including mistakes, as the context for subsequent predictions. Supplying a
reference adds scoring after decoding; it never corrects the model's history.

The report prints JSON and optionally saves it. Accuracy rates are fractions
from 0 to 1:

| Metric | Denominator |
| --- | --- |
| `letter_accuracy` | ASCII letters whose values the model had to guess |
| `word_accuracy` | Contiguous ASCII a-z runs; a word must match completely |
| `character_accuracy` | All characters, including supplied literals |
| `message_accuracy` | Complete records; the file prediction CLI treats its input as one record |

Counts accompany every metric. Letter and word accuracy are the useful measures
of T9 disambiguation; spaces and punctuation are already supplied. A rate with
an empty denominator is JSON null.

## T9 typing simulator and activation viewer

Open the desktop app with a trained T9 model:

```bash
python visualize_t9.py models/t9-custom/model.t9nn
python visualize_t9.py models/t9-custom/model.t9nn --text "I slow cook to make life soft"
python visualize_t9.py models/t9-custom/model.t9nn --keys 4663 --beam-width 8
```

Run without a model argument to choose a `.t9nn` file in the app. The installed
entry point is `t9-visualize`. NumPy and Tkinter are required; PyTorch is not.
On Debian/Ubuntu, install `python3-tk` if Tkinter is missing. A desktop display
is required.

- Click the phone keypad or edit/paste keystrokes directly. `0` inserts a space;
  punctuation is literal. Escape original digits/backslashes as described below.
  The keyboard space bar inserts `0`. Backspace removes a complete escaped
  token. Clear starts a new message.
- Enter ordinary text and click **Convert text → keys** to try a sentence.
  The original text is used only for the accuracy comparison; predictions feed
  back their own guessed text. The comparison disappears when the keys differ.
- Predictions update as keys arrive. **Replay typing** re-enters the current
  keystrokes one token at a time, showing how additional keys revise a word.
  Pause leaves the currently typed prefix in the editor.
- Change **Beam width** (1–64) to compare greedy decoding with wider search.
  The selected word's alternatives show summed conditional log probabilities
  (`Log P`), not normalized confidence across all possible spellings.
- Use Previous/Next, the position field, or the slider to inspect a character.
  Positions count decoded characters from zero, including supplied literals.
  The selected letter is highlighted in the prediction. Inspection sees the
  supplied remainder of the current word; replay shows incremental lookahead.
- Inspect separate text/key embeddings, each tanh hidden layer, and the 26
  masked letter probabilities. Hover over tiles for values and input slots.
  **Tiles per layer** controls grouping for display only. All neurons still run.

The letter probability table is conditioned on the chosen spelling's preceding
history. The beam's chosen letter can differ from the local highest probability
because the beam scores complete words. Spaces, punctuation, and Unicode
literals are copied rather than predicted, so they have no activation trace.
Model loading and prediction run in a background worker; editing invalidates
older results. Predictions use the same NumPy decoder as `predict_t9.py`.

## Keystroke notation

This is a text notation for ambiguous letter keys and explicitly supplied
literal characters:

| Key | Letters |
| --- | --- |
| 2 | abc |
| 3 | def |
| 4 | ghi |
| 5 | jkl |
| 6 | mno |
| 7 | pqrs |
| 8 | tuv |
| 9 | wxyz |
| 0 | space |

For example, `Hello world!` becomes `43556096753!`. Predictions are ASCII
lowercase: capitalization is discarded before both training and scoring.

Punctuation, tabs, CR/LF, accented letters, emoji, and other non-ASCII characters
are preserved as supplied literals. They are not predicted by this English
alphabet model. There is no accent folding or general Unicode case folding.
Original digits are backslash-escaped (`Room 2` becomes `76660\2`), and an
original backslash becomes two backslashes. Raw `1`, ASCII letters, and raw
spaces are rejected in a keystroke file to catch accidental text input.

This mapping is deliberately ambiguous: `good` and `home` both become
`4663`. The model chooses a spelling; exact recovery is not guaranteed.

## Model and evaluation

Each training example predicts one letter from:

1. A fixed window of preceding decoded UTF-8 bytes (left-padded with BOS).
2. The current key and a fixed window of upcoming keys in the same word
   (right-padded with zero at a word boundary).

The model concatenates learned text/key embeddings and applies an MLP with
one to three tanh hidden layers and 26 output logits. A mask restricts the
softmax to the three or four letters on the current key. Training minimizes
this conditional cross-entropy, so model capacity is spent on ambiguities that
a T9 user encounters. Supplied literals are context rather than loss targets.

Prediction evaluates only letters consistent with each key. Beam search keeps
the most probable candidate prefixes, then commits the best complete spelling
before moving to the next word. `--beam-width 1` uses greedy decoding; wider
beams approximate the best spelling more thoroughly at greater runtime cost.
No dictionary or word vocabulary is required, so names and unseen words are
possible. The decoder uses known remaining keys in the current word; it does
not see future words or generate next-word completions.

The default checkpoint criterion is **free-running word accuracy of exported
weights on validation data**, with letter accuracy breaking ties. Set
`--selection-metric letter_accuracy` to reverse those priorities. Exact ties
retain the earlier checkpoint. Test data is evaluated once after selecting the
best model, and never selects a checkpoint. Early stopping counts scheduled
validation checks without improvement; `--patience 0` disables it.
`--validate-every N` reduces validation frequency; the final epoch is always
validated. Training loss uses true preceding text, while accuracy uses predicted
preceding text.

JSONL/CSV and plain-text data are lowercased before grouping identical text into
disjoint training/validation/test splits (default 80/10/10, ordered by first
occurrence). `--keep-duplicates` preserves occurrences within their assigned
split. Plain text is read as line records; long lines are segmented on UTF-8
boundaries, preferring spaces. Corpora with fewer than three distinct records
are segmented further if possible. JSONL/CSV records over
`--max-message-bytes` are skipped. Records with no ASCII letters are skipped.
Prepared corpora retain their predefined partitions; normalized copies of
selected evaluation messages are excluded from training, and validation copies
are excluded from test sampling.

Validation/test default to deterministic reservoir samples of up to 128 records
per split. Use `--evaluation-messages 0` to score the full held-out splits.
Training examples stream in bounded batches; prepared corpora are re-iterable
and are not loaded into RAM. Ordinary text/JSONL/CSV splitting groups unique
records in RAM, as the compression trainers do.
`--train-bytes` limits training to whole records per epoch.

Text history resets at training/evaluation record boundaries. The standalone
prediction CLI decodes its entire file continuously, including across newlines;
its scores can therefore differ from per-message corpus scores.

## Storage and dependencies

Inference and conversion require NumPy; training additionally requires PyTorch.
The `.t9nn` format has its own versioned header and SHA-256 integrity check.
It is separate from compression `.mcnn` models.

Int8 files use one symmetric scale per tensor. The runtime dequantizes them to
NumPy float32 arrays, using approximately four times the weight storage in RAM,
plus working buffers. This implementation targets compact parameter storage;
it does not include an integer-only Pico/PY32 firmware port.

## Tests

```bash
python -m pytest tests/test_t9.py tests/test_t9_inspection.py tests/test_t9_visualizer.py -q
```

Tests cover notation and literals, constrained greedy/beam decoding, predicted
history, word lookahead boundaries, metric denominators, normalized split
isolation, model budgets through 10 MiB, quantization/export integrity,
PyTorch/NumPy agreement, CUDA launcher selection, and end-to-end CPU training
with plain text, JSONL, CSV, and prepared corpora.
Inspection tests compare activation logits and word scores with the deployed
decoder. GUI tests cover typing, replay, navigation, and model/input errors;
they skip when no desktop display is available.
