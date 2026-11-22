# Pico W neural compression demo

This runs a trained byte predictor and a lossless arithmetic codec on stock
MicroPython. The tested model is
`models/nn-cuda-sweep-mc-small/32k-context3/model.mcnn`: 32,740 bytes of int8
weights/scales/header, context 3, embedding 8, and 108 hidden units. Each byte
prediction requires 30,240 dense multiply-accumulates.

Both dense layers, the hidden activation, and probability allocation use
`@micropython.viper` and contiguous integer buffers.
There is no floating-point inference: hidden activations use an 8 KiB tanh
lookup table, and softmax uses an 8 KiB exponential lookup table. Integer
normalization produces 256 positive frequencies summing to 65,536. The exported
model is 50,212 bytes including these tables and converted biases.

The format is `MCPNN001`. Its probability arithmetic differs from the existing
float64 `.mcnn` codec. Both sender and receiver must use the same `.pnn` file and
integer runtime. The trained weights are retained; this does not retrain the
network. Model/table bytes are pre-shared, and message length and integrity
framing are handled externally.

## Prepare and deploy

Run from the repository root. Preparation needs NumPy; deployment needs
`pyserial`, already installed in this project's `.venv` environment. The scripts
do not require CUDA or replace the Pico's firmware.

```bash
.venv-cuda/bin/python pico/prepare_model.py
.venv/bin/python pico/run_benchmarks.py --port /dev/ttyACM0 --soft-reset
```

`prepare_model.py` supports `--model`, `--output`, `--corpus`, and `--messages`.
It samples 30 nonempty MeshCore bodies with seed 42, creates reference payloads,
and checks compression quality against the original predictor. Sample bodies
are private: generated files stay under ignored `models/pico-32k-context3/`.

`run_benchmarks.py` supports `--assets`, `--iterations`, and `--timeout`. Add
`--soft-reset` to reset the interpreter and release previous diagnostic objects
before measurement. It
interrupts the serial REPL, uploads only into `/llm_pico_demo`, verifies SHA-256
for uploads, and returns to the REPL when finished. It leaves `boot.py`,
`main.py`, other files, firmware, CPU frequency, and Wi-Fi configuration alone.
No program is installed to start at boot.

Results are saved to `models/pico-32k-context3/pico-benchmark.json`. The report
contains aggregate metrics and per-message lengths/times, without message text.
It records firmware, CPU frequency, memory, stage timing, Python versus Viper
speed, full encode/decode speed, and agreement with desktop predictions.

## Use at the Pico REPL

```python
import sys
sys.path.insert(0, '/llm_pico_demo')
from nn_runtime import Model
from nn_codec import compress, decompress

model = Model('/llm_pico_demo/model.pnn')
message = b'Hello from the Pico W'
payload = compress(message, model)
assert decompress(payload, len(message), model) == message
print(len(message), len(payload))
```

`Model(..., backend='python')` selects the slower reference dense loops.
The same runtime can run on CPython with that backend; it has no NumPy dependency.
`cdf()` reuses its output buffer, so copy it if retaining multiple predictions.

## Correctness and measurement limits

The device compares complete CDF checksums for 32 contexts and compressed
payloads for all sampled messages against the desktop integer implementation.
Extra vectors cover empty messages, all byte values, UTF-8, and BOS padding.
Host tests also compare this codec to the project's existing arithmetic coder.

CPU cycle estimates are elapsed microseconds multiplied by the current clock
frequency, not hardware cycle-counter readings. Benchmarks use one core and
include normal interpreter/interrupt overhead. Payload bits per byte include
arithmetic termination and byte padding, but exclude original length,
checksums, and transport framing. The sampled source messages may overlap model
training, so these results are an implementation benchmark rather than a new
held-out compression-quality evaluation.

MicroPython's [Viper documentation](https://docs.micropython.org/en/latest/reference/speed_python.html#the-viper-code-emitter)
describes its machine-word integer arithmetic. The exporter bounds intermediate
products to avoid signed 32-bit overflow. The arithmetic coder deliberately
uses Python integers because interval products exceed 32 bits.

## Measured on the connected Pico W

MicroPython 1.26.1, RP2040 at its existing 125 MHz clock, one core:

| Measurement | Result |
| --- | ---: |
| Viper prediction, fixed BOS context | 27.26 ms/byte |
| Plain MicroPython prediction, same context | 644.13 ms/byte |
| Prediction speedup | 23.6x |
| Estimated prediction clock cycles | 3.41 million/byte |
| Sample encoding throughput | 37.02 original bytes/s |
| Sample decoding throughput | 36.22 original bytes/s |
| Model and persistent inference heap delta | 63,280 bytes |
| Free heap after benchmark | 101,584 bytes |
| Free flash after deployment | 208,896 bytes |

All 30 sampled messages (1,298 UTF-8 bytes) encoded to 524 payload bytes:
**3.2296 bits per original byte**, excluding external framing. On the same
messages, independently compressed gzip level 9 produced 11.1556 bits/byte and
zlib level 9 produced 8.9368 bits/byte. Those standard codecs have no pre-shared
trained model.

The original float64 neural predictor scored 3.13653 bits/byte on this sample;
the integer predictor scored 3.13682 bits/byte before arithmetic stream rounding.
All 30 payloads and all 32 CDF checks matched the desktop integer implementation.
Empty, UTF-8, and all-byte-value correctness vectors also passed.

Average stage times for the fixed context were 0.82 ms for embedding lookup,
1.79 ms for the first dense layer, 0.25 ms for the tanh lookup, 17.72 ms for the
output dense layer, and 6.89 ms for softmax/frequency allocation. These include
measurement overhead. The tanh lookup is under 1% of total prediction time.

See [benchmark-results.json](benchmark-results.json) for aggregate measurements,
firmware details, and model/source hashes. The original report, including
per-message lengths and timings, remains under ignored `models/`.
