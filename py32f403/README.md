# PY32F403 neural compression firmware

Native, bare-metal C firmware for **PY32F403V1DT6**, Cortex-M4, 384 KiB flash,
64 KiB SRAM. Python scripts build, program, verify, and benchmark it through
ST-Link and the existing Raspberry Pi Pico W. The Pico retains MicroPython.

The implementation supports the Pico's integer `MCPNN001` (one hidden layer)
and `MCPNN002`/`MCPNN003` (two/three hidden layers) codecs. New training defaults
to three layers. The existing device measurements use one hidden layer in
32 KiB and 64 KiB/context-3 models. **Weights, biases, and lookup tables live in
read-only internal flash**: 50,140 bytes for the 32 KiB model and 83,084 bytes for
the 64 KiB model. They are never copied to RAM. The device currently runs the
64 KiB model; the original build and measurements are preserved below.
All dense layers use the Cortex-M4 `SMLAD` dual multiply-accumulate instruction.
The second hidden layer uses one additional `int16_t[hidden]` RAM buffer, and
the third reuses the first layer's buffer. The linker continues to enforce the 64 KiB SRAM limit, including the reserved stack.
Tanh and exponentials use the same integer tables as the Pico; no floating point,
heap allocation, operating system, or neural runtime library is required.

## Wiring

| Pico W | PY32 |
| --- | --- |
| GPIO0, UART0 TX | PA10, USART1 RX |
| GPIO1, UART0 RX | PA9, USART1 TX |
| GND | GND |

Use 3.3 V logic. This mapping was confirmed by passively observing the PY32's
boot-banner activity on Pico GPIO1 before enabling the Pico transmitter.
USART1 uses AF2 and 115200 baud, 8 data bits, no parity, one stop bit. ST-Link
connects through the PY32 programming header using SWD. The USB connection to
the Pico is `/dev/ttyACM0` on the tested machine.

## Build and verify

Run these commands from the repository root. Requirements: Python, the GNU Arm
Embedded compiler (`arm-none-eabi-gcc`, `objcopy`, `size`), OpenOCD 0.12 or later,
and `pyserial` for the USB bridge. Host verification additionally needs `gcc`.
The build itself uses only Python's standard library.

```bash
# Existing Pico export; creating it initially also needs NumPy and source weights.
.venv-cuda/bin/python pico/prepare_model.py

# Fetch the pinned device SDK if absent and build both firmware and RAM loader.
.venv-cuda/bin/python py32f403/build.py --fetch-sdk

# Check portable C against the desktop/Pico probability and payload vectors.
.venv-cuda/bin/python py32f403/verify_host.py

# Back up current flash, program only image pages, read back every byte, reset.
.venv/bin/python py32f403/flash.py

# Compare hardware CDFs and payloads, then measure DWT hardware cycle counts.
.venv/bin/python py32f403/benchmark.py
```

The source model defaults to `models/pico-32k-context3/model.pnn`. Use
`build.py --model PATH --output PATH` for a different exported model;
`--sdk PATH` selects an already-fetched SDK. The linker checks the entire RAM
budget, including an 8 KiB stack reservation. MCU accumulator checks reject
models whose integer intermediates could overflow. Context/embedding/hidden
dimensions are generated at build time, so the firmware carries no generic
maximum-sized buffers for unused model dimensions.

Generated files go into ignored `py32f403/build/`:

- `nn-py32f403.elf`, `.bin`, `.map`: executable, flash image, and memory map.
- `model.c`, `model.h`: flash-resident trained weights/tables and dimensions.
- `flash-loader.elf`: a small RAM-resident PY32 page programmer.
- `build.json`, `flash-report.json`, `benchmark-results.json`: provenance and results.

The SDK is pinned to
[`982757404a416dd1e0fde921750d61167aed09ee`](https://github.com/OpenPuya/PY32F4xx_Firmware/tree/982757404a416dd1e0fde921750d61167aed09ee)
under `.cache/py32f4xx-firmware`. The build retains Puya's factory-trim
initialization. Its sole source adaptation replaces the HAL reset wrapper with
the equivalent CMSIS `NVIC_SystemReset()`, avoiding a full HAL dependency.

## Memory layout and measured results

The model is the same one tested on the Pico: context 3, embedding dimension 8,
108 hidden units, 30,240 dense multiply-accumulates per predicted byte.

| Resource | Bytes |
| --- | ---: |
| Complete firmware flash image | 57,724 |
| Model tensors/tables within that image | 50,140 |
| NN working state | 2,836 |
| All static RAM, including UART/message buffers | 6,696 |
| Reserved stack budget | 8,192 |
| Total reserved RAM | **14,888 of 65,536** |
| Conservative measured stack high-water | 608 |

The NN unions logits, scores, frequencies, and the final CDF in a single
reused buffer. Hidden activations and byte embeddings use int16 buffers;
frequency residuals use uint32 and sort indices use uint16. The inference state
includes context history and all scratch space. The application adds a bounded
UART command line, a 512-byte message buffer, and a 1,100-byte payload buffer.

On the connected PY32, using nominal HSI 8 MHz x18 PLL = **144 MHz**, one core:

| Benchmark | Result |
| --- | ---: |
| Prediction, 200 repetitions at BOS context | **461,066 cycles/byte** |
| Prediction time | **3.202 ms/byte** |
| MeshCore sample encoding | **333.73 original bytes/s** |
| MeshCore sample decoding | **332.92 original bytes/s** |
| Payload compression | **3.2296 bits/original byte** |

The 30-message sample contains 1,298 UTF-8 bytes and compresses to 524 payload
bytes, exactly matching the Pico output. All 32 CDF vectors and 34 encode/decode
vectors passed, including empty input, UTF-8, and every byte value. The host C
checks run with the undefined-behavior sanitizer. Hardware times use the DWT
cycle counter and exclude UART transfer and host/Pico bridge overhead. The HSI
frequency is nominal; it has not been measured with an external frequency counter.
The stack estimate conservatively includes a 512-byte unpainted safety margin.

For comparison, the earlier Pico MicroPython implementation took 27.26 ms per
prediction and encoded at 37.02 bytes/s. This native C implementation's fixed
context prediction is about 8.5x faster and its sample encoding about 9x faster;
the MCU, CPU frequency, language, memory placement, and kernels all differ.

[benchmark-results.json](benchmark-results.json) contains aggregate measurements
and hashes without private messages. Per-message lengths/timings remain in the
ignored build directory. The sample may overlap training: it measures this port,
not fresh held-out model quality. Payload sizes exclude original length,
integrity checks, and radio framing; the model is pre-shared.

## Reproduce the 64 KiB/context-3 run

The 64 KiB model uses 224 hidden units instead of 108, keeping context 3 and
embedding dimension 8. Its firmware occupies 90,672 flash bytes and reserves
15,120 RAM bytes including the same 8 KiB stack budget. NN working state grows
from 2,836 to 3,068 bytes. The larger weights and lookup tables remain in flash.
Both model sizes are benchmarked on the same 30 messages and correctness inputs.

```bash
.venv-cuda/bin/python pico/prepare_model.py \
  --model models/nn-cuda-sweep-mc-small/64k-context3/model.mcnn \
  --output models/pico-64k-context3
.venv-cuda/bin/python py32f403/build.py \
  --model models/pico-64k-context3/model.pnn --output py32f403/build/64k-context3
.venv-cuda/bin/python py32f403/verify_host.py \
  --build py32f403/build/64k-context3 --vectors models/pico-64k-context3/vectors.json
.venv/bin/python py32f403/flash.py --build py32f403/build/64k-context3
.venv/bin/python py32f403/benchmark.py \
  --build py32f403/build/64k-context3 --vectors models/pico-64k-context3/vectors.json
```

Measured at the same nominal 144 MHz clock:

| Measurement | 32 KiB model | 64 KiB model |
| --- | ---: | ---: |
| Prediction cycles/byte | 461,066 | 787,100 |
| Prediction ms/byte | 3.202 | 5.466 |
| Encoding original bytes/s | 333.73 | 187.88 |
| Decoding original bytes/s | 332.92 | 187.79 |
| Payload bits/original byte | 3.2296 | **2.9399** |
| Sample payload bytes | 524 | **477** |
| NN working state bytes | 2,836 | 3,068 |
| All static RAM bytes | 6,696 | 6,928 |
| Total RAM reserved, including stack | 14,888 | 15,120 |
| Firmware flash bytes | 57,724 | 90,672 |

The larger model reduces this sample's payload by **8.97%**, while prediction
takes about 1.71x as long. It still uses under one quarter of available RAM,
including the stack budget. All 32 CDF checks and 34 encode/decode vectors
matched the desktop 64 KiB integer reference; flash readback matched exactly
and readable configuration remained unchanged.

The original 32 KiB build remains in `py32f403/build/`; the 64 KiB build is kept
separately in its `64k-context3/` subdirectory. Use the matching `--build` and
`--vectors` options when benchmarking the model currently on the device.
The [32 KiB baseline report](benchmark-results-32k-context3.json) is preserved;
[benchmark-results.json](benchmark-results.json) records the latest 64 KiB run.

## Programming and backups

OpenOCD's STM32 flash drivers are not suitable for the PY32 flash controller.
`openocd.cfg` provides only generic Cortex-M4 SWD access. `flash.py` loads a
small page programmer into SRAM; it follows Puya's 256-byte page erase/program
sequence and uses the internal HSI clock during programming. It programs only
pages covered by the image and never performs mass erase or option programming.

Every invocation backs up all 384 KiB flash plus readable configuration to a
timestamped directory under `.cache/py32-backup/`. It reads back the programmed
bytes and compares them directly, and checks configuration reads before/after.
The initial device firmware was preserved at
`.cache/py32-backup/original-flash.bin` and
`.cache/py32-backup/20261122-172843/flash.bin` on the development machine.
The backups and generated firmware are excluded from Git because they can
contain private application code or a model learned from private messages.

## UART command interface

The firmware remains running after programming and accepts newline-terminated
ASCII commands. Hex payloads contain bytes, so arbitrary binary data is supported.

| Command | Response |
| --- | --- |
| `I` | CPU clock, NN/static RAM bytes, stack high-water, model bytes/hash, kernel |
| `P 010001000100` | Cycles and full CDF for three BOS symbols (uint16 big-endian context input) |
| `B 200` | Iteration count and total prediction cycles, resetting context to BOS |
| `E 48656c6c6f` | Encode `Hello`: cycles, payload length, payload hex |
| `D LENGTH HEX` | Decode: cycles, recovered length, recovered bytes as hex |

`P` returns the 257 CDF entries as little-endian uint32 bytes in hex.
The transport is a development console; messages must fit the 512-byte limit.
Malformed command lengths/hex and undersized codec output buffers are rejected.
The arithmetic format relies on external original-length and integrity checks,
as with the Pico demo. Compressed payloads are compatible with that integer
runtime, but not the original floating-point `.mcnn` codec.

Device pin/clock/flash definitions come from the pinned SDK and the
[Puya PY32F403 datasheet](https://www.puyasemi.com/download_path/%E6%95%B0%E6%8D%AE%E6%89%8B%E5%86%8C/MCU/PY32F403_Datasheet_V1.8.pdf).
The SDK uses BSD-3-Clause licensing, with CMSIS under Apache-2.0; the build copies
the relevant license files alongside its generated artifacts.
