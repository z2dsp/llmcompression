# Compressed MeshCore messages

The Python client connects over USB serial or TCP and sends/receives binary messages on **`#llmcompression`**. Sending always uses flood routing. Specify a region on every send, or explicitly use `*` for an unscoped flood.

## Setup and shared profile

Use the Python environment from [README.md](README.md), then install the optional client dependency:

```bash
.venv/bin/python -m pip install -e '.[mesh,test]'
```

## Desktop app

Launch the GUI:

```bash
.venv/bin/python -m llm_compression.meshcore_app
# Or, after installing the project:
.venv/bin/llm-mesh-gui
```

Launching `llm-mesh` without arguments also opens the GUI. The `send`, `listen`, `inspect`, and `profile` subcommands remain available for scripts, including `send_every_60s.sh`.

The window provides:

- **TCP or USB:** host/port, a refreshable serial-device list (or manually entered device), baud rate, and command timeout. The TCP default is `192.168.0.1:5051`.
- **Shared profile:** choose a profile JSON file, set the model cache folder, and enable or disable offline-only loading. **Create shared profile…** lets you choose the Hugging Face model, revision, and context size without using the command line. Disable offline mode to download a model that is not yet cached.
- **Connect / Disconnect:** load and validate the model, connect to the companion, and start receiving automatically. Connection/profile controls are locked while connected; region scope remains editable for each send.
- **Conversation and composer:** display incoming and outgoing messages, open a UTF-8 file as a message, and send with the button or **Ctrl+Enter** (Cmd+Enter on macOS). Enter inserts a newline. The channel is fixed to `#llmcompression`, and transport is always flood.
- **Region scope:** enter a hash region or choose `*` for explicitly unscoped flooding. The default test region is `#llmcompression`; choose the region you intend to use.
- **Check size:** compress and verify the draft locally without transmitting. A successful send displays the compressed byte count and companion acceptance, which does not confirm remote delivery. Failed sends preserve the draft.
- **Save settings:** remember connection, profile, cache, offline mode, and region locally in `.cache/mesh-gui-settings.json`. Settings are restored on the next launch; the app does not automatically connect. Conversation history stays in memory unless you choose **Save chat…**.

The model stays loaded for the connected session. One background asyncio loop handles the radio, and one dedicated inference worker serializes complete compression/decompression operations. The GUI remains responsive while loading models, sending, and receiving. Closing disconnects the radio and waits for any model operation already running to finish.

The GUI uses Python's Tkinter toolkit. Check availability with `python -m tkinter`. On Linux, Tk support may need an OS package such as `python3-tk`; the virtual environment must use a Python interpreter built with Tk support. The CLI still works on systems without a desktop display.

## Shared profiles and CLI

Commands below use `python -m llm_compression.meshcore_app`; the installed `llm-mesh` command is equivalent.

The included `llmcompression-profile.json` pins this workspace's SmolLM2-135M model and runtime. Copy the **same profile file** to every participant before messaging. It includes the model/tokenizer commit, context size, integer probability scheme, inference settings, runtime versions, and a calibration digest of actual integer probability tables. These are checked locally and never sent over the radio. The displayed profile fingerprint is also local only.

To create a profile for a fresh setup:

```bash
.venv/bin/python -m llm_compression.meshcore_app profile --output shared-profile.json
```

Use `--offline` if the model is already cached. Existing profile files are never overwritten. To swap models, add `--model MODEL_ID`; use `--revision COMMIT` to choose an explicit checkpoint. Share the resulting file and pass `--profile shared-profile.json` to both sender and receiver.

Participants need matching Python/library versions and CPU environments. Calibration catches many numerical differences, and checksums catch incorrect reconstructions, but matching a short calibration does not guarantee every later probability table matches. See [README.md](README.md) for the reproducibility limits of floating-point model inference.

## TCP commands

Inspect the device without changing settings or transmitting:

```bash
.venv/bin/python -m llm_compression.meshcore_app inspect --tcp 192.168.0.1:5051
```

Send within your region:

```bash
.venv/bin/python -m llm_compression.meshcore_app send \
    --tcp 192.168.0.1:5051 --region '#au-nsw-ntl' \
    --text 'The weather is pleasant today.' --offline
```

Replace the example region with your actual region. An omitted `#` prefix is added automatically; names are otherwise preserved exactly. Quote `#...` and `*` in shell commands.

Send a UTF-8 file, preserving all its bytes:

```bash
.venv/bin/python -m llm_compression.meshcore_app send \
    --tcp 192.168.0.1:5051 --region '#au-nsw-ntl' \
    --file examples/english.txt --offline
```

Add **`--dry-run`** to compress and verify without opening a connection, changing device settings, or transmitting. The 1,259-byte example fits in **155 application datagram bytes**: 149 arithmetic bytes plus 6 framing bytes.

Receive until Ctrl+C:

```bash
.venv/bin/python -m llm_compression.meshcore_app listen \
    --tcp 192.168.0.1:5051 --offline
```

Use `--count 1` to stop after one valid message, `--duration 30` to limit the listening interval, or `--json` for JSON Lines output. Duration starts after loading the model and connecting. Region selection is a send option: received packets must already have reached the companion under its network/region configuration.

## Send every 60 seconds

From the project folder, run:

```bash
./send_every_60s.sh --tcp 192.168.0.1:5051 --region '#au-nsw-ntl' \
    --text 'The weather is pleasant today.' --offline
```

Choose your region and message. The script sends immediately and starts subsequent attempts 60 seconds apart, accounting for model loading and sending time. If an attempt takes longer than 60 seconds, the next begins when it finishes; attempts never overlap. Failed sends are logged and retried at the next interval. Stop with Ctrl+C.

The script accepts the same arguments as `send`, including USB `--serial`, `--file`, `--profile`, and `--dry-run`. Paths are resolved relative to the project folder. Each attempt starts the Python client and reloads the model.

## USB serial

Replace `--tcp HOST:PORT` with `--serial DEVICE`:

```bash
.venv/bin/python -m llm_compression.meshcore_app send \
    --serial /dev/ttyUSB0 --baudrate 115200 --region '#au-nsw-ntl' \
    --text 'Hello over USB.' --offline

.venv/bin/python -m llm_compression.meshcore_app listen \
    --serial /dev/ttyUSB0 --offline
```

On Windows use a device such as `COM3` and the virtual environment's `Scripts/python.exe`.

Sending/listening reuses an existing correctly keyed `#llmcompression` slot or configures the first empty slot after slot 0. Existing channels are not overwritten. A mismatched key or full channel table is reported as an error.

## Minimal wire format

The firmware sends `PAYLOAD_TYPE_GRP_DATA` through `CMD_SEND_CHANNEL_DATA` (`62`), with experimental application `data_type=0xFFFF`. MeshCore carries the application type and payload length; our application payload is:

| Field | Bytes | Purpose |
| --- | --- | --- |
| Random nonce | 2 | Distinguish repeated identical messages for mesh duplicate suppression |
| Token count | 1–2, unsigned LEB128 | Tell the arithmetic decoder when to stop |
| CRC-16/CCITT-FALSE | 2, little-endian | Check nonce + encoded token count + original UTF-8 bytes |
| Arithmetic stream | Remaining bytes | Compressed tokens, termination, and byte padding |

Framing costs **5 bytes for fewer than 128 tokens**, otherwise 6 bytes. There are no model names, JSON headers, profile IDs, timestamps, or sender labels on the air. The sole shared profile defines the format. Every message starts a fresh model context; packet loss cannot desynchronize later messages. The receiver also checks that the arithmetic stream is canonical.

The two-byte nonce trades size against collision probability: two sends of identical text can still collide with probability 1/65,536. It does not authenticate the sender. This is a publicly derivable hash-channel, and received messages carry no authenticated sender identity.

These sizes exclude MeshCore's application type/length fields, channel hash, MAC, encryption padding, routing header/path, region transport codes, and LoRa framing. Actual airtime depends on those fields and the radio configuration.

The app caps each payload at **163 bytes**, matching the pinned MeshCore Python library. Oversized messages fail before connection/transmission and report the required size; shorten the text. There is no fragmentation, plaintext fallback, direct send, or automatic retransmission. Decode limits are 2,048 tokens and 8,192 original UTF-8 bytes. Model work runs on one dedicated worker thread, with a bounded incoming queue so inference does not block transport processing.

## Flood regions and delivery

Each send sets the transient flood scope, then calls `send_channel_data(..., path=None)`. This encodes a flood path descriptor of `0xFF`; an empty path would request zero-hop direct transport and is never used. Named regions produce transport-scoped flood packets through the companion. `--region '*'` explicitly ignores any persisted default scope for that send.

If setting the scope fails, **nothing is sent**. After the send, the client resets the transient override to the device's configured default. It does not change the persisted region, radio parameters, or advertise the node. Scope is device-wide, so use a dedicated companion session; another connected application changing scope concurrently can interfere.

The companion must support binary channel data and flood scopes. Unsupported commands fail explicitly. Reception uses `RESP_CODE_CHANNEL_DATA_RECV` (`27`) and polls the companion queue, including previously queued messages. Queue synchronization also consumes nonmatching queued messages, which this single-channel app ignores. Text-only MeshCore clients will not display compressed datagrams; recipients need this app and the shared profile.

A successful send means **the companion accepted the datagram**. Group datagrams have no delivery acknowledgment, and the client does not synthesize a received message from a successful send.

## Verification

Desktop tests exercise connection controls, simultaneous sending/receiving through the TCP simulator, draft preservation, size checks without transmission, settings persistence, and clean shutdown. These tests need a working desktop display and are skipped when one is unavailable.

```bash
.venv/bin/python -m pytest -q
HF_HUB_OFFLINE=1 RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -q
```

Tests cover compact framing, corruption/truncation, limits, profile mismatches, free-slot selection, channel/application filtering, USB/TCP factory selection, scope failures, flood-only commands, and receive output. A TCP simulator exercises actual fragmented companion framing through the real MeshCore library, including scoped sending and binary reception. It is skipped when loopback sockets are unavailable; real-model tests are opt-in.

Live testing against `192.168.0.1:5051` successfully connected to `VK2DDS Compression`, configured slot 1 for `#llmcompression`, and submitted a 60-byte test message as one **22-byte compressed datagram**, scoped to region `#llmcompression`. This confirms companion acceptance; remote RF delivery requires a receiving peer. USB hardware was not attached for a physical test.

References: [MeshCore Python client](https://github.com/meshcore-dev/meshcore_py), [MeshCore payload format](https://github.com/meshcore-dev/MeshCore/blob/main/docs/payloads.md), and [OpenHop companion protocol](https://openhop-dev.github.io/openhop_core/companion/).
