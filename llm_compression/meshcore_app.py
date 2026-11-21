"""Basic MeshCore app: python -m llm_compression.meshcore_app --help."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
import json
from pathlib import Path
import sys

from .mesh_wire import CHANNEL, DATA_TYPE, MAX_DATAGRAM, SharedProfile, decode_message, encode_message
from .model import DEFAULT_CACHE, DEFAULT_MODEL, HuggingFacePredictor

DEFAULT_PROFILE = Path("llmcompression-profile.json")


def positive(value: str) -> float:
    result = float(value)
    if not 0 < result < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def tcp_address(value: str) -> tuple[str, int]:
    try:
        host, port = value.rsplit(":", 1)
        host = host.removeprefix("[").removesuffix("]")
        port = int(port)
        if not host or not 1 <= port <= 65535:
            raise ValueError
        return host, port
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use HOST:PORT, for example 192.168.0.1:5051") from exc


def _connection_options(command: argparse.ArgumentParser) -> None:
    connection = command.add_mutually_exclusive_group(required=True)
    connection.add_argument("--tcp", type=tcp_address, metavar="HOST:PORT")
    connection.add_argument("--serial", metavar="DEVICE", help="USB serial device, e.g. /dev/ttyUSB0 or COM3")
    command.add_argument("--baudrate", type=int, default=115200)
    command.add_argument("--timeout", type=positive, default=10, help="companion command timeout in seconds")


def _profile_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    command.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    command.add_argument("--offline", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=f"Compressed binary MeshCore messages on {CHANNEL}; flood only.")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("gui", help="open the desktop messaging app (also the default with no arguments)")
    profile = commands.add_parser("profile", help="create a fixed profile to share with every participant")
    profile.add_argument("--output", type=Path, default=DEFAULT_PROFILE)
    profile.add_argument("--model", default=DEFAULT_MODEL)
    profile.add_argument("--revision")
    profile.add_argument("--context-tokens", type=int, default=256)
    profile.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    profile.add_argument("--offline", action="store_true")
    inspect = commands.add_parser("inspect", help="read device and channel information without changing settings")
    _connection_options(inspect)
    send = commands.add_parser("send", help="send one compressed group datagram")
    _connection_options(send)
    _profile_options(send)
    content = send.add_mutually_exclusive_group(required=True)
    content.add_argument("--text", help="message text")
    content.add_argument("--file", type=Path, help="UTF-8 message file (preserves newlines)")
    send.add_argument("--region", required=True, help="hash region such as #au-nsw-ntl; * explicitly forces unscoped flood")
    send.add_argument("--dry-run", action="store_true", help="compress and verify without connecting or transmitting")
    listen = commands.add_parser("listen", help="receive and decompress group datagrams until Ctrl+C")
    _connection_options(listen)
    _profile_options(listen)
    listen.add_argument("--duration", type=positive, help="stop after this many seconds")
    listen.add_argument("--count", type=int, help="stop after this many valid messages")
    listen.add_argument("--json", action="store_true", help="output received messages as JSON Lines")
    return root


async def _receive(radio, predictor, executor, args) -> int:
    loop = asyncio.get_running_loop()
    received = 0
    deadline = loop.time() + args.duration if args.duration else None
    radio.subscribe_messages()
    poller = asyncio.create_task(radio.poll())
    disconnected = asyncio.create_task(radio.disconnected.wait())
    try:
        while args.count is None or received < args.count:
            timeout = max(0, deadline - loop.time()) if deadline is not None else None
            pending = asyncio.create_task(radio.incoming.get())
            try:
                done, _ = await asyncio.wait({pending, poller, disconnected}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if poller in done:
                    await poller
                    raise ConnectionError("Message polling stopped")
                if disconnected in done:
                    raise ConnectionError("MeshCore companion disconnected")
                if pending not in done:
                    break
                packet, reception = pending.result()
            finally:
                if not pending.done():
                    pending.cancel()
                    with suppress(asyncio.CancelledError):
                        await pending
            try:
                text = await loop.run_in_executor(executor, decode_message, packet, predictor)
            except (ValueError, RuntimeError, UnicodeError) as exc:
                print(f"Discarded datagram: {exc}", file=sys.stderr)
                continue
            received += 1
            if args.json:
                print(json.dumps({"channel": CHANNEL, "text": text, "wire_bytes": len(packet), **reception}, ensure_ascii=True), flush=True)
            else:
                print(f"[{CHANNEL}; {len(packet)} bytes; SNR {reception['snr']}; hops {reception['hops']}] {text}", flush=True)
    finally:
        for task in (poller, disconnected):
            task.cancel()
        await asyncio.gather(poller, disconnected, return_exceptions=True)
        if radio.dropped:
            print(f"Dropped {radio.dropped} datagrams because the decode queue filled.", file=sys.stderr)
    return received


async def run(args) -> int:
    # Keep the optional MeshCore dependency out of the standalone compressor.
    from .mesh_radio import MeshRadio, normalize_region

    if args.baudrate <= 0:
        raise ValueError("Baud rate must be positive")
    if args.command == "listen" and args.count is not None and args.count <= 0:
        raise ValueError("Receive count must be positive")
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-mesh") as executor:
        loop = asyncio.get_running_loop()
        predictor = None
        prepared = None
        if args.command != "inspect":
            if args.command == "send":
                args.region = normalize_region(args.region)
                data = args.file.read_bytes() if args.file else args.text.encode("utf-8")
                text = data.decode("utf-8")
            profile = SharedProfile.load(args.profile)
            print(f"Loading shared profile {profile.fingerprint[:16]}...", file=sys.stderr)
            predictor = await loop.run_in_executor(executor, partial(profile.load_predictor, args.cache_dir, args.offline))
            if args.command == "send":
                prepared = await loop.run_in_executor(executor, encode_message, text, predictor)
                restored = await loop.run_in_executor(executor, decode_message, prepared.packet, predictor)
                if restored != text:
                    raise ValueError("Local message round-trip failed")
                print(f"{prepared.text_bytes} UTF-8 bytes -> {len(prepared.packet)}/{MAX_DATAGRAM} datagram bytes "
                      f"({prepared.arithmetic_bytes} arithmetic); scope {args.region}; flood.", file=sys.stderr)
                if args.dry_run:
                    print("Dry run: exact local round-trip; no connection or transmission.")
                    return 0
        radio = await MeshRadio.connect(tcp=args.tcp, serial=args.serial, baudrate=args.baudrate, timeout=args.timeout)
        try:
            if args.command == "inspect":
                slots = await radio.channels()
                print(json.dumps({
                    "name": radio.client.self_info.get("name"),
                    "device": {key: value for key, value in radio.device.items() if key != "ble_pin"},
                    "channels": [{"index": slot["channel_idx"], "name": slot.get("channel_name", "")} for slot in slots],
                }, indent=2))
                return 0
            index = await radio.ensure_channel()
            print(f"Connected to {radio.client.self_info.get('name', 'MeshCore')}; {CHANNEL} in slot {index}.", file=sys.stderr)
            if args.command == "send":
                await radio.send(prepared.packet, args.region)
                print("Companion accepted one binary flood datagram. Group messages have no delivery acknowledgment.")
                return 0
            print(f"Listening for application 0x{DATA_TYPE:04x} on {CHANNEL}...", file=sys.stderr)
            await _receive(radio, predictor, executor, args)
            return 0
        finally:
            await radio.close()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv or ["gui"])
    try:
        if args.command == "gui":
            return gui_main([])
        if args.command == "profile":
            if args.output.exists():
                raise ValueError(f"Profile already exists: {args.output}; choose a new output path")
            predictor = HuggingFacePredictor(args.model, args.revision, args.context_tokens, args.cache_dir, args.offline)
            profile = SharedProfile.create(predictor)
            profile.save(args.output)
            print(f"Saved shared profile to {args.output}; SHA-256 {profile.fingerprint}")
            print("Share this exact file with every participant before sending. The profile is never transmitted over MeshCore.")
            return 0
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def gui_main(argv=None) -> int:
    try:
        from .mesh_gui import main as launch
    except ImportError as exc:
        print(f"Cannot load desktop GUI: {exc}. Install Python's Tk support (python3-tk on Debian/Ubuntu).", file=sys.stderr)
        return 1
    return launch(argv)


if __name__ == "__main__":
    raise SystemExit(main())
