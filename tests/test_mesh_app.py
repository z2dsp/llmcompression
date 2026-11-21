import asyncio
from concurrent.futures import Future
import json
from types import SimpleNamespace

import pytest

from llm_compression.meshcore_app import _receive, parser, tcp_address
from llm_compression.mesh_wire import encode_message
from test_mesh_wire import UniformPredictor


class InlineExecutor:
    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_receive_discards_bad_crc_then_displays_valid_text(capsys):
    async def run():
        valid = encode_message("Hello 🌊", UniformPredictor()).packet
        invalid = bytearray(valid)
        invalid[3] ^= 1
        incoming = asyncio.Queue()
        incoming.put_nowait((bytes(invalid), {"snr": 1.0, "hops": 2}))
        incoming.put_nowait((valid, {"snr": 1.0, "hops": 2}))
        async def poll():
            await asyncio.Event().wait()
        radio = SimpleNamespace(
            incoming=incoming, disconnected=asyncio.Event(), dropped=0,
            subscribe_messages=lambda: None, poll=poll,
        )
        args = SimpleNamespace(duration=None, count=1, json=True)
        assert await _receive(radio, UniformPredictor(), InlineExecutor(), args) == 1
    asyncio.run(run())
    output = capsys.readouterr()
    assert json.loads(output.out)["text"] == "Hello 🌊"
    assert "Discarded datagram" in output.err


def test_listen_duration_exits_cleanly():
    async def run():
        async def poll():
            await asyncio.Event().wait()
        radio = SimpleNamespace(
            incoming=asyncio.Queue(), disconnected=asyncio.Event(), dropped=0,
            subscribe_messages=lambda: None, poll=poll,
        )
        args = SimpleNamespace(duration=0.01, count=None, json=False)
        assert await _receive(radio, UniformPredictor(), InlineExecutor(), args) == 0
    asyncio.run(run())


def test_send_requires_region_and_exactly_one_connection():
    with pytest.raises(SystemExit):
        parser().parse_args(["send", "--tcp", "localhost:5051", "--text", "hello"])
    with pytest.raises(SystemExit):
        parser().parse_args(["send", "--tcp", "localhost:5051", "--serial", "COM3", "--text", "hello", "--region", "*"])


def test_tcp_addresses():
    assert tcp_address("192.168.0.1:5051") == ("192.168.0.1", 5051)
    assert tcp_address("[::1]:5051") == ("::1", 5051)
