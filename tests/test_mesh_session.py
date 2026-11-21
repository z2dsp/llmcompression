import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("meshcore")

from llm_compression.mesh_session import MeshSession, Settings
from llm_compression.mesh_wire import encode_message
from test_mesh_app import InlineExecutor
from test_mesh_wire import UniformPredictor


class FakeRadio:
    def __init__(self):
        self.client = SimpleNamespace(self_info={"name": "Test companion"})
        self.incoming = asyncio.Queue()
        self.disconnected = asyncio.Event()
        self.dropped = 0
        self.ensure_channel = AsyncMock(return_value=1)
        self.send = AsyncMock()
        self.close = AsyncMock()
        self.subscribed = False

    def subscribe_messages(self):
        self.subscribed = True

    async def poll(self):
        await asyncio.Event().wait()


def make_session(emit=None):
    radio = FakeRadio()
    predictor = UniformPredictor()
    profile = SimpleNamespace(
        predictor=predictor.metadata, fingerprint="a" * 64,
        load_predictor=lambda *args: predictor,
    )
    events = []
    callback = emit or (lambda kind, data: events.append((kind, data)))
    session = MeshSession(callback, InlineExecutor(), radio_factory=AsyncMock(return_value=radio), profile_loader=lambda path: profile)
    return session, radio, events


def test_persistent_session_send_check_and_receive():
    async def run():
        received = asyncio.Event()
        events = []
        def emit(kind, data):
            events.append((kind, data))
            if kind == "received":
                received.set()
        session, radio, _ = make_session(emit)
        await session.connect(Settings())
        assert session.connected and radio.subscribed
        await session.send("Hello", "test-region", dry_run=True)
        radio.send.assert_not_awaited()
        await session.send("Hello", "test-region")
        radio.send.assert_awaited_once()
        assert radio.send.await_args.args[1] == "#test-region"
        packet = encode_message("Reply 🌊", UniformPredictor()).packet
        radio.incoming.put_nowait((packet, {"snr": 3.5, "hops": 2}))
        await asyncio.wait_for(received.wait(), 1)
        assert next(data for kind, data in events if kind == "received")["text"] == "Reply 🌊"
        assert [kind for kind, data in events].count("connected") == 1
        await session.disconnect()
        assert not session.connected
        radio.close.assert_awaited_once()
        assert session.predictor is None
    asyncio.run(run())


def test_corrupt_receive_does_not_stop_session():
    async def run():
        received = asyncio.Event()
        kinds = []
        def emit(kind, data):
            kinds.append(kind)
            if kind == "received":
                received.set()
        session, radio, _ = make_session(emit)
        await session.connect(Settings())
        bad = bytearray(encode_message("hello", UniformPredictor()).packet)
        bad[3] ^= 1
        radio.incoming.put_nowait((bytes(bad), {}))
        radio.incoming.put_nowait((encode_message("valid", UniformPredictor()).packet, {}))
        await asyncio.wait_for(received.wait(), 1)
        assert "warning" in kinds
        assert session.connected
        await session.disconnect()
    asyncio.run(run())


def test_disconnect_during_compression_prevents_transmission():
    async def run():
        session, radio, _ = make_session()
        await session.connect(Settings())
        started, finish = asyncio.Event(), asyncio.Event()
        original = session.model_work
        async def delayed(function, *args):
            started.set()
            await finish.wait()
            return await original(function, *args)
        session.model_work = delayed
        sending = asyncio.create_task(session.send("Hello", "#test"))
        await started.wait()
        closing = asyncio.create_task(session.disconnect())
        await asyncio.sleep(0)
        finish.set()
        with pytest.raises(ConnectionError, match="not sent"):
            await sending
        await closing
        radio.send.assert_not_awaited()
        radio.close.assert_awaited_once()
    asyncio.run(run())


def test_disconnect_and_connection_loss_cleanup_once():
    async def run():
        session, radio, _ = make_session()
        await session.connect(Settings())
        radio.disconnected.set()
        await asyncio.gather(session.disconnect(), session.disconnect())
        assert not session.connected
        radio.close.assert_awaited_once()
    asyncio.run(run())


def test_channel_setup_failure_closes_connection():
    async def run():
        session, radio, _ = make_session()
        radio.ensure_channel.side_effect = ValueError("No free channel")
        with pytest.raises(ValueError, match="No free channel"):
            await session.connect(Settings())
        radio.close.assert_awaited_once()
        assert session.radio is None
    asyncio.run(run())


def test_failed_send_can_be_retried_and_does_not_stop_receiving():
    async def run():
        session, radio, _ = make_session()
        await session.connect(Settings())
        radio.send.side_effect = RuntimeError("queue full")
        with pytest.raises(RuntimeError, match="queue full"):
            await session.send("hello", "#test")
        assert session.connected
        radio.send.side_effect = None
        await session.send("second", "#test")
        await session.disconnect()
    asyncio.run(run())


def test_settings_round_trip(tmp_path):
    settings = replace(Settings(), region="#test", transport="USB", serial="/dev/ttyACM0")
    path = tmp_path / "settings.json"
    settings.save(path)
    assert Settings.load(path) == settings


@pytest.mark.parametrize("changes", [
    {"port": 0}, {"port": 65536}, {"port": True}, {"timeout": float("nan")},
    {"timeout": -1}, {"region": ""}, {"transport": "BLE"}, {"offline": "yes"},
    {"transport": "USB", "serial": ""}, {"profile": ""}, {"cache_dir": 42},
])
def test_invalid_settings_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(Settings(), **changes).validate()
