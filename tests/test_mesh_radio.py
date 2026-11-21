import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

meshcore = pytest.importorskip("meshcore")
from meshcore import EventType
from meshcore.events import Event

from llm_compression.mesh_radio import MeshRadio, normalize_region
from llm_compression.mesh_wire import CHANNEL, CHANNEL_KEY, DATA_TYPE


def slot(index, name="", key=b"\x00" * 16):
    return {"channel_idx": index, "channel_name": name, "channel_secret": key}


def fake_radio(slots):
    commands = SimpleNamespace(
        get_channel=AsyncMock(side_effect=lambda index: Event(EventType.CHANNEL_INFO, slots[index])),
        set_channel=AsyncMock(return_value=Event(EventType.OK, {})),
        set_flood_scope=AsyncMock(return_value=Event(EventType.OK, {})),
        send_channel_data=AsyncMock(return_value=Event(EventType.OK, {})),
    )
    client = SimpleNamespace(commands=commands, subscribe=lambda kind, callback: callback)
    radio = MeshRadio(client)
    radio.device = {"max_channels": len(slots)}
    return radio


@pytest.mark.parametrize("name,expected", [("sydney", "#sydney"), ("#au-nsw-ntl", "#au-nsw-ntl"), ("*", "*")])
def test_scope_names(name, expected):
    assert normalize_region(name) == expected


@pytest.mark.parametrize("name", ["", "0", "None", "#", " has spaces", "two words", "é", "x" * 65])
def test_bad_scope_names(name):
    with pytest.raises(ValueError):
        normalize_region(name)


def test_existing_channel_reused_without_mutation():
    async def run():
        radio = fake_radio([slot(0, "Public"), slot(1, CHANNEL, CHANNEL_KEY)])
        assert await radio.ensure_channel() == 1
        radio.client.commands.set_channel.assert_not_awaited()
    asyncio.run(run())


def test_free_slot_is_used_and_public_is_untouched():
    async def run():
        slots = [slot(0, "Public"), slot(1)]
        radio = fake_radio(slots)
        async def set_channel(index, name, key):
            slots[index] = slot(index, name, key)
            return Event(EventType.OK, {})
        radio.client.commands.set_channel.side_effect = set_channel
        assert await radio.ensure_channel() == 1
        assert slots[0]["channel_name"] == "Public"
        radio.client.commands.set_channel.assert_awaited_once_with(1, CHANNEL, CHANNEL_KEY)
    asyncio.run(run())


def test_wrong_key_and_full_channel_table_are_not_overwritten():
    async def run():
        for slots in ([slot(0, "Public"), slot(1, CHANNEL, b"x" * 16)], [slot(0, "Public"), slot(1, "Other")]):
            radio = fake_radio(slots)
            with pytest.raises(ValueError):
                await radio.ensure_channel()
            radio.client.commands.set_channel.assert_not_awaited()
    asyncio.run(run())


def test_send_forces_flood_and_cleans_up_transient_scope():
    async def run():
        radio = fake_radio([])
        radio.channel_index = 4
        await radio.send(b"binary", "au-nsw-ntl")
        radio.client.commands.send_channel_data.assert_awaited_once_with(4, DATA_TYPE, b"binary", path=None)
        assert [call.args for call in radio.client.commands.set_flood_scope.await_args_list] == [("#au-nsw-ntl",), (None,)]
    asyncio.run(run())


def test_scope_failure_prevents_any_transmission():
    async def run():
        radio = fake_radio([])
        radio.channel_index = 1
        radio.client.commands.set_flood_scope.return_value = Event(EventType.ERROR, {"error_code": 1})
        with pytest.raises(RuntimeError, match="scope"):
            await radio.send(b"binary", "#test")
        radio.client.commands.send_channel_data.assert_not_awaited()
    asyncio.run(run())


def test_send_failure_still_restores_default_scope():
    async def run():
        radio = fake_radio([])
        radio.channel_index = 1
        radio.client.commands.send_channel_data.return_value = Event(EventType.ERROR, {"error_code": 3})
        with pytest.raises(RuntimeError):
            await radio.send(b"binary", "*")
        radio.client.commands.set_flood_scope.assert_awaited_with(None)
    asyncio.run(run())


def test_receive_filters_channel_application_and_direct_packets():
    async def run():
        radio = fake_radio([])
        radio.channel_index = 1
        radio.subscribe_messages()
        callback = radio.subscriptions[-1]
        data = {"channel_idx": 1, "data_type": DATA_TYPE, "path_len": 2, "payload": "000100000040", "data_len": 6, "SNR": 3.5}
        for change in ({"channel_idx": 2}, {"data_type": 123}, {"path_len": 255}, {"data_len": 7}, {"payload": "bad"}):
            callback(Event(EventType.CHANNEL_DATA_RECV, {**data, **change}))
        assert radio.incoming.empty()
        callback(Event(EventType.CHANNEL_DATA_RECV, data))
        packet, reception = radio.incoming.get_nowait()
        assert packet == bytes.fromhex(data["payload"])
        assert reception == {"snr": 3.5, "hops": 2}
    asyncio.run(run())


def test_usb_and_tcp_use_the_correct_meshcore_factories(monkeypatch):
    async def run():
        client = SimpleNamespace(
            commands=SimpleNamespace(send_device_query=AsyncMock(return_value=Event(EventType.DEVICE_INFO, {"max_channels": 8}))),
            subscribe=lambda *args: None,
        )
        tcp = AsyncMock(return_value=client)
        serial = AsyncMock(return_value=client)
        monkeypatch.setattr("llm_compression.mesh_radio.MeshCore.create_tcp", tcp)
        monkeypatch.setattr("llm_compression.mesh_radio.MeshCore.create_serial", serial)
        await MeshRadio.connect(tcp=("localhost", 5051))
        await MeshRadio.connect(serial="/dev/ttyUSB0", baudrate=57600)
        assert tcp.await_args.args == ("localhost", 5051)
        assert serial.await_args.args == ("/dev/ttyUSB0", 57600)
    asyncio.run(run())
