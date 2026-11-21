"""MeshCore USB/TCP adapter, restricted to one channel and flood sends."""

import asyncio
from contextlib import suppress

from meshcore import EventType, MeshCore

from .mesh_wire import CHANNEL, CHANNEL_KEY, DATA_TYPE, MAX_DATAGRAM


def checked(event, expected, operation: str):
    if event is None:
        raise ConnectionError(f"{operation}: no response from companion")
    if event.type != expected:
        raise RuntimeError(f"{operation}: {event.type.value}: {event.payload}")
    return event.payload


def normalize_region(region: str) -> str:
    if region == "*":
        return region
    if not region or region.strip() != region or region in {"0", "None"}:
        raise ValueError("Use a region name such as #sydney, or * for explicitly unscoped flood")
    canonical = region if region.startswith("#") else "#" + region
    if len(canonical) < 2 or len(canonical.encode("ascii")) > 64 or any(c.isspace() for c in canonical):
        raise ValueError("Region must be an ASCII hash name of 2–64 bytes with no whitespace")
    return canonical


class MeshRadio:
    def __init__(self, client) -> None:
        self.client = client
        self.command_lock = asyncio.Lock()
        self.channel_index = None
        self.device = {}
        self.incoming = asyncio.Queue(maxsize=64)
        self.disconnected = asyncio.Event()
        self.subscriptions = []
        self.dropped = 0

    @classmethod
    async def connect(cls, *, tcp: tuple[str, int] | None = None, serial: str | None = None,
                      baudrate: int = 115200, timeout: float = 10) -> "MeshRadio":
        if (tcp is None) == (serial is None):
            raise ValueError("Choose exactly one TCP or USB serial connection")
        options = {"default_timeout": timeout, "only_error": True, "auto_reconnect": False}
        if tcp is not None:
            client = await asyncio.wait_for(MeshCore.create_tcp(*tcp, **options), timeout + 5)
        else:
            client = await asyncio.wait_for(MeshCore.create_serial(serial, baudrate, **options), timeout + 5)
        if client is None:
            raise ConnectionError("Companion handshake failed")
        radio = cls(client)
        radio.subscriptions.append(client.subscribe(EventType.DISCONNECTED, lambda event: radio.disconnected.set()))
        try:
            radio.device = checked(await client.commands.send_device_query(), EventType.DEVICE_INFO, "Query device")
        except BaseException:
            await radio.close()
            raise
        return radio

    async def channels(self) -> list[dict]:
        count = self.device.get("max_channels", 8)
        if type(count) is not int or not 1 <= count <= 256:
            raise ValueError("Companion advertised an invalid channel capacity")
        result = []
        async with self.command_lock:
            for index in range(count):
                result.append(checked(await self.client.commands.get_channel(index), EventType.CHANNEL_INFO, f"Read channel {index}"))
        return result

    async def ensure_channel(self) -> int:
        slots = await self.channels()
        empty = None
        for slot in slots:
            index = slot["channel_idx"]
            secret = slot.get("channel_secret", b"")
            if isinstance(secret, str):
                secret = bytes.fromhex(secret)
            if slot.get("channel_name") == CHANNEL:
                if secret != CHANNEL_KEY:
                    raise ValueError(f"Channel {index} has the right name but the wrong hash-channel key")
                self.channel_index = index
                return index
            if index > 0 and not slot.get("channel_name") and not any(secret) and empty is None:
                empty = index
        if empty is None:
            raise ValueError(f"No free channel slot for {CHANNEL}; existing channels were left intact")
        async with self.command_lock:
            checked(await self.client.commands.set_channel(empty, CHANNEL, CHANNEL_KEY), EventType.OK, "Configure hash channel")
            verified = checked(await self.client.commands.get_channel(empty), EventType.CHANNEL_INFO, "Verify hash channel")
        if verified.get("channel_name") != CHANNEL or verified.get("channel_secret") != CHANNEL_KEY:
            raise ValueError("Companion did not retain the expected hash-channel settings")
        self.channel_index = empty
        return empty

    async def send(self, payload: bytes, region: str) -> None:
        if self.channel_index is None:
            raise ValueError("Hash channel is not configured")
        if not 1 <= len(payload) <= MAX_DATAGRAM:
            raise ValueError("Invalid binary channel payload length")
        region = normalize_region(region)
        async with self.command_lock:
            checked(await self.client.commands.set_flood_scope(region), EventType.OK, "Set flood region scope")
            try:
                # None is critical: b'' would request a zero-hop DIRECT packet.
                checked(await self.client.commands.send_channel_data(
                    self.channel_index, DATA_TYPE, payload, path=None,
                ), EventType.OK, "Send binary flood datagram")
            finally:
                # Restore the device's configured default, never persist a scope.
                checked(await self.client.commands.set_flood_scope(None), EventType.OK, "Restore default flood scope")

    def subscribe_messages(self) -> None:
        if self.channel_index is None:
            raise ValueError("Hash channel is not configured")

        def received(event) -> None:
            data = event.payload
            if data.get("channel_idx") != self.channel_index or data.get("data_type") != DATA_TYPE:
                return
            if data.get("path_len") == 255:  # Companion's marker for a direct receive.
                return
            try:
                payload = data["payload"]
                payload = bytes.fromhex(payload) if isinstance(payload, str) else bytes(payload)
                if len(payload) != data.get("data_len") or not 6 <= len(payload) <= MAX_DATAGRAM:
                    return
                self.incoming.put_nowait((payload, {"snr": data.get("SNR"), "hops": data.get("path_len")}))
            except (KeyError, TypeError, ValueError):
                return
            except asyncio.QueueFull:
                self.dropped += 1

        self.subscriptions.append(self.client.subscribe(EventType.CHANNEL_DATA_RECV, received))

    async def poll(self) -> None:
        """Drain the companion queue; polling also catches missed push events."""
        while not self.disconnected.is_set():
            async with self.command_lock:
                event = await self.client.commands.get_msg()
            if event is None or event.type == EventType.ERROR:
                checked(event, EventType.NO_MORE_MSGS, "Receive messages")
            if event.type == EventType.NO_MORE_MSGS:
                await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(0)
        raise ConnectionError("MeshCore companion disconnected")

    async def close(self) -> None:
        for subscription in self.subscriptions:
            with suppress(Exception):
                self.client.unsubscribe(subscription)
        self.subscriptions.clear()
        await self.client.disconnect()
