"""Exercise real MeshCore TCP framing against an in-process companion."""

import asyncio
from contextlib import suppress
import struct

import pytest

pytest.importorskip("meshcore")

from llm_compression.mesh_radio import MeshRadio
from llm_compression.mesh_wire import CHANNEL, CHANNEL_KEY, DATA_TYPE, decode_message, encode_message
from test_mesh_wire import UniformPredictor


class CompanionServer:
    def __init__(self):
        self.channels = [("Public", b"p" * 16), ("", b"\x00" * 16), ("Other", b"o" * 16)]
        self.commands = []
        self.messages = []
        self.finished = asyncio.Event()

    async def handle(self, reader, writer):
        try:
            while True:
                header = await reader.readexactly(3)
                assert header[0] == 0x3C
                command = await reader.readexactly(int.from_bytes(header[1:], "little"))
                self.commands.append(command)
                response = self.respond(command)
                frame = b">" + len(response).to_bytes(2, "little") + response
                # Split within the serial/TCP frame header and payload.
                for part in (frame[:2], frame[2:5], frame[5:]):
                    writer.write(part)
                    await writer.drain()
                    await asyncio.sleep(0)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self.finished.set()

    def respond(self, command):
        code = command[0]
        if code == 1:
            return (b"\x05\x01\x14\x16" + b"k" * 32 + b"\x00" * 12
                    + struct.pack("<IIBB", 915075, 125000, 9, 5) + b"TCP simulator")
        if code == 22:
            return b"\x0d\x0e\x10\x03" + b"\x00" * 4 + b"test".ljust(12, b"\x00") + b"simulator".ljust(40, b"\x00") + b"14.0".ljust(20, b"\x00") + b"\x00\x00"
        if code == 31:
            index = command[1]
            name, key = self.channels[index]
            return bytes([18, index]) + name.encode().ljust(32, b"\x00") + key
        if code == 32:
            self.channels[command[1]] = (command[2:34].rstrip(b"\x00").decode(), command[34:50])
            return b"\x00"
        if code == 54:
            return b"\x00"
        if code == 62:
            assert command[2] == 0xFF, "A binary channel send must use flood"
            self.messages.append(command[5:])
            return b"\x00"
        if code == 10:
            if not self.messages:
                return b"\x0a"
            payload = self.messages.pop(0)
            return bytes([27, 8, 0, 0, 1, 2]) + DATA_TYPE.to_bytes(2, "little") + bytes([len(payload)]) + payload
        raise AssertionError(f"Unexpected command {code}")


def test_tcp_binary_flood_and_receive_round_trip():
    async def run():
        companion = CompanionServer()
        try:
            server = await asyncio.start_server(companion.handle, "127.0.0.1", 0)
        except OSError:
            pytest.skip("sandbox denies loopback sockets; run this test with local networking enabled")
        async with server:
            port = server.sockets[0].getsockname()[1]
            radio = await MeshRadio.connect(tcp=("127.0.0.1", port), timeout=2)
            poller = None
            try:
                assert await radio.ensure_channel() == 1
                assert companion.channels[0][0] == "Public"
                assert companion.channels[1] == (CHANNEL, CHANNEL_KEY)
                assert companion.channels[2][0] == "Other"
                packet = encode_message("Hello over TCP!", UniformPredictor()).packet
                await radio.send(packet, "#test-region")
                wire_send = next(command for command in companion.commands if command[0] == 62)
                assert wire_send == bytes([62, 1, 255, 255, 255]) + packet
                scope = [command for command in companion.commands if command[0] == 54]
                import hashlib
                assert scope[0] == b"\x36\x00" + hashlib.sha256(b"#test-region").digest()[:16]
                assert scope[-1] == b"\x36\x00" + bytes(16)
                radio.subscribe_messages()
                poller = asyncio.create_task(radio.poll())
                received, reception = await asyncio.wait_for(radio.incoming.get(), timeout=2)
                assert decode_message(received, UniformPredictor()) == "Hello over TCP!"
                assert reception == {"snr": 2.0, "hops": 2}
            finally:
                if poller:
                    poller.cancel()
                    await asyncio.gather(poller, return_exceptions=True)
                await radio.close()
            await asyncio.wait_for(companion.finished.wait(), timeout=2)
    asyncio.run(run())
