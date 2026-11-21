import asyncio
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

tk = pytest.importorskip("tkinter")
pytest.importorskip("meshcore")

from llm_compression.mesh_gui import MeshWindow
from llm_compression.mesh_session import Settings


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip("desktop display unavailable")
    window.withdraw()
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass


class FakeWorker:
    def __init__(self, emit):
        self.thread = SimpleNamespace(is_alive=lambda: False)
        self.session = SimpleNamespace(connect=self.noop, disconnect=self.noop, send=self.noop)
        self.requests = []

    async def noop(self, *args):
        pass

    def submit(self, coroutine, operation):
        self.requests.append(operation)
        coroutine.close()

    def stop(self):
        pass


def connect_event(window):
    window._handle_event("connected", {"name": "Test companion", "slot": 1, "fingerprint": "a" * 64, "model": "test/model"})


def test_gui_connection_settings_and_send_state(root, tmp_path):
    window = MeshWindow(root, worker_factory=FakeWorker, settings_path=tmp_path / "settings.json")
    assert window.send_button.instate(["disabled"])
    window.toggle_connection()
    assert window.busy
    assert window.worker.requests == ["connect"]
    connect_event(window)
    assert window.send_button.instate(["!disabled"])
    assert window.serial_entry.instate(["disabled"])
    window.composer.insert("1.0", "Hello 🌊")
    window.send()
    assert window.sending
    assert window.send_button.instate(["disabled"])
    window._handle_event("sent", {"text": "Hello 🌊", "original_bytes": 10, "wire_bytes": 8, "region": "#test"})
    assert not window.sending
    assert window.composer.get("1.0", "end-1c") == ""
    assert "accepted by companion" in window.transcript.get("1.0", "end")


def test_gui_preserves_draft_edits_and_failed_messages(root, tmp_path):
    window = MeshWindow(root, worker_factory=FakeWorker, settings_path=tmp_path / "settings.json")
    connect_event(window)
    window.composer.insert("1.0", "Next draft")
    window._handle_event("sent", {"text": "Earlier message", "original_bytes": 15, "wire_bytes": 9, "region": "#test"})
    assert window.composer.get("1.0", "end-1c") == "Next draft"
    window._handle_event("operation_failed", {"operation": "send", "text": "queue full"})
    assert window.composer.get("1.0", "end-1c") == "Next draft"
    assert window.send_button.instate(["!disabled"])


def test_gui_settings_save_and_validation(root, tmp_path):
    path = tmp_path / "settings.json"
    window = MeshWindow(root, worker_factory=FakeWorker, settings_path=path)
    window.variables["region"].set("#test")
    window.save_settings()
    assert Settings.load(path).region == "#test"
    window.variables["port"].set("invalid")
    window.toggle_connection()
    assert not window.worker.requests
    assert "must be integers" in window.status.get()


def test_gui_size_check_does_not_clear_draft(root, tmp_path):
    window = MeshWindow(root, worker_factory=FakeWorker, settings_path=tmp_path / "settings.json")
    connect_event(window)
    window.composer.insert("1.0", "hello")
    window._handle_event("checked", {"text": "hello", "original_bytes": 5, "wire_bytes": 6, "region": "#test"})
    assert window.composer.get("1.0", "end-1c") == "hello"
    assert "no transmission" in window.status.get()


def test_gui_tcp_send_and_receive_with_real_worker(root, tmp_path):
    from test_mesh_tcp import CompanionServer
    from test_mesh_wire import UniformPredictor

    window = MeshWindow(root, settings_path=tmp_path / "settings.json")
    worker = window.worker
    profile = SimpleNamespace(
        predictor=UniformPredictor().metadata, fingerprint="a" * 64,
        load_predictor=lambda *args: UniformPredictor(),
    )
    worker.session.profile_loader = lambda path: profile
    async def start_server():
        companion = CompanionServer()
        server = await asyncio.start_server(companion.handle, "127.0.0.1", 0)
        return companion, server
    def until(predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            if time.monotonic() > deadline:
                pytest.fail("GUI operation timed out: " + window.status.get())
            root.update()
            time.sleep(0.01)
    server = None
    try:
        companion, server = asyncio.run_coroutine_threadsafe(start_server(), worker.loop).result(timeout=5)
        window.variables["host"].set("127.0.0.1")
        window.variables["port"].set(str(server.sockets[0].getsockname()[1]))
        window.variables["region"].set("#test-region")
        window.toggle_connection()
        until(lambda: window.connected)
        window.composer.insert("1.0", "GUI message 🌊")
        window.send(dry_run=True)
        until(lambda: not window.sending)
        assert not [command for command in companion.commands if command[0] == 62]
        window.send()
        until(lambda: "Channel · received" in window.transcript.get("1.0", "end"))
        assert "GUI message 🌊" in window.transcript.get("1.0", "end")
        assert "accepted by companion" in window.transcript.get("1.0", "end")
        assert len([command for command in companion.commands if command[0] == 62]) == 1
        window.toggle_connection()
        until(lambda: not window.connected)
    finally:
        if server is not None:
            worker.loop.call_soon_threadsafe(server.close)
        worker.stop()
        worker.thread.join(timeout=5)
    assert not worker.thread.is_alive()
