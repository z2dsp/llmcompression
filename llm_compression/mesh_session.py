"""Persistent chat session shared by the desktop UI and its tests.

All radio work runs on one asyncio loop. A single executor serializes complete
model operations, since the predictor's KV cache is mutable. No Tk calls are
made here or on the inference thread.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
import threading

from .mesh_wire import SharedProfile, decode_message, encode_message
from .model import DEFAULT_CACHE, DEFAULT_MODEL, HuggingFacePredictor

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS = PROJECT_DIR / ".cache" / "mesh-gui-settings.json"


@dataclass(frozen=True)
class Settings:
    transport: str = "TCP"
    host: str = "192.168.0.1"
    port: int = 5051
    serial: str = ""
    baudrate: int = 115200
    timeout: float = 10
    profile: str = str(PROJECT_DIR / "llmcompression-profile.json")
    cache_dir: str = str(DEFAULT_CACHE)
    offline: bool = True
    region: str = "#llmcompression"

    def validate(self) -> None:
        from .mesh_radio import normalize_region

        for key in ("transport", "host", "serial", "profile", "cache_dir", "region"):
            if not isinstance(getattr(self, key), str):
                raise ValueError(f"{key} must be text")
        if self.transport not in {"TCP", "USB"}:
            raise ValueError("Choose TCP or USB")
        if not self.host.strip() and self.transport == "TCP":
            raise ValueError("Enter the TCP host")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("TCP port must be between 1 and 65535")
        if not self.serial.strip() and self.transport == "USB":
            raise ValueError("Choose or enter a USB serial device")
        if type(self.baudrate) is not int or self.baudrate <= 0:
            raise ValueError("Baud rate must be positive")
        if not 0 < self.timeout < float("inf"):
            raise ValueError("Timeout must be a positive finite number")
        if not self.profile.strip() or not self.cache_dir.strip():
            raise ValueError("Choose a shared profile and model cache folder")
        if type(self.offline) is not bool:
            raise ValueError("Offline mode must be a boolean")
        normalize_region(self.region)

    def save(self, path: Path = DEFAULT_SETTINGS) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path = DEFAULT_SETTINGS) -> "Settings":
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) - set(cls.__dataclass_fields__):
            raise ValueError("Invalid GUI settings file")
        result = cls(**document)
        # An empty serial field is valid when configuring TCP.
        result.validate()
        return result


class MeshSession:
    def __init__(self, emit, executor, *, radio_factory=None, profile_loader=None):
        self.emit = emit
        self.executor = executor
        self.radio_factory = radio_factory
        self.profile_loader = profile_loader or SharedProfile.load
        self.radio = None
        self.predictor = None
        self.connected = False
        self._tasks = []
        self._monitor = None
        self._disconnect_task = None
        self._send_lock = asyncio.Lock()

    async def model_work(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    async def connect(self, settings: Settings) -> None:
        if self.radio is not None:
            raise ValueError("Disconnect before opening another connection")
        settings.validate()
        self.emit("status", {"text": "Loading shared profile and language model…"})
        def load():
            profile = self.profile_loader(Path(settings.profile).expanduser())
            predictor = profile.load_predictor(Path(settings.cache_dir).expanduser(), settings.offline)
            return profile, predictor
        profile, predictor = await self.model_work(load)
        self.predictor = predictor
        factory = self.radio_factory
        if factory is None:
            from .mesh_radio import MeshRadio
            factory = MeshRadio.connect
        self.emit("status", {"text": "Connecting to companion…"})
        try:
            radio = await factory(
                tcp=(settings.host.strip(), settings.port) if settings.transport == "TCP" else None,
                serial=settings.serial.strip() if settings.transport == "USB" else None,
                baudrate=settings.baudrate, timeout=settings.timeout,
            )
        except BaseException:
            self.predictor = None
            raise
        self.radio = radio
        try:
            index = await radio.ensure_channel()
            radio.subscribe_messages()
        except BaseException:
            await self.disconnect()
            raise
        self.connected = True
        self._tasks = [
            asyncio.create_task(radio.poll()),
            asyncio.create_task(self._receive(radio, predictor)),
            asyncio.create_task(radio.disconnected.wait()),
        ]
        self._monitor = asyncio.create_task(self._watch())
        self.emit("connected", {
            "name": radio.client.self_info.get("name", "MeshCore"),
            "slot": index, "fingerprint": profile.fingerprint,
            "model": profile.predictor["model_id"],
        })

    async def _receive(self, radio, predictor):
        while True:
            packet, reception = await radio.incoming.get()
            try:
                text = await self.model_work(decode_message, packet, predictor)
            except (ValueError, RuntimeError, UnicodeError) as exc:
                self.emit("warning", {"text": f"Discarded datagram: {exc}"})
            else:
                self.emit("received", {"text": text, "wire_bytes": len(packet), **reception})

    async def _watch(self):
        done, _ = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                self.emit("error", {"text": str(task.exception())})
                break
        else:
            self.emit("warning", {"text": "Companion disconnected."})
        await self.disconnect()

    async def send(self, text: str, region: str, dry_run: bool = False):
        from .mesh_radio import normalize_region

        region = normalize_region(region)
        if not text:
            raise ValueError("Enter a message")
        async with self._send_lock:
            if not self.connected:
                raise ConnectionError("Connect before sending")
            predictor = self.predictor
            def prepare():
                prepared = encode_message(text, predictor)
                if decode_message(prepared.packet, predictor) != text:
                    raise ValueError("Local message round-trip failed")
                return prepared
            prepared = await self.model_work(prepare)
            # Disconnect may have been requested during inference.
            if not self.connected:
                raise ConnectionError("Disconnected before transmission; message was not sent")
            if not dry_run:
                await self.radio.send(prepared.packet, region)
            result = {
                "text": text, "region": region, "wire_bytes": len(prepared.packet),
                "original_bytes": prepared.text_bytes, "arithmetic_bytes": prepared.arithmetic_bytes,
            }
            self.emit("checked" if dry_run else "sent", result)
            return result

    async def disconnect(self):
        self.connected = False
        # UI disconnects, connection-loss callbacks, and shutdown can coincide.
        # Keep one cleanup task alive even when one of its callers is cancelled.
        if self._disconnect_task is None or self._disconnect_task.done():
            self._disconnect_task = asyncio.create_task(self._disconnect())
        await asyncio.shield(self._disconnect_task)

    async def _disconnect(self):
        monitor, self._monitor = self._monitor, None
        current = asyncio.current_task()
        if monitor is not None and monitor is not current:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        # Let an already submitted send finish before closing its transport.
        async with self._send_lock:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
            radio, self.radio = self.radio, None
            try:
                if radio is not None:
                    await radio.close()
            finally:
                self.predictor = None
                if radio is not None and radio.dropped:
                    self.emit("warning", {"text": f"Receive queue overflow: {radio.dropped} datagrams dropped."})
                self.emit("disconnected", {})


class SessionWorker:
    """Own the asyncio loop and model executor outside Tk's main thread."""

    def __init__(self, emit):
        self.emit = emit
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-inference")
        self.session = MeshSession(emit, self.executor)
        self._pending = set()
        self._stopping = False
        self.thread = threading.Thread(target=self._run, name="meshcore-io", daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.loop.close()

    def submit(self, coroutine, operation: str):
        if self._stopping:
            coroutine.close()
            return
        async def guarded():
            task = asyncio.current_task()
            self._pending.add(task)
            try:
                await coroutine
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.emit("operation_failed", {"operation": operation, "text": str(exc)})
            finally:
                self._pending.discard(task)
        return asyncio.run_coroutine_threadsafe(guarded(), self.loop)

    def create_profile(self, path: Path, model: str = DEFAULT_MODEL, revision=None,
                       context: int = 256, cache_dir=DEFAULT_CACHE, offline=True):
        async def create():
            def work():
                if path.exists():
                    raise ValueError("Profile already exists; choose another output file")
                predictor = HuggingFacePredictor(model, revision, context, cache_dir, offline)
                profile = SharedProfile.create(predictor)
                profile.save(path)
                return profile.fingerprint
            fingerprint = await self.session.model_work(work)
            self.emit("profile_created", {"path": str(path), "fingerprint": fingerprint})
        return self.submit(create(), "profile")

    def stop(self):
        if self._stopping:
            return
        self._stopping = True
        async def finish():
            pending = list(self._pending)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            try:
                await self.session.disconnect()
            finally:
                self.loop.stop()
        asyncio.run_coroutine_threadsafe(finish(), self.loop)
