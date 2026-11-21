"""Tk desktop client. Launch with python -m llm_compression.meshcore_app."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .mesh_session import DEFAULT_SETTINGS, SessionWorker, Settings
from .mesh_wire import CHANNEL, MAX_DATAGRAM, MAX_TEXT_BYTES
from .model import DEFAULT_MODEL


class MeshWindow:
    def __init__(self, root, *, worker_factory=SessionWorker, settings_path=DEFAULT_SETTINGS):
        self.root = root
        self.settings_path = Path(settings_path)
        self.events = queue.Queue()
        self.worker = worker_factory(lambda kind, data: self.events.put((kind, data)))
        self.connected = self.busy = self.sending = self.closing = False
        self.profile_dialog = None
        self.config_widgets = []
        self.tcp_widgets = []
        self.usb_widgets = []
        self.boot_error = None
        try:
            settings = Settings.load(self.settings_path) if self.settings_path.exists() else Settings()
        except (OSError, ValueError, TypeError) as exc:
            settings = Settings()
            self.boot_error = f"Saved settings could not be loaded: {exc}"
        self.variables = {
            key: tk.BooleanVar(root, value=value) if isinstance(value, bool) else tk.StringVar(root, value=str(value))
            for key, value in asdict(settings).items()
        }
        self.status = tk.StringVar(root, value="Disconnected — choose your connection and shared profile.")
        self.connection_label = tk.StringVar(root, value="DISCONNECTED")
        self.profile_info = tk.StringVar(root, value="Share the same profile file with everyone on this channel.")
        self.byte_count = tk.StringVar(root, value="0 UTF-8 bytes")
        self._build()
        self.variables["transport"].trace_add("write", lambda *_: self._refresh_controls())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(80, self._drain_events)
        if self.boot_error:
            self._append("Settings", self.boot_error, "warning")
        self._refresh_controls()

    def _build(self):
        root = self.root
        root.title("LLM Mesh · #llmcompression")
        root.geometry("1120x840")
        root.minsize(980, 720)
        root.configure(background="#eef2f6")
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#eef2f6")
        style.configure("TLabel", background="#eef2f6", foreground="#243447")
        style.configure("TLabelframe", background="#eef2f6")
        style.configure("TLabelframe.Label", background="#eef2f6", foreground="#243447")
        style.configure("TCheckbutton", background="#eef2f6")
        style.configure("TButton", padding=(10, 5))
        style.configure("Accent.TButton", background="#176b66", foreground="white")
        style.map("Accent.TButton", background=[("active", "#10524e"), ("disabled", "#d5dde3")])
        style.configure("Heading.TLabel", font=("TkDefaultFont", 19, "bold"))
        style.configure("Muted.TLabel", foreground="#617287")

        header = ttk.Frame(root, padding=(22, 18))
        header.pack(fill="x")
        ttk.Label(header, text="LLM Mesh", style="Heading.TLabel").pack(side="left")
        ttk.Label(header, text=f"{CHANNEL}  ·  Flood transport", style="Muted.TLabel").pack(side="left", padx=18)
        ttk.Label(header, textvariable=self.connection_label).pack(side="right")

        body = ttk.Frame(root, padding=(18, 0, 18, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        sidebar.rowconfigure(0, weight=1)
        sidebar.columnconfigure(0, weight=1)
        canvas = tk.Canvas(sidebar, width=305, highlightthickness=0, background="#eef2f6")
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)
        left = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        connection = ttk.LabelFrame(left, text="Connection", padding=12)
        connection.pack(fill="x")
        connection.columnconfigure(1, weight=1)
        self._field(connection, 0, "Transport", "transport", values=("TCP", "USB"))
        self._field(connection, 1, "TCP host", "host", group=self.tcp_widgets)
        self._field(connection, 2, "TCP port", "port", group=self.tcp_widgets)
        self.serial_entry = self._field(connection, 3, "USB device", "serial", values=(), group=self.usb_widgets, editable=True)
        self.scan_button = ttk.Button(connection, text="Refresh USB devices", command=self._scan_serial)
        self.scan_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 7))
        self.config_widgets.append(self.scan_button)
        self.usb_widgets.append(self.scan_button)
        self._field(connection, 5, "Baud rate", "baudrate", group=self.usb_widgets)
        self._field(connection, 6, "Timeout (s)", "timeout")

        compression = ttk.LabelFrame(left, text="Compression profile", padding=12)
        compression.pack(fill="x", pady=12)
        compression.columnconfigure(0, weight=1)
        self._path_field(compression, 0, "Shared profile", "profile", self._browse_profile)
        self._path_field(compression, 2, "Model cache folder", "cache_dir", self._browse_cache)
        offline = ttk.Checkbutton(compression, text="Use cached model only (offline)", variable=self.variables["offline"])
        offline.grid(row=4, column=0, columnspan=2, sticky="w", pady=8)
        self.config_widgets.append(offline)
        new_profile = ttk.Button(compression, text="Create shared profile…", command=self._create_profile_dialog)
        new_profile.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.config_widgets.append(new_profile)
        ttk.Label(compression, textvariable=self.profile_info, wraplength=290, style="Muted.TLabel").grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.connect_button = ttk.Button(left, text="Connect", style="Accent.TButton", command=self.toggle_connection)
        self.connect_button.pack(fill="x")
        self.save_button = ttk.Button(left, text="Save settings", command=self.save_settings)
        self.save_button.pack(fill="x", pady=7)

        toolbar = ttk.Frame(right)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(toolbar, text="Conversation", font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Button(toolbar, text="Save chat…", command=self._save_chat).pack(side="right")
        ttk.Button(toolbar, text="Clear", command=self._clear_chat).pack(side="right", padx=6)
        self.transcript = scrolledtext.ScrolledText(right, wrap="word", state="disabled", font=("TkDefaultFont", 11),
                                                  background="white", foreground="#243447", relief="flat", padx=16, pady=14)
        self.transcript.grid(row=1, column=0, sticky="nsew")
        self.transcript.tag_configure("sent", foreground="#176b66", font=("TkDefaultFont", 10, "bold"))
        self.transcript.tag_configure("received", foreground="#245ab0", font=("TkDefaultFont", 10, "bold"))
        self.transcript.tag_configure("system", foreground="#617287", font=("TkDefaultFont", 10))
        self.transcript.tag_configure("warning", foreground="#a04525", font=("TkDefaultFont", 10))

        composer = ttk.Frame(right, padding=(0, 12, 0, 0))
        composer.grid(row=2, column=0, sticky="ew")
        composer.columnconfigure(1, weight=1)
        ttk.Label(composer, text="Region scope").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.region_entry = ttk.Combobox(composer, textvariable=self.variables["region"], values=("#llmcompression", "#au-nsw-ntl", "*"))
        self.region_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(composer, text="* = unscoped", style="Muted.TLabel").grid(row=0, column=2, padx=(8, 0))
        self.composer = tk.Text(composer, height=5, wrap="word", font=("TkDefaultFont", 11), relief="solid", borderwidth=1, padx=10, pady=8)
        self.composer.grid(row=1, column=0, columnspan=3, sticky="ew", pady=8)
        self.composer.bind("<<Modified>>", self._text_changed)
        self.composer.bind("<Control-Return>", lambda event: self._keyboard_send())
        self.composer.bind("<Command-Return>", lambda event: self._keyboard_send())
        actions = ttk.Frame(composer)
        actions.grid(row=2, column=0, columnspan=3, sticky="ew")
        ttk.Button(actions, text="Open text…", command=self._open_text).pack(side="left")
        ttk.Label(actions, textvariable=self.byte_count, style="Muted.TLabel").pack(side="left", padx=10)
        self.send_button = ttk.Button(actions, text="Send", style="Accent.TButton", command=self.send)
        self.send_button.pack(side="right")
        self.check_button = ttk.Button(actions, text="Check size", command=lambda: self.send(dry_run=True))
        self.check_button.pack(side="right", padx=6)
        ttk.Label(composer, text="Ctrl+Enter to send · Enter adds a line · Acceptance does not confirm remote delivery", style="Muted.TLabel").grid(row=3, column=0, columnspan=3, sticky="w", pady=(9, 0))

        footer = ttk.Frame(root, padding=(22, 8, 22, 14))
        footer.pack(side="bottom", fill="x", before=body)
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=90)
        self.progress.pack(side="right", padx=(14, 0))
        ttk.Label(footer, textvariable=self.status, wraplength=850).pack(side="left")

    def _field(self, parent, row, label, key, *, values=None, group=None, editable=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=2)
        if values is None:
            widget = ttk.Entry(parent, textvariable=self.variables[key], width=21)
        else:
            widget = ttk.Combobox(parent, textvariable=self.variables[key], values=values, width=19,
                                  state="normal" if editable else "readonly")
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        self.config_widgets.append(widget)
        if group is not None:
            group.append(widget)
        return widget

    def _path_field(self, parent, row, label, key, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 3))
        entry = ttk.Entry(parent, textvariable=self.variables[key], width=24)
        entry.grid(row=row + 1, column=0, sticky="ew", padx=(0, 5))
        button = ttk.Button(parent, text="…", width=3, command=command)
        button.grid(row=row + 1, column=1)
        self.config_widgets.extend((entry, button))

    def settings(self):
        values = {key: variable.get() for key, variable in self.variables.items()}
        try:
            for key in ("port", "baudrate"):
                values[key] = int(values[key])
            values["timeout"] = float(values["timeout"])
        except ValueError as exc:
            raise ValueError("Port and baud rate must be integers; timeout must be a number") from exc
        settings = Settings(**values)
        settings.validate()
        return settings

    def _refresh_controls(self):
        locked = self.connected or self.busy or self.closing
        for widget in self.config_widgets:
            widget.configure(state="disabled" if locked else "normal")
        transport = self.variables["transport"].get()
        if not locked:
            # The transport choice itself must remain readonly.
            for widget in self.config_widgets:
                if isinstance(widget, ttk.Combobox) and widget is not self.serial_entry:
                    widget.configure(state="readonly")
            for widget in (self.usb_widgets if transport == "TCP" else self.tcp_widgets):
                widget.configure(state="disabled")
        self.connect_button.configure(text="Disconnect" if self.connected else "Connect", state="disabled" if self.busy or self.closing else "normal")
        for button in (self.send_button, self.check_button):
            button.configure(state="normal" if self.connected and not self.busy and not self.sending and not self.closing else "disabled")
        self.save_button.configure(state="disabled" if self.closing else "normal")
        if self.busy or self.sending or self.closing:
            self.progress.start(12)
        else:
            self.progress.stop()

    def toggle_connection(self):
        if self.connected:
            self.busy = True
            self.status.set("Disconnecting…")
            self.worker.submit(self.worker.session.disconnect(), "disconnect")
        else:
            try:
                settings = self.settings()
            except (ValueError, TypeError) as exc:
                self._error(str(exc))
                return
            self.busy = True
            self.status.set("Loading shared profile and language model…")
            self.worker.submit(self.worker.session.connect(settings), "connect")
        self._refresh_controls()

    def send(self, dry_run=False):
        if not self.connected or self.sending or self.busy or self.closing:
            return
        text = self.composer.get("1.0", "end-1c")
        if not text:
            self.status.set("Enter a message first.")
            return
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            self._error(f"Message exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
            return
        self.sending = True
        self.status.set("Compressing and verifying message…")
        self.worker.submit(self.worker.session.send(text, self.variables["region"].get(), dry_run), "send")
        self._refresh_controls()

    def _keyboard_send(self):
        self.send()
        return "break"

    def _text_changed(self, event=None):
        if self.composer.edit_modified():
            text = self.composer.get("1.0", "end-1c")
            self.byte_count.set(f"{len(text.encode('utf-8')):,} UTF-8 bytes")
            self.composer.edit_modified(False)

    def _append(self, label, text, tag="system", details=""):
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        self.transcript.configure(state="normal")
        self.transcript.insert("end", f"{timestamp}  {label}" + (f" · {details}" if details else "") + "\n", tag)
        self.transcript.insert("end", text + "\n\n")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _error(self, text):
        self.status.set(text)
        self._append("Error", text, "warning")

    def _handle_event(self, kind, data):
        if kind == "status":
            self.status.set(data["text"])
        elif kind == "connected":
            self.connected, self.busy = True, False
            self.connection_label.set("CONNECTED")
            self.status.set(f"Connected to {data['name']} · {CHANNEL} in slot {data['slot']} · listening")
            self.profile_info.set(f"{data['model']}\nProfile: {data['fingerprint'][:16]}")
            self._append("Connected", self.status.get())
        elif kind == "disconnected":
            self.connected = self.busy = self.sending = False
            self.connection_label.set("DISCONNECTED")
            self.status.set("Disconnected.")
        elif kind == "received":
            details = f"{data['wire_bytes']} bytes · SNR {data.get('snr')} · {data.get('hops')} hops"
            self._append("Channel · received", data["text"], "received", details)
        elif kind in {"sent", "checked"}:
            self.sending = False
            detail = f"{data['original_bytes']} → {data['wire_bytes']}/{MAX_DATAGRAM} bytes · {data['region']} · flood"
            if kind == "sent":
                self._append("You · accepted by companion", data["text"], "sent", detail)
                if self.composer.get("1.0", "end-1c") == data["text"]:
                    self.composer.delete("1.0", "end")
                self.status.set("Companion accepted the message; remote delivery is unconfirmed. " + detail)
            else:
                self.status.set("Size check passed; no transmission. " + detail)
                self._append("Size check", "Exact local round-trip. " + detail)
        elif kind == "operation_failed":
            if data["operation"] in {"connect", "disconnect", "profile"}:
                self.busy = False
            if data["operation"] == "send":
                self.sending = False
            self._error(data["text"])
        elif kind in {"error", "warning"}:
            self._error(data["text"])
        elif kind == "profile_created":
            self.busy = False
            self.variables["profile"].set(data["path"])
            self.profile_info.set("Profile: " + data["fingerprint"][:16])
            self.status.set("Profile created. Share this exact file with participants before connecting.")
            self._append("Shared profile", self.status.get() + "\n" + data["path"])
        self._refresh_controls()

    def _drain_events(self):
        try:
            for _ in range(100):
                kind, data = self.events.get_nowait()
                self._handle_event(kind, data)
        except queue.Empty:
            pass
        if self.closing and not self.worker.thread.is_alive():
            self.root.destroy()
            return
        self.root.after(80, self._drain_events)

    def save_settings(self):
        try:
            self.settings().save(self.settings_path)
        except (OSError, ValueError, TypeError) as exc:
            self._error(str(exc))
        else:
            self.status.set(f"Settings saved to {self.settings_path}")

    def _scan_serial(self):
        try:
            from serial.tools.list_ports import comports
            devices = sorted(port.device for port in comports())
            self.serial_entry.configure(values=devices)
            if devices and not self.variables["serial"].get():
                self.variables["serial"].set(devices[0])
            self.status.set(f"Found {len(devices)} USB/serial devices. You can also enter a path manually.")
        except (ImportError, OSError) as exc:
            self._error(str(exc))

    def _browse_profile(self):
        path = filedialog.askopenfilename(parent=self.root, title="Shared compression profile", filetypes=[("JSON profile", "*.json"), ("All files", "*")])
        if path:
            self.variables["profile"].set(path)

    def _browse_cache(self):
        path = filedialog.askdirectory(parent=self.root, title="Model cache folder")
        if path:
            self.variables["cache_dir"].set(path)

    def _open_text(self):
        path = filedialog.askopenfilename(parent=self.root, title="Open UTF-8 message")
        if path:
            try:
                data = Path(path).read_bytes()
                if len(data) > MAX_TEXT_BYTES:
                    raise ValueError(f"Message exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
                text = data.decode("utf-8")
            except (OSError, ValueError) as exc:
                self._error(str(exc))
                return
            self.composer.delete("1.0", "end")
            self.composer.insert("1.0", text)

    def _save_chat(self):
        path = filedialog.asksaveasfilename(parent=self.root, title="Save conversation", defaultextension=".txt", filetypes=[("Text file", "*.txt")])
        if path:
            try:
                Path(path).write_text(self.transcript.get("1.0", "end-1c"), encoding="utf-8")
            except OSError as exc:
                self._error(str(exc))

    def _clear_chat(self):
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    def _create_profile_dialog(self):
        if self.busy or self.connected:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Create shared compression profile")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)
        fields = {}
        for row, (label, value) in enumerate((("Hugging Face model", DEFAULT_MODEL), ("Revision (optional)", ""), ("Context tokens", "256"))):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=7, padx=(0, 12))
            fields[label] = ttk.Entry(frame, width=38)
            fields[label].insert(0, value)
            fields[label].grid(row=row, column=1, pady=7)
        ttk.Label(frame, text="Uses the main window's model cache and offline setting.\nShare the created file with every participant.", style="Muted.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=12)
        def create():
            try:
                context = int(fields["Context tokens"].get())
                if context <= 0:
                    raise ValueError("Context tokens must be positive")
                model = fields["Hugging Face model"].get().strip()
                if not model:
                    raise ValueError("Enter a model ID")
            except ValueError as exc:
                messagebox.showerror("Invalid profile settings", str(exc), parent=dialog)
                return
            path = filedialog.asksaveasfilename(parent=dialog, title="Create a new shared profile", defaultextension=".json", initialfile="shared-profile.json")
            if not path:
                return
            revision = fields["Revision (optional)"].get().strip() or None
            dialog.destroy()
            self.busy = True
            self.status.set("Creating profile and calibrating the language model…")
            self._refresh_controls()
            self.worker.create_profile(Path(path), model, revision, context,
                                       Path(self.variables["cache_dir"].get()).expanduser(), self.variables["offline"].get())
        ttk.Button(frame, text="Create profile…", command=create, style="Accent.TButton").grid(row=4, column=1, sticky="e")
        dialog.grab_set()

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.status.set("Closing connection; waiting for the current model operation to finish…")
        self._refresh_controls()
        self.worker.stop()


def main(argv=None) -> int:
    import argparse
    import sys

    argparse.ArgumentParser(description="Desktop MeshCore chat on #llmcompression (USB/TCP, flood only).").parse_args(argv)
    try:
        import meshcore  # noqa: F401
    except ImportError as exc:
        print(f"The GUI requires MeshCore support: {exc}. Install with: python -m pip install -e '.[mesh]'", file=sys.stderr)
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Cannot open desktop GUI: {exc}. Run in a desktop session, or use llm-mesh send/listen.", file=sys.stderr)
        return 1
    MeshWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
