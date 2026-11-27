"""Tk desktop T9 typing simulator and exported .t9nn activation viewer."""
import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
import queue
import threading

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, scrolledtext, ttk
except ImportError:
    tk = None

from .nn_inspection import activation_color, byte_label, group_activations
from .t9 import Accuracy, KEY_LETTERS, encode_text, parse_keys
from .t9_inspection import inspect_position, simulate
from .t9_model import T9Model


def backspace_keys(keys):
    """Remove one notation token, including a digit/backslash escape pair."""
    starts, offset = [], 0
    while offset < len(keys):
        starts.append(offset)
        offset += 2 if keys[offset] == '\\' and offset + 1 < len(keys) else 1
    return keys[:starts[-1]] if starts else ''


def evaluate_input(model, keys, beam, position, cache, *, cancel):
    # Navigating an existing prediction only needs one forward pass.
    simulation = cache if cache is not None else simulate(model, keys, beam_width=beam, cancel=cancel)
    if position is None:
        position = next((i for i in range(len(simulation.tokens) - 1, -1, -1)
                         if isinstance(simulation.tokens[i], int)), len(simulation.tokens))
    position = min(position, len(simulation.tokens))
    return simulation, position, inspect_position(model, simulation, position, cancel=cancel)


class T9Window:
    def __init__(self, root, *, text='Hello world!', keys=None, beam_width=8, tile_limit=256):
        self.root = root
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='t9-viewer')
        self.events = queue.Queue()
        self.cancel_event = threading.Event()
        self.future = None
        self.request_id = 0
        self.model = self.simulation = self.trace = self.cache = None
        self.position = 0
        self.closing = self.syncing = False
        self.update_job = self.draw_job = self.poll_job = self.play_job = None
        self.replay = []
        self.cells = {}
        self.path = tk.StringVar(root)
        self.metadata = tk.StringVar(root, value='Choose an exported model.t9nn file.')
        self.status = tk.StringVar(root, value='Ready. Load a model, then type keys or use the phone keypad.')
        self.context_info = tk.StringVar(root)
        self.position_info = tk.StringVar(root)
        self.metrics = tk.StringVar(root)
        self.detail = tk.StringVar(root, value='Hover over a tile to inspect it. Blue: negative; amber: positive; green: letter probability.')
        self.beam = tk.StringVar(root, value=str(beam_width))
        self.tiles = tk.StringVar(root, value=str(tile_limit))
        self.position_var = tk.StringVar(root, value='0')
        self._build(text, encode_text(text) if keys is None else keys)
        self.beam.trace_add('write', self._edited_settings)
        self.poll_job = root.after(30, self._poll)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def _build(self, text, keys):
        root = self.root
        root.title('T9 · Typing simulator & model explorer')
        root.geometry('1420x940')
        root.minsize(1080, 800)
        style = ttk.Style(root)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure('Title.TLabel', font=('TkDefaultFont', 18, 'bold'))
        style.configure('Muted.TLabel', foreground='#58677d')
        body = ttk.Frame(root, padding=14)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text='T9 typing simulator', style='Title.TLabel').pack(anchor='w')
        ttk.Label(body, text='Type ambiguous phone keys and watch the model choose a spelling.', style='Muted.TLabel').pack(anchor='w', pady=(2, 10))
        files = ttk.Frame(body)
        files.pack(fill='x')
        ttk.Entry(files, textvariable=self.path).pack(side='left', fill='x', expand=True)
        ttk.Button(files, text='Browse…', command=self.browse).pack(side='left', padx=6)
        ttk.Button(files, text='Load model', command=self.load_model).pack(side='left')
        ttk.Label(body, textvariable=self.metadata, style='Muted.TLabel').pack(anchor='w', pady=6)

        inputs = ttk.Frame(body)
        inputs.pack(fill='x', pady=(0, 8))
        inputs.columnconfigure(1, weight=1)
        phone = ttk.LabelFrame(inputs, text='Phone keypad', padding=8)
        phone.grid(row=0, column=0, sticky='ns', padx=(0, 10))
        for index, key in enumerate(range(1, 10)):
            label = '1\n—' if key == 1 else f'{key}\n{KEY_LETTERS[key].upper()}'
            button = ttk.Button(phone, text=label, width=8, command=lambda k=key: self.type_key(str(k)))
            button.grid(row=index // 3, column=index % 3, padx=2, pady=2, sticky='nsew')
            if key == 1:
                button.state(['disabled'])
        for column, (label, command) in enumerate((('⌫', self.backspace), ('0\nSPACE', lambda: self.type_key('0')), ('.', lambda: self.type_key('.')))):
            ttk.Button(phone, text=label, width=8, command=command).grid(row=3, column=column, padx=2, pady=2, sticky='nsew')
        ttk.Button(phone, text='Clear keys', command=lambda: self.replace_keys('')).grid(row=4, column=0, columnspan=3, sticky='ew', pady=(5, 0))

        editors = ttk.Frame(inputs)
        editors.grid(row=0, column=1, sticky='nsew')
        ttk.Label(editors, text='Keystrokes · 2–9 letters, 0 space, punctuation literal; escape original digits with \\').pack(anchor='w')
        self.composer = scrolledtext.ScrolledText(editors, height=2, wrap='word', font=('TkFixedFont', 11), undo=True)
        self.composer.pack(fill='x', pady=(3, 6))
        self.composer.insert('1.0', keys)
        self.composer.edit_modified(False)
        self.composer.bind('<<Modified>>', self._edited)
        self.composer.bind('<BackSpace>', self._keyboard_backspace)
        self.composer.bind('<space>', self._keyboard_space)
        reference_row = ttk.Frame(editors)
        reference_row.pack(fill='x')
        ttk.Label(reference_row, text='Try ordinary text').pack(side='left')
        ttk.Button(reference_row, text='Convert text → keys', command=self.convert_text).pack(side='right')
        self.reference = scrolledtext.ScrolledText(editors, height=2, wrap='word', font=('TkFixedFont', 10))
        self.reference.pack(fill='x', pady=(3, 6))
        self.reference.insert('1.0', text)
        self.reference.edit_modified(False)
        self.reference.bind('<<Modified>>', self._reference_edited)
        ttk.Label(editors, text='Predicted text · updates after each key; the active word may change as keys arrive').pack(anchor='w')
        self.output = scrolledtext.ScrolledText(editors, height=2, wrap='word', font=('TkFixedFont', 13), state='disabled')
        self.output.pack(fill='x', pady=(3, 0))
        self.output.tag_configure('selected', background='#f9ce87', foreground='#182333')

        alternatives = ttk.LabelFrame(inputs, text='Selected word · beam alternatives', padding=8)
        alternatives.grid(row=0, column=2, sticky='nsew', padx=(10, 0))
        self.candidates = ttk.Treeview(alternatives, columns=('word', 'score'), show='headings', height=7, selectmode='none')
        self.candidates.heading('word', text='Spelling')
        self.candidates.heading('score', text='Log P')
        self.candidates.column('word', width=150)
        self.candidates.column('score', width=80, anchor='e')
        self.candidates.tag_configure('winner', background='#d8f0e7')
        self.candidates.pack(fill='both', expand=True)
        ttk.Label(alternatives, text='Higher scores rank first.\nBeam results are approximate.', style='Muted.TLabel').pack(anchor='w', pady=(5, 0))

        controls = ttk.Frame(body)
        controls.pack(fill='x')
        self.previous_button = ttk.Button(controls, text='◀ Previous', command=lambda: self.set_position(self.position - 1))
        self.previous_button.pack(side='left')
        self.next_button = ttk.Button(controls, text='Next ▶', command=lambda: self.set_position(self.position + 1))
        self.next_button.pack(side='left', padx=5)
        ttk.Label(controls, text='Inspect character').pack(side='left', padx=(10, 4))
        self.position_spin = ttk.Spinbox(controls, from_=0, to=0, width=7, textvariable=self.position_var, command=self._position_entered)
        self.position_spin.pack(side='left')
        self.position_spin.bind('<Return>', self._position_entered)
        self.play_button = ttk.Button(controls, text='Replay typing', command=self.toggle_replay)
        self.play_button.pack(side='left', padx=12)
        ttk.Label(controls, text='Beam width').pack(side='left')
        beam = ttk.Spinbox(controls, from_=1, to=64, width=5, textvariable=self.beam, command=self._edited_settings)
        beam.pack(side='left', padx=5)
        beam.bind('<Return>', self._edited_settings)
        beam.bind('<FocusOut>', self._edited_settings)
        ttk.Label(controls, text='Tiles per layer').pack(side='left', padx=(10, 4))
        tiles = ttk.Combobox(controls, textvariable=self.tiles, values=('64', '128', '256', '512', '1024'), width=6, state='readonly')
        tiles.pack(side='left')
        tiles.bind('<<ComboboxSelected>>', lambda _: self._schedule_draw())
        self.slider = ttk.Scale(body, from_=0, to=1, command=self._slider_changed)
        self.slider.pack(fill='x', pady=(4, 0))
        ttk.Label(body, textvariable=self.position_info).pack(anchor='w')
        ttk.Label(body, textvariable=self.context_info, style='Muted.TLabel', wraplength=1350).pack(anchor='w', pady=3)
        ttk.Label(body, textvariable=self.metrics, style='Muted.TLabel').pack(anchor='w', pady=(0, 6))

        footer = ttk.Frame(body)
        footer.pack(side='bottom', fill='x')
        ttk.Label(footer, text='Hidden tiles: fixed −1…+1 scale. Embeddings scale per frame. Grouped tiles show peak absolute activation. Output shows all 26 letters; off-key letters have zero probability.', style='Muted.TLabel', wraplength=1350).pack(anchor='w', pady=(6, 3))
        self.status_label = ttk.Label(footer, textvariable=self.status, style='Muted.TLabel', wraplength=1350)
        self.status_label.pack(anchor='w')
        panels = ttk.Frame(body)
        panels.pack(fill='both', expand=True)
        panels.columnconfigure(0, weight=1)
        panels.rowconfigure(0, weight=1)
        diagram = ttk.Frame(panels)
        diagram.grid(row=0, column=0, sticky='nsew')
        diagram.columnconfigure(0, weight=1)
        diagram.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(diagram, background='#101923', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        horizontal = ttk.Scrollbar(diagram, orient='horizontal', command=self.canvas.xview)
        horizontal.grid(row=1, column=0, sticky='ew')
        vertical = ttk.Scrollbar(diagram, command=self.canvas.yview)
        vertical.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.bind('<Configure>', lambda _: self._schedule_draw())
        self.canvas.bind('<Motion>', self._hover)
        sidebar = ttk.Frame(panels, padding=(12, 0, 0, 0))
        sidebar.grid(row=0, column=1, sticky='nsew')
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(2, weight=1)
        ttk.Label(sidebar, text='Selected letter probabilities', font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, sticky='w')
        self.probabilities = ttk.Treeview(sidebar, columns=('letter', 'probability'), show='headings', height=4, selectmode='none')
        self.probabilities.heading('letter', text='Letter')
        self.probabilities.heading('probability', text='Probability')
        self.probabilities.column('letter', width=70)
        self.probabilities.column('probability', width=125, anchor='e')
        self.probabilities.tag_configure('winner', background='#d8f0e7')
        self.probabilities.grid(row=1, column=0, sticky='ew', pady=6)
        self.detail_view = scrolledtext.ScrolledText(sidebar, height=9, width=27, wrap='word', state='disabled', relief='flat')
        self.detail_view.grid(row=2, column=0, sticky='nsew')
        self.detail.trace_add('write', lambda *_: self._show_detail())
        self._show_detail()
        self._controls()

    def browse(self):
        name = filedialog.askopenfilename(parent=self.root, title='Open T9 predictive model', filetypes=[('T9 model', '*.t9nn'), ('All files', '*')])
        if name:
            self.path.set(name)
            self.load_model()

    def _invalidate(self):
        self.request_id += 1
        self.cancel_event.set()
        if self.future is not None:
            self.future.cancel()

    def _submit(self, kind, function, *args):
        self._invalidate()
        token = self.request_id
        self.cancel_event = threading.Event()
        kwargs = {'cancel': self.cancel_event} if kind == 'input' else {}
        self.future = self.executor.submit(function, *args, **kwargs)
        self.future.add_done_callback(lambda future: self.events.put((token, kind, future)))

    def load_model(self):
        name = self.path.get().strip()
        if not name:
            self.status.set('Choose a model.t9nn file first.')
            return
        self.pause()
        self.model = self.simulation = self.trace = self.cache = None
        self._clear_result()
        self.metadata.set('Loading model…')
        self.status.set('Loading model…')
        self._controls()
        self._submit('model', T9Model, Path(name).expanduser())

    def _poll(self):
        if self.closing:
            return
        while True:
            try:
                token, kind, future = self.events.get_nowait()
            except queue.Empty:
                break
            if token != self.request_id:
                continue
            try:
                result = future.result()
            except CancelledError:
                continue
            except Exception as exc:
                self.pause()
                self._clear_result()
                if kind == 'model':
                    self.metadata.set('No model loaded. Choose a valid model.t9nn file.')
                self.status.set(f'Could not {"load model" if kind == "model" else "predict input"}: {exc}')
                continue
            if kind == 'model':
                self.model = result
                self.metadata.set(f'{result.size_bytes / 1024:,.1f} KiB · {result.storage} · text context {result.context} bytes · key context {result.key_context} · embedding {result.embedding} · {result.hidden_layers} hidden layers × {result.hidden:,} neurons')
                self._controls()
                self.request_update()
            else:
                self.simulation, self.position, self.trace = result
                self.cache = (self.model, self.simulation.keystrokes, int(self.beam.get()), self.simulation)
                self._show_result()
                if self.replay:
                    self.play_job = self.root.after(350, self._replay_step)
                else:
                    self.pause()
        self.poll_job = self.root.after(30, self._poll)

    def _clear_result(self):
        self.simulation = self.trace = None
        self._set_output('')
        for table in (self.candidates, self.probabilities):
            table.delete(*table.get_children())
        self.context_info.set('')
        self.position_info.set('')
        self.metrics.set('')
        self._schedule_draw()

    def _keys(self):
        return self.composer.get('1.0', 'end-1c')

    def _edited(self, _event=None):
        if not self.composer.edit_modified():
            return
        self.composer.edit_modified(False)
        self.pause()
        self._schedule_update()

    def _edited_settings(self, *_args):
        self.pause()
        self._schedule_update()

    def _schedule_update(self, position=None):
        # Input edits during loading must not invalidate the model-load result.
        if self.model is not None:
            self._invalidate()
        if self.update_job:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(160, lambda: self.request_update(position))

    def request_update(self, position=None):
        if self.update_job:
            self.root.after_cancel(self.update_job)
            self.update_job = None
        if self.closing or self.model is None:
            return
        try:
            beam = int(self.beam.get())
            if not 1 <= beam <= 64:
                raise ValueError()
        except ValueError:
            self.pause()
            self._clear_result()
            self.status.set('Beam width must be an integer from 1 to 64.')
            return
        keys = self._keys()
        cached = self.cache[3] if self.cache is not None and self.cache[:3] == (self.model, keys, beam) else None
        self.status.set('Predicting…')
        self._submit('input', evaluate_input, self.model, keys, beam, position, cached)

    def replace_keys(self, keys, *, replay=False):
        if not replay:
            self.pause()
        self.composer.delete('1.0', 'end')
        self.composer.insert('1.0', keys)
        self.composer.mark_set('insert', 'end-1c')
        self.composer.edit_modified(False)
        self._schedule_update()

    def type_key(self, key):
        self.pause()
        try:
            self.composer.delete('sel.first', 'sel.last')
        except tk.TclError:
            pass
        self.composer.insert('insert', key)
        self.composer.see('insert')
        self.composer.edit_modified(False)
        self._schedule_update()

    def backspace(self):
        self.pause()
        try:
            self.composer.delete('sel.first', 'sel.last')
        except tk.TclError:
            prefix = self.composer.get('1.0', 'insert')
            shortened = backspace_keys(prefix)
            self.composer.delete(f'insert-{len(prefix) - len(shortened)}c', 'insert')
        self.composer.edit_modified(False)
        self._schedule_update()

    def _keyboard_backspace(self, _event):
        self.backspace()
        return 'break'

    def _keyboard_space(self, _event):
        self.type_key('0')
        return 'break'

    def convert_text(self):
        self.replace_keys(encode_text(self.reference.get('1.0', 'end-1c')))

    def _reference_edited(self, _event=None):
        if self.reference.edit_modified():
            self.reference.edit_modified(False)
            self._show_metrics()

    def _show_metrics(self):
        self.metrics.set('')
        if self.simulation is None:
            return
        reference = self.reference.get('1.0', 'end-1c')
        if encode_text(reference) != self.simulation.keystrokes:
            self.metrics.set('Convert ordinary text to keys to compare the prediction with that text.')
            return
        accuracy = Accuracy()
        accuracy.add(reference, self.simulation.prediction)
        report = accuracy.report()
        letter, word = report['letter_accuracy'], report['word_accuracy']
        self.metrics.set(f'Reference comparison · letters: {letter:.1%} · words: {word:.1%}' if letter is not None else 'No ambiguous letters to score.')

    def set_position(self, position):
        self.pause()
        if self.simulation is None:
            return
        position = min(max(0, position), max(0, len(self.simulation.tokens) - 1))
        self._schedule_update(position)

    def _position_entered(self, _event=None):
        try:
            self.set_position(int(self.position_var.get()))
        except ValueError:
            self.status.set('Character position must be an integer.')

    def _slider_changed(self, value):
        if not self.syncing:
            self.set_position(round(float(value)))

    def toggle_replay(self):
        if self.replay or self.play_job:
            self.pause()
            return
        if self.model is None:
            return
        try:
            tokens = parse_keys(self._keys())
        except ValueError as exc:
            self.status.set(f'Cannot replay: {exc}')
            return
        self.replay = [str(token) if isinstance(token, int) else encode_text(token) for token in tokens]
        if not self.replay:
            return
        self.play_button.configure(text='Pause replay')
        self.replace_keys('', replay=True)

    def _replay_step(self):
        self.play_job = None
        if self.replay:
            key = self.replay.pop(0)
            self.replace_keys(self._keys() + key, replay=True)

    def pause(self):
        self.replay = []
        self.play_button.configure(text='Replay typing')
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def _controls(self):
        state = ['!disabled'] if self.model is not None else ['disabled']
        for widget in (self.previous_button, self.next_button, self.position_spin, self.slider, self.play_button):
            widget.state(state)

    def _set_output(self, text):
        self.output.configure(state='normal')
        self.output.delete('1.0', 'end')
        self.output.insert('1.0', text)
        self.output.configure(state='disabled')

    def _show_result(self):
        result = self.simulation
        self._set_output(result.prediction)
        # Derive Tk indices from the prefix so multibyte/astral literals stay aligned.
        self.output.configure(state='normal')
        prefix = result.prediction[:self.position]
        index = self.output.index(f'1.0 + {self.root.tk.call("string", "length", prefix)} chars')
        self.output.tag_add('selected', index, f'{index}+1c')
        self.output.see(index)
        self.output.configure(state='disabled')
        self.syncing = True
        self.position_var.set(str(self.position))
        self.position_spin.configure(to=max(0, len(result.tokens) - 1))
        self.slider.configure(to=max(1, len(result.tokens) - 1))
        self.slider.set(self.position)
        self.syncing = False
        self.candidates.delete(*self.candidates.get_children())
        word = result.word_at(self.position)
        if word:
            for rank, (spelling, score) in enumerate(word.candidates):
                self.candidates.insert('', 'end', values=(spelling, f'{score:.4f}'), tags=('winner',) if rank == 0 else ())
        self.probabilities.delete(*self.probabilities.get_children())
        trace = self.trace
        if trace is None:
            self.position_info.set(f'Character {self.position} · supplied literal or end of input; no network prediction here.')
            self.context_info.set('Spaces, punctuation, and Unicode literals are copied and become context for later letters.')
        else:
            key = int(trace.keys[0])
            self.position_info.set(f'Character {self.position} / {len(result.tokens)} · key {key} ({KEY_LETTERS[key]}) · chosen letter {trace.letter!r} · positions start at 0')
            context = ' | '.join(byte_label(int(b)) for b in trace.history[-20:]) or '(no text context)'
            self.context_info.set(f'Predicted history, oldest → newest: {"… " if len(trace.history) > 20 else ""}{context}\nCurrent/upcoming word keys: {" ".join(map(str, trace.keys[:32]))}{" …" if len(trace.keys) > 32 else ""} · 0 = padding')
            for letter in sorted(KEY_LETTERS[key], key=lambda letter: -trace.probabilities[ord(letter) - 97]):
                self.probabilities.insert('', 'end', values=(letter, f'{trace.probabilities[ord(letter) - 97]:.3%}'), tags=('winner',) if letter == trace.letter else ())
        self._show_metrics()
        self.detail.set('Hover over an activation tile. Letter probabilities use the chosen spelling’s preceding history; beam search can choose a letter that is not the local maximum.')
        self.status.set('Ready · Predictions use their own history. Inspection sees the supplied remainder of this word. Replay shows predictions as keys arrive.')
        self._schedule_draw()

    def _schedule_draw(self):
        if not self.closing:
            if self.draw_job:
                self.root.after_cancel(self.draw_job)
            self.draw_job = self.root.after(30, self._draw)

    def _draw(self):
        self.draw_job = None
        canvas = self.canvas
        canvas.delete('all')
        self.cells.clear()
        if self.trace is None:
            canvas.create_text(25, 25, text='Load a T9 model and inspect a letter key to see activations.', anchor='nw', fill='#a6b8cd', font=('TkDefaultFont', 12))
            return
        trace = self.trace
        stages = [('Text embeddings', trace.text_features, 'text'), ('Key embeddings', trace.key_features, 'key')]
        stages += [(f'Hidden {i + 1} · tanh', values, i) for i, values in enumerate(trace.hidden)]
        stages += [('Letter probabilities', trace.probabilities, 'output')]
        gap, margin = 22, 18
        width = max(165, min(220, (canvas.winfo_width() - margin * 2 - gap * (len(stages) - 1)) / len(stages)))
        bottom = 0
        for stage_index, (name, values, layer) in enumerate(stages):
            x, y = margin + stage_index * (width + gap), 18
            groups = group_activations(values, 26 if layer == 'output' else int(self.tiles.get()))
            columns = 7 if layer == 'output' else 12
            cell = (width - 20) / columns
            height = max(1, (len(groups) + columns - 1) // columns) * cell + 96
            canvas.create_rectangle(x, y, x + width, y + height, fill='#142131', outline='#314258')
            canvas.create_text(x + 10, y + 13, text=name, anchor='nw', fill='#edf4ff', font=('TkDefaultFont', 10, 'bold'))
            canvas.create_text(x + 10, y + 36, text=f'{len(values):,} units', anchor='nw', fill='#9fb2ca')
            scale = 1.0 if isinstance(layer, int) else max(float(np.max(np.abs(values), initial=0)), 1e-12)
            for index, group in enumerate(groups):
                left, top = x + 10 + index % columns * cell, y + 60 + index // columns * cell
                tag = f'tile_{stage_index}_{index}'
                canvas.create_rectangle(left, top, left + cell - 1, top + cell - 1,
                                        fill=activation_color(group.value, scale, probability=layer == 'output'), outline='', tags=(tag,))
                self.cells[tag] = (name, layer, group)
                if layer == 'output':
                    canvas.create_text(left + cell / 2, top + cell / 2, text=chr(97 + index), fill='#ffffff', tags=(tag,))
            label = '1 tile / unit' if len(groups) == len(values) else f'{len(groups)} groups · peak |value|'
            canvas.create_text(x + 10, y + height - 20, text=label, anchor='nw', fill='#9fb2ca', font=('TkDefaultFont', 8))
            bottom = max(bottom, y + height)
        canvas.configure(scrollregion=(0, 0, margin + len(stages) * (width + gap) - gap, bottom + 20))

    def _hover(self, event):
        if self.trace is None:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        for item in reversed(self.canvas.find_overlapping(x, y, x, y)):
            for tag in self.canvas.gettags(item):
                if tag not in self.cells:
                    continue
                name, layer, group = self.cells[tag]
                if layer == 'output':
                    letter = chr(97 + group.start)
                    self.detail.set(f'Letter: {letter}\nProbability: {self.trace.probabilities[group.start]:.6%}\nRaw logit: {self.trace.logits[group.start]:.6g}\nChosen in word: {letter == self.trace.letter}\n\nOff-key letters are masked to zero probability.')
                else:
                    detail = f'{name}\nUnits {group.start}…{group.stop - 1}\nPeak unit: {group.peak_index}\nActivation: {group.value:+.6f}\nMean: {group.mean:+.6f}\nRange: {group.minimum:+.4f} … {group.maximum:+.4f}'
                    if layer in ('text', 'key'):
                        slot, component = divmod(group.peak_index, self.model.embedding)
                        value = byte_label(int(self.trace.history[slot])) if layer == 'text' else str(self.trace.keys[slot])
                        detail += f'\n\nInput slot: {slot}\nInput: {value}\nEmbedding component: {component}'
                    else:
                        detail += f'\n\nBefore tanh: {self.trace.preactivations[layer][group.peak_index]:+.6f}'
                    self.detail.set(detail)
                return

    def _show_detail(self):
        self.detail_view.configure(state='normal')
        self.detail_view.delete('1.0', 'end')
        self.detail_view.insert('1.0', self.detail.get())
        self.detail_view.configure(state='disabled')

    def close(self):
        if self.closing:
            return
        self.closing = True
        self._invalidate()
        self.pause()
        for job in (self.update_job, self.draw_job, self.poll_job):
            if job:
                self.root.after_cancel(job)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', type=Path, help='exported model.t9nn; otherwise choose one in the GUI')
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument('--text', default=None, help='ordinary text to convert to initial keystrokes')
    initial.add_argument('--keys', help='initial keystrokes in text_to_t9.py notation')
    parser.add_argument('--beam-width', type=int, default=8, help='word beam width, 1–64 (default 8)')
    parser.add_argument('--tiles', type=int, choices=(64, 128, 256, 512, 1024), default=256)
    args = parser.parse_args(argv)
    if not 1 <= args.beam_width <= 64:
        parser.error('--beam-width must be between 1 and 64')
    if args.keys is not None:
        try:
            parse_keys(args.keys)
        except ValueError as exc:
            parser.error(str(exc))
    if tk is None:
        parser.exit(2, 'Tkinter is required. On Debian/Ubuntu install python3-tk.\n')
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(2, f'Cannot open the desktop display: {exc}\n')
    window = T9Window(root, text=args.text if args.text is not None else ('' if args.keys is not None else 'Hello world!'),
                      keys=args.keys, beam_width=args.beam_width, tile_limit=args.tiles)
    if args.model:
        window.path.set(str(args.model))
        window.load_model()
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
