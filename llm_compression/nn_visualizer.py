"""Tk desktop viewer for exported .mcnn neural compression model activations."""
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

from .nn_inspection import MessageInspector, activation_color, byte_label, group_activations
from .nn_model import NeuralModel


class NetworkWindow:
    def __init__(self, root, *, text='Hello mesh!', position=0, tile_limit=256):
        self.root = root
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nn-viewer')
        self.events = queue.Queue()
        self.future = None
        self.cancel_event = threading.Event()
        self.request_id = 0
        self.model = self.trace = None
        self.inspector = self.inspection = None
        self.message = text.encode('utf-8')
        self.position = min(max(0, position), len(self.message))
        self.playing = self.closing = self.syncing = False
        self.update_job = self.play_job = self.poll_job = self.draw_job = None
        self.cells = {}
        self.path = tk.StringVar(root)
        self.metadata = tk.StringVar(root, value='Choose an exported model.mcnn file. One, two, and three hidden layers are supported.')
        self.status = tk.StringVar(root, value='Ready to load a model. All computation stays on this computer.')
        self.position_var = tk.StringVar(root, value=str(self.position))
        self.position_info = tk.StringVar(root)
        self.cost_info = tk.StringVar(root)
        self.context_info = tk.StringVar(root)
        self.detail = tk.StringVar(root, value='Hover over a tile to inspect its neuron or group.\n\nBlue: negative activation\nAmber: positive activation\nDark: near zero\nGreen: next-byte probability')
        self.tiles = tk.StringVar(root, value=str(tile_limit))
        self.delay = tk.StringVar(root, value='400')
        self.prediction_info = tk.StringVar(root)
        self._build(text)
        self._read_message()
        self.poll_job = root.after(30, self._poll)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def _build(self, text):
        root = self.root
        root.title('Neural compression · Activation viewer')
        root.geometry('1320x860')
        root.minsize(1000, 700)
        style = ttk.Style(root)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure('Title.TLabel', font=('TkDefaultFont', 18, 'bold'))
        style.configure('Muted.TLabel', foreground='#58677d')
        body = ttk.Frame(root, padding=16)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text='Neural compression explorer', style='Title.TLabel').pack(anchor='w')
        ttk.Label(body, text='Follow a message through the network, one UTF-8 byte at a time.', style='Muted.TLabel').pack(anchor='w', pady=(3, 12))
        files = ttk.Frame(body)
        files.pack(fill='x')
        ttk.Label(files, text='Model').pack(side='left', padx=(0, 8))
        ttk.Entry(files, textvariable=self.path).pack(side='left', fill='x', expand=True)
        ttk.Button(files, text='Browse…', command=self.browse).pack(side='left', padx=6)
        ttk.Button(files, text='Load', command=self.load_model).pack(side='left')
        ttk.Label(body, textvariable=self.metadata, style='Muted.TLabel').pack(anchor='w', pady=(6, 12))

        entry = ttk.LabelFrame(body, text='Message', padding=8)
        entry.pack(fill='x')
        self.composer = scrolledtext.ScrolledText(entry, height=3, wrap='word', font=('TkFixedFont', 11), undo=True)
        self.composer.pack(fill='x')
        self.composer.insert('1.0', text)
        self.composer.edit_modified(False)
        self.composer.bind('<<Modified>>', self._edited)
        controls = ttk.Frame(body)
        controls.pack(fill='x', pady=(10, 4))
        self.previous_button = ttk.Button(controls, text='◀ Previous', command=lambda: self.set_position(self.position - 1))
        self.previous_button.pack(side='left')
        self.play_button = ttk.Button(controls, text='Play', command=self.toggle_play)
        self.play_button.pack(side='left', padx=6)
        self.next_button = ttk.Button(controls, text='Next ▶', command=lambda: self.set_position(self.position + 1))
        self.next_button.pack(side='left')
        ttk.Label(controls, text='Byte').pack(side='left', padx=(18, 4))
        self.position_spin = ttk.Spinbox(controls, from_=0, to=len(self.message), width=7,
                                        textvariable=self.position_var, command=self._position_entered)
        self.position_spin.pack(side='left')
        self.position_spin.bind('<Return>', self._position_entered)
        ttk.Label(controls, text='Delay (ms)').pack(side='left', padx=(18, 4))
        ttk.Spinbox(controls, from_=50, to=3000, increment=50, width=6, textvariable=self.delay).pack(side='left')
        ttk.Label(controls, text='Neuron tile limit').pack(side='left', padx=(18, 4))
        selector = ttk.Combobox(controls, textvariable=self.tiles, values=('64', '128', '256', '512', '1024'), width=6, state='readonly')
        selector.pack(side='left')
        selector.bind('<<ComboboxSelected>>', lambda _: self._schedule_draw())
        self.slider = ttk.Scale(body, from_=0, to=max(1, len(self.message)), command=self._slider_changed)
        self.slider.pack(fill='x')
        ttk.Label(body, textvariable=self.position_info).pack(anchor='w', pady=(4, 2))
        ttk.Label(body, textvariable=self.cost_info, font=('TkDefaultFont', 10, 'bold'),
                  wraplength=1200).pack(anchor='w', pady=(0, 3))
        ttk.Label(body, textvariable=self.context_info, style='Muted.TLabel').pack(anchor='w', pady=(0, 10))

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
        sidebar = ttk.Frame(panels, width=255, padding=(14, 0, 0, 0))
        sidebar.grid(row=0, column=1, sticky='ns')
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1, minsize=80)
        ttk.Label(sidebar, text='Next-byte predictions', font=('TkDefaultFont', 11, 'bold')).grid(row=0, column=0, sticky='w')
        prediction_frame = ttk.Frame(sidebar)
        prediction_frame.grid(row=1, column=0, sticky='ew', pady=(6, 4))
        self.predictions = ttk.Treeview(prediction_frame, columns=('byte', 'probability'), show='headings', height=6, selectmode='none')
        self.predictions.heading('byte', text='Byte')
        self.predictions.heading('probability', text='Probability')
        self.predictions.column('byte', width=112, stretch=True)
        self.predictions.column('probability', width=105, anchor='e', stretch=True)
        self.predictions.tag_configure('actual', background='#d8f0e7')
        self.predictions.pack(side='left', fill='both', expand=True)
        prediction_scroll = ttk.Scrollbar(prediction_frame, command=self.predictions.yview)
        prediction_scroll.pack(side='right', fill='y')
        self.predictions.configure(yscrollcommand=prediction_scroll.set)
        ttk.Label(sidebar, textvariable=self.prediction_info, wraplength=232, justify='left').grid(row=2, column=0, sticky='w', pady=(0, 12))
        ttk.Separator(sidebar).grid(row=3, column=0, sticky='ew', pady=6)
        sidebar.rowconfigure(4, weight=1, minsize=60)
        self.detail_view = scrolledtext.ScrolledText(sidebar, height=5, width=24, wrap='word',
                                                   font=('TkDefaultFont', 10), relief='flat', state='disabled')
        self.detail_view.grid(row=4, column=0, sticky='nsew', pady=6)
        self.detail.trace_add('write', lambda *_: self._show_detail())
        self._show_detail()
        ttk.Label(body, text='Hidden colors use a fixed −1…+1 scale. Embedding and output brightness scale per frame. Grouped tiles show the strongest absolute activation.',
                  style='Muted.TLabel', wraplength=1200).pack(anchor='w', pady=(9, 4))
        ttk.Label(body, textvariable=self.status, style='Muted.TLabel', wraplength=1200).pack(anchor='w')
        self._controls()

    def browse(self):
        name = filedialog.askopenfilename(parent=self.root, title='Open trained neural model',
                                          filetypes=[('Neural compression model', '*.mcnn'), ('All files', '*')])
        if name:
            self.path.set(name)
            self.load_model()

    def _submit(self, kind, function, *args):
        self.request_id += 1
        token = self.request_id
        if self.future is not None:
            self.future.cancel()
        self.cancel_event.set()
        self.cancel_event = threading.Event()
        kwargs = {'cancel': self.cancel_event} if kind == 'trace' else {}
        self.future = self.executor.submit(function, *args, **kwargs)
        self.future.add_done_callback(lambda future: self.events.put((token, kind, future)))

    def load_model(self):
        name = self.path.get().strip()
        if not name:
            self.status.set('Choose a model.mcnn file first.')
            return
        self.pause()
        self.model = self.trace = None
        self.inspector = self.inspection = None
        self.cost_info.set('')
        self.cells.clear()
        self.metadata.set('Loading model…')
        self.predictions.delete(*self.predictions.get_children())
        self.prediction_info.set('')
        self._controls()
        self._schedule_draw()
        self.status.set('Loading model…')
        self._submit('model', NeuralModel, Path(name).expanduser())

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
                if kind == 'model':
                    self.metadata.set('No model loaded. Choose another model.mcnn file.')
                self.status.set(f'Could not {"load model" if kind == "model" else "evaluate input"}: {exc}')
                self._controls()
                continue
            if kind == 'model':
                self.model = result
                self.metadata.set(f'{result.size_bytes / 1024:,.1f} KiB · {result.storage} · context {result.context:,} bytes · '
                                  f'embedding {result.embedding} · {result.hidden_layers} hidden layers × {result.hidden:,} neurons')
                self._read_message()
                self._controls()
                self.request_update()
            else:
                self.inspection = result
                self.trace = result.trace
                self._show_trace()
                if self.playing and self.position < len(self.message):
                    try:
                        delay = max(50, min(3000, int(self.delay.get())))
                    except ValueError:
                        delay = 400
                    self.play_job = self.root.after(delay, self._play_step)
                else:
                    self.pause()
        self.poll_job = self.root.after(30, self._poll)

    def _edited(self, _event=None):
        if not self.composer.edit_modified():
            return
        self.composer.edit_modified(False)
        self.pause()
        # Invalidate results for the previous draft before the debounce expires.
        if self.model is not None:
            self.request_id += 1
            self.cancel_event.set()
        self.cost_info.set('')
        if self.update_job:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(180, self.request_update)

    def _read_message(self):
        self.message = self.composer.get('1.0', 'end-1c').encode('utf-8')
        self.position = min(self.position, len(self.message))
        self.syncing = True
        self.position_var.set(str(self.position))
        self.position_spin.configure(to=len(self.message))
        self.slider.configure(to=max(1, len(self.message)))
        self.slider.set(self.position)
        self.syncing = False
        self.position_info.set(f'Predicting byte {self.position:,} of {len(self.message):,} UTF-8 bytes (positions start at 0).')

    def request_update(self):
        self.update_job = None
        if self.closing:
            return
        try:
            self._read_message()
        except UnicodeError as exc:
            self.status.set(f'Input is not valid UTF-8: {exc}')
            return
        if self.model is not None:
            if self.inspector is None or self.inspector.model is not self.model or self.inspector.message != self.message:
                self.inspector = MessageInspector(self.model, self.message)
            self.status.set('Evaluating input…')
            self.cost_info.set('Calculating prefix cost…')
            self._submit('trace', self.inspector.inspect, self.position)

    def _slider_changed(self, value):
        if not self.syncing:
            self.set_position(round(float(value)))

    def _position_entered(self, _event=None):
        try:
            self.set_position(int(self.position_var.get()))
        except ValueError:
            self.status.set('Byte position must be an integer.')

    def set_position(self, position, *, playback=False):
        if not playback:
            self.pause()
        self.position = min(max(0, position), len(self.composer.get('1.0', 'end-1c').encode('utf-8')))
        if self.model is not None:
            self.request_id += 1
            self.cancel_event.set()
        if self.update_job:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(25, self.request_update)

    def toggle_play(self):
        if self.playing:
            self.pause()
        elif self.model is not None:
            self.playing = True
            self.play_button.configure(text='Pause')
            if self.position >= len(self.message):
                self.position = 0
            self.request_update()

    def _play_step(self):
        self.play_job = None
        if self.playing:
            self.set_position(self.position + 1, playback=True)

    def pause(self):
        self.playing = False
        self.play_button.configure(text='Play')
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def _controls(self):
        state = ['!disabled'] if self.model is not None else ['disabled']
        for widget in (self.previous_button, self.next_button, self.play_button, self.position_spin, self.slider):
            widget.state(state)

    def _show_trace(self):
        trace = self.trace
        context = trace.history.tolist()
        visible = context[-24:]
        prefix = f'… {len(context) - 24:,} earlier context bytes · ' if len(context) > 24 else ''
        self.context_info.set('Context, oldest → newest: ' + prefix + (' | '.join(byte_label(b) for b in visible) or '(none; unconditional model)'))
        actual = byte_label(trace.actual_byte)
        cost = f' · this byte: {trace.next_byte_bits:.3f} bits' if trace.actual_byte is not None else ' · predicting beyond the message'
        self.position_info.set(f'Byte {trace.position:,} / {len(self.message):,} · actual: {actual}{cost}')
        prefix = f'Through byte {trace.position:,}' if trace.actual_byte is not None else 'Complete message'
        input_unit = 'byte' if self.inspection.encoded_bytes == 1 else 'bytes'
        payload_unit = 'byte' if self.inspection.payload_bytes == 1 else 'bytes'
        self.cost_info.set(f'{prefix} ({self.inspection.encoded_bytes:,} {input_unit}): {self.inspection.total_bits:.3f} bits '
                           f'({self.inspection.total_bits / 8:.3f} bytes equivalent) · Payload if ended here: {self.inspection.payload_bytes:,} {payload_unit}')
        self.predictions.delete(*self.predictions.get_children())
        top = np.argsort(-trace.coding_probabilities, kind='stable')[:10].tolist()
        if trace.actual_byte is not None and trace.actual_byte not in top:
            top.append(trace.actual_byte)
        for byte in top:
            self.predictions.insert('', 'end', values=(byte_label(byte), f'{trace.coding_probabilities[byte]:.4%}'),
                                    tags=('actual',) if byte == trace.actual_byte else ())
        comparison = (f'Actual {byte_label(trace.actual_byte)}: {trace.coding_probabilities[trace.actual_byte]:.4%}'
                      if trace.actual_byte is not None else 'End of input; no actual next byte.')
        self.prediction_info.set(f'Codec entropy: {trace.entropy_bits:.3f} bits\n{comparison}')
        self.status.set('Ready · Total uses codec probabilities. Payload includes termination/padding; excludes framing and the shared model.')
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
            canvas.create_text(30, 35, text='Load a model to see its activations.', anchor='nw', fill='#a6b8cd', font=('TkDefaultFont', 14))
            return
        trace = self.trace
        stages = [('Embeddings', trace.features, None)]
        stages += [(f'Hidden {i + 1} · tanh', values, i) for i, values in enumerate(trace.hidden)]
        stages += [('Next byte', trace.coding_probabilities, 'output')]
        gap, margin = 28, 22
        width = max(156, min(225, (canvas.winfo_width() - margin * 2 - gap * (len(stages) - 1)) / len(stages)))
        bottom = 0
        for stage_index, (name, values, layer) in enumerate(stages):
            x, y = margin + stage_index * (width + gap), 20
            limit = 256 if layer == 'output' else int(self.tiles.get())
            groups = group_activations(values, limit)
            cell = (width - 20) / 16
            height = max(1, (len(groups) + 15) // 16) * cell + 98
            canvas.create_rectangle(x, y, x + width, y + height, fill='#142131', outline='#314258')
            canvas.create_text(x + 10, y + 14, text=name, anchor='nw', fill='#edf4ff', font=('TkDefaultFont', 11, 'bold'))
            count_label = f'{len(values):,} ' + ('bytes' if layer == 'output' else 'units')
            canvas.create_text(x + 10, y + 38, text=count_label, anchor='nw', fill='#9fb2ca', font=('TkDefaultFont', 9))
            grouped = len(values) > len(groups)
            scale = max(float(np.max(np.abs(values), initial=0)), 1e-12) if layer is None or layer == 'output' else 1.0
            for index, group in enumerate(groups):
                left, top = x + 10 + (index % 16) * cell, y + 64 + (index // 16) * cell
                item = canvas.create_rectangle(left, top, left + cell - 1, top + cell - 1,
                                               fill=activation_color(group.value, scale, probability=layer == 'output'), outline='', tags='activation')
                self.cells[item] = (name, layer, group)
            label = '1 tile / byte' if layer == 'output' else (f'{len(groups)} groups · peak |value|' if grouped else '1 tile / unit')
            canvas.create_text(x + 10, y + height - 22, text=label, anchor='nw', fill='#9fb2ca', font=('TkDefaultFont', 8))
            if not len(values):
                canvas.create_text(x + 10, y + 65, text='No context features', anchor='nw', fill='#9fb2ca', font=('TkDefaultFont', 8))
            if stage_index < len(stages) - 1:
                canvas.create_line(x + width + 3, y + 100, x + width + gap - 3, y + 100, arrow='last', fill='#7b93ac', width=2)
            bottom = max(bottom, y + height)
        canvas.configure(scrollregion=(0, 0, margin + len(stages) * (width + gap) - gap, bottom + 20))

    def _hover(self, event):
        if self.trace is None or self.model is None:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        for item in reversed(self.canvas.find_overlapping(x, y, x, y)):
            if item not in self.cells:
                continue
            name, layer, group = self.cells[item]
            if layer == 'output':
                byte = group.start
                self.detail.set(f'Next byte: {byte_label(byte)}\n\nCodec probability: {self.trace.coding_probabilities[byte]:.6%}\n'
                                f'Softmax probability: {self.trace.probabilities[byte]:.6%}\nLogit: {self.trace.logits[byte]:.6g}')
            else:
                units = f'Unit {group.start}' if group.stop == group.start + 1 else f'Units {group.start}–{group.stop - 1}'
                detail = f'{name}\n{units}\nActivation: {group.value:+.6f}\nPeak unit: {group.peak_index}\n'
                if group.stop > group.start + 1:
                    detail += f'Mean: {group.mean:+.6f}\nRange: {group.minimum:+.4f} … {group.maximum:+.4f}\n'
                if layer is None:
                    position, component = divmod(group.peak_index, self.model.embedding)
                    detail += f'\nContext slot: {position}\nEmbedding component: {component}\nInput: {byte_label(int(self.trace.history[position]))}'
                else:
                    detail += f'\nBefore tanh: {self.trace.preactivations[layer][group.peak_index]:+.6f}'
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
        self.cancel_event.set()
        self.pause()
        for job in (self.update_job, self.draw_job, self.poll_job):
            if job:
                self.root.after_cancel(job)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', type=Path, help='exported model.mcnn; otherwise choose one in the GUI')
    parser.add_argument('--text', default='Hello mesh!', help='initial input message')
    parser.add_argument('--position', type=int, default=0, help='initial UTF-8 byte position, starting at 0')
    parser.add_argument('--tiles', type=int, choices=(64, 128, 256, 512, 1024), default=256,
                        help='maximum hidden/embedding tiles per layer; output always shows all 256 bytes')
    args = parser.parse_args(argv)
    if args.position < 0:
        parser.error('--position must be nonnegative')
    if tk is None:
        parser.exit(2, 'Tkinter is required. On Debian/Ubuntu install python3-tk.\n')
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(2, f'Cannot open the desktop display: {exc}\n')
    window = NetworkWindow(root, text=args.text, position=args.position, tile_limit=args.tiles)
    if args.model:
        window.path.set(str(args.model))
        window.load_model()
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
