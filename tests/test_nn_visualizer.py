from concurrent.futures import Future
from types import SimpleNamespace
import time

import numpy as np
import pytest

tk = pytest.importorskip('tkinter')
from llm_compression import nn_visualizer
from llm_compression.nn_inspection import inspect_position
from llm_compression.nn_model import NeuralModel, compress_payload, export_model, shapes


@pytest.fixture
def window():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('desktop display unavailable; run with a real or virtual X display')
    root.withdraw()
    app = nn_visualizer.NetworkWindow(root, text='Hi café 📻')
    errors = []
    root.report_callback_exception = lambda *error: errors.append(error)
    yield app
    app.close()
    assert not errors, errors


@pytest.fixture
def model_path(tmp_path):
    rng = np.random.default_rng(42)
    arrays = [rng.normal(0, 0.2, shape).astype('float32') for shape in shapes(3, 2, 6)]
    path = tmp_path / 'model.mcnn'
    export_model(path, arrays, 3, 2, 6, 'int8')
    return path


def until(app, condition):
    deadline = time.monotonic() + 5
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail('Viewer timed out: ' + app.status.get())
        app.root.update()
        time.sleep(0.005)
    app.root.update()


def load(app, path):
    app.path.set(str(path))
    app.load_model()
    until(app, lambda: app.trace is not None)


def test_load_step_edit_empty_input_and_playback(window, model_path):
    app = window
    assert app.play_button.instate(['disabled'])
    load(app, model_path)
    assert app.play_button.instate(['!disabled'])
    assert app.trace.position == 0 and app.trace.history.tolist() == [256] * 3
    assert app.inspection.encoded_bytes == 1
    assert app.inspection.total_bits == app.trace.next_byte_bits
    assert 'Through byte 0 (1 byte)' in app.cost_info.get()
    assert '3 hidden layers' in app.metadata.get()
    app.set_position(7)  # Inside the UTF-8 encoding of é, rather than at a character boundary.
    until(app, lambda: app.trace.position == 7)
    np.testing.assert_array_equal(app.trace.history, list(app.message[4:7]))
    expected = sum(inspect_position(app.model, app.message, i).next_byte_bits for i in range(8))
    assert app.inspection.total_bits == pytest.approx(expected)
    assert app.inspection.payload_bytes == len(compress_payload(app.message[:8], app.model))
    assert len(app.predictions.get_children()) in (10, 11)
    app.delay.set('50')
    app.toggle_play()
    until(app, lambda: app.trace.position == len(app.message) and not app.playing)
    assert app.trace.actual_byte is None
    assert app.inspection.encoded_bytes == len(app.message)
    assert app.inspection.payload_bytes == len(compress_payload(app.message, app.model))
    assert 'Complete message' in app.cost_info.get()
    old_inspector = app.inspector
    app.composer.delete('1.0', 'end')
    until(app, lambda: app.message == b'' and app.trace.position == 0)
    assert app.trace.actual_byte is None
    assert app.trace.history.tolist() == [256] * 3
    assert app.inspector is not old_inspector
    assert app.inspection.total_bits == app.inspection.encoded_bytes == 0


def test_render_tiles_and_hover(window, model_path):
    app = window
    app.root.deiconify()
    load(app, model_path)
    until(app, lambda: len(app.cells) == 6 + 3 * 6 + 256)
    assert app.detail_view.winfo_height() >= 60
    counts = set(layer for _, layer, _ in app.cells.values())
    assert counts == {None, 0, 1, 2, 'output'}
    for wanted in (None, 0, 'output'):
        item = next(item for item, (_, layer, _) in app.cells.items() if layer == wanted)
        left, top, right, bottom = app.canvas.coords(item)
        app._hover(SimpleNamespace(x=(left + right) / 2, y=(top + bottom) / 2))
        assert 'Embedding component' in app.detail.get() if wanted is None else (
            'Codec probability' in app.detail.get() if wanted == 'output' else 'Before tanh' in app.detail.get())


def test_bad_model_and_invalid_position_are_reported(window, tmp_path):
    app = window
    app.path.set(str(tmp_path / 'missing.mcnn'))
    app.load_model()
    until(app, lambda: 'Could not load model' in app.status.get())
    assert app.model is None and app.play_button.instate(['disabled'])
    app.position_var.set('not a number')
    app._position_entered()
    assert 'integer' in app.status.get()


def test_editing_while_loading_does_not_discard_model(window, model_path, monkeypatch):
    app = window
    model = NeuralModel(model_path)
    pending = Future()
    monkeypatch.setattr(app.executor, 'submit', lambda *args: pending)
    app.path.set(str(model_path))
    app.load_model()
    token = app.request_id
    app.composer.insert('end', '!')
    app.root.update()
    assert app.request_id == token
    # Restore normal inference after completing the deliberately delayed load.
    monkeypatch.undo()
    pending.set_result(model)
    until(app, lambda: app.trace is not None and app.message.endswith(b'!'))


def test_stale_worker_result_is_ignored(window, model_path):
    app = window
    load(app, model_path)
    old = app.trace
    stale = Future()
    stale.set_exception(RuntimeError('old failure must not overwrite current state'))
    app.events.put((app.request_id - 1, 'trace', stale))
    app.root.update()
    until(app, lambda: app.events.empty())
    assert app.trace is old and 'old failure' not in app.status.get()


def test_cli_help_and_invalid_position_do_not_open_a_window(monkeypatch, capsys):
    monkeypatch.setattr(nn_visualizer.tk, 'Tk', lambda: pytest.fail('Must not open a window'))
    with pytest.raises(SystemExit) as error:
        nn_visualizer.main(['--help'])
    assert error.value.code == 0
    assert '--tiles' in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        nn_visualizer.main(['--position', '-1'])
    assert error.value.code == 2
