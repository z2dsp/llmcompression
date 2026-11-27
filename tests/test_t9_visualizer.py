from concurrent.futures import Future
from types import SimpleNamespace
import time

import pytest

from llm_compression import t9_visualizer
from llm_compression.t9 import decode_keys, encode_text
from llm_compression.t9_visualizer import backspace_keys, evaluate_input
from test_t9_inspection import make_model


@pytest.fixture
def model(tmp_path):
    return make_model(tmp_path / 'model.t9nn')


@pytest.mark.parametrize(('keys', 'expected'), [('', ''), ('23', '2'), ('23\\2', '23'),
                                            ('23\\\\', '23'), ('23\\', '23'), ('23📻', '23')])
def test_backspace_removes_one_token(keys, expected):
    assert backspace_keys(keys) == expected


def test_evaluate_navigation_reuses_prediction(model, monkeypatch):
    result, position, trace = evaluate_input(model, '466304663!', 4, None, None, cancel=None)
    assert position == trace.position == 8
    assert result.prediction == decode_keys(model, result.keystrokes, beam_width=4)
    monkeypatch.setattr(t9_visualizer, 'simulate', lambda *a, **kw: pytest.fail('Navigation must reuse simulation'))
    reused, position, trace = evaluate_input(model, result.keystrokes, 4, 0, result, cancel=None)
    assert reused is result and position == trace.position == 0


@pytest.fixture
def window():
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('desktop display unavailable; run with a real or virtual X display')
    root.withdraw()
    app = t9_visualizer.T9Window(root, text='Hi 📻! café 2\\ home')
    errors = []
    root.report_callback_exception = lambda *error: errors.append(error)
    yield app
    app.close()
    assert not errors, errors


def until(app, condition):
    deadline = time.monotonic() + 8
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail('T9 viewer timed out: ' + app.status.get())
        app.root.update()
        time.sleep(0.005)
    app.root.update()


def load(app, model):
    # The model fixture always writes its file to tmp_path.
    app.path.set(str(model))
    app.load_model()
    until(app, lambda: app.simulation is not None)


def test_load_type_convert_inspect_and_clear(window, model, tmp_path):
    app = window
    assert app.play_button.instate(['disabled'])
    load(app, tmp_path / 'model.t9nn')
    assert '2 hidden layers' in app.metadata.get()
    assert app.simulation.prediction == decode_keys(app.model, app._keys())
    assert 'Reference comparison' in app.metrics.get()
    app.set_position(0)
    until(app, lambda: app.trace is not None and app.trace.position == 0)
    assert app.trace.history.tolist() == [256] * 6
    assert 3 <= len(app.probabilities.get_children()) <= 4
    app.set_position(3)  # The radio emoji is a supplied literal, not a target.
    until(app, lambda: app.position == 3 and app.trace is None)
    assert 'supplied literal' in app.position_info.get()
    app.replace_keys('4663')
    until(app, lambda: app.simulation is not None and app.simulation.keystrokes == '4663')
    app.type_key('0')
    app.type_key('2')
    until(app, lambda: app.simulation is not None and app.simulation.keystrokes == '466302')
    assert app.trace.history[-1] == ord(' ')
    app.type_key('\\2')
    app.backspace()
    assert app._keys() == '466302'
    app.reference.delete('1.0', 'end')
    app.reference.insert('1.0', 'Good home')
    app.convert_text()
    until(app, lambda: app.simulation is not None and app.simulation.keystrokes == encode_text('Good home'))
    assert 'Reference comparison' in app.metrics.get()
    app.replace_keys('')
    until(app, lambda: app.simulation is not None and app.simulation.keystrokes == '')
    assert app.simulation.prediction == '' and app.trace is None


def test_render_hover_and_replay(window, model, tmp_path):
    app = window
    app.root.deiconify()
    load(app, tmp_path / 'model.t9nn')
    until(app, lambda: len(app.cells) == 18 + 12 + 2 * 7 + 26)
    assert app.detail_view.winfo_height() >= 60
    assert app.status_label.winfo_ismapped()
    assert app.status_label.winfo_rooty() + app.status_label.winfo_height() <= app.root.winfo_rooty() + app.root.winfo_height()
    for layer in ('text', 'key', 0, 'output'):
        tag = next(tag for tag, (_, item_layer, _) in app.cells.items() if item_layer == layer)
        item = app.canvas.find_withtag(tag)[0]
        left, top, right, bottom = app.canvas.coords(item)
        app._hover(SimpleNamespace(x=(left + right) / 2, y=(top + bottom) / 2))
        assert ('Embedding component' if layer in ('text', 'key') else 'Probability' if layer == 'output' else 'Before tanh') in app.detail.get()
    app.replace_keys('2\\2📻')
    until(app, lambda: app.simulation is not None and app.simulation.keystrokes == '2\\2📻')
    app.toggle_replay()
    until(app, lambda: app.simulation is not None and app.simulation.keystrokes == '')
    until(app, lambda: not app.replay and app.play_job is None and app.simulation.keystrokes == '2\\2📻')
    assert app._keys() == '2\\2📻'


def test_invalid_input_beam_and_model_recover(window, model, tmp_path):
    app = window
    load(app, tmp_path / 'model.t9nn')
    app.replace_keys('hello')
    until(app, lambda: 'Could not predict input' in app.status.get())
    assert app.trace is None and app.simulation is None
    app.replace_keys('4663')
    until(app, lambda: app.simulation is not None)
    app.beam.set('bad')
    until(app, lambda: 'Beam width must' in app.status.get())
    app.beam.set('1')
    until(app, lambda: app.simulation is not None)
    assert len(app.candidates.get_children()) == 1
    app.path.set(str(tmp_path / 'missing.t9nn'))
    app.load_model()
    until(app, lambda: 'Could not load model' in app.status.get())
    assert app.model is None and app.play_button.instate(['disabled'])


def test_edits_during_loading_and_stale_results(window, model, tmp_path, monkeypatch):
    app = window
    pending = Future()
    monkeypatch.setattr(app.executor, 'submit', lambda *args: pending)
    app.path.set(str(tmp_path / 'model.t9nn'))
    app.load_model()
    token = app.request_id
    app.replace_keys('22')
    app.root.update()
    assert app.request_id == token
    monkeypatch.undo()
    pending.set_result(model)
    until(app, lambda: app.simulation is not None)
    assert app.simulation.keystrokes == '22'
    old = app.simulation
    stale = Future()
    stale.set_exception(RuntimeError('stale failure'))
    app.events.put((app.request_id - 1, 'input', stale))
    until(app, lambda: app.events.empty())
    assert app.simulation is old and 'stale failure' not in app.status.get()


def test_cli_validation_without_display(monkeypatch, capsys):
    if t9_visualizer.tk is not None:
        monkeypatch.setattr(t9_visualizer.tk, 'Tk', lambda: pytest.fail('Must not open a window'))
    for args, code in ((['--help'], 0), (['--beam-width', '0'], 2),
                       (['--keys', 'hello'], 2), (['--text', 'hi', '--keys', '44'], 2)):
        with pytest.raises(SystemExit) as exc:
            t9_visualizer.main(args)
        assert exc.value.code == code
    assert '--keys' in capsys.readouterr().out
