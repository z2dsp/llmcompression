"""Verify inspection against deployed T9 predictions and exported weights."""
from concurrent.futures import CancelledError
from dataclasses import replace
import threading

import numpy as np
import pytest

from llm_compression.t9 import CLASS_KEYS, decode_keys, encode_text
from llm_compression.t9_inspection import inspect_position, simulate
from llm_compression.t9_model import T9Model, export_model, shapes


def make_model(path, context=6, layers=2, storage='int8'):
    arrays = [np.random.default_rng(i).normal(0, 0.15, shape).astype('float32')
              for i, shape in enumerate(shapes(context, 4, 3, 7, layers))]
    export_model(path, arrays, context, 4, 3, 7, storage, layers)
    return T9Model(path)


@pytest.mark.parametrize('layers', [1, 2, 3])
@pytest.mark.parametrize('context', [0, 6])
@pytest.mark.parametrize('storage', ['int8', 'float32'])
def test_simulation_and_traces_match_runtime(tmp_path, layers, context, storage):
    model = make_model(tmp_path / 'model.t9nn', context, layers, storage)
    keys = encode_text('Hi 📻! café 2\\ GOOD home.')
    simulation = simulate(model, keys, beam_width=4)
    assert simulation.prediction == decode_keys(model, keys, beam_width=4)
    assert encode_text(simulation.prediction) == keys
    for word in simulation.words:
        assert word.candidates[0][0] == simulation.prediction[word.start:word.stop]
        assert [score for _, score in word.candidates] == sorted((score for _, score in word.candidates), reverse=True)
        # Every returned word score equals the sum of its conditional letter scores.
        for spelling, score in word.candidates:
            alternate = replace(simulation, prediction=simulation.prediction[:word.start] + spelling + simulation.prediction[word.stop:])
            expected = 0.0
            for position in range(word.start, word.stop):
                trace = inspect_position(model, alternate, position)
                expected += float(np.log(trace.probabilities[ord(trace.letter) - 97]))
            assert score == pytest.approx(expected, abs=2e-6)
    for position, token in enumerate(simulation.tokens):
        trace = inspect_position(model, simulation, position)
        if isinstance(token, str):
            assert trace is None
            continue
        assert len(trace.hidden) == len(trace.preactivations) == layers
        history = ([256] * context + list(simulation.prediction[:position].encode()))[-context:] if context else []
        assert trace.history.tolist() == history
        word = simulation.word_at(position)
        following = list(simulation.tokens[position:min(word.stop, position + model.key_context)])
        assert trace.keys.tolist() == following + [0] * (model.key_context - len(following))
        expected = model.logits([trace.history], [trace.keys])[0]
        np.testing.assert_allclose(trace.logits, expected, atol=1e-7)
        mask = CLASS_KEYS == token
        assert np.all(trace.probabilities[~mask] == 0)
        assert trace.probabilities.sum() == pytest.approx(1)
        weights = np.exp(expected[mask] - expected[mask].max())
        np.testing.assert_allclose(trace.probabilities[mask], weights / weights.sum(), atol=1e-7)
    assert inspect_position(model, simulation, len(simulation.tokens)) is None


def test_empty_literal_input_invalid_positions_and_cancellation(tmp_path):
    model = make_model(tmp_path / 'model.t9nn')
    for keys in ('', '0!📻\\2\\\\'):
        result = simulate(model, keys)
        assert result.words == ()
        assert inspect_position(model, result, len(result.tokens)) is None
        for position in (-1, len(result.tokens) + 1):
            with pytest.raises(ValueError, match='Position'):
                inspect_position(model, result, position)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        simulate(model, '4663', cancel=cancel)
    with pytest.raises(CancelledError):
        inspect_position(model, simulate(model, '4663'), 0, cancel=cancel)


def test_decoder_checks_cancellation_during_a_word(tmp_path):
    model = make_model(tmp_path / 'model.t9nn')
    cancel = threading.Event()
    original = model.logits
    calls = []

    def logits(*args):
        calls.append(1)
        cancel.set()
        return original(*args)

    model.logits = logits
    with pytest.raises(CancelledError):
        simulate(model, '4663', cancel=cancel)
    assert len(calls) == 1
