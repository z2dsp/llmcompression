import math
from concurrent.futures import CancelledError
import threading

import numpy as np
import pytest

from llm_compression import nn_inspection
from llm_compression.nn_inspection import MessageInspector, activation_color, byte_label, group_activations, inspect_position
from llm_compression.nn_model import NeuralModel, TOTAL, compress_payload, export_model, shapes


@pytest.fixture(params=[(depth, storage, context) for depth in (1, 2, 3)
                        for storage in ('int8', 'float32') for context in (0, 3, 32)])
def model(tmp_path, request):
    depth, storage, context = request.param
    rng = np.random.default_rng(81)
    arrays = [rng.normal(0, 0.2, shape).astype('float32') for shape in shapes(context, 2, 5, depth)]
    path = tmp_path / 'model.mcnn'
    export_model(path, arrays, context, 2, 5, storage, depth)
    return NeuralModel(path)


def test_trace_matches_codec_at_every_utf8_position_and_never_looks_ahead(model):
    message = 'Hi café 📻'.encode('utf-8')
    for position in range(len(message) + 1):
        trace = inspect_position(model, message, position)
        expected = ([256] * model.context + list(message[:position]))[-model.context:] if model.context else []
        assert trace.history.tolist() == expected
        assert trace.features.shape == (model.context * model.embedding,)
        assert len(trace.hidden) == len(trace.preactivations) == model.hidden_layers
        for z, hidden in zip(trace.preactivations, trace.hidden):
            np.testing.assert_array_equal(np.tanh(z), hidden)
            assert hidden.shape == (model.hidden,) and np.max(np.abs(hidden)) <= 1
        np.testing.assert_array_equal(trace.coding_probabilities * TOTAL, np.diff(model.cdf(expected)))
        assert trace.probabilities.sum() == pytest.approx(1)
        assert trace.coding_probabilities.sum() == 1 and trace.coding_probabilities.min() > 0
        assert 0 <= trace.entropy_bits <= 8
        if position < len(message):
            assert trace.actual_byte == message[position]
            assert trace.next_byte_bits == -math.log2(trace.coding_probabilities[message[position]])
        else:
            assert trace.actual_byte is trace.next_byte_bits is None
        changed_suffix = message[:position] + b'Entirely different future'
        other = inspect_position(model, changed_suffix, position)
        np.testing.assert_array_equal(trace.logits, other.logits)


@pytest.mark.parametrize('position', [-1, 2, 0.5])
def test_invalid_position_is_rejected(model, position):
    with pytest.raises(ValueError, match='Position'):
        inspect_position(model, b'x', position)


def test_display_groups_preserve_coverage_and_peak_sign_without_changing_values():
    values = np.array([0.1, -0.9, 0.2, 0.4, -0.8, 0.1, 0.0])
    before = values.copy()
    groups = group_activations(values, limit=3)
    assert [(g.start, g.stop) for g in groups] == [(0, 2), (2, 4), (4, 7)]
    assert [g.value for g in groups] == [-0.9, 0.4, -0.8]
    assert [g.peak_index for g in groups] == [1, 3, 4]
    assert groups[0].mean == pytest.approx(-0.4)
    np.testing.assert_array_equal(values, before)
    assert group_activations([], 256) == []
    assert len(group_activations(np.zeros(257), 256)) == 256
    assert len(group_activations(values, 100)) == len(values)
    with pytest.raises(ValueError, match='positive'):
        group_activations(values, 0)


def test_byte_labels_and_activation_colors():
    assert byte_label(256) == 'BOS'
    assert byte_label(32) == '20 SPACE'
    assert byte_label(10) == '0A LF'
    assert byte_label(97) == "61 'a'"
    assert byte_label(255) == 'FF'
    assert byte_label(None) == 'End of input'
    assert activation_color(0) == '#1d2939'
    assert activation_color(-1) == '#49a8ff'
    assert activation_color(1) == '#ffb05c'
    assert activation_color(2) == activation_color(1)
    assert activation_color(0.3, 0.3, probability=True) == '#48dda6'
    assert activation_color(0, 0) == '#1d2939'


def test_prefix_totals_and_payload_size_match_standalone_encoding(model):
    message = 'café 📻'.encode('utf-8')
    inspector = MessageInspector(model, message)
    expected_bits = 0.0
    for position in range(len(message) + 1):
        if position < len(message):
            expected_bits += inspect_position(model, message, position).next_byte_bits
        result = inspector.inspect(position)
        count = min(position + 1, len(message))
        assert result.trace.position == position
        assert result.encoded_bytes == count
        assert result.total_bits == pytest.approx(expected_bits)
        assert result.payload_bytes == len(compress_payload(message[:count], model))
    # Backtracking must choose the previous prefix, rather than retain the full total.
    assert inspector.inspect(0).total_bits == inspect_position(model, message, 0).next_byte_bits
    empty = MessageInspector(model, b'').inspect(0)
    assert empty.encoded_bytes == empty.total_bits == 0
    assert empty.trace.actual_byte is None
    assert empty.payload_bytes == len(compress_payload(b'', model))


def test_prefix_navigation_reuses_already_encoded_bytes(model, monkeypatch):
    calls = []
    def counted(*args):
        calls.append(args[2])
        return inspect_position(*args)
    monkeypatch.setattr(nn_inspection, 'inspect_position', counted)
    inspector = MessageInspector(model, b'abcdef')
    inspector.inspect(3)
    assert calls == [0, 1, 2, 3]
    result = inspector.inspect(1)
    assert calls == [0, 1, 2, 3, 1]
    assert inspector.inspect(1) is not None
    assert calls == [0, 1, 2, 3, 1]
    inspector.inspect(5)
    assert calls == [0, 1, 2, 3, 1, 4, 5]
    assert result.payload_bytes == len(compress_payload(b'ab', model))


def test_cancelled_prefix_can_resume_without_double_counting(model, monkeypatch):
    event, calls = threading.Event(), []
    def interrupted(*args):
        position = args[2]
        calls.append(position)
        if position == 1:
            event.set()
        return inspect_position(*args)
    monkeypatch.setattr(nn_inspection, 'inspect_position', interrupted)
    message = b'abcdef'
    inspector = MessageInspector(model, message)
    with pytest.raises(CancelledError):
        inspector.inspect(5, cancel=event)
    assert calls == [0, 1]
    event.clear()
    result = inspector.inspect(5, cancel=event)
    assert calls == list(range(6))
    assert result.total_bits == pytest.approx(sum(inspect_position(model, message, i).next_byte_bits for i in range(6)))
    assert result.payload_bytes == len(compress_payload(message, model))
