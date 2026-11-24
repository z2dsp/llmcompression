"""Inspect exported neural predictors without PyTorch or GUI dependencies."""
from dataclasses import dataclass
from concurrent.futures import CancelledError
import math

import numpy as np

from .nn_model import NeuralModel, TOTAL, probabilities_to_cdf
from .arithmetic import ArithmeticEncoder


@dataclass(frozen=True)
class ActivationTrace:
    position: int
    history: np.ndarray
    features: np.ndarray
    preactivations: tuple
    hidden: tuple
    logits: np.ndarray
    probabilities: np.ndarray
    coding_probabilities: np.ndarray
    cdf: tuple
    actual_byte: int | None

    @property
    def next_byte_bits(self):
        return None if self.actual_byte is None else -math.log2(float(self.coding_probabilities[self.actual_byte]))

    @property
    def entropy_bits(self):
        return float(-np.sum(self.coding_probabilities * np.log2(self.coding_probabilities)))


def inspect_position(model: NeuralModel, message: bytes, position: int) -> ActivationTrace:
    """Predict byte at position from preceding bytes only, including BOS padding.

    Position len(message) predicts the byte after the message. UTF-8 is deliberately
    inspected byte by byte, including positions inside multibyte characters.
    """
    if not isinstance(position, int) or not 0 <= position <= len(message):
        raise ValueError('Position must be between zero and the message byte length')
    history = np.full(model.context, 256, dtype=np.int64)
    preceding = np.frombuffer(message[max(0, position - model.context):position], dtype=np.uint8)
    if len(preceding):
        history[-len(preceding):] = preceding
    features = model.arrays[0][history].reshape(-1)
    x, preactivations, hidden = features, [], []
    for index in range(1, len(model.arrays) - 2, 2):
        z = x @ model.arrays[index] + model.arrays[index + 1]
        x = np.tanh(z)
        preactivations.append(z)
        hidden.append(x)
    logits = x @ model.arrays[-2] + model.arrays[-1]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    cdf = probabilities_to_cdf(probabilities)
    coding_probabilities = np.diff(cdf).astype(np.float64) / TOTAL
    return ActivationTrace(position, history, features, tuple(preactivations), tuple(hidden), logits,
                           probabilities, coding_probabilities, tuple(cdf), message[position] if position < len(message) else None)


@dataclass(frozen=True)
class PrefixInspection:
    trace: ActivationTrace
    encoded_bytes: int
    total_bits: float
    payload_bytes: int


class MessageInspector:
    """Cache coding totals for one model/message; call from a single worker.

    Totals include the selected byte, when present. At the end-of-input position,
    they describe the complete message without encoding an extra byte.
    Only one activation trace is retained, along with scalar prefix totals.
    """

    def __init__(self, model: NeuralModel, message: bytes):
        self.model, self.message = model, bytes(message)
        self._encoder = ArithmeticEncoder()
        self._bits = [0.0]
        self._sizes = [self._encoder.finished_size()]
        self._last_trace = None

    def inspect(self, position: int, *, cancel=None) -> PrefixInspection:
        if not isinstance(position, int) or not 0 <= position <= len(self.message):
            raise ValueError('Position must be between zero and the message byte length')
        count = min(position + 1, len(self.message))
        while len(self._bits) <= count:
            if cancel is not None and cancel.is_set():
                raise CancelledError()
            index = len(self._bits) - 1
            trace = inspect_position(self.model, self.message, index)
            self._encoder.write(trace.cdf, self.message[index])
            self._bits.append(self._bits[-1] + trace.next_byte_bits)
            self._sizes.append(self._encoder.finished_size())
            self._last_trace = trace
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        if self._last_trace is None or self._last_trace.position != position:
            self._last_trace = inspect_position(self.model, self.message, position)
        return PrefixInspection(self._last_trace, count, self._bits[count], self._sizes[count])


@dataclass(frozen=True)
class ActivationGroup:
    start: int
    stop: int  # exclusive
    peak_index: int
    value: float
    minimum: float
    maximum: float
    mean: float


def group_activations(values, limit=256):
    """Consecutive display groups; preserve the sign of the strongest activation.

    These are display summaries only. Inference always evaluates every neuron.
    """
    if limit < 1:
        raise ValueError('Tile limit must be positive')
    values = np.asarray(values).reshape(-1)
    boundaries = np.linspace(0, len(values), min(limit, len(values)) + 1, dtype=int)
    groups = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        section = values[start:stop]
        peak = int(start + np.argmax(np.abs(section)))
        groups.append(ActivationGroup(int(start), int(stop), peak, float(values[peak]),
                                      float(section.min()), float(section.max()), float(section.mean())))
    return groups


def byte_label(value):
    if value is None:
        return 'End of input'
    if value == 256:
        return 'BOS'
    special = {0: 'NUL', 9: 'TAB', 10: 'LF', 13: 'CR', 32: 'SPACE', 127: 'DEL'}
    label = special.get(value, repr(chr(value)) if 33 <= value <= 126 else '')
    return f'{value:02X}' + (f' {label}' if label else '')


def activation_color(value, scale=1.0, *, probability=False):
    """Blue=negative, amber=positive; green=relative output probability."""
    strength = min(1.0, abs(float(value)) / max(float(scale), 1e-12))
    base = (29, 41, 57)
    target = (72, 221, 166) if probability else ((255, 176, 92) if value >= 0 else (73, 168, 255))
    return '#' + ''.join(f'{round(a + (b - a) * strength):02x}' for a, b in zip(base, target))
