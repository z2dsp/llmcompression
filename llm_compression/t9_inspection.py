"""T9 simulation and activation inspection using the deployed NumPy decoder."""
from concurrent.futures import CancelledError
from dataclasses import dataclass

import numpy as np

from .t9 import CLASS_KEYS, decode_keys, parse_keys


@dataclass(frozen=True)
class WordPrediction:
    start: int
    stop: int
    candidates: tuple  # (spelling, summed conditional log probability)


@dataclass(frozen=True)
class Simulation:
    keystrokes: str
    tokens: tuple
    prediction: str
    words: tuple

    def word_at(self, position):
        return next((word for word in self.words if word.start <= position < word.stop), None)


@dataclass(frozen=True)
class T9Trace:
    position: int
    letter: str
    history: np.ndarray
    keys: np.ndarray
    text_features: np.ndarray
    key_features: np.ndarray
    preactivations: tuple
    hidden: tuple
    logits: np.ndarray
    probabilities: np.ndarray


def simulate(model, keystrokes, *, beam_width=8, cancel=None):
    words = []

    def record(start, stop, beams):
        words.append(WordPrediction(start, stop, tuple((word, score) for score, word, _ in beams)))

    prediction = decode_keys(model, keystrokes, beam_width=beam_width, cancel=cancel, on_word=record)
    return Simulation(keystrokes, tuple(parse_keys(keystrokes)), prediction, tuple(words))


def inspect_position(model, simulation, position, *, cancel=None):
    """Trace a chosen spelling's letter; literal/empty positions return None.

    History contains predicted bytes, never reference text. Lookahead contains
    only remaining keys in the same word, exactly as in decode_keys.
    """
    if type(position) is not int or not 0 <= position <= len(simulation.tokens):
        raise ValueError('Position must be between zero and the parsed token count')
    if cancel is not None and cancel.is_set():
        raise CancelledError()
    word = simulation.word_at(position)
    if word is None:
        return None
    prefix = simulation.prediction[:position].encode('utf-8')
    history = np.full(model.context, 256, dtype=np.int64)
    preceding = np.frombuffer(prefix, dtype=np.uint8)[-model.context:] if model.context else []
    if len(preceding):
        history[-len(preceding):] = preceding
    keys = np.zeros(model.key_context, dtype=np.int64)
    following = simulation.tokens[position:min(word.stop, position + model.key_context)]
    keys[:len(following)] = following
    text_features = model.arrays[0][history].reshape(-1)
    key_features = model.arrays[1][keys].reshape(-1)
    features = np.concatenate((text_features, key_features))
    preactivations, hidden = [], []
    for index in range(2, len(model.arrays) - 2, 2):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        z = features @ model.arrays[index] + model.arrays[index + 1]
        features = np.tanh(z)
        preactivations.append(z)
        hidden.append(features)
    logits = features @ model.arrays[-2] + model.arrays[-1]
    allowed = CLASS_KEYS == keys[0]
    scores = logits[allowed]
    if not np.isfinite(scores).all():
        raise ValueError('Invalid T9 model scores')
    weights = np.exp(scores - scores.max())
    probabilities = np.zeros(26, dtype=np.float32)
    probabilities[allowed] = weights / weights.sum()
    return T9Trace(position, simulation.prediction[position], history, keys, text_features,
                   key_features, tuple(preactivations), tuple(hidden), logits, probabilities)
