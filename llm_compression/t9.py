"""T9 notation, constrained word decoding, and free-running accuracy metrics.

ASCII letters share keys 2..9; 0 is space. Other characters are literal,
except original digits and backslashes, which are backslash-escaped.
"""
import argparse
from concurrent.futures import CancelledError
import json
from pathlib import Path
import re
import sys

import numpy as np

KEY_LETTERS = {2: 'abc', 3: 'def', 4: 'ghi', 5: 'jkl',
               6: 'mno', 7: 'pqrs', 8: 'tuv', 9: 'wxyz'}
LETTER_KEY = {letter: key for key, letters in KEY_LETTERS.items() for letter in letters}
LOWERCASE = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')
BYTE_KEYS = np.zeros(256, dtype=np.int64)
for _letter, _key in LETTER_KEY.items():
    BYTE_KEYS[ord(_letter)] = _key
CLASS_KEYS = BYTE_KEYS[ord('a'):ord('z') + 1]
WORD = re.compile('[a-z]+')


def normalize_text(text):
    """Fold ASCII case only; preserve whitespace, accents, and other Unicode."""
    return text.translate(LOWERCASE)


def encode_text(text):
    result = []
    for character in normalize_text(text):
        if character in LETTER_KEY:
            result.append(str(LETTER_KEY[character]))
        elif character == ' ':
            result.append('0')
        elif character in '0123456789\\':
            result.append('\\' + character)
        else:
            result.append(character)
    return ''.join(result)


def parse_keys(keystrokes):
    """Return integer ambiguous keys and string literal characters."""
    tokens = []
    position = 0
    while position < len(keystrokes):
        character = keystrokes[position]
        if character == '\\':
            position += 1
            if position == len(keystrokes) or keystrokes[position] not in '0123456789\\':
                raise ValueError('A backslash must escape an original digit or backslash')
            tokens.append(keystrokes[position])
        elif character in '23456789':
            tokens.append(int(character))
        elif character == '0':
            tokens.append(' ')
        elif character == '1' or character == ' ' or character in LETTER_KEY or character in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            raise ValueError(f'Invalid keystroke {character!r} at offset {position}; use text_to_t9.py first')
        else:
            tokens.append(character)
        position += 1
    return tokens


def _advance(history, text, context):
    if not context:
        return ()
    return (history + tuple(text.encode('utf-8')))[-context:]


def decode_keys(model, keystrokes, *, beam_width=8, cancel=None, on_word=None):
    """Choose one spelling per completed word, feeding predictions into context.

    Only the current word's remaining keys are visible. Beam scores are sums
    of masked conditional log probabilities. Beam width 1 is greedy; wider
    beams approximate the most probable spelling, without a vocabulary list.

    Optional cancel Event and on_word(start, stop, beams) callback support
    interactive inspection. Beams contain (log probability, spelling, history)
    in descending score order; start/stop index parsed tokens, not raw notation.
    """
    if type(beam_width) is not int or beam_width < 1:
        raise ValueError('Beam width must be a positive integer')
    tokens = parse_keys(keystrokes)
    history = (256,) * model.context
    output = []
    position = 0
    while position < len(tokens):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        token = tokens[position]
        if isinstance(token, str):
            output.append(token)
            history = _advance(history, token, model.context)
            position += 1
            continue
        end = position
        while end < len(tokens) and isinstance(tokens[end], int):
            end += 1
        word_keys = tokens[position:end]
        beams = [(0.0, '', history)]
        for offset, key in enumerate(word_keys):
            if cancel is not None and cancel.is_set():
                raise CancelledError()
            window = word_keys[offset:offset + model.key_context]
            window += [0] * (model.key_context - len(window))
            histories = np.asarray([item[2] for item in beams], dtype=np.int64)
            windows = np.tile(window, (len(beams), 1))
            logits = np.asarray(model.logits(histories, windows))
            candidates = KEY_LETTERS[key]
            indices = [ord(letter) - ord('a') for letter in candidates]
            scores = logits[:, indices]
            if scores.shape != (len(beams), len(indices)) or not np.isfinite(scores).all():
                raise ValueError('Invalid T9 model scores')
            shifted = scores - scores.max(axis=1, keepdims=True)
            log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            expanded = []
            for row, (score, prefix, previous) in enumerate(beams):
                for column, letter in enumerate(candidates):
                    expanded.append((score + float(log_probabilities[row, column]), prefix + letter,
                                     _advance(previous, letter, model.context)))
            expanded.sort(key=lambda item: (-item[0], item[1]))
            beams = expanded[:beam_width]
        _, word, history = beams[0]
        if on_word is not None:
            on_word(position, end, tuple(beams))
        output.append(word)
        position = end
    return ''.join(output)


class Accuracy:
    def __init__(self):
        self.counts = dict(messages=0, exact_messages=0, characters=0, correct_characters=0,
                           letters=0, correct_letters=0, words=0, correct_words=0)

    def add(self, reference, prediction):
        reference = normalize_text(reference)
        if len(reference) != len(prediction) or encode_text(reference) != encode_text(prediction):
            raise ValueError('Prediction and reference do not represent the same T9 keystrokes')
        counts = self.counts
        counts['messages'] += 1
        counts['exact_messages'] += reference == prediction
        counts['characters'] += len(reference)
        counts['correct_characters'] += sum(a == b for a, b in zip(reference, prediction))
        for index, letter in enumerate(reference):
            if letter in LETTER_KEY:
                counts['letters'] += 1
                counts['correct_letters'] += letter == prediction[index]
        for match in WORD.finditer(reference):
            counts['words'] += 1
            counts['correct_words'] += match.group() == prediction[match.start():match.end()]

    def report(self):
        counts = self.counts
        rates = {name + '_accuracy': counts[numerator] / counts[denominator] if counts[denominator] else None
                 for name, numerator, denominator in (
                     ('character', 'correct_characters', 'characters'),
                     ('letter', 'correct_letters', 'letters'),
                     ('word', 'correct_words', 'words'),
                     ('message', 'exact_messages', 'messages'))}
        return {**counts, **rates}


def evaluate(model, messages, *, beam_width=8):
    accuracy = Accuracy()
    for message in messages:
        text = message.decode('utf-8') if isinstance(message, bytes) else message
        # Reference text is used only after decoding.
        prediction = decode_keys(model, encode_text(text), beam_width=beam_width)
        accuracy.add(text, prediction)
    result = accuracy.report()
    if not result['letters']:
        raise ValueError('Evaluation split contains no ASCII letters')
    return result


def _read(path):
    return Path(path).read_bytes().decode('utf-8')


def _write_new(path, text):
    with Path(path).open('xb') as stream:
        stream.write(text.encode('utf-8'))


def encode_main(argv=None):
    parser = argparse.ArgumentParser(description='Convert UTF-8 text to T9: 2-9 letters, 0 space, punctuation literal.')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    try:
        _write_new(args.output, encode_text(_read(args.input)))
    except (OSError, ValueError) as exc:
        parser.exit(2, f'T9 conversion failed: {exc}\n')
    return 0


def predict_main(argv=None):
    parser = argparse.ArgumentParser(description='Decode a T9 file with a trained model and optionally measure accuracy.')
    parser.add_argument('model', type=Path)
    parser.add_argument('input', type=Path, help='keystrokes produced by text_to_t9.py')
    parser.add_argument('output', type=Path, help='new UTF-8 prediction file')
    parser.add_argument('--reference', type=Path, help='original UTF-8 text; used only to score the completed prediction')
    parser.add_argument('--report', type=Path, help='write accuracy JSON to a new file (requires --reference)')
    parser.add_argument('--beam-width', type=int, default=8, help='hypotheses per word; 1 is greedy (default: 8)')
    args = parser.parse_args(argv)
    if args.report and not args.reference:
        parser.error('--report requires --reference')
    try:
        from .t9_model import T9Model
        if args.output.exists() or args.report and args.report.exists():
            raise ValueError('Output/report already exists; choose new file names')
        if args.report and args.report.resolve() == args.output.resolve():
            raise ValueError('Prediction and report paths must differ')
        keys = _read(args.input)
        reference = _read(args.reference) if args.reference else None
        if reference is not None and parse_keys(encode_text(reference)) != parse_keys(keys):
            raise ValueError('Reference text does not match the input keystrokes')
        model = T9Model(args.model)
        prediction = decode_keys(model, keys, beam_width=args.beam_width)
        report = None
        if reference is not None:
            accuracy = Accuracy()
            accuracy.add(reference, prediction)
            report = {**accuracy.report(), 'model_bytes': model.size_bytes, 'model_sha256': model.sha256,
                      'beam_width': args.beam_width, 'normalization': 'ASCII lowercase only',
                      'decoding': 'free-running; word beam search; no reference feedback'}
        _write_new(args.output, prediction)
        if report is not None:
            rendered = json.dumps(report, indent=2) + '\n'
            if args.report:
                _write_new(args.report, rendered)
            sys.stdout.write(rendered)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as exc:
        parser.exit(2, f'T9 prediction failed: {exc}\n')
    return 0
