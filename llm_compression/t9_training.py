"""Train compact, key-conditioned T9 networks for best-guess word accuracy."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from .corpus import read_messages
from .nn_cuda import Float32Sum
from .nn_training import resolve_device
from .t9 import BYTE_KEYS, CLASS_KEYS, evaluate, normalize_text
from .t9_model import T9Model, export_model, layout
from .training_corpus import load_prepared, reservoir, take_bytes


def normalized_bytes(text):
    return normalize_text(text).encode('utf-8')


def _digest(data):
    return hashlib.sha256(data).digest()


def _segments(data, size):
    """Split on UTF-8 boundaries, preferring whitespace near the size limit."""
    while data:
        end = min(size, len(data))
        while end < len(data) and end > 0 and data[end] & 0xC0 == 0x80:
            end -= 1
        if end == 0:
            raise ValueError('Text segment size is smaller than a UTF-8 character')
        if end < len(data):
            space = data.rfind(b' ', end // 2, end)
            if space >= 0:
                end = space + 1
        yield data[:end]
        data = data[end:]


class NormalizedMessages:
    """Re-iterable prepared messages, optionally excluding normalized holdouts."""
    def __init__(self, source, forbidden=()):
        self.source, self.forbidden = source, set(forbidden)

    def __iter__(self):
        for message in self.source:
            data = normalized_bytes(message.decode('utf-8'))
            if any(97 <= byte <= 122 for byte in data) and _digest(data) not in self.forbidden:
                yield data


def load_corpus(args):
    if args.input.is_dir() or args.input.name == 'manifest.json':
        if args.limit_messages is not None or args.keep_duplicates:
            raise ValueError('Prepared splits are fixed; use --train-bytes for a development limit')
        sources, report = load_prepared(args.input, max_message_bytes=args.max_message_bytes)
        validation = list(reservoir(NormalizedMessages(sources['validation']),
                                    args.evaluation_messages, args.seed))
        validation_hashes = {_digest(data) for data in validation}
        test = list(reservoir(NormalizedMessages(sources['test'], validation_hashes),
                              args.evaluation_messages, args.seed + 1))
        forbidden = validation_hashes | {_digest(data) for data in test}
        train = NormalizedMessages(sources['train'], forbidden)
        report = {**report, 'normalization': 'ASCII lowercase only',
                  'normalized_overlap_policy': 'selected validation/test messages excluded from training; test excludes validation'}
        return {'train': train, 'validation': validation, 'test': test}, report

    def texts():
        if args.input.suffix.lower() in ('.jsonl', '.csv'):
            yield from read_messages(args.input, args.text_field)
        else:
            with args.input.open(encoding='utf-8', newline='') as stream:
                yield from stream

    groups = Counter()
    input_rows = empty = oversized = 0
    plain_text = args.input.suffix.lower() not in ('.jsonl', '.csv')
    for index, text in enumerate(texts()):
        if args.limit_messages is not None and index >= args.limit_messages:
            break
        input_rows += 1
        data = normalized_bytes(text)
        if not any(97 <= byte <= 122 for byte in data):
            empty += 1
            continue
        if not plain_text and args.max_message_bytes and len(data) > args.max_message_bytes:
            oversized += 1
            continue
        pieces = _segments(data, args.max_message_bytes or 8192) if plain_text else (data,)
        for piece in pieces:
            if any(97 <= byte <= 122 for byte in piece):
                groups[piece] += 1
    segmented = False
    if 0 < len(groups) < 3:
        target = max(4, min(args.max_message_bytes or 8192, sum(map(len, groups)) // 10))
        originals = groups
        while True:
            smaller = Counter()
            for data, occurrences in originals.items():
                for piece in _segments(data, target):
                    if any(97 <= byte <= 122 for byte in piece):
                        smaller[piece] += occurrences
            if len(smaller) >= 3 or target <= 4:
                break
            target = max(4, target // 2)
        groups, segmented = smaller, True
    unique = list(groups)
    if len(unique) < 3:
        raise ValueError('Corpus needs at least three distinct messages/text segments containing ASCII letters')
    if args.split == 'random':
        random.Random(args.seed).shuffle(unique)
    nval = max(1, int(len(unique) * args.validation_fraction))
    ntest = max(1, int(len(unique) * args.test_fraction))
    ntrain = len(unique) - nval - ntest
    if ntrain < 1:
        raise ValueError('No training messages remain after splitting')
    partitions = {'train': unique[:ntrain], 'validation': unique[ntrain:ntrain + nval],
                  'test': unique[ntrain + nval:]}
    if args.keep_duplicates:
        partitions = {name: [data for data in records for _ in range(groups[data])]
                      for name, records in partitions.items()}
    report = {'path': str(args.input.resolve()), 'input_rows': input_rows, 'unique_messages': len(groups),
              'no_letters_skipped': empty, 'oversized_skipped': oversized,
              'duplicates': sum(groups.values()) - len(groups), 'split_strategy': args.split,
              'normalization': 'ASCII lowercase before grouping and splitting',
              'automatic_segmentation': segmented,
              'splits': {name: {'messages': len(records), 'utf8_bytes': sum(map(len, records))}
                         for name, records in partitions.items()}}
    for index, name in enumerate(('validation', 'test')):
        partitions[name] = list(reservoir(partitions[name], args.evaluation_messages, args.seed + index))
    return partitions, report


def training_batches(messages, context, key_context, batch_size, rng=None):
    """Bounded batches of true text history, remaining word keys, target letter.

    Only ASCII letters are prediction targets. Literal bytes still enter text
    history. Future key features stop at punctuation, spaces, and record ends.
    """
    if context < 0 or key_context < 1 or batch_size < 1:
        raise ValueError('Invalid training context or batch size')
    histories, windows, targets, count = [], [], [], 0

    def packed():
        x, k, y = np.concatenate(histories), np.concatenate(windows), np.concatenate(targets)
        order = rng.permutation(len(y)) if rng is not None else np.arange(len(y))
        return x[order], k[order], y[order]

    for message in messages:
        # Normalization is idempotent, including for callers outside the trainer.
        data = np.frombuffer(normalized_bytes(message.decode('utf-8')), dtype=np.uint8)
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            selected = np.flatnonzero(BYTE_KEYS[data[start:end]])
            if not len(selected):
                continue
            preceding = data[max(0, start - context):end].astype(np.int64)
            preceding = np.pad(preceding, (max(0, context - start), 0), constant_values=256)
            x = np.lib.stride_tricks.sliding_window_view(preceding, context + 1)[selected, :context].copy()
            following = BYTE_KEYS[data[start:min(len(data), end + key_context - 1)]]
            following = np.pad(following, (0, max(0, end + key_context - 1 - len(data))))
            k = np.lib.stride_tricks.sliding_window_view(following, key_context)[selected].copy()
            k *= np.cumprod(k != 0, axis=1)
            y = data[start:end][selected].astype(np.int64) - ord('a')
            offset = 0
            while offset < len(y):
                take = min(batch_size - count, len(y) - offset)
                histories.append(x[offset:offset + take])
                windows.append(k[offset:offset + take])
                targets.append(y[offset:offset + take])
                count += take
                offset += take
                if count == batch_size:
                    yield packed()
                    histories, windows, targets, count = [], [], [], 0
    if count:
        yield packed()


def make_network(torch, context, key_context, embedding, hidden, hidden_layers=3):
    if context < 0 or key_context < 1 or embedding < 1 or hidden < 1 or hidden_layers not in (1, 2, 3):
        raise ValueError('Invalid T9 architecture')

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_embedding = torch.nn.Embedding(257, embedding)
            self.key_embedding = torch.nn.Embedding(10, embedding)
            widths = [(context + key_context) * embedding] + [hidden] * hidden_layers + [26]
            self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
            self.register_buffer('class_keys', torch.tensor(CLASS_KEYS.copy()))

        def forward(self, history, keys):
            text = self.text_embedding(history).reshape(len(history), context * embedding)
            keypad = self.key_embedding(keys).reshape(len(keys), key_context * embedding)
            features = torch.cat((text, keypad), dim=1)
            for layer in self.layers[:-1]:
                features = torch.tanh(layer(features))
            logits = self.layers[-1](features)
            return logits.masked_fill(self.class_keys[None, :] != keys[:, :1], -torch.inf)

        def arrays(self):
            arrays = [self.text_embedding.weight.detach().cpu().numpy(),
                      self.key_embedding.weight.detach().cpu().numpy()]
            for layer in self.layers:
                arrays.extend((layer.weight.detach().cpu().numpy().T, layer.bias.detach().cpu().numpy()))
            return arrays

    return Network()


def train(args):
    # Validate before creating output files.
    positive = (args.epochs, args.batch_size, args.threads, args.embedding_dim, args.key_context,
                args.validate_every, args.beam_width, args.learning_rate)
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError('Epochs, batch size, threads, embedding, key context, validation interval, beam and learning rate must be positive')
    if min(args.context_chars, args.patience, args.evaluation_messages, args.max_message_bytes) < 0 or args.seed < 0:
        raise ValueError('Context, patience, evaluation/message limits and seed must be nonnegative')
    if any(value is not None and value <= 0 for value in (args.limit_messages, args.train_bytes)):
        raise ValueError('Input limits must be positive')
    if not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1
            and args.validation_fraction + args.test_fraction < 1):
        raise ValueError('Validation/test fractions must be positive and sum to less than one')
    hidden, planned_size = layout(args.model_kib, args.context_chars, args.key_context,
                                  args.embedding_dim, args.storage, args.hidden_layers)
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    device = resolve_device(torch, args.device)
    output = args.output or Path(f'models/t9-{args.model_kib}k-context{args.context_chars}-{args.storage}')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Output is not an empty directory: {output}')
    partitions, corpus = load_corpus(args)
    if not partitions['validation'] or not partitions['test']:
        raise ValueError('Validation and test must contain ASCII letters after normalization/overlap filtering')
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    network = make_network(torch, args.context_chars, args.key_context, args.embedding_dim,
                           hidden, args.hidden_layers).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    model_path, candidate_path = output / 'model.t9nn', output / 'candidate.t9nn'
    epochs, best_score, best_epoch, bad_checks = [], None, None, 0
    started = time.perf_counter()
    print(f'T9 model: {planned_size:,} bytes, text context {args.context_chars} UTF-8 bytes, '
          f'{args.key_context} word keys, {args.hidden_layers} layers of width {hidden}, '
          f'{args.storage}; training on {device}', flush=True)
    for epoch in range(1, args.epochs + 1):
        network.train()
        source = take_bytes(partitions['train'], args.train_bytes) if args.train_bytes else partitions['train']
        losses = Float32Sum(torch, device)
        letters = 0
        epoch_started = last_progress = time.perf_counter()
        for history, keys, target in training_batches(source, args.context_chars, args.key_context,
                                                       args.batch_size, np.random.default_rng(args.seed + epoch)):
            history, keys, target = (torch.from_numpy(array).to(device) for array in (history, keys, target))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(network(history, keys), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            losses.add(loss.detach() * len(target))
            letters += len(target)
            if time.perf_counter() - last_progress >= 10:
                print(f'Epoch {epoch}: {letters:,} letters trained', flush=True)
                last_progress = time.perf_counter()
        if not letters:
            raise ValueError('No training letters fit the configured limits')
        loss_value = losses.item() / letters
        if not math.isfinite(loss_value):
            raise ValueError('Nonfinite training loss; lower the learning rate')
        row = {'epoch': epoch, 'training_letters': letters, 'training_nats_per_letter': loss_value,
               'training_seconds': time.perf_counter() - epoch_started, 'validation': None}
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            export_model(candidate_path, network.arrays(), args.context_chars, args.key_context,
                         args.embedding_dim, hidden, args.storage, args.hidden_layers)
            deployed = T9Model(candidate_path)
            if deployed.size_bytes != planned_size or deployed.size_bytes > args.model_kib * 1024:
                raise ValueError('Exported model exceeds storage budget')
            print(f'Epoch {epoch}: decoding held-out validation keystrokes (beam {args.beam_width})', flush=True)
            validation_started = time.perf_counter()
            row['validation'] = evaluate(deployed, partitions['validation'], beam_width=args.beam_width)
            row['validation_seconds'] = time.perf_counter() - validation_started
            secondary = 'letter_accuracy' if args.selection_metric == 'word_accuracy' else 'word_accuracy'
            score = (row['validation'][args.selection_metric], row['validation'][secondary])
            if best_score is None or score > best_score:
                best_score, best_epoch, bad_checks = score, epoch, 0
                model_path.write_bytes(candidate_path.read_bytes())
            else:
                bad_checks += 1
            print(f'Epoch {epoch}: loss {loss_value:.4f}; '
                  f'word accuracy {row["validation"]["word_accuracy"]:.2%}; '
                  f'letter accuracy {row["validation"]["letter_accuracy"]:.2%}', flush=True)
        else:
            print(f'Epoch {epoch}: loss {loss_value:.4f}; validation skipped', flush=True)
        epochs.append(row)
        (output / 'epochs.json').write_text(json.dumps(epochs, indent=2) + '\n', encoding='utf-8')
        if args.patience and bad_checks >= args.patience:
            print(f'Early stopping; retaining epoch {best_epoch}', flush=True)
            break
    candidate_path.unlink()
    deployed = T9Model(model_path)
    print(f'Decoding held-out test keystrokes with best epoch {best_epoch}', flush=True)
    test = evaluate(deployed, partitions['test'], beam_width=args.beam_width)
    report = {
        'architecture': 'text-history and remaining-word-key embeddings, tanh MLP, key-masked 26-letter softmax',
        'normalization': 'ASCII lowercase only; other Unicode/punctuation/digits are literal',
        'requested_model_kib': args.model_kib,
        'model': {'path': str(model_path), 'model_bytes': deployed.size_bytes, 'sha256': deployed.sha256,
                  'parameter_count': sum(parameter.numel() for parameter in network.parameters()),
                  'storage': args.storage, 'context_utf8_bytes': args.context_chars,
                  'key_context': args.key_context, 'embedding_dim': args.embedding_dim,
                  'hidden_dim': hidden, 'hidden_layers': args.hidden_layers},
        'corpus': corpus, 'selection_metric': args.selection_metric, 'best_epoch': best_epoch,
        'validation': epochs[best_epoch - 1]['validation'], 'test': test, 'epochs': epochs,
        'completed_epochs': len(epochs), 'stopped_early': len(epochs) < args.epochs,
        'evaluated_messages': {'validation': len(partitions['validation']), 'test': len(partitions['test'])},
        'beam_width': args.beam_width, 'seed': args.seed,
        'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'training_device': device,
                    'training_precision': 'float32', 'inference': 'NumPy float32; exported/dequantized weights'},
        'training_seconds': time.perf_counter() - started,
        'notes': ['Best checkpoint chosen by free-running validation accuracy of the exported model; test used only after selection.',
                  'Training uses correct preceding letters; validation/test feed back predicted letters without reference corrections.',
                  'Each word is committed before decoding the next; beam search is approximate and dictionary-free.',
                  'File size includes all metadata, weights, biases and int8 scales, excluding optimizer/training state.',
                  'Int8 storage is dequantized to float32 in RAM for inference; this is not an integer MCU runtime.',
                  'Word accuracy measures contiguous ASCII a-z runs; letter accuracy excludes supplied literals.',
                  'Text context resets at corpus record boundaries; decoding a text file uses continuous context.',
                  'Future keys are limited to the current completed word; this is not next-word completion.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'model_bytes': deployed.size_bytes, 'best_epoch': best_epoch,
                      'test_word_accuracy': test['word_accuracy'], 'test_letter_accuracy': test['letter_accuracy']}, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='UTF-8 text, JSONL/CSV, or prepared Wikipedia directory/manifest.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-kib', type=int, default=1024, help='complete exported file budget, KiB (default 1024)')
    parser.add_argument('--context-chars', '--context-bytes', dest='context_chars', type=int, default=32,
                        help='preceding predicted UTF-8 bytes, including spaces (default 32)')
    parser.add_argument('--key-context', type=int, default=16,
                        help='current and upcoming keys within the current word (default 16)')
    parser.add_argument('--embedding-dim', type=int, default=8)
    parser.add_argument('--hidden-layers', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--storage', choices=('int8', 'float32'), default='int8')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--patience', '--early-stopping-patience', type=int, default=3,
                        help='validation checks without accuracy improvement; 0 disables (default 3)')
    parser.add_argument('--validate-every', type=int, default=1)
    parser.add_argument('--selection-metric', choices=('word_accuracy', 'letter_accuracy'), default='word_accuracy')
    parser.add_argument('--beam-width', type=int, default=8, help='same word beam for validation/test and prediction')
    parser.add_argument('--evaluation-messages', type=int, default=128,
                        help='seeded sample per held-out split; 0 uses all (default 128)')
    parser.add_argument('--text-field', choices=('body', 'text'), default='body')
    parser.add_argument('--split', choices=('chronological', 'random'), default='chronological')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--keep-duplicates', action='store_true', help='retain occurrences within one split')
    parser.add_argument('--limit-messages', type=int)
    parser.add_argument('--max-message-bytes', type=int, default=8192, help='0 removes the input record limit')
    parser.add_argument('--train-bytes', type=int, help='whole-message training byte limit per epoch')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train(args)
    except KeyboardInterrupt:
        return 130
    except ImportError as exc:
        parser.exit(2, f'Missing dependency: {exc}. Training needs torch and numpy.\n')
    except (ValueError, OSError, RuntimeError, MemoryError, OverflowError) as exc:
        parser.exit(2, f'T9 training failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
