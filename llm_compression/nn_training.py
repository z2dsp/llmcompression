"""Train bounded-storage neural byte predictors on MeshCore or prepared Wikipedia."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from .bit_training import file_sha256, split_corpus
from .training_corpus import load_prepared, reservoir, take_bytes
from .nn_model import NeuralModel, TOTAL, advance, compress_payload, decompress_payload, export_model, layout

VALIDATION_DEFAULTS = {'validate_every': 1, 'validation_device': 'cpu', 'validation_batch_size': 4096,
                       'skip_cpu_validation': False}
LEGACY_EARLY_STOPPING_DEFAULTS = {'early_stopping_patience': 0, 'early_stopping_min_delta': 0.0}


class EarlyStopping:
    """Stop after consecutive validation checks without a significant improvement.

    The significant-improvement anchor is separate from checkpoint selection:
    even a smaller improvement than min_delta can still be the best model.
    """

    def __init__(self, patience=3, min_delta=0.0):
        if type(patience) is not int or patience < 0 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('Early stopping patience must be nonnegative; minimum improvement must be finite and nonnegative')
        self.patience, self.min_delta = patience, min_delta
        self.best_loss = None
        self.bad_checks = self.checks = 0

    def update(self, loss):
        if not math.isfinite(loss):
            raise ValueError('Nonfinite validation loss')
        self.checks += 1
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return self.patience > 0 and self.bad_checks >= self.patience


def byte_batches(messages, batch_size):
    """Pack bounded byte batches with each byte's position in its own message."""
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    pieces, lengths, bases = [], [], []
    count = 0
    for message in messages:
        offset = 0
        while offset < len(message):
            take = min(batch_size - count, len(message) - offset)
            pieces.append(message[offset:offset + take])
            lengths.append(take)
            bases.append(offset - count)
            count += take
            offset += take
            if count == batch_size:
                yield (np.frombuffer(b''.join(pieces), dtype=np.uint8),
                       np.arange(count, dtype=np.int64) + np.repeat(bases, lengths))
                pieces, lengths, bases = [], [], []
                count = 0
    if count:
        yield (np.frombuffer(b''.join(pieces), dtype=np.uint8),
               np.arange(count, dtype=np.int64) + np.repeat(bases, lengths))


def training_batches(messages, context, batch_size, rng=None):
    """Vectorized contexts with the same examples, batch boundaries, and shuffle.

    Memory scales with one batch and its context, even for whole-article input.
    No context crosses a message or prepared-record boundary.
    """
    if context < 0 or batch_size < 1:
        raise ValueError('Context must be nonnegative and batch size positive')
    history = np.full(context, 256, dtype=np.int64)
    required_history = context - np.arange(context, dtype=np.int64)
    for data, positions in byte_batches(messages, batch_size):
        padded = np.empty(context + len(data), dtype=np.int64)
        padded[:context] = history
        padded[context:] = data
        x = np.lib.stride_tricks.sliding_window_view(padded, context + 1)[:, :context].copy()
        # Mask bytes from earlier messages, including boundaries within a batch.
        x[positions[:, None] < required_history] = 256
        history = padded[-context:].copy() if context else padded[:0]
        yield _batch(x, data, context, rng)


def _batch(x, y, context, rng):
    x = np.asarray(x, dtype=np.int64).reshape(len(y), context)
    y = np.asarray(y, dtype=np.int64)
    if rng is not None:
        order = rng.permutation(len(y))
        x, y = x[order], y[order]
    return x, y


def evaluate(messages, model):
    bits = byte_count = message_count = 0
    for message in messages:
        message_count += 1
        history = [256] * model.context
        for byte in message:
            cdf = model.cdf(history)
            bits -= math.log2((cdf[byte + 1] - cdf[byte]) / TOTAL)
            byte_count += 1
            advance(history, byte)
    if not byte_count:
        raise ValueError('Cannot evaluate an empty split')
    return {'messages': message_count, 'utf8_bytes': byte_count, 'predicted_bits': bits,
            'bits_per_utf8_byte': bits / byte_count}


def target_frequencies(torch, probabilities, targets):
    """The codec's integer frequency for each target, without sorting all bytes.

    Rank fractional remainders with ascending byte values breaking ties, exactly
    as the reference stable sort does. Given identical probabilities, these
    integer frequencies agree with NeuralModel.cdf().
    """
    scaled = probabilities * (TOTAL - 256)
    floors = torch.floor(scaled)
    fractions = scaled - floors
    remaining = TOTAL - 256 - floors.to(torch.int64).sum(dim=1)
    target = targets[:, None]
    selected_fraction = fractions.gather(1, target)
    classes = torch.arange(256, device=targets.device)
    ranks = ((fractions > selected_fraction)
             | ((fractions == selected_fraction) & (classes < target))).sum(dim=1)
    return floors.gather(1, target).squeeze(1).to(torch.int64) + 1 + (ranks < remaining).to(torch.int64)


def evaluate_cuda(messages, model, batch_size=4096):
    """Batched exported-weight scoring; float64 GPU rounding can differ from CPU.

    Used for checkpoint selection. Optional final validation and final test/coding
    use the CPU reference. No training weights or generation loss are scored.
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    resolve_device(torch, 'cuda')
    if batch_size < 1:
        raise ValueError('Validation batch size must be positive')
    message_count = byte_count = 0
    def counted_messages():
        nonlocal message_count
        for message in messages:
            message_count += 1
            yield message
    with torch.inference_mode():
        embedding, w1, b1, w2, b2 = [torch.from_numpy(array).to('cuda') for array in model.arrays]
        bits = torch.zeros((), device='cuda', dtype=torch.float64)
        for x, y in training_batches(counted_messages(), model.context, batch_size):
            history = torch.from_numpy(x).to('cuda', non_blocking=True)
            targets = torch.from_numpy(y).to('cuda', non_blocking=True)
            features = embedding[history].reshape(len(y), model.context * model.embedding)
            logits = torch.tanh(features @ w1 + b1) @ w2 + b2
            probabilities = torch.exp(logits - logits.amax(dim=1, keepdim=True))
            probabilities /= probabilities.sum(dim=1, keepdim=True)
            frequencies = target_frequencies(torch, probabilities, targets)
            bits.sub_(torch.log2(frequencies.to(torch.float64) / TOTAL).sum())
            byte_count += len(y)
        value = bits.item()
    if not byte_count:
        raise ValueError('Cannot evaluate an empty split')
    if not math.isfinite(value):
        raise ValueError('Nonfinite CUDA validation loss')
    return {'messages': message_count, 'utf8_bytes': byte_count, 'predicted_bits': value,
            'bits_per_utf8_byte': value / byte_count}


def make_network(torch, context, embedding, hidden):
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(257, embedding)
            self.hidden = torch.nn.Linear(context * embedding, hidden)
            self.output = torch.nn.Linear(hidden, 256)

        def forward(self, x):
            x = self.embedding(x).reshape(x.shape[0], context * embedding)
            return self.output(torch.tanh(self.hidden(x)))

        def arrays(self):
            return [self.embedding.weight.detach().cpu().numpy(),
                    self.hidden.weight.detach().cpu().numpy().T,
                    self.hidden.bias.detach().cpu().numpy(),
                    self.output.weight.detach().cpu().numpy().T,
                    self.output.bias.detach().cpu().numpy()]
    return Network()


def resolve_device(torch, requested):
    available = torch.cuda.is_available()
    if requested == 'auto':
        return 'cuda' if available else 'cpu'
    if requested == 'cuda' and not available:
        if torch.version.cuda is None:
            local_runtime = Path(__file__).resolve().parent.parent / '.cache' / 'cuda-runtime' / 'torch' / '__init__.py'
            if sys.platform == 'win32' and local_runtime.is_file():
                raise ValueError('This Python has CPU-only PyTorch. Run train_nn_cuda.py with the same arguments to use the project-local CUDA runtime.')
            raise ValueError(
                'This Python has CPU-only PyTorch. Install the CUDA build with: '
                f'"{sys.executable}" -m pip install --upgrade torch==2.5.1+cu124 '
                '--index-url https://download.pytorch.org/whl/cu124')
        raise ValueError(
            f'PyTorch includes CUDA {torch.version.cuda}, but no CUDA device is available. '
            'Check nvidia-smi and the NVIDIA driver using the same Python environment.')
    return requested


def train(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    hidden, planned_size = layout(args.model_kib, args.context_bytes, args.embedding_dim, args.storage)
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Epochs, batch size, threads, and learning rate must be positive')
    if args.evaluation_messages < 0 or args.benchmark_messages < 0 or args.max_message_bytes < 0:
        raise ValueError('Evaluation/benchmark/message limits must be nonnegative')
    if args.validate_every < 1 or args.validation_batch_size < 1:
        raise ValueError('Validation interval and batch size must be positive')
    if args.skip_cpu_validation and args.validation_device != 'cuda':
        raise ValueError('--skip-cpu-validation requires --validation-device cuda')
    stopping = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)
    if args.train_bytes is not None and args.train_bytes <= 0 or args.limit_messages is not None and args.limit_messages <= 0:
        raise ValueError('Training byte and input message limits must be positive')
    requested_device = args.device
    args.device = resolve_device(torch, requested_device)
    args.validation_device = resolve_device(torch, args.validation_device)
    uses_cuda = args.device == 'cuda' or args.validation_device == 'cuda'
    output = args.output or Path(f'models/nn-{args.model_kib}k-context{args.context_bytes}-{args.storage}')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output is not empty: {output}; choose a new directory')
    prepared = args.input.is_dir() or args.input.name == 'manifest.json'
    if prepared:
        if args.limit_messages is not None or args.keep_duplicates:
            raise ValueError('Prepared splits are fixed; use --train-bytes for a development limit')
        partitions, corpus = load_prepared(args.input, max_message_bytes=args.max_message_bytes)
        source = Path(corpus['prepared_manifest'])
    else:
        partitions, corpus = split_corpus(args.input, text_field=args.text_field,
            validation_fraction=args.validation_fraction, test_fraction=args.test_fraction,
            split=args.split, seed=args.seed, keep_duplicates=args.keep_duplicates,
            limit=args.limit_messages, max_message_bytes=args.max_message_bytes)
        source = args.input
    validation = reservoir(partitions['validation'], args.evaluation_messages, args.seed)
    test = reservoir(partitions['test'], args.evaluation_messages, args.seed + 1)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if uses_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        print(f'CUDA GPU: {torch.cuda.get_device_name(0)}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}', flush=True)
    network = make_network(torch, args.context_bytes, args.embedding_dim, hidden).to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    print(f'Neural model: {planned_size:,} bytes, context {args.context_bytes}, embedding {args.embedding_dim}, hidden {hidden}, {args.storage}; training on {args.device}', flush=True)
    started, epochs, best_loss, best_epoch = time.perf_counter(), [], float('inf'), None
    stopped_early = False
    model_path, candidate_path = output / 'model.mcnn', output / 'candidate.mcnn'
    for epoch in range(1, args.epochs + 1):
        network.train()
        source_messages = take_bytes(partitions['train'], args.train_bytes) if args.train_bytes else partitions['train']
        # Keep the loss accumulator on device. Reading loss.item() or testing
        # a CUDA boolean every batch forces the CPU to wait for the GPU.
        loss_total = torch.zeros((), device=args.device, dtype=torch.float64)
        byte_count = 0
        train_started = time.perf_counter()
        last_progress = time.perf_counter()
        rng = np.random.default_rng(args.seed + epoch)
        for x, y in training_batches(source_messages, args.context_bytes, args.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            logits = network(torch.from_numpy(x).to(args.device, non_blocking=True))
            loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y).to(args.device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            loss_total.add_(loss.detach().to(torch.float64) * len(y))
            byte_count += len(y)
            if time.perf_counter() - last_progress >= 10:
                if not math.isfinite(loss_total.item()):
                    raise ValueError('Nonfinite training loss; lower the learning rate')
                elapsed = time.perf_counter() - train_started
                print(f'Epoch {epoch}: {byte_count:,} bytes trained ({byte_count / elapsed:,.0f} bytes/s)', flush=True)
                last_progress = time.perf_counter()
        if not byte_count:
            raise ValueError('No training bytes fit the configured limit')
        loss_sum = loss_total.item()  # Synchronizes once at the epoch boundary.
        if not math.isfinite(loss_sum):
            raise ValueError('Nonfinite training loss; lower the learning rate')
        train_seconds = time.perf_counter() - train_started
        val = None
        should_stop = False
        export_seconds = validation_seconds = 0.0
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            export_started = time.perf_counter()
            export_model(candidate_path, network.arrays(), args.context_bytes, args.embedding_dim, hidden, args.storage)
            deployed = NeuralModel(candidate_path)
            if deployed.size_bytes != planned_size or deployed.size_bytes > args.model_kib * 1024:
                raise ValueError('Exported model exceeds configured storage budget')
            export_seconds = time.perf_counter() - export_started
            print(f'Epoch {epoch}: evaluating exported model on {args.validation_device}', flush=True)
            validation_started = time.perf_counter()
            val = (evaluate_cuda(validation, deployed, args.validation_batch_size)
                   if args.validation_device == 'cuda' else evaluate(validation, deployed))
            validation_seconds = time.perf_counter() - validation_started
            should_stop = stopping.update(val['bits_per_utf8_byte'])
            if val['bits_per_utf8_byte'] < best_loss:
                best_loss, best_epoch = val['bits_per_utf8_byte'], epoch
                model_path.write_bytes(candidate_path.read_bytes())
        epochs.append({'epoch': epoch, 'training_bytes': byte_count,
                       'training_seconds': train_seconds, 'training_bytes_per_second': byte_count / train_seconds,
                       'export_seconds': export_seconds, 'validation_seconds': validation_seconds,
                       'early_stopping_bad_checks': stopping.bad_checks if val is not None else None,
                       'training_bits_per_utf8_byte': loss_sum / byte_count / math.log(2), 'validation': val})
        val_text = f'exported validation {val["bits_per_utf8_byte"]:.4f}' if val else 'validation skipped'
        print(f'Epoch {epoch}: train {epochs[-1]["training_bits_per_utf8_byte"]:.4f}, {val_text}; '
              f'training {train_seconds:.2f}s ({byte_count / train_seconds:,.0f} bytes/s), validation {validation_seconds:.2f}s', flush=True)
        (output / 'epochs.json').write_text(json.dumps(epochs, indent=2) + '\n', encoding='utf-8')
        if should_stop and epoch < args.epochs:
            stopped_early = True
            print(f'Early stopping at epoch {epoch}: validation did not improve by more than '
                  f'{args.early_stopping_min_delta:g} bits/byte for {stopping.bad_checks} consecutive '
                  f'validation checks. Keeping best epoch {best_epoch} ({best_loss:.4f} bits/byte).', flush=True)
            break
    candidate_path.unlink()
    deployed = NeuralModel(model_path)
    selection_validation = epochs[best_epoch - 1]['validation']
    reference_validation = selection_validation
    reference_validation_seconds = 0.0
    if args.skip_cpu_validation:
        reference_validation = None
        print('Skipping final CPU validation; retaining the CUDA selection score.', flush=True)
    elif args.validation_device == 'cuda':
        print(f'Checking best epoch {best_epoch} with CPU reference validation', flush=True)
        reference_started = time.perf_counter()
        reference_validation = evaluate(validation, deployed)
        reference_validation_seconds = time.perf_counter() - reference_started
    print(f'Evaluating best epoch {best_epoch} on test data', flush=True)
    test_started = time.perf_counter()
    test_result = evaluate(test, deployed)
    test_seconds = time.perf_counter() - test_started
    # Reuse precisely the statistical trainer's sampling, framing, and round-trip checks.
    selected = reservoir(test, args.benchmark_messages, args.seed) if not isinstance(test, list) else test
    if isinstance(selected, list) and args.benchmark_messages and len(selected) > args.benchmark_messages:
        selected = random.Random(args.seed).sample(selected, args.benchmark_messages)
    original = payload_size = framed_size = count = 0
    bench_start = time.perf_counter()
    for message in selected:
        payload = compress_payload(message, deployed)
        if decompress_payload(payload, len(message), deployed) != message:
            raise ValueError('Neural arithmetic round trip failed')
        count += 1
        original += len(message)
        payload_size += len(payload)
        framed_size += len(payload) + 4 + max(1, (len(message).bit_length() + 6) // 7)
    if not original:
        raise ValueError('Empty benchmark')
    actual = {'messages': count, 'original_utf8_bytes': original, 'arithmetic_payload_bytes': payload_size,
              'minimal_framed_bytes': framed_size, 'payload_fraction_of_original': payload_size / original,
              'framed_fraction_of_original': framed_size / original, 'all_round_trips_exact': True,
              'reference_python_encode_decode_seconds': time.perf_counter() - bench_start}
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    report = {'architecture': 'Fixed preceding-byte embeddings, flattened context, tanh hidden layer, 256-way byte softmax',
              'requested_model_kib': args.model_kib,
              'model': {'model_bytes': deployed.size_bytes, 'parameter_count': parameter_count,
                        'context_order': args.context_bytes, 'embedding_dim': args.embedding_dim,
                        'hidden_dim': hidden, 'storage': args.storage, 'sha256': deployed.sha256},
              'corpus': {'sha256': file_sha256(source), **corpus}, 'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'best_epoch': best_epoch, 'validation': reference_validation,
              'selection_validation': selection_validation,
              'cpu_validation_skipped': args.skip_cpu_validation,
              'early_stopping': {'enabled': args.early_stopping_patience > 0,
                                 'patience': args.early_stopping_patience,
                                 'min_delta_bits_per_utf8_byte': args.early_stopping_min_delta,
                                 'stopped_early': stopped_early, 'completed_epochs': len(epochs),
                                 'max_epochs': args.epochs, 'validation_checks': stopping.checks,
                                 'checks_without_significant_improvement': stopping.bad_checks,
                                 'reason': 'validation_no_improvement' if stopped_early else 'max_epochs_reached'},
              'test': test_result, 'benchmark': actual, 'epochs': epochs,
              'training_seconds': time.perf_counter() - started,
              'phase_seconds': {'training': sum(row['training_seconds'] for row in epochs),
                               'export': sum(row['export_seconds'] for row in epochs),
                               'validation': sum(row['validation_seconds'] for row in epochs),
                               'reference_validation': reference_validation_seconds,
                               'test': test_seconds, 'benchmark': actual['reference_python_encode_decode_seconds']},
              'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'training_device': args.device, 'requested_device': requested_device,
                          'validation_device': args.validation_device,
                          'cuda_version': torch.version.cuda,
                          'gpu_name': torch.cuda.get_device_name(0) if uses_cuda else None,
                          'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if uses_cuda else 0},
              'notes': ['File size includes header, weights, biases, and int8 scales; excludes optimizer and training state.',
                        'Training uses float32; validation/test use exported, dequantized weights and integer coding probabilities.',
                        'NumPy float64 CPU inference reference; native MCU port and cross-platform probability agreement are not verified.',
                        'Optional CUDA float64 validation uses exported weights and codec frequency allocation, but floating-point rounding may differ. Checkpoint selection uses selection_validation. validation is the CPU reference score, or null when skipped; test and coding always use CPU.',
                        'Only scheduled validation epochs and the final epoch compete for checkpoint selection; skipped epochs have validation=null.',
                        'Early stopping monitors validation coding loss, never test loss. Patience counts validation checks, and the lowest measured validation checkpoint is retained independently of min_delta.',
                        'Context resets for each message/prepared record; no history crosses splits or record boundaries.',
                        'Shared model and radio overhead excluded; framing estimates nonce(2)+ULEB128 length+CRC16(2).']}
    vectors = []
    for message in (b'', b'Hello mesh!', bytes(range(256))):
        payload = compress_payload(message, deployed)
        vectors.append({'input_hex': message.hex(), 'payload_hex': payload.hex()})
    (output / 'test-vectors.json').write_text(json.dumps(vectors, indent=2) + '\n', encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'model_bytes': deployed.size_bytes, 'best_epoch': best_epoch,
                      'completed_epochs': len(epochs), 'stopped_early': stopped_early,
                      'test_bits_per_utf8_byte': test_result['bits_per_utf8_byte'], **actual}, indent=2))
    return report


def build_parser(*, sweep=False):
    parser = argparse.ArgumentParser(description=__doc__)
    if sweep:
        parser.add_argument('input', type=Path, nargs='?', default=Path('data/messages.jsonl'),
                            help='MeshCore JSONL/CSV or prepared Wikipedia directory/manifest.json')
    else:
        parser.add_argument('input', type=Path, help='MeshCore JSONL/CSV or prepared Wikipedia directory/manifest.json')
    parser.add_argument('--output', type=Path)
    if not sweep:
        parser.add_argument('--model-kib', type=int, default=256, help='maximum complete exported model size, KiB')
        parser.add_argument('--context-bytes', '--order', dest='context_bytes', type=int, default=4)
    parser.add_argument('--storage', choices=('float32', 'int8'), default='int8')
    parser.add_argument('--embedding-dim', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10,
                        help='maximum training epochs; early stopping may finish sooner (default: 10)')
    parser.add_argument('--early-stopping-patience', '--patience', type=int, default=3,
                        help='stop after N validation checks without improvement; default 3; 0 disables early stopping')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                        help='minimum significant decrease in validation bits/byte; default 0 (strict improvement)')
    parser.add_argument('--batch-size', type=int, default=1024, help='training examples (one example per input byte)')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    if not sweep:
        parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                            help='auto selects CUDA when available, otherwise CPU (default: auto)')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--text-field', choices=('body', 'text'), default='body')
    parser.add_argument('--split', choices=('chronological', 'random'), default='chronological')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--keep-duplicates', action='store_true')
    parser.add_argument('--limit-messages', type=int)
    parser.add_argument('--max-message-bytes', type=int, default=8192)
    parser.add_argument('--train-bytes', type=int, help='whole-message training byte limit per epoch; default all training data')
    parser.add_argument('--evaluation-messages', type=int, default=1000, help='reservoir per held-out split; 0 evaluates all')
    parser.add_argument('--validate-every', type=int, default=VALIDATION_DEFAULTS['validate_every'],
                        help='validate/export every N epochs; the final epoch is always validated (default: 1)')
    parser.add_argument('--validation-device', choices=('cpu', 'cuda'), default=VALIDATION_DEFAULTS['validation_device'],
                        help='checkpoint selection scoring: exact CPU reference (default), or batched float64 CUDA')
    parser.add_argument('--skip-cpu-validation', action='store_true', default=VALIDATION_DEFAULTS['skip_cpu_validation'],
                        help='skip the final CPU validation pass after CUDA selection; requires --validation-device cuda; test scoring and coding still run')
    parser.add_argument('--validation-batch-size', type=int, default=VALIDATION_DEFAULTS['validation_batch_size'],
                        help='byte examples per CUDA validation batch (default: 4096)')
    parser.add_argument('--benchmark-messages', type=int, default=100, help='arithmetic round trips; 0 uses all evaluated test messages')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train(args)
    except ImportError as exc:
        parser.exit(2, f'Missing dependency: {exc}. Install torch and numpy with the Python used to run this script.\n')
    except (ValueError, OSError, RuntimeError, MemoryError, OverflowError) as exc:
        parser.exit(2, f'Neural training failed: {exc}\n')
    return 0
