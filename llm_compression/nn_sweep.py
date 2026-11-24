"""Sequential, resumable CUDA neural compression sweeps through train_nn_cuda.py."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import uuid

from .bit_sweep import integer_list
from .bit_training import file_sha256
from .nn_model import NeuralModel, layout
from .nn_training import LEGACY_EARLY_STOPPING_DEFAULTS, VALIDATION_DEFAULTS, GPU_PIPELINE_DEFAULTS, build_parser


SWEEP_OPTIONS = {'input', 'output', 'sizes_kib', 'contexts', 'resume', 'plot_only'}
LEGACY_ARCHITECTURE_DEFAULTS = {'hidden_layers': 1}


def run_command(args, size, context, directory):
    launcher = Path(__file__).resolve().parent.parent / 'train_nn_cuda.py'
    if not launcher.is_file():
        raise ValueError(f'CUDA training launcher is missing: {launcher}')
    command = [sys.executable, '-u', str(launcher), str(args.input.resolve()),
               '--device', 'cuda', '--model-kib', str(size),
               '--context-bytes', str(context), '--output', str(directory.resolve())]
    for key, value in vars(args).items():
        if key in SWEEP_OPTIONS or value is None:
            continue
        option = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(option)
            elif key == 'skip_cpu_validation':
                command.append('--no-skip-cpu-validation')
        else:
            command.extend((option, str(value)))
    return command


def checked_report(directory, command, corpus_sha256, verification_sha256=None):
    """Check model integrity and every trainer setting before reusing results."""
    expected = vars(build_parser(cuda_defaults=True).parse_args(command[3:]))
    expected = {key: str(value) if isinstance(value, Path) else value for key, value in expected.items()}
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    model = NeuralModel(directory / 'model.mcnn')
    hidden, size = layout(expected['model_kib'], expected['context_bytes'],
                          expected['embedding_dim'], expected['storage'], expected['hidden_layers'])
    metadata = {'sha256': model.sha256, 'context_order': model.context,
                'embedding_dim': model.embedding, 'hidden_dim': model.hidden,
                'storage': model.storage, 'model_bytes': model.size_bytes}
    if ({**VALIDATION_DEFAULTS, **GPU_PIPELINE_DEFAULTS, **LEGACY_EARLY_STOPPING_DEFAULTS, **LEGACY_ARCHITECTURE_DEFAULTS, **{'verification_input':None}, **report['settings']} != expected
            or report['corpus']['sha256'] != corpus_sha256
            or report['corpus'].get('verification',{}).get('sha256') != verification_sha256
            or report['requested_model_kib'] != expected['model_kib']
            or report['runtime']['training_device'] != 'cuda'
            or report.get('cpu_validation_skipped', False) != expected['skip_cpu_validation']
            or (report.get('validation') is None) != expected['skip_cpu_validation']
            or not report['benchmark']['all_round_trips_exact']
            or any(report['model'].get(key) != value for key, value in metadata.items())
            or report['model'].get('hidden_layers', 1) != model.hidden_layers
            or model.hidden_layers != expected['hidden_layers']
            or (model.context, model.embedding, model.hidden, model.storage, model.size_bytes)
            != (expected['context_bytes'], expected['embedding_dim'], hidden, expected['storage'], size)):
        raise ValueError(f'Completed run does not match configuration: {directory}')
    return report


def summarize(report, directory):
    training_phase = report.get('phase_seconds', {}).get('training')
    training_bytes = sum(epoch['training_bytes'] for epoch in report['epochs'])
    validation = report.get('validation') or report['selection_validation']
    validation_score_device = 'cpu' if report.get('validation') is not None else 'cuda'
    return {
        'requested_model_kib': report['requested_model_kib'],
        'context_bytes': report['model']['context_order'],
        'model_bytes': report['model']['model_bytes'],
        'storage': report['model']['storage'],
        'embedding_dim': report['model']['embedding_dim'],
        'hidden_dim': report['model']['hidden_dim'],
        'hidden_layers': report['model'].get('hidden_layers', 1),
        'parameter_count': report['model']['parameter_count'],
        'best_epoch': report['best_epoch'],
        'completed_epochs': len(report['epochs']),
        'stopped_early': report.get('early_stopping', {}).get('stopped_early', False),
        'validation_bits_per_utf8_byte': validation['bits_per_utf8_byte'],
        'validation_score_device': validation_score_device,
        'selection_validation_bits_per_utf8_byte': report.get('selection_validation', report['validation'])['bits_per_utf8_byte'],
        'validation_device': report['runtime'].get('validation_device', 'cpu'),
        'test_bits_per_utf8_byte': report['test']['bits_per_utf8_byte'],
        'test_device': report['runtime'].get('test_device', 'cpu'),
        'validation_precision': report['runtime'].get('validation_precision', 'float64'),
        'test_precision': report['runtime'].get('test_precision', 'float64'),
        'payload_percent': 100 * report['benchmark']['payload_fraction_of_original'],
        'framed_percent': 100 * report['benchmark']['framed_fraction_of_original'],
        'test_messages': report['test']['messages'],
        'benchmark_messages': report['benchmark']['messages'],
        'training_seconds': report['training_seconds'],
        'training_phase_seconds': training_phase,
        'training_bytes_per_second': training_bytes / training_phase if training_phase else None,
        'gpu_name': report['runtime']['gpu_name'],
        'peak_cuda_allocated_bytes': report['runtime']['peak_cuda_allocated_bytes'],
        'report': str(directory / 'report.json'),
    }


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_summary(output, rows, skipped):
    write_json(output / 'summary.json', {
        'results': rows, 'skipped': skipped,
        'notes': ['Lower is better for compression metrics; select parameters using validation loss.',
                  'Test comparisons are exploratory; use fresh traffic for a final evaluation.',
                  'Payload and estimated framing exclude the shared model and radio overhead.',
                  'Validation uses the CPU reference when available, otherwise the CUDA selection estimate; validation_score_device identifies the metric. Test score device/precision are recorded separately; exact coding uses the NumPy CPU reference.',
                  'Context zero is a learned neural baseline and still varies with model budget.'],
    })
    if rows:
        temporary = output / 'summary.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / 'summary.csv')


def plot_results(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    try:
        score_devices = {row.get('validation_score_device', 'cpu') for row in rows}
        validation_label = 'Validation bits / UTF-8 byte'
        if score_devices == {'cuda'}:
            validation_label = 'CUDA validation estimate (bits / UTF-8 byte)'
        elif 'cuda' in score_devices:
            validation_label += ' (CPU/CUDA)'
        for context in sorted({row['context_bytes'] for row in rows}):
            group = sorted((row for row in rows if row['context_bytes'] == context),
                           key=lambda row: row['model_bytes'])
            axes[0, 0].plot([row['model_bytes'] / 1024 for row in group],
                            [row['validation_bits_per_utf8_byte'] for row in group],
                            'o-', label=f'Context {context}')
        for size in sorted({row['requested_model_kib'] for row in rows}):
            group = sorted((row for row in rows if row['requested_model_kib'] == size),
                           key=lambda row: row['context_bytes'])
            for ax, metric in ((axes[0, 1], 'validation_bits_per_utf8_byte'),
                               (axes[1, 0], 'payload_percent'), (axes[1, 1], 'framed_percent')):
                ax.plot([row['context_bytes'] for row in group], [row[metric] for row in group],
                        'o-', label=f'{size:g} KiB budget')
        axes[0, 0].set(xscale='log', xlabel='Actual model size (KiB; logarithmic)',
                       ylabel=validation_label, title='Model size vs compression')
        axes[0, 1].set(xlabel='Context (preceding bytes)', ylabel=validation_label,
                       title='Context depth vs compression')
        axes[1, 0].set(xlabel='Context (preceding bytes)', ylabel='Payload / original (%)',
                       title='Actual test arithmetic payload')
        axes[1, 1].set(xlabel='Context (preceding bytes)', ylabel='Estimated framed / original (%)',
                       title='Test payload plus estimated framing')
        for ax in axes.flat:
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize='small')
        for ax in axes[1]:
            ax.axhline(100, color='gray', linestyle=':')
        fig.suptitle('CUDA neural compression sweep — lower is better\n'
                     'Shared model excluded; dotted lines: original size', fontsize=12)
        for suffix in ('png', 'svg'):
            fig.savefig(output / f'compression.{suffix}', dpi=160)
    finally:
        plt.close(fig)


def main(argv=None):
    parser = build_parser(sweep=True, cuda_defaults=True)
    parser.description = 'Sweep train_nn_cuda.py over model budgets and preceding-byte contexts.'
    parser.add_argument('--sizes-kib', type=integer_list, default=integer_list('16,64,256,1024'),
                        help='comma-separated model budgets in KiB (default: 16,64,256,1024)')
    parser.add_argument('--contexts', '--orders', type=integer_list, default=integer_list('0,1,2,4,8,16'),
                        help='comma-separated context depths (default: 0,1,2,4,8,16)')
    parser.add_argument('--resume', action='store_true', help='reuse verified runs with the same corpus/settings')
    parser.add_argument('--plot-only', action='store_true', help='redraw summary plots without training or CUDA')
    parser.set_defaults(output=Path('models/nn-cuda-sweep'))
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        # Fail before training if the plotting dependency is unavailable.
        import matplotlib  # noqa: F401
        if args.plot_only:
            rows = json.loads((output / 'summary.json').read_text(encoding='utf-8'))['results']
            if not rows:
                raise ValueError('No completed runs to plot')
            plot_results(rows, output)
            return 0
        if min(args.sizes_kib) < 1:
            raise ValueError('Model budgets must be positive')
        source = args.input / 'manifest.json' if args.input.is_dir() else args.input
        config = {key: str(value.resolve()) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if key not in ('output', 'resume', 'plot_only')}
        config.update(format='nn-cuda-sweep-v2-fp32', input_sha256=file_sha256(source), device='cuda')
        if args.verification_input is not None:
            config['verification_sha256']=file_sha256(args.verification_input)
        manifest = output / 'sweep.json'
        if output.exists() and any(output.iterdir()):
            if not args.resume:
                raise ValueError('Output is not empty; use --resume or choose a new --output')
            if not manifest.is_file() or {**VALIDATION_DEFAULTS, **GPU_PIPELINE_DEFAULTS, **LEGACY_EARLY_STOPPING_DEFAULTS, **LEGACY_ARCHITECTURE_DEFAULTS,
                                          **{'verification_input':None}, **json.loads(manifest.read_text(encoding='utf-8'))} != config:
                raise ValueError('Resume configuration/corpus differs (including CUDA backend); choose a new --output')
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest, config)
        rows, skipped = [], []
        if not (output / 'summary.json').exists():
            save_summary(output, rows, skipped)
        for size in args.sizes_kib:
            for context in args.contexts:
                try:
                    layout(size, context, args.embedding_dim, args.storage, args.hidden_layers)
                except ValueError as exc:
                    skipped.append({'requested_model_kib': size, 'context_bytes': context, 'reason': str(exc)})
                    if not args.resume or rows:
                        save_summary(output, rows, skipped)
                    print(f'Skipping {size} KiB / context {context}: {exc}', file=sys.stderr)
                    continue
                directory = output / f'{size}k-context{context}'
                if directory.resolve().parent != output:
                    raise ValueError(f'Run directory escapes sweep output: {directory}')
                command = run_command(args, size, context, directory)
                print(f'Sweep: {size} KiB / context {context}', file=sys.stderr, flush=True)
                reuse = args.resume and (directory / 'report.json').is_file()
                if not reuse:
                    if directory.exists() and any(directory.iterdir()):
                        archived = directory.with_name(directory.name + '.incomplete-' + uuid.uuid4().hex[:8])
                        directory.rename(archived)
                        print(f'Preserved interrupted run at {archived}', file=sys.stderr)
                    subprocess.run(command, check=True)
                report = checked_report(directory, command, config['input_sha256'], config.get('verification_sha256'))
                if reuse:
                    print('Reusing verified completed run.', file=sys.stderr)
                rows.append(summarize(report, directory))
                save_summary(output, rows, skipped)
        if not rows:
            save_summary(output, rows, skipped)
            raise ValueError('No feasible configurations; increase budgets or reduce context/embedding sizes')
        plot_results(rows, output)
        print(f'Results: {output / "summary.csv"}\nGraphs: {output / "compression.png"} and compression.svg')
        print('Choose parameters using validation loss; test comparisons are exploratory.')
    except ImportError as exc:
        parser.exit(2, f'Missing dependency: {exc}. Install numpy and matplotlib in this Python environment.\n')
    except subprocess.CalledProcessError as exc:
        parser.exit(2, f'CUDA training exited with status {exc.returncode}; see the trainer error above. '
                    'Correct the problem and use --resume to retry the interrupted run.\n')
    except (ValueError, OSError, RuntimeError, MemoryError, OverflowError, KeyError, TypeError) as exc:
        parser.exit(2, f'Neural sweep failed: {exc}\n')
    except KeyboardInterrupt:
        print('Sweep interrupted. Repeat the same command with --resume to continue.', file=sys.stderr)
        return 130
    return 0
