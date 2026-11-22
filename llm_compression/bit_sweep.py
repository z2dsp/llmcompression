"""Sequential, resumable parameter sweeps for the fixed MCU bit-context model."""

import argparse
import csv
import json
from pathlib import Path
import sys
import uuid

from .bit_model import BitModel, layout
from .bit_training import build_parser, file_sha256, train


def integer_list(value):
    try:
        result = sorted(set(int(part.strip()) for part in value.split(',')))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected comma-separated integers') from exc
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError('Expected nonnegative integers')
    return result


def summarize(report, directory):
    return {
        'requested_model_kib': report['requested_model_kib'],
        'order': report['model']['context_order'],
        'model_bytes': report['model']['model_bytes'],
        'slots_per_prefix_per_order': report['model']['slots_per_prefix_per_order'],
        'selected_strength': report['selected_strength'],
        'validation_bits_per_utf8_byte': min(
            item['bits_per_utf8_byte'] for item in report['validation_candidates']),
        'test_bits_per_utf8_byte': report['test']['bits_per_utf8_byte'],
        'unigram_test_bits_per_utf8_byte': report['unigram_baseline']['test']['bits_per_utf8_byte'],
        'payload_percent': 100 * report['benchmark']['payload_fraction_of_original'],
        'framed_percent': 100 * report['benchmark']['framed_fraction_of_original'],
        'test_messages': report['test']['messages'],
        'benchmark_messages': report['benchmark']['messages'],
        'training_seconds': report['training_seconds'],
        'report': str(directory / 'report.json'),
    }


def save_summary(output, rows, skipped):
    data = {'results': rows, 'skipped': skipped,
            'notes': ['Lower is better for all compression metrics.',
                      'Payload and estimated framing exclude the shared model and radio overhead.',
                      'Order zero always uses the same 1088-byte model, regardless of budget.',
                      'Use validation to choose parameters; repeated test comparisons make the test set exploratory.']}
    (output / 'summary.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    if rows:
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_results(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    sizes = sorted({row['requested_model_kib'] for row in rows})
    orders = sorted({row['order'] for row in rows})
    for order in orders:
        group = sorted((r for r in rows if r['order'] == order), key=lambda r: r['model_bytes'])
        # Order zero is independent of the requested budget: show one point.
        points = {r['model_bytes']: r for r in group}
        group = list(points.values())
        axes[0, 0].plot([r['model_bytes'] / 1024 for r in group],
                        [r['test_bits_per_utf8_byte'] for r in group], 'o-', label=f'Order {order}')
    for size in sizes:
        group = sorted((r for r in rows if r['requested_model_kib'] == size), key=lambda r: r['order'])
        x = [r['order'] for r in group]
        for ax, metric in ((axes[0, 1], 'test_bits_per_utf8_byte'),
                           (axes[1, 0], 'payload_percent'), (axes[1, 1], 'framed_percent')):
            ax.plot(x, [r[metric] for r in group], 'o-', label=f'{size:g} KiB budget')
    axes[0, 0].set(xscale='log', xlabel='Actual model size (KiB; logarithmic)', ylabel='Test bits / UTF-8 byte', title='Model size vs compression')
    axes[0, 1].set(xlabel='Context order (preceding bytes)', ylabel='Test bits / UTF-8 byte', title='Context depth vs compression')
    axes[1, 0].set(xlabel='Context order (preceding bytes)', ylabel='Payload / original (%)', title='Actual arithmetic payload')
    axes[1, 1].set(xlabel='Context order (preceding bytes)', ylabel='Estimated framed / original (%)', title='Payload plus estimated framing')
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize='small')
    for ax in axes[0]:
        ax.axhline(rows[0]['unigram_test_bits_per_utf8_byte'], color='gray', linestyle=':', label='_nolegend_')
    for ax in axes[1]:
        ax.axhline(100, color='gray', linestyle=':')
    fig.suptitle('MCU compression sweep — lower is better\nShared model excluded; dotted lines: unigram loss / original size', fontsize=12)
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'compression.{suffix}', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = build_parser(sweep=True)
    parser.description = 'Train and graph a grid of MCU model budgets and preceding-byte context orders.'
    parser.add_argument('--sizes-kib', type=integer_list, default=integer_list('16,64,256,1024'),
                        help='model budgets in KiB, comma-separated (default: 16,64,256,1024)')
    parser.add_argument('--orders', type=integer_list, default=integer_list('0,1,2,4,8'),
                        help='context depths, comma-separated (default: 0,1,2,4,8)')
    parser.add_argument('--resume', action='store_true', help='reuse verified completed runs with the same configuration')
    parser.add_argument('--plot-only', action='store_true', help='regenerate graphs from an existing summary.json without training')
    parser.set_defaults(output=Path('models/mcu-sweep'))
    args = parser.parse_args(argv)
    output = args.output
    try:
        # Check the plotting dependency before starting potentially expensive training.
        import matplotlib  # noqa: F401
        if args.plot_only:
            rows = json.loads((output / 'summary.json').read_text(encoding='utf-8'))['results']
            if not rows:
                raise ValueError('No completed runs to plot')
            plot_results(rows, output)
            return 0
        if min(args.sizes_kib) < 2:
            raise ValueError('Model budgets must be at least 2 KiB')
        source = args.input / 'manifest.json' if args.input.is_dir() else args.input
        config = {key: str(value.resolve()) if isinstance(value, Path) else value
                  for key, value in vars(args).items()
                  if key not in ('output', 'resume', 'plot_only', 'model_kib', 'order')}
        config['input_sha256'] = file_sha256(source)
        manifest = output / 'sweep.json'
        if output.exists() and any(output.iterdir()):
            if not args.resume:
                raise ValueError('Output is not empty; use --resume or choose a new --output')
            if not manifest.exists() or json.loads(manifest.read_text(encoding='utf-8')) != config:
                raise ValueError('Resume configuration/corpus differs; choose a new --output')
        output.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        rows, skipped = [], []
        for size in args.sizes_kib:
            for order in args.orders:
                try:
                    layout(size, order)
                except ValueError as exc:
                    skipped.append({'requested_model_kib': size, 'order': order, 'reason': str(exc)})
                    print(f'Skipping {size} KiB / order {order}: {exc}', file=sys.stderr)
                    continue
                directory = output / f'{size}k-order{order}'
                report_path = directory / 'report.json'
                print(f'Sweep: {size} KiB / order {order}', file=sys.stderr, flush=True)
                if args.resume and report_path.exists():
                    report = json.loads(report_path.read_text(encoding='utf-8'))
                    with BitModel(directory / 'model.mcblm') as model:
                        if (model.sha256 != report['model']['sha256'] or model.order != order
                                or report['requested_model_kib'] != size
                                or report['corpus']['sha256'] != config['input_sha256']):
                            raise ValueError(f'Completed run does not match configuration: {directory}')
                else:
                    if directory.exists() and any(directory.iterdir()):
                        # Preserve interrupted artifacts before retrying; both paths must stay in this sweep.
                        archived = directory.with_name(directory.name + '.incomplete-' + uuid.uuid4().hex[:8])
                        if directory.resolve().parent != output.resolve() or archived.resolve().parent != output.resolve():
                            raise ValueError('Run directory escapes sweep output')
                        directory.rename(archived)
                        print(f'Preserved interrupted run at {archived}', file=sys.stderr)
                    run_args = argparse.Namespace(**vars(args))
                    run_args.model_kib, run_args.order, run_args.output = size, order, directory
                    report = train(run_args)
                rows.append(summarize(report, directory))
                save_summary(output, rows, skipped)
        save_summary(output, rows, skipped)
        if not rows:
            raise ValueError('No feasible configurations; increase budgets or reduce context orders')
        plot_results(rows, output)
        print(f'Results: {output / "summary.csv"}\nGraphs: {output / "compression.png"} and compression.svg')
        print('Use validation loss to select parameters; treat the test plots as exploratory.')
    except ImportError as exc:
        parser.exit(2, f'Missing dependency: {exc}. Install numpy and matplotlib (or pip install -e ".[sweep]").\n')
    except (ValueError, OSError, MemoryError, OverflowError) as exc:
        parser.exit(2, f'Sweep failed: {exc}\n')
    return 0
