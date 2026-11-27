"""T9 conversion, constrained inference, export, and training integration."""
import json

import numpy as np
import pytest

from llm_compression.t9 import (Accuracy, CLASS_KEYS, KEY_LETTERS, decode_keys, encode_main,
                                encode_text, evaluate, normalize_text, parse_keys, predict_main)
from llm_compression.t9_model import T9Model, export_model, layout, model_size, shapes
from llm_compression.t9_training import build_parser, load_corpus, main, make_network, training_batches


def make_model(path, storage='int8', context=3, key_context=4, hidden_layers=2, budget=4):
    hidden, size = layout(budget, context, key_context, 2, storage, hidden_layers)
    rng = np.random.default_rng(31)
    arrays = [rng.normal(0, 0.1, shape).astype('float32')
              for shape in shapes(context, key_context, 2, hidden, hidden_layers)]
    export_model(path, arrays, context, key_context, 2, hidden, storage, hidden_layers)
    return T9Model(path), size


def test_notation_preserves_literals_and_normalizes_ascii_only():
    text = 'Hello World! 0123456789\\\r\n\tcaf\u00e9 \U0001f642 A\u0301'
    keys = encode_text(text)
    assert keys.startswith('43556096753!0' + ''.join('\\' + str(i) for i in range(10)))
    tokens = parse_keys(keys)
    expected = normalize_text(text)
    assert len(tokens) == len(expected)
    for token, character in zip(tokens, expected):
        if isinstance(token, int):
            assert character in KEY_LETTERS[token]
        else:
            assert token == character
    assert normalize_text(text) == 'hello world! 0123456789\\\r\n\tcaf\u00e9 \U0001f642 a\u0301'


@pytest.mark.parametrize('keys', ['hello', '1', '\\', '\\a', '2 3', 'ABC'])
def test_malformed_keys_rejected(keys):
    with pytest.raises(ValueError, match='keystroke|backslash'):
        parse_keys(keys)


class UniformModel:
    context = 3
    key_context = 4

    def logits(self, histories, keys):
        return np.zeros((len(histories), 26), dtype=np.float32)


def test_decoder_constraints_deterministic_ties_and_literals():
    text = 'Hello 0! 29\\\r\ncaf\u00e9'
    model = UniformModel()
    for width in (1, 2, 8):
        prediction = decode_keys(model, encode_text(text), beam_width=width)
        assert encode_text(prediction) == encode_text(text)
        assert prediction.endswith('aad\u00e9')
        assert '\r\n' in prediction
    assert decode_keys(model, '') == ''
    with pytest.raises(ValueError, match='Beam'):
        decode_keys(model, '2', beam_width=0)


def test_beam_search_can_recover_a_word_greedy_prunes():
    class Conditional(UniformModel):
        context = 1

        def logits(self, histories, keys):
            result = np.zeros((len(histories), 26), dtype=np.float32)
            for row, history in enumerate(histories):
                if history[-1] == 256:
                    result[row, :3] = np.log([0.51, 0.48, 0.01])
                elif history[-1] == ord('b'):
                    result[row, :3] = np.log([0.01, 0.98, 0.01])
                else:
                    result[row, :3] = np.log([0.34, 0.33, 0.33])
            return result

    assert decode_keys(Conditional(), '22', beam_width=1) == 'aa'
    assert decode_keys(Conditional(), '22', beam_width=2) == 'bb'


def test_lookahead_stops_at_literals_and_history_uses_predictions():
    class Recording(UniformModel):
        def __init__(self):
            self.calls = []

        def logits(self, histories, keys):
            self.calls.append((histories.copy(), keys.copy()))
            return super().logits(histories, keys)

    model = Recording()
    result = evaluate(model, ['BB DE'])
    assert result['letter_accuracy'] == 0.25
    calls = model.calls
    assert calls[0][0].tolist() == [[256, 256, 256]]
    assert calls[0][1].tolist() == [[2, 2, 0, 0]]
    assert calls[1][1].tolist() == [[2, 0, 0, 0]] * 3
    # After the first word commits "aa", context is "aa ", not reference "bb ".
    assert calls[2][0].tolist() == [[97, 97, 32]]
    assert calls[2][1].tolist() == [[3, 3, 0, 0]]


def test_accuracy_denominators_exclude_literals_for_letter_and_word_metrics():
    accuracy = Accuracy()
    accuracy.add('Good 2!', 'home 2!')  # both encode to 4663
    result = accuracy.report()
    assert result['letters'] == 4 and result['correct_letters'] == 1
    assert result['letter_accuracy'] == 0.25
    assert result['character_accuracy'] == 4 / 7
    assert result['word_accuracy'] == 0
    assert result['message_accuracy'] == 0
    accuracy.add('', '')
    assert accuracy.report()['message_accuracy'] == 0.5
    assert Accuracy().report()['letter_accuracy'] is None
    with pytest.raises(ValueError, match='same T9'):
        accuracy.add('hi', 'no')


@pytest.mark.parametrize('storage', ['int8', 'float32'])
@pytest.mark.parametrize('budget', [1, 32, 1024, 10240])
@pytest.mark.parametrize('layers', [1, 2, 3])
def test_largest_width_fits_budget(storage, budget, layers):
    if model_size(32, 16, 8, 1, storage, layers) > budget * 1024:
        with pytest.raises(ValueError, match='Budget too small'):
            layout(budget, 32, 16, 8, storage, layers)
        return
    hidden, size = layout(budget, 32, 16, 8, storage, layers)
    assert size <= budget * 1024 < model_size(32, 16, 8, hidden + 1, storage, layers)


@pytest.mark.parametrize('storage', ['int8', 'float32'])
@pytest.mark.parametrize('context', [0, 3])
@pytest.mark.parametrize('layers', [1, 2, 3])
def test_export_size_runtime_and_corruption(tmp_path, storage, context, layers):
    path = tmp_path / 'model.t9nn'
    model, size = make_model(path, storage, context=context, hidden_layers=layers)
    assert path.stat().st_size == model.size_bytes == size <= 4096
    prediction = decode_keys(model, encode_text('Hello, world! 42'))
    assert encode_text(prediction) == encode_text('Hello, world! 42')
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match='checksum'):
        T9Model(path)


@pytest.mark.parametrize('context', [0, 2, 10])
@pytest.mark.parametrize('key_context', [1, 4, 15])
@pytest.mark.parametrize('batch_size', [1, 3, 50])
def test_batch_features_match_independent_reference(context, key_context, batch_size):
    messages = [b'Ab cd! ', b'EF', 'z\u00e9 hi'.encode(), b'123']
    expected = []
    for message in messages:
        data = normalize_text(message.decode()).encode()
        for position, byte in enumerate(data):
            if 97 <= byte <= 122:
                previous = ([256] * context + list(data[:position]))[-context:] if context else []
                keys = []
                for following in data[position:position + key_context]:
                    if not 97 <= following <= 122:
                        break
                    keys.append(int(CLASS_KEYS[following - 97]))
                keys += [0] * (key_context - len(keys))
                expected.append((previous, keys, byte - 97))
    actual = []
    for x, keys, y in training_batches(messages, context, key_context, batch_size):
        assert len(y) <= batch_size
        actual.extend(zip(x.tolist(), keys.tolist(), y.tolist()))
    assert actual == expected


@pytest.mark.parametrize('layers', [1, 2, 3])
def test_torch_forward_mask_gradients_and_numpy_agree(tmp_path, layers):
    torch = pytest.importorskip('torch')
    torch.manual_seed(6)
    network = make_network(torch, 3, 4, 2, 5, layers)
    history = torch.tensor([[256, 97, 98], [32, 99, 100]])
    keys = torch.tensor([[2, 3, 0, 0], [9, 0, 0, 0]])
    logits = network(history, keys)
    assert torch.isfinite(logits).sum().item() == 7
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 25])).backward()
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in network.parameters())
    path = tmp_path / 'model.t9nn'
    export_model(path, network.arrays(), 3, 4, 2, 5, 'float32', layers)
    exported = T9Model(path).logits(history.numpy(), keys.numpy())
    allowed = np.isfinite(logits.detach().numpy())
    np.testing.assert_allclose(exported[allowed], logits.detach().numpy()[allowed], atol=1e-6)


def test_normalized_duplicates_never_cross_splits(tmp_path):
    source = tmp_path / 'messages.jsonl'
    source.write_text(''.join(json.dumps({'body': text}) + '\n'
                              for text in ('HELLO', 'hello', 'Bye', 'good', 'home', 'test', 'Test')))
    args = build_parser().parse_args([str(source), '--keep-duplicates', '--split', 'random'])
    partitions, report = load_corpus(args)
    assert report['unique_messages'] == 5 and report['duplicates'] == 2
    assert not set(partitions['train']) & set(partitions['validation'])
    assert not set(partitions['train']) & set(partitions['test'])
    assert not set(partitions['validation']) & set(partitions['test'])
    assert sum(len(records) for records in partitions.values()) == 7


def test_single_paragraph_is_segmented_for_training(tmp_path):
    source = tmp_path / 'text.txt'
    source.write_text('the quick brown fox jumps over the lazy dog. hello world. ' * 20)
    partitions, report = load_corpus(build_parser().parse_args([str(source)]))
    assert report['automatic_segmentation']
    assert all(partitions.values())
    assert not set(partitions['train']) & set(partitions['validation'])
    assert not set(partitions['train']) & set(partitions['test'])


@pytest.mark.parametrize('kind', ['txt', 'jsonl', 'csv', 'prepared'])
def test_training_and_cli_end_to_end(tmp_path, kind):
    if kind == 'prepared':
        from llm_compression.training_corpus import RecordWriter
        source = tmp_path / 'prepared'
        source.mkdir()
        splits = {}
        for name in ('train', 'validation', 'test'):
            writer = RecordWriter(source / f'{name}.mccorpus')
            for number in range(5):
                writer.write(f'{name} hello world {number}'.encode())
            stats = writer.close()
            splits[name] = {**stats, 'files': [{'path': f'{name}.mccorpus', **stats}]}
        (source / 'manifest.json').write_text(json.dumps({
            'format': 'mesh-compression-corpus-v1', 'source': {'test': True},
            'preparation': {'sample_bytes': 160}, 'splits': splits}))
    else:
        source = tmp_path / ('messages.' + kind)
        texts = [f'hello world number {number}' for number in range(20)]
        if kind == 'csv':
            import csv
            with source.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['body'])
                writer.writeheader()
                writer.writerows({'body': text} for text in texts)
        elif kind == 'jsonl':
            source.write_text(''.join(json.dumps({'body': text}) + '\n' for text in texts))
        else:
            source.write_text('\n'.join(texts))
    output = tmp_path / 'trained'
    arguments = [str(source), '--output', str(output), '--model-kib', '4', '--context-bytes', '3',
                 '--key-context', '4', '--embedding-dim', '2', '--hidden-layers', '2', '--epochs', '2',
                 '--batch-size', '64', '--evaluation-messages', '2', '--threads', '1', '--device', 'cpu',
                 '--validate-every', '2']
    assert main(arguments) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['best_epoch'] == 2 and report['epochs'][0]['validation'] is None
    assert report['model']['model_bytes'] <= 4096
    model_path = output / 'model.t9nn'
    deployed = T9Model(model_path)
    assert report['model']['sha256'] == deployed.sha256
    partitions, _ = load_corpus(build_parser().parse_args(arguments))
    assert report['test'] == evaluate(deployed, partitions['test'], beam_width=8)

    reference, keys, prediction, metrics = (tmp_path / name for name in ('input.txt', 'input.t9', 'guess.txt', 'score.json'))
    reference.write_bytes('Hello, world! 42\\\r\ncaf\u00e9'.encode('utf-8'))
    assert encode_main([str(reference), str(keys)]) == 0
    assert predict_main([str(model_path), str(keys), str(prediction),
                         '--reference', str(reference), '--report', str(metrics)]) == 0
    assert encode_text(prediction.read_bytes().decode()) == keys.read_bytes().decode()
    accuracy = json.loads(metrics.read_text())
    assert accuracy['letters'] == 13
    assert accuracy['model_bytes'] <= 4096
    with pytest.raises(SystemExit):
        encode_main([str(reference), str(keys)])
    with pytest.raises(SystemExit):
        predict_main([str(model_path), str(keys), str(prediction)])
    with pytest.raises(SystemExit):
        main(arguments)


def test_cli_reference_mismatch_does_not_write_output(tmp_path):
    model_path = tmp_path / 'model.t9nn'
    make_model(model_path)
    keys, reference, output = (tmp_path / name for name in ('keys.t9', 'wrong.txt', 'prediction.txt'))
    keys.write_text('22')
    reference.write_text('hello')
    with pytest.raises(SystemExit):
        predict_main([str(model_path), str(keys), str(output), '--reference', str(reference)])
    assert not output.exists()


def test_cuda_launcher_reuses_cached_runtime(tmp_path, monkeypatch):
    import sys
    import train_t9_cuda
    from llm_compression import t9_training
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    runtime = tmp_path / 'llm-compression/cuda-2.5.1-cu124-py311'
    (runtime / 'torch').mkdir(parents=True)
    (runtime / 'torch/__init__.py').touch()
    (runtime / '.ready').touch()
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(sys, 'version_info', (3, 11, 0))
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    calls = []
    monkeypatch.setattr(t9_training, 'main', lambda args: calls.append(args) or 0)
    assert train_t9_cuda.main(['messages.jsonl', '--epochs', '1']) == 0
    assert calls == [['--device', 'cuda', 'messages.jsonl', '--epochs', '1']]
    assert str(runtime) in sys.path


def test_best_checkpoint_selected_by_accuracy_and_test_never_competes(tmp_path, monkeypatch):
    from llm_compression import t9_training
    source = tmp_path / 'messages.jsonl'
    source.write_text(''.join(json.dumps({'body': f'hello world {i}'}) + '\n' for i in range(20)))
    calls = []

    def scoring(model, messages, **kwargs):
        calls.append(model.sha256)
        rate = [0.25, 0.75, 0.5, 1.0][len(calls) - 1]
        return {'word_accuracy': rate, 'letter_accuracy': rate, 'letters': 10}

    monkeypatch.setattr(t9_training, 'evaluate', scoring)
    output = tmp_path / 'model'
    assert main([str(source), '--output', str(output), '--model-kib', '4', '--epochs', '5',
                 '--context-bytes', '2', '--key-context', '3', '--embedding-dim', '2',
                 '--threads', '1', '--device', 'cpu', '--patience', '1']) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['best_epoch'] == 2
    assert report['completed_epochs'] == 3 and report['stopped_early']
    assert report['validation']['word_accuracy'] == 0.75
    assert report['test']['word_accuracy'] == 1
    assert T9Model(output / 'model.t9nn').sha256 == calls[1] == calls[3]
