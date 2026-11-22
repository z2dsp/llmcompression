import json

import numpy as np
import pytest

from llm_compression import nn_training


def reference_batches(messages, context, batch_size, rng):
    x, y = [], []
    for message in messages:
        history = [256] * context
        for byte in message:
            x.append(history.copy())
            y.append(byte)
            if history:
                history.pop(0)
                history.append(byte)
            if len(y) == batch_size:
                yield nn_training._batch(x, y, context, rng)
                x, y = [], []
    if y:
        yield nn_training._batch(x, y, context, rng)


@pytest.mark.parametrize('context', [0, 1, 4, 16, 257])
@pytest.mark.parametrize('batch_size', [1, 7, 64, 1024])
@pytest.mark.parametrize('shuffle', [False, True])
def test_vectorized_batches_exactly_match_original_examples_and_shuffle(context, batch_size, shuffle):
    generator = np.random.default_rng(12)
    messages = [b'', b'ab', b'c', 'Café 🌊'.encode(), bytes(range(256)) * 5,
                *[generator.bytes(size) for size in generator.integers(0, 40, size=100)]]
    actual = list(nn_training.training_batches(iter(messages), context, batch_size,
                  np.random.default_rng(42) if shuffle else None))
    expected = list(reference_batches(messages, context, batch_size,
                    np.random.default_rng(42) if shuffle else None))
    assert len(actual) == len(expected)
    for (x, y), (old_x, old_y) in zip(actual, expected):
        np.testing.assert_array_equal(x, old_x)
        np.testing.assert_array_equal(y, old_y)
        assert x.dtype == y.dtype == np.int64


def test_batch_generation_is_lazy_and_keeps_returned_arrays_valid():
    consumed = []
    def messages():
        for i in range(1000):
            consumed.append(i)
            yield b'abcdefgh'
    source = nn_training.training_batches(messages(), 4, 16)
    x, y = next(source)
    original_x, original_y = x.copy(), y.copy()
    assert len(consumed) == 2
    for _ in range(10):
        next(source)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    source.close()


def test_nonfinite_loss_is_rejected_before_export(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip('torch')
    corpus = tmp_path / 'messages.jsonl'
    corpus.write_text(''.join(json.dumps({'body': f'hello {i}'}) + '\n' for i in range(20)))
    original = nn_training.make_network
    def broken_network(*args):
        model = original(*args)
        with torch.no_grad():
            model.output.weight.fill_(float('nan'))
        return model
    monkeypatch.setattr(nn_training, 'make_network', broken_network)
    output = tmp_path / 'model'
    with pytest.raises(SystemExit) as error:
        nn_training.main([str(corpus), '--device', 'cpu', '--epochs', '1', '--model-kib', '4',
                          '--context-bytes', '2', '--batch-size', '16', '--output', str(output)])
    assert error.value.code == 2
    assert 'Nonfinite training loss' in capsys.readouterr().err
    assert not (output / 'model.mcnn').exists()
