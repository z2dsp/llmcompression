"""Architecture, storage budgets, and compatibility of neural byte models."""
import numpy as np
import pytest

from llm_compression.nn_model import NeuralModel, TOTAL, export_model, layout, model_size, shapes
from llm_compression.nn_training import build_parser, make_network


@pytest.mark.parametrize('storage', ['int8', 'float32'])
@pytest.mark.parametrize('context', [0, 3, 257, 10000])
@pytest.mark.parametrize('budget', [32, 64, 1024, 65537, 2**40])
@pytest.mark.parametrize('hidden_layers', [1, 2, 3])
def test_budget_fits_largest_equal_width_without_allocating_weights(storage, context, budget, hidden_layers):
    if model_size(context, 8, 1, storage, hidden_layers) > budget * 1024:
        with pytest.raises(ValueError, match='Budget too small'):
            layout(budget, context, 8, storage, hidden_layers)
        return
    hidden, size = layout(budget, context, 8, storage, hidden_layers)
    assert hidden >= 1
    assert size <= budget * 1024 < model_size(context, 8, hidden + 1, storage, hidden_layers)
    tensors = shapes(context, 8, hidden, hidden_layers)
    expected = 60 + sum(np.prod(shape, dtype=object) for shape in tensors) * (1 if storage == 'int8' else 4)
    expected += 4 * len(tensors) if storage == 'int8' else 0
    assert size == expected


@pytest.mark.parametrize('hidden_layers', [1, 2, 3])
def test_network_forward_and_gradients_use_every_dense_layer(hidden_layers):
    torch = pytest.importorskip('torch')
    torch.manual_seed(12)
    network = make_network(torch, 3, 2, 5, hidden_layers)
    history = torch.tensor([[256, 10, 11], [10, 11, 12]])
    logits = network(history)
    assert sum(isinstance(layer, torch.nn.Linear) for layer in network.modules()) == hidden_layers + 1
    torch.nn.functional.cross_entropy(logits, torch.tensor([12, 13])).backward()
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in network.parameters())


@pytest.mark.parametrize('hidden_layers', [1, 2, 3])
def test_exported_predictor_matches_independent_dense_computation(tmp_path, hidden_layers):
    rng = np.random.default_rng(31)
    arrays = [rng.normal(0, 0.3, shape).astype('float32') for shape in shapes(3, 2, 5, hidden_layers)]
    path = tmp_path / 'model.mcnn'
    export_model(path, arrays, 3, 2, 5, 'float32', hidden_layers)
    model = NeuralModel(path)
    history = [256, 97, 98]
    arrays = [a.astype('float64') for a in arrays]
    x = np.tanh(arrays[0][history].reshape(-1) @ arrays[1] + arrays[2])
    if hidden_layers >= 2:
        x = np.tanh(x @ arrays[3] + arrays[4])
    if hidden_layers == 3:
        x = np.tanh(x @ arrays[5] + arrays[6])
    logits = x @ arrays[-2] + arrays[-1]
    probabilities = np.exp(logits - logits.max())
    scaled = probabilities / probabilities.sum() * (TOTAL - 256)
    counts = np.floor(scaled).astype('int64') + 1
    for symbol in sorted(range(256), key=lambda b: (-(scaled[b] % 1), b))[:TOTAL - int(counts.sum())]:
        counts[symbol] += 1
    np.testing.assert_array_equal(np.diff(model.cdf(history)), counts)


def test_default_is_three_hidden_layers_and_invalid_depth_is_rejected():
    assert build_parser().parse_args(['messages.jsonl']).hidden_layers == 3
    assert build_parser().parse_args(['messages.jsonl', '--hidden-layers', '1']).hidden_layers == 1
    with pytest.raises(SystemExit):
        build_parser().parse_args(['messages.jsonl', '--hidden-layers', '4'])
    with pytest.raises(ValueError, match='one, two, or three'):
        layout(32, 3, 8, 'int8', 4)


@pytest.mark.parametrize('budget', [32, 64, 1024])
@pytest.mark.parametrize('storage', ['int8', 'float32'])
def test_three_layer_files_obey_actual_storage_budget(tmp_path, budget, storage):
    hidden, planned_size = layout(budget, 3, 8, storage)
    path = tmp_path / 'model.mcnn'
    arrays = [np.zeros(shape, dtype='float32') for shape in shapes(3, 8, hidden)]
    export_model(path, arrays, 3, 8, hidden, storage)
    model = NeuralModel(path)
    assert path.stat().st_size == model.size_bytes == planned_size <= budget * 1024
    assert model.hidden_layers == 3 and len(model.arrays) == 9


@pytest.mark.parametrize('hidden_layers', [1, 2, 3])
def test_export_rejects_wrong_tensor_count_and_middle_shape(tmp_path, hidden_layers):
    arrays = [np.zeros(shape, dtype='float32') for shape in shapes(3, 2, 5, hidden_layers)]
    path = tmp_path / 'bad.mcnn'
    with pytest.raises(ValueError, match='tensors'):
        export_model(path, arrays[:-1], 3, 2, 5, 'int8', hidden_layers)
    arrays[3] = np.zeros((1, 1), dtype='float32')
    with pytest.raises(ValueError, match='weights'):
        export_model(path, arrays, 3, 2, 5, 'int8', hidden_layers)
    assert not path.exists()
