import io

import numpy as np
import pytest

from llm_compression import nn_generation
from llm_compression.nn_generation import generate_bytes, greedy_bytes, main
from llm_compression.nn_inspection import inspect_position
from llm_compression.nn_model import NeuralModel, export_model, shapes


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('storage', ['int8', 'float32'])
@pytest.mark.parametrize('context', [0, 3])
def test_generation_matches_exported_model_scores_and_repeats_exactly(tmp_path, depth, storage, context):
    rng = np.random.default_rng(81)
    arrays = [rng.normal(0, 0.2, shape).astype('float32') for shape in shapes(context, 2, 5, depth)]
    path = tmp_path / 'model.mcnn'
    export_model(path, arrays, context, 2, 5, storage, depth)
    model = NeuralModel(path)
    prompt = 'Mesh café: '.encode('utf-8')
    generated = bytes(greedy_bytes(model, prompt, 20))
    assert len(generated) == 20
    assert bytes(greedy_bytes(model, prompt, 20)) == generated
    text = prompt
    for byte in generated:
        trace = inspect_position(model, text, len(text))
        assert byte == int(np.argmax(trace.logits))
        text += bytes([byte])


class FollowingModel:
    context = 3

    def __init__(self):
        self.histories = []

    def logits(self, history):
        self.histories.append(list(history))
        scores = np.zeros(256)
        # Alternate the two bytes in UTF-8 é, using the generated history.
        scores[0xA9 if history[-1] == 0xC3 else 0xC3] = 1
        return scores


def test_bos_prompt_truncation_and_generated_byte_feedback():
    model = FollowingModel()
    assert bytes(greedy_bytes(model, b'A', 3)) == b'\xc3\xa9\xc3'
    assert model.histories == [[256, 256, 65], [256, 65, 195], [65, 195, 169]]
    model = FollowingModel()
    bytes(greedy_bytes(model, 'café'.encode(), 1))
    assert model.histories == [list('café'.encode()[-3:])]
    model = FollowingModel()
    bytes(greedy_bytes(model, b'', 1))
    assert model.histories == [[256, 256, 256]]


def test_raw_logit_argmax_is_used_even_when_codec_quantization_ties(tmp_path):
    arrays = [np.zeros(shape, dtype='float32') for shape in shapes(0, 1, 1)]
    arrays[-1][220] = 1e-8
    path = tmp_path / 'model.mcnn'
    export_model(path, arrays, 0, 1, 1, 'float32')
    model = NeuralModel(path)
    assert int(np.argmax(np.diff(model.cdf([])))) == 0
    assert bytes(greedy_bytes(model, b'', 3)) == bytes([220]) * 3
    arrays[-1][:] = 0
    export_model(path, arrays, 0, 1, 1, 'float32')
    assert bytes(greedy_bytes(NeuralModel(path), b'', 3)) == b'\x00' * 3


def test_zero_length_does_not_evaluate_model_and_negative_length_is_rejected():
    assert bytes(greedy_bytes(None, b'prompt', 0)) == b''
    with pytest.raises(ValueError, match='nonnegative'):
        bytes(greedy_bytes(None, b'prompt', -1))


def test_cli_decodes_utf8_incrementally_and_preserves_exact_file_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(nn_generation, 'NeuralModel', lambda _: FollowingModel())
    stdout = io.StringIO()
    monkeypatch.setattr(nn_generation.sys, 'stdout', stdout)
    assert main(['model.mcnn', 'A', '--bytes', '4']) == 0
    assert stdout.getvalue() == 'Aéé'
    stdout.seek(0)
    stdout.truncate()
    assert main(['model.mcnn', 'A', '--bytes', '1', '--continuation-only']) == 0
    assert stdout.getvalue() == '\ufffd'
    output = tmp_path / 'generated.bin'
    assert main(['model.mcnn', 'A', '--bytes', '1', '--output', str(output)]) == 0
    assert output.read_bytes() == b'A\xc3'
    with pytest.raises(SystemExit) as error:
        main(['model.mcnn', 'A', '--bytes', '2', '--output', str(output)])
    assert error.value.code == 2 and output.read_bytes() == b'A\xc3'
    continuation = tmp_path / 'continuation.bin'
    assert main(['model.mcnn', 'A', '--bytes', '2', '--continuation-only', '--output', str(continuation)]) == 0
    assert continuation.read_bytes() == b'\xc3\xa9'


def test_cli_validation_and_model_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / 'missing.mcnn'), '--bytes', '-1'])
    assert error.value.code == 2 and '--bytes must be nonnegative' in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / 'missing.mcnn'), 'Hello'])
    assert error.value.code == 2 and 'Generation failed' in capsys.readouterr().err


class TwoByteModel:
    context = 1

    def logits(self, history):
        scores = np.full(256, -np.inf)
        scores[65], scores[66] = np.log(3), 0
        return scores


def test_temperature_sampling_matches_expected_distribution_and_seed():
    model = TwoByteModel()
    actual = bytes(generate_bytes(model, b'', 512, temperature=0.5, seed=91))
    # At T=0.5 the odds 3:1 become 9:1. Compare to independent categorical draws.
    expected = np.random.default_rng(91).choice([65, 66], size=512, p=[0.9, 0.1])
    assert actual == bytes(expected.tolist())
    assert set(actual) == {65, 66}
    assert actual == bytes(generate_bytes(model, b'', 512, temperature=0.5, seed=91))
    assert actual != bytes(generate_bytes(model, b'', 512, temperature=0.5, seed=92))
    warmer = bytes(generate_bytes(model, b'', 512, temperature=2.0, seed=91))
    assert warmer.count(66) > actual.count(66) + 50


def test_zero_temperature_preserves_greedy_generation_and_sampling_feeds_back():
    assert bytes(generate_bytes(TwoByteModel(), b'', 50, temperature=0, seed=99)) == b'A' * 50
    assert bytes(generate_bytes(FollowingModel(), b'A', 20, temperature=0)) == bytes(greedy_bytes(FollowingModel(), b'A', 20))
    model = FollowingModel()
    sampled = bytes(generate_bytes(model, b'A', 20, temperature=0.8, seed=10))
    for position, history in enumerate(model.histories):
        assert history == ([256] * 3 + list(b'A' + sampled[:position]))[-3:]


@pytest.mark.parametrize('temperature', [1e-320, 1e308])
def test_extreme_positive_temperatures_are_numerically_stable(temperature):
    with np.errstate(all='raise'):
        result = bytes(generate_bytes(TwoByteModel(), b'', 50, temperature=temperature, seed=10))
    assert len(result) == 50 and set(result) <= {65, 66}


@pytest.mark.parametrize('temperature', [-0.1, float('nan'), float('inf'), -float('inf')])
def test_invalid_temperatures_rejected_before_model_loading(tmp_path, temperature, capsys):
    with pytest.raises(ValueError, match='Temperature'):
        bytes(generate_bytes(None, b'', 0, temperature=temperature))
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / 'missing.mcnn'), f'--temperature={temperature}'])
    assert error.value.code == 2 and '--temperature must be finite and nonnegative' in capsys.readouterr().err


def test_cli_forwards_temperature_and_seed_to_sampling(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nn_generation, 'NeuralModel', lambda _: TwoByteModel())
    expected = bytes(generate_bytes(TwoByteModel(), b'Prompt: ', 40, temperature=1.2, seed=17))
    assert main(['model.mcnn', 'Prompt: ', '--bytes', '40', '--temperature', '1.2', '--seed', '17']) == 0
    assert capsys.readouterr().out == 'Prompt: ' + expected.decode('ascii')
    output = tmp_path / 'sample.bin'
    assert main(['model.mcnn', 'Prompt: ', '--bytes', '40', '--temperature', '1.2', '--seed', '17',
                 '--output', str(output), '--continuation-only']) == 0
    assert output.read_bytes() == expected
    with pytest.raises(SystemExit) as error:
        main(['model.mcnn', '--seed', '-1'])
    assert error.value.code == 2 and '--seed must be nonnegative' in capsys.readouterr().err
    with pytest.raises(ValueError, match='Seed'):
        bytes(generate_bytes(None, b'', 0, seed=-1))
