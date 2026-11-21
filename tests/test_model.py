import os

import numpy as np
import pytest

from llm_compression.codec import compress, decompress
from llm_compression.model import HuggingFacePredictor, TOTAL, probabilities_to_cdf


@pytest.mark.parametrize("probs", [[0.1, 0.2, 0.7], [1.0], [0, 1, 0], [1e-300, 1, 1e-300], [1] * 49152])
def test_quantization(probs):
    cdf = probabilities_to_cdf(probs)
    assert cdf[0] == 0
    assert cdf[-1] == TOTAL
    assert all(a < b for a, b in zip(cdf, cdf[1:]))
    assert len(cdf) == len(probs) + 1
    assert cdf == probabilities_to_cdf(probs)


@pytest.mark.parametrize("probs", [[], [0, 0], [-0.1, 1.1], [float("nan"), 1], [float("inf"), 1], [[1, 2]]])
def test_invalid_probabilities(probs):
    with pytest.raises(ValueError):
        probabilities_to_cdf(probs)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run the real model")
def test_real_model_round_trips():
    encoder = HuggingFacePredictor(context_tokens=8)
    decoder = HuggingFacePredictor(context_tokens=8)
    assert sum(p.numel() for p in encoder.model.parameters()) < 1_000_000_000
    for text in [
        "", "Hello, world!", "\ufeffLine one\r\n\tLine two\n  \n",
        "Café — 日本語 🌊 e\u0301\x00\n", "<|endoftext|>\n",
        "The weather was warm, and the sky was blue.\n" * 5,
    ]:
        data = text.encode("utf-8")
        result = compress(data, encoder)
        assert decompress(result.archive, decoder) == data
        assert compress(data, encoder).archive == result.archive


def test_cdf_preserves_probabilities_with_a_small_rounding_error():
    probabilities = np.array([0.01, 0.04, 0.2, 0.75])
    frequencies = np.diff(probabilities_to_cdf(probabilities))
    expected = probabilities * (TOTAL - len(probabilities)) + 1
    assert np.max(np.abs(frequencies - expected)) <= 1
