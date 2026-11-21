import math
import random

import pytest

from llm_compression.arithmetic import ArithmeticDecoder, ArithmeticEncoder


def round_trip(symbols, cdfs):
    encoder = ArithmeticEncoder()
    for symbol, cdf in zip(symbols, cdfs):
        encoder.write(cdf, symbol)
    payload = encoder.finish()
    decoder = ArithmeticDecoder(payload)
    assert [decoder.read(cdf) for cdf in cdfs] == symbols
    return payload


@pytest.mark.parametrize("seed", range(10))
def test_changing_distributions(seed):
    randomizer = random.Random(seed)
    cdfs, symbols = [], []
    for _ in range(500):
        cdf = [0]
        for _ in range(randomizer.randint(2, 50)):
            cdf.append(cdf[-1] + randomizer.randint(1, 10000))
        symbols.append(randomizer.randrange(len(cdf) - 1))
        cdfs.append(cdf)
    round_trip(symbols, cdfs)


@pytest.mark.parametrize("symbols", [[], [0], [1], [0] * 1000, [1] * 1000, [0, 1] * 1000])
def test_extreme_frequencies(symbols):
    round_trip(symbols, [[0, 1, 1 << 24]] * len(symbols))


def test_single_symbol_alphabet():
    payload = round_trip([0] * 1000, [[0, 100]] * 1000)
    assert len(payload) == 1


def test_coding_length_matches_information():
    symbols = [0] * 900 + [1] * 100
    random.Random(42).shuffle(symbols)
    payload = round_trip(symbols, [[0, 9, 10]] * len(symbols))
    ideal_bits = -900 * math.log2(0.9) - 100 * math.log2(0.1)
    assert abs(len(payload) * 8 - ideal_bits) < 10


def test_rejects_invalid_symbols_and_reuse():
    encoder = ArithmeticEncoder()
    with pytest.raises(ValueError):
        encoder.write([0, 1, 2], 2)
    with pytest.raises(ValueError):
        encoder.write([0, 0, 2], 0)
    encoder.finish()
    with pytest.raises(ValueError):
        encoder.write([0, 1], 0)
    with pytest.raises(ValueError):
        encoder.finish()
