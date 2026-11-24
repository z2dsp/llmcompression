"""Generate bytes from a trained .mcnn model with optional temperature sampling."""
import argparse
import codecs
import math
from pathlib import Path
import sys

import numpy as np

from .nn_model import NeuralModel, advance


def greedy_bytes(model, initial_context: bytes, length: int):
    """Original greedy API; ties choose the lower byte value."""
    return generate_bytes(model, initial_context, length)


def generate_bytes(model, initial_context: bytes, length: int, *, temperature=0.0, seed=None):
    """Yield length bytes, using greedy selection or softmax(logits/temperature).

    Prompt bytes older than the model's fixed context window are not used for
    prediction. Short prompts are left-padded with BOS, just as in the codec.
    Temperature zero preserves greedy selection. Positive values sample with a
    local RNG; a seed reproduces the same choices. No repetition penalty, UTF-8
    filtering, or EOS token is applied.
    """
    if not isinstance(length, int) or length < 0:
        raise ValueError('Generation length must be a nonnegative number of bytes')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Temperature must be finite and nonnegative')
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ValueError('Seed must be a nonnegative integer')
    if length == 0:
        return
    history = [256] * model.context
    if model.context:
        tail = initial_context[-model.context:]
        if tail:
            history[-len(tail):] = tail
    rng = np.random.default_rng(seed) if temperature > 0 else None
    for _ in range(length):
        logits = model.logits(history)
        if temperature == 0:
            byte = int(np.argmax(logits))
        else:
            # Subtract before dividing so even tiny positive temperatures are
            # stable. Large negative scores may become -inf, giving zero mass.
            with np.errstate(over='ignore', under='ignore'):
                weights = np.exp((logits - np.max(logits)) / temperature)
            probabilities = weights / weights.sum()
            byte = int(rng.choice(256, p=probabilities))
        advance(history, byte)
        yield byte


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path, help='trained model.mcnn (one, two, or three hidden layers)')
    parser.add_argument('context', nargs='?', default='', help='initial text context; quote spaces (default: empty/BOS)')
    parser.add_argument('--bytes', dest='length', type=int, default=256,
                        help='number of new bytes to generate (default: 256; excludes the initial context)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='0 selects the most likely byte; positive values sample (e.g. 0.8); default: 0')
    parser.add_argument('--seed', type=int, help='nonnegative random seed for reproducible sampling')
    parser.add_argument('--continuation-only', action='store_true', help='omit the initial context from the output')
    parser.add_argument('--output', type=Path, help='save exact bytes to a new file instead of printing text')
    args = parser.parse_args(argv)
    if args.length < 0:
        parser.error('--bytes must be nonnegative')
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error('--temperature must be finite and nonnegative')
    if args.seed is not None and args.seed < 0:
        parser.error('--seed must be nonnegative')
    try:
        model = NeuralModel(args.model)
        prompt = args.context.encode('utf-8')
        initial_output = b'' if args.continuation_only else prompt
        generated = generate_bytes(model, prompt, args.length, temperature=args.temperature, seed=args.seed)
        if args.output is not None:
            with args.output.open('xb') as stream:
                stream.write(initial_output)
                for byte in generated:
                    stream.write(bytes([byte]))
        else:
            # Decode incrementally: a UTF-8 character can span several greedy
            # byte choices. Replacement is for display only, never fed back.
            decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            sys.stdout.write(decoder.decode(initial_output))
            for byte in generated:
                sys.stdout.write(decoder.decode(bytes([byte])))
                sys.stdout.flush()
            sys.stdout.write(decoder.decode(b'', final=True))
            sys.stdout.flush()
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(2, f'Generation failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
