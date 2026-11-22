#!/usr/bin/env python3
"""Launch neural training with CUDA required.

On Linux, use .venv-cuda/bin/python. On Windows, Python 3.11 can use the
project-local wheel cache. All normal neural trainer arguments are accepted.
"""
from pathlib import Path
import sys


def main(argv=None):
    runtime = Path(__file__).resolve().parent / '.cache' / 'cuda-runtime'
    # The shared project can contain Windows DLLs; Linux must use its own venv.
    if sys.platform == 'win32' and (runtime / 'torch' / '__init__.py').is_file():
        if sys.version_info[:2] != (3, 11):
            raise SystemExit('The project-local CUDA build requires Windows Python 3.11.')
        sys.path.insert(0, str(runtime))
    from llm_compression.nn_training import main as train_main
    return train_main(['--device', 'cuda', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    raise SystemExit(main())
