#!/usr/bin/env python3
"""Sweep neural compression model budgets and contexts with CUDA training."""
from pathlib import Path
import os
import subprocess
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if sys.platform == 'win32' and sys.version_info[:2] != (3, 11):
        local = Path(os.environ.get('LOCALAPPDATA', Path.home()))
        runtime = local / 'llm-compression/cuda-2.5.1-cu124-py311/.ready'
        python = local / 'Programs/Python/Python311/python.exe'
        if runtime.is_file() and python.is_file():
            print(f'Using CUDA sweep Python: {python}', flush=True)
            return subprocess.run([str(python), '-u', str(Path(__file__).resolve()), *arguments], check=False).returncode
    if sys.platform == 'win32':
        os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache' / 'matplotlib'))
    from llm_compression.nn_sweep import main as sweep_main
    return sweep_main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
