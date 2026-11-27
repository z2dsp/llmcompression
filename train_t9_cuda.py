#!/usr/bin/env python3
"""Launch T9 training with CUDA required.

On Linux, use .venv-cuda/bin/python. On Windows, Python 3.11 can use the
compressed local runtime. Other Windows Python versions relaunch Python 3.11
automatically. All T9 trainer arguments are accepted.
"""
from pathlib import Path
import os
import subprocess
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    runtime = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'llm-compression' / 'cuda-2.5.1-cu124-py311'
    # The shared project can contain Windows DLLs; Linux must use its own venv.
    if sys.platform == 'win32' and (runtime / 'torch' / '__init__.py').is_file():
        if not (runtime / '.ready').is_file():
            raise SystemExit('CUDA runtime installation is incomplete; run install_nn_cuda.py first.')
        if sys.version_info[:2] != (3, 11):
            python311 = Path(os.environ['LOCALAPPDATA']) / 'Programs' / 'Python' / 'Python311' / 'python.exe'
            if not python311.is_file():
                raise SystemExit('The local CUDA build requires Windows Python 3.11; install Python 3.11 or run with its executable.')
            print(f'Using CUDA Python: {python311}', flush=True)
            return subprocess.run([str(python311), '-u', str(Path(__file__).resolve()), *arguments], check=False).returncode
        sys.path.insert(0, str(runtime))
    from llm_compression.t9_training import main as train_main
    return train_main(['--device', 'cuda', *arguments])


if __name__ == '__main__':
    raise SystemExit(main())
