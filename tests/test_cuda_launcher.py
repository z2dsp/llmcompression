import sys

import pytest

import train_nn_cuda
from llm_compression import nn_training


@pytest.mark.parametrize('platform', ['linux', 'win32'])
def test_launcher_selects_cached_windows_runtime_only_on_windows(tmp_path, monkeypatch, platform):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    runtime = tmp_path / 'llm-compression' / 'cuda-2.5.1-cu124-py311'
    (runtime / 'torch').mkdir(parents=True)
    (runtime / 'torch' / '__init__.py').touch()
    (runtime / '.ready').touch()
    monkeypatch.setattr(train_nn_cuda, '__file__', str(tmp_path / 'train_nn_cuda.py'))
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setattr(sys, 'version_info', (3, 11, 0))
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    captured = []
    monkeypatch.setattr(nn_training, 'main', lambda args, **kwargs: captured.append((args, kwargs)) or 0)

    assert train_nn_cuda.main(['messages.jsonl', '--epochs', '1']) == 0
    assert captured == [(['--device', 'cuda', 'messages.jsonl', '--epochs', '1'], {'cuda_defaults': True})]
    assert (str(runtime) in sys.path) == (platform == 'win32')


def test_cached_windows_runtime_requires_matching_python(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    runtime = tmp_path / 'llm-compression' / 'cuda-2.5.1-cu124-py311'
    (runtime / 'torch').mkdir(parents=True)
    (runtime / 'torch' / '__init__.py').touch()
    (runtime / '.ready').touch()
    monkeypatch.setattr(train_nn_cuda, '__file__', str(tmp_path / 'train_nn_cuda.py'))
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(sys, 'version_info', (3, 12, 0))
    with pytest.raises(SystemExit, match='Windows Python 3.11'):
        train_nn_cuda.main([])


def test_wrong_windows_python_relaunches_preserving_arguments(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    runtime = tmp_path / 'llm-compression' / 'cuda-2.5.1-cu124-py311'
    (runtime / 'torch').mkdir(parents=True)
    (runtime / 'torch' / '__init__.py').touch()
    (runtime / '.ready').touch()
    python = tmp_path / 'Programs/Python/Python311/python.exe'
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(sys, 'version_info', (3, 10, 0))
    calls = []
    monkeypatch.setattr(train_nn_cuda.subprocess, 'run', lambda command, **kwargs:
                        calls.append((command, kwargs)) or SimpleNamespace(returncode=7))
    assert train_nn_cuda.main(['messages with spaces.jsonl', '--epochs', '1']) == 7
    assert calls[0][0] == [str(python), '-u', str(train_nn_cuda.Path(train_nn_cuda.__file__).resolve()),
                          'messages with spaces.jsonl', '--epochs', '1']
    assert calls[0][1] == {'check': False}


def test_incomplete_local_runtime_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    runtime = tmp_path / 'llm-compression' / 'cuda-2.5.1-cu124-py311'
    (runtime / 'torch').mkdir(parents=True)
    (runtime / 'torch' / '__init__.py').touch()
    monkeypatch.setattr(sys, 'platform', 'win32')
    with pytest.raises(SystemExit, match='install_nn_cuda.py'):
        train_nn_cuda.main([])

def test_sweep_launcher_relaunches_before_loading_numpy(tmp_path, monkeypatch):
    from pathlib import Path
    import sweep_nn_cuda
    from types import SimpleNamespace
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    marker=tmp_path/'llm-compression/cuda-2.5.1-cu124-py311/.ready'
    marker.parent.mkdir(parents=True); marker.touch()
    python=tmp_path/'Programs/Python/Python311/python.exe'
    python.parent.mkdir(parents=True); python.touch()
    monkeypatch.setattr(sys,'platform','win32')
    monkeypatch.setattr(sys,'version_info',(3,10,0))
    calls=[]
    monkeypatch.setattr(sweep_nn_cuda.subprocess,'run',lambda command,**kwargs:
                        calls.append(command) or SimpleNamespace(returncode=4))
    assert sweep_nn_cuda.main(['messages with spaces.jsonl','--sizes-kib','4'])==4
    assert calls[0]==[str(python),'-u',str(Path(sweep_nn_cuda.__file__).resolve()),
                      'messages with spaces.jsonl','--sizes-kib','4']
