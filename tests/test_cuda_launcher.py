import sys

import pytest

import train_nn_cuda
from llm_compression import nn_training


@pytest.mark.parametrize('platform', ['linux', 'win32'])
def test_launcher_selects_cached_windows_runtime_only_on_windows(tmp_path, monkeypatch, platform):
    runtime = tmp_path / '.cache' / 'cuda-runtime'
    (runtime / 'torch').mkdir(parents=True)
    (runtime / 'torch' / '__init__.py').touch()
    monkeypatch.setattr(train_nn_cuda, '__file__', str(tmp_path / 'train_nn_cuda.py'))
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setattr(sys, 'version_info', (3, 11, 0))
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    captured = []
    monkeypatch.setattr(nn_training, 'main', lambda args: captured.append(args) or 0)

    assert train_nn_cuda.main(['messages.jsonl', '--epochs', '1']) == 0
    assert captured == [['--device', 'cuda', 'messages.jsonl', '--epochs', '1']]
    assert (str(runtime) in sys.path) == (platform == 'win32')


def test_cached_windows_runtime_requires_matching_python(tmp_path, monkeypatch):
    (tmp_path / '.cache' / 'cuda-runtime' / 'torch').mkdir(parents=True)
    (tmp_path / '.cache' / 'cuda-runtime' / 'torch' / '__init__.py').touch()
    monkeypatch.setattr(train_nn_cuda, '__file__', str(tmp_path / 'train_nn_cuda.py'))
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(sys, 'version_info', (3, 12, 0))
    with pytest.raises(SystemExit, match='Windows Python 3.11'):
        train_nn_cuda.main([])
