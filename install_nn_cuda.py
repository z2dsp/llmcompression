"""Install the official Windows Python 3.11 CUDA wheel in project-local storage.

Large buffered transfers avoid small network-share writes. Existing system
PyTorch supplies dependencies and is left untouched. No executable installer runs.
"""
import hashlib
import html
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile


def main():
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 11):
        raise SystemExit('Run with Windows Python 3.11.')
    root = Path(__file__).resolve().parent
    wheel_name = 'torch-2.5.1+cu124-cp311-cp311-win_amd64.whl'
    index = 'https://download.pytorch.org/whl/cu124/torch/'
    with urllib.request.urlopen(index, timeout=60) as response:
        page = response.read().decode()
    links = [html.unescape(link) for link in re.findall(r'href="([^"]+)"', page)]
    link = next(link for link in links if urllib.parse.unquote(urllib.parse.urlsplit(link).path).endswith('/' + wheel_name))
    url = urllib.parse.urljoin(index, link)
    expected = urllib.parse.parse_qs(urllib.parse.urlsplit(url).fragment)['sha256'][0]
    wheels = root / '.cache' / 'cuda-wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    wheel = wheels / wheel_name
    candidate = wheel
    if not wheel.exists():
        candidate = wheel.with_suffix('.partial')
        count = candidate.stat().st_size if candidate.exists() else 0
        request = urllib.request.Request(url.split('#')[0], headers={'Range': f'bytes={count}-'} if count else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            if count and response.status == 206:
                if not response.headers.get('Content-Range', '').startswith(f'bytes {count}-'):
                    raise SystemExit('Server returned an incorrect download range.')
            elif count:
                count = 0  # Server ignored Range; replace the partial download.
            last = time.monotonic()
            print(f'Downloading from {count / 1048576:.0f} MiB', flush=True)
            with candidate.open('ab' if count else 'wb', buffering=16 * 1024 * 1024) as output:
                while True:
                    chunk = response.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    count += len(chunk)
                    if time.monotonic() - last >= 10:
                        print(f'Downloaded {count / 1048576:.0f} MiB', flush=True)
                        last = time.monotonic()
    print('Verifying official wheel SHA-256', flush=True)
    digest = hashlib.sha256()
    with candidate.open('rb') as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit('Wheel SHA-256 mismatch; runtime was not installed.')
    if candidate != wheel:
        candidate.replace(wheel)
    target = (root / '.cache' / 'cuda-runtime').resolve()
    extract_wheel(wheel, target)
    print(f'CUDA PyTorch installed in {target}. Run train_nn_cuda.py with this Python.', flush=True)


def extract_wheel(wheel, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        entries = [(info, target / info.filename) for info in archive.infolist()]
        if any(info.filename.startswith(('/', '\\')) or ':' in info.filename
               or '..' in info.filename.replace('\\', '/').split('/')
               or (info.external_attr >> 16) & 0o170000 == 0o120000
               for info, _ in entries):
            raise SystemExit('Wheel path escapes installation directory.')
        last = time.monotonic()
        for info, path in entries:
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, path.open('wb', buffering=16 * 1024 * 1024) as output:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
            if time.monotonic() - last >= 10:
                print(f'Extracting {info.filename}', flush=True)
                last = time.monotonic()


if __name__ == '__main__':
    main()

