#!/usr/bin/env python3
"""Deploy to an existing MicroPython Pico via USB raw REPL and run benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import serial

DEVICE_DIR = '/llm_pico_demo'


class RawRepl:
    def __init__(self, port):
        self.serial = serial.Serial(port, 115200, timeout=0.1, write_timeout=5, exclusive=True)

    def until(self, marker, timeout=10):
        result = bytearray()
        line = bytearray()
        deadline = time.monotonic() + timeout
        while not result.endswith(marker):
            if time.monotonic() > deadline:
                raise TimeoutError('Pico did not respond before timeout')
            chunk = self.serial.read(1)
            result.extend(chunk)
            line.extend(chunk)
            if chunk == b'\n':
                if line.startswith(b'# '):
                    print(line.decode().strip(), flush=True)
                line = bytearray()
        return bytes(result[:-len(marker)])

    def __enter__(self):
        self.serial.write(b'\r\x03\x03')
        time.sleep(0.2)
        self.serial.reset_input_buffer()
        self.serial.write(b'\r\x01')
        self.until(b'raw REPL; CTRL-B to exit\r\n>')
        return self

    def __exit__(self, *args):
        if args[0] is not None:
            self.serial.write(b'\x03')
        self.serial.write(b'\x02')
        self.serial.close()

    def execute(self, code, timeout=30):
        data = code.encode()
        for offset in range(0, len(data), 256):
            self.serial.write(data[offset:offset + 256])
            time.sleep(0.003)
        self.serial.write(b'\x04')
        if self.serial.read(2) != b'OK':
            raise RuntimeError('Raw REPL did not accept command')
        stdout = self.until(b'\x04', timeout)
        stderr = self.until(b'\x04', timeout)
        self.until(b'>', timeout)
        if stderr:
            raise RuntimeError(stderr.decode(errors='replace'))
        return stdout.decode()

    def upload(self, local, remote):
        content = Path(local).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        code = ("import hashlib,binascii\n"
                "try:\n"
                " f=open(%r,'rb'); d=hashlib.sha256()\n"
                " while True:\n"
                "  b=f.read(1024)\n"
                "  if not b: break\n"
                "  d.update(b)\n"
                " f.close();print(binascii.hexlify(d.digest()).decode())\n"
                "except OSError: print('MISSING')" % remote)
        if self.execute(code).strip() == digest:
            print('Already current:', remote, flush=True)
            return
        temporary = remote + '.uploading'
        self.execute('f=open(%r,"wb")' % temporary)
        for offset in range(0, len(content), 512):
            self.execute('f.write(bytes.fromhex(%r))' % content[offset:offset + 512].hex())
        self.execute('f.close()')
        actual = self.execute(code.replace(repr(remote), repr(temporary))).strip()
        if actual != digest:
            raise RuntimeError('Upload checksum mismatch: ' + remote)
        self.execute('import os;os.rename(%r,%r)' % (temporary, remote))
        print('Uploaded:', remote, len(content), 'bytes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--assets', type=Path, default=Path('models/pico-32k-context3'))
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--soft-reset', action='store_true',
                        help='reset the MicroPython interpreter before deployment for a clean heap')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    base = Path(__file__).resolve().parent
    with RawRepl(args.port) as device:
        if args.soft_reset:
            device.serial.write(b'\x04')
            device.until(b'raw REPL; CTRL-B to exit\r\n>', timeout=10)
        print(device.execute("import os,gc,sys,machine;gc.collect();print(sys.implementation);print('CPU',machine.freq(),'RAM',gc.mem_free(),'FLASH',os.statvfs('/'))"), flush=True)
        device.execute("import os\ntry: os.mkdir(%r)\nexcept OSError: pass" % DEVICE_DIR)
        for name in ('nn_runtime.py', 'nn_codec.py', 'benchmark.py'):
            device.upload(base / name, DEVICE_DIR + '/' + name)
        for name in ('model.pnn', 'vectors.json'):
            device.upload(args.assets / name, DEVICE_DIR + '/' + name)
        # Remove our prior in-memory modules only. Preserve unrelated files/apps.
        device.execute("import sys,gc\nglobals().pop('benchmark',None)\nfor name in ('benchmark','nn_codec','nn_runtime'):\n if name in sys.modules: del sys.modules[name]\ngc.collect()\nif %r not in sys.path: sys.path.insert(0,%r)" % (DEVICE_DIR, DEVICE_DIR))
        print('Running device benchmarks...', flush=True)
        output = device.execute("import benchmark,json;print(json.dumps(benchmark.run(%r,%d)))" % (DEVICE_DIR, args.iterations), timeout=args.timeout)
        result = json.loads(output.strip().splitlines()[-1])
    result['model_manifest'] = json.loads((args.assets / 'manifest.json').read_text())
    for name in ('nn_runtime.py', 'nn_codec.py', 'benchmark.py'):
        result.setdefault('source_sha256', {})[name] = hashlib.sha256((base / name).read_bytes()).hexdigest()
    path = args.assets / 'pico-benchmark.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('sample','model_manifest')}, indent=2))
    print('Sample:', json.dumps({k:v for k,v in result['sample'].items() if k!='rows'}, indent=2))
    print('Saved:', path)


if __name__ == '__main__':
    main()
