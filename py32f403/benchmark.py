#!/usr/bin/env python3
"""Benchmark PY32 firmware through the Pico W's USB-to-USART bridge."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pico'))
from run_benchmarks import RawRepl

BRIDGE = '''
from machine import UART,Pin
import time,binascii,gc
py32_uart=UART(0,115200,tx=Pin(0),rx=Pin(1),rxbuf=8192,timeout=10)
def py32_request(data):
 while py32_uart.any(): py32_uart.read()
 py32_uart.write(data+b'\\n')
 received=bytearray()
 started=time.ticks_ms()
 while time.ticks_diff(time.ticks_ms(),started)<10000:
  if py32_uart.any():
   received.extend(py32_uart.read(py32_uart.any()))
   if b'\\n' in received: return received
  else: time.sleep_ms(1)
 raise RuntimeError('PY32 UART timeout')
'''


def request(device, command):
    output = device.execute('print(binascii.hexlify(py32_request(%r)).decode())' % command.encode(), timeout=15)
    response = bytes.fromhex(output.strip()).decode().strip()
    if response.startswith('ERR'):
        raise RuntimeError('Target rejected request: ' + response)
    return response.split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--build', type=Path, default=HERE / 'build')
    parser.add_argument('--vectors', type=Path, default=HERE.parent / 'models/pico-32k-context3/vectors.json')
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 1000:
        parser.error('--iterations must be between 1 and 1000')
    vectors = json.loads(args.vectors.read_text())
    manifest = json.loads((args.build / 'build.json').read_text())
    rows = []
    with RawRepl(args.port) as device:
        device.execute(BRIDGE)
        info = request(device, 'I')
        if info[0] != 'I' or info[6] != manifest['pnn_sha256']:
            raise ValueError('Target firmware/model identity mismatch')
        cpu = int(info[1])
        for item in vectors['cdfs']:
            context = ''.join('%04x' % value for value in item['history'])
            response = request(device, 'P ' + context)
            if response[0] != 'P' or hashlib.sha256(bytes.fromhex(response[2])).hexdigest() != item['sha256']:
                raise ValueError('PY32/desktop CDF mismatch')
        print('All 32 desktop CDF checks passed.', flush=True)
        timing = request(device, 'B ' + str(args.iterations))
        total_cycles = int(timing[2])
        for group, cases in [('checks', vectors['checks']), ('sample', vectors['cases'])]:
            for item in cases:
                original = bytes.fromhex(item['message_hex'])
                expected = bytes.fromhex(item['payload_hex'])
                response = request(device, 'E ' + original.hex())
                encoded = bytes.fromhex(response[3]) if len(response) > 3 else b''
                if response[0] != 'E' or encoded != expected or len(encoded) != int(response[2]):
                    raise ValueError('PY32/desktop payload mismatch')
                encode_cycles = int(response[1])
                response = request(device, 'D %d %s' % (len(original), expected.hex()))
                restored = bytes.fromhex(response[3]) if len(response) > 3 else b''
                if response[0] != 'D' or restored != original or int(response[2]) != len(original):
                    raise ValueError('PY32/desktop decode mismatch')
                if group == 'sample':
                    rows.append({'original_bytes': len(original), 'payload_bytes': len(encoded),
                                 'encode_cycles': encode_cycles, 'decode_cycles': int(response[1])})
            print('Checked', len(cases), group, 'messages.', flush=True)
        final_info = request(device, 'I')
    original_bytes = sum(row['original_bytes'] for row in rows)
    encoded_bytes = sum(row['payload_bytes'] for row in rows)
    enc_cycles = sum(row['encode_cycles'] for row in rows)
    dec_cycles = sum(row['decode_cycles'] for row in rows)
    report = {
        'target': 'PY32F403V1DT6', 'cpu_hz': cpu, 'kernel': info[7],
        'pnn_sha256': info[6], 'model_bytes': int(info[5]),
        'memory': {'inference_state_bytes': int(info[2]), 'static_ram_bytes': int(info[3]),
                   'conservative_stack_high_water_bytes': int(final_info[4]),
                   'ram_capacity_bytes': 65536, 'linker_stack_budget_bytes': 8192},
        'prediction': {'iterations': int(timing[1]), 'cycles_per_byte': total_cycles / int(timing[1]),
                       'microseconds_per_byte': total_cycles * 1000000 / cpu / int(timing[1])},
        'sample': {'messages': len(rows), 'original_bytes': original_bytes,
                   'payload_bytes': encoded_bytes, 'bits_per_original_byte': 8 * encoded_bytes / original_bytes,
                   'encode_cycles': enc_cycles, 'decode_cycles': dec_cycles,
                   'encode_bytes_per_second': original_bytes * cpu / enc_cycles,
                   'decode_bytes_per_second': original_bytes * cpu / dec_cycles,
                   'cdf_contexts_match_desktop': len(vectors['cdfs']),
                   'extra_correctness_vectors': len(vectors['checks']),
                   'all_payloads_match_desktop': True, 'all_round_trips_exact': True, 'rows': rows},
        'notes': ['DWT hardware cycle-counter timings exclude UART transfer and host bridge latency.',
                  '144 MHz clock uses nominal internal HSI 8 MHz x18 PLL; oscillator accuracy is not independently measured.',
                  'Model and lookup tables stay in internal flash; no heap allocation.',
                  'Sample may overlap training; compression payload excludes external length/checksum/radio framing.'],
    }
    for name in ('nn.c', 'codec.c', 'main.c', 'platform.c', 'startup.c'):
        report.setdefault('source_sha256', {})[name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    report['firmware_sha256'] = hashlib.sha256((args.build / 'nn-py32f403.bin').read_bytes()).hexdigest()
    out = args.build / 'benchmark-results.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    aggregate = {**report, 'sample': {k:v for k,v in report['sample'].items() if k != 'rows'}}
    print(json.dumps(aggregate, indent=2))
    print('Saved:', out)


if __name__ == '__main__':
    main()
