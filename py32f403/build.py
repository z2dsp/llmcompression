#!/usr/bin/env python3
"""Build a flash-resident integer neural codec for PY32F403V1DT6 (64 KiB RAM)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SDK_COMMIT = '982757404a416dd1e0fde921750d61167aed09ee'


def generate_model(source, output):
    data = Path(source).read_bytes()
    if len(data) < 72:
        raise ValueError('Truncated PNN model')
    magic = data[:8]
    if magic not in (b'MCPNN001', b'MCPNN002', b'MCPNN003'):
        raise ValueError('PNN format/checksum mismatch')
    hidden_layers = magic[-1] - 48
    header_size = 72 + 12 * (hidden_layers - 1)
    if len(data) < header_size:
        raise ValueError('Truncated PNN model')
    header = struct.unpack_from('<8s' + str(8 + 3 * (hidden_layers - 1)) + 'I32s', data)
    if hashlib.sha256(data[:header_size - 32] + data[header_size:]).digest() != header[-1]:
        raise ValueError('PNN format/checksum mismatch')
    context, embedding, hidden, mul1, shift1, mul2, shift2, pre2 = header[1:9]
    # Validate the same signed accumulator bounds as the MicroPython runtime.
    max1, max2 = context * embedding * 128 * 128, hidden * 4096 * 128
    if min(context, embedding, hidden) < 1 or not all(0 <= s <= 24 for s in (shift1, shift2, pre2)):
        raise ValueError('Invalid model dimensions/scaling')
    if (max(max1, max2) >= 1 << 30 or not 0 < mul1 < 1 << 30 or not 0 < mul2 < 1 << 30
            or max1 * mul1 >= 1 << 30 or ((max2 + (1 << pre2) - 1) >> pre2) * mul2 >= 1 << 30):
        raise ValueError('Model exceeds signed 32-bit accumulator bounds')
    middle_constants = {}
    for index in range(hidden_layers - 1):
        mul_mid, shift_mid, pre_mid = header[9 + 3 * index:12 + 3 * index]
        if not (0 < mul_mid < 1 << 30 and 0 <= shift_mid <= 24 and 0 <= pre_mid <= 24):
            raise ValueError('Invalid middle layer scaling')
        if ((max2 + (1 << pre_mid) - 1) >> pre_mid) * mul_mid >= 1 << 30:
            raise ValueError('Middle layer exceeds signed 32-bit accumulator bounds')
        suffix = '' if index == 0 else str(index + 1)
        middle_constants.update({f'MUL_MID{suffix}': mul_mid, f'SHIFT_MID{suffix}': shift_mid,
                                 f'PRE_MID{suffix}': pre_mid})
    digest = hashlib.sha256(data).hexdigest()
    constants = dict(CONTEXT=context, EMBEDDING=embedding, HIDDEN=hidden, MUL1=mul1,
                     SHIFT1=shift1, MUL2=mul2, SHIFT2=shift2, PRE2=pre2,
                     HIDDEN_LAYERS=hidden_layers, MODEL_BYTES=len(data), **middle_constants)
    declarations, definitions = [], ['#include "model.h"']
    offset = header_size
    tensors = [('embedding', 'int8_t', 'B', 257 * embedding),
               ('w1', 'int8_t', 'B', context * embedding * hidden),
               ('b1', 'int32_t', 'i', hidden)]
    for index in range(hidden_layers - 1):
        suffix = '' if index == 0 else str(index + 1)
        tensors += [(f'w_mid{suffix}', 'int8_t', 'B', hidden * hidden),
                    (f'b_mid{suffix}', 'int32_t', 'i', hidden)]
    tensors += [('w2', 'int8_t', 'B', hidden * 256),
               ('b2', 'int32_t', 'i', 256), ('tanh', 'int16_t', 'h', 4097),
               ('exp', 'uint16_t', 'H', 4097)]
    for name, ctype, fmt, count in tensors:
        size = struct.calcsize('<' + fmt) * count
        values = list(struct.unpack_from('<' + str(count) + fmt, data, offset))
        offset += size
        if fmt == 'B': values = [value - 128 for value in values]
        declarations.append(f'extern const {ctype} nn_{name}[{count}];')
        definitions.append(f'const {ctype} nn_{name}[{count}] __attribute__((aligned(4))) = {{')
        definitions += [','.join(str(value) for value in values[i:i + 24]) + ',' for i in range(0, count, 24)]
        definitions.append('};')
    if offset != len(data):
        raise ValueError('Unexpected PNN trailing data')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'model.h').write_text('#ifndef NN_MODEL_H\n#define NN_MODEL_H\n#include <stdint.h>\n'
        + '\n'.join(f'#define NN_{key} {value}' for key, value in constants.items())
        + f'\n#define NN_MODEL_SHA "{digest}"\n' + '\n'.join(declarations) + '\n#endif\n')
    (output / 'model.c').write_text('\n'.join(definitions) + '\n')
    return {**{k.lower(): v for k,v in constants.items()}, 'pnn_sha256': digest,
            'model_source': str(Path(source).resolve()), 'flash_tensor_bytes': len(data) - header_size,
            'dense_macs_per_byte': hidden * (context * embedding + 256) + (hidden_layers - 1) * hidden**2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/pico-32k-context3/model.pnn')
    parser.add_argument('--sdk', type=Path, default=ROOT / '.cache/py32f4xx-firmware')
    parser.add_argument('--output', type=Path, default=HERE / 'build')
    parser.add_argument('--fetch-sdk', action='store_true')
    args = parser.parse_args()
    if args.fetch_sdk and not args.sdk.exists():
        subprocess.run(['git', 'clone', 'https://github.com/OpenPuya/PY32F4xx_Firmware.git', str(args.sdk)], check=True)
        subprocess.run(['git', '-C', str(args.sdk), 'checkout', '--detach', SDK_COMMIT], check=True)
    sdk = args.sdk.resolve()
    if not (sdk / 'Drivers/CMSIS/Device/PUYA/PY32F403/Include/py32f403xD.h').is_file():
        parser.error('Device SDK missing; rerun with --fetch-sdk or specify --sdk')
    actual = subprocess.check_output(['git', '-C', str(sdk), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != SDK_COMMIT:
        parser.error('SDK revision differs from the verified revision: ' + SDK_COMMIT)
    out = args.output.resolve()
    meta = generate_model(args.model, out)
    # Keep the vendor factory-trim initialization and its original copyright.
    system = (sdk / 'Templates/PY32F403xx_Templates/Src/system_py32f403.c').read_text()
    system = system.replace('HAL_NVIC_SystemReset();', 'NVIC_SystemReset();')
    (out / 'system_py32f403.c').write_text(system)
    includes = [HERE, out, sdk / 'Drivers/CMSIS/Include',
                sdk / 'Drivers/CMSIS/Device/PUYA/PY32F403/Include']
    flags = ['-mcpu=cortex-m4', '-mthumb', '-mfloat-abi=soft', '-O3', '-g3',
             '-std=gnu11', '-ffreestanding', '-fno-builtin', '-ffunction-sections',
             '-fdata-sections', '-fno-strict-aliasing', '-Wall', '-Wextra', '-DPY32F403xD']
    flags += [f'-I{p}' for p in includes]
    objects = []
    for name, source in [('nn', HERE / 'nn.c'), ('codec', HERE / 'codec.c'),
                         ('model', out / 'model.c'), ('main', HERE / 'main.c'),
                         ('platform', HERE / 'platform.c'), ('startup', HERE / 'startup.c'),
                         ('system', out / 'system_py32f403.c')]:
        obj = out / (name + '.o')
        subprocess.run(['arm-none-eabi-gcc', *flags, '-c', str(source), '-o', str(obj)], check=True)
        objects.append(str(obj))
    elf, binary = out / 'nn-py32f403.elf', out / 'nn-py32f403.bin'
    subprocess.run(['arm-none-eabi-gcc', *flags, '-nostdlib', *objects, '-lgcc',
                    '-T' + str(HERE / 'linker.ld'), '-Wl,--gc-sections,--print-memory-usage',
                    '-Wl,-Map=' + str(out / 'nn-py32f403.map'), '-o', str(elf)], check=True)
    subprocess.run(['arm-none-eabi-objcopy', '-O', 'binary', str(elf), str(binary)], check=True)
    subprocess.run(['arm-none-eabi-gcc', *flags, '-nostdlib', str(HERE / 'flash_loader.c'),
                    '-T' + str(HERE / 'flash_loader.ld'), '-Wl,--gc-sections', '-lgcc',
                    '-o', str(out / 'flash-loader.elf')], check=True)
    meta.update(sdk_commit=actual, firmware_bytes=binary.stat().st_size,
                firmware_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                cpu_hz=144000000, sram_bytes=65536, stack_reserved_bytes=8192,
                compiler=subprocess.check_output(['arm-none-eabi-gcc', '--version'], text=True).splitlines()[0])
    meta['size_report'] = subprocess.check_output(['arm-none-eabi-size', str(elf)], text=True)
    (out / 'build.json').write_text(json.dumps(meta, indent=2) + '\n')
    shutil.copy(sdk / 'LICENSE', out / 'SDK-LICENSE')
    shutil.copy(sdk / 'Drivers/CMSIS/LICENSE.txt', out / 'CMSIS-LICENSE')
    print(meta['size_report'])
    print('Built:', binary)


if __name__ == '__main__':
    main()
