#!/usr/bin/env python3
"""Back up, page-program, and verify PY32F403 flash through a RAM-resident loader."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent


def program(build, backup_root):
    build = Path(build).resolve()
    backup = Path(backup_root).resolve() / time.strftime('%Y%m%d-%H%M%S')
    backup.mkdir(parents=True, exist_ok=False)
    firmware = (build / 'nn-py32f403.bin').read_bytes()
    if not 0 < len(firmware) <= 384 * 1024:
        raise ValueError('Invalid firmware size')
    firmware += b'\xff' * (-len(firmware) % 256)
    # Tcl paths are braced, and directory names containing Tcl delimiters rejected.
    def path(p):
        value = str(Path(p).resolve())
        if any(ch in value for ch in '{}\n\r'):
            raise ValueError('Unsupported Tcl path')
        return '{' + value + '}'
    commands = ['init', 'halt',
                f'dump_image {path(backup / "flash.bin")} 0x08000000 393216',
                f'dump_image {path(backup / "options.bin")} 0x1fff5000 4096',
                f'load_image {path(build / "flash-loader.elf")}',
                'reg msp 0x2000fff0', 'reg xpsr 0x01000000']
    for offset in range(0, len(firmware), 4096):
        block = firmware[offset:offset + 4096]
        image = build / ('flash-block-%05x.bin' % offset)
        image.write_bytes(block)
        commands += [f'load_image {path(image)} 0x20004000 bin',
                     f'write_memory 0x20003f00 32 {{ {0x08000000 + offset} {len(block)//256} 536887296 4294967295 0 }}',
                     'reg r0 0x20003f00', 'reg msp 0x2000fff0',
                     'resume 0x20000000', 'wait_halt 10000',
                     'set status [read_memory 0x20003f0c 32 1]',
                     f'if {{$status != 0}} {{error "Flash loader error at offset {offset}: $status"}}',
                     f'echo "Programmed {min(offset+4096,len(firmware))}/{len(firmware)} bytes"']
    padded = build / 'firmware-padded.bin'
    padded.write_bytes(firmware)
    commands += [f'dump_image {path(backup / "programmed-readback.bin")} 0x08000000 {len(firmware)}',
                 f'dump_image {path(backup / "options-after.bin")} 0x1fff5000 4096',
                 'reset run', 'shutdown']
    script = build / 'program.tcl'
    script.write_text('\n'.join(commands) + '\n')
    print('Backup:', backup, flush=True)
    subprocess.run(['openocd', '-f', str(HERE / 'openocd.cfg'), '-f', str(script)], check=True)
    if (backup / 'programmed-readback.bin').read_bytes() != firmware:
        raise RuntimeError('Flash readback differs from the requested firmware')
    before, after = (backup / 'options.bin').read_bytes(), (backup / 'options-after.bin').read_bytes()
    if before != after:
        raise RuntimeError('Configuration readback changed; inspect backup')
    report = {'backup': str(backup), 'original_flash_sha256': hashlib.sha256((backup / 'flash.bin').read_bytes()).hexdigest(),
              'programmed_bytes': len(firmware), 'programmed_sha256': hashlib.sha256(firmware).hexdigest(),
              'options_unchanged': True, 'readback_verified': True}
    (build / 'flash-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=HERE / 'build')
    parser.add_argument('--backups', type=Path, default=HERE.parent / '.cache/py32-backup')
    args = parser.parse_args()
    print(json.dumps(program(args.build, args.backups), indent=2))
