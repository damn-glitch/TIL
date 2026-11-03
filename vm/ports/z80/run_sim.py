#!/usr/bin/env python3
"""Compile a TIL program, link it with the TIL VM for the Z80 (SDCC) and run it
on the ucsim Z80 simulator (sz80). Prints the program output and statistics.

    python3 run_sim.py program.til [--keep]
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import til  # noqa: E402

BUILD = os.path.join(HERE, 'build')


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.stdout


def dump(cmd_out: str, start: int, count: int) -> bytes:
    data = {}
    for line in cmd_out.splitlines():
        m = re.match(r'^\s*0x([0-9a-fA-F]+)\s+((?:[0-9a-fA-F]{2}\s+)+)', line)
        if m:
            addr = int(m.group(1), 16)
            for i, b in enumerate(m.group(2).split()):
                data[addr + i] = int(b, 16)
    return bytes(data.get(start + i, 0) for i in range(count))


def main():
    src_path = sys.argv[1]
    os.makedirs(BUILD, exist_ok=True)
    if not os.path.exists(os.path.join(BUILD, 'tilvm.rel')):
        sh(['make', '-C', HERE])     # compiles the VM once (several minutes with SDCC)
    with open(src_path, encoding='utf-8') as f:
        module = til.compile_to_bytecode(f.read(), src_path)
    tbc = module.serialize(strip=True)
    with open(os.path.join(BUILD, 'program.h'), 'w') as f:
        f.write(til.bytes_to_c_array('til_program', tbc))
    sh(['sdcc', '-mz80', '--std-c99', '--code-loc', '0x0200', '--data-loc', '0xC000',
        '-DTILVM_LEAN', '-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=0', '-DTILVM_NO_STDIO', '-DTILVM_NO_TIME',
        '-DTILVM_HEAP_SIZE=4096', '-DTILVM_STACK_SIZE=192', '-DTILVM_MAX_FRAMES=32',
        '-DTILVM_MAX_CONSTS=192', '-DTILVM_MAX_TYPES=24', '-DTILVM_MAX_TRY=8', '-DTILVM_MARK_STACK=16',
        '-I', os.path.join(HERE, '..', '..'), '-I', BUILD,
        os.path.join(HERE, 'main_z80.c'), os.path.join(BUILD, 'tilvm.rel'),
        '-o', os.path.join(BUILD, 'prog.ihx')], cwd=BUILD)
    code_end = 0
    for line in open(os.path.join(BUILD, 'prog.map'), errors='replace'):
        m = re.match(r'^_CODE\s+([0-9A-F]{8})\s+([0-9A-F]{8})', line)
        if m:
            code_end = int(m.group(1), 16) + int(m.group(2), 16)
    if code_end > 0xC000:
        raise SystemExit(f"code ends at 0x{code_end:X}: does not fit below the VM data at 0xC000")
    cmds = "run\ndump rom 0xdbf0 0xdbf3\ndump rom 0xdc00 0xf3ff\ninfo registers\nquit\n"
    out = subprocess.run(['sz80', '-b', '-c', '-', os.path.join(BUILD, 'prog.ihx')],
                         input=cmds, capture_output=True, text=True, timeout=600).stdout
    hdr = dump(out, 0xdbf0, 3)
    n = hdr[0] | (hdr[1] << 8)
    text = dump(out, 0xdc00, n)
    sys.stdout.write(text.decode('utf-8', 'replace'))
    ticks = re.search(r'Simulated (\d+) ticks', out)
    code_size = os.path.getsize(os.path.join(BUILD, 'prog.ihx'))
    sys.stderr.write(f"[z80] exit={hdr[2]} code=0x0200-0x{code_end:04X} ({code_end - 0x200} bytes) "
                     f"bytecode={len(tbc)} bytes cycles={ticks.group(1) if ticks else '?'}\n")
    return 0 if hdr[2] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
