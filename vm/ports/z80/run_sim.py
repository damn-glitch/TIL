#!/usr/bin/env python3
"""Run TIL programs on the TIL VM for the Z80 (16-bit lean build, SDCC) in the
ucsim Z80 simulator `sz80` (shipped with SDCC).

    python3 run_sim.py program.til [...]     program output on stdout, statistics on stderr
    python3 run_sim.py --json program.til    results as JSON (used by tests/test_z80.py)
    python3 run_sim.py --size                memory map, code size, interpreter stack frame

The VM image (build/tilvm.ihx: VM + test harness, see main_z80.c) is built once
with `make`; each program's bytecode is placed in RAM at 0xD800, like a program
on a calculator, so running another program never rebuilds the VM.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
BUILD = os.path.join(HERE, 'build')
IHX = os.path.join(BUILD, 'tilvm.ihx')

CODE_LIMIT = 0xC000            # VM code + constants must end below (RAM starts here)
DATA_LIMIT = 0xD800            # VM state must end below the program
PROG_ADDR, PROG_MAX = 0xD800, 0x17F0
RES_ADDR, OUT_ADDR, OUT_MAX = 0xEFF0, 0xF000, 0x0800
CODE_AREAS = ('_CODE', '_HOME', '_INITIALIZER', '_GSINIT', '_GSFINAL')
DATA_AREAS = ('_DATA', '_INITIALIZED', '_BSEG', '_BSS', '_HEAP')
SOURCES = [os.path.join(ROOT, 'vm', 'tilvm.c'), os.path.join(ROOT, 'vm', 'tilvm.h'),
           os.path.join(HERE, 'main_z80.c'), os.path.join(HERE, 'Makefile')]


def build():
    """Build the VM image unless it is up to date (SDCC needs a few minutes)."""
    if os.path.exists(IHX) and all(os.path.getmtime(IHX) >= os.path.getmtime(s) for s in SOURCES):
        return
    r = subprocess.run(['make', '-C', HERE], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(IHX):
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        raise SystemExit("building the Z80 VM failed")


def memory_map():
    """Areas of the linked image: {name: (address, size)}."""
    areas = {}
    with open(os.path.join(BUILD, 'tilvm.map'), errors='replace') as f:
        for line in f:
            m = re.match(r'^(_\w+)\s+([0-9A-F]{8})\s+([0-9A-F]{8})\s+=', line)
            if m:
                areas[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16))
    return areas


def frame_size(func='tilvm_run_steps'):
    """Bytes of locals in a function's stack frame (from its prologue)."""
    with open(os.path.join(BUILD, 'tilvm.asm'), errors='replace') as f:
        asm = f.read()
    m = re.search(r'^_%s::?\s*\n((?:.*\n){1,12})' % func, asm, re.M)
    if not m:
        return None
    body = m.group(1)
    m2 = re.search(r'ld\s+(?:iy|hl),\s*#-(\d+)', body)
    if m2:
        return int(m2.group(1))
    return 2 * len(re.findall(r'^\s*push\s+af', body, re.M))   # small frames: push af per 2 bytes


def sizes():
    areas = memory_map()
    code = [(a, s) for n, (a, s) in areas.items() if n in CODE_AREAS and s]
    data = [(a, s) for n, (a, s) in areas.items() if n in DATA_AREAS and s]
    code_start = min(a for a, _ in code)
    code_end = max(a + s for a, s in code)
    data_end = max(a + s for a, s in data)
    return {'code_start': code_start, 'code_end': code_end, 'code_bytes': code_end - code_start,
            'vm_code_bytes': areas['_CODE'][1], 'data_start': min(a for a, _ in data),
            'data_end': data_end, 'data_bytes': sum(s for _, s in data),
            'run_steps_frame': frame_size()}


def check_layout(info):
    if info['code_end'] > CODE_LIMIT:
        raise SystemExit(f"code ends at 0x{info['code_end']:04X}: does not fit below 0x{CODE_LIMIT:04X}")
    if info['data_end'] > DATA_LIMIT:
        raise SystemExit(f"VM state ends at 0x{info['data_end']:04X}: overlaps the program at 0x{DATA_LIMIT:04X}")


def ihx_records(addr, data):
    out = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        a = addr + i
        rec = bytes([len(chunk), (a >> 8) & 255, a & 255, 0]) + chunk
        out.append(':%s%02X' % (rec.hex().upper(), (-sum(rec)) & 255))
    return out


def dump(text, start, count):
    data = {}
    for line in text.splitlines():
        m = re.match(r'^\s*0x([0-9a-fA-F]+)\s+((?:[0-9a-fA-F]{2}\s+)+)', line)
        if m:
            addr = int(m.group(1), 16)
            for i, b in enumerate(m.group(2).split()):
                data[addr + i] = int(b, 16)
    return bytes(data.get(start + i, 0) for i in range(count))


def compile_til(path):
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    import til  # noqa: E402
    with open(path, encoding='utf-8') as f:
        return til.compile_to_bytecode(f.read(), path).serialize()   # with line numbers


def run_tbc(tbc, timeout=600):
    """Run bytecode on the simulated Z80; returns a dict of results."""
    if len(tbc) + 2 > PROG_MAX:
        raise ValueError(f"program too large for the simulator layout ({len(tbc)} bytes)")
    with open(IHX) as f:
        image = [line.strip() for line in f if line.strip() and not line.startswith(':00000001')]
    image += ihx_records(PROG_ADDR, bytes([len(tbc) & 255, len(tbc) >> 8]) + tbc) + [':00000001FF']
    path = os.path.join(BUILD, 'run.ihx')
    with open(path, 'w') as f:
        f.write('\n'.join(image) + '\n')
    cmds = (f"run\ndump rom 0x{RES_ADDR:x} 0x{RES_ADDR + 15:x}\n"
            f"dump rom 0x{OUT_ADDR:x} 0x{OUT_ADDR + OUT_MAX - 1:x}\nquit\n")
    sim = subprocess.run(['sz80', '-b', '-c', '-', path], input=cmds, capture_output=True,
                         text=True, timeout=timeout).stdout
    res = dump(sim, RES_ADDR, 16)
    n = min(res[0] | (res[1] << 8), OUT_MAX)
    ticks = re.search(r'Simulated (\d+) ticks', sim)
    return {
        'output': dump(sim, OUT_ADDR, n).decode('utf-8', 'replace'),
        'exit_code': res[2], 'load_error': res[3],
        'steps': int.from_bytes(res[4:8], 'little'),
        'heap_used': res[8] | (res[9] << 8), 'gc_count': res[10] | (res[11] << 8),
        'c_stack_peak': 0x10000 - (res[12] | (res[13] << 8)),
        'cycles': int(ticks.group(1)) if ticks else None,
        'bytecode_bytes': len(tbc),
    }


def main(argv):
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    build()
    info = sizes()
    if argv[0] == '--size':
        print(json.dumps(info, indent=2) if '--json' in argv else
              f"code 0x{info['code_start']:04X}-0x{info['code_end']:04X} ({info['code_bytes']} bytes, "
              f"VM _CODE {info['vm_code_bytes']}), limit 0x{CODE_LIMIT:04X}\n"
              f"data 0x{info['data_start']:04X}-0x{info['data_end']:04X} ({info['data_bytes']} bytes)\n"
              f"tilvm_run_steps frame: {info['run_steps_frame']} bytes (must stay below 128)")
        return 0
    as_json = argv[0] == '--json'
    files = argv[1:] if as_json else argv
    check_layout(info)
    results, rc = [], 0
    for path in files:
        r = run_tbc(compile_til(path))
        r['file'] = path
        results.append(r)
        if not as_json:
            sys.stdout.write(r['output'])
            sys.stderr.write(f"[z80] {os.path.basename(path)}: exit={r['exit_code']} steps={r['steps']} "
                             f"cycles={r['cycles']} heap={r['heap_used']} gc={r['gc_count']} "
                             f"C-stack={r['c_stack_peak']} bytecode={r['bytecode_bytes']}\n")
        rc = rc or (0 if r['exit_code'] == 0 else 1)
    if as_json:
        print(json.dumps({'sizes': info, 'runs': results}, indent=2))
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
