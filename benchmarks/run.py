#!/usr/bin/env python3
"""
TIL VM benchmarks: run every benchmarks/*.til program several times on the
C VM (desktop build, -O2), the reference Python VM and optionally the native
backend, and print a table of median times.

    python3 benchmarks/run.py                       # C VM + Python VM (this checkout)
    python3 benchmarks/run.py --baseline 6f4d00b    # ... compared with an older commit
    python3 benchmarks/run.py --vms c,c-nofuse,c-base --repeat 11 --only fib,loops
    python3 benchmarks/run.py --native --markdown   # also the native backend; Markdown table

VMs (--vms):
    c          C VM of this checkout, fused bytecode (the default compiler output)
    c-nofuse   C VM of this checkout, bytecode built with --no-fuse
    py         reference VM (src/til.py, class TILVM), fused bytecode
    c-base     C VM of the --baseline commit, bytecode from that commit's compiler
    py-base    Python VM of the --baseline commit
    native     native backend (til build: C via gcc/clang)

Times are CPU seconds (user + sys) of the process running the program; for the
Python VMs only TILVM.run() is timed (no interpreter start-up or compilation).
Every VM must print the same output; a mismatch is reported and the exit code is 1.
The baseline sources are read with `git show REV:path` (the checkout is untouched).
"""

import argparse
import glob
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PY_DRIVER = r'''
import sys, time
sys.path.insert(0, sys.argv[1])
sys.setrecursionlimit(20000)
import til
with open(sys.argv[2], 'rb') as f:
    module = til.TBCModule.from_bytes(f.read())
out = []
t0 = time.process_time()
rc = til.TILVM(module, write=out.append).run()
t1 = time.process_time()
sys.stdout.buffer.write(b"".join(out))
sys.stdout.flush()
sys.stderr.write("TIME %.6f\n" % (t1 - t0))
sys.exit(rc)
'''

LABELS = {
    'c-base': 'C VM base', 'c': 'C VM', 'c-nofuse': 'C VM no-fuse',
    'py-base': 'Py VM base', 'py': 'Py VM', 'native': 'native',
}


def find_cc():
    for cc in (os.environ.get('CC'), 'gcc', 'clang', 'cc'):
        if cc and shutil.which(cc):
            return cc
    return None


def build_c_vm(cc, vm_dir, exe):
    cmd = [cc, '-std=c99', '-O2', '-o', exe, os.path.join(vm_dir, 'tilvm.c'),
           os.path.join(vm_dir, 'main.c'), '-lm']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"C VM build failed: {' '.join(cmd)}\n{r.stderr}")


def extract_baseline(rev, dest):
    """Copy the sources of commit `rev` (git show) into dest/{src,vm}."""
    for rel in ('src/til.py', 'vm/tilvm.c', 'vm/tilvm.h', 'vm/main.c'):
        r = subprocess.run(['git', '-C', ROOT, 'show', f'{rev}:{rel}'], capture_output=True)
        if r.returncode != 0:
            sys.exit(f"cannot read {rel} at {rev}: {r.stderr.decode(errors='replace')}")
        path = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(r.stdout)


def compile_tbc(til_py, source, out, extra=()):
    r = subprocess.run([sys.executable, til_py, 'build', '--target', 'vm', *extra, source, '-o', out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"compile failed ({til_py} {source}):\n{r.stdout}{r.stderr}")


def timed_run(cmd):
    """Run cmd; return (cpu seconds, stdout bytes, returncode)."""
    if hasattr(os, 'wait4'):
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            p = subprocess.Popen(cmd, stdout=out, stderr=err)
            _pid, status, usage = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status) if hasattr(os, 'waitstatus_to_exitcode') \
                else (status >> 8)
            out.seek(0)
            err.seek(0)
            stdout, stderr = out.read(), err.read()
            cpu = usage.ru_utime + usage.ru_stime
    else:                                     # Windows: wall-clock time
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True)
        cpu = time.perf_counter() - t0
        stdout, stderr, p = r.stdout, r.stderr, r
    for line in stderr.decode('utf-8', 'replace').splitlines():
        if line.startswith('TIME '):          # Python VM driver: time of TILVM.run() only
            cpu = float(line.split()[1])
    return cpu, stdout, p.returncode, stderr


def machine_info(cc):
    cpu = platform.processor() or platform.machine()
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    ccv = ''
    if cc:
        r = subprocess.run([cc, '--version'], capture_output=True, text=True)
        ccv = (r.stdout.splitlines() or [''])[0]
    return {'cpu': cpu, 'cores': os.cpu_count(), 'os': platform.platform(), 'cc': ccv,
            'python': platform.python_version()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repeat', type=int, default=0, help="runs per benchmark and VM (default: 7 for "
                    "C VMs/native, 3 for Python VMs)")
    ap.add_argument('--vms', default='', help="comma-separated VMs (default: c,py; with --baseline also "
                    "c-base,py-base)")
    ap.add_argument('--baseline', metavar='REV', help="git revision to compare with (e.g. 6f4d00b = TIL 3.0)")
    ap.add_argument('--native', action='store_true', help="also run the native backend")
    ap.add_argument('--only', default='', help="comma-separated benchmark names (default: all)")
    ap.add_argument('--markdown', action='store_true', help="print a Markdown table")
    ap.add_argument('--json', metavar='FILE', help="write all measurements to FILE")
    args = ap.parse_args(argv)

    cc = find_cc()
    vms = [v for v in args.vms.split(',') if v] if args.vms else \
        (['c-base', 'c', 'py-base', 'py'] if args.baseline else ['c', 'py'])
    if args.native and 'native' not in vms:
        vms.append('native')
    for v in vms:
        if v not in LABELS:
            sys.exit(f"unknown VM '{v}' (choose from {', '.join(LABELS)})")
    if any(v.endswith('-base') for v in vms) and not args.baseline:
        sys.exit("c-base / py-base need --baseline REV")
    if cc is None and any(v.startswith('c') or v == 'native' for v in vms):
        sys.exit("no C compiler found (set CC)")

    benches = sorted(glob.glob(os.path.join(HERE, '*.til')))
    if args.only:
        wanted = set(args.only.split(','))
        benches = [b for b in benches if os.path.splitext(os.path.basename(b))[0] in wanted]
    if not benches:
        sys.exit("no benchmarks selected")

    work = tempfile.mkdtemp(prefix='til-bench-')
    try:
        # snapshot this checkout's compiler (edits during a long run cannot skew it)
        cur_src = os.path.join(work, 'cur', 'src')
        shutil.copytree(os.path.join(ROOT, 'src'), cur_src,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        til_py = os.path.join(cur_src, 'til.py')
        exes = {}
        if 'c' in vms or 'c-nofuse' in vms:
            exes['c'] = os.path.join(work, 'tilvm')
            build_c_vm(cc, os.path.join(ROOT, 'vm'), exes['c'])
        base_dir = os.path.join(work, 'base')
        if args.baseline:
            extract_baseline(args.baseline, base_dir)
            if 'c-base' in vms:
                exes['c-base'] = os.path.join(work, 'tilvm-base')
                build_c_vm(cc, os.path.join(base_dir, 'vm'), exes['c-base'])

        plan = {}             # (bench, vm) -> command
        for b in benches:
            name = os.path.splitext(os.path.basename(b))[0]
            fused = os.path.join(work, name + '.tbc')
            compile_tbc(til_py, b, fused)
            if 'c' in vms:
                plan[name, 'c'] = [exes['c'], fused]
            if 'py' in vms:
                plan[name, 'py'] = [sys.executable, '-c', PY_DRIVER, cur_src, fused]
            if 'c-nofuse' in vms:
                nofuse = os.path.join(work, name + '.nofuse.tbc')
                compile_tbc(til_py, b, nofuse, ['--no-fuse'])
                plan[name, 'c-nofuse'] = [exes['c'], nofuse]
            if args.baseline and ('c-base' in vms or 'py-base' in vms):
                base_tbc = os.path.join(work, name + '.base.tbc')
                compile_tbc(os.path.join(base_dir, 'src', 'til.py'), b, base_tbc)
                if 'c-base' in vms:
                    plan[name, 'c-base'] = [exes['c-base'], base_tbc]
                if 'py-base' in vms:
                    plan[name, 'py-base'] = [sys.executable, '-c', PY_DRIVER, os.path.join(base_dir, 'src'),
                                             base_tbc]
            if 'native' in vms:
                exe = os.path.join(work, name + '-native')
                r = subprocess.run([sys.executable, til_py, 'build', b, '-o', exe], capture_output=True, text=True)
                if r.returncode == 0 and os.path.exists(exe):
                    plan[name, 'native'] = [exe]
                else:
                    print(f"[bench] {name}: the native backend cannot compile it (skipped)", file=sys.stderr)

        times = {k: [] for k in plan}
        outputs = {}
        failed = False
        max_rep = max(args.repeat or 7, 1)
        for rep in range(max_rep):
            for b in benches:                 # interleaved: load changes hit every VM alike
                name = os.path.splitext(os.path.basename(b))[0]
                for vm in vms:
                    if (name, vm) not in plan:
                        continue
                    want = args.repeat or (3 if vm.startswith('py') else 7)
                    if rep >= want:
                        continue
                    cpu, out, rc, err = timed_run(plan[name, vm])
                    if rc != 0:
                        print(f"[bench] {name} on {vm}: exit code {rc}\n{err.decode(errors='replace')}",
                              file=sys.stderr)
                        failed = True
                    times[name, vm].append(cpu)
                    outputs.setdefault(name, {})[vm] = out
            print(f"[bench] round {rep + 1}/{max_rep} done", file=sys.stderr)

        for name, per_vm in outputs.items():
            if len(set(per_vm.values())) > 1:
                failed = True
                print(f"[bench] OUTPUT MISMATCH in {name}: " +
                      ", ".join(f"{vm}={out[:60]!r}" for vm, out in per_vm.items()), file=sys.stderr)

        names = [os.path.splitext(os.path.basename(b))[0] for b in benches]
        med = {k: statistics.median(v) for k, v in times.items() if v}
        cols = [vm for vm in vms if any((n, vm) in med for n in names)]
        speedups = []
        if 'c-base' in cols and 'c' in cols:
            speedups.append(('c-base', 'c', 'C speedup'))
        if 'c-base' in cols and 'c-nofuse' in cols:
            speedups.append(('c-base', 'c-nofuse', 'C no-fuse speedup'))
        if 'py-base' in cols and 'py' in cols:
            speedups.append(('py-base', 'py', 'Py speedup'))
        header = ['benchmark'] + [LABELS[c] + ' (s)' for c in cols] + [s[2] for s in speedups]
        rows = []
        for n in names:
            row = [n] + [f"{med[n, c]:.3f}" if (n, c) in med else "-" for c in cols]
            for a, b, _t in speedups:
                row.append(f"{med[n, a] / med[n, b]:.2f}x" if (n, a) in med and (n, b) in med and med[n, b] > 0
                           else "-")
            rows.append(row)
        info = machine_info(cc)
        print(f"\nmachine: {info['cpu']} ({info['cores']} cores), {info['os']}")
        print(f"C compiler: {info['cc']} (-std=c99 -O2); Python {info['python']}")
        print(f"runs: median of {args.repeat or '7 (C/native) / 3 (Python)'}; CPU seconds\n")
        if args.markdown:
            print("| " + " | ".join(header) + " |")
            print("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
            for row in rows:
                print("| " + " | ".join(row) + " |")
        else:
            widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
            print("  ".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(header, widths))))
            for row in rows:
                print("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths))))
        if args.json:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump({'machine': info, 'baseline': args.baseline,
                           'times': {f"{n}/{vm}": v for (n, vm), v in times.items()},
                           'median': {f"{n}/{vm}": v for (n, vm), v in med.items()}}, f, indent=1)
        return 1 if failed else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
