#!/usr/bin/env python3
"""
Native backend coverage: which TIL programs compile with the native (C)
backend, and does the native executable print exactly what the reference VM
prints?

    python3 tools/native_coverage.py                 # summary table
    python3 tools/native_coverage.py --details       # one line per program
    python3 tools/native_coverage.py --json out.json # per-program results
    python3 tools/native_coverage.py --compare tools/native_coverage_baseline.json

Corpus (default)
----------------
* every TIL program written in the test-suite files tests/test_compiler.py,
  tests/test_v3.py and tests/test_foundation.py (more files can be given as
  arguments), and
* every examples/**/*.til file.

How programs are taken out of the test files (static, deterministic, no test
is executed): the file is parsed with Python's `ast` module. Every string
literal is a candidate, and so is every `+` concatenation whose operands are
string literals or names bound to string literals in the same function or at
module level (tests build programs as `base + '...'`). A candidate counts as
a program when it parses, passes the verifier and compiles to bytecode, and
contains at least one call -- expected outputs such as "3\\n2" or "true" are
expressions without calls and are skipped; deliberately invalid programs
(compile-error tests) fail to compile and are skipped too. Programs are
de-duplicated by text; a program's id is the SHA-1 of its text (12 hex digits).

What is measured per program
----------------------------
* reference: the Python VM (no program arguments, empty stdin, a budget of
  5,000,000 steps) -> exit code and stdout;
* native: TILCompiler.compile -> C compiler (same driver as `til run`) ->
  run the executable (empty stdin, timeout) -> exit code and stdout.

Both runs use the same scratch working directory (deleted afterwards), so a
program that writes files (`write_file("NOTE", ...)`) never touches the
checkout.

Status: `native-ok` (same exit code and stdout as the VM), `native-mismatch`,
`fallback` (the native backend cannot compile it: `til run` uses the VM), or
`vm-limit` (the VM did not finish within the step budget or stopped with a
fatal error such as its recursion limit: native limits differ, not compared).

The recorded baseline (`tools/native_coverage_baseline.json`) was measured on
the TIL 3.0 native backend; docs/TIL_TYPES.md has the numbers before/after.
"""

import ast
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import til  # noqa: E402

DEFAULT_TEST_FILES = ['tests/test_compiler.py', 'tests/test_v3.py', 'tests/test_foundation.py']
VM_STEPS = 5_000_000
OUTPUT_CAP = 1 << 20          # bytes of stdout kept per run
NATIVE_TIMEOUT = 20           # seconds


class Program:
    __slots__ = ('id', 'label', 'source', 'filename')

    def __init__(self, label: str, source: str, filename: str = "<test>"):
        self.id = hashlib.sha1(source.encode('utf-8')).hexdigest()[:12]
        self.label = label
        self.source = source
        self.filename = filename        # imports are resolved next to it


# ── corpus extraction ───────────────────────────────────────────────────────

class _Folder:
    """Folds string literals, names bound to strings and `+` concatenations."""

    MAX_VALUES = 8

    def __init__(self, module_env):
        self.module_env = module_env

    def fold(self, node, env):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.JoinedStr) and all(isinstance(v, ast.Constant) for v in node.values):
            return [''.join(str(v.value) for v in node.values)]
        if isinstance(node, ast.Name):
            return env.get(node.id) or self.module_env.get(node.id) or []
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.fold(node.left, env), self.fold(node.right, env)
            return [a + b for a in left for b in right][:self.MAX_VALUES]
        return []


def _bindings(body, folder, env):
    """Names assigned string values in a statement list (flow-insensitive)."""
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not stmt:
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                values = folder.fold(node.value, env)
                if values:
                    seen = env.setdefault(node.targets[0].id, [])
                    for v in values:
                        if v not in seen and len(seen) < _Folder.MAX_VALUES:
                            seen.append(v)


def test_file_candidates(path):
    """(line, text) string candidates of one Python test file."""
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read(), path)
    module_env = {}
    folder = _Folder(module_env)
    _bindings([s for s in tree.body if isinstance(s, ast.Assign)], folder, module_env)
    out = []
    scopes = [(tree, {})]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            env = {}
            _bindings(node.body, folder, env)
            scopes.append((node, env))
    for scope, env in scopes:
        for node in ast.walk(scope):
            if isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr)):
                for text in folder.fold(node, env):
                    out.append((getattr(node, 'lineno', 0), text))
    return out


def is_program(source, filename="<test>"):
    """Valid TIL program with at least one call (see the module docstring)."""
    if '\n' not in source.strip() and '(' not in source:
        return False
    saved_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')      # "could not find module" warnings of test fragments
    try:
        program = til.resolve_imports(til.parse_source(source, filename), filename)
        if til.SemanticChecker(source, filename).check(program):
            return False
        if not any(isinstance(n, til.Call) for st in program.statements for n in til.walk(st)):
            return False
        til.BytecodeCompiler(source, filename).compile_program(program)
        return True
    except BaseException:       # SyntaxError, CompileError, parser crashes on non-TIL text
        return False
    finally:
        sys.stderr.close()
        sys.stderr = saved_stderr


def collect_corpus(test_files=None, examples=True):
    programs, seen = [], set()

    def add(prog):
        if prog.id not in seen:
            seen.add(prog.id)
            programs.append(prog)

    for rel in (DEFAULT_TEST_FILES if test_files is None else test_files):
        path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        shown = os.path.relpath(path, ROOT)
        for line, text in test_file_candidates(path):
            if is_program(text):
                add(Program(f"{shown}:{line}", text))
    if examples:
        for dirpath, _dirs, files in sorted(os.walk(os.path.join(ROOT, 'examples'))):
            for name in sorted(files):
                if not name.endswith('.til'):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding='utf-8') as f:
                    text = f.read()
                if is_program(text, path):
                    add(Program(os.path.relpath(path, ROOT), text, path))
    return programs


# ── measuring ───────────────────────────────────────────────────────────────

class _OutputCap(Exception):
    pass


def run_reference(prog):
    """(exit code, stdout bytes) on the Python VM, or None when the VM hit a limit
    (step budget, output cap, or a fatal error such as its recursion limit)."""
    module = til.TBCModule.from_bytes(til.compile_to_bytecode(prog.source, prog.filename).serialize())
    out = []
    size = [0]

    def write(data):
        out.append(data)
        size[0] += len(data)
        if size[0] > OUTPUT_CAP:
            raise _OutputCap()
    vm = til.TILVM(module, write=write, read_line=lambda: b"", max_steps=VM_STEPS)
    reports = []
    vm.report = lambda msg, pc: reports.append(msg)
    try:
        rc = vm.run()
    except _OutputCap:
        return None
    if vm.max_steps and vm.steps > vm.max_steps:
        return None
    if any(m.startswith(b"fatal error:") for m in reports):
        return None     # VM resource limit (recursion depth, ...): native limits differ
    return rc, b"".join(out)


def first_error_line(text):
    for line in text.splitlines():
        if 'error' in line:
            return line.split('error:', 1)[-1].strip()[:160]
    return (text.strip().splitlines() or ["?"])[0][:160]


def run_native(prog, workdir, opt=0):
    """('ok', rc, stdout) | ('fallback', reason) | ('timeout', reason)."""
    compiler = til.TILCompiler()
    compiler.optimize = opt
    exe = os.path.join(workdir, 'prog_' + prog.id)
    try:
        ok = compiler.compile_to_executable(prog.source, exe, prog.filename, quiet=True)
    except Exception as e:      # the code generator gave up on this program
        return ('fallback', f"codegen: {type(e).__name__}: {str(e).strip().splitlines()[0][:160] if str(e).strip() else ''}")
    if not ok:
        return ('fallback', "cc: " + first_error_line(getattr(compiler, 'last_c_error', '') or ''))
    out_path = exe + '.out'
    try:
        with open(out_path, 'wb') as out:
            # same working directory as the in-process VM run (measure(): a scratch one)
            r = subprocess.run([exe], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.DEVNULL,
                               timeout=NATIVE_TIMEOUT)
        with open(out_path, 'rb') as f:
            data = f.read(OUTPUT_CAP + 1)
        return ('ok', r.returncode, data)
    except subprocess.TimeoutExpired:
        return ('timeout', f"native run exceeded {NATIVE_TIMEOUT}s")
    finally:
        for p in (exe, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def measure_one(prog, workdir, opt=0):
    result = {'id': prog.id, 'label': prog.label}
    try:
        ref = run_reference(prog)
    except Exception as e:
        result.update(status='vm-error', reason=f"{type(e).__name__}: {e}")
        return result
    if ref is None:
        result.update(status='vm-limit', reason=f"VM limit: step budget ({VM_STEPS}), output cap or a fatal "
                                                f"error such as the recursion limit")
        return result
    native = run_native(prog, workdir, opt)
    if native[0] == 'fallback':
        result.update(status='fallback', reason=native[1])
    elif native[0] == 'timeout':
        result.update(status='native-mismatch', reason=native[1])
    else:
        _, rc, out = native
        if (rc, out) == ref:
            result.update(status='native-ok')
        else:
            why = []
            if rc != ref[0]:
                why.append(f"exit {rc} vs VM {ref[0]}")
            if out != ref[1]:
                why.append(f"stdout {out[:60]!r} vs VM {ref[1][:60]!r}")
            result.update(status='native-mismatch', reason="; ".join(why))
    return result


def measure(programs, jobs=None, opt=0):
    """Measure every program. The programs run in a scratch working directory
    (the VM in this process, the executables as child processes), so files
    they write never land in the checkout."""
    jobs = jobs or max(1, min(8, os.cpu_count() or 1))
    with tempfile.TemporaryDirectory(prefix='til-cov-') as workdir:
        rundir = os.path.join(workdir, 'cwd')
        os.mkdir(rundir)
        saved = os.getcwd()
        os.chdir(rundir)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
                return list(pool.map(lambda p: measure_one(p, workdir, opt), programs))
        finally:
            os.chdir(saved)


def summarize(results):
    counts = {}
    for r in results:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    compared = sum(counts.get(s, 0) for s in ('native-ok', 'native-mismatch', 'fallback'))
    native = counts.get('native-ok', 0) + counts.get('native-mismatch', 0)
    return {
        'programs': len(results),
        'compared': compared,
        'native': native,
        'native_ok': counts.get('native-ok', 0),
        'native_mismatch': counts.get('native-mismatch', 0),
        'fallback': counts.get('fallback', 0),
        'vm_limit': counts.get('vm-limit', 0),
        'vm_error': counts.get('vm-error', 0),
    }


def format_summary(summary, title="native backend coverage"):
    s = summary
    pct = (100.0 * s['native'] / s['compared']) if s['compared'] else 0.0
    ok_pct = (100.0 * s['native_ok'] / s['compared']) if s['compared'] else 0.0
    return "\n".join([
        f"{title}",
        f"  programs measured          {s['programs']}",
        f"  compared with the VM       {s['compared']}",
        f"  compiled natively          {s['native']} ({pct:.1f}%)",
        f"    output == VM             {s['native_ok']} ({ok_pct:.1f}%)",
        f"    output differs           {s['native_mismatch']}",
        f"  fall back to the VM        {s['fallback']}",
        f"  not compared (VM limit)    {s['vm_limit']}",
    ])


def reason_histogram(results, top=25):
    hist = {}
    for r in results:
        if r['status'] in ('fallback', 'native-mismatch'):
            key = r['status'] + ": " + (r.get('reason') or '?')
            hist[key] = hist.get(key, 0) + 1
    return sorted(hist.items(), key=lambda kv: -kv[1])[:top]


def compare(results, baseline):
    """Programs whose status got worse than in the baseline file."""
    rank = {'native-ok': 2, 'native-mismatch': 0, 'fallback': 1}
    before = {p['id']: p['status'] for p in baseline.get('programs', [])}
    worse = []
    for r in results:
        old = before.get(r['id'])
        if old in rank and r['status'] in rank:
            if old == 'native-ok' and r['status'] != 'native-ok':
                worse.append((r, old))
    return worse


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('test_files', nargs='*', help="test files to extract programs from "
                                                  "(default: %s)" % ", ".join(DEFAULT_TEST_FILES))
    ap.add_argument('--no-examples', action='store_true')
    ap.add_argument('--jobs', type=int, default=0)
    ap.add_argument('-O', '--opt', type=int, default=0, help="C optimisation level (default 0)")
    ap.add_argument('--json', help="write per-program results to this file")
    ap.add_argument('--compare', help="baseline JSON: report programs that regressed; exit status 1 "
                                      "on a regression or on any native output that differs from the VM")
    ap.add_argument('--details', action='store_true', help="one line per program")
    ap.add_argument('--reasons', action='store_true', help="most common fallback / mismatch reasons")
    ap.add_argument('--filter', help="only programs whose label contains this text")
    args = ap.parse_args(argv)

    programs = collect_corpus(args.test_files or None, not args.no_examples)
    if args.filter:
        programs = [p for p in programs if args.filter in p.label]
    results = measure(programs, args.jobs or None, args.opt)
    summary = summarize(results)
    print(format_summary(summary))
    if args.details:
        for r in results:
            print(f"{r['status']:<16} {r['id']} {r['label']:<32} {r.get('reason', '')}")
    if args.reasons:
        print("\nmost common reasons:")
        for key, n in reason_histogram(results, 60):
            print(f"  {n:4d}  {key}")
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'programs': results}, f, indent=1, ensure_ascii=False)
            f.write("\n")
    rc = 0
    if args.compare:
        with open(args.compare, encoding='utf-8') as f:
            baseline = json.load(f)
        worse = compare(results, baseline)
        print(f"\ncompared with {args.compare}: {len(worse)} regression(s)")
        for r, old in worse:
            print(f"  {r['id']} {r['label']}: {old} -> {r['status']} {r.get('reason', '')}")
        # native output that differs from the VM fails the check too, also for a
        # program that was not native in the baseline (never a silent wrong result)
        wrong = [r for r in results if r['status'] == 'native-mismatch']
        for r in wrong:
            print(f"  native output differs from the VM: {r['id']} {r['label']} {r.get('reason', '')}")
        rc = 1 if worse or wrong else 0
    return rc


if __name__ == '__main__':
    sys.exit(main())
