#!/usr/bin/env python3
"""
Effects with resumption (docs/TIL_EFFECTS.md) together with the faster VM
(superinstructions 0x80-0xBF, docs/TIL_BYTECODE_V2.md): performs inside fused
loops, aborts out of FOR_ITER_STORE loops and through fused returns, resumes that
continue directly in a fused compare-and-branch, and clauses made of fused code.

Every program runs fused and unfused (--no-fuse) on the reference VM and the C VM
(several build profiles and interpreter variants): output, errors, exit codes and
step counts must be identical everywhere.

Run: python3 -m pytest tests/test_effects_fused.py -v
"""

import os
import re
import subprocess
import tempfile

import pytest

from tilhelpers import CC, VM_DIR, cvm_binary, needs_cc, run_cvm, til

Op = til.Op
EFFECT_OPS = (Op.HANDLE, Op.END_HANDLE, Op.PERFORM, Op.RESUME)


def build(src: str, fuse: bool = True) -> bytes:
    return til.compile_to_bytecode(src, "<test>", fuse=fuse).serialize()


def py_run(data: bytes, max_steps: int = 0):
    """Reference VM: (exit code, stdout, error report, steps)."""
    module = til.TBCModule.from_bytes(data)
    out, err = [], []
    vm = til.TILVM(module, write=out.append, max_steps=max_steps)
    vm.report = lambda msg, pc: err.append(msg + b"\n" + (vm.traceback(pc) + b"\n" if pc >= 0 else b""))
    rc = vm.run()
    return rc, b"".join(out).decode('utf-8', 'replace'), b"".join(err).decode('utf-8', 'replace'), vm.steps


def parse_stats(rc: int, out: bytes, err: bytes):
    err = err.decode('utf-8', 'replace')
    m = re.search(r'\[tilvm\] steps=(\d+)', err)
    return rc, out.decode('utf-8', 'replace'), re.sub(r'\[tilvm\] steps=.*\n?', '', err), \
        int(m.group(1)) if m else -1


def c_run(data: bytes, profile: str = "host", vm_args=()):
    """C VM with --stats: (exit code, stdout, stderr without the stats line, steps)."""
    rc, out, err = run_cvm(data, profile=profile, vm_args=('--stats',) + tuple(vm_args))
    return parse_stats(rc, out.encode(), err.encode())


def run_exe(exe: str, data: bytes, vm_args=()):
    fd, path = tempfile.mkstemp(suffix='.tbc')
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    try:
        r = subprocess.run([exe, '--stats', *vm_args, path], capture_output=True, timeout=120)
    finally:
        os.unlink(path)
    return parse_stats(r.returncode, r.stdout, r.stderr)


def same_everywhere(src: str, max_steps: int = 0):
    """fused == unfused on both VMs (output, error report, exit code, steps) and
    C VM == reference VM (output, exit code, steps). Returns the reference result."""
    fused, plain = build(src, True), build(src, False)
    assert any(op >= 0x80 for op in opcodes(fused)), "no superinstruction in the fused code"
    pf, pp = py_run(fused, max_steps), py_run(plain, max_steps)
    assert pf == pp, f"reference VM: fused {pf!r} != unfused {pp!r}"
    if CC:
        args = ('--max-steps', str(max_steps)) if max_steps else ()
        cf, cp = c_run(fused, vm_args=args), c_run(plain, vm_args=args)
        assert cf == cp, f"C VM: fused {cf!r} != unfused {cp!r}"
        assert (cf[0], cf[1], cf[3]) == (pf[0], pf[1], pf[3]), f"C VM {cf!r} != reference VM {pf!r}"
    return pf


def opcodes(data: bytes):
    m = til.TBCModule.from_bytes(data)
    seen = set()
    for f in m.funcs:
        seen |= {ins[1] for ins in til._decode_insns(m.code[f.offset:f.offset + len(f.code)])}
    return seen


def listing(src: str) -> str:
    return til.disassemble(til.compile_to_bytecode(src, "<test>"))


def after(text: str, opname: str):
    """Opcode names of the instructions that directly follow each `opname`."""
    rows = [line.split()[1] for line in text.splitlines() if re.match(r'\s+\d{4}  [A-Z]', line)]
    return [rows[i + 1] for i, name in enumerate(rows[:-1]) if name == opname]


# ═══════════════════════════════════════════════════════════════
# PROGRAMS (small enough for the 16-bit LEAN "tiny" profile)
# ═══════════════════════════════════════════════════════════════

# performs inside loops whose test, increment and accumulation are fused
# (JUMP_IF_NOT_LL / JUMP_IF_NOT_LI, INC_LOCAL, ADD_STORE_LOCAL, LOAD_LOCAL2), and
# resumes that land directly on a fused instruction
FUSED_LOOPS = '''
count_to(n: int) -> int
    var i = 0
    var total = 0
    while i < n
        total += perform Tick(i)
        i += 1
    return total

main()
    var log = 0
    let r = handle
        count_to(60)
    with
        Tick(k) =>
            log += 1
            resume(k * 2)
    print(r, log)
    var total = 0
    var j = 0
    let s = handle
        while j < 30
            total += perform Get(j) + (100 if j < 5 else 1)
            j += 1
        total
    with
        Get(v) => resume(v * 2)
    print(s)
    let hits = handle
        var h = 0
        var q = 0
        while q < 40
            if perform Get(q) < 10
                h += 1
            if q < perform Big(q)
                h += 2
            q += 1
        h
    with
        Get(v) => resume(v % 20)
        Big(v) => resume(v * 2 % 50)
    print(hits)
'''

# a clause that finishes without resume aborts the handle from inside a
# FOR_ITER_STORE loop; the abort lands on CAST_RETURN (typed function result).
# 300 rounds: nothing may leak (stack slots, handler entries, iterator locals).
ABORT_LOOPS = '''
first_index(xs: [int], target: int) -> int
    return handle
        var idx = 0
        for x in xs
            if x == target
                perform Found(idx)
            idx += 1
        -1
    with
        Found(i) => i * 10

sum_until(xs: [int], limit: int) -> int
    var total = 0
    let r = handle
        for x in xs
            total += x
            if total > limit
                perform Stop(total)
        total
    with
        Stop(t) => 0 - t
    return r

main()
    let xs = [5, 3, 9, 3, 7]
    var acc = 0
    for round in 0..300
        acc += first_index(xs, round % 10)
    print(acc)
    var neg = 0
    for round in 0..300
        let v = sum_until(xs, round % 30)
        if v < 0
            neg += 1
    print(neg, sum_until(xs, 12), sum_until(xs, 100))
    var broke = 0
    for round in 0..200
        for x in xs
            handle
                if x == 9
                    perform Skip()
                broke += x
            with
                Skip() => 0
        broke += 1
    print(broke)
'''

# `return` out of a handled body through RETURN_LOCAL / CAST_RETURN (the handle
# record of the returning frame is dropped), and `?` inside a handled body
RETURNS = '''
pick(xs: [int], want: int) -> int
    handle
        for x in xs
            let y = perform Map(x)
            if y == want
                return y
        0
    with
        Map(v) => resume(v + 1)
    return -1

find_local(xs: [int]) -> int
    let found = handle
        var best = 0
        for x in xs
            let y = perform Map(x)
            if y > best
                best = y
        best
    with
        Map(v) => resume(v * 3)
    return found

half(n: int) -> Result<int, str>
    if n % 2 == 1
        return Err("odd " + str(n))
    return Ok(n / 2)

halves(xs: [int]) -> Result<int, str>
    var t = 0
    handle
        for x in xs
            t += half(perform Twice(x))?
        t
    with
        Twice(v) => resume(v * 2 if v < 50 else v + 1)
    if t > 100
        return half(t + 1)
    return Ok(t)

main()
    let xs = [1, 4, 6, 2]
    var total = 0
    for round in 0..300
        total += pick(xs, round % 8)
    print(total)
    var t2 = 0
    for round in 0..100
        t2 += find_local(xs)
    print(t2)
    print(halves(xs))
    print(halves([40, 45, 30]))
    var errs = 0
    for round in 0..100
        let h = halves([2, 60, 4])
        errs += match h
            Ok(v) => 100
            Err(e) => 1
    print(errs)
    print(halves([60]))
'''

# clauses made of fused code: loops, field loads, indexing, print (NATIVE_POP),
# local arithmetic; an abort through RETURN_LOCAL (`Bail(x) => x`), nested
# handles (masking) and an inner perform inside a clause loop
CLAUSES = '''
struct P
    x: int
    y: int

main()
    let pts = [P{x: 1, y: 2}, P{x: 3, y: 4}, P{x: 5, y: 6}]
    var trace = 0
    let r = handle
        var acc = 0
        for i in 0..3
            acc += perform Sum(pts, i)
        let inner = handle
            var k = 0
            while k < 4
                acc += perform Sum(pts, k % 3)
                k += 1
            perform Bail(acc)
            999
        with
            Sum(ps, i) =>
                var s = 0
                for p in ps
                    s += p.x
                resume(s + perform Sum(ps, i))
            Bail(v) => v
        inner * 2
    with
        Sum(ps, i) =>
            let p = ps[i]
            var s = p.x * p.y
            var j = 0
            while j < i
                s += ps[j].y
                j += 1
            trace += 1
            print("sum", i, s)
            resume(s)
        Bail(v) => 0 - v
    print(r, trace)
'''

# runtime errors raised inside fused instructions in a handled body and in a
# clause: the recovery point around the handle drops its handler entries
ERRORS = '''
risky(xs: [int], i: int) -> int
    return handle
        var t = 0
        var k = 0
        while k <= i
            t += perform Get(xs, k)
            k += 1
        t
    with
        Get(a, k) => resume(a[k] * 2)

main()
    let xs = [1, 2, 3]
    var ok = 0
    var failed = 0
    for i in 0..6
        let r = retry(1)
            risky(xs, i)
        else
            failed += 1
            -1
        if r >= 0
            ok += r
    print(ok, failed)
    print(risky(xs, 7))
'''

PROGRAMS = {
    'fused_loops': (FUSED_LOOPS, "3540\n60\n1395\n68\n"),
    'abort_loops': (ABORT_LOOPS, None),
    'returns': (RETURNS, None),
    'clauses': (CLAUSES, None),
    'errors': (ERRORS, None),
}


# ═══════════════════════════════════════════════════════════════
# THE BYTECODE: fused code around and inside effects
# ═══════════════════════════════════════════════════════════════

class TestFusedBytecode:
    def test_the_programs_use_the_intended_superinstructions(self):
        loops = listing(FUSED_LOOPS)
        # resumes continue directly in fused instructions
        follow = set(after(loops, 'PERFORM'))
        assert {'JUMP_IF_NOT_LI', 'JUMP_IF_NOT_I', 'JUMP_IF_NOT', 'ADD_STORE_LOCAL'} <= follow, follow
        for name in ('JUMP_IF_NOT_LL', 'INC_LOCAL', 'LOAD_LOCAL2'):
            assert re.search(rf'\b{name}\b', loops), name
        aborts = listing(ABORT_LOOPS)
        assert 'FOR_ITER_STORE' in aborts
        # the abort target of `return handle ...` in a typed function is CAST_RETURN
        assert 'CAST_RETURN' in after(aborts, 'END_HANDLE')
        rets = listing(RETURNS)
        assert 'RETURN_LOCAL' in rets or 'CAST_RETURN' in rets
        clauses = listing(CLAUSES)
        clause_code = clauses[clauses.index('<clause '):]
        for name in ('FOR_ITER_STORE', 'LOAD_LOCAL_FIELD', 'NATIVE_POP', 'JUMP_IF_NOT_LL', 'INC_LOCAL',
                     'RETURN_LOCAL'):
            assert re.search(rf'\b{name}\b', clause_code), name

    @pytest.mark.parametrize("name", sorted(PROGRAMS))
    def test_effect_instructions_are_never_fused(self, name):
        """Every HANDLE/END_HANDLE/PERFORM/RESUME survives the pass unchanged, at the
        new pc of its old position; HANDLE abort targets and the resume points
        (the instruction after each PERFORM) are instruction starts there."""
        src = PROGRAMS[name][0]
        plain = til.compile_to_bytecode(src, "<test>", fuse=False)
        for f in plain.funcs:
            old = til._decode_insns(f.code)
            g = til.TBCFunction(f.name)
            g.code, g.lines = bytearray(f.code), list(f.lines)
            pc_map = til.fuse_superinstructions(g)
            if pc_map is None:
                assert not any(ins[1] in EFFECT_OPS for ins in old)
                continue
            new = {ins[0]: ins for ins in til._decode_insns(g.code)}
            for pc, op, vals, end in old:
                if op not in EFFECT_OPS:
                    continue
                ins = new[pc_map[pc]]
                assert ins[1] == op and (ins[2] == vals if op != Op.HANDLE else ins[2][0] == vals[0])
                assert pc_map[end] == ins[3], "an effect instruction was fused with its successor"
                assert pc_map[end] in new or pc_map[end] == len(g.code)
                if op == Op.HANDLE:
                    assert ins[3] + ins[2][1] == pc_map[end + vals[1]]

    def test_the_pass_refuses_effect_instructions_even_if_a_pattern_matched(self, monkeypatch):
        """Defence in depth: with a (hypothetical) rule LOAD_LOCAL b; RESUME -> <op>_L,
        the pass still leaves RESUME alone."""
        f = til.TBCFunction("t")
        f.n_locals = 1
        f.flags = til.TBCFunction.FLAG_EFFECT_CLAUSE
        f.code = bytearray([Op.LOAD_LOCAL, 0, Op.RESUME, Op.LOAD_LOCAL, 0, Op.ADD, Op.RETURN])
        monkeypatch.setitem(til._FUSED_INDEX, Op.RESUME, 0)
        til.fuse_superinstructions(f)
        ops = [ins[1] for ins in til._decode_insns(f.code)]
        assert ops[:2] == [Op.LOAD_LOCAL, Op.RESUME]
        assert Op.ADD_L in ops                       # ordinary fusion still happens


# ═══════════════════════════════════════════════════════════════
# BEHAVIOUR: fused == unfused, C VM == reference VM, steps included
# ═══════════════════════════════════════════════════════════════

class TestSameEverywhere:
    @pytest.mark.parametrize("name", sorted(PROGRAMS))
    def test_program(self, name):
        src, expected = PROGRAMS[name]
        rc, out, err, steps = same_everywhere(src)
        if name == 'errors':
            assert rc == 1 and "index out of bounds" in err
            assert out == "20\n3\n"
        else:
            assert rc == 0, err
            assert steps > 0
        if expected is not None:
            assert out == expected

    def test_expected_results(self):
        assert py_run(build(ABORT_LOOPS))[1] == "%d\n%d\n-17\n27\n%d\n" % (
            sum(({5: 0, 3: 10, 9: 20, 7: 40}.get(r % 10, -1)) for r in range(300)),
            sum(1 for r in range(300) if r % 30 < 27), 200 * (5 + 3 + 3 + 7 + 1))
        out = py_run(build(RETURNS))[1].splitlines()
        assert out[0] == str(sum(({2: 2, 5: 5, 7: 7, 3: 3}.get(r % 8, -1)) for r in range(300)))
        assert out[1] == str(100 * 18)
        assert out[2:] == ["Ok(13)", "Ok(58)", "100", 'Err("odd 61")']
        out = py_run(build(CLAUSES))[1].splitlines()
        assert out[-2:] == ["284", "7"]          # (2 + 14 + 36 + 11 + 23 + 45 + 11) * 2; 7 outer clause runs

    @pytest.mark.parametrize("limit", [1, 2, 7, 30, 61, 62, 63, 100, 155])
    def test_max_steps_inside_fused_loops_with_performs(self, limit):
        """The sandbox stops at the same step on both VMs, fused or not (performs
        count like calls, fused backward branches like JUMP_IF_FALSE)."""
        rc, out, err, steps = same_everywhere(FUSED_LOOPS, max_steps=limit)
        assert rc == 1 and "step limit exceeded" in err and steps == limit + 1

    def test_steps_count_performs_and_back_edges(self):
        src = ('main()\n    var i = 0\n    let r = handle\n        while i < 10\n'
               '            i += perform One()\n        i\n    with\n        One() => resume(1)\n    print(r)\n')
        rc, out, _err, steps = same_everywhere(src)
        assert (rc, out) == (0, "10\n")
        assert steps == 1 + 10 + 10          # main, 10 performs, 10 back-edges (resume/HANDLE: free)


@needs_cc
class TestCVM:
    @pytest.mark.parametrize("profile", ["tiny", "small", "gcstress"])
    def test_profiles(self, profile):
        for name in sorted(PROGRAMS):
            src = PROGRAMS[name][0]
            for fuse in (True, False):
                data = build(src, fuse)
                ref = py_run(data)
                got = c_run(data, profile=profile)
                assert (got[0], got[1], got[3]) == (ref[0], ref[1], ref[3]), f"{name} fuse={fuse}: {got[2]}"

    @pytest.mark.parametrize("variant", ["switch", "slow-paths", "notables", "handlers8"])
    def test_interpreter_variants(self, variant):
        """Switch dispatch, no integer fast paths (superinstruction slow paths), no
        function table / caches, and an explicit -DTILVM_MAX_HANDLERS: the same
        results and steps as the default build."""
        flags = {"switch": ['-DTILVM_COMPUTED_GOTO=0'], "slow-paths": ['-DTILVM_FAST_PATHS=0'],
                 "notables": ['-DTILVM_MAX_FUNCS=0', '-DTILVM_FIELD_CACHE=0'],
                 "handlers8": ['-DTILVM_MAX_HANDLERS=8']}[variant]
        exe = cvm_binary("host", flags + ['-O1'])
        for name in sorted(PROGRAMS):
            data = build(PROGRAMS[name][0])
            assert run_exe(exe, data) == c_run(data), name

    def test_snapshot_resume_at_every_step(self, tmp_path):
        """Raw snapshots taken at every step of fused code with handlers, running
        clauses and resumes in flight: part 1 + part 2 == an uninterrupted run."""
        src = FUSED_LOOPS.replace('count_to(60)', 'count_to(12)').replace('j < 30', 'j < 6') \
            .replace('q < 40', 'q < 6')
        exe = cvm_binary("host")
        prog = tmp_path / "p.tbc"
        prog.write_bytes(build(src))
        full = subprocess.run([exe, str(prog)], capture_output=True, timeout=60)
        assert full.returncode == 0
        assert full.stdout.decode() == py_run(build(src))[1]
        steps = c_run(build(src))[3]
        assert 40 < steps < 200
        snap = tmp_path / "s.snap"
        for n in range(1, steps):
            r1 = subprocess.run([exe, '--snapshot-after', str(n), str(snap), str(prog)],
                                capture_output=True, timeout=60)
            assert r1.returncode == 3, r1.stderr
            r2 = subprocess.run([exe, '--resume', str(snap), str(prog)], capture_output=True, timeout=60)
            assert r2.returncode == 0, r2.stderr
            assert r1.stdout + r2.stdout == full.stdout, f"after {n} steps"

    def test_no_effects_build_runs_fused_code_until_the_handle(self):
        """TILVM_NO_EFFECTS: fused code runs, `handle` reports the build explicitly."""
        exe = cvm_binary("host", ['-DTILVM_NO_EFFECTS', '-O1'])
        src = ('Log(n: int) -> int\n    return n * 2\nmain()\n    var t = 0\n    var i = 0\n'
               '    while i < 5\n        t += perform Log(i)\n        i += 1\n    print(t)\n'
               '    let r = handle\n        1\n    with\n        A() => 2\n    print(r)\n')
        for fuse in (True, False):
            rc, out, err, _steps = run_exe(exe, build(src, fuse))
            assert (rc, out) == (1, "20\n")
            assert "effect handlers are not available in this VM build" in err

    @pytest.mark.skipif(not os.environ.get('TIL_TEST_SLOW'), reason="slow: set TIL_TEST_SLOW=1")
    def test_sanitizers_with_a_small_heap(self):
        """AddressSanitizer + UBSan (aborting on the first report), 4 KB heap."""
        exe = cvm_binary("host", ['-g', '-O1', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                                  '-fno-omit-frame-pointer', '-DTILVM_HEAP_SIZE=4096'])
        for name in sorted(PROGRAMS):
            for fuse in (True, False):
                data = build(PROGRAMS[name][0], fuse)
                ref = py_run(data)
                got = run_exe(exe, data)
                assert "Sanitizer" not in got[2] and not re.search(r'\.c:\d+:\d+: runtime error', got[2]), got[2]
                assert (got[0], got[1], got[3]) == (ref[0], ref[1], ref[3]), f"{name}: {got[2]}"
