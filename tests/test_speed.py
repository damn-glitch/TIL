#!/usr/bin/env python3
"""
TIL 3.1 faster VM (docs/dev/TIL_3_1_DESIGN.md section 13): superinstructions
(opcodes 0x80-0xBF, docs/TIL_BYTECODE_V2.md), the peephole pass that emits them,
and the reworked C VM interpreter (computed goto / switch, function table,
lookup caches, step accounting).

Every program runs fused and unfused (--no-fuse) on both VMs; output, errors,
error lines, exit codes and step counts must not change.

Run: python3 -m pytest tests/test_speed.py -v
Performance check against TIL 3.0 (slow, needs git history): TIL_BENCH=1
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

from tilhelpers import CC, ROOT, VM_DIR, cli, cvm_binary, needs_cc, run_cvm, til

Op = til.Op
C_SOURCE = open(os.path.join(VM_DIR, 'tilvm.c'), encoding='utf-8').read()
EXAMPLES = sorted(glob.glob(os.path.join(ROOT, 'examples', '**', '*.til'), recursive=True))
BENCHMARKS = sorted(glob.glob(os.path.join(ROOT, 'benchmarks', '*.til')))
SUPER = sorted(v for k, v in vars(Op).items() if not k.startswith('_') and v >= 0x80)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def build(src: str, fuse: bool = True, filename: str = "<test>") -> bytes:
    return til.compile_to_bytecode(src, filename, fuse=fuse).serialize()


def py_run(data: bytes, max_steps: int = 0):
    """Reference VM: (exit code, stdout, error report, steps)."""
    module = til.TBCModule.from_bytes(data)
    out, err = [], []
    vm = til.TILVM(module, write=out.append, max_steps=max_steps)
    vm.report = lambda msg, pc: err.append(msg + b"\n" + (vm.traceback(pc) + b"\n" if pc >= 0 else b""))
    rc = vm.run()
    return rc, b"".join(out).decode('utf-8', 'replace'), b"".join(err).decode('utf-8', 'replace'), vm.steps


def c_run(data: bytes, profile: str = "host", vm_args=()):
    """C VM with --stats: (exit code, stdout, stderr without the stats line, steps)."""
    rc, out, err = run_cvm(data, profile=profile, vm_args=('--stats',) + tuple(vm_args))
    m = re.search(r'\[tilvm\] steps=(\d+)', err)
    steps = int(m.group(1)) if m else -1
    err = re.sub(r'\[tilvm\] steps=.*\n?', '', err)
    return rc, out, err, steps


def opcodes(data: bytes):
    m = til.TBCModule.from_bytes(data)
    found = set()
    for f in m.funcs:
        for ins in til._decode_insns(f.code):
            found.add(ins[1])
    return found


def assert_same_everywhere(src: str, filename: str = "<test>", max_steps: int = 0):
    """fused == unfused on the reference VM and on the C VM (output, error report,
    exit code, steps), and the C VM == the reference VM (output, exit code, steps).
    Programs with timeout(...) blocks run until a clock deadline: their step
    counts depend on the speed of the VM and are not compared."""
    timed = 'timeout(' in src
    keep = (lambda r: r[:3]) if timed else (lambda r: r)
    fused, plain = build(src, True, filename), build(src, False, filename)
    pf, pp = py_run(fused, max_steps), py_run(plain, max_steps)
    assert keep(pf) == keep(pp), f"reference VM: fused {pf!r} != unfused {pp!r}"
    if CC:
        args = ('--max-steps', str(max_steps)) if max_steps else ()
        cf, cp = c_run(fused, vm_args=args), c_run(plain, vm_args=args)
        assert keep(cf) == keep(cp), f"C VM: fused {cf!r} != unfused {cp!r}"
        assert (cf[0], cf[1]) == (pf[0], pf[1]), f"C VM {cf!r} != reference VM {pf!r}"
        if not timed:
            assert cf[3] == pf[3], f"steps: C VM {cf[3]} != reference VM {pf[3]}"
    return pf


# ═══════════════════════════════════════════════════════════════
# CORPUS: programs that exercise every superinstruction, with the int fast
# paths and the general paths (floats, strings, mixed types, errors)
# ═══════════════════════════════════════════════════════════════

CORPUS = {
    'counters': '''
main()
    var i = 0
    var total = 0
    while i < 300
        total += i
        i += 1
    var f = 1.5
    f += 1
    var s = "x"
    s += 1
    var big = 9223372036854775807
    big += 1
    print(total, f, s, big)
    for k in 0..=5
        total -= k
    print(total)
''',
    'compare_branches': '''
check(a, b)
    var r = ""
    if a < b
        r += "<"
    if a <= b
        r += "<="
    if a > b
        r += ">"
    if a >= b
        r += ">="
    if a == b
        r += "=="
    if a != b
        r += "!="
    return r

main()
    print(check(1, 2), check(2, 2), check(3, 2))
    print(check(1.5, 2), check(2, 2.0), check("b", "a"), check("a", "a"))
    let nan = 0.0 / 0.0
    print(check(nan, 1.0), check(1, nan))
    let xs = [1, 2]
    let ys = [1, 2]
    if xs == ys
        print("arrays equal")
    if xs != [2]
        print("arrays differ")
    var n = 0
    for i in 0..40
        if i % 3 == 0
            n += 1
        if i < 10
            n += 2
        if 100 <= i
            n = 0
        if i != 7
            n += 1
    print(n)
    let x = 5
    if x < 300
        print("i16 immediate")
    if x >= -1000
        print("negative immediate")
''',
    'compare_errors': '''
main()
    let r = retry(1)
        let a = "text"
        if a < 3
            print("never")
        "unreachable"
    else
        error
    print(r)
    var xs = [3, 1, 2]
    let q = retry(1)
        if xs[0] > "a"
            print("never")
        "no"
    else
        error
    print(q)
''',
    'binops_immediate': '''
main()
    let a = 100
    let f = 7.5
    print(a + 3, a - 3, a * 3, a / 3, a % 3, a & 6, a | 3, a ^ 5, a << 2, a >> 2)
    print(a == 100, a != 100, a < 3, a <= 100, a > 3, a >= 101)
    print(-a / 7, -a % 7, a * 1000, a % 1000, a & 255, a + 20000)
    print(f + 1, f - 1, f * 2, f / 2, f % 2, f < 8, f == 7.5)
    print("ab" + 1, "ab" * 3)
    let e = retry(1)
        a / 0
    else
        error
    print(e)
    let m = retry(1)
        a % 0
    else
        error
    print(m)
    let b = retry(1)
        f & 1
    else
        error
    print(b)
    let t = retry(1)
        "s" - 1
    else
        error
    print(t)
''',
    'binops_local': '''
main()
    let a = 77
    let b = 5
    let z = 0
    let f = 2.5
    let s = "str"
    print(a + b, a - b, a * b, a / b, a % b, a & b, a | b, a ^ b, a << b, a >> b)
    print(a == b, a != b, a < b, a <= b, a > b, a >= b)
    print(f + b, b - f, f * b, b / f, a % f, f < b, f == f)
    print(s + b, s * b, b + s)
    let e = retry(1)
        a / z
    else
        error
    print(e)
    let m = retry(1)
        a % z
    else
        error
    print(m)
    let x = retry(1)
        s & b
    else
        error
    print(x)
    var w = 9223372036854775807
    var one = 1
    print(w + one, w * a)
''',
    'casts': '''
to_int(x: int) -> int
    return x

to_float(x: float) -> float
    return x

half(x: int) -> float
    return x / 2

whole(x: float) -> int
    return x * 2

main()
    print(to_int(3), to_int(2.9), to_int("12"), to_int(true))
    print(to_float(3), to_float(2.5), to_float("1.25"))
    print(half(7), whole(1.75))
    let e = retry(1)
        to_int("abc")
    else
        error
    print(e)
''',
    'locals_and_fields': '''
struct P
    x: int
    y: int
    name: str

impl P
    sum(self) -> int
        return self.x + self.y
    rename(self, n: str)
        self.name = n

struct Q
    name: str
    x: float

main()
    var p = P { x: 3, y: 4, name: "p" }
    let q = Q { name: "q", x: 1.5 }
    print(p.x, p.y, p.name, q.x, q.name, p.sum())
    p.rename("pp")
    p.x += 10
    print(p.x, p.name)
    var total = 0
    for i in 0..50
        let a = P { x: i, y: i * 2, name: "" }
        total += a.x + a.y
    print(total)
    let n = 5
    let e = retry(1)
        n.x
    else
        error
    print(e)
''',
    'indexing': '''
main()
    var xs = [10, 20, 30, 40]
    var i = 2
    print(xs[i], xs[0], xs[3])
    let grid = [[1, 2, 3], [4, 5, 6]]
    print(grid[1][i], grid[0][i - 1])
    let s = "hello"
    print(s[i], s[0])
    var m = {"a": 1, 2: "two"}
    let k = "a"
    print(m[k], m[2])
    var total = 0
    for j in 0..len(xs)
        total += xs[j]
    print(total)
    let e = retry(1)
        xs[i * 10]
    else
        error
    print(e)
    let f = retry(1)
        xs[-1]
    else
        error
    print(f)
    let g = retry(1)
        m["zz"]
    else
        error
    print(g)
''',
    'iteration': '''
main()
    var total = 0
    for x in [1, 2, 3, 4]
        total += x
    for c in "abc"
        total += c
    var keys = ""
    for k in {"x": 1, "y": 2}
        keys += k
    var nested = 0
    for row in [[1, 2], [3, 4]]
        for v in row
            nested += v
    print(total, keys, nested)
    let fs = []
    for v in [1, 2, 3]
        fs.push(|n: int| n + v)
    print(fs[0](10), fs[2](10))
''',
    'returns': '''
first(xs) -> int
    return xs[0]

ident(v)
    return v

pick(a: int, b: int) -> int
    if a > b
        return a
    return b

main()
    print(first([7, 8]), ident("s"), ident(None), pick(3, 9), pick(9, 3))
''',
    'natives_and_pop': '''
main()
    print("a")
    println("b")
    let xs = [3, 1, 2]
    print(len(xs), min(xs), max(xs), sum(xs), str(42), int("7"), abs(-3))
    print(format(3.14159, ".2f"), type_of(xs), chr(65), ord("A"))
''',
    'closures': '''
make_counter()
    var n = 10
    return |step: int| n + step

adder(k: int)
    return |x: int| x + k

main()
    let c = make_counter()
    print(c(1), c(2))
    let add5 = adder(5)
    var acc = 0
    for i in 0..10
        acc += add5(i)
    print(acc)
    let xs = [1, 2, 3, 4, 5, 6]
    print(xs.map(|x| x * x).filter(|x| x % 2 == 0).reduce(|a, b| a + b, 0))
''',
    'recursion': '''
fib(n: int) -> int
    if n < 2
        return n
    return fib(n - 1) + fib(n - 2)

ack(m: int, n: int) -> int
    if m == 0
        return n + 1
    if n == 0
        return ack(m - 1, 1)
    return ack(m - 1, ack(m, n - 1))

main()
    print(fib(18), ack(2, 3))
''',
    'error_line_in_loop': '''
main()
    var xs = [1, 2, 3]
    var i = 0
    var total = 0
    while i < 5
        total += xs[i]
        i += 1
    print(total)
''',
    'error_in_fused_compare': '''
check(v)
    if v < 10
        return "small"
    return "big"

main()
    print(check(3))
    print(check("x"))
''',
    'error_in_cast': '''
need_int(x: int) -> int
    return x + 1

main()
    print(need_int(1))
    print(need_int("nope"))
''',
    'deep_recursion': '''
down(n: int) -> int
    if n == 0
        return 0
    return down(n - 1) + 1

main()
    print(down(100))
    print(down(900))
''',
    'retry_and_timeout': '''
var attempts = 0

flaky() -> int
    attempts += 1
    assert attempts >= 3, "warming up"
    return 42

main()
    let v = retry(5)
        flaky()
    print(v, attempts)
    let t = timeout(30)
        var i = 0
        loop
            i += 1
        i
    else
        "stopped: " + error
    print(t)
''',
    'match_and_options': '''
enum Shape
    Circle(float)
    Rect(float, float)
    Dot

area(s: Shape) -> float
    return match s
        Shape.Circle(r) => 3.0 * r * r
        Shape.Rect(w, h) => w * h
        Shape.Dot => 0.0

parse(s: str) -> Result<int, str>
    if s == ""
        return Err("empty")
    return Ok(s.to_int())

twice(s: str) -> Result<int, str>
    let v = parse(s)?
    return Ok(v * 2)

main()
    for s in [Shape.Circle(1.0), Shape.Rect(2.0, 3.0), Shape.Dot]
        print(area(s))
    for t in ["21", ""]
        match twice(t)
            Ok(v) => print(v)
            Err(e) => print("err " + e)
    for n in [1, 5, 50, 500]
        let label = match n
            1 => "one"
            2..10 => "few"
            _ => "many"
        print(label)
''',
    'default_args': '''
power(base: int, exp: int = 2) -> int
    var r = 1
    for i in 0..exp
        r = r * base
    return r

main()
    print(power(3), power(2, 10))
''',
    'strings_and_maps': '''
main()
    var words = "tea coffee tea juice tea coffee".split(" ")
    var votes = {}
    for w in words
        votes[w] = votes.get_or(w, 0) + 1
    for k in votes
        print(f"{k}={votes[k]}")
    var acc = ""
    for i in 0..30
        acc = acc + str(i % 10)
    print(acc.find("789"), acc.contains("0123"), acc.count("5"), len(acc))
    print(f"{1}-{2.5}-{true}-{None}-{[1, 2]}-{acc.slice(0, 3)}")
''',
    'globals': '''
var counter = 0
let LIMIT = 50

bump(k: int)
    counter += k

main()
    for i in 0..LIMIT
        bump(i)
    print(counter)
''',
}

ERROR_PROGRAMS = {'error_line_in_loop', 'error_in_fused_compare', 'error_in_cast'}


# ═══════════════════════════════════════════════════════════════
# OPCODE TABLES
# ═══════════════════════════════════════════════════════════════

class TestOpcodeTables:
    def test_superinstructions_are_in_the_reserved_range(self):
        assert SUPER and all(0x80 <= op <= 0xBF for op in SUPER)
        for op in SUPER:
            assert op in til.OP_NAMES and op in til.OP_OPERANDS

    def test_fused_binop_families(self):
        assert len(til.FUSED_BINOPS) == 16
        for i, base in enumerate(til.FUSED_BINOPS):
            name = til.OP_NAMES[base]
            assert til.OP_NAMES[0xA0 + i] == name + "_I" and til.OP_OPERANDS[0xA0 + i] == 's'
            assert til.OP_NAMES[0xB0 + i] == name + "_L" and til.OP_OPERANDS[0xB0 + i] == 'b'

    def test_c_enum_and_tables_match(self):
        enum = dict((int(v, 16), n) for n, v in re.findall(r'OP_(\w+) = (0x[0-9A-Fa-f]+)', C_SOURCE))
        for op in SUPER:
            assert enum.get(op) == til.OP_NAMES[op], hex(op)
        table = re.search(r'FUSED_BINOP\[16\] = \{(.*?)\};', C_SOURCE, re.S).group(1)
        assert re.findall(r'OP_(\w+)', table) == [til.OP_NAMES[o] for o in til.FUSED_BINOPS]

    def test_computed_goto_table(self):
        """dispatch[256]: entry i is &&L_OP_<name of opcode i> or &&L_BAD."""
        body = re.search(r'dispatch\[256\] = \{(.*?)\};', C_SOURCE, re.S).group(1)
        labels = re.findall(r'&&L_(\w+)', body)
        assert len(labels) == 256
        for i, label in enumerate(labels):
            if label != 'BAD':
                assert label == 'OP_' + til.OP_NAMES.get(i, '?'), hex(i)
        for op in SUPER:
            assert labels[op] != 'BAD', f"superinstruction 0x{op:02x} missing from dispatch[]"
        handlers = set(re.findall(r'CASE\(OP_(\w+)\)', C_SOURCE))
        for i, label in enumerate(labels):
            if label != 'BAD':
                assert label[3:] in handlers, label


# ═══════════════════════════════════════════════════════════════
# PEEPHOLE PASS
# ═══════════════════════════════════════════════════════════════

class TestPeephole:
    def test_disassembler_shows_superinstructions(self):
        src = CORPUS['compare_branches'] + '\n' + CORPUS['casts'].replace('main()', 'main2()')
        listing = til.disassemble(til.compile_to_bytecode(src, "<test>"))
        for name in ('JUMP_IF_NOT_L', 'JUMP_IF_NOT_LI', 'CAST_LOCAL', 'CAST_RETURN', 'INC_LOCAL',
                     'ADD_STORE_LOCAL', 'MOD_I', 'NATIVE_POP'):
            assert re.search(rf'\b{name}\b', listing), name
        assert re.search(r'JUMP_IF_NOT_LI +\d+ \d+ \d+ -?\d+ +; (LT|LE|GT|GE|EQ|NE) -> \d{4}', listing)
        assert re.search(r'CAST_LOCAL +\d+ 0 +; int', listing)

    def test_cli_dis_and_no_fuse(self, tmp_path):
        prog = tmp_path / "f.til"
        prog.write_text(CORPUS['recursion'], encoding='utf-8')
        fused = cli('dis', str(prog))
        plain = cli('dis', '--no-fuse', str(prog))
        assert fused.returncode == 0 and plain.returncode == 0
        assert 'CAST_LOCAL' in fused.stdout and 'JUMP_IF_NOT' in fused.stdout
        assert 'CAST_LOCAL' not in plain.stdout and 'JUMP_IF_NOT' not in plain.stdout
        out = tmp_path / "f.tbc"
        assert cli('build', '--target', 'vm', '--no-fuse', str(prog), '-o', str(out)).returncode == 0
        assert not any(op >= 0x80 for op in opcodes(out.read_bytes()))
        assert cli('build', '--target', 'vm', str(prog), '-o', str(out)).returncode == 0
        assert any(op >= 0x80 for op in opcodes(out.read_bytes()))
        r = cli('run', '--vm', '--no-fuse', str(prog))
        assert r.returncode == 0 and r.stdout == "2584\n9\n"

    def test_every_superinstruction_is_emitted(self):
        """The corpus + examples + benchmarks use every superinstruction."""
        seen = set()
        for src in CORPUS.values():
            seen |= opcodes(build(src))
        for path in EXAMPLES + BENCHMARKS:
            seen |= opcodes(build(open(path, encoding='utf-8').read(), filename=path))
        missing = [til.OP_NAMES[op] for op in SUPER if op not in seen]
        assert not missing, missing

    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_fused_code_is_smaller_and_lines_are_kept(self, name):
        plain = til.compile_to_bytecode(CORPUS[name], "<test>", fuse=False)
        fused = til.compile_to_bytecode(CORPUS[name], "<test>", fuse=True)
        assert len(fused.code) <= len(plain.code)
        for fp, ff in zip(plain.funcs, fused.funcs):
            n_plain, n_fused = len(til._decode_insns(fp.code)), len(til._decode_insns(ff.code))
            assert n_fused <= n_plain
            # the same lines, in the same order (a dropped `JUMP 0` may merge two entries)
            seq = [line for _pc, line in ff.lines]
            plain_seq = [line for _pc, line in fp.lines]
            it = iter(plain_seq)
            assert all(any(line == p for p in it) for line in seq), (seq, plain_seq)

    def test_pass_is_idempotent_and_keeps_targets(self):
        """Fusing already fused code changes nothing; group starts keep jump targets."""
        m = til.compile_to_bytecode(CORPUS['compare_branches'], "<test>", fuse=True)
        for f in m.funcs:
            code, lines = bytes(f.code), list(f.lines)
            til.fuse_superinstructions(f)
            assert bytes(f.code) == code and f.lines == lines

    def test_never_fuses_across_a_jump_target(self):
        """LOAD_LOCAL 0; [target:] PUSH_I8 1; ADD; STORE_LOCAL 0 must not become INC_LOCAL."""
        f = til.TBCFunction("t")
        f.n_locals = 1
        code = bytearray()
        code += bytes([Op.PUSH_I8, 0, Op.STORE_LOCAL, 0])
        code += bytes([Op.LOAD_LOCAL, 0])
        target = len(code)
        code += bytes([Op.PUSH_I8, 1, Op.ADD, Op.STORE_LOCAL, 0])
        jump_at = len(code)
        rel = target - (jump_at + 3)
        code += bytes([Op.JUMP]) + rel.to_bytes(2, 'little', signed=True)
        code += bytes([Op.RETURN_NIL])
        f.code = code
        pc_map = til.fuse_superinstructions(f)
        ops = [ins[1] for ins in til._decode_insns(f.code)]
        assert Op.INC_LOCAL not in ops
        assert len(ops) == 7                       # PUSH_I8 1; ADD; STORE_LOCAL 0 -> two instructions
        starts = [ins[0] for ins in til._decode_insns(f.code)]
        jmp = [ins for ins in til._decode_insns(f.code) if ins[1] == Op.JUMP][0]
        assert jmp[3] + jmp[2][0] == pc_map[target] and pc_map[target] in starts

    def test_never_fuses_across_a_line_entry(self):
        f = til.TBCFunction("t")
        f.n_locals = 2
        f.code = bytearray([Op.LOAD_LOCAL, 0, Op.LOAD_LOCAL, 1, Op.ADD, Op.POP, Op.RETURN_NIL])
        f.lines = [(0, 1), (2, 2)]                 # the second LOAD_LOCAL starts line 2
        til.fuse_superinstructions(f)
        ops = [ins[1] for ins in til._decode_insns(f.code)]
        assert ops[0] == Op.LOAD_LOCAL and Op.LOAD_LOCAL2 not in ops
        assert Op.ADD_L in ops                     # LOAD_LOCAL 1; ADD (both on line 2)
        assert f.lines == [(0, 1), (2, 2)]


# ═══════════════════════════════════════════════════════════════
# DIFFERENTIAL: fused == unfused, C VM == reference VM
# ═══════════════════════════════════════════════════════════════

class TestSameBehaviour:
    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_corpus(self, name):
        rc, out, err, steps = assert_same_everywhere(CORPUS[name])
        if name in ERROR_PROGRAMS:
            assert rc == 1 and "error" in err
        else:
            assert rc == 0, err
            assert out

    def test_error_lines(self):
        """Errors inside fused instructions report the line of the original instruction."""
        _rc, _out, err, _s = py_run(build(CORPUS['error_line_in_loop']))
        assert "index out of bounds" in err and "(line 7)" in err
        _rc, _out, err, _s = py_run(build(CORPUS['error_in_fused_compare']))
        assert "cannot compare str and int" in err and "(line 3)" in err
        _rc, _out, err, _s = py_run(build(CORPUS['error_in_cast']))
        # parameter conversions run before the first line entry (fused or not)
        assert 'invalid integer: "nope"' in err and "at need_int\n  at main (line 7)" in err
        if CC:
            for name, where in (('error_line_in_loop', "at main (line 7)"), ('error_in_fused_compare',
                                "at check (line 3)"), ('error_in_cast', "at need_int\n  at main (line 7)")):
                rc, _out, cerr, _s = c_run(build(CORPUS[name]))
                assert rc == 1 and where in cerr, cerr

    @pytest.mark.parametrize("path", EXAMPLES, ids=[os.path.relpath(p, ROOT) for p in EXAMPLES])
    def test_examples(self, path):
        rc, out, err, _steps = assert_same_everywhere(open(path, encoding='utf-8').read(), filename=path)
        assert rc == 0, err

    @needs_cc
    @pytest.mark.parametrize("path", BENCHMARKS, ids=[os.path.basename(p) for p in BENCHMARKS])
    def test_benchmarks_on_c_vm(self, path):
        src = open(path, encoding='utf-8').read()
        cf = c_run(build(src, True, path))
        cp = c_run(build(src, False, path))
        assert cf == cp and cf[0] == 0


# ═══════════════════════════════════════════════════════════════
# STEP COUNTS, SANDBOX LIMITS, SNAPSHOTS
# ═══════════════════════════════════════════════════════════════

LOOP_PROGRAM = '''
main()
    var total = 0
    for i in 0..20
        for j in 0..i
            total += j
        print(total)
'''


SNAPSHOT_PROGRAM = '''
fib(n: int) -> int
    if n < 2
        return n
    return fib(n - 1) + fib(n - 2)

main()
    var total = 0
    for i in 0..20
        for j in 0..i
            total += j
        print(total, fib(i % 8))
'''


class TestSteps:
    @pytest.mark.parametrize("name", ['counters', 'recursion', 'closures', 'iteration', 'match_and_options'])
    def test_step_counts_identical(self, name):
        _rc, _out, _err, steps = assert_same_everywhere(CORPUS[name])
        assert steps > 0

    @pytest.mark.parametrize("limit", [1, 2, 25, 77, 150, 209, 210, 211, 400])
    def test_max_steps(self, limit):
        """--max-steps stops at the same step, with the same output and error line,
        fused or not, on both VMs (steps: 1 call to main + 209 back-edges + 1 = 211)."""
        rc, out, err, steps = assert_same_everywhere(LOOP_PROGRAM, max_steps=limit)
        if limit < 211:
            assert rc == 1 and "step limit exceeded" in err and steps == limit + 1
        else:
            assert rc == 0 and out.count("\n") == 20

    @needs_cc
    def test_cli_max_steps(self, tmp_path):
        prog = tmp_path / "l.til"
        prog.write_text(LOOP_PROGRAM, encoding='utf-8')
        runs = [cli('run', vm, *fuse, '--max-steps', '100', str(prog))
                for vm in ('--vm', '--cvm') for fuse in ((), ('--no-fuse',))]
        assert all(r.returncode == 1 for r in runs)
        assert len({r.stdout for r in runs}) == 1
        assert all("step limit exceeded" in r.stderr for r in runs)

    def test_backward_fused_branch_counts_steps(self):
        """A hand-written loop whose back-edge is a fused compare-and-branch: same
        steps and same sandbox stop as the unfused sequence (both VMs)."""
        enc = til._encode_insn

        def module(fused):
            head = enc(Op.PUSH_I8, [0]) + enc(Op.STORE_LOCAL, [0])
            if fused:
                body = enc(Op.INC_LOCAL, [0, 1])
                branch = lambda rel: enc(Op.JUMP_IF_NOT_LI, [Op.GE, 0, 100, rel])
            else:
                body = enc(Op.LOAD_LOCAL, [0]) + enc(Op.PUSH_I8, [1]) + enc(Op.ADD, []) + \
                    enc(Op.STORE_LOCAL, [0]) + enc(Op.LOAD_LOCAL, [0]) + enc(Op.PUSH_I8, [100]) + enc(Op.GE, [])
                branch = lambda rel: enc(Op.JUMP_IF_FALSE, [rel])
            size = len(branch(0))
            back = -(len(body) + size)                  # to the start of `body`
            tail = enc(Op.LOAD_LOCAL, [0]) + enc(Op.NATIVE, [til.NATIVE_ID['print'], 1]) + \
                enc(Op.POP, []) + enc(Op.RETURN_NIL, [])
            m = til.TBCModule()
            f = til.TBCFunction("main")
            f.n_locals = 1
            f.code = bytearray(head + body + branch(back) + tail)
            f.lines = [(0, 1)]
            m.funcs.append(f)
            start = til.TBCFunction("__start__")
            start.code = bytearray(enc(Op.CALL, [0, 0]) + enc(Op.POP, []) + enc(Op.HALT, []))
            m.funcs.append(start)
            m.entry = 1
            return m.serialize()
        for limit in (100000, 50):
            pf, pp = py_run(module(True), limit), py_run(module(False), limit)
            assert pf == pp
            if limit == 50:
                assert pf[0] == 1 and pf[3] == limit + 1
            else:
                assert pf[1] == "100\n" and pf[3] == 100       # 1 call + 99 back-edges
            if CC:
                args = ('--max-steps', str(limit))
                cf, cp = c_run(module(True), vm_args=args), c_run(module(False), vm_args=args)
                assert cf == cp and (cf[0], cf[1], cf[3]) == (pf[0], pf[1], pf[3])

    @needs_cc
    @pytest.mark.parametrize("pause", [1, 3, 50, 199, 200, 201, 333])
    def test_snapshot_resume(self, tmp_path, pause):
        """Raw snapshots: pause after N steps, save, resume - fused and unfused give
        the same split of the output, and part1 + part2 == an uninterrupted run."""
        src = SNAPSHOT_PROGRAM
        vm = cvm_binary()
        parts = []
        for fuse in (True, False):
            tbc = tmp_path / f"p{int(fuse)}.tbc"
            tbc.write_bytes(build(src, fuse))
            full = subprocess.run([vm, str(tbc)], capture_output=True).stdout
            snap = tmp_path / f"s{int(fuse)}.snap"
            p1 = subprocess.run([vm, '--snapshot-after', str(pause), str(snap), str(tbc)], capture_output=True)
            if p1.returncode == 3:
                p2 = subprocess.run([vm, '--resume', str(snap), str(tbc)], capture_output=True)
                assert p2.returncode == 0, p2.stderr
                assert p1.stdout + p2.stdout == full
                parts.append((p1.stdout, p2.stdout))
            else:
                assert p1.returncode == 0 and p1.stdout == full
                parts.append((p1.stdout, b""))
        assert parts[0] == parts[1]

    @needs_cc
    def test_timeouts_still_fire(self):
        src = ('main()\n    let r = timeout(20)\n        var i = 0\n        loop\n            i += 1\n'
               '        i\n    else\n        "stopped: " + error\n    print(r)\n')
        for fuse in (True, False):
            rc, out, _err, _s = py_run(build(src, fuse))
            assert (rc, out) == (0, "stopped: timeout\n")
            crc, cout, cerr, _s = c_run(build(src, fuse))
            assert (crc, cout) == (0, "stopped: timeout\n"), cerr


# ═══════════════════════════════════════════════════════════════
# AGAINST TIL 3.0 (commit 6f4d00b, when the git history is available)
# ═══════════════════════════════════════════════════════════════

BASELINE_REV = os.environ.get('TIL_BENCH_BASELINE', '6f4d00b')
_BASELINE = {}


def baseline():
    """TIL 3.0 sources (git show) + its desktop C VM; skips without git history."""
    if 'dir' not in _BASELINE:
        base = tempfile.mkdtemp(prefix='til30-')
        for rel in ('src/til.py', 'vm/tilvm.c', 'vm/tilvm.h', 'vm/main.c'):
            r = subprocess.run(['git', '-C', ROOT, 'show', f'{BASELINE_REV}:{rel}'], capture_output=True)
            if r.returncode != 0:
                _BASELINE['dir'] = None
                break
            os.makedirs(os.path.dirname(os.path.join(base, rel)), exist_ok=True)
            with open(os.path.join(base, rel), 'wb') as f:
                f.write(r.stdout)
        else:
            exe = os.path.join(base, 'tilvm-3.0')
            r = subprocess.run([CC, '-std=c99', '-O1', '-o', exe, os.path.join(base, 'vm', 'tilvm.c'),
                                os.path.join(base, 'vm', 'main.c'), '-lm'], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            _BASELINE['dir'], _BASELINE['exe'] = base, exe
    if _BASELINE['dir'] is None:
        pytest.skip(f"git history with {BASELINE_REV} is not available")
    return _BASELINE['dir'], _BASELINE['exe']


@needs_cc
class TestAgainstTil30:
    PROGRAMS = ('counters', 'recursion', 'closures', 'iteration', 'locals_and_fields', 'compare_branches')

    def _old_tbc(self, tmp_path, name, src):
        base, _exe = baseline()
        path = tmp_path / f"{name}.til"
        path.write_text(src, encoding='utf-8')
        out = tmp_path / f"{name}.old.tbc"
        r = subprocess.run([sys.executable, os.path.join(base, 'src', 'til.py'), 'build', '--target', 'vm',
                            str(path), '-o', str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return out.read_bytes()

    @pytest.mark.parametrize("name", PROGRAMS)
    def test_no_fuse_is_the_til_3_0_bytecode(self, tmp_path, name):
        """--no-fuse output is byte-identical to what the TIL 3.0 compiler wrote."""
        assert build(CORPUS[name], False, "<test>") != b""
        old = self._old_tbc(tmp_path, name, CORPUS[name])
        new = til.compile_to_bytecode(CORPUS[name], str(tmp_path / f"{name}.til"), fuse=False).serialize()
        assert new == old

    @pytest.mark.parametrize("name", PROGRAMS)
    def test_steps_and_limits_as_in_til_3_0(self, tmp_path, name):
        """Steps (--stats), --max-steps stops and snapshot pause points of the TIL 3.0
        C VM are those of the new C VM, fused or not."""
        _base, old_exe = baseline()
        old_tbc = self._old_tbc(tmp_path, name, CORPUS[name])
        fused = build(CORPUS[name])
        ref = run_exe(old_exe, old_tbc)
        assert c_run(old_tbc) == ref                    # new VM, TIL 3.0 bytecode
        assert c_run(fused) == ref                      # new VM, superinstructions
        steps = ref[3]
        for limit in sorted({1, steps // 3, steps // 2, steps - 1}):
            if limit <= 0:
                continue
            old = subprocess.run([old_exe, '--max-steps', str(limit), str(tmp_path / f"{name}.old.tbc")],
                                 capture_output=True)
            new = c_run(fused, vm_args=('--max-steps', str(limit)))
            assert (new[0], new[1]) == (old.returncode, old.stdout.decode('utf-8', 'replace'))
            snap_old, snap_new = tmp_path / "o.snap", tmp_path / "n.snap"
            (tmp_path / "f.tbc").write_bytes(fused)
            p_old = subprocess.run([old_exe, '--snapshot-after', str(limit), str(snap_old),
                                    str(tmp_path / f"{name}.old.tbc")], capture_output=True)
            p_new = subprocess.run([cvm_binary(), '--snapshot-after', str(limit), str(snap_new),
                                    str(tmp_path / "f.tbc")], capture_output=True)
            assert (p_new.returncode, p_new.stdout) == (p_old.returncode, p_old.stdout)


# ═══════════════════════════════════════════════════════════════
# C VM BUILD VARIANTS
# ═══════════════════════════════════════════════════════════════

VARIANTS = {
    "switch": ['-DTILVM_COMPUTED_GOTO=0'],
    "slow-paths": ['-DTILVM_FAST_PATHS=0'],
    "notables": ['-DTILVM_MAX_FUNCS=0', '-DTILVM_FIELD_CACHE=0'],
    "small-tables": ['-DTILVM_FIELD_CACHE=2', '-DTILVM_MAX_FUNCS=64'],
}
_VARIANT_EXE = {}


def variant_exe(name: str) -> str:
    """The desktop VM built with VARIANTS[name] and the strict flags (once per session)."""
    if name not in _VARIANT_EXE:
        exe = os.path.join(tempfile.mkdtemp(prefix='tilvm-var-'), 'tilvm-' + name)
        r = subprocess.run([CC, '-std=c99', '-Wall', '-Wextra', '-pedantic', '-Werror', *VARIANTS[name],
                            '-o', exe, os.path.join(VM_DIR, 'tilvm.c'), os.path.join(VM_DIR, 'main.c'), '-lm'],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        _VARIANT_EXE[name] = exe
    return _VARIANT_EXE[name]


def run_exe(exe: str, data: bytes):
    fd, path = tempfile.mkstemp(suffix='.tbc')
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    try:
        r = subprocess.run([exe, '--stats', path], capture_output=True, timeout=60)
    finally:
        os.unlink(path)
    err = r.stderr.decode('utf-8', 'replace')
    m = re.search(r'\[tilvm\] steps=(\d+)', err)
    return (r.returncode, r.stdout.decode('utf-8', 'replace'), re.sub(r'\[tilvm\] steps=.*\n?', '', err),
            int(m.group(1)) if m else -1)


@needs_cc
class TestCBuilds:
    @pytest.mark.parametrize("variant", sorted(VARIANTS))
    def test_variants_agree(self, variant):
        """Switch dispatch, no function table / caches, a 2-entry cache (collisions):
        the same results (and steps) as the default build."""
        exe = variant_exe(variant)
        for name in ('compare_branches', 'locals_and_fields', 'closures', 'binops_local', 'error_in_cast',
                     'iteration', 'indexing'):
            data = build(CORPUS[name])
            assert run_exe(exe, data) == c_run(data), name

    def test_lean_profile_runs_fused_code(self):
        src = ('fib(n: int) -> int\n    if n < 2\n        return n\n    return fib(n - 1) + fib(n - 2)\n'
               'main()\n    var t = 0\n    for i in 0..15\n        t += fib(i) % 7\n    print(t)\n'
               '    var xs = [5, 6, 7]\n    for x in xs\n        t += x * 3\n    print(t, xs[1] - 2)\n')
        data = build(src)
        assert any(op >= 0x80 for op in opcodes(data))
        rc, out, _err, _steps = py_run(data)
        crc, cout, cerr = run_cvm(data, profile="tiny")
        assert (crc, cout) == (rc, out), cerr

    def test_too_many_functions_is_a_load_error(self):
        """TILVM_MAX_FUNCS bounds the decoded function table: explicit load error."""
        exe = variant_exe("small-tables")                  # TILVM_MAX_FUNCS=64
        src = "".join(f"f{i}() -> int\n    return {i}\n" for i in range(70)) + \
            "main()\n    print(f0() + f69())\n"
        rc, _out, err, _steps = run_exe(exe, build(src))
        assert rc == 2 and "error -10" in err and "TILVM_MAX_FUNCS" in err
        rc, out, _err, _steps = run_exe(variant_exe("notables"), build(src))    # no table: no limit
        assert (rc, out) == (0, "69\n")

    def test_bad_comparison_operand_is_fatal(self, tmp_path):
        """Malformed bytecode (a fused compare with a non-comparison operator) is
        rejected the same way by both VMs."""
        m = til.compile_to_bytecode('main()\n    var a = 1\n    var b = 2\n    if a < b\n        print(1)\n',
                                    "<test>")
        for f in m.funcs:
            for ins in til._decode_insns(f.code):
                if ins[1] == Op.JUMP_IF_NOT_LL:
                    f.code[ins[0] + 1] = Op.ADD
        data = m.serialize()
        rc, _out, err, _s = py_run(data)
        assert rc == 1 and "bad opcode" in err
        crc, _cout, cerr, _s = c_run(data)
        assert crc == 1 and "bad opcode" in cerr

    @pytest.mark.parametrize("compiler, flags", [
        ("clang", ['-DTILVM_COMPUTED_GOTO=0']),
        ("gcc", ['-DTILVM_LEAN', '-DTILVM_COMPUTED_GOTO=1', '-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0']),
        ("clang", ['-DTILVM_LEAN', '-DTILVM_COMPUTED_GOTO=1', '-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0']),
        ("gcc", ['-Os', '-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1']),
    ], ids=["switch-clang", "lean-cgoto-gcc", "lean-cgoto-clang", "Os-gcc"])
    def test_strict_builds(self, compiler, flags, tmp_path):
        if not shutil.which(compiler):
            pytest.skip(f"{compiler} not installed")
        r = subprocess.run([compiler, '-std=c99', '-Wall', '-Wextra', '-pedantic', '-Werror', *flags,
                            '-c', os.path.join(VM_DIR, 'tilvm.c'), '-o', str(tmp_path / 'tilvm.o')],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


# ═══════════════════════════════════════════════════════════════
# PERFORMANCE (TIL_BENCH=1): against TIL 3.0 (commit 6f4d00b)
# ═══════════════════════════════════════════════════════════════


def _cpu_time(cmd, repeat):
    """Best CPU time of `repeat` runs (getrusage: microseconds; else wall clock).
    The minimum is used because other load on the machine only ever adds time."""
    try:
        import resource
    except ImportError:            # Windows
        resource = None
    times = []
    for _ in range(repeat):
        if resource:
            u0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True)
        t1 = time.perf_counter()
        assert r.returncode == 0, r.stderr
        if resource:
            u1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            times.append((u1.ru_utime - u0.ru_utime) + (u1.ru_stime - u0.ru_stime))
        else:
            times.append(t1 - t0)
    return min(times), r.stdout


@pytest.mark.skipif(os.environ.get('TIL_BENCH') != '1', reason="performance check: set TIL_BENCH=1")
@needs_cc
def test_speedup_against_til_3_0(tmp_path):
    """C VM at least 2x faster than TIL 3.0 on fib / loops / arrays / strings
    (desktop build, -O2); the reference VM not slower."""
    base = tmp_path / "base"
    for rel in ('src/til.py', 'vm/tilvm.c', 'vm/tilvm.h', 'vm/main.c'):
        r = subprocess.run(['git', '-C', ROOT, 'show', f'{BASELINE_REV}:{rel}'], capture_output=True)
        if r.returncode != 0:
            pytest.skip(f"git history with {BASELINE_REV} is not available")
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_bytes(r.stdout)
    exes = {}
    for name, vm_dir in (('new', VM_DIR), ('old', str(base / 'vm'))):
        exes[name] = str(tmp_path / f"tilvm-{name}")
        subprocess.run([CC, '-std=c99', '-O2', '-o', exes[name], os.path.join(vm_dir, 'tilvm.c'),
                        os.path.join(vm_dir, 'main.c'), '-lm'], check=True)
    results = {}
    for bench in ('fib', 'loops', 'arrays', 'strings'):
        src = os.path.join(ROOT, 'benchmarks', bench + '.til')
        new_tbc, old_tbc = tmp_path / f"{bench}.tbc", tmp_path / f"{bench}.old.tbc"
        assert cli('build', '--target', 'vm', src, '-o', str(new_tbc)).returncode == 0
        subprocess.run([sys.executable, str(base / 'src' / 'til.py'), 'build', '--target', 'vm', src,
                        '-o', str(old_tbc)], check=True, capture_output=True)
        t_new, out_new = _cpu_time([exes['new'], str(new_tbc)], 9)
        t_old, out_old = _cpu_time([exes['old'], str(old_tbc)], 9)
        assert out_new == out_old
        results[bench] = t_old / t_new
    print("C VM speedups vs TIL 3.0:", {k: round(v, 2) for k, v in results.items()})
    for bench, speedup in results.items():
        assert speedup >= 2.0, f"{bench}: only {speedup:.2f}x faster than TIL 3.0"
    # reference VM: not slower (5% tolerance for timer noise)
    src = os.path.join(ROOT, 'benchmarks', 'fib.til')
    driver = ("import sys, time; sys.path.insert(0, sys.argv[1]); import til; "
              "m = til.TBCModule.from_bytes(open(sys.argv[2], 'rb').read()); t = time.process_time(); "
              "til.TILVM(m, write=lambda b: None).run(); print(time.process_time() - t)")
    old_tbc = tmp_path / "fib.old.tbc"
    new_tbc = tmp_path / "fib.tbc"
    py_times = {}
    for name, srcdir, tbc in (('new', os.path.join(ROOT, 'src'), new_tbc), ('old', str(base / 'src'), old_tbc)):
        runs = [float(subprocess.run([sys.executable, '-c', driver, srcdir, str(tbc)], capture_output=True,
                                     text=True, check=True).stdout) for _ in range(3)]
        py_times[name] = min(runs)
    print("reference VM fib(27):", py_times)
    assert py_times['new'] <= py_times['old'] * 1.05
