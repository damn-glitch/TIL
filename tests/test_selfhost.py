#!/usr/bin/env python3
"""
TIL 3.1 #1: the self-hosted compiler selfhost/tilc.til (TIL written in TIL).

* Bootstrap fixed point: stage1 = the Python compiler builds tilc.tbc;
  stage2 = stage1 compiles tilc.til on the C VM; stage3 = stage2 compiles
  tilc.til again. stage2 == stage3 byte for byte (and both equal stage1).
* Every example and every program of tests/selfhost_corpus.txt (programs the
  test-suite compiles) is compiled by tilc (stage 2) and by the Python
  compiler: the bytecode must be identical, or - where it legitimately
  differs (folding of float constant expressions) - behave identically on
  both VMs. Programs outside "TIL Core" must be rejected with
  `tilc: FILE:LINE: unsupported: WHAT`, invalid programs with `... error: ...`.
* Calculator configuration: tilc.tbc runs on an int32/float32 C VM with a
  64 KB heap.
* TIL 3.1: superinstructions (and --no-fuse), effects with resumption,
  `from` imports with their closure, E-IMPORT, packages in til_modules/, and
  the two constant-folding fixes shared with src/til.py - all byte-identical.

Slow checks (the Python VM compiling all of tilc.til, ~60 s; 33,000 and 66,000
constants on a 64 MB heap, ~20 s) run with TIL_TEST_SLOW=1.

Run: python3 -m pytest tests/test_selfhost.py -v
"""

import os
import re
import subprocess
import sys
import tempfile
import time

import pytest

from tilhelpers import CC, ROOT, cvm_binary, needs_cc, run_cvm, run_vm, til

TILC_SRC = os.path.join(ROOT, 'selfhost', 'tilc.til')
CORPUS = os.path.join(ROOT, 'tests', 'selfhost_corpus.txt')
EXAMPLES = os.path.join(ROOT, 'examples')
BIG_HEAP = ('-DTILVM_HEAP_SIZE=4194304',)          # self-compilation needs ~1 MB
CALC_FLAGS = ('-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1', '-DTILVM_HEAP_SIZE=65536')
SLOW = bool(os.environ.get('TIL_TEST_SLOW'))

# Constructs outside TIL Core: tilc must reject them with "unsupported: <what>"
# (see selfhost/README.md). Any other "unsupported" message fails the tests.
UNSUPPORTED = ("(module not found next to the importing file or in til_modules/; tilc does not search "
               "TIL_PATH or the standard library)",)

# tilc emits exactly what the Python compiler emits (superinstructions,
# effect handlers and all). The one deliberate difference: a float constant
# expression that folds to NaN is left to run time (Python's NaN bit pattern
# depends on the CPU); then only the behaviour is compared.
NAN_FOLDING = re.compile(r'\d\.\d|\d[eE][-+]?\d')
def py_compile(src: str, name: str = "<test>") -> bytes:
    return til.compile_to_bytecode(src, name).serialize()


class Tilc:
    """A tilc.tbc + the C VM that runs it; compiles TIL sources to .tbc."""

    def __init__(self, tbc: bytes, vm: str):
        self.dir = tempfile.mkdtemp(prefix='tilc-')
        self.path = os.path.join(self.dir, 'tilc.tbc')
        with open(self.path, 'wb') as f:
            f.write(tbc)
        self.vm = vm
        self.tbc = tbc

    def compile_file(self, src_path: str, out_path: str, vm=None, timeout=120):
        r = subprocess.run([vm or self.vm, self.path, src_path, out_path], capture_output=True,
                           timeout=timeout)
        data = None
        if r.returncode == 0 and os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                data = f.read()
            os.unlink(out_path)
        return r.returncode, r.stdout.decode('utf-8', 'replace'), data

    def compile(self, src: str, name: str = "prog.til", vm=None):
        """-> (exit code, tilc output, .tbc bytes or None)"""
        path = os.path.join(self.dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        return self.compile_file(path, os.path.join(self.dir, 'out.tbc'), vm)


@pytest.fixture(scope="module")
def stage1() -> bytes:
    with open(TILC_SRC, encoding='utf-8') as f:
        return py_compile(f.read(), TILC_SRC)


@pytest.fixture(scope="module")
def big_vm() -> str:
    return cvm_binary("host", BIG_HEAP)


@pytest.fixture(scope="module")
def tilc(stage1, big_vm) -> Tilc:
    """Stage 2: tilc compiled by itself (stage 1 running on the C VM)."""
    t1 = Tilc(stage1, big_vm)
    rc, out, stage2 = t1.compile_file(TILC_SRC, os.path.join(t1.dir, 'tilc2.tbc'))
    assert rc == 0 and stage2, out
    return Tilc(stage2, big_vm)


def read_corpus():
    with open(CORPUS, encoding='utf-8') as f:
        text = f.read()
    progs, cur, lines = [], None, []
    for line in text.split('\n'):
        if line.startswith('#==== '):
            if cur:
                progs.append((cur[0], cur[1], '\n'.join(lines).rstrip('\n') + '\n'))
            parts = line.split()
            cur, lines = (parts[1], parts[2] if len(parts) > 2 else '?'), []
        elif cur:
            lines.append(line)
    if cur:
        progs.append((cur[0], cur[1], '\n'.join(lines).rstrip('\n') + '\n'))
    return progs


def run_both_vms(data: bytes, cwd: str):
    """(rc, stdout, stderr) on the Python VM and on the C VM."""
    module = til.TBCModule.from_bytes(data)
    out, err = [], []
    old = os.getcwd()
    os.chdir(cwd)
    try:
        import io
        saved = sys.stderr
        sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')
        try:
            rc = til.TILVM(module, write=out.append, read_line=lambda: b"", max_steps=2_000_000).run()
            sys.stderr.flush()
            err_text = sys.stderr.buffer.getvalue().decode('utf-8', 'replace')
        finally:
            sys.stderr = saved
    finally:
        os.chdir(old)
    py = (rc, b"".join(out).decode('utf-8', 'replace'), err_text)
    c = None
    if CC:
        # (more steps: with superinstructions the C VM runs the "recovery"
        # program's endless loop 2,000,000 times before its 20 ms timeout)
        c = run_cvm(data, vm_args=('--max-steps', '200000000'), cwd=cwd)
    return py, c


def check_program(tilc: Tilc, name: str, src: str):
    """Compile with both compilers; return 'same' | 'behaviour' | 'unsupported:<what>' |
    'rejected'. Fails the test on any mismatch."""
    try:
        ref = py_compile(src, name)
        py_error = None
    except (til.CompileError, SyntaxError) as e:
        ref, py_error = None, str(e)
    rc, out, data = tilc.compile(src)
    if py_error is not None:
        assert rc == 1 and data is None, f"{name}: Python rejects it but tilc does not:\n{py_error}"
        assert re.match(r'tilc: \S+:\d+: (error|unsupported): ', out), f"{name}: bad message {out!r}"
        return 'rejected'
    if rc != 0:
        m = re.match(r'tilc: \S+:(\d+): unsupported: (.*)', out.strip())
        assert m and m.group(2).endswith(UNSUPPORTED), f"{name}: tilc failed:\n{out}"
        return 'unsupported:' + m.group(2)
    if data == ref:
        return 'same'
    # different bytecode: only expected for NaN folding; the programs must
    # behave the same
    assert NAN_FOLDING.search(src), f"{name}: bytecode differs from the Python compiler's"
    cwd = tempfile.mkdtemp(prefix='tilc-run-')
    ref_py, ref_c = run_both_vms(ref, cwd)
    got_py, got_c = run_both_vms(data, cwd)
    assert got_py == ref_py, f"{name}: Python VM output differs\n{got_py}\n{ref_py}"
    assert got_c == ref_c, f"{name}: C VM output differs\n{got_c}\n{ref_c}"
    return 'behaviour'


# ═══════════════════════════════════════════════════════════════
# TABLES SHARED WITH src/til.py
# ═══════════════════════════════════════════════════════════════

def tilc_text() -> str:
    with open(TILC_SRC, encoding='utf-8') as f:
        return f.read()


def til_literal(text: str, name: str, fn: bool = False):
    """The value of `var NAME = <literal>` (or `NAME()` returning one) in tilc.til,
    with T_* / OP_* constants resolved to their names."""
    src = tilc_text()
    if fn:
        m = re.search(r'^' + name + r'\(\)\n    return (\{.*?\}|\[.*?\])\n', src, re.S | re.M)
    else:
        m = re.search(r'^var ' + name + r' = (\{.*?\}|\[.*?\])\n', src, re.S | re.M)
    assert m, name
    lit = re.sub(r'\b(T|OP)_([A-Z_0-9]+)\b', r'"\1_\2"', m.group(1))
    import ast
    return ast.literal_eval(lit.replace('\n    ', ' '))


class TestTablesInSync:
    """tilc.til keeps copies of tables of src/til.py: when one changes (a new
    native, keyword, prelude function...), tilc.til must be updated too."""

    def test_natives_and_arity(self):
        assert til_literal(tilc_text(), 'NATIVES') == til.NATIVES
        lo, hi = til_literal(tilc_text(), 'NAT_MIN'), til_literal(tilc_text(), 'NAT_MAX')
        assert [(a, None if b < 0 else b) for a, b in zip(lo, hi)] == [til.NATIVE_ARITY[n] for n in til.NATIVES]

    def test_methods_and_prelude(self):
        assert til_literal(tilc_text(), 'METHODS') == til.METHODS
        assert til_literal(tilc_text(), 'PRELUDE_SRC') == til.PRELUDE_SOURCE

    def test_keywords(self):
        def names(table):
            return {k: v.name for k, v in table.items()}
        def mine(name, fn=False):
            return {k: v[2:] for k, v in til_literal(tilc_text(), name, fn).items()}
        assert mine('KW_EN') == names(til.ENGLISH_KEYWORDS)
        assert mine('KW_KK') == names(til.KAZAKH_KEYWORDS)
        assert mine('kw_kk_latn', True) == names(til.KAZAKH_LATIN_KEYWORDS)
        assert mine('kw_ru', True) == names(til.RUSSIAN_KEYWORDS)
        assert set(til_literal(tilc_text(), 'TRUE_WORDS')) == til.TRUE_WORDS
        assert til_literal(tilc_text(), 'AL_KK') == til.KAZAKH_ALIASES
        tt = til_literal(tilc_text(), 'TT_NAMES')
        assert tt == [t.name for t in til.TokenType], "token types (TT_NAMES)"

    def test_opcodes_and_io_builtins(self):
        consts = dict(re.findall(r'^const OP_([A-Z_0-9]+) = (\d+)$', tilc_text(), re.M))
        for name, value in consts.items():
            assert getattr(til.Op, name) == int(value), name
        assert set(til_literal(tilc_text(), 'IO_BUILTINS')) == til.SemanticChecker.IO_BUILTINS
        assert set(til_literal(tilc_text(), 'PRIM')) == set(til.TYPE_MAP)


# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP
# ═══════════════════════════════════════════════════════════════

@needs_cc
class TestBootstrap:
    def test_fixed_point(self, stage1, big_vm):
        t1 = Tilc(stage1, big_vm)
        start = time.time()
        rc, out, stage2 = t1.compile_file(TILC_SRC, os.path.join(t1.dir, 'tilc2.tbc'))
        t_stage2 = time.time() - start
        assert rc == 0, out
        t2 = Tilc(stage2, big_vm)
        rc, out, stage3 = t2.compile_file(TILC_SRC, os.path.join(t2.dir, 'tilc3.tbc'))
        assert rc == 0, out
        assert stage2 == stage3, "stage2 != stage3: the bootstrap has no fixed point"
        # the self-hosted compiler emits exactly what the Python compiler emits
        assert stage2 == stage1
        print(f"\n[selfhost] tilc.til -> {len(stage2)} bytes; stage 2 on the C VM: {t_stage2:.2f}s")

    def test_fixed_point_on_int32_vm(self, stage1):
        """The calculator-class integer size (int32, float32): the same stages."""
        vm = cvm_binary("int32", ('-DTILVM_HEAP_SIZE=2097152',))
        t1 = Tilc(stage1, vm)
        rc, out, stage2 = t1.compile_file(TILC_SRC, os.path.join(t1.dir, 'tilc2.tbc'))
        assert rc == 0, out
        assert stage2 == stage1
        rc, out, stage3 = Tilc(stage2, vm).compile_file(TILC_SRC, os.path.join(t1.dir, 'tilc3.tbc'))
        assert rc == 0, out
        assert stage3 == stage2

    def test_stage2_runs_programs(self, tilc):
        rc, out, data = tilc.compile('main()\n    print("hi from stage 2")\n')
        assert rc == 0, out
        assert run_vm(data) == (0, "hi from stage 2\n")

    def test_python_vm_compiles_like_c_vm(self, stage1, tilc):
        """The reference VM running stage 1 produces the same bytes (small program)."""
        src_path = os.path.join(EXAMPLES, '04_structs.til')
        out_path = os.path.join(tilc.dir, 'py.tbc')
        module = til.TBCModule.from_bytes(stage1)
        rc = til.TILVM(module, write=lambda b: None, argv=[src_path, out_path]).run()
        assert rc == 0
        with open(out_path, 'rb') as f:
            data = f.read()
        rc, out, c_data = tilc.compile_file(src_path, os.path.join(tilc.dir, 'c.tbc'))
        assert data == c_data
        with open(src_path, encoding='utf-8') as f:
            ref = py_compile(f.read(), src_path)
        assert data == ref

    @pytest.mark.skipif(not SLOW, reason="slow (~45 s): set TIL_TEST_SLOW=1")
    def test_python_vm_bootstrap(self, stage1, tilc):
        """The reference VM running stage 1 compiles all of tilc.til to stage 2."""
        out_path = os.path.join(tilc.dir, 'py-stage2.tbc')
        module = til.TBCModule.from_bytes(stage1)
        start = time.time()
        rc = til.TILVM(module, write=lambda b: None, argv=[TILC_SRC, out_path]).run()
        elapsed = time.time() - start
        assert rc == 0
        with open(out_path, 'rb') as f:
            assert f.read() == tilc.tbc
        print(f"\n[selfhost] Python VM: tilc.til compiled in {elapsed:.1f}s")

    def test_cli_run_vm(self, tmp_path):
        """til run --vm selfhost/tilc.til input.til output.tbc (Python VM)."""
        prog = tmp_path / "in.til"
        prog.write_text('let x = 6\nprint(x * 7)\n', encoding='utf-8')
        out = tmp_path / "out.tbc"
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'til.py'), 'run', '--vm', TILC_SRC,
                            str(prog), str(out)], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        assert run_vm(out.read_bytes()) == (0, "42\n")


# ═══════════════════════════════════════════════════════════════
# EXAMPLES AND THE TEST-SUITE CORPUS
# ═══════════════════════════════════════════════════════════════

def example_files():
    out = []
    for d, _subdirs, files in sorted(os.walk(EXAMPLES)):
        for f in sorted(files):
            if f.endswith('.til'):
                out.append(os.path.join(d, f))
    return out


@needs_cc
class TestPrograms:
    def test_examples(self, tilc):
        results = {}
        for path in example_files():
            with open(path, encoding='utf-8') as f:
                src = f.read()
            name = os.path.relpath(path, ROOT)
            results[name] = check_program(tilc, name, src)
        print("\n[selfhost] examples:", {n: r for n, r in results.items()})
        for must in ('examples/01_hello.til', 'examples/03_functions.til', 'examples/04_structs.til',
                     'examples/05_control_flow.til', 'examples/06_enums.til',
                     'examples/09_v3_features.til', 'examples/10_verified.til',
                     'examples/kazakh/salam.til', 'examples/kazakh/salam_latn.til',
                     'examples/russian/privet.til'):
            assert results[must] in ('same', 'behaviour'), (must, results[must])

    def test_examples_in_place(self, tilc):
        """Each example compiled where it is (its imports resolve): the bytes
        of the Python compiler, also for examples/08_imports.til."""
        for path in example_files():
            assert compare_file(tilc, path) == 'same', path

    def test_corpus(self, tilc):
        progs = read_corpus()
        assert len(progs) > 200
        stats = {}
        for pid, origin, src in progs:
            r = check_program(tilc, f"{pid} ({origin})", src)
            key = r.split(':', 1)[0]
            stats[key] = stats.get(key, 0) + 1
        print("\n[selfhost] corpus:", stats)
        # the whole corpus is inside TIL Core (the invalid programs are rejected)
        assert stats.get('unsupported', 0) == 0, stats
        assert stats.get('same', 0) + stats.get('behaviour', 0) == 244, stats


# ═══════════════════════════════════════════════════════════════
# LANGUAGE DETAILS (byte-identical to the Python compiler)
# ═══════════════════════════════════════════════════════════════

IDENTICAL = {
    "floats": """
main()
    print(0.1)
    print(1e23)
    print(9007199254740993.0)
    print(1.7976931348623157e308)
    print(1e309)
    print(2.2250738585072011e-308)
    print(4.9406564584124654e-324)
    print(2.4703282292062327e-324)
    print(2.4703282292062328e-324)
    print(1e-400)
    print(123456789012345678901234567890.0)
    print(1_000.5_5)
    print(-0.0)
    print(-2.5e-3)
""",
    "float_folding": """
main()
    print(0.1 + 0.2, 1e308 * 10.0, -0.0 + 0.0, -0.0 + -0.0, -0.0 * 5, 1 / 3.0)
    print(9007199254740993 + 0.0, 9007199254740995 * 1.0, 1e-320 * 0.5, 5e-324 / 2)
    print(1.5e-323 / 2, 2.0 - 2.0, 1e16 + 1.0, 1e16 + 2.0, 123.456 - 0.456)
    print(9007199254740993 == 9007199254740992.0, 9007199254740993 > 9007199254740992.0)
    print(true == 1.0, "a" == 1.0, 1e400 > 1e308, -1e400 < -1e308, 1.5 / 0, 1.5 % 2.0)
    print(1e400 * -2.0, 3.0 / 1e400, -3 / 1e400, 0.5 < 1, 2 >= 2.0, -0.0 == 0.0)
    print(not 0.0, not -0.0, 0.0 or 1.5, 2.5 and 0, 1.0 != 1, -(-1.5))
    print(4611686018427387903 * 2.0, -9223372036854775808 * 1.0)
    print(1.7976931348623157e308 + 9.979201547673598e291, 1.0e-5 * 1e-5 * 1e-300)
    print(2.2250738585072014e-308 - 2.2250738585072009e-308, 3.14159 * 2.0)
""",
    "ints": """
main()
    print(9223372036854775807)
    print(-9223372036854775808)
    print(18446744073709551615)
    print(340282366920938463463374607431768211457)
    print(0xffff_ffff_ffff_ffff)
    print(0b1011 + 0o777 + 0XfF)
    print(2147483648 * 3)
    print(4611686018427387904 * 4)
    print(3000000000 * 3000000000)
    print(-9223372036854775808 / -1)
    print(-7 / 2)
    print(-7 % 3)
    print(~5)
    print(not "")
    print(true == 1)
    print(len("héllo") * 2)
    print('a' + 'ә' + '\\u{1F600}')
""",
    "strings": """
let NAME = "TIL"
const PI = 3.14
main()
    print("tab\\there\\nnew \\"q\\" \\\\ \\x41\\xff \\u{416} \\q \\{")
    print(f"{NAME} {PI:.1f} [{7:>5}] {{lit}} {1 + 2} {255:x}")
    print(f"")
    print(f'single {NAME}')
    print("ключ" + "қазақ")
""",
    "script": """
let base = 10
print("start")
var total = base * 2
let derived = total + 1
for i in 0..3
    total += i
print(total)
helper(x) => x * base
print(helper(derived))
""",
    "structures": """
struct Config
    name: str = "default"
    size: int = 4
    ratio: float
    items: [int]

enum Color
    Red
    Green = 5
    Blue

enum Shape
    Circle(float)
    Empty

greet(name: str, greeting: str = "Hello", times = 1) -> str
    var out = ""
    for i in 0..times
        out += greeting + ", " + name + "! "
    return out

safe_div(a: int, b: int) -> Result<int, str>
    if b == 0
        return Err("div by zero")
    return Ok(a / b)

chain(a, b) -> Result<int, str>
    let q = safe_div(a, b)?
    return Ok(q + 1)

brace(n) {
    var s = 0; s += n; return s * 2
}

main()
    let c = Config { ratio: 2 }
    print(c)
    print(Color.Green + Color.Blue)
    print(Shape.Circle(1.5))
    print(greet("A", "Hi"))
    print(chain(9, 3))
    print(chain(9, 0))
    print(brace(21))
    print([1, 2, 3] |> len)
    var vec = Vec<int>.new()
    vec.push(4)
    print((1, "two"))
    if c.size > 3 print("big")
    var n = 0
    loop
        n += 1
        if n == 2
            continue
        if n > 4
            break
        print(n)
    var m = {"a": 1}
    for k in m
        print(k + "=" + str(m[k]))
    var arr = [1, 2, 3]
    arr[0] += 10
    c.size *= 3
    print(arr, c.size, 3.9 as int, 1 as str, -2 ** 2)
    assert c.size == 12, "size"
""",
    "match": """
enum Shape
    Circle(float)
    Rect(float, float)
    Dot

struct P
    x: int
    y: int

let ORIGIN = 0

area(s: Shape) -> float
    return match s
        Shape.Circle(r) => 3.0 * r * r
        Shape.Rect(w, h) => w * h
        Shape.Dot => 0.0

describe(v)
    match v
        None => print("none")
        0 | 1 => print("small")
        2..=9 => print("digit")
        ORIGIN => print("never")
        Some(x) if x > 100 => print(f"big some {x}")
        Some(x) => print(f"some {x}")
        Ok(x) => print(f"ok {x}")
        Err(e) => print(f"err {e}")
        (a, b) => print(a + b)
        [a, _, c] => print(a * c)
        P { x: px, y: 2 } => print(f"P with y=2, x={px}")
        "text" => print("a string")
        _ => print("other")

main()
    for s in [Shape.Circle(1.0), Shape.Rect(2.0, 3.5), Shape.Dot]
        print(area(s))
    for v in [None, 1, 5, Some(500), Some(5), Ok(1), Err("bad"), (1, 2), [3, 4, 5], P { x: 7, y: 2 }, "text", 3.5]
        describe(v)
    let label = match 3
        1 => "one"
        3 => "three"
    print(label)
""",
    "closures": """
make_counter()
    var n = 0
    let inc = |k| n + k
    n = 10
    return inc

apply(fun, v) => fun(v)

# (calling a captured parameter directly, |x| f(x), is rejected by both
# compilers: see selfhost/README.md)
compose(f, g)
    return |x| apply(f, apply(g, x))

twice(x) => x * 2

main()
    let c = make_counter()
    print(c(5))
    let add = |a, b| a + b
    print(add(2, 3))
    let h = compose(|x| x + 1, twice)
    print(h(10))
    let base = 100
    let nested = |x| |y| x + y + base
    print(nested(1)(2))
    let nums = [1, 2, 3, 4, 5, 6]
    print(nums.map(|n| n * n))
    print(nums.filter(|n| n % 2 == 0).map(twice))
    print(nums.reduce(|a, b| a + b, 0))
    print(nums.any(|n| n > 5), nums.all(|n| n > 0))
    nums.each(|n| print(n))
    print([x * 10 for x in nums if x % 2 == 1])
    print([c for c in "abc"])
    var fns = []
    for i in 0..3
        fns.push(|x| x + i)
    print(fns[2](1))
    print((|x: int| -> int { x + 1 })(41))
""",
    "recovery": """
var attempts = 0

flaky() -> int
    attempts += 1
    assert attempts >= 3, "warming up"
    return 42

trait Greeter
    name(self) -> str
    greet(self) -> str
        return "Hello, " + self.name() + "!"

struct Student
    first: str

impl Greeter for Student
    name(self) -> str
        return self.first

main()
    let reading = retry(5)
        flaky()
    print(f"{reading} after {attempts}")
    let r = timeout(20)
        var i = 0
        loop
            i += 1
        i
    else
        "stopped: " + error
    print(r)
    let msg = retry(1)
        intent "load the class list"
            let classes = ["7A", "7B"]
            classes[5]
    else
        error
    print(msg)
    let v = retry(2)
        if attempts > 100
            "many"
        else
            "few"
    print(v)
    print(Student { first: "Aigerim" }.greet())
    intent "statement form"
        print("inside intent")
""",
    "contracts": """
#[requires: b != 0]
safe_div(a: int, b: int) -> int
    return a / b

#[ensures: result >= 0]
abs_diff(a: int, b: int) -> int
    if a > b
        return a - b
    return b - a

#[requires: n >= 0]
#[ensures: result >= 1]
fact(n: int) -> int
    if n <= 1
        return 1
    return n * fact(n - 1)

main()
    print(safe_div(10, 2), abs_diff(3, 8), fact(5))
    var zero = 0
    print(safe_div(1, zero))
""",
    # every pattern form, on values that reach their arm (in "match" above,
    # describe() stops at Some(500): `2..=9` compares a struct with an int)
    "match_run": """
enum Shape
    Circle(float)
    Rect(float, float)
    Dot

enum Color
    Red
    Green = 5
    Blue

struct P
    x: int
    y: int

let ORIGIN = 0
var hits = 0

bump(n)
    hits += n

area(s: Shape) -> float
    return match s
        Shape.Circle(r) => 3.0 * r * r
        Shape.Rect(w, h) => w * h
        Shape.Dot => 0.0

# patterns that compare with < and > see ints only
number(n: int) -> str
    return match n
        ORIGIN => "origin"
        1 | 2 => "one or two"
        3..5 => "three or four"
        5..=9 => "five to nine"
        x if x < 0 => f"negative {x}"
        x if x % 2 == 0 => f"even {x}"
        _ => "big odd"

describe(v)
    match v
        None => print("none")
        Some(x) if x > 100 => print(f"big some {x}")
        Some(x) => print(f"some {x}")
        Ok(x) => print(f"ok {x}")
        Err(e) => print(f"err {e}")
        (a, b) => print(f"pair {a + b}")
        [a, _, c] => print(f"triple {a * c}")
        [] => print("empty")
        P { x: px, y: 2 } => print(f"P with y=2, x={px}")
        P { x: 0, y: py } => print(f"P on the y axis at {py}")
        "text" => print("a string")
        true => print("true")
        _ => print(f"other {v}")

main()
    for s in [Shape.Circle(1.0), Shape.Rect(2.0, 3.5), Shape.Dot]
        print(area(s))
    for n in [0, 1, 2, 3, 4, 5, 9, -7, 12, 13]
        print(number(n))
    for v in [None, Some(500), Some(5), Ok(1), Err("bad"), (1, 2), [3, 4, 5], [], P { x: 7, y: 2 },
              P { x: 0, y: 9 }, P { x: 1, y: 1 }, "text", true, 3.5]
        describe(v)
    let c = Color.Green
    match c
        Color.Red => print("red")
        Color.Green => print(f"green {c}")
        _ => print("blue")
    let label = match 3
        1 => "one"
        3 => "three"
    print(label)
    for i in 0..8
        match i % 4
            0 => print(f"zero at {i}")
            1 | 3 => bump(1)
            _ => bump(100)
    print(hits)
""",
}

# what the IDENTICAL programs print when they run (both VMs, both compilers)
IDENTICAL_RC = {"contracts": 1, "match": 1}
MATCH_RUN_OUT = """3
7
0
origin
one or two
one or two
three or four
three or four
five to nine
five to nine
negative -7
even 12
big odd
none
big some 500
some 5
ok 1
err bad
pair 3
triple 15
empty
P with y=2, x=7
P on the y axis at 9
other P { x: 1, y: 1 }
a string
true
other 3.5
green 5
three
zero at 0
zero at 4
204
"""


@needs_cc
class TestLanguage:
    @pytest.mark.parametrize("name", sorted(IDENTICAL))
    def test_identical_bytecode(self, tilc, name):
        assert check_program(tilc, name, IDENTICAL[name]) in ('same', 'behaviour')
        rc, out, data = tilc.compile(IDENTICAL[name])
        ref = py_compile(IDENTICAL[name])
        assert data == ref

    def test_identical_programs_run(self, tilc, tmp_path):
        """The programs above also run (not only compile to the same bytes):
        the same exit code and output on both VMs."""
        for name, src in sorted(IDENTICAL.items()):
            rc, out, data = tilc.compile(src)
            assert rc == 0, (name, out)
            py, c = run_both_vms(data, str(tmp_path))
            assert py[0] == IDENTICAL_RC.get(name, 0), (name, py)
            assert c[:2] == py[:2], (name, py, c)
            if name == "match_run":
                assert py[1] == MATCH_RUN_OUT

    def test_runtime_errors_report_the_same_lines(self, tilc):
        src = "inner(a)\n    let arr = [1, 2]\n    return arr[a]\n\nmain()\n    print(inner(1))\n    print(inner(5))\n"
        rc, out, data = tilc.compile(src)
        assert rc == 0
        crc, cout, cerr = run_cvm(data)
        assert crc == 1 and cout == "2\n"
        assert "at inner (line 3)" in cerr and "at main (line 7)" in cerr

    def test_imports(self, tilc, tmp_path):
        files = {
            "geom.til": "import util\nstruct Point\n    x: int\n    y: int\nimpl Point\n"
                        "    norm1(self) -> int\n        return abs(self.x) + abs(self.y)\n"
                        "let ORIGIN_X = 0\nvar counter = 0\nmake(x, y) -> Point\n    counter += 1\n"
                        "    return Point { x: x, y: y }\nmain()\n    print(\"not imported\")\n",
            "util.til": "import geom\ntwice(v) => v * 2\n",
            "rus.til": "#[lang: ru]\nфункция удвоить(х)\n    вернуть х * 2\n",
            "pkg/sub/mod.til": "deep() -> str\n    return \"deep\"\n",
            "pkg/leaf.til": "leaf_val() => 41\n",
            "main.til": "import geom as g\nfrom util import twice\nimport rus\nimport pkg.sub\n"
                        "from pkg.leaf import *\nmain()\n    let p = g.make(3, -4)\n    print(p.norm1())\n"
                        "    print(g.counter, twice(21), удвоить(5), deep(), sub.deep(), leaf_val())\n"
                        "    print(geom.ORIGIN_X)\n",
            "bad.til": "from geom import Point, make\nmain()\n    print(make(1, 2))\n",
        }
        for name, text in files.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
        main = str(tmp_path / "main.til")
        rc, out, data = tilc.compile_file(main, str(tmp_path / "out.tbc"))
        assert rc == 0, out
        ref = py_compile(files["main.til"], main)
        assert data == ref
        assert run_vm(data) == (0, "7\n1\n42\n10\ndeep\ndeep\n41\n0\n")
        # `from geom import Point, make` also brings in what make uses (`counter`,
        # TIL 3.1 packages); it used to be "unknown name 'counter'" in both compilers
        bad = str(tmp_path / "bad.til")
        rc, out, data = tilc.compile_file(bad, str(tmp_path / "out.tbc"))
        assert rc == 0, out
        assert data == py_compile(files["bad.til"], bad)
        assert run_vm(data) == (0, "Point { x: 1, y: 2 }\n")

    def test_float_folding_nan_keeps_behaviour(self, tilc):
        src = "main()\n    print(1e400 - 1e400)\n    print(-1e400 * 0.0)\n    print(2.0 * 3.5)\n"
        assert check_program(tilc, "nan", src) == 'behaviour'


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

ERRORS = [
    ("main()\n    print(totl)\n", 2, "error: unknown name 'totl'"),
    ("main()\n    let x = 1\n    x = 2\n", 3, "error: cannot assign twice to immutable variable 'x'"),
    ("let G = 1\nmain()\n    G = 2\n", 3, "error: cannot assign to constant 'G'"),
    ("f(a, b = 1)\n    return a\nmain()\n    f(1, 2, 3)\n", 4, "error: 'f' expects 1 to 2 argument(s), got 3"),
    ("main()\n    print(len())\n", 2, "error: 'len' takes 1 argument(s), got 0"),
    ("main()\n    nope(1)\n", 2, "error: unknown function 'nope'"),
    ("main()\n    break\n", 2, "error: 'break' outside of a loop"),
    ("f()\n    return 1\nf()\n    return 2\n", 3, "error: function 'f' is defined twice"),
    ("struct P\n    x: int\nmain()\n    let p = P { y: 1 }\n", 4, "error: struct 'P' has no field 'y'"),
    ("main()\n    print(\"abc)\n", 2, "error: Unterminated string"),
    ("main()\n    let x = 0x\n", 2, "error: Invalid hexadecimal literal: '0x'"),
    ("main()\n    let x = (1 + \n", 3, "error: Unexpected token: DEDENT"),
    ("main()\n    let x = 1 $ 2\n", 2, "error: Unexpected character: '$'"),
    ("main(x)\n    print(x)\n", 1, "error: main() takes no parameters"),
    ("struct P\n    x: int\nmain()\n    print(P)\n", 4, "error: 'P' is a type, not a value"),
    ("main()\n    [1].frobnicate()\n", 2, "error: unknown method 'frobnicate'"),
    ("#[requires: b != 0]\nd(a, b)\n    return a / b\nmain()\n    print(d(1, 0))\n", 5,
     "error: contract violation: call to 'd' (a=1, b=0) breaks #[requires: b != 0]"),
    ('#[requires: s != "xy" and f > 0.5 and n < 0]\nd(s, f, n)\n    return s\nmain()\n'
     '    print(d("xy", 1e-7, 12345678901234))\n', 5,
     "error: contract violation: call to 'd' (s='xy', f=1e-07, n=12345678901234) breaks "
     '#[requires: s != "xy" and f > 0.5 and n < 0]'),
    ("main()\n    let f = |x| x\n    let y = 1\n    y = f(2)\n", 4,
     "error: cannot assign twice to immutable variable 'y'"),
    ("compose(f)\n    return |x| f(x)\n", 2, "error: unknown function 'f'"),
    ("enum E\n    A(int)\nmain()\n    match E.A(1)\n        E.B(x) => print(x)\n", 5,
     "error: enum 'E' has no variant 'B'"),
    ("main()\n    match Some(1)\n        Some(a, b) => print(a)\n", 3, "error: pattern expects 1 field(s), got 2"),
    ("#[pure]\nsq(x)\n    log(x)\n    return x * x\nlog(v)\n    print(v)\n", 3,
     "error: pure function 'sq' performs effect 'io' (through the call to 'log')"),
    ("#[effects: fs]\nf()\n    print(1)\n", 3, "error: function 'f' performs effect 'io', but declares only #[effects: fs]"),
    ("cap Net { send }\n#[cap: Net]\nfetch()\n    return 1\nsneaky()\n    return fetch()\n", 6,
     "error: 'sneaky' uses 'fetch', which requires capability 'Net', but 'sneaky' does not hold it"),
    ("#[cap: Net]\nfetch()\n    return 1\nlaunder()\n    let f = fetch\n    return f\n", 5,
     "error: 'launder' uses 'fetch', which requires capability 'Net', but 'launder' does not hold it"),
]

UNSUPPORTED_PROGRAMS = [
    ("import no_such_module\n", 1, "import of 'no_such_module' (module not found next to the importing "
     "file or in til_modules/; tilc does not search TIL_PATH or the standard library)"),
]


@needs_cc
class TestDiagnostics:
    @pytest.mark.parametrize("src,line,msg", ERRORS)
    def test_errors(self, tilc, src, line, msg):
        with pytest.raises((til.CompileError, SyntaxError)):
            py_compile(src)                    # the Python compiler rejects it too
        rc, out, data = tilc.compile(src, "bad.til")
        assert rc == 1 and data is None
        assert out.strip().endswith(msg), out
        assert out.startswith(f"tilc: {os.path.join(tilc.dir, 'bad.til')}:{line}: "), out

    @pytest.mark.parametrize("src,line,what", UNSUPPORTED_PROGRAMS)
    def test_unsupported(self, tilc, src, line, what):
        rc, out, data = tilc.compile(src, "u.til")
        assert rc == 1 and data is None
        assert out.strip() == f"tilc: {os.path.join(tilc.dir, 'u.til')}:{line}: unsupported: {what}"

    def test_usage_and_missing_file(self, tilc):
        r = subprocess.run([tilc.vm, tilc.path], capture_output=True, text=True)
        assert r.returncode == 2 and "usage: tilc INPUT.til OUTPUT.tbc" in r.stdout
        r = subprocess.run([tilc.vm, tilc.path, "/nonexistent/x.til", "o.tbc"], capture_output=True, text=True)
        assert r.returncode == 1 and "tilc: cannot read /nonexistent/x.til" in r.stdout


# ═══════════════════════════════════════════════════════════════
# EFFECT / CAPABILITY CHECKS FOR EVERY SPELLING (review round 1)
# ═══════════════════════════════════════════════════════════════

# tilc used to decide from the raw text whether the SemanticChecker was
# needed and missed localized imports, tabs and some attribute layouts: the
# checks were skipped and the program compiled. Now the token stream decides.
SC_MODULES = {
    "netcap.til": "cap Net { send }\n#[cap: Net]\nfetch()\n    return 42\n",
    "netpure.til": "#[pure]\ncalc(x)\n    print(\"side effect\")\n    return x\n",
}
SNEAKY = "sneaky()\n    return fetch()\nmain()\n    print(sneaky())\n"
SC_PROGRAMS = {
    "kk_import": "импорт netcap\n" + SNEAKY,
    "kk_from": "бастап netcap импорт fetch\n" + SNEAKY,
    "ru_import": "#[lang: ru]\nимпорт netcap\n" + SNEAKY,
    "ru_from": "#[lang: ru]\nиз netcap импорт fetch\n" + SNEAKY,
    "kk_latn_from": "#[lang: kk-latn]\nbastap netcap import fetch\n" + SNEAKY,
    "tab_import": "import\tnetcap\n" + SNEAKY,
    "tab_from": "from\tnetcap import fetch\n" + SNEAKY,
    "module_effect": "import\tnetpure\nmain()\n    print(calc(1))\n",
    "two_attrs_cap": "cap Net { send }\n#[inline] #[cap: Net]\nfetch()\n    return 1\n" + SNEAKY,
    "two_attrs_pure": "#[inline] #[pure]\nsq(x)\n    print(x)\n    return x * x\nmain()\n    print(sq(3))\n",
    "spaces_effects": "#[    effects: fs]\nf()\n    print(1)\nmain()\n    f()\n",
    "spaces_cap": "cap Net { send }\n#[         cap: Net]\nfetch()\n    return 1\n" + SNEAKY,
    "trailing_attr": "x() => 1 #[pure]\nsq(x)\n    print(x)\n    return x * x\nmain()\n    print(sq(3))\n",
    "attr_and_comment": "#[ pure ]  # no side effects\nsq(x)\n    print(x)\n    return x * x\nmain()\n    print(sq(3))\n",
    "method_attrs": "struct P\n    x: int\nimpl P\n    #[inline] #[pure]\n    get(self)\n"
                    "        print(self.x)\n        return self.x\nmain()\n    print(P { x: 1 }.get())\n",
}


def py_error(src: str, path: str):
    """(code, message, line) of the Python compiler's first error."""
    try:
        py_compile(src, path)
    except (til.CompileError, SyntaxError) as e:
        text = str(e)
        m = re.match(r'error\[(E-[A-Z]+)\]: (.*)\n --> .*:(\d+):\d+', text)
        assert m, text
        return m.group(1), m.group(2), int(m.group(3))
    return None


@needs_cc
class TestEffectChecksAnySpelling:
    @pytest.mark.parametrize("name", sorted(SC_PROGRAMS))
    def test_rejected_like_python(self, tilc, tmp_path, name):
        for mod, text in SC_MODULES.items():
            (tmp_path / mod).write_text(text, encoding='utf-8')
        main = tmp_path / (name + ".til")
        main.write_text(SC_PROGRAMS[name], encoding='utf-8')
        err = py_error(SC_PROGRAMS[name], str(main))
        assert err and err[0] in ("E-CAP", "E-EFFECT"), err
        rc, out, data = tilc.compile_file(str(main), str(tmp_path / "out.tbc"))
        assert rc == 1 and data is None, f"tilc accepted it ({err})"
        assert out.strip() == f"tilc: {main}:{err[2]}: error: {err[1]}"

    def test_valid_programs_still_identical(self, tilc, tmp_path):
        """The same spellings without a violation compile to the Python compiler's bytes."""
        (tmp_path / "netcap.til").write_text(SC_MODULES["netcap.til"], encoding='utf-8')
        ok = "#[cap: Net]\nsneaky()\n    return fetch()\nmain()\n    print(sneaky())\n"
        for name, src in (("kk", "импорт netcap\n" + ok), ("ru", "#[lang: ru]\nиз netcap импорт fetch\n" + ok),
                          ("tab", "import\tnetcap\n" + ok),
                          ("attrs", "#[inline] #[pure]\nsq(x) => x * x\nmain()\n    print(sq(3))\n")):
            main = tmp_path / (name + ".til")
            main.write_text(src, encoding='utf-8')
            rc, out, data = tilc.compile_file(str(main), str(tmp_path / "out.tbc"))
            assert rc == 0, (name, out)
            ref = py_compile(src, str(main))
            assert data == ref, name


# ═══════════════════════════════════════════════════════════════
# UNICODE, CONSTANT FOLDING, LIMITS (review round 1, continued)
# ═══════════════════════════════════════════════════════════════

UNICODE_TABLES = os.path.join(ROOT, 'selfhost', 'unicode_tables.py')


def unicode_tables():
    import importlib.util
    spec = importlib.util.spec_from_file_location('unicode_tables', UNICODE_TABLES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestUnicodeTables:
    """tilc.til embeds tables of str.isalpha/isdecimal/isdigit/isalnum and
    isprintable generated by selfhost/unicode_tables.py."""

    def test_in_sync_with_this_python(self):
        # same Unicode version: byte for byte; older Pythons: every character
        # they know has the same class (newer versions only add characters)
        assert unicode_tables().check(tilc_text()) == []
        r = subprocess.run([sys.executable, UNICODE_TABLES, '--check'], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout

    def test_check_finds_a_wrong_entry(self):
        ut = unicode_tables()
        text = tilc_text()
        a, b = ut.block_span(text)
        block = text[a:b]
        assert '("##3#' in block            # U+0080 starts a run of class 0
        bad = text[:a] + block.replace('("##3#', '("##3$', 1) + text[b:]   # U+0080 a letter
        assert ut.check(bad)


def python_result(src: str, path: str):
    """('ok', bytes) or ('error', line, message) of the Python compiler."""
    try:
        return ('ok', py_compile(src, path))
    except (til.CompileError, SyntaxError) as e:
        m = re.match(r'(?:error(?:\[E-[A-Z]+\])?: )(.*?)\n --> .*?:(\d+):\d+', str(e), re.S)
        assert m, str(e)
        return ('error', int(m.group(2)), m.group(1))


def assert_like_python(tilc: Tilc, src: str, name: str = "u.til"):
    """tilc gives the Python compiler's bytes, or its first error at the same line."""
    path = os.path.join(tilc.dir, name)
    ref = python_result(src, path)
    if ref[0] == 'ok':
        assert check_program(tilc, name, src) in ('same', 'behaviour')
    else:
        rc, out, data = tilc.compile(src, name)
        assert rc == 1 and data is None, f"tilc accepted it; Python: {ref}"
        assert out.strip() == f"tilc: {path}:{ref[1]}: error: {ref[2]}"


P1 = 'main()\n    print(%s)\n'
# characters of Unicode 13 or older (the oldest Python of the CI matrix)
LEXICAL = {
    "bom": '\ufeffmain()\n    print("hi")\n',
    "bom_inside": 'main()\n    print("hi")\ufeff\n',
    "nbsp": 'main()\n    let x\u00a0= 1\n',
    "zero_width": 'main()\n    let x\u200b = 1\n',
    "line_separator": 'main()\n    let x = 1\u2028\n',
    "emoji_name": 'main()\n    let a\U0001f600 = 2\n',
    "roman_numeral_start": 'main()\n    let \u216b = 2\n',
    "letters": 'main()\n    let na\u00efve_\u65e5\u672c = 3\n    let x\u216b\u0663 = 4\n    let \u01c5 = 5\n'
               '    print(na\u00efve_\u65e5\u672c, x\u216b\u0663, \u01c5)\n',
    "arabic_indic": P1 % '\u0661\u0662\u0663 + 1',
    "devanagari_float": P1 % '\u0663.\u0665e-\u0662',
    "math_digits": P1 % '\U0001d7d9\U0001d7da_0',
    "superscript": P1 % '1\u00b2',
    "superscript_exponent": P1 % '1e\u00b2',
    "superscript_start": P1 % '\u00b2',
    "hex_unicode_digit": P1 % '0x\u0661',
    "int_4300_digits": P1 % ('7' * 4300),
    "int_4301_digits": P1 % ('7' * 4301),
    "float_4400_digits": P1 % ('7' * 4400 + '.5'),
    "escape_spaces": P1 % '"\\u{ 41 }\\u{\u300042\u3000}\\u{43\t}"',
    "escape_sign_prefix": P1 % '"\\u{+41}\\u{0x42}\\u{0X_43}\\u{-0}\\u{4_4}\\u{\u0664\u0665}"',
    "escape_negative": P1 % '"\\u{-1}"',
    "escape_double_underscore": P1 % '"\\u{4__1}"',
    "escape_trailing_underscore": P1 % '"\\u{41_}"',
    "escape_superscript": P1 % '"\\u{\u00b2}"',
    "escape_x1c": P1 % '"\\u{\x1c41}"',
    "spec_unicode_width": 'main()\n    let x = 5\n    print(f"{x:\u0665}|{x:.\u0662f}")\n',
    "spec_superscript": 'main()\n    let x = 5\n    print(f"{x:\u00b2}")\n',
    "pragma_unicode_spaces": '#[\u3000lang\u3000:\u3000ru\u2003]\n\u0444\u0443\u043d\u043a\u0446\u0438\u044f f()\n'
                             '    \u0432\u0435\u0440\u043d\u0443\u0442\u044c 1\nmain()\n    print(f())\n',
    "pragma_kelvin_sign": '#[lang: \u212ak-latn]\nnegizgi()\n    basyp_shygaru(1)\n',
    "attribute_nbsp": '#[\u00a0pure\u00a0]\nsq(x)\n    print(x)\n    return x * x\nmain()\n    print(sq(3))\n',
}


@needs_cc
class TestUnicodeLexing:
    """The lexer classifies characters like Python's str methods: a BOM or a
    zero-width space is an unexpected character (tilc used to take every
    byte >= 128 for a letter: a BOM renamed main() and the program did
    nothing), other scripts' digits are numbers, white space is Unicode
    white space, and \\u{...} is read with int(text, 16)."""

    @pytest.mark.parametrize("name", sorted(LEXICAL))
    def test_like_python(self, tilc, name):
        assert_like_python(tilc, LEXICAL[name])

    def test_bom_is_not_a_silent_no_op(self, tilc):
        rc, out, data = tilc.compile(LEXICAL["bom"], "bom.til")
        assert rc == 1 and out.strip().endswith(":1: error: Unexpected character: '\ufeff'"), out

    def test_invalid_utf8(self, tilc, tmp_path):
        """Python reads sources with encoding="utf-8": the UnicodeDecodeError text."""
        for raw in (b'main()\n    print("a\xff")\n', b'main()\r\n    print("\xe2\x82")\r\n',
                    b'main()\n    print(1)\n\xf4\x90\x80\x80\n'):
            path = tmp_path / "bad8.til"
            path.write_bytes(raw)
            with pytest.raises(UnicodeDecodeError) as e:
                raw.decode('utf-8')
            line = raw[:e.value.start].replace(b'\r\n', b'\n').count(b'\n') + 1
            rc, out, data = tilc.compile_file(str(path), str(tmp_path / "out.tbc"))
            assert rc == 1 and out.strip() == f"tilc: {path}:{line}: error: {e.value}"


CONTRACT_STR = '#[requires: len(s) == 0]\nf(s)\n    return s\nmain()\n    print(f(%s))\n'


@needs_cc
class TestContractMessages:
    """E-CONTRACT shows arguments with Python's repr(): characters that are not
    str.isprintable() are escaped (\\xhh, \\uhhhh, \\Uhhhhhhhh)."""

    @pytest.mark.parametrize("arg", [
        '"nbsp\\u{a0}|nel\\x85|c1\\u{80}|ls\\u{2028}"',
        '"\\u{ad}\\u{3000}\\u{180e}\\u{61c}\\u{200b}\\u{feff}\\u{2066}"',
        '"\\u{e000}\\u{f0000}\\u{10ffff}\\u{378}\\u{e0001}\\u{9f}"',
        '"printable: \u0259 \U0001f600 \\u{300} \\u{1d400} \\u{d7ff}"',
        '"quotes \' and \\" \\x1b \\x7f \\t"',
    ])
    def test_repr_like_python(self, tilc, arg):
        assert_like_python(tilc, CONTRACT_STR % arg)


BIG = '18446744073709551616'
CONTRACT_X = '#[requires: %s]\nf(x)\n    return x\nmain()\n    print(f(%s))\n'
ACCEPT_REJECT = {
    # the SemanticChecker resolves `Vec<int>.new()` by method name: with a
    # user new(x) it reports a (false) arity error; tilc does the same
    "vec_new_user_new": "struct P\n    x: int\nimpl P\n    new(x) -> P\n        return P { x: x }\nmain()\n"
                        "    var v = Vec<int>.new()\n    v.push(1)\n    print(v, P.new(2))\n",
    "hashmap_new_user_new": "struct P\n    x: int\nimpl P\n    new(x) -> P\n        return P { x: x }\nmain()\n"
                            "    var m = HashMap<str, int>.new()\n    print(m)\n",
    "vec_new_no_conflict": "struct P\n    x: int\nimpl P\n    new() -> P\n        return P { x: 0 }\nmain()\n"
                           "    var v = Vec<int>.new()\n    print(v, P.new())\n",
    # a nan argument is a constant of the contract check
    "nan_argument": CONTRACT_X % ('x == x', '1e309 - 1e309'),
    "nan_argument_ok": CONTRACT_X % ('x != x', '1e309 * 0.0'),
    "nan_global": 'let N = 1e309 - 1e309\n' + CONTRACT_X % ('x == x', 'N'),
    # literals of 2^63 and more are exact in constant folding
    "big_argument_ok": CONTRACT_X % ('x > 0', '9223372036854775808'),
    "big_argument_bad": CONTRACT_X % ('x < 0', '9223372036854775808'),
    "big_argument_div": CONTRACT_X % ('x / 2 == 0', BIG),
    "big_argument_float": CONTRACT_X % ('x < 1.0', BIG),
    "big_folding": P1 % ', '.join([
        BIG + ' / 2', BIG + ' % 7', '-' + BIG + ' / 3', '9223372036854775808 / -1', '100 / ' + BIG,
        '-7 % ' + BIG, BIG + ' == 0', 'not ' + BIG, BIG + ' or 3', BIG + ' < 18446744073709551617',
        BIG + ' * 1.0', BIG + ' == 1.8446744073709552e19', '0x1_0000_0000_0000_0000 / 16',
        '340282366920938463463374607431768211457 % 1000', '-' + BIG, '~' + BIG, BIG + ' + 1']),
    "big_division_by_zero": P1 % (BIG + ' / 0'),
    "big_const": 'const BIG = ' + BIG + '\nmain()\n    print(BIG / 2, BIG > 0, BIG % 10)\n',
    # nan comparisons and truthiness fold (only a nan result is left to run time)
    "nan_comparisons": P1 % '(1e309 - 1e309) != 0.0, not (1e309 - 1e309), (1e309 - 1e309) < 1',
}


@needs_cc
class TestAcceptRejectLikePython:
    @pytest.mark.parametrize("name", sorted(ACCEPT_REJECT))
    def test_like_python(self, tilc, name):
        assert_like_python(tilc, ACCEPT_REJECT[name])


@needs_cc
class TestLimits:
    """Valid programs that tilc used to fail on with VM errors."""

    def test_function_over_64k_of_bytecode(self, tilc):
        # the code buffer was one VM array (at most 65535 elements)
        src = "main()\n" + "".join(f"    print(i_{i % 7} % 100)\n" for i in range(12000))
        src = "".join(f"let i_{k} = {k}\n" for k in range(7)) + src
        ref = py_compile(src)
        assert len(ref) > 70000
        rc, out, data = tilc.compile(src)
        assert rc == 0, out
        assert data == ref

    def test_long_operator_chains(self, tilc):
        # flat chains are folded and compiled with loops (no deep recursion)
        # (sizes the Python compiler handles under pytest; it stops near 1000)
        for src in (P1 % ' + '.join(['1'] * 850), P1 % ('not ' * 800 + 'true'), P1 % ('-' * 800 + '1'),
                    'main()\n    let x = 3\n    print(' + ' + '.join(['x', '1'] * 150) + ')\n',
                    'main()\n    let x = 3\n    print(' + ' and '.join(['x > 1'] * 300) + ')\n',
                    'main()\n    let f = |y| ' + ' + '.join(['y'] * 300) + '\n    print(f(2))\n'):
            assert_like_python(tilc, src)

    def test_chain_longer_than_python_can_compile(self, tilc):
        """The Python compiler stops at ~1000 terms (recursion limit); tilc does not."""
        src = "main()\n    let x = 2\n    print(" + " + ".join(["x"] * 3000) + ")\n"
        rc, out, data = tilc.compile(src)
        assert rc == 0, out
        assert run_vm(data) == (0, "6000\n")

    def test_unwritable_output(self, tilc):
        src_path = os.path.join(tilc.dir, "ok.til")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write("main()\n    print(1)\n")
        rc, out, data = tilc.compile_file(src_path, "/nonexistent-dir/out.tbc")
        assert rc == 1 and out.startswith("tilc: cannot write /nonexistent-dir/out.tbc ("), out

    def test_long_programs_on_the_calculator(self, tilc):
        """64 KB heap: script statements share one item, code and line tables
        are kept as bytes (tilc used to stop after ~100 script statements or
        ~240 statements in a function)."""
        vm = cvm_binary("host", CALC_FLAGS)
        for src in ("".join(f"print({i})\n" for i in range(1000)),
                    "main()\n" + "".join(f"    print({i})\n" for i in range(800))):
            rc, out, data = tilc.compile(src, "long.til", vm=vm)
            assert rc == 0, out
            assert data == py_compile(src)

    @pytest.mark.skipif(not SLOW, reason="slow (~20 s, 64 MB heap): set TIL_TEST_SLOW=1")
    def test_constant_table(self, tilc):
        """More than 32767 constants (a C VM map holds at most that many) and the
        65535-constant limit of the format, reported like the Python compiler."""
        vm = cvm_binary("host", ('-DTILVM_HEAP_SIZE=67108864',))
        for n in (33000, 66000):
            src = "main()\n" + "".join(f"    print({100000 + i})\n" for i in range(n))
            path = os.path.join(tilc.dir, f"consts{n}.til")
            ref = python_result(src, path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)
            rc, out, data = tilc.compile_file(path, os.path.join(tilc.dir, "out.tbc"), vm=vm, timeout=300)
            if ref[0] == 'ok':
                assert rc == 0 and data == ref[1], out
            else:
                assert out.strip() == f"tilc: {path}:{ref[1]}: error: {ref[2]}"


# ═══════════════════════════════════════════════════════════════
# C VM: measuring the text of a huge value (review round 1)
# ═══════════════════════════════════════════════════════════════

# 16 references to 16 references ... to a 64 KB string: the text of `a` is
# 4 GB. The C VM measured it in a uint32 that wrapped, allocated a small
# string and wrote 4 GB into it (heap overflow, segfault). The count now
# stops above the 16 MB string limit: "string too long".
HUGE_TEXT = """
main()
    let s = "x".repeat(65532)
    var a = [s]
    for level in 0..LEVELS
        var b = []
        for i in 0..16
            b.push(a)
        a = b
    print(len(a))
    let t = EXPR
    print(len(t))
"""


@needs_cc
class TestVMHugeText:
    @pytest.mark.parametrize("levels,expr", [(4, '[a].join("")'), (4, 'str(a)'), (4, 'f"{a}"'),
                                             (4, 'a.join(",")'), (15, 'str(a)')])
    def test_string_too_long(self, levels, expr):
        src = HUGE_TEXT.replace("LEVELS", str(levels)).replace("EXPR", expr)
        rc, out, err = run_cvm(src, timeout=30)
        assert (rc, out) == (1, "16\n"), (rc, out, err)
        assert err.startswith("runtime error: string too long\n  at main (line 11)"), err

    def test_sanitizers(self, tmp_path):
        """The same programs under AddressSanitizer / UBSan (skipped without them)."""
        exe = str(tmp_path / "tilvm-asan")
        r = subprocess.run([CC, '-std=c99', '-O1', '-g', '-fsanitize=address,undefined',
                            '-fno-sanitize-recover=undefined', '-o', exe,
                            os.path.join(ROOT, 'vm', 'tilvm.c'), os.path.join(ROOT, 'vm', 'main.c'), '-lm'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip("no sanitizer runtime: " + r.stderr[-200:])
        probe = subprocess.run([exe, '--version'], capture_output=True, text=True)
        if "AddressSanitizer" in probe.stderr:
            pytest.skip("AddressSanitizer cannot run here: " + probe.stderr[-200:])
        for expr in ('[a].join("")', 'str(a)'):
            path = tmp_path / "huge.tbc"
            path.write_bytes(py_compile(HUGE_TEXT.replace("LEVELS", "4").replace("EXPR", expr)))
            r = subprocess.run([exe, str(path)], capture_output=True, text=True, timeout=60)
            assert r.returncode == 1 and "runtime error: string too long" in r.stderr, r.stderr[-2000:]


# ═══════════════════════════════════════════════════════════════
# C VM FIX: big arrays (found while building tilc's output)
# ═══════════════════════════════════════════════════════════════

LARGE_ARRAYS = """
main()
    var parts = []
    for i in 0..10000
        parts.push(str(i % 10))
    let s = parts.join("")
    print(len(s), s.slice(0, 12), len(parts.join(", ")))
    print(len(parts.copy()), len(parts.slice(1, 9500)), len(parts + parts))
    var m = {}
    for i in 0..9000
        m[i] = i * 2
    print(len(m.keys()), m.values()[8999], len(m.copy()))
    var n = 0
    for k in m
        n += 1
    print(n, len(range(20000)), range(5, 20000)[19994])
    print(len(s.split("")), len(("a," * 12000).split(",")), len((" x" * 9000).split()))
"""


@needs_cc
class TestVMLargeArrays:
    """join/copy/slice/+/keys/values/copy/iter/range/split staged every element
    on the C VM's value stack (8192 slots): arrays of more than ~8000 elements
    failed with "stack overflow" on the C VM but worked on the Python VM."""

    # small stack: nothing may be staged on it; small heap: the GC runs often
    @pytest.mark.parametrize("flags", [('-DTILVM_HEAP_SIZE=4194304',),
                                       ('-DTILVM_HEAP_SIZE=2097152', '-DTILVM_STACK_SIZE=1024')])
    def test_both_vms_agree(self, flags):
        expected = "".join(f"{v}\n" for v in (10000, "012345678901", 29998, 10000, 9499, 20000, 9000,
                                                17998, 9000, 9000, 20000, 19999, 10000, 12001, 9000))
        data = py_compile(LARGE_ARRAYS)
        assert run_vm(data) == (0, expected)
        path = os.path.join(tempfile.mkdtemp(prefix='tilvm-big-'), 'big.tbc')
        with open(path, 'wb') as f:
            f.write(data)
        r = subprocess.run([cvm_binary("host", flags), path], capture_output=True, timeout=60)
        assert r.returncode == 0, r.stderr
        assert r.stdout.decode() == expected


# ═══════════════════════════════════════════════════════════════
# CALCULATOR CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CALC_PROGRAM = """
# a small calculator program: compiled ON the calculator
struct Account
    owner: str
    balance: float

impl Account
    deposit(self, amount: float)
        self.balance += amount

interest(principal: float, rate: float, years: int) -> float
    var total = principal
    for y in 0..years
        total = total * (1.0 + rate)
    return total

fib(n: int) -> int
    var a = 0
    var b = 1
    for i in 0..n
        let t = a + b
        a = b
        b = t
    return a

main()
    let acc = Account { owner: "Aigerim", balance: 100.0 }
    acc.deposit(25.5)
    print(f"{acc.owner}: {acc.balance}")
    print(interest(1000.0, 0.05, 10))
    var fibs = []
    for i in 0..15
        fibs.push(fib(i))
    print(fibs)
    let big = 3000000000
    print(big * 2)
"""


@needs_cc
class TestCalculator:
    def test_compile_on_int32_float32_64k_heap(self, tilc):
        vm = cvm_binary("host", CALC_FLAGS)
        rc, out, data = tilc.compile(CALC_PROGRAM, "calc.til", vm=vm)
        assert rc == 0, out
        # the same bytes as tilc on a PC (and as the Python compiler)
        assert data == tilc.compile(CALC_PROGRAM, "calc.til")[2]
        ref = py_compile(CALC_PROGRAM)
        assert data == ref
        # the result runs on a 64-bit VM (the int32 VM rejects 6000000000 by design)
        rc, out = run_vm(data)
        assert rc == 0 and out.splitlines()[0] == "Aigerim: 125.5"

    def test_examples_on_calculator_vm(self, tilc):
        vm = cvm_binary("host", CALC_FLAGS)
        for name in ('01_hello.til', '03_functions.til', '04_structs.til', '05_control_flow.til',
                     '07_multilevel.til', '09_v3_features.til', 'basic/control_flow.til'):
            path = os.path.join(EXAMPLES, name)
            rc, out, data = tilc.compile_file(path, os.path.join(tilc.dir, 'calc_out.tbc'), vm=vm)
            assert rc == 0, (name, out)
            assert data == tilc.compile_file(path, os.path.join(tilc.dir, 'pc_out.tbc'))[2], name


# ═══════════════════════════════════════════════════════════════
# TIL 3.1 IN TILC: superinstructions, effects, imports, folding fixes
# ═══════════════════════════════════════════════════════════════

def tilc_string(name: str) -> str:
    m = re.search(r'^var ' + name + r' = "(.*)"$', tilc_text(), re.M)
    assert m, name
    return m.group(1)


class TestTablesInSync31:
    """Tables tilc.til copies from src/til.py for the TIL 3.1 features."""

    def test_contextual_keywords(self):
        names = {'en': 'CTX_EN', 'kk': 'CTX_KK', 'kk-latn': 'CTX_KK_LATN', 'ru': 'CTX_RU'}
        assert set(names) == set(til.CONTEXTUAL_KEYWORDS)
        for code, var in names.items():
            mine = {k: v[2:] for k, v in til_literal(tilc_text(), var).items()}
            assert mine == {k: t.name for k, t in til.CONTEXTUAL_KEYWORDS[code].items()}, code

    def test_superinstruction_tables(self):
        size = ''.join(chr(48 + 1 + sum(til._OPERAND_SIZE[k] for k in til.OP_OPERANDS.get(op, '')))
                       if op in til.OP_NAMES else '0' for op in range(256))
        assert tilc_string('OP_SIZE') == size
        assert tilc_string('OP_JUMPS') == ''.join('1' if op in til._JUMP_OPS else '0' for op in range(256))
        assert tilc_string('FUSED_IX') == ''.join(chr(65 + til.FUSED_BINOPS.index(op))
                                                  if op in til.FUSED_BINOPS else '.' for op in range(64))
        assert re.search(r'^const F_COST = (\d+)$', tilc_text(), re.M).group(1) == str(til._COST)

    def test_stack_effect_tables(self):
        fx = ''.join(chr(80 + til._STACK_EFFECT[op]) if op in til._STACK_EFFECT else
                     '?' if op in til._OPERAND_EFFECT else '.' for op in range(128))
        assert tilc_string('DEPTH_FX') == fx
        assert tilc_string('JUMP_LAND') == ''.join(chr(80 + til._JUMP_LANDING[op]) if op in til._JUMP_LANDING
                                                   else '.' for op in range(128))
        assert tilc_string('ENDS_FLOW') == ''.join('1' if op in til._ENDS_FLOW else '0' for op in range(128))
        # dep_n: the operand-dependent effects tilc knows
        assert {op: spec for op, spec in til._OPERAND_EFFECT.items() if op < 0x80} == {
            til.Op.CALL: (1, 1, 1), til.Op.CALL_VALUE: (0, 1, 0), til.Op.INVOKE: (2, 1, 0),
            til.Op.NATIVE: (1, 1, 1), til.Op.NEW_ARRAY: (0, 1, 1), til.Op.NEW_MAP: (0, 2, 1),
            til.Op.NEW_STRUCT: (1, 1, 1), til.Op.MAKE_CLOSURE: (1, 1, 1), til.Op.CONCAT: (0, 1, 1),
            til.Op.PERFORM: (2, 1, 1)}


EFFECT_PROGRAMS = {
    'handle_forms': '''
greet() -> str
    let name = perform Ask("name?")
    perform Log("hello " + name)
    return name

counter(n: int) -> int
    var total = 0
    for i in 0..n
        total += perform Tick(i)
    return total

main()
    let r = handle
        greet()
        "done"
    with
        Ask(question) => resume("Aigerim")
        Log(msg) =>
            print(msg)
            resume(None)
        Fail(reason) => "failed: " + reason
    print(r)
    print(handle counter(4) with Tick(k: float) => resume(k * 2))
    let aborted = handle
        perform Fail("no disk")
        "never"
    with
        Fail(reason) => "failed: " + reason
    print(aborted)
    print(handle perform Ask() with Ask => resume(7))
''',
    'nested_and_masking': '''
Log(m)
    print("default " + m)

inner() -> int
    perform Log("inner")
    return perform Get() + 1

main()
    let base = 10
    var seen = []
    let v = handle
        handle
            inner() * 2
        with
            Get() => resume(perform Get() + base)
    with
        Get() => resume(100)
        Log(m) =>
            seen.push(m)
            resume()
    print(v, seen)
    perform Log("outside")
    let r = handle
        let a = perform Outer()
        let b = retry(1)
            perform Inner()
        else
            error
        f"{a} {b}"
    with
        Outer() =>
            handle
                resume(1)
            with
                Inner() => resume(2)
    print(r)
''',
    'loops_and_closures': '''
count_even(n: int) -> int
    var total = 0
    for x in 0..n
        total += handle
            if x % 2 == 1
                continue
            perform One()
        with
            One() => resume(1)
    return total

first_big(xs) -> int
    var found = -1
    for x in xs
        found = handle
            if perform Big(x)
                break
            x
        with
            Big(v) => resume(v > 10)
    return found

main()
    print(count_even(9), first_big([1, 5, 20, 3]))
    let k = 5
    let f = |x| x + k
    let r = handle
        [1, 2, 3].map(|v| perform Scale(v)).reduce(|a, b| a + b, 0)
    with
        Scale(v) => resume(v * k + len(str(f)))
    print(r)
    var log = []
    let n = handle
        perform Note("a")
        perform Note("b")
        len(log)
    with
        Note(t) =>
            log.push(t + str(k))
            resume(len(log))
    print(n, log)
''',
    'localized': '''#[lang: ru]
#[lang: kk-latn]
функция спросить() -> int
    вернуть выполнить Число(2) * 10

main()
    let a = обработать спросить() с Число(x) => возобновить(x + 1)
    let b = өңдеу орындау Ж() арқылы Ж() => жалғау(7)
    let c = ondeu
        oryndau Q()
    arqyly
        Q() => zhalgau(9)
    let with = 1
    let resume = 2
    print(a + b + c + with + resume)
''',
    'statement_and_values': '''
struct Box
    v: int

Tick()
    return 1

main()
    perform Tick()
    handle
        print(perform Tick())
    with
        Tick() => resume(5)
    let b = Box { v: handle perform Tick() with Tick() => resume(3) }
    print(b.v)
    let m = match handle perform Tick() with Tick() => resume(2)
        2 => "two"
        _ => "other"
    print(m)
    let t = handle
        timeout(1000)
            perform Tick()
        else
            -1
    with
        Tick() => resume(8)
    print(t)
''',
    'unhandled_effect': '''
main()
    print("before")
    perform Missing(1)
    print("after")
''',
}

EFFECT_ERRORS = {
    'duplicate_clause': 'main()\n    let r = handle\n        1\n    with\n        A() => 1\n        A() => 2\n',
    'duplicate_param': 'main()\n    let r = handle\n        1\n    with\n        A(x, x) => 1\n',
    'no_with': 'main()\n    let r = handle\n        1\n    print(r)\n',
    'no_clause_arrow': 'main()\n    let r = handle 1 with A() 2\n',
    'arity': 'main()\n    let r = handle\n        perform A(1, 2)\n    with\n        A(x) => resume(x)\n',
    'return_in_clause': 'main()\n    let r = handle\n        perform A()\n    with\n        A() =>\n            return 1\n',
    'break_in_clause': 'main()\n    for i in 0..3\n        let r = handle\n            perform A()\n        with\n            A() =>\n                break\n',
    'resume_outside': 'main()\n    resume(1)\n',
    'resume_in_lambda': 'main()\n    let r = handle\n        perform A()\n    with\n        A() => (|x| resume(x))(1)\n',
    'resume_param': 'main()\n    let r = handle\n        perform A(1)\n    with\n        A(resume) => 1\n',
    'resume_let': 'main()\n    let r = handle\n        perform A()\n    with\n        A() =>\n            let resume = 1\n            resume(1)\n',
    'resume_assign': 'var resume = 0\nmain()\n    let r = handle\n        perform A()\n    with\n        A() =>\n            resume = 1\n            resume(1)\n',
    'value_lost_let': 'main()\n    let r = handle\n        perform A()\n    with\n        A() =>\n            let v = resume(1)\n            v\n',
    'value_lost_call': 'main()\n    let r = handle\n        perform A()\n    with\n        A() => print(resume(1))\n',
    'value_lost_binop': 'main()\n    let r = handle\n        perform A()\n    with\n        A() =>\n            resume(1) + 1\n',
    'value_lost_and': 'main()\n    let r = handle\n        perform A(true)\n    with\n        A(x) =>\n            resume(1) and x\n',
    'resume_resume': 'main()\n    let r = handle\n        perform A()\n    with\n        A() => resume(resume(1))\n',
    'after_if_else': 'main()\n    let r = handle\n        perform A(true)\n    with\n        A(x) =>\n            if x\n                resume(1)\n            else\n                resume(2)\n            print("dead")\n',
    'if_cond': 'main()\n    let r = handle\n        perform A()\n    with\n        A() =>\n            if resume(1)\n                print("x")\n',
    'while_cond': 'main()\n    let r = handle\n        perform A()\n    with\n        A() =>\n            while resume(1)\n                print("x")\n',
    'match_subject': 'main()\n    let r = handle\n        perform A()\n    with\n        A() =>\n            match resume(1)\n                1 => print("x")\n                _ => print("y")\n',
    'after_resume': 'main()\n    let r = handle\n        perform A()\n    with\n        A() =>\n            resume(1)\n            print("x")\n',
    'nested_clause_first': 'main()\n    let r = handle\n        perform A()\n    with\n        A() =>\n            let q = handle\n                perform B()\n            with\n                B() =>\n                    resume(2)\n                    print("inner")\n            resume(q)\n            print("outer")\n',
    'reserved_perform': 'let perform = 1\n',
    'reserved_handle_method': 'struct T\n    x: int\nimpl T\n    handle(self) -> int\n        return 1\n',
    'pure_perform': '#[pure]\nf() -> int\n    return perform Get()\nmain()\n    print(handle f() with Get() => resume(1))\n',
    'pure_handled': '#[pure]\nf() -> int\n    return handle perform Get() with Get() => resume(1)\nmain()\n    print(f())\n',
    'effects_declared': '#[effects: Get]\nf() -> int\n    return perform Get()\n#[effects: none]\ng() -> int\n    return f()\nmain()\n    print(handle g() with Get() => resume(1))\n',
    'handled_call': '#[effects: Get]\nf() -> int\n    return perform Get()\n#[pure]\ng() -> int\n    return handle f() with Get() => resume(4)\nmain()\n    print(g())\n',
    'io_clause_does_not_hide_print': '#[pure]\nf() -> int\n    return handle print(1) with io() => 0\nmain()\n    print(f())\n',
    'io_shadowed_by_param': '#[pure]\napply(print, x)\n    return print(x)\nmain()\n    print(apply(|v| v * 2, 4))\n',
    'io_shadowed_by_function': 'print_it(x)\n    return x\n#[pure]\nf(x)\n    return print_it(x)\nmain()\n    print(f(1))\n',
    'cap_perform': 'cap Secret { read }\n#[cap: Secret]\nSecret() -> str\n    return "s"\nleak() -> str\n    return perform Secret()\nmain()\n    print(leak())\n',
    'cap_perform_handled': 'cap Secret { read }\n#[cap: Secret]\nSecret() -> str\n    return "s"\nleak() -> str\n    return handle perform Secret() with Secret() => resume("fake")\nmain()\n    print(leak())\n',
    'cap_lambda_escapes': 'cap Secret { read }\n#[cap: Secret]\nSecret() -> str\n    return "s"\nleak() -> str\n    let f = handle\n        |x| perform Secret()\n    with\n        Secret() => resume("fake")\n    return f(0)\nmain()\n    print(leak())\n',
    'cap_masked_in_clause': 'cap Secret { read }\n#[cap: Secret]\nSecret() -> str\n    return "s"\nleak() -> str\n    return handle\n        perform Secret()\n    with\n        Secret() => resume(perform Secret())\nmain()\n    print(leak())\n',
    'cap_struct_field': 'struct Box\n    v: str\ncap Secret { read }\n#[cap: Secret]\nSecret() -> str\n    return "s"\nleak() -> str\n    let b = Box { v: Secret() }\n    return b.v\n',
    'pure_struct_field': 'struct Box\n    v: str\n#[pure]\nnoisy() -> int\n    let b = Box { v: str(print("io")) }\n    return 1\n',
    'arity_struct_field': 'struct Box\n    v: int\nf(a) => a\nmain()\n    print(Box { v: f(1, 2) })\n',
}

FOLDING_FIXES = {
    'shadowing_param': ('let N = 10\nf(N: int) -> int\n    return N + 1\nmain()\n    print(f(5), N + 1)\n', '6\n11\n'),
    'shadowing_let_lambda_for': ('const N = 5\nf() -> int\n    let N = 3\n    return N + 1\nmain()\n    let g = |N| N * 2\n    var r = 0\n    for N in 0..2\n        r = N * 1\n    let M = 2\n    let h = |x| x + N + M * 10\n    print(f(), g(4), r, h(1), N * 2)\n', '4\n8\n1\n26\n10\n'),
    'bool_equality': ('main()\n    print(true == 1, true != 1, 1 == true, true == 1.0, false == 0, true == true, 1 == 1.0)\n', 'false\ntrue\nfalse\nfalse\nfalse\ntrue\ntrue\n'),
}


@needs_cc
class TestEffects:
    """handle / with / resume / perform: the Python compiler's bytes, the same
    output on both VMs; the parser's and the checkers' errors at the same line."""

    @pytest.mark.parametrize("name", sorted(EFFECT_PROGRAMS))
    def test_identical_and_runs(self, tilc, tmp_path, name):
        src = EFFECT_PROGRAMS[name]
        rc, out, data = tilc.compile(src)
        assert rc == 0, out
        assert data == py_compile(src)
        py, c = run_both_vms(data, str(tmp_path))
        assert c[:2] == py[:2] and py[0] == (1 if name == "unhandled_effect" else 0), (py, c)

    @pytest.mark.parametrize("name", sorted(EFFECT_ERRORS))
    def test_like_python(self, tilc, name):
        assert_like_python(tilc, EFFECT_ERRORS[name])

    def test_examples_on_calculator_vm(self, tilc):
        vm = cvm_binary("host", CALC_FLAGS)
        path = os.path.join(EXAMPLES, '12_effects.til')
        rc, out, data = tilc.compile_file(path, os.path.join(tilc.dir, 'calc_eff.tbc'), vm=vm)
        assert rc == 0, out
        with open(path, encoding='utf-8') as f:
            assert data == py_compile(f.read(), path)


BREAK_IN_EXPRESSIONS = """
f(a, b) => a + b
main()
    var total = 0
    for i in 0..6
        total += retry(2)
            if i == 3
                continue
            i
    var arr = [0, 0, 0]
    var i = 0
    while i < 10
        arr[i % 3] += timeout(1000)
            if i == 7
                break
            i
        i += 1
    var s = 0
    loop
        s = f(s, 1) + retry(1)
            if s > 20
                break
            s + 5
    print(total, arr, s)
"""


@needs_cc
class TestSuperinstructions:
    """The peephole pass (fuse_superinstructions) and the operand stack depth
    (break / continue inside expressions pop what the expression pushed)."""

    def test_break_and_continue_inside_expressions(self, tilc, tmp_path):
        assert check_program(tilc, "break", BREAK_IN_EXPRESSIONS) == 'same'
        rc, out, data = tilc.compile(BREAK_IN_EXPRESSIONS)
        py, c = run_both_vms(data, str(tmp_path))
        assert py[:2] == (0, "12\n[9, 5, 7]\n42\n") and c[:2] == py[:2]

    def test_every_superinstruction(self, tilc):
        src = ("struct P\n    x: int\nmain()\n    var a = [1, 2, 3]\n    var i = 0\n    var t = 0.0\n"
               "    let p = P { x: 4 }\n    while i < 3\n        t = t + a[i]\n        i += 1\n"
               "    for v in a\n        if v == 2\n            print(a[v] + p.x)\n        if v > i\n"
               "            print(v)\n        if 2 < v\n            print(v - 1)\n    let k = 2\n"
               "    var c = 0\n    c = c + k\n    print(c * 3, i << 1, k == 2, a[i - 1], t)\n"
               "f(x: float) -> int\n    var y: float = x\n    return y\ng(v) => v\n")
        rc, out, data = tilc.compile(src)
        assert rc == 0, out
        assert data == py_compile(src)
        ops = set()
        for f in til.TBCModule.from_bytes(data).funcs:
            pc = 0
            while pc < len(f.code):
                ops.add(f.code[pc])
                pc += 1 + sum(til._OPERAND_SIZE[k] for k in til.OP_OPERANDS.get(f.code[pc], ''))
        assert len([op for op in ops if op >= 0x80]) >= 10, sorted(ops)

    def test_no_fuse(self, tilc):
        src = "main()\n    var s = 0\n    for i in 0..10\n        s += i\n    print(s)\n"
        path = os.path.join(tilc.dir, "nofuse.til")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        out_path = os.path.join(tilc.dir, "nofuse.tbc")
        r = subprocess.run([tilc.vm, tilc.path, "--no-fuse", path, out_path], capture_output=True, timeout=60)
        assert r.returncode == 0, r.stdout
        with open(out_path, "rb") as f:
            data = f.read()
        assert data == til.compile_to_bytecode(src, path, fuse=False).serialize() != py_compile(src)
        assert run_vm(data) == (0, "45\n")

    def test_long_lines(self, tilc):
        """A run of thousands of instructions (no line entry, no jump target)
        is cut only where no superinstruction can span the cut."""
        for items in (["a"] * 9001, ["a + 1", "a", "f(a)", "a < 3"] * 1500):
            src = ("f(x) => x\nmain()\n    let a = 3\n    let b = [" + ", ".join(items) +
                   "]\n    print(len(b))\n")
            assert check_program(tilc, "long", src) == 'same'


@needs_cc
class TestSharedCompilerFixes:
    """Two bugs both compilers had: constant folding used a module constant
    that a local shadows, and folded `~true` / `true == 1` like Python, not
    like the VMs. Folding now follows emit_load_var and the VMs' semantics."""

    @pytest.mark.parametrize("name", sorted(FOLDING_FIXES))
    def test_folding_like_run_time(self, tilc, tmp_path, name):
        src, out = FOLDING_FIXES[name]
        assert check_program(tilc, name, src) == 'same'
        py, c = run_both_vms(py_compile(src), str(tmp_path))
        assert py[:2] == (0, out) and c[:2] == py[:2]

    def test_bitwise_not_of_a_bool_is_a_run_time_error(self, tilc, tmp_path):
        src = "main()\n    print(~5)\n    print(~true)\n"
        assert check_program(tilc, "bnot", src) == 'same'
        py, c = run_both_vms(py_compile(src), str(tmp_path))
        assert py[:2] == (1, "-6\n") and c[:2] == py[:2]
        assert "~ needs an integer" in py[2] and "~ needs an integer" in c[2]

    def test_contracts_compare_like_the_vms(self, tilc):
        # #[requires: flag == 1] with flag = true is false at run time: also at compile time
        assert_like_python(tilc, "#[requires: flag == 1]\nf(flag)\n    return flag\nmain()\n    print(f(true))\n")


def write_tree(root, files):
    for rel, text in files.items():
        path = os.path.join(str(root), rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def compare_file(tilc, main_path):
    """tilc on a file: the Python compiler's bytes, or its first error (message,
    file and line)."""
    with open(main_path, encoding="utf-8") as f:
        src = f.read()
    rc, out, data = tilc.compile_file(main_path, os.path.join(tilc.dir, "in_place.tbc"))
    try:
        ref = py_compile(src, main_path)
    except (til.CompileError, SyntaxError) as e:
        m = re.match(r'(?:error(?:\[E-[A-Z]+\])?: )(.*?)\n --> (.*?):(\d+):\d+', str(e), re.S)
        assert m, str(e)
        assert out.strip() == f"tilc: {m.group(2)}:{m.group(3)}: error: {m.group(1)}"
        return 'error'
    assert rc == 0, out
    assert data == ref
    return 'same'


IMPORT_TREES = {
    "from_closure": ({
        "m.til": "let K = 10\nhelper(x) => x * K\nunused() => 99\nstruct P\n    v: int\nimpl P\n"
                 "    get(self) => self.v\nf(x) -> int\n    let p = P { v: helper(x) }\n    return p.get()\n",
        "main.til": "from m import f\nmain()\n    print(f(4))\n"}, "40\n"),
    "from_enum_variant": ({
        "m.til": "enum Color\n    Red\n    Green\nfav() => Color.Green\n",
        "main.til": "from m import Green, fav\nmain()\n    print(fav())\n"}, "1\n"),
    "clash_function": ({
        "m.til": "helper() => 1\nf() => helper()\n",
        "main.til": "import m\nhelper() => 2\nmain()\n    print(f(), helper())\n"}, None),
    "clash_from_closure": ({
        "m.til": "const SCALE = 3\nf(x) => x * SCALE\n",
        "main.til": "from m import f\nconst SCALE = 4\nmain()\n    print(f(1))\n"}, None),
    "clash_kinds": ({
        "m.til": "cap Net { send }\nstruct Thing\n    x: int\n",
        "main.til": "import m\nvar Thing = 3\nmain()\n    print(Thing)\n"}, None),
    "clash_method": ({
        "m.til": "struct P\n    x: int\nimpl P\n    get(self) => self.x\n",
        "n.til": "import m\nimpl P\n    get(self) => 0\n",
        "main.til": "import n\nmain()\n    print(P { x: 1 }.get())\n"}, None),
    "clash_with_attributes": ({
        "m.til": "#[pure]\n#[inline]\nsq(x) => x * x\n",
        "main.til": "import m\n\n#[pure]\nsq(x) => x + x\nmain()\n    print(sq(3))\n"}, None),
    "clash_two_modules": ({
        "a.til": "const SCALE = 1\n", "b.til": "var SCALE = 2\n",
        "main.til": "import a\nimport b\nmain()\n    print(SCALE)\n"}, None),
    "one_module_twice_and_cycle": ({
        "m.til": "import n\nf() => 1\n", "n.til": "import m\ng() => f() + 1\n",
        "main.til": "import m\nimport n\nmain()\n    print(f(), g())\n"}, "1\n2\n"),
    "package_entry_name": ({
        "til.json": '{"name": "app", "dependencies": {"geo": "../geo"}}',
        "til_modules/geo/geo.til": "area(w, h) => w * h\n",
        "main.til": "import geo\nmain()\n    print(geo.area(2, 3), area(1, 1))\n"}, "6\n1\n"),
    "package_manifest_main": ({
        "til.json": '{"name": "app"}',
        "til_modules/geo/til.json": '\ufeff{"name": "geo", "main": "src\\\\geo_main.til", "x": [1, {"y": null}]}',
        "til_modules/geo/src/geo_main.til": "import shapes\narea(w, h) => w * h + sq(1)\n",
        "til_modules/geo/src/shapes.til": "sq(x) => x * x\n",
        "main.til": "from geo import area\nmain()\n    print(area(2, 3))\n"}, "7\n"),
    "package_lib_and_submodule": ({
        "til.lock": "{}",
        "til_modules/geo/lib.til": "one() => 1\n",
        "til_modules/geo/shapes/square.til": "square(x) => x * x\n",
        "main.til": "import geo\nimport geo.shapes.square\nmain()\n    print(one(), square.square(3))\n"}, "1\n9\n"),
    "package_from_a_subdirectory": ({
        "til.json": '{"name": "app"}', "til_modules/util/mod.til": "u() => 5\n",
        "src/app/main.til": "import util\nmain()\n    print(util.u())\n"}, "5\n"),
    "package_nearest_project_root": ({
        "proj/til.json": '{"name": "app"}', "proj/til_modules/geo/geo.til": "g() => 3\n",
        "til_modules/geo/geo.til": "g() => 4\n", "til_modules/other/other.til": "o() => 5\n",
        "proj/main.til": "import geo\nmain()\n    print(g())\n"}, "3\n"),
    "local_module_shadows_package": ({
        "til.json": '{"name": "app"}', "til_modules/geo/geo.til": "g() => 7\n", "geo.til": "g() => 8\n",
        "main.til": "import geo\nmain()\n    print(g())\n"}, "8\n"),
    "invalid_manifests_are_empty": ({
        "til.json": '{"name": "app",}', "til_modules/geo/til.json": '{"main": "x.til" "y"}',
        "til_modules/geo/geo.til": "g() => 7\n",
        "main.til": "import geo\nmain()\n    print(g())\n"}, "7\n"),
    "package_dependencies_are_pinned": ({
        "til.json": '{"name": "app", "dependencies": {"geo": "x", "units": "y"}}',
        "til_modules/geo/til.json": '{"name": "geo", "dependencies": {"units": "*"}}',
        "til_modules/geo/geo.til": "import helper\nimport units\ngeo_area() => helper_two() * units_scale()\n",
        "til_modules/geo/helper.til": "helper_two() => 2\n",
        "til_modules/units/units.til": "units_scale() => 10\n",
        "units.til": "units_scale() => 1000\n",
        "main.til": "import geo\nmain()\n    print(geo_area())\n"}, "20\n"),
    "packages_clash": ({
        "til.json": '{"name": "app"}',
        "til_modules/a/a.til": "const SCALE = 1\na1() => SCALE\n",
        "til_modules/b/b.til": "const SCALE = 2\nb1() => SCALE\n",
        "main.til": "import a\nimport b\nmain()\n    print(a1(), b1())\n"}, None),
}


@needs_cc
class TestImports31:
    """TIL 3.1 imports (docs/TIL_PACKAGES.md): `from m import a` brings what a
    uses, a name two files define is E-IMPORT (same message and line), and
    packages come from til_modules/ (nearest first, up to the project root)."""

    @pytest.mark.parametrize("name", sorted(IMPORT_TREES))
    def test_like_python(self, tilc, tmp_path, name):
        files, out = IMPORT_TREES[name]
        write_tree(tmp_path, files)
        main = [rel for rel in files if rel == "main.til" or rel.endswith("/main.til")][0]
        path = str(tmp_path / main)
        kind = compare_file(tilc, path)
        if out is None:
            assert kind == 'error'
        else:
            assert kind == 'same'
            rc, _out, data = tilc.compile_file(path, str(tmp_path / "out.tbc"))
            assert run_vm(data) == (0, out)
