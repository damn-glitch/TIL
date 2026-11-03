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

Slow checks (the Python VM compiling all of tilc.til, ~30 s) run with
TIL_TEST_SLOW=1.

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
UNSUPPORTED = (
    "retry", "timeout", "intent", "handle", "perform", "trait", "impl Trait for Type (traits)",
    "import (tilc compiles one file)", "capability declaration (cap)",
    "(effect checking; use til check)", "(capabilities)",
)

# Bytecode may differ from the Python compiler's only through constant folding
# of float expressions (tilc folds integers, strings and booleans exactly).
FLOAT_FOLDING = re.compile(r'\d\.\d|\d[eE][-+]?\d')


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
        c = run_cvm(data, vm_args=('--max-steps', '2000000'), cwd=cwd)
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
    # different bytecode: allowed only for float constant expressions, and then
    # the programs must behave identically on both VMs
    assert FLOAT_FOLDING.search(src), f"{name}: bytecode differs from the Python compiler's"
    cwd = tempfile.mkdtemp(prefix='tilc-run-')
    ref_py, ref_c = run_both_vms(ref, cwd)
    got_py, got_c = run_both_vms(data, cwd)
    assert got_py == ref_py, f"{name}: Python VM output differs\n{got_py}\n{ref_py}"
    assert got_c == ref_c, f"{name}: C VM output differs\n{got_c}\n{ref_c}"
    return 'behaviour'


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
            assert data == py_compile(f.read(), src_path)

    @pytest.mark.skipif(not SLOW, reason="slow (~30 s): set TIL_TEST_SLOW=1")
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
        same = [n for n, r in results.items() if r == 'same']
        print("\n[selfhost] examples:", {n: r for n, r in results.items()})
        assert len(same) >= 16, results
        for must in ('examples/01_hello.til', 'examples/03_functions.til', 'examples/04_structs.til',
                     'examples/05_control_flow.til', 'examples/06_enums.til',
                     'examples/kazakh/salam.til', 'examples/kazakh/salam_latn.til',
                     'examples/russian/privet.til'):
            assert results[must] == 'same', (must, results[must])

    def test_corpus(self, tilc):
        progs = read_corpus()
        assert len(progs) > 200
        stats = {}
        for pid, origin, src in progs:
            r = check_program(tilc, f"{pid} ({origin})", src)
            key = r.split(':', 1)[0]
            stats[key] = stats.get(key, 0) + 1
        print("\n[selfhost] corpus:", stats)
        # most of the test-suite is inside TIL Core and compiles to identical bytes
        assert stats.get('same', 0) >= 200, stats


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
}


@needs_cc
class TestLanguage:
    @pytest.mark.parametrize("name", sorted(IDENTICAL))
    def test_identical_bytecode(self, tilc, name):
        assert check_program(tilc, name, IDENTICAL[name]) == 'same'

    def test_runtime_errors_report_the_same_lines(self, tilc):
        src = "inner(a)\n    let arr = [1, 2]\n    return arr[a]\n\nmain()\n    print(inner(1))\n    print(inner(5))\n"
        rc, out, data = tilc.compile(src)
        assert rc == 0 and data == py_compile(src)
        crc, cout, cerr = run_cvm(data)
        assert crc == 1 and cout == "2\n"
        assert "at inner (line 3)" in cerr and "at main (line 7)" in cerr

    def test_float_folding_keeps_behaviour(self, tilc):
        src = "main()\n    print(2.0 * 3.14159)\n    print(1.5 * 2 + 0.25)\n"
        assert check_program(tilc, "fold", src) in ('same', 'behaviour')


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
]

UNSUPPORTED_PROGRAMS = [
    ("import math_utils\n", 1, "import (tilc compiles one file)"),
    ("#[pure]\nf(x)\n    return x\n", 2, "#[pure] (effect checking; use til check)"),
    ("main()\n    let r = retry(3)\n        1\n", 2, "retry"),
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
        assert data == py_compile(CALC_PROGRAM)       # same bytes as on a PC
        # the result runs on a 64-bit VM (the int32 VM rejects 6000000000 by design)
        rc, out = run_vm(data)
        assert rc == 0 and out.splitlines()[0] == "Aigerim: 125.5"

    def test_examples_on_calculator_vm(self, tilc):
        vm = cvm_binary("host", CALC_FLAGS)
        for name in ('01_hello.til', '03_functions.til', '04_structs.til', '05_control_flow.til',
                     '07_multilevel.til', 'basic/control_flow.til'):
            path = os.path.join(EXAMPLES, name)
            rc, out, data = tilc.compile_file(path, os.path.join(tilc.dir, 'calc_out.tbc'), vm=vm)
            assert rc == 0, (name, out)
            with open(path, encoding='utf-8') as f:
                assert data == py_compile(f.read(), path), name
