#!/usr/bin/env python3
"""
TIL 3.0 tests: VM backend, C VM, verifier, multilingual syntax, bug fixes.

Most programs run on THREE engines - the native C backend, the reference
Python VM and the portable C VM - and must produce byte-identical output.

Run: python3 -m pytest tests/test_v3.py -v
Optional cross-target checks (need toolchains): TIL_TEST_CROSS=1
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import til  # noqa: E402

TIL = os.path.join(ROOT, 'src', 'til.py')
VM_DIR = os.path.join(ROOT, 'vm')
CC = shutil.which('gcc') or shutil.which('clang') or shutil.which('cc')
needs_cc = pytest.mark.skipif(CC is None, reason="no C compiler")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def run_native(src: str) -> str:
    compiler = til.TILCompiler()
    compiler.check_types = False
    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as f:
        exe = f.name
    try:
        assert compiler.compile_to_executable(src, exe, "<test>"), "native compilation failed"
        r = subprocess.run([exe], capture_output=True, timeout=30)
        return r.stdout.decode('utf-8', 'replace')
    finally:
        if os.path.exists(exe):
            os.unlink(exe)


def compile_vm(src: str, filename: str = "<test>") -> til.TBCModule:
    return til.compile_to_bytecode(src, filename)


def run_vm(src: str, max_steps: int = 0):
    """Run on the reference VM; returns (exit code, stdout)."""
    module = til.TBCModule.from_bytes(compile_vm(src).serialize())
    out = []
    rc = til.TILVM(module, write=out.append, max_steps=max_steps).run()
    return rc, b"".join(out).decode('utf-8', 'replace')


_CVM = {}


def cvm_binary(profile: str = "host") -> str:
    """Build the C VM once per test session (per profile)."""
    if profile in _CVM:
        return _CVM[profile]
    flags = {
        "host": [],
        "tiny": ['-DTILVM_LEAN', '-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0', '-DTILVM_HEAP_SIZE=2048',
                 '-DTILVM_STACK_SIZE=128', '-DTILVM_MAX_FRAMES=24', '-DTILVM_MAX_CONSTS=128',
                 '-DTILVM_MAX_TYPES=16', '-DTILVM_MAX_TRY=4'],
        "small": ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1', '-DTILVM_HEAP_SIZE=16384',
                  '-DTILVM_STACK_SIZE=512', '-DTILVM_MAX_FRAMES=64'],
        "gcstress": ['-DTILVM_HEAP_SIZE=12000'],
    }[profile]
    exe = os.path.join(tempfile.mkdtemp(prefix='tilvm-'), 'tilvm-' + profile)
    cmd = [CC, '-std=c99', '-O2', '-Wall', '-Wextra', '-pedantic', '-Werror'] + flags + \
          ['-o', exe, os.path.join(VM_DIR, 'tilvm.c'), os.path.join(VM_DIR, 'main.c'), '-lm']
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"C VM build failed ({profile}):\n{r.stderr}"
    _CVM[profile] = exe
    return exe


def run_cvm(src: str, profile: str = "host", args=()):
    data = compile_vm(src).serialize()
    with tempfile.NamedTemporaryFile(suffix='.tbc', delete=False) as f:
        f.write(data)
        path = f.name
    try:
        r = subprocess.run([cvm_binary(profile), *args, path], capture_output=True, timeout=60)
        return r.returncode, r.stdout.decode('utf-8', 'replace'), r.stderr.decode('utf-8', 'replace')
    finally:
        os.unlink(path)


def run_everywhere(src: str, native: bool = True) -> str:
    """Run on every available engine and require identical output."""
    rc, vm_out = run_vm(src)
    assert rc == 0, f"VM exit {rc}"
    if CC:
        crc, c_out, c_err = run_cvm(src)
        assert crc == 0, c_err
        assert c_out == vm_out, f"C VM differs from reference VM:\n{c_out!r}\n{vm_out!r}"
        if native:
            n_out = run_native(src)
            assert n_out == vm_out, f"native backend differs from VM:\n{n_out!r}\n{vm_out!r}"
    return vm_out.strip()


def expect_compile_error(src: str, *fragments: str):
    with pytest.raises((til.CompileError, SyntaxError)) as info:
        compile_vm(src)
    text = str(info.value)
    for frag in fragments:
        assert frag in text, f"{frag!r} not in:\n{text}"
    return text


def cli(*args, cwd=None, input_text=None):
    return subprocess.run([sys.executable, TIL, *args], capture_output=True, text=True,
                          cwd=cwd, input=input_text, timeout=120)


EXAMPLES = sorted(
    os.path.join(dp, f) for dp, _dn, fs in os.walk(os.path.join(ROOT, 'examples'))
    for f in fs if f.endswith('.til') and f != 'math_utils.til')


# ═══════════════════════════════════════════════════════════════
# BUG FIXES (all engines)
# ═══════════════════════════════════════════════════════════════

class TestFixedBugs:
    def test_fstring_with_strings_and_floats(self):
        out = run_everywhere('main()\n    let name = "World"\n    let pi = 3.14\n'
                             '    print(f"Hello {name}! pi={pi} n={2 + 3}")\n')
        assert out == "Hello World! pi=3.14 n=5"

    def test_string_concat_equality_and_append(self):
        out = run_everywhere('main()\n    let a = "he" + "llo"\n    print(a == "hello")\n'
                             '    print(a != "hello")\n    var s = "a"\n    s += "b"\n    s += "c"\n'
                             '    print(s)\n    print("n=" + 42)\n')
        assert out == "true\nfalse\nabc\nn=42"

    def test_loops_over_string_and_float_arrays(self):
        out = run_everywhere('main()\n    let names = ["ann", "bob"]\n    for n in names\n'
                             '        print(n)\n    for x in [1.5, 2.25]\n        print(x)\n'
                             '    for c in "AB"\n        print(c)\n')
        assert out == "ann\nbob\n1.5\n2.25\n65\n66"

    def test_integer_power_and_precedence(self):
        out = run_everywhere('main()\n    print(2 ** 10)\n    print(3 ** 40)\n    print(-2 ** 2)\n'
                             '    print(2.0 ** 0.5)\n')
        assert out.splitlines() == ["1024", "12157665459056928801"[:0] or str(til.wrap_int64(3 ** 40)),
                                    "-4", "1.41421"]

    def test_chained_comparisons(self):
        out = run_everywhere('main()\n    let i = 5\n    print(0 <= i < 10)\n    print(3 < 2 < 1)\n'
                             '    print(1 < 2 < 3 < 4)\n')
        assert out == "true\nfalse\ntrue"

    def test_one_line_function_bodies(self):
        # Block([...]) used to put the statements into the dataclass `line` field:
        # `f(x) -> int => x + 1` compiled to an empty function returning 0.
        out = run_everywhere('add(a: int, b: int) -> int => a + b\nmain()\n    print(add(2, 3))\n')
        assert out == "5"

    def test_trailing_blank_lines(self):
        assert run_everywhere('main()\n    print(1)\n\n\n\n') == "1"

    def test_else_if(self):
        src = ('main()\n    for x in [1, 5, 50]\n        if x > 10\n            print("big")\n'
               '        else if x > 3\n            print("mid")\n        else\n            print("small")\n')
        assert run_everywhere(src) == "small\nmid\nbig"

    def test_nested_struct_float_field(self):
        src = ('struct V\n    x: float\n    y: float\nstruct P\n    pos: V\n'
               'main()\n    let p = P { pos: V { x: 0.5, y: 1.5 } }\n    print(p.pos.x)\n    print(p.pos.y)\n')
        assert run_everywhere(src) == "0.5\n1.5"

    def test_assert_message_cannot_inject_c(self):
        # used to close the C string literal and inject code
        src = 'main()\n    assert 1 > 2, "bad \\" ); exit(42); //"\n'
        compiler = til.TILCompiler()
        with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as f:
            exe = f.name
        try:
            if CC:
                assert compiler.compile_to_executable(src, exe, "<t>")
                r = subprocess.run([exe], capture_output=True, text=True)
                assert r.returncode == 1 and 'exit(42)' in r.stderr
        finally:
            os.unlink(exe)
        rc, _ = run_vm(src)
        assert rc == 1

    def test_intent_text_cannot_inject_c(self):
        c = til.TILCompiler().compile('main()\n    intent "x\\nexit(123);"\n        print(1)\n')
        assert "exit(123);" not in c.replace("/* INTENT: x exit(123); */", "")

    def test_numeric_literals(self):
        assert run_everywhere('main()\n    print(0x1F)\n    print(0b101)\n    print(0o17)\n'
                              '    print(1_000_000)\n    print(2.5e3)\n') == "31\n5\n15\n1000000\n2500"
        with pytest.raises(SyntaxError):
            compile_vm('main()\n    print(0x)\n')

    def test_option_result_native(self):
        src = ('find(id: int) -> Option<str>\n    if id == 1\n        return Some("found")\n    return None\n'
               'div(a: int, b: int) -> Result<int, str>\n    if b == 0\n        return Err("div by zero")\n'
               '    return Ok(a / b)\n'
               'calc(a: int, b: int) -> Result<int, str>\n    let q = div(a, b)?\n    return Ok(q * 10)\n'
               'main()\n    print(find(1).unwrap())\n    print(find(2).is_none())\n'
               '    print(find(2).unwrap_or("none"))\n    print(calc(8, 2).unwrap())\n'
               '    print(calc(1, 0).is_err())\n')
        assert run_everywhere(src) == "found\ntrue\nnone\n40\ntrue"

    def test_match_on_strings(self):
        src = ('hi(lang: str) -> str\n    return match lang\n        "kk" => "Сәлем"\n'
               '        "ru" => "Привет"\n        _ => "Hello"\nmain()\n    print(hi("kk"))\n'
               '    print(hi("ru"))\n    print(hi("xx"))\n')
        assert run_everywhere(src) == "Сәлем\nПривет\nHello"


class TestScriptMode:
    def test_top_level_statements_are_the_program(self):
        assert run_everywhere('print("hi")\nprint(1 + 2)\n') == "hi\n3"

    def test_globals_initialize_in_source_order(self):
        src = ('var primes = []\nfor n in 2..20\n    var ok = true\n    for d in primes\n'
               '        if n % d == 0\n            ok = false\n    if ok\n        primes.push(n)\n'
               'let summary = {"count": primes.len(), "last": primes.last()}\nprint(summary)\n')
        rc, out = run_vm(src)
        assert out.strip() == '{"count": 8, "last": 19}'

    def test_functions_see_script_globals(self):
        src = 'square(x: int) -> int\n    return x * x\nlet total = square(12)\nprint(total)\n'
        assert run_everywhere(src) == "144"


class TestModules:
    def test_imports(self, tmp_path):
        (tmp_path / 'geo.til').write_text('const UNIT = 10\narea(w: int, h: int) -> int\n    return w * h\n')
        (tmp_path / 'shapes.til').write_text('import geo\nsquare(a: int) -> int\n    return area(a, a)\n')
        main = tmp_path / 'main.til'
        main.write_text('import shapes\nimport geo as g\nmain()\n    print(square(3))\n'
                        '    print(g.area(2, UNIT))\n    print(geo.area(1, 1))\n')
        for flags in ([], ['--vm']):
            r = cli('run', *flags, str(main), cwd=str(tmp_path))
            assert r.stdout.split() == ["9", "20", "1"], r.stderr


# ═══════════════════════════════════════════════════════════════
# MULTILINGUAL SYNTAX
# ═══════════════════════════════════════════════════════════════

class TestMultilingual:
    def test_kazakh_latin_readme_example(self):
        src = ('#[lang: kk-latn]\nnegizgi()\n    turaqty aty = "Alem"\n    turaqty zhasy = 25\n'
               '    eger zhasy > 18\n        basyp_shygaru("Salam, " + aty + "!")\n    aitpese\n'
               '        basyp_shygaru("Salemetsiz be!")\n    ushin i ishinde 1..3\n        basyp_shygaru(i)\n')
        assert run_everywhere(src) == "Salam, Alem!\n1\n2"

    def test_kazakh_else_if(self):
        # 'немесе' used to be both elif and or (it silently became or)
        src = ('негізгі()\n    тұрақты x = 5\n    егер x > 10\n        басып_шығару("үлкен")\n'
               '    әйтпесе егер x > 3 немесе x < 0\n        басып_шығару("орташа")\n'
               '    әйтпесе\n        басып_шығару("кіші")\n')
        assert run_everywhere(src) == "орташа"

    def test_russian(self):
        src = ('#[lang: ru]\nфункция факт(н: int) -> int\n    если н <= 1\n        вернуть 1\n'
               '    вернуть н * факт(н - 1)\nглавная()\n    пусть х = 5\n    если х > 3 и истина\n'
               '        печать("да")\n    иначе\n        печать("нет")\n    печать(факт(6))\n')
        assert run_everywhere(src) == "да\n720"

    def test_russian_is_opt_in(self):
        # without the pragma, 'и' is an ordinary identifier (used in Kazakh code)
        assert run_everywhere('main()\n    var и = 0\n    for к in 0..4\n        и += к\n    print(и)\n') == "6"

    @pytest.mark.parametrize("path", [p for p in EXAMPLES if '/kazakh/' not in p
                                      and 'import' not in open(p, encoding='utf-8').read()][:8])
    def test_translation_round_trip(self, path):
        src = open(path, encoding='utf-8').read()
        ru = til.translate_source(src, 'ru')
        kk = til.translate_source(ru, 'kk-latn')
        en = til.translate_source(kk, 'en')
        assert en == til.translate_source(src, 'en')
        assert run_vm(ru)[1] == run_vm(src)[1]


# ═══════════════════════════════════════════════════════════════
# VERIFIER: effects, capabilities, contracts, arity
# ═══════════════════════════════════════════════════════════════

class TestVerifier:
    def test_pure_function_cannot_print(self):
        expect_compile_error('#[pure]\nf(x: int) -> int\n    print(x)\n    return x\nmain()\n    print(f(1))\n',
                             "E-EFFECT", "pure function 'f'")

    def test_effects_propagate_through_calls(self):
        expect_compile_error('log(m: str)\n    print(m)\n#[pure]\nsq(x: int) -> int\n    log("sq")\n'
                             '    return x * x\nmain()\n    print(sq(2))\n',
                             "through the call to 'log'")

    def test_declared_effects_are_an_upper_bound(self):
        expect_compile_error('#[effects: fs]\nf()\n    print("x")\nmain()\n    f()\n', "E-EFFECT", "'io'")
        assert run_everywhere('#[effects: io]\nf()\n    print("x")\nmain()\n    f()\n') == "x"

    def test_capabilities(self):
        base = 'cap FileRead { read }\n#[cap: FileRead]\nread_cfg() -> str\n    return "cfg"\n'
        expect_compile_error(base + 'helper() -> str\n    return read_cfg()\nmain()\n    print(helper())\n',
                             "E-CAP", "requires capability 'FileRead'")
        assert run_everywhere(base + '#[cap: FileRead]\nhelper() -> str\n    return read_cfg()\n'
                                     'main()\n    print(helper())\n') == "cfg"

    def test_capability_cannot_be_laundered_through_a_value(self):
        expect_compile_error('#[cap: Net]\nfetch() -> int\n    return 1\nsneaky() -> int\n'
                             '    let f = fetch\n    return f()\nmain()\n    print(sneaky())\n', "E-CAP")

    def test_contracts_checked_at_compile_time(self):
        src = '#[requires: b != 0]\ndiv(a: int, b: int) -> int\n    return a / b\nmain()\n'
        expect_compile_error(src + '    print(div(10, 0))\n', "E-CONTRACT", "b != 0")
        assert run_everywhere(src + '    print(div(10, 2))\n') == "5"

    def test_contracts_checked_at_run_time_and_recoverable(self):
        src = ('#[requires: n >= 0]\n#[ensures: result >= 1]\nfact(n: int) -> int\n    if n <= 1\n'
               '        return 1\n    return n * fact(n - 1)\nmain()\n    var k = 0 - 3\n'
               '    let r = retry(2)\n        fact(k)\n    else\n        error\n    print(r)\n')
        out = run_everywhere(src, native=False)
        assert out == "contract violation: fact requires n >= 0"

    def test_arity(self):
        expect_compile_error('f(a: int, b: int = 2) -> int\n    return a + b\nmain()\n    print(f(1, 2, 3))\n',
                             "E-ARITY")
        assert run_everywhere('f(a: int, b: int = 2) -> int\n    return a + b\nmain()\n    print(f(1))\n'
                              '    print(f(1, 5))\n') == "3\n6"

    def test_check_command(self, tmp_path):
        bad = tmp_path / 'bad.til'
        bad.write_text('#[pure]\nf()\n    print(1)\nmain()\n    f()\n')
        good = tmp_path / 'good.til'
        good.write_text('main()\n    print(1)\n')
        assert cli('check', str(bad)).returncode == 1
        assert cli('check', str(good)).returncode == 0


# ═══════════════════════════════════════════════════════════════
# VM LANGUAGE FEATURES (Python VM == C VM)
# ═══════════════════════════════════════════════════════════════

def vm_both(src: str) -> str:
    return run_everywhere(src, native=False)


class TestVMFeatures:
    def test_closures_capture_by_reference(self):
        src = ('make_counter()\n    var n = 0\n    return |step: int| n + step\n'
               'adder(k: int)\n    return |x: int| x + k\nmain()\n    let add5 = adder(5)\n'
               '    print(add5(10))\n    var base = 1\n    let f = |x: int| x + base\n    base = 100\n'
               '    print(f(1))\n')
        assert vm_both(src) == "15\n101"

    def test_higher_order_methods(self):
        src = ('main()\n    let xs = [1, 2, 3, 4, 5, 6]\n    print(xs.map(|x| x * x))\n'
               '    print(xs.filter(|x| x % 2 == 0))\n    print(xs.reduce(|a, b| a + b, 0))\n'
               '    print(xs.any(|x| x > 5))\n    print(xs.all(|x| x > 0))\n    xs.each(|x| print(x * 0))\n'
               '    print(xs |> len)\n')
        assert vm_both(src) == "[1, 4, 9, 16, 25, 36]\n[2, 4, 6]\n21\ntrue\ntrue\n" + "0\n" * 6 + "6"

    def test_traits_and_dynamic_dispatch(self):
        src = ('trait Shape\n    area(self) -> float\n    describe(self) -> str\n'
               '        return "shape with area " + str(self.area())\n'
               'struct Sq\n    s: float\nstruct Ci\n    r: float\n'
               'impl Shape for Sq\n    area(self) -> float\n        return self.s * self.s\n'
               'impl Shape for Ci\n    area(self) -> float\n        return 3.0 * self.r * self.r\n'
               '    describe(self) -> str\n        return "circle"\n'
               'main()\n    for sh in [Sq { s: 2.0 }, Ci { r: 1.0 }]\n        print(sh.describe())\n')
        assert vm_both(src) == "shape with area 4\ncircle"

    def test_algebraic_data_types(self):
        src = ('enum Expr\n    Num(int)\n    Add(Expr, Expr)\n    Mul(Expr, Expr)\n'
               'eval(e: Expr) -> int\n    return match e\n        Expr.Num(n) => n\n'
               '        Expr.Add(a, b) => eval(a) + eval(b)\n        Expr.Mul(a, b) => eval(a) * eval(b)\n'
               'main()\n    let e = Expr.Add(Expr.Num(2), Expr.Mul(Expr.Num(3), Expr.Num(4)))\n'
               '    print(eval(e))\n    print(Expr.Num(7))\n')
        assert vm_both(src) == "14\nExpr.Num(7)"

    def test_option_result_match_and_propagation(self):
        src = ('parse(s: str) -> Result<int, str>\n    if s == ""\n        return Err("empty")\n'
               '    return Ok(s.to_int())\n'
               'twice(s: str) -> Result<int, str>\n    let v = parse(s)?\n    return Ok(v * 2)\n'
               'main()\n    for s in ["21", ""]\n        match twice(s)\n'
               '            Ok(v) => print(f"ok {v}")\n            Err(e) => print(f"err {e}")\n'
               '    let o = Some(3)\n    match o\n        Some(x) if x > 2 => print("big")\n'
               '        Some(x) => print("small")\n        None => print("none")\n')
        assert vm_both(src) == "ok 42\nerr empty\nbig"

    def test_retry_really_retries(self):
        src = ('var attempts = 0\nflaky() -> int\n    attempts += 1\n    assert attempts >= 3, "not yet"\n'
               '    return attempts\nmain()\n    let r = retry(5)\n        flaky()\n    print(r)\n'
               '    let g = retry(2)\n        panic("broken")\n    else\n        "fallback: " + error\n'
               '    print(g)\n')
        assert vm_both(src) == "3\nfallback: broken"
        assert run_native(src).strip() == "3\nfallback: broken"

    def test_retry_without_fallback_reraises(self):
        rc, _ = run_vm('main()\n    let r = retry(2)\n        panic("nope")\n    print(r)\n')
        assert rc == 1

    def test_timeout_interrupts_infinite_loop(self):
        src = ('spin() -> int\n    var i = 0\n    loop\n        i += 1\n    return i\n'
               'main()\n    let t = timeout(30)\n        spin()\n    else\n        "gave up: " + error\n'
               '    print(t)\n')
        assert vm_both(src) == "gave up: timeout"
        assert run_native(src).strip() == "gave up: timeout"

    def test_intent_adds_context_to_errors(self):
        rc, _ = run_vm('main()\n    intent "load the settings"\n        let xs = [1]\n        print(xs[5])\n')
        assert rc == 1
        src = ('main()\n    let r = retry(1)\n        intent "parse config"\n            panic("bad token")\n'
               '    else\n        error\n    print(r)\n')
        assert vm_both(src) == "intent 'parse config' failed: bad token"

    def test_runtime_error_has_line_and_trace(self):
        src = 'f(xs: [int]) -> int\n    return xs[10]\nmain()\n    print(f([1, 2]))\n'
        module = til.TBCModule.from_bytes(compile_vm(src).serialize())
        tb = []
        vm = til.TILVM(module, write=lambda b: None)
        orig = vm.report
        vm.report = lambda msg, pc: tb.append(msg + b"\n" + vm.traceback(pc))
        assert vm.run() == 1
        text = tb[0].decode()
        assert "index out of bounds: index 10, length 2" in text
        assert "at f (line 2)" in text and "at main (line 4)" in text

    def test_maps(self):
        src = ('main()\n    var m = {"b": 2, "a": 1}\n    m["c"] = 3\n    m["a"] += 10\n'
               '    m.delete("b")\n    print(m)\n    print(m.keys())\n    print(m.get_or("zz", 0))\n'
               '    print(m.has("c"))\n    for k in m\n        print(k)\n')
        assert vm_both(src) == '{"a": 11, "c": 3}\n["a", "c"]\n0\ntrue\na\nc'

    def test_string_methods(self):
        src = ('main()\n    let s = "  Hello, World  "\n    print(s.trim().to_upper())\n'
               '    print("a,b,,c".split(","))\n    print(["x", "y"].join("-"))\n'
               '    print("banana".count("an"))\n    print("banana".replace("an", "AN"))\n'
               '    print("abc".slice(1, 10))\n    print("ab" * 3)\n    print("x y  z".split())\n')
        assert vm_both(src) == ('HELLO, WORLD\n["a", "b", "", "c"]\nx-y\n2\nbANANa\nbc\nababab\n'
                                '["x", "y", "z"]')

    def test_format_specs(self):
        src = 'main()\n    print(f"{3.14159:.2f}|{255:x}|{7:03}|{0.25:%}|{42:<5}|{42:>5}|{1234.5:.1e}")\n'
        assert run_everywhere(src) == "3.14|ff|007|25.000000%|42   |   42|1.2e+03"

    def test_structural_equality(self):
        src = ('struct P\n    x: int\n    y: int\nmain()\n    print([1, [2, 3]] == [1, [2, 3]])\n'
               '    print(P { x: 1, y: 2 } == P { x: 1, y: 2 })\n    print({"a": 1} == {"a": 2})\n')
        assert vm_both(src) == "true\ntrue\nfalse"

    def test_immutability_is_enforced(self):
        expect_compile_error('main()\n    let x = 1\n    x = 2\n', "cannot assign twice to immutable", "var")

    def test_unknown_name_suggestion(self):
        expect_compile_error('main()\n    let total = 1\n    print(totl)\n', "unknown name 'totl'", "total")

    def test_step_limit_sandbox(self):
        rc, _ = run_vm('main()\n    loop\n        print(1)\n', max_steps=50)
        assert rc == 1

    def test_serialization_round_trip(self):
        m = compile_vm(open(EXAMPLES[0], encoding='utf-8').read())
        data = m.serialize()
        assert til.TBCModule.from_bytes(data).serialize() == data
        assert data[:4] == b"TILB"

    def test_disassembler(self):
        listing = til.disassemble(compile_vm('main()\n    print(1 + 2)\n'))
        assert "fn #" in listing and "main" in listing and "NATIVE" in listing


# ═══════════════════════════════════════════════════════════════
# ALL EXAMPLES: native == Python VM == C VM
# ═══════════════════════════════════════════════════════════════

class TestExamplesEverywhere:
    @pytest.mark.parametrize("path", EXAMPLES, ids=[os.path.relpath(p, ROOT) for p in EXAMPLES])
    def test_example(self, path):
        src = open(path, encoding='utf-8').read()
        rc, vm_out = run_vm_file(path)
        assert rc == 0
        if CC:
            r = cli('run', path, cwd=os.path.dirname(path))
            assert r.returncode == 0, r.stderr
            assert r.stdout == vm_out, "native and VM output differ"
            m = til.compile_to_bytecode(src, path)
            with tempfile.NamedTemporaryFile(suffix='.tbc', delete=False) as f:
                f.write(m.serialize())
            try:
                c = subprocess.run([cvm_binary(), f.name], capture_output=True, timeout=60)
                assert c.stdout.decode('utf-8', 'replace') == vm_out, "C VM differs"
            finally:
                os.unlink(f.name)


def run_vm_file(path):
    src = open(path, encoding='utf-8').read()
    module = til.TBCModule.from_bytes(til.compile_to_bytecode(src, path).serialize())
    out = []
    rc = til.TILVM(module, write=out.append).run()
    return rc, b"".join(out).decode('utf-8', 'replace')


# ═══════════════════════════════════════════════════════════════
# C VM SPECIFICS
# ═══════════════════════════════════════════════════════════════

@needs_cc
class TestCVM:
    def test_tables_match_compiler(self):
        """Opcodes, natives and builtin methods: src/til.py == vm/tilvm.c."""
        c = open(os.path.join(VM_DIR, 'tilvm.c'), encoding='utf-8').read()
        ops = dict((name, int(val, 16)) for name, val in re.findall(r'OP_(\w+) = (0x[0-9A-Fa-f]+)', c))
        for name, val in vars(til.Op).items():
            if not name.startswith('_'):
                assert ops.get(name) == val, f"opcode {name}"
        natives = re.search(r'NATIVE_NAMES\[\] = \{(.*?)\};', c, re.S).group(1)
        assert re.findall(r'"(\w+)"', natives) == til.NATIVES
        methods = re.search(r'METHOD_NAMES\[\] = \{(.*?)\};', c, re.S).group(1)
        assert re.findall(r'"(\w+)"', methods) == til.METHODS

    def test_gc_stress_small_heap(self):
        src = ('main()\n    var keep = []\n    var total = 0\n    for i in 0..2000\n'
               '        let s = f"item-{i}-" + str(i * i)\n        if i % 100 == 0\n'
               '            keep.push(s.split("-")[1])\n        total += len(s)\n'
               '    print(total)\n    print(keep.join(","))\n')
        rc, ref = run_vm(src)
        crc, out, err = run_cvm(src, profile="gcstress", args=("--stats",))
        assert crc == 0 and out == ref
        gcs = int(re.search(r'gc=(\d+)', err).group(1))
        assert gcs > 5, "the collector should have run many times"

    def test_snapshot_and_resume(self, tmp_path):
        src = ('var primes = []\nvar n = 2\nwhile primes.len() < 30\n    var ok = true\n'
               '    for d in primes\n        if n % d == 0\n            ok = false\n            break\n'
               '    if ok\n        primes.push(n)\n        if primes.len() % 10 == 0\n'
               '            print(f"{primes.len()} primes, last {n}")\n    n += 1\n')
        tbc = tmp_path / 'p.tbc'
        tbc.write_bytes(compile_vm(src).serialize())
        vm = cvm_binary()
        full = subprocess.run([vm, str(tbc)], capture_output=True).stdout
        snap = tmp_path / 'p.snap'
        part1 = subprocess.run([vm, '--snapshot-after', '200', str(snap), str(tbc)], capture_output=True)
        assert part1.returncode == 3 and snap.exists()
        part2 = subprocess.run([vm, '--resume', str(snap), str(tbc)], capture_output=True)
        assert part2.returncode == 0
        assert part1.stdout + part2.stdout == full

    def test_tiny_profile(self):
        src = ('fib(n: int) -> int\n    if n < 2\n        return n\n    return fib(n - 1) + fib(n - 2)\n'
               'main()\n    print(fib(15))\n    print([1, 2, 3].map(|x| x * 10))\n')
        rc, out, _ = run_cvm(src, profile="tiny")
        assert rc == 0 and out.strip() == "610\n[10, 20, 30]"
        rc, _, err = run_cvm('main()\n    print(1.5)\n', profile="tiny")
        assert rc == 2 and "floats" in err

    def test_small_profile_32bit(self):
        rc, out, _ = run_cvm('main()\n    print(100000 * 1000)\n    print(1.5 * 2)\n', profile="small")
        assert rc == 0 and out.strip() == "100000000\n3"

    def test_freestanding_builds(self, tmp_path):
        for defs in (['-DTILVM_NO_STDIO', '-DTILVM_NO_TIME'],
                     ['-DTILVM_NO_STDIO', '-DTILVM_NO_MATH', '-DTILVM_FLOAT=1'],
                     ['-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0', '-DTILVM_NO_STDIO']):
            r = subprocess.run([CC, '-std=c99', '-Wall', '-Wextra', '-pedantic', '-Werror', *defs, '-c',
                                os.path.join(VM_DIR, 'tilvm.c'), '-o', str(tmp_path / 'x.o')],
                               capture_output=True, text=True)
            assert r.returncode == 0, f"{defs}: {r.stderr}"

    def test_amalgamated_c_vm_target(self, tmp_path):
        prog = tmp_path / 'hello.til'
        prog.write_text('main()\n    print("hello from one C file")\n    print([1, 2].map(|x| x + 1))\n')
        r = cli('build', '--target', 'c-vm', str(prog), '-o', str(tmp_path / 'hello_vm.c'))
        assert r.returncode == 0, r.stderr
        exe = tmp_path / 'hello_vm'
        c = subprocess.run([CC, '-std=c99', '-O1', '-o', str(exe), str(tmp_path / 'hello_vm.c'), '-lm'],
                           capture_output=True, text=True)
        assert c.returncode == 0, c.stderr
        assert subprocess.run([str(exe)], capture_output=True, text=True).stdout == \
            "hello from one C file\n[2, 3]\n"

    def test_cli_run_cvm(self, tmp_path):
        prog = tmp_path / 'p.til'
        prog.write_text('print(sum([1, 2, 3]))\n')
        r = cli('run', '--cvm', str(prog))
        assert r.returncode == 0 and r.stdout.strip() == "6", r.stderr


# ═══════════════════════════════════════════════════════════════
# CROSS TARGETS (calculator / microcontroller CPUs, in simulators)
# ═══════════════════════════════════════════════════════════════

cross = pytest.mark.skipif(not os.environ.get('TIL_TEST_CROSS'), reason="set TIL_TEST_CROSS=1")


@cross
class TestCrossTargets:
    PROGRAM = os.path.join(ROOT, 'examples', '03_functions.til')

    def reference(self):
        return run_vm_file(self.PROGRAM)[1]

    @pytest.mark.skipif(not shutil.which('qemu-system-arm'), reason="no qemu-system-arm")
    def test_cortex_m(self):
        port = os.path.join(VM_DIR, 'ports', 'cortex-m')
        subprocess.run(['rm', '-f', os.path.join(port, 'build', 'program.h')])
        subprocess.run(['make', '-s', '-C', port, f'PROGRAM={self.PROGRAM}'], check=True, capture_output=True)
        r = subprocess.run(['qemu-system-arm', '-M', 'lm3s6965evb', '-nographic', '-semihosting-config',
                            'enable=on,target=native', '-kernel', os.path.join(port, 'build', 'tilvm.elf')],
                           capture_output=True, text=True, timeout=120)
        out = "".join(l + "\n" for l in (r.stdout + r.stderr).splitlines() if "Timer with period" not in l)
        assert out == self.reference()

    @pytest.mark.skipif(not shutil.which('simavr'), reason="no simavr")
    def test_avr(self):
        r = subprocess.run([sys.executable, os.path.join(VM_DIR, 'ports', 'avr', 'run_sim.py'), self.PROGRAM],
                           capture_output=True, text=True, timeout=600)
        assert r.stdout == self.reference()

    @pytest.mark.skipif(not shutil.which('sz80'), reason="no SDCC ucsim")
    def test_z80(self):
        r = subprocess.run([sys.executable, os.path.join(VM_DIR, 'ports', 'z80', 'run_sim.py'), self.PROGRAM],
                           capture_output=True, text=True, timeout=7200)
        assert r.stdout == self.reference()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
