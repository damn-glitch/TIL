#!/usr/bin/env python3
"""
TIL 3.1 typed native backend (docs/TIL_TYPES.md): gradual type inference,
E-TYPE for definite type errors, monomorphised generics, and a native
backend whose output is byte-identical to the VMs.

Run: python3 -m pytest tests/test_types.py -v
The full native-vs-VM coverage measurement over the whole corpus (about a
minute of C compilations) runs with TIL_TEST_SLOW=1.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

from tilhelpers import CC, ROOT, cli, needs_cc, run_both, run_vm, til

sys.path.insert(0, os.path.join(ROOT, 'tools'))
import native_coverage  # noqa: E402

tiltypes = til.feature_module('tiltypes')
slow = pytest.mark.skipif(not os.environ.get('TIL_TEST_SLOW'), reason="set TIL_TEST_SLOW=1")
BASELINE = os.path.join(ROOT, 'tools', 'native_coverage_baseline.json')


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def program(src: str):
    return til.resolve_imports(til.parse_source(src, "<test>"), "<test>")


def infer(src: str):
    return tiltypes.infer_program(program(src), src, "<test>")


def main_locals(src: str) -> dict:
    return infer(src).specialize('main', []).locals()


def c_code(src: str) -> str:
    return til.TILCompiler().compile(src, "<test>")


def run_native(src: str, args=()):
    """Compile with the native backend (must succeed) and run: (exit code, stdout)."""
    compiler = til.TILCompiler()
    fd, exe = tempfile.mkstemp(suffix='.exe')
    os.close(fd)
    try:
        ok = compiler.compile_to_executable(src, exe, "<test>", quiet=True)
        assert ok, f"the native backend did not compile it: {compiler.last_c_error}"
        r = subprocess.run([exe, *args], stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        return r.returncode, r.stdout.decode('utf-8', 'replace')
    finally:
        os.unlink(exe)


def native_matches_vm(src: str, expect_rc: int = 0) -> str:
    """Native output and exit code == Python VM == C VM; returns the output."""
    vm_out = run_both(src, expect_rc=expect_rc)
    rc, out = run_native(src)
    assert rc == expect_rc, f"native exit {rc}"
    assert out == vm_out, f"native differs from the VM:\n{out!r}\n{vm_out!r}"
    return out


def etype_errors(src: str):
    return infer(src).errors


# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

class TestInference:
    def test_literals_and_operators(self):
        types = main_locals('main()\n    let a = 1\n    let b = 2.5\n    let c = a + b\n    let d = 7 / 2\n'
                            '    let e = "n=" + a\n    let f = a < b\n    let g = a ** 2\n    let h = -b\n'
                            '    let i = not a\n    let j = 5 % 3\n    let k = 5 & 3\n    let ch = \'x\'\n')
        assert types == {'a': 'int', 'b': 'float', 'c': 'float', 'd': 'int', 'e': 'str', 'f': 'bool',
                         'g': 'int', 'h': 'float', 'i': 'bool', 'j': 'int', 'k': 'int', 'ch': 'int'}

    def test_containers_learn_their_element_types_from_use(self):
        types = main_locals('main()\n    var xs = []\n    xs.push("a")\n    var m = {}\n    m[1] = 2.5\n'
                            '    let sq = [x * x for x in 0..3]\n    let lens = xs.map(|s| len(s))\n'
                            '    let words = "a b".split(" ")\n    let total = sq.reduce(|a, b| a + b, 0)\n'
                            '    let keys = m.keys()\n    let first = xs[0]\n')
        assert types == {'xs': '[str]', 'm': '{int: float}', 'sq': '[int]', 'lens': '[int]',
                         'words': '[str]', 'total': 'int', 'keys': '[int]', 'first': 'str'}

    def test_structs_methods_and_option_result(self):
        src = ('struct P\n    x: float\n    name: str\nimpl P\n    norm(self) -> float\n        return self.x * 2\n'
               'find(id: int) -> Option<P>\n    if id == 1\n        return Some(P { x: 1.0, name: "a" })\n'
               '    return None\n'
               'parse(s: str) -> Result<int, str>\n    if s == ""\n        return Err("empty")\n    return Ok(s.to_int())\n'
               'main()\n    let p = P { x: 1.5, name: "p" }\n    let n = p.norm()\n    let nm = p.name\n'
               '    let o = find(1)\n    let q = o.unwrap()\n    let r = parse("4")\n    let v = r.unwrap_or(0)\n')
        assert main_locals(src) == {'p': 'P', 'n': 'float', 'nm': 'str', 'o': 'Option<P>', 'q': 'P',
                                    'r': 'Result<int, str>', 'v': 'int'}

    def test_natives(self):
        types = main_locals('main()\n    let a = len("x")\n    let b = str(1)\n    let c = abs(-2)\n'
                            '    let d = abs(-2.5)\n    let e = sqrt(2)\n    let f = min(1, 2)\n    let g = sum([1.5])\n'
                            '    let h = range(3)\n    let i = args()\n    let j = str_bytes("a")\n'
                            '    let k = type_of(1)\n    let l = int("3")\n    let m2 = max([2, 3])\n')
        assert types == {'a': 'int', 'b': 'str', 'c': 'int', 'd': 'float', 'e': 'float', 'f': 'int',
                         'g': 'float', 'h': '[int]', 'i': '[str]', 'j': '[int]', 'k': 'str', 'l': 'int',
                         'm2': 'int'}

    def test_unannotated_parameters_make_a_function_generic(self):
        ti = infer('twice(x)\n    return x + x\nlabel(x: int) -> str\n    return str(x)\n'
                   'main()\n    print(twice(21))\n    print(twice("ab"))\n    print(label(1))\n')
        assert ti.is_generic('twice') and not ti.is_generic('label')
        assert ti.specialize('twice', [tiltypes.INT]).signature() == "fn(int) -> int"
        assert ti.specialize('twice', [tiltypes.STR]).signature() == "fn(str) -> str"
        assert ti.specialize('twice', [tiltypes.FLOAT]).signature() == "fn(float) -> float"
        calls = [c for c in ti.specialize('main', []).ctx.calls.values() if c[0] == 'twice']
        assert sorted(tiltypes.show(c[1][0]) for c in calls) == ['int', 'str']

    def test_explicit_type_parameters(self):
        ti = infer('max_of<T>(a: T, b: T) -> T\n    if a > b\n        return a\n    return b\n'
                   '#[generic: K]\nfirst(xs: [K]) -> K\n    return xs[0]\n'
                   'main()\n    print(max_of(1, 2))\n    print(first(["a"]))\n')
        assert ti.is_generic('max_of') and ti.is_generic('first')
        assert ti.specialize('max_of', [tiltypes.STR, tiltypes.STR]).signature() == "fn(str, str) -> str"
        assert ti.specialize('first', [tiltypes.arr(tiltypes.INT)]).signature() == "fn([int]) -> int"

    def test_conflicts_are_dynamic_not_errors(self):
        src = ('main()\n    var x = 1\n    x = "now a string"\n    let mixed = [1, "a"]\n    var y = 0\n'
               '    y = y + 0.5\n    let pick = 1 if true else "one"\n')
        ti = infer(src)
        assert ti.errors == []
        assert ti.specialize('main', []).locals() == {'x': 'dynamic', 'mixed': '[dynamic]', 'y': 'dynamic',
                                                      'pick': 'dynamic'}

    def test_values_flow_one_way(self):
        # using i in a heterogeneous array does not make i itself dynamic
        types = main_locals('main()\n    let i = 1\n    let mixed = [i, "s"]\n    var j = i\n    j = "x"\n'
                            '    let k = i + 1\n')
        assert types == {'i': 'int', 'mixed': '[dynamic]', 'j': 'dynamic', 'k': 'int'}

    def test_infinite_types_are_dynamic(self, tmp_path):
        # polymorphic recursion (nest([v])) and an array that contains itself:
        # the type would be infinite, so it is dynamic (no crash, no E-TYPE)
        src = ('nest(v, n: int)\n    if n == 0\n        return str(v)\n    return nest([v], n - 1)\n'
               'grow(xs)\n    xs.push(xs)\n    return len(xs)\n'
               'main()\n    print(nest(1, 3))\n    var a = []\n    print(grow(a))\n')
        assert etype_errors(src) == []
        with pytest.raises(til.NativeUnsupported) as info:
            c_code(src)
        assert "dynamically typed" in str(info.value)
        prog = tmp_path / 'nest.til'
        prog.write_text(src)
        r = cli('check', str(prog))
        assert r.returncode == 0 and "internal error" not in r.stderr, r.stderr
        r = cli('run', '--vm', str(prog))
        assert r.stdout == "[[[1]]]\n1\n" and "internal error" not in r.stderr, r.stderr

    def test_private_constraints_do_not_grow_schemes(self):
        # f0's open constraint (g applied to an element of unknown type) only
        # concerns f0's own variables: callers do not copy it (it used to double
        # at every level: 12288 constraints in f12's scheme)
        lines = ['g(x)', '    return x + 1', 'f0()', '    var v = []', '    return len(str(g(v[0])))']
        for i in range(1, 13):
            lines += [f'f{i}()', f'    return f{i - 1}() + f{i - 1}()']
        src = '\n'.join(lines + ['main()', '    print(f12() * 0)']) + '\n'
        ti = infer(src)
        assert ti.errors == []
        assert len(ti.funcs['f12'].scheme.templates) == 0
        assert ti.funcs['g'].scheme.templates                    # x + 1 decides g's result

    def test_retry_with_a_string_fallback(self):
        types = main_locals('main()\n    let r = retry(2)\n        6 * 7\n    else\n        "failed: " + error\n'
                            '    let s = retry(1)\n        panic("x")\n    else\n        error\n')
        assert types == {'r': 'str or int', 's': 'str'}

    def test_every_corpus_program_infers_without_error(self):
        progs = native_coverage.collect_corpus()
        assert len(progs) > 200
        for p in progs:
            ti = tiltypes.infer_program(til.resolve_imports(til.parse_source(p.source, p.filename),
                                                            p.filename), p.source, p.filename)
            assert ti.errors == [], f"E-TYPE on a valid program ({p.label}):\n{ti.errors[0]}"
            if 'main' in ti.funcs:
                ti.specialize('main', [])


# ═══════════════════════════════════════════════════════════════
# E-TYPE: definite errors only
# ═══════════════════════════════════════════════════════════════

STRUCTS = 'struct P\n    x: int\nstruct Q\n    y: int\n'


class TestEType:
    @pytest.mark.parametrize("src, fragment", [
        ('main()\n    let x: int = "a"\n', "'x' is int, but the value is the string 'a', which is not a number"),
        ('main()\n    let s: str = 5\n', "'s' is str, but the value is int"),
        (STRUCTS + 'main()\n    let p = P { x: 1 }\n    print(p.z)\n', "struct 'P' has no field 'z'"),
        (STRUCTS + 'main()\n    let p = P { x: 1 }\n    p.nope()\n', "struct 'P' has no method 'nope'"),
        (STRUCTS + 'main()\n    var p = P { x: 1 }\n    p.w = 2\n', "struct 'P' has no field 'w'"),
        (STRUCTS + 'main()\n    let p: P = Q { y: 1 }\n', "'p' is P, but the value is Q"),
        ('main()\n    print(1 - "a")\n', "unsupported operand types for -: int and str"),
        ('main()\n    print("a" * "b")\n', "unsupported operand types for *: str and str"),
        ('main()\n    let t = true + 1\n', "unsupported operand types for +: bool and int"),
        ('f(s: str) -> int\n    return len(s)\nmain()\n    print(f(5))\n',
         "argument 's' of 'f' is str, but the value is int"),
        ('main()\n    var n: int = 0\n    n = "s"\n', "'n' is int, but the value is str"),
        ('f() -> int\n    return "x"\nmain()\n    print(f())\n', "the return value of 'f' is int"),
        ('f() -> [int]\n    return 5\nmain()\n    print(f())\n', "is [int], but the value is int"),
        (STRUCTS + 'main()\n    let p = P { x: "one" }\n', "field 'x' of 'P' is int"),
    ])
    def test_definite_errors(self, src, fragment):
        errors = etype_errors(src)
        assert errors, "expected an E-TYPE error"
        assert "E-TYPE" in errors[0] and fragment in errors[0], errors[0]

    @pytest.mark.parametrize("src", [
        # depends on the generic parameter: checked at run time by the VMs
        'f(x)\n    return x - "a"\nmain()\n    print(f(1))\n',
        # recovery blocks are where failing on purpose is fine
        'main()\n    let r = retry(1)\n        1 - "a"\n    else\n        0\n    print(r)\n',
        # the VMs convert numbers at annotated declarations and parameters
        'main()\n    let x: float = 1\n    let y: int = 2.5\n    let z: int = "12"\n    print(x, y, z)\n',
        'to_int(x: int) -> int\n    return x\nmain()\n    print(to_int("12"), to_int(true))\n',
        # a failing conversion of an argument is the callee's run-time error
        'need(x: int) -> int\n    return x\nmain()\n    print(need("nope"))\n',
        # None fits references
        'struct P\n    x: int\nmain()\n    let s: str = None\n    var p: P = None\n    print(s, p)\n',
        # a number given to `bool` stays a number on the VMs (no conversion, no error)
        'h(b: bool, n: int) -> bool\n    if n <= 0\n        return b\n    return h(n, n - 1)\n'
        'f(b: bool = 5) -> bool\n    let k: bool = 0\n    return b\nmain()\n    print(h(true, 1), f())\n',
        # dynamic code
        'main()\n    var v = 1\n    v = "x"\n    print(v * 2)\n',
        'main()\n    let m = {"a": 1}\n    print(m.get_or(5, 0), m.has(true))\n',
        'main()\n    print(true == 1, 1 == "1", [1] == "x")\n',
        # comparisons, calls of non-functions etc. are reported by the VMs at run time
        'main()\n    print(1 < "a")\n',
        'main()\n    let f = 5\n    print(f())\n',
    ])
    def test_valid_programs_have_no_etype(self, src):
        assert etype_errors(src) == []

    def test_error_format(self):
        err = etype_errors('main()\n    let total = 1\n    let x: int = "a"\n')[0]
        assert "error[E-TYPE]" in err and "<test>:3:5" in err and "hint:" in err

    def test_vm_compile_check_and_native_report_etype(self, tmp_path):
        bad = tmp_path / 'bad.til'
        bad.write_text('main()\n    let s: str = 5\n    print(s)\n')
        for args in (['check'], ['run', '--vm'], ['run'], ['build', '--target', 'c']):
            r = cli(*args, str(bad))
            assert r.returncode == 1 and "E-TYPE" in r.stderr, (args, r.stderr)
        # --no-types: the VM just runs it (annotations are not enforced there)
        r = cli('run', '--vm', '--no-types', str(bad))
        assert r.returncode == 0 and r.stdout == "5\n", r.stderr

    def test_compile_to_bytecode_raises(self):
        with pytest.raises(til.CompileError) as info:
            til.compile_to_bytecode('main()\n    let x: int = "a"\n')
        assert "E-TYPE" in str(info.value)


# ═══════════════════════════════════════════════════════════════
# MONOMORPHISATION
# ═══════════════════════════════════════════════════════════════

@needs_cc
class TestGenerics:
    def test_generic_function_used_with_int_and_str(self):
        src = ('twice(x)\n    return x + x\nshow(v)\n    print(f"value: {v}")\n'
               'main()\n    show(twice(21))\n    show(twice("ab"))\n    print(twice(1.5))\n')
        c = c_code(src)
        for name in ('til_twice__int(', 'til_twice__str(', 'til_twice__float(', 'til_show__int(',
                     'til_show__str('):
            assert name in c, name
        assert native_matches_vm(src) == "value: 42\nvalue: abab\n3\n"

    def test_explicit_generic_instances(self):
        src = ('max_of<T>(a: T, b: T) -> T\n    if a > b\n        return a\n    return b\n'
               'main()\n    print(max_of(3, 7), max_of("pear", "apple"), max_of(2.5, 1.5))\n')
        c = c_code(src)
        assert 'til_max_of__int(int64_t a, int64_t b)' in c
        assert 'til_max_of__str(const char* a, const char* b)' in c
        assert 'til_max_of__float(double a, double b)' in c
        assert native_matches_vm(src) == "7\npear\n2.5\n"

    def test_generic_recursion_and_containers(self):
        src = ('first(xs)\n    return xs[0]\nwrap(v)\n    return [v, v]\nfact(n)\n    if n <= 1\n        return 1\n'
               '    return n * fact(n - 1)\ncount_if(xs, limit)\n    var n = 0\n    for x in xs\n'
               '        if x > limit\n            n += 1\n    return n\n'
               'main()\n    print(first([3, 4]), first(["s"]), wrap(1), wrap("a"))\n    print(fact(20))\n'
               '    print(count_if([1, 5, 9], 4), count_if(["a", "z"], "m"))\n')
        assert native_matches_vm(src) == '3\ns\n[1, 1]\n["a", "a"]\n2432902008176640000\n2\n1\n'

    def test_an_instance_with_path_dependent_types_falls_back(self):
        # fact(5.0) is a float on one path and the int 1 on the other: the VM runs it
        with pytest.raises(til.NativeUnsupported):
            c_code('fact(n)\n    if n <= 1\n        return 1\n    return n * fact(n - 1)\n'
                   'main()\n    print(fact(5.0))\n')

    def test_generic_method(self):
        src = ('struct Box\n    items: [str]\nimpl Box\n    add(self, v)\n        self.items.push(str(v))\n'
               '        return len(self.items)\n'
               'main()\n    let b = Box { items: [] }\n    b.add(1)\n    b.add("x")\n    print(b.add(2.5), b.items)\n')
        assert native_matches_vm(src) == '3\n["1", "x", "2.5"]\n'


# ═══════════════════════════════════════════════════════════════
# NATIVE == VM
# ═══════════════════════════════════════════════════════════════

# programs from the test-suite that used to fall back to the VM (baseline)
FORMERLY_FALLBACK = {
    'script_primes': ('var primes = []\nfor n in 2..20\n    var ok = true\n    for d in primes\n'
                      '        if n % d == 0\n            ok = false\n    if ok\n        primes.push(n)\n'
                      'let summary = {"count": primes.len(), "last": primes.last()}\nprint(summary)\n'),
    'higher_order': ('main()\n    let xs = [1, 2, 3, 4, 5, 6]\n    print(xs.map(|x| x * x))\n'
                     '    print(xs.filter(|x| x % 2 == 0))\n    print(xs.reduce(|a, b| a + b, 0))\n'
                     '    print(xs.any(|x| x > 5))\n    print(xs.all(|x| x > 0))\n    xs.each(|x| print(x * 0))\n'
                     '    print(xs |> len)\n'),
    'maps': ('main()\n    var m = {"b": 2, "a": 1}\n    m["c"] = 3\n    m["a"] += 10\n'
             '    m.delete("b")\n    print(m)\n    print(m.keys())\n    print(m.get_or("zz", 0))\n'
             '    print(m.has("c"))\n    for k in m\n        print(k)\n'),
    'string_methods': ('main()\n    let s = "  Hello, World  "\n    print(s.trim().to_upper())\n'
                       '    print("a,b,,c".split(","))\n    print(["x", "y"].join("-"))\n'
                       '    print("banana".count("an"))\n    print("banana".replace("an", "AN"))\n'
                       '    print("abc".slice(1, 10))\n    print("ab" * 3)\n    print("x y  z".split())\n'),
    'option_result_match': ('parse(s: str) -> Result<int, str>\n    if s == ""\n        return Err("empty")\n'
                            '    return Ok(s.to_int())\n'
                            'twice(s: str) -> Result<int, str>\n    let v = parse(s)?\n    return Ok(v * 2)\n'
                            'main()\n    for s in ["21", ""]\n        match twice(s)\n'
                            '            Ok(v) => print(f"ok {v}")\n            Err(e) => print(f"err {e}")\n'
                            '    let o = Some(3)\n    match o\n        Some(x) if x > 2 => print("big")\n'
                            '        Some(x) => print("small")\n        None => print("none")\n'),
    'adt': ('enum Expr\n    Num(int)\n    Add(Expr, Expr)\n    Mul(Expr, Expr)\n'
            'eval(e: Expr) -> int\n    return match e\n        Expr.Num(n) => n\n'
            '        Expr.Add(a, b) => eval(a) + eval(b)\n        Expr.Mul(a, b) => eval(a) * eval(b)\n'
            'main()\n    let e = Expr.Add(Expr.Num(2), Expr.Mul(Expr.Num(3), Expr.Num(4)))\n'
            '    print(eval(e))\n    print(Expr.Num(7))\n'),
    'structural_equality': ('struct P\n    x: int\n    y: int\nmain()\n    print([1, 2] == [1, 2])\n'
                            '    print(P { x: 1, y: 2 } == P { x: 1, y: 2 })\n    print({"a": 1} == {"a": 2})\n'),
    'gc_style': ('main()\n    var keep = []\n    var total = 0\n    for i in 0..2000\n'
                 '        let s = f"item-{i}-" + str(i * i)\n        if i % 100 == 0\n'
                 '            keep.push(s.split("-")[1])\n        total += len(s)\n'
                 '    print(total)\n    print(keep.join(","))\n'),
    'retry_text': ('#[requires: n >= 0]\nfact(n: int) -> int\n    if n <= 1\n        return 1\n'
                   '    return n * fact(n - 1)\nmain()\n    var k = 0 - 3\n    let r = retry(2)\n        fact(k)\n'
                   '    else\n        error\n    print(r)\n'),
    'intent_error': ('main()\n    let r = retry(1)\n        intent "parse config"\n            panic("bad token")\n'
                     '    else\n        error\n    print(r)\n'),
    'some_value': 'main()\n    let x = Some(42)\n    print(x, x.unwrap() + 1)\n',
    'closures_as_values': ('make_counter()\n    var n = 0\n    return |step: int| n + step\n'
                           'adder(k: int)\n    return |x: int| x + k\nmain()\n    let add5 = adder(5)\n'
                           '    print(add5(10))\n    var base = 1\n    let f = |x: int| x + base\n    base = 100\n'
                           '    print(f(1))\n    print(make_counter()(3), [1, 2].map(add5), add5)\n'),
    'ranges': 'print(range(4))\nfor i in range(2)\n    print(i)\nprint(sum([1, 2, 3]))\n',
    'args_files': ('let a = args()\nprint(len(a))\nfor x in a\n    print("[" + x + "]")\n'
                   'print(file_exists("/no/such/file"))\n'),
}


@needs_cc
class TestNativeEqualsVM:
    @pytest.mark.parametrize("name", sorted(FORMERLY_FALLBACK))
    def test_formerly_fell_back(self, name):
        native_matches_vm(FORMERLY_FALLBACK[name])

    def test_former_silent_mismatches_are_fixed(self):
        # the classic backend printed a pointer, 1 and 2.5 here
        out = native_matches_vm('main()\n    let xs = [1, 2, 3]\n    print(xs)\n    print(min(1.5, 2.5))\n'
                                '    print(abs(-5) / 2)\n    print("b" > "a", "é"[0], -7 / 2, -7 % 3)\n')
        assert out == "[1, 2, 3]\n1.5\n2\ntrue\n195\n-3\n-1\n"

    def test_runtime_errors_have_the_vm_messages(self):
        src = ('main()\n    var zero = 0\n    var empty = []\n    let errs = [\n'
               '        retry(1)\n            str(10 / zero)\n        else\n            error\n'
               '    ]\n    print(errs)\n    let b = retry(1)\n        str([1, 2][5])\n    else\n        error\n'
               '    print(b)\n    let c = retry(1)\n        str({"a": 1}["q"])\n    else\n        error\n'
               '    print(c)\n    let d = retry(1)\n        str("x1".to_int())\n    else\n        error\n    print(d)\n'
               '    let e = retry(1)\n        str(empty.pop())\n    else\n        error\n    print(e)\n')
        assert native_matches_vm(src) == ('["division by zero"]\nindex out of bounds: index 5, length 2\n'
                                          'key not found: q\ninvalid integer: "x1"\npop() from empty array\n')

    def test_uncaught_error_exit_code(self):
        native_matches_vm('main()\n    print("before")\n    let xs = [1]\n    print(xs[3])\n', expect_rc=1)

    def test_structs_are_references(self):
        src = ('struct C\n    n: int\nbump(c: C)\n    c.n += 100\nmain()\n    var a = C { n: 1 }\n'
               '    let b = a\n    b.n = 5\n    bump(a)\n    let cs = [C { n: 1 }, C { n: 2 }]\n'
               '    for c in cs\n        c.n *= 10\n    print(a.n, b.n, cs)\n')
        assert native_matches_vm(src) == '105\n105\n[C { n: 10 }, C { n: 20 }]\n'

    def test_evaluation_order(self):
        src = ('var log = []\nnote(s: str) -> int\n    log.push(s)\n    return len(log)\n'
               'main()\n    let a = note("x") + note("y") * 10\n    var xs = [1, 2, 3]\n'
               '    print(a, xs, xs.pop(), xs)\n    print([note("p"), note("q")], log)\n')
        assert native_matches_vm(src) == '21\n[1, 2]\n3\n[1, 2]\n[3, 4]\n["x", "y", "p", "q"]\n'

    def test_numbers_like_the_vm(self):
        src = ('main()\n    print(9223372036854775807 + 1, 2 ** 64, 3 ** 40, 1 << 64, -16 >> 2)\n'
               '    print(0.1 + 0.2, 1.0 / 0.0, 0.0 / 0.0, -0.0, 1e300 * 1e10, round(-2.5))\n'
               '    print(true == 1, int(-3.9), f"{255:x}|{0.25:%}|{7:03}|{2.5:^7}|")\n')
        native_matches_vm(src)

    def test_exact_int_float_comparison_like_the_reference_vm(self):
        # Python semantics (the reference VM): 2^53 + 1 is not 2^53 as a float.
        # (The C VM currently rounds the int first; see docs/TIL_TYPES.md.)
        src = ('main()\n    var big = 9007199254740993\n'
               '    print(big == 9007199254740992.0, big > 9007199254740992.0, f"{1e100:.1f}")\n')
        rc, vm_out = run_vm(src)
        assert rc == 0 and vm_out.startswith("false\ntrue\n10000000000000000159")
        assert run_native(src) == (0, vm_out)

    def test_nul_bytes_are_explicit_errors(self):
        """Native strings are C strings: a NUL byte from chr(0) or input() stops
        the program with an explicit error instead of truncating the string."""
        assert run_vm('main()\n    let z = 0\n    print(len(chr(z)))\n') == (0, "1\n")
        src = 'main()\n    let z = 0\n    print(len(chr(z)))\n'
        compiler = til.TILCompiler()
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, 'nul')
            assert compiler.compile_to_executable(src, exe, "<test>", quiet=True)
            r = subprocess.run([exe], capture_output=True, timeout=60)
            assert r.returncode == 1 and r.stdout == b"" and b"NUL" in r.stderr
            assert compiler.compile_to_executable('main()\n    let s = input()\n    print(len(s))\n', exe,
                                                  "<test>", quiet=True)
            r = subprocess.run([exe], input=b"ab\x00c\n", capture_output=True, timeout=60)
            assert r.returncode == 1 and r.stdout == b"" and b"NUL" in r.stderr
            r = subprocess.run([exe], input=b"abc\n", capture_output=True, timeout=60)
            assert r.returncode == 0 and r.stdout == b"3\n"

    def test_sorting_nan_is_an_explicit_error(self):
        # where nan ends up depends on the sort algorithm (host Python's on the reference VM)
        src = 'main()\n    let z = 0.0\n    var xs = [3.0, z / z, 1.0]\n    xs.sort()\n    print(len(xs))\n'
        assert run_vm(src) == (0, "3\n")
        rc, out = run_native(src)
        assert rc == 1 and out == ""
        assert native_matches_vm('main()\n    var xs = [3.0, 0.0, -0.0, -1.5]\n    xs.sort()\n    print(xs)\n') == \
            "[-1.5, 0, -0, 3]\n"

    def test_integer_literals_beyond_64_bits_fold_like_the_bytecode_compiler(self):
        # const_eval folds with the exact value; at run time the literal is wrapped
        src = ('const BIG = 18446744073709551616\nmain()\n    print(BIG / 2, BIG > 0, BIG % 10, BIG)\n'
               '    let big = 18446744073709551616\n    print(big > 0, 18446744073709551616 > 0, 0 < BIG < 10)\n')
        assert native_matches_vm(src) == "-9223372036854775808\ntrue\n6\n0\nfalse\ntrue\nfalse\n"

    def test_floor_ceil_and_integer_formats_of_big_floats(self):
        # floor/ceil are float(math.floor(x)) on the reference VM (never -0);
        # {x:d} of a float beyond int64 is the exact integer (Python's int(x)).
        # Compared with the reference VM only: the C VM differs here
        # (docs/TIL_TYPES.md, "Known differences of the C VM").
        src = ('main()\n    let a = 0.0 - 0.00001\n    let z = 0.0 - 0.0\n'
               '    print(ceil(a), floor(z), ceil(-0.5), floor(-0.5), round(a))\n'
               '    let big = 1e100\n    let neg = 0.0 - 9223372036854775808.0\n'
               '    print(f"{big:d}", f"{neg:x}", f"{2.0 ** 70:b}")\n'
               '    print(f"{123.9:d}", f"{-0.5:d}", f"{1e19:X}")\n')
        rc, out = run_vm(src)
        assert rc == 0 and run_native(src) == (0, out)
        assert out.startswith("0\n0\n0\n-1\n-0\n1000000000000000015902891109759918046836080856394528138978132755"
                              "7747838772170381060813469985856815104\n-8000000000000000\n1" + "0" * 70 + "\n123\n0\n")

    def test_kazakh_program(self):
        native_matches_vm('қосу(а: int, б: int) -> int\n    return а + б\nнегізгі()\n'
                          '    тұрақты нәтиже = қосу(2, 3)\n    басып_шығару(нәтиже)\n')

    def test_none_until_assigned_and_implicit_returns(self):
        # the VMs start `var x` (and non-primitive annotations) as None, primitive
        # annotations as 0 / "" / false, and a function that ends without
        # returning a value returns None
        src = ('struct P\n    x: int\nlabel(n: int)\n    if n > 0\n        return "pos"\n'
               'pick(n: int) -> str\n    if n > 0\n        return "yes"\n    return\n'
               'var names: [str]\n'
               'main()\n    var s\n    print(s)\n    s = "hi"\n    var p: P\n    var xs: [int]\n'
               '    print(s, p, xs, names, label(-1), label(2), pick(0), pick(1))\n'
               '    var n: int\n    var t: str\n    print(n, len(t))\n')
        assert native_matches_vm(src) == 'None\nhi\nNone\nNone\nNone\nNone\npos\nNone\nyes\n0\n0\n'


# ═══════════════════════════════════════════════════════════════
# FALLBACK TO THE VM
# ═══════════════════════════════════════════════════════════════

DYNAMIC_PROGRAM = 'main()\n    let mixed = [1, "a", 2.5]\n    print(mixed)\n'


class TestFallback:
    def test_unsupported_programs_raise_native_unsupported(self):
        with pytest.raises(til.NativeUnsupported) as info:
            c_code(DYNAMIC_PROGRAM)
        assert "native backend:" in str(info.value) and "dynamically typed" in str(info.value)
        with pytest.raises(til.NativeUnsupported):
            c_code('main()\n    var x = 1\n    x = "a"\n    print(x)\n')
        # a closure cannot copy a variable that changes after it is created
        with pytest.raises(til.NativeUnsupported) as info:
            c_code('make()\n    var total = 0\n    let f = |x: int| x + total\n    total = 10\n    return f\n'
                   'main()\n    print(make()(1))\n')
        assert "changes later" in str(info.value)

    @pytest.mark.parametrize("src, vm_out", [
        ('let A = B == NULL\nlet B = 5 + 0\nmain()\n    print(A)\n', "true\n"),
        ('let A = f()\nlet B = 5 + 0\nf()\n    return B\nmain()\n    print(A, B)\n', "None\n5\n"),
        ('let A = B\nlet B = C + 1\nlet C = 2\nmain()\n    print(A, B, C)\n', "None\n3\n2\n"),
        # through a struct field default
        ('struct S\n    v: int = L\nlet A = S {}\nlet L = 3 + 0\nmain()\n    print(A.v)\n', "None\n"),
        # a `var` with a literal value is set in order too (only `let` literals are inlined)
        ('let A = f()\nvar B = 5\nf()\n    return B\nmain()\n    print(A, B)\n', "None\n5\n"),
        ('#[requires: x < LIMIT]\nf(x: int) -> int\n    return x\nlet A = retry(1)\n        f(1)\n'
         '    else\n        0 - 1\nlet LIMIT = 10 + 0\nmain()\n    print(A)\n', "-1\n"),
        # script mode: a top-level statement before the line that sets X
        ('show()\n    print(X)\nshow()\nlet X = 5 + 0\nshow()\n', "None\n5\n"),
    ])
    def test_a_global_read_before_it_is_set(self, src, vm_out):
        """Literal globals are set before anything runs (in the VMs and as C
        initialisers); a computed one is None until its line has run, which a
        native int cannot hold."""
        assert run_both(src) == vm_out
        with pytest.raises(til.NativeUnsupported) as info:
            c_code(src)
        assert "is read before its value is set" in str(info.value)

    @needs_cc
    def test_globals_read_after_they_are_set_stay_native(self):
        src = ('let B = 5 + 0\nlet A = f() + C\nlet C = 2\nf() -> int\n    return B\n'
               'main()\n    print(A, B, f())\n')
        assert native_matches_vm(src) == "7\n5\n5\n"
        assert native_matches_vm('print("start")\nlet X = 5 + 0\nf() -> int\n    return X\nprint(f())\n') == \
            "start\n5\n"
        # a string (any reference) read before its line has run is None natively too
        assert native_matches_vm('let A = g()\nvar S = "s"\nlet T = "t"\ng()\n    return [S, T]\n'
                                 'main()\n    print(A, S)\n') == '[None, "t"]\ns\n'

    def test_a_number_given_to_a_bool_parameter_runs_on_the_vm(self):
        src = 'show(b: bool)\n    print(b, b == true)\nmain()\n    show(5)\n    show(true)\n'
        assert run_both(src) == "5\nfalse\ntrue\ntrue\n"
        with pytest.raises(til.NativeUnsupported):
            c_code(src)

    @needs_cc
    def test_lambda_called_where_its_captured_variable_is_shadowed(self):
        # the VMs use the variable the lambda captured, not the one that shadows it
        src = ('run(f, v: int) -> int\n    return f(v)\n'
               'main()\n    let base = 1\n    let f = |x: int| x + base\n    print(f(1))\n'
               '    if true\n        let base = 50\n        print(f(1), base)\n    print(f(2))\n')
        assert run_both(src) == "2\n2\n50\n3\n"
        with pytest.raises(til.NativeUnsupported) as info:
            c_code(src)
        assert "captured variable 'base' is not visible" in str(info.value)
        # used as a value (passed on) where the captured variable is shadowed
        with pytest.raises(til.NativeUnsupported) as info:
            c_code(src.replace('print(f(1), base)', 'print(run(f, 1), base)'))
        assert "captured variable 'base' is not visible" in str(info.value)
        # not shadowed: native
        assert native_matches_vm(src.replace('let base = 50', 'let other = 50')
                                    .replace('print(f(1), base)', 'print(run(f, 1), other)')) == "2\n2\n50\n3\n"

    @needs_cc
    def test_run_falls_back_and_explains(self, tmp_path):
        prog = tmp_path / 'mixed.til'
        prog.write_text(DYNAMIC_PROGRAM)
        r = cli('run', str(prog))
        assert r.returncode == 0 and r.stdout == '[1, "a", 2.5]\n'
        assert "running it on the TIL VM" in r.stderr
        r = cli('run', '-v', str(prog))
        assert "dynamically typed" in r.stderr
        r = cli('build', str(prog), '-o', str(tmp_path / 'mixed'))
        assert r.returncode == 1 and "native backend:" in r.stderr

    @pytest.mark.parametrize("src", [
        'main()\n    var x\n    print(x)\n    x = 5\n    print(x)\n',
        'f(x: int) -> int\n    if x > 0\n        return 1\nmain()\n    print(f(-1))\n',
        'f(x: int) -> int\n    while x > 0\n        return x\nmain()\n    print(f(0))\n',
        'f(x: int) -> float\n    if x > 0\n        return 1.5\n    return\nmain()\n    print(f(0))\n',
        'var h: int\nmain()\n    print(h)\n',
        'enum E\n    A\n    B\nmain()\n    var e: E\n    print(e)\n',
        '#[generic: T]\nf(a: T) -> T\n    var y: T\n    print(y)\n    return a\nmain()\n    print(f(1))\n',
    ])
    def test_none_that_a_native_number_cannot_hold(self, src):
        """The VMs give None here (an int / float variable or result that was
        never set): the native backend cannot represent it, so it falls back."""
        assert run_vm(src)[1].startswith("None\n")
        with pytest.raises(til.NativeUnsupported) as info:
            c_code(src)
        assert "None" in str(info.value) or "dynamically typed" in str(info.value)

    def test_dead_code_does_not_force_a_fallback(self):
        c = c_code('unused(k: int)\n    let mixed = [k, "a"]\n    print(mixed)\nmain()\n    print(1)\n')
        assert 'til_main' in c and 'til_unused' not in c

    @needs_cc
    def test_classic_behaviour_without_the_module(self, tmp_path):
        """TIL_NO_TYPES (like an installation without src/tiltypes.py): no
        E-TYPE checks and the classic C backend."""
        bad = tmp_path / 'bad.til'
        bad.write_text('main()\n    let s: str = 5\n')
        assert cli('check', str(bad)).returncode == 1
        assert cli('check', str(bad), env={'TIL_NO_TYPES': '1'}).returncode == 0
        prog = tmp_path / 'p.til'
        prog.write_text('main()\n    let xs = [1, 2]\n    print(len("ab") + 3)\n')
        r = cli('build', '--target', 'c', str(prog), env={'TIL_NO_TYPES': '1'})
        assert r.returncode == 0 and 'TIL_Arr*' not in r.stdout and 'int64_t xs[2]' in r.stdout
        r = cli('run', str(prog), env={'TIL_NO_TYPES': '1'})
        assert r.returncode == 0 and r.stdout == "5\n", r.stderr


# ═══════════════════════════════════════════════════════════════
# COVERAGE (docs/TIL_TYPES.md)
# ═══════════════════════════════════════════════════════════════

def baseline():
    with open(BASELINE, encoding='utf-8') as f:
        return json.load(f)


class TestCoverage:
    def test_corpus_extraction(self):
        progs = native_coverage.collect_corpus()
        labels = {p.label.split(':')[0] for p in progs}
        assert {'tests/test_compiler.py', 'tests/test_v3.py', 'tests/test_foundation.py'} <= labels
        assert any(lbl.startswith('examples/') for lbl in labels)
        assert len({p.id for p in progs}) == len(progs)

    def test_c_is_generated_for_everything_that_compiled_before(self):
        """Fast check (no C compiler): no program that compiled natively in the
        baseline is rejected by the typed backend, and more are accepted."""
        before = {p['id']: p['status'] for p in baseline()['programs']}
        accepted, regressed = 0, []
        for prog in native_coverage.collect_corpus():
            try:
                til.TILCompiler().compile(prog.source, prog.filename)
                accepted += 1
            except til.NativeUnsupported as e:
                if before.get(prog.id) == 'native-ok':
                    regressed.append(f"{prog.label}: {e}")
        assert not regressed, "\n".join(regressed)
        assert accepted > baseline()['summary']['native']

    @slow
    @needs_cc
    def test_full_measurement_is_at_least_the_baseline(self):
        results = native_coverage.measure(native_coverage.collect_corpus())
        summary = native_coverage.summarize(results)
        assert native_coverage.compare(results, baseline()) == []
        assert summary['native'] >= baseline()['summary']['native']
        assert summary['native_mismatch'] == 0, [r for r in results if r['status'] == 'native-mismatch']
