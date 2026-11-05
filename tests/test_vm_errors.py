#!/usr/bin/env python3
"""
TIL 3.1 review fixes: builtins with wrong arguments on both VMs.

A builtin (native or builtin method) called with arguments of the wrong type or
count is a TIL runtime error with the same text on the reference VM (src/til.py)
and the C VM (vm/tilvm.c): `retry`/`timeout` catch it, an uncaught one is the same
report and exit code 1. The reference VM used to crash with an internal Python
exception instead (`Error: expected string or bytes-like object`, not catchable);
where the VMs accepted different inputs, the stricter behaviour is now the rule
of both (chr(65.0), sleep_ms(1.5), exit(2.5), timeout("abc"), slice(1.0), ...).
A differential fuzz calls every native and every builtin method with argument
tuples of all kinds of values (ints, floats, inf, nan, huge ints, strings, bools,
nil, arrays, maps, structs, closures, Option/Result) on both VMs.

Run: python3 -m pytest tests/test_vm_errors.py -v
"""

import io
import os
import subprocess
import sys
import tempfile

import pytest

from tilhelpers import CC, cvm_binary, needs_cc, run_cvm, tbc_bytes, til


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def run_py(data: bytes, cwd=None):
    """The reference VM: (exit code, stdout, stderr). A Python exception escaping
    the VM (an internal error) fails the test."""
    old_err, old_cwd = sys.stderr, os.getcwd()
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')
    out = []
    try:
        if cwd:
            os.chdir(cwd)
        module = til.TBCModule.from_bytes(bytes(data))
        rc = til.TILVM(module, write=out.append, read_line=lambda: b"").run()
        sys.stderr.flush()
        err = sys.stderr.buffer.getvalue().decode('utf-8', 'replace')
    finally:
        sys.stderr = old_err
        os.chdir(old_cwd)
    return rc, b"".join(out).decode('utf-8', 'replace'), err


def run_both(src: str, expect_rc: int = 0) -> str:
    """Both VMs: the same exit code, output and error report; returns the output."""
    data = tbc_bytes(src)
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err = run_py(data, cwd=tmp)
    assert rc == expect_rc, f"reference VM exit {rc}\n{out}\n{err}"
    if CC:
        with tempfile.TemporaryDirectory() as tmp:
            crc, cout, cerr = run_cvm(data, cwd=tmp)
        assert crc == rc, f"C VM exit {crc}, reference VM {rc}\n{cerr}"
        if cout != out:
            pl, cl = out.splitlines(), cout.splitlines()
            diff = [f"  py: {a}\n  c : {b}" for a, b in zip(pl, cl) if a != b][:20]
            pytest.fail("C VM output differs from the reference VM:\n" + "\n".join(diff) +
                        f"\n({len(pl)} / {len(cl)} lines)")
        assert cerr == err, f"error reports differ:\n{err}\n{cerr}"
    return out


STRUCT_P = "struct P\n    x: int\n\n"


def caught_program(exprs) -> str:
    """One function per expression: `retry(1) str(expr) else "E:" + error`; main
    prints their results in order (inf and nan are local variables)."""
    lines = [STRUCT_P]
    for i, e in enumerate(exprs):
        lines += [f"c{i}()", "    let inf = 1e308 * 10.0", "    let nan = inf - inf",
                  "    return retry(1)", f"        str({e})", "    else", "        \"E:\" + error", ""]
    lines.append("main()")
    lines += [f"    print(c{i}())" for i in range(len(exprs))]
    return "\n".join(lines) + "\n"


def check_cases(cases):
    out = run_both(caught_program([e for e, _ in cases])).splitlines()
    assert len(out) == len(cases)
    bad = [f"{e}: got {got!r}, expected {want!r}" for (e, want), got in zip(cases, out) if got != want]
    assert not bad, "\n".join(bad)


# ═══════════════════════════════════════════════════════════════
# THE REVIEW'S REPRODUCERS
# ═══════════════════════════════════════════════════════════════

class TestReproducers:
    def test_format_with_a_non_string_spec(self):          # repro fm.til
        src = """main()
    let r = retry(1)
        format(100.0, 0)
    else
        "E:" + error
    print(r)
    print("next")
"""
        assert run_both(src) == "E:format() expects a spec string, got int\nnext\n"

    def test_int_format_of_inf_and_nan(self):              # repro fm2.til
        src = """main()
    let x = 1e308 * 10.0
    let r = retry(1)
        f"{x:b}"
    else
        "E:" + error
    print(r)
    let y = x - x
    let q = retry(1)
        f"{y:x}"
    else
        "E:" + error
    print(q)
    print("next")
"""
        assert run_both(src) == "E:cannot convert nan/inf to int\n" * 2 + "next\n"

    def test_wrong_types_of_natives_and_methods(self):     # repro sl.til
        src = """main()
    var r = ""
    r = retry(1)
        sleep_ms(1.5)
        "slept"
    else
        "E:" + error
    print(r)
    r = retry(1)
        seed(2.5)
        "seeded"
    else
        "E:" + error
    print(r)
    r = retry(1)
        str(range(0.5, 3))
    else
        "E:" + error
    print(r)
    r = retry(1)
        str(chr(65.0))
    else
        "E:" + error
    print(r)
    r = retry(1)
        str("abc".repeat(2.0))
    else
        "E:" + error
    print(r)
    r = retry(1)
        str([1,2,3][1.0])
    else
        "E:" + error
    print(r)
    r = retry(1)
        str("abcdef".slice(1.0, 3))
    else
        "E:" + error
    print(r)
    r = retry(1)
        exit(2.5)
        "x"
    else
        "E:" + error
    print(r)
"""
        assert run_both(src) == (
            "E:sleep_ms() needs an integer\nE:seed() needs an integer\nE:range bounds must be integers\n"
            "E:chr() needs an integer\nE:repeat() needs an integer\nE:index must be an integer\n"
            "E:index must be an integer\nE:exit() needs an integer\n")

    def test_messages_that_differed(self):                 # repro em.til (the builtins part)
        check_cases([
            ('-[]', 'E:cannot negate array'),
            ('-true', 'E:cannot negate bool'),
            ('pow("a", 2)', 'E:pow expects a number, got str'),
            ('sum([None])', 'E:sum() needs numbers'),
            ('chr("x")', 'E:chr() needs an integer'),
            ('iter(5)', 'E:int is not iterable'),
        ])

    def test_timeout_needs_a_number(self):                 # repro to.til
        # the duration is evaluated before the block starts: its own `else` does not
        # catch the error, an enclosing retry does
        src = """main()
    let r = retry(1)
        let z = timeout("abc")
            5
        else
            "T:" + error
        str(z)
    else
        "E:" + error
    print(r)
    let q = timeout(2.5)
        7
    else
        "T:" + error
    print(q)
"""
        assert run_both(src) == "E:timeout expects a number, got str\n7\n"
        src = 'main()\n    let z = timeout("abc")\n        5\n    else\n        "T"\n    print(z)\n'
        run_both(src, expect_rc=1)

    def test_methods_with_missing_or_wrong_arguments(self):
        check_cases([
            ('"abc".repeat(2.0)', 'E:repeat() needs an integer'),
            ('"ab".repeat("x")', 'E:repeat() needs an integer'),
            ('"abc".repeat()', 'E:repeat() needs an integer'),
            ('"s".slice(1.0, 3)', 'E:index must be an integer'),
            ('"s".slice(0, {})', 'E:index must be an integer'),
            ('{1: "x"}.get_or(false)', 'E:get_or() needs a key and a default'),
        ])

    def test_uncaught_error_report(self):
        out = run_both('main()\n    print("a")\n    print(format(100.0, 0))\n', expect_rc=1)
        assert out == "a\n"
        rc, _out, err = run_py(tbc_bytes('main()\n    print("a")\n    print(format(100.0, 0))\n'))
        assert err.startswith("runtime error: format() expects a spec string, got int\n  at main (line 3)")


# ═══════════════════════════════════════════════════════════════
# ONE BEHAVIOUR FOR BOTH VMS
# ═══════════════════════════════════════════════════════════════

class TestUnifiedBehaviour:
    def test_natives(self):
        check_cases([
            # format(): the spec is a string; int kinds convert a float like int()
            ('format(1, None)', 'E:format() expects a spec string, got nil'),
            ('format(P { x: 1 }, [])', 'E:format() expects a spec string, got array'),
            ('f"{-inf:d}"', 'E:cannot convert nan/inf to int'),
            ('f"{-2.9:d}"', '-2'),
            ('f"{255:08X}"', '000000FF'),
            ('f"{7:>100}"', ' ' * 59 + '7'),                      # widths above 60 count as 60
            ('f"{1.5:.40f}"', '1.' + '5' + '0' * 29),            # precisions above 30 as 30
            ('format(2.5, ".f")', '2'),                          # "." alone: precision 0
            ('format(2.5, "5z")', '2.5'),                        # not a spec: the value's text
            # seed / sleep_ms / random / exit take integers
            ('seed("x")', 'E:seed() needs an integer'),
            ('sleep_ms(None)', 'E:sleep_ms() needs an integer'),
            ('random(1.5, 3)', 'E:random(lo, hi) needs lo <= hi'),
            ('random("a", 1)', 'E:random(lo, hi) needs lo <= hi'),
            ('random(-9223372036854775807 - 1, 9223372036854775807) != 0.5', 'true'),   # C VM: SIGFPE
            # chr: an integer, a Unicode scalar value (no surrogates)
            ('chr(true)', 'E:chr() needs an integer'),
            ('chr(-1)', 'E:chr() argument out of range'),
            ('chr(55296)', 'E:chr() argument out of range'),
            ('chr(1114112)', 'E:chr() argument out of range'),
            ('chr(233)', 'é'),
            # ord reads the first UTF-8 sequence without validating it
            ('ord(bytes_to_str([255]))', '255'),
            ('ord(bytes_to_str([195, 169]))', '233'),
            ('ord(bytes_to_str([226, 130]))', '130'),
            # numbers
            ('pow(2, None)', 'E:pow expects a number, got nil'),
            ('pow(-2, 1e300)', 'inf'),
            ('pow(0, -2)', 'inf'),
            ('sum(["a"])', 'E:sum() needs numbers'),
            ('sum([1, true])', 'E:sum() needs numbers'),
            ('sum(1, 2.5)', '3.5'),
            ('round(inf)', 'inf'),
            ('round(-inf)', '-inf'),
            ('inf % 2.0', 'nan'),
            ('int([1])', 'E:cannot convert array to int'),
            ('float({})', 'E:cannot convert map to float'),
            # sizes
            ('range(0, 9223372036854775807)', 'E:range too large'),
            ('range(-9223372036854775807 - 1, 1)', 'E:range too large'),
            ('range(3, 3, true)', '[3]'),
            ('"x".repeat(4294967296)', 'E:string too long'),
            ('"ab" * 4294967296', 'E:string too long'),
            ('"ab".repeat(-1)', ''),
        ])

    def test_structs_are_named_by_their_type(self):
        check_cases([
            ('len(P { x: 1 })', 'E:len() of P'),
            ('abs(Err("e"))', 'E:abs expects a number, got Err'),
            ('int(Some(1))', 'E:cannot convert Some to int'),
            ('iter(P { x: 1 })', 'E:P is not iterable'),
            ('-P { x: 1 }', 'E:cannot negate P'),
            ('P { x: 1 } < 1', 'E:cannot compare P and int'),
            ('P { x: 1 } + 1', 'E:unsupported operand types for +: P and int'),
            ('{P { x: 1 }: 1}', 'E:P cannot be a map key'),
            ('str_bytes(Some(2))', 'E:str_bytes() expects a string, got Some'),
            ('read_file(P { x: 1 })', 'E:read_file() expects a path string, got P'),
            ('P { x: 1 }.len()', "E:no method 'len' on P"),
            ('Some(1).len()', "E:no method 'len' on Some"),
            ('5.len()', "E:no method 'len' on int"),
        ])

    def test_methods(self):
        check_cases([
            ('"abcdef".slice(true)', 'E:index must be an integer'),
            ('[1, 2].slice(1.0)', 'E:index must be an integer'),
            ('[1, 2, 3].slice(1)', '[2, 3]'),
            ('"abcdef".slice(-2, 99)', 'abcdef'),
            ('{1: "x"}.get_or()', 'E:get_or() needs a key and a default'),
            ('{1: "x"}.get_or(1)', 'E:get_or() needs a key and a default'),
            ('{1: "x"}.get_or(2, "d")', 'd'),
            ('None.unwrap_or()', 'E:unwrap_or() needs a default'),
            ('Some(1).unwrap_or()', 'E:unwrap_or() needs a default'),
            ('None.unwrap_or(3)', '3'),
            ('Err("e").expect()', 'E:expect() failed'),
            ('Err("e").expect("boom")', 'E:boom'),
            ('[].push()', 'E:push() needs a value'),
            ('[1].set(0)', 'E:set() needs an index and a value'),
            ('[1].set(1.0, 2)', 'E:index must be an integer'),
            ('{}.set(1)', 'E:set() needs a key and a value'),
            ('[1].insert(0)', 'E:insert() needs an index and a value'),
            ('[1].insert(1.5, 0)', 'E:index must be an integer'),
            ('[1].insert(5, 0)', 'E:insert index out of bounds: 5'),
            ('[1].remove(1.0)', 'E:index must be an integer'),
            ('[1].contains()', 'E:method needs a value'),
            ('{1: 2}.get()', 'E:map method needs a key'),
            ('"abc".replace("a")', 'E:replace() needs two strings'),
            ('"abc".replace(1, 2)', 'E:replace() needs two strings'),
            ('"a,b".split(1)', 'E:split() needs a string separator'),
            ('"abc".contains()', 'E:str.contains expects a string argument'),
            ('"abc".get()', 'E:index must be an integer'),
            ('{1: 2}.get([1])', 'E:array cannot be a map key'),
            ('{1: 2}.has({})', 'E:map cannot be a map key'),
            ('{1: 2}[[1]]', 'E:array cannot be a map key'),
            ('{1: "x"}.get(true)', 'E:key not found: true'),
        ])

    def test_bool_map_keys_are_not_ints(self):
        src = """main()
    var m = {1: "one", true: "yes", 0: "zero", false: "no", 1.0: "float one", None: "nil"}
    print(m, len(m), m.keys(), m.values())
    print(m[1], m[true], m[0], m[false], m.get_or(true, "-"), m.has(false), m.has(2))
    m.delete(true)
    print(m, m.has(true), m.has(1))
    for k in m
        print(k, type_of(k))
    print({true: 1} == {1: 1}, {true: 1} == {true: 1})
"""
        assert run_both(src) == (
            '{1: "float one", true: "yes", 0: "zero", false: "no", None: "nil"}\n5\n[1, true, 0, false, None]\n'
            '["float one", "yes", "zero", "no", "nil"]\n'
            'float one\nyes\nzero\nno\nyes\ntrue\nfalse\n'
            '{1: "float one", 0: "zero", false: "no", None: "nil"}\nfalse\ntrue\n'
            '1\nint\n0\nint\nfalse\nbool\nNone\nnil\n'
            'false\ntrue\n')

    def test_bool_map_keys_in_portable_snapshots(self, tmp_path):
        src = """main()
    var m = {1: "one", true: "yes", false: "no", 0: "zero"}
    var t = 0
    for i in 0..20
        t += i
    print(m, m[true], m[1], t)
"""
        expected = '{1: "one", true: "yes", false: "no", 0: "zero"}\nyes\none\n190\n'
        data = tbc_bytes(src)
        vm = til.TILVM(til.TBCModule.from_bytes(data), write=[].append)
        assert vm.run_steps(5) == vm.YIELD
        snap = vm.save_portable()
        out = []
        vm2 = til.TILVM(til.TBCModule.from_bytes(data), write=out.append)
        vm2.load_portable(snap)
        assert vm2.run() == 0 and b"".join(out).decode() == expected
        if CC:
            (tmp_path / "p.tbc").write_bytes(data)
            (tmp_path / "p.tilp").write_bytes(snap)
            r = subprocess.run([cvm_binary(), '--resume-portable', str(tmp_path / "p.tilp"),
                                str(tmp_path / "p.tbc")], capture_output=True, timeout=60)
            assert (r.returncode, r.stdout.decode()) == (0, expected), r.stderr

    def test_sort_with_nan_and_split_on_whitespace(self):
        src = """main()
    let inf = 1e308 * 10.0
    let nan = inf - inf
    var a = [3, nan, 1, 2.5, nan, -1]
    a.sort()
    print(a)
    print(bytes_to_str([97, 11, 98, 32, 99, 9, 10, 100]).split())
"""
        assert run_both(src) == '[3, nan, 1, 2.5, nan, -1]\n["a\x0bb", "c", "d"]\n'

    def test_functions_without_upvalues_are_equal(self):
        src = """mk()
    return |x| x + 1

main()
    let f = mk()
    let g = mk()
    let n = 5
    let h = |x| x + n
    print(f == g, [f].contains(g), [h].contains(|x| x + n), [h].index_of(h))
"""
        assert run_both(src) == "true\ntrue\nfalse\n0\n"

    def test_random_is_the_same_generator(self):
        src = """main()
    seed(42)
    print(random(1, 1000000), random(-5, 5), random(-9223372036854775807 - 1, 9223372036854775807))
    print(f"{random():.15f}")
    seed(-7)
    print(random(0, 9223372036854775807))
    seed(0)
    print(random(1, 100))
    let r = retry(1)
        random(5, 1)
    else
        error
    print(r, random(1, 100))
"""
        out = run_both(src)
        # pin the reference VM's generator (xorshift64*)
        assert out.splitlines()[0] == "693601"

    def test_exit_codes(self):
        for arg, rc in (('2', 2), ('300', 44), ('-1', 255), ('9223372036854775807', 255)):
            assert run_both(f'main()\n    print("a")\n    exit({arg})\n    print("b")\n', expect_rc=rc) == "a\n"
        out = run_both('main()\n    exit("x")\n', expect_rc=1)
        assert out == ""


# ═══════════════════════════════════════════════════════════════
# THE SAFETY NET
# ═══════════════════════════════════════════════════════════════

class TestSafetyNet:
    """A Python exception that escapes a builtin of the reference VM (a check it
    lacks) is a TIL runtime error, not an internal crash."""

    @staticmethod
    def run(src):
        rc, out, err = run_py(tbc_bytes(src))
        return rc, out, err

    def test_native(self, monkeypatch):
        def broken(v, spec, to_str):
            raise TypeError("an internal problem")
        monkeypatch.setattr(til, 'til_format_spec', broken)
        src = 'main()\n    let r = retry(1)\n        format(1, "x")\n    else\n        "E:" + error\n    print(r)\n'
        assert self.run(src)[:2] == (0, "E:format(): invalid arguments\n")
        rc, out, err = self.run('main()\n    print(format(1, "x"))\n')
        assert rc == 1 and err.startswith("runtime error: format(): invalid arguments\n  at main (line 2)")

    def test_method(self, monkeypatch):
        def broken(self, args, n):
            raise IndexError("an internal problem")
        monkeypatch.setattr(til.TILVM, 'slice_bounds', broken)
        src = 'main()\n    let r = retry(1)\n        "abc".slice(1)\n    else\n        "E:" + error\n    print(r)\n'
        assert self.run(src)[:2] == (0, "E:slice(): invalid arguments\n")

    def test_out_of_memory(self):
        # [x] * n: the reference VM (and the native backend) repeat arrays
        src = 'main()\n    let r = retry(1)\n        str([1, 2] * 9223372036854775807)\n' \
              '    else\n        "E:" + error\n    print(r)\n'
        assert self.run(src)[:2] == (0, "E:out of memory\n")

    def test_other_instructions_still_report_vm_bugs(self, monkeypatch):
        def broken(self, op, a, b):
            raise ValueError("a VM bug")
        monkeypatch.setattr(til.TILVM, 'arith', broken)
        with pytest.raises(ValueError, match="a VM bug"):
            self.run('main()\n    let a = 2.5\n    print(a * 2)\n')

    def test_host_control_flow_passes(self, monkeypatch):
        """Only Python's own exceptions are converted: a host's (the debugger pauses
        with one at an instruction boundary) goes through."""
        class HostStop(Exception):
            pass

        def stop(self, nid, args):
            raise HostStop()
        monkeypatch.setattr(til.TILVM, 'call_native', stop)
        with pytest.raises(HostStop):
            self.run('main()\n    let r = retry(1)\n        len("x")\n    else\n        "E"\n    print(r)\n')

    def test_debugger_has_the_safety_net(self, monkeypatch):
        """`til debug` runs a copy of TILVM.execute(): the same conversion."""
        import tildebug

        def broken(v, spec, to_str):
            raise TypeError("an internal problem")
        monkeypatch.setattr(til, 'til_format_spec', broken)
        module = til.TBCModule.from_bytes(tbc_bytes(
            'main()\n    let r = retry(1)\n        format(1, "x")\n    else\n        "E:" + error\n    print(r)\n'))
        out = []
        vm = tildebug.DebugVM(module, out.append, lambda: b"")
        assert vm.resume(lambda ev, line, depth, top: False) == 0
        assert b"".join(out) == b"E:format(): invalid arguments\n"


# ═══════════════════════════════════════════════════════════════
# DIFFERENTIAL FUZZ: every native and builtin method, all kinds of values
# ═══════════════════════════════════════════════════════════════

VALUES = ['3', '-2', '0', '2.5', '-1.5', '"ab"', '""', 'true', 'None', '[1, 2]', '[]', '{1: "x"}',
          'P { x: 1 }', '|x| x', 'inf', 'nan', '-inf', '9223372036854775807', '-9223372036854775807 - 1',
          '4294967296', '65', 'Some(1)', 'Err("e")', '" 12 "', '1e300', '"x:>8"', '1.0', '"a"']
MID = [0, 1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 21]  # methods with 2 arguments
SMALL = [0, 3, 5, 8, 9, 11, 12, 14, 17]        # natives with 3 arguments
TINY = [0, 5, 9, 14]                            # 4 or more arguments (methods: 3)
FUZZ_HEADER = STRUCT_P + """values()
    let inf = 1e308 * 10.0
    let nan = inf - inf
    return [%s]

""" % ", ".join(VALUES)
# not fuzzed: input (stdin), wait_key (blocks), clock_ms (time), sleep_ms with huge
# values, exit and energy_mode (see TestUnifiedBehaviour / below)
FUZZ_SKIP = {'input', 'wait_key', 'clock_ms', 'sleep_ms', 'exit', 'energy_mode'}
FUZZ_RECEIVERS = [5, 9, 11, 8, 21, 22, 0, 12, 13]   # str array map nil Some Err int struct fn


def fuzz_calls(call: str, arity: int, receiver=None):
    """TIL lines: `call` with every tuple of values; prints the tuple and the result or error."""
    if receiver is not None:
        idx = list(range(len(VALUES))) if arity <= 1 else MID if arity == 2 else TINY
    else:
        idx = list(range(len(VALUES))) if arity <= 2 else SMALL if arity == 3 else TINY
    names = 'abcdefgh'
    lines, ind = [], '    '
    for k in range(arity):
        lines.append(ind + f"for {names[k]} in [{', '.join(map(str, idx))}]")
        ind += '    '
    argv = ", ".join(f"V[{names[k]}]" for k in range(arity))
    label = " + \" \" + ".join([f"str({names[k]})" for k in range(arity)] or ['""'])
    target = f"values()[{receiver}].{call}({argv})" if receiver is not None else f"{call}({argv})"
    lines += [ind + "let r = retry(1)", ind + f"    str({target})", ind + "else", ind + "    \"E:\" + error",
              ind + f"print(\"{receiver} \" + {label} + \" | \" + r)"]
    return lines


def fuzz_program(*blocks) -> str:
    """One function per block of lines (a function holds at most 250 locals)."""
    parts = [f"part{k}(V)\n" + "\n".join(b) + "\n" for k, b in enumerate(blocks)]
    return FUZZ_HEADER + "\n".join(parts) + "\nmain()\n    let V = values()\n" + \
        "".join(f"    part{k}(V)\n" for k in range(len(blocks)))


def native_cases():
    for nid, name in enumerate(til.NATIVES):
        if name in FUZZ_SKIP:
            continue
        lo, hi = til.NATIVE_ARITY[name]
        hi = 3 if hi is None else hi
        blocks = [["    screen_init(8, 8)"]] if til.GFX_FIRST_NATIVE <= nid <= til.GFX_LAST_NATIVE else []
        for arity in range(1 if name == 'range' else lo, hi + 1):
            blocks.append(fuzz_calls(name, arity))
        yield name, blocks


@needs_cc
class TestFuzz:
    @pytest.mark.parametrize("name,blocks", list(native_cases()), ids=[n for n, _ in native_cases()])
    def test_native(self, name, blocks):
        run_both(fuzz_program(*blocks))

    @pytest.mark.parametrize("method", til.METHODS)
    def test_method(self, method):
        blocks = [fuzz_calls(method, arity, receiver=r)
                  for r in FUZZ_RECEIVERS for arity in range(0, 3 if r not in (5, 9, 11) else 4)]
        run_both(fuzz_program(*blocks))

    @pytest.mark.parametrize("op", ['-V[a]', 'V[a] + V[b]', 'V[a] - V[b]', 'V[a] * V[b]', 'V[a] / V[b]',
                                    'V[a] % V[b]', 'V[a] ** V[b]', 'V[a] < V[b]', 'V[a] == V[b]',
                                    'V[a][V[b]]', '{V[a]: V[b]}', 'V[a](1)'])
    def test_operator(self, op):
        arity = 2 if 'V[b]' in op else 1
        if op == 'V[a] * V[b]':
            skip = "if a == 9 or a == 10\n" + "        continue"     # arrays * int: not in the C VM
        else:
            skip = None
        lines = fuzz_calls('OP', arity)
        lines = [ln.replace('OP(' + ", ".join(f"V[{n}]" for n in 'ab'[:arity]) + ')', op) for ln in lines]
        if skip:
            lines.insert(1, "        " + skip.replace("\n", "\n    "))
        run_both(fuzz_program(lines))

    def test_format_specs(self):
        specs = ['', 'd', 'x', 'X', 'o', 'b', 'e', 'f', '%', '.2f', '.0f', '.f', '>8', '<8', '^9', '08', '08.3f',
                 '+5', '>100', '.40f', '.99999999999999999999f', '99999999999999999999', 'z', '5z', '.2', '.0',
                 '010x', '^0', '-8', '>08', '0', '00', '.3e', '12.4%', '>', '<<']
        # not here: nan (the C VM's printf writes "-nan" for a negative nan) and 1e300
        # with f/% (the C VM's fixed-size buffer): numeric formatting, fixed separately
        values = [str(i) for i, v in enumerate(VALUES) if v not in ('nan', '1e300')]
        body = ["    for i in [%s]" % ", ".join(values)]
        for k, spec in enumerate(specs):
            body += [f"        let r{k} = retry(1)", f"            format(V[i], \"{spec}\")", "        else",
                     "            \"E:\" + error", f"        print(str(i) + \" [{spec}] \" + r{k})"]
        run_both(fuzz_program(body))

    def test_sleep_and_energy_mode_arguments(self):
        small = [i for i, v in enumerate(VALUES) if v not in ('9223372036854775807', '4294967296', '65', '3')]
        body = ["    for a in [%s]" % ", ".join(map(str, small)),
                "        let r = retry(1)", "            str(sleep_ms(V[a]))", "        else",
                "            \"E:\" + error", "        print(str(a) + \" \" + r)",
                "        let e = retry(1)", "            str(energy_mode(V[a]))", "        else",
                "            \"E:\" + error", "        energy_mode(0)", "        print(str(a) + \" \" + e)"]
        run_both(fuzz_program(body))

    @pytest.mark.parametrize("value", VALUES)
    def test_exit_argument(self, value):
        src = FUZZ_HEADER + "main()\n    print(\"a\")\n    let r = retry(1)\n" \
            f"        exit(values()[{VALUES.index(value)}])\n        \"x\"\n" \
            "    else\n        \"E:\" + error\n    print(r)\n"
        data = tbc_bytes(src)
        rc, out, err = run_py(data)
        crc, cout, cerr = run_cvm(data)
        assert (rc, out, err) == (crc, cout, cerr)
