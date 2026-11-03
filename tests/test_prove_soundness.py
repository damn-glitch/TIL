#!/usr/bin/env python3
"""
TIL 3.1 contract proofs (src/tilprove.py): regression tests for the first
adversarial review of `til prove`.

Each class is one finding.  The pattern is always the same: a program that the
prover used to call `proved` while it fails on both VMs (or a false
counterexample, a crash, a resource blowup) - and the program is run on the
reference VM and on the C VM to show what really happens.

Run: python3 -m pytest tests/test_prove_soundness.py -v
"""

import os
import subprocess
import sys

try:
    import resource              # POSIX
except ImportError:              # pragma: no cover - Windows
    resource = None

import pytest

from tilhelpers import CC, ROOT, TIL, cli, run_both, run_cvm, run_vm, til   # puts src/ on sys.path

import tilprove  # noqa: E402

HAVE_Z3 = tilprove.load_z3()
pytestmark = pytest.mark.skipif(not HAVE_Z3, reason="z3-solver is not installed")

MAX = (1 << 63) - 1
MIN = -(1 << 63)


def prove(src, **kw):
    return {r.key: r for r in tilprove.prove_source(src, "<test>", **kw)}


def statuses(src, **kw):
    return {k: r.status for k, r in prove(src, **kw).items()}


def fails_on_both_vms(src: str, message: str):
    """The program stops with this runtime error on both VMs."""
    rc, _out = run_vm(src)
    assert rc == 1
    if CC:
        crc, _cout, cerr = run_cvm(src)
        assert crc == 1 and message in cerr, cerr


def failure(rep, status='counterexample'):
    return next(f for f in rep.failures if f.status == status)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return str(path)


def limited_run(args, timeout=240, memory_mb=3000, env=None):
    """`til ARGS` in a subprocess whose address space is limited (so that a
    regression cannot exhaust the machine)."""
    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
    full_env = dict(os.environ)
    full_env.update(env or {})
    linux = sys.platform.startswith('linux') and resource is not None
    return subprocess.run([sys.executable, TIL, *args], capture_output=True, text=True, timeout=timeout,
                          preexec_fn=limit if linux else None, env=full_env)


# ═══════════════════════════════════════════════════════════════
# 1. a callee's #[ensures] that reads a name the callee binds again
# ═══════════════════════════════════════════════════════════════

SHADOW_PARAM = '''#[requires: x >= 0]
#[ensures: result >= x]
grow(x: int) -> int
    let x = x / 2
    return x + 1

#[level: 4]
#[requires: n >= 0 and n <= 1000]
#[ensures: result >= n]
use(n: int) -> int
    return grow(n)

main()
    print(use(10))
'''


class TestShadowedCalleeEnsures:
    def test_let_shadowing_a_parameter(self, tmp_path):
        reps = prove(SHADOW_PARAM)
        assert reps['grow'].status == 'proved'          # true: in grow, `x` at the return is the local
        use = reps['use']
        assert use.status == 'counterexample', use.status
        assert failure(use).confirmed
        fails_on_both_vms(SHADOW_PARAM, "contract violation: use ensures result >= n")
        # plain `til check` proves #[level: 4] functions: it must fail too
        r = cli('check', write(tmp_path, 's.til', SHADOW_PARAM))
        assert r.returncode == 1 and "E-PROVE" in r.stderr and "'use'" in r.stderr

    @pytest.mark.parametrize("callee", [
        # a `for` variable named like the parameter
        "#[ensures: result == x]\nf(x: int) -> int\n    for x in 0..1\n        return x\n    return x\n",
        # a local shadowing a global constant
        "#[ensures: result == LIMIT]\nf(x: int) -> int\n    let LIMIT = 3\n    return 3\n",
        # shadowing in a nested block only
        "#[ensures: result == x]\nf(x: int) -> int\n    if x > 0\n        let x = 7\n        return 7\n    return x\n",
        # an assignment in a nested lambda-free block
        "#[ensures: result == x]\nf(x: int) -> int\n    if true\n        x = 7\n    return 7\n",
    ])
    def test_other_rebindings(self, callee):
        src = ("const LIMIT = 10\n" + callee +
               "#[ensures: result == 7]\ng() -> int\n    return f(7) * 0 + 7 + f(7) - f(7)\n"
               "#[ensures: result == LIMIT]\nh() -> int\n    return f(7)\n")
        reps = prove(src)
        assert reps['h'].status != 'proved', [(x.ob.what, x.status) for x in reps['h'].failures]

    def test_unshadowed_ensures_are_still_used(self):
        src = ('#[requires: x >= 0 and x <= 1000]\n#[ensures: result >= x]\ndbl(x: int) -> int\n'
               '    let y = x * 2\n    return y\n'
               '#[requires: n >= 0 and n <= 50]\n#[ensures: result >= 0]\nuse(n: int) -> int\n'
               '    return dbl(n) - n\n')
        assert statuses(src) == {'dbl': 'proved', 'use': 'proved'}


# ═══════════════════════════════════════════════════════════════
# 2. a `-> int` callee without #[ensures] can return nil
# ═══════════════════════════════════════════════════════════════

NIL_RESULT = '''lookup(key: int) -> int
    if key == 1
        return 10
    if key == 2
        return 20

#[ensures: result >= 0]
price(key: int) -> int
    let p = lookup(key)
    if p == NULL
        return -1
    return 0

main()
    print(price(3))
'''


class TestNilResults:
    def test_missing_default_case_is_found(self):
        rep = prove(NIL_RESULT)['price']
        assert rep.status == 'counterexample' and failure(rep).confirmed
        fails_on_both_vms(NIL_RESULT, "contract violation: price ensures result >= 0")

    def test_nil_check_on_the_result_is_not_proved(self):
        src = ('maybe(x: int) -> int\n    if x > 0\n        return 1\n'
               'f(x: int) -> int\n    assert maybe(x) != NULL\n    return 0\n'
               'main()\n    print(f(0))\n')
        assert prove(src)['f'].status != 'proved'
        fails_on_both_vms(src, "Assertion failed")

    @pytest.mark.parametrize("body", [
        "    if x > 0\n        return 1\n    else\n        return 2\n",
        "    loop\n        if x > 0\n            return 1\n        return 2\n",
        "    if x > 0\n        return 1\n    panic(\"no\")\n",
    ])
    def test_callees_that_always_return_still_give_ints(self, body):
        src = (f"helper(x: int) -> int\n{body}"
               "f(x: int) -> int\n    assert helper(x) != NULL\n    return 0\n")
        assert prove(src)['f'].status == 'proved'

    def test_callee_with_ensures_is_cast_to_int(self):
        # with #[ensures] the VM casts every return to int: ending without
        # `return` gives 0, not nil
        src = ('#[ensures: result >= 0]\ng(x: int) -> int\n    if x > 0\n        return x\n'
               'f(x: int) -> int\n    assert g(x) != NULL\n    return 0\n'
               'main()\n    print(g(-5))\n    print(f(-5))\n')
        assert prove(src)['f'].status == 'proved'
        assert run_both(src) == "0\n0\n"


# ═══════════════════════════════════════════════════════════════
# 3. `x ** e` with a variable exponent
# ═══════════════════════════════════════════════════════════════

POWERS = '''#[requires: x >= 0 and x <= 10 and e >= 0 and e <= 3]
#[ensures: result >= 0]
small(x: int, e: int) -> int
    return x ** e

#[requires: e >= 0]
#[ensures: result == 1]
one(e: int) -> int
    return 1 ** e

#[requires: x % 2 == 0 and e >= 64]
#[ensures: result == 0]
even(x: int, e: int) -> int
    return x ** e

#[requires: x >= 0 and x <= 10]
#[ensures: result >= 0]
huge(x: int) -> int
    return x ** 9223372036854775807

#[ensures: result >= 0]
any_power(x: int, e: int) -> int
    return x ** e
'''


class TestPowers:
    def test_variable_exponents_are_bounded_and_exact_when_small(self, tmp_path):
        r = limited_run(['prove', write(tmp_path, 'p.til', POWERS), '--timeout', '5000'], timeout=300)
        assert "small: proved" in r.stdout and "one: proved" in r.stdout and "even: proved" in r.stdout, \
            r.stdout + r.stderr
        # 3 ** (2**63 - 1) wraps to a negative number; (-1) ** 9 is -1
        assert "huge: counterexample" in r.stdout and "any_power: counterexample" in r.stdout, r.stdout
        assert "Traceback" not in r.stderr and "internal error" not in r.stdout

    def test_counterexample_values_from_the_unknown_power_are_not_shown(self):
        rep = prove(POWERS, only='huge', timeout_ms=5000)['huge']
        fail = failure(rep)
        assert fail.confirmed
        # `result` comes from the solver's choice for the unknown power: not what the VM computes
        assert 'result' not in dict(fail.extras)

    def test_the_vms_agree_with_the_counterexamples(self):
        out = run_both("main()\n    print(3 ** 9223372036854775807 >= 0)\n    print((0 - 1) ** 9)\n"
                       "    print(6 ** 64)\n")
        assert out == "false\n-1\n0\n"


# ═══════════════════════════════════════════════════════════════
# 4. `-> bool` callees that return a parameter or a shadowed name
# ═══════════════════════════════════════════════════════════════

BOOLS = '''check(flag: bool) -> bool
    if flag
        return true
    return flag

weird(b: bool) -> bool
    let b = 5
    return b

f(x: int) -> int
    let r = check(x & 4)
    assert r == true or r == false
    return 0

g() -> int
    assert weird(true) != 5
    return 0

h(b: bool) -> int
    let r = check(b)
    assert r == true or r == false
    return 0

main()
    print(h(true))
    print(f(0))
'''


class TestBoolResults:
    def test_int_argument_to_a_bool_parameter(self):
        reps = prove(BOOLS)
        assert reps['f'].status != 'proved'
        assert reps['g'].status != 'proved'
        assert reps['h'].status == 'proved'           # a bool argument: the result is a bool
        fails_on_both_vms(BOOLS, "Assertion failed")
        fails_on_both_vms(BOOLS.replace("print(f(0))", "print(g())"), "Assertion failed")


# ═══════════════════════════════════════════════════════════════
# 5. immutable globals computed by __start__
# ═══════════════════════════════════════════════════════════════

class TestGlobals:
    def test_a_global_read_before_it_is_initialized(self):
        src = ('let A = f()\nlet C = 5 + 0\n#[ensures: result == 5]\nf() -> int\n    return C\n'
               'main()\n    print(A)\n')
        assert prove(src)['f'].status != 'proved'
        fails_on_both_vms(src, "contract violation: f ensures result == 5")
        # in the other order C is set before f runs
        ok = src.replace("let A = f()\nlet C = 5 + 0\n", "let C = 5 + 0\nlet A = f()\n")
        assert prove(ok)['f'].status == 'proved'
        assert run_both(ok) == "5\n"

    def test_functions_that_do_not_run_at_startup_know_late_globals(self):
        src = ('let A = f()\nlet C = 5 + 0\nf() -> int\n    return 1\n'
               '#[ensures: result == 5]\ng() -> int\n    return C\nmain()\n    print(g())\n')
        assert prove(src)['g'].status == 'proved'          # only f runs while the globals are set
        assert run_both(src) == "5\n"

    def test_ints_and_bools_are_never_equal_at_run_time(self):
        src = ('let X = 0 + 1\nlet B = X == true\n#[ensures: result]\nf() -> bool\n    return B\n'
               'main()\n    print(f())\n')
        rep = prove(src)['f']
        assert rep.status == 'counterexample' and failure(rep).confirmed
        fails_on_both_vms(src, "contract violation: f ensures result")

    def test_a_global_read_by_an_earlier_initializer_is_nil(self):
        src = ('let A = B == NULL\nlet B = 5 + 0\n#[ensures: result]\nf() -> bool\n    return A\n'
               'main()\n    print(f())\n')
        assert prove(src)['f'].status == 'proved'          # A is true: B was still nil
        assert run_both(src) == "true\n"

    def test_computed_globals_are_still_known(self):
        src = ('let N = 10 * 1000\nlet HALF = N / 2\n#[ensures: result == 5000]\nf() -> int\n    return HALF\n')
        assert prove(src)['f'].status == 'proved'


# ═══════════════════════════════════════════════════════════════
# 6. calls in struct field defaults and in global initializers
# ═══════════════════════════════════════════════════════════════

STRUCT_DEFAULT = '''let A = 0 - 1

#[requires: x >= 0]
half(x: int) -> int
    return x / 2

struct P
    v: int = half(A)

#[ensures: result == 1]
make() -> int
    let p = P { }
    return 1

main()
    print(make())
'''

GLOBAL_CALL = '''let A = 0 - 1
let G = half(A)

#[requires: x >= 0]
half(x: int) -> int
    return x / 2

main()
    print(G)
'''


class TestDefaultsAndInitializers:
    def test_struct_field_default(self):
        rep = prove(STRUCT_DEFAULT)['make']
        assert rep.status == 'counterexample' and failure(rep).confirmed
        assert "default value of field 'v' of 'P'" in failure(rep).ob.what
        fails_on_both_vms(STRUCT_DEFAULT, "contract violation: half requires x >= 0")
        given = STRUCT_DEFAULT.replace("P { }", "P { v: 3 }")
        assert prove(given)['make'].status == 'proved'

    def test_a_struct_default_makes_a_function_a_target(self):
        src = STRUCT_DEFAULT.replace("#[ensures: result == 1]\n", "")
        assert prove(src)['make'].status == 'counterexample'

    def test_fields_are_evaluated_in_the_order_of_the_struct(self):
        # pos(x) is evaluated after need(x) at run time: its #[ensures] cannot justify need's #[requires]
        src = ('#[ensures: x > 0]\npos(x: int) -> int\n    return 1\n'
               '#[requires: x > 0]\nneed(x: int) -> int\n    return 1\n'
               'struct P\n    a: int\n    b: int\n'
               '#[ensures: result == 1]\nf(x: int) -> int\n    let p = P { b: pos(x), a: need(x) }\n    return 1\n'
               'main()\n    print(f(0))\n')
        rep = prove(src)['f']
        assert rep.status == 'counterexample' and "of 'need'" in failure(rep).ob.what
        fails_on_both_vms(src, "contract violation: need requires x > 0")

    def test_global_initializers(self, tmp_path):
        rep = prove(GLOBAL_CALL)[tilprove.START]
        assert rep.status == 'counterexample' and failure(rep).confirmed
        fails_on_both_vms(GLOBAL_CALL, "contract violation: half requires x >= 0")
        path = write(tmp_path, 'g.til', GLOBAL_CALL)
        r = cli('prove', path)
        assert r.returncode == 1 and "__start__ (global initializers): counterexample" in r.stdout
        r = cli('check', '--prove', path)
        assert r.returncode == 1 and "'__start__' (the global initializers)" in r.stderr
        ok = GLOBAL_CALL.replace("let A = 0 - 1", "let A = 0 + 4")
        assert prove(ok)[tilprove.START].status == 'proved'
        assert run_both(ok) == "2\n"

    def test_no_initializer_target_without_contract_calls(self):
        assert tilprove.START not in prove('let A = 1 + 2\n#[ensures: result == 3]\nf() -> int\n    return A\n')


# ═══════════════════════════════════════════════════════════════
# 7. script mode: the probe does not run the script
# ═══════════════════════════════════════════════════════════════

SCRIPT = '''print("start")
let K = int("5")

#[ensures: result > 0]
f(x: int) -> int
    if K == 5
        return 1
    return 0

#[ensures: result > x]
g(x: int) -> int
    print(K)
    return x + 1

print(f(1))
print(f(-7))
'''


class TestScriptMode:
    def test_no_false_counterexample(self):
        reps = prove(SCRIPT)
        assert reps['f'].status == 'not proved'
        assert "the script initializes" in failure(reps['f'], 'not proved').outcome
        assert run_both(SCRIPT) == "start\n1\n1\n"

    def test_failures_that_do_not_depend_on_the_script_are_counterexamples(self):
        rep = prove(SCRIPT)['g']
        assert rep.status == 'counterexample' and rep.counterexample == {'x': MAX}


# ═══════════════════════════════════════════════════════════════
# 8. the same file gives the same results in every run
# ═══════════════════════════════════════════════════════════════

NONDET = '''helper(a: int, b: int) -> int
    var r = 0
    if a == 9223372036854775807 and b == -1
        r = 7
    return r

#[ensures: result != 7]
f(a: int, b: int) -> int
    return helper(a, b)
'''


def test_results_do_not_depend_on_the_hash_seed(tmp_path):
    path = write(tmp_path, 'd.til', NONDET)
    outs = {cli('prove', path, env={'PYTHONHASHSEED': str(seed)}).stdout for seed in (1, 2, 3)}
    assert len(outs) == 1, outs


# ═══════════════════════════════════════════════════════════════
# 9. counterexamples for functions with parameters that are not int/bool
# ═══════════════════════════════════════════════════════════════

def test_other_parameters_get_values_and_the_run_confirms():
    src = '#[ensures: result > x]\ninc(name: str, xs: [int], x: int, r: float) -> int\n    return x + 1\n'
    rep = prove(src)['inc']
    fail = failure(rep)
    assert fail.confirmed and dict(fail.others) == {'name': "", 'xs': [], 'r': 0.0}
    assert tilprove.summary_line(rep).startswith('inc: counterexample: name = "", xs = [], '
                                                 'x = 9223372036854775807, r = 0.0')
    call = 'main()\n    print(inc("", [], 9223372036854775807, 0.0))\n'
    fails_on_both_vms(src + call, "contract violation: inc ensures result > x")


# ═══════════════════════════════════════════════════════════════
# 11. range() over values the prover does not know
# ═══════════════════════════════════════════════════════════════

def test_range_of_len():
    src = ('#[requires: n >= 0 and n <= 100]\n#[ensures: result == n + 1]\nf(n: int, xs: [int]) -> int\n'
           '    for i in range(len(xs))\n        assert i >= 0\n        print(xs[i])\n    return n + 1\n'
           'g(xs: [int]) -> int\n    var s = -1\n    for i in range(len(xs))\n        invariant s >= 0\n'
           '        s = 0\n    return s\n')
    reps = prove(src)
    assert reps['f'].status == 'proved'
    assert reps['g'].status == 'not proved'           # s = -1 when the loop is reached


def test_an_empty_range_does_not_hide_the_code_after_the_loop():
    # range(len(xs), MIN) is empty: after the loop x can be MIN
    src = ('f(x: int, xs: [int]) -> int\n    for i in range(len(xs), x)\n        print(i)\n'
           '    assert x != -9223372036854775807 - 1\n    return 0\n')
    rep = prove(src)['f']
    assert rep.status == 'counterexample' and failure(rep).confirmed
    assert rep.counterexample == {'x': MIN}
    fails_on_both_vms(src + 'main()\n    print(f(-9223372036854775807 - 1, []))\n', "Assertion failed")


# ═══════════════════════════════════════════════════════════════
# 12. the command line
# ═══════════════════════════════════════════════════════════════

class TestCommandLine:
    def test_unreadable_inputs(self, tmp_path):
        r = cli('prove', str(tmp_path))
        assert r.returncode == 1 and r.stderr.startswith("Error:") and "Traceback" not in r.stderr
        bad = tmp_path / 'bad.til'
        bad.write_bytes(b'\xff\xfe bad\n')
        r = cli('prove', str(bad))
        assert r.returncode == 1 and r.stderr.startswith("Error:") and "Traceback" not in r.stderr

    def test_help_version_and_missing_values(self, tmp_path):
        r = cli('prove', '--help')
        assert r.returncode == 0 and r.stdout.startswith("usage: til prove")
        r = cli('prove', 'x.til', '--version')
        assert r.returncode == 0 and "til prove (TIL" in r.stdout and "Z3" in r.stdout
        path = write(tmp_path, 'a.til', '#[ensures: result == 1]\nf() -> int\n    return 1\n')
        for opt in ('--timeout', '--function', '--memory'):
            r = cli('prove', path, opt)
            assert r.returncode == 2 and f"{opt} needs a value" in r.stderr
        assert cli('prove', path, '--memory', '0').returncode == 2
        assert cli('--help').stdout.startswith("\nTIL")            # til's own help is unchanged


# ═══════════════════════════════════════════════════════════════
# 13/14. running counterexamples: sandbox, memory, edge values
# ═══════════════════════════════════════════════════════════════

class TestCounterexampleRuns:
    def test_files_sleep_and_exit_are_blocked(self, tmp_path):
        target = tmp_path / 'written.txt'
        src = (f'#[ensures: result > x]\nw(x: int) -> int\n    write_file("{target}", "x")\n    return x + 1\n'
               '#[ensures: result > x]\ns(x: int) -> int\n    sleep_ms(100000)\n    return x + 1\n'
               '#[ensures: result > x]\ne(x: int) -> int\n    if x == 9223372036854775807\n        exit(3)\n'
               '    return x + 1\n')
        reps = prove(src)
        assert not target.exists()
        for key in ('w', 's'):
            fail = reps[key].failures[0]
            assert fail.confirmed is False and "is not run while checking a counterexample" in fail.outcome
        assert reps['e'].status in ('counterexample', 'not proved')     # and this process is still here

    @pytest.mark.parametrize("doubling", ["s = s + s", 's = f"{s}{s}"', "s = s.repeat(2)"])
    def test_memory_of_a_run_is_bounded(self, tmp_path, doubling):
        src = ('#[level: 4]\n#[ensures: result > x]\ninc(x: int) -> int\n    var s = "ab"\n    var i = 0\n'
               f'    while i < 40\n        {doubling}\n        i += 1\n    return x + 1\n')
        r = limited_run(['check', write(tmp_path, 'm.til', src)], timeout=120, memory_mb=2500)
        assert r.returncode == 1 and "E-PROVE" in r.stderr, r.stderr
        assert "Traceback" not in r.stderr

    def test_edge_values_find_what_the_abstraction_hides(self):
        # the solver's candidate comes from an unknown callee result (no
        # #[ensures]) and does not reproduce; the edge value MAX does
        src = ('helper(a: int) -> int\n    var r = a\n    r += 1\n    return r\n'
               '#[ensures: result != -9223372036854775807 - 1]\nf(a: int) -> int\n    return helper(a)\n')
        rep = prove(src)['f']
        fail = failure(rep)
        assert fail.confirmed and "found by running edge values" in fail.outcome
        assert rep.counterexample == {'a': MAX}
        fails_on_both_vms(src + "main()\n    print(f(9223372036854775807))\n",
                          "contract violation: f ensures result != -9223372036854775807 - 1")


def test_z3_running_out_of_memory_is_unknown(monkeypatch):
    # Z3 throws "out of memory" when it reaches memory_max_size (the prover
    # sets it: --memory, TIL_PROVE_MEMORY_MB); that is an `unknown` result
    def oom(self, *args):
        raise tilprove.z3.Z3Exception(b"out of memory")
    monkeypatch.setattr(tilprove.z3.Solver, 'check', oom)
    monkeypatch.setattr(tilprove.z3.Solver, 'add', lambda self, *args: None)
    rep = prove('#[ensures: result >= 0]\nf(x: int) -> int\n    return x * x\n')['f']
    assert rep.status == 'unknown' and "out of memory" in rep.failures[0].reason
    errors = tilprove.format_errors(rep, til.ErrorReporter("", "f.til"), 1000)
    assert "raise --memory" in errors[0]


def test_memory_option_has_a_floor(tmp_path, monkeypatch):
    path = write(tmp_path, 'a.til', '#[ensures: result == 1]\nf() -> int\n    return 1\n')
    r = cli('prove', path, '--memory', '16')            # Z3 itself crashes with so little memory
    assert r.returncode == 2 and "at least 256" in r.stderr
    monkeypatch.setenv('TIL_PROVE_MEMORY_MB', '1')
    assert tilprove.memory_limit_mb() == tilprove.MIN_MEMORY_MB


# ═══════════════════════════════════════════════════════════════
# the example runs the same natively and on both VMs
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_example_runs_natively_too(tmp_path):
    path = os.path.join(ROOT, 'examples', '13_proofs.til')
    with open(path, encoding='utf-8') as f:
        src = f.read()
    vm_out = run_both(src)
    compiler = til.TILCompiler()
    exe = str(tmp_path / 'proofs')
    assert compiler.compile_to_executable(src, exe, path)
    native = subprocess.run([exe], capture_output=True, timeout=60)
    assert native.returncode == 0 and native.stdout.decode() == vm_out


def test_a_failure_while_the_globals_are_initialized_confirms_nothing():
    # __start__ calls f(-5), which fails; the probe's own call f(0) never runs
    src = ('let A = f(-5)\nhelper(x: int) -> int\n    var r = x\n    r += 0\n    return r\n'
           '#[ensures: result > -3]\nf(x: int) -> int\n    return helper(x)\nmain()\n    print(A)\n')
    rep = prove(src)['f']
    assert rep.status == 'not proved'
    assert "while the global initializers run" in rep.failures[0].outcome
    fails_on_both_vms(src, "contract violation: f ensures result > -3")
