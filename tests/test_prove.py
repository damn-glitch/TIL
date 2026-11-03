#!/usr/bin/env python3
"""
TIL 3.1: contracts proved for ALL inputs with Z3 (src/tilprove.py,
docs/TIL_PROOFS.md) - `til prove`, `til check --prove`, #[level: 4] in
`til check`.

Every counterexample found here is run again on the reference VM and on the C
VM with the counterexample inputs, and must break the contract there exactly
as reported.  Tests that need Z3 are skipped when z3-solver is not installed;
the "no Z3" behaviour is tested with a stand-in module that fails to import.

Run: python3 -m pytest tests/test_prove.py -v
     TIL_TEST_SLOW=1 python3 -m pytest tests/test_prove.py -k fuzz   (more random cases)
"""

import os
import random

import pytest

from tilhelpers import ROOT, cli, run_both, til   # importing tilhelpers puts src/ on sys.path

import tilprove  # noqa: E402

HAVE_Z3 = tilprove.load_z3()          # False when z3-solver is missing or broken
needs_z3 = pytest.mark.skipif(not HAVE_Z3, reason="z3-solver is not installed")
SLOW = os.environ.get('TIL_TEST_SLOW') == '1'

MAX = (1 << 63) - 1
MIN = -(1 << 63)


def prove(src, **kw):
    """{function: report} for the functions in scope (explicit mode)."""
    return {r.key: r for r in tilprove.prove_source(src, "<test>", **kw)}


def lit(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v == MIN:
        return "(-9223372036854775807 - 1)"
    return str(v) if v >= 0 else f"({v})"


def run_call(src: str, func: str, args) -> str:
    """Call func(*args) on both VMs (they must agree); returns the runtime error
    message, or "" when the call succeeds."""
    call = f"{func}({', '.join(lit(v) for v in args)})"
    prog = (src + "\nmain()\n    let r = retry(1)\n        " + call +
            '\n    else\n        print("ERROR: " + error)\n        0\n    print("done")\n')
    out = run_both(prog)
    assert out.endswith("done\n"), out
    lines = [ln for ln in out.splitlines() if ln.startswith("ERROR: ")]
    return lines[0][len("ERROR: "):] if lines else ""


def assert_real(src: str, rep):
    """Each counterexample of the report breaks the contract on both VMs."""
    found = [f for f in rep.failures if f.status == 'counterexample']
    assert found, f"{rep.key}: no counterexample ({rep.status})"
    for fail in found:
        assert fail.confirmed, f"{rep.key}: {fail.ob.what}: {fail.outcome}"
        error = run_call(src, rep.key, [v for _n, v in fail.inputs])
        assert error == fail.ob.expect[0], (rep.key, fail.inputs, error, fail.ob.expect)


# ═══════════════════════════════════════════════════════════════
# PROVED FOR ALL INPUTS
# ═══════════════════════════════════════════════════════════════

PROVED = r'''
#[requires: lo <= hi]
#[ensures: lo <= result and result <= hi]
clamp(x: int, lo: int, hi: int) -> int
    if x < lo
        return lo
    elif x > hi
        return hi
    return x

#[requires: x > -9223372036854775807 - 1]
#[ensures: result >= 0 and (result == x or result == -x)]
my_abs(x: int) -> int
    if x < 0
        return -x
    return x

#[ensures: result >= a and result >= b and result >= c]
max3(a: int, b: int, c: int) -> int
    return max(a, max(b, c))

#[ensures: result == -1 or result == 0 or result == 1]
sign(x: int) -> int
    if x > 0
        return 1
    elif x == 0
        return 0
    else
        return -1

#[ensures: result == (a and not b) or (b and not a)]
xor(a: bool, b: bool) -> bool
    return a != b

#[requires: b != 0]
#[ensures: result == a / b]
quotient(a: int, b: int) -> int
    return a / b

#[ensures: result >= 0 and result < 256]
low_byte(x: int) -> int
    return x & 255

#[ensures: result == x * 8]
times8(x: int) -> int
    return x << 3

#[ensures: result == 1024]
kilo() -> int
    return 2 ** 10

square(x: int) -> int
    return x * x

#[requires: x >= -3000000000 and x <= 3000000000]
#[ensures: result >= 0]
uses_inlined(x: int) -> int
    return square(x)

#[requires: n >= 0 and n <= 1000000]
#[ensures: result >= n]
uses_contract(n: int) -> int
    return my_abs(n) + quotient(n, 1)

#[requires: lo <= hi]
#[ensures: result == 1]
calls_ok(lo: int, hi: int) -> int
    let c = clamp(5, lo, hi)
    assert lo <= c and c <= hi
    return 1
'''


@needs_z3
class TestProved:
    def test_functions_proved(self):
        reps = prove(PROVED)
        for key in ('clamp', 'my_abs', 'max3', 'sign', 'xor', 'quotient', 'low_byte', 'times8', 'kilo',
                    'uses_inlined', 'uses_contract', 'calls_ok'):
            rep = reps[key]
            assert rep.status == 'proved', (key, rep.status, rep.reason,
                                            [(f.ob.what, f.inputs, f.outcome) for f in rep.failures])
        assert 'square' not in reps            # no contract: nothing to prove

    def test_the_proved_functions_run(self):
        out = run_both(PROVED + "main()\n    print(clamp(15, 0, 10))\n    print(my_abs(-4))\n"
                       "    print(max3(1, 9, 3))\n    print(times8(-2))\n    print(uses_contract(7))\n")
        assert out == "10\n4\n9\n-16\n14\n"

    def test_globals_enums_and_static_methods(self):
        src = ('const LIMIT = 100\nlet BIG = 10 * 1000\nenum Color\n    Red\n    Green = 5\n    Blue\n'
               'struct P\n    x: int\nimpl P\n    #[requires: a >= 0 and a < LIMIT]\n    #[ensures: result == a + 1]\n'
               '    grow(a: int) -> int\n        return a + 1\n'
               '#[ensures: result == BIG + Color.Blue]\nbig() -> int\n    return 10006\n'
               '#[requires: a >= 0 and a < 50]\n#[ensures: result >= 2]\ntwice(a: int) -> int\n'
               '    return P.grow(P.grow(a))\n')
        reps = prove(src)
        assert {k: r.status for k, r in reps.items()} == {'P.grow': 'proved', 'big': 'proved', 'twice': 'proved'}

    def test_script_mode_main_is_checked(self):
        src = ('#[requires: b != 0]\ndiv(a: int, b: int) -> int\n    return a / b\n'
               'print(f"{div(10, 2)} {div(9, 3)}")\nlet k = 4\nprint(div(1, k))\n')
        reps = prove(src)
        assert reps['main'].status == 'proved' and reps['main'].n_obligations == 3

    def test_localized_keywords(self):
        kk = ('#[requires: n >= 0 және n <= 1000]\n#[ensures: result == n]\nсанау(n: int) -> int\n'
              '    айнымалы c = 0\n    кезінде c < n\n        invariant c >= 0 және c <= n\n        c += 1\n'
              '    тексеру c == n\n    қайтару c\n')
        ru = ('#[lang: ru]\n#[requires: n >= 0 и n <= 1000]\n#[ensures: result == n]\nсчёт(n: int) -> int\n'
              '    перем c = 0\n    пока c < n\n        инвариант c >= 0 и c <= n\n        c += 1\n'
              '    проверить c == n\n    вернуть c\n')
        assert prove(kk)['санау'].status == 'proved'
        assert prove(ru)['счёт'].status == 'proved'

    def test_unsatisfiable_requires_is_reported_as_vacuous(self):
        rep = prove('#[requires: x > 0 and x < 0]\n#[ensures: result == 42]\nf(x: int) -> int\n    return x\n')['f']
        assert rep.status == 'proved' and "vacuous" in rep.notes[0]


# ═══════════════════════════════════════════════════════════════
# THE INTEGER MODEL: exactly the VMs' semantics
# ═══════════════════════════════════════════════════════════════

MIN_LIT = "(-9223372036854775807 - 1)"

# facts that hold for every int in TIL (64-bit wrap-around, C-like / and %,
# shift amounts masked to 0..63) - and false variants
IDENTITIES = [
    "(a + b) - b == a",
    "b <= 0 or b > 1000 or (a / b) * b + a % b == a",
    "b == 0 or a % b == 0 or (a % b > 0) == (a > 0)",
    "(-7) / 2 == -3 and (-7) % 2 == -1 and 7 % (-2) == 1",
    f"a != {MIN_LIT} or a / (-1) == a",
    f"a != {MIN_LIT} or (abs(a) == a and -a == a)",
    "(a << 64) == a and (a << 65) == a * 2 and (a >> 64) == a",
    "(a >> 63) == 0 or (a >> 63) == -1",
    "~a == -a - 1",
    "((a & b) | (a & ~b)) == a",
    f"9223372036854775807 + 1 == {MIN_LIT}",
    "min(a, b) <= max(a, b)",
    "a ** 2 == a * a and a ** 0 == 1",
    "(a == true) == false and (a == NULL) == false",     # values of different types differ
]
FALSE_FACTS = [
    ("(a + 1) > a", {'a': MAX}),
    ("a % 2 == 0 or a % 2 == 1", None),          # -3 % 2 == -1
    ("abs(a) >= 0", {'a': MIN}),
    ("(a * 2) / 2 == a", None),
    ("(a << 1) >> 1 == a", None),
    ("-a != a or a == 0", {'a': MIN}),
]


@needs_z3
class TestIntegerModel:
    @pytest.mark.parametrize("fact", IDENTITIES)
    def test_identity_is_proved(self, fact):
        rep = prove(f"f(a: int, b: int) -> int\n    assert {fact}\n    return 0\n")['f']
        assert rep.status == 'proved', (fact, [(x.ob.what, x.inputs, x.outcome) for x in rep.failures])

    @pytest.mark.parametrize("fact, expected", FALSE_FACTS)
    def test_false_fact_has_a_real_counterexample(self, fact, expected):
        src = f"f(a: int) -> int\n    assert {fact}\n    return 0\n"
        rep = prove(src)['f']
        assert rep.status == 'counterexample'
        if expected is not None:
            assert rep.counterexample == expected
        assert_real(src, rep)

    def test_overflowing_increment(self):
        # the example from the spec: x + 1 > x is false for the largest int
        src = "#[ensures: result > x]\ninc(x: int) -> int\n    return x + 1\n"
        rep = prove(src)['inc']
        assert rep.status == 'counterexample'
        assert rep.counterexample == {'x': MAX}
        fail = rep.failures[0]
        assert fail.ob.what == "#[ensures: result > x]" and fail.ob.line == 1
        assert dict(fail.extras)['result'] == MIN
        assert_real(src, rep)


# ═══════════════════════════════════════════════════════════════
# COUNTEREXAMPLES (each one is run on both VMs)
# ═══════════════════════════════════════════════════════════════

BUGGY = r'''
#[requires: b != 0]
safe_div(a: int, b: int) -> int
    return a / b

caller(n: int) -> int
    return safe_div(100, n - 3)

#[ensures: result >= 0]
abs_diff(a: int, b: int) -> int
    if a > b
        return a - b
    return b - a

#[requires: a <= b]
#[ensures: a <= result and result <= b]
midpoint(a: int, b: int) -> int
    return (a + b) / 2

ratio(a: int, b: int) -> int
    assert a >= 0, "a must not be negative"
    return a % (b - b / 2 * 2)

#[ensures: result != 0]
pick(flag: bool, x: int) -> int
    var r = 1
    if flag
        r = x
    return r
'''


@needs_z3
class TestCounterexamples:
    def test_kinds_of_failures_and_their_lines(self):
        reps = prove(BUGGY)
        assert reps['safe_div'].status == 'proved'
        caller = reps['caller']
        assert caller.status == 'counterexample' and caller.counterexample == {'n': 3}
        fail = caller.failures[0]
        assert fail.ob.kind == 'call' and fail.ob.what == "#[requires: b != 0] of 'safe_div'"
        assert fail.ob.line == 7 and ('safe_div: b', 0) in fail.extras
        assert reps['abs_diff'].status == 'counterexample'
        assert reps['midpoint'].status == 'counterexample'
        ratio = reps['ratio']
        kinds = sorted(f.ob.kind for f in ratio.failures)
        assert kinds == ['assert', 'div'], kinds
        pick = reps['pick']
        assert pick.status == 'counterexample' and pick.counterexample == {'flag': True, 'x': 0}

    @pytest.mark.parametrize("func", ['caller', 'abs_diff', 'midpoint', 'ratio', 'pick'])
    def test_each_counterexample_is_real(self, func):
        assert_real(BUGGY, prove(BUGGY)[func])

    def test_assert_message_and_division_by_zero(self):
        rep = prove(BUGGY)['ratio']
        msgs = sorted(f.ob.expect[0] for f in rep.failures)
        assert msgs == ["Assertion failed: a must not be negative", "division by zero"]

    def test_requires_evaluation_can_divide_by_zero(self):
        src = "#[requires: a / b > 0]\nf(a: int, b: int) -> int\n    return 1\n"
        rep = prove(src)['f']
        assert rep.status == 'counterexample' and rep.counterexample['b'] == 0
        assert rep.failures[0].ob.what == "division by zero in #[requires: a / b > 0]"
        assert_real(src, rep)

    def test_wrong_loop_result(self):
        src = ('#[requires: n >= 1 and n <= 1000]\n#[ensures: result == n * (n + 1) / 2]\n'
               'sum_to(n: int) -> int\n    var s = 0\n    for i in 0..n\n        invariant s == i * (i - 1) / 2\n'
               '        s += i\n    return s\n')     # off by one: sums 0..n-1
        rep = prove(src)['sum_to']
        assert rep.status == 'counterexample'
        assert_real(src, rep)


# ═══════════════════════════════════════════════════════════════
# LOOPS AND INVARIANTS
# ═══════════════════════════════════════════════════════════════

LOOPS = r'''
#[requires: n >= 0 and n <= 1000000]
#[ensures: 2 * result == n * (n + 1)]
sum_to(n: int) -> int
    var i = 0
    var s = 0
    while i < n
        invariant 0 <= i and i <= n and 2 * s == i * (i + 1)
        i += 1
        s += i
    return s

#[requires: n >= 0]
#[ensures: result == 0]
count_down(n: int) -> int
    var k = n
    while k > 0
        invariant k >= 0
        k -= 1
    return k

#[requires: a > 0 and b > 0]
#[ensures: result > 0]
gcd(a: int, b: int) -> int
    var x = a
    var y = b
    while y != 0
        invariant x > 0 and y >= 0
        let t = x % y
        x = y
        y = t
    return x

#[requires: n >= 0 and n <= 100000]
#[ensures: result == n * n]
odd_sum(n: int) -> int
    var total = 0
    for i in 0..n
        invariant total == i * i
        total += 2 * i + 1
    return total

#[requires: n >= 1 and n <= 100]
#[ensures: result == n]
inclusive(n: int) -> int
    var c = 0
    for i in 1..=n
        invariant c == i - 1
        c += 1
    return c

#[requires: n >= 0 and n <= 100]
#[ensures: result == n]
by_range(n: int) -> int
    var c = 0
    for i in range(n)
        invariant c == i
        c += 1
    return c

#[requires: n >= 0 and n <= 1000]
#[ensures: result >= 0 and result <= n]
first_multiple_of_7(n: int) -> int
    var i = 0
    while i < n
        invariant 0 <= i and i <= n
        if i % 7 == 6
            break
        i += 1
    return i

#[requires: n >= 0 and n <= 1000]
#[ensures: result >= 0 and result <= n]
evens(n: int) -> int
    var s = 0
    for i in 0..n
        invariant s >= 0 and s <= i
        if i % 2 == 1
            continue
        s += 1
    return s

#[ensures: result == 10]
ten() -> int
    var k = 0
    loop
        invariant k >= 0 and k <= 10
        if k == 10
            break
        k += 1
    return k
'''


@needs_z3
class TestLoops:
    def test_loops_with_invariants_are_proved(self):
        reps = prove(LOOPS)
        for key in ('sum_to', 'count_down', 'gcd', 'odd_sum', 'inclusive', 'by_range', 'first_multiple_of_7',
                    'evens', 'ten'):
            assert reps[key].status == 'proved', (key, [(f.ob.what, f.outcome) for f in reps[key].failures])

    def test_loop_programs_run_on_both_vms(self):
        out = run_both(LOOPS + "main()\n    print(sum_to(100))\n    print(gcd(84, 36))\n    print(odd_sum(12))\n"
                       "    print(first_multiple_of_7(50))\n    print(evens(9))\n    print(ten())\n")
        assert out == "5050\n12\n144\n6\n5\n10\n"

    def test_invariant_that_is_not_inductive(self):
        # s >= 0 holds while the program runs, but it does not prove itself:
        # from s = MAX the next iteration wraps around
        src = ('#[requires: n >= 0 and n <= 1000]\nf(n: int) -> int\n    var s = 0\n    for i in 0..n\n'
               '        invariant s >= 0\n        s += i\n    return s\n')
        rep = prove(src)['f']
        assert rep.status == 'not proved'
        fail = rep.failures[0]
        assert fail.ob.kind == 'loop-step' and fail.confirmed is False
        assert "does not fail" in fail.outcome
        assert any(kind == 'loop' for kind, _t in fail.depends)

    def test_invariant_false_on_entry(self):
        src = ('f(n: int) -> int\n    var i = 5\n    while i < n\n        invariant i >= 6\n'
               '        i += 1\n    return i\n')
        rep = prove(src)['f']
        assert rep.status == 'counterexample'
        fail = rep.failures[0]
        assert fail.ob.kind == 'loop-entry' and fail.inputs[0][1] > 5    # the loop body runs: VM fails
        assert_real(src, rep)

    def test_loop_without_invariant_is_sound(self):
        src = ('#[ensures: result >= 0]\nf(n: int) -> int\n    var i = 0\n    while i < n\n        i += 1\n'
               '    return i\n')
        rep = prove(src)['f']
        assert rep.status == 'not proved'            # true, but unprovable without an invariant
        assert "loop invariant" in tilprove.format_errors(rep, til.ErrorReporter(src, "f.til"), 1000)[0]
        src_bad = src.replace("var i = 0", "var i = -1")
        rep = prove(src_bad)['f']
        assert rep.status == 'counterexample'      # n <= -1: the loop does not run and -1 is returned
        assert_real(src_bad, rep)


# ═══════════════════════════════════════════════════════════════
# CALLS: modular reasoning
# ═══════════════════════════════════════════════════════════════

@needs_z3
class TestCalls:
    def test_callee_ensures_are_assumed(self):
        src = ('#[requires: x >= 0 and x <= 1000]\n#[ensures: result >= x]\ndouble(x: int) -> int\n'
               '    return x * 2\n'
               '#[requires: n >= 0 and n <= 50]\n#[ensures: result >= 0]\nuse(n: int) -> int\n'
               '    return double(n) - n\n')
        assert prove(src)['use'].status == 'proved'

    def test_weak_callee_contract_is_not_a_counterexample(self):
        src = ('#[requires: a >= 0]\n#[ensures: result >= a]\ngrow(a: int) -> int\n    return a + 1\n'
               '#[ensures: result >= 1]\nf(a: int) -> int\n    if a < 0\n        return 1\n    return grow(a)\n')
        rep = prove(src)['f']
        assert rep.status == 'not proved'
        fail = rep.failures[0]
        assert fail.confirmed is False and ('contract', "the result of 'grow' (only its #[ensures] is known)") \
            in fail.depends

    def test_callee_without_contract_is_unknown(self):
        src = ('helper(x: int) -> int\n    var y = x\n    y += 1\n    return y\n'
               '#[ensures: result > x]\nf(x: int) -> int\n    return helper(x)\n')
        rep = prove(src)['f']
        assert rep.status == 'counterexample'       # helper(MAX) really wraps around
        assert_real(src, rep)
        src_ok = src.replace("y += 1", "y = 5").replace("result > x", "result == result")
        assert prove(src_ok)['f'].status == 'proved'

    def test_recursion_through_the_contract(self):
        src = ('#[requires: n >= 0 and n <= 20]\n#[ensures: result >= 1]\nfact(n: int) -> int\n    if n <= 1\n'
               '        return 1\n    return n * fact(n - 1)\n')
        rep = prove(src)['fact']
        assert rep.status in ('not proved', 'proved')       # the ensures is too weak to bound n * r
        unbounded = src.replace("n <= 20", "n <= 100")
        rep = prove(unbounded)['fact']
        assert rep.status in ('counterexample', 'not proved')
        if rep.status == 'counterexample':
            assert_real(unbounded, rep)

    def test_ensures_reading_a_reassigned_parameter_is_not_assumed(self):
        # f's #[ensures] sees x after `x = 5`: a caller must not read it as "result == argument"
        src = ('#[ensures: result == x]\nf(x: int) -> int\n    x = 5\n    return 5\n'
               '#[ensures: result == 7]\ng() -> int\n    return f(7)\n')
        reps = prove(src)
        assert reps['f'].status == 'proved'
        assert reps['g'].status == 'counterexample'          # f(7) is 5
        assert_real(src, reps['g'])


# ═══════════════════════════════════════════════════════════════
# UNSUPPORTED: never proved by accident
# ═══════════════════════════════════════════════════════════════

@needs_z3
class TestUnsupported:
    @pytest.mark.parametrize("src, reason", [
        ('#[ensures: result > 0.0]\nf(x: float) -> float\n    return x * x\n', "float"),
        ('f(s: str) -> int\n    assert s == "a"\n    return 1\n', "parameter 's' of type str"),
        ('#[ensures: result >= 0]\nf(x) -> int\n    return x\n', "no type"),
        ('#[ensures: result == 1]\nf(x: int) -> int\n    return match x\n        _ => 1\n', "'match'"),
        ('#[ensures: result == 1]\nf(x: int) -> int\n    let g = |y: int| y\n    return 1\n', "closures"),
        ('#[ensures: result >= 0]\nf(x: int) -> int\n    let r = retry(2)\n        x\n    else\n        0\n'
         '    return r\n', "retry"),
        ('var counter = 0\n#[ensures: result >= 0]\nf() -> int\n    return counter\n', "mutable global"),
        ('#[ensures: result >= 0]\nf(xs: [int]) -> int\n    return 10 / xs[0]\n', "division by"),
        ('struct P\n    x: int\nimpl P\n    #[ensures: result >= 0]\n    get(self) -> int\n        return self.x\n',
         "self"),
        ('#[requires: g(x) > 0]\ng(x: int) -> int\n    return x\n', "calls 'g' itself"),
    ])
    def test_unsupported(self, src, reason):
        reps = prove(src)
        rep = next(iter(reps.values()))
        assert rep.status == 'unsupported', (rep.status, rep.reason)
        assert reason in rep.reason, rep.reason

    def test_internal_errors_are_reported_not_raised(self, monkeypatch):
        def boom(self):
            raise RuntimeError("boom")
        monkeypatch.setattr(tilprove._Encoder, 'encode', boom)
        rep = prove('#[ensures: result == 1]\nf() -> int\n    return 1\n')['f']
        assert rep.status == 'unsupported' and "internal error in the prover (RuntimeError: boom)" in rep.reason

    def test_unmodeled_values_are_fine_when_contracts_do_not_use_them(self):
        src = ('#[requires: b != 0]\ndiv(a: int, b: int) -> int\n    return a / b\n'
               '#[ensures: result == 3]\nf(name: str, xs: [int]) -> int\n    print(f"hello {name} {xs}")\n'
               '    if name == "x"\n        print("x")\n    return div(9, 3)\n')
        assert prove(src)['f'].status == 'not proved'     # div's result is unknown without #[ensures]
        assert prove(src.replace("return a / b", "return a / b").replace(
            "#[requires: b != 0]", "#[requires: b != 0]\n#[ensures: result == a / b]"))['f'].status == 'proved'


# ═══════════════════════════════════════════════════════════════
# THE til COMMAND LINE
# ═══════════════════════════════════════════════════════════════

def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return str(path)


LEVEL4_BAD = ('#[level: 4]\n#[ensures: result > x]\ninc(x: int) -> int\n    return x + 1\n'
              '#[ensures: result >= 0]\nloose(x: int) -> int\n    return x\nmain()\n    print(inc(1))\n')
LEVEL4_GOOD = ('#[level: 4]\n#[requires: x < 100]\n#[ensures: result > x]\ninc(x: int) -> int\n    return x + 1\n'
               'main()\n    print(inc(1))\n')


class Run:
    """Result of an in-process command (faster than starting Python)."""

    def __init__(self, code, out, err):
        self.returncode, self.stdout, self.stderr = code, out, err


@pytest.fixture
def prove_cli(capsys):
    """`til prove ARGS` in-process (til.py dispatches straight to tilprove.main)."""
    def run(*args):
        code = tilprove.main(list(args))
        out = capsys.readouterr()
        return Run(code, out.out, out.err)
    return run


@pytest.fixture
def check_cli(capsys, monkeypatch):
    """`til check ARGS` in-process, through til.main()."""
    def run(*args):
        monkeypatch.setattr(til.sys, 'argv', ['til', 'check', *args])
        code = til.main()
        out = capsys.readouterr()
        return Run(code, out.out, out.err)
    return run


@needs_z3
class TestProveCommand:
    def test_til_prove_dispatch_everything_proved_exits_0(self):
        r = cli('prove', os.path.join(ROOT, 'examples', '13_proofs.til'))      # a real `til prove` process
        assert r.returncode == 0, r.stdout + r.stderr
        assert "Result: 8 proved" in r.stdout and "gcd: proved" in r.stdout

    def test_counterexample_exits_1_with_e_prove(self, tmp_path, prove_cli):
        r = prove_cli(write(tmp_path, 'a.til', LEVEL4_BAD))
        assert r.returncode == 1
        assert "inc: counterexample: x = 9223372036854775807 (#[ensures: result > x] fails, line 2)" in r.stdout
        assert "loose: counterexample: x = -" in r.stdout
        assert "error[E-PROVE]: 'inc': #[ensures: result > x] fails for x = 9223372036854775807" in r.stderr
        assert "a.til:2:1" in r.stderr and "result = -9223372036854775808" in r.stderr

    def test_function_option(self, tmp_path, prove_cli):
        path = write(tmp_path, 'a.til', LEVEL4_BAD)
        r = prove_cli(path, '--function', 'main')
        assert r.returncode == 0 and "main: proved" in r.stdout and "inc:" not in r.stdout
        r = prove_cli(path, '--function=nope')
        assert r.returncode == 1 and "no function named 'nope'" in r.stderr

    def test_allow_unsupported(self, tmp_path, prove_cli):
        path = write(tmp_path, 'u.til', '#[ensures: result > 0.0]\nf(x: float) -> float\n    return x * x\n'
                                        '#[ensures: result == 2]\ng() -> int\n    return 2\n')
        r = prove_cli(path)
        assert r.returncode == 1 and "f: unsupported:" in r.stdout and "g: proved" in r.stdout
        assert prove_cli(path, '--allow-unsupported').returncode == 0

    def test_timeout_gives_unknown(self, tmp_path, prove_cli):
        path = write(tmp_path, 'h.til', '#[requires: b != 0]\n#[ensures: result * b + a % b == a]\n'
                                        'divmod(a: int, b: int) -> int\n    return a / b\n')
        r = prove_cli(path, '--timeout', '1')
        assert r.returncode == 1 and "divmod: unknown (timeout)" in r.stdout
        assert "could not prove #[ensures: result * b + a % b == a]" in r.stderr

    def test_usage_errors(self, tmp_path, prove_cli):
        path = write(tmp_path, 'a.til', LEVEL4_GOOD)
        assert prove_cli(path, '--timeout', 'soon').returncode == 2
        assert prove_cli(path, '--bogus').returncode == 2
        assert prove_cli().returncode == 2
        assert prove_cli(str(tmp_path / 'missing.til')).returncode == 1
        r = prove_cli('--help')           # (`til prove --help` shows til's own help)
        assert r.returncode == 0 and "--allow-unsupported" in r.stdout

    def test_nothing_to_prove(self, tmp_path, prove_cli):
        r = prove_cli(write(tmp_path, 'n.til', 'main()\n    print(1)\n'))
        assert r.returncode == 0 and "Nothing to prove" in r.stdout

    def test_invalid_programs_are_rejected(self, tmp_path, prove_cli):
        assert prove_cli(write(tmp_path, 's.til', 'f(x: int) -> int\n    return (x\n')).returncode == 1
        r = prove_cli(write(tmp_path, 'e.til', '#[pure]\nf()\n    print(1)\nmain()\n    f()\n'))
        assert r.returncode == 1 and "E-EFFECT" in r.stderr

    def test_verbose_lists_conditions(self, prove_cli):
        r = prove_cli('-v', os.path.join(ROOT, 'examples', '13_proofs.til'), '--function', 'sum_to')
        assert r.returncode == 0
        assert "loop invariant `0 <= i and i <= n and 2 * s == i * (i + 1)` after an iteration: proved" in r.stdout


@needs_z3
class TestCheckCommand:
    def test_til_check_process_proves_level_4(self, tmp_path):
        r = cli('check', write(tmp_path, 'bad.til', LEVEL4_BAD))     # a real `til check` process
        assert r.returncode == 1 and "E-PROVE" in r.stderr and "'inc'" in r.stderr

    def test_plain_check_proves_level_4_only(self, tmp_path, check_cli):
        r = check_cli(write(tmp_path, 'bad.til', LEVEL4_BAD))
        assert r.returncode == 1 and "E-PROVE" in r.stderr and "'inc'" in r.stderr
        assert "'loose'" not in r.stderr                      # not level 4: not proved by plain check
        good = write(tmp_path, 'good.til', LEVEL4_GOOD + '#[ensures: result >= 0]\nloose(x: int) -> int\n'
                                                         '    return x\n')
        r = check_cli(good)
        assert r.returncode == 0 and "OK:" in r.stdout

    def test_check_prove_covers_every_contract(self, tmp_path, check_cli):
        r = check_cli('--prove', write(tmp_path, 'bad.til', LEVEL4_BAD))
        assert r.returncode == 1 and "'inc'" in r.stderr and "'loose'" in r.stderr
        r = check_cli('--prove', write(tmp_path, 'good.til', LEVEL4_GOOD))
        assert r.returncode == 0 and "proved: inc, main" in r.stdout

    def test_check_without_level_4_is_unchanged(self, tmp_path, check_cli):
        r = check_cli(write(tmp_path, 'p.til', '#[ensures: result > x]\ninc(x: int) -> int\n'
                                               '    return x + 1\nmain()\n    print(inc(1))\n'))
        assert r.returncode == 0

    def test_check_program_api(self):
        prog = til.resolve_imports(til.parse_source(LEVEL4_BAD, "x.til"), "x.til")
        errors = tilprove.check_program(prog, LEVEL4_BAD, "x.til", False)
        assert len(errors) == 1 and errors[0].startswith("error[E-PROVE]")
        assert tilprove.check_program(prog, LEVEL4_BAD, "x.til", True)[1].count("E-PROVE") == 1


class TestWithoutZ3:
    """z3 is replaced by a module that fails to import (works with or without
    z3-solver installed)."""

    @pytest.fixture
    def no_z3_env(self, tmp_path):
        stub = tmp_path / 'stub'
        stub.mkdir()
        (stub / 'z3.py').write_text('raise ImportError("No module named z3")\n')
        return {'PYTHONPATH': str(stub)}

    def test_prove_explains_how_to_install(self, tmp_path, no_z3_env):
        r = cli('prove', write(tmp_path, 'a.til', LEVEL4_GOOD), env=no_z3_env)
        assert r.returncode == 1 and "pip install z3-solver" in r.stderr

    def test_check_prove_fails(self, tmp_path, no_z3_env):
        r = cli('check', '--prove', write(tmp_path, 'a.til', LEVEL4_GOOD), env=no_z3_env)
        assert r.returncode == 1 and "error[E-PROVE]" in r.stderr and "pip install z3-solver" in r.stderr

    def test_plain_check_warns_but_succeeds(self, tmp_path, no_z3_env):
        r = cli('check', write(tmp_path, 'a.til', LEVEL4_BAD), env=no_z3_env)
        assert r.returncode == 0 and "OK:" in r.stdout
        assert "warning:" in r.stderr and "pip install z3-solver" in r.stderr

    def test_plain_check_without_level_4_is_silent(self, tmp_path, no_z3_env):
        r = cli('check', write(tmp_path, 'a.til', 'main()\n    print(1)\n'), env=no_z3_env)
        assert r.returncode == 0 and r.stderr == ""


# ═══════════════════════════════════════════════════════════════
# THE EXAMPLE
# ═══════════════════════════════════════════════════════════════

EXAMPLE = os.path.join(ROOT, 'examples', '13_proofs.til')


def test_example_runs_identically_on_both_vms():
    with open(EXAMPLE, encoding='utf-8') as f:
        src = f.read()
    out = run_both(src)
    assert out.startswith("clamp(15, 0, 10) = 10\ndivide(17, 5) = 3\n")
    assert "sum_to(100) = 5050\ngcd(84, 36) = 12\n" in out


@needs_z3
def test_example_is_proved_by_check(check_cli):
    r = check_cli(EXAMPLE)             # sum_to and gcd are #[level: 4]
    assert r.returncode == 0, r.stderr


# ═══════════════════════════════════════════════════════════════
# RANDOM PROGRAMS: proved => never fails; counterexample => really fails
# ═══════════════════════════════════════════════════════════════

EDGE = [0, 1, -1, 2, 7, 63, 64, MAX, MIN, 1 << 62, -5, 1000]


def _expr(r, names, depth=0):
    if depth >= 2 or r.random() < 0.4:
        return r.choice(names) if r.random() < 0.7 else lit(r.choice(EDGE))
    op = r.choice(['+', '-', '*', '&', '^', '<<', '>>', '|'])
    if r.random() < 0.8:
        return f"({_expr(r, names, depth + 1)} {op} {_expr(r, names, depth + 1)})"
    return f"{r.choice(['abs', '-'])}({_expr(r, names, depth + 1)})"


def _cond(r, names):
    return f"({_expr(r, names, 1)} {r.choice(['<', '<=', '>', '>=', '==', '!='])} {_expr(r, names, 1)})"


def _random_function(r, idx, mutate):
    names = [f"p{i}" for i in range(r.randint(1, 3))]
    head = f"f{idx}({', '.join(n + ': int' for n in names)}) -> int"
    c, e1, e2, e3 = _cond(r, names), _expr(r, names), _expr(r, names), _expr(r, names)
    if r.random() < 0.5:
        body = [f"    var v = {e3}", f"    if {c}", f"        v = {e1}", "    else", f"        v = {e2}"]
        fact = f"v == ({e1} if {c} else {e2})" if not mutate else f"v == ({e2} if {c} else {e1})"
        return names, "\n".join([head] + body + [f"    assert {fact}", "    return v"]) + "\n"
    ens = f"result == ({e1} if {c} else {e2})" if not mutate else f"result == ({e1} if not {c} else {e2})"
    return names, "\n".join([f"#[ensures: {ens}]", head, f"    if {c}", f"        return {e1}",
                             f"    return {e2}"]) + "\n"


@needs_z3
@pytest.mark.parametrize("seed", [1, 2] + ([3, 4, 5, 6, 7, 8] if SLOW else []))
def test_fuzz_proofs_agree_with_the_vm(seed):
    r = random.Random(seed)
    for idx in range(12 if not SLOW else 40):
        mutate = r.random() < 0.4
        names, src = _random_function(r, idx, mutate)
        rep = prove(src, timeout_ms=4000)[f"f{idx}"]
        if not mutate:
            assert rep.status in ('proved', 'unknown'), (src, rep.status, [f.inputs for f in rep.failures])
        if rep.status == 'proved':
            for _ in range(4):
                args = [r.choice(EDGE) for _n in names]
                assert run_call(src, f"f{idx}", args) == "", (src, args)
        elif rep.status == 'counterexample':
            assert_real(src, rep)
