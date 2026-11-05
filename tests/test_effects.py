#!/usr/bin/env python3
"""
TIL 3.1 - effect handlers with resumption (docs/TIL_EFFECTS.md).

    handle
        <body>              perform E(args) anywhere below (even in callees)
    with
        E(params) => ...    resume(v) continues the body; no resume aborts it

Every program runs on the reference VM and the C VM (several build profiles)
and must produce identical output, exit codes and step counts.

Run: python3 -m pytest tests/test_effects.py -v
"""

import os
import re
import subprocess

import pytest

from tilhelpers import (CC, ROOT, cli, compile_vm, cvm_binary, expect_compile_error, needs_cc, run_both,
                        run_cvm, run_vm, tbc_bytes, til)

SPEC_EXAMPLE = '''
let r = handle
    let name = perform Ask("name?")
    perform Log("hello " + name)
    "done"
with
    Ask(question) => resume("Aigerim")
    Log(msg) =>
        print(msg)
        resume(None)
    Fail(reason) => "failed: " + reason        # no resume: aborts the handle
print(r)
let f = handle
    perform Log("before")
    perform Fail("no network")
    "unreachable"
with
    Log(msg) =>
        print(msg)
        resume(None)
    Fail(reason) => "failed: " + reason
print(f)
'''


def both(src, expect_rc=0, profile="host"):
    return run_both(src, profile=profile, expect_rc=expect_rc)


def write(tmp_path, src, name="prog.til"):
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return str(p)


# ═══════════════════════════════════════════════════════════════
# SEMANTICS (both VMs, identical output)
# ═══════════════════════════════════════════════════════════════

class TestBasics:
    def test_spec_example(self):
        assert both(SPEC_EXAMPLE) == "hello Aigerim\ndone\nbefore\nfailed: no network\n"

    def test_value_of_handle_is_the_body_value(self):
        src = '''
let a = handle
    1 + perform Get()
with
    Get() => resume(41)
print(a)
let b = handle
    let x = 5
with
    Get() => resume(0)
print(b)
compute() -> int
    return handle
        perform Get() * 2
    with
        Get() => resume(21)
print(compute())
handle
    print(perform Get())
with
    Get() => resume("statement form")
'''
        assert both(src) == "42\nNone\n42\nstatement form\n"

    def test_resume_without_value_and_parameterless_clauses(self):
        src = '''
let a = handle
    perform Ping()
with
    Ping() => resume()
print(a)
let b = handle
    [perform Tick, perform Tick(), perform Tock]
with
    Tick => resume(1)
    Tock() => resume
print(b)
'''
        assert both(src) == "None\n[1, 1, None]\n"

    def test_abort_without_resume_discards_the_frames_in_between(self):
        src = '''
search(xs) -> int
    for x in xs
        if x > 10
            perform Found(x)
    return -1

outer(xs) -> int
    return search(xs) + 1000

let r = handle
    outer([1, 5, 42, 7])
with
    Found(v) => v * 100
print(r)
let none = handle
    outer([1, 2])
with
    Found(v) => v
print(none)
var log = []
let s = handle
    log.push("start")
    perform Stop()
    log.push("never")
    "never"
with
    Stop() =>
        log.push("clause")
        "stopped"
print(s)
print(log)
'''
        assert both(src) == "4200\n999\nstopped\n[\"start\", \"clause\"]\n"

    def test_clause_without_resume_and_without_value_gives_none(self):
        src = '''
let r = handle
    perform Log("x")
    "body"
with
    Log(m) =>
        if m == ""
            resume()
print(r)
'''
        assert both(src) == "None\n"

    def test_deep_handlers_stay_installed_after_resume(self):
        src = '''
var n = 0
let r = handle
    for i in 0..5
        n += perform Next()
    n
with
    Next() => resume(10)
print(r)
'''
        assert both(src) == "50\n"

    def test_state_effect_with_captured_variables(self):
        src = '''
counter() -> int
    var n = 0
    return handle
        perform Inc()
        perform Inc()
        perform Inc()
        perform Get()
    with
        Inc() =>
            n += 1
            resume()
        Get => resume(n * 10)
print(counter())
var out = []
handle
    for i in 0..5
        perform Yield(i * i)
with
    Yield(v) =>
        out.push(v)
        resume()
print(out)
'''
        assert both(src) == "30\n[0, 1, 4, 9, 16]\n"

    def test_typed_clause_parameters_are_converted(self):
        src = '''
let r = handle
    perform Half(7)
with
    Half(x: float) => resume(x / 2)
print(r)
'''
        assert both(src) == "3.5\n"

    def test_inline_forms(self):
        src = '''
compute() -> int
    return perform Ask() * 2
print(handle compute() with Ask() => resume(21))
let x = handle compute()
with
    Ask => resume(5)
print(x)
'''
        assert both(src) == "42\n10\n"


class TestBlockExpressionsEndTheirStatement:
    """A statement starting with `if`, `[` or `-` after an expression that ends with
    an indented block (handle / match / retry / timeout) is a new statement - it
    used to be glued to the expression (`x if c else y`, indexing, minus)."""

    def test_if_index_and_minus_after_block_expressions(self):
        src = '''
let h = handle
    perform A()
with
    A() => resume(5)
if h == 5
    print("handle ok")
let x = 1
let v = match x
    1 => "one"
    _ => "other"
if true
    print(v)
let r = retry(2) 7
if r == 7
    print("retry inline ok")
let t = timeout(100)
    8
[1, 2].each(|i| print(i + t))
let w = handle
    perform A()
with
    A() => resume(1)
-1 |> print
'''
        assert both(src) == "handle ok\none\nretry inline ok\n9\n10\n-1\n"


class TestDynamicScope:
    def test_performs_from_nested_calls_closures_methods_and_prelude(self):
        src = '''
struct Acc
    total: int

impl Acc
    add(self, x: int)
        self.total += perform Scale(x)

level3(x: int) -> int
    return perform Scale(x) + 1
level2(x: int) -> int
    return level3(x) * 2
level1(x: int) -> int
    return level2(x) + 3

let acc = Acc { total: 0 }
let res = handle
    acc.add(1)
    acc.add(2)
    let f = |v| perform Scale(v) - 1
    [level1(1), f(5), acc.total, [1, 2, 3].map(|v| perform Scale(v))]
with
    Scale(v) => resume(v * 1000)
print(res)
'''
        assert both(src) == "[2005, 4999, 3000, [1000, 2000, 3000]]\n"

    def test_methods_with_handlers_capture_self(self):
        src = '''
struct Box
    v: int

impl Box
    twice(self) -> int
        return handle
            perform Get() + perform Get()
        with
            Get() => resume(self.v)

print(Box { v: 4 }.twice())
'''
        assert both(src) == "8\n"

    def test_abort_from_inside_a_prelude_callback(self):
        src = '''
check(x) -> int
    if x > 10
        perform Stop(x)
    return x

let big = handle
    [1, 50, 3].each(|x| check(x))
    -1
with
    Stop(v) => v
print(big)
'''
        assert both(src) == "50\n"

    def test_innermost_handler_wins(self):
        src = '''
emit() -> str
    return perform Who()
let r = handle
    let a = emit()
    let b = handle
        emit()
    with
        Who() => resume("inner")
    a + "/" + b + "/" + emit()
with
    Who() => resume("outer")
print(r)
'''
        assert both(src) == "outer/inner/outer\n"


class TestNestingAndMasking:
    def test_same_effect_inside_a_clause_goes_to_the_outer_handler(self):
        src = '''
let m = handle
    handle
        perform Log("a")
        "inner done"
    with
        Log(msg) =>
            perform Log("inner saw " + msg)
            resume()
with
    Log(msg) =>
        print("outer: " + msg)
        resume()
print(m)
'''
        assert both(src) == "outer: inner saw a\ninner done\n"

    def test_the_whole_owner_and_handlers_inside_its_body_are_masked(self):
        # While A's clause runs, the owning handle (all of its clauses) and every
        # handle installed inside its body are invisible.
        src = '''
let r = handle
    handle
        handle
            perform A()
        with
            B() => resume("B from inside the owner's body")
    with
        A() => resume(perform B())
        B() => resume("B from the owner itself")
with
    B() => resume("B from outside the owner")
print(r)
let s = retry(1)
    handle
        handle
            perform A()
        with
            B() => resume("inner")
    with
        A() => resume(perform B())
        B() => resume("owner")
else
    error
print(s)
'''
        assert both(src) == "B from outside the owner\nunhandled effect 'B'\n"

    def test_handler_installed_inside_a_clause_is_visible(self):
        src = '''
let r = handle
    perform A()
with
    A() =>
        let v = handle
            perform A()
        with
            A() => resume(5)
        resume(v + 1)
print(r)
'''
        assert both(src) == "6\n"

    def test_masked_effect_falls_back_to_the_default_function(self):
        src = '''
A() -> int
    return 100
let r = handle
    perform A()
with
    A() => resume(perform A() + 1)
print(r)
'''
        assert both(src) == "101\n"

    def test_after_resume_the_handler_stack_is_as_before_the_perform(self):
        src = '''
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
'''
        assert both(src) == "1 unhandled effect 'Inner'\n"

    def test_clause_aborting_an_outer_handle_unwinds_everything(self):
        src = '''
let r = handle
    let x = handle
        perform Inner()
        "inner body"
    with
        Inner() =>
            perform Quit("from inner clause")
            resume()
    "outer body " + x
with
    Quit(m) => "quit: " + m
print(r)
'''
        assert both(src) == "quit: from inner clause\n"

    def test_deep_recursive_handlers(self):
        src = '''
deep(n: int) -> int
    if n == 0
        return perform Leaf()
    return handle
        deep(n - 1) + 1
    with
        Other() => resume(0)

let a = handle
    deep(10)
with
    Leaf() => resume(100)
print(a)
'''
        assert both(src) == "110\n"


class TestControlFlow:
    def test_break_continue_return_and_question_mark_leave_the_handle(self):
        src = '''
var s = 0
for i in 0..10
    handle
        if i == 1
            continue
        if i == 4
            break
        s += perform Val(i)
    with
        Val(x) => resume(x * 2)
print(s)

f() -> int
    handle
        return perform Ask() + 1
    with
        Ask() => resume(41)
print(f())

maybe(x) -> Option<int>
    if x > 0
        return Some(x)
    return None

h(x) -> Option<int>
    let r = handle
        let v = maybe(x)?
        Some(v + perform Ask())
    with
        Ask() => resume(100)
    return r
print(h(1))
print(h(0))
let q = retry(1)
    perform Ask()
else
    error
print(q)
'''
        assert both(src) == "10\n42\nSome(101)\nNone\nunhandled effect 'Ask'\n"

    def test_resume_inside_a_loop_in_the_clause(self):
        src = '''
let r = handle
    perform Find(7)
with
    Find(target) =>
        for i in 0..100
            if i * i > target
                resume(i)
        resume(-1)
print(r)
'''
        assert both(src) == "3\n"


class TestRetryTimeout:
    def test_abort_out_of_a_retry_body(self):
        src = '''
var attempts = 0
let r1 = handle
    retry(3)
        attempts += 1
        perform Fail("x")
        "never"
    else
        "fallback"
with
    Fail(m) => "aborted: " + m
print(r1)
print(attempts)
let after = retry(1)
    panic("later")
else
    "caught " + error
print(after)
'''
        assert both(src) == "aborted: x\n1\ncaught later\n"

    def test_panic_inside_a_clause_is_caught_around_the_perform(self):
        src = '''
var tries = 0
let r2 = handle
    retry(3)
        tries += 1
        perform Boom()
    else
        "gave up: " + error
with
    Boom() => panic("bad")
print(r2)
print(tries)
let r3 = retry(1)
    handle
        perform Boom()
    with
        Boom() => panic("escapes the handle")
else
    "outside: " + error
print(r3)
let next = retry(1)
    perform Boom()
else
    error
print(next)
'''
        assert both(src) == "gave up: bad\n3\noutside: escapes the handle\nunhandled effect 'Boom'\n"

    def test_retry_and_resume_inside_clauses(self):
        src = '''
var k = 0
let v = handle
    perform Get() + 1
with
    Get() =>
        let x = retry(5)
            k += 1
            assert k >= 3, "warming"
            k
        resume(x * 10)
print(v)
let w = handle
    perform Q()
with
    Q() =>
        retry(2)
            resume(7)
print(w)
let e = retry(1)
    panic("z")
else
    error
print(e)
'''
        assert both(src) == "31\n7\nz\n"

    def test_timeout_around_a_handled_body(self):
        src = '''
let t = timeout(30)
    handle
        loop
            perform Tick()
    with
        Tick() => resume()
else
    "stopped: " + error
print(t)
let u = retry(1)
    perform Tick()
else
    error
print(u)
'''
        assert both(src) == "stopped: timeout\nunhandled effect 'Tick'\n"

    def test_abort_out_of_a_timeout_drops_its_deadline(self):
        src = '''
let r = handle
    timeout(1000)
        perform Stop(7)
with
    Stop(v) => v * 6
print(r)
var s = 0
for i in 0..20000
    s += i
print(s)
'''
        assert both(src) == "42\n199990000\n"


class TestDefaultsAndErrors:
    def test_default_top_level_function_keeps_v3_programs_working(self):
        src = '''
Greet(name: str) -> str
    return "hi " + name
Log(msg: str)
    print("[log] " + msg)
main()
    perform Log("plain call")
    print(perform Greet("Aru"))
    let g = handle
        perform Greet("Aru")
    with
        Greet(n) => resume("hello " + n)
    print(g)
'''
        assert both(src) == "[log] plain call\nhi Aru\nhello Aru\n"

    def test_unhandled_effect_is_catchable(self):
        src = '''
let u = retry(1)
    perform Nope(1)
else
    error
print(u)
'''
        assert both(src) == "unhandled effect 'Nope'\n"

    def test_uncaught_unhandled_effect_both_vms(self, tmp_path):
        src = 'print("before")\nperform Missing(1, 2)\nprint("after")\n'
        r = cli("run", "--vm", write(tmp_path, src))
        assert r.returncode == 1 and r.stdout == "before\n"
        assert r.stderr.startswith("runtime error: unhandled effect 'Missing'\n")
        if CC:
            crc, cout, cerr = run_cvm(src)
            assert (crc, cout) == (1, "before\n")
            assert cerr.splitlines()[0] == "runtime error: unhandled effect 'Missing'"

    def test_argument_count_mismatch_at_run_time(self):
        src = '''
call_it() -> int
    return perform Two(1)
let a = retry(1)
    handle
        call_it()
    with
        Two(x, y) => resume(x + y)
else
    error
print(a)
'''
        assert both(src) == "effect 'Two' expects 2 argument(s), got 1\n"

    def test_too_many_nested_handlers(self):
        src = '''
deep(n: int) -> int
    if n == 0
        return perform Leaf()
    return handle
        deep(n - 1) + 1
    with
        Other() => resume(0)

let b = retry(1)
    handle
        deep(40)
    with
        Leaf() => resume(100)
else
    error
print(b)
print(handle deep(30) with Leaf() => resume(0))
'''
        assert til.TILVM.MAX_HANDLERS == 32
        assert both(src) == "too many nested handlers\n30\n"

    def test_long_effect_names_are_cut_identically(self):
        name = "E" + "x" * 80
        src = f'let r = retry(1)\n    perform {name}()\nelse\n    error\nprint(r)\n'
        assert both(src) == f"unhandled effect '{name[:64]}'\n"


# ═══════════════════════════════════════════════════════════════
# COMPILE-TIME CHECKS
# ═══════════════════════════════════════════════════════════════

class TestCompileErrors:
    def test_duplicate_clause(self):
        expect_compile_error('let r = handle\n    1\nwith\n    A() => 1\n    A(x) => 2\n',
                             "duplicate clause for effect 'A'")

    def test_resume_outside_a_clause(self):
        expect_compile_error('resume(1)\n', "resume(...) outside an effect clause")
        expect_compile_error('let r = handle\n    resume(1)\nwith\n    A() => 1\n',
                             "resume(...) outside an effect clause")

    def test_resume_in_a_lambda_inside_a_clause(self):
        expect_compile_error('let r = handle\n    perform A()\nwith\n    A() => resume([1].map(|x| resume(x)))\n',
                             "resume(...) outside an effect clause")

    def test_code_after_resume(self):
        expect_compile_error('let r = handle\n    perform A()\nwith\n    A() =>\n        resume(1)\n'
                             '        print("x")\n', "unreachable code after resume(...)", "6:9")

    def test_return_and_break_in_a_clause(self):
        expect_compile_error('f()\n    let r = handle\n        perform A()\n    with\n        A() =>\n'
                             '            return 1\n', "'return' is not allowed in an effect clause")
        expect_compile_error('for i in 0..3\n    let r = handle\n        perform A()\n    with\n'
                             '        A() =>\n            break\n', "'break' cannot leave an effect clause")

    def test_static_argument_count(self):
        expect_compile_error('let r = handle\n    perform A(1, 2)\nwith\n    A(x) => resume(x)\n',
                             "effect 'A' expects 1 argument(s), got 2")
        # a clause performing A sees the outer handle's clause, not its own
        expect_compile_error('let r = handle\n    handle\n        perform A(1)\n    with\n'
                             '        A(x) => resume(perform A(x))\nwith\n    A(x, y) => resume(x)\n',
                             "effect 'A' expects 2 argument(s), got 1")
        # inside a lambda the handler is only known at run time: no error
        compile_vm('let r = handle\n    let f = |v| perform A(v, v)\n    f(1)\nwith\n    A(x) => resume(x)\n')

    def test_syntax_errors(self):
        expect_compile_error('let r = handle\n    perform A()\nprint(r)\n', "Expected 'with'")
        expect_compile_error('let r = handle\n    perform A(1, 2)\nwith\n    A(x, x) => resume(x)\n',
                             "duplicate parameter 'x'")
        expect_compile_error('let r = handle\n    perform A()\nwith\n    A() => resume(1, 2)\n',
                             "resume takes at most one value")

    def test_with_and_resume_are_contextual_keywords(self):
        # (review round 1: hard keywords broke TIL 3.0 programs; outside a clause
        # `with` / `resume` are ordinary names again - more in test_effects_review.py)
        assert both('let with = 1\nlet resume = with + 1\nprint(resume)\n') == "2\n"
        expect_compile_error('let r = handle\n    perform A()\nwith\n    A() =>\n        let resume = 1\n'
                             '        resume(resume)\n', "'resume' is reserved inside an effect clause")


class TestEffectChecker:
    def test_handled_effects_are_not_effects_of_the_function(self):
        src = '''
#[pure]
compute() -> int
    return handle
        perform Ask() + 1
    with
        Ask() => resume(41)

#[effects: Log]
f()
    perform Log("x")

#[pure]
g() -> int
    return handle
        f()
        7
    with
        Log(m) => resume()

#[pure]
h() -> int
    return handle
        [1, 2].map(|x| perform Twice(x)).sum()
    with
        Twice(x) => resume(x * 2)

main()
    print(compute())
    print(g())
    print(h())
'''
        assert both(src) == "42\n7\n6\n"

    def test_effects_of_clauses_and_unhandled_effects_count(self):
        io_in_clause = ('#[pure]\nbad() -> int\n    return handle\n        perform Ask()\n    with\n'
                        '        Ask() =>\n            print("asked")\n            resume(1)\n')
        expect_compile_error(io_in_clause, "E-EFFECT", "pure function 'bad' performs effect 'io'")
        other = '#[pure]\nleak() -> int\n    return handle\n        perform Other()\n    with\n        Ask() => resume(1)\n'
        expect_compile_error(other, "E-EFFECT", "effect 'Other'")
        masked = ('#[pure]\nmasked() -> int\n    return handle\n        perform Ask()\n    with\n'
                  '        Ask() =>\n            perform Ask()\n            resume(1)\n')
        expect_compile_error(masked, "E-EFFECT", "pure function 'masked' performs effect 'Ask'")

    def test_declared_effects_account_for_handlers(self):
        src = ('#[effects: io]\nshow()\n    handle\n        print(perform Ask())\n    with\n'
               '        Ask() => resume("ok")\n    perform Other()\n\nmain()\n    show()\n')
        expect_compile_error(src, "E-EFFECT", "performs effect 'Other'", "declares only #[effects: io]")


# ═══════════════════════════════════════════════════════════════
# C VM PROFILES, SNAPSHOTS, STEP COUNTS
# ═══════════════════════════════════════════════════════════════

MANY_PERFORMS = '''
var total = 0
let r = handle
    for i in 0..10000
        total += perform Get(i)
    total
with
    Get(x) => resume(x % 7)
print(r)
var installs = 0
for i in 0..10000
    installs += handle
        perform One()
    with
        One() => resume(1)
print(installs)
var aborted = 0
for i in 0..10000
    aborted += handle
        perform Stop(i)
        0
    with
        Stop(v) => 1
print(aborted)
var broke = 0
for i in 0..10000
    while true
        handle
            perform Tick()
            break
        with
            Tick() => resume()
    broke += 1
print(broke)
deep(n: int) -> int
    if n == 0
        return 0
    return deep(n - 1) + 1
print(deep(15))
'''

GC_CLOSURES = '''
run(rounds: int) -> str
    var count = 0
    var last = ""
    for round in 0..rounds
        var acc = ""
        let r = handle
            for i in 0..8
                let s = perform Emit(str(i) + ":" + str(round))
                acc = acc + s
            acc
        with
            Emit(s) =>
                let t = "<" + s + ">"
                count += 1
                last = t
                resume(t)
        if round % 50 == 0
            print(r)
    return str(count) + " " + last

probe(seed: int) -> int
    let data = [seed, seed * 2, seed * 3]
    return handle
        var junk = ""
        for k in 0..40
            junk = junk + "xxxxxxxxxx"
        perform Sum() + len(junk)
    with
        Sum() => resume(data[0] + data[1] + data[2])

print(run(200))
var total = 0
for n in 0..30
    total += probe(n)
print(total)
'''

SNAPSHOT_PROGRAM = '''
var log = []
work(n: int) -> int
    var total = 0
    for i in 0..n
        total += perform Get(i)
        if i % 3 == 0
            perform Note("i=" + str(i))
    return total

let r = handle
    let a = work(12)
    let b = handle
        work(5)
    with
        Get(x) =>
            let y = perform Get(x + 100)
            resume(y + 1)
    a * 1000 + b
with
    Get(x) =>
        var s = 0
        for k in 0..3
            s += x
        resume(s)
    Note(msg) =>
        log.push(msg)
        print(msg)
        resume()
print(r)
print(log)
let ab = handle
    for i in 0..100
        if i == 7
            perform Stop(i)
    -1
with
    Stop(v) => v * 2
print(ab)
'''


def c_steps(data: bytes, profile: str = "host") -> int:
    rc, _out, err = run_cvm(data, profile=profile, vm_args=['--stats'])
    m = re.search(r'steps=(\d+)', err)
    assert m, err
    return int(m.group(1))


def run_exe(exe: str, data: bytes, tmp_path, timeout: int = 120):
    """Run a specially built C VM binary on bytecode."""
    path = tmp_path / "prog.tbc"
    path.write_bytes(data)
    return subprocess.run([exe, str(path)], capture_output=True, timeout=timeout)


@needs_cc
class TestCVM:
    def test_10000_performs_on_a_small_stack_profile(self):
        # resumes, handle installs, aborts and breaks 10000 times each: nothing may
        # leak (512 stack slots / 64 frames, and the 16-bit LEAN build: 128 / 24)
        expected = "29994\n10000\n10000\n10000\n15\n"
        data = tbc_bytes(MANY_PERFORMS)
        assert run_vm(data) == (0, expected)
        for profile in ("small", "tiny"):
            crc, cout, cerr = run_cvm(data, profile=profile)
            assert (crc, cout) == (0, expected), f"{profile}: {cerr}"

    def test_gcstress_with_closures_captured_by_clauses(self):
        out = both(GC_CLOSURES, profile="gcstress")
        assert out.splitlines()[-2:] == ["1600 <7:199>", str(sum(6 * n + 400 for n in range(30)))]
        rc, _o, err = run_cvm(GC_CLOSURES, profile="gcstress", vm_args=['--stats'])
        assert rc == 0 and int(re.search(r'gc=(\d+)', err).group(1)) > 0, err

    def test_identical_step_counts(self):
        for src in (SPEC_EXAMPLE, SNAPSHOT_PROGRAM, MANY_PERFORMS):
            data = tbc_bytes(src)
            vm = til.TILVM(til.TBCModule.from_bytes(data), write=lambda b: None)
            assert vm.run() == 0
            assert vm.steps == c_steps(data)

    def test_step_budget_counts_performs_like_calls(self):
        src = 'let r = handle\n    perform A() + perform A()\nwith\n    A() => resume(1)\nprint(r)\n'
        data = tbc_bytes(src)
        assert c_steps(data) == 3          # main() + two clause entries; resume is free
        rc, out = run_vm(data, max_steps=1)
        assert rc == 1
        crc, _cout, cerr = run_cvm(data, vm_args=['--max-steps', '1'])
        assert crc == 1 and "step limit exceeded" in cerr

    def test_raw_snapshot_in_the_middle_of_handled_bodies(self, tmp_path):
        data = tbc_bytes(SNAPSHOT_PROGRAM)
        prog = tmp_path / "p.tbc"
        prog.write_bytes(data)
        exe = cvm_binary("host")
        full = subprocess.run([exe, str(prog)], capture_output=True, timeout=60)
        assert full.returncode == 0
        assert full.stdout.decode() == run_vm(data)[1]
        steps = c_steps(data)
        assert steps > 80
        snap = tmp_path / "s.snap"
        for n in range(1, steps):
            r1 = subprocess.run([exe, '--snapshot-after', str(n), str(snap), str(prog)],
                                capture_output=True, timeout=60)
            assert r1.returncode == 3, r1.stderr
            r2 = subprocess.run([exe, '--resume', str(snap), str(prog)], capture_output=True, timeout=60)
            assert r2.returncode == 0, r2.stderr
            assert r1.stdout + r2.stdout == full.stdout, f"after {n} steps"

    def test_no_effects_build(self, tmp_path):
        """TILVM_NO_EFFECTS (for the smallest 8-bit builds) reports handlers as not
        available - explicitly - and keeps TIL 3.0 performs working."""
        exe = cvm_binary("host", ['-DTILVM_NO_EFFECTS', '-O0'])     # -O0: quick to build
        handled = tbc_bytes('let r = retry(1)\n    handle\n        1\n    with\n        A() => 2\nelse\n'
                            '    error\nprint(r)\n')
        r = run_exe(exe, handled, tmp_path)
        assert (r.returncode, r.stdout.decode()) == (0, "effect handlers are not available in this VM build\n")
        r = run_exe(exe, tbc_bytes('Log(m: str)\n    print("log " + m)\nperform Log("v3")\nperform Nope()\n'),
                    tmp_path)
        assert (r.returncode, r.stdout.decode()) == (1, "log v3\n")
        assert "unhandled effect 'Nope'" in r.stderr.decode()

    @pytest.mark.skipif(not os.environ.get('TIL_TEST_SLOW'), reason="slow: set TIL_TEST_SLOW=1")
    def test_sanitizers_with_a_tiny_heap(self, tmp_path):
        exe = cvm_binary("host", ['-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                                  '-DTILVM_HEAP_SIZE=4096'])
        for src in (SPEC_EXAMPLE, GC_CLOSURES, SNAPSHOT_PROGRAM, MANY_PERFORMS):
            data = tbc_bytes(src)
            r = run_exe(exe, data, tmp_path)
            assert r.returncode == 0, r.stderr
            assert r.stdout.decode() == run_vm(data)[1]


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("flags", [['-DTILVM_NO_EFFECTS'],
                                   ['-DTILVM_NO_EFFECTS', '-DTILVM_LEAN', '-DTILVM_INT_BITS=16',
                                    '-DTILVM_FLOAT=0', '-DTILVM_NO_STDIO'],
                                   ['-DTILVM_MAX_HANDLERS=4', '-DTILVM_LEAN', '-DTILVM_INT_BITS=16',
                                    '-DTILVM_FLOAT=0', '-DTILVM_NO_STDIO']],
                         ids=["noeffects", "noeffects-lean16", "handlers4-lean16"])
def test_strict_builds_of_the_effect_options(compiler, flags, tmp_path):
    import shutil
    if not shutil.which(compiler):
        pytest.skip(f"{compiler} not installed")
    r = subprocess.run([compiler, '-std=c99', '-Wall', '-Wextra', '-pedantic', '-Werror', *flags,
                        '-c', os.path.join(ROOT, 'vm', 'tilvm.c'), '-o', str(tmp_path / 'tilvm.o')],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ═══════════════════════════════════════════════════════════════
# TOOLS: disassembler, native fallback, languages, example
# ═══════════════════════════════════════════════════════════════

class TestDisassembler:
    def test_listing(self, tmp_path):
        src = ('A() -> int\n    return 1\nlet r = handle\n    perform A() + perform B(2)\nwith\n'
               '    B(x) => resume(x)\n    C => 0\nprint(r)\n')
        r = cli("dis", write(tmp_path, src))
        assert r.returncode == 0, r.stderr
        text = r.stdout
        assert re.search(r'HANDLE\s+2 \d+\s+; 2 clause\(s\), abort -> \d{4}', text)
        assert re.search(r'PERFORM\s+\d+ \d+ 0\s+; A, default A', text)
        assert re.search(r'PERFORM\s+\d+ 65535 1\s+; B, no default', text)
        assert "END_HANDLE" in text and "RESUME" in text
        assert re.search(r'fn #\d+ <clause B:6>\s+params=1 required=1 .*\[effect-clause\]', text)
        assert til.disassemble(compile_vm(src)) == text.rstrip("\n")


class TestNativeBackend:
    def test_native_code_generator_refuses_handlers(self):
        prog = til.resolve_imports(til.parse_source(SPEC_EXAMPLE, "x.til"), "x.til")
        with pytest.raises(til.NativeUnsupported):
            til.CCodeGenerator().generate(prog)
        assert issubclass(til.NativeUnsupported, til.CompileError)

    @needs_cc
    def test_run_falls_back_to_the_vm(self, tmp_path):
        r = cli("run", write(tmp_path, SPEC_EXAMPLE))
        assert r.returncode == 0, r.stderr
        assert r.stdout == "hello Aigerim\ndone\nbefore\nfailed: no network\n"
        assert "running it on the TIL VM" in r.stderr

    @needs_cc
    def test_native_build_reports_vm_only_feature(self, tmp_path):
        out = tmp_path / "exe"
        r = cli("build", write(tmp_path, SPEC_EXAMPLE), "-o", str(out))
        assert r.returncode == 1
        assert "native backend does not support effect handlers" in r.stderr
        assert not out.exists()

    @needs_cc
    def test_v3_perform_without_handlers_still_compiles_natively(self, tmp_path):
        src = ('Log(m: str)\n    print("log: " + m)\n\nTwice(n: int) -> int\n    return n * 2\n\n'
               'main()\n    perform Log("native")\n    let x = perform Twice(21)\n    print(x)\n')
        r = cli("run", write(tmp_path, src))
        assert (r.returncode, r.stdout) == (0, "log: native\n42\n"), r.stderr
        assert "TIL VM" not in r.stderr


LANG_PROGRAM = '''ask_twice() -> int
    let a = perform Ask(1)
    let b = perform Ask(2)
    return a + b

let r = handle
    ask_twice()
with
    Ask(n) => resume(n * 10)
    Stop =>
        print("stopped")
        0
print(r)
'''


class TestLanguages:
    def test_kazakh_keywords(self):
        src = '''тұрақты r = өңдеу
    орындау Сұрау(1) + орындау Сұрау(2)
арқылы
    Сұрау(n) => жалғау(n * 10)
басып_шығару(r)
'''
        assert both(src) == "30\n"

    def test_kazakh_latin_keywords(self):
        src = '''#[lang: kk-latn]
turaqty r = ondeu
    oryndau Surau(1) + oryndau Surau(2)
arqyly
    Surau(n) => zhalgau(n * 10)
print(r)
'''
        assert both(src) == "30\n"

    def test_russian_keywords(self):
        src = '''#[lang: ru]
пусть r = обработать
    выполнить Спросить(1) + выполнить Спросить(2)
с
    Спросить(n) => возобновить(n * 10)
печать(r)
'''
        assert both(src) == "30\n"

    @pytest.mark.parametrize("lang", ["kk", "kk-latn", "ru"])
    def test_translate_round_trip(self, lang):
        translated = til.translate_source(LANG_PROGRAM, lang)
        assert "handle" not in translated and "resume" not in translated and "perform" not in translated
        assert both(translated) == both(LANG_PROGRAM) == "30\n"
        assert til.translate_source(translated, "en") == LANG_PROGRAM

    def test_translate_cli(self, tmp_path):
        r = cli("translate", "--to", "kk", write(tmp_path, LANG_PROGRAM))
        assert r.returncode == 0, r.stderr
        assert "өңдеу" in r.stdout and "арқылы" in r.stdout and "жалғау" in r.stdout and "орындау" in r.stdout


class TestExample:
    def test_example_12_on_both_vms(self):
        path = os.path.join(ROOT, 'examples', '12_effects.til')
        with open(path, encoding='utf-8') as f:
            src = f.read()
        out = both(src)
        assert out.startswith("== ask / log ==\n[log] greeting Aigerim\nHello, Aigerim from Almaty!\n")
        assert "first negative: -4" in out and "[25, 16, 9, 4, 1]" in out and "total = 150" in out
        assert "[outer] [inner] inner work\ndone\n" in out and out.endswith("sensor = 42 after 3 calls\n")

    def test_example_12_cli_run(self):
        r = cli("run", "12_effects.til", cwd=os.path.join(ROOT, 'examples'))
        vm = cli("run", "--vm", "12_effects.til", cwd=os.path.join(ROOT, 'examples'))
        assert r.returncode == 0 and vm.returncode == 0, r.stderr + vm.stderr
        assert r.stdout == vm.stdout
