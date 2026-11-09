#!/usr/bin/env python3
"""
TIL 3.1 effects - regression tests for the findings of review round 1
(docs/TIL_EFFECTS.md). One class per finding:

  1. break/continue leaving a `handle` (or retry/timeout/match) that is part of
     a larger expression no longer leaks operand-stack slots (the C VM used to
     die with "stack overflow" where the Python VM grew silently)
  2. `perform E(...)` is a use of the top-level function E for #[cap: ...]
  3. a clause named `io` does not hide real I/O from #[pure] / #[effects]
  4. the REPL evaluates a statement-form `handle ... with` block
  5. no new reserved words: with/resume and the localized effect words are
     contextual keywords (TIL 3.0 programs keep compiling)
  6. code that could only run after resume(...) is rejected, also when the
     value of resume(...) would be used
  7. the design table describes the real HANDLE / END_HANDLE stack effects

Run: python3 -m pytest tests/test_effects_review.py -v
"""

import os
import re

import pytest

from tilhelpers import (CC, ROOT, cli, compile_vm, expect_compile_error, needs_cc, run_both, run_cvm,
                        run_vm, tbc_bytes, til)


def write(tmp_path, src, name="prog.til"):
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return str(p)


def c_steps(data: bytes, profile: str = "host") -> int:
    rc, _out, err = run_cvm(data, profile=profile, vm_args=['--stats'])
    m = re.search(r'steps=(\d+)', err)
    assert m, err
    return int(m.group(1))


# ═══════════════════════════════════════════════════════════════
# 1. break / continue out of expressions: no operand stack leak
# ═══════════════════════════════════════════════════════════════

REVIEW_PROGRAM = '''
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
print(count_even(N))
'''

# every function runs >128 loop iterations that leave an expression with
# break/continue, so a leak of one slot per iteration overflows the 128-slot
# stack of the "tiny" (16-bit LEAN) C VM profile; `len("probe:..")` records the
# stack depth on the Python VM at the start of every iteration
LEAK_VARIANTS = '''
count_even(n: int) -> int
    var total = 0
    for x in 0..n
        let _p = len("probe:count_even")
        total += handle
            if x % 2 == 1
                continue
            perform One()
        with
            One() => resume(1)
    return total

first_big(xs, reps: int) -> int
    var found = -1
    for r in 0..reps
        for x in xs
            let _p = len("probe:first_big")
            found = 1 + 2 * handle
                if x > 100
                    break
                perform Get(x)
            with
                Get(v) => resume(v)
    return found

nested(n: int) -> int
    var acc = 0
    var i = 0
    while i < n
        let _p = len("probe:nested")
        i += 1
        acc += handle
            10 + handle
                if i % 3 == 0
                    continue
                perform A() + perform B()
            with
                A() => resume(1)
        with
            B() => resume(2)
    return acc

retry_sum(n: int) -> int
    var s = 0
    for i in 0..n
        let _p = len("probe:retry_sum")
        s += retry(2)
            if i % 2 == 0
                continue
            1
        else
            0
    return s

timeout_sum(n: int) -> int
    var s = 0
    var i = 0
    loop
        let _p = len("probe:timeout_sum")
        i += 1
        if i > n
            break
        s += timeout(30000)
            if i % 5 == 0
                continue
            1
        else
            0
    return s

match_loop(n: int) -> int
    var t = 0
    for i in 0..n
        let _p = len("probe:match_loop")
        t += 7 * match i % 3
            0 => handle
                if i > 2
                    continue
                perform Z()
            with
                Z() => resume(100)
            _ => 1
    return t

mixed(n: int) -> int
    var t = 0
    for i in 0..n
        let _p = len("probe:mixed")
        t += handle
            3 * retry(2)
                if i % 4 == 3
                    continue
                if i == n - 1
                    break
                perform W(i)
            else
                0
        with
            W(v) => resume(v % 4)
    return t

arr_loop(xs, reps: int) -> int
    var t = 0
    for r in 0..reps
        for x in xs
            let _p = len("probe:arr_loop")
            t = t + handle
                handle
                    if x == 3
                        continue
                    perform Q() * perform R()
                with
                    Q() => resume(2)
            with
                R() => resume(x)
    return t

main()
    print(count_even(300))
    var small = []
    for k in 0..30
        small.push(k % 10)
    small.push(500)
    small.push(7)
    print(first_big(small, 8))
    print(nested(300))
    print(retry_sum(300))
    print(timeout_sum(300))
    print(match_loop(300))
    print(mixed(301))
    var ys = []
    for k in 0..40
        ys.push(k % 7)
    print(arr_loop(ys, 6))
'''
LEAK_EXPECTED = "150\n19\n2600\n150\n240\n2100\n675\n1164\n"


class ProbeVM(til.TILVM):
    """Reference VM that records the operand stack depth at every
    `len("probe:NAME")`: per probe the depth must never change."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depths = {}

    def call_native(self, nid, args):
        if nid == til.NATIVE_ID['len'] and args and isinstance(args[0], bytes) and args[0].startswith(b"probe:"):
            self.depths.setdefault(args[0][6:].decode(), set()).add(len(self.stack))
        return super().call_native(nid, args)


class TestNoStackLeak:
    def test_review_program_on_both_vms(self):
        data = tbc_bytes(REVIEW_PROGRAM.replace("N", "20000"))
        assert run_vm(data) == (0, "10000\n")
        if CC:
            assert run_cvm(data)[:2] == (0, "10000\n")
            vm = til.TILVM(til.TBCModule.from_bytes(data), write=lambda b: None)
            assert vm.run() == 0 and vm.steps == c_steps(data)

    @needs_cc
    def test_review_program_on_the_small_stack_profiles(self):
        # used to die after ~150 continues with 128-512 stack slots
        data = tbc_bytes(REVIEW_PROGRAM.replace("N", "2000"))
        for profile in ("tiny", "small"):
            rc, out, err = run_cvm(data, profile=profile)
            assert (rc, out) == (0, "1000\n"), f"{profile}: {err}"

    def test_continue_path_pops_the_pending_value(self):
        listing = til.disassemble(compile_vm(REVIEW_PROGRAM.replace("N", "10")))
        ops = re.findall(r'^\s+\d{4}\s+([A-Z_]+)', listing, re.M)
        text = " ".join(ops)
        # LOAD_LOCAL total, the clause pair, HANDLE ... then on `continue`:
        assert "PUSH_NIL END_HANDLE POP POP JUMP" in text, listing

    def test_every_variant_on_both_vms_and_the_tiny_profile(self):
        data = tbc_bytes(LEAK_VARIANTS)
        assert run_vm(data) == (0, LEAK_EXPECTED)
        if CC:
            for profile in ("host", "tiny"):
                rc, out, err = run_cvm(data, profile=profile)
                assert (rc, out) == (0, LEAK_EXPECTED), f"{profile}: {err}"
            vm = til.TILVM(til.TBCModule.from_bytes(data), write=lambda b: None)
            assert vm.run() == 0 and vm.steps == c_steps(data)

    def test_reference_vm_stack_depth_is_constant_per_iteration(self):
        vm = ProbeVM(til.TBCModule.from_bytes(tbc_bytes(LEAK_VARIANTS)), write=lambda b: None)
        assert vm.run() == 0
        assert len(vm.depths) == 8
        for name, depths in vm.depths.items():
            assert len(depths) == 1, f"{name}: the stack grows: {sorted(depths)[:5]}..."

    def test_dead_break_after_return_still_compiles(self):
        src = ('f(n: int) -> int\n    var t = 0\n    for i in 0..n\n        t += handle\n'
               '            return i\n            break\n        with\n            A() => 0\n    return t\n'
               'print(f(3))\n')
        assert run_both(src) == "0\n"


# depth tracking is exact for every program of the corpus: statements leave the
# stack as they found it, expressions push one value, jumps agree where they land
def _corpus():
    progs = [("<leak>", LEAK_VARIANTS), ("<review>", REVIEW_PROGRAM.replace("N", "5"))]
    for root, _dirs, files in sorted(os.walk(os.path.join(ROOT, 'examples'))):
        for name in sorted(files):
            if name.endswith('.til'):
                path = os.path.join(root, name)        # imports resolve next to the file
                with open(path, encoding='utf-8') as f:
                    progs.append((path, f.read()))
    return progs


class TestDepthTracking:
    def test_corpus_is_tracked_exactly(self, monkeypatch):
        problems = []
        states = []
        orig_init = til._FuncState.__init__
        orig_stmt = til.BytecodeCompiler.compile_stmt
        orig_expr = til.BytecodeCompiler.compile_expr

        def init(self, *a, **k):
            orig_init(self, *a, **k)
            states.append(self)

        def stmt(self, node):
            fs, before, dead = self.fs, self.fs.depth, self.fs.dead
            orig_stmt(self, node)
            if fs.depth_ok and not fs.dead and not dead and fs.depth != before:
                problems.append(("statement", type(node).__name__, node.line, before, fs.depth))

        def expr(self, node, parent=None):
            fs, before, dead = self.fs, self.fs.depth, self.fs.dead
            orig_expr(self, node, parent)
            if fs.depth_ok and not fs.dead and not dead and fs.depth != before + 1:
                problems.append(("expression", type(node).__name__, getattr(node, 'line', 0), before, fs.depth))

        monkeypatch.setattr(til._FuncState, '__init__', init)
        monkeypatch.setattr(til.BytecodeCompiler, 'compile_stmt', stmt)
        monkeypatch.setattr(til.BytecodeCompiler, 'compile_expr', expr)
        count = 0
        for name, src in _corpus():
            til.compile_to_bytecode(src, name)
            count += 1
        assert count >= 20
        lost = [fs.proto.name for fs in states if not fs.depth_ok]
        assert not lost, f"depth tracking lost in {lost}"
        assert not problems, problems[:10]

    def test_every_opcode_has_a_stack_effect(self):
        known = set(til._STACK_EFFECT) | set(til._OPERAND_EFFECT)
        ops = {v for k, v in vars(til.Op).items() if not k.startswith('_')}
        assert ops <= known, sorted(til.OP_NAMES[o] for o in ops - known)


# ═══════════════════════════════════════════════════════════════
# 2. capabilities: perform E(...) uses the default function E
# ═══════════════════════════════════════════════════════════════

SECRET = '#[cap: FileRead]\nSecret() -> str\n    return "top secret"\n\n'


class TestCapabilities:
    def test_perform_of_a_capability_function_needs_the_capability(self):
        src = SECRET + 'leak() -> str\n    return perform Secret()\n\nmain()\n    print(leak())\n'
        expect_compile_error(src, "E-CAP", "'leak' uses 'Secret', which requires capability 'FileRead'",
                             "perform Secret(...)` calls the function 'Secret'")
        # the same as a direct call
        expect_compile_error(src.replace("perform Secret()", "Secret()"), "E-CAP")

    def test_holding_the_capability_or_main_is_fine(self):
        ok = SECRET + '#[cap: FileRead]\nread() -> str\n    return perform Secret()\n\nmain()\n' \
                      '    print(read())\n    print(perform Secret())\n'
        assert run_both(ok) == "top secret\ntop secret\n"

    def test_a_statically_handled_perform_never_calls_the_function(self):
        src = SECRET + ('safe() -> str\n    return handle\n        perform Secret()\n    with\n'
                        '        Secret() => resume("fake")\n\nmain()\n    print(safe())\n')
        assert run_both(src) == "fake\n"

    def test_a_lambda_may_escape_the_handle(self):
        src = SECRET + ('leak() -> str\n    let f = handle\n        |x| perform Secret()\n    with\n'
                        '        Secret() => resume("fake")\n    return f(0)\n\nmain()\n    print(leak())\n')
        expect_compile_error(src, "E-CAP", "'leak' uses 'Secret'")

    def test_a_masked_perform_inside_its_own_clause_calls_the_function(self):
        src = SECRET + ('leak() -> str\n    return handle\n        perform Secret()\n    with\n'
                        '        Secret() => resume(perform Secret())\n\nmain()\n    print(leak())\n')
        expect_compile_error(src, "E-CAP", "'leak' uses 'Secret'")

    def test_struct_literal_fields_are_checked(self):
        struct = 'struct Box\n    v: str\n\n'
        perform = struct + SECRET + 'leak() -> str\n    let b = Box { v: perform Secret() }\n    return b.v\n'
        expect_compile_error(perform, "E-CAP", "'leak' uses 'Secret'")
        call = struct + SECRET + 'leak() -> str\n    let b = Box { v: Secret() }\n    return b.v\n'
        expect_compile_error(call, "E-CAP", "'leak' uses 'Secret'")
        pure = struct + '#[pure]\nnoisy() -> int\n    let b = Box { v: str(print("io")) }\n    return 1\n'
        expect_compile_error(pure, "E-EFFECT", "pure function 'noisy' performs effect 'io'")

    def test_effects_of_default_functions_are_unchanged(self):
        # TIL 3.0: `#[effects: Log]` + a printing default function Log stays valid
        src = ('Log(m: str)\n    print("log " + m)\n\n#[effects: Log]\nwork()\n    perform Log("x")\n\n'
               'main()\n    work()\n')
        assert run_both(src) == "log x\n"


# ═══════════════════════════════════════════════════════════════
# 3. the builtin effect io cannot be handled by a clause
# ═══════════════════════════════════════════════════════════════

class TestIoIsNotHandleable:
    NOISY = 'noisy() -> int\n    print("I/O happens here")\n    return 1\n\n'

    def test_io_clause_does_not_hide_a_callee_that_prints(self):
        src = self.NOISY + ('#[pure]\nclaims_pure() -> int\n    return handle\n        noisy()\n    with\n'
                            '        io() => resume(0)\n\nmain()\n    print(claims_pure())\n')
        expect_compile_error(src, "E-EFFECT", "pure function 'claims_pure' performs effect 'io'",
                             "through the call to 'noisy'")
        declared = src.replace("#[pure]", "#[effects: Other]")
        expect_compile_error(declared, "E-EFFECT", "performs effect 'io'")

    def test_direct_print_and_perform_io_count_too(self):
        direct = ('#[pure]\nf() -> int\n    return handle\n        print("x")\n        1\n    with\n'
                  '        io() => resume(0)\n')
        expect_compile_error(direct, "E-EFFECT", "performs effect 'io'")
        performed = ('#[pure]\ng() -> int\n    return handle\n        perform io()\n    with\n'
                     '        io() => resume(0)\n')
        expect_compile_error(performed, "E-EFFECT", "performs effect 'io'")

    def test_user_effects_are_still_handled(self):
        src = ('ask() -> int\n    return perform Ask()\n\n#[pure]\nf() -> int\n    return handle\n'
               '        ask()\n    with\n        Ask() => resume(5)\n\nmain()\n    print(f())\n')
        assert run_both(src) == "5\n"


# ═══════════════════════════════════════════════════════════════
# 4. REPL
# ═══════════════════════════════════════════════════════════════

class TestRepl:
    def repl(self, text):
        r = cli("repl", input_text=text)
        assert r.returncode == 0, r.stderr
        return r.stdout

    def test_statement_form_handle_block(self):
        out = self.repl('handle\n    print(perform B())\nwith\n    B() => resume("stmt")\n\n'
                        'handle\n    perform V()\nwith\n    V() => resume(7)\n\n')
        assert "stmt\n" in out and "7\n" in out and "error" not in out

    def test_let_handle_and_localized_blocks(self):
        out = self.repl('let r = handle\n    perform A(1)\nwith\n    A(x) => resume(x + 41)\n\nprint(r)\n'
                        'өңдеу(x: int) -> int\n    return x + 1\n\nөңдеу(9)\n'
                        'тұрақты k = өңдеу\n    орындау Сұрау(1)\nарқылы\n    Сұрау(n) => жалғау(n * 30)\n\n'
                        'басып_шығару(k)\n')
        assert "42\n" in out and "10\n" in out and "30\n" in out and "error" not in out

    def test_repl_program_keeps_relative_indentation(self):
        src, is_expr = til.repl_program([], [], 'handle\n    1 + perform A()\nwith\n    A() => resume(2)')
        assert is_expr
        assert "    let __v = handle\n        1 + perform A()\n    with\n        A() => resume(2)\n" in src
        assert til.compile_to_bytecode(src, "<repl>")


# ═══════════════════════════════════════════════════════════════
# 5. contextual keywords: TIL 3.0 programs keep compiling
# ═══════════════════════════════════════════════════════════════

TIMER = '''struct Timer
    running: bool

impl Timer
    resume(self)
        self.running = true

    with(self, x: int) -> int
        return x + 1

main()
    let t = Timer { running: false }
    t.resume()
    print(t.running)
    let with = t.with(41)
    let resume = with + 1
    print(resume)
    let r = handle
        perform Ask()
    with
        Ask() =>
            t.resume()
            resume(with)
    print(r)
'''


class TestContextualKeywords:
    def test_with_and_resume_as_names(self):
        assert run_both(TIMER) == "true\n43\n42\n"

    @needs_cc
    def test_with_and_resume_as_names_natively(self, tmp_path):
        r = cli("run", write(tmp_path, TIMER.split("    let r = handle")[0]))
        assert (r.returncode, r.stdout) == (0, "true\n43\n"), r.stderr
        assert "TIL VM" not in r.stderr

    def test_localized_words_as_names(self):
        kk = ('орындау(x: int) -> int\n    return x * 2\n\nөңдеу(x: int) -> int\n    return x + 1\n\n'
              'жалғау(a: int) -> int\n    return a\n\nарқылы(a: int) -> int\n    return a\n\n'
              'басып_шығару(орындау(20) + өңдеу(1) + жалғау(0) + арқылы(0))\n')
        assert run_both(kk) == "42\n"
        assert run_both('#[lang: ru]\nпусть с = 5\nпечать(с + 1)\n') == "6\n"
        assert run_both('#[lang: ru]\nпусть выполнить = 2\nпусть обработать = выполнить * 3\n'
                        'пусть возобновить = обработать\nпечать(возобновить)\n') == "6\n"
        assert run_both('#[lang: kk-latn]\nturaqty oryndau = 1\nturaqty ondeu = oryndau + 1\n'
                        'turaqty arqyly = ondeu\nturaqty zhalgau = arqyly\nprint(zhalgau)\n') == "2\n"

    def test_localized_handle_forms(self):
        src = ('тұрақты a = өңдеу 5 арқылы Ж() => 0\n'
               'тұрақты b = өңдеу орындау Ж() арқылы Ж() => жалғау(7)\n'
               'тұрақты c = өңдеу\n    орындау Ж()\nарқылы\n    Ж() => жалғау(9)\n'
               'тұрақты өңдеу = 1\nбасып_шығару(a + b + c + өңдеу)\n')
        assert run_both(src) == "22\n"

    def test_resume_is_reserved_only_inside_clauses(self):
        head = 'let r = handle\n    perform A()\nwith\n    A() =>\n'
        expect_compile_error(head + '        let resume = 1\n        resume(1)\n',
                             "'resume' is reserved inside an effect clause")
        expect_compile_error(head + '        for resume in [1]\n            print(1)\n        resume(1)\n',
                             "'resume' is reserved inside an effect clause")
        expect_compile_error('let r = handle\n    perform A(1)\nwith\n    A(resume) => resume\n',
                             "'resume' is reserved inside an effect clause")
        expect_compile_error('var resume = 0\n' + head + '        resume = 1\n        resume(1)\n',
                             "'resume' is reserved inside an effect clause")
        # outside a clause an undefined resume(...) is still explained
        expect_compile_error('resume(1)\n', "resume(...) outside an effect clause")

    def test_reserved_words_are_named_in_errors(self):
        expect_compile_error('let perform = 1\n', "'perform' is a reserved word")
        expect_compile_error('struct T\n    x: int\nimpl T\n    handle(self) -> int\n        return 1\n',
                             "Expected method definition in impl block", "'handle' is a reserved word")

    def test_translate_keeps_names_and_translates_keywords(self):
        for lang in ("kk", "kk-latn", "ru"):
            translated = til.translate_source(TIMER, lang)
            assert "t.resume()" in translated and "t.with(41)" in translated
            assert "let with" not in translated               # `let` is translated, the name is not
            assert run_both(translated) == "true\n43\n42\n"
            assert til.translate_source(translated, "en") == TIMER


# ═══════════════════════════════════════════════════════════════
# 6. nothing may follow resume(...)
# ═══════════════════════════════════════════════════════════════

def clause(body: str) -> str:
    return 'let r = handle\n    perform A(true)\nwith\n    A(x) =>\n' + \
        "".join("        " + line + "\n" for line in body.split("\n")) + 'print(r)\n'


VALUE = "the value of resume(...) is never available"
AFTER = "unreachable code after resume(...)"


class TestUnreachableAfterResume:
    @pytest.mark.parametrize("body, message, where", [
        ('let v = resume(1)\nprint("never runs")\nv', VALUE, "5:17"),     # the review example
        ('print(resume(1))', VALUE, "5:15"),
        ('resume(1) + 1', VALUE, "5:9"),
        ('x = resume(1)\nresume(2)', VALUE, "5:13"),
        ('resume(1) and x', VALUE, "5:9"),
        ('resume(resume(1))', AFTER, "5:9"),
        ('if x\n    resume(1)\nelse\n    resume(2)\nprint("dead")', AFTER, "9:9"),
        ('if resume(1)\n    print("x")', AFTER, "6:13"),
        ('while resume(1)\n    print("x")', AFTER, "6:13"),
        ('match resume(1)\n    1 => print("x")\n    _ => print("y")', "the match arms are never reached", "5:9"),
        ('resume(1)\nprint("x")', AFTER, "6:9"),
    ])
    def test_rejected(self, body, message, where):
        expect_compile_error(clause(body), message, where)

    @pytest.mark.parametrize("body, out", [
        ('if x\n    resume(1)\nprint("x false")\nresume(2)', "1\n"),
        ('resume(1) if x else 0', "1\n"),
        ('x and resume(3)\nresume(4)', "3\n"),
        ('not x or resume(5)\nresume(6)', "5\n"),
        ('for i in 0..3\n    resume(i + 10)\nresume(9)', "10\n"),
        ('let v = retry(2)\n    resume(11)\nelse\n    0\nresume(v)', "11\n"),
        ('let v = handle\n    perform B()\n    resume(1)\nwith\n    B() => 12\nresume(v)', "12\n"),
        ('match x\n    true => resume(13)\n    _ => resume(0)', "13\n"),
    ])
    def test_reachable_code_is_fine(self, body, out):
        assert run_both(clause(body)) == out


# ═══════════════════════════════════════════════════════════════
# 7. the shared design table matches the implementation
# ═══════════════════════════════════════════════════════════════

class TestDesignTable:
    def test_handle_rows_describe_the_pairs_on_the_stack(self):
        with open(os.path.join(ROOT, 'docs', 'dev', 'TIL_3_1_DESIGN.md'), encoding='utf-8') as f:
            rows = {line.split('|')[1].strip(): line for line in f if line.startswith('| ')}
        assert "stay on the value stack" in rows['HANDLE'] and "stack effect 0" in rows['HANDLE']
        assert "−2n" in rows['END_HANDLE']
        # ... and so does the compiler's own stack accounting
        assert til._STACK_EFFECT[til.Op.HANDLE] == 0
        assert til._JUMP_LANDING[til.Op.HANDLE] == 1        # abort: pairs cut away (-2n), value pushed
