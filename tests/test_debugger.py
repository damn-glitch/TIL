#!/usr/bin/env python3
"""
TIL 3.1 #4: the time-travel debugger (`til debug`, src/tildebug.py) and the
debug-info section of the bytecode (docs/TIL_DEBUGGER.md).

* scripted sessions: breakpoints, s / n / out, back, rc, p of locals, globals,
  arrays, structs, upvalues, bt in recursion, unknown commands;
* time travel: no output printed twice after back + c, deterministic replay of
  input() (stdin and --input), clock_ms(), sleep_ms(), timeout blocks, keys;
* effects, retry, closures and graphics under the debugger;
* debug info: round trip, live ranges after superinstructions, globals past
  slot 255, --strip omits it, default bytecode unchanged, the C VM runs files
  with the section, files without it (selfhost compiler output) still load.

Run: python3 -m pytest tests/test_debugger.py -v
"""

import os
import re
import subprocess
import sys
import tempfile

import pytest

from tilhelpers import ROOT, SRC, TIL, cli, cvm_binary, needs_cc, run_cvm, run_vm, til

sys.path.insert(0, SRC)
import tildebug  # noqa: E402

TMP = tempfile.mkdtemp(prefix='til-debug-')


def write(name: str, text: str) -> str:
    path = os.path.join(TMP, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def debug(src: str, commands: str, stdin: str = "", args=(), extra=(), name: str = "prog.til",
          env=None):
    """til debug PROG --commands ... -> (exit code, stdout, stderr)."""
    path = write(name, src)
    r = cli('debug', path, '--commands', commands, *extra, *(('--',) + tuple(args) if args else ()),
            input_text=stdin, env=env)
    return r.returncode, r.stdout, r.stderr


def stops(out: str):
    """The 'FUNC at line N' / 'Breakpoint K, FUNC at line N' stop lines of a session."""
    return re.findall(r'^(?:Breakpoint \d+, )?(\S+) at line (\d+)$', out, re.M)


def answer(out: str, command: str, nth: int = 0) -> str:
    """What the session printed after the nth `(til) command` line (up to the next command)."""
    parts = out.split(f"(til) {command}\n")
    assert len(parts) > nth + 1, f"no ({nth}) `{command}` in:\n{out}"
    return parts[nth + 1].split("(til) ")[0]


LOOP = """\
var total = 0
let names = ["a", "b"]

add(a, b)
    let s = a + b
    return s

main()
    for i in 0..4
        let sq = i * i
        total = add(total, sq)
        print(total)
    print("done")
"""

FACT = """\
fact(n: int) -> int
    if n <= 1
        return 1
    return n * fact(n - 1)

main()
    let r = fact(4)
    print(r)
"""


# ═══════════════════════════════════════════════════════════════════════════
# scripted sessions
# ═══════════════════════════════════════════════════════════════════════════

class TestSession:
    def test_breakpoint_continue_print(self):
        rc, out, err = debug(LOOP, "b 11; c; p i; p sq; p total; p names; p names[1]; locals; c; p total; q")
        assert rc == 0, err
        assert out.startswith("__start__ at line 1\n    1 | var total = 0\n(til) b 11\nBreakpoint 1 at line 11\n")
        assert answer(out, "c") == "Breakpoint 1, main at line 11\n   11 |         total = add(total, sq)\n"
        assert answer(out, "p i") == "i = 0\n"
        assert answer(out, "p sq") == "sq = 0\n"
        assert answer(out, "p names") == 'names = ["a", "b"]\n'      # strings quoted inside containers
        assert answer(out, "p names[1]") == "names[1] = b\n"          # like print(): not at the top
        assert answer(out, "locals") == "i = 0\nsq = 0\n"
        assert answer(out, "c", 1) == "0\nBreakpoint 1, main at line 11\n   11 |         total = add(total, sq)\n"
        assert answer(out, "p total") == "total = 0\n"

    def test_full_transcript(self):
        """The whole output of a session is deterministic (the --commands format)."""
        rc, out, err = debug(FACT, "b 4; c; p n; c; bt; back; p n; q")
        assert rc == 0, err
        assert out == """\
main at line 7
    7 |     let r = fact(4)
(til) b 4
Breakpoint 1 at line 4
(til) c
Breakpoint 1, fact at line 4
    4 |     return n * fact(n - 1)
(til) p n
n = 4
(til) c
Breakpoint 1, fact at line 4
    4 |     return n * fact(n - 1)
(til) bt
  at fact(n=3) (line 4)
  at fact(n=4) (line 4)
  at main() (line 7)
  at __start__()
(til) back
fact at line 2
    2 |     if n <= 1
(til) p n
n = 3
(til) q
"""

    def test_step_next_out(self):
        rc, out, err = debug(LOOP, "b 11; c; s; s; s; n; n; n; d 11; out; n; q")
        assert rc == 0, err
        assert stops(out) == [('__start__', '1'), ('main', '11'), ('add', '5'), ('add', '6'),
                              ('main', '12'),          # s at `return s`: the next line event anywhere
                              ('main', '9'),           # n: the loop header, then the body
                              ('main', '10'), ('main', '11'), ('__start__', '2')]
        assert answer(out, "n", 2) == "Breakpoint 1, main at line 11\n   11 |         total = add(total, sq)\n"
        # out from main: the rest of main runs, then the stop in its caller
        assert answer(out, "out") == ("1\n5\n14\ndone\n__start__ at line 2\n    2 | let names = [\"a\", \"b\"]\n"
                                      "main returned None\n")
        assert answer(out, "n", 3) == "Program exited with code 0\n"

    def test_next_steps_over_calls_and_out_returns_to_the_caller(self):
        rc, out, err = debug(FACT, "s; s; s; s; bt; out; bt; out; n; q")
        assert rc == 0, err
        assert stops(out)[:5] == [('main', '7'), ('fact', '2'), ('fact', '4'), ('fact', '2'), ('fact', '4')]
        assert answer(out, "bt") == ("  at fact(n=3) (line 4)\n  at fact(n=4) (line 4)\n"
                                     "  at main() (line 7)\n  at __start__()\n")
        # out: back in the caller, in the middle of the line of the call, with the value
        assert answer(out, "out") == "fact at line 4\n    4 |     return n * fact(n - 1)\nfact returned 6\n"
        assert answer(out, "bt", 1) == "  at fact(n=4) (line 4)\n  at main() (line 7)\n  at __start__()\n"
        assert answer(out, "out", 1) == "main at line 7\n    7 |     let r = fact(4)\nfact returned 24\n"
        assert answer(out, "n") == "main at line 8\n    8 |     print(r)\n"

    def test_next_over_recursion_stays_in_the_frame(self):
        rc, out, err = debug(FACT, "s; n; n; n; n; q")
        assert rc == 0, err
        # fact(4): line 2, then line 4 (n steps over the recursive call: no stop inside),
        # then main again (line 8, then the implicit return on the `main()` line)
        assert stops(out) == [('main', '7'), ('fact', '2'), ('fact', '4'), ('main', '8'), ('main', '6')]
        assert answer(out, "n", 2) == "24\nmain at line 6\n    6 | main()\n"
        assert answer(out, "n", 3) == "Program exited with code 0\n"

    def test_bt_in_recursion(self):
        rc, out, err = debug(FACT, "b 3; c; bt; q")
        assert rc == 0, err
        assert answer(out, "bt") == ("  at fact(n=1) (line 3)\n  at fact(n=2) (line 4)\n  at fact(n=3) (line 4)\n"
                                     "  at fact(n=4) (line 4)\n  at main() (line 7)\n  at __start__()\n")

    def test_print_structs_maps_and_paths(self):
        src = """\
struct Point
    x: int
    y: int

main()
    var p = Point { x: 1, y: 2 }
    let m = {"k": [1, "two"], "n": None}
    let grid = [[1, 2], [3, 4]]
    let j = 1
    p.x = 7
    print(p)
"""
        rc, out, err = debug(src, 'b 10; c; p p; p p.y; p m; p m["k"]; p m.k; p grid[1][0]; p grid[j]; '
                                  'p grid[5]; p p.z; p q; p 1+; n; p p; p p.x; q')
        assert rc == 0, err
        assert answer(out, "p p") == "p = Point { x: 1, y: 2 }\n"
        assert answer(out, "p p.y") == "p.y = 2\n"
        assert answer(out, "p m") == 'm = {"k": [1, "two"], "n": None}\n'
        assert answer(out, 'p m["k"]') == 'm["k"] = [1, "two"]\n'
        assert answer(out, "p m.k") == 'm.k = [1, "two"]\n'
        assert answer(out, "p grid[1][0]") == "grid[1][0] = 3\n"
        assert answer(out, "p grid[j]") == "grid[j] = [3, 4]\n"
        assert "index out of bounds" in answer(out, "p grid[5]")
        assert "no field 'z' on Point" in answer(out, "p p.z")
        assert answer(out, "p q") == "No variable 'q' here\n"
        assert "Cannot evaluate '1+'" in answer(out, "p 1+")
        assert answer(out, "p p", 1) == "p = Point { x: 7, y: 2 }\n"

    def test_locals_live_ranges(self):
        src = """\
main()
    let a = 1
    if a > 0
        let b = 2
        print(b)
    let c = 3
    print(a + c)
"""
        rc, out, err = debug(src, "locals; n; locals; n; n; locals; n; locals; n; locals; q")
        assert rc == 0, err
        assert answer(out, "locals") == "No local variables\n"          # line 2: a is not set yet
        assert answer(out, "locals", 1) == "a = 1\n"
        assert answer(out, "locals", 2) == "a = 1\nb = 2\n"             # line 5
        assert answer(out, "locals", 3) == "a = 1\n"                    # line 6: b's block ended
        assert answer(out, "locals", 4) == "a = 1\nc = 3\n"

    def test_breakpoint_on_a_line_without_code(self):
        rc, out, err = debug(LOOP, "b 3; b 7; b 99; b; d 11; d 5; b; d; b; c; q")
        assert rc == 0, err
        assert answer(out, "b 3") == "Breakpoint 1 at line 4 (line 3 has no code)\n"
        assert answer(out, "b 7") == "Breakpoint 2 at line 8 (line 7 has no code)\n"
        assert answer(out, "b 99") == "No code at or after line 99\n"
        assert answer(out, "b") == "Breakpoint 1 at line 4\nBreakpoint 2 at line 8\n"
        assert answer(out, "d 11") == "No breakpoint at line 11\n"
        assert answer(out, "d") == "Deleted all breakpoints\n"
        assert answer(out, "b", 2) == "No breakpoints\n"
        assert "Program exited with code 0" in answer(out, "c")

    def test_unknown_command_short_help(self):
        rc, out, err = debug(LOOP, "frobnicate; b x; p; help; q")
        assert rc == 0, err
        assert answer(out, "frobnicate") == (
            "Unknown command 'frobnicate'\n"
            "commands: b LINE, d LINE, c, s, n, out, back, rc, p NAME, locals, bt, q (help: details)\n")
        assert answer(out, "b x") == "Usage: b LINE\n"
        assert answer(out, "p") == "Usage: p NAME\n"
        assert "  back     step back one line\n" in answer(out, "help")

    def test_end_of_program(self):
        rc, out, err = debug(LOOP, "c; c; s; p total; bt; q")
        assert rc == 0, err
        assert answer(out, "c") == "0\n1\n5\n14\ndone\nProgram exited with code 0\n"
        assert answer(out, "c", 1) == "The program has finished (back and rc go back in time, q quits)\n"
        assert answer(out, "p total") == "total = 14\n"          # globals stay readable

    def test_exit_code_and_runtime_error(self):
        src = "divide(a, b)\n    return a / b\n\nmain()\n    for x in [4, 0]\n        print(divide(8, x))\n"
        rc, out, err = debug(src, "c; bt; p b; back; p b; back; back; c; c; back; c; q")
        assert rc == 0
        assert answer(out, "c") == "2\nProgram ended with an error (exit code 1)\n"
        assert answer(out, "bt") == "  at divide(a=8, b=0) (line 2)\n  at main() (line 6)\n  at __start__()\n"
        assert answer(out, "p b") == "b = 0\n"                    # the state where it failed
        assert err.count("runtime error: division by zero") == 1, err     # reported once
        assert out.count("\n2\n") == 1

    def test_interactive_from_stdin(self):
        path = write("inter.til", LOOP)
        r = cli('debug', path, input_text="b 12\nc\n\np total\nq\n")
        assert r.returncode == 0, r.stderr
        assert r.stdout.startswith("TIL time-travel debugger (type help for the commands)\n__start__ at line 1\n")
        # the prompt, an empty line repeats the last command (c)
        assert r.stdout.count("Breakpoint 1, main at line 12") == 2
        assert "(til) total = 1\n" in r.stdout

    @pytest.mark.skipif(os.name != 'posix', reason="SIGINT to a child process")
    def test_ctrl_c_stops_at_the_next_line_event(self):
        import signal
        import time
        path = write("forever.til", "main()\n    var n = 0\n    while true\n        n += 1\n")
        p = subprocess.Popen([sys.executable, TIL, 'debug', path], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            seen = b""
            deadline = time.time() + 30
            while b"(til) " not in seen and time.time() < deadline:      # the first prompt
                seen += os.read(p.stdout.fileno(), 4096)
            p.stdin.write(b"c\n")
            p.stdin.flush()
            time.sleep(0.5)
            p.send_signal(signal.SIGINT)
            out, err = p.communicate(b"p n\nq\n", timeout=60)
        finally:
            if p.poll() is None:
                p.kill()
        text = (seen + out).decode()
        assert p.returncode == 0, err
        assert "Interrupted\nmain at line " in text
        assert re.search(r"\(til\) n = \d+", text)

    def test_localized_messages(self):
        rc, out, err = debug(FACT, "b 4; c; xyz; q", extra=('--lang', 'ru'))
        assert rc == 0, err
        assert "Точка останова 1: строка 4" in out
        assert "Точка останова 1, fact, строка 4" in out
        assert "Неизвестная команда 'xyz'" in out
        rc, out, err = debug(FACT, "s; bt; q", env={'TIL_LANG': 'kk'})
        assert "fact, 2-жол" in out and "  fact(n=4) ішінде (2-жол)" in out

    def test_main_in_process_keeps_the_language(self, capsys):
        before = til.MESSAGE_LANG
        assert tildebug.main([write("inproc.til", FACT), '--commands', 'b 4; c; q', '--lang', 'ru']) == 0
        assert "Точка останова 1, fact, строка 4" in capsys.readouterr().out
        assert til.MESSAGE_LANG == before
        tildebug.load_module(write("pragma.til", "#[lang: ru]\n" + FACT))
        assert til.MESSAGE_LANG == before

    def test_usage_errors(self):
        r = cli('debug')
        assert r.returncode == 2 and "no program given" in r.stderr
        r = cli('debug', os.path.join(TMP, 'missing.til'))
        assert r.returncode == 2 and "File not found" in r.stderr
        r = cli('debug', write("u.til", LOOP), '--bogus')
        assert r.returncode == 2 and "unknown option --bogus" in r.stderr
        r = cli('debug', write("syntax.til", "main()\n    let = 3\n"), '--commands', 'q')
        assert r.returncode == 1 and "error" in r.stderr.lower()


# ═══════════════════════════════════════════════════════════════════════════
# time travel
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeTravel:
    def test_back_then_continue_prints_nothing_twice(self):
        rc, out, err = debug(LOOP, "b 12; c; c; c; back; back; back; back; back; back; p total; c; c; c; "
                                   "back; back; c; q")
        assert rc == 0, err
        program = [ln for ln in out.split("\n") if re.fullmatch(r'\d+|done', ln)]
        assert program == ["0", "1", "5", "14", "done"]
        # six line events back from line 12 of the third iteration: line 12 of the second
        assert answer(out, "p total") == "total = 1\n"

    def test_back_steps_through_history(self):
        rc, out, err = debug(LOOP, "s; s; s; s; s; s; s; back; back; back; back; back; back; back; back; q")
        assert rc == 0, err
        forward = stops(out)[:8]
        backward = stops(out)[8:]
        assert forward == [('__start__', '1'), ('__start__', '2'), ('main', '9'), ('main', '10'),
                           ('main', '11'), ('add', '5'), ('add', '6'), ('main', '12')]
        assert backward == list(reversed(forward[:7])), out
        assert answer(out, "back", 7) == "Already at the start of the program\n"

    def test_reverse_continue(self):
        rc, out, err = debug(LOOP, "b 11; c; c; c; p i; rc; p i; rc; p i; rc; p i; d 11; c; rc; q")
        assert rc == 0, err
        assert [answer(out, "p i", k) for k in range(4)] == ["i = 2\n", "i = 1\n", "i = 0\n",
                                                             "No variable 'i' here\n"]
        assert answer(out, "rc", 2) == ("No earlier breakpoint hit: back to the start of the program\n"
                                        "__start__ at line 1\n    1 | var total = 0\n")
        assert "No earlier breakpoint hit" in answer(out, "rc", 3)     # no breakpoints at all

    def test_back_from_the_end(self):
        rc, out, err = debug(LOOP, "c; back; back; p total; c; q")
        assert rc == 0, err
        assert stops(answer(out, "back")) == [('main', '8')]            # the implicit return of main
        assert stops(answer(out, "back", 1)) == [('main', '13')]
        assert answer(out, "p total") == "total = 14\n"
        assert answer(out, "c", 1) == "done\n" or answer(out, "c", 1) == "Program exited with code 0\n"
        assert out.count("done\n") == 1

    def test_replay_input_and_clock(self):
        src = """\
main()
    let t0 = clock_ms()
    sleep_ms(20)
    let t1 = clock_ms()
    let a = input("a? ")
    let b = input()
    let r = random(1, 1000000)
    print(a + "/" + b)
    print(t1 - t0 >= 20)
"""
        cmds = "b 8; c; p t0; p t1; p a; p b; p r; back; back; back; back; back; p t0; c; p t1; p a; p b; p r; c; q"
        rc, out, err = debug(src, cmds, stdin="first\nsecond\n")
        assert rc == 0, err
        for name in ("t1", "a", "b", "r"):
            assert answer(out, f"p {name}") == answer(out, f"p {name}", 1), name
        assert answer(out, "p t0") == answer(out, "p t0", 1)
        assert answer(out, "p a") == "a = first\n" and answer(out, "p b") == "b = second\n"
        assert out.count("\na? ") == 1 and out.count("first/second") == 1    # the prompt, once
        assert "true\nProgram exited with code 0" in out

    def test_input_from_file(self):
        src = "main()\n    let x = input()\n    print(\"got \" + x)\n"
        inp = write("in.txt", "from the file\n")
        rc, out, err = debug(src, "n; n; back; back; c; q", extra=('--input', inp))
        assert rc == 0, err
        assert out.count("got from the file") == 1

    def test_args(self):
        rc, out, err = debug("main()\n    let a = args()\n    print(a)\n", "n; p a; q", args=("x", "y z"))
        assert rc == 0, err
        assert answer(out, "p a") == 'a = ["x", "y z"]\n'

    def test_timeout_replays_identically(self):
        src = """\
main()
    var n = 0
    let r = timeout(30)
        while true
            n += 1
        0
    else
        n
    print(r == n)
"""
        rc, out, err = debug(src, "b 9; c; p n; back; back; back; p n; c; p n; q")
        assert rc == 0, err
        values = {answer(out, "p n", k) for k in range(3)}
        assert len(values) == 1, out             # the timeout fired at the same iteration
        assert int(values.pop().split("= ")[1]) > 0

    def test_small_snapshot_interval_and_thinning(self, monkeypatch):
        src = "main()\n    var t = 0\n    for i in 0..3000\n        t += i\n    print(t)\n"
        module, _src = tildebug.load_module(write("thin.til", src))
        out = []
        vm = tildebug.DebugVM(module, out.append, lambda: b"", every=1)
        monkeypatch.setattr(tildebug, 'MAX_SNAPSHOTS', 16)
        assert vm.resume(lambda ev, line, depth, top: False) == 0
        assert b"".join(out) == b"4498500\n"
        assert 2 <= len(vm.snaps) <= 17 and 0 in vm.snaps and vm.every > 1
        events = len(vm.ev_lines)
        for target in (events - 1, events // 2, 1, 0):
            assert vm.travel(target) is None and vm.ev == target
            assert vm.at_event and vm.paused
        # forward again from event 0 to the end: the same output, printed once
        assert vm.resume(lambda ev, line, depth, top: False) == 0
        assert b"".join(out) == b"4498500\n"


# ═══════════════════════════════════════════════════════════════════════════
# effects, retry, closures, graphics
# ═══════════════════════════════════════════════════════════════════════════

EFFECTS = """\
greet() -> str
    let name = perform Ask("name?")
    perform Log("hello " + name)
    return name

main()
    var count = 0
    let r = handle
        greet()
        "done"
    with
        Ask(question) => resume("Aigerim")
        Log(msg) =>
            count += 1
            print(msg)
            resume(None)
    print(r)
    let f = handle
        perform Fail("no")
        "not reached"
    with
        Fail(why) => "failed: " + why
    print(f)
    print(count)
"""


class TestFeatures:
    def test_effects(self):
        rc, out, err = debug(EFFECTS, "b 14; b 22; c; p msg; p count; bt; n; n; p count; c; c; p why; bt; "
                                      "rc; p count; rc; back; c; c; c; c; q")
        assert rc == 0, err
        assert answer(out, "c") == "Breakpoint 1, <clause Log:13> at line 14\n   14 |             count += 1\n"
        assert answer(out, "p msg") == "msg = hello Aigerim\n"
        assert answer(out, "p count") == "count = 0\n"                       # an upvalue of the clause
        assert answer(out, "bt") == ("  at <clause Log:13>(msg=\"hello Aigerim\") (line 14)\n"
                                     "  at greet() (line 3)\n  at main() (line 9)\n  at __start__()\n")
        assert stops(answer(out, "n")) == [('<clause', '15')] or "at line 15" in answer(out, "n")
        assert answer(out, "n", 1).startswith("hello Aigerim\n")
        assert answer(out, "p count", 1) == "count = 1\n"
        # line 22 has code twice: main makes the clause, the clause runs (after the abort: no resume)
        assert answer(out, "c", 1).endswith("Breakpoint 2, main at line 22\n"
                                            "   22 |         Fail(why) => \"failed: \" + why\n")
        assert answer(out, "c", 2).startswith("Breakpoint 2, <clause Fail:22> at line 22\n")
        assert answer(out, "p why") == "why = no\n"
        assert answer(out, "bt", 1) == "  at <clause Fail:22>(why=\"no\") (line 22)\n  at main() (line 19)\n" \
                                       "  at __start__()\n"
        assert answer(out, "rc").startswith("Breakpoint 2, main at line 22\n")
        assert answer(out, "rc", 1).startswith("Breakpoint 1, <clause Log:13> at line 14\n")
        assert answer(out, "back").startswith("greet at line 3\n")           # back through the perform
        program = [ln for ln in out.split("\n") if ln in ("hello Aigerim", "done", "failed: no", "1")]
        assert program == ["hello Aigerim", "done", "failed: no", "1"], out
        assert out.endswith("failed: no\n1\nProgram exited with code 0\n(til) q\n")

    def test_retry_and_closures(self):
        src = """\
flaky(n: int) -> int
    if n < 3
        panic("not yet")
    return n * 10

main()
    var tries = 0
    let v = retry(5)
        tries += 1
        flaky(tries)
    else
        -1
    let k = 2
    let scaled = [1, 2, 3].map(|x| x * k + v)
    print(scaled)
"""
        rc, out, err = debug(src, "b 4; b 14; c; p n; bt; c; p v; s; p x; p k; locals; bt; back; back; p tries; q")
        assert rc == 0, err
        assert answer(out, "p n") == "n = 3\n"
        assert answer(out, "p v") == "v = 30\n"
        assert stops(answer(out, "s")) == [('<lambda:14>', '14')]       # through the prelude's map
        assert answer(out, "p x") == "x = 1\n" and answer(out, "p k") == "k = 2\n"
        assert answer(out, "locals") == "x = 1\nk = 2\nv = 30\n"
        assert "  at __prelude_map(" in answer(out, "bt", 1)
        assert answer(out, "p tries") == "tries = 3\n"

    def test_graphics_and_keys(self):
        src = """\
main()
    screen_init(8, 6)
    screen_clear(0x102030)
    var x = 0
    while true
        let k = wait_key()
        if k == 27
            break
        if k == 4
            x += 1
        draw_pixel(x, 2, 0xFFFFFF)
    print(get_pixel(2, 2))
"""
        path = write("gfx.til", src)
        run_ppm, dbg_ppm = os.path.join(TMP, "run.ppm"), os.path.join(TMP, "dbg.ppm")
        r = cli('run', '--vm', path, '--keys', 'RIGHT,RIGHT,a,RIGHT,ESC', '--screen', run_ppm)
        assert r.returncode == 0, r.stderr
        r2 = cli('debug', path, '--keys', 'RIGHT,RIGHT,a,RIGHT,ESC', '--screen', dbg_ppm, '--commands',
                 'b 11; c; c; p x; p k; back; back; back; p k; rc; p k; rc; d 11; c; back; back; c; q')
        assert r2.returncode == 0, r2.stderr
        assert answer(r2.stdout, "p x") == "x = 2\n"
        assert answer(r2.stdout, "p k") == "k = 4\n"
        assert answer(r2.stdout, "p k", 2) == "k = 4\n"          # the first key again
        assert r2.stdout.count(r.stdout) == 1
        with open(run_ppm, 'rb') as f1, open(dbg_ppm, 'rb') as f2:
            assert f1.read() == f2.read()

    def test_unfused_bytecode_debugs_the_same(self):
        cmds = "b 11; c; c; p sq; p total; s; s; p s; back; back; back; p i; q"
        a = debug(LOOP, cmds)
        b = debug(LOOP, cmds, extra=('--no-fuse',))
        assert a == b and a[0] == 0

    def test_tbc_with_and_without_debug_info(self):
        src_path = write("built.til", LOOP)
        tbc = os.path.join(TMP, "built.tbc")
        r = cli('build', '--target', 'vm', '--debug-info', src_path, '-o', tbc)
        assert r.returncode == 0, r.stderr
        r = cli('debug', tbc, '--commands', 'b 11; c; p sq; p total; q')
        assert r.returncode == 0, r.stderr
        assert "sq = 0" in r.stdout and "total = 0" in r.stdout
        assert "   11 |         total = add(total, sq)" in r.stdout        # source found next to it
        plain = os.path.join(TMP, "plain.tbc")
        r = cli('build', '--target', 'vm', src_path, '-o', plain)
        r = cli('debug', plain, '--commands', 'b 11; c; p sq; p $2; p g0; q')
        assert r.returncode == 0, r.stderr
        assert "No variable 'sq' here" in r.stdout                    # no names without the section
        assert "$2 = 0" in r.stdout and "g0 = 0" in r.stdout          # slots and globals by number


# ═══════════════════════════════════════════════════════════════════════════
# debug info in the bytecode
# ═══════════════════════════════════════════════════════════════════════════

class TestDebugInfo:
    def test_round_trip(self):
        m = til.compile_to_bytecode(LOOP, "loop.til", debug_info=True)
        data = m.serialize()
        m2 = til.TBCModule.from_bytes(data)
        assert m2.debug_info and m2.serialize() == data
        assert m2.globals == ["total", "names"]
        names = {f.name: sorted((r[1], r[0]) for r in f.locals) for f in m2.funcs}
        assert names["add"] == [("a", 0), ("b", 1), ("s", 2)]
        assert names["main"] == [("i", 0), ("sq", 2)]
        for f in m2.funcs:
            for slot, name, start, end in f.locals:
                assert 0 <= start < end <= len(f.code), (f.name, name)
                assert slot < f.n_locals
        # the section: n u16, then 13-byte records, right after the line table
        recs = m2.debug_records()
        assert data.endswith(b"".join(
            til._struct.pack('<HBHII', fn, slot, m2.const(name), s, e) for fn, slot, name, s, e in recs))
        assert (til.DEBUG_GLOBAL, 0, "names", 1, 1) in recs

    def test_default_bytecode_unchanged(self):
        plain = til.compile_to_bytecode(LOOP, "loop.til").serialize()
        assert not til.TBCModule.from_bytes(plain).debug_info
        dbg = til.compile_to_bytecode(LOOP, "loop.til", debug_info=True)
        # the same code and line table; the section and the names come after
        assert dbg.code == til.TBCModule.from_bytes(plain).code
        assert len(dbg.serialize()) > len(plain)

    def test_strip_omits_the_section(self):
        dbg = til.compile_to_bytecode(LOOP, "loop.til", debug_info=True).serialize(strip=True)
        plain = til.compile_to_bytecode(LOOP, "loop.til").serialize(strip=True)
        assert dbg == plain
        src_path = write("strip.til", LOOP)
        a, b = os.path.join(TMP, "s1.tbc"), os.path.join(TMP, "s2.tbc")
        assert cli('build', '--target', 'vm', '--debug-info', '--strip', src_path, '-o', a).returncode == 0
        assert cli('build', '--target', 'vm', '--strip', src_path, '-o', b).returncode == 0
        with open(a, 'rb') as f1, open(b, 'rb') as f2:
            assert f1.read() == f2.read()

    def test_dis_shows_names(self):
        r = cli('dis', '--debug-info', write("dis.til", LOOP))
        assert r.returncode == 0, r.stderr
        assert "; global #0 total" in r.stdout
        assert re.search(r";   local sq: slot 2, live \d{4}-\d{4}", r.stdout)
        plain = cli('dis', write("dis2.til", LOOP)).stdout
        assert ";   local " not in plain and "; global" not in plain

    def test_live_ranges_after_superinstructions(self):
        # `let s = a + b` ends with ADD_STORE_LOCAL; `return s` is RETURN_LOCAL: s must be
        # live at RETURN_LOCAL (its range ended inside the fused pair)
        m = til.compile_to_bytecode(LOOP, "loop.til", debug_info=True)
        add = next(f for f in m.funcs if f.name == "add")
        assert any(op == til.Op.RETURN_LOCAL for op in add.code)
        s = next(r for r in add.locals if r[1] == "s")
        ret = add.code.index(til.Op.RETURN_LOCAL)
        assert s[2] <= ret < s[3]

    def test_globals_past_slot_255(self):
        src = "".join(f"var g{i} = {i}\n" for i in range(300)) + "main()\n    g299 += 1\n    print(g299)\n"
        m = til.compile_to_bytecode(src, "many.til", debug_info=True)
        m2 = til.TBCModule.from_bytes(m.serialize())
        assert m2.globals[299] == "g299" and m2.globals[256] == "g256"
        rc, out, err = debug(src, "b 302; c; p g299; p g256; n; p g299; q")
        assert rc == 0, err
        assert answer(out, "p g299") == "g299 = 299\n" and answer(out, "p g256") == "g256 = 256\n"
        assert answer(out, "p g299", 1) == "g299 = 300\n"

    def test_upvalue_names(self):
        src = "make(k)\n    return |x| x + k\n\nmain()\n    print(make(2)(3))\n"
        m = til.TBCModule.from_bytes(til.compile_to_bytecode(src, "up.til", debug_info=True).serialize())
        lam = next(f for f in m.funcs if f.name.startswith("<lambda"))
        assert lam.upval_names == ["k"]

    def test_bad_sections_are_reported(self):
        data = til.compile_to_bytecode(LOOP, "loop.til", debug_info=True).serialize()
        with pytest.raises(ValueError, match="truncated debug-info section"):
            til.TBCModule.from_bytes(data[:-3])

    @needs_cc
    def test_c_vm_runs_files_with_the_section(self):
        for src in (LOOP, FACT, EFFECTS):
            plain = til.compile_to_bytecode(src, "p.til").serialize()
            dbg = til.compile_to_bytecode(src, "p.til", debug_info=True).serialize()
            assert dbg != plain
            for profile in ("host", "int32"):
                rc1, out1, err1 = run_cvm(dbg, profile=profile)
                rc2, out2, _ = run_cvm(plain, profile=profile)
                assert (rc1, out1) == (rc2, out2) and rc1 == 0, err1
            assert run_vm(dbg) == run_vm(plain)

    @needs_cc
    def test_selfhost_output_without_the_section_loads_and_debugs(self):
        tilc_src = os.path.join(ROOT, 'selfhost', 'tilc.til')
        with open(tilc_src, encoding='utf-8') as f:
            tilc = til.compile_to_bytecode(f.read(), tilc_src).serialize()
        tilc_path = os.path.join(TMP, "tilc.tbc")
        with open(tilc_path, 'wb') as f:
            f.write(tilc)
        prog = write("sh.til", FACT)
        out_tbc = os.path.join(TMP, "sh.tbc")
        vm = cvm_binary("host", ('-DTILVM_HEAP_SIZE=4194304',))
        r = subprocess.run([vm, tilc_path, prog, out_tbc], capture_output=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        with open(out_tbc, 'rb') as f:
            data = f.read()
        m = til.TBCModule.from_bytes(data)
        assert not m.debug_info and m.lines
        r = cli('debug', out_tbc, '--commands', 'b 4; c; p $0; bt; c; c; c; c; q')
        assert r.returncode == 0, r.stderr
        assert "$0 = 4" in r.stdout and "24\nProgram exited with code 0" in r.stdout


# ═══════════════════════════════════════════════════════════════════════════
# the debug path and the state copier
# ═══════════════════════════════════════════════════════════════════════════

class TestInternals:
    def test_normal_execute_has_only_the_marker(self):
        import inspect
        src = inspect.getsource(til.TILVM.execute)
        assert src.count("# @line-event") == 1
        assert "dbg_" not in src                    # nothing of the debugger in the fast loop
        code = tildebug.debug_execute_code()
        assert any('dbg_event' in c.co_varnames for c in code.co_consts if hasattr(c, 'co_varnames'))

    def test_copy_state_keeps_sharing(self):
        shared = [1, 2]
        cell = til.VMCell(shared)
        clo = til.VMClosure(3, (cell,))
        st = til.VMStruct(4, [shared, clo, til._UNDEF])
        state = [[shared, st, {b"k": shared}], [cell, clo]]
        c, size = tildebug.copy_state(state)
        a, b, m = c[0]
        assert a == [1, 2] and a is not shared
        assert b.f[0] is a and m[b"k"] is a                        # still one array
        assert b.f[1] is c[1][1] and c[1][1].up[0] is c[1][0]     # one closure, one cell
        assert c[1][0].v is a
        assert b.f[2] is til._UNDEF
        a.append(3)
        assert shared == [1, 2]

    def test_copy_state_long_chains(self):
        node = None
        for i in range(50000):
            node = til.VMStruct(5, [i, node])
        c, size = tildebug.copy_state([node])
        n, k = c[0], 0
        while n is not None:
            k += 1
            n = n.f[1]
        assert k == 50000

    def test_debug_vm_matches_the_normal_vm(self):
        for name in ("fib", "closures", "structs"):
            with open(os.path.join(ROOT, 'benchmarks', name + '.til'), encoding='utf-8') as f:
                src = f.read()
            if name == "fib":
                src = src.replace("27", "16")
            path = write(name + ".til", src)
            module, _ = tildebug.load_module(path)
            out = []
            vm = tildebug.DebugVM(module, out.append, lambda: b"")
            rc = vm.resume(lambda ev, line, depth, top: False)
            assert (rc, b"".join(out).decode()) == run_vm(til.compile_to_bytecode(src, path).serialize())
