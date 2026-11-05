#!/usr/bin/env python3
"""
TIL 3.1 #11: battery saving (docs/dev/TIL_3_1_DESIGN.md section 11, docs/TIL_ENERGY.md).

* `#[energy: low]` directly above a function sets TBCFunction.FLAG_ENERGY_LOW;
  a top-level `#[energy: low|balanced|performance]` pragma before every
  definition sets the program's initial mode (bits 2-3 of the .tbc header
  flags byte); both compilers (src/til.py, selfhost/tilc.til) agree.
* energy_mode() / energy_mode(m) (native 54) behave identically on both VMs.
* Both VMs idle 1 ms at every step that is a multiple of 512 (mode 2, or
  while an energy-low function is active) or of 4096 (mode 1): the idle
  counters (TILVM.idle_count, `tilvm --stats`) are equal, and equal to a
  model of the rule computed from a step trace.
* A CPU-bound loop in mode 2 uses much less CPU per wall second than in
  mode 0 (skipped with TIL_SKIP_TIMING=1); outputs are identical; timeouts
  fire on time while the VM idles; sleep_ms() idles instead of spinning.

Run: python3 -m pytest tests/test_energy.py -v
"""

import os
import re
import signal
import subprocess
import sys
import tempfile
import time

import pytest

from tilhelpers import CC, ROOT, TIL, VM_DIR, cli, cvm_binary, expect_compile_error, needs_cc, til

FLAG_LOW = til.TBCFunction.FLAG_ENERGY_LOW
SKIP_TIMING = bool(os.environ.get('TIL_SKIP_TIMING'))
needs_timing = pytest.mark.skipif(SKIP_TIMING, reason="TIL_SKIP_TIMING=1")
NO_ENERGY = ('-DTILVM_NO_ENERGY',)
EXAMPLE = os.path.join(ROOT, 'examples', '14_energy.til')
BAD_MODE = "energy_mode() needs a mode 0 (performance), 1 (balanced) or 2 (low)"


def build(src: str, filename: str = "<test>", fuse: bool = True) -> bytes:
    return til.compile_to_bytecode(src, filename, fuse=fuse).serialize()


def module_of(src: str) -> 'til.TBCModule':
    return til.TBCModule.from_bytes(build(src))


def low_funcs(src: str):
    return sorted(f.name for f in module_of(src).funcs if f.flags & FLAG_LOW)


class TraceVM(til.TILVM):
    """The reference VM without real sleeps, plus an independent model of the
    idle rule: at step s, the effective mode is 2 while any frame belongs to an
    #[energy: low] function, else the program's mode; it idles when s is a
    multiple of 512 (mode 2) or 4096 (mode 1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_idles = 0
        self.slept = 0

    def idle(self, ms):
        self.slept += ms

    def check_budget(self, pc):
        s = self.steps + 1
        low = any(self.m.funcs[fr[0]].flags & FLAG_LOW for fr in self.frames)
        mode = 2 if low else self.energy
        period = (0, 4096, 512)[mode]
        if period and s % period == 0 and not (self.max_steps and s > self.max_steps):
            self.model_idles += 1
        super().check_budget(pc)


def py_run(data: bytes, argv=None, max_steps: int = 0, vm_class=TraceVM):
    """(exit code, stdout, error report, steps, idles, model idles, final mode)"""
    module = til.TBCModule.from_bytes(data)
    out, err = [], []
    vm = vm_class(module, write=out.append, argv=argv, max_steps=max_steps)
    vm.report = lambda msg, pc: err.append(msg + b"\n" + (vm.traceback(pc) + b"\n" if pc >= 0 else b""))
    rc = vm.run()
    return (rc, b"".join(out).decode('utf-8', 'replace'), b"".join(err).decode('utf-8', 'replace'),
            vm.steps, vm.idle_count, getattr(vm, 'model_idles', None), vm.energy)


def c_run(data: bytes, profile: str = "host", extra=(), vm_args=(), prog_args=(), timeout: int = 120):
    """C VM with --stats: (exit code, stdout, stderr without the stats lines, steps, idles, mode)"""
    fd, path = tempfile.mkstemp(suffix='.tbc')
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    try:
        r = subprocess.run([cvm_binary(profile, extra), '--stats', *vm_args, path, *prog_args],
                           capture_output=True, timeout=timeout)
    finally:
        os.unlink(path)
    err = r.stderr.decode('utf-8', 'replace')
    steps = re.search(r'\[tilvm\] steps=(\d+)', err)
    energy = re.search(r'\[tilvm\] energy_mode=(\d+) idle=(\d+)', err)
    err = re.sub(r'\[tilvm\] [^\n]*\n', '', err)
    return (r.returncode, r.stdout.decode('utf-8', 'replace'), err, int(steps.group(1)) if steps else None,
            int(energy.group(2)) if energy else 0, int(energy.group(1)) if energy else 0)


# ═══════════════════════════════════════════════════════════════
# SYNTAX: the attribute, the pragma, the header bits
# ═══════════════════════════════════════════════════════════════

LOOP = "var s = 0\nfor i in 0..100\n    s += i\nprint(s)\n"


class TestSyntax:
    def test_attribute_sets_the_function_flag(self):
        src = ("#[energy: low]\nwork(n: int) -> int\n    return n * 2\n\n"
               "fast(n: int) -> int\n    return n + 1\n\nprint(work(2) + fast(3))\n")
        assert low_funcs(src) == ['work']
        assert module_of(src).energy == 0

    def test_every_kind_of_function(self):
        src = ("#[energy: low]\npub a() -> int\n    return 1\n"
               "#[energy: low]\nfn b() -> int\n    return 2\n"
               "#[energy: low]\n#[pure]\nc() -> int\n    return 3\n"
               "#[pure]\n#[energy: low]\nd() -> int\n    return 4\n"
               "#[energy: low]\n# a comment line between them is fine\ne() -> int\n    return 5\n"
               "#[energy: low] f() -> int\n    return 6\n"
               "struct P\n    x: int\nimpl P\n    #[energy: low]\n    get(self) -> int\n        return self.x\n"
               "#[energy: low]\nmain()\n    print(a() + b() + c() + d() + e() + f() + P { x: 7 }.get())\n")
        assert low_funcs(src) == ['P.get', 'a', 'b', 'c', 'd', 'e', 'f', 'main']

    def test_legacy_levels_mark_nothing_for_the_vm(self):
        # TIL 3.0: medium / high only steer the native backend
        src = "#[energy: high]\nf() -> int\n    return 1\n#[energy: medium]\ng() -> int\n    return 2\nprint(f() + g())\n"
        assert low_funcs(src) == [] and module_of(src).energy == 0

    @pytest.mark.parametrize("name, mode", [('performance', 0), ('balanced', 1), ('low', 2)])
    def test_pragma_sets_the_header_bits(self, name, mode):
        src = f"#[energy: {name}]\n" + LOOP
        data = build(src)
        plain = build("# (the same line numbers)\n" + LOOP)
        assert data[5] == plain[5] | (mode << 2) and (data[5] >> 2) & 3 == mode
        assert data[:5] + data[6:] == plain[:5] + plain[6:], "only the flags byte differs"
        assert til.TBCModule.from_bytes(data).energy == mode
        stripped = til.compile_to_bytecode(src, "<test>").serialize(strip=True)
        assert stripped[5] == mode << 2
        floats = build(f"#[energy: {name}]\nprint(1.5)\n")
        assert floats[5] == 1 | 2 | (mode << 2)

    def test_where_the_pragma_goes(self):
        # an empty line, a statement or another item after it: the program's mode
        assert module_of("#[energy: low]\n\nmain()\n    print(1)\n").energy == 2
        assert low_funcs("#[energy: low]\n\nmain()\n    print(1)\n") == []
        assert module_of("#[energy: low]\n# about it\n\nf() -> int\n    return 1\nprint(f())\n").energy == 2
        assert module_of("#[energy: low]\nstruct P\n    x: int\nprint(P { x: 1 }.x)\n").energy == 2
        assert module_of("print(0)\n#[energy: balanced]\nprint(1)\n").energy == 1
        # balanced / performance never mark a function: directly above one they are the pragma
        assert module_of("#[energy: balanced]\nmain()\n    print(1)\n").energy == 1
        assert module_of("#[energy: performance]\nmain()\n    print(1)\n").energy == 0
        # the pragma and a function attribute together
        src = "#[energy: balanced]\n\n#[energy: low]\nwork() -> int\n    return 5\nprint(work())\n"
        assert module_of(src).energy == 1 and low_funcs(src) == ['work']

    def test_misplaced_or_unknown(self):
        expect_compile_error("f() -> int\n    return 1\n#[energy: low]\n\nprint(f())\n",
                             "must be directly above a function, or come before every definition")
        expect_compile_error("f() -> int\n    return 1\n#[energy: balanced]\ng() -> int\n    return 2\n",
                             "must be directly above a function")
        expect_compile_error("import math\n#[energy: low]\nprint(1)\n", "come before every definition")
        expect_compile_error("#[energy: turbo]\nprint(1)\n",
                             "unknown energy mode 'turbo' (use low, balanced or performance)")
        expect_compile_error("#[energy: high]\n\nprint(1)\n", "unknown energy mode 'high'")
        expect_compile_error("#[energy: lwo]\nf() -> int\n    return 1\nprint(f())\n",
                             "unknown energy level 'lwo' for a function (use #[energy: low])")
        expect_compile_error("struct P\n    x: int\nimpl P\n    #[energy: balanced]\n    get(self) -> int\n"
                             "        return self.x\nprint(P { x: 1 }.get())\n", "unknown energy level 'balanced'")
        expect_compile_error("#[energy: low]\nprint(1)\n#[energy: low]\nprint(2)\n",
                             "the program's energy mode is set twice")
        expect_compile_error("#[energy: low]\n#[energy: balanced]\nprint(1)\n", "set twice")

    def test_a_modules_pragma_is_not_the_programs(self, tmp_path):
        (tmp_path / "helper.til").write_text("#[energy: low]\n\n#[energy: low]\nslow(n: int) -> int\n"
                                             "    return n + 1\n", encoding='utf-8')
        main = tmp_path / "main.til"
        main.write_text("import helper\nprint(slow(1))\nprint(energy_mode())\n", encoding='utf-8')
        m = til.compile_to_bytecode(main.read_text(encoding='utf-8'), str(main))
        assert m.energy == 0 and [f.name for f in m.funcs if f.flags & FLAG_LOW] == ['slow']

    def test_disassembler(self, tmp_path):
        src = "#[energy: low]\n\n#[energy: low]\nwork() -> int\n    return 1\nprint(work())\n"
        prog = tmp_path / "d.til"
        prog.write_text(src, encoding='utf-8')
        tbc = tmp_path / "d.tbc"
        r = cli('build', '--target', 'vm', str(prog), '-o', str(tbc))
        assert r.returncode == 0, r.stdout + r.stderr
        r = cli('dis', str(tbc))
        assert r.returncode == 0, r.stderr
        assert "; energy mode: low (2, #[energy: low])" in r.stdout
        assert re.search(r'fn #\d+ work .*\[energy-low\]', r.stdout)
        assert "energy mode" not in til.disassemble(module_of(LOOP))

    @needs_cc
    def test_bad_header_bits_are_rejected(self, tmp_path):
        data = bytearray(build(LOOP))
        data[5] |= 3 << 2
        with pytest.raises(ValueError, match=r"bad bytecode header \(energy mode 3\)"):
            til.TBCModule.from_bytes(bytes(data))
        path = tmp_path / "bad.tbc"
        path.write_bytes(bytes(data))
        r = subprocess.run([cvm_binary(), str(path)], capture_output=True, text=True)
        assert r.returncode == 2 and "error -5" in r.stderr
        # a build without energy saving ignores the bits
        r = subprocess.run([cvm_binary("host", NO_ENERGY), str(path)], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout == "4950\n"


# ═══════════════════════════════════════════════════════════════
# energy_mode(): the same on both VMs
# ═══════════════════════════════════════════════════════════════

MODE_API = """
#[energy: balanced]

#[energy: low]
inner() -> int
    # the program's mode (not the effective one); it can be changed here
    let before = energy_mode()
    let prev = energy_mode(0)
    let after = energy_mode()
    energy_mode(prev)
    return before * 100 + prev * 10 + after

main()
    print(energy_mode())
    print(energy_mode(2))
    print(energy_mode())
    print(f"{energy_mode(0)} {energy_mode(1)} {energy_mode(2)} {energy_mode(1)}")
    print(inner())
    print(energy_mode())
    for bad in [3, -1, 2.0, "2", true, None, [1]]
        let r = retry(1)
            energy_mode(bad)
            "accepted"
        else
            error
        print(r)
    print(energy_mode())
    energy_mode(7)
    print("not reached")
"""
MODE_API_OUT = ("1\n1\n2\n2 0 1 2\n110\n1\n" + (BAD_MODE + "\n") * 7 + "1\n")
MODE_API_LINE = MODE_API.split("\n").index("    energy_mode(7)") + 1


class TestEnergyMode:
    def test_python_vm(self):
        rc, out, err, *_rest = py_run(build(MODE_API))
        assert (rc, out) == (1, MODE_API_OUT)
        assert err.startswith("runtime error: " + BAD_MODE + "\n") and f"(line {MODE_API_LINE})" in err

    @needs_cc
    @pytest.mark.parametrize("profile", ["host", "int32", "small"])
    def test_c_vm_is_identical(self, profile):
        data = build(MODE_API)
        py = py_run(data)
        c = c_run(data, profile)
        assert c[:3] == py[:3]
        assert c[5] == py[6] == 1, "the final mode"

    @needs_cc
    def test_lean_build(self):
        src = MODE_API.replace('2.0, ', '')        # no floats in the 16-bit lean build
        data = build(src)
        py = py_run(data)
        assert c_run(data, "tiny")[:3] == py[:3]

    @needs_cc
    def test_build_without_energy_saving(self):
        rc, out, err, *_ = c_run(build("print(1)\nprint(energy_mode())\n"), "host", NO_ENERGY)
        assert (rc, out) == (1, "1\n") and "energy_mode() is not available in this VM build" in err
        # programs that only ask for energy saving run unchanged, at full speed
        src = "#[energy: low]\n\n#[energy: low]\nf(n: int) -> int\n    var s = 0\n    for i in 0..n\n" \
              "        s += i\n    return s\nprint(f(5000))\n"
        c = c_run(build(src), "host", NO_ENERGY)
        assert c[:3] == (0, "12497500\n", "") and c[4] == 0
        assert c_run(build(src))[4] == py_run(build(src))[4] > 0


    @needs_cc
    def test_z80_profile_leaves_energy_saving_out(self):
        """vm/ports/z80/Makefile builds with -DTILVM_NO_ENERGY (the image would not fit
        below 0xC000 with it): the desktop build of that profile behaves the same"""
        import test_z80
        assert '-DTILVM_NO_ENERGY' in test_z80.Z80_PROFILE
        flags = tuple(test_z80.HOSTED)
        rc, out, err, *_ = c_run(build("print(1)\nprint(energy_mode())\n"), "host", flags)
        assert (rc, out) == (1, "1\n") and "energy_mode() is not available in this VM build" in err
        src = "#[energy: low]\n\nvar s = 0\nfor i in 0..3000\n    s += i % 5\nprint(s)\n"
        assert c_run(build(src), "host", flags)[:3] == (0, "6000\n", "")


# ═══════════════════════════════════════════════════════════════
# IDLES: exactly per the rule, the same count on both VMs
# ═══════════════════════════════════════════════════════════════

IDLE_PROGRAMS = {
    'low': "#[energy: low]\n\nvar s = 0\nfor i in 0..6000\n    s += i % 3\nprint(s)\n",
    'balanced': "#[energy: balanced]\n\nvar s = 0\nfor i in 0..30000\n    s += i % 3\nprint(s)\n",
    'performance': "#[energy: performance]\n\nvar s = 0\nfor i in 0..6000\n    s += i % 3\nprint(s)\n",
    'switching': """
var s = 0
for i in 0..12000
    if i % 700 == 0
        energy_mode((i / 700) % 3)
    s += i % 5
print(s, energy_mode())
""",
    'low_function': """
#[energy: low]
fib(n: int) -> int
    if n < 2
        return n
    return fib(n - 1) + fib(n - 2)

#[energy: low]
background(n: int) -> int
    var h = 7
    for i in 0..n
        h = (h * 31 + i) % 10007
    return h

fast(n: int) -> int
    var t = 0
    for i in 0..n
        t += i % 7
    return t

main()
    var total = 0
    for k in 0..6
        total += background(700) + fast(900)
    print(total, fib(14))
""",
    'closures_and_lambdas': """
#[energy: low]
make(k: int)
    # the lambda is not energy-low itself: it saves energy only while make() runs
    let add = |x: int| x + k
    var t = 0
    for i in 0..800
        t = add(t) % 1000
    print(t)
    return add

main()
    let f = make(3)
    var u = 0
    for i in 0..3000
        u = f(u) % 1000
    print(u)
""",
    'methods': """
struct Counter
    n: int

impl Counter
    #[energy: low]
    run(self, k: int) -> int
        var t = self.n
        for i in 0..k
            t += i % 3
        return t

main()
    let c = Counter { n: 1 }
    var t = 0
    for j in 0..5
        t += c.run(1500)
    print(t)
""",
    'effects': """
#[energy: low]
produce(n: int)
    for i in 0..n
        perform Emit(i)

main()
    var total = 0
    handle
        produce(2500)
    with
        Emit(x) =>
            total += x % 9
            resume()
    print(total)
    let first = handle
        produce(5000)
        -1
    with
        Emit(x) =>
            if x != 1300
                resume()
            x
    print(first)
""",
    'unwinding': """
#[energy: low]
deep(n: int) -> int
    var t = 0
    for i in 0..n
        t += 1
    if n > 0
        panic("stop " + str(n))
    return t

main()
    var caught = 0
    for k in 0..8
        let r = retry(1)
            deep(400 + k)
        else
            caught += 1
            0
    var t = 0
    for i in 0..5000
        t += i % 2
    print(caught, t, energy_mode())
""",
    'mode_inside_low': """
#[energy: low]
job() -> int
    energy_mode(2)
    var t = 0
    for i in 0..1000
        t += 1
    energy_mode(0)
    for i in 0..1000
        t += 1
    return t

main()
    print(job())
    var t = 0
    for i in 0..3000
        t += 1
    print(t, energy_mode())
""",
}


def assert_idles_agree(src: str, profiles=("host",), max_steps: int = 0):
    data = build(src)
    py = py_run(data, max_steps=max_steps)
    assert py[4] == py[5], f"reference VM idled {py[4]} times, the rule says {py[5]}"
    unfused = py_run(build(src, fuse=False), max_steps=max_steps)
    assert unfused[:2] == py[:2] and unfused[3:5] == py[3:5]
    if CC:
        args = ('--max-steps', str(max_steps)) if max_steps else ()
        for profile in profiles:
            c = c_run(data, profile, vm_args=args)
            assert (c[0], c[1], c[3], c[4]) == (py[0], py[1], py[3], py[4]), \
                f"{profile}: C VM (rc, out, steps, idles) {c} != reference VM {py}"
    return py


class TestIdleCounts:
    @pytest.mark.parametrize("name", sorted(IDLE_PROGRAMS))
    def test_both_vms_idle_exactly_per_the_rule(self, name):
        py = assert_idles_agree(IDLE_PROGRAMS[name], ("host", "int32"))
        assert py[0] == 0, py
        if name == 'performance':
            assert py[4] == 0
        else:
            assert py[4] > 0, f"{name}: no idles"

    def test_simple_counts(self):
        # mode 2 from the start: one idle per 512 steps; mode 1: per 4096
        py = py_run(build(IDLE_PROGRAMS['low']))
        assert py[4] == py[3] // 512 and py[3] > 6000
        py = py_run(build(IDLE_PROGRAMS['balanced']))
        assert py[4] == py[3] // 4096 and py[3] > 30000

    @needs_cc
    @pytest.mark.parametrize("name", ['low', 'switching', 'low_function', 'mode_inside_low'])
    def test_lean_build(self, name):
        src = IDLE_PROGRAMS[name]
        if name == 'low_function':
            src = src.replace('10007', '101').replace('fib(14)', 'fib(9)')
        assert_idles_agree(src, ("tiny",))

    def test_step_limit_comes_first(self):
        # the step that exceeds --max-steps fails before it could idle
        py = assert_idles_agree(IDLE_PROGRAMS['low'], ("host",), max_steps=1024)
        assert py[0] == 1 and "step limit exceeded" in py[2] and py[4] == 2

    @needs_cc
    def test_the_example(self):
        with open(EXAMPLE, encoding='utf-8') as f:
            src = f.read()
        py = assert_idles_agree(src, ("host", "int32"))
        assert py[0] == 0 and py[4] > 0 and py[6] == 2
        assert "mode at start: balanced" in py[1] and "refused: " + BAD_MODE in py[1]

    @needs_cc
    def test_debugger_counts_without_sleeping(self, tmp_path):
        tildebug = til.feature_module('tildebug')
        path = tmp_path / "dbg.til"
        path.write_text(IDLE_PROGRAMS['low_function'], encoding='utf-8')
        module, _ = tildebug.load_module(str(path))
        out = []
        vm = tildebug.DebugVM(module, out.append, lambda: b"", every=1)
        start = time.perf_counter()
        assert vm.resume(lambda ev, line, depth, top: False) == 0
        py = py_run(build(IDLE_PROGRAMS['low_function']))
        assert (b"".join(out).decode(), vm.steps, vm.idle_count) == (py[1], py[3], py[4])
        assert 'energy' in vm.STATE_NUMBERS and 'idle_count' in vm.STATE_NUMBERS
        assert vm.travel(0) is None and vm.idle_count == 0 and vm.steps < 5
        assert vm.resume(lambda ev, line, depth, top: False) == 0
        assert vm.idle_count == py[4]
        assert time.perf_counter() - start < py[4] / 1000.0 + 30


# ═══════════════════════════════════════════════════════════════
# SNAPSHOTS: the mode is program state
# ═══════════════════════════════════════════════════════════════

@needs_cc
def test_raw_snapshot_keeps_the_mode(tmp_path):
    src = ("var s = 0\nenergy_mode(2)\nfor i in 0..4000\n    s += i\nprint(f\"{s} {energy_mode()}\")\n"
           "energy_mode(1)\nfor i in 0..9000\n    s += 1\nprint(f\"{s} {energy_mode()}\")\n")
    data = build(src)
    full = c_run(data)
    assert full[1] == "7998000 2\n8007000 1\n" and full[4] > 0
    snap = str(tmp_path / "s.snap")
    first = c_run(data, vm_args=('--snapshot-after', '3000', snap))
    assert first[0] == 3 and first[1] == ""
    second = c_run(data, vm_args=('--resume', snap))
    assert second[1] == full[1], "the resumed program is still in mode 2"
    assert full[4] - second[4] == 3000 // 512, "idles of the two parts = idles of one run"


# ═══════════════════════════════════════════════════════════════
# TIME: less CPU, timeouts on time, sleeping without spinning
# ═══════════════════════════════════════════════════════════════

BUSY = "energy_mode(int(args()[0]))\nvar s = 0\nfor i in 0..{n}\n    s += i % 7\nprint(s)\n"


def cpu_per_wall(cmd, timeout=120):
    """(stdout, child CPU seconds / wall seconds, wall seconds)"""
    t0 = time.monotonic()
    with tempfile.TemporaryFile() as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.DEVNULL)
        _pid, _status, usage = os.wait4(p.pid, 0)
        wall = time.monotonic() - t0
        out.seek(0)
        text = out.read().decode()
    return text, (usage.ru_utime + usage.ru_stime) / wall, wall


@needs_timing
@pytest.mark.skipif(not hasattr(os, 'wait4'), reason="needs os.wait4 (getrusage of the child)")
class TestTiming:
    @needs_cc
    def test_c_vm_mode_low_uses_less_cpu(self, tmp_path):
        tbc = tmp_path / "busy.tbc"
        tbc.write_bytes(build(BUSY.format(n=400000)))
        out0, r0, _ = cpu_per_wall([cvm_binary(), str(tbc), "0"])
        out2, r2, wall2 = cpu_per_wall([cvm_binary(), str(tbc), "2"])
        assert out0 == out2 == "1199997\n"
        assert wall2 > 0.5, "400000 steps in mode 2: about 780 idles of 1 ms"
        assert r2 < 0.5 and r2 < r0 / 3, f"CPU per wall second: mode 0 {r0:.2f}, mode 2 {r2:.2f}"

    def test_python_vm_mode_low_uses_less_cpu(self, tmp_path):
        tbc = tmp_path / "busy.tbc"
        tbc.write_bytes(build(BUSY.format(n=150000)))
        out0, r0, _ = cpu_per_wall([sys.executable, TIL, 'exec', str(tbc), "0"])
        out2, r2, _ = cpu_per_wall([sys.executable, TIL, 'exec', str(tbc), "2"])
        assert out0 == out2 == "449994\n"
        # 512 steps take the reference VM ~2 ms of CPU, then it sleeps 1 ms (other processes
        # on the machine lower both ratios)
        assert r2 < 0.85 and r2 < r0 * 0.9, f"CPU per wall second: mode 0 {r0:.2f}, mode 2 {r2:.2f}"

    TIMEOUT = ("#[energy: low]\n\nlet t0 = clock_ms()\nlet r = timeout(150)\n    var x = 0\n    while true\n"
               "        x += 1\n    x\nelse\n    -1\nlet took = clock_ms() - t0\n"
               "print(f\"{r} {took >= 150} {took < 1500}\")\n")

    @needs_cc
    def test_timeouts_fire_on_time_while_idling(self, tmp_path):
        data = build(self.TIMEOUT)
        t0 = time.monotonic()
        c = c_run(data)
        wall = time.monotonic() - t0
        assert c[:3] == (0, "-1 true true\n", "") and c[4] > 50, c
        assert wall < 3
        py = py_run(data, vm_class=til.TILVM)
        assert py[:3] == (0, "-1 true true\n", "") and py[4] > 10

    @needs_cc
    def test_sleep_idles_instead_of_spinning(self, tmp_path):
        src = "let t0 = clock_ms()\nsleep_ms(400)\nlet d = clock_ms() - t0\nprint(f\"{d >= 400} {d < 2000}\")\n"
        tbc = tmp_path / "sleep.tbc"
        tbc.write_bytes(build(src))
        out, ratio, wall = cpu_per_wall([cvm_binary(), str(tbc)])
        assert out == "true true\n" and wall >= 0.4
        assert ratio < 0.25, f"sleep_ms(400) used {ratio * wall:.3f} s of CPU"
        assert py_run(build(src), vm_class=til.TILVM)[1] == "true true\n"

    @needs_cc
    @pytest.mark.skipif(sys.platform == 'win32', reason="POSIX signals")
    @pytest.mark.parametrize("what", ["idle", "sleep"])
    def test_ctrl_c_while_waiting(self, tmp_path, what):
        body = "var s = 0\nwhile true\n    s += 1\n" if what == "idle" else "sleep_ms(20000)\n"
        tbc = tmp_path / "wait.tbc"
        tbc.write_bytes(build("#[energy: low]\n\nprint(\"start\")\n" + body))
        p = subprocess.Popen([cvm_binary(), str(tbc)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(0.4)
        t0 = time.monotonic()
        p.send_signal(signal.SIGINT)
        out, _err = p.communicate(timeout=10)
        assert time.monotonic() - t0 < 1.5
        assert p.returncode == -signal.SIGINT and out == b"start\n"


# ═══════════════════════════════════════════════════════════════
# THE SELF-HOSTED COMPILER: the same bytes, the same rejections
# ═══════════════════════════════════════════════════════════════

SELFHOST_PROGRAMS = [
    "#[energy: low]\nprint(energy_mode())\n",
    "#[energy: balanced]\n\nmain()\n    print(energy_mode())\n",
    "#[energy: performance]\nmain()\n    print(1)\n",
    "#[energy: low]\nmain()\n    print(energy_mode())\n",
    "#[energy: low]\n# comment\nf() -> int\n    return 1\nprint(f())\n",
    "#[energy: low]\n# comment\n\nf() -> int\n    return 1\nprint(f())\n",
    "#[energy: low]\n\n#[pure]\nf() -> int\n    return 1\nprint(f())\n",
    "#[energy: low]\npub f() -> int\n    return 1\n#[energy: low]\nfn g() -> int\n    return 2\nprint(f() + g())\n",
    "#[energy: low] f() -> int\n    return 1\nprint(f())\n",
    "#[energy: high]\nf() -> int\n    return 1\n#[energy: medium]\ng() -> int\n    return 1\nprint(f())\n",
    "struct P\n    x: int\nimpl P\n    #[energy: low]\n    get(self) -> int\n        return self.x\nprint(P { x: 3 }.get())\n",
    "print(0)\n#[energy: low]\nprint(energy_mode())\n",
    "#[ energy:   balanced ]\n\nprint(1.5)\n",
    "#[energy: balanced]\n\n#[energy: low]\nwork() -> int\n    return 5\nprint(energy_mode(), work())\n",
] + list(IDLE_PROGRAMS.values()) + [MODE_API]

SELFHOST_REJECTED = [
    "f() -> int\n    return 1\n#[energy: low]\n\nprint(f())\n",
    "f() -> int\n    return 1\n#[energy: balanced]\ng() -> int\n    return 1\nprint(f())\n",
    "#[energy: turbo]\nprint(1)\n",
    "#[energy: high]\n\nprint(1)\n",
    "#[energy: lwo]\nf() -> int\n    return 1\nprint(f())\n",
    "#[energy: low]\nprint(1)\n#[energy: low]\nprint(2)\n",
    "#[energy: low]\n#[energy: balanced]\nprint(1)\n",
    "struct P\n    x: int\nimpl P\n    #[energy: balanced]\n    get(self) -> int\n        return self.x\n",
]


@pytest.fixture(scope="module")
def tilc():
    """tilc.til compiled by the Python compiler, running on the C VM (stage 1;
    tests/test_selfhost.py shows stage 2 == stage 1)"""
    import test_selfhost
    vm = cvm_binary("host", test_selfhost.BIG_HEAP)
    with open(test_selfhost.TILC_SRC, encoding='utf-8') as f:
        stage1 = test_selfhost.py_compile(f.read(), test_selfhost.TILC_SRC)
    return test_selfhost.Tilc(stage1, vm)


@needs_cc
class TestSelfhost:
    def test_same_bytes(self, tilc):
        for src in SELFHOST_PROGRAMS:
            rc, out, data = tilc.compile(src)
            assert rc == 0, f"tilc failed: {out}\n{src}"
            assert data == build(src, "prog.til"), f"tilc differs from the Python compiler:\n{src}"

    def test_same_rejections(self, tilc):
        for src in SELFHOST_REJECTED:
            with pytest.raises((til.CompileError, SyntaxError)):
                build(src)
            rc, out, data = tilc.compile(src)
            assert rc == 1 and data is None and re.match(r'tilc: \S+:\d+: error: ', out), (src, out)

    def test_the_example(self, tilc):
        rc, out, data = tilc.compile_file(EXAMPLE, os.path.join(tilc.dir, 'ex.tbc'))
        with open(EXAMPLE, encoding='utf-8') as f:
            assert rc == 0 and data == build(f.read(), EXAMPLE), out


# ═══════════════════════════════════════════════════════════════
# THE INTERPRETER: energy saving rides on the fuel counter
# ═══════════════════════════════════════════════════════════════

C_SOURCE = open(os.path.join(VM_DIR, 'tilvm.c'), encoding='utf-8').read()


def test_no_work_per_instruction():
    """The only energy test in the interpreter loop is in the slow path of
    `backedge` (after the fuel counter ran out), and after energy_mode()."""
    loop = C_SOURCE[C_SOURCE.index('int tilvm_run_steps('):C_SOURCE.index('int tilvm_run(tilvm *vm)')]
    lean = loop[loop.index('backedge:'):loop.index('#else', loop.index('backedge:'))]   # lean: every step
    assert loop.count('energy_point(') == 2 and loop.count('ENERGY_ON(') == 2
    assert 'energy_point(' in lean
    slow = loop[loop.index('if (--fuel != 0 && !st->n_deadlines) NEXT;'):]
    assert 'energy_point(' in slow.split('stack_overflow:')[0]
    assert loop.count('N_ENERGY_MODE') == 1
    refuel = C_SOURCE[C_SOURCE.index('uint32_t refuel(const tilvm *vm, uint32_t budget) {'):]
    assert 'ENERGY_ON(vm)' in refuel[:refuel.index('return fuel;')]
    assert 'static TV_ALWAYS_INLINE uint32_t refuel(' in C_SOURCE


@needs_cc
def test_strict_builds_without_energy_saving(tmp_path):
    for cc in ('gcc', 'clang'):
        if not __import__('shutil').which(cc):
            continue
        for flags in ([], ['-DTILVM_LEAN', '-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0', '-DTILVM_NO_STDIO']):
            r = subprocess.run([cc, '-std=c99', '-Wall', '-Wextra', '-pedantic', '-Werror', '-DTILVM_NO_ENERGY',
                                *flags, '-c', os.path.join(VM_DIR, 'tilvm.c'), '-o', str(tmp_path / 'x.o')],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr


# ═══════════════════════════════════════════════════════════════
# THE PLAYGROUND: idling never pauses; sleep_ms still does
# ═══════════════════════════════════════════════════════════════

try:
    from test_playground import builds, compile_file, desktop_run, drive, needs_node, node_run  # noqa: F401
except Exception:                                   # pragma: no cover - the playground tests are optional
    builds = None
    needs_node = pytest.mark.skip(reason="tests/test_playground.py is not importable")


@needs_node
@needs_cc
def test_playground_energy_and_sleep(builds, tmp_path):
    """The real VM (no test patch): in mode 2 the VM idles at loop back-edges -
    the playground's idle hook never pauses there (the frame's saved pc is
    stale) - and sleep_ms() still pauses through its clock reads."""
    if not builds['fresh']:
        pytest.skip("no wasm toolchain")
    src = ('#[energy: low]\n\nf() -> int\n    print("start")\n    sleep_ms(5)\n    var n = 0\n    var one = 1\n'
           '    while n < 10000\n        n += one\n    return n\n'
           'print(f"n = {f()}")\nvar m = 0\nlet one = 1\nsleep_ms(3)\nfor i in 0..5000\n    m += one\n'
           'print(m)\nprint(energy_mode())\n')
    tbc = compile_file(src, str(tmp_path), 'energy')
    rc, wout, werr, wsteps = node_run(builds['wasm'], tbc)
    drc, dout, derr, dsteps = desktop_run(tbc)
    assert (rc, wout, werr) == (drc, dout, derr) == (0, 'start\nn = 10000\n5000\n2\n', '')
    assert wsteps == dsteps
    log = drive(builds['wasm'], '#[energy: low]\n\nprint("a")\nsleep_ms(400)\nprint("b")\n',
                [{}, {'wait': 420}], tmp_path)
    assert [e['st'] for e in log] == ['sleep', 'done'] and log[0]['ms'] < 300
