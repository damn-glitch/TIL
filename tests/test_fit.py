#!/usr/bin/env python3
"""
TIL 3.1 feature 10: `til fit` - will a program fit a device?

* device profiles (and the C VM struct-size model, checked against C compilers)
* static analysis: operand-stack depth by abstract interpretation, worst-case
  value-stack slots / call frames / recovery points, recursion, integer width,
  floats, lean builtins, tables, storage, allocations in loops
* static bounds compared with the high-water marks the C VM measures
  (TILVM_STATS) on a corpus: exact for programs without recursion
* --run on the C VM built with a device's limits; robustness against unknown
  opcodes; `til build --target vm --device NAME`

Run: python3 -m pytest tests/test_fit.py -v
Slower checks (more device builds): TIL_TEST_SLOW=1; cross compilers: TIL_TEST_CROSS=1
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from tilhelpers import CC, ROOT, VM_DIR, cli, compile_vm, cvm_binary, needs_cc, run_vm, tbc_bytes, til

fit = til.feature_module('tilfit')
slow = pytest.mark.skipif(not os.environ.get('TIL_TEST_SLOW'), reason="set TIL_TEST_SLOW=1")


@pytest.fixture(autouse=True, scope="module")
def quick_device_vms():
    """Device VMs for --run are built with -O1 here (3x faster to compile; cached)."""
    saved, fit.VM_OPT = fit.VM_OPT, '-O1'
    yield
    fit.VM_OPT = saved


def device_vm(name: str) -> str:
    exe, err = fit.build_device_vm(fit.DEVICES[name])
    assert exe, err
    return exe


def analyze(src: str, strip: bool = False) -> 'fit.Analysis':
    return fit.Analysis(compile_vm(src).serialize(strip=strip), "<test>")


def report(src: str, device: str) -> 'fit.Report':
    return fit.check(analyze(src), fit.DEVICES[device])


def fit_main(capsys, *args):
    """Run tilfit.main in-process; returns (exit code, stdout, stderr)."""
    rc = fit.main(list(args))
    out = capsys.readouterr()
    return rc, out.out, out.err


def flat(text: str) -> str:
    return " ".join(text.split())


def write(tmp_path, text: str, name: str = "p.til") -> str:
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return str(p)


# ═══════════════════════════════════════════════════════════════
# DEVICE PROFILES
# ═══════════════════════════════════════════════════════════════

NAMES = ['desktop', 'ti84ce', 'numworks', 'nspire', 'casio-cg50', 'cortex-m3-64k', 'atmega1284',
         'arduino-mega', 'esp32', 'rp2040', 'z80-16']


class TestDevices:
    def test_all_profiles_exist(self):
        assert list(fit.DEVICES) == NAMES

    @pytest.mark.parametrize("name", NAMES)
    def test_profile_is_valid(self, name):
        d = fit.DEVICES[name]
        d.validate()
        assert d.title and d.source
        flags = d.c_flags()
        assert f"-DTILVM_STACK_SIZE={d.stack}" in flags and f"-DTILVM_HEAP_SIZE={d.heap}" in flags
        assert ('-DTILVM_LEAN' in flags) == d.lean
        # the VM itself must fit the device's RAM with room for a program
        layout = fit.vm_struct_layout(d)
        if d.ram is not None:
            assert layout['tilvm'] + d.vm_static_ram + d.c_stack + 400 <= d.ram, layout
        # a tiny program fits every device
        r = report('print("hi")', name)
        assert r.fits, r.verdict()

    def test_profiles_from_the_ports(self):
        """The verified profiles use exactly the flags of vm/ports/*/Makefile."""
        for port, name in (('cortex-m', 'cortex-m3-64k'), ('avr', 'atmega1284')):
            text = open(os.path.join(VM_DIR, 'ports', port, 'Makefile'), encoding='utf-8').read()
            defs = dict(re.findall(r'-D(TILVM_\w+)=(\d+)', text))
            flags = dict(re.findall(r'-D(TILVM_\w+)=(\d+)', " ".join(fit.DEVICES[name].c_flags())))
            for k, v in defs.items():
                assert flags[k] == v, (name, k)

    def test_list_devices(self, capsys):
        rc, out, _ = fit_main(capsys, '--list-devices')
        assert rc == 0
        for name in NAMES:
            assert re.search(rf'^{re.escape(name)}\s', out, re.M), name
        rc, out, _ = fit_main(capsys, '--list-devices', '--json')
        data = json.loads(out)
        assert [d['name'] for d in data] == NAMES and all(d['vm_struct_bytes'] > 0 for d in data)

    def test_custom_device_file(self, tmp_path, capsys):
        dev = tmp_path / "board.json"
        dev.write_text(json.dumps({"base": "cortex-m3-64k", "name": "myboard", "stack": 64}))
        prog = write(tmp_path, "main()\n    print([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])\n")
        rc, out, _ = fit_main(capsys, prog, '--device', str(dev))
        assert rc == 0 and "on myboard" in out
        dev.write_text(json.dumps({"base": "cortex-m3-64k", "stack": 8}))
        rc, _out, err = fit_main(capsys, prog, '--device', str(dev))
        assert rc == 2 and "stack must be" in err
        dev.write_text(json.dumps({"base": "cortex-m3-64k", "flash": 1}))
        rc, _out, err = fit_main(capsys, prog, '--device', str(dev))
        assert rc == 2 and "unknown device field" in err


# ═══════════════════════════════════════════════════════════════
# THE C VM'S OWN RAM: struct layout model == the C compilers
# ═══════════════════════════════════════════════════════════════

PROBE = '#include "tilvm.h"\nchar tilfit_probe[sizeof(tilvm)];\nchar tilfit_probe_val[sizeof(tv_val)];\n'


def compiler_sizeof(cmd_prefix, flags, tmp_path, sdcc=False):
    src = tmp_path / "probe.c"
    src.write_text(PROBE)
    out = tmp_path / ("probe.asm" if sdcc else "probe.s")
    cmd = list(cmd_prefix) + ['-S', '-I', VM_DIR] + list(flags) + [str(src), '-o', str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    asm = out.read_text()
    if sdcc:
        sizes = dict(re.findall(r'_(tilfit_probe\w*)::\s*\n\s*\.ds (\d+)', asm))
    else:
        sizes = dict(re.findall(r'\.size\s+(tilfit_probe\w*),\s*(\d+)', asm))
        sizes.update(re.findall(r'\.comm\s+(tilfit_probe\w*),(\d+)', asm))
    return int(sizes['tilfit_probe']), int(sizes['tilfit_probe_val'])


def host_abi() -> str:
    return 'lp64' if sys.maxsize > 2 ** 32 else 'ilp32'


@needs_cc
@pytest.mark.parametrize("name", NAMES)
def test_struct_model_matches_host_compiler(name, tmp_path):
    """The model of vm/tilvm.h (all of a profile's macros) == sizeof() on this host."""
    if shutil.which('gcc') is None:
        pytest.skip("gcc not installed")
    dev = fit.DEVICES[name]
    values = dict(dev.as_dict(), abi=host_abi())
    del values['name']
    want = fit.vm_struct_layout(fit.Device('host', **values))
    got = compiler_sizeof(['gcc', '-std=c99'], dev.c_flags(), tmp_path)
    assert got == (want['tilvm'], want['tv_val']), \
        "vm/tilvm.h changed: update vm_struct_layout() in src/tilfit.py"


CROSS_COMPILERS = [
    ('atmega1284', ['avr-gcc', '-mmcu=atmega1284p', '-std=gnu99'], False),
    ('arduino-mega', ['avr-gcc', '-mmcu=atmega2560', '-std=gnu99'], False),
    ('cortex-m3-64k', ['arm-none-eabi-gcc', '-mcpu=cortex-m3', '-mthumb', '-std=c99'], False),
    ('numworks', ['arm-none-eabi-gcc', '-mcpu=cortex-m7', '-mthumb', '-std=c99'], False),
    ('rp2040', ['arm-none-eabi-gcc', '-mcpu=cortex-m0plus', '-mthumb', '-std=c99'], False),
    ('z80-16', ['sdcc', '-mz80', '--std-c99'], True),
]


@pytest.mark.parametrize("name, cmd, sdcc", CROSS_COMPILERS, ids=[c[0] for c in CROSS_COMPILERS])
def test_struct_model_matches_cross_compilers(name, cmd, sdcc, tmp_path):
    """sizeof(tilvm) of the device's own compiler == the model for its ABI."""
    if shutil.which(cmd[0]) is None:
        pytest.skip(f"{cmd[0]} not installed")
    dev = fit.DEVICES[name]
    got = compiler_sizeof(cmd, dev.c_flags(), tmp_path, sdcc)
    want = fit.vm_struct_layout(dev)
    assert got == (want['tilvm'], want['tv_val'])


# ═══════════════════════════════════════════════════════════════
# STATIC ANALYSIS
# ═══════════════════════════════════════════════════════════════

class TestStackAnalysis:
    def test_expression_depth(self):
        an = analyze('main()\n    let a = 1\n    print(a + (a * (a - (a + 2))))\n')
        main = next(an.facts(f) for f in an.reachable if an.module.funcs[f].name == 'main')
        # a, a, a, a, 2 are on the stack at the deepest point
        assert main.max_occ == 5 and main.complete

    def test_array_literal_needs_its_elements(self):
        n = 40
        an = analyze('main()\n    print([' + ", ".join(str(i) for i in range(n)) + '])\n')
        main = next(an.facts(f) for f in an.reachable if an.module.funcs[f].name == 'main')
        assert main.max_occ == n + 1          # make_array keeps the buffer on the stack
        assert main.max_need == n + 2         # ... after checking for 2 free slots

    def test_call_chain_adds_frames(self):
        src = "".join(f"f{i}(x: int) -> int\n    return f{i + 1}(x + 1) + 1\n" for i in range(10))
        src += "f10(x: int) -> int\n    return x\nprint(f0(1))\n"
        an = analyze(src)
        assert an.frames == 1 + 1 + 11 and not an.recursions and an.complete

    def test_recovery_points_add_up(self):
        src = ("inner() -> int\n    return retry(2)\n        retry(2)\n            7\n"
               "main()\n    let v = retry(3)\n        inner()\n    print(v)\n")
        an = analyze(src)
        assert an.tries == 3

    def test_recursion_is_reported(self):
        an = analyze('count(n: int) -> int\n    if n == 0\n        return 0\n    return 1 + count(n - 1)\n'
                     'print(count(5))\n')
        assert len(an.recursions) == 1
        rec = an.recursions[0]
        assert [an.module.funcs[f].name for f in rec.members] == ['count']
        assert rec.slots_per_level == 2        # local n + the pending 1; n-1 becomes the next frame's n
        r = fit.check(an, fit.DEVICES['arduino-mega'])
        assert r.fits
        text = " ".join(w.text for w in r.warnings)
        assert "count calls itself" in text and "depends on the input" in text
        assert r.recursion[0]['levels'] == 14 and r.recursion[0]['limited_by'] == "call frames"

    def test_mutual_recursion(self):
        an = analyze('even(n: int) -> bool\n    if n == 0\n        return true\n    return odd(n - 1)\n'
                     'odd(n: int) -> bool\n    if n == 0\n        return false\n    return even(n - 1)\n'
                     'print(even(10))\n')
        assert len(an.recursions) == 1 and len(an.recursions[0].members) == 2
        r = fit.check(an, fit.DEVICES['desktop'])
        assert any("call each other" in w.text for w in r.warnings)

    def test_indirect_calls_are_conservative(self):
        src = ('big(x: int) -> int\n    let a = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]\n    return len(a)\n'
               'small(x: int) -> int\n    return x\n'
               'main()\n    var f = small\n    if len(args()) > 5\n        f = big\n    print(f(1))\n')
        an = analyze(src)
        names = {an.module.funcs[f].name for f in an.reachable}
        assert {'big', 'small'} <= names
        main = next(an.facts(f) for f in an.reachable if an.module.funcs[f].name == 'main')
        site = next(s for s in main.calls if s.op == til.Op.CALL_VALUE)
        assert {an.module.funcs[g].name for g, _c in site.targets} == {'big', 'small'}
        assert an.slots_occ >= 16

    def test_methods_and_closures_are_followed(self):
        src = ('struct P\n    x: int\nimpl P\n    deep(self) -> int\n        return [1, 2, 3, 4, 5, 6, 7, 8, 9].len()\n'
               'main()\n    let p = P { x: 1 }\n    let k = 2\n    let g = |v| v * k\n    print(p.deep())\n'
               '    print([1, 2].map(g))\n')
        an = analyze(src)
        names = {an.module.funcs[f].name for f in an.reachable}
        assert {'P.deep', '__prelude_map'} <= names and any(n.startswith('<lambda') for n in names)

    def test_unreachable_functions_are_ignored(self):
        r = report('unused() -> str\n    return chr(65)\nprint(1)\n', 'z80-16')
        assert r.fits, r.verdict()


# ═══════════════════════════════════════════════════════════════
# STATIC BOUNDS vs MEASURED HIGH-WATER MARKS (TILVM_STATS)
# ═══════════════════════════════════════════════════════════════

CORPUS = {
    'closures': 'make(k: int)\n    return |x: int| x * k + 1\nmain()\n    let f = make(3)\n    var t = 0\n'
                '    for i in 0..5\n        t += f(i)\n    print(t)\n',
    'structs': 'struct V\n    x: int\n    y: int\nimpl V\n    add(self, o: V) -> V\n        return V { x: self.x + o.x, y: self.y + o.y }\n'
               'main()\n    var v = V { x: 1, y: 2 }\n    for i in 0..4\n        v = v.add(V { x: i, y: i * 2 })\n'
               '    print(v.x, v.y)\n',
    'maps': 'main()\n    var m = {"a": 1, "b": 2}\n    m["c"] = 3\n    for k in m\n        print(k, m[k])\n'
            '    print(m.keys(), m.get_or("z", 0))\n',
    'strings': 'main()\n    let s = "one,two,three"\n    let parts = s.split(",")\n    print(parts.join("+"))\n'
               '    print(f"{len(s)} chars, {parts[1]:>6}|")\n    print(s.slice(0, 3).to_upper())\n',
    'retry': 'var n = 0\nflaky() -> int\n    n += 1\n    assert n >= 3, "warming up"\n    return n\n'
             'main()\n    let v = retry(5)\n        flaky()\n    print(v)\n    let w = retry(2)\n'
             '        intent "read"\n            [1][4]\n    else\n        error\n    print(w)\n',
    'match': 'enum S\n    A(int)\n    B(int, int)\n    C\nf(s: S) -> int\n    return match s\n'
             '        S.A(x) => x\n        S.B(x, y) => x + y\n        S.C => 0\n'
             'main()\n    for s in [S.A(1), S.B(2, 3), S.C]\n        print(f(s))\n',
    'higher_order': 'main()\n    let xs = [1, 2, 3, 4, 5, 6]\n    print(xs.filter(|x| x % 2 == 0).map(|x| x * x))\n'
                    '    print(xs.reduce(|a, b| a + b, 0), sum(xs), min(xs), max(4, 9, 2))\n',
    'range_const': 'main()\n    var t = 0\n    for i in range(50)\n        t += i\n    print(t)\n',
    'comprehension': 'main()\n    let sq = [x * x for x in 0..10 if x % 3 == 0]\n    print(sq)\n',
    'globals': 'var counter = 0\nlet names = ["a", "b"]\nbump(k: int)\n    counter += k\n'
               'main()\n    for i in 0..3\n        bump(i)\n    print(counter, names)\n',
    'defaults': 'pad(s: str, n: int = 6, c: str = ".") -> str\n    var out = s\n    while len(out) < n\n'
                '        out = out + c\n    return out\nmain()\n    print(pad("ab"), pad("x", 3), pad("q", 4, "-"))\n',
    'option': 'find(xs, v: int) -> Option<int>\n    for i in 0..len(xs)\n        if xs[i] == v\n            return Some(i)\n'
              '    return None\nfirst(xs) -> Option<int>\n    let i = find(xs, 3)?\n    return Some(i * 10)\n'
              'main()\n    print(first([1, 3]), first([2]))\n',
}


def stats_vm() -> str:
    """The desktop C VM with TILVM_STATS (default limits)."""
    return device_vm('desktop')


def measure(data: bytes, tmp_path, exe=None):
    path = tmp_path / "m.tbc"
    path.write_bytes(data)
    r = subprocess.run([exe or stats_vm(), '--stats', str(path)], capture_output=True, text=True, timeout=60)
    m = re.search(r'peak stack=(\d+)/\d+ frames=(\d+)/\d+ try=(\d+)/\d+ heap=(\d+)/\d+ live=(\d+)', r.stderr)
    assert m, r.stderr
    return r, dict(zip(('stack', 'frames', 'try', 'heap', 'live'), (int(x) for x in m.groups())))


EXAMPLES = sorted(p for p in glob.glob(os.path.join(ROOT, 'examples', '**', '*.til'), recursive=True))


@needs_cc
class TestStaticVersusMeasured:
    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_corpus_exact(self, name, tmp_path):
        data = compile_vm(CORPUS[name]).serialize()
        an = fit.Analysis(data, name)
        r, peak = measure(data, tmp_path)
        assert r.returncode == 0, r.stderr
        assert an.complete and not an.recursions
        assert an.frames == peak['frames'] and an.tries == peak['try']
        assert an.slots_need >= an.slots_occ
        if any(ff.data_dependent for ff in an.reachable_facts()):
            # split()/join()/map iteration...: the analysis warns that these grow with the data
            assert an.slots_occ <= peak['stack']
            assert any("grows with the data" in w.text for w in fit.check(an, fit.DEVICES['desktop']).warnings)
        else:
            # every path runs in these programs: the bound is exact
            assert an.slots_occ == peak['stack']

    @pytest.mark.parametrize("path", EXAMPLES, ids=[os.path.relpath(p, ROOT) for p in EXAMPLES])
    def test_examples(self, path, tmp_path):
        src = open(path, encoding='utf-8').read()
        data = til.compile_to_bytecode(src, path).serialize()
        an = fit.Analysis(data, path)
        path_tbc = tmp_path / "e.tbc"
        path_tbc.write_bytes(data)
        r = subprocess.run([stats_vm(), '--stats', str(path_tbc)], capture_output=True, text=True,
                           timeout=60, cwd=os.path.dirname(path))
        peak = dict(zip(('stack', 'frames', 'try'), (int(x) for x in re.search(
            r'peak stack=(\d+)/\d+ frames=(\d+)/\d+ try=(\d+)', r.stderr).groups())))
        assert an.complete
        if not an.recursions and not any(ff.data_dependent for ff in an.reachable_facts()):
            assert (an.slots_occ, an.frames, an.tries) == (peak['stack'], peak['frames'], peak['try'])
        elif not an.recursions:
            assert an.frames == peak['frames'] and an.tries == peak['try']
        else:
            assert an.slots_occ <= peak['stack'] and an.frames <= peak['frames']

    @pytest.mark.parametrize("n", [1, 10, 57])
    def test_recursion_levels_formula(self, n, tmp_path):
        """k levels of a recursion need at most base + (k - 1) * per_level + top slots
        (the deepest level takes the base case, so one slot less here)."""
        src = ('count(n: int) -> int\n    if n == 0\n        return 0\n    return 1 + count(n - 1)\n'
               f'main()\n    print(count({n}))\n')
        data = compile_vm(src).serialize()
        an = fit.Analysis(data, "<rec>")
        rec = an.recursions[0]
        _r, peak = measure(data, tmp_path)
        k = n + 1
        assert peak['frames'] == rec.frames_below - 1 + k
        assert peak['stack'] == rec.base + (k - 1) * rec.slots_per_level + rec.top_occ - 1

    def test_levels_prediction_is_exact_on_the_device(self, tmp_path, capsys):
        """The predicted number of levels is the number the device's VM runs."""
        dev = fit.DEVICES['arduino-mega']
        src = 'count(n: int) -> int\n    if n == 0\n        return 0\n    return 1 + count(n - 1)\nprint(count(N))\n'
        levels = fit.check(analyze(src.replace('N', '1')), dev).recursion[0]['levels']
        exe = device_vm('arduino-mega')
        ok, _ = measure(compile_vm(src.replace('N', str(levels - 1))).serialize(), tmp_path, exe)
        bad, _ = measure(compile_vm(src.replace('N', str(levels))).serialize(), tmp_path, exe)
        assert ok.returncode == 0 and bad.returncode == 1
        assert "recursion too deep" in bad.stderr


# ═══════════════════════════════════════════════════════════════
# VERDICTS (static)
# ═══════════════════════════════════════════════════════════════

class TestVerdicts:
    def test_deep_call_chain_does_not_fit_frames(self):
        src = "".join(f"f{i}() -> int\n    return f{i + 1}() + 1\n" for i in range(20)) + \
            "f20() -> int\n    return 0\nprint(f0())\n"
        r = report(src, 'arduino-mega')
        assert not r.fits and "call frames 23 > 16" in r.verdict()
        assert report(src, 'cortex-m3-64k').fits

    def test_big_array_literal(self):
        src = 'main()\n    let a = [' + ", ".join(str(i % 100) for i in range(200)) + ']\n    print(len(a))\n'
        r = report(src, 'atmega1284')
        assert not r.fits and "value stack" in r.verdict()
        assert report(src, 'cortex-m3-64k').fits

    def test_range_with_a_constant_size(self):
        src = 'main()\n    var t = 0\n    for i in range(1000)\n        t += i\n    print(t)\n'
        r = report(src, 'cortex-m3-64k')
        assert not r.fits and "value stack" in r.verdict()
        assert report(src, 'desktop').fits
        assert report(src.replace('range(1000)', '0..1000'), 'cortex-m3-64k').fits

    @pytest.mark.parametrize("device", ['cortex-m3-64k', 'atmega1284', 'esp32', 'rp2040', 'ti84ce'])
    def test_64_bit_constant_on_int32_devices(self, device):
        src = 'var big = 5000000000\nprint(big + 1)\n'
        r = report(src, device)
        assert not r.fits
        assert "integer constant 5000000000 needs 64-bit integers" in r.verdict()
        assert report(src, 'nspire').fits and report(src, 'desktop').fits

    @pytest.mark.parametrize("device", ['arduino-mega', 'z80-16'])
    def test_32_bit_constant_on_int16_devices(self, device):
        r = report('var x = 100000\nprint(x)\n', device)
        assert "needs 32-bit integers" in r.verdict()
        assert report('var x = 30000\nprint(x)\n', device).fits

    def test_floats_on_z80(self):
        r = report('print(1.5 * 2)\n', 'z80-16')
        assert not r.fits and "float constants" in r.verdict()
        r = report('half(x: float) -> float\n    return x / 2\nprint(half(3))\n', 'z80-16')
        assert "float" in r.verdict()
        r = report('print(sqrt(16))\n', 'z80-16')
        assert "sqrt() need floats" in r.verdict()
        assert report('print(sqrt(16))\n', 'arduino-mega').fits     # float32 there

    @pytest.mark.parametrize("device", ['z80-16', 'arduino-mega'])
    def test_lean_missing_builtins(self, device):
        for src, name in (('print(chr(65))\n', 'chr()'), ('print(ord("A"))\n', 'ord()'),
                          ('print("a,b".split(","))\n', '.split()'), ('var a = [3, 1]\na.sort()\nprint(a)\n', '.sort()'),
                          ('print(["a", "b"].join("-"))\n', '.join()'), ('print(type_of(1))\n', 'type_of()'),
                          ('print("x".repeat(3))\n', '.repeat()')):
            r = report(src, device)
            assert not r.fits and name in r.verdict(), (src, r.verdict())
            assert report(src, 'cortex-m3-64k').fits, src

    def test_lean_format_specs(self):
        r = report('let x = 7\nprint(f"{x:>5}|")\n', 'arduino-mega')
        assert not r.fits and "format specs" in r.verdict()
        assert report('let x = 7\nprint(f"{x}|")\n', 'arduino-mega').fits

    def test_lean_methods_that_depend_on_the_receiver(self):
        r = report('print("hello".slice(1, 3))\n', 'z80-16')
        assert r.fits and any(".slice()" in w.text and "array" in w.text for w in r.warnings)
        src = ('struct Bag\n    n: int\nimpl Bag\n    sort(self) -> int\n        return self.n\n'
               'main()\n    print(Bag { n: 2 }.sort())\n')
        r = report(src, 'z80-16')
        assert r.fits and any("your own struct" in w.text for w in r.warnings)

    def test_constant_and_type_tables(self):
        src = "main()\n" + "".join(f'    print("s{i}")\n' for i in range(100))
        r = report(src, 'arduino-mega')
        assert not r.fits and re.search(r'\d+ constants > 96', r.verdict())
        src = "".join(f"struct T{i}\n    v: int\n" for i in range(20)) + "print(T3 { v: 1 }.v)\n"
        r = report(src, 'arduino-mega')
        assert not r.fits and "23 struct types > 16" in r.verdict()

    def test_too_many_globals(self):
        src = "".join(f"var g{i} = [{i}]\n" for i in range(90)) + "print(g1)\n"
        r = report(src, 'arduino-mega')
        assert not r.fits and "90 globals > 88" in r.verdict()

    def test_nested_recovery_points(self):
        body, ind = "1", "    "
        for _ in range(9):
            body = f"retry(2)\n{ind}    " + body.replace("\n", "\n    ")
        src = "main()\n    let v = " + body + "\n    print(v)\n"
        r = report(src, 'cortex-m3-64k')
        assert not r.fits and "recovery points 9 > 8" in r.verdict()
        assert report(src, 'desktop').fits

    def test_storage_and_strip(self, tmp_path, capsys):
        src = "main()\n" + "".join(f"    print({i} + {i})\n" for i in range(300))
        full = len(compile_vm(src).serialize())
        stripped = len(compile_vm(src).serialize(strip=True))
        dev = tmp_path / "small.json"
        dev.write_text(json.dumps({"base": "cortex-m3-64k", "storage": (full + stripped) // 2}))
        prog = write(tmp_path, src)
        rc, out, _ = fit_main(capsys, prog, '--device', str(dev))
        assert rc == 1 and "--strip would save" in flat(out) and "storage" in out
        rc, out, _ = fit_main(capsys, prog, '--device', str(dev), '--strip')
        assert rc == 0, out

    def test_bytecode_in_ram_on_avr(self):
        src = "main()\n" + "".join(f'    print("line number {i} of a long program")\n' for i in range(150))
        an = analyze(src, strip=True)
        r = fit.check(an, fit.DEVICES['atmega1284'])
        assert not r.fits and "VM RAM" in r.verdict() and an.size < fit.DEVICES['atmega1284'].storage
        assert fit.check(an, fit.DEVICES['cortex-m3-64k']).fits

    def test_device_services(self):
        r = report('screen_init(10, 10)\nprint(read_file("x"))\n', 'esp32')
        texts = " ".join(w.text for w in r.warnings)
        assert "need a screen" in texts and "file system" in texts
        r = report('let t = timeout(10)\n    1\nprint(t)\n', 'cortex-m3-64k')
        assert any("counts VM steps" in w.text for w in r.warnings)

    def test_allocations_in_loops(self):
        r = report('main()\n    var xs = []\n    for i in 0..100\n        xs.push([i])\n    print(len(xs))\n',
                   'cortex-m3-64k')
        assert r.fits
        assert any("loop at line 3 in main keeps allocating" in w.text and "array push at line 4" in w.text
                   for w in r.warnings), [w.text for w in r.warnings]
        r = report('main()\n    for i in 0..3\n        print(f"i = {i}")\n', 'cortex-m3-64k')
        assert any("temporary objects" in n for n in r.notes)

    def test_data_dependent_stack(self):
        r = report('main()\n    let parts = "a b c".split(" ")\n    print(parts)\n', 'desktop')
        assert any("grows with the data" in w.text and ".split()" in w.text for w in r.warnings)

    def test_json_output(self, tmp_path, capsys):
        prog = write(tmp_path, 'fib(n: int) -> int\n    if n < 2\n        return n\n'
                               '    return fib(n - 1) + fib(n - 2)\nprint(fib(10))\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'rp2040', '--json')
        data = json.loads(out)
        assert rc == 0 and data['fits'] is True and data['verdict'] == 'fits'
        assert data['device']['name'] == 'rp2040' and data['vm']['struct_bytes'] > 0
        st = data['static']
        assert st['complete'] and st['stack_slots'] > 0 and st['recursion'][0]['functions'] == ['fib']
        assert {r['key'] for r in data['resources']} >= {'bytecode', 'ram', 'stack', 'frames', 'heap'}
        assert data['run'] is None

    def test_text_report_and_functions_table(self, tmp_path, capsys):
        prog = write(tmp_path, 'main()\n    print([1, 2, 3])\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'esp32', '--functions')
        assert rc == 0 and out.rstrip().endswith("verdict: fits")
        assert "value stack" in out and "functions (reachable" in out and "main" in out

    def test_usage_errors(self, tmp_path, capsys):
        prog = write(tmp_path, 'print(1)\n')
        assert fit_main(capsys, prog)[0] == 2                                   # no --device
        assert fit_main(capsys, prog, '--device', 'pdp11')[0] == 2
        assert fit_main(capsys, prog, '--device', 'esp32', '--bogus')[0] == 2
        assert fit_main(capsys, str(tmp_path / "none.til"), '--device', 'esp32')[0] == 2
        bad = write(tmp_path, 'print(\n', "bad.til")
        assert fit_main(capsys, bad, '--device', 'esp32')[0] == 2
        assert fit_main(capsys, '--help')[0] == 0
        assert fit_main(capsys)[0] == 2


# ═══════════════════════════════════════════════════════════════
# ROBUSTNESS: opcodes the analysis does not know (superinstructions,
# effect opcodes from other branches, corrupt files)
# ═══════════════════════════════════════════════════════════════

def patched_tbc(new_op: int) -> bytes:
    """`x = x + 2` in main with the ADD replaced by another opcode."""
    m = compile_vm('main()\n    var x = 1\n    x = x + 2\n    print(x)\n')
    data = bytearray(m.serialize())
    mod = til.TBCModule.from_bytes(bytes(data))
    main = next(f for f in mod.funcs if f.name == 'main')
    code_start = len(data) - 2 - 6 * len(mod.lines) - len(mod.code)
    at = code_start + main.offset + bytes(mod.code[main.offset:]).index(bytes([til.Op.ADD]))
    data[at] = new_op
    return bytes(data)


class TestRobustness:
    @pytest.mark.parametrize("op", [0x80, 0xB0, 0xBF, 0x7E, 0xFF])
    def test_unknown_opcode_is_incomplete_not_a_crash(self, op, tmp_path, capsys):
        path = tmp_path / "p.tbc"
        path.write_bytes(patched_tbc(op))
        rc, out, err = fit_main(capsys, str(path), '--device', 'cortex-m3-64k')
        assert rc == 1, err
        assert f"unknown opcode 0x{op:02X}" in out and "analysis incomplete" in out
        assert "verdict: does not fit" in out
        rc, out, _ = fit_main(capsys, str(path), '--device', 'cortex-m3-64k', '--json')
        data = json.loads(out)
        assert data['static']['complete'] is False and data['static']['stack_slots'] is None

    def test_known_layout_without_stack_effect(self, tmp_path, capsys):
        """HANDLE has an operand layout but (in this branch) no stack-effect entry."""
        path = tmp_path / "p.tbc"
        path.write_bytes(patched_tbc(til.Op.HANDLE))
        rc, out, _ = fit_main(capsys, str(path), '--device', 'desktop')
        assert rc == 1 and "no stack-effect entry for opcode HANDLE" in out

    def test_new_opcode_is_a_one_line_addition(self, monkeypatch):
        """A superinstruction registered in til (Op/OP_NAMES/OP_OPERANDS) + one line in
        STACK_EFFECTS makes the analysis complete again."""
        op = 0x80
        monkeypatch.setitem(til.OP_NAMES, op, 'ADD_I8')          # ADD_I8 n: TOS += n
        monkeypatch.setitem(til.OP_OPERANDS, op, 'c')
        data = bytearray(patched_tbc(op))
        an = fit.Analysis(bytes(data), "<p>")
        assert not an.complete
        monkeypatch.setitem(fit.STACK_EFFECTS, op, fit.Effect(1, 1))
        an = fit.Analysis(bytes(data), "<p>")
        assert an.complete, an.problems()

    def test_stack_underflow_and_bad_jumps(self):
        data = bytearray(patched_tbc(til.Op.POP))
        # POP instead of ADD leaves the stack consistent; DUP2 at the start underflows
        mod = til.TBCModule.from_bytes(bytes(data))
        main = next(f for f in mod.funcs if f.name == 'main')
        code_start = len(data) - 2 - 6 * len(mod.lines) - len(mod.code)
        data2 = bytearray(data)
        data2[code_start + main.offset] = til.Op.DUP2
        an = fit.Analysis(bytes(data2), "<p>")
        assert not an.complete and any("pops 2" in p for p in an.problems())
        data3 = bytearray(data)
        data3[code_start + main.offset] = til.Op.JUMP
        data3[code_start + main.offset + 1:code_start + main.offset + 3] = (30000).to_bytes(2, 'little')
        an = fit.Analysis(bytes(data3), "<p>")
        assert not an.complete and any("outside the function" in p for p in an.problems())

    def test_corrupt_files(self, tmp_path, capsys):
        good = tbc_bytes('print(1)\n')
        for name, data in (("trunc.tbc", good[:20]), ("magic.tbc", b"XXXX" + good[4:]), ("empty.tbc", b"")):
            path = tmp_path / name
            path.write_bytes(data)
            rc, _out, err = fit_main(capsys, str(path), '--device', 'desktop')
            assert rc == 2 and "til fit:" in err and "Traceback" not in err


# ═══════════════════════════════════════════════════════════════
# --run: THE DEVICE'S LIMITS, MEASURED
# ═══════════════════════════════════════════════════════════════

@needs_cc
class TestRun:
    def test_run_reports_high_water_marks(self, tmp_path, capsys):
        prog = write(tmp_path, CORPUS['structs'])
        rc, out, _ = fit_main(capsys, prog, '--device', 'cortex-m3-64k', '--run', '--json')
        data = json.loads(out)
        assert rc == 0 and data['fits']
        run = data['run']
        assert run['exit_code'] == 0 and run['desktop']['same_output'] is True
        st = run['stats']
        assert st['peak_stack'] == data['static']['stack_slots_written']
        assert st['peak_frames'] == data['static']['call_frames']
        assert 0 < st['peak_heap'] <= 16384 and 'steps' in st and 'gc' in st
        rc, out, _ = fit_main(capsys, prog, '--device', 'cortex-m3-64k', '--run')
        assert "measured" in out and re.search(r'value stack\s+\d+ slots\s+\d+\s+512', out)

    def test_run_finds_heap_exhaustion(self, tmp_path, capsys):
        prog = write(tmp_path, 'var items = []\nfor i in 0..2000\n    items.push(i)\nprint(len(items))\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'arduino-mega')
        assert rc == 0 and "keeps allocating" in out          # static: a risk, not a verdict
        rc, out, _ = fit_main(capsys, prog, '--device', 'arduino-mega', '--run')
        assert rc == 1 and "verdict: does not fit: run: out of memory" in out

    def test_run_finds_deep_recursion(self, tmp_path, capsys):
        prog = write(tmp_path, 'count(n: int) -> int\n    if n == 0\n        return 0\n'
                               '    return 1 + count(n - 1)\nprint(count(100))\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'arduino-mega', '--run')
        assert rc == 1 and "stack overflow (recursion too deep)" in out
        rc, out, _ = fit_main(capsys, prog, '--device', 'desktop', '--run')
        assert rc == 0, out

    def test_run_compares_with_the_desktop(self, tmp_path, capsys):
        prog = write(tmp_path, 'var a = 300\nprint(a * a)\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'arduino-mega', '--run', '--show-output')
        assert rc == 0 and out.startswith("24464\n")             # 16-bit wrap-around
        assert "output differs from the desktop VM" in out and "'24464' instead of '90000'" in out

    def test_program_errors_are_not_device_errors(self, tmp_path, capsys):
        prog = write(tmp_path, 'let xs = [1, 2, 3]\nprint(xs[5])\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'arduino-mega', '--run')
        assert rc == 0 and "on the desktop VM too" in out

    def test_run_load_error(self, tmp_path, capsys):
        prog = write(tmp_path, 'let big = 5000000000\nprint(big)\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'cortex-m3-64k', '--run')
        assert rc == 1 and "refuses to load the program" in out and "(error -6)" in out

    def test_max_steps_and_arguments(self, tmp_path, capsys):
        prog = write(tmp_path, 'print(len(args()))\nvar i = 0\nwhile true\n    i += 1\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'cortex-m3-64k', '--run', '--max-steps', '5000',
                              '--show-output', '--', 'a', 'b')
        assert rc == 0 and out.startswith("2\n") and "stopped at --max-steps" in out

    def test_cli_dispatch(self, tmp_path):
        prog = write(tmp_path, 'print("hello")\n')
        r = cli('fit', prog, '--device', 'z80-16')
        assert r.returncode == 0 and "verdict: fits" in r.stdout, r.stderr
        r = cli('fit', prog, '--device', 'z80-16', '--json')
        assert r.returncode == 0 and json.loads(r.stdout)['fits'] is True
        r = cli('fit', '--list-devices')
        assert r.returncode == 0 and 'casio-cg50' in r.stdout

    @slow
    @pytest.mark.parametrize("device", NAMES)
    def test_every_device_vm_builds_and_runs(self, device, tmp_path, capsys):
        prog = write(tmp_path, CORPUS['strings'] if not fit.DEVICES[device].lean else
                     'main()\n    var t = 0\n    for i in 0..10\n        t += i\n    print(t)\n')
        rc, out, _ = fit_main(capsys, prog, '--device', device, '--run')
        assert rc == 0, out


# ═══════════════════════════════════════════════════════════════
# THE C VM: TILVM_STATS, TILVM_PACK, the desktop runner
# ═══════════════════════════════════════════════════════════════

STRICT = ['-std=c99', '-Wall', '-Wextra', '-pedantic', '-Werror']
CONFIGS = [
    [],
    ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1'],
    ['-DTILVM_LEAN', '-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0', '-DTILVM_NO_STDIO'],
    ['-DTILVM_NO_STDIO', '-DTILVM_NO_TIME'],
    ['-DTILVM_NO_STDIO', '-DTILVM_NO_MATH', '-DTILVM_FLOAT=1'],
]


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("flags", CONFIGS, ids=["default", "int32", "lean16", "nostdio", "nomath"])
def test_strict_builds_with_stats_and_pack(compiler, flags, tmp_path):
    if not shutil.which(compiler):
        pytest.skip(f"{compiler} not installed")
    for src in ('tilvm.c', 'main.c'):
        r = subprocess.run([compiler] + STRICT + flags + ['-DTILVM_STATS', '-DTILVM_PACK', '-fsyntax-only',
                                                          os.path.join(VM_DIR, src)],
                           capture_output=True, text=True)
        assert r.returncode == 0, (src, r.stderr)


@needs_cc
class TestCVM:
    def test_stats_lines(self, tmp_path):
        data = compile_vm(CORPUS['closures']).serialize()
        r, peak = measure(data, tmp_path)
        lines = [ln for ln in r.stderr.splitlines() if ln.startswith('[tilvm]')]
        assert re.fullmatch(r'\[tilvm\] steps=\d+ gc=\d+ heap=\d+/\d+ bytes', lines[0])
        assert lines[1].startswith('[tilvm] peak stack=') and len(lines) == 2
        # a normal build keeps the one-line format
        tbc = tmp_path / "c.tbc"
        tbc.write_bytes(data)
        r = subprocess.run([cvm_binary(), '--stats', str(tbc)], capture_output=True, text=True)
        assert [ln for ln in r.stderr.splitlines() if ln.startswith('[tilvm]')] == lines[:1]

    def test_peaks_after_gc(self, tmp_path):
        src = ('main()\n    var keep = []\n    for i in 0..3000\n        let s = f"item {i}"\n'
               '        if i % 100 == 0\n            keep.push(s)\n    print(len(keep))\n')
        r, peak = measure(compile_vm(src).serialize(), tmp_path, device_vm('cortex-m3-64k'))
        gcs = int(re.search(r'gc=(\d+)', r.stderr).group(1))
        assert r.returncode == 0 and gcs > 0, r.stderr
        assert 0 < peak['live'] < peak['heap'] <= 16384

    def test_no_stdio_runner_uses_the_console(self, tmp_path):
        """vm/main.c gives a TILVM_NO_STDIO core (a device profile) a console."""
        assert fit.DEVICES['cortex-m3-64k'].no_stdio
        tbc = tmp_path / "io.tbc"
        tbc.write_bytes(compile_vm('let name = input("name? ")\nprint("hi " + name)\nprint([1][3])\n').serialize())
        r = subprocess.run([device_vm('cortex-m3-64k'), str(tbc)], input="Aigerim\n", capture_output=True,
                           text=True)
        assert r.stdout == "name? hi Aigerim\n" and "index out of bounds" in r.stderr and r.returncode == 1

    def test_packed_values_run_identically(self, tmp_path):
        """TILVM_PACK (the 8-bit layout, used to emulate AVR/Z80/eZ80) changes no behaviour."""
        assert '-DTILVM_PACK' in fit.device_vm_flags(fit.DEVICES['arduino-mega'])
        exe = device_vm('arduino-mega')
        for name in ('closures', 'structs', 'maps', 'higher_order', 'match', 'option', 'defaults'):
            data = compile_vm(CORPUS[name]).serialize()
            rc, ref = run_vm(data)
            path = tmp_path / f"{name}.tbc"
            path.write_bytes(data)
            r = subprocess.run([exe, str(path)], capture_output=True, text=True, timeout=60)
            assert (r.returncode, r.stdout) == (rc, ref), (name, r.stderr)

    @needs_cc
    def test_packed_layout_is_the_8_bit_layout(self, tmp_path):
        """With TILVM_PACK a value takes as many bytes on the desktop as on AVR / Z80."""
        if shutil.which('gcc') is None:
            pytest.skip("gcc not installed")
        for name in ('atmega1284', 'arduino-mega', 'z80-16', 'ti84ce'):
            dev = fit.DEVICES[name]
            _vm, val = compiler_sizeof(['gcc', '-std=c99'], dev.c_flags() + ['-DTILVM_PACK'], tmp_path)
            assert val == fit.vm_struct_layout(dev)['tv_val'], name


# ═══════════════════════════════════════════════════════════════
# THE LEAN TABLES == vm/tilvm.c
# ═══════════════════════════════════════════════════════════════

METHOD_PROBES = {
    'str': ('"a,b"', {'len': '', 'is_empty': '', 'contains': '"a"', 'starts_with': '"a"', 'ends_with': '"b"',
                      'find': '"b"', 'index_of': '"b"', 'count': '"a"', 'eq': '"x"', 'get': '0', 'trim': '',
                      'repeat': '2', 'to_upper': '', 'to_lower': '', 'slice': '0, 1', 'replace': '"a", "c"',
                      'split': '","', 'to_int': '', 'to_float': ''}),
    'array': ('[3, 1, 2]', {'len': '', 'is_empty': '', 'push': '4', 'pop': '', 'remove': '0', 'insert': '0, 9',
                            'get': '0', 'set': '0, 5', 'contains': '1', 'index_of': '1', 'count': '1',
                            'reverse': '', 'sort': '', 'first': '', 'last': '', 'clear': '', 'sum': '',
                            'min': '', 'max': '', 'copy': '', 'slice': '0, 2', 'join': '"-"'}),
    'map': ('{"k": 1}', {'len': '', 'is_empty': '', 'set': '"j", 2', 'get': '"k"', 'get_or': '"z", 0',
                         'has': '"k"', 'contains': '"k"', 'delete': '"k"', 'remove': '"k"', 'values': '',
                         'copy': '', 'keys': '', 'clear': ''}),
}


@needs_cc
def test_lean_tables_match_the_c_vm(tmp_path):
    """LEAN_MISSING_METHODS / LEAN_MISSING_NATIVES are exactly what a TILVM_LEAN build
    lacks (probed on the arduino-mega VM, which is lean, against a full int32 build)."""
    probes = []
    for recv, (value, calls) in METHOD_PROBES.items():
        for name, args in calls.items():
            assert name in fit.METHOD_RECEIVERS[recv], (recv, name)
            probes.append((f"{recv}.{name}", f"{value}.{name}({args})",
                           name in fit.LEAN_MISSING_METHODS.get(recv, ())))
    natives = {'input': '', 'chr': '65', 'ord': '"A"', 'type_of': '1', 'clock_ms': '', 'random': '1, 2',
               'seed': '1', 'sleep_ms': '0', 'file_exists': '"x"', 'args': '', 'bytes_to_str': '[65]',
               'str_bytes': '"A"', 'len': '"ab"', 'abs': '-1', 'min': '1, 2', 'str': '5', 'int': '"7"',
               'bool': '1', 'sum': '[1, 2]', 'range': '0, 2'}
    assert set(fit.LEAN_MISSING_NATIVES) - {'read_file', 'write_file'} <= set(natives)
    for name, args in natives.items():
        probes.append((f"{name}()", f"{name}({args})", name in fit.LEAN_MISSING_NATIVES))
    lean_vm, full_vm = device_vm('arduino-mega'), cvm_binary("int32")
    for start in range(0, len(probes), 12):          # small programs: the lean VM holds 96 constants
        chunk = probes[start:start + 12]
        src = "".join(f'p{i}()\n    let r = retry(1)\n        {expr}\n    else\n        error\n'
                      f'    print("{what}=" + str(r))\n' for i, (what, expr, _m) in enumerate(chunk))
        src += "main()\n" + "".join(f"    p{i}()\n" for i in range(len(chunk)))
        path = tmp_path / f"probe{start}.tbc"
        path.write_bytes(compile_vm(src).serialize())
        lean = subprocess.run([lean_vm, str(path)], capture_output=True, text=True, input="", timeout=60)
        full = subprocess.run([full_vm, str(path)], capture_output=True, text=True, input="", timeout=60)
        assert lean.returncode == 0 and full.returncode == 0, (lean.stderr, full.stderr)
        for what, _expr, missing in chunk:
            lean_line = next(ln for ln in lean.stdout.splitlines() if ln.startswith(what + "="))
            full_line = next(ln for ln in full.stdout.splitlines() if ln.startswith(what + "="))
            unavailable = "no method '" in lean_line or "is not available in this VM build" in lean_line
            assert unavailable == missing, (what, lean_line)
            assert "no method '" not in full_line and "not available in this VM build" not in full_line


# ═══════════════════════════════════════════════════════════════
# til build --target vm --device NAME
# ═══════════════════════════════════════════════════════════════

class TestBuildIntegration:
    def test_build_fits(self, tmp_path):
        prog = write(tmp_path, CORPUS['structs'])
        out = tmp_path / "p.tbc"
        r = cli('build', '--target', 'vm', '--device', 'cortex-m3-64k', prog, '-o', str(out))
        assert r.returncode == 0, r.stderr
        assert "Fits cortex-m3-64k" in r.stdout and out.exists()
        assert til.TBCModule.from_bytes(out.read_bytes()).funcs

    def test_build_does_not_fit(self, tmp_path):
        prog = write(tmp_path, 'print(chr(65) + "x".repeat(2))\n')
        out = tmp_path / "p.tbc"
        r = cli('build', '--target', 'vm', '--device=z80-16', prog, '-o', str(out))
        assert r.returncode == 1 and not out.exists()
        assert "verdict: does not fit" in r.stderr and "not in the lean VM: chr(), .repeat()" in r.stderr
        r = cli('build', '--target', 'vm', prog, '-o', str(out))     # without --device: no check
        assert r.returncode == 0 and out.exists()

    def test_build_c_vm_and_strip(self, tmp_path):
        prog = write(tmp_path, 'print("hi")\n')
        out = tmp_path / "p_vm.c"
        r = cli('build', '--target', 'c-vm', '--device', 'atmega1284', '--strip', prog, '-o', str(out))
        assert r.returncode == 0 and out.exists() and "Fits atmega1284" in r.stdout, r.stderr

    def test_device_needs_a_vm_target(self, tmp_path):
        prog = write(tmp_path, 'print("hi")\n')
        for args in (('build', '--device', 'esp32', prog), ('run', '--vm', '--device', 'esp32', prog)):
            r = cli(*args)
            assert r.returncode == 1 and "--device works with" in r.stderr

    def test_unknown_device(self, tmp_path):
        prog = write(tmp_path, 'print("hi")\n')
        r = cli('build', '--target', 'vm', '--device', 'toaster', prog, '-o', str(tmp_path / "x.tbc"))
        assert r.returncode == 1 and "unknown device 'toaster'" in r.stderr

    def test_til_help_lists_fit(self):
        r = cli('--help')
        assert "fit ..." in r.stdout and "--device" in r.stdout
