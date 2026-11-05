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
DEFAULT_VM_OPT = fit.VM_OPT          # what `til fit` uses (the fixture below speeds up in-process builds)


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
        sizes = dict(re.findall(r'\.size\s+_?(tilfit_probe\w*),\s*(\d+)', asm))
        sizes.update(re.findall(r'\.comm\s+_?(tilfit_probe\w*),\s*(\d+)', asm))
    return int(sizes['tilfit_probe']), int(sizes['tilfit_probe_val'])


def host_sizes(configs, tmp_path):
    """{key: (sizeof(tilvm), sizeof(tv_val))} on this host for several sets of -D flags:
    one translation unit per set, linked into one probe program and run."""
    files, calls = [], []
    for i, (key, flags) in enumerate(sorted(configs.items())):
        defines = "".join("#define " + f[2:].replace('=', ' ', 1) + "\n" for f in flags)
        unit = tmp_path / f"size{i}.c"
        unit.write_text(defines + '#include "tilvm.h"\n'
                        f'unsigned long vm_size{i}(void) {{ return (unsigned long)sizeof(tilvm); }}\n'
                        f'unsigned long val_size{i}(void) {{ return (unsigned long)sizeof(tv_val); }}\n')
        files.append(str(unit))
        calls.append((key, i))
    main = tmp_path / "sizes.c"
    main.write_text("#include <stdio.h>\n" +
                    "".join(f"unsigned long vm_size{i}(void);\nunsigned long val_size{i}(void);\n" for _k, i in calls) +
                    "int main(void) {\n" +
                    "".join(f'    printf("%lu %lu\\n", vm_size{i}(), val_size{i}());\n' for _k, i in calls) +
                    "    return 0;\n}\n")
    exe = tmp_path / "sizes"
    r = subprocess.run([CC, '-std=c99', '-I', VM_DIR, str(main)] + files + ['-o', str(exe)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = subprocess.run([str(exe)], capture_output=True, text=True).stdout.split("\n")
    return {key: tuple(int(x) for x in lines[i].split()) for key, i in calls}


_HOST_SIZES = {}


def host_sizes_cached(tmp_path_factory):
    """sizeof() for every device profile (and with TILVM_PACK) on this host, computed once."""
    if not _HOST_SIZES:
        configs = {}
        for name, dev in fit.DEVICES.items():
            configs[name] = dev.c_flags()
            configs[name + "+pack"] = dev.c_flags() + ['-DTILVM_PACK']
        _HOST_SIZES.update(host_sizes(configs, tmp_path_factory.mktemp("sizes")))
    return _HOST_SIZES


def host_abi() -> str:
    return 'lp64' if sys.maxsize > 2 ** 32 else 'ilp32'


@needs_cc
@pytest.mark.parametrize("name", NAMES)
def test_struct_model_matches_host_compiler(name, tmp_path_factory):
    """The model of vm/tilvm.h (all of a profile's macros) == sizeof() on this host."""
    dev = fit.DEVICES[name]
    values = dict(dev.as_dict(), abi=host_abi())
    del values['name']
    want = fit.vm_struct_layout(fit.Device('host', **values))
    got = host_sizes_cached(tmp_path_factory)[name]
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


cross = pytest.mark.skipif(not os.environ.get('TIL_TEST_CROSS'), reason="set TIL_TEST_CROSS=1")

PORT_BUILDS = [
    ('atmega1284', ['avr-gcc', '-mmcu=atmega1284p', '-Os', '-std=gnu99'], 'avr', 'avr-size'),
    ('arduino-mega', ['avr-gcc', '-mmcu=atmega2560', '-Os', '-std=gnu99'], 'avr', 'avr-size'),
    ('cortex-m3-64k', ['arm-none-eabi-gcc', '-mcpu=cortex-m3', '-mthumb', '-Os', '-std=c99'], 'cortex-m',
     'arm-none-eabi-size'),
]


@cross
@pytest.mark.parametrize("name, cmd, port, size_tool", PORT_BUILDS, ids=[b[0] for b in PORT_BUILDS])
def test_ram_model_matches_the_linked_port(name, cmd, port, size_tool, tmp_path):
    """Link the real port with a program: its static RAM (.data + .bss) is what the
    model counts (VM struct + VM tables [+ bytecode copied to RAM])."""
    if shutil.which(cmd[0]) is None:
        pytest.skip(f"{cmd[0]} not installed")
    dev = fit.DEVICES[name]
    data = compile_vm(open(os.path.join(ROOT, 'examples', '05_control_flow.til')).read()).serialize(strip=True)
    (tmp_path / "program.h").write_text(til.bytes_to_c_array('til_program', data))
    pdir = os.path.join(VM_DIR, 'ports', port)
    srcs = [os.path.join(pdir, 'main_avr.c')] if port == 'avr' else \
        [os.path.join(pdir, 'main_arm.c'), os.path.join(pdir, 'startup.c')]
    extra = [] if port == 'avr' else ['-T', os.path.join(pdir, 'lm3s6965.ld'), '-nostartfiles',
                                      '--specs=nano.specs', '--specs=nosys.specs']
    elf = tmp_path / "t.elf"
    r = subprocess.run(cmd + ['-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections'] + dev.c_flags() +
                       ['-I', VM_DIR, '-I', str(tmp_path)] + extra + ['-o', str(elf)] + srcs +
                       [os.path.join(VM_DIR, 'tilvm.c'), '-lm'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    sizes = dict((m.group(1), int(m.group(2))) for m in re.finditer(
        r'^\.(data|bss|text)\s+(\d+)', subprocess.run([size_tool, '-A', str(elf)], capture_output=True,
                                                     text=True).stdout, re.M))
    layout = fit.vm_struct_layout(dev)
    assert sizes['bss'] >= layout['tilvm'] and sizes['bss'] - layout['tilvm'] < 512
    static = sizes['data'] - (len(data) if dev.bytecode_in_ram else 0)
    assert abs(static - dev.vm_static_ram) <= 160, (static, dev.vm_static_ram)
    assert sizes['text'] + (0 if dev.bytecode_in_ram else len(data)) + (dev.storage or 0) <= \
        (128 if name == 'atmega1284' else 256) * 1024 + len(data)


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


    def test_stack_limited_recursion_is_exact(self, tmp_path):
        """Same when the value stack (not the call depth) limits the recursion."""
        values = fit.DEVICES['desktop'].as_dict()
        del values['name']
        values.update(stack=300, heap=8192, title="small stack")
        dev = fit.Device('small-stack', **values)
        src = ('deep(n: int) -> int\n    let a = n + 1\n    let b = a * 2\n    let c = b - 1\n'
               '    if n == 0\n        return c\n    return deep(n - 1) + a + b + c\nprint(deep(N))\n')
        rec = fit.check(analyze(src.replace('N', '1')), dev).recursion[0]
        assert rec["limited_by"] == "value stack" and rec["slots_per_level"] == 4   # n, a, b, c
        exe, err = fit.build_device_vm(dev)
        assert exe, err
        ok, _ = measure(compile_vm(src.replace('N', str(rec['levels'] - 1))).serialize(), tmp_path, exe)
        bad, _ = measure(compile_vm(src.replace('N', str(rec['levels']))).serialize(), tmp_path, exe)
        assert ok.returncode == 0, ok.stderr
        assert bad.returncode == 1 and "stack overflow" in bad.stderr and "too deep" not in bad.stderr


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
        assert "sqrt() needs floats" in r.verdict()
        r = report('print(float(3))\n', 'z80-16')
        assert "float() needs floats" in r.verdict()
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

    def test_packed_layout_is_the_8_bit_layout(self, tmp_path_factory):
        """With TILVM_PACK a value takes as many bytes on the desktop as on AVR / Z80 / eZ80."""
        sizes = host_sizes_cached(tmp_path_factory)
        for name in ('atmega1284', 'arduino-mega', 'z80-16', 'ti84ce'):
            assert sizes[name + "+pack"][1] == fit.vm_struct_layout(fit.DEVICES[name])['tv_val'], name


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


# ═══════════════════════════════════════════════════════════════
# REVIEW ROUND 1: regressions
# ═══════════════════════════════════════════════════════════════

def custom_device(tmp_path, name: str, **values) -> str:
    """A device JSON file based on `values['base']`; returns its path."""
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(dict(values, name=name)), encoding='utf-8')
    return str(path)


def run_tbc(exe: str, data: bytes, tmp_path, name: str = "p.tbc", args=()):
    path = tmp_path / name
    path.write_bytes(data)
    return subprocess.run([exe, str(path), *args], capture_output=True, text=True, timeout=60)


CONCAT = 'main()\n    var xs = []\n    for i in 0..N\n        xs = xs + [i]\n    print(len(xs), xs[N - 1])\n'


@needs_cc
class TestArrayConcatenation:
    """Finding 1: array + used one value-stack slot per element in the C VM."""

    def test_concat_needs_no_stack_for_its_elements(self, tmp_path):
        # 200 elements on a VM with 160 stack slots (atmega1284): it ran out of stack before
        data = compile_vm(CONCAT.replace('N', '200')).serialize()
        an = fit.Analysis(data, "<concat>")
        r, peak = measure(data, tmp_path, device_vm('atmega1284'))
        assert r.returncode == 0 and r.stdout == "200\n199\n", r.stderr
        assert peak['stack'] == an.slots_occ                    # exact: + is constant stack use now
        b = compile_vm('main()\n    let a = [1, 2, 3]\n    let b = [4, 5]\n    print(a + b)\n').serialize()
        _r, peak = measure(b, tmp_path, device_vm('atmega1284'))
        assert peak['stack'] == fit.Analysis(b, "<b>").slots_occ

    def test_both_vms_agree(self, tmp_path):
        src = ('main()\n    var a = [1, [2, 3], "s"]\n    let b = a + a + [] + ["z"]\n    b[1][0] = 9\n'
               '    a = a + [4]\n    print(a, b, len(b), [] + [], [[1]] + [[2]])\n'
               '    var big = []\n    for i in 0..100\n        big = big + [i, i * 2]\n'
               '    print(len(big), big[199], big[0])\n')      # 200 elements, 160 stack slots
        data = compile_vm(src).serialize()
        rc, ref = run_vm(data)
        assert rc == 0
        for exe in (device_vm('atmega1284'), cvm_binary("gcstress"), cvm_binary()):
            r = run_tbc(exe, data, tmp_path)
            assert (r.returncode, r.stdout) == (0, ref), (exe, r.stderr)

    def test_accumulating_loops_are_growing_allocations(self):
        for src, kind in ((CONCAT.replace('N', '50'), "array built up with + at line 4"),
                          ('main()\n    var s = ""\n    for i in 0..3000\n        s = s + "ab"\n    print(len(s))\n',
                           "string built up with + at line 4"),
                          ('main()\n    var s = ""\n    for i in 0..30\n        s += str(i)\n    print(s)\n',
                           "string built up with + at line 4"),
                          ('main()\n    var xs = [0]\n    for i in 0..5\n        xs = [i] + xs\n    print(xs)\n',
                           "array built up with + at line 4"),
                          ('struct Log\n    items: [int]\nimpl Log\n    add(self, x: int)\n'
                           '        self.items = self.items + [x]\nmain()\n    var log = Log { items: [] }\n'
                           '    for i in 0..3\n        log.add(i)\n    print(log.items)\n', None),
                          ('main()\n    var s = "x"\n    while len(s) < 100\n        s = s * 2\n    print(len(s))\n',
                           "string built up with * at line 4")):
            r = report(src, 'atmega1284')
            texts = [w.text for w in r.warnings]
            if kind is None:      # in a method called from the loop: not a loop allocation site
                assert not any("keeps allocating" in t for t in texts), texts
                continue
            assert any("keeps allocating" in t and kind in t for t in texts), (src, texts)
            assert not any("temporary objects" in n for n in r.notes), r.notes
        # temporaries: a new string per pass, nothing kept
        r = report('main()\n    for i in 0..9\n        print("item " + str(i))\n', 'atmega1284')
        assert any("1 loop allocates temporary objects on every pass (string)" in n for n in r.notes), r.notes
        assert not any("keeps allocating" in w.text for w in r.warnings)
        # numbers do not allocate
        r = report('main()\n    var t = 0\n    for i in 0..9\n        t = t + i * 2\n    print(t)\n', 'atmega1284')
        assert not r.notes or not any("temporary" in n for n in r.notes)

    def test_reviewer_repro(self, tmp_path, capsys):
        prog = write(tmp_path, CONCAT.replace('N', '200'))
        rc, out, _ = fit_main(capsys, prog, '--device', 'atmega1284', '--run')
        assert rc == 0 and "analysis bug" not in out and "stack overflow" not in out
        assert "keeps allocating (array built up with +" in flat(out)
        out_tbc = tmp_path / "c.tbc"
        r = cli('build', '--target', 'vm', '--device', 'atmega1284', prog, '-o', str(out_tbc))
        assert r.returncode == 0 and out_tbc.exists(), r.stderr
        r = subprocess.run([device_vm('atmega1284'), str(out_tbc)], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and r.stdout == "200\n199\n", r.stderr


class TestValueKinds:
    """The dataflow behind the type-dependent checks."""

    def kinds_of(self, src):
        an = analyze(src)
        assert an.kinds is not None
        return an, {an.where(f, pc): (fit.kind_names(a), fit.kind_names(b), fit.kind_names(r))
                    for (f, pc), (a, b, r) in an.kinds.binop.items()}

    def test_locals_params_returns_fields_globals(self):
        src = ('struct P\n    x: float\n    y: float\nimpl P\n    norm(self) -> float\n'
               '        return self.x * self.x + self.y * self.y\n'
               'fib(n: int) -> int\n    if n < 2\n        return n\n    return fib(n - 1) + fib(n - 2)\n'
               'var label = "n="\n'
               'main()\n    let p = P { x: 3.0, y: 4.0 }\n    var xs = [1]\n    xs = xs + [2]\n'
               '    print(label + str(fib(10)), p.norm(), xs)\n')
        _an, k = self.kinds_of(src)
        assert k['line 6 in P.norm'][2] == "float"
        assert k['line 10 in fib'] == ("int", "int", "int")
        assert k['line 15 in main'] == ("array", "array", "array")
        assert k['line 16 in main'] == ("nil/str", "str", "str")      # globals start as nil

    def test_lambdas_get_what_the_prelude_passes(self):
        _an, k = self.kinds_of('main()\n    let xs = [[1], [2]]\n    print(xs.reduce(|a, b| a + b, []))\n')
        a, b, r = k['line 3 in <lambda:3>']
        assert "array" in a and b == "any" and "array" in r

    def test_unknown_opcode_disables_kinds(self):
        an = fit.Analysis(patched_tbc(0xB0), "<p>")
        assert not an.complete and an.kinds is None


class TestLeanDifferences:
    """Finding 2: array + and map literals behave differently in TILVM_LEAN builds."""

    @pytest.mark.parametrize("device", ['z80-16', 'arduino-mega'])
    def test_array_concatenation_is_reported(self, device):
        for src in ('main()\n    var xs = [1, 2]\n    xs = xs + [3]\n    print(xs)\n',
                    'main()\n    var log = []\n    for i in 0..5\n        log += [i]\n    print(len(log))\n',
                    'join(a, b)\n    return a + b\nmain()\n    print(join([1], [2]))\n'):
            r = report(src, device)
            assert not r.fits and "not in the lean VM: array + array" in r.verdict(), (src, r.verdict())
            assert r.get_row('builtins')['status'] == "FAIL"
            assert report(src, 'cortex-m3-64k').fits
        # operand types only known at run time: a warning
        r = report('main()\n    let m = [[1], [2]]\n    print(m[0] + m[1])\n', device)
        assert r.fits and any("may join two arrays" in w.text for w in r.warnings)
        # no false alarms for numbers, strings and struct fields
        for src in ('fib(n: int) -> int\n    if n < 2\n        return n\n    return fib(n - 1) + fib(n - 2)\n'
                    'print(fib(10))\n',
                    'struct V\n    x: int\nimpl V\n    add(self, o: V) -> V\n        return V { x: self.x + o.x }\n'
                    'main()\n    print(V { x: 1 }.add(V { x: 2 }).x, "a" + "b")\n'):
            r = report(src, device)
            assert r.fits and not any("join two arrays" in w.text for w in r.warnings), r.verdict()

    @pytest.mark.parametrize("device", ['z80-16', 'arduino-mega'])
    def test_map_literals(self, device):
        r = report('main()\n    let m = {"a": 1, "b": 2, "a": 3}\n    print(len(m), m)\n', device)
        assert not r.fits and "repeated keys in a map literal" in r.verdict()
        assert any('repeats the key "a"' in e.text for e in r.errors)
        r = report('main()\n    let m = {1: "x", 1.0: "y"}\n    print(m)\n', device.replace('z80-16', 'arduino-mega'))
        assert not r.fits and "repeated keys" in r.verdict()
        r = report('main()\n    let k = "a"\n    let j = "b"\n    let m = {k: 1, j: 2}\n    print(m)\n', device)
        assert r.fits and any("computed keys" in w.text and "keeps both entries" in w.text for w in r.warnings)
        r = report('main()\n    let m = {"a": 1, "b": 2}\n    print(m)\n', device)
        assert r.fits and not r.warnings, [w.text for w in r.warnings]
        assert report('main()\n    let m = {"a": 1, "a": 3}\n    print(m)\n', 'cortex-m3-64k').fits

    def test_float_keys(self):
        r = report('main()\n    let m = {2.0: "x"}\n    print(m)\n', 'arduino-mega')
        assert not r.fits and "float keys in a map literal" in r.verdict()

    @needs_cc
    def test_the_lean_vm_really_differs(self, tmp_path):
        """What the checks describe is what a TILVM_LEAN build does (update the checks
        when the lean build gains these)."""
        lean, full = device_vm('arduino-mega'), device_vm('cortex-m3-64k')
        data = compile_vm('main()\n    print([1, 2] + [3])\n').serialize()
        r = run_tbc(lean, data, tmp_path)
        assert r.returncode == 1 and "unsupported operand types for +: array and array" in r.stderr
        assert run_tbc(full, data, tmp_path).stdout == "[1, 2, 3]\n"
        data = compile_vm('main()\n    let m = {"a": 1, "b": 2, "a": 3}\n    print(len(m))\n').serialize()
        assert run_tbc(lean, data, tmp_path).stdout == "3\n"
        assert run_tbc(full, data, tmp_path).stdout == "2\n"

    def test_build_for_a_lean_device_fails(self, tmp_path):
        prog = write(tmp_path, 'main()\n    var xs = [1, 2]\n    xs = xs + [3]\n    print(xs)\n')
        out = tmp_path / "p.tbc"
        r = cli('build', '--target', 'vm', '--device', 'z80-16', prog, '-o', str(out))
        assert r.returncode == 1 and not out.exists() and "array + array" in r.stderr


@needs_cc
class TestRunVerdicts:
    """Finding 3: device limits hit by the desktop VM too still mean "does not fit"."""

    def test_out_of_memory_everywhere(self, tmp_path, capsys):
        prog = write(tmp_path, 'var items = []\nfor i in 0..60000\n    items.push([i, i, i])\nprint(len(items))\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'esp32', '--run', '--json')
        data = json.loads(out)
        assert rc == 1 and data['fits'] is False and data['run']['error'] == "out of memory"
        assert data['run']['desktop']['error'] == "out of memory"
        assert any("the desktop VM stops there too" in e for e in data['errors'])

    def test_deep_recursion_everywhere(self, tmp_path, capsys):
        prog = write(tmp_path, 'f(n: int) -> int\n    if n == 0\n        return 0\n    return 1 + f(n - 1)\n'
                               'print(f(5000))\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'z80-16', '--run')
        assert rc == 1 and "verdict: does not fit: run: stack overflow (recursion too deep)" in out

    def test_program_errors_stay_warnings(self, tmp_path, capsys):
        prog = write(tmp_path, 'let xs = [1, 2, 3]\nprint(xs[5])\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'cortex-m3-64k', '--run')
        assert rc == 0 and "the program's own error" in flat(out)

    def test_killed_by_signal(self, tmp_path):
        r = fit.check(analyze('print(1)\n'), fit.DEVICES['esp32'])
        fit.apply_run(r, {'exit_code': -9, 'stdout': '', 'errors': '', 'stats': {}, 'error': None,
                          'timeout': False, 'desktop': None}, compared=False)
        assert not r.fits and "run: the VM was killed by signal 9" in r.verdict()

    @pytest.mark.skipif(sys.platform == 'win32', reason="POSIX signals")
    def test_killed_by_signal_end_to_end(self, tmp_path):
        fake = tmp_path / "fakevm"
        fake.write_text("#!/bin/sh\nkill -9 $$\n")
        fake.chmod(0o755)
        tbc = tmp_path / "p.tbc"
        tbc.write_bytes(tbc_bytes('print(1)\n'))
        run = fit._exec_vm(str(fake), str(tbc), [], 0, b"", 30, True)
        run['desktop'] = None
        r = fit.check(analyze('print(1)\n'), fit.DEVICES['esp32'])
        fit.apply_run(r, run, compared=False)
        assert "killed by signal 9" in r.verdict()

    def test_timeout_skips_the_desktop_run(self, monkeypatch):
        calls = []

        def fake_exec(exe, tbc, prog_args, max_steps, stdin_data, timeout, stats):
            calls.append(exe)
            return {'exit_code': None, 'stdout': '', 'errors': '', 'stats': {}, 'timeout': True}
        monkeypatch.setattr(fit, '_exec_vm', fake_exec)
        run = fit.run_on_device(fit.DEVICES['esp32'], tbc_bytes('print(1)\n'), [], timeout=0.5)
        assert len(calls) == 1 and run['desktop'] is None
        r = fit.check(analyze('print(1)\n'), fit.DEVICES['esp32'])
        fit.apply_run(r, run, compared=True)
        assert any("did not finish within the --timeout" in w.text for w in r.warnings)

    @slow
    def test_timeout_end_to_end(self, tmp_path, capsys):
        import time
        prog = write(tmp_path, 'var i = 0\nwhile true\n    i += 1\n')
        device_vm('cortex-m3-64k'), device_vm('desktop')           # built and cached: time the run only
        t0 = time.time()
        rc, out, _ = fit_main(capsys, prog, '--device', 'cortex-m3-64k', '--run', '--timeout', '1')
        assert rc == 0 and "did not finish within the --timeout" in flat(out)
        assert time.time() - t0 < 1.9                # one timeout, not two

    def test_stdin(self, tmp_path, capsys):
        prog = write(tmp_path, 'let name = input("name? ")\nprint("hi " + name)\n')
        text = tmp_path / "in.txt"
        text.write_text("Aigerim\n")
        rc, out, _ = fit_main(capsys, prog, '--device', 'esp32', '--run', '--stdin', str(text), '--show-output')
        assert rc == 0 and out.startswith("name? hi Aigerim\n")
        assert fit_main(capsys, prog, '--device', 'esp32', '--run', '--stdin', str(tmp_path / "none"))[0] == 2

    def test_output_difference_names_the_likely_cause(self, tmp_path, capsys):
        prog = write(tmp_path, 'var a = 300\nprint(a * a)\n')
        rc, out, _ = fit_main(capsys, prog, '--device', 'arduino-mega', '--run')
        assert "(integer width, float precision or the lean VM?)" in flat(out)

    def test_live_data_warning(self, tmp_path, capsys):
        dev = custom_device(tmp_path, "mega-1400", base="arduino-mega", heap=1400)
        prog = write(tmp_path, 'main()\n    var keep = []\n    for k in 0..3\n        var row = []\n'
                               '        for i in 0..33\n            row.push(i)\n        keep.push(row)\n'
                               '    var t = 0\n    for j in 0..300\n        let s = f"{j}"\n        t += len(s)\n'
                               '    print(len(keep), t)\n')
        rc, out, _ = fit_main(capsys, prog, '--device', dev, '--run')
        assert rc == 0 and "live data reached" in flat(out), out


class TestArgs:
    """Finding 4: args() puts every argument on the value stack."""

    @needs_cc
    def test_arguments_are_counted(self, tmp_path, capsys):
        prog = write(tmp_path, 'main()\n    let a = args()\n    print(len(a))\n')
        args = [str(i) for i in range(10)]
        rc, out, _ = fit_main(capsys, prog, '--device', 'cortex-m3-64k', '--run', '--json', '--', *args)
        data = json.loads(out)
        assert rc == 0 and data['run']['stdout'] == "10\n"
        assert data['run']['stats']['peak_stack'] == data['static']['stack_slots_written']
        assert not any("analysis bug" in w for w in data['warnings'])
        assert any("no command line" in n for n in data['notes'])

    def test_devices_with_a_command_line_warn(self):
        r = report('print(len(args()))\n', 'desktop')
        assert any("args() puts every program argument" in w.text for w in r.warnings)
        an = fit.Analysis(tbc_bytes('print(len(args()))\n'), "<a>", n_args=7)
        assert an.slots_occ == analyze('print(len(args()))\n').slots_occ + 7
        assert fit.DEVICES['nspire'].args and not fit.DEVICES['esp32'].args


TREE = ('struct Node\n    v: int\n    kids: [Node]\ndepth(n: Node) -> int\n    let ds = n.kids.map(|k| depth(k))\n'
        '    var best = 0\n    for d in ds\n        if d > best\n            best = d\n    return best + 1\n'
        'build(d: int) -> Node\n    if d == 0\n        return Node { v: 0, kids: [] }\n'
        '    return Node { v: d, kids: [build(d - 1)] }\nmain()\n    print(depth(build(DEPTH)))\n')
MUTUAL = ('ev(n: int) -> bool\n    let a = n\n    let b = n\n    let c = n\n    let d = n\n    if n == 0\n'
          '        return true\n    return od(n - 1)\n'
          'od(n: int) -> bool\n    if n == 0\n        return false\n    return ev(n - 1)\nmain()\n    print(ev(N))\n')


class TestRecursionLevels:
    """Finding 5: a level is one way around the recursion (frames and slots summed)."""

    def small_stack(self):
        values = fit.DEVICES['desktop'].as_dict()
        del values['name']
        values.update(stack=300, heap=8192, title="small stack")
        return fit.Device('small-stack', **values)

    def test_recursion_through_map(self, tmp_path):
        dev = fit.DEVICES['cortex-m3-64k']
        r = fit.check(analyze(TREE.replace('DEPTH', '3')), dev)
        rec = next(x for x in r.recursion if x['functions'][0] == 'depth')
        assert rec['frames_per_level'] == 3 and rec['unit'] == 'levels'
        assert rec['cycle'].startswith("depth -> .map() -> <lambda")
        assert any("Each level (depth -> .map() -> <lambda" in w.text for w in r.warnings)
        if fit.til.TILCompiler().find_c_compiler() is None:
            return
        levels = rec['levels']                 # a tree of `levels` levels surely fits
        ok, _ = measure(compile_vm(TREE.replace('DEPTH', str(levels - 1))).serialize(), tmp_path,
                        device_vm('cortex-m3-64k'))
        bad, _ = measure(compile_vm(TREE.replace('DEPTH', str(levels + 1))).serialize(), tmp_path,
                         device_vm('cortex-m3-64k'))
        assert ok.returncode == 0, ok.stderr
        assert bad.returncode == 1 and "recursion too deep" in bad.stderr

    @needs_cc
    def test_asymmetric_mutual_recursion_is_exact(self, tmp_path):
        dev = self.small_stack()
        rec = fit.check(analyze(MUTUAL.replace('N', '2')), dev).recursion[0]
        assert (rec['slots_per_level'], rec['frames_per_level'], rec['cycle']) == (6, 2, "ev -> od -> ev")
        exe, err = fit.build_device_vm(dev)
        assert exe, err
        levels = rec['levels']                 # ev(n) runs n // 2 + 1 levels
        ok, _ = measure(compile_vm(MUTUAL.replace('N', str(2 * levels - 1))).serialize(), tmp_path, exe)
        bad, _ = measure(compile_vm(MUTUAL.replace('N', str(2 * levels))).serialize(), tmp_path, exe)
        assert ok.returncode == 0, ok.stderr
        assert bad.returncode == 1 and "stack overflow" in bad.stderr

    @needs_cc
    def test_nested_map_is_not_recursion(self, tmp_path):
        data = compile_vm('main()\n    let m = [[1, 2], [3, 4]]\n    let d = m.map(|row| row.map(|x| x * 2))\n'
                          '    print(d)\n').serialize()
        an = fit.Analysis(data, "<nest>")
        assert not an.recursions and an.complete
        _r, peak = measure(data, tmp_path)
        assert (an.slots_occ, an.frames) == (peak['stack'], peak['frames'])

    @needs_cc
    def test_prelude_function_called_directly(self, tmp_path):
        """The callback edges hang on every call of a prelude function, not on its name."""
        src = ('big(x: int) -> int\n    let a = [' + ", ".join(["x"] * 20) + ']\n    return len(a)\n'
               'main()\n    print([1, 2].map(|v| v + 1))\n    print(__prelude_map([1, 2], big))\n')
        data = compile_vm(src).serialize()
        an = fit.Analysis(data, "<direct>")
        r, peak = measure(data, tmp_path)
        assert r.returncode == 0 and an.complete and not an.recursions
        assert (an.slots_occ, an.frames) == (peak['stack'], peak['frames'])

    @pytest.mark.parametrize("call", ["__prelude_map([n - 1], f)", "[n - 1].map(f)"])
    def test_recursion_through_a_prelude_function(self, tmp_path, call):
        """f -> __prelude_map -> f is a recursion also when the prelude function is called
        directly (it used to be missed: no callback edge without INVOKE)."""
        src = ('g() -> [int]\n    return [1].map(|x| x)\n'
               f'f(n: int) -> int\n    if n == 0\n        return 0\n    return {call}[0] + 1\n'
               'main()\n    print(g())\n    print(f(N))\n')
        r = fit.check(analyze(src.replace('N', '3')), fit.DEVICES['cortex-m3-64k'])
        rec = next(x for x in r.recursion if x['functions'] == ['f'])
        assert (rec['frames_per_level'], rec['cycle'], rec['unit']) == (2, "f -> .map() -> f", 'levels')
        if fit.til.TILCompiler().find_c_compiler() is None:
            return
        exe = device_vm('cortex-m3-64k')         # f(N) enters f N + 1 times: exact here
        ok, _ = measure(compile_vm(src.replace('N', str(rec['levels'] - 1))).serialize(), tmp_path, exe)
        bad, _ = measure(compile_vm(src.replace('N', str(rec['levels']))).serialize(), tmp_path, exe)
        assert ok.returncode == 0, ok.stderr
        assert bad.returncode == 1 and "recursion too deep" in bad.stderr

    @needs_cc
    @pytest.mark.parametrize("shape, src, device", [
        ("expression before the call",            # the deepest level returns before pushing n, n, n, 1
         'deep(n: int) -> int\n    if n == 0\n        return 0\n'
         '    return n + (n * (n + (n - (1 + deep(n - 1)))))\nmain()\n    print(deep(N))\n', 'small-stack'),
        ("retry",                                 # ... before entering the retry block
         'r(n: int) -> int\n    if n == 0\n        return 0\n    let v = retry(1)\n        r(n - 1)\n'
         '    return v + 1\nmain()\n    print(r(N))\n', 'small-stack'),
        ("retry on avr", 'r(n: int) -> int\n    if n == 0\n        return 0\n    let v = retry(1)\n'
         '        r(n - 1)\n    return v + 1\nmain()\n    print(r(N))\n', 'atmega1284'),
        ("base case calls a chain",
         'l2(x: int) -> int\n    return x + 1\nl1(x: int) -> int\n    return l2(x) + 1\n'
         'count(n: int) -> int\n    if n == 0\n        return l1(0)\n    return 1 + count(n - 1)\n'
         'main()\n    print(count(N))\n', 'arduino-mega'),
    ])
    def test_levels_are_a_guarantee(self, tmp_path, shape, src, device):
        """The reported number of levels always runs on the device's VM, and at most one
        more does (the review's shapes; N + 1 levels for f(N))."""
        dev = self.small_stack() if device == 'small-stack' else fit.DEVICES[device]
        rec = fit.check(analyze(src.replace('N', '1')), dev).recursion[0]
        exe, err = fit.build_device_vm(dev)
        assert exe, err
        levels = rec['levels']
        ok, _ = measure(compile_vm(src.replace('N', str(levels - 1))).serialize(), tmp_path, exe)
        bad, _ = measure(compile_vm(src.replace('N', str(levels + 1))).serialize(), tmp_path, exe)
        assert ok.returncode == 0, (shape, ok.stderr)
        assert bad.returncode == 1, (shape, levels)

    def test_cycles_that_avoid_the_entry_count_calls(self):
        src = ('a(n: int) -> int\n    if n <= 0\n        return 0\n    return b(n - 1) + 1\n'
               'b(n: int) -> int\n    if n % 2 == 0\n        return a(n)\n    return c(n - 1)\n'
               'c(n: int) -> int\n    return b(n) + 1\nprint(a(9))\n')
        r = report(src, 'cortex-m3-64k')
        rec = r.recursion[0]
        assert rec['unit'] == 'calls' and set(rec['functions']) == {'a', 'b', 'c'}
        assert any("in more than one cycle" in w.text and "nested calls fit" in w.text for w in r.warnings)

    def test_frames_row_shows_frames_per_level(self):
        r = fit.check(analyze(TREE.replace('DEPTH', '3')), fit.DEVICES['cortex-m3-64k'])
        assert r.get_row('frames')['note'] == "+3 per recursion level"


class TestCliAndDevices:
    def test_build_with_device_rejects_bytecode_input(self, tmp_path):
        prog = write(tmp_path, 'print(chr(65) + "x".repeat(2))\n')
        tbc = tmp_path / "p.tbc"
        assert cli('build', '--target', 'vm', prog, '-o', str(tbc)).returncode == 0
        out = tmp_path / "out.tbc"
        r = cli('build', '--target', 'vm', '--device', 'z80-16', str(tbc), '-o', str(out))
        assert r.returncode == 1 and "on a .til source" in r.stderr and not out.exists()
        assert "Axx" not in r.stdout                            # the program did not run

    def test_fit_help(self):
        r = cli('fit', '--help')
        assert r.returncode == 0 and r.stdout.startswith("usage: til fit")

    def test_program_arguments_after_double_dash(self, tmp_path):
        prog = write(tmp_path, 'print("ok")\n')
        r = cli('fit', prog, '--device', 'esp32', '--', '--help')
        assert r.returncode == 0 and "verdict: fits" in r.stdout

    @pytest.mark.parametrize("values, message", [
        ({"float": True}, "float must be a whole number"),
        ({"int_bits": 16.0}, "int_bits must be a whole number"),
        ({"lean": "no"}, "lean must be true or false"),
        ({"heap": "4096"}, "heap must be a whole number"),
        ({"title": 3}, "title must be a string"),
        ({"base": "ti84ce", "float": 2}, "float 2 (double) is 32-bit with the ez80 toolchain"),
        ({"base": 7}, "unknown base device"),
        ({"name": 12}, "name must be a string"),
    ])
    def test_device_json_types(self, tmp_path, capsys, values, message):
        dev = tmp_path / "d.json"
        dev.write_text(json.dumps(dict({"base": "cortex-m3-64k"}, **values)))
        prog = write(tmp_path, 'print(1.5)\n')
        rc, _out, err = fit_main(capsys, prog, '--device', str(dev))
        assert rc == 2 and message in err, err

    def test_no_math(self, tmp_path, capsys):
        dev = custom_device(tmp_path, "nomath", base="cortex-m3-64k", no_math=True)
        prog = write(tmp_path, 'print(sin(1.0))\n')
        rc, out, _ = fit_main(capsys, prog, '--device', dev)
        assert rc == 1 and "sin() needs <math.h>" in flat(out) and "(no math library)" in out
        prog = write(tmp_path, 'print(sin(1.0), cos(1.0))\n')
        rc, out, _ = fit_main(capsys, prog, '--device', dev)
        assert "sin(), cos() need <math.h>" in flat(out)
        prog = write(tmp_path, 'let x = 2\nprint(x ** 0.5)\n')
        rc, out, _ = fit_main(capsys, prog, '--device', dev)
        assert rc == 0 and "integer powers only" in flat(out)

    @needs_cc
    def test_no_math_run(self, tmp_path, capsys):
        dev = custom_device(tmp_path, "nomath", base="cortex-m3-64k", no_math=True)
        prog = write(tmp_path, 'print(sin(1.0))\n')
        rc, out, _ = fit_main(capsys, prog, '--device', dev, '--run')
        assert rc == 1 and "need a build with <math.h>" in out

    def test_device_service_warnings(self, tmp_path):
        r = report('print(input("x? "))\n', 'cortex-m3-64k')
        assert any("input() returns an empty string" in w.text for w in r.warnings)
        dev = fit.load_device(custom_device(tmp_path, "nofloat", base="cortex-m3-64k", float=0))
        r = fit.check(analyze('print(random())\n'), dev)
        assert any("random() returns an integer 0..32767" in w.text for w in r.warnings)

    def test_little_room_left(self):
        src = 'main()\n    let a = [' + ", ".join("0" for _ in range(130)) + ']\n    print(len(a))\n'
        r = report(src, 'atmega1284')
        assert r.fits and any("little room left" in w.text and "value-stack slots" in w.text for w in r.warnings)

    def test_unknown_builtin(self):
        data = bytearray(tbc_bytes('print(len("ab"))\n'))
        mod = til.TBCModule.from_bytes(bytes(data))
        code_start = len(data) - 2 - 6 * len(mod.lines) - len(mod.code)
        at = code_start + bytes(mod.code).index(bytes([til.Op.NATIVE, til.NATIVES.index('len')]))
        data[at + 1] = 99
        r = fit.check(fit.Analysis(bytes(data), "<p>"), fit.DEVICES['desktop'])
        assert any("unknown builtins (native#99)" in w.text for w in r.warnings)

    def test_grammar(self):
        r = report('main()\n    for i in 0..3\n        print(f"i = {i}")\n', 'cortex-m3-64k')
        assert any(n.startswith("1 loop allocates temporary objects") for n in r.notes), r.notes


@needs_cc
class TestPackedBigHeap:
    """Finding 8: TILVM_PACK with a heap above 64 KB (eZ80 custom devices)."""

    def test_no_misaligned_headers(self, tmp_path):
        exe = str(tmp_path / "bigpack")
        r = subprocess.run([CC, '-std=c99', '-g', '-O1', '-fsanitize=undefined', '-fno-sanitize-recover=undefined',
                            '-DTILVM_STATS', '-DTILVM_PACK', '-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1',
                            '-DTILVM_HEAP_SIZE=100000', '-o', exe, os.path.join(VM_DIR, 'tilvm.c'),
                            os.path.join(VM_DIR, 'main.c'), '-lm'], capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip("no UBSan runtime: " + r.stderr[-200:])
        for name in ('structs', 'closures', 'maps', 'strings'):
            res = run_tbc(exe, compile_vm(CORPUS[name]).serialize(), tmp_path)
            assert res.returncode == 0 and "runtime error" not in res.stderr, (name, res.stderr)

    def test_custom_ez80_device_runs(self, tmp_path, capsys):
        dev = custom_device(tmp_path, "ce-big", base="ti84ce", heap=100000, ram=262144)
        prog = write(tmp_path, CORPUS['structs'])
        rc, out, _ = fit_main(capsys, prog, '--device', dev, '--run')
        assert rc == 0 and "verdict: fits" in out, out


class TestCacheDirectory:
    """Finding 10: cached VM executables never come from a shared, predictable place."""

    @pytest.mark.skipif(not hasattr(os, 'getuid'), reason="POSIX ownership")
    def test_private_fallback(self, tmp_path, monkeypatch):
        shared = tmp_path / "shared"
        shared.mkdir()
        monkeypatch.setenv('HOME', '/proc/nonexistent')
        monkeypatch.delenv('TIL_FIT_CACHE', raising=False)
        monkeypatch.setattr(til.tempfile, 'tempdir', str(shared))
        d = fit._cache_dir()
        assert os.path.dirname(d) == str(shared) and os.stat(d).st_mode & 0o077 == 0
        assert os.stat(d).st_uid == os.getuid()
        # a directory with that name that others can write to is not used
        os.chmod(d, 0o777)
        d2 = til.user_cache_dir()
        assert d2 != d and os.stat(d2).st_mode & 0o077 == 0

    @pytest.mark.skipif(not hasattr(os, 'getuid'), reason="POSIX ownership")
    @needs_cc
    def test_planted_binary_is_not_run(self, tmp_path):
        shared = tmp_path / "shared"
        shared.mkdir()
        dev = fit.DEVICES['esp32']
        cc = til.TILCompiler().find_c_compiler()
        import hashlib
        digest = hashlib.sha256((cc + ' ' + DEFAULT_VM_OPT + ' ' + ' '.join(fit.device_vm_flags(dev))).encode())
        for n in ('tilvm.c', 'tilvm.h', 'main.c'):
            digest.update(open(os.path.join(VM_DIR, n), 'rb').read())
        planted = shared / f"tilvm-fit-esp32-{digest.hexdigest()[:16]}"
        planted.write_text('#!/bin/sh\necho PLANTED >&2\nexit 0\n')
        planted.chmod(0o755)
        prog = write(tmp_path, 'print("hi")\n')
        env = dict(os.environ, HOME='/proc/nonexistent', TMPDIR=str(shared))
        env.pop('TIL_FIT_CACHE', None)
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'til.py'), 'fit', prog, '--device', 'esp32',
                            '--run', '--json'], capture_output=True, text=True, env=env, timeout=300)
        data = json.loads(r.stdout)
        assert "PLANTED" not in data['run']['errors'] and data['run']['stdout'] == "hi\n"
        assert os.path.isdir(shared / f"til-cache-{os.getuid()}")      # built in the private directory
