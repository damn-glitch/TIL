"""
TIL 3.1 feature 9: the TIL VM on the Z80 (16-bit lean build, SDCC) and the
TI-84 Plus CE port. See vm/ports/z80/RESULTS.md and vm/ports/ti84ce/README.md.

Always on (fast):
  * the Z80 and CE VM profiles compile with -std=c99 -Wall -Wextra -pedantic
    -Werror (gcc and clang);
  * desktop builds with the Z80 profile (TILVM_LEAN + TILVM_ADDR16 + 16-bit ints)
    and the related lean / 16-bit-offset profiles run a corpus of programs and
    print exactly what the reference VM prints;
  * tbc2appvar.py writes valid AppVar files, and the TI-84 Plus CE host code
    (vm/ports/ti84ce/src/main.c) runs programs from AppVars against desktop
    stand-ins of the CE libraries.

Slow, TIL_TEST_CROSS=1 (needs sdcc + sz80 from SDCC 4.2, or the CE toolchain):
  * the Z80 image builds, its code ends below 0xC000, the interpreter's stack
    frame stays below 128 bytes (SDCC bug, see vm/tilvm.c), and programs and
    examples give the same output on the simulated Z80 as on the reference VM;
  * the CE program builds with CEdev (only where the toolchain is installed).
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from tilhelpers import ROOT, VM_DIR, CC, needs_cc, run_vm, tbc_bytes, til

Z80_DIR = os.path.join(VM_DIR, 'ports', 'z80')
CE_DIR = os.path.join(VM_DIR, 'ports', 'ti84ce')
STRICT = ['-std=c99', '-Wall', '-Wextra', '-pedantic', '-Werror']
cross = pytest.mark.skipif(not os.environ.get('TIL_TEST_CROSS'), reason="set TIL_TEST_CROSS=1")


def make_var(path, var):
    """Flags of a make variable (with continuation lines), e.g. the VM profile."""
    with open(path) as f:
        text = f.read()
    m = re.search(r'^%s\s*=\s*((?:.*\\\n)*.*)$' % var, text, re.M)
    assert m, f"{var} not found in {path}"
    return m.group(1).replace('\\\n', ' ').split()


Z80_PROFILE = make_var(os.path.join(Z80_DIR, 'Makefile'), 'PROFILE')
CE_PROFILE = make_var(os.path.join(CE_DIR, 'makefile'), 'TIL_PROFILE')
HOSTED = [f for f in Z80_PROFILE if f != '-DTILVM_NO_STDIO']   # desktop runner needs stdio output

# desktop profiles that run the corpus: the exact Z80 one, plus lean without
# 16-bit offsets and 16-bit offsets without lean (the code paths they select)
DESKTOP_PROFILES = {
    'z80': HOSTED,
    'lean32': ['-DTILVM_LEAN', '-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1', '-DTILVM_HEAP_SIZE=8192',
               '-DTILVM_STACK_SIZE=256', '-DTILVM_MAX_FRAMES=48'],
    'addr16': ['-DTILVM_ADDR16', '-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1', '-DTILVM_HEAP_SIZE=16384',
               '-DTILVM_STACK_SIZE=512', '-DTILVM_MAX_FRAMES=64'],
}

PROGRAMS = {
    'ops': '''\
main()
    var a = 1234
    let b = -77
    print(a + b, a - b, a * 3, a / 7, a % 7, -a, 2 ** 10)
    print(a / b, a % b, (-a) / 7, (-a) % 7, 7 / -2, -7 % 3)
    print(a & 255, a | 4096, a ^ 1023, ~a, a << 3, a >> 2, b >> 1)
    print(a < b, a <= a, a > b, a >= 5000, a == 1234, a != 1234)
    print(not true, not 0, not "", not [1], true and false, true or false)
    var x = 5
    x += 3
    x *= 2
    x -= 1
    x /= 3
    print(x)
''',
    'strings': '''\
main()
    let s = "Hello, Z80"
    let n = 42
    print(s + "!", s * 2, "ab" * 3, "" * 5, "x" * 0)
    print(f"{s} has {len(s)} bytes; n={n} n*2={n * 2}")
    print(s[0], s[4], len(""), s.len(), s.is_empty(), "".is_empty())
    print(s.contains("Z8"), s.starts_with("Hell"), s.ends_with("80"), s.find("lo"), s.index_of("q"))
    print(s.to_upper(), s.to_lower(), s.slice(7, 10), s.slice(3), s.slice(-5, 99), s.eq("Hello, Z80"))
    print("123".to_int() + 1, int("-45") * 2, str(99) + "!", bool(""), bool("a"))
    print("abc" < "abd", "b" > "abc", "abc" == "abc", "x" != "y")
    var total = 0
    for c in "AB"
        total += c
    print(total)
''',
    'arrays': '''\
main()
    var a = [5, 3, 9]
    a.push(1)
    a[0] = 50
    a[1] += 7
    print(a, len(a), a.len(), a[2], a.get(3), a.first(), a.last())
    print(a.pop(), a, a.contains(9), a.index_of(9), a.index_of(77), a.count(9))
    a.set(0, 1)
    print(a, a.is_empty(), [].is_empty(), [1, [2, [3, "x"]]], [1, 2] == [1, 2], [1, 2] == [2, 1])
    let nums = [1, 2, 3, 4, 5, 6]
    print(nums.map(|x| x * 10))
    print(nums.filter(|x| x % 2 == 0))
    print(nums.reduce(|s, x| s + x, 0))
    print(nums.any(|x| x > 5), nums.all(|x| x > 0))
    nums.each(|x| print(x * x))
    var b = []
    for i in 0..5
        b.push(i * i)
    for i in 3..=4
        b.push(-i)
    print(b, range(4), range(2, 5))
    b.clear()
    print(b, len(b))
''',
    'maps': '''\
main()
    var m = {"one": 1, "two": 2}
    m["three"] = 3
    m["one"] = 11
    print(m, len(m), m["two"], m.get("three"), m.get_or("four", 4))
    print(m.has("one"), m.contains("zero"), m.keys())
    m.delete("two")
    m.remove("nothing")
    print(m, m.is_empty(), {}.is_empty())
    var total = 0
    for k in m
        total += m[k]
    print(total)
    let p = {1: "a", 2: "b"}
    let q = {2: "b", 1: "a"}
    print(p == q, p == {1: "a"}, p != {1: "a", 2: "c"}, p[2])
    var inv = {}
    for k in p
        inv[p[k]] = k
    print(inv)
''',
    'structs': '''\
struct Point
    x: int
    y: int

impl Point
    sum(self) -> int
        return self.x + self.y
    moved(self, dx: int, dy: int) -> Point
        return Point { x: self.x + dx, y: self.y + dy }

enum Shape
    Square(int)
    Rect(int, int)
    Dot

area(s: Shape) -> int
    return match s
        Shape.Square(a) => a * a
        Shape.Rect(w, h) => w * h
        Shape.Dot => 0

describe(v) -> str
    return match v
        [] => "empty"
        [a] => f"one: {a}"
        [a, b] => f"two: {a} {b}"
        _ => "many"

main()
    var p = Point { x: 3, y: 4 }
    p.x += 10
    print(p, p.x, p.sum(), p.moved(1, 1))
    print(p == Point { x: 13, y: 4 }, p != Point { x: 0, y: 0 })
    for s in [Shape.Square(5), Shape.Rect(2, 7), Shape.Dot]
        print(s, area(s))
    print(describe([]), describe([1]), describe([1, 2]), describe([1, 2, 3]))
''',
    'closures': '''\
make_adder(k: int)
    return |x: int| x + k

greet(name: str, greeting: str = "Hello") -> str
    return f"{greeting}, {name}"

var calls = 0

bump() -> int
    calls += 1
    return calls

apply(f, x)
    return f(x)

main()
    var base = 1
    let f = |x: int| x + base
    base = 100
    print(f(1))
    let add5 = make_adder(5)
    print(add5(10), [1, 2, 3].map(add5), apply(add5, 1))
    print(greet("Aigerim"), greet("Ivan", "Privet"))
    bump()
    bump()
    print(calls, bump())
    var fs = []
    for i in 0..3
        fs.push(|x| x * i)
    print(fs.map(|g| g(10)))
''',
    'results': '''\
parse(s: str) -> Result<int, str>
    if s == ""
        return Err("empty")
    return Ok(s.to_int())

double(s: str) -> Result<int, str>
    let v = parse(s)?
    return Ok(v * 2)

find(xs, target: int) -> Option<int>
    for i in 0..len(xs)
        if xs[i] == target
            return Some(i)
    return None

main()
    print(double("21"), double(""), parse("7").unwrap(), parse("").unwrap_or(-1))
    print(parse("3").is_ok(), parse("").is_err(), parse("").error())
    let f = find([4, 5, 6], 6)
    print(f, f.is_some(), find([1], 9), find([1], 9).is_none(), find([1], 9).unwrap_or(0))
    print(Some(3).expect("must"), Ok("fine").unwrap())
''',
    'errors': '''\
risky(n: int) -> int
    if n > 2
        panic(f"too big: {n}")
    return n

main()
    var tries = 0
    retry(3)
        tries += 1
        assert tries >= 2, "warming up"
        print(f"ok after {tries}")
    else
        print("never")
    retry(1)
        let x = [1, 2][5]
    else
        print(f"caught: {error}")
    retry(1)
        let m = {"a": 1}
        print(m["b"])
    else
        print(f"caught: {error}")
    retry(1)
        print(10 / (tries - tries))
    else
        print(f"caught: {error}")
    retry(1)
        risky(5)
    else
        print(f"caught: {error}")
    retry(1)
        let g = 5
        g(1)
    else
        print(f"caught: {error}")
    print(risky(2))
    let none_value = None
    print(none_value.unwrap())
''',
    'gc': '''\
fib(n: int) -> int
    if n < 2
        return n
    return fib(n - 1) + fib(n - 2)

main()
    var keep = []
    for i in 0..300
        let s = f"item-{i}"
        let a = [i, i + 1, s]
        if i % 50 == 0
            keep.push(a)
    print(keep)
    var acc = ""
    for i in 0..40
        acc = acc + str(i % 10)
    print(acc, len(acc))
    print(fib(15))
    exit(3)
''',
    'deep': '''\
depth(n: int) -> int
    if n == 0
        return 0
    return 1 + depth(n - 1)

main()
    var a = [0]
    for i in 0..14
        a = [a, i]
    print(a)
    print(a == a, [a] == [a])
    print(depth(25))
''',
}

EXAMPLES_16BIT = ['01_hello.til', '05_control_flow.til', '06_enums.til', '08_imports.til',
                  '10_verified.til', os.path.join('kazakh', 'salam.til')]


def _build(name, flags, sources, extra=()):
    exe = os.path.join(tempfile.mkdtemp(prefix='tilvm-z80-'), name)
    r = subprocess.run([CC, *STRICT, '-O1', *flags, *extra, '-o', exe, *sources, '-lm'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


_BINARIES = {}


def desktop_vm(profile):
    if profile not in _BINARIES:
        _BINARIES[profile] = _build('tilvm-' + profile, DESKTOP_PROFILES[profile],
                                    [os.path.join(VM_DIR, 'tilvm.c'), os.path.join(VM_DIR, 'main.c')])
    return _BINARIES[profile]


def run_c(exe, src, cwd=None):
    fd, path = tempfile.mkstemp(suffix='.tbc')
    with os.fdopen(fd, 'wb') as f:
        f.write(src if isinstance(src, bytes) else tbc_bytes(src))
    try:
        r = subprocess.run([exe, path], capture_output=True, timeout=120, cwd=cwd)
        return r.returncode, r.stdout.decode('utf-8', 'replace'), r.stderr.decode('utf-8', 'replace')
    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════
# STRICT BUILDS OF THE PORT PROFILES
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("profile", ["z80", "ce"])
def test_port_profiles_build_strictly(compiler, profile, tmp_path):
    if not shutil.which(compiler):
        pytest.skip(f"{compiler} not installed")
    flags = Z80_PROFILE if profile == 'z80' else CE_PROFILE
    r = subprocess.run([compiler, *STRICT, *flags, '-c', os.path.join(VM_DIR, 'tilvm.c'),
                        '-o', str(tmp_path / 'tilvm.o')], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_profiles_are_the_documented_ones():
    assert '-DTILVM_ADDR16' in Z80_PROFILE and '-DTILVM_LEAN' in Z80_PROFILE
    assert '-DTILVM_INT_BITS=16' in Z80_PROFILE and '-DTILVM_FLOAT=0' in Z80_PROFILE
    assert '-DTILVM_ADDR16' in CE_PROFILE and '-DTILVM_FLOAT=1' in CE_PROFILE


# ═══════════════════════════════════════════════════════════════
# DESKTOP BUILDS WITH THE 8-BIT PROFILES == REFERENCE VM
# ═══════════════════════════════════════════════════════════════

@needs_cc
@pytest.mark.parametrize("profile", sorted(DESKTOP_PROFILES))
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_profile_matches_reference_vm(profile, name):
    src = PROGRAMS[name]
    rc, out = run_vm(src)
    crc, cout, cerr = run_c(desktop_vm(profile), src)
    assert (crc, cout) == (rc, out), cerr


@needs_cc
def test_z80_profile_error_reports_and_limits():
    exe = desktop_vm('z80')
    rc, out, err = run_c(exe, 'main()\n    var a = [1]\n    print(a[3])\n')
    assert rc == 1 and err.startswith("runtime error: index out of bounds: index 3, length 1")
    rc, _, err = run_c(exe, 'main()\n    print(70000)\n')          # does not fit 16 bits
    assert rc == 2 and "too large" in err
    rc, _, err = run_c(exe, 'main()\n    print(1.5)\n')
    assert rc == 2 and "floats" in err
    rc, out, err = run_c(exe, 'main()\n    print("a b".count("b"))\n')   # not in lean builds
    assert rc == 1 and "no method 'count' on str" in err
    rc, out, err = run_c(exe, 'main()\n    var s = "x"\n    while true\n        s = s + s\n')
    assert rc == 1 and "out of memory" in err
    # the Z80 image leaves out effect handlers and the screen (TILVM_NO_EFFECTS, TILVM_NO_GFX)
    rc, out, err = run_c(exe, 'let r = handle\n    perform Ask()\nwith\n    Ask => resume(1)\nprint(r)\n')
    assert rc == 1 and "effect handlers are not available in this VM build" in err
    rc, out, err = run_c(exe, 'Ask() -> int\n    return 7\nprint(perform Ask())\n')   # default handler
    assert (rc, out) == (0, "7\n"), err
    rc, out, err = run_c(exe, 'main()\n    screen_init(8, 8)\n')
    assert rc == 1 and "screen_init() is not available in this VM build" in err


@needs_cc
def test_addr16_rejects_programs_over_64k():
    data = tbc_bytes('main()\n' + ''.join('    print("%s")\n' % (c * 30000) for c in 'xyz'))
    assert len(data) > 0xFFFF
    rc, out, err = run_c(desktop_vm('addr16'), data)
    assert rc == 2 and "error -10" in err and "larger than 64 KB" in err
    rc, out, err = run_c(desktop_vm('lean32'), data)                  # 32-bit offsets: fine
    assert (rc, len(out)) == (0, 90003), err


# ═══════════════════════════════════════════════════════════════
# LEAN BUILDS RUN EVERY OPCODE LIKE THE DEFAULT BUILD
# ═══════════════════════════════════════════════════════════════

# programs of the speed corpus that lean builds cannot run like the default build
LEAN_SKIP = {'natives_and_pop': "type_of() is not in lean builds",
             'strings_and_maps': "str.split is not in lean builds",
             'retry_and_timeout': "its step count depends on the clock"}


def run_stats(exe, data):
    """(exit code, stdout, stderr) of `tilvm --stats` (the stats line has the steps)."""
    fd, path = tempfile.mkstemp(suffix='.tbc')
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    try:
        r = subprocess.run([exe, '--stats', path], capture_output=True, timeout=120)
        return r.returncode, r.stdout.decode('utf-8', 'replace'), r.stderr.decode('utf-8', 'replace')
    finally:
        os.unlink(path)


@needs_cc
def test_lean_build_runs_every_opcode_like_the_default_build():
    """Lean builds run the data opcodes and every superinstruction in lean_op() and
    the control flow (calls, returns, effects) in a smaller loop (vm/tilvm.c): the
    same output, errors, steps and heap use as the default build. The speed corpus
    uses every superinstruction; the effect programs use HANDLE / PERFORM / RESUME."""
    import test_effects
    import test_speed
    sources = [os.path.join(VM_DIR, 'tilvm.c'), os.path.join(VM_DIR, 'main.c')]
    lean = _build('tilvm-lean64', ['-DTILVM_LEAN'], sources)
    full = _build('tilvm-full64', [], sources)
    progs = {n: s for n, s in test_speed.CORPUS.items() if n not in LEAN_SKIP}
    for n in ('SPEC_EXAMPLE', 'GC_CLOSURES', 'MANY_PERFORMS'):
        progs[n] = getattr(test_effects, n)
    seen = set()
    for name, src in sorted(progs.items()):
        data = tbc_bytes(src)
        seen |= test_speed.opcodes(data)
        assert run_stats(lean, data) == run_stats(full, data), name
    supers = {v for k, v in vars(til.Op).items() if isinstance(v, int) and 0x80 <= v <= 0xBF}
    assert supers and supers <= seen, sorted(hex(op) for op in supers - seen)
    assert {til.Op.HANDLE, til.Op.END_HANDLE, til.Op.PERFORM, til.Op.RESUME} <= seen


def example_tbc(rel):
    path = os.path.join(ROOT, 'examples', rel)
    with open(path, encoding='utf-8') as f:
        return til.compile_to_bytecode(f.read(), path).serialize()


@needs_cc
def test_examples_on_the_z80_profile():
    for rel in EXAMPLES_16BIT:
        data = example_tbc(rel)
        rc, out = run_vm(data)
        crc, cout, cerr = run_c(desktop_vm('z80'), data)
        assert (crc, cout) == (rc, out), f"{rel}: {cerr}"


# ═══════════════════════════════════════════════════════════════
# TI-84 PLUS CE: APPVARS AND THE HOST CODE (desktop stand-ins)
# ═══════════════════════════════════════════════════════════════

def tbc2appvar():
    spec = importlib.util.spec_from_file_location('tbc2appvar', os.path.join(CE_DIR, 'tbc2appvar.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_appvar_format():
    t = tbc2appvar()
    data = tbc_bytes('main()\n    print("hi")\n')
    blob = t.make_appvar('HELLO', data)
    assert blob[:11] == b'**TI83F*\x1a\x0a\x00' and len(blob) == 55 + 19 + len(data) + 2
    assert blob[55:57] == b'\x0d\x00' and blob[59] == 0x15 and blob[60:68] == b'HELLO\0\0\0'
    assert blob[69] == 0x80                                   # archived by default
    assert sum(blob[55:-2]) & 0xFFFF == blob[-2] | (blob[-1] << 8)
    assert t.read_appvar(blob) == ('HELLO', data, True)
    assert t.read_appvar(t.make_appvar('a1', b'TILB', archived=False)) == ('a1', b'TILB', False)
    for bad in ('', '1ABC', 'TOOLONGNAME', 'A-B'):
        with pytest.raises(ValueError):
            t.make_appvar(bad, data)
    with pytest.raises(ValueError):
        t.make_appvar('BIG', b'x' * 70000)
    with pytest.raises(ValueError):
        t.read_appvar(blob[:-1] + bytes([blob[-1] ^ 1]))       # checksum


def test_tbc2appvar_cli(tmp_path):
    prog = tmp_path / 'p.til'
    prog.write_text('main()\n    print(1)\n')
    r = subprocess.run([sys.executable, os.path.join(CE_DIR, 'tbc2appvar.py'), str(prog), 'PROG',
                        '-o', str(tmp_path / 'PROG.8xv')], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    name, data, archived = tbc2appvar().read_appvar((tmp_path / 'PROG.8xv').read_bytes())
    assert name == 'PROG' and data[:4] == b'TILB' and archived


@needs_cc
def test_ce_host_runs_appvars(tmp_path):
    exe = _build('tilce', CE_PROFILE + ['-I' + os.path.join(CE_DIR, 'desktop'), '-I' + VM_DIR],
                 [os.path.join(CE_DIR, 'src', 'main.c'), os.path.join(CE_DIR, 'src', 'tilvm_ce.c'),
                  os.path.join(CE_DIR, 'desktop', 'ce_shim.c')])
    t = tbc2appvar()
    progs = {'AMAPS': PROGRAMS['maps'], 'BFILES': 'main()\n    write_file("NOTE", "saved")\n'
             '    print(read_file("NOTE"))\n    print(file_exists("NOTE"), file_exists("NOPE"))\n'
             '    print(sqrt(2.0) > 1.41)\n'}
    for name, src in progs.items():
        (tmp_path / (name + '.8xv')).write_bytes(t.make_appvar(name, tbc_bytes(src)))
    (tmp_path / 'OTHER.8xv').write_bytes(t.make_appvar('OTHER', b'not bytecode'))
    env = dict(os.environ, TILCE_DIR=str(tmp_path))
    # menu: down + enter picks the second program
    for keys, name in (('0x09', 'AMAPS'), ('0x01,0x09', 'BFILES')):
        env['TILCE_KEYS'] = keys
        r = subprocess.run([exe], capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stderr
        screens = r.stdout.split('\f')                  # os_ClrHome() starts a screen
        assert screens[-1] == ''                            # cleared when leaving
        assert screens[-3].startswith('TIL programs:\n') and f'> {name}\n' in screens[-3]
        assert 'OTHER' not in r.stdout                      # not bytecode: not listed
        body = screens[-2]
        assert body.endswith('\n[press a key]')
        cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            rc, expected = run_vm(progs[name])
        finally:
            os.chdir(cwd)
        assert rc == 0 and body[:-len('\n[press a key]')] == expected
    assert t.read_appvar((tmp_path / 'NOTE.8xv').read_bytes())[:2] == ('NOTE', b'saved')


# ═══════════════════════════════════════════════════════════════
# SIMULATED Z80 (slow: SDCC builds the VM in a few minutes)
# ═══════════════════════════════════════════════════════════════

def run_sim_module():
    spec = importlib.util.spec_from_file_location('z80_run_sim', os.path.join(Z80_DIR, 'run_sim.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


needs_sdcc = pytest.mark.skipif(not (shutil.which('sdcc') and shutil.which('sz80')),
                                reason="needs sdcc and sz80 (SDCC + ucsim)")


@pytest.fixture(scope='module')
def sim():
    mod = run_sim_module()
    mod.build()                                     # make -C vm/ports/z80 (minutes)
    return mod


@cross
@needs_sdcc
class TestZ80Simulator:
    def test_image_fits(self, sim):
        info = sim.sizes()
        assert info['code_end'] <= sim.CODE_LIMIT, f"code ends at 0x{info['code_end']:04X}"
        assert info['data_end'] <= sim.DATA_LIMIT, f"VM state ends at 0x{info['data_end']:04X}"
        assert info['run_steps_frame'] is not None and info['run_steps_frame'] < 128

    @pytest.mark.parametrize("name", ['ops', 'strings', 'arrays', 'maps', 'structs', 'closures',
                                      'results', 'errors', 'gc'])
    def test_program(self, sim, name):
        src = PROGRAMS[name]
        crc, cout, cerr = run_c(desktop_vm('z80'), src)       # == reference VM (tests above)
        r = sim.run_tbc(tbc_bytes(src))
        assert r['load_error'] == 0
        assert r['output'] == cout + cerr and r['exit_code'] == crc

    def test_examples(self, sim):
        for rel in EXAMPLES_16BIT:
            data = example_tbc(rel)
            rc, out = run_vm(data)
            r = sim.run_tbc(data)
            assert (r['exit_code'], r['output']) == (rc, out), rel


@cross
@pytest.mark.skipif(not shutil.which('cedev-config'), reason="needs the CE C toolchain (cedev-config)")
def test_ce_program_builds():
    r = subprocess.run(['make', '-C', CE_DIR], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    size = os.path.getsize(os.path.join(CE_DIR, 'bin', 'TILVM.8xp'))
    assert size < 65535, f"TILVM.8xp is {size} bytes"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
