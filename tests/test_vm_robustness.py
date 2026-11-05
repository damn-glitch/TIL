"""
Robustness of the two VMs (TIL 3.1 review fixes):

  1. loading a damaged .tbc is memory safe: vm/tilvm.c checks every section
     against the file size and every record (constant, function, type, method,
     prelude, line table) and fails with -5; the reference VM raises
     ValueError("damaged bytecode file ..."). Fuzzed under AddressSanitizer.
  2. int(x) of a float (also typed parameters and f"{x:d}") is exact and wraps
     like the reference VM's wrap_int64(int(x)); nan/inf is a catchable error.
  3. timeouts of 2^31 ms and more, and deadlines that wrap to 0.
  4. signed zeros: abs(-0.0), floor/ceil/round of zeros as the reference VM.
  5. float formatting: exact digits (ties to even on the binary value) in
     every C build, with or without stdio, at any precision and magnitude.
  6. TILVM_NO_MATH: exact floor/ceil/fmod, correctly rounded sqrt, pow.
  7. forged portable snapshots: frames, recovery points and handler records
     are checked against the bytecode (-4 in both VMs).
"""

import ctypes
import math
import os
import random
import re
import struct
import subprocess
import tempfile
import zlib

import pytest

from tilhelpers import CC, ROOT, VM_DIR, cvm_binary, needs_cc, run_both, run_cvm, run_vm, til, tbc_bytes

TILVM_C = os.path.join(VM_DIR, 'tilvm.c')
MAIN_C = os.path.join(VM_DIR, 'main.c')

_BUILDS = {}


def build(name: str, flags=(), sanitize=False, sources=(MAIN_C,)) -> str:
    """A C VM (or a driver: sources=[driver.c]) with the strict flags; sanitize=True
    adds AddressSanitizer + UBSan (float-cast-overflow too) when the compiler has
    them (else a plain build). Cached per session."""
    key = (name, tuple(flags), sanitize, tuple(sources))
    if key in _BUILDS:
        return _BUILDS[key]
    if CC is None:
        pytest.skip("no C compiler")
    exe = os.path.join(tempfile.mkdtemp(prefix='tilrob-'), name)
    base = [CC, '-std=c99', '-Wall', '-Wextra', '-pedantic', '-Werror', '-I', VM_DIR] + list(flags)
    cmd = None
    if sanitize:
        san = base + ['-g', '-O1', '-fno-omit-frame-pointer', '-fsanitize=address,undefined,float-cast-overflow',
                      '-fno-sanitize-recover=all', '-o', exe, TILVM_C, *sources, '-lm']
        if subprocess.run(san, capture_output=True).returncode == 0:
            cmd = san
    if cmd is None:
        cmd = base + ['-O2', '-o', exe, TILVM_C, *sources, '-lm']
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    _BUILDS[key] = exe
    return exe


def run_exe(exe: str, data: bytes, *args, timeout=120):
    fd, path = tempfile.mkstemp(suffix='.tbc')
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    try:
        r = subprocess.run([exe, *args, path], capture_output=True, timeout=timeout)
        return r.returncode, r.stdout.decode('utf-8', 'replace'), r.stderr.decode('utf-8', 'replace')
    finally:
        os.unlink(path)


def py_out(src) -> str:
    rc, out = run_vm(src)
    assert rc == 0, out
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 1. loading damaged bytecode
# ═══════════════════════════════════════════════════════════════════════════

RICH = '''
struct P
    x: int
    name: str

impl P
    show(self) -> str
        return f"{self.name}={self.x}"

enum Shape
    Circle(int)
    Dot

const BIG = 123456789012

area(s: Shape) -> int
    return match s
        Shape.Circle(r) => 3 * r * r
        Shape.Dot => 0

main()
    let p = P { x: 4, name: "p" }
    print(p.show(), area(Shape.Circle(2)), BIG, 2.5, "text")
    print([1, 2, 3].map(|v| v * 2).filter(|v| v > 2))
    let r = retry(2)
        panic("boom")
    else
        "E:" + error
    print(r)
'''

SMALL = '''
struct Q
    a: int

twice(n: int) -> int
    return n * 2

main()
    let q = Q { a: 3 }
    print(twice(q.a), [1, 2].map(|v| v + 1), "s")
'''


def sections(data: bytes) -> dict:
    """Offsets of the .tbc sections (docs/TIL_BYTECODE.md, independent of both VMs)."""
    p = 6
    out = {}
    n_globals, entry, n_consts = struct.unpack_from('<HHH', data, p)
    out['entry'] = p + 2
    p += 6
    consts = []
    for _ in range(n_consts):
        tag = data[p]
        consts.append((tag, p))
        p += 1 + (2 + struct.unpack_from('<H', data, p + 1)[0] if tag == 3 else 8)
    out['consts'] = consts
    out['n_funcs'] = p
    n_funcs = struct.unpack_from('<H', data, p)[0]
    out['funcs'] = [p + 2 + 11 * i for i in range(n_funcs)]
    p += 2 + 11 * n_funcs
    out['n_types'] = p
    types = []
    for _ in range(struct.unpack_from('<H', data, p)[0] if True else 0):
        pass
    n_types = struct.unpack_from('<H', data, p)[0]
    p += 2
    for _ in range(n_types):
        types.append(p)
        p += 3 + 2 * data[p + 2]
    out['types'] = types
    out['n_methods'] = p
    n_methods = struct.unpack_from('<H', data, p)[0]
    out['methods'] = [p + 2 + 6 * i for i in range(n_methods)]
    p += 2 + 6 * n_methods
    out['n_prelude'] = p
    out['prelude'] = [p + 1 + 3 * i for i in range(data[p])]
    p += 1 + 3 * data[p]
    out['code_len'] = p
    code_len = struct.unpack_from('<I', data, p)[0]
    out['code'] = p + 4
    p += 4 + code_len
    out['n_lines'] = p
    p += 2 + 6 * struct.unpack_from('<H', data, p)[0]
    out['end'] = p
    return out


def patch(data: bytes, off: int, fmt: str, value) -> bytes:
    b = bytearray(data)
    struct.pack_into('<' + fmt, b, off, value)
    return bytes(b)


def damaged_files(data: bytes) -> dict:
    """One file per record check of the loaders."""
    s = sections(data)
    n_funcs = len(s['funcs'])
    int_const = next(i for i, (tag, _) in enumerate(s['consts']) if tag != 3)
    code_len = struct.unpack_from('<I', data, s['code_len'])[0]
    files = {
        'entry out of range': patch(data, s['entry'], 'H', n_funcs),
        'function name not a string': patch(data, s['funcs'][0], 'H', int_const),
        'function name out of range': patch(data, s['funcs'][0], 'H', 0xFFFF),
        'function code outside the code': patch(data, s['funcs'][-1] + 7, 'I', code_len),
        'type name not a string': patch(data, s['types'][0], 'H', int_const),
        'type field out of range': patch(data, s['types'][3] + 3, 'H', len(s['consts'])),
        'type with 255 fields': patch(data, s['types'][-1] + 2, 'B', 255),
        'too many types': patch(data, s['n_types'], 'H', 4000),
        'method function out of range': patch(data, s['methods'][0] + 4, 'H', n_funcs),
        'too many methods': patch(data, s['n_methods'], 'H', 0xFFFF),
        'prelude function out of range': patch(data, s['prelude'][0] + 1, 'H', n_funcs),
        'code longer than the file': patch(data, s['code_len'], 'I', 0xFFFFFFF0),
        'line table longer than the file': patch(data, s['n_lines'], 'H', 0xFFFF),
        'string constant longer than the file': patch(data, s['consts'][-1][1] + 1, 'H', 0xFFFF)
        if s['consts'][-1][0] == 3 else data[:20],
        'too many functions': patch(data, s['n_funcs'], 'H', 0xFFFF),
        'truncated in the constants': data[:s['n_funcs'] - 3],
        'truncated in the functions': data[:s['funcs'][-1] + 5],
        'truncated in the code': data[:s['code'] + 7],
        'truncated in the line table': data[:s['end'] - 3],
        'header only': data[:16],
    }
    return files


@pytest.fixture(scope='module')
def rich_tbc():
    return tbc_bytes(RICH)


def test_sections_helper_matches_the_writer(rich_tbc):
    s = sections(rich_tbc)
    assert s['end'] == len(rich_tbc)
    assert len(s['types']) >= 5 and s['methods'] and s['prelude']


@pytest.mark.parametrize("what", sorted(damaged_files(tbc_bytes(RICH))))
def test_damaged_record_is_a_load_error(rich_tbc, what):
    data = damaged_files(rich_tbc)[what]
    with pytest.raises(ValueError) as info:
        til.TBCModule.from_bytes(data)
    assert "damaged bytecode file" in str(info.value) or "bad constant tag" in str(info.value)
    if CC:
        rc, out, err = run_exe(build('asan-vm', sanitize=True), data)
        assert rc == 2 and out == '', err
        assert re.search(r"cannot load \S+ \(error -(5|7|10): ", err), err


@needs_cc
def test_load_error_messages(rich_tbc, tmp_path):
    data = rich_tbc[:len(rich_tbc) // 2]
    rc, out, err = run_exe(cvm_binary(), data)
    assert rc == 2 and "(error -5: truncated or malformed file)" in err
    path = tmp_path / 'half.tbc'
    path.write_bytes(data)
    r = subprocess.run([til.sys.executable, os.path.join(ROOT, 'src', 'til.py'), 'exec', str(path)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "Error: damaged bytecode file (truncated or malformed)" in r.stderr
    r = subprocess.run([til.sys.executable, os.path.join(ROOT, 'src', 'til.py'), 'exec', '--lang', 'ru',
                        str(path)], capture_output=True, text=True)
    assert r.returncode == 1 and "damaged" not in r.stderr, r.stderr


FUZZ_DRIVER = r'''
/* load every truncation and many mutations of a .tbc: tilvm_load must return 0
 * or a negative code and never read outside the buffer (built with ASan) */
#include "tilvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static tilvm vm;
static unsigned long loads, loaded;
static void try_load(const unsigned char *src, unsigned long n) {
    unsigned char *buf = (unsigned char *)malloc(n ? n : 1);    /* exactly n bytes */
    int rc;
    memcpy(buf, src, n);
    rc = tilvm_load(&vm, buf, (uint32_t)n);
    loads++;
    if (rc == 0) loaded++;
    if (rc > 0) { printf("bad result %d\n", rc); exit(1); }
    free(buf);
}
int main(int argc, char **argv) {
    static unsigned char data[1 << 18], copy[1 << 18];
    unsigned long n, i, j, k;
    FILE *f = fopen(argv[1], "rb");
    static const unsigned char vals[] = {0, 1, 2, 3, 0x7F, 0x80, 0xFE, 0xFF};
    if (!f || argc < 3) return 2;
    n = (unsigned long)fread(data, 1, sizeof data, f);
    fclose(f);
    for (i = 0; i <= n; i++) try_load(data, i);                 /* every truncation */
    for (i = 4; i < n; i++)                                      /* every byte, 8 values */
        for (j = 0; j < sizeof vals; j++) {
            unsigned char save = data[i];
            data[i] = vals[j];
            try_load(data, n);
            data[i] = save;
        }
    for (i = 6; i + 1 < n; i++) {                               /* 16-bit counts and indexes */
        unsigned char a = data[i], b = data[i + 1];
        data[i] = 0xFF; data[i + 1] = 0xFF; try_load(data, n);
        data[i] = 0x00; data[i + 1] = 0x10; try_load(data, n);
        data[i] = a; data[i + 1] = b;
    }
    srand((unsigned)atoi(argv[2]));
    for (i = 0; i < 20000; i++) {                               /* random damage */
        memcpy(copy, data, n);
        for (k = 0; k < 1 + (unsigned long)(rand() % 4); k++) copy[6 + (unsigned long)rand() % (n - 6)] = (unsigned char)rand();
        try_load(copy, rand() % 8 ? n : (unsigned long)rand() % (n + 1));
    }
    printf("%lu loads, %lu loaded\n", loads, loaded);
    return 0;
}
'''

LEAN16 = ['-DTILVM_LEAN', '-DTILVM_ADDR16', '-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0', '-DTILVM_NO_STDIO',
          '-DTILVM_HEAP_SIZE=4096', '-DTILVM_STACK_SIZE=192', '-DTILVM_MAX_CONSTS=192', '-DTILVM_MAX_TYPES=24']


@needs_cc
@pytest.mark.parametrize("profile", ['default', 'lean16'])
def test_fuzz_loading_under_asan(tmp_path, profile):
    driver = tmp_path / 'fuzzload.c'
    driver.write_text(FUZZ_DRIVER)
    exe = build('fuzzload-' + profile, LEAN16 if profile == 'lean16' else [], sanitize=True,
                sources=(str(driver),))
    srcs = [RICH, SMALL] if profile == 'default' else [SMALL]
    for k, src in enumerate(srcs):
        for debug in (False, True):
            path = tmp_path / f'p{k}{int(debug)}.tbc'
            path.write_bytes(til.compile_to_bytecode(src, 'p.til', debug_info=debug).serialize())
            r = subprocess.run([exe, str(path), str(k + 1)], capture_output=True, text=True, timeout=600)
            assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
            m = re.search(r'(\d+) loads, (\d+) loaded', r.stdout)
            assert m and int(m.group(2)) > 0, r.stdout


@pytest.mark.parametrize("src", [RICH, SMALL])
def test_fuzz_loading_reference_vm(src):
    rnd = random.Random(7)
    for debug in (False, True):
        data = til.compile_to_bytecode(src, 'p.til', debug_info=debug).serialize()
        cases = [data[:i] for i in range(len(data) + 1)]
        for i in range(4, len(data)):
            for v in (0, 1, 3, 0x80, 0xFF):
                d = bytearray(data)
                d[i] = v
                cases.append(bytes(d))
        for _ in range(1500):
            d = bytearray(data)
            for _ in range(rnd.randint(1, 4)):
                d[rnd.randrange(6, len(d))] = rnd.randrange(256)
            cases.append(bytes(d))
        for d in cases:
            try:
                til.TBCModule.from_bytes(d)
            except ValueError:
                pass


@needs_cc
def test_valid_programs_still_load_in_every_profile():
    for src in (RICH, SMALL):
        out = run_both(src)
        for flags in ([], ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1'], ['-DTILVM_ADDR16', '-DTILVM_HEAP_SIZE=60000']):
            if src is RICH and '-DTILVM_INT_BITS=32' in flags:
                continue                                    # BIG does not fit 32 bits
            rc, cout, cerr = run_exe(build('v', flags), tbc_bytes(src))
            assert (rc, cout) == (0, out), cerr
    rc, cout, cerr = run_exe(build('lean16', LEAN16), tbc_bytes(SMALL))
    assert (rc, cout) == (0, py_out(SMALL)), cerr


# ═══════════════════════════════════════════════════════════════════════════
# 2. float -> int
# ═══════════════════════════════════════════════════════════════════════════

FLOAT_TO_INT = '''
g(x: int) -> int
    return x

h(y) -> int
    return y

main()
    let f = -9223372036854774784.0
    print(int(f), g(f), h(f))
    print(int(-9.21e18), int(-9.2e18 - 4096.0), int(9.2e18 + 4096.0), int(-9223372036854775808.0))
    print(int(9223372036854775808.0), int(1e19), int(-1e19), int(1e30), int(2.0 ** 63), int(2.0 ** 64 + 4096.0))
    print(int(0.999), int(-0.999), int(-0.0), int(4503599627370497.0), int(-123456.75))
    print(f"{1e30:d} {-9223372036854774784.0:x} {2.0 ** 70:b} {-1.5:d} {3e9:o} {1e19:X}")
    let inf = 1e308 * 10.0
    for x in [inf, -inf, inf - inf]
        let r = retry(1)
            f"{x:b}"
        else
            "E:" + error
        let s = retry(1)
            str(int(x))
        else
            "E:" + error
        print(r, s)
'''


def wrap64(n: int) -> int:
    n &= (1 << 64) - 1
    return n - (1 << 64) if n >= 1 << 63 else n


def signed_hex(n: int, fmt: str) -> str:
    return ('-' if n < 0 else '') + fmt % abs(n)


def test_float_to_int_wraps_exactly():
    out = run_both(FLOAT_TO_INT)
    expect = ["-9223372036854774784"] * 3
    expect += [str(v) for v in (int(-9.21e18), int(-9.2e18 - 4096.0), int(9.2e18 + 4096.0), -2 ** 63)]
    expect += [str(wrap64(int(v))) for v in (2.0 ** 63, 1e19, -1e19, 1e30, 2.0 ** 63, 2.0 ** 64 + 4096.0)]
    expect += ["0", "0", "0", "4503599627370497", "-123456"]
    expect.append(" ".join([str(wrap64(int(1e30))), signed_hex(-9223372036854774784, '%x'), "0", "-1",
                            "%o" % 3000000000, signed_hex(wrap64(int(1e19)), '%X')]))
    expect += ["E:cannot convert nan/inf to int"] * 6
    assert out.splitlines() == expect


@needs_cc
@pytest.mark.parametrize("flags", [['-DTILVM_NO_MATH'], ['-DTILVM_NO_STDIO', '-DTILVM_NO_MATH']])
def test_float_to_int_without_libm(flags):
    rc, out, err = run_exe(build('f2i', flags, sanitize=True), tbc_bytes(FLOAT_TO_INT))
    assert (rc, out) == (0, py_out(FLOAT_TO_INT)), err


@needs_cc
def test_float_to_int_32_bit_build_wraps_to_32_bits():
    src = '''
main()
    print(int(1e10), int(-1e10), int(3e9), int(4294967296.0 * 3.0 + 1024.0), int(2147483520.0))
    print(f"{3e9:d}")
'''
    rc, out, err = run_exe(build('f2i32', ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1'], sanitize=True),
                           tbc_bytes(src))
    w32 = lambda n: (n & 0xFFFFFFFF) - (1 << 32 if n & 0x80000000 else 0)   # noqa: E731
    assert rc == 0, err
    assert out.split() == [str(w32(v)) for v in (10 ** 10, -10 ** 10, 3 * 10 ** 9, 3 * 2 ** 32 + 1024,
                                                 2 ** 31 - 128, 3 * 10 ** 9)]


@needs_cc
def test_float_casts_are_not_undefined_behaviour():
    """UBSan's float-cast-overflow: map keys, timeouts, f"{x:d}" and int() of
    huge floats, inf and nan (the sanitizer build aborts on the first report)."""
    src = '''
main()
    let inf = 1e308 * 10.0
    var m = {1e300: "a", 2.0: "b", inf: "c"}
    print(m.get_or(1e300, "?"), m.get_or(2, "?"), m.get_or(inf, "?"))
    let r = timeout(1e300)
        5
    else
        "E:" + error
    let q = timeout(inf - inf)
        6
    else
        "E:" + error
    print(r, q, int(1e300), f"{-1e300:x}")
'''
    exe = build('ubsan', sanitize=True)
    rc, out, err = run_exe(exe, tbc_bytes(src))
    assert rc == 0 and "runtime error" not in err, err
    assert out.splitlines() == ["a", "b", "c", "5", "6", str(wrap64(int(1e300))),
                                signed_hex(wrap64(int(-1e300)), '%x')]
    assert out == py_out(src)


# ═══════════════════════════════════════════════════════════════════════════
# 3. timeouts
# ═══════════════════════════════════════════════════════════════════════════

LONG_TIMEOUTS = '''
work() -> int
    var s = 0
    for i in 0..2000
        s += i
    return s

main()
    let r = timeout(3000000000)
        work()
    else
        "E:" + error
    let q = timeout(10000000000000)
        work()
    else
        "E:" + error
    let f = timeout(2147483648.5)
        work()
    else
        "E:" + error
    print(r, q, f)
    let z = retry(1)
        timeout("abc")
            5
        else
            "inner:" + error
    else
        "E:" + error
    print(z)
'''


def test_long_timeouts_do_not_fire():
    out = run_both(LONG_TIMEOUTS)
    assert out.splitlines() == ["1999000"] * 3 + ["E:timeout expects a number, got str"]


CLOCK_DRIVER = r'''
/* a fake clock: start, step per call (argv); runs the program with a step budget */
#include "tilvm.h"
#include <stdio.h>
#include <stdlib.h>
static tilvm vm;
static uint32_t now, step;
static unsigned long calls;
static uint32_t fake_clock(void *ctx) { (void)ctx; calls++; now += step; return now; }
int main(int argc, char **argv) {
    static unsigned char code[1 << 16];
    FILE *f = fopen(argv[1], "rb");
    unsigned long n;
    uint32_t start;
    int rc;
    if (!f || argc < 5) return 2;
    n = (unsigned long)fread(code, 1, sizeof code, f);
    fclose(f);
    if (tilvm_load(&vm, code, (uint32_t)n) != 0) return 2;
    now = start = (uint32_t)strtoul(argv[2], 0, 0);
    step = (uint32_t)strtoul(argv[3], 0, 0);
    vm.clock_ms = fake_clock;
    rc = tilvm_run_steps(&vm, (uint32_t)strtoul(argv[4], 0, 0));
    printf("status %d elapsed %lu\n", rc, (unsigned long)(uint32_t)(now - start));
    return 0;
}
'''

SPIN = '''
main()
    let r = timeout(%s)
        var n = 0
        while true
            n += 1
        n
    else
        "fired: " + error
    print(r)
'''


def clock_run(tmp_path, ms: str, start: int, step: int, budget: int, flags=()):
    driver = tmp_path / 'clock.c'
    driver.write_text(CLOCK_DRIVER)
    exe = build('clock', flags, sanitize=True, sources=(str(driver),))
    path = tmp_path / 'spin.tbc'
    path.write_bytes(tbc_bytes(SPIN % ms))
    r = subprocess.run([exe, str(path), str(start), str(step), str(budget)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    m = re.search(r'status (\d+) elapsed (\d+)', r.stdout)
    return r.stdout, int(m.group(1)), int(m.group(2))


@needs_cc
def test_deadline_that_wraps_to_zero_still_fires(tmp_path):
    # now + 4 + 1 == 0 (mod 2^32): the deadline must not read as "no deadline"
    out, status, elapsed = clock_run(tmp_path, '4', 0xFFFFFFFF - 5, 1, 1000000)
    assert status == 0 and out.startswith("fired: timeout"), out


@needs_cc
@pytest.mark.parametrize("flags", [[], ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1'], LEAN16[:4] + LEAN16[5:]])
def test_timeout_fires_after_its_time_only(tmp_path, flags):
    big = flags == [] or '-DTILVM_INT_BITS=32' in flags
    ms = 3000000000 if flags == [] else 2147483647 if big else 30000      # the largest int32: 2^31 - 1
    step = 1 << 26 if big else 64
    out, status, elapsed = clock_run(tmp_path, str(ms) if big else str(ms), 0xF0000000, step, 50000000, flags)
    assert status == 0 and out.startswith("fired: timeout"), out
    assert ms < elapsed <= ms + 3 * step, (elapsed, ms)       # a few clock reads after the deadline


@needs_cc
def test_huge_timeout_never_fires(tmp_path):
    for ms in ('10000000000000', '1e300', '(1e308 * 10.0)', '((1e308 * 10.0) - (1e308 * 10.0))'):
        out, status, elapsed = clock_run(tmp_path, ms, 123, 1 << 30, 200000)
        assert status == 2 and out.startswith("status 2"), (ms, out)      # TILVM_YIELD: still running


# ═══════════════════════════════════════════════════════════════════════════
# 4. signed zeros
# ═══════════════════════════════════════════════════════════════════════════

ZEROS = '''
main()
    let z = 0.0 * -1.0
    print(abs(-0.0), abs(z), -0.0, z, str(z), f"{z}", [z], z / 3.0)
    print(sqrt(abs((sqrt(abs((-0.0))) / 3.14159))), sqrt(-0.0))
    print(ceil(-0.5), floor(-0.0), ceil(-0.0), floor(0.3), ceil(-0.3))
    print(round(-0.0), round(-0.3), round(0.3), round(-2.5), round(2.5), round(-0.5))
    print(f"{z:.2f} {z:.1e} {z:.3} {-0.0:g}", abs(-1.5), abs(-7))
'''


def test_signed_zeros():
    out = run_both(ZEROS)
    assert out.splitlines() == ["0", "0", "-0", "-0", "-0", "-0", "[-0]", "-0",
                                "0", "-0",
                                "0", "0", "0", "0", "0",
                                "-0", "-0", "0", "-3", "3", "-1",
                                "-0.00 -0.0e+00 -0 -0", "1.5", "7"]


@needs_cc
@pytest.mark.parametrize("flags", [['-DTILVM_NO_MATH'], ['-DTILVM_NO_STDIO', '-DTILVM_NO_MATH', '-DTILVM_FLOAT=1',
                                                         '-DTILVM_INT_BITS=32']])
def test_signed_zeros_without_libm(flags):
    rc, out, err = run_exe(build('zeros', flags), tbc_bytes(ZEROS))
    assert (rc, out) == (0, py_out(ZEROS)), err


# ═══════════════════════════════════════════════════════════════════════════
# 5. float formatting: exact digits in every build
# ═══════════════════════════════════════════════════════════════════════════

# builds whose float output must equal the reference VM byte for byte
FLOAT_BUILDS = {
    'stdio': [],
    'nostdio': ['-DTILVM_NO_STDIO'],
    'nomath': ['-DTILVM_NO_MATH'],
    'nostdio-nomath': ['-DTILVM_NO_STDIO', '-DTILVM_NO_MATH'],
}

REVIEW_FLOATS = '''
main()
    print(-0.0 * 1.0, 999999.5, 123456.5, 65081.35, 2877580 / 8.0, 7158252 / 8.0, 65023 / 8.0)
    print(f"{1e45:.1f}")
    print(f"{1e21:.1%}")
    print(f"{0.005:.2f} {-0.015:.2f} {1e100:.3f} {0.125:.2f} {0.375:.2f} {2.5:.0f} {3.5:.0f}")
    print(f"{72465 / 100.0:.3e} {1.0 / 3.0:.17f} {5e-324:.3e} {1.7976931348623157e308:.30e}")
    print(f"{1e100:.2f}")
    print(len(f"{1e300:.1f}"), len(f"{1.7976931348623157e308:.30f}"))
    let z = 1e308 * 10.0
    let n = z - z
    print(f"{n:.3e}", f"{n:.1f}", f"{-n:.1f}", str(n), str(-n), f"{n}", f"{n:.5}", f"{z:.2f}", f"{-z:e}")
    print(f"{n:08.2f}|{-z:>8.1f}|{0.1:.30}|{1e-310:.20}|{2.0 ** -1074:.25e}")
    print(1e16, 1e-5, 0.0001, 123456789.0, 1.5e300, 5e-324, 2.2250738585072014e-308)
'''


@pytest.fixture(scope='module')
def review_floats_ref():
    out = py_out(REVIEW_FLOATS)
    lines = out.splitlines()
    assert lines[:7] == ['-0', '1e+06', '123456', '65081.3', '359698', '894782', '8127.88']
    assert lines[7] == '%.1f' % 1e45 and lines[8] == '%.1f%%' % (1e21 * 100)
    assert lines[9].split()[:2] == ['0.01', '-0.01'] and lines[11] == '%.2f' % 1e100
    assert lines[12:14] == ['303', str(len('%.30f' % 1.7976931348623157e308))]
    assert lines[14:21] == ['nan'] * 7
    return out


@needs_cc
@pytest.mark.parametrize("name", sorted(FLOAT_BUILDS))
def test_review_float_cases(review_floats_ref, name):
    rc, out, err = run_exe(build('fmt-' + name, FLOAT_BUILDS[name]), tbc_bytes(REVIEW_FLOATS))
    assert (rc, out) == (0, review_floats_ref), err


def random_doubles(rnd, n):
    special = [0.5, 1.5, 2.5, 0.125, 999999.5, 123456.5, 65081.35, 0.005, 1e21, 1e45, 1e100, 5e-324,
               2.2250738585072014e-308, 1.7976931348623157e308, 0.1, 0.3, 9.5, 99.95, 1e15, 1e16, 1e17]
    vals = []
    while len(vals) < n:
        k = rnd.randrange(8)
        if k == 0:
            v = struct.unpack('<d', struct.pack('<Q', rnd.getrandbits(64)))[0]
        elif k == 1:
            v = rnd.choice(special)
        elif k == 2:
            v = rnd.randrange(-10 ** 6, 10 ** 6) + rnd.choice([0, 0.5, 0.25, 0.125, 0.05])
        elif k == 3:
            v = float(rnd.randrange(1, 10 ** 17)) * 10.0 ** rnd.randrange(-30, 30)
        else:
            v = rnd.uniform(1, 10) * 10.0 ** rnd.randrange(-320, 308) * rnd.choice([1, -1])
        if v == v and v not in (float('inf'), float('-inf')):
            vals.append(v)
    return vals


SPECS = ['.0f', '.1f', '.2f', '.6f', '.17f', '.30f', '.0e', '.3e', '.16e', '.30e', '.1', '.6', '.17', '.30',
         '.0%', '.3%', '12.3f', '012.3f', '<15.2e', '^20.4', 'd', 'x']


def format_program(vals) -> str:
    src = ['main()']
    for i, v in enumerate(vals):
        src.append(f'    let v{i} = {v!r}')
        src += [f'    print(f"{{v{i}:{sp}}}")' for sp in SPECS]
        src += [f'    print(v{i})', f'    print(str(-v{i}))']
    return '\n'.join(src) + '\n'


@needs_cc
@pytest.mark.parametrize("seed", [1, 2])
def test_random_float_formatting_matches_the_reference_vm(seed):
    """Random doubles of every magnitude in %g, .Nf, .Ne, .Ng, %, d, x."""
    src = format_program(random_doubles(random.Random(seed), 70))
    ref = py_out(src)
    for name in ('stdio', 'nostdio-nomath'):
        rc, out, err = run_exe(build('fmt-' + name, FLOAT_BUILDS[name], sanitize=True), tbc_bytes(src))
        assert rc == 0, err
        for a, b in zip(ref.splitlines(), out.splitlines()):
            assert a == b, name
        assert out == ref


def f32(x: float) -> float:
    return ctypes.c_float(x).value


@needs_cc
@pytest.mark.parametrize("flags", [['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1'],
                                   ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1', '-DTILVM_NO_STDIO', '-DTILVM_NO_MATH']])
def test_float32_builds_print_the_float32_value(flags):
    """The rule of TILVM_FLOAT=1 builds: a float prints exactly as the reference VM
    prints the same float32 value (converted to a double: exact)."""
    rnd = random.Random(5)
    vals = []
    while len(vals) < 80:
        k = rnd.randrange(4)
        if k == 0:
            v = struct.unpack('<f', struct.pack('<I', rnd.getrandbits(32)))[0]
        elif k == 1:
            v = rnd.choice([0.5, 1.5, 2.5, 999999.5, 123456.5, 65081.35, 0.005, 1e21, 1e38, 1e-45, 3.4e38, 0.1])
        elif k == 2:
            v = rnd.randrange(-10 ** 6, 10 ** 6) + rnd.choice([0, 0.5, 0.25, 0.05])
        else:
            v = rnd.uniform(1, 10) * 10.0 ** rnd.randrange(-44, 38)
        v = f32(v)
        if v == v and abs(v) != float('inf'):
            vals.append(v)
    specs = ['.0f', '.1f', '.2f', '.6f', '.12f', '.0e', '.3e', '.8e', '.1', '.6', '.9', '.0%', '.3%', '012.3f']
    src, expect = ['main()'], []
    for i, v in enumerate(vals):
        src.append(f'    let v{i} = float("{v!r}")')                # parsed: correctly rounded to float32
        src.append(f'    let c{i} = {v!r}')                          # a constant: rounded when it loads
        for sp in specs:
            src.append(f'    print(f"{{v{i}:{sp}}}")')
            x = f32(v * 100.0) if sp.endswith('%') else v
            expect.append(til.til_format_spec(x, sp.replace('%', 'f').encode(), til.fmt_g).decode()
                          + ('%' if sp.endswith('%') else ''))
        src += [f'    print(v{i})', f'    print(str(-v{i}))', f'    print(c{i})']
        expect += [til.fmt_g(v).decode(), til.fmt_g(-v).decode(), til.fmt_g(v).decode()]
    rc, out, err = run_exe(build('f32', flags), tbc_bytes('\n'.join(src) + '\n'))
    assert rc == 0, err
    assert out.splitlines() == expect


@needs_cc
def test_float32_constants_are_correctly_rounded():
    rnd = random.Random(9)
    vals = [rnd.uniform(1, 10) * 10.0 ** rnd.randrange(-50, 39) for _ in range(300)]
    vals += [struct.unpack('<d', struct.pack('<Q', rnd.getrandbits(64)))[0] for _ in range(300)]
    vals = [v for v in vals if v == v and abs(v) != float('inf')]
    src = 'main()\n' + ''.join(f'    print(f"{{{v!r}:.9e}}")\n' for v in vals)
    rc, out, err = run_exe(build('f32', ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1']), tbc_bytes(src))
    assert rc == 0, err
    assert out.splitlines() == ['%.9e' % f32(v) for v in vals]


# ═══════════════════════════════════════════════════════════════════════════
# 6. TILVM_NO_MATH
# ═══════════════════════════════════════════════════════════════════════════

def nomath_values(rnd, n):
    vals = [1e30, 2.0, 0.5, 3.0, 1e-300, 5e-324, 1.7976931348623157e308, 2.2250738585072014e-308, 1e19,
            -1e19, 4503599627370495.5, -4503599627370495.5, 1e20, 0.1]
    while len(vals) < n:
        k = rnd.randrange(3)
        if k == 0:
            v = struct.unpack('<d', struct.pack('<Q', rnd.getrandbits(63)))[0]
        elif k == 1:
            v = rnd.uniform(0, 10) * 10.0 ** rnd.randrange(-310, 308)
        else:
            v = rnd.uniform(-1e6, 1e6)
        if v == v and abs(v) != float('inf'):
            vals.append(v)
    return vals


@needs_cc
@pytest.mark.parametrize("flags", [['-DTILVM_NO_MATH'], ['-DTILVM_NO_STDIO', '-DTILVM_NO_MATH']])
def test_nomath_floor_ceil_fmod_sqrt_are_exact(flags):
    rnd = random.Random(11)
    vals = nomath_values(rnd, 150)
    divs = [3.0, 0.1, 1e-300, 7.5e10, 2.5, 1e300, 5e-324, 1.0]
    src = ['main()']
    for i, v in enumerate(vals):
        d = divs[i % len(divs)]
        src.append(f'    let v{i} = {v!r}')
        src.append(f'    print(f"{{sqrt(abs(v{i})):.17}} {{floor(v{i}):.17}} {{ceil(v{i}):.17}} '
                   f'{{v{i} % {d!r}:.17}} {{-v{i} % {d!r}:.17}} {{round(v{i}):.17}}")')
    src = '\n'.join(src) + '\n'
    rc, out, err = run_exe(build('nomath', flags, sanitize=True), tbc_bytes(src))
    assert rc == 0, err
    assert out == py_out(src)
    assert run_vm(f'main()\n    print(f"{{sqrt(1e30):.17}} {{floor(1e19):.17}} {{1e20 % 3.0}}")\n')[1] == \
        "1000000000000000 1e+19 1\n"


@needs_cc
def test_nomath_float32_sqrt_is_correctly_rounded():
    rnd = random.Random(12)
    vals = [f32(abs(struct.unpack('<f', struct.pack('<I', rnd.getrandbits(31)))[0])) for _ in range(300)]
    vals = [v for v in vals if v == v and v != float('inf')]
    src = 'main()\n    var v = 0.0\n' + ''.join(f'    v = float("{v!r}")\n    print(f"{{sqrt(v):.9e}}")\n'
                                                for v in vals)
    flags = ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1', '-DTILVM_NO_MATH', '-DTILVM_NO_STDIO']
    rc, out, err = run_exe(build('nomath32', flags), tbc_bytes(src))
    assert rc == 0, err
    assert out.splitlines() == ['%.9e' % f32(math.sqrt(v)) for v in vals]


POW_CASES = '''
main()
    let z = 0.0
    let inf = 1e308 * 10.0
    print(2.0 ** 60.0, 0.5 ** 1074.0, (-3.0) ** 3.0, 2.0 ** -3.0, 10.0 ** 22.0, 2.0 ** 1e10, (-2.0) ** 1e10)
    print(z ** -1.0, (-z) ** -1.0, z ** -2.0, z ** 0.5, z ** -0.5, inf ** 0.5, inf ** -0.5, 1.0 ** 0.5)
    print((-8.0) ** (1.0 / 3.0), 0.5 ** inf, 2.0 ** inf, 2.0 ** -inf, (-1.0) ** inf, (inf - inf) ** 0.0, 1.0 ** (inf - inf))
    print(pow(2, 10), pow(1.5, 2.0), 1e308 ** 2.0, (-1e308) ** 3.0, 10.0 ** 400.0)
    let r = retry(1)
        str(2.0 ** 0.5)
    else
        "E:" + error
    print(r)
'''


@needs_cc
def test_nomath_pow():
    ref = py_out(POW_CASES).splitlines()
    assert ref[-1] == '1.41421'
    rc, out, err = run_exe(build('nomath', ['-DTILVM_NO_MATH'], sanitize=True), tbc_bytes(POW_CASES))
    assert rc == 0, err
    assert out.splitlines() == ref[:-1] + ["E:pow with a fractional exponent is not available in this VM build"]
    rc, out, err = run_exe(cvm_binary(), tbc_bytes(POW_CASES))
    assert (rc, out.splitlines()) == (0, ref), err


# ═══════════════════════════════════════════════════════════════════════════
# 7. portable snapshots: frames, recovery points and handlers vs the bytecode
# ═══════════════════════════════════════════════════════════════════════════

def tp_ops_from_python() -> list:
    """TP_OPS of vm/tilvm.c, computed from the compiler's stack-effect tables."""
    rows = []
    for op in range(0xC0):
        if op not in til._STACK_EFFECT and op not in til._OPERAND_EFFECT:
            rows.append(7)
            continue
        n = sum(til._OPERAND_SIZE[k] for k in til.OP_OPERANDS.get(op, ''))
        eff = 0 if op in til._OPERAND_EFFECT else til._STACK_EFFECT[op]
        rows.append(n | (eff + 3) << 3 | (op in til._JUMP_LANDING) << 6 |
                    (op in til._ENDS_FLOW or op == til.Op.END_HANDLE) << 7)
    return rows


def test_snapshot_depth_table_matches_the_compiler():
    src = open(TILVM_C, encoding='utf-8').read()
    body = re.search(r'static const uint8_t TP_OPS\[0xC0\] = \{(.*?)\};', src, re.S).group(1)
    assert [int(x, 16) for x in re.findall(r'0x([0-9A-F]{2})', body)] == tp_ops_from_python()
    # the operand-dependent effects tp_depth computes are exactly _OPERAND_EFFECT's
    names = {til.Op.CALL: 'OP_CALL', til.Op.CALL_VALUE: 'OP_CALL_VALUE', til.Op.INVOKE: 'OP_INVOKE',
             til.Op.NATIVE: 'OP_NATIVE', til.Op.NEW_ARRAY: 'OP_NEW_ARRAY', til.Op.NEW_MAP: 'OP_NEW_MAP',
             til.Op.NEW_STRUCT: 'OP_NEW_STRUCT', til.Op.MAKE_CLOSURE: 'OP_MAKE_CLOSURE', til.Op.CONCAT: 'OP_CONCAT',
             til.Op.PERFORM: 'OP_PERFORM', til.Op.NATIVE_POP: 'OP_NATIVE_POP'}
    assert set(names) == set(til._OPERAND_EFFECT)
    depth_fn = src[src.index('static long tp_depth('):src.index('static int tp_at(')]
    for op, name in names.items():
        assert name in depth_fn, name


CALLS = '''
struct Box
    f: fn(int) -> int
    k: int

impl Box
    apply(self, x: int) -> int
        var t = 0
        for i in 0..x
            t += i
        return self.f(t) + self.k

twice(n: int) -> int
    var s = 0
    for i in 0..n
        s += 2
    return s

main()
    let add = |v| twice(v) + 1
    let b = Box { f: add, k: 7 }
    var total = 0
    for i in 0..4
        total += b.apply(i) + add(i) + b.f(i)
    let r = retry(2)
        let t = timeout(100000)
            twice(total)
        else
            -1
        t + [1, 2, 3].map(|q| twice(q)).sum()
    else
        0
    let h = handle
        perform Ask(3) + twice(2)
    with
        Ask(n) =>
            var w = 0
            for j in 0..n
                w += twice(j)
            resume(w)
    print(total, r, h)
'''


def snapshot_layout(data: bytes):
    """Offsets of the frame, recovery point and handler records of a .tilp."""
    p = 30
    frames = []
    for _ in range(struct.unpack_from('<H', data, 28)[0]):
        frames.append(p)
        p += 13
    tries = []
    for _ in range(struct.unpack_from('<H', data, p)[0]):
        tries.append(p + 2 + 14 * len(tries))
    p += 2 + 14 * len(tries)
    handlers = [p + 2 + 13 * i for i in range(struct.unpack_from('<H', data, p)[0])]
    return frames, tries, handlers


def resign(b: bytearray) -> bytes:
    struct.pack_into('<I', b, len(b) - 4, zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF)
    return bytes(b)


def forgeries(data: bytes):
    frames, tries, handlers = snapshot_layout(data)
    out = []
    for i, off in enumerate(frames):
        if i:
            for delta in (1, -1, 2):
                b = bytearray(data)
                struct.pack_into('<H', b, off + 6, (struct.unpack_from('<H', b, off + 6)[0] + delta) & 0xFFFF)
                out.append(('base', resign(b)))
            b = bytearray(data)
            b[off + 8] ^= 1
            out.append(('extra', resign(b)))
    for off in tries:
        b = bytearray(data)
        struct.pack_into('<H', b, off + 10, struct.unpack_from('<H', b, off + 10)[0] + 1)
        out.append(('try sp', resign(b)))
    for off in handlers:
        if data[off] == 0:
            b = bytearray(data)
            struct.pack_into('<H', b, off + 3, struct.unpack_from('<H', b, off + 3)[0] + 2)
            out.append(('handler sp', resign(b)))
    return out


def test_forged_snapshots_are_refused_by_the_reference_vm():
    tbc = tbc_bytes(CALLS)
    vm = til.TILVM(til.TBCModule.from_bytes(tbc), write=lambda b: None)
    n = 0
    kinds = set()
    while vm.run_steps(3) == vm.YIELD:
        data = vm.save_portable()
        fresh = til.TILVM(til.TBCModule.from_bytes(tbc), write=lambda b: None)
        fresh.load_portable(data)                               # the genuine one loads
        for what, forged in forgeries(data):
            bad = til.TILVM(til.TBCModule.from_bytes(tbc), write=lambda b: None)
            try:
                bad.load_portable(forged)
            except til.SnapshotError as e:
                assert e.code in (-4, -5), (what, e)           # -5: a stack limit of the forged base
                n += 1
                kinds.add(what)
                continue
            pytest.fail(f"forged {what} accepted at step {vm.steps}")
    assert n > 100 and kinds == {'base', 'extra', 'try sp', 'handler sp'}


@needs_cc
def test_forged_snapshots_are_refused_by_the_c_vm(tmp_path):
    tbc = tbc_bytes(CALLS)
    tbc_path = tmp_path / 'calls.tbc'
    tbc_path.write_bytes(tbc)
    exe = build('asan-vm', sanitize=True)
    out = []
    vm = til.TILVM(til.TBCModule.from_bytes(tbc), write=out.append)
    ref = run_vm(tbc)[1]
    checked = 0
    while vm.run_steps(7) == vm.YIELD:
        data = vm.save_portable()
        snap = tmp_path / 's.tilp'
        snap.write_bytes(data)
        r = subprocess.run([exe, '--resume-portable', str(snap), str(tbc_path)], capture_output=True, timeout=60)
        assert r.returncode == 0, r.stderr
        assert b"".join(out).decode() + r.stdout.decode() == ref        # continues where it paused
        for what, forged in forgeries(data)[:6]:
            snap.write_bytes(forged)
            r = subprocess.run([exe, '--resume-portable', str(snap), str(tbc_path)], capture_output=True,
                               timeout=60)
            assert r.returncode == 2, (what, r.stdout, r.stderr)
            assert b"the snapshot is damaged or incomplete" in r.stderr or b"does not fit" in r.stderr, r.stderr
            checked += 1
    assert checked > 20


@needs_cc
def test_reviewer_forged_frame_base(tmp_path):
    """The repro: frame 1 (main) moved up by 2 slots, checksum fixed - main's caller
    (__start__) calls it with nothing on its operand stack."""
    tbc = tbc_bytes(CALLS)
    vm = til.TILVM(til.TBCModule.from_bytes(tbc), write=lambda b: None)
    assert vm.run_steps(3) == vm.YIELD
    data = bytearray(vm.save_portable())
    frames, _, _ = snapshot_layout(bytes(data))
    struct.pack_into('<H', data, frames[1] + 6, struct.unpack_from('<H', data, frames[1] + 6)[0] + 2)
    forged = resign(data)
    with pytest.raises(til.SnapshotError) as info:
        til.TILVM(til.TBCModule.from_bytes(tbc), write=lambda b: None).load_portable(forged)
    assert info.value.code == -4
    (tmp_path / 'f.tilp').write_bytes(forged)
    (tmp_path / 'f.tbc').write_bytes(tbc)
    r = subprocess.run([cvm_binary(), '--resume-portable', str(tmp_path / 'f.tilp'), str(tmp_path / 'f.tbc')],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "the snapshot is damaged or incomplete" in r.stderr
