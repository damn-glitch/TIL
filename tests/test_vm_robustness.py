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
    assert ms < elapsed <= ms + 2 * step, (elapsed, ms)


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
