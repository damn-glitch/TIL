"""
Portable snapshots (.tilp) - TIL 3.1 feature 2 (docs/TIL_SNAPSHOTS.md).

Pause a program on one VM, continue it on another: the reference VM
(TILVM.run_steps / save_portable / load_portable) and the C VM
(tilvm_save_portable / tilvm_load_portable, `tilvm --save-portable-after N
FILE`, `--resume-portable FILE`) in several builds. Part 1 + part 2 of the
output must equal an uninterrupted run, and both VMs must write the same bytes
for the same pause point.
"""

import os
import shutil
import struct
import subprocess
import sys
import zlib

import pytest

from tilhelpers import CC, ROOT, VM_DIR, cli, cvm_binary, needs_cc, til, tbc_bytes

# Non-lean C VM builds (lean builds leave portable snapshots out)
TINY2K = ['-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0', '-DTILVM_HEAP_SIZE=2048', '-DTILVM_STACK_SIZE=128',
          '-DTILVM_MAX_FRAMES=24', '-DTILVM_MAX_CONSTS=128', '-DTILVM_MAX_TYPES=16', '-DTILVM_MAX_TRY=4']
INT16 = ['-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0']

# ═══════════════════════════════════════════════════════════════════════════
# programs
# ═══════════════════════════════════════════════════════════════════════════

MIXED = '''
struct P
    x: int
    y: int

enum Shape
    Circle(int)
    Rect(int, int)
    Dot

fib(n: int) -> int
    if n < 2
        return n
    return fib(n - 1) + fib(n - 2)

area(s: Shape) -> int
    return match s
        Shape.Circle(r) => 3 * r * r
        Shape.Rect(w, h) => w * h
        Shape.Dot => 0

base() -> int
    var t = 0
    for i in 0..3
        t += i
    return t

scaled(x: int, k: int = base()) -> int
    return x * k

main()
    var total = 0
    var m = {"a": 1, "b": 2}
    let arr = [P { x: 1, y: 2 }, P { x: 3, y: 4 }]
    var shared = [1, 2, 3]
    let both = [shared, shared]
    let shapes = [Shape.Circle(2), Shape.Rect(2, 5), Shape.Dot]
    var opt = Some(7)
    var letters = 0
    for ch in "hello"
        letters += ch
    for i in 0..18
        total += fib(i % 9) + scaled(i) + area(shapes[i % 3])
        m["k" + str(i % 5)] = i
        both[0].push(i)
        let name = ("x" + "hello").slice(1, 6)
        let fv = fib
        print(f"{i} {total} {len(both[1])} {name} {opt.unwrap()} {letters} {fv(5)}")
    print(m)
    print(arr[1].y)
'''

CLOSURES = '''
make_adder(k: int)
    return |x: int| x + k

apply_all(fs, x: int) -> int
    var total = 0
    for f in fs
        total += f(x)
    return total

twice(f, x: int) -> int
    return f(f(x))

main()
    var fs = []
    for i in 0..6
        fs.push(make_adder(i))
    var count = 0
    let peek = |d: int| count * 1000 + d
    let peek2 = peek
    var acc = 0
    for j in 0..14
        acc += apply_all(fs, j)
        count += 1
        print(f"{j} {acc} {peek(j)} {peek2(1)} {twice(fs[3], j)} {twice(|y: int| y * 2, j)}")
'''

EFFECTS = '''
countdown(n: int)
    var i = n
    while i > 0
        perform Yield(i)
        i -= 1

walk(depth: int) -> int
    if depth == 0
        return perform Ask("leaf")
    return walk(depth - 1) + 1

main()
    var squares = []
    handle
        countdown(6)
    with
        Yield(v) =>
            var k = 0
            while k < 3
                k += 1
            squares.push(v * v + k)
            resume()
    print(squares)
    var counter = 0
    let total = handle
        for i in 1..=6
            perform Add(i)
            print(f"added {i} -> {counter}")
        perform Get()
    with
        Add(x) =>
            counter += x
            resume()
        Get => resume(counter * 10)
    print(f"total = {total}")
    let r = handle
        handle
            var s = 0
            for q in 0..4
                s += walk(q)
            perform Log(f"inner work {s}")
            "done"
        with
            Log(msg) =>
                perform Log("[inner] " + msg)
                resume()
            Ask(w) => resume(100)
    with
        Log(msg) =>
            print("[outer] " + msg)
            resume()
    print(r)
    let early = handle
        for z in 0..50
            if z == 7
                perform Stop(z)
        -1
    with
        Stop(x) =>
            var t = 0
            for u in 0..x
                t += u
            t
    print(f"early = {early}")
'''

RETRY = '''
var attempts = 0

flaky(n: int) -> int
    attempts += 1
    var w = 0
    while w < 6
        w += 1
    assert attempts % 3 == 0, "busy"
    return n * 2

main()
    var results = []
    for i in 0..4
        let v = retry(5)
            flaky(i)
        else
            -1
        results.push(v)
        print(f"{i}: {v} after {attempts}")
    let msg = retry(2)
        intent "load"
            let xs = [1, 2]
            var q = 0
            while q < 5
                q += 1
            xs[5]
    else
        error
    print(msg)
    print(results)
'''

MAPS = '''
struct Node
    name: str
    next: any

main()
    var votes = {}
    for word in "tea coffee tea juice tea coffee water milk tea juice".split(" ")
        votes[word] = votes.get_or(word, 0) + 1
    votes.delete("water")
    votes["zz"] = 1
    votes[2.0] = "two"
    for k in votes
        print(f"{k} {votes[k]}")
    var a = [1, 2]
    let b = [a, a, {"x": a}]
    var cyc = []
    cyc.push(cyc)
    let n1 = Node { name: "one", next: None }
    let n2 = Node { name: "two", next: n1 }
    n1.next = n2
    var text = ""
    for i in 0..12
        a.push(i)
        text = text + str(i % 10)
        let inner = b[2]
        let key = "x"
        cyc[0].push(i)
        print(f"{len(b[0])} {len(b[1])} {len(inner[key])} {len(cyc[0][0])} {n1.next.next.name} {text}")
    print(votes)
'''

TIMEOUT1 = '''
main()
    let r = timeout(100000)
        var s = 0
        for k in 0..300
            s += k
        s
    else
        "stopped: " + error
    print(r)
'''

TIMEOUT = '''
main()
    let r = timeout(100000)
        var s = 0
        for k in 0..300
            s += k
        s
    else
        "stopped: " + error
    print(r)
    let r2 = timeout(40)
        var n = 0
        loop
            n += 1
        n
    else
        "stopped: " + error
    print(r2)
'''

GRAPHICS = '''
main()
    screen_init(32, 24)
    var frame = 0
    while frame < 25
        screen_clear(0x102030)
        fill_rect(frame % 32, 4, 6, 6, 0xFF0000)
        draw_line(0, 0, 31, frame % 24, 0x00FF00)
        draw_text(1, 14, str(frame), 0xFFFFFF)
        screen_show()
        frame += 1
    print(f"{screen_width()}x{screen_height()} {get_pixel(frame % 32 - 1, 5)}")
'''

BIG_INT = '''
main()
    var big = 1
    for i in 0..40
        big = big * 2
    var k = 0
    while k < 5
        k += 1
    print(big + k)
'''

FLOATY = '''
main()
    let f = float(3)
    var s = 0
    for i in 0..20
        s += i
    print(s)
'''

INTS = '''
main()
    var s = 0
    var xs = []
    for i in 0..60
        s += i * 3
        xs.push(i % 7)
        if i % 10 == 0
            print(f"{i} {s} {len(xs)}")
    print(s)
'''

PROGRAMS = {'mixed': MIXED, 'closures': CLOSURES, 'effects': EFFECTS, 'retry': RETRY, 'maps': MAPS}


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

_TBC = {}


def program(src: str) -> bytes:
    if src not in _TBC:
        _TBC[src] = tbc_bytes(src)
    return _TBC[src]


def new_vm(tbc: bytes, out: list, **kw) -> 'til.TILVM':
    return til.TILVM(til.TBCModule.from_bytes(tbc), write=out.append, **kw)


def reference(tbc: bytes):
    out = []
    vm = new_vm(tbc, out)
    rc = vm.run()
    return rc, b"".join(out), vm.steps


def py_snapshots(tbc: bytes, every: int = 1):
    """Run on the reference VM, saving after every `every` steps:
    [(steps, output so far, snapshot)]."""
    out = []
    vm = new_vm(tbc, out)
    snaps = []
    while vm.run_steps(every) == vm.YIELD:
        snaps.append((vm.steps, b"".join(out), vm.save_portable()))
    return snaps


def py_resume(tbc: bytes, data: bytes, **kw):
    out = []
    vm = new_vm(tbc, out, **kw)
    vm.load_portable(data)
    rc = vm.run()
    return rc, b"".join(out), vm


def write(tmp_path, name: str, data: bytes) -> str:
    path = os.path.join(str(tmp_path), name)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def c_save(exe: str, tbc_path: str, steps: int, snap_path: str, *extra):
    r = subprocess.run([exe, *extra, '--save-portable-after', str(steps), snap_path, tbc_path],
                       capture_output=True, timeout=60)
    data = b""
    if r.returncode == 3:
        with open(snap_path, 'rb') as f:
            data = f.read()
    return r, data


def c_resume(exe: str, tbc_path: str, snap_path: str, *extra):
    return subprocess.run([exe, *extra, '--resume-portable', snap_path, tbc_path], capture_output=True,
                          timeout=60)


def masked(data: bytes, rng: bool = False) -> bytes:
    """A snapshot without the parts that may differ between VMs that agree: the
    time left of timeout blocks (clock), the generator state of 32-bit builds
    (rng=True), and the checksum."""
    d = decode(data)
    b = bytearray(data[:-4])
    if rng:
        b[14:22] = bytes(8)
    for off in d['deadline_offsets']:
        b[off:off + 4] = bytes(4)
    return bytes(b)


def decode(data: bytes) -> dict:
    """Parse a .tilp with the byte layout of docs/TIL_SNAPSHOTS.md - an
    implementation of the document independent of both VMs."""
    p = 0

    def rd(fmt):
        nonlocal p
        v = struct.unpack_from('<' + fmt, data, p)
        p += struct.calcsize('<' + fmt)
        return v if len(v) > 1 else v[0]

    n_ids = 0

    def ref(i):
        nonlocal n_ids
        assert 1 <= i <= n_ids + 1, f"object id {i} out of order"
        n_ids = max(n_ids, i)
        return ('ref', i)

    def value():
        tag = rd('B')
        if tag <= 2:
            return (None, False, True)[tag]
        if tag == 3:
            return rd('q')
        if tag == 4:
            return rd('d')
        if tag == 5:
            return ('const', rd('H'))
        if tag == 6:
            return ref(rd('I'))
        if tag == 7:
            return ('func', rd('H'))
        assert tag == 8, f"bad tag {tag}"
        return ('undef',)

    assert data[:4] == b"TILP"
    p = 4
    d = {}
    d['version'], d['flags'], d['hash'] = rd('BBI')
    d['steps'], d['rng'], d['tick'], d['energy'], d['reserved'] = rd('IQHBB')
    n_globals = rd('H')
    d['frames'] = []
    for _ in range(rd('H')):
        func, pc, base, extra, clos = rd('HIHBI')
        d['frames'].append((func, pc, base, extra, ref(clos) if clos else 0))
    d['tries'], d['deadline_offsets'] = [], []
    for _ in range(rd('H')):
        d['deadline_offsets'].append(p + 4)
        d['tries'].append(rd('IIHHH'))
    d['handlers'] = [rd('BHHHIH') for _ in range(rd('H'))]
    d['globals'] = [value() for _ in range(n_globals)]
    d['stack'] = [value() for _ in range(rd('H'))]
    d['screen'] = rd('HH')
    objs = []
    while len(objs) < n_ids:
        kind = rd('B')
        if kind == 1:
            n = rd('I')
            objs.append(('str', data[p:p + n]))
            p += n
        elif kind in (2, 3):
            n = rd('I')
            objs.append(('array' if kind == 2 else 'map', [value() for _ in range(n * (kind - 1))]))
        elif kind in (4, 5):
            a, n = rd('HH')
            objs.append(('struct' if kind == 4 else 'closure', a, [value() for _ in range(n)]))
        else:
            assert kind == 6, f"bad object kind {kind}"
            objs.append(('cell', value()))
    d['objects'] = objs
    d['size'] = p
    assert rd('I') == len(objs)
    body = p
    assert rd('I') == zlib.crc32(data[:body]) & 0xFFFFFFFF
    assert p == len(data)
    return d


def all_values(d: dict):
    vals = list(d['globals']) + list(d['stack'])
    for o in d['objects']:
        if o[0] in ('array', 'map'):
            vals += o[1]
        elif o[0] in ('struct', 'closure'):
            vals += o[2]
        elif o[0] == 'cell':
            vals.append(o[1])
    return vals


# ═══════════════════════════════════════════════════════════════════════════
# the format
# ═══════════════════════════════════════════════════════════════════════════

class TestFormat:
    def test_layout(self):
        tbc = program(MIXED)
        snaps = py_snapshots(tbc, 7)
        assert snaps
        for steps, _out, data in snaps:
            d = decode(data)
            assert d['size'] + 8 == len(data)
            assert (d['version'], d['flags'], d['reserved'], d['energy']) == (1, 0, 0, 0)
            assert d['hash'] == til.fnv1a32(tbc)
            assert d['steps'] == steps
            assert d['frames'][0][2:4] == (0, 0)       # frame 0: base 0, no callee slot
            assert d['screen'] == (0, 0)

    def test_fnv1a(self):
        assert til.fnv1a32(b"") == 0x811C9DC5
        assert til.fnv1a32(b"a") == 0xE40C292C
        assert til.fnv1a32(b"foobar") == 0xBF9CF968

    def test_canonical_values(self):
        """Strings equal to a constant are constants; equal strings share one
        object; functions without upvalues are function values; a parameter not
        passed yet is a missing-argument value."""
        tbc = program(MIXED)
        consts = til.TBCModule.from_bytes(tbc).consts
        seen_undef = seen_func = seen_abc = False
        for _steps, _out, data in py_snapshots(tbc, 1):
            d = decode(data)
            strs = [o[1] for o in d['objects'] if o[0] == 'str']
            assert len(strs) == len(set(strs))
            assert not set(strs) & {c for c in consts if isinstance(c, bytes)}
            vals = all_values(d)
            seen_undef = seen_undef or ('undef',) in vals
            seen_func = seen_func or any(isinstance(v, tuple) and v[0] == 'func' for v in vals)
            seen_abc = seen_abc or ('const', consts.index(b"hello")) in vals   # also a slice made at run time
        assert seen_undef and seen_func and seen_abc

    def test_closures_and_cells_are_objects(self):
        d = decode(py_snapshots(program(CLOSURES), 300)[0][2])
        kinds = [o[0] for o in d['objects']]
        assert 'closure' in kinds and 'cell' in kinds
        for o in d['objects']:
            if o[0] == 'closure':
                assert all(d['objects'][v[1] - 1][0] == 'cell' for v in o[2])


# ═══════════════════════════════════════════════════════════════════════════
# the reference VM
# ═══════════════════════════════════════════════════════════════════════════

class TestReferenceVM:
    @pytest.mark.parametrize('name', sorted(PROGRAMS))
    def test_roundtrip_at_every_step(self, name):
        tbc = program(PROGRAMS[name])
        rc, ref, _steps = reference(tbc)
        assert rc == 0
        snaps = py_snapshots(tbc, 1)
        assert len(snaps) > 20
        for steps, prefix, data in snaps:
            rc2, rest, vm = py_resume(tbc, data)
            assert rc2 == 0 and prefix + rest == ref, f"pause after {steps} steps"
            # the same state again: the same bytes (deadlines aside)
            vm2 = new_vm(tbc, [])
            vm2.load_portable(data)
            assert masked(vm2.save_portable()) == masked(data)

    def test_run_steps(self):
        tbc = program(MIXED)
        _rc, ref, total = reference(tbc)
        out = []
        vm = new_vm(tbc, out)
        assert vm.run_steps(25) == vm.YIELD and vm.steps == 25
        assert vm.run_steps(1) == vm.YIELD and vm.steps == 26
        assert vm.run_steps(0) == vm.DONE and vm.steps == total and vm.exit_code == 0
        assert b"".join(out) == ref
        assert vm.run_steps(5) == vm.DONE
        with pytest.raises(til.SnapshotError):
            vm.save_portable()

    def test_exit_and_error_status(self):
        out = []
        vm = new_vm(program("main()\n    for i in 0..10\n        print(i)\n    exit(4)\n"), out)
        assert vm.run_steps(3) == vm.YIELD
        assert vm.run_steps(0) == vm.EXIT and vm.exit_code == 4
        vm = new_vm(program("main()\n    var xs = [1]\n    for i in 0..3\n        print(xs[i])\n"), [])
        assert vm.run_steps(0) == vm.ERROR and vm.exit_code == 1

    def test_save_before_the_first_step(self):
        tbc = program(INTS)
        vm = new_vm(tbc, [])
        data = vm.save_portable()
        assert decode(data)['steps'] == 0
        assert py_resume(tbc, data)[1] == reference(tbc)[1]

    def test_load_after_start_is_refused(self):
        tbc = program(INTS)
        vm = new_vm(tbc, [])
        vm.run_steps(3)
        data = vm.save_portable()
        with pytest.raises(til.SnapshotError):
            vm.load_portable(data)

    def test_identity_and_insertion_order(self):
        """Shared and cyclic references stay one object; maps keep their order."""
        tbc = program(MAPS)
        steps, _prefix, data = py_snapshots(tbc, 1)[-1]
        vm = new_vm(tbc, [])
        vm.load_portable(data)
        stack = vm.stack
        arrays = [v for v in stack if type(v) is list]
        both = [v for v in arrays if len(v) == 3 and type(v[0]) is list and v[0] is v[1]]
        assert both and both[0][2][b'x'] is both[0][0]
        cyc = [v for v in arrays if v and v[0] is v]
        assert cyc
        nodes = [v for v in stack if type(v) is til.VMStruct and vm.type_names[v.t] == 'Node']
        assert nodes and nodes[0].f[1].f[1] is nodes[0]
        votes = [v for v in stack if type(v) is dict and b"tea" in v][0]
        assert list(votes) == [b"tea", b"coffee", b"juice", b"milk", b"zz", 2]


# ═══════════════════════════════════════════════════════════════════════════
# reference VM <-> C VM
# ═══════════════════════════════════════════════════════════════════════════

def cross_check(tmp_path, src: str, exe: str, every: int = 1, identical: bool = True, rng: bool = False,
                limit: int = 0):
    """Reference VM -> C VM and C VM -> reference VM at every `every`-th step:
    the outputs add up to an uninterrupted run, and both VMs write the same
    snapshot (identical=True; deadlines and, with rng=True, the generator of a
    32-bit C build aside)."""
    tbc = program(src)
    tbc_path = write(tmp_path, 'p.tbc', tbc)
    rc, ref, _ = reference(tbc)
    snaps = py_snapshots(tbc, every)
    if limit and len(snaps) > limit:
        snaps = snaps[::len(snaps) // limit + 1]
    assert snaps
    for steps, prefix, data in snaps:
        snap = write(tmp_path, 'py.tilp', data)
        r = c_resume(exe, tbc_path, snap)
        assert r.returncode == rc and prefix + r.stdout == ref, \
            f"reference VM -> C VM, {steps} steps: {r.stderr.decode()}"
        r1, cdata = c_save(exe, tbc_path, steps, os.path.join(str(tmp_path), 'c.tilp'))
        assert r1.returncode == 3, r1.stderr.decode()
        assert r1.stdout == prefix
        if identical:
            assert masked(cdata, rng) == masked(data, rng), f"snapshots differ after {steps} steps"
        rc2, rest, _vm = py_resume(tbc, cdata)
        assert rc2 == rc and r1.stdout + rest == ref, f"C VM -> reference VM, {steps} steps"
    return snaps


@needs_cc
class TestCrossVM:
    @pytest.mark.parametrize('name', sorted(PROGRAMS))
    def test_desktop_build_every_step(self, tmp_path, name):
        cross_check(tmp_path, PROGRAMS[name], cvm_binary('host'), 1)

    @pytest.mark.parametrize('profile', ['int32', 'small', 'gcstress'])
    @pytest.mark.parametrize('name', sorted(PROGRAMS))
    def test_other_builds(self, tmp_path, name, profile):
        cross_check(tmp_path, PROGRAMS[name], cvm_binary(profile), 3, rng=profile != 'gcstress', limit=60)

    @pytest.mark.parametrize('scratch', ['1', '2'])
    @pytest.mark.parametrize('name', ['closures', 'maps', 'effects'])
    def test_queue_and_table_fallbacks(self, tmp_path, name, scratch):
        """A tiny save queue (heap rescans, as after a mark-stack overflow) and a
        tiny id table (loading finds objects by walking the heap) give the same
        bytes and the same results."""
        exe = cvm_binary('host', ['-DTILVM_P_SCRATCH=' + scratch])
        cross_check(tmp_path, PROGRAMS[name], exe, 2)

    def test_mark_stack_queue(self, tmp_path):
        """When the free end of the arena is not larger than the GC's mark stack,
        the save queues objects there (here: a mark stack larger than the arena)."""
        exe = cvm_binary('host', ['-DTILVM_HEAP_SIZE=16384', '-DTILVM_MARK_STACK=20000'])
        cross_check(tmp_path, MAPS, exe, 2)

    def test_tiny_heap_2k(self, tmp_path):
        exe = cvm_binary('host', TINY2K)
        cross_check(tmp_path, INTS, exe, 4, rng=True)
        cross_check(tmp_path, '''
main()
    var cells = []
    for i in 0..30
        cells.push([i, i * 2])
    var s = 0
    for c in cells
        s += c[0] + c[1]
    print(s)
''', exe, 3, rng=True)

    def test_does_not_fit_a_tiny_heap(self, tmp_path):
        tbc = program('''
main()
    var xs = []
    for i in 0..400
        xs.push([i])
    var k = 0
    while k < 3
        k += 1
    print(len(xs))
''')
        out = []
        vm = new_vm(tbc, out)
        assert vm.run_steps(401) == vm.YIELD
        snap = write(tmp_path, 's.tilp', vm.save_portable())
        r = c_resume(cvm_binary('host', TINY2K), write(tmp_path, 'p.tbc', tbc), snap)
        assert r.returncode == 2
        assert "does not fit this VM" in r.stderr.decode()

    def test_several_hops(self, tmp_path):
        """reference VM -> C VM -> reference VM -> 32-bit C VM -> reference VM"""
        tbc = program(MIXED)
        tbc_path = write(tmp_path, 'p.tbc', tbc)
        _rc, ref, _total = reference(tbc)
        out = []
        vm = new_vm(tbc, out)
        assert vm.run_steps(40) == vm.YIELD
        got = b"".join(out)
        snap = write(tmp_path, 'a.tilp', vm.save_portable())
        r = subprocess.run([cvm_binary('host'), '--resume-portable', snap, '--save-portable-after', '30',
                            os.path.join(str(tmp_path), 'b.tilp'), tbc_path], capture_output=True)
        assert r.returncode == 3, r.stderr
        got += r.stdout
        with open(os.path.join(str(tmp_path), 'b.tilp'), 'rb') as f:
            data = f.read()
        out = []
        vm = new_vm(tbc, out)
        vm.load_portable(data)
        assert vm.run_steps(25) == vm.YIELD
        got += b"".join(out)
        snap = write(tmp_path, 'c.tilp', vm.save_portable())
        r = subprocess.run([cvm_binary('int32'), '--resume-portable', snap, '--save-portable-after', '20',
                            os.path.join(str(tmp_path), 'd.tilp'), tbc_path], capture_output=True)
        assert r.returncode == 3, r.stderr
        got += r.stdout
        with open(os.path.join(str(tmp_path), 'd.tilp'), 'rb') as f:
            rc, rest, _vm = py_resume(tbc, f.read())
        assert rc == 0 and got + rest == ref

    def test_effect_handlers_inside_clause_and_body(self, tmp_path):
        """Pauses inside a running clause (a kind-1 handler entry) and inside a
        handled body (kind-0 entries only) - both covered, both work."""
        snaps = cross_check(tmp_path, EFFECTS, cvm_binary('host'), 1, limit=0)
        kinds = [{h[0] for h in decode(data)['handlers']} for _s, _o, data in snaps]
        assert any(1 in k for k in kinds)
        assert any(k == {0} for k in kinds)

    def test_no_effects_build(self, tmp_path):
        tbc = program(EFFECTS)
        data = [d for _s, _o, d in py_snapshots(tbc, 1) if decode(d)['handlers']][0]
        r = c_resume(cvm_binary('host', ['-DTILVM_NO_EFFECTS']), write(tmp_path, 'p.tbc', tbc),
                     write(tmp_path, 's.tilp', data))
        assert r.returncode == 2 and "effect handlers" in r.stderr.decode()

    def test_timeout_keeps_the_time_left(self, tmp_path):
        cross_check(tmp_path, TIMEOUT1, cvm_binary('host'), 1)
        tbc = program(TIMEOUT)
        tbc_path = write(tmp_path, 'p.tbc', tbc)
        rc, ref, _ = reference(tbc)
        assert ref.endswith(b"stopped: timeout\n")
        out = []
        vm = new_vm(tbc, out)
        assert vm.run_steps(100) == vm.YIELD
        d = decode(vm.save_portable())
        assert len(d['tries']) == 1 and 99000 <= d['tries'][0][1] <= 100000
        # a pause inside the block that times out: the rest of its 40 ms runs out after the resume
        out = []
        vm = new_vm(tbc, out)
        while not (out and vm.recover):
            assert vm.run_steps(1) == vm.YIELD
        assert vm.run_steps(3) == vm.YIELD
        d = decode(vm.save_portable())
        assert len(d['tries']) == 1 and 1 <= d['tries'][0][1] <= 40
        snap = write(tmp_path, 's.tilp', vm.save_portable())
        r = c_resume(cvm_binary('host'), tbc_path, snap)
        assert b"".join(out) + r.stdout == ref
        assert py_resume(tbc, vm.save_portable())[1] == r.stdout

    def test_graphics_screen_is_re_created(self, tmp_path):
        tbc = program(GRAPHICS)
        tbc_path = write(tmp_path, 'p.tbc', tbc)
        out = []
        vm = new_vm(tbc, out)
        assert vm.run() == 0
        ref, ref_ppm = b"".join(out), vm.screen.ppm()
        snaps = py_snapshots(tbc, 9)
        assert snaps
        for steps, prefix, data in snaps:
            assert decode(data)['screen'] == (32, 24)
            ppm = os.path.join(str(tmp_path), 'c.ppm')
            r = c_resume(cvm_binary('host'), tbc_path, write(tmp_path, 's.tilp', data), '--screen', ppm)
            assert r.returncode == 0 and prefix + r.stdout == ref
            with open(ppm, 'rb') as f:
                assert f.read() == ref_ppm
            r1, cdata = c_save(cvm_binary('host'), tbc_path, steps, os.path.join(str(tmp_path), 'c.tilp'))
            assert masked(cdata) == masked(data)
            rc, rest, vm2 = py_resume(tbc, cdata)
            assert rc == 0 and r1.stdout + rest == ref and vm2.screen.ppm() == ref_ppm

    def test_screen_too_large_for_the_build(self, tmp_path):
        tbc = program(GRAPHICS)
        data = py_snapshots(tbc, 9)[0][2]
        tbc_path = write(tmp_path, 'p.tbc', tbc)
        snap = write(tmp_path, 's.tilp', data)
        r = c_resume(cvm_binary('host', ['-DTILVM_GFX_MAX_W=16', '-DTILVM_GFX_MAX_H=16']), tbc_path, snap)
        assert r.returncode == 2 and "screen" in r.stderr.decode()
        r = c_resume(cvm_binary('host', ['-DTILVM_NO_GFX']), tbc_path, snap)
        assert r.returncode == 2 and "screen" in r.stderr.decode()
        with pytest.raises(til.SnapshotError) as e:
            py_resume(tbc, data, screen=til.TILScreen(max_w=16, max_h=16))
        assert e.value.code == -8

    def test_strings_over_64k(self, tmp_path):
        src = '''
main()
    var s = "ab".repeat(40000)
    var parts = [s, s, s.slice(0, 70000)]
    var n = 0
    for i in 0..9
        n += len(parts[i % 3])
    print(n)
    print(len(s + "x"))
'''
        snaps = cross_check(tmp_path, src, cvm_binary('host'), 2)
        d = decode(snaps[-1][2])
        strs = [o[1] for o in d['objects'] if o[0] == 'str']
        assert len(strs) == 2 and sorted(map(len, strs)) == [70000, 80000]   # s once (shared)
        big = cvm_binary('host', ['-DTILVM_HEAP_SIZE=1048576'])
        cross_check(tmp_path, src.replace('"ab".repeat(40000)', '"abc".repeat(100000)'), big, 4)
        tbc = program(src)
        r = c_resume(cvm_binary('host', ['-DTILVM_HEAP_SIZE=60000']), write(tmp_path, 'p.tbc', tbc),
                     write(tmp_path, 's.tilp', snaps[-1][2]))
        assert r.returncode == 2 and "does not fit this VM" in r.stderr.decode()

    def test_int64_into_a_16_bit_vm(self, tmp_path):
        tbc = program(BIG_INT)
        tbc_path = write(tmp_path, 'p.tbc', tbc)
        exe = cvm_binary('host', INT16)
        snaps = py_snapshots(tbc, 1)
        # before the value grows past 16 bits it loads (and the run continues in 16 bits)
        r = c_resume(exe, tbc_path, write(tmp_path, 'a.tilp', snaps[2][2]))
        assert r.returncode == 0, r.stderr
        r = c_resume(exe, tbc_path, write(tmp_path, 'b.tilp', snaps[-1][2]))
        assert r.returncode == 2
        assert "integer that does not fit this VM" in r.stderr.decode()
        r = c_resume(cvm_binary('int32'), tbc_path, write(tmp_path, 'b.tilp', snaps[-1][2]))
        assert r.returncode == 2 and "integer" in r.stderr.decode()
        assert c_resume(cvm_binary('host'), tbc_path, write(tmp_path, 'b.tilp', snaps[-1][2])).returncode == 0

    def test_float_into_a_build_without_floats(self, tmp_path):
        tbc = program(FLOATY)
        assert not til.TBCModule.from_bytes(tbc).uses_float
        data = py_snapshots(tbc, 5)[0][2]
        assert any(type(v) is float for v in all_values(decode(data)))
        r = c_resume(cvm_binary('host', INT16), write(tmp_path, 'p.tbc', tbc), write(tmp_path, 's.tilp', data))
        assert r.returncode == 2 and "floats" in r.stderr.decode()

    def test_floats_in_a_float32_build(self, tmp_path):
        """A `float` (binary32) C VM rounds the file's doubles to nearest-even
        when it loads them (subnormals, overflow to inf included) and writes the
        exact double of its value."""
        src = '''
main()
    let xs = [0.5, -1.25, 3.0e39, -3.0e39, 1.0e-40, 1.0e-46, 0.1, 123456.789, 2.0e-38, 0.0, -0.0]
    var k = 0
    while k < 3
        k += 1
    print(len(xs))
'''
        tbc = program(src)
        tbc_path = write(tmp_path, 'p.tbc', tbc)
        out = []
        vm = new_vm(tbc, out)
        assert vm.run_steps(2) == vm.YIELD
        xs = [v for v in vm.stack if type(v) is list][0]
        r = subprocess.run([cvm_binary('int32'), '--resume-portable', write(tmp_path, 's.tilp', vm.save_portable()),
                            '--save-portable-after', '1', os.path.join(str(tmp_path), 'c.tilp'), tbc_path],
                           capture_output=True)
        assert r.returncode == 3, r.stderr
        with open(os.path.join(str(tmp_path), 'c.tilp'), 'rb') as f:
            d = decode(f.read())
        floats = [v for o in d['objects'] if o[0] == 'array' for v in o[1] if type(v) is float]
        want = [struct.unpack('<f', struct.pack('<f', x))[0] if abs(x) < 3.5e38 else
                (float('inf') if x > 0 else float('-inf')) for x in xs]
        assert len(floats) == len(want)
        for got, w in zip(floats, want):
            assert struct.pack('<d', got) == struct.pack('<d', w), (got, w)

    def test_wrong_program(self, tmp_path):
        data = py_snapshots(program(INTS), 5)[0][2]
        other = program(INTS.replace('i * 3', 'i * 4'))
        with pytest.raises(til.SnapshotError) as e:
            py_resume(other, data)
        assert e.value.code == -3
        r = c_resume(cvm_binary('host'), write(tmp_path, 'o.tbc', other), write(tmp_path, 's.tilp', data))
        assert r.returncode == 2 and "different program" in r.stderr.decode()

    def test_damaged_snapshots_same_codes(self, tmp_path):
        """Truncated, altered or foreign data: both VMs refuse it with the same code."""
        tbc = program(MIXED)
        tbc_path = write(tmp_path, 'p.tbc', tbc)
        data = py_snapshots(tbc, 30)[0][2]
        bad = {
            -1: [b"XILP" + data[4:], b"\x7fELF" + bytes(40)],
            -2: [data[:4] + b"\x02" + data[5:], data[:5] + b"\x01" + data[6:]],
            -4: [data[:n] for n in (0, 3, 9, 30, len(data) // 2, len(data) - 1)] +
                [data[:-1] + bytes([data[-1] ^ 1]), data + b"\x00", data[:40] + bytes([data[40] ^ 0x10]) + data[41:]],
        }
        messages = {code: til.TILP_ERRORS[code] for code in bad}
        for code, cases in bad.items():
            for case in cases:
                with pytest.raises(til.SnapshotError) as e:
                    py_resume(tbc, case)
                assert e.value.code == code, (code, case[:12])
                r = c_resume(cvm_binary('host'), tbc_path, write(tmp_path, 'x.tilp', case))
                assert r.returncode == 2 and messages[code] in r.stderr.decode(), (code, r.stderr)

    def test_lean_builds_report_not_available(self, tmp_path):
        tbc = program(INTS)
        tbc_path = write(tmp_path, 'p.tbc', tbc)
        exe = cvm_binary('tiny')
        r = c_resume(exe, tbc_path, write(tmp_path, 's.tilp', py_snapshots(tbc, 5)[0][2]))
        assert r.returncode == 2 and "not available in this VM build" in r.stderr.decode()
        r, _data = c_save(exe, tbc_path, 5, os.path.join(str(tmp_path), 'c.tilp'))
        assert r.returncode == 2 and "not available in this VM build" in r.stderr.decode()

    def test_save_at_step_0_and_program_ending_first(self, tmp_path):
        tbc = program(INTS)
        tbc_path = write(tmp_path, 'p.tbc', tbc)
        _rc, ref, total = reference(tbc)
        r, data = c_save(cvm_binary('host'), tbc_path, 0, os.path.join(str(tmp_path), 'c.tilp'))
        assert r.returncode == 3 and r.stdout == b"" and decode(data)['steps'] == 0
        assert masked(data) == masked(new_vm(tbc, []).save_portable())
        assert py_resume(tbc, data)[1] == ref
        r, data = c_save(cvm_binary('host'), tbc_path, total + 5, os.path.join(str(tmp_path), 'd.tilp'))
        assert r.returncode == 0 and r.stdout == ref and data == b""
        assert b"ended before the pause point" in r.stderr


# ═══════════════════════════════════════════════════════════════════════════
# the C API (vm/tilvm.h) from a host program
# ═══════════════════════════════════════════════════════════════════════════

HOST_C = r'''
#include "tilvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static tilvm vm;
static unsigned char *buf;
static uint32_t len, cap;
static long limit = -1;

static int put_mem(void *ctx, const uint8_t *p, uint32_t n) {
    (void)ctx;
    if (limit >= 0 && len + n > (uint32_t)limit) return -1;
    if (len + n > cap) { cap = 2 * (len + n); buf = (unsigned char *)realloc(buf, cap); }
    memcpy(buf + len, p, n);
    len += n;
    return 0;
}

struct src { const unsigned char *p; uint32_t left; };
static int get_mem(void *ctx, uint8_t *b, uint32_t n) {
    struct src *s = (struct src *)ctx;
    if (n > s->left) return -1;
    memcpy(b, s->p, n);
    s->p += n;
    s->left -= n;
    return 0;
}

static unsigned char *slurp(const char *path, long *n) {
    FILE *f = fopen(path, "rb");
    unsigned char *b;
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = (unsigned char *)malloc((size_t)*n + 1);
    if (fread(b, 1, (size_t)*n, f) != (size_t)*n) exit(9);
    fclose(f);
    return b;
}

/* host save PROG N OUT: run N steps, save twice (same bytes), fail a save
 * half-way (0, VM unchanged), save again, run to the end.
 * host load PROG SNAP: load from memory, run to the end. */
int main(int argc, char **argv) {
    long n;
    unsigned char *code, *first;
    uint32_t a, b, c;
    if (argc < 4) return 6;
    code = slurp(argv[2], &n);
    if (tilvm_load(&vm, code, (uint32_t)n) != 0) return 8;
    if (!strcmp(argv[1], "load")) {
        long sn;
        unsigned char *snap = slurp(argv[3], &sn);
        struct src s;
        int rc;
        s.p = snap;
        s.left = (uint32_t)sn;
        rc = tilvm_load_portable(&vm, get_mem, &s);
        if (rc) { fprintf(stderr, "load error %d\n", rc); return 2; }
        return tilvm_run(&vm);
    }
    if (tilvm_run_steps(&vm, (uint32_t)atol(argv[3])) != TILVM_YIELD) return 7;
    a = tilvm_save_portable(&vm, put_mem, 0);
    first = (unsigned char *)malloc(a);
    memcpy(first, buf, a);
    len = 0;
    b = tilvm_save_portable(&vm, put_mem, 0);
    if (a == 0 || a != b || memcmp(first, buf, a) != 0) { fprintf(stderr, "second save differs\n"); return 5; }
    len = 0;
    limit = (long)a / 2;
    if (tilvm_save_portable(&vm, put_mem, 0) != 0) { fprintf(stderr, "failing put\n"); return 5; }
    limit = -1;
    len = 0;
    c = tilvm_save_portable(&vm, put_mem, 0);
    if (c != a || memcmp(first, buf, a) != 0) { fprintf(stderr, "save after a failed save differs\n"); return 5; }
    {
        FILE *f = fopen(argv[4], "wb");
        fwrite(first, 1, a, f);
        fclose(f);
    }
    fprintf(stderr, "saved %lu\n", (unsigned long)a);
    return tilvm_run(&vm);
}
'''


@needs_cc
class TestCAPI:
    @pytest.fixture(scope='class')
    @classmethod
    def host(cls, tmp_path_factory):
        d = tmp_path_factory.mktemp('host')
        src = os.path.join(str(d), 'host.c')
        with open(src, 'w') as f:
            f.write(HOST_C)
        exe = os.path.join(str(d), 'host')
        r = subprocess.run([CC, '-std=c99', '-O1', '-Wall', '-Wextra', '-pedantic', '-Werror', '-DTILVM_HEAP_SIZE=8192',
                            '-I', VM_DIR, '-o', exe, src, os.path.join(VM_DIR, 'tilvm.c'), '-lm'],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return exe

    @pytest.mark.parametrize('name', ['closures', 'maps', 'effects'])
    def test_save_leaves_the_vm_unchanged(self, tmp_path, host, name):
        """Saving twice gives the same bytes, a failed `put` changes nothing, and
        the program continues as if it had not been saved."""
        tbc = program(PROGRAMS[name])
        tbc_path = write(tmp_path, 'p.tbc', tbc)
        _rc, ref, total = reference(tbc)
        snap = os.path.join(str(tmp_path), 's.tilp')
        for steps in (1, total // 3, total // 2, total - 1):
            r = subprocess.run([host, 'save', tbc_path, str(steps), snap], capture_output=True)
            assert r.returncode == 0 and r.stdout == ref, r.stderr
            with open(snap, 'rb') as f:
                data = f.read()
            r2 = subprocess.run([host, 'load', tbc_path, snap], capture_output=True)
            assert r2.returncode == 0, r2.stderr
            rc, rest, _vm = py_resume(tbc, data)
            assert ref.endswith(r2.stdout) and r2.stdout == rest


# ═══════════════════════════════════════════════════════════════════════════
# command line
# ═══════════════════════════════════════════════════════════════════════════

def til_file(tmp_path, src: str, name: str = 'prog.til') -> str:
    path = os.path.join(str(tmp_path), name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src)
    return path


class TestCommandLine:
    def test_run_vm_pause_save_resume(self, tmp_path):
        prog = til_file(tmp_path, MIXED)
        snap = os.path.join(str(tmp_path), 's.tilp')
        ref = cli('run', '--vm', prog)
        assert ref.returncode == 0
        a = cli('run', '--vm', prog, '--pause-after', '60', '--save', snap)
        assert a.returncode == 3
        assert "til: paused after 60 steps; saved " + snap in a.stderr
        b = cli('run', '--vm', prog, '--resume', snap)
        assert b.returncode == 0 and a.stdout + b.stdout == ref.stdout
        assert decode(open(snap, 'rb').read())['steps'] == 60

    def test_exec_tbc(self, tmp_path):
        prog = til_file(tmp_path, MIXED)
        tbc = os.path.join(str(tmp_path), 'prog.tbc')
        assert cli('build', '--target', 'vm', prog, '-o', tbc).returncode == 0
        snap = os.path.join(str(tmp_path), 's.tilp')
        a = cli('exec', tbc, '--pause-after', '33', '--save', snap)
        assert a.returncode == 3
        b = cli('exec', tbc, '--resume', snap)
        assert a.stdout + b.stdout == cli('exec', tbc).stdout
        # `til run --vm prog.til` makes the same .tbc: its snapshots are interchangeable
        c = cli('run', '--vm', prog, '--resume', snap)
        assert c.returncode == 0 and c.stdout == b.stdout

    @needs_cc
    def test_cvm_and_vm_in_turns(self, tmp_path):
        prog = til_file(tmp_path, CLOSURES)
        ref = cli('run', '--vm', prog).stdout
        s1, s2, s3 = (os.path.join(str(tmp_path), f's{i}.tilp') for i in range(3))
        a = cli('run', '--vm', prog, '--pause-after', '50', '--save', s1)
        b = cli('run', '--cvm', prog, '--resume', s1, '--pause-after', '70', '--save', s2)
        assert b.returncode == 3 and "tilvm: paused after 70 steps; saved" in b.stderr
        c = cli('run', '--vm', prog, '--resume', s2, '--pause-after', '40', '--save', s3)
        d = cli('run', '--cvm', prog, '--resume', s3)
        assert (a.returncode, c.returncode, d.returncode) == (3, 3, 0)
        assert a.stdout + b.stdout + c.stdout + d.stdout == ref

    def test_options_go_together(self, tmp_path):
        prog = til_file(tmp_path, INTS)
        r = cli('run', '--vm', prog, '--pause-after', '5')
        assert r.returncode == 2 and "--pause-after N and --save FILE go together" in r.stderr
        r = cli('run', '--vm', prog, '--save', 'x.tilp')
        assert r.returncode == 2
        r = cli('run', '--vm', prog, '--pause-after', 'many', '--save', 'x.tilp')
        assert r.returncode == 2 and "needs a number of steps" in r.stderr

    def test_without_vm_flag_uses_the_vm(self, tmp_path):
        prog = til_file(tmp_path, INTS)
        snap = os.path.join(str(tmp_path), 's.tilp')
        a = cli('run', prog, '--pause-after', '10', '--save', snap)
        assert a.returncode == 3 and os.path.exists(snap)

    def test_program_ends_first(self, tmp_path):
        prog = til_file(tmp_path, INTS)
        snap = os.path.join(str(tmp_path), 's.tilp')
        r = cli('run', '--vm', prog, '--pause-after', '100000', '--save', snap)
        assert r.returncode == 0 and "ended before the pause point" in r.stderr
        assert not os.path.exists(snap)

    def test_bad_snapshot_messages(self, tmp_path):
        prog = til_file(tmp_path, INTS)
        other = til_file(tmp_path, MIXED, 'other.til')
        snap = os.path.join(str(tmp_path), 's.tilp')
        cli('run', '--vm', prog, '--pause-after', '10', '--save', snap)
        r = cli('run', '--vm', other, '--resume', snap)
        assert r.returncode == 2
        assert r.stderr.strip() == f"til: cannot resume {snap}: the snapshot was saved by a different program"
        r = cli('run', '--vm', prog, '--resume', os.path.join(str(tmp_path), 'missing.tilp'))
        assert r.returncode == 2 and "cannot read" in r.stderr

    @pytest.mark.parametrize('vm', ['--vm', '--cvm'])
    def test_localized_messages(self, tmp_path, vm):
        if vm == '--cvm' and CC is None:
            pytest.skip("no C compiler")
        prog = til_file(tmp_path, INTS)
        snap = os.path.join(str(tmp_path), 's.tilp')
        r = cli('run', vm, prog, '--pause-after', '10', '--save', snap, '--lang', 'ru')
        assert r.returncode == 3 and "программа приостановлена (шагов: 10); сохранено:" in r.stderr
        r = cli('run', vm, prog, '--pause-after', '10', '--save', snap, '--lang', 'kk')
        assert r.returncode == 3 and "бағдарлама кідіртілді (қадам саны: 10)" in r.stderr
        wrong = til_file(tmp_path, MIXED, 'other.til')
        r = cli('run', vm, wrong, '--resume', snap, '--lang', 'ru')
        assert r.returncode == 2 and "снимок сохранён другой программой" in r.stderr


# ═══════════════════════════════════════════════════════════════════════════
# start on a PC, continue on a calculator (ARM Cortex-M on QEMU)
# ═══════════════════════════════════════════════════════════════════════════

cross = pytest.mark.skipif(not os.environ.get('TIL_TEST_CROSS'), reason="set TIL_TEST_CROSS=1")


@cross
@pytest.mark.skipif(not (shutil.which('qemu-system-arm') and shutil.which('arm-none-eabi-gcc')),
                    reason="needs arm-none-eabi-gcc and qemu-system-arm")
def test_continue_on_cortex_m(tmp_path):
    """docs/TIL_SNAPSHOTS.md §4: pause on the desktop C VM, continue on the
    Cortex-M port (int32, float, 16 KB heap, no stdio, no clock) in QEMU."""
    with open(os.path.join(ROOT, 'examples', '03_functions.til'), encoding='utf-8') as f:
        tbc = til.compile_to_bytecode(f.read()).serialize(strip=True)
    tbc_path = write(tmp_path, 'prog.tbc', tbc)
    snap = os.path.join(str(tmp_path), 'state.tilp')
    r1, _data = c_save(cvm_binary('host'), tbc_path, 40, snap)
    assert r1.returncode == 3
    port = os.path.join(VM_DIR, 'ports', 'cortex-m')
    subprocess.run(['make', '-s', '-C', port, 'clean'], check=True, capture_output=True)
    try:
        subprocess.run(['make', '-s', '-C', port, f'TBC={tbc_path}', f'SNAPSHOT={snap}'], check=True,
                       capture_output=True)
        r = subprocess.run(['qemu-system-arm', '-M', 'lm3s6965evb', '-nographic', '-semihosting-config',
                            'enable=on,target=native', '-kernel', os.path.join(port, 'build', 'tilvm.elf')],
                           capture_output=True, timeout=120)
    finally:
        subprocess.run(['make', '-s', '-C', port, 'clean'], capture_output=True)
    rest = b"".join(ln + b"\n" for ln in (r.stdout + r.stderr).splitlines() if b"Timer with period" not in ln)
    _rc, ref, _steps = reference(tbc)
    assert r1.stdout + rest == ref
