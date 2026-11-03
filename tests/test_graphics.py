#!/usr/bin/env python3
"""
TIL 3.1 screen graphics and keys (natives 40-53, docs/TIL_GRAPHICS.md).

Shapes and the 5x7 font are rasterized inside the VMs, so the reference VM
(src/til.py) and the C VM (vm/tilvm.c + vm/main.c) must produce identical
pixels (PPM bytes), identical terminal frames, identical key sequences and
identical errors.

Run: python3 -m pytest tests/test_graphics.py -v
"""

import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from tilhelpers import (CC, PROFILES, ROOT, VM_DIR, cli, cvm_binary, expect_compile_error, needs_cc,
                        run_cvm, tbc_bytes, til)

C_SOURCE = open(os.path.join(VM_DIR, 'tilvm.c'), encoding='utf-8').read()
STRICT = ['-std=c99', '-Wall', '-Wextra', '-pedantic', '-Werror']


# ═══════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════

_QUICK_VMS = {}


def quick_vm(*flags):
    """vm/tilvm.c + vm/main.c built without optimization (fast to build) with the
    strict flags every build must pass; one binary per flag set and session."""
    if CC is None:
        pytest.skip("no C compiler")
    if flags not in _QUICK_VMS:
        exe = os.path.join(tempfile.mkdtemp(prefix='tilgfx-vm-'), 'tilvm')
        r = subprocess.run([CC, *STRICT, '-O0', *flags, '-o', exe, os.path.join(VM_DIR, 'tilvm.c'),
                            os.path.join(VM_DIR, 'main.c'), '-lm'], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        _QUICK_VMS[flags] = exe
    return _QUICK_VMS[flags]


def vm_exe(profile):
    """host / int32: the shared optimized builds; other profiles: quick builds."""
    return cvm_binary(profile) if profile in ("host", "int32") else quick_vm(*PROFILES[profile])


def py_run(src, keys="", ansi=False):
    """Reference VM -> (exit code, stdout bytes, PPM bytes or None)."""
    data = src if isinstance(src, (bytes, bytearray)) else tbc_bytes(src)
    module = til.TBCModule.from_bytes(bytes(data))
    out = []
    screen = til.TILScreen(ansi=ansi)
    rc = til.TILVM(module, write=out.append, screen=screen, keys=keys).run()
    return rc, b"".join(out), (screen.ppm() if screen.w else None)


def c_run(src, keys="", ansi=False, profile="host", exe=None):
    """C VM -> (exit code, stdout bytes, stderr text, PPM bytes or None)."""
    data = src if isinstance(src, (bytes, bytearray)) else tbc_bytes(src)
    exe = exe or vm_exe(profile)
    work = tempfile.mkdtemp(prefix='tilgfx-')
    try:
        prog, ppm = os.path.join(work, 'p.tbc'), os.path.join(work, 'screen.ppm')
        with open(prog, 'wb') as f:
            f.write(data)
        args = [exe, '--screen', ppm, '--keys', keys] + (['--screen-ansi'] if ansi else []) + [prog]
        r = subprocess.run(args, capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
        image = None
        if os.path.exists(ppm):
            with open(ppm, 'rb') as f:
                image = f.read()
        return r.returncode, r.stdout, r.stderr.decode('utf-8', 'replace'), image
    finally:
        shutil.rmtree(work, ignore_errors=True)


def both(src, keys="", ansi=False, profile="host", expect_rc=0):
    """Run on both VMs; require identical exit code, stdout and PPM bytes."""
    data = tbc_bytes(src)
    rc, out, ppm = py_run(data, keys, ansi)
    assert rc == expect_rc, out
    if CC:
        crc, cout, cerr, cppm = c_run(data, keys, ansi, profile)
        assert crc == expect_rc, cerr
        assert cout == out, f"C VM output differs:\n{cout!r}\n{out!r}"
        assert cppm == ppm, "C VM pixels differ from the reference VM"
    return out.decode('utf-8'), ppm


def pixels(ppm: bytes):
    """PPM bytes -> (w, h, {(x, y): 0xRRGGBB})."""
    magic, size, maxval, data = ppm.split(b"\n", 3)
    assert magic == b"P6" and maxval == b"255"
    w, h = map(int, size.split())
    assert len(data) == 3 * w * h
    return w, h, {(i % w, i // w): (data[3 * i] << 16) | (data[3 * i + 1] << 8) | data[3 * i + 2]
                  for i in range(w * h)}


def lit(ppm: bytes, bg: int = 0):
    """Set of (x, y) whose colour differs from the background."""
    _w, _h, px = pixels(ppm)
    return {p for p, c in px.items() if c != bg}


# ═══════════════════════════════════════════════════════════════
# PIXELS: identical in both VMs
# ═══════════════════════════════════════════════════════════════

LINES = '''
screen_init(64, 48)
let ends = [[56, 30], [50, 44], [36, 46], [20, 45], [6, 36], [4, 20], [12, 6], [26, 2],
            [38, 1], [52, 5], [60, 16], [62, 24], [8, 24], [32, 47], [32, 1], [55, 47], [9, 1]]
let colors = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF, 0xFF8000, 0x8000FF,
              0x0080FF, 0x80FF00, 0xFF0080, 0x808080, 0xC0C0C0, 0x804000, 0x004080, 0x408000, 0xFFFFFF]
var i = 0
for e in ends
    draw_line(32, 24, e[0], e[1], colors[i])
    draw_line(e[0] + 1, e[1], 33, 25, colors[i])
    i += 1
draw_line(-20, 60, 80, -12, 0x123456)
draw_line(5, 5, 5, 5, 0xABCDEF)
draw_line(-100, -100, -1, -1, 0xFFFFFF)
draw_line(-32768, 10, 32767, 20, 0x654321)
draw_line(63, -32768, 0, 32767, 0x0F0F0F)
screen_show()
print(get_pixel(32, 24), get_pixel(5, 5), get_pixel(0, 0))
'''

SHAPES = '''
screen_init(50, 40)
screen_clear(0x101820)
draw_rect(2, 2, 10, 6, 0xFF0000)
draw_rect(14, 2, 1, 1, 0x00FF00)
draw_rect(16, 2, 5, 1, 0x00FF00)
draw_rect(22, 2, 1, 5, 0x00FF00)
draw_rect(24, 2, 0, 5, 0xFFFFFF)
draw_rect(24, 2, -3, 5, 0xFFFFFF)
draw_rect(-5, 30, 12, 20, 0x00FFFF)
draw_rect(40, -3, 20, 8, 0xFFFF00)
fill_rect(30, 10, 8, 5, 0x0000FF)
fill_rect(-10, 15, 14, 3, 0xFF00FF)
fill_rect(45, 35, 100, 100, 0x808080)
fill_rect(10, 20, 0, 4, 0xFFFFFF)
fill_rect(10, 20, 4, -4, 0xFFFFFF)
fill_rect(-32768, -32768, 32767, 32767, 0x202020)
draw_circle(25, 25, 0, 0xFFFFFF)
draw_circle(30, 25, 1, 0xFFFFFF)
draw_circle(36, 25, 2, 0xFFFFFF)
draw_circle(20, 30, 7, 0xFF8800)
draw_circle(0, 0, 9, 0x88FF00)
draw_circle(49, 39, 12, 0x0088FF)
draw_circle(25, 20, -3, 0xFFFFFF)
draw_circle(25, 32767, 32760, 0xFFC0CB)
draw_pixel(0, 39, 0xFFFFFF)
draw_pixel(-1, 0, 0xFFFFFF)
draw_pixel(50, 0, 0xFFFFFF)
draw_pixel(2000000000, 3, 0xFFFFFF)
screen_show()
'''

TEXT = '''
screen_init(120, 72)
var line = ""
var y = 0
for c in 32..127
    line = line + chr(c)
    if len(line) == 19
        draw_text(1, y, line, 0xFFFFFF)
        line = ""
        y += 8
draw_text(1, y, line, 0xFFFFFF)
draw_text(70, 41, "two\\nlines", 0x00FF00)
draw_text(1, 50, "Сәлем ✓ é\\tx", 0xFFFF00)
draw_text(-8, 60, "clip", 0xFF0000)
draw_text(100, 66, "edge", 0x00FFFF)
draw_text(50, -4, "top", 0xFF00FF)
draw_text(80, 58, 12345, 0x8080FF)
draw_text(-32768, 0, "far away", 0xFFFFFF)
screen_show()
'''


@pytest.mark.parametrize("name,src", [("lines", LINES), ("shapes", SHAPES), ("text", TEXT)])
def test_identical_pixels(name, src):
    _out, ppm = both(src)
    assert ppm is not None and ppm.startswith(b"P6\n")


@needs_cc
@pytest.mark.parametrize("name,src", [("lines", LINES), ("shapes", SHAPES), ("text", TEXT)])
def test_identical_pixels_int32_build(name, src):
    both(src, profile="int32")


def test_ppm_format():
    _out, ppm = both('screen_init(3, 2)\ndraw_pixel(2, 1, 0x102030)\n')
    assert ppm == b"P6\n3 2\n255\n" + bytes(15) + b"\x10\x20\x30"


def test_lines_in_all_octants_are_bresenham():
    """Random lines: end points included, 8-connected, max(|dx|,|dy|)+1 pixels."""
    rnd = random.Random(31)
    for _ in range(400):
        x0, y0, x1, y1 = (rnd.randint(-40, 40) for _ in range(4))
        pts = list(til.gfx_line_points(x0, y0, x1, y1))
        assert pts[0] == (x0, y0) and pts[-1] == (x1, y1)
        assert len(pts) == max(abs(x1 - x0), abs(y1 - y0)) + 1
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            assert max(abs(bx - ax), abs(by - ay)) == 1
    # the error term starts at dx - |dy|: ties step diagonally first (so A->B and
    # B->A may differ in a pixel; both VMs draw exactly these)
    assert list(til.gfx_line_points(0, 0, 4, 2)) == [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2)]
    assert list(til.gfx_line_points(4, 2, 0, 0)) == [(4, 2), (3, 1), (2, 1), (1, 0), (0, 0)]
    assert list(til.gfx_line_points(0, 0, -2, -4)) == [(0, 0), (-1, -1), (-1, -2), (-2, -3), (-2, -4)]


def test_circle_is_symmetric_midpoint():
    for r in range(0, 30):
        pts = set(til.gfx_circle_points(0, 0, r))
        assert pts == {(y, x) for x, y in pts} == {(-x, y) for x, y in pts}
        if r:
            assert (r, 0) in pts and (0, -r) in pts
            # every pixel lies within half a pixel of the true circle
            assert all(abs((x * x + y * y) ** 0.5 - r) < 0.75 for x, y in pts)
    assert set(til.gfx_circle_points(5, 6, 0)) == {(5, 6)}
    assert list(til.gfx_circle_points(5, 6, -1)) == []


GLYPH_A = ['.###.', '#...#', '#...#', '#...#', '#####', '#...#', '#...#']
REPLACEMENT = ['#####', '#...#', '#...#', '#...#', '#...#', '#...#', '#####']


def art(points, x0=0, y0=0, w=5, h=7):
    return [''.join('#' if (x0 + x, y0 + y) in points else '.' for x in range(w)) for y in range(h)]


def test_font_glyphs_and_layout():
    _out, ppm = both('screen_init(30, 18)\ndraw_text(1, 2, "AA\\nә", 0xFFFFFF)\n')
    on = lit(ppm)
    assert art(on, 1, 2) == GLYPH_A
    assert art(on, 7, 2) == GLYPH_A                  # 6 px advance
    assert art(on, 1, 10) == REPLACEMENT             # "\n": 8 px lower; one glyph per UTF-8 char
    assert len(on) == 2 * 18 + 20


def test_font_tables_match():
    table = re.search(r'FONT5X7\[96 \* 5\] = \{(.*?)\};', C_SOURCE, re.S).group(1)
    table = re.sub(r'/\*.*?\*/', '', table, flags=re.S)
    assert bytes(int(x, 16) for x in re.findall(r'0x([0-9A-Fa-f]{2})', table)) == til.FONT5X7
    assert len(til.FONT5X7) == 96 * 5
    assert all(b < 0x80 for b in til.FONT5X7)       # 7 rows


def test_screen_init_resizes_and_clears():
    out, ppm = both('''
screen_init(8, 8)
screen_clear(0xFFFFFF)
print(screen_width(), screen_height(), get_pixel(7, 7))
screen_init(5, 3)
print(screen_width(), screen_height(), get_pixel(4, 2), get_pixel(7, 7))
screen_clear()
draw_pixel(4, 2, 0x00FF00)
''')
    assert out == "8\n8\n16777215\n5\n3\n0\n0\n"
    assert pixels(ppm)[:2] == (5, 3) and lit(ppm) == {(4, 2)}


def test_text_of_any_value_like_print():
    out, ppm = both('''
struct P
    x: int
screen_init(160, 10)
draw_text(0, 0, P { x: 7 }, 0xFFFFFF)
draw_text(80, 0, [1, true, None], 0xFFFFFF)
print(len(str(P { x: 7 })))
''')
    assert out == "10\n"


def test_get_pixel_and_draw_pixel_accept_any_int():
    out, _ppm = both('''
screen_init(4, 4)
draw_pixel(3, 3, 0xFEDCBA)
draw_pixel(-9000000000, 9000000000, 0xFFFFFF)
print(get_pixel(3, 3), get_pixel(4, 0), get_pixel(0, -1), get_pixel(-9000000000, 0))
''')
    assert out == "16702650\n0\n0\n0\n"


# ═══════════════════════════════════════════════════════════════
# KEYS
# ═══════════════════════════════════════════════════════════════

KEYS_PROGRAM = '''
var got = []
got.push(key_pressed())
got.push(key_pressed())
got.push(wait_key())
got.push(key_pressed())
got.push(wait_key())
got.push(wait_key())
got.push(key_pressed())
got.push(wait_key())
print(got)
'''


def test_scripted_keys_and_exhaustion():
    out, ppm = both(KEYS_PROGRAM, keys="UP, none,Down,NONE,NONE,a,ESC")
    assert out == "[1, 0, 2, 0, 97, 27, 0, 27]\n" and ppm is None   # keys need no screen


def test_key_names_and_characters():
    out, _ = both('var k = []\nfor i in 0..12\n    k.push(key_pressed())\nprint(k)\n',
                  keys="left,RIGHT,Enter,space,TAB,backspace,COMMA,A,z,5,ә,ESC")
    assert out == "[3, 4, 13, 32, 9, 8, 44, 65, 122, 53, 1241, 27]\n"


def test_empty_script():
    assert both('print(key_pressed(), wait_key(), wait_key())\n', keys="")[0] == "0\n27\n27\n"


def test_parse_key_script():
    assert til.parse_key_script("UP,DOWN") == [1, 2]
    assert til.parse_key_script("  ") == []
    assert til.parse_key_script(" x , Y ") == [120, 89]
    for bad in ("UP,,DOWN", "UP,", "FOO", "ab", "ſpace"):
        with pytest.raises(ValueError, match="bad --keys entry"):
            til.parse_key_script(bad)


STREAM_PROGRAM = '''
var got = []
for i in 0..8
    got.push(wait_key())
got.push(key_pressed())
print(got)
'''
STREAM_INPUT = b"\x1b[Aa\n\x1bx\xd3\x99\x7f"
STREAM_KEYS = "[1, 97, 13, 27, 120, 1241, 8, 27, 0]\n"


def test_keys_from_stdin_bytes(tmp_path):
    """Without --keys and without a terminal, keys are stdin's bytes (arrows as
    escape sequences, UTF-8, Enter); at the end wait_key() -> 27, key_pressed() -> 0."""
    prog = tmp_path / "k.til"
    prog.write_text(STREAM_PROGRAM, encoding='utf-8')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'til.py'), 'run', '--vm', str(prog)],
                       input=STREAM_INPUT, capture_output=True, timeout=60)
    assert r.returncode == 0 and r.stdout.decode() == STREAM_KEYS, r.stderr
    if CC:
        tbc = tmp_path / "k.tbc"
        tbc.write_bytes(tbc_bytes(STREAM_PROGRAM))
        r = subprocess.run([cvm_binary(), str(tbc)], input=STREAM_INPUT, capture_output=True, timeout=60)
        assert r.returncode == 0 and r.stdout.decode() == STREAM_KEYS, r.stderr


def test_input_still_works_with_stdin_keys(tmp_path):
    prog = tmp_path / "k.til"
    prog.write_text('print(wait_key())\nprint(input("name? "))\nprint(wait_key())\n', encoding='utf-8')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'til.py'), 'run', '--vm', str(prog)],
                       input=b"qAigerim\n", capture_output=True, timeout=60)
    assert r.stdout.decode() == "113\nname? Aigerim\n27\n", r.stderr
    if CC:
        tbc = tmp_path / "k.tbc"
        tbc.write_bytes(tbc_bytes(prog.read_text(encoding='utf-8')))
        r = subprocess.run([cvm_binary(), str(tbc)], input=b"qAigerim\n", capture_output=True, timeout=60)
        assert r.stdout.decode() == "113\nname? Aigerim\n27\n", r.stderr


# ═══════════════════════════════════════════════════════════════
# ERRORS: identical messages, catchable
# ═══════════════════════════════════════════════════════════════

ERRORS = '''
struct P
    x: int

check(f)
    let r = retry(1)
        f()
        "ok"
    else
        error
    print(r)

check(|| draw_pixel(1, 1, 0))
check(|| screen_show())
check(|| get_pixel(0, 0))
print(screen_width(), screen_height(), key_pressed())
check(|| screen_init(0, 10))
check(|| screen_init(10, 4097))
check(|| screen_init(10.5, 10))
check(|| screen_init("8", 8))
check(|| screen_init(4096, 1))
screen_init(16, 16)
check(|| draw_pixel(1, 2, -1))
check(|| draw_pixel(1, 2, 0x1000000))
check(|| draw_pixel(1.0, 2, 3))
check(|| draw_pixel(true, 2, 3))
check(|| draw_line(0, 0, 32768, 0, 1))
check(|| draw_line(-32769, 0, 0, 0, 1))
check(|| draw_line(-32768, 32767, 0, 0, 0xFFFFFF))
check(|| draw_rect(0, 0, 5, None, 1))
check(|| fill_rect(0, 0, [1], 1, 1))
check(|| draw_circle(P { x: 1 }, 0, 1, 1))
check(|| draw_circle(0, 0, 40000, 1))
check(|| draw_text(0, 0, "x", "red"))
check(|| draw_text(70000, 0, "x", 1))
check(|| draw_text(0, 0, None, 1))
check(|| get_pixel(0.5, 0))
check(|| screen_clear(16777216))
check(|| screen_clear({}))
check(|| draw_pixel(1, 2, 3))
'''

ERRORS_OUT = """screen not initialized (call screen_init)
screen not initialized (call screen_init)
screen not initialized (call screen_init)
0
0
0
screen_init() argument 1 must be 1..4096, got 0
screen_init() argument 2 must be 1..4096, got 4097
screen_init() argument 1 must be an int, got float
screen_init() argument 1 must be an int, got str
ok
draw_pixel() argument 3 must be a color 0..0xFFFFFF, got -1
draw_pixel() argument 3 must be a color 0..0xFFFFFF, got 16777216
draw_pixel() argument 1 must be an int, got float
draw_pixel() argument 1 must be an int, got bool
draw_line() argument 3 must be -32768..32767, got 32768
draw_line() argument 1 must be -32768..32767, got -32769
ok
draw_rect() argument 4 must be an int, got nil
fill_rect() argument 3 must be an int, got array
draw_circle() argument 1 must be an int, got P
draw_circle() argument 3 must be -32768..32767, got 40000
draw_text() argument 4 must be an int, got str
draw_text() argument 1 must be -32768..32767, got 70000
ok
get_pixel() argument 1 must be an int, got float
screen_clear() argument 1 must be a color 0..0xFFFFFF, got 16777216
screen_clear() argument 1 must be an int, got map
ok
"""


def test_errors_identical_and_catchable():
    out, _ppm = both(ERRORS)
    assert out == ERRORS_OUT


@needs_cc
def test_uncaught_error_report_identical(tmp_path):
    src = 'draw_line(0, 0, 5, 5, 0xFF)\n'
    path = tmp_path / "e.tbc"
    path.write_bytes(tbc_bytes(src))
    r = cli("exec", str(path))
    crc, _out, cerr = run_cvm(tbc_bytes(src))
    assert r.returncode == crc == 1
    assert r.stderr == cerr
    assert cerr.startswith("runtime error: screen not initialized (call screen_init)\n")


def test_compile_time_arity():
    expect_compile_error('screen_init(10, 10)\ndraw_line(1, 2, 3, 4)\n', "'draw_line' takes 5 argument(s), got 4")
    expect_compile_error('screen_clear(1, 2)\n', "'screen_clear' takes 0 to 1 argument(s), got 2")


@pytest.mark.parametrize("call", ["draw_pixel(1, 1, 0)", "key_pressed()", "screen_width()"])
def test_screen_natives_are_io(call):
    src = f'#[pure]\nf() -> int\n    {call}\n    return 1\n\nmain()\n    print(f())\n'
    assert "io" in expect_compile_error(src, "E-EFFECT")


# ═══════════════════════════════════════════════════════════════
# COMMAND LINE: --screen, --screen-ansi, --keys, headless
# ═══════════════════════════════════════════════════════════════

ANIMATION = '''
screen_init(6, 5)
for i in 0..3
    screen_clear(0x000040)
    fill_rect(i, 1, 2, 2, 0xFF8000)
    draw_pixel(5, 4, 0x00FF00)
    screen_show()
print("done")
'''


def _write(tmp_path, text, name="g.til"):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return str(p)


def til_main(monkeypatch, capsys, *argv):
    """`til ARGV...` in this process (no interpreter start-up): (rc, out, err)."""
    monkeypatch.setattr(til.sys, 'argv', ['til', *argv])
    rc = til.main()
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def tilvm(*args, **kw):
    """The C VM runner itself: CompletedProcess with text output."""
    return subprocess.run([cvm_binary(), *args], capture_output=True, text=True, timeout=60,
                          stdin=subprocess.DEVNULL, **kw)


def test_headless_run(tmp_path):
    """No screen option: drawing goes to an offscreen framebuffer, screen_show()
    shows nothing, no files appear."""
    prog = _write(tmp_path, ANIMATION)
    r = cli("run", "--vm", prog, cwd=str(tmp_path))
    assert r.returncode == 0 and r.stdout == "done\n" and r.stderr == ""
    assert sorted(os.listdir(tmp_path)) == ["g.til"]


def test_screen_option_all_paths(tmp_path):
    prog = _write(tmp_path, LINES)
    tbc = str(tmp_path / "g.tbc")
    assert cli("build", "--target", "vm", prog, "-o", tbc).returncode == 0
    runs = [("run", "--vm", prog), ("exec", tbc)] + ([("run", "--cvm", prog)] if CC else [])
    images = []
    for k, args in enumerate(runs):
        out = str(tmp_path / f"o{k}.ppm")
        r = cli(*args, "--screen", out, "--keys", "")
        assert r.returncode == 0, (args, r.stderr)
        with open(out, 'rb') as f:
            images.append(f.read())
    assert len(set(images)) == 1
    assert images[0] == py_run(LINES)[2]


def test_screen_option_without_screen(tmp_path, monkeypatch, capsys):
    prog = _write(tmp_path, 'print("no screen")\n')
    out = str(tmp_path / "x.ppm")
    rc, stdout, err = til_main(monkeypatch, capsys, "run", "--vm", prog, "--screen", out)
    assert (rc, stdout) == (0, "no screen\n")
    assert "did not call screen_init (no image written)" in err
    if CC:
        tbc = tmp_path / "x.tbc"
        tbc.write_bytes(tbc_bytes('print("no screen")\n'))
        r = tilvm("--screen", out, str(tbc))
        assert (r.returncode, r.stdout) == (0, "no screen\n")
        assert "did not call screen_init (no image written)" in r.stderr
    assert not os.path.exists(out)


def test_screen_option_unwritable(tmp_path, monkeypatch, capsys):
    prog = _write(tmp_path, 'screen_init(2, 2)\n')
    out = str(tmp_path / "missing" / "x.ppm")
    rc, _stdout, err = til_main(monkeypatch, capsys, "run", "--vm", prog, "--screen", out)
    assert rc == 2 and "cannot write" in err
    if CC:
        tbc = tmp_path / "x.tbc"
        tbc.write_bytes(tbc_bytes('screen_init(2, 2)\n'))
        r = tilvm("--screen", out, str(tbc))
        assert r.returncode == 2 and "cannot write" in r.stderr


def test_screen_ansi(tmp_path, monkeypatch, capsys):
    """--screen-ansi: one frame of 24-bit colour half blocks per screen_show()."""
    prog = _write(tmp_path, ANIMATION)
    rc, text, err = til_main(monkeypatch, capsys, "run", "--vm", prog, "--screen-ansi", "--keys", "")
    assert rc == 0, err
    assert text.startswith("\x1b[2J\x1b[H") and text.endswith("done\n")
    assert text.count("\x1b[H") == 3 and text.count("\x1b[2J") == 1      # one frame per screen_show()
    assert text.count("▀") == 3 * 6 * 3                           # 6 columns x 3 rows of half blocks
    assert "\x1b[38;2;255;128;0m" in text and "\x1b[48;2;0;0;64m" in text
    assert "\x1b[49m" in text                                           # odd height: last half row
    if CC:
        tbc = tmp_path / "a.tbc"
        tbc.write_bytes(tbc_bytes(ANIMATION))
        r = tilvm("--screen-ansi", "--keys", "", str(tbc))
        assert r.returncode == 0 and r.stdout == text


def test_bad_keys_option(tmp_path, monkeypatch, capsys):
    prog = _write(tmp_path, 'print(key_pressed())\n')
    rc, _out, err = til_main(monkeypatch, capsys, "run", "--vm", prog, "--keys", "UP,JUMP")
    assert rc == 2 and "bad --keys entry 'JUMP'" in err
    if CC:
        tbc = tmp_path / "k.tbc"
        tbc.write_bytes(tbc_bytes('print(key_pressed())\n'))
        r = tilvm("--keys", "UP,JUMP", str(tbc))
        assert r.returncode == 2 and "bad --keys entry 'JUMP'" in r.stderr
    # graphics options make `til run` use the VM (the native backend has no screen)
    rc, out, err = til_main(monkeypatch, capsys, "run", prog, "--keys", "UP,x")
    assert (rc, out) == (0, "1\n") and "native" not in err


# ═══════════════════════════════════════════════════════════════
# THE EXAMPLE
# ═══════════════════════════════════════════════════════════════

EXAMPLE = os.path.join(ROOT, 'examples', '11_graphics.til')
EXAMPLE_OUT = ("screen 96x64\nsun edge  ffd23f\ndoor      000000\nwindow    a8dadc\noutside   0\n"
               "checksum  440b5111\n")


def test_example_identical_on_both_vms():
    with open(EXAMPLE, encoding='utf-8') as f:
        src = f.read()
    out, ppm = both(src)
    assert out == EXAMPLE_OUT
    assert pixels(ppm)[:2] == (96, 64)


# ═══════════════════════════════════════════════════════════════
# BUILDS: lean 16-bit (RGB565), int32, no graphics, custom hosts, snapshots
# ═══════════════════════════════════════════════════════════════

def rgb565_program(red, green, blue, white):
    return f'''
let RED = {red}
let GREEN = {green}
let BLUE = {blue}
let WHITE = {white}
screen_init(40, 24)
screen_clear(BLUE)
draw_line(0, 0, 39, 23, RED)
draw_rect(2, 2, 12, 8, GREEN)
fill_rect(30, 2, 6, 6, WHITE)
draw_circle(20, 12, 7, GREEN)
draw_text(1, 15, "Hi 16!", WHITE)
print(get_pixel(0, 0) == RED, get_pixel(5, 12) == BLUE, get_pixel(39, 23))
'''


@needs_cc
def test_lean_16bit_build_uses_rgb565():
    """TILVM_LEAN keeps the graphics natives. 16-bit ints cannot hold 0xRRGGBB,
    so colours are RGB565 there: the same picture as the 24-bit program."""
    big = rgb565_program(0xFF0000, 0x00FF00, 0x0000FF, 0xFFFFFF)
    small = rgb565_program(-2048, 2016, 31, -1)          # 0xF800 0x07E0 0x001F 0xFFFF
    rc, out, ppm = py_run(big)
    assert rc == 0 and out == b"true\ntrue\n16711680\n"
    crc, cout, cerr, cppm = c_run(small, profile="tiny")
    assert crc == 0, cerr
    assert cout == b"true\ntrue\n-2048\n"
    assert cppm == ppm
    crc, _cout, cerr, _ = c_run('screen_init(4097, 1)\n', profile="tiny")
    assert crc == 1 and "screen_init() argument 1 must be 1..4096, got 4097" in cerr


def expand565(v):
    v &= 0xFFFF
    r, g, b = v >> 11, (v >> 5) & 63, v & 31
    return bytes(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)))


@needs_cc
def test_lean_rgb565_round_trip():
    values = [0, -1, 31, 2016, -2048, 12345, -12345, 32767, -32768, 1]
    src = f"let cs = {values}\nscreen_init({len(values)}, 1)\nvar x = 0\nfor c in cs\n" \
          "    draw_pixel(x, 0, c)\n    x += 1\nvar back = []\nfor i in 0..len(cs)\n" \
          "    back.push(get_pixel(i, 0))\nprint(back == cs)\n"
    crc, cout, cerr, cppm = c_run(src, profile="tiny")
    assert (crc, cout) == (0, b"true\n"), cerr
    assert cppm == b"P6\n%d 1\n255\n" % len(values) + b"".join(expand565(v) for v in values)


@needs_cc
def test_lean_16bit_keys_and_text():
    crc, cout, cerr, cppm = c_run('screen_init(30, 9)\ndraw_text(0, 1, 12345, -1)\n' + KEYS_PROGRAM,
                                  keys="UP,NONE,DOWN,NONE,NONE,a,ESC", profile="tiny")
    assert crc == 0, cerr
    assert cout == b"[1, 0, 2, 0, 97, 27, 0, 27]\n"
    assert cppm == py_run('screen_init(30, 9)\ndraw_text(0, 1, "12345", 0xFFFFFF)\n')[2]


@needs_cc
def test_no_gfx_build_reports_not_available():
    exe = quick_vm('-DTILVM_NO_GFX')
    for name in ("screen_width", "key_pressed"):
        crc, _out, cerr, _ = c_run(f'print({name}())\n', exe=exe)
        assert crc == 1 and f"{name}() is not available in this VM build" in cerr


@needs_cc
def test_gc_stress_text_conversion():
    """draw_text converts non-strings on the VM heap (may collect garbage)."""
    src = """
screen_init(64, 8)
var junk = []
for i in 0..300
    junk.push([i, str(i)])
    draw_text(0, 0, [i, "x"], 0xFFFFFF)
    if len(junk) > 20
        junk = []
print(get_pixel(1, 1))
"""
    both(src, profile="gcstress")


# A device host made from `til build --target c-vm`: "none" installs no screen or
# key hooks, "partial" only gfx_pixel + key_pressed + idle (no gfx_init, gfx_get,
# gfx_show, wait_key).
DEVICE_HOST = r"""
#define TILVM_NO_MAIN 1
#include "prog_vm.c"
#include <stdio.h>
#include <string.h>
static unsigned long px[4][6];
static int polls;
static void pixel(void *ctx, int x, int y, unsigned long rgb) { (void)ctx; px[y][x] = rgb; }
static int pressed(void *ctx) { (void)ctx; return ++polls < 4 ? 0 : 'k'; }
static void idle(void *ctx, unsigned long ms) { (void)ctx; (void)ms; }
int main(int argc, char **argv) {
    static tilvm vm;
    int x, y, rc;
    for (y = 0; y < 4; y++) for (x = 0; x < 6; x++) px[y][x] = 0xABCDEFul;
    if (tilvm_load(&vm, til_program, (uint32_t)til_program_size) != 0) return 9;
    if (argc > 1 && !strcmp(argv[1], "partial")) {
        vm.host.gfx_pixel = pixel;
        vm.host.key_pressed = pressed;
        vm.host.idle = idle;
    }
    rc = tilvm_run(&vm);
    for (y = 0; y < 4; y++) { for (x = 0; x < 6; x++) printf("%06lx ", px[y][x]); printf("\n"); }
    printf("polls=%d rc=%d\n", polls, rc);
    return 0;
}
"""

DEVICE_PROGRAM = """
attempt(f)
    let r = retry(1)
        f()
    else
        error
    print(r)

print(screen_width())
attempt(|| screen_init(5, 3))
if screen_width() > 0
    draw_line(0, 0, 4, 2, 0x112233)
    screen_show()
attempt(|| wait_key())
attempt(|| get_pixel(0, 0))
"""


@needs_cc
def test_device_hosts_with_missing_hooks(tmp_path):
    """No hooks: the screen/keys natives are 'not available on this device'.
    Without gfx_init the VM clears the screen itself; without wait_key it polls
    key_pressed (calling idle); without gfx_get, get_pixel is not available."""
    prog = _write(tmp_path, DEVICE_PROGRAM)
    r = cli("build", "--target", "c-vm", prog, "-o", str(tmp_path / "prog_vm.c"))
    assert r.returncode == 0, r.stderr
    (tmp_path / "host.c").write_text(DEVICE_HOST)
    exe = str(tmp_path / "host")
    b = subprocess.run([CC, *STRICT, '-O0', '-I', str(tmp_path), '-o', exe, str(tmp_path / "host.c"), '-lm'],
                       capture_output=True, text=True)
    assert b.returncode == 0, b.stderr
    untouched = ["abcdef " * 6] * 4
    r = subprocess.run([exe, "none"], capture_output=True, text=True, timeout=30)
    assert r.stdout.splitlines() == [
        "0", "screen_init() is not available on this device", "wait_key() is not available on this device",
        "get_pixel() is not available on this device", *untouched, "polls=0 rc=0"]
    r = subprocess.run([exe, "partial"], capture_output=True, text=True, timeout=30)
    assert r.stdout.splitlines() == [
        "0", "None", "107", "get_pixel() is not available on this device",
        "112233 000000 000000 000000 000000 abcdef ",
        "000000 112233 112233 000000 000000 abcdef ",
        "000000 000000 000000 112233 112233 abcdef ",
        "abcdef abcdef abcdef abcdef abcdef abcdef ", "polls=4 rc=0"]


@needs_cc
def test_snapshot_keeps_the_screen_size(tmp_path):
    """A raw snapshot keeps the screen size; the host screen is re-created on
    resume, and a program that redraws every frame ends with the same picture."""
    src = '''
screen_init(24, 10)
var frame = 0
while frame < 200
    screen_clear(0x000010)
    fill_rect(frame % 24, 3, 4, 4, 0xFF8800)
    draw_text(0, 0, str(frame), 0xFFFFFF)
    frame += 1
print(frame, screen_width(), screen_height())
'''
    tbc = tmp_path / "s.tbc"
    tbc.write_bytes(tbc_bytes(src))
    exe = cvm_binary()
    full = subprocess.run([exe, '--screen', str(tmp_path / "full.ppm"), str(tbc)], capture_output=True)
    assert full.returncode == 0 and full.stdout == b"200\n24\n10\n"
    snap = str(tmp_path / "s.snap")
    first = subprocess.run([exe, '--snapshot-after', '80', snap, '--screen', str(tmp_path / "half.ppm"),
                            str(tbc)], capture_output=True)
    assert first.returncode == 3, first.stderr
    second = subprocess.run([exe, '--resume', snap, '--screen', str(tmp_path / "end.ppm"), str(tbc)],
                            capture_output=True)
    assert second.returncode == 0 and second.stdout == b"200\n24\n10\n", second.stderr
    assert (tmp_path / "end.ppm").read_bytes() == (tmp_path / "full.ppm").read_bytes()
    assert (tmp_path / "half.ppm").read_bytes() != (tmp_path / "full.ppm").read_bytes()


# ═══════════════════════════════════════════════════════════════
# THE DOCUMENTATION'S EXAMPLES RUN AS WRITTEN
# ═══════════════════════════════════════════════════════════════

DOC = open(os.path.join(ROOT, 'docs', 'TIL_GRAPHICS.md'), encoding='utf-8').read()


def doc_block(lang, marker):
    blocks = re.findall(r'```' + lang + r'\n(.*?)```', DOC, re.S)
    found = [b for b in blocks if marker in b]
    assert len(found) == 1, marker
    return found[0]


def test_doc_dot_demo_with_scripted_keys():
    src = doc_block('til', '# move a dot with the arrow keys')
    out, ppm = both(src, keys="RIGHT,RIGHT,DOWN,SPACE,LEFT,ESC")
    assert out == "5 keys, dot at 17,13\n"
    w, h, px = pixels(ppm)
    assert (w, h) == (32, 24) and px[(17, 13)] == 0xFF4040 and px[(16, 12)] == 0xFFFFFF
    assert both(src)[0] == "0 keys, dot at 16,12\n"          # no keys: wait_key() -> ESC


def test_doc_opening_example():
    src = doc_block('til', 'screen_init(96, 64)                 # open a 96 x 64')
    assert both(src)[0] == "11066076\n"


DOC_HOST_MAIN = r"""
#define TILVM_NO_MAIN 1
#include "prog_vm.c"
#include <stdio.h>
%s
int main(void) {
    static tilvm vm;
    int y;
    if (tilvm_load(&vm, til_program, (uint32_t)til_program_size) != 0) return 9;
    install_screen(&vm);
    if (tilvm_run(&vm) != 0) return 8;
    printf("P6\n%%d %%d\n255\n", (int)vm.st.scr_w, (int)vm.st.scr_h);
    for (y = 0; y < vm.st.scr_h; y++) fwrite(screen_rgb[y], 3, vm.st.scr_w, stdout);
    return 0;
}
"""


@needs_cc
def test_doc_host_example_draws_the_same_pixels(tmp_path):
    """The C host of docs/TIL_GRAPHICS.md, built into a `til build --target c-vm`
    program, gives the reference VM's picture."""
    host = doc_block('c', 'host example (tested by tests/test_graphics.py)')
    prog = _write(tmp_path, SHAPES)
    assert cli("build", "--target", "c-vm", prog, "-o", str(tmp_path / "prog_vm.c")).returncode == 0
    (tmp_path / "main.c").write_text(DOC_HOST_MAIN % host)
    exe = str(tmp_path / "doc_host")
    b = subprocess.run([CC, *STRICT, '-O0', '-I', str(tmp_path), '-I', VM_DIR, '-o', exe,
                        str(tmp_path / "main.c"), '-lm'], capture_output=True, text=True)
    assert b.returncode == 0, b.stderr
    r = subprocess.run([exe], capture_output=True, timeout=30)
    assert r.returncode == 0
    assert r.stdout == py_run(SHAPES)[2]


# ═══════════════════════════════════════════════════════════════
# REAL TERMINAL (opt-in: timing-sensitive, POSIX only)
# ═══════════════════════════════════════════════════════════════

TTY_PROGRAM = '''
var got = []
for i in 0..4
    got.push(wait_key())
got.push(key_pressed())
print(got)
print(input("name? "))
print(wait_key())
'''


@pytest.mark.skipif(not os.environ.get('TIL_TEST_TTY') or os.name != 'posix', reason="set TIL_TEST_TTY=1")
def test_terminal_keys_in_a_pty(tmp_path):
    """Raw mode in a pseudo-terminal: arrows, characters, Enter and a lone ESC;
    no echo while keys are read, a normal line for input(), the terminal mode
    restored at exit (checked with stty in the same terminal)."""
    import pty
    import select
    import time
    prog = _write(tmp_path, TTY_PROGRAM)
    tbc = tmp_path / "t.tbc"
    tbc.write_bytes(tbc_bytes(TTY_PROGRAM))
    stty = "stty -a | tr ' ' '\\n' | grep -E '^-?(icanon|echo)$' | tr '\\n' ' '"
    commands = [f"{sys.executable} {os.path.join(ROOT, 'src', 'til.py')} run --vm {prog}"]
    if CC:
        commands.append(f"{cvm_binary()} {tbc}")
    for command in commands:
        pid, fd = pty.fork()
        if pid == 0:
            os.execvp('sh', ['sh', '-c', f"{command}; {stty}"])
        out = b""

        def pump(seconds):
            nonlocal out
            end = time.time() + seconds
            while time.time() < end:
                if select.select([fd], [], [], 0.02)[0]:
                    try:
                        data = os.read(fd, 4096)
                    except OSError:
                        return
                    if not data:
                        return
                    out += data
        pump(2.0)
        for chunk in (b"\x1b[A", b"x", b"\r", b"\x1b", b"Aigerim\r", b"q"):
            os.write(fd, chunk)
            pump(0.3)
        pump(1.5)
        os.waitpid(pid, 0)
        text = out.decode('utf-8', 'replace').replace("\r\n", "\n")
        assert "[1, 120, 13, 27, 0]\nname? Aigerim\nAigerim\n113\n" in text, text
        assert text.rstrip().endswith("icanon echo"), text
