#!/usr/bin/env python3
"""
TIL 3.1 feature 5: the online playground (website/playground/, design §5).

  * the committed build (tilvm.wasm, build-info.json, tilc.tbc or the
    precompiled examples) matches its sources and is reproducible;
  * the WebAssembly VM (unmodified vm/tilvm.c + wasm_host.c, run by Node with
    website/playground/test/tilvm_node.js) prints, fails, exits and counts
    steps exactly like the desktop C VM;
  * input(), sleep_ms() and wait_key() pause the program without blocking the
    JavaScript event loop, and a VM build that cannot pause is reported
    explicitly;
  * the page itself (website/playground/test/playground.spec.mjs: Node +
    Playwright + Chromium): hello world, runtime error line, Stop, input,
    share links, canvas, keys, and the compile pipeline with a stand-in for
    tilc.tbc.

Needs node (+ Playwright and Chromium for the browser tests) and clang with
the wasm32 target (for fresh builds); the tests skip what is not installed.

Run: python3 -m pytest tests/test_playground.py -v
     TIL_SLOW_TESTS=1 python3 -m pytest tests/test_playground.py   (+ the tests that need extra builds)
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from tilhelpers import CC, ROOT, cvm_binary, til

PG = os.path.join(ROOT, 'website', 'playground')
BUILD_SH = os.path.join(PG, 'build.sh')
TILVM_JS = os.path.join(PG, 'tilvm.js')
NODE_RUNNER = os.path.join(PG, 'test', 'tilvm_node.js')
SPEC = os.path.join(PG, 'test', 'playground.spec.mjs')
MOCK_GFX = os.path.join(PG, 'test', 'mock_gfx.inc.c')
MOCK_PROGRAMS = os.path.join(PG, 'test', 'mock_programs.json')
VM_C = os.path.join(ROOT, 'vm', 'tilvm.c')
NODE = shutil.which('node')
CLANG = shutil.which('clang')
BASH = shutil.which('bash')

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
# tests that need an extra WebAssembly build of their own (opt-in)
slow = pytest.mark.skipif(not os.environ.get('TIL_SLOW_TESTS'), reason="slow: set TIL_SLOW_TESTS=1")


def _probe(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


_WASM_OK = None


def wasm_toolchain() -> bool:
    """clang can compile and link a freestanding wasm32 module (wasm-ld)."""
    global _WASM_OK
    if _WASM_OK is None:
        _WASM_OK = False
        if CLANG and BASH:
            d = tempfile.mkdtemp(prefix='tilwasm-probe-')
            src = os.path.join(d, 'p.c')
            with open(src, 'w') as f:
                f.write('__attribute__((export_name("f"))) int f(int x) { return x + 1; }\n')
            _WASM_OK = _probe([CLANG, '--target=wasm32', '-nostdlib', '-Wl,--no-entry', '-o',
                               os.path.join(d, 'p.wasm'), src])
            shutil.rmtree(d, ignore_errors=True)
    return _WASM_OK


def has_playwright() -> bool:
    return NODE is not None and _probe([NODE, '-e', "require('playwright')"], cwd=tempfile.gettempdir())


def has_real_graphics(vm_source: str) -> bool:
    return re.search(r'case\s+N_SCREEN_INIT\s*:', vm_source) is not None


def sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ═══════════════════════════════════════════════════════════════
# BUILDS (fresh ones go to a temporary directory, never into the tree)
# ═══════════════════════════════════════════════════════════════

def run_build(out_dir: str, tilvm_c: str = None, tilc_til: str = None, examples: bool = True):
    env = dict(os.environ, OUT=out_dir, PYTHON=sys.executable, EXAMPLES='1' if examples else '0')
    if tilvm_c:
        env['TILVM_C'] = tilvm_c
    if tilc_til:
        env['TILC_TIL'] = tilc_til
    r = subprocess.run([BASH, BUILD_SH], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, f"build.sh failed:\n{r.stdout}\n{r.stderr}"
    return r


def add_mock_graphics(src: str) -> str:
    """TEST ONLY: paste the minimal screen/key natives of mock_gfx.inc.c into a
    copy of vm/tilvm.c that does not implement natives 40-53 yet."""
    with open(MOCK_GFX, encoding='utf-8') as f:
        mock = f.read()
    anchor = '/* args are STK[base .. base+argc); the result replaces them */\nstatic int call_native('
    default = ('    default:\n        return failn(vm, id < N_NATIVES ? NATIVE_NAMES[id] : "?", '
               '"() is not available in this VM build",')
    if anchor not in src or default not in src:
        return None
    cases = ('    case N_SCREEN_INIT: case N_SCREEN_CLEAR: case N_DRAW_PIXEL: case N_DRAW_LINE: case N_DRAW_RECT:\n'
             '    case N_FILL_RECT: case N_DRAW_CIRCLE: case N_DRAW_TEXT: case N_GET_PIXEL: case N_SCREEN_SHOW:\n'
             '    case N_KEY_PRESSED: case N_WAIT_KEY: case N_SCREEN_WIDTH: case N_SCREEN_HEIGHT:\n'
             '        if (!mock_gfx(vm, id, base, argc, &res)) return 0;\n        break;\n')
    return src.replace(anchor, mock + '\n' + anchor, 1).replace(default, cases + default, 1)


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    """Fresh builds from the current sources (or the committed one without a
    wasm toolchain). Returns a dict:
       wasm       tilvm.wasm of the unmodified vm/tilvm.c
       site       a copy of website/ whose playground has a build with graphics
                  (the real natives, or the test mock while they do not exist)
       site_wasm  the site's tilvm.wasm (fresh builds only)
       graphics   'real' | 'mock' | ''   (what the site's VM can draw)
       fresh      True when built here"""
    tmp = tmp_path_factory.mktemp('playground')
    with open(VM_C, encoding='utf-8') as f:
        vm_source = f.read()
    real = has_real_graphics(vm_source)
    site = str(tmp / 'site')
    shutil.copytree(os.path.join(ROOT, 'website'), site)
    if not wasm_toolchain():
        if not committed_build_current():
            pytest.skip("no clang wasm32 toolchain and the committed tilvm.wasm is stale")
        return {'wasm': os.path.join(PG, 'tilvm.wasm'), 'site': site,
                'graphics': 'real' if real else '', 'fresh': False}
    site_wasm = os.path.join(site, 'playground', 'tilvm.wasm')
    result = {'wasm': site_wasm, 'site': site, 'site_wasm': site_wasm, 'fresh': True}
    patched = None if real else add_mock_graphics(vm_source)
    if patched is None:                     # one build serves everything
        run_build(os.path.join(site, 'playground'))
        result['graphics'] = 'real' if real else ''
    else:                                   # the site gets the test mock; the VM tests use the plain VM
        plain = str(tmp / 'plain')
        run_build(plain, examples=False)
        result['wasm'] = os.path.join(plain, 'tilvm.wasm')
        mock_c = str(tmp / 'tilvm_mock_graphics.c')
        with open(mock_c, 'w', encoding='utf-8') as f:
            f.write(patched)
        run_build(os.path.join(site, 'playground'), tilvm_c=mock_c)
        result['graphics'] = 'mock'
    return result


# ═══════════════════════════════════════════════════════════════
# THE COMMITTED BUILD
# ═══════════════════════════════════════════════════════════════

def build_info():
    with open(os.path.join(PG, 'build-info.json'), encoding='utf-8') as f:
        return json.load(f)


def expected_inputs():
    inputs = {'vm/tilvm.c': sha256(VM_C), 'vm/tilvm.h': sha256(os.path.join(ROOT, 'vm', 'tilvm.h'))}
    for rel in ('website/playground/wasm_host.c', 'website/playground/shim/libc.c',
                'website/playground/shim/string.h', 'website/playground/shim/math.h',
                'website/playground/build.sh'):
        inputs[rel] = sha256(os.path.join(ROOT, rel))
    tilc = os.path.join(ROOT, 'selfhost', 'tilc.til')
    if os.path.exists(tilc):
        inputs['selfhost/tilc.til'] = sha256(tilc)
    return inputs


def committed_build_current() -> bool:
    try:
        info = build_info()
    except OSError:
        return False
    return info.get('inputs') == expected_inputs() and 'vm_source' not in info and 'tilc_source' not in info


STALE = ("website/playground/{what} is out of date with its sources: run website/playground/build.sh "
         "and commit tilvm.wasm, build-info.json, tilc.tbc (when built) and examples/*.tbc")


def test_committed_build_matches_sources():
    info = build_info()
    assert 'vm_source' not in info and 'tilc_source' not in info, "built from a non-default source"
    assert info['inputs'] == expected_inputs(), STALE.format(what='tilvm.wasm')
    assert info['outputs']['tilvm.wasm'] == sha256(os.path.join(PG, 'tilvm.wasm')), \
        STALE.format(what='tilvm.wasm')


def test_committed_compiler_or_fallback_examples():
    info = build_info()
    examples = sorted(f[:-4] for f in os.listdir(os.path.join(PG, 'examples')) if f.endswith('.til'))
    with open(os.path.join(PG, 'examples', 'index.json'), encoding='utf-8') as f:
        assert sorted(e['id'] for e in json.load(f)) == examples
    tbc = sorted(f[:-4] for f in os.listdir(os.path.join(PG, 'examples')) if f.endswith('.tbc'))
    if os.path.exists(os.path.join(ROOT, 'selfhost', 'tilc.til')):
        assert info['compiler'] == 'tilc', STALE.format(what='tilc.tbc')
        assert info['outputs'].get('tilc.tbc') == sha256(os.path.join(PG, 'tilc.tbc')), STALE.format(what='tilc.tbc')
        assert tbc == [], "examples/*.tbc are only a fallback while there is no tilc.tbc"
    else:
        assert info['compiler'] == 'precompiled-examples'
        assert not os.path.exists(os.path.join(PG, 'tilc.tbc'))
        assert tbc == examples, STALE.format(what='examples/*.tbc')
        for name in examples:
            with open(os.path.join(PG, 'examples', name + '.til'), encoding='utf-8') as f:
                src = f.read()
            want = til.compile_to_bytecode(src, os.path.join(PG, 'examples', name + '.til')).serialize()
            with open(os.path.join(PG, 'examples', name + '.tbc'), 'rb') as f:
                assert f.read() == want, STALE.format(what=f'examples/{name}.tbc')


def test_build_is_reproducible(builds):
    if not builds['fresh']:
        pytest.skip("no wasm toolchain")
    info = build_info()
    version = subprocess.run([CLANG, '--version'], capture_output=True, text=True).stdout.splitlines()[0]
    if version != info['clang'] or not committed_build_current():
        pytest.skip(f"committed build made with {info['clang']!r} or stale; this machine has {version!r}")
    assert sha256(builds['wasm']) == info['outputs']['tilvm.wasm']


def test_uses_the_unmodified_vm_source():
    with open(BUILD_SH, encoding='utf-8') as f:
        script = f.read()
    assert 'TILVM_C="${TILVM_C:-$ROOT/vm/tilvm.c}"' in script and '-DTILVM_NO_STDIO' in script
    assert '--target=wasm32' in script and '-nostdlib' in script
    assert not os.path.exists(os.path.join(PG, 'tilvm.c')), "no forked copy of the VM"


# ═══════════════════════════════════════════════════════════════
# WEBASSEMBLY VM == DESKTOP C VM
# ═══════════════════════════════════════════════════════════════

def node_run(wasm, tbc_path, args=(), stdin='', keys=None, timeout=60):
    cmd = [NODE, NODE_RUNNER, '--wasm', wasm, '--stats']
    if keys:
        cmd += ['--keys', keys]
    r = subprocess.run(cmd + [tbc_path, *args], input=stdin.encode('utf-8'), capture_output=True, timeout=timeout)
    err = r.stderr.decode('utf-8', 'replace')
    m = re.search(r'\[tilvm_node\] status=(\S+) steps=(\d+)', err)
    err = re.sub(r'\[tilvm_node\][^\n]*\n', '', err)
    return r.returncode, r.stdout.decode('utf-8', 'replace'), err, int(m.group(2)) if m else None


def desktop_run(tbc_path, args=(), stdin='', timeout=60):
    r = subprocess.run([cvm_binary(), '--stats', tbc_path, *args], input=stdin.encode('utf-8'),
                       capture_output=True, timeout=timeout)
    err = r.stderr.decode('utf-8', 'replace')
    m = re.search(r'\[tilvm\] steps=(\d+)', err)
    err = re.sub(r'\[tilvm\][^\n]*\n', '', err)
    return r.returncode, r.stdout.decode('utf-8', 'replace'), err, int(m.group(1)) if m else None


def compile_file(src: str, path: str, name: str, filename: str = None) -> str:
    out = os.path.join(path, name + '.tbc')
    with open(out, 'wb') as f:
        f.write(til.compile_to_bytecode(src, filename or name + '.til').serialize())
    return out


DIFF_PROGRAMS = {
    'closures': '''
make_adder(k: int)
    return |x: int| x + k
let add5 = make_adder(5)
let nums = [5, 3, 8, 1, 9, 2]
print(nums.map(add5))
print(nums.filter(|n| n % 2 == 1))
print(nums.reduce(|a, b| a + b, 0))
var sorted = nums.copy()
sorted.sort()
print(sorted)
print(nums.any(|n| n > 8), nums.all(|n| n > 0))
''',
    'strings': '''
let s = "  Hello, TIL World  "
print(s.trim().to_upper(), s.trim().to_lower())
print(s.trim().split(", "))
print("abc".repeat(3), "banana".replace("an", "AN"), "banana".count("a"))
print("banana".find("na"), "banana".slice(1, 4), "x".is_empty(), "".is_empty())
print(len("сәлем"), "сәлем".to_upper(), str_bytes("hi"), bytes_to_str([84, 73, 76]))
print(chr(1199), ord("ә"), "42".to_int() + 1, "2.5".to_float() * 2)
let n = 255
print(f"[{n:>6}] [{n:<6}] [{n:^7}] [{n:06d}] [{n:x}] [{n:X}] [{n:o}] [{n:b}]")
print(f"{3.14159:.2f} {2.5:.1e} {0.25:.0%} {1234.5:>10.1f}")
''',
    'maps_and_structs': '''
struct Item
    name: str
    qty: int
var stock = {"apple": 3, "pear": 0}
stock["plum"] = 7
stock.delete("pear")
print(stock.keys(), stock.values(), stock.has("pear"), stock.get_or("kiwi", -1))
let items = [Item { name: "apple", qty: 3 }, Item { name: "plum", qty: 7 }]
var total = 0
for it in items
    total += it.qty
print(total, items[1].name, items)
''',
    'enums_options': '''
enum Shape
    Circle(float)
    Rect(float, float)
area(s: Shape) -> float
    return match s
        Shape.Circle(r) => 3.0 * r * r
        Shape.Rect(w, h) => w * h
print(area(Shape.Circle(2.0)), area(Shape.Rect(2.0, 3.5)))
parse(s: str) -> Result<int, str>
    if s == ""
        return Err("empty")
    return Ok(s.to_int())
for x in ["12", ""]
    match parse(x)
        Ok(v) => print(f"ok {v}")
        Err(e) => print(f"err {e}")
''',
    'retry': '''
var tries = 0
flaky() -> int
    tries += 1
    assert tries >= 3, "warming up"
    return 42
let v = retry(5)
    flaky()
print(v, tries)
let r = retry(1)
    [1, 2][5]
else
    "recovered: " + error
print(r)
''',
    'gc_stress': '''
var keep = []
for i in 0..3000
    let s = f"item-{i}-" + "x".repeat(i % 50)
    if i % 100 == 0
        keep.push(s)
    var tmp = [i, i * 2, s]
    tmp.push(len(s))
print(len(keep), keep[29])
''',
    'deep_recursion': '''
down(n: int) -> int
    return down(n + 1) + 1
print("start")
print(down(0))
''',
    'exit_code': 'print("bye")\nexit(7)\nprint("never")\n',
    'args_and_files': '''
let a = args()
print(len(a), a[1], a[2])
let path = a[0]
write_file(path, "line 1\\nсәлем\\n")
print(file_exists(path), read_file(path).split("\\n"))
print(file_exists(path + ".missing"))
''',
    'input_lines': '''
let first = input("first? ")
print(f"[{first}]")
let long = input()
print(len(long))
let rest = input()
print(len(rest))
let after_eof = input("more? ")
print(f"<{after_eof}>")
''',
    'floats_common': '''
print(3.14, 0.1 + 0.2, 2.0 ** 10, sqrt(2.0), 1.0 / 3.0, 10.0 / 4.0)
print(sin(1.0), cos(0.5), exp(1.0), log(10.0), floor(-2.5), ceil(2.1), round(2.4))
print(int(7.9), float(3), 1e10, 1.5e-7, -0.75, 100.0 * 1.1)
print(pow(2.0, 0.5), abs(-3.5), min(2.5, 1.5), max([1.0, 9.5, 3.0]))
''',
    'sleep': 'print("a")\nsleep_ms(30)\nprint("b")\n',
    'runtime_error_trace': '''
inner(xs) -> int
    return xs[10]
outer() -> int
    return inner([1, 2, 3]) + 1
print("before")
print(outer())
''',
}


def diff_corpus():
    """(name, source, filename) for the differential test: targeted programs,
    the playground's examples (those without randomness, input or a screen)
    and the repository's examples."""
    corpus = []
    for name in sorted(DIFF_PROGRAMS):
        corpus.append((name, DIFF_PROGRAMS[name], None))
    files = [('example_' + n, os.path.join(PG, 'examples', n + '.til'))
             for n in ('hello', 'basics', 'structs', 'kazakh', 'russian', 'errors')]
    examples = os.path.join(ROOT, 'examples')
    files += [('repo_' + f[:-4], os.path.join(examples, f)) for f in sorted(os.listdir(examples))
              if f.endswith('.til') and f != 'math_utils.til']
    for name, path in files:
        with open(path, encoding='utf-8') as f:
            corpus.append((name, f.read(), path))
    return corpus


CORPUS = diff_corpus()


@needs_node
@pytest.mark.skipif(CC is None, reason="no C compiler for the desktop VM")
@pytest.mark.parametrize("name,source,filename", CORPUS, ids=[c[0] for c in CORPUS])
def test_wasm_vm_matches_desktop_vm(builds, tmp_path, name, source, filename):
    try:
        tbc = compile_file(source, str(tmp_path), name, filename)
    except (til.CompileError, SyntaxError) as e:
        pytest.skip(f"not a VM program: {str(e).splitlines()[0]}")
    args, stdin = (), ''
    if name == 'args_and_files':
        args = (str(tmp_path / 'out.txt'), 'b c', 'сәлем')
    if name == 'input_lines':
        stdin = 'Aigerim\n' + 'x' * 300 + '\n'
    wrc, wout, werr, wsteps = node_run(builds['wasm'], tbc, args, stdin)
    if name == 'args_and_files':
        os.unlink(str(tmp_path / 'out.txt'))
    drc, dout, derr, dsteps = desktop_run(tbc, args, stdin)
    assert (wrc, wout, werr) == (drc, dout, derr), f"wasm:\n{wout}{werr}\ndesktop:\n{dout}{derr}"
    if 'timeout(' not in source and 'clock_ms(' not in source:
        assert wsteps == dsteps, f"step counts differ: wasm {wsteps}, desktop {dsteps}"


# Known differences of the freestanding float formatting in vm/tilvm.c
# (#ifdef TILVM_NO_STDIO: fmt_gp/fmt_fixed/fmt_exp, also used by calculator
# builds) from printf: exact ties, near-ties, -0.0 and huge fixed-point
# numbers. Not a playground bug; tracked here so a fix shows up as XPASS.
@needs_node
@pytest.mark.skipif(CC is None, reason="no C compiler for the desktop VM")
@pytest.mark.xfail(reason="TILVM_NO_STDIO float formatting in vm/tilvm.c differs from printf "
                          "(see website/playground/README.md)", strict=False)
def test_float_formatting_matches_printf(builds, tmp_path):
    src = '''
print(123456.5, 999999.5, 65081.35, -0.0)
print(f"{0.005:.2f} {-0.015:.2f} {1e100:.3f}")
'''
    tbc = compile_file(src, str(tmp_path), 'floats')
    wasm = node_run(builds['wasm'], tbc)[:3]
    desktop = desktop_run(tbc)[:3]
    assert wasm == desktop


# ═══════════════════════════════════════════════════════════════
# PAUSING WITHOUT BLOCKING (tilvm.js driven directly)
# ═══════════════════════════════════════════════════════════════

PAUSE_DRIVER = r'''
const fs = require('fs');
const T = require(process.argv[2]);
(async () => {
    const rt = await T.Runtime.create(fs.readFileSync(process.argv[3]));
    const vm = rt.createVM();
    let out = '';
    vm.onOutput = (t) => { out += t; };
    vm.load(fs.readFileSync(process.argv[4]));
    vm.setArgs([]);
    const log = [];
    const step = () => {
        const t0 = performance.now();
        const st = vm.run(1000000);
        log.push({ st: T.STATUS_NAMES[st], ms: performance.now() - t0, out, sleep: vm.sleepMs });
        out = '';
        return st;
    };
    for (const action of JSON.parse(process.argv[5])) {
        if (action.key) vm.pushKey(action.key);
        if (action.line !== undefined) vm.pushLine(action.line);
        if (action.wait) await new Promise((r) => setTimeout(r, action.wait));
        step();
    }
    console.log(JSON.stringify(log));
})().catch((e) => { console.error(e); process.exit(1); });
'''


def drive(wasm, src, actions, tmp_path):
    tbc = compile_file(src, str(tmp_path), 'paused')
    driver = tmp_path / 'driver.js'
    driver.write_text(PAUSE_DRIVER)
    r = subprocess.run([NODE, str(driver), TILVM_JS, wasm, tbc, json.dumps(actions)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@needs_node
def test_input_pauses_and_prints_the_prompt_once(builds, tmp_path):
    src = ('let a = input("Name? ")\nprint(f"Hi {a}")\ninput("Press Enter ")\n'
           'let b = input("Age? ")\nprint(b)\n')
    log = drive(builds['wasm'], src, [{}, {'line': 'Aigerim'}, {'line': ''}, {}, {'line': '12'}], tmp_path)
    assert [e['st'] for e in log] == ['wait-line', 'wait-line', 'wait-line', 'wait-line', 'done']
    assert ''.join(e['out'] for e in log) == 'Name? Hi Aigerim\nPress Enter Age? 12\n'
    assert log[0]['ms'] < 1000


@needs_node
def test_timeouts_after_a_sleep_are_not_sleeps(builds, tmp_path):
    """Clock reads of timeout() setup in a frame whose last native was
    sleep_ms() must not be taken for that sleep."""
    src = ('f() -> int\n    sleep_ms(5)\n    let r = 2500 + timeout(1000)\n        10\n    return r\n'
           'print(f())\n')
    log = drive(builds['wasm'], src, [{}, {'wait': 10}, {}], tmp_path)
    assert [e['st'] for e in log[:2]] == ['sleep', 'done'], log
    assert ''.join(e['out'] for e in log) == '2510\n'
    assert log[1]['ms'] < 1000


@needs_node
def test_sleep_returns_to_the_event_loop(builds, tmp_path):
    log = drive(builds['wasm'], 'let t0 = clock_ms()\nprint("a")\nsleep_ms(400)\nprint("b")\n'
                                'print(clock_ms() - t0 >= 400)\n',
                [{}, {'wait': 420}], tmp_path)
    assert [e['st'] for e in log] == ['sleep', 'done']
    assert log[0]['out'] == 'a\n' and log[1]['out'] == 'b\ntrue\n'
    assert log[0]['ms'] < 300, "sleep_ms() must not busy-wait inside wh_run"
    assert 0 < log[0]['sleep'] <= 400, "the page is told how long to wait"


@needs_node
def test_wait_key_pauses(builds, tmp_path):
    if not builds.get('graphics') or not builds.get('site_wasm'):
        pytest.skip("no screen/key natives in this VM build")
    src = ('screen_init(4, 4)\nprint("ready")\nprint(key_pressed())\nlet k = wait_key()\nprint(k)\n'
           'print(wait_key())\nwait_key()\nprint(key_pressed())\n')
    log = drive(builds['site_wasm'], src, [{}, {}, {'key': 4}, {'key': 27}, {'key': 13}], tmp_path)
    assert [e['st'] for e in log] == ['wait-key', 'wait-key', 'wait-key', 'wait-key', 'done']
    assert ''.join(e['out'] for e in log) == 'ready\n0\n4\n27\n0\n'
    assert log[0]['ms'] < 1000


@slow
@needs_node
def test_a_vm_that_cannot_pause_is_reported(builds, tmp_path):
    """If a future vm/tilvm.c stops saving the pc before natives, input() must
    fail with an explicit message instead of returning a wrong value."""
    if not builds['fresh']:
        pytest.skip("no wasm toolchain")
    with open(VM_C, encoding='utf-8') as f:
        src = f.read()
    m = re.search(r'(case OP_NATIVE: \{\n\s*uint8_t id = READ8\(\), argc = READ8\(\);\n)(\s*fr->pc = pc;\n)', src)
    if not m:
        pytest.skip("OP_NATIVE changed shape; nothing to simulate")
    broken = tmp_path / 'tilvm.c'
    broken.write_text(src.replace(m.group(0), m.group(1), 1), encoding='utf-8')
    out = str(tmp_path / 'out')
    env = dict(os.environ, OUT=out, TILVM_C=str(broken), PYTHON=sys.executable, WERROR='0', EXAMPLES='0')
    r = subprocess.run([BASH, BUILD_SH], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr
    tbc = compile_file('print("x")\nprint(input("? "))\n', str(tmp_path), 'p')
    rc, wout, werr, _ = node_run(os.path.join(out, 'tilvm.wasm'), tbc, stdin='')
    assert rc == 1 and wout == 'x\n? '
    assert 'input() cannot pause in this VM build' in werr


# ═══════════════════════════════════════════════════════════════
# THE PAGE (Playwright)
# ═══════════════════════════════════════════════════════════════

def til_bytes(data: bytes) -> str:
    return 'bytes_to_str([' + ', '.join(str(b) for b in data) + '])'


def mock_tilc_source(programs) -> str:
    """TEST ONLY: a TIL program with tilc's interface (args: input.til output.tbc)
    that 'compiles' the fixed programs of mock_programs.json by writing the
    bytecode src/til.py produced for them (or printing its compile error)."""
    lines = ['# TEST ONLY: stands in for selfhost/tilc.til (knows a fixed set of programs)',
             'let a = args()', 'let src = read_file(a[0])']
    for i, prog in enumerate(programs):
        try:
            data = til.compile_to_bytecode(prog['source'], '/main.til').serialize()
            body = [f'    write_file(a[1], {til_bytes(data)})']
        except (til.CompileError, SyntaxError) as e:
            body = [f'    print({til_bytes(str(e).encode("utf-8"))})', '    exit(1)']
        lines.append(('if' if i == 0 else 'elif') + f' src == {til_bytes(prog["source"].encode("utf-8"))}')
        lines += body
    lines += ['else', '    print("mock compiler: unknown program")', '    exit(1)']
    return '\n'.join(lines) + '\n'


@needs_node
def test_browser_playground(builds, tmp_path):
    if not has_playwright():
        pytest.skip("Playwright is not installed (npm install playwright; npx playwright install chromium)")
    assets = tmp_path / 'assets'
    assets.mkdir()
    with open(MOCK_PROGRAMS, encoding='utf-8') as f:
        programs = json.load(f)
    (assets / 'mock-tilc.tbc').write_bytes(
        til.compile_to_bytecode(mock_tilc_source(programs), 'mock_tilc.til').serialize())
    env = dict(os.environ, SITE_DIR=builds['site'], TEST_ASSETS=str(assets), GRAPHICS=builds.get('graphics', ''))
    r = subprocess.run([NODE, '--test-reporter=spec', SPEC], capture_output=True, text=True, env=env,
                       timeout=600, cwd=ROOT)
    if r.returncode != 0 and 'Executable doesn\'t exist' in (r.stdout + r.stderr):
        pytest.skip("Playwright's Chromium is not installed (npx playwright install chromium)")
    assert r.returncode == 0, f"browser tests failed:\n{r.stdout}\n{r.stderr}"
    assert '✖' not in r.stdout, r.stdout
