#!/usr/bin/env python3
"""
TIL Studio, the browser IDE: its frontend (ide/studio/, design
docs/dev/TIL_STUDIO_DESIGN.md §6).

  * the UI strings: ide/studio/i18n/en.json, ru.json and kk.json have exactly
    the same keys, no empty strings and the same {placeholders}; every key
    the sources use exists;
  * ide/studio/tables.json (keywords of every language, natives, methods,
    attributes for the editor) matches the compiler (gen_tables.py --check);
  * the committed single-file build ide/studio.html is self-contained, under
    1.5 MB, the same page as website/studio/index.html, and what build.mjs
    makes from the sources (checked when ide/studio/node_modules is installed:
    cd ide/studio && npm ci);
  * the page itself: Node + Playwright + Chromium run
    ide/studio/test/studio.spec.mjs twice: the desktop face against
    test/mock_server.mjs and the web face against a static server of
    website/; then the desktop face against the real server (src/tilstudio.py,
    StudioServer over a temporary project, STUDIO_URL). The screenshot tour is
    skipped here; TIL_STUDIO_SHOTS=DIR keeps it (mock run) and writes the
    screenshots to DIR.

Needs node (+ the npm packages for the build check, + Playwright and Chromium
for the browser tests); the tests skip what is not installed.

Run: python3 -m pytest tests/test_studio_ui.py -v
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from tilhelpers import ROOT, SRC

STUDIO = os.path.join(ROOT, 'ide', 'studio')
BUILT = os.path.join(ROOT, 'ide', 'studio.html')
WEB_BUILT = os.path.join(ROOT, 'website', 'studio', 'index.html')
I18N = os.path.join(STUDIO, 'i18n')
SPEC = os.path.join(STUDIO, 'test', 'studio.spec.mjs')
LOCALES = ('en', 'ru', 'kk')
NODE = shutil.which('node')

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def load_catalog(lang):
    with open(os.path.join(I18N, lang + '.json'), encoding='utf-8') as f:
        return json.load(f)


def placeholders(text):
    return set(re.findall(r'\{(\w+)\}', text))


# ═══════════════════════════════════════════════════════════════
# UI STRINGS
# ═══════════════════════════════════════════════════════════════

def test_i18n_catalogs_have_the_same_keys():
    cats = {lang: load_catalog(lang) for lang in LOCALES}
    keys = set(cats['en'])
    assert len(keys) > 500
    for lang in LOCALES[1:]:
        missing = sorted(keys - set(cats[lang]))
        extra = sorted(set(cats[lang]) - keys)
        assert not missing and not extra, f"{lang}.json: missing {missing[:20]}, extra {extra[:20]}"


def test_i18n_strings_are_complete():
    cats = {lang: load_catalog(lang) for lang in LOCALES}
    for lang, cat in cats.items():
        empty = [k for k, v in cat.items() if not isinstance(v, str) or not v.strip()]
        assert not empty, f"{lang}.json: empty strings {empty[:20]}"
    for key, text in cats['en'].items():
        want = placeholders(text)
        for lang in LOCALES[1:]:
            got = placeholders(cats[lang][key])
            if re.search(r'\.(zero|one|two|few|many|other)$', key):
                # plural forms: English "one" says "1"; other languages show {n}
                want_cmp, got_cmp = want - {'n'}, got - {'n'}
            else:
                want_cmp, got_cmp = want, got
            assert want_cmp == got_cmp, f"{lang}.json {key!r}: placeholders {sorted(got)} != {sorted(want)}"


def test_i18n_keys_used_by_the_sources_exist():
    en = load_catalog('en')
    plural_bases = {k.rsplit('.', 1)[0] for k in en if re.search(r'\.(one|other)$', k)}
    missing = []
    call = re.compile(r"""\b(t|tn|L|tip|setText)\(\s*(?:[^,()'"]+,\s*)?'([a-zA-Z][\w-]*(?:\.[\w-]+)+)'""")
    for path in glob.glob(os.path.join(STUDIO, 'src', '**', '*.js'), recursive=True):
        with open(path, encoding='utf-8') as f:
            text = f.read()
        for m in call.finditer(text):
            fn, key = m.group(1), m.group(2)
            ok = key in plural_bases if fn == 'tn' else key in en
            if not ok:
                missing.append(f"{os.path.relpath(path, STUDIO)}: {fn}('{key}')")
    assert not missing, "keys missing from en.json:\n" + "\n".join(missing)


# ═══════════════════════════════════════════════════════════════
# TABLES AND THE BUILD
# ═══════════════════════════════════════════════════════════════

def test_tables_match_the_compiler():
    r = subprocess.run([sys.executable, os.path.join(STUDIO, 'gen_tables.py'), '--check'],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, f"ide/studio/tables.json is out of date (python3 ide/studio/gen_tables.py):\n{r.stdout}{r.stderr}"
    with open(os.path.join(STUDIO, 'tables.json'), encoding='utf-8') as f:
        tables = json.load(f)
    for lang in ('en', 'kk', 'kk-latn', 'ru'):
        assert tables['keywords'][lang], lang
    assert 'print' in tables['natives']


def test_the_built_page_is_self_contained():
    with open(BUILT, encoding='utf-8') as f:
        html = f.read()
    assert len(html.encode('utf-8')) < 1.5 * 1024 * 1024, "ide/studio.html is over 1.5 MB"
    assert html.lstrip().lower().startswith('<!doctype html>')
    assert not re.search(r'<script[^>]*\bsrc=', html, re.I), "the page loads an external script"
    assert not re.search(r'<link[^>]*rel=["\']?stylesheet', html, re.I), "the page loads an external stylesheet"
    assert 'MIT' in html and 'CodeMirror' in html, "the license header of the bundled packages is missing"
    with open(WEB_BUILT, encoding='utf-8') as f:
        assert f.read() == html, "website/studio/index.html differs from ide/studio.html (node ide/studio/build.mjs)"


@needs_node
def test_the_built_page_is_up_to_date(tmp_path):
    if not os.path.isdir(os.path.join(STUDIO, 'node_modules', 'esbuild')):
        pytest.skip("ide/studio/node_modules is not installed (cd ide/studio && npm ci)")
    out = tmp_path / 'studio.html'
    r = subprocess.run([NODE, 'build.mjs', '--out', str(out)], capture_output=True, text=True,
                       timeout=300, cwd=STUDIO)
    assert r.returncode == 0, f"build.mjs failed:\n{r.stdout}\n{r.stderr}"
    with open(BUILT, encoding='utf-8') as f:
        committed = f.read()
    assert out.read_text(encoding='utf-8') == committed, \
        "ide/studio.html is out of date: run `node build.mjs` in ide/studio and commit the result"


# ═══════════════════════════════════════════════════════════════
# THE PAGE IN A BROWSER
# ═══════════════════════════════════════════════════════════════

def has_playwright() -> bool:
    if NODE is None:
        return False
    try:
        return subprocess.run([NODE, '-e', "require('playwright')"], capture_output=True, text=True,
                              timeout=60, cwd=tempfile.gettempdir()).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_spec(env, cwd=ROOT, timeout=900):
    r = subprocess.run([NODE, '--test', '--test-reporter=spec', SPEC], capture_output=True, text=True,
                       env=env, timeout=timeout, cwd=cwd)
    if r.returncode != 0 and "Executable doesn't exist" in (r.stdout + r.stderr):
        pytest.skip("Playwright's Chromium is not installed (npx playwright install chromium)")
    return r


@needs_node
def test_browser_studio(tmp_path):
    """The desktop face against test/mock_server.mjs, the web face against website/."""
    if not has_playwright():
        pytest.skip("Playwright is not installed (npm install playwright; npx playwright install chromium)")
    env = dict(os.environ)
    env.pop('STUDIO_URL', None)
    shots = os.environ.get('TIL_STUDIO_SHOTS')
    if shots:
        env['SCREENSHOTS'] = os.path.abspath(shots)
    else:
        env['SKIP_SHOTS'] = '1'
        env['SCREENSHOTS'] = str(tmp_path / 'screenshots')
    r = run_spec(env)
    assert r.returncode == 0, f"browser tests failed:\n{r.stdout[-6000:]}\n{r.stderr[-3000:]}"
    assert '✖' not in r.stdout, r.stdout[-6000:]


@needs_node
def test_browser_studio_real_server(tmp_path):
    """The desktop face against the real server: src/tilstudio.py serving the
    built ide/studio.html over a temporary project (the spec writes its files
    through the API), with the real language server, debugger, VMs and tools."""
    if not has_playwright():
        pytest.skip("Playwright is not installed (npm install playwright; npx playwright install chromium)")
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import tilstudio
    project = tmp_path / 'project'
    project.mkdir()
    log_path = tmp_path / 'server.log'
    with open(log_path, 'w', encoding='utf-8') as log:
        server = tilstudio.StudioServer(str(project), frontend=BUILT, log=log).start()
        try:
            env = dict(os.environ, STUDIO_URL=server.url, SKIP_WEB='1', SKIP_SHOTS='1',
                       SCREENSHOTS=str(tmp_path / 'screenshots'))
            r = run_spec(env)
        finally:
            server.stop()
    server_log = log_path.read_text(encoding='utf-8')
    assert r.returncode == 0, (f"browser tests against the real server failed:\n{r.stdout[-6000:]}\n"
                               f"{r.stderr[-3000:]}\nserver log:\n{server_log[-3000:]}")
    assert '✖' not in r.stdout, r.stdout[-6000:]
    assert 'Traceback' not in server_log, server_log[-3000:]
