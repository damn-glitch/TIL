#!/usr/bin/env python3
"""
The VS Code extension (editors/vscode/, TIL Studio design section 7).

  (a) package.json: valid and consistent - every declared command registered in
      the sources (and at run time), menus, keybindings, the `til` debugger
      (launch attributes = the ones `til dap` accepts), configuration (every
      setting read by the sources), snippets that parse as TIL 3.1;
  (b) the TextMate grammar tokenizes TIL as VS Code does (vscode-textmate +
      vscode-oniguruma from node_modules): keywords of all four keyword
      languages (Russian / Latin-script Kazakh only after their #[lang] pragma),
      contextual effect words, strings, f-strings, comments, attributes,
      numbers, durations; the Sublime syntax's regexes compile;
  (c) the committed bundle dist/extension.js is up to date with src/ (rebuilt
      into a temporary folder and compared) and the sources type-check against
      the VS Code 1.82 API;
  (d) node smoke test (editors/vscode/test/smoke.js): the bundle activated
      against a stand-in `vscode` module registers every command, the debug
      type and the time-travel view; the server-launch logic computes the right
      command line (til.path, PATH, the checkout, workspace trust, Windows .bat,
      shell quoting); the bundled language client starts `til lsp`; LSP and DAP
      sessions over the computed command lines (diagnostics, til.translate,
      locale switch, breakpoints, step back, til/timeline, til/goto,
      til/inputRequested -> til/stdin);
  (e) the generated grammar / Vim / Sublime keyword lists match src/til.py
      (gen_grammar.py --check, the word lists, and Vim's highlighting itself).

Needs node (else skipped); (b) and (c) also need the dev dependencies
(`cd editors/vscode && npm ci`, else skipped); the Vim test needs vim.

Run: python3 -m pytest tests/test_vscode_ext.py -v
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from tilhelpers import ROOT, til

EXT = os.path.join(ROOT, 'editors', 'vscode')
SRC = os.path.join(EXT, 'src')
NODE = shutil.which('node')
VIM = shutil.which('vim')
MODULES = os.path.join(EXT, 'node_modules')
HAS_TEXTMATE = os.path.isdir(os.path.join(MODULES, 'vscode-textmate')) and \
    os.path.isdir(os.path.join(MODULES, 'vscode-oniguruma'))
HAS_ESBUILD = os.path.isdir(os.path.join(MODULES, 'esbuild')) and \
    os.path.isdir(os.path.join(MODULES, 'vscode-languageclient'))
HAS_TSC = os.path.isfile(os.path.join(MODULES, 'typescript', 'bin', 'tsc')) and \
    os.path.isdir(os.path.join(MODULES, '@types', 'vscode'))

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
needs_textmate = pytest.mark.skipif(NODE is None or not HAS_TEXTMATE,
                                    reason="no vscode-textmate in editors/vscode/node_modules (npm ci)")
needs_esbuild = pytest.mark.skipif(NODE is None or not HAS_ESBUILD,
                                   reason="no esbuild in editors/vscode/node_modules (npm ci)")
needs_tsc = pytest.mark.skipif(NODE is None or not HAS_TSC,
                               reason="no typescript in editors/vscode/node_modules (npm ci)")

sys.path.insert(0, os.path.join(EXT, 'scripts'))
import gen_grammar  # noqa: E402


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


PKG = load_json(os.path.join(EXT, 'package.json'))
CONTRIB = PKG['contributes']
DECLARED = [c['command'] for c in CONTRIB['commands']]
SOURCES = {name: read(os.path.join(SRC, name)) for name in sorted(os.listdir(SRC)) if name.endswith('.js')}
ALL_SOURCE = '\n'.join(SOURCES.values())


def version_tuple(v):
    return tuple(int(x) for x in re.findall(r'\d+', v)[:3])


# ═══════════════════════════════════════════════════════════════════════════
# (a) the manifest
# ═══════════════════════════════════════════════════════════════════════════

class TestManifest:
    def test_identity_and_entry_point(self):
        assert PKG['version'] == til.TIL_VERSION == '3.1.0'
        assert PKG['name'] == 'til-lang' and PKG['publisher'] == 'alisher-beisembekov'
        assert PKG['main'] == './dist/extension.js'
        assert os.path.isfile(os.path.join(EXT, 'dist', 'extension.js'))
        for event in ('onLanguage:til', 'onDebugResolve:til'):
            assert event in PKG['activationEvents']

    def test_engine_fits_the_dependencies(self):
        engine = version_tuple(PKG['engines']['vscode'])
        lock = load_json(os.path.join(EXT, 'package-lock.json'))
        lc = lock['packages']['node_modules/vscode-languageclient']
        assert lc['version'] == PKG['dependencies']['vscode-languageclient']
        assert version_tuple(lc['engines']['vscode']) <= engine
        # vsce refuses @types/vscode newer than the engine
        assert version_tuple(PKG['devDependencies']['@types/vscode']) <= engine
        assert lock['packages']['']['dependencies'] == PKG['dependencies']
        assert lock['packages']['']['devDependencies'] == PKG['devDependencies']

    def test_commands_declared_and_registered(self):
        assert len(DECLARED) == len(set(DECLARED))
        for c in CONTRIB['commands']:
            assert c['command'].startswith('til.') and c.get('category') == 'TIL' and c.get('title')
        registered = set(re.findall(r"(?:registerCommand|\br)\('(til\.[\w.]+)'", ALL_SOURCE))
        assert registered == set(DECLARED)
        # the code lenses of the language server use these three, with {uri, function}
        for c in ('til.run', 'til.debug', 'til.prove'):
            assert c in DECLARED

    def test_required_commands(self):
        for c in ('til.run', 'til.runVM', 'til.runCVM', 'til.debug', 'til.prove', 'til.proveFunction', 'til.fit',
                  'til.translate', 'til.openStudio', 'til.buildBytecode', 'til.buildCVM', 'til.restartServer',
                  'til.gotoTime'):
            assert c in DECLARED, c

    def test_menus_and_keybindings_name_known_commands(self):
        submenus = {s['id'] for s in CONTRIB['submenus']}
        for where, items in CONTRIB['menus'].items():
            assert where in ('commandPalette', 'editor/title/run', 'editor/context', 'explorer/context',
                             'debug/toolBar') or where in submenus, where
            for item in items:
                assert item.get('command') in DECLARED or item.get('submenu') in submenus, item
        keys = {k['key']: k for k in CONTRIB['keybindings']}
        for k in CONTRIB['keybindings']:
            assert k['command'] in DECLARED
            assert 'til' in k['when']
        # F5 / Ctrl+F5 belong to VS Code (debug / run without debugging): not taken
        assert 'f5' not in keys and 'ctrl+f5' not in keys
        assert keys['f6']['command'] == 'til.build' and keys['f7']['command'] == 'til.check'
        assert '!inDebugMode' in keys['f6']['when']           # F6 is Pause while debugging

    def test_debugger(self):
        (dbg,) = CONTRIB['debuggers']
        assert dbg['type'] == 'til' and dbg['languages'] == ['til']
        launch = dbg['configurationAttributes']['launch']
        assert launch['required'] == ['program']
        props = launch['properties']
        documented = ('program', 'args', 'stdin', 'keys', 'cwd', 'lang', 'stopOnEntry', 'stopAtEnd', 'noFuse')
        for a in documented:
            assert a in props, a
        assert props['lang']['enum'] == ['en', 'ru', 'kk']
        # every attribute is one that `til dap` reads in its launch request
        dap = read(os.path.join(ROOT, 'src', 'tildap.py'))
        launch_code = dap[dap.index('def _launch('):dap.index('def _compile_failed(')]
        for a in props:
            assert re.search(r"(?:_str_arg|_bool_arg|_int_arg)\(args, '%s'|args\.get\('%s'" % (a, a), launch_code), a
        for conf in dbg['initialConfigurations']:
            assert conf['type'] == 'til' and conf['request'] == 'launch' and conf['program'] == '${file}'
        for snip in dbg['configurationSnippets']:
            assert snip['body']['type'] == 'til' and snip['body']['request'] == 'launch'
            assert set(snip['body']) - {'type', 'request', 'name'} <= set(props)
        assert CONTRIB['breakpoints'] == [{'language': 'til'}]
        views = CONTRIB['views']['debug']
        assert views == [{'type': 'webview', 'id': 'til.timeline', 'name': 'TIL Time Travel',
                          'when': "debugType == 'til'"}]
        assert "registerWebviewViewProvider('til.timeline'" in ALL_SOURCE
        assert "registerDebugAdapterDescriptorFactory('til'" in ALL_SOURCE
        # the custom DAP messages of src/tildap.py the extension handles
        for name in ('til/inputRequested', 'til/stdin', 'til/screen', 'til/timeline', 'til/goto'):
            assert name in SOURCES['debug.js'] + SOURCES['webviews.js'], name
            assert name in dap, name

    def test_configuration(self):
        props = CONTRIB['configuration']['properties']
        for key in ('til.path', 'til.python', 'til.lsp.enabled', 'til.locale', 'til.trace.server'):
            assert key in props, key
        assert props['til.locale']['enum'] == ['auto', 'en', 'ru', 'kk'] and props['til.locale']['default'] == 'auto'
        assert props['til.trace.server']['enum'] == ['off', 'messages', 'verbose']
        assert props['til.lsp.enabled']['default'] is True
        for key in props:
            short = key[len('til.'):]
            if key == 'til.trace.server':
                continue                       # read by vscode-languageclient (client id 'til')
            assert re.search(r"get\('%s'" % re.escape(short), ALL_SOURCE), key
        assert "new lc.LanguageClient('til'" in SOURCES['client.js']
        assert "initializationOptions: { locale:" in SOURCES['client.js']
        restricted = PKG['capabilities']['untrustedWorkspaces']['restrictedConfigurations']
        assert {'til.path', 'til.python'} <= set(restricted)

    def test_language_files(self):
        (lang,) = CONTRIB['languages']
        assert lang['id'] == 'til' and lang['extensions'] == ['.til']
        conf = load_json(os.path.join(EXT, lang['configuration']))
        assert conf['comments']['lineComment'] == '#'
        assert conf['folding']['offSide'] is True
        (grammar,) = CONTRIB['grammars']
        assert grammar['scopeName'] == 'source.til'
        assert load_json(os.path.join(EXT, grammar['path']))['scopeName'] == 'source.til'
        for s in CONTRIB['snippets']:
            load_json(os.path.join(EXT, s['path']))

    def test_package_files(self):
        ignore = read(os.path.join(EXT, '.vscodeignore'))
        for pattern in ('src/**', 'node_modules/**', 'scripts/**', 'test/**'):
            assert pattern in ignore
        gitignore = read(os.path.join(EXT, '.gitignore'))
        assert '!dist/' in gitignore and 'node_modules/' in gitignore
        if shutil.which('git') is None:
            return
        out = subprocess.run(['git', 'check-ignore', '-q', os.path.join(EXT, 'dist', 'extension.js')], cwd=ROOT,
                             capture_output=True)
        if out.returncode == 128:
            return                                       # not a git checkout
        assert out.returncode == 1, "dist/extension.js must not be git-ignored (it is committed)"

    def test_bundle_has_no_unresolved_requires(self):
        text = read(os.path.join(EXT, 'dist', 'extension.js'))
        assert text.startswith(f'/* TIL for VS Code {PKG["version"]} ')
        required = set(re.findall(r'require\("([^"]+)"\)', text))
        builtins = {'fs', 'os', 'path', 'child_process', 'crypto', 'net', 'util', 'url', 'events', 'assert',
                    'stream', 'buffer', 'string_decoder', 'readline', 'tty', 'http', 'https', 'zlib'}
        assert 'vscode' in required
        assert required - {'vscode'} <= builtins | {'node:' + b for b in builtins}, required
        notices = read(os.path.join(EXT, 'ThirdPartyNotices.txt'))
        assert 'vscode-languageclient 9.0.1' in notices


# ── snippets ────────────────────────────────────────────────────────────────

def render_snippet(text, i=0, inside=False):
    """VS Code's snippet syntax, every placeholder at its default (a choice: the
    first one): `$1`, `${1}`, `${1:default}` (nested; the first unescaped `}`
    closes it), `${1|a,b|}`, and the escapes `\\$`, `\\}`, `\\\\`."""
    out = []
    while i < len(text):
        c = text[i]
        if c == '\\' and i + 1 < len(text) and text[i + 1] in '$}\\':
            out.append(text[i + 1])
            i += 2
            continue
        if inside and c == '}':
            return ''.join(out), i + 1
        if c == '$':
            m = re.match(r'\$(\d+)|\$\{(\d+)\}', text[i:])
            if m:
                i += len(m.group(0))
                continue
            m = re.match(r'\$\{\d+\|([^|]*)\|\}', text[i:])
            if m:
                out.append(m.group(1).split(',')[0])
                i += len(m.group(0))
                continue
            m = re.match(r'\$\{\d+:', text[i:])
            if m:
                inner, i = render_snippet(text, i + len(m.group(0)), True)
                out.append(inner)
                continue
        out.append(c)
        i += 1
    return ''.join(out), i


def expand_snippet(body):
    """A snippet body as VS Code inserts it with the defaults; a line that is only
    a tab stop becomes a statement, so that blocks are not empty."""
    lines = body if isinstance(body, list) else [body]
    out = []
    for line in lines:
        text, _ = render_snippet(line)
        if not text.strip() and line.strip():
            text = line[:len(line) - len(line.lstrip())] + 'print(0)'
        out.append(text)
    return '\n'.join(out) + '\n'


SNIPPETS = load_json(os.path.join(EXT, 'snippets', 'til.json'))


def snippet_program(name, text):
    """A program around a snippet: definitions as they are, statements in main()."""
    first = text.lstrip().split('\n')[0]
    top = re.match(r'(#\[|struct |impl |trait |enum |cap |const |import |from |main\(|негізгі\(|[\w]+\(.*\)( ->|$))', first)
    if name in ('Requires', 'Ensures', 'Effects'):
        return text + 'f(n: int) -> int\n    return n\n'
    if name == 'Energy Mode':
        return text + 'main()\n    print(1)\n'
    if name == 'Closure':
        return 'main()\n    let f = ' + text
    if top and not text.lstrip().startswith(('print(', 'perform ')):
        return text
    return 'main()\n' + ''.join('    ' + ln + '\n' if ln else '\n' for ln in text.rstrip('\n').split('\n'))


class TestSnippets:
    def test_prefixes_unique(self):
        prefixes = [s['prefix'] for s in SNIPPETS.values()]
        assert len(prefixes) == len(set(prefixes))
        for p in ('handle', 'retry', 'timeout', 'fnc', 'requires', 'ensures', 'struct', 'impl', 'match', 'screen',
                  'energy'):
            assert p in prefixes, p

    @pytest.mark.parametrize('name', sorted(SNIPPETS))
    def test_snippet_parses(self, name):
        src = snippet_program(name, expand_snippet(SNIPPETS[name]['body']))
        try:
            til.parse_source(src, f'<snippet {name}>')
        except SyntaxError as e:
            pytest.fail(f"snippet {name!r} does not parse:\n{src}\n{e}")

    @pytest.mark.parametrize('name', ['Graphics Loop', 'Kazakh Main', 'Russian Main', 'Function with Contracts',
                                      'Struct with Impl', 'Enum', 'Trait'])
    def test_snippet_compiles(self, name):
        src = snippet_program(name, expand_snippet(SNIPPETS[name]['body']))
        if 'main()' not in src and 'негізгі()' not in src and 'главная()' not in src:
            src += '\nmain()\n    print(1)\n'
        til.compile_to_bytecode(src, f'<snippet {name}>')


# ═══════════════════════════════════════════════════════════════════════════
# (e) generated syntax files = the compiler's tables
# ═══════════════════════════════════════════════════════════════════════════

def all_keywords(langs=('en', 'kk', 'kk-latn', 'ru')):
    words = {}
    for lang in langs:
        table = til.ENGLISH_KEYWORDS if lang == 'en' else til.LANGUAGES[lang][0]
        for w in table:
            words[w] = lang
        for w in til.CONTEXTUAL_KEYWORDS.get(lang, {}):
            words[w] = lang
    return words


class TestGenerated:
    def test_generated_files_are_up_to_date(self):
        r = subprocess.run([sys.executable, os.path.join(EXT, 'scripts', 'gen_grammar.py'), '--check'],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    def test_textmate_word_lists(self):
        g = load_json(os.path.join(EXT, 'syntaxes', 'til.tmLanguage.json'))
        repo = g['repository']

        def words_of(key):
            text = json.dumps(repo[key], ensure_ascii=False)
            return set(re.findall(r'[\w]+', text))
        for lang in ('en', 'kk', 'kk-latn', 'ru'):
            have = words_of('keywords-' + lang)
            table = til.ENGLISH_KEYWORDS if lang == 'en' else til.LANGUAGES[lang][0]
            for w in list(table) + list(til.CONTEXTUAL_KEYWORDS.get(lang, {})):
                assert w in have, (lang, w)
            if lang != 'en':
                assert set(til.LANGUAGES[lang][1]) <= words_of('aliases-' + lang)
        builtins = words_of('builtin')
        assert set(til.NATIVES) <= builtins
        assert {'Some', 'Ok', 'Err', 'None'} <= builtins
        assert set(til.TYPE_MAP) <= words_of('type')
        import tills
        assert set(tills.ATTRIBUTE_DOCS) <= words_of('attribute')
        # Russian / Latin-script Kazakh keywords only inside their pragma contexts
        default = json.dumps([repo[k] for k in ('keywords-en', 'keywords-kk', 'expression')], ensure_ascii=False)
        assert 'если' not in default and 'eger' not in default

    def test_vim_and_sublime_word_lists(self):
        vim = read(gen_grammar.VIM_PATH)
        sublime = read(gen_grammar.SUBLIME_PATH)
        guard_ru = vim.index("search('\\m\\c^\\s*#\\[\\s*lang\\s*:\\%([^\\],]*,\\)*\\s*ru")
        always = vim[:guard_ru]
        for w, lang in all_keywords().items():
            assert re.search(r'(?<![\w])%s(?![\w])' % re.escape(w), vim), w
            assert re.search(r'(?<![\w])%s(?![\w])' % re.escape(w), sublime), w
            if lang in ('ru', 'kk-latn') and w not in all_keywords(('en', 'kk')):
                assert not re.search(r'(?<![\w])%s(?![\w])' % re.escape(w), always), w
        for name in til.NATIVES:
            assert name in vim and name in sublime
        assert f'TIL (The Intelligent Language) {til.TIL_VERSION}' in vim

    def test_language_configuration_word_lists(self):
        conf = json.dumps(load_json(gen_grammar.LANGCONF_PATH)['indentationRules'], ensure_ascii=False)
        for name in ('IF', 'ELSE', 'FOR', 'WHILE', 'LOOP', 'MATCH', 'STRUCT', 'IMPL', 'RETRY', 'TIMEOUT'):
            for lang in ('en', 'kk', 'kk-latn', 'ru'):
                table = til.ENGLISH_KEYWORDS if lang == 'en' else til.LANGUAGES[lang][0]
                for w, tt in table.items():
                    if tt.name == name:
                        assert w in conf, (lang, w)

    @pytest.mark.skipif(VIM is None, reason="vim is not installed")
    def test_vim_highlighting(self, tmp_path):
        sample = ('#[lang: ru]\n#[requires: b != 0]\nstruct Point\n    x: float\n'
                  'main()\n    let s = f"{n:.2f} {{x}}"\n    var y = 0x1F + 3.5 + 5s\n'
                  '    егер x\n        басып_шығару(x)\n    если y\n        печать(int(y))\n'
                  '    let t = handle\n        greet()\n    with\n        Ask(k) => resume("x")\n')
        plain = 'если y\n    eger x\nегер z\n'
        names = {}
        for label, text in (('ru', sample), ('plain', plain)):
            src = tmp_path / f'{label}.til'
            src.write_text(text, encoding='utf-8')
            out = tmp_path / f'{label}.out'
            script = tmp_path / f'{label}.vim'
            script.write_text(
                'set nocompatible\nset encoding=utf-8\nsyntax on\n'
                f'edit {src}\nsyntax clear\nunlet! b:current_syntax\nsource {gen_grammar.VIM_PATH}\n'
                'let out = []\nfor l in range(1, line("$"))\n  let text = getline(l)\n  let cols = []\n'
                '  let c = 1\n  while c <= len(text)\n    call add(cols, synIDattr(synID(l, c, 1), "name"))\n'
                '    let c += 1\n  endwhile\n  call add(out, join(cols, ","))\nendfor\n'
                f'call writefile(out, "{out}")\nqa!\n', encoding='utf-8')
            subprocess.run([VIM, '-Nu', 'NONE', '-i', 'NONE', '-n', '-es', '-S', str(script)], timeout=60,
                           capture_output=True)
            lines = text.split('\n')
            groups = out.read_text(encoding='utf-8').split('\n')
            per_line = []
            for line, g in zip(lines, groups):
                cols = g.split(',') if g else []
                words = {}
                for m in re.finditer(r'[\w.]+|[^\w\s]+', line):
                    start = len(line[:m.start()].encode('utf-8'))
                    words.setdefault(m.group(), cols[start] if start < len(cols) else '')
                per_line.append(words)
            names[label] = per_line
        ru = names['ru']
        assert ru[0]['lang'] == 'tilAttrName'
        assert ru[1]['requires'] == 'tilAttrName' and ru[1]['0'] == 'tilNumber'
        assert ru[2]['struct'] == 'tilStorage' and ru[2]['Point'] == 'tilDeclType'
        assert ru[3]['float'] == 'tilType'
        assert ru[5]['let'] == 'tilStorage' and ru[5]['n'] == 'tilInterp'
        assert ru[6]['0x1F'] == 'tilNumber' and ru[6]['3.5'] == 'tilFloat' and ru[6]['5s'] == 'tilDuration'
        assert ru[7]['егер'] == 'tilConditional' and ru[8]['басып_шығару'] == 'tilBuiltin'
        assert ru[9]['если'] == 'tilConditional' and ru[10]['печать'] == 'tilBuiltin' and ru[10]['int'] == 'tilBuiltin'
        assert ru[11]['handle'] == 'tilEffect' and ru[13]['with'] == 'tilEffect' and ru[14]['resume'] == 'tilEffect'
        plain_lines = names['plain']
        assert plain_lines[0]['если'] == '' and plain_lines[1]['eger'] == ''      # no pragma: identifiers
        assert plain_lines[2]['егер'] == 'tilConditional'

    def test_sublime_syntax_structure(self):
        yaml = pytest.importorskip('yaml')
        with open(gen_grammar.SUBLIME_PATH, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        assert data['scope'] == 'source.til' and data['file_extensions'] == ['til']
        assert data['contexts'] == gen_grammar.sublime_syntax(gen_grammar.Tables())['contexts']
        for ctx in data['contexts'].values():
            for rule in ctx:
                if 'include' in rule:
                    assert rule['include'] in data['contexts'], rule
                for k in ('push', 'set'):
                    if k in rule:
                        assert rule[k] in data['contexts'], rule


# ═══════════════════════════════════════════════════════════════════════════
# (b) the grammar, tokenized as VS Code does
# ═══════════════════════════════════════════════════════════════════════════

SAMPLES = {
    'english': '''# a comment with #[not an attribute]
#[pure]
#[requires: b != 0 and a[0] > 1]
safe_div(a: int, b: int) -> int
    return a / b

struct Point
    x: float

main()
    let s = f"{n:.2f} and {{x}} {d['k']} {p.len()}"
    var y = 0x1F + 0b101 + 0o17 + 1_000 + 3.5e3 + 5s + 100ms
    if y == 1 and not false
        print("hi\\n", 'c')
    for i in 0..=10
        y += i
    let r = retry(3)
        flaky()
    else
        -1
    let t = handle
        greet()
    with
        Ask(key) => resume("x")
    let v = Shape.Circle(1.0)
    print(nums.map(triple), int(y), BLACK, Some(1), None)
''',
    'kazakh': '''құрылым Нүкте
    x: int
негізгі()
    тұрақты x = 5
    егер x > 3 және ақиқат
        басып_шығару("үлкен")
    әйтпесе
        шығару(x)
''',
    'russian': '''#[lang: ru]
главная()
    пусть x = 5
    если x > 3 и истина
        печать("больше")
    иначе
        вывод(x)
    пусть t = обработать
        работа()
    с
        Спросить(к) => возобновить(1)
''',
    'latin': '''#[lang: kk-latn]
negizgi()
    turaqty x = 5
    eger x > 3 zhane aqiqat
        basyp_shygaru(x)
''',
    'both': '''#[lang: ru, kk-latn]
eger x
    если y
''',
    'no_pragma': '''если x
    eger y
    с = 1
    печать(x)
''',
}


@pytest.fixture(scope='module')
def tokens():
    if NODE is None or not HAS_TEXTMATE:
        pytest.skip("no vscode-textmate in editors/vscode/node_modules (npm ci)")
    req = {'grammar': os.path.join(EXT, 'syntaxes', 'til.tmLanguage.json'), 'samples': SAMPLES,
           'regexes': gen_grammar.sublime_regexes(gen_grammar.Tables())}
    r = subprocess.run([NODE, os.path.join(EXT, 'test', 'tokenize.js')], input=json.dumps(req), capture_output=True,
                       text=True, encoding='utf-8', timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def scope(tokens, sample, line, text):
    """The innermost scope of the token `text` on a line of a sample."""
    for t in tokens['samples'][sample][line]:
        if t['text'] == text:
            return t['scopes'][-1]
    raise AssertionError(f"no token {text!r} in line {line} of {sample}: "
                         f"{[t['text'] for t in tokens['samples'][sample][line]]}")


def scopes(tokens, sample, line, text):
    for t in tokens['samples'][sample][line]:
        if t['text'] == text:
            return t['scopes']
    raise AssertionError(f"no token {text!r} in line {line} of {sample}")


class TestGrammar:
    def test_comments_and_attributes(self, tokens):
        assert scope(tokens, 'english', 0, '#') == 'punctuation.definition.comment.til'
        assert scopes(tokens, 'english', 0, ' a comment with #[not an attribute]')[-1] == 'comment.line.number-sign.til'
        assert scope(tokens, 'english', 1, '#[') == 'punctuation.definition.attribute.begin.til'
        assert scope(tokens, 'english', 1, 'pure') == 'entity.other.attribute-name.til'
        assert scope(tokens, 'english', 2, 'requires') == 'entity.other.attribute-name.til'
        assert scope(tokens, 'english', 2, '!=') == 'keyword.operator.comparison.til'
        assert scope(tokens, 'english', 2, 'and') == 'keyword.operator.logical.til'
        assert 'meta.attribute.til' in scopes(tokens, 'english', 2, '1')
        last = tokens['samples']['english'][2][-1]
        assert last['text'] == ']' and last['scopes'][-1] == 'punctuation.definition.attribute.end.til'

    def test_english_keywords_and_names(self, tokens):
        assert scope(tokens, 'english', 3, 'safe_div') == 'entity.name.function.til'
        assert scope(tokens, 'english', 3, 'int') == 'support.type.primitive.til'
        assert scope(tokens, 'english', 3, '->') == 'keyword.operator.arrow.til'
        assert scope(tokens, 'english', 4, 'return') == 'keyword.control.flow.til'
        assert scope(tokens, 'english', 6, 'struct') == 'storage.type.til'
        assert scope(tokens, 'english', 6, 'Point') == 'entity.name.type.struct.til'
        assert scope(tokens, 'english', 10, 'let') == 'storage.type.til'
        assert scope(tokens, 'english', 12, 'if') == 'keyword.control.conditional.til'
        assert scope(tokens, 'english', 12, 'not') == 'keyword.operator.logical.til'
        assert scope(tokens, 'english', 12, 'false') == 'constant.language.boolean.til'
        assert scope(tokens, 'english', 14, 'for') == 'keyword.control.loop.til'
        assert scope(tokens, 'english', 14, 'in') == 'keyword.control.loop.til'
        assert scope(tokens, 'english', 14, '..=') == 'keyword.operator.range.til'
        assert scope(tokens, 'english', 16, 'retry') == 'keyword.control.recovery.til'
        assert scope(tokens, 'english', 18, 'else') == 'keyword.control.conditional.til'
        assert scope(tokens, 'english', 24, 'Shape') == 'entity.name.type.til'
        assert scope(tokens, 'english', 24, 'Circle') == 'variable.other.enummember.til'
        assert scope(tokens, 'english', 25, 'print') == 'support.function.builtin.til'
        assert scope(tokens, 'english', 25, 'map') == 'entity.name.function.member.til'
        assert scope(tokens, 'english', 25, 'int') == 'support.function.builtin.til'
        assert scope(tokens, 'english', 25, 'BLACK') == 'variable.other.constant.til'
        assert scope(tokens, 'english', 25, 'Some') == 'support.function.builtin.constructor.til'
        assert scope(tokens, 'english', 25, 'None') == 'constant.language.none.til'

    def test_effects(self, tokens):
        assert scope(tokens, 'english', 20, 'handle') == 'keyword.control.effect.til'
        assert scope(tokens, 'english', 22, 'with') == 'keyword.control.effect.til'
        assert scope(tokens, 'english', 23, 'Ask') == 'entity.name.type.til'
        assert scope(tokens, 'english', 23, 'resume') == 'keyword.control.effect.til'
        assert scope(tokens, 'russian', 7, 'обработать') == 'keyword.control.effect.til'
        assert scope(tokens, 'russian', 9, 'с') == 'keyword.control.effect.til'
        assert scope(tokens, 'russian', 10, 'возобновить') == 'keyword.control.effect.til'
        # `с` is only `with` alone on its line (a variable elsewhere)
        assert not any(t['text'] == 'с' for t in tokens['samples']['no_pragma'][2])

    def test_strings_and_fstrings(self, tokens):
        line = tokens['samples']['english'][10]
        assert scope(tokens, 'english', 10, 'f') == 'storage.type.string.til'
        assert 'string.interpolated.double.til' in scopes(tokens, 'english', 10, ' and ')
        assert scope(tokens, 'english', 10, '{') == 'punctuation.section.interpolation.begin.til'
        assert scope(tokens, 'english', 10, '.2f') == 'constant.other.format-spec.til'
        assert scope(tokens, 'english', 10, '{{') == 'constant.character.escape.til'
        inner = scopes(tokens, 'english', 10, 'k')
        assert inner[-1] == 'string.quoted.single.til' and 'meta.interpolation.til' in inner
        assert scope(tokens, 'english', 10, 'len') == 'entity.name.function.member.til'
        assert line[-1]['scopes'][-1] == 'punctuation.definition.string.end.til'
        assert scope(tokens, 'english', 13, '\\n') == 'constant.character.escape.til'
        assert scope(tokens, 'english', 13, 'hi') == 'string.quoted.double.til'
        assert scope(tokens, 'english', 13, 'c') == 'string.quoted.single.til'
        assert scope(tokens, 'kazakh', 5, 'үлкен') == 'string.quoted.double.til'

    def test_numbers_and_durations(self, tokens):
        assert scope(tokens, 'english', 11, '0x1F') == 'constant.numeric.hex.til'
        assert scope(tokens, 'english', 11, '0b101') == 'constant.numeric.binary.til'
        assert scope(tokens, 'english', 11, '0o17') == 'constant.numeric.octal.til'
        assert scope(tokens, 'english', 11, '1_000') == 'constant.numeric.integer.til'
        assert scope(tokens, 'english', 11, '3.5e3') == 'constant.numeric.float.til'
        assert scope(tokens, 'english', 11, '5') == 'constant.numeric.duration.til'
        assert scope(tokens, 'english', 11, 's') == 'keyword.other.unit.til'
        assert scope(tokens, 'english', 11, 'ms') == 'keyword.other.unit.til'
        assert scope(tokens, 'english', 14, '10') == 'constant.numeric.integer.til'

    def test_kazakh_keywords(self, tokens):
        assert scope(tokens, 'kazakh', 0, 'құрылым') == 'storage.type.til'
        assert scope(tokens, 'kazakh', 0, 'Нүкте') == 'entity.name.type.struct.til'
        assert scope(tokens, 'kazakh', 2, 'негізгі') == 'support.function.builtin.til'
        assert scope(tokens, 'kazakh', 3, 'тұрақты') == 'storage.type.til'
        assert scope(tokens, 'kazakh', 4, 'егер') == 'keyword.control.conditional.til'
        assert scope(tokens, 'kazakh', 4, 'және') == 'keyword.operator.logical.til'
        assert scope(tokens, 'kazakh', 4, 'ақиқат') == 'constant.language.boolean.til'
        assert scope(tokens, 'kazakh', 5, 'басып_шығару') == 'support.function.builtin.til'
        assert scope(tokens, 'kazakh', 6, 'әйтпесе') == 'keyword.control.conditional.til'

    def test_russian_and_latin_keywords_after_their_pragma(self, tokens):
        assert scope(tokens, 'russian', 0, ' ru') == 'support.constant.language-code.til'
        assert 'meta.attribute.til' in scopes(tokens, 'russian', 0, 'lang')
        assert scope(tokens, 'russian', 2, 'пусть') == 'storage.type.til'
        assert scope(tokens, 'russian', 3, 'если') == 'keyword.control.conditional.til'
        assert scope(tokens, 'russian', 3, 'и') == 'keyword.operator.logical.til'
        assert scope(tokens, 'russian', 3, 'истина') == 'constant.language.boolean.til'
        assert scope(tokens, 'russian', 4, 'печать') == 'support.function.builtin.til'
        assert scope(tokens, 'russian', 5, 'иначе') == 'keyword.control.conditional.til'
        assert scope(tokens, 'russian', 6, 'вывод') == 'support.function.builtin.til'
        assert scope(tokens, 'latin', 2, 'turaqty') == 'storage.type.til'
        assert scope(tokens, 'latin', 3, 'eger') == 'keyword.control.conditional.til'
        assert scope(tokens, 'latin', 3, 'zhane') == 'keyword.operator.logical.til'
        assert scope(tokens, 'latin', 4, 'basyp_shygaru') == 'support.function.builtin.til'
        assert scope(tokens, 'both', 1, 'eger') == 'keyword.control.conditional.til'
        assert scope(tokens, 'both', 2, 'если') == 'keyword.control.conditional.til'

    def test_no_pragma_no_russian_keywords(self, tokens):
        # like the compiler: without #[lang: ru] / #[lang: kk-latn] these are identifiers
        for line, word in ((0, 'если'), (1, 'eger')):
            toks = tokens['samples']['no_pragma'][line]
            assert all(word not in t['text'] or t['scopes'] == ['source.til'] for t in toks), toks
        assert scope(tokens, 'no_pragma', 3, 'печать') == 'entity.name.function.til'

    def test_sublime_regexes_compile(self, tokens):
        assert tokens['regexes'] == []


# ═══════════════════════════════════════════════════════════════════════════
# (c) the bundle is built from the sources
# ═══════════════════════════════════════════════════════════════════════════

class TestBundle:
    @needs_esbuild
    def test_bundle_up_to_date(self):
        r = subprocess.run([NODE, os.path.join(EXT, 'scripts', 'build.mjs'), '--check'], cwd=EXT,
                           capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stdout + r.stderr

    @needs_tsc
    def test_sources_typecheck(self):
        r = subprocess.run([NODE, os.path.join(MODULES, 'typescript', 'bin', 'tsc'), '-p',
                            os.path.join(EXT, 'jsconfig.json')], cwd=EXT, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout + r.stderr


# ═══════════════════════════════════════════════════════════════════════════
# (d) the extension in node: activation, launch logic, LSP and DAP
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def smoke():
    if NODE is None:
        pytest.skip("node is not installed")
    env = dict(os.environ)
    env.pop('TIL_LANG', None)
    r = subprocess.run([NODE, os.path.join(EXT, 'test', 'smoke.js'), ROOT, '--python', sys.executable],
                       capture_output=True, text=True, encoding='utf-8', timeout=240, env=env)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert not data.get('error'), data.get('error')
    return data


TIL_PY = os.path.join(ROOT, 'src', 'til.py')


@needs_node
class TestSmoke:
    def test_activation_registers_everything(self, smoke):
        act = smoke['activation']
        assert act['commands'] == sorted(DECLARED)
        assert act['debugTypes'] == ['til'] and act['trackers'] == ['til']
        assert act['debugProviders'] == ['til', 'til'] and act['dynamicProvider']
        assert act['webviewViews'] == ['til.timeline']
        assert act['api'] == ['lspServerCommand', 'tilCommand']

    def test_debug_adapter_descriptor(self, smoke):
        adapter = smoke['activation']['adapter']
        assert adapter['command'] == sys.executable and adapter['args'] == [TIL_PY, 'dap']

    def test_f5_without_launch_json(self, smoke):
        f5 = smoke['activation']['f5']
        assert f5['type'] == 'til' and f5['request'] == 'launch' and f5['program'] == '${file}'
        assert smoke['activation']['f5kk']['lang'] == 'kk'           # til.locale -> the adapter's messages

    def test_language_client_starts_the_server(self, smoke):
        client = smoke['client']
        assert any(f'language server running: til-lsp {til.TIL_VERSION}' in line for line in client['log']), client
        assert client['status'][-1] == f'$(check) TIL {til.TIL_VERSION}'

    def test_launch_resolution(self, smoke):
        r = smoke['resolve']
        assert r['settingPy']['command'] == sys.executable and r['settingPy']['args'] == [TIL_PY, 'lsp']
        assert r['onPath']['til']['source'] == 'path' and r['onPath']['args'] == ['lsp']
        assert r['onPath']['command'].endswith(os.path.join('bin', 'til.cmd' if os.name == 'nt' else 'til'))
        ext = r['fromExtension']
        assert ext['til']['source'] == 'extension' and ext['args'] == [TIL_PY, 'lsp']
        assert os.path.basename(ext['command']).startswith('python')
        assert r['settingFolder']['args'] == [TIL_PY, 'lsp']
        assert not r['settingMissing']['til']['found'] and 'til.path' in r['settingMissing']['til']['problem']
        assert r['workspaceTrusted']['til']['source'] == 'workspace'
        assert not r['workspaceUntrusted']['til']['found']               # Restricted Mode: not from the workspace
        assert r['home']['til']['source'] == 'home' and r['home']['args'][-2].endswith('til.py')
        assert r['legacy']['args'] == [TIL_PY, 'lsp']
        assert r['windowsBat'] == {'command': 'C:\\Python312\\python.exe', 'args': ['C:\\Users\\a\\.til\\bin\\til.py'],
                                   'found': True, 'source': 'path', 'tilPy': 'C:\\Users\\a\\.til\\bin\\til.py',
                                   'pythonFound': True}
        assert r['windowsCmdOnly']['command'].lower().endswith('cmd.exe')
        assert r['windowsCmdOnly']['args'] == ['/d', '/c', 'D:\\tools\\til.cmd']
        assert r['windowsPySetting']['command'] == 'py' and r['windowsPySetting']['args'][0] == '-3'
        assert r['windowsPySetting']['pythonFound']
        missing = r['missingPython']
        assert missing['til']['found'] and missing['til']['pythonFound'] is False
        assert 'Python 3 was not found' in missing['problem'] and 'til.python' in missing['problem']

    def test_terminal_quoting(self, smoke):
        q = smoke['quoting']
        assert q['posix'] == "/usr/bin/python3 '/home/a b/til.py' run 'it'\\''s.til'"
        assert q['pwsh'] == "& 'C:\\Program Files\\Python\\python.exe' C:\\til\\til.py run 'C:\\a b\\it''s.til'"
        assert q['cmd'] == 'C:\\Python\\python.exe "C:\\a b\\til.py" run x.til'
        assert q['defaultWin'] == 'pwsh'

    def test_lsp_session(self, smoke):
        lsp = smoke['lsp']
        assert lsp['serverInfo'] == {'name': 'til-lsp', 'version': til.TIL_VERSION}
        assert lsp['positionEncoding'] == 'utf-16'
        (d,) = lsp['diagnostics']
        assert d['code'] == 'E-VM' and d['line'] == 2 and "'y'" in d['message']
        assert re.search('[а-я]', d['message'])                           # initializationOptions.locale = ru
        assert lsp['lensCommands'] == ['til.run', 'til.debug']
        assert all(args[0]['function'] == 'main' and args[0]['uri'].startswith('file://')
                   for args in lsp['lensArguments'])
        assert lsp['translated'].startswith('#[lang: ru]') and 'главная()' in lsp['translated']
        assert lsp['runCommandResult'] is None
        assert 'басып_шығару' in lsp['hoverKk'] and re.search('[әғқңөұүһі]', lsp['hoverKk'])
        assert lsp['exitCode'] == 0

    def test_dap_session(self, smoke):
        dap = smoke['dap']
        assert dap['supportsStepBack'] and dap['supportsConfigurationDone'] and dap['launch']
        assert dap['breakpoints'] == [True] and dap['firstStop'] == 'breakpoint'
        assert dap['stepBack']
        tl = dap['timeline']
        assert set(tl) == {'index', 'total', 'complete'} and 0 < tl['index'] < tl['total']
        assert dap['goto'] and dap['gotoReason'] == 'goto'
        assert dap['inputRequested'] and dap['stdin'] and dap['output'] == 'Aigerim: 3'
        assert dap['endReason'] == 'exit' and dap['exitCode'] == 0


# ── language configuration regexes (JavaScript) ────────────────────────────

INDENT_CASES = [
    ('if x > 1', True), ('    for i in 0..10', True), ('while true', True), ('loop', True), ('else', True),
    ('struct Point', True), ('impl Greeter for Student', True), ('main()', True), ('add(a: int, b: int) -> int', True),
    ('    distance(self) -> float', True), ('    let t = handle', True), ('    let r = retry(3)', True),
    ('    return match s', True), ('    Ask(key) =>', True), ('    with', True), ('егер x', True),
    ('әйтпесе', True), ('если x', True), ('eger x', True), ('    өңдеу', True),
    ('    print(x)', False), ('    let x = 1', False), ('    return x + 1', False), ('    x = handle_it(1)', False),
    ('    Ask(key) => resume(1)', False), ('    өңдеу(x)', False), ('# if not a block', False),
]
DEDENT_CASES = [('    else', True), ('    иначе', True), ('    with', True), ('    әйтпесе', True),
                ('    elsewhere()', False), ('    с = 1', False)]


@needs_node
def test_language_configuration_regexes():
    conf = load_json(os.path.join(EXT, 'language-configuration.json'))
    script = (
        "const conf = JSON.parse(require('fs').readFileSync(process.argv[1], 'utf8'));\n"
        "const cases = JSON.parse(process.argv[2]);\n"
        "const inc = new RegExp(conf.indentationRules.increaseIndentPattern);\n"
        "const dec = new RegExp(conf.indentationRules.decreaseIndentPattern);\n"
        "const out = {inc: cases.inc.map(([l]) => inc.test(l)), dec: cases.dec.map(([l]) => dec.test(l)),\n"
        "  enter: conf.onEnterRules.map((r) => String(new RegExp(r.beforeText).test('    return x')))};\n"
        "console.log(JSON.stringify(out));\n")
    cases = {'inc': INDENT_CASES, 'dec': DEDENT_CASES}
    r = subprocess.run([NODE, '-e', script, os.path.join(EXT, 'language-configuration.json'), json.dumps(cases)],
                       capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out['inc'] == [want for _line, want in INDENT_CASES], list(zip(INDENT_CASES, out['inc']))
    assert out['dec'] == [want for _line, want in DEDENT_CASES], list(zip(DEDENT_CASES, out['dec']))
    assert out['enter'] == ['false', 'true']             # outdent after `return x`, not after break rules
    assert conf['onEnterRules'][0]['action'] == {'indent': 'outdent'}


# ── the pure modules in node: launcher quoting edge cases, report parsers ──

@needs_node
def test_report_parsers():
    prove_out = ("Proving p.til with Z3 4.13 (64-bit integers, 10000 ms per query)\n"
                 "  add: counterexample: a = 1, b = 0 (#[ensures: result > a] fails, line 2)\n"
                 "  half: unsupported: division by a float (line 7)\n"
                 "  __start__ (global initializers): proved (1 condition)\n"
                 "Result: 1 proved, 1 counterexample, 1 unsupported\n")
    prove_err = ("error[E-PROVE]: 'add': #[ensures: result > a] fails for a = 1, b = 0\n --> p.til:2:1\n"
                 "  2 | #[ensures: result > a]\n    | ^\n  = hint: confirmed on the TIL VM\n"
                 "ошибка[E-PROVE]: cannot prove 'half': unsupported\n --> p.til:7:14\n  = подсказка: see docs\n"
                 "Syntax Error: error: Expected RPAREN, got DEDENT\n --> q.til:4:1\n"
                 "warning: unused\n --> q.til:9:3\n")
    fit = {'program': 'g.til', 'device': {'name': 'z80-16', 'title': 'Z80'}, 'fits': False, 'verdict': 'does not fit',
           'errors': ['the integer constant 65521 (line 31 in checksum, line 32 in checksum) does not fit'],
           'warnings': ['screen natives need a screen'], 'notes': ['integers are 16-bit'],
           'resources': [{'resource': 'bytecode', 'needs': '1 KB', 'limit': '64 KB', 'status': 'ok', 'note': '',
                          'measured': None}]}
    script = (
        "const r = require(process.argv[1]);\n"
        "const [out, err, fit] = JSON.parse(process.argv[2]);\n"
        "console.log(JSON.stringify({summary: r.parseProveSummary(out), diags: r.parseToolDiagnostics(err),\n"
        "  fitDiags: r.fitDiagnostics(fit), fitText: r.fitReportText(fit), built: r.builtArtifact('Built: g.tbc (1111 bytes)'),\n"
        "  sev: ['counterexample', 'not proved', 'unsupported'].map(r.proofSeverity), fn: r.proofFunction(\"'Acc.add': x\")}));\n")
    r = subprocess.run([NODE, '-e', script, os.path.join(SRC, 'reports.js'), json.dumps([prove_out, prove_err, fit])],
                       capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    fns = out['summary']['functions']
    assert [(f['name'], f['status'], f['line']) for f in fns] == [
        ('add', 'counterexample', 2), ('half', 'unsupported', 7), ('__start__', 'proved', 0)]
    assert out['summary']['result'] == '1 proved, 1 counterexample, 1 unsupported'
    diags = out['diags']
    assert [(d['severity'], d['code'], d['file'], d['line'], d['col']) for d in diags] == [
        ('error', 'E-PROVE', 'p.til', 2, 1), ('error', 'E-PROVE', 'p.til', 7, 14), ('error', '', 'q.til', 4, 1),
        ('warning', '', 'q.til', 9, 3)]
    assert diags[0]['hint'] == 'confirmed on the TIL VM' and diags[1]['hint'] == 'see docs'
    assert [(d['severity'], d['line']) for d in out['fitDiags']] == [('error', 31), ('warning', 1), ('information', 1)]
    assert 'verdict: does not fit' in out['fitText'] and 'bytecode' in out['fitText']
    assert out['built'] == 'g.tbc'
    assert out['sev'] == ['error', 'warning', 'information'] and out['fn'] == 'Acc.add'


# ── the webviews' scripts against a fake DOM ────────────────────────────────

@needs_node
def test_webview_scripts():
    r = subprocess.run([NODE, os.path.join(EXT, 'test', 'webviews.js')], capture_output=True, text=True,
                       encoding='utf-8', timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    screen = out['screen']
    assert screen['ready'] == {'type': 'ready'} and screen['csp']
    assert screen['canvas'] == [4, 2] and screen['pixel5'] == [50, 200, 250, 255]   # RGB24 -> RGBA
    # 'Fit' in a 400 x 300 panel at devicePixelRatio 2: 196 device pixels per program pixel
    assert screen['cssWidth'] == '392px' and screen['cssHeight'] == '196px'
    assert screen['zoom3'] == '12px'
    assert not screen['hidden'] and screen['emptyHidden'] and not screen['saveDisabled']
    assert screen['info'].startswith('game') and screen['ended'].endswith('ended')
    assert screen['save']['type'] == 'savePng'
    tl = out['timeline']
    assert tl['ready'] == {'type': 'ready'} and tl['idleShown'] and tl['uiShown']
    assert tl['slider'] == {'max': '19', 'value': '7', 'disabled': False}
    assert tl['pos'] == 'line event 7 of 20'
    assert tl['posted'] == [{'type': 'goto', 'index': 3}, {'type': 'goto', 'index': 0}, {'type': 'goto', 'index': 19},
                            {'type': 'command', 'id': 'workbench.action.debug.stepBack'},
                            {'type': 'goto', 'index': 19}]                      # 99 is clamped to the history
    assert tl['completeMax'] == '20'                                              # complete: index = total is the end
    assert tl['runningDisabled'] and tl['state'] == 'running' and tl['idleAgain']
