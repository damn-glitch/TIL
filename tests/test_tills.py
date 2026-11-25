#!/usr/bin/env python3
"""
TIL language services (src/tills.py): diagnostics from the real compiler with
exact ranges, completion, hover, signature help, definition / references /
rename across files, symbols, folding, formatting, inlay hints, code actions,
code lenses, semantic tokens -- in English, Russian and Kazakh programs,
with UTF-16 positions (Cyrillic, emoji) and broken code.

Run: python3 -m pytest tests/test_tills.py -v
"""

import glob
import os
import sys
import time

import pytest

from tilhelpers import ROOT, til

sys.path.insert(0, os.path.join(ROOT, 'src'))
import tills  # noqa: E402

EXAMPLES = sorted(glob.glob(os.path.join(ROOT, 'examples', '**', '*.til'), recursive=True))
CORPUS_FILE = os.path.join(ROOT, 'tests', 'selfhost_corpus.txt')


def corpus():
    """(id, source) of every program of tests/selfhost_corpus.txt."""
    out, cur = [], None
    with open(CORPUS_FILE, encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line.startswith('#==== '):
                if cur is not None:
                    out.append((cur[0], '\n'.join(cur[1])))
                cur = [line.split()[1], []]
            elif cur is not None:
                cur[1].append(line)
    if cur is not None:
        out.append((cur[0], '\n'.join(cur[1])))
    return out


CORPUS = corpus()


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def open_doc(text, uri='untitled:test.til', locale='en', root=None, ws=None):
    ws = ws or tills.Workspace(root, locale)
    ws.open(uri, text, 1)
    return ws, uri


def at(text, needle, k=0, delta=0):
    """LSP (line, character) of the k-th occurrence of `needle` (+delta code points)."""
    i = -1
    for _ in range(k + 1):
        i = text.index(needle, i + 1)
    pm = tills.PositionMap(text)
    p = pm.position(i + delta)
    return p['line'], p['character']


def cursor(text):
    """Text without the `|` marker, and the marker's LSP position."""
    i = text.index('|')
    clean = text[:i] + text[i + 1:]
    p = tills.PositionMap(clean).position(i)
    return clean, p['line'], p['character']


def complete(text, locale='en', ws=None, uri='untitled:c.til'):
    clean, line, ch = cursor(text)
    ws, uri = open_doc(clean, uri, locale, ws=ws)
    return ws.completion(uri, line, ch)


def labels(items):
    return [i['label'] for i in sorted(items, key=lambda i: i.get('sortText', ''))]


def text_of(text, rng):
    pm = tills.PositionMap(text)
    return text[pm.offset(rng['start']['line'], rng['start']['character']):
                pm.offset(rng['end']['line'], rng['end']['character'])]


def diags(text, locale='en'):
    ws, uri = open_doc(text, locale=locale)
    return ws.diagnostics(uri)


def one_diag(text, locale='en'):
    ds = diags(text, locale)
    assert len(ds) == 1, ds
    d = ds[0]
    return d, text_of(text, d['range'])


def apply_edits(text, edits):
    pm = tills.PositionMap(text)
    spans = []
    for e in edits:
        r = e['range']
        spans.append((pm.offset(r['start']['line'], r['start']['character']),
                      pm.offset(r['end']['line'], r['end']['character']), e['newText']))
    for a, b, new in sorted(spans, reverse=True):
        text = text[:a] + new + text[b:]
    return text


def decode_tokens(text, data):
    """Semantic tokens -> [(text, type, modifiers)]."""
    pm = tills.PositionMap(text)
    out, line, ch = [], 0, 0
    for k in range(0, len(data), 5):
        dl, dc, length, ttype, mods = data[k:k + 5]
        line += dl
        ch = ch + dc if dl == 0 else dc
        start = pm.offset(line, ch)
        end = pm.offset(line, ch + length)
        mod_names = {tills.TOKEN_MODIFIERS[b] for b in range(len(tills.TOKEN_MODIFIERS)) if mods & (1 << b)}
        out.append((text[start:end], tills.TOKEN_TYPES[ttype], mod_names))
    return out


@pytest.fixture
def project(tmp_path):
    """A workspace folder with files that import each other."""
    files = {
        'geo.til': ("# A point in the plane\nstruct Point\n    x: float\n    y: float\n\nimpl Point\n"
                    "    fn new(x: float, y: float) -> Point\n        return Point { x: x, y: y }\n\n"
                    "    # Distance from the origin\n    fn norm(self) -> float\n"
                    "        return sqrt(self.x * self.x + self.y * self.y)\n\n"
                    "#[requires: n >= 0]\n#[ensures: result >= 1]\nfact(n: int) -> int\n    if n <= 1\n"
                    "        return 1\n    return n * fact(n - 1)\n"),
        'app.til': ("import geo\n\nmain()\n    let p = Point.new(3.0, 4.0)\n    let d = p.norm()\n"
                    "    print(d, fact(5), p.x)\n"),
        'other.til': "import geo\n\nshow(q: Point)\n    print(q.norm())\n",
        'lonely.til': "fact(n: int) -> int\n    return n\n",
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding='utf-8')
    ws = tills.Workspace(str(tmp_path))
    uris = {}
    for name in files:
        uris[name] = tills.path_to_uri(str(tmp_path / name))
    for name in ('app.til', 'geo.til'):
        ws.open(uris[name], files[name], 1)
    return ws, uris, files, tmp_path


# ═══════════════════════════════════════════════════════════════
# POSITIONS (UTF-16)
# ═══════════════════════════════════════════════════════════════

class TestPositionMap:
    def test_ascii_round_trip(self):
        pm = tills.PositionMap("ab\ncd\n")
        assert pm.position(4) == {'line': 1, 'character': 1}
        assert pm.offset(1, 1) == 4

    def test_cyrillic_is_one_unit_per_letter(self):
        pm = tills.PositionMap("пусть х = 1")
        assert pm.offset(0, 6) == 6
        assert pm.position(6) == {'line': 0, 'character': 6}

    def test_emoji_counts_two_utf16_units(self):
        text = 's = "😀x"'
        pm = tills.PositionMap(text)
        i = text.index('x')
        assert pm.position(i) == {'line': 0, 'character': i + 1}
        assert pm.offset(0, i + 1) == i

    def test_offset_inside_a_surrogate_pair_snaps_to_the_character(self):
        text = '"😀"'
        pm = tills.PositionMap(text)
        assert pm.offset(0, 2) == 1          # between the two halves of the emoji

    def test_crlf_lines(self):
        pm = tills.PositionMap("a\r\nbc\r\n")
        assert pm.line_count == 3
        assert pm.line_text(1) == "bc"
        assert pm.position(pm.offset(1, 2)) == {'line': 1, 'character': 2}

    def test_compiler_columns(self):
        pm = tills.PositionMap("x\n  😀y")
        assert pm.til_to_lsp(2, 4) == {'line': 1, 'character': 4}     # col 4 = 'y' (code points)
        assert pm.lsp_to_til(1, 4) == (2, 4)

    def test_clamping(self):
        pm = tills.PositionMap("ab")
        assert pm.offset(5, 0) == 2
        assert pm.offset(0, 99) == 2
        assert pm.position(99) == {'line': 0, 'character': 2}

    def test_uri_round_trip(self, tmp_path):
        p = str(tmp_path / "файл 1.til")
        uri = tills.path_to_uri(p)
        assert uri.startswith('file://') and ' ' not in uri
        assert tills.uri_to_path(uri) == p
        assert tills.uri_to_path('untitled:Untitled-1') is None


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

class TestDiagnostics:
    def test_clean_program_has_none(self):
        assert diags("main()\n    print(1)\n") == []

    @pytest.mark.parametrize('path', EXAMPLES, ids=lambda p: os.path.relpath(p, ROOT))
    def test_examples_are_clean(self, path):
        with open(path, encoding='utf-8') as f:
            text = f.read()
        ws = tills.Workspace(os.path.dirname(path))
        uri = tills.path_to_uri(path)
        ws.open(uri, text, 1)
        assert ws.diagnostics(uri) == []

    def test_agrees_with_the_compiler_on_the_whole_corpus(self):
        ws = tills.Workspace(None)
        for k, (pid, src) in enumerate(CORPUS):
            uri = f'untitled:corpus{k}.til'
            ws.open(uri, src, 1)
            errors = [d for d in ws.diagnostics(uri) if d['severity'] == 1]
            try:
                til.compile_to_bytecode(src, '<corpus>')
                ok = True
            except (til.CompileError, SyntaxError):
                ok = False
            assert ok == (not errors), (pid, errors)
            for d in errors:
                assert text_of(src, d['range']), (pid, d)        # never an empty range
                assert d['source'] == 'til'
            ws.close(uri)

    def test_unexpected_token_range_and_hint(self):
        d, seg = one_diag("main()\n    let x = 1 +\n")
        assert d['message'].startswith("Unexpected token: NEWLINE")
        assert d['data']['hint'] and d['message'].endswith(d['data']['hint'])
        assert seg == "+"
        assert 'code' not in d                        # the compiler gives syntax errors no code

    def test_unterminated_string_spans_from_the_quote(self):
        d, seg = one_diag('main()\n    let s = "abc\n')
        assert d['message'] == "Unterminated string"
        assert seg == '"abc'

    def test_unexpected_character_emoji(self):
        text = 'main()\n    let 😀 = 1\n'
        d, seg = one_diag(text)
        assert seg == '😀'
        assert d['range']['end']['character'] - d['range']['start']['character'] == 2

    def test_invalid_hex_literal(self):
        d, seg = one_diag("main()\n    print(0x)\n")
        assert "Invalid hexadecimal literal" in d['message'] and seg == "0x"

    def test_unclosed_bracket_points_at_the_bracket(self):
        text = "main()\n    let x = [1, 2\n    print(x)\n"
        d, seg = one_diag(text)
        assert seg == "[" and d['range']['start']['line'] == 1

    def test_error_inside_an_fstring_expression(self):
        text = 'main()\n    let a = 1\n    print(f"v={a +}")\n'
        d, seg = one_diag(text)
        assert d['range']['start']['line'] == 2
        line = text.split('\n')[2]
        assert line.index('{') < d['range']['start']['character'] < line.index('}') + 1

    def test_error_inside_a_contract(self):
        text = "#[requires: n >]\nf(n: int) -> int\n    return n\nmain()\n    print(f(1))\n"
        d, seg = one_diag(text)
        assert d['range']['start']['line'] == 0 and seg in (']', '>')

    def test_e_effect_pure_function(self):
        d, seg = one_diag("#[pure]\nf(x: int) -> int\n    print(x)\n    return x\nmain()\n    print(f(1))\n")
        assert d['code'] == 'E-EFFECT' and seg == 'print(x)'
        assert d['data']['hint'] == "remove #[pure] or declare #[effects: io]"

    def test_e_effect_declared_effects(self):
        d, seg = one_diag("#[effects: fs]\nf()\n    print(1)\nmain()\n    f()\n")
        assert d['code'] == 'E-EFFECT' and "declares only" in d['message']

    def test_e_cap(self):
        text = ("cap Net { send }\n#[cap: Net]\nfetch() -> int\n    return 1\nhelper() -> int\n    return fetch()\n"
                "main()\n    print(helper())\n")
        d, seg = one_diag(text)
        assert d['code'] == 'E-CAP' and seg == 'fetch'
        assert d['range']['start']['line'] == 5

    def test_e_contract(self):
        text = "#[requires: b != 0]\ndiv(a: int, b: int) -> int\n    return a / b\nmain()\n    print(div(10, 0))\n"
        d, seg = one_diag(text)
        assert d['code'] == 'E-CONTRACT' and seg == 'div'

    def test_e_arity(self):
        d, seg = one_diag("f(a: int) -> int\n    return a\nmain()\n    print(f(1, 2))\n")
        assert d['code'] == 'E-ARITY' and seg == 'f'

    def test_e_vm_unknown_name_with_suggestion(self):
        d, seg = one_diag("main()\n    let total = 1\n    print(totl)\n")
        assert d['code'] == 'E-VM' and seg == 'totl'
        assert d['data']['enHint'] == "did you mean 'total'?"

    def test_e_vm_unknown_function(self):
        d, seg = one_diag("main()\n    frobnicate(1)\n")
        assert d['code'] == 'E-VM' and seg == 'frobnicate'

    def test_e_vm_assign_twice(self):
        d, seg = one_diag("main()\n    let x = 1\n    x = 2\n")
        assert d['code'] == 'E-VM' and seg == 'x' and d['range']['start']['line'] == 2

    def test_e_vm_native_arity(self):
        d, seg = one_diag("main()\n    print(len())\n")
        assert d['code'] == 'E-VM' and seg == 'len'

    @pytest.mark.skipif(til.types_module() is None, reason="tiltypes is not installed")
    def test_e_type(self):
        d, seg = one_diag('main()\n    let x: int = "a"\n    print(x)\n')
        assert d['code'] == 'E-TYPE' and seg == 'x'

    def test_e_import_missing_module_is_a_warning(self):
        d, seg = one_diag("import nowhere_to_be_found\nmain()\n    print(1)\n")
        assert d['code'] == 'E-IMPORT' and d['severity'] == tills.SEV_WARNING
        assert seg == 'nowhere_to_be_found'

    def test_errors_in_an_imported_file(self, tmp_path):
        (tmp_path / 'mod.til').write_text("helper(x: int) -> int\n    return undefined_thing + x\n")
        (tmp_path / 'main.til').write_text("import mod\nmain()\n    print(helper(1))\n")
        ws = tills.Workspace(str(tmp_path))
        uri = tills.path_to_uri(str(tmp_path / 'main.til'))
        ws.open(uri, (tmp_path / 'main.til').read_text(), 1)
        by = ws.diagnostics_by_uri(uri)
        mod_uri = tills.path_to_uri(str(tmp_path / 'mod.til'))
        assert [d['code'] for d in by[mod_uri]] == ['E-VM']
        assert by[mod_uri][0]['range']['start'] == {'line': 1, 'character': 11}
        note = by[uri][0]
        assert note['severity'] == tills.SEV_INFO and 'mod' in note['message']
        assert note['range']['start']['line'] == 0

    def test_syntax_error_in_an_imported_file(self, tmp_path):
        (tmp_path / 'bad.til').write_text("helper(x: int) -> int\n    return x +\n")
        (tmp_path / 'main.til').write_text("import bad\nmain()\n    print(helper(1))\n")
        ws = tills.Workspace(str(tmp_path))
        uri = tills.path_to_uri(str(tmp_path / 'main.til'))
        ws.open(uri, (tmp_path / 'main.til').read_text(), 1)
        by = ws.diagnostics_by_uri(uri)
        bad = by[tills.path_to_uri(str(tmp_path / 'bad.til'))]
        assert bad[0]['message'].startswith("Unexpected token") and bad[0]['range']['start']['line'] == 1
        assert by[uri][0]['severity'] == tills.SEV_INFO

    def test_namespace_clash_is_e_import(self, tmp_path):
        (tmp_path / 'm1.til').write_text("dup(x: int) -> int\n    return x\n")
        (tmp_path / 'b.til').write_text("import m1\ndup(y: int) -> int\n    return y\nmain()\n    print(dup(1))\n")
        ws = tills.Workspace(str(tmp_path))
        uri = tills.path_to_uri(str(tmp_path / 'b.til'))
        ws.open(uri, (tmp_path / 'b.til').read_text(), 1)
        d = ws.diagnostics(uri)
        assert d[0]['code'] == 'E-IMPORT' and "defined in two files" in d[0]['message']
        assert d[0]['range']['start']['line'] == 1

    def test_russian_messages(self):
        d, _seg = one_diag("main()\n    let total = 1\n    print(totl)\n", locale='ru')
        assert d['message'].startswith("неизвестное имя 'totl'")
        assert "возможно" in d['message']
        assert d['data']['en'] == "unknown name 'totl'"

    def test_kazakh_messages(self):
        d, _seg = one_diag("main()\n    print(len())\n", locale='kk')
        assert "аргумент" in d['message']

    def test_locale_codes(self):
        assert tills.message_lang('ru-RU') == 'ru'
        assert tills.message_lang('kk') == 'kk'
        assert tills.message_lang('kk-latn') == 'kk'
        assert tills.message_lang('de') == 'en'

    def test_diagnostics_in_russian_keyword_file(self):
        d, seg = one_diag("#[lang: ru]\nглавная()\n    пусть х = 1\n    х = 2\n")
        assert seg == 'х' and d['code'] == 'E-VM'

    def test_utf16_range_after_emoji(self):
        text = 'main()\n    let s = "😀😀"\n    print(s, nope)\n'
        d, seg = one_diag(text)
        assert seg == 'nope'
        assert d['range']['start'] == {'line': 2, 'character': 13}

    def test_cached_per_version(self):
        ws, uri = open_doc("main()\n    print(1)\n")
        ws.diagnostics(uri)
        sem = ws.docs[uri].semantic
        ws.diagnostics(uri)
        assert ws.docs[uri].semantic is sem
        ws.change(uri, "main()\n    print(2)\n", 2)
        ws.diagnostics(uri)
        assert ws.docs[uri].semantic is not sem

    def test_crlf_document(self):
        text = "main()\r\n    let total = 1\r\n    print(totl)\r\n"
        ws, uri = open_doc(text)
        d = ws.diagnostics(uri)[0]
        assert d['range']['start'] == {'line': 2, 'character': 10}

    def test_classic_type_warnings_without_tiltypes(self, monkeypatch):
        monkeypatch.setenv('TIL_NO_TYPES', '1')
        ws, uri = open_doc('main()\n    let x: int = "a"\n    print(x)\n')
        ds = ws.diagnostics(uri)
        assert ds and all(d['severity'] == tills.SEV_WARNING for d in ds)
        assert "Type mismatch" in ds[0]['message']


# ═══════════════════════════════════════════════════════════════
# COMPLETION
# ═══════════════════════════════════════════════════════════════

class TestCompletion:
    SRC = ("struct Point\n    x: float\n    y: float\n\nimpl Point\n    fn new(x: float, y: float) -> Point\n"
           "        return Point { x: x, y: y }\n    fn norm(self) -> float\n        return self.x\n\n"
           "enum Color\n    Red\n    Green\n\ntwice(n: int) -> int\n    return n * 2\n\n")

    def test_keywords_builtins_and_names(self):
        items = complete(self.SRC + "main()\n    let total = 1\n    |\n")
        ls = labels(items)
        for name in ('total', 'twice', 'Point', 'Color', 'print', 'len', 'if', 'for', 'егер', 'None'):
            assert name in ls, name
        assert ls.index('total') < ls.index('twice') < ls.index('print') < ls.index('if')

    def test_locals_declared_later_are_not_offered(self):
        ls = labels(complete("main()\n    let a = 1\n    |\n    let later = 2\n"))
        assert 'a' in ls and 'later' not in ls

    def test_parameters(self):
        ls = labels(complete("f(alpha: int, beta: str) -> int\n    return al|\n"))
        assert ls[:2] == ['alpha', 'beta']

    def test_builtin_items_have_signature_docs_and_snippets(self):
        items = complete("main()\n    pr|\n")
        p = next(i for i in items if i['label'] == 'print')
        assert p['detail'] == 'print(values...)'
        assert p['insertTextFormat'] == tills.INSERT_SNIPPET and p['insertText'] == 'print($1)$0'
        assert 'Prints' in p['documentation']['value']
        clock = next(i for i in items if i['label'] == 'clock_ms')
        assert clock['insertText'] == 'clock_ms()$0'

    def test_user_function_detail(self):
        items = complete(self.SRC + "main()\n    tw|\n")
        t = next(i for i in items if i['label'] == 'twice')
        assert t['detail'] == 'twice(n: int) -> int' and t['kind'] == tills.CompletionKind.FUNCTION

    def test_struct_members_after_dot(self):
        ls = labels(complete(self.SRC + "main()\n    let p = Point.new(1.0, 2.0)\n    p.|\n"))
        assert set(ls) >= {'x', 'y', 'norm'} and 'push' not in ls and 'new' not in ls

    def test_static_functions_after_type_name(self):
        ls = labels(complete(self.SRC + "main()\n    let p = Point.|\n"))
        assert 'new' in ls and 'x' not in ls

    def test_enum_variants(self):
        assert labels(complete(self.SRC + "main()\n    let c = Color.|\n")) == ['Green', 'Red']

    def test_self_members(self):
        ls = labels(complete("struct P\n    a: int\nimpl P\n    fn get(self) -> int\n        return self.|\n"))
        assert 'a' in ls and 'get' in ls

    def test_string_methods(self):
        ls = labels(complete('main()\n    let s = "abc"\n    s.|\n'))
        assert 'to_upper' in ls and 'push' not in ls

    def test_array_methods(self):
        ls = labels(complete('main()\n    let xs = [1, 2]\n    xs.|\n'))
        assert 'push' in ls and 'to_upper' not in ls

    def test_chained_receiver_type(self):
        ls = labels(complete(self.SRC + "main()\n    let p = Point.new(1.0, 2.0)\n    p.norm().|\n"))
        assert ls == ['to_string']

    def test_field_chain_to_string_methods(self):
        ls = labels(complete('struct P\n    name: str\nmain()\n    let p = P { name: "a" }\n    p.name.|\n'))
        assert 'trim' in ls and 'name' not in ls

    def test_unknown_receiver_offers_everything(self):
        ls = labels(complete(self.SRC + "show(v)\n    v.|\n"))
        assert 'norm' in ls and 'push' in ls

    def test_module_members(self, project):
        ws, uris, files, root = project
        text = "import geo\nmain()\n    geo.|\n"
        clean, line, ch = cursor(text)
        uri = tills.path_to_uri(str(root / 'mod_use.til'))
        ws.open(uri, clean, 1)
        ls = labels(ws.completion(uri, line, ch))
        assert 'fact' in ls and 'Point' in ls and 'main' not in ls

    def test_imported_module_names(self, project):
        ws, uris, files, root = project
        ls = labels(ws.completion(uris['app.til'], 3, 4))
        assert 'geo' in ls

    def test_names_of_imported_files(self, project):
        ws, uris, files, root = project
        doc = ws.docs[uris['app.til']]
        text = doc.text + "    fa"
        ws.change(uris['app.til'], text, 5)
        pm = tills.PositionMap(text)
        last = pm.line_count - 1
        ls = labels(ws.completion(uris['app.til'], last, 6))
        assert 'fact' in ls and 'Point' in ls

    def test_import_module_names(self, project):
        ws, uris, files, root = project
        clean, line, ch = cursor("import |\n")
        uri = tills.path_to_uri(str(root / 'new.til'))
        ws.open(uri, clean, 1)
        ls = labels(ws.completion(uri, line, ch))
        assert 'geo' in ls and 'app' in ls and 'math' in ls        # math: the standard library

    def test_from_import_names(self, project):
        ws, uris, files, root = project
        clean, line, ch = cursor("from geo import |\n")
        uri = tills.path_to_uri(str(root / 'new2.til'))
        ws.open(uri, clean, 1)
        ls = labels(ws.completion(uri, line, ch))
        assert 'fact' in ls and 'Point' in ls

    def test_attribute_names(self):
        ls = labels(complete("#[|\nf() -> int\n    return 1\n"))
        assert set(ls) >= {'pure', 'effects', 'cap', 'requires', 'ensures', 'level', 'energy', 'lang'}

    def test_attribute_snippets(self):
        items = complete("#[|\n")
        req = next(i for i in items if i['label'] == 'requires')
        assert req['insertText'].startswith('requires: ') and req['insertTextFormat'] == tills.INSERT_SNIPPET

    @pytest.mark.parametrize('attr, expected', [
        ('lang', {'ru', 'kk-latn', 'kk'}),
        ('level', {'0', '1', '2', '3', '4'}),
        ('energy', {'low', 'balanced', 'performance'}),
        ('effects', {'io'}),
    ])
    def test_attribute_values(self, attr, expected):
        assert set(labels(complete(f"#[{attr}: |\n"))) >= expected

    def test_capability_names(self):
        ls = labels(complete("cap Storage { read }\n#[cap: |\nf()\n    print(1)\n"))
        assert 'Storage' in ls

    def test_contract_names(self):
        ls = labels(complete("#[ensures: re|]\nf(n: int) -> int\n    return n\n"))
        assert 'result' in ls and 'n' in ls

    def test_type_position(self):
        ls = labels(complete(self.SRC + "g(p: |) -> int\n    return 1\n"))
        assert {'int', 'str', 'Point', 'Color', 'Option'} <= set(ls) and 'print' not in ls

    def test_type_after_arrow(self):
        assert 'Point' in labels(complete(self.SRC + "g() -> |\n    return 1\n"))

    def test_let_annotation(self):
        assert 'float' in labels(complete("main()\n    let x: |\n"))

    def test_no_completion_in_strings_and_comments(self):
        assert complete('main()\n    let s = "ab|c"\n') == []
        assert complete('main()\n    # comm|ent\n') == []

    def test_fstring_expression(self):
        ls = labels(complete('main()\n    let count = 5\n    print(f"n={co|}")\n'))
        assert 'count' in ls

    def test_snippets_at_statement_start_only(self):
        items = complete("main()\n    |\n")
        snips = [i for i in items if i['kind'] == tills.CompletionKind.SNIPPET]
        assert {s['label'] for s in snips} >= {'for', 'while', 'match', 'retry', 'handle', 'fn', 'struct'}
        assert not [i for i in complete("main()\n    let x = |\n") if i['kind'] == tills.CompletionKind.SNIPPET]

    def test_snippets_in_the_files_language(self):
        items = complete("#[lang: ru]\nглавная()\n    |\n")
        loop = next(i for i in items if i['label'] == 'for' and i['kind'] == tills.CompletionKind.SNIPPET)
        assert loop['insertText'].startswith('для ') and ' в ' in loop['insertText']

    def test_russian_keywords_first_in_a_russian_file(self):
        items = complete("#[lang: ru]\nглавная()\n    |\n")
        kws = [i for i in sorted(items, key=lambda i: i['sortText']) if i['kind'] == tills.CompletionKind.KEYWORD]
        assert kws[0]['label'] in tills.til.RUSSIAN_KEYWORDS
        assert 'если' in [k['label'] for k in kws] and 'егер' in [k['label'] for k in kws]

    def test_kazakh_keywords_always_available(self):
        ls = labels(complete("main()\n    |\n"))
        assert 'егер' in ls and 'если' not in ls          # Russian needs #[lang: ru]

    def test_kazakh_file_gets_kazakh_first(self):
        items = complete("негізгі()\n    тұрақты а = 1\n    егер а > 0\n        басып_шығару(а)\n    |\n")
        kws = [i for i in sorted(items, key=lambda i: i['sortText']) if i['kind'] == tills.CompletionKind.KEYWORD]
        assert kws[0]['label'] in tills.til.KAZAKH_KEYWORDS

    def test_keyword_detail_lists_translations(self):
        items = complete("main()\n    |\n")
        kw = next(i for i in items if i['label'] == 'if')
        assert 'егер' in kw['detail'] and 'если' in kw['detail'] and 'eger' in kw['detail']

    def test_localized_documentation(self):
        items = complete("main()\n    |\n", locale='ru')
        p = next(i for i in items if i['label'] == 'print')
        assert 'Печатает' in p['documentation']['value']
        kw = next(i for i in items if i['label'] == 'while')
        assert 'Повторяет' in kw['documentation']['value']

    def test_kazakh_builtin_aliases(self):
        ls = labels(complete("main()\n    басып|\n"))
        assert 'басып_шығару' in ls

    def test_broken_line_keeps_scope(self):
        ls = labels(complete("f(alpha: int) -> int\n    let gamma = alpha +\n    return ga|\n"))
        assert 'alpha' in ls and 'gamma' in ls

    def test_function_without_a_body_yet(self):
        ls = labels(complete("f(alpha: int, beta: str) -> int\n    |\n"))
        assert ls[:2] == ['alpha', 'beta']

    def test_resolve_completion(self):
        ws = tills.Workspace(None)
        item = ws.resolve_completion({'label': 'len', 'data': {'tills': 'builtin', 'name': 'len'}})
        assert 'Length' in item['documentation']['value']
        assert ws.resolve_completion({'label': 'x'}) == {'label': 'x'}

    def test_fast_after_a_one_line_change(self):
        lines = [f"f{k}(a: int) -> int\n    let b = a + {k}\n    return b\n" for k in range(300)]
        text = ''.join(lines) + "main()\n    let p = [1, 2]\n    print(p)\n"
        ws, uri = open_doc(text)
        ws.diagnostics(uri)
        broken = text.replace("    print(p)\n", "    p.\n")
        ws.change(uri, broken, 2)
        line = broken.count('\n') - 1
        t0 = time.perf_counter()
        ls = labels(ws.completion(uri, line, 6))
        assert time.perf_counter() - t0 < 0.5
        assert 'push' in ls


# ═══════════════════════════════════════════════════════════════
# HOVER
# ═══════════════════════════════════════════════════════════════

class TestHover:
    def hover(self, text, needle, k=0, delta=0, locale='en'):
        ws, uri = open_doc(text, locale=locale)
        h = ws.hover(uri, *at(text, needle, k, delta))
        return h['contents']['value'] if h else None, h

    def test_local_with_inferred_type(self):
        md, h = self.hover("main()\n    let d = 2.5 * 2\n    print(d)\n", "d)")
        assert "let d: float" in md and "local variable" in md
        assert h['range']['start'] == {'line': 2, 'character': 10}

    def test_function_signature_doc_contracts_effects(self):
        text = ("# Factorial of n\n#[requires: n >= 0]\n#[effects: io]\nfact(n: int) -> int\n    print(n)\n"
                "    return 1\nmain()\n    print(fact(3))\n")
        md, _h = self.hover(text, "fact(3)")
        assert "fact(n: int) -> int" in md and "Factorial of n" in md
        assert "`n >= 0`" in md and "io" in md

    @pytest.mark.skipif(til.types_module() is None, reason="tiltypes is not installed")
    def test_inferred_parameter_and_return_types(self):
        text = "f(a: int, b) -> str\n    return str(a) + b\nlabel(x: int)\n    return str(x)\nmain()\n" \
               "    print(f(1, \"x\"), label(2))\n"
        md, _h = self.hover(text, "f(1")
        assert "f(a: int, b) -> str" in md
        md, _h = self.hover(text, "label(2")
        assert "label(x: int) -> str" in md

    def test_builtin(self):
        md, _h = self.hover("main()\n    print(len(\"x\"))\n", "len")
        assert "len(x) -> int" in md and "builtin" in md

    def test_builtin_alias_in_kazakh(self):
        md, _h = self.hover("негізгі()\n    басып_шығару(1)\n", "басып")
        assert "print" in md and "басып_шығару" in md

    def test_keyword_translations(self):
        md, _h = self.hover("main()\n    if true\n        print(1)\n", "if")
        assert "`if` = `егер` = `eger` = `если`" in md

    def test_kazakh_keyword(self):
        md, _h = self.hover("негізгі()\n    егер ақиқат\n        басып_шығару(1)\n", "егер", locale='kk')
        assert "**егер**" in md and "`if`" in md and "кілт сөз" in md

    def test_russian_hover_text(self):
        md, _h = self.hover("main()\n    while false\n        print(1)\n", "while", locale='ru')
        assert "ключевое слово" in md and "Повторяет" in md

    def test_attribute(self):
        md, _h = self.hover("#[pure]\nf() -> int\n    return 1\n", "pure")
        assert "#[pure]" in md

    def test_struct_method_and_field(self):
        text = ("struct P\n    # the x coordinate\n    x: int\nimpl P\n    fn twice(self) -> int\n"
                "        return self.x * 2\nmain()\n    let p = P { x: 1 }\n    print(p.twice(), p.x)\n")
        md, _h = self.hover(text, "twice()")
        assert "P.twice(self) -> int" in md and "method" in md
        md, _h = self.hover(text, "x)")
        assert "P.x: int" in md

    def test_self(self):
        md, _h = self.hover("struct P\n    x: int\nimpl P\n    fn get(self) -> int\n        return self.x\n", "self.x")
        assert "self: P" in md

    def test_module(self, project):
        ws, uris, files, root = project
        h = ws.hover(uris['app.til'], 0, 8)
        assert "import geo" in h['contents']['value']

    def test_cross_file_function(self, project):
        ws, uris, files, root = project
        h = ws.hover(uris['app.til'], *at(files['app.til'], "fact"))
        assert "fact(n: int) -> int" in h['contents']['value'] and "geo.til" in h['contents']['value']

    def test_nothing_on_whitespace(self):
        _md, h = self.hover("main()\n    print(1)\n", "    print", 0, 1)
        assert h is None


# ═══════════════════════════════════════════════════════════════
# SIGNATURE HELP
# ═══════════════════════════════════════════════════════════════

class TestSignatureHelp:
    def sig(self, text):
        clean, line, ch = cursor(text)
        ws, uri = open_doc(clean)
        return ws.signature_help(uri, line, ch)

    def test_user_function_active_parameter(self):
        s = self.sig("f(a: int, b: str) -> int\n    return a\nmain()\n    f(1, |)\n")
        assert s['signatures'][0]['label'] == "f(a: int, b: str) -> int"
        assert s['activeParameter'] == 1
        lbl = s['signatures'][0]['label']
        a, b = s['signatures'][0]['parameters'][1]['label']
        assert lbl[a:b] == "b: str"

    def test_builtin(self):
        s = self.sig("main()\n    draw_line(1, 2, |)\n")
        assert s['signatures'][0]['label'].startswith("draw_line(x0, y0, x1, y1, color)")
        assert s['activeParameter'] == 2

    def test_nested_call(self):
        s = self.sig("main()\n    print(len(|))\n")
        assert s['signatures'][0]['label'].startswith("len(")

    def test_method(self):
        s = self.sig("struct P\n    x: int\nimpl P\n    fn add(self, k: int) -> int\n        return self.x + k\n"
                     "main()\n    let p = P { x: 1 }\n    p.add(|)\n")
        assert s['signatures'][0]['label'] == "P.add(k: int) -> int"

    def test_builtin_method(self):
        s = self.sig('main()\n    let s = "a"\n    s.replace("a", |)\n')
        assert s['signatures'][0]['label'].startswith(".replace(old, new)") and s['activeParameter'] == 1

    def test_struct_constructor(self):
        s = self.sig("struct P\n    x: int\n    y: int\nmain()\n    let p = P(1, |)\n")
        assert s['signatures'][0]['label'] == "P(x: int, y: int)"

    def test_commas_in_strings_do_not_count(self):
        s = self.sig('f(a: str, b: int) -> int\n    return b\nmain()\n    f("x, y", |)\n')
        assert s['activeParameter'] == 1

    def test_outside_a_call(self):
        assert self.sig("main()\n    let x = 1|\n") is None


# ═══════════════════════════════════════════════════════════════
# DEFINITION / REFERENCES / HIGHLIGHTS
# ═══════════════════════════════════════════════════════════════

class TestDefinition:
    def defn(self, text, needle, k=0, delta=0):
        ws, uri = open_doc(text)
        locs = ws.definition(uri, *at(text, needle, k, delta))
        return [text_of(text, l['range']) for l in locs], locs

    def test_local(self):
        text = "main()\n    let x = 1\n    print(x)\n"
        segs, locs = self.defn(text, "x)")
        assert segs == ['x'] and locs[0]['range']['start'] == {'line': 1, 'character': 8}

    def test_shadowing_is_scope_correct(self):
        text = "main()\n    let x = 1\n    if true\n        let x = 2\n        print(x)\n    print(x)\n"
        _s, inner = self.defn(text, "print(x)", 0, 6)
        _s, outer = self.defn(text, "print(x)", 1, 6)
        assert inner[0]['range']['start']['line'] == 3 and outer[0]['range']['start']['line'] == 1

    def test_initializer_sees_the_previous_binding(self):
        text = "main()\n    let x = 1\n    if true\n        let x = x + 1\n        print(x)\n"
        _s, locs = self.defn(text, "x + 1")
        assert locs[0]['range']['start']['line'] == 1

    def test_parameter_and_global(self):
        text = "var counter = 0\nbump(step: int)\n    counter += step\nmain()\n    bump(1)\n"
        _s, p = self.defn(text, "step\n")
        assert p[0]['range']['start'] == {'line': 1, 'character': 5}
        _s, g = self.defn(text, "counter +=")
        assert g[0]['range']['start'] == {'line': 0, 'character': 4}

    def test_function(self):
        text = "helper() -> int\n    return 1\nmain()\n    print(helper())\n"
        segs, locs = self.defn(text, "helper()", 1)
        assert locs[0]['range']['start'] == {'line': 0, 'character': 0} and segs == ['helper']

    def test_struct_field_method_and_init(self):
        text = ("struct P\n    x: int\nimpl P\n    fn get(self) -> int\n        return self.x\n"
                "main()\n    let p = P { x: 1 }\n    print(p.get(), p.x)\n")
        _s, f = self.defn(text, "x)")
        assert f[0]['range']['start'] == {'line': 1, 'character': 4}
        _s, m = self.defn(text, "get()")
        assert m[0]['range']['start'] == {'line': 3, 'character': 7}
        _s, i = self.defn(text, "x: 1")
        assert i[0]['range']['start'] == {'line': 1, 'character': 4}
        _s, t = self.defn(text, "P {")
        assert t[0]['range']['start'] == {'line': 0, 'character': 7}

    def test_type_in_annotation(self):
        text = "struct P\n    x: int\nshow(p: P) -> P\n    return p\n"
        _s, locs = self.defn(text, "P)")
        assert locs[0]['range']['start'] == {'line': 0, 'character': 7}

    def test_enum_variant(self):
        text = "enum Color\n    Red\n    Green\nmain()\n    print(Color.Green)\n"
        _s, locs = self.defn(text, "Green)")
        assert locs[0]['range']['start'] == {'line': 2, 'character': 4}

    def test_for_lambda_comprehension_and_match(self):
        text = ("main()\n    for i in 0..3\n        print(i)\n    let f = |v| v * 2\n"
                "    let sq = [w * w for w in 0..3]\n    match 5\n        n if n > 1 => print(n)\n"
                "        _ => print(0)\n")
        _s, a = self.defn(text, "print(i)", 0, 6)
        assert a[0]['range']['start'] == {'line': 1, 'character': 8}
        _s, b = self.defn(text, "v * 2")
        assert b[0]['range']['start'] == {'line': 3, 'character': 13}
        _s, c = self.defn(text, "w * w")
        assert c[0]['range']['start']['character'] == text.split('\n')[4].index('for w') + 4
        _s, d = self.defn(text, "print(n)", 0, 6)
        assert d[0]['range']['start'] == {'line': 6, 'character': 8}

    def test_retry_error_and_ensures_result(self):
        text = ("#[ensures: result >= 0]\nf(n: int) -> int\n    return n\nmain()\n    let r = retry(2)\n"
                "        f(1)\n    else\n        print(error)\n        0\n")
        ws, uri = open_doc(text)
        assert ws.definition(uri, *at(text, "error)")) == []        # implicit: no source
        assert "error" in ws.hover(uri, *at(text, "error)"))['contents']['value']
        assert "result" in ws.hover(uri, *at(text, "result"))['contents']['value']

    def test_names_inside_fstrings_and_contracts(self):
        text = "#[requires: limit > 0]\nf(limit: int) -> int\n    return limit\nmain()\n    let who = 1\n" \
               "    print(f\"hi {who}\")\n"
        _s, a = self.defn(text, "who}")
        assert a[0]['range']['start'] == {'line': 4, 'character': 8}
        _s, b = self.defn(text, "limit > 0")
        assert b[0]['range']['start'] == {'line': 1, 'character': 2}

    def test_effect_with_a_default_handler(self):
        text = "Log(msg: str)\n    print(msg)\nmain()\n    perform Log(\"x\")\n"
        _s, locs = self.defn(text, "Log(\"x")
        assert locs[0]['range']['start'] == {'line': 0, 'character': 0}

    def test_across_files(self, project):
        ws, uris, files, root = project
        locs = ws.definition(uris['app.til'], *at(files['app.til'], "norm()"))
        assert locs[0]['uri'] == uris['geo.til'] and locs[0]['range']['start'] == {'line': 10, 'character': 7}
        locs = ws.definition(uris['app.til'], *at(files['app.til'], "Point"))
        assert locs[0]['uri'] == uris['geo.til'] and locs[0]['range']['start']['line'] == 1

    def test_import_goes_to_the_module_file(self, project):
        ws, uris, files, root = project
        locs = ws.definition(uris['app.til'], 0, 8)
        assert locs == [{'uri': uris['geo.til'], 'range': {'start': {'line': 0, 'character': 0},
                                                           'end': {'line': 0, 'character': 0}}}]

    def test_builtins_have_no_definition(self):
        _s, locs = self.defn("main()\n    print(1)\n", "print")
        assert locs == []

    def test_kazakh_program(self):
        text = "қосу(а: int, б: int) -> int\n    қайтару а + б\nнегізгі()\n    басып_шығару(қосу(1, 2))\n"
        segs, locs = self.defn(text, "қосу(1")
        assert segs == ['қосу'] and locs[0]['range']['start'] == {'line': 0, 'character': 0}


class TestReferences:
    def test_local_references(self):
        text = "main()\n    var n = 0\n    n += 1\n    print(n)\n"
        ws, uri = open_doc(text)
        refs = ws.references(uri, *at(text, "n += "))
        assert [r['range']['start']['line'] for r in refs] == [1, 2, 3]
        refs = ws.references(uri, *at(text, "n += "), include_declaration=False)
        assert [r['range']['start']['line'] for r in refs] == [2, 3]

    def test_function_across_files(self, project):
        ws, uris, files, root = project
        refs = ws.references(uris['geo.til'], *at(files['geo.til'], "fact(n"))
        where = sorted((r['uri'].rsplit('/', 1)[1], r['range']['start']['line']) for r in refs)
        assert where == [('app.til', 5), ('geo.til', 15), ('geo.til', 18)]   # not lonely.til's own fact

    def test_method_across_files(self, project):
        ws, uris, files, root = project
        refs = ws.references(uris['geo.til'], *at(files['geo.til'], "norm(self"))
        files_hit = sorted(r['uri'].rsplit('/', 1)[1] for r in refs)
        assert files_hit == ['app.til', 'geo.til', 'other.til']

    def test_field_references(self):
        text = "struct P\n    x: int\nmain()\n    let p = P { x: 1 }\n    print(p.x)\n"
        ws, uri = open_doc(text)
        refs = ws.references(uri, *at(text, "x: int"))
        assert sorted(r['range']['start']['line'] for r in refs) == [1, 3, 4]

    def test_struct_type_references(self):
        text = "struct P\n    x: int\nimpl P\n    fn new() -> P\n        return P { x: 0 }\nshow(p: P)\n    print(p.x)\n"
        ws, uri = open_doc(text)
        refs = ws.references(uri, *at(text, "P\n"))
        assert len(refs) == 5

    def test_document_highlights(self):
        text = "main()\n    var n = 0\n    n = n + 1\n"
        ws, uri = open_doc(text)
        hl = ws.document_highlights(uri, *at(text, "n + 1"))
        kinds = sorted((h['range']['start']['line'], h['range']['start']['character'], h['kind']) for h in hl)
        assert kinds == [(1, 8, 3), (2, 4, 3), (2, 8, 2)]


# ═══════════════════════════════════════════════════════════════
# RENAME
# ═══════════════════════════════════════════════════════════════

class TestRename:
    def rename(self, text, needle, new, k=0, delta=0):
        ws, uri = open_doc(text)
        edit = ws.rename(uri, *at(text, needle, k, delta), new)
        return apply_edits(text, edit['changes'].get(uri, []))

    def test_local(self):
        text = "main()\n    let total = 1\n    print(total)\n    print(\"total\")  # total\n"
        out = self.rename(text, "total =", "amount")
        assert out == "main()\n    let amount = 1\n    print(amount)\n    print(\"total\")  # total\n"

    def test_parameter_only_in_its_function(self):
        text = "f(x: int) -> int\n    return x + 1\ng(x: int) -> int\n    return x\n"
        out = self.rename(text, "x + 1", "y")
        assert out == "f(y: int) -> int\n    return y + 1\ng(x: int) -> int\n    return x\n"

    def test_shadowed_inner_binding(self):
        text = "main()\n    let a = 1\n    if true\n        let a = 2\n        print(a)\n    print(a)\n"
        out = self.rename(text, "a = 2", "b")
        assert out == "main()\n    let a = 1\n    if true\n        let b = 2\n        print(b)\n    print(a)\n"

    def test_method_and_field(self):
        text = ("struct P\n    x: int\nimpl P\n    fn get(self) -> int\n        return self.x\n"
                "main()\n    let p = P { x: 1 }\n    print(p.get() + p.x)\n")
        out = self.rename(text, "get(self", "value")
        assert out.count("value") == 2 and "get" not in out
        out = self.rename(text, "x: int", "pos")
        assert out.count("pos") == 4 and "self.pos" in out and "P { pos: 1 }" in out

    def test_enum_variant(self):
        text = "enum C\n    Red\nmain()\n    print(C.Red)\n"
        assert self.rename(text, "Red\n", "Crimson") == "enum C\n    Crimson\nmain()\n    print(C.Crimson)\n"

    def test_struct_everywhere(self):
        text = "struct P\n    x: int\nimpl P\n    fn new() -> P\n        return P { x: 0 }\n"
        assert self.rename(text, "P\n", "Q").count("Q") == 4

    def test_across_files(self, project):
        ws, uris, files, root = project
        edit = ws.rename(uris['geo.til'], *at(files['geo.til'], "fact(n"), "factorial")
        changed = {u.rsplit('/', 1)[1]: len(e) for u, e in edit['changes'].items()}
        assert changed == {'geo.til': 2, 'app.til': 1}
        assert apply_edits(files['app.til'], edit['changes'][uris['app.til']]).count("factorial(5)") == 1

    def test_method_across_files(self, project):
        ws, uris, files, root = project
        edit = ws.rename(uris['app.til'], *at(files['app.til'], "norm()"), "length")
        assert sorted(u.rsplit('/', 1)[1] for u in edit['changes']) == ['app.til', 'geo.til', 'other.til']

    def test_russian_program(self):
        text = "#[lang: ru]\nглавная()\n    пусть счёт = 1\n    печать(счёт)\n"
        out = self.rename(text, "счёт =", "итог")
        assert out == "#[lang: ru]\nглавная()\n    пусть итог = 1\n    печать(итог)\n"

    def test_utf16_edit_ranges(self):
        text = 'main()\n    let s = "😀"; let q = 1\n'.replace('; let q = 1', '')
        text = 'main()\n    let a = "😀😀"\n    let b = a\n    print(a, b)\n'
        out = self.rename(text, "a, b", "c")
        assert out == 'main()\n    let c = "😀😀"\n    let b = c\n    print(c, b)\n'

    @pytest.mark.parametrize('text, needle', [
        ("main()\n    if true\n        print(1)\n", "if"),
        ("main()\n    print(1)\n", "print"),
        ("main()\n    print(1)\n", "main"),
        ("struct P\n    x: int\nimpl P\n    fn get(self) -> int\n        return self.x\n", "self.x"),
        ("main()\n    let s = \"a\"\n    print(s.len())\n", "len()"),
        ("main()\n    print(unknown_thing)\n", "unknown_thing"),
    ], ids=['keyword', 'builtin', 'main', 'self', 'builtin-method', 'undefined'])
    def test_refused(self, text, needle):
        ws, uri = open_doc(text)
        with pytest.raises(tills.RenameError):
            ws.rename(uri, *at(text, needle), "other")

    def test_refuses_a_module(self, project):
        ws, uris, files, root = project
        with pytest.raises(tills.RenameError) as e:
            ws.prepare_rename(uris['app.til'], 0, 8)
        assert "module" in str(e.value)

    @pytest.mark.parametrize('new', ['1abc', 'a-b', '', 'if', 'для', 'print', 'Some', 'печать'])
    def test_refuses_bad_new_names(self, new):
        text = "main()\n    let x = 1\n    print(x)\n"
        ws, uri = open_doc(text)
        with pytest.raises(tills.RenameError):
            ws.rename(uri, *at(text, "x ="), new)

    def test_refuses_a_clash_in_scope(self):
        text = "main()\n    let a = 1\n    let b = 2\n    print(a + b)\n"
        ws, uri = open_doc(text)
        with pytest.raises(tills.RenameError) as e:
            ws.rename(uri, *at(text, "a ="), "b")
        assert "already defined" in str(e.value)

    def test_refuses_capturing_a_global(self):
        text = "var limit = 3\nf() -> int\n    let x = 1\n    return x + limit\n"
        ws, uri = open_doc(text)
        with pytest.raises(tills.RenameError):
            ws.rename(uri, *at(text, "x ="), "limit")

    def test_refuses_a_top_level_clash_across_files(self, project):
        ws, uris, files, root = project
        with pytest.raises(tills.RenameError):
            ws.rename(uris['geo.til'], *at(files['geo.til'], "fact(n"), "Point")

    def test_refuses_a_member_clash(self):
        text = "struct P\n    x: int\n    y: int\n"
        ws, uri = open_doc(text)
        with pytest.raises(tills.RenameError):
            ws.rename(uri, *at(text, "x:"), "y")

    def test_localized_refusal(self):
        text = "main()\n    print(1)\n"
        ws, uri = open_doc(text, locale='ru')
        with pytest.raises(tills.RenameError) as e:
            ws.rename(uri, *at(text, "print"), "x")
        assert "нельзя переименовать" in str(e.value)

    def test_prepare_rename(self):
        text = "main()\n    let total = 1\n    print(total)\n"
        ws, uri = open_doc(text)
        r = ws.prepare_rename(uri, *at(text, "total)", 0, 2))
        assert r['placeholder'] == 'total' and r['range']['start'] == {'line': 2, 'character': 10}
        assert ws.prepare_rename(uri, 0, 6) is None                 # on ')'


# ═══════════════════════════════════════════════════════════════
# SYMBOLS
# ═══════════════════════════════════════════════════════════════

class TestSymbols:
    SRC = ("const MAX = 10\nstruct P\n    x: int\n    y: int\nenum C\n    Red\n    Blue\ntrait Shape\n"
           "    area(self) -> int\nimpl P\n    fn sum(self) -> int\n        return self.x + self.y\n"
           "f(a: int) -> int\n    return a\nmain()\n    print(f(1))\n")

    def test_document_symbols_hierarchy(self):
        ws, uri = open_doc(self.SRC)
        syms = ws.document_symbols(uri)
        names = [(s['name'], s['kind']) for s in syms]
        assert names == [('MAX', tills.SymbolKind.CONSTANT), ('P', tills.SymbolKind.STRUCT),
                         ('C', tills.SymbolKind.ENUM), ('Shape', tills.SymbolKind.INTERFACE),
                         ('impl P', tills.SymbolKind.CLASS), ('f', tills.SymbolKind.FUNCTION),
                         ('main', tills.SymbolKind.FUNCTION)]
        assert [c['name'] for c in syms[1]['children']] == ['x', 'y']
        assert [c['name'] for c in syms[2]['children']] == ['Red', 'Blue']
        assert [c['name'] for c in syms[4]['children']] == ['sum']
        assert syms[5]['detail'] == 'f(a: int) -> int'

    def test_selection_range_inside_range(self):
        ws, uri = open_doc(self.SRC)

        def inside(s):
            r, sel = s['range'], s['selectionRange']
            key = lambda p: (p['line'], p['character'])  # noqa: E731
            assert key(r['start']) <= key(sel['start']) and key(sel['end']) <= key(r['end'])
            for c in s.get('children', []):
                inside(c)
        for s in ws.document_symbols(uri):
            inside(s)

    def test_function_range_covers_its_body(self):
        ws, uri = open_doc(self.SRC)
        f = next(s for s in ws.document_symbols(uri) if s['name'] == 'f')
        assert f['range']['start']['line'] == 12 and f['range']['end']['line'] == 13

    def test_attributes_and_doc_comments_belong_to_the_function(self):
        text = "# doc\n#[pure]\nf() -> int\n    return 1\n"
        ws, uri = open_doc(text)
        assert ws.document_symbols(uri)[0]['range']['start']['line'] == 1

    def test_workspace_symbols(self, project):
        ws, uris, files, root = project
        found = ws.workspace_symbols("nrm")
        assert [(s['name'], s['containerName']) for s in found] == [('norm', 'Point')]
        assert len(ws.workspace_symbols("fact")) == 2              # geo.til and lonely.til
        assert ws.workspace_symbols("")

    def test_kazakh_symbols(self):
        text = "құрылым Нүкте\n    х: int\nнегізгі()\n    басып_шығару(1)\n"
        ws, uri = open_doc(text)
        assert [s['name'] for s in ws.document_symbols(uri)] == ['Нүкте', 'main']


# ═══════════════════════════════════════════════════════════════
# FOLDING
# ═══════════════════════════════════════════════════════════════

class TestFolding:
    def test_blocks_comments_brackets_imports(self):
        text = ("import a\nimport b\n# one\n# two\n# three\nf(x: int) -> int\n    if x > 0\n        return 1\n"
                "    return 0\nmain()\n    let xs = [\n        1,\n        2,\n    ]\n    print(xs)\n")
        ws, uri = open_doc(text)
        got = {(r['startLine'], r['endLine'], r.get('kind')) for r in ws.folding_ranges(uri)}
        assert (0, 1, 'imports') in got
        assert (2, 4, 'comment') in got
        assert (5, 8, None) in got and (6, 7, None) in got
        assert (10, 12, 'region') in got

    def test_no_single_line_ranges(self):
        ws, uri = open_doc("main()\n    print(1)\n")
        for r in ws.folding_ranges(uri):
            assert r['endLine'] > r['startLine']


# ═══════════════════════════════════════════════════════════════
# FORMATTING
# ═══════════════════════════════════════════════════════════════

def fmt(text, tab=4):
    ws, uri = open_doc(text)
    return apply_edits(text, ws.formatting(uri, tab))


class TestFormatting:
    def test_indentation_to_four_spaces(self):
        assert fmt("main()\n  if true\n\tprint(1)\n") is not None
        assert fmt("main()\n  let x = 1\n  if x > 0\n     print(x)\n") == \
            "main()\n    let x = 1\n    if x > 0\n        print(x)\n"

    def test_operator_and_comma_spacing(self):
        assert fmt("main()\n    let x=1+2*3\n    print(x ,x,x)\n") == \
            "main()\n    let x = 1 + 2 * 3\n    print(x, x, x)\n"

    def test_unary_minus_and_generics_untouched(self):
        src = "main()\n    let a = -1\n    let v = Vec<int>.new()\n    let m = HashMap<str, int>.new()\n" \
              "    print(a * -2, [-1, 2], 0..10)\n"
        assert fmt(src) == src

    def test_strings_and_comments_untouched(self):
        src = 'main()\n    print("a+b ,c")   # x=1  ,y\n    print(f"{1+2}")\n'
        assert fmt(src) == src

    def test_blank_lines_trailing_whitespace_final_newline(self):
        assert fmt("main()   \n    print(1)\n\n\n\n\nf()\n    print(2)\n\n\n") == \
            "main()\n    print(1)\n\n\nf()\n    print(2)\n"
        assert fmt("main()\n    print(1)") == "main()\n    print(1)\n"

    def test_continuation_lines_keep_their_alignment(self):
        src = "main()\n  print(1,\n        2)\n"
        assert fmt(src) == "main()\n    print(1,\n          2)\n"

    def test_comment_lines_follow_the_code(self):
        assert fmt("main()\n  # note\n  print(1)\n") == "main()\n    # note\n    print(1)\n"

    def test_tab_size(self):
        assert fmt("main()\n    print(1)\n", tab=2) == "main()\n  print(1)\n"

    def test_lambda_match_and_arrows(self):
        assert fmt("f(x: int)->int\n    let g = |y| y*2\n    return match x\n        0=>1\n        _ =>g(x)\n") == \
            "f(x: int) -> int\n    let g = |y| y * 2\n    return match x\n        0 => 1\n        _ => g(x)\n"

    def test_russian_keywords(self):
        assert fmt("#[lang: ru]\nглавная()\n  пусть а=1\n  печать(а)\n") == \
            "#[lang: ru]\nглавная()\n    пусть а = 1\n    печать(а)\n"

    def test_nothing_when_it_does_not_lex(self):
        ws, uri = open_doc('main()\n  print("abc)\n')
        assert ws.formatting(uri) == []

    def test_already_formatted_gives_no_edits(self):
        ws, uri = open_doc("main()\n    print(1)\n")
        assert ws.formatting(uri) == []

    def test_minimal_edits(self):
        ws, uri = open_doc("main()\n    print(1)\n    let x=1\n    print(x)\n")
        edits = ws.formatting(uri)
        assert len(edits) == 1 and edits[0]['range']['start']['line'] == 2

    def test_crlf_document_keeps_crlf(self):
        ws, uri = open_doc("main()\r\n  print(1)\r\n")
        edits = ws.formatting(uri)
        assert edits and all('\r\n' in e['newText'] or not e['newText'] for e in edits)

    @pytest.mark.parametrize('name, src', [(os.path.relpath(p, ROOT), open(p, encoding='utf-8').read())
                                           for p in EXAMPLES] + [(f"corpus-{pid}", s) for pid, s in CORPUS],
                             ids=lambda x: x if isinstance(x, str) and len(x) < 60 else None)
    def test_idempotent_and_meaning_preserving(self, name, src):
        out = tills.format_source(src)
        if out is None:                     # only a program the lexer rejects is left alone
            with pytest.raises(SyntaxError):
                til.Lexer(src, "<t>").tokenize()
            return
        assert tills.format_source(out) == out
        a = til.Lexer(src, "<t>").tokenize()
        b = til.Lexer(out, "<t>").tokenize()
        assert tills._same_tokens(a, b)
        try:
            before = til.compile_to_bytecode(src, os.path.join(ROOT, 'examples', 'x.til')).serialize(strip=True)
        except (til.CompileError, SyntaxError):
            return
        after = til.compile_to_bytecode(out, os.path.join(ROOT, 'examples', 'x.til')).serialize(strip=True)
        assert before == after

    def test_selfhost_compiler_is_already_formatted(self):
        with open(os.path.join(ROOT, 'selfhost', 'tilc.til'), encoding='utf-8') as f:
            src = f.read()
        out = tills.format_source(src)
        assert out is not None and [l.rstrip() for l in out.split('\n')] == \
            [l.rstrip() for l in src.split('\n')][:len(out.split('\n'))]


# ═══════════════════════════════════════════════════════════════
# INLAY HINTS
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(til.types_module() is None, reason="tiltypes is not installed")
class TestInlayHints:
    def test_let_var_for_and_return_types(self):
        text = ("twice(x: int)\n    return x * 2\nmain()\n    let a = 1\n    var s = \"x\"\n"
                "    let b: int = 2\n    for i in 0..3\n        print(i)\n    print(a, s, b, twice(1))\n")
        ws, uri = open_doc(text)
        hints = {(h['position']['line'], h['label']) for h in ws.inlay_hints(uri, None)}
        assert hints == {(0, ' -> int'), (3, ': int'), (4, ': str'), (6, ': int')}

    def test_position_is_after_the_name(self):
        text = "main()\n    let total = 2.5\n    print(total)\n"
        ws, uri = open_doc(text)
        h = ws.inlay_hints(uri, None)[0]
        assert h['position'] == {'line': 1, 'character': 13} and h['label'] == ': float'

    def test_range_filter(self):
        text = "main()\n    let a = 1\n    let b = 2\n    print(a + b)\n"
        ws, uri = open_doc(text)
        rng = {'start': {'line': 2, 'character': 0}, 'end': {'line': 3, 'character': 0}}
        assert [h['position']['line'] for h in ws.inlay_hints(uri, rng)] == [2]

    def test_struct_type_and_none_for_unknown(self):
        text = "struct P\n    x: int\nmain()\n    let p = P { x: 1 }\n    print(p.x)\n"
        ws, uri = open_doc(text)
        assert [h['label'] for h in ws.inlay_hints(uri, None)] == [': P']

    def test_no_hints_for_broken_code(self):
        ws, uri = open_doc("main()\n    let a = 1 +\n")
        assert ws.inlay_hints(uri, None) == []


# ═══════════════════════════════════════════════════════════════
# CODE ACTIONS
# ═══════════════════════════════════════════════════════════════

def actions_for(text, locale='en', root=None, uri='untitled:a.til', ws=None):
    ws, uri = open_doc(text, uri=uri, locale=locale, root=root, ws=ws)
    ds = ws.diagnostics(uri)
    out = []
    for d in ds:
        out += ws.code_actions(uri, d['range'], [d])
    return ws, uri, [a for a in out if a['kind'] == 'quickfix']


class TestCodeActions:
    def test_let_to_var(self):
        text = "main()\n    let x = 1\n    x = 2\n"
        ws, uri, acts = actions_for(text)
        assert acts[0]['title'] == "Change 'let' to 'var'" and acts[0]['isPreferred']
        fixed = apply_edits(text, acts[0]['edit']['changes'][uri])
        assert fixed == "main()\n    var x = 1\n    x = 2\n"
        assert ws.diagnostics(uri) and tills.Workspace(None).diagnostics(uri) == []

    @pytest.mark.parametrize('text, new', [
        ("негізгі()\n    тұрақты х = 1\n    х = 2\n", "айнымалы"),
        ("#[lang: ru]\nглавная()\n    пусть х = 1\n    х = 2\n", "перем"),
        ("#[lang: kk-latn]\nnegizgi()\n    turaqty x = 1\n    x = 2\n", "aynymaly"),
    ], ids=['kk', 'ru', 'kk-latn'])
    def test_let_to_var_in_every_language(self, text, new):
        _ws, uri, acts = actions_for(text)
        assert apply_edits(text, acts[0]['edit']['changes'][uri]).count(new) == 1

    def test_global_let_to_var(self):
        text = "let g = 1\nmain()\n    g = 2\n"
        _ws, uri, acts = actions_for(text)
        assert apply_edits(text, acts[0]['edit']['changes'][uri]).startswith("var g = 1")

    def test_did_you_mean(self):
        text = "main()\n    let total = 1\n    print(totl)\n"
        ws, uri, acts = actions_for(text)
        assert acts[0]['title'] == "Replace 'totl' with 'total'"
        assert apply_edits(text, acts[0]['edit']['changes'][uri]) == "main()\n    let total = 1\n    print(total)\n"

    def test_add_missing_import(self, project):
        ws, uris, files, root = project
        text = "main()\n    print(fact(3))\n"
        uri = tills.path_to_uri(str(root / 'needs_import.til'))
        _ws, uri, acts = actions_for(text, uri=uri, ws=ws)
        titles = [a['title'] for a in acts]
        assert "Add an import of 'geo'" in titles and "Add an import of 'lonely'" in titles
        act = acts[titles.index("Add an import of 'geo'")]
        fixed = apply_edits(text, act['edit']['changes'][uri])
        assert fixed == "import geo\nmain()\n    print(fact(3))\n"

    def test_import_goes_after_the_existing_imports(self, project):
        ws, uris, files, root = project
        (root / 'util.til').write_text("helper2() -> int\n    return 1\n")
        ws.files_changed()
        text = "import geo\n\nmain()\n    print(helper2())\n"
        uri = tills.path_to_uri(str(root / 'needs2.til'))
        _ws, uri, acts = actions_for(text, uri=uri, ws=ws)
        act = next(a for a in acts if a['title'] == "Add an import of 'util'")
        assert apply_edits(text, act['edit']['changes'][uri]).startswith("import geo\nimport util\n")

    def test_pure_function_with_effects(self):
        text = "#[pure]\nf(x: int) -> int\n    print(x)\n    return x\nmain()\n    print(f(1))\n"
        _ws, uri, acts = actions_for(text)
        titles = [a['title'] for a in acts]
        assert titles == ["Remove #[pure]", "Replace #[pure] with #[effects: io]"]
        assert apply_edits(text, acts[0]['edit']['changes'][uri]).startswith("f(x: int)")
        assert apply_edits(text, acts[1]['edit']['changes'][uri]).startswith("#[effects: io]\nf(x: int)")

    def test_add_effect_to_the_declared_list(self):
        text = "#[effects: fs]\nf()\n    print(1)\nmain()\n    f()\n"
        _ws, uri, acts = actions_for(text)
        assert acts[0]['title'] == "Add 'io' to #[effects: ...]"
        assert apply_edits(text, acts[0]['edit']['changes'][uri]).startswith("#[effects: fs, io]\n")

    def test_add_capability(self):
        text = ("cap Net { send }\n#[cap: Net]\nfetch() -> int\n    return 1\nhelper() -> int\n    return fetch()\n"
                "main()\n    print(helper())\n")
        ws, uri, acts = actions_for(text)
        assert acts[0]['title'] == "Add #[cap: Net] to 'helper'"
        fixed = apply_edits(text, acts[0]['edit']['changes'][uri])
        assert "#[cap: Net]\nhelper() -> int" in fixed
        ws.change(uri, fixed, 9)
        assert ws.diagnostics(uri) == []

    def test_translate_actions(self):
        text = "main()\n    if true\n        print(1)\n"
        ws, uri = open_doc(text)
        acts = [a for a in ws.code_actions(uri, None, []) if a['kind'] == 'refactor.rewrite']
        titles = [a['title'] for a in acts]
        assert titles == ["Translate file to Қазақша", "Translate file to Qazaqsha (latyn)",
                          "Translate file to Русский"]
        ru = apply_edits(text, acts[2]['edit']['changes'][uri])
        assert ru == til.translate_source(text, 'ru') and "если" in ru

    def test_translate_back_to_english_from_russian(self):
        text = "#[lang: ru]\nглавная()\n    если истина\n        печать(1)\n"
        ws, uri = open_doc(text)
        titles = [a['title'] for a in ws.code_actions(uri, None, [])]
        assert "Translate file to English" in titles and "Translate file to Русский" not in titles

    def test_localized_titles(self):
        text = "main()\n    let x = 1\n    x = 2\n"
        _ws, _uri, acts = actions_for(text, locale='ru')
        assert acts[0]['title'] == "Заменить 'let' на 'var'"

    def test_diagnostics_without_data_are_matched(self):
        text = "main()\n    let x = 1\n    x = 2\n"
        ws, uri = open_doc(text)
        d = dict(ws.diagnostics(uri)[0])
        d.pop('data')
        acts = [a for a in ws.code_actions(uri, d['range'], [d]) if a['kind'] == 'quickfix']
        assert acts and acts[0]['title'] == "Change 'let' to 'var'"


# ═══════════════════════════════════════════════════════════════
# CODE LENSES
# ═══════════════════════════════════════════════════════════════

class TestCodeLenses:
    def test_main_and_contracts(self):
        text = ("#[requires: n >= 0]\nfact(n: int) -> int\n    return 1\nstruct P\n    x: int\nimpl P\n"
                "    #[ensures: result >= 0]\n    fn get(self) -> int\n        return 1\nmain()\n    print(fact(1))\n")
        ws, uri = open_doc(text)
        lenses = ws.code_lenses(uri)
        got = [(l['range']['start']['line'], l['command']['title'], l['command']['command'],
                l['command']['arguments'][0]['function']) for l in lenses]
        assert got == [(9, '▶ Run', 'til.run', 'main'), (9, 'Debug', 'til.debug', 'main'),
                       (1, 'Prove', 'til.prove', 'fact'), (7, 'Prove', 'til.prove', 'P.get')]
        assert lenses[0]['command']['arguments'][0]['uri'] == uri

    def test_script_without_main(self):
        ws, uri = open_doc("# a script\nlet x = 1\nprint(x)\n")
        lenses = ws.code_lenses(uri)
        assert [(l['range']['start']['line'], l['command']['command']) for l in lenses] == \
            [(2, 'til.run'), (2, 'til.debug')]
        assert lenses[0]['command']['arguments'][0]['function'] is None

    def test_localized(self):
        ws, uri = open_doc("main()\n    print(1)\n", locale='kk')
        assert ws.code_lenses(uri)[0]['command']['title'] == "▶ Іске қосу"


# ═══════════════════════════════════════════════════════════════
# SEMANTIC TOKENS
# ═══════════════════════════════════════════════════════════════

class TestSemanticTokens:
    def tokens(self, text):
        ws, uri = open_doc(text)
        return decode_tokens(text, ws.semantic_tokens(uri)['data'])

    def test_legend(self):
        assert tills.TOKEN_TYPES[:3] == ['namespace', 'type', 'struct']
        assert 'keyword' in tills.TOKEN_TYPES and 'declaration' in tills.TOKEN_MODIFIERS

    def test_kinds(self):
        text = ("# c\nstruct P\n    x: int\nenum C\n    Red\n#[pure]\nf(a: int) -> int\n    return a\n"
                "main()\n    let p = P { x: 1 }\n    print(f(p.x), C.Red, \"s\", 42)\n")
        toks = self.tokens(text)
        kinds = {(t, k) for t, k, _m in toks}
        assert ('# c', 'comment') in kinds and ('struct', 'keyword') in kinds
        assert ('P', 'struct') in kinds and ('x', 'property') in kinds and ('Red', 'enumMember') in kinds
        assert ('#[pure]', 'decorator') in kinds and ('a', 'parameter') in kinds
        assert ('f', 'function') in kinds and ('"s"', 'string') in kinds and ('42', 'number') in kinds
        assert ('C', 'enum') in kinds and ('int', 'type') in kinds
        p = [m for t, k, m in toks if t == 'print']
        assert p and 'defaultLibrary' in p[0]
        decl = [m for t, k, m in toks if t == 'f' and k == 'function']
        assert 'declaration' in decl[0]
        let_p = [m for t, k, m in toks if t == 'p' and k == 'variable']
        assert 'readonly' in let_p[0]

    def test_kazakh_keywords_and_alias(self):
        toks = self.tokens("негізгі()\n    егер ақиқат\n        басып_шығару(1)\n")
        assert ('егер', 'keyword', set()) in toks and ('ақиқат', 'keyword', set()) in toks
        assert any(t == 'басып_шығару' and k == 'function' for t, k, _m in toks)

    def test_contextual_effect_keywords(self):
        toks = self.tokens("main()\n    let r = handle\n        perform Ask(1)\n    with\n        Ask(x) => resume(x)\n")
        words = {t for t, k, _m in toks if k == 'keyword'}
        assert {'handle', 'perform', 'with', 'resume'} <= words

    def test_utf16_lengths(self):
        text = 'main()\n    let s = "😀"\n    print(s)\n'
        ws, uri = open_doc(text)
        data = ws.semantic_tokens(uri)['data']
        toks = decode_tokens(text, data)
        assert ('"😀"', 'string', set()) in toks
        k = [t for t, _k, _m in toks].index('"😀"')
        assert data[5 * k + 2] == 4                     # quote + 2 units + quote

    def test_fstring_parts(self):
        toks = self.tokens('main()\n    let n = 1\n    print(f"n={n}!")\n')
        assert ('f"n=', 'string', set()) in toks
        assert any(t == 'n' and k == 'variable' for t, k, _m in toks[-3:])

    def test_sorted_and_not_overlapping(self):
        ws, uri = open_doc(TestCompletion.SRC + "main()\n    print(twice(1))\n")
        data = ws.semantic_tokens(uri)['data']
        assert len(data) % 5 == 0
        for k in range(0, len(data), 5):
            assert data[k] >= 0 and (data[k] > 0 or k == 0 or data[k + 1] >= 0)


# ═══════════════════════════════════════════════════════════════
# BROKEN CODE
# ═══════════════════════════════════════════════════════════════

class TestBrokenCode:
    GOOD = "struct P\n    x: int\n\nf(a: int) -> int\n    let b = a * 2\n    return b\n\nmain()\n" \
           "    let p = P { x: 1 }\n    print(f(p.x))\n"

    def test_outline_hover_and_definition_survive_a_syntax_error(self):
        ws, uri = open_doc(self.GOOD)
        broken = self.GOOD.replace("    let b = a * 2", "    let b = a *")
        ws.change(uri, broken, 2)
        assert len(ws.diagnostics(uri)) == 1
        assert [s['name'] for s in ws.document_symbols(uri)] == ['P', 'f', 'main']
        loc = ws.definition(uri, *at(broken, "f(p.x"))
        assert loc[0]['range']['start'] == {'line': 3, 'character': 0}
        assert "f(a: int) -> int" in ws.hover(uri, *at(broken, "f(p.x"))['contents']['value']

    def test_outline_from_the_last_good_version(self):
        ws, uri = open_doc(self.GOOD)
        ws.document_symbols(uri)
        ws.change(uri, "main()\n  let x = [1, 2\n  print(x\n   )))\n  ", 2)
        assert [s['name'] for s in ws.document_symbols(uri)] == ['P', 'f', 'main']

    def test_unlexable_line(self):
        text = "f(a: int) -> int\n    return a\nmain()\n    let s = \"open\n    print(f(1))\n"
        ws, uri = open_doc(text)
        assert ws.diagnostics(uri)[0]['message'] == "Unterminated string"
        assert ws.definition(uri, *at(text, "f(1"))[0]['range']['start'] == {'line': 0, 'character': 0}

    def test_hover_on_a_name_declared_on_the_broken_line(self):
        text = "main()\n    let b = 1 +\n    print(b)\n"
        ws, uri = open_doc(text)
        h = ws.hover(uri, *at(text, "b)"))
        assert h is not None and "b" in h['contents']['value'] and "local variable" in h['contents']['value']

    def test_completion_in_a_half_written_call(self):
        ls = labels(complete("count(n: int) -> int\n    return n\nmain()\n    let value = 3\n    print(count(va|\n"))
        assert 'value' in ls

    def test_garbage_does_not_crash(self):
        ws, uri = open_doc("\x00\x01 )))( ]] ## \n\t\t\n  ¤¤¤ \"\n" * 5)
        assert ws.diagnostics(uri)
        for fn in (ws.document_symbols, ws.folding_ranges, ws.code_lenses, ws.semantic_tokens, ws.formatting):
            fn(uri)
        ws.completion(uri, 1, 1)
        ws.hover(uri, 0, 3)
        ws.inlay_hints(uri, None)
        ws.code_actions(uri, None, [])

    def test_empty_document(self):
        ws, uri = open_doc("")
        assert ws.diagnostics(uri) == [] and ws.document_symbols(uri) == []
        assert ws.completion(uri, 0, 0)
        assert ws.formatting(uri) == []


# ═══════════════════════════════════════════════════════════════
# PERFORMANCE
# ═══════════════════════════════════════════════════════════════

def big_program(lines=2000):
    out, k = [], 0
    while len(out) < lines:
        out += [f"# helper {k}", "#[requires: n >= 0]", f"func_{k}(n: int, s: str) -> int", "    var total = 0",
                f"    let label = s + \"_{k}\"", "    for i in 0..n", "        if i % 2 == 0",
                f"            total += i * {k}", "        else", "            total -= 1", "    let arr = [1, 2, 3]",
                "    arr.push(total)", "    return total + len(label)", "", f"struct P{k}", "    x: int",
                "    y: float", "", f"impl P{k}", "    fn norm(self) -> float", "        return self.y * 2.0 + self.x", ""]
        k += 1
    return '\n'.join(out + ["main()"] + [f"    print(func_{j}({j}, \"a\"))" for j in range(0, k, 7)]) + '\n'


class TestPerformance:
    def test_diagnostics_of_2000_lines(self):
        text = big_program()
        assert text.count('\n') >= 2000
        ws, uri = open_doc(text)
        t0 = time.perf_counter()
        assert ws.diagnostics(uri) == []
        cold = time.perf_counter() - t0
        ws.change(uri, text.replace("+ len(label)", "+ len(label) + 1", 1), 2)
        t0 = time.perf_counter()
        ws.diagnostics(uri)
        warm = time.perf_counter() - t0
        print(f"\n2000-line diagnostics: {cold * 1000:.0f} ms, after a change {warm * 1000:.0f} ms")
        assert warm < 3.0                       # spec: < 1 s (measured ~0.3 s; generous for slow CI)

    def test_completion_in_2000_lines(self):
        text = big_program()
        ws, uri = open_doc(text)
        ws.diagnostics(uri)
        lines = text.split('\n')
        k = next(i for i, l in enumerate(lines) if 'arr.push(total)' in l)
        times = []
        for n in range(3):
            lines[k] = "    arr." + ("" if n == 0 else "p" * n)
            ws.change(uri, '\n'.join(lines), 3 + n)
            t0 = time.perf_counter()
            items = ws.completion(uri, k, len(lines[k]))
            times.append(time.perf_counter() - t0)
            assert 'push' in labels(items)
        print(f"\n2000-line completion while typing: {', '.join(f'{t * 1000:.1f}' for t in times)} ms")
        assert max(times) < 0.5                 # spec: < 150 ms (measured a few ms)
