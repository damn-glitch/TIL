#!/usr/bin/env python3
"""
TIL 3.1 localized messages (docs/TIL_I18N.md): compiler diagnostics and the
error reports of both VMs in Russian and Kazakh.

* the catalog (src/tilmsg.py) is consistent and translates real diagnostics;
* language selection: --lang > TIL_LANG > #[lang: ...] pragma > English;
* `til check / build / run / exec / repl` print localized diagnostics, the
  reference VM and the C VM produce identical localized reports, and values
  that programs see (`error` in retry/timeout/intent) stay English;
* unknown messages fall back to English, a missing tilmsg.py means English;
* tools/i18n_coverage.py: every diagnostic of src/til.py and vm/tilvm.c has a
  translation, and a new message without one is reported.

Run: python3 -m pytest tests/test_i18n.py -v
"""

import io
import os
import re
import shutil
import subprocess
import sys

import pytest

from tilhelpers import CC, ROOT, SRC, TIL, cli, compile_vm, needs_cc, run_cvm, til

TOOLS = os.path.join(ROOT, 'tools')
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import tilmsg              # noqa: E402  (src/ is on sys.path via tilhelpers)
import i18n_coverage       # noqa: E402

CYRILLIC = re.compile(r'[А-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]')

# ── programs ────────────────────────────────────────────────────────────────
IMMUTABLE = "main()\n    let x = 1\n    x = 2\n    print(x)\n"
SYNTAX = "main()\n    let x = $\n"
RUNTIME = 'f(a: int) -> int\n    return 10 / a\nmain()\n    print("start")\n    print(f(0))\n'
VERIFY = ("#[pure]\nf(x: int) -> int\n    print(x)\n    return x\n"
          "#[cap: Net]\nsend(x: int)\n    print(x)\ng()\n    send(1)\n"
          "#[requires: n > 0]\nh(n: int) -> int\n    return n\n"
          "main()\n    print(f(1))\n    g()\n    print(h(0))\n    print(h(1, 2))\n")
VISIBLE = ('main()\n'
           '    let a = retry(2)\n        let xs = [1, 2]\n        xs[7]\n    else\n        "caught: " + error\n'
           '    print(a)\n'
           '    let b = retry(1)\n        intent "load the class list"\n            let d = {"x": 1}\n'
           '            d["nope"]\n    else\n        error\n    print(b)\n'
           '    let c = timeout(10)\n        var i = 0\n        loop\n            i += 1\n        i\n'
           '    else\n        "stopped: " + error\n    print(c)\n'
           '    print(10 / 0)\n')
VISIBLE_OUT = ("caught: index out of bounds: index 7, length 2\n"
               "intent 'load the class list' failed: key not found: nope\n"
               "stopped: timeout\n")

IMMUTABLE_HEADER = {
    'en': "error[E-VM]: cannot assign twice to immutable variable 'x'",
    'ru': "ошибка[E-VM]: нельзя повторно присвоить значение неизменяемой переменной 'x'",
    'kk': "қате[E-VM]: өзгермейтін 'x' айнымалысына екінші рет мән меншіктеуге болмайды",
}
IMMUTABLE_HINT = {
    'en': "  = hint: declare it with 'var' to make it mutable",
    'ru': "  = подсказка: объявите её через 'var', чтобы она стала изменяемой",
    'kk': "  = кеңес: өзгермелі болуы үшін оны 'var' арқылы жариялаңыз",
}
RUNTIME_REPORT = {
    'en': "runtime error: division by zero\n  at f (line 2)\n  at main (line 5)\n  at __start__ (line 3)\n",
    'ru': "ошибка выполнения: деление на ноль\n  в f (строка 2)\n  в main (строка 5)\n  в __start__ (строка 3)\n",
    'kk': "орындау қатесі: нөлге бөлу\n  f ішінде (2-жол)\n  main ішінде (5-жол)\n  __start__ ішінде (3-жол)\n",
}


def write(tmp_path, name: str, text: str) -> str:
    path = os.path.join(str(tmp_path), name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def run_til(*args, til_lang: str = '', **kw):
    """The CLI with a controlled TIL_LANG ('' = unset: the developer's own setting must not leak in)."""
    env = dict(kw.pop('env', None) or {})
    env['TIL_LANG'] = til_lang
    return cli(*args, env=env, **kw)


# ═══════════════════════════════════════════════════════════════
# THE CATALOG
# ═══════════════════════════════════════════════════════════════

class TestCatalog:
    def test_catalog_is_consistent(self):
        assert tilmsg.check_catalog() == []

    def test_languages(self):
        assert tilmsg.LANGUAGES == til.MESSAGE_LANGUAGES == ('en', 'ru', 'kk')
        assert set(tilmsg.FRAME) == {'ru', 'kk'}

    def test_every_entry_is_really_translated(self):
        for en, ru, kk in tilmsg.MESSAGES:
            assert CYRILLIC.search(ru), f"Russian text of {en!r} has no Cyrillic: {ru!r}"
            assert CYRILLIC.search(kk), f"Kazakh text of {en!r} has no Cyrillic: {kk!r}"
            assert ru != kk, f"Kazakh text of {en!r} is the Russian text"

    def test_no_suffix_glued_to_a_placeholder(self):
        # Kazakh (and Russian) endings depend on the word they attach to, so a
        # translation never glues one to a placeholder: '{name}' функциясы.
        for en, ru, kk in tilmsg.MESSAGES:
            for text in (ru, kk):
                assert not re.search(r'\}[А-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]', text), text

    def test_bad_templates_are_reported(self):
        saved = list(tilmsg.MESSAGES)
        try:
            tilmsg.MESSAGES.append(("broken {name}", "сломано", "бұзылған {name}"))
            tilmsg.MESSAGES.append(("lone { brace", "одинокая { скобка", "жалғыз { жақша"))
            problems = tilmsg.check_catalog()
            assert any("ru placeholders" in p for p in problems), problems
            assert any("unescaped '{'" in p for p in problems), problems
        finally:
            tilmsg.MESSAGES[:] = saved


# ═══════════════════════════════════════════════════════════════
# TRANSLATING DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

KNOWN = [
    ("error[E-EFFECT]: pure function 'f' performs effect 'io'",
     "ошибка[E-EFFECT]: чистая функция 'f' выполняет эффект 'io'",
     "қате[E-EFFECT]: 'f' таза функциясы 'io' әсерін тудырады"),
    ("  = hint: did you mean 'count'?",
     "  = подсказка: возможно, имелось в виду 'count'?",
     "  = кеңес: 'count' деп жазғыңыз келді ме?"),
    ("runtime error: index out of bounds: index 5, length 3",
     "ошибка выполнения: индекс вне границ: индекс 5, длина 3",
     "орындау қатесі: индекс шектен тыс: индекс 5, ұзындық 3"),
    ("fatal error: step limit exceeded (1000)",
     "фатальная ошибка: превышен лимит шагов (1000)",
     "қатерлі қате: қадам шегінен асып кетті (1000)"),
    ("  at Point.dist (line 7)",
     "  в Point.dist (строка 7)",
     "  Point.dist ішінде (7-жол)"),
    ("Syntax Error: error: Expected INDENT, got NEWLINE",
     "Синтаксическая ошибка: ожидалось INDENT, получено NEWLINE",
     "Синтаксистік қате: INDENT күтілді, бірақ NEWLINE кездесті"),
    ("Warning: Line 3: Type mismatch: expected int, got str",
     "Предупреждение: строка 3: несоответствие типов: ожидался int, получен str",
     "Ескерту: 3-жол: типтер сәйкес емес: int күтілді, бірақ str алынды"),
    # a nested message inside a message
    ("runtime error: intent 'load' failed: key not found: \"k\"",
     "ошибка выполнения: намерение 'load' не выполнено: ключ не найден: \"k\"",
     "орындау қатесі: 'load' ниеті орындалмады: кілт табылмады: \"k\""),
    # the most specific template wins over a generic one
    ("error[E-ARITY]: 'f' expects 1 to 2 argument(s), got 3",
     "ошибка[E-ARITY]: 'f' ожидает аргументов: от 1 до 2, передано: 3",
     "қате[E-ARITY]: 'f' 1–2 аргумент күтеді, бірақ 3 берілді"),
    ("Syntax Error: error: Invalid hexadecimal literal: '0xZZ'",
     "Синтаксическая ошибка: неверный шестнадцатеричный литерал: '0xZZ'",
     "Синтаксистік қате: жарамсыз он алтылық литерал: '0xZZ'"),
    ("runtime error: value is not iterable",
     "ошибка выполнения: значение не является итерируемым",
     "орындау қатесі: бұл мән бойынша итерация жасауға болмайды"),
    ("runtime error: field 'cb' is not callable",
     "ошибка выполнения: поле 'cb' нельзя вызвать",
     "орындау қатесі: 'cb' өрісін шақыруға болмайды"),
    # identifiers, quotes and code inside placeholders are copied unchanged
    ("error[E-VM]: unknown name 'көрсеткіш'",
     "ошибка[E-VM]: неизвестное имя 'көрсеткіш'",
     "қате[E-VM]: белгісіз атау: 'көрсеткіш'"),
    ("error[E-CONTRACT]: contract violation: call to 'sqrt_i' (n=-4) breaks #[requires: n >= 0]",
     "ошибка[E-CONTRACT]: нарушение контракта: вызов 'sqrt_i' (n=-4) не удовлетворяет #[requires: n >= 0]",
     "қате[E-CONTRACT]: келісімшарт бұзылды: 'sqrt_i' шақыруы (n=-4) #[requires: n >= 0] шартын бұзады"),
    ("Error: File not found: prog.til",
     "Ошибка: файл не найден: prog.til",
     "Қате: файл табылмады: prog.til"),
    ("fatal error: native recursion limit",
     "фатальная ошибка: превышен внутренний предел рекурсии",
     "қатерлі қате: ішкі рекурсия шегінен асып кетті"),
    # effects with resumption (the til31/effects branch)
    ("runtime error: unhandled effect 'Log'",
     "ошибка выполнения: необработанный эффект 'Log'",
     "орындау қатесі: өңделмеген әсер: 'Log'"),
    ("error[E-VM]: effect 'Ask' expects 1 argument(s), got 2",
     "ошибка[E-VM]: эффект 'Ask' ожидает аргументов: 1, передано: 2",
     "қате[E-VM]: 'Ask' әсері 1 аргумент күтеді, бірақ 2 берілді"),
    ("Syntax Error: error: duplicate clause for effect 'Ask' in one handle",
     "Синтаксическая ошибка: повторная ветвь для эффекта 'Ask' в одном handle",
     "Синтаксистік қате: бір handle ішінде 'Ask' әсерінің тармағы қайталанған"),
    # graphics (the til31/gfx branch)
    ("runtime error: draw_pixel() argument 3 must be a color 0..0xFFFFFF, got -1",
     "ошибка выполнения: draw_pixel(): аргумент 3 должен быть цветом 0..0xFFFFFF, получено -1",
     "орындау қатесі: draw_pixel(): 3-аргумент 0..0xFFFFFF аралығындағы түс болуы керек, бірақ -1 берілді"),
]


class TestLocalize:
    @pytest.mark.parametrize("en,ru,kk", KNOWN, ids=[k[0][:40] for k in KNOWN])
    def test_known_messages(self, en, ru, kk):
        assert tilmsg.localize(en, 'ru') == ru
        assert tilmsg.localize(en, 'kk') == kk
        assert til.localize(en, 'ru') == ru          # the compiler's entry point
        assert tilmsg.localize(en, 'en') == en

    def test_unknown_message_stays_english_inside_a_translated_frame(self):
        text = "error[E-TYPE]: some message nobody has translated yet\n  = hint: neither this one"
        assert tilmsg.localize(text, 'ru') == ("ошибка[E-TYPE]: some message nobody has translated yet\n"
                                               "  = подсказка: neither this one")
        assert tilmsg.translate("some message nobody has translated yet", 'kk') is None
        assert tilmsg.localize("a line that is no diagnostic at all", 'kk') == "a line that is no diagnostic at all"

    def test_source_excerpts_are_never_translated(self):
        text = ("runtime error: division by zero\n"
                " --> prog.til:3:11\n"
                '  3 |     print("division by zero")\n'
                "    |           ^\n"
                "  4 | xs cannot be indexed")
        out = tilmsg.localize(text, 'ru').split('\n')
        assert out[0] == "ошибка выполнения: деление на ноль"
        assert out[1:] == text.split('\n')[1:]

    def test_a_generic_template_never_swallows_free_text(self):
        # "{type} cannot be indexed" must not translate a sentence that ends the same way
        assert tilmsg.localize("runtime error: my list cannot be indexed", 'ru') == \
            "ошибка выполнения: my list cannot be indexed"
        assert tilmsg.localize("runtime error: Point cannot be indexed", 'ru') == \
            "ошибка выполнения: Point не поддерживает индексацию"

    def test_colors_are_kept(self):
        reporter = til.ErrorReporter("main()\n    let x = 1\n    x = 2\n", "p.til")
        reporter.use_color = True                    # as on a terminal
        text = reporter.format_error(3, 7, "cannot assign twice to immutable variable 'x'", "E-VM",
                                     "declare it with 'var' to make it mutable")
        out = tilmsg.localize(text, 'kk')
        assert out.count('\x1b[') == text.count('\x1b[')
        plain = re.sub(r'\x1b\[[0-9;]*m', '', out).split('\n')
        assert plain[0] == IMMUTABLE_HEADER['kk']
        assert plain[-1] == IMMUTABLE_HINT['kk']
        assert plain[3] == re.sub(r'\x1b\[[0-9;]*m', '', text).split('\n')[3]      # source line kept

    def test_windows_line_endings(self):
        assert tilmsg.localize("runtime error: division by zero\r\n  at main (line 2)\r\n", 'ru') == \
            "ошибка выполнения: деление на ноль\r\n  в main (строка 2)\r\n"

    def test_english_and_unknown_languages_are_identity(self):
        text = "error[E-VM]: unknown name 'x'"
        assert tilmsg.localize(text, 'en') == text
        assert tilmsg.localize(text, 'de') == text
        assert til.localize(text, None) == text      # main() chose nothing: English
        assert til.localize_bytes(b"runtime error: \xff\n", 'ru') == "ошибка выполнения: \udcff\n".encode(
            'utf-8', 'surrogateescape')

    def test_a_broken_catalog_never_hides_a_diagnostic(self, monkeypatch):
        class Broken:
            @staticmethod
            def localize(text, lang):
                raise RuntimeError("bug in a translation")
        monkeypatch.setattr(til, '_MESSAGE_CATALOG', [Broken])
        assert til.localize("error: x", 'ru') == "error: x"

    @pytest.mark.parametrize("src,en,ru,kk", [
        (SYNTAX, "Syntax Error: error: Unexpected character: '$'",
         "Синтаксическая ошибка: неожиданный символ: '$'", "Синтаксистік қате: күтілмеген таңба: '$'"),
        ("main()\n    let s = \"abc\n", "Syntax Error: error: Unterminated string",
         "Синтаксическая ошибка: незакрытый строковый литерал", "Синтаксистік қате: жол литералы жабылмаған"),
        ("main()\n    print(nmae)\n", "error[E-VM]: unknown name 'nmae'",
         "ошибка[E-VM]: неизвестное имя 'nmae'", "қате[E-VM]: белгісіз атау: 'nmae'"),
        ("main()\n    frobnicate(1)\n", "error[E-VM]: unknown function 'frobnicate'",
         "ошибка[E-VM]: неизвестная функция 'frobnicate'", "қате[E-VM]: белгісіз функция: 'frobnicate'"),
        ("main()\n    print(abs(1, 2))\n", "error[E-VM]: 'abs' takes 1 argument(s), got 2",
         "ошибка[E-VM]: 'abs' принимает аргументов: 1, передано: 2",
         "қате[E-VM]: 'abs' 1 аргумент қабылдайды, бірақ 2 берілді"),
        ("main()\n    break\n", "error[E-VM]: 'break' outside of a loop",
         "ошибка[E-VM]: 'break' вне цикла", "қате[E-VM]: 'break' циклден тыс"),
    ])
    def test_real_compile_errors(self, src, en, ru, kk):
        with pytest.raises((til.CompileError, SyntaxError)) as info:
            compile_vm(src, "p.til")
        text = ("Syntax Error: " if isinstance(info.value, SyntaxError) else "") + str(info.value)
        assert text.split('\n')[0] == en
        assert tilmsg.localize(text, 'ru').split('\n')[0] == ru
        assert tilmsg.localize(text, 'kk').split('\n')[0] == kk


# ═══════════════════════════════════════════════════════════════
# CHOOSING THE LANGUAGE
# ═══════════════════════════════════════════════════════════════

class TestLanguageSelection:
    @pytest.mark.parametrize("value,code", [
        ('ru', 'ru'), ('RU', 'ru'), ('ru_RU.UTF-8', 'ru'), ('kk', 'kk'), ('kk-latn', 'kk'), ('kz', 'kk'),
        ('kk_KZ', 'kk'), ('en', 'en'), ('en_US.UTF-8', 'en'), ('fr', None), ('', None), (None, None),
        ('russian', None)])
    def test_language_codes(self, value, code):
        assert til.message_language_code(value) == code

    @pytest.mark.parametrize("source,code", [
        ("#[lang: ru]\nmain()\n", 'ru'), ("#[lang: kk-latn]\n", 'kk'), ("#[lang: kk]\n", 'kk'),
        ("#[lang: en]\n", 'en'), ("  #[lang: ru, kk-latn]\n", 'ru'), ("main()\n", None),
        ("#[lang: fr]\n", None)])
    def test_pragma(self, source, code):
        assert til.pragma_message_language(source) == code

    def test_precedence(self):
        ru_prog = "#[lang: ru]\nmain()\n"
        assert til.message_language('kk', ru_prog, {'TIL_LANG': 'en'}) == 'kk'      # --lang first
        assert til.message_language(None, ru_prog, {'TIL_LANG': 'kk'}) == 'kk'      # then TIL_LANG
        assert til.message_language(None, ru_prog, {'TIL_LANG': ''}) == 'ru'        # then the pragma
        assert til.message_language(None, ru_prog, {'TIL_LANG': 'xx'}) == 'ru'      # (unknown env ignored)
        assert til.message_language(None, "main()\n", {}) == 'en'                   # else English
        assert til.message_language('en', ru_prog, {'TIL_LANG': 'ru'}) == 'en'


# ═══════════════════════════════════════════════════════════════
# THE COMMAND LINE
# ═══════════════════════════════════════════════════════════════

class Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.fixture
def til_main(monkeypatch, capfd):
    """`til ARGS...` run by til.main() in this process (no interpreter start-up per
    call). Output is captured at the file-descriptor level, so the C VM process
    started by --cvm is captured too."""
    def run(*args, til_lang: str = ''):
        monkeypatch.setattr(sys, 'argv', ['til', *args])
        monkeypatch.setenv('TIL_LANG', til_lang)
        monkeypatch.setattr(til, 'MESSAGE_LANG', None)
        monkeypatch.setattr(til, '_MESSAGE_LANG_FROM_USER', False)
        capfd.readouterr()
        rc = til.main()
        sys.stdout.flush()
        sys.stderr.flush()
        out, err = capfd.readouterr()
        return Result(rc, out, err)
    return run


class TestCommandLine:
    def test_real_command_line(self, tmp_path):
        # the same paths as below, through a real `python3 src/til.py` process
        r = run_til('check', '--lang', 'kk', write(tmp_path, 'p.til', IMMUTABLE))
        assert r.returncode == 1 and r.stderr.split('\n')[0] == IMMUTABLE_HEADER['kk']
        r = run_til('run', '--vm', write(tmp_path, 'r.til', RUNTIME), til_lang='ru')
        assert (r.returncode, r.stdout, r.stderr) == (1, "start\n", RUNTIME_REPORT['ru'])

    @pytest.mark.parametrize("lang", ['en', 'ru', 'kk'])
    def test_check_with_lang_option(self, tmp_path, til_main, lang):
        r = til_main('check', '--lang', lang, write(tmp_path, 'p.til', IMMUTABLE))
        assert r.returncode == 1
        lines = r.stderr.split('\n')
        assert lines[0] == IMMUTABLE_HEADER[lang]
        assert IMMUTABLE_HINT[lang] in lines
        assert "  3 |     x = 2" in lines                    # the source excerpt is unchanged

    def test_til_lang_environment(self, tmp_path, til_main):
        r = til_main('run', '--vm', write(tmp_path, 'p.til', IMMUTABLE), til_lang='kk')
        assert r.returncode == 1 and r.stderr.split('\n')[0] == IMMUTABLE_HEADER['kk']
        r = til_main('build', '--target', 'vm', write(tmp_path, 's.til', SYNTAX), til_lang='ru_RU.UTF-8')
        assert r.returncode == 1
        assert r.stderr.startswith("Синтаксическая ошибка: неожиданный символ: '$'\n --> ")

    @pytest.mark.parametrize("pragma,first_line", [
        ("#[lang: ru]\nглавная()\n    печать(неизвестно)\n", "ошибка[E-VM]: неизвестное имя 'неизвестно'"),
        ("#[lang: kk-latn]\nnegizgi()\n    basyp_shygaru(belgisiz)\n", "қате[E-VM]: белгісіз атау: 'belgisiz'"),
        ("#[lang: kk]\nнегізгі()\n    басып_шығару(белгісіз)\n", "қате[E-VM]: белгісіз атау: 'белгісіз'"),
    ], ids=['ru', 'kk-latn', 'kk'])
    def test_program_pragma(self, tmp_path, til_main, pragma, first_line):
        r = til_main('check', write(tmp_path, 'p.til', pragma))
        assert r.returncode == 1 and r.stderr.split('\n')[0] == first_line

    def test_precedence(self, tmp_path, til_main):
        prog = write(tmp_path, 'p.til', "#[lang: ru]\n" + IMMUTABLE)
        first = {}
        for name, args, env in (('pragma', (), ''), ('env beats pragma', (), 'kk'),
                                ('option beats env', ('--lang', 'en'), 'kk'),
                                ('option beats pragma', ('--lang=kk',), '')):
            first[name] = til_main('check', *args, prog, til_lang=env).stderr.split('\n')[0]
        assert first == {'pragma': IMMUTABLE_HEADER['ru'], 'env beats pragma': IMMUTABLE_HEADER['kk'],
                         'option beats env': IMMUTABLE_HEADER['en'],
                         'option beats pragma': IMMUTABLE_HEADER['kk']}

    def test_unknown_language(self, tmp_path, til_main):
        prog = write(tmp_path, 'p.til', 'print(1)\n')
        r = til_main('run', '--vm', '--lang', 'fr', prog)
        assert r.returncode == 1 and r.stdout == ""
        assert r.stderr.strip() == "Error: unknown message language 'fr' (use: en, ru, kk)"
        r = til_main('run', '--vm', prog, til_lang='klingon')          # a bad TIL_LANG only warns
        assert r.returncode == 0 and r.stdout == "1\n"
        assert "TIL_LANG='klingon' is not a message language" in r.stderr

    def test_semantic_errors_in_check(self, tmp_path, til_main):
        r = til_main('check', '--lang', 'ru', write(tmp_path, 'v.til', VERIFY))
        assert r.returncode == 1
        headers = [l for l in r.stderr.split('\n') if l.startswith('ошибка')]
        assert sorted(headers) == sorted([
            "ошибка[E-CONTRACT]: нарушение контракта: вызов 'h' (n=0) не удовлетворяет #[requires: n > 0]",
            "ошибка[E-ARITY]: 'h' ожидает аргументов: 1, передано: 2",
            "ошибка[E-EFFECT]: чистая функция 'f' выполняет эффект 'io'",
            "ошибка[E-CAP]: 'g' использует функцию 'send', которой требуется полномочие 'Net', "
            "но у 'g' его нет",
        ])
        assert not [l for l in r.stderr.split('\n') if l.startswith('error') or '= hint:' in l]
        assert r.stderr.count("= подсказка: ") == 3

    def test_runtime_report_reference_vm(self, tmp_path, til_main):
        prog = write(tmp_path, 'r.til', RUNTIME)
        for lang in ('en', 'ru', 'kk'):
            r = til_main('run', '--vm', '--lang', lang, prog)
            assert (r.returncode, r.stdout, r.stderr) == (1, "start\n", RUNTIME_REPORT[lang])

    @needs_cc
    def test_runtime_report_c_vm(self, tmp_path, til_main):
        prog = write(tmp_path, 'r.til', RUNTIME)
        for lang in ('en', 'ru', 'kk'):
            r = til_main('run', '--cvm', prog, til_lang=lang)
            assert (r.returncode, r.stdout, r.stderr) == (1, "start\n", RUNTIME_REPORT[lang])

    def test_exec_bytecode(self, tmp_path, til_main):
        tbc = os.path.join(str(tmp_path), 'r.tbc')
        r = til_main('build', '--target', 'vm', '-o', tbc, write(tmp_path, 'r.til', RUNTIME))
        assert r.returncode == 0, r.stderr
        r = til_main('exec', '--lang', 'ru', tbc)
        assert (r.returncode, r.stdout, r.stderr) == (1, "start\n", RUNTIME_REPORT['ru'])

    @pytest.mark.parametrize("vm", ['--vm', pytest.param('--cvm', marks=needs_cc)])
    def test_program_visible_errors_stay_english(self, tmp_path, til_main, vm):
        r = til_main('run', vm, '--lang', 'ru', write(tmp_path, 'v.til', VISIBLE))
        assert r.returncode == 1
        assert r.stdout == VISIBLE_OUT                     # `error` values: English
        assert r.stderr == "ошибка выполнения: деление на ноль\n  в main (строка 23)\n  в __start__ (строка 1)\n"

    def test_language_lasts_for_one_main_call(self, tmp_path, monkeypatch, capsys):
        # main() may run in-process (tests do): the language it picks must not leak
        monkeypatch.setattr(sys, 'argv', ['til', 'check', '--lang', 'kk', write(tmp_path, 'p.til', IMMUTABLE)])
        assert til.main() == 1
        assert capsys.readouterr().err.split('\n')[0] == IMMUTABLE_HEADER['kk']
        assert til.MESSAGE_LANG is None and til.localize("error: x") == "error: x"

    def test_extra_commands(self, monkeypatch, capsys):
        # `til pkg/fit/debug/prove ... --lang ru`: the option is the compiler's, the tool
        # reads the language from til.MESSAGE_LANG (and its own arguments stay intact)
        seen = {}

        class Tool:
            @staticmethod
            def main(argv):
                seen['argv'], seen['lang'] = argv, til.MESSAGE_LANG
                print(til.localize("error: Unexpected token: COLON"), file=sys.stderr)
                return 0
        monkeypatch.setitem(til.EXTRA_COMMANDS, 'zzz', ('til_zzz_tool', 'test'))
        monkeypatch.setitem(sys.modules, 'til_zzz_tool', Tool)
        monkeypatch.setattr(sys, 'argv', ['til', 'zzz', 'add', '--lang', 'kk', 'x', '--lang=ru', '--',
                                          '--lang', 'en'])
        monkeypatch.setenv('TIL_LANG', '')
        assert til.main() == 0
        assert seen == {'argv': ['add', 'x', '--', '--lang', 'en'], 'lang': 'ru'}
        assert capsys.readouterr().err == "ошибка: неожиданный токен: COLON\n"
        assert til.without_lang_option(['--lang']) == ['--lang']      # no value: left to the tool

    def test_repl(self):
        r = run_til('repl', input_text="print(y)\nlet z = 1 / 0\nexit\n", til_lang='ru')
        assert "ошибка[E-VM]: неизвестное имя 'y'" in r.stdout
        assert "ошибка выполнения: деление на ноль" in r.stderr

    def test_without_the_catalog_messages_stay_english(self, tmp_path):
        alone = tmp_path / 'alone'
        alone.mkdir()
        shutil.copy(TIL, str(alone / 'til.py'))                  # til.py without tilmsg.py
        prog = write(tmp_path, 'p.til', IMMUTABLE)
        r = subprocess.run([sys.executable, str(alone / 'til.py'), 'check', '--lang', 'ru', prog],
                           capture_output=True, text=True,
                           env=dict(os.environ, TIL_LANG='', PYTHONPATH=''))
        assert r.returncode == 1
        lines = r.stderr.split('\n')
        assert lines[0] == "note: message translations are not installed (tilmsg.py is missing); " \
                           "messages stay in English"
        assert lines[1] == IMMUTABLE_HEADER['en'] and IMMUTABLE_HINT['en'] in lines
        r = subprocess.run([sys.executable, str(alone / 'til.py'), 'check', prog], capture_output=True,
                           text=True, env=dict(os.environ, TIL_LANG='kk', PYTHONPATH=''))
        assert r.stderr.split('\n')[0] == IMMUTABLE_HEADER['en']   # no note without --lang


# ═══════════════════════════════════════════════════════════════
# BOTH VMs: IDENTICAL LOCALIZED REPORTS
# ═══════════════════════════════════════════════════════════════

ERROR_PROGRAMS = [
    'main()\n    let xs = [1, 2, 3]\n    print(xs[5])\n',
    'main()\n    let m = {"a": 1}\n    print(m["b"])\n',
    'main()\n    let o = None\n    print(o.unwrap())\n',
    'main()\n    assert 1 > 2, "math is broken"\n',
    '#[requires: n > 0]\nf(n: int) -> int\n    return n\nmain()\n    let k = 0 - 1\n    print(f(k))\n',
    'main()\n    intent "parse the input"\n        let v = int("12x")\n        print(v)\n',
    'main()\n    print([].pop())\n',
    'main()\n    print(chr(-1))\n',
]


def reference_vm_report(src: str):
    """(exit code, stdout, stderr) of the reference VM, in-process (English report)."""
    module = til.TBCModule.from_bytes(compile_vm(src).serialize())
    out = []
    err = io.BytesIO()

    class Stderr:
        buffer = err

        @staticmethod
        def write(text):
            err.write(text.encode('utf-8'))

        @staticmethod
        def flush():
            pass
    saved = sys.stderr
    sys.stderr = Stderr()
    try:
        rc = til.TILVM(module, write=out.append).run()
    finally:
        sys.stderr = saved
    return rc, b"".join(out).decode(), err.getvalue().decode('utf-8', 'replace')


class TestBothVMs:
    @pytest.mark.parametrize("src", ERROR_PROGRAMS)
    def test_identical_localized_reports(self, src):
        rc, out, err = reference_vm_report(src)
        assert rc == 1 and err.startswith("runtime error: ")
        localized = {lang: tilmsg.localize(err, lang) for lang in ('ru', 'kk')}
        for lang, text in localized.items():
            first = text.split('\n')[0]
            assert first != err.split('\n')[0] and CYRILLIC.search(first.split(':', 1)[1]), \
                f"{lang}: message not translated: {first!r}"
            assert " at " not in text and "line" not in text
        if CC:
            crc, cout, cerr = run_cvm(src)
            assert (crc, cout, cerr) == (rc, out, err), "the VMs report differently"
            assert {lang: tilmsg.localize(cerr, lang) for lang in ('ru', 'kk')} == localized


# ═══════════════════════════════════════════════════════════════
# COVERAGE: EVERY DIAGNOSTIC HAS A TRANSLATION
# ═══════════════════════════════════════════════════════════════

def _texts(templates):
    return {t.text() for t in templates}


_EXTRACTED = {}


def extracted(which: str):
    """The templates of src/til*.py ('py') or vm/tilvm.c ('c'), extracted once per session."""
    if which not in _EXTRACTED:
        _EXTRACTED[which] = i18n_coverage.extract_python() if which == 'py' else i18n_coverage.extract_c()
    return list(_EXTRACTED[which])


class TestCoverage:
    def test_every_diagnostic_has_a_translation(self):
        report = i18n_coverage.analyse(extracted('py') + extracted('c'))
        missing = i18n_coverage.enforced_missing(report)
        assert not missing, (
            "diagnostics without a Russian/Kazakh translation:\n" +
            "\n".join(f"  [{t.category}] {t.text()!r}  ({', '.join(t.where[:2])})" for t in missing) +
            "\nadd them to MESSAGES in src/tilmsg.py (python3 tools/i18n_coverage.py --skeleton prints "
            "entries to fill in), or to UNTRANSLATED if they are internal")
        assert len(report['covered']) > 150

    def test_extraction_finds_the_known_sites(self):
        py = _texts(extracted('py'))
        for t in ("division by zero", "unknown function '{}'", "unknown struct '{}'",
                  "'{}' expects {} to {} argument(s), got {}", "Missing closing parenthesis ')'",
                  "contract violation: {} requires {}", "intent '{}' failed: ",
                  "Unterminated '{{' in f-string", "no field '{}' on {}", "{}() is missing arguments",
                  "pure function '{}' performs effect '{}' (through the call to '{}')"):
            assert t in py, t
        c = _texts(extracted('c'))
        for t in ("division by zero", "unsupported operand types for {}: {} and {}", "key not found: {}",
                  "index out of bounds: index {}, length {}", "{}() takes {} argument(s), got {}",
                  'invalid integer: "{}"', "cannot compare {} and {}", "min() of empty sequence"):
            assert t in c, t

    def test_a_new_python_message_is_reported(self):
        src = ("class TILVM:\n    def panic(self, msg):\n        pass\n"
               "    def run(self, x):\n        self.panic(f\"the flux capacitor of {x} overheated\")\n"
               "        self.panic(\"division by zero\")\n")
        templates = i18n_coverage.PyExtractor(src, os.path.join(SRC, 'tilnew.py')).run()
        report = i18n_coverage.analyse(templates)
        assert _texts(report['missing']) == {"the flux capacitor of {} overheated"}
        assert _texts(report['covered']) == {"division by zero"}

    def test_a_new_c_message_is_reported(self, tmp_path):
        path = write(tmp_path, 'vm.c', 'static int f(tilvm *vm, int x) {\n'
                                       '    if (x) return fail(vm, "warp core breach");\n'
                                       '    return failn(vm, "no method \'", "x", "\' on ", name, 0, 0);\n}\n')
        report = i18n_coverage.analyse(i18n_coverage.extract_c(path))
        assert _texts(report['missing']) == {"warp core breach"}
        assert _texts(report['covered']) == {"no method 'x' on {}"}

    def test_c_forwarding_helpers_are_expanded_per_call_site(self, tmp_path):
        # like gfx_arg_error(vm, id, k, " must be 1..4096, got ", ...): the message tail is a parameter
        path = write(tmp_path, 'vm.c',
                     'static int arg_error(tilvm *vm, uint8_t id, const char *what, const char *got) {\n'
                     '    return failn(vm, NATIVE_NAMES[id], "() argument", what, got, 0, 0);\n}\n'
                     'static int check(tilvm *vm, int x, char *buf) {\n'
                     '    if (x < 0) return arg_error(vm, 3, " must be positive, got ", fmt_int(buf, x));\n'
                     '    return arg_error(vm, 3, " must be small, got ", fmt_int(buf, x));\n}\n')
        assert _texts(i18n_coverage.extract_c(path)) == {"{}() argument must be positive, got {}",
                                                          "{}() argument must be small, got {}"}

    def test_a_half_english_translation_is_not_coverage(self):
        # without its own entry, "expects 1 to 2" would come out as "ожидает аргументов: 1 to 2"
        generic = tilmsg.Entry("'{func}' expects {count} argument(s), got {got}", "x {func} {count} {got}",
                               "y {func} {count} {got}")
        t = i18n_coverage.Template(["'", None, "' expects ", None, " to ", None, " argument(s), got ", None],
                                   'semantic', 'x:1')
        assert i18n_coverage.coverage_of(tilmsg, t, [generic])[0] is False
        assert i18n_coverage.coverage_of(tilmsg, t)[0] is True

    def test_allow_list(self):
        report = i18n_coverage.analyse([i18n_coverage.Template(["panic"], 'runtime', 'x:1')])
        assert _texts(report['allowed']) == {"panic"} and not report['missing']

    def test_tool_command_line(self, capsys):
        r = subprocess.run([sys.executable, os.path.join(TOOLS, 'i18n_coverage.py')],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "total" in r.stdout
        listed = [l for l in r.stdout.split('\n') if l.startswith('  [')]
        assert all(l.endswith('(not enforced)') for l in listed), listed     # plain CLI prints only
        assert i18n_coverage.skeleton(i18n_coverage.Template(["'", None, "' broke"], 'runtime', 'x:1')) == (
            '    ("\'{x1}\' broke",\n     "\'{x1}\' broke",   # ru: TODO\n     "\'{x1}\' broke"),  # kk: TODO\n')
