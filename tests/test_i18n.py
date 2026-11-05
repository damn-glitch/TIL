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
import time

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
        saved = list(tilmsg.COMMAND_LINE_MESSAGES)
        try:
            tilmsg.COMMAND_LINE_MESSAGES.append(("broken {name}", "сломано", "бұзылған {name}"))
            tilmsg.COMMAND_LINE_MESSAGES.append(("lone { brace", "одинокая { скобка", "жалғыз { жақша"))
            tilmsg.COMMAND_LINE_MESSAGES.append(("{a}{b} glued", "{a}{b} склеены", "{a}{b} жабысқан"))
            tilmsg.COMMAND_LINE_MESSAGES.append(("{rest:text} at the end", "{rest} в конце", "соңында {rest}"))
            tilmsg.COMMAND_LINE_MESSAGES.append(("bad kind {n:number}", "плохой {n}", "жаман {n}"))
            problems = tilmsg.check_catalog()
            assert any("ru placeholders" in p for p in problems), problems
            assert any("unescaped '{'" in p for p in problems), problems
            assert any("adjacent placeholders" in p for p in problems), problems
            assert any("free-text placeholder rest" in p for p in problems), problems
            assert any("unknown placeholder kind 'number'" in p for p in problems), problems
            # MESSAGES is the three lists: an entry appended to it alone would never be used
            assert any("MESSAGES is not" in p for p in problems), problems
        finally:
            tilmsg.COMMAND_LINE_MESSAGES[:] = saved
        assert tilmsg.check_catalog() == []


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
        assert til.localize_report_bytes(b"runtime error: \xff\n", 'ru') == \
            "ошибка выполнения: \udcff\n".encode('utf-8', 'surrogateescape')

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
        # TIL_LANG and the program's pragma ask for a translation too: the note says why there
        # is none (review round 1: an installation without tilmsg.py ignored them silently)
        r = subprocess.run([sys.executable, str(alone / 'til.py'), 'check', prog], capture_output=True,
                           text=True, env=dict(os.environ, TIL_LANG='kk', PYTHONPATH=''))
        assert r.stderr.split('\n')[:2] == [lines[0], IMMUTABLE_HEADER['en']]
        pragma = write(tmp_path, 'ru.til', "#[lang: ru]\n" + IMMUTABLE)
        r = subprocess.run([sys.executable, str(alone / 'til.py'), 'check', pragma], capture_output=True,
                           text=True, env=dict(os.environ, TIL_LANG='', PYTHONPATH=''))
        assert r.stderr.split('\n')[:2] == [lines[0], IMMUTABLE_HEADER['en']]
        assert r.stderr.count("note:") == 1                    # once, however many diagnostics follow
        ok = write(tmp_path, 'ok.til', "#[lang: ru]\nprint(1)\n")
        r = subprocess.run([sys.executable, str(alone / 'til.py'), 'run', '--vm', ok], capture_output=True,
                           text=True, env=dict(os.environ, TIL_LANG='kk', PYTHONPATH=''))
        assert (r.returncode, r.stdout, r.stderr) == (0, "1\n", "")   # nothing to translate: no note


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
        localized = {lang: tilmsg.localize_report(err, lang) for lang in ('ru', 'kk')}
        for lang, text in localized.items():
            first = text.split('\n')[0]
            assert first != err.split('\n')[0] and CYRILLIC.search(first.split(':', 1)[1]), \
                f"{lang}: message not translated: {first!r}"
            assert " at " not in text and "line" not in text
        if CC:
            crc, cout, cerr = run_cvm(src)
            assert (crc, cout, cerr) == (rc, out, err), "the VMs report differently"
            assert {lang: tilmsg.localize_report(cerr, lang) for lang in ('ru', 'kk')} == localized


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


# ═══════════════════════════════════════════════════════════════
# REVIEW ROUND 1: REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════

# Reports that took the round-0 matcher cubic or quadratic time (minutes to
# hours at these sizes: a line of blanks, a message repeating a template's
# text, colour codes); each now takes milliseconds. The time limit leaves a
# margin of 100x and more, so it only fails if the work stops being linear.
PATHOLOGICAL = {
    'blank line': "runtime error: bad record:\n" + " " * 20000 + "x\n  at main (line 3)\n",
    'blanks and tabs': "runtime error: x\n" + " \t" * 10000 + "7 x\n",
    'quoted template repeated': "runtime error: '" + "x' takes y' takes " * 8000 + "\n",
    'expects repeated': "runtime error: " + "'f' expects " * 10000 + "\n",
    'colour codes': "runtime error: " + "\x1b[m" * 50000 + "x\n",
    'header colour codes': "error: " + "\x1b[1m" * 50000 + "x",
    'huge message': "runtime error: " + "word " * 200000 + "\n",
    'huge key': "runtime error: key not found: " + "a.b'c " * 20000 + "\n",
    'nested intents': "runtime error: " + "intent 'a' failed: " * 500 + "division by zero\n",
    'many lines': "runtime error: x\n" + "a line of program data\n" * 100000 + "  at main (line 2)\n",
}

# A program's own panic messages that look like catalog templates: they stay
# English (review round 1 saw "ожидалось 3 items, получено 2").
PROGRAM_TEXT = [
    "Expected 3 items, got 2",                       # parser: "Expected {expected}, got {got}"
    "cannot convert negative balance to int",        # VM: "cannot convert {type:type} to int"
    "cannot compare apples and oranges",             # VM: "cannot compare {left:type} and {right:type}"
    "Parser expects a number, got letters",          # VM: "{what:builtin} expects a number, got {type:type}"
    "File not found: config.txt",                    # command line: "File not found: {path:text}"
    "Unexpected token: foo",                         # parser: "Unexpected token: {token}"
    "Type mismatch: expected a number, got a string",
    "my list cannot be indexed",
]


def broken_installation(tmp_path, catalog_edit):
    """til.py with a tilmsg.py changed by `catalog_edit(text) -> text`, in a directory of its own."""
    here = tmp_path / 'inst'
    here.mkdir()
    shutil.copy(TIL, str(here / 'til.py'))
    with open(os.path.join(SRC, 'tilmsg.py'), encoding='utf-8') as f:
        original = f.read()
    catalog = catalog_edit(original)
    assert catalog != original
    (here / 'tilmsg.py').write_text(catalog, encoding='utf-8')
    return str(here / 'til.py')


def run_installed(til_py, *args, til_lang=''):
    return subprocess.run([sys.executable, til_py, *args], capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, TIL_LANG=til_lang, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1'))


class TestLinearTime:
    """Finding 1 (major): localizing a report could take cubic time."""

    @pytest.mark.parametrize("name", sorted(PATHOLOGICAL))
    def test_pathological_reports(self, name):
        text = PATHOLOGICAL[name]
        start = time.perf_counter()
        for lang in ('ru', 'kk'):
            out = tilmsg.localize_report(text, lang)
            assert out.count('\n') == text.count('\n')
            if name != 'many lines':      # (compiler diagnostics: short; reports: localize_report)
                assert tilmsg.localize(text, lang).count('\n') == text.count('\n')
        assert time.perf_counter() - start < 5.0, f"{name}: not linear"

    def test_program_data_in_a_report_is_copied(self):
        text = PATHOLOGICAL['blank line']
        assert tilmsg.localize_report(text, 'ru') == \
            "ошибка выполнения: bad record:\n" + " " * 20000 + "x\n  в main (строка 3)\n"
        # lines after the report's first one are the program's message: never looked up
        text = "runtime error: x\ndivision by zero\nerror: unknown name 'q'\n  at main (line 2)\n"
        assert tilmsg.localize_report(text, 'kk') == \
            "орындау қатесі: x\ndivision by zero\nerror: unknown name 'q'\n  main ішінде (2-жол)\n"

    def test_long_messages_are_not_looked_up(self):
        key = "k" * tilmsg.MAX_LOOKUP
        assert tilmsg.localize_report("runtime error: key not found: " + key + "\n", 'ru') == \
            "ошибка выполнения: key not found: " + key + "\n"           # the frame is still translated
        key = "k" * (tilmsg.MAX_LOOKUP - 100)
        assert tilmsg.localize_report("runtime error: key not found: " + key + "\n", 'ru') == \
            "ошибка выполнения: ключ не найден: " + key + "\n"

    @pytest.mark.parametrize("vm", ['--vm', pytest.param('--cvm', marks=needs_cc)])
    def test_a_program_cannot_stall_its_own_report(self, tmp_path, til_main, vm):
        # the program's pragma turns translation on; --max-steps does not bound the report
        prog = write(tmp_path, 'p.til', '#[lang: ru]\nmain()\n    let pad = " ".repeat(20000)\n'
                                        '    panic("bad record:\\n" + pad + "x")\n')
        start = time.perf_counter()
        r = til_main('run', vm, '--max-steps', '50', prog)
        assert time.perf_counter() - start < 30, "the report took too long"
        assert (r.returncode, r.stdout) == (1, "")
        assert r.stderr == ("ошибка выполнения: bad record:\n" + " " * 20000 + "x\n"
                            "  в main (строка 4)\n  в __start__ (строка 2)\n")

    def test_streamed_and_whole_reports_agree(self):
        # run_localized feeds the C VM's stderr line by line until the report starts, then the
        # rest at once; TILVM.report localizes the whole text: the results must be identical
        texts = [RUNTIME_REPORT['en'], PATHOLOGICAL['blank line'], PATHOLOGICAL['many lines'][:2000],
                 RUNTIME_REPORT['en'].replace('\n', '\r\n'),
                 "tilvm: cannot write /tmp/x\n[tilvm] paused\nfatal error: step limit exceeded (50)\n"
                 "  at f (line 9)\n  at main\n",
                 "runtime error: intent 'x' failed: key not found: a\nb\n  at main (line 2)",
                 "runtime error: no traceback", "", "\n\n"]
        for text in texts:
            for lang in ('ru', 'kk'):
                whole = tilmsg.localize_report(text, lang)
                lines = text.splitlines(keepends=True)
                loc = tilmsg.ReportLocalizer(lang)
                out, i = [], 0
                while i < len(lines) and not loc.in_report:
                    out.append(loc.feed(lines[i]))
                    i += 1
                out.append(loc.feed(''.join(lines[i:])))
                assert ''.join(out) == whole, text[:60]
                loc = tilmsg.ReportLocalizer(lang)
                assert ''.join(loc.feed(line) for line in lines) == whole, text[:60]

    def test_program_output_is_flushed_before_the_report_is_translated(self, monkeypatch):
        # the round-0 TILVM.report translated first and flushed the program's output after it
        events = []

        class Stdout(io.StringIO):
            def flush(self):
                events.append('flush')

        def localize_report_bytes(data, lang=None):
            events.append('translate')
            return data
        monkeypatch.setattr(sys, 'stdout', Stdout())
        monkeypatch.setattr(sys, 'stderr', io.StringIO())
        monkeypatch.setattr(til, 'localize_report_bytes', localize_report_bytes)
        module = til.TBCModule.from_bytes(compile_vm(RUNTIME).serialize())
        assert til.TILVM(module).run() == 1
        assert sys.stdout.getvalue() == "start\n"
        assert events[:2] == ['flush', 'translate']


def _drop_first_comma(text):
    # a missing comma between two entries: "TypeError: 'tuple' object is not callable"
    return text.replace("\"күтілмеген таңба: '{ch}'\"),", "\"күтілмеген таңба: '{ch}'\")", 1)


class TestCatalogFailures:
    """Finding 2: a tilmsg.py that fails to load must not replace diagnostics with a traceback."""

    BROKEN = {
        'missing comma': _drop_first_comma,
        'syntax error': lambda s: s.replace("LANGUAGES = ('en', 'ru', 'kk')", "LANGUAGES = ('en', 'ru', 'kk'", 1),
        'bad template': lambda s: s.replace('("division by zero",', '("division by zero {",', 1),
        'failing import': lambda s: s.replace("import re\n", "import re\nimport tilmsg_part_that_is_missing\n", 1),
    }

    @pytest.mark.parametrize("name", sorted(BROKEN))
    def test_english_with_a_note(self, tmp_path, name):
        til_py = broken_installation(tmp_path, self.BROKEN[name])
        imm = write(tmp_path, 'imm.til', IMMUTABLE)
        div = write(tmp_path, 'div.til', RUNTIME)
        for args, env, english in ((('check', '--lang', 'ru', imm), '', [IMMUTABLE_HEADER['en']]),
                                   (('check', imm), 'kk', [IMMUTABLE_HEADER['en']]),
                                   (('run', '--vm', div), 'kk', RUNTIME_REPORT['en'].split('\n'))):
            r = run_installed(til_py, *args, til_lang=env)
            assert r.returncode == 1 and "Traceback" not in r.stderr, r.stderr
            lines = r.stderr.split('\n')
            at = [i for i, line in enumerate(lines) if line.startswith("note: ")]
            assert len(at) == 1, r.stderr
            # before the note: only Python's own SyntaxWarning about the catalog, if any
            assert all("SyntaxWarning" in line or line.startswith("  ") for line in lines[:at[0]]), r.stderr
            note = lines[at[0]]
            assert note.startswith("note: message translations could not be loaded (tilmsg.py"), note
            assert note.endswith("); messages stay in English"), note
            assert lines[at[0] + 1:at[0] + 1 + len(english)] == english, r.stderr

    def test_a_failing_import_in_process(self, monkeypatch, capsys):
        def broken(name):
            raise SyntaxError("'(' was never closed", ('tilmsg.py', 38, 13, 'LANGUAGES = (\n'))
        monkeypatch.setattr(til, 'feature_module', broken)
        monkeypatch.setattr(til, '_MESSAGE_CATALOG', [])
        monkeypatch.setattr(til, '_CATALOG_NOTE_SHOWN', False)
        assert til.localize("error: x", 'ru') == "error: x"
        assert til.localize_report_bytes(b"runtime error: division by zero\n", 'kk') == \
            b"runtime error: division by zero\n"
        assert capsys.readouterr().err == ("note: message translations could not be loaded (tilmsg.py, "
                                           "line 38: SyntaxError: '(' was never closed); messages stay "
                                           "in English\n")


    @pytest.mark.parametrize("catalog,note", [
        (None, "note: message translations are not installed (tilmsg.py is missing); messages stay in English\n"),
        (object(), ""),                                   # a tilmsg.py of another version: plain English
    ], ids=['missing', 'other version'])
    def test_c_vm_stderr_without_a_usable_catalog(self, monkeypatch, capfd, catalog, note):
        report = "runtime error: division by zero\n  at main (line 2)\n"
        script = "import sys; sys.stderr.write(" + repr(report) + "); sys.exit(1)"
        monkeypatch.setattr(til, 'MESSAGE_LANG', 'kk')
        monkeypatch.setattr(til, '_MESSAGE_CATALOG', [catalog, None])
        monkeypatch.setattr(til, '_CATALOG_NOTE_SHOWN', False)
        capfd.readouterr()
        assert til.run_localized([sys.executable, '-c', script]) == 1
        assert capfd.readouterr().err == note + report


class TestProgramText:
    """Finding 3: a program's own message is not translated piecemeal."""

    @pytest.mark.parametrize("message", PROGRAM_TEXT)
    def test_free_text_stays_english(self, message):
        for lang, frame in (('ru', "ошибка выполнения: "), ('kk', "орындау қатесі: ")):
            assert tilmsg.localize_report(f"runtime error: {message}\n", lang) == f"{frame}{message}\n"
            nested = f"runtime error: intent 'load' failed: {message}\n"
            assert tilmsg.localize_report(nested, lang).endswith(f": {message}\n")

    @pytest.mark.parametrize("vm", ['--vm', pytest.param('--cvm', marks=needs_cc)])
    def test_programs(self, tmp_path, til_main, vm):
        prog = write(tmp_path, 'p.til',
                     'main()\n    let n = 2\n    if n != 3\n        panic(f"Expected 3 items, got {n}")\n')
        r = til_main('run', vm, '--lang', 'ru', prog)
        assert r.stderr.split('\n')[0] == "ошибка выполнения: Expected 3 items, got 2"
        prog = write(tmp_path, 'q.til', 'check(n: int)\n    if n < 0\n'
                                        '        panic("cannot convert negative balance to int")\n'
                                        'main()\n    check(0 - 5)\n')
        r = til_main('run', vm, '--lang', 'kk', prog)
        assert r.stderr == ("орындау қатесі: cannot convert negative balance to int\n"
                            "  check ішінде (3-жол)\n  main ішінде (5-жол)\n  __start__ ішінде (4-жол)\n")

    def test_real_messages_still_match(self):
        # the same templates still translate what the VMs and the compiler really say
        for en, ru in (
                ("runtime error: cannot convert str to int", "ошибка выполнения: нельзя преобразовать str в int"),
                ("runtime error: sqrt expects a number, got str",
                 "ошибка выполнения: sqrt ожидает число, получено str"),
                ("runtime error: key not found: my key", "ошибка выполнения: ключ не найден: my key"),
                ("runtime error: Assertion failed: x > 0 is false",
                 "ошибка выполнения: проверка не пройдена: x > 0 is false"),
                ("runtime error: intent 'load the data' failed: division by zero",
                 "ошибка выполнения: намерение 'load the data' не выполнено: деление на ноль")):
            assert tilmsg.localize_report(en, 'ru') == ru
        for en, ru in (
                ("Syntax Error: error: Expected INDENT, got NEWLINE",
                 "Синтаксическая ошибка: ожидалось INDENT, получено NEWLINE"),
                ("Error: File not found: my file.til", "Ошибка: файл не найден: my file.til"),
                ("Warning: Line 2: Type mismatch: expected [int; 3], got fn(int) -> int",
                 "Предупреждение: строка 2: несоответствие типов: ожидался [int; 3], получен fn(int) -> int")):
            assert tilmsg.localize(en, 'ru') == ru

    def test_runtime_templates_never_take_prose(self):
        # a placeholder that is not free text takes one word: a sentence written by the
        # program does not fit it (free-text placeholders follow a distinctive literal)
        sample = {'int': '3', 'type': 'int', 'builtin': 'abs', 'word': 'x', 'quoted': 'x', 'text': 'x'}
        checked = 0
        for e in tilmsg.entries():
            if e.scope != 'runtime':
                continue
            for i, p in enumerate(e.parts):
                if isinstance(p, str) or e.shapes[i] in ('text', 'quoted'):
                    continue
                message = ''.join(q if isinstance(q, str) else
                                  "negative balance" if q[0] == p[0] else sample[e.shapes[j]]
                                  for j, q in enumerate(e.parts))
                assert tilmsg.find(message, 'runtime') is None, (e.en, message)
                checked += 1
        assert checked > 50

    def test_placeholder_kinds(self):
        def match(template, message):
            return tilmsg.Entry(template, template, template).match(message)
        assert match("cannot convert {type} to int", "cannot convert str to int") == {'type': 'str'}
        assert match("cannot convert {type} to int", "cannot convert my str to int") is None     # a word
        assert match("unknown name '{name}'", "unknown name 'a b'") == {'name': 'a b'}            # quoted
        assert match("key not found: {key:text}", "key not found: a b") == {'key': 'a b'}
        assert match("got {n:int} items", "got -12 items") == {'n': '-12'}
        assert match("got {n:int} items", "got twelve items") is None
        assert match("{f:builtin} expects", "abs_2 expects") == {'f': 'abs_2'}
        assert match("{f:builtin} expects", "Abs expects") is None
        assert match("'{a}' uses '{b}', but '{a}' is", "'f' uses 'g', but 'f' is") == {'a': 'f', 'b': 'g'}
        assert match("'{a}' uses '{b}', but '{a}' is", "'f' uses 'g', but 'h' is") is None
        assert match("{a} to {b}", "x to y to z") is None       # the first " to "; "y to z" is no word
        for value, ok in (('int', True), ('struct', True), ('Point', True), ('Shape.Circle', True),
                          ('Нүкте', True), ('apples', False), ('point', False), ('Big Point', False)):
            assert (match("cannot negate {t:type}", "cannot negate " + value) is not None) == ok, value
        assert tilmsg.find("cannot convert str to int", 'compiler') is None                       # scopes
        assert tilmsg.find("cannot convert str to int", 'runtime')[0].en == "cannot convert {type:type} to int"
        with pytest.raises(ValueError):
            tilmsg.find("x", 'nowhere')


class TestRunnerAndWarnings:
    """Findings 4 and 5: the TIL_LANG warning, and the runners' 'tilvm:' / 'til:' messages."""

    def test_invalid_til_lang_warning_is_true(self, tmp_path, til_main):
        prog = write(tmp_path, 'p.til', "#[lang: ru]\nглавная()\n    пусть х = 1\n    х = 2\n")
        r = til_main('check', prog, til_lang='fr')
        lines = r.stderr.split('\n')
        assert lines[0] == "warning: TIL_LANG='fr' is not a message language (en, ru, kk); it is ignored"
        assert lines[1] == "ошибка[E-VM]: нельзя повторно присвоить значение неизменяемой переменной 'х'"

    KEYS = ("bad --keys entry 'FOO' (use UP, DOWN, LEFT, RIGHT, ENTER, ESC, SPACE, TAB, BACKSPACE, COMMA, "
            "NONE or one character)")
    KEYS_RU = ("неверный элемент --keys 'FOO' (используйте UP, DOWN, LEFT, RIGHT, ENTER, ESC, SPACE, TAB, "
               "BACKSPACE, COMMA, NONE или один символ)")

    def test_runner_prefixes(self):
        assert tilmsg.localize("Error: " + self.KEYS, 'ru') == "Ошибка: " + self.KEYS_RU
        assert tilmsg.localize("tilvm: " + self.KEYS, 'ru') == "tilvm: " + self.KEYS_RU
        assert tilmsg.localize_report("tilvm: " + self.KEYS + "\n", 'ru') == "tilvm: " + self.KEYS_RU + "\n"
        assert tilmsg.localize("til: --screen: the program did not call screen_init (no image written)", 'kk') \
            == "til: --screen: бағдарлама screen_init шақырмады (сурет жазылмады)"
        assert tilmsg.localize("til: cannot write /tmp/my image.ppm", 'ru') == \
            "til: не удаётся записать /tmp/my image.ppm"
        assert tilmsg.localize_report("tilvm: not enough memory for a 4096x4096 screen\n", 'ru') == \
            "tilvm: недостаточно памяти для экрана 4096x4096\n"
        # a 'tilvm:' line inside a program's message is the program's text
        assert tilmsg.localize_report("runtime error: x\ntilvm: " + self.KEYS + "\n", 'ru') == \
            "ошибка выполнения: x\ntilvm: " + self.KEYS + "\n"

    def test_c_vm_stderr_is_translated(self, monkeypatch, capfd):
        # run_localized() with a stand-in for the C VM's runner that rejects its options
        script = ("import sys; sys.stderr.write(" + repr("tilvm: " + self.KEYS + "\n") + "); "
                  "sys.stderr.flush(); sys.exit(2)")
        monkeypatch.setattr(til, 'MESSAGE_LANG', 'ru')
        capfd.readouterr()
        assert til.run_localized([sys.executable, '-c', script]) == 2
        assert capfd.readouterr().err == "tilvm: " + self.KEYS_RU + "\n"


class TestMergedBranches:
    """Messages of the branches merged after this one (effects, typed, prove, vmcore, gfx, z80):
    translated ahead of the merge, and extracted correctly by tools/i18n_coverage.py."""

    @pytest.mark.parametrize("en,ru,kk", [
        ("Syntax Error: error: Expected IDENT, got MATCH ('match' is a reserved word and cannot be used as a name)",
         "Синтаксическая ошибка: ожидалось IDENT, получено MATCH ('match' — зарезервированное слово, его нельзя "
         "использовать как имя)",
         "Синтаксистік қате: IDENT күтілді, бірақ MATCH кездесті ('match' — резервтелген сөз, оны атау ретінде "
         "қолдануға болмайды)"),
        ("  = hint: `perform Log(...)` calls the function 'Log' when no handler is active; add #[cap: Net] to 'f' "
         "(only main holds every capability)",
         "  = подсказка: `perform Log(...)` вызывает функцию 'Log', когда нет активного обработчика; добавьте "
         "#[cap: Net] к 'f' (все полномочия есть только у main)",
         "  = кеңес: белсенді өңдеуші болмаса, `perform Log(...)` 'Log' функциясын шақырады; 'f' функциясына "
         "#[cap: Net] қосыңыз (барлық өкілеттік тек main функциясында бар)"),
        ("Syntax Error: error: 'resume' is reserved inside an effect clause",
         "Синтаксическая ошибка: 'resume' — зарезервированное слово внутри ветви эффекта",
         "Синтаксистік қате: әсер тармағының ішінде 'resume' сөзі резервтелген"),
        ("error[E-PROVE]: 'f': could not prove #[ensures: result > 0] (unknown: timeout, 500 ms per query)",
         "ошибка[E-PROVE]: 'f': не удалось доказать #[ensures: result > 0] (неизвестно: истекло время, 500 мс на "
         "запрос)",
         "қате[E-PROVE]: 'f': #[ensures: result > 0] дәлелденбеді (белгісіз: уақыт бітті, бір сұрауға 500 мс)"),
    ])
    def test_translated_ahead(self, en, ru, kk):
        assert tilmsg.localize(en, 'ru') == ru
        assert tilmsg.localize(en, 'kk') == kk

    def test_free_text_at_the_start_of_a_template(self):
        saved = (list(tilmsg.RUNTIME_MESSAGES), list(tilmsg.COMPILER_MESSAGES))
        try:
            tilmsg.RUNTIME_MESSAGES.append(("{what:text} is broken beyond any repair whatsoever",
                                            "{what} сломано", "{what} бұзылған"))
            tilmsg.COMPILER_MESSAGES.append(("{what:msg} (x)", "{what} (х)", "{what} (х)"))
            problems = tilmsg.check_catalog()
            assert any("runtime entry" in p and "a program's own sentence" in p for p in problems), problems
            assert any("compiler entry" in p and "too little text" in p for p in problems), problems
        finally:
            tilmsg.RUNTIME_MESSAGES[:], tilmsg.COMPILER_MESSAGES[:] = saved

    def test_a_self_referential_hint_is_expanded_once(self):
        src = ("class SemanticChecker:\n"
               "    def error(self, node, msg, code='', hint=''):\n        pass\n"
               "    def check(self, node, key, callee, cap):\n"
               "        hint = f\"add #[cap: {cap}] to '{key}'\"\n"
               "        if node:\n"
               "            hint = f\"`perform {callee}(...)` calls '{callee}'; \" + hint\n"
               "        self.error(node, 'no capability', 'E-CAP', hint)\n")
        texts = _texts(i18n_coverage.PyExtractor(src, os.path.join(SRC, 'tilnew.py')).run())
        assert texts == {"no capability", "add #[cap: {}] to '{}'", "`perform {}(...)` calls '{}'; add #[cap: {}] to '{}'"}

    def test_methods_returning_text_are_expanded(self):
        src = ("class Parser:\n"
               "    def error(self, msg):\n        pass\n"
               "    def note(self):\n"
               "        if self.x:\n            return f\" ('{self.x}' is reserved)\"\n"
               "        return ''\n"
               "    def parse(self):\n"
               "        self.error('Expected a name' + self.note())\n")
        texts = _texts(i18n_coverage.PyExtractor(src, os.path.join(SRC, 'tilnew.py')).run())
        assert texts == {"Expected a name", "Expected a name ('{}' is reserved)"}

    def test_c_string_constants(self, tmp_path):
        # the z80 branch shares message text in constants to save ROM
        path = write(tmp_path, 'vm.c', 'static const char E_NO_METHOD[] = "no method \'";\n'
                                       'static int f(tilvm *vm, int mid) {\n'
                                       '    return failn(vm, E_NO_METHOD, METHOD_NAMES[mid], "\' on str", 0, 0, 0);\n}\n')
        assert _texts(i18n_coverage.extract_c(path)) == {"no method '{}' on str"}

    def test_c_conditional_literals(self, tmp_path):
        # til31/gfx: failn(vm, k ? "... argument 2 must be 1.." : "... argument 1 must be 1..", ...)
        path = write(tmp_path, 'vm.c',
                     'static int f(tilvm *vm, int k, int x) {\n    char lim[32], buf[32];\n'
                     '    return failn(vm, k ? "screen_init() argument 2 must be 1.." : '
                     '"screen_init() argument 1 must be 1..",\n'
                     '                 fmt_int(lim, x), ", got ", fmt_int(buf, x), 0, 0);\n}\n')
        templates = i18n_coverage.extract_c(path)
        assert _texts(templates) == {"screen_init() argument 1 must be 1..{}, got {}",
                                     "screen_init() argument 2 must be 1..{}, got {}"}
        for t in templates:                               # and the catalog has them
            assert i18n_coverage.coverage_of(tilmsg, t)[0] is True, t.text()

    def test_nested_messages_must_be_covered(self):
        outer = tilmsg.Entry("intent '{intent}' failed: {error:msg}", "a {intent} {error}", "b {intent} {error}",
                             scope='runtime')
        inner = tilmsg.Entry("division by zero", "деление на ноль", "нөлге бөлу", scope='runtime')
        broken = i18n_coverage.Template(["intent '", None, "' failed: the flux capacitor broke"], 'runtime', 'x:1')
        known = i18n_coverage.Template(["intent '", None, "' failed: division by zero"], 'runtime', 'x:1')
        assert i18n_coverage.coverage_of(tilmsg, broken, [outer, inner])[0] is False
        assert i18n_coverage.coverage_of(tilmsg, known, [outer, inner])[0] is True
