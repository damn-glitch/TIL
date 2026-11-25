#!/usr/bin/env python3
"""
TIL language services: the analysis core of TIL Studio and `til lsp`.

Everything here comes from the real compiler -- src/til.py's Lexer, Parser,
resolve_imports, SemanticChecker and BytecodeCompiler, plus src/tiltypes.py
(types, E-TYPE) when it is installed -- never from regular expressions over
the source:

    ws = Workspace(root, locale='ru')
    ws.open(uri, text, version)
    ws.diagnostics(uri)                    # LSP Diagnostic dicts (messages in Russian)
    ws.completion(uri, line, character)    # CompletionItem dicts
    ...

Positions are LSP positions: 0-based line, character offset in UTF-16 code
units (PositionMap converts to and from the compiler's 1-based line / column
in code points). Every result is a plain dict shaped like the LSP type of the
same name, so `til lsp` (src/tillsp.py) sends them as they are.

How diagnostics are collected (structurally, never by parsing formatted text):
ErrorReporter.format_error is wrapped once so that, while an analysis runs in
the current thread, every call records its arguments (file, line, column,
message, code, hint); the checkers are subclassed only to remember the AST
node an error is about (errors inside imported functions are reported on the
imported file). Outside an analysis the wrapper only calls the original.
"""

import bisect
import difflib
import os
import re
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote, unquote, urlparse

if 'til' in sys.modules:            # til.py loaded us (also as __main__): use that module
    til = sys.modules['til']
else:                               # direct import (tests, tools, lsp/til_lsp.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import til                      # noqa: E402

TT = til.TokenType

# ═══════════════════════════════════════════════════════════════════════════════
#                               LSP CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SEV_ERROR, SEV_WARNING, SEV_INFO, SEV_HINT = 1, 2, 3, 4


class CompletionKind:
    TEXT, METHOD, FUNCTION, CONSTRUCTOR, FIELD, VARIABLE, CLASS, INTERFACE, MODULE = 1, 2, 3, 4, 5, 6, 7, 8, 9
    PROPERTY, UNIT, VALUE, ENUM, KEYWORD, SNIPPET, COLOR, FILE, REFERENCE = 10, 11, 12, 13, 14, 15, 16, 17, 18
    FOLDER, ENUM_MEMBER, CONSTANT, STRUCT, EVENT, OPERATOR, TYPE_PARAMETER = 19, 20, 21, 22, 23, 24, 25


class SymbolKind:
    FILE, MODULE, NAMESPACE, PACKAGE, CLASS, METHOD, PROPERTY, FIELD, CONSTRUCTOR = 1, 2, 3, 4, 5, 6, 7, 8, 9
    ENUM, INTERFACE, FUNCTION, VARIABLE, CONSTANT, STRING, NUMBER, BOOLEAN, ARRAY = 10, 11, 12, 13, 14, 15, 16, 17, 18
    OBJECT, KEY, NULL, ENUM_MEMBER, STRUCT, EVENT, OPERATOR, TYPE_PARAMETER = 19, 20, 21, 22, 23, 24, 25, 26


INSERT_PLAIN, INSERT_SNIPPET = 1, 2

# Semantic tokens legend (semanticTokensProvider.legend). A token's type is an
# index into TOKEN_TYPES; its modifiers a bit set over TOKEN_MODIFIERS.
TOKEN_TYPES = [
    'namespace',      # 0  module names (import m, m.f())
    'type',           # 1  builtin types (int, str, ...), type aliases
    'struct',         # 2  struct names
    'enum',           # 3  enum names
    'interface',      # 4  traits
    'typeParameter',  # 5  generic parameters <T>
    'parameter',      # 6  function / lambda / clause parameters, self
    'variable',       # 7  locals, globals, constants
    'property',       # 8  struct fields
    'enumMember',     # 9  enum variants
    'function',       # 10 functions (builtins: + defaultLibrary)
    'method',         # 11 methods (builtin methods: + defaultLibrary)
    'keyword',        # 12 keywords in every language, contextual effect words
    'comment',        # 13 comments, #[lang: ...] pragmas
    'string',         # 14 string, char and f-string text
    'number',         # 15 numeric literals
    'operator',       # 16 operators
    'decorator',      # 17 attributes #[...]
]
TOKEN_MODIFIERS = [
    'declaration',     # bit 0: the defining occurrence
    'readonly',        # bit 1: let / const / immutable globals
    'defaultLibrary',  # bit 2: builtins (print, len, push, ...)
    'static',          # bit 3: Type.function without self
    'modification',    # bit 4: assigned here
]
_TT_INDEX = {name: i for i, name in enumerate(TOKEN_TYPES)}
MOD_DECL, MOD_READONLY, MOD_BUILTIN, MOD_STATIC, MOD_WRITE = 1, 2, 4, 8, 16


# ═══════════════════════════════════════════════════════════════════════════════
#                       POSITIONS (LSP UTF-16  <->  compiler)
# ═══════════════════════════════════════════════════════════════════════════════

def utf16_len(s: str) -> int:
    """Length of `s` in UTF-16 code units (astral characters count 2)."""
    if s.isascii():
        return len(s)
    return len(s) + sum(1 for c in s if ord(c) > 0xFFFF)


class PositionMap:
    """Converts between code point offsets into `text`, LSP positions
    (0-based line, UTF-16 character) and the compiler's 1-based line / column
    (code points). Line ends: '\\n', '\\r\\n' and '\\r' (as LSP)."""

    def __init__(self, text: str):
        self.text = text
        starts = [0]
        for m in re.finditer(r'\r\n|\r|\n', text):
            starts.append(m.end())
        self.line_starts = starts
        # end of each line's content (before its line break)
        self.line_ends = []
        for i, s in enumerate(starts):
            if i + 1 < len(starts):
                e = starts[i + 1] - 1
                if e > s and text[e] == '\n' and text[e - 1] == '\r':
                    e -= 1
                self.line_ends.append(e)
            else:
                self.line_ends.append(len(text))

    @property
    def line_count(self) -> int:
        return len(self.line_starts)

    def line_text(self, line: int) -> str:
        if not 0 <= line < len(self.line_starts):
            return ""
        return self.text[self.line_starts[line]:self.line_ends[line]]

    def line_of(self, offset: int) -> int:
        offset = max(0, min(offset, len(self.text)))
        return bisect.bisect_right(self.line_starts, offset) - 1

    def position(self, offset: int) -> dict:
        """Code point offset -> LSP Position."""
        offset = max(0, min(offset, len(self.text)))
        line = self.line_of(offset)
        start = self.line_starts[line]
        end = self.line_ends[line]
        return {'line': line, 'character': utf16_len(self.text[start:min(offset, end)]) +
                max(0, offset - end)}

    def offset(self, line: int, character: int) -> int:
        """LSP Position -> code point offset (clamped to the line)."""
        if line < 0:
            return 0
        if line >= len(self.line_starts):
            return len(self.text)
        start, end = self.line_starts[line], self.line_ends[line]
        seg = self.text[start:end]
        if character <= 0:
            return start
        if seg.isascii():
            return start + min(character, len(seg))
        units = 0
        for i, c in enumerate(seg):
            if units >= character:
                return start + i
            units += 2 if ord(c) > 0xFFFF else 1
            if units > character:          # inside a surrogate pair: the character's start
                return start + i
        return end

    def range(self, start: int, end: int) -> dict:
        return {'start': self.position(start), 'end': self.position(max(start, end))}

    def til_to_offset(self, line: int, col: int) -> int:
        """Compiler 1-based line / column (code points) -> offset."""
        if line < 1:
            return 0
        if line > len(self.line_starts):
            return len(self.text)
        start = self.line_starts[line - 1]
        return min(start + max(col, 1) - 1, self.line_ends[line - 1] + 1, len(self.text))

    def til_to_lsp(self, line: int, col: int) -> dict:
        return self.position(self.til_to_offset(line, col))

    def lsp_to_til(self, line: int, character: int) -> Tuple[int, int]:
        off = self.offset(line, character)
        ln = self.line_of(off)
        return ln + 1, off - self.line_starts[ln] + 1


# ═══════════════════════════════════════════════════════════════════════════════
#                                  URIS
# ═══════════════════════════════════════════════════════════════════════════════

def uri_to_path(uri: str) -> Optional[str]:
    """file:///home/x/a.til -> /home/x/a.til (None for untitled:, inmemory:, ...)."""
    if not uri:
        return None
    if '://' not in uri and not uri.startswith('file:'):
        return os.path.abspath(uri) if os.path.isabs(uri) else None
    p = urlparse(uri)
    if p.scheme != 'file':
        return None
    path = unquote(p.path)
    if p.netloc and p.netloc != 'localhost':          # UNC: file://server/share/x
        path = '//' + p.netloc + path
    if os.name == 'nt' or re.match(r'^/[A-Za-z]:', path):
        if re.match(r'^/[A-Za-z]:', path):
            path = path[1:]
        if os.name == 'nt':
            path = path.replace('/', '\\')
    return path


def path_to_uri(path: str) -> str:
    path = os.path.abspath(path)
    if os.name == 'nt':
        path = path.replace('\\', '/')
        if re.match(r'^[A-Za-z]:', path):
            path = '/' + path
    return 'file://' + quote(path, safe='/:')


def path_key(path: Optional[str]) -> str:
    return til.module_file_key(path) if path else ""


# ═══════════════════════════════════════════════════════════════════════════════
#                     STRUCTURAL CAPTURE OF COMPILER DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

class RawDiag:
    """One diagnostic as the compiler produced it (arguments of format_error)."""
    __slots__ = ('filename', 'source', 'line', 'col', 'text', 'code', 'tip', 'node', 'severity')

    def __init__(self, filename, source, line, col, text, code="", tip="", node=None, severity=SEV_ERROR):
        self.filename = filename or ""
        self.source = source or ""
        self.line = line or 0
        self.col = col or 1
        self.text = text or ""
        self.code = code or ""
        self.tip = tip or ""
        self.node = node
        self.severity = severity

    def __repr__(self):
        return f"RawDiag({self.filename}:{self.line}:{self.col} [{self.code}] {self.text!r})"


_TLS = threading.local()
_HOOK_LOCK = threading.Lock()


def _install_capture():
    """Wrap ErrorReporter.format_error once: while a sink is active in this
    thread, record its arguments (it still returns the same text)."""
    cls = til.ErrorReporter
    current = cls.__dict__.get('format_error')
    if getattr(current, '_tills_capture', False):
        return
    with _HOOK_LOCK:
        current = cls.__dict__.get('format_error')
        if getattr(current, '_tills_capture', False):
            return
        original = current

        def format_error(self, line, col, msg, error_code="", hint=""):
            sink = getattr(_TLS, 'sink', None)
            if sink is not None:
                node = getattr(_TLS, 'node', None)
                _TLS.node = None
                sink.append(RawDiag(getattr(self, 'filename', ''), getattr(self, 'source', ''),
                                    line, col, msg, error_code, hint, node))
            return original(self, line, col, msg, error_code, hint)

        format_error._tills_capture = True
        format_error.__wrapped__ = original
        cls.format_error = format_error


class _Capture:
    """`with _Capture() as raws:` collects every format_error call of this thread."""

    def __enter__(self) -> List[RawDiag]:
        _install_capture()
        self.prev = getattr(_TLS, 'sink', None)
        self.items: List[RawDiag] = []
        _TLS.sink = self.items
        _TLS.node = None
        return self.items

    def __exit__(self, *exc):
        _TLS.sink = self.prev
        _TLS.node = None
        return False


class _QuietStderr:
    """Swallow what the compiler prints to stderr from THIS thread (resolve_imports
    warns about missing modules itself; the language server reports them as
    diagnostics). Other threads keep writing to the real stream."""

    class _Proxy:
        def __init__(self, real, owner):
            self._real = real
            self._owner = owner
            self.captured: List[str] = []

        def write(self, s):
            if threading.get_ident() == self._owner:
                self.captured.append(s)
                return len(s)
            return self._real.write(s)

        def flush(self):
            if threading.get_ident() != self._owner:
                self._real.flush()

        def __getattr__(self, name):
            return getattr(self._real, name)

    def __enter__(self):
        self.real = sys.stderr
        self.proxy = self._Proxy(self.real, threading.get_ident())
        sys.stderr = self.proxy
        return self.proxy

    def __exit__(self, *exc):
        if sys.stderr is self.proxy:
            sys.stderr = self.real
        return False


class _Checker(til.SemanticChecker):
    """SemanticChecker that remembers the node of each error (for the file)."""

    def error(self, node, *args, **kwargs):
        _TLS.node = node
        try:
            return super().error(node, *args, **kwargs)
        finally:
            _TLS.node = None


class _Bytecode(til.BytecodeCompiler):
    """BytecodeCompiler that remembers the node of its (first, fatal) error."""

    def error(self, node, *args, **kwargs):
        _TLS.node = node
        try:
            return super().error(node, *args, **kwargs)
        finally:
            _TLS.node = None


class _ClassicTypes(til.TypeChecker):
    """The classic type checker (used by the compiler only without tiltypes):
    its warnings, structurally."""

    def __init__(self):
        super().__init__()
        self.records: List[Tuple[str, Any]] = []

    def error(self, *args, **kwargs):
        text = args[0] if args else kwargs.get('msg', '')
        node = args[1] if len(args) > 1 else kwargs.get('node')
        self.records.append((text, node))
        return super().error(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
#                     KEYWORDS OF EVERY LANGUAGE (from til.py)
# ═══════════════════════════════════════════════════════════════════════════════

KEYWORD_LANGS = ('en', 'kk', 'kk-latn', 'ru')
LANG_NAMES = {'en': 'English', 'kk': 'Қазақша', 'kk-latn': 'Qazaqsha (latyn)', 'ru': 'Русский'}


def _keyword_table() -> Dict[str, Dict[str, List[str]]]:
    """'IF' -> {'en': ['if'], 'kk': ['егер'], 'kk-latn': ['eger'], 'ru': ['если']}.
    true/false are 'TRUE' / 'FALSE'; contextual effect words are included."""
    out: Dict[str, Dict[str, List[str]]] = {}

    def add(tt, word, lang):
        name = tt.name
        if tt == TT.BOOL:
            name = 'TRUE' if word in til.TRUE_WORDS else 'FALSE'
        words = out.setdefault(name, {}).setdefault(lang, [])
        if word not in words:
            words.append(word)

    for word, tt in til.ENGLISH_KEYWORDS.items():
        if word in ('True', 'False'):
            continue
        add(tt, word, 'en')
    for lang in ('kk', 'kk-latn', 'ru'):
        for word, tt in til.LANGUAGES[lang][0].items():
            add(tt, word, lang)
    for lang, table in til.CONTEXTUAL_KEYWORDS.items():
        for word, tt in table.items():
            add(tt, word, lang)
    return out


KEYWORDS_BY_TYPE = _keyword_table()
# word -> (keyword type name, language)
KEYWORD_WORDS: Dict[str, Tuple[str, str]] = {}
for _name, _langs in KEYWORDS_BY_TYPE.items():
    for _lang, _words in _langs.items():
        for _w in _words:
            KEYWORD_WORDS.setdefault(_w, (_name, _lang))
KEYWORD_WORDS['True'] = ('TRUE', 'en')
KEYWORD_WORDS['False'] = ('FALSE', 'en')
CONTEXTUAL_WORDS = {w for table in til.CONTEXTUAL_KEYWORDS.values() for w in table}
# builtin aliases: localized spelling -> (canonical builtin, language)
ALIAS_WORDS: Dict[str, Tuple[str, str]] = {}
for _lang in ('kk', 'kk-latn', 'ru'):
    for _alias, _canon in til.LANGUAGES[_lang][1].items():
        ALIAS_WORDS.setdefault(_alias, (_canon, _lang))


def keyword_spelling(name: str, lang: str) -> str:
    """The first spelling of a keyword type in a language (English if none).
    ELIF is written `else if` outside English."""
    if name == 'ELIF' and lang != 'en':
        return f"{keyword_spelling('ELSE', lang)} {keyword_spelling('IF', lang)}"
    words = KEYWORDS_BY_TYPE.get(name, {})
    if lang in words:
        return words[lang][0]
    return words.get('en', [name.lower()])[0]


def message_lang(locale: Optional[str]) -> str:
    """'ru-RU' / 'kk' / 'kk-latn' / 'en_US' -> 'ru' / 'kk' / 'en' (messages)."""
    return til.message_language_code(locale) or 'en'


def localize(text: str, lang: str) -> str:
    """A message in the workspace language (src/tilmsg.py; English stays English)."""
    if not text or lang == 'en':
        return text
    return til.localize(text, lang)


# ═══════════════════════════════════════════════════════════════════════════════
#                 BUILT-IN DOCUMENTATION (en / ru / kk), compact tables
# ═══════════════════════════════════════════════════════════════════════════════
# Keyword docs: keyword type -> (English, Russian, Kazakh). Short, one or two
# sentences; the hover adds the keyword's spelling in every language.

KEYWORD_DOCS: Dict[str, Tuple[str, str, str]] = {
    'IF': ("Runs the indented block when the condition is true; continue with `elif` / `else`.",
           "Выполняет вложенный блок, если условие истинно; продолжение — `elif` / `else`.",
           "Шарт ақиқат болса, ішкі блокты орындайды; жалғасы — `elif` / `else`."),
    'ELSE': ("The branch taken when no condition above was true; also the fallback of `retry` / `timeout`.",
             "Ветка, если ни одно условие выше не выполнено; также запасной вариант `retry` / `timeout`.",
             "Жоғарыдағы шарттардың ешқайсысы орындалмаса таңдалатын тармақ; `retry` / `timeout` үшін "
             "қосалқы нұсқа."),
    'ELIF': ("Another condition after `if` (in Kazakh and Russian written as `else if`).",
             "Ещё одно условие после `if` (по-русски пишется `иначе если`).",
             "`if` кейінгі тағы бір шарт (қазақша `әйтпесе егер` деп жазылады)."),
    'FOR': ("Loops over a range, array, string or map: `for x in 0..10`.",
            "Цикл по диапазону, массиву, строке или словарю: `for x in 0..10`.",
            "Ауқым, массив, жол немесе сөздік бойынша цикл: `for x in 0..10`."),
    'WHILE': ("Repeats the block while the condition is true.",
              "Повторяет блок, пока условие истинно.",
              "Шарт ақиқат болғанша блокты қайталайды."),
    'LOOP': ("Repeats the block forever; leave it with `break` or `return`.",
             "Бесконечный цикл; выход — `break` или `return`.",
             "Шексіз цикл; одан `break` немесе `return` арқылы шығады."),
    'IN': ("Separates the loop variable from what `for` iterates over.",
           "Отделяет переменную цикла `for` от того, по чему идёт цикл.",
           "`for` цикл айнымалысын қайталанатын мәннен бөледі."),
    'RETURN': ("Leaves the function, giving back a value.",
               "Выходит из функции и возвращает значение.",
               "Функциядан шығып, мән қайтарады."),
    'BREAK': ("Leaves the innermost loop.", "Выходит из ближайшего цикла.", "Ең ішкі циклдан шығады."),
    'CONTINUE': ("Starts the next iteration of the innermost loop.",
                 "Переходит к следующей итерации ближайшего цикла.",
                 "Ең ішкі циклдың келесі қадамына өтеді."),
    'FN': ("Defines a function (optional: `name(params) -> T` works without it).",
           "Определяет функцию (необязательно: `имя(параметры) -> T` работает и без него).",
           "Функция анықтайды (міндетті емес: `аты(параметрлер) -> T` онсыз да жұмыс істейді)."),
    'LET': ("Declares an immutable variable: `let x = 1`.",
            "Объявляет неизменяемую переменную: `let x = 1`.",
            "Өзгермейтін айнымалы жариялайды: `let x = 1`."),
    'VAR': ("Declares a mutable variable: `var n = 0`.",
            "Объявляет изменяемую переменную: `var n = 0`.",
            "Өзгермелі айнымалы жариялайды: `var n = 0`."),
    'CONST': ("Declares a compile-time constant: `const MAX = 100`.",
              "Объявляет константу времени компиляции: `const MAX = 100`.",
              "Компиляция кезіндегі тұрақтыны жариялайды: `const MAX = 100`."),
    'MUT': ("Marks a parameter or reference as mutable.",
            "Помечает параметр или ссылку как изменяемые.",
            "Параметрді немесе сілтемені өзгермелі деп белгілейді."),
    'PUB': ("Marks a definition as public.", "Помечает определение как публичное.",
            "Анықтаманы ашық деп белгілейді."),
    'STRUCT': ("Defines a struct: named fields with types.",
               "Определяет структуру: именованные поля с типами.",
               "Құрылым анықтайды: типтері бар аталған өрістер."),
    'ENUM': ("Defines an enum; variants may carry data: `Circle(float)`.",
             "Определяет перечисление; варианты могут нести данные: `Circle(float)`.",
             "Санақ анықтайды; нұсқалар деректер ұстай алады: `Circle(float)`."),
    'IMPL': ("Adds methods to a struct: `impl Point` or `impl Trait for Point`.",
             "Добавляет методы структуре: `impl Point` или `impl Trait for Point`.",
             "Құрылымға әдістер қосады: `impl Point` немесе `impl Trait for Point`."),
    'TRAIT': ("Defines a trait: methods a type promises, with optional default bodies.",
              "Определяет типаж: методы, которые обещает тип (возможны реализации по умолчанию).",
              "Қасиет анықтайды: тип уәде ететін әдістер (әдепкі денесі болуы мүмкін)."),
    'MATCH': ("Pattern matching: the first arm whose pattern fits (and guard holds) runs.",
              "Сопоставление с образцом: выполняется первая подходящая ветка (с выполненным условием).",
              "Үлгімен салыстыру: үлгісі сәйкес келген (шарты орындалған) бірінші тармақ орындалады."),
    'IMPORT': ("Imports a module: `import math_utils`, `import m as alias`.",
               "Импортирует модуль: `import math_utils`, `import m as alias`.",
               "Модульді импорттайды: `import math_utils`, `import m as alias`."),
    'FROM': ("Imports chosen names: `from m import a, b`.",
             "Импортирует выбранные имена: `from m import a, b`.",
             "Таңдалған атауларды импорттайды: `from m import a, b`."),
    'AS': ("Type cast (`x as float`) or import alias (`import m as k`).",
           "Приведение типа (`x as float`) или псевдоним импорта (`import m as k`).",
           "Типті түрлендіру (`x as float`) немесе импорт бүркеншік аты (`import m as k`)."),
    'TYPE': ("Defines a type alias: `type Score = int`.",
             "Определяет псевдоним типа: `type Score = int`.",
             "Тип бүркеншік атын анықтайды: `type Score = int`."),
    'SELF': ("The value a method is called on.", "Значение, у которого вызван метод.",
             "Әдіс шақырылған мән."),
    'AND': ("Logical and (short-circuit).", "Логическое И (с сокращённым вычислением).",
            "Логикалық ЖӘНЕ (қысқартылған есептеу)."),
    'OR': ("Logical or (short-circuit).", "Логическое ИЛИ (с сокращённым вычислением).",
           "Логикалық НЕМЕСЕ (қысқартылған есептеу)."),
    'NOT': ("Logical negation.", "Логическое отрицание.", "Логикалық терістеу."),
    'TRUE': ("The boolean value true.", "Логическое значение «истина».", "«Ақиқат» логикалық мәні."),
    'FALSE': ("The boolean value false.", "Логическое значение «ложь».", "«Жалған» логикалық мәні."),
    'ASSERT': ("Checks a condition at run time: `assert x > 0, \"message\"` (retry can catch it).",
               "Проверяет условие во время выполнения: `assert x > 0, \"сообщение\"` (retry может перехватить).",
               "Орындау кезінде шартты тексереді: `assert x > 0, \"хабар\"` (retry ұстай алады)."),
    'INVARIANT': ("States a condition that must always hold (checked at run time).",
                  "Условие, которое должно выполняться всегда (проверяется во время выполнения).",
                  "Әрқашан орындалуы тиіс шарт (орындау кезінде тексеріледі)."),
    'INTENT': ("Names what a block is trying to do; errors inside report the intent.",
               "Описывает цель блока; ошибки внутри сообщают об этой цели.",
               "Блоктың мақсатын атайды; ішіндегі қателер сол мақсатты хабарлайды."),
    'RETRY': ("Runs the body again after a runtime error, up to n times: `retry(3) ... else ...`.",
              "Повторяет тело после ошибки выполнения до n раз: `retry(3) ... else ...`.",
              "Орындау қатесінен кейін денені n ретке дейін қайталайды: `retry(3) ... else ...`."),
    'FALLBACK': ("A default value used when the expression fails.",
                 "Значение по умолчанию, если выражение завершилось ошибкой.",
                 "Өрнек қатемен аяқталса қолданылатын әдепкі мән."),
    'TIMEOUT': ("Aborts the body after the given milliseconds: `timeout(100) ... else ...`.",
                "Прерывает тело через заданное число миллисекунд: `timeout(100) ... else ...`.",
                "Денені берілген миллисекундтан кейін тоқтатады: `timeout(100) ... else ...`."),
    'CAP': ("Declares a capability: `cap Storage { read, write }`; functions require it with `#[cap: Storage]`.",
            "Объявляет право доступа: `cap Storage { read, write }`; функции требуют его через `#[cap: Storage]`.",
            "Құқықты жариялайды: `cap Storage { read, write }`; функциялар оны `#[cap: Storage]` арқылы талап етеді."),
    'PERFORM': ("Performs an effect: `perform Ask(\"name\")`; the nearest handler answers.",
                "Выполняет эффект: `perform Ask(\"name\")`; отвечает ближайший обработчик.",
                "Эффект орындайды: `perform Ask(\"name\")`; ең жақын өңдеуші жауап береді."),
    'HANDLE': ("Runs a body with effect clauses installed: `handle ... with Name(x) => ...`.",
               "Выполняет тело с обработчиками эффектов: `handle ... with Name(x) => ...`.",
               "Денені эффект өңдеушілерімен орындайды: `handle ... with Name(x) => ...`."),
    'WITH': ("Starts the effect clauses of a `handle`.",
             "Начинает обработчики эффектов в `handle`.",
             "`handle` ішіндегі эффект өңдеушілерін бастайды."),
    'RESUME': ("Inside an effect clause: continue the handled body with a value: `resume(v)`.",
               "В обработчике эффекта: продолжить обрабатываемое тело со значением: `resume(v)`.",
               "Эффект өңдеушісінде: өңделетін денені мәнмен жалғастыру: `resume(v)`."),
}

# Builtin functions: name -> (parameters, return type, English, Russian, Kazakh).
# Parameters: a tuple of names (a trailing '...' = any number of arguments,
# '?name' = optional); arity itself comes from til.NATIVE_ARITY.
BUILTIN_DOCS: Dict[str, Tuple[Tuple[str, ...], str, str, str, str]] = {
    'print': (('values', '...'), '', "Prints each value on its own line (no arguments: an empty line).",
              "Печатает каждое значение на отдельной строке (без аргументов — пустую строку).",
              "Әр мәнді жеке жолға шығарады (аргументсіз — бос жол)."),
    'println': (('values', '...'), '', "Same as print.", "То же, что print.", "print-пен бірдей."),
    'len': (('x',), 'int', "Length of a string (in bytes), an array or a map.",
            "Длина строки (в байтах), массива или словаря.",
            "Жолдың (байтпен), массивтің немесе сөздіктің ұзындығы."),
    'str': (('x',), 'str', "The text form of any value.", "Текстовое представление значения.",
            "Кез келген мәннің мәтіндік түрі."),
    'int': (('x',), 'int', "Converts a number, bool or numeric string to an integer.",
            "Преобразует число, bool или числовую строку в целое.",
            "Санды, bool мәнін немесе сандық жолды бүтін санға түрлендіреді."),
    'float': (('x',), 'float', "Converts a number or numeric string to a float.",
              "Преобразует число или числовую строку в дробное число.",
              "Санды немесе сандық жолды бөлшек санға түрлендіреді."),
    'bool': (('x',), 'bool', "Truthiness of a value (0, \"\", [], None are false).",
             "Истинность значения (0, \"\", [], None — ложь).",
             "Мәннің ақиқаттығы (0, \"\", [], None — жалған)."),
    'abs': (('x',), 'int | float', "Absolute value (an int stays an int).",
            "Абсолютное значение (целое остаётся целым).", "Абсолют мән (бүтін сан бүтін болып қалады)."),
    'sqrt': (('x',), 'float', "Square root.", "Квадратный корень.", "Квадрат түбір."),
    'pow': (('base', 'exponent'), 'float', "base raised to exponent.", "Возведение base в степень exponent.",
            "base санын exponent дәрежесіне шығару."),
    'sin': (('x',), 'float', "Sine (radians).", "Синус (в радианах).", "Синус (радианмен)."),
    'cos': (('x',), 'float', "Cosine (radians).", "Косинус (в радианах).", "Косинус (радианмен)."),
    'tan': (('x',), 'float', "Tangent (radians).", "Тангенс (в радианах).", "Тангенс (радианмен)."),
    'log': (('x',), 'float', "Natural logarithm.", "Натуральный логарифм.", "Натурал логарифм."),
    'exp': (('x',), 'float', "e raised to x.", "e в степени x.", "e санының x дәрежесі."),
    'floor': (('x',), 'float', "Rounds down.", "Округление вниз.", "Төмен қарай дөңгелектеу."),
    'ceil': (('x',), 'float', "Rounds up.", "Округление вверх.", "Жоғары қарай дөңгелектеу."),
    'round': (('x',), 'float', "Rounds to the nearest whole number (halves away from zero).",
              "Округление до ближайшего целого (половины — от нуля).",
              "Ең жақын бүтін санға дөңгелектеу (жартысы нөлден алысқа)."),
    'min': (('values', '...'), 'value', "The smallest argument (or the smallest element of one array).",
            "Наименьший аргумент (или наименьший элемент массива).",
            "Ең кіші аргумент (немесе массивтің ең кіші элементі)."),
    'max': (('values', '...'), 'value', "The largest argument (or the largest element of one array).",
            "Наибольший аргумент (или наибольший элемент массива).",
            "Ең үлкен аргумент (немесе массивтің ең үлкен элементі)."),
    'sum': (('values', '...'), 'int | float', "Sum of numbers (the arguments or one array).",
            "Сумма чисел (аргументов или одного массива).", "Сандардың қосындысы (аргументтер немесе бір массив)."),
    'input': (('?prompt',), 'str', "Reads a line from standard input (prints the prompt first).",
              "Читает строку со стандартного ввода (сначала печатает подсказку).",
              "Стандарт кірістен жол оқиды (алдымен шақыруды шығарады)."),
    'chr': (('code',), 'str', "The character with this Unicode code point.",
            "Символ с этим кодом Unicode.", "Осы Unicode кодындағы таңба."),
    'ord': (('s',), 'int', "The Unicode code point of the first character.",
            "Код Unicode первого символа.", "Бірінші таңбаның Unicode коды."),
    'range': (('start', 'end', '?inclusive'), '[int]', "Integers start..end (range(n) = 0..n).",
              "Целые числа start..end (range(n) = 0..n).", "start..end бүтін сандары (range(n) = 0..n)."),
    'type_of': (('x',), 'str', "The name of a value's type.", "Имя типа значения.", "Мән типінің атауы."),
    'format': (('value', 'spec'), 'str', "Formats a value with an f-string spec: \".2f\", \"5\", \"x\".",
               "Форматирует значение как в f-строке: \".2f\", \"5\", \"x\".",
               "Мәнді f-жолдағыдай пішімдейді: \".2f\", \"5\", \"x\"."),
    'clock_ms': ((), 'int', "Milliseconds of a monotonic clock.", "Миллисекунды монотонных часов.",
                 "Монотонды сағаттың миллисекундтары."),
    'random': (('?lo', '?hi'), 'float | int', "random(): a float in [0, 1); random(lo, hi): an int in lo..=hi.",
               "random(): дробное в [0, 1); random(lo, hi): целое в lo..=hi.",
               "random(): [0, 1) аралығындағы бөлшек; random(lo, hi): lo..=hi аралығындағы бүтін."),
    'seed': (('n',), '', "Seeds the random number generator.", "Задаёт начальное значение генератора случайных чисел.",
             "Кездейсоқ сандар генераторының бастапқы мәнін береді."),
    'sleep_ms': (('ms',), '', "Pauses the program.", "Приостанавливает программу.", "Бағдарламаны кідіртеді."),
    'exit': (('?code',), '', "Ends the program with an exit code.", "Завершает программу с кодом выхода.",
             "Бағдарламаны шығу кодымен аяқтайды."),
    'panic': (('?message',), '', "Raises a runtime error (retry / timeout can catch it).",
              "Вызывает ошибку выполнения (retry / timeout могут её перехватить).",
              "Орындау қатесін тудырады (retry / timeout оны ұстай алады)."),
    'iter': (('x',), '[value]', "What a for loop iterates: the keys of a map, an array, a string.",
             "То, по чему идёт цикл for: ключи словаря, массив, строка.",
             "for циклы жүретін мән: сөздік кілттері, массив, жол."),
    'read_file': (('path',), 'str', "The contents of a file.", "Содержимое файла.", "Файлдың мазмұны."),
    'write_file': (('path', 'text'), '', "Writes text to a file (replacing it).",
                   "Записывает текст в файл (заменяя его).", "Мәтінді файлға жазады (оны ауыстырып)."),
    'file_exists': (('path',), 'bool', "True when the file can be opened.", "True, если файл можно открыть.",
                    "Файлды ашуға болса True."),
    'args': ((), '[str]', "The program's command-line arguments.", "Аргументы командной строки программы.",
             "Бағдарламаның командалық жол аргументтері."),
    'bytes_to_str': (('bytes',), 'str', "A string from an array of byte values 0..255.",
                     "Строка из массива байтов 0..255.", "0..255 байттар массивінен жол."),
    'str_bytes': (('s',), '[int]', "The UTF-8 bytes of a string.", "Байты UTF-8 строки.", "Жолдың UTF-8 байттары."),
    'screen_init': (('w', 'h'), '', "Opens (or resizes) a w×h screen, all black.",
                    "Открывает (или меняет размер) экран w×h, весь чёрный.",
                    "w×h экранды ашады (немесе өлшемін өзгертеді), толығымен қара."),
    'screen_clear': (('?color',), '', "Fills the screen (default black).", "Заливает экран (по умолчанию чёрным).",
                     "Экранды бояйды (әдепкі — қара)."),
    'draw_pixel': (('x', 'y', 'color'), '', "Sets one pixel (color 0xRRGGBB).",
                   "Рисует один пиксель (цвет 0xRRGGBB).", "Бір пиксель салады (түсі 0xRRGGBB)."),
    'draw_line': (('x0', 'y0', 'x1', 'y1', 'color'), '', "Draws a line, both end points included.",
                  "Рисует линию, включая оба конца.", "Екі шетін қоса сызық сызады."),
    'draw_rect': (('x', 'y', 'w', 'h', 'color'), '', "Draws the outline of a w×h rectangle.",
                  "Рисует контур прямоугольника w×h.", "w×h тіктөртбұрыштың контурын сызады."),
    'fill_rect': (('x', 'y', 'w', 'h', 'color'), '', "Draws a filled rectangle.",
                  "Рисует закрашенный прямоугольник.", "Боялған тіктөртбұрыш салады."),
    'draw_circle': (('cx', 'cy', 'r', 'color'), '', "Draws a circle outline.", "Рисует контур окружности.",
                    "Шеңбердің контурын сызады."),
    'draw_text': (('x', 'y', 'text', 'color'), '', "Draws text in the 5×7 font; (x, y) is the top-left corner.",
                  "Рисует текст шрифтом 5×7; (x, y) — левый верхний угол.",
                  "Мәтінді 5×7 қаріппен салады; (x, y) — сол жақ жоғарғы бұрыш."),
    'get_pixel': (('x', 'y'), 'int', "The color of a pixel (0 outside the screen).",
                  "Цвет пикселя (0 за пределами экрана).", "Пиксельдің түсі (экраннан тыс — 0)."),
    'screen_show': ((), '', "Presents the frame.", "Показывает кадр.", "Кадрды көрсетеді."),
    'key_pressed': ((), 'int', "A waiting key code, or 0 (never blocks).",
                    "Код нажатой клавиши или 0 (не ждёт).", "Басылған перне коды немесе 0 (күтпейді)."),
    'wait_key': ((), 'int', "Waits for a key and returns its code.", "Ждёт нажатия клавиши и возвращает её код.",
                 "Пернені күтеді және оның кодын қайтарады."),
    'screen_width': ((), 'int', "Width of the screen (0 before screen_init).",
                     "Ширина экрана (0 до screen_init).", "Экранның ені (screen_init-ке дейін 0)."),
    'screen_height': ((), 'int', "Height of the screen (0 before screen_init).",
                      "Высота экрана (0 до screen_init).", "Экранның биіктігі (screen_init-ке дейін 0)."),
    'energy_mode': (('?mode',), 'int', "Gets or sets the energy mode: 0 performance, 1 balanced, 2 low.",
                    "Читает или задаёт режим энергии: 0 производительность, 1 баланс, 2 экономия.",
                    "Энергия режимін оқиды немесе орнатады: 0 өнімділік, 1 теңгерім, 2 үнемдеу."),
    'Some': (('value',), 'Option', "An Option that holds a value.", "Option со значением.", "Мәні бар Option."),
    'Ok': (('value',), 'Result', "A successful Result.", "Успешный Result.", "Сәтті Result."),
    'Err': (('error',), 'Result', "A failed Result carrying an error.", "Неуспешный Result с ошибкой.",
            "Қатесі бар сәтсіз Result."),
}

# Builtin methods: name -> (receivers, parameters, return, English, Russian, Kazakh)
# receivers: s = str, a = array, m = map, o = Option / Result
METHOD_DOCS: Dict[str, Tuple[str, Tuple[str, ...], str, str, str, str]] = {
    'len': ('sam', (), 'int', "Number of bytes / elements / entries.", "Число байтов / элементов / записей.",
            "Байттар / элементтер / жазбалар саны."),
    'is_empty': ('sam', (), 'bool', "True when it has no elements.", "True, если пусто.", "Бос болса True."),
    'push': ('a', ('value',), '', "Appends a value.", "Добавляет значение в конец.", "Соңына мән қосады."),
    'pop': ('a', (), 'value', "Removes and returns the last element.", "Удаляет и возвращает последний элемент.",
            "Соңғы элементті алып тастап, қайтарады."),
    'get': ('sam', ('index',), 'value', "The element at an index (or the value of a key).",
            "Элемент по индексу (или значение по ключу).", "Индекс бойынша элемент (немесе кілт бойынша мән)."),
    'set': ('am', ('index', 'value'), '', "Replaces the element at an index (or sets a key).",
            "Заменяет элемент по индексу (или задаёт ключ).", "Индекс бойынша элементті ауыстырады (немесе кілтті орнатады)."),
    'contains': ('sam', ('value',), 'bool', "True when the value (substring / element / key) is present.",
                 "True, если значение (подстрока / элемент / ключ) есть.",
                 "Мән (ішкі жол / элемент / кілт) бар болса True."),
    'starts_with': ('s', ('prefix',), 'bool', "True when the string starts with prefix.",
                    "True, если строка начинается с prefix.", "Жол prefix-пен басталса True."),
    'ends_with': ('s', ('suffix',), 'bool', "True when the string ends with suffix.",
                  "True, если строка заканчивается на suffix.", "Жол suffix-пен аяқталса True."),
    'trim': ('s', (), 'str', "Without leading and trailing blanks.", "Без пробелов в начале и в конце.",
             "Басындағы және соңындағы бос орынсыз."),
    'to_upper': ('s', (), 'str', "In upper case.", "В верхнем регистре.", "Бас әріптермен."),
    'to_lower': ('s', (), 'str', "In lower case.", "В нижнем регистре.", "Кіші әріптермен."),
    'replace': ('s', ('old', 'new'), 'str', "Every old replaced by new.", "Все old заменены на new.",
                "Барлық old new-мен ауыстырылған."),
    'slice': ('sa', ('?start', '?end'), 'str | array', "The part start..end (clamped).",
              "Часть start..end (с ограничением границ).", "start..end бөлігі (шекаралар шектеледі)."),
    'find': ('s', ('sub',), 'int', "Byte index of sub, or -1.", "Байтовый индекс sub или -1.",
             "sub-тың байттық индексі немесе -1."),
    'eq': ('s', ('other',), 'bool', "True when both strings are equal.", "True, если строки равны.",
           "Жолдар тең болса True."),
    'split': ('s', ('?separator',), '[str]', "Parts between separators (no separator: between blanks).",
              "Части между разделителями (без разделителя — между пробелами).",
              "Бөлгіштер арасындағы бөліктер (бөлгішсіз — бос орындар арасында)."),
    'repeat': ('s', ('times',), 'str', "The string repeated n times.", "Строка, повторённая n раз.",
               "n рет қайталанған жол."),
    'to_int': ('s', (), 'int', "The integer the string spells.", "Целое число из строки.", "Жолдағы бүтін сан."),
    'to_float': ('s', (), 'float', "The float the string spells.", "Дробное число из строки.",
                 "Жолдағы бөлшек сан."),
    'insert': ('a', ('index', 'value'), '', "Inserts a value before index.", "Вставляет значение перед index.",
               "Мәнді index алдына кірістіреді."),
    'remove': ('am', ('index',), 'value', "Removes the element at index (or a key).",
               "Удаляет элемент по индексу (или ключ).", "Индекс бойынша элементті (немесе кілтті) жояды."),
    'index_of': ('sa', ('value',), 'int', "Index of the first equal element, or -1.",
                 "Индекс первого равного элемента или -1.", "Бірінші тең элементтің индексі немесе -1."),
    'reverse': ('a', (), '', "Reverses the array in place.", "Переворачивает массив на месте.",
                "Массивті орнында кері аударады."),
    'sort': ('a', (), '', "Sorts the array in place (numbers or strings).",
             "Сортирует массив на месте (числа или строки).", "Массивті орнында сұрыптайды (сандар немесе жолдар)."),
    'join': ('a', ('?separator',), 'str', "The elements as text, joined by separator.",
             "Элементы как текст, соединённые разделителем.", "Бөлгішпен біріктірілген элементтер мәтіні."),
    'first': ('a', (), 'value', "The first element (None when empty).", "Первый элемент (None, если пусто).",
              "Бірінші элемент (бос болса None)."),
    'last': ('a', (), 'value', "The last element (None when empty).", "Последний элемент (None, если пусто).",
             "Соңғы элемент (бос болса None)."),
    'clear': ('am', (), '', "Removes every element.", "Удаляет все элементы.", "Барлық элементтерді жояды."),
    'sum': ('a', (), 'int | float', "Sum of the elements.", "Сумма элементов.", "Элементтердің қосындысы."),
    'min': ('a', (), 'value', "The smallest element.", "Наименьший элемент.", "Ең кіші элемент."),
    'max': ('a', (), 'value', "The largest element.", "Наибольший элемент.", "Ең үлкен элемент."),
    'has': ('m', ('key',), 'bool', "True when the key is present.", "True, если ключ есть.", "Кілт бар болса True."),
    'delete': ('m', ('key',), '', "Removes a key.", "Удаляет ключ.", "Кілтті жояды."),
    'keys': ('m', (), '[key]', "The keys in insertion order.", "Ключи в порядке добавления.",
             "Қосылу ретіндегі кілттер."),
    'values': ('m', (), '[value]', "The values in insertion order.", "Значения в порядке добавления.",
               "Қосылу ретіндегі мәндер."),
    'get_or': ('m', ('key', 'default'), 'value', "The value of a key, or default.",
               "Значение ключа или default.", "Кілттің мәні немесе default."),
    'unwrap': ('o', (), 'value', "The value inside Some / Ok (a runtime error for None / Err).",
               "Значение внутри Some / Ok (ошибка выполнения для None / Err).",
               "Some / Ok ішіндегі мән (None / Err үшін орындау қатесі)."),
    'expect': ('o', ('message',), 'value', "Like unwrap, with your error message.",
               "Как unwrap, но с вашим сообщением об ошибке.", "unwrap сияқты, бірақ сіздің қате хабарыңызбен."),
    'unwrap_or': ('o', ('default',), 'value', "The value inside, or default.", "Значение внутри или default.",
                  "Ішіндегі мән немесе default."),
    'is_some': ('o', (), 'bool', "True for Some(...).", "True для Some(...).", "Some(...) үшін True."),
    'is_none': ('o', (), 'bool', "True for None.", "True для None.", "None үшін True."),
    'is_ok': ('o', (), 'bool', "True for Ok(...).", "True для Ok(...).", "Ok(...) үшін True."),
    'is_err': ('o', (), 'bool', "True for Err(...).", "True для Err(...).", "Err(...) үшін True."),
    'error': ('o', (), 'value', "The error inside Err (None otherwise).", "Ошибка внутри Err (иначе None).",
              "Err ішіндегі қате (әйтпесе None)."),
    'to_string': ('samo', (), 'str', "The text form of the value.", "Текстовое представление значения.",
                  "Мәннің мәтіндік түрі."),
    'copy': ('am', (), 'array | map', "A shallow copy.", "Поверхностная копия.", "Беткі көшірме."),
    'count': ('sa', ('value',), 'int', "How many times the value occurs.", "Сколько раз встречается значение.",
              "Мәннің неше рет кездесетіні."),
    'map': ('a', ('f',), '[value]', "A new array of f(x) for each element.", "Новый массив из f(x) для каждого элемента.",
            "Әр элемент үшін f(x) мәндерінің жаңа массиві."),
    'filter': ('a', ('f',), '[value]', "The elements for which f(x) is true.", "Элементы, для которых f(x) истинно.",
               "f(x) ақиқат болатын элементтер."),
    'reduce': ('a', ('f', 'initial'), 'value', "Folds the array: f(acc, x) from initial.",
               "Свёртка массива: f(acc, x) начиная с initial.", "Массивті жинақтау: initial-дан бастап f(acc, x)."),
    'each': ('a', ('f',), '', "Calls f(x) for each element.", "Вызывает f(x) для каждого элемента.",
             "Әр элемент үшін f(x) шақырады."),
    'any': ('a', ('f',), 'bool', "True when f(x) is true for some element.",
            "True, если f(x) истинно хотя бы для одного элемента.", "f(x) кемінде бір элемент үшін ақиқат болса True."),
    'all': ('a', ('f',), 'bool', "True when f(x) is true for every element.",
            "True, если f(x) истинно для всех элементов.", "f(x) барлық элементтер үшін ақиқат болса True."),
}

# Attributes `#[name: ...]`: name -> (insert text, English, Russian, Kazakh)
ATTRIBUTE_DOCS: Dict[str, Tuple[str, str, str, str]] = {
    'pure': ('pure', "The function has no effects (checked: E-EFFECT).",
             "Функция без эффектов (проверяется: E-EFFECT).", "Эффектісіз функция (тексеріледі: E-EFFECT)."),
    'effects': ('effects: ${1:io}', "The effects the function may perform, e.g. `#[effects: io]` (an upper bound).",
                "Эффекты, которые может выполнять функция, напр. `#[effects: io]` (верхняя граница).",
                "Функция орындай алатын эффектілер, мысалы `#[effects: io]` (жоғарғы шек)."),
    'cap': ('cap: ${1:Capability}', "The function requires a capability; callers must hold it (E-CAP).",
            "Функции нужно право доступа; вызывающие должны его иметь (E-CAP).",
            "Функцияға құқық қажет; шақырушыларда ол болуы тиіс (E-CAP)."),
    'requires': ('requires: ${1:condition}', "Precondition, checked on entry (and proved by `til prove`).",
                 "Предусловие: проверяется при входе (и доказывается `til prove`).",
                 "Алғышарт: кіргенде тексеріледі (және `til prove` дәлелдейді)."),
    'ensures': ('ensures: ${1:result >= 0}', "Postcondition on `result`, checked on return.",
                "Постусловие для `result`, проверяется при возврате.",
                "`result` үшін соңғы шарт, қайтарғанда тексеріледі."),
    'level': ('level: ${1|0,1,2,3,4|}', "The function's level: 0 hardware, 1 systems, 2 safe, 3 script, 4 formal.",
              "Уровень функции: 0 железо, 1 системный, 2 безопасный, 3 скрипт, 4 формальный.",
              "Функция деңгейі: 0 жабдық, 1 жүйелік, 2 қауіпсіз, 3 скрипт, 4 формалды."),
    'energy': ('energy: ${1|low,balanced,performance|}',
               "Battery saving: `low` above a function; `low`/`balanced`/`performance` as the program's mode.",
               "Экономия батареи: `low` над функцией; `low`/`balanced`/`performance` — режим программы.",
               "Батареяны үнемдеу: функция үстінде `low`; `low`/`balanced`/`performance` — бағдарлама режимі."),
    'lang': ('lang: ${1|ru,kk-latn,kk|}', "Keyword language of the file (ru, kk-latn; Kazakh is always on).",
             "Язык ключевых слов файла (ru, kk-latn; казахский включён всегда).",
             "Файлдың кілт сөздер тілі (ru, kk-latn; қазақ тілі әрқашан қосулы)."),
    'generic': ('generic: ${1:T}', "Type parameters, same as `name<T>(...)`.",
                "Параметры типа, как `имя<T>(...)`.", "Тип параметрлері, `аты<T>(...)` сияқты."),
    'inline': ('inline', "Hint for the native backend: inline this function.",
               "Подсказка нативному бэкенду: встроить функцию.", "Нативті бэкендке кеңес: функцияны кірістіру."),
    'noinline': ('noinline', "Hint for the native backend: never inline this function.",
                 "Подсказка нативному бэкенду: не встраивать функцию.",
                 "Нативті бэкендке кеңес: функцияны кірістірмеу."),
    'cross_level': ('cross_level: ${1:2} -> ${2:0}', "A cross-level contract between two levels.",
                    "Контракт между двумя уровнями.", "Екі деңгей арасындағы келісім."),
}

# Words shown in hovers / details (en, ru, kk). Not diagnostics: labels.
LABELS: Dict[str, Tuple[str, str, str]] = {
    'function': ("function", "функция", "функция"),
    'method': ("method", "метод", "әдіс"),
    'struct': ("struct", "структура", "құрылым"),
    'enum': ("enum", "перечисление", "санақ"),
    'variant': ("enum variant", "вариант перечисления", "санақ нұсқасы"),
    'field': ("field", "поле", "өріс"),
    'trait': ("trait", "типаж", "қасиет"),
    'alias': ("type alias", "псевдоним типа", "тип бүркеншік аты"),
    'global': ("global variable", "глобальная переменная", "глобалды айнымалы"),
    'const': ("constant", "константа", "тұрақты"),
    'local': ("local variable", "локальная переменная", "жергілікті айнымалы"),
    'param': ("parameter", "параметр", "параметр"),
    'capability': ("capability", "право доступа", "құқық"),
    'module': ("module", "модуль", "модуль"),
    'typeparam': ("type parameter", "параметр типа", "тип параметрі"),
    'builtin': ("builtin function", "встроенная функция", "кіріктірілген функция"),
    'builtin_method': ("builtin method", "встроенный метод", "кіріктірілген әдіс"),
    'keyword': ("keyword", "ключевое слово", "кілт сөз"),
    'attribute': ("attribute", "атрибут", "атрибут"),
    'snippet': ("snippet", "шаблон", "үлгі"),
    'requires': ("requires", "требует", "талап етеді"),
    'ensures': ("ensures", "гарантирует", "кепілдік береді"),
    'effects': ("effects", "эффекты", "эффектілер"),
    'capabilities': ("capabilities", "права доступа", "құқықтар"),
    'in': ("in", "в", "ішінде"),
    'type': ("type", "тип", "тип"),
    'mutable': ("mutable", "изменяемая", "өзгермелі"),
}
_LANG_COLUMN = {'en': 0, 'ru': 1, 'kk': 2}


def label(key: str, lang: str) -> str:
    row = LABELS.get(key)
    if row is None:
        return key
    return row[_LANG_COLUMN.get(lang, 0)]


def doc_text(row, lang: str, first: int) -> str:
    """The (en, ru, kk) text of a doc-table row whose English text is at row[first]."""
    return row[first + _LANG_COLUMN.get(lang, 0)]


# Snippets: (label, keyword types used, body template, English, Russian, Kazakh).
# `{IF}` etc. are replaced by the keyword in the document's language.
SNIPPETS: List[Tuple[str, str, str, str, str]] = [
    ('fn', "${1:name}(${2:x: int}) -> ${3:int}\n    ${0:{RETURN} x}",
     "Function definition", "Определение функции", "Функция анықтамасы"),
    ('main', "{MAIN}()\n    $0", "Program entry point", "Точка входа программы", "Бағдарламаның кіру нүктесі"),
    ('struct', "{STRUCT} ${1:Name}\n    ${2:field}: ${3:int}$0", "Struct definition", "Определение структуры",
     "Құрылым анықтамасы"),
    ('impl', "{IMPL} ${1:Name}\n    ${2:method}({SELF}) -> ${3:int}\n        $0", "Methods of a struct",
     "Методы структуры", "Құрылым әдістері"),
    ('enum', "{ENUM} ${1:Name}\n    ${2:First}\n    ${3:Second}$0", "Enum definition",
     "Определение перечисления", "Санақ анықтамасы"),
    ('if', "{IF} ${1:condition}\n    $0", "If block", "Блок если", "Егер блогы"),
    ('ifelse', "{IF} ${1:condition}\n    ${2}\n{ELSE}\n    $0", "If / else", "Если / иначе", "Егер / әйтпесе"),
    ('for', "{FOR} ${1:i} {IN} ${2:0..10}\n    $0", "For loop", "Цикл для", "Үшін циклы"),
    ('while', "{WHILE} ${1:condition}\n    $0", "While loop", "Цикл пока", "Кезінде циклы"),
    ('match', "{MATCH} ${1:value}\n    ${2:pattern} => ${3:result}\n    _ => ${0:other}", "Match expression",
     "Выражение выбора", "Сәйкестік өрнегі"),
    ('retry', "{RETRY}(${1:3})\n    ${2:body}\n{ELSE}\n    ${0:fallback}", "Retry with a fallback",
     "Повтор с запасным вариантом", "Қосалқы нұсқасы бар қайталау"),
    ('timeout', "{TIMEOUT}(${1:100})\n    ${2:body}\n{ELSE}\n    ${0:fallback}", "Timeout with a fallback",
     "Таймаут с запасным вариантом", "Қосалқы нұсқасы бар уақыт шегі"),
    ('handle', "{HANDLE}\n    ${1:body}\n{WITH}\n    ${2:Effect}(${3:x}) => {RESUME}(${0:x})",
     "Effect handler", "Обработчик эффектов", "Эффект өңдеушісі"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#                     LEXING (tolerant) AND PARSING (with repair)
# ═══════════════════════════════════════════════════════════════════════════════

LAYOUT = frozenset((TT.NEWLINE, TT.INDENT, TT.DEDENT, TT.EOF))
OPENERS = {TT.LPAREN: TT.RPAREN, TT.LBRACKET: TT.RBRACKET, TT.LBRACE: TT.RBRACE}
CLOSERS = frozenset(OPENERS.values())
KEYWORD_TYPES = frozenset(tt for tt in TT if tt not in (
    TT.INT, TT.FLOAT, TT.STRING, TT.FSTRING, TT.CHAR, TT.IDENT, TT.ATTRIBUTE,
    TT.NEWLINE, TT.INDENT, TT.DEDENT, TT.EOF) and tt.name.isalpha() and tt.name not in (
    'PLUS', 'MINUS', 'STAR', 'SLASH', 'PERCENT', 'POWER', 'EQ', 'EQEQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE',
    'PLUSEQ', 'MINUSEQ', 'STAREQ', 'SLASHEQ', 'PERCENTEQ', 'ARROW', 'RANGE', 'AMP', 'PIPE', 'CARET',
    'TILDE', 'SHL', 'SHR', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'LBRACE', 'RBRACE', 'COMMA',
    'COLON', 'SEMICOLON', 'DOT', 'QUESTION', 'AT', 'HASH'))
MAX_LEX_REPAIRS = 8
MAX_PARSE_REPAIRS = 10


class _Lexer(til.Lexer):
    """The compiler's lexer, also recording comments and `#[lang: ...]` pragmas
    (which produce no tokens) as (start, end, kind) spans."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comments: List[Tuple[int, int, str]] = []

    def skip_comment(self):
        start = self.pos
        super().skip_comment()
        if self.pos > start:
            self.comments.append((start, self.pos, 'comment'))

    def read_attribute(self):
        start, count = self.pos, len(self.tokens)
        super().read_attribute()
        if len(self.tokens) == count:           # #[lang: ...] (a lexer pragma, no token)
            self.comments.append((start, self.pos, 'pragma'))


def _blank_lines(text: str, lines: Iterable[int]) -> str:
    """Replace the content of 0-based lines with spaces (offsets stay valid)."""
    lines = set(lines)
    if not lines:
        return text
    parts = text.split('\n')
    for ln in lines:
        if 0 <= ln < len(parts):
            parts[ln] = ' ' * len(parts[ln])
    return '\n'.join(parts)


class Lexed:
    """Tokens of a text (lines the lexer rejects are blanked and lexed again)."""

    def __init__(self):
        self.tokens: List[Any] = []
        self.comments: List[Tuple[int, int, str]] = []
        self.languages: List[str] = list(til.DEFAULT_LANGUAGES)
        self.error: Optional[RawDiag] = None          # the first lexer error
        self.error_span: Optional[Tuple[int, int]] = None
        self.blanked: Set[int] = set()
        self.text = ""                                # the text that was lexed


def lex(text: str, filename: str = "<lsp>") -> Lexed:
    out = Lexed()
    cur = text
    for _attempt in range(MAX_LEX_REPAIRS + 1):
        lexer = _Lexer(cur, filename)
        with _Capture() as raws:
            try:
                tokens = lexer.tokenize()
            except SyntaxError:
                tokens = None
            except RecursionError:
                tokens = None
        out.languages = list(lexer.languages)
        if tokens is not None:
            out.tokens, out.comments, out.text = tokens, lexer.comments, cur
            return out
        line0 = max(lexer.line - 1, 0)
        if out.error is None:
            raw = raws[-1] if raws else RawDiag(filename, cur, lexer.line, lexer.col, "Unexpected character")
            out.error = raw
            pm_line_start = _line_start(cur, line0)
            start = pm_line_start + max(lexer.tok_col - 1, 0) if lexer.tok_line == lexer.line else \
                pm_line_start
            end = pm_line_start + max(lexer.col - 1, 0)
            if end <= start:
                end = start + 1
            out.error_span = (start, end)
        if line0 in out.blanked:
            break
        out.blanked.add(line0)
        cur = _blank_lines(cur, [line0])
    out.tokens, out.comments, out.text = [], [], cur
    return out


def _line_start(text: str, line0: int) -> int:
    pos = 0
    for _ in range(line0):
        k = text.find('\n', pos)
        if k < 0:
            return len(text)
        pos = k + 1
    return pos


class Parsed:
    """A program parsed from a text, possibly after blanking broken lines."""

    def __init__(self):
        self.program = None
        self.statements: List[Any] = []
        self.error: Optional[RawDiag] = None          # the first parser error (as the compiler reports it)
        self.error_tok = None
        self.blanked: Set[int] = set()
        self.lexed: Optional[Lexed] = None            # tokens of the text the program came from


def _parse_tokens(tokens, text, filename, languages):
    parser = til.Parser(list(tokens), filename, source=text, languages=languages)
    with _Capture() as raws:
        try:
            program = parser.parse()
            return program, None, None
        except SyntaxError:
            pass
        except (RecursionError, IndexError, AttributeError, TypeError, ValueError):
            pass
    tok = parser.current() if parser.tokens else None
    raw = raws[-1] if raws else None
    return None, raw, tok


def _indent_width(line: str) -> int:
    w = 0
    for ch in line:
        if ch == ' ':
            w += 1
        elif ch == '\t':
            w += 4
        else:
            break
    return w


def _is_blank_or_comment(line: str) -> bool:
    s = line.strip()
    return not s or (s.startswith('#') and not s.startswith('#['))


def parse(text: str, lexed: Lexed, filename: str = "<lsp>") -> Parsed:
    """Parse; on a syntax error blank the broken line (or block, or the open
    bracket's line) and try again, so the rest of the file is still analyzed."""
    out = Parsed()
    out.blanked = set(lexed.blanked)
    cur_lexed = lexed
    cur_text = lexed.text
    lines = cur_text.split('\n')
    for attempt in range(MAX_PARSE_REPAIRS + 1):
        if not cur_lexed.tokens:
            break
        program, raw, tok = _parse_tokens(cur_lexed.tokens, cur_text, filename, cur_lexed.languages)
        if program is not None:
            out.program = program
            out.statements = list(program.statements)
            out.lexed = cur_lexed
            return out
        if out.error is None and attempt == 0 and not lexed.blanked:
            out.error = raw
            out.error_tok = tok
        bad = _lines_to_blank(cur_lexed, tok, raw, lines)
        bad -= out.blanked
        if not bad:
            # nothing new to remove: drop the rest of the top-level item
            ln = (tok.line - 1) if tok is not None else 0
            bad = set(_toplevel_item_lines(lines, ln)) - out.blanked
            if not bad:
                break
        out.blanked |= bad
        cur_text = _blank_lines(cur_text, bad)
        lines = cur_text.split('\n')
        cur_lexed = lex(cur_text, filename)
        out.blanked |= cur_lexed.blanked
    out.lexed = cur_lexed
    return out


def _lines_to_blank(lexed: Lexed, tok, raw: Optional[RawDiag], lines: List[str]) -> Set[int]:
    if tok is None:
        return set()
    tokens = lexed.tokens
    idx = _token_index(tokens, tok)
    text = (raw.text if raw is not None else "") or ""
    # an unclosed bracket swallows the following lines: blank the line that opened it
    depth_stack: List[Any] = []
    for t in tokens[:idx]:
        if t.type in OPENERS:
            depth_stack.append(t)
        elif t.type in CLOSERS and depth_stack:
            depth_stack.pop()
    if depth_stack:
        return {depth_stack[-1].line - 1}
    line = tok.line - 1
    if tok.type == TT.INDENT or text.startswith("Unexpected token: INDENT"):
        return _block_lines(lines, line)
    if text.startswith("Expected INDENT") or text.startswith("Expected function body"):
        prev = _prev_code_line(lines, line)
        return {prev} if prev is not None else {line}
    if tok.type in (TT.DEDENT, TT.EOF):
        # the statement before the end of a block / file is incomplete
        prev = _prev_code_line(lines, len(lines) if tok.type == TT.EOF else line)
        return {prev} if prev is not None else set()
    if 0 <= line < len(lines) and _is_blank_or_comment(lines[line]):
        prev = _prev_code_line(lines, line)
        return {prev} if prev is not None else set()
    result = {line}
    # a header line (the next code line is indented deeper): its block goes too
    nxt = _next_code_line(lines, line)
    if nxt is not None and 0 <= line < len(lines) and \
            _indent_width(lines[nxt]) > _indent_width(lines[line]):
        result |= _block_lines(lines, nxt)
    return result


def _token_index(tokens, tok) -> int:
    for i, t in enumerate(tokens):
        if t is tok:
            return i
    starts = [t.start for t in tokens]
    return max(0, min(bisect.bisect_left(starts, tok.start), len(tokens)))


def _prev_code_line(lines: List[str], line: int) -> Optional[int]:
    k = min(line, len(lines)) - 1
    while k >= 0:
        if not _is_blank_or_comment(lines[k]):
            return k
        k -= 1
    return None


def _next_code_line(lines: List[str], line: int) -> Optional[int]:
    k = line + 1
    while k < len(lines):
        if not _is_blank_or_comment(lines[k]):
            return k
        k += 1
    return None


def _block_lines(lines: List[str], line: int) -> Set[int]:
    """`line` and the following lines indented at least as deep (an indented block)."""
    if not 0 <= line < len(lines):
        return set()
    w = _indent_width(lines[line])
    out = {line}
    k = line + 1
    while k < len(lines):
        if not _is_blank_or_comment(lines[k]):
            if _indent_width(lines[k]) < w:
                break
            out.add(k)
        k += 1
    return out


def _toplevel_item_lines(lines: List[str], line: int) -> List[int]:
    """The lines of the top-level item around `line` (up to the next unindented code line)."""
    start = line
    while start > 0 and (_is_blank_or_comment(lines[start]) or _indent_width(lines[start]) > 0):
        start -= 1
    end = start + 1
    while end < len(lines) and (_is_blank_or_comment(lines[end]) or _indent_width(lines[end]) > 0):
        end += 1
    return [k for k in range(start, end) if k < len(lines) and not _is_blank_or_comment(lines[k])]


# ═══════════════════════════════════════════════════════════════════════════════
#                                 SYMBOL INDEX
# ═══════════════════════════════════════════════════════════════════════════════

class Sym:
    """A definition: function, method, struct, field, enum, variant, trait,
    type alias, global, constant, local, parameter, capability, module."""

    def __init__(self, name: str, kind: str, start: int, end: int):
        self.name = name
        self.kind = kind
        self.start = start
        self.end = end
        self.full_start = start
        self.full_end = end
        self.container: Optional['Sym'] = None
        self.container_name: Optional[str] = None
        self.node = None
        self.scope: Optional['Scope'] = None
        self.owner: Optional['Sym'] = None            # the function of a local / parameter
        self.typ = None                               # syntactic type (see _desc)
        self.mutable = False
        self.uri = ""
        self.doc = ""
        self.key: Optional[str] = None                # tiltypes function key: 'f', 'Point.norm'
        self.params: List['Sym'] = []
        self.static = False
        self.implicit = False                         # self / error / result: not renamable
        self.children: List['Sym'] = []
        self.decl_word: Optional[Tuple[int, int, str]] = None   # (start, end, text) of let/var
        self.header_end = end                         # end of the function header (code lenses)
        self.default_text: Optional[str] = None       # a parameter's default value, as written
        self.index: Optional['Index'] = None

    @property
    def qualname(self) -> str:
        cont = self.container.name if self.container is not None else self.container_name
        return f"{cont}.{self.name}" if cont and self.kind in ('method', 'field', 'variant') else self.name

    def __repr__(self):
        return f"Sym({self.kind} {self.qualname} @{self.start})"


class Ref:
    """An occurrence of a name. `sym` is set when it resolves inside the file;
    otherwise `kind` and `recv` say how to resolve it across files."""
    __slots__ = ('start', 'end', 'name', 'sym', 'kind', 'write', 'scope', 'recv', 'decl', 'guess', 'builtin')

    def __init__(self, start, end, name, sym=None, kind='name', write=False, scope=None, recv=None,
                 decl=False):
        self.start = start
        self.end = end
        self.name = name
        self.sym = sym
        self.kind = kind          # name | member | type | module | module-member | effect
        self.write = write
        self.scope = scope
        self.recv = recv          # member: type name of the receiver; module-member: module name
        self.decl = decl
        self.guess = False        # member resolved by the "only one struct has it" rule
        self.builtin = None       # 'function' / 'method' / 'type' / 'const' for builtins

    def __repr__(self):
        return f"Ref({self.name}@{self.start} -> {self.sym!r} {self.kind})"


class Scope:
    def __init__(self, kind: str, parent: Optional['Scope'], start: int, end: int, indent: int = 0,
                 owner: Optional[Sym] = None):
        self.kind = kind
        self.parent = parent
        self.start = start
        self.end = end
        self.indent = indent
        self.owner = owner
        self.decls: List[Tuple[int, Sym]] = []
        self.children: List['Scope'] = []
        if parent is not None:
            parent.children.append(self)

    def add(self, sym: Sym, at: int):
        self.decls.append((at, sym))
        sym.scope = self

    def find(self, name: str, at: Optional[int]) -> Optional[Sym]:
        for off, s in reversed(self.decls):
            if s.name == name and (at is None or self.kind == 'module' or off <= at):
                return s
        return None

    def lookup(self, name: str, at: Optional[int]) -> Optional[Sym]:
        sc = self
        while sc is not None:
            s = sc.find(name, at)
            if s is not None:
                return s
            sc = sc.parent
        return None

    def visible(self, at: int) -> List[Sym]:
        """Every declaration visible at `at` (innermost first, shadowed ones dropped)."""
        out, seen = [], set()
        sc = self
        while sc is not None:
            for off, s in reversed(sc.decls):
                if s.name in seen:
                    continue
                if sc.kind == 'module' or off <= at:
                    seen.add(s.name)
                    out.append(s)
            sc = sc.parent
        return out


class ImportInfo:
    def __init__(self, module: str, alias: Optional[str], items: Optional[List[str]], start: int, end: int,
                 line: int):
        self.module = module
        self.alias = alias
        self.items = items
        self.start = start            # the module path in the source
        self.end = end
        self.line = line              # 0-based line of the import statement
        self.item_spans: List[Tuple[str, int, int]] = []
        self.path: Optional[str] = None


class Index:
    def __init__(self, uri: str):
        self.uri = uri
        self.syms: List[Sym] = []
        self.refs: List[Ref] = []
        self.starts: List[int] = []
        self.toplevel: Dict[str, Sym] = {}
        self.members: Dict[Tuple[str, str], Sym] = {}
        self.module_scope = Scope('module', None, 0, 1 << 60)
        self.script_scope: Optional[Scope] = None
        self.scopes: List[Scope] = [self.module_scope]
        self.imports: List[ImportInfo] = []
        self.outline: List[Sym] = []                  # top-level items, in source order
        self.impls: List[Tuple[str, Optional[str], int, int, int, int, List[Sym]]] = []
        self.functions: List[Sym] = []                # functions and methods (code lenses, types)
        self.effects: Set[str] = set()
        self.caps: Set[str] = set()
        self.has_script = False
        self.script_line = 0                          # first line of top-level script code (0-based)
        self.main: Optional[Sym] = None
        self.node_refs: Dict[int, Ref] = {}

    def finish(self):
        self.refs.sort(key=lambda r: (r.start, r.end))
        self.starts = [r.start for r in self.refs]
        for s in self.syms:
            s.index = self

    def ref_at(self, offset: int) -> Optional[Ref]:
        """The reference whose range contains `offset` (or ends right at it)."""
        i = bisect.bisect_right(self.starts, offset) - 1
        best = None
        for k in (i, i - 1):
            if 0 <= k < len(self.refs):
                r = self.refs[k]
                if r.start <= offset < r.end:
                    return r
                if r.end == offset and best is None:
                    best = r
        return best

    def scope_at(self, offset: int, indent: Optional[int] = None) -> Scope:
        sc = self.module_scope
        while True:
            nxt = None
            for ch in sc.children:
                if ch.start <= offset <= ch.end and (indent is None or ch.kind not in ('block', 'function', 'script')
                                                     or ch.indent <= indent or ch.kind == 'script'):
                    nxt = ch
            if nxt is None:
                return sc
            sc = nxt


def _desc(t):
    """A til type annotation -> a small syntactic type: ('int',), ('str',),
    ('array', elem), ('map', k, v), ('option', x), ('result', ok, err),
    ('struct', name) (a struct, enum or alias name), ('fn', ret) or None."""
    if t is None or isinstance(t, til.UnknownType):
        return None
    if isinstance(t, til.VoidType):
        return ('void',)
    if isinstance(t, til.PrimitiveType):
        n = t.name
        if n in ('float', 'f32', 'f64'):
            return ('float',)
        if n in ('str', 'bool', 'char', 'void'):
            return ('int',) if n == 'char' else (n,)
        return ('int',)
    if isinstance(t, til.ArrayType):
        return ('array', _desc(t.element_type))
    if isinstance(t, til.OptionType):
        return ('option', _desc(t.inner))
    if isinstance(t, til.ResultType):
        return ('result', _desc(t.ok), _desc(t.err))
    if isinstance(t, til.GenericType):
        if t.name in ('Vec', 'Array', 'List') and t.params:
            return ('array', _desc(t.params[0]))
        if t.name in ('HashMap', 'Map', 'Dict') and len(t.params) == 2:
            return ('map', _desc(t.params[0]), _desc(t.params[1]))
        return ('struct', t.name)
    if isinstance(t, til.FunctionType):
        return ('fn', _desc(t.ret))
    if isinstance(t, til.PointerType):
        return _desc(t.pointee)
    if isinstance(t, til.StructType):
        return ('struct', t.name)
    return None


def show_desc(d) -> str:
    if d is None:
        return '?'
    k = d[0]
    if k in ('int', 'float', 'str', 'bool', 'void'):
        return k
    if k == 'array':
        return f"[{show_desc(d[1])}]"
    if k == 'map':
        return f"{{{show_desc(d[1])}: {show_desc(d[2])}}}"
    if k == 'option':
        return f"Option<{show_desc(d[1])}>"
    if k == 'result':
        return f"Result<{show_desc(d[1])}, {show_desc(d[2])}>"
    if k == 'fn':
        return 'fn'
    if k in ('struct', 'enum'):
        return d[1]
    return '?'


def _type_text(t) -> str:
    """A til annotation as written (for signatures)."""
    if t is None or isinstance(t, til.UnknownType):
        return ''
    return str(t)


# Result kinds of builtin methods by receiver (for member completion on chains)
_STR_METHOD_RESULT = {'trim': ('str',), 'to_upper': ('str',), 'to_lower': ('str',), 'slice': ('str',),
                      'repeat': ('str',), 'replace': ('str',), 'to_string': ('str',), 'get': ('str',),
                      'split': ('array', ('str',)), 'len': ('int',), 'find': ('int',), 'index_of': ('int',),
                      'count': ('int',), 'to_int': ('int',), 'to_float': ('float',), 'contains': ('bool',),
                      'starts_with': ('bool',), 'ends_with': ('bool',), 'is_empty': ('bool',), 'eq': ('bool',)}
_BUILTIN_RESULT = {'str': ('str',), 'input': ('str',), 'format': ('str',), 'type_of': ('str',), 'chr': ('str',),
                   'read_file': ('str',), 'bytes_to_str': ('str',), 'len': ('int',), 'int': ('int',),
                   'ord': ('int',), 'clock_ms': ('int',), 'float': ('float',), 'sqrt': ('float',),
                   'pow': ('float',), 'sin': ('float',), 'cos': ('float',), 'tan': ('float',), 'log': ('float',),
                   'exp': ('float',), 'floor': ('float',), 'ceil': ('float',), 'round': ('float',),
                   'bool': ('bool',), 'file_exists': ('bool',), 'range': ('array', ('int',)),
                   'args': ('array', ('str',)), 'str_bytes': ('array', ('int',)), 'get_pixel': ('int',),
                   'key_pressed': ('int',), 'wait_key': ('int',), 'screen_width': ('int',),
                   'screen_height': ('int',), 'energy_mode': ('int',)}
BUILTIN_NAMES = frozenset(til.NATIVES) | {'Some', 'Ok', 'Err'}
BUILTIN_TYPE_NAMES = ('int', 'float', 'str', 'bool', 'char', 'void', 'i8', 'i16', 'i32', 'i64', 'u8', 'u16',
                      'u32', 'u64', 'uint', 'f32', 'f64', 'Option', 'Result', 'Vec', 'HashMap')


def _method_result(recv, name: str):
    if recv is None:
        return None
    k = recv[0]
    if k == 'str':
        return _STR_METHOD_RESULT.get(name)
    if k == 'array':
        elem = recv[1] if len(recv) > 1 else None
        if name in ('map',):
            return ('array', None)
        if name in ('filter', 'copy', 'slice'):
            return recv
        if name in ('first', 'last', 'pop', 'get', 'min', 'max'):
            return elem
        if name in ('join', 'to_string'):
            return ('str',)
        if name in ('len', 'index_of', 'count'):
            return ('int',)
        if name in ('contains', 'is_empty', 'any', 'all'):
            return ('bool',)
        return None
    if k == 'map':
        if name == 'keys':
            return ('array', recv[1] if len(recv) > 1 else None)
        if name == 'values':
            return ('array', recv[2] if len(recv) > 2 else None)
        if name in ('get', 'get_or'):
            return recv[2] if len(recv) > 2 else None
        if name in ('has', 'contains', 'is_empty'):
            return ('bool',)
        if name == 'len':
            return ('int',)
        if name == 'copy':
            return recv
        return None
    if k in ('option', 'result'):
        if name in ('unwrap', 'expect', 'unwrap_or'):
            return recv[1] if len(recv) > 1 else None
        if name in ('is_some', 'is_none', 'is_ok', 'is_err'):
            return ('bool',)
        if name == 'error':
            return recv[2] if k == 'result' and len(recv) > 2 else None
    return None


class _View:
    """Tokens of the file (or of a fragment: an f-string expression, a contract
    in an attribute) with their offsets in the document."""

    def __init__(self, tokens, base: int = 0):
        self.t = tokens
        self.base = base
        self.pos: Dict[Tuple[int, int], int] = {}
        self.match: Dict[int, int] = {}
        self.indent_match: Dict[int, int] = {}
        pos = self.pos
        match = self.match
        brackets: List[int] = []
        indents: List[int] = []
        nl, ind, ded, eof = TT.NEWLINE, TT.INDENT, TT.DEDENT, TT.EOF
        lp, lb, lc, rp, rb, rc = TT.LPAREN, TT.LBRACKET, TT.LBRACE, TT.RPAREN, TT.RBRACKET, TT.RBRACE
        pair = {rp: lp, rb: lb, rc: lc}
        for i, tok in enumerate(tokens):
            tt = tok.type
            if tt is nl or tt is eof:
                continue
            if tt is ind:
                indents.append(i)
                continue
            if tt is ded:
                if indents:
                    self.indent_match[indents.pop()] = i
                continue
            key = (tok.line, tok.col)
            if key not in pos:
                pos[key] = i
            if tt is lp or tt is lb or tt is lc:
                brackets.append(i)
            elif tt is rp or tt is rb or tt is rc:
                want = pair[tt]
                while brackets:
                    j = brackets.pop()
                    if tokens[j].type is want:
                        match[j] = i
                        match[i] = j
                        break
        n = len(tokens)
        while indents:                                 # unbalanced: close at EOF
            self.indent_match[indents.pop()] = n - 1

    def at(self, line: int, col: int) -> Optional[int]:
        return self.pos.get((line, col))

    def span(self, i: int) -> Tuple[int, int]:
        tok = self.t[i]
        return self.base + tok.start, self.base + tok.end

    def typ(self, i: int):
        return self.t[i].type if 0 <= i < len(self.t) else TT.EOF

    def find(self, i: int, types, limit: Optional[int] = None) -> Optional[int]:
        """First index >= i (at bracket depth 0 relative to i) with a type in `types`."""
        n = len(self.t) if limit is None else min(limit, len(self.t))
        depth = 0
        while 0 <= i < n:
            tt = self.t[i].type
            if depth == 0 and tt in types:
                return i
            if tt in OPENERS:
                depth += 1
            elif tt in CLOSERS:
                depth -= 1
                if depth < 0:
                    return None
            i += 1
        return None

    def line_end(self, i: int) -> int:
        """Index of the NEWLINE (or EOF) ending the logical line of token i."""
        if 0 <= i < len(self.t) and self.t[i].type in CLOSERS:
            i += 1                       # the scan counts brackets from token i on
        j = self.find(i, (TT.NEWLINE, TT.EOF, TT.INDENT, TT.DEDENT))
        return j if j is not None else len(self.t) - 1

    def block_after(self, i: int) -> Optional[Tuple[int, int]]:
        """(INDENT, DEDENT) of the indented block that follows the line of token i."""
        j = self.line_end(i)
        k = j
        while k < len(self.t) and self.t[k].type == TT.NEWLINE:
            k += 1
        if k < len(self.t) and self.t[k].type == TT.INDENT:
            return k, self.indent_match.get(k, len(self.t) - 1)
        return None

    def last_real_before(self, i: int) -> int:
        k = min(i, len(self.t)) - 1
        while k > 0 and self.t[k].type in LAYOUT:
            k -= 1
        return max(k, 0)


def _fstring_expr_offsets(raw: str) -> List[int]:
    """Offsets (in the token's source text f"...") where each {expression} starts."""
    if len(raw) < 2:
        return []
    quote = raw[1]
    i, n = 2, len(raw)
    out = []
    while i < n:
        ch = raw[i]
        if ch == quote:
            break
        if ch == '\\':
            nx = raw[i + 1:i + 2]
            if nx == 'u' and raw[i + 2:i + 3] == '{':
                k = raw.find('}', i + 3)
                i = (k + 1) if k >= 0 else n
            elif nx == 'x':
                i += 4
            else:
                i += 2
            continue
        if ch in '{}' and raw[i + 1:i + 2] == ch:
            i += 2
            continue
        if ch == '{':
            out.append(i + 1)
            depth, i = 1, i + 1
            while i < n and depth > 0:
                c = raw[i]
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                i += 1
            continue
        i += 1
    return out


class Indexer:
    """Builds the Index of one file from its tokens and (own) AST statements."""

    def __init__(self, text: str, tokens, statements, uri: str, languages, comments):
        self.text = text
        self.uri = uri
        self.languages = languages
        self.main_view = _View(tokens)
        self.v = self.main_view
        self.ix = Index(uri)
        self.statements = statements
        self.node_sym: Dict[int, Any] = {}
        self.comment_lines: Dict[int, str] = {}
        lines = text.split('\n')
        self.lines = lines
        self.line_starts = [0]
        for ln in lines[:-1]:
            self.line_starts.append(self.line_starts[-1] + len(ln) + 1)
        for start, end, kind in comments:
            if kind != 'comment':
                continue
            ln = bisect.bisect_right(self.line_starts, start) - 1
            if 0 <= ln < len(lines) and lines[ln][:start - self.line_starts[ln]].strip() == "":
                self.comment_lines[ln] = text[start:end]
        self.fn_stack: List[Sym] = []
        self.skip: Set[int] = set()         # statements walked elsewhere (script code in main)

    # ── helpers ─────────────────────────────────────────────────────────────
    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self.line_starts, offset) - 1

    def add_sym(self, sym: Sym) -> Sym:
        sym.uri = self.uri
        self.ix.syms.append(sym)
        return sym

    def add_ref(self, start: int, end: int, name: str, sym=None, **kw) -> Ref:
        r = Ref(start, end, name, sym, **kw)
        self.ix.refs.append(r)
        return r

    def tok_span(self, i: Optional[int]) -> Optional[Tuple[int, int]]:
        if i is None or not 0 <= i < len(self.v.t):
            return None
        return self.v.span(i)

    def node_tok(self, node) -> Optional[int]:
        line, col = getattr(node, 'line', 0), getattr(node, 'col', 0)
        if not line:
            return None
        return self.v.at(line, col)

    def ident_after(self, i: Optional[int], types) -> Optional[int]:
        """Index of the IDENT right after the first token of `types` at/after i."""
        if i is None:
            return None
        j = self.v.find(i, types)
        if j is None or j + 1 >= len(self.v.t):
            return None
        k = j + 1
        if self.v.t[k].type in (TT.IDENT, TT.SELF):
            return k
        return None

    def doc_above(self, first_line: int) -> str:
        out = []
        k = first_line - 1
        while k >= 0 and k in self.comment_lines:
            out.append(self.comment_lines[k])
            k -= 1
        out.reverse()
        cleaned = []
        for c in out:
            c = c.lstrip()
            if c.startswith('#'):
                c = c[1:]
            if c.startswith(' '):
                c = c[1:]
            cleaned.append(c.rstrip())
        return '\n'.join(cleaned).strip('\n')

    def first_line_with_attrs(self, i: int) -> int:
        """0-based first line of a definition whose name/keyword is token i,
        including the attribute lines above it."""
        k = i - 1
        first = self.v.t[i].line - 1
        while k >= 0:
            tt = self.v.t[k].type
            if tt == TT.ATTRIBUTE:
                first = self.v.t[k].line - 1
            elif tt not in (TT.NEWLINE, TT.PUB, TT.FN, TT.INDENT, TT.DEDENT):
                break
            k -= 1
        return first

    def attrs_before(self, i: int) -> List[int]:
        """ATTRIBUTE token indexes that belong to the definition starting at token i."""
        out = []
        k = i - 1
        while k >= 0:
            tt = self.v.t[k].type
            if tt == TT.ATTRIBUTE:
                out.append(k)
            elif tt not in (TT.NEWLINE, TT.PUB, TT.FN, TT.INDENT, TT.DEDENT):
                break
            k -= 1
        out.reverse()
        return out

    def def_end(self, i: int) -> int:
        """End offset of a definition whose header starts at token i."""
        blk = self.v.block_after(i)
        if blk is not None:
            return self.v.span(self.v.last_real_before(blk[1]))[1]
        j = self.v.line_end(i)
        return self.v.span(self.v.last_real_before(j))[1] if j > i else self.v.span(i)[1]

    def block_scope(self, body, parent: Scope, kind: str = 'block', owner=None) -> Scope:
        """A scope for an indented block (extent from the tokens)."""
        start, end, indent = parent.start, parent.end, parent.indent + 1
        stmts = body.statements if isinstance(body, til.Block) else ([body] if body is not None else [])
        first = None
        for st in stmts:
            first = self.node_tok(st)
            if first is not None:
                break
        if first is not None and self.v is self.main_view:
            k = first - 1
            while k >= 0 and self.v.t[k].type == TT.NEWLINE:
                k -= 1
            if k >= 0 and self.v.t[k].type == TT.INDENT:
                ln = self.v.t[first].line - 1
                start = self.line_starts[ln] if 0 <= ln < len(self.line_starts) else self.v.span(first)[0]
                ded = self.v.indent_match.get(k, len(self.v.t) - 1)
                end = self.v.span(ded)[0]
                indent = self.v.t[first].col - 1
            elif k >= 0 and self.v.t[k].type == TT.LBRACE:
                start = self.v.span(k)[0]
                end = self.v.span(self.v.match.get(k, len(self.v.t) - 1))[1]
                indent = parent.indent
            else:
                start = self.v.span(first)[0]
                end = self.v.span(self.v.line_end(first))[0]
                indent = parent.indent
        sc = Scope(kind, parent, start, end, indent, owner)
        self.ix.scopes.append(sc)
        return sc

    # ── types ───────────────────────────────────────────────────────────────
    def resolve_desc(self, d, scope: Optional[Scope] = None, depth: int = 0):
        """('struct', name) of an alias -> its target; of an enum -> ('enum', name)."""
        if d is None or d[0] != 'struct' or depth > 8:
            return d
        s = (scope or self.ix.module_scope).lookup(d[1], None)
        if s is None:
            return d
        if s.kind == 'alias' and s.node is not None:
            return self.resolve_desc(_desc(s.node.type), scope, depth + 1)
        if s.kind == 'enum':
            return ('enum', d[1])
        if s.kind == 'typeparam':
            return None
        return d

    def type_of(self, e, scope: Scope, depth: int = 0):
        if e is None or depth > 10:
            return None
        t = type(e)
        if t in (til.StringLit, til.FStringLit):
            return ('str',)
        if t in (til.IntLit, til.CharLit):
            return ('int',)
        if t is til.FloatLit:
            return ('float',)
        if t is til.BoolLit:
            return ('bool',)
        if t is til.ArrayLit:
            return ('array', self.type_of(e.elements[0], scope, depth + 1) if e.elements else None)
        if t is til.ListComprehension:
            return ('array', None)
        if t is til.Range:
            return ('array', ('int',))
        if t is til.DictLit:
            if e.pairs:
                return ('map', self.type_of(e.pairs[0][0], scope, depth + 1),
                        self.type_of(e.pairs[0][1], scope, depth + 1))
            return ('map', None, None)
        if t is til.StructInit:
            return self.resolve_desc(('struct', e.name), scope)
        if t is til.Identifier:
            if e.name.startswith('__Vec__'):
                return ('array', None)
            if e.name.startswith('__HashMap__'):
                return ('map', None, None)
            s = self.node_sym.get(id(e))
            if isinstance(s, Sym):
                if s.kind in ('struct',):
                    return None
                return self.resolve_desc(s.typ, scope)
            return None
        if t is til.Call:
            f = e.func
            if isinstance(f, til.Identifier):
                s = self.node_sym.get(id(f))
                if isinstance(s, Sym):
                    if s.kind == 'struct':
                        return ('struct', s.name)
                    if s.kind in ('function', 'method') and s.node is not None:
                        return self.resolve_desc(_desc(s.node.ret_type), scope)
                    if s.kind in ('local', 'param', 'global', 'const') and s.typ and s.typ[0] == 'fn':
                        return s.typ[1] if len(s.typ) > 1 else None
                    return None
                if f.name in _BUILTIN_RESULT:
                    return _BUILTIN_RESULT[f.name]
                if f.name == 'Some':
                    return ('option', self.type_of(e.args[0], scope, depth + 1) if e.args else None)
                if f.name in ('Ok', 'Err'):
                    return ('result', None, None)
                if f.name in ('min', 'max', 'abs', 'sum') and e.args:
                    return self.type_of(e.args[0], scope, depth + 1)
                return None
            if isinstance(f, til.Attribute):
                target = self.node_sym.get(id(f))
                if isinstance(target, Sym):
                    if target.kind == 'method' and target.node is not None:
                        rt = _desc(target.node.ret_type)
                        return self.resolve_desc(rt, scope)
                    if target.kind == 'variant' and target.container is not None:
                        return ('enum', target.container.name)
                    return None
                recv = self.type_of(f.obj, scope, depth + 1)
                return _method_result(recv, f.attr)
            return None
        if t is til.Attribute:
            target = self.node_sym.get(id(e))
            if isinstance(target, Sym):
                if target.kind == 'field' and target.node is not None:
                    return self.resolve_desc(_desc(target.node.type), scope)
                if target.kind == 'variant' and target.container is not None:
                    return ('enum', target.container.name)
            return None
        if t is til.Index:
            obj = self.type_of(e.obj, scope, depth + 1)
            if obj is None:
                return None
            if obj[0] == 'array':
                return obj[1] if len(obj) > 1 else None
            if obj[0] == 'map':
                return obj[2] if len(obj) > 2 else None
            if obj[0] == 'str':
                return ('str',)
            return None
        if t is til.Cast:
            return self.resolve_desc(_desc(e.target_type), scope)
        if t is til.NullCheck:
            inner = self.type_of(e.expr, scope, depth + 1)
            if inner and inner[0] in ('option', 'result'):
                return inner[1] if len(inner) > 1 else None
            return None
        if t is til.IfExpr:
            return self.type_of(e.then_expr, scope, depth + 1) or self.type_of(e.else_expr, scope, depth + 1)
        if t is til.BinaryOp:
            if e.op in ('==', '!=', '<', '>', '<=', '>=', 'and', 'or'):
                return ('bool',)
            a = self.type_of(e.left, scope, depth + 1)
            b = self.type_of(e.right, scope, depth + 1)
            if e.op == '+' and (a == ('str',) or b == ('str',)):
                return ('str',)
            if a == ('float',) or b == ('float',):
                return ('float',)
            if a == ('int',) and b == ('int',):
                return ('int',)
            return None
        if t is til.UnaryOp:
            if e.op == 'not':
                return ('bool',)
            return self.type_of(e.operand, scope, depth + 1)
        if t is til.Lambda:
            return ('fn', _desc(e.ret_type))
        return None

    # ── names ───────────────────────────────────────────────────────────────
    def name_ref(self, node, scope: Scope, write=False) -> Optional[Ref]:
        """Resolve an Identifier node."""
        name = node.name
        i = self.node_tok(node)
        sp = self.tok_span(i)
        if sp is None:
            return None
        if name == 'NULL':
            r = self.add_ref(sp[0], sp[1], 'None', kind='name')
            r.builtin = 'const'
            return r
        if name.startswith('__Vec__') or name.startswith('__HashMap__'):
            r = self.add_ref(sp[0], sp[1], self.text[sp[0]:sp[1]], kind='type')
            r.builtin = 'type'
            return r
        sym = scope.lookup(name, sp[0])
        if sym is not None:
            self.node_sym[id(node)] = sym
            r = self.add_ref(sp[0], sp[1], name, sym, kind='name', write=write, scope=scope)
        else:
            r = self.add_ref(sp[0], sp[1], name, None, kind='name', write=write, scope=scope)
            if name in BUILTIN_NAMES:
                r.builtin = 'function'
            elif name == 'self':
                r.builtin = 'const'
        self.ix.node_refs[id(node)] = r
        return r

    def declare(self, scope: Scope, name: str, kind: str, i: Optional[int], node=None, typ=None,
                mutable=False, at: Optional[int] = None) -> Optional[Sym]:
        sp = self.tok_span(i)
        if sp is None:
            return None
        sym = self.add_sym(Sym(name, kind, sp[0], sp[1]))
        sym.node = node
        sym.typ = typ
        sym.mutable = mutable
        sym.owner = self.fn_stack[-1] if self.fn_stack else None
        scope.add(sym, sp[0] if at is None else at)
        self.add_ref(sp[0], sp[1], name, sym, kind='name', scope=scope, decl=True)
        return sym

    def type_refs(self, i: Optional[int], stop_types, scope: Scope, limit: Optional[int] = None):
        """IDENT tokens of a type written from token i up to a stop token (depth 0)."""
        if i is None:
            return
        n = len(self.v.t) if limit is None else min(limit, len(self.v.t))
        depth = 0
        k = i
        while k < n:
            tok = self.v.t[k]
            tt = tok.type
            if depth == 0 and tt in stop_types:
                break
            if tt in (TT.LT, TT.LBRACKET, TT.LPAREN):
                depth += 1
            elif tt in (TT.GT, TT.RBRACKET, TT.RPAREN):
                depth -= 1
                if depth < 0:
                    break
            elif tt == TT.SHR:
                depth -= 2
                if depth < 0:
                    break
            elif tt in (TT.NEWLINE, TT.INDENT, TT.DEDENT, TT.EOF):
                break
            elif tt == TT.IDENT:
                self.type_name_ref(k, tok.value, scope)
            k += 1

    def type_name_ref(self, k: int, name: str, scope: Scope):
        sp = self.v.span(k)
        sym = scope.lookup(name, sp[0])
        if sym is not None and sym.kind in ('struct', 'enum', 'alias', 'trait', 'typeparam', 'capability'):
            self.add_ref(sp[0], sp[1], name, sym, kind='type', scope=scope)
        elif name in BUILTIN_TYPE_NAMES:
            r = self.add_ref(sp[0], sp[1], name, kind='type', scope=scope)
            r.builtin = 'type'
        else:
            self.add_ref(sp[0], sp[1], name, None, kind='type', scope=scope)

    # ── build ───────────────────────────────────────────────────────────────
    def build(self) -> Index:
        stmts = sorted(self.statements, key=lambda s: (getattr(s, 'line', 0) or 0, getattr(s, 'col', 0) or 0))
        for st in stmts:
            try:
                self.collect(st)
            except Exception:           # an index problem must never hide the rest of the file
                if os.environ.get('TILLS_DEBUG'):
                    raise
        self.link_traits()
        for st in stmts:
            try:
                self.walk_toplevel(st)
            except Exception:
                if os.environ.get('TILLS_DEBUG'):
                    raise
        self.ix.finish()
        return self.ix

    def toplevel(self, sym: Optional[Sym], outline=True) -> Optional[Sym]:
        if sym is None:
            return None
        self.ix.module_scope.add(sym, sym.start)
        self.ix.toplevel.setdefault(sym.name, sym)
        if outline:
            self.ix.outline.append(sym)
        return sym

    def collect(self, st):
        v = self.v
        i = self.node_tok(st)
        if isinstance(st, til.FuncDef):
            if st.name == 'main' and (i is None or v.t[i].type != TT.IDENT):
                return                               # the synthetic main of a script
            self.collect_function(st, i, None)
        elif isinstance(st, til.StructDef):
            ni = self.ident_after(i, (TT.STRUCT,))
            sym = self.toplevel(self.make(st.name, 'struct', ni, st))
            if sym is None:
                return
            for fi, fld in zip(self.member_lines(ni), st.fields):
                f = self.make(fld.name, 'field', fi, fld)
                if f is not None and f.name == fld.name:
                    f.container = sym
                    f.typ = _desc(fld.type)
                    f.mutable = True
                    sym.children.append(f)
                    self.ix.members.setdefault((sym.name, f.name), f)
                    self.ix.refs.append(Ref(f.start, f.end, f.name, f, kind='member', decl=True))
        elif isinstance(st, til.EnumDef):
            ni = self.ident_after(i, (TT.ENUM,))
            sym = self.toplevel(self.make(st.name, 'enum', ni, st))
            if sym is None:
                return
            for vi, var in zip(self.member_lines(ni), st.variants):
                m = self.make(var.name, 'variant', vi, var)
                if m is not None and m.name == var.name:
                    m.container = sym
                    sym.children.append(m)
                    self.ix.members.setdefault((sym.name, m.name), m)
                    self.ix.refs.append(Ref(m.start, m.end, m.name, m, kind='member', decl=True))
        elif isinstance(st, til.ImplBlock):
            ii = v.find(i, (TT.IMPL,)) if i is not None else None
            methods = []
            for m in st.methods:
                mi = self.node_tok(m)
                ms = self.collect_function(m, mi, st.type_name)
                if ms is not None:
                    methods.append(ms)
            if ii is not None:
                first = ii + 1
                trait = st.trait_name
                start = v.span(ii)[0]
                end = self.def_end(ii)
                self.ix.impls.append((st.type_name, trait, start, end, first, ii, methods))
        elif isinstance(st, til.TraitDef):
            ni = self.ident_after(i, (TT.TRAIT,))
            sym = self.toplevel(self.make(st.name, 'trait', ni, st))
            if sym is None:
                return
            for m in st.methods:
                mi = self.node_tok(m)
                ms = self.collect_function(m, mi, None, trait=sym)
                if ms is not None:
                    sym.children.append(ms)
        elif isinstance(st, til.VarDecl):
            ni = self.ident_after(i, (TT.LET, TT.VAR, TT.CONST))
            kind = 'const' if st.is_const else 'global'
            sym = self.toplevel(self.make(st.name, kind, ni, st))
            if sym is not None:
                sym.mutable = st.mutable
                sym.typ = _desc(st.type_ann)
                kw = v.find(i, (TT.LET, TT.VAR, TT.CONST))
                if kw is not None:
                    a, b = v.span(kw)
                    sym.decl_word = (a, b, self.text[a:b])
        elif isinstance(st, til.TypeAlias):
            ni = self.ident_after(i, (TT.TYPE,))
            self.toplevel(self.make(st.name, 'alias', ni, st))
        elif isinstance(st, til.CapabilityType):
            ni = self.ident_after(i, (TT.CAP,))
            sym = self.toplevel(self.make(st.name, 'capability', ni, st))
            if sym is not None:
                self.ix.caps.add(st.name)
        elif isinstance(st, til.Import):
            self.collect_import(st, i)

    def make(self, name: str, kind: str, i: Optional[int], node) -> Optional[Sym]:
        sp = self.tok_span(i)
        if sp is None:
            return None
        sym = self.add_sym(Sym(name, kind, sp[0], sp[1]))
        sym.node = node
        first = self.first_line_with_attrs(i)
        sym.full_start = self.line_starts[first] if first < len(self.line_starts) else sp[0]
        while sym.full_start < sp[0] and self.text[sym.full_start] in ' \t':
            sym.full_start += 1
        if kind in ('struct', 'enum', 'trait', 'function', 'method'):
            sym.full_end = self.def_end(i)
        else:
            j = self.v.line_end(i)
            sym.full_end = self.v.span(self.v.last_real_before(j))[1] if j > i else sp[1]
        sym.full_end = max(sym.full_end, sp[1])
        sym.doc = self.doc_above(first)
        if kind not in ('field', 'variant'):
            self.ix.refs.append(Ref(sp[0], sp[1], name, sym, kind='name', decl=True))
        return sym

    def member_lines(self, ni: Optional[int]) -> List[int]:
        """Name tokens of the members (fields / variants) in the block after token ni."""
        out = []
        if ni is None:
            return out
        blk = self.v.block_after(ni)
        if blk is None:
            return out
        k, end = blk[0] + 1, blk[1]
        at_start = True
        depth = 0
        while k < end:
            tok = self.v.t[k]
            if tok.type in (TT.NEWLINE,):
                at_start = depth == 0
            elif tok.type == TT.INDENT:
                depth += 1
            elif tok.type == TT.DEDENT:
                depth -= 1
                at_start = depth == 0
            elif at_start and depth == 0:
                if tok.type == TT.PUB:
                    k += 1
                    continue
                if tok.type == TT.IDENT:
                    out.append(k)
                at_start = False
            k += 1
        return out

    def collect_function(self, fn, i: Optional[int], owner_type: Optional[str], trait: Optional[Sym] = None):
        if i is None:
            return None
        kind = 'method' if owner_type or trait is not None else 'function'
        sym = self.make(fn.name, kind, i, fn)
        if sym is None:
            return None
        sym.key = f"{owner_type}.{fn.name}" if owner_type else fn.name
        if trait is not None:
            sym.container = trait
            sym.key = None
            self.ix.members.setdefault((trait.name, fn.name), sym)
        elif owner_type:
            sym.container_name = owner_type
            self.ix.members.setdefault((owner_type, fn.name), sym)
            sym.static = not (fn.params and fn.params[0].name == 'self')
        else:
            self.toplevel(sym)
            if fn.name == 'main':
                self.ix.main = sym
        # header end: the closing ')' (and return type)
        lp = self.v.find(i, (TT.LPAREN,))
        if lp is not None and lp in self.v.match:
            rp = self.v.match[lp]
            sym.header_end = self.v.span(rp)[1]
            j = self.v.line_end(rp)
            sym.header_end = self.v.span(self.v.last_real_before(j))[1] if j > rp else sym.header_end
        self.ix.functions.append(sym)
        return sym

    def link_traits(self):
        for type_name, trait, _s, _e, _f, _i, _m in self.ix.impls:
            if not trait:
                continue
            tsym = self.ix.toplevel.get(trait)
            if tsym is None or tsym.kind != 'trait':
                continue
            for m in tsym.children:
                self.ix.members.setdefault((type_name, m.name), m)

    def collect_import(self, st, i: Optional[int]):
        v = self.v
        if i is None:
            return
        kw = v.find(i, (TT.IMPORT, TT.FROM))
        if kw is None:
            return
        k = kw + 1
        start = v.span(k)[0] if k < len(v.t) else v.span(kw)[1]
        end = start
        while k < len(v.t) and v.t[k].type in (TT.IDENT, TT.DOT):
            end = v.span(k)[1]
            k += 1
        info = ImportInfo(st.module, st.alias, st.items, start, end, v.t[kw].line - 1)
        self.ix.imports.append(info)
        r = self.add_ref(start, end, st.module, kind='module')
        r.recv = st.module
        if v.t[kw].type == TT.FROM:
            imp = v.find(k, (TT.IMPORT,))
            if imp is not None:
                j = imp + 1
                while j < len(v.t) and v.t[j].type in (TT.IDENT, TT.COMMA):
                    if v.t[j].type == TT.IDENT:
                        a, b = v.span(j)
                        info.item_spans.append((v.t[j].value, a, b))
                        rr = self.add_ref(a, b, v.t[j].value, kind='module-member')
                        rr.recv = st.module
                    j += 1
        else:
            name = st.alias or st.module.split('.')[-1]
            if st.alias:
                ai = v.find(k, (TT.AS,))
                ai = ai + 1 if ai is not None and ai + 1 < len(v.t) else None
                sym = self.make(name, 'module', ai, st)
            else:
                sym = self.add_sym(Sym(name, 'module', start, end))
                sym.node = st
            if sym is not None:
                sym.implicit = True
                self.ix.module_scope.add(sym, 0)
                self.ix.toplevel.setdefault(name, sym)

    # ── bodies ──────────────────────────────────────────────────────────────
    def walk_toplevel(self, st):
        ms = self.ix.module_scope
        i = self.node_tok(st)
        if isinstance(st, til.FuncDef):
            if st.name == 'main' and (i is None or self.v.t[i].type != TT.IDENT):
                self.walk_script(st.body.statements if isinstance(st.body, til.Block) else [])
                return
            sym = self.func_sym(st)
            if st.name == 'main' and isinstance(st.body, til.Block):
                # top-level script statements were moved into main's body by the parser
                script = [x for x in st.body.statements if self.is_script_stmt(x)]
                if script:
                    self.walk_script(script)
                    self.skip.update(id(x) for x in script)
            self.walk_function(st, sym, ms, None)
        elif isinstance(st, til.StructDef):
            ni = self.ident_after(i, (TT.STRUCT,))
            for fi, fld in zip(self.member_lines(ni), st.fields):
                c = self.v.find(fi, (TT.COLON,))
                if c is not None:
                    self.type_refs(c + 1, (TT.EQ,), ms)
                if fld.default is not None:
                    self.expr(fld.default, ms)
        elif isinstance(st, til.EnumDef):
            ni = self.ident_after(i, (TT.ENUM,))
            for vi, var in zip(self.member_lines(ni), st.variants):
                if var.fields and vi + 1 < len(self.v.t) and self.v.t[vi + 1].type == TT.LPAREN:
                    self.type_refs(vi + 2, (TT.RPAREN,), ms)
                if var.value is not None:
                    self.expr(var.value, ms)
        elif isinstance(st, til.ImplBlock):
            ii = self.v.find(i, (TT.IMPL,)) if i is not None else None
            if ii is not None:
                k = ii + 1
                names = [st.trait_name, st.type_name] if st.trait_name else [st.type_name]
                for name in names:
                    if k < len(self.v.t) and self.v.t[k].type == TT.IDENT:
                        self.type_name_ref(k, name, ms)
                        k += 1
                    if k < len(self.v.t) and self.v.t[k].type == TT.FOR:
                        k += 1
            for m in st.methods:
                self.walk_function(m, self.func_sym(m), ms, st.type_name)
        elif isinstance(st, til.TraitDef):
            for m in st.methods:
                self.walk_function(m, self.func_sym(m), ms, st.name)
        elif isinstance(st, til.VarDecl):
            ni = self.ident_after(i, (TT.LET, TT.VAR, TT.CONST))
            if st.type_ann is not None and ni is not None:
                self.type_refs(ni + 2, (TT.EQ,), ms)
            if st.value is not None:
                sym = self.ix.toplevel.get(st.name)
                self.expr(st.value, ms)
                if sym is not None and sym.typ is None:
                    sym.typ = self.type_of(st.value, ms)
        elif isinstance(st, til.TypeAlias):
            ni = self.ident_after(i, (TT.TYPE,))
            if ni is not None:
                eq = self.v.find(ni, (TT.EQ,))
                if eq is not None:
                    self.type_refs(eq + 1, (), ms)

    def func_sym(self, fn) -> Optional[Sym]:
        i = self.node_tok(fn)
        if i is None:
            return None
        sp = self.v.span(i)
        for s in self.ix.functions:
            if s.start == sp[0]:
                return s
        return None

    def walk_script(self, stmts):
        ms = self.ix.module_scope
        script = [s for s in stmts if self.is_script_stmt(s)]
        rest = [s for s in stmts if not self.is_script_stmt(s)]
        if script:
            self.ix.has_script = True
            first = self.node_tok(script[0])
            self.ix.script_line = (self.v.t[first].line - 1) if first is not None else 0
            sc = Scope('script', ms, 0, len(self.text) + 1, 0)
            self.ix.scopes.append(sc)
            self.ix.script_scope = sc
            for s in script:
                if not isinstance(s, til.GlobalInit):
                    self.stmt(s, sc)
        return rest

    def is_script_stmt(self, s) -> bool:
        if isinstance(s, til.GlobalInit):
            return True
        i = self.node_tok(s)
        if i is None:
            return False
        tok = self.v.t[i]
        ln = tok.line - 1
        return 0 <= ln < len(self.lines) and _indent_width(self.lines[ln]) == 0 and \
            self.lines[ln][:max(tok.col - 1, 0)].strip() == ""

    def walk_function(self, fn, sym: Optional[Sym], parent: Scope, self_type: Optional[str]):
        i = self.node_tok(fn)
        if i is None:
            return
        v = self.v
        start = sym.full_start if sym is not None else v.span(i)[0]
        blk = v.block_after(i)
        # the scope reaches the next dedented line (blank lines at the end of the body belong to it)
        end = v.span(blk[1])[0] if blk is not None else (sym.full_end if sym is not None else self.def_end(i))
        body_indent = v.t[blk[0] + 1].col - 1 if blk is not None and blk[0] + 1 < len(v.t) else 4
        fsc = Scope('function', parent, start, end, body_indent if blk is not None else 0, sym)
        self.ix.scopes.append(fsc)
        if sym is not None:
            self.fn_stack.append(sym)
        try:
            # generic type parameters: name<T, U>(...) or #[generic: T]
            k = i + 1
            if k < len(v.t) and v.t[k].type == TT.LT:
                k += 1
                while k < len(v.t) and v.t[k].type in (TT.IDENT, TT.COMMA):
                    if v.t[k].type == TT.IDENT:
                        self.declare(fsc, v.t[k].value, 'typeparam', k, at=fsc.start)
                    k += 1
            for attr in getattr(fn, 'attributes', None) or []:
                if attr.startswith('generic:'):
                    for tname in attr[len('generic:'):].split(','):
                        tname = tname.strip()
                        if tname and fsc.find(tname, None) is None:
                            ts = self.add_sym(Sym(tname, 'typeparam', v.span(i)[0], v.span(i)[0]))
                            ts.implicit = True
                            fsc.add(ts, 0)
            # parameters
            lp = v.find(i, (TT.LPAREN,))
            rp = v.match.get(lp) if lp is not None else None
            if lp is not None and rp is not None:
                self.walk_params(fn, sym, fsc, lp, rp, self_type)
                j = rp + 1
                if j < len(v.t) and v.t[j].type == TT.ARROW:
                    self.type_refs(j + 1, (TT.NEWLINE, TT.INDENT, TT.LBRACE, TT.FAT_ARROW), fsc)
            # contracts (in the attributes above the function)
            self.walk_contracts(fn, i, fsc)
            self.block(fn.body, fsc, kind='body')
        finally:
            if sym is not None:
                self.fn_stack.pop()

    def walk_params(self, fn, sym, fsc, lp, rp, self_type):
        v = self.v
        starts = []
        depth = 0
        k = lp + 1
        expect = True
        while k < rp:
            tt = v.t[k].type
            if tt in OPENERS:
                depth += 1
            elif tt in CLOSERS:
                depth -= 1
            elif depth == 0 and tt == TT.COMMA:
                expect = True
            elif expect and depth == 0 and tt in (TT.IDENT, TT.SELF):
                starts.append(k)
                expect = False
            k += 1
        for p, k in zip(fn.params, starts):
            # parameters are visible in the whole function (its contracts above it too)
            if p.name == 'self':
                ps = self.declare(fsc, 'self', 'param', k, p,
                                  ('struct', self_type) if self_type else None, mutable=p.mutable, at=fsc.start)
                if ps is not None:
                    ps.implicit = True
            else:
                ps = self.declare(fsc, p.name, 'param', k, p, _desc(p.type), mutable=True, at=fsc.start)
            if ps is not None and sym is not None:
                sym.params.append(ps)
            # the annotation and the default value
            c = k + 1
            if c < rp and v.t[c].type == TT.COLON:
                self.type_refs(c + 1, (TT.COMMA, TT.EQ, TT.RPAREN), fsc, limit=rp)
            if p.default is not None:
                self.expr(p.default, fsc)
                eq = v.find(k + 1, (TT.EQ,), limit=rp)
                if eq is not None and ps is not None:
                    end = v.find(eq + 1, (TT.COMMA,), limit=rp)
                    end = rp if end is None else end
                    a = v.span(eq + 1)[0] if eq + 1 < end else v.span(eq)[1]
                    b = v.span(v.last_real_before(end))[1] if end > eq + 1 else a
                    ps.default_text = self.text[a:b].strip() if v is self.main_view else None

    def walk_contracts(self, fn, i: int, fsc: Scope):
        if not (getattr(fn, 'requires', None) or getattr(fn, 'ensures', None)):
            return
        attrs = self.attrs_before(i)
        req = list(fn.requires)
        ens = list(fn.ensures)
        res = None
        for ai in attrs:
            tok = self.v.t[ai]
            value = tok.value
            if not (value.startswith('requires:') or value.startswith('ensures:')):
                continue
            kind, expr_text = value.split(':', 1)
            expr_text = expr_text.strip()
            node = (req.pop(0) if req else None) if kind == 'requires' else (ens.pop(0) if ens else None)
            if node is None:
                continue
            raw = self.text[tok.start:tok.end]
            colon = raw.find(':')
            off = raw.find(expr_text, colon + 1) if expr_text else -1
            if off < 0:
                continue
            scope = fsc
            if kind == 'ensures':
                if res is None:
                    res = Scope('ensures', fsc, fsc.start, fsc.end, fsc.indent)
                    self.ix.scopes.append(res)
                    rsym = self.add_sym(Sym('result', 'local', tok.start, tok.start))
                    rsym.implicit = True
                    rsym.typ = _desc(fn.ret_type)
                    rsym.owner = self.fn_stack[-1] if self.fn_stack else None
                    res.add(rsym, 0)
                scope = res
            self.fragment(expr_text, tok.start + off, max(self.v.t[i].line - 1, 0), node, scope)

    def fragment(self, text: str, base: int, line_offset: int, node, scope: Scope):
        """Walk a sub-parsed expression (f-string part, contract) with its own tokens."""
        try:
            toks = til.Lexer(text, "<fragment>", languages=self.languages, line_offset=line_offset).tokenize()
        except Exception:
            return
        saved = self.v
        self.v = _View(toks, base)
        try:
            self.expr(node, scope)
        finally:
            self.v = saved

    def block(self, body, scope: Scope, kind: str = 'block'):
        if body is None:
            return
        if isinstance(body, til.Block):
            stmts = [st for st in body.statements if id(st) not in self.skip] if self.skip else body.statements
            sc = self.block_scope(til.Block(statements=stmts) if len(stmts) != len(body.statements) else body,
                                  scope, kind)
            for st in stmts:
                self.stmt(st, sc)
        else:
            self.stmt(body, scope)

    def stmt(self, n, scope: Scope):
        if n is None:
            return
        t = type(n)
        try:
            if t is til.VarDecl:
                self.var_decl(n, scope)
            elif t is til.Assignment:
                self.expr(n.value, scope)
                if isinstance(n.target, til.Identifier):
                    self.name_ref(n.target, scope, write=True)
                else:
                    self.expr(n.target, scope)
                    r = self.ix.node_refs.get(id(n.target))
                    if r is not None:
                        r.write = True
            elif t is til.If:
                self.expr(n.condition, scope)
                self.block(n.then_body, scope)
                for cond, body in n.elifs:
                    self.expr(cond, scope)
                    self.block(body, scope)
                self.block(n.else_body, scope)
            elif t is til.For:
                self.for_loop(n, scope)
            elif t is til.While:
                self.expr(n.condition, scope)
                self.block(n.body, scope)
            elif t is til.Loop:
                self.block(n.body, scope)
            elif t is til.Return:
                self.expr(n.value, scope)
            elif t is til.ExprStmt:
                self.expr(n.expr, scope)
            elif t is til.Block:
                self.block(n, scope)
            elif t in (til.Break, til.Continue, til.GlobalInit):
                pass
            elif t in (til.AssertStmt, til.InvariantStmt):
                self.expr(n.condition, scope)
            else:
                self.expr(n, scope)
        except Exception:
            if os.environ.get('TILLS_DEBUG'):
                raise

    def var_decl(self, n, scope: Scope):
        i = self.node_tok(n)
        kw = self.v.find(i, (TT.LET, TT.VAR, TT.CONST)) if i is not None else None
        ni = kw + 1 if kw is not None and kw + 1 < len(self.v.t) and self.v.t[kw + 1].type == TT.IDENT else None
        if n.type_ann is not None and ni is not None:
            self.type_refs(ni + 2, (TT.EQ,), scope)
        self.expr(n.value, scope)
        typ = _desc(n.type_ann) or self.type_of(n.value, scope)
        kind = 'const' if n.is_const else 'local'
        sym = self.declare(scope, n.name, kind, ni, n, typ, mutable=n.mutable)
        if sym is not None and kw is not None:
            a, b = self.v.span(kw)
            sym.decl_word = (a, b, self.text[a:b] if self.v is self.main_view else '')

    def for_loop(self, n, scope: Scope):
        self.expr(n.iter, scope)
        i = self.node_tok(n)
        vi = self.ident_after(i, (TT.FOR,))
        end = scope.end
        if i is not None:
            blk = self.v.block_after(i)
            end = self.v.span(blk[1])[0] if blk is not None else self.v.span(self.v.line_end(i))[0]
        start = self.v.span(i)[0] if i is not None else scope.start
        fsc = Scope('for', scope, start, end, scope.indent)
        self.ix.scopes.append(fsc)
        elem = None
        it = self.type_of(n.iter, scope)
        if it is not None:
            if it[0] == 'array':
                elem = it[1] if len(it) > 1 else None
            elif it[0] == 'str':
                elem = ('str',)
            elif it[0] == 'map':
                elem = it[1] if len(it) > 1 else None
        self.declare(fsc, n.var, 'local', vi, n, elem)
        self.block(n.body, fsc)

    # ── expressions ─────────────────────────────────────────────────────────
    def expr(self, e, scope: Scope):
        if e is None or not isinstance(e, til.ASTNode):
            return
        t = type(e)
        if t is til.Identifier:
            self.name_ref(e, scope)
        elif t is til.Attribute:
            self.attribute(e, scope)
        elif t is til.Call:
            self.expr(e.func, scope)
            for a in e.args:
                self.expr(a, scope)
        elif t is til.StructInit:
            self.struct_init(e, scope)
        elif t is til.Lambda:
            self.lambda_(e, scope)
        elif t is til.ListComprehension:
            self.comprehension(e, scope)
        elif t is til.FStringLit:
            self.fstring(e, scope)
        elif t is til.MatchExpr:
            self.match_(e, scope)
        elif t is til.Cast:
            self.expr(e.expr, scope)
            i = self.node_tok(e)
            if i is not None and self.v.t[i].type == TT.AS:
                self.type_refs(i + 1, (TT.NEWLINE, TT.RPAREN, TT.COMMA, TT.RBRACKET, TT.DOT, TT.PLUS, TT.MINUS,
                                       TT.STAR, TT.SLASH, TT.EQEQ, TT.NEQ, TT.IF, TT.ELSE, TT.AND, TT.OR,
                                       TT.RBRACE, TT.COLON, TT.QUESTION), scope)
        elif t in (til.RetryExpr, til.TimeoutExpr):
            self.expr(e.count if t is til.RetryExpr else e.duration, scope)
            self.body_or_expr(e.body, scope)
            if e.fallback is not None:
                fsc = Scope('fallback', scope, scope.start, scope.end, scope.indent)
                self.ix.scopes.append(fsc)
                err = self.add_sym(Sym('error', 'local', 0, 0))
                err.implicit = True
                err.typ = ('str',)
                err.owner = self.fn_stack[-1] if self.fn_stack else None
                fsc.add(err, 0)
                self.body_or_expr(e.fallback, fsc)
        elif t is til.EffectHandler:
            self.body_or_expr(e.expr, scope)
            for clause in e.handlers:
                self.clause(clause, scope)
        elif t is til.PerformEffect:
            i = self.node_tok(e)
            if i is not None:
                ni = i + 1 if i + 1 < len(self.v.t) and self.v.t[i + 1].type == TT.IDENT else None
                if ni is not None:
                    self.effect_ref(ni, e.name, scope)
            for a in e.args:
                self.expr(a, scope)
        elif t is til.IntentBlock:
            self.body_or_expr(e.body, scope)
        elif t is til.Block:
            self.block(e, scope)
        elif t in (til.IntLit, til.FloatLit, til.StringLit, til.BoolLit, til.CharLit, til.DurationLit):
            pass
        else:
            for child in til.iter_children(e):
                if isinstance(child, (til.Block,)):
                    self.block(child, scope)
                else:
                    self.expr(child, scope)

    def body_or_expr(self, body, scope: Scope):
        if isinstance(body, til.Block):
            self.block(body, scope)
        else:
            self.expr(body, scope)

    def effect_ref(self, k: int, name: str, scope: Scope):
        sp = self.v.span(k)
        self.ix.effects.add(name)
        target = self.ix.module_scope.find(name, None)
        if target is not None and target.kind == 'function':
            self.add_ref(sp[0], sp[1], name, target, kind='effect', scope=scope)
        else:
            self.add_ref(sp[0], sp[1], name, None, kind='effect', scope=scope)

    def clause(self, c, scope: Scope):
        i = self.node_tok(c)
        csc = Scope('clause', scope, scope.start, scope.end, scope.indent)
        self.ix.scopes.append(csc)
        if i is not None:
            self.effect_ref(i, c.name, scope)
            if i + 1 < len(self.v.t) and self.v.t[i + 1].type == TT.LPAREN:
                rp = self.v.match.get(i + 1, i + 1)
                k = i + 2
                names = list(c.params)
                while k < rp and names:
                    if self.v.t[k].type == TT.IDENT and self.v.t[k].value == names[0]:
                        self.declare(csc, names.pop(0), 'param', k, c)
                        if k + 1 < rp and self.v.t[k + 1].type == TT.COLON:
                            self.type_refs(k + 2, (TT.COMMA, TT.RPAREN), csc, limit=rp)
                    k += 1
            blk = self.v.block_after(i)
            csc.start = self.v.span(i)[0]
            csc.end = self.v.span(blk[1])[0] if blk is not None else self.v.span(self.v.line_end(i))[0]
        self.body_or_expr(c.body, csc)

    def attribute(self, e, scope: Scope):
        self.expr(e.obj, scope)
        i = self.node_tok(e)
        if i is None or i + 1 >= len(self.v.t) or self.v.t[i].type != TT.DOT:
            return
        k = i + 1
        if self.v.t[k].type != TT.IDENT:
            return
        a, b = self.v.span(k)
        target = self.member_target(e.obj, e.attr, scope)
        r = self.add_ref(a, b, e.attr, kind='member', scope=scope)
        self.ix.node_refs[id(e)] = r
        if isinstance(target, Sym):
            r.sym = target
            self.node_sym[id(e)] = target
        elif isinstance(target, tuple):
            what, name = target[0], target[1]
            if what == 'module':
                r.kind = 'module-member'
                r.recv = name
            elif what == 'type':
                r.recv = name
            elif what == 'builtin':
                r.builtin = 'method'
                r.recv = name
            elif what == 'guess':
                r.sym = target[2]
                r.guess = True
                self.node_sym[id(e)] = target[2]

    def member_target(self, obj, attr: str, scope: Scope):
        if isinstance(obj, til.Identifier):
            s = self.node_sym.get(id(obj))
            if isinstance(s, Sym):
                if s.kind in ('enum', 'struct', 'trait'):
                    m = self.ix.members.get((s.name, attr))
                    return m if m is not None else ('type', s.name)
                if s.kind == 'module':
                    return ('module', s.node.module if s.node is not None else s.name)
            elif obj.name not in BUILTIN_NAMES and obj.name != 'self':
                # a name defined in another file: a type, an enum or a module
                return ('type-or-module', obj.name)
        t = self.type_of(obj, scope)
        if t is not None:
            if t[0] in ('struct', 'enum'):
                m = self.ix.members.get((t[1], attr))
                return m if m is not None else ('type', t[1])
            if t[0] in ('str', 'array', 'map', 'option', 'result'):
                return ('builtin', t[0])
        cands = [s for (c, m), s in self.ix.members.items() if m == attr and s.kind in ('field', 'method')]
        if len(cands) == 1:
            return ('guess', attr, cands[0])
        if attr in METHOD_DOCS and not cands:
            return ('builtin', '')
        return None

    def struct_init(self, e, scope: Scope):
        i = self.node_tok(e)
        if i is None:
            return
        self.type_name_ref(i, e.name, scope)
        struct = scope.lookup(e.name, None)
        lb = i + 1
        if lb < len(self.v.t) and self.v.t[lb].type == TT.LBRACE:
            rb = self.v.match.get(lb, lb)
            k = lb + 1
            depth = 0
            while k < rb:
                tt = self.v.t[k].type
                if tt in OPENERS:
                    depth += 1
                elif tt in CLOSERS:
                    depth -= 1
                elif depth == 0 and tt == TT.IDENT and k + 1 < rb and self.v.t[k + 1].type == TT.COLON and \
                        (k == lb + 1 or self.v.t[k - 1].type == TT.COMMA or self.v.t[k - 1].type == TT.NEWLINE):
                    a, b = self.v.span(k)
                    fname = self.v.t[k].value
                    target = self.ix.members.get((e.name, fname)) if struct is not None else None
                    r = self.add_ref(a, b, fname, target, kind='member', scope=scope)
                    if target is None:
                        r.recv = e.name
                k += 1
        for value in e.fields.values():
            self.expr(value, scope)

    def lambda_(self, e, scope: Scope):
        i = self.node_tok(e)
        lsc = Scope('lambda', scope, scope.start, scope.end, scope.indent)
        self.ix.scopes.append(lsc)
        if i is not None and self.v.t[i].type == TT.PIPE:
            k = i + 1
            names = [p[0] for p in e.params]
            while k < len(self.v.t) and self.v.t[k].type != TT.PIPE and names:
                if self.v.t[k].type == TT.IDENT and self.v.t[k].value == names[0]:
                    self.declare(lsc, names.pop(0), 'param', k, e)
                    if k + 1 < len(self.v.t) and self.v.t[k + 1].type == TT.COLON:
                        self.type_refs(k + 2, (TT.COMMA, TT.PIPE), lsc)
                k += 1
            lsc.start = self.v.span(i)[0]
            end_i = self.v.line_end(i)
            lsc.end = self.v.span(end_i)[0]
            close = self.v.find(i + 1, (TT.PIPE,))
            if close is not None and close + 1 < len(self.v.t) and self.v.t[close + 1].type == TT.ARROW:
                self.type_refs(close + 2, (TT.LBRACE, TT.NEWLINE), lsc)
        self.body_or_expr(e.body, lsc)

    def comprehension(self, e, scope: Scope):
        i = self.node_tok(e)
        self.expr(e.iter, scope)
        csc = Scope('comprehension', scope, scope.start, scope.end, scope.indent)
        self.ix.scopes.append(csc)
        if i is not None and self.v.t[i].type == TT.LBRACKET:
            close = self.v.match.get(i, len(self.v.t) - 1)
            csc.start, csc.end = self.v.span(i)[0], self.v.span(close)[1]
            fi = self.v.find(i + 1, (TT.FOR,), limit=close)
            vi = fi + 1 if fi is not None and fi + 1 < close and self.v.t[fi + 1].type == TT.IDENT else None
            elem = None
            it = self.type_of(e.iter, scope)
            if it is not None and it[0] == 'array':
                elem = it[1] if len(it) > 1 else None
            self.declare(csc, e.var, 'local', vi, e, elem, at=self.v.span(i)[0])
        self.expr(e.expr, csc)
        self.expr(e.condition, csc)

    def fstring(self, e, scope: Scope):
        i = self.node_tok(e)
        if i is None or self.v is not self.main_view:
            return
        tok = self.v.t[i]
        if tok.type != TT.FSTRING:
            return
        raw = self.text[tok.start:tok.end]
        offs = _fstring_expr_offsets(raw)
        k = 0
        for part in e.parts:
            if not (isinstance(part, (tuple, list)) and part and part[0] == 'expr'):
                continue
            if k >= len(offs):
                break
            node = part[1]
            # the expression text the parser saw (before the :format spec)
            text_parts = [p for p in tok.value if p[0] == 'expr']
            ptext = text_parts[k][1] if k < len(text_parts) else ''
            self.fragment(ptext, tok.start + offs[k], max(tok.line - 1, 0), node, scope)
            k += 1

    def match_(self, e, scope: Scope):
        self.expr(e.value, scope)
        for arm in e.arms:
            pat, guard, body = arm[0], arm[1], arm[2]
            asc = Scope('arm', scope, scope.start, scope.end, scope.indent)
            self.ix.scopes.append(asc)
            pi = self.node_tok(pat)
            if pi is not None:
                asc.start = self.v.span(pi)[0]
                blk = self.v.block_after(pi)
                asc.end = self.v.span(blk[1])[0] if blk is not None else self.v.span(self.v.line_end(pi))[0]
            self.pattern(pat, asc)
            self.expr(guard, asc)
            self.body_or_expr(body, asc)

    def pattern(self, p, scope: Scope):
        if p is None:
            return
        if isinstance(p, til.Identifier):
            if p.name in ('_', 'NULL'):
                if p.name == 'NULL':
                    self.name_ref(p, scope)
                return
            existing = scope.lookup(p.name, None)
            if existing is not None and existing.kind in ('global', 'const') and \
                    existing.scope is self.ix.module_scope:
                self.name_ref(p, scope)
                return
            i = self.node_tok(p)
            sym = self.declare(scope, p.name, 'local', i, p)
            if sym is not None:
                self.node_sym[id(p)] = sym
            return
        if isinstance(p, til.Attribute):
            self.attribute(p, scope)
            return
        if isinstance(p, til.Call):
            self.expr(p.func, scope)
            for a in p.args:
                self.pattern(a, scope)
            return
        if isinstance(p, (til.TupleLit, til.ArrayLit)):
            for el in p.elements:
                self.pattern(el, scope)
            return
        if isinstance(p, til.StructInit):
            self.struct_init_pattern(p, scope)
            return
        self.expr(p, scope)

    def struct_init_pattern(self, p, scope: Scope):
        i = self.node_tok(p)
        if i is not None:
            self.type_name_ref(i, p.name, scope)
        for value in p.fields.values():
            self.pattern(value, scope)


def build_index(text: str, lexed: Lexed, statements, uri: str) -> Index:
    return Indexer(text, lexed.tokens, statements, uri, lexed.languages, lexed.comments).build()


# ═══════════════════════════════════════════════════════════════════════════════
#                                  DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

class RenameError(Exception):
    """A rename (or prepareRename) was refused; str(e) is the localized reason."""


class Document:
    def __init__(self, uri: str, text: str, version: int = 0, is_open: bool = True,
                 path: Optional[str] = None):
        self.uri = uri
        self.path = path if path is not None else uri_to_path(uri)
        self.is_open = is_open
        self.version = version
        self.mtime: Optional[Tuple[float, int]] = None
        self.syntax: Optional['Syntax'] = None
        self.semantic: Optional['Semantic'] = None
        self.last_good: Optional['Syntax'] = None
        self.last_types: Optional['Semantic'] = None   # the newest analysis with types
        self.prev_syntax: Optional['Syntax'] = None    # the newest analysis of any version
        self.translations: Dict[str, Any] = {}          # code actions: translate_source() per version
        self.set_text(text, version)

    def set_text(self, text: str, version: int):
        self.raw = text
        self.version = version
        self.eol = '\r\n' if '\r\n' in text else '\n'
        self.text = text.replace('\r\n', '\n').replace('\r', '\n')
        self.pmap = PositionMap(self.text)
        if self.syntax is not None and self.syntax.ok:
            self.last_good = self.syntax
        self.syntax = None
        self.semantic = None


class Syntax:
    """Tier 1 of an analysis: tokens, the (repaired) parse, the symbol index."""

    def __init__(self):
        self.version = 0
        self.lexed: Optional[Lexed] = None
        self.parsed: Optional[Parsed] = None
        self.index: Optional[Index] = None
        self.ok = False                    # parsed without repairs
        self.error: Optional[RawDiag] = None
        self.error_span: Optional[Tuple[int, int]] = None
        self.languages: List[str] = list(til.DEFAULT_LANGUAGES)
        self.primary = 'en'                # keyword language the file is written in
        self.view: Optional[_View] = None  # tokens of the (repaired) text
        self.pmap: Optional[PositionMap] = None   # positions of the text it was built from
        self.text: Optional[str] = None
        self.time = 0.0


class Semantic:
    """Tier 2: diagnostics of the whole pipeline and inferred types."""

    def __init__(self):
        self.version = 0
        self.diags: Dict[str, List[dict]] = {}
        self.types = None                  # tiltypes.TypeInfer, or None
        self.tt = None                     # the tiltypes module
        self.instances: Dict[str, Any] = {}
        self.time = 0.0


def _primary_language(lexed: Lexed) -> str:
    """The keyword language a file is written in: its #[lang: ...] pragma, else
    Kazakh when its keywords are Kazakh, else English."""
    for code in lexed.languages:
        if code not in til.DEFAULT_LANGUAGES:
            return code
    kk = en = 0
    text = lexed.text
    for tok in lexed.tokens:
        if tok.type in KEYWORD_TYPES or tok.type == TT.BOOL:
            word = text[tok.start:tok.end]
            if word in til.KAZAKH_KEYWORDS:
                kk += 1
            elif word in til.ENGLISH_KEYWORDS:
                en += 1
    return 'kk' if kk > en else 'en'


# ═══════════════════════════════════════════════════════════════════════════════
#                                  WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════════

_SKIP_DIRS = {'.git', '.hg', '.svn', 'node_modules', '__pycache__', 'til_modules', '.venv', 'venv', 'build',
              'dist', '.idea', '.vscode'}
MAX_WORKSPACE_FILES = 3000
_WORD = re.compile(r'[^\W\d]\w*')
_QUOTED = re.compile(r"'([^'\s]+)'")


class Workspace:
    """All language features over the open documents (and the .til files of
    the root folder, for cross-file definition / references / rename)."""

    def __init__(self, root: Optional[str] = None, locale: str = 'en'):
        self.root = os.path.abspath(root) if root else None
        self.docs: Dict[str, Document] = {}
        self._disk: Dict[str, Document] = {}
        self._files_cache: Tuple[float, List[str]] = (0.0, [])
        self._gen = 0                       # bumped by every change: caches of the import graph
        self._graph_cache: Dict[Tuple[str, str], Tuple[int, Any, float]] = {}
        self._cheap = False                 # completion, signature help, outline: no whole-pipeline runs
        self._reuse = False                 # completion, signature help: reuse the last analysis when
                                            # only the cursor line changed
        try:
            self.tt = til.types_module()      # None: no tiltypes (or TIL_NO_TYPES)
        except Exception:                     # a broken tiltypes.py: work without types
            self.tt = None
        self.set_locale(locale)

    def set_locale(self, locale: Optional[str]):
        self.locale = locale or 'en'
        self.lang = message_lang(self.locale)
        for doc in list(self.docs.values()) + list(self._disk.values()):
            doc.semantic = None

    # ── documents ───────────────────────────────────────────────────────────
    def open(self, uri: str, text: str, version: int = 0) -> None:
        doc = self.docs.get(uri)
        if doc is None:
            self.docs[uri] = Document(uri, text, version, True)
        else:
            doc.set_text(text, version)
        self._invalidate_dependents(uri)

    def change(self, uri: str, text: str, version: int) -> None:
        doc = self.docs.get(uri)
        if doc is None:
            self.open(uri, text, version)
            return
        if doc.raw == text and doc.version == version:
            return
        doc.set_text(text, version)
        self._invalidate_dependents(uri)

    def close(self, uri: str) -> None:
        self.docs.pop(uri, None)
        self._invalidate_dependents(uri)

    def saved(self, uri: str) -> None:
        """A document was saved: files that import it see the new text (the
        compiler reads imports from disk)."""
        doc = self.docs.get(uri)
        if doc is not None and doc.path:
            self._disk.pop(path_key(doc.path), None)
        self._invalidate_dependents(uri)

    def files_changed(self, paths: Iterable[str] = ()) -> None:
        self._gen += 1
        for p in paths:
            self._disk.pop(path_key(p), None)
        self._files_cache = (0.0, [])
        for doc in self.docs.values():
            doc.semantic = None

    def _invalidate_dependents(self, uri: str):
        self._gen += 1
        for d in self.docs.values():
            if d.uri != uri:
                d.semantic = None

    def _doc(self, uri: str) -> Optional[Document]:
        doc = self.docs.get(uri)
        if doc is not None:
            return doc
        path = uri_to_path(uri)
        if path:
            return self._doc_for_path(path)
        return None

    def _doc_for_path(self, path: str, prefer_open: bool = True) -> Optional[Document]:
        """The open document of a file, or (prefer_open=False: always) the file as
        it is on disk -- what the compiler reads for an import."""
        key = path_key(path)
        if prefer_open:
            for d in self.docs.values():
                if d.path and path_key(d.path) == key:
                    return d
        try:
            st = os.stat(path)
        except OSError:
            return None
        stamp = (st.st_mtime, st.st_size)
        doc = self._disk.get(key)
        if doc is not None and doc.mtime == stamp:
            return doc
        if st.st_size > 4 * 1024 * 1024:
            return None
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except OSError:
            return None
        doc = Document(path_to_uri(path), text, 0, False, path=os.path.abspath(path))
        doc.mtime = stamp
        self._disk[key] = doc
        return doc

    def _filename(self, doc: Document) -> str:
        if doc.path:
            return doc.path
        base = self.root or os.getcwd()
        return os.path.join(base, '__untitled__.til')

    # ── tier 1: syntax + symbols ────────────────────────────────────────────
    def _syntax(self, doc: Document) -> Syntax:
        syn = doc.syntax
        if syn is not None and syn.version == doc.version:
            return syn
        if self._reuse:
            prev = doc.prev_syntax
            if prev is not None and prev.text is not None and _one_line_change(prev.text, doc.text):
                # typing on one line: the last analysis is valid for every other line
                # (offsets up to the end of the changed line are the same)
                return prev
        t0 = time.perf_counter()
        syn = Syntax()
        syn.version = doc.version
        filename = self._filename(doc)
        lexed = lex(doc.text, filename)
        syn.lexed = lexed
        syn.languages = lexed.languages
        syn.primary = _primary_language(lexed)
        if lexed.error is not None:
            syn.error = lexed.error
            syn.error_span = lexed.error_span
        parsed = parse(doc.text, lexed, filename)
        syn.parsed = parsed
        if syn.error is None and parsed.error is not None:
            syn.error = parsed.error
            syn.error_span = self._parse_error_span(doc, lexed, parsed)
        view_lexed = parsed.lexed or lexed
        if parsed.program is not None:
            syn.ok = syn.error is None
            try:
                syn.index = build_index(doc.text, view_lexed, parsed.statements, doc.uri)
            except Exception:
                if os.environ.get('TILLS_DEBUG'):
                    raise
                syn.index = None
        if syn.index is None:
            syn.index = Index(doc.uri)
            syn.index.finish()
            if doc.last_good is not None and doc.last_good.index is not None:
                syn.index = doc.last_good.index
        syn.view = _View(view_lexed.tokens)
        syn.pmap = doc.pmap
        syn.text = doc.text
        doc.prev_syntax = syn
        syn.time = time.perf_counter() - t0
        doc.syntax = syn
        if syn.ok:
            doc.last_good = syn
        return syn

    def _parse_error_span(self, doc: Document, lexed: Lexed, parsed: Parsed) -> Optional[Tuple[int, int]]:
        raw = parsed.error
        if raw is None:
            return None
        if raw.filename in ('<attr>', '<fstring>'):
            sp = self._fragment_error_span(doc, lexed, raw)
            if sp is not None:
                return sp
        tok = parsed.error_tok
        view = _View(lexed.tokens)
        if tok is not None:
            i = view.at(tok.line, tok.col) if tok.type not in LAYOUT else None
            if i is None:
                for k, t in enumerate(lexed.tokens):
                    if t is tok:
                        i = k
                        break
            if i is not None:
                closer = re.match(r"Expected (RPAREN|RBRACKET|RBRACE), got ", raw.text)
                if closer is not None:
                    # the bracket was never closed: point at the one that was opened
                    want = getattr(TT, closer.group(1))
                    opener = next(o for o, c in OPENERS.items() if c == want)
                    stack = []
                    for k in range(i):
                        tt = view.t[k].type
                        if tt in OPENERS:
                            stack.append(k)
                        elif tt in CLOSERS and stack:
                            stack.pop()
                    for k in reversed(stack):
                        if view.t[k].type == opener:
                            if view.t[k].line < view.t[i].line or view.t[i].type in LAYOUT:
                                return view.span(k)
                            break
                return self._token_range(doc.text, view, i, raw.text)
        return self._span_at(doc.text, view, raw.line, raw.col, raw.text, doc.pmap)

    def _fragment_error_span(self, doc: Document, lexed: Lexed, raw: RawDiag) -> Optional[Tuple[int, int]]:
        """A syntax error inside an f-string expression or a #[requires/ensures]
        contract: the compiler reports it relative to the fragment."""
        text = doc.text
        line_toks = [t for t in lexed.tokens if t.line == raw.line] if raw.filename == '<fstring>' else []
        if raw.filename == '<fstring>':
            for t in line_toks:
                if t.type != TT.FSTRING:
                    continue
                src = text[t.start:t.end]
                offs = _fstring_expr_offsets(src)
                parts = [p for p in t.value if p[0] == 'expr']
                for off, part in zip(offs, parts):
                    if _fragment_fails(part[1], lexed.languages):
                        a = t.start + off + max(raw.col - 1, 0)
                        return a, min(a + 1, t.end)
            return None
        # <attr>: the contract attributes above the function named on raw.line
        attrs = [t for t in lexed.tokens if t.type == TT.ATTRIBUTE and t.line <= raw.line and
                 (t.value.startswith('requires:') or t.value.startswith('ensures:'))]
        for t in reversed(attrs):
            expr_text = t.value.split(':', 1)[1].strip()
            if _fragment_fails(expr_text, lexed.languages):
                src = text[t.start:t.end]
                off = src.find(expr_text, src.find(':') + 1)
                a = t.start + max(off, 0) + max(raw.col - 1, 0)
                return a, min(max(a + 1, a), t.end) if a < t.end else (t.start, t.end)
        return None

    def _token_range(self, text: str, view: _View, i: int, message: str) -> Tuple[int, int]:
        tok = view.t[i]
        if tok.type in LAYOUT or tok.end <= tok.start:
            # a line end / indentation: point at the last token before it on its line
            k = i - 1
            while k >= 0 and (view.t[k].type in LAYOUT or view.t[k].end <= view.t[k].start):
                k -= 1
            if k >= 0 and view.t[k].line in (tok.line, tok.line - 1):
                return view.span(k)
            a = tok.start
            return (a, a + 1) if a < len(text) and text[a] != '\n' else (max(a - 1, 0), max(a, 1))
        a, b = view.span(i)
        names = _QUOTED.findall(message or "")
        if names and text[a:b] not in names:
            # the compiler points at a call's '(' or an operator: prefer the quoted name nearby
            best = None
            for k in range(max(0, i - 3), min(len(view.t), i + 3)):
                t = view.t[k]
                if t.line == tok.line and t.type in (TT.IDENT, TT.SELF) and text[t.start:t.end] in names:
                    if best is None or abs(k - i) < abs(best - i) or (abs(k - i) == abs(best - i) and k < i):
                        best = k
            if best is not None:
                return view.span(best)
            if tok.type == TT.LPAREN and i > 0 and view.t[i - 1].type == TT.IDENT and i in view.match:
                return view.span(i - 1)[0], view.span(view.match[i])[1]
        return a, b

    def _span_at(self, text: str, view: _View, line: int, col: int, message: str,
                 pm: Optional[PositionMap] = None) -> Tuple[int, int]:
        """Range of the token the compiler reported at (line, col): the whole
        word, never empty."""
        if line < 1:
            line, col = 1, 1
        i = view.at(line, col)
        if i is not None:
            return self._token_range(text, view, i, message)
        pm = pm or PositionMap(text)
        off = pm.til_to_offset(line, col)
        starts = [t.start for t in view.t]
        k = bisect.bisect_right(starts, off) - 1
        if 0 <= k < len(view.t) and view.t[k].start <= off < view.t[k].end:
            return self._token_range(text, view, k, message)
        m = _WORD.match(text, off) if off < len(text) else None
        if m and m.end() > m.start():
            return m.start(), m.end()
        line_end = text.find('\n', off)
        line_end = len(text) if line_end < 0 else line_end
        if off < line_end:
            return off, off + 1
        return max(off - 1, 0), max(off, 1) if off > 0 else (0, min(1, len(text)))

    # ── tier 2: the whole compiler pipeline ─────────────────────────────────
    def _semantic(self, doc: Document) -> Semantic:
        sem = doc.semantic
        if sem is not None and sem.version == doc.version:
            return sem
        t0 = time.perf_counter()
        sem = Semantic()
        sem.version = doc.version
        try:
            self._run_pipeline(doc, sem)
        except Exception:
            if os.environ.get('TILLS_DEBUG'):
                raise
        sem.time = time.perf_counter() - t0
        doc.semantic = sem
        if sem.types is not None:
            doc.last_types = sem
        return sem

    def _add_diag(self, sem: Semantic, uri: str, d: dict):
        lst = sem.diags.setdefault(uri, [])
        key = (d['range']['start']['line'], d['range']['start']['character'], d.get('code'), d['message'])
        for e in lst:
            if (e['range']['start']['line'], e['range']['start']['character'], e.get('code'), e['message']) == key:
                return
        lst.append(d)

    def _make_diag(self, doc: Document, span: Tuple[int, int], raw: RawDiag, severity: Optional[int] = None) -> dict:
        text = localize(raw.text, self.lang)
        tip = localize(raw.tip, self.lang) if raw.tip else ""
        message = text + ("\n" + tip if tip else "")
        d = {'range': doc.pmap.range(span[0], max(span[1], span[0] + 1) if span[1] <= span[0] else span[1]),
             'severity': severity or raw.severity, 'source': 'til', 'message': message,
             'data': {'en': raw.text}}
        if raw.code:
            d['code'] = raw.code
        if tip:
            d['data']['hint'] = tip
            d['data']['enHint'] = raw.tip
        return d

    def _diag_in(self, doc: Document, raw: RawDiag, severity: Optional[int] = None) -> dict:
        syn = self._syntax(doc)
        span = self._span_at(doc.text, syn.view, raw.line, raw.col, raw.text, doc.pmap)
        return self._make_diag(doc, span, raw, severity)

    def _run_pipeline(self, doc: Document, sem: Semantic):
        syn = self._syntax(doc)
        uri = doc.uri
        sem.diags[uri] = []
        if syn.error is not None:
            span = syn.error_span or self._span_at(doc.text, syn.view, syn.error.line, syn.error.col,
                                                   syn.error.text, doc.pmap)
            self._add_diag(sem, uri, self._make_diag(doc, span, syn.error))
            return
        # a fresh Program over the parsed statements: resolve_imports merges into the
        # program it is given, and this analysis may run again (after a save of an import)
        parsed = syn.parsed.program
        program = til.Program(statements=list(syn.parsed.statements), energy=getattr(parsed, 'energy', None),
                              line=getattr(parsed, 'line', 0), col=getattr(parsed, 'col', 0))
        filename = self._filename(doc)
        own_nodes: Set[int] = set()
        for st in syn.parsed.statements:
            for n in til.walk(st):
                own_nodes.add(id(n))
        # ── imports: missing modules, syntax errors in imported files ──────
        imported, stop = self._check_imports(doc, syn, sem, filename)
        if stop:
            return
        # ── the compiler: resolve imports (one namespace), verify, compile, types
        with _Capture() as raws, _QuietStderr():
            try:
                program = til.resolve_imports(program, filename)
            except (SyntaxError, til.CompileError):
                for raw in raws:
                    self._report_raw_file(doc, sem, raw, imported)
                return
            except Exception:
                if os.environ.get('TILLS_DEBUG'):
                    raise
                return
        stmt_files = self._imported_statement_files(program, syn.parsed.statements, imported)
        node_file = _NodeFiles(own_nodes, stmt_files, program)
        # SemanticChecker: effects, capabilities, contracts, arity
        sem_errors = []
        with _Capture() as raws, _QuietStderr():
            try:
                _Checker(doc.text, filename).check(program)
            except Exception:
                if os.environ.get('TILLS_DEBUG'):
                    raise
        for raw in raws:
            sem_errors.append(raw)
            self._report_node_raw(doc, sem, raw, node_file, imported)
        # BytecodeCompiler: names, immutability, arity of natives (its first error)
        with _Capture() as raws, _QuietStderr():
            try:
                _Bytecode(doc.text, filename).compile_program(program)
            except til.CompileError as e:
                if not raws:
                    raws.append(RawDiag(filename, doc.text, 1, 1, str(e).split('\n')[0], 'E-VM'))
            except Exception:
                if os.environ.get('TILLS_DEBUG'):
                    raise
        for raw in raws:
            if any(s.node is raw.node and s.code == 'E-ARITY' for s in sem_errors if raw.node is not None):
                continue
            self._report_node_raw(doc, sem, raw, node_file, imported)
        # types: E-TYPE (tiltypes), or the classic checker's warnings without it
        tt = self.tt
        if tt is not None:
            with _Capture() as raws, _QuietStderr():
                ti = None
                try:
                    ti = tt.TypeInfer(program, doc.text, filename)
                    ti.run()
                except Exception:
                    if os.environ.get('TILLS_DEBUG'):
                        raise
                    ti = None
            if ti is not None:
                sem.types = ti
                sem.tt = tt
                suspects = list(getattr(ti, 'suspects', []) or [])
                for raw in raws:
                    if raw.node is None:
                        for c in suspects:
                            n = getattr(c, 'node', None)
                            if n is not None and (getattr(n, 'line', 0), getattr(n, 'col', 0) or 1) == \
                                    (raw.line, raw.col):
                                raw.node = n
                                if id(n) in own_nodes:
                                    break
                    self._report_node_raw(doc, sem, raw, node_file, imported)
        else:
            checker = _ClassicTypes()
            try:
                checker.check(program)
            except Exception:
                pass
            for text, node in checker.records:
                raw = RawDiag(filename, doc.text, getattr(node, 'line', 0) or 1, getattr(node, 'col', 0) or 1,
                              text, "", "", node, SEV_WARNING)
                self._report_node_raw(doc, sem, raw, node_file, imported)

    def _check_imports(self, doc: Document, syn: Syntax, sem: Semantic, filename: str):
        """Missing modules (warnings) and imported files that do not parse (errors
        reported on that file, plus a note on the import line). Returns the
        imported files {path key: (path, import info of this document)}."""
        imported: Dict[str, Tuple[str, ImportInfo]] = {}
        resolver = til.ModuleResolver()
        stop = False
        todo: List[Tuple[str, ImportInfo, str]] = []      # (module file, own import, from file)
        for info in syn.index.imports:
            path = self._resolve_module(resolver, info.module, filename)
            info.path = path
            if path is None:
                raw = RawDiag(filename, doc.text, 0, 0, f"Could not find module '{info.module}'", "E-IMPORT",
                              til.package_install_hint(info.module, filename, '.') or "", None, SEV_WARNING)
                self._add_diag(sem, doc.uri, self._make_diag(doc, (info.start, max(info.end, info.start + 1)),
                                                             raw))
                continue
            todo.append((path, info, filename))
        seen = {path_key(filename)}
        while todo:
            path, info, _from = todo.pop(0)
            key = path_key(path)
            if key in seen:
                continue
            seen.add(key)
            imported[key] = (path, info)
            mdoc = self._doc_for_path(path, prefer_open=False)
            if mdoc is None:
                continue
            msyn = self._syntax(mdoc)
            if msyn.error is not None:
                span = msyn.error_span or self._span_at(mdoc.text, msyn.view, msyn.error.line, msyn.error.col,
                                                       msyn.error.text, mdoc.pmap)
                self._add_diag(sem, mdoc.uri, self._make_diag(mdoc, span, msyn.error))
                self._note_on_import(doc, sem, info, msyn.error)
                stop = True
                continue
            for sub in msyn.index.imports:
                sp = self._resolve_module(resolver, sub.module, path)
                if sp is None:
                    raw = RawDiag(path, mdoc.text, 0, 0, f"Could not find module '{sub.module}'", "E-IMPORT",
                                  "", None, SEV_WARNING)
                    self._add_diag(sem, mdoc.uri, self._make_diag(mdoc, (sub.start, max(sub.end, sub.start + 1)),
                                                                  raw))
                    self._note_on_import(doc, sem, info, raw)
                    continue
                todo.append((sp, info, path))
        return imported, stop

    @staticmethod
    def _resolve_module(resolver, module: str, from_file: str) -> Optional[str]:
        try:
            with _QuietStderr():
                p = resolver.resolve(module, from_file)
        except Exception:
            return None
        return os.path.abspath(p) if p else None

    def _note_on_import(self, doc: Document, sem: Semantic, info: ImportInfo, raw: RawDiag):
        first = (raw.text or "").split('\n')[0]
        note = RawDiag(doc.path or "", doc.text, 0, 0,
                       f"problem in the imported module '{info.module}': {first}", raw.code, "", None, SEV_INFO)
        self._add_diag(sem, doc.uri, self._make_diag(doc, (info.start, max(info.end, info.start + 1)), note,
                                                     SEV_INFO))

    def _report_raw_file(self, doc: Document, sem: Semantic, raw: RawDiag, imported):
        """A diagnostic the compiler formatted against `raw.filename`."""
        fname = raw.filename
        own = path_key(self._filename(doc))
        key = path_key(os.path.abspath(fname)) if fname and not fname.startswith('<') else own
        if key == own or fname in ('<attr>', '<fstring>'):
            self._add_diag(sem, doc.uri, self._diag_in(doc, raw))
            return
        mdoc = self._doc_for_path(os.path.abspath(fname), prefer_open=False)
        if mdoc is None:
            return
        self._add_diag(sem, mdoc.uri, self._diag_in(mdoc, raw))
        entry = imported.get(key)
        if entry is not None:
            self._note_on_import(doc, sem, entry[1], raw)
        elif self._syntax(doc).index.imports:
            self._note_on_import(doc, sem, self._syntax(doc).index.imports[0], raw)

    def _report_node_raw(self, doc: Document, sem: Semantic, raw: RawDiag, node_file: '_NodeFiles', imported):
        path = node_file.file_of(raw.node) if raw.node is not None else None
        if path is None:
            self._add_diag(sem, doc.uri, self._diag_in(doc, raw, raw.severity))
            return
        mdoc = self._doc_for_path(path, prefer_open=False)
        if mdoc is None:
            return
        self._add_diag(sem, mdoc.uri, self._diag_in(mdoc, raw, raw.severity))
        entry = imported.get(path_key(path))
        if entry is not None:
            self._note_on_import(doc, sem, entry[1], raw)

    def _imported_statement_files(self, program, own_statements, imported) -> Dict[int, str]:
        """id(imported top-level statement) -> its file (matched by kind, name and
        position against each imported file's own parse)."""
        own = {id(s) for s in own_statements}
        sigs: Dict[Tuple[str, str, int, int], str] = {}
        for key, (path, _info) in imported.items():
            mdoc = self._doc_for_path(path, prefer_open=False)
            if mdoc is None:
                continue
            msyn = self._syntax(mdoc)
            if msyn.parsed is None:
                continue
            for st in msyn.parsed.statements:
                sigs.setdefault(_stmt_sig(st), path)
        out = {}
        for st in program.statements:
            if id(st) in own:
                continue
            p = sigs.get(_stmt_sig(st))
            if p is not None:
                out[id(st)] = p
        return out

    # ── public: diagnostics ────────────────────────────────────────────────
    def diagnostics(self, uri) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        return list(self._semantic(doc).diags.get(doc.uri, []))

    def diagnostics_by_uri(self, uri) -> Dict[str, List[dict]]:
        """Diagnostics of an analysis of `uri`, by file: the document itself and
        imported files with problems (the language server publishes them all)."""
        doc = self._doc(uri)
        if doc is None:
            return {}
        sem = self._semantic(doc)
        out = {k: list(v) for k, v in sem.diags.items()}
        out.setdefault(doc.uri, [])
        return out

    # ── cross-file: the files that share a namespace ────────────────────────
    def _workspace_paths(self) -> List[str]:
        now = time.monotonic()
        stamp, paths = self._files_cache
        if paths and now - stamp < 2.0:
            return paths
        out: List[str] = []
        if self.root and os.path.isdir(self.root):
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith('.'))
                for fn in sorted(filenames):
                    if fn.endswith('.til'):
                        out.append(os.path.join(dirpath, fn))
                        if len(out) >= MAX_WORKSPACE_FILES:
                            break
                if len(out) >= MAX_WORKSPACE_FILES:
                    break
        for d in self.docs.values():
            if d.path and d.path not in out:
                out.append(d.path)
        self._files_cache = (now, out)
        return out

    def _imports_of(self, path: str) -> List[str]:
        doc = self._doc_for_path(path)
        if doc is None:
            return []
        syn = self._syntax(doc)
        cached = getattr(syn, '_import_paths', None)
        if cached is not None:
            return cached
        resolver = til.ModuleResolver()
        out = []
        for info in syn.index.imports:
            p = self._resolve_module(resolver, info.module, path)
            if p:
                out.append(p)
        syn._import_paths = out
        return out

    def _cached(self, kind: str, path: str, compute):
        key = (kind, path_key(path))
        hit = self._graph_cache.get(key)
        if hit is not None and hit[0] == self._gen and (kind != 'group' or hit[2] == self._files_cache[0]):
            return hit[1]
        value = compute(path)
        self._graph_cache[key] = (self._gen, value, self._files_cache[0])
        return value

    def _closure(self, path: str) -> List[str]:
        """Files a file imports, directly or not (BFS order, without itself)."""
        return self._cached('closure', path, self._compute_closure)

    def _compute_closure(self, path: str) -> List[str]:
        seen = {path_key(path)}
        out = []
        todo = list(self._imports_of(path))
        while todo:
            p = todo.pop(0)
            k = path_key(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
            todo.extend(self._imports_of(p))
        return out

    def _group(self, path: str) -> List[str]:
        """The files that can be merged into one program with `path` (they import
        each other, directly or through other files): one namespace."""
        self._workspace_paths()
        return self._cached('group', path, self._compute_group)

    def _compute_group(self, path: str) -> List[str]:
        paths = self._workspace_paths()
        adj: Dict[str, Set[str]] = {}
        names: Dict[str, str] = {}
        for p in paths + [path]:
            k = path_key(p)
            names[k] = p
            for q in self._imports_of(p):
                qk = path_key(q)
                names.setdefault(qk, q)
                adj.setdefault(k, set()).add(qk)
                adj.setdefault(qk, set()).add(k)
        start = path_key(path)
        seen = {start}
        todo = [start]
        while todo:
            k = todo.pop()
            for n in adj.get(k, ()):
                if n not in seen:
                    seen.add(n)
                    todo.append(n)
        return [names[k] for k in seen if k in names]

    def _in_workspace(self, path: Optional[str]) -> bool:
        if not path:
            return True                     # an untitled document
        if self.root is None:
            return True
        try:
            rel = os.path.relpath(os.path.realpath(path), os.path.realpath(self.root))
        except ValueError:
            return False
        if rel == os.pardir or rel.startswith(os.pardir + os.sep):
            return any(d.path and path_key(d.path) == path_key(path) for d in self.docs.values())
        return til.PACKAGE_DIR not in rel.split(os.sep)

    # ── resolution of a reference ───────────────────────────────────────────
    def _resolve_free(self, doc: Document, name: str) -> Optional[Sym]:
        """A top-level name a file uses but does not define: the files it imports
        (nearest first), then the files that import it."""
        syn = self._syntax(doc)
        s = syn.index.toplevel.get(name)
        if s is not None and s.kind != 'module':
            return s
        if not doc.path:
            return None
        for p in self._closure(doc.path):
            d = self._doc_for_path(p)
            if d is None:
                continue
            s = self._syntax(d).index.toplevel.get(name)
            if s is not None and s.kind != 'module':
                return s
        found = []
        for p in ([] if self._cheap else self._group(doc.path)):
            if path_key(p) == path_key(doc.path):
                continue
            d = self._doc_for_path(p)
            if d is None:
                continue
            s = self._syntax(d).index.toplevel.get(name)
            if s is not None and s.kind != 'module':
                found.append(s)
        return found[0] if len(found) == 1 else None

    def _related_docs(self, doc: Document) -> List[Document]:
        """`doc` and the files sharing its namespace. Completion / signature help
        (cheap mode) look only at the files it imports: finding the files that
        import it would read the whole workspace."""
        out = [doc]
        if not doc.path:
            return out
        paths = self._closure(doc.path) if self._cheap else self._group(doc.path)
        for p in paths:
            if path_key(p) != path_key(doc.path):
                d = self._doc_for_path(p)
                if d is not None:
                    out.append(d)
        return out

    def _members_named(self, doc: Document, type_name: Optional[str], member: str) -> List[Sym]:
        """Members `member` of a type (fields, methods, variants), in every file of
        the namespace (impl blocks may live in other files)."""
        files = self._related_docs(doc)
        out = []
        for d in files:
            ix = self._syntax(d).index
            if type_name is not None:
                s = ix.members.get((type_name, member))
                if s is not None and s not in out:
                    out.append(s)
            else:
                for (_c, m), s in ix.members.items():
                    if m == member and s.kind in ('field', 'method') and s not in out:
                        out.append(s)
        return out

    def _target(self, doc: Document, ref: Ref):
        """What a reference names: a Sym, ('module', path), ('builtin', kind, name) or None."""
        if ref.sym is not None:
            s = ref.sym
            if s.kind == 'module' and s.node is not None:
                return ('module', self._module_path(doc, s.node.module), s.node.module)
            if ref.guess and doc.path:
                cands = self._members_named(doc, None, ref.name)
                if len(cands) > 1:
                    return None
            return s
        if ref.builtin:
            return ('builtin', ref.builtin, ref.name, ref.recv)
        if ref.kind == 'module':
            return ('module', self._module_path(doc, ref.recv or ref.name), ref.recv or ref.name)
        if ref.kind == 'module-member':
            p = self._module_path(doc, ref.recv or "")
            if p:
                d = self._doc_for_path(p)
                if d is not None:
                    s = self._syntax(d).index.toplevel.get(ref.name)
                    if s is not None:
                        return s
                    return self._resolve_free(d, ref.name)
            return None
        if ref.kind in ('name', 'type', 'effect'):
            s = self._resolve_free(doc, ref.name)
            if ref.kind == 'effect' and s is not None and s.kind != 'function':
                return None
            return s
        if ref.kind == 'member':
            if ref.recv:
                owner = self._resolve_free(doc, ref.recv)
                tname = owner.name if owner is not None else ref.recv
                cands = self._members_named(doc, tname, ref.name)
                if cands:
                    return cands[0]
                if owner is not None and owner.kind == 'module':
                    return None
                return None
            cands = self._members_named(doc, None, ref.name)
            if len(cands) == 1:
                return cands[0]
            if not cands and ref.name in METHOD_DOCS:
                return ('builtin', 'method', ref.name, None)
        return None

    def _module_path(self, doc: Document, module: str) -> Optional[str]:
        if not module:
            return None
        return self._resolve_module(til.ModuleResolver(), module, self._filename(doc))

    def _ref_at(self, doc: Document, line: int, character: int) -> Tuple[Syntax, Optional[Ref], int]:
        syn = self._syntax(doc)
        off = doc.pmap.offset(line, character)
        ref = syn.index.ref_at(off)
        if ref is not None and not (ref.start <= off <= ref.end):
            ref = None
        return syn, ref, off

    def _location(self, sym: Sym) -> Optional[dict]:
        doc = self._doc(sym.uri)
        if doc is None:
            return None
        return {'uri': doc.uri, 'range': doc.pmap.range(sym.start, sym.end)}

    # ── public: definition / references / highlights ────────────────────────
    def definition(self, uri, line, character) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        _syn, ref, _off = self._ref_at(doc, line, character)
        if ref is None:
            return []
        t = self._target(doc, ref)
        if isinstance(t, Sym):
            if t.implicit and t.kind != 'module':
                if t.name == 'self' and t.owner is not None:
                    loc = self._location(t)
                    return [loc] if loc else []
                return []
            loc = self._location(t)
            return [loc] if loc else []
        if isinstance(t, tuple) and t[0] == 'module' and t[1]:
            return [{'uri': self._uri_for_path(t[1]), 'range': {'start': {'line': 0, 'character': 0},
                                                              'end': {'line': 0, 'character': 0}}}]
        return []

    def _uri_for_path(self, path: str) -> str:
        key = path_key(path)
        for d in self.docs.values():
            if d.path and path_key(d.path) == key:
                return d.uri
        return path_to_uri(path)

    def _all_refs(self, doc: Document, target: Sym) -> List[Tuple[Document, Ref]]:
        """Every occurrence of `target` (definition included), across files."""
        tdoc = self._doc(target.uri) or doc
        local = target.kind in ('local', 'param', 'typeparam') or target.scope is not None and \
            target.scope.kind != 'module' and target.kind not in ('method', 'field', 'variant')
        files: List[Document] = [tdoc]
        if not local and tdoc.path:
            for p in self._group(tdoc.path):
                if path_key(p) == path_key(tdoc.path):
                    continue
                d = self._doc_for_path(p)
                if d is not None:
                    files.append(d)
        out = []
        for d in files:
            ix = self._syntax(d).index
            for r in ix.refs:
                if r.name != target.name:
                    continue
                if r.sym is target:
                    out.append((d, r))
                    continue
                if local:
                    continue
                if r.sym is not None and r.sym is not target and not r.guess:
                    continue
                t = self._target(d, r)
                if t is target:
                    out.append((d, r))
        return out

    def references(self, uri, line, character, include_declaration=True) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        _syn, ref, _off = self._ref_at(doc, line, character)
        if ref is None:
            return []
        t = self._target(doc, ref)
        if not isinstance(t, Sym):
            return []
        out = []
        seen = set()
        for d, r in self._all_refs(doc, t):
            if r.decl and not include_declaration:
                continue
            key = (d.uri, r.start, r.end)
            if key in seen:
                continue
            seen.add(key)
            out.append({'uri': d.uri, 'range': d.pmap.range(r.start, r.end)})
        return out

    def document_highlights(self, uri, line, character) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        _syn, ref, _off = self._ref_at(doc, line, character)
        if ref is None:
            return []
        t = self._target(doc, ref)
        if not isinstance(t, Sym):
            return []
        out = []
        for d, r in self._all_refs(doc, t):
            if d.uri != doc.uri:
                continue
            out.append({'range': doc.pmap.range(r.start, r.end), 'kind': 3 if (r.write or r.decl) else 2})
        return out

    # ── public: rename ──────────────────────────────────────────────────────
    def _rename_target(self, doc: Document, line: int, character: int):
        syn, ref, off = self._ref_at(doc, line, character)
        if ref is None:
            word = self._word_at(doc, off)
            if word and word[2] in KEYWORD_WORDS:
                raise RenameError(localize(f"cannot rename '{word[2]}': it is a keyword", self.lang))
            return None, None
        t = self._target(doc, ref)
        name = doc.text[ref.start:ref.end]
        if isinstance(t, tuple) or t is None:
            if isinstance(t, tuple) and t[0] == 'module':
                raise RenameError(localize(f"cannot rename the module '{name}' (rename its file instead)",
                                           self.lang))
            if isinstance(t, tuple) or ref.builtin or name in BUILTIN_NAMES:
                raise RenameError(localize(f"cannot rename '{name}': it is a builtin", self.lang))
            raise RenameError(localize(f"cannot rename '{name}': its definition is not known", self.lang))
        if t.implicit:
            raise RenameError(localize(f"cannot rename '{name}': it is a builtin", self.lang))
        if t.kind == 'function' and t.name == 'main' and t.scope is not None and t.scope.kind == 'module':
            raise RenameError(localize(f"cannot rename '{name}': it is the entry point of the program", self.lang))
        if not self._in_workspace(self._doc(t.uri).path if self._doc(t.uri) else None):
            raise RenameError(localize(f"cannot rename '{name}': it is defined outside the workspace", self.lang))
        if name != t.name:
            # a localized alias of the entry point (негізгі, главная): keep it
            raise RenameError(localize(f"cannot rename '{name}': it is a builtin", self.lang))
        return ref, t

    def _word_at(self, doc: Document, off: int) -> Optional[Tuple[int, int, str]]:
        text = doc.text
        a = off
        while a > 0 and (text[a - 1].isalnum() or text[a - 1] == '_'):
            a -= 1
        b = off
        while b < len(text) and (text[b].isalnum() or text[b] == '_'):
            b += 1
        if b > a:
            return a, b, text[a:b]
        return None

    def prepare_rename(self, uri, line, character) -> Optional[dict]:
        doc = self._doc(uri)
        if doc is None:
            return None
        ref, t = self._rename_target(doc, line, character)
        if ref is None:
            return None
        return {'range': doc.pmap.range(ref.start, ref.end), 'placeholder': doc.text[ref.start:ref.end]}

    def rename(self, uri, line, character, new_name) -> Optional[dict]:
        doc = self._doc(uri)
        if doc is None:
            return None
        ref, target = self._rename_target(doc, line, character)
        if ref is None:
            return None
        new_name = new_name.strip() if isinstance(new_name, str) else ''
        if not re.fullmatch(r'[^\W\d]\w*', new_name):
            raise RenameError(localize(f"'{new_name}' is not a valid name", self.lang))
        if new_name == target.name:
            return {'changes': {}}
        if new_name in KEYWORD_WORDS or new_name in ('self', 'None', 'main'):
            raise RenameError(localize(f"'{new_name}' is a keyword", self.lang))
        if new_name in BUILTIN_NAMES or new_name in ALIAS_WORDS or new_name in BUILTIN_TYPE_NAMES or \
                (target.kind in ('method', 'field') and new_name in METHOD_DOCS):
            raise RenameError(localize(f"'{new_name}' is the name of a builtin", self.lang))
        occurrences = self._all_refs(doc, target)
        self._check_clash(doc, target, new_name, occurrences)
        changes: Dict[str, List[dict]] = {}
        seen = set()
        for d, r in occurrences:
            if not self._in_workspace(d.path):
                raise RenameError(localize(f"cannot rename '{target.name}': it is used outside the workspace",
                                           self.lang))
            key = (d.uri, r.start)
            if key in seen:
                continue
            seen.add(key)
            changes.setdefault(d.uri, []).append({'range': d.pmap.range(r.start, r.end), 'newText': new_name})
        return {'changes': changes}

    def _check_clash(self, doc: Document, target: Sym, new_name: str, occurrences):
        what = None
        if target.kind in ('method', 'field', 'variant'):
            cont = target.container.name if target.container is not None else target.container_name
            if self._members_named(doc, cont, new_name):
                what = cont
        elif target.scope is not None and target.scope.kind == 'module':
            tdoc = self._doc(target.uri) or doc
            files = [tdoc]
            if tdoc.path:
                files += [d for d in (self._doc_for_path(p) for p in self._group(tdoc.path)) if d is not None]
            for d in files:
                s = self._syntax(d).index.toplevel.get(new_name)
                if s is not None:
                    what = new_name
                    break
        if what is None:
            # a reference that would then mean something else, or another name captured
            for d, r in occurrences:
                if r.scope is None:
                    continue
                other = r.scope.lookup(new_name, r.start)
                # `new_name` is declared at least as close to this reference as the target
                if other is not None and other is not target and other.scope is not None and \
                        target.scope is not None and _scope_depth(other.scope) >= _scope_depth(target.scope):
                    what = new_name
                    break
            if what is None and target.scope is not None and target.scope.kind != 'module':
                ix = self._syntax(self._doc(target.uri) or doc).index
                for r in ix.refs:
                    if r.name != new_name or r.scope is None or r.sym is target:
                        continue
                    if _scope_within(r.scope, target.scope) and r.start >= target.start:
                        what = new_name
                        break
        if what is not None:
            raise RenameError(localize(f"'{new_name}' is already defined: renaming '{target.name}' would change "
                                       f"what a name refers to", self.lang))

    # ── types from tiltypes (tier 2) ───────────────────────────────────────
    def _instance(self, sem: Semantic, key: Optional[str]):
        if sem.types is None or not key:
            return None
        if key in sem.instances:
            return sem.instances[key]
        ti = sem.types
        inst = None
        try:
            if key in ti.funcs:
                info = ti.funcs[key]
                params = ti.scheme_params(key) if info.scheme is not None else []
                with _QuietStderr():
                    inst = ti.specialize(key, params)
        except Exception:
            if os.environ.get('TILLS_DEBUG'):
                raise
            inst = None
        sem.instances[key] = inst
        return inst

    def _types_for(self, doc: Document) -> Optional[Semantic]:
        """This version's analysis with types (computed when needed, except in
        completion / signature help, which must stay fast: then only a cached
        one), else None (the callers fall back to the last version's types)."""
        if self.tt is None:
            return None
        sem = doc.semantic
        if sem is not None and sem.version == doc.version:
            return sem if sem.types is not None else None
        if self._cheap or not self._syntax(doc).ok:
            return None
        sem = self._semantic(doc)
        return sem if sem.types is not None else None

    def _function_types(self, doc: Document) -> Optional[Semantic]:
        """Types for function signatures: this version's, else the last ones."""
        sem = self._types_for(doc)
        if sem is None and doc.last_types is not None and doc.last_types.types is not None:
            return doc.last_types
        return sem

    def _inferred(self, doc: Document, sym: Sym):
        """(tiltypes term, show text) of a symbol, or (None, None)."""
        sem = self._types_for(doc)
        if sem is None:
            t = self._stale_inferred(doc, sym)
            if t is None or self.tt is None:
                return None, None
            tt = self.tt
            try:
                t = tt.deep(t, tt.UNK)
                text = tt.show(t)
            except Exception:
                return None, None
            return (t, None) if text in ('unknown', 'dynamic', 'None') or '?' in text else (t, text)
        tt = sem.tt
        ti = sem.types
        t = None
        try:
            if sym.kind in ('local', 'param', 'const') and sym.scope is not None and sym.scope.kind != 'module':
                key = sym.owner.key if sym.owner is not None else 'main'
                inst = self._instance(sem, key)
                if inst is None:
                    return None, None
                if sym.kind == 'param' and sym.owner is not None and sym.owner.node is not None:
                    names = [p.name for p in sym.owner.node.params]
                    if sym.name in names:
                        t = inst.params[names.index(sym.name)]
                elif isinstance(sym.node, (til.VarDecl, til.For)):
                    t = inst.decl(sym.node)
                elif sym.node is not None:
                    t = inst.t(sym.node)
            elif sym.kind in ('global', 'const'):
                g = ti.globals.get(sym.name)
                if g is not None:
                    t = g.frozen if g.frozen is not None else g.type
            elif sym.kind == 'field' and sym.container is not None:
                info = ti.structs.get(sym.container.name)
                if info is not None:
                    t = info.frozen.get(sym.name) or info.ann.get(sym.name)
        except Exception:
            return None, None
        if t is None:
            return None, None
        t = tt.deep(t, tt.UNK)
        text = tt.show(t)
        if text in ('unknown', 'dynamic', 'None') or '?' in text:
            return t, None
        return t, text

    def _tt_desc(self, t, depth: int = 0):
        tt = self.tt
        if tt is None or t is None or depth > 8:
            return None
        t = tt.resolve(t)
        if not isinstance(t, tt.TCon):
            return None
        n = t.name
        if n in ('int', 'float', 'str', 'bool'):
            return (n,)
        if n == 'array':
            return ('array', self._tt_desc(t.args[0], depth + 1) if t.args else None)
        if n == 'map':
            return ('map', self._tt_desc(t.args[0], depth + 1), self._tt_desc(t.args[1], depth + 1))
        if n == 'option':
            return ('option', self._tt_desc(t.args[0], depth + 1))
        if n == 'result':
            return ('result', self._tt_desc(t.args[0], depth + 1), self._tt_desc(t.args[1], depth + 1))
        if n.startswith('S:'):
            return ('struct', n[2:])
        if n.startswith('E:'):
            return ('enum', n[2:])
        return None

    def _sym_desc(self, doc: Document, sym: Optional[Sym], depth: int = 0):
        """Syntactic type of a symbol: written, else from its value (across
        files), else inferred by tiltypes (this version, or the last one that
        had types when the file is broken now)."""
        if sym is None or depth > 12:
            return None
        sdoc = self._doc(sym.uri) or doc
        d = sym.typ
        if d is None and isinstance(sym.node, til.VarDecl) and sym.node.value is not None:
            d = self._node_desc(sdoc, sym.node.value, depth + 1)
        if d is None:
            t, _text = self._inferred(sdoc, sym)
            d = self._tt_desc(t)
        if d is None:
            d = self._tt_desc(self._stale_inferred(sdoc, sym))
        return self._resolve_named_desc(sdoc, d)

    def _resolve_named_desc(self, doc: Document, d, depth: int = 0):
        if d is None or d[0] != 'struct' or depth > 8:
            return d
        ix = self._syntax(doc).index
        s = ix.toplevel.get(d[1]) or self._resolve_free(doc, d[1])
        if s is not None and s.kind == 'enum':
            return ('enum', d[1])
        if s is not None and s.kind == 'alias' and s.node is not None:
            return self._resolve_named_desc(self._doc(s.uri) or doc, _desc(s.node.type), depth + 1)
        return d

    def _node_desc(self, doc: Document, node, depth: int = 0):
        """Syntactic type of an expression of `doc`'s current parse, resolving
        names across files (a struct, a function or a method of another file)."""
        if node is None or depth > 12:
            return None
        ix = self._syntax(doc).index
        t = type(node)
        if t in (til.StringLit, til.FStringLit):
            return ('str',)
        if t in (til.IntLit, til.CharLit):
            return ('int',)
        if t is til.FloatLit:
            return ('float',)
        if t is til.BoolLit:
            return ('bool',)
        if t in (til.ArrayLit, til.ListComprehension, til.Range):
            return ('array', ('int',) if t is til.Range else None)
        if t is til.DictLit:
            return ('map', None, None)
        if t is til.StructInit:
            return self._resolve_named_desc(doc, ('struct', node.name))
        if t is til.Cast:
            return _desc(node.target_type)
        if t is til.Call:
            f = node.func
            r = ix.node_refs.get(id(f))
            tgt = self._target(doc, r) if r is not None else None
            if isinstance(tgt, Sym):
                if tgt.kind in ('function', 'method') and isinstance(tgt.node, til.FuncDef):
                    rt = tgt.node.ret_type
                    if not isinstance(rt, til.VoidType):
                        return self._resolve_named_desc(self._doc(tgt.uri) or doc, _desc(rt))
                    return None
                if tgt.kind == 'struct':
                    return ('struct', tgt.name)
                if tgt.kind == 'variant' and tgt.container is not None:
                    return ('enum', tgt.container.name)
                if tgt.kind in ('local', 'param', 'global', 'const'):
                    fd = self._sym_desc(doc, tgt, depth + 1)
                    return fd[1] if fd is not None and fd[0] == 'fn' and len(fd) > 1 else None
                return None
            if isinstance(tgt, tuple) and tgt[0] == 'builtin' and tgt[1] == 'function':
                if tgt[2] == 'Some':
                    return ('option', self._node_desc(doc, node.args[0], depth + 1) if node.args else None)
                if tgt[2] in ('Ok', 'Err'):
                    return ('result', None, None)
                return _BUILTIN_RESULT.get(tgt[2])
            if isinstance(f, til.Attribute):
                recv = self._node_desc(doc, f.obj, depth + 1)
                return _method_result(recv, f.attr)
            return None
        if t is til.Identifier:
            r = ix.node_refs.get(id(node))
            tgt = self._target(doc, r) if r is not None else None
            if isinstance(tgt, Sym) and tgt.kind in ('local', 'param', 'global', 'const'):
                return self._sym_desc(doc, tgt, depth + 1)
            return None
        if t is til.Attribute:
            r = ix.node_refs.get(id(node))
            tgt = self._target(doc, r) if r is not None else None
            if isinstance(tgt, Sym):
                if tgt.kind == 'field':
                    return self._resolve_named_desc(self._doc(tgt.uri) or doc, tgt.typ)
                if tgt.kind == 'variant' and tgt.container is not None:
                    return ('enum', tgt.container.name)
            return None
        if t is til.Index:
            obj = self._node_desc(doc, node.obj, depth + 1)
            if obj is not None and obj[0] == 'array':
                return obj[1] if len(obj) > 1 else None
            if obj is not None and obj[0] == 'map':
                return obj[2] if len(obj) > 2 else None
            return ('str',) if obj == ('str',) else None
        if t is til.NullCheck:
            inner = self._node_desc(doc, node.expr, depth + 1)
            return inner[1] if inner is not None and inner[0] in ('option', 'result') and len(inner) > 1 else None
        if t is til.BinaryOp:
            if node.op in ('==', '!=', '<', '>', '<=', '>=', 'and', 'or'):
                return ('bool',)
            a = self._node_desc(doc, node.left, depth + 1)
            b = self._node_desc(doc, node.right, depth + 1)
            if node.op == '+' and ('str',) in (a, b):
                return ('str',)
            if ('float',) in (a, b):
                return ('float',)
            return ('int',) if a == ('int',) and b == ('int',) else None
        if t is til.IfExpr:
            return self._node_desc(doc, node.then_expr, depth + 1)
        return None

    def _stale_inferred(self, doc: Document, sym: Sym):
        """The type tiltypes gave a variable of the same name in the same
        function in the last analysis that had types (the file is broken now)."""
        sem = doc.last_types
        if sem is None or sem.types is None or sym.kind not in ('local', 'param', 'global', 'const'):
            return None
        try:
            if sym.scope is not None and sym.scope.kind == 'module':
                g = sem.types.globals.get(sym.name)
                return g.frozen if g is not None else None
            key = sym.owner.key if sym.owner is not None else 'main'
            inst = self._instance(sem, key)
            if inst is None or inst.info is None:
                return None
            if sym.kind == 'param':
                names = [p.name for p in inst.info.fdef.params]
                return inst.params[names.index(sym.name)] if sym.name in names else None
            found = None
            for n in til.walk(inst.info.fdef.body):
                if isinstance(n, til.VarDecl) and n.name == sym.name or isinstance(n, til.For) and n.var == sym.name:
                    found = inst.decl(n)
            return found
        except Exception:
            return None

    # ── signatures ──────────────────────────────────────────────────────────
    def _signature(self, doc: Document, sym: Sym) -> str:
        fn = sym.node
        if fn is None or not isinstance(fn, til.FuncDef):
            return sym.qualname
        sem = self._function_types(doc)
        info = None
        if sem is not None and sym.key and sym.key in sem.types.funcs:
            info = sem.types.funcs[sym.key]
        tt = self.tt
        parts = []
        for k, p in enumerate(fn.params):
            if p.name == 'self':
                parts.append(('mut ' if p.mutable else '') + 'self')
                continue
            t = _type_text(p.type)
            if not t and info is not None and info.scheme is not None:
                try:
                    term = tt.deep(info.scheme.sig.params[k], None)
                    if not tt.contains(term, ('var',)):
                        s = tt.show(term)
                        if s not in ('unknown', 'dynamic'):
                            t = s
                except Exception:
                    pass
            item = f"{p.name}: {t}" if t else p.name
            psym = next((s for s in sym.params if s.name == p.name), None)
            default = getattr(psym, 'default_text', None) if psym is not None else None
            if default:
                item += f" = {default}"
            parts.append(item)
        ret = ''
        if not isinstance(fn.ret_type, til.VoidType):
            ret = _type_text(fn.ret_type)
        elif info is not None and info.scheme is not None and fn.name != 'main':
            try:
                term = tt.deep(info.scheme.sig.ret, None)
                if not tt.contains(term, ('var',)):
                    s = tt.show(term)
                    if s not in ('unknown', 'dynamic', 'None'):
                        ret = s
            except Exception:
                pass
        head = sym.qualname
        tparams = [s.name for s in (sym.index.syms if sym.index else []) if s.kind == 'typeparam' and
                   s.owner is sym and not s.implicit] if sym.index is not None else []
        if tparams:
            head += '<' + ', '.join(tparams) + '>'
        return f"{head}({', '.join(parts)})" + (f" -> {ret}" if ret else "")

    def _sym_detail(self, doc: Document, sym: Sym) -> str:
        k = sym.kind
        if k in ('function', 'method'):
            return self._signature(doc, sym)
        if k in ('struct', 'enum', 'trait', 'capability', 'module', 'typeparam'):
            word = {'struct': 'struct', 'enum': 'enum', 'trait': 'trait', 'capability': 'cap',
                    'module': 'import', 'typeparam': 'type'}[k]
            return f"{word} {sym.name}"
        if k == 'alias':
            return f"type {sym.name} = {_type_text(sym.node.type) if sym.node is not None else '?'}"
        if k == 'variant':
            node = sym.node
            fields = ', '.join(_type_text(t) for t in getattr(node, 'fields', []) or [])
            cont = sym.container.name if sym.container is not None else ''
            return f"{cont}.{sym.name}" + (f"({fields})" if fields else "")
        if k == 'field':
            cont = sym.container.name if sym.container is not None else ''
            t = _type_text(sym.node.type) if sym.node is not None else ''
            return f"{cont}.{sym.name}: {t}" if t else f"{cont}.{sym.name}"
        # variables
        declared = None
        node = sym.node
        if isinstance(node, til.VarDecl) and node.type_ann is not None:
            declared = _type_text(node.type_ann)
        elif isinstance(node, til.FuncParam) and node.type is not None:
            declared = _type_text(node.type)
        if not declared:
            _t, declared = self._inferred(doc, sym)
        if not declared and sym.typ is not None:
            s = show_desc(sym.typ)
            declared = s if '?' not in s else None
        word = ''
        if k in ('local', 'global', 'const'):
            if isinstance(node, til.VarDecl):
                word = 'const ' if node.is_const else ('var ' if node.mutable else 'let ')
        return f"{word}{sym.name}" + (f": {declared}" if declared else "")

    def _sym_markdown(self, doc: Document, sym: Sym) -> str:
        lang = self.lang
        detail = self._sym_detail(doc, sym)
        kind_label = label(sym.kind, lang)
        lines = ["```til", detail, "```", f"*{kind_label}*"]
        sdoc = self._doc(sym.uri)
        if sdoc is not None and sdoc.uri != doc.uri and sdoc.path:
            lines[-1] += f" · `{os.path.basename(sdoc.path)}`"
        if sym.doc:
            lines += ["", sym.doc]
        fn = sym.node if isinstance(sym.node, til.FuncDef) else None
        if fn is not None:
            extra = []
            for text in fn.requires_text or []:
                extra.append(f"**{label('requires', lang)}** `{text}`")
            for text in fn.ensures_text or []:
                extra.append(f"**{label('ensures', lang)}** `{text}`")
            if fn.declared_effects is not None:
                effs = ', '.join(fn.declared_effects) or ('pure' if 'pure' in (fn.effects or []) else '—')
                extra.append(f"**{label('effects', lang)}** {effs}")
            if fn.caps:
                extra.append(f"**{label('capabilities', lang)}** {', '.join(fn.caps)}")
            if extra:
                lines += [""] + ["  \n".join(extra)]
        return "\n".join(lines)

    # ── hover ───────────────────────────────────────────────────────────────
    def _keyword_markdown(self, name: str, word: str) -> str:
        lang = self.lang
        spellings = []
        for code in KEYWORD_LANGS:
            if name == 'ELIF' and code != 'en':
                spellings.append(f"`{keyword_spelling('ELIF', code)}`")
                continue
            for w in KEYWORDS_BY_TYPE.get(name, {}).get(code, [])[:1]:
                spellings.append(f"`{w}`")
        row = KEYWORD_DOCS.get(name)
        text = doc_text(row, lang, 0) if row else ""
        out = [f"**{word}** — *{label('keyword', lang)}*", " = ".join(dict.fromkeys(spellings))]
        if text:
            out += ["", text]
        return "\n".join(out)

    def _builtin_markdown(self, name: str, shown: Optional[str] = None) -> str:
        lang = self.lang
        row = BUILTIN_DOCS.get(name)
        sig = self._builtin_signature(name)
        out = ["```til", sig, "```", f"*{label('builtin', lang)}*"]
        aliases = [a for a, (c, _l) in ALIAS_WORDS.items() if c == name]
        if aliases or (shown and shown != name):
            out[-1] += " · " + " = ".join(f"`{w}`" for w in dict.fromkeys([name] + aliases))
        if row:
            out += ["", doc_text(row, lang, 2)]
        return "\n".join(out)

    @staticmethod
    def _builtin_signature(name: str) -> str:
        row = BUILTIN_DOCS.get(name)
        if row is None:
            return f"{name}(...)"
        params = []
        for p in row[0]:
            if p == '...':
                if params:
                    params[-1] += '...'
                continue
            params.append(p[1:] + '?' if p.startswith('?') else p)
        ret = f" -> {row[1]}" if row[1] else ""
        return f"{name}({', '.join(params)}){ret}"

    def _method_markdown(self, name: str) -> str:
        lang = self.lang
        row = METHOD_DOCS.get(name)
        if row is None:
            return ""
        recv = {'s': 'str', 'a': 'array', 'm': 'map', 'o': 'Option / Result'}
        on = ', '.join(recv[c] for c in row[0] if c in recv)
        params = ', '.join(p[1:] + '?' if p.startswith('?') else p for p in row[1])
        ret = f" -> {row[2]}" if row[2] else ""
        return "\n".join(["```til", f".{name}({params}){ret}", "```",
                          f"*{label('builtin_method', lang)}* · {on}", "", doc_text(row, lang, 3)])

    def hover(self, uri, line, character) -> Optional[dict]:
        doc = self._doc(uri)
        if doc is None:
            return None
        syn = self._syntax(doc)
        off = doc.pmap.offset(line, character)
        view = syn.view
        starts = [t.start for t in view.t]
        k = bisect.bisect_right(starts, off) - 1
        tok = view.t[k] if 0 <= k < len(view.t) and view.t[k].start <= off < max(view.t[k].end, view.t[k].start + 1) \
            else None
        ref = syn.index.ref_at(off)
        if ref is not None and ref.start <= off <= ref.end:
            md = self._ref_markdown(doc, ref)
            if md:
                return {'contents': {'kind': 'markdown', 'value': md}, 'range': doc.pmap.range(ref.start, ref.end)}
            if not syn.ok and ref.kind == 'name' and ref.sym is None:
                # broken code: the declaring line may be the one that was left out
                for n, kind, typ in _scan_locals(doc.text, off):
                    if n == ref.name:
                        shown = f"{n}: {show_desc(typ)}" if typ is not None else n
                        md = f"```til\n{shown}\n```\n*{label('param' if kind == 'param' else 'local', self.lang)}*"
                        return {'contents': {'kind': 'markdown', 'value': md},
                                'range': doc.pmap.range(ref.start, ref.end)}
        if tok is None:
            return None
        word = doc.text[tok.start:tok.end]
        if tok.type in KEYWORD_TYPES or tok.type == TT.BOOL:
            name = 'TRUE' if tok.type == TT.BOOL and tok.value else ('FALSE' if tok.type == TT.BOOL else tok.type.name)
            if tok.type == TT.ELSE and k + 1 < len(view.t) and view.t[k + 1].type == TT.IF:
                name = 'ELSE'
            md = self._keyword_markdown(name, word)
            return {'contents': {'kind': 'markdown', 'value': md}, 'range': doc.pmap.range(tok.start, tok.end)}
        if tok.type == TT.ATTRIBUTE:
            inner = tok.value.split(':', 1)[0].strip()
            row = ATTRIBUTE_DOCS.get(inner)
            if row is not None:
                md = f"**#[{inner}]** — *{label('attribute', self.lang)}*\n\n{doc_text(row, self.lang, 1)}"
                return {'contents': {'kind': 'markdown', 'value': md}, 'range': doc.pmap.range(tok.start, tok.end)}
        if tok.type == TT.IDENT and word in ALIAS_WORDS:
            return {'contents': {'kind': 'markdown', 'value': self._builtin_markdown(ALIAS_WORDS[word][0], word)},
                    'range': doc.pmap.range(tok.start, tok.end)}
        return None

    def _ref_markdown(self, doc: Document, ref: Ref) -> Optional[str]:
        t = self._target(doc, ref)
        if isinstance(t, Sym):
            if t.implicit and t.name == 'self':
                ty = t.typ[1] if t.typ and len(t.typ) > 1 else None
                return "```til\nself" + (f": {ty}" if ty else "") + "\n```\n*" + label('param', self.lang) + "*"
            if t.implicit and t.name in ('error', 'result'):
                detail = self._sym_detail(doc, t)
                return f"```til\n{detail}\n```\n*{label('local', self.lang)}*"
            return self._sym_markdown(doc, t)
        if isinstance(t, tuple):
            if t[0] == 'module':
                p = t[1]
                where = f" · `{os.path.relpath(p, self.root) if self.root and p else p}`" if p else ""
                return f"```til\nimport {t[2]}\n```\n*{label('module', self.lang)}*{where}"
            if t[0] == 'builtin':
                kind, name = t[1], t[2]
                if kind == 'function':
                    shown = doc.text[ref.start:ref.end]
                    return self._builtin_markdown(name, shown)
                if kind == 'method':
                    return self._method_markdown(name) or None
                if kind == 'type':
                    return f"```til\n{doc.text[ref.start:ref.end]}\n```\n*{label('type', self.lang)}*"
                if kind == 'const' and ref.name == 'None':
                    return "```til\nNone\n```"
        return None

    # ── signature help ─────────────────────────────────────────────────────
    def signature_help(self, uri, line, character) -> Optional[dict]:
        saved = self._cheap, self._reuse
        self._cheap = self._reuse = True
        try:
            return self._signature_help(uri, line, character)
        finally:
            self._cheap, self._reuse = saved

    def _signature_help(self, uri, line, character) -> Optional[dict]:
        doc = self._doc(uri)
        if doc is None:
            return None
        syn = self._syntax(doc)
        off = doc.pmap.offset(line, character)
        call = _open_call(doc.text, off)
        if call is None:
            return None
        paren, active = call
        callee = _callee_before(doc.text, paren)
        if callee is None:
            return None
        recv_text, name = callee
        sig = self._call_signature(doc, syn, paren, recv_text, name)
        if sig is None:
            return None
        label_text, params, documentation = sig
        param_info = []
        for a, b in params:
            param_info.append({'label': [utf16_len(label_text[:a]), utf16_len(label_text[:b])]})
        sig_d = {'label': label_text, 'parameters': param_info}
        if documentation:
            sig_d['documentation'] = {'kind': 'markdown', 'value': documentation}
        if params:
            active = min(active, len(params) - 1) if not label_text.rstrip(')').endswith('...') else active
        return {'signatures': [sig_d], 'activeSignature': 0,
                'activeParameter': min(active, max(len(params) - 1, 0))}

    def _call_signature(self, doc: Document, syn: Syntax, paren: int, recv_text: Optional[str], name: str):
        lang = self.lang
        scope = syn.index.scope_at(paren)
        if recv_text is None:
            sym = scope.lookup(name, paren) or self._resolve_free(doc, name)
            if sym is None and name in ALIAS_WORDS:
                name = ALIAS_WORDS[name][0]
            if isinstance(sym, Sym):
                if sym.kind in ('function', 'method') and sym.node is not None:
                    return self._fn_sig_parts(doc, sym)
                if sym.kind == 'struct' and sym.node is not None:
                    parts = [f"{f.name}: {_type_text(f.type)}" for f in sym.node.fields]
                    return _sig_from_parts(sym.name, parts, '') + (None,)
                return None
            if name in BUILTIN_DOCS:
                row = BUILTIN_DOCS[name]
                parts = []
                for p in row[0]:
                    if p == '...':
                        if parts:
                            parts[-1] += '...'
                        continue
                    parts.append(p[1:] + '?' if p.startswith('?') else p)
                (lbl, spans) = _sig_from_parts(name, parts, row[1])
                return lbl, spans, doc_text(row, lang, 2)
            return None
        # a method / static function / variant: resolve the receiver
        recv = self._eval_receiver(doc, syn, recv_text, paren)
        if recv is None:
            return None
        kind, value = recv
        if kind == 'static':
            sym_list = self._members_named(doc, value, name)
            if sym_list and sym_list[0].kind in ('method', 'function'):
                return self._fn_sig_parts(doc, sym_list[0])
            if sym_list and sym_list[0].kind == 'variant':
                v = sym_list[0]
                parts = [_type_text(t) for t in getattr(v.node, 'fields', []) or []]
                return _sig_from_parts(f"{value}.{name}", parts, '') + (None,)
            return None
        if kind == 'module':
            d = self._doc_for_path(value) if value else None
            if d is not None:
                s = self._syntax(d).index.toplevel.get(name)
                if s is not None and s.kind == 'function':
                    return self._fn_sig_parts(d, s)
            return None
        desc = value
        if desc is not None and desc[0] in ('struct', 'enum'):
            ms = self._members_named(doc, desc[1], name)
            if ms and ms[0].kind == 'method':
                return self._fn_sig_parts(doc, ms[0])
        row = METHOD_DOCS.get(name)
        if row is not None:
            parts = [p[1:] + '?' if p.startswith('?') else p for p in row[1]]
            lbl, spans = _sig_from_parts('.' + name, parts, row[2])
            return lbl, spans, doc_text(row, lang, 3)
        return None

    def _fn_sig_parts(self, doc: Document, sym: Sym):
        sdoc = self._doc(sym.uri) or doc
        full = self._signature(sdoc, sym)
        head = full.split('(', 1)[0]
        inner = full[len(head) + 1:]
        depth = 0
        end = 0
        for i, c in enumerate(inner):
            if c in '([{<':
                depth += 1
            elif c in ')]}>':
                if depth == 0 and c == ')':
                    end = i
                    break
                depth -= 1
        params_text = inner[:end]
        tail = inner[end + 1:]
        parts = _split_params(params_text)
        if parts and parts[0].startswith('self') or (parts and parts[0].startswith('mut self')):
            parts = parts[1:]
        ret = tail.strip()[3:] if tail.strip().startswith('->') else ''
        lbl, spans = _sig_from_parts(head, parts, ret)
        return lbl, spans, sym.doc or None

    def _eval_receiver(self, doc: Document, syn: Syntax, recv_text: str, at: int):
        """Type of the expression before a '.': ('static', type name) for a type,
        ('module', path), ('value', desc) or None."""
        chain = _receiver_chain(recv_text)
        if not chain:
            return None
        scope = syn.index.scope_at(at)
        first = chain[0]
        cur = None
        i = 1
        if first[0] == 'name':
            name = first[1]
            sym = scope.lookup(name, at)
            if sym is None:
                sym = self._scan_name(doc, at, name)
            if sym is None:
                sym = self._resolve_free(doc, name)
            if sym is None and name == 'self':
                return None
            if isinstance(sym, Sym):
                if sym.kind in ('struct', 'enum', 'trait') and len(chain) == 1:
                    return ('static', sym.name)
                if sym.kind == 'module':
                    return ('module', self._module_path(doc, sym.node.module if sym.node is not None else name))
                if sym.kind in ('struct', 'enum'):
                    if len(chain) > 1 and chain[1][0] == 'name':
                        ms = self._members_named(doc, sym.name, chain[1][1])
                        if ms and ms[0].kind == 'method' and ms[0].node is not None:
                            cur = _desc(ms[0].node.ret_type)
                        elif ms and ms[0].kind == 'variant':
                            cur = ('enum', sym.name)
                        i = 2
                        if i < len(chain) and chain[i][0] == 'call':
                            i += 1
                    else:
                        return ('static', sym.name)
                elif sym.kind in ('function',) and len(chain) > 1 and chain[1][0] == 'call':
                    cur = _desc(sym.node.ret_type) if sym.node is not None else None
                    i = 2
                else:
                    cur = self._sym_desc(doc, sym) if isinstance(sym, Sym) else None
            elif isinstance(sym, tuple):
                cur = sym[1] if len(sym) > 1 else None
            elif name in _BUILTIN_RESULT and len(chain) > 1 and chain[1][0] == 'call':
                cur = _BUILTIN_RESULT[name]
                i = 2
            else:
                return ('value', None)
        elif first[0] == 'lit':
            cur = (first[1],) if first[1] != 'array' else ('array', None)
        else:
            return ('value', None)
        while i < len(chain):
            step = chain[i]
            if step[0] == 'name':
                is_call = i + 1 < len(chain) and chain[i + 1][0] == 'call'
                cur = self._member_type(doc, cur, step[1], is_call)
                i += 2 if is_call else 1
            elif step[0] == 'index':
                if cur is not None and cur[0] == 'array':
                    cur = cur[1] if len(cur) > 1 else None
                elif cur is not None and cur[0] == 'map':
                    cur = cur[2] if len(cur) > 2 else None
                elif cur is not None and cur[0] == 'str':
                    cur = ('str',)
                else:
                    cur = None
                i += 1
            elif step[0] == 'call':
                cur = None
                i += 1
            elif step[0] == 'try':
                cur = cur[1] if cur is not None and cur[0] in ('option', 'result') and len(cur) > 1 else None
                i += 1
            else:
                i += 1
        if cur is not None and cur[0] == 'struct':
            s = self._resolve_free(doc, cur[1])
            if s is not None and s.kind == 'enum':
                cur = ('enum', cur[1])
            elif s is not None and s.kind == 'alias' and s.node is not None:
                cur = _desc(s.node.type)
        return ('value', cur)

    def _member_type(self, doc: Document, cur, name: str, is_call: bool):
        if cur is None:
            return None
        if cur[0] in ('struct', 'enum'):
            ms = self._members_named(doc, cur[1], name)
            if not ms:
                return None
            m = ms[0]
            if m.kind == 'field':
                return m.typ
            if m.kind == 'method' and m.node is not None:
                return _desc(m.node.ret_type)
            return None
        return _method_result(cur, name)

    def _scan_name(self, doc: Document, at: int, name: str):
        """Token-scan fallback (broken code): a let/var/for/parameter named `name`
        above `at` in the enclosing definition, with its written type."""
        for n, kind, typ in _scan_locals(doc.text, at):
            if n == name:
                return ('scan', typ)
        return None

    # ── completion ──────────────────────────────────────────────────────────
    def completion(self, uri, line, character) -> List[dict]:
        saved = self._cheap, self._reuse
        self._cheap = self._reuse = True
        try:
            return self._completion(uri, line, character)
        finally:
            self._cheap, self._reuse = saved

    def _completion(self, uri, line, character) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        syn = self._syntax(doc)
        off = doc.pmap.offset(line, character)
        line_start = doc.pmap.line_starts[doc.pmap.line_of(off)]
        prefix = doc.text[line_start:off]
        state, info = _prefix_state(prefix)
        if state in ('comment', 'str'):
            return []
        if state == 'attr':
            return self._complete_attribute(doc, syn, off, prefix[info + 2:])
        expr_prefix = prefix[info:] if state == 'fexpr' else prefix
        m = re.search(r'([^\W\d]\w*)?$', expr_prefix)
        word = (m.group(1) or "") if m else ""
        before = expr_prefix[:len(expr_prefix) - len(word)]
        # import / from
        imp = self._complete_import(doc, syn, prefix, word)
        if imp is not None:
            return imp
        stripped = before.rstrip()
        if stripped.endswith('.') and not stripped.endswith('..'):
            return self._complete_member(doc, syn, off, stripped[:-1])
        if _is_type_position(before, syn.lexed.languages if syn.lexed else None):
            return self._complete_types(doc, syn, off)
        at_stmt = before.strip() == "" and state == 'code'
        return self._complete_names(doc, syn, off, prefix, at_stmt)

    def resolve_completion(self, item: dict) -> dict:
        out = dict(item)
        data = item.get('data') or {}
        if 'documentation' not in out:
            k = data.get('tills')
            name = data.get('name', item.get('label'))
            md = None
            if k == 'builtin':
                md = self._builtin_markdown(name)
            elif k == 'method':
                md = self._method_markdown(name)
            elif k == 'keyword':
                md = self._keyword_markdown(data.get('kw', ''), item.get('label', ''))
            if md:
                out['documentation'] = {'kind': 'markdown', 'value': md}
        return out

    def _item(self, label_text: str, kind: int, detail: str = "", md: str = "", sort: str = "5",
              insert: Optional[str] = None, snippet: bool = False, data: Optional[dict] = None) -> dict:
        it = {'label': label_text, 'kind': kind, 'sortText': f"{sort}_{label_text}"}
        if detail:
            it['detail'] = detail
        if md:
            it['documentation'] = {'kind': 'markdown', 'value': md}
        if insert is not None:
            it['insertText'] = insert
            it['insertTextFormat'] = INSERT_SNIPPET if snippet else INSERT_PLAIN
        if data:
            it['data'] = data
        return it

    def _sym_item(self, doc: Document, s: Sym, sort: str, call_snippets: bool = True) -> dict:
        kinds = {'function': CompletionKind.FUNCTION, 'method': CompletionKind.METHOD,
                 'struct': CompletionKind.STRUCT, 'enum': CompletionKind.ENUM,
                 'variant': CompletionKind.ENUM_MEMBER, 'field': CompletionKind.FIELD,
                 'trait': CompletionKind.INTERFACE, 'alias': CompletionKind.CLASS,
                 'global': CompletionKind.VARIABLE, 'const': CompletionKind.CONSTANT,
                 'local': CompletionKind.VARIABLE, 'param': CompletionKind.VARIABLE,
                 'capability': CompletionKind.CONSTANT, 'module': CompletionKind.MODULE,
                 'typeparam': CompletionKind.TYPE_PARAMETER}
        sdoc = self._doc(s.uri) or doc
        detail = self._sym_detail(sdoc, s)
        insert = None
        snippet = False
        if call_snippets and s.kind in ('function', 'method') and s.node is not None:
            nparams = len([p for p in s.node.params if p.name != 'self'])
            insert = f"{s.name}($1)$0" if nparams else f"{s.name}()$0"
            snippet = True
        md = s.doc or ""
        return self._item(s.name, kinds.get(s.kind, CompletionKind.TEXT), detail, md, sort, insert, snippet)

    def _complete_names(self, doc: Document, syn: Syntax, off: int, prefix: str, at_stmt: bool) -> List[dict]:
        items: List[dict] = []
        seen: Set[str] = set()
        cur_indent = _indent_width(prefix)            # the cursor's column decides the block it is in
        scope = syn.index.scope_at(off, cur_indent)
        for s in scope.visible(off):
            if s.name in seen or (s.implicit and s.kind != 'module' and s.name not in ('self', 'error', 'result')):
                continue
            seen.add(s.name)
            sort = '0' if s.scope is not None and s.scope.kind != 'module' else '1'
            items.append(self._sym_item(doc, s, sort))
        # broken code: names a token scan finds above the cursor
        for n, kind, typ in _scan_locals(doc.text, off):
            if n in seen:
                continue
            seen.add(n)
            items.append(self._item(n, CompletionKind.VARIABLE, f"{n}: {typ}" if typ else n, sort='0'))
        if not syn.ok and doc.last_good is not None and doc.last_good.index is not None and \
                doc.last_good.index is not syn.index:
            for name, s in doc.last_good.index.toplevel.items():
                if name not in seen:
                    seen.add(name)
                    items.append(self._sym_item(doc, s, '1'))
        # top-level names of imported files
        if doc.path:
            for p in self._closure(doc.path):
                d = self._doc_for_path(p)
                if d is None:
                    continue
                for name, s in self._syntax(d).index.toplevel.items():
                    if name in seen or s.kind == 'module' or name == 'main':
                        continue
                    seen.add(name)
                    items.append(self._sym_item(d, s, '2'))
        # builtins (and their localized spellings in this file's languages)
        langs = syn.languages
        for name in sorted(BUILTIN_NAMES):
            if name in seen:
                continue
            seen.add(name)
            items.append(self._builtin_item(name, name))
        for alias, (canon, alang) in ALIAS_WORDS.items():
            if alang in langs and alias not in seen and canon != 'main':
                seen.add(alias)
                items.append(self._builtin_item(alias, canon))
        if 'None' not in seen:
            items.append(self._item('None', CompletionKind.CONSTANT, 'None', sort='3'))
        items += self._keyword_items(syn, at_stmt)
        if at_stmt:
            items += self._snippet_items(syn)
        return items

    def _builtin_item(self, shown: str, name: str) -> dict:
        row = BUILTIN_DOCS.get(name)
        arity = til.NATIVE_ARITY.get(name, (1, 1))
        insert = f"{shown}()$0" if arity[1] == 0 else f"{shown}($1)$0"
        return self._item(shown, CompletionKind.FUNCTION, self._builtin_signature(name),
                          doc_text(row, self.lang, 2) if row else "", '3', insert, True,
                          {'tills': 'builtin', 'name': name})

    def _keyword_items(self, syn: Syntax, at_stmt: bool) -> List[dict]:
        out = []
        primary = syn.primary
        langs = list(dict.fromkeys([primary] + [c for c in syn.languages]))
        seen = set()
        for code in langs:
            for name, words in KEYWORDS_BY_TYPE.items():
                for w in words.get(code, []):
                    if w in seen:
                        continue
                    seen.add(w)
                    row = KEYWORD_DOCS.get(name)
                    sort = '4' if code == primary else '6'
                    others = [keyword_spelling(name, c) for c in KEYWORD_LANGS if c != code]
                    detail = f"{w} = " + " = ".join(dict.fromkeys(others))
                    out.append(self._item(w, CompletionKind.KEYWORD, detail,
                                          doc_text(row, self.lang, 0) if row else "", sort,
                                          data={'tills': 'keyword', 'kw': name}))
        return out

    def _snippet_items(self, syn: Syntax) -> List[dict]:
        lang = syn.primary
        out = []
        main_word = next((a for a, (c, l) in ALIAS_WORDS.items() if c == 'main' and l == lang), 'main')

        def fill(m):
            key = m.group(1)
            if key == 'MAIN':
                return main_word
            return keyword_spelling(key, lang)
        for name, body, en, ru, kk in SNIPPETS:
            text = re.sub(r'\{([A-Z]+)\}', fill, body)
            descr = {'en': en, 'ru': ru, 'kk': kk}.get(self.lang, en)
            out.append(self._item(name, CompletionKind.SNIPPET, descr, f"```til\n{re.sub(r'[$][{]?[0-9]+:?|[}]', '', text)}\n```",
                                  '5', text, True, {'tills': 'snippet'}))
        return out

    def _complete_member(self, doc: Document, syn: Syntax, off: int, recv_text: str) -> List[dict]:
        recv = self._eval_receiver(doc, syn, recv_text, off)
        items: List[dict] = []
        if recv is None:
            return items
        kind, value = recv
        if kind == 'static':
            for s in self._type_members(doc, value):
                if s.kind == 'field':
                    continue
                items.append(self._sym_item(self._doc(s.uri) or doc, s, '0' if s.static or s.kind == 'variant'
                                            else '1'))
            return items
        if kind == 'module':
            d = self._doc_for_path(value) if value else None
            if d is not None:
                for name, s in self._syntax(d).index.toplevel.items():
                    if s.kind != 'module' and name != 'main':
                        items.append(self._sym_item(d, s, '0'))
            return items
        desc = value
        if desc is not None and desc[0] in ('struct', 'enum'):
            for s in self._type_members(doc, desc[1]):
                if s.static and s.kind == 'method':
                    continue
                items.append(self._sym_item(self._doc(s.uri) or doc, s, '0'))
            items.append(self._method_item('to_string'))
            return items
        if desc is not None and desc[0] in ('int', 'float', 'bool', 'void', 'fn'):
            return [self._method_item('to_string')] if desc[0] not in ('void', 'fn') else []
        letter = {'str': 's', 'array': 'a', 'map': 'm', 'option': 'o', 'result': 'o'}.get(desc[0]) if desc else None
        if letter is not None:
            for name, row in METHOD_DOCS.items():
                if letter in row[0]:
                    items.append(self._method_item(name))
            return items
        # unknown receiver: every user member and every builtin method
        seen = set()
        files = self._related_docs(doc)
        for d in files:
            for (_c, _m), s in self._syntax(d).index.members.items():
                if s.kind in ('field', 'method') and not s.static and s.name not in seen:
                    seen.add(s.name)
                    items.append(self._sym_item(d, s, '0'))
        for name in METHOD_DOCS:
            if name not in seen:
                items.append(self._method_item(name))
        return items

    def _type_members(self, doc: Document, type_name: str) -> List[Sym]:
        out: List[Sym] = []
        files = self._related_docs(doc)
        seen = set()
        for d in files:
            for (c, m), s in self._syntax(d).index.members.items():
                if c == type_name and m not in seen:
                    seen.add(m)
                    out.append(s)
        return out

    def _method_item(self, name: str) -> dict:
        row = METHOD_DOCS[name]
        params = [p for p in row[1]]
        insert = f"{name}($1)$0" if params else f"{name}()$0"
        sig = f".{name}({', '.join(p[1:] + '?' if p.startswith('?') else p for p in params)})" + \
            (f" -> {row[2]}" if row[2] else "")
        return self._item(name, CompletionKind.METHOD, sig, doc_text(row, self.lang, 3), '2', insert, True,
                          {'tills': 'method', 'name': name})

    def _complete_types(self, doc: Document, syn: Syntax, off: int) -> List[dict]:
        items = []
        seen = set()
        for t in ('int', 'float', 'str', 'bool', 'char', 'void', 'Option', 'Result', 'Vec', 'HashMap',
                  'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'f32', 'f64'):
            seen.add(t)
            insert = f"{t}<$1>$0" if t in ('Option', 'Vec') else (f"{t}<$1, $2>$0" if t in ('Result', 'HashMap')
                                                                     else None)
            items.append(self._item(t, CompletionKind.CLASS if t[0].isupper() else CompletionKind.KEYWORD,
                                    t, '', '1', insert, insert is not None))
        scope = syn.index.scope_at(off)
        for s in scope.visible(off):
            if s.kind in ('struct', 'enum', 'alias', 'trait', 'typeparam') and s.name not in seen:
                seen.add(s.name)
                items.append(self._sym_item(doc, s, '0'))
        if doc.path:
            for p in self._closure(doc.path):
                d = self._doc_for_path(p)
                if d is None:
                    continue
                for name, s in self._syntax(d).index.toplevel.items():
                    if s.kind in ('struct', 'enum', 'alias', 'trait') and name not in seen:
                        seen.add(name)
                        items.append(self._sym_item(d, s, '0'))
        return items

    def _complete_attribute(self, doc: Document, syn: Syntax, off: int, inside: str) -> List[dict]:
        m = re.match(r'\s*([A-Za-z_]+)\s*:\s*(.*)$', inside)
        if m is None:
            out = []
            for name, row in ATTRIBUTE_DOCS.items():
                out.append(self._item(name, CompletionKind.PROPERTY, f"#[{name}]", doc_text(row, self.lang, 1),
                                      '0', row[0], '$' in row[0], {'tills': 'attribute'}))
            return out
        name, value = m.group(1), m.group(2)
        if name == 'lang':
            return [self._item(c, CompletionKind.ENUM_MEMBER, LANG_NAMES[c], '', '0') for c in ('ru', 'kk-latn', 'kk')]
        if name == 'level':
            return [self._item(str(n), CompletionKind.ENUM_MEMBER, f"level {n}", '', '0') for n in range(5)]
        if name == 'energy':
            return [self._item(w, CompletionKind.ENUM_MEMBER, w, '', '0')
                    for w in ('low', 'medium', 'high', 'balanced', 'performance')]
        if name == 'effects':
            names = {'io'} | set(syn.index.effects)
            return [self._item(e, CompletionKind.EVENT, e, '', '0') for e in sorted(names)]
        if name == 'cap':
            caps = set(syn.index.caps)
            if doc.path:
                for p in self._closure(doc.path):
                    d = self._doc_for_path(p)
                    if d is not None:
                        caps |= self._syntax(d).index.caps
            return [self._item(c, CompletionKind.CONSTANT, f"cap {c}", '', '0') for c in sorted(caps)]
        if name in ('requires', 'ensures'):
            items = []
            fn = None
            for s in syn.index.functions:
                if s.full_start <= off <= s.full_end:
                    fn = s
            if fn is None:
                nxt = [s for s in syn.index.functions if s.full_start >= off]
                fn = min(nxt, key=lambda s: s.full_start) if nxt else None
            if fn is not None:
                for p in fn.params:
                    if p.name != 'self':
                        items.append(self._sym_item(doc, p, '0'))
                if name == 'ensures':
                    items.append(self._item('result', CompletionKind.VARIABLE, 'result', '', '0'))
            for n in sorted(BUILTIN_NAMES):
                items.append(self._builtin_item(n, n))
            return items
        return []

    def _complete_import(self, doc: Document, syn: Syntax, prefix: str, word: str) -> Optional[List[dict]]:
        imp_words = set(KEYWORDS_BY_TYPE.get('IMPORT', {}).get(c, [''])[0] for c in KEYWORD_LANGS
                        if KEYWORDS_BY_TYPE.get('IMPORT', {}).get(c))
        from_words = set(w for c in KEYWORD_LANGS for w in KEYWORDS_BY_TYPE.get('FROM', {}).get(c, []))
        imp_words |= {w for c in KEYWORD_LANGS for w in KEYWORDS_BY_TYPE.get('IMPORT', {}).get(c, [])}
        alt = '|'.join(re.escape(w) for w in sorted(imp_words | from_words, key=len, reverse=True))
        m = re.match(r'^\s*(%s)\s+([\w.]*)$' % alt, prefix)
        if m:
            return self._module_items(doc, m.group(2))
        imp_alt = '|'.join(re.escape(w) for w in sorted(imp_words, key=len, reverse=True))
        from_alt = '|'.join(re.escape(w) for w in sorted(from_words, key=len, reverse=True))
        m = re.match(r'^\s*(%s)\s+([\w.]+)\s+(%s)\s+([\w\s,]*)$' % (from_alt, imp_alt), prefix)
        if m:
            path = self._module_path(doc, m.group(2))
            d = self._doc_for_path(path) if path else None
            if d is None:
                return []
            return [self._sym_item(d, s, '0', call_snippets=False)
                    for name, s in self._syntax(d).index.toplevel.items() if s.kind != 'module' and name != 'main']
        return None

    def _module_items(self, doc: Document, typed: str) -> List[dict]:
        names: Dict[str, str] = {}
        base = os.path.dirname(self._filename(doc))
        dirs = [base]
        if self.root and self.root != base:
            dirs.append(self.root)
        for home in til.til_home_dirs():
            sd = os.path.join(home, 'stdlib')
            if os.path.isdir(sd):
                dirs.append(sd)
        own = path_key(doc.path) if doc.path else None
        for d in dirs:
            for dirpath, dirnames, filenames in os.walk(d):
                rel = os.path.relpath(dirpath, d)
                depth = 0 if rel == '.' else rel.count(os.sep) + 1
                dirnames[:] = [x for x in dirnames if x not in _SKIP_DIRS and not x.startswith('.')] \
                    if depth < 2 else []
                for fn in filenames:
                    if not fn.endswith('.til'):
                        continue
                    full = os.path.join(dirpath, fn)
                    if own and path_key(full) == own:
                        continue
                    parts = ([] if rel == '.' else rel.split(os.sep)) + [fn[:-4]]
                    if parts[-1] == 'mod' and len(parts) > 1:
                        parts = parts[:-1]
                    if not all(re.fullmatch(r'[^\W\d]\w*', p) for p in parts):
                        continue
                    names.setdefault('.'.join(parts), full)
        for pd in til.package_search_dirs(self._filename(doc), '.') if hasattr(til, 'package_search_dirs') else []:
            try:
                for entry in os.listdir(pd):
                    if os.path.isdir(os.path.join(pd, entry)) and re.fullmatch(r'[^\W\d]\w*', entry):
                        names.setdefault(entry, os.path.join(pd, entry))
            except OSError:
                pass
        out = []
        for name, full in sorted(names.items()):
            rel = os.path.relpath(full, self.root) if self.root else full
            out.append(self._item(name, CompletionKind.MODULE, rel, '', '0'))
        return out

    # ── symbols ─────────────────────────────────────────────────────────────
    _SK = {'function': SymbolKind.FUNCTION, 'method': SymbolKind.METHOD, 'struct': SymbolKind.STRUCT,
           'enum': SymbolKind.ENUM, 'variant': SymbolKind.ENUM_MEMBER, 'field': SymbolKind.FIELD,
           'trait': SymbolKind.INTERFACE, 'alias': SymbolKind.TYPE_PARAMETER, 'global': SymbolKind.VARIABLE,
           'const': SymbolKind.CONSTANT, 'capability': SymbolKind.KEY, 'module': SymbolKind.MODULE}

    def _doc_symbol(self, doc: Document, s: Sym, children: Optional[List[dict]] = None,
                    pmap: Optional[PositionMap] = None) -> dict:
        kind = self._SK.get(s.kind, SymbolKind.VARIABLE)
        if s.kind == 'global' and not s.mutable:
            kind = SymbolKind.CONSTANT
        pm = pmap or doc.pmap
        d = {'name': s.name, 'kind': kind, 'detail': self._sym_detail(doc, s) if s.kind in ('function', 'method')
             else '', 'range': self._clamped_range(doc, pm, min(s.full_start, s.start), max(s.full_end, s.end)),
             'selectionRange': self._clamped_range(doc, pm, s.start, s.end)}
        if children:
            d['children'] = children
        return d

    @staticmethod
    def _clamped_range(doc: Document, pm: PositionMap, a: int, b: int) -> dict:
        """A range from another version's offsets (`pm`), kept inside `doc`."""
        r = pm.range(a, b)
        if pm is doc.pmap:
            return r
        last = doc.pmap.line_count - 1
        for k in ('start', 'end'):
            ln = min(r[k]['line'], last)
            r[k] = {'line': ln, 'character': min(r[k]['character'], utf16_len(doc.pmap.line_text(ln)))}
        if (r['end']['line'], r['end']['character']) < (r['start']['line'], r['start']['character']):
            r['end'] = dict(r['start'])
        return r

    def document_symbols(self, uri) -> List[dict]:
        saved, self._cheap = self._cheap, True
        try:
            return self._document_symbols(uri)
        finally:
            self._cheap = saved

    def _document_symbols(self, uri) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        syn = self._syntax(doc)
        ix = syn.index
        items: List[Tuple[int, dict]] = []
        for s in ix.outline:
            if s.kind in ('function',) and s.container is None and s.container_name is None:
                items.append((s.full_start, self._doc_symbol(doc, s)))
            elif s.kind in ('struct', 'enum', 'trait'):
                kids = [self._doc_symbol(doc, c) for c in s.children]
                items.append((s.full_start, self._doc_symbol(doc, s, kids)))
            elif s.kind in ('global', 'const', 'alias', 'capability'):
                items.append((s.full_start, self._doc_symbol(doc, s)))
        for type_name, trait, start, end, first_i, impl_i, methods in ix.impls:
            name = f"impl {trait} for {type_name}" if trait else f"impl {type_name}"
            kids = [self._doc_symbol(doc, m) for m in methods]
            sel_a, sel_b = start, start + len(doc.text[start:end].split('\n', 1)[0])
            d = {'name': name, 'kind': SymbolKind.CLASS, 'detail': '',
                 'range': doc.pmap.range(start, max(end, sel_b)), 'selectionRange': doc.pmap.range(sel_a, sel_b)}
            if kids:
                d['children'] = kids
            items.append((start, d))
        if not syn.ok and doc.last_good is not None and doc.last_good.index is not None and \
                doc.last_good.index is not ix and doc.last_good.pmap is not None:
            # broken now: the definitions the repaired parse lost, from the last good parse
            have = {d['name'] for _s, d in items}
            old_pm = doc.last_good.pmap
            for s in doc.last_good.index.outline:
                if s.name in have or s.kind not in ('function', 'struct', 'enum', 'trait', 'global', 'const',
                                                    'alias', 'capability'):
                    continue
                kids = [self._doc_symbol(doc, c, pmap=old_pm) for c in s.children] if s.children else None
                d = self._doc_symbol(doc, s, kids, pmap=old_pm)
                line = d['range']['start']['line']
                items.append((doc.pmap.line_starts[min(line, doc.pmap.line_count - 1)], d))
        items.sort(key=lambda x: x[0])
        return [d for _s, d in items]

    def workspace_symbols(self, query) -> List[dict]:
        q = (query or "").lower()
        out = []
        paths = self._workspace_paths()
        docs: List[Document] = []
        seen_keys = set()
        for d in self.docs.values():
            docs.append(d)
            if d.path:
                seen_keys.add(path_key(d.path))
        for p in paths:
            if path_key(p) in seen_keys:
                continue
            d = self._doc_for_path(p)
            if d is not None:
                docs.append(d)
        for d in docs:
            ix = self._syntax(d).index
            for s in ix.syms:
                if s.kind not in self._SK or s.kind == 'module' or s.implicit:
                    continue
                if s.scope is not None and s.scope.kind != 'module' and s.kind not in ('method', 'field', 'variant'):
                    continue
                if q and not _fuzzy(q, s.name.lower()):
                    continue
                cont = s.container.name if s.container is not None else s.container_name
                item = {'name': s.name, 'kind': self._SK.get(s.kind, SymbolKind.VARIABLE),
                        'location': {'uri': d.uri, 'range': d.pmap.range(s.start, s.end)}}
                if cont:
                    item['containerName'] = cont
                out.append(item)
                if len(out) >= 500:
                    return out
        return out

    # ── folding ─────────────────────────────────────────────────────────────
    def folding_ranges(self, uri) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        syn = self._syntax(doc)
        view = syn.view
        toks = view.t
        out = []
        seen = set()
        for i, j in view.indent_match.items():
            k = i - 1
            while k >= 0 and toks[k].type in LAYOUT:
                k -= 1
            if k < 0:
                continue
            start = toks[k].line - 1
            last = view.last_real_before(j)
            end = toks[last].line - 1
            if end > start and (start, end) not in seen:
                seen.add((start, end))
                out.append({'startLine': start, 'endLine': end})
        for i, j in view.match.items():
            if j <= i or toks[i].type not in OPENERS:
                continue
            a, b = toks[i].line - 1, toks[j].line - 1
            if b - 1 > a and (a, b - 1) not in seen:
                seen.add((a, b - 1))
                out.append({'startLine': a, 'endLine': b - 1, 'kind': 'region'})
        lines = doc.text.split('\n')
        run: List[int] = []
        imports: List[int] = []

        def flush(rng: List[int], kind: str):
            if len(rng) >= 2:
                out.append({'startLine': rng[0], 'endLine': rng[-1], 'kind': kind})
        imp_words = {w for c in KEYWORD_LANGS for n in ('IMPORT', 'FROM') for w in KEYWORDS_BY_TYPE.get(n, {}).get(c, [])}
        for n, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith('#') and not s.startswith('#['):
                run.append(n)
                continue
            flush(run, 'comment')
            run = []
            first = s.split(' ', 1)[0] if s else ''
            if first in imp_words:
                imports.append(n)
            elif s:
                flush(imports, 'imports')
                imports = []
        flush(run, 'comment')
        flush(imports, 'imports')
        out.sort(key=lambda r: (r['startLine'], r['endLine']))
        return out

    # ── formatting ──────────────────────────────────────────────────────────
    def formatting(self, uri, tab_size=4) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        new = format_source(doc.text, tab_size)
        if new is None or new == doc.text:
            return []
        return _line_edits(doc, new)

    # ── inlay hints ─────────────────────────────────────────────────────────
    def inlay_hints(self, uri, rng) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        syn = self._syntax(doc)
        if not syn.ok:
            return []
        if rng:
            a = doc.pmap.offset(rng['start']['line'], rng['start']['character'])
            b = doc.pmap.offset(rng['end']['line'], rng['end']['character'])
        else:
            a, b = 0, len(doc.text)
        out = []
        for s in syn.index.syms:
            if not (a <= s.start <= b):
                continue
            if s.kind in ('local', 'global', 'const') and isinstance(s.node, til.VarDecl) and \
                    s.node.type_ann is None and not s.implicit:
                _t, text = self._inferred(doc, s)
                if text:
                    out.append({'position': doc.pmap.position(s.end), 'label': f": {text}", 'kind': 1,
                                'paddingLeft': False, 'paddingRight': False})
            elif s.kind == 'local' and isinstance(s.node, til.For):
                _t, text = self._inferred(doc, s)
                if text:
                    out.append({'position': doc.pmap.position(s.end), 'label': f": {text}", 'kind': 1})
            elif s.kind in ('function', 'method') and isinstance(s.node, til.FuncDef) and \
                    isinstance(s.node.ret_type, til.VoidType) and s.name != 'main':
                ret = self._inferred_return(doc, s)
                if ret:
                    out.append({'position': doc.pmap.position(s.header_end), 'label': f" -> {ret}", 'kind': 1})
        out.sort(key=lambda h: (h['position']['line'], h['position']['character']))
        return out

    def _inferred_return(self, doc: Document, s: Sym) -> Optional[str]:
        sem = self._types_for(doc)
        if sem is None or not s.key or s.key not in sem.types.funcs:
            return None
        info = sem.types.funcs[s.key]
        if info.scheme is None:
            return None
        tt = sem.tt
        try:
            term = tt.deep(info.scheme.sig.ret, None)
            if tt.contains(term, ('var',)):
                return None
            text = tt.show(term)
        except Exception:
            return None
        if text in ('None', 'unknown', 'dynamic') or '?' in text:
            return None
        # was the return type written (`-> void`)? then no hint
        view = self._syntax(doc).view
        lp = None
        for i, t in enumerate(view.t):
            if t.start == s.start:
                lp = view.find(i, (TT.LPAREN,))
                break
        if lp is not None and lp in view.match:
            rp = view.match[lp]
            if rp + 1 < len(view.t) and view.t[rp + 1].type == TT.ARROW:
                return None
        return text

    # ── code lenses ─────────────────────────────────────────────────────────
    def code_lenses(self, uri) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        syn = self._syntax(doc)
        ix = syn.index
        out = []
        run_title = localize("▶ Run", self.lang)
        debug_title = localize("Debug", self.lang)
        prove_title = localize("Prove", self.lang)

        def lens(start: int, end: int, title: str, command: str, function: Optional[str]):
            out.append({'range': doc.pmap.range(start, end),
                        'command': {'title': title, 'command': command,
                                    'arguments': [{'uri': doc.uri, 'function': function}]}})
        if ix.main is not None:
            lens(ix.main.start, ix.main.end, run_title, 'til.run', 'main')
            lens(ix.main.start, ix.main.end, debug_title, 'til.debug', 'main')
        elif ix.has_script:
            ls = doc.pmap.line_starts[min(ix.script_line, doc.pmap.line_count - 1)]
            le = doc.pmap.line_ends[min(ix.script_line, doc.pmap.line_count - 1)]
            lens(ls, max(le, ls), run_title, 'til.run', None)
            lens(ls, max(le, ls), debug_title, 'til.debug', None)
        for s in ix.functions:
            fn = s.node
            if isinstance(fn, til.FuncDef) and (fn.requires or fn.ensures) and s.key:
                lens(s.start, s.end, prove_title, 'til.prove', s.key)
        return out

    # ── code actions ────────────────────────────────────────────────────────
    def code_actions(self, uri, rng, diagnostics) -> List[dict]:
        doc = self._doc(uri)
        if doc is None:
            return []
        syn = self._syntax(doc)
        out: List[dict] = []
        diags = list(diagnostics or [])
        if rng is not None:
            a = doc.pmap.offset(rng['start']['line'], rng['start']['character'])
            b = doc.pmap.offset(rng['end']['line'], rng['end']['character'])
        else:
            a, b = 0, len(doc.text)
        if not diags:
            # the client did not send them: ours that touch the range
            for d in self.diagnostics(uri):
                da = doc.pmap.offset(d['range']['start']['line'], d['range']['start']['character'])
                db = doc.pmap.offset(d['range']['end']['line'], d['range']['end']['character'])
                if da <= b and db >= a:
                    diags.append(d)
        need = any(not (isinstance(d, dict) and isinstance(d.get('data'), dict) and 'en' in d['data'])
                   for d in diags)
        known = self.diagnostics(uri) if need else []
        for d in diags:
            if not isinstance(d, dict) or 'range' not in d:
                continue
            data = d.get('data') if isinstance(d.get('data'), dict) else None
            if data is None or 'en' not in data:
                match = next((k for k in known if k['range'] == d['range'] and k.get('code') == d.get('code')), None)
                if match is None:
                    continue
                data = match.get('data', {})
            try:
                out += self._quick_fixes(doc, syn, d, data.get('en', ''), data.get('enHint', ''))
            except Exception:
                if os.environ.get('TILLS_DEBUG'):
                    raise
        out += self._translate_actions(doc, syn)
        return out

    def _edit(self, doc: Document, start: int, end: int, text: str) -> dict:
        return {'range': doc.pmap.range(start, end), 'newText': text.replace('\n', doc.eol)}

    def _action(self, doc: Document, title: str, edits: List[dict], diag: Optional[dict] = None,
                kind: str = 'quickfix', preferred: bool = False) -> dict:
        a = {'title': title, 'kind': kind, 'edit': {'changes': {doc.uri: edits}}}
        if diag is not None:
            a['diagnostics'] = [diag]
        if preferred:
            a['isPreferred'] = True
        return a

    def _quick_fixes(self, doc: Document, syn: Syntax, d: dict, en: str, en_hint: str) -> List[dict]:
        out = []
        lang = self.lang
        da = doc.pmap.offset(d['range']['start']['line'], d['range']['start']['character'])
        db = doc.pmap.offset(d['range']['end']['line'], d['range']['end']['character'])
        m = re.match(r"cannot assign (?:twice )?to (?:immutable (?:variable|captured variable|global)|constant) "
                     r"'([^']+)'", en)
        if m:
            name = m.group(1)
            sym = syn.index.scope_at(da).lookup(name, da) or syn.index.toplevel.get(name)
            if isinstance(sym, Sym) and sym.decl_word is not None:
                ws, we, word = sym.decl_word
                info = KEYWORD_WORDS.get(word)
                wlang = info[1] if info else 'en'
                new = keyword_spelling('VAR', wlang)
                out.append(self._action(doc, localize(f"Change '{word}' to '{new}'", lang),
                                        [self._edit(doc, ws, we, new)], d, preferred=True))
        m = re.match(r"did you mean '([^']+)'\?", en_hint or "")
        if m and db > da:
            new = m.group(1)
            old = doc.text[da:db]
            out.append(self._action(doc, localize(f"Replace '{old}' with '{new}'", lang),
                                    [self._edit(doc, da, db, new)], d, preferred=True))
        m = re.match(r"unknown (?:function|name|struct) '([^']+)'", en)
        if m:
            out += self._import_fixes(doc, syn, m.group(1), d)
        m = re.match(r"pure function '([^']+)' performs effect '([^']+)'", en)
        if m:
            out += self._effect_fixes(doc, syn, m.group(1), m.group(2), d, pure=True)
        m = re.match(r"function '([^']+)' performs effect '([^']+)'.*but declares only", en)
        if m:
            out += self._effect_fixes(doc, syn, m.group(1), m.group(2), d, pure=False)
        m = re.match(r"add #\[cap: ([^\]]+)\] to '([^']+)'", en_hint or "")
        if m:
            cap, func = m.group(1), m.group(2)
            fsym = self._function_named(syn, func)
            if fsym is not None:
                ls = doc.pmap.line_starts[doc.pmap.line_of(fsym.full_start)]
                indent = doc.text[ls:fsym.full_start]
                if indent.strip():
                    indent = ''
                out.append(self._action(doc, localize(f"Add #[cap: {cap}] to '{func}'", lang),
                                        [self._edit(doc, ls, ls, f"{indent}#[cap: {cap}]\n")], d))
        return out

    def _function_named(self, syn: Syntax, key: str) -> Optional[Sym]:
        for s in syn.index.functions:
            if s.key == key or (s.key is None and s.qualname == key):
                return s
        return None

    def _import_fixes(self, doc: Document, syn: Syntax, name: str, d: dict) -> List[dict]:
        out = []
        if not doc.path:
            return out
        base = os.path.dirname(doc.path)
        have = {i.module for i in syn.index.imports}
        found: List[Tuple[str, str]] = []
        for p in self._workspace_paths():
            if path_key(p) == path_key(doc.path):
                continue
            mdoc = self._doc_for_path(p)
            if mdoc is None:
                continue
            s = self._syntax(mdoc).index.toplevel.get(name)
            stem = os.path.splitext(os.path.basename(p))[0]
            if (s is None or s.kind == 'module') and stem != name:
                continue
            rel = os.path.relpath(p, base)
            if rel.startswith(os.pardir):
                continue
            parts = rel[:-4].split(os.sep)
            if parts[-1] == 'mod' and len(parts) > 1:
                parts = parts[:-1]
            module = '.'.join(parts)
            if module in have or not all(re.fullmatch(r'[^\W\d]\w*', x) for x in parts):
                continue
            found.append((module, p))
        for module, _p in sorted(set(found))[:5]:
            word = keyword_spelling('IMPORT', syn.primary)
            line = f"{word} {module}\n"
            at = self._import_insert_offset(doc, syn)
            out.append(self._action(doc, localize(f"Add an import of '{module}'", self.lang),
                                    [self._edit(doc, at, at, line)], d))
        return out

    def _import_insert_offset(self, doc: Document, syn: Syntax) -> int:
        if syn.index.imports:
            last = max(syn.index.imports, key=lambda i: i.line)
            ln = last.line + 1
            return doc.pmap.line_starts[ln] if ln < doc.pmap.line_count else len(doc.text)
        lines = doc.text.split('\n')
        k = 0
        while k < len(lines) and (lines[k].startswith('#') or not lines[k].strip()):
            if lines[k].startswith('#[') and not re.match(r'#\[\s*lang\s*:', lines[k]):
                break
            k += 1
        k = min(k, len(lines))
        while k > 0 and not lines[k - 1].strip():
            k -= 1
        return doc.pmap.line_starts[k] if k < doc.pmap.line_count else len(doc.text)

    def _effect_fixes(self, doc: Document, syn: Syntax, func: str, eff: str, d: dict, pure: bool) -> List[dict]:
        out = []
        fsym = self._function_named(syn, func)
        if fsym is None:
            return out
        view = syn.view
        idx = None
        for i, t in enumerate(view.t):
            if t.start == fsym.start:
                idx = i
                break
        if idx is None:
            return out
        attrs = Indexer.attrs_before(_AttrHelper(view), idx)
        for ai in attrs:
            tok = view.t[ai]
            if pure and tok.value == 'pure':
                ls = doc.pmap.line_starts[doc.pmap.line_of(tok.start)]
                le = doc.text.find('\n', tok.end)
                le = len(doc.text) if le < 0 else le + 1
                alone = doc.text[ls:tok.start].strip() == "" and doc.text[tok.end:le].strip() == ""
                rm = self._edit(doc, ls, le, "") if alone else self._edit(doc, tok.start, tok.end, "")
                out.append(self._action(doc, localize("Remove #[pure]", self.lang), [rm], d))
                out.append(self._action(doc, localize(f"Replace #[pure] with #[effects: {eff}]", self.lang),
                                        [self._edit(doc, tok.start, tok.end, f"#[effects: {eff}]")], d,
                                        preferred=True))
            elif not pure and tok.value.startswith('effects:'):
                current = [e.strip() for e in tok.value[len('effects:'):].split(',') if e.strip()]
                new = ', '.join(current + [eff])
                out.append(self._action(doc, localize(f"Add '{eff}' to #[effects: ...]", self.lang),
                                        [self._edit(doc, tok.start, tok.end, f"#[effects: {new}]")], d,
                                        preferred=True))
        return out

    def _translate_actions(self, doc: Document, syn: Syntax) -> List[dict]:
        out = []
        if syn.lexed is None or syn.lexed.error is not None:
            return out
        cache = doc.translations
        if cache.get('version') != doc.version:
            cache.clear()
            cache['version'] = doc.version
        for target in KEYWORD_LANGS:
            if target == syn.primary:
                continue
            if target not in cache:
                try:
                    with _Capture(), _QuietStderr():
                        cache[target] = til.translate_source(doc.text, target, self._filename(doc))
                except Exception:
                    cache[target] = None
            text = cache[target]
            if text is None:
                continue
            if text == doc.text:
                continue
            edit = self._edit(doc, 0, len(doc.text), text)
            out.append(self._action(doc, localize(f"Translate file to {LANG_NAMES[target]}", self.lang), [edit],
                                    None, kind='refactor.rewrite'))
        return out

    # ── semantic tokens ─────────────────────────────────────────────────────
    def semantic_tokens(self, uri) -> dict:
        doc = self._doc(uri)
        if doc is None:
            return {'data': []}
        syn = self._syntax(doc)
        text = doc.text
        spans: List[Tuple[int, int, int, int]] = []          # (start, end, type, modifiers)
        refs_by_start = {}
        for r in syn.index.refs:
            refs_by_start.setdefault(r.start, r)
        kind_type = {'function': 'function', 'method': 'method', 'struct': 'struct', 'enum': 'enum',
                     'variant': 'enumMember', 'field': 'property', 'trait': 'interface', 'alias': 'type',
                     'global': 'variable', 'const': 'variable', 'local': 'variable', 'param': 'parameter',
                     'capability': 'type', 'module': 'namespace', 'typeparam': 'typeParameter'}
        for tok in syn.view.t:
            tt = tok.type
            if tt in LAYOUT or tok.end <= tok.start:
                continue
            if tt in KEYWORD_TYPES or tt == TT.BOOL:
                spans.append((tok.start, tok.end, _TT_INDEX['keyword'], 0))
            elif tt in (TT.INT, TT.FLOAT):
                spans.append((tok.start, tok.end, _TT_INDEX['number'], 0))
            elif tt in (TT.STRING, TT.CHAR):
                spans.append((tok.start, tok.end, _TT_INDEX['string'], 0))
            elif tt == TT.FSTRING:
                spans += self._fstring_spans(syn, tok, refs_by_start, kind_type)
            elif tt == TT.ATTRIBUTE:
                spans.append((tok.start, tok.end, _TT_INDEX['decorator'], 0))
            elif tt == TT.IDENT:
                r = refs_by_start.get(tok.start)
                sp = self._ref_span_type(doc, r, kind_type) if r is not None else None
                if sp is not None:
                    spans.append((tok.start, tok.end) + sp)
        lexed = syn.parsed.lexed if syn.parsed is not None and syn.parsed.lexed is not None else syn.lexed
        for start, end, _kind in (lexed.comments if lexed is not None else []):
            spans.append((start, end, _TT_INDEX['comment'], 0))
        spans.sort()
        return {'data': _encode_tokens(doc, spans)}

    def _ref_span_type(self, doc: Document, r: Ref, kind_type) -> Optional[Tuple[int, int]]:
        if r.builtin == 'function':
            return _TT_INDEX['function'], MOD_BUILTIN
        if r.builtin == 'method':
            return _TT_INDEX['method'], MOD_BUILTIN
        if r.builtin == 'type':
            return _TT_INDEX['type'], MOD_BUILTIN
        if r.builtin == 'const':
            return _TT_INDEX['keyword'], 0
        s = r.sym
        if s is None:
            if r.kind == 'module':
                return _TT_INDEX['namespace'], 0
            if r.kind == 'type':
                return _TT_INDEX['type'], 0
            if r.kind in ('member', 'module-member'):
                return _TT_INDEX['property'] if r.kind == 'member' else _TT_INDEX['function'], 0
            if r.kind == 'effect':
                return _TT_INDEX['function'], 0
            return _TT_INDEX['variable'], 0
        mods = 0
        if r.decl:
            mods |= MOD_DECL
        if r.write:
            mods |= MOD_WRITE
        if s.kind in ('global', 'local', 'const') and not s.mutable:
            mods |= MOD_READONLY
        if s.static:
            mods |= MOD_STATIC
        name = kind_type.get(s.kind, 'variable')
        if s.kind == 'struct' and r.kind == 'type' or s.kind == 'struct':
            name = 'struct'
        return _TT_INDEX[name], mods

    def _fstring_spans(self, syn: Syntax, tok, refs_by_start, kind_type) -> List[Tuple[int, int, int, int]]:
        doc_text = syn.lexed.text if syn.lexed is not None else ""
        raw = doc_text[tok.start:tok.end]
        offs = _fstring_expr_offsets(raw)
        out = []
        pos = tok.start
        for off in offs:
            a = tok.start + off - 1            # the '{'
            if a > pos:
                out.append((pos, a, _TT_INDEX['string'], 0))
            # the expression: identifiers from the index
            depth, k = 1, tok.start + off
            while k < tok.end and depth > 0:
                c = doc_text[k]
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            for start in sorted(s for s in refs_by_start if tok.start + off <= s < k):
                r = refs_by_start[start]
                sp = self._ref_span_type(None, r, kind_type)
                if sp is not None:
                    out.append((r.start, r.end) + sp)
            pos = k
        if pos < tok.end:
            out.append((pos, tok.end, _TT_INDEX['string'], 0))
        return out


class _AttrHelper:
    """Lets code actions reuse Indexer.attrs_before on a plain view."""

    def __init__(self, view: _View):
        self.v = view


# ═══════════════════════════════════════════════════════════════════════════════
#                                   HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class _NodeFiles:
    """Which file an AST node of the merged program comes from (None: this one)."""

    def __init__(self, own_nodes: Set[int], stmt_files: Dict[int, str], program):
        self.own = own_nodes
        self.stmt_files = stmt_files
        self.program = program
        self._map: Optional[Dict[int, str]] = None

    def file_of(self, node) -> Optional[str]:
        if node is None or id(node) in self.own:
            return None
        if self._map is None:
            self._map = {}
            for st in self.program.statements:
                p = self.stmt_files.get(id(st))
                if p is None:
                    continue
                for n in til.walk(st):
                    self._map[id(n)] = p
                if isinstance(st, til.FuncDef):
                    for prm in st.params:
                        if prm.default is not None:
                            for n in til.walk(prm.default):
                                self._map[id(n)] = p
        return self._map.get(id(node))


def _stmt_sig(st) -> Tuple[str, str, int, int]:
    name = til._definition_name(st) if hasattr(til, '_definition_name') else getattr(st, 'name', '')
    return (type(st).__name__, str(name or ''), getattr(st, 'line', 0) or 0, getattr(st, 'col', 0) or 0)


def _fragment_fails(text: str, languages) -> bool:
    try:
        with _Capture():
            toks = til.Lexer(text, "<fragment>", languages=languages).tokenize()
            til.Parser(toks, "<fragment>", languages=languages).parse_expression()
        return False
    except Exception:
        return True


def _scope_depth(sc: Scope) -> int:
    n = 0
    while sc.parent is not None:
        n += 1
        sc = sc.parent
    return n


def _scope_within(inner: Scope, outer: Scope) -> bool:
    sc = inner
    while sc is not None:
        if sc is outer:
            return True
        sc = sc.parent
    return False


def _one_line_change(old: str, new: str) -> bool:
    """`new` differs from `old` inside one line at most (same lines elsewhere)."""
    if old == new:
        return True
    if old.count('\n') != new.count('\n'):
        return False
    i = 0
    n = min(len(old), len(new))
    # common prefix, then common suffix: the difference must not contain a line break
    while i < n and old[i] == new[i]:
        i += 1
    j = 0
    while j < n - i and old[len(old) - 1 - j] == new[len(new) - 1 - j]:
        j += 1
    return '\n' not in old[i:len(old) - j] and '\n' not in new[i:len(new) - j]


def _fuzzy(q: str, name: str) -> bool:
    if q in name:
        return True
    i = 0
    for c in name:
        if i < len(q) and c == q[i]:
            i += 1
    return i == len(q)


def _encode_tokens(doc: Document, spans) -> List[int]:
    data: List[int] = []
    prev_line, prev_char = 0, 0
    last_end = -1
    for start, end, ttype, mods in spans:
        if start < last_end:            # overlapping (never expected): keep the first
            continue
        # split tokens that cross lines (never in TIL) at the line end
        line = doc.pmap.line_of(start)
        line_end = doc.pmap.line_ends[line]
        end = min(end, line_end)
        if end <= start:
            continue
        p = doc.pmap.position(start)
        length = utf16_len(doc.text[start:end])
        dl = p['line'] - prev_line
        dc = p['character'] - prev_char if dl == 0 else p['character']
        data += [dl, dc, length, ttype, mods]
        prev_line, prev_char = p['line'], p['character']
        last_end = end
    return data


def _sig_from_parts(head: str, parts: List[str], ret: str):
    label_text = head + '('
    spans = []
    for k, p in enumerate(parts):
        if k:
            label_text += ', '
        a = len(label_text)
        label_text += p
        spans.append((a, len(label_text)))
    label_text += ')'
    if ret:
        label_text += f" -> {ret}"
    return label_text, spans


def _split_params(text: str) -> List[str]:
    parts, depth, cur = [], 0, ''
    for c in text:
        if c in '([{<':
            depth += 1
        elif c in ')]}>':
            depth -= 1
        if c == ',' and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += c
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _prefix_state(prefix: str):
    """('code' | 'comment' | 'str' | 'fexpr' | 'attr', start offset) of the end of a line prefix."""
    state, quote, fstr, brace = 'code', None, False, 0
    start = 0
    i, n = 0, len(prefix)
    while i < n:
        c = prefix[i]
        if state == 'code':
            if c == '#':
                if prefix[i + 1:i + 2] == '[':
                    state, start, brace = 'attr', i, 1
                    i += 2
                    continue
                return 'comment', i
            if c in '"\'':
                fstr = i > 0 and prefix[i - 1] == 'f' and (i < 2 or not (prefix[i - 2].isalnum() or
                                                                         prefix[i - 2] == '_'))
                state, quote, start = 'str', c, i
        elif state == 'str':
            if c == '\\':
                i += 2
                continue
            if c == quote:
                state = 'code'
            elif fstr and c == '{':
                if prefix[i + 1:i + 2] == '{':
                    i += 2
                    continue
                state, brace, start = 'fexpr', 1, i + 1
        elif state == 'fexpr':
            if c == '{':
                brace += 1
            elif c == '}':
                brace -= 1
                if brace == 0:
                    state = 'str'
            elif c == quote:
                state = 'code'
        elif state == 'attr':
            if c == '[':
                brace += 1
            elif c == ']':
                brace -= 1
                if brace == 0:
                    state = 'code'
        i += 1
    return state, start


def _is_type_position(before: str, languages) -> bool:
    """Is the cursor where a type is written (`x: |`, `-> |`, `Option<|`)?"""
    s = before.rstrip()
    if s.endswith('->'):
        return True
    if s.endswith('<') and re.search(r'(Option|Result|Vec|HashMap)\s*<$', s):
        return True
    if not s.endswith(':') or s.endswith('::'):
        return False
    head = s[:-1].rstrip()
    m = re.search(r'([^\W\d]\w*)$', head)
    if not m:
        return False
    rest = head[:m.start()].rstrip()
    if rest.endswith('(') or rest.endswith(',') or rest.endswith('|'):
        # a parameter (but not a struct initializer / dict on one line)
        return not re.search(r'[{]\s*[^}]*$', rest)
    words = rest.split()
    last = words[-1] if words else ''
    if last in KEYWORD_WORDS and KEYWORD_WORDS[last][0] in ('LET', 'VAR', 'CONST', 'MUT'):
        return True
    if not rest.strip():
        return True             # a struct field line
    return False


def _open_call(text: str, off: int) -> Optional[Tuple[int, int]]:
    """(offset of the unmatched '(' before `off`, index of the argument) or None."""
    depth = 0
    commas = 0
    i = off - 1
    lines = 0
    limit = max(0, off - 4000)
    while i >= limit:
        c = text[i]
        if c == '\n':
            lines += 1
            if lines > 30:
                return None
        elif c in '"\'':
            # skip a string backwards (same line)
            j = i - 1
            while j >= 0 and text[j] != c and text[j] != '\n':
                j -= 1
            if j >= 0 and text[j] == c:
                i = j - 1
                continue
        elif c in ')]}':
            depth += 1
        elif c in '([{':
            if depth == 0:
                if c == '(':
                    return i, commas
                return None
            depth -= 1
        elif c == ',' and depth == 0:
            commas += 1
        elif c == '#' and _in_comment(text, i):
            pass
        i -= 1
    return None


def _in_comment(text: str, i: int) -> bool:
    ls = text.rfind('\n', 0, i) + 1
    state, _ = _prefix_state(text[ls:i + 1])
    return state == 'comment'


def _callee_before(text: str, paren: int) -> Optional[Tuple[Optional[str], str]]:
    """(receiver text or None, name) of the call whose '(' is at `paren`."""
    j = paren
    while j > 0 and text[j - 1] in ' \t':
        j -= 1
    k = j
    while k > 0 and (text[k - 1].isalnum() or text[k - 1] == '_'):
        k -= 1
    name = text[k:j]
    if not name or name[0].isdigit():
        return None
    if k > 0 and text[k - 1] == '.':
        ls = text.rfind('\n', 0, k) + 1
        return text[ls:k - 1], name
    return None, name


def _receiver_chain(recv_text: str) -> List[Tuple[str, Any]]:
    """The postfix chain that ends a receiver text: [('name', 'p'), ('name', 'pos'),
    ('call',), ('index',), ...] (read backwards from the end, tokens of the line)."""
    try:
        toks = til.Lexer(recv_text, "<recv>").tokenize()
    except Exception:
        toks = _rough_tokens(recv_text)
    toks = [t for t in toks if t.type not in LAYOUT]
    items: List[Tuple[str, Any]] = []
    k = len(toks) - 1
    while k >= 0:
        t = toks[k]
        if t.type in (TT.RPAREN, TT.RBRACKET):
            want = TT.LPAREN if t.type == TT.RPAREN else TT.LBRACKET
            depth = 0
            j = k
            while j >= 0:
                if toks[j].type == t.type:
                    depth += 1
                elif toks[j].type == want:
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            if j < 0:
                break
            prev = toks[j - 1] if j > 0 else None
            operand = prev is not None and prev.type in (TT.IDENT, TT.SELF, TT.RPAREN, TT.RBRACKET, TT.STRING)
            if t.type == TT.RBRACKET and not operand:
                items.append(('lit', 'array'))
                break
            if t.type == TT.RPAREN and not operand:
                break
            items.append(('call',) if t.type == TT.RPAREN else ('index',))
            k = j - 1
            continue
        if t.type == TT.QUESTION:
            items.append(('try',))
            k -= 1
            continue
        if t.type in (TT.IDENT, TT.SELF):
            items.append(('name', 'self' if t.type == TT.SELF else t.value))
            k -= 1
            if k >= 0 and toks[k].type == TT.DOT:
                k -= 1
                continue
            break
        if t.type in (TT.STRING, TT.FSTRING):
            items.append(('lit', 'str'))
            break
        if t.type == TT.INT:
            items.append(('lit', 'int'))
            break
        if t.type == TT.FLOAT:
            items.append(('lit', 'float'))
            break
        break
    items.reverse()
    return items


class _RoughTok:
    def __init__(self, type_, value, start, end):
        self.type, self.value, self.start, self.end = type_, value, start, end
        self.line, self.col = 1, start + 1


def _rough_tokens(text: str) -> List[_RoughTok]:
    out = []
    for m in re.finditer(r'[^\W\d]\w*|\d+(?:\.\d+)?|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|\S', text):
        s = m.group(0)
        if s[0].isalpha() or s[0] == '_':
            tt = TT.SELF if s == 'self' else TT.IDENT
        elif s[0].isdigit():
            tt = TT.FLOAT if '.' in s else TT.INT
        elif s[0] in '"\'':
            tt = TT.STRING
        else:
            tt = {'(': TT.LPAREN, ')': TT.RPAREN, '[': TT.LBRACKET, ']': TT.RBRACKET, '.': TT.DOT,
                  '?': TT.QUESTION}.get(s, TT.COMMA)
        out.append(_RoughTok(tt, s, m.start(), m.end()))
    return out


_LET_WORDS = {w for c in KEYWORD_LANGS for n in ('LET', 'VAR', 'CONST') for w in KEYWORDS_BY_TYPE.get(n, {}).get(c, [])}
_FOR_WORDS = {w for c in KEYWORD_LANGS for w in KEYWORDS_BY_TYPE.get('FOR', {}).get(c, [])}


def _scan_locals(text: str, off: int) -> List[Tuple[str, str, Optional[Any]]]:
    """Token-scan fallback for broken code: (name, kind, syntactic type) of the
    parameters of the definition around `off` and the let/var/const/for names
    above `off` in it."""
    ls = text.rfind('\n', 0, off) + 1
    lines = text[:ls].split('\n')[:-1]
    cur_line = text[ls:text.find('\n', off) if text.find('\n', off) >= 0 else len(text)]
    cur_w = _indent_width(cur_line) if cur_line.strip() else _indent_width(text[ls:off])
    out: List[Tuple[str, str, Optional[Any]]] = []
    seen = set()
    k = len(lines) - 1
    let_alt = '|'.join(re.escape(w) for w in sorted(_LET_WORDS, key=len, reverse=True))
    for_alt = '|'.join(re.escape(w) for w in sorted(_FOR_WORDS, key=len, reverse=True))
    rx_let = re.compile(r'^\s*(?:%s)\s+([^\W\d]\w*)\s*(?::\s*([^=]+?))?\s*(?:=|$)' % let_alt)
    rx_for = re.compile(r'^\s*(?:%s)\s+([^\W\d]\w*)\b' % for_alt)
    rx_head = re.compile(r'^\s*(?:(?:fn|функция|funktsiya|pub)\s+)*([^\W\d]\w*)\s*(?:<[^>]*>)?\s*\((.*)\)\s*(?:->.*)?$')
    while k >= 0:
        ln = lines[k]
        if ln.strip() and not ln.strip().startswith('#'):
            w = _indent_width(ln)
            m = rx_let.match(ln) or rx_for.match(ln)
            if m and w <= cur_w:
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    typ = None
                    if m.re is rx_let and m.lastindex and m.lastindex >= 2 and m.group(2):
                        typ = _rough_type(m.group(2).strip())
                    out.append((name, 'local', typ))
            if w == 0 or w < cur_w:
                h = rx_head.match(ln)
                if h:
                    for p in _split_params(h.group(2)):
                        pm = re.match(r'(?:mut\s+)?([^\W\d]\w*)\s*(?::\s*(.+?))?\s*(?:=.*)?$', p)
                        if pm and pm.group(1) not in seen:
                            seen.add(pm.group(1))
                            out.append((pm.group(1), 'param', _rough_type(pm.group(2)) if pm.group(2) else None))
                    if w == 0:
                        break
                if w == 0:
                    break
                cur_w = min(cur_w, w)
        k -= 1
    return out


def _rough_type(text: Optional[str]):
    if not text:
        return None
    try:
        toks = til.Lexer(text, "<type>").tokenize()
        return _desc(til.Parser(toks, "<type>").parse_type())
    except Exception:
        return None


def _line_edits(doc: Document, new_text: str) -> List[dict]:
    old_lines = doc.text.splitlines(True)
    new_lines = new_text.splitlines(True)
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    edits = []
    starts = []
    pos = 0
    for ln in old_lines:
        starts.append(pos)
        pos += len(ln)
    starts.append(pos)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            continue
        a = starts[i1]
        b = starts[i2]
        text = ''.join(new_lines[j1:j2])
        edits.append({'range': doc.pmap.range(a, b), 'newText': text.replace('\n', doc.eol)})
    return edits


# ═══════════════════════════════════════════════════════════════════════════════
#                     FORMATTER (token based, verified by re-lexing)
# ═══════════════════════════════════════════════════════════════════════════════

_BINARY_OPS = frozenset((TT.PLUS, TT.MINUS, TT.STAR, TT.SLASH, TT.PERCENT, TT.POWER, TT.EQEQ, TT.NEQ, TT.LT,
                         TT.GT, TT.LTE, TT.GTE, TT.EQ, TT.PLUSEQ, TT.MINUSEQ, TT.STAREQ, TT.SLASHEQ,
                         TT.PERCENTEQ, TT.ARROW, TT.FAT_ARROW, TT.PIPE_GT, TT.CARET, TT.AMP))
_MAYBE_UNARY = frozenset((TT.MINUS, TT.STAR, TT.AMP))
_OPERAND_END = frozenset((TT.IDENT, TT.INT, TT.FLOAT, TT.STRING, TT.FSTRING, TT.CHAR, TT.BOOL, TT.SELF,
                          TT.RPAREN, TT.RBRACKET, TT.RBRACE, TT.QUESTION))
_GENERIC_INSIDE = frozenset((TT.IDENT, TT.COMMA, TT.LBRACKET, TT.RBRACKET, TT.SEMICOLON, TT.INT, TT.STAR,
                             TT.MUT, TT.FN, TT.LPAREN, TT.RPAREN, TT.ARROW, TT.LT, TT.GT))


def _lex_for_format(text: str):
    lx = _Lexer(text, "<format>")
    try:
        with _Capture():
            toks = lx.tokenize()
    except Exception:
        return None, None
    return toks, lx


def _without_trailing_newlines(toks):
    """Tokens without the NEWLINEs that only precede the final DEDENTs / EOF
    (a file with or without a final newline lexes the same otherwise)."""
    end = len(toks)
    while end > 0 and toks[end - 1].type in (TT.DEDENT, TT.EOF):
        end -= 1
    k = end
    while k > 0 and toks[k - 1].type == TT.NEWLINE:
        k -= 1
    return toks[:k] + toks[end:]


def _same_tokens(a, b) -> bool:
    a = _without_trailing_newlines(a)
    b = _without_trailing_newlines(b)
    if len(a) != len(b):
        return False
    for t1, t2 in zip(a, b):
        if t1.type != t2.type:
            return False
        if t1.type in (TT.INDENT, TT.DEDENT):
            continue
        if t1.value != t2.value:
            return False
    return True


def format_source(text: str, tab_size: int = 4) -> Optional[str]:
    """The formatted text, or None when it cannot be formatted safely (it does
    not lex, or the result would not lex to the very same tokens)."""
    try:
        tab_size = max(1, min(int(tab_size or 4), 8))
    except (TypeError, ValueError):
        tab_size = 4
    norm = text.replace('\r\n', '\n').replace('\r', '\n')
    if not norm.strip():
        return norm
    toks, lx = _lex_for_format(norm)
    if toks is None:
        return None
    lines = norm.split('\n')
    line_starts = [0]
    for ln in lines[:-1]:
        line_starts.append(line_starts[-1] + len(ln) + 1)
    n = len(toks)
    # generic brackets (Vec<int>, Option<T>, max_of<T>): never spaced
    generic: Set[int] = set()
    for i, t in enumerate(toks):
        if t.type != TT.LT or i == 0 or toks[i - 1].type != TT.IDENT or toks[i - 1].end != t.start:
            continue
        depth, j, ok = 1, i + 1, False
        while j < n:
            tt = toks[j].type
            if tt == TT.LT:
                depth += 1
            elif tt == TT.GT:
                depth -= 1
                if depth == 0:
                    ok = True
                    break
            elif tt == TT.SHR:
                depth -= 2
                ok = depth == 0
                break
            elif tt not in _GENERIC_INSIDE:
                break
            j += 1
        if ok:
            for k in range(i, j + 1):
                if toks[k].type in (TT.LT, TT.GT, TT.SHR):
                    generic.add(k)
    unary: Set[int] = set()
    for i, t in enumerate(toks):
        if t.type in _MAYBE_UNARY:
            prev = toks[i - 1] if i > 0 else None
            if prev is None or prev.type in LAYOUT or prev.type not in _OPERAND_END:
                unary.add(i)
    # tokens per line, and the indentation level of logical line starts
    by_line: Dict[int, List[int]] = {}
    level: Dict[int, Optional[int]] = {}
    depth = 0
    pdepth = 0
    for i, t in enumerate(toks):
        tt = t.type
        if tt == TT.INDENT:
            depth += 1
            continue
        if tt == TT.DEDENT:
            depth = max(0, depth - 1)
            continue
        if tt in (TT.NEWLINE, TT.EOF) or t.end <= t.start:
            continue
        ln = t.line - 1
        if ln not in by_line:
            by_line[ln] = []
            level[ln] = depth if pdepth == 0 else None
        by_line[ln].append(i)
        if tt in OPENERS:
            pdepth += 1
        elif tt in CLOSERS:
            pdepth = max(0, pdepth - 1)
    old_w: Dict[int, int] = {}
    new_w: Dict[int, int] = {}
    last_logical = None
    for ln in sorted(by_line):
        line = lines[ln]
        old = _indent_width(line)
        old_w[ln] = old
        if level[ln] is not None:
            new_w[ln] = level[ln] * tab_size
            last_logical = ln
        elif last_logical is not None:
            new_w[ln] = max(0, old + new_w[last_logical] - old_w[last_logical])
        else:
            new_w[ln] = old
    code_lines = sorted(by_line)
    out_lines: List[str] = []

    def gap_between(a: int, b: int, gap: str) -> str:
        if '\n' in gap or '#' in gap:
            return gap
        ta, tb = toks[a], toks[b]
        if tb.type == TT.COMMA:
            return '' if ta.type not in OPENERS else gap
        if ta.type == TT.COMMA:
            return gap if tb.type in CLOSERS else ' '

        def binary(k):
            t = toks[k]
            return t.type in _BINARY_OPS and k not in unary and k not in generic
        if binary(a) or binary(b):
            return ' '
        return gap

    for ln, line in enumerate(lines):
        if ln in by_line:
            idxs = by_line[ln]
            first = toks[idxs[0]]
            lead = norm[line_starts[ln]:first.start]
            if lead.strip():                  # something without a token before (a pragma): keep the line
                out_lines.append(line.rstrip())
                continue
            out = ' ' * new_w[ln]
            prev = None
            for i in idxs:
                t = toks[i]
                if prev is not None:
                    out += gap_between(prev, i, norm[toks[prev].end:t.start])
                out += norm[t.start:t.end]
                prev = i
            end_of_line = line_starts[ln] + len(line)
            tail = norm[toks[prev].end:end_of_line]
            out += tail.rstrip()
            out_lines.append(out.rstrip())
        else:
            s = line.strip()
            if not s:
                out_lines.append('')
                continue
            w = _indent_width(line)
            nxt = next((c for c in code_lines if c > ln), None)
            prv = next((c for c in reversed(code_lines) if c < ln), None)
            if nxt is not None and old_w.get(nxt) == w:
                w = new_w[nxt]
            elif prv is not None and old_w.get(prv) == w:
                w = new_w[prv]
            else:
                out_lines.append(line.rstrip())
                continue
            out_lines.append(' ' * w + s)
    # blank lines: at most two in a row, none at the end; one final newline
    result_lines: List[str] = []
    blanks = 0
    for ln in out_lines:
        if ln == '':
            blanks += 1
            if blanks > 2:
                continue
        else:
            blanks = 0
        result_lines.append(ln)
    while result_lines and result_lines[-1] == '':
        result_lines.pop()
    result = '\n'.join(result_lines) + '\n'
    # verify: the very same tokens (indentation widths aside), comments and languages
    toks2, lx2 = _lex_for_format(result)
    if toks2 is None or not _same_tokens(toks, toks2):
        return None
    c1 = [norm[a:b].rstrip() for a, b, _k in lx.comments]
    c2 = [result[a:b].rstrip() for a, b, _k in lx2.comments]
    if c1 != c2 or lx.languages != lx2.languages:
        return None
    return result
