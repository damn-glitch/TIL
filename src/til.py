#!/usr/bin/env python3
"""
TIL v2.0 - The Intelligent Language
====================================
Complete Multi-Level Programming Language

Author: Alisher Beisembekov
Version: 2.0.0

"Проще Python. Быстрее C. Умнее всех."
(Simpler than Python. Faster than C. Smarter than all.)

MULTI-LEVEL ARCHITECTURE:
  Level 0: Hardware  - SIMD, inline assembly, direct memory
  Level 1: Systems   - Like C, manual memory, pointers
  Level 2: Safe      - Like Rust, bounds checking, null safety [DEFAULT]
  Level 3: Script    - Like Python, dynamic, GC
  Level 4: Formal    - Contracts, invariants, proofs

This is Mixed Martial Programming - use the right level for each task.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
from abc import ABC, abstractmethod

TIL_VERSION = "3.1.0"

# ═══════════════════════════════════════════════════════════════════════════════
#                              ERROR REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorReporter:
    """Beautiful error messages with source context and hints."""

    # ANSI colors (used when output is a tty)
    RESET = "\033[0m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def __init__(self, source: str = "", filename: str = "<stdin>", use_color: bool = True):
        self.source = source
        self.filename = filename
        self.lines = source.split('\n') if source else []
        self.use_color = use_color and sys.stderr.isatty()

    def _c(self, code: str, text: str) -> str:
        """Wrap text in ANSI color if color is enabled."""
        if self.use_color:
            return f"{code}{text}{self.RESET}"
        return text

    def format_error(self, line: int, col: int, msg: str,
                     error_code: str = "", hint: str = "") -> str:
        """Format a beautiful error message with source context."""
        parts = []

        # Error header
        code_str = f"[{error_code}]" if error_code else ""
        parts.append(self._c(self.BOLD + self.RED, f"error{code_str}: {msg}"))

        # Location
        parts.append(self._c(self.CYAN, f" --> {self.filename}:{line}:{col}"))

        # Source context
        if 1 <= line <= len(self.lines):
            line_num_width = len(str(line + 1))
            pad = " " * line_num_width

            # Line before (context)
            if line >= 2:
                prev = self.lines[line - 2]
                parts.append(self._c(self.DIM, f"  {line-1:>{line_num_width}} | {prev}"))

            # Error line
            error_line = self.lines[line - 1]
            parts.append(f"  {self._c(self.CYAN, f'{line:>{line_num_width}}')} | {error_line}")

            # Pointer
            pointer = " " * max(0, col - 1) + "^"
            parts.append(f"  {pad} | {self._c(self.RED, pointer)}")

            # Line after (context)
            if line < len(self.lines):
                nxt = self.lines[line]
                parts.append(self._c(self.DIM, f"  {line+1:>{line_num_width}} | {nxt}"))

        # Hint
        if hint:
            parts.append(self._c(self.YELLOW, f"  = hint: {hint}"))

        return "\n".join(parts)

    def format_warning(self, line: int, col: int, msg: str, hint: str = "") -> str:
        """Format a warning message."""
        parts = []
        parts.append(self._c(self.BOLD + self.YELLOW, f"warning: {msg}"))
        parts.append(self._c(self.CYAN, f" --> {self.filename}:{line}:{col}"))

        if 1 <= line <= len(self.lines):
            error_line = self.lines[line - 1]
            line_num_width = len(str(line))
            parts.append(f"  {self._c(self.CYAN, f'{line:>{line_num_width}}')} | {error_line}")
            pointer = " " * max(0, col - 1) + "^"
            parts.append(f"  {' ' * line_num_width} | {self._c(self.YELLOW, pointer)}")

        if hint:
            parts.append(self._c(self.YELLOW, f"  = hint: {hint}"))

        return "\n".join(parts)


# Common error hints
ERROR_HINTS = {
    "Expected IDENT": "Variable or function name expected here",
    "Expected RPAREN": "Missing closing parenthesis ')'",
    "Expected RBRACKET": "Missing closing bracket ']'",
    "Expected RBRACE": "Missing closing brace '}'",
    "Expected INDENT": "Expected an indented block (use 4 spaces)",
    "Unexpected token: NEWLINE": "Unexpected end of line. Check for missing operators or parentheses",
    "Unexpected character": "This character is not valid in TIL",
}

def get_hint_for_error(msg: str) -> str:
    """Get a helpful hint for a given error message."""
    for pattern, hint in ERROR_HINTS.items():
        if pattern in msg:
            return hint
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
#                                  TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

class TokenType(Enum):
    # Literals
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    FSTRING = auto()  # f-string interpolation
    CHAR = auto()
    BOOL = auto()
    
    # Identifiers & Keywords
    IDENT = auto()
    
    # Keywords
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    FOR = auto()
    WHILE = auto()
    LOOP = auto()
    IN = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    FN = auto()
    LET = auto()
    VAR = auto()
    CONST = auto()
    MUT = auto()
    PUB = auto()
    STRUCT = auto()
    ENUM = auto()
    IMPL = auto()
    TRAIT = auto()
    MATCH = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    TYPE = auto()
    SELF = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    POWER = auto()
    
    EQ = auto()
    EQEQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    
    PLUSEQ = auto()
    MINUSEQ = auto()
    STAREQ = auto()
    SLASHEQ = auto()
    PERCENTEQ = auto()
    PIPE_GT = auto()   # |>  pipeline
    
    ARROW = auto()
    FAT_ARROW = auto()
    RANGE = auto()
    RANGE_INCL = auto()
    
    AMP = auto()
    PIPE = auto()
    CARET = auto()
    TILDE = auto()
    SHL = auto()
    SHR = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    DOT = auto()
    QUESTION = auto()
    AT = auto()
    HASH = auto()
    
    # Special
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()
    
    # V2.0 Advanced Keywords
    ASSERT = auto()
    INVARIANT = auto()
    INTENT = auto()
    RETRY = auto()
    FALLBACK = auto()
    TIMEOUT = auto()
    CAP = auto()
    PERFORM = auto()
    HANDLE = auto()
    WITH = auto()      # handle ... with <clauses>   (TIL 3.1 effects)
    RESUME = auto()    # resume(v) inside an effect clause

    # Attributes
    ATTRIBUTE = auto()

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    col: int
    start: int = -1   # offset of the first character in the source
    end: int = -1     # offset just past the last character

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"

# ═══════════════════════════════════════════════════════════════════════════════
#                                  LEXER
# ═══════════════════════════════════════════════════════════════════════════════

KEYWORDS = {
    'if': TokenType.IF, 'else': TokenType.ELSE, 'elif': TokenType.ELIF,
    'for': TokenType.FOR, 'while': TokenType.WHILE, 'loop': TokenType.LOOP,
    'in': TokenType.IN, 'return': TokenType.RETURN, 'break': TokenType.BREAK,
    'continue': TokenType.CONTINUE, 'fn': TokenType.FN, 'let': TokenType.LET,
    'var': TokenType.VAR, 'const': TokenType.CONST, 'mut': TokenType.MUT,
    'pub': TokenType.PUB, 'struct': TokenType.STRUCT, 'enum': TokenType.ENUM,
    'impl': TokenType.IMPL, 'trait': TokenType.TRAIT, 'match': TokenType.MATCH,
    'import': TokenType.IMPORT, 'from': TokenType.FROM, 'as': TokenType.AS,
    'type': TokenType.TYPE, 'self': TokenType.SELF,
    'and': TokenType.AND, 'or': TokenType.OR, 'not': TokenType.NOT,
    'true': TokenType.BOOL, 'false': TokenType.BOOL,
    'True': TokenType.BOOL, 'False': TokenType.BOOL,
    # V2.0 advanced keywords
    'assert': TokenType.ASSERT, 'invariant': TokenType.INVARIANT,
    'intent': TokenType.INTENT, 'retry': TokenType.RETRY,
    'fallback': TokenType.FALLBACK, 'timeout': TokenType.TIMEOUT,
    'cap': TokenType.CAP, 'perform': TokenType.PERFORM, 'handle': TokenType.HANDLE,
    # TIL 3.1 `with` / `resume` are contextual: see CONTEXTUAL_KEYWORDS
}

# Казахские ключевые слова (Kazakh language keywords)
# TIL supports writing code in Kazakh — Patent № 66853
KAZAKH_KEYWORDS = {
    'егер': TokenType.IF,         # if
    'әйтпесе': TokenType.ELSE,   # else
    'немесе': TokenType.ELIF,     # elif
    'үшін': TokenType.FOR,        # for
    'кезінде': TokenType.WHILE,   # while
    'цикл': TokenType.LOOP,       # loop
    'ішінде': TokenType.IN,       # in
    'қайтару': TokenType.RETURN,  # return
    'тоқтату': TokenType.BREAK,   # break
    'жалғастыру': TokenType.CONTINUE,  # continue
    'функция': TokenType.FN,      # fn
    'тұрақты': TokenType.LET,     # let (immutable)
    'айнымалы': TokenType.VAR,    # var (mutable)
    'тұрақтама': TokenType.CONST, # const
    'өзгермелі': TokenType.MUT,   # mut
    'ашық': TokenType.PUB,        # pub
    'құрылым': TokenType.STRUCT,  # struct
    'санақ': TokenType.ENUM,      # enum
    'іске': TokenType.IMPL,       # impl
    'қасиет': TokenType.TRAIT,    # trait
    'сәйкестік': TokenType.MATCH, # match
    'импорт': TokenType.IMPORT,   # import
    'бастап': TokenType.FROM,     # from
    'ретінде': TokenType.AS,      # as
    'тип': TokenType.TYPE,        # type
    'өзін': TokenType.SELF,       # self
    'және': TokenType.AND,        # and
    'немесе': TokenType.OR,       # or   ("elif" is written "әйтпесе егер")
    'емес': TokenType.NOT,        # not
    'ақиқат': TokenType.BOOL,     # true
    'жалған': TokenType.BOOL,     # false
    'тексеру': TokenType.ASSERT,  # assert
    'қайталау': TokenType.RETRY,  # retry
    'ниет': TokenType.INTENT,     # intent
    # effects (TIL 3.1): contextual, see CONTEXTUAL_KEYWORDS
}
# NOTE: 'немесе' used to be registered twice (ELIF and OR) in one dict literal,
# so it silently became OR and Kazakh `elif` never worked. TIL v3 writes elif
# as "else if" in every language: `әйтпесе егер`, `иначе если`, `aitpese eger`.

# Kazakh in ASCII Latin transliteration (keyboard / calculator friendly).
# Enabled per file with the pragma `#[lang: kk-latn]`.
KAZAKH_LATIN_KEYWORDS = {
    'eger': TokenType.IF, 'aitpese': TokenType.ELSE, 'ushin': TokenType.FOR,
    'kezinde': TokenType.WHILE, 'tsikl': TokenType.LOOP, 'ishinde': TokenType.IN,
    'qaitaru': TokenType.RETURN, 'toqtatu': TokenType.BREAK,
    'zhalgastyru': TokenType.CONTINUE, 'funktsiya': TokenType.FN,
    'turaqty': TokenType.LET, 'aynymaly': TokenType.VAR,
    'turaqtama': TokenType.CONST, 'ozgermeli': TokenType.MUT,
    'ashyq': TokenType.PUB, 'qurylym': TokenType.STRUCT, 'sanaq': TokenType.ENUM,
    'iske': TokenType.IMPL, 'qasiet': TokenType.TRAIT,
    'saikestik': TokenType.MATCH, 'bastap': TokenType.FROM,
    'retinde': TokenType.AS, 'ozin': TokenType.SELF, 'zhane': TokenType.AND,
    'nemese': TokenType.OR, 'emes': TokenType.NOT, 'aqiqat': TokenType.BOOL,
    'zhalgan': TokenType.BOOL, 'tekseru': TokenType.ASSERT,
    'qaitalau': TokenType.RETRY, 'niet': TokenType.INTENT,
    # effects (TIL 3.1): contextual, see CONTEXTUAL_KEYWORDS
}

# Russian keywords. Enabled per file with the pragma `#[lang: ru]`
# (opt-in, because short words like "и"/"в" would otherwise collide with
# identifiers in Kazakh programs).
RUSSIAN_KEYWORDS = {
    'если': TokenType.IF, 'иначе': TokenType.ELSE, 'для': TokenType.FOR,
    'пока': TokenType.WHILE, 'цикл': TokenType.LOOP, 'в': TokenType.IN,
    'вернуть': TokenType.RETURN, 'прервать': TokenType.BREAK,
    'продолжить': TokenType.CONTINUE, 'функция': TokenType.FN,
    'пусть': TokenType.LET, 'перем': TokenType.VAR, 'конст': TokenType.CONST,
    'изм': TokenType.MUT, 'публ': TokenType.PUB, 'структура': TokenType.STRUCT,
    'перечисление': TokenType.ENUM, 'реализация': TokenType.IMPL,
    'типаж': TokenType.TRAIT, 'выбор': TokenType.MATCH, 'импорт': TokenType.IMPORT,
    'из': TokenType.FROM, 'как': TokenType.AS, 'тип': TokenType.TYPE,
    'сам': TokenType.SELF, 'и': TokenType.AND, 'или': TokenType.OR,
    'не': TokenType.NOT, 'истина': TokenType.BOOL, 'ложь': TokenType.BOOL,
    'проверить': TokenType.ASSERT, 'инвариант': TokenType.INVARIANT,
    'повторить': TokenType.RETRY, 'таймаут': TokenType.TIMEOUT,
    'намерение': TokenType.INTENT,
    # effects (TIL 3.1): contextual, see CONTEXTUAL_KEYWORDS
}

# TIL 3.1 effect words are *contextual keywords* (docs/TIL_EFFECTS.md): the lexer
# reads them as identifiers and the parser gives them their keyword meaning only
# where an identifier could not stand in a TIL 3.0 program, so no 3.0 program
# changes meaning (variables, functions, methods and fields may still be called
# `with`, `resume`, `с`, `орындау`, ...):
#   WITH     right after the body of a `handle`
#   RESUME   inside an effect clause (there it always means resume)
#   PERFORM  followed by an identifier on the same line (the effect name)
#   HANDLE   followed by a newline and an indented body, or on the same line by
#            an identifier, a literal or perform/handle/match/retry/timeout
# English `perform` and `handle` have been hard keywords since TIL 3.0.
CONTEXTUAL_KEYWORDS = {
    'en': {'with': TokenType.WITH, 'resume': TokenType.RESUME},
    'kk': {'орындау': TokenType.PERFORM, 'өңдеу': TokenType.HANDLE,
           'арқылы': TokenType.WITH, 'жалғау': TokenType.RESUME},
    'kk-latn': {'oryndau': TokenType.PERFORM, 'ondeu': TokenType.HANDLE,
                'arqyly': TokenType.WITH, 'zhalgau': TokenType.RESUME},
    'ru': {'выполнить': TokenType.PERFORM, 'обработать': TokenType.HANDLE,
           'с': TokenType.WITH, 'возобновить': TokenType.RESUME},
}
# every spelling of `handle` (the REPL reads a multi-line block after one) and resume
HANDLE_WORDS = ('handle',) + tuple(w for tbl in CONTEXTUAL_KEYWORDS.values()
                                   for w, t in tbl.items() if t == TokenType.HANDLE)
RESUME_WORDS = frozenset(w for tbl in CONTEXTUAL_KEYWORDS.values()
                         for w, t in tbl.items() if t == TokenType.RESUME)

# Values of localized boolean literals.
TRUE_WORDS = {'true', 'True', 'ақиқат', 'aqiqat', 'истина'}

# Builtin-name aliases: localized spellings of builtin functions and of the
# program entry point. The lexer rewrites them to the canonical name, so the
# rest of the compiler (and both backends) only ever see `print`, `main`, ...
KAZAKH_ALIASES = {
    'басып_шығару': 'print', 'шығару': 'print', 'негізгі': 'main',
    'ұзындық': 'len', 'енгізу': 'input',
}
KAZAKH_LATIN_ALIASES = {
    'basyp_shygaru': 'print', 'shygaru': 'print', 'negizgi': 'main',
    'uzyndyq': 'len', 'engizu': 'input',
}
RUSSIAN_ALIASES = {
    'печать': 'print', 'вывод': 'print', 'главная': 'main',
    'длина': 'len', 'ввод': 'input',
}

# language code -> (keyword table, alias table)
LANGUAGES = {
    'en': ({}, {}),
    'kk': (KAZAKH_KEYWORDS, KAZAKH_ALIASES),
    'kk-latn': (KAZAKH_LATIN_KEYWORDS, KAZAKH_LATIN_ALIASES),
    'ru': (RUSSIAN_KEYWORDS, RUSSIAN_ALIASES),
}
DEFAULT_LANGUAGES = ('en', 'kk')   # Cyrillic Kazakh is always on (backward compatible)

_LANG_PRAGMA_RE = re.compile(r'^[ \t]*#\[\s*lang\s*:\s*([^\]]*)\]', re.MULTILINE)

def detect_languages(source: str) -> List[str]:
    """Languages enabled for a source file: defaults + `#[lang: ...]` pragmas."""
    langs = list(DEFAULT_LANGUAGES)
    for m in _LANG_PRAGMA_RE.finditer(source):
        for code in m.group(1).split(','):
            code = code.strip().lower()
            if code in LANGUAGES and code not in langs:
                langs.append(code)
    return langs

ENGLISH_KEYWORDS = dict(KEYWORDS)

# Merge Kazakh keywords into the main keyword map (backward compatible)
KEYWORDS.update(KAZAKH_KEYWORDS)

class Lexer:
    def __init__(self, source: str, filename: str = "<stdin>",
                 languages: Optional[List[str]] = None, line_offset: int = 0):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1 + line_offset
        self.col = 1
        self.tokens: List[Token] = []
        self.indent_stack = [0]
        self.at_line_start = True
        self.paren_depth = 0  # Track () [] {} depth to suppress NEWLINE inside
        self.error_reporter = ErrorReporter(source, filename)
        # Keyword languages: English + Kazakh by default, more via #[lang: ...]
        self.languages = list(languages) if languages else detect_languages(source)
        self.keywords: Dict[str, TokenType] = dict(ENGLISH_KEYWORDS)
        self.aliases: Dict[str, str] = {}
        for code in self.languages:
            kw, al = LANGUAGES.get(code, ({}, {}))
            self.keywords.update(kw)
            self.aliases.update(al)
        # Start of the token currently being read
        self.tok_start = 0
        self.tok_line = self.line
        self.tok_col = 1

    def error(self, msg: str):
        hint = get_hint_for_error(msg)
        formatted = self.error_reporter.format_error(self.line, self.col, msg, hint=hint)
        raise SyntaxError(formatted)
    
    def current(self) -> str:
        return self.source[self.pos] if self.pos < len(self.source) else '\0'
    
    def peek(self, n: int = 1) -> str:
        pos = self.pos + n
        return self.source[pos] if pos < len(self.source) else '\0'
    
    def advance(self) -> str:
        ch = self.current()
        self.pos += 1
        self.col += 1
        return ch
    
    def add_token(self, type: TokenType, value: Any = None):
        if type in (TokenType.INDENT, TokenType.DEDENT, TokenType.EOF):
            tok = Token(type, value, self.line, self.col, self.pos, self.pos)
        else:
            tok = Token(type, value, self.tok_line, self.tok_col, self.tok_start, self.pos)
        self.tokens.append(tok)
        self.at_line_start = False

    def mark_token_start(self):
        self.tok_start = self.pos
        self.tok_line = self.line
        self.tok_col = self.col

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            if self.at_line_start and self.paren_depth == 0:
                self.handle_indentation()
                if self.pos >= len(self.source):
                    break  # trailing blank lines / comments at end of file
            elif self.at_line_start and self.paren_depth > 0:
                # Inside parens: skip whitespace but don't emit INDENT/DEDENT
                self.at_line_start = False
                while self.pos < len(self.source) and self.current() in ' \t':
                    self.advance()
                if self.pos >= len(self.source):
                    break

            ch = self.current()
            self.mark_token_start()
            
            # Whitespace (not newline)
            if ch in ' \t' and not self.at_line_start:
                self.advance()
                continue
            
            # Comments
            if ch == '#':
                if self.peek() == '[':
                    self.read_attribute()
                else:
                    self.skip_comment()
                continue
            
            # Newline
            if ch == '\n':
                if self.paren_depth == 0:
                    self.add_token(TokenType.NEWLINE, '\n')
                    self.at_line_start = True
                self.advance()
                self.line += 1
                self.col = 1
                continue
            
            # String / f-string
            if ch == 'f' and self.peek() in '"\'':
                self.advance()  # skip 'f'
                self.read_fstring(self.current())
                continue
            if ch in '"\'':
                self.read_string(ch)
                continue
            
            # Number
            if ch.isdigit():
                self.read_number()
                continue
            
            # Identifier or keyword
            if ch.isalpha() or ch == '_':
                self.read_identifier()
                continue
            
            # Operators and delimiters
            if self.read_operator():
                continue
            
            self.error(f"Unexpected character: '{ch}'")
        
        # Close remaining indents
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.add_token(TokenType.DEDENT)
        
        self.add_token(TokenType.EOF)
        return self.tokens
    
    def handle_indentation(self):
        spaces = 0
        while self.pos < len(self.source):
            ch = self.current()
            if ch == ' ':
                spaces += 1
                self.advance()
            elif ch == '\t':
                spaces += 4
                self.advance()
            elif ch == '\n':
                # Empty line, reset
                spaces = 0
                self.advance()
                self.line += 1
                self.col = 1
            elif ch == '#':
                if self.peek() == '[':
                    break  # Attribute, process normally
                self.skip_comment()
                spaces = 0
            else:
                break
        
        if self.pos >= len(self.source):
            return
        
        current_indent = self.indent_stack[-1]
        
        if spaces > current_indent:
            self.indent_stack.append(spaces)
            self.add_token(TokenType.INDENT, spaces)
        elif spaces < current_indent:
            while len(self.indent_stack) > 1 and self.indent_stack[-1] > spaces:
                self.indent_stack.pop()
                self.add_token(TokenType.DEDENT, spaces)
        
        self.at_line_start = False
    
    def skip_comment(self):
        while self.pos < len(self.source) and self.current() != '\n':
            self.advance()
    
    def read_attribute(self):
        start_line, start_col = self.line, self.col
        self.advance()  # #
        self.advance()  # [
        
        content = []
        depth = 1
        while self.pos < len(self.source) and depth > 0:
            ch = self.current()
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    self.advance()
                    break
            elif ch == '\n':
                self.error("Unterminated attribute")
            content.append(ch)
            self.advance()

        text = ''.join(content).strip()
        if re.match(r'lang\s*:', text):
            return  # `#[lang: ru]` is a lexer pragma (see detect_languages), not an attribute
        self.add_token(TokenType.ATTRIBUTE, text)
    
    def read_string(self, quote: str):
        start_line, start_col = self.line, self.col
        self.advance()  # Opening quote
        
        result = []
        while self.pos < len(self.source):
            ch = self.current()
            if ch == quote:
                self.advance()
                value = ''.join(result)
                # Single-quoted single char = CharLit
                if quote == "'" and len(value) == 1:
                    self.add_token(TokenType.CHAR, value)
                else:
                    self.add_token(TokenType.STRING, value)
                return
            elif ch == '\\':
                result.append(self.read_escape())
            elif ch == '\n':
                self.error("Unterminated string")
            else:
                result.append(ch)
                self.advance()

        self.error("Unterminated string")

    def read_escape(self) -> str:
        """Read an escape sequence starting at a backslash; return its character.
        Supports \\n \\t \\r \\0 \\\\ \\" \\' \\{ \\} \\xNN and \\u{XXXX}."""
        self.advance()  # backslash
        esc = self.current()
        simple = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"',
                  "'": "'", '0': '\0', '{': '{', '}': '}', 'a': '\a', 'b': '\b',
                  'e': '\x1b'}
        if esc == 'x':
            self.advance()
            digits = self.source[self.pos:self.pos + 2]
            if len(digits) < 2 or not all(c in '0123456789abcdefABCDEF' for c in digits):
                self.error("Invalid \\x escape: expected two hex digits")
            self.advance()
            self.advance()
            return chr(int(digits, 16))
        if esc == 'u' and self.peek() == '{':
            self.advance()
            self.advance()
            digits = []
            while self.current() not in '}\n\0':
                digits.append(self.advance())
            if self.current() != '}' or not digits:
                self.error("Invalid \\u{...} escape")
            self.advance()
            try:
                return chr(int(''.join(digits), 16))
            except ValueError:
                self.error("Invalid \\u{...} escape")
        self.advance()
        return simple.get(esc, esc)
    
    def read_fstring(self, quote: str):
        """Parse f"text {expr} text" into STRING_INTERP token with parts list."""
        self.advance()  # opening quote
        parts = []  # list of (type, value): ('str', text) or ('expr', text)
        current_str = []

        while self.pos < len(self.source):
            ch = self.current()
            if ch == quote:
                self.advance()
                if current_str:
                    parts.append(('str', ''.join(current_str)))
                self.add_token(TokenType.FSTRING, parts)
                return
            elif ch == '{' and self.peek() == '{':
                current_str.append('{')       # {{ -> literal {
                self.advance()
                self.advance()
            elif ch == '}' and self.peek() == '}':
                current_str.append('}')       # }} -> literal }
                self.advance()
                self.advance()
            elif ch == '{':
                if current_str:
                    parts.append(('str', ''.join(current_str)))
                    current_str = []
                self.advance()  # skip {
                expr_chars = []
                depth = 1
                while self.pos < len(self.source) and depth > 0:
                    c = self.current()
                    if c == '\n' or c == quote:
                        self.error("Unterminated '{' in f-string")
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            self.advance()
                            break
                    expr_chars.append(c)
                    self.advance()
                text = ''.join(expr_chars)
                # Optional format spec: {value:.2f} {n:5} {n:x}
                m = re.match(r'^(.*?):([<>^]?\d*(?:\.\d+)?[fdxXobe%]?)$', text, re.S)
                if m and m.group(1).strip() and m.group(2):
                    parts.append(('expr', m.group(1), m.group(2)))
                else:
                    parts.append(('expr', text))
            elif ch == '\\':
                current_str.append(self.read_escape())
            elif ch == '\n':
                self.error("Unterminated f-string")
            else:
                current_str.append(ch)
                self.advance()
        self.error("Unterminated f-string")

    def read_number(self):
        start = self.pos
        is_float = False
        
        # Hex / binary / octal
        for prefixes, digits, base, name in (('xX', '0123456789abcdefABCDEF_', 16, 'hexadecimal'),
                                             ('bB', '01_', 2, 'binary'),
                                             ('oO', '01234567_', 8, 'octal')):
            if self.current() == '0' and self.peek() in prefixes:
                self.advance()
                self.advance()
                while self.current() in digits:
                    self.advance()
                text = self.source[start:self.pos]
                try:
                    self.add_token(TokenType.INT, int(text[2:].replace('_', ''), base))
                except ValueError:
                    self.error(f"Invalid {name} literal: '{text}'")
                return

        # Decimal
        while self.current().isdigit() or self.current() == '_':
            self.advance()

        if self.current() == '.' and self.peek().isdigit():
            is_float = True
            self.advance()
            while self.current().isdigit() or self.current() == '_':
                self.advance()

        if self.current() in 'eE' and (self.peek().isdigit() or
                                       (self.peek() in '+-' and self.peek(2).isdigit())):
            is_float = True
            self.advance()
            if self.current() in '+-':
                self.advance()
            while self.current().isdigit():
                self.advance()

        text = self.source[start:self.pos].replace('_', '')
        try:
            if is_float:
                self.add_token(TokenType.FLOAT, float(text))
            else:
                self.add_token(TokenType.INT, int(text))
        except ValueError:
            self.error(f"Invalid numeric literal: '{self.source[start:self.pos]}'")

    def read_identifier(self):
        start = self.pos
        while self.current().isalnum() or self.current() == '_':
            self.advance()

        text = self.source[start:self.pos]

        if text in self.keywords:
            token_type = self.keywords[text]
            if token_type == TokenType.BOOL:
                self.add_token(token_type, text in TRUE_WORDS)
            else:
                self.add_token(token_type, text)
        else:
            # Localized builtin names (басып_шығару -> print, негізгі -> main, ...)
            self.add_token(TokenType.IDENT, self.aliases.get(text, text))
    
    def read_operator(self) -> bool:
        ch = self.current()
        nxt = self.peek()
        
        # Two-character operators
        two_char = ch + nxt
        two_char_ops = {
            '==': TokenType.EQEQ, '!=': TokenType.NEQ,
            '<=': TokenType.LTE, '>=': TokenType.GTE,
            '+=': TokenType.PLUSEQ, '-=': TokenType.MINUSEQ,
            '*=': TokenType.STAREQ, '/=': TokenType.SLASHEQ,
            '->': TokenType.ARROW, '=>': TokenType.FAT_ARROW,
            '..': TokenType.RANGE, '<<': TokenType.SHL, '>>': TokenType.SHR,
            '|>': TokenType.PIPE_GT, '%=': TokenType.PERCENTEQ,
        }
        
        # Three-character (..=)
        if ch == '.' and nxt == '.' and self.peek(2) == '=':
            self.advance()
            self.advance()
            self.advance()
            self.add_token(TokenType.RANGE_INCL, '..=')
            return True
        
        if two_char in two_char_ops:
            self.advance()
            self.advance()
            self.add_token(two_char_ops[two_char], two_char)
            return True
        
        # Power operator **
        if ch == '*' and nxt == '*':
            self.advance()
            self.advance()
            self.add_token(TokenType.POWER, '**')
            return True
        
        # Single-character operators
        one_char_ops = {
            '+': TokenType.PLUS, '-': TokenType.MINUS,
            '*': TokenType.STAR, '/': TokenType.SLASH,
            '%': TokenType.PERCENT, '=': TokenType.EQ,
            '<': TokenType.LT, '>': TokenType.GT,
            '(': TokenType.LPAREN, ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET, ']': TokenType.RBRACKET,
            '{': TokenType.LBRACE, '}': TokenType.RBRACE,
            ',': TokenType.COMMA, ':': TokenType.COLON,
            ';': TokenType.SEMICOLON, '.': TokenType.DOT,
            '&': TokenType.AMP, '|': TokenType.PIPE,
            '^': TokenType.CARET, '~': TokenType.TILDE,
            '?': TokenType.QUESTION, '@': TokenType.AT,
        }
        
        if ch in one_char_ops:
            # Track paren depth for multi-line expression support
            if ch in '([{':
                self.paren_depth += 1
            elif ch in ')]}':
                self.paren_depth = max(0, self.paren_depth - 1)
            self.advance()
            self.add_token(one_char_ops[ch], ch)
            return True
        
        return False

# ═══════════════════════════════════════════════════════════════════════════════
#                                  TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class Type(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    def __eq__(self, other):
        return type(self) == type(other) and str(self) == str(other)
    
    def __hash__(self):
        return hash(str(self))

class PrimitiveType(Type):
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.name

class ArrayType(Type):
    def __init__(self, element_type: Type, size: Optional[int] = None):
        self.element_type = element_type
        self.size = size
    
    def __str__(self):
        if self.size:
            return f"[{self.element_type}; {self.size}]"
        return f"[{self.element_type}]"

class PointerType(Type):
    def __init__(self, pointee: Type, mutable: bool = False):
        self.pointee = pointee
        self.mutable = mutable
    
    def __str__(self):
        if self.mutable:
            return f"*mut {self.pointee}"
        return f"*{self.pointee}"

class FunctionType(Type):
    def __init__(self, params: List[Type], ret: Type):
        self.params = params
        self.ret = ret
    
    def __str__(self):
        params = ", ".join(str(p) for p in self.params)
        return f"fn({params}) -> {self.ret}"

class StructType(Type):
    def __init__(self, name: str, fields: Dict[str, Type] = None):
        self.name = name
        self.fields = fields or {}
    
    def __str__(self):
        return self.name

class GenericType(Type):
    def __init__(self, name: str, params: List[Type]):
        self.name = name
        self.params = params
    
    def __str__(self):
        params = ", ".join(str(p) for p in self.params)
        return f"{self.name}<{params}>"

class OptionType(Type):
    def __init__(self, inner: Type):
        self.inner = inner
    
    def __str__(self):
        return f"Option<{self.inner}>"

class ResultType(Type):
    def __init__(self, ok: Type, err: Type):
        self.ok = ok
        self.err = err
    
    def __str__(self):
        return f"Result<{self.ok}, {self.err}>"

class UnknownType(Type):
    def __str__(self):
        return "?"

class VoidType(Type):
    def __str__(self):
        return "void"

# Type constants
T_INT = PrimitiveType("int")
T_I8 = PrimitiveType("i8")
T_I16 = PrimitiveType("i16")
T_I32 = PrimitiveType("i32")
T_I64 = PrimitiveType("i64")
T_U8 = PrimitiveType("u8")
T_U16 = PrimitiveType("u16")
T_U32 = PrimitiveType("u32")
T_U64 = PrimitiveType("u64")
T_FLOAT = PrimitiveType("float")
T_F32 = PrimitiveType("f32")
T_F64 = PrimitiveType("f64")
T_BOOL = PrimitiveType("bool")
T_STR = PrimitiveType("str")
T_CHAR = PrimitiveType("char")
T_VOID = VoidType()
T_UNKNOWN = UnknownType()

TYPE_MAP = {
    'int': T_INT, 'i8': T_I8, 'i16': T_I16, 'i32': T_I32, 'i64': T_I64,
    'uint': PrimitiveType("uint"), 'u8': T_U8, 'u16': T_U16, 'u32': T_U32, 'u64': T_U64,
    'float': T_FLOAT, 'f32': T_F32, 'f64': T_F64,
    'bool': T_BOOL, 'str': T_STR, 'char': T_CHAR,
    'void': T_VOID,
}

# ═══════════════════════════════════════════════════════════════════════════════
#                                   AST
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ASTNode:
    line: int = 0
    col: int = 0

# Expressions
@dataclass
class IntLit(ASTNode):
    value: int = 0
    type: Type = field(default_factory=lambda: T_INT)

@dataclass
class FloatLit(ASTNode):
    value: float = 0.0
    type: Type = field(default_factory=lambda: T_FLOAT)

@dataclass
class StringLit(ASTNode):
    value: str = ""
    type: Type = field(default_factory=lambda: T_STR)

@dataclass
class FStringLit(ASTNode):
    """f"text {expr} text" - interpolated string"""
    parts: List = field(default_factory=list)  # list of (type, ASTNode_or_str)
    type: Type = field(default_factory=lambda: T_STR)

@dataclass
class BoolLit(ASTNode):
    value: bool = False
    type: Type = field(default_factory=lambda: T_BOOL)

@dataclass
class CharLit(ASTNode):
    value: str = ""
    type: Type = field(default_factory=lambda: T_CHAR)

@dataclass
class Identifier(ASTNode):
    name: str = ""
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class BinaryOp(ASTNode):
    op: str = ""
    left: ASTNode = None
    right: ASTNode = None
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class UnaryOp(ASTNode):
    op: str = ""
    operand: ASTNode = None
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class Call(ASTNode):
    func: ASTNode = None
    args: List[ASTNode] = field(default_factory=list)
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class Index(ASTNode):
    obj: ASTNode = None
    index: ASTNode = None
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class Attribute(ASTNode):
    obj: ASTNode = None
    attr: str = ""
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class ArrayLit(ASTNode):
    elements: List[ASTNode] = field(default_factory=list)
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class DictLit(ASTNode):
    pairs: List[Tuple[ASTNode, ASTNode]] = field(default_factory=list)
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class Range(ASTNode):
    start: ASTNode = None
    end: ASTNode = None
    inclusive: bool = False
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class StructInit(ASTNode):
    name: str = ""
    fields: Dict[str, ASTNode] = field(default_factory=dict)
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class Lambda(ASTNode):
    params: List[Tuple[str, Optional[Type]]] = field(default_factory=list)
    body: ASTNode = None
    ret_type: Optional[Type] = None
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class IfExpr(ASTNode):
    condition: ASTNode = None
    then_expr: ASTNode = None
    else_expr: Optional[ASTNode] = None
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class MatchExpr(ASTNode):
    value: ASTNode = None
    arms: List = field(default_factory=list)  # list of (pattern, guard_or_None, expr)
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass 
class Cast(ASTNode):
    expr: ASTNode = None
    target_type: Type = None
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class NullCheck(ASTNode):
    """expr? - unwrap optional or propagate null"""
    expr: ASTNode = None
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class TupleLit(ASTNode):
    """(a, b, c) - tuple literal"""
    elements: List[ASTNode] = field(default_factory=list)
    type: Type = field(default_factory=lambda: T_UNKNOWN)

@dataclass
class ListComprehension(ASTNode):
    """[expr for var in iter if condition]"""
    expr: ASTNode = None
    var: str = ""
    iter: ASTNode = None
    condition: Optional[ASTNode] = None
    type: Type = field(default_factory=lambda: T_UNKNOWN)

# Statements
@dataclass
class Block(ASTNode):
    statements: List[ASTNode] = field(default_factory=list)

@dataclass
class VarDecl(ASTNode):
    name: str = ""
    type_ann: Optional[Type] = None
    value: Optional[ASTNode] = None
    mutable: bool = True
    is_const: bool = False
    deferred: bool = False      # script mode: initialized where it appears (GlobalInit)

@dataclass
class GlobalInit(ASTNode):
    """Script mode: initialize a top-level variable at its position in the program."""
    decl: Optional['VarDecl'] = None

@dataclass
class Assignment(ASTNode):
    target: ASTNode = None
    value: ASTNode = None
    op: str = "="  # =, +=, -=, etc.

@dataclass
class If(ASTNode):
    condition: ASTNode = None
    then_body: ASTNode = None
    elifs: List[Tuple[ASTNode, ASTNode]] = field(default_factory=list)
    else_body: Optional[ASTNode] = None

@dataclass
class For(ASTNode):
    var: str = ""
    iter: ASTNode = None
    body: ASTNode = None

@dataclass
class While(ASTNode):
    condition: ASTNode = None
    body: ASTNode = None

@dataclass
class Loop(ASTNode):
    body: ASTNode = None

@dataclass
class Return(ASTNode):
    value: Optional[ASTNode] = None

@dataclass
class Break(ASTNode):
    pass

@dataclass
class Continue(ASTNode):
    pass

@dataclass
class ExprStmt(ASTNode):
    expr: ASTNode = None

# Definitions
@dataclass
class FuncParam:
    name: str
    type: Type
    default: Optional[ASTNode] = None
    mutable: bool = False

@dataclass
class FuncDef(ASTNode):
    name: str = ""
    params: List[FuncParam] = field(default_factory=list)
    ret_type: Type = None
    body: ASTNode = None
    is_pub: bool = False
    level: int = 2  # Default to safe level
    attributes: List[str] = field(default_factory=list)
    is_method: bool = False
    self_type: Optional[str] = None
    effects: Optional[List[str]] = None
    requires: List[ASTNode] = field(default_factory=list)
    ensures: List[ASTNode] = field(default_factory=list)
    requires_text: List[str] = field(default_factory=list)
    ensures_text: List[str] = field(default_factory=list)
    caps: List[str] = field(default_factory=list)               # #[cap: X] - required capabilities
    declared_effects: Optional[List[str]] = None                # #[pure] -> [], #[effects: io] -> ['io']

@dataclass
class StructField:
    name: str
    type: Type
    default: Optional[ASTNode] = None
    is_pub: bool = False

@dataclass
class StructDef(ASTNode):
    name: str = ""
    fields: List[StructField] = field(default_factory=list)
    methods: List[FuncDef] = field(default_factory=list)
    is_pub: bool = False

@dataclass
class EnumVariant:
    name: str
    fields: List[Type] = field(default_factory=list)
    value: Optional[ASTNode] = None

@dataclass
class EnumDef(ASTNode):
    name: str = ""
    variants: List[EnumVariant] = field(default_factory=list)
    is_pub: bool = False

@dataclass
class ImplBlock(ASTNode):
    type_name: str = ""
    methods: List[FuncDef] = field(default_factory=list)
    trait_name: Optional[str] = None

@dataclass
class TraitDef(ASTNode):
    name: str = ""
    methods: List[FuncDef] = field(default_factory=list)
    is_pub: bool = False

@dataclass
class Import(ASTNode):
    module: str = ""
    items: Optional[List[str]] = None  # None means import all
    alias: Optional[str] = None

@dataclass
class TypeAlias(ASTNode):
    name: str = ""
    type: Type = None

# ═══════════════════════════════════════════════════════════════
# V2.0 ADVANCED AST NODES
# ═══════════════════════════════════════════════════════════════

@dataclass
class GenericFunc(ASTNode):
    """Generic function: max<T>(a: T, b: T) -> T"""
    name: str = ""
    type_params: List[str] = field(default_factory=list)
    func: FuncDef = None

@dataclass
class EffectClause(ASTNode):
    """One clause of `handle ... with`:  Name(params) => body   (TIL 3.1)"""
    name: str = ""
    params: List[str] = field(default_factory=list)
    param_types: List[Any] = field(default_factory=list)
    body: ASTNode = None          # Block (indented clause) or a single expression

@dataclass
class EffectHandler(ASTNode):
    """handle <body> with <clauses>: the body runs with the clauses installed
    (docs/TIL_EFFECTS.md). Its value is the body's value, or the value of a
    clause that finishes without resume()."""
    expr: ASTNode = None
    handlers: List[EffectClause] = field(default_factory=list)

@dataclass
class PerformEffect(ASTNode):
    """perform EffectName(args) - trigger an effect (an expression: its value is
    what the handling clause passes to resume)"""
    name: str = ""
    args: List[ASTNode] = field(default_factory=list)

@dataclass
class ResumeExpr(ASTNode):
    """resume(v) / resume() inside an effect clause: v becomes the value of the
    `perform` and the handled body continues (tail resumption: the clause ends)"""
    value: Optional[ASTNode] = None

@dataclass
class CapabilityType(ASTNode):
    """cap FileRead, cap NetAccess — capability declaration"""
    name: str = ""
    permissions: List[str] = field(default_factory=list)

@dataclass
class RequiresCap(ASTNode):
    """#[cap: FileRead] — function requires capability"""
    cap_name: str = ""

@dataclass
class DurationLit(ASTNode):
    """5s, 100ms, 2m — temporal duration literal"""
    value: float = 0.0
    unit: str = "ms"  # ms, s, m, h
    type: Type = field(default_factory=lambda: T_INT)

@dataclass
class TimeoutExpr(ASTNode):
    """timeout(duration) { expr } — timeout wrapper"""
    duration: ASTNode = None
    body: ASTNode = None
    fallback: Optional[ASTNode] = None

@dataclass
class AssertStmt(ASTNode):
    """assert expr, "message" — runtime assertion"""
    condition: ASTNode = None
    message: Optional[str] = None

@dataclass
class InvariantStmt(ASTNode):
    """invariant expr — loop/struct invariant"""
    condition: ASTNode = None
    message: Optional[str] = None

@dataclass
class RetryExpr(ASTNode):
    """retry(n) { expr } — retry on failure"""
    count: ASTNode = None
    body: ASTNode = None
    fallback: Optional[ASTNode] = None

@dataclass
class FallbackExpr(ASTNode):
    """expr fallback default_val — fallback on error"""
    expr: ASTNode = None
    default: ASTNode = None

@dataclass
class ChannelType(ASTNode):
    """Channel<T> — typed channel for message passing"""
    elem_type: str = "int"

@dataclass
class SendExpr(ASTNode):
    """ch.send(val) — send to channel"""
    channel: ASTNode = None
    value: ASTNode = None

@dataclass
class RecvExpr(ASTNode):
    """ch.recv() — receive from channel"""
    channel: ASTNode = None

@dataclass
class IntentBlock(ASTNode):
    """intent "description" { body } — intent-based programming"""
    description: str = ""
    body: ASTNode = None
    constraints: List[ASTNode] = field(default_factory=list)

@dataclass
class Program(ASTNode):
    statements: List[ASTNode] = field(default_factory=list)
    energy: Optional[int] = None    # the #[energy: ...] pragma of the file (ENERGY_MODES value)

# `#[energy: X]` on a function: low = the VMs idle while it runs (FLAG_ENERGY_LOW);
# medium / high are TIL 3.0 hints for the native backend only
FUNCTION_ENERGY_LEVELS = ('low', 'medium', 'high')

# Top-level nodes that define things (everything else at the top level is a
# script statement that runs at program start - see Parser.hoist_script_statements).
DEFINITION_NODES = (FuncDef, StructDef, EnumDef, ImplBlock, TraitDef, Import,
                    TypeAlias, VarDecl, CapabilityType)

# ═══════════════════════════════════════════════════════════════════════════════
#                                  PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class Parser:
    def __init__(self, tokens: List[Token], filename: str = "<stdin>", source: str = "",
                 languages: Optional[List[str]] = None):
        self.tokens = tokens
        self.filename = filename
        self.pos = 0
        self.current_level = 2  # Default level
        self.pending_attributes: List[str] = []
        self.energy: Optional[int] = None   # the file's #[energy: ...] pragma (Program.energy)
        self.seen_definition = False        # a top-level definition was parsed (pragmas come first)
        self.error_reporter = ErrorReporter(source, filename)
        self.languages = languages
        self.in_clause = False   # parsing the code of an effect clause (resume allowed)
        # contextual keywords of the file's languages (identifier -> keyword type)
        self.ctx_words: Dict[str, TokenType] = {}
        for code in (languages or DEFAULT_LANGUAGES):
            self.ctx_words.update(CONTEXTUAL_KEYWORDS.get(code, {}))

    # ── contextual keywords (TIL 3.1 effects, see CONTEXTUAL_KEYWORDS) ────────
    # token types that can follow a localized `handle` word on the same line
    _HANDLE_BODY_START = frozenset((
        TokenType.IDENT, TokenType.INT, TokenType.FLOAT, TokenType.STRING, TokenType.FSTRING,
        TokenType.CHAR, TokenType.BOOL, TokenType.SELF, TokenType.PERFORM, TokenType.HANDLE,
        TokenType.MATCH, TokenType.RETRY, TokenType.TIMEOUT))

    def word_is(self, tok: Token, kind: TokenType) -> bool:
        """`tok` is the keyword `kind`: a hard keyword or a contextual word."""
        return tok.type == kind or (tok.type == TokenType.IDENT and self.ctx_words.get(tok.value) == kind)

    def take_word(self, kind: TokenType) -> Token:
        """Consume a (contextual) keyword. The token is retyped, so that `til
        translate` and later error messages see the keyword."""
        tok = self.current()
        if not self.word_is(tok, kind):
            self.error(f"Expected {kind.name}, got {tok.type.name}")
        tok.type = kind
        self.pos += 1
        return tok

    def contextual_start(self) -> Optional[TokenType]:
        """At the start of a primary expression: the effect keyword a contextual
        word stands for here (see CONTEXTUAL_KEYWORDS), or None."""
        tok = self.current()
        kind = self.ctx_words.get(tok.value) if tok.type == TokenType.IDENT else None
        if kind is None:
            return None
        nxt = self.peek()
        if kind == TokenType.RESUME:
            return kind if self.in_clause else None
        if kind == TokenType.PERFORM:
            return kind if nxt.type == TokenType.IDENT and nxt.line == tok.line else None
        if kind == TokenType.HANDLE:
            if nxt.type == TokenType.NEWLINE:
                k = self.pos + 1
                while k < len(self.tokens) and self.tokens[k].type == TokenType.NEWLINE:
                    k += 1
                return kind if k < len(self.tokens) and self.tokens[k].type == TokenType.INDENT else None
            if nxt.line == tok.line and nxt.type in self._HANDLE_BODY_START:
                return kind
        return None

    def reserved_note(self) -> str:
        """' (`x` is a reserved word ...)' when the current token is a keyword."""
        tok = self.current()
        if tok.type != TokenType.IDENT and isinstance(tok.value, str) and tok.value.isidentifier():
            return f" ('{tok.value}' is a reserved word and cannot be used as a name)"
        return ""

    def check_binding_name(self, tok: Token):
        """Inside an effect clause the resume words mean resume: no variables."""
        if self.in_clause and self.ctx_words.get(tok.value) == TokenType.RESUME:
            self._syntax_error_at(tok, f"'{tok.value}' is reserved inside an effect clause",
                                  "there it always means resume(...); choose another name")

    @staticmethod
    def _mark(node, tok: Token):
        """Attach a source position to an AST node (if it has none yet)."""
        if isinstance(node, ASTNode) and not node.line and tok is not None:
            node.line = tok.line
            node.col = tok.col
        return node

    def error(self, msg: str):
        self.error_at(self.current(), msg)

    def error_at(self, tok: Token, msg: str):
        hint = get_hint_for_error(msg)
        formatted = self.error_reporter.format_error(tok.line, tok.col, msg, hint=hint)
        raise SyntaxError(formatted)
    
    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF
    
    def peek(self, n: int = 1) -> Token:
        pos = self.pos + n
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]
    
    def match(self, *types: TokenType) -> bool:
        return self.current().type in types
    
    def consume(self, type: TokenType, msg: str = None) -> Token:
        if self.current().type == type:
            tok = self.current()
            self.pos += 1
            return tok
        note = self.reserved_note() if type == TokenType.IDENT else ""
        if msg:
            self.error(msg + note)
        self.error(f"Expected {type.name}, got {self.current().type.name}{note}")
    
    def advance(self) -> Token:
        tok = self.current()
        self.pos += 1
        return tok
    
    def skip_newlines(self):
        while self.match(TokenType.NEWLINE):
            self.advance()
    
    def parse(self) -> Program:
        statements = []
        self.skip_newlines()

        while not self.match(TokenType.EOF):
            stmt = self.parse_top_level()
            if stmt:
                statements.append(stmt)
                if isinstance(stmt, DEFINITION_NODES):
                    self.seen_definition = True
            self.skip_newlines()

        return Program(statements=self.hoist_script_statements(statements), energy=self.energy)

    @staticmethod
    def hoist_script_statements(statements: List[ASTNode]) -> List[ASTNode]:
        """Script mode: executable statements written at the top level run first,
        in source order. Without a `main()` they *are* the program (so `print("hi")`
        is a complete TIL program); with a `main()` they run before its body.
        Top-level variables declared after a script statement stay globals (so
        functions can use them) but are initialized at their position."""
        if not any(not isinstance(s, DEFINITION_NODES) for s in statements):
            return statements
        script, rest, seen_stmt = [], [], False
        for st in statements:
            if not isinstance(st, DEFINITION_NODES):
                seen_stmt = True
                script.append(st)
            elif (isinstance(st, VarDecl) and seen_stmt and st.value is not None and
                  not isinstance(st.value, (IntLit, FloatLit, StringLit, BoolLit, CharLit))):
                st.deferred = True
                rest.append(st)
                script.append(GlobalInit(decl=st, line=st.line, col=st.col))
            else:
                rest.append(st)
        main = next((s for s in rest if isinstance(s, FuncDef) and s.name == "main"), None)
        if main is None:
            main = FuncDef(name="main", params=[], ret_type=T_VOID,
                           body=Block(statements=script), line=script[0].line)
            rest.append(main)
        else:
            body = main.body if isinstance(main.body, Block) else Block(statements=[main.body])
            main.body = Block(statements=script + body.statements, line=body.line)
        return rest

    def parse_top_level(self) -> Optional[ASTNode]:
        tok = self.current()
        node = self._parse_top_level()
        return self._mark(node, tok)

    def _parse_top_level(self) -> Optional[ASTNode]:
        self.skip_newlines()
        
        # Collect attributes
        energy_toks = []
        while self.match(TokenType.ATTRIBUTE):
            tok = self.advance()
            attr = tok.value
            self.pending_attributes.append(attr)
            if attr.startswith("energy:"):
                energy_toks.append(tok)
            
            # Parse level
            if attr.startswith("level:"):
                try:
                    self.current_level = int(attr.split(":")[1].strip())
                except:
                    pass
            
            self.skip_newlines()
        if energy_toks:
            self.place_energy_attributes(energy_toks)
        
        if self.match(TokenType.IMPORT, TokenType.FROM):
            return self.parse_import()
        
        if self.match(TokenType.PUB):
            self.advance()
            return self.parse_pub_item()
        
        if self.match(TokenType.STRUCT):
            return self.parse_struct()
        
        if self.match(TokenType.ENUM):
            return self.parse_enum()
        
        if self.match(TokenType.IMPL):
            return self.parse_impl()
        
        if self.match(TokenType.TRAIT):
            return self.parse_trait()
        
        if self.match(TokenType.FN):
            return self.parse_fn()
        
        if self.match(TokenType.TYPE):
            return self.parse_type_alias()
        
        if self.match(TokenType.CONST):
            return self.parse_const()

        if self.match(TokenType.CAP):
            return self.parse_capability()

        if self.match(TokenType.IDENT):
            # Could be function def or statement
            if self.is_func_def():
                return self.parse_func_def_simple()
            return self.parse_statement()
        
        return self.parse_statement()
    
    def place_energy_attributes(self, toks: List[Token]):
        """`#[energy: ...]` before a top-level item (docs/TIL_ENERGY.md): it marks
        the function defined next when it is directly above it (no empty line in
        between); otherwise - or with the program modes `balanced` / `performance`
        - it is the program's energy-mode pragma, which must come before every
        definition of the file (a module's pragma only counts when the module
        is the program)."""
        func_next = self.at_function_definition()
        for tok in toks:
            level = tok.value[len("energy:"):].strip()
            attached = func_next and not self.blank_line_after(tok)
            if attached and level not in ('balanced', 'performance'):
                continue                      # an attribute of the function (parse_func_body)
            self.pending_attributes.remove(tok.value)
            if self.seen_definition:
                self.error_at(tok, "#[energy: ...] must be directly above a function, or come before "
                                   "every definition (the program's energy mode)")
            if level not in ENERGY_MODES:
                self.error_at(tok, f"unknown energy mode '{level}' (use low, balanced or performance)")
            if self.energy is not None:
                self.error_at(tok, "the program's energy mode is set twice")
            self.energy = ENERGY_MODES[level]

    def at_function_definition(self) -> bool:
        """The next top-level item is a function definition."""
        if self.match(TokenType.FN):
            return True
        if self.match(TokenType.PUB):
            nxt = self.peek()
            if nxt.type == TokenType.FN:
                return True
            if nxt.type != TokenType.IDENT:
                return False
            self.pos += 1
            try:
                return self.is_func_def()
            finally:
                self.pos -= 1
        return self.match(TokenType.IDENT) and self.is_func_def()

    def blank_line_after(self, tok: Token) -> bool:
        """An empty line comes between `tok` and the next token (comment lines do not count)."""
        src = self.error_reporter.source or ""
        i, n = tok.end, len(src)
        if i < 0:
            return False
        while i < n and src[i] != '\n':          # the rest of tok's line
            if src[i] == '#' and src[i + 1:i + 2] != '[':
                while i < n and src[i] != '\n':
                    i += 1
                break
            if src[i] not in ' \t\r':
                return False                      # a token on the same line
            i += 1
        while i < n:                              # src[i] is a newline
            i += 1
            while i < n and src[i] in ' \t\r':
                i += 1
            if i >= n or src[i] == '\n':
                return True
            if src[i] != '#' or src[i + 1:i + 2] == '[':
                return False
            while i < n and src[i] != '\n':      # a comment line
                i += 1
        return True

    def parse_pub_item(self) -> ASTNode:
        if self.match(TokenType.STRUCT):
            node = self.parse_struct()
            node.is_pub = True
            return node
        if self.match(TokenType.ENUM):
            node = self.parse_enum()
            node.is_pub = True
            return node
        if self.match(TokenType.FN):
            node = self.parse_fn()
            node.is_pub = True
            return node
        if self.match(TokenType.TRAIT):
            node = self.parse_trait()
            node.is_pub = True
            return node
        if self.match(TokenType.IDENT):
            if self.is_func_def():
                node = self.parse_func_def_simple()
                node.is_pub = True
                return node
        self.error("Expected struct, enum, fn, or trait after 'pub'")
    
    def is_func_def(self) -> bool:
        """Check if current position is a function definition"""
        if not self.match(TokenType.IDENT):
            return False
        
        # Save position
        pos = self.pos
        self.advance()  # name

        if self.match(TokenType.LT):        # generic parameters: name<T, U>(...)
            self.advance()
            while self.match(TokenType.IDENT, TokenType.COMMA):
                self.advance()
            if not self.match(TokenType.GT):
                self.pos = pos
                return False
            self.advance()
        
        if not self.match(TokenType.LPAREN):
            self.pos = pos
            return False
        
        # Find matching paren
        depth = 1
        self.advance()
        while depth > 0 and not self.match(TokenType.EOF):
            if self.match(TokenType.LPAREN):
                depth += 1
            elif self.match(TokenType.RPAREN):
                depth -= 1
            self.advance()

        # A definition is `name(...)` followed by `-> T`, `{`, `=>`, or an
        # indented body (NEWLINE INDENT). A bare `name(...)` + NEWLINE is a call
        # statement (script mode), e.g. `print("hi")` at the top level.
        result = self.match(TokenType.ARROW, TokenType.LBRACE, TokenType.FAT_ARROW)
        if not result and self.match(TokenType.NEWLINE):
            self.skip_newlines()
            result = self.match(TokenType.INDENT)

        self.pos = pos
        return result
    
    def parse_fn(self) -> FuncDef:
        self.consume(TokenType.FN)
        name = self.consume(TokenType.IDENT).value
        self.parse_type_params()
        return self.parse_func_body(name)
    
    def parse_func_def_simple(self) -> FuncDef:
        name = self.consume(TokenType.IDENT).value
        self.parse_type_params()
        return self.parse_func_body(name)

    def parse_type_params(self):
        """`name<T, U>(...)`: same as #[generic: T, U] (TIL 3.1)."""
        if not self.match(TokenType.LT):
            return
        self.advance()
        names = []
        while not self.match(TokenType.GT):
            names.append(self.consume(TokenType.IDENT, "Expected a type parameter name").value)
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        self.consume(TokenType.GT, "Expected '>' after the type parameters")
        if names:
            self.pending_attributes.append("generic:" + ",".join(names))
    
    def parse_func_body(self, name: str, allow_signature: bool = False) -> FuncDef:
        name_tok = self.tokens[self.pos - 1] if self.pos > 0 else self.current()
        self.consume(TokenType.LPAREN)
        params = self.parse_params()
        self.consume(TokenType.RPAREN)

        # Return type
        ret_type = T_VOID
        if self.match(TokenType.ARROW):
            self.advance()
            ret_type = self.parse_type()

        self.skip_newlines()

        # Body
        if self.match(TokenType.INDENT):
            body = self.parse_block()
        elif self.match(TokenType.LBRACE):
            body = self.parse_brace_block()
        elif self.match(TokenType.FAT_ARROW):
            fat = self.advance()
            expr = self.parse_expression()
            body = Block(statements=[self._mark(Return(value=expr), fat)])
        elif allow_signature:
            body = Block(statements=[])     # trait method signature (no default body)
        else:
            self.error("Expected function body")

        # Apply pending attributes
        attrs = self.pending_attributes
        level = self.current_level
        self.pending_attributes = []
        self.current_level = 2

        # Parse effects and contracts from attributes
        effects = []
        requires_list = []
        ensures_list = []
        requires_text = []
        ensures_text = []
        caps = []
        declared_effects = None   # None = not declared (gradual: inferred, unchecked)
        remaining_attrs = []
        for attr in attrs:
            if attr == "pure":
                effects.append("pure")
                declared_effects = declared_effects or []
            elif attr.startswith("effects:"):
                declared_effects = declared_effects or []
                for e in attr[len("effects:"):].strip().split(","):
                    if e.strip():
                        effects.append(e.strip())
                        declared_effects.append(e.strip())
            elif attr.startswith("requires:") or attr.startswith("ensures:"):
                kind, expr_text = attr.split(":", 1)
                expr_text = expr_text.strip()
                sub_tokens = Lexer(expr_text, "<attr>", languages=self.languages,
                                   line_offset=max(name_tok.line - 1, 0)).tokenize()
                sub_parser = Parser(sub_tokens, "<attr>", languages=self.languages)
                cexpr = sub_parser.parse_expression()
                if kind == "requires":
                    requires_list.append(cexpr)
                    requires_text.append(expr_text)
                else:
                    ensures_list.append(cexpr)
                    ensures_text.append(expr_text)
            elif attr.startswith("cap:"):
                # Capability annotation: #[cap: FileRead] or #[cap: A, B]
                for cap_name in attr[len("cap:"):].split(","):
                    cap_name = cap_name.strip()
                    if cap_name:
                        effects.append(f"cap:{cap_name}")
                        caps.append(cap_name)
            elif attr.startswith("energy:"):
                # Energy annotation: #[energy: low] (the VMs idle while it runs, see
                # docs/TIL_ENERGY.md); medium / high only steer the native backend
                energy_level = attr[len("energy:"):].strip()
                if energy_level not in FUNCTION_ENERGY_LEVELS:
                    self.error_at(name_tok, f"unknown energy level '{energy_level}' for a function "
                                            f"(use #[energy: low])")
                remaining_attrs.append(f"energy:{energy_level}")
            elif attr.startswith("generic:"):
                # Generic type params: #[generic: T, U]
                type_params = [t.strip() for t in attr[len("generic:"):].strip().split(",")]
                remaining_attrs.append(f"generic:{','.join(type_params)}")
            elif attr.startswith("cross_level:"):
                # Cross-level contract: #[cross_level: from_level -> to_level]
                remaining_attrs.append(attr)
            else:
                remaining_attrs.append(attr)

        return FuncDef(name=name, params=params, ret_type=ret_type, body=body,
                       level=level, attributes=remaining_attrs,
                       effects=effects if effects else None,
                       requires=requires_list, ensures=ensures_list,
                       requires_text=requires_text, ensures_text=ensures_text,
                       caps=caps, declared_effects=declared_effects,
                       line=name_tok.line, col=name_tok.col)

    def parse_params(self) -> List[FuncParam]:
        params = []
        
        if self.match(TokenType.RPAREN):
            return params
        
        while True:
            mutable = False
            if self.match(TokenType.MUT):
                self.advance()
                mutable = True
            
            if self.match(TokenType.SELF):
                self.advance()
                params.append(FuncParam("self", T_UNKNOWN, mutable=mutable))
            else:
                name = self.consume(TokenType.IDENT).value
                
                param_type = T_UNKNOWN
                if self.match(TokenType.COLON):
                    self.advance()
                    param_type = self.parse_type()
                
                default = None
                if self.match(TokenType.EQ):
                    self.advance()
                    default = self.parse_expression()
                
                params.append(FuncParam(name, param_type, default, mutable))
            
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        
        return params
    
    def parse_type(self) -> Type:
        # Pointer
        if self.match(TokenType.STAR):
            self.advance()
            mutable = False
            if self.match(TokenType.MUT):
                self.advance()
                mutable = True
            inner = self.parse_type()
            return PointerType(inner, mutable)
        
        # Array
        if self.match(TokenType.LBRACKET):
            self.advance()
            elem_type = self.parse_type()
            size = None
            if self.match(TokenType.SEMICOLON):
                self.advance()
                size = self.consume(TokenType.INT).value
            self.consume(TokenType.RBRACKET)
            return ArrayType(elem_type, size)
        
        # Function type
        if self.match(TokenType.FN):
            self.advance()
            self.consume(TokenType.LPAREN)
            params = []
            if not self.match(TokenType.RPAREN):
                while True:
                    params.append(self.parse_type())
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
            self.consume(TokenType.RPAREN)
            ret = T_VOID
            if self.match(TokenType.ARROW):
                self.advance()
                ret = self.parse_type()
            return FunctionType(params, ret)
        
        # Named type
        name = self.consume(TokenType.IDENT).value
        
        # Generic
        if self.match(TokenType.LT):
            self.advance()
            params = []
            while True:
                params.append(self.parse_type())
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            self.consume(TokenType.GT)
            
            if name == "Option":
                return OptionType(params[0])
            if name == "Result":
                return ResultType(params[0], params[1] if len(params) > 1 else T_STR)
            return GenericType(name, params)
        
        # Primitive or struct
        base_type = TYPE_MAP.get(name, StructType(name))

        # Support type[] syntax for arrays (e.g., float[], int[])
        if self.match(TokenType.LBRACKET):
            self.advance()
            size = None
            if self.match(TokenType.INT):
                size = self.advance().value
            self.consume(TokenType.RBRACKET)
            return ArrayType(base_type, size)

        return base_type
    
    def parse_block(self) -> Block:
        self.consume(TokenType.INDENT)
        statements = []
        
        while not self.match(TokenType.DEDENT, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.DEDENT, TokenType.EOF):
                break
            
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
        
        if self.match(TokenType.DEDENT):
            self.advance()
        
        return Block(statements=statements)
    
    def parse_brace_block(self) -> Block:
        self.consume(TokenType.LBRACE)
        self.skip_newlines()
        statements = []
        
        while not self.match(TokenType.RBRACE, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
            if self.match(TokenType.SEMICOLON):
                self.advance()
            self.skip_newlines()
        
        self.consume(TokenType.RBRACE)
        return Block(statements=statements)
    
    def parse_statement(self) -> Optional[ASTNode]:
        self.skip_newlines()
        tok = self.current()
        return self._mark(self._parse_statement(), tok)

    def _parse_statement(self) -> Optional[ASTNode]:
        self.skip_newlines()
        
        if self.match(TokenType.IF):
            return self.parse_if()
        
        if self.match(TokenType.FOR):
            return self.parse_for()
        
        if self.match(TokenType.WHILE):
            return self.parse_while()
        
        if self.match(TokenType.LOOP):
            return self.parse_loop()
        
        if self.match(TokenType.MATCH):
            return self.parse_match()
        
        if self.match(TokenType.RETURN):
            return self.parse_return()
        
        if self.match(TokenType.BREAK):
            self.advance()
            return Break()
        
        if self.match(TokenType.CONTINUE):
            self.advance()
            return Continue()
        
        if self.match(TokenType.LET, TokenType.VAR, TokenType.CONST):
            return self.parse_var_decl()

        # V2.0 advanced statements
        if self.match(TokenType.ASSERT):
            return self.parse_assert()

        if self.match(TokenType.INVARIANT):
            return self.parse_invariant()

        if self.match(TokenType.INTENT):
            return self.parse_intent_block()

        # Expression or assignment
        expr = self.parse_expression()
        
        # Check for assignment
        if self.match(TokenType.EQ, TokenType.PLUSEQ, TokenType.MINUSEQ,
                      TokenType.STAREQ, TokenType.SLASHEQ, TokenType.PERCENTEQ):
            op_tok = self.advance()
            value = self.parse_expression()
            return self._mark(Assignment(target=expr, value=value, op=op_tok.value), op_tok)
        
        return ExprStmt(expr=expr)
    
    def parse_var_decl(self) -> VarDecl:
        is_const = self.match(TokenType.CONST)
        mutable = self.match(TokenType.VAR)
        self.advance()  # let/var/const
        
        name_tok = self.consume(TokenType.IDENT)
        self.check_binding_name(name_tok)
        name = name_tok.value
        
        type_ann = None
        if self.match(TokenType.COLON):
            self.advance()
            type_ann = self.parse_type()
        
        value = None
        if self.match(TokenType.EQ):
            self.advance()
            value = self.parse_expression()
        
        return VarDecl(name=name, type_ann=type_ann, value=value, mutable=mutable, is_const=is_const)
    
    def parse_if(self) -> If:
        self.consume(TokenType.IF)
        condition = self.parse_expression()
        
        self.skip_newlines()
        if self.match(TokenType.INDENT):
            then_body = self.parse_block()
        elif self.match(TokenType.LBRACE):
            then_body = self.parse_brace_block()
        else:
            then_body = Block(statements=[self.parse_statement()])
        
        elifs = []
        else_body = None
        
        self.skip_newlines()
        
        # `elif` - or `else if` in any language (әйтпесе егер / иначе если / aitpese eger)
        while self.match(TokenType.ELIF) or (self.match(TokenType.ELSE) and
                                            self.peek().type == TokenType.IF):
            if self.match(TokenType.ELSE):
                self.advance()
            self.advance()
            elif_cond = self.parse_expression()
            self.skip_newlines()
            if self.match(TokenType.INDENT):
                elif_body = self.parse_block()
            elif self.match(TokenType.LBRACE):
                elif_body = self.parse_brace_block()
            else:
                elif_body = Block(statements=[self.parse_statement()])
            elifs.append((elif_cond, elif_body))
            self.skip_newlines()
        
        if self.match(TokenType.ELSE):
            self.advance()
            self.skip_newlines()
            if self.match(TokenType.INDENT):
                else_body = self.parse_block()
            elif self.match(TokenType.LBRACE):
                else_body = self.parse_brace_block()
            else:
                else_body = Block(statements=[self.parse_statement()])
        
        return If(condition=condition, then_body=then_body, elifs=elifs, else_body=else_body)
    
    def parse_for(self) -> For:
        self.consume(TokenType.FOR)
        var_tok = self.consume(TokenType.IDENT)
        self.check_binding_name(var_tok)
        var = var_tok.value
        self.consume(TokenType.IN)
        iter_expr = self.parse_expression()
        
        self.skip_newlines()
        if self.match(TokenType.INDENT):
            body = self.parse_block()
        elif self.match(TokenType.LBRACE):
            body = self.parse_brace_block()
        else:
            body = Block(statements=[self.parse_statement()])
        
        return For(var=var, iter=iter_expr, body=body)
    
    def parse_while(self) -> While:
        self.consume(TokenType.WHILE)
        condition = self.parse_expression()
        
        self.skip_newlines()
        if self.match(TokenType.INDENT):
            body = self.parse_block()
        elif self.match(TokenType.LBRACE):
            body = self.parse_brace_block()
        else:
            body = Block(statements=[self.parse_statement()])
        
        return While(condition=condition, body=body)
    
    def parse_loop(self) -> Loop:
        self.consume(TokenType.LOOP)
        
        self.skip_newlines()
        if self.match(TokenType.INDENT):
            body = self.parse_block()
        elif self.match(TokenType.LBRACE):
            body = self.parse_brace_block()
        else:
            body = Block(statements=[self.parse_statement()])
        
        return Loop(body=body)
    
    def parse_match(self) -> MatchExpr:
        self.consume(TokenType.MATCH)
        value = self.parse_expression()

        self.skip_newlines()
        self.consume(TokenType.INDENT)

        arms = []
        while not self.match(TokenType.DEDENT, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.DEDENT, TokenType.EOF):
                break

            # Use parse_or to avoid ternary consuming 'if' (needed for guard syntax)
            pattern = self.parse_or()

            # Parse optional guard: pattern if condition =>
            guard = None
            if self.match(TokenType.IF):
                self.advance()
                guard = self.parse_or()

            self.consume(TokenType.FAT_ARROW)

            if self.match(TokenType.INDENT):
                expr = self.parse_block()
            else:
                expr = self.parse_expression()

            arms.append((pattern, guard, expr))
            self.skip_newlines()

        if self.match(TokenType.DEDENT):
            self.advance()

        self.end_block_expr()
        return MatchExpr(value=value, arms=arms)

    def end_block_expr(self):
        """An expression that ends with an indented block (match, retry, timeout,
        handle) also ends its statement: an `if`, `[`, `(` or `-` on the next line
        starts a new statement instead of continuing the expression (`x if c else
        y`, indexing, ...). The lexer emits no NEWLINE after a DEDENT (and the
        parsers above skip the NEWLINE before a possible `else`), so add one."""
        if self.pos > 0 and self.tokens[self.pos - 1].type in (TokenType.DEDENT, TokenType.NEWLINE) and \
                not self.match(TokenType.NEWLINE, TokenType.DEDENT, TokenType.EOF):
            tok = self.current()
            self.tokens.insert(self.pos, Token(TokenType.NEWLINE, '\n', tok.line, tok.col))

    # ═══════════════════════════════════════════════════════════
    # V2.0 ADVANCED PARSERS
    # ═══════════════════════════════════════════════════════════

    def parse_assert(self) -> AssertStmt:
        """Parse: assert condition, "message" """
        self.consume(TokenType.ASSERT)
        condition = self.parse_expression()
        message = None
        if self.match(TokenType.COMMA):
            self.advance()
            msg_tok = self.consume(TokenType.STRING)
            message = msg_tok.value
        return AssertStmt(condition=condition, message=message)

    def parse_invariant(self) -> InvariantStmt:
        """Parse: invariant condition, "message" """
        self.consume(TokenType.INVARIANT)
        condition = self.parse_expression()
        message = None
        if self.match(TokenType.COMMA):
            self.advance()
            msg_tok = self.consume(TokenType.STRING)
            message = msg_tok.value
        return InvariantStmt(condition=condition, message=message)

    def parse_capability(self) -> CapabilityType:
        """Parse: cap CapName { read, write, ... }"""
        self.consume(TokenType.CAP)
        name = self.consume(TokenType.IDENT).value
        permissions = []
        if self.match(TokenType.LBRACE):
            self.advance()
            while not self.match(TokenType.RBRACE):
                perm = self.consume(TokenType.IDENT).value
                permissions.append(perm)
                if self.match(TokenType.COMMA):
                    self.advance()
            self.consume(TokenType.RBRACE)
        return CapabilityType(name=name, permissions=permissions)

    def parse_retry_expr(self) -> RetryExpr:
        """Parse: retry(n) { body } else { fallback }"""
        self.consume(TokenType.RETRY)
        self.consume(TokenType.LPAREN)
        count = self.parse_expression()
        self.consume(TokenType.RPAREN)
        self.skip_newlines()
        body = self.parse_block() if self.match(TokenType.INDENT) else self.parse_expression()
        fallback = None
        self.skip_newlines()
        if self.match(TokenType.ELSE):
            self.advance()
            self.skip_newlines()
            fallback = self.parse_block() if self.match(TokenType.INDENT) else self.parse_expression()
        self.end_block_expr()
        return RetryExpr(count=count, body=body, fallback=fallback)

    def parse_timeout_expr(self) -> TimeoutExpr:
        """Parse: timeout(duration_ms) { body } else { fallback }"""
        self.consume(TokenType.TIMEOUT)
        self.consume(TokenType.LPAREN)
        duration = self.parse_expression()
        self.consume(TokenType.RPAREN)
        self.skip_newlines()
        body = self.parse_block() if self.match(TokenType.INDENT) else self.parse_expression()
        fallback = None
        self.skip_newlines()
        if self.match(TokenType.ELSE):
            self.advance()
            self.skip_newlines()
            fallback = self.parse_block() if self.match(TokenType.INDENT) else self.parse_expression()
        self.end_block_expr()
        return TimeoutExpr(duration=duration, body=body, fallback=fallback)

    def parse_intent_block(self) -> IntentBlock:
        """Parse: intent "description" { body }"""
        self.consume(TokenType.INTENT)
        desc_tok = self.consume(TokenType.STRING)
        self.skip_newlines()
        body = self.parse_block() if self.match(TokenType.INDENT) else self.parse_expression()
        return IntentBlock(description=desc_tok.value, body=body)

    def parse_perform(self) -> PerformEffect:
        """Parse: perform EffectName(args)"""
        self.take_word(TokenType.PERFORM)
        name = self.consume(TokenType.IDENT, "Expected the effect name after 'perform'").value
        args = []
        if self.match(TokenType.LPAREN):
            self.advance()
            if not self.match(TokenType.RPAREN):
                while True:
                    args.append(self.parse_expression())
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
            self.consume(TokenType.RPAREN)
        return PerformEffect(name=name, args=args)

    def parse_handle_expr(self) -> EffectHandler:
        """Parse an effect handler (TIL 3.1, docs/TIL_EFFECTS.md):

            handle                          handle <expression>
                <body>                      with
            with                                Name(params) => <expression>
                Name(params) => <expression>
                Name(params) =>
                    <block>
                Tick => <expression>        (no parameters: `Tick` or `Tick()`)
        """
        self.take_word(TokenType.HANDLE)
        self.skip_newlines()
        body = self.parse_block() if self.match(TokenType.INDENT) else self.parse_expression()
        self.skip_newlines()
        if not self.word_is(self.current(), TokenType.WITH):
            self.error("Expected 'with' and effect clauses after the body of 'handle'")
        self.take_word(TokenType.WITH)
        self.skip_newlines()
        clauses: List[EffectClause] = []
        if self.match(TokenType.INDENT):
            self.advance()
            while not self.match(TokenType.DEDENT, TokenType.EOF):
                self.skip_newlines()
                if self.match(TokenType.DEDENT, TokenType.EOF):
                    break
                clauses.append(self.parse_effect_clause(clauses))
                self.skip_newlines()
            if self.match(TokenType.DEDENT):
                self.advance()
        else:
            clauses.append(self.parse_effect_clause(clauses))
        if not clauses:
            self.error("'handle ... with' needs at least one effect clause: Name(params) => ...")
        self.end_block_expr()
        return EffectHandler(expr=body, handlers=clauses)

    def _syntax_error_at(self, node_or_tok, msg: str, hint: str = ""):
        line = getattr(node_or_tok, 'line', 0) or 0
        col = getattr(node_or_tok, 'col', 0) or 1
        raise SyntaxError(self.error_reporter.format_error(line, col, msg, hint=hint))

    def parse_effect_clause(self, seen: List[EffectClause]) -> EffectClause:
        """Name(params) => expression   |   Name(params) => NEWLINE <block>"""
        name_tok = self.consume(TokenType.IDENT, "Expected an effect clause: Name(params) => ...")
        params: List[str] = []
        ptypes: List[Any] = []
        if self.match(TokenType.LPAREN):
            self.advance()
            if not self.match(TokenType.RPAREN):
                while True:
                    ptok = self.consume(TokenType.IDENT, "Expected a parameter name in the effect clause")
                    if self.ctx_words.get(ptok.value) == TokenType.RESUME:
                        self._syntax_error_at(ptok, f"'{ptok.value}' is reserved inside an effect clause",
                                              "there it always means resume(...); choose another name")
                    ptype = None
                    if self.match(TokenType.COLON):
                        self.advance()
                        ptype = self.parse_type()
                    if ptok.value in params:
                        self._syntax_error_at(ptok, f"duplicate parameter '{ptok.value}' in the clause "
                                                    f"for effect '{name_tok.value}'")
                    params.append(ptok.value)
                    ptypes.append(ptype)
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
            self.consume(TokenType.RPAREN, "Expected ')' after the clause parameters")
        if any(c.name == name_tok.value for c in seen):
            self._syntax_error_at(name_tok, f"duplicate clause for effect '{name_tok.value}' in one handle",
                                  "each effect may have only one clause per 'handle ... with'")
        self.consume(TokenType.FAT_ARROW, "Expected '=>' after the effect clause head")
        saved = self.in_clause
        self.in_clause = True
        try:
            if self.match(TokenType.NEWLINE):
                self.skip_newlines()
                if not self.match(TokenType.INDENT):
                    self.error("Expected an expression or an indented block after '=>'")
                body = self.parse_block()
            else:
                body = self.parse_expression()
        finally:
            self.in_clause = saved
        self._check_code_after_resume(body)
        return EffectClause(name=name_tok.value, params=params, param_types=ptypes, body=body,
                            line=name_tok.line, col=name_tok.col)

    # nodes that evaluate all their children, in order, and then do their own
    # work: when a child always reaches resume(...), that work never happens
    _EAGER_NODES = (Call, UnaryOp, Index, Attribute, ArrayLit, TupleLit, DictLit, StructInit,
                    Range, Cast, NullCheck, FStringLit, PerformEffect, VarDecl, Assignment,
                    AssertStmt, InvariantStmt)

    def _check_code_after_resume(self, body):
        """resume() is a tail call: the clause ends there. Code that could only run
        after it - a following statement, or the expression that would use the
        value of resume(...) - is a compile error, never silently dropped."""
        def unreachable(node, what="unreachable code after resume(...)"):
            at = next((n for n in walk(node) if getattr(n, 'line', 0)), node)
            self._syntax_error_at(
                at, what, "resume ends the clause and continues the handled body at once "
                          "(tail resumption); it does not return here, so move this code before "
                          "resume (write resume(v) as the clause's last statement)")
        value_lost = "the value of resume(...) is never available: resume ends the clause"

        def ends(node) -> bool:
            """Check `node`; True when evaluating it always reaches resume(...)."""
            if node is None or isinstance(node, (Lambda, EffectClause)):
                return False      # separate code (a nested clause is checked on its own)
            if isinstance(node, ResumeExpr):
                if ends(node.value):
                    unreachable(node)
                return True
            if isinstance(node, Block):
                stmts = node.statements
                for k, st in enumerate(stmts):
                    if ends(st):
                        if k + 1 < len(stmts):
                            unreachable(stmts[k + 1])
                        return True
                return False
            if isinstance(node, ExprStmt):
                return ends(node.expr)
            if isinstance(node, Assignment) and isinstance(node.target, ResumeExpr):
                self._syntax_error_at(node, "'resume' is reserved inside an effect clause",
                                      "there it always means resume(...); use another variable name")
            if isinstance(node, BinaryOp) and node.op in ('and', 'or'):
                if ends(node.left):
                    unreachable(node.left, value_lost)
                ends(node.right)          # only sometimes evaluated
                return False
            if isinstance(node, self._EAGER_NODES) or isinstance(node, BinaryOp):
                for child in iter_children(node):
                    if ends(child):
                        unreachable(child, value_lost)
                return False
            if isinstance(node, (If, IfExpr)):
                if isinstance(node, If):
                    tests = [(node.condition, node.then_body)] + list(node.elifs)
                    other = node.else_body
                else:
                    tests, other = [(node.condition, node.then_expr)], node.else_expr
                every = True
                for cond, branch in tests:
                    if ends(cond):
                        unreachable(branch)
                    every = ends(branch) and every
                return ends(other) and every
            if isinstance(node, (While, For)):
                if ends(node.condition if isinstance(node, While) else node.iter):
                    unreachable(node.body)
                ends(node.body)
                return False
            if isinstance(node, MatchExpr):
                if ends(node.value):
                    unreachable(node, "the match arms are never reached: resume(...) ends the clause")
            # anything else (match arms, loop bodies, retry/timeout/intent and nested
            # handle bodies - a clause may abort them, an error may skip to the
            # fallback): checked inside, but it is not known to always resume
            for child in iter_children(node):
                ends(child)
            return False

        ends(body)

    def parse_resume(self) -> ResumeExpr:
        """resume(v) / resume() / resume: only inside an effect clause."""
        tok = self.take_word(TokenType.RESUME)
        if not self.in_clause:
            self._syntax_error_at(tok, "resume(...) outside an effect clause",
                                  "resume continues the body of a 'handle' and may only be used in "
                                  "one of its 'with' clauses (not in a nested lambda)")
        value = None
        if self.match(TokenType.LPAREN):
            self.advance()
            if not self.match(TokenType.RPAREN):
                value = self.parse_expression()
            self.consume(TokenType.RPAREN, "resume takes at most one value: resume(v)")
        return ResumeExpr(value=value)

    def parse_return(self) -> Return:
        ret_tok = self.consume(TokenType.RETURN)
        if self.in_clause:
            self._syntax_error_at(ret_tok, "'return' is not allowed in an effect clause",
                                  "use resume(v) to continue the handled body, or end the clause "
                                  "with a value: that value becomes the value of the whole handle")

        value = None
        if not self.match(TokenType.NEWLINE, TokenType.DEDENT, TokenType.EOF, TokenType.RBRACE):
            value = self.parse_expression()
        
        return Return(value=value)
    
    def parse_struct(self) -> StructDef:
        self.consume(TokenType.STRUCT)
        name = self.consume(TokenType.IDENT).value
        
        self.skip_newlines()
        self.consume(TokenType.INDENT)
        
        fields = []
        while not self.match(TokenType.DEDENT, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.DEDENT, TokenType.EOF):
                break
            
            is_pub = False
            if self.match(TokenType.PUB):
                self.advance()
                is_pub = True
            
            field_name = self.consume(TokenType.IDENT).value
            self.consume(TokenType.COLON)
            field_type = self.parse_type()
            
            default = None
            if self.match(TokenType.EQ):
                self.advance()
                default = self.parse_expression()
            
            fields.append(StructField(field_name, field_type, default, is_pub))
            self.skip_newlines()
        
        if self.match(TokenType.DEDENT):
            self.advance()
        
        return StructDef(name=name, fields=fields)
    
    def parse_enum(self) -> EnumDef:
        self.consume(TokenType.ENUM)
        name = self.consume(TokenType.IDENT).value
        
        self.skip_newlines()
        self.consume(TokenType.INDENT)
        
        variants = []
        while not self.match(TokenType.DEDENT, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.DEDENT, TokenType.EOF):
                break
            
            var_name = self.consume(TokenType.IDENT).value

            # Handle enum variant value: Name = value
            value = None
            if self.match(TokenType.EQ):
                self.advance()
                value = self.parse_expression()

            fields = []
            if self.match(TokenType.LPAREN):
                self.advance()
                if not self.match(TokenType.RPAREN):
                    while True:
                        fields.append(self.parse_type())
                        if not self.match(TokenType.COMMA):
                            break
                        self.advance()
                self.consume(TokenType.RPAREN)

            variants.append(EnumVariant(var_name, fields, value))
            self.skip_newlines()
        
        if self.match(TokenType.DEDENT):
            self.advance()
        
        return EnumDef(name=name, variants=variants)
    
    def parse_impl(self) -> ImplBlock:
        self.consume(TokenType.IMPL)
        
        # Check for trait impl: impl Trait for Type
        first_name = self.consume(TokenType.IDENT).value
        
        trait_name = None
        type_name = first_name
        
        if self.match(TokenType.FOR):
            self.advance()
            trait_name = first_name
            type_name = self.consume(TokenType.IDENT).value
        
        self.skip_newlines()
        self.consume(TokenType.INDENT)
        
        methods = []
        while not self.match(TokenType.DEDENT, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.DEDENT, TokenType.EOF):
                break
            
            # Collect attributes for method
            while self.match(TokenType.ATTRIBUTE):
                attr = self.advance().value
                self.pending_attributes.append(attr)
                self.skip_newlines()
            
            if self.match(TokenType.FN):
                method = self.parse_fn()
            elif self.match(TokenType.IDENT) and self.is_func_def():
                method = self.parse_func_def_simple()
            else:
                self.error("Expected method definition in impl block" + self.reserved_note())
            
            method.is_method = True
            method.self_type = type_name
            methods.append(method)
            self.skip_newlines()
        
        if self.match(TokenType.DEDENT):
            self.advance()
        
        return ImplBlock(type_name=type_name, methods=methods, trait_name=trait_name)
    
    def parse_trait(self) -> TraitDef:
        self.consume(TokenType.TRAIT)
        name = self.consume(TokenType.IDENT).value
        
        self.skip_newlines()
        self.consume(TokenType.INDENT)
        
        methods = []
        while not self.match(TokenType.DEDENT, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.DEDENT, TokenType.EOF):
                break
            
            if self.match(TokenType.FN):
                self.advance()
            if self.match(TokenType.IDENT) and self.peek().type == TokenType.LPAREN:
                method_name = self.advance().value
                method = self.parse_func_body(method_name, allow_signature=True)
            else:
                self.error("Expected method definition in trait" + self.reserved_note())
            
            methods.append(method)
            self.skip_newlines()
        
        if self.match(TokenType.DEDENT):
            self.advance()
        
        return TraitDef(name=name, methods=methods)
    
    def parse_import(self) -> Import:
        if self.match(TokenType.FROM):
            self.advance()
            module = self.consume(TokenType.IDENT).value
            while self.match(TokenType.DOT):
                self.advance()
                module += "." + self.consume(TokenType.IDENT).value
            
            self.consume(TokenType.IMPORT)
            
            items = []
            if self.match(TokenType.STAR):
                self.advance()
                items = None  # Import all
            else:
                while True:
                    items.append(self.consume(TokenType.IDENT).value)
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
            
            return Import(module=module, items=items)
        else:
            self.consume(TokenType.IMPORT)
            module = self.consume(TokenType.IDENT).value
            while self.match(TokenType.DOT):
                self.advance()
                module += "." + self.consume(TokenType.IDENT).value

            alias = None
            if self.match(TokenType.AS):
                self.advance()
                alias = self.consume(TokenType.IDENT).value

            return Import(module=module, alias=alias)
    
    def parse_type_alias(self) -> TypeAlias:
        self.consume(TokenType.TYPE)
        name = self.consume(TokenType.IDENT).value
        self.consume(TokenType.EQ)
        type_val = self.parse_type()
        return TypeAlias(name=name, type=type_val)
    
    def parse_const(self) -> VarDecl:
        self.consume(TokenType.CONST)
        name = self.consume(TokenType.IDENT).value
        
        type_ann = None
        if self.match(TokenType.COLON):
            self.advance()
            type_ann = self.parse_type()
        
        self.consume(TokenType.EQ)
        value = self.parse_expression()
        
        return VarDecl(name=name, type_ann=type_ann, value=value, mutable=False, is_const=True)
    
    # Expression parsing with precedence climbing.
    # Every node gets the source position of its operator / first token.
    def parse_expression(self) -> ASTNode:
        return self.parse_ternary()
    
    def parse_ternary(self) -> ASTNode:
        expr = self.parse_pipe()
        
        if self.match(TokenType.IF):
            if_tok = self.advance()
            condition = self.parse_pipe()
            self.consume(TokenType.ELSE)
            else_expr = self.parse_ternary()
            return self._mark(IfExpr(condition=condition, then_expr=expr, else_expr=else_expr), if_tok)
        
        return expr

    def parse_pipe(self) -> ASTNode:
        """Pipeline operator: `x |> f` == `f(x)`, `x |> f(a)` == `f(x, a)`."""
        left = self.parse_or()
        while self.match(TokenType.PIPE_GT):
            op_tok = self.advance()
            right = self.parse_or()
            if isinstance(right, Call):
                right.args.insert(0, left)
                left = right
            else:
                left = self._mark(Call(func=right, args=[left]), op_tok)
        return left
    
    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        while self.match(TokenType.OR):
            op_tok = self.advance()
            right = self.parse_and()
            left = self._mark(BinaryOp(op="or", left=left, right=right), op_tok)
        return left
    
    def parse_and(self) -> ASTNode:
        left = self.parse_not()
        while self.match(TokenType.AND):
            op_tok = self.advance()
            right = self.parse_not()
            left = self._mark(BinaryOp(op="and", left=left, right=right), op_tok)
        return left
    
    def parse_not(self) -> ASTNode:
        if self.match(TokenType.NOT):
            op_tok = self.advance()
            return self._mark(UnaryOp(op="not", operand=self.parse_not()), op_tok)
        return self.parse_comparison()
    
    def parse_comparison(self) -> ASTNode:
        """Comparisons chain like in Python/math: `0 <= i < n` means
        `0 <= i and i < n` (it used to silently mean `(0 <= i) < n`)."""
        left = self.parse_bitwise_or()
        result = None
        while self.match(TokenType.EQEQ, TokenType.NEQ, TokenType.LT,
                         TokenType.GT, TokenType.LTE, TokenType.GTE):
            op_tok = self.advance()
            right = self.parse_bitwise_or()
            cmp = self._mark(BinaryOp(op=op_tok.value, left=left, right=right), op_tok)
            result = cmp if result is None else self._mark(
                BinaryOp(op="and", left=result, right=cmp), op_tok)
            left = right
        return left if result is None else result
    
    def _binary_level(self, next_level, ops):
        left = next_level()
        while self.match(*ops):
            op_tok = self.advance()
            right = next_level()
            left = self._mark(BinaryOp(op=op_tok.value, left=left, right=right), op_tok)
        return left

    def parse_bitwise_or(self) -> ASTNode:
        return self._binary_level(self.parse_bitwise_xor, (TokenType.PIPE,))
    
    def parse_bitwise_xor(self) -> ASTNode:
        return self._binary_level(self.parse_bitwise_and, (TokenType.CARET,))
    
    def parse_bitwise_and(self) -> ASTNode:
        return self._binary_level(self.parse_shift, (TokenType.AMP,))
    
    def parse_shift(self) -> ASTNode:
        return self._binary_level(self.parse_range, (TokenType.SHL, TokenType.SHR))
    
    def parse_range(self) -> ASTNode:
        left = self.parse_additive()
        
        if self.match(TokenType.RANGE, TokenType.RANGE_INCL):
            op_tok = self.current()
            inclusive = op_tok.type == TokenType.RANGE_INCL
            self.advance()
            right = self.parse_additive()
            return self._mark(Range(start=left, end=right, inclusive=inclusive), op_tok)
        
        return left
    
    def parse_additive(self) -> ASTNode:
        return self._binary_level(self.parse_multiplicative, (TokenType.PLUS, TokenType.MINUS))
    
    def parse_multiplicative(self) -> ASTNode:
        return self._binary_level(self.parse_unary,
                                  (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT))
    
    def parse_unary(self) -> ASTNode:
        # Unary operators bind looser than `**` (like Python): -2 ** 2 == -(2 ** 2)
        tok = self.current()
        if self.match(TokenType.MINUS):
            self.advance()
            return self._mark(UnaryOp(op="-", operand=self.parse_unary()), tok)
        if self.match(TokenType.TILDE):
            self.advance()
            return self._mark(UnaryOp(op="~", operand=self.parse_unary()), tok)
        if self.match(TokenType.AMP):
            self.advance()
            mutable = False
            if self.match(TokenType.MUT):
                self.advance()
                mutable = True
            return self._mark(UnaryOp(op="&mut" if mutable else "&", operand=self.parse_unary()), tok)
        if self.match(TokenType.STAR):
            self.advance()
            return self._mark(UnaryOp(op="*", operand=self.parse_unary()), tok)
        
        return self.parse_power()

    def parse_power(self) -> ASTNode:
        left = self.parse_postfix()
        if self.match(TokenType.POWER):
            op_tok = self.advance()
            right = self.parse_unary()  # right associative; allows 2 ** -1
            left = self._mark(BinaryOp(op="**", left=left, right=right), op_tok)
        return left
    
    def parse_postfix(self) -> ASTNode:
        expr = self.parse_primary()
        
        while True:
            op_tok = self.current()
            if self.match(TokenType.LPAREN):
                # Function call
                self.advance()
                args = []
                if not self.match(TokenType.RPAREN):
                    while True:
                        args.append(self.parse_expression())
                        if not self.match(TokenType.COMMA):
                            break
                        self.advance()
                self.consume(TokenType.RPAREN)
                expr = self._mark(Call(func=expr, args=args), op_tok)
            
            elif self.match(TokenType.LBRACKET):
                # Index
                self.advance()
                index = self.parse_expression()
                self.consume(TokenType.RBRACKET)
                expr = self._mark(Index(obj=expr, index=index), op_tok)
            
            elif self.match(TokenType.DOT):
                # Attribute access (`self` and keywords are not valid here, plain names only)
                self.advance()
                attr = self.consume(TokenType.IDENT).value
                expr = self._mark(Attribute(obj=expr, attr=attr), op_tok)
            
            elif self.match(TokenType.AS):
                # Type cast: expr as Type
                self.advance()
                target_type = self.parse_type()
                expr = self._mark(Cast(expr=expr, target_type=target_type, type=target_type), op_tok)

            elif self.match(TokenType.QUESTION):
                # `?` - unwrap Option/Result, propagating None/Err to the caller
                self.advance()
                expr = self._mark(NullCheck(expr=expr), op_tok)

            else:
                break

        return expr
    
    def parse_primary(self) -> ASTNode:
        tok = self.current()
        return self._mark(self._parse_primary(), tok)

    def _parse_fstring_expr(self, text: str, line: int) -> ASTNode:
        # Parse with THIS module's Lexer/Parser. (A `from til import ...` here
        # loaded a second copy of the module when the compiler ran as __main__,
        # so f-string sub-expressions were AST nodes of a *different* class and
        # every isinstance() check in the code generator failed: f"Hello {name}"
        # printed the string's address as an integer.)
        sub_tokens = Lexer(text, "<fstring>", languages=self.languages,
                           line_offset=max(line - 1, 0)).tokenize()
        sub_parser = Parser(sub_tokens, "<fstring>", languages=self.languages)
        sub_parser.in_clause = self.in_clause       # `resume` means the same inside {...}
        return sub_parser.parse_expression()

    def _parse_primary(self) -> ASTNode:
        # Literals
        if self.match(TokenType.INT):
            return IntLit(value=self.advance().value)
        
        if self.match(TokenType.FLOAT):
            return FloatLit(value=self.advance().value)
        
        if self.match(TokenType.STRING):
            return StringLit(value=self.advance().value)

        if self.match(TokenType.CHAR):
            return CharLit(value=self.advance().value)

        if self.match(TokenType.FSTRING):
            token = self.advance()
            # Parse expression parts within the f-string
            parts = []
            for part in token.value:
                kind, text = part[0], part[1]
                if kind == 'str':
                    parts.append(('str', StringLit(value=text, line=token.line, col=token.col)))
                else:
                    expr = self._parse_fstring_expr(text, token.line)
                    if len(part) > 2:
                        parts.append(('expr', expr, part[2]))   # with format spec
                    else:
                        parts.append(('expr', expr))
            return FStringLit(parts=parts)

        if self.match(TokenType.BOOL):
            return BoolLit(value=self.advance().value)
        
        # Parenthesized expression or tuple
        if self.match(TokenType.LPAREN):
            self.advance()
            if self.match(TokenType.RPAREN):
                self.advance()
                return ArrayLit(elements=[])  # Empty tuple
            
            expr = self.parse_expression()
            
            if self.match(TokenType.COMMA):
                # Tuple: (a, b, c)
                elements = [expr]
                while self.match(TokenType.COMMA):
                    self.advance()
                    if self.match(TokenType.RPAREN):
                        break
                    elements.append(self.parse_expression())
                self.consume(TokenType.RPAREN)
                return TupleLit(elements=elements)
            
            self.consume(TokenType.RPAREN)
            return expr
        
        # Array literal
        if self.match(TokenType.LBRACKET):
            self.advance()
            elements = []
            
            if not self.match(TokenType.RBRACKET):
                first = self.parse_expression()
                
                # Check for list comprehension: [expr for x in iter if cond]
                if self.match(TokenType.FOR):
                    self.advance()
                    var = self.consume(TokenType.IDENT).value
                    self.consume(TokenType.IN)
                    # Parse only non-ternary expression for iter to avoid consuming 'if'
                    iter_expr = self.parse_or()

                    condition = None
                    if self.match(TokenType.IF):
                        self.advance()
                        condition = self.parse_or()

                    self.consume(TokenType.RBRACKET)
                    return ListComprehension(expr=first, var=var, iter=iter_expr, condition=condition)
                
                elements.append(first)
                while self.match(TokenType.COMMA):
                    self.advance()
                    if self.match(TokenType.RBRACKET):
                        break
                    elements.append(self.parse_expression())
            
            self.consume(TokenType.RBRACKET)
            return ArrayLit(elements=elements)
        
        # Dict literal
        if self.match(TokenType.LBRACE):
            self.advance()
            pairs = []
            
            if not self.match(TokenType.RBRACE):
                while True:
                    key = self.parse_expression()
                    self.consume(TokenType.COLON)
                    value = self.parse_expression()
                    pairs.append((key, value))
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
            
            self.consume(TokenType.RBRACE)
            return DictLit(pairs=pairs)
        
        # Lambda: |x, y| x + y  OR  |x: int| -> int { x + 1 }
        if self.match(TokenType.PIPE):
            self.advance()
            params = []
            if not self.match(TokenType.PIPE):
                while True:
                    name = self.consume(TokenType.IDENT).value
                    type_ann = None
                    if self.match(TokenType.COLON):
                        self.advance()
                        type_ann = self.parse_type()
                    params.append((name, type_ann))
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
            self.consume(TokenType.PIPE)
            # Optional return type annotation
            ret_type = None
            if self.match(TokenType.ARROW):
                self.advance()
                ret_type = self.parse_type()
            # Body: either { expr } or just expr
            saved_clause, self.in_clause = self.in_clause, False   # a lambda is not the clause
            if self.match(TokenType.LBRACE):
                self.advance()
                self.skip_newlines()
                body = self.parse_expression()
                self.skip_newlines()
                self.consume(TokenType.RBRACE)
            else:
                body = self.parse_expression()
            self.in_clause = saved_clause
            return Lambda(params=params, body=body, ret_type=ret_type)
        
        # Self keyword (used in method bodies)
        if self.match(TokenType.SELF):
            self.advance()
            return Identifier(name="self")

        # contextual effect keywords (TIL 3.1): localized perform/handle, resume
        kind = self.contextual_start() if self.match(TokenType.IDENT) else None
        if kind == TokenType.HANDLE:
            return self.parse_handle_expr()
        if kind == TokenType.PERFORM:
            return self.parse_perform()
        if kind == TokenType.RESUME:
            return self.parse_resume()

        # None literal
        if self.match(TokenType.IDENT) and self.current().value == "None":
            self.advance()
            return Identifier(name="NULL")

        # Identifier or struct init
        if self.match(TokenType.IDENT):
            name = self.advance().value

            # Vec<T> or HashMap<K,V> generic constructors
            if name in ("Vec", "HashMap") and self.match(TokenType.LT):
                self.advance()  # <
                type_args = []
                while True:
                    type_args.append(self.consume(TokenType.IDENT).value)
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
                self.consume(TokenType.GT)
                encoded = f"__{name}__{'_'.join(type_args)}"
                return Identifier(name=encoded)

            # Struct initialization: Name { field: value }
            if self.match(TokenType.LBRACE):
                self.advance()
                fields = {}
                
                if not self.match(TokenType.RBRACE):
                    while True:
                        field_name = self.consume(TokenType.IDENT).value
                        self.consume(TokenType.COLON)
                        field_value = self.parse_expression()
                        fields[field_name] = field_value
                        if not self.match(TokenType.COMMA):
                            break
                        self.advance()
                
                self.consume(TokenType.RBRACE)
                return StructInit(name=name, fields=fields)
            
            return Identifier(name=name)

        # Match expression (can appear as expression: let r = match x ...)
        if self.match(TokenType.MATCH):
            return self.parse_match()

        # V2.0 expression forms
        if self.match(TokenType.RETRY):
            return self.parse_retry_expr()

        if self.match(TokenType.TIMEOUT):
            return self.parse_timeout_expr()

        if self.match(TokenType.HANDLE):
            return self.parse_handle_expr()

        # effects (TIL 3.1): `perform E(args)` is an expression, resume only in clauses
        if self.match(TokenType.PERFORM):
            return self.parse_perform()

        self.error(f"Unexpected token: {self.current().type.name}")

# ═══════════════════════════════════════════════════════════════════════════════
#                              TYPE CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Symbol:
    name: str
    type: Type
    mutable: bool = True
    is_const: bool = False
    level: int = 2

class TypeChecker:
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]
        self.functions: Dict[str, FunctionType] = {}
        self.structs: Dict[str, StructType] = {}
        self.current_function: Optional[FuncDef] = None
        self.errors: List[str] = []
    
    def error(self, msg: str, node: ASTNode = None):
        if node:
            self.errors.append(f"Line {node.line}: {msg}")
        else:
            self.errors.append(msg)
    
    def enter_scope(self):
        self.scopes.append({})
    
    def exit_scope(self):
        self.scopes.pop()
    
    def define(self, name: str, type: Type, mutable: bool = True, is_const: bool = False):
        self.scopes[-1][name] = Symbol(name, type, mutable, is_const)
    
    def lookup(self, name: str) -> Optional[Symbol]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def check(self, program: Program) -> List[str]:
        # First pass: collect all type definitions
        for stmt in program.statements:
            if isinstance(stmt, StructDef):
                fields = {f.name: f.type for f in stmt.fields}
                self.structs[stmt.name] = StructType(stmt.name, fields)
            elif isinstance(stmt, FuncDef):
                param_types = [p.type for p in stmt.params]
                self.functions[stmt.name] = FunctionType(param_types, stmt.ret_type)
        
        # Second pass: type check
        for stmt in program.statements:
            self.check_node(stmt)
        
        return self.errors
    
    def check_node(self, node: ASTNode) -> Type:
        method = f"check_{type(node).__name__}"
        if hasattr(self, method):
            return getattr(self, method)(node)
        return T_UNKNOWN
    
    def check_Program(self, node: Program) -> Type:
        # First pass: collect trait definitions
        self._trait_defs: Dict[str, TraitDef] = {}
        for stmt in node.statements:
            if isinstance(stmt, TraitDef):
                self._trait_defs[stmt.name] = stmt

        for stmt in node.statements:
            self.check_node(stmt)
        return T_VOID
    
    def check_FuncDef(self, node: FuncDef) -> Type:
        self.current_function = node
        self.enter_scope()
        
        for param in node.params:
            self.define(param.name, param.type, param.mutable)
        
        self.check_node(node.body)
        
        self.exit_scope()
        self.current_function = None
        return T_VOID
    
    def check_Block(self, node: Block) -> Type:
        for stmt in node.statements:
            self.check_node(stmt)
        return T_VOID
    
    def check_VarDecl(self, node: VarDecl) -> Type:
        var_type = node.type_ann or T_UNKNOWN
        
        if node.value:
            value_type = self.check_node(node.value)
            if var_type == T_UNKNOWN:
                var_type = value_type
            elif not self.types_compatible(var_type, value_type):
                self.error(f"Type mismatch: expected {var_type}, got {value_type}", node)
        
        self.define(node.name, var_type, node.mutable, node.is_const)
        return T_VOID
    
    def check_Assignment(self, node: Assignment) -> Type:
        target_type = self.check_node(node.target)
        value_type = self.check_node(node.value)
        
        if isinstance(node.target, Identifier):
            sym = self.lookup(node.target.name)
            if sym and not sym.mutable:
                self.error(f"Cannot assign to immutable variable '{node.target.name}'", node)
        
        return T_VOID
    
    def check_If(self, node: If) -> Type:
        cond_type = self.check_node(node.condition)
        if cond_type != T_BOOL and cond_type != T_UNKNOWN:
            self.error(f"Condition must be bool, got {cond_type}", node)
        
        self.check_node(node.then_body)
        
        for elif_cond, elif_body in node.elifs:
            self.check_node(elif_cond)
            self.check_node(elif_body)
        
        if node.else_body:
            self.check_node(node.else_body)
        
        return T_VOID
    
    def check_For(self, node: For) -> Type:
        iter_type = self.check_node(node.iter)
        
        elem_type = T_UNKNOWN
        if isinstance(iter_type, ArrayType):
            elem_type = iter_type.element_type
        elif isinstance(node.iter, Range):
            elem_type = T_INT
        
        self.enter_scope()
        self.define(node.var, elem_type)
        self.check_node(node.body)
        self.exit_scope()
        
        return T_VOID
    
    def check_While(self, node: While) -> Type:
        self.check_node(node.condition)
        self.check_node(node.body)
        return T_VOID
    
    def check_Return(self, node: Return) -> Type:
        if node.value:
            return self.check_node(node.value)
        return T_VOID
    
    def check_BinaryOp(self, node: BinaryOp) -> Type:
        left = self.check_node(node.left)
        right = self.check_node(node.right)
        
        if node.op in ('+', '-', '*', '/', '%', '**'):
            if left in (T_INT, T_FLOAT) or right in (T_INT, T_FLOAT):
                node.type = T_FLOAT if T_FLOAT in (left, right) else T_INT
                return node.type
        
        if node.op in ('==', '!=', '<', '>', '<=', '>='):
            node.type = T_BOOL
            return T_BOOL
        
        if node.op in ('and', 'or'):
            node.type = T_BOOL
            return T_BOOL
        
        node.type = left
        return left
    
    def check_UnaryOp(self, node: UnaryOp) -> Type:
        operand = self.check_node(node.operand)
        
        if node.op == '-':
            node.type = operand
        elif node.op == 'not':
            node.type = T_BOOL
        elif node.op == '&':
            node.type = PointerType(operand)
        elif node.op == '*':
            if isinstance(operand, PointerType):
                node.type = operand.pointee
            else:
                node.type = T_UNKNOWN
        else:
            node.type = operand
        
        return node.type
    
    def check_Call(self, node: Call) -> Type:
        if isinstance(node.func, Identifier):
            name = node.func.name
            
            # Check for built-in functions
            builtins = {
                'print': T_VOID, 'println': T_VOID,
                'len': T_INT, 'str': T_STR, 'int': T_INT, 'float': T_FLOAT,
                'bool': T_BOOL, 'abs': T_FLOAT, 'sqrt': T_FLOAT,
                'min': T_FLOAT, 'max': T_FLOAT, 'sum': T_FLOAT,
                'input': T_STR, 'type': T_STR,
            }
            
            if name in builtins:
                node.type = builtins[name]
                return node.type
            
            if name in self.functions:
                node.type = self.functions[name].ret
                return node.type
            
            # Struct constructor
            if name in self.structs:
                node.type = self.structs[name]
                return node.type
        
        for arg in node.args:
            self.check_node(arg)
        
        return T_UNKNOWN
    
    def check_Index(self, node: Index) -> Type:
        obj_type = self.check_node(node.obj)
        self.check_node(node.index)
        
        if isinstance(obj_type, ArrayType):
            node.type = obj_type.element_type
        elif obj_type == T_STR:
            node.type = T_CHAR
        else:
            node.type = T_UNKNOWN
        
        return node.type
    
    def check_Attribute(self, node: Attribute) -> Type:
        obj_type = self.check_node(node.obj)
        
        if isinstance(obj_type, StructType):
            if node.attr in obj_type.fields:
                node.type = obj_type.fields[node.attr]
                return node.type
        
        return T_UNKNOWN
    
    def check_Identifier(self, node: Identifier) -> Type:
        sym = self.lookup(node.name)
        if sym:
            node.type = sym.type
            return sym.type
        return T_UNKNOWN
    
    def check_IntLit(self, node: IntLit) -> Type:
        return T_INT
    
    def check_FloatLit(self, node: FloatLit) -> Type:
        return T_FLOAT
    
    def check_StringLit(self, node: StringLit) -> Type:
        return T_STR
    
    def check_BoolLit(self, node: BoolLit) -> Type:
        return T_BOOL
    
    def check_ArrayLit(self, node: ArrayLit) -> Type:
        if not node.elements:
            node.type = ArrayType(T_UNKNOWN)
            return node.type
        
        elem_type = self.check_node(node.elements[0])
        for elem in node.elements[1:]:
            self.check_node(elem)
        
        node.type = ArrayType(elem_type, len(node.elements))
        return node.type
    
    def check_Range(self, node: Range) -> Type:
        self.check_node(node.start)
        self.check_node(node.end)
        node.type = ArrayType(T_INT)
        return node.type
    
    def check_StructDef(self, node: StructDef) -> Type:
        return T_VOID
    
    def check_ImplBlock(self, node: ImplBlock) -> Type:
        # Check if this impl is for a trait
        if node.trait_name and hasattr(self, '_trait_defs') and node.trait_name in self._trait_defs:
            trait = self._trait_defs[node.trait_name]
            impl_method_names = {m.name for m in node.methods}
            for trait_method in trait.methods:
                if trait_method.name not in impl_method_names:
                    self.errors.append(
                        f"Missing trait method '{trait_method.name}' in impl {node.trait_name} for {node.type_name}")

        for method in node.methods:
            self.check_node(method)
        return T_VOID
    
    def types_compatible(self, expected: Type, actual: Type) -> bool:
        if expected == T_UNKNOWN or actual == T_UNKNOWN:
            return True
        if expected == actual:
            return True
        # Numeric promotion
        if expected == T_FLOAT and actual == T_INT:
            return True
        return False

# ═══════════════════════════════════════════════════════════════════════════════
#                     SEMANTIC VERIFIER (effects, capabilities, contracts)
# ═══════════════════════════════════════════════════════════════════════════════

def iter_children(node):
    """Yield the direct AST children of a node (works for every node class)."""
    if not isinstance(node, ASTNode) and not isinstance(node, (FuncParam, StructField)):
        return
    for value in vars(node).values():
        if isinstance(value, ASTNode):
            yield value
        elif isinstance(value, dict):            # StructInit.fields: {name: value}
            for item in value.values():
                if isinstance(item, ASTNode):
                    yield item
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, ASTNode):
                    yield item
                elif isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, ASTNode):
                            yield sub


def walk(node):
    """Depth-first iteration over a subtree (node included)."""
    stack = [node]
    while stack:
        n = stack.pop()
        if n is None:
            continue
        yield n
        stack.extend(reversed(list(iter_children(n))))


INT64_MASK = (1 << 64) - 1

def wrap_int64(x: int) -> int:
    """Two's-complement 64-bit wrap-around (TIL `int` semantics)."""
    x &= INT64_MASK
    return x - (1 << 64) if x >= (1 << 63) else x

def til_idiv(a: int, b: int) -> int:
    """Integer division truncating toward zero (C semantics, not Python's floor)."""
    q = abs(a) // abs(b)
    return wrap_int64(q if (a >= 0) == (b >= 0) else -q)

def til_imod(a: int, b: int) -> int:
    """Remainder with the sign of the dividend (C semantics)."""
    return wrap_int64(a - til_idiv(a, b) * b)


class NotConstant(Exception):
    pass


def const_eval(node, env: Dict[str, Any]):
    """Evaluate an expression at compile time, or raise NotConstant."""
    if isinstance(node, (IntLit, FloatLit, StringLit, BoolLit)):
        return node.value
    if isinstance(node, CharLit):
        return ord(node.value)
    if isinstance(node, Identifier):
        if node.name in env:
            return env[node.name]
        raise NotConstant(node.name)
    if isinstance(node, UnaryOp):
        v = const_eval(node.operand, env)
        if node.op == '-' and isinstance(v, (int, float)) and not isinstance(v, bool):
            return wrap_int64(-v) if isinstance(v, int) else -v
        if node.op == 'not':
            return not v
        if node.op == '~' and type(v) is int:      # the VMs reject ~ on a bool
            return wrap_int64(~v)
        raise NotConstant(node.op)
    if isinstance(node, BinaryOp):
        if node.op == 'and':
            return bool(const_eval(node.left, env)) and bool(const_eval(node.right, env))
        if node.op == 'or':
            return bool(const_eval(node.left, env)) or bool(const_eval(node.right, env))
        a, b = const_eval(node.left, env), const_eval(node.right, env)
        num = (int, float)
        if node.op in ('==', '!='):
            # the VMs' equality: a bool equals only a bool (`true == 1` is false)
            if isinstance(a, bool) or isinstance(b, bool):
                eq = type(a) is type(b) and a == b
            else:
                eq = a == b
            return eq if node.op == '==' else not eq
        if node.op in ('<', '>', '<=', '>=') and type(a) in num and type(b) in num:
            return {'<': a < b, '>': a > b, '<=': a <= b, '>=': a >= b}[node.op]
        if type(a) is int and type(b) is int:
            if node.op == '+': return wrap_int64(a + b)
            if node.op == '-': return wrap_int64(a - b)
            if node.op == '*': return wrap_int64(a * b)
            if node.op in ('/', '%'):
                if b == 0:
                    raise NotConstant("division by zero")
                return til_idiv(a, b) if node.op == '/' else til_imod(a, b)
        if type(a) in num and type(b) in num and node.op in ('+', '-', '*', '/'):
            if node.op == '/' and b == 0:
                raise NotConstant("division by zero")
            return {'+': a + b, '-': a - b, '*': a * b, '/': a / b if b else 0}[node.op]
        if node.op == '+' and isinstance(a, str) and isinstance(b, str):
            return a + b
        raise NotConstant(node.op)
    if isinstance(node, Call) and isinstance(node.func, Identifier) and node.func.name == 'len' \
            and len(node.args) == 1:
        v = const_eval(node.args[0], env)
        if isinstance(v, str):
            return len(v.encode('utf-8'))
    raise NotConstant(type(node).__name__)


class SemanticChecker:
    """Compile-time verification shared by both backends.

    * Effects (gradual): a function's effects are inferred from its body
      (print/input -> io, `perform X` -> X, plus everything its callees do).
      `#[pure]` / `#[effects: ...]` declare an upper bound that is ENFORCED.
    * Capabilities: a function marked `#[cap: X]` may only be called (or
      referenced) by functions that also hold X; `main` holds every capability.
    * Contracts: calls whose arguments are compile-time constants are checked
      against `#[requires: ...]` at compile time.
    * Arity: calls to known functions must pass a valid number of arguments.
    """

    IO_BUILTINS = {'print', 'println', 'input', 'read_file', 'write_file', 'file_exists',
                   'screen_init', 'screen_clear', 'draw_pixel', 'draw_line', 'draw_rect',
                   'fill_rect', 'draw_circle', 'draw_text', 'get_pixel', 'screen_show',
                   'key_pressed', 'wait_key', 'energy_mode', 'screen_width', 'screen_height'}
    # effects of builtins: no `handle` clause intercepts them (print never performs)
    BUILTIN_EFFECTS = frozenset({'io'})

    def __init__(self, source: str = "", filename: str = "<stdin>"):
        self.reporter = ErrorReporter(source, filename)
        self.errors: List[str] = []

    def error(self, node, msg: str, code: str = "", hint: str = ""):
        line = getattr(node, 'line', 0) or 0
        col = getattr(node, 'col', 0) or 1
        self.errors.append(self.reporter.format_error(line, col, msg, code, hint))

    @staticmethod
    def describe(key: str) -> str:
        return f"'{key}'"

    def check(self, program: Program) -> List[str]:
        self.funcs: Dict[str, FuncDef] = {}
        self.methods_by_name: Dict[str, List[str]] = {}
        self.struct_names: Set[str] = set()
        self.enum_names: Set[str] = set()
        self.module_names = set(getattr(program, 'module_names', set()) or ())
        self.consts: Dict[str, Any] = {}
        for stmt in program.statements:
            if isinstance(stmt, FuncDef):
                self.funcs[stmt.name] = stmt
            elif isinstance(stmt, StructDef):
                self.struct_names.add(stmt.name)
            elif isinstance(stmt, EnumDef):
                self.enum_names.add(stmt.name)
            elif isinstance(stmt, ImplBlock):
                for m in stmt.methods:
                    key = f"{stmt.type_name}.{m.name}"
                    self.funcs[key] = m
                    self.methods_by_name.setdefault(m.name, []).append(key)
            elif isinstance(stmt, VarDecl) and stmt.value is not None and not stmt.mutable:
                try:
                    self.consts[stmt.name] = const_eval(stmt.value, self.consts)
                except NotConstant:
                    pass

        # Per function: base effects and resolved call edges
        self.base_effects: Dict[str, Set[str]] = {}
        self.edges: Dict[str, List[Tuple[str, ASTNode, bool]]] = {}  # (callee, node, precise)
        self.effect_origin: Dict[Tuple[str, str], ASTNode] = {}
        # id(call node) -> effects handled by the `handle` expressions around it
        self.edge_handled: Dict[int, frozenset] = {}
        # function -> [(E, perform node)]: performs that may call the function E
        self.perform_uses: Dict[str, List[Tuple[str, ASTNode]]] = {}
        for key, fn in self.funcs.items():
            self.scan_function(key, fn)

        self.check_effects()
        self.check_capabilities()
        return self.errors

    # ── call graph ──────────────────────────────────────────────────────────
    def resolve_call(self, owner_key: str, call: Call) -> List[Tuple[str, bool]]:
        f = call.func
        if isinstance(f, Identifier):
            if f.name in self.funcs and '.' not in f.name:
                return [(f.name, True)]
            return []
        if isinstance(f, Attribute):
            obj = f.obj
            if isinstance(obj, Identifier):
                if obj.name in self.struct_names or obj.name in self.enum_names:
                    key = f"{obj.name}.{f.attr}"
                    return [(key, True)] if key in self.funcs else []
                if obj.name in self.module_names and f.attr in self.funcs:
                    return [(f.attr, True)]
                if obj.name == "self" and '.' in owner_key:
                    key = f"{owner_key.split('.')[0]}.{f.attr}"
                    if key in self.funcs:
                        return [(key, True)]
            candidates = self.methods_by_name.get(f.attr, [])
            return [(c, len(candidates) == 1) for c in candidates]
        return []

    @classmethod
    def handled_effects(cls, body, through_lambdas: bool = True) -> Dict[int, frozenset]:
        """id(node) -> effects handled by the `handle ... with` expressions that
        lexically enclose the node. A clause body only sees the handles outside its
        own handle (the owner is masked while its clause runs). Lambdas inside a
        handle body count as handled (they are assumed to run inside it) unless
        `through_lambdas` is false: a lambda may escape the handle and run later.
        Builtin effects (io) are never handled: no clause intercepts print."""
        out: Dict[int, frozenset] = {}
        todo: List[Tuple[Any, frozenset]] = [(body, frozenset())]
        while todo:
            node, handled = todo.pop()
            if node is None:
                continue
            if not through_lambdas and isinstance(node, Lambda):
                handled = frozenset()
            if handled:
                out[id(node)] = handled
            if isinstance(node, EffectHandler):
                names = frozenset(c.name for c in node.handlers) - cls.BUILTIN_EFFECTS
                todo.append((node.expr, handled | names))
                todo.extend((c, handled) for c in node.handlers)
                continue
            todo.extend((child, handled) for child in iter_children(node))
        return out

    def scan_function(self, key: str, fn: FuncDef):
        effects: Set[str] = set()
        edges: List[Tuple[str, ASTNode, bool]] = []
        callee_nodes = set()
        performs: List[PerformEffect] = []      # performs whose default is a function
        handled = self.handled_effects(fn.body)
        # a user function or a parameter with a builtin's name is what gets called
        # (its own effects count through the call edge), not the io builtin
        shadowing = {p.name for p in fn.params}
        for node in walk(fn.body):
            if isinstance(node, Call):
                if isinstance(node.func, Identifier) and node.func.name in self.IO_BUILTINS \
                        and node.func.name not in self.funcs and node.func.name not in shadowing:
                    effects.add('io')
                    self.effect_origin.setdefault((key, 'io'), node)
                for callee, precise in self.resolve_call(key, node):
                    edges.append((callee, node, precise))
                    if precise:
                        self.check_call_site(callee, node)
                if id(node) in handled:
                    self.edge_handled[id(node)] = handled[id(node)]
                callee_nodes.add(id(node.func))
            elif isinstance(node, PerformEffect):
                if node.name in self.funcs and '.' not in node.name:
                    performs.append(node)
                if node.name in handled.get(id(node), ()):
                    continue          # handled by an enclosing `handle` in this function
                effects.add(node.name)
                self.effect_origin.setdefault((key, node.name), node)
        # Taking a function as a value (e.g. `let f = read_file`) counts as a
        # reference for capability purposes: no capability laundering.
        for node in walk(fn.body):
            if isinstance(node, Identifier) and id(node) not in callee_nodes \
                    and node.name in self.funcs and self.funcs[node.name].caps:
                edges.append((node.name, node, True))
                if id(node) in handled:
                    self.edge_handled[id(node)] = handled[id(node)]
        # `perform E(...)` calls the top-level function E when no handler is active:
        # for capabilities that is a use of E, unless a handle with a clause for E
        # surrounds the perform in this function (not through a lambda, which may
        # escape the handle). Effects are unchanged: perform E is the effect E.
        if performs:
            strict = self.handled_effects(fn.body, through_lambdas=False)
            self.perform_uses[key] = [(n.name, n) for n in performs
                                      if n.name not in strict.get(id(n), ())]
        self.base_effects[key] = effects
        self.edges[key] = edges

    # ── effects ─────────────────────────────────────────────────────────────
    def declared(self, key: str) -> Optional[Set[str]]:
        fn = self.funcs[key]
        if fn.declared_effects is None:
            return None
        return set(fn.declared_effects)

    def check_effects(self):
        # Interface effects: declared if present, else inferred (fixpoint).
        inferred = {k: set(v) for k, v in self.base_effects.items()}
        changed = True
        while changed:
            changed = False
            for key, edges in self.edges.items():
                for callee, _node, _precise in edges:
                    callee_eff = self.declared(callee)
                    if callee_eff is None:
                        callee_eff = inferred.get(callee, set())
                    new = callee_eff - self.edge_handled.get(id(_node), frozenset()) - inferred[key]
                    if new:
                        inferred[key] |= new
                        changed = True
        for key, fn in self.funcs.items():
            allowed = self.declared(key)
            if allowed is None:
                continue
            actual = set(self.base_effects[key])
            for callee, node, _precise in self.edges[key]:
                callee_eff = self.declared(callee)
                if callee_eff is None:
                    callee_eff = inferred.get(callee, set())
                for eff in callee_eff - allowed - self.edge_handled.get(id(node), frozenset()):
                    if eff not in actual:
                        actual.add(eff)
                        self.effect_origin.setdefault((key, eff), node)
                        self.effect_origin[(key, eff, 'via')] = callee
            for eff in sorted(actual - allowed):
                origin = self.effect_origin.get((key, eff), fn)
                via = self.effect_origin.get((key, eff, 'via'))
                how = f" (through the call to '{via}')" if via else ""
                if 'pure' in (fn.effects or []) and not fn.declared_effects:
                    msg = f"pure function '{key}' performs effect '{eff}'{how}"
                    hint = f"remove #[pure] or declare #[effects: {eff}]"
                else:
                    msg = (f"function '{key}' performs effect '{eff}'{how}, "
                           f"but declares only #[effects: {', '.join(sorted(allowed)) or 'none'}]")
                    hint = f"add '{eff}' to the #[effects: ...] list"
                self.error(origin, msg, "E-EFFECT", hint)

    # ── capabilities ────────────────────────────────────────────────────────
    def check_capabilities(self):
        for key, edges in self.edges.items():
            if key == "main":
                continue  # the program entry point holds all authority
            held = set(self.funcs[key].caps)
            uses = [(callee, node) for callee, node, precise in edges if precise]
            uses += self.perform_uses.get(key, [])
            for callee, node in uses:
                missing = [c for c in self.funcs[callee].caps if c not in held]
                for cap in missing:
                    hint = f"add #[cap: {cap}] to '{key}' (only main holds every capability)"
                    if isinstance(node, PerformEffect):
                        hint = (f"`perform {callee}(...)` calls the function '{callee}' when no "
                                "handler is active; " + hint)
                    self.error(node,
                               f"'{key}' uses '{callee}', which requires capability '{cap}', "
                               f"but '{key}' does not hold it", "E-CAP", hint)

    # ── call sites: arity + constant contracts ─────────────────────────────
    def check_call_site(self, callee: str, call: Call):
        fn = self.funcs[callee]
        params = [p for p in fn.params if p.name != "self"]
        is_instance_call = (isinstance(call.func, Attribute) and not (
            isinstance(call.func.obj, Identifier) and (
                call.func.obj.name in self.struct_names or call.func.obj.name in self.module_names)))
        if len(fn.params) != len(params) and not is_instance_call:
            return  # static reference to an instance method; leave to the backend
        required = sum(1 for p in params if p.default is None)
        if not (required <= len(call.args) <= len(params)):
            expected = f"{required}" if required == len(params) else f"{required} to {len(params)}"
            self.error(call, f"'{callee}' expects {expected} argument(s), got {len(call.args)}",
                       "E-ARITY")
            return
        if not fn.requires:
            return
        env = dict(self.consts)
        try:
            for i, p in enumerate(params):
                arg = call.args[i] if i < len(call.args) else p.default
                env[p.name] = const_eval(arg, self.consts)
        except NotConstant:
            return  # not all arguments are known: checked at run time instead
        for expr, text in zip(fn.requires, fn.requires_text or [""] * len(fn.requires)):
            try:
                ok = const_eval(expr, env)
            except NotConstant:
                continue
            if not ok:
                shown = ", ".join(f"{p.name}={env[p.name]!r}" for p in params)
                self.error(call, f"contract violation: call to '{callee}' ({shown}) "
                                 f"breaks #[requires: {text}]",
                           "E-CONTRACT", "proved false at compile time; fix the arguments")


class CompileError(Exception):
    """A fatal, already formatted, compile-time error."""
    pass


class NativeUnsupported(CompileError):
    """The native (C) backend cannot compile this program, but the TIL VM can:
    `til run` falls back to the VM, `til build` reports it."""
    pass


# ── gradual type inference (TIL 3.1, src/tiltypes.py; optional) ─────────────────
def types_module():
    """The type inference / typed native backend module, or None when this
    installation does not have it (then: no E-TYPE checks, classic C backend)."""
    if os.environ.get('TIL_NO_TYPES'):
        return None
    try:
        return feature_module('tiltypes')
    except ImportError as e:
        if getattr(e, 'name', None) != 'tiltypes':
            raise
        return None


def type_errors(program: 'Program', source: str, filename: str) -> List[str]:
    """E-TYPE errors (definite type errors only; empty when tiltypes is missing)."""
    tt = types_module()
    if tt is None:
        return []
    try:
        return tt.check_program(program, source, filename)
    except Exception as e:          # an inference bug must never reject a program
        if os.environ.get('TIL_DEBUG_TYPES'):
            raise
        print(localize(f"[TIL] type inference skipped (internal error: {type(e).__name__}: {e})"), file=sys.stderr)
        return []

# ═══════════════════════════════════════════════════════════════════════════════
#                              CODE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def c_string_literal(text: str) -> str:
    """Encode text as a C string literal. Every byte that is not printable
    ASCII is emitted as an octal escape, so no input can terminate the literal
    or inject code (the old escaper missed \\r, \\0 and quote/backslash order)."""
    out = ['"']
    for byte in text.encode('utf-8'):
        ch = chr(byte)
        if ch == '"':
            out.append('\\"')
        elif ch == '\\':
            out.append('\\\\')
        elif ch == '\n':
            out.append('\\n')
        elif ch == '\t':
            out.append('\\t')
        elif 32 <= byte < 127 and ch != '?':   # '?' avoided: no C trigraphs
            out.append(ch)
        else:
            out.append('\\%03o' % byte)
    out.append('"')
    return ''.join(out)


def c_format_spec(spec: str, ptype: str) -> Tuple[str, str]:
    """Translate an f-string spec (`.2f`, `5`, `x`, `08.3f`) into a printf format.
    Returns (format, 'float'|'int')."""
    m = re.match(r'^([<>^]?)(\d*)(?:\.(\d+))?([fdxXobe%]?)$', spec)
    if not m:
        return ("%g", "float") if ptype == "float" else ("%lld", "int")
    align, width, prec, conv = m.groups()
    if conv == '%':
        return (f"%.{prec or 6}f%%", "percent")
    flags = '-' if align == '<' else ''
    if width.startswith('0') and len(width) > 1:
        flags += '0'
    w = width.lstrip('0') if width.startswith('0') else width
    if conv in ('f', 'e', '%') or (not conv and (prec or ptype == "float")):
        c = conv if conv in ('f', 'e') else ('f' if prec else 'g')
        p = f".{prec}" if prec else ""
        return (f"%{flags}{w}{p}{c}", "float")
    c = {'x': 'llx', 'X': 'llX', 'o': 'llo', 'b': 'lld'}.get(conv, 'lld')
    return (f"%{flags}{w}{c}", "int")


def c_comment(text: str) -> str:
    """Make arbitrary text safe inside a /* ... */ C comment."""
    return '/* ' + text.replace('*/', '* /').replace('\n', ' ').replace('\r', ' ') + ' */'


class CCodeGenerator:
    def __init__(self):
        self.output: List[str] = []
        self.indent = 0
        self.structs: Dict[str, StructDef] = {}
        self.enums: Dict[str, EnumDef] = {}
        self.functions: Dict[str, FuncDef] = {}
        self.impl_methods: Dict[str, List[FuncDef]] = {}
        self.declared_vars: Set[str] = set()
        self.string_vars: Set[str] = set()
        self.float_vars: Set[str] = set()
        self.bool_vars: Set[str] = set()
        self.struct_vars: Dict[str, str] = {}  # var_name -> struct_type_name
        self.array_vars: Dict[str, int] = {}  # name -> length
        self.array_elem_c: Dict[str, str] = {}  # array name -> C element type
        self._global_string_vars: Set[str] = set()
        self._global_array_vars: Dict[str, int] = {}
        self._global_array_elem_c: Dict[str, str] = {}
        self._global_float_vars: Set[str] = set()
        self._global_bool_vars: Set[str] = set()
        self.current_level = 2
        self.in_method: bool = False
        self.current_struct: Optional[str] = None
        # V2: container tracking
        self.dynarray_vars: Dict[str, str] = {}  # var_name -> elem_type
        self.hashmap_vars: Dict[str, tuple] = {}  # var_name -> (key_type, val_type)
        # V2: lambda/closure tracking
        self._lambda_var_map: Dict[str, str] = {}  # var_name -> lambda_c_name
        self._lambda_captures: Dict[str, list] = {}  # lambda_name -> [capture_names]
        self._lambda_ret_types: Dict[str, str] = {}  # lambda_c_name -> C return type
        # V2: contract tracking
        self._current_ensures: list = []
        self._current_ensures_ret_type: str = "int64_t"
        self._in_ensures: bool = False
        self._current_ret_type: Optional[Type] = None   # TIL return type of the function being generated
        self.optresult_vars: Dict[str, str] = {}        # var -> TIL_Option_x / TIL_Result_x
        self.module_names: Set[str] = set()             # imported module names (for mod.func())
        self._recover_stack: List[Tuple[str, int]] = []  # active retry/timeout recovery points
        self._loop_depth = 0
        self._uses_timeout = False
    
    @staticmethod
    def mangle_name(name: str) -> str:
        """Mangle a Unicode identifier to a valid C identifier."""
        if name.isascii() and name.replace('_', 'a').isalnum():
            return name  # Already a valid C identifier
        # Convert non-ASCII chars to _uXXXX hex encoding
        result = []
        for ch in name:
            if ch.isascii() and (ch.isalnum() or ch == '_'):
                result.append(ch)
            else:
                result.append(f"_u{ord(ch):04x}")
        mangled = ''.join(result)
        # Ensure it starts with a letter or underscore
        if mangled and mangled[0].isdigit():
            mangled = '_' + mangled
        return mangled

    def emit(self, line: str):
        self.output.append("    " * self.indent + line)

    def emit_raw(self, line: str):
        self.output.append(line)
    
    def generate(self, program: Program) -> str:
        self.output = []
        self.module_names = set(getattr(program, 'module_names', set()) or ())
        self._uses_timeout = any(isinstance(n, TimeoutExpr) for st in program.statements for n in walk(st))
        # Effect handlers with resumption (TIL 3.1) only run on the TIL VM.
        for st in program.statements:
            for n in walk(st):
                if isinstance(n, (EffectHandler, ResumeExpr)):
                    raise NativeUnsupported(
                        f"line {n.line}: the native backend does not support effect handlers "
                        "(handle ... with / resume); run it on the TIL VM (til run, til run --vm) "
                        "or build bytecode (til build --target vm)")

        # First pass: collect definitions
        self._globals: List[VarDecl] = []
        self._type_aliases: Dict[str, TypeAlias] = {}
        self._trait_defs: Dict[str, 'TraitDef'] = {}
        for stmt in program.statements:
            if isinstance(stmt, StructDef):
                self.structs[stmt.name] = stmt
            elif isinstance(stmt, FuncDef):
                self.functions[stmt.name] = stmt
            elif isinstance(stmt, EnumDef):
                self.enums[stmt.name] = stmt
            elif isinstance(stmt, ImplBlock):
                if stmt.type_name not in self.impl_methods:
                    self.impl_methods[stmt.type_name] = []
                self.impl_methods[stmt.type_name].extend(stmt.methods)
            elif isinstance(stmt, VarDecl):
                self._globals.append(stmt)
            elif isinstance(stmt, TypeAlias):
                self._type_aliases[stmt.name] = stmt
            elif isinstance(stmt, TraitDef):
                self._trait_defs[stmt.name] = stmt
            elif isinstance(stmt, CapabilityType):
                if not hasattr(self, '_capabilities'):
                    self._capabilities = {}
                self._capabilities[stmt.name] = stmt

        # Build function return type map for print inference
        self._func_ret_types: Dict[str, str] = {}
        for name, func in self.functions.items():
            self._func_ret_types[name] = self.type_to_c(func.ret_type)
        for type_name, methods in self.impl_methods.items():
            for method in methods:
                self._func_ret_types[f"{type_name}_{method.name}"] = self.type_to_c(method.ret_type)

        self._deferred_global_inits: List[VarDecl] = []

        # Initialize lambda storage
        self._lambda_defs = []
        self._lambda_counter = 0

        # Generate code
        self.emit_header()
        self.emit_types()
        self.emit_enums()
        # Helpers define Option/Result/Vec/HashMap typedefs, so they must come
        # before type aliases and function prototypes that mention those types.
        self.emit_helpers()
        self.emit_optresult_methods()
        self.emit_type_aliases()
        self.emit_forward_declarations()
        self.emit_lambdas()
        self.emit_structs()
        self.emit_globals()
        self.emit_capabilities()
        self.emit_functions(program)
        # Emit any lambdas created during code generation
        if self._lambda_defs:
            # Lambdas need to be before the functions that use them
            # Re-insert them before the functions section
            self._fixup_lambdas()
        self.emit_main_wrapper()
        
        return '\n'.join(self.output)
    
    def emit_header(self):
        self.emit_raw("// TIL v2.0 - Generated C Code")
        self.emit_raw("// Author: Alisher Beisembekov")
        self.emit_raw("// Multi-Level Programming: Mixed Martial Programming")
        self.emit_raw("")
        self.emit_raw("#include <stdio.h>")
        self.emit_raw("#include <stdlib.h>")
        self.emit_raw("#include <string.h>")
        self.emit_raw("#include <stdbool.h>")
        self.emit_raw("#include <stdint.h>")
        self.emit_raw("#include <math.h>")
        self.emit_raw("")
    
    def emit_types(self):
        self.emit_raw("// TIL Type Definitions")
        self.emit_raw("typedef struct TIL_String { char* data; size_t len; size_t cap; } TIL_String;")
        self.emit_raw("typedef struct TIL_Array { void* data; size_t len; size_t cap; size_t elem_size; } TIL_Array;")
        self.emit_raw("")
    
    def emit_enums(self):
        if not self.enums:
            return
        self.emit_raw("// Enum Definitions")
        for name, enum_def in self.enums.items():
            self.emit_raw(f"typedef enum {{")
            for i, variant in enumerate(enum_def.variants):
                suffix = "," if i < len(enum_def.variants) - 1 else ""
                if variant.value is not None:
                    val = self.generate_node(variant.value)
                    self.emit_raw(f"    {self.mangle_name(name)}_{self.mangle_name(variant.name)} = {val}{suffix}")
                else:
                    self.emit_raw(f"    {self.mangle_name(name)}_{self.mangle_name(variant.name)} = {i}{suffix}")
            self.emit_raw(f"}} {self.mangle_name(name)};")
            self.emit_raw("")

    def emit_lambdas(self):
        """Emit lambda function definitions (placeholder, filled during codegen)."""
        pass

    def _fixup_lambdas(self):
        """Insert lambda definitions before the TIL Functions section."""
        if not self._lambda_defs:
            return
        # Find the "// TIL Functions" marker and insert lambdas before it
        marker = "// TIL Functions"
        for i, line in enumerate(self.output):
            if line.strip() == marker:
                insert_lines = ["// Lambda Functions"]
                insert_lines.extend(self._lambda_defs)
                insert_lines.append("")
                self.output[i:i] = insert_lines
                break

    def emit_forward_declarations(self):
        self.emit_raw("// Forward Declarations")

        # Forward declare structs (so methods can reference them)
        for name in self.structs:
            c_name = self.mangle_name(name)
            self.emit_raw(f"typedef struct {c_name} {c_name};")

        # Forward declare functions
        for name, func in self.functions.items():
            c_name = self.mangle_name(name)
            ret = self.type_to_c(func.ret_type)
            params = self.params_to_c(func.params)
            level = func.level
            level_names = {0: "Hardware", 1: "Systems", 2: "Safe", 3: "Script", 4: "Formal"}
            self.emit_raw(f"{ret} til_{c_name}({params});  // Level {level}: {level_names.get(level, '?')}")

        # Forward declare all impl methods
        for type_name, methods in self.impl_methods.items():
            for method in methods:
                ret = self.type_to_c(method.ret_type)
                has_self = any(p.name == "self" for p in method.params)
                params = []
                if has_self:
                    params.append(f"{self.mangle_name(type_name)}* self")
                for p in method.params:
                    if p.name != "self":
                        params.append(f"{self.type_to_c(p.type)} {self.mangle_name(p.name)}")
                params_str = ", ".join(params) if params else "void"
                self.emit_raw(f"static {ret} {self.mangle_name(type_name)}_{self.mangle_name(method.name)}({params_str});")

        self.emit_raw("")
    
    def emit_helpers(self):
        self.emit_raw("// Built-in Functions")
        self.emit_raw("""
// Print functions
static void til_print_int(int64_t x) { printf("%lld\\n", (long long)x); }
static void til_print_float(double x) { printf("%g\\n", x); }
static void til_print_str(const char* s) { printf("%s\\n", s); }
static void til_print_bool(bool b) { printf("%s\\n", b ? "true" : "false"); }
static void til_print_char(char c) { printf("%c\\n", c); }

// Math functions
static double til_sqrt(double x) { return sqrt(x); }
static double til_abs(double x) { return fabs(x); }
static double til_pow(double base, double exp) { return pow(base, exp); }
static double til_sin(double x) { return sin(x); }
static double til_cos(double x) { return cos(x); }
static double til_tan(double x) { return tan(x); }
static double til_log(double x) { return log(x); }
static double til_exp(double x) { return exp(x); }
static double til_floor(double x) { return floor(x); }
static double til_ceil(double x) { return ceil(x); }
static double til_round(double x) { return round(x); }

// Integer power (exact, wraps like other int64 arithmetic)
static int64_t til_ipow(int64_t base, int64_t exp) {
    if (exp < 0) return (base == 1) ? 1 : (base == -1) ? ((exp & 1) ? -1 : 1) : 0;
    uint64_t result = 1, b = (uint64_t)base;
    while (exp > 0) { if (exp & 1) result *= b; b *= b; exp >>= 1; }
    return (int64_t)result;
}

// Panic: every runtime error goes through here, so `retry`/`timeout`
// blocks can recover from it (setjmp/longjmp based recovery points).
#include <setjmp.h>
#include <time.h>
typedef struct TIL_Recover { jmp_buf env; double deadline; struct TIL_Recover* prev; } TIL_Recover;
static TIL_Recover* til_recover_top = NULL;
static const char* til_panic_msg = "";
static char til_panic_buf[256];
static void til_panic(const char* msg) {
    til_panic_msg = msg;
    if (til_recover_top) longjmp(til_recover_top->env, 1);
    fflush(stdout);
    fprintf(stderr, "%s\\n", msg);
    exit(1);
}
static double til_now_ms(void) { return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC; }
// Called at loop back-edges and function entries when the program uses `timeout`:
// unwinds straight to the OUTERMOST expired timeout block.
static void til_check_deadline(void) {
    static unsigned til_tick = 0;
    TIL_Recover* expired = NULL;
    double now;
    if (!til_recover_top || (++til_tick & 255u) != 0) return;
    now = til_now_ms();
    for (TIL_Recover* r = til_recover_top; r; r = r->prev)
        if (r->deadline > 0 && now > r->deadline) expired = r;
    if (expired) {
        til_panic_msg = "timeout";
        til_recover_top = expired;
        longjmp(expired->env, 1);
    }
}

// Value -> string conversions (for "text" + number, f-strings, str())
static char* til_int_to_str(int64_t x) { char* b = (char*)malloc(24); snprintf(b, 24, "%lld", (long long)x); return b; }
static char* til_float_to_str(double x) { char* b = (char*)malloc(32); snprintf(b, 32, "%g", x); return b; }
static const char* til_bool_to_str(bool x) { return x ? "true" : "false"; }
static char* til_fmt_float(double x, const char* fmt) { char* b = (char*)malloc(64); snprintf(b, 64, fmt, x); return b; }
static char* til_fmt_int(int64_t x, const char* fmt) { char* b = (char*)malloc(64); snprintf(b, 64, fmt, (long long)x); return b; }

// Min/max
static int64_t til_min_int(int64_t a, int64_t b) { return a < b ? a : b; }
static int64_t til_max_int(int64_t a, int64_t b) { return a > b ? a : b; }
static double til_min_float(double a, double b) { return a < b ? a : b; }
static double til_max_float(double a, double b) { return a > b ? a : b; }

// Array helpers
static size_t til_len_array(TIL_Array* arr) { return arr ? arr->len : 0; }
static size_t til_len_str(const char* s) { return s ? strlen(s) : 0; }

// String helpers
static char* til_str_concat(const char* a, const char* b) {
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    char* result = (char*)malloc(len_a + len_b + 1);
    strcpy(result, a);
    strcat(result, b);
    return result;
}

static bool til_str_eq(const char* a, const char* b) {
    return strcmp(a, b) == 0;
}

// String methods
static bool til_str_contains(const char* s, const char* sub) {
    return strstr(s, sub) != NULL;
}

static bool til_str_starts_with(const char* s, const char* prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static bool til_str_ends_with(const char* s, const char* suffix) {
    size_t slen = strlen(s), suflen = strlen(suffix);
    if (suflen > slen) return false;
    return strcmp(s + slen - suflen, suffix) == 0;
}

static char* til_str_trim(const char* s) {
    while (*s == ' ' || *s == '\\t' || *s == '\\n' || *s == '\\r') s++;
    size_t len = strlen(s);
    while (len > 0 && (s[len-1] == ' ' || s[len-1] == '\\t' || s[len-1] == '\\n' || s[len-1] == '\\r')) len--;
    char* result = (char*)malloc(len + 1);
    strncpy(result, s, len);
    result[len] = '\\0';
    return result;
}

static char* til_str_to_upper(const char* s) {
    size_t len = strlen(s);
    char* result = (char*)malloc(len + 1);
    for (size_t i = 0; i < len; i++)
        result[i] = (s[i] >= 'a' && s[i] <= 'z') ? s[i] - 32 : s[i];
    result[len] = '\\0';
    return result;
}

static char* til_str_to_lower(const char* s) {
    size_t len = strlen(s);
    char* result = (char*)malloc(len + 1);
    for (size_t i = 0; i < len; i++)
        result[i] = (s[i] >= 'A' && s[i] <= 'Z') ? s[i] + 32 : s[i];
    result[len] = '\\0';
    return result;
}

static char* til_str_replace(const char* s, const char* old, const char* new_s) {
    size_t slen = strlen(s), olen = strlen(old), nlen = strlen(new_s);
    size_t count = 0;
    const char* p = s;
    while ((p = strstr(p, old)) != NULL) { count++; p += olen; }
    char* result = (char*)malloc(slen + count * (nlen - olen) + 1);
    char* w = result;
    p = s;
    while (*p) {
        if (strncmp(p, old, olen) == 0) {
            memcpy(w, new_s, nlen);
            w += nlen;
            p += olen;
        } else {
            *w++ = *p++;
        }
    }
    *w = '\\0';
    return result;
}

static char* til_str_slice(const char* s, int64_t start, int64_t end) {
    size_t len = strlen(s);
    if (start < 0) start = 0;
    if (end > (int64_t)len) end = len;
    if (start >= end) { char* r = (char*)malloc(1); r[0] = '\\0'; return r; }
    size_t n = end - start;
    char* result = (char*)malloc(n + 1);
    strncpy(result, s + start, n);
    result[n] = '\\0';
    return result;
}

static int64_t til_str_find(const char* s, const char* sub) {
    const char* p = strstr(s, sub);
    return p ? (int64_t)(p - s) : -1;
}

// Dynamic array helpers
typedef struct TIL_IntArray { int64_t* data; size_t len; size_t cap; } TIL_IntArray;
typedef struct TIL_FloatArray { double* data; size_t len; size_t cap; } TIL_FloatArray;
typedef struct TIL_StrArray { const char** data; size_t len; size_t cap; } TIL_StrArray;

static TIL_IntArray til_int_array_new(size_t cap) {
    TIL_IntArray a; a.data = (int64_t*)malloc(sizeof(int64_t) * (cap > 0 ? cap : 8));
    a.len = 0; a.cap = cap > 0 ? cap : 8; return a;
}
static void til_int_array_push(TIL_IntArray* a, int64_t val) {
    if (a->len >= a->cap) { a->cap *= 2; a->data = (int64_t*)realloc(a->data, sizeof(int64_t) * a->cap); }
    a->data[a->len++] = val;
}
static int64_t til_int_array_pop(TIL_IntArray* a) {
    if (a->len == 0) { til_panic("Pop from empty array"); }
    return a->data[--a->len];
}
static int64_t til_int_array_get(TIL_IntArray* a, size_t i) {
    if (i >= a->len) { til_panic("Array index out of bounds"); }
    return a->data[i];
}

// Option<T> - tagged union for optional values
typedef struct TIL_Option_int { bool has_value; int64_t value; } TIL_Option_int;
typedef struct TIL_Option_float { bool has_value; double value; } TIL_Option_float;
typedef struct TIL_Option_str { bool has_value; const char* value; } TIL_Option_str;
typedef struct TIL_Option_bool { bool has_value; bool value; } TIL_Option_bool;

static TIL_Option_int til_Some_int(int64_t v) { return (TIL_Option_int){true, v}; }
static TIL_Option_float til_Some_float(double v) { return (TIL_Option_float){true, v}; }
static TIL_Option_str til_Some_str(const char* v) { return (TIL_Option_str){true, v}; }
static TIL_Option_bool til_Some_bool(bool v) { return (TIL_Option_bool){true, v}; }
static TIL_Option_int til_None_int(void) { return (TIL_Option_int){false, 0}; }
static TIL_Option_float til_None_float(void) { return (TIL_Option_float){false, 0.0}; }
static TIL_Option_str til_None_str(void) { return (TIL_Option_str){false, NULL}; }
static TIL_Option_bool til_None_bool(void) { return (TIL_Option_bool){false, false}; }

static int64_t til_unwrap_int(TIL_Option_int opt, const char* msg) {
    if (!opt.has_value) { snprintf(til_panic_buf, sizeof til_panic_buf, "Unwrap failed: %s", msg); til_panic(til_panic_buf); }
    return opt.value;
}
static double til_unwrap_float(TIL_Option_float opt, const char* msg) {
    if (!opt.has_value) { snprintf(til_panic_buf, sizeof til_panic_buf, "Unwrap failed: %s", msg); til_panic(til_panic_buf); }
    return opt.value;
}
static const char* til_unwrap_str(TIL_Option_str opt, const char* msg) {
    if (!opt.has_value) { snprintf(til_panic_buf, sizeof til_panic_buf, "Unwrap failed: %s", msg); til_panic(til_panic_buf); }
    return opt.value;
}

// Result<T, E> - tagged union for error handling
typedef struct TIL_Result_int { bool is_ok; int64_t value; const char* error; } TIL_Result_int;
typedef struct TIL_Result_float { bool is_ok; double value; const char* error; } TIL_Result_float;
typedef struct TIL_Result_str { bool is_ok; const char* value; const char* error; } TIL_Result_str;

static TIL_Result_int til_Ok_int(int64_t v) { return (TIL_Result_int){true, v, NULL}; }
static TIL_Result_float til_Ok_float(double v) { return (TIL_Result_float){true, v, NULL}; }
static TIL_Result_str til_Ok_str(const char* v) { return (TIL_Result_str){true, v, NULL}; }
typedef struct TIL_Result_bool { bool is_ok; bool value; const char* error; } TIL_Result_bool;
static TIL_Result_bool til_Ok_bool(bool v) { return (TIL_Result_bool){true, v, NULL}; }
static TIL_Result_int til_Err_int(const char* e) { return (TIL_Result_int){false, 0, e}; }
static TIL_Result_float til_Err_float(const char* e) { return (TIL_Result_float){false, 0.0, e}; }
static TIL_Result_str til_Err_str(const char* e) { return (TIL_Result_str){false, NULL, e}; }

// Memory (Level 1+)
static void* til_alloc(size_t size) { return malloc(size); }
static void til_free(void* ptr) { free(ptr); }
static void* til_realloc(void* ptr, size_t size) { return realloc(ptr, size); }

// Bounds checking (Level 2)
static void til_bounds_check(size_t index, size_t len, const char* msg) {
    if (index >= len) {
        snprintf(til_panic_buf, sizeof til_panic_buf, "Bounds check failed: %s (index %zu, len %zu)", msg, index, len);
        til_panic(til_panic_buf);
    }
}

// ═══ Vec<T> - Dynamic Arrays ═══
typedef struct TIL_DynArray_int { int64_t* data; size_t len; size_t cap; } TIL_DynArray_int;
static TIL_DynArray_int til_dynarray_int_new() {
    TIL_DynArray_int a; a.data = (int64_t*)malloc(sizeof(int64_t) * 8);
    a.len = 0; a.cap = 8; return a;
}
static void til_dynarray_int_push(TIL_DynArray_int* a, int64_t val) {
    if (a->len >= a->cap) { a->cap *= 2; a->data = (int64_t*)realloc(a->data, sizeof(int64_t) * a->cap); }
    a->data[a->len++] = val;
}
static int64_t til_dynarray_int_pop(TIL_DynArray_int* a) {
    if (a->len == 0) { til_panic("Pop from empty Vec"); }
    return a->data[--a->len];
}
static int64_t til_dynarray_int_get(TIL_DynArray_int* a, size_t i) {
    if (i >= a->len) { til_panic("Vec index out of bounds"); }
    return a->data[i];
}

typedef struct TIL_DynArray_float { double* data; size_t len; size_t cap; } TIL_DynArray_float;
static TIL_DynArray_float til_dynarray_float_new() {
    TIL_DynArray_float a; a.data = (double*)malloc(sizeof(double) * 8);
    a.len = 0; a.cap = 8; return a;
}
static void til_dynarray_float_push(TIL_DynArray_float* a, double val) {
    if (a->len >= a->cap) { a->cap *= 2; a->data = (double*)realloc(a->data, sizeof(double) * a->cap); }
    a->data[a->len++] = val;
}
static double til_dynarray_float_pop(TIL_DynArray_float* a) {
    if (a->len == 0) { til_panic("Pop from empty Vec"); }
    return a->data[--a->len];
}
static double til_dynarray_float_get(TIL_DynArray_float* a, size_t i) {
    if (i >= a->len) { til_panic("Vec index out of bounds"); }
    return a->data[i];
}

typedef struct TIL_DynArray_str { const char** data; size_t len; size_t cap; } TIL_DynArray_str;
static TIL_DynArray_str til_dynarray_str_new() {
    TIL_DynArray_str a; a.data = (const char**)malloc(sizeof(const char*) * 8);
    a.len = 0; a.cap = 8; return a;
}
static void til_dynarray_str_push(TIL_DynArray_str* a, const char* val) {
    if (a->len >= a->cap) { a->cap *= 2; a->data = (const char**)realloc(a->data, sizeof(const char*) * a->cap); }
    a->data[a->len++] = val;
}
static const char* til_dynarray_str_pop(TIL_DynArray_str* a) {
    if (a->len == 0) { til_panic("Pop from empty Vec"); }
    return a->data[--a->len];
}
static const char* til_dynarray_str_get(TIL_DynArray_str* a, size_t i) {
    if (i >= a->len) { til_panic("Vec index out of bounds"); }
    return a->data[i];
}

// ═══ HashMap<str, int/str> ═══
static size_t _til_hash_str(const char* s) {
    size_t h = 5381;
    while (*s) { h = ((h << 5) + h) + (unsigned char)*s++; }
    return h;
}

typedef struct TIL_HM_si_Entry { const char* key; int64_t value; bool occupied; } TIL_HM_si_Entry;
typedef struct TIL_HashMap_str_int { TIL_HM_si_Entry* entries; size_t cap; size_t len; } TIL_HashMap_str_int;

static TIL_HashMap_str_int til_hashmap_str_int_new() {
    TIL_HashMap_str_int m;
    m.cap = 64; m.len = 0;
    m.entries = (TIL_HM_si_Entry*)calloc(m.cap, sizeof(TIL_HM_si_Entry));
    return m;
}
static void til_hashmap_str_int_set(TIL_HashMap_str_int* m, const char* key, int64_t val) {
    size_t idx = _til_hash_str(key) % m->cap;
    for (size_t i = 0; i < m->cap; i++) {
        size_t j = (idx + i) % m->cap;
        if (!m->entries[j].occupied || (m->entries[j].key && strcmp(m->entries[j].key, key) == 0)) {
            if (!m->entries[j].occupied) m->len++;
            m->entries[j].key = key; m->entries[j].value = val; m->entries[j].occupied = true;
            return;
        }
    }
}
static int64_t til_hashmap_str_int_get(TIL_HashMap_str_int* m, const char* key) {
    size_t idx = _til_hash_str(key) % m->cap;
    for (size_t i = 0; i < m->cap; i++) {
        size_t j = (idx + i) % m->cap;
        if (!m->entries[j].occupied) break;
        if (strcmp(m->entries[j].key, key) == 0) return m->entries[j].value;
    }
    snprintf(til_panic_buf, sizeof til_panic_buf, "Key not found: %s", key); til_panic(til_panic_buf); return 0;
}
static bool til_hashmap_str_int_has(TIL_HashMap_str_int* m, const char* key) {
    size_t idx = _til_hash_str(key) % m->cap;
    for (size_t i = 0; i < m->cap; i++) {
        size_t j = (idx + i) % m->cap;
        if (!m->entries[j].occupied) return false;
        if (strcmp(m->entries[j].key, key) == 0) return true;
    }
    return false;
}
static void til_hashmap_str_int_delete(TIL_HashMap_str_int* m, const char* key) {
    size_t idx = _til_hash_str(key) % m->cap;
    for (size_t i = 0; i < m->cap; i++) {
        size_t j = (idx + i) % m->cap;
        if (!m->entries[j].occupied) return;
        if (strcmp(m->entries[j].key, key) == 0) { m->entries[j].occupied = false; m->len--; return; }
    }
}

typedef struct TIL_HM_ss_Entry { const char* key; const char* value; bool occupied; } TIL_HM_ss_Entry;
typedef struct TIL_HashMap_str_str { TIL_HM_ss_Entry* entries; size_t cap; size_t len; } TIL_HashMap_str_str;

static TIL_HashMap_str_str til_hashmap_str_str_new() {
    TIL_HashMap_str_str m;
    m.cap = 64; m.len = 0;
    m.entries = (TIL_HM_ss_Entry*)calloc(m.cap, sizeof(TIL_HM_ss_Entry));
    return m;
}
static void til_hashmap_str_str_set(TIL_HashMap_str_str* m, const char* key, const char* val) {
    size_t idx = _til_hash_str(key) % m->cap;
    for (size_t i = 0; i < m->cap; i++) {
        size_t j = (idx + i) % m->cap;
        if (!m->entries[j].occupied || (m->entries[j].key && strcmp(m->entries[j].key, key) == 0)) {
            if (!m->entries[j].occupied) m->len++;
            m->entries[j].key = key; m->entries[j].value = val; m->entries[j].occupied = true;
            return;
        }
    }
}
static const char* til_hashmap_str_str_get(TIL_HashMap_str_str* m, const char* key) {
    size_t idx = _til_hash_str(key) % m->cap;
    for (size_t i = 0; i < m->cap; i++) {
        size_t j = (idx + i) % m->cap;
        if (!m->entries[j].occupied) break;
        if (strcmp(m->entries[j].key, key) == 0) return m->entries[j].value;
    }
    snprintf(til_panic_buf, sizeof til_panic_buf, "Key not found: %s", key); til_panic(til_panic_buf); return NULL;
}
static bool til_hashmap_str_str_has(TIL_HashMap_str_str* m, const char* key) {
    size_t idx = _til_hash_str(key) % m->cap;
    for (size_t i = 0; i < m->cap; i++) {
        size_t j = (idx + i) % m->cap;
        if (!m->entries[j].occupied) return false;
        if (strcmp(m->entries[j].key, key) == 0) return true;
    }
    return false;
}
""")
        self.emit_raw("")
    
    def emit_optresult_methods(self):
        """unwrap / unwrap_or / expect for every Option/Result instantiation."""
        ctypes = {"int": "int64_t", "float": "double", "str": "const char*", "bool": "bool"}
        self.emit_raw("// Option/Result methods")
        for suffix, ct in ctypes.items():
            self.emit_raw(f"static {ct} til_opt_unwrap_{suffix}(TIL_Option_{suffix} o) "
                          f"{{ if (!o.has_value) til_panic(\"unwrap() called on None\"); return o.value; }}")
            self.emit_raw(f"static {ct} til_opt_expect_{suffix}(TIL_Option_{suffix} o, const char* m) "
                          f"{{ if (!o.has_value) til_panic(m); return o.value; }}")
            self.emit_raw(f"static {ct} til_opt_unwrap_or_{suffix}(TIL_Option_{suffix} o, {ct} d) "
                          f"{{ return o.has_value ? o.value : d; }}")
            self.emit_raw(f"static {ct} til_res_unwrap_{suffix}(TIL_Result_{suffix} r) "
                          f"{{ if (!r.is_ok) til_panic(r.error); return r.value; }}")
            self.emit_raw(f"static {ct} til_res_expect_{suffix}(TIL_Result_{suffix} r, const char* m) "
                          f"{{ if (!r.is_ok) til_panic(m); return r.value; }}")
            self.emit_raw(f"static {ct} til_res_unwrap_or_{suffix}(TIL_Result_{suffix} r, {ct} d) "
                          f"{{ return r.is_ok ? r.value : d; }}")
        self.emit_raw("")

    def emit_type_aliases(self):
        """Emit C typedef for type aliases."""
        if not self._type_aliases:
            return
        self.emit_raw("// Type Aliases")
        for name, alias in self._type_aliases.items():
            c_name = self.mangle_name(name)
            c_type = self.type_to_c(alias.type) if alias.type else "int64_t"
            self.emit_raw(f"typedef {c_type} {c_name};")
        self.emit_raw("")

    def emit_globals(self):
        """Emit module-level variables and constants as C static variables."""
        if not self._globals:
            return
        self.emit_raw("// Global Variables")
        for var in self._globals:
            c_type = self.infer_c_type(var)
            c_name = self.mangle_name(var.name)
            self._track_var_type(var, c_type)
            self.declared_vars.add(var.name)
            # Track global type info for print inference across functions
            if c_type == "const char*":
                self._global_string_vars.add(var.name)
            elif c_type in ("double", "float"):
                self._global_float_vars.add(var.name)
            elif c_type == "bool":
                self._global_bool_vars.add(var.name)
            if var.deferred:
                self.emit_raw(f"static {c_type} {c_name};")
                continue
            if var.value and not self._is_c_constant(var.value):
                # Not a C constant expression (calls, other globals, string ops):
                # declare here, initialize at program start (inside main).
                self.emit_raw(f"static {c_type} {c_name};")
                self._deferred_global_inits.append(var)
                continue
            if var.value:
                val = self.generate_node(var.value)
                if isinstance(var.value, ArrayLit):
                    elem_type = self.infer_array_elem_type(var.value)
                    n = len(var.value.elements)
                    self.emit_raw(f"static {elem_type} {c_name}[{n}] = {val};")
                    self.emit_raw(f"static size_t {c_name}_len = {n};")
                    self.array_vars[var.name] = n
                    self._global_array_vars[var.name] = n
                    self._global_array_elem_c[var.name] = elem_type
                else:
                    self.emit_raw(f"static {c_type} {c_name} = {val};")
            else:
                self.emit_raw(f"static {c_type} {c_name};")
        self.emit_raw("")

    def _is_c_constant(self, node) -> bool:
        """Is this TIL expression a valid C static initializer?"""
        if isinstance(node, (IntLit, FloatLit, StringLit, BoolLit, CharLit)):
            return True
        if isinstance(node, UnaryOp) and node.op in ('-', '~', 'not'):
            return self._is_c_constant(node.operand)
        if isinstance(node, BinaryOp) and node.op in ('-', '*', '/', '%', '&', '|', '^', '<<', '>>',
                                                       '==', '!=', '<', '>', '<=', '>=', 'and', 'or'):
            return self._is_c_constant(node.left) and self._is_c_constant(node.right) \
                and not (self._is_string_expr(node.left) or self._is_string_expr(node.right))
        if isinstance(node, BinaryOp) and node.op == '+':
            return (self._is_c_constant(node.left) and self._is_c_constant(node.right)
                    and not (self._is_string_expr(node.left) or self._is_string_expr(node.right)))
        if isinstance(node, ArrayLit):
            return all(self._is_c_constant(e) for e in node.elements)
        return False

    def emit_capabilities(self):
        """Emit capability type definitions."""
        caps = getattr(self, '_capabilities', {})
        if not caps:
            return
        self.emit_raw("// Capability Definitions")
        for name, cap in caps.items():
            self.gen_CapabilityType(cap)
        self.emit_raw("")

    def emit_structs(self):
        if not self.structs:
            return
        
        self.emit_raw("// Struct Definitions")
        
        for name, struct in self.structs.items():
            self.emit_raw(f"struct {self.mangle_name(name)} {{")
            for fld in struct.fields:
                c_type = self.type_to_c(fld.type)
                self.emit_raw(f"    {c_type} {fld.name};")
            self.emit_raw(f"}};")
            self.emit_raw("")

            # Constructor
            params = ", ".join(f"{self.type_to_c(f.type)} _{f.name}" for f in struct.fields)
            cname = self.mangle_name(name)
            self.emit_raw(f"static {cname} {cname}_create({params}) {{")
            self.emit_raw(f"    {cname} _self;")
            for fld in struct.fields:
                self.emit_raw(f"    _self.{fld.name} = _{fld.name};")
            self.emit_raw(f"    return _self;")
            self.emit_raw(f"}}")
            self.emit_raw("")
        
        # Methods
        for type_name, methods in self.impl_methods.items():
            for method in methods:
                self.emit_method(type_name, method)
    
    def emit_method(self, type_name: str, method: FuncDef):
        ret = self.type_to_c(method.ret_type)
        has_self = any(p.name == "self" for p in method.params)

        # Build parameter list
        params = []
        if has_self:
            params.append(f"{self.mangle_name(type_name)}* self")
        for p in method.params:
            if p.name != "self":
                params.append(f"{self.type_to_c(p.type)} {self.mangle_name(p.name)}")

        params_str = ", ".join(params) if params else "void"

        self.emit_raw(f"static {ret} {self.mangle_name(type_name)}_{self.mangle_name(method.name)}({params_str}) {{")
        self.indent += 1

        # Save and set context
        old_declared = self.declared_vars
        old_string = self.string_vars
        old_float = self.float_vars
        old_bool = self.bool_vars
        old_struct = self.struct_vars
        old_in_method = self.in_method
        old_current_struct = self.current_struct

        self.declared_vars = {"self"} if has_self else set()
        self.string_vars = set(self._global_string_vars)
        self.float_vars = set(self._global_float_vars)
        self.bool_vars = set(self._global_bool_vars)
        self.struct_vars = {}
        self.in_method = has_self
        self.current_struct = type_name if has_self else None

        # Track parameter types
        for p in method.params:
            if p.name != "self":
                self.declared_vars.add(p.name)
                self._track_param_type(p)

        self.generate_node(method.body)

        # Restore context
        self.declared_vars = old_declared
        self.string_vars = old_string
        self.float_vars = old_float
        self.bool_vars = old_bool
        self.struct_vars = old_struct
        self.in_method = old_in_method
        self.current_struct = old_current_struct

        self.indent -= 1
        self.emit_raw("}")
        self.emit_raw("")

    def _track_param_type(self, p):
        """Track parameter type for print/string inference."""
        if isinstance(p.type, PrimitiveType):
            if p.type.name == "str":
                self.string_vars.add(p.name)
            elif p.type.name in ("float", "f32", "f64"):
                self.float_vars.add(p.name)
            elif p.type.name == "bool":
                self.bool_vars.add(p.name)
        elif isinstance(p.type, StructType):
            self.struct_vars[p.name] = p.type.name
    
    def emit_functions(self, program: Program):
        self.emit_raw("// TIL Functions")
        
        for stmt in program.statements:
            if isinstance(stmt, FuncDef):
                self.generate_function(stmt)
    
    def generate_function(self, func: FuncDef):
        self.current_level = func.level
        self.declared_vars = set()
        self.string_vars = set(self._global_string_vars)
        self.float_vars = set(self._global_float_vars)
        self.bool_vars = set(self._global_bool_vars)
        self.struct_vars = {}
        self.array_vars = dict(self._global_array_vars)
        self.array_elem_c = dict(self._global_array_elem_c)
        self.dynarray_vars = {}
        self.hashmap_vars = {}
        self.optresult_vars = {}
        self._lambda_var_map = {}
        self.in_method = False
        self.current_struct = None

        # Add parameters to declared vars and track types
        for p in func.params:
            self.declared_vars.add(p.name)
            self._track_param_type(p)

        ret = self.type_to_c(func.ret_type)
        params = self.params_to_c(func.params)

        # Level-specific attributes
        attrs = []
        level_names = {0: "Hardware/SIMD", 1: "Systems", 2: "Safe", 3: "Script", 4: "Formal"}

        if func.level == 0:
            attrs.append("__attribute__((always_inline)) inline")
        elif func.level == 1:
            attrs.append("inline")

        # Effect system: #[pure] adds __attribute__((pure))
        if func.effects and 'pure' in func.effects:
            attrs.append("__attribute__((pure))")

        energy_level = None
        cap_annotations = []
        generic_params = []
        cross_level_info = None

        for attr in func.attributes:
            if attr == "inline" and "inline" not in ' '.join(attrs):
                attrs.append("inline")
            elif attr == "noinline":
                attrs.append("__attribute__((noinline))")
            elif attr == "hot":
                attrs.append("__attribute__((hot))")
            elif attr == "cold":
                attrs.append("__attribute__((cold))")
            elif attr.startswith("energy:"):
                energy_level = attr[len("energy:"):].strip()
                # Energy-aware: low = -O0, medium = -O1, high = -O3
                energy_opts = {"low": "__attribute__((optimize(\"Os\")))",
                               "medium": "",
                               "high": "__attribute__((optimize(\"O3\")))"}
                opt = energy_opts.get(energy_level, "")
                if opt:
                    attrs.append(opt)
            elif attr.startswith("generic:"):
                generic_params = [t.strip() for t in attr[len("generic:"):].split(",")]
            elif attr.startswith("cross_level:"):
                cross_level_info = attr

        # Capability annotations from effects
        if func.effects:
            for eff in func.effects:
                if eff.startswith("cap:"):
                    cap_annotations.append(eff[4:])

        attr_str = ' '.join(attrs)
        if attr_str:
            attr_str += ' '

        c_name = self.mangle_name(func.name)

        # Generate comments for advanced annotations
        comments = [f"// Level {func.level}: {level_names.get(func.level, 'Unknown')}"]
        if energy_level:
            comments.append(f"// Energy: {energy_level}")
        if cap_annotations:
            comments.append(f"// Capabilities: {', '.join(cap_annotations)}")
        if generic_params:
            comments.append(f"// Generic<{', '.join(generic_params)}>")
        if cross_level_info:
            comments.append(f"// {cross_level_info}")
        for c in comments:
            self.emit_raw(c)
        self.emit_raw(f"{attr_str}{ret} til_{c_name}({params}) {{")
        self.indent += 1

        # Emit capability checks
        if cap_annotations:
            for cap in cap_annotations:
                cap_c = self.mangle_name(cap)
                self.emit(f'// Capability check: {cap}')

        # Cross-level validation
        if cross_level_info:
            self.emit(f'// Cross-level contract: {cross_level_info}')

        if self._uses_timeout and func.name != "main":
            self.emit("til_check_deadline();")

        # Program start: initialize globals whose values are not C constants
        if func.name == "main":
            for var in self._deferred_global_inits:
                self.emit(f"{self.mangle_name(var.name)} = {self.generate_node(var.value)};")

        # Emit requires contract checks at function entry
        if func.requires:
            for k, req_expr in enumerate(func.requires):
                req_code = self.generate_node(req_expr)
                text = func.requires_text[k] if k < len(func.requires_text) else ""
                msg = f"contract violation: {func.name} requires {text}".strip()
                self.emit(f'if (!({req_code})) {{ til_panic({c_string_literal(msg)}); }}')

        # Set up ensures for gen_Return to use
        old_ensures = self._current_ensures
        old_ensures_ret = self._current_ensures_ret_type
        old_ret_type = self._current_ret_type
        self._current_ensures = func.ensures if func.ensures else []
        self._current_ensures_text = func.ensures_text or []
        self._current_func_name = func.name
        self._current_ensures_ret_type = ret
        self._current_ret_type = func.ret_type

        self.generate_node(func.body)

        # Restore ensures state
        self._current_ensures = old_ensures
        self._current_ensures_ret_type = old_ensures_ret
        self._current_ret_type = old_ret_type
        
        # Add default return if needed
        if isinstance(func.ret_type, VoidType) or (isinstance(func.ret_type, PrimitiveType) and func.ret_type.name == "void"):
            pass
        elif func.name == "main":
            pass
        else:
            # Check if last statement is a return
            if isinstance(func.body, Block) and func.body.statements:
                last = func.body.statements[-1]
                if not isinstance(last, Return):
                    self.emit("return 0;")
        
        self.indent -= 1
        self.emit_raw("}")
        self.emit_raw("")
    
    def emit_main_wrapper(self):
        if "main" in self.functions:
            self.emit_raw("// Main Entry Point")
            self.emit_raw("int main(int argc, char** argv) {")
            self.emit_raw("    til_main();")
            self.emit_raw("    return 0;")
            self.emit_raw("}")
    
    def generate_node(self, node: ASTNode) -> str:
        method = f"gen_{type(node).__name__}"
        if hasattr(self, method):
            return getattr(self, method)(node)
        return ""
    
    def gen_Block(self, node: Block) -> str:
        for stmt in node.statements:
            self.generate_node(stmt)
        return ""
    
    def gen_VarDecl(self, node: VarDecl) -> str:
        c_type = self.infer_c_type(node)
        c_name = self.mangle_name(node.name)

        # Track variable types for print/method inference
        self._track_var_type(node, c_type)

        # Handle lambda assignment: don't emit a C variable, just track mapping
        if isinstance(node.value, Lambda):
            lambda_name = self.generate_node(node.value)
            self._lambda_var_map[node.name] = lambda_name
            self.declared_vars.add(node.name)
            return ""

        if node.name in self.declared_vars:
            # Just assign
            if node.value:
                val = self.generate_node(node.value)
                self.emit(f"{c_name} = {val};")
        else:
            self.declared_vars.add(node.name)

            if isinstance(node.value, ArrayLit):
                self.array_vars[node.name] = len(node.value.elements)

            if isinstance(node.value, ListComprehension):
                self.array_vars[node.name] = 0  # dynamic

            if node.value:
                val = self.generate_node(node.value)

                # Handle array initialization
                if isinstance(node.value, ArrayLit):
                    elem_type = self.infer_array_elem_type(node.value)
                    n = len(node.value.elements)
                    self.array_elem_c[node.name] = elem_type
                    self.emit(f"{elem_type} {c_name}[{n if n else 1}] = {val if n else '{0}'};")
                    self.emit(f"size_t {c_name}_len = {n};")
                elif isinstance(node.value, ListComprehension):
                    # Comprehension already emitted the array; just assign the pointer and len
                    self.emit(f"{c_type} {c_name} = {val};")
                    self.emit(f"size_t {c_name}_len = {val}_len;")
                else:
                    self.emit(f"{c_type} {c_name} = {val};")
            else:
                self.emit(f"{c_type} {c_name};")

        return ""
    
    def _track_var_type(self, node: VarDecl, c_type: str):
        """Track variable type for print/method call inference."""
        if c_type == "const char*":
            self.string_vars.add(node.name)
        elif c_type in ("double", "float"):
            self.float_vars.add(node.name)
        elif c_type == "bool":
            self.bool_vars.add(node.name)
        # Track struct type from struct init, constructor call, or type annotation
        if isinstance(node.value, StructInit):
            self.struct_vars[node.name] = node.value.name
        elif isinstance(node.value, Call):
            # Check for Type.method() calls like Point.new(...)
            if isinstance(node.value.func, Attribute) and isinstance(node.value.func.obj, Identifier):
                sname = node.value.func.obj.name
                if sname in self.structs:
                    self.struct_vars[node.name] = sname
            # Check struct constructor call like Point(...)
            elif isinstance(node.value.func, Identifier) and node.value.func.name in self.structs:
                self.struct_vars[node.name] = node.value.func.name
        elif node.type_ann and isinstance(node.type_ann, StructType):
            self.struct_vars[node.name] = node.type_ann.name
        # Check c_type against known struct names (C names may be mangled)
        sname = self._struct_of_ctype(c_type)
        if sname:
            self.struct_vars[node.name] = sname
        # Track enum types
        if c_type in self.enums:
            pass  # enums are int-like in C
        if c_type.startswith("TIL_Option_") or c_type.startswith("TIL_Result_"):
            self.optresult_vars[node.name] = c_type
        # Track Vec<T> dynamic array variables
        if c_type.startswith("TIL_DynArray_"):
            elem_type = c_type[len("TIL_DynArray_"):]
            self.dynarray_vars[node.name] = elem_type
        # Track HashMap<K,V> variables
        elif c_type.startswith("TIL_HashMap_"):
            parts = c_type[len("TIL_HashMap_"):].split("_", 1)
            if len(parts) == 2:
                self.hashmap_vars[node.name] = (parts[0], parts[1])

    def gen_Assignment(self, node: Assignment) -> str:
        target = self.generate_node(node.target)
        value = self.generate_node(node.value)

        # `s += x` on a string lowers to `s = til_str_concat(s, x)`
        if node.op == "+=" and self._is_string_expr(node.target):
            self.emit(f"{target} = til_str_concat({target}, {self._to_c_string(node.value, value)});")
            return ""
        
        if node.op == "=":
            self.emit(f"{target} = {value};")
        else:
            self.emit(f"{target} {node.op} {value};")
        
        return ""
    
    def gen_If(self, node: If) -> str:
        cond = self.generate_node(node.condition)
        self.emit(f"if ({cond}) {{")
        self.indent += 1
        self.generate_node(node.then_body)
        self.indent -= 1
        
        for elif_cond, elif_body in node.elifs:
            cond = self.generate_node(elif_cond)
            self.emit(f"}} else if ({cond}) {{")
            self.indent += 1
            self.generate_node(elif_body)
            self.indent -= 1
        
        if node.else_body:
            self.emit("} else {")
            self.indent += 1
            self.generate_node(node.else_body)
            self.indent -= 1
        
        self.emit("}")
        return ""
    
    def gen_For(self, node: For) -> str:
        c_var = self.mangle_name(node.var)
        if isinstance(node.iter, Range):
            start = self.generate_node(node.iter.start)
            end = self.generate_node(node.iter.end)
            op = "<=" if node.iter.inclusive else "<"
            self.emit(f"for (int64_t {c_var} = {start}; {c_var} {op} {end}; {c_var}++) {{")
        elif not isinstance(node.iter, Identifier) and self._is_string_expr(node.iter):
            # for ch in "text" / for ch in s.to_upper() -> byte codes
            tmp = f"_str_{c_var}"
            self.emit(f"const char* {tmp} = {self.generate_node(node.iter)};")
            self.emit(f"for (size_t _i_{c_var} = 0; {tmp}[_i_{c_var}]; _i_{c_var}++) {{")
            self.indent += 1
            self.emit(f"int64_t {c_var} = (unsigned char){tmp}[_i_{c_var}];")
            self.indent -= 1
        elif isinstance(node.iter, Identifier):
            name = node.iter.name
            c_arr = self.mangle_name(name)
            if name in self.dynarray_vars:
                # for x in vec
                elem_c = {"int": "int64_t", "float": "double", "str": "const char*"}.get(
                    self.dynarray_vars[name], "int64_t")
                self.emit(f"for (size_t _i_{c_var} = 0; _i_{c_var} < {c_arr}.len; _i_{c_var}++) {{")
                self.indent += 1
                self.emit(f"{elem_c} {c_var} = {c_arr}.data[_i_{c_var}];")
                self.indent -= 1
                self._track_loop_var(node.var, elem_c)
            elif name in self.string_vars:
                # for ch in "text" -> byte codes (TIL chars are integers, like C)
                self.emit(f"for (size_t _i_{c_var} = 0; {c_arr}[_i_{c_var}]; _i_{c_var}++) {{")
                self.indent += 1
                self.emit(f"int64_t {c_var} = (unsigned char){c_arr}[_i_{c_var}];")
                self.indent -= 1
            else:
                elem_c = self.array_elem_c.get(name, "int64_t")
                len_var = f"{c_arr}_len"
                self.emit(f"for (size_t _i_{c_var} = 0; _i_{c_var} < {len_var}; _i_{c_var}++) {{")
                self.indent += 1
                self.emit(f"{elem_c} {c_var} = {c_arr}[_i_{c_var}];")
                self.indent -= 1
                self._track_loop_var(node.var, elem_c)
        elif isinstance(node.iter, ArrayLit):
            n = len(node.iter.elements)
            elem_type = self.infer_array_elem_type(node.iter)
            arr_name = f"_arr_{c_var}"
            arr_val = self.generate_node(node.iter)
            self.emit(f"{elem_type} {arr_name}[{n}] = {arr_val};")
            self.emit(f"for (size_t _i_{c_var} = 0; _i_{c_var} < {n}; _i_{c_var}++) {{")
            self.indent += 1
            self.emit(f"{elem_type} {c_var} = {arr_name}[_i_{c_var}];")
            self.indent -= 1
            self._track_loop_var(node.var, elem_type)
        elif isinstance(node.iter, Attribute):
            # Iterate over field: for x in obj.items
            iter_expr = self.generate_node(node.iter)
            len_expr = f"{iter_expr}_len"
            self.emit(f"for (size_t _i_{node.var} = 0; _i_{node.var} < {len_expr}; _i_{node.var}++) {{")
            self.indent += 1
            self.emit(f"int64_t {node.var} = {iter_expr}[_i_{node.var}];")
            self.indent -= 1
        elif isinstance(node.iter, Call):
            # Iterate over function result
            iter_expr = self.generate_node(node.iter)
            self.emit(f"// Iteration over call result")
            self.emit(f"for (size_t _i_{node.var} = 0; _i_{node.var} < 0; _i_{node.var}++) {{")
        else:
            # Generic iterable — try to iterate
            iter_expr = self.generate_node(node.iter)
            self.emit(f"for (size_t _i_{node.var} = 0; _i_{node.var} < {iter_expr}_len; _i_{node.var}++) {{")
            self.indent += 1
            self.emit(f"int64_t {node.var} = {iter_expr}[_i_{node.var}];")
            self.indent -= 1
        
        self.indent += 1
        self.declared_vars.add(node.var)
        if self._uses_timeout:
            self.emit("til_check_deadline();")
        self._loop_depth += 1
        self.generate_node(node.body)
        self._loop_depth -= 1
        self.indent -= 1
        self.emit("}")
        return ""
    
    def _track_loop_var(self, name: str, elem_c: str):
        """Register a loop variable's C type so print/concat/== treat it right."""
        for tracked in (self.string_vars, self.float_vars, self.bool_vars):
            tracked.discard(name)
        if elem_c == "const char*":
            self.string_vars.add(name)
        elif elem_c in ("double", "float"):
            self.float_vars.add(name)
        elif elem_c == "bool":
            self.bool_vars.add(name)

    def gen_GlobalInit(self, node: GlobalInit) -> str:
        self.emit(f"{self.mangle_name(node.decl.name)} = {self.generate_node(node.decl.value)};")
        return ""

    def gen_While(self, node: While) -> str:
        cond = self.generate_node(node.condition)
        self.emit(f"while ({cond}) {{")
        self.indent += 1
        if self._uses_timeout:
            self.emit("til_check_deadline();")
        self._loop_depth += 1
        self.generate_node(node.body)
        self._loop_depth -= 1
        self.indent -= 1
        self.emit("}")
        return ""
    
    def gen_Loop(self, node: Loop) -> str:
        self.emit("while (1) {")
        self.indent += 1
        if self._uses_timeout:
            self.emit("til_check_deadline();")
        self._loop_depth += 1
        self.generate_node(node.body)
        self._loop_depth -= 1
        self.indent -= 1
        self.emit("}")
        return ""
    
    def gen_Return(self, node: Return) -> str:
        if self._recover_stack:
            self.emit(f"til_recover_top = {self._recover_stack[0][0]}.prev;")
        if node.value:
            if isinstance(node.value, Identifier) and node.value.name == "NULL":
                suffix = self._optresult_suffix_from_context('None')
                if suffix:
                    self.emit(f"return til_None_{suffix}();")
                    return ""
            val = self.generate_node(node.value)
            # Check if current function has ensures contracts
            if self._current_ensures:
                ret_type = self._current_ensures_ret_type
                self.emit(f"{{ {ret_type} _result = {val};")
                old_in_ensures = self._in_ensures
                self._in_ensures = True
                texts = self._current_ensures_text
                for k, ens_expr in enumerate(self._current_ensures):
                    ens_code = self.generate_node(ens_expr)
                    msg = f"contract violation: {self._current_func_name} ensures " \
                          f"{texts[k] if k < len(texts) else ''}"
                    self.emit(f'  if (!({ens_code})) {{ til_panic({c_string_literal(msg.strip())}); }}')
                self._in_ensures = old_in_ensures
                self.emit(f"  return _result; }}")
            else:
                self.emit(f"return {val};")
        else:
            self.emit("return;")
        return ""
    
    def _leave_guarded_regions_in_loop(self):
        """break/continue out of retry/timeout blocks opened inside the loop."""
        inner = [rec for rec, depth in self._recover_stack if depth == self._loop_depth]
        if inner:
            self.emit(f"til_recover_top = {inner[0]}.prev;")

    def gen_Break(self, node: Break) -> str:
        self._leave_guarded_regions_in_loop()
        self.emit("break;")
        return ""
    
    def gen_Continue(self, node: Continue) -> str:
        self._leave_guarded_regions_in_loop()
        self.emit("continue;")
        return ""
    
    def gen_ExprStmt(self, node: ExprStmt) -> str:
        # Check if it's a print call (needs special handling)
        if isinstance(node.expr, Call) and isinstance(node.expr.func, Identifier):
            if node.expr.func.name in ("print", "println"):
                self.gen_print_call(node.expr)
                return ""
        
        expr = self.generate_node(node.expr)
        if expr:
            self.emit(f"{expr};")
        return ""
    
    def _infer_expr_print_type(self, arg) -> str:
        """Infer the C print type for an expression: 'str', 'float', 'bool', or 'int'."""
        if isinstance(arg, (RetryExpr, TimeoutExpr)):
            return {"const char*": "str", "double": "float", "bool": "bool"}.get(
                self._guarded_value_type(arg), "int")
        if isinstance(arg, StringLit):
            return "str"
        if isinstance(arg, FStringLit):
            return "str"
        if isinstance(arg, IntLit):
            return "int"
        if isinstance(arg, FloatLit):
            return "float"
        if isinstance(arg, BoolLit):
            return "bool"
        if isinstance(arg, Cast):
            c = self.type_to_c(arg.target_type) if arg.target_type else ""
            if c in ("double", "float"):
                return "float"
            if c == "const char*":
                return "str"
            if c == "bool":
                return "bool"
            return "int"
        if isinstance(arg, Identifier):
            if arg.name in self.string_vars:
                return "str"
            if arg.name in self.float_vars:
                return "float"
            if arg.name in self.bool_vars:
                return "bool"
            return "int"
        if isinstance(arg, Attribute):
            # self.x / p.x / a.b.c - resolve the struct type of the object expression
            if True:
                stype = self._resolve_expr_struct_type(arg.obj)
                if stype and stype in self.structs:
                    for fld in self.structs[stype].fields:
                        if fld.name == arg.attr:
                            ct = self.type_to_c(fld.type)
                            if ct in ("double", "float"):
                                return "float"
                            if ct == "const char*":
                                return "str"
                            if ct == "bool":
                                return "bool"
                            return "int"
            return "int"
        if isinstance(arg, Call):
            # Check function return types
            ret_c = self._infer_call_ret_type(arg)
            if ret_c in ("double", "float"):
                return "float"
            if ret_c == "const char*":
                return "str"
            if ret_c == "bool":
                return "bool"
            return "int"
        if isinstance(arg, BinaryOp):
            if arg.op in ("==", "!=", "<", ">", "<=", ">=", "and", "or"):
                return "bool"
            # If either side is float, result is float
            lt = self._infer_expr_print_type(arg.left)
            rt = self._infer_expr_print_type(arg.right)
            if lt == "str" or rt == "str":
                return "str"
            if lt == "float" or rt == "float":
                return "float"
            return "int"
        if isinstance(arg, UnaryOp):
            if arg.op == "not":
                return "bool"
            return self._infer_expr_print_type(arg.operand)
        return "int"

    def _infer_call_ret_type(self, node: Call) -> str:
        """Infer C return type of a function/method call."""
        if isinstance(node.func, Identifier):
            name = node.func.name
            # Built-in math functions return double
            if name in ('sqrt', 'abs', 'pow', 'sin', 'cos', 'tan', 'log', 'exp', 'floor', 'ceil', 'round',
                        'min', 'max'):
                return "double"
            if name in ('len',):
                return "int64_t"
            # Check lambda return types
            if name in self._lambda_var_map:
                lambda_name = self._lambda_var_map[name]
                if lambda_name in self._lambda_ret_types:
                    return self._lambda_ret_types[lambda_name]
            key = name
            if key in self._func_ret_types:
                return self._func_ret_types[key]
        if isinstance(node.func, Attribute):
            method_name = node.func.attr

            # DynArray/HashMap method return types
            if isinstance(node.func.obj, Identifier):
                obj_name = node.func.obj.name
                if obj_name in self.dynarray_vars:
                    elem = self.dynarray_vars[obj_name]
                    type_map = {"int": "int64_t", "float": "double", "str": "const char*"}
                    if method_name in ("len",):
                        return "int64_t"
                    if method_name in ("get", "pop"):
                        return type_map.get(elem, "int64_t")
                if obj_name in self.hashmap_vars:
                    key_t, val_t = self.hashmap_vars[obj_name]
                    type_map = {"int": "int64_t", "float": "double", "str": "const char*"}
                    if method_name == "get":
                        return type_map.get(val_t, "int64_t")
                    if method_name == "has":
                        return "bool"
                    if method_name == "len":
                        return "int64_t"

            # Option / Result method return types
            opt_ct = self._optresult_ctype(node.func.obj)
            if opt_ct:
                if method_name in ("is_some", "is_none", "is_ok", "is_err"):
                    return "bool"
                if method_name in ("unwrap", "expect", "unwrap_or"):
                    return {"int": "int64_t", "float": "double", "str": "const char*",
                            "bool": "bool"}.get(opt_ct.split("_")[-1], "int64_t")
                if method_name == "error":
                    return "const char*"

            # String method return types
            if self._is_string_expr(node.func.obj):
                str_bool_methods = {'contains', 'starts_with', 'ends_with', 'eq'}
                str_str_methods = {'trim', 'to_upper', 'to_lower', 'replace', 'slice'}
                str_int_methods = {'len', 'find'}
                if method_name in str_bool_methods:
                    return "bool"
                if method_name in str_str_methods:
                    return "const char*"
                if method_name in str_int_methods:
                    return "int64_t"

            if (isinstance(node.func.obj, Identifier) and node.func.obj.name in self.module_names
                    and method_name in self._func_ret_types):
                return self._func_ret_types[method_name]

            # obj.method() — resolve struct type of obj, then look up method return type
            stype = self._resolve_expr_struct_type(node.func.obj)
            if stype:
                key = f"{stype}_{method_name}"
                if key in self._func_ret_types:
                    return self._func_ret_types[key]
        return "int64_t"

    def gen_print_call(self, node: Call):
        for arg in node.args:
            val = self.generate_node(arg)
            ptype = self._infer_expr_print_type(arg)

            if ptype == "str":
                self.emit(f'til_print_str({val});')
            elif ptype == "float":
                self.emit(f'til_print_float({val});')
            elif ptype == "bool":
                self.emit(f'til_print_bool({val});')
            else:
                self.emit(f'til_print_int((int64_t){val});')
    
    def gen_Call(self, node: Call) -> str:
        if isinstance(node.func, Identifier):
            name = node.func.name

            # Handle print/println as special case (they're not regular C functions)
            if name in ("print", "println"):
                self.gen_print_call(node)
                return ""

            args = [self.generate_node(a) for a in node.args]
            args_str = ", ".join(args)

            # Check if this is a lambda variable with captures
            if name in self._lambda_var_map:
                lambda_name = self._lambda_var_map[name]
                captures = self._lambda_captures.get(lambda_name, [])
                if captures:
                    cap_args = [self.mangle_name(c) for c in captures]
                    all_args = args + cap_args
                    return f"{lambda_name}({', '.join(all_args)})"
                return f"{lambda_name}({args_str})"

            # Built-in functions
            builtins = {
                'sqrt': 'til_sqrt', 'abs': 'til_abs', 'pow': 'til_pow',
                'sin': 'til_sin', 'cos': 'til_cos', 'tan': 'til_tan',
                'log': 'til_log', 'exp': 'til_exp',
                'floor': 'til_floor', 'ceil': 'til_ceil', 'round': 'til_round',
                'len': 'til_len_str',
            }

            if name in builtins:
                return f"{builtins[name]}({args_str})"

            if name == 'panic':
                return f"til_panic({self._to_c_string(node.args[0], args[0]) if node.args else chr(34) + 'panic' + chr(34)})"
            if name == 'exit':
                return f"exit((int)({args[0] if args else '0'}))"
            if name == 'min':
                return f"til_min_int({args_str})"
            if name == 'max':
                return f"til_max_int({args_str})"

            # Option/Result constructors (typed by the enclosing function's
            # return type when it is an Option/Result, else by the argument)
            if name in ('Some', 'Ok', 'Err'):
                suffix = self._optresult_suffix_from_context(name)
                if suffix is None:
                    if name == 'Err':
                        suffix = "int"
                    else:
                        inner_type = self._infer_expr_print_type(node.args[0]) if node.args else "int"
                        suffix = inner_type if inner_type in ("int", "float", "str", "bool") else "int"
                if name == 'Err' and suffix == "bool":
                    suffix = "int"
                return f"til_{name}_{suffix}({args_str})"

            # Struct constructor
            if name in self.structs:
                c_sname = self.mangle_name(name)
                return f"{c_sname}_create({args_str})"

            # Fill in default parameters if fewer args provided
            if name in self.functions:
                func_def = self.functions[name]
                if len(node.args) < len(func_def.params):
                    for i in range(len(node.args), len(func_def.params)):
                        p = func_def.params[i]
                        if p.default:
                            args.append(self.generate_node(p.default))
                    args_str = ", ".join(args)

            c_fname = self.mangle_name(name)
            return f"til_{c_fname}({args_str})"
        
        if isinstance(node.func, Attribute):
            obj = self.generate_node(node.func.obj)
            method_name = node.func.attr
            args = [self.generate_node(a) for a in node.args]

            # Vec<T>.new() and HashMap<K,V>.new() constructors
            if isinstance(node.func.obj, Identifier):
                obj_name = node.func.obj.name
                if obj_name.startswith("__Vec__"):
                    elem_type = obj_name[7:]
                    if method_name == "new":
                        return f"til_dynarray_{elem_type}_new()"
                if obj_name.startswith("__HashMap__"):
                    type_parts = obj_name[11:]  # "str_int" or "str_str"
                    if method_name == "new":
                        return f"til_hashmap_{type_parts}_new()"

                # Dynamic array method calls
                if obj_name in self.dynarray_vars:
                    elem_type = self.dynarray_vars[obj_name]
                    c_obj = self.mangle_name(obj_name)
                    if method_name == "push":
                        self.emit(f"til_dynarray_{elem_type}_push(&{c_obj}, {args[0]});")
                        return ""
                    elif method_name == "pop":
                        return f"til_dynarray_{elem_type}_pop(&{c_obj})"
                    elif method_name == "get":
                        return f"til_dynarray_{elem_type}_get(&{c_obj}, {args[0]})"
                    elif method_name == "len":
                        return f"(int64_t){c_obj}.len"

                # HashMap method calls
                if obj_name in self.hashmap_vars:
                    key_type, val_type = self.hashmap_vars[obj_name]
                    c_obj = self.mangle_name(obj_name)
                    hm_prefix = f"til_hashmap_{key_type}_{val_type}"
                    if method_name == "set":
                        self.emit(f"{hm_prefix}_set(&{c_obj}, {args[0]}, {args[1]});")
                        return ""
                    elif method_name == "get":
                        return f"{hm_prefix}_get(&{c_obj}, {args[0]})"
                    elif method_name == "has":
                        return f"{hm_prefix}_has(&{c_obj}, {args[0]})"
                    elif method_name == "delete":
                        self.emit(f"{hm_prefix}_delete(&{c_obj}, {args[0]});")
                        return ""
                    elif method_name == "len":
                        return f"(int64_t){c_obj}.len"

            # Option / Result methods
            opt_ct = self._optresult_ctype(node.func.obj)
            if opt_ct:
                is_opt = opt_ct.startswith("TIL_Option_")
                suffix = opt_ct.split("_")[-1]
                kind = "opt" if is_opt else "res"
                if method_name in ("is_some", "is_ok"):
                    return f"({obj}).{'has_value' if is_opt else 'is_ok'}"
                if method_name in ("is_none", "is_err"):
                    return f"(!({obj}).{'has_value' if is_opt else 'is_ok'})"
                if method_name in ("unwrap", "expect", "unwrap_or"):
                    return f"til_{kind}_{method_name}_{suffix}({', '.join([obj] + args)})"
                if method_name == "error" and not is_opt:
                    return f"({obj}).error"

            # String method calls
            if self._is_string_expr(node.func.obj):
                str_methods = {
                    'len': ('til_len_str', [obj]),
                    'contains': ('til_str_contains', [obj] + args),
                    'starts_with': ('til_str_starts_with', [obj] + args),
                    'ends_with': ('til_str_ends_with', [obj] + args),
                    'trim': ('til_str_trim', [obj]),
                    'to_upper': ('til_str_to_upper', [obj]),
                    'to_lower': ('til_str_to_lower', [obj]),
                    'replace': ('til_str_replace', [obj] + args),
                    'slice': ('til_str_slice', [obj] + args),
                    'find': ('til_str_find', [obj] + args),
                    'eq': ('til_str_eq', [obj] + args),
                }
                if method_name in str_methods:
                    func_name, func_args = str_methods[method_name]
                    return f"{func_name}({', '.join(func_args)})"

            # Resolve the struct type of the object
            resolved_type = self._resolve_expr_struct_type(node.func.obj)

            if resolved_type:
                if resolved_type in self.enums:
                    # Enum member access: Color.Red -> Color_Red
                    return f"{self.mangle_name(resolved_type)}_{self.mangle_name(method_name)}"
                # Check if method has self parameter (instance method vs static)
                is_instance = False
                if resolved_type in self.impl_methods:
                    for m in self.impl_methods[resolved_type]:
                        if m.name == method_name:
                            is_instance = any(p.name == "self" for p in m.params)
                            break
                if is_instance:
                    # If obj is already a pointer (self in method), don't add &
                    if isinstance(node.func.obj, Identifier) and node.func.obj.name == "self" and self.in_method:
                        self_arg = "self"  # already a pointer
                    else:
                        self_arg = f"&{obj}"
                    args_str = ", ".join([self_arg] + args)
                else:
                    args_str = ", ".join(args)
                return f"{self.mangle_name(resolved_type)}_{self.mangle_name(method_name)}({args_str})"

            # Module-qualified call: `math_utils.square(5)` (imports are merged,
            # so this is a call of the imported top-level function)
            if (isinstance(node.func.obj, Identifier) and node.func.obj.name in self.module_names
                    and method_name in self.functions):
                func_def = self.functions[method_name]
                for i in range(len(node.args), len(func_def.params)):
                    if func_def.params[i].default:
                        args.append(self.generate_node(func_def.params[i].default))
                return f"til_{self.mangle_name(method_name)}({', '.join(args)})"

            args_str = ", ".join(args)
            return f"{obj}.{method_name}({args_str})"
        
        func = self.generate_node(node.func)
        args = [self.generate_node(a) for a in node.args]
        return f"{func}({', '.join(args)})"
    
    def _optresult_suffix_from_context(self, ctor: str) -> Optional[str]:
        """Suffix (int/float/str/bool) for Some/None/Ok/Err from the current
        function's declared Option/Result return type."""
        rt = self._current_ret_type
        if ctor in ('Some', 'None') and isinstance(rt, OptionType):
            return self.type_to_c(rt)[len("TIL_Option_"):]
        if ctor in ('Ok', 'Err') and isinstance(rt, ResultType):
            return self.type_to_c(rt)[len("TIL_Result_"):]
        return None

    def _optresult_ctype(self, node) -> Optional[str]:
        """C type (TIL_Option_x / TIL_Result_x) of an Option/Result expression."""
        if isinstance(node, Identifier) and node.name in self.optresult_vars:
            return self.optresult_vars[node.name]
        if isinstance(node, Call):
            ct = self._infer_call_ret_type(node)
            if ct.startswith("TIL_Option_") or ct.startswith("TIL_Result_"):
                return ct
        return None

    def gen_BinaryOp(self, node: BinaryOp) -> str:
        left = self.generate_node(node.left)
        right = self.generate_node(node.right)
        
        op_map = {
            'and': '&&', 'or': '||',
            '==': '==', '!=': '!=',
            '<': '<', '>': '>', '<=': '<=', '>=': '>=',
            '+': '+', '-': '-', '*': '*', '/': '/', '%': '%',
            '&': '&', '|': '|', '^': '^',
            '<<': '<<', '>>': '>>',
        }
        
        if node.op == '**':
            if (self._infer_expr_print_type(node.left) == "int" and
                    self._infer_expr_print_type(node.right) == "int"):
                return f"til_ipow({left}, {right})"     # exact integer power
            return f"pow({left}, {right})"
        
        if node.op == '+':
            # String concat - works with literals AND variables; a non-string
            # operand is converted ("Score: " + 42 == "Score: 42")
            if self._is_string_expr(node.left) or self._is_string_expr(node.right):
                return f'til_str_concat({self._to_c_string(node.left, left)}, {self._to_c_string(node.right, right)})'

        # String equality compares contents, not pointers
        if node.op in ('==', '!=') and (self._is_string_expr(node.left) or self._is_string_expr(node.right)):
            eq = f'til_str_eq({left}, {right})'
            return eq if node.op == '==' else f'(!{eq})'
        
        c_op = op_map.get(node.op, node.op)
        return f"({left} {c_op} {right})"

    def _to_c_string(self, node, code: str) -> str:
        """C expression converting an already-generated value to a C string."""
        ptype = self._infer_expr_print_type(node)
        if ptype == "str":
            return code
        if ptype == "float":
            return f"til_float_to_str({code})"
        if ptype == "bool":
            return f"til_bool_to_str({code})"
        return f"til_int_to_str((int64_t)({code}))"
    
    def _resolve_expr_struct_type(self, node) -> Optional[str]:
        """Resolve the struct type name of an expression, or None."""
        if isinstance(node, Identifier):
            name = node.name
            if name == "self" and self.current_struct:
                return self.current_struct
            if name in self.struct_vars:
                return self.struct_vars[name]
            if name in self.structs:
                return name  # static call: Point.new(...)
            if name in self.enums:
                return name  # enum access
            return None
        if isinstance(node, Attribute):
            # Chained access: e.g., self.center -> look up field type
            parent_type = self._resolve_expr_struct_type(node.obj)
            if parent_type and parent_type in self.structs:
                for fld in self.structs[parent_type].fields:
                    if fld.name == node.attr:
                        if isinstance(fld.type, StructType):
                            return fld.type.name
                        return self._struct_of_ctype(self.type_to_c(fld.type))
            return None
        if isinstance(node, Call):
            # Method call result: e.g., v1.add(v2) -> Vector2D
            return self._struct_of_ctype(self._infer_call_ret_type(node))
        return None

    def _struct_of_ctype(self, c_type: str) -> Optional[str]:
        """TIL struct name for a C type name (struct names are mangled in C)."""
        if c_type in self.structs:
            return c_type
        for name in self.structs:
            if self.mangle_name(name) == c_type:
                return name
        return None

    def _is_string_expr(self, node) -> bool:
        """Check if an expression produces a string value."""
        if isinstance(node, (StringLit, FStringLit)):
            return True
        if isinstance(node, Cast):
            return self.type_to_c(node.target_type) == "const char*"
        if isinstance(node, IfExpr):
            return self._is_string_expr(node.then_expr)
        if isinstance(node, Identifier):
            return node.name in self.string_vars
        if isinstance(node, BinaryOp) and node.op == '+':
            return self._is_string_expr(node.left) or self._is_string_expr(node.right)
        if isinstance(node, Call):
            return self._infer_call_ret_type(node) == "const char*"
        if isinstance(node, Attribute):
            stype = self._resolve_expr_struct_type(node.obj)
            if stype and stype in self.structs:
                for fld in self.structs[stype].fields:
                    if fld.name == node.attr:
                        return self.type_to_c(fld.type) == "const char*"
        return False

    def gen_UnaryOp(self, node: UnaryOp) -> str:
        operand = self.generate_node(node.operand)
        
        if node.op == 'not':
            return f"(!{operand})"
        if node.op == '-':
            return f"(-{operand})"
        if node.op == '~':
            return f"(~{operand})"
        if node.op == '&':
            return f"(&{operand})"
        if node.op == '&mut':
            return f"(&{operand})"
        if node.op == '*':
            return f"(*{operand})"
        
        return f"({node.op}{operand})"
    
    def gen_Index(self, node: Index) -> str:
        obj = self.generate_node(node.obj)
        index = self.generate_node(node.index)

        # Add bounds checking for Level 2+ (Safe level and above)
        if self.current_level >= 2:
            if isinstance(node.obj, Identifier):
                arr_name = node.obj.name
                if arr_name in self.array_vars:
                    self.emit(f'til_bounds_check({index}, {arr_name}_len, "{arr_name}");')
                elif arr_name in self.string_vars:
                    self.emit(f'til_bounds_check({index}, til_len_str({arr_name}), "{arr_name}");')

        return f"{obj}[{index}]"
    
    def gen_Attribute(self, node: Attribute) -> str:
        obj = self.generate_node(node.obj)
        # Use -> for pointer access (self in methods)
        if isinstance(node.obj, Identifier) and node.obj.name == "self" and self.in_method:
            return f"self->{node.attr}"
        # Check if it's an enum member access: Color.Red -> Color_Red
        if isinstance(node.obj, Identifier) and node.obj.name in self.enums:
            return f"{self.mangle_name(node.obj.name)}_{self.mangle_name(node.attr)}"
        return f"{obj}.{node.attr}"
    
    def gen_ListComprehension(self, node: ListComprehension) -> str:
        """Generate C code for [expr for var in iter if condition].
        Returns the array variable name; also emits _len variable."""
        if not hasattr(self, '_comp_counter'):
            self._comp_counter = 0
        self._comp_counter += 1
        arr = f"_comp_{self._comp_counter}"

        if isinstance(node.iter, Range):
            start = self.generate_node(node.iter.start)
            end = self.generate_node(node.iter.end)
            op = "<=" if node.iter.inclusive else "<"
            # Calculate max size
            size_expr = f"({end} - {start}" + (" + 1" if node.iter.inclusive else "") + ")"
            self.emit(f"size_t {arr}_cap = {size_expr};")
            self.emit(f"int64_t* {arr} = (int64_t*)malloc(sizeof(int64_t) * {arr}_cap);")
            self.emit(f"size_t {arr}_len = 0;")
            self.emit(f"for (int64_t {node.var} = {start}; {node.var} {op} {end}; {node.var}++) {{")
            self.indent += 1
            if node.condition:
                cond = self.generate_node(node.condition)
                self.emit(f"if ({cond}) {{")
                self.indent += 1
            expr_val = self.generate_node(node.expr)
            self.emit(f"{arr}[{arr}_len++] = {expr_val};")
            if node.condition:
                self.indent -= 1
                self.emit("}")
            self.indent -= 1
            self.emit("}")
        else:
            # Array iteration
            iter_expr = self.generate_node(node.iter)
            len_name = f"{iter_expr}_len" if isinstance(node.iter, Identifier) else "0"
            self.emit(f"size_t {arr}_cap = {len_name};")
            self.emit(f"int64_t* {arr} = (int64_t*)malloc(sizeof(int64_t) * ({arr}_cap > 0 ? {arr}_cap : 16));")
            self.emit(f"size_t {arr}_len = 0;")
            self.emit(f"for (size_t _ci = 0; _ci < {len_name}; _ci++) {{")
            self.indent += 1
            self.emit(f"int64_t {node.var} = {iter_expr}[_ci];")
            if node.condition:
                cond = self.generate_node(node.condition)
                self.emit(f"if ({cond}) {{")
                self.indent += 1
            expr_val = self.generate_node(node.expr)
            self.emit(f"{arr}[{arr}_len++] = {expr_val};")
            if node.condition:
                self.indent -= 1
                self.emit("}")
            self.indent -= 1
            self.emit("}")

        # Track as array var
        self.array_vars[arr] = 0  # dynamic size

        return arr

    def gen_Lambda(self, node: Lambda) -> str:
        """Generate a lambda with closure support (captured outer variables)."""
        if not hasattr(self, '_lambda_counter'):
            self._lambda_counter = 0
        self._lambda_counter += 1
        name = f"_til_lambda_{self._lambda_counter}"

        if not hasattr(self, '_lambda_defs'):
            self._lambda_defs = []

        # Collect parameter names
        param_names = {pname for pname, _ in node.params}

        # Save and set up tracking for lambda scope
        old_string_vars = set(self.string_vars)
        old_float_vars = set(self.float_vars)
        old_bool_vars = set(self.bool_vars)
        old_declared = set(self.declared_vars)

        for pname, ptype in node.params:
            self.declared_vars.add(pname)
            if ptype:
                c_type = self.type_to_c(ptype)
                if c_type == "const char*":
                    self.string_vars.add(pname)
                elif c_type in ("double", "float"):
                    self.float_vars.add(pname)
                elif c_type == "bool":
                    self.bool_vars.add(pname)

        # Detect captured outer variables
        captures = []
        self._collect_captures(node.body, param_names, captures)

        # Build C parameter list (params + captures)
        params = []
        for pname, ptype in node.params:
            c_type = self.type_to_c(ptype) if ptype else "int64_t"
            params.append(f"{c_type} {pname}")

        for cap_name in captures:
            c_type = self._infer_var_c_type(cap_name)
            params.append(f"{c_type} {cap_name}")

        params_str = ", ".join(params) if params else "void"

        # Determine return type
        if node.ret_type:
            ret_type = self.type_to_c(node.ret_type)
        else:
            ret_type = "int64_t"

        # Generate body expression
        body_code = self.generate_node(node.body)

        # Restore tracking state
        self.string_vars = old_string_vars
        self.float_vars = old_float_vars
        self.bool_vars = old_bool_vars
        self.declared_vars = old_declared

        self._lambda_defs.append(f"static {ret_type} {name}({params_str}) {{ return {body_code}; }}")
        self._lambda_captures[name] = captures
        self._lambda_ret_types[name] = ret_type

        return name

    def _collect_captures(self, node, param_names, captures):
        """Walk AST to find identifiers referencing outer scope variables (captures)."""
        if node is None:
            return
        if isinstance(node, Identifier):
            if (node.name not in param_names and node.name in self.declared_vars
                    and node.name not in captures
                    and node.name not in self.functions
                    and node.name != "self"):
                captures.append(node.name)
        elif isinstance(node, BinaryOp):
            self._collect_captures(node.left, param_names, captures)
            self._collect_captures(node.right, param_names, captures)
        elif isinstance(node, UnaryOp):
            self._collect_captures(node.operand, param_names, captures)
        elif isinstance(node, Call):
            if isinstance(node.func, Attribute):
                self._collect_captures(node.func.obj, param_names, captures)
            elif isinstance(node.func, Identifier):
                pass  # Don't capture function names
            else:
                self._collect_captures(node.func, param_names, captures)
            for arg in node.args:
                self._collect_captures(arg, param_names, captures)
        elif isinstance(node, Attribute):
            self._collect_captures(node.obj, param_names, captures)
        elif isinstance(node, Block):
            for stmt in node.statements:
                self._collect_captures(stmt, param_names, captures)
        elif isinstance(node, ExprStmt):
            self._collect_captures(node.expr, param_names, captures)
        elif isinstance(node, Return):
            if node.value:
                self._collect_captures(node.value, param_names, captures)
        elif isinstance(node, IfExpr):
            self._collect_captures(node.condition, param_names, captures)
            self._collect_captures(node.then_expr, param_names, captures)
            if node.else_expr:
                self._collect_captures(node.else_expr, param_names, captures)
        elif isinstance(node, If):
            self._collect_captures(node.condition, param_names, captures)
            self._collect_captures(node.then_body, param_names, captures)
            if node.else_body:
                self._collect_captures(node.else_body, param_names, captures)
        elif isinstance(node, Index):
            self._collect_captures(node.obj, param_names, captures)
            self._collect_captures(node.index, param_names, captures)

    def _infer_var_c_type(self, name: str) -> str:
        """Infer the C type of a variable from tracking info."""
        if name in self.string_vars:
            return "const char*"
        if name in self.float_vars:
            return "double"
        if name in self.bool_vars:
            return "bool"
        return "int64_t"

    def gen_IfExpr(self, node: IfExpr) -> str:
        cond = self.generate_node(node.condition)
        then_val = self.generate_node(node.then_expr)
        else_val = self.generate_node(node.else_expr) if node.else_expr else "0"
        return f"({cond} ? {then_val} : {else_val})"

    def gen_Cast(self, node: Cast) -> str:
        expr = self.generate_node(node.expr)
        c_type = self.type_to_c(node.target_type)
        return f"(({c_type}){expr})"

    def gen_Identifier(self, node: Identifier) -> str:
        if node.name == "self":
            return "self"
        if node.name == "result" and self._in_ensures:
            return "_result"
        return self.mangle_name(node.name)
    
    def gen_IntLit(self, node: IntLit) -> str:
        return str(node.value)
    
    def gen_FloatLit(self, node: FloatLit) -> str:
        return str(node.value)
    
    def gen_StringLit(self, node: StringLit) -> str:
        return c_string_literal(node.value)
    
    def gen_BoolLit(self, node: BoolLit) -> str:
        return "true" if node.value else "false"

    def gen_CharLit(self, node: CharLit) -> str:
        ch = node.value
        if ch == "'":
            return "'\\''"
        elif ch == '\\':
            return "'\\\\'"
        elif ch == '\n':
            return "'\\n'"
        elif ch == '\t':
            return "'\\t'"
        return f"'{ch}'"
    
    def gen_TupleLit(self, node: TupleLit) -> str:
        """Generate tuple as a C struct literal."""
        if not hasattr(self, '_tuple_types'):
            self._tuple_types = {}
        n = len(node.elements)
        # Get element types
        types = []
        vals = []
        for elem in node.elements:
            t = self._infer_expr_print_type(elem)
            type_map = {"int": "int64_t", "float": "double", "str": "const char*", "bool": "bool"}
            types.append(type_map.get(t, "int64_t"))
            vals.append(self.generate_node(elem))

        # Create a unique tuple type name
        key = "_".join(types)
        if key not in self._tuple_types:
            tuple_name = f"TIL_Tuple_{len(self._tuple_types)}"
            self._tuple_types[key] = tuple_name
            # Will be emitted in _fixup_tuples

        tuple_name = self._tuple_types[key]
        fields = ", ".join(vals)
        return f"({tuple_name}){{{fields}}}"

    def gen_FStringLit(self, node: FStringLit) -> str:
        """Generate an f-string as til_str_concat calls; supports {x:.2f}-style specs."""
        if not node.parts:
            return '""'

        result = None
        for part in node.parts:
            kind, value = part[0], part[1]
            spec = part[2] if len(part) > 2 else None
            if kind == 'str':
                val = self.gen_StringLit(value)
            else:
                ptype = self._infer_expr_print_type(value)
                expr_val = self.generate_node(value)
                if spec:
                    cspec = c_format_spec(spec, ptype)
                    if cspec[1] == "percent":
                        val = f"til_fmt_float((double)({expr_val}) * 100.0, {c_string_literal(cspec[0])})"
                    elif cspec[1] == "float":
                        val = f"til_fmt_float((double)({expr_val}), {c_string_literal(cspec[0])})"
                    else:
                        val = f"til_fmt_int((int64_t)({expr_val}), {c_string_literal(cspec[0])})"
                else:
                    val = self._to_c_string(value, expr_val)
            result = val if result is None else f"til_str_concat({result}, {val})"

        return result

    def gen_ArrayLit(self, node: ArrayLit) -> str:
        elements = [self.generate_node(e) for e in node.elements]
        return "{" + ", ".join(elements) + "}"
    
    def gen_NullCheck(self, node: NullCheck) -> str:
        """`expr?` - real error propagation: if expr is None/Err, return it from the
        enclosing function (when that function returns Option/Result), otherwise
        panic. Evaluates to the unwrapped value."""
        ct = self._optresult_ctype(node.expr)
        if ct is None:
            expr = self.generate_node(node.expr)
            return f"({expr})"
        self._q_counter = getattr(self, '_q_counter', 0) + 1
        q = f"_q_{self._q_counter}"
        self.emit(f"{ct} {q} = {self.generate_node(node.expr)};")
        rt = self._current_ret_type
        if ct.startswith("TIL_Option_"):
            if isinstance(rt, OptionType):
                self.emit(f"if (!{q}.has_value) return til_None_{self.type_to_c(rt)[len('TIL_Option_'):]}();")
            else:
                self.emit(f'if (!{q}.has_value) til_panic("? applied to None");')
        else:
            if isinstance(rt, ResultType):
                rsuffix = self.type_to_c(rt)[len('TIL_Result_'):]
                self.emit(f"if (!{q}.is_ok) return til_Err_{'int' if rsuffix == 'bool' else rsuffix}({q}.error);")
            else:
                self.emit(f'if (!{q}.is_ok) til_panic({q}.error);')
        return f"{q}.value"

    def gen_DictLit(self, node: DictLit) -> str:
        # Basic dict literal - generate as comment for now (needs hash table impl)
        self.emit("// TODO: Dict literal requires hash table implementation")
        return "NULL"

    def gen_Range(self, node: Range) -> str:
        # Ranges are handled in for loops
        start = self.generate_node(node.start)
        end = self.generate_node(node.end)
        return f"/* range {start}..{end} */"
    
    def gen_StructInit(self, node: StructInit) -> str:
        args = []
        if node.name in self.structs:
            for fld in self.structs[node.name].fields:
                if fld.name in node.fields:
                    args.append(self.generate_node(node.fields[fld.name]))
                elif fld.default:
                    args.append(self.generate_node(fld.default))
                else:
                    args.append("0")
        else:
            args = [self.generate_node(v) for v in node.fields.values()]

        return f"{self.mangle_name(node.name)}_create({', '.join(args)})"
    
    def gen_MatchExpr(self, node: MatchExpr) -> str:
        val = self.generate_node(node.value)

        # Normalize arms to 3-tuples (pattern, guard, body)
        arms = []
        for arm in node.arms:
            if len(arm) == 2:
                arms.append((arm[0], None, arm[1]))
            else:
                arms.append(arm)

        # Determine if this match produces a value (expression context)
        # Heuristic: if any arm body is a non-Block expression, it's a value match
        is_expr = any(not isinstance(body, Block) for _, _, body in arms)

        # Infer result type from first arm body
        result_c_type = "int64_t"
        if is_expr:
            first_body = arms[0][2] if arms else None
            if first_body:
                if isinstance(first_body, StringLit):
                    result_c_type = "const char*"
                elif isinstance(first_body, FloatLit):
                    result_c_type = "double"
                elif isinstance(first_body, BoolLit):
                    result_c_type = "bool"

        # Check if we need if-else chain (guards, wildcards, or variable bindings)
        has_guards = any(guard is not None for _, guard, _ in arms)
        has_wildcard = any(isinstance(p, Identifier) and p.name == "_" for p, _, _ in arms)
        has_var_bind = any(isinstance(p, Identifier) and p.name != "_"
                          and not isinstance(p, (IntLit, FloatLit, BoolLit, StringLit))
                          for p, _, _ in arms)

        subject_c = {"str": "const char*", "float": "double", "bool": "bool"}.get(
            self._infer_expr_print_type(node.value), "int64_t")
        has_str_pat = any(isinstance(p, StringLit) for p, _, _ in arms)
        if subject_c == "const char*" or has_str_pat:
            subject_c = "const char*"   # strings can't be switch()ed: compare contents
        if has_guards or has_wildcard or has_var_bind or subject_c != "int64_t":
            return self._gen_match_complex(arms, val, is_expr, result_c_type, subject_c)
        else:
            return self._gen_match_switch(arms, val, is_expr, result_c_type)

    def _gen_match_arm_body(self, body, result_var=None):
        """Generate the body of a match arm. If result_var set, assign expression to it."""
        if isinstance(body, Block):
            self.generate_node(body)
        elif result_var:
            result = self.generate_node(body)
            if result:
                self.emit(f"{result_var} = {result};")
        else:
            # Could be a Call to print — generate_node handles it
            result = self.generate_node(body)
            if result:
                self.emit(f"{result};")

    def _gen_match_complex(self, arms, val, is_expr=False, result_c_type="int64_t",
                           subject_c="int64_t"):
        """Generate match as cascading if-else with guards/variable bindings."""
        self._match_counter = getattr(self, '_match_counter', 0) + 1
        result_var = f"_match_result_{self._match_counter}" if is_expr else None

        # Declare result var BEFORE the block so it's visible after
        if result_var:
            if result_c_type == "const char*":
                self.emit(f'{result_c_type} {result_var} = "";')
            else:
                self.emit(f"{result_c_type} {result_var} = 0;")

        self.emit("{")
        self.indent += 1
        self.emit(f"{subject_c} _match_val = {val};")
        self.emit("bool _matched = false;")

        for pat, guard, body in arms:
            is_wildcard = isinstance(pat, Identifier) and pat.name == "_"
            is_var_bind = (isinstance(pat, Identifier) and pat.name != "_"
                          and not isinstance(pat, (IntLit, FloatLit, BoolLit, StringLit)))

            if is_wildcard:
                self.emit("if (!_matched) {")
                self.indent += 1
                self._gen_match_arm_body(body, result_var)
                self.indent -= 1
                self.emit("}")
            elif is_var_bind and guard:
                var_name = self.mangle_name(pat.name)
                self.declared_vars.add(pat.name)
                self._track_loop_var(pat.name, subject_c)
                self.emit("if (!_matched) {")
                self.indent += 1
                self.emit(f"{subject_c} {var_name} = _match_val;")
                guard_code = self.generate_node(guard)
                self.emit(f"if ({guard_code}) {{")
                self.indent += 1
                self._gen_match_arm_body(body, result_var)
                self.emit("_matched = true;")
                self.indent -= 1
                self.emit("}")
                self.indent -= 1
                self.emit("}")
            else:
                # Literal pattern (possibly with guard)
                pat_val = self.generate_node(pat)
                cond = (f"til_str_eq(_match_val, {pat_val})" if subject_c == "const char*"
                        else f"_match_val == {pat_val}")
                if guard:
                    guard_code = self.generate_node(guard)
                    cond = f"({cond}) && ({guard_code})"
                self.emit(f"if (!_matched && ({cond})) {{")
                self.indent += 1
                self._gen_match_arm_body(body, result_var)
                self.emit("_matched = true;")
                self.indent -= 1
                self.emit("}")

        self.indent -= 1
        self.emit("}")
        if result_var:
            return result_var
        return ""

    def _gen_match_switch(self, arms, val, is_expr=False, result_c_type="int64_t"):
        """Generate match as switch statement."""
        self._match_counter = getattr(self, '_match_counter', 0) + 1
        result_var = f"_match_result_{self._match_counter}" if is_expr else None

        if result_var:
            if result_c_type == "const char*":
                self.emit(f'{result_c_type} {result_var} = "";')
            else:
                self.emit(f"{result_c_type} {result_var} = 0;")

        self.emit(f"switch ({val}) {{")
        self.indent += 1

        for pat, guard, body in arms:
            if isinstance(pat, Identifier) and pat.name == "_":
                self.emit("default:")
            else:
                pat_val = self.generate_node(pat)
                self.emit(f"case {pat_val}:")

            self.indent += 1
            self._gen_match_arm_body(body, result_var)
            self.emit("break;")
            self.indent -= 1

        self.indent -= 1
        self.emit("}")
        if result_var:
            return result_var
        return ""
    
    # Helper methods
    def type_to_c(self, t: Type) -> str:
        if isinstance(t, PrimitiveType):
            type_map = {
                'int': 'int64_t', 'i8': 'int8_t', 'i16': 'int16_t',
                'i32': 'int32_t', 'i64': 'int64_t',
                'uint': 'uint64_t', 'u8': 'uint8_t', 'u16': 'uint16_t',
                'u32': 'uint32_t', 'u64': 'uint64_t',
                'float': 'double', 'f32': 'float', 'f64': 'double',
                'bool': 'bool', 'str': 'const char*', 'char': 'char',
                'void': 'void',
            }
            return type_map.get(t.name, 'int64_t')
        
        if isinstance(t, VoidType):
            return 'void'
        
        if isinstance(t, ArrayType):
            elem = self.type_to_c(t.element_type)
            if t.size:
                return f"{elem}*"
            return f"{elem}*"
        
        if isinstance(t, PointerType):
            inner = self.type_to_c(t.pointee)
            return f"{inner}*"
        
        if isinstance(t, StructType):
            return self.mangle_name(t.name)

        if isinstance(t, OptionType):
            inner_c = self.type_to_c(t.inner)
            type_map = {'int64_t': 'int', 'double': 'float', 'const char*': 'str', 'bool': 'bool'}
            suffix = type_map.get(inner_c, 'int')
            return f'TIL_Option_{suffix}'

        if isinstance(t, ResultType):
            ok_c = self.type_to_c(t.ok)
            type_map = {'int64_t': 'int', 'double': 'float', 'const char*': 'str', 'bool': 'bool'}
            suffix = type_map.get(ok_c, 'int')
            return f'TIL_Result_{suffix}'

        if isinstance(t, UnknownType):
            return 'int64_t'

        return 'int64_t'
    
    def params_to_c(self, params: List[FuncParam]) -> str:
        if not params:
            return 'void'

        parts = []
        for p in params:
            c_type = self.type_to_c(p.type)
            if isinstance(p.type, UnknownType):
                c_type = 'int64_t'
            c_pname = self.mangle_name(p.name)
            parts.append(f"{c_type} {c_pname}")

        return ", ".join(parts)
    
    def infer_c_type(self, node: VarDecl) -> str:
        if node.type_ann:
            return self.type_to_c(node.type_ann)

        if node.value:
            if isinstance(node.value, IntLit):
                return 'int64_t'
            if isinstance(node.value, FloatLit):
                return 'double'
            if isinstance(node.value, StringLit):
                return 'const char*'
            if isinstance(node.value, BoolLit):
                return 'bool'
            if isinstance(node.value, ArrayLit):
                return self.infer_array_elem_type(node.value) + '*'
            if isinstance(node.value, StructInit):
                return self.mangle_name(node.value.name)  # struct name as C type
            if isinstance(node.value, Cast):
                return self.type_to_c(node.value.target_type)
            if isinstance(node.value, Call):
                # Check for Vec<T>.new() and HashMap<K,V>.new() constructors
                if isinstance(node.value.func, Attribute) and isinstance(node.value.func.obj, Identifier):
                    cname = node.value.func.obj.name
                    if cname.startswith("__Vec__"):
                        elem_type = cname[7:]
                        return f"TIL_DynArray_{elem_type}"
                    if cname.startswith("__HashMap__"):
                        type_parts = cname[11:]
                        return f"TIL_HashMap_{type_parts}"
                # Use unified call return type inference
                ret = self._infer_call_ret_type(node.value)
                if ret and ret != 'void':
                    return ret
            if isinstance(node.value, ListComprehension):
                return 'int64_t*'
            if isinstance(node.value, UnaryOp):
                if node.value.op == 'not':
                    return 'bool'
                return self.infer_c_type(VarDecl(value=node.value.operand))
            if isinstance(node.value, (RetryExpr, TimeoutExpr)):
                return self._guarded_value_type(node.value)
            if isinstance(node.value, FStringLit):
                return 'const char*'
            if isinstance(node.value, CharLit):
                return 'int64_t'
            if isinstance(node.value, BinaryOp):
                if node.value.op in ('==', '!=', '<', '>', '<=', '>=', 'and', 'or'):
                    return 'bool'
                ptype = self._infer_expr_print_type(node.value)
                if ptype == "str":
                    return 'const char*'
                if ptype == "float":
                    return 'double'
            if isinstance(node.value, IfExpr):
                return self.infer_c_type(VarDecl(value=node.value.then_expr))
            if isinstance(node.value, MatchExpr):
                # Infer from first arm body
                if node.value.arms:
                    first_arm = node.value.arms[0]
                    body = first_arm[2] if len(first_arm) == 3 else first_arm[1]
                    return self.infer_c_type(VarDecl(value=body))

        return 'int64_t'
    
    def infer_array_elem_type(self, node: ArrayLit) -> str:
        if not node.elements:
            return 'int64_t'
        
        first = node.elements[0]
        if isinstance(first, IntLit):
            return 'int64_t'
        if isinstance(first, FloatLit):
            return 'double'
        if isinstance(first, StringLit):
            return 'const char*'
        if isinstance(first, BoolLit):
            return 'bool'
        
        return 'int64_t'

    # ═══════════════════════════════════════════════════════════════
    # V2.0 ADVANCED CODE GENERATION
    # ═══════════════════════════════════════════════════════════════

    def gen_AssertStmt(self, node: AssertStmt) -> str:
        """assert cond, "msg" -> recoverable runtime check (via til_panic)."""
        cond = self.generate_node(node.condition)
        msg = f"Assertion failed: {node.message}" if node.message else "Assertion failed"
        msg += f" (line {node.line})" if node.line else ""
        self.emit(f'if (!({cond})) til_panic({c_string_literal(msg)});')
        return ""

    def gen_InvariantStmt(self, node: InvariantStmt) -> str:
        """invariant cond, "msg" -> recoverable runtime check (via til_panic)."""
        cond = self.generate_node(node.condition)
        msg = f"Invariant violated: {node.message}" if node.message else "Invariant violated"
        msg += f" (line {node.line})" if node.line else ""
        self.emit(f'if (!({cond})) til_panic({c_string_literal(msg)});')
        return ""

    def gen_CapabilityType(self, node: CapabilityType) -> str:
        """Emit capability type as a capability bitmask constant."""
        cap_name = self.mangle_name(node.name)
        bits = 0
        perm_map = {"read": 1, "write": 2, "execute": 4, "network": 8, "fs": 16}
        for p in node.permissions:
            bits |= perm_map.get(p, 0)
        self.emit(f"#define TIL_CAP_{cap_name} {bits}")
        self.emit(f"static uint64_t _cap_{cap_name}_mask = {bits};")
        return ""

    # ── retry / timeout: real semantics via recovery points (setjmp/longjmp) ──
    def _block_value_expr(self, body):
        expr = body
        if isinstance(body, Block):
            if not body.statements or not isinstance(body.statements[-1], ExprStmt):
                return None
            expr = body.statements[-1].expr
        if self._is_void_call(expr):
            return None
        return expr

    def _guarded_value_type(self, node) -> str:
        """C type of a retry/timeout expression: common type of body and fallback
        (the VM is dynamically typed; C needs one type: str > float > int)."""
        types = []
        for part in (node.body, node.fallback):
            expr = self._block_value_expr(part) if part is not None else None
            if expr is not None:
                types.append({"str": "const char*", "float": "double", "bool": "bool"}.get(
                    self._infer_expr_print_type(expr), "int64_t"))
        if not types:
            return "int64_t"
        for preferred in ("const char*", "double"):
            if preferred in types:
                return preferred
        return types[0]

    def _is_void_call(self, expr) -> bool:
        if not isinstance(expr, Call):
            return False
        if isinstance(expr.func, Identifier) and expr.func.name in ("print", "println", "panic", "exit"):
            return True
        return self._infer_call_ret_type(expr) == "void"

    def _emit_guarded_body(self, body, result_var: str, ctype: str = ""):
        """Body of retry/timeout; its value (last expression) goes to result_var."""
        if not isinstance(body, Block) and self._is_void_call(body):
            body = Block(statements=[ExprStmt(expr=body)])
        if isinstance(body, Block):
            stmts = body.statements
            for st in stmts[:-1]:
                self.generate_node(st)
            if stmts:
                last = stmts[-1]
                if isinstance(last, ExprStmt) and not self._is_void_call(last.expr):
                    val = self.generate_node(last.expr)
                    if val:
                        if ctype == "const char*":
                            val = self._to_c_string(last.expr, val)
                        self.emit(f"{result_var} = {val};")
                else:
                    self.generate_node(last)
        else:
            val = self.generate_node(body)
            if val:
                if ctype == "const char*":
                    val = self._to_c_string(body, val)
                self.emit(f"{result_var} = {val};")

    def _emit_fallback(self, fallback, result_var: str, ctype: str = ""):
        if fallback is None:
            self.emit("til_panic(til_panic_msg);   /* re-raise */")
            return
        self.emit("{")
        self.indent += 1
        self.emit("const char* error = til_panic_msg;")
        self.emit("(void)error;")
        self.string_vars.add("error")
        self._emit_guarded_body(fallback, result_var, ctype)
        self.indent -= 1
        self.emit("}")

    def gen_RetryExpr(self, node: RetryExpr) -> str:
        """retry(n) { body } else { fallback }: on a runtime error (assert, contract,
        bounds, unwrap, panic, timeout) the body really runs again, up to n times."""
        self._retry_counter = getattr(self, '_retry_counter', 0) + 1
        k = self._retry_counter
        ctype = self._guarded_value_type(node)
        result_var, rec = f"_retry_result_{k}", f"_rec_{k}"
        zero = '""' if ctype == "const char*" else "0"
        self.emit(f"{ctype} volatile {result_var} = {zero};")
        self.emit("{")
        self.indent += 1
        self.emit(f"volatile int64_t _retry_limit_{k} = {self.generate_node(node.count)};")
        self.emit(f"volatile int64_t _retry_i_{k} = 0;")
        self.emit(f"TIL_Recover {rec};")
        # labels instead of a C loop, so `break`/`continue` in the body still
        # refer to the enclosing TIL loop
        self.emit(f"_retry_again_{k}:")
        self.emit(f"{rec}.prev = til_recover_top; {rec}.deadline = 0; til_recover_top = &{rec};")
        self.emit(f"if (setjmp({rec}.env) == 0) {{")
        self.indent += 1
        self._recover_stack.append((rec, self._loop_depth))
        self._emit_guarded_body(node.body, result_var, ctype)
        self._recover_stack.pop()
        self.emit(f"til_recover_top = {rec}.prev;")
        self.indent -= 1
        self.emit("} else {")
        self.indent += 1
        self.emit(f"til_recover_top = {rec}.prev;")
        self.emit(f"if (++_retry_i_{k} < _retry_limit_{k}) goto _retry_again_{k};")
        self._emit_fallback(node.fallback, result_var, ctype)
        self.indent -= 1
        self.emit("}")
        self.indent -= 1
        self.emit("}")
        return result_var

    def gen_TimeoutExpr(self, node: TimeoutExpr) -> str:
        """timeout(ms) { body } else { fallback }: the body is interrupted once it
        runs longer than ms milliseconds (checked at loop back-edges / calls)."""
        self._timeout_counter = getattr(self, '_timeout_counter', 0) + 1
        k = self._timeout_counter
        ctype = self._guarded_value_type(node)
        result_var, rec = f"_timeout_result_{k}", f"_trec_{k}"
        zero = '""' if ctype == "const char*" else "0"
        self.emit(f"{ctype} volatile {result_var} = {zero};")
        self.emit("{")
        self.indent += 1
        self.emit(f"TIL_Recover {rec};")
        self.emit(f"{rec}.prev = til_recover_top; {rec}.deadline = til_now_ms() + (double)({self.generate_node(node.duration)});")
        self.emit(f"til_recover_top = &{rec};")
        self.emit(f"if (setjmp({rec}.env) == 0) {{")
        self.indent += 1
        self._recover_stack.append((rec, self._loop_depth))
        self._emit_guarded_body(node.body, result_var, ctype)
        self._recover_stack.pop()
        self.emit(f"til_recover_top = {rec}.prev;")
        self.indent -= 1
        self.emit("} else {")
        self.indent += 1
        self.emit(f"til_recover_top = {rec}.prev;")
        self._emit_fallback(node.fallback, result_var, ctype)
        self.indent -= 1
        self.emit("}")
        self.indent -= 1
        self.emit("}")
        return result_var

    def gen_IntentBlock(self, node: IntentBlock) -> str:
        """Generate intent block as documented code block."""
        self.emit(c_comment(f"INTENT: {node.description}"))
        self.emit("{")
        self.indent += 1
        if isinstance(node.body, Block):
            self.generate_node(node.body)
        else:
            val = self.generate_node(node.body)
            if val:
                self.emit(f"{val};")
        self.indent -= 1
        self.emit("}")
        return ""

    def gen_PerformEffect(self, node: PerformEffect) -> str:
        """A program without `handle`: `perform E(args)` calls the top-level
        function E, its default handler (TIL 3.0 behaviour)."""
        if node.name not in self.functions:
            raise NativeUnsupported(f"line {node.line}: effect '{node.name}' has no default handler "
                                    "function; the TIL VM reports unhandled effects at run time")
        name = self.mangle_name(node.name)
        args = ", ".join(self.generate_node(a) for a in node.args)
        return f"til_{name}({args})"

    def gen_EffectHandler(self, node: EffectHandler) -> str:
        raise NativeUnsupported(f"line {node.line}: the native backend does not support effect "
                                "handlers (handle ... with); use the TIL VM")

    def gen_DurationLit(self, node: DurationLit) -> str:
        """Convert duration to milliseconds as int64."""
        multipliers = {"ms": 1, "s": 1000, "m": 60000, "h": 3600000}
        ms = int(node.value * multipliers.get(node.unit, 1))
        return str(ms)

    def gen_FallbackExpr(self, node: FallbackExpr) -> str:
        """Generate fallback as ternary-like expression."""
        expr = self.generate_node(node.expr)
        default = self.generate_node(node.default)
        return f"({expr} ? {expr} : {default})"

    def gen_GenericFunc(self, node: GenericFunc) -> str:
        """Generate generic function as monomorphized C (currently emits base version)."""
        if node.func:
            self.generate_function(node.func)
        return ""

    def gen_RequiresCap(self, node: RequiresCap) -> str:
        """Generate capability check."""
        cap_name = self.mangle_name(node.cap_name)
        self.emit(f'// Requires capability: {node.cap_name}')
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
#              TIL VM - PORTABLE BYTECODE BACKEND ("TIL Micro")
# ═══════════════════════════════════════════════════════════════════════════════
#
# The second backend of TIL. Instead of C it emits compact bytecode (.tbc) that
# runs on a tiny virtual machine:
#   * the reference VM below (pure Python: `til run --vm`, no C compiler needed)
#   * vm/tilvm.c - a portable C99 VM with no malloc, a fixed arena, a compacting
#     GC and pluggable I/O, small enough for graphing calculators (Z80 / eZ80 /
#     SH4 / ARM) and microcontrollers.
# Values are dynamically tagged, so every language feature has the same meaning
# on every device. Format: docs/TIL_BYTECODE.md.

import struct as _struct
import math as _math
import time as _time

TBC_MAGIC = b"TILB"
TBC_VERSION = 2          # written by the compiler (TIL 3.1)
TBC_MIN_VERSION = 1      # oldest version the VMs still load (TIL 3.0)


class Op:
    NOP = 0x00; PUSH_NIL = 0x01; PUSH_TRUE = 0x02; PUSH_FALSE = 0x03
    PUSH_I8 = 0x04; PUSH_I16 = 0x05; PUSH_CONST = 0x06; POP = 0x07
    DUP = 0x08; DUP2 = 0x09; SWAP = 0x0A
    LOAD_LOCAL = 0x10; STORE_LOCAL = 0x11; LOAD_GLOBAL = 0x12; STORE_GLOBAL = 0x13
    LOAD_UPVAL = 0x14; STORE_UPVAL = 0x15; LOAD_CELL = 0x16; STORE_CELL = 0x17
    MAKE_CELL = 0x18; LOAD_UPVAL_RAW = 0x19; ARG_DEFAULT = 0x1A
    ADD = 0x20; SUB = 0x21; MUL = 0x22; DIV = 0x23; MOD = 0x24; POW = 0x25
    NEG = 0x26; NOT = 0x27; BAND = 0x28; BOR = 0x29; BXOR = 0x2A; BNOT = 0x2B
    SHL = 0x2C; SHR = 0x2D
    EQ = 0x30; NE = 0x31; LT = 0x32; LE = 0x33; GT = 0x34; GE = 0x35
    JUMP = 0x40; JUMP_IF_FALSE = 0x41; JUMP_IF_TRUE = 0x42
    CALL = 0x50; CALL_VALUE = 0x51; INVOKE = 0x52; RETURN = 0x53
    RETURN_NIL = 0x54; NATIVE = 0x55
    NEW_ARRAY = 0x60; NEW_MAP = 0x61; NEW_STRUCT = 0x62; GET_INDEX = 0x63
    SET_INDEX = 0x64; GET_FIELD = 0x65; SET_FIELD = 0x66; GET_SLOT = 0x67
    MAKE_CLOSURE = 0x68; PUSH_FUNC = 0x69; CONCAT = 0x6A; FOR_ITER = 0x6B
    MATCH_TYPE = 0x6C; CAST = 0x6D; IS_ARRAY_LEN = 0x6E
    TRY = 0x70; TRY_TIMEOUT = 0x71; END_TRY = 0x72; PANIC = 0x73
    UNWRAP_OR_RETURN = 0x74; UNWRAP_OR_PANIC = 0x75
    # effects with resumption (TIL 3.1, see docs/dev/TIL_3_1_DESIGN.md §8)
    HANDLE = 0x76; END_HANDLE = 0x77; PERFORM = 0x78; RESUME = 0x79
    HALT = 0x7F
    # 0x80-0xBF: superinstructions (TIL 3.1, docs/TIL_BYTECODE_V2.md). The
    # compiler's peephole pass (fuse_superinstructions) emits them; each does
    # exactly what the sequence in its comment does.
    LOAD_LOCAL2 = 0x80          # LOAD_LOCAL a; LOAD_LOCAL b
    LOAD_LOCAL_FIELD = 0x81     # LOAD_LOCAL a; GET_FIELD name
    INC_LOCAL = 0x82            # LOAD_LOCAL a; PUSH_I8 k; ADD; STORE_LOCAL a
    CAST_LOCAL = 0x83           # LOAD_LOCAL a; CAST kind; STORE_LOCAL a
    CAST_RETURN = 0x84          # CAST kind; RETURN
    ADD_STORE_LOCAL = 0x85      # ADD; STORE_LOCAL a
    NATIVE_POP = 0x86           # NATIVE id argc; POP
    INDEX_L = 0x87              # LOAD_LOCAL b; GET_INDEX
    INDEX_LL = 0x88             # LOAD_LOCAL a; LOAD_LOCAL b; GET_INDEX
    RETURN_LOCAL = 0x89         # LOAD_LOCAL a; RETURN
    FOR_ITER_STORE = 0x8A       # FOR_ITER it off; STORE_LOCAL x
    JUMP_IF_NOT = 0x90          # <cmp>; JUMP_IF_FALSE off        (cmp = EQ..GE opcode)
    JUMP_IF_NOT_L = 0x91        # LOAD_LOCAL b; <cmp>; JUMP_IF_FALSE off
    JUMP_IF_NOT_I = 0x92        # PUSH_I8/PUSH_I16 k; <cmp>; JUMP_IF_FALSE off
    JUMP_IF_NOT_LL = 0x93       # LOAD_LOCAL a; LOAD_LOCAL b; <cmp>; JUMP_IF_FALSE off
    JUMP_IF_NOT_LI = 0x94       # LOAD_LOCAL a; PUSH_I8/PUSH_I16 k; <cmp>; JUMP_IF_FALSE off
    # <op>_I k = PUSH_I8/PUSH_I16 k; <op>   (k: i16; 0xA0 + index in FUSED_BINOPS)
    ADD_I = 0xA0; SUB_I = 0xA1; MUL_I = 0xA2; DIV_I = 0xA3; MOD_I = 0xA4; BAND_I = 0xA5
    BOR_I = 0xA6; BXOR_I = 0xA7; SHL_I = 0xA8; SHR_I = 0xA9; EQ_I = 0xAA; NE_I = 0xAB
    LT_I = 0xAC; LE_I = 0xAD; GT_I = 0xAE; GE_I = 0xAF
    # <op>_L b = LOAD_LOCAL b; <op>   (0xB0 + index in FUSED_BINOPS)
    ADD_L = 0xB0; SUB_L = 0xB1; MUL_L = 0xB2; DIV_L = 0xB3; MOD_L = 0xB4; BAND_L = 0xB5
    BOR_L = 0xB6; BXOR_L = 0xB7; SHL_L = 0xB8; SHR_L = 0xB9; EQ_L = 0xBA; NE_L = 0xBB
    LT_L = 0xBC; LE_L = 0xBD; GT_L = 0xBE; GE_L = 0xBF


OP_NAMES = {v: k for k, v in vars(Op).items() if not k.startswith('_')}

# base operators of the <op>_I / <op>_L superinstructions (opcode & 15)
FUSED_BINOPS = [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR,
                Op.EQ, Op.NE, Op.LT, Op.LE, Op.GT, Op.GE]
CMP_OPS = (Op.EQ, Op.NE, Op.LT, Op.LE, Op.GT, Op.GE)

# Operand layout per opcode: b=u8 c=i8 h=u16 s=i16 (little endian)
OP_OPERANDS = {
    Op.PUSH_I8: 'c', Op.PUSH_I16: 's', Op.PUSH_CONST: 'h',
    Op.LOAD_LOCAL: 'b', Op.STORE_LOCAL: 'b', Op.LOAD_GLOBAL: 'h', Op.STORE_GLOBAL: 'h',
    Op.LOAD_UPVAL: 'b', Op.STORE_UPVAL: 'b', Op.LOAD_CELL: 'b', Op.STORE_CELL: 'b',
    Op.MAKE_CELL: 'b', Op.LOAD_UPVAL_RAW: 'b', Op.ARG_DEFAULT: 'bs',
    Op.JUMP: 's', Op.JUMP_IF_FALSE: 's', Op.JUMP_IF_TRUE: 's',
    Op.CALL: 'hb', Op.CALL_VALUE: 'b', Op.INVOKE: 'hbb', Op.NATIVE: 'bb',
    Op.NEW_ARRAY: 'h', Op.NEW_MAP: 'h', Op.NEW_STRUCT: 'hb', Op.GET_FIELD: 'h',
    Op.SET_FIELD: 'h', Op.GET_SLOT: 'b', Op.MAKE_CLOSURE: 'hb', Op.PUSH_FUNC: 'h',
    Op.CONCAT: 'b', Op.FOR_ITER: 'bs', Op.MATCH_TYPE: 'h', Op.CAST: 'b',
    Op.IS_ARRAY_LEN: 'h', Op.TRY: 's', Op.TRY_TIMEOUT: 's',
    Op.HANDLE: 'bs', Op.PERFORM: 'hhb',
    # superinstructions
    Op.LOAD_LOCAL2: 'bb', Op.LOAD_LOCAL_FIELD: 'bh', Op.INC_LOCAL: 'bc', Op.CAST_LOCAL: 'bb',
    Op.CAST_RETURN: 'b', Op.ADD_STORE_LOCAL: 'b', Op.NATIVE_POP: 'bb', Op.INDEX_L: 'b',
    Op.INDEX_LL: 'bb', Op.RETURN_LOCAL: 'b', Op.FOR_ITER_STORE: 'bbs',
    Op.JUMP_IF_NOT: 'bs', Op.JUMP_IF_NOT_L: 'bbs', Op.JUMP_IF_NOT_I: 'bss', Op.JUMP_IF_NOT_LL: 'bbbs',
    Op.JUMP_IF_NOT_LI: 'bbss',
}
for _i in range(16):
    OP_OPERANDS[0xA0 + _i] = 's'      # <op>_I k (i16)
    OP_OPERANDS[0xB0 + _i] = 'b'      # <op>_L b
del _i
_OPERAND_SIZE = {'b': 1, 'c': 1, 'h': 2, 's': 2}

# Builtin functions callable with NATIVE <id> <argc>. The order is part of the
# bytecode format (shared with vm/tilvm.c).
NATIVES = [
    'print', 'len', 'str', 'int', 'float', 'abs', 'sqrt', 'pow', 'sin', 'cos',
    'tan', 'log', 'exp', 'floor', 'ceil', 'round', 'min', 'max', 'input', 'chr',
    'ord', 'range', 'type_of', 'format', 'clock_ms', 'random', 'seed', 'sleep_ms',
    'exit', 'bool', 'sum', 'panic', 'iter', 'println',
    # TIL 3.1 (bytecode v2): files and bytes
    'read_file', 'write_file', 'file_exists', 'args', 'bytes_to_str', 'str_bytes',
    # screen and keys (calculators, terminals, browser canvas)
    'screen_init', 'screen_clear', 'draw_pixel', 'draw_line', 'draw_rect', 'fill_rect',
    'draw_circle', 'draw_text', 'get_pixel', 'screen_show', 'key_pressed', 'wait_key',
    'screen_width', 'screen_height',
    # battery saving
    'energy_mode',
]
NATIVE_ID = {name: i for i, name in enumerate(NATIVES)}

# (min, max) argument counts; max None = any number. The compiler checks both,
# the VMs check the minimum (bytecode may come from other compilers).
NATIVE_ARITY = {
    'print': (0, None), 'len': (1, 1), 'str': (1, 1), 'int': (1, 1), 'float': (1, 1),
    'abs': (1, 1), 'sqrt': (1, 1), 'pow': (2, 2), 'sin': (1, 1), 'cos': (1, 1),
    'tan': (1, 1), 'log': (1, 1), 'exp': (1, 1), 'floor': (1, 1), 'ceil': (1, 1),
    'round': (1, 1), 'min': (1, None), 'max': (1, None), 'input': (0, 1), 'chr': (1, 1),
    'ord': (1, 1), 'range': (1, 3), 'type_of': (1, 1), 'format': (2, 2), 'clock_ms': (0, 0),
    'random': (0, 2), 'seed': (1, 1), 'sleep_ms': (1, 1), 'exit': (0, 1), 'bool': (1, 1),
    'sum': (1, None), 'panic': (0, 1), 'iter': (1, 1), 'println': (0, None),
    'read_file': (1, 1), 'write_file': (2, 2), 'file_exists': (1, 1), 'args': (0, 0),
    'bytes_to_str': (1, 1), 'str_bytes': (1, 1),
    'screen_init': (2, 2), 'screen_clear': (0, 1), 'draw_pixel': (3, 3), 'draw_line': (5, 5),
    'draw_rect': (5, 5), 'fill_rect': (5, 5), 'draw_circle': (4, 4), 'draw_text': (4, 4),
    'get_pixel': (2, 2), 'screen_show': (0, 0), 'key_pressed': (0, 0), 'wait_key': (0, 0),
    'screen_width': (0, 0), 'screen_height': (0, 0), 'energy_mode': (0, 1),
}
# runtime minimum (range always gets 2..3 arguments from the compiler)
NATIVE_MIN_ARGS = [2 if n == 'range' else NATIVE_ARITY[n][0] for n in NATIVES]

# Builtin methods (INVOKE ... <method id> ...) of str / array / map / Option / Result.
METHODS = [
    'len', 'push', 'pop', 'get', 'set', 'contains', 'starts_with', 'ends_with',
    'trim', 'to_upper', 'to_lower', 'replace', 'slice', 'find', 'eq', 'split',
    'repeat', 'to_int', 'to_float', 'is_empty', 'insert', 'remove', 'index_of',
    'reverse', 'sort', 'join', 'first', 'last', 'clear', 'sum', 'min', 'max',
    'has', 'delete', 'keys', 'values', 'get_or', 'unwrap', 'expect', 'unwrap_or',
    'is_some', 'is_none', 'is_ok', 'is_err', 'error', 'to_string', 'copy', 'count',
    'map', 'filter', 'reduce', 'each', 'any', 'all',
]
METHOD_ID = {name: i for i, name in enumerate(METHODS)}
NO_METHOD = 0xFF

# Higher-order array methods are written in TIL itself (no native->bytecode
# re-entrancy is needed in the C VM). Compiled into a module only when used.
PRELUDE_SOURCE = {
    'map': "__prelude_map(a, f)\n    var out = []\n    for x in a\n        out.push(f(x))\n    return out\n",
    'filter': "__prelude_filter(a, f)\n    var out = []\n    for x in a\n        if f(x)\n            out.push(x)\n    return out\n",
    'reduce': "__prelude_reduce(a, f, init)\n    var acc = init\n    for x in a\n        acc = f(acc, x)\n    return acc\n",
    'each': "__prelude_each(a, f)\n    for x in a\n        f(x)\n",
    'any': "__prelude_any(a, f)\n    for x in a\n        if f(x)\n            return true\n    return false\n",
    'all': "__prelude_all(a, f)\n    for x in a\n        if not f(x)\n            return false\n    return true\n",
}

CAST_KINDS = {'int': 0, 'float': 1, 'str': 2, 'bool': 3}
BUILTIN_TYPES = [('Some', ['value']), ('Ok', ['value']), ('Err', ['error'])]
TYPE_SOME, TYPE_OK, TYPE_ERR = 0, 1, 2

# Optional debug-info section (TIL 3.1; docs/TIL_DEBUGGER.md), right after the
# line table:  n u16, then n records  func u16, slot u8, name u16, start_pc u32, end_pc u32
# (name = index of a string constant; pcs absolute in the code section, like the
# line table). Three kinds of record:
#   * a local: func = function index, slot = its local slot; live while
#     start_pc <= pc < end_pc (from just after its first store to the end of its scope)
#   * a global: func = DEBUG_GLOBAL; start_pc = end_pc = the global's index (the u16
#     operand of LOAD_GLOBAL - the u8 slot field, 0 here, could not hold indices > 255)
#   * an upvalue: func = DEBUG_UPVAL, slot = the upvalue's index; start_pc = end_pc =
#     the index of the function (lambda / effect clause) whose closure holds it
# Written only on request (`til build --debug-info`, `til debug`), never with
# --strip, so default bytecode is unchanged. Both VMs ignore it (the C VM stops
# reading after the line table); TBCModule.from_bytes reads it.
DEBUG_GLOBAL = 0xFFFF
DEBUG_UPVAL = 0xFFFE
DEBUG_RECORD = '<HBHII'                 # 13 bytes per record


# Energy saving (docs/TIL_ENERGY.md): the program's mode (the .tbc header
# flags byte, bits 2-3: the #[energy: ...] pragma; energy_mode(m) at run time).
# Both VMs idle 1 ms at every step that is a multiple of 512 in mode low, of
# 4096 in mode balanced, never in mode performance; while a frame of an
# #[energy: low] function is active they behave as in mode low.
ENERGY_PERFORMANCE, ENERGY_BALANCED, ENERGY_LOW = 0, 1, 2
ENERGY_MODES = {'performance': ENERGY_PERFORMANCE, 'balanced': ENERGY_BALANCED, 'low': ENERGY_LOW}
TBC_ENERGY_SHIFT = 2


class TBCFunction:
    FLAG_METHOD = 1      # param 0 is `self`
    FLAG_PROPAGATE = 2   # `?` inside returns None/Err from this function
    FLAG_LAMBDA = 4
    FLAG_ENERGY_LOW = 8      # #[energy: low]: the VM idles while running it
    FLAG_EFFECT_CLAUSE = 16  # a `handle ... with E(x) => ...` clause

    def __init__(self, name: str):
        self.name = name
        self.n_params = 0
        self.n_required = 0
        self.n_locals = 0
        self.n_upvals = 0
        self.flags = 0
        self.code = bytearray()
        self.lines: List[Tuple[int, int]] = []   # (offset in code, source line)
        self.offset = 0                           # absolute offset in module code
        # debug info (BytecodeCompiler(debug_info=True) or a file's debug section):
        # [slot, name, start, end] = local `name` lives in `slot` while start <= pc < end
        # (offsets in code, like `lines`); upval_names[i] = name of upvalue i
        self.locals: List[list] = []
        self.upval_names: List[str] = []


class TBCModule:
    """An in-memory bytecode module (what a .tbc file contains)."""

    def __init__(self):
        self.consts: List[Any] = []          # int | float | bytes
        self._const_index: Dict[Any, int] = {}
        self.funcs: List[TBCFunction] = []
        self.types: List[Tuple[str, List[str]]] = list(BUILTIN_TYPES)
        self.methods: List[Tuple[int, int, int]] = []   # (type, name const, func)
        self.globals: List[str] = []
        self.entry = 0
        self.prelude: Dict[int, int] = {}    # method id -> func index
        self.code = b""
        self.lines: List[Tuple[int, int]] = []
        self.uses_float = False
        self.energy = ENERGY_PERFORMANCE     # initial energy mode (#[energy: ...] pragma)
        self.version = TBC_VERSION           # of a loaded file (written files: TBC_VERSION)
        self.debug_info = False              # write the debug-info section (see DEBUG_RECORD)
        self.raw: Optional[bytes] = None     # from_bytes(): the bytes it was read from

    def const(self, value) -> int:
        if isinstance(value, str):
            value = value.encode('utf-8')
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, float):
            key = ('f', _struct.pack('<d', value))
            self.uses_float = True
        elif isinstance(value, int):
            key = ('i', value)
        else:
            key = ('s', bytes(value))
        if key not in self._const_index:
            self._const_index[key] = len(self.consts)
            self.consts.append(value)
        return self._const_index[key]

    def link(self):
        """Concatenate function code into one code section."""
        code = bytearray()
        self.lines = []
        for f in self.funcs:
            f.offset = len(code)
            code += f.code
            for off, line in f.lines:
                self.lines.append((f.offset + off, line))
        self.code = bytes(code)
        self.methods.sort(key=lambda m: (m[0], m[1]))

    def debug_records(self) -> List[Tuple[int, int, str, int, int]]:
        """Records of the debug-info section, in file order: (func, slot, name,
        start_pc, end_pc) with absolute pcs (call link() first)."""
        recs = []
        for i, f in enumerate(self.funcs):
            for slot, name, start, end in sorted(f.locals, key=lambda r: (r[2], r[0], r[3], r[1])):
                if end > start:
                    recs.append((i, slot, name, f.offset + start, f.offset + end))
        for i, f in enumerate(self.funcs):
            for k, name in enumerate(f.upval_names):
                recs.append((DEBUG_UPVAL, k, name, i, i))
        for g, name in enumerate(self.globals):
            recs.append((DEBUG_GLOBAL, 0, name, g, g))
        return recs

    # ── serialization ───────────────────────────────────────────────────────
    def to_bytes(self, strip: bool = False) -> bytes:
        """Write the module (call serialize(): it interns all names first)."""
        self.link()
        out = bytearray()
        flags = (1 if self.uses_float else 0) | (0 if strip else 2) | (self.energy << TBC_ENERGY_SHIFT)
        out += TBC_MAGIC + bytes([TBC_VERSION, flags])
        out += _struct.pack('<HH', len(self.globals), self.entry)
        out += _struct.pack('<H', len(self.consts))
        for c in self.consts:
            if isinstance(c, float):
                out += b'\x02' + _struct.pack('<d', c)
            elif isinstance(c, int):
                out += b'\x01' + _struct.pack('<q', wrap_int64(c))
            else:
                if len(c) > 0xFFFF:
                    raise CompileError("string constant longer than 65535 bytes")
                out += b'\x03' + _struct.pack('<H', len(c)) + c
        out += _struct.pack('<H', len(self.funcs))
        for f in self.funcs:
            out += _struct.pack('<HBBBBBI', self.const(f.name), f.n_params, f.n_required,
                                f.n_locals, f.n_upvals, f.flags, f.offset)
        out += _struct.pack('<H', len(self.types))
        for name, fields in self.types:
            out += _struct.pack('<HB', self.const(name), len(fields))
            for fld in fields:
                out += _struct.pack('<H', self.const(fld))
        out += _struct.pack('<H', len(self.methods))
        for t, n, fidx in self.methods:
            out += _struct.pack('<HHH', t, n, fidx)
        out += bytes([len(self.prelude)])
        for mid, fidx in sorted(self.prelude.items()):
            out += _struct.pack('<BH', mid, fidx)
        out += _struct.pack('<I', len(self.code)) + self.code
        lines = [] if strip else self.lines
        out += _struct.pack('<H', min(len(lines), 0xFFFF))
        for pc, line in lines[:0xFFFF]:
            out += _struct.pack('<IH', pc, min(line, 0xFFFF))
        if self.debug_info and not strip:
            recs = self.debug_records()
            if len(recs) > 0xFFFF or len(self.funcs) >= DEBUG_UPVAL:
                raise CompileError("program too large for debug info (more than 65535 variables or "
                                   "65533 functions); build without --debug-info")
            out += _struct.pack('<H', len(recs))
            for func, slot, name, start, end in recs:
                out += _struct.pack(DEBUG_RECORD, func, slot, self.const(name), start, end)
        return bytes(out)

    def serialize(self, strip: bool = False) -> bytes:
        for f in self.funcs:
            self.const(f.name)
        for name, fields in self.types:
            self.const(name)
            for fld in fields:
                self.const(fld)
        if self.debug_info and not strip:       # variable names: string constants
            self.link()
            for rec in self.debug_records():
                self.const(rec[2])
        return self.to_bytes(strip)

    @staticmethod
    def from_bytes(data: bytes) -> 'TBCModule':
        m = TBCModule()
        m.types = []
        if data[:4] != TBC_MAGIC:
            raise ValueError("not a TIL bytecode file (bad magic)")
        damaged = "damaged bytecode file (truncated or malformed)"
        if len(data) < 16:
            raise ValueError(damaged)
        if not TBC_MIN_VERSION <= data[4] <= TBC_VERSION:
            raise ValueError(f"unsupported bytecode version {data[4]} "
                             f"(this VM loads versions {TBC_MIN_VERSION}..{TBC_VERSION})")
        m.version = data[4]
        flags = data[5]
        m.uses_float = bool(flags & 1)
        m.energy = (flags >> TBC_ENERGY_SHIFT) & 3
        if m.energy > ENERGY_LOW:
            raise ValueError(f"bad bytecode header (energy mode {m.energy})")
        pos = 6

        def rd(fmt):
            nonlocal pos
            size = _struct.calcsize('<' + fmt)
            if pos + size > len(data):
                raise ValueError(damaged)
            vals = _struct.unpack_from('<' + fmt, data, pos)
            pos += size
            return vals if len(vals) > 1 else vals[0]

        def name(ix: int) -> str:
            # names of functions, types and fields: string constants (as vm/tilvm.c checks)
            if ix >= len(m.consts) or not isinstance(m.consts[ix], bytes):
                raise ValueError(damaged)
            return m.consts[ix].decode('utf-8', 'replace')
        n_globals, m.entry = rd('HH')
        m.globals = [f"g{i}" for i in range(n_globals)]
        for _ in range(rd('H')):
            tag = rd('B')
            if tag == 1:
                value = rd('q')
            elif tag == 2:
                value = rd('d')
            elif tag == 3:
                n = rd('H')
                if pos + n > len(data):
                    raise ValueError(damaged)
                value = bytes(data[pos:pos + n])
                pos += n
            else:
                raise ValueError(f"bad constant tag {tag}")
            key = (('f', _struct.pack('<d', value)) if isinstance(value, float) else
                   ('i', value) if isinstance(value, int) else ('s', value))
            m._const_index.setdefault(key, len(m.consts))
            m.consts.append(value)
        for _ in range(rd('H')):
            name_ix, np_, nr, nl, nu, fl, off = rd('HBBBBBI')
            f = TBCFunction(name(name_ix))
            f.n_params, f.n_required, f.n_locals, f.n_upvals, f.flags, f.offset = np_, nr, nl, nu, fl, off
            m.funcs.append(f)
        if m.entry >= len(m.funcs):
            raise ValueError(damaged)
        for _ in range(rd('H')):
            name_ix, nf = rd('HB')
            tname = name(name_ix)
            m.types.append((tname, [name(rd('H')) for _ in range(nf)]))
        for _ in range(rd('H')):
            m.methods.append(rd('HHH'))
            if m.methods[-1][2] >= len(m.funcs):
                raise ValueError(damaged)
        for _ in range(rd('B')):
            mid, fidx = rd('BH')
            if fidx >= len(m.funcs):
                raise ValueError(damaged)
            m.prelude[mid] = fidx
        n = rd('I')
        if pos + n > len(data) or any(f.offset >= n for f in m.funcs):
            raise ValueError(damaged)
        m.code = bytes(data[pos:pos + n])
        pos += n
        for _ in range(rd('H')):
            m.lines.append(rd('IH'))
        debug = None
        if pos < len(data):                     # the optional debug-info section
            size = _struct.calcsize(DEBUG_RECORD)
            if pos + 2 > len(data) or pos + 2 + size * _struct.unpack_from('<H', data, pos)[0] > len(data):
                raise ValueError("truncated debug-info section")
            debug = [rd(DEBUG_RECORD[1:]) for _ in range(rd('H'))]
        # split the code section back into functions (so the module can be re-linked)
        order = sorted(range(len(m.funcs)), key=lambda i: m.funcs[i].offset)
        for k, i in enumerate(order):
            f = m.funcs[i]
            end = m.funcs[order[k + 1]].offset if k + 1 < len(order) else len(m.code)
            f.code = bytearray(m.code[f.offset:end])
            f.lines = [(pc - f.offset, line) for pc, line in m.lines if f.offset <= pc < end]
        if debug is not None:
            m.read_debug_records(debug)
        m.raw = bytes(data)         # the file itself (portable snapshots hash it)
        return m

    def read_debug_records(self, recs):
        """Debug-info records (func, slot, name const, start_pc, end_pc) of a file ->
        per-function locals and upvalue names, names of the globals."""
        self.debug_info = True

        def name_of(ix):
            if ix >= len(self.consts) or not isinstance(self.consts[ix], bytes):
                raise ValueError(f"bad debug-info section (name {ix} is not a string constant)")
            return self.consts[ix].decode('utf-8', 'replace')
        for func, slot, name, start, end in recs:
            if func == DEBUG_GLOBAL:
                if start < len(self.globals):
                    self.globals[start] = name_of(name)
            elif func == DEBUG_UPVAL:
                if start < len(self.funcs):
                    names = self.funcs[start].upval_names
                    names.extend("?" for _ in range(slot + 1 - len(names)))
                    names[slot] = name_of(name)
            elif func < len(self.funcs):
                f = self.funcs[func]
                f.locals.append([slot, name_of(name), start - f.offset, end - f.offset])
            else:
                raise ValueError(f"bad debug-info section (function {func} does not exist)")


# ═══════════════════════════════════════════════════════════════════════════════
#             PEEPHOLE PASS: SUPERINSTRUCTIONS (TIL 3.1, §13 of the design)
# ═══════════════════════════════════════════════════════════════════════════════
# Frequent instruction sequences are replaced by one superinstruction (opcodes
# 0x80-0xBF, docs/TIL_BYTECODE_V2.md) that does exactly the same thing in one
# dispatch. Rules that keep every observable behaviour unchanged:
#   * a sequence never spans a jump target or a line-table entry: only its first
#     instruction may be one, so jumps, error lines and tracebacks stay the same;
#   * sequences contain no calls, no TRY/END_TRY (recovery points) and no effect
#     instruction (HANDLE/END_HANDLE/PERFORM/RESUME, also checked explicitly:
#     _NEVER_FUSED); their jumps count a step when they go backwards, like
#     JUMP_IF_FALSE. A perform returns (resume) to the instruction after it and an
#     aborted handle continues at its HANDLE target, both instruction starts;
#   * jump offsets and the line table are rebuilt for the shorter code.
# `JUMP 0` (a jump to the next instruction) is dropped. The choice of sequences
# minimises the number of instructions per function (dynamic programming).

# instructions whose last operand is a jump offset (relative to the instruction end)
_JUMP_OPS = frozenset([Op.JUMP, Op.JUMP_IF_FALSE, Op.JUMP_IF_TRUE, Op.TRY, Op.TRY_TIMEOUT,
                       Op.FOR_ITER, Op.ARG_DEFAULT, Op.HANDLE, Op.JUMP_IF_NOT, Op.JUMP_IF_NOT_L,
                       Op.JUMP_IF_NOT_I, Op.JUMP_IF_NOT_LL, Op.JUMP_IF_NOT_LI, Op.FOR_ITER_STORE])
_PUSH_INT = (Op.PUSH_I8, Op.PUSH_I16)       # small integer constants (immediates of fused ops)
# never part of a fused sequence (effects: docs/TIL_EFFECTS.md)
_NEVER_FUSED = frozenset([Op.HANDLE, Op.END_HANDLE, Op.PERFORM, Op.RESUME])
_FUSED_INDEX = {op: i for i, op in enumerate(FUSED_BINOPS)}
# cost of one instruction when the pass chooses sequences (LOAD_LOCAL2 costs
# 1/20 more: LOAD_LOCAL + <op>_L, which also saves a push, is preferred)
_COST = 20


def _decode_insns(code) -> Optional[List[list]]:
    """[pc, op, operands, end] per instruction; None if the code cannot be decoded."""
    out = []
    pc, n = 0, len(code)
    while pc < n:
        op = code[pc]
        if op not in OP_NAMES:
            return None
        p = pc + 1
        vals = []
        for kind in OP_OPERANDS.get(op, ''):
            if p + _OPERAND_SIZE[kind] > n:
                return None
            if kind == 'b':
                vals.append(code[p])
            elif kind == 'c':
                vals.append(code[p] - 256 if code[p] > 127 else code[p])
            else:
                v = code[p] | (code[p + 1] << 8)
                vals.append(v - 0x10000 if kind == 's' and v >= 0x8000 else v)
            p += _OPERAND_SIZE[kind]
        out.append([pc, op, vals, p])
        pc = p
    return out


def _fusions_at(insns: List[list], i: int, boundary: Set[int]):
    """Superinstructions that can replace insns[i:i+n]: (n, op, operands, cost).
    Jump operands are absolute targets here (relocated by the caller). Costs are
    integers (one instruction = _COST): the choice is made in exact arithmetic,
    which the self-hosted compiler (selfhost/tilc.til) reproduces."""
    def ops(n):
        if i + n > len(insns) or any(insns[j][0] in boundary for j in range(i + 1, i + n)):
            return None
        group = [insns[j] for j in range(i, i + n)]
        if any(ins[1] in _NEVER_FUSED for ins in group):
            return None
        return group

    found = []
    g = ops(4)
    if g:
        (_, o0, v0, _e0), (_, o1, v1, _e1), (_, o2, v2, _e2), (_, o3, v3, e3) = g
        if o0 == Op.LOAD_LOCAL and o1 == Op.PUSH_I8 and o2 == Op.ADD and o3 == Op.STORE_LOCAL \
                and v0[0] == v3[0]:
            found.append((4, Op.INC_LOCAL, [v0[0], v1[0]], _COST))
        if o0 == Op.LOAD_LOCAL and o2 in CMP_OPS and o3 == Op.JUMP_IF_FALSE:
            if o1 == Op.LOAD_LOCAL:
                found.append((4, Op.JUMP_IF_NOT_LL, [o2, v0[0], v1[0], e3 + v3[0]], _COST))
            elif o1 in _PUSH_INT:
                found.append((4, Op.JUMP_IF_NOT_LI, [o2, v0[0], v1[0], e3 + v3[0]], _COST))
    g = ops(3)
    if g:
        (_, o0, v0, _e0), (_, o1, v1, _e1), (_, o2, v2, e2) = g
        if o0 == Op.LOAD_LOCAL and o1 == Op.CAST and o2 == Op.STORE_LOCAL and v0[0] == v2[0]:
            found.append((3, Op.CAST_LOCAL, [v0[0], v1[0]], _COST))
        if o1 in CMP_OPS and o2 == Op.JUMP_IF_FALSE:
            if o0 == Op.LOAD_LOCAL:
                found.append((3, Op.JUMP_IF_NOT_L, [o1, v0[0], e2 + v2[0]], _COST))
            elif o0 in _PUSH_INT:
                found.append((3, Op.JUMP_IF_NOT_I, [o1, v0[0], e2 + v2[0]], _COST))
        if o0 == Op.LOAD_LOCAL and o1 == Op.LOAD_LOCAL and o2 == Op.GET_INDEX:
            found.append((3, Op.INDEX_LL, [v0[0], v1[0]], _COST))
    g = ops(2)
    if g:
        (_, o0, v0, e0), (_, o1, v1, e1) = g
        if o0 in CMP_OPS and o1 == Op.JUMP_IF_FALSE:
            found.append((2, Op.JUMP_IF_NOT, [o0, e1 + v1[0]], _COST))
        elif o0 == Op.LOAD_LOCAL:
            if o1 == Op.GET_INDEX:
                found.append((2, Op.INDEX_L, [v0[0]], _COST))
            elif o1 == Op.GET_FIELD:
                found.append((2, Op.LOAD_LOCAL_FIELD, [v0[0], v1[0]], _COST))
            elif o1 == Op.RETURN:
                found.append((2, Op.RETURN_LOCAL, [v0[0]], _COST))
            elif o1 in _FUSED_INDEX:
                found.append((2, 0xB0 + _FUSED_INDEX[o1], [v0[0]], _COST))
            elif o1 == Op.LOAD_LOCAL:
                # slightly dearer than LOAD_LOCAL + <op>_L, which also saves the push
                found.append((2, Op.LOAD_LOCAL2, [v0[0], v1[0]], _COST + 1))
        elif o0 in _PUSH_INT and o1 in _FUSED_INDEX:
            found.append((2, 0xA0 + _FUSED_INDEX[o1], [v0[0]], _COST))
        elif o0 == Op.FOR_ITER and o1 == Op.STORE_LOCAL:
            found.append((2, Op.FOR_ITER_STORE, [v0[0], v1[0], e0 + v0[1]], _COST))
        elif o0 == Op.CAST and o1 == Op.RETURN:
            found.append((2, Op.CAST_RETURN, [v0[0]], _COST))
        elif o0 == Op.ADD and o1 == Op.STORE_LOCAL:
            found.append((2, Op.ADD_STORE_LOCAL, [v1[0]], _COST))
        elif o0 == Op.NATIVE and o1 == Op.POP:
            found.append((2, Op.NATIVE_POP, [v0[0], v0[1]], _COST))
    return found


def _encode_insn(op: int, vals: List[int]) -> bytes:
    out = bytearray([op])
    for kind, val in zip(OP_OPERANDS.get(op, ''), vals):
        if kind == 'b':
            out.append(val)
        elif kind == 'c':
            out += _struct.pack('<b', val)
        elif kind == 'h':
            out += _struct.pack('<H', val)
        else:
            out += _struct.pack('<h', val)
    return bytes(out)


def _remap_live_ranges(new_pc: Dict[int, int], records: List[list]):
    """Debug info after fuse_superinstructions(): move the [slot, name, start, end]
    live ranges to the new pcs. A range that ends inside a superinstruction lasts
    until that instruction ends (`y` in `LOAD_LOCAL y; RETURN` -> RETURN_LOCAL y)."""
    import bisect
    olds = sorted(new_pc)
    first_of: Dict[int, int] = {}            # new pc -> first old instruction that went there
    for old in olds:
        first_of.setdefault(new_pc[old], old)

    def moved(pc: int, is_end: bool) -> int:
        k = bisect.bisect_left(olds, pc)
        if k >= len(olds):
            return new_pc[olds[-1]]
        at = new_pc[olds[k]]
        if is_end and olds[k] == pc and first_of[at] != pc:
            while k < len(olds) and new_pc[olds[k]] == at:
                k += 1
            return new_pc[olds[k]] if k < len(olds) else at
        return at
    for rec in records:
        rec[2], rec[3] = moved(rec[2], False), moved(rec[3], True)


def fuse_superinstructions(func: 'TBCFunction') -> Optional[Dict[int, int]]:
    """Rewrite one function's code with superinstructions (in place: code and
    line table). Returns the map old pc -> new pc (function-relative), or None
    when the code was left unchanged."""
    insns = _decode_insns(func.code)
    if not insns:
        return None
    end = len(func.code)
    targets = {ins[3] + ins[2][-1] for ins in insns if ins[1] in _JUMP_OPS}
    if any(t < 0 or t > end for t in targets):
        return None
    boundary = targets | {pc for pc, _line in func.lines} | {0}
    n = len(insns)
    best = [0] * (n + 1)
    choice: List[Any] = [None] * n
    for i in range(n - 1, -1, -1):
        best[i] = _COST + best[i + 1]
        if insns[i][1] == Op.JUMP and insns[i][2][0] == 0:
            best[i] = best[i + 1]               # a jump to the next instruction
            choice[i] = 'drop'
        for length, op, vals, cost in _fusions_at(insns, i, boundary):
            if cost + best[i + length] < best[i]:
                best[i] = cost + best[i + length]
                choice[i] = (length, op, vals)
    # layout: new pc of every old instruction start (and of the end)
    new_pc: Dict[int, int] = {}
    plan = []                                   # (op, operands with absolute old targets)
    i = pc = 0
    while i < n:
        c = choice[i]
        if c == 'drop':
            new_pc[insns[i][0]] = pc
            i += 1
            continue
        if c is None:
            length, op, vals = 1, insns[i][1], list(insns[i][2])
            if op in _JUMP_OPS:
                vals[-1] = insns[i][3] + vals[-1]
        else:
            length, op, vals = c
        for j in range(i, i + length):
            new_pc[insns[j][0]] = pc
        plan.append((pc, op, vals))
        pc += 1 + sum(_OPERAND_SIZE[k] for k in OP_OPERANDS.get(op, ''))
        i += length
    new_pc[end] = pc
    code = bytearray()
    for at, op, vals in plan:
        if op in _JUMP_OPS:
            vals = list(vals)
            size = 1 + sum(_OPERAND_SIZE[k] for k in OP_OPERANDS[op])
            rel = new_pc[vals[-1]] - (at + size)
            if not -32768 <= rel <= 32767:
                raise CompileError("function too large: jump offset exceeds 32 KB")
            vals[-1] = rel
        code += _encode_insn(op, vals)
    lines: List[Tuple[int, int]] = []
    for old, line in func.lines:
        at = new_pc[old]
        if lines and lines[-1][0] == at:
            lines[-1] = (at, line)              # a dropped JUMP shared its line entry
        else:
            lines.append((at, line))
    func.code = code
    func.lines = lines
    return new_pc


# ═══════════════════════════════════════════════════════════════════════════════
#                         BYTECODE COMPILER (AST -> .tbc)
# ═══════════════════════════════════════════════════════════════════════════════

class _Local:
    __slots__ = ('slot', 'mutable', 'celled', 'name', 'dbg')

    def __init__(self, name, slot, mutable, celled):
        self.name, self.slot, self.mutable, self.celled = name, slot, mutable, celled
        self.dbg = None           # its debug-info record [slot, name, start, end] (debug_info=True)


# ── operand stack depth while compiling ─────────────────────────────────────
# The compiler follows how many values the code it emits leaves on the operand
# stack (above the locals). `break`/`continue` written inside an expression
# (`total += handle ... continue ...`, `x = 1 + 2 * retry(2) ... break ...`) pop
# the values that expression had pushed, so a loop never leaks stack slots.
# Net stack effect per instruction; the ones that depend on an operand are in
# _OPERAND_EFFECT. END_HANDLE is set by the compiler (it drops its handle's pairs).
_STACK_EFFECT = {
    Op.NOP: 0, Op.PUSH_NIL: 1, Op.PUSH_TRUE: 1, Op.PUSH_FALSE: 1, Op.PUSH_I8: 1,
    Op.PUSH_I16: 1, Op.PUSH_CONST: 1, Op.POP: -1, Op.DUP: 1, Op.DUP2: 2, Op.SWAP: 0,
    Op.LOAD_LOCAL: 1, Op.STORE_LOCAL: -1, Op.LOAD_GLOBAL: 1, Op.STORE_GLOBAL: -1,
    Op.LOAD_UPVAL: 1, Op.STORE_UPVAL: -1, Op.LOAD_CELL: 1, Op.STORE_CELL: -1,
    Op.MAKE_CELL: 0, Op.LOAD_UPVAL_RAW: 1, Op.ARG_DEFAULT: 0,
    Op.ADD: -1, Op.SUB: -1, Op.MUL: -1, Op.DIV: -1, Op.MOD: -1, Op.POW: -1,
    Op.NEG: 0, Op.NOT: 0, Op.BAND: -1, Op.BOR: -1, Op.BXOR: -1, Op.BNOT: 0,
    Op.SHL: -1, Op.SHR: -1,
    Op.EQ: -1, Op.NE: -1, Op.LT: -1, Op.LE: -1, Op.GT: -1, Op.GE: -1,
    Op.JUMP: 0, Op.JUMP_IF_FALSE: -1, Op.JUMP_IF_TRUE: -1,
    Op.RETURN: -1, Op.RETURN_NIL: 0,
    Op.GET_INDEX: -1, Op.SET_INDEX: -3, Op.GET_FIELD: 0, Op.SET_FIELD: -2, Op.GET_SLOT: 0,
    Op.PUSH_FUNC: 1, Op.FOR_ITER: 1, Op.MATCH_TYPE: 0, Op.CAST: 0, Op.IS_ARRAY_LEN: 0,
    Op.TRY: 0, Op.TRY_TIMEOUT: -1, Op.END_TRY: 0, Op.PANIC: -1,
    Op.UNWRAP_OR_RETURN: 0, Op.UNWRAP_OR_PANIC: 0,
    Op.HANDLE: 0, Op.END_HANDLE: 0, Op.RESUME: 0, Op.HALT: 0,
    # superinstructions (emitted only by fuse_superinstructions, after compiling):
    # the net effect of the sequence each one replaces
    Op.LOAD_LOCAL2: 2, Op.LOAD_LOCAL_FIELD: 1, Op.INC_LOCAL: 0, Op.CAST_LOCAL: 0,
    Op.CAST_RETURN: -1, Op.ADD_STORE_LOCAL: -2, Op.INDEX_L: 0, Op.INDEX_LL: 1,
    Op.RETURN_LOCAL: 0, Op.FOR_ITER_STORE: 0, Op.JUMP_IF_NOT: -2, Op.JUMP_IF_NOT_L: -1,
    Op.JUMP_IF_NOT_I: -1, Op.JUMP_IF_NOT_LL: 0, Op.JUMP_IF_NOT_LI: 0,
}
_STACK_EFFECT.update({0xA0 + _i: 0 for _i in range(16)})     # <op>_I k
_STACK_EFFECT.update({0xB0 + _i: 0 for _i in range(16)})     # <op>_L b
# (operand index, factor, constant): effect = constant - factor * operands[index]
_OPERAND_EFFECT = {
    Op.CALL: (1, 1, 1), Op.CALL_VALUE: (0, 1, 0), Op.INVOKE: (2, 1, 0), Op.NATIVE: (1, 1, 1),
    Op.NEW_ARRAY: (0, 1, 1), Op.NEW_MAP: (0, 2, 1), Op.NEW_STRUCT: (1, 1, 1),
    Op.MAKE_CLOSURE: (1, 1, 1), Op.CONCAT: (0, 1, 1), Op.PERFORM: (2, 1, 1),
    Op.NATIVE_POP: (1, 1, 0),
}
# depth where a jump lands, relative to the depth before the instruction
# (TRY: the error message is pushed; HANDLE: an abort cuts the stack back to the
# clause pairs and pushes the clause value - minus 2n, done in track_depth)
_JUMP_LANDING = {
    Op.JUMP: 0, Op.JUMP_IF_FALSE: -1, Op.JUMP_IF_TRUE: -1, Op.TRY: 1, Op.TRY_TIMEOUT: 0,
    Op.FOR_ITER: 0, Op.ARG_DEFAULT: 0, Op.HANDLE: 1,
    Op.FOR_ITER_STORE: 0, Op.JUMP_IF_NOT: -2, Op.JUMP_IF_NOT_L: -1, Op.JUMP_IF_NOT_I: -1,
    Op.JUMP_IF_NOT_LL: 0, Op.JUMP_IF_NOT_LI: 0,
}
# the next instruction is only reached through a jump
_ENDS_FLOW = frozenset((Op.JUMP, Op.RETURN, Op.RETURN_NIL, Op.PANIC, Op.RESUME, Op.HALT,
                        Op.CAST_RETURN, Op.RETURN_LOCAL))


def code_depths(code: bytes, start: int, until: int) -> Dict[int, tuple]:
    """Operand stack depths for checking a portable snapshot's call frames,
    recovery points and handler records against the bytecode (docs/TIL_SNAPSHOTS.md
    3.4). One linear pass from a function's first instruction `start` (depth 0:
    nothing above the locals) to `until`, with the effects of _STACK_EFFECT, that
    follows forward jumps (the compiler places loop heads before their back-edges,
    so every reachable instruction gets its depth); the depth after END_HANDLE and
    after an instruction that ends the flow comes from a jump landing there, or is
    unknown. Returns {pc: (depth or None, pc of the previous instruction)} for the
    instruction starts in [start, until]. vm/tilvm.c (tp_depth) computes the same."""
    out: Dict[int, tuple] = {}
    pending: Dict[int, int] = {}            # forward jump target -> depth (the first one wins)
    depth: Optional[int] = 0
    pos, prev = start, -1
    while pos <= until and pos < len(code):
        if depth is None:
            depth = pending.get(pos)
        out[pos] = (depth, prev)
        op = code[pos]
        fmt = OP_OPERANDS.get(op, '')
        size = 1 + sum(_OPERAND_SIZE[k] for k in fmt)
        if op not in _STACK_EFFECT and op not in _OPERAND_EFFECT or pos + size > len(code):
            break
        if depth is not None:
            if op in _OPERAND_EFFECT:
                idx, factor, const = _OPERAND_EFFECT[op]
                at = pos + 1 + sum(_OPERAND_SIZE[k] for k in fmt[:idx])
                n = code[at] | (code[at + 1] << 8 if fmt[idx] == 'h' else 0)
                after = depth + const - factor * n
            else:
                after = depth + _STACK_EFFECT[op]
            if op in _JUMP_LANDING:
                rel = code[pos + size - 2] | (code[pos + size - 1] << 8)
                target = pos + size + (rel - 0x10000 if rel >= 0x8000 else rel)
                landing = depth + _JUMP_LANDING[op] - (2 * code[pos + 1] if op == Op.HANDLE else 0)
                if target > pos and landing >= 0:
                    pending.setdefault(target, landing)
            depth = None if op in _ENDS_FLOW or op == Op.END_HANDLE or after < 0 else after
        prev = pos
        pos += size
    return out


class _FoldEnv:
    """The constants try_fold may use: the inline constants that are not
    shadowed by a local of the function being compiled (const_eval only tests
    `name in env` and reads env[name])."""
    __slots__ = ('compiler',)

    def __init__(self, compiler: 'BytecodeCompiler'):
        self.compiler = compiler

    def __contains__(self, name) -> bool:
        return self.compiler.names_a_constant(name)

    def __getitem__(self, name):
        return self.compiler.inline_consts[name]


class _FuncState:
    def __init__(self, proto: TBCFunction, parent: Optional['_FuncState'], fdef=None):
        self.proto = proto
        self.parent = parent
        self.fdef = fdef
        self.scopes: List[Dict[str, _Local]] = [{}]
        self.next_slot = 0
        self.upvals: List[Tuple[str, bool, int, bool]] = []   # (name, from_parent_local, index, mutable)
        self.loops: List[dict] = []
        self.try_depth = 0
        self.captured: Set[str] = set()
        self.last_line = -1
        self.result_slot: Optional[int] = None   # `result` inside #[ensures]
        self.is_clause = False    # an effect clause (resume allowed, return not)
        self.handle_depth = 0     # enclosing `handle` bodies in this function
        # operand stack depth (see _STACK_EFFECT): values above the locals at the
        # current point; `dead` = the current point is only reachable by a jump
        self.depth = 0
        self.dead = False
        self.depth_ok = True                      # False: lost track (never expected)
        self.jump_depths: Dict[int, Optional[int]] = {}   # patch pos -> landing depth
        self.handle_sps: List[int] = []          # depth of the pairs of each enclosing handle


class BytecodeCompiler:
    """Compiles a (resolved) TIL Program into a TBCModule."""

    def __init__(self, source: str = "", filename: str = "<stdin>", fuse: bool = True,
                 debug_info: bool = False):
        self.reporter = ErrorReporter(source, filename)
        self.filename = filename
        self.m = TBCModule()
        self.fuse = fuse          # peephole pass: superinstructions (til ... --no-fuse turns it off)
        # record variable names and live ranges (the module's debug-info section:
        # til build --debug-info, til debug); off by default: the bytecode is unchanged
        self.debug_info = debug_info

    # ── errors ──────────────────────────────────────────────────────────────
    def error(self, node, msg: str, hint: str = ""):
        line = getattr(node, 'line', 0) or 0
        col = getattr(node, 'col', 0) or 1
        raise CompileError(self.reporter.format_error(line, col, msg, "E-VM", hint))

    # ── emission helpers ────────────────────────────────────────────────────
    def emit(self, op: int, *operands, node=None):
        fs = self.fs
        line = getattr(node, 'line', 0) if node is not None else 0
        if line and line != fs.last_line:
            fs.proto.lines.append((len(fs.proto.code), line))
            fs.last_line = line
        code = fs.proto.code
        code.append(op)
        fmt = OP_OPERANDS.get(op, '')
        for kind, val in zip(fmt, operands):
            if kind == 'b':
                if not 0 <= val <= 0xFF:
                    self.error(node, f"operand out of range ({val}); too many locals/arguments?")
                code.append(val)
            elif kind == 'c':
                code += _struct.pack('<b', val)
            elif kind == 'h':
                if not 0 <= val <= 0xFFFF:
                    self.error(node, "program too large for the bytecode format")
                code += _struct.pack('<H', val)
            elif kind == 's':
                code += _struct.pack('<h', val)
        self.track_depth(op, operands)
        return len(code)

    def track_depth(self, op: int, operands):
        """Follow the operand stack depth over one emitted instruction (see
        _STACK_EFFECT). Jumps remember the depth where they land."""
        fs = self.fs
        effect = _STACK_EFFECT.get(op)
        if effect is None:
            spec = _OPERAND_EFFECT.get(op)
            if spec is None or len(operands) <= spec[0]:
                fs.depth_ok = False      # an instruction this table does not know
                return
            effect = spec[2] - spec[1] * operands[spec[0]]
        before = fs.depth
        fs.depth = before + effect
        landing = _JUMP_LANDING.get(op)
        if landing is not None:
            if op == Op.HANDLE:
                landing -= 2 * operands[0]
            fs.jump_depths[len(fs.proto.code) - 2] = None if fs.dead else before + landing
        if op in _ENDS_FLOW:
            fs.dead = True

    def land_jump(self, at: int):
        """A forward jump (patch position `at`) lands at the current position."""
        fs = self.fs
        landing = fs.jump_depths.pop(at, None)
        if landing is None:
            return                       # the jump itself is unreachable
        if fs.dead:
            fs.depth, fs.dead = landing, False
        elif landing != fs.depth:
            fs.depth_ok = False          # paths disagree (a compiler bug)

    def emit_jump(self, op: int, node=None) -> int:
        """Emit a forward jump (single i16 operand); returns the patch position."""
        self.emit(op, 0, node=node)
        return len(self.fs.proto.code) - 2

    def patch_jump(self, at: int, target: Optional[int] = None):
        code = self.fs.proto.code
        if target is None:
            self.land_jump(at)
        else:
            self.fs.jump_depths.pop(at, None)     # backward jump: the target is known
        target = len(code) if target is None else target
        rel = target - (at + 2)
        if not -32768 <= rel <= 32767:
            raise CompileError("function too large: jump offset exceeds 32 KB")
        code[at:at + 2] = _struct.pack('<h', rel)

    def emit_loop(self, target: int, node=None):
        self.emit(Op.JUMP, 0, node=node)
        self.patch_jump(len(self.fs.proto.code) - 2, target)

    def here(self) -> int:
        return len(self.fs.proto.code)

    def push_int(self, v: int, node=None):
        v = wrap_int64(v)
        if -128 <= v <= 127:
            self.emit(Op.PUSH_I8, v, node=node)
        elif -32768 <= v <= 32767:
            self.emit(Op.PUSH_I16, v, node=node)
        else:
            self.emit(Op.PUSH_CONST, self.m.const(v), node=node)

    def push_const(self, value, node=None):
        if isinstance(value, bool):
            self.emit(Op.PUSH_TRUE if value else Op.PUSH_FALSE, node=node)
        elif value is None:
            self.emit(Op.PUSH_NIL, node=node)
        elif isinstance(value, int):
            self.push_int(value, node)
        else:
            self.emit(Op.PUSH_CONST, self.m.const(value), node=node)

    # ── scopes ──────────────────────────────────────────────────────────────
    def begin_scope(self):
        self.fs.scopes.append({})

    def end_scope(self):
        scope = self.fs.scopes.pop()
        # slots are not reused, which keeps closures/cells simple and safe
        if self.debug_info:
            end = len(self.fs.proto.code)
            for local in scope.values():
                if local.dbg is not None and local.dbg[3] is None:
                    local.dbg[3] = end          # the variable's scope ends here

    def declare_local(self, name: str, mutable: bool = True, node=None) -> _Local:
        fs = self.fs
        slot = fs.next_slot
        fs.next_slot += 1
        if fs.next_slot > 250:
            self.error(node, "too many local variables in one function (max 250)")
        fs.proto.n_locals = max(fs.proto.n_locals, fs.next_slot)
        local = _Local(name, slot, mutable, name in fs.captured)
        if self.debug_info and not name.startswith(' '):      # not a compiler temporary
            here = len(fs.proto.code)
            old = fs.scopes[-1].get(name)
            if old is not None and old.dbg is not None and old.dbg[3] is None:
                old.dbg[3] = here               # `let x` again in the same scope hides it
            local.dbg = [slot, name, here, None]   # parameters: live from the start
            fs.proto.locals.append(local.dbg)
        fs.scopes[-1][name] = local
        return local

    def finish_debug_info(self, fs: '_FuncState'):
        """End of a function: close the ranges of its remaining locals, name its upvalues."""
        if self.debug_info:
            end = len(fs.proto.code)
            for rec in fs.proto.locals:
                if rec[3] is None:
                    rec[3] = end
            fs.proto.upval_names = [u[0] for u in fs.upvals]

    def hidden_local(self) -> int:
        return self.declare_local(f" tmp{self.fs.next_slot}").slot

    def find_local(self, fs: _FuncState, name: str) -> Optional[_Local]:
        for scope in reversed(fs.scopes):
            if name in scope:
                return scope[name]
        return None

    def resolve_upval(self, fs: _FuncState, name: str) -> Optional[int]:
        if fs.parent is None:
            return None
        for i, (n, _from_local, _idx, _mut) in enumerate(fs.upvals):
            if n == name:
                return i
        local = self.find_local(fs.parent, name)
        if local is not None:
            if not local.celled:
                return None   # cannot happen: capture analysis cells captured names
            fs.upvals.append((name, True, local.slot, local.mutable))
            return len(fs.upvals) - 1
        idx = self.resolve_upval(fs.parent, name)
        if idx is None:
            return None
        fs.upvals.append((name, False, idx, fs.parent.upvals[idx][3]))
        return len(fs.upvals) - 1

    def store_new_local(self, local: _Local, node=None):
        """Value on the stack -> freshly declared local (celled if captured)."""
        self.emit(Op.STORE_LOCAL, local.slot, node=node)
        if local.celled:
            self.emit(Op.MAKE_CELL, local.slot, node=node)
        if local.dbg is not None:
            local.dbg[2] = len(self.fs.proto.code)     # live once it holds its value

    # ── program ─────────────────────────────────────────────────────────────
    def compile_program(self, program: Program) -> TBCModule:
        m = self.m
        m.energy = program.energy or ENERGY_PERFORMANCE   # the #[energy: ...] pragma
        self.funcs: Dict[str, int] = {}          # function name -> index
        self.fdefs: Dict[str, FuncDef] = {}
        self.struct_types: Dict[str, int] = {}   # struct name -> type index
        self.struct_fields: Dict[str, List[StructField]] = {}
        self.enum_values: Dict[str, Dict[str, int]] = {}   # C-like enums
        self.adt_variants: Dict[str, Dict[str, Tuple[int, int]]] = {}  # enum -> variant -> (type, arity)
        self.global_slots: Dict[str, int] = {}
        self.global_mutable: Dict[str, bool] = {}
        self.inline_consts: Dict[str, Any] = {}
        self.module_names = set(getattr(program, 'module_names', set()) or ())
        self.user_method_names: Set[str] = set()
        self.pending: List[Tuple[int, Any, Optional[str]]] = []   # (func idx, FuncDef/lambda, owner type)
        # effect -> clause arity of the handles that lexically enclose the code being
        # compiled (same function): performs they handle are checked at compile time
        self.static_effects: List[Dict[str, int]] = []

        traits: Dict[str, TraitDef] = {}
        impls: List[ImplBlock] = []
        globals_: List[VarDecl] = []
        for stmt in program.statements:
            if isinstance(stmt, StructDef):
                self.struct_types[stmt.name] = len(m.types)
                self.struct_fields[stmt.name] = stmt.fields
                m.types.append((stmt.name, [f.name for f in stmt.fields]))
            elif isinstance(stmt, EnumDef):
                self.declare_enum(stmt)
            elif isinstance(stmt, TraitDef):
                traits[stmt.name] = stmt
            elif isinstance(stmt, ImplBlock):
                impls.append(stmt)
            elif isinstance(stmt, VarDecl):
                globals_.append(stmt)

        # functions
        for stmt in program.statements:
            if isinstance(stmt, FuncDef):
                if stmt.name in self.funcs:
                    self.error(stmt, f"function '{stmt.name}' is defined twice")
                self.add_function(stmt.name, stmt, None)
        # methods (+ trait default methods)
        for impl in impls:
            if impl.type_name not in self.struct_types:
                self.error(impl, f"impl for unknown struct '{impl.type_name}'")
            tidx = self.struct_types[impl.type_name]
            defined = set()
            for meth in impl.methods:
                fidx = self.add_function(f"{impl.type_name}.{meth.name}", meth, impl.type_name)
                defined.add(meth.name)
                self.user_method_names.add(meth.name)
                if any(p.name == "self" for p in meth.params):
                    m.methods.append((tidx, m.const(meth.name), fidx))
            if impl.trait_name and impl.trait_name in traits:
                for tm in traits[impl.trait_name].methods:
                    if tm.name in defined:
                        continue
                    if not (isinstance(tm.body, Block) and tm.body.statements):
                        self.error(impl, f"impl {impl.trait_name} for {impl.type_name} is missing "
                                         f"method '{tm.name}'")
                    fidx = self.add_function(f"{impl.type_name}.{tm.name}", tm, impl.type_name)
                    self.user_method_names.add(tm.name)
                    if any(p.name == "self" for p in tm.params):
                        m.methods.append((tidx, m.const(tm.name), fidx))

        # globals: literal constants are inlined, everything else gets a slot
        for g in globals_:
            if not g.mutable and isinstance(g.value, (IntLit, FloatLit, StringLit, BoolLit, CharLit)):
                self.inline_consts[g.name] = ord(g.value.value) if isinstance(g.value, CharLit) \
                    else g.value.value
                continue
            self.global_slots[g.name] = len(m.globals)
            self.global_mutable[g.name] = g.mutable
            m.globals.append(g.name)

        # entry point: initialise globals, then call main
        start = TBCFunction("__start__")
        m.entry = len(m.funcs)
        m.funcs.append(start)
        self.fs = _FuncState(start, None)
        for g in globals_:
            if g.name in self.global_slots and not g.deferred:
                self.compile_expr(g.value, g) if g.value is not None else self.emit(Op.PUSH_NIL)
                self.emit(Op.STORE_GLOBAL, self.global_slots[g.name], node=g)
        if "main" in self.funcs:
            self.emit(Op.CALL, self.funcs["main"], 0)
            self.emit(Op.POP)
        self.emit(Op.HALT)
        start_fs = self.fs

        # compile bodies (lambdas and prelude functions are appended as we go)
        i = 0
        while i < len(self.pending):
            fidx, fdef, owner = self.pending[i]
            self.compile_function(fidx, fdef, owner)
            i += 1
        self.finish_debug_info(start_fs)
        m.debug_info = self.debug_info
        if self.fuse:
            for f in m.funcs:
                new_pc = fuse_superinstructions(f)
                if new_pc and f.locals:          # the shorter code moved the live ranges
                    _remap_live_ranges(new_pc, f.locals)
        return m

    def declare_enum(self, e: EnumDef):
        if any(v.fields for v in e.variants):
            # Algebraic data type: every variant is a (possibly empty) record
            variants = {}
            for v in e.variants:
                tidx = len(self.m.types)
                self.m.types.append((f"{e.name}.{v.name}", [str(i) for i in range(len(v.fields))]))
                variants[v.name] = (tidx, len(v.fields))
            self.adt_variants[e.name] = variants
        else:
            values, nxt = {}, 0
            for v in e.variants:
                if v.value is not None:
                    try:
                        nxt = const_eval(v.value, {})
                    except NotConstant:
                        self.error(v.value if isinstance(v.value, ASTNode) else e,
                                   "enum values must be constants")
                values[v.name] = nxt
                nxt += 1
            self.enum_values[e.name] = values

    def add_function(self, name: str, fdef, owner: Optional[str]) -> int:
        proto = TBCFunction(name)
        idx = len(self.m.funcs)
        self.m.funcs.append(proto)
        self.funcs[name] = idx
        self.fdefs[name] = fdef
        self.pending.append((idx, fdef, owner))
        return idx

    def prelude_function(self, method: str) -> int:
        mid = METHOD_ID[method]
        if mid in self.m.prelude:
            return self.m.prelude[mid]
        fdef = parse_source(PRELUDE_SOURCE[method], "<prelude>").statements[0]
        idx = self.add_function(fdef.name, fdef, None)
        self.m.prelude[mid] = idx
        return idx

    # ── capture analysis ────────────────────────────────────────────────────
    @staticmethod
    def captured_names(body) -> Set[str]:
        """Names referenced inside lambdas and effect clauses (they must live in cells)."""
        names: Set[str] = set()
        for node in walk(body):
            if isinstance(node, (Lambda, EffectClause)):
                params = set(node.params) if isinstance(node, EffectClause) else {p for p, _ in node.params}
                for inner in walk(node.body):
                    if isinstance(inner, Identifier) and inner.name not in params:
                        names.add(inner.name)
        return names

    # ── functions ───────────────────────────────────────────────────────────
    def compile_function(self, fidx: int, fdef, owner: Optional[str]):
        proto = self.m.funcs[fidx]
        parent_fs = getattr(self, 'fs', None)
        fs = _FuncState(proto, None, fdef)
        self.fs = fs
        fs.captured = self.captured_names(fdef.body)
        for c in fdef.requires + fdef.ensures:
            fs.captured |= self.captured_names(c)
        params = fdef.params
        proto.n_params = len(params)
        proto.n_required = 0
        seen_default = False
        for i, p in enumerate(params):
            if p.default is None and not seen_default:
                proto.n_required = i + 1
            elif p.default is not None:
                seen_default = True
        if fdef.name == "main" and params:
            self.error(fdef, "main() takes no parameters")
        if params and params[0].name == "self":
            proto.flags |= TBCFunction.FLAG_METHOD
        if isinstance(fdef.ret_type, (OptionType, ResultType)):
            proto.flags |= TBCFunction.FLAG_PROPAGATE
        if "energy:low" in getattr(fdef, 'attributes', ()):
            proto.flags |= TBCFunction.FLAG_ENERGY_LOW
        for p in params:
            self.declare_local(p.name, mutable=True, node=fdef)
        # default arguments are evaluated by the callee when the slot is unset
        for i, p in enumerate(params):
            if p.default is not None:
                at = self.emit(Op.ARG_DEFAULT, i, 0, node=fdef) - 2
                self.compile_expr(p.default, fdef)
                self.emit(Op.STORE_LOCAL, i)
                self.patch_jump(at)
        # annotated numeric parameters are converted like C does (f(x: float) with f(3))
        for i, p in enumerate(params):
            cast = self.numeric_cast_for(p.type)
            if cast is not None:
                self.emit(Op.LOAD_LOCAL, i)
                self.emit(Op.CAST, cast)
                self.emit(Op.STORE_LOCAL, i)
        for p in params:
            local = self.find_local(fs, p.name)
            if local.celled:
                self.emit(Op.MAKE_CELL, local.slot)
        # contracts: preconditions at entry
        for cond, text in zip(fdef.requires, fdef.requires_text or [""] * len(fdef.requires)):
            self.compile_expr(cond, fdef)
            skip = self.emit_jump(Op.JUMP_IF_TRUE, node=cond)
            self.push_const(f"contract violation: {fdef.name} requires {text}".strip(), cond)
            self.emit(Op.PANIC, node=cond)
            self.patch_jump(skip)
        self.compile_block_stmts(fdef.body)
        self.emit_return_nil(fdef)
        self.finish_debug_info(fs)
        self.fs = parent_fs

    def emit_return_nil(self, node):
        fdef = self.fs.fdef
        if isinstance(fdef, FuncDef) and fdef.ensures and not isinstance(fdef.ret_type, VoidType):
            self.emit(Op.PUSH_NIL)
            self.emit_return_value(node)
        else:
            self.emit(Op.RETURN_NIL, node=node)

    def emit_return_value(self, node):
        """Return the value on the stack, checking #[ensures] first."""
        fs = self.fs
        fdef = fs.fdef
        if isinstance(fdef, FuncDef):
            cast = self.numeric_cast_for(fdef.ret_type)
            if cast is not None:
                self.emit(Op.CAST, cast, node=node)   # `-> float` returning an int, etc.
        if isinstance(fdef, FuncDef) and fdef.ensures:
            if fs.result_slot is None:
                fs.result_slot = self.hidden_local()
            self.emit(Op.STORE_LOCAL, fs.result_slot, node=node)
            fs.ensures_active = True
            for cond, text in zip(fdef.ensures, fdef.ensures_text or [""] * len(fdef.ensures)):
                self.compile_expr(cond, fdef)
                ok = self.emit_jump(Op.JUMP_IF_TRUE, node=cond)
                self.push_const(f"contract violation: {fdef.name} ensures {text}".strip(), cond)
                self.emit(Op.PANIC, node=cond)
                self.patch_jump(ok)
            fs.ensures_active = False
            self.emit(Op.LOAD_LOCAL, fs.result_slot)
        self.emit(Op.RETURN, node=node)

    def compile_lambda(self, node: Lambda) -> None:
        proto = TBCFunction(f"<lambda:{node.line}>")
        proto.flags |= TBCFunction.FLAG_LAMBDA
        fidx = len(self.m.funcs)
        self.m.funcs.append(proto)
        parent = self.fs
        fs = _FuncState(proto, parent, node)
        fs.captured = self.captured_names(node.body)
        self.fs = fs
        saved_effects, self.static_effects = self.static_effects, []   # called later: dynamic
        proto.n_params = proto.n_required = len(node.params)
        for pname, _ptype in node.params:
            self.declare_local(pname, node=node)
        for pname, _ptype in node.params:
            local = self.find_local(fs, pname)
            if local.celled:
                self.emit(Op.MAKE_CELL, local.slot)
        if isinstance(node.body, Block):
            self.compile_block_value(node.body)
        else:
            self.compile_expr(node.body, node)
        self.emit(Op.RETURN, node=node)
        self.finish_debug_info(fs)
        self.static_effects = saved_effects
        proto.n_upvals = len(fs.upvals)
        upvals = fs.upvals
        self.fs = parent
        if not upvals:
            self.emit(Op.PUSH_FUNC, fidx, node=node)
            return
        for _name, from_local, index, _mut in upvals:
            if from_local:
                self.emit(Op.LOAD_LOCAL, index, node=node)       # pushes the cell itself
            else:
                self.emit(Op.LOAD_UPVAL_RAW, index, node=node)
        self.emit(Op.MAKE_CLOSURE, fidx, len(upvals), node=node)


    # ── blocks & statements ─────────────────────────────────────────────────
    def compile_block_stmts(self, body):
        stmts = body.statements if isinstance(body, Block) else [body]
        self.begin_scope()
        for s in stmts:
            self.compile_stmt(s)
        self.end_scope()

    def compile_block_value(self, body):
        """A block used as an expression evaluates to its last expression."""
        stmts = body.statements if isinstance(body, Block) else [body]
        self.begin_scope()
        if not stmts:
            self.emit(Op.PUSH_NIL)
        for s in stmts[:-1]:
            self.compile_stmt(s)
        if stmts:
            last = stmts[-1]
            if isinstance(last, ExprStmt):
                self.compile_expr(last.expr, last)
            elif isinstance(last, MatchExpr):
                self.compile_match(last, True)
            elif isinstance(last, If) and last.else_body is not None:
                self.compile_if(last, want_value=True)
            elif isinstance(last, (Block, IntentBlock)):
                self.compile_expr(last, last)
            else:
                self.compile_stmt(last)
                self.emit(Op.PUSH_NIL)
        self.end_scope()

    def compile_stmt(self, node):
        t = type(node)
        if t is ExprStmt:
            self.compile_expr(node.expr, node)
            self.emit(Op.POP)
        elif t is VarDecl:
            self.compile_var_decl(node)
        elif t is Assignment:
            self.compile_assignment(node)
        elif t is If:
            self.compile_if(node, want_value=False)
        elif t is While:
            self.compile_while(node)
        elif t is Loop:
            self.compile_while(node, infinite=True)
        elif t is For:
            self.compile_for(node)
        elif t is Break or t is Continue:
            self.compile_break_continue(node, t is Break)
        elif t is Return:
            if self.fs.is_clause:
                self.error(node, "'return' is not allowed in an effect clause")
            # (handler records of this frame are dropped by RETURN itself)
            for _ in range(self.fs.try_depth):
                self.emit(Op.END_TRY)
            if node.value is not None:
                self.compile_expr(node.value, node)
                self.emit_return_value(node)
            else:
                self.emit_return_nil(node)
        elif t is MatchExpr:
            self.compile_match(node, False)
        elif t is AssertStmt or t is InvariantStmt:
            self.compile_expr(node.condition, node)
            ok = self.emit_jump(Op.JUMP_IF_TRUE, node)
            kind = "Assertion failed" if t is AssertStmt else "Invariant violated"
            self.push_const(f"{kind}: {node.message}" if node.message else kind, node)
            self.emit(Op.PANIC, node=node)
            self.patch_jump(ok)
        elif t is IntentBlock:
            self.compile_intent(node, False)
        elif t is GlobalInit:
            self.compile_expr(node.decl.value, node)
            cast = self.numeric_cast_for(node.decl.type_ann)
            if cast is not None:
                self.emit(Op.CAST, cast, node=node)
            self.emit(Op.STORE_GLOBAL, self.global_slots[node.decl.name], node=node)
        elif t is Block:
            self.compile_block_stmts(node)
        elif t in (FuncDef, StructDef, EnumDef, ImplBlock, TraitDef):
            self.error(node, "definitions are only allowed at the top level")
        elif t in (TypeAlias, Import, CapabilityType):
            pass
        else:
            self.compile_expr(node, node)
            self.emit(Op.POP)

    def numeric_cast_for(self, type_ann) -> Optional[int]:
        if isinstance(type_ann, PrimitiveType):
            if type_ann.name in ('float', 'f32', 'f64'):
                return CAST_KINDS['float']
            if type_ann.name in ('int', 'i8', 'i16', 'i32', 'i64', 'uint', 'u8', 'u16', 'u32', 'u64'):
                return CAST_KINDS['int']
        return None

    def compile_var_decl(self, node: VarDecl):
        if node.value is None:
            self.push_const(self.zero_value(node.type_ann), node)
        else:
            self.compile_expr(node.value, node)
            cast = self.numeric_cast_for(node.type_ann)
            if cast is not None and not isinstance(node.value, (IntLit if cast == 0 else FloatLit)):
                self.emit(Op.CAST, cast, node=node)
        local = self.declare_local(node.name, mutable=node.mutable, node=node)
        self.store_new_local(local, node)

    @staticmethod
    def zero_value(type_ann):
        if isinstance(type_ann, PrimitiveType):
            return {'float': 0.0, 'f32': 0.0, 'f64': 0.0, 'str': "", 'bool': False,
                    'char': 0}.get(type_ann.name, 0)
        return None

    def emit_load_var(self, name: str, node) -> bool:
        fs = self.fs
        if name == 'result' and getattr(fs, 'ensures_active', False):
            self.emit(Op.LOAD_LOCAL, fs.result_slot, node=node)
            return True
        local = self.find_local(fs, name)
        if local is not None:
            self.emit(Op.LOAD_CELL if local.celled else Op.LOAD_LOCAL, local.slot, node=node)
            return True
        up = self.resolve_upval(fs, name)
        if up is not None:
            self.emit(Op.LOAD_UPVAL, up, node=node)
            return True
        if name in self.inline_consts:
            self.push_const(self.inline_consts[name], node)
            return True
        if name in self.global_slots:
            self.emit(Op.LOAD_GLOBAL, self.global_slots[name], node=node)
            return True
        return False

    def emit_store_var(self, name: str, node):
        fs = self.fs
        local = self.find_local(fs, name)
        hint = "declare it with 'var' to make it mutable"
        if local is not None:
            if not local.mutable:
                self.error(node, f"cannot assign twice to immutable variable '{name}'", hint)
            self.emit(Op.STORE_CELL if local.celled else Op.STORE_LOCAL, local.slot, node=node)
            return
        up = self.resolve_upval(fs, name)
        if up is not None:
            if not fs.upvals[up][3]:
                self.error(node, f"cannot assign to immutable captured variable '{name}'", hint)
            self.emit(Op.STORE_UPVAL, up, node=node)
            return
        if name in self.global_slots:
            if not self.global_mutable[name]:
                self.error(node, f"cannot assign to immutable global '{name}'", hint)
            self.emit(Op.STORE_GLOBAL, self.global_slots[name], node=node)
            return
        if name in self.inline_consts:
            self.error(node, f"cannot assign to constant '{name}'")
        self.unknown_name(node, name)

    def unknown_name(self, node, name: str, what: str = "name"):
        import difflib
        known = set(self.funcs) | set(self.global_slots) | set(self.inline_consts) | set(NATIVES)
        for scope in self.fs.scopes:
            known |= {k for k in scope if not k.startswith(' ')}
        close = difflib.get_close_matches(name, [k for k in known if '.' not in k], n=1)
        if what == "function" and name in RESUME_WORDS:
            self.error(node, f"unknown function '{name}': resume(...) outside an effect clause",
                       "resume continues the body of a 'handle' and may only be used in one of "
                       "its 'with' clauses (not in a nested lambda)")
        self.error(node, f"unknown {what} '{name}'", f"did you mean '{close[0]}'?" if close else "")

    BINOPS = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL, '/': Op.DIV, '%': Op.MOD, '**': Op.POW,
              '&': Op.BAND, '|': Op.BOR, '^': Op.BXOR, '<<': Op.SHL, '>>': Op.SHR,
              '==': Op.EQ, '!=': Op.NE, '<': Op.LT, '<=': Op.LE, '>': Op.GT, '>=': Op.GE}

    def compile_assignment(self, node: Assignment):
        target, op = node.target, node.op
        binop = None if op == '=' else self.BINOPS[op[:-1]]
        if isinstance(target, Identifier):
            if binop is not None:
                if not self.emit_load_var(target.name, target):
                    self.unknown_name(target, target.name)
                self.compile_expr(node.value, node)
                self.emit(binop, node=node)
            else:
                self.compile_expr(node.value, node)
            self.emit_store_var(target.name, node)
        elif isinstance(target, Index):
            self.compile_expr(target.obj, target)
            self.compile_expr(target.index, target)
            if binop is not None:
                self.emit(Op.DUP2)
                self.emit(Op.GET_INDEX, node=target)
                self.compile_expr(node.value, node)
                self.emit(binop, node=node)
            else:
                self.compile_expr(node.value, node)
            self.emit(Op.SET_INDEX, node=node)
        elif isinstance(target, Attribute):
            self.compile_expr(target.obj, target)
            name = self.m.const(target.attr)
            if binop is not None:
                self.emit(Op.DUP)
                self.emit(Op.GET_FIELD, name, node=target)
                self.compile_expr(node.value, node)
                self.emit(binop, node=node)
            else:
                self.compile_expr(node.value, node)
            self.emit(Op.SET_FIELD, name, node=node)
        else:
            self.error(node, "invalid assignment target")

    def compile_if(self, node: If, want_value: bool):
        ends = []
        branches = [(node.condition, node.then_body)] + list(node.elifs)
        for cond, body in branches:
            self.compile_expr(cond, node)
            nxt = self.emit_jump(Op.JUMP_IF_FALSE, cond)
            if want_value:
                self.compile_block_value(body)
            else:
                self.compile_block_stmts(body)
            ends.append(self.emit_jump(Op.JUMP))
            self.patch_jump(nxt)
        if node.else_body is not None:
            if want_value:
                self.compile_block_value(node.else_body)
            else:
                self.compile_block_stmts(node.else_body)
        elif want_value:
            self.emit(Op.PUSH_NIL)
        for e in ends:
            self.patch_jump(e)

    def push_loop(self, continue_target: Optional[int]):
        ctx = {'breaks': [], 'continues': [], 'continue_target': continue_target,
               'try_depth': self.fs.try_depth, 'handle_depth': self.fs.handle_depth,
               'depth': self.fs.depth}
        self.fs.loops.append(ctx)
        return ctx

    def pop_loop(self, ctx):
        self.fs.loops.pop()
        for b in ctx['breaks']:
            self.patch_jump(b)

    def compile_while(self, node, infinite: bool = False):
        top = self.here()
        exit_jump = None
        if not infinite:
            self.compile_expr(node.condition, node)
            exit_jump = self.emit_jump(Op.JUMP_IF_FALSE, node)
        ctx = self.push_loop(top)
        self.compile_block_stmts(node.body)
        self.emit_loop(top, node)
        if exit_jump is not None:
            self.patch_jump(exit_jump)
        self.pop_loop(ctx)

    def compile_break_continue(self, node, is_break: bool):
        fs = self.fs
        if not fs.loops and fs.is_clause:
            self.error(node, f"'{'break' if is_break else 'continue'}' cannot leave an effect clause",
                       "finish the clause with a value (or resume) instead")
        if not fs.loops:
            self.error(node, f"'{'break' if is_break else 'continue'}' outside of a loop")
        ctx = fs.loops[-1]
        for _ in range(fs.try_depth - ctx['try_depth']):
            self.emit(Op.END_TRY)
        leave = fs.handle_depth - ctx['handle_depth']
        if leave:
            # leaving `handle` bodies: END_HANDLE drops each record and its clause
            # pairs (it keeps the value on top - here a dummy nil)
            self.emit(Op.PUSH_NIL, node=node)
            for k in range(leave):
                self.emit_end_handle(fs.handle_sps[-1 - k], node)
            self.emit(Op.POP, node=node)
        # values pushed by the expression around the break/continue (`total +=
        # handle ... continue`): drop them, the loop continues at its own depth
        if fs.depth_ok and not fs.dead:
            for _ in range(fs.depth - ctx['depth']):
                self.emit(Op.POP, node=node)
        if is_break:
            ctx['breaks'].append(self.emit_jump(Op.JUMP, node))
        elif ctx['continue_target'] is not None:
            self.emit_loop(ctx['continue_target'], node)
        else:
            ctx['continues'].append(self.emit_jump(Op.JUMP, node))

    def compile_for(self, node: For):
        self.begin_scope()
        if isinstance(node.iter, Range):
            self.compile_expr(node.iter.start, node)
            var = self.declare_local(node.var, mutable=True, node=node)
            self.store_new_local(var, node)
            self.compile_expr(node.iter.end, node)
            end_slot = self.hidden_local()
            self.emit(Op.STORE_LOCAL, end_slot)
            top = self.here()
            self.emit(Op.LOAD_CELL if var.celled else Op.LOAD_LOCAL, var.slot, node=node)
            self.emit(Op.LOAD_LOCAL, end_slot)
            self.emit(Op.LE if node.iter.inclusive else Op.LT)
            exit_jump = self.emit_jump(Op.JUMP_IF_FALSE, node)
            ctx = self.push_loop(None)
            self.compile_block_stmts(node.body)
            for c in ctx['continues']:
                self.patch_jump(c)
            self.emit(Op.LOAD_CELL if var.celled else Op.LOAD_LOCAL, var.slot)
            self.emit(Op.PUSH_I8, 1)
            self.emit(Op.ADD)
            self.emit(Op.STORE_CELL if var.celled else Op.STORE_LOCAL, var.slot)
            self.emit_loop(top, node)
            self.patch_jump(exit_jump)
            self.pop_loop(ctx)
        else:
            self.compile_expr(node.iter, node)
            self.emit(Op.NATIVE, NATIVE_ID['iter'], 1, node=node)
            it = self.hidden_local()
            idx = self.hidden_local()
            assert idx == it + 1
            self.emit(Op.STORE_LOCAL, it)
            self.emit(Op.PUSH_I8, 0)
            self.emit(Op.STORE_LOCAL, idx)
            top = self.here()
            at = self.emit(Op.FOR_ITER, it, 0, node=node) - 2
            var = self.declare_local(node.var, mutable=True, node=node)
            self.store_new_local(var, node)
            ctx = self.push_loop(top)
            self.compile_block_stmts(node.body)
            self.emit_loop(top, node)
            self.patch_jump(at)
            self.pop_loop(ctx)
        self.end_scope()

    # ── pattern matching ────────────────────────────────────────────────────
    def compile_match(self, node: MatchExpr, want_value: bool):
        self.begin_scope()
        self.compile_expr(node.value, node)
        subj = self.hidden_local()
        self.emit(Op.STORE_LOCAL, subj)
        ends = []
        for arm in node.arms:
            pat, guard, body = arm if len(arm) == 3 else (arm[0], None, arm[1])
            self.begin_scope()
            fails: List[int] = []
            self.compile_pattern(pat, subj, fails)
            if guard is not None:
                self.compile_expr(guard, guard)
                fails.append(self.emit_jump(Op.JUMP_IF_FALSE, guard))
            if want_value:
                if isinstance(body, Block):
                    self.compile_block_value(body)
                else:
                    self.compile_expr(body, node)
            else:
                if isinstance(body, Block):
                    self.compile_block_stmts(body)
                else:
                    self.compile_expr(body, node)
                    self.emit(Op.POP)
            ends.append(self.emit_jump(Op.JUMP))
            for f in fails:
                self.patch_jump(f)
            self.end_scope()
        if want_value:
            self.emit(Op.PUSH_NIL)
        for e in ends:
            self.patch_jump(e)
        self.end_scope()

    def pattern_type(self, callee) -> Optional[Tuple[int, int]]:
        """(type index, arity) for constructor patterns like Some(x), Shape.Circle(r)."""
        if isinstance(callee, Identifier):
            if callee.name in ('Some', 'Ok', 'Err'):
                return ({'Some': TYPE_SOME, 'Ok': TYPE_OK, 'Err': TYPE_ERR}[callee.name], 1)
            if callee.name in self.struct_types:
                return (self.struct_types[callee.name], len(self.struct_fields[callee.name]))
        if isinstance(callee, Attribute) and isinstance(callee.obj, Identifier):
            variants = self.adt_variants.get(callee.obj.name)
            if variants and callee.attr in variants:
                return variants[callee.attr]
        return None

    def compile_pattern(self, pat, subj: int, fails: List[int]):
        def compare_with(value_node):
            self.emit(Op.LOAD_LOCAL, subj)
            self.compile_expr(value_node, pat)
            self.emit(Op.EQ)
            fails.append(self.emit_jump(Op.JUMP_IF_FALSE, pat))

        if isinstance(pat, Identifier):
            name = pat.name
            if name == '_':
                return
            if name == 'NULL':
                self.emit(Op.LOAD_LOCAL, subj)
                self.emit(Op.PUSH_NIL)
                self.emit(Op.EQ)
                fails.append(self.emit_jump(Op.JUMP_IF_FALSE, pat))
                return
            if (self.find_local(self.fs, name) is None and
                    (name in self.inline_consts or name in self.global_slots)):
                compare_with(pat)
                return
            local = self.declare_local(name, mutable=False, node=pat)
            self.emit(Op.LOAD_LOCAL, subj)
            self.store_new_local(local, pat)
            return
        if isinstance(pat, Attribute) and isinstance(pat.obj, Identifier) \
                and pat.obj.name in self.adt_variants:
            variant = self.adt_variants[pat.obj.name].get(pat.attr)
            if variant is None:
                self.error(pat, f"enum '{pat.obj.name}' has no variant '{pat.attr}'")
            self.emit(Op.LOAD_LOCAL, subj)
            self.emit(Op.MATCH_TYPE, variant[0])
            fails.append(self.emit_jump(Op.JUMP_IF_FALSE, pat))
            return
        if isinstance(pat, Call):
            info = self.pattern_type(pat.func)
            if info is None:
                compare_with(pat)
                return
            tidx, arity = info
            if len(pat.args) != arity:
                self.error(pat, f"pattern expects {arity} field(s), got {len(pat.args)}")
            self.emit(Op.LOAD_LOCAL, subj)
            self.emit(Op.MATCH_TYPE, tidx)
            fails.append(self.emit_jump(Op.JUMP_IF_FALSE, pat))
            self.destructure(pat.args, subj, fails, lambda i: self.emit(Op.GET_SLOT, i))
            return
        if isinstance(pat, (TupleLit, ArrayLit)):
            self.emit(Op.LOAD_LOCAL, subj)
            self.emit(Op.IS_ARRAY_LEN, len(pat.elements))
            fails.append(self.emit_jump(Op.JUMP_IF_FALSE, pat))
            self.destructure(pat.elements, subj, fails, lambda i: self.emit(Op.GET_SLOT, i))
            return
        if isinstance(pat, StructInit) and pat.name in self.struct_types:
            self.emit(Op.LOAD_LOCAL, subj)
            self.emit(Op.MATCH_TYPE, self.struct_types[pat.name])
            fails.append(self.emit_jump(Op.JUMP_IF_FALSE, pat))
            names = list(pat.fields.keys())
            self.destructure(list(pat.fields.values()), subj, fails,
                             lambda i: self.emit(Op.GET_FIELD, self.m.const(names[i])))
            return
        if isinstance(pat, Range):
            self.emit(Op.LOAD_LOCAL, subj)
            self.compile_expr(pat.start, pat)
            self.emit(Op.GE)
            fails.append(self.emit_jump(Op.JUMP_IF_FALSE, pat))
            self.emit(Op.LOAD_LOCAL, subj)
            self.compile_expr(pat.end, pat)
            self.emit(Op.LE if pat.inclusive else Op.LT)
            fails.append(self.emit_jump(Op.JUMP_IF_FALSE, pat))
            return
        if isinstance(pat, BinaryOp) and pat.op == '|':
            alts, stack = [], [pat]
            while stack:
                p = stack.pop()
                if isinstance(p, BinaryOp) and p.op == '|':
                    stack.extend([p.right, p.left])
                else:
                    alts.append(p)
            oks = []
            for alt in alts:
                self.emit(Op.LOAD_LOCAL, subj)
                self.compile_expr(alt, pat)
                self.emit(Op.EQ)
                oks.append(self.emit_jump(Op.JUMP_IF_TRUE, alt))
            fails.append(self.emit_jump(Op.JUMP, pat))
            for o in oks:
                self.patch_jump(o)
            return
        compare_with(pat)

    def destructure(self, subpatterns, subj, fails, emit_get):
        for i, sub in enumerate(subpatterns):
            if isinstance(sub, Identifier) and sub.name == '_':
                continue
            tmp = self.hidden_local()
            self.emit(Op.LOAD_LOCAL, subj)
            emit_get(i)
            self.emit(Op.STORE_LOCAL, tmp)
            self.compile_pattern(sub, tmp, fails)

    # ── error-handling forms (real semantics) ──────────────────────────────
    def compile_guarded_body(self, body):
        self.fs.try_depth += 1
        if isinstance(body, Block):
            self.compile_block_value(body)
        else:
            self.compile_expr(body, body)
        self.fs.try_depth -= 1

    def compile_fallback(self, fallback, err_slot: int, node):
        """`else` branch of retry/timeout: the error message is bound to `error`."""
        if fallback is None:
            self.emit(Op.LOAD_LOCAL, err_slot)
            self.emit(Op.PANIC, node=node)      # re-raise
            return
        self.begin_scope()
        local = self.declare_local("error", mutable=False, node=node)
        self.emit(Op.LOAD_LOCAL, err_slot)
        self.store_new_local(local, node)
        if isinstance(fallback, Block):
            self.compile_block_value(fallback)
        else:
            self.compile_expr(fallback, node)
        self.end_scope()

    def compile_retry(self, node: RetryExpr):
        """retry(n) { body } else { fallback }: run body; on any runtime error
        (failed assert/contract, bounds, None unwrap, timeout, panic(...)) run it
        again, up to n attempts; then evaluate the fallback (or re-raise)."""
        self.begin_scope()
        limit, count, err = self.hidden_local(), self.hidden_local(), self.hidden_local()
        self.compile_expr(node.count, node)
        self.emit(Op.STORE_LOCAL, limit)
        self.emit(Op.PUSH_I8, 0)
        self.emit(Op.STORE_LOCAL, count)
        top = self.here()
        handler = self.emit_jump(Op.TRY, node)
        self.compile_guarded_body(node.body)
        self.emit(Op.END_TRY)
        done = self.emit_jump(Op.JUMP)
        self.patch_jump(handler)
        self.emit(Op.STORE_LOCAL, err)
        self.emit(Op.LOAD_LOCAL, count)
        self.emit(Op.PUSH_I8, 1)
        self.emit(Op.ADD)
        self.emit(Op.DUP)
        self.emit(Op.STORE_LOCAL, count)
        self.emit(Op.LOAD_LOCAL, limit)
        self.emit(Op.LT)
        again = self.emit_jump(Op.JUMP_IF_TRUE, node)
        self.patch_jump(again, top)
        self.compile_fallback(node.fallback, err, node)
        self.patch_jump(done)
        self.end_scope()

    def compile_timeout(self, node: TimeoutExpr):
        """timeout(ms) { body } else { fallback }: abort the body when it runs
        longer than ms milliseconds (checked at loop back-edges and calls)."""
        self.begin_scope()
        err = self.hidden_local()
        self.compile_expr(node.duration, node)
        handler = self.emit_jump(Op.TRY_TIMEOUT, node)
        self.compile_guarded_body(node.body)
        self.emit(Op.END_TRY)
        done = self.emit_jump(Op.JUMP)
        self.patch_jump(handler)
        self.emit(Op.STORE_LOCAL, err)
        self.compile_fallback(node.fallback, err, node)
        self.patch_jump(done)
        self.end_scope()

    def compile_intent(self, node: IntentBlock, want_value: bool):
        """intent "goal" { body }: errors raised inside carry the intent, so a
        failure reports *what the code was trying to do*, not just where."""
        handler = self.emit_jump(Op.TRY, node)
        self.fs.try_depth += 1
        if want_value:
            self.compile_block_value(node.body) if isinstance(node.body, Block) \
                else self.compile_expr(node.body, node)
        else:
            self.compile_block_stmts(node.body) if isinstance(node.body, Block) \
                else (self.compile_expr(node.body, node), self.emit(Op.POP))
        self.fs.try_depth -= 1
        self.emit(Op.END_TRY)
        done = self.emit_jump(Op.JUMP)
        self.patch_jump(handler)
        self.push_const(f"intent '{node.description}' failed: ", node)
        self.emit(Op.SWAP)
        self.emit(Op.CONCAT, 2)
        self.emit(Op.PANIC, node=node)
        self.patch_jump(done)

    # ── effect handlers with resumption (TIL 3.1, docs/TIL_EFFECTS.md) ──────
    def compile_handle(self, node: EffectHandler):
        """handle <body> with <clauses>  compiles to

               <clause 1> PUSH_CONST "E1" ... <clause n> PUSH_CONST "En"
               HANDLE n, end      the n (clause, name) pairs stay on the stack under
                                  the body; a handler record points at them
               <body value>
               END_HANDLE         drop the record and the pairs, keep the body value
           end:                   a clause that finishes without resume() lands here
                                  with its value (record, pairs and frames discarded)"""
        if len(node.handlers) > 255:
            self.error(node, "too many effect clauses in one handle (max 255)")
        arities: Dict[str, int] = {}
        pairs_depth = self.fs.depth          # the handle record's `sp`
        for clause in node.handlers:
            self.compile_clause(clause)
            self.emit(Op.PUSH_CONST, self.m.const(clause.name), node=clause)
            arities[clause.name] = len(clause.params)
        at = self.emit(Op.HANDLE, len(node.handlers), 0, node=node) - 2
        self.fs.handle_depth += 1
        self.fs.handle_sps.append(pairs_depth)
        self.static_effects.append(arities)
        if isinstance(node.expr, Block):
            self.compile_block_value(node.expr)
        else:
            self.compile_expr(node.expr, node)
        self.static_effects.pop()
        self.fs.handle_sps.pop()
        self.fs.handle_depth -= 1
        self.emit_end_handle(pairs_depth, node)
        self.patch_jump(at)

    def emit_end_handle(self, pairs_depth: int, node):
        """END_HANDLE keeps the value on top and cuts the stack back to the pairs."""
        self.emit(Op.END_HANDLE, node=node)
        self.fs.depth = pairs_depth + 1

    def compile_clause(self, clause: EffectClause):
        """A clause is a function with FLAG_EFFECT_CLAUSE that closes over the
        variables it uses (like a lambda). Its parameters are the perform's
        arguments; returning from it (= finishing without resume) aborts the handle."""
        proto = TBCFunction(f"<clause {clause.name}:{clause.line}>")
        proto.flags |= TBCFunction.FLAG_EFFECT_CLAUSE
        fidx = len(self.m.funcs)
        self.m.funcs.append(proto)
        parent = self.fs
        fs = _FuncState(proto, parent, clause)
        fs.captured = self.captured_names(clause.body)
        fs.is_clause = True
        self.fs = fs
        proto.n_params = proto.n_required = len(clause.params)
        for pname in clause.params:
            self.declare_local(pname, node=clause)
        for i, ptype in enumerate(clause.param_types):
            cast = self.numeric_cast_for(ptype)
            if cast is not None:
                self.emit(Op.LOAD_LOCAL, i)
                self.emit(Op.CAST, cast)
                self.emit(Op.STORE_LOCAL, i)
        for pname in clause.params:
            local = self.find_local(fs, pname)
            if local.celled:
                self.emit(Op.MAKE_CELL, local.slot)
        # Performs in a clause skip the handle that owns it (it is masked while the
        # clause runs); the handles around that one stay statically known.
        if isinstance(clause.body, Block):
            self.compile_block_value(clause.body)
        else:
            self.compile_expr(clause.body, clause)
        self.emit(Op.RETURN, node=clause)
        self.finish_debug_info(fs)
        proto.n_upvals = len(fs.upvals)
        upvals = fs.upvals
        self.fs = parent
        if not upvals:
            self.emit(Op.PUSH_FUNC, fidx, node=clause)
            return
        for _name, from_local, index, _mut in upvals:
            if from_local:
                self.emit(Op.LOAD_LOCAL, index, node=clause)       # pushes the cell itself
            else:
                self.emit(Op.LOAD_UPVAL_RAW, index, node=clause)
        self.emit(Op.MAKE_CLOSURE, fidx, len(upvals), node=clause)

    def compile_perform(self, node: PerformEffect):
        """perform E(args) -> PERFORM name, default, argc. At run time the VM finds
        the innermost active handler with a clause for E; without one it calls the
        top-level function E (TIL 3.0 behaviour, `default`), else `unhandled effect`."""
        argc = len(node.args)
        if argc > 255:
            self.error(node, "too many arguments (max 255)")
        for scope in reversed(self.static_effects):
            if node.name in scope:
                if scope[node.name] != argc:
                    self.error(node, f"effect '{node.name}' expects {scope[node.name]} argument(s), got {argc}",
                               "the clause of the enclosing handle takes a different number of parameters")
                break
        self.compile_args(node.args, node)
        default = self.funcs.get(node.name, 0xFFFF) if '.' not in node.name else 0xFFFF
        if default != 0xFFFF and default >= 0xFFFF:
            self.error(node, "program too large for the bytecode format")
        self.emit(Op.PERFORM, self.m.const(node.name), default, argc, node=node)

    def compile_resume(self, node: ResumeExpr):
        if not self.fs.is_clause:
            self.error(node, "resume(...) outside an effect clause")
        if node.value is None:
            self.emit(Op.PUSH_NIL, node=node)
        else:
            self.compile_expr(node.value, node)
        self.emit(Op.RESUME, node=node)

    # ── expressions ─────────────────────────────────────────────────────────
    def try_fold(self, node):
        if isinstance(node, (BinaryOp, UnaryOp)):
            try:
                v = const_eval(node, _FoldEnv(self))
            except (NotConstant, OverflowError, ValueError, ZeroDivisionError):
                return None
            if isinstance(v, (bool, int, float, str)):
                return v
        return None

    def names_a_constant(self, name: str) -> bool:
        """`name` here is the inline constant: no local, parameter, captured
        variable or `result` (in #[ensures]) shadows it (as emit_load_var)."""
        fs = self.fs
        if name not in self.inline_consts:
            return False
        if name == 'result' and getattr(fs, 'ensures_active', False):
            return False
        if self.find_local(fs, name) is not None:
            return False
        while fs.parent is not None:              # resolve_upval, without recording one
            if any(u[0] == name for u in fs.upvals):
                return False
            local = self.find_local(fs.parent, name)
            if local is not None:
                return not local.celled
            fs = fs.parent
        return True

    def compile_expr(self, node, parent=None):
        t = type(node)
        if t is IntLit:
            self.push_int(node.value, node)
        elif t is FloatLit:
            self.emit(Op.PUSH_CONST, self.m.const(float(node.value)), node=node)
        elif t is StringLit:
            self.emit(Op.PUSH_CONST, self.m.const(node.value), node=node)
        elif t is BoolLit:
            self.emit(Op.PUSH_TRUE if node.value else Op.PUSH_FALSE, node=node)
        elif t is CharLit:
            self.push_int(ord(node.value), node)
        elif t is Identifier:
            if node.name == 'NULL':
                self.emit(Op.PUSH_NIL, node=node)
            elif not self.emit_load_var(node.name, node):
                if node.name in self.funcs and '.' not in node.name:
                    self.emit(Op.PUSH_FUNC, self.funcs[node.name], node=node)
                elif node.name in self.struct_types or node.name in self.enum_values:
                    self.error(node, f"'{node.name}' is a type, not a value")
                else:
                    self.unknown_name(node, node.name)
        elif t is BinaryOp:
            self.compile_binary(node)
        elif t is UnaryOp:
            folded = self.try_fold(node)
            if folded is not None:
                self.push_const(folded, node)
                return
            if node.op in ('&', '&mut', '*'):
                # Structs, arrays and maps are references already: &x / *x are identity
                self.compile_expr(node.operand, node)
                return
            self.compile_expr(node.operand, node)
            self.emit({'-': Op.NEG, 'not': Op.NOT, '~': Op.BNOT}[node.op], node=node)
        elif t is Call:
            self.compile_call(node)
        elif t is Index:
            self.compile_expr(node.obj, node)
            self.compile_expr(node.index, node)
            self.emit(Op.GET_INDEX, node=node)
        elif t is Attribute:
            self.compile_attribute(node)
        elif t is ArrayLit or t is TupleLit:
            for e in node.elements:
                self.compile_expr(e, node)
            self.emit(Op.NEW_ARRAY, len(node.elements), node=node)
        elif t is DictLit:
            for k, v in node.pairs:
                self.compile_expr(k, node)
                self.compile_expr(v, node)
            self.emit(Op.NEW_MAP, len(node.pairs), node=node)
        elif t is Range:
            self.compile_expr(node.start, node)
            self.compile_expr(node.end, node)
            self.emit(Op.PUSH_TRUE if node.inclusive else Op.PUSH_FALSE)
            self.emit(Op.NATIVE, NATIVE_ID['range'], 3, node=node)
        elif t is StructInit:
            self.compile_struct_init(node)
        elif t is Lambda:
            self.compile_lambda(node)
        elif t is IfExpr:
            self.compile_expr(node.condition, node)
            jf = self.emit_jump(Op.JUMP_IF_FALSE, node)
            self.compile_expr(node.then_expr, node)
            end = self.emit_jump(Op.JUMP)
            self.patch_jump(jf)
            if node.else_expr is not None:
                self.compile_expr(node.else_expr, node)
            else:
                self.emit(Op.PUSH_NIL)
            self.patch_jump(end)
        elif t is MatchExpr:
            self.compile_match(node, True)
        elif t is Cast:
            self.compile_expr(node.expr, node)
            target = node.target_type
            if isinstance(target, PrimitiveType):
                name = target.name
                kind = ('float' if name in ('float', 'f32', 'f64') else
                        'str' if name == 'str' else 'bool' if name == 'bool' else 'int')
                self.emit(Op.CAST, CAST_KINDS[kind], node=node)
        elif t is NullCheck:
            self.compile_expr(node.expr, node)
            propagate = self.fs.proto.flags & TBCFunction.FLAG_PROPAGATE
            self.emit(Op.UNWRAP_OR_RETURN if propagate else Op.UNWRAP_OR_PANIC, node=node)
        elif t is ListComprehension:
            self.compile_comprehension(node)
        elif t is FStringLit:
            self.compile_fstring(node)
        elif t is RetryExpr:
            self.compile_retry(node)
        elif t is TimeoutExpr:
            self.compile_timeout(node)
        elif t is IntentBlock:
            self.compile_intent(node, True)
        elif t is Block:
            self.compile_block_value(node)
        elif t is EffectHandler:
            self.compile_handle(node)
        elif t is PerformEffect:
            self.compile_perform(node)
        elif t is ResumeExpr:
            self.compile_resume(node)
        elif t is DurationLit:
            mult = {"ms": 1, "s": 1000, "m": 60000, "h": 3600000}.get(node.unit, 1)
            self.push_int(int(node.value * mult), node)
        else:
            self.error(node if isinstance(node, ASTNode) else parent,
                       f"{t.__name__} is not supported by the VM backend")

    def compile_binary(self, node: BinaryOp):
        folded = self.try_fold(node)
        if folded is not None:
            self.push_const(folded, node)
            return
        op = node.op
        if op == 'and':
            self.compile_expr(node.left, node)
            f1 = self.emit_jump(Op.JUMP_IF_FALSE, node)
            self.compile_expr(node.right, node)
            f2 = self.emit_jump(Op.JUMP_IF_FALSE, node)
            self.emit(Op.PUSH_TRUE)
            end = self.emit_jump(Op.JUMP)
            self.patch_jump(f1)
            self.patch_jump(f2)
            self.emit(Op.PUSH_FALSE)
            self.patch_jump(end)
            return
        if op == 'or':
            self.compile_expr(node.left, node)
            t1 = self.emit_jump(Op.JUMP_IF_TRUE, node)
            self.compile_expr(node.right, node)
            t2 = self.emit_jump(Op.JUMP_IF_TRUE, node)
            self.emit(Op.PUSH_FALSE)
            end = self.emit_jump(Op.JUMP)
            self.patch_jump(t1)
            self.patch_jump(t2)
            self.emit(Op.PUSH_TRUE)
            self.patch_jump(end)
            return
        if op not in self.BINOPS:
            self.error(node, f"unsupported operator '{op}'")
        self.compile_expr(node.left, node)
        self.compile_expr(node.right, node)
        self.emit(self.BINOPS[op], node=node)

    def shadowed(self, name: str) -> bool:
        return self.find_local(self.fs, name) is not None or \
            any(n == name for n, *_ in self.fs.upvals)

    def compile_args(self, args, node):
        if len(args) > 255:
            self.error(node, "too many arguments (max 255)")
        for a in args:
            self.compile_expr(a, node)

    def check_arity(self, node, name: str, argc: int):
        fdef = self.fdefs.get(name)
        if fdef is None or isinstance(fdef, Lambda):
            return
        params = [p for p in fdef.params if p.name != 'self']
        required = sum(1 for p in params if p.default is None)
        if not required <= argc <= len(params):
            self.error(node, f"'{name}' expects {required if required == len(params) else f'{required} to {len(params)}'}"
                             f" argument(s), got {argc}")

    def compile_call(self, node: Call):
        f = node.func
        argc = len(node.args)
        if isinstance(f, Identifier):
            name = f.name
            if self.shadowed(name):
                self.emit_load_var(name, f)
                self.compile_args(node.args, node)
                self.emit(Op.CALL_VALUE, argc, node=node)
                return
            if name in self.funcs and '.' not in name:
                self.check_arity(node, name, argc)
                self.compile_args(node.args, node)
                self.emit(Op.CALL, self.funcs[name], argc, node=node)
                return
            if name in self.struct_types:
                nfields = len(self.struct_fields[name])
                if argc != nfields:
                    self.error(node, f"'{name}(...)' needs {nfields} field value(s), got {argc}")
                self.compile_args(node.args, node)
                self.emit(Op.NEW_STRUCT, self.struct_types[name], argc, node=node)
                return
            if name in ('Some', 'Ok', 'Err'):
                if argc > 1:
                    self.error(node, f"{name}(...) takes one value")
                self.compile_args(node.args, node)
                if argc == 0:
                    self.emit(Op.PUSH_NIL)
                self.emit(Op.NEW_STRUCT, {'Some': TYPE_SOME, 'Ok': TYPE_OK, 'Err': TYPE_ERR}[name], 1,
                          node=node)
                return
            if name in self.global_slots:
                self.emit(Op.LOAD_GLOBAL, self.global_slots[name], node=f)
                self.compile_args(node.args, node)
                self.emit(Op.CALL_VALUE, argc, node=node)
                return
            if name in NATIVE_ID:
                lo, hi = NATIVE_ARITY[name]
                if argc < lo or (hi is not None and argc > hi):
                    want = (f"{lo}" if lo == hi else f"{lo} to {hi}" if hi is not None
                            else f"at least {lo}")
                    self.error(node, f"'{name}' takes {want} argument(s), got {argc}")
                if name == 'range' and argc == 1:          # range(n) = range(0, n)
                    self.emit(Op.PUSH_I8, 0, node=node)
                    self.compile_args(node.args, node)
                    self.emit(Op.PUSH_FALSE)
                    self.emit(Op.NATIVE, NATIVE_ID['range'], 3, node=node)
                    return
                self.compile_args(node.args, node)
                self.emit(Op.NATIVE, NATIVE_ID[name], argc, node=node)
                return
            self.unknown_name(f, name, "function")
        if isinstance(f, Attribute):
            obj, meth = f.obj, f.attr
            if isinstance(obj, Identifier) and not self.shadowed(obj.name):
                oname = obj.name
                if oname in self.struct_types:
                    key = f"{oname}.{meth}"
                    if key not in self.funcs:
                        self.error(node, f"struct '{oname}' has no function '{meth}'")
                    self.check_arity(node, key, argc)
                    self.compile_args(node.args, node)
                    self.emit(Op.CALL, self.funcs[key], argc, node=node)
                    return
                if oname in self.adt_variants:
                    variant = self.adt_variants[oname].get(meth)
                    if variant is None:
                        self.error(node, f"enum '{oname}' has no variant '{meth}'")
                    if argc != variant[1]:
                        self.error(node, f"'{oname}.{meth}' takes {variant[1]} value(s), got {argc}")
                    self.compile_args(node.args, node)
                    self.emit(Op.NEW_STRUCT, variant[0], argc, node=node)
                    return
                if oname in self.module_names and oname not in self.global_slots:
                    if meth not in self.funcs:
                        self.error(node, f"module '{oname}' has no function '{meth}'")
                    self.check_arity(node, meth, argc)
                    self.compile_args(node.args, node)
                    self.emit(Op.CALL, self.funcs[meth], argc, node=node)
                    return
                if oname.startswith('__Vec__') and meth == 'new':
                    self.emit(Op.NEW_ARRAY, 0, node=node)
                    return
                if oname.startswith('__HashMap__') and meth == 'new':
                    self.emit(Op.NEW_MAP, 0, node=node)
                    return
            # dynamic method call: user struct method or builtin method
            mid = METHOD_ID.get(meth, NO_METHOD)
            known_field = any(meth in fields for _n, fields in self.m.types)
            if mid == NO_METHOD and meth not in self.user_method_names and not known_field:
                self.error(node, f"unknown method '{meth}'")
            if meth in PRELUDE_SOURCE:
                self.prelude_function(meth)
            self.compile_expr(obj, node)
            self.compile_args(node.args, node)
            self.emit(Op.INVOKE, self.m.const(meth), mid, argc, node=node)
            return
        # calling an arbitrary expression: f(x)(y), arr[0](x), ...
        self.compile_expr(f, node)
        self.compile_args(node.args, node)
        self.emit(Op.CALL_VALUE, argc, node=node)

    def compile_attribute(self, node: Attribute):
        obj = node.obj
        if isinstance(obj, Identifier) and not self.shadowed(obj.name):
            oname = obj.name
            if oname in self.enum_values:
                values = self.enum_values[oname]
                if node.attr not in values:
                    self.error(node, f"enum '{oname}' has no variant '{node.attr}'")
                self.push_int(values[node.attr], node)
                return
            if oname in self.adt_variants:
                variant = self.adt_variants[oname].get(node.attr)
                if variant is None:
                    self.error(node, f"enum '{oname}' has no variant '{node.attr}'")
                if variant[1] != 0:
                    self.error(node, f"'{oname}.{node.attr}' needs {variant[1]} value(s)")
                self.emit(Op.NEW_STRUCT, variant[0], 0, node=node)
                return
            if oname in self.module_names and oname not in self.global_slots:
                self.compile_expr(Identifier(name=node.attr, line=node.line, col=node.col), node)
                return
        self.compile_expr(obj, node)
        self.emit(Op.GET_FIELD, self.m.const(node.attr), node=node)

    def compile_struct_init(self, node: StructInit):
        if node.name not in self.struct_types:
            self.unknown_name(node, node.name, "struct")
        fields = self.struct_fields[node.name]
        names = {f.name for f in fields}
        for given in node.fields:
            if given not in names:
                self.error(node, f"struct '{node.name}' has no field '{given}'")
        for fld in fields:
            if fld.name in node.fields:
                self.compile_expr(node.fields[fld.name], node)
                cast = self.numeric_cast_for(fld.type)
                if cast == CAST_KINDS['float'] and not isinstance(node.fields[fld.name], FloatLit):
                    self.emit(Op.CAST, cast)
            elif fld.default is not None:
                self.compile_expr(fld.default, node)
            else:
                self.push_const(self.zero_value(fld.type), node)
        self.emit(Op.NEW_STRUCT, self.struct_types[node.name], len(fields), node=node)

    def compile_comprehension(self, node: ListComprehension):
        self.begin_scope()
        self.emit(Op.NEW_ARRAY, 0, node=node)
        res = self.hidden_local()
        self.emit(Op.STORE_LOCAL, res)
        body_stmts = []
        push_call = Call(func=Attribute(obj=Identifier(name=f" res", line=node.line), attr="push"),
                         args=[node.expr], line=node.line)
        body = ExprStmt(expr=push_call, line=node.line)
        if node.condition is not None:
            body = If(condition=node.condition, then_body=Block(statements=[body]), line=node.line)
        self.fs.scopes[-1][" res"] = _Local(" res", res, False, False)
        self.compile_for(For(var=node.var, iter=node.iter, body=Block(statements=[body]), line=node.line))
        self.emit(Op.LOAD_LOCAL, res)
        self.end_scope()

    def compile_fstring(self, node: FStringLit):
        count = 0
        for part in node.parts:
            kind, value = part[0], part[1]
            if kind == 'str':
                self.emit(Op.PUSH_CONST, self.m.const(value.value), node=node)
            else:
                self.compile_expr(value, node)
                if len(part) > 2:
                    self.emit(Op.PUSH_CONST, self.m.const(part[2]))
                    self.emit(Op.NATIVE, NATIVE_ID['format'], 2, node=node)
            count += 1
        if count == 0:
            self.emit(Op.PUSH_CONST, self.m.const(""), node=node)
        elif count > 255:
            self.error(node, "f-string has too many parts")
        else:
            self.emit(Op.CONCAT, count, node=node)


# ═══════════════════════════════════════════════════════════════════════════════
#                       REFERENCE VM (pure Python)
# ═══════════════════════════════════════════════════════════════════════════════

class TILPanic(Exception):
    """A recoverable TIL runtime error (caught by retry / timeout / intent)."""
    def __init__(self, msg: bytes, target: Optional[int] = None):
        super().__init__(msg)
        self.msg = msg if isinstance(msg, bytes) else str(msg).encode('utf-8')
        self.target = target          # recovery-frame index (deadline expiry)
        self.pc = -1


class TILFatal(Exception):
    """A non-recoverable VM error (step budget exhausted, stack overflow)."""
    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg.encode('utf-8')
        self.pc = -1


class TILExit(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


class TILPause(Exception):
    """Raised by the step check when TILVM.run_steps() reaches its pause point
    (the instruction that counted the step has completed and the top frame's pc
    is saved). Not an error: run_steps() returns TILVM.YIELD."""


class SnapshotError(ValueError):
    """A portable snapshot (.tilp, docs/TIL_SNAPSHOTS.md) cannot be saved or
    loaded. `code` is what tilvm_load_portable() returns for the same problem
    (TILVM_P_* in vm/tilvm.h); the message is TILP_ERRORS[code] plus details."""
    def __init__(self, code: int, detail: str = ""):
        base = TILP_ERRORS.get(code)        # code 0: a save problem, the detail says it all
        super().__init__(detail if base is None else base + (f" ({detail})" if detail else ""))
        self.code = code


TILP_MAGIC = b"TILP"
TILP_VERSION = 1
# value tags and object kinds of the .tilp format (docs/TIL_SNAPSHOTS.md §3.1)
TILP_NIL, TILP_FALSE, TILP_TRUE, TILP_INT, TILP_FLOAT, TILP_CONST, TILP_REF, TILP_FUNC, TILP_UNDEF = range(9)
TILP_STR, TILP_ARRAY, TILP_MAP, TILP_STRUCT, TILP_CLOSURE, TILP_CELL = range(1, 7)
TILP_ERRORS = {     # code -> message; vm/main.c prints the same texts
    -1: "not a portable TIL snapshot (.tilp)",
    -2: "unsupported snapshot version",
    -3: "the snapshot was saved by a different program",
    -4: "the snapshot is damaged or incomplete",
    -5: "the program's state does not fit this VM",
    -6: "the snapshot holds an integer that does not fit this VM",
    -7: "the snapshot holds floats, but this VM has no floats",
    -8: "the snapshot's screen is larger than this VM can show",
    -9: "portable snapshots are not available in this VM build",
    -10: "the snapshot uses effect handlers, which this VM build leaves out",
}


def fnv1a32(data: bytes) -> int:
    """FNV-1a-32: identifies the program in snapshots (vm->bc_hash in the C VM)."""
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


class _PRef:
    """An object reference read from a snapshot before its object is built."""
    __slots__ = ('i',)

    def __init__(self, i: int):
        self.i = i


class VMStruct:
    __slots__ = ('t', 'f')

    def __init__(self, t: int, f: list):
        self.t, self.f = t, f


class VMClosure:
    __slots__ = ('fn', 'up')

    def __init__(self, fn: int, up: tuple = ()):
        self.fn, self.up = fn, up


class VMCell:
    __slots__ = ('v',)

    def __init__(self, v):
        self.v = v


_UNDEF = object()   # argument slot not supplied by the caller (use the default)


def fmt_g(x: float) -> bytes:
    """C printf("%g") formatting (TIL's float output on every backend)."""
    if x != x:
        return b"nan"
    if x in (float('inf'), float('-inf')):
        return b"inf" if x > 0 else b"-inf"
    return ('%g' % x).encode()


def c_pow(a: float, b: float) -> float:
    """C99's pow() (the C VM's results): math.pow, with the overflow and pole
    results instead of Python's exceptions."""
    try:
        return _math.pow(a, b)
    except OverflowError:                          # |result| too large: inf, negative for x < 0 ** odd n
        odd = b == _math.floor(b) and _math.fmod(b, 2.0) != 0.0
        return float('-inf') if a < 0 and odd else float('inf')
    except ValueError:
        if a == 0.0:                               # 0 ** negative: a pole
            odd = b == _math.floor(b) and _math.fmod(b, 2.0) != 0.0
            return _math.copysign(float('inf'), a) if odd else float('inf')
        return float('nan')                        # negative ** fractional


def til_format_spec(v, spec: bytes, to_str, to_int=None) -> bytes:
    """f-string format specs: [<>^][0][width][.prec][f|e|%|d|x|X|o|b].
    `to_int` converts a float for d/x/X/o/b as int() does (the VM's cast: wraps
    to 64 bits, an error for nan/inf)."""
    m = re.match(rb'^([<>^]?)(0?)(\d*)(?:\.(\d+))?([fdxXobe%]?)$', spec)
    if not m:
        return to_str(v)
    align, zero, width, prec, kind = m.groups()
    width = int(width) if width else 0
    prec = int(prec) if prec is not None else None
    is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
    if kind in (b'f', b'e', b'%') and is_num:
        p = 6 if prec is None else prec
        x = float(v) * (100.0 if kind == b'%' else 1.0)
        body = (('%.' + str(p) + ('e' if kind == b'e' else 'f')) % x).encode()
        if kind == b'%':
            body += b'%'
    elif kind in (b'd', b'x', b'X', b'o', b'b') and is_num:
        n = v if type(v) is int else (to_int(v) if to_int else wrap_int64(int(v)))
        digits = {b'd': '%d', b'x': '%x', b'X': '%X', b'o': '%o'}.get(kind)
        body = (digits % abs(n)).encode() if digits else bin(abs(n))[2:].encode()
        if n < 0:
            body = b'-' + body
    elif isinstance(v, float) and prec is not None:
        body = (('%.' + str(prec) + 'g') % v).encode()
    else:
        body = to_str(v)
    if len(body) >= width:
        return body
    pad = width - len(body)
    if zero and is_num and not align:
        sign = body[:1] if body[:1] in (b'-', b'+') else b''
        return sign + b'0' * pad + body[len(sign):]
    if not align:
        align = b'>' if is_num else b'<'
    if align == b'<':
        return body + b' ' * pad
    if align == b'>':
        return b' ' * pad + body
    left = pad // 2
    return b' ' * left + body + b' ' * (pad - left)


# ── screen and keys (TIL 3.1, docs/TIL_GRAPHICS.md) ─────────────────────────
# The VM rasterizes shapes and text and clips them to the screen; the host
# (TILScreen) only stores pixels. vm/tilvm.c uses the same algorithms, limits
# and messages, so both VMs produce identical pixels.

# 5x7 font: ASCII 32..126, then the replacement glyph used for every other
# character. 5 columns per glyph, bit 0 = top row. Same table as FONT5X7 in
# vm/tilvm.c (the test-suite compares them).
FONT5X7 = bytes.fromhex(
    "000000000000004f00000007000700147f147f14242a7f2a12231308646236495522500005030000"
    "001c2241000041221c0014083e081408083e08080050300000080808080800606000002010080402"
    "3e5149453e00427f400042615149462141454b311814127f1027454545393c4a4949300171090503"
    "3649494936064949291e003636000000563600000814224100141414141400412214080201510906"
    "324979413e7e1111117e7f494949363e414141227f4141221c7f494949417f090909013e4149497a"
    "7f0808087f00417f41002040413f017f081422417f404040407f020c027f7f0408107f3e4141413e"
    "7f090909063e4151215e7f09192946464949493101017f01013f4040403f1f2040201f3f4038403f"
    "631408146307087008076151494543007f41410002040810200041417f0004020102044040404040"
    "000102040020545454787f484444383844444420384444487f3854545418087e090102085454543c"
    "7f0804047800447d40002040443d007f1028440000417f40007c041804787c080404783844444438"
    "7c14141408081414187c7c080404084854545420043f4440203c4040207c1c2040201c3c4030403c"
    "44281028440c5050503c4464544c44000836410000007f0000004136080008040810087f4141417f")
GFX_MAX_SIZE = 4096                        # screen_init: 1..4096 pixels per side
GFX_COORD_MIN, GFX_COORD_MAX = -32768, 32767   # shape coordinates and sizes
# argument kinds: i any int, c coordinate/size, s screen size, k color 0xRRGGBB,
# t any value (drawn like print shows it)
GFX_KINDS = {'screen_init': 'ss', 'screen_clear': 'k', 'draw_pixel': 'iik', 'draw_line': 'cccck',
             'draw_rect': 'cccck', 'fill_rect': 'cccck', 'draw_circle': 'ccck', 'draw_text': 'cctk',
             'get_pixel': 'ii', 'screen_show': ''}
GFX_FIRST_NATIVE, GFX_LAST_NATIVE = NATIVE_ID['screen_init'], NATIVE_ID['screen_height']
# --keys names (any case); a single character stands for its code point
KEY_NAMES = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'ENTER': 13, 'ESC': 27, 'SPACE': 32,
             'TAB': 9, 'BACKSPACE': 8, 'COMMA': 44, 'NONE': 0}
GFX_RUN_OPTIONS = ('screen', 'keys', 'screen_ansi')     # run options that need the VM
SNAPSHOT_RUN_OPTIONS = ('pause_after', 'save', 'resume')  # portable snapshots: also need the VM


def gfx_line_points(x0: int, y0: int, x1: int, y1: int):
    """Bresenham line, all octants, both end points included."""
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def gfx_circle_points(cx: int, cy: int, r: int):
    """Midpoint circle outline (r < 0: nothing, r = 0: the centre)."""
    x, y, d = 0, r, 1 - r
    while x <= y:
        yield cx + x, cy + y
        yield cx - x, cy + y
        yield cx + x, cy - y
        yield cx - x, cy - y
        yield cx + y, cy + x
        yield cx - y, cy + x
        yield cx + y, cy - x
        yield cx - y, cy - x
        if d < 0:
            d += 2 * x + 3
        else:
            d += 2 * (x - y) + 5
            y -= 1
        x += 1


def gfx_text_points(x: int, y: int, text: bytes):
    """Text in the 5x7 font: 6 px advance, b"\\n" starts a line 8 px lower. One
    glyph per byte below 0x80 or UTF-8 lead byte (continuation bytes are
    skipped); characters outside ASCII 32..126 use the replacement glyph."""
    cx, cy = x, y
    for c in text:
        if c == 10:
            cx, cy = x, cy + 8
            continue
        if 0x80 <= c < 0xC0:
            continue
        g = 5 * (c - 32 if 32 <= c <= 126 else 95)
        for col in range(5):
            bits = FONT5X7[g + col]
            for row in range(7):
                if (bits >> row) & 1:
                    yield cx + col, cy + row
        cx += 6


class TILScreen:
    """Framebuffer of the reference VM (the host side of the screen natives):
    w x h pixels, 3 bytes each. With `ansi=True` every screen_show() draws the
    frame into the program's output with 24-bit colour half blocks, byte for
    byte like `tilvm --screen-ansi`. `max_w` x `max_h` is the largest screen
    screen_init accepts, like a C VM built with -DTILVM_GFX_MAX_W/H for a
    device's display (default: the language limit 4096 x 4096)."""

    def __init__(self, ansi: bool = False, max_w: int = GFX_MAX_SIZE, max_h: int = GFX_MAX_SIZE):
        if not (1 <= max_w <= GFX_MAX_SIZE and 1 <= max_h <= GFX_MAX_SIZE):
            raise ValueError(f"TILScreen: max_w and max_h must be 1..{GFX_MAX_SIZE}")
        self.w = self.h = 0
        self.fb = bytearray()
        self.ansi = ansi
        self.max_w, self.max_h = max_w, max_h
        self.frames = 0                  # ANSI frames since screen_init

    def init(self, w: int, h: int):
        self.w, self.h = w, h
        self.fb = bytearray(3 * w * h)
        self.frames = 0

    def pixel(self, x: int, y: int, rgb: int):
        i = 3 * (y * self.w + x)
        fb = self.fb
        fb[i] = rgb >> 16
        fb[i + 1] = (rgb >> 8) & 255
        fb[i + 2] = rgb & 255

    def get(self, x: int, y: int) -> int:
        i = 3 * (y * self.w + x)
        fb = self.fb
        return (fb[i] << 16) | (fb[i + 1] << 8) | fb[i + 2]

    def fill(self, x0: int, y0: int, x1: int, y1: int, rgb: int):
        """Fill [x0, x1) x [y0, y1), clipped to the screen."""
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.w), min(y1, self.h)
        if x0 >= x1 or y0 >= y1:
            return
        row = bytes((rgb >> 16, (rgb >> 8) & 255, rgb & 255)) * (x1 - x0)
        for y in range(y0, y1):
            i = 3 * (y * self.w + x0)
            self.fb[i:i + len(row)] = row

    def show(self, write):
        if self.ansi:
            write(self.ansi_frame())

    def ansi_frame(self) -> bytes:
        """One frame: each character cell shows two pixels (U+2580 upper half
        block: foreground = upper pixel, background = lower pixel)."""
        out = [b"\x1b[H" if self.frames else b"\x1b[2J\x1b[H"]
        self.frames += 1
        w, h, fb = self.w, self.h, self.fb
        for y in range(0, h, 2):
            fg = bg = -1
            row = []
            for x in range(w):
                i = 3 * (y * w + x)
                top = (fb[i] << 16) | (fb[i + 1] << 8) | fb[i + 2]
                if top != fg:
                    row.append(b"\x1b[38;2;%d;%d;%dm" % (fb[i], fb[i + 1], fb[i + 2]))
                if y + 1 < h:
                    j = i + 3 * w
                    bot = (fb[j] << 16) | (fb[j + 1] << 8) | fb[j + 2]
                    if bot != bg:
                        row.append(b"\x1b[48;2;%d;%d;%dm" % (fb[j], fb[j + 1], fb[j + 2]))
                else:
                    bot = -2
                    if bot != bg:
                        row.append(b"\x1b[49m")
                fg, bg = top, bot
                row.append(b"\xe2\x96\x80")
            out.append(b"".join(row) + b"\x1b[0m\n")
        return b"".join(out)

    def ppm(self) -> bytes:
        return b"P6\n%d %d\n255\n" % (self.w, self.h) + bytes(self.fb)


def parse_key_script(spec: str) -> List[int]:
    """--keys "UP,UP,ENTER,a,ESC": comma-separated names (any case) or single
    characters -> key codes (NONE = 0); "" = no keys. Raises ValueError."""
    if not spec.strip(' \t'):
        return []
    codes = []
    for raw in spec.split(','):
        tok = raw.strip(' \t')
        name = tok.encode('utf-8', 'surrogateescape').upper().decode('utf-8', 'replace')
        if len(tok) == 1 and not 0xD800 <= ord(tok) <= 0xDFFF:
            codes.append(ord(tok))
        elif name in KEY_NAMES:
            codes.append(KEY_NAMES[name])
        else:
            raise ValueError(f"bad --keys entry '{tok}' (use UP, DOWN, LEFT, RIGHT, ENTER, ESC, SPACE, "
                             f"TAB, BACKSPACE, COMMA, NONE or one character)")
    return codes


class KeyScript:
    """Scripted keys (--keys): key_pressed() takes the next entry (0 once the
    script is used up); wait_key() skips NONE entries and returns 27 (ESC) when
    the script has ended, so programs that wait for keys always finish."""

    def __init__(self, codes=()):
        self.codes = list(codes)
        self.pos = 0

    def pressed(self) -> int:
        if self.pos < len(self.codes):
            self.pos += 1
            return self.codes[self.pos - 1]
        return 0

    def wait(self) -> int:
        while self.pos < len(self.codes):
            self.pos += 1
            if self.codes[self.pos - 1]:
                return self.codes[self.pos - 1]
        return 27

    def close(self):
        pass


class TerminalKeys:
    """Keys from the terminal (raw mode while the program reads keys, restored by
    close(), before input() and on SIGTERM) or, when stdin is not a terminal, from
    stdin's bytes (at the end: key_pressed() -> 0, wait_key() -> 27). Arrows
    (ESC [ A..D) -> 1..4, CR/LF -> 13, DEL -> 8, UTF-8 -> code point: the same
    decoding as vm/main.c. Keys and input() lines (cooked()) take the same
    buffered bytes, so they never overtake each other. key_pressed() only takes
    bytes that are already there, so it never blocks, also with a pipe (Windows
    cannot poll a pipe: there the read waits); a regular file is always ready."""

    def __init__(self):
        self.mode = None            # 'tty' (POSIX terminal), 'win' (Windows console), 'stream'
        self.fd = -1
        self.saved = None           # termios settings to restore
        self.old_sigterm = None     # SIGTERM handler before raw mode
        self.buf = b""              # bytes read from stdin; buf[pos:] are not used yet
        self.pos = 0

    def _start_mode(self):
        if self.mode is None:
            try:
                self.fd = sys.stdin.fileno()
                tty = os.isatty(self.fd)
            except (AttributeError, OSError, ValueError):
                self.fd, tty = -1, False
            self.mode = ('win' if os.name == 'nt' else 'tty') if tty else 'stream'

    def _start(self):
        self._start_mode()
        if self.mode == 'tty' and self.saved is None:
            import termios
            self._catch_sigterm()               # before the terminal changes
            try:
                self.saved = termios.tcgetattr(self.fd)
                raw = termios.tcgetattr(self.fd)
                raw[3] &= ~(termios.ICANON | termios.ECHO)
                raw[6][termios.VMIN] = 1
                raw[6][termios.VTIME] = 0
                termios.tcsetattr(self.fd, termios.TCSANOW, raw)
            except termios.error:
                self.saved = None

    def _catch_sigterm(self):
        """SIGTERM while the terminal is raw: restore it, then end by SIGTERM (as
        vm/main.c does)."""
        import signal

        def on_sigterm(signum, frame):
            self.close()
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        try:
            old = signal.signal(signal.SIGTERM, on_sigterm)
        except (ValueError, OSError, AttributeError):     # not the main thread
            return
        if self.old_sigterm is None:
            self.old_sigterm = old

    def close(self):
        """Give the terminal back its normal mode (echo, line editing)."""
        if self.saved is not None:
            import termios
            try:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)
            except termios.error:
                pass
            self.saved = None
        if self.old_sigterm is not None:
            import signal
            try:
                signal.signal(signal.SIGTERM, self.old_sigterm)
            except (ValueError, OSError):
                pass
            self.old_sigterm = None

    def cooked(self, read_line):
        """Wrap input(): the terminal is in normal mode while a line is read, and
        the line comes from the same buffered bytes as the keys."""
        def read():
            self.close()
            self._start_mode()
            if self.mode == 'win':
                return read_line()
            try:
                sys.stdout.flush()
            except (AttributeError, OSError, ValueError):
                pass
            return self.read_line()
        return read

    def _read(self, timeout) -> Optional[bytes]:
        """One read from stdin: None = no data within `timeout` seconds (None:
        wait), b"" = end of input (or stdin cannot be read)."""
        if self.fd < 0:             # stdin without a file descriptor (replaced in-process)
            stdin = sys.stdin
            try:
                buf = getattr(stdin, 'buffer', None)
                if buf is not None:
                    return buf.read1(4096) if hasattr(buf, 'read1') else buf.read(4096)
                return stdin.read(4096).encode('utf-8')
            except (AttributeError, OSError, ValueError):
                return b""
        import select
        while True:
            if timeout is not None:
                try:
                    if not select.select([self.fd], [], [], timeout)[0]:
                        return None
                except (OSError, ValueError):
                    pass            # cannot be polled (a pipe on Windows): the read waits
            try:
                return os.read(self.fd, 4096)
            except BlockingIOError:     # a non-blocking stdin
                if timeout is not None:
                    return None
                select.select([self.fd], [], [])
            except OSError:
                return b""

    def _byte(self, timeout) -> int:
        """Next byte of stdin: -1 = none within `timeout` seconds (None: wait),
        -2 = end of input."""
        if self.pos < len(self.buf):
            self.pos += 1
            return self.buf[self.pos - 1]
        data = self._read(timeout)
        if data is None:
            return -1
        if not data:
            return -2
        self.buf, self.pos = data, 1
        return data[0]

    def read_line(self) -> bytes:
        """input(): the next line without its newline (b"" at the end of input),
        like sys.stdin.buffer.readline()."""
        parts = []
        while True:
            if self.pos >= len(self.buf):
                data = self._read(None)
                if not data:
                    return b"".join(parts)
                self.buf, self.pos = data, 0
            i = self.buf.find(b"\n", self.pos)
            if i >= 0:
                parts.append(self.buf[self.pos:i])
                self.pos = i + 1
                return b"".join(parts)
            parts.append(self.buf[self.pos:])
            self.pos = len(self.buf)

    def _decode(self, c: int) -> int:
        """One key, `c` being its first byte; -1 = bytes that are not a key. A lone
        ESC is one that no '[' or 'O' follows within 30 ms."""
        nxt = self._byte
        if c == 27:
            c2 = nxt(0.03)
            if c2 not in (0x5B, 0x4F):              # '[' 'O'
                if c2 >= 0:
                    self.pos -= 1                    # a lone ESC: c2 is the next key
                return 27
            c3 = nxt(0.03)
            if 0x41 <= c3 <= 0x44:                   # A B C D
                return (1, 2, 4, 3)[c3 - 0x41]
            while c3 >= 0 and not 0x40 <= c3 <= 0x7E:
                c3 = nxt(0.03)
            return -1
        if 0xC0 <= c < 0xF8:
            extra = 3 if c >= 0xF0 else 2 if c >= 0xE0 else 1
            cp = c & (0x3F >> extra)
            for _ in range(extra):
                b = nxt(0.03)
                if not 0x80 <= b <= 0xBF:
                    return -1
                cp = (cp << 6) | (b & 0x3F)
            return cp
        if c >= 0x80:
            return -1
        if c in (10, 13):
            return 13
        return 8 if c == 127 else c

    def _win_key(self, block: bool) -> int:
        import msvcrt
        while True:
            if not block and not msvcrt.kbhit():
                return 0
            ch = msvcrt.getwch()
            if ch in ('\x00', '\xe0'):               # arrows: prefix + scan code
                k = {72: 1, 80: 2, 75: 3, 77: 4}.get(ord(msvcrt.getwch()))
                if k:
                    return k
                continue
            return 13 if ch in ('\r', '\n') else ord(ch)

    def pressed(self) -> int:
        self._start()
        if self.mode == 'win':
            return self._win_key(False)
        c = self._byte(0)
        if c < 0:
            return 0
        k = self._decode(c)
        return k if k > 0 else 0

    def wait(self) -> int:
        self._start()
        try:
            sys.stdout.flush()
        except (AttributeError, OSError, ValueError):
            pass
        if self.mode == 'win':
            return self._win_key(True)
        while True:
            c = self._byte(None)
            if c == -2:
                return 27
            if c >= 0:
                k = self._decode(c)
                if k > 0:
                    return k


class TILVM:
    """Reference interpreter for TIL bytecode modules."""

    CHECK_INTERVAL = 256
    MAX_FRAMES = 2000
    MAX_HANDLERS = 32     # = TILVM_MAX_HANDLERS of the default C VM build
    # run_steps() results: the same numbers as TILVM_DONE ... TILVM_EXIT in vm/tilvm.h
    DONE, ERROR, YIELD, EXIT = 0, 1, 2, 3

    def __init__(self, module: TBCModule, write=None, read_line=None, max_steps: int = 0,
                 argv: Optional[List[Any]] = None, screen: Optional['TILScreen'] = None, keys=None,
                 tbc: Optional[bytes] = None):
        self.m = module
        # the .tbc bytes (portable snapshots name the program by their hash):
        # given, else the file the module was read from, else module.serialize()
        self._tbc = tbc
        # program arguments for args(): bytes (str is encoded as UTF-8)
        self.argv = [a.encode('utf-8') if isinstance(a, str) else bytes(a) for a in (argv or [])]
        # screen and keys: an offscreen framebuffer and no keys unless the host passes
        # others (keys: a --keys string, a list of key codes, or a key source object)
        self.screen = screen if screen is not None else TILScreen()
        self.keys = keys if hasattr(keys, 'pressed') else KeyScript(
            parse_key_script(keys) if isinstance(keys, str) else (keys or ()))
        self.code = module.code
        self.consts = module.consts
        self._own_stdout = write is None
        self.write = write or self._stdout_write
        self.read_line = read_line or self._stdin_line
        self.steps = 0
        # energy saving (docs/TIL_ENERGY.md): `energy` is the program's mode (0 performance,
        # 1 balanced, 2 low; C VM: st.energy), `idle_count` the idles so far (tilvm --stats)
        self.idle_count = 0
        self.low_funcs = frozenset(i for i, f in enumerate(module.funcs)
                                   if f.flags & TBCFunction.FLAG_ENERGY_LOW)
        self._max_steps = self._energy = 0
        self.max_steps = max_steps
        self.energy = getattr(module, 'energy', ENERGY_PERFORMANCE)
        self.globals = [None] * len(module.globals)
        self.methods = {(t, n): f for t, n, f in module.methods}
        const_of = {}
        for i, c in enumerate(module.consts):
            if isinstance(c, bytes):
                const_of.setdefault(c, i)
        self.type_names = [name for name, _ in module.types]
        # per type: field-name const index -> slot
        self.field_slots = []
        for _name, fields in module.types:
            slots = {}
            for i, fname in enumerate(fields):
                key = fname.encode('utf-8') if isinstance(fname, str) else fname
                if key in const_of:
                    slots[const_of[key]] = i
            self.field_slots.append(slots)
        self.field_names = [[f.encode('utf-8') if isinstance(f, str) else f for f in fields]
                            for _n, fields in module.types]
        self.rng = 0x9E3779B97F4A7C15
        # [handler_pc, n_frames, sp, deadline_ms or 0, n_handlers]
        self.recover: List[list] = []
        # Effect handler stack (docs/TIL_EFFECTS.md); both kinds of entry have 6 fields:
        #   [0, n_frames, sp, n_try, abort_pc, n]     handle: n (clause, name) pairs at stack[sp:]
        #   [1, n_frames, 0,  n_try, 0,     owner]   a clause of handler #owner is running
        #                                             (masks #owner and everything above it)
        self.handlers: List[list] = []
        self.deadlines = 0
        self.tick = 0
        self.stack: list = []
        self.frames: List[list] = []      # [func, pc, base, closure, extra]
        self.started = False              # entry frame set up (start() or load_portable())
        self.status: Optional[int] = None  # DONE / ERROR / EXIT once the program has ended
        self.exit_code: Optional[int] = None
        self.pause_at = 0                 # run_steps(): pause when steps reaches this (0: never)

    # max_steps and energy are properties: assigning either keeps `_slow` right, the
    # flag check_budget tests at every step before it looks at the step limit and at
    # energy saving (a program with neither pays one test per step, as in TIL 3.0)
    @property
    def max_steps(self) -> int:
        """Sandbox: stop after this many steps (0 = no limit)."""
        return self._max_steps

    @max_steps.setter
    def max_steps(self, n: int):
        self._max_steps = n
        self._slow = bool(n or self._energy or self.low_funcs)

    @property
    def energy(self) -> int:
        """The program's energy mode: 0 performance, 1 balanced, 2 low (C VM: st.energy)."""
        return self._energy

    @energy.setter
    def energy(self, mode: int):
        self._energy = mode
        self._slow = bool(self._max_steps or mode or self.low_funcs)

    @staticmethod
    def _stdout_write(data: bytes):
        out = getattr(sys.stdout, 'buffer', None)
        if out is not None:
            out.write(data)
        else:
            sys.stdout.write(data.decode('utf-8', 'replace'))

    def flush(self):
        if self._own_stdout:
            try:
                sys.stdout.flush()
            except Exception:
                pass

    def _stdin_line(self):
        self.flush()
        line = sys.stdin.buffer.readline()
        return line[:-1] if line.endswith(b'\n') else line

    # ── value helpers ───────────────────────────────────────────────────────
    def type_name(self, v) -> bytes:
        if v is None:
            return b"nil"
        if v is True or v is False:
            return b"bool"
        if type(v) is int:
            return b"int"
        if type(v) is float:
            return b"float"
        if type(v) is bytes:
            return b"str"
        if type(v) is list:
            return b"array"
        if type(v) is dict:
            return b"map"
        if type(v) is VMClosure:
            return b"fn"
        if type(v) is VMStruct:
            return self.type_names[v.t].encode('utf-8')
        return b"?"

    def to_str(self, v, depth: int = 0, quote: bool = False) -> bytes:
        if v is None:
            return b"None"
        if v is True:
            return b"true"
        if v is False:
            return b"false"
        tv = type(v)
        if tv is int:
            return str(v).encode()
        if tv is float:
            return fmt_g(v)
        if tv is bytes:
            return b'"' + v + b'"' if quote else v
        if depth > 16:
            return b"..."
        if tv is list:
            return b"[" + b", ".join(self.to_str(x, depth + 1, True) for x in v) + b"]"
        if tv is dict:
            return b"{" + b", ".join(self.to_str(k, depth + 1, True) + b": " +
                                     self.to_str(x, depth + 1, True) for k, x in v.items()) + b"}"
        if tv is VMStruct:
            name = self.type_names[v.t].encode('utf-8')
            fields = self.field_names[v.t]
            if v.t <= TYPE_ERR or (fields and fields[0] == b"0"):
                inner = b", ".join(self.to_str(x, depth + 1, True) for x in v.f)
                return name + b"(" + inner + b")" if fields else name
            if not fields:
                return name
            return name + b" { " + b", ".join(
                fn + b": " + self.to_str(x, depth + 1, True) for fn, x in zip(fields, v.f)) + b" }"
        if tv is VMClosure:
            return b"<fn " + self.m.funcs[v.fn].name.encode('utf-8') + b">"
        return b"?"

    @staticmethod
    def truthy(v) -> bool:
        if v is None or v is False:
            return False
        if v is True:
            return True
        tv = type(v)
        if tv is int or tv is float:
            return v != 0
        if tv is bytes or tv is list or tv is dict:
            return len(v) != 0
        return True

    def equal(self, a, b, depth: int = 0) -> bool:
        ta, tb = type(a), type(b)
        if ta is bool or tb is bool:
            return ta is tb and a == b
        if (ta is int or ta is float) and (tb is int or tb is float):
            return a == b
        if ta is not tb:
            return False
        if a is b:
            return True
        if depth > 32:
            return False
        if ta is list:
            return len(a) == len(b) and all(self.equal(x, y, depth + 1) for x, y in zip(a, b))
        if ta is dict:
            if len(a) != len(b):
                return False
            for k, v in a.items():
                if k not in b or not self.equal(v, b[k], depth + 1):
                    return False
            return True
        if ta is VMStruct:
            return a.t == b.t and all(self.equal(x, y, depth + 1) for x, y in zip(a.f, b.f))
        return a == b

    def panic(self, msg):
        raise TILPanic(msg if isinstance(msg, bytes) else msg.encode('utf-8'))

    def num(self, v, what: str):
        if type(v) is int or type(v) is float:
            return v
        self.panic(f"{what} expects a number, got {self.type_name(v).decode()}")

    def arith(self, op: int, a, b):
        ta, tb = type(a), type(b)
        if ta is int and tb is int:
            if op == Op.ADD:
                return wrap_int64(a + b)
            if op == Op.SUB:
                return wrap_int64(a - b)
            if op == Op.MUL:
                return wrap_int64(a * b)
            if op == Op.DIV:
                if b == 0:
                    self.panic("division by zero")
                return til_idiv(a, b)
            if op == Op.MOD:
                if b == 0:
                    self.panic("division by zero")
                return til_imod(a, b)
            if op == Op.POW:
                if b < 0:
                    return 1 if a == 1 else ((-1 if b & 1 else 1) if a == -1 else 0)
                return wrap_int64(pow(a, b, 1 << 64))
        if (ta is int or ta is float) and (tb is int or tb is float):
            a, b = float(a), float(b)
            if op == Op.ADD:
                return a + b
            if op == Op.SUB:
                return a - b
            if op == Op.MUL:
                return a * b
            if op == Op.DIV:
                if b == 0.0:
                    if a == 0.0 or a != a:
                        return float('nan')
                    return _math.copysign(float('inf'), a) * _math.copysign(1.0, b)
                return a / b
            if op == Op.MOD:            # C's fmod (vm/tilvm.c): nan for x % 0 and inf % y
                return _math.fmod(a, b) if b != 0.0 and a - a == 0.0 else float('nan')
            if op == Op.POW:
                return c_pow(a, b)
        if op == Op.ADD:
            if ta is bytes or tb is bytes:
                return self.to_str(a) + self.to_str(b)
            if ta is list and tb is list:
                return a + b
        if op == Op.MUL:
            if ta is bytes and tb is int:
                return a * max(b, 0)
            if ta is list and tb is int:
                return a * max(b, 0)
        names = {Op.ADD: '+', Op.SUB: '-', Op.MUL: '*', Op.DIV: '/', Op.MOD: '%', Op.POW: '**'}
        self.panic(f"unsupported operand types for {names[op]}: "
                   f"{self.type_name(a).decode()} and {self.type_name(b).decode()}")

    def compare(self, op: int, a, b) -> bool:
        ta, tb = type(a), type(b)
        if (ta is int or ta is float) and (tb is int or tb is float):
            pass
        elif ta is bytes and tb is bytes:
            pass
        else:
            self.panic(f"cannot compare {self.type_name(a).decode()} and {self.type_name(b).decode()}")
        if op == Op.LT:
            return a < b
        if op == Op.LE:
            return a <= b
        if op == Op.GT:
            return a > b
        return a >= b

    def bitwise(self, op: int, a, b):
        if type(a) is not int or type(b) is not int:
            self.panic("bitwise operators need integers")
        if op == Op.BAND:
            return a & b
        if op == Op.BOR:
            return a | b
        if op == Op.BXOR:
            return a ^ b
        if op == Op.SHL:
            return wrap_int64(a << (b & 63))
        return a >> (b & 63)

    def cast(self, kind: int, v):
        if kind == 0:   # int
            tv = type(v)
            if tv is int:
                return v
            if v is True or v is False:
                return int(v)
            if tv is float:
                if v != v or v in (float('inf'), float('-inf')):
                    self.panic("cannot convert nan/inf to int")
                return wrap_int64(int(v))
            if tv is bytes:
                try:
                    return wrap_int64(int(v.strip()))
                except ValueError:
                    self.panic(b"invalid integer: \"" + v + b"\"")
            if v is None:
                return 0
            self.panic(f"cannot convert {self.type_name(v).decode()} to int")
        if kind == 1:   # float
            tv = type(v)
            if tv is float:
                return v
            if tv is int or v is True or v is False:
                return float(v)
            if tv is bytes:
                try:
                    return float(v.strip())
                except ValueError:
                    self.panic(b"invalid number: \"" + v + b"\"")
            if v is None:
                return 0.0
            self.panic(f"cannot convert {self.type_name(v).decode()} to float")
        if kind == 2:
            return self.to_str(v)
        return self.truthy(v)

    def index_get(self, obj, idx):
        to = type(obj)
        if to is list or to is bytes:
            if type(idx) is not int:
                self.panic("index must be an integer")
            if not 0 <= idx < len(obj):
                self.panic(f"index out of bounds: index {idx}, length {len(obj)}")
            return obj[idx]
        if to is dict:
            key = self.map_key(idx)
            if key not in obj:
                self.panic(b"key not found: " + self.to_str(idx))
            return obj[key]
        if to is VMStruct and type(idx) is int and 0 <= idx < len(obj.f):
            return obj.f[idx]
        self.panic(f"{self.type_name(obj).decode()} cannot be indexed")

    def map_key(self, k):
        tk = type(k)
        if tk is bytes or tk is int or k is None or tk is bool:
            return k
        if tk is float:      # 2.0 -> 2 (a whole number in int64 range; inf, nan and 1e300 stay floats)
            return int(k) if k - k == 0.0 and k == int(k) and -2.0 ** 63 <= k < 2.0 ** 63 else k
        self.panic(f"{self.type_name(k).decode()} cannot be a map key")

    def index_set(self, obj, idx, val):
        to = type(obj)
        if to is list:
            if type(idx) is not int:
                self.panic("index must be an integer")
            if not 0 <= idx < len(obj):
                self.panic(f"index out of bounds: index {idx}, length {len(obj)}")
            obj[idx] = val
        elif to is dict:
            obj[self.map_key(idx)] = val
        elif to is bytes:
            self.panic("strings are immutable")
        else:
            self.panic(f"{self.type_name(obj).decode()} does not support item assignment")

    def get_field(self, obj, name_const: int):
        if type(obj) is VMStruct:
            slot = self.field_slots[obj.t].get(name_const)
            if slot is not None:
                return obj.f[slot]
        self.panic(b"no field '" + self.consts[name_const] + b"' on " + self.type_name(obj))

    def set_field(self, obj, name_const: int, val):
        if type(obj) is VMStruct:
            slot = self.field_slots[obj.t].get(name_const)
            if slot is not None:
                obj.f[slot] = val
                return
        self.panic(b"no field '" + self.consts[name_const] + b"' on " + self.type_name(obj))

    # ── natives ─────────────────────────────────────────────────────────────
    def call_native(self, nid: int, args: list):
        name = NATIVES[nid] if nid < len(NATIVES) else '?'
        if nid < len(NATIVES) and len(args) < NATIVE_MIN_ARGS[nid]:
            self.panic(f"{name}() is missing arguments")
        if GFX_FIRST_NATIVE <= nid <= GFX_LAST_NATIVE:
            return self.gfx_native(name, args)
        if name in ('print', 'println'):
            if not args:
                self.write(b"\n")
            for a in args:
                self.write(self.to_str(a) + b"\n")
            return None
        if name == 'iter':
            v = args[0]
            if type(v) is dict:
                return list(v.keys())
            if type(v) in (list, bytes):
                return v
            self.panic(f"{self.type_name(v).decode()} is not iterable")
        if name == 'len':
            v = args[0]
            if type(v) in (bytes, list, dict):
                return len(v)
            self.panic(f"len() of {self.type_name(v).decode()}")
        if name == 'str':
            return self.to_str(args[0])
        if name == 'int':
            return self.cast(0, args[0])
        if name == 'float':
            return self.cast(1, args[0])
        if name == 'bool':
            return self.truthy(args[0])
        if name == 'abs':
            v = self.num(args[0], 'abs')
            return wrap_int64(abs(v)) if type(v) is int else abs(v)
        if name in ('sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'floor', 'ceil', 'round'):
            x = float(self.num(args[0], name))
            try:
                if name == 'sqrt':
                    return _math.sqrt(x) if x >= 0 else float('nan')
                if name == 'log':
                    return _math.log(x) if x > 0 else (float('-inf') if x == 0 else float('nan'))
                if name == 'round':
                    return _math.copysign(_math.floor(abs(x) + 0.5), x) if x == x else x
                if name in ('floor', 'ceil') and (x != x or x in (float('inf'), float('-inf'))):
                    return x
                return float(getattr(_math, name)(x))
            except (OverflowError, ValueError):
                return float('inf') if name == 'exp' else float('nan')
        if name == 'pow':
            return self.arith(Op.POW, float(self.num(args[0], 'pow')), float(self.num(args[1], 'pow')))
        if name in ('min', 'max'):
            vals = args[0] if len(args) == 1 and type(args[0]) is list else args
            if not vals:
                self.panic(f"{name}() of empty sequence")
            best = vals[0]
            for v in vals[1:]:
                if self.compare(Op.LT if name == 'min' else Op.GT, v, best):
                    best = v
            return best
        if name == 'sum':
            total = 0
            for v in args[0] if len(args) == 1 and type(args[0]) is list else args:
                total = self.arith(Op.ADD, total, v)
            return total
        if name == 'input':
            if args:
                self.write(self.to_str(args[0]))
            return self.read_line()
        if name == 'chr':
            n = self.num(args[0], 'chr')
            try:
                return chr(int(n)).encode('utf-8')
            except (ValueError, OverflowError):
                self.panic("chr() argument out of range")
        if name == 'ord':
            s = args[0]
            if type(s) is not bytes or not s:
                self.panic("ord() expects a non-empty string")
            return ord(s.decode('utf-8', 'replace')[0]) if s[0] >= 0x80 else s[0]
        if name == 'range':
            a, b = args[0], args[1]
            incl = len(args) > 2 and self.truthy(args[2])
            if type(a) is not int or type(b) is not int:
                self.panic("range bounds must be integers")
            return list(range(a, b + 1 if incl else b))
        if name == 'type_of':
            return self.type_name(args[0])
        if name == 'format':
            return til_format_spec(args[0], args[1], self.to_str, lambda x: self.cast(0, x))
        if name == 'clock_ms':
            return int(_time.monotonic() * 1000)
        if name == 'random':
            self.rng ^= (self.rng >> 12)
            self.rng ^= (self.rng << 25) & INT64_MASK
            self.rng ^= (self.rng >> 27)
            r = (self.rng * 0x2545F4914F6CDD1D) & INT64_MASK
            if len(args) >= 2:
                lo, hi = args[0], args[1]
                if hi < lo:
                    self.panic("random(lo, hi) needs lo <= hi")
                return lo + (r % (hi - lo + 1))
            return (r >> 11) / float(1 << 53)
        if name == 'seed':
            self.rng = (args[0] & INT64_MASK) or 0x9E3779B97F4A7C15
            return None
        if name == 'sleep_ms':
            _time.sleep(max(0, self.num(args[0], 'sleep_ms')) / 1000.0)
            return None
        if name == 'exit':
            raise TILExit(int(args[0]) if args else 0)
        if name == 'panic':
            self.panic(self.to_str(args[0]) if args else b"panic")
        if name in ('read_file', 'write_file', 'file_exists'):
            path = self.path_arg(name, args[0])
            if name == 'file_exists':
                try:
                    with open(path, 'rb'):
                        return True
                except (OSError, ValueError):
                    return False
            if name == 'read_file':
                try:
                    with open(path, 'rb') as f:
                        return f.read()
                except (OSError, ValueError):
                    self.panic(b"cannot read file: " + args[0])
            data = args[1] if type(args[1]) is bytes else self.to_str(args[1])
            try:
                with open(path, 'wb') as f:
                    f.write(data)
            except (OSError, ValueError):
                self.panic(b"cannot write file: " + args[0])
            return None
        if name == 'args':
            return list(self.argv)
        if name == 'bytes_to_str':
            arr = args[0]
            if type(arr) is not list or any(type(x) is not int or not 0 <= x <= 255 for x in arr):
                self.panic("bytes_to_str() expects an array of ints 0..255")
            return bytes(arr)
        if name == 'str_bytes':
            if type(args[0]) is not bytes:
                self.panic(f"str_bytes() expects a string, got {self.type_name(args[0]).decode()}")
            return list(args[0])
        if name == 'energy_mode':
            prev = self.energy
            if args:
                m = args[0]
                if type(m) is not int or not ENERGY_PERFORMANCE <= m <= ENERGY_LOW:
                    self.panic("energy_mode() needs a mode 0 (performance), 1 (balanced) or 2 (low)")
                self.energy = m
            return prev
        self.panic(f"{name}() is not available in this VM build")

    def path_arg(self, who: str, v) -> str:
        """A file path argument (same rules as the C VM: non-empty, < 256 bytes, no NUL)."""
        if type(v) is not bytes:
            self.panic(f"{who}() expects a path string, got {self.type_name(v).decode()}")
        if not v or len(v) >= 256 or b"\0" in v:
            self.panic(f"{who}(): invalid path")
        return os.fsdecode(v)

    def gfx_native(self, name: str, args: list):
        """Screen and keys natives (ids 40-53): same checks, order and messages as
        native_gfx in vm/tilvm.c."""
        scr = self.screen
        if name == 'screen_width':
            return scr.w
        if name == 'screen_height':
            return scr.h
        if name in ('key_pressed', 'wait_key'):
            k = self.keys.pressed() if name == 'key_pressed' else self.keys.wait()
            if type(k) is not int or k < 0:
                self.panic("key code out of range")
            return k
        if name != 'screen_init' and not scr.w:
            self.panic("screen not initialized (call screen_init)")
        kinds = GFX_KINDS[name]
        for k, v in enumerate(args[:len(kinds)]):
            kind = kinds[k]
            if kind == 't':
                continue
            if type(v) is not int:
                self.panic(f"{name}() argument {k + 1} must be an int, got {self.type_name(v).decode()}")
            if kind == 's':                         # screen_init: the device's largest screen
                most = getattr(scr, 'max_h' if k else 'max_w', GFX_MAX_SIZE)
                if not 1 <= v <= most:
                    self.panic(f"{name}() argument {k + 1} must be 1..{most}, got {v}")
            if kind == 'c' and not GFX_COORD_MIN <= v <= GFX_COORD_MAX:
                self.panic(f"{name}() argument {k + 1} must be -32768..32767, got {v}")
            if kind == 'k' and not 0 <= v <= 0xFFFFFF:
                self.panic(f"{name}() argument {k + 1} must be a color 0..0xFFFFFF, got {v}")
        w, h = scr.w, scr.h
        if name == 'screen_init':
            scr.init(args[0], args[1])
        elif name == 'screen_clear':
            scr.fill(0, 0, w, h, args[0] if args else 0)
        elif name == 'draw_pixel':
            if 0 <= args[0] < w and 0 <= args[1] < h:
                scr.pixel(args[0], args[1], args[2])
        elif name == 'get_pixel':
            return scr.get(args[0], args[1]) if 0 <= args[0] < w and 0 <= args[1] < h else 0
        elif name in ('draw_rect', 'fill_rect'):
            x, y, rw, rh, c = args[:5]
            if rw > 0 and rh > 0:
                if name == 'fill_rect':
                    scr.fill(x, y, x + rw, y + rh, c)
                else:
                    scr.fill(x, y, x + rw, y + 1, c)                    # top
                    scr.fill(x, y + rh - 1, x + rw, y + rh, c)          # bottom
                    scr.fill(x, y + 1, x + 1, y + rh - 1, c)            # left
                    scr.fill(x + rw - 1, y + 1, x + rw, y + rh - 1, c)  # right
        elif name == 'screen_show':
            scr.show(self.write)
            if scr.ansi:
                self.flush()
        else:
            if name == 'draw_line':
                points, c = gfx_line_points(*args[:4]), args[4]
            elif name == 'draw_circle':
                points, c = gfx_circle_points(*args[:3]), args[3]
            else:                                                       # draw_text
                text = args[2] if type(args[2]) is bytes else self.to_str(args[2])
                points, c = gfx_text_points(args[0], args[1], text), args[3]
            plot = scr.pixel
            for x, y in points:
                if 0 <= x < w and 0 <= y < h:
                    plot(x, y, c)
        return None

    # ── builtin methods ─────────────────────────────────────────────────────
    def call_method(self, mid: int, recv, args: list):
        name = METHODS[mid] if mid < len(METHODS) else '?'
        argc = len(args)
        tr = type(recv)
        if name == 'to_string':
            return self.to_str(recv)
        if tr is bytes:
            s = recv
            if name == 'len':
                return len(s)
            if name in ('contains', 'starts_with', 'ends_with', 'find', 'index_of', 'count',
                        'eq', 'split', 'replace') and argc and type(args[0]) is not bytes:
                self.panic(f"str.{name} expects a string argument")
            if name == 'contains':
                return args[0] in s
            if name == 'starts_with':
                return s.startswith(args[0])
            if name == 'ends_with':
                return s.endswith(args[0])
            if name == 'trim':
                return s.strip(b' \t\r\n')
            if name == 'to_upper':
                return s.upper()
            if name == 'to_lower':
                return s.lower()
            if name == 'replace':
                return s if not args[0] else s.replace(args[0], self.to_str(args[1]))
            if name == 'slice':
                n = len(s)
                a = max(0, min(args[0], n))
                b = max(0, min(args[1] if argc > 1 else n, n))
                return s[a:b] if a < b else b""
            if name in ('find', 'index_of'):
                return s.find(args[0])
            if name == 'count':
                return s.count(args[0]) if args[0] else 0
            if name == 'eq':
                return s == args[0]
            if name == 'split':
                if not argc:
                    return s.split()
                return s.split(args[0]) if args[0] else [bytes([c]) for c in s]
            if name == 'repeat':
                return s * max(args[0], 0)
            if name == 'to_int':
                return self.cast(0, s)
            if name == 'to_float':
                return self.cast(1, s)
            if name == 'is_empty':
                return len(s) == 0
            if name == 'get':
                return self.index_get(s, args[0])
        elif tr is list:
            a = recv
            if name == 'len':
                return len(a)
            if name == 'push':
                a.append(args[0])
                return None
            if name == 'pop':
                if not a:
                    self.panic("pop() from empty array")
                return a.pop()
            if name == 'get':
                return self.index_get(a, args[0])
            if name == 'set':
                self.index_set(a, args[0], args[1])
                return None
            if name == 'contains':
                return any(self.equal(x, args[0]) for x in a)
            if name == 'index_of':
                for i, x in enumerate(a):
                    if self.equal(x, args[0]):
                        return i
                return -1
            if name == 'count':
                return sum(1 for x in a if self.equal(x, args[0]))
            if name == 'insert':
                i = args[0]
                if type(i) is not int or not 0 <= i <= len(a):
                    self.panic(f"insert index out of bounds: {i}")
                a.insert(i, args[1])
                return None
            if name == 'remove':
                i = args[0]
                if type(i) is not int or not 0 <= i < len(a):
                    self.panic(f"remove index out of bounds: {i}")
                return a.pop(i)
            if name == 'reverse':
                a.reverse()
                return None
            if name == 'sort':
                self.sort_array(a)
                return None
            if name == 'join':
                sep = args[0] if argc else b""
                return self.to_str(sep).join(self.to_str(x) for x in a)
            if name == 'slice':
                n = len(a)
                lo = max(0, min(args[0], n))
                hi = max(0, min(args[1] if argc > 1 else n, n))
                return a[lo:hi] if lo < hi else []
            if name == 'first':
                return a[0] if a else None
            if name == 'last':
                return a[-1] if a else None
            if name == 'clear':
                a.clear()
                return None
            if name == 'is_empty':
                return len(a) == 0
            if name == 'copy':
                return list(a)
            if name in ('sum', 'min', 'max'):
                return self.call_native(NATIVE_ID[name], [a])
        elif tr is dict:
            d = recv
            if name == 'len':
                return len(d)
            if name == 'set':
                d[self.map_key(args[0])] = args[1]
                return None
            if name == 'get':
                return self.index_get(d, args[0])
            if name == 'get_or':
                return d.get(self.map_key(args[0]), args[1])
            if name in ('has', 'contains'):
                return self.map_key(args[0]) in d
            if name in ('delete', 'remove'):
                d.pop(self.map_key(args[0]), None)
                return None
            if name == 'keys':
                return list(d.keys())
            if name == 'values':
                return list(d.values())
            if name == 'is_empty':
                return len(d) == 0
            if name == 'clear':
                d.clear()
                return None
            if name == 'copy':
                return dict(d)
        elif recv is None or (tr is VMStruct and recv.t <= TYPE_ERR):
            kind = -1 if recv is None else recv.t
            ok = kind in (TYPE_SOME, TYPE_OK)
            if name in ('is_some', 'is_ok'):
                return ok
            if name in ('is_none', 'is_err'):
                return not ok
            if name == 'unwrap':
                if ok:
                    return recv.f[0]
                self.panic(b"unwrap() called on None" if recv is None else
                           b"unwrap() called on Err: " + self.to_str(recv.f[0]))
            if name == 'expect':
                if ok:
                    return recv.f[0]
                self.panic(self.to_str(args[0]))
            if name == 'unwrap_or':
                return recv.f[0] if ok else args[0]
            if name == 'error':
                return recv.f[0] if kind == TYPE_ERR else None
        self.panic(f"no method '{name}' on {self.type_name(recv).decode()}")

    def sort_array(self, a: list):
        if not a:
            return
        if all(type(x) in (int, float) for x in a) or all(type(x) is bytes for x in a):
            a.sort()
        else:
            self.panic("sort() needs an array of numbers or of strings")

    # ── execution ───────────────────────────────────────────────────────────
    def line_for(self, pc: int) -> int:
        lines = self.m.lines
        lo, hi, best = 0, len(lines) - 1, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if lines[mid][0] <= pc:
                best = lines[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def func_at(self, pc: int) -> str:
        best = None
        for f in self.m.funcs:
            if f.offset <= pc and (best is None or f.offset > best.offset):
                best = f
        return best.name if best else "?"

    def traceback(self, pc: int) -> bytes:
        parts = []
        line = self.line_for(pc)
        parts.append(f"  at {self.func_at(pc)}" + (f" (line {line})" if line else ""))
        for fr in reversed(self.frames[:-1]):
            ln = self.line_for(fr[1] - 1)
            parts.append(f"  at {self.m.funcs[fr[0]].name}" + (f" (line {ln})" if ln else ""))
        return ("\n".join(parts[:8])).encode()

    def start(self):
        """Set up the entry frame (once; load_portable() sets up a saved program instead)."""
        if self.started:
            return
        self.started = True
        if self.frames:                   # set up by the caller (til debug)
            return
        entry = self.m.funcs[self.m.entry]
        self.frames = [[self.m.entry, entry.offset, 0, None, 0]]
        self.stack = [None] * entry.n_locals

    def run(self) -> int:
        """Run the module (to the end, also after run_steps() paused it); returns
        the process exit code."""
        if self.status is not None:
            return self.exit_code
        self.start()
        try:
            return self._run()
        finally:
            self.flush()

    def run_steps(self, budget: int = 0) -> int:
        """Run at most `budget` steps (loop back-edges and calls, counted exactly as
        the C VM's tilvm_run_steps() counts them; 0 = no limit). Returns DONE, ERROR,
        EXIT (see self.exit_code) or YIELD: paused after exactly `budget` steps,
        at an instruction boundary - call again to continue, or save_portable()."""
        if self.status is not None:
            return self.status
        self.start()
        self.pause_at = self.steps + budget if budget > 0 else 0
        try:
            self._run()
            return self.status
        except TILPause:
            return self.YIELD
        finally:
            self.pause_at = 0
            self.flush()

    def _end(self, status: int, code: int) -> int:
        self.status, self.exit_code = status, code
        return code

    def _run(self) -> int:
        try:
            while True:
                try:
                    self.execute()
                    return self._end(self.DONE, 0)
                except TILPanic as p:
                    if not self.recover_from(p):
                        self.report(b"runtime error: " + p.msg, p.pc)
                        return self._end(self.ERROR, 1)
        except TILFatal as f:
            self.report(b"fatal error: " + f.msg, f.pc)
            return self._end(self.ERROR, 1)
        except TILExit as e:
            return self._end(self.EXIT, e.code)
        except RecursionError:
            self.report(b"fatal error: native recursion limit", -1)
            return self._end(self.ERROR, 1)

    def report(self, msg: bytes, pc: int):
        out = getattr(sys.stderr, 'buffer', None)
        text = msg + b"\n" + (self.traceback(pc) + b"\n" if pc >= 0 else b"")
        self.flush()                     # the program's output first: translation never holds it back
        text = localize_report_bytes(text)   # the CLI's message language (programs still see English)
        if out is not None:
            out.write(text)
            out.flush()
        else:
            sys.stderr.write(text.decode('utf-8', 'replace'))

    def recover_from(self, p: TILPanic) -> bool:
        if not self.recover:
            return False
        if p.target is not None:
            for rf in self.recover[p.target + 1:]:
                if rf[3]:
                    self.deadlines -= 1
            del self.recover[p.target + 1:]
        handler, n_frames, sp, deadline, n_handlers = self.recover.pop()
        if deadline:
            self.deadlines -= 1
        del self.frames[n_frames:]
        del self.stack[sp:]
        del self.handlers[n_handlers:]     # handles / running clauses inside the block
        self.stack.append(p.msg)
        self.frames[-1][1] = handler
        return True

    # ── effect handlers (TIL 3.1, docs/TIL_EFFECTS.md) ──────────────────────
    EFFECT_NAME_MAX = 64      # effect names in error messages (same cut in the C VM)

    def find_clause(self, name: bytes):
        """Innermost visible clause for effect `name` -> (handler index, clause value).
        While a clause runs, its marker hides the owning handler and everything
        installed after it, so the search continues below the owner."""
        hs = self.handlers
        stack = self.stack
        i = len(hs) - 1
        while i >= 0:
            h = hs[i]
            if h[0] == 1:
                i = h[5] - 1
                continue
            sp = h[2]
            for j in range(h[5]):
                if stack[sp + 2 * j + 1] == name:
                    return i, stack[sp + 2 * j]
            i -= 1
        return None

    def pop_recover_to(self, n: int):
        while len(self.recover) > n:
            if self.recover.pop()[3]:
                self.deadlines -= 1

    def leave_frame(self, depth: int, fidx: int, result) -> bool:
        """RETURN with handler records present: drop the handles the frame installed.
        Returning from an effect clause (it finished without resume) aborts the
        handle that owns it: its value becomes the value of the handle expression.
        Returns True after such an abort (the VM continues at the handle's end)."""
        hs = self.handlers
        while hs and hs[-1][1] >= depth:
            hs.pop()
        if not self.m.funcs[fidx].flags & TBCFunction.FLAG_EFFECT_CLAUSE:
            return False
        if not hs or hs[-1][0] != 1 or hs[-1][1] != depth - 1:
            raise TILFatal("effect clause returned outside its handler")
        owner = hs[-1][5]
        rec = hs[owner]
        del hs[owner:]
        self.pop_recover_to(rec[3])
        del self.frames[rec[1]:]
        del self.stack[rec[2]:]
        self.stack.append(result)
        self.frames[-1][1] = rec[4]
        return True

    def check_budget(self, pc: int):
        """One step (a loop back-edge or a call). `pc` = where the program goes on
        (after the jump / at the called function's start): a pause saves it."""
        self.steps += 1
        if self._slow:                  # a step limit or energy saving (see max_steps, energy)
            steps = self.steps
            if self._max_steps and steps > self._max_steps:
                raise TILFatal(f"step limit exceeded ({self._max_steps})")
            if not steps & 511 and (self._energy or self.low_funcs):
                self.energy_point()
        if self.deadlines:
            self.tick += 1
            if self.tick >= self.CHECK_INTERVAL:
                self.tick = 0
                now = _time.monotonic() * 1000.0
                for i, rf in enumerate(self.recover):
                    if rf[3] and now > rf[3]:
                        if self.pause_at:
                            self.pause_at += 1     # as in C: a timeout's step is not part of the budget
                        raise TILPanic(b"timeout", target=i)
        if self.steps == self.pause_at:
            self.frames[-1][1] = pc
            raise TILPause()

    # ── portable snapshots (.tilp, TIL 3.1; docs/TIL_SNAPSHOTS.md) ──────────
    # The byte layout, the traversal order and the canonical encodings are
    # specified there; vm/tilvm.c (tilvm_save_portable / tilvm_load_portable)
    # writes and checks exactly the same, so both VMs write the same bytes.

    def program_bytes(self) -> bytes:
        """The .tbc file this VM runs: snapshots name the program by its hash."""
        if self._tbc is None:
            raw = getattr(self.m, 'raw', None)
            self._tbc = raw if raw is not None else self.m.serialize()
        return self._tbc

    def program_hash(self) -> int:
        h = getattr(self, '_tbc_hash', None)
        if h is None:
            h = self._tbc_hash = fnv1a32(self.program_bytes())
        return h

    def save_portable(self) -> bytes:
        """The paused program as a portable snapshot: after run_steps() returned
        YIELD, or before the program started (step 0)."""
        if self.status is not None:
            raise SnapshotError(0, "the program has ended: there is nothing to save")
        self.program_hash()               # serialize() first: it may add name constants
        self.start()
        import zlib
        pack = _struct.pack
        out = bytearray()
        consts: Dict[bytes, int] = {}
        for i, c in enumerate(self.consts):
            if type(c) is bytes and c not in consts:
                consts[c] = i             # strings equal to a constant: its lowest index
        ids: Dict[int, int] = {}          # id(object) -> object id
        strs: Dict[bytes, int] = {}       # string bytes -> object id (one object per content)
        objs: list = []                   # objects in id order (also keeps them alive)

        def ref(o) -> int:
            k = ids.get(id(o))
            if k is None:
                objs.append(o)
                k = ids[id(o)] = len(objs)
            return k

        def value(v):
            if v is None:
                out.append(TILP_NIL)
            elif v is False:
                out.append(TILP_FALSE)
            elif v is True:
                out.append(TILP_TRUE)
            else:
                t = type(v)
                if t is int:
                    out.append(TILP_INT)
                    out.extend(pack('<q', v))
                elif t is float:
                    out.append(TILP_FLOAT)
                    out.extend(pack('<d', v))
                elif t is bytes:
                    k = consts.get(v)
                    if k is not None:
                        out.append(TILP_CONST)
                        out.extend(pack('<H', k))
                        return
                    k = strs.get(v)
                    if k is None:
                        objs.append(v)
                        k = strs[v] = len(objs)
                    out.append(TILP_REF)
                    out.extend(pack('<I', k))
                elif t is VMClosure and not v.up:
                    out.append(TILP_FUNC)
                    out.extend(pack('<H', v.fn))
                elif t is list or t is dict or t is VMStruct or t is VMClosure or t is VMCell:
                    out.append(TILP_REF)
                    out.extend(pack('<I', ref(v)))
                elif v is _UNDEF:
                    out.append(TILP_UNDEF)
                else:
                    raise SnapshotError(0, f"cannot save a value of type {t.__name__}")

        try:
            out += TILP_MAGIC + bytes((TILP_VERSION, 0)) + pack('<I', self.program_hash())
            # state: steps, rng, tick, energy mode (design §11), reserved
            out += pack('<IQHBB', self.steps & 0xFFFFFFFF, self.rng & INT64_MASK, self.tick, self.energy, 0)
            out += pack('<H', len(self.globals))
            out += pack('<H', len(self.frames))
            for func, pc, base, clos, extra in self.frames:
                out += pack('<HIHBI', func, pc, base, extra, ref(clos) if clos is not None and clos.up else 0)
            now = _time.monotonic() * 1000.0
            out += pack('<H', len(self.recover))
            for handler, n_frames, sp, deadline, n_handlers in self.recover:
                rem = deadline - now                    # nan / inf deadlines: never (the maximum)
                left = 0 if not deadline else 1 if rem < 1 else int(_math.ceil(rem)) if rem < 0xFFFFFFFF \
                    else 0xFFFFFFFF
                out += pack('<IIHHH', handler, left, n_frames, sp, n_handlers)
            out += pack('<H', len(self.handlers))
            for kind, n_frames, sp, n_try, abort_pc, n in self.handlers:
                out += pack('<BHHHIH', kind, n_frames, sp, n_try, abort_pc, n)
            for v in self.globals:
                value(v)
            out += pack('<H', len(self.stack))
            for v in self.stack:
                value(v)
            out += pack('<HH', self.screen.w, self.screen.h)
            k = 0
            while k < len(objs):              # breadth-first: objects first referenced here
                o = objs[k]                   # get the next ids
                k += 1
                t = type(o)
                if t is bytes:
                    out.append(TILP_STR)
                    out += pack('<I', len(o))
                    out += o
                elif t is list:
                    out.append(TILP_ARRAY)
                    out += pack('<I', len(o))
                    for x in o:
                        value(x)
                elif t is dict:
                    out.append(TILP_MAP)
                    out += pack('<I', len(o))
                    for x, y in o.items():
                        value(x)
                        value(y)
                elif t is VMStruct:
                    out.append(TILP_STRUCT)
                    out += pack('<HH', o.t, len(o.f))
                    for x in o.f:
                        value(x)
                elif t is VMClosure:
                    out.append(TILP_CLOSURE)
                    out += pack('<HH', o.fn, len(o.up))
                    for x in o.up:
                        value(x)
                else:
                    out.append(TILP_CELL)
                    value(o.v)
            out += pack('<I', len(objs))
        except _struct.error as e:            # a count or index the format cannot hold
            raise SnapshotError(0, f"the program's state cannot be written as a snapshot ({e})")
        out += pack('<I', zlib.crc32(out) & 0xFFFFFFFF)
        return bytes(out)

    def _check_snapshot_depths(self, frames, recover, handlers, depth: int):
        """A snapshot's call frames, recovery points and handler records must sit
        where the bytecode puts them (docs/TIL_SNAPSHOTS.md 3.4): every frame's pc
        has a known operand stack depth (code_depths); a frame below the top is
        at a call whose result slot is the next frame's base minus its callee slot
        (CALL: the called function, no callee slot; CALL_VALUE: a callee slot;
        PERFORM: no callee slot, a clause or the default function); the top frame's
        depth ends at the stack depth; a recovery point's handler and a handle's
        abort pc are where its saved stack depth plus the pushed message / value
        is. So a forged file cannot make the VM pop below a frame or return into
        the wrong slot. -4 otherwise (vm/tilvm.c checks the same)."""
        m, code, funcs = self.m, self.m.code, self.m.funcs
        cache: Dict[int, Dict[int, tuple]] = {}

        def at(fi: int, pc: int):
            func = frames[fi][0]
            known = cache.get(func)
            if known is None or pc not in known and pc > max(known):
                known = cache[func] = code_depths(code, funcs[func].offset, pc if known is None
                                                  else max(pc, max(known)))
            d, prev = known.get(pc, (None, -1))
            if d is None:
                raise SnapshotError(-4, "a frame, recovery point or handler is not at a known stack depth")
            return frames[fi][2] + funcs[func].n_locals + d, prev

        n_frames = len(frames)
        for i in range(n_frames):
            top, prev = at(i, frames[i][1])
            if i == n_frames - 1:
                if top != depth:
                    raise SnapshotError(-4, "bad frame")
                continue
            callee, base, extra = frames[i + 1][0], frames[i + 1][2], frames[i + 1][4]
            op = code[prev] if prev >= 0 else -1
            ok = base - extra == top - 1 and (
                (op == Op.CALL and extra == 0 and callee == code[prev + 1] | (code[prev + 2] << 8)) or
                (op == Op.CALL_VALUE and extra == 1) or op == Op.INVOKE or
                (op == Op.PERFORM and extra == 0 and
                 (callee == code[prev + 3] | (code[prev + 4] << 8) or
                  funcs[callee].flags & TBCFunction.FLAG_EFFECT_CLAUSE)))
            if not ok:
                raise SnapshotError(-4, "bad frame")
        for handler, nf, sp, _left, _nh in recover:
            if at(nf - 1, handler)[0] != sp + 1:
                raise SnapshotError(-4, "bad recovery point")
        for kind, nf, sp, _n_try, abort_pc, _n in handlers:
            if kind == 0:
                ok = at(nf - 1, abort_pc)[0] == sp + 1
            else:
                prev = at(nf - 1, frames[nf - 1][1])[1]
                ok = code[prev] == Op.PERFORM and bool(funcs[frames[nf][0]].flags & TBCFunction.FLAG_EFFECT_CLAUSE)
            if not ok:
                raise SnapshotError(-4, "bad handler record")

    def load_portable(self, data: bytes):
        """Continue a program saved by save_portable() or by the C VM's
        tilvm_save_portable(): call after construction, before running (then
        run() or run_steps()). Raises SnapshotError (.code as in C)."""
        if self.started:
            raise SnapshotError(0, "load_portable() must come before the program runs")
        import zlib
        data = bytes(data)
        unpack = _struct.unpack_from
        m = self.m
        funcs, consts, code_size = m.funcs, self.consts, len(m.code)
        pos = 0
        n_ids = 0                          # object ids assigned so far

        def take(fmt: str, size: int):
            nonlocal pos
            if pos + size > len(data):
                raise SnapshotError(-4, "truncated")
            v = unpack(fmt, data, pos)
            pos += size
            return v

        def new_id(i: int) -> _PRef:
            nonlocal n_ids
            if i == 0 or i > n_ids + 1:
                raise SnapshotError(-4, f"object id {i} out of order")
            if i > n_ids:
                n_ids = i
            return _PRef(i)

        func_values: Dict[int, VMClosure] = {}

        def value():
            nonlocal pos
            if pos >= len(data):
                raise SnapshotError(-4, "truncated")
            tag = data[pos]
            pos += 1
            if tag <= TILP_TRUE:
                return None if tag == TILP_NIL else tag == TILP_TRUE
            if tag == TILP_INT:
                return take('<q', 8)[0]
            if tag == TILP_FLOAT:
                return take('<d', 8)[0]
            if tag == TILP_CONST:
                k = take('<H', 2)[0]
                if k >= len(consts) or type(consts[k]) is not bytes:
                    raise SnapshotError(-4, f"constant {k} is not a string constant")
                return consts[k]
            if tag == TILP_REF:
                return new_id(take('<I', 4)[0])
            if tag == TILP_FUNC:
                k = take('<H', 2)[0]
                if k >= len(funcs):
                    raise SnapshotError(-4, f"function {k} out of range")
                f = func_values.get(k)
                if f is None:
                    f = func_values[k] = VMClosure(k)
                return f
            if tag == TILP_UNDEF:
                return _UNDEF
            raise SnapshotError(-4, f"bad value tag {tag}")

        # header (checked in the order of tilvm_load_portable: the same error codes)
        if len(data) < 4:
            raise SnapshotError(-4, "truncated")
        if data[:4] != TILP_MAGIC:
            raise SnapshotError(-1)
        pos = 4
        version, flags = take('<BB', 2)
        if version != TILP_VERSION or flags != 0:
            raise SnapshotError(-2, f"version {version}, flags {flags}")
        if take('<I', 4)[0] != self.program_hash():
            raise SnapshotError(-3)
        steps, rng, tick, energy, _reserved = take('<IQHBB', 16)
        if energy > 2 or tick > 255:
            raise SnapshotError(-4, "bad state")
        if take('<H', 2)[0] != len(self.globals):
            raise SnapshotError(-4, "global count")
        # frames, recovery points, handler records
        n_frames = take('<H', 2)[0]
        if n_frames == 0:
            raise SnapshotError(-4, "no frames")
        if n_frames > self.MAX_FRAMES:
            raise SnapshotError(-5, f"{n_frames} call frames, at most {self.MAX_FRAMES}")
        frames = []
        for _ in range(n_frames):
            func, pc, base, extra, clos = take('<HIHBI', 13)
            if func >= len(funcs) or pc >= code_size or extra > 1:
                raise SnapshotError(-4, "bad frame")
            frames.append([func, pc, base, new_id(clos) if clos else None, extra])
        recover = []
        for _ in range(take('<H', 2)[0]):
            handler, left, nf, sp, nh = take('<IIHHH', 14)
            if handler >= code_size or not 1 <= nf <= n_frames or (recover and nf < recover[-1][1]):
                raise SnapshotError(-4, "bad recovery point")
            recover.append([handler, nf, sp, left, nh])
        n_handlers = take('<H', 2)[0]
        if n_handlers > self.MAX_HANDLERS:
            raise SnapshotError(-5, f"{n_handlers} effect handler entries, at most {self.MAX_HANDLERS}")
        handlers: List[list] = []
        for i in range(n_handlers):
            kind, nf, sp, n_try, abort_pc, n = take('<BHHHIH', 13)
            if kind > 1 or nf > n_frames or n_try > len(recover) or (handlers and nf < handlers[-1][1]):
                raise SnapshotError(-4, "bad handler record")
            if kind == 0 and (nf == 0 or abort_pc >= code_size):
                raise SnapshotError(-4, "bad handler record")
            if kind == 1 and (nf == 0 or nf >= n_frames or n >= i or handlers[n][0] != 0 or sp or abort_pc):
                raise SnapshotError(-4, "bad handler record")
            handlers.append([kind, nf, sp, n_try, abort_pc, n])
        # values
        globals_ = [value() for _ in range(len(self.globals))]
        depth = take('<H', 2)[0]
        stack = [value() for _ in range(depth)]
        for i, (func, _pc, base, _clos, extra) in enumerate(frames):
            # the locals of a frame lie below the next frame's arguments (and its callee slot)
            top = frames[i + 1][2] - frames[i + 1][4] if i + 1 < n_frames else depth
            if base < extra or (i == 0 and base) or base + funcs[func].n_locals > top:
                raise SnapshotError(-4, "bad frame")
        for rec in recover:
            if rec[2] > depth or rec[4] > n_handlers:
                raise SnapshotError(-4, "bad recovery point")
        for h in handlers:
            if h[0] == 0 and h[2] + 2 * h[5] > depth:
                raise SnapshotError(-4, "bad handler record")
        self._check_snapshot_depths(frames, recover, handlers, depth)
        width, height = take('<HH', 4)
        if (width == 0) != (height == 0) or width > GFX_MAX_SIZE or height > GFX_MAX_SIZE:
            raise SnapshotError(-4, "bad screen size")
        if width > self.screen.max_w or height > self.screen.max_h:
            raise SnapshotError(-8, f"{width}x{height}")
        # the object table: build every object, then resolve the references
        objs: list = []
        fill = []
        while len(objs) < n_ids:
            kind = take('<B', 1)[0]
            if kind == TILP_STR:
                n = take('<I', 4)[0]
                if pos + n > len(data):
                    raise SnapshotError(-4, "truncated")
                objs.append(data[pos:pos + n])
                pos += n
                continue
            if kind == TILP_ARRAY:
                o = []
                raw = [value() for _ in range(take('<I', 4)[0])]
            elif kind == TILP_MAP:
                o = {}
                raw = [(value(), value()) for _ in range(take('<I', 4)[0])]
            elif kind == TILP_STRUCT:
                t, n = take('<HH', 4)
                if t >= len(m.types) or n != len(m.types[t][1]):
                    raise SnapshotError(-4, f"bad struct (type {t}, {n} fields)")
                o = VMStruct(t, [])
                raw = [value() for _ in range(n)]
            elif kind == TILP_CLOSURE:
                fn, n = take('<HH', 4)
                if fn >= len(funcs) or n != funcs[fn].n_upvals:
                    raise SnapshotError(-4, f"bad closure (function {fn}, {n} upvalues)")
                o = VMClosure(fn)
                raw = [value() for _ in range(n)]
            elif kind == TILP_CELL:
                o = VMCell(None)
                raw = value()
            else:
                raise SnapshotError(-4, f"bad object kind {kind}")
            objs.append(o)
            fill.append((o, raw))
        if take('<I', 4)[0] != n_ids:
            raise SnapshotError(-4, "object count")
        if take('<I', 4)[0] != zlib.crc32(data[:pos - 4]) & 0xFFFFFFFF or pos != len(data):
            raise SnapshotError(-4, "checksum")

        def res(v):
            return objs[v.i - 1] if type(v) is _PRef else v

        for o, raw in fill:
            t = type(o)
            if t is list:
                o.extend([res(x) for x in raw])
            elif t is dict:
                for k, v in raw:
                    k = res(k)
                    if not (k is None or type(k) in (bool, int, float, bytes)):
                        raise SnapshotError(-4, "bad map key")
                    o[k] = res(v)
            elif t is VMStruct:
                o.f = [res(x) for x in raw]
            elif t is VMClosure:
                o.up = tuple(res(x) for x in raw)
                if any(type(c) is not VMCell for c in o.up):
                    raise SnapshotError(-4, "an upvalue is not a cell")
            else:
                o.v = res(raw)
        for fr in frames:
            if fr[3] is not None:
                fr[3] = res(fr[3])
                if type(fr[3]) is not VMClosure or fr[3].fn != fr[0]:
                    raise SnapshotError(-4, "a frame's closure is not a closure of its function")
        # commit
        now = _time.monotonic() * 1000.0
        self.steps = steps
        self.rng = rng or 0x9E3779B97F4A7C15
        self.tick = tick
        self.energy = energy                     # energy mode (design §11)
        self.globals = [res(v) for v in globals_]
        self.stack = [res(v) for v in stack]
        self.frames = frames
        self.recover = [[r[0], r[1], r[2], now + r[3] if r[3] else 0.0, r[4]] for r in recover]
        self.deadlines = sum(1 for r in recover if r[3])
        self.handlers = handlers
        if width:
            self.screen.init(width, height)
        self.started = True

    def energy_point(self):
        """At a step that is a multiple of 512 (as energy_point in vm/tilvm.c): idle
        1 ms if the energy mode asks for it - mode 2, or mode 1 at a multiple of
        4096; mode 2 while a frame of an #[energy: low] function is active."""
        mode = self._energy
        if mode != ENERGY_LOW and self.low_funcs:
            low = self.low_funcs
            for fr in reversed(self.frames):
                if fr[0] in low:
                    mode = ENERGY_LOW
                    break
        if mode == ENERGY_LOW or (mode == ENERGY_BALANCED and not self.steps & 4095):
            self.idle_count += 1
            self.idle(1)

    def idle(self, ms: int):
        """Energy saving: give the CPU away for `ms` milliseconds (C VM: vm->host.idle)."""
        _time.sleep(ms / 1000.0)

    def binop(self, op: int, a, b):
        """Binary operator `op` (ADD..GE) exactly as its plain instruction computes it
        (used by the <op>_I / <op>_L superinstructions)."""
        if op <= Op.MUL:
            if type(a) is int and type(b) is int:
                r = a + b if op == Op.ADD else (a - b if op == Op.SUB else a * b)
                if not -0x8000000000000000 <= r <= 0x7FFFFFFFFFFFFFFF:
                    r = wrap_int64(r)
                return r
            return self.arith(op, a, b)
        if op <= Op.POW:
            return self.arith(op, a, b)
        if op == Op.EQ:
            return self.equal(a, b)
        if op == Op.NE:
            return not self.equal(a, b)
        if Op.LT <= op <= Op.GE:
            if type(a) is int and type(b) is int:
                return (a < b if op == Op.LT else a <= b if op == Op.LE else
                        a > b if op == Op.GT else a >= b)
            return self.compare(op, a, b)
        return self.bitwise(op, a, b)

    def cmp_values(self, cmp: int, a, b) -> bool:
        """a <cmp> b for the compare-and-branch superinstructions (cmp = EQ..GE)."""
        if cmp == Op.EQ:
            return self.equal(a, b)
        if cmp == Op.NE:
            return not self.equal(a, b)
        if not Op.LT <= cmp <= Op.GE:
            raise TILFatal(f"bad opcode (comparison 0x{cmp:02x})")
        if type(a) is int and type(b) is int:
            return a < b if cmp == Op.LT else a <= b if cmp == Op.LE else a > b if cmp == Op.GT else a >= b
        return self.compare(cmp, a, b)

    def execute(self):
        code = self.code
        consts = self.consts
        stack = self.stack
        frames = self.frames
        funcs = self.m.funcs
        handlers = self.handlers          # effect handler entries (changed in place only)
        frame = frames[-1]
        pc = frame[1]
        base = frame[2]
        closure = frame[3]
        op_pc = pc
        binops = FUSED_BINOPS
        # opcodes as locals: the dispatch chain below compares against them (fast)
        LOAD_LOCAL, STORE_LOCAL, PUSH_I8, PUSH_CONST = Op.LOAD_LOCAL, Op.STORE_LOCAL, Op.PUSH_I8, Op.PUSH_CONST
        ADD, SUB, MUL, LT, LE, GT, GE = Op.ADD, Op.SUB, Op.MUL, Op.LT, Op.LE, Op.GT, Op.GE
        JUMP, JUMP_IF_FALSE, JUMP_IF_TRUE = Op.JUMP, Op.JUMP_IF_FALSE, Op.JUMP_IF_TRUE
        CALL, CALL_VALUE, INVOKE, RETURN, RETURN_NIL = Op.CALL, Op.CALL_VALUE, Op.INVOKE, Op.RETURN, Op.RETURN_NIL
        PERFORM = Op.PERFORM
        LOAD_LOCAL2, LOAD_LOCAL_FIELD, INC_LOCAL, CAST_LOCAL = \
            Op.LOAD_LOCAL2, Op.LOAD_LOCAL_FIELD, Op.INC_LOCAL, Op.CAST_LOCAL
        CAST_RETURN, ADD_STORE_LOCAL, NATIVE_POP, RETURN_LOCAL = \
            Op.CAST_RETURN, Op.ADD_STORE_LOCAL, Op.NATIVE_POP, Op.RETURN_LOCAL
        INDEX_L, INDEX_LL, FOR_ITER_STORE = Op.INDEX_L, Op.INDEX_LL, Op.FOR_ITER_STORE
        NATIVE, FOR_ITER, LOAD_UPVAL, GET_INDEX, POP = Op.NATIVE, Op.FOR_ITER, Op.LOAD_UPVAL, Op.GET_INDEX, Op.POP
        LOAD_GLOBAL, STORE_GLOBAL, GET_FIELD, SET_FIELD = Op.LOAD_GLOBAL, Op.STORE_GLOBAL, Op.GET_FIELD, Op.SET_FIELD
        EQ, NE, DIV, MOD, POW, CONCAT, NEW_STRUCT = Op.EQ, Op.NE, Op.DIV, Op.MOD, Op.POW, Op.CONCAT, Op.NEW_STRUCT
        BAND, BXOR = Op.BAND, Op.BXOR
        JUMP_IF_NOT, JUMP_IF_NOT_L, JUMP_IF_NOT_I, JUMP_IF_NOT_LL, JUMP_IF_NOT_LI = \
            Op.JUMP_IF_NOT, Op.JUMP_IF_NOT_L, Op.JUMP_IF_NOT_I, Op.JUMP_IF_NOT_LL, Op.JUMP_IF_NOT_LI
        try:
            while True:
                op_pc = pc
                # @line-event: `til debug` (src/tildebug.py) runs a copy of execute() with its
                # line-event check put here; normal runs have nothing here and pay nothing
                op = code[pc]
                pc += 1
                if op == LOAD_LOCAL:
                    stack.append(stack[base + code[pc]])
                    pc += 1
                elif op == JUMP:
                    rel = code[pc] | (code[pc + 1] << 8)
                    if rel >= 0x8000:
                        pc += rel - 0xFFFE          # 2 + rel - 0x10000: a loop back-edge (a step)
                        self.check_budget(pc)
                    else:
                        pc += 2 + rel
                elif op == INC_LOCAL:
                    # LOAD_LOCAL a; PUSH_I8 k; ADD; STORE_LOCAL a
                    slot = base + code[pc]
                    k = code[pc + 1]
                    if k > 127:
                        k -= 256
                    a = stack[slot]
                    if type(a) is int:
                        r = a + k
                        if not -0x8000000000000000 <= r <= 0x7FFFFFFFFFFFFFFF:
                            r = wrap_int64(r)
                        stack[slot] = r
                    else:
                        stack[slot] = self.arith(ADD, a, k)
                    pc += 2
                elif op == JUMP_IF_NOT_LL:
                    # LOAD_LOCAL a; LOAD_LOCAL b; <cmp>; JUMP_IF_FALSE
                    cmp = code[pc]
                    a = stack[base + code[pc + 1]]
                    b = stack[base + code[pc + 2]]
                    if type(a) is int and type(b) is int and LT <= cmp <= GE:
                        t = (a < b if cmp == LT else a <= b if cmp == LE else
                             a > b if cmp == GT else a >= b)
                    else:
                        t = self.cmp_values(cmp, a, b)
                    if t:
                        pc += 5
                    else:
                        rel = code[pc + 3] | (code[pc + 4] << 8)
                        if rel >= 0x8000:
                            pc += rel - 0xFFFB      # 5 + rel - 0x10000
                            self.check_budget(pc)
                        else:
                            pc += 5 + rel
                elif op == LOAD_LOCAL_FIELD:
                    # LOAD_LOCAL a; GET_FIELD name
                    stack.append(self.get_field(stack[base + code[pc]], code[pc + 1] | (code[pc + 2] << 8)))
                    pc += 3
                elif op == ADD_STORE_LOCAL:
                    # ADD; STORE_LOCAL a
                    b = stack.pop()
                    a = stack.pop()
                    if type(a) is int and type(b) is int:
                        r = a + b
                        if not -0x8000000000000000 <= r <= 0x7FFFFFFFFFFFFFFF:
                            r = wrap_int64(r)
                    else:
                        r = self.arith(ADD, a, b)
                    stack[base + code[pc]] = r
                    pc += 1
                elif op == LOAD_LOCAL2:
                    stack.append(stack[base + code[pc]])
                    stack.append(stack[base + code[pc + 1]])
                    pc += 2
                elif op >= 0xA0:
                    # <op>_I k = PUSH_I8/PUSH_I16 k; <op>      <op>_L b = LOAD_LOCAL b; <op>
                    if op >= 0xB0:
                        if op > 0xBF:
                            raise TILFatal(f"bad opcode 0x{op:02x} at {op_pc}")
                        b = stack[base + code[pc]]
                        pc += 1
                    else:
                        b = code[pc] | (code[pc + 1] << 8)
                        if b >= 0x8000:
                            b -= 0x10000
                        pc += 2
                    a = stack[-1]
                    bop = binops[op & 15]
                    if type(a) is int and type(b) is int:
                        if bop == ADD:
                            r = a + b
                        elif bop == SUB:
                            r = a - b
                        elif bop == LT:
                            stack[-1] = a < b
                            continue
                        elif bop == MUL:
                            r = a * b
                        elif bop == MOD and a >= 0 and b > 0:
                            stack[-1] = a % b           # C and Python agree here
                            continue
                        elif bop == BAND:
                            stack[-1] = a & b
                            continue
                        elif bop == BXOR:
                            stack[-1] = a ^ b
                            continue
                        else:
                            stack[-1] = self.binop(bop, a, b)
                            continue
                        if not -0x8000000000000000 <= r <= 0x7FFFFFFFFFFFFFFF:
                            r = wrap_int64(r)
                        stack[-1] = r
                    else:
                        stack[-1] = self.binop(bop, a, b)
                elif op == STORE_LOCAL:
                    stack[base + code[pc]] = stack.pop()
                    pc += 1
                elif op == ADD or op == SUB or op == MUL:
                    b = stack.pop()
                    a = stack[-1]
                    if type(a) is int and type(b) is int:
                        r = a + b if op == ADD else (a - b if op == SUB else a * b)
                        if not -0x8000000000000000 <= r <= 0x7FFFFFFFFFFFFFFF:
                            r = wrap_int64(r)
                        stack[-1] = r
                    else:
                        stack[-1] = self.arith(op, a, b)
                elif op == PUSH_CONST:
                    stack.append(consts[code[pc] | (code[pc + 1] << 8)])
                    pc += 2
                elif op == CALL or op == CALL_VALUE or op == INVOKE:
                    if op == CALL:
                        fidx = code[pc] | (code[pc + 1] << 8)
                        argc = code[pc + 2]
                        pc += 3
                        new_closure = None
                        extra = 0
                    elif op == CALL_VALUE:
                        argc = code[pc]
                        pc += 1
                        callee = stack[-argc - 1]
                        if type(callee) is not VMClosure:
                            self.panic(f"{self.type_name(callee).decode()} is not callable")
                        fidx = callee.fn
                        new_closure = callee
                        extra = 1
                    else:
                        name_c = code[pc] | (code[pc + 1] << 8)
                        mid = code[pc + 2]
                        argc = code[pc + 3]
                        pc += 4
                        recv = stack[-argc - 1]
                        fidx = None
                        if type(recv) is VMStruct:
                            fidx = self.methods.get((recv.t, name_c))
                            if fidx is not None:
                                argc += 1          # receiver becomes `self`
                                new_closure = None
                                extra = 0
                            elif recv.t > TYPE_ERR and name_c in self.field_slots[recv.t]:
                                callee = recv.f[self.field_slots[recv.t][name_c]]
                                if type(callee) is not VMClosure:
                                    self.panic(b"field '" + consts[name_c] + b"' is not callable")
                                fidx = callee.fn
                                new_closure = callee
                                extra = 1
                        if fidx is None:
                            if mid in self.m.prelude and type(recv) is list:
                                fidx = self.m.prelude[mid]
                                argc += 1
                                new_closure = None
                                extra = 0
                            elif mid == NO_METHOD:
                                self.panic(b"no method '" + consts[name_c] + b"' on " + self.type_name(recv))
                            else:
                                args = stack[len(stack) - argc:]
                                del stack[len(stack) - argc - 1:]
                                stack.append(self.call_method(mid, recv, args))
                                continue
                    fn = funcs[fidx]
                    if argc > fn.n_params or argc < fn.n_required:
                        self.panic(f"{fn.name}() takes {fn.n_params} argument(s), got {argc}")
                    if len(frames) >= self.MAX_FRAMES:
                        raise TILFatal("stack overflow (recursion too deep)")
                    frame[1] = pc
                    new_base = len(stack) - argc
                    if argc < fn.n_params:
                        stack.extend([_UNDEF] * (fn.n_params - argc))
                    if fn.n_locals > fn.n_params:
                        stack.extend([None] * (fn.n_locals - fn.n_params))
                    frame = [fidx, fn.offset, new_base, new_closure, extra]
                    frames.append(frame)
                    pc = fn.offset
                    base = new_base
                    closure = new_closure
                    self.check_budget(pc)
                elif op == RETURN or op == RETURN_NIL or op == CAST_RETURN or op == RETURN_LOCAL:
                    if op == RETURN:
                        result = stack.pop()
                    elif op == CAST_RETURN:
                        # CAST kind; RETURN
                        result = self.cast(code[pc], stack[-1])
                        stack.pop()
                    elif op == RETURN_LOCAL:
                        # LOAD_LOCAL a; RETURN
                        result = stack[base + code[pc]]
                    else:
                        result = None
                    depth = len(frames)
                    while self.recover and self.recover[-1][1] >= depth:
                        if self.recover.pop()[3]:
                            self.deadlines -= 1
                    if handlers and self.leave_frame(depth, frame[0], result):
                        frame = frames[-1]          # a clause aborted its handle
                        pc = frame[1]
                        base = frame[2]
                        closure = frame[3]
                        continue
                    frames.pop()
                    del stack[base - frame[4]:]
                    stack.append(result)
                    if not frames:
                        return
                    frame = frames[-1]
                    pc = frame[1]
                    base = frame[2]
                    closure = frame[3]
                elif op == CAST_LOCAL:
                    # LOAD_LOCAL a; CAST kind; STORE_LOCAL a
                    slot = base + code[pc]
                    v = stack[slot]
                    if not (type(v) is int and code[pc + 1] == 0):
                        stack[slot] = self.cast(code[pc + 1], v)
                    pc += 2
                elif op == JUMP_IF_NOT_LI or op == JUMP_IF_NOT_L or op == JUMP_IF_NOT_I or op == JUMP_IF_NOT:
                    cmp = code[pc]
                    if op == JUMP_IF_NOT_LI:
                        # LOAD_LOCAL a; PUSH_I8/PUSH_I16 k; <cmp>; JUMP_IF_FALSE
                        a = stack[base + code[pc + 1]]
                        b = code[pc + 2] | (code[pc + 3] << 8)
                        if b >= 0x8000:
                            b -= 0x10000
                        pc += 4
                    elif op == JUMP_IF_NOT_L:
                        # LOAD_LOCAL b; <cmp>; JUMP_IF_FALSE
                        a = stack[-1]
                        b = stack[base + code[pc + 1]]
                        pc += 2
                    elif op == JUMP_IF_NOT_I:
                        # PUSH_I8/PUSH_I16 k; <cmp>; JUMP_IF_FALSE
                        a = stack[-1]
                        b = code[pc + 1] | (code[pc + 2] << 8)
                        if b >= 0x8000:
                            b -= 0x10000
                        pc += 3
                    else:
                        # <cmp>; JUMP_IF_FALSE
                        a = stack[-2]
                        b = stack[-1]
                        pc += 1
                    if type(a) is int and type(b) is int and LT <= cmp <= GE:
                        t = (a < b if cmp == LT else a <= b if cmp == LE else
                             a > b if cmp == GT else a >= b)
                    else:
                        t = self.cmp_values(cmp, a, b)
                    if op == JUMP_IF_NOT:
                        del stack[-2:]
                    elif op != JUMP_IF_NOT_LI:
                        stack.pop()
                    if t:
                        pc += 2
                    else:
                        rel = code[pc] | (code[pc + 1] << 8)
                        if rel >= 0x8000:
                            pc += rel - 0xFFFE
                            self.check_budget(pc)
                        else:
                            pc += 2 + rel
                elif op == PUSH_I8:
                    v = code[pc]
                    stack.append(v - 256 if v > 127 else v)
                    pc += 1
                elif op == NATIVE or op == NATIVE_POP:
                    nid, argc = code[pc], code[pc + 1]
                    pc += 2
                    args = stack[len(stack) - argc:] if argc else []
                    if argc:
                        del stack[len(stack) - argc:]
                    frame[1] = pc
                    v = self.call_native(nid, args)
                    if op != NATIVE_POP:
                        stack.append(v)
                elif op == INDEX_LL:
                    # LOAD_LOCAL a; LOAD_LOCAL b; GET_INDEX
                    stack.append(self.index_get(stack[base + code[pc]], stack[base + code[pc + 1]]))
                    pc += 2
                elif op == FOR_ITER or op == FOR_ITER_STORE:
                    slot = base + code[pc]
                    container = stack[slot]
                    i = stack[slot + 1]
                    if op == FOR_ITER_STORE:
                        # FOR_ITER it off; STORE_LOCAL x
                        if i < len(container):
                            stack[base + code[pc + 1]] = container[i]
                            stack[slot + 1] = i + 1
                            pc += 4
                        else:
                            rel = code[pc + 2] | (code[pc + 3] << 8)
                            pc += 4 + (rel - 0x10000 if rel >= 0x8000 else rel)
                    elif i < len(container):
                        stack.append(container[i])
                        stack[slot + 1] = i + 1
                        pc += 3
                    else:
                        rel = code[pc + 1] | (code[pc + 2] << 8)
                        pc += 3 + (rel - 0x10000 if rel >= 0x8000 else rel)
                elif op == LOAD_UPVAL:
                    stack.append(closure.up[code[pc]].v)
                    pc += 1
                elif op == GET_INDEX:
                    idx = stack.pop()
                    stack[-1] = self.index_get(stack[-1], idx)
                elif op == JUMP_IF_FALSE or op == JUMP_IF_TRUE:
                    v = stack.pop()
                    t = v is True or (v is not False and v is not None and self.truthy(v))
                    if t == (op == JUMP_IF_TRUE):
                        rel = code[pc] | (code[pc + 1] << 8)
                        if rel >= 0x8000:
                            pc += rel - 0xFFFE
                            self.check_budget(pc)
                        else:
                            pc += 2 + rel
                    else:
                        pc += 2
                elif op == LT or op == LE or op == GT or op == GE:
                    b = stack.pop()
                    a = stack[-1]
                    if type(a) is int and type(b) is int:
                        stack[-1] = (a < b if op == LT else a <= b if op == LE else
                                     a > b if op == GT else a >= b)
                    else:
                        stack[-1] = self.compare(op, a, b)
                elif op == INDEX_L:
                    # LOAD_LOCAL b; GET_INDEX
                    stack[-1] = self.index_get(stack[-1], stack[base + code[pc]])
                    pc += 1
                elif op == POP:
                    stack.pop()
                elif op == LOAD_GLOBAL:
                    stack.append(self.globals[code[pc] | (code[pc + 1] << 8)])
                    pc += 2
                elif op == STORE_GLOBAL:
                    self.globals[code[pc] | (code[pc + 1] << 8)] = stack.pop()
                    pc += 2
                elif op == GET_FIELD:
                    stack[-1] = self.get_field(stack[-1], code[pc] | (code[pc + 1] << 8))
                    pc += 2
                elif op == SET_FIELD:
                    val = stack.pop()
                    self.set_field(stack.pop(), code[pc] | (code[pc + 1] << 8), val)
                    pc += 2
                elif op == EQ:
                    b = stack.pop()
                    stack[-1] = self.equal(stack[-1], b)
                elif op == NE:
                    b = stack.pop()
                    stack[-1] = not self.equal(stack[-1], b)
                elif op == DIV or op == MOD or op == POW:
                    b = stack.pop()
                    a = stack[-1]
                    if op == MOD and type(a) is int and type(b) is int and a >= 0 and b > 0:
                        stack[-1] = a % b               # C and Python agree here
                    else:
                        stack[-1] = self.arith(op, a, b)
                elif op == CONCAT:
                    n = code[pc]
                    pc += 1
                    parts = stack[len(stack) - n:]
                    del stack[len(stack) - n:]
                    stack.append(b"".join(p if type(p) is bytes else self.to_str(p) for p in parts))
                elif op == NEW_STRUCT:
                    t = code[pc] | (code[pc + 1] << 8)
                    n = code[pc + 2]
                    pc += 3
                    fields = stack[len(stack) - n:] if n else []
                    if n:
                        del stack[len(stack) - n:]
                    stack.append(VMStruct(t, fields))
                elif op == Op.PUSH_NIL:
                    stack.append(None)
                elif op == Op.PUSH_TRUE:
                    stack.append(True)
                elif op == Op.PUSH_FALSE:
                    stack.append(False)
                elif op == Op.PUSH_I16:
                    v = code[pc] | (code[pc + 1] << 8)
                    stack.append(v - 0x10000 if v >= 0x8000 else v)
                    pc += 2
                elif op == Op.DUP:
                    stack.append(stack[-1])
                elif op == Op.DUP2:
                    stack.extend(stack[-2:])
                elif op == Op.SWAP:
                    stack[-1], stack[-2] = stack[-2], stack[-1]
                elif op == Op.NEG:
                    v = stack[-1]
                    if type(v) is int:
                        stack[-1] = wrap_int64(-v)
                    elif type(v) is float:
                        stack[-1] = -v
                    else:
                        self.panic(f"cannot negate {self.type_name(v).decode()}")
                elif op == Op.NOT:
                    stack[-1] = not self.truthy(stack[-1])
                elif op == Op.BNOT:
                    v = stack[-1]
                    if type(v) is not int:
                        self.panic("~ needs an integer")
                    stack[-1] = ~v
                elif op in (Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR):
                    b = stack.pop()
                    stack[-1] = self.bitwise(op, stack[-1], b)
                elif op == Op.LOAD_CELL:
                    stack.append(stack[base + code[pc]].v)
                    pc += 1
                elif op == Op.STORE_CELL:
                    stack[base + code[pc]].v = stack.pop()
                    pc += 1
                elif op == Op.MAKE_CELL:
                    slot = base + code[pc]
                    stack[slot] = VMCell(stack[slot])
                    pc += 1
                elif op == Op.STORE_UPVAL:
                    closure.up[code[pc]].v = stack.pop()
                    pc += 1
                elif op == Op.LOAD_UPVAL_RAW:
                    stack.append(closure.up[code[pc]])
                    pc += 1
                elif op == Op.ARG_DEFAULT:
                    if stack[base + code[pc]] is not _UNDEF:
                        rel = code[pc + 1] | (code[pc + 2] << 8)
                        pc += 3 + (rel - 0x10000 if rel >= 0x8000 else rel)
                    else:
                        pc += 3
                elif op == Op.NEW_ARRAY:
                    n = code[pc] | (code[pc + 1] << 8)
                    pc += 2
                    items = stack[len(stack) - n:] if n else []
                    if n:
                        del stack[len(stack) - n:]
                    stack.append(items)
                elif op == Op.NEW_MAP:
                    n = code[pc] | (code[pc + 1] << 8)
                    pc += 2
                    d = {}
                    items = stack[len(stack) - 2 * n:] if n else []
                    if n:
                        del stack[len(stack) - 2 * n:]
                    for i in range(0, len(items), 2):
                        d[self.map_key(items[i])] = items[i + 1]
                    stack.append(d)
                elif op == Op.SET_INDEX:
                    val = stack.pop()
                    idx = stack.pop()
                    self.index_set(stack.pop(), idx, val)
                elif op == Op.GET_SLOT:
                    v = stack[-1]
                    i = code[pc]
                    pc += 1
                    if type(v) is VMStruct and i < len(v.f):
                        stack[-1] = v.f[i]
                    elif type(v) is list and i < len(v):
                        stack[-1] = v[i]
                    else:
                        self.panic("pattern slot out of range")
                elif op == Op.MAKE_CLOSURE:
                    fidx = code[pc] | (code[pc + 1] << 8)
                    n = code[pc + 2]
                    pc += 3
                    ups = tuple(stack[len(stack) - n:])
                    del stack[len(stack) - n:]
                    stack.append(VMClosure(fidx, ups))
                elif op == Op.PUSH_FUNC:
                    stack.append(VMClosure(code[pc] | (code[pc + 1] << 8)))
                    pc += 2
                elif op == Op.MATCH_TYPE:
                    t = code[pc] | (code[pc + 1] << 8)
                    pc += 2
                    v = stack[-1]
                    stack[-1] = type(v) is VMStruct and v.t == t
                elif op == Op.IS_ARRAY_LEN:
                    n = code[pc] | (code[pc + 1] << 8)
                    pc += 2
                    v = stack[-1]
                    stack[-1] = type(v) is list and len(v) == n
                elif op == Op.CAST:
                    stack[-1] = self.cast(code[pc], stack[-1])
                    pc += 1
                elif op == Op.TRY or op == Op.TRY_TIMEOUT:
                    rel = code[pc] | (code[pc + 1] << 8)
                    pc += 2
                    handler = pc + (rel - 0x10000 if rel >= 0x8000 else rel)
                    deadline = 0.0
                    if op == Op.TRY_TIMEOUT:
                        ms = self.num(stack.pop(), 'timeout')
                        deadline = _time.monotonic() * 1000.0 + max(ms, 0)
                        self.deadlines += 1
                    self.recover.append([handler, len(frames), len(stack), deadline, len(handlers)])
                elif op == Op.END_TRY:
                    if self.recover and self.recover.pop()[3]:
                        self.deadlines -= 1
                elif op == Op.PANIC:
                    v = stack.pop()
                    self.panic(v if type(v) is bytes else self.to_str(v))
                elif op == Op.UNWRAP_OR_RETURN or op == Op.UNWRAP_OR_PANIC:
                    v = stack[-1]
                    if type(v) is VMStruct and (v.t == TYPE_SOME or v.t == TYPE_OK):
                        stack[-1] = v.f[0]
                    elif v is None or (type(v) is VMStruct and v.t == TYPE_ERR):
                        if op == Op.UNWRAP_OR_PANIC:
                            self.panic(b"? applied to None" if v is None else
                                       b"? applied to Err: " + self.to_str(v.f[0]))
                        # early return of None / Err from the current function
                        result = stack.pop()
                        depth = len(frames)
                        while self.recover and self.recover[-1][1] >= depth:
                            if self.recover.pop()[3]:
                                self.deadlines -= 1
                        if handlers and self.leave_frame(depth, frame[0], result):
                            frame = frames[-1]
                            pc = frame[1]
                            base = frame[2]
                            closure = frame[3]
                            continue
                        frames.pop()
                        del stack[base - frame[4]:]
                        stack.append(result)
                        if not frames:
                            return
                        frame = frames[-1]
                        pc = frame[1]
                        base = frame[2]
                        closure = frame[3]
                    # any other value passes through unchanged
                elif op == PERFORM:
                    # PERFORM name, default, argc: enter the innermost visible clause for
                    # the effect (a call: RESUME returns here), else the default function.
                    # (Kept out of the CALL branch: the hot opcodes after it in this chain
                    # would pay one more comparison each.)
                    name = consts[code[pc] | (code[pc + 1] << 8)]
                    fidx = code[pc + 2] | (code[pc + 3] << 8)
                    argc = code[pc + 4]
                    pc += 5
                    new_closure = None
                    found = self.find_clause(name) if handlers else None
                    if found is not None:
                        new_closure = found[1]
                        if type(new_closure) is not VMClosure:
                            self.panic(b"effect '" + name[:self.EFFECT_NAME_MAX] + b"' has no callable clause")
                        fidx = new_closure.fn
                        fn = funcs[fidx]
                        if argc > fn.n_params or argc < fn.n_required:
                            self.panic(b"effect '" + name[:self.EFFECT_NAME_MAX] + b"' expects " +
                                       str(fn.n_params).encode() + b" argument(s), got " + str(argc).encode())
                        if len(handlers) >= self.MAX_HANDLERS:
                            self.panic("too many nested handlers")
                        # mark the running clause: it masks its own handler and above
                        handlers.append([1, len(frames), 0, len(self.recover), 0, found[0]])
                    elif fidx == 0xFFFF:
                        self.panic(b"unhandled effect '" + name[:self.EFFECT_NAME_MAX] + b"'")
                    # enter the function exactly like CALL (no callee slot)
                    fn = funcs[fidx]
                    if argc > fn.n_params or argc < fn.n_required:
                        self.panic(f"{fn.name}() takes {fn.n_params} argument(s), got {argc}")
                    if len(frames) >= self.MAX_FRAMES:
                        raise TILFatal("stack overflow (recursion too deep)")
                    frame[1] = pc
                    new_base = len(stack) - argc
                    if argc < fn.n_params:
                        stack.extend([_UNDEF] * (fn.n_params - argc))
                    if fn.n_locals > fn.n_params:
                        stack.extend([None] * (fn.n_locals - fn.n_params))
                    frame = [fidx, fn.offset, new_base, new_closure, 0]
                    frames.append(frame)
                    pc = fn.offset
                    base = new_base
                    closure = new_closure
                    self.check_budget(pc)
                elif op == Op.HANDLE:
                    # HANDLE n, rel_end: the n (clause, name) pairs below stay on the stack
                    n = code[pc]
                    rel = code[pc + 1] | (code[pc + 2] << 8)
                    pc += 3
                    if len(handlers) >= self.MAX_HANDLERS:
                        self.panic("too many nested handlers")
                    handlers.append([0, len(frames), len(stack) - 2 * n, len(self.recover),
                                          pc + (rel - 0x10000 if rel >= 0x8000 else rel), n])
                elif op == Op.END_HANDLE:
                    # the body finished: drop the record and its pairs, keep the value
                    hs = handlers
                    if not hs or hs[-1][0] != 0 or hs[-1][1] != len(frames):
                        raise TILFatal("END_HANDLE without an active handle")
                    sp = hs.pop()[2]
                    v = stack.pop()
                    del stack[sp:]
                    stack.append(v)
                elif op == Op.RESUME:
                    # inside a clause: TOS becomes the value of the perform, the
                    # handler stack is exactly as it was at the perform
                    v = stack.pop()
                    hs = handlers
                    depth = len(frames)
                    k = len(hs) - 1
                    while k >= 0 and hs[k][0] != 1:
                        k -= 1
                    if k < 0 or hs[k][1] != depth - 1 or \
                            not funcs[frame[0]].flags & TBCFunction.FLAG_EFFECT_CLAUSE:
                        raise TILFatal("resume outside an effect clause")
                    n_try = hs[k][3]
                    del hs[k:]
                    self.pop_recover_to(n_try)
                    frames.pop()
                    del stack[base - frame[4]:]
                    stack.append(v)
                    frame = frames[-1]
                    pc = frame[1]
                    base = frame[2]
                    closure = frame[3]
                elif op == Op.HALT:
                    return
                elif op == Op.NOP:
                    pass
                else:
                    raise TILFatal(f"bad opcode 0x{op:02x} at {op_pc}")
        except (TILPanic, TILFatal) as e:
            if e.pc < 0:
                e.pc = op_pc
            frame[1] = pc
            raise


def disassemble(m: TBCModule) -> str:
    """Human-readable listing of a bytecode module (`til dis`)."""
    if not m.code:
        m.link()
    out = [f"; TIL bytecode v{getattr(m, 'version', TBC_VERSION)}: {len(m.funcs)} functions, {len(m.consts)} constants, "
           f"{len(m.globals)} globals, {len(m.code)} bytes of code"]
    energy = getattr(m, 'energy', ENERGY_PERFORMANCE)
    if energy:
        name = {v: k for k, v in ENERGY_MODES.items()}[energy]
        out.append(f"; energy mode: {name} ({energy}, #[energy: {name}])")
    for i, (name, fields) in enumerate(m.types):
        if i > TYPE_ERR:
            out.append(f"; type #{i} {name} {{ {', '.join(fields)} }}")
    if m.debug_info:
        for g, name in enumerate(m.globals):
            out.append(f"; global #{g} {name}")
    bounds = sorted((f.offset, i) for i, f in enumerate(m.funcs))
    ends = {idx: (bounds[k + 1][0] if k + 1 < len(bounds) else len(m.code))
            for k, (_off, idx) in enumerate(bounds)}
    line_at = dict(m.lines)

    def show_const(ix):
        c = m.consts[ix]
        if isinstance(c, bytes):
            text = c.decode('utf-8', 'replace')
            return '"' + (text if len(text) <= 40 else text[:37] + '...').replace('\n', '\\n') + '"'
        return repr(c)

    for i, f in enumerate(m.funcs):
        flags = [n for bit, n in ((1, 'method'), (2, 'propagates'), (4, 'lambda'), (8, 'energy-low'),
                                  (16, 'effect-clause')) if f.flags & bit]
        out.append("")
        out.append(f"fn #{i} {f.name}  params={f.n_params} required={f.n_required} "
                   f"locals={f.n_locals} upvals={f.n_upvals}" + (f" [{', '.join(flags)}]" if flags else ""))
        if m.debug_info:
            for slot, lname, start, stop in sorted(f.locals, key=lambda r: (r[2], r[0])):
                out.append(f";   local {lname}: slot {slot}, live {start:04d}-{stop:04d}")
            for k, uname in enumerate(f.upval_names):
                out.append(f";   upvalue {uname}: #{k}")
        pc, end = f.offset, ends[i]
        while pc < end:
            op = m.code[pc]
            name = OP_NAMES.get(op, f"0x{op:02x}")
            fmt = OP_OPERANDS.get(op, '')
            p = pc + 1
            vals = []
            for kind in fmt:
                if kind == 'b':
                    vals.append(m.code[p]); p += 1
                elif kind == 'c':
                    vals.append(_struct.unpack_from('<b', m.code, p)[0]); p += 1
                elif kind == 'h':
                    vals.append(_struct.unpack_from('<H', m.code, p)[0]); p += 2
                else:
                    vals.append(_struct.unpack_from('<h', m.code, p)[0]); p += 2
            note = ""
            if op == Op.PUSH_CONST:
                note = show_const(vals[0])
            elif op in (Op.CALL, Op.PUSH_FUNC, Op.MAKE_CLOSURE):
                note = m.funcs[vals[0]].name if vals[0] < len(m.funcs) else "?"
            elif op == Op.INVOKE:
                note = "." + m.consts[vals[0]].decode('utf-8', 'replace') + "()"
            elif op in (Op.GET_FIELD, Op.SET_FIELD):
                note = "." + m.consts[vals[0]].decode('utf-8', 'replace')
            elif op == Op.LOAD_LOCAL_FIELD:
                note = "." + m.consts[vals[1]].decode('utf-8', 'replace') if vals[1] < len(m.consts) else "?"
            elif op in (Op.NATIVE, Op.NATIVE_POP):
                note = NATIVES[vals[0]] + "()" if vals[0] < len(NATIVES) else "?"
            elif op in (Op.JUMP_IF_NOT, Op.JUMP_IF_NOT_L, Op.JUMP_IF_NOT_I, Op.JUMP_IF_NOT_LL,
                        Op.JUMP_IF_NOT_LI):
                note = f"{OP_NAMES.get(vals[0], '?')} -> {p + vals[-1] - f.offset:04d}"
            elif op in (Op.CAST_LOCAL, Op.CAST_RETURN):
                note = {v: k for k, v in CAST_KINDS.items()}.get(vals[-1], "?")
            elif op in (Op.NEW_STRUCT, Op.MATCH_TYPE):
                note = m.types[vals[0]][0] if vals[0] < len(m.types) else "?"
            elif op == Op.HANDLE:
                note = f"{vals[0]} clause(s), abort -> {p + vals[-1] - f.offset:04d}"
            elif op in (Op.JUMP, Op.JUMP_IF_FALSE, Op.JUMP_IF_TRUE, Op.TRY, Op.TRY_TIMEOUT):
                note = f"-> {p + vals[-1] - f.offset:04d}"
            elif op in (Op.FOR_ITER, Op.ARG_DEFAULT):
                note = f"-> {p + vals[-1] - f.offset:04d}"
            elif op == Op.CAST:
                note = {v: k for k, v in CAST_KINDS.items()}.get(vals[0], "?")
            elif op == Op.PERFORM:
                note = m.consts[vals[0]].decode('utf-8', 'replace') if vals[0] < len(m.consts) else "?"
                note += (", no default" if vals[1] == 0xFFFF else
                         f", default {m.funcs[vals[1]].name if vals[1] < len(m.funcs) else '?'}")
            line = f"  {pc - f.offset:04d}  {name:<18}{' '.join(str(v) for v in vals):<12}"
            if note:
                line += f"; {note}"
            if pc in line_at:
                line = f"{line:<62} @ line {line_at[pc]}"
            out.append(line.rstrip())
            pc = p
    return "\n".join(out)


def compile_to_bytecode(source: str, filename: str = "<stdin>", verify: bool = True,
                        fuse: bool = True, debug_info: bool = False) -> TBCModule:
    """TIL source -> bytecode module (parse, resolve imports, verify, compile).
    fuse=False skips the superinstruction pass (til ... --no-fuse); debug_info=True
    records variable names for the debugger (til build --debug-info)."""
    program = resolve_imports(parse_source(source, filename), filename)
    if verify:
        errors = SemanticChecker(source, filename).check(program)
        if errors:
            raise CompileError("\n\n".join(errors))
    module = BytecodeCompiler(source, filename, fuse=fuse, debug_info=debug_info).compile_program(program)
    if verify:
        errors = type_errors(program, source, filename)
        if errors:
            raise CompileError("\n\n".join(errors))
    module.link()
    return module


def find_vm_sources() -> Optional[str]:
    """Directory holding the portable C VM (vm/tilvm.c, vm/tilvm.h)."""
    for home in til_home_dirs():
        d = os.path.join(home, 'vm')
        if os.path.exists(os.path.join(d, 'tilvm.c')) and os.path.exists(os.path.join(d, 'tilvm.h')):
            return d
    return None


def bytes_to_c_array(name: str, data: bytes) -> str:
    rows = []
    for i in range(0, len(data), 16):
        rows.append("    " + ", ".join(f"0x{b:02x}" for b in data[i:i + 16]) + ",")
    return (f"static const unsigned char {name}[{len(data)}] = {{\n" + "\n".join(rows) +
            f"\n}};\nstatic const unsigned long {name}_size = {len(data)}UL;\n")


def amalgamate_c_vm(tbc: bytes, source_name: str = "program", with_main: bool = True) -> str:
    """One self-contained C file: the TIL VM + this program's bytecode.
    Compile it with any C99 compiler for any device (`til build --target c-vm`)."""
    vm_dir = find_vm_sources()
    if vm_dir is None:
        raise CompileError("cannot find the TIL C VM sources (vm/tilvm.c). "
                           "Set TIL_HOME to the TIL directory.")
    with open(os.path.join(vm_dir, 'tilvm.h'), encoding='utf-8') as f:
        header = f.read()
    with open(os.path.join(vm_dir, 'tilvm.c'), encoding='utf-8') as f:
        impl = f.read().replace('#include "tilvm.h"', '/* tilvm.h inlined above */')
    parts = [
        f"/* Generated by TIL {TIL_VERSION} from {os.path.basename(source_name)}:",
        "   the portable TIL virtual machine + the program's bytecode in one file.",
        "   Build: cc -O2 -o program this_file.c -lm   (or with any calculator / MCU toolchain) */",
        "#define TILVM_AMALGAMATED 1",
        header,
        impl,
        bytes_to_c_array("til_program", tbc),
    ]
    if with_main:
        parts.append('''
#ifndef TILVM_NO_MAIN
int main(void) {
    static tilvm vm;
    int rc = tilvm_load(&vm, til_program, (unsigned long)til_program_size);
    if (rc != 0) { tilvm_default_write(NULL, "TIL VM: cannot load program\\n", 29); return 1; }
    return tilvm_run(&vm);
}
#endif
''')
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#                                COMPILER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_source(source: str, filename: str = "<stdin>") -> Program:
    """Lex + parse a TIL source file (honouring `#[lang: ...]` pragmas)."""
    lexer = Lexer(source, filename)
    tokens = lexer.tokenize()
    parser = Parser(tokens, filename, source=source, languages=lexer.languages)
    return parser.parse()


def til_home_dirs() -> List[str]:
    """Directories that may hold the TIL distribution (stdlib/, vm/):
    the repository checkout this compiler lives in, $TIL_HOME, ~/.til."""
    dirs = []
    compiler_dir = os.path.dirname(os.path.abspath(__file__))
    dirs.append(os.path.dirname(compiler_dir))          # repo layout: src/til.py
    dirs.append(compiler_dir)                           # flat layout
    if os.environ.get('TIL_HOME'):
        dirs.append(os.environ['TIL_HOME'])
    dirs.append(os.path.join(os.path.expanduser('~'), '.til'))
    return dirs


class ModuleResolver:
    """Resolves and loads TIL module files."""

    def __init__(self, base_path: str = "."):
        self.base_path = base_path
        self.loaded: Dict[str, Program] = {}  # module path -> parsed AST

    def resolve(self, module_name: str, from_file: str = "") -> Optional[str]:
        """Resolve a module name to a file path."""
        # Convert module.path to file path
        parts = module_name.replace(".", os.sep)

        def find(dirs: List[str]) -> Optional[str]:
            for d in dirs:
                # Try module_name.til, then module_name/mod.til (package)
                for candidate in (os.path.join(d, parts + ".til"), os.path.join(d, parts, "mod.til")):
                    if os.path.exists(candidate):
                        return candidate
            return None

        # Search: importing file's dir, base path, $TIL_PATH, (declared packages,) stdlib dirs
        search_dirs = []
        if from_file and not from_file.startswith("<"):
            search_dirs.append(os.path.dirname(os.path.abspath(from_file)))
        # a file of an installed package never searches the current directory or
        # $TIL_PATH: its imports must not depend on where `til` runs (til.lock pins them)
        if installed_package_dirs(from_file) is None:
            search_dirs.append(self.base_path)
            for extra in os.environ.get('TIL_PATH', '').split(os.pathsep):
                if extra:
                    search_dirs.append(extra)
        found = find(search_dirs)
        if found:
            return found
        # a dependency declared in the til.json that governs the importing file is
        # pinned by til.lock: only til_modules/ provides it - never the standard
        # library ($TIL_HOME/stdlib and ~/.til/stdlib differ between machines)
        if module_name.split(".")[0] in declared_dependencies(from_file, self.base_path):
            return find_package_module(module_name, from_file, self.base_path)
        stdlib_dirs = [os.path.join(home, 'stdlib') for home in til_home_dirs()]
        found = find([d for d in stdlib_dirs if os.path.isdir(d)])
        # the other installed packages (til_modules/, `til pkg`) come after everything above
        return found or find_package_module(module_name, from_file, self.base_path)

    def load_module(self, module_name: str, from_file: str = "") -> Optional[Program]:
        """Load and parse a module, returning its AST (cached per file)."""
        path = self.resolve(module_name, from_file)
        if not path:
            hint = package_install_hint(module_name, from_file, self.base_path)
            if hint:
                print(localize(hint), file=sys.stderr)
            return None
        key = module_file_key(path)         # one AST per file, whatever path reaches it
        if key in self.loaded:
            return self.loaded[key]

        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast = parse_source(source, path)
        ast.filename = path
        self.loaded[key] = ast
        return ast


# ── packages: til_modules/ (installed by `til pkg`, see docs/TIL_PACKAGES.md) ──
# `import NAME` / `from NAME import x` look for til_modules/NAME/ after the
# search paths above - but a NAME declared in the "dependencies" of the
# importing package's til.json (or, for a program, of its project's til.json)
# comes from til_modules/ only, before the standard library is searched
# (declared_dependencies):
#   * from a program: in the importing file's directory, then each parent
#     directory (nearest first), up to the nearest project root (the first
#     directory with a til.json or til.lock) - never above it;
#   * from a file of an installed package (PROJECT/til_modules/PKG/...): only
#     in PROJECT/til_modules/ - the flat directory til.lock describes; never a
#     til_modules/ nested inside an installed package.
# Inside a package:
#   import NAME      -> til.json "main", else NAME.til, lib.til, mod.til
#   import NAME.a.b  -> a/b.til or a/b/mod.til (package root first, then the
#                       directory of the entry file)
PACKAGE_DIR = 'til_modules'


def _package_manifest(directory: str) -> Dict[str, Any]:
    """til.json of a directory as a dict ({} if missing or unreadable)."""
    import json
    try:
        with open(os.path.join(directory, 'til.json'), encoding='utf-8-sig') as f:
            data = json.load(f)
    except (OSError, ValueError, RecursionError):      # RecursionError: JSON nested too deeply
        return {}
    return data if isinstance(data, dict) else {}


def _package_start_dir(from_file: str, base_path: str) -> str:
    if from_file and not from_file.startswith("<"):
        return os.path.dirname(os.path.abspath(from_file))
    return os.path.abspath(base_path)


def package_entry_file(pkg_dir: str, name: str) -> Optional[str]:
    """The file `import NAME` loads from a package directory: til.json "main"
    (a relative path inside the package; '/' or '\\' separate directories on
    every OS, as `til pkg` checks it), else NAME.til, lib.til, mod.til."""
    main = _package_manifest(pkg_dir).get('main')
    if isinstance(main, str) and main:
        root = os.path.abspath(pkg_dir)
        parts = main.replace('\\', '/').split('/')
        path = os.path.abspath(os.path.join(root, *parts))
        if (os.path.isabs(main) or main.startswith(('/', '\\')) or ':' in parts[0]
                or not path.startswith(root + os.sep) or not os.path.isfile(path)):
            return None             # `til pkg` rejects such packages; never look elsewhere
        return path
    for candidate in (name + ".til", "lib.til", "mod.til"):
        path = os.path.join(pkg_dir, candidate)
        if os.path.isfile(path):
            return path
    return None


def _is_project_root(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, 'til.json')) or \
        os.path.isfile(os.path.join(directory, 'til.lock'))


def installed_package_dirs(from_file: str) -> Optional[Tuple[str, str]]:
    """(package directory, project til_modules/) when `from_file` belongs to an
    installed package - it lies in PROJECT/til_modules/PKG/ and PROJECT has a
    til.json or til.lock - else None."""
    if not from_file or from_file.startswith("<"):
        return None
    d = os.path.dirname(os.path.abspath(from_file))
    while True:
        parent = os.path.dirname(d)
        if parent == d:
            return None
        if os.path.basename(parent) == PACKAGE_DIR and _is_project_root(os.path.dirname(parent)):
            return d, parent
        d = parent


def package_search_dirs(from_file: str = "", base_path: str = ".") -> List[str]:
    """The til_modules/ directories `import NAME` looks in, nearest first."""
    owner = installed_package_dirs(from_file)
    if owner is not None:
        return [owner[1]]           # a package's dependencies: the project's flat til_modules/ only
    dirs = []
    d = _package_start_dir(from_file, base_path)
    while True:
        dirs.append(os.path.join(d, PACKAGE_DIR))
        parent = os.path.dirname(d)
        if _is_project_root(d) or parent == d:
            return dirs             # never above the nearest project root
        d = parent


def find_package_module(module_name: str, from_file: str = "", base_path: str = ".") -> Optional[str]:
    """Resolve `import NAME[.sub...]` against installed packages: the first
    til_modules/NAME/ directory of package_search_dirs()."""
    parts = module_name.split(".")
    for modules in package_search_dirs(from_file, base_path):
        pkg_dir = os.path.join(modules, parts[0])
        if os.path.isdir(pkg_dir):
            entry = package_entry_file(pkg_dir, parts[0])
            if len(parts) == 1:
                return entry
            roots = [pkg_dir]
            if entry and os.path.dirname(entry) != os.path.abspath(pkg_dir):
                roots.append(os.path.dirname(entry))
            sub = os.path.join(*parts[1:])
            for root in roots:
                for candidate in (os.path.join(root, sub + ".til"), os.path.join(root, sub, "mod.til")):
                    if os.path.isfile(candidate):
                        return candidate
            return None             # the nearest package of that name wins
    return None


def declared_dependencies(from_file: str = "", base_path: str = ".") -> Set[str]:
    """The names in "dependencies" of the til.json that governs `from_file`:
    that of the installed package it belongs to, else that of the nearest
    project root (the first directory upwards with a til.json or til.lock)."""
    owner = installed_package_dirs(from_file)
    if owner is not None:
        directory = owner[0]
    else:
        directory = _package_start_dir(from_file, base_path)
        while not _is_project_root(directory):
            parent = os.path.dirname(directory)
            if parent == directory:
                return set()
            directory = parent
    deps = _package_manifest(directory).get('dependencies')
    return set(deps) if isinstance(deps, dict) else set()


def package_install_hint(module_name: str, from_file: str = "", base_path: str = ".") -> Optional[str]:
    """A hint for a module that was not found: an installed package lacks that
    (sub)module, or the nearest til.json declares it as a dependency that was
    never installed (`til pkg install` was not run)."""
    parts = module_name.split(".")
    name = parts[0]
    for modules in package_search_dirs(from_file, base_path):
        pkg_dir = os.path.join(modules, name)
        if os.path.isdir(pkg_dir):
            if len(parts) > 1:
                sub = "/".join(parts[1:])
                return (f"hint: the package '{name}' ({pkg_dir}) has no module '{'.'.join(parts[1:])}' "
                        f"({sub}.til or {sub}/mod.til)")
            return (f"hint: the package '{name}' ({pkg_dir}) has no entry file: til.json \"main\" if it is "
                    f"set, else {name}.til, lib.til or mod.til")
    owner = installed_package_dirs(from_file)
    d = owner[0] if owner is not None else _package_start_dir(from_file, base_path)
    while True:
        if _is_project_root(d):
            deps = _package_manifest(d).get('dependencies')
            if not (isinstance(deps, dict) and name in deps):
                return None
            project = os.path.dirname(owner[1]) if owner is not None else d
            return (f"hint: '{name}' is a dependency in {os.path.join(d, 'til.json')} but is not "
                    f"installed - run `til pkg install` in {project}")
        parent = os.path.dirname(d)
        if parent == d or owner is not None:
            return None             # a package's own til.json is at its root
        d = parent


IMPORTABLE_NODES = (FuncDef, StructDef, EnumDef, ImplBlock, TraitDef, TypeAlias,
                    VarDecl, CapabilityType)

def _definition_name(node) -> Optional[str]:
    if isinstance(node, ImplBlock):
        return node.type_name
    return getattr(node, 'name', None)


def module_file_key(path: str) -> str:
    """The identity of a source file: the same file reached through different
    paths (symbolic links, '..', letter case on Windows) is one module."""
    if not path or path.startswith("<"):
        return path
    return os.path.normcase(os.path.realpath(path))


def _definition_kind(node) -> str:
    if isinstance(node, VarDecl):
        return 'const' if node.is_const else ('var' if node.mutable else 'let')
    for cls, kind in ((FuncDef, 'function'), (StructDef, 'struct'), (EnumDef, 'enum'), (TraitDef, 'trait'),
                      (TypeAlias, 'type alias'), (CapabilityType, 'capability')):
        if isinstance(node, cls):
            return kind
    return 'definition'


MAX_NAMESPACE_ERRORS = 10


def _shown_path(path: str) -> str:
    """A source path in messages: relative to the current directory when the
    file is below it, else as it is."""
    if not os.path.isabs(path):
        return path
    try:
        rel = os.path.relpath(path)
    except ValueError:              # another drive (Windows)
        return path
    return path if rel == os.pardir or rel.startswith(os.pardir + os.sep) else rel


def namespace_errors(definitions: List[Tuple[str, ASTNode]]) -> List[str]:
    """TIL has ONE namespace for a program and everything it imports (its
    modules and packages, directly or indirectly): there are no module-private
    names, and merged definitions see each other. So a top-level name
    (function, const/let/var, struct, enum, trait, type alias, capability) -
    or a method of a type - that two different files bring into the program
    would make code of one file silently use the other file's definition.
    Each such clash is an error naming both files. `definitions`: (file,
    definition) for everything merged into the program, each once, in merge
    order. (Duplicates inside one file are left to the backends, as before.)"""
    first: Dict[Tuple[str, str], Tuple[str, str, ASTNode, str]] = {}
    errors: List[str] = []
    more = 0
    reporters: Dict[str, ErrorReporter] = {}
    for path, stmt in definitions:
        if isinstance(stmt, ImplBlock):
            entries = [(('method', f"{stmt.type_name}.{meth.name}"), meth, 'method') for meth in stmt.methods]
        elif isinstance(stmt, IMPORTABLE_NODES) and _definition_name(stmt):
            entries = [(('name', _definition_name(stmt)), stmt, _definition_kind(stmt))]
        else:
            continue
        key = module_file_key(path)
        for ident, node, kind in entries:
            prev = first.setdefault(ident, (path, key, node, kind))
            if prev[1] == key:
                continue
            if len(errors) >= MAX_NAMESPACE_ERRORS:
                more += 1
                continue
            name = ident[1]
            there = f"{_shown_path(prev[0])}:{getattr(prev[2], 'line', 0) or '?'}"
            here = f"{_shown_path(path)}:{getattr(node, 'line', 0) or '?'}"
            if kind == prev[3]:
                msg = f"{kind} '{name}' is defined in two files: {there} and {here}"
            else:
                msg = f"'{name}' is defined in two files: {prev[3]} at {there}, {kind} at {here}"
            if key not in reporters:
                try:
                    with open(path, encoding='utf-8') as f:
                        source = f.read()
                except (OSError, ValueError):
                    source = ""
                reporters[key] = ErrorReporter(source, _shown_path(path))
            hint = ("a program and everything it imports (modules, packages, also indirectly) share one "
                    "namespace - TIL has no module-private names: rename one of the two (packages should "
                    "prefix their internal names with the package name)" if not errors else "")
            errors.append(reporters[key].format_error(getattr(node, 'line', 0) or 0, getattr(node, 'col', 0) or 1,
                                                      msg, "E-IMPORT", hint))
    if more:
        errors.append(f"error[E-IMPORT]: ... and {more} more name{'s' if more != 1 else ''} defined in two files")
    return errors


def _names_used(definition: ASTNode) -> Set[str]:
    """Every string inside a definition (identifiers, but also literals and
    type names): an over-approximation of the top-level names it uses. Merging
    a definition too many is harmless - a clash is never silent (namespace_errors)."""
    from dataclasses import is_dataclass
    names: Set[str] = set()
    todo: List[Any] = [definition]
    seen: Set[int] = set()
    while todo:
        n = todo.pop()
        if isinstance(n, str):
            names.add(n)
        elif id(n) in seen or n is None or isinstance(n, (int, float, bool, bytes)):
            continue
        elif isinstance(n, (list, tuple, set, frozenset)):
            seen.add(id(n))
            todo.extend(n)
        elif isinstance(n, dict):
            seen.add(id(n))
            todo.extend(n.keys())
            todo.extend(n.values())
        elif isinstance(n, Type) or (is_dataclass(n) and not isinstance(n, type)):
            seen.add(id(n))             # AST nodes, types, FuncParam, StructField, EnumVariant, ...
            todo.extend(getattr(n, '__dict__', {}).values())
    return names


def from_import_closure(definitions: List[ASTNode], items: List[str]) -> List[ASTNode]:
    """`from m import a, b` merges a, b and every definition of m (and of what
    m imports) that they use, transitively: a function imported this way still
    finds its module's constants, helpers, structs and methods (it used to fail
    with "unknown function", or - worse - silently use the program's own
    definition of that name; namespace_errors now reports that clash)."""
    wanted = set(items)
    kept: Set[int] = set()
    changed = True
    while changed:
        changed = False
        for d in definitions:
            if id(d) in kept:
                continue
            if isinstance(d, ImplBlock):
                hit = d.type_name in wanted or any(meth.name in wanted for meth in d.methods)
            elif isinstance(d, EnumDef):
                hit = d.name in wanted or any(v.name in wanted for v in d.variants)
            else:
                hit = _definition_name(d) in wanted
            if hit:
                kept.add(id(d))
                wanted |= _names_used(d)
                changed = True
    return [d for d in definitions if id(d) in kept]


def resolve_imports(program: Program, filename: str = "<stdin>") -> Program:
    """Process import statements: load modules (recursively) and merge their
    definitions - functions, structs, enums, impls, traits, consts, aliases.

    `import m` / `import m as alias` also enable qualified calls `m.f()`;
    the module names are recorded in `program.module_names`. Everything
    merged shares one namespace: a name two files bring in is an error
    (namespace_errors)."""
    resolver = ModuleResolver()
    seen_defs: Set[int] = set()
    module_names: Set[str] = set(getattr(program, 'module_names', set()) or ())

    def merge(prog: Program, fname: str, stack: Tuple[str, ...]) -> List[ASTNode]:
        out = []
        for stmt in prog.statements:
            if not isinstance(stmt, Import):
                out.append(stmt)
                continue
            mod_ast = resolver.load_module(stmt.module, fname)
            if mod_ast is None:
                print(localize(f"Warning: Could not find module '{stmt.module}'"), file=sys.stderr)
                continue
            module_names.add(stmt.module.split('.')[-1])
            if stmt.alias:
                module_names.add(stmt.alias)
            mod_path = getattr(mod_ast, 'filename', stmt.module)
            mod_key = module_file_key(mod_path)
            if mod_key in stack:
                continue  # import cycle: definitions are already being merged
            mod_stmts = [s for s in merge(mod_ast, mod_path, stack + (mod_key,))
                         if isinstance(s, IMPORTABLE_NODES)
                         # a module's own entry point / script code is not imported
                         and not (isinstance(s, FuncDef) and s.name == "main")]
            if stmt.items is not None:
                mod_stmts = from_import_closure(mod_stmts, stmt.items)
            out.extend(mod_stmts)
        return out

    merged = merge(program, filename, (module_file_key(filename),))
    program.statements = []
    for stmt in merged:                 # a module imported several times is merged once
        if id(stmt) not in seen_defs:
            seen_defs.add(id(stmt))
            program.statements.append(stmt)
    program.module_names = module_names
    origin = {id(s): mod.filename for mod in resolver.loaded.values() for s in mod.statements}
    clashes = namespace_errors([(origin.get(id(s), filename), s) for s in program.statements])
    if clashes:
        raise CompileError("\n\n".join(clashes))
    return program


class TILCompiler:
    def __init__(self):
        self.verbose = False
        self.optimize = 2
        self.keep_c = False
        self.check_types = True
        self.verify = True      # effects / capabilities / constant contracts (fatal errors)
        self.typed = True       # TIL 3.1: type inference + typed C backend (src/tiltypes.py)
        self.last_c_error = ""

    def compile(self, source: str, filename: str = "<stdin>") -> str:
        """Compile TIL source to C code"""

        if self.verbose:
            print(f"[TIL] Lexing {filename}...")

        if self.verbose:
            print(f"[TIL] Parsing...")
        ast = parse_source(source, filename)

        # Resolve imports
        if self.verbose:
            print(f"[TIL] Resolving imports...")
        ast = resolve_imports(ast, filename)

        if self.verify:
            if self.verbose:
                print(f"[TIL] Verifying effects, capabilities and contracts...")
            errors = SemanticChecker(source, filename).check(ast)
            if errors:
                raise CompileError("\n\n".join(errors))

        tt = types_module() if self.typed else None
        self.typed_c = tt is not None
        if tt is not None:
            if self.verbose:
                print(f"[TIL] Inferring types...")
            try:
                inference = tt.infer_program(ast, source, filename)
            except Exception as e:
                if os.environ.get('TIL_DEBUG_TYPES'):
                    raise
                raise NativeUnsupported(f"native backend: type inference failed "
                                        f"({type(e).__name__}: {e})")
            if self.verify and inference.errors:
                raise CompileError("\n\n".join(inference.errors))
            if self.verbose:
                print(f"[TIL] Generating C code (typed backend)...")
            try:
                return tt.generate_c(ast, source, filename, inference)
            except NativeUnsupported:
                raise
            except Exception as e:      # a bug in the typed backend: the VM still runs the program
                if os.environ.get('TIL_DEBUG_TYPES'):
                    raise
                raise NativeUnsupported(f"native backend: internal error ({type(e).__name__}: {e})")

        if self.check_types:
            if self.verbose:
                print(f"[TIL] Type checking...")
            
            checker = TypeChecker()
            errors = checker.check(ast)
            
            for err in errors:
                print(localize(f"Warning: {err}"), file=sys.stderr)
        
        if self.verbose:
            print(f"[TIL] Generating C code...")
        
        codegen = CCodeGenerator()
        c_code = codegen.generate(ast)
        
        return c_code
    
    def compile_to_executable(self, source: str, output: str, filename: str = "<stdin>",
                              quiet: bool = False) -> bool:
        """Compile TIL source to native executable"""
        
        try:
            c_code = self.compile(source, filename)
        except NativeUnsupported as e:
            self.last_c_error = str(e)
            if not quiet:
                print(localize(f"{e}"), file=sys.stderr)
                print(localize("hint: run it on the TIL VM: `til run --vm` or `til build --target vm`"), file=sys.stderr)
            return False
        
        # Find C compiler
        cc = self.find_c_compiler()
        if not cc:
            print(localize("Error: No C compiler found (tried gcc, clang, cc)"), file=sys.stderr)
            return False
        
        # Write C code to temp file (UTF-8 for string literals with Kazakh/Unicode)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False,
                                         encoding='utf-8', newline='\n') as f:
            f.write(c_code)
            c_file = f.name
        
        try:
            # Compile C to executable
            opt_flags = ["-O0", "-O1", "-O2", "-O3"][min(self.optimize, 3)]
            # -fwrapv: TIL int arithmetic wraps around (two's complement), like the VMs
            cmd = [cc, c_file, "-o", output, opt_flags, "-fwrapv", "-lm"]
            if getattr(self, 'typed_c', False) and os.path.basename(cc) in ('gcc', 'clang'):
                # typed backend: mixing up ints and pointers is a bug -> fall back to the VM
                cmd[4:4] = ["-Werror=int-conversion", "-Werror=incompatible-pointer-types"]
            
            if self.verbose:
                print(f"[TIL] Compiling: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                   encoding='utf-8', errors='replace')
            
            if result.returncode != 0:
                self.last_c_error = result.stderr
                if not quiet:
                    print(localize("C compilation failed:") + f"\n{result.stderr}", file=sys.stderr)
                    print(localize("hint: this program may use features only the TIL VM supports; "
                                   "try `til run --vm` or `til build --target vm`"), file=sys.stderr)
                return False
            
            if self.verbose:
                print(f"[TIL] Generated: {output}")
            
            return True
        
        finally:
            if not self.keep_c:
                os.unlink(c_file)
            else:
                c_output = output + ".c"
                shutil.move(c_file, c_output)
                print(f"[TIL] C code saved to: {c_output}")
    
    def find_c_compiler(self) -> Optional[str]:
        for cc in ['gcc', 'clang', 'cc']:
            if shutil.which(cc):
                return cc
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#                                  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_help():
    extra_commands = "".join(f"  {name + ' ...':<24}{text}\n" for name, (_mod, text) in EXTRA_COMMANDS.items())
    print(f"""
TIL {TIL_VERSION} - The Intelligent Language
Author: Alisher Beisembekov

Usage: til <command> [options] <file>

Commands:
  run <file.til>          Compile and run (native via C; falls back to the VM
                          automatically when no C compiler is installed)
  build <file.til>        Build: --target native (default) | vm | c-vm | c
  exec <file.tbc>         Run a compiled bytecode file on the TIL VM
  dis <file>              Disassemble a program (.til or .tbc)
  check <file.til>        Parse + verify effects, capabilities, contracts
  translate <file.til>    Translate keywords: --to en | kk | kk-latn | ru
  repl                    Interactive REPL (runs on the VM, no C compiler needed)
{extra_commands}  version                 Show version

Options:
  -o <file>               Output file name
  -c                      Output C code only (same as --target c)
  --vm                    run: execute on the portable TIL VM (pure Python)
  --cvm                   run: execute on the C VM (vm/tilvm.c, built on demand)
  --target <t>            build: native | vm (.tbc) | c-vm (one C file: VM +
                          program, for calculators/MCUs) | c (generated C)
  --strip                 vm/c-vm: drop line tables (smallest bytecode)
  --debug-info            vm/c-vm/dis: add variable names for debuggers (til debug)
  --no-fuse               VM bytecode without superinstructions (0x80-0xBF)
  --device <name>         vm/c-vm: fail the build unless it fits the device (til fit)
  --max-steps <n>         VM sandbox: stop after n loop iterations/calls
  --screen <file.ppm>     graphics: save the final screen as a PPM image
  --screen-ansi           graphics: draw the screen in the terminal (24-bit colour)
  --keys <list>           graphics: scripted keys, e.g. "UP,UP,ENTER,a,ESC"
  --pause-after <n> --save <file.tilp>
                          VM: run n steps, save a portable snapshot, exit 3
  --resume <file.tilp>    VM: continue a portable snapshot (any VM, e.g. a calculator's)
  -O0, -O1, -O2, -O3      Optimization level for native builds (default -O2)
  --keep-c                Keep generated C file
  --no-check              Disable type warnings
  --no-verify             Disable effect/capability/contract/type verification
  --no-types              No type inference: classic C backend, no E-TYPE checks
  -v, --verbose           Verbose output (shows why a program runs on the VM)
  --                      Everything after it is passed to the program (args())

Program arguments:        til run prog.til a b c   ->  args() == ["a", "b", "c"]

Multi-Level Programming:  #[level: 0..4]  Hardware, Systems, Safe (default), Script, Formal
Languages:                English + Kazakh always; #[lang: ru] / #[lang: kk-latn] per file
Messages:                 --lang en|ru|kk or TIL_LANG (else the program's #[lang: ...], else English)

Examples:
  til run hello.til                     # native (or VM if no C compiler)
  til run --vm hello.til                # portable VM
  til build --target vm game.til        # -> game.tbc (runs on any TIL VM)
  til build --target c-vm game.til      # -> game_vm.c (compile for a calculator)
  til translate --to ru hello.til       # Russian keywords
""")


# ── VM helpers ───────────────────────────────────────────────────────────────

def user_cache_dir() -> str:
    """A directory for cached VM executables that only this user can write to:
    ~/.til/cache, else <temp>/til-cache-<uid> (created 0700, owner and mode checked),
    else a new private temporary directory.  Cached executables are run, so they must
    never come from a shared, predictable place another user could have planted."""
    def private(path: str, forbidden: int) -> bool:
        import stat
        try:
            st = os.lstat(path)
        except OSError:
            return False
        if not stat.S_ISDIR(st.st_mode):
            return False
        if not hasattr(os, 'getuid'):          # Windows: per-user profile and temp directories
            return True
        return st.st_uid == os.getuid() and not st.st_mode & forbidden
    home = os.path.join(os.path.expanduser('~'), '.til', 'cache')
    try:
        os.makedirs(home, exist_ok=True)
        if private(home, 0o002):
            return home
    except OSError:
        pass
    tmp = os.path.join(tempfile.gettempdir(), f"til-cache-{os.getuid() if hasattr(os, 'getuid') else 0}")
    try:
        os.mkdir(tmp, 0o700)
    except OSError:
        pass
    if private(tmp, 0o077):
        return tmp
    return tempfile.mkdtemp(prefix='til-cache-')


def _cvm_binary(verbose: bool = False) -> Optional[str]:
    """Build (once) and return the host C VM executable."""
    vm_dir = find_vm_sources()
    cc = TILCompiler().find_c_compiler()
    if vm_dir is None or cc is None:
        return None
    cache = user_cache_dir()
    sources = [os.path.join(vm_dir, n) for n in ('tilvm.c', 'tilvm.h', 'main.c')]
    # one cached binary per VM source version (several checkouts can share ~/.til/cache)
    import hashlib
    digest = hashlib.sha256()
    for path in sources:
        with open(path, 'rb') as f:
            digest.update(f.read())
    exe = os.path.join(cache, f"tilvm-{digest.hexdigest()[:16]}" + ('.exe' if sys.platform == 'win32' else ''))
    if not os.path.exists(exe):
        tmp_exe = f"{exe}.{os.getpid()}.tmp"
        cmd = [cc, '-O2', '-std=c99', '-o', tmp_exe, sources[0], sources[2], '-lm']
        if verbose:
            print(f"[TIL] Building C VM: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(localize("C VM build failed:") + f"\n{result.stderr}", file=sys.stderr)
            return None
        os.replace(tmp_exe, exe)        # atomic: parallel builds never see half a file
    return exe


def run_bytecode(module: TBCModule, max_steps: int = 0, argv: Optional[List[str]] = None,
                 opts: Optional[Dict[str, Any]] = None) -> int:
    """Run a module on the reference VM. `argv` = program arguments (args()),
    `opts` = run options parsed by main() (see RUN_VALUE_OPTIONS)."""
    opts = opts or {}
    try:
        pause = portable_pause(opts)
        screen, keys = screen_and_keys(opts)
    except ValueError as e:
        print(localize(f"Error: {e}"), file=sys.stderr)
        return 2
    vm = TILVM(module, max_steps=max_steps, argv=argv, screen=screen, keys=keys)
    if opts.get('resume'):              # continue a portable snapshot (docs/TIL_SNAPSHOTS.md)
        path = opts['resume']
        try:
            with open(path, 'rb') as f:
                data = f.read()
            vm.load_portable(data)
        except OSError:
            keys.close()
            print(localize(f"til: cannot read {path}"), file=sys.stderr)
            return 2
        except SnapshotError as e:
            keys.close()
            print(localize(f"til: cannot resume {path}: {e}"), file=sys.stderr)
            return 2
    if isinstance(keys, TerminalKeys):
        vm.read_line = keys.cooked(vm.read_line)
    ctrl_c = CtrlC()
    try:
        try:
            if pause is None:
                rc = vm.run()
            else:
                rc = run_and_save(vm, pause, opts['save'])
            ctrl_c.running = False
        except KeyboardInterrupt:   # Ctrl-C: the program ends; the cleanup below still runs
            ctrl_c.running, ctrl_c.hit, rc = False, True, 130
        keys.close()
        rc = save_screen(screen, opts.get('screen'), rc)
    finally:
        keys.close()
        ctrl_c.restore()
    return end_by_sigint() if ctrl_c.hit else rc


def portable_pause(opts: Dict[str, Any]) -> Optional[int]:
    """--pause-after N --save FILE: N (>= 0), or None without them. ValueError if
    only one of the two is given or N is not a number."""
    if opts.get('pause_after') is None and opts.get('save') is None:
        return None
    if opts.get('pause_after') is None or not opts.get('save'):
        raise ValueError("--pause-after N and --save FILE go together")
    try:
        n = int(opts['pause_after'])
    except ValueError:
        n = -1
    if n < 0:
        raise ValueError("--pause-after needs a number of steps")
    return n


def run_and_save(vm: 'TILVM', steps: int, path: str) -> int:
    """Run `steps` more steps (0: none), then save the paused program as a portable
    snapshot: exit code 3. A program that ends first ends as usual (nothing saved)."""
    status = vm.run_steps(steps) if steps else (vm.YIELD if vm.status is None else vm.status)
    if status != vm.YIELD:
        print(localize("til: the program ended before the pause point (nothing saved)"), file=sys.stderr)
        return vm.exit_code
    try:
        data = vm.save_portable()
        with open(path, 'wb') as f:
            f.write(data)
    except SnapshotError as e:
        print(localize(f"til: cannot save {path}: {e}"), file=sys.stderr)
        return 2
    except OSError:
        print(localize(f"til: cannot write {path}"), file=sys.stderr)
        return 2
    print(localize(f"til: paused after {steps} steps; saved {path} ({len(data)} bytes)"), file=sys.stderr)
    return 3


def cvm_snapshot_flags(opts: Dict[str, Any]) -> List[str]:
    """The portable snapshot options for the C VM runner (vm/main.c)."""
    flags = []
    pause = portable_pause(opts)
    if pause is not None:
        flags += ['--save-portable-after', str(pause), opts['save']]
    if opts.get('resume'):
        flags += ['--resume-portable', opts['resume']]
    return flags


class CtrlC:
    """SIGINT while a program runs raises KeyboardInterrupt (Python's default);
    after that (`running = False`: terminal, --screen image) it is only
    remembered in `hit`, as vm/main.c does, so the cleanup always finishes."""

    def __init__(self):
        import signal
        self.running, self.hit, self.old = True, False, None
        try:
            if signal.getsignal(signal.SIGINT) is not signal.SIG_IGN:
                self.old = signal.signal(signal.SIGINT, self._handler)
        except (ValueError, OSError, AttributeError):      # not the main thread
            self.old = None

    def _handler(self, signum, frame):
        self.hit = True
        if self.running:
            raise KeyboardInterrupt

    def restore(self):
        if self.old is not None:
            import signal
            try:
                signal.signal(signal.SIGINT, self.old)
            except (ValueError, OSError):
                pass
            self.old = None


def end_by_sigint() -> int:
    """After Ctrl-C and the cleanup: flush the output, then end this process by
    SIGINT as an uncaught Ctrl-C does - a shell then also stops the script that
    ran it, and `$?` is 130. Where that cannot be done (Windows, not the main
    thread) the exit code is 130."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    if os.name == 'posix':
        import signal
        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGINT)
        except (ValueError, OSError):
            pass
    return 130


def run_foreground(cmd: List[str]) -> int:
    """Run a child (the C VM) in the foreground, like system(): a Ctrl-C reaches
    the child too, so wait for it to finish its own cleanup instead of raising
    KeyboardInterrupt here. Returns its exit status (-N: ended by signal N)."""
    ctrl_c = CtrlC()
    ctrl_c.running = False
    try:
        return subprocess.Popen(cmd).wait()
    finally:
        ctrl_c.restore()


def foreground_exit(rc: int) -> int:
    """Exit code for a child's status: a child ended by Ctrl-C ends this process
    the same way (end_by_sigint), one ended by another signal N gives 128 + N."""
    import signal
    if rc == -signal.SIGINT:
        return end_by_sigint()
    return 128 - rc if rc < 0 else rc


# ── graphics run options: --screen FILE.ppm, --screen-ansi, --keys LIST ──────

def screen_and_keys(opts: Dict[str, Any]) -> Tuple[TILScreen, Any]:
    """Screen and key source for a run: scripted keys with --keys, else the
    terminal (or stdin's bytes). Raises ValueError for a bad --keys list."""
    screen = TILScreen(ansi=bool(opts.get('screen_ansi')))
    if opts.get('keys') is not None:
        return screen, KeyScript(parse_key_script(opts['keys']))
    return screen, TerminalKeys()


def save_screen(screen: TILScreen, path: Optional[str], rc: int) -> int:
    """--screen FILE: the framebuffer at the end of the run as a binary PPM (P6)."""
    if not path:
        return rc
    if not screen.w:
        print(localize("til: --screen: the program did not call screen_init (no image written)"), file=sys.stderr)
        return rc
    try:
        with open(path, 'wb') as f:
            f.write(screen.ppm())
    except OSError:
        print(localize(f"til: cannot write {path}"), file=sys.stderr)
        return rc or 2
    return rc


def cvm_screen_flags(opts: Dict[str, Any]) -> List[str]:
    """The same graphics options for the C VM runner (vm/main.c)."""
    flags = ['--screen', opts['screen']] if opts.get('screen') else []
    if opts.get('screen_ansi'):
        flags.append('--screen-ansi')
    if opts.get('keys') is not None:
        flags += ['--keys', opts['keys']]
    return flags


# ── feature commands and run options (TIL 3.1) ───────────────────────────────
# Larger tools live in their own modules next to til.py (src/); each exposes
# main(argv) -> exit code. `til <command> ...` dispatches to them.
EXTRA_COMMANDS: Dict[str, Tuple[str, str]] = {
    'pkg': ('tilpkg', "Packages: init | add NAME SPEC | install | remove NAME | list | verify"),
    'prove': ('tilprove', "Prove #[requires]/#[ensures] for all inputs (needs z3-solver)"),
    'fit': ('tilfit', "Will the program fit a device?  --device NAME [--run]"),
    'debug': ('tildebug', "Time-travel debugger: breakpoints, step, step back, reverse-continue"),
}

# Options of run/exec/build that take a value (stored in opts[key]) or are flags.
RUN_VALUE_OPTIONS = {
    '--screen': 'screen',            # graphics: write the final frame as a PPM image
    '--keys': 'keys',                # graphics: scripted key presses "UP,UP,ENTER,a"
    '--pause-after': 'pause_after',  # portable snapshots: pause after N steps ...
    '--save': 'save',                # ... and save the running program (.tilp)
    '--resume': 'resume',            # continue a saved program
    '--lang': 'lang',                # messages: en | ru | kk
    '--device': 'device',            # build --target vm: check the program fits a device
}
RUN_FLAG_OPTIONS = {
    '--screen-ansi': 'screen_ansi',  # graphics: draw the screen in the terminal
    '--prove': 'prove',              # check: prove contracts for all inputs
    '--no-fuse': 'no_fuse',          # run/build/dis: bytecode without superinstructions
    '--debug-info': 'debug_info',    # build/dis: add variable names (the debug-info section)
}


def feature_module(name: str):
    """Import a TIL tool module that lives next to til.py (src/tilpkg.py, ...).
    The tools `import til`; make that the module that is already running."""
    sys.modules.setdefault('til', sys.modules[__name__])
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import importlib
    return importlib.import_module(name)


# ── localized messages (TIL 3.1) ─────────────────────────────────────────────
# Diagnostics are written in English everywhere in this file. The command line
# translates what it prints (compile errors, warnings, hints, VM error reports)
# with the catalog in src/tilmsg.py. Language: --lang > TIL_LANG > the
# program's #[lang: ...] pragma (ru -> ru, kk / kk-latn -> kk) > English.
# Values that programs see (`error` in a retry/timeout else branch) stay English.
# Without tilmsg.py (or with one that cannot be loaded) everything stays English,
# and a note says so before the first diagnostic that would have been translated.

MESSAGE_LANGUAGES = ('en', 'ru', 'kk')
MESSAGE_LANG: Optional[str] = None      # set by main(); None / 'en' = English
_MESSAGE_LANG_FROM_USER = False          # --lang or TIL_LANG given (they beat the pragma)
_MESSAGE_CATALOG: List[Any] = []         # [tilmsg module or None, why it is None] once loaded
_CATALOG_NOTE_SHOWN = False              # the note about a missing catalog (once per main() call)


def message_language_code(value: Optional[str]) -> Optional[str]:
    """'ru', 'RU', 'ru_RU.UTF-8', 'kk', 'kk-latn', 'kz', 'en_US' -> 'ru' / 'kk' / 'en';
    None for anything else."""
    if not value:
        return None
    v = value.strip().lower()
    if v in ('kk-latn', 'kk_latn', 'kz'):
        return 'kk'
    v = re.split(r'[_.@-]', v, maxsplit=1)[0]
    return v if v in MESSAGE_LANGUAGES else None


def pragma_message_language(source: str) -> Optional[str]:
    """The message language named by the first `#[lang: ...]` pragma of a program."""
    for m in _LANG_PRAGMA_RE.finditer(source or ""):
        for code in m.group(1).split(','):
            lang = message_language_code(code) if code.strip().lower() in LANGUAGES else None
            if lang:
                return lang
    return None


def message_language(cli_value: Optional[str] = None, source: str = "",
                     environ: Optional[Dict[str, str]] = None) -> str:
    """--lang value > TIL_LANG > `#[lang: ...]` pragma of `source` > 'en'."""
    env = os.environ if environ is None else environ
    return (message_language_code(cli_value) or message_language_code(env.get('TIL_LANG'))
            or pragma_message_language(source) or 'en')


def message_catalog():
    """src/tilmsg.py, or None when it is missing or cannot be loaded (then
    messages stay English). Loading builds the catalog, so a broken entry shows
    up here and never in the middle of a diagnostic."""
    if not _MESSAGE_CATALOG:
        module, problem = None, None
        try:
            module = feature_module('tilmsg')
        except ImportError as e:
            if getattr(e, 'name', None) != 'tilmsg':      # tilmsg.py is there but imports something missing
                problem = _catalog_problem(e)
        except Exception as e:      # a syntax error, a missing comma between two entries, ...
            problem = _catalog_problem(e)
        if module is not None:
            try:
                module.entries()
            except Exception as e:  # a malformed template
                module, problem = None, f"tilmsg.py: {type(e).__name__}: {e}"
        _MESSAGE_CATALOG[:] = [module, problem]
    return _MESSAGE_CATALOG[0]


def _catalog_problem(e: Exception) -> str:
    """'tilmsg.py, line 73: TypeError: ...' for an exception raised while importing tilmsg.py."""
    line = e.lineno if isinstance(e, SyntaxError) else None
    text = e.msg if isinstance(e, SyntaxError) else str(e)
    tb = e.__traceback__
    while tb is not None:
        if os.path.basename(tb.tb_frame.f_code.co_filename) == 'tilmsg.py':
            line = tb.tb_lineno
        tb = tb.tb_next
    where = "tilmsg.py, line %d" % line if line else "tilmsg.py"
    return f"{where}: {type(e).__name__}: {text}"


def _catalog_note():
    """A translation was asked for but the catalog cannot be used: say so once
    (per main() call), before the first diagnostic that stays English."""
    global _CATALOG_NOTE_SHOWN
    if _CATALOG_NOTE_SHOWN:
        return
    _CATALOG_NOTE_SHOWN = True
    problem = _MESSAGE_CATALOG[1] if len(_MESSAGE_CATALOG) > 1 else None
    if problem is None:
        note = "note: message translations are not installed (tilmsg.py is missing); messages stay in English"
    else:
        note = f"note: message translations could not be loaded ({problem}); messages stay in English"
    print(note, file=sys.stderr, flush=True)


def localize(text: str, lang: Optional[str] = None) -> str:
    """Translate a compiler or command-line diagnostic (any number of lines) into
    `lang` (default: the language chosen by main()). Unknown messages stay English."""
    lang = MESSAGE_LANG if lang is None else lang
    if not text or not lang or lang == 'en':
        return text
    try:
        catalog = message_catalog()
        if catalog is None:
            _catalog_note()
            return text
        return catalog.localize(text, lang)
    except Exception:          # a failure in a translation must never hide the diagnostic
        return text


def localize_report_bytes(data: bytes, lang: Optional[str] = None) -> bytes:
    """A VM's uncaught-error report (bytes; invalid UTF-8 is kept as it is) in
    `lang`: the frame, the known message, the traceback. The rest of a program's
    own message is copied unchanged (tilmsg.ReportLocalizer)."""
    lang = MESSAGE_LANG if lang is None else lang
    if not data or not lang or lang == 'en':
        return data
    try:
        catalog = message_catalog()
        if catalog is None:
            _catalog_note()
            return data
        text = catalog.localize_report(data.decode('utf-8', 'surrogateescape'), lang)
        return text.encode('utf-8', 'surrogateescape')
    except Exception:
        return data


def set_message_language(args: List[str]) -> Optional[str]:
    """main(): choose the message language from --lang / TIL_LANG (a program's
    pragma is applied when its source is read). Returns an error for a bad --lang."""
    global MESSAGE_LANG, _MESSAGE_LANG_FROM_USER
    value = None
    for i, arg in enumerate(args):
        if arg == '--':
            break
        if arg == '--lang' and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith('--lang='):
            value = arg.split('=', 1)[1]
    env_value = os.environ.get('TIL_LANG', '').strip()
    if value is not None and message_language_code(value) is None:
        MESSAGE_LANG = message_language(None, "")
        return localize(f"Error: unknown message language '{value}' (use: en, ru, kk)")
    if env_value and value is None and message_language_code(env_value) is None:
        # (English: the language is not chosen yet; a program's pragma may still choose one)
        print(f"warning: TIL_LANG={env_value!r} is not a message language (en, ru, kk); it is ignored",
              file=sys.stderr)
    MESSAGE_LANG = message_language(value, "")
    _MESSAGE_LANG_FROM_USER = message_language_code(value) is not None or \
        message_language_code(env_value) is not None
    return None


def without_lang_option(args: List[str]) -> List[str]:
    """The arguments for an extra command (`til pkg`, `til debug`, ...): --lang is
    the compiler's option (the tool finds the language in til.MESSAGE_LANG)."""
    out: List[str] = []
    i = 0
    while i < len(args):
        if args[i] == '--':
            return out + args[i:]
        if args[i] == '--lang' and i + 1 < len(args):
            i += 2
            continue
        if not args[i].startswith('--lang='):
            out.append(args[i])
        i += 1
    return out


def scoped_message_language(func):
    """Decorator for main(): the message language it chooses lasts for that call
    only (main() also runs in-process, e.g. in tests; library use stays English)."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global MESSAGE_LANG, _MESSAGE_LANG_FROM_USER, _CATALOG_NOTE_SHOWN
        saved = (MESSAGE_LANG, _MESSAGE_LANG_FROM_USER, _CATALOG_NOTE_SHOWN)
        _CATALOG_NOTE_SHOWN = False
        try:
            return func(*args, **kwargs)
        finally:
            MESSAGE_LANG, _MESSAGE_LANG_FROM_USER, _CATALOG_NOTE_SHOWN = saved
    return wrapper


def use_program_message_language(source: str):
    """main(): a program's `#[lang: ru]` / `#[lang: kk]` pragma selects the message
    language unless --lang or TIL_LANG chose one."""
    global MESSAGE_LANG
    if not _MESSAGE_LANG_FROM_USER:
        MESSAGE_LANG = pragma_message_language(source) or 'en'


class _StderrTranslator:
    """The C VM's standard error in the message language, as it arrives: lines
    before the report one by one, the report's tail at once (linear time, the
    same translation as TILVM.report's: tilmsg.ReportLocalizer)."""

    def __init__(self, lang: str):
        catalog = message_catalog()
        self.usable = True
        try:
            self.loc = None if catalog is None else catalog.ReportLocalizer(lang)
        except Exception:          # a tilmsg.py from another version: English, as without one
            self.loc, self.usable = None, False

    def feed(self, data: bytes) -> bytes:
        if self.loc is None:
            if self.usable:
                _catalog_note()
            return data
        try:
            return self.loc.feed(data.decode('utf-8', 'surrogateescape')).encode('utf-8', 'surrogateescape')
        except Exception:          # never hide the report
            return data


def run_localized(cmd: List[str]) -> int:
    """Run a TIL executable (the C VM): with a non-English message language its
    standard error (runner messages, the uncaught-error report) is translated."""
    if not MESSAGE_LANG or MESSAGE_LANG == 'en':
        return run_foreground(cmd)
    ctrl_c = CtrlC()                 # like run_foreground: the child handles Ctrl-C itself
    ctrl_c.running = False
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    out = getattr(sys.stderr, 'buffer', None)

    def emit(data: bytes):
        if out is not None:
            out.write(data)
            out.flush()
        else:
            sys.stderr.write(data.decode('utf-8', 'replace'))
            sys.stderr.flush()
    try:
        translator = _StderrTranslator(MESSAGE_LANG)
        for line in iter(proc.stderr.readline, b''):
            emit(translator.feed(line))
            if translator.loc is None:                 # no catalog (after its note): copy as it comes
                for chunk in iter(lambda: proc.stderr.read1(65536), b''):
                    emit(chunk)
                break
            if translator.loc.in_report:               # the report's tail: at once, in linear time
                emit(translator.feed(proc.stderr.read()))
                break
    except BaseException:            # Ctrl-C, closed stderr, ...: do not leave the VM running
        proc.kill()
        raise
    finally:
        proc.stderr.close()
        rc = proc.wait()
        ctrl_c.restore()
    return rc


def run_extra_command(command: str, argv: List[str]) -> int:
    module_name = EXTRA_COMMANDS[command][0]
    try:
        module = feature_module(module_name)
    except ImportError as e:
        if getattr(e, 'name', None) != module_name:
            raise
        print(localize(f"til {command}: not available in this installation ({module_name}.py is missing)"),
              file=sys.stderr)
        return 1
    try:
        return module.main(argv)
    except BrokenPipeError:              # `til fit ... | head`: the reader is gone
        try:
            sys.stdout = open(os.devnull, 'w')
        except OSError:
            pass
        return 0


def run_proofs(program: 'Program', source: str, filename: str, explicit: bool) -> List[str]:
    """`til check`: prove the contracts of #[level: 4] functions (of every function
    with --prove). Returns formatted errors (empty = everything proved)."""
    try:
        prover = feature_module('tilprove')
    except ImportError as e:
        if getattr(e, 'name', None) != 'tilprove':
            raise
        return ["til check --prove: the prover (tilprove.py) is not installed"] if explicit else []
    return prover.check_program(program, source, filename, explicit)


def run_fit_check(data: bytes, device: str, filename: str) -> bool:
    """`til build --target vm|c-vm --device NAME`: does the built bytecode fit the
    device? Prints the result; False = it does not (the build fails)."""
    try:
        fit = feature_module('tilfit')
    except ImportError as e:
        if getattr(e, 'name', None) != 'tilfit':
            raise
        print(localize("Error: --device needs the fit checker (tilfit.py), which is not installed"),
              file=sys.stderr)
        return False
    return fit.check_build(data, device, filename)


# ── translate ────────────────────────────────────────────────────────────────

def translate_source(source: str, target: str, filename: str = "<stdin>") -> str:
    """Rewrite TIL source with the keywords of another language (en, kk,
    kk-latn, ru). Identifiers, strings and comments are kept as they are."""
    if target not in LANGUAGES:
        raise CompileError(f"unknown language '{target}' (use: {', '.join(LANGUAGES)})")
    lexer = Lexer(source, filename)
    tokens = lexer.tokenize()
    # contextual keywords (with/resume, localized perform/handle) are known only to
    # the parser: it retypes the identifier tokens it reads as keywords. A file that
    # does not parse keeps those words as they are.
    try:
        Parser(list(tokens), filename, source=source, languages=lexer.languages).parse()
    except Exception:          # noqa: BLE001 - best effort, see above
        pass
    kw_table = ENGLISH_KEYWORDS if target == 'en' else LANGUAGES[target][0]
    alias_table = LANGUAGES[target][1]
    word_for: Dict[TokenType, str] = {}
    for word, ttype in ENGLISH_KEYWORDS.items():
        word_for.setdefault(ttype, word)
    if target != 'en':
        for word, ttype in kw_table.items():
            if ttype not in word_for or word_for[ttype] in ENGLISH_KEYWORDS:
                word_for[ttype] = word
    for word, ttype in CONTEXTUAL_KEYWORDS.get(target, {}).items():
        word_for[ttype] = word
    ctx_words = {w for tbl in CONTEXTUAL_KEYWORDS.values() for w in tbl}
    true_word = next((w for w, t in kw_table.items() if t == TokenType.BOOL and w in TRUE_WORDS), 'true')
    false_word = next((w for w, t in kw_table.items() if t == TokenType.BOOL and w not in TRUE_WORDS), 'false')
    if target == 'en':
        true_word, false_word = 'true', 'false'
    canonical_to_alias: Dict[str, str] = {}
    for alias, canon in alias_table.items():
        canonical_to_alias.setdefault(canon, alias)      # first spelling is preferred
    replacements = []
    skip = set()
    for k, tok in enumerate(tokens):
        if tok.start < 0 or tok.end <= tok.start or k in skip:
            continue
        text = source[tok.start:tok.end]
        nxt = tokens[k + 1] if k + 1 < len(tokens) else None
        if (target == 'en' and tok.type == TokenType.ELSE and nxt is not None
                and nxt.type == TokenType.IF and nxt.line == tok.line):
            replacements.append((tok.start, nxt.end, "elif"))   # else if -> elif
            skip.add(k + 1)
            continue
        if tok.type == TokenType.BOOL:
            new = true_word if tok.value else false_word
        elif tok.type == TokenType.ELIF and target != 'en':
            new = f"{word_for[TokenType.ELSE]} {word_for[TokenType.IF]}"
        elif tok.type == TokenType.IDENT:
            canon = tok.value
            new = canonical_to_alias.get(canon, canon) if target != 'en' or text != canon else text
            if target == 'en':
                new = canon
        elif text in ENGLISH_KEYWORDS or text in KEYWORDS or text in ctx_words or \
                any(text in LANGUAGES[c][0] for c in LANGUAGES):
            new = word_for.get(tok.type, text)
        else:
            continue
        if new != text:
            replacements.append((tok.start, tok.end, new))
    out = source
    for start, end, new in sorted(replacements, reverse=True):
        out = out[:start] + new + out[end:]
    out = _LANG_PRAGMA_RE.sub('', out).lstrip('\n')
    if target in ('ru', 'kk-latn'):
        out = f"#[lang: {target}]\n" + out
    return out


# ── REPL (on the VM: instant, no C compiler needed) ─────────────────────────

_REPL_MARK = "\x00til-repl\x00"
_REPL_DEF_RE = re.compile(r'^(struct|enum|impl|trait|fn|pub|cap)\b|^#\[|'
                          r'^[^\W\d]\w*\s*\([^)]*\)\s*->|^[^\W\d]\w*\s*\(\s*\w+\s*:')
# a line starting with `handle` or ending in `handle` (any language: the localized
# words are contextual, `өңдеу(x)` is a call) opens a multi-line block
_REPL_HANDLE_RE = re.compile(r'^handle\b|\b(%s)$' % '|'.join(HANDLE_WORDS))


def repl_program(definitions: List[str], history: List[str], block: str) -> Tuple[str, bool]:
    """Build the program for one REPL input. Returns (source, is_expression)."""
    body = "\n".join("    " + l for l in block.split("\n"))
    probe = parse_source("main()\n" + body + "\n", "<repl>")
    stmts = next(s for s in probe.statements if isinstance(s, FuncDef) and s.name == "main").body.statements
    is_expr = (len(stmts) == 1 and isinstance(stmts[0], ExprStmt) and not (
        isinstance(stmts[0].expr, Call) and isinstance(stmts[0].expr.func, Identifier)
        and stmts[0].expr.func.name in ('print', 'println')))
    if is_expr:
        # a multi-line expression (`handle` + body + `with` + clauses): its other
        # lines keep their indentation relative to the first one
        lines = block.rstrip().split("\n")
        ind = len(lines[0]) - len(lines[0].lstrip())
        expr = "\n".join([lines[0].strip()] + [
            "    " + (l[ind:] if not l[:ind].strip() else l.lstrip()) for l in lines[1:]])
        body = f"    let __v = {expr}\n    if __v != None\n        print(__v)"
    replay = "\n".join("    " + l for h in history for l in h.split("\n"))
    source = "\n".join(definitions) + "\nmain()\n" + (replay + "\n" if replay else "") + \
        f'    print("{_REPL_MARK}")\n' + body + "\n"
    return source, is_expr


def run_repl():
    """Interactive REPL on the VM (instant, no C compiler needed).
    Definitions persist; earlier statements are replayed silently so that
    variables persist too."""
    print(f"TIL {TIL_VERSION} REPL - The Intelligent Language (VM)")
    print('Expressions print their value. A blank line ends a multi-line block. "exit" quits.')
    definitions: List[str] = []
    history: List[str] = []
    out = getattr(sys.stdout, 'buffer', None)

    def read_block(first: str) -> str:
        lines = [first]
        while True:
            try:
                more = input("...  ")
            except (EOFError, KeyboardInterrupt):
                break
            if not more.strip():
                break
            lines.append(more)
        return "\n".join(lines)

    while True:
        try:
            line = input("til> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0
        stripped = line.strip()
        if not stripped or (stripped.startswith('#') and not stripped.startswith('#[')):
            continue
        if stripped in ('exit', 'quit', 'exit()', 'quit()'):
            print("Bye!")
            return 0
        if re.match(r'^(import|from|type)\b', stripped) or _REPL_DEF_RE.match(stripped):
            block = line if re.match(r'^(import|from|type)\b', stripped) else read_block(line)
            try:
                compile_to_bytecode("\n".join(definitions + [block]) + "\n", "<repl>")
            except (SyntaxError, CompileError) as e:
                print(localize(str(e)))
                continue
            definitions.append(block)
            print("ok")
            continue
        if re.match(r'^(if|for|while|loop|match|intent)\b', stripped) or \
                _REPL_HANDLE_RE.search(stripped):        # `handle` / `let x = handle` + body + with
            block = read_block(line)
        else:
            block = line
        try:
            source, is_expr = repl_program(definitions, history, block)
            module = compile_to_bytecode(source, "<repl>")
        except (SyntaxError, CompileError) as e:
            print(localize(str(e)))
            continue
        state = {'live': False, 'buf': b""}
        mark = _REPL_MARK.encode() + b"\n"

        def write(data: bytes):
            if state['live']:
                out.write(data) if out else sys.stdout.write(data.decode('utf-8', 'replace'))
                return
            state['buf'] += data
            if mark in state['buf']:
                state['live'] = True
                rest = state['buf'].split(mark, 1)[1]
                if rest:
                    write(rest)

        rc = TILVM(module, write=write, max_steps=50_000_000).run()
        sys.stdout.flush()
        if rc == 0 and not is_expr:
            history.append(block)


# ── main ─────────────────────────────────────────────────────────────────────

_COMMANDS = ('run', 'build', 'check', 'repl', 'exec', 'dis', 'translate', 'version')
_VALUE_FLAGS = ('-o', '--target', '--to', '--max-steps')


def _command_first(args: List[str]) -> List[str]:
    """Options written before the command (`til -v --lang ru run x.til`) move
    behind it. Only leading options are moved; nothing after the command changes."""
    i = 0
    while i < len(args) and args[i].startswith('-') and args[i] != '--':
        i += 2 if (args[i] in _VALUE_FLAGS or args[i] in RUN_VALUE_OPTIONS) else 1
    if 0 < i < len(args) and (args[i] in _COMMANDS or args[i] in EXTRA_COMMANDS):
        return [args[i]] + args[:i] + args[i + 1:]
    return args


@scoped_message_language
def main():
    """The `til` command line. `--no-types` switches type inference off for this
    call only (the environment is restored afterwards, also for in-process callers)."""
    saved = os.environ.get('TIL_NO_TYPES')
    try:
        return _main()
    finally:
        if saved is None:
            os.environ.pop('TIL_NO_TYPES', None)
        else:
            os.environ['TIL_NO_TYPES'] = saved


def _main():
    args = sys.argv[1:]

    args = _command_first(args)                    # `til --lang ru run x.til` == `til run --lang ru x.til`
    lang_error = set_message_language(args)       # messages: --lang / TIL_LANG (pragma: below)
    if lang_error:
        print(lang_error, file=sys.stderr)
        return 1

    if args and args[0] in EXTRA_COMMANDS:      # before -h/--help: `til pkg --help` is the tool's own help
        return run_extra_command(args[0], without_lang_option(args[1:]))

    til_options = args[:args.index('--')] if '--' in args else args     # after `--`: the program's
    if not args or '-h' in til_options or '--help' in til_options:
        print_help()
        return 0

    if '--version' in til_options or args[0] == 'version':
        print(f"TIL Compiler v{TIL_VERSION}")
        print("Author: Alisher Beisembekov")
        print("Backends: native (C via gcc/clang), TIL VM (bytecode: Python, C99 / calculators)")
        return 0

    command = None
    if args and args[0] in ('run', 'build', 'check', 'repl', 'exec', 'dis', 'translate'):
        command = args[0]
        args = args[1:]

    if command == 'repl':
        return run_repl()

    compiler = TILCompiler()
    input_file = None
    output_file = None
    target = None
    use_vm = None          # None = auto, 'py', 'c'
    max_steps = 0
    strip = False
    lang_to = None
    opts: Dict[str, Any] = {}       # RUN_VALUE_OPTIONS / RUN_FLAG_OPTIONS
    prog_args: List[str] = []       # arguments for the program (args())

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--':
            prog_args.extend(args[i + 1:])
            break
        if arg in RUN_VALUE_OPTIONS and i + 1 < len(args):
            opts[RUN_VALUE_OPTIONS[arg]] = args[i + 1]
            i += 2
            continue
        if arg.startswith('--') and arg.split('=', 1)[0] in RUN_VALUE_OPTIONS and '=' in arg:
            key, value = arg.split('=', 1)
            opts[RUN_VALUE_OPTIONS[key]] = value
            i += 1
            continue
        if arg in RUN_FLAG_OPTIONS:
            opts[RUN_FLAG_OPTIONS[arg]] = True
            i += 1
            continue
        if arg == '-o' and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
            continue
        if arg == '--target' and i + 1 < len(args):
            target = args[i + 1]
            i += 2
            continue
        if arg.startswith('--target='):
            target = arg.split('=', 1)[1]
        elif arg == '--to' and i + 1 < len(args):
            lang_to = args[i + 1]
            i += 2
            continue
        elif arg.startswith('--to='):
            lang_to = arg.split('=', 1)[1]
        elif arg == '--max-steps' and i + 1 < len(args):
            try:
                max_steps = int(args[i + 1])
            except ValueError:
                max_steps = -1
            if max_steps < 0:
                print(localize(f"Error: --max-steps needs a number >= 0, got '{args[i + 1]}'"), file=sys.stderr)
                return 2
            i += 2
            continue
        elif arg == '-c':
            target = 'c'
        elif arg == '--vm':
            use_vm = 'py'
        elif arg == '--cvm':
            use_vm = 'c'
        elif arg == '--strip':
            strip = True
        elif arg == '--keep-c':
            compiler.keep_c = True
        elif arg == '--no-check':
            compiler.check_types = False
        elif arg == '--no-verify':
            compiler.verify = False
        elif arg == '--no-types':
            compiler.typed = False          # no type inference: classic C backend, no E-TYPE
            os.environ['TIL_NO_TYPES'] = '1'
        elif arg in ('-v', '--verbose'):
            compiler.verbose = True
        elif arg in ('-O0', '-O1', '-O2', '-O3'):
            compiler.optimize = int(arg[2])
        elif input_file is None and not arg.startswith('-'):
            input_file = arg
        elif input_file is not None:
            prog_args.append(arg)        # after the program file: the program's arguments
        else:
            print(localize(f"Error: unknown option {arg}"), file=sys.stderr)
            return 1
        i += 1

    if not input_file:
        print(localize("Error: No input file specified"), file=sys.stderr)
        return 1
    if not os.path.exists(input_file):
        print(localize(f"Error: File not found: {input_file}"), file=sys.stderr)
        return 1
    if opts.get('device') and (command != 'build' or target not in ('vm', 'tbc', 'c-vm', 'calc', 'embed')
                               or input_file.endswith('.tbc')):
        print(localize("Error: --device works with `til build --target vm|c-vm` on a .til source; for a report "
                       "on any program (also .tbc) use `til fit FILE --device NAME`"), file=sys.stderr)
        return 1

    try:
        # bytecode files
        if input_file.endswith('.tbc'):
            with open(input_file, 'rb') as f:
                module = TBCModule.from_bytes(f.read())
            if command == 'dis':
                print(disassemble(module))
                return 0
            if use_vm == 'c':                          # til exec prog.tbc --cvm: the C VM runs the file
                try:
                    snap_flags = cvm_snapshot_flags(opts)
                except ValueError as e:
                    print(localize(f"Error: {e}"), file=sys.stderr)
                    return 2
                exe = _cvm_binary(compiler.verbose)
                if exe is None:
                    print(localize("Error: cannot build the C VM (need vm/ sources and a C compiler)"),
                          file=sys.stderr)
                    return 1
                cmd = [exe] + (['--max-steps', str(max_steps)] if max_steps else []) + \
                    cvm_screen_flags(opts) + snap_flags + [input_file] + prog_args
                return foreground_exit(run_localized(cmd))
            return run_bytecode(module, max_steps, prog_args, opts)

        with open(input_file, 'r', encoding='utf-8') as f:
            source = f.read()
        use_program_message_language(source)          # #[lang: ru] -> Russian messages
        fuse = not opts.get('no_fuse')

        if command == 'check':
            ast = resolve_imports(parse_source(source, input_file), input_file)
            errors = SemanticChecker(source, input_file).check(ast)
            if errors:
                for err in errors:
                    print(localize(err), file=sys.stderr)
                return 1
            BytecodeCompiler(source, input_file).compile_program(ast)   # names, arity, immutability
            errors = type_errors(ast, source, input_file) if compiler.typed else []
            if errors:
                for err in errors:
                    print(localize(err), file=sys.stderr)
                return 1
            proof_errors = run_proofs(ast, source, input_file, bool(opts.get('prove')))
            if proof_errors:
                for err in proof_errors:
                    print(localize(err), file=sys.stderr)
                return 1
            print(f"OK: {input_file}")
            return 0

        if command == 'translate':
            text = translate_source(source, lang_to or 'en', input_file)
            if output_file:
                with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(text)
                print(f"Translated: {output_file}")
            else:
                sys.stdout.write(text)
            return 0

        if command == 'dis':
            print(disassemble(compile_to_bytecode(source, input_file, compiler.verify, fuse,
                                                  bool(opts.get('debug_info')))))
            return 0

        base = os.path.splitext(os.path.basename(input_file))[0]

        if command == 'run' and use_vm is None and compiler.find_c_compiler() is None:
            print(localize("[TIL] No C compiler found - running on the TIL VM"), file=sys.stderr)
            use_vm = 'py'
        if command == 'run' and use_vm is None and any(o in opts for o in GFX_RUN_OPTIONS):
            use_vm = 'py'           # screen/keys options: the VM implements the graphics natives
        if command == 'run' and use_vm is None and any(o in opts for o in SNAPSHOT_RUN_OPTIONS):
            use_vm = 'py'           # portable snapshots: only the VMs can pause and resume

        if command == 'run' and use_vm:
            module = compile_to_bytecode(source, input_file, compiler.verify, fuse)
            if use_vm == 'py':
                return run_bytecode(module, max_steps, prog_args, opts)
            try:
                snap_flags = cvm_snapshot_flags(opts)
            except ValueError as e:
                print(localize(f"Error: {e}"), file=sys.stderr)
                return 2
            exe = _cvm_binary(compiler.verbose)
            if exe is None:
                print(localize("Error: cannot build the C VM (need vm/ sources and a C compiler)"),
                      file=sys.stderr)
                return 1
            fd, tbc_path = tempfile.mkstemp(suffix='.tbc')
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(module.serialize())
                cmd = [exe] + (['--max-steps', str(max_steps)] if max_steps else []) + \
                    cvm_screen_flags(opts) + snap_flags + [tbc_path] + prog_args
                rc = run_localized(cmd)                # C VM error reports in the message language
            finally:
                os.unlink(tbc_path)
            return foreground_exit(rc)

        if target in ('vm', 'tbc'):
            module = compile_to_bytecode(source, input_file, compiler.verify, fuse,
                                         bool(opts.get('debug_info')))
            data = module.serialize(strip=strip)
            out = output_file or base + '.tbc'
            if opts.get('device') and not run_fit_check(data, opts['device'], input_file):
                print(localize(f"Build failed: {out} not written"), file=sys.stderr)
                return 1
            with open(out, 'wb') as f:
                f.write(data)
            print(f"Built: {out} ({len(data)} bytes of TIL bytecode)")
            return 0

        if target in ('c-vm', 'calc', 'embed'):
            module = compile_to_bytecode(source, input_file, compiler.verify, fuse,
                                         bool(opts.get('debug_info')))
            data = module.serialize(strip=strip)
            if opts.get('device') and not run_fit_check(data, opts['device'], input_file):
                print(localize("Build failed: nothing written"), file=sys.stderr)
                return 1
            text = amalgamate_c_vm(data, input_file)
            out = output_file or base + '_vm.c'
            with open(out, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
            print(f"Built: {out} (TIL VM + program in one C file; compile with any C99 compiler)")
            return 0

        if target == 'c':
            c_code = compiler.compile(source, input_file)
            if output_file:
                with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(c_code)
                print(f"Generated: {output_file}")
            else:
                print(c_code)
            return 0

        if target not in (None, 'native'):
            print(localize(f"Error: unknown target '{target}' (native, vm, c-vm, c)"), file=sys.stderr)
            return 1

        # native build (C backend)
        if not output_file:
            output_file = base
        exe_file = output_file
        if sys.platform == 'win32' and not exe_file.endswith('.exe'):
            exe_file = output_file + '.exe'
        if command == 'run':
            fd, tmp_exe = tempfile.mkstemp(suffix='.exe' if sys.platform == 'win32' else '')
            os.close(fd)
            try:
                if not compiler.compile_to_executable(source, tmp_exe, input_file, quiet=True):
                    # The native backend covers a subset of TIL; the VM runs everything.
                    print(localize("[TIL] native backend cannot compile this program; running it on the TIL VM"),
                          file=sys.stderr)
                    if compiler.verbose:
                        print(getattr(compiler, 'last_c_error', ''), file=sys.stderr)
                    return run_bytecode(compile_to_bytecode(source, input_file, compiler.verify, fuse), max_steps,
                                        prog_args, opts)
                return subprocess.run([tmp_exe] + prog_args).returncode
            finally:
                try:
                    os.unlink(tmp_exe)
                except OSError:
                    pass
        if not compiler.compile_to_executable(source, output_file, input_file):
            return 1
        print(f"Compiled: {exe_file}")
        return 0

    except SyntaxError as e:
        print(localize(f"Syntax Error: {e}"), file=sys.stderr)
        return 1
    except CompileError as e:
        print(localize(str(e)), file=sys.stderr)
        return 1
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, 'w')
        except OSError:
            pass
        return 0
    except (ValueError, OSError) as e:
        print(localize(f"Error: {e}"), file=sys.stderr)
        return 1
    except Exception as e:
        print(localize(f"Error: {e}"), file=sys.stderr)
        if compiler.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
