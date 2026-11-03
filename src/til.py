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

TIL_VERSION = "3.0.0"

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
}

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
class EffectHandler(ASTNode):
    """effect handler block: handle expr with { effect_name(args) => body }"""
    expr: ASTNode = None
    handlers: List[tuple] = field(default_factory=list)  # [(name, params, body), ...]

@dataclass
class PerformEffect(ASTNode):
    """perform EffectName(args) - trigger an effect"""
    name: str = ""
    args: List[ASTNode] = field(default_factory=list)

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
        self.error_reporter = ErrorReporter(source, filename)
        self.languages = languages

    @staticmethod
    def _mark(node, tok: Token):
        """Attach a source position to an AST node (if it has none yet)."""
        if isinstance(node, ASTNode) and not node.line and tok is not None:
            node.line = tok.line
            node.col = tok.col
        return node

    def error(self, msg: str):
        tok = self.current()
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
        if msg:
            self.error(msg)
        self.error(f"Expected {type.name}, got {self.current().type.name}")
    
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
            self.skip_newlines()

        return Program(statements=self.hoist_script_statements(statements))

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
        while self.match(TokenType.ATTRIBUTE):
            attr = self.advance().value
            self.pending_attributes.append(attr)
            
            # Parse level
            if attr.startswith("level:"):
                try:
                    self.current_level = int(attr.split(":")[1].strip())
                except:
                    pass
            
            self.skip_newlines()
        
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
        return self.parse_func_body(name)
    
    def parse_func_def_simple(self) -> FuncDef:
        name = self.consume(TokenType.IDENT).value
        return self.parse_func_body(name)
    
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
                # Energy annotation: #[energy: low|medium|high]
                energy_level = attr[len("energy:"):].strip()
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

        if self.match(TokenType.PERFORM):
            return ExprStmt(expr=self.parse_perform())

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
        
        name = self.consume(TokenType.IDENT).value
        
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
        var = self.consume(TokenType.IDENT).value
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

        return MatchExpr(value=value, arms=arms)

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
        self.consume(TokenType.PERFORM)
        name = self.consume(TokenType.IDENT).value
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
        """Parse: handle { body } with { EffectName(params) => handler_body }"""
        self.consume(TokenType.HANDLE)
        self.skip_newlines()
        expr = self.parse_block() if self.match(TokenType.INDENT) else self.parse_expression()
        # Expect 'with' as an identifier (not a keyword)
        # We'll check for it manually
        self.skip_newlines()
        handlers = []
        # If there's a 'with' identifier followed by handlers, parse them
        # For simplicity, return the handler with empty handlers list
        return EffectHandler(expr=expr, handlers=handlers)

    def parse_return(self) -> Return:
        self.consume(TokenType.RETURN)
        
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
                self.error("Expected method definition in impl block")
            
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
                self.error("Expected method definition in trait")
            
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
            if self.match(TokenType.LBRACE):
                self.advance()
                self.skip_newlines()
                body = self.parse_expression()
                self.skip_newlines()
                self.consume(TokenType.RBRACE)
            else:
                body = self.parse_expression()
            return Lambda(params=params, body=body, ret_type=ret_type)
        
        # Self keyword (used in method bodies)
        if self.match(TokenType.SELF):
            self.advance()
            return Identifier(name="self")

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
        if node.op == '~' and isinstance(v, int):
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
            return (a == b) if node.op == '==' else (a != b)
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
                   'key_pressed', 'wait_key', 'energy_mode'}

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

    def scan_function(self, key: str, fn: FuncDef):
        effects: Set[str] = set()
        edges: List[Tuple[str, ASTNode, bool]] = []
        callee_nodes = set()
        for node in walk(fn.body):
            if isinstance(node, Call):
                if isinstance(node.func, Identifier) and node.func.name in self.IO_BUILTINS:
                    effects.add('io')
                    self.effect_origin.setdefault((key, 'io'), node)
                for callee, precise in self.resolve_call(key, node):
                    edges.append((callee, node, precise))
                    if precise:
                        self.check_call_site(callee, node)
                callee_nodes.add(id(node.func))
            elif isinstance(node, PerformEffect):
                effects.add(node.name)
                self.effect_origin.setdefault((key, node.name), node)
        # Taking a function as a value (e.g. `let f = read_file`) counts as a
        # reference for capability purposes: no capability laundering.
        for node in walk(fn.body):
            if isinstance(node, Identifier) and id(node) not in callee_nodes \
                    and node.name in self.funcs and self.funcs[node.name].caps:
                edges.append((node.name, node, True))
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
                    new = callee_eff - inferred[key]
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
                for eff in callee_eff - allowed:
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
            for callee, node, precise in edges:
                if not precise:
                    continue
                missing = [c for c in self.funcs[callee].caps if c not in held]
                for cap in missing:
                    self.error(node,
                               f"'{key}' uses '{callee}', which requires capability '{cap}', "
                               f"but '{key}' does not hold it",
                               "E-CAP", f"add #[cap: {cap}] to '{key}' (only main holds every capability)")

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
        """Generate effect invocation as function call."""
        name = self.mangle_name(node.name)
        args = ", ".join(self.generate_node(a) for a in node.args)
        return f"til_{name}({args})"

    def gen_EffectHandler(self, node: EffectHandler) -> str:
        """Generate effect handler as try-execute block."""
        self.emit("// Effect handler")
        self.emit("{")
        self.indent += 1
        if isinstance(node.expr, Block):
            self.generate_node(node.expr)
        else:
            val = self.generate_node(node.expr)
            if val:
                self.emit(f"{val};")
        self.indent -= 1
        self.emit("}")
        return ""

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
    # 0x80-0xBF: reserved for superinstructions (§13)


OP_NAMES = {v: k for k, v in vars(Op).items() if not k.startswith('_')}

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
}
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
        self.version = TBC_VERSION           # of a loaded file (written files: TBC_VERSION)

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

    # ── serialization ───────────────────────────────────────────────────────
    def to_bytes(self, strip: bool = False) -> bytes:
        """Write the module (call serialize(): it interns all names first)."""
        self.link()
        out = bytearray()
        flags = (1 if self.uses_float else 0) | (0 if strip else 2)
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
        return bytes(out)

    def serialize(self, strip: bool = False) -> bytes:
        for f in self.funcs:
            self.const(f.name)
        for name, fields in self.types:
            self.const(name)
            for fld in fields:
                self.const(fld)
        return self.to_bytes(strip)

    @staticmethod
    def from_bytes(data: bytes) -> 'TBCModule':
        m = TBCModule()
        m.types = []
        if data[:4] != TBC_MAGIC:
            raise ValueError("not a TIL bytecode file (bad magic)")
        if not TBC_MIN_VERSION <= data[4] <= TBC_VERSION:
            raise ValueError(f"unsupported bytecode version {data[4]} "
                             f"(this VM loads versions {TBC_MIN_VERSION}..{TBC_VERSION})")
        m.version = data[4]
        flags = data[5]
        m.uses_float = bool(flags & 1)
        pos = 6
        def rd(fmt):
            nonlocal pos
            vals = _struct.unpack_from('<' + fmt, data, pos)
            pos += _struct.calcsize('<' + fmt)
            return vals if len(vals) > 1 else vals[0]
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
                value = bytes(data[pos:pos + n])
                pos += n
            else:
                raise ValueError(f"bad constant tag {tag}")
            key = (('f', _struct.pack('<d', value)) if isinstance(value, float) else
                   ('i', value) if isinstance(value, int) else ('s', value))
            m._const_index.setdefault(key, len(m.consts))
            m.consts.append(value)
        for _ in range(rd('H')):
            name, np_, nr, nl, nu, fl, off = rd('HBBBBBI')
            f = TBCFunction(m.consts[name].decode('utf-8', 'replace'))
            f.n_params, f.n_required, f.n_locals, f.n_upvals, f.flags, f.offset = np_, nr, nl, nu, fl, off
            m.funcs.append(f)
        for _ in range(rd('H')):
            name, nf = rd('HB')
            fields = [m.consts[rd('H')].decode('utf-8', 'replace') for _ in range(nf)]
            m.types.append((m.consts[name].decode('utf-8', 'replace'), fields))
        for _ in range(rd('H')):
            m.methods.append(rd('HHH'))
        for _ in range(rd('B')):
            mid, fidx = rd('BH')
            m.prelude[mid] = fidx
        n = rd('I')
        m.code = bytes(data[pos:pos + n])
        pos += n
        for _ in range(rd('H')):
            m.lines.append(rd('IH'))
        # split the code section back into functions (so the module can be re-linked)
        order = sorted(range(len(m.funcs)), key=lambda i: m.funcs[i].offset)
        for k, i in enumerate(order):
            f = m.funcs[i]
            end = m.funcs[order[k + 1]].offset if k + 1 < len(order) else len(m.code)
            f.code = bytearray(m.code[f.offset:end])
            f.lines = [(pc - f.offset, line) for pc, line in m.lines if f.offset <= pc < end]
        return m


# ═══════════════════════════════════════════════════════════════════════════════
#                         BYTECODE COMPILER (AST -> .tbc)
# ═══════════════════════════════════════════════════════════════════════════════

class _Local:
    __slots__ = ('slot', 'mutable', 'celled', 'name')

    def __init__(self, name, slot, mutable, celled):
        self.name, self.slot, self.mutable, self.celled = name, slot, mutable, celled


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


class BytecodeCompiler:
    """Compiles a (resolved) TIL Program into a TBCModule."""

    def __init__(self, source: str = "", filename: str = "<stdin>"):
        self.reporter = ErrorReporter(source, filename)
        self.filename = filename
        self.m = TBCModule()

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
        return len(code)

    def emit_jump(self, op: int, node=None) -> int:
        """Emit a forward jump (single i16 operand); returns the patch position."""
        self.emit(op, 0, node=node)
        return len(self.fs.proto.code) - 2

    def patch_jump(self, at: int, target: Optional[int] = None):
        code = self.fs.proto.code
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
        self.fs.scopes.pop()
        # slots are not reused, which keeps closures/cells simple and safe

    def declare_local(self, name: str, mutable: bool = True, node=None) -> _Local:
        fs = self.fs
        slot = fs.next_slot
        fs.next_slot += 1
        if fs.next_slot > 250:
            self.error(node, "too many local variables in one function (max 250)")
        fs.proto.n_locals = max(fs.proto.n_locals, fs.next_slot)
        local = _Local(name, slot, mutable, name in fs.captured)
        fs.scopes[-1][name] = local
        return local

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

    # ── program ─────────────────────────────────────────────────────────────
    def compile_program(self, program: Program) -> TBCModule:
        m = self.m
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

        # compile bodies (lambdas and prelude functions are appended as we go)
        i = 0
        while i < len(self.pending):
            fidx, fdef, owner = self.pending[i]
            self.compile_function(fidx, fdef, owner)
            i += 1
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
        """Names referenced inside lambdas (they must live in cells)."""
        names: Set[str] = set()
        for node in walk(body):
            if isinstance(node, Lambda):
                params = {p for p, _ in node.params}
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
               'try_depth': self.fs.try_depth}
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
        if not fs.loops:
            self.error(node, f"'{'break' if is_break else 'continue'}' outside of a loop")
        ctx = fs.loops[-1]
        for _ in range(fs.try_depth - ctx['try_depth']):
            self.emit(Op.END_TRY)
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

    # ── expressions ─────────────────────────────────────────────────────────
    def try_fold(self, node):
        if isinstance(node, (BinaryOp, UnaryOp)):
            try:
                v = const_eval(node, self.inline_consts)
            except (NotConstant, OverflowError, ValueError, ZeroDivisionError):
                return None
            if isinstance(v, (bool, int, float, str)):
                return v
        return None

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
            if isinstance(node.expr, Block):
                self.compile_block_value(node.expr)
            else:
                self.compile_expr(node.expr, node)
        elif t is PerformEffect:
            if node.name not in self.funcs:
                self.error(node, f"no handler function '{node.name}' for effect '{node.name}'")
            for a in node.args:
                self.compile_expr(a, node)
            self.emit(Op.CALL, self.funcs[node.name], len(node.args), node=node)
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


def til_format_spec(v, spec: bytes, to_str) -> bytes:
    """f-string format specs: [<>^][0][width][.prec][f|e|%|d|x|X|o|b]."""
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
        n = int(v)
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


class TILVM:
    """Reference interpreter for TIL bytecode modules."""

    CHECK_INTERVAL = 256
    MAX_FRAMES = 2000

    def __init__(self, module: TBCModule, write=None, read_line=None, max_steps: int = 0,
                 argv: Optional[List[Any]] = None):
        self.m = module
        # program arguments for args(): bytes (str is encoded as UTF-8)
        self.argv = [a.encode('utf-8') if isinstance(a, str) else bytes(a) for a in (argv or [])]
        self.code = module.code
        self.consts = module.consts
        self._own_stdout = write is None
        self.write = write or self._stdout_write
        self.read_line = read_line or self._stdin_line
        self.max_steps = max_steps
        self.steps = 0
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
        self.recover: List[list] = []     # [handler_pc, n_frames, sp, deadline_ms or 0]
        self.deadlines = 0
        self.tick = 0
        self.stack: list = []
        self.frames: List[list] = []      # [func, pc, base, closure, extra]

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
            if op == Op.MOD:
                return _math.fmod(a, b) if b != 0.0 else float('nan')
            if op == Op.POW:
                try:
                    return _math.pow(a, b)
                except (OverflowError, ValueError):
                    return float('inf') if a > 1 else float('nan')
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
        if tk is float:
            return int(k) if k == int(k) else k
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
            return til_format_spec(args[0], args[1], self.to_str)
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
        self.panic(f"{name}() is not available in this VM build")

    def path_arg(self, who: str, v) -> str:
        """A file path argument (same rules as the C VM: non-empty, < 256 bytes, no NUL)."""
        if type(v) is not bytes:
            self.panic(f"{who}() expects a path string, got {self.type_name(v).decode()}")
        if not v or len(v) >= 256 or b"\0" in v:
            self.panic(f"{who}(): invalid path")
        return os.fsdecode(v)

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

    def run(self) -> int:
        """Run the module; returns the process exit code."""
        entry = self.m.funcs[self.m.entry]
        self.frames = [[self.m.entry, entry.offset, 0, None, 0]]
        self.stack = [None] * entry.n_locals
        try:
            return self._run()
        finally:
            self.flush()

    def _run(self) -> int:
        try:
            while True:
                try:
                    self.execute()
                    return 0
                except TILPanic as p:
                    if not self.recover_from(p):
                        self.report(b"runtime error: " + p.msg, p.pc)
                        return 1
        except TILFatal as f:
            self.report(b"fatal error: " + f.msg, f.pc)
            return 1
        except TILExit as e:
            return e.code
        except RecursionError:
            self.report(b"fatal error: native recursion limit", -1)
            return 1

    def report(self, msg: bytes, pc: int):
        out = getattr(sys.stderr, 'buffer', None)
        text = msg + b"\n" + (self.traceback(pc) + b"\n" if pc >= 0 else b"")
        self.flush()
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
        handler, n_frames, sp, deadline = self.recover.pop()
        if deadline:
            self.deadlines -= 1
        del self.frames[n_frames:]
        del self.stack[sp:]
        self.stack.append(p.msg)
        self.frames[-1][1] = handler
        return True

    def check_budget(self, pc: int):
        self.steps += 1
        if self.max_steps and self.steps > self.max_steps:
            err = TILFatal(f"step limit exceeded ({self.max_steps})")
            raise err
        if self.deadlines:
            self.tick += 1
            if self.tick >= self.CHECK_INTERVAL:
                self.tick = 0
                now = _time.monotonic() * 1000.0
                for i, rf in enumerate(self.recover):
                    if rf[3] and now > rf[3]:
                        raise TILPanic(b"timeout", target=i)

    def execute(self):
        code = self.code
        consts = self.consts
        stack = self.stack
        frames = self.frames
        funcs = self.m.funcs
        frame = frames[-1]
        pc = frame[1]
        base = frame[2]
        closure = frame[3]
        op_pc = pc
        unpack_h = _struct.unpack_from
        try:
            while True:
                op_pc = pc
                op = code[pc]
                pc += 1
                if op == Op.LOAD_LOCAL:
                    stack.append(stack[base + code[pc]])
                    pc += 1
                elif op == Op.STORE_LOCAL:
                    stack[base + code[pc]] = stack.pop()
                    pc += 1
                elif op == Op.PUSH_I8:
                    v = code[pc]
                    stack.append(v - 256 if v > 127 else v)
                    pc += 1
                elif op == Op.PUSH_CONST:
                    stack.append(consts[code[pc] | (code[pc + 1] << 8)])
                    pc += 2
                elif op == Op.ADD or op == Op.SUB or op == Op.MUL:
                    b = stack.pop()
                    a = stack[-1]
                    if type(a) is int and type(b) is int:
                        r = a + b if op == Op.ADD else (a - b if op == Op.SUB else a * b)
                        if not -0x8000000000000000 <= r <= 0x7FFFFFFFFFFFFFFF:
                            r = wrap_int64(r)
                        stack[-1] = r
                    else:
                        stack[-1] = self.arith(op, a, b)
                elif op == Op.LT or op == Op.LE or op == Op.GT or op == Op.GE:
                    b = stack.pop()
                    a = stack[-1]
                    if type(a) is int and type(b) is int:
                        stack[-1] = (a < b if op == Op.LT else a <= b if op == Op.LE else
                                     a > b if op == Op.GT else a >= b)
                    else:
                        stack[-1] = self.compare(op, a, b)
                elif op == Op.JUMP:
                    rel = code[pc] | (code[pc + 1] << 8)
                    if rel >= 0x8000:
                        rel -= 0x10000
                        self.check_budget(op_pc)
                    pc += 2 + rel
                elif op == Op.JUMP_IF_FALSE or op == Op.JUMP_IF_TRUE:
                    v = stack.pop()
                    t = v is True or (v is not False and v is not None and self.truthy(v))
                    if t == (op == Op.JUMP_IF_TRUE):
                        rel = code[pc] | (code[pc + 1] << 8)
                        if rel >= 0x8000:
                            rel -= 0x10000
                            self.check_budget(op_pc)
                        pc += 2 + rel
                    else:
                        pc += 2
                elif op == Op.LOAD_GLOBAL:
                    stack.append(self.globals[code[pc] | (code[pc + 1] << 8)])
                    pc += 2
                elif op == Op.STORE_GLOBAL:
                    self.globals[code[pc] | (code[pc + 1] << 8)] = stack.pop()
                    pc += 2
                elif op == Op.EQ:
                    b = stack.pop()
                    stack[-1] = self.equal(stack[-1], b)
                elif op == Op.NE:
                    b = stack.pop()
                    stack[-1] = not self.equal(stack[-1], b)
                elif op == Op.CALL or op == Op.CALL_VALUE or op == Op.INVOKE:
                    if op == Op.CALL:
                        fidx = code[pc] | (code[pc + 1] << 8)
                        argc = code[pc + 2]
                        pc += 3
                        new_closure = None
                        extra = 0
                    elif op == Op.CALL_VALUE:
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
                    for _ in range(fn.n_params - argc):
                        stack.append(_UNDEF)
                    for _ in range(fn.n_locals - fn.n_params):
                        stack.append(None)
                    frame = [fidx, fn.offset, new_base, new_closure, extra]
                    frames.append(frame)
                    pc = fn.offset
                    base = new_base
                    closure = new_closure
                    self.check_budget(op_pc)
                elif op == Op.RETURN or op == Op.RETURN_NIL:
                    result = stack.pop() if op == Op.RETURN else None
                    depth = len(frames)
                    while self.recover and self.recover[-1][1] >= depth:
                        if self.recover.pop()[3]:
                            self.deadlines -= 1
                    frames.pop()
                    del stack[base - frame[4]:]
                    stack.append(result)
                    if not frames:
                        return
                    frame = frames[-1]
                    pc = frame[1]
                    base = frame[2]
                    closure = frame[3]
                elif op == Op.POP:
                    stack.pop()
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
                elif op == Op.DIV or op == Op.MOD or op == Op.POW:
                    b = stack.pop()
                    stack[-1] = self.arith(op, stack[-1], b)
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
                elif op == Op.LOAD_UPVAL:
                    stack.append(closure.up[code[pc]].v)
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
                elif op == Op.NATIVE:
                    nid, argc = code[pc], code[pc + 1]
                    pc += 2
                    args = stack[len(stack) - argc:] if argc else []
                    if argc:
                        del stack[len(stack) - argc:]
                    frame[1] = pc
                    stack.append(self.call_native(nid, args))
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
                elif op == Op.NEW_STRUCT:
                    t = code[pc] | (code[pc + 1] << 8)
                    n = code[pc + 2]
                    pc += 3
                    fields = stack[len(stack) - n:] if n else []
                    if n:
                        del stack[len(stack) - n:]
                    stack.append(VMStruct(t, fields))
                elif op == Op.GET_INDEX:
                    idx = stack.pop()
                    stack[-1] = self.index_get(stack[-1], idx)
                elif op == Op.SET_INDEX:
                    val = stack.pop()
                    idx = stack.pop()
                    self.index_set(stack.pop(), idx, val)
                elif op == Op.GET_FIELD:
                    stack[-1] = self.get_field(stack[-1], code[pc] | (code[pc + 1] << 8))
                    pc += 2
                elif op == Op.SET_FIELD:
                    val = stack.pop()
                    self.set_field(stack.pop(), code[pc] | (code[pc + 1] << 8), val)
                    pc += 2
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
                elif op == Op.CONCAT:
                    n = code[pc]
                    pc += 1
                    parts = stack[len(stack) - n:]
                    del stack[len(stack) - n:]
                    stack.append(b"".join(p if type(p) is bytes else self.to_str(p) for p in parts))
                elif op == Op.FOR_ITER:
                    slot = base + code[pc]
                    container = stack[slot]
                    i = stack[slot + 1]
                    if i < len(container):
                        stack.append(container[i])
                        stack[slot + 1] = i + 1
                        pc += 3
                    else:
                        rel = code[pc + 1] | (code[pc + 2] << 8)
                        pc += 3 + (rel - 0x10000 if rel >= 0x8000 else rel)
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
                    self.recover.append([handler, len(frames), len(stack), deadline])
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
    for i, (name, fields) in enumerate(m.types):
        if i > TYPE_ERR:
            out.append(f"; type #{i} {name} {{ {', '.join(fields)} }}")
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
            elif op == Op.NATIVE:
                note = NATIVES[vals[0]] + "()" if vals[0] < len(NATIVES) else "?"
            elif op in (Op.NEW_STRUCT, Op.MATCH_TYPE):
                note = m.types[vals[0]][0] if vals[0] < len(m.types) else "?"
            elif op in (Op.JUMP, Op.JUMP_IF_FALSE, Op.JUMP_IF_TRUE, Op.TRY, Op.TRY_TIMEOUT, Op.HANDLE):
                note = f"-> {p + vals[-1] - f.offset:04d}"
            elif op in (Op.FOR_ITER, Op.ARG_DEFAULT):
                note = f"-> {p + vals[-1] - f.offset:04d}"
            elif op == Op.CAST:
                note = {v: k for k, v in CAST_KINDS.items()}.get(vals[0], "?")
            elif op == Op.PERFORM:
                note = m.consts[vals[0]].decode('utf-8', 'replace') if vals[0] < len(m.consts) else "?"
            line = f"  {pc - f.offset:04d}  {name:<18}{' '.join(str(v) for v in vals):<12}"
            if note:
                line += f"; {note}"
            if pc in line_at:
                line = f"{line:<62} @ line {line_at[pc]}"
            out.append(line.rstrip())
            pc = p
    return "\n".join(out)


def compile_to_bytecode(source: str, filename: str = "<stdin>", verify: bool = True) -> TBCModule:
    """TIL source -> bytecode module (parse, resolve imports, verify, compile)."""
    program = resolve_imports(parse_source(source, filename), filename)
    if verify:
        errors = SemanticChecker(source, filename).check(program)
        if errors:
            raise CompileError("\n\n".join(errors))
    module = BytecodeCompiler(source, filename).compile_program(program)
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

        # Search: importing file's dir, base path, $TIL_PATH, then stdlib dirs
        search_dirs = []
        if from_file and not from_file.startswith("<"):
            search_dirs.append(os.path.dirname(os.path.abspath(from_file)))
        search_dirs.append(self.base_path)
        for extra in os.environ.get('TIL_PATH', '').split(os.pathsep):
            if extra:
                search_dirs.append(extra)
        for home in til_home_dirs():
            stdlib_dir = os.path.join(home, 'stdlib')
            if os.path.isdir(stdlib_dir):
                search_dirs.append(stdlib_dir)

        for d in search_dirs:
            # Try module_name.til
            candidate = os.path.join(d, parts + ".til")
            if os.path.exists(candidate):
                return candidate
            # Try module_name/mod.til (package)
            candidate = os.path.join(d, parts, "mod.til")
            if os.path.exists(candidate):
                return candidate
        return None

    def load_module(self, module_name: str, from_file: str = "") -> Optional[Program]:
        """Load and parse a module, returning its AST (cached per file)."""
        path = self.resolve(module_name, from_file)
        if not path:
            return None
        key = os.path.abspath(path)
        if key in self.loaded:
            return self.loaded[key]

        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast = parse_source(source, path)
        ast.filename = path
        self.loaded[key] = ast
        return ast


IMPORTABLE_NODES = (FuncDef, StructDef, EnumDef, ImplBlock, TraitDef, TypeAlias,
                    VarDecl, CapabilityType)

def _definition_name(node) -> Optional[str]:
    if isinstance(node, ImplBlock):
        return node.type_name
    return getattr(node, 'name', None)


def resolve_imports(program: Program, filename: str = "<stdin>") -> Program:
    """Process import statements: load modules (recursively) and merge their
    definitions - functions, structs, enums, impls, traits, consts, aliases.

    `import m` / `import m as alias` also enable qualified calls `m.f()`;
    the module names are recorded in `program.module_names`."""
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
                print(f"Warning: Could not find module '{stmt.module}'", file=sys.stderr)
                continue
            module_names.add(stmt.module.split('.')[-1])
            if stmt.alias:
                module_names.add(stmt.alias)
            mod_path = getattr(mod_ast, 'filename', stmt.module)
            if mod_path in stack:
                continue  # import cycle: definitions are already being merged
            mod_stmts = merge(mod_ast, mod_path, stack + (mod_path,))
            for mod_stmt in mod_stmts:
                if not isinstance(mod_stmt, IMPORTABLE_NODES):
                    continue
                if isinstance(mod_stmt, FuncDef) and mod_stmt.name == "main":
                    continue  # a module's own entry point / script code is not imported
                if stmt.items is not None and _definition_name(mod_stmt) not in stmt.items:
                    continue
                out.append(mod_stmt)
        return out

    merged = merge(program, filename, (os.path.abspath(filename)
                                       if not filename.startswith("<") else filename,))
    program.statements = []
    for stmt in merged:                 # a module imported several times is merged once
        if id(stmt) not in seen_defs:
            seen_defs.add(id(stmt))
            program.statements.append(stmt)
    program.module_names = module_names
    return program


class TILCompiler:
    def __init__(self):
        self.verbose = False
        self.optimize = 2
        self.keep_c = False
        self.check_types = True
        self.verify = True      # effects / capabilities / constant contracts (fatal errors)

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

        if self.check_types:
            if self.verbose:
                print(f"[TIL] Type checking...")
            
            checker = TypeChecker()
            errors = checker.check(ast)
            
            for err in errors:
                print(f"Warning: {err}", file=sys.stderr)
        
        if self.verbose:
            print(f"[TIL] Generating C code...")
        
        codegen = CCodeGenerator()
        c_code = codegen.generate(ast)
        
        return c_code
    
    def compile_to_executable(self, source: str, output: str, filename: str = "<stdin>",
                              quiet: bool = False) -> bool:
        """Compile TIL source to native executable"""
        
        c_code = self.compile(source, filename)
        
        # Find C compiler
        cc = self.find_c_compiler()
        if not cc:
            print("Error: No C compiler found (tried gcc, clang, cc)", file=sys.stderr)
            return False
        
        # Write C code to temp file (UTF-8 for string literals with Kazakh/Unicode)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False,
                                         encoding='utf-8', newline='\n') as f:
            f.write(c_code)
            c_file = f.name
        
        try:
            # Compile C to executable
            opt_flags = ["-O0", "-O1", "-O2", "-O3"][min(self.optimize, 3)]
            cmd = [cc, c_file, "-o", output, opt_flags, "-lm"]
            
            if self.verbose:
                print(f"[TIL] Compiling: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                   encoding='utf-8', errors='replace')
            
            if result.returncode != 0:
                self.last_c_error = result.stderr
                if not quiet:
                    print(f"C compilation failed:\n{result.stderr}", file=sys.stderr)
                    print("hint: this program may use features only the TIL VM supports; "
                          "try `til run --vm` or `til build --target vm`", file=sys.stderr)
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
  --max-steps <n>         VM sandbox: stop after n loop iterations/calls
  -O0, -O1, -O2, -O3      Optimization level for native builds (default -O2)
  --keep-c                Keep generated C file
  --no-check              Disable type warnings
  --no-verify             Disable effect/capability/contract verification
  -v, --verbose           Verbose output
  --                      Everything after it is passed to the program (args())

Program arguments:        til run prog.til a b c   ->  args() == ["a", "b", "c"]

Multi-Level Programming:  #[level: 0..4]  Hardware, Systems, Safe (default), Script, Formal
Languages:                English + Kazakh always; #[lang: ru] / #[lang: kk-latn] per file

Examples:
  til run hello.til                     # native (or VM if no C compiler)
  til run --vm hello.til                # portable VM
  til build --target vm game.til        # -> game.tbc (runs on any TIL VM)
  til build --target c-vm game.til      # -> game_vm.c (compile for a calculator)
  til translate --to ru hello.til       # Russian keywords
""")


# ── VM helpers ───────────────────────────────────────────────────────────────

def _cvm_binary(verbose: bool = False) -> Optional[str]:
    """Build (once) and return the host C VM executable."""
    vm_dir = find_vm_sources()
    cc = TILCompiler().find_c_compiler()
    if vm_dir is None or cc is None:
        return None
    cache = os.path.join(os.path.expanduser('~'), '.til', 'cache')
    try:
        os.makedirs(cache, exist_ok=True)
    except OSError:
        cache = tempfile.gettempdir()
    exe = os.path.join(cache, 'tilvm' + ('.exe' if sys.platform == 'win32' else ''))
    sources = [os.path.join(vm_dir, n) for n in ('tilvm.c', 'tilvm.h', 'main.c')]
    newest = max(os.path.getmtime(f) for f in sources if os.path.exists(f))
    if not os.path.exists(exe) or os.path.getmtime(exe) < newest:
        cmd = [cc, '-O2', '-std=c99', '-o', exe, sources[0], sources[2], '-lm']
        if verbose:
            print(f"[TIL] Building C VM: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"C VM build failed:\n{result.stderr}", file=sys.stderr)
            return None
    return exe


def run_bytecode(module: TBCModule, max_steps: int = 0, argv: Optional[List[str]] = None,
                 opts: Optional[Dict[str, Any]] = None) -> int:
    """Run a module on the reference VM. `argv` = program arguments (args()),
    `opts` = run options parsed by main() (see RUN_VALUE_OPTIONS)."""
    return TILVM(module, max_steps=max_steps, argv=argv).run()


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


def run_extra_command(command: str, argv: List[str]) -> int:
    module_name = EXTRA_COMMANDS[command][0]
    try:
        module = feature_module(module_name)
    except ImportError as e:
        if getattr(e, 'name', None) != module_name:
            raise
        print(f"til {command}: not available in this installation ({module_name}.py is missing)",
              file=sys.stderr)
        return 1
    return module.main(argv)


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


# ── translate ────────────────────────────────────────────────────────────────

def translate_source(source: str, target: str, filename: str = "<stdin>") -> str:
    """Rewrite TIL source with the keywords of another language (en, kk,
    kk-latn, ru). Identifiers, strings and comments are kept as they are."""
    if target not in LANGUAGES:
        raise CompileError(f"unknown language '{target}' (use: {', '.join(LANGUAGES)})")
    tokens = Lexer(source, filename).tokenize()
    kw_table = ENGLISH_KEYWORDS if target == 'en' else LANGUAGES[target][0]
    alias_table = LANGUAGES[target][1]
    word_for: Dict[TokenType, str] = {}
    for word, ttype in ENGLISH_KEYWORDS.items():
        word_for.setdefault(ttype, word)
    if target != 'en':
        for word, ttype in kw_table.items():
            if ttype not in word_for or word_for[ttype] in ENGLISH_KEYWORDS:
                word_for[ttype] = word
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
        elif text in ENGLISH_KEYWORDS or text in KEYWORDS or any(text in LANGUAGES[c][0] for c in LANGUAGES):
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


def repl_program(definitions: List[str], history: List[str], block: str) -> Tuple[str, bool]:
    """Build the program for one REPL input. Returns (source, is_expression)."""
    body = "\n".join("    " + l for l in block.split("\n"))
    probe = parse_source("main()\n" + body + "\n", "<repl>")
    stmts = next(s for s in probe.statements if isinstance(s, FuncDef) and s.name == "main").body.statements
    is_expr = (len(stmts) == 1 and isinstance(stmts[0], ExprStmt) and not (
        isinstance(stmts[0].expr, Call) and isinstance(stmts[0].expr.func, Identifier)
        and stmts[0].expr.func.name in ('print', 'println')))
    if is_expr:
        body = f"    let __v = {block.strip()}\n    if __v != None\n        print(__v)"
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
                print(e)
                continue
            definitions.append(block)
            print("ok")
            continue
        if re.match(r'^(if|for|while|loop|match|intent)\b', stripped):
            block = read_block(line)
        else:
            block = line
        try:
            source, is_expr = repl_program(definitions, history, block)
            module = compile_to_bytecode(source, "<repl>")
        except (SyntaxError, CompileError) as e:
            print(e)
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

def main():
    args = sys.argv[1:]

    if not args or '-h' in args or '--help' in args:
        print_help()
        return 0

    if '--version' in args or args[0] == 'version':
        print(f"TIL Compiler v{TIL_VERSION}")
        print("Author: Alisher Beisembekov")
        print("Backends: native (C via gcc/clang), TIL VM (bytecode: Python, C99 / calculators)")
        return 0

    if args[0] in EXTRA_COMMANDS:
        return run_extra_command(args[0], args[1:])

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
            max_steps = int(args[i + 1])
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
        elif arg in ('-v', '--verbose'):
            compiler.verbose = True
        elif arg in ('-O0', '-O1', '-O2', '-O3'):
            compiler.optimize = int(arg[2])
        elif input_file is None and not arg.startswith('-'):
            input_file = arg
        elif input_file is not None:
            prog_args.append(arg)        # after the program file: the program's arguments
        else:
            print(f"Error: unknown option {arg}", file=sys.stderr)
            return 1
        i += 1

    if not input_file:
        print("Error: No input file specified", file=sys.stderr)
        return 1
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        return 1

    try:
        # bytecode files
        if input_file.endswith('.tbc'):
            with open(input_file, 'rb') as f:
                module = TBCModule.from_bytes(f.read())
            if command == 'dis':
                print(disassemble(module))
                return 0
            return run_bytecode(module, max_steps, prog_args, opts)

        with open(input_file, 'r', encoding='utf-8') as f:
            source = f.read()

        if command == 'check':
            ast = resolve_imports(parse_source(source, input_file), input_file)
            errors = SemanticChecker(source, input_file).check(ast)
            if errors:
                for err in errors:
                    print(err, file=sys.stderr)
                return 1
            BytecodeCompiler(source, input_file).compile_program(ast)   # names, arity, immutability
            proof_errors = run_proofs(ast, source, input_file, bool(opts.get('prove')))
            if proof_errors:
                for err in proof_errors:
                    print(err, file=sys.stderr)
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
            print(disassemble(compile_to_bytecode(source, input_file, compiler.verify)))
            return 0

        base = os.path.splitext(os.path.basename(input_file))[0]

        if command == 'run' and use_vm is None and compiler.find_c_compiler() is None:
            print("[TIL] No C compiler found - running on the TIL VM", file=sys.stderr)
            use_vm = 'py'

        if command == 'run' and use_vm:
            module = compile_to_bytecode(source, input_file, compiler.verify)
            if use_vm == 'py':
                return run_bytecode(module, max_steps, prog_args, opts)
            exe = _cvm_binary(compiler.verbose)
            if exe is None:
                print("Error: cannot build the C VM (need vm/ sources and a C compiler)", file=sys.stderr)
                return 1
            fd, tbc_path = tempfile.mkstemp(suffix='.tbc')
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(module.serialize())
                cmd = [exe] + (['--max-steps', str(max_steps)] if max_steps else []) + [tbc_path] + prog_args
                return subprocess.run(cmd).returncode
            finally:
                os.unlink(tbc_path)

        if target in ('vm', 'tbc'):
            module = compile_to_bytecode(source, input_file, compiler.verify)
            data = module.serialize(strip=strip)
            out = output_file or base + '.tbc'
            with open(out, 'wb') as f:
                f.write(data)
            print(f"Built: {out} ({len(data)} bytes of TIL bytecode)")
            return 0

        if target in ('c-vm', 'calc', 'embed'):
            module = compile_to_bytecode(source, input_file, compiler.verify)
            text = amalgamate_c_vm(module.serialize(strip=strip), input_file)
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
            print(f"Error: unknown target '{target}' (native, vm, c-vm, c)", file=sys.stderr)
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
                    print("[TIL] native backend cannot compile this program; running it on the TIL VM",
                          file=sys.stderr)
                    if compiler.verbose:
                        print(getattr(compiler, 'last_c_error', ''), file=sys.stderr)
                    return run_bytecode(compile_to_bytecode(source, input_file, compiler.verify), max_steps,
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
        print(f"Syntax Error: {e}", file=sys.stderr)
        return 1
    except CompileError as e:
        print(str(e), file=sys.stderr)
        return 1
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, 'w')
        except OSError:
            pass
        return 0
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if compiler.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
