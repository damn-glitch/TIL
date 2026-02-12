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
        code_str = f"[{error_code}] " if error_code else ""
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
    
    # Attributes
    ATTRIBUTE = auto()

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    col: int
    
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
    'немесе': TokenType.OR,       # or (also elif, last definition wins)
    'емес': TokenType.NOT,        # not
    'ақиқат': TokenType.BOOL,     # true
    'жалған': TokenType.BOOL,     # false
}

# Merge Kazakh keywords into the main keyword map
KEYWORDS.update(KAZAKH_KEYWORDS)

class Lexer:
    def __init__(self, source: str, filename: str = "<stdin>"):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []
        self.indent_stack = [0]
        self.at_line_start = True
        self.paren_depth = 0  # Track () [] {} depth to suppress NEWLINE inside
        self.error_reporter = ErrorReporter(source, filename)

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
        self.tokens.append(Token(type, value, self.line, self.col))
        self.at_line_start = False
    
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            if self.at_line_start and self.paren_depth == 0:
                self.handle_indentation()
            elif self.at_line_start and self.paren_depth > 0:
                # Inside parens: skip whitespace but don't emit INDENT/DEDENT
                self.at_line_start = False
                while self.pos < len(self.source) and self.current() in ' \t':
                    self.advance()
                if self.pos >= len(self.source):
                    break
            
            ch = self.current()
            
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
        
        self.add_token(TokenType.ATTRIBUTE, ''.join(content).strip())
    
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
                self.advance()
                esc = self.current()
                escapes = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', 
                          '"': '"', "'": "'", '0': '\0'}
                result.append(escapes.get(esc, esc))
                self.advance()
            elif ch == '\n':
                self.error("Unterminated string")
            else:
                result.append(ch)
                self.advance()
        
        self.error("Unterminated string")
    
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
            elif ch == '{':
                if current_str:
                    parts.append(('str', ''.join(current_str)))
                    current_str = []
                self.advance()  # skip {
                expr_chars = []
                depth = 1
                while self.pos < len(self.source) and depth > 0:
                    c = self.current()
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            self.advance()
                            break
                    expr_chars.append(c)
                    self.advance()
                parts.append(('expr', ''.join(expr_chars)))
            elif ch == '\\':
                self.advance()
                esc = self.current()
                escapes = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}
                current_str.append(escapes.get(esc, esc))
                self.advance()
            elif ch == '\n':
                self.error("Unterminated f-string")
            else:
                current_str.append(ch)
                self.advance()
        self.error("Unterminated f-string")

    def read_number(self):
        start = self.pos
        is_float = False
        
        # Hex
        if self.current() == '0' and self.peek() in 'xX':
            self.advance()
            self.advance()
            while self.current() in '0123456789abcdefABCDEF_':
                self.advance()
            self.add_token(TokenType.INT, int(self.source[start:self.pos].replace('_', ''), 16))
            return
        
        # Binary
        if self.current() == '0' and self.peek() in 'bB':
            self.advance()
            self.advance()
            while self.current() in '01_':
                self.advance()
            self.add_token(TokenType.INT, int(self.source[start:self.pos].replace('_', ''), 2))
            return
        
        # Decimal
        while self.current().isdigit() or self.current() == '_':
            self.advance()
        
        if self.current() == '.' and self.peek().isdigit():
            is_float = True
            self.advance()
            while self.current().isdigit() or self.current() == '_':
                self.advance()
        
        if self.current() in 'eE':
            is_float = True
            self.advance()
            if self.current() in '+-':
                self.advance()
            while self.current().isdigit():
                self.advance()
        
        text = self.source[start:self.pos].replace('_', '')
        if is_float:
            self.add_token(TokenType.FLOAT, float(text))
        else:
            self.add_token(TokenType.INT, int(text))
    
    def read_identifier(self):
        start = self.pos
        while self.current().isalnum() or self.current() == '_':
            self.advance()
        
        text = self.source[start:self.pos]
        
        if text in KEYWORDS:
            token_type = KEYWORDS[text]
            if token_type == TokenType.BOOL:
                self.add_token(token_type, text in ('true', 'True'))
            else:
                self.add_token(token_type, text)
        else:
            self.add_token(TokenType.IDENT, text)
    
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

@dataclass
class Program(ASTNode):
    statements: List[ASTNode] = field(default_factory=list)

# ═══════════════════════════════════════════════════════════════════════════════
#                                  PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class Parser:
    def __init__(self, tokens: List[Token], filename: str = "<stdin>", source: str = ""):
        self.tokens = tokens
        self.filename = filename
        self.pos = 0
        self.current_level = 2  # Default level
        self.pending_attributes: List[str] = []
        self.error_reporter = ErrorReporter(source, filename)

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
        
        return Program(statements=statements)
    
    def parse_top_level(self) -> Optional[ASTNode]:
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
        
        # Check for -> or NEWLINE INDENT
        result = self.match(TokenType.ARROW, TokenType.NEWLINE)
        
        self.pos = pos
        return result
    
    def parse_fn(self) -> FuncDef:
        self.consume(TokenType.FN)
        name = self.consume(TokenType.IDENT).value
        return self.parse_func_body(name)
    
    def parse_func_def_simple(self) -> FuncDef:
        name = self.consume(TokenType.IDENT).value
        return self.parse_func_body(name)
    
    def parse_func_body(self, name: str) -> FuncDef:
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
            self.advance()
            expr = self.parse_expression()
            body = Block([Return(expr)])
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
        remaining_attrs = []
        for attr in attrs:
            if attr == "pure":
                effects.append("pure")
            elif attr.startswith("effects:"):
                for e in attr[len("effects:"):].strip().split(","):
                    effects.append(e.strip())
            elif attr.startswith("requires:"):
                expr_text = attr[len("requires:"):].strip()
                sub_tokens = Lexer(expr_text, "<attr>").tokenize()
                sub_parser = Parser(sub_tokens, "<attr>")
                requires_list.append(sub_parser.parse_expression())
            elif attr.startswith("ensures:"):
                expr_text = attr[len("ensures:"):].strip()
                sub_tokens = Lexer(expr_text, "<attr>").tokenize()
                sub_parser = Parser(sub_tokens, "<attr>")
                ensures_list.append(sub_parser.parse_expression())
            else:
                remaining_attrs.append(attr)

        return FuncDef(name=name, params=params, ret_type=ret_type, body=body,
                       level=level, attributes=remaining_attrs,
                       effects=effects if effects else None,
                       requires=requires_list, ensures=ensures_list)
    
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
        
        # Expression or assignment
        expr = self.parse_expression()
        
        # Check for assignment
        if self.match(TokenType.EQ, TokenType.PLUSEQ, TokenType.MINUSEQ,
                      TokenType.STAREQ, TokenType.SLASHEQ):
            op = self.advance().value
            value = self.parse_expression()
            return Assignment(target=expr, value=value, op=op)
        
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
            then_body = Block([self.parse_statement()])
        
        elifs = []
        else_body = None
        
        self.skip_newlines()
        
        while self.match(TokenType.ELIF):
            self.advance()
            elif_cond = self.parse_expression()
            self.skip_newlines()
            if self.match(TokenType.INDENT):
                elif_body = self.parse_block()
            elif self.match(TokenType.LBRACE):
                elif_body = self.parse_brace_block()
            else:
                elif_body = Block([self.parse_statement()])
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
                else_body = Block([self.parse_statement()])
        
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
            body = Block([self.parse_statement()])
        
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
            body = Block([self.parse_statement()])
        
        return While(condition=condition, body=body)
    
    def parse_loop(self) -> Loop:
        self.consume(TokenType.LOOP)
        
        self.skip_newlines()
        if self.match(TokenType.INDENT):
            body = self.parse_block()
        elif self.match(TokenType.LBRACE):
            body = self.parse_brace_block()
        else:
            body = Block([self.parse_statement()])
        
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
                method = self.parse_fn()
            elif self.match(TokenType.IDENT) and self.is_func_def():
                method = self.parse_func_def_simple()
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
    
    # Expression parsing with precedence climbing
    def parse_expression(self) -> ASTNode:
        return self.parse_ternary()
    
    def parse_ternary(self) -> ASTNode:
        expr = self.parse_or()
        
        if self.match(TokenType.IF):
            self.advance()
            condition = self.parse_or()
            self.consume(TokenType.ELSE)
            else_expr = self.parse_ternary()
            return IfExpr(condition=condition, then_expr=expr, else_expr=else_expr)
        
        return expr
    
    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        while self.match(TokenType.OR):
            self.advance()
            right = self.parse_and()
            left = BinaryOp(op="or", left=left, right=right)
        return left
    
    def parse_and(self) -> ASTNode:
        left = self.parse_not()
        while self.match(TokenType.AND):
            self.advance()
            right = self.parse_not()
            left = BinaryOp(op="and", left=left, right=right)
        return left
    
    def parse_not(self) -> ASTNode:
        if self.match(TokenType.NOT):
            self.advance()
            return UnaryOp(op="not", operand=self.parse_not())
        return self.parse_comparison()
    
    def parse_comparison(self) -> ASTNode:
        left = self.parse_bitwise_or()
        
        while self.match(TokenType.EQEQ, TokenType.NEQ, TokenType.LT, 
                        TokenType.GT, TokenType.LTE, TokenType.GTE):
            op = self.advance().value
            right = self.parse_bitwise_or()
            left = BinaryOp(op=op, left=left, right=right)
        
        return left
    
    def parse_bitwise_or(self) -> ASTNode:
        left = self.parse_bitwise_xor()
        while self.match(TokenType.PIPE):
            self.advance()
            right = self.parse_bitwise_xor()
            left = BinaryOp(op="|", left=left, right=right)
        return left
    
    def parse_bitwise_xor(self) -> ASTNode:
        left = self.parse_bitwise_and()
        while self.match(TokenType.CARET):
            self.advance()
            right = self.parse_bitwise_and()
            left = BinaryOp(op="^", left=left, right=right)
        return left
    
    def parse_bitwise_and(self) -> ASTNode:
        left = self.parse_shift()
        while self.match(TokenType.AMP):
            self.advance()
            right = self.parse_shift()
            left = BinaryOp(op="&", left=left, right=right)
        return left
    
    def parse_shift(self) -> ASTNode:
        left = self.parse_range()
        while self.match(TokenType.SHL, TokenType.SHR):
            op = self.advance().value
            right = self.parse_range()
            left = BinaryOp(op=op, left=left, right=right)
        return left
    
    def parse_range(self) -> ASTNode:
        left = self.parse_additive()
        
        if self.match(TokenType.RANGE, TokenType.RANGE_INCL):
            inclusive = self.current().type == TokenType.RANGE_INCL
            self.advance()
            right = self.parse_additive()
            return Range(start=left, end=right, inclusive=inclusive)
        
        return left
    
    def parse_additive(self) -> ASTNode:
        left = self.parse_multiplicative()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplicative()
            left = BinaryOp(op=op, left=left, right=right)
        
        return left
    
    def parse_multiplicative(self) -> ASTNode:
        left = self.parse_power()
        
        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.advance().value
            right = self.parse_power()
            left = BinaryOp(op=op, left=left, right=right)
        
        return left
    
    def parse_power(self) -> ASTNode:
        left = self.parse_unary()
        
        if self.match(TokenType.POWER):
            self.advance()
            right = self.parse_power()  # Right associative
            left = BinaryOp(op="**", left=left, right=right)
        
        return left
    
    def parse_unary(self) -> ASTNode:
        if self.match(TokenType.MINUS):
            self.advance()
            return UnaryOp(op="-", operand=self.parse_unary())
        if self.match(TokenType.TILDE):
            self.advance()
            return UnaryOp(op="~", operand=self.parse_unary())
        if self.match(TokenType.AMP):
            self.advance()
            mutable = False
            if self.match(TokenType.MUT):
                self.advance()
                mutable = True
            return UnaryOp(op="&mut" if mutable else "&", operand=self.parse_unary())
        if self.match(TokenType.STAR):
            self.advance()
            return UnaryOp(op="*", operand=self.parse_unary())
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> ASTNode:
        expr = self.parse_primary()
        
        while True:
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
                expr = Call(func=expr, args=args)
            
            elif self.match(TokenType.LBRACKET):
                # Index
                self.advance()
                index = self.parse_expression()
                self.consume(TokenType.RBRACKET)
                expr = Index(obj=expr, index=index)
            
            elif self.match(TokenType.DOT):
                # Attribute access
                self.advance()
                attr = self.consume(TokenType.IDENT).value
                expr = Attribute(obj=expr, attr=attr)
            
            elif self.match(TokenType.AS):
                # Type cast: expr as Type
                self.advance()
                target_type = self.parse_type()
                expr = Cast(expr=expr, target_type=target_type, type=target_type)

            elif self.match(TokenType.QUESTION):
                # Null check / unwrap
                self.advance()
                expr = NullCheck(expr)

            else:
                break

        return expr
    
    def parse_primary(self) -> ASTNode:
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
            for kind, text in token.value:
                if kind == 'str':
                    parts.append(('str', StringLit(value=text)))
                else:
                    # Parse the expression text
                    from til import Lexer as SubLexer, Parser as SubParser
                    sub_tokens = SubLexer(text, "<fstring>").tokenize()
                    sub_parser = SubParser(sub_tokens, "<fstring>")
                    expr = sub_parser.parse_expression()
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
            return DictLit(pairs)
        
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
#                              CODE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

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
        self._global_string_vars: Set[str] = set()
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

        # Build function return type map for print inference
        self._func_ret_types: Dict[str, str] = {}
        for name, func in self.functions.items():
            self._func_ret_types[name] = self.type_to_c(func.ret_type)
        for type_name, methods in self.impl_methods.items():
            for method in methods:
                self._func_ret_types[f"{type_name}_{method.name}"] = self.type_to_c(method.ret_type)

        # Initialize lambda storage
        self._lambda_defs = []
        self._lambda_counter = 0

        # Generate code
        self.emit_header()
        self.emit_types()
        self.emit_enums()
        self.emit_type_aliases()
        self.emit_forward_declarations()
        self.emit_helpers()
        self.emit_lambdas()
        self.emit_structs()
        self.emit_globals()
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
                    self.emit_raw(f"    {name}_{variant.name} = {val}{suffix}")
                else:
                    self.emit_raw(f"    {name}_{variant.name} = {i}{suffix}")
            self.emit_raw(f"}} {name};")
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
                    params.append(f"{type_name}* self")
                for p in method.params:
                    if p.name != "self":
                        params.append(f"{self.type_to_c(p.type)} {p.name}")
                params_str = ", ".join(params) if params else "void"
                self.emit_raw(f"static {ret} {type_name}_{method.name}({params_str});")

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
    if (a->len == 0) { fprintf(stderr, "Pop from empty array\\n"); exit(1); }
    return a->data[--a->len];
}
static int64_t til_int_array_get(TIL_IntArray* a, size_t i) {
    if (i >= a->len) { fprintf(stderr, "Array index out of bounds\\n"); exit(1); }
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
    if (!opt.has_value) { fprintf(stderr, "Unwrap failed: %s\\n", msg); exit(1); }
    return opt.value;
}
static double til_unwrap_float(TIL_Option_float opt, const char* msg) {
    if (!opt.has_value) { fprintf(stderr, "Unwrap failed: %s\\n", msg); exit(1); }
    return opt.value;
}
static const char* til_unwrap_str(TIL_Option_str opt, const char* msg) {
    if (!opt.has_value) { fprintf(stderr, "Unwrap failed: %s\\n", msg); exit(1); }
    return opt.value;
}

// Result<T, E> - tagged union for error handling
typedef struct TIL_Result_int { bool is_ok; int64_t value; const char* error; } TIL_Result_int;
typedef struct TIL_Result_float { bool is_ok; double value; const char* error; } TIL_Result_float;
typedef struct TIL_Result_str { bool is_ok; const char* value; const char* error; } TIL_Result_str;

static TIL_Result_int til_Ok_int(int64_t v) { return (TIL_Result_int){true, v, NULL}; }
static TIL_Result_float til_Ok_float(double v) { return (TIL_Result_float){true, v, NULL}; }
static TIL_Result_str til_Ok_str(const char* v) { return (TIL_Result_str){true, v, NULL}; }
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
        fprintf(stderr, "Bounds check failed: %s (index %zu, len %zu)\\n", msg, index, len);
        exit(1);
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
    if (a->len == 0) { fprintf(stderr, "Pop from empty Vec\\n"); exit(1); }
    return a->data[--a->len];
}
static int64_t til_dynarray_int_get(TIL_DynArray_int* a, size_t i) {
    if (i >= a->len) { fprintf(stderr, "Vec index out of bounds\\n"); exit(1); }
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
    if (a->len == 0) { fprintf(stderr, "Pop from empty Vec\\n"); exit(1); }
    return a->data[--a->len];
}
static double til_dynarray_float_get(TIL_DynArray_float* a, size_t i) {
    if (i >= a->len) { fprintf(stderr, "Vec index out of bounds\\n"); exit(1); }
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
    if (a->len == 0) { fprintf(stderr, "Pop from empty Vec\\n"); exit(1); }
    return a->data[--a->len];
}
static const char* til_dynarray_str_get(TIL_DynArray_str* a, size_t i) {
    if (i >= a->len) { fprintf(stderr, "Vec index out of bounds\\n"); exit(1); }
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
    fprintf(stderr, "Key not found: %s\\n", key); exit(1); return 0;
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
    fprintf(stderr, "Key not found: %s\\n", key); exit(1); return NULL;
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
            if isinstance(var.value, StringLit):
                self._global_string_vars.add(var.name)
            elif isinstance(var.value, FloatLit):
                self._global_float_vars.add(var.name)
            elif isinstance(var.value, BoolLit):
                self._global_bool_vars.add(var.name)
            if var.value:
                val = self.generate_node(var.value)
                if isinstance(var.value, ArrayLit):
                    elem_type = self.infer_array_elem_type(var.value)
                    n = len(var.value.elements)
                    self.emit_raw(f"static {elem_type} {c_name}[{n}] = {val};")
                    self.emit_raw(f"static size_t {c_name}_len = {n};")
                    self.array_vars[var.name] = n
                else:
                    self.emit_raw(f"static {c_type} {c_name} = {val};")
            else:
                self.emit_raw(f"static {c_type} {c_name};")
        self.emit_raw("")

    def emit_structs(self):
        if not self.structs:
            return
        
        self.emit_raw("// Struct Definitions")
        
        for name, struct in self.structs.items():
            self.emit_raw(f"struct {name} {{")
            for fld in struct.fields:
                c_type = self.type_to_c(fld.type)
                self.emit_raw(f"    {c_type} {fld.name};")
            self.emit_raw(f"}};")
            self.emit_raw("")

            # Constructor
            params = ", ".join(f"{self.type_to_c(f.type)} _{f.name}" for f in struct.fields)
            self.emit_raw(f"static {name} {name}_create({params}) {{")
            self.emit_raw(f"    {name} _self;")
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
            params.append(f"{type_name}* self")
        for p in method.params:
            if p.name != "self":
                params.append(f"{self.type_to_c(p.type)} {p.name}")

        params_str = ", ".join(params) if params else "void"

        self.emit_raw(f"static {ret} {type_name}_{method.name}({params_str}) {{")
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
        self.array_vars = {}
        self.dynarray_vars = {}
        self.hashmap_vars = {}
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

        for attr in func.attributes:
            if attr == "inline" and "inline" not in ' '.join(attrs):
                attrs.append("inline")
            elif attr == "noinline":
                attrs.append("__attribute__((noinline))")
            elif attr == "hot":
                attrs.append("__attribute__((hot))")
            elif attr == "cold":
                attrs.append("__attribute__((cold))")

        attr_str = ' '.join(attrs)
        if attr_str:
            attr_str += ' '

        c_name = self.mangle_name(func.name)
        self.emit_raw(f"// Level {func.level}: {level_names.get(func.level, 'Unknown')}")
        self.emit_raw(f"{attr_str}{ret} til_{c_name}({params}) {{")
        self.indent += 1

        # Emit requires contract checks at function entry
        if func.requires:
            for req_expr in func.requires:
                req_code = self.generate_node(req_expr)
                self.emit(f'if (!({req_code})) {{ fprintf(stderr, "Contract violation: requires failed\\n"); exit(1); }}')

        # Set up ensures for gen_Return to use
        old_ensures = self._current_ensures
        old_ensures_ret = self._current_ensures_ret_type
        self._current_ensures = func.ensures if func.ensures else []
        self._current_ensures_ret_type = ret

        self.generate_node(func.body)

        # Restore ensures state
        self._current_ensures = old_ensures
        self._current_ensures_ret_type = old_ensures_ret
        
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
                    self.emit(f"{elem_type} {c_name}[{n}] = {val};")
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
        # Check c_type against known struct names
        if c_type in self.structs:
            self.struct_vars[node.name] = c_type
        # Track enum types
        if c_type in self.enums:
            pass  # enums are int-like in C
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
        elif isinstance(node.iter, Identifier):
            c_arr = self.mangle_name(node.iter.name)
            len_var = f"{c_arr}_len"
            self.emit(f"for (size_t _i_{c_var} = 0; _i_{c_var} < {len_var}; _i_{c_var}++) {{")
            self.indent += 1
            self.emit(f"int64_t {c_var} = {c_arr}[_i_{c_var}];")
            self.indent -= 1
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
        self.generate_node(node.body)
        self.indent -= 1
        self.emit("}")
        return ""
    
    def gen_While(self, node: While) -> str:
        cond = self.generate_node(node.condition)
        self.emit(f"while ({cond}) {{")
        self.indent += 1
        self.generate_node(node.body)
        self.indent -= 1
        self.emit("}")
        return ""
    
    def gen_Loop(self, node: Loop) -> str:
        self.emit("while (1) {")
        self.indent += 1
        self.generate_node(node.body)
        self.indent -= 1
        self.emit("}")
        return ""
    
    def gen_Return(self, node: Return) -> str:
        if node.value:
            val = self.generate_node(node.value)
            # Check if current function has ensures contracts
            if self._current_ensures:
                ret_type = self._current_ensures_ret_type
                self.emit(f"{{ {ret_type} _result = {val};")
                old_in_ensures = self._in_ensures
                self._in_ensures = True
                for ens_expr in self._current_ensures:
                    ens_code = self.generate_node(ens_expr)
                    self.emit(f'  if (!({ens_code})) {{ fprintf(stderr, "Contract violation: ensures\\n"); exit(1); }}')
                self._in_ensures = old_in_ensures
                self.emit(f"  return _result; }}")
            else:
                self.emit(f"return {val};")
        else:
            self.emit("return;")
        return ""
    
    def gen_Break(self, node: Break) -> str:
        self.emit("break;")
        return ""
    
    def gen_Continue(self, node: Continue) -> str:
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
            # self.x in method context — check struct field type
            if isinstance(arg.obj, Identifier):
                obj_name = arg.obj.name
                stype = None
                if obj_name == "self" and self.current_struct:
                    stype = self.current_struct
                elif obj_name in self.struct_vars:
                    stype = self.struct_vars[obj_name]
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

            if name == 'min':
                return f"til_min_int({args_str})"
            if name == 'max':
                return f"til_max_int({args_str})"

            # Option/Result constructors
            if name == 'Some':
                inner_type = self._infer_expr_print_type(node.args[0]) if node.args else "int"
                type_map = {"int": "int", "float": "float", "str": "str", "bool": "bool"}
                return f"til_Some_{type_map.get(inner_type, 'int')}({args_str})"
            if name == 'Ok':
                inner_type = self._infer_expr_print_type(node.args[0]) if node.args else "int"
                type_map = {"int": "int", "float": "float", "str": "str", "bool": "bool"}
                return f"til_Ok_{type_map.get(inner_type, 'int')}({args_str})"
            if name == 'Err':
                return f'til_Err_int({args_str})'

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
                    return f"{resolved_type}_{method_name}"
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
                return f"{resolved_type}_{method_name}({args_str})"

            args_str = ", ".join(args)
            return f"{obj}.{method_name}({args_str})"
        
        func = self.generate_node(node.func)
        args = [self.generate_node(a) for a in node.args]
        return f"{func}({', '.join(args)})"
    
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
            return f"pow({left}, {right})"
        
        if node.op == '+':
            # Check for string concat — works with literals AND variables
            if self._is_string_expr(node.left) or self._is_string_expr(node.right):
                return f'til_str_concat({left}, {right})'
        
        c_op = op_map.get(node.op, node.op)
        return f"({left} {c_op} {right})"
    
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
                        ct = self.type_to_c(fld.type)
                        if ct in self.structs:
                            return ct
            return None
        if isinstance(node, Call):
            # Method call result: e.g., v1.add(v2) -> Vector2D
            ret = self._infer_call_ret_type(node)
            if ret in self.structs:
                return ret
            return None
        return None

    def _is_string_expr(self, node) -> bool:
        """Check if an expression produces a string value."""
        if isinstance(node, StringLit):
            return True
        if isinstance(node, Identifier):
            return node.name in self.string_vars
        if isinstance(node, BinaryOp) and node.op == '+':
            return self._is_string_expr(node.left) or self._is_string_expr(node.right)
        if isinstance(node, Call):
            return self._infer_call_ret_type(node) == "const char*"
        if isinstance(node, Attribute):
            if isinstance(node.obj, Identifier):
                stype = None
                if node.obj.name == "self" and self.current_struct:
                    stype = self.current_struct
                elif node.obj.name in self.struct_vars:
                    stype = self.struct_vars[node.obj.name]
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
            return f"{node.obj.name}_{node.attr}"
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
        escaped = node.value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
        return f'"{escaped}"'
    
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
        """Generate f-string as a series of til_str_concat calls."""
        if not node.parts:
            return '""'

        result = None
        for kind, part in node.parts:
            if kind == 'str':
                val = self.gen_StringLit(part)
            else:
                # Expression — convert to string based on type
                ptype = self._infer_expr_print_type(part)
                expr_val = self.generate_node(part)
                if ptype == "str":
                    val = expr_val
                elif ptype == "int":
                    if not hasattr(self, '_fstr_counter'):
                        self._fstr_counter = 0
                    self._fstr_counter += 1
                    buf = f"_fstr_buf_{self._fstr_counter}"
                    self.emit(f"char {buf}[32]; snprintf({buf}, 32, \"%lld\", (long long){expr_val});")
                    val = buf
                elif ptype == "float":
                    if not hasattr(self, '_fstr_counter'):
                        self._fstr_counter = 0
                    self._fstr_counter += 1
                    buf = f"_fstr_buf_{self._fstr_counter}"
                    self.emit(f"char {buf}[32]; snprintf({buf}, 32, \"%g\", {expr_val});")
                    val = buf
                elif ptype == "bool":
                    val = f"({expr_val} ? \"true\" : \"false\")"
                else:
                    val = expr_val

            if result is None:
                result = val
            else:
                result = f"til_str_concat({result}, {val})"

        return result

    def gen_ArrayLit(self, node: ArrayLit) -> str:
        elements = [self.generate_node(e) for e in node.elements]
        return "{" + ", ".join(elements) + "}"
    
    def gen_NullCheck(self, node: NullCheck) -> str:
        expr = self.generate_node(node.expr)
        return f"({expr} != NULL ? {expr} : (fprintf(stderr, \"Null unwrap failed\\n\"), exit(1), (void*)0))"

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

        return f"{node.name}_create({', '.join(args)})"
    
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

        if has_guards or has_wildcard or has_var_bind:
            return self._gen_match_complex(arms, val, is_expr, result_c_type)
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

    def _gen_match_complex(self, arms, val, is_expr=False, result_c_type="int64_t"):
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
        self.emit(f"int64_t _match_val = {val};")
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
                self.emit("if (!_matched) {")
                self.indent += 1
                self.emit(f"int64_t {var_name} = _match_val;")
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
                cond = f"_match_val == {pat_val}"
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
            return t.name

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
                return node.value.name  # Return struct name as C type
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
            if isinstance(node.value, BinaryOp):
                if node.value.op in ('==', '!=', '<', '>', '<=', '>=', 'and', 'or'):
                    return 'bool'
                # If either side is float, result is float
                lt = self._infer_expr_print_type(node.value.left)
                if lt == "float":
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

# ═══════════════════════════════════════════════════════════════════════════════
#                                COMPILER
# ═══════════════════════════════════════════════════════════════════════════════

class ModuleResolver:
    """Resolves and loads TIL module files."""

    def __init__(self, base_path: str = "."):
        self.base_path = base_path
        self.loaded: Dict[str, Program] = {}  # module_name -> parsed AST

    def resolve(self, module_name: str, from_file: str = "") -> Optional[str]:
        """Resolve a module name to a file path."""
        # Convert module.path to file path
        parts = module_name.replace(".", os.sep)

        # Search paths: relative to importing file, then base path, then stdlib
        search_dirs = []
        if from_file and from_file != "<stdin>":
            search_dirs.append(os.path.dirname(os.path.abspath(from_file)))
        search_dirs.append(self.base_path)
        # Add stdlib directory (relative to compiler location)
        compiler_dir = os.path.dirname(os.path.abspath(__file__))
        stdlib_dir = os.path.join(os.path.dirname(compiler_dir), 'stdlib')
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
        """Load and parse a module, returning its AST."""
        if module_name in self.loaded:
            return self.loaded[module_name]

        path = self.resolve(module_name, from_file)
        if not path:
            return None

        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()

        lexer = Lexer(source, path)
        tokens = lexer.tokenize()
        parser = Parser(tokens, path, source=source)
        ast = parser.parse()

        self.loaded[module_name] = ast
        return ast


def resolve_imports(program: Program, filename: str = "<stdin>") -> Program:
    """Process import statements: load modules and merge their definitions."""
    resolver = ModuleResolver()
    new_statements = []

    for stmt in program.statements:
        if isinstance(stmt, Import):
            mod_ast = resolver.load_module(stmt.module, filename)
            if mod_ast is None:
                print(f"Warning: Could not find module '{stmt.module}'", file=sys.stderr)
                continue

            # Collect exported definitions from the module
            for mod_stmt in mod_ast.statements:
                if isinstance(mod_stmt, (FuncDef, StructDef, EnumDef, ImplBlock)):
                    if stmt.items is None:
                        # import all
                        new_statements.append(mod_stmt)
                    elif isinstance(mod_stmt, FuncDef) and mod_stmt.name in stmt.items:
                        new_statements.append(mod_stmt)
                    elif isinstance(mod_stmt, StructDef) and mod_stmt.name in stmt.items:
                        new_statements.append(mod_stmt)
                    elif isinstance(mod_stmt, EnumDef) and mod_stmt.name in stmt.items:
                        new_statements.append(mod_stmt)
                    elif isinstance(mod_stmt, ImplBlock) and mod_stmt.type_name in stmt.items:
                        new_statements.append(mod_stmt)
        else:
            new_statements.append(stmt)

    program.statements = new_statements
    return program


class TILCompiler:
    def __init__(self):
        self.verbose = False
        self.optimize = 2
        self.keep_c = False
        self.check_types = True

    def compile(self, source: str, filename: str = "<stdin>") -> str:
        """Compile TIL source to C code"""

        if self.verbose:
            print(f"[TIL] Lexing {filename}...")

        lexer = Lexer(source, filename)
        tokens = lexer.tokenize()

        if self.verbose:
            print(f"[TIL] Parsing...")
        
        parser = Parser(tokens, filename, source=source)
        ast = parser.parse()

        # Resolve imports
        if self.verbose:
            print(f"[TIL] Resolving imports...")
        ast = resolve_imports(ast, filename)

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
    
    def compile_to_executable(self, source: str, output: str, filename: str = "<stdin>") -> bool:
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
                print(f"C compilation failed:\n{result.stderr}", file=sys.stderr)
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
    print("""
TIL v2.0 Compiler - The Intelligent Language
Author: Alisher Beisembekov

Usage: til [options] <input.til>

Options:
  -o <file>       Output file name
  -c              Output C code only (don't compile)
  -O0, -O1, -O2, -O3  Optimization level (default: -O2)
  --keep-c        Keep generated C file
  --no-check      Disable type checking
  -v, --verbose   Verbose output
  -h, --help      Show this help
  --version       Show version

Multi-Level Programming:
  Use #[level: N] attribute to set function level:
    0 = Hardware (SIMD, assembly)
    1 = Systems (like C)
    2 = Safe (like Rust) [DEFAULT]
    3 = Script (like Python)
    4 = Formal (contracts, proofs)

Examples:
  til program.til              # Compile to ./program
  til -o myapp program.til     # Compile to ./myapp
  til -c program.til           # Output C code to stdout
  til -O3 program.til          # Compile with max optimization

Commands:
  run <file>        Compile and run
  build <file>      Compile to executable
  check <file>      Syntax check only
  repl              Interactive REPL
""")

def run_repl():
    """Interactive REPL for TIL"""
    print("TIL v2.0 REPL - The Intelligent Language")
    print("Author: Alisher Beisembekov")
    print('Type expressions or statements. Use "exit" or Ctrl+D to quit.')
    print()

    compiler = TILCompiler()
    compiler.check_types = False  # Less strict for REPL
    history_funcs = []  # Accumulated function definitions
    history_vars = []   # Accumulated variable assignments

    while True:
        try:
            line = input("til> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0

        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line in ('exit', 'quit', 'exit()', 'quit()'):
            print("Bye!")
            return 0

        # Build a program from accumulated state + current input
        parts = []
        for func_def in history_funcs:
            parts.append(func_def)

        # Check if this looks like a function definition
        if ('(' in line and line.endswith(':') and not line.startswith(' ') and
                not line.startswith('if') and not line.startswith('while') and
                not line.startswith('for')):
            # Multi-line function definition
            func_lines = [line]
            while True:
                try:
                    cont = input("...  ")
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    return 0
                if cont.strip() == '':
                    break
                func_lines.append(cont)
            func_def = '\n'.join(func_lines)
            history_funcs.append(func_def)
            print("OK")
            continue

        # Build main function with accumulated vars + current line
        main_lines = ["main()"]
        for var_line in history_vars:
            main_lines.append(f"    {var_line}")

        # Determine if this is a print, assignment, or expression
        is_assignment = False
        if '=' in line and not line.startswith('print') and not '==' in line.split('=')[0]:
            # Likely an assignment like "let x = 5" or "x = 5"
            is_assignment = True
            main_lines.append(f"    {line}")
            # Also print result for let statements
            if line.startswith('let '):
                var_name = line.split('=')[0].replace('let ', '').strip()
                if ':' in var_name:
                    var_name = var_name.split(':')[0].strip()
                main_lines.append(f"    print({var_name})")
        elif line.startswith('print'):
            main_lines.append(f"    {line}")
        else:
            # Treat as expression - wrap in print
            main_lines.append(f"    print({line})")

        source = '\n'.join(parts + main_lines) + '\n'

        try:
            success = compiler.compile_to_executable(source, '/tmp/til_repl', '<repl>')
            if success:
                result = subprocess.run(['/tmp/til_repl'], capture_output=True, text=True,
                                       encoding='utf-8', errors='replace')
                if result.stdout:
                    print(result.stdout, end='')
                if result.stderr:
                    print(result.stderr, end='', file=sys.stderr)
                if result.returncode != 0 and not result.stdout and not result.stderr:
                    print(f"[exit code: {result.returncode}]")

                # If assignment succeeded, remember it
                if is_assignment:
                    history_vars.append(line)

                try:
                    os.remove('/tmp/til_repl')
                except:
                    pass
        except Exception as e:
            print(f"Error: {e}")

    return 0


def main():
    args = sys.argv[1:]

    if not args or '-h' in args or '--help' in args:
        print_help()
        return 0

    if '--version' in args:
        print("TIL Compiler v2.0.0")
        print("Author: Alisher Beisembekov")
        print("Multi-Level Programming: Mixed Martial Programming")
        return 0

    # Check for command (run, build, check, repl)
    command = None
    if args and args[0] in ('run', 'build', 'check', 'repl'):
        command = args[0]
        args = args[1:]

    if command == 'repl':
        return run_repl()
    
    compiler = TILCompiler()
    input_file = None
    output_file = None
    c_only = False
    
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg == '-o' and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
            continue
        elif arg == '-c':
            c_only = True
        elif arg == '--keep-c':
            compiler.keep_c = True
        elif arg == '--no-check':
            compiler.check_types = False
        elif arg in ('-v', '--verbose'):
            compiler.verbose = True
        elif arg == '-O0':
            compiler.optimize = 0
        elif arg == '-O1':
            compiler.optimize = 1
        elif arg == '-O2':
            compiler.optimize = 2
        elif arg == '-O3':
            compiler.optimize = 3
        elif not arg.startswith('-'):
            input_file = arg
        
        i += 1
    
    if not input_file:
        print("Error: No input file specified", file=sys.stderr)
        return 1
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        return 1
    
    with open(input_file, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        # Check command - just syntax check
        if command == 'check':
            lexer = Lexer(source, input_file)
            tokens = lexer.tokenize()
            parser = Parser(tokens, input_file, source=source)
            parser.parse()
            print(f"OK: {input_file}")
            return 0
        
        if c_only:
            c_code = compiler.compile(source, input_file)
            if output_file:
                with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(c_code)
                print(f"Generated: {output_file}")
            else:
                print(c_code)
        else:
            if not output_file:
                output_file = os.path.splitext(os.path.basename(input_file))[0]
            
            # Add .exe on Windows
            exe_file = output_file
            if sys.platform == 'win32' and not exe_file.endswith('.exe'):
                exe_file = output_file + '.exe'
            
            success = compiler.compile_to_executable(source, output_file, input_file)
            if not success:
                return 1
            
            # Run command - compile and execute
            if command == 'run':
                # Determine executable path
                if sys.platform == 'win32':
                    exe_path = exe_file if os.path.exists(exe_file) else output_file + '.exe'
                else:
                    exe_path = './' + output_file if not output_file.startswith('./') else output_file
                
                # Run the executable
                try:
                    result = subprocess.run([exe_path], capture_output=False)
                    
                    # Clean up executable after running
                    try:
                        os.remove(exe_path)
                    except:
                        pass
                    
                    return result.returncode
                except Exception as e:
                    print(f"Error running {exe_path}: {e}", file=sys.stderr)
                    return 1
            else:
                # Build command - just compile
                print(f"Compiled: {output_file}")
    
    except SyntaxError as e:
        print(f"Syntax Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if compiler.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
