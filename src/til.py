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
            
            # String
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
                self.add_token(TokenType.STRING, ''.join(result))
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
    arms: List[Tuple[ASTNode, ASTNode]] = field(default_factory=list)  # (pattern, expr)
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
        
        return FuncDef(name=name, params=params, ret_type=ret_type, body=body, level=level, attributes=attrs)
    
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
            
            pattern = self.parse_expression()
            self.consume(TokenType.FAT_ARROW)
            
            if self.match(TokenType.INDENT):
                expr = self.parse_block()
            else:
                expr = self.parse_expression()
            
            arms.append((pattern, expr))
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
        
        return TraitDef(name, methods)
    
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
            
            return Import(module, items)
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
            
            return Import(module, alias=alias)
    
    def parse_type_alias(self) -> TypeAlias:
        self.consume(TokenType.TYPE)
        name = self.consume(TokenType.IDENT).value
        self.consume(TokenType.EQ)
        type_val = self.parse_type()
        return TypeAlias(name, type_val)
    
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
                # Tuple
                elements = [expr]
                while self.match(TokenType.COMMA):
                    self.advance()
                    if self.match(TokenType.RPAREN):
                        break
                    elements.append(self.parse_expression())
                self.consume(TokenType.RPAREN)
                return ArrayLit(elements=elements)
            
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
        
        # Lambda: |x, y| x + y
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
            body = self.parse_expression()
            return Lambda(params=params, body=body)
        
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
        self.current_level = 2
        self.in_method: bool = False
        self.current_struct: Optional[str] = None
    
    def emit(self, line: str):
        self.output.append("    " * self.indent + line)
    
    def emit_raw(self, line: str):
        self.output.append(line)
    
    def generate(self, program: Program) -> str:
        self.output = []
        
        # First pass: collect definitions
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
        self.emit_forward_declarations()
        self.emit_helpers()
        self.emit_lambdas()  # emit lambda defs collected from first pass (may be empty)
        self.emit_structs()
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
            self.emit_raw(f"typedef struct {name} {name};")

        # Forward declare functions
        for name, func in self.functions.items():
            ret = self.type_to_c(func.ret_type)
            params = self.params_to_c(func.params)
            level = func.level
            level_names = {0: "Hardware", 1: "Systems", 2: "Safe", 3: "Script", 4: "Formal"}
            self.emit_raw(f"{ret} til_{name}({params});  // Level {level}: {level_names.get(level, '?')}")

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
""")
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
        self.string_vars = set()
        self.float_vars = set()
        self.bool_vars = set()
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
        self.string_vars = set()
        self.float_vars = set()
        self.bool_vars = set()
        self.struct_vars = {}
        self.array_vars = {}
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
        
        self.emit_raw(f"// Level {func.level}: {level_names.get(func.level, 'Unknown')}")
        self.emit_raw(f"{attr_str}{ret} til_{func.name}({params}) {{")
        self.indent += 1
        
        self.generate_node(func.body)
        
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

        # Track variable types for print/method inference
        self._track_var_type(node, c_type)

        if node.name in self.declared_vars:
            # Just assign
            if node.value:
                val = self.generate_node(node.value)
                self.emit(f"{node.name} = {val};")
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
                    self.emit(f"{elem_type} {node.name}[{n}] = {val};")
                    self.emit(f"size_t {node.name}_len = {n};")
                elif isinstance(node.value, ListComprehension):
                    # Comprehension already emitted the array; just assign the pointer and len
                    self.emit(f"{c_type} {node.name} = {val};")
                    self.emit(f"size_t {node.name}_len = {val}_len;")
                else:
                    self.emit(f"{c_type} {node.name} = {val};")
            else:
                self.emit(f"{c_type} {node.name};")
        
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
        if isinstance(node.iter, Range):
            start = self.generate_node(node.iter.start)
            end = self.generate_node(node.iter.end)
            op = "<=" if node.iter.inclusive else "<"
            self.emit(f"for (int64_t {node.var} = {start}; {node.var} {op} {end}; {node.var}++) {{")
        elif isinstance(node.iter, Identifier):
            arr_name = node.iter.name
            len_var = f"{arr_name}_len"
            self.emit(f"for (size_t _i_{node.var} = 0; _i_{node.var} < {len_var}; _i_{node.var}++) {{")
            self.indent += 1
            # Get element type
            if arr_name in self.array_vars:
                self.emit(f"int64_t {node.var} = {arr_name}[_i_{node.var}];")
            else:
                self.emit(f"int64_t {node.var} = {arr_name}[_i_{node.var}];")
            self.indent -= 1
        elif isinstance(node.iter, ArrayLit):
            # Iterate over array literal: for x in [1, 2, 3]
            n = len(node.iter.elements)
            elem_type = self.infer_array_elem_type(node.iter)
            arr_name = f"_arr_{node.var}"
            arr_val = self.generate_node(node.iter)
            self.emit(f"{elem_type} {arr_name}[{n}] = {arr_val};")
            self.emit(f"for (size_t _i_{node.var} = 0; _i_{node.var} < {n}; _i_{node.var}++) {{")
            self.indent += 1
            self.emit(f"{elem_type} {node.var} = {arr_name}[_i_{node.var}];")
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
            key = name
            if key in self._func_ret_types:
                return self._func_ret_types[key]
        if isinstance(node.func, Attribute):
            method_name = node.func.attr

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
            
            # Struct constructor
            if name in self.structs:
                return f"{name}_create({args_str})"

            # Fill in default parameters if fewer args provided
            if name in self.functions:
                func_def = self.functions[name]
                if len(node.args) < len(func_def.params):
                    for i in range(len(node.args), len(func_def.params)):
                        p = func_def.params[i]
                        if p.default:
                            args.append(self.generate_node(p.default))
                    args_str = ", ".join(args)

            return f"til_{name}({args_str})"
        
        if isinstance(node.func, Attribute):
            obj = self.generate_node(node.func.obj)
            method_name = node.func.attr
            args = [self.generate_node(a) for a in node.args]

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
        # Generate a unique static function for the lambda
        if not hasattr(self, '_lambda_counter'):
            self._lambda_counter = 0
        self._lambda_counter += 1
        name = f"_til_lambda_{self._lambda_counter}"

        # Infer parameter and return types from the body
        params = []
        for pname, ptype in node.params:
            if ptype:
                c_type = self.type_to_c(ptype)
            else:
                c_type = "int64_t"  # default
            params.append(f"{c_type} {pname}")
        params_str = ", ".join(params) if params else "void"

        # Store the lambda function to be emitted at top level
        if not hasattr(self, '_lambda_defs'):
            self._lambda_defs = []

        # Generate the body expression
        body_code = self.generate_node(node.body)
        self._lambda_defs.append(f"static int64_t {name}({params_str}) {{ return {body_code}; }}")

        return name

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
        return node.name
    
    def gen_IntLit(self, node: IntLit) -> str:
        return str(node.value)
    
    def gen_FloatLit(self, node: FloatLit) -> str:
        return str(node.value)
    
    def gen_StringLit(self, node: StringLit) -> str:
        escaped = node.value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
        return f'"{escaped}"'
    
    def gen_BoolLit(self, node: BoolLit) -> str:
        return "true" if node.value else "false"
    
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
        # Generate as switch or if-else chain
        val = self.generate_node(node.value)
        
        self.emit(f"switch ({val}) {{")
        self.indent += 1
        
        for pattern, expr in node.arms:
            if isinstance(pattern, Identifier) and pattern.name == "_":
                self.emit("default:")
            else:
                pat = self.generate_node(pattern)
                self.emit(f"case {pat}:")
            
            self.indent += 1
            if isinstance(expr, Block):
                self.generate_node(expr)
            else:
                result = self.generate_node(expr)
                if result:
                    self.emit(f"{result};")
            self.emit("break;")
            self.indent -= 1
        
        self.indent -= 1
        self.emit("}")
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
            parts.append(f"{c_type} {p.name}")
        
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
        
        # Write C code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False, 
                                         encoding='ascii', errors='replace', newline='\n') as f:
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
""")

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
    
    # Check for command (run, build, check)
    command = None
    if args and args[0] in ('run', 'build', 'check'):
        command = args[0]
        args = args[1:]
    
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
                with open(output_file, 'w', encoding='ascii', errors='replace', newline='\n') as f:
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
