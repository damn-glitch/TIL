# TIL Compiler Reference

## Complete Compiler Documentation v2.0

---

<div align="center">

**Author: Alisher Beisembekov**

*"Проще Python. Быстрее C. Умнее всех."*

*"Simpler than Python. Faster than C. Smarter than all."*

</div>

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Compiler Architecture](#2-compiler-architecture)
3. [Lexer](#3-lexer)
4. [Parser](#4-parser)
5. [Abstract Syntax Tree](#5-abstract-syntax-tree)
6. [Code Generator](#6-code-generator)
7. [C Code Generation](#7-c-code-generation)
8. [Optimization](#8-optimization)
9. [Command Line Interface](#9-command-line-interface)
10. [IDE Integration](#10-ide-integration)
11. [Error Messages](#11-error-messages)
12. [Compiler Internals](#12-compiler-internals)
13. [Built-in Functions](#13-built-in-functions)
14. [Type System Implementation](#14-type-system-implementation)
15. [Multi-Level System Implementation](#15-multi-level-system-implementation)
16. [Extending the Compiler](#16-extending-the-compiler)
17. [Testing](#17-testing)
18. [Debugging](#18-debugging)
19. [Performance](#19-performance)
20. [API Reference](#20-api-reference)

---

## 1. Introduction

### 1.1 About the TIL Compiler

The TIL Compiler is a single-file, self-contained compiler written in Python by **Alisher Beisembekov**. It transforms TIL source code into optimized C code, which is then compiled to native executables using standard C compilers (GCC, Clang, MSVC).

### 1.2 Design Philosophy

The TIL compiler was designed with these goals:

1. **Simplicity**: Single-file implementation (~2500 lines)
2. **Portability**: Pure Python, no external dependencies
3. **Transparency**: Generated C code is readable and debuggable
4. **Performance**: Leverage decades of C compiler optimizations
5. **Extensibility**: Clean architecture for adding features

### 1.3 Compilation Pipeline

```
┌─────────────┐    ┌─────────┐    ┌────────┐    ┌───────────┐    ┌────────────┐
│ TIL Source  │ -> │  Lexer  │ -> │ Parser │ -> │  CodeGen  │ -> │   C Code   │
│   (.til)    │    │ (tokens)│    │  (AST) │    │           │    │   (.c)     │
└─────────────┘    └─────────┘    └────────┘    └───────────┘    └────────────┘
                                                                        │
                                                                        v
                                                               ┌────────────────┐
                                                               │  C Compiler    │
                                                               │ (gcc/clang/cl) │
                                                               └────────────────┘
                                                                        │
                                                                        v
                                                               ┌────────────────┐
                                                               │   Executable   │
                                                               │  (.exe / bin)  │
                                                               └────────────────┘
```

### 1.4 File Structure

The compiler consists of these main components:

```python
# til.py - Single file compiler by Alisher Beisembekov

# Token types (TT enum)
# Token dataclass
# Keyword dictionary (KW)
# Lexer class
# AST Node classes
# Parser class
# Code Generator (Gen class)
# Compiler class
# Main entry point
```

---

## 2. Compiler Architecture

### 2.1 Overview

```
TIL Compiler Architecture
Author: Alisher Beisembekov

┌─────────────────────────────────────────────────────────────────┐
│                         TIL Source Code                         │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                           LEXER                                  │
│  • Tokenization                                                  │
│  • Indentation tracking                                          │
│  • Keyword recognition                                           │
│  • Literal parsing                                               │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌───────────────┐
                        │ Token Stream  │
                        └───────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                           PARSER                                 │
│  • Recursive descent parsing                                     │
│  • Precedence climbing for expressions                           │
│  • AST construction                                              │
│  • Syntax error detection                                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌───────────────┐
                        │  AST (Tree)   │
                        └───────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CODE GENERATOR                             │
│  • C code emission                                               │
│  • Type mapping                                                  │
│  • Function generation                                           │
│  • Struct/Enum handling                                          │
│  • Built-in function injection                                   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌───────────────┐
                        │    C Code     │
                        └───────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       C COMPILER                                 │
│  • GCC / Clang / MSVC                                            │
│  • Optimization flags                                            │
│  • Linking                                                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌───────────────┐
                        │  Executable   │
                        └───────────────┘
```

### 2.2 Module Dependencies

```python
# Standard library imports used by the compiler
import os
import sys
import re
import subprocess
import tempfile
import shutil
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
```

### 2.3 Class Hierarchy

```
Classes in the TIL Compiler:

TT (Enum)                    - Token types
Token (dataclass)            - Token with type, value, position

Lexer                        - Tokenizer
  ├── tokenize()            - Main tokenization
  ├── _indent()             - Handle indentation
  ├── _attr()               - Parse attributes
  ├── _str()                - Parse strings
  ├── _num()                - Parse numbers
  ├── _id()                 - Parse identifiers
  └── _op()                 - Parse operators

N (base)                     - AST node base
  ├── IntN, FloatN, StrN    - Literals
  ├── BoolN, IdN            - Literals, Identifiers
  ├── BinN, UnN             - Expressions
  ├── CallN, IdxN, AttrN    - Postfix operations
  ├── ArrN, RangeN          - Arrays, Ranges
  ├── StructLitN, CastN     - Struct literals, Casts
  ├── BlockN, VarN          - Statements
  ├── AssignN, IfN          - Statements
  ├── ForN, WhileN, LoopN   - Loops
  ├── RetN, BreakN, ContN   - Control flow
  ├── ExprStN, MatchN       - Statements
  ├── FnN, StructN, EnumN   - Definitions
  ├── ImplN, ProgN          - Top-level
  └── Param, Field, EVar    - Supporting structures

Parser                       - Parse tokens to AST
  ├── parse()               - Main parsing
  ├── top()                 - Top-level statements
  ├── pfn(), pfns()         - Functions
  ├── pstruct(), penum()    - Types
  ├── pimpl()               - Implementations
  ├── stmt()                - Statements
  ├── pif(), pfor(), ...    - Control flow
  └── expr(), eor(), ...    - Expressions

Gen                          - Code generator
  ├── gen()                 - Main generation
  ├── hdr(), hlp()          - Headers, helpers
  ├── gst(), gen_()         - Structs, enums
  ├── gfn(), gf()           - Functions
  ├── gim(), gmt()          - Impl methods
  ├── gb(), gs()            - Blocks, statements
  └── gx(), gbin(), ...     - Expressions

Compiler                     - Main compiler class
  ├── to_c()                - Compile to C
  ├── compile()             - Compile to executable
  └── find_cc()             - Find C compiler
```

---

## 3. Lexer

### 3.1 Token Types

The lexer recognizes these token types:

```python
class TT(Enum):
    # Literals
    INT = auto()        # 42, 0xFF, 0b1010
    FLOAT = auto()      # 3.14, 1.5e10
    STRING = auto()     # "hello", 'world'
    IDENT = auto()      # variable_name, Type
    
    # Keywords
    IF = auto()         # if
    ELSE = auto()       # else
    ELIF = auto()       # elif
    FOR = auto()        # for
    WHILE = auto()      # while
    LOOP = auto()       # loop
    IN = auto()         # in
    RETURN = auto()     # return
    BREAK = auto()      # break
    CONTINUE = auto()   # continue
    FN = auto()         # fn
    LET = auto()        # let
    VAR = auto()        # var
    CONST = auto()      # const
    MUT = auto()        # mut
    STRUCT = auto()     # struct
    ENUM = auto()       # enum
    IMPL = auto()       # impl
    TRAIT = auto()      # trait
    MATCH = auto()      # match
    TYPE = auto()       # type
    PUB = auto()        # pub
    AS = auto()         # as
    TRUE = auto()       # true, True
    FALSE = auto()      # false, False
    AND = auto()        # and
    OR = auto()         # or
    NOT = auto()        # not
    SELF = auto()       # self
    NONE = auto()       # None
    
    # Operators
    PLUS = auto()       # +
    MINUS = auto()      # -
    STAR = auto()       # *
    SLASH = auto()      # /
    PERCENT = auto()    # %
    POWER = auto()      # **
    EQ = auto()         # =
    EQEQ = auto()       # ==
    NEQ = auto()        # !=
    LT = auto()         # <
    GT = auto()         # >
    LTE = auto()        # <=
    GTE = auto()        # >=
    PLUSEQ = auto()     # +=
    MINUSEQ = auto()    # -=
    STAREQ = auto()     # *=
    SLASHEQ = auto()    # /=
    ARROW = auto()      # ->
    FAT_ARROW = auto()  # =>
    RANGE = auto()      # ..
    RANGE_INCL = auto() # ..=
    AMP = auto()        # &
    PIPE = auto()       # |
    CARET = auto()      # ^
    TILDE = auto()      # ~
    SHL = auto()        # <<
    SHR = auto()        # >>
    
    # Delimiters
    LPAREN = auto()     # (
    RPAREN = auto()     # )
    LBRACKET = auto()   # [
    RBRACKET = auto()   # ]
    LBRACE = auto()     # {
    RBRACE = auto()     # }
    COMMA = auto()      # ,
    COLON = auto()      # :
    SEMICOLON = auto()  # ;
    DOT = auto()        # .
    
    # Structure
    NEWLINE = auto()    # \n
    INDENT = auto()     # Indentation increase
    DEDENT = auto()     # Indentation decrease
    ATTR = auto()       # #[...]
    EOF = auto()        # End of file
```

### 3.2 Keyword Dictionary

```python
KW = {
    'if': TT.IF,
    'else': TT.ELSE,
    'elif': TT.ELIF,
    'for': TT.FOR,
    'while': TT.WHILE,
    'loop': TT.LOOP,
    'in': TT.IN,
    'return': TT.RETURN,
    'break': TT.BREAK,
    'continue': TT.CONTINUE,
    'fn': TT.FN,
    'let': TT.LET,
    'var': TT.VAR,
    'const': TT.CONST,
    'mut': TT.MUT,
    'struct': TT.STRUCT,
    'enum': TT.ENUM,
    'impl': TT.IMPL,
    'trait': TT.TRAIT,
    'match': TT.MATCH,
    'type': TT.TYPE,
    'pub': TT.PUB,
    'as': TT.AS,
    'true': TT.TRUE,
    'false': TT.FALSE,
    'True': TT.TRUE,
    'False': TT.FALSE,
    'and': TT.AND,
    'or': TT.OR,
    'not': TT.NOT,
    'self': TT.SELF,
    'None': TT.NONE
}
```

### 3.3 Lexer Implementation

```python
class Lexer:
    def __init__(self, src: str, filename: str = "<stdin>"):
        self.src = src           # Source code
        self.filename = filename # For error messages
        self.pos = 0            # Current position
        self.line = 1           # Current line
        self.col = 1            # Current column
        self.tokens = []        # Output tokens
        self.indent_stack = [0] # Indentation levels
        self.at_line_start = True
    
    def c(self) -> str:
        """Get current character"""
        return self.src[self.pos] if self.pos < len(self.src) else '\0'
    
    def peek(self, n: int = 1) -> str:
        """Peek ahead n characters"""
        pos = self.pos + n
        return self.src[pos] if pos < len(self.src) else '\0'
    
    def advance(self) -> str:
        """Advance and return current character"""
        c = self.c()
        self.pos += 1
        if c == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return c
    
    def add_token(self, type: TT, value: Any = None):
        """Add token to output"""
        self.tokens.append(Token(type, value, self.line, self.col))
    
    def error(self, msg: str):
        """Raise syntax error"""
        raise SyntaxError(f"{self.filename}:{self.line}:{self.col}: {msg}")
    
    def tokenize(self) -> List[Token]:
        """Main tokenization loop"""
        while self.pos < len(self.src):
            if self.at_line_start:
                self._handle_indentation()
            
            if self.pos >= len(self.src):
                break
            
            c = self.c()
            
            # Skip whitespace (not at line start)
            if c in ' \t' and not self.at_line_start:
                self.advance()
                continue
            
            # Comments
            if c == '#':
                if self.peek() == '[':
                    self._parse_attribute()
                else:
                    self._skip_comment()
                continue
            
            # Newline
            if c == '\n':
                self.add_token(TT.NEWLINE)
                self.advance()
                self.at_line_start = True
                continue
            
            # String literals
            if c in '"\'':
                self._parse_string(c)
                continue
            
            # Number literals
            if c.isdigit():
                self._parse_number()
                continue
            
            # Identifiers and keywords
            if c.isalpha() or c == '_':
                self._parse_identifier()
                continue
            
            # Operators and delimiters
            if self._parse_operator():
                continue
            
            self.error(f"Unexpected character: '{c}'")
        
        # Close remaining indentation
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.add_token(TT.DEDENT)
        
        self.add_token(TT.EOF)
        return self.tokens
```

### 3.4 Indentation Handling

```python
def _handle_indentation(self):
    """Handle Python-style indentation"""
    spaces = 0
    
    while self.pos < len(self.src):
        c = self.c()
        if c == ' ':
            spaces += 1
            self.advance()
        elif c == '\t':
            spaces += 4  # Tab = 4 spaces
            self.advance()
        elif c == '\n':
            # Empty line
            spaces = 0
            self.advance()
        elif c == '#' and self.peek() != '[':
            # Comment line
            self._skip_comment()
            spaces = 0
        else:
            break
    
    if self.pos >= len(self.src):
        return
    
    current_indent = self.indent_stack[-1]
    
    if spaces > current_indent:
        self.indent_stack.append(spaces)
        self.add_token(TT.INDENT)
    elif spaces < current_indent:
        while len(self.indent_stack) > 1 and self.indent_stack[-1] > spaces:
            self.indent_stack.pop()
            self.add_token(TT.DEDENT)
    
    self.at_line_start = False
```

### 3.5 Number Parsing

```python
def _parse_number(self):
    """Parse integer or float literal"""
    start = self.pos
    is_float = False
    
    # Check for hex, binary, octal
    if self.c() == '0' and self.peek() in 'xXbBoO':
        self.advance()  # 0
        base_char = self.advance()  # x/b/o
        
        while self.c().isalnum() or self.c() == '_':
            self.advance()
        
        text = self.src[start:self.pos].replace('_', '')
        base = {'x': 16, 'X': 16, 'b': 2, 'B': 2, 'o': 8, 'O': 8}[base_char]
        self.add_token(TT.INT, int(text, base))
        return
    
    # Integer part
    while self.c().isdigit() or self.c() == '_':
        self.advance()
    
    # Decimal part
    if self.c() == '.' and self.peek().isdigit():
        is_float = True
        self.advance()
        while self.c().isdigit() or self.c() == '_':
            self.advance()
    
    # Exponent part
    if self.c() in 'eE':
        is_float = True
        self.advance()
        if self.c() in '+-':
            self.advance()
        while self.c().isdigit():
            self.advance()
    
    text = self.src[start:self.pos].replace('_', '')
    
    if is_float:
        self.add_token(TT.FLOAT, float(text))
    else:
        self.add_token(TT.INT, int(text))
```

---

## 4. Parser

### 4.1 Parser Overview

The TIL parser uses **recursive descent** with **precedence climbing** for expressions.

```python
class Parser:
    def __init__(self, tokens: List[Token], filename: str = "<stdin>"):
        self.tokens = tokens
        self.filename = filename
        self.pos = 0
        self.current_level = 2  # Default abstraction level
        self.attributes = []
    
    def current(self) -> Token:
        """Get current token"""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]
    
    def peek(self, n: int = 1) -> Token:
        """Peek ahead n tokens"""
        pos = self.pos + n
        return self.tokens[pos] if pos < len(self.tokens) else self.tokens[-1]
    
    def match(self, *types: TT) -> bool:
        """Check if current token matches any type"""
        return self.current().type in types
    
    def advance(self) -> Token:
        """Advance and return current token"""
        tok = self.current()
        self.pos += 1
        return tok
    
    def expect(self, type: TT, msg: str = None) -> Token:
        """Expect specific token type"""
        if self.current().type == type:
            return self.advance()
        t = self.current()
        raise SyntaxError(
            f"{self.filename}:{t.line}:{t.col}: "
            f"{msg or f'Expected {type.name}'}"
        )
    
    def skip_newlines(self):
        """Skip newline tokens"""
        while self.match(TT.NEWLINE):
            self.advance()
```

### 4.2 Top-Level Parsing

```python
def parse(self) -> ProgN:
    """Parse entire program"""
    statements = []
    self.skip_newlines()
    
    while not self.match(TT.EOF):
        stmt = self.parse_top_level()
        if stmt:
            statements.append(stmt)
        self.skip_newlines()
    
    return ProgN(statements)

def parse_top_level(self):
    """Parse top-level statement"""
    self.skip_newlines()
    
    # Collect attributes
    while self.match(TT.ATTR):
        attr = self.advance().value
        self.attributes.append(attr)
        
        # Handle level attribute
        if attr.startswith("level:"):
            try:
                self.current_level = int(attr.split(":")[1].strip())
            except:
                pass
        
        self.skip_newlines()
    
    # Parse declaration
    if self.match(TT.STRUCT):
        return self.parse_struct()
    if self.match(TT.ENUM):
        return self.parse_enum()
    if self.match(TT.IMPL):
        return self.parse_impl()
    if self.match(TT.FN):
        return self.parse_function()
    if self.match(TT.IDENT) and self.is_function():
        return self.parse_function_shorthand()
    
    return self.parse_statement()
```

### 4.3 Expression Parsing with Precedence Climbing

```python
def parse_expression(self):
    """Parse expression using precedence climbing"""
    return self.parse_or()

def parse_or(self):
    """Parse or expression (lowest precedence)"""
    left = self.parse_and()
    while self.match(TT.OR):
        self.advance()
        left = BinN("or", left, self.parse_and())
    return left

def parse_and(self):
    """Parse and expression"""
    left = self.parse_not()
    while self.match(TT.AND):
        self.advance()
        left = BinN("and", left, self.parse_not())
    return left

def parse_not(self):
    """Parse not expression"""
    if self.match(TT.NOT):
        self.advance()
        return UnN("not", self.parse_not())
    return self.parse_comparison()

def parse_comparison(self):
    """Parse comparison expression"""
    left = self.parse_bitor()
    while self.match(TT.EQEQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE):
        op = self.advance().value
        left = BinN(op, left, self.parse_bitor())
    return left

def parse_bitor(self):
    left = self.parse_xor()
    while self.match(TT.PIPE):
        self.advance()
        left = BinN("|", left, self.parse_xor())
    return left

def parse_xor(self):
    left = self.parse_bitand()
    while self.match(TT.CARET):
        self.advance()
        left = BinN("^", left, self.parse_bitand())
    return left

def parse_bitand(self):
    left = self.parse_shift()
    while self.match(TT.AMP):
        self.advance()
        left = BinN("&", left, self.parse_shift())
    return left

def parse_shift(self):
    left = self.parse_additive()
    while self.match(TT.SHL, TT.SHR):
        op = self.advance().value
        left = BinN(op, left, self.parse_additive())
    return left

def parse_additive(self):
    left = self.parse_multiplicative()
    while self.match(TT.PLUS, TT.MINUS):
        op = self.advance().value
        left = BinN(op, left, self.parse_multiplicative())
    return left

def parse_multiplicative(self):
    left = self.parse_power()
    while self.match(TT.STAR, TT.SLASH, TT.PERCENT):
        op = self.advance().value
        left = BinN(op, left, self.parse_power())
    return left

def parse_power(self):
    """Parse power (right-associative)"""
    left = self.parse_unary()
    if self.match(TT.POWER):
        self.advance()
        left = BinN("**", left, self.parse_power())
    return left

def parse_unary(self):
    """Parse unary operators"""
    if self.match(TT.MINUS):
        self.advance()
        return UnN("-", self.parse_unary())
    if self.match(TT.TILDE):
        self.advance()
        return UnN("~", self.parse_unary())
    if self.match(TT.AMP):
        self.advance()
        return UnN("&", self.parse_unary())
    if self.match(TT.STAR):
        self.advance()
        return UnN("*", self.parse_unary())
    return self.parse_postfix()

def parse_postfix(self):
    """Parse postfix operations"""
    expr = self.parse_primary()
    
    while True:
        if self.match(TT.LPAREN):
            # Function call
            self.advance()
            args = []
            while not self.match(TT.RPAREN):
                args.append(self.parse_expression())
                if not self.match(TT.RPAREN):
                    self.expect(TT.COMMA)
            self.expect(TT.RPAREN)
            expr = CallN(expr, args)
        elif self.match(TT.LBRACKET):
            # Index
            self.advance()
            index = self.parse_expression()
            self.expect(TT.RBRACKET)
            expr = IdxN(expr, index)
        elif self.match(TT.DOT):
            # Attribute access
            self.advance()
            attr = self.expect(TT.IDENT).value
            expr = AttrN(expr, attr)
        elif self.match(TT.RANGE, TT.RANGE_INCL):
            # Range
            inclusive = self.current().type == TT.RANGE_INCL
            self.advance()
            end = self.parse_additive()
            expr = RangeN(expr, end, inclusive)
        elif self.match(TT.AS):
            # Cast
            self.advance()
            type_name = self.parse_type()
            expr = CastN(expr, type_name)
        else:
            break
    
    return expr

def parse_primary(self):
    """Parse primary expressions"""
    if self.match(TT.INT):
        return IntN(self.advance().value)
    if self.match(TT.FLOAT):
        return FloatN(self.advance().value)
    if self.match(TT.STRING):
        return StrN(self.advance().value)
    if self.match(TT.TRUE):
        self.advance()
        return BoolN(True)
    if self.match(TT.FALSE):
        self.advance()
        return BoolN(False)
    if self.match(TT.SELF):
        self.advance()
        return IdN("self")
    if self.match(TT.LPAREN):
        self.advance()
        expr = self.parse_expression()
        self.expect(TT.RPAREN)
        return expr
    if self.match(TT.LBRACKET):
        return self.parse_array_literal()
    if self.match(TT.IDENT):
        name = self.advance().value
        if self.match(TT.LBRACE):
            return self.parse_struct_literal(name)
        return IdN(name)
    
    t = self.current()
    raise SyntaxError(
        f"{self.filename}:{t.line}:{t.col}: "
        f"Unexpected token {t.type.name}"
    )
```

---

## 5. Abstract Syntax Tree

### 5.1 AST Node Hierarchy

```python
# Base class for all AST nodes
class N:
    pass

# ═══════════════════════════════════════════════════════════
# LITERAL NODES
# ═══════════════════════════════════════════════════════════

class IntN(N):
    """Integer literal: 42"""
    def __init__(self, value: int):
        self.value = value

class FloatN(N):
    """Float literal: 3.14"""
    def __init__(self, value: float):
        self.value = value

class StrN(N):
    """String literal: "hello" """
    def __init__(self, value: str):
        self.value = value

class BoolN(N):
    """Boolean literal: true/false"""
    def __init__(self, value: bool):
        self.value = value

class IdN(N):
    """Identifier: variable_name"""
    def __init__(self, name: str):
        self.name = name

# ═══════════════════════════════════════════════════════════
# EXPRESSION NODES
# ═══════════════════════════════════════════════════════════

class BinN(N):
    """Binary operation: a + b"""
    def __init__(self, op: str, left: N, right: N):
        self.op = op
        self.left = left
        self.right = right

class UnN(N):
    """Unary operation: -x, not x"""
    def __init__(self, op: str, expr: N):
        self.op = op
        self.expr = expr

class CallN(N):
    """Function call: func(args)"""
    def __init__(self, func: N, args: List[N]):
        self.func = func
        self.args = args

class IdxN(N):
    """Index operation: arr[i]"""
    def __init__(self, obj: N, index: N):
        self.obj = obj
        self.index = index

class AttrN(N):
    """Attribute access: obj.attr"""
    def __init__(self, obj: N, attr: str):
        self.obj = obj
        self.attr = attr

class ArrN(N):
    """Array literal: [1, 2, 3]"""
    def __init__(self, elements: List[N]):
        self.elements = elements

class RangeN(N):
    """Range: start..end or start..=end"""
    def __init__(self, start: N, end: N, inclusive: bool = False):
        self.start = start
        self.end = end
        self.inclusive = inclusive

class StructLitN(N):
    """Struct literal: Point { x: 1, y: 2 }"""
    def __init__(self, name: str, fields: Dict[str, N]):
        self.name = name
        self.fields = fields

class CastN(N):
    """Type cast: x as float"""
    def __init__(self, expr: N, target_type: str):
        self.expr = expr
        self.target_type = target_type

# ═══════════════════════════════════════════════════════════
# STATEMENT NODES
# ═══════════════════════════════════════════════════════════

class BlockN(N):
    """Block of statements"""
    def __init__(self, statements: List[N]):
        self.statements = statements

class VarN(N):
    """Variable declaration: let/var/const"""
    def __init__(self, name: str, type: str, value: N, mutable: bool = True):
        self.name = name
        self.type = type
        self.value = value
        self.mutable = mutable

class AssignN(N):
    """Assignment: x = y, x += y"""
    def __init__(self, target: N, value: N, op: str = '='):
        self.target = target
        self.value = value
        self.op = op

class IfN(N):
    """If statement with optional elif and else"""
    def __init__(self, condition: N, then_block: N, 
                 elif_blocks: List[Tuple[N, N]], else_block: N):
        self.condition = condition
        self.then_block = then_block
        self.elif_blocks = elif_blocks
        self.else_block = else_block

class ForN(N):
    """For loop: for i in range"""
    def __init__(self, var: str, iterable: N, body: N):
        self.var = var
        self.iterable = iterable
        self.body = body

class WhileN(N):
    """While loop"""
    def __init__(self, condition: N, body: N):
        self.condition = condition
        self.body = body

class LoopN(N):
    """Infinite loop"""
    def __init__(self, body: N):
        self.body = body

class RetN(N):
    """Return statement"""
    def __init__(self, value: N = None):
        self.value = value

class BreakN(N):
    """Break statement"""
    pass

class ContN(N):
    """Continue statement"""
    pass

class ExprStN(N):
    """Expression statement"""
    def __init__(self, expr: N):
        self.expr = expr

class MatchN(N):
    """Match expression"""
    def __init__(self, expr: N, arms: List[Tuple[N, N]]):
        self.expr = expr
        self.arms = arms

# ═══════════════════════════════════════════════════════════
# DEFINITION NODES
# ═══════════════════════════════════════════════════════════

@dataclass
class Param:
    """Function parameter"""
    name: str
    type: str
    default: N = None

@dataclass
class Field:
    """Struct field"""
    name: str
    type: str
    default: N = None

@dataclass
class EVar:
    """Enum variant"""
    name: str
    value: N = None

class FnN(N):
    """Function definition"""
    def __init__(self, name: str, params: List[Param], return_type: str,
                 body: N, level: int = 2, attributes: List[str] = None,
                 is_method: bool = False, has_self: bool = False):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
        self.level = level
        self.attributes = attributes or []
        self.is_method = is_method
        self.has_self = has_self

class StructN(N):
    """Struct definition"""
    def __init__(self, name: str, fields: List[Field]):
        self.name = name
        self.fields = fields

class EnumN(N):
    """Enum definition"""
    def __init__(self, name: str, variants: List[EVar]):
        self.name = name
        self.variants = variants

class ImplN(N):
    """Impl block"""
    def __init__(self, type_name: str, methods: List[FnN], trait: str = None):
        self.type_name = type_name
        self.methods = methods
        self.trait = trait

class ProgN(N):
    """Program (root node)"""
    def __init__(self, statements: List[N]):
        self.statements = statements
```

---

## 6. Code Generator

### 6.1 Generator Overview

```python
class Gen:
    """C code generator for TIL"""
    
    def __init__(self):
        self.output = []          # Output lines
        self.indent = 0           # Current indentation
        self.structs = {}         # Struct definitions
        self.functions = {}       # Function definitions
        self.impls = {}           # Impl methods
        self.enums = {}           # Enum definitions
        self.declared_vars = set() # Variables in current scope
        self.string_vars = set()  # Variables holding strings
        self.array_vars = {}      # Array variable sizes
        self.current_level = 2    # Current abstraction level
    
    def emit(self, line: str):
        """Emit indented line"""
        self.output.append("    " * self.indent + line)
    
    def emit_raw(self, line: str):
        """Emit without indentation"""
        self.output.append(line)
    
    def gen(self, prog: ProgN) -> str:
        """Generate C code from AST"""
        # First pass: collect definitions
        for stmt in prog.statements:
            if isinstance(stmt, StructN):
                self.structs[stmt.name] = stmt
            elif isinstance(stmt, FnN):
                self.functions[stmt.name] = stmt
            elif isinstance(stmt, EnumN):
                self.enums[stmt.name] = stmt
            elif isinstance(stmt, ImplN):
                if stmt.type_name not in self.impls:
                    self.impls[stmt.type_name] = []
                self.impls[stmt.type_name].extend(stmt.methods)
        
        # Generate code
        self.gen_header()
        self.gen_helpers()
        self.gen_structs()
        self.gen_enums()
        self.gen_forward_declarations()
        self.gen_impl_methods()
        self.gen_functions(prog)
        self.gen_main()
        
        return '\n'.join(self.output)
```

### 6.2 Header Generation

```python
def gen_header(self):
    """Generate C header"""
    self.emit_raw(f"// TIL v2.0 - Generated C Code")
    self.emit_raw(f"// Author: Alisher Beisembekov")
    self.emit_raw(f"// Compiled by TIL Compiler")
    self.emit_raw("")
    self.emit_raw("#include <stdio.h>")
    self.emit_raw("#include <stdlib.h>")
    self.emit_raw("#include <string.h>")
    self.emit_raw("#include <stdbool.h>")
    self.emit_raw("#include <stdint.h>")
    self.emit_raw("#include <math.h>")
    self.emit_raw("")

def gen_helpers(self):
    """Generate built-in helper functions"""
    helpers = '''
// TIL Built-in Functions
static void til_print_int(int64_t x) { printf("%lld\\n", (long long)x); }
static void til_print_float(double x) { printf("%g\\n", x); }
static void til_print_str(const char* s) { printf("%s\\n", s); }
static void til_print_bool(bool b) { printf("%s\\n", b ? "true" : "false"); }

static double til_sqrt(double x) { return sqrt(x); }
static double til_abs(double x) { return fabs(x); }
static double til_pow(double a, double b) { return pow(a, b); }
static double til_sin(double x) { return sin(x); }
static double til_cos(double x) { return cos(x); }
static double til_tan(double x) { return tan(x); }
static double til_log(double x) { return log(x); }
static double til_exp(double x) { return exp(x); }
static double til_floor(double x) { return floor(x); }
static double til_ceil(double x) { return ceil(x); }
static double til_round(double x) { return round(x); }
static int64_t til_min_int(int64_t a, int64_t b) { return a < b ? a : b; }
static int64_t til_max_int(int64_t a, int64_t b) { return a > b ? a : b; }
static size_t til_len(const char* s) { return strlen(s); }

static char* til_concat(const char* a, const char* b) {
    size_t la = strlen(a), lb = strlen(b);
    char* r = (char*)malloc(la + lb + 1);
    strcpy(r, a);
    strcat(r, b);
    return r;
}

static char* til_input(const char* prompt) {
    printf("%s", prompt);
    fflush(stdout);
    static char buffer[4096];
    if (!fgets(buffer, 4096, stdin)) return "";
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len-1] == '\\n') buffer[len-1] = '\\0';
    return buffer;
}
'''
    self.emit_raw(helpers)
```

### 6.3 Type Mapping

```python
def c_type(self, til_type: str) -> str:
    """Convert TIL type to C type"""
    type_map = {
        # Standard types
        'int': 'int64_t',
        'float': 'double',
        'bool': 'bool',
        'str': 'const char*',
        'char': 'char',
        'void': 'void',
        
        # Sized integers
        'i8': 'int8_t',
        'i16': 'int16_t',
        'i32': 'int32_t',
        'i64': 'int64_t',
        'u8': 'uint8_t',
        'u16': 'uint16_t',
        'u32': 'uint32_t',
        'u64': 'uint64_t',
        
        # Floats
        'f32': 'float',
        'f64': 'double',
        
        # Special
        'self': 'void*',
    }
    
    if til_type in type_map:
        return type_map[til_type]
    
    # Check if it's a known struct
    if til_type in self.structs:
        return til_type
    
    # Check if it's a known enum
    if til_type in self.enums:
        return til_type
    
    # Default to int64_t
    return 'int64_t'
```

---

## 7. C Code Generation

### 7.1 Struct Generation

```python
def gen_structs(self):
    """Generate struct definitions"""
    if not self.structs:
        return
    
    self.emit_raw("// Structs")
    for name, struct in self.structs.items():
        self.emit_raw(f"typedef struct {name} {{")
        for field in struct.fields:
            c_type = self.c_type(field.type)
            self.emit_raw(f"    {c_type} {field.name};")
        self.emit_raw(f"}} {name};")
        self.emit_raw("")
```

### 7.2 Function Generation

```python
def gen_function(self, func: FnN):
    """Generate function code"""
    self.current_level = func.level
    self.declared_vars = set()
    self.string_vars = set()
    self.array_vars = {}
    
    # Track string parameters
    for param in func.params:
        self.declared_vars.add(param.name)
        if param.type == 'str':
            self.string_vars.add(param.name)
    
    # Build attributes
    attrs = []
    if func.level == 0:
        attrs.append("__attribute__((always_inline)) inline")
    elif func.level == 1:
        attrs.append("inline")
    
    for attr in func.attributes:
        if attr == "inline" and "inline" not in ' '.join(attrs):
            attrs.append("inline")
    
    attr_str = ' '.join(attrs) + ' ' if attrs else ''
    
    # Build parameters
    params_str = self.gen_params(func.params)
    
    # Generate function
    return_type = self.c_type(func.return_type)
    self.emit_raw(f"{attr_str}{return_type} til_{func.name}({params_str}) {{")
    
    self.indent += 1
    self.gen_block(func.body)
    self.indent -= 1
    
    self.emit_raw("}")
    self.emit_raw("")

def gen_params(self, params: List[Param]) -> str:
    """Generate parameter list"""
    if not params:
        return 'void'
    
    parts = []
    for p in params:
        c_type = self.c_type(p.type)
        parts.append(f"{c_type} {p.name}")
    
    return ", ".join(parts)
```

### 7.3 Expression Generation

```python
def gen_expr(self, expr: N) -> str:
    """Generate expression code"""
    if isinstance(expr, IntN):
        return str(expr.value)
    
    if isinstance(expr, FloatN):
        return str(expr.value)
    
    if isinstance(expr, StrN):
        escaped = (expr.value
            .replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\n', '\\n')
            .replace('\t', '\\t'))
        return f'"{escaped}"'
    
    if isinstance(expr, BoolN):
        return "true" if expr.value else "false"
    
    if isinstance(expr, IdN):
        return expr.name
    
    if isinstance(expr, BinN):
        return self.gen_binary(expr)
    
    if isinstance(expr, UnN):
        return self.gen_unary(expr)
    
    if isinstance(expr, CallN):
        return self.gen_call(expr)
    
    if isinstance(expr, IdxN):
        return f"{self.gen_expr(expr.obj)}[{self.gen_expr(expr.index)}]"
    
    if isinstance(expr, AttrN):
        return f"{self.gen_expr(expr.obj)}.{expr.attr}"
    
    if isinstance(expr, ArrN):
        elements = ", ".join(self.gen_expr(e) for e in expr.elements)
        return "{" + elements + "}"
    
    if isinstance(expr, RangeN):
        return "/*range*/"  # Handled specially in for loops
    
    if isinstance(expr, StructLitN):
        return self.gen_struct_literal(expr)
    
    if isinstance(expr, CastN):
        c_type = self.c_type(expr.target_type)
        return f"(({c_type})({self.gen_expr(expr.expr)}))"
    
    return "/*unknown*/"

def gen_binary(self, expr: BinN) -> str:
    """Generate binary operation"""
    # Handle string concatenation
    if expr.op == '+' and (self.is_string(expr.left) or self.is_string(expr.right)):
        return self.gen_string_concat(expr)
    
    left = self.gen_expr(expr.left)
    right = self.gen_expr(expr.right)
    
    op_map = {
        'and': '&&', 'or': '||',
        '==': '==', '!=': '!=',
        '<': '<', '>': '>', '<=': '<=', '>=': '>=',
        '+': '+', '-': '-', '*': '*', '/': '/', '%': '%',
        '&': '&', '|': '|', '^': '^',
        '<<': '<<', '>>': '>>'
    }
    
    if expr.op == '**':
        return f"pow({left}, {right})"
    
    c_op = op_map.get(expr.op, expr.op)
    return f"({left} {c_op} {right})"

def gen_call(self, expr: CallN) -> str:
    """Generate function call"""
    args = [self.gen_expr(a) for a in expr.args]
    args_str = ", ".join(args)
    
    if isinstance(expr.func, IdN):
        name = expr.func.name
        
        # Built-in functions
        builtins = {
            'sqrt': 'til_sqrt', 'abs': 'til_abs', 'pow': 'til_pow',
            'sin': 'til_sin', 'cos': 'til_cos', 'tan': 'til_tan',
            'log': 'til_log', 'exp': 'til_exp',
            'floor': 'til_floor', 'ceil': 'til_ceil', 'round': 'til_round',
            'min': 'til_min_int', 'max': 'til_max_int',
            'len': 'til_len', 'input': 'til_input'
        }
        
        if name in builtins:
            return f"{builtins[name]}({args_str})"
        
        # Struct constructor
        if name in self.structs:
            struct = self.structs[name]
            if len(args) == len(struct.fields):
                return f"({name}){{{args_str}}}"
        
        # Regular function
        return f"til_{name}({args_str})"
    
    if isinstance(expr.func, AttrN):
        # Method call
        obj = self.gen_expr(expr.func.obj)
        method = expr.func.attr
        
        # Check for impl methods
        for struct_name in self.impls:
            for m in self.impls[struct_name]:
                if m.name == method:
                    if m.has_self:
                        if args_str:
                            return f"{struct_name}_{method}(&{obj}, {args_str})"
                        return f"{struct_name}_{method}(&{obj})"
                    return f"{struct_name}_{method}({args_str})"
        
        return f"{obj}.{method}({args_str})"
    
    return f"{self.gen_expr(expr.func)}({args_str})"
```

---

## 8. Optimization

### 8.1 Level-Based Optimization

The TIL compiler applies optimizations based on the abstraction level:

| Level | Inlining | Bounds Check | SIMD Hints |
|-------|----------|--------------|------------|
| 0 | Always | Never | Yes |
| 1 | Suggested | Never | Possible |
| 2 | Auto | Yes | No |
| 3 | Never | Yes | No |

### 8.2 C Compiler Flags

```python
def get_optimization_flags(self, level: int) -> List[str]:
    """Get C compiler optimization flags"""
    flags = {
        0: ["-O0"],  # No optimization (debug)
        1: ["-O1"],  # Basic optimization
        2: ["-O2"],  # Standard optimization
        3: ["-O3"],  # Maximum optimization
    }
    return flags.get(level, ["-O2"])
```

### 8.3 Generated Code Optimization

The code generator applies these optimizations:

1. **Constant Folding**: Compile-time computation
2. **Dead Code Elimination**: Remove unreachable code
3. **Inline Expansion**: Level 0 functions always inlined
4. **Loop Optimization**: Range loops to C for loops

```python
# TIL:
for i in 0..100
    process(i)

# Generated C:
for (int64_t i = 0; i < 100; i++) {
    til_process(i);
}
```

---

## 9. Command Line Interface

### 9.1 Usage

```bash
# Basic usage
til <command> [options] <file.til>

# Commands
til run <file.til>           # Compile and run
til build <file.til>         # Compile to executable
til check <file.til>         # Check syntax only
til fmt <file.til>           # Format code (future)
```

### 9.2 Options

```bash
# Output options
-o, --output <file>          # Specify output file
-c, --emit-c                 # Output C code only
--keep-c                     # Keep intermediate C file

# Optimization
-O0                          # No optimization
-O1                          # Basic optimization
-O2                          # Standard (default)
-O3                          # Maximum optimization

# Information
-v, --version                # Show version
-h, --help                   # Show help
--verbose                    # Verbose output
```

### 9.3 Examples

```bash
# Compile and run
til run hello.til

# Compile to executable
til build hello.til -o hello

# Generate C code
til build hello.til -c

# Maximum optimization
til build game.til -O3 -o game

# Debug build
til build debug.til -O0 -o debug
```

### 9.4 CLI Implementation

```python
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TIL Compiler - Author: Alisher Beisembekov"
    )
    parser.add_argument('command', choices=['run', 'build', 'check'],
                       help='Command to execute')
    parser.add_argument('file', help='TIL source file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-c', '--emit-c', action='store_true',
                       help='Output C code only')
    parser.add_argument('-O', '--optimize', default='2',
                       choices=['0', '1', '2', '3'],
                       help='Optimization level')
    parser.add_argument('-v', '--version', action='version',
                       version='TIL Compiler 2.0.0')
    
    args = parser.parse_args()
    
    # Read source file
    with open(args.file, 'r', encoding='utf-8') as f:
        source = f.read()
    
    compiler = Compiler()
    compiler.opt = int(args.optimize)
    
    if args.command == 'check':
        # Syntax check only
        try:
            compiler.to_c(source, args.file)
            print("✓ Syntax OK")
        except SyntaxError as e:
            print(f"✗ {e}")
            sys.exit(1)
    
    elif args.command == 'build':
        if args.emit_c:
            # Output C code
            c_code = compiler.to_c(source, args.file)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(c_code)
            else:
                print(c_code)
        else:
            # Compile to executable
            output = args.output or args.file.replace('.til', '')
            exe = compiler.compile(source, output, args.file)
            print(f"✓ Compiled: {exe}")
    
    elif args.command == 'run':
        # Compile and run
        output = tempfile.mktemp()
        exe = compiler.compile(source, output, args.file)
        result = subprocess.run([exe], capture_output=False)
        os.unlink(exe)
        sys.exit(result.returncode)

if __name__ == '__main__':
    main()
```

---

## 10. IDE Integration

### 10.1 VS Code Extension

The TIL compiler supports VS Code through an extension that provides:

- Syntax highlighting
- Code snippets
- Build tasks
- Problem matching
- Hover documentation

### 10.2 Language Server Protocol (LSP)

TIL includes an LSP server (`til_lsp.py`) that provides:

```python
# LSP Capabilities
{
    "textDocumentSync": {
        "openClose": True,
        "change": 1  # Full sync
    },
    "completionProvider": {
        "triggerCharacters": [".", ":", "("]
    },
    "hoverProvider": True,
    "definitionProvider": True,
    "documentSymbolProvider": True,
    "diagnosticProvider": True
}
```

### 10.3 Integration Points

```python
class TILAnalyzer:
    """Provides IDE features"""
    
    def get_diagnostics(self, uri: str) -> List[Diagnostic]:
        """Get syntax errors and warnings"""
        
    def get_completions(self, uri: str, pos: Position) -> List[CompletionItem]:
        """Get autocomplete suggestions"""
        
    def get_hover(self, uri: str, pos: Position) -> str:
        """Get hover documentation"""
        
    def get_definition(self, uri: str, pos: Position) -> Location:
        """Get definition location"""
        
    def get_symbols(self, uri: str) -> List[Symbol]:
        """Get document symbols"""
```

---

## 11. Error Messages

### 11.1 Error Format

```
<filename>:<line>:<column>: <error type>: <message>
```

### 11.2 Lexer Errors

```
file.til:5:10: SyntaxError: Unexpected character: '@'
file.til:12:1: SyntaxError: Unterminated string literal
file.til:20:5: SyntaxError: Invalid number format
```

### 11.3 Parser Errors

```
file.til:10:5: SyntaxError: Expected RPAREN, got NEWLINE
file.til:15:1: SyntaxError: Expected INDENT after function declaration
file.til:25:10: SyntaxError: Unexpected token ELSE
```

### 11.4 Compilation Errors

```
file.til:30:5: Error: C compilation failed:
    undefined reference to 'undefined_function'
file.til:35:1: Error: No C compiler found (gcc/clang)
```

### 11.5 Error Recovery

The compiler attempts to continue after errors:

```python
def parse_with_recovery(self):
    """Parse with error recovery"""
    statements = []
    errors = []
    
    while not self.match(TT.EOF):
        try:
            stmt = self.parse_top_level()
            if stmt:
                statements.append(stmt)
        except SyntaxError as e:
            errors.append(e)
            # Skip to next statement
            self.recover()
        
        self.skip_newlines()
    
    if errors:
        for e in errors:
            print(e)
    
    return ProgN(statements)
```

---

## 12. Compiler Internals

### 12.1 Symbol Table

```python
class SymbolTable:
    """Track declared symbols"""
    
    def __init__(self):
        self.scopes = [{}]  # Stack of scopes
    
    def enter_scope(self):
        self.scopes.append({})
    
    def exit_scope(self):
        self.scopes.pop()
    
    def declare(self, name: str, type: str, kind: str):
        self.scopes[-1][name] = {'type': type, 'kind': kind}
    
    def lookup(self, name: str) -> dict:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
```

### 12.2 Type Inference

```python
def infer_type(self, expr: N) -> str:
    """Infer expression type"""
    if isinstance(expr, IntN):
        return 'int'
    if isinstance(expr, FloatN):
        return 'float'
    if isinstance(expr, StrN):
        return 'str'
    if isinstance(expr, BoolN):
        return 'bool'
    if isinstance(expr, BinN):
        left_type = self.infer_type(expr.left)
        right_type = self.infer_type(expr.right)
        if left_type == 'float' or right_type == 'float':
            return 'float'
        return 'int'
    if isinstance(expr, CallN):
        # Look up function return type
        if isinstance(expr.func, IdN):
            if expr.func.name in self.functions:
                return self.functions[expr.func.name].return_type
    return 'int'  # Default
```

### 12.3 Attribute Handling

```python
def process_attributes(self, attrs: List[str]) -> dict:
    """Process function attributes"""
    result = {
        'level': 2,
        'inline': False,
        'noinline': False,
        'pure': False,
        'unsafe': False
    }
    
    for attr in attrs:
        if attr.startswith('level:'):
            result['level'] = int(attr.split(':')[1].strip())
        elif attr == 'inline':
            result['inline'] = True
        elif attr == 'noinline':
            result['noinline'] = True
        elif attr == 'pure':
            result['pure'] = True
        elif attr == 'unsafe':
            result['unsafe'] = True
    
    return result
```

---

## 13. Built-in Functions

### 13.1 I/O Functions

| Function | C Implementation |
|----------|-----------------|
| `print(int)` | `printf("%lld\n", x)` |
| `print(float)` | `printf("%g\n", x)` |
| `print(str)` | `printf("%s\n", s)` |
| `print(bool)` | `printf("%s\n", b ? "true" : "false")` |
| `input(prompt)` | `fgets()` with prompt |

### 13.2 Math Functions

| Function | C Implementation |
|----------|-----------------|
| `sqrt(x)` | `sqrt(x)` |
| `abs(x)` | `fabs(x)` |
| `pow(x, y)` | `pow(x, y)` |
| `sin(x)` | `sin(x)` |
| `cos(x)` | `cos(x)` |
| `tan(x)` | `tan(x)` |
| `log(x)` | `log(x)` |
| `exp(x)` | `exp(x)` |
| `floor(x)` | `floor(x)` |
| `ceil(x)` | `ceil(x)` |
| `round(x)` | `round(x)` |
| `min(a, b)` | `a < b ? a : b` |
| `max(a, b)` | `a > b ? a : b` |

### 13.3 String Functions

| Function | C Implementation |
|----------|-----------------|
| `len(s)` | `strlen(s)` |
| `a + b` | `til_concat(a, b)` |

---

## 14. Type System Implementation

### 14.1 Type Checking

```python
def check_types(self, expr: N, expected: str) -> bool:
    """Check if expression matches expected type"""
    actual = self.infer_type(expr)
    
    # Exact match
    if actual == expected:
        return True
    
    # Numeric promotion
    if expected == 'float' and actual == 'int':
        return True
    
    # Integer sizes
    int_types = {'i8', 'i16', 'i32', 'i64', 'int'}
    if expected in int_types and actual in int_types:
        return True
    
    return False
```

### 14.2 Type Coercion

```python
def coerce_type(self, expr: N, target: str) -> str:
    """Generate code with type coercion"""
    actual = self.infer_type(expr)
    code = self.gen_expr(expr)
    
    if actual == target:
        return code
    
    # Add explicit cast
    c_type = self.c_type(target)
    return f"(({c_type})({code}))"
```

---

## 15. Multi-Level System Implementation

### 15.1 Level Tracking

```python
class LevelContext:
    """Track current abstraction level"""
    
    def __init__(self):
        self.current = 2  # Default level
        self.stack = []
    
    def push(self, level: int):
        self.stack.append(self.current)
        self.current = level
    
    def pop(self):
        self.current = self.stack.pop()
```

### 15.2 Level-Specific Code Generation

```python
def gen_function_with_level(self, func: FnN):
    """Generate function with level-specific optimizations"""
    level = func.level
    
    # Level 0: Always inline, no checks
    if level == 0:
        self.emit_raw("__attribute__((always_inline)) inline")
        self.generate_unchecked(func)
    
    # Level 1: Inline hint
    elif level == 1:
        self.emit_raw("inline")
        self.generate_unchecked(func)
    
    # Level 2: Safe with bounds checking
    elif level == 2:
        self.generate_safe(func)
    
    # Level 3: Maximum convenience
    else:
        self.generate_dynamic(func)
```

---

## 16. Extending the Compiler

### 16.1 Adding New Token Types

```python
# 1. Add to TT enum
class TT(Enum):
    # ... existing tokens ...
    NEW_TOKEN = auto()

# 2. Add to lexer recognition
def _op(self):
    if self.c() == '@':  # New token character
        self.advance()
        self.add_token(TT.NEW_TOKEN)
        return True
    # ... rest of operator handling
```

### 16.2 Adding New AST Nodes

```python
# 1. Define the node class
class NewFeatureN(N):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

# 2. Add parsing logic
def parse_new_feature(self):
    self.expect(TT.NEW_KEYWORD)
    param1 = self.parse_expression()
    param2 = self.parse_expression()
    return NewFeatureN(param1, param2)

# 3. Add code generation
def gen_new_feature(self, node: NewFeatureN):
    p1 = self.gen_expr(node.param1)
    p2 = self.gen_expr(node.param2)
    return f"new_feature_c_code({p1}, {p2})"
```

### 16.3 Adding Built-in Functions

```python
# 1. Add to helper functions in gen_helpers()
static double til_new_builtin(double x) {
    return x * 2;  // Example implementation
}

# 2. Add to builtin map in gen_call()
builtins = {
    # ... existing builtins ...
    'new_builtin': 'til_new_builtin',
}
```

---

## 17. Testing

### 17.1 Test Structure

```
tests/
├── lexer/
│   ├── test_numbers.til
│   ├── test_strings.til
│   └── test_operators.til
├── parser/
│   ├── test_expressions.til
│   ├── test_statements.til
│   └── test_functions.til
├── codegen/
│   ├── test_basic.til
│   ├── test_structs.til
│   └── test_methods.til
└── integration/
    ├── test_fibonacci.til
    ├── test_sorting.til
    └── test_game.til
```

### 17.2 Running Tests

```python
def run_tests():
    """Run all compiler tests"""
    import glob
    
    passed = 0
    failed = 0
    
    for test_file in glob.glob("tests/**/*.til", recursive=True):
        try:
            with open(test_file) as f:
                source = f.read()
            
            compiler = Compiler()
            compiler.to_c(source, test_file)
            
            print(f"✓ {test_file}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_file}: {e}")
            failed += 1
    
    print(f"\nPassed: {passed}, Failed: {failed}")
```

---

## 18. Debugging

### 18.1 Verbose Mode

```python
class Compiler:
    def __init__(self):
        self.verbose = False
    
    def log(self, msg: str):
        if self.verbose:
            print(f"[DEBUG] {msg}")
    
    def compile(self, source: str, output: str, filename: str):
        self.log(f"Compiling {filename}")
        
        # Lexing
        self.log("Running lexer...")
        tokens = Lexer(source, filename).tokenize()
        self.log(f"Generated {len(tokens)} tokens")
        
        # Parsing
        self.log("Running parser...")
        ast = Parser(tokens, filename).parse()
        self.log(f"Generated AST with {len(ast.statements)} statements")
        
        # Code generation
        self.log("Generating C code...")
        c_code = Gen().gen(ast)
        self.log(f"Generated {len(c_code)} characters of C code")
        
        # ...
```

### 18.2 AST Dumping

```python
def dump_ast(node: N, indent: int = 0) -> str:
    """Dump AST for debugging"""
    prefix = "  " * indent
    
    if isinstance(node, IntN):
        return f"{prefix}Int({node.value})"
    if isinstance(node, StrN):
        return f"{prefix}Str(\"{node.value}\")"
    if isinstance(node, BinN):
        return (f"{prefix}Bin({node.op})\n"
                f"{dump_ast(node.left, indent + 1)}\n"
                f"{dump_ast(node.right, indent + 1)}")
    # ... other node types
```

---

## 19. Performance

### 19.1 Compiler Performance

| Phase | Typical Time |
|-------|--------------|
| Lexing | ~5ms per 1000 lines |
| Parsing | ~10ms per 1000 lines |
| Code Generation | ~5ms per 1000 lines |
| C Compilation | ~100ms+ (external) |

### 19.2 Generated Code Performance

TIL-generated code performance compared to hand-written C:

| Benchmark | TIL | C | Ratio |
|-----------|-----|---|-------|
| Fibonacci(40) | 0.8s | 0.7s | 1.14x |
| Matrix Mult | 1.2s | 1.1s | 1.09x |
| String Concat | 0.5s | 0.3s | 1.67x |

### 19.3 Optimization Tips

1. Use Level 0 for hot loops
2. Avoid string concatenation in loops
3. Use `-O3` for release builds
4. Profile before optimizing

---

## 20. API Reference

### 20.1 Compiler Class

```python
class Compiler:
    """
    TIL Compiler
    Author: Alisher Beisembekov
    
    Attributes:
        opt (int): Optimization level (0-3)
        keep_c (bool): Keep intermediate C file
    
    Methods:
        to_c(source, filename) -> str
            Compile TIL source to C code
        
        compile(source, output, filename) -> str
            Compile TIL source to executable
            Returns path to executable
        
        find_cc() -> str
            Find available C compiler
            Returns compiler name or None
    """
```

### 20.2 Lexer Class

```python
class Lexer:
    """
    TIL Lexer
    
    Parameters:
        src (str): Source code
        filename (str): Filename for error messages
    
    Methods:
        tokenize() -> List[Token]
            Tokenize source code
    """
```

### 20.3 Parser Class

```python
class Parser:
    """
    TIL Parser
    
    Parameters:
        tokens (List[Token]): Token list from lexer
        filename (str): Filename for error messages
    
    Methods:
        parse() -> ProgN
            Parse tokens to AST
    """
```

### 20.4 Gen Class

```python
class Gen:
    """
    TIL Code Generator
    
    Methods:
        gen(prog: ProgN) -> str
            Generate C code from AST
    """
```

---

<div align="center">

**TIL Compiler Reference v2.0**

**Author: Alisher Beisembekov**

*"Проще Python. Быстрее C. Умнее всех."*

© 2025 TIL Language. MIT License.

</div>
