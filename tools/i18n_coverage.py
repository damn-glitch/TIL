#!/usr/bin/env python3
"""
TIL i18n coverage: which compiler / VM diagnostics have no Russian and Kazakh
translation in the message catalog (src/tilmsg.py).

    python3 tools/i18n_coverage.py              # summary + missing messages; exit 1 if any
    python3 tools/i18n_coverage.py --list       # every extracted message template
    python3 tools/i18n_coverage.py --skeleton   # catalog entries to paste into tilmsg.py
    python3 tools/i18n_coverage.py --unused     # catalog entries no extracted message needs
    python3 tools/i18n_coverage.py --json       # machine-readable report

Message templates are extracted statically:

* src/til.py and the feature modules next to it (src/til*.py), with Python's
  `ast`: the message (and hint) argument of every `self.method(...)` call
  whose method takes a `msg`/`message` (and `hint`) parameter -- `error`,
  `panic`, `consume(type, msg)`, ...; positions are read from the method's
  own signature, so new checker classes are picked up automatically --, of
  `ErrorReporter.format_error/format_warning(...)`, `TILPanic(...)`,
  `TILFatal(...)`, `CompileError(...)`, `ValueError(...)` (src/til.py only),
  the literal messages `BytecodeCompiler` compiles into programs (`push_const`
  before a PANIC: contracts, asserts, intents), the `ERROR_HINTS` table, and
  `print(..., file=sys.stderr)` (category 'cli': reported, not enforced).
  f-strings, `+` concatenations and conditional expressions are followed;
  local variables holding message text (`msg = f"..."`, `hint = ...`; inside
  `hint = "x; " + hint` the name stands for its other values), parameters that
  only ever receive literal values (`what="name"`) and methods of the same
  class that only return text (`self.reserved_note()`) are expanded; every
  other interpolated expression becomes a placeholder.
* vm/tilvm.c, with regular expressions: `fail(vm, "...")`, `failn(vm, ...)`,
  `failf(vm, ...)`, `type_error2(vm, "...", ...)`, `invalid_number(vm, ...,
  "...")` and the few messages assembled by hand in `vm->errbuf` (C_HANDBUILT
  below; each is used only while its anchor literal is still in the source).
  A helper that passes one of its `const char *` parameters on to
  fail/failn/failf is expanded at each of its call sites, and a string
  constant (`static const char E_NO_METHOD[] = "no method '";`) passed to
  them stands for its text.

A template is covered when the catalog translates it: each of its lines is
rendered with placeholder markers and looked up exactly as `til` does at run
time (tilmsg.split_frame + tilmsg.find; a marker fits every placeholder kind),
and no word or text placeholder of the matching entry may swallow several
words of the template (a generic "expects {n} argument(s)" does not cover
"expects 1 to 2 argument(s)": that would print a half-English message).
Run-time messages (categories 'runtime' and 'runtime-c') must be covered by
tilmsg.RUNTIME_MESSAGES: a program's error report is looked up there only.
Templates matched by `tilmsg.UNTRANSLATED` are English on purpose.
"""

import ast
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
TIL_PY = os.path.join(SRC, 'til.py')
TILVM_C = os.path.join(ROOT, 'vm', 'tilvm.c')


def python_sources() -> List[str]:
    """src/til.py and the feature modules next to it (tilprove.py, tiltypes.py, ...)."""
    names = sorted(n for n in os.listdir(SRC) if n.startswith('til') and n.endswith('.py')
                   and n != 'tilmsg.py')
    return [os.path.join(SRC, n) for n in names]

# Categories that must be fully translated; 'cli' (plain prints to stderr) and
# 'internal' are reported but not enforced. 'localized' = text passed to localize().
ENFORCED = ('syntax', 'semantic', 'vm-compile', 'type-warning', 'compile', 'runtime', 'runtime-c',
            'hint', 'localized')
# Categories printed in a VM's error report: looked up in tilmsg.RUNTIME_MESSAGES only.
RUNTIME_CATEGORIES = ('runtime', 'runtime-c')
LOCALIZE_FUNCS = ('localize', 'localize_report_bytes')

PH = None          # a placeholder part inside a template
MARK = '\x00'      # placeholders when a template is rendered as text


class Template:
    """One extracted message: literal parts and placeholders (PH)."""

    def __init__(self, parts: Sequence[Optional[str]], category: str, where: str):
        self.parts = normalize(parts)
        self.category = category
        self.where = [where]

    @property
    def key(self) -> Tuple:
        return tuple(self.parts)

    def text(self) -> str:
        """Catalog syntax: literal braces doubled, placeholders {}."""
        out = []
        for p in self.parts:
            out.append('{}' if p is PH else p.replace('{', '{{').replace('}', '}}'))
        return ''.join(out)

    def render(self) -> str:
        return ''.join(MARK if p is PH else p for p in self.parts)

    def regex(self) -> 're.Pattern':
        return re.compile(''.join('.*?' if p is PH else re.escape(p) for p in self.parts), re.S)

    def literal_size(self) -> int:
        return sum(len(p) for p in self.parts if p is not PH)


def normalize(parts: Sequence[Optional[str]]) -> List[Optional[str]]:
    """Merge adjacent literals, collapse adjacent placeholders, drop empty literals."""
    out: List[Optional[str]] = []
    for p in parts:
        if p is PH:
            if out and out[-1] is PH:
                continue
            out.append(PH)
        elif p:
            if out and out[-1] is not PH:
                out[-1] += p
            else:
                out.append(p)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Python extraction (src/til.py)
# ═══════════════════════════════════════════════════════════════════════════

MAX_ALTERNATIVES = 16


def _product(left: List[List], right: List[List]) -> List[List]:
    out = [a + b for a in left for b in right]
    return out[:MAX_ALTERNATIVES]


def _dedupe(alts: List[List]) -> List[List]:
    out: List[List] = []
    for a in alts:
        if a not in out:
            out.append(a)
    return out


class PyExtractor(ast.NodeVisitor):
    """Collects message templates from one Python module."""

    # classes whose error() reports syntax errors / warnings, and the category
    CLASS_CATEGORY = {'Lexer': 'syntax', 'Parser': 'syntax', 'SemanticChecker': 'semantic',
                      'BytecodeCompiler': 'vm-compile', 'TypeChecker': 'type-warning', 'TILVM': 'runtime'}
    MESSAGE_PARAMS = ('msg', 'message')
    RAISE_CATEGORY = {'TILPanic': 'runtime', 'TILFatal': 'runtime', 'CompileError': 'compile',
                      'ValueError': 'compile'}

    def __init__(self, source: str, filename: str):
        self.filename = os.path.relpath(filename, ROOT)
        self.main_module = os.path.basename(filename) == 'til.py'
        self.tree = ast.parse(source, filename)
        self.templates: List[Template] = []
        self.class_stack: List[ast.ClassDef] = []
        self.func_stack: List[ast.FunctionDef] = []
        # per class: method name -> (FunctionDef, [Call nodes of self.<method>(...)])
        self.methods: Dict[str, Dict[str, ast.FunctionDef]] = {}
        self.self_calls: Dict[str, Dict[str, List[ast.Call]]] = {}
        self.resolving: List[str] = []           # variables whose values are being expanded
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                meths = {f.name: f for f in node.body if isinstance(f, ast.FunctionDef)}
                self.methods[node.name] = meths
                calls: Dict[str, List[ast.Call]] = {}
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                            and isinstance(sub.func.value, ast.Name) and sub.func.value.id == 'self':
                        calls.setdefault(sub.func.attr, []).append(sub)
                self.self_calls[node.name] = calls

    def run(self) -> List[Template]:
        self.visit(self.tree)
        self.extract_error_hints()
        return self.templates

    # ── traversal ───────────────────────────────────────────────────────────
    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.func_stack.append(node)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def add(self, expr: ast.AST, category: str, node: ast.AST):
        where = f"{self.filename}:{node.lineno}"
        for parts in self.flatten(expr, 0):
            t = Template(parts, category, where)
            if any(p is not PH and re.search(r'[A-Za-z]', p) for p in t.parts):
                self.templates.append(t)

    def visit_Call(self, node: ast.Call):
        f = node.func
        cls = self.class_stack[-1].name if self.class_stack else None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'self' and cls:
            method = self.methods.get(cls, {}).get(f.attr)
            if method is not None and any(a.arg in self.MESSAGE_PARAMS for a in method.args.args):
                # error(msg), panic(msg), consume(type, msg), _syntax_error_at(node, msg, hint), ...
                self.extract_message_call(cls, method, node)
            elif f.attr == 'push_const' and cls == 'BytecodeCompiler' and node.args \
                    and self.is_text(node.args[0]):
                self.add(node.args[0], 'runtime', node)       # messages compiled into PANIC
        elif isinstance(f, ast.Attribute) and f.attr in ('format_error', 'format_warning'):
            # ErrorReporter used directly (feature modules): (line, col, msg, [code,] hint)
            hint_at = 4 if f.attr == 'format_error' else 3
            for name, at, category in (('msg', 2, 'semantic'), ('hint', hint_at, 'hint')):
                arg = next((k.value for k in node.keywords if k.arg == name), None)
                if arg is None and at < len(node.args):
                    arg = node.args[at]
                if arg is not None:
                    self.add(arg, category, node)
        elif isinstance(f, ast.Name) and f.id in self.RAISE_CATEGORY and node.args:
            category = self.RAISE_CATEGORY[f.id]
            if f.id == 'ValueError' and not self.main_module:
                category = 'internal'          # feature modules: plumbing, not diagnostics
            self.add(node.args[0], category, node)
        elif isinstance(f, ast.Name) and f.id in LOCALIZE_FUNCS and node.args:
            self.add(node.args[0], 'localized', node)
        elif isinstance(f, ast.Name) and f.id == 'print' and node.args and any(
                k.arg == 'file' and ast.unparse(k.value) == 'sys.stderr' for k in node.keywords) \
                and not self.calls_localize(node.args[0]):
            self.add(node.args[0], 'cli', node)
        self.generic_visit(node)

    def extract_message_call(self, cls: str, method: ast.FunctionDef, node: ast.Call):
        """A call of a method that takes a message (and maybe a hint) parameter."""
        category = self.CLASS_CATEGORY.get(cls, 'semantic')
        msg = self.argument(method, node, self.MESSAGE_PARAMS)
        if msg is not None:
            self.add(msg, category, node)
        hint = self.argument(method, node, ('hint',))
        if hint is not None:
            self.add(hint, 'hint', node)

    @staticmethod
    def argument(method: ast.FunctionDef, call: ast.Call, names: Tuple[str, ...]) -> Optional[ast.AST]:
        params = [a.arg for a in method.args.args][1:]          # without self
        for kw in call.keywords:
            if kw.arg in names:
                return kw.value
        for i, p in enumerate(params):
            if p in names and i < len(call.args):
                return call.args[i]
        return None

    def extract_error_hints(self):
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ERROR_HINTS'
                                                    for t in node.targets) and isinstance(node.value, ast.Dict):
                for v in node.value.values:
                    self.add(v, 'hint', v)

    @staticmethod
    def calls_localize(expr: ast.AST) -> bool:
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in LOCALIZE_FUNCS
                   for n in ast.walk(expr))

    # ── expressions -> alternatives of parts ────────────────────────────────
    def is_text(self, expr: ast.AST) -> bool:
        """A message-like expression: literals, f-strings, concatenations, if-else of those."""
        if isinstance(expr, ast.Constant):
            return isinstance(expr.value, (str, bytes))
        if isinstance(expr, ast.JoinedStr):
            return True
        if isinstance(expr, ast.IfExp):
            return self.is_text(expr.body) or self.is_text(expr.orelse)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return self.is_text(expr.left) or self.is_text(expr.right)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) \
                and expr.func.attr in ('strip', 'rstrip', 'lstrip'):
            return self.is_text(expr.func.value)
        return False

    def flatten(self, expr: ast.AST, depth: int) -> List[List[Optional[str]]]:
        if depth > 6:
            return [[PH]]
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, str):
                return [[expr.value]]
            if isinstance(expr.value, bytes):
                return [[expr.value.decode('utf-8', 'replace')]]
            return [[PH]]
        if isinstance(expr, ast.JoinedStr):
            alts: List[List[Optional[str]]] = [[]]
            for v in expr.values:
                if isinstance(v, ast.Constant):
                    alts = _product(alts, [[v.value]])
                elif isinstance(v, ast.FormattedValue):
                    if v.format_spec is not None or v.conversion not in (-1, None):
                        alts = _product(alts, [[PH]])
                    else:
                        alts = _product(alts, self.flatten_value(v.value, depth + 1))
            return alts
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return _product(self.flatten(expr.left, depth + 1), self.flatten(expr.right, depth + 1))
        if isinstance(expr, ast.IfExp):
            return (self.flatten(expr.body, depth + 1) + self.flatten(expr.orelse, depth + 1))[:MAX_ALTERNATIVES]
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) \
                and expr.func.attr in ('strip', 'rstrip', 'lstrip') and self.is_text(expr.func.value):
            return self.flatten(expr.func.value, depth + 1)
        if isinstance(expr, ast.Call):
            returned = self.method_texts(expr)
            if returned is not None:
                method, values = returned
                alts = []
                self.func_stack.append(method)
                try:
                    for v in values:
                        alts.extend(self.flatten(v, depth + 1))
                finally:
                    self.func_stack.pop()
                return _dedupe(alts)[:MAX_ALTERNATIVES]
        if isinstance(expr, ast.Name):
            return self.resolve_name(expr.id, depth)
        return [[PH]]

    def method_texts(self, call: ast.Call) -> Optional[Tuple[ast.FunctionDef, List[ast.AST]]]:
        """`self.reserved_note()`: a method of the same class whose every `return`
        gives text (literals, f-strings, ...) stands for those texts."""
        f = call.func
        if not (self.class_stack and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == 'self'):
            return None
        method = self.methods.get(self.class_stack[-1].name, {}).get(f.attr)
        if method is None or method in self.func_stack:
            return None
        values = [n.value for n in ast.walk(method) if isinstance(n, ast.Return)]
        if not values or not all(v is not None and self.is_text(v) for v in values):
            return None
        return method, values

    def flatten_value(self, expr: ast.AST, depth: int) -> List[List[Optional[str]]]:
        """An interpolated value: text-valued variables are expanded, the rest is a placeholder."""
        if isinstance(expr, ast.Name):
            return self.resolve_name(expr.id, depth)
        if self.is_text(expr):
            return self.flatten(expr, depth)
        return [[PH]]

    def resolve_name(self, name: str, depth: int) -> List[List[Optional[str]]]:
        if not self.func_stack:
            return [[PH]]
        func = self.func_stack[-1]
        values = []
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
                values.append(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id == name and node.value is not None:
                values.append(node.value)
        if values and name in self.resolving:
            # `hint = "prefix; " + hint`: inside its own value, a name has its other values
            values = [v for v in values if not any(isinstance(n, ast.Name) and n.id == name
                                                   for n in ast.walk(v))]
            if not values:
                return [[PH]]
        if values:
            if all(self.is_text(v) for v in values):
                alts: List[List[Optional[str]]] = []
                self.resolving.append(name)
                try:
                    for v in values:
                        alts.extend(self.flatten(v, depth + 1))
                finally:
                    self.resolving.pop()
                return _dedupe([self.letterless_as_placeholder(a) for a in alts])[:MAX_ALTERNATIVES]
            return [[PH]]
        literals = self.parameter_literals(func, name)
        if literals:
            return _dedupe([self.letterless_as_placeholder([s]) for s in literals])
        return [[PH]]

    @staticmethod
    def letterless_as_placeholder(parts: List[Optional[str]]) -> List[Optional[str]]:
        """`name = NATIVES[i] if ok else "?"`: a fallback value without letters is
        not message text, it stands for the (identifier) value of the variable."""
        if len(parts) == 1 and parts[0] is not PH and parts[0] and not re.search(r'[A-Za-z]', parts[0]):
            return [PH]
        return parts

    def parameter_literals(self, func: ast.FunctionDef, name: str) -> Optional[List[str]]:
        """Values of a parameter with a string default when every call site passes a literal."""
        if not self.class_stack:
            return None
        params = [a.arg for a in func.args.args]
        if name not in params:
            return None
        idx = params.index(name)
        defaults = func.args.defaults
        first_default = len(params) - len(defaults)
        if idx < first_default:
            return None
        default = defaults[idx - first_default]
        if not (isinstance(default, ast.Constant) and isinstance(default.value, str)):
            return None
        found = [default.value]
        for call in self.self_calls.get(self.class_stack[-1].name, {}).get(func.name, []):
            value = None
            for kw in call.keywords:
                if kw.arg == name:
                    value = kw.value
            if value is None and idx - 1 < len(call.args):
                value = call.args[idx - 1]
            if value is None:
                continue
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                return None
            if value.value not in found:
                found.append(value.value)
        return found


def extract_python(path: Optional[str] = None) -> List[Template]:
    """Templates of one Python file, or of every python_sources() file."""
    out: List[Template] = []
    for p in ([path] if path else python_sources()):
        with open(p, encoding='utf-8') as f:
            source = f.read()
        out.extend(PyExtractor(source, p).run())
    return out


# ═══════════════════════════════════════════════════════════════════════════
# C extraction (vm/tilvm.c)
# ═══════════════════════════════════════════════════════════════════════════

# Messages assembled by hand in vm->errbuf: (literal that must still exist, template parts)
C_HANDBUILT = [
    ('"unsupported operand types for "', ["unsupported operand types for ", PH, ": ", PH, " and ", PH]),
    ('" cannot be indexed"', [PH, " cannot be indexed"]),
    ('"key not found: "', ["key not found: ", PH]),
    ('"no field \'"', ["no field '", PH, "' on ", PH]),
]

_C_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '0': '\0', '\\': '\\', '"': '"', "'": "'", '?': '?'}


def strip_c_comments(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == '\\' else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            i = n if j < 0 else j + 2
            out.append(' ')
        elif text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def split_c_args(text: str, start: int) -> Tuple[List[str], int]:
    """Arguments of the call whose '(' is at text[start]; returns (args, end)."""
    args, depth, cur, i = [], 0, [], start + 1
    while i < len(text):
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < len(text) and text[j] != c:
                j += 2 if text[j] == '\\' else 1
            cur.append(text[i:j + 1])
            i = j + 1
            continue
        if c in '([{':
            depth += 1
        elif c in ')]}':
            if depth == 0:
                args.append(''.join(cur).strip())
                return args, i
            depth -= 1
        elif c == ',' and depth == 0:
            args.append(''.join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    return args, i


def c_literal(arg: str) -> Optional[str]:
    """The value of an argument made only of (adjacent) string literals, else None."""
    arg = arg.strip()
    if not arg.startswith('"'):
        return None
    out, i = [], 0
    while i < len(arg):
        if arg[i].isspace():
            i += 1
            continue
        if arg[i] != '"':
            return None
        i += 1
        while i < len(arg) and arg[i] != '"':
            if arg[i] == '\\' and i + 1 < len(arg):
                nxt = arg[i + 1]
                if nxt == 'x':
                    m = re.match(r'[0-9a-fA-F]{1,2}', arg[i + 2:])
                    out.append(chr(int(m.group(0), 16)) if m else 'x')
                    i += 2 + (len(m.group(0)) if m else 0)
                    continue
                out.append(_C_ESCAPES.get(nxt, nxt))
                i += 2
                continue
            out.append(arg[i])
            i += 1
        i += 1
    return ''.join(out)


def c_alternatives(arg: str) -> List[List[Optional[str]]]:
    """An argument: literal, `cond ? "a" : "b"` (both literals) or a placeholder."""
    lit = c_literal(arg)
    if lit is not None:
        return [[lit]]
    m = re.match(r'^(.*?)\?\s*("(?:[^"\\]|\\.)*")\s*:\s*("(?:[^"\\]|\\.)*")\s*$', arg, re.S)
    if m:
        return [[c_literal(m.group(2))], [c_literal(m.group(3))]]
    return [[PH]]


def _is_null(arg: str) -> bool:
    return arg.strip() in ('0', 'NULL', '(const char *)0')


_C_FAIL_CALL = re.compile(r'\b(fail|failn|failf|type_error2|invalid_number)\s*\(')
_C_STRING_CONST = re.compile(r'\bstatic\s+const\s+char\s+(?:\*\s*const\s+(\w+)|(\w+)\s*\[\s*\])\s*=\s*'
                             r'((?:"(?:[^"\\\n]|\\.)*"\s*)+);')
_C_HELPER = re.compile(r'\bstatic\s+int\s+(\w+)\s*\(\s*tilvm\s*\*\s*vm\s*,([^)]*)\)\s*\{')


def _c_block_end(text: str, brace: int) -> int:
    """Index just past the '}' that closes the '{' at text[brace]."""
    depth, i = 0, brace
    while i < len(text):
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < len(text) and text[j] != c:
                j += 2 if text[j] == '\\' else 1
            i = j + 1
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def c_message_parts(name: str, args: List[str]) -> List[List[Optional[str]]]:
    """Template alternatives of one fail/failn/failf/... call (arguments after vm)."""
    if name == 'fail':
        if args and args[0].strip() != 'vm->errbuf':
            alts = c_alternatives(args[0])
            if alts != [[PH]]:
                return alts
        return []
    if name == 'failn':
        alts: List[List[Optional[str]]] = [[]]
        for a in args:
            if not _is_null(a):
                alts = _product(alts, c_alternatives(a) if c_literal(a) is not None else [[PH]])
        return alts
    if name == 'failf' and len(args) == 5:
        a, _n1, b, _n2, c = args
        parts: List[Optional[str]] = [c_literal(a) or PH, PH]
        if not _is_null(b):
            parts += [c_literal(b) or PH, PH]
            if not _is_null(c):
                parts.append(c_literal(c) or PH)
        return [parts]
    if name == 'type_error2' and len(args) == 3:
        return [[c_literal(args[0]) or PH, PH, " and ", PH]]
    if name == 'invalid_number' and len(args) == 2:
        return [[c_literal(args[1]) or PH, '"', PH, '"']]
    return []


def extract_c(path: str = TILVM_C) -> List[Template]:
    """Templates of vm/tilvm.c. A helper `static int f(tilvm *vm, ..., const char *p, ...)`
    that passes its parameter p on to fail/failn/failf is expanded per call site
    (gfx_arg_error(vm, id, k, " must be 1..4096, got ", ...)), not reported itself."""
    with open(path, encoding='utf-8') as f:
        raw = f.read()
    text = strip_c_comments(raw)
    rel = os.path.relpath(path, ROOT)
    out: List[Template] = []

    def line_of(pos: int) -> int:
        return text.count('\n', 0, pos) + 1

    def add(alts: List[List[Optional[str]]], pos: int):
        for parts in alts:
            t = Template(parts, 'runtime-c', f"{rel}:{line_of(pos)}")
            if any(p is not PH and re.search(r'[A-Za-z]', p) for p in t.parts):
                out.append(t)

    # string constants shared by several messages (`static const char E_NO_METHOD[] = "no method '";`)
    consts = {(c.group(1) or c.group(2)): c.group(3).strip() for c in _C_STRING_CONST.finditer(text)}

    def c_args(start: int) -> List[str]:
        args, _end = split_c_args(text, start)
        return [consts.get(a.strip(), a) for a in args]

    def fail_calls(lo: int, hi: int):
        for m in _C_FAIL_CALL.finditer(text, lo, hi):
            if re.search(r'(?:int|static|void)\s+$', text[max(0, m.start() - 20):m.start()]):
                continue                                   # the definition itself
            args = c_args(m.end() - 1)
            if args and args[0] == 'vm':
                yield m, args[1:]

    # forwarding helpers: name -> (parameter names, [(call name, call args)], body range)
    helpers: Dict[str, Tuple[List[str], List[Tuple[str, List[str]]], Tuple[int, int]]] = {}
    for h in _C_HELPER.finditer(text):
        params = [re.findall(r'\w+', p)[-1] if re.findall(r'\w+', p) else '' for p in h.group(2).split(',')]
        strings = {p for p, decl in zip(params, h.group(2).split(',')) if 'char' in decl and '*' in decl}
        body = (h.end() - 1, _c_block_end(text, h.end() - 1))
        forwards = [(m.group(1), args) for m, args in fail_calls(*body)
                    if any(a.strip() in strings for a in args)]
        if forwards:
            helpers[h.group(1)] = (params, forwards, body)

    for m, args in fail_calls(0, len(text)):
        if any(lo <= m.start() < hi for _p, _f, (lo, hi) in helpers.values()):
            continue                                       # expanded at its call sites below
        add(c_message_parts(m.group(1), args), m.start())
    for name, (params, forwards, (lo, hi)) in helpers.items():
        for m in re.finditer(r'\b' + name + r'\s*\(', text):
            if lo - 200 <= m.start() < hi:
                continue                                   # the definition (and recursion)
            args = c_args(m.end() - 1)
            if not args or args[0] != 'vm':
                continue
            given = dict(zip(params, args[1:]))
            for call, call_args in forwards:
                add(c_message_parts(call, [given.get(a.strip(), a) for a in call_args]), m.start())
    for anchor, parts in C_HANDBUILT:
        pos = text.find(anchor)
        if pos >= 0:
            add([list(parts)], pos)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Catalog matching
# ═══════════════════════════════════════════════════════════════════════════

def load_catalog():
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import tilmsg
    return tilmsg


def merge(templates: List[Template]) -> List[Template]:
    """One template per distinct text (locations merged); categories keep the first."""
    seen: Dict[Tuple, Template] = {}
    for t in templates:
        if t.key in seen:
            for w in t.where:
                if w not in seen[t.key].where:
                    seen[t.key].where.append(w)
        else:
            seen[t.key] = t
    return list(seen.values())


def _absorbs_text(entry, groups: Dict[str, str]) -> bool:
    """True when a word or text placeholder of `entry` swallowed several words
    of the extracted template (`expects {n} argument(s)` matching `expects 1 to
    2 argument(s)`): the message would come out half English."""
    for name, value in groups.items():
        if entry.kinds.get(name) in ('', 'text') and re.search(r'[A-Za-z]', value.replace(MARK, '')) \
                and re.search(r'\s', value):
            return True
    return False


def scope_of(t: Template) -> Optional[str]:
    """The catalog list a template's lookup uses at run time (None: all of them)."""
    return 'runtime' if t.category in RUNTIME_CATEGORIES else None


def coverage_of(cat, t: Template, entries=None) -> Tuple[bool, List[str]]:
    """Is every line of template `t` translated, and by which catalog entries?
    Uses the catalog's own lookup (tilmsg.split_frame + tilmsg.find), so the
    check matches what `til` does with a real message."""
    used: List[str] = []
    for line in t.render().split('\n'):
        if not re.search(r'[A-Za-z]', line.replace(MARK, '')):
            continue
        kind, message = cat.split_frame(line)
        if kind in ('source', 'at') or not re.search(r'[A-Za-z]', message.replace(MARK, '')):
            continue                                     # pure frame: "Error: {}", "Syntax Error: {}"
        if not _translated(cat, message, scope_of(t), entries, used):
            return False, used
    return True, used


def _translated(cat, message: str, scope: Optional[str], entries, used: List[str], depth: int = 0) -> bool:
    """Does `message` (placeholders marked) come out fully translated? A nested
    {x:msg} value is looked up too, as `til` does."""
    hit = cat.find(message, scope, any_value=MARK) if entries is None else _find_in(entries, message)
    if hit is None or _absorbs_text(hit[0], hit[1]):
        return False
    used.append(hit[0].en)
    for name, value in hit[1].items():
        if hit[0].kinds.get(name) == 'msg' and depth < 8 and re.search(r'[A-Za-z]', value.replace(MARK, '')):
            if not _translated(cat, value, scope, entries, used, depth + 1):
                return False
    return True


def _find_in(entries, message: str):
    for e in sorted(entries, key=lambda e: -e.literal_size):
        groups = e.match(message, MARK)
        if groups is not None:
            return e, groups
    return None


def analyse(templates: Optional[List[Template]] = None) -> dict:
    cat = load_catalog()
    if templates is None:
        templates = extract_python() + extract_c()
    templates = merge(templates)
    allowed = [cat.Entry(text, *([text] * (len(cat.COLUMNS) - 1))) for text in cat.UNTRANSLATED]
    used = set()
    missing, covered, allowed_hits = [], [], []
    for t in templates:
        ok, hits = coverage_of(cat, t)
        if ok:
            covered.append(t)
            used.update(hits)
        elif coverage_of(cat, t, allowed)[0]:
            allowed_hits.append(t)
        else:
            missing.append(t)
    unused = [e for e in cat.entries() if e.en not in used]
    return {'templates': templates, 'covered': covered, 'missing': missing,
            'allowed': allowed_hits, 'unused': unused}


def enforced_missing(report: dict) -> List[Template]:
    return [t for t in report['missing'] if t.category in ENFORCED]


def catalog_list(t: Template) -> str:
    """The list of src/tilmsg.py a new entry for `t` belongs to."""
    if t.category in RUNTIME_CATEGORIES:
        return 'RUNTIME_MESSAGES'
    return 'COMMAND_LINE_MESSAGES' if t.category == 'cli' else 'COMPILER_MESSAGES'


def skeleton(t: Template) -> str:
    names = iter(f"x{i}" for i in range(1, 100))
    text = ''.join('{' + next(names) + '}' if p is PH else p.replace('{', '{{').replace('}', '}}')
                   for p in t.parts)
    return f"    ({text!r},\n     {text!r},   # ru: TODO\n     {text!r}),  # kk: TODO\n"


def main(argv: Sequence[str]) -> int:
    report = analyse()
    if '--json' in argv:
        def dump(ts):
            return [{'template': t.text(), 'category': t.category, 'where': t.where} for t in ts]
        print(json.dumps({'missing': dump(report['missing']), 'covered': len(report['covered']),
                          'allowed': dump(report['allowed']),
                          'unused': [e.en for e in report['unused']]}, ensure_ascii=False, indent=2))
        return 1 if enforced_missing(report) else 0
    if '--list' in argv:
        for t in sorted(report['templates'], key=lambda t: (t.category, t.text())):
            print(f"{t.category:<13} {t.text()!r:<70} {t.where[0]}")
        return 0
    if '--unused' in argv:
        for e in report['unused']:
            print(repr(e.en))
        return 0
    if '--skeleton' in argv:
        for t in enforced_missing(report):
            sys.stdout.write(f"    # {catalog_list(t)}: {', '.join(t.where)}\n" + skeleton(t))
        return 0
    by_cat: Dict[str, List[int]] = {}
    for key in ('covered', 'missing', 'allowed'):
        for t in report[key]:
            row = by_cat.setdefault(t.category, [0, 0, 0])
            row[('covered', 'missing', 'allowed').index(key)] += 1
    print(f"{'category':<14}{'translated':>11}{'missing':>9}{'english-only':>14}")
    for name in sorted(by_cat):
        c, m, a = by_cat[name]
        note = "" if name in ENFORCED else "   (not enforced)"
        print(f"{name:<14}{c:>11}{m:>9}{a:>14}{note}")
    total = sum(r[0] for r in by_cat.values()), sum(r[1] for r in by_cat.values())
    print(f"{'total':<14}{total[0]:>11}{total[1]:>9}")
    missing = report['missing']
    if missing:
        print("\nmissing translations:")
        for t in sorted(missing, key=lambda t: (t.category, t.text())):
            flag = "" if t.category in ENFORCED else " (not enforced)"
            print(f"  [{t.category}] {t.text()!r}  -- {', '.join(t.where[:3])}{flag}")
    if report['unused']:
        print(f"\n{len(report['unused'])} catalog entries are not used by any extracted message "
              f"(run with --unused to list them)")
    return 1 if enforced_missing(report) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
