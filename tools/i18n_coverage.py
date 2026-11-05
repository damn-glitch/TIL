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

* src/til.py and the feature modules next to it (src/til*.py: tilprove,
  tilpkg, tilfit, tildebug, tiltypes, ...), with Python's `ast`:
  - the message (and hint) argument of every call of a method or function of
    the module that takes a `msg`/`message` (and `hint`) parameter -- `error`,
    `panic`, `consume(type, msg)`, `Constraint.fail(s, message)` from a
    subclass, `c.fail(...)` when only one class has that method --; positions
    are read from the definition, so new checker classes are picked up
    automatically;
  - `ErrorReporter.format_error/format_warning(...)`, `TILPanic(...)`,
    `TILFatal(...)`, `CompileError(...)`, `NativeUnsupported(...)`,
    `SnapshotError(code, detail)`, `ValueError(...)` (src/til.py); in the
    other modules their own public exception classes (`PkgError(...)`,
    `Unsupported(...)`, `DebugError(...)`; category 'tool') and the builtin
    ones they print (`except ValueError as e: print(f"til fit: {e}")`);
  - `localize(...)` / `til.localize(...)` (category 'localized'), and the
    arguments of every function that passes a parameter on to such a sink
    (a "wrapper": `_(text)`, `self.say(text)`, `self.unsupported(what)`,
    `check_name(name, where)`): it is expanded at each call site with the
    parameter bound to the argument (a wrapper nothing in the module calls
    keeps its own messages);
  - MESSAGE_ARGUMENTS (an argument that is a message although its parameter
    is not called msg: `Obligation(..., what, ...)`), MESSAGE_LISTS
    (`fetcher.warnings.append(text)`), the literal messages
    `BytecodeCompiler` compiles into programs (`push_const` before a PANIC:
    contracts, asserts, intents), the `ERROR_HINTS` and `TILP_ERRORS` tables,
    and `print(..., file=sys.stderr)` without localize() (category 'cli':
    reported, not enforced).
  f-strings, `+` concatenations, `x or "default"` and conditional expressions
  are followed; local variables holding message text (`msg = f"..."`, `hint =
  ...`; inside `hint = "x; " + hint` the name stands for its other values),
  parameters that only ever receive literal values (`what="name"`), one-line
  text constants (`SHORT_HELP = "..."`), attributes only ever set to text
  (`self.quoted = f"'{key}'"`), methods of the same class and functions of
  the module that only return text (`self.reserved_note()`,
  `package_install_hint()`) are expanded; an attribute set to sentences only
  sometimes (`fail.outcome = "..."`, also `ok, fail.outcome = self.confirm()`)
  is a placeholder whose sentences are messages of their own; every other
  interpolated expression becomes a placeholder.
* vm/tilvm.c, with regular expressions: `fail(vm, "...")`, `failn(vm, ...)`,
  `failf(vm, ...)`, `type_error2(vm, "...", ...)`, `invalid_number(vm, ...,
  "...")`, `fail_type(vm, ...)` / `fail_types(vm, ...)` (a value's type) and the few messages assembled by hand in `vm->errbuf` (C_HANDBUILT
  below; each is used only while its anchor literal is still in the source).
  A helper that passes one of its `const char *` parameters on to
  fail/failn/failf is expanded at each of its call sites, and a string
  constant (`static const char E_NO_METHOD[] = "no method '";`) passed to
  them stands for its text.
* vm/main.c (the C VM's runner, category 'runner'): `fprintf(stderr,
  "tilvm: ...")` with printf conversions as placeholders, the reasons of
  `portable_error()` and RUNNER_HANDBUILT.

A template is covered when the catalog translates it: each of its lines is
rendered with placeholder markers and looked up exactly as `til` does at run
time (tilmsg.split_frame + tilmsg.find; a marker fits every placeholder kind;
the first line of a 'tool' message follows a frame such as "til pkg: error: "
and is looked up as it is), and no word or text placeholder of the matching
entry may swallow several words of the template (a generic "expects {n}
argument(s)" does not cover "expects 1 to 2 argument(s)": that would print a
half-English message); a nested {x:msg} value is checked too.
Run-time messages (categories 'runtime' and 'runtime-c') must be covered by
tilmsg.RUNTIME_MESSAGES: a program's error report is looked up there only.
Templates matched by `tilmsg.UNTRANSLATED` are English on purpose; so may be
a nested {x:msg} value (`str.{method}()` in "native backend: str.upper()") or
a line of a longer message (`  url#sha256=...`).
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
MAIN_C = os.path.join(ROOT, 'vm', 'main.c')


def python_sources() -> List[str]:
    """src/til.py and the feature modules next to it (tilprove.py, tiltypes.py, ...)."""
    names = sorted(n for n in os.listdir(SRC) if n.startswith('til') and n.endswith('.py')
                   and n != 'tilmsg.py')
    return [os.path.join(SRC, n) for n in names]

# Categories that must be fully translated; 'cli' (plain prints to stderr) and
# 'internal' are reported but not enforced. 'localized' = text passed to localize().
ENFORCED = ('syntax', 'semantic', 'vm-compile', 'type-warning', 'compile', 'runtime', 'runtime-c',
            'hint', 'localized', 'tool', 'runner')
# Messages printed after a frame ("til pkg: error: {e}"): their first line is
# looked up as it is ("til pkg add x: dependency ..." is no `til pkg add` frame).
BARE_CATEGORIES = ('tool',)
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


def is_message(t: 'Template') -> bool:
    """Text with a word in it (`{}x{}`, a screen size, is not a message)."""
    return any(p is not PH and re.search(r'[A-Za-z]{2}', p) for p in t.parts)


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


# Exceptions whose message argument is a diagnostic, and its category. The
# message is the first argument (RAISE_MESSAGE_ARG: another one).
RAISE_CATEGORY = {'TILPanic': 'runtime', 'TILFatal': 'runtime', 'CompileError': 'compile',
                  'NativeUnsupported': 'compile', 'ValueError': 'compile', 'SnapshotError': 'localized'}
RAISE_MESSAGE_ARG = {'SnapshotError': 1}           # SnapshotError(code, detail)
# Builtin exceptions: in a feature module their text is a diagnostic only when
# the module prints it (`except ValueError as e: print(f"til fit: {e}")`).
BUILTIN_EXCEPTIONS = ('Exception', 'ValueError', 'RuntimeError', 'KeyError', 'TypeError', 'OSError',
                      'LookupError', 'IndexError')
# Arguments that hold message text although their parameter is not called
# msg/message: (module, callee) -> (parameter, how). 'message': a message of
# its own (category: the module's default); 'attribute': the text of
# `self.<parameter>` in the methods of that class (CheckC.message() starts
# with `{self.what}`: "argument 'x' of 'f' is int, but ...").
MESSAGE_ARGUMENTS = {
    ('tilprove.py', 'Obligation'): ('what', 'message'),    # "division by zero" in an E-PROVE error
    ('tilprove.py', '_opaque'): ('what', 'message'),       # "a string": a value the prover does not model
    ('tiltypes.py', 'CheckC'): ('what', 'attribute'),      # "argument 'x' of 'f'" in an E-TYPE error
}
# Lists of messages printed later: `<obj>.<name>.append(text)` (module -> names).
MESSAGE_LISTS = {
    'tilpkg.py': ('warnings',),        # printed as "til pkg: warning: {}"
}
# Category of the messages of a module's own classes and exceptions (default:
# 'semantic' in til.py, 'tool' in the other modules: `til pkg`, `til prove`, ...).
MODULE_CATEGORY = {'til.py': 'semantic', 'tiltypes.py': 'semantic'}
# A call obj.name(...) (not self.name) is resolved by the method's name only
# when no builtin type has a method of that name (`xs.append(m)` is not a message).
_BUILTIN_METHODS = frozenset(n for t in (list, dict, str, set, bytes, bytearray, tuple, frozenset)
                             for n in dir(t)) | {'write', 'read', 'readline', 'flush', 'close', 'seek', 'tell'}
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _terminal(expr: ast.AST) -> str:
    """`PkgError` / `til.CompileError` -> the last name; '' for anything else."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return ''


class _Wrapper:
    """A function that passes one of its parameters on to a message sink
    (`def unsupported(self, what, node): raise NativeUnsupported(f"native
    backend: {what}...")`, `def _(text): return til.localize(text)`): it is
    expanded at each call site, with the parameter bound to the argument."""

    def __init__(self, func: ast.AST, cls: Optional[str], params: List[str], calls: List[ast.Call]):
        self.func, self.cls, self.params, self.calls = func, cls, params, calls


class PyExtractor(ast.NodeVisitor):
    """Collects message templates from one Python module."""

    # classes whose error() reports syntax errors / warnings, and the category
    CLASS_CATEGORY = {'Lexer': 'syntax', 'Parser': 'syntax', 'SemanticChecker': 'semantic',
                      'BytecodeCompiler': 'vm-compile', 'TypeChecker': 'type-warning', 'TILVM': 'runtime'}
    MESSAGE_PARAMS = ('msg', 'message')
    RAISE_CATEGORY = RAISE_CATEGORY

    def __init__(self, source: str, filename: str):
        self.filename = os.path.relpath(filename, ROOT)
        self.basename = os.path.basename(filename)
        self.main_module = self.basename == 'til.py'
        self.default_category = MODULE_CATEGORY.get(self.basename, 'semantic' if self.main_module else 'tool')
        self.tree = ast.parse(source, filename)
        self.templates: List[Template] = []
        self.class_stack: List[Optional[str]] = []        # class context (names) of the code being read
        self.func_stack: List[ast.AST] = []
        self.methods: Dict[str, Dict[str, ast.AST]] = {}
        self.bases: Dict[str, List[str]] = {}
        self.self_calls: Dict[str, Dict[str, List[ast.Call]]] = {}
        self.functions: Dict[str, ast.AST] = {}          # functions that are not methods (any depth)
        self.func_class: Dict[int, Optional[str]] = {}   # id(function) -> class context
        self.method_ids: set = set()
        self.attr_values: Dict[str, List[Tuple[ast.AST, Optional[ast.AST], Optional[str]]]] = {}
        self.module_texts: Dict[str, ast.AST] = {}        # NAME = "one line of text" at the top level
        self.tuple_attrs: Dict[str, List[Tuple[ast.Call, int, Optional[ast.AST], Optional[str]]]] = {}
        self.resolving: List[str] = []                     # variables whose values are being expanded
        self.bindings: Dict[int, Dict[str, List[List[Optional[str]]]]] = {}   # wrapper parameters
        self.fragments: Dict[str, List[List[Optional[str]]]] = {}   # 'CheckC.what' -> its texts
        self.collecting = False
        self._collect(self.tree, None, None)
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) \
                    and '\n' not in node.value.value:
                self.module_texts[node.targets[0].id] = node.value
        self.exceptions = self._find_exceptions()
        self.wrappers: Dict[int, _Wrapper] = {}
        self._find_wrappers()

    # ── what the module defines ─────────────────────────────────────────────
    def _collect(self, node: ast.AST, cls: Optional[str], func: Optional[ast.AST]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self.bases[child.name] = [_terminal(b) for b in child.bases]
                meths = {f.name: f for f in child.body if isinstance(f, _FUNCS)}
                self.methods[child.name] = meths
                self.method_ids.update(id(f) for f in meths.values())
                calls: Dict[str, List[ast.Call]] = {}
                for sub in ast.walk(child):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                            and isinstance(sub.func.value, ast.Name) and sub.func.value.id == 'self':
                        calls.setdefault(sub.func.attr, []).append(sub)
                self.self_calls[child.name] = calls
                self._collect(child, child.name, func)
            elif isinstance(child, _FUNCS):
                self.func_class[id(child)] = cls
                if id(child) not in self.method_ids:
                    self.functions.setdefault(child.name, child)
                self._collect(child, cls, child)
            else:
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    for t in targets:
                        if isinstance(t, ast.Attribute) and child.value is not None:
                            self.attr_values.setdefault(t.attr, []).append((child.value, func, cls))
                        elif isinstance(t, ast.Tuple) and isinstance(child.value, ast.Call):
                            for i, elt in enumerate(t.elts):     # a, x.attr = self.method(): its returns
                                if isinstance(elt, ast.Attribute):
                                    self.tuple_attrs.setdefault(elt.attr, []).append((child.value, i, func, cls))
                self._collect(child, cls, func)

    def lookup_method(self, cls: Optional[str], name: str) -> Optional[Tuple[str, ast.AST]]:
        """The method `name` of class `cls` or of its bases in this module: (owner, def)."""
        todo, seen = [cls], set()
        while todo:
            c = todo.pop(0)
            if c is None or c in seen or c not in self.methods:
                continue
            seen.add(c)
            if name in self.methods[c]:
                return c, self.methods[c][name]
            todo.extend(self.bases.get(c, []))
        return None

    def _find_exceptions(self) -> Dict[str, str]:
        """Exception classes of a feature module whose text reaches the user ->
        category: its public exception classes (PkgError, Unsupported, ...) and
        the builtin ones it prints (`except ValueError as e: print(f"...{e}")`)."""
        out: Dict[str, str] = {}
        if self.main_module:
            return out
        exc: set = set()
        changed = True
        while changed:
            changed = False
            for name, bases in self.bases.items():
                if name not in exc and any(b in exc or b in BUILTIN_EXCEPTIONS or b in RAISE_CATEGORY
                                           or b.endswith('Error') or b.endswith('Exception') for b in bases):
                    exc.add(name)
                    changed = True
        for name in exc:
            if not name.startswith('_') and name not in RAISE_CATEGORY:
                out[name] = self.default_category
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ExceptHandler) and node.name and node.type is not None:
                types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
                if self._shows(node.body, node.name):
                    for t in types:
                        name = _terminal(t)
                        if name in BUILTIN_EXCEPTIONS and name not in out:
                            out[name] = self.default_category
        return out

    @staticmethod
    def _shows(body: List[ast.AST], name: str) -> bool:
        """Does an except block print the exception `name` (print, localize, _, say)?"""
        for stmt in body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Call) and _terminal(n.func) in ('print', '_', 'say') + LOCALIZE_FUNCS \
                        and any(isinstance(x, ast.Name) and x.id == name
                                for a in list(n.args) + [k.value for k in n.keywords] for x in ast.walk(a)):
                    return True
        return False

    # ── calls ───────────────────────────────────────────────────────────────
    def resolve_callee(self, call: ast.Call, cls: Optional[str]) -> Optional[Tuple[ast.AST, Optional[str], str]]:
        """(definition, owner class, 'function' | 'method' | 'class') of a call to
        something this module defines, else None."""
        f = call.func
        if isinstance(f, ast.Name):
            if f.id in self.functions:
                return self.functions[f.id], None, 'function'
            if f.id in self.methods:
                hit = self.lookup_method(f.id, '__init__')
                return (hit[1], hit[0], 'class') if hit else None
            return None
        if not isinstance(f, ast.Attribute):
            return None
        if isinstance(f.value, ast.Name) and f.value.id == 'self':
            hit = self.lookup_method(cls, f.attr)
            return (hit[1], hit[0], 'method') if hit else None
        if isinstance(f.value, ast.Name) and f.value.id in self.methods:      # Class.method(...)
            hit = self.lookup_method(f.value.id, f.attr)
            return (hit[1], hit[0], 'method') if hit else None
        if f.attr in _BUILTIN_METHODS or f.attr.startswith('__'):
            return None
        found = [(c, m[f.attr]) for c, m in self.methods.items() if f.attr in m]
        if len(found) == 1:                       # c.fail(s, msg): the only method of that name
            func = found[0][1]
            if self.message_params(func, 'method') or id(func) in self.wrappers:
                return func, found[0][0], 'method'
        return None

    @staticmethod
    def call_params(func: ast.AST, kind: str) -> List[str]:
        """The parameters a call's positional arguments fill (without self / cls)."""
        params = [a.arg for a in func.args.args]
        static = any(_terminal(d) == 'staticmethod' for d in func.decorator_list)
        if kind in ('method', 'class') and not static and params:
            params = params[1:]
        return params

    def message_params(self, func: ast.AST, kind: str) -> List[str]:
        return [p for p in self.call_params(func, kind) if p in self.MESSAGE_PARAMS]

    @staticmethod
    def param_default(func: ast.AST, name: str) -> Optional[ast.AST]:
        a = func.args
        names = [x.arg for x in a.args]
        if name in names:
            i = names.index(name) - (len(names) - len(a.defaults))
            return a.defaults[i] if i >= 0 else None
        names = [x.arg for x in a.kwonlyargs]
        return a.kw_defaults[names.index(name)] if name in names else None

    @staticmethod
    def argument_for(func: ast.AST, params: List[str], call: ast.Call, name: str) -> Optional[ast.AST]:
        """The expression a call passes for parameter `name` (None: not given)."""
        for kw in call.keywords:
            if kw.arg == name:
                return kw.value
        if name in params:
            i = params.index(name)
            if i < len(call.args) and not any(isinstance(a, ast.Starred) for a in call.args[:i + 1]):
                return call.args[i]
        return None

    def base_sinks(self, call: ast.Call, cls: Optional[str]) -> List[Tuple[ast.AST, str]]:
        """The message arguments of one call and their categories (without wrappers)."""
        f = call.func
        name = _terminal(f)
        out: List[Tuple[ast.AST, str]] = []
        if name in self.exceptions or name in RAISE_CATEGORY:
            # raise PkgError(...), TILPanic(...), SnapshotError(code, detail), ...
            category = self.exceptions.get(name)
            if category is None:
                category = RAISE_CATEGORY[name]
                if name in BUILTIN_EXCEPTIONS and not self.main_module:
                    category = 'internal'          # feature modules: plumbing, unless printed (above)
            at = RAISE_MESSAGE_ARG.get(name, 0)
            if at < len(call.args):
                out.append((call.args[at], category))
            return out
        if isinstance(f, ast.Attribute) and f.attr in ('format_error', 'format_warning'):
            # ErrorReporter used directly (feature modules): (line, col, msg, [code,] hint)
            hint_at = 4 if f.attr == 'format_error' else 3
            for pname, at, category in (('msg', 2, 'semantic'), ('hint', hint_at, 'hint')):
                arg = next((k.value for k in call.keywords if k.arg == pname), None)
                if arg is None and at < len(call.args):
                    arg = call.args[at]
                if arg is not None:
                    out.append((arg, category))
            return out
        if name in LOCALIZE_FUNCS and call.args:            # localize(...), til.localize(...)
            return [(call.args[0], 'localized')]
        if isinstance(f, ast.Name) and f.id == 'print':
            if call.args and any(k.arg == 'file' and ast.unparse(k.value) == 'sys.stderr' for k in call.keywords) \
                    and not self.calls_localize(call.args[0], cls):
                out.append((call.args[0], 'cli'))
            return out
        if isinstance(f, ast.Attribute) and f.attr == 'append' and isinstance(f.value, ast.Attribute) \
                and f.value.attr in MESSAGE_LISTS.get(self.basename, ()) and call.args:
            return [(call.args[0], self.default_category)]      # fetcher.warnings.append(text)
        if cls == 'BytecodeCompiler' and isinstance(f, ast.Attribute) and f.attr == 'push_const' \
                and isinstance(f.value, ast.Name) and f.value.id == 'self' and call.args \
                and self.is_text(call.args[0]):
            return [(call.args[0], 'runtime')]                # messages compiled into PANIC
        callee = self.resolve_callee(call, cls)
        if callee is not None:
            func, owner, kind = callee
            params = self.call_params(func, kind)
            if self.message_params(func, kind):
                # error(msg), panic(msg), consume(type, msg), _syntax_error_at(node, msg, hint), ...
                category = self.CLASS_CATEGORY.get(cls) or self.CLASS_CATEGORY.get(owner) or self.default_category
                msg = next((a for a in (self.argument_for(func, params, call, p) for p in self.MESSAGE_PARAMS)
                            if a is not None), None)
                if msg is not None:
                    out.append((msg, category))
                hint = self.argument_for(func, params, call, 'hint')
                if hint is not None:
                    out.append((hint, 'hint'))
                return out
            spec = MESSAGE_ARGUMENTS.get((self.basename, name))
            if spec is not None:
                arg = self.argument_for(func, params, call, spec[0])
                if arg is not None:
                    out.append((arg, self.default_category if spec[1] == 'message'
                                else f"attribute:{name}.{spec[0]}"))
        return out

    def wrapper_call(self, call: ast.Call, cls: Optional[str]) -> Optional[Tuple[_Wrapper, List[str]]]:
        callee = self.resolve_callee(call, cls)
        if callee is None or id(callee[0]) not in self.wrappers:
            return None
        return self.wrappers[id(callee[0])], self.call_params(callee[0], callee[2])

    def sink_arguments(self, call: ast.Call, cls: Optional[str]) -> List[ast.AST]:
        """Expressions of a call that end up in a message (wrappers included)."""
        args = [a for a, _c in self.base_sinks(call, cls)]
        w = self.wrapper_call(call, cls)
        if w is not None:
            wrapper, params = w
            args.extend(a for a in (self.argument_for(wrapper.func, params, call, p) for p in wrapper.params)
                        if a is not None)
        return args

    def _find_wrappers(self):
        """Functions that pass a parameter on to a message sink (to a fixed point:
        `say(text)` -> `_(text)` -> `til.localize(text)`)."""
        funcs = [f for f in ast.walk(self.tree) if isinstance(f, _FUNCS)]
        for _round in range(10):
            changed = False
            for func in funcs:
                kind = 'method' if id(func) in self.method_ids else 'function'
                if self.message_params(func, kind):
                    continue                      # error(msg), panic(msg): a sink of its own
                params = set(self.call_params(func, kind)) | {a.arg for a in func.args.kwonlyargs}
                cls = self.func_class.get(id(func))
                used: List[str] = []
                calls: List[ast.Call] = []
                for call in self._calls_in(func):
                    names: set = set()
                    for arg in self.sink_arguments(call, cls):
                        names |= self.direct_names(arg, func, set())
                    hit = sorted(names & params)
                    if hit:
                        calls.append(call)
                        used.extend(p for p in hit if p not in used)
                old = self.wrappers.get(id(func))
                if calls and (old is None or len(old.calls) != len(calls) or old.params != used):
                    self.wrappers[id(func)] = _Wrapper(func, cls, used, calls)
                    changed = True
            if not changed:
                break
        # a wrapper nothing here calls (dispatched by name, called from another
        # module) keeps its own messages, with its parameters as placeholders
        called: set = set()
        for func in funcs + [self.tree]:
            cls = self.func_class.get(id(func)) if func is not self.tree else None
            calls = self._calls_in(func) if func is not self.tree else \
                [n for stmt in self.tree.body if not isinstance(stmt, _FUNCS + (ast.ClassDef,))
                 for n in ast.walk(stmt) if isinstance(n, ast.Call)]
            for call in calls:
                callee = self.resolve_callee(call, cls)
                if callee is not None and id(callee[0]) in self.wrappers:
                    called.add(id(callee[0]))
        for key in [k for k in self.wrappers if k not in called]:
            del self.wrappers[key]

    @staticmethod
    def _calls_in(func: ast.AST) -> List[ast.Call]:
        """The calls in a function's body (not in the functions and classes it defines)."""
        out: List[ast.Call] = []
        todo = list(func.body)
        while todo:
            n = todo.pop()
            if isinstance(n, _FUNCS + (ast.ClassDef,)):
                continue
            if isinstance(n, ast.Call):
                out.append(n)
            todo.extend(ast.iter_child_nodes(n))
        return out

    def direct_names(self, expr: ast.AST, func: ast.AST, seen: set) -> set:
        """Names whose text becomes part of a message (f-string parts, + operands,
        if-else branches; a local variable holding text stands for its values)."""
        if isinstance(expr, ast.Name):
            values = self._local_values(func, expr.id)
            if values and expr.id not in seen and all(self.is_text(v) for v in values):
                out: set = set()
                for v in values:
                    out |= self.direct_names(v, func, seen | {expr.id})
                return out
            return {expr.id}
        if isinstance(expr, ast.JoinedStr):
            out = set()
            for v in expr.values:
                if isinstance(v, ast.FormattedValue) and v.format_spec is None and v.conversion in (-1, None):
                    out |= self.direct_names(v.value, func, seen)
            return out
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return self.direct_names(expr.left, func, seen) | self.direct_names(expr.right, func, seen)
        if isinstance(expr, ast.IfExp):
            return self.direct_names(expr.body, func, seen) | self.direct_names(expr.orelse, func, seen)
        if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
            out = set()
            for v in expr.values:
                out |= self.direct_names(v, func, seen)
            return out
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) \
                and expr.func.attr in ('strip', 'rstrip', 'lstrip'):
            return self.direct_names(expr.func.value, func, seen)
        return set()

    @staticmethod
    def _local_values(func: ast.AST, name: str) -> List[ast.AST]:
        values = []
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
                values.append(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id == name and node.value is not None:
                values.append(node.value)
        return values

    def call_messages(self, call: ast.Call, cls: Optional[str], depth: int = 0
                      ) -> List[Tuple[List[List[Optional[str]]], str]]:
        """(alternatives, category) of every message one call produces; a wrapper
        call is expanded with its parameters bound to the call's arguments."""
        out = []
        for arg, category in self.base_sinks(call, cls):
            out.append((self.flatten(arg, 0), category))
            if not self.main_module:
                out.extend((alts, category) for alts in self.attribute_messages(arg, cls))
        w = self.wrapper_call(call, cls) if depth < 6 and not out else None
        if w is not None:
            wrapper, params = w
            bound: Dict[str, List[List[Optional[str]]]] = {}
            for p in wrapper.params:
                arg = self.argument_for(wrapper.func, params, call, p)
                if arg is None:                     # not given: the parameter's default
                    arg = self.param_default(wrapper.func, p)
                alts = self.flatten_value(arg, 0) if arg is not None else [[PH]]
                # an empty value (`where=""`) only chooses another branch (`f"{where}: " if where else ""`)
                bound[p] = _dedupe([a if any(x is PH or x for x in a) else [PH] for a in alts])
            self.func_stack.append(wrapper.func)
            self.class_stack.append(wrapper.cls)
            saved = self.bindings.get(id(wrapper.func))
            self.bindings[id(wrapper.func)] = bound
            try:
                for inner in wrapper.calls:
                    out.extend(self.call_messages(inner, wrapper.cls, depth + 1))
            finally:
                if saved is None:
                    self.bindings.pop(id(wrapper.func), None)
                else:
                    self.bindings[id(wrapper.func)] = saved
                self.class_stack.pop()
                self.func_stack.pop()
        return out

    def run(self) -> List[Template]:
        if any(m == self.basename and how == 'attribute' for (m, _c), (_p, how) in MESSAGE_ARGUMENTS.items()):
            self.collecting = True            # first the texts of the 'attribute' arguments
            self.visit(self.tree)
            self.collecting = False
        self.visit(self.tree)
        self.extract_error_hints()
        self.extract_message_tables()
        return self.templates

    # ── traversal ───────────────────────────────────────────────────────────
    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.func_stack.append(node)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def add(self, expr: ast.AST, category: str, node: ast.AST):
        self.add_alternatives(self.flatten(expr, 0), category, node)

    def add_alternatives(self, alts: List[List[Optional[str]]], category: str, node: ast.AST):
        where = f"{self.filename}:{node.lineno}"
        for parts in alts:
            t = Template(parts, category, where)
            if is_message(t):
                self.templates.append(t)

    def visit_Call(self, node: ast.Call):
        cls = self.class_stack[-1] if self.class_stack else None
        func = self.func_stack[-1] if self.func_stack else None
        wrapper = self.wrappers.get(id(func)) if func is not None else None
        if wrapper is None or not any(node is c for c in wrapper.calls):    # else: expanded at call sites
            for alts, category in self.call_messages(node, cls):
                if category.startswith('attribute:'):
                    if self.collecting:
                        known = self.fragments.setdefault(category[len('attribute:'):], [])
                        known.extend(a for a in alts if a not in known and any(p is not PH for p in a))
                elif not self.collecting:
                    self.add_alternatives(alts, category, node)
        self.generic_visit(node)

    def extract_error_hints(self):
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ERROR_HINTS'
                                                    for t in node.targets) and isinstance(node.value, ast.Dict):
                for v in node.value.values:
                    self.add(v, 'hint', v)

    def extract_message_tables(self):
        """TILP_ERRORS = {code: message}: the base messages of SnapshotError."""
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TILP_ERRORS'
                                                    for t in node.targets) and isinstance(node.value, ast.Dict):
                for v in node.value.values:
                    self.add(v, 'localized', v)

    def calls_localize(self, expr: ast.AST, cls: Optional[str]) -> bool:
        """Does the expression translate its text (localize(), or a wrapper of it)?"""
        for n in ast.walk(expr):
            if isinstance(n, ast.Call):
                if _terminal(n.func) in LOCALIZE_FUNCS:
                    return True
                w = self.wrapper_call(n, cls)
                if w is not None and self._localizes(w[0], set()):
                    return True
        return False

    def _localizes(self, wrapper: _Wrapper, seen: set) -> bool:
        if id(wrapper) in seen:
            return False
        seen.add(id(wrapper))
        for call in wrapper.calls:
            if _terminal(call.func) in LOCALIZE_FUNCS:
                return True
            w = self.wrapper_call(call, wrapper.cls)
            if w is not None and self._localizes(w[0], seen):
                return True
        return False

    # ── expressions -> alternatives of parts ────────────────────────────────
    def is_text(self, expr: ast.AST, calls: bool = True) -> bool:
        """A message-like expression: literals, f-strings, concatenations, if-else of
        those, a call of a function that returns text (`calls`)."""
        if calls and isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) \
                and expr.func.id in self.functions and self.method_texts(expr) is not None:
            return True
        if isinstance(expr, ast.Constant):
            return isinstance(expr.value, (str, bytes))
        if isinstance(expr, ast.JoinedStr):
            return True
        if isinstance(expr, ast.IfExp):
            return self.is_text(expr.body) or self.is_text(expr.orelse)
        if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):         # x or "default"
            return any(self.is_text(v) for v in expr.values)
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
        if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
            alts = []
            for v in expr.values:
                alts.extend(self.flatten_value(v, depth + 1))
            return _dedupe(alts)[:MAX_ALTERNATIVES]
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
        if isinstance(expr, ast.Attribute):
            return self.resolve_attribute(expr.attr, depth)
        return [[PH]]

    def method_texts(self, call: ast.Call) -> Optional[Tuple[ast.AST, List[ast.AST]]]:
        """`self.reserved_note()`: a method of the same class whose every `return`
        gives text (literals, f-strings, ...) stands for those texts; so does a
        function of the module that returns text or None (`package_install_hint()`)."""
        f = call.func
        cls = self.class_stack[-1] if self.class_stack else None
        if isinstance(f, ast.Name) and f.id in self.functions:
            func = self.functions[f.id]
            if func in self.func_stack:
                return None
            values = [n.value for n in ast.walk(func) if isinstance(n, ast.Return)
                      and not (n.value is None or (isinstance(n.value, ast.Constant) and n.value.value is None))]
            if not values or not all(self.is_text(v, False) for v in values):
                return None
            return func, values
        if not (cls and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'self'):
            return None
        hit = self.lookup_method(cls, f.attr)
        if hit is None or hit[1] in self.func_stack:
            return None
        method = hit[1]
        values = [n.value for n in ast.walk(method) if isinstance(n, ast.Return)]
        if not values or not all(v is not None and self.is_text(v, False) for v in values):
            return None
        return method, values

    def flatten_value(self, expr: ast.AST, depth: int) -> List[List[Optional[str]]]:
        """An interpolated value: text-valued variables are expanded, the rest is a placeholder."""
        if isinstance(expr, ast.Name):
            return self.resolve_name(expr.id, depth)
        if isinstance(expr, ast.Attribute):
            return self.resolve_attribute(expr.attr, depth)
        if self.is_text(expr):
            return self.flatten(expr, depth)
        return [[PH]]

    def attribute_messages(self, expr: ast.AST, cls: Optional[str]) -> List[List[List[Optional[str]]]]:
        """`hint = f"..., but {fail.outcome}"`: an attribute in a message that is not
        always text (so it is a placeholder) is a message of its own where it is
        set to a sentence (`fail.outcome = "the call passes an int ..."`, also
        through `ok, fail.outcome = self.confirm(...)`: the method's returns)."""
        out: List[List[List[Optional[str]]]] = []
        func = self.func_stack[-1] if self.func_stack else None
        for attr in self._text_attributes(expr, func, set()):
            values = self.attr_values.get(attr, [])
            if values and attr not in self.tuple_attrs and all(self.is_text(v) for v, _f, _c in values):
                continue                                      # expanded in place (resolve_attribute)
            sources = [(v, f, c) for v, f, c in values if self.is_text(v)]
            for call, i, f, c in self.tuple_attrs.get(attr, []):
                callee = self.resolve_callee(call, c)
                if callee is not None:
                    for r in ast.walk(callee[0]):
                        if isinstance(r, ast.Return) and isinstance(r.value, ast.Tuple) \
                                and i < len(r.value.elts) and self.is_text(r.value.elts[i]):
                            sources.append((r.value.elts[i], callee[0], self.func_class.get(id(callee[0]))))
            for value, f, c in sources:
                self.func_stack.append(f)
                self.class_stack.append(c)
                try:
                    alts = self.flatten(value, 1)
                finally:
                    self.class_stack.pop()
                    self.func_stack.pop()
                out.append([a for a in alts if any(p is not PH and ' ' in p for p in a)])
        return out

    def _text_attributes(self, expr: ast.AST, func: Optional[ast.AST], seen: set) -> set:
        """Attribute names at the text positions of an expression (through local variables)."""
        if isinstance(expr, ast.Attribute):
            return {expr.attr}
        if isinstance(expr, ast.Name):
            if func is None or expr.id in seen:
                return set()
            out: set = set()
            for v in self._local_values(func, expr.id):
                if self.is_text(v):
                    out |= self._text_attributes(v, func, seen | {expr.id})
            return out
        if isinstance(expr, ast.JoinedStr):
            out = set()
            for v in expr.values:
                if isinstance(v, ast.FormattedValue) and v.format_spec is None and v.conversion in (-1, None):
                    out |= self._text_attributes(v.value, func, seen)
            return out
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return self._text_attributes(expr.left, func, seen) | self._text_attributes(expr.right, func, seen)
        if isinstance(expr, ast.IfExp):
            return self._text_attributes(expr.body, func, seen) | self._text_attributes(expr.orelse, func, seen)
        if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
            out = set()
            for v in expr.values:
                out |= self._text_attributes(v, func, seen)
            return out
        return set()

    def resolve_attribute(self, attr: str, depth: int) -> List[List[Optional[str]]]:
        """`rep.quoted`: an attribute that is only ever assigned text (`self.quoted =
        f"'{key}'"`) stands for those texts; any other attribute is a placeholder."""
        cls = self.class_stack[-1] if self.class_stack else None
        if f"{cls}.{attr}" in self.fragments:
            return self.fragments[f"{cls}.{attr}"][:MAX_ALTERNATIVES]
        values = self.attr_values.get(attr)
        if not values or depth > 4 or len(values) > 4 or attr in self.tuple_attrs \
                or not all(self.is_text(v) for v, _f, _c in values):
            return [[PH]]
        alts: List[List[Optional[str]]] = []
        for value, func, cls in values:
            self.func_stack.append(func)
            self.class_stack.append(cls)
            try:
                alts.extend(self.flatten(value, depth + 1))
            finally:
                self.class_stack.pop()
                self.func_stack.pop()
        return _dedupe([self.letterless_as_placeholder(a) for a in alts])[:MAX_ALTERNATIVES]

    def resolve_name(self, name: str, depth: int) -> List[List[Optional[str]]]:
        func = self.func_stack[-1] if self.func_stack else None
        if func is None:
            return self.module_text(name, depth)
        bound = self.bindings.get(id(func))
        if bound is not None and name in bound:
            return bound[name]
        values = self._local_values(func, name)
        if values and name in self.resolving:
            # `hint = "prefix; " + hint`: inside its own value, a name has its other values
            values = [v for v in values if not any(isinstance(n, ast.Name) and n.id == name
                                                   for n in ast.walk(v))]
            if not values:
                return [[PH]]
        if len(values) > 1:          # `label = ""` before the real value: not a message's text
            values = [v for v in values if not (isinstance(v, ast.Constant) and v.value == "")] or values
        if values:
            if any(self.is_text(v) for v in values):
                # every text value; a value computed otherwise ("; ".join(parts)) is a placeholder
                alts: List[List[Optional[str]]] = []
                self.resolving.append(name)
                try:
                    for v in values:
                        alts.extend(self.flatten(v, depth + 1) if self.is_text(v) else [[PH]])
                finally:
                    self.resolving.pop()
                return _dedupe([self.letterless_as_placeholder(a) for a in alts])[:MAX_ALTERNATIVES]
            return [[PH]]
        literals = self.parameter_literals(func, name)
        if literals:
            return _dedupe([self.letterless_as_placeholder([s]) for s in literals])
        if name in [a.arg for a in func.args.args + func.args.kwonlyargs]:
            return [[PH]]
        return self.module_text(name, depth)

    def module_text(self, name: str, depth: int) -> List[List[Optional[str]]]:
        """A top-level constant holding one line of text (`SHORT_HELP = "..."`)."""
        value = self.module_texts.get(name)
        return [[value.value]] if value is not None else [[PH]]

    @staticmethod
    def letterless_as_placeholder(parts: List[Optional[str]]) -> List[Optional[str]]:
        """`name = NATIVES[i] if ok else "?"`: a fallback value without letters is
        not message text, it stands for the (identifier) value of the variable."""
        if len(parts) == 1 and parts[0] is not PH and parts[0] and not re.search(r'[A-Za-z]', parts[0]):
            return [PH]
        return parts

    def parameter_literals(self, func: ast.AST, name: str) -> Optional[List[str]]:
        """Values of a parameter with a string default when every call site passes a literal."""
        cls = self.class_stack[-1] if self.class_stack else None
        if not cls:
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
        for call in self.self_calls.get(cls, {}).get(func.name, []):
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
            end = n if j < 0 else j + 2
            out.append(' ' + '\n' * text.count('\n', i, end))     # (line numbers stay right)
            i = end
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


_C_FAIL_CALL = re.compile(r'\b(fail|failn|failf|type_error2|invalid_number|fail_type|fail_types)\s*\(')
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
                alts = _product(alts, c_alternatives(a))       # (`k ? "a" : "b"` gives both)
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
    if name in ('fail_type', 'fail_types') and len(args) == (4 if name == 'fail_type' else 5):
        # fail_type(vm, p0, p1, value, p3) / fail_types(vm, p0, p1, a, p3, b): the values'
        # types are placeholders (any argument may be 0)
        alts: List[List[Optional[str]]] = [[]]
        for k, a in enumerate(args):
            if _is_null(a):
                continue
            alts = _product(alts, [[PH]] if k in (2, 4) else c_alternatives(a))
        return alts
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
            if is_message(t):
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


# The C VM's runner (vm/main.c): `tilvm: ...` lines, translated by `til run --cvm`.
# Functions whose string literals are messages (the reasons of the .tilp error codes):
RUNNER_MESSAGE_FUNCTIONS = ('portable_error',)
# `tilvm: cannot load %s (error %d%s)`: the reasons chosen by `rc == -3 ? "..." : ...`
RUNNER_HANDBUILT = [
    ('"tilvm: cannot load %s (error %d%s)\\n"', [
        ["tilvm: cannot load ", PH, " (error ", PH, ")"],
        ["tilvm: cannot load ", PH, " (error ", PH, ": program uses floats, VM built without them)"],
        ["tilvm: cannot load ", PH, " (error ", PH, ": integer constant too large for this VM)"],
        ["tilvm: cannot load ", PH, " (error ", PH, ": too many functions for this VM build (TILVM_MAX_FUNCS))"],
        ["tilvm: cannot load ", PH, " (error ", PH, ": program larger than 64 KB (TILVM_ADDR16 build))"],
    ]),
]
_PRINTF_SPEC = re.compile(r'%%|%[-+ #0]*(?:[0-9]+|\*)?(?:\.(?:[0-9]+|\*))?(?:hh|h|ll|l|z|j|t|L)?[diouxXcsfFeEgGp]')


def printf_parts(fmt: str) -> List[Optional[str]]:
    """'cannot read %s\\n' -> ['cannot read ', PH] (the line end dropped)."""
    parts: List[Optional[str]] = []
    pos = 0
    for m in _PRINTF_SPEC.finditer(fmt):
        parts.append(fmt[pos:m.start()])
        parts.append('%' if m.group(0) == '%%' else PH)
        pos = m.end()
    parts.append(fmt[pos:].rstrip('\n'))
    return parts


def extract_runner(path: str = MAIN_C) -> List[Template]:
    """Messages of the C VM's runner: `fprintf(stderr, "tilvm: ...")` (its usage
    line and the `[tilvm]` statistics of --stats are not messages), the string
    literals of RUNNER_MESSAGE_FUNCTIONS and RUNNER_HANDBUILT."""
    with open(path, encoding='utf-8') as f:
        text = strip_c_comments(f.read())
    rel = os.path.relpath(path, ROOT)
    out: List[Template] = []

    def add(parts: List[Optional[str]], pos: int):
        t = Template(parts, 'runner', f"{rel}:{text.count(chr(10), 0, pos) + 1}")
        if is_message(t):
            out.append(t)

    for m in re.finditer(r'\bfprintf\s*\(', text):
        args, _end = split_c_args(text, m.end() - 1)
        fmt = c_literal(args[1]) if len(args) > 1 and args[0] == 'stderr' else None
        if fmt is not None and fmt.startswith('tilvm: '):
            add(printf_parts(fmt), m.start())
    for name in RUNNER_MESSAGE_FUNCTIONS:
        m = re.search(r'\b' + name + r'\s*\([^)]*\)\s*\{', text)
        if m:
            body = text[m.end() - 1:_c_block_end(text, m.end() - 1)]
            for lit in re.finditer(r'"(?:[^"\\\n]|\\.)*"', body):
                value = c_literal(lit.group(0))
                if value and ' ' in value:
                    add([value], m.start() + lit.start())
    for anchor, alternatives in RUNNER_HANDBUILT:
        pos = text.find(anchor)
        if pos >= 0:
            for parts in alternatives:
                add(list(parts), pos)
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


def coverage_of(cat, t: Template, entries=None, allowed=None) -> Tuple[bool, List[str]]:
    """Is every line of template `t` translated, and by which catalog entries?
    Uses the catalog's own lookup (tilmsg.split_frame + tilmsg.find), so the
    check matches what `til` does with a real message. A nested {x:msg} value
    may also be one of the `allowed` (English on purpose: `str.{method}()` in
    "native backend: str.upper() (line 3)" is code)."""
    used: List[str] = []
    english = 0                     # lines that are English on purpose (`  url#sha256=...` is code)
    lines = t.render().split('\n')
    for i, line in enumerate(lines):
        if not re.search(r'[A-Za-z]', line.replace(MARK, '')):
            continue
        if i == 0 and t.category in BARE_CATEGORIES:
            kind, message = None, line          # it follows a frame: "til pkg: error: <message>"
        else:
            kind, message = cat.split_frame(line)
        if kind in ('source', 'at') or not re.search(r'[A-Za-z]', message.replace(MARK, '')):
            continue                                     # pure frame: "Error: {}", "Syntax Error: {}"
        if not _translated(cat, message, scope_of(t), entries, used, allowed=allowed):
            if not (allowed and len(lines) > 1 and _find_in(allowed, message) is not None):
                return False, used
            english += 1
    return not english or bool(used), used


def _translated(cat, message: str, scope: Optional[str], entries, used: List[str], depth: int = 0,
                allowed=None) -> bool:
    """Does `message` (placeholders marked) come out fully translated? A nested
    {x:msg} value is looked up too, as `til` does."""
    hit = cat.find(message, scope, any_value=MARK) if entries is None else _find_in(entries, message)
    if hit is None and depth and allowed:
        return _find_in(allowed, message) is not None         # an English part on purpose
    if hit is None or _absorbs_text(hit[0], hit[1]):
        return False
    used.append(hit[0].en)
    for name, value in hit[1].items():
        if hit[0].kinds.get(name) == 'msg' and depth < 8 and re.search(r'[A-Za-z]', value.replace(MARK, '')):
            if not _translated(cat, value, scope, entries, used, depth + 1, allowed):
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
        templates = extract_python() + extract_c() + extract_runner()
    templates = merge(templates)
    allowed = [cat.Entry(text, *([text] * (len(cat.COLUMNS) - 1))) for text in cat.UNTRANSLATED]
    used = set()
    missing, covered, allowed_hits = [], [], []
    for t in templates:
        ok, hits = coverage_of(cat, t, allowed=allowed)
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
