#!/usr/bin/env python3
"""
tilprove - prove TIL contracts for ALL inputs with the Z3 SMT solver (TIL 3.1).

    til prove prog.til [--timeout MS] [--function NAME] [--allow-unsupported] [-v]
    til check --prove prog.til     every function with contracts, assert or invariant
    til check prog.til             only #[level: 4] functions (when Z3 is installed)

For every function in scope the prover shows, for every int/bool input that
satisfies the function's #[requires], that

  * each #[ensures] holds whenever the function returns,
  * each `assert` / `invariant` holds (loop invariants: on entry and after
    every iteration),
  * each call satisfies the callee's #[requires],
  * no division or modulo by zero happens,

with TIL's exact integer semantics: 64-bit two's complement wrap-around, `/`
and `%` truncating like C, shifts masked to 0..63 - the semantics of both TIL
VMs.  A counterexample is confirmed by running the function with it on the
reference VM.  Constructs the prover does not model make the function
`unsupported`; they are never proved by accident.

The integer model, what exactly is proved and the limitations are described in
docs/TIL_PROOFS.md.
"""

import os
import re
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple

import til

z3 = None                     # imported on first use (plain `til check` stays fast)

INT_MIN = -(1 << 63)
INT_MAX = (1 << 63) - 1
DEFAULT_TIMEOUT_MS = 10000    # per solver query
PROBE_STEPS = 300000          # step budget when a counterexample is run on the VM
SEARCH_RUNS = 40              # edge-value runs when the solver's candidate does not reproduce
SEARCH_STEPS = 20000          # step budget of each of those runs
INSTALL_HINT = "pip install z3-solver"
ERROR_CODE = "E-PROVE"

INT, BOOL, NIL, OPAQUE = 'int', 'bool', 'nil', 'opaque'


def load_z3() -> bool:
    """Import the Z3 Python bindings; False when they are not installed."""
    global z3
    if z3 is not None:
        return True
    try:
        import z3 as _z3
    except Exception:          # ImportError, or a broken native library
        return False
    z3 = _z3
    return True


def z3_version() -> str:
    return z3.get_version_string() if load_z3() else "not installed"


# ═══════════════════════════════════════════════════════════════════════════════
#                         SYMBOLIC VALUES AND STATES
# ═══════════════════════════════════════════════════════════════════════════════

class Unsupported(Exception):
    """The function uses something the prover does not model."""

    def __init__(self, reason: str, node=None):
        super().__init__(reason)
        self.reason = reason
        self.node = node
        self.loc: Optional[Tuple[int, int]] = None     # set by the encoder (attribute lines...)


class _NotTranslatable(Exception):
    """The integer encoding cannot express an operation (bit operations...);
    only the bit-vector encoding is used for the function."""


class _V:
    """A symbolic TIL value: an int (a 64-bit bit-vector term, or a
    mathematical integer term in the integer encoding), a bool, nil, or an
    opaque value the prover does not model (string, float, array, ...).
    Opaque values may flow through a function freely; a contract, assert or
    division that depends on one makes the function unsupported."""
    __slots__ = ('sort', 'z', 'what')

    def __init__(self, sort: str, z=None, what: str = ""):
        self.sort = sort
        self.z = z
        self.what = what


_NIL = _V(NIL)


def _opaque(what: str) -> _V:
    return _V(OPAQUE, None, what)


class _State:
    """A program point: path condition (tuple of z3 Bool conjuncts) and the
    local variables (one dict per lexical scope)."""
    __slots__ = ('pc', 'scopes')

    def __init__(self, pc: tuple, scopes: List[Dict[str, _V]]):
        self.pc = pc
        self.scopes = scopes

    def copy(self) -> '_State':
        return _State(self.pc, [dict(s) for s in self.scopes])

    def lookup(self, name: str) -> Optional[_V]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def assign(self, name: str, value: _V) -> bool:
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = value
                return True
        return False

    def declare(self, name: str, value: _V):
        self.scopes[-1][name] = value


class _LoopCtx:
    def __init__(self, depth: int):
        self.depth = depth
        self.breaks: List[_State] = []
        self.continues: List[_State] = []


class Obligation:
    """One condition to prove: `pc => goal`."""
    __slots__ = ('kind', 'what', 'line', 'col', 'pc', 'goal', 'expect', 'show', 'unmodeled', 'loop', 'prefer')

    def __init__(self, kind, what, line, col, pc, goal, expect, show, unmodeled, loop, prefer=None):
        # ensures | call | div | assert | invariant | loop-entry | loop-step | range | overflow
        self.kind = kind
        self.what = what            # human readable: "#[ensures: result > x]"
        self.line = line            # where to report it (in the file being proved)
        self.col = col
        self.pc = pc
        self.goal = goal
        self.expect = expect        # (panic message, [(function, line or 0), ...]) seen on the VM, or None
        self.show = show            # [(label, _V)] extra values to show with a counterexample
        self.unmodeled = unmodeled  # number of unmodeled operations before it (exactness)
        self.loop = loop            # inside / after a loop abstraction
        self.prefer = prefer        # a counterexample satisfying this is preferred (it can be run)


def _and(conjuncts):
    if not conjuncts:
        return z3.BoolVal(True)
    if len(conjuncts) == 1:
        return conjuncts[0]
    return z3.And(*conjuncts)


def _tdiv(a, b):
    """Integer division truncating toward zero (C, both TIL VMs) on Z3 Ints
    (SMT-LIB `div` is Euclidean)."""
    return z3.If(a >= 0, z3.If(b >= 0, a / b, -(a / -b)),
                 z3.If(b >= 0, -((-a) / b), (-a) / (-b)))


def _free_consts(exprs) -> Set[str]:
    """Names of the uninterpreted constants occurring in z3 expressions."""
    out: Set[str] = set()
    seen: Set[int] = set()
    stack = list(exprs)
    while stack:
        e = stack.pop()
        k = e.get_id()
        if k in seen:
            continue
        seen.add(k)
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            out.add(e.decl().name())
        else:
            stack.extend(e.children())
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#                       INTEGER MODELS (bit-vector / integer)
# ═══════════════════════════════════════════════════════════════════════════════

class _BVDomain:
    """TIL ints as 64-bit bit-vectors: exactly the VM semantics."""
    name = "bit-vector"

    def __init__(self):
        self.nonlinear = False      # products/quotients of two variables seen

    def const(self, v: int):
        return z3.BitVecVal(v, 64)

    def fresh(self, name: str):
        return z3.BitVec(name, 64)

    def range(self, z):
        return None

    def is_value(self, z) -> bool:
        return z3.is_bv_value(z)

    def value(self, z) -> int:
        return z.as_signed_long()

    def _nl(self, a, b):
        if not (z3.is_bv_value(a) or z3.is_bv_value(b)):
            self.nonlinear = True

    def add(self, enc, st, a, b, node):
        return a + b

    def sub(self, enc, st, a, b, node):
        return a - b

    def mul(self, enc, st, a, b, node):
        self._nl(a, b)
        return a * b

    def neg(self, enc, st, a, node):
        return -a

    def div(self, enc, st, a, b, node):
        self._nl(a, b)
        return a / b                       # bvsdiv: truncates, MIN / -1 == MIN

    def rem(self, enc, st, a, b, node):
        self._nl(a, b)
        return z3.SRem(a, b)               # sign of the dividend, MIN % -1 == 0

    def band(self, enc, st, a, b, node):
        return a & b

    def bor(self, enc, st, a, b, node):
        return a | b

    def bxor(self, enc, st, a, b, node):
        return a ^ b

    def bnot(self, enc, st, a, node):
        return ~a

    def shl(self, enc, st, a, b, node):
        return a << (b & 63)               # both VMs mask the shift amount

    def shr(self, enc, st, a, b, node):
        return a >> (b & 63)               # arithmetic shift

    def abs(self, enc, st, a, node):
        return z3.If(a < 0, -a, a)         # abs(MIN) == MIN

    def pow(self, enc, st, a, b, node):
        one = self.const(1)
        if z3.is_bv_value(b):
            e = b.as_signed_long()
            if e < 0:
                return z3.If(a == 1, one, z3.If(a == -1, self.const(-1 if e & 1 else 1), self.const(0)))
            r, base = one, a
            while e:
                if e & 1:
                    r = self.mul(enc, st, r, base, node)
                e >>= 1
                if e:
                    base = self.mul(enc, st, base, base, node)
            return r
        self.nonlinear = True
        r, base = one, a
        for i in range(63):                # square and multiply over the exponent bits
            r = z3.If(z3.Extract(i, i, b) == 1, r * base, r)
            base = base * base
        neg = z3.If(a == 1, one, z3.If(a == -1, z3.If(z3.Extract(0, 0, b) == 1, self.const(-1), one),
                                       self.const(0)))
        return z3.If(b < 0, neg, r)


class _IntDomain:
    """TIL ints as mathematical integers, valid only when no operation
    overflows: +, -, * and ** add an `overflow` obligation (the result fits in
    64 bits); unary -, abs and / overflow only for MIN (-MIN, abs(MIN) and
    MIN / -1 are MIN in both VMs), which is modeled exactly.  When all
    overflow obligations are proved the function behaves exactly like with
    wrap-around, and nonlinear arithmetic (loop invariants such as
    2 * s == i * (i + 1)) becomes easy for Z3."""
    name = "integer"
    nonlinear = False

    def const(self, v: int):
        return z3.IntVal(v)

    def fresh(self, name: str):
        return z3.Int(name)

    def range(self, z):
        return z3.And(z >= INT_MIN, z <= INT_MAX)

    def is_value(self, z) -> bool:
        return z3.is_int_value(z)

    def value(self, z) -> int:
        return z.as_long()

    def add(self, enc, st, a, b, node):
        return enc.overflow(st, a + b, node)

    def sub(self, enc, st, a, b, node):
        return enc.overflow(st, a - b, node)

    def mul(self, enc, st, a, b, node):
        return enc.overflow(st, a * b, node)

    def neg(self, enc, st, a, node):
        return z3.If(a == INT_MIN, a, -a)

    def div(self, enc, st, a, b, node):
        return z3.If(z3.And(a == INT_MIN, b == -1), a, _tdiv(a, b))

    def rem(self, enc, st, a, b, node):
        return a - b * _tdiv(a, b)         # always fits; MIN % -1 == 0 like the VMs

    def _bits(self, *_args):
        raise _NotTranslatable("bit operations")

    band = bor = bxor = shl = shr = _bits

    def bnot(self, enc, st, a, node):
        raise _NotTranslatable("bit operations")

    def abs(self, enc, st, a, node):
        return z3.If(a == INT_MIN, a, z3.If(a < 0, -a, a))

    def pow(self, enc, st, a, b, node):
        if not z3.is_int_value(b):
            raise _NotTranslatable("'**' with a variable exponent")
        e = b.as_long()
        if e < 0:
            return z3.If(a == 1, 1, z3.If(a == -1, -1 if e & 1 else 1, 0))
        if e > 64:                          # only -1, 0, 1 have such powers in range
            enc.overflow_cond(st, z3.And(a >= -1, a <= 1), node)
            return z3.If(a == 0, 0, z3.If(a == 1, 1, -1 if e & 1 else 1))
        r = self.const(1)
        for _ in range(e):
            r = r * a
        # |a ** k| <= |a ** e| for every k <= e: one condition covers the steps
        return enc.overflow(st, r, node)


# ═══════════════════════════════════════════════════════════════════════════════
#                     ENCODER (symbolic execution of one function)
# ═══════════════════════════════════════════════════════════════════════════════

_NO_FOLD = object()
_NARROW_INTS = ('i8', 'i16', 'i32', 'u8', 'u16', 'u32', 'u64', 'uint')
_FLOATS = ('float', 'f32', 'f64')


def _type_kind(t) -> Optional[str]:
    """'int' / 'bool' for the types the prover models exactly, else None."""
    if isinstance(t, til.PrimitiveType):
        if t.name in ('int', 'i64'):
            return 'int'
        if t.name == 'bool':
            return 'bool'
    return None


def _body_stmts(body) -> list:
    return list(body.statements) if isinstance(body, til.Block) else [body]


def _split_invariants(stmts: list) -> Tuple[list, list]:
    """Leading `invariant` statements of a loop body are its loop invariants."""
    k = 0
    while k < len(stmts) and isinstance(stmts[k], til.InvariantStmt):
        k += 1
    return stmts[:k], stmts[k:]


def _assigned_names(stmts) -> Set[str]:
    names: Set[str] = set()
    for s in stmts:
        for n in til.walk(s):
            if isinstance(n, til.Assignment) and isinstance(n.target, til.Identifier):
                names.add(n.target.name)
    return names


def _conjuncts(expr) -> list:
    """Top-level operands of `a and b and c` (evaluated left to right)."""
    if isinstance(expr, til.BinaryOp) and expr.op == 'and':
        return _conjuncts(expr.left) + _conjuncts(expr.right)
    return [expr]


def _names_in(expr) -> List[str]:
    """Identifiers read by an expression (not counting called function names)."""
    callees = set()
    out: List[str] = []
    for n in til.walk(expr):
        if isinstance(n, til.Call) and isinstance(n.func, til.Identifier):
            callees.add(id(n.func))
    for n in til.walk(expr):
        if isinstance(n, til.Identifier) and id(n) not in callees and n.name not in out:
            out.append(n.name)
    return out


class _Encoder:
    """Symbolic execution of one function: produces the obligations (path
    condition => goal) whose validity means the function is correct."""

    def __init__(self, prover: 'Prover', key: str, fdef, dom):
        self.p = prover
        self.key = key
        self.fdef = fdef
        self.dom = dom
        self.obligations: List[Obligation] = []
        self.inputs: List[Tuple[str, str, Any, Any]] = []     # (name, sort, z3 term, declared type)
        self.entry_pc: tuple = ()
        self.loops: List[_LoopCtx] = []
        self.mode = 'check'          # check: conditions are obligations | assume: they are assumed
        self.ovf = 'check'           # integer encoding, overflow conditions: check | collect | drop
        self.collected: list = []    # overflow conditions gathered in 'collect' mode
        self.strict = False          # evaluating a contract / assert: opaque values are unsupported
        self.result: Optional[_V] = None
        self.sites: list = []        # call nodes while inside callee contracts / inlined callees
        self.inline_stack: List[Tuple[str, Any]] = []
        self.attr: Optional[Tuple[str, int]] = None
        self.suffix = ""
        self.frames: Optional[list] = None
        self.unmodeled = 0
        self.loop_depth = 0          # > 0: obligations depend on a loop abstraction
        self.after_loop = False
        self.n_fresh = 0
        self.fresh_desc: Dict[str, Tuple[str, str]] = {}   # fresh constant -> (kind, description)
        self.contracts: List[str] = []   # callees whose contract is being evaluated (recursion guard)

    # ── context ─────────────────────────────────────────────────────────────
    @contextmanager
    def context(self, **kw):
        saved = {k: getattr(self, k) for k in kw}
        for k, v in kw.items():
            setattr(self, k, v)
        try:
            yield
        except Unsupported as u:
            if u.loc is None:
                u.loc = self.location(u.node)
            raise
        finally:
            for k, v in saved.items():
                setattr(self, k, v)

    def location(self, node) -> Tuple[int, int]:
        """Where to report a condition: the function's own #[requires] /
        #[ensures] line, else the outermost call site in the body (callee
        contracts, inlined callees), else the node itself."""
        if self.attr is not None:
            return self.p.attr_location(self.fdef, self.attr)
        if self.sites:
            site = self.sites[0]
            target = site.func if isinstance(site, til.Call) and getattr(site.func, 'line', 0) else site
            return target.line or self.fdef.line, target.col or 1
        line = getattr(node, 'line', 0) or self.fdef.line
        return line, getattr(node, 'col', 0) or 1

    def cur_func(self) -> str:
        return self.inline_stack[-1][0] if self.inline_stack else self.key

    def frames_here(self, line: int) -> list:
        """Expected VM frames (function, line), innermost first, for a runtime
        error at `line` of the function being encoded (or an inlined callee)."""
        if self.frames is not None:
            return list(self.frames)
        out = [(self.cur_func(), line)]
        for i in range(len(self.inline_stack) - 1, -1, -1):
            caller = self.inline_stack[i - 1][0] if i > 0 else self.key
            out.append((caller, self.inline_stack[i][1].line))
        return out

    # ── obligations ─────────────────────────────────────────────────────────
    @staticmethod
    def dead(st: _State) -> bool:
        return bool(st.pc) and z3.is_false(st.pc[-1])

    def add_ob(self, st: _State, kind: str, goal, node, what: str, expect=None, show=(), prefer=None):
        """Record `st.pc => goal` (in check mode) and continue assuming goal:
        the VM stops the program when the check fails."""
        if z3.is_true(goal):
            return
        if self.mode == 'check':
            if not self.dead(st):
                line, col = self.location(node)
                self.obligations.append(Obligation(
                    kind, what, line, col, st.pc, goal, expect, list(show), self.unmodeled,
                    self.loop_depth > 0 or self.after_loop, prefer))
        st.pc = st.pc + (goal,)

    def overflow(self, st: _State, term, node):
        self.overflow_cond(st, z3.And(term >= INT_MIN, term <= INT_MAX), node)
        return term

    def overflow_cond(self, st: _State, cond, node):
        """Integer encoding: `cond` says an operation does not overflow.

        check:   an obligation, not assumed afterwards: the integer results are
                 used only when every overflow obligation of the function is
                 proved, and then each one already follows from its own path;
        collect: an assumed contract is evaluated: `assume_conjuncts` assumes
                 `no overflow => contract` (the VM checked it with wrap-around);
        drop:    a loop invariant at the loop head: every real head state
                 satisfies it in integer arithmetic too, because entry and
                 preservation checked its evaluation for overflow."""
        if self.ovf == 'drop' or z3.is_true(z3.simplify(cond)) or self.dead(st):
            return
        if self.ovf == 'collect':
            self.collected.append(cond)
            return
        line, col = self.location(node)
        self.obligations.append(Obligation('overflow', "integer overflow", line, col, st.pc, cond, None, [],
                                           self.unmodeled, self.loop_depth > 0))

    # ── values ──────────────────────────────────────────────────────────────
    def fresh(self, st: _State, sort: str, base: str, what: str = "",
              desc: Optional[Tuple[str, str]] = None) -> _V:
        """A new unknown value (loop heads, call results). `desc` = (kind,
        text) says where it comes from, for messages about unproved conditions."""
        self.n_fresh += 1
        name = f"{base}@{self.n_fresh}"
        if desc is not None:
            self.fresh_desc[name] = desc
        if sort == INT:
            z = self.dom.fresh(name)
            rng = self.dom.range(z)
            if rng is not None:
                st.pc = st.pc + (rng,)
            return _V(INT, z)
        if sort == BOOL:
            return _V(BOOL, z3.Bool(name))
        if sort == NIL:
            return _NIL
        return _opaque(what or "an unknown value")

    def int_const(self, v: int) -> _V:
        return _V(INT, self.dom.const(til.wrap_int64(v)))

    def bool_const(self, b: bool) -> _V:
        return _V(BOOL, z3.BoolVal(bool(b)))

    def const_value(self, v) -> _V:
        if isinstance(v, bool):
            return self.bool_const(v)
        if isinstance(v, int):
            return self.int_const(v)
        if v is None:
            return _NIL
        if isinstance(v, float):
            return _opaque("a float")
        if isinstance(v, str):
            return _opaque("a string")
        return _opaque("a constant")

    def truth(self, v: _V, node):
        """Truthiness like the VMs (JUMP_IF_FALSE / not / and / or)."""
        if v.sort == BOOL:
            return v.z
        if v.sort == INT:
            return v.z != 0
        if v.sort == NIL:
            return z3.BoolVal(False)
        if self.strict:
            raise Unsupported(f"the condition depends on {v.what}, which the prover does not model", node)
        # a branch on something unmodeled: both ways are possible (sound)
        self.unmodeled += 1
        self.n_fresh += 1
        name = f"branch@{self.n_fresh}"
        self.fresh_desc[name] = ('opaque', f"a condition on {v.what} (line {getattr(node, 'line', 0)})")
        return z3.Bool(name)

    def cast_int(self, v: _V) -> _V:
        """CAST int (int(x), `-> int` returns, `x: int` parameters)."""
        if v.sort == INT:
            return v
        if v.sort == BOOL:
            return _V(INT, z3.If(v.z, self.dom.const(1), self.dom.const(0)))
        if v.sort == NIL:
            return self.int_const(0)
        self.unmodeled += 1
        return _opaque(v.what)

    def convert_annotated(self, v: _V, t, what: str) -> _V:
        """Value stored under a type annotation (parameter, let x: T, return)."""
        if isinstance(t, til.PrimitiveType):
            if t.name in ('int', 'i64'):
                return self.cast_int(v)
            if t.name in _NARROW_INTS:
                return _opaque(f"{what} of type {t.name} (narrower than 64 bits in native code)")
            if t.name in _FLOATS:
                return v if v.sort == OPAQUE else _opaque(f"{what} of type float")
        return v

    # ── function ────────────────────────────────────────────────────────────
    def encode(self) -> List[Obligation]:
        fdef = self.fdef
        st = _State((), [{}])
        for p in fdef.params:
            if p.name == 'self':
                raise Unsupported("methods with 'self' are not supported (the receiver is a struct)", fdef)
            st.declare(p.name, self.param_value(p, st))
        for k, cond in enumerate(fdef.requires):
            text = _text(fdef.requires_text, k)
            # assumed (the VM checks it at entry); its own divisions must not fail
            with self.context(attr=('requires', k), suffix=f" in #[requires: {text}]",
                              frames=[(self.key, 0)], strict=True):
                self.assume_conjuncts(cond, st)
        self.entry_pc = st.pc
        end = self.exec_block(fdef.body, st)
        if end is not None and not self.dead(end):
            self.fall_off(end)
        return self.obligations

    def assume_conjuncts(self, cond, st: _State):
        """Assume a condition the VM has checked (#[requires] at entry, a
        callee's #[ensures] after the call), one top-level `and` operand at a
        time.  In the integer encoding each operand is assumed only when its
        evaluation does not overflow: `no overflow => operand`."""
        for part in _conjuncts(cond):
            bucket: list = []
            with self.context(ovf='collect', collected=bucket):
                t = self.truth(self.ev(part, st), part)
            st.pc = st.pc + ((z3.Implies(_and(bucket), t) if bucket else t),)

    def param_value(self, p, st: _State) -> _V:
        kind = _type_kind(p.type)
        if kind == 'int':
            z = self.dom.fresh(p.name)
            rng = self.dom.range(z)
            if rng is not None:
                st.pc = st.pc + (rng,)
            v = _V(INT, z)
        elif kind == 'bool':
            v = _V(BOOL, z3.Bool(p.name))
        elif isinstance(p.type, til.UnknownType):
            v = _opaque(f"parameter '{p.name}' (it has no type; annotate it as int or bool)")
        else:
            v = _opaque(f"parameter '{p.name}' of type {p.type}")
        self.inputs.append((p.name, v.sort, v.z, p.type))
        return v

    def ret_value(self, v: _V) -> _V:
        """The returned value as the caller sees it (the VM casts `-> int`)."""
        return self.convert_annotated(v, self.fdef.ret_type, "a result")

    def fall_off(self, st: _State):
        """End of the body without `return`: the VM returns nil, cast to the
        return type; #[ensures] is checked unless the function returns nothing."""
        if isinstance(self.fdef.ret_type, til.VoidType):
            return
        self.check_ensures(self.ret_value(_NIL), st, self.fdef)

    def check_ensures(self, v: _V, st: _State, node):
        fdef = self.fdef
        for k, cond in enumerate(fdef.ensures):
            text = _text(fdef.ensures_text, k)
            with self.context(attr=('ensures', k), suffix=f" in #[ensures: {text}]",
                              frames=[(self.key, 0)], result=v, strict=True):
                t = self.truth(self.ev(cond, st), cond)
                show = [('result', v)] + self.show_names(cond, st, skip_inputs=True)
                msg = f"contract violation: {fdef.name} ensures {text}".strip()
                self.add_ob(st, 'ensures', t, cond, f"#[ensures: {text}]", (msg, [(self.key, 0)]), show)

    def show_names(self, expr, st: _State, skip_inputs: bool = False) -> list:
        """(name, value) of the int/bool variables an expression reads (inputs
        still holding their initial value are already shown with the inputs)."""
        out = []
        initial = {name: z for name, _s, z, _t in self.inputs} if skip_inputs else {}
        for name in _names_in(expr):
            if name == 'result':
                continue
            v = st.lookup(name)
            if v is None or v.sort not in (INT, BOOL):
                continue
            if name in initial:
                if v.z is initial[name]:
                    continue
                name = f"{name} (when it returns)"
            out.append((name, v))
        return out

    # ── statements ──────────────────────────────────────────────────────────
    def exec_block(self, body, st: _State) -> Optional[_State]:
        st.scopes.append({})
        for s in _body_stmts(body):
            st = self.exec_stmt(s, st)
            if st is None or self.dead(st):
                return None
        st.scopes.pop()
        return st

    def exec_stmt(self, node, st: _State) -> Optional[_State]:
        T = til
        t = type(node)
        if t is T.ExprStmt:
            self.ev(node.expr, st)
            return st
        if t is T.VarDecl:
            if node.value is None:
                v = self.zero_value(node.type_ann)
            else:
                v = self.convert_annotated(self.ev(node.value, st), node.type_ann, f"variable '{node.name}'")
            st.declare(node.name, v)
            return st
        if t is T.Assignment:
            return self.exec_assign(node, st)
        if t is T.If:
            return self.exec_if(node, st)
        if t is T.While:
            return self.exec_while(node, st, infinite=False)
        if t is T.Loop:
            return self.exec_while(node, st, infinite=True)
        if t is T.For:
            return self.exec_for(node, st)
        if t is T.Break or t is T.Continue:
            if not self.loops:
                raise Unsupported("'break'/'continue' outside of a loop", node)
            ctx = self.loops[-1]
            st.scopes = st.scopes[:ctx.depth]
            (ctx.breaks if t is T.Break else ctx.continues).append(st)
            return None
        if t is T.Return:
            self.exec_return(node, st)
            return None
        if t is T.AssertStmt or t is T.InvariantStmt:
            self.exec_check(node, st)
            return st
        if t is T.GlobalInit:
            self.ev(node.decl.value, st)
            return st
        if t is T.Block:
            return self.exec_block(node, st)
        if t in (T.TypeAlias, T.Import, T.CapabilityType):
            return st
        if t is T.MatchExpr:
            raise Unsupported("'match' is not supported", node)
        if t in (T.FuncDef, T.StructDef, T.EnumDef, T.ImplBlock, T.TraitDef):
            raise Unsupported("nested definitions are not supported", node)
        self.ev(node, st)
        return st

    def zero_value(self, type_ann) -> _V:
        if isinstance(type_ann, til.PrimitiveType):
            name = type_ann.name
            if name == 'bool':
                return self.bool_const(False)
            if name in ('int', 'i64', 'char'):
                return self.int_const(0)
            if name in _NARROW_INTS:
                return _opaque(f"a value of type {name}")
            if name in _FLOATS:
                return _opaque("a float")
            if name == 'str':
                return _opaque("a string")
            return self.int_const(0)
        return _NIL

    def exec_assign(self, node, st: _State) -> _State:
        tgt = node.target
        if isinstance(tgt, til.Identifier):
            local = st.lookup(tgt.name)
            if node.op == '=':
                v = self.ev(node.value, st)
            else:
                cur = local if local is not None else self.ev_name(tgt, st)
                v = self.binop(node.op[:-1], cur, self.ev(node.value, st), node, st)
            if local is not None:
                st.assign(tgt.name, v)
            # globals are not tracked: reading a mutable global is opaque
            return st
        if isinstance(tgt, til.Index):
            self.ev(tgt.obj, st)
            self.ev(tgt.index, st)
            if node.op != '=':
                self.binop(node.op[:-1], _opaque("an element"), self.ev(node.value, st), node, st)
            else:
                self.ev(node.value, st)
            self.unmodeled += 1          # index errors are not modeled
            return st
        if isinstance(tgt, til.Attribute):
            self.ev(tgt.obj, st)
            if node.op != '=':
                self.binop(node.op[:-1], _opaque("a field"), self.ev(node.value, st), node, st)
            else:
                self.ev(node.value, st)
            self.unmodeled += 1
            return st
        raise Unsupported("this assignment target is not supported", node)

    def exec_if(self, node, st: _State) -> Optional[_State]:
        results = []
        cur = st
        for cond, body in [(node.condition, node.then_body)] + list(node.elifs):
            c = self.truth(self.ev(cond, cur), cond)
            if not z3.is_false(c):
                branch = cur.copy()
                branch.pc = cur.pc + (c,)
                results.append(self.exec_block(body, branch))
            if z3.is_true(c):
                cur = None
                break
            cur.pc = cur.pc + (z3.Not(c),)
        if cur is not None:
            results.append(self.exec_block(node.else_body, cur) if node.else_body is not None else cur)
        return self.merge(results)

    def exec_return(self, node, st: _State):
        v = self.ev(node.value, st) if node.value is not None else _NIL
        if node.value is None and isinstance(self.fdef.ret_type, til.VoidType):
            return                       # RETURN_NIL: the VM checks no #[ensures]
        self.check_ensures(self.ret_value(v), st, node)

    def exec_check(self, node, st: _State):
        is_assert = isinstance(node, til.AssertStmt)
        with self.context(strict=True):
            t = self.truth(self.ev(node.condition, st), node.condition)
        msg = ("Assertion failed" if is_assert else "Invariant violated") + \
            (f": {node.message}" if node.message else "")
        self.add_ob(st, 'assert' if is_assert else 'invariant', t, node, self.p.stmt_text(node),
                    (msg, self.frames_here(node.line)), self.show_names(node.condition, st))

    # ── loops ───────────────────────────────────────────────────────────────
    def exec_while(self, node, st: _State, infinite: bool) -> Optional[_State]:
        invs, rest = _split_invariants(_body_stmts(node.body))

        def cond_fn(h: _State):
            if infinite:
                return z3.BoolVal(True)
            return self.truth(self.ev(node.condition, h), node.condition)
        return self.run_loop(node, st, rest, cond_fn, invs, _assigned_names(rest), None, [])

    def exec_for(self, node, st: _State) -> Optional[_State]:
        invs, rest = _split_invariants(_body_stmts(node.body))
        assigned = _assigned_names(rest)
        var = node.var
        it = node.iter
        st.scopes.append({})
        rng = self.range_call(it, st)
        if isinstance(it, til.Range):
            # `for i in a..b`: i itself is the counter (i = a; while i < b: body; i += 1)
            a = self.ev(it.start, st)
            st.declare(var, a)
            b = self.ev(it.end, st)
            op = '<=' if it.inclusive else '<'

            def cond_fn(h: _State):
                return self.truth(self.binop(op, h.lookup(var), b, node, h), node)

            def step_fn(s: _State):
                s.assign(var, self.binop('+', s.lookup(var), self.int_const(1), node, s))
                return s
            implicit = []
            if var not in assigned and a.sort == INT and b.sort == INT:
                implicit.append(self.range_bound(var, a, b, it.inclusive))
            res = self.run_loop(node, st, rest, cond_fn, invs, assigned | {var}, step_fn, implicit)
        elif rng is not None:
            # `for i in range(a, b)`: a hidden counter; i is re-bound each iteration
            a, b = rng
            counter = f" k{node.line}:{node.col}"
            st.declare(counter, a)
            st.declare(var, a)

            def cond_fn(h: _State):
                return self.truth(self.binop('<', h.lookup(counter), b, node, h), node)

            def step_fn(s: _State):
                k = self.binop('+', s.lookup(counter), self.int_const(1), node, s)
                s.assign(counter, k)
                s.assign(var, k)
                return s

            def head_fix(h: _State):
                h.assign(var, h.lookup(counter))
            res = self.run_loop(node, st, rest, cond_fn, invs, assigned | {counter, var}, step_fn,
                                [self.range_bound(counter, a, b, False)], head_fix)
        else:
            # `for x in <array/string/map>`: any number of iterations, opaque elements
            self.ev(it, st)
            self.unmodeled += 1
            elem = _opaque(f"an element of the loop at line {node.line}")
            st.declare(var, elem)

            def cond_fn(h: _State):
                self.n_fresh += 1
                name = f"more@{self.n_fresh}"
                self.fresh_desc[name] = ('loop', f"the number of iterations of the loop at line {node.line}")
                return z3.Bool(name)

            def step_fn(s: _State):
                s.assign(var, elem)
                return s
            res = self.run_loop(node, st, rest, cond_fn, invs, assigned, step_fn, [])
        if res is not None:
            res.scopes.pop()
        return res

    def range_call(self, it, st: _State):
        """(start, end) of `range(n)` / `range(a, b)` over ints, else None."""
        if not (isinstance(it, til.Call) and isinstance(it.func, til.Identifier)
                and it.func.name == 'range' and 1 <= len(it.args) <= 2):
            return None
        if st.lookup('range') is not None or 'range' in self.p.funcs:
            return None
        args = [self.ev(a, st) for a in it.args]
        if any(a.sort != INT for a in args):
            raise Unsupported("range() over values that are not ints", it)
        return (self.int_const(0), args[0]) if len(args) == 1 else (args[0], args[1])

    def range_bound(self, var: str, a: _V, b: _V, inclusive: bool):
        """Implicit invariant of a counted loop whose counter the body never
        assigns: a <= i <= max(a, b) (exclusive), a <= i <= max(a, b + 1)
        (inclusive; no bound when b == MAX: the loop never ends then)."""
        dom = self.dom

        def bound(s: _State):
            i = s.lookup(var).z
            if not inclusive:
                return z3.And(a.z <= i, i <= z3.If(a.z > b.z, a.z, b.z))
            return z3.Or(b.z == dom.const(INT_MAX),
                          z3.And(a.z <= i, i <= z3.If(a.z > b.z, a.z, b.z + 1)))
        return bound

    def run_loop(self, node, st: _State, body: list, cond_fn, invs: list, modified: Set[str],
                 step_fn, implicit: list, head_fix=None) -> Optional[_State]:
        """The loop rule: invariants hold on entry; the state at the loop head
        is any state satisfying them (modified variables havocked); one
        iteration from there re-establishes them; after the loop they hold and
        the condition is false.  Without invariants this is a sound
        over-approximation (the modified variables are unknown)."""
        prefer = None
        if invs:
            # the VM checks the invariants only when the body runs: prefer
            # counterexamples to the entry condition where it does
            saved = (self.unmodeled, len(self.obligations))
            try:
                with self.context(mode='assume', ovf='drop', strict=False):
                    prefer = cond_fn(st.copy())
            except Unsupported:
                prefer = None
            self.unmodeled = saved[0]
            del self.obligations[saved[1]:]
        for inv in invs:
            self.check_loop_inv(inv, st, 'entry', prefer=prefer)
        for bound in implicit:
            self.add_ob(st, 'range', bound(st), node, "the bounds of the for loop")
        h = st.copy()
        head: Dict[str, _V] = {}
        for name in sorted(modified):
            old = h.lookup(name)
            if old is None:
                continue                 # declared inside the body
            shown = "the counter" if name.startswith(' ') else f"'{name}'"
            new = self.fresh(h, old.sort, name.strip().replace(' ', '_') or 'v',
                             old.what if old.sort == OPAQUE else "",
                             ('loop', f"{shown} at the start of an iteration of the loop at line {node.line}"))
            h.assign(name, new)
            head[name] = new
        if head_fix is not None:
            head_fix(h)
        for bound in implicit:
            h.pc = h.pc + (bound(h),)
        for inv in invs:
            with self.context(mode='assume', ovf='drop', strict=True):
                t = self.truth(self.ev(inv.condition, h), inv.condition)
            h.pc = h.pc + (t,)
        self.loop_depth += 1
        try:
            c = cond_fn(h)
            body_st = h.copy()
            body_st.pc = h.pc + (c,)
            ctx = _LoopCtx(len(body_st.scopes))
            self.loops.append(ctx)
            try:
                end = self.exec_block(til.Block(statements=body), body_st) if not z3.is_false(c) else None
            finally:
                self.loops.pop()
            for back in [end] + ctx.continues:
                if back is None or self.dead(back):
                    continue
                if step_fn is not None:
                    back = step_fn(back)
                self.check_sorts(back, head, node)
                for bound in implicit:
                    self.add_ob(back, 'range', bound(back), node, "the bounds of the for loop")
                for inv in invs:
                    self.check_loop_inv(inv, back, 'step', head)
            for brk in ctx.breaks:
                self.check_sorts(brk, head, node)
        finally:
            self.loop_depth -= 1
        self.after_loop = True
        h.pc = h.pc + (z3.Not(c),)
        return self.merge([h] + ctx.breaks)

    def check_loop_inv(self, inv, st: _State, phase: str, head: Optional[Dict[str, _V]] = None, prefer=None):
        with self.context(strict=True):
            t = self.truth(self.ev(inv.condition, st), inv.condition)
        text = self.p.stmt_text(inv, keyword=False)
        msg = "Invariant violated" + (f": {inv.message}" if inv.message else "")
        if phase == 'entry':
            what = f"loop invariant `{text}` when the loop is reached"
            show = self.show_names(inv.condition, st)
        else:
            what = f"loop invariant `{text}` after an iteration"
            show = [(f"{n} (iteration start)", head[n]) for n in _names_in(inv.condition)
                    if head and n in head and head[n].sort in (INT, BOOL)]
            show += [(f"{n} (iteration end)", v) for n, v in self.show_names(inv.condition, st)]
        self.add_ob(st.copy() if phase == 'step' else st, 'loop-entry' if phase == 'entry' else 'loop-step',
                    t, inv, what, (msg, self.frames_here(inv.line)), show, prefer)

    def check_sorts(self, st: _State, head: Dict[str, _V], node):
        for name, hv in head.items():
            v = st.lookup(name)
            if v is not None and v.sort != hv.sort and not (v.sort == OPAQUE and hv.sort == OPAQUE):
                raise Unsupported(f"variable '{name.strip()}' changes its type inside the loop", node)

    # ── merging paths ───────────────────────────────────────────────────────
    def merge(self, states: list) -> Optional[_State]:
        live = [s for s in states if s is not None and not self.dead(s)]
        if not live:
            return None
        if len(live) == 1:
            return live[0]
        n = min(len(s.pc) for s in live)
        k = 0
        while k < n and all(s.pc[k] is live[0].pc[k] for s in live):
            k += 1
        guards = [_and(s.pc[k:]) for s in live]
        pc = live[0].pc[:k] + (z3.Or(*guards),)
        scopes = []
        for level in range(min(len(s.scopes) for s in live)):
            scope = {}
            for name in live[0].scopes[level]:
                vals = [s.scopes[level].get(name) for s in live]
                if any(v is None for v in vals):
                    continue
                scope[name] = self.merge_values(name, vals, guards)
            scopes.append(scope)
        return _State(pc, scopes)

    @staticmethod
    def merge_values(name: str, vals: List[_V], guards: list) -> _V:
        first = vals[0]
        if all(v is first for v in vals):
            return first
        sorts = {v.sort for v in vals}
        if len(sorts) > 1:
            return _opaque(f"variable '{name}' (it holds different types on different paths)")
        if first.sort == NIL:
            return _NIL
        if first.sort == OPAQUE:
            return first
        z = vals[-1].z
        for v, g in zip(reversed(vals[:-1]), reversed(guards[:-1])):
            z = z3.If(g, v.z, z)
        return _V(first.sort, z)

    # ── expressions ─────────────────────────────────────────────────────────
    def ev(self, node, st: _State) -> _V:
        T = til
        t = type(node)
        if t is T.BinaryOp or t is T.UnaryOp:
            folded = self.p.fold(node)
            if folded is not _NO_FOLD:
                return self.const_value(folded)
            return self.ev_binary(node, st) if t is T.BinaryOp else self.ev_unary(node, st)
        if t is T.IntLit:
            return self.int_const(node.value)
        if t is T.BoolLit:
            return self.bool_const(node.value)
        if t is T.CharLit:
            return self.int_const(ord(node.value))
        if t is T.Identifier:
            return self.ev_name(node, st)
        if t is T.Call:
            return self.ev_call(node, st)
        if t is T.StringLit:
            return _opaque("a string")
        if t is T.FloatLit:
            return _opaque("a float")
        if t is T.FStringLit:
            for part in node.parts:
                if part[0] == 'expr':
                    self.ev(part[1], st)
            return _opaque("a string")
        if t is T.IfExpr:
            return self.ev_ifexpr(node, st)
        if t is T.Cast:
            return self.ev_cast(node, st)
        if t is T.DurationLit:
            mult = {"ms": 1, "s": 1000, "m": 60000, "h": 3600000}.get(node.unit, 1)
            return self.int_const(int(node.value * mult))
        if t is T.ArrayLit or t is T.TupleLit:
            for e in node.elements:
                self.ev(e, st)
            return _opaque("an array")
        if t is T.DictLit:
            for k, v in node.pairs:
                self.ev(k, st)
                self.ev(v, st)
            return _opaque("a map")
        if t is T.StructInit:
            for v in node.fields.values():
                self.ev(v, st)
            return _opaque(f"a {node.name} value")
        if t is T.Index:
            self.ev(node.obj, st)
            self.ev(node.index, st)
            self.unmodeled += 1
            return _opaque("an element of an array, string or map")
        if t is T.Attribute:
            return self.ev_attribute(node, st)
        if t is T.Range:
            self.ev(node.start, st)
            self.ev(node.end, st)
            return _opaque("a range")
        names = {
            'Lambda': "closures (lambdas)", 'MatchExpr': "'match'", 'NullCheck': "'?' (Option/Result)",
            'RetryExpr': "retry blocks", 'TimeoutExpr': "timeout blocks", 'IntentBlock': "'intent' blocks",
            'ListComprehension': "list comprehensions", 'EffectHandler': "effect handlers",
            'PerformEffect': "'perform'", 'Block': "block expressions", 'SendExpr': "channels",
            'RecvExpr': "channels", 'FallbackExpr': "'fallback'",
        }
        raise Unsupported(f"{names.get(t.__name__, t.__name__)} are not supported"
                          if t.__name__ in names else f"{t.__name__} is not supported", node)

    def ev_name(self, node, st: _State) -> _V:
        name = node.name
        if name == 'NULL':
            return _NIL
        if name == 'result' and self.result is not None:
            return self.result
        v = st.lookup(name)
        if v is not None:
            return v
        g = self.p.global_value(name)
        if g is not None:
            return g if isinstance(g, _V) else self.const_value(g[0])
        if name in self.p.funcs and '.' not in name:
            if self.p.funcs[name].requires:
                raise Unsupported(f"function '{name}' (which has #[requires]) is used as a value; "
                                  f"calls through function values are not checked", node)
            return _opaque(f"the function value '{name}'")
        if name in self.p.struct_names or name in self.p.enum_values or name in self.p.adt_names:
            return _opaque(f"the type '{name}'")
        raise Unsupported(f"unknown name '{name}'", node)

    def ev_attribute(self, node, st: _State) -> _V:
        obj = node.obj
        if isinstance(obj, til.Identifier) and st.lookup(obj.name) is None:
            values = self.p.enum_values.get(obj.name)
            if values is not None and node.attr in values:
                return self.int_const(values[node.attr])
            if obj.name in self.p.adt_names:
                return _opaque(f"{obj.name}.{node.attr}")
        self.ev(obj, st)
        self.unmodeled += 1
        return _opaque(f"the field '{node.attr}'")

    def ev_ifexpr(self, node, st: _State) -> _V:
        c = self.truth(self.ev(node.condition, st), node.condition)
        ts = _State(st.pc + (c,), st.scopes)
        tv = self.ev(node.then_expr, ts)
        es = _State(st.pc + (z3.Not(c),), st.scopes)
        fv = self.ev(node.else_expr, es) if node.else_expr is not None else _NIL
        base = len(st.pc) + 1
        for cond, sub in ((c, ts), (z3.Not(c), es)):
            extra = sub.pc[base:]
            if extra:
                st.pc = st.pc + (z3.Implies(cond, _and(extra)),)
        if tv.sort != fv.sort:
            return _opaque("a conditional expression with values of different types")
        if tv.sort in (INT, BOOL):
            return _V(tv.sort, z3.If(c, tv.z, fv.z))
        return tv if tv.sort == OPAQUE else _NIL

    def ev_cast(self, node, st: _State) -> _V:
        v = self.ev(node.expr, st)
        target = node.target_type
        if not isinstance(target, til.PrimitiveType):
            return v                      # the VM emits no cast
        name = target.name
        if name in _FLOATS:
            return _opaque("a float")
        if name == 'str':
            return _opaque("a string")
        if name == 'bool':
            if v.sort == OPAQUE:
                return v
            return _V(BOOL, self.truth(v, node))
        if name in ('int', 'i64'):
            return self.cast_int(v)
        return _opaque(f"a value cast to {name} (narrower than 64 bits in native code)")

    def ev_unary(self, node, st: _State) -> _V:
        op = node.op
        if op in ('&', '&mut', '*'):
            return self.ev(node.operand, st)      # references are identity in the VM
        v = self.ev(node.operand, st)
        if op == 'not':
            if v.sort == OPAQUE and not self.strict:
                return v
            return _V(BOOL, z3.Not(self.truth(v, node.operand)))
        if v.sort == INT:
            if op == '-':
                return _V(INT, self.dom.neg(self, st, v.z, node))
            if op == '~':
                return _V(INT, self.dom.bnot(self, st, v.z, node))
        self.unmodeled += 1
        return v if v.sort == OPAQUE else _opaque(f"'{op}' applied to a {v.sort}")

    def ev_binary(self, node, st: _State) -> _V:
        op = node.op
        if op in ('and', 'or'):
            left = self.ev(node.left, st)
            if left.sort == OPAQUE and not self.strict:
                self.ev(node.right, _State(st.pc, st.scopes))
                return left
            lt = self.truth(left, node.left)
            cond = lt if op == 'and' else z3.Not(lt)
            sub = _State(st.pc + (cond,), st.scopes)
            right = self.ev(node.right, sub)
            extra = sub.pc[len(st.pc) + 1:]
            if extra:
                st.pc = st.pc + (z3.Implies(cond, _and(extra)),)
            if right.sort == OPAQUE and not self.strict:
                return right
            rt = self.truth(right, node.right)
            return _V(BOOL, z3.And(lt, rt) if op == 'and' else z3.Or(lt, rt))
        a = self.ev(node.left, st)
        b = self.ev(node.right, st)
        return self.binop(op, a, b, node, st)

    def binop(self, op: str, a: _V, b: _V, node, st: _State) -> _V:
        if op in ('==', '!='):
            if a.sort == OPAQUE or b.sort == OPAQUE:
                return a if a.sort == OPAQUE else b
            if a.sort != b.sort:
                e = z3.BoolVal(False)     # values of different types are never equal
            elif a.sort == NIL:
                e = z3.BoolVal(True)
            else:
                e = a.z == b.z
            return _V(BOOL, e if op == '==' else z3.Not(e))
        if a.sort == INT and b.sort == INT:
            d, x, y = self.dom, a.z, b.z
            if op in ('<', '<=', '>', '>='):
                return _V(BOOL, x < y if op == '<' else x <= y if op == '<=' else x > y if op == '>' else x >= y)
            if op == '+':
                return _V(INT, d.add(self, st, x, y, node))
            if op == '-':
                return _V(INT, d.sub(self, st, x, y, node))
            if op == '*':
                return _V(INT, d.mul(self, st, x, y, node))
            if op in ('/', '%'):
                self.check_divisor(st, b, node, op)
                return _V(INT, (d.div if op == '/' else d.rem)(self, st, x, y, node))
            if op == '**':
                return _V(INT, d.pow(self, st, x, y, node))
            fn = {'&': d.band, '|': d.bor, '^': d.bxor, '<<': d.shl, '>>': d.shr}.get(op)
            if fn is None:
                raise Unsupported(f"operator '{op}' is not supported", node)
            return _V(INT, fn(self, st, x, y, node))
        if op in ('/', '%'):
            if b.sort == INT:
                self.check_divisor(st, b, node, op)
            elif b.sort == OPAQUE:
                raise Unsupported(f"division by {b.what}: the prover cannot rule out division by zero", node)
        # anything else on unmodeled values (strings, floats...) or a VM type error
        self.unmodeled += 1
        if a.sort == OPAQUE or b.sort == OPAQUE:
            return a if a.sort == OPAQUE else b
        return _opaque(f"'{op}' applied to a {a.sort} and a {b.sort} (a run-time type error)")

    def check_divisor(self, st: _State, b: _V, node, op: str):
        what = ("division by zero" if op == '/' else "modulo by zero") + self.suffix
        self.add_ob(st, 'div', b.z != self.dom.const(0), node, what,
                    ("division by zero", self.frames_here(getattr(node, 'line', 0))))

    # ── calls ───────────────────────────────────────────────────────────────
    def ev_call(self, node, st: _State) -> _V:
        f = node.func
        p = self.p
        if isinstance(f, til.Identifier):
            name = f.name
            if st.lookup(name) is not None:
                self.ev_args(node, st)
                self.unmodeled += 1
                return _opaque(f"the result of calling the value '{name}'")
            if name in p.funcs and '.' not in name:
                return self.call_user(name, node, st)
            if name in p.struct_names or name in ('Some', 'Ok', 'Err'):
                self.ev_args(node, st)
                return _opaque(f"a {name} value")
            if p.global_value(name) is not None:
                self.ev_args(node, st)
                self.unmodeled += 1
                return _opaque(f"the result of calling the global '{name}'")
            if name in til.NATIVE_ID:
                return self.call_native(name, node, st)
            raise Unsupported(f"unknown function '{name}'", node)
        if isinstance(f, til.Attribute):
            obj = f.obj
            if isinstance(obj, til.Identifier) and st.lookup(obj.name) is None:
                oname = obj.name
                if oname in p.struct_names and f"{oname}.{f.attr}" in p.funcs:
                    return self.call_user(f"{oname}.{f.attr}", node, st)
                if oname in p.adt_names:
                    self.ev_args(node, st)
                    return _opaque(f"{oname}.{f.attr}(...)")
                if oname in p.module_names and p.global_value(oname) is None and f.attr in p.funcs:
                    return self.call_user(f.attr, node, st)
                if oname.startswith('__Vec__') or oname.startswith('__HashMap__'):
                    return _opaque("a collection")
            if f.attr in p.methods_with_requires:
                raise Unsupported(f"method call '.{f.attr}(...)': methods with #[requires] are not supported",
                                  node)
            self.ev(obj, st)
            self.ev_args(node, st)
            self.unmodeled += 1
            return _opaque(f"the result of .{f.attr}()")
        self.ev(f, st)
        self.ev_args(node, st)
        self.unmodeled += 1
        return _opaque("the result of calling a function value")

    def ev_args(self, node, st: _State) -> List[_V]:
        return [self.ev(a, st) for a in node.args]

    def call_native(self, name: str, node, st: _State) -> _V:
        args = self.ev_args(node, st)
        if name in ('print', 'println'):
            return _NIL
        if name == 'abs' and len(args) == 1 and args[0].sort == INT:
            return _V(INT, self.dom.abs(self, st, args[0].z, node))
        if name in ('min', 'max') and args and all(a.sort == INT for a in args):
            best = args[0].z
            for a in args[1:]:
                best = z3.If(a.z < best if name == 'min' else a.z > best, a.z, best)
            return _V(INT, best)
        if name == 'int' and len(args) == 1:
            return self.cast_int(args[0])
        if name == 'bool' and len(args) == 1 and args[0].sort != OPAQUE:
            return _V(BOOL, self.truth(args[0], node))
        self.unmodeled += 1
        return _opaque(f"the result of {name}()")

    def call_user(self, key: str, node, st: _State) -> _V:
        p = self.p
        callee = p.funcs[key]
        args = self.ev_args(node, st)
        if callee.requires or callee.ensures:
            return self.call_with_contract(key, callee, args, node, st)
        if p.inlinable(key) and key not in [k for k, _ in self.inline_stack] and key != self.key:
            saved_pc, saved_n, saved_um = st.pc, len(self.obligations), self.unmodeled
            try:
                return self.inline(key, callee, args, node, st)
            except Unsupported:
                st.pc = saved_pc
                del self.obligations[saved_n:]
                self.unmodeled = saved_um
        self.unmodeled += 1
        return self.call_result(key, callee, st)

    def call_result(self, key: str, callee, st: _State) -> _V:
        """An unconstrained result of the callee's declared return type."""
        kind = _type_kind(callee.ret_type)
        short = key.replace('.', '_')
        if callee.ensures:
            desc = ('contract', f"the result of '{key}' (only its #[ensures] is known)")
        else:
            desc = ('call', f"the result of '{key}' (it has no #[ensures])")
        if kind == 'int':
            return self.fresh(st, INT, short, desc=desc)
        if kind == 'bool':
            if self.p.certainly_bool(key):
                return self.fresh(st, BOOL, short, desc=desc)
            return _opaque(f"the result of '{key}' (its `-> bool` returns are not all booleans)")
        if isinstance(callee.ret_type, til.VoidType) and not self.p.returns_value(key):
            return _NIL
        return _opaque(f"the result of '{key}'")

    def callee_env(self, callee, args: List[_V], node, st: _State) -> Dict[str, _V]:
        env: Dict[str, _V] = {}
        for i, prm in enumerate(callee.params):
            if i < len(args):
                v = args[i]
            elif prm.default is not None:
                v = self.ev(prm.default, _State(st.pc, [dict(env)]))
            else:
                raise Unsupported(f"call passes too few arguments", node)
            env[prm.name] = self.convert_annotated(v, prm.type, f"parameter '{prm.name}'")
        return env

    def call_with_contract(self, key: str, callee, args: List[_V], node, st: _State) -> _V:
        """Modular reasoning: the call must satisfy the callee's #[requires];
        the result is unknown except for what the callee's #[ensures] says
        (sound for any callee: the VM checks #[ensures] before returning)."""
        if key in self.contracts:
            raise Unsupported(f"the contract of '{key}' calls '{key}' itself (it would never finish)", node)
        env = self.callee_env(callee, args, node, st)
        call_frames = [(key, 0)] + self.frames_here(node.line)
        shown = [(f"{key}: {n}", v) for n, v in env.items() if v.sort in (INT, BOOL)]
        for k, cond in enumerate(callee.requires):
            text = _text(callee.requires_text, k)
            sub = _State(st.pc, [dict(env)])
            with self.context(sites=self.sites + [node], suffix=f" in #[requires: {text}] of '{key}'",
                              frames=call_frames, result=None, strict=True,
                              contracts=self.contracts + [key]):
                t = self.truth(self.ev(cond, sub), cond)
            st.pc = sub.pc
            msg = f"contract violation: {callee.name} requires {text}".strip()
            via = f" (called from '{self.cur_func()}')" if self.inline_stack else ""
            with self.context(sites=self.sites + [node]):
                self.add_ob(st, 'call', t, node, f"#[requires: {text}] of '{key}'{via}",
                            (msg, call_frames), shown)
        self.unmodeled += 1              # the callee may fail or not return
        res = self.call_result(key, callee, st)
        if res.sort in (INT, BOOL):
            for cond, text in self.p.usable_ensures(key):
                for part in _conjuncts(cond):
                    sub = _State(st.pc, [dict(env)])
                    try:
                        with self.context(sites=self.sites + [node], mode='assume', result=res,
                                          strict=True, frames=call_frames, contracts=self.contracts + [key],
                                          suffix=f" in #[ensures: {text}] of '{key}'"):
                            self.assume_conjuncts(part, sub)
                    except Unsupported:
                        continue         # not usable here: assume less (sound)
                    st.pc = sub.pc
        return res

    def inline(self, key: str, callee, args: List[_V], node, st: _State) -> _V:
        """A callee without contracts whose body is `return <expr>`: evaluate
        the expression itself (exact, including its divisions)."""
        env = self.callee_env(callee, args, node, st)
        sub = _State(st.pc, [env])
        expr = callee.body.statements[0].value
        with self.context(sites=self.sites + [node], inline_stack=self.inline_stack + [(key, node)],
                          suffix=f" in '{key}'", result=None, frames=None):
            v = self.ev(expr, sub)
            v = self.convert_annotated(v, callee.ret_type, f"the result of '{key}'")
        st.pc = sub.pc
        return v


def _text(texts: List[str], k: int) -> str:
    return texts[k] if texts and k < len(texts) else ""


# ═══════════════════════════════════════════════════════════════════════════════
#                     RUNNING A COUNTEREXAMPLE ON THE VM
# ═══════════════════════════════════════════════════════════════════════════════

# Natives a counterexample run may use; everything else (files, screen, keys,
# sleeping) is refused so that checking a counterexample has no side effects.
_PROBE_NATIVES = {
    'print', 'println', 'len', 'str', 'int', 'float', 'abs', 'sqrt', 'pow', 'sin', 'cos',
    'tan', 'log', 'exp', 'floor', 'ceil', 'round', 'min', 'max', 'input', 'chr', 'ord',
    'range', 'type_of', 'format', 'random', 'seed', 'bool', 'sum', 'panic', 'iter', 'args',
    'bytes_to_str', 'str_bytes', 'exit',
}
_PROBE_MAX_ITEMS = 1000000


def _make_probe_vm_class():
    class _ProbeVM(til.TILVM):
        """The reference VM, sandboxed: output discarded, no input, no files,
        bounded steps and allocations; the runtime error is recorded."""

        def __init__(self, module, argv=None, max_steps=PROBE_STEPS):
            super().__init__(module, write=lambda data: None, read_line=lambda: b"",
                             max_steps=max_steps, argv=argv)
            self.failure = None

        def report(self, msg, pc):
            text = msg.decode('utf-8', 'replace')
            if text.startswith("runtime error: "):
                text = text[len("runtime error: "):]
            frames = []
            if pc >= 0:
                frames.append((self.func_at(pc), self.line_for(pc)))
                for fr in reversed(self.frames[:-1]):
                    frames.append((self.m.funcs[fr[0]].name, self.line_for(fr[1] - 1)))
            self.failure = (text, frames)

        def call_native(self, nid, args):
            name = til.NATIVES[nid] if nid < len(til.NATIVES) else '?'
            if name not in _PROBE_NATIVES:
                self.panic(f"{name}() is not run while checking a counterexample")
            if name == 'range' and len(args) >= 2 and type(args[0]) is int and type(args[1]) is int \
                    and args[1] - args[0] > _PROBE_MAX_ITEMS:
                self.panic("range too large")
            return super().call_native(nid, args)

        def arith(self, op, a, b):
            if op == til.Op.MUL and type(a) in (bytes, list) and type(b) is int and \
                    len(a) * b > _PROBE_MAX_ITEMS:
                self.panic("value too large")
            return super().arith(op, a, b)

        def call_method(self, mid, recv, args):
            if til.METHODS[mid] == 'repeat' and args and type(args[0]) is int and \
                    type(recv) is bytes and len(recv) * args[0] > _PROBE_MAX_ITEMS:
                self.panic("value too large")
            return super().call_method(mid, recv, args)
    return _ProbeVM


_PROBE_VM = None


# ═══════════════════════════════════════════════════════════════════════════════
#                                  RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

class Failure:
    """An obligation that is not proved."""

    def __init__(self, ob: Obligation, status: str):
        self.ob = ob
        self.status = status          # counterexample | not proved | unknown
        self.inputs: List[Tuple[str, Any]] = []
        self.extras: List[Tuple[str, Any]] = []
        self.confirmed: Optional[bool] = None
        self.outcome = ""             # what the VM did with the inputs
        self.reason = ""              # unknown: why the solver gave up
        self.domain = ""
        self.depends: List[Tuple[str, str]] = []   # (kind, text) of the unknowns it relies on
        self.candidate: Optional[list] = None      # the solver's inputs, when edge values were used instead

    @property
    def values(self) -> Dict[str, Any]:
        return dict(self.inputs)


class FunctionReport:
    def __init__(self, key: str, fdef):
        self.key = key
        self.fdef = fdef
        self.status = 'proved'        # proved | counterexample | not proved | unknown | unsupported
        self.failures: List[Failure] = []
        self.reason = ""              # unsupported: why
        self.line = fdef.line
        self.col = fdef.col or 1
        self.n_obligations = 0
        self.notes: List[str] = []
        self.details: List[Tuple[Obligation, str, str]] = []   # (obligation, status, model)
        self.seconds = 0.0

    @property
    def counterexample(self) -> Optional[Dict[str, Any]]:
        for f in self.failures:
            if f.status == 'counterexample':
                return f.values
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#                                   PROVER
# ═══════════════════════════════════════════════════════════════════════════════

_ATTR_RE = re.compile(r'#\[\s*(requires|ensures)\s*:')


class Prover:
    def __init__(self, program, source: str, filename: str, timeout_ms: int = DEFAULT_TIMEOUT_MS,
                 run_counterexamples: bool = True):
        if not load_z3():
            raise RuntimeError(f"the prover needs Z3: {INSTALL_HINT}")
        self.program = program
        self.source = source
        self.filename = filename
        self.lines = source.split('\n')
        self.timeout = int(timeout_ms)
        self.run_counterexamples = run_counterexamples
        self.reporter = til.ErrorReporter(source, filename)
        self._collect()
        self._usable: Dict[str, list] = {}
        self._bool: Dict[str, bool] = {}
        self._attr_cache: Dict[int, Dict[Tuple[str, int], Tuple[int, int]]] = {}
        self._probe_cache: Dict[Any, Any] = {}

    # ── program tables ──────────────────────────────────────────────────────
    def _collect(self):
        prog = self.program
        self.funcs: Dict[str, Any] = {}
        self.struct_names: Set[str] = set()
        self.enum_values: Dict[str, Dict[str, int]] = {}
        self.adt_names: Set[str] = set()
        self.module_names: Set[str] = set(getattr(prog, 'module_names', set()) or ())
        self.inline_consts: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}       # name -> (python value,) | _V (opaque)
        traits = {}
        for s in prog.statements:
            if isinstance(s, til.FuncDef):
                self.funcs[s.name] = s
            elif isinstance(s, til.StructDef):
                self.struct_names.add(s.name)
            elif isinstance(s, til.TraitDef):
                traits[s.name] = s
            elif isinstance(s, til.EnumDef):
                if any(v.fields for v in s.variants):
                    self.adt_names.add(s.name)
                else:
                    values, nxt = {}, 0
                    for v in s.variants:
                        if v.value is not None:
                            try:
                                nxt = til.const_eval(v.value, {})
                            except Exception:
                                pass
                        values[v.name] = nxt
                        nxt = nxt + 1 if isinstance(nxt, int) else nxt
                    self.enum_values[s.name] = values
        for s in prog.statements:
            if isinstance(s, til.ImplBlock):
                defined = set()
                for m in s.methods:
                    self.funcs[f"{s.type_name}.{m.name}"] = m
                    defined.add(m.name)
                if s.trait_name in traits:
                    for tm in traits[s.trait_name].methods:
                        if tm.name not in defined and isinstance(tm.body, til.Block) and tm.body.statements:
                            self.funcs[f"{s.type_name}.{tm.name}"] = tm
        self.methods_with_requires = {k.split('.', 1)[1] for k, f in self.funcs.items()
                                      if '.' in k and f.requires}
        consts: Dict[str, Any] = {}
        for s in prog.statements:
            if not isinstance(s, til.VarDecl):
                continue
            if not s.mutable and isinstance(s.value, (til.IntLit, til.FloatLit, til.StringLit,
                                                      til.BoolLit, til.CharLit)):
                value = ord(s.value.value) if isinstance(s.value, til.CharLit) else s.value.value
                self.inline_consts[s.name] = value
                consts[s.name] = value
                self.globals[s.name] = (value,)
                continue
            if s.mutable:
                self.globals[s.name] = _opaque(f"the mutable global '{s.name}'")
            elif s.deferred:
                self.globals[s.name] = _opaque(f"the global '{s.name}' (initialized by the script)")
            elif s.value is None:
                self.globals[s.name] = (None,)
            else:
                try:
                    value = til.const_eval(s.value, consts)
                except Exception:
                    self.globals[s.name] = _opaque(f"the global '{s.name}' (computed at run time)")
                    continue
                if isinstance(value, (bool, int)):
                    value = value if isinstance(value, bool) else til.wrap_int64(value)
                    consts[s.name] = value
                    self.globals[s.name] = (value,)
                else:
                    self.globals[s.name] = _opaque(f"the global '{s.name}'")

    def global_value(self, name: str):
        return self.globals.get(name)

    def fold(self, node):
        """Mirror of BytecodeCompiler.try_fold: constant subexpressions are
        computed at compile time (with the compiler's own rules)."""
        try:
            v = til.const_eval(node, self.inline_consts)
        except Exception:                   # NotConstant, ZeroDivisionError, ...
            return _NO_FOLD
        if isinstance(v, (bool, int, float, str)):
            return v
        return _NO_FOLD

    def inlinable(self, key: str) -> bool:
        f = self.funcs[key]
        stmts = f.body.statements if isinstance(f.body, til.Block) else []
        return (len(stmts) == 1 and isinstance(stmts[0], til.Return) and stmts[0].value is not None
                and not any(p.name == 'self' for p in f.params))

    def returns_value(self, key: str) -> bool:
        return any(isinstance(n, til.Return) and n.value is not None for n in til.walk(self.funcs[key].body))

    def certainly_bool(self, key: str) -> bool:
        """Every `return` of this `-> bool` function returns a bool (the VM does
        not convert bool results), and it cannot end without `return`."""
        if key in self._bool:
            return self._bool[key]
        self._bool[key] = True               # recursion: assume, the base cases decide
        f = self.funcs[key]
        bool_params = {p.name for p in f.params if _type_kind(p.type) == 'bool'}

        def is_bool(e) -> bool:
            if isinstance(e, til.BoolLit):
                return True
            if isinstance(e, til.BinaryOp):
                return e.op in ('==', '!=', '<', '<=', '>', '>=', 'and', 'or')
            if isinstance(e, til.UnaryOp):
                return e.op == 'not'
            if isinstance(e, til.Identifier):
                return e.name in bool_params
            if isinstance(e, til.IfExpr):
                return e.else_expr is not None and is_bool(e.then_expr) and is_bool(e.else_expr)
            if isinstance(e, til.Cast):
                return isinstance(e.target_type, til.PrimitiveType) and e.target_type.name == 'bool'
            if isinstance(e, til.Call) and isinstance(e.func, til.Identifier):
                if e.func.name in self.funcs:
                    g = self.funcs[e.func.name]
                    return _type_kind(g.ret_type) == 'bool' and self.certainly_bool(e.func.name)
                return e.func.name == 'bool'
            return False
        ok = _always_returns(_body_stmts(f.body))
        for n in til.walk(f.body):
            if isinstance(n, til.Return) and (n.value is None or not is_bool(n.value)):
                ok = False
        self._bool[key] = ok
        return ok

    def usable_ensures(self, key: str) -> list:
        """#[ensures] clauses a caller may assume: those that read only
        `result`, parameters the callee never assigns, and global constants
        (others see values the caller does not know)."""
        if key in self._usable:
            return self._usable[key]
        f = self.funcs[key]
        params = {p.name for p in f.params}
        assigned = _assigned_names(_body_stmts(f.body))
        out = []
        for k, cond in enumerate(f.ensures):
            ok = True
            for name in _names_in(cond):
                if name == 'result' or name == 'NULL':
                    continue
                if name in params:
                    if name in assigned:
                        ok = False
                elif self.global_value(name) is None and name not in self.funcs and \
                        name not in self.enum_values and name not in self.module_names:
                    ok = False
            if ok:
                out.append((cond, _text(f.ensures_text, k)))
        self._usable[key] = out
        return out

    # ── locations and texts ─────────────────────────────────────────────────
    def attr_location(self, fdef, attr: Tuple[str, int]) -> Tuple[int, int]:
        """Source position of the k-th #[requires]/#[ensures] of a function."""
        cache = self._attr_cache.get(id(fdef))
        if cache is None:
            found: Dict[str, List[Tuple[int, int]]] = {'requires': [], 'ensures': []}
            ln = fdef.line - 1
            while ln >= 1 and ln <= len(self.lines):
                text = self.lines[ln - 1]
                stripped = text.strip()
                if stripped.startswith('#['):
                    hits = [(m.group(1), m.start() + 1) for m in _ATTR_RE.finditer(text)]
                    for kind, col in reversed(hits):
                        found[kind].append((ln, col))
                elif stripped and not stripped.startswith('#'):
                    break
                ln -= 1
            cache = {}
            for kind, locs in found.items():
                for k, loc in enumerate(reversed(locs)):
                    cache[(kind, k)] = loc
            self._attr_cache[id(fdef)] = cache
        return cache.get(attr, (fdef.line, fdef.col or 1))

    def stmt_text(self, node, keyword: bool = True) -> str:
        """Source text of a statement (`assert x > 0`); without the keyword
        (any language) when keyword=False."""
        line = getattr(node, 'line', 0)
        text = self.lines[line - 1].strip() if 1 <= line <= len(self.lines) else type(node).__name__
        if not keyword:
            parts = text.split(None, 1)
            text = parts[1] if len(parts) == 2 else text
        return text

    # ── scope ───────────────────────────────────────────────────────────────
    def local_keys(self) -> List[str]:
        """Functions defined in the file itself (not imported), in order."""
        try:
            own = til.parse_source(self.source, self.filename)
        except Exception:
            return list(self.funcs)
        keys = []
        for s in own.statements:
            if isinstance(s, til.FuncDef):
                keys.append(s.name)
            elif isinstance(s, til.ImplBlock):
                keys.extend(f"{s.type_name}.{m.name}" for m in s.methods)
        return [k for k in keys if k in self.funcs]

    def has_contract_work(self, key: str) -> bool:
        f = self.funcs[key]
        if f.requires or f.ensures:
            return True
        for n in til.walk(f.body):
            if isinstance(n, (til.AssertStmt, til.InvariantStmt)):
                return True
            if isinstance(n, til.Call) and self._calls_contract(n):
                return True
        return False

    def _calls_contract(self, call) -> bool:
        f = call.func
        if isinstance(f, til.Identifier):
            g = self.funcs.get(f.name)
            return g is not None and bool(g.requires)
        if isinstance(f, til.Attribute):
            if isinstance(f.obj, til.Identifier):
                g = self.funcs.get(f"{f.obj.name}.{f.attr}") or (
                    self.funcs.get(f.attr) if f.obj.name in self.module_names else None)
                if g is not None and g.requires:
                    return True
            return f.attr in self.methods_with_requires
        return False

    def targets(self, explicit: bool, only: Optional[str] = None) -> List[str]:
        keys = self.local_keys()
        if only is not None:
            return [k for k in keys if k == only or k.split('.')[-1] == only]
        if explicit:
            return [k for k in keys if self.has_contract_work(k)]
        return [k for k in keys if getattr(self.funcs[k], 'level', 2) == 4]

    # ── solving ─────────────────────────────────────────────────────────────
    def solve(self, ob: Obligation, solver: str = 'pipeline'):
        """Is `pc and not goal` unsatisfiable?  ('proved'|'sat'|'unknown',
        model, reason).  `pipeline` eliminates equations before searching
        (10-100x faster on the integer, nonlinear queries); `default` is Z3's
        own configuration (bit-blasting for bit-vectors)."""
        if z3.is_true(ob.goal):
            return 'proved', None, ""
        if solver == 'pipeline':
            s = z3.Then('simplify', 'propagate-values', 'solve-eqs', 'smt').solver()
        else:
            s = z3.Solver()
        s.set('timeout', self.timeout)
        s.add(*ob.pc)
        s.add(z3.Not(ob.goal))
        r = s.check()
        if r == z3.unsat:
            return 'proved', None, ""
        if r == z3.sat:
            return 'sat', s.model(), ""
        return 'unknown', None, s.reason_unknown()

    def _int_encoding(self, key: str, fdef, n_regular: int):
        """The integer encoding, usable only when every overflow obligation
        is proved; (None, None) otherwise (the bit-vector results stand)."""
        try:
            enc = _Encoder(self, key, fdef, _IntDomain())
            obs = enc.encode()
        except Exception:               # _NotTranslatable (bit operations...), Unsupported
            return None, None
        regular = [o for o in obs if o.kind != 'overflow']
        if len(regular) != n_regular:
            return None, None
        for o in obs:
            if o.kind == 'overflow' and self.solve(o)[0] != 'proved':
                return None, None
        return regular, enc

    def prove_function(self, key: str) -> FunctionReport:
        try:
            return self._prove_function(key)
        except Exception as e:          # never crash `til check`: report it, prove nothing
            rep = FunctionReport(key, self.funcs[key])
            rep.status = 'unsupported'
            rep.reason = f"internal error in the prover ({type(e).__name__}: {e})"
            return rep

    def _prove_function(self, key: str) -> FunctionReport:
        fdef = self.funcs[key]
        rep = FunctionReport(key, fdef)
        t0 = time.time()
        try:
            bv = _Encoder(self, key, fdef, _BVDomain())
            obs = bv.encode()
        except Unsupported as u:
            rep.status = 'unsupported'
            rep.reason = u.reason
            if u.loc is not None:
                rep.line, rep.col = u.loc
            elif u.node is not None and getattr(u.node, 'line', 0):
                rep.line, rep.col = u.node.line, u.node.col or 1
            rep.seconds = time.time() - t0
            return rep
        rep.n_obligations = len(obs)
        int_cache: Dict[str, Any] = {}

        def int_obs():
            if 'v' not in int_cache:
                int_cache['v'] = self._int_encoding(key, fdef, len(obs))
            return int_cache['v']
        # each attempt has the full time limit; nonlinear arithmetic is much
        # easier for Z3 on integers (when no operation overflows)
        attempts = [('bit-vector', 'pipeline'), ('integer', 'pipeline'), ('bit-vector', 'default')]
        if bv.dom.nonlinear:
            attempts[0], attempts[1] = attempts[1], attempts[0]
        for i, ob in enumerate(obs):
            decided = None
            reason = ""
            for dom, solver in attempts:
                if dom == 'bit-vector':
                    o, enc = ob, bv
                else:
                    iobs, ienc = int_obs()
                    if iobs is None:
                        continue
                    o, enc = iobs[i], ienc
                status, model, why = self.solve(o, solver)
                if status != 'unknown':
                    decided = (status, model, o, enc, dom)
                    break
                reason = why or reason
            if decided is None:
                fail = Failure(ob, 'unknown')
                fail.reason = reason or "unknown"
                rep.failures.append(fail)
                rep.details.append((ob, 'unknown', ''))
                continue
            status, model, o, enc, dom = decided
            if status == 'proved':
                rep.details.append((ob, 'proved', dom))
                continue
            if o.prefer is not None:
                alt = Obligation(o.kind, o.what, o.line, o.col, o.pc + (o.prefer,), o.goal, o.expect, o.show,
                                 o.unmodeled, o.loop)
                found, alt_model, _why = self.solve(alt)
                if found == 'sat':
                    model = alt_model
            fail = self._failure(key, ob, o, enc, model, dom)
            rep.failures.append(fail)
            rep.details.append((ob, fail.status, dom))
        if not rep.failures:
            s = z3.Solver()
            s.set('timeout', self.timeout)
            s.add(*bv.entry_pc)
            if fdef.requires and s.check() == z3.unsat:
                rep.notes.append("its #[requires] can never hold: every call fails, so the proof is vacuous")
        for kind in ('counterexample', 'not proved', 'unknown'):
            if any(f.status == kind for f in rep.failures):
                rep.status = kind
                break
        rep.seconds = time.time() - t0
        return rep

    def _failure(self, key: str, ob: Obligation, o: Obligation, enc: _Encoder, model, dom: str) -> Failure:
        def value(v: _V):
            r = model.eval(v.z, model_completion=True)
            if v.sort == BOOL:
                return z3.is_true(r)
            return enc.dom.value(r)
        fail = Failure(ob, 'counterexample')
        fail.domain = dom
        runnable = True
        for name, sort, z, _t in enc.inputs:
            if sort in (INT, BOOL):
                fail.inputs.append((name, value(_V(sort, z))))
            else:
                runnable = False
        fail.extras = [(label, value(v)) for label, v in o.show if v.sort in (INT, BOOL)]
        free = _free_consts(list(o.pc) + [o.goal])
        for name in sorted(free, key=lambda n: (len(n), n)):
            dep = enc.fresh_desc.get(name)
            if dep is not None and dep not in fail.depends:
                fail.depends.append(dep)
        exact = o.unmodeled == 0 and not o.loop and free <= {name for name, *_ in enc.inputs}
        if self.run_counterexamples and runnable and ob.expect is not None:
            fail.confirmed, fail.outcome = self.confirm(key, fail.inputs, ob.expect)
            if fail.confirmed is False:
                found = self.search_counterexample(key, fail.inputs, ob.expect)
                if found is not None:
                    fail.candidate = fail.inputs
                    fail.inputs = [(n, v) for (n, _old), v in zip(fail.inputs, found)]
                    fail.extras = []
                    fail.confirmed = True
                    fail.outcome = f"running it fails with: {ob.expect[0]} (found by running edge values; " \
                                   f"the solver's candidate relied on an abstraction)"
        elif not runnable:
            fail.outcome = "it cannot be run with these inputs (some parameters are not int or bool)"
        if fail.confirmed:
            fail.status = 'counterexample'
        elif fail.confirmed is None and exact:
            fail.status = 'counterexample'
        else:
            fail.status = 'not proved'
        return fail

    # ── running counterexamples ─────────────────────────────────────────────
    def confirm(self, key: str, inputs: List[Tuple[str, Any]], expect) -> Tuple[Optional[bool], str]:
        """Run the function with the inputs on the reference VM. Returns
        (True, outcome) when it fails exactly as predicted, (False, what
        happened instead), or (None, why) when it cannot be run."""
        try:
            failure = self.run_function(key, [v for _n, v in inputs])
        except Exception as e:           # the program cannot be rebuilt / run here
            return None, f"it could not be run ({e})"
        if failure is None:
            return False, "running it with these inputs does not fail"
        msg, frames = failure
        if _matches(failure, expect):
            return True, f"running it fails with: {msg}"
        if msg.startswith("fatal error: step limit"):
            return False, f"running it with these inputs did not finish within {PROBE_STEPS} steps"
        where = f" (in '{frames[0][0]}', line {frames[0][1]})" if frames else ""
        return False, f"running it with these inputs fails differently: {msg}{where}"

    def search_counterexample(self, key: str, inputs: List[Tuple[str, Any]], expect) -> Optional[list]:
        """The solver's candidate did not fail when run (it relies on an
        abstraction).  Try edge values instead (0, +-1, MIN, MAX, ...): each
        hit is a real counterexample, checked by running the function."""
        if not inputs:
            return None
        rng = __import__('random').Random(len(inputs) * 7919 + sum(hash(n) & 0xFFFF for n, _ in inputs))
        pools = []
        for _name, v in inputs:
            if isinstance(v, bool):
                pools.append([not v, v])
            else:
                pools.append(list(dict.fromkeys([v, 0, 1, -1, 2, INT_MAX, INT_MIN, INT_MAX - 1, INT_MIN + 1,
                                                 10, 100, -100, 1000000, 1 << 32])))
        seen = set()
        candidates = []
        for i, pool in enumerate(pools):           # vary one input at a time ...
            for v in pool:
                vals = [x for _n, x in inputs]
                vals[i] = v
                candidates.append(vals)
        for _ in range(SEARCH_RUNS):               # ... then random combinations
            candidates.append([rng.choice(pool) for pool in pools])
        runs = 0
        for vals in candidates:
            t = tuple(vals)
            if t in seen:
                continue
            seen.add(t)
            runs += 1
            if runs > SEARCH_RUNS:
                break
            try:
                failure = self.run_function(key, vals, SEARCH_STEPS)
            except Exception:
                return None
            if failure is not None and _matches(failure, expect):
                return vals
        return None

    def _probe_module(self, key: str, n_args: int, literal_args: Optional[list]):
        """The program with a `main` that calls the function: with the values
        as literals, or (compiled once) reading them from args()."""
        T = til
        cache_key = (key, n_args) if literal_args is None else None
        if cache_key is not None and cache_key in self._probe_cache:
            return self._probe_cache[cache_key]
        if key == 'main' and n_args == 0:
            prog = self.program
        else:
            stmts = [s for s in self.program.statements if not (isinstance(s, T.FuncDef) and s.name == 'main')]
            if '.' in key:
                tname, mname = key.split('.', 1)
                func = T.Attribute(obj=T.Identifier(name=tname), attr=mname)
            else:
                func = T.Identifier(name=key)
            if literal_args is not None:
                args = [T.BoolLit(value=v) if isinstance(v, bool) else T.IntLit(value=v) for v in literal_args]
            else:
                args = []
                for i, (_n, sort, _z, _t) in enumerate(self._inputs_of(key)):
                    item = T.Index(obj=T.Call(func=T.Identifier(name='args'), args=[]), index=T.IntLit(value=i))
                    if sort == BOOL:
                        args.append(T.BinaryOp(op='==', left=item, right=T.StringLit(value="true")))
                    else:
                        args.append(T.Call(func=T.Identifier(name='int'), args=[item]))
            main = T.FuncDef(name='main', params=[], ret_type=T.T_VOID,
                             body=T.Block(statements=[T.ExprStmt(expr=T.Call(func=func, args=args))]))
            prog = T.Program(statements=stmts + [main])
            prog.module_names = getattr(self.program, 'module_names', set())
        module = T.BytecodeCompiler(self.source, self.filename).compile_program(prog)
        module.link()
        if cache_key is not None:
            self._probe_cache[cache_key] = module
        return module

    def _inputs_of(self, key: str):
        f = self.funcs[key]
        return [(p.name, INT if _type_kind(p.type) == 'int' else BOOL, None, p.type) for p in f.params]

    def run_function(self, key: str, values: List[Any], max_steps: int = PROBE_STEPS):
        """Run key(*values) on the sandboxed reference VM; returns the runtime
        error (message, [(function, line), ...]) or None."""
        global _PROBE_VM
        if _PROBE_VM is None:
            _PROBE_VM = _make_probe_vm_class()
        if values and ('args' in self.funcs or 'int' in self.funcs):
            module, argv = self._probe_module(key, len(values), list(values)), []
        else:
            module = self._probe_module(key, len(values), None)
            argv = [("true" if v else "false") if isinstance(v, bool) else str(v) for v in values]
        vm = _PROBE_VM(module, argv, max_steps)
        vm.run()
        return vm.failure


def _matches(failure, expect) -> bool:
    """Is the VM's runtime error the predicted one (message, and the
    innermost frames: function names, and lines where known)?"""
    msg, frames = failure
    exp_msg, exp_frames = expect
    if msg != exp_msg or len(frames) < len(exp_frames):
        return False
    for (ef, el), (af, al) in zip(exp_frames, frames):
        if ef != af or (el and al and el != al):
            return False
    return True


def _always_returns(stmts: list) -> bool:
    """Syntactically, does every path through the statements end in `return`?"""
    for s in stmts:
        if isinstance(s, til.Return):
            return True
        if isinstance(s, til.If) and s.else_body is not None:
            bodies = [s.then_body] + [b for _c, b in s.elifs] + [s.else_body]
            if all(_always_returns(_body_stmts(b)) for b in bodies):
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#                                  OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _assignments(pairs) -> str:
    """`x = 1, y = true`; labels `f: a` are grouped into a call `f(a = 1, b = 0)`."""
    parts: List[str] = []
    call, args = None, []
    for n, v in list(pairs) + [(None, None)]:
        prefix, _, arg = n.partition(': ') if n else ('', '', '')
        if call is not None and (not arg or prefix != call):
            parts.append(f"the call {call}({', '.join(args)})")
            call, args = None, []
        if n is None:
            break
        if arg:
            call = prefix
            args.append(f"{arg} = {_fmt(v)}")
        else:
            parts.append(f"{n} = {_fmt(v)}")
    return ", ".join(parts)


def _inputs_text(fail: Failure) -> str:
    return _assignments(fail.inputs) if fail.inputs else "no inputs (it fails on every run)"


def summary_line(rep: FunctionReport) -> str:
    if rep.status == 'proved':
        n = rep.n_obligations
        text = f"{rep.key}: proved" + (f" ({n} condition{'s' if n != 1 else ''})" if n else
                                       " (nothing to check)")
        return text + (f" - note: {rep.notes[0]}" if rep.notes else "")
    if rep.status == 'unsupported':
        return f"{rep.key}: unsupported: {rep.reason} (line {rep.line})"
    first = next(f for f in rep.failures if f.status == rep.status)
    more = len(rep.failures) - 1
    tail = f"; {more} more problem{'s' if more != 1 else ''}" if more > 0 else ""
    ob = first.ob
    if rep.status == 'counterexample':
        return f"{rep.key}: counterexample: {_inputs_text(first)} ({ob.what} fails, line {ob.line}){tail}"
    if rep.status == 'not proved':
        return f"{rep.key}: not proved: {ob.what} (line {ob.line}){tail}"
    why = "timeout" if first.reason in ('timeout', 'canceled') else first.reason
    return f"{rep.key}: unknown ({why}): {ob.what} (line {ob.line}){tail}"


def format_errors(rep: FunctionReport, reporter, timeout_ms: int) -> List[str]:
    """E-PROVE errors (ErrorReporter style) for a function that is not proved."""
    out = []
    if rep.status == 'unsupported':
        out.append(reporter.format_error(
            rep.line, rep.col, f"cannot prove '{rep.key}': unsupported: {rep.reason}", ERROR_CODE,
            "the prover models int and bool values, if/elif/else, loops with `invariant`, and calls; "
            "see docs/TIL_PROOFS.md (til prove --allow-unsupported skips such functions)"))
        return out
    for fail in rep.failures:
        ob = fail.ob
        extras = _assignments(fail.extras)
        if fail.status == 'counterexample':
            msg = f"'{rep.key}': {ob.what} fails for {_inputs_text(fail)}"
            parts = [f"with these inputs {extras}"] if extras else []
            if fail.confirmed:
                parts.append(f"confirmed on the TIL VM ({fail.outcome})")
            else:                     # exact encoding, but it could not be run here
                parts.append(f"not run on the VM: {fail.outcome}" if fail.outcome else "not run on the VM")
            hint = "; ".join(parts)
        elif fail.status == 'not proved':
            msg = f"'{rep.key}': cannot prove {ob.what}"
            found = f"the solver found {_inputs_text(fail)}" + (f" ({extras})" if extras else "")
            kinds = [k for k, _t in fail.depends]
            if ob.kind in ('loop-step', 'loop-entry') or 'loop' in kinds:
                advice = "the loop invariant (`invariant ...` first in the loop body) may be missing or too weak"
            elif 'contract' in kinds:
                advice = "the #[ensures] of the called function may be too weak"
            elif 'call' in kinds:
                advice = "add #[ensures] to the called function: without it its result is unknown"
            elif ob.unmodeled or 'opaque' in kinds:
                advice = "it depends on values or operations the prover does not model"
            else:
                advice = "add #[requires] bounds or a stronger contract"
            relies = ""
            if fail.depends:
                relies = "; it relies on " + ", ".join(t for _k, t in fail.depends[:3])
            hint = f"{found}, but {fail.outcome or 'this is not a real run'}{relies}; {advice}"
        else:
            why = "timeout" if fail.reason in ('timeout', 'canceled') else fail.reason
            msg = f"'{rep.key}': could not prove {ob.what} (unknown: {why}, {timeout_ms} ms per query)"
            hint = "raise --timeout, add bounds with #[requires: ...], or split the computation"
        out.append(reporter.format_error(ob.line, ob.col, msg, ERROR_CODE, hint))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#                                ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def prove_program(program, source: str, filename: str, explicit: bool = True, only: Optional[str] = None,
                  timeout_ms: int = DEFAULT_TIMEOUT_MS, run_counterexamples: bool = True) -> List[FunctionReport]:
    """Prove the functions in scope; one report per function."""
    prover = Prover(program, source, filename, timeout_ms, run_counterexamples)
    return [prover.prove_function(k) for k in prover.targets(explicit, only)]


def prove_source(source: str, filename: str = "<test>", **kw) -> List[FunctionReport]:
    """Parse, resolve imports, verify and prove (for tests and tools)."""
    program = til.resolve_imports(til.parse_source(source, filename), filename)
    errors = til.SemanticChecker(source, filename).check(program)
    if errors:
        raise til.CompileError("\n\n".join(errors))
    til.BytecodeCompiler(source, filename).compile_program(program)
    return prove_program(program, source, filename, **kw)


def check_program(program, source: str, filename: str, explicit: bool) -> List[str]:
    """`til check` (explicit=False: #[level: 4] functions) and `til check
    --prove` (explicit=True: every function with contracts). Returns the
    formatted errors; an empty list means everything in scope is proved."""
    probe = _ScopeProbe(program, source, filename)
    keys = probe.targets(explicit)
    if not keys:
        return []
    if not load_z3():
        if explicit:
            return [f"error[{ERROR_CODE}]: til check --prove needs the Z3 SMT solver, which is not installed\n"
                    f"  = hint: {INSTALL_HINT}"]
        print(f"warning: {len(keys)} #[level: 4] function(s) in {filename} were not proved: "
              f"the Z3 SMT solver is not installed ({INSTALL_HINT})", file=sys.stderr)
        return []
    prover = Prover(program, source, filename)
    errors: List[str] = []
    proved = []
    for key in keys:
        rep = prover.prove_function(key)
        if rep.status == 'proved':
            proved.append(key)
        else:
            errors.extend(format_errors(rep, prover.reporter, prover.timeout))
    if explicit and proved:
        print(f"proved: {', '.join(proved)}")
    return errors


class _ScopeProbe(Prover):
    """Target selection without Z3 (so `til check` can warn when it is missing)."""

    def __init__(self, program, source: str, filename: str):
        self.program = program
        self.source = source
        self.filename = filename
        self.lines = source.split('\n')
        self._collect()


USAGE = f"""usage: til prove <file.til> [options]

Proves, for ALL int/bool inputs, that every #[ensures] holds when the function
returns, every assert/invariant holds, every call satisfies the callee's
#[requires] and nothing divides by zero (64-bit wrap-around semantics).

options:
  --timeout MS           solver time limit per condition (default {DEFAULT_TIMEOUT_MS})
  --function NAME        prove only this function (also for functions without contracts)
  --allow-unsupported    unsupported functions do not make the exit code nonzero
  -v, --verbose          list every proof condition and its result
  -h, --help             this help

Exit code 0 when everything in scope is proved. Needs Z3: {INSTALL_HINT}
"""


def main(argv: List[str]) -> int:
    path = None
    timeout = DEFAULT_TIMEOUT_MS
    only = None
    allow_unsupported = False
    verbose = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('-h', '--help'):
            print(USAGE, end="")
            return 0
        if a in ('--timeout', '--function') and i + 1 < len(argv):
            value = argv[i + 1]
            i += 2
        elif a.startswith('--timeout=') or a.startswith('--function='):
            a, value = a.split('=', 1)
            i += 1
        elif a == '--allow-unsupported':
            allow_unsupported = True
            i += 1
            continue
        elif a in ('-v', '--verbose'):
            verbose = True
            i += 1
            continue
        elif not a.startswith('-') and path is None:
            path = a
            i += 1
            continue
        else:
            print(f"til prove: unknown option {a}\n\n{USAGE}", end="", file=sys.stderr)
            return 2
        if a == '--timeout':
            try:
                timeout = int(value)
                if timeout <= 0:
                    raise ValueError
            except ValueError:
                print(f"til prove: --timeout needs a positive number of milliseconds, got {value!r}",
                      file=sys.stderr)
                return 2
        else:
            only = value
    if path is None:
        print(USAGE, end="", file=sys.stderr)
        return 2
    if not os.path.exists(path):
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1
    if not load_z3():
        print(f"til prove needs the Z3 SMT solver, which is not installed.\n"
              f"Install it with:  {INSTALL_HINT}", file=sys.stderr)
        return 1
    try:
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        program = til.resolve_imports(til.parse_source(source, path), path)
        errors = til.SemanticChecker(source, path).check(program)
        if errors:
            for err in errors:
                print(err, file=sys.stderr)
            return 1
        til.BytecodeCompiler(source, path).compile_program(program)
    except SyntaxError as e:
        print(f"Syntax Error: {e}", file=sys.stderr)
        return 1
    except til.CompileError as e:
        print(str(e), file=sys.stderr)
        return 1
    prover = Prover(program, source, path, timeout)
    if only is not None and not prover.targets(False, only):
        print(f"til prove: no function named '{only}' in {path}", file=sys.stderr)
        return 1
    keys = prover.targets(True, only)
    print(f"Proving {path} with Z3 {z3_version()} (64-bit integers, {timeout} ms per query)")
    if not keys:
        print("Nothing to prove: no function has #[requires], #[ensures], assert or invariant "
              "or calls a function with #[requires].")
        return 0
    reports = []
    for key in keys:
        rep = prover.prove_function(key)
        reports.append(rep)
        print("  " + summary_line(rep))
        if verbose:
            for ob, status, dom in rep.details:
                how = f" [{dom}]" if dom and status == 'proved' else ""
                print(f"      line {ob.line}: {ob.what}: {status}{how}")
            if rep.status == 'proved' or verbose:
                print(f"      ({rep.seconds:.2f} s)")
        sys.stdout.flush()
        for err in format_errors(rep, prover.reporter, prover.timeout):
            print(err, file=sys.stderr)
    counts: Dict[str, int] = {}
    for rep in reports:
        counts[rep.status] = counts.get(rep.status, 0) + 1
    order = ['proved', 'counterexample', 'not proved', 'unknown', 'unsupported']
    parts = []
    for st in order:
        if st in counts:
            n = counts[st]
            label = st if st != 'counterexample' else ('counterexample' if n == 1 else 'counterexamples')
            parts.append(f"{n} {label}")
    print("Result: " + ", ".join(parts))
    bad = [r for r in reports if r.status != 'proved' and not (allow_unsupported and r.status == 'unsupported')]
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
