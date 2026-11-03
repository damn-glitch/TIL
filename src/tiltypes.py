"""
TIL 3.1 -- gradual type inference (TypeInfer) and the typed native backend.

This module is optional: src/til.py loads it when it is present (like the
other TIL 3.1 tool modules) and keeps its old behaviour without it.

Part 1 (this file, TYPE INFERENCE): a Hindley-Milner style inference over the
whole program with *gradual* semantics, because TIL itself is dynamically
typed (the VMs check types at run time):

* annotations give types; unannotated parameters, locals and return types get
  type variables; unification (union-find) connects them;
* operators, indexing, fields, methods and iteration are *deferred
  constraints* that are solved once the operand types are known (so `+` on two
  unknowns stays open until a call site decides between int, float and str);
* top-level functions are generalised per strongly connected component of the
  call graph (with their open constraints), so a function with unannotated
  parameters is generic and every call site instantiates it afresh;
* a conflict never stops the compiler: the conflicting type widens to `dyn`
  (dynamic) -- the program is still valid, it just cannot be compiled to
  native code with static types (the native backend then falls back to the
  VM);
* E-TYPE errors are reported only for DEFINITE errors: a value whose type is
  fully known and can never satisfy an annotation (`let x: int = "a"`), an
  unknown field or method on a known struct type, an operator applied to
  concrete types it never accepts (`1 - "a"`). Types that depend on generic
  parameters, code inside `retry`/`timeout` bodies (where failing on purpose is
  how recoverable errors are produced) and the arguments of builtin natives are
  never reported -- the VMs check those at run time.

Phase A infers the whole program (schemes + E-TYPE). Phase B re-infers one
function body for one concrete instantiation (monomorphisation) and records
the type of every expression for the code generator.

Part 2 (TYPED NATIVE BACKEND) turns the per-instance types into C.
"""

import os
import sys

if 'til' in sys.modules:            # til.py loaded us (also as __main__): use that module
    til = sys.modules['til']
else:                               # pragma: no cover - direct import from tests/tools
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import til                      # noqa: E402

from typing import Any, Dict, List, Optional, Set, Tuple  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
#                         TYPE TERMS AND UNIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TVar:
    """A type variable (union-find node). Only a root holds the binding."""
    __slots__ = ('id', 'parent', 'value', 'waiters')
    counter = 0

    def __init__(self):
        TVar.counter += 1
        self.id = TVar.counter
        self.parent = None      # union-find link to another TVar
        self.value = None       # binding (a TCon) - meaningful at the root only
        self.waiters = None     # constraints blocked on this variable

    def __repr__(self):
        return show(self)


class TCon:
    """A type constructor applied to arguments: int, [T], {K: V}, fn, struct..."""
    __slots__ = ('name', 'args')

    def __init__(self, name: str, args=()):
        self.name = name
        self.args = tuple(args)

    def __repr__(self):
        return show(self)


INT = TCon('int')
FLOAT = TCon('float')
BOOL = TCon('bool')
STR = TCon('str')
NIL = TCon('nil')       # None / no value
DYN = TCon('dyn')       # dynamically typed (conflicting static types)
UNK = TCon('unk')       # never constrained: any representation will do

NUMERIC = ('int', 'float')
SCALAR_NAMES = ('int', 'float', 'bool', 'str')


def arr(elem) -> TCon:
    return TCon('array', (elem,))


def mapt(key, val) -> TCon:
    return TCon('map', (key, val))


def opt(inner) -> TCon:
    return TCon('option', (inner,))


def resultt(ok, err) -> TCon:
    return TCon('result', (ok, err))


def fnt(ret, params) -> TCon:
    return TCon('fn', (ret,) + tuple(params))


def structt(name: str) -> TCon:
    return TCon('S:' + name)


def adtt(name: str) -> TCon:
    return TCon('E:' + name)


def is_struct(t) -> bool:
    return isinstance(t, TCon) and t.name.startswith('S:')


def is_adt(t) -> bool:
    return isinstance(t, TCon) and t.name.startswith('E:')


def nominal_name(t) -> str:
    return t.name[2:]


def root(v: TVar) -> TVar:
    r = v
    while r.parent is not None:
        r = r.parent
    while v.parent is not None and v.parent is not r:     # path compression
        nxt = v.parent
        v.parent = r
        v = nxt
    return r


def resolve(t):
    """-> a TCon, or the unbound root TVar."""
    if isinstance(t, TVar):
        r = root(t)
        return r if r.value is None else r.value
    return t


def deep(t, unbound=None):
    """Fully resolved copy of a term; unbound variables become `unbound` (UNK)."""
    t = resolve(t)
    if isinstance(t, TVar):
        return t if unbound is None else unbound
    if not t.args:
        return t
    return TCon(t.name, [deep(a, unbound) for a in t.args])


def nullable(t) -> bool:
    """Types whose values may be None (references, or Option)."""
    return isinstance(t, TCon) and (t.name in ('str', 'array', 'map', 'option', 'fn', 'nil') or
                                    is_struct(t) or is_adt(t))


def show(t) -> str:
    """User-facing spelling of a type (used in E-TYPE messages)."""
    t = resolve(t)
    if isinstance(t, TVar):
        return f"?{t.id}"
    n = t.name
    if n in ('int', 'float', 'bool', 'str'):
        return n
    if n == 'nil':
        return 'None'
    if n == 'dyn':
        return 'dynamic'
    if n == 'unk':
        return 'unknown'
    if n == 'array':
        return f"[{show(t.args[0])}]"
    if n == 'map':
        return f"{{{show(t.args[0])}: {show(t.args[1])}}}"
    if n == 'option':
        return f"Option<{show(t.args[0])}>"
    if n == 'result':
        return f"Result<{show(t.args[0])}, {show(t.args[1])}>"
    if n == 'fn':
        return f"fn({', '.join(show(a) for a in t.args[1:])}) -> {show(t.args[0])}"
    if n == 'strish':
        return f"str or {show(t.args[0])}"
    if is_struct(t) or is_adt(t):
        return nominal_name(t)
    return n


def canon(t) -> str:
    """Canonical C-identifier-safe spelling (instance keys, C names)."""
    t = resolve(t)
    if isinstance(t, TVar):
        return 'unk'
    n = t.name
    if not t.args:
        if is_struct(t) or is_adt(t):
            return til.CCodeGenerator.mangle_name(nominal_name(t))
        return n
    inner = '_'.join(canon(a) for a in t.args)
    short = {'array': 'arr', 'map': 'map', 'option': 'opt', 'result': 'res', 'fn': 'fn'}.get(n, n)
    return f"{short}_{inner}"


def contains(t, names) -> bool:
    """Does a (resolved) type mention one of these constructor names?"""
    t = resolve(t)
    if isinstance(t, TVar):
        return 'var' in names
    if t.name in names:
        return True
    return any(contains(a, names) for a in t.args)


class Solver:
    """Unification + a wake-up queue for deferred constraints."""

    def __init__(self, phase: str = 'A'):
        self.phase = phase
        self.ready: List['Constraint'] = []
        self.running = False
        self.conflicts: List[Tuple[Any, str]] = []   # (node, text): hard conflicts (phase B)
        self.current_node = None
        self.sink: Optional[list] = None    # every constraint created goes here (generalisation)
        self.all: list = []                 # phase B: all constraints (settled at the end)

    # ── variables ───────────────────────────────────────────────────────────
    def fresh(self) -> TVar:
        return TVar()

    def bind(self, r: TVar, value: TCon):
        r.value = value
        self.wake(r)

    def union(self, a: TVar, b: TVar):
        """Merge class a into class b (a unbound)."""
        a.parent = b
        if a.waiters:
            self.wake(a)

    def wake(self, r: TVar):
        if r.waiters:
            self.ready.extend(r.waiters)
            r.waiters = None

    def wait(self, var: TVar, c: 'Constraint'):
        r = root(var)
        if r.waiters is None:
            r.waiters = []
        r.waiters.append(c)

    # ── unification ─────────────────────────────────────────────────────────
    @staticmethod
    def parts(t):
        if isinstance(t, TVar):
            r = root(t)
            return r, r.value
        return None, t

    def unify(self, a, b) -> bool:
        """Make a and b the same type. A conflict widens the variables involved
        to `dyn` (returns True: absorbed); only a conflict between two bare type
        constructors, with no variable to widen, returns False."""
        va, ta = self.parts(a)
        vb, tb = self.parts(b)
        if va is not None and va is vb:
            return True
        if va is not None and vb is not None:
            if ta is None:
                self.union(va, vb)
                return True
            if tb is None:
                self.union(vb, va)
                return True
            # both bound: merge the classes, then reconcile the two values
            va.parent = vb
            waiters = va.waiters
            va.waiters = None
            if waiters:
                self.ready.extend(waiters)
            self.reconcile(vb, tb, ta)
            return True
        if va is not None:
            if ta is None:
                self.bind(va, tb)
                return True
            self.reconcile(va, ta, tb)
            return True
        if vb is not None:
            if tb is None:
                self.bind(vb, ta)
                return True
            self.reconcile(vb, tb, ta)
            return True
        ok = self.match_cons(ta, tb)
        if not ok and self.phase == 'B':
            self.conflicts.append((self.current_node, f"{show(ta)} vs {show(tb)}"))
        return ok

    def reconcile(self, r: TVar, cur: TCon, other: TCon):
        """Class r has value `cur` and must also be `other`."""
        if cur is other or other is UNK or cur is DYN:
            return
        if cur is UNK:
            return
        if other is DYN:
            self.bind(r, DYN)
            return
        if cur is NIL and nullable(other):
            self.bind(r, other)           # None upgraded to a nullable type
            return
        if other is NIL and nullable(cur):
            return
        if cur.name == other.name and len(cur.args) == len(other.args):
            ok = True
            for x, y in zip(cur.args, other.args):
                if not self.unify(x, y):
                    ok = False
            if not ok:
                self.bind(r, DYN)
            return
        self.bind(r, DYN)                 # conflict: the value is dynamically typed

    def match_cons(self, ta: TCon, tb: TCon) -> bool:
        if ta is tb or ta is DYN or tb is DYN or ta is UNK or tb is UNK:
            return True
        if (ta is NIL and nullable(tb)) or (tb is NIL and nullable(ta)):
            return True
        if ta.name != tb.name or len(ta.args) != len(tb.args):
            return False
        ok = True
        for x, y in zip(ta.args, tb.args):
            if not self.unify(x, y):
                ok = False
        return ok

    def flow(self, src, dst):
        """dst receives src's values (one way): dst is unified with src's type
        constructor once it is known, never with src's variable, so widening the
        destination later (a variable assigned something else) leaves src alone."""
        v = resolve(src)
        if isinstance(v, TVar):
            self.add(FlowC(src, dst))
        else:
            self.unify(dst, v)

    # ── constraints ─────────────────────────────────────────────────────────
    def add(self, c: 'Constraint'):
        self.track(c)
        c.attempt(self)
        self.run()

    def track(self, c: 'Constraint'):
        if self.sink is not None:
            self.sink.append(c)
        if self.phase == 'B':
            self.all.append(c)

    def run(self):
        if self.running:
            return
        self.running = True
        try:
            while self.ready:
                c = self.ready.pop()
                if not c.done:
                    c.attempt(self)
        finally:
            self.running = False


# ═══════════════════════════════════════════════════════════════════════════════
#                              DEFERRED CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

ARITH_OPS = ('+', '-', '*', '/', '%', '**')
CMP_OPS = ('<', '<=', '>', '>=')
BIT_OPS = ('&', '|', '^', '<<', '>>')


def arith_result(op: str, a: TCon, b: TCon, s: Solver):
    """Result type of `a op b` with the VM's rules, or None (always an error)."""
    if a is DYN or b is DYN:
        return DYN
    if a is UNK or b is UNK:
        return UNK
    if a.name == 'strish' or b.name == 'strish':
        return STR if op == '+' and (a is STR or b is STR) else DYN
    if a.name in NUMERIC and b.name in NUMERIC:
        return INT if (a is INT and b is INT) else FLOAT
    if op == '+':
        if a is STR or b is STR:
            return STR
        if a.name == 'array' and b.name == 'array':
            s.unify(a.args[0], b.args[0])
            return a
        return None
    if op == '*':
        if a is STR and b is INT:
            return STR
        if a.name == 'array' and b is INT:
            return a
    return None


class Constraint:
    """A deferred typing rule. solve() returns None when done, else a TVar to wait on."""
    done = False
    report = True
    reportable = True               # may this kind of rule produce E-TYPE at all?
    node = None
    ctx = None
    error: Optional[str] = None     # message when the rule found a definite error

    def attempt(self, s: Solver):
        if self.done:
            return
        waiting = self.solve(s)
        if waiting is None:
            self.done = True
        else:
            s.wait(waiting, self)

    def solve(self, s: Solver):
        raise NotImplementedError

    def terms(self) -> list:
        return []

    def copy(self, sub) -> 'Constraint':
        raise NotImplementedError

    def outputs(self) -> list:
        return []

    def fail(self, s: Solver, message: str):
        """Record a (possible) definite error; checked again once types settle."""
        self.error = message
        if self.ctx is not None and self.report and self.reportable and self.ctx.phase == 'E':
            self.ctx.infer.suspects.append(self)


class Arith(Constraint):
    def __init__(self, op, a, b, r, node=None):
        self.op, self.a, self.b, self.r, self.node = op, a, b, r, node

    def terms(self):
        return [self.a, self.b, self.r]

    def outputs(self):
        return [self.r]

    def copy(self, sub):
        return Arith(self.op, sub(self.a), sub(self.b), sub(self.r))

    def solve(self, s):
        a, b = resolve(self.a), resolve(self.b)
        if self.op == '+' and (a is STR or b is STR):
            s.unify(self.r, STR)
            return None
        if isinstance(a, TVar):
            return a
        if isinstance(b, TVar):
            return b
        rt = arith_result(self.op, a, b, s)
        if rt is None:
            self.fail(s, f"unsupported operand types for {self.op}: {show(a)} and {show(b)}")
            rt = DYN
        s.unify(self.r, rt)
        return None

    def still_wrong(self):
        a, b = resolve(self.a), resolve(self.b)
        if not (isinstance(a, TCon) and isinstance(b, TCon)):
            return False
        return arith_result(self.op, a, b, Solver('check')) is None


class Compare(Constraint):
    """`<`, `<=`, `>`, `>=`: numbers with numbers or strings with strings."""

    reportable = False      # the VMs report it at run time (not an E-TYPE category)

    def __init__(self, op, a, b, node=None):
        self.op, self.a, self.b, self.node = op, a, b, node

    def terms(self):
        return [self.a, self.b]

    def copy(self, sub):
        return Compare(self.op, sub(self.a), sub(self.b))

    @staticmethod
    def wrong(a, b):
        if a in (DYN, UNK) or b in (DYN, UNK) or a.name == 'strish' or b.name == 'strish':
            return False
        if a.name in NUMERIC and b.name in NUMERIC:
            return False
        return not (a is STR and b is STR)

    def solve(self, s):
        a, b = resolve(self.a), resolve(self.b)
        if isinstance(a, TVar):
            return a
        if isinstance(b, TVar):
            return b
        if self.wrong(a, b):
            self.fail(s, f"cannot compare {show(a)} and {show(b)} with {self.op}")
        return None

    def still_wrong(self):
        a, b = resolve(self.a), resolve(self.b)
        return isinstance(a, TCon) and isinstance(b, TCon) and self.wrong(a, b)


class Bitwise(Constraint):
    reportable = False      # the VMs report it at run time (not an E-TYPE category)

    def __init__(self, op, a, b, node=None):
        self.op, self.a, self.b, self.node = op, a, b, node

    def terms(self):
        return [self.a, self.b]

    def copy(self, sub):
        return Bitwise(self.op, sub(self.a), sub(self.b))

    @staticmethod
    def wrong(t):
        return t not in (INT, DYN, UNK)

    def solve(self, s):
        for t in (resolve(self.a), resolve(self.b)):
            if isinstance(t, TVar):
                return t
        a, b = resolve(self.a), resolve(self.b)
        if self.wrong(a) or self.wrong(b):
            self.fail(s, f"bitwise operator {self.op} needs integers, got {show(a)} and {show(b)}")
        return None

    def still_wrong(self):
        a, b = resolve(self.a), resolve(self.b)
        return isinstance(a, TCon) and isinstance(b, TCon) and (self.wrong(a) or self.wrong(b))


class Unary(Constraint):
    """`-x` (int/float) and `~x` (int)."""

    reportable = False      # the VMs report it at run time (not an E-TYPE category)

    def __init__(self, op, a, r, node=None):
        self.op, self.a, self.r, self.node = op, a, r, node

    def terms(self):
        return [self.a, self.r]

    def outputs(self):
        return [self.r]

    def copy(self, sub):
        return Unary(self.op, sub(self.a), sub(self.r))

    def ok(self, a):
        if a in (DYN, UNK):
            return True
        return a is INT or (self.op == '-' and a is FLOAT)

    def solve(self, s):
        a = resolve(self.a)
        if isinstance(a, TVar):
            return a
        if not self.ok(a):
            self.fail(s, f"cannot apply unary {self.op} to {show(a)}")
            s.unify(self.r, DYN)
        else:
            s.unify(self.r, a)
        return None

    def still_wrong(self):
        a = resolve(self.a)
        return isinstance(a, TCon) and not self.ok(a)


class IndexC(Constraint):
    """obj[idx] -> r (also used for assignments obj[idx] = r)."""

    reportable = False      # the VMs report it at run time (not an E-TYPE category)

    def __init__(self, obj, idx, r, node=None, store=False):
        self.obj, self.idx, self.r, self.node, self.store = obj, idx, r, node, store

    def terms(self):
        return [self.obj, self.idx, self.r]

    def outputs(self):
        return [] if self.store else [self.r]

    def copy(self, sub):
        return IndexC(sub(self.obj), sub(self.idx), sub(self.r), store=self.store)

    def solve(self, s):
        o = resolve(self.obj)
        if isinstance(o, TVar):
            return o
        if o.name == 'array':
            if self.store:
                s.flow(self.r, o.args[0])
            else:
                s.unify(self.r, o.args[0])
            i = resolve(self.idx)
            if isinstance(i, TCon) and i not in (INT, DYN, UNK):
                self.fail(s, f"array index must be an integer, got {show(i)}")
        elif o is STR:
            if self.store:
                self.fail(s, "strings are immutable")
            else:
                s.unify(self.r, INT)
        elif o.name == 'map':
            if self.store:                  # a lookup never changes which keys are stored
                s.flow(self.idx, o.args[0])
                s.flow(self.r, o.args[1])
            else:
                s.unify(self.r, o.args[1])
        elif o in (DYN, UNK):
            if not self.store:
                s.unify(self.r, o)
        elif is_struct(o) or is_adt(o) or o.name in ('option', 'result'):
            if not self.store:
                s.unify(self.r, DYN)
        else:
            self.fail(s, f"{show(o)} cannot be indexed")
            if not self.store:
                s.unify(self.r, DYN)
        return None

    def still_wrong(self):
        o = resolve(self.obj)
        if not isinstance(o, TCon):
            return False
        if o.name == 'array':
            i = resolve(self.idx)
            return isinstance(i, TCon) and i not in (INT, DYN, UNK)
        if o is STR:
            return self.store
        return o.name in ('int', 'float', 'bool', 'nil', 'fn')


class IterC(Constraint):
    """`for x in obj`: the element type."""

    reportable = False      # the VMs report it at run time (not an E-TYPE category)

    def __init__(self, obj, elem, node=None):
        self.obj, self.elem, self.node = obj, elem, node

    def terms(self):
        return [self.obj, self.elem]

    def outputs(self):
        return [self.elem]

    def copy(self, sub):
        return IterC(sub(self.obj), sub(self.elem))

    def solve(self, s):
        o = resolve(self.obj)
        if isinstance(o, TVar):
            return o
        if o.name == 'array':
            s.flow(o.args[0], self.elem)
        elif o is STR:
            s.unify(self.elem, INT)
        elif o.name == 'map':
            s.flow(o.args[0], self.elem)
        elif o in (DYN, UNK):
            s.unify(self.elem, o)
        else:
            self.fail(s, f"{show(o)} is not iterable")
            s.unify(self.elem, DYN)
        return None

    def still_wrong(self):
        o = resolve(self.obj)
        return isinstance(o, TCon) and o.name in ('int', 'float', 'bool', 'nil', 'fn')


class UnwrapC(Constraint):
    """`e?`: Option<T> / Result<T, E> -> T (other values pass through)."""

    def __init__(self, a, r, node=None):
        self.a, self.r, self.node = a, r, node

    def terms(self):
        return [self.a, self.r]

    def outputs(self):
        return [self.r]

    def copy(self, sub):
        return UnwrapC(sub(self.a), sub(self.r))

    def solve(self, s):
        a = resolve(self.a)
        if isinstance(a, TVar):
            return a
        if a.name in ('option', 'result'):
            s.unify(self.r, a.args[0])
        elif a is NIL:
            pass                        # always returns/panics: no value
        else:
            s.unify(self.r, a)
        return None


class AbsC(Constraint):
    def __init__(self, a, r):
        self.a, self.r = a, r

    def terms(self):
        return [self.a, self.r]

    def outputs(self):
        return [self.r]

    def copy(self, sub):
        return AbsC(sub(self.a), sub(self.r))

    def solve(self, s):
        a = resolve(self.a)
        if isinstance(a, TVar):
            return a
        s.unify(self.r, a if a in (INT, FLOAT, UNK) else DYN)
        return None


class FoldC(Constraint):
    """min/max/sum over arguments or one array: the result type."""

    def __init__(self, kind, args, r):
        self.kind, self.args, self.r = kind, list(args), r

    def terms(self):
        return self.args + [self.r]

    def outputs(self):
        return [self.r]

    def copy(self, sub):
        return FoldC(self.kind, [sub(a) for a in self.args], sub(self.r))

    def solve(self, s):
        vals = [resolve(a) for a in self.args]
        for v in vals:
            if isinstance(v, TVar):
                return v
        if len(vals) == 1:
            v = vals[0]
            if v.name == 'array':
                e = resolve(v.args[0])
                if isinstance(e, TVar):
                    return e
                vals = [e]
            elif self.kind == 'sum':
                s.unify(self.r, arith_result('+', INT, v, s) or DYN)
                return None
        if self.kind == 'sum':
            acc = INT
            for v in vals:
                acc = arith_result('+', acc, v, s) or DYN
            s.unify(self.r, acc)
            return None
        first = vals[0] if vals else UNK
        if all(v is first for v in vals):
            s.unify(self.r, first)
        else:
            s.unify(self.r, DYN)        # min(1, 2.5) is an int or a float at run time
        return None


class CallValueC(Constraint):
    """Calling a value (closure / function reference)."""

    reportable = False      # the VMs report it at run time (not an E-TYPE category)

    def __init__(self, f, args, r, node=None):
        self.f, self.args, self.r, self.node = f, list(args), r, node

    def terms(self):
        return [self.f, self.r] + self.args

    def outputs(self):
        return [self.r]

    def copy(self, sub):
        return CallValueC(sub(self.f), [sub(a) for a in self.args], sub(self.r))

    def solve(self, s):
        f = resolve(self.f)
        if isinstance(f, TVar):
            # an unknown callee: assume a function of these arguments
            s.unify(f, fnt(self.r, self.args))
            return None
        if f.name == 'fn':
            params = f.args[1:]
            if len(params) != len(self.args):
                self.fail(s, f"this function takes {len(params)} argument(s), got {len(self.args)}")
                s.unify(self.r, DYN)
                return None
            for p, a in zip(params, self.args):
                s.flow(a, p)
            s.unify(self.r, f.args[0])
        elif f in (DYN, UNK):
            s.unify(self.r, f)
        else:
            self.fail(s, f"{show(f)} is not callable")
            s.unify(self.r, DYN)
        return None

    def still_wrong(self):
        f = resolve(self.f)
        if not isinstance(f, TCon):
            return False
        if f.name == 'fn':
            return len(f.args) - 1 != len(self.args)
        return f not in (DYN, UNK)


class FieldC(Constraint):
    """obj.name (read, or write when store=True)."""

    def __init__(self, infer, obj, name, r, node=None, store=False):
        self.infer, self.obj, self.name, self.r, self.node, self.store = infer, obj, name, r, node, store

    def terms(self):
        return [self.obj, self.r]

    def outputs(self):
        return [] if self.store else [self.r]

    def copy(self, sub):
        return FieldC(self.infer, sub(self.obj), self.name, sub(self.r), store=self.store)

    def solve(self, s):
        o = resolve(self.obj)
        if isinstance(o, TVar):
            return o
        if is_struct(o):
            info = self.infer.structs.get(nominal_name(o))
            ft = info.field_type(self.name, s.phase) if info else None
            if ft is None:
                self.fail(s, f"struct '{nominal_name(o)}' has no field '{self.name}'")
                if not self.store:
                    s.unify(self.r, DYN)
            elif self.store:
                if self.ctx is not None:
                    self.ctx.check(info.ann.get(self.name), self.r, self.node,
                                   f"field '{self.name}' of '{nominal_name(o)}'", self.report,
                                   getattr(self.node, 'value', None))
                s.flow(self.r, ft)
            else:
                s.unify(self.r, ft)
        elif o.name in ('option', 'result') and self.name in ('value', 'error') and not self.store:
            s.unify(self.r, o.args[0] if self.name == 'value' else
                    (o.args[1] if o.name == 'result' else DYN))
        elif o in (DYN, UNK):
            if not self.store:
                s.unify(self.r, o)
        elif o.name in ('int', 'float', 'bool', 'str', 'nil', 'array', 'map', 'fn'):
            if not self.store:
                s.unify(self.r, DYN)
        elif not self.store:
            s.unify(self.r, DYN)
        return None

    def still_wrong(self):
        o = resolve(self.obj)
        if not isinstance(o, TCon):
            return False
        if is_struct(o):
            info = self.infer.structs.get(nominal_name(o))
            return info is not None and info.field_type(self.name, 'A') is None
        return False


class MethodC(Constraint):
    """obj.name(args...): user methods (static dispatch on the receiver type)
    and the builtin methods of str / array / map / Option / Result."""

    def __init__(self, ctx, obj, name, args, r, node=None):
        self.ctx, self.obj, self.name, self.args, self.r, self.node = ctx, obj, name, list(args), r, node
        self.infer = ctx.infer if ctx is not None else None

    def terms(self):
        return [self.obj, self.r] + self.args

    def outputs(self):
        return [self.r]

    def copy(self, sub):
        c = MethodC(None, sub(self.obj), self.name, [sub(a) for a in self.args], sub(self.r))
        c.infer = self.infer
        return c

    def solve(self, s):
        o = resolve(self.obj)
        if isinstance(o, TVar):
            return o
        return self.infer.solve_method(s, self, o)

    def still_wrong(self):
        o = resolve(self.obj)
        if not isinstance(o, TCon) or not is_struct(o):
            return False
        return self.infer.method_missing(o, self.name)


class FlowC(Constraint):
    """See Solver.flow."""

    def __init__(self, src, dst, node=None):
        self.src, self.dst, self.node = src, dst, node

    def terms(self):
        return [self.src, self.dst]

    def outputs(self):
        return [self.dst]

    def copy(self, sub):
        return FlowC(sub(self.src), sub(self.dst))

    def solve(self, s):
        v = resolve(self.src)
        if isinstance(v, TVar):
            return v
        s.unify(self.dst, v)
        return None


class JoinC(Constraint):
    """The value of retry/timeout: the body's or the fallback's. A string
    fallback for a scalar body (`retry ... else "failed: " + error`) gives a
    `strish` value: printable and concatenable, nothing else is known."""

    def __init__(self, a, b, r, node=None):
        self.a, self.b, self.r, self.node = a, b, r, node

    def terms(self):
        return [self.a, self.b, self.r]

    def outputs(self):
        return [self.r]

    def copy(self, sub):
        return JoinC(sub(self.a), sub(self.b), sub(self.r))

    def solve(self, s):
        a, b = resolve(self.a), resolve(self.b)
        if isinstance(a, TVar) and isinstance(b, TVar):
            return a
        if isinstance(a, TCon) and isinstance(b, TCon) and a is not b:
            pair = (a, b) if b is STR else (b, a)
            if pair[1] is STR and pair[0] in (INT, FLOAT, BOOL):
                s.unify(self.r, TCon('strish', (pair[0],)))
                return None
            if a.name == 'strish' and b in (STR, a.args[0]):
                s.unify(self.r, a)
                return None
            if b.name == 'strish' and a in (STR, b.args[0]):
                s.unify(self.r, b)
                return None
        s.flow(self.a, self.r)
        s.flow(self.b, self.r)
        return None


class CheckC(Constraint):
    """An annotation check made when the value type is known (E-TYPE).
    `converts`: the VMs convert the value here (declarations, parameters,
    returns): a string is then only definitely wrong for a number when it is a
    literal that does not parse as one (`let x: int = "a"`; `f("12")` is fine)."""

    def __init__(self, declared, value, node, what, value_node=None, converts=False):
        self.declared, self.value, self.node, self.what = declared, value, node, what
        self.value_node, self.converts = value_node, converts
        self.argument = False       # call arguments: a failing conversion is the callee's run-time error

    def terms(self):
        return [self.declared, self.value]

    def copy(self, sub):
        return CheckC(sub(self.declared), sub(self.value), None, self.what, None, self.converts)

    def mismatch(self) -> bool:
        d, v = resolve(self.declared), resolve(self.value)
        if not (isinstance(d, TCon) and isinstance(v, TCon)):
            return False
        if d.name in NUMERIC and v is STR and self.converts:
            vn = self.value_node
            return not self.argument and isinstance(vn, til.StringLit) and not parses_as(vn.value, d)
        return kind_mismatch(d, v)

    def solve(self, s):
        v = resolve(self.value)
        if isinstance(v, TVar):
            return v
        if self.mismatch():
            self.fail(s, self.message())
        return None

    def message(self):
        v = resolve(self.value)
        if v is STR and self.converts and isinstance(self.value_node, til.StringLit) and \
                resolve(self.declared).name in NUMERIC:
            return (f"{self.what} is {show(self.declared)}, but the value is the string "
                    f"{self.value_node.value!r}, which is not a number")
        return f"{self.what} is {show(self.declared)}, but the value is {show(self.value)}"

    def still_wrong(self):
        return self.mismatch()


def parses_as(text: str, t) -> bool:
    """Would the VMs convert this string to the number type t?"""
    try:
        if t is INT:
            int(text.strip(' \t\n\r\x0b\x0c'))
        else:
            float(text.strip(' \t\n\r\x0b\x0c'))
        return True
    except ValueError:
        return False


def kind_mismatch(d, v) -> bool:
    """Can a value of (resolved) type v never satisfy the declared type d?
    Numbers convert into each other (the VMs cast), None fits references."""
    if not (isinstance(d, TCon) and isinstance(v, TCon)):
        return False
    if d in (DYN, UNK) or v in (DYN, UNK) or v.name == 'strish':
        return False
    if d.name in NUMERIC:
        return v.name not in ('int', 'float', 'bool', 'nil')
    if v is NIL:
        return not nullable(d) and d is not BOOL
    if d is STR or d is BOOL:
        return v is not d
    if is_struct(d) or is_adt(d):
        return v.name != d.name
    if d.name in ('array', 'map', 'option', 'result', 'fn'):
        return v.name != d.name
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#                         PROGRAM INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

INT_NAMES = {'int', 'i8', 'i16', 'i32', 'i64', 'uint', 'u8', 'u16', 'u32', 'u64', 'char',
             'isize', 'usize'}

NATIVE_RESULT = {
    'print': NIL, 'println': NIL, 'seed': NIL, 'sleep_ms': NIL, 'write_file': NIL,
    'len': INT, 'int': INT, 'ord': INT, 'clock_ms': INT,
    'str': STR, 'input': STR, 'chr': STR, 'type_of': STR, 'format': STR, 'read_file': STR,
    'bytes_to_str': STR,
    'float': FLOAT, 'sqrt': FLOAT, 'sin': FLOAT, 'cos': FLOAT, 'tan': FLOAT, 'log': FLOAT,
    'exp': FLOAT, 'floor': FLOAT, 'ceil': FLOAT, 'round': FLOAT, 'pow': FLOAT,
    'bool': BOOL, 'file_exists': BOOL,
    'screen_init': NIL, 'screen_clear': NIL, 'draw_pixel': NIL, 'draw_line': NIL,
    'draw_rect': NIL, 'fill_rect': NIL, 'draw_circle': NIL, 'draw_text': NIL,
    'screen_show': NIL, 'get_pixel': INT, 'key_pressed': INT, 'wait_key': INT,
    'screen_width': INT, 'screen_height': INT, 'energy_mode': INT,
}


class StructInfo:
    def __init__(self, sdef):
        self.name = sdef.name
        self.sdef = sdef
        self.order: List[str] = [f.name for f in sdef.fields]
        self.ann: Dict[str, TCon] = {}          # declared field types
        self.flow: Dict[str, TVar] = {}         # what is really stored (phase A)
        self.frozen: Dict[str, TCon] = {}       # settled types (phase B)
        self.methods: Dict[str, str] = {}       # instance methods: name -> function key
        self.statics: Dict[str, str] = {}       # Name.f(...) functions without self

    def field_type(self, name: str, phase: str):
        if phase in ('B', 'E'):
            return self.frozen.get(name)
        return self.flow.get(name)


class AdtInfo:
    def __init__(self, edef):
        self.name = edef.name
        self.edef = edef
        self.variants: Dict[str, List[TCon]] = {}       # declared payload types
        self.flow: Dict[str, List[TVar]] = {}
        self.frozen: Dict[str, List[TCon]] = {}

    def payload(self, variant: str, phase: str):
        return (self.frozen if phase in ('B', 'E') else self.flow).get(variant)


class GlobalInfo:
    def __init__(self, decl):
        self.decl = decl
        self.type = None
        self.declared = None
        self.frozen = None

    def type_for(self, phase: str):
        return self.frozen if phase in ('B', 'E') else self.type


class Sig:
    __slots__ = ('params', 'modes', 'ret', 'ret_mode', 'tparams')

    def __init__(self, params, modes, ret, ret_mode, tparams=None):
        self.params, self.modes, self.ret, self.ret_mode = params, modes, ret, ret_mode
        self.tparams = tparams or {}


class Scheme:
    """A generalised function type: quantified variables + open constraints."""

    def __init__(self, quantified, sig: Sig, templates):
        self.q = quantified
        self.sig = sig
        self.templates = templates

    def instantiate(self, s: Solver):
        mapping: Dict[TVar, TVar] = {}

        def sub(t):
            t = resolve(t)
            if isinstance(t, TVar):
                if t in self.q:
                    v = mapping.get(t)
                    if v is None:
                        v = mapping[t] = TVar()
                    return v
                return UNK if s.phase in ('B', 'E') else t
            if not t.args:
                return t
            return TCon(t.name, [sub(a) for a in t.args])
        params = [sub(p) for p in self.sig.params]
        ret = sub(self.sig.ret)
        if self.templates:
            copies = []
            for c in self.templates:
                cc = c.copy(sub)
                cc.report = False
                copies.append(cc)
            for cc in copies:
                s.track(cc)
                cc.attempt(s)
            s.run()
        return params, ret


class FuncInfo:
    def __init__(self, key: str, fdef, owner: Optional[str]):
        self.key = key
        self.fdef = fdef
        self.owner = owner
        self.has_self = bool(fdef.params) and fdef.params[0].name == 'self'
        self.generic_names: List[str] = []
        for attr in fdef.attributes or []:
            if attr.startswith('generic:'):
                self.generic_names += [t.strip() for t in attr[len('generic:'):].split(',') if t.strip()]
        self.mono: Optional[Sig] = None
        self.scheme: Optional[Scheme] = None

    @property
    def name(self) -> str:
        return self.fdef.name


def children(node):
    """Direct AST children (like til.iter_children, but also inside dicts such as
    StructInit.fields)."""
    if not isinstance(node, til.ASTNode):
        return
    for value in vars(node).values():
        if isinstance(value, til.ASTNode):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, til.ASTNode):
                    yield item
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, til.ASTNode):
                    yield item
                elif isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, til.ASTNode):
                            yield sub


def walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        if n is None:
            continue
        yield n
        stack.extend(reversed(list(children(n))))


def falls_through(body) -> bool:
    """Can control reach the end of this statement list? (conservative: True)"""
    stmts = body.statements if isinstance(body, til.Block) else [body]
    for st in stmts:
        if not _stmt_falls_through(st):
            return False
    return True


def _stmt_falls_through(st) -> bool:
    if isinstance(st, til.Return):
        return False
    if isinstance(st, til.ExprStmt) and isinstance(st.expr, til.Call) and \
            isinstance(st.expr.func, til.Identifier) and st.expr.func.name in ('panic', 'exit'):
        return False
    if isinstance(st, til.If):
        if st.else_body is None:
            return True
        return (falls_through(st.then_body) or any(falls_through(b) for _c, b in st.elifs) or
                falls_through(st.else_body))
    if isinstance(st, til.Loop):
        return _has_break(st.body)
    if isinstance(st, til.While) and isinstance(st.condition, til.BoolLit) and st.condition.value:
        return _has_break(st.body)
    if isinstance(st, til.Block):
        return falls_through(st)
    if isinstance(st, til.IntentBlock):
        return falls_through(st.body)
    if isinstance(st, til.MatchExpr):
        if not any(_irrefutable(arm[0]) and (len(arm) < 3 or arm[1] is None) for arm in st.arms):
            return True
        return any(falls_through(arm[-1]) if isinstance(arm[-1], til.Block) else True for arm in st.arms)
    return True


def _has_break(body) -> bool:
    stack = [body]
    while stack:
        n = stack.pop()
        if isinstance(n, til.Break):
            return True
        if isinstance(n, (til.Loop, til.While, til.For, til.Lambda)):
            continue        # a break in a nested loop leaves that loop only
        stack.extend(children(n))
    return False


def _irrefutable(pat) -> bool:
    return isinstance(pat, til.Identifier) and pat.name != 'NULL'


# ═══════════════════════════════════════════════════════════════════════════════
#                     FUNCTION BODIES: STATEMENTS AND EXPRESSIONS
# ═══════════════════════════════════════════════════════════════════════════════

class Local:
    __slots__ = ('type', 'mutable', 'declared')

    def __init__(self, t, mutable=True, declared=None):
        self.type, self.mutable, self.declared = t, mutable, declared


class FuncCtx:
    """Inference of one function body (phase A: generic, phase B: one instance)."""

    def __init__(self, infer: 'TypeInfer', s: Solver, info: Optional[FuncInfo], phase: str,
                 report: bool, scc=None):
        self.infer = infer
        self.s = s
        self.info = info
        self.phase = phase
        self.report = report
        self.scc = scc or set()
        self.guard = 0
        self.scopes: List[Dict[str, Local]] = [{}]
        self.types: Dict[int, Any] = {}         # id(expression) -> type term
        self.decls: Dict[int, Any] = {}         # id(VarDecl / For / comprehension) -> variable type
        self.calls: Dict[int, Tuple[str, list]] = {}   # id(call or reference) -> (function key, params)
        self.open_matches: Set[int] = set()     # value matches that may match nothing
        self.ret = None
        self.ret_mode = None
        self.tparams: Dict[str, Any] = {}

    # ── helpers ─────────────────────────────────────────────────────────────
    def fresh(self) -> TVar:
        return TVar()

    def unify(self, a, b, node=None):
        self.s.current_node = node
        return self.s.unify(a, b)

    def flow(self, src, dst, node=None):
        self.s.current_node = node
        self.s.flow(src, dst)

    def reporting(self) -> bool:
        return self.report and self.guard == 0

    def add(self, c: Constraint, report=None):
        c.ctx = self
        c.report = self.reporting() if report is None else report
        self.s.add(c)

    def check(self, declared, value, node, what: str, report=None, value_node=None, converts=False,
              argument=False):
        """E-TYPE if the value can never be of the declared type (phase E only)."""
        if report is None:
            report = self.reporting()
        if self.phase == 'E' and report and declared is not None and not isinstance(declared, TVar):
            c = CheckC(declared, value, node, what, value_node, converts)
            c.argument = argument
            self.add(c, report)

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def declare(self, name: str, t, mutable=True, declared=None):
        self.scopes[-1][name] = Local(t, mutable, declared)

    def lookup(self, name: str) -> Optional[Local]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def ann(self, t):
        return self.infer.ann(t, self.tparams)

    def instantiate(self, key: str):
        info = self.infer.funcs[key]
        if info.scheme is None:              # same SCC (phase A): monomorphic recursion
            return list(info.mono.params), info.mono.ret, info.mono.modes
        if self.phase == 'E' and self.infer.is_generic(key):
            return self.infer.opaque(key)
        params, ret = info.scheme.instantiate(self.s)
        return params, ret, info.scheme.sig.modes

    def bind_args(self, key: str, params, modes, args, node, skip_self=False, report=None, arg_nodes=None):
        fdef = self.infer.funcs[key].fdef
        names = [p.name for p in fdef.params]
        offset = 1 if skip_self else 0
        for i, at in enumerate(args):
            k = i + offset
            if k >= len(params):
                break
            mode = modes[k]
            what = f"argument '{names[k]}' of '{key}'"
            an = arg_nodes[i] if arg_nodes is not None and i < len(arg_nodes) else None
            if mode == 'num':
                self.check(params[k], at, node, what, report, an, converts=True, argument=True)
            elif mode == 'ann':
                self.check(params[k], at, node, what, report, an, converts=True, argument=True)
                self.flow(at, params[k], node)
            else:
                self.flow(at, params[k], node)

    # ── statements ──────────────────────────────────────────────────────────
    def block(self, body):
        self.push()
        for st in (body.statements if isinstance(body, til.Block) else [body]):
            self.stmt(st)
        self.pop()

    def stmt(self, n):
        t = type(n)
        m = getattr(self, 'st_' + t.__name__, None)
        if m is not None:
            m(n)
        elif t in (til.FuncDef, til.StructDef, til.EnumDef, til.ImplBlock, til.TraitDef,
                   til.TypeAlias, til.Import, til.CapabilityType, til.Break, til.Continue):
            pass
        else:
            self.expr(n)

    def st_ExprStmt(self, n):
        self.expr(n.expr)

    def st_Block(self, n):
        self.block(n)

    def st_VarDecl(self, n):
        declared = self.ann(n.type_ann) if n.type_ann is not None else None
        t = self.fresh()
        if isinstance(declared, TVar):      # `let y: T = ...` inside a generic function
            self.unify(t, declared, n)
            declared = None
        if n.value is not None:
            vt = self.expr(n.value)
            if declared is not None:
                self.check(declared, vt, n, f"'{n.name}'", value_node=n.value, converts=True)
                self.s.bind(t, declared)
                if declared.name not in NUMERIC:
                    self.flow(vt, t, n)      # numbers are converted at the declaration
            else:
                self.flow(vt, t, n)
        elif declared is not None:
            self.s.bind(t, declared)        # zero value of the declared type
        self.declare(n.name, t, n.mutable, declared)
        self.decls[id(n)] = t

    def st_GlobalInit(self, n):
        decl = n.decl
        g = self.infer.globals.get(decl.name)
        vt = self.expr(decl.value)
        if g is None:
            return
        gt = g.type_for(self.phase)
        if g.declared is not None:
            self.check(g.declared, vt, n, f"'{decl.name}'", value_node=decl.value, converts=True)
            if g.declared.name in NUMERIC:
                return                      # converted (the global already has the declared type)
        self.flow(vt, gt, n)

    def st_Assignment(self, n):
        target = n.target
        if n.op == '=':
            vt = self.expr(n.value)
        else:
            cur = self.expr(target)
            rv = self.expr(n.value)
            vt = self.fresh()
            self.add(Arith(n.op[:-1], cur, rv, vt, n))
        if isinstance(target, til.Identifier):
            loc = self.lookup(target.name)
            if loc is not None:
                if loc.declared is not None:
                    self.check(loc.declared, vt, n, f"'{target.name}'", value_node=n.value)
                self.flow(vt, loc.type, n)
                self.types[id(target)] = loc.type
                return
            g = self.infer.globals.get(target.name)
            if g is not None:
                if g.declared is not None:
                    self.check(g.declared, vt, n, f"'{target.name}'", value_node=n.value)
                self.flow(vt, g.type_for(self.phase), n)
                self.types[id(target)] = g.type_for(self.phase)
            return
        if isinstance(target, til.Index):
            if n.op == '=':
                ot, it = self.expr(target.obj), self.expr(target.index)
            else:
                ot, it = self.types[id(target.obj)], self.types[id(target.index)]
            self.add(IndexC(ot, it, vt, n, store=True))
        elif isinstance(target, til.Attribute):
            ot = self.expr(target.obj) if n.op == '=' else self.types.get(id(target.obj))
            if ot is None:
                ot = self.expr(target.obj)
            self.add(FieldC(self.infer, ot, target.attr, vt, n, store=True))
        else:
            self.expr(target)

    def st_If(self, n):
        for cond, body in [(n.condition, n.then_body)] + list(n.elifs):
            self.expr(cond)
            self.block(body)
        if n.else_body is not None:
            self.block(n.else_body)

    def st_While(self, n):
        self.expr(n.condition)
        self.block(n.body)

    def st_Loop(self, n):
        self.block(n.body)

    def st_For(self, n):
        self.push()
        v = self.fresh()
        if isinstance(n.iter, til.Range):
            st = self.expr(n.iter.start)
            self.expr(n.iter.end)
            self.unify(v, st, n)
        else:
            it = self.expr(n.iter)
            self.add(IterC(it, v, n))
        self.declare(n.var, v, True)
        self.decls[id(n)] = v
        self.block(n.body)
        self.pop()

    def st_Return(self, n):
        if n.value is None:
            if self.ret_mode is None:
                self.unify(self.ret, NIL, n)
            return
        vt = self.expr(n.value)
        self.return_value(vt, n, n.value)

    def return_value(self, vt, node, value_node=None):
        what = f"the return value of '{self.info.key if self.info else '?'}'"
        if self.ret_mode == 'main':
            return                          # the VM ignores main's result
        if self.ret_mode == 'num':
            self.check(self.ret, vt, node, what, value_node=value_node, converts=True)
        elif self.ret_mode == 'ann':
            self.check(self.ret, vt, node, what, value_node=value_node, converts=True)
            self.flow(vt, self.ret, node)
        else:
            self.flow(vt, self.ret, node)

    def st_MatchExpr(self, n):
        self.match(n, False)

    def st_AssertStmt(self, n):
        self.expr(n.condition)

    def st_InvariantStmt(self, n):
        self.expr(n.condition)

    def st_IntentBlock(self, n):
        if isinstance(n.body, til.Block):
            self.block(n.body)
        else:
            self.expr(n.body)

    # ── values of blocks ────────────────────────────────────────────────────
    def block_value(self, body):
        """A block used as an expression evaluates to its last expression."""
        if not isinstance(body, til.Block):
            return self.expr(body)
        stmts = body.statements
        self.push()
        t = NIL
        for st in stmts[:-1]:
            self.stmt(st)
        if stmts:
            last = stmts[-1]
            if isinstance(last, til.ExprStmt):
                t = self.expr(last.expr)
            elif isinstance(last, til.MatchExpr):
                t = self.match(last, True)
                self.types[id(last)] = t
            elif isinstance(last, til.If) and last.else_body is not None:
                t = self.if_value(last)
            elif isinstance(last, (til.Block, til.IntentBlock)):
                t = self.expr(last)
            else:
                self.stmt(last)
        self.pop()
        return t

    def if_value(self, n):
        r = self.fresh()
        for cond, body in [(n.condition, n.then_body)] + list(n.elifs):
            self.expr(cond)
            self.flow(self.block_value(body), r, n)
        self.flow(self.block_value(n.else_body), r, n)
        self.types[id(n)] = r
        return r

    # ── pattern matching ────────────────────────────────────────────────────
    def match(self, n, want_value: bool):
        subj = self.expr(n.value)
        results = []
        for arm in n.arms:
            pat, guard, body = arm if len(arm) == 3 else (arm[0], None, arm[1])
            self.push()
            self.pattern(pat, subj)
            if guard is not None:
                self.expr(guard)
            if want_value:
                results.append(self.block_value(body) if isinstance(body, til.Block) else self.expr(body))
            elif isinstance(body, til.Block):
                self.block(body)
            else:
                self.expr(body)
            self.pop()
        if not want_value:
            return NIL
        r = self.fresh()
        for t in results:
            self.flow(t, r, n)
        if not self.exhaustive(n, subj):
            self.open_matches.add(id(n))
        return r

    def exhaustive(self, n, subj) -> bool:
        arms = [arm if len(arm) == 3 else (arm[0], None, arm[1]) for arm in n.arms]
        plain = [pat for pat, guard, _b in arms if guard is None]
        if any(_irrefutable(p) and self.binds(p) for p in plain):
            return True
        s = resolve(subj)
        names = set()
        for p in plain:
            if isinstance(p, til.Attribute) and isinstance(p.obj, til.Identifier):
                names.add(p.attr)
            elif isinstance(p, til.Call):
                f = p.func
                if all(_irrefutable(a) and self.binds(a) for a in p.args):
                    if isinstance(f, til.Identifier):
                        names.add(f.name)
                    elif isinstance(f, til.Attribute):
                        names.add(f.attr)
            elif isinstance(p, til.Identifier) and p.name == 'NULL':
                names.add('None')
            elif isinstance(p, til.BoolLit):
                names.add(str(p.value))
        if isinstance(s, TCon):
            if is_adt(s):
                info = self.infer.adts.get(nominal_name(s))
                return info is not None and set(info.variants) <= names
            if s.name == 'option':
                return {'Some', 'None'} <= names
            if s.name == 'result':
                return {'Ok', 'Err'} <= names
            if s is BOOL:
                return {'True', 'False'} <= names
        return False

    def binds(self, pat) -> bool:
        """An identifier pattern binds (rather than compares with a constant)."""
        return pat.name == '_' or self.lookup(pat.name) is not None or pat.name not in self.infer.globals

    def pattern(self, pat, subj):
        if isinstance(pat, til.Identifier):
            name = pat.name
            if name in ('_', 'NULL'):
                return
            if self.lookup(name) is None and name in self.infer.globals:
                self.expr(pat)              # compared with the constant
                return
            v = self.fresh()
            self.unify(v, subj, pat)
            self.declare(name, v, False)
            self.types[id(pat)] = v
            return
        infer = self.infer
        if isinstance(pat, til.Attribute) and isinstance(pat.obj, til.Identifier) \
                and pat.obj.name in infer.adts:
            self.unify(subj, adtt(pat.obj.name), pat)
            return
        if isinstance(pat, til.Call):
            f = pat.func
            if isinstance(f, til.Identifier) and f.name in ('Some', 'Ok', 'Err'):
                if f.name == 'Some':
                    inner = self.fresh()
                    self.unify(subj, opt(inner), pat)
                else:
                    ok, err = self.fresh(), self.fresh()
                    self.unify(subj, resultt(ok, err), pat)
                    inner = ok if f.name == 'Ok' else err
                for p in pat.args[:1]:
                    self.pattern(p, inner)
                self.types[id(pat)] = subj
                return
            if isinstance(f, til.Identifier) and f.name in infer.structs:
                info = infer.structs[f.name]
                self.unify(subj, structt(f.name), pat)
                for p, fname in zip(pat.args, info.order):
                    self.pattern(p, info.field_type(fname, self.phase))
                return
            if isinstance(f, til.Attribute) and isinstance(f.obj, til.Identifier) \
                    and f.obj.name in infer.adts:
                info = infer.adts[f.obj.name]
                self.unify(subj, adtt(f.obj.name), pat)
                payload = info.payload(f.attr, self.phase) or []
                for p, pt in zip(pat.args, payload):
                    self.pattern(p, pt)
                return
            self.expr(pat)
            return
        if isinstance(pat, (til.TupleLit, til.ArrayLit)):
            e = self.fresh()
            self.unify(subj, arr(e), pat)
            for p in pat.elements:
                self.pattern(p, e)
            return
        if isinstance(pat, til.StructInit) and pat.name in infer.structs:
            info = infer.structs[pat.name]
            self.unify(subj, structt(pat.name), pat)
            for fname, p in pat.fields.items():
                ft = info.field_type(fname, self.phase)
                if ft is not None:
                    self.pattern(p, ft)
            return
        if isinstance(pat, til.Range):
            self.expr(pat.start)
            self.expr(pat.end)
            return
        if isinstance(pat, til.BinaryOp) and pat.op == '|':
            stack = [pat]
            while stack:
                p = stack.pop()
                if isinstance(p, til.BinaryOp) and p.op == '|':
                    stack.extend([p.right, p.left])
                else:
                    self.expr(p)
            return
        self.expr(pat)

    # ── expressions ─────────────────────────────────────────────────────────
    def expr(self, n):
        m = getattr(self, 'ex_' + type(n).__name__, None)
        if m is None:
            t = self.ex_unknown(n)
        else:
            t = m(n)
        self.types[id(n)] = t
        return t

    def ex_unknown(self, n):
        """A construct this pass does not know (e.g. added by another feature):
        infer its parts without reporting anything; its own value is dynamic."""
        self.guard += 1
        try:
            for child in children(n):
                if isinstance(child, til.Block):
                    self.block(child)
                elif isinstance(child, (til.VarDecl, til.Assignment, til.If, til.While, til.For,
                                        til.Loop, til.Return, til.ExprStmt)):
                    self.stmt(child)
                else:
                    self.expr(child)
        finally:
            self.guard -= 1
        return DYN

    def ex_IntLit(self, n):
        return INT

    def ex_FloatLit(self, n):
        return FLOAT

    def ex_StringLit(self, n):
        return STR

    def ex_BoolLit(self, n):
        return BOOL

    def ex_CharLit(self, n):
        return INT

    def ex_DurationLit(self, n):
        return INT

    def ex_FStringLit(self, n):
        for part in n.parts:
            if part[0] != 'str':
                self.expr(part[1])
        return STR

    def ex_Identifier(self, n):
        name = n.name
        if name == 'NULL':
            v = self.fresh()
            self.s.bind(v, NIL)
            return v
        loc = self.lookup(name)
        if loc is not None:
            return loc.type
        g = self.infer.globals.get(name)
        if g is not None:
            return g.type_for(self.phase)
        if name in self.infer.funcs and '.' not in name:
            params, ret, _modes = self.instantiate(name)
            self.calls[id(n)] = (name, params)
            return fnt(ret, params)
        return DYN

    def ex_BinaryOp(self, n):
        op = n.op
        lt = self.expr(n.left)
        rt = self.expr(n.right)
        if op in ('and', 'or', '==', '!='):
            return BOOL
        if op in CMP_OPS:
            self.add(Compare(op, lt, rt, n))
            return BOOL
        if op in ARITH_OPS:
            r = self.fresh()
            self.add(Arith(op, lt, rt, r, n))
            return r
        if op in BIT_OPS:
            self.add(Bitwise(op, lt, rt, n))
            return INT
        return DYN

    def ex_UnaryOp(self, n):
        ot = self.expr(n.operand)
        if n.op == 'not':
            return BOOL
        if n.op in ('-', '~'):
            r = self.fresh()
            self.add(Unary(n.op, ot, r, n))
            return r
        return ot                       # & / &mut / * are identity on references

    def ex_Index(self, n):
        ot = self.expr(n.obj)
        it = self.expr(n.index)
        r = self.fresh()
        self.add(IndexC(ot, it, r, n))
        return r

    def shadowed(self, name: str) -> bool:
        return self.lookup(name) is not None

    def ex_Attribute(self, n):
        obj = n.obj
        infer = self.infer
        if isinstance(obj, til.Identifier) and not self.shadowed(obj.name):
            name = obj.name
            if name in infer.enums:
                return INT
            if name in infer.adts:
                payload = infer.adts[name].variants.get(n.attr)
                return adtt(name) if payload == [] else DYN
            if name in infer.module_names and name not in infer.globals:
                return self.expr(til.Identifier(name=n.attr, line=n.line, col=n.col))
            if name in infer.structs:
                return DYN
        ot = self.expr(obj)
        r = self.fresh()
        self.add(FieldC(infer, ot, n.attr, r, n))
        return r

    def ex_ArrayLit(self, n):
        e = self.fresh()
        for el in n.elements:
            self.flow(self.expr(el), e, el)
        return arr(e)

    ex_TupleLit = ex_ArrayLit

    def ex_DictLit(self, n):
        k, v = self.fresh(), self.fresh()
        for ke, ve in n.pairs:
            self.flow(self.expr(ke), k, ke)
            self.flow(self.expr(ve), v, ve)
        return mapt(k, v)

    def ex_Range(self, n):
        self.expr(n.start)
        self.expr(n.end)
        return arr(INT)

    def ex_StructInit(self, n):
        info = self.infer.structs.get(n.name)
        if info is None:
            for v in n.fields.values():
                self.expr(v)
            return DYN
        for fld in info.sdef.fields:
            ft = info.field_type(fld.name, self.phase)
            what = f"field '{fld.name}' of '{n.name}'"
            if fld.name in n.fields:
                vt = self.expr(n.fields[fld.name])
                declared = info.ann[fld.name]
                self.check(declared, vt, n, what, value_node=n.fields[fld.name],
                           converts=declared is FLOAT)
                if declared is not FLOAT:           # float fields are converted by the VM
                    self.flow(vt, ft, n)
            elif fld.default is not None:
                self.flow(self.expr(fld.default), ft, n)
        return structt(n.name)

    def ex_Lambda(self, n):
        self.push()
        params = []
        for pname, ptype in n.params:
            t = self.fresh()
            if ptype is not None:
                at = self.ann(ptype)
                if at is not None:
                    self.unify(t, at, n)
            self.declare(pname, t, True)
            params.append(t)
        bt = self.block_value(n.body) if isinstance(n.body, til.Block) else self.expr(n.body)
        if n.ret_type is not None:
            rt = self.ann(n.ret_type)
            if rt is not None and rt is not NIL:
                self.unify(bt, rt, n)
        self.pop()
        return fnt(bt, params)

    def ex_IfExpr(self, n):
        self.expr(n.condition)
        r = self.fresh()
        self.flow(self.expr(n.then_expr), r, n)
        self.flow(self.expr(n.else_expr) if n.else_expr is not None else NIL, r, n)
        return r

    def ex_MatchExpr(self, n):
        return self.match(n, True)

    def ex_Cast(self, n):
        et = self.expr(n.expr)
        t = n.target_type
        if isinstance(t, til.PrimitiveType):
            if t.name in ('float', 'f32', 'f64'):
                return FLOAT
            if t.name == 'str':
                return STR
            if t.name == 'bool':
                return BOOL
            return INT
        return et

    def ex_NullCheck(self, n):
        r = self.fresh()
        self.add(UnwrapC(self.expr(n.expr), r, n))
        return r

    def ex_ListComprehension(self, n):
        self.push()
        v = self.fresh()
        if isinstance(n.iter, til.Range):
            st = self.expr(n.iter.start)
            self.expr(n.iter.end)
            self.unify(v, st, n)
        else:
            self.add(IterC(self.expr(n.iter), v, n))
        self.declare(n.var, v, True)
        self.decls[id(n)] = v
        if n.condition is not None:
            self.expr(n.condition)
        et = self.expr(n.expr)
        self.pop()
        return arr(et)

    def guarded(self, n):
        self.expr(n.count) if isinstance(n, til.RetryExpr) else self.expr(n.duration)
        self.guard += 1
        try:
            bt = self.block_value(n.body)
        finally:
            self.guard -= 1
        if n.fallback is None:
            return bt
        self.push()
        self.declare('error', STR, False)
        ft = self.block_value(n.fallback)
        self.pop()
        r = self.fresh()
        self.add(JoinC(bt, ft, r, n))
        return r

    ex_RetryExpr = guarded
    ex_TimeoutExpr = guarded

    def ex_IntentBlock(self, n):
        return self.block_value(n.body)

    def ex_Block(self, n):
        return self.block_value(n)

    def ex_EffectHandler(self, n):
        return self.block_value(n.expr)

    def ex_PerformEffect(self, n):
        args = [self.expr(a) for a in n.args]
        if n.name in self.infer.funcs:
            params, ret, modes = self.instantiate(n.name)
            self.bind_args(n.name, params, modes, args, n, arg_nodes=n.args)
            self.calls[id(n)] = (n.name, params)
            return ret
        return DYN

    # ── calls ───────────────────────────────────────────────────────────────
    def ex_Call(self, n):
        f = n.func
        infer = self.infer
        if isinstance(f, til.Identifier):
            name = f.name
            if self.shadowed(name):
                ft = self.expr(f)
                return self.call_value(ft, n)
            if name in infer.funcs and '.' not in name:
                args = [self.expr(a) for a in n.args]
                params, ret, modes = self.instantiate(name)
                self.bind_args(name, params, modes, args, n, arg_nodes=n.args)
                self.calls[id(n)] = (name, params)
                return ret
            if name in infer.structs:
                info = infer.structs[name]
                for a, fname in zip(n.args, info.order):
                    self.flow(self.expr(a), info.field_type(fname, self.phase), n)
                for a in n.args[len(info.order):]:
                    self.expr(a)
                return structt(name)
            if name in ('Some', 'Ok', 'Err'):
                inner = self.expr(n.args[0]) if n.args else NIL
                for a in n.args[1:]:
                    self.expr(a)
                if name == 'Some':
                    return opt(inner)
                if name == 'Ok':
                    return resultt(inner, self.fresh())
                return resultt(self.fresh(), inner)
            if name in infer.globals:
                return self.call_value(self.expr(f), n)
            if name in til.NATIVE_ID:
                return self.call_native(name, n)
            for a in n.args:
                self.expr(a)
            return DYN
        if isinstance(f, til.Attribute):
            obj = f.obj
            meth = f.attr
            if isinstance(obj, til.Identifier) and not self.shadowed(obj.name):
                oname = obj.name
                if oname in infer.structs:
                    key = f"{oname}.{meth}"
                    args = [self.expr(a) for a in n.args]
                    if key in infer.funcs:
                        params, ret, modes = self.instantiate(key)
                        self.bind_args(key, params, modes, args, n, arg_nodes=n.args)
                        self.calls[id(n)] = (key, params)
                        return ret
                    return DYN
                if oname in infer.adts:
                    info = infer.adts[oname]
                    payload = info.payload(meth, self.phase)
                    args = [self.expr(a) for a in n.args]
                    if payload is not None:
                        for pt, at in zip(payload, args):
                            self.flow(at, pt, n)
                        return adtt(oname)
                    return DYN
                if oname in infer.module_names and oname not in infer.globals:
                    args = [self.expr(a) for a in n.args]
                    if meth in infer.funcs:
                        params, ret, modes = self.instantiate(meth)
                        self.bind_args(meth, params, modes, args, n, arg_nodes=n.args)
                        self.calls[id(n)] = (meth, params)
                        return ret
                    return DYN
                if oname.startswith('__Vec__') and meth == 'new':
                    return arr(infer.type_name(oname[len('__Vec__'):]))
                if oname.startswith('__HashMap__') and meth == 'new':
                    parts = oname[len('__HashMap__'):].split('_', 1)
                    if len(parts) == 2:
                        return mapt(infer.type_name(parts[0]), infer.type_name(parts[1]))
                    return mapt(self.fresh(), self.fresh())
            ot = self.expr(obj)
            args = [self.expr(a) for a in n.args]
            r = self.fresh()
            self.add(MethodC(self, ot, meth, args, r, n))
            return r
        ft = self.expr(f)
        return self.call_value(ft, n)

    def call_value(self, ft, n):
        args = [self.expr(a) for a in n.args]
        r = self.fresh()
        self.add(CallValueC(ft, args, r, n))
        return r

    def call_native(self, name: str, n):
        args = [self.expr(a) for a in n.args]
        if name in NATIVE_RESULT:
            return NATIVE_RESULT[name]
        r = self.fresh()
        if name == 'abs':
            self.add(AbsC(args[0] if args else UNK, r))
        elif name in ('min', 'max', 'sum'):
            self.add(FoldC(name, args, r))
        elif name == 'range':
            return arr(INT)
        elif name == 'args':
            return arr(STR)
        elif name == 'str_bytes':
            return arr(INT)
        elif name == 'random':
            return FLOAT if not args else args[0]
        elif name == 'iter':
            return args[0] if args else DYN
        # exit / panic never return: their value can be of any type
        return r


# ═══════════════════════════════════════════════════════════════════════════════
#                         WHOLE-PROGRAM INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

class Instance:
    """One monomorphic instantiation of a function, with every expression typed."""

    def __init__(self, key: str, info: Optional[FuncInfo], params, ret, ctx: FuncCtx):
        self.key = key
        self.info = info
        self.params = [deep(p, UNK) for p in params]
        self.ret = deep(ret, UNK)
        self.ctx = ctx
        self.conflicts = list(ctx.s.conflicts)
        self.c_name = ""
        self._cache: Dict[int, TCon] = {}

    def t(self, node):
        """Settled type of an expression node in this instance (UNK if untyped)."""
        k = id(node)
        t = self._cache.get(k)
        if t is None:
            term = self.ctx.types.get(k)
            t = self._cache[k] = UNK if term is None else deep(term, UNK)
        return t

    def decl(self, node):
        term = self.ctx.decls.get(id(node))
        return UNK if term is None else deep(term, UNK)

    def call(self, node):
        """(function key, settled parameter types) of a user-function call/reference."""
        target = self.ctx.calls.get(id(node))
        if target is None:
            return None
        key, params = target
        return key, [deep(p, UNK) for p in params]

    def is_open_match(self, node) -> bool:
        return id(node) in self.ctx.open_matches

    def signature(self) -> str:
        return f"fn({', '.join(show(p) for p in self.params)}) -> {show(self.ret)}"

    def locals(self) -> Dict[str, str]:
        """name -> type of every variable declared in the body (for tests/tools)."""
        out = {}
        fdef = self.info.fdef if self.info else None
        if fdef is None:
            return out
        for node in walk(fdef.body):
            if isinstance(node, (til.VarDecl, til.For)):
                name = node.name if isinstance(node, til.VarDecl) else node.var
                out[name] = show(self.decl(node))
        return out


class TypeInfer:
    """Gradual type inference for a whole (import-resolved) program."""

    MAX_INSTANCES_PER_FUNCTION = 32

    def __init__(self, program, source: str = "", filename: str = "<stdin>"):
        self.program = program
        self.reporter = til.ErrorReporter(source, filename)
        self.errors: List[str] = []
        self.suspects: List[Constraint] = []
        self.structs: Dict[str, StructInfo] = {}
        self.adts: Dict[str, AdtInfo] = {}
        self.enums: Dict[str, Dict[str, int]] = {}
        self.traits: Dict[str, Any] = {}
        self.funcs: Dict[str, FuncInfo] = {}
        self.globals: Dict[str, GlobalInfo] = {}
        self.aliases: Dict[str, Any] = {}
        self.module_names: Set[str] = set(getattr(program, 'module_names', set()) or ())
        self.solver = Solver('A')
        self.globals_ctx: Optional[FuncCtx] = None
        self.instances: Dict[Tuple[str, Tuple[str, ...]], Instance] = {}
        self._alias_depth = 0
        self.collect()

    # ── declarations ────────────────────────────────────────────────────────
    def collect(self):
        impls = []
        for st in self.program.statements:
            if isinstance(st, til.StructDef):
                self.structs[st.name] = StructInfo(st)
            elif isinstance(st, til.EnumDef):
                if any(v.fields for v in st.variants):
                    self.adts[st.name] = AdtInfo(st)
                else:
                    self.enums[st.name] = {v.name: i for i, v in enumerate(st.variants)}
            elif isinstance(st, til.TraitDef):
                self.traits[st.name] = st
            elif isinstance(st, til.ImplBlock):
                impls.append(st)
            elif isinstance(st, til.VarDecl):
                self.globals[st.name] = GlobalInfo(st)
            elif isinstance(st, til.TypeAlias):
                self.aliases[st.name] = st.type
            elif isinstance(st, til.FuncDef):
                if st.name not in self.funcs:
                    self.funcs[st.name] = FuncInfo(st.name, st, None)
        s = self.solver
        for info in self.structs.values():
            for fld in info.sdef.fields:
                declared = self.ann(fld.type) or DYN
                info.ann[fld.name] = declared
                v = TVar()
                s.bind(v, declared)
                info.flow[fld.name] = v
        for info in self.adts.values():
            for variant in info.edef.variants:
                declared = [self.ann(t) or DYN for t in variant.fields]
                info.variants[variant.name] = declared
                flows = []
                for d in declared:
                    v = TVar()
                    s.bind(v, d)
                    flows.append(v)
                info.flow[variant.name] = flows
        for impl in impls:
            sinfo = self.structs.get(impl.type_name)
            if sinfo is None:
                continue
            defined = set()
            for m in impl.methods:
                key = f"{impl.type_name}.{m.name}"
                self.funcs[key] = FuncInfo(key, m, impl.type_name)
                defined.add(m.name)
                if m.params and m.params[0].name == 'self':
                    sinfo.methods[m.name] = key
                else:
                    sinfo.statics[m.name] = key
            trait = self.traits.get(impl.trait_name) if impl.trait_name else None
            if trait is not None:
                for tm in trait.methods:
                    if tm.name in defined or not (isinstance(tm.body, til.Block) and tm.body.statements):
                        continue
                    key = f"{impl.type_name}.{tm.name}"
                    self.funcs[key] = FuncInfo(key, tm, impl.type_name)
                    if tm.params and tm.params[0].name == 'self':
                        sinfo.methods[tm.name] = key
                    else:
                        sinfo.statics[tm.name] = key
        for name, g in self.globals.items():
            decl = g.decl
            g.declared = self.ann(decl.type_ann) if decl.type_ann is not None else None
            v = TVar()
            if not decl.mutable and isinstance(decl.value, (til.IntLit, til.FloatLit, til.StringLit,
                                                            til.BoolLit, til.CharLit)):
                # inlined constant: the literal itself (no conversion)
                lit = {til.IntLit: INT, til.FloatLit: FLOAT, til.StringLit: STR,
                       til.BoolLit: BOOL, til.CharLit: INT}[type(decl.value)]
                s.bind(v, lit)
            elif decl.value is None:
                s.bind(v, g.declared if g.declared is not None else NIL)
            elif decl.deferred and g.declared is not None and g.declared.name in NUMERIC:
                s.bind(v, g.declared)       # script-mode globals are converted like locals
            g.type = v

    def type_name(self, name: str):
        """Type from a bare name (Vec<T> / HashMap<K, V> arguments)."""
        if name in til.TYPE_MAP:
            return self.ann(til.TYPE_MAP[name]) or DYN
        return self.ann(til.StructType(name)) or DYN

    def ann(self, t, tparams=None):
        """Annotation (AST type) -> type term, or None when there is none."""
        if t is None or isinstance(t, til.UnknownType):
            return None
        if isinstance(t, til.VoidType):
            return NIL
        if isinstance(t, til.PrimitiveType):
            n = t.name
            if n in INT_NAMES:
                return INT
            if n in ('float', 'f32', 'f64'):
                return FLOAT
            return {'bool': BOOL, 'str': STR, 'void': NIL}.get(n, DYN)
        if isinstance(t, til.ArrayType):
            return arr(self.ann(t.element_type, tparams) or DYN)
        if isinstance(t, til.PointerType):
            return self.ann(t.pointee, tparams)
        if isinstance(t, til.FunctionType):
            ret = self.ann(t.ret, tparams)
            return fnt(NIL if ret is None else ret, [self.ann(p, tparams) or DYN for p in t.params])
        if isinstance(t, til.OptionType):
            return opt(self.ann(t.inner, tparams) or DYN)
        if isinstance(t, til.ResultType):
            return resultt(self.ann(t.ok, tparams) or DYN, self.ann(t.err, tparams) or DYN)
        if isinstance(t, til.GenericType):
            ps = [self.ann(p, tparams) or DYN for p in t.params]
            if t.name in ('Vec', 'Array', 'List') and len(ps) == 1:
                return arr(ps[0])
            if t.name in ('HashMap', 'Map', 'Dict') and len(ps) == 2:
                return mapt(ps[0], ps[1])
            return DYN
        if isinstance(t, til.StructType):
            n = t.name
            if tparams and n in tparams:
                return tparams[n]
            if n in self.aliases and self._alias_depth < 16:
                self._alias_depth += 1
                try:
                    return self.ann(self.aliases[n], tparams)
                finally:
                    self._alias_depth -= 1
            if n in self.structs:
                return structt(n)
            if n in self.adts:
                return adtt(n)
            if n in self.enums:
                return INT
            return DYN
        return DYN

    # ── signatures ──────────────────────────────────────────────────────────
    def signature(self, info: FuncInfo, given=None, s: Optional[Solver] = None) -> Sig:
        tparams = {name: TVar() for name in info.generic_names}
        params, modes = [], []
        for i, p in enumerate(info.fdef.params):
            if p.name == 'self' and info.owner:
                t, mode = structt(info.owner), 'self'
            else:
                t = self.ann(p.type, tparams)
                if t is None:
                    t, mode = TVar(), None
                elif isinstance(t, TVar):
                    mode = 'tparam'         # annotated with an explicit type parameter
                else:
                    mode = 'num' if t.name in NUMERIC else 'ann'
            if given is not None and i < len(given):
                if s is not None:
                    s.unify(t, given[i])        # binds explicit type parameters
                t = given[i]
            params.append(t)
            modes.append(mode)
        rt = self.ann(info.fdef.ret_type, tparams)
        if info.fdef.name == 'main' and info.owner is None:
            ret, ret_mode = NIL, 'main'
        elif rt is None or rt is NIL or isinstance(rt, TVar):
            ret, ret_mode = (TVar() if rt is None or rt is NIL else rt), None
        else:
            ret, ret_mode = rt, ('num' if rt.name in NUMERIC else 'ann')
        return Sig(params, modes, ret, ret_mode, tparams)

    def infer_body(self, info: FuncInfo, sig: Sig, s: Solver, phase: str, report: bool,
                   scc=None) -> FuncCtx:
        ctx = FuncCtx(self, s, info, phase, report, scc)
        ctx.tparams = sig.tparams
        ctx.ret, ctx.ret_mode = sig.ret, sig.ret_mode
        fdef = info.fdef
        for p, t, mode in zip(fdef.params, sig.params, sig.modes):
            if mode in ('num', 'ann') or phase == 'B':
                local = TVar()          # the body may store other values in a parameter
                s.bind(local, resolve(t)) if isinstance(resolve(t), TCon) else s.unify(local, t)
                ctx.declare(p.name, local, True, resolve(t) if mode in ('num', 'ann') else None)
            else:
                ctx.declare(p.name, t, True)
        for p, t, mode in zip(fdef.params, sig.params, sig.modes):
            if p.default is not None:
                dt = ctx.expr(p.default)
                if mode == 'num':
                    ctx.check(t, dt, p.default, f"the default of '{p.name}'", value_node=p.default, converts=True)
                elif mode == 'ann':
                    ctx.check(t, dt, p.default, f"the default of '{p.name}'", value_node=p.default, converts=True)
                    ctx.unify(t, dt, p.default)
                else:
                    ctx.unify(t, dt, p.default)
        for cond in fdef.requires:
            ctx.expr(cond)
        ctx.block(fdef.body)
        if ctx.ret_mode is None and falls_through(fdef.body):
            ctx.unify(ctx.ret, NIL, fdef)
        if fdef.ensures:
            ctx.push()
            ctx.declare('result', ctx.ret, False)
            for cond in fdef.ensures:
                ctx.expr(cond)
            ctx.pop()
        return ctx

    # ── the call graph ──────────────────────────────────────────────────────
    def references(self, nodes, method_keys) -> Set[str]:
        out: Set[str] = set()
        for root_node in nodes:
            if root_node is None:
                continue
            for node in walk(root_node):
                if isinstance(node, til.Identifier) and node.name in self.funcs and '.' not in node.name:
                    out.add(node.name)
                elif isinstance(node, til.PerformEffect) and node.name in self.funcs:
                    out.add(node.name)
                elif isinstance(node, til.Call) and isinstance(node.func, til.Attribute):
                    f = node.func
                    if isinstance(f.obj, til.Identifier):
                        key = f"{f.obj.name}.{f.attr}"
                        if key in self.funcs:
                            out.add(key)
                        if f.obj.name in self.module_names and f.attr in self.funcs:
                            out.add(f.attr)
                    out.update(method_keys.get(f.attr, ()))
        return out

    def function_nodes(self, info: FuncInfo):
        fdef = info.fdef
        return ([fdef.body] + [p.default for p in fdef.params if p.default is not None] +
                list(fdef.requires) + list(fdef.ensures))

    def sccs(self) -> List[List[str]]:
        method_keys: Dict[str, List[str]] = {}
        for key in self.funcs:
            if '.' in key:
                method_keys.setdefault(key.split('.', 1)[1], []).append(key)
        graph = {key: sorted(self.references(self.function_nodes(info), method_keys))
                 for key, info in self.funcs.items()}
        graph['<globals>'] = sorted(self.references(
            [g.decl.value for g in self.globals.values() if g.decl.value is not None and not g.decl.deferred],
            method_keys))
        # iterative Tarjan: SCCs come out callees first
        index: Dict[str, int] = {}
        low: Dict[str, int] = {}
        on_stack: Set[str] = set()
        stack: List[str] = []
        out: List[List[str]] = []
        counter = 0
        for start in graph:
            if start in index:
                continue
            work = [(start, 0)]
            while work:
                v, i = work.pop()
                if i == 0:
                    index[v] = low[v] = counter
                    counter += 1
                    stack.append(v)
                    on_stack.add(v)
                succ = graph.get(v, [])
                recurse = False
                while i < len(succ):
                    w = succ[i]
                    i += 1
                    if w not in index:
                        work.append((v, i))
                        work.append((w, 0))
                        recurse = True
                        break
                    if w in on_stack:
                        low[v] = min(low[v], index[w])
                if recurse:
                    continue
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    out.append(comp)
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[v])
        return out

    # ── phase A ─────────────────────────────────────────────────────────────
    def run(self) -> 'TypeInfer':
        s = self.solver
        for comp in self.sccs():
            if comp == ['<globals>']:
                self.infer_globals()
                continue
            infos = [self.funcs[k] for k in comp]
            s.sink = []
            for info in infos:
                info.mono = self.signature(info)
            for info in infos:
                self.infer_body(info, info.mono, s, 'A', True, set(comp))
            s.run()
            self.generalize(infos, s.sink)
            s.sink = None
        s.run()
        self.freeze()
        self.phase_e()
        self.collect_errors()
        return self

    # ── phase E: definite errors ────────────────────────────────────────────
    def opaque(self, key: str):
        """A generic function seen from phase E: its annotated parameters are
        checked, its result is dynamic (it depends on the instantiation)."""
        sig = self.funcs[key].scheme.sig
        params = []
        for p, mode in zip(sig.params, sig.modes):
            t = deep(p, None)
            params.append(t if mode in ('num', 'ann', 'self') and not contains(t, ('var',)) else DYN)
        return params, DYN, sig.modes

    def phase_e(self):
        """Re-infer every body with unannotated parameters dynamic: a type error
        found now does not depend on how a generic function is instantiated."""
        for key, info in self.funcs.items():
            s = Solver('E')
            sig = self.signature(info)
            for t, mode in zip(sig.params, sig.modes):
                if mode is None and isinstance(resolve(t), TVar):
                    s.bind(resolve(t), DYN)
            for tv in sig.tparams.values():
                if isinstance(resolve(tv), TVar):
                    s.bind(resolve(tv), DYN)
            self.infer_body(info, sig, s, 'E', True)
            s.run()
        s = Solver('E')
        ctx = FuncCtx(self, s, None, 'E', True)
        for g in self.globals.values():
            decl = g.decl
            if decl.value is None or decl.deferred:
                continue
            vt = ctx.expr(decl.value)
            if g.declared is not None:
                ctx.check(g.declared, vt, decl, f"'{decl.name}'", value_node=decl.value)
        s.run()

    def infer_globals(self):
        self.solver.sink = None
        ctx = FuncCtx(self, self.solver, None, 'A', True)
        self.globals_ctx = ctx
        for g in self.globals.values():
            decl = g.decl
            if decl.value is None or decl.deferred:
                continue
            vt = ctx.expr(decl.value)
            ctx.unify(g.type, vt, decl)

    def env_vars(self) -> Set[TVar]:
        acc: Set[TVar] = set()
        terms = [g.type for g in self.globals.values() if g.type is not None]
        for info in self.structs.values():
            terms.extend(info.flow.values())
        for info in self.adts.values():
            for flows in info.flow.values():
                terms.extend(flows)
        for t in terms:
            _free(t, acc)
        return acc

    def generalize(self, infos: List[FuncInfo], created: List[Constraint]):
        env = self.env_vars()
        pending = [c for c in created if not c.done]
        # variables tied to the environment through open constraints are not
        # generalised (the constraint must stay live to inform the globals)
        cvars = []
        for c in pending:
            fv: Set[TVar] = set()
            for t in c.terms():
                _free(t, fv)
            cvars.append(fv)
        changed = True
        while changed:
            changed = False
            for fv in cvars:
                if fv & env and not fv <= env:
                    env |= fv
                    changed = True
        cand: Set[TVar] = set()
        for info in infos:
            for t in list(info.mono.params) + [info.mono.ret]:
                _free(t, cand)
        for c in pending:
            for t in c.terms():
                _free(t, cand)
        q = cand - env
        templates = []
        for c in pending:
            fv: Set[TVar] = set()
            for t in c.terms():
                _free(t, fv)
            if fv & q:
                templates.append(c)
                c.done = True           # lives on in the scheme only
        for info in infos:
            info.scheme = Scheme(q, info.mono, templates)

    def freeze(self):
        for g in self.globals.values():
            if g.type is not None:
                g.frozen = deep(g.type, UNK)
        for info in self.structs.values():
            info.frozen = {k: deep(v, UNK) for k, v in info.flow.items()}
        for info in self.adts.values():
            info.frozen = {k: [deep(v, UNK) for v in vs] for k, vs in info.flow.items()}

    def collect_errors(self):
        seen = set()
        for c in self.suspects:
            if not c.still_wrong():
                continue
            node = c.node
            line = getattr(node, 'line', 0) or 0
            col = getattr(node, 'col', 0) or 1
            msg = c.error if not isinstance(c, CheckC) else c.message()
            key = (line, col, msg)
            if key in seen:
                continue
            seen.add(key)
            self.errors.append(self.reporter.format_error(line, col, msg, "E-TYPE", self.hint(c)))

    @staticmethod
    def hint(c) -> str:
        if isinstance(c, CheckC):
            return "convert the value explicitly (str(x), int(x), float(x)) or change the annotation"
        if isinstance(c, (FieldC, MethodC)):
            return "check the spelling, or add the field / method to the struct"
        return ""

    # ── methods (shared by both phases) ─────────────────────────────────────
    def method_missing(self, o: TCon, name: str) -> bool:
        """Is `o.name(...)` definitely an error?"""
        if name == 'to_string':
            return False
        if is_struct(o):
            info = self.structs.get(nominal_name(o))
            if info is None:
                return False
            return name not in info.methods and info.field_type(name, 'A') is None
        if o.name in ('int', 'float', 'bool', 'nil', 'fn'):
            return o is not NIL or name not in ('is_some', 'is_none', 'is_ok', 'is_err', 'unwrap',
                                                'expect', 'unwrap_or', 'error')
        return False

    def solve_method(self, s: Solver, c: MethodC, o: TCon):
        name, args, r = c.name, c.args, c.r
        if o in (DYN, UNK):
            s.unify(r, o)
            return None
        if o.name == 'strish':
            s.unify(r, STR if name == 'to_string' else DYN)
            return None
        if name == 'to_string':
            s.unify(r, STR)
            return None
        if is_struct(o):
            sname = nominal_name(o)
            info = self.structs.get(sname)
            key = info.methods.get(name) if info else None
            if key is not None:
                finfo = self.funcs[key]
                if finfo.scheme is None:
                    params, ret, modes = list(finfo.mono.params), finfo.mono.ret, finfo.mono.modes
                elif s.phase == 'E' and self.is_generic(key):
                    params, ret, modes = self.opaque(key)
                else:
                    params, ret = finfo.scheme.instantiate(s)
                    modes = finfo.scheme.sig.modes
                s.unify(params[0], o)
                if c.ctx is not None:
                    c.ctx.bind_args(key, params, modes, args, c.node, skip_self=True, report=c.report,
                                    arg_nodes=getattr(c.node, 'args', None))
                    c.ctx.calls[id(c.node)] = (key, params)
                else:
                    for p, a in zip(params[1:], args):
                        s.unify(p, a)
                s.unify(r, ret)
                return None
            ft = info.field_type(name, s.phase) if info else None
            if ft is not None:             # a field holding a closure
                s.add(CallValueC(ft, args, r))
                return None
            c.fail(s, f"struct '{sname}' has no method '{name}'")
            s.unify(r, DYN)
            return None
        handler = BUILTIN_METHODS.get(o.name)
        if handler is not None:
            handler(self, s, c, o)
            return None
        s.unify(r, DYN)
        return None

    # ── phase B: monomorphic instances ──────────────────────────────────────
    def specialize(self, key: str, params) -> Instance:
        ikey = (key, tuple(canon(p) for p in params))
        inst = self.instances.get(ikey)
        if inst is not None:
            return inst
        info = self.funcs[key]
        s = Solver('B')
        sig = self.signature(info, given=list(params), s=s)
        ctx = self.infer_body(info, sig, s, 'B', False)
        s.run()
        self.settle(s)
        inst = Instance(key, info, sig.params, sig.ret, ctx)
        inst.tparams = {name: deep(v, UNK) for name, v in sig.tparams.items()}
        inst.modes = list(sig.modes)
        self.instances[ikey] = inst
        return inst

    def specialize_globals(self) -> Instance:
        """Typing of the global initialisers (evaluated before main)."""
        s = Solver('B')
        ctx = FuncCtx(self, s, None, 'B', False)
        for g in self.globals.values():
            decl = g.decl
            if decl.value is None or decl.deferred:
                continue
            ctx.unify(g.frozen, ctx.expr(decl.value), decl)
        s.run()
        self.settle(s)
        return Instance('<globals>', None, [], NIL, ctx)

    @staticmethod
    def settle(s: Solver):
        """Undecided constraints: their results are unknown (any representation)."""
        for _round in range(64):
            progress = False
            for c in s.all:
                if c.done:
                    continue
                for out in c.outputs():
                    o = resolve(out)
                    if isinstance(o, TVar):
                        s.bind(o, UNK)
                        progress = True
                s.run()
            if not progress:
                break
        s.run()

    def is_generic(self, key: str) -> bool:
        info = self.funcs[key]
        if info.scheme is None:
            return False
        acc: Set[TVar] = set()
        for p in info.scheme.sig.params:
            _free(p, acc)
        return bool(acc & info.scheme.q)

    def scheme_params(self, key: str):
        info = self.funcs[key]
        return [deep(p, UNK) for p in info.scheme.sig.params]


def _free(t, acc: Set[TVar]):
    t = resolve(t)
    if isinstance(t, TVar):
        acc.add(t)
        return
    for a in t.args:
        _free(a, acc)


# ── builtin methods ─────────────────────────────────────────────────────────

def _m_str(infer, s, c, o):
    name, r = c.name, c.r
    if name in ('len', 'find', 'index_of', 'count', 'get'):
        s.unify(r, INT)
    elif name in ('contains', 'starts_with', 'ends_with', 'eq', 'is_empty'):
        s.unify(r, BOOL)
    elif name in ('trim', 'to_upper', 'to_lower', 'replace', 'slice', 'repeat'):
        s.unify(r, STR)
    elif name == 'split':
        s.unify(r, arr(STR))
    elif name == 'to_int':
        s.unify(r, INT)
    elif name == 'to_float':
        s.unify(r, FLOAT)
    else:
        s.unify(r, DYN)


def _m_array(infer, s, c, o):
    name, args, r = c.name, c.args, c.r
    e = o.args[0]
    if name in ('len', 'index_of', 'count'):
        s.unify(r, INT)
    elif name in ('push', 'set', 'insert'):
        if args:
            s.flow(args[-1], e)
        s.unify(r, NIL)
    elif name in ('pop', 'get', 'remove', 'first', 'last'):
        s.unify(r, e)
    elif name in ('contains', 'is_empty'):
        s.unify(r, BOOL)
    elif name in ('reverse', 'sort', 'clear'):
        s.unify(r, NIL)
    elif name == 'join':
        s.unify(r, STR)
    elif name in ('slice', 'copy'):
        s.unify(r, arr(e))
    elif name in ('sum', 'min', 'max'):
        s.add(FoldC(name, [o], r))
    elif name == 'map':
        out = TVar()
        if args:
            s.unify(args[0], fnt(out, [e]))
        s.unify(r, arr(out))
    elif name == 'filter':
        if args:
            s.unify(args[0], fnt(TVar(), [e]))
        s.unify(r, arr(e))
    elif name == 'reduce':
        acc = TVar()
        if len(args) > 1:
            s.flow(args[1], acc)
        if args:
            s.unify(args[0], fnt(acc, [acc, e]))
        s.unify(r, acc)
    elif name == 'each':
        if args:
            s.unify(args[0], fnt(TVar(), [e]))
        s.unify(r, NIL)
    elif name in ('any', 'all'):
        if args:
            s.unify(args[0], fnt(TVar(), [e]))
        s.unify(r, BOOL)
    else:
        s.unify(r, DYN)


def _m_map(infer, s, c, o):
    name, args, r = c.name, c.args, c.r
    k, v = o.args
    if name == 'len':
        s.unify(r, INT)
    elif name == 'set':
        if len(args) == 2:
            s.flow(args[0], k)
            s.flow(args[1], v)
        s.unify(r, NIL)
    elif name in ('get', 'get_or'):
        if name == 'get_or' and len(args) > 1:
            s.flow(v, r)
            s.flow(args[1], r)
        else:
            s.unify(r, v)
    elif name in ('has', 'contains'):
        s.unify(r, BOOL)
    elif name in ('delete', 'remove'):
        s.unify(r, NIL)
    elif name == 'keys':
        s.unify(r, arr(k))
    elif name == 'values':
        s.unify(r, arr(v))
    elif name == 'is_empty':
        s.unify(r, BOOL)
    elif name == 'clear':
        s.unify(r, NIL)
    elif name == 'copy':
        s.unify(r, mapt(k, v))
    else:
        s.unify(r, DYN)


def _m_optres(infer, s, c, o):
    name, args, r = c.name, c.args, c.r
    t = o.args[0]
    if name in ('is_some', 'is_none', 'is_ok', 'is_err'):
        s.unify(r, BOOL)
    elif name in ('unwrap', 'expect'):
        s.unify(r, t)
    elif name == 'unwrap_or':
        s.flow(t, r)
        if args:
            s.flow(args[0], r)
    elif name == 'error' and o.name == 'result':
        s.unify(r, o.args[1])
    else:
        s.unify(r, DYN)


def _m_nil(infer, s, c, o):
    name, args, r = c.name, c.args, c.r
    if name in ('is_some', 'is_none', 'is_ok', 'is_err'):
        s.unify(r, BOOL)
    elif name == 'unwrap_or' and args:
        s.unify(r, args[0])
    elif name in ('unwrap', 'expect', 'error'):
        pass
    else:
        s.unify(r, DYN)


BUILTIN_METHODS = {'str': _m_str, 'array': _m_array, 'map': _m_map, 'option': _m_optres,
                   'result': _m_optres, 'nil': _m_nil}


# ── public API (used by til.py) ─────────────────────────────────────────────

def infer_program(program, source: str = "", filename: str = "<stdin>") -> TypeInfer:
    return TypeInfer(program, source, filename).run()


def check_program(program, source: str = "", filename: str = "<stdin>") -> List[str]:
    """E-TYPE errors (formatted) of a parsed, import-resolved program."""
    return infer_program(program, source, filename).errors


# ═══════════════════════════════════════════════════════════════════════════════
#                      TYPED NATIVE BACKEND: C RUNTIME
# ═══════════════════════════════════════════════════════════════════════════════
#
# Emitted after the classic prelude of CCodeGenerator (til_panic, retry/timeout
# recovery points, string helpers, Option/Result types). Every runtime error
# uses the VM's message text, so `error` in a retry/timeout `else` branch is
# the same on every backend.

TYPED_RUNTIME = r"""
// ═══ typed native backend runtime ═══
#include <errno.h>
#include <limits.h>
static int til_argc = 0;
static char** til_argv = NULL;

static void* til_alloc_obj(size_t n) {
    void* p = calloc(1, n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}
static char* til_strdup_n(const char* s, size_t n) {
    char* r = (char*)til_alloc_obj(n + 1);
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}
static void til_panicf(const char* fmt, long long a, long long b) {
    snprintf(til_panic_buf, sizeof til_panic_buf, fmt, a, b);
    til_panic(til_panic_buf);
}
/* index errors: the message is the VMs' (seen by `error`); uncaught, the
   classic native report "Bounds check failed: ..." is printed */
static void til_panic_bounds(long long i, long long n) {
    snprintf(til_panic_buf, sizeof til_panic_buf, "index out of bounds: index %lld, length %lld", i, n);
    if (!til_recover_top) {
        fflush(stdout);
        fprintf(stderr, "Bounds check failed: %s\n", til_panic_buf);
        exit(1);
    }
    til_panic(til_panic_buf);
}
static void til_panic_str(const char* prefix, const char* s, const char* suffix) {
    size_t a = strlen(prefix), b = strlen(s), c = strlen(suffix);
    char* m = (char*)til_alloc_obj(a + b + c + 1);
    memcpy(m, prefix, a); memcpy(m + a, s, b); memcpy(m + a + b, suffix, c);
    til_panic(m);
}

// numbers (TIL int is 64-bit two's complement; float printing is C %g with nan/inf like the VMs)
static int64_t til_div_i(int64_t a, int64_t b) {
    if (b == 0) til_panic("division by zero");
    if (b == -1) return (int64_t)(0 - (uint64_t)a);
    return a / b;
}
static int64_t til_mod_i(int64_t a, int64_t b) {
    if (b == 0) til_panic("division by zero");
    if (b == -1) return 0;
    return a % b;
}
/* int vs float compared exactly (like Python): -1, 0, 1, or 2 when b is NaN */
static int til_cmp_if(int64_t a, double b) {
    double t, frac;
    int64_t bi;
    if (b != b) return 2;
    if (b >= 9223372036854775808.0) return -1;
    if (b < -9223372036854775808.0) return 1;
    t = trunc(b);
    bi = (int64_t)t;
    if (a < bi) return -1;
    if (a > bi) return 1;
    frac = b - t;
    return frac > 0 ? -1 : (frac < 0 ? 1 : 0);
}
static int til_cmp_fi(double a, int64_t b) { int c = til_cmp_if(b, a); return c == 2 ? 2 : -c; }
static int64_t til_shl(int64_t a, int64_t b) { return (int64_t)((uint64_t)a << (b & 63)); }
static int64_t til_shr(int64_t a, int64_t b) { return a >> (b & 63); }
static int64_t til_abs_i(int64_t a) { return a < 0 ? (int64_t)(0 - (uint64_t)a) : a; }
static double til_fmod(double a, double b) { return b != 0.0 ? fmod(a, b) : NAN; }
static double til_fpow(double a, double b) {
    double r;
    if (a == 0.0 && b < 0.0) return NAN;                 /* Python: math domain error */
    r = pow(a, b);
    if (isinf(r) && !isinf(a) && !isinf(b)) return a > 1.0 ? INFINITY : NAN;   /* overflow */
    if (isnan(r)) return NAN;
    return r;
}
static int64_t til_f2i(double x) {
    if (x != x || isinf(x)) til_panic("cannot convert nan/inf to int");
    if (x >= 9223372036854775808.0 || x < -9223372036854775808.0) {
        double m = fmod(x, 18446744073709551616.0);      /* wrap like Python int -> int64 */
        if (m < 0) m += 18446744073709551616.0;
        return (int64_t)(uint64_t)m;
    }
    return (int64_t)x;
}
static const char* til_g_buf(double x, char* buf, size_t n) {
    if (x != x) return "nan";
    if (isinf(x)) return x > 0 ? "inf" : "-inf";
    snprintf(buf, n, "%g", x);
    return buf;
}
static char* til_float_str(double x) { char b[40]; const char* s = til_g_buf(x, b, sizeof b); return til_strdup_n(s, strlen(s)); }
static void til_print_g(double x) { char b[40]; puts(til_g_buf(x, b, sizeof b)); }
static const char* til_s(const char* s) { return s ? s : "None"; }
static void til_print_s(const char* s) { puts(s ? s : "None"); }

// containers: one 8-byte slot per element; the static type says which member
typedef union TIL_Slot { int64_t i; double f; const char* s; void* p; } TIL_Slot;
static TIL_Slot til_slot_i(int64_t v) { TIL_Slot s; s.i = v; return s; }
static TIL_Slot til_slot_f(double v) { TIL_Slot s; s.f = v; return s; }
static TIL_Slot til_slot_s(const char* v) { TIL_Slot s; s.s = v; return s; }
static TIL_Slot til_slot_p(void* v) { TIL_Slot s; s.p = v; return s; }

typedef struct TIL_Arr { int64_t len, cap; TIL_Slot* data; } TIL_Arr;
static TIL_Arr* til_arr_new(int64_t cap) {
    TIL_Arr* a = (TIL_Arr*)til_alloc_obj(sizeof(TIL_Arr));
    a->cap = cap > 4 ? cap : 4;
    a->data = (TIL_Slot*)til_alloc_obj(sizeof(TIL_Slot) * (size_t)a->cap);
    return a;
}
static void til_arr_reserve(TIL_Arr* a, int64_t n) {
    if (n <= a->cap) return;
    while (a->cap < n) a->cap *= 2;
    a->data = (TIL_Slot*)realloc(a->data, sizeof(TIL_Slot) * (size_t)a->cap);
    if (!a->data) { fprintf(stderr, "out of memory\n"); exit(1); }
}
static TIL_Arr* til_arr_lit(int64_t n, const TIL_Slot* items) {
    TIL_Arr* a = til_arr_new(n);
    if (n) memcpy(a->data, items, sizeof(TIL_Slot) * (size_t)n);
    a->len = n;
    return a;
}
static TIL_Arr* til_arr_chk(TIL_Arr* a) { if (!a) til_panic("None is not an array"); return a; }
static int64_t til_arr_len(TIL_Arr* a) { return til_arr_chk(a)->len; }
static void til_arr_push(TIL_Arr* a, TIL_Slot v) {
    til_arr_chk(a);
    til_arr_reserve(a, a->len + 1);
    a->data[a->len++] = v;
}
static TIL_Slot til_arr_pop(TIL_Arr* a) {
    if (til_arr_chk(a)->len == 0) til_panic("pop() from empty array");
    return a->data[--a->len];
}
static TIL_Slot* til_arr_at(TIL_Arr* a, int64_t i) {
    if (i < 0 || i >= til_arr_chk(a)->len) til_panic_bounds((long long)i, (long long)a->len);
    return &a->data[i];
}
static void til_arr_insert(TIL_Arr* a, int64_t i, TIL_Slot v) {
    if (i < 0 || i > til_arr_chk(a)->len) til_panicf("insert index out of bounds: %lld", (long long)i, 0);
    til_arr_reserve(a, a->len + 1);
    memmove(&a->data[i + 1], &a->data[i], sizeof(TIL_Slot) * (size_t)(a->len - i));
    a->data[i] = v;
    a->len++;
}
static TIL_Slot til_arr_remove(TIL_Arr* a, int64_t i) {
    TIL_Slot v;
    if (i < 0 || i >= til_arr_chk(a)->len) til_panicf("remove index out of bounds: %lld", (long long)i, 0);
    v = a->data[i];
    memmove(&a->data[i], &a->data[i + 1], sizeof(TIL_Slot) * (size_t)(a->len - i - 1));
    a->len--;
    return v;
}
static void til_arr_reverse(TIL_Arr* a) {
    int64_t i, j;
    til_arr_chk(a);
    for (i = 0, j = a->len - 1; i < j; i++, j--) { TIL_Slot t = a->data[i]; a->data[i] = a->data[j]; a->data[j] = t; }
}
static TIL_Arr* til_arr_slice(TIL_Arr* a, int64_t lo, int64_t hi) {
    int64_t n = til_arr_chk(a)->len;
    lo = lo < 0 ? 0 : (lo > n ? n : lo);
    hi = hi < 0 ? 0 : (hi > n ? n : hi);
    return lo < hi ? til_arr_lit(hi - lo, a->data + lo) : til_arr_new(0);
}
static TIL_Arr* til_arr_copy(TIL_Arr* a) { return til_arr_lit(til_arr_chk(a)->len, a->data); }
static TIL_Arr* til_arr_concat(TIL_Arr* a, TIL_Arr* b) {
    TIL_Arr* r = til_arr_new(til_arr_chk(a)->len + til_arr_chk(b)->len);
    memcpy(r->data, a->data, sizeof(TIL_Slot) * (size_t)a->len);
    memcpy(r->data + a->len, b->data, sizeof(TIL_Slot) * (size_t)b->len);
    r->len = a->len + b->len;
    return r;
}
static TIL_Arr* til_arr_repeat(TIL_Arr* a, int64_t n) {
    TIL_Arr* r = til_arr_new(0);
    int64_t k;
    til_arr_chk(a);
    for (k = 0; k < n; k++) {
        til_arr_reserve(r, r->len + a->len);
        memcpy(r->data + r->len, a->data, sizeof(TIL_Slot) * (size_t)a->len);
        r->len += a->len;
    }
    return r;
}
static void til_arr_clear(TIL_Arr* a) { til_arr_chk(a)->len = 0; }
static TIL_Arr* til_range(int64_t a, int64_t b, bool incl) {
    int64_t end = incl ? b + 1 : b, i;
    TIL_Arr* r = til_arr_new(end > a ? end - a : 0);
    for (i = a; i < end; i++) r->data[r->len++].i = i;
    return r;
}
/* stable merge sort (Python's sort is stable) */
typedef int (*til_cmp_fn)(TIL_Slot, TIL_Slot);
static int til_cmp_i(TIL_Slot a, TIL_Slot b) { return a.i < b.i ? -1 : a.i > b.i; }
static int til_cmp_f(TIL_Slot a, TIL_Slot b) { return a.f < b.f ? -1 : a.f > b.f; }
static int til_cmp_s(TIL_Slot a, TIL_Slot b) { return strcmp(a.s, b.s); }
static void til_arr_sort(TIL_Arr* a, til_cmp_fn cmp) {
    int64_t n = til_arr_chk(a)->len, w, i;
    TIL_Slot* tmp;
    if (n < 2) return;
    tmp = (TIL_Slot*)til_alloc_obj(sizeof(TIL_Slot) * (size_t)n);
    for (w = 1; w < n; w *= 2) {
        for (i = 0; i < n; i += 2 * w) {
            int64_t lo = i, mid = i + w < n ? i + w : n, hi = i + 2 * w < n ? i + 2 * w : n;
            int64_t p = lo, q = mid, k = lo;
            while (p < mid && q < hi) tmp[k++] = cmp(a->data[q], a->data[p]) < 0 ? a->data[q++] : a->data[p++];
            while (p < mid) tmp[k++] = a->data[p++];
            while (q < hi) tmp[k++] = a->data[q++];
        }
        memcpy(a->data, tmp, sizeof(TIL_Slot) * (size_t)n);
    }
    free(tmp);
}

/* insertion-ordered hash map (like the VMs' maps); key kind 0 = int/bool, 1 = str */
typedef struct TIL_Map { int64_t len, cap; TIL_Slot* keys; TIL_Slot* vals; int64_t* index; int64_t icap; } TIL_Map;
static uint64_t til_hash(TIL_Slot k, int kind) {
    uint64_t h;
    if (kind == 1) {
        const unsigned char* p = (const unsigned char*)k.s;
        h = 1469598103934665603ULL;
        while (*p) { h ^= *p++; h *= 1099511628211ULL; }
        return h;
    }
    h = (uint64_t)k.i + 0x9E3779B97F4A7C15ULL;
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
    h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
    return h ^ (h >> 31);
}
static bool til_key_eq(TIL_Slot a, TIL_Slot b, int kind) {
    return kind == 1 ? strcmp(a.s, b.s) == 0 : a.i == b.i;
}
static void til_map_reindex(TIL_Map* m, int kind) {
    int64_t i;
    m->icap = 16;
    while (m->icap < m->cap * 2) m->icap *= 2;
    free(m->index);
    m->index = (int64_t*)til_alloc_obj(sizeof(int64_t) * (size_t)m->icap);
    for (i = 0; i < m->len; i++) {
        uint64_t h = til_hash(m->keys[i], kind) & (uint64_t)(m->icap - 1);
        while (m->index[h]) h = (h + 1) & (uint64_t)(m->icap - 1);
        m->index[h] = i + 1;
    }
}
static TIL_Map* til_map_new(void) {
    TIL_Map* m = (TIL_Map*)til_alloc_obj(sizeof(TIL_Map));
    m->cap = 8;
    m->keys = (TIL_Slot*)til_alloc_obj(sizeof(TIL_Slot) * 8);
    m->vals = (TIL_Slot*)til_alloc_obj(sizeof(TIL_Slot) * 8);
    return m;
}
static TIL_Map* til_map_chk(TIL_Map* m) { if (!m) til_panic("None is not a map"); return m; }
static int64_t til_map_find(TIL_Map* m, TIL_Slot k, int kind) {
    uint64_t h;
    if (til_map_chk(m)->len == 0) return -1;
    if (!m->index) til_map_reindex(m, kind);
    h = til_hash(k, kind) & (uint64_t)(m->icap - 1);
    while (m->index[h]) {
        int64_t i = m->index[h] - 1;
        if (til_key_eq(m->keys[i], k, kind)) return i;
        h = (h + 1) & (uint64_t)(m->icap - 1);
    }
    return -1;
}
static void til_map_set(TIL_Map* m, TIL_Slot k, TIL_Slot v, int kind) {
    int64_t i = til_map_find(m, k, kind);
    if (i >= 0) { m->vals[i] = v; return; }
    if (m->len == m->cap) {
        m->cap *= 2;
        m->keys = (TIL_Slot*)realloc(m->keys, sizeof(TIL_Slot) * (size_t)m->cap);
        m->vals = (TIL_Slot*)realloc(m->vals, sizeof(TIL_Slot) * (size_t)m->cap);
        if (!m->keys || !m->vals) { fprintf(stderr, "out of memory\n"); exit(1); }
    }
    m->keys[m->len] = k;
    m->vals[m->len] = v;
    m->len++;
    if (m->index && m->len * 2 <= m->icap) {
        uint64_t h = til_hash(k, kind) & (uint64_t)(m->icap - 1);
        while (m->index[h]) h = (h + 1) & (uint64_t)(m->icap - 1);
        m->index[h] = m->len;
    } else {
        til_map_reindex(m, kind);
    }
}
static const char* til_key_str(TIL_Slot k, int kind) {
    if (kind == 1) return k.s;
    if (kind == 2) return k.i ? "true" : "false";
    return til_int_to_str(k.i);
}
static TIL_Slot til_map_get(TIL_Map* m, TIL_Slot k, int kind) {
    int64_t i = til_map_find(m, k, kind < 2 ? kind : 0);
    if (i < 0) til_panic_str("key not found: ", til_key_str(k, kind), "");
    return m->vals[i];
}
static void til_map_delete(TIL_Map* m, TIL_Slot k, int kind) {
    int64_t i = til_map_find(m, k, kind);
    if (i < 0) return;
    memmove(&m->keys[i], &m->keys[i + 1], sizeof(TIL_Slot) * (size_t)(m->len - i - 1));
    memmove(&m->vals[i], &m->vals[i + 1], sizeof(TIL_Slot) * (size_t)(m->len - i - 1));
    m->len--;
    til_map_reindex(m, kind);
}
static TIL_Map* til_map_lit(int64_t n, int kind, const TIL_Slot* kv) {
    TIL_Map* m = til_map_new();
    int64_t i;
    for (i = 0; i < n; i++) til_map_set(m, kv[2 * i], kv[2 * i + 1], kind);
    return m;
}
static TIL_Arr* til_map_keys(TIL_Map* m) { return til_arr_lit(til_map_chk(m)->len, m->keys); }
static TIL_Arr* til_map_values(TIL_Map* m) { return til_arr_lit(til_map_chk(m)->len, m->vals); }
static TIL_Map* til_map_copy(TIL_Map* m, int kind) {
    TIL_Map* r = til_map_new();
    int64_t i;
    for (i = 0; i < til_map_chk(m)->len; i++) til_map_set(r, m->keys[i], m->vals[i], kind);
    return r;
}
static void til_map_clear(TIL_Map* m) { til_map_chk(m)->len = 0; free(m->index); m->index = NULL; }

/* growable string buffer (to_str of containers, join, f-strings) */
typedef struct TIL_Buf { char* p; size_t n, cap; } TIL_Buf;
static void til_buf_put(TIL_Buf* b, const char* s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2;
        b->p = (char*)realloc(b->p, b->cap);
        if (!b->p) { fprintf(stderr, "out of memory\n"); exit(1); }
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = '\0';
}
static void til_buf_add(TIL_Buf* b, const char* s) { til_buf_put(b, s, strlen(s)); }
static void til_buf_q(TIL_Buf* b, const char* s) {     /* a string inside a container: quoted */
    if (!s) { til_buf_add(b, "None"); return; }
    til_buf_add(b, "\""); til_buf_add(b, s); til_buf_add(b, "\"");
}
static char* til_buf_done(TIL_Buf* b) { if (!b->p) return til_strdup_n("", 0); return b->p; }

/* strings */
static int64_t til_str_len(const char* s) { if (!s) til_panic("len() of nil"); return (int64_t)strlen(s); }
static int64_t til_str_at(const char* s, int64_t i) {
    int64_t n = til_str_len(s);
    if (i < 0 || i >= n) til_panic_bounds((long long)i, (long long)n);
    return (int64_t)(unsigned char)s[i];
}
static bool til_str_eqn(const char* a, const char* b) {
    if (!a || !b) return a == b;
    return strcmp(a, b) == 0;
}
static int til_str_cmp(const char* a, const char* b) { return strcmp(a, b); }
static char* til_str_rep(const char* s, int64_t n) {
    size_t len = strlen(s), k;
    char* r;
    if (n <= 0 || len == 0) return til_strdup_n("", 0);
    r = (char*)til_alloc_obj(len * (size_t)n + 1);
    for (k = 0; k < (size_t)n; k++) memcpy(r + k * len, s, len);
    r[len * (size_t)n] = '\0';
    return r;
}
static char* til_str_replace2(const char* s, const char* old, const char* new_s) {
    if (!*old) return (char*)s;
    return til_str_replace(s, old, new_s);
}
static int64_t til_str_count(const char* s, const char* sub) {
    size_t n = strlen(sub);
    int64_t c = 0;
    const char* p = s;
    if (!n) return 0;
    while ((p = strstr(p, sub)) != NULL) { c++; p += n; }
    return c;
}
static bool til_is_ws(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 11 || c == 12; }
static TIL_Arr* til_str_split(const char* s, const char* sep) {
    TIL_Arr* r = til_arr_new(0);
    size_t n = strlen(sep);
    if (!n) {
        for (; *s; s++) til_arr_push(r, til_slot_s(til_strdup_n(s, 1)));
        return r;
    }
    for (;;) {
        const char* p = strstr(s, sep);
        if (!p) { til_arr_push(r, til_slot_s(til_strdup_n(s, strlen(s)))); return r; }
        til_arr_push(r, til_slot_s(til_strdup_n(s, (size_t)(p - s))));
        s = p + n;
    }
}
static TIL_Arr* til_str_split_ws(const char* s) {
    TIL_Arr* r = til_arr_new(0);
    for (;;) {
        const char* start;
        while (*s && til_is_ws((unsigned char)*s)) s++;
        if (!*s) return r;
        start = s;
        while (*s && !til_is_ws((unsigned char)*s)) s++;
        til_arr_push(r, til_slot_s(til_strdup_n(start, (size_t)(s - start))));
    }
}
static const char* til_strip_ws(const char* s, size_t* n) {
    size_t len;
    while (*s && til_is_ws((unsigned char)*s)) s++;
    len = strlen(s);
    while (len && til_is_ws((unsigned char)s[len - 1])) len--;
    *n = len;
    return s;
}
static int64_t til_str_to_int(const char* s) {
    size_t n, i = 0;
    const char* p = til_strip_ws(s, &n);
    bool neg = false, digit = false, under = false;
    uint64_t v = 0;
    if (n && (p[0] == '+' || p[0] == '-')) { neg = p[0] == '-'; i = 1; }
    for (; i < n; i++) {
        char c = p[i];
        if (c >= '0' && c <= '9') { v = v * 10 + (uint64_t)(c - '0'); digit = true; under = false; }
        else if (c == '_' && digit && !under) under = true;
        else { digit = false; break; }
    }
    if (!digit || under) til_panic_str("invalid integer: \"", s, "\"");
    return (int64_t)(neg ? 0 - v : v);
}
static double til_str_to_float(const char* s) {
    size_t n, i, k = 0;
    const char* p = til_strip_ws(s, &n);
    char* buf = (char*)til_alloc_obj(n + 1);
    char* end;
    double v;
    for (i = 0; i < n; i++) {                     /* digits may be separated by single '_' */
        if (p[i] == '_' && i > 0 && i + 1 < n && p[i - 1] >= '0' && p[i - 1] <= '9'
                && p[i + 1] >= '0' && p[i + 1] <= '9') continue;
        if (p[i] == 'x' || p[i] == 'X' || p[i] == 'p' || p[i] == 'P' || p[i] == '_') { k = 0; break; }
        buf[k++] = p[i];
    }
    buf[k] = '\0';
    errno = 0;
    v = k ? strtod(buf, &end) : 0.0;
    if (!k || *end) til_panic_str("invalid number: \"", s, "\"");
    return v;
}
static const char* til_chr(int64_t c) {
    char b[5];
    if (c < 0 || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) til_panic("chr() argument out of range");
    if (c == 0) til_panic("chr(0): strings containing NUL bytes are not supported by the native backend");
    if (c < 0x80) { b[0] = (char)c; return til_strdup_n(b, 1); }
    if (c < 0x800) { b[0] = (char)(0xC0 | (c >> 6)); b[1] = (char)(0x80 | (c & 63)); return til_strdup_n(b, 2); }
    if (c < 0x10000) {
        b[0] = (char)(0xE0 | (c >> 12)); b[1] = (char)(0x80 | ((c >> 6) & 63)); b[2] = (char)(0x80 | (c & 63));
        return til_strdup_n(b, 3);
    }
    b[0] = (char)(0xF0 | (c >> 18)); b[1] = (char)(0x80 | ((c >> 12) & 63));
    b[2] = (char)(0x80 | ((c >> 6) & 63)); b[3] = (char)(0x80 | (c & 63));
    return til_strdup_n(b, 4);
}
static int64_t til_ord(const char* s) {
    const unsigned char* p = (const unsigned char*)s;
    int need, lo = 0x80, hi = 0xBF, k;
    int64_t c;
    if (!s || !*s) til_panic("ord() expects a non-empty string");
    if (p[0] < 0x80) return p[0];
    if (p[0] >= 0xC2 && p[0] <= 0xDF) { need = 1; c = p[0] & 0x1F; }
    else if (p[0] >= 0xE0 && p[0] <= 0xEF) {
        need = 2; c = p[0] & 0x0F;
        if (p[0] == 0xE0) lo = 0xA0;
        if (p[0] == 0xED) hi = 0x9F;
    } else if (p[0] >= 0xF0 && p[0] <= 0xF4) {
        need = 3; c = p[0] & 0x07;
        if (p[0] == 0xF0) lo = 0x90;
        if (p[0] == 0xF4) hi = 0x8F;
    } else return 0xFFFD;
    for (k = 1; k <= need; k++) {
        if (p[k] < lo || p[k] > hi) return 0xFFFD;
        lo = 0x80; hi = 0xBF;
        c = (c << 6) | (p[k] & 0x3F);
    }
    return c;
}

/* f-string format specs: [<>^][0][width][.prec][f|e|%|d|x|X|o|b] (same rules as the VMs) */
static const char* til_fmt_spec(int kind, int64_t iv, double fv, const char* body, const char* spec) {
    const char* p = spec;
    char align = 0, conv = 0, num[80];
    char* big = NULL;
    int zero = 0, width = 0, prec = -1, is_num = kind == 0 || kind == 1;
    size_t blen, pad, k;
    char* r;
    if (*p == '<' || *p == '>' || *p == '^') align = *p++;
    if (*p == '0') { zero = 1; p++; }
    while (*p >= '0' && *p <= '9') width = width * 10 + (*p++ - '0');
    if (*p == '.') {
        p++;
        if (!(*p >= '0' && *p <= '9')) return body;
        prec = 0;
        while (*p >= '0' && *p <= '9') prec = prec * 10 + (*p++ - '0');
    }
    if (*p && strchr("fdxXobe%", *p)) conv = *p++;
    if (*p) return body;                                   /* not a valid spec: plain text */
    if (conv && strchr("fe%", conv) && is_num) {
        double x = kind == 0 ? (double)iv : fv;
        char f[16];
        if (conv == '%') x *= 100.0;
        if (x != x) snprintf(num, sizeof num, "nan");
        else if (isinf(x)) snprintf(num, sizeof num, x > 0 ? "inf" : "-inf");
        else {
            int need;
            snprintf(f, sizeof f, "%%.%d%c", prec < 0 ? 6 : prec, conv == 'e' ? 'e' : 'f');
            need = snprintf(NULL, 0, f, x);
            big = (char*)til_alloc_obj((size_t)need + 2);
            snprintf(big, (size_t)need + 1, f, x);
        }
        if (big) {
            if (conv == '%') strcat(big, "%");
            body = big;
        } else {
            if (conv == '%') strcat(num, "%");
            body = num;
        }
    } else if (conv && strchr("dxXob", conv) && is_num) {
        int64_t n = kind == 0 ? iv : til_f2i(fv);
        uint64_t u = n < 0 ? 0 - (uint64_t)n : (uint64_t)n;
        char digits[72];
        int d = 0;
        int base = conv == 'd' ? 10 : conv == 'o' ? 8 : conv == 'b' ? 2 : 16;
        const char* set = conv == 'X' ? "0123456789ABCDEF" : "0123456789abcdef";
        do { digits[d++] = set[u % (uint64_t)base]; u /= (uint64_t)base; } while (u);
        k = 0;
        if (n < 0) num[k++] = '-';
        while (d) num[k++] = digits[--d];
        num[k] = '\0';
        body = num;
    } else if (kind == 1 && prec >= 0) {
        if (fv != fv) snprintf(num, sizeof num, "nan");
        else if (isinf(fv)) snprintf(num, sizeof num, fv > 0 ? "inf" : "-inf");
        else {
            char f[16];
            int need;
            snprintf(f, sizeof f, "%%.%dg", prec);
            need = snprintf(NULL, 0, f, fv);
            big = (char*)til_alloc_obj((size_t)need + 1);
            snprintf(big, (size_t)need + 1, f, fv);
        }
        body = big ? big : num;
    }
    blen = strlen(body);
    if ((int)blen >= width) return til_strdup_n(body, blen);
    pad = (size_t)width - blen;
    r = (char*)til_alloc_obj((size_t)width + 1);
    if (zero && is_num && !align) {
        size_t sign = (body[0] == '-' || body[0] == '+') ? 1 : 0;
        memcpy(r, body, sign);
        memset(r + sign, '0', pad);
        memcpy(r + sign + pad, body + sign, blen - sign);
    } else {
        if (!align) align = is_num ? '>' : '<';
        if (align == '<') { memcpy(r, body, blen); memset(r + blen, ' ', pad); }
        else if (align == '>') { memset(r, ' ', pad); memcpy(r + pad, body, blen); }
        else { size_t left = pad / 2; memset(r, ' ', left); memcpy(r + left, body, blen); memset(r + left + blen, ' ', pad - left); }
    }
    r[width] = '\0';
    return r;
}

/* natives */
static const char* til_input(const char* prompt) {
    TIL_Buf b = {0, 0, 0};
    int c;
    if (prompt) { fputs(prompt, stdout); }
    fflush(stdout);
    while ((c = getchar()) != EOF && c != '\n') { char ch = (char)c; til_buf_put(&b, &ch, 1); }
    return til_buf_done(&b);
}
static uint64_t til_rng = 0x9E3779B97F4A7C15ULL;
static uint64_t til_rng_next(void) {
    til_rng ^= til_rng >> 12;
    til_rng ^= til_rng << 25;
    til_rng ^= til_rng >> 27;
    return til_rng * 0x2545F4914F6CDD1DULL;
}
static double til_random(void) { return (double)(til_rng_next() >> 11) / 9007199254740992.0; }
static int64_t til_random_range(int64_t lo, int64_t hi) {
    uint64_t r = til_rng_next();
    if (hi < lo) til_panic("random(lo, hi) needs lo <= hi");
    return lo + (int64_t)(r % ((uint64_t)hi - (uint64_t)lo + 1));
}
static void til_seed(int64_t s) { til_rng = (uint64_t)s ? (uint64_t)s : 0x9E3779B97F4A7C15ULL; }
static int64_t til_clock_ms(void) {
#if defined(CLOCK_MONOTONIC) && !defined(_WIN32)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
#else
    return (int64_t)(til_now_ms());
#endif
}
static const char* til_path(const char* who, const char* p) {
    if (!p || !*p || strlen(p) >= 256) {
        snprintf(til_panic_buf, sizeof til_panic_buf, "%s(): invalid path", who);
        til_panic(til_panic_buf);
    }
    return p;
}
static const char* til_read_file(const char* path) {
    FILE* f = fopen(til_path("read_file", path), "rb");
    TIL_Buf b = {0, 0, 0};
    char chunk[4096];
    size_t n;
    if (!f) til_panic_str("cannot read file: ", path, "");
    while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) {
        if (memchr(chunk, 0, n)) { fclose(f); til_panic_str("read_file(): ", path, " contains NUL bytes, which the native backend cannot represent"); }
        til_buf_put(&b, chunk, n);
    }
    if (ferror(f)) { fclose(f); til_panic_str("cannot read file: ", path, ""); }
    fclose(f);
    return til_buf_done(&b);
}
static void til_write_file(const char* path, const char* data) {
    FILE* f = fopen(til_path("write_file", path), "wb");
    size_t n = strlen(data);
    if (!f) til_panic_str("cannot write file: ", path, "");
    if (fwrite(data, 1, n, f) != n) { fclose(f); til_panic_str("cannot write file: ", path, ""); }
    if (fclose(f) != 0) til_panic_str("cannot write file: ", path, "");
}
static bool til_file_exists(const char* path) {
    FILE* f;
    if (!path || !*path || strlen(path) >= 256) til_path("file_exists", path);
    f = fopen(path, "rb");
    if (!f) return false;
    fclose(f);
    return true;
}
static TIL_Arr* til_args(void) {
    TIL_Arr* r = til_arr_new(til_argc);
    int i;
    for (i = 1; i < til_argc; i++) til_arr_push(r, til_slot_s(til_argv[i]));
    return r;
}
static int til_tostr_depth = 0;
static bool til_truthy_s(const char* s) { return s && *s; }
static bool til_truthy_arr(TIL_Arr* a) { return a && a->len; }
static bool til_truthy_map(TIL_Map* m) { return m && m->len; }
static void* til_nn(void* p, const char* field) { if (!p) til_panic_str("no field '", field, "' on nil"); return p; }
static void* til_nm(void* p, const char* method) { if (!p) til_panic_str("no method '", method, "' on nil"); return p; }
static double til_round_vm(double x) { return x != x ? x : copysign(floor(fabs(x) + 0.5), x); }
static int64_t til_min_i(int64_t a, int64_t b) { return b < a ? b : a; }
static int64_t til_max_i(int64_t a, int64_t b) { return b > a ? b : a; }
static double til_min_f(double a, double b) { return b < a ? b : a; }
static double til_max_f(double a, double b) { return b > a ? b : a; }
static const char* til_min_s(const char* a, const char* b) { return strcmp(b, a) < 0 ? b : a; }
static const char* til_max_s(const char* a, const char* b) { return strcmp(b, a) > 0 ? b : a; }
static TIL_Arr* til_str_bytes(const char* s) {
    TIL_Arr* r = til_arr_new((int64_t)strlen(s));
    for (; *s; s++) til_arr_push(r, til_slot_i((unsigned char)*s));
    return r;
}
static void til_sleep_ms(int64_t ms) {
#if !defined(_WIN32)
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = (time_t)(ms / 1000);
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {}
#else
    clock_t end = clock() + (clock_t)((double)ms * CLOCKS_PER_SEC / 1000.0);
    while (ms > 0 && clock() < end) {}
#endif
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#                      TYPED NATIVE BACKEND: C CODE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

if hasattr(til, 'NativeUnsupported'):
    NativeUnsupported = til.NativeUnsupported
else:                                               # til.py without the hook (older)
    class NativeUnsupported(til.CompileError):
        pass

PTR_KINDS = ('str', 'array', 'map', 'nil')
SCALAR_OPT = {'int': 'int', 'float': 'float', 'str': 'str', 'bool': 'bool', 'unk': 'int'}
UNSUPPORTED_NATIVES = {'bytes_to_str', 'screen_init', 'screen_clear', 'draw_pixel', 'draw_line',
                       'draw_rect', 'fill_rect', 'draw_circle', 'draw_text', 'get_pixel',
                       'screen_show', 'key_pressed', 'wait_key', 'screen_width', 'screen_height',
                       'energy_mode', 'iter'}


C_RESERVED = set("""
auto break case char const continue default do double else enum extern float for goto if inline int long
register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while
_Bool bool true false NULL errno stdin stdout stderr EOF INFINITY NAN main argc argv
printf puts malloc calloc realloc free strlen strcmp strncmp strstr strchr strcat strcpy memcpy memmove memset
memchr snprintf fflush fputs fprintf getchar fopen fclose fread fwrite ferror exit setjmp longjmp jmp_buf
sqrt pow fabs floor ceil round log exp sin cos tan fmod copysign isnan isinf clock time abs labs
nanosleep clock_gettime strtod error self _result
""".split())


def c_local(name: str) -> str:
    """C identifier for a TIL local: TIL names never collide with C keywords,
    the C library or the generated temporaries (which start with '_')."""
    m = til.CCodeGenerator.mangle_name(name)
    if m in C_RESERVED or m.startswith('_') or m.lower().startswith('til_'):
        return 'v_' + m
    return m


def c_global(name: str) -> str:
    return 'g_' + til.CCodeGenerator.mangle_name(name)


def c_string_opaque(text: str) -> str:
    """A C string literal for user text where only letters, digits and spaces stay
    readable (user text never appears as C tokens in the generated file)."""
    out = ['"']
    for byte in text.encode('utf-8'):
        ch = chr(byte)
        out.append(ch if (ch.isascii() and (ch.isalnum() or ch == ' ')) else '\\%03o' % byte)
    out.append('"')
    return ''.join(out)


def is_ptr(t) -> bool:
    """Values represented by a C pointer (NULL = None)."""
    return t.name in PTR_KINDS or is_struct(t) or is_adt(t)


def literal_default(node) -> bool:
    """A default argument that can be evaluated at the call site."""
    for n in walk(node):
        if isinstance(n, til.Identifier) and n.name not in ('NULL',):
            return False
        if not isinstance(n, (til.IntLit, til.FloatLit, til.StringLit, til.BoolLit, til.CharLit,
                              til.UnaryOp, til.Identifier, til.BinaryOp)):
            return False
    return True


class TypedCGen(til.CCodeGenerator):
    """C code generation driven by the inferred types (one C function per
    function instance; generic functions are monomorphised: f__int, f__str)."""

    MAX_INSTANCES = 32

    def __init__(self, ti: TypeInfer):
        super().__init__()
        self.ti = ti
        self.inst: Optional[Instance] = None
        self.helpers: Dict[str, str] = {}          # name -> C definition (generated per type)
        self.helper_protos: Dict[str, str] = {}
        self.typedefs: Dict[str, str] = {}         # Option/Result types for other payloads
        self.lambda_defs: List[str] = []
        self.uid = 0
        self.scopes: List[Dict[str, Tuple[str, Any]]] = []   # TIL name -> (C name, type)
        self.cnames_used: List[Set[str]] = []
        self.lambdas: Dict[str, Any] = {}          # C-level id -> lambda binding
        self.insts: Dict[Tuple[str, Tuple[str, ...]], Instance] = {}
        self.inst_names: Dict[Tuple[str, Tuple[str, ...]], str] = {}
        self.loop_depth = 0

    # ── errors ──────────────────────────────────────────────────────────────
    def unsupported(self, what: str, node=None):
        line = getattr(node, 'line', 0) if node is not None else 0
        where = f" (line {line})" if line else ""
        fn = f" in '{self.inst.key}'" if self.inst is not None and self.inst.key != '<globals>' else ""
        raise NativeUnsupported(f"native backend: {what}{fn}{where}")

    def fresh_id(self) -> int:
        self.uid += 1
        return self.uid

    # ── types ───────────────────────────────────────────────────────────────
    def T(self, node):
        return self.inst.t(node)

    def ct(self, t, node=None, what="a value") -> str:
        """C type of a TIL type."""
        if t is INT or t is UNK:
            return "int64_t"
        if t is FLOAT:
            return "double"
        if t is BOOL:
            return "bool"
        if t is STR:
            return "const char*"
        if t is NIL:
            return "void*"
        n = t.name
        if n == 'strish':
            return "const char*"
        if n == 'array':
            return "TIL_Arr*"
        if n == 'map':
            return "TIL_Map*"
        if is_struct(t):
            return self.mangle_name(nominal_name(t)) + "*"
        if n == 'option':
            return self.option_type(t, node)
        if n == 'result':
            return self.result_type(t, node)
        if t is DYN:
            self.unsupported(f"{what} has no single static type (it is dynamically typed)", node)
        if n == 'fn':
            self.unsupported(f"{what} is a function value (closures as values run on the VM)", node)
        if is_adt(t):
            return self.adt_c(nominal_name(t)) + "*"
        self.unsupported(f"{what} has type {show(t)}", node)

    def adt_c(self, name: str) -> str:
        return f"TIL_E_{self.mangle_name(name)}"

    def adt_ctor(self, name: str, variant: str, node) -> str:
        """Constructor function of an enum variant (payload in 8-byte slots)."""
        info = self.ti.adts[name]
        vnames = list(info.variants)
        tag = vnames.index(variant)
        payload = info.frozen[variant]
        c = self.adt_c(name)
        fname = f"til_mk_{self.mangle_name(name)}_{self.mangle_name(variant)}"
        if fname not in self.helpers:
            params = []
            sets = []
            for i, pt in enumerate(payload):
                self.check_slot_type(pt, node, f"a payload of {name}.{variant}")
                params.append(f"{self.ct(pt, node)} p{i}")
                sets.append(f"v->f[{i}] = {self.slot_make(pt, f'p{i}')};")
            self.add_helper(fname, f"static {c}* {fname}({', '.join(params) or 'void'})",
                            "{ " + f"{c}* v = ({c}*)til_alloc_obj(sizeof({c})); v->tag = {tag}; " +
                            " ".join(sets) + " return v; }")
        return fname

    def adt_defs(self) -> List[str]:
        lines = []
        for name, info in self.ti.adts.items():
            c = self.adt_c(name)
            width = max([len(v) for v in info.variants.values()] + [1])
            lines.append(f"struct {c} {{ int tag; TIL_Slot f[{width}]; }};")
            names = ", ".join(til.c_string_literal(f"{name}.{v}") for v in info.variants)
            lines.append(f"static const char* {c}_names[] = {{{names}}};")
        return lines

    def ret_ct(self, t) -> str:
        return "void" if t is NIL else self.ct(t, None, "a return value")

    def zero(self, t) -> str:
        if t is INT or t is UNK:
            return "0"
        if t is FLOAT:
            return "0.0"
        if t is BOOL:
            return "false"
        if t is STR:
            return '""'
        if t.name == 'option':
            return f"{self.option_ctor(t, None)}()"
        if t.name == 'result':
            return f"({self.result_type(t)}){{0}}"
        return "NULL"

    def option_type(self, t, node=None) -> str:
        inner = t.args[0]
        if inner.name in SCALAR_OPT:
            return f"TIL_Option_{SCALAR_OPT[inner.name]}"
        name = f"TIL_Opt_{canon(inner)}"
        if name not in self.typedefs:
            cti = self.ct(inner, node, "an Option payload")
            self.typedefs[name] = f"typedef struct {name} {{ bool has_value; {cti} value; }} {name};"
            sfx = canon(inner)
            self.add_helper(f"til_Some_{sfx}", f"static {name} til_Some_{sfx}({cti} v)",
                            f"{{ {name} o; o.has_value = true; o.value = v; return o; }}")
            self.add_helper(f"til_None_{sfx}", f"static {name} til_None_{sfx}(void)",
                            f"{{ {name} o; memset(&o, 0, sizeof o); return o; }}")
        return name

    def option_ctor(self, t, node, some=False) -> str:
        inner = t.args[0]
        self.option_type(t, node)
        sfx = SCALAR_OPT.get(inner.name) or canon(inner)
        return f"til_{'Some' if some else 'None'}_{sfx}"

    def result_type(self, t, node=None) -> str:
        ok, err = t.args
        name = f"TIL_Res_{canon(ok)}__{canon(err)}"
        if name not in self.typedefs:
            c_ok = self.ct(ok, node, "a Result value") if ok is not NIL else "int64_t"
            c_err = self.ct(err, node, "a Result error") if err is not NIL else "int64_t"
            self.typedefs[name] = (f"typedef struct {name} {{ bool is_ok; {c_ok} value; {c_err} error; }} "
                                   f"{name};")
            self.add_helper(f"til_Ok_{name}", f"static {name} til_Ok_{name}({c_ok} v)",
                            f"{{ {name} r; memset(&r, 0, sizeof r); r.is_ok = true; r.value = v; return r; }}")
            self.add_helper(f"til_Err_{name}", f"static {name} til_Err_{name}({c_err} e)",
                            f"{{ {name} r; memset(&r, 0, sizeof r); r.is_ok = false; r.error = e; return r; }}")
        return name

    def add_helper(self, name: str, proto: str, body: str):
        if name not in self.helpers:
            self.helper_protos[name] = proto + ";"
            self.helpers[name] = proto + " " + body

    # ── slots (array / map storage) ─────────────────────────────────────────
    def check_slot_type(self, t, node, what="an element"):
        if t.name in ('option', 'result'):
            self.unsupported(f"{what} of type {show(t)} inside a container", node)
        self.ct(t, node, what)

    def slot_get(self, t, slot: str) -> str:
        if t is INT or t is UNK:
            return f"({slot}).i"
        if t is BOOL:
            return f"(({slot}).i != 0)"
        if t is FLOAT:
            return f"({slot}).f"
        if t is STR:
            return f"({slot}).s"
        return f"(({self.ct(t)})({slot}).p)"

    def slot_make(self, t, code: str) -> str:
        if t is INT or t is UNK or t is BOOL:
            return f"til_slot_i((int64_t)({code}))"
        if t is FLOAT:
            return f"til_slot_f({code})"
        if t is STR:
            return f"til_slot_s({code})"
        return f"til_slot_p((void*)({code}))"

    def slot_field(self, t) -> str:
        return {'int': 'i', 'bool': 'i', 'unk': 'i', 'float': 'f', 'str': 's'}.get(t.name, 'p')

    def key_kind(self, t, node) -> int:
        if t is STR:
            return 1
        if t is BOOL:
            return 2
        if t is INT or t is UNK:
            return 0
        self.unsupported(f"map keys of type {show(t)}", node)

    # ── conversions ─────────────────────────────────────────────────────────
    def coerce(self, code: str, src, dst, node=None) -> str:
        """Convert a value for a place of type dst (the VM converts numbers at
        annotated declarations, parameters and returns)."""
        if dst is UNK or src is dst:
            return code
        if dst.name == 'strish':
            if src is STR or (src.name == 'strish' and src.args[0] is dst.args[0]):
                return code
            if src is dst.args[0] or src is UNK:
                return self.to_s(code, src, node)
            self.unsupported(f"a value of type {show(src)} is used where {show(dst)} is expected", node)
        if dst is FLOAT and src in (INT, BOOL, UNK):
            return f"(double)({code})"
        if dst is INT and src is FLOAT:
            return f"til_f2i({code})"
        if dst is INT and src in (BOOL, UNK):
            return f"(int64_t)({code})"
        if dst is FLOAT and src is NIL:
            return "0.0"
        if dst is INT and src is NIL:
            return "0"
        if src is NIL:
            if is_ptr(dst):
                return f"((void)({code}), ({self.ct(dst)})NULL)" if code not in ('NULL',) else "NULL"
            if dst.name == 'option':
                return f"{self.option_ctor(dst, node)}()"
        if src is UNK and dst is not NIL:
            return code
        if self.ct(src, node) == self.ct(dst, node):
            return code
        self.unsupported(f"a value of type {show(src)} is used where {show(dst)} is expected", node)

    def to_s(self, code: str, t, node=None) -> str:
        """C string (unquoted, like str(x) / print) of a value."""
        if t is STR or t.name == 'strish':
            return f"til_s({code})"
        if t is INT or t is UNK:
            return f"til_int_to_str((int64_t)({code}))"
        if t is FLOAT:
            return f"til_float_str({code})"
        if t is BOOL:
            return f"til_bool_to_str({code})"
        if t is NIL:
            return f"((void)({code}), \"None\")"
        return f"{self.tostr_helper(t, node)}({code})"

    def append_quoted(self, buf: str, code: str, t, node) -> str:
        if t is STR:
            return f"til_buf_q({buf}, {code});"
        return f"til_buf_add({buf}, {self.to_s(code, t, node)});"

    def tostr_helper(self, t, node=None) -> str:
        name = f"til_tostr_{canon(t)}"
        if name in self.helpers:
            return name
        ctype = self.ct(t, node, "a printed value")
        proto = f"static const char* {name}({ctype} x)"
        self.helper_protos[name] = proto + ";"
        self.helpers[name] = proto + " { return \"\"; }"     # placeholder (recursive types)
        lines = ["{", "    TIL_Buf b = {0, 0, 0};", "    const char* r;"]
        if t.name == 'array':
            e = t.args[0]
            self.check_slot_type(e, node)
            lines += ["    int64_t i;", "    if (!x) return \"None\";",
                      "    if (til_tostr_depth > 16) return \"...\";", "    til_tostr_depth++;",
                      "    til_buf_add(&b, \"[\");",
                      "    for (i = 0; i < x->len; i++) {",
                      "        if (i) til_buf_add(&b, \", \");",
                      "        " + self.append_quoted("&b", self.slot_get(e, "x->data[i]"), e, node),
                      "    }", "    til_buf_add(&b, \"]\");"]
        elif t.name == 'map':
            k, v = t.args
            self.check_slot_type(k, node, "a map key")
            self.check_slot_type(v, node, "a map value")
            lines += ["    int64_t i;", "    if (!x) return \"None\";",
                      "    if (til_tostr_depth > 16) return \"...\";", "    til_tostr_depth++;",
                      "    til_buf_add(&b, \"{\");",
                      "    for (i = 0; i < x->len; i++) {",
                      "        if (i) til_buf_add(&b, \", \");",
                      "        " + self.append_quoted("&b", self.slot_get(k, "x->keys[i]"), k, node),
                      "        til_buf_add(&b, \": \");",
                      "        " + self.append_quoted("&b", self.slot_get(v, "x->vals[i]"), v, node),
                      "    }", "    til_buf_add(&b, \"}\");"]
        elif is_struct(t):
            sname = nominal_name(t)
            info = self.ti.structs[sname]
            lines += ["    if (!x) return \"None\";", "    if (til_tostr_depth > 16) return \"...\";",
                      "    til_tostr_depth++;"]
            if not info.order:
                lines.append(f"    til_buf_add(&b, {til.c_string_literal(sname)});")
            else:
                lines.append(f"    til_buf_add(&b, {til.c_string_literal(sname + ' { ')});")
                for i, fname in enumerate(info.order):
                    ft = info.frozen[fname]
                    sep = ", " if i else ""
                    lines.append(f"    til_buf_add(&b, {til.c_string_literal(sep + fname + ': ')});")
                    lines.append("    " + self.append_quoted("&b", f"x->{fname}", ft, node))
                lines.append("    til_buf_add(&b, \" }\");")
        elif is_adt(t):
            aname = nominal_name(t)
            info = self.ti.adts[aname]
            lines += ["    if (!x) return \"None\";", "    if (til_tostr_depth > 16) return \"...\";",
                      "    til_tostr_depth++;", f"    til_buf_add(&b, {self.adt_c(aname)}_names[x->tag]);",
                      "    switch (x->tag) {"]
            for tag, (vname, payload) in enumerate(info.frozen.items()):
                if not payload:
                    continue
                lines.append(f"    case {tag}:")
                lines.append("        til_buf_add(&b, \"(\");")
                for i, pt in enumerate(payload):
                    self.check_slot_type(pt, node, f"a payload of {aname}.{vname}")
                    if i:
                        lines.append("        til_buf_add(&b, \", \");")
                    lines.append("        " + self.append_quoted("&b", self.slot_get(pt, f"x->f[{i}]"), pt, node))
                lines.append("        til_buf_add(&b, \")\");")
                lines.append("        break;")
            lines += ["    default: break;", "    }"]
        elif t.name == 'option':
            inner = t.args[0]
            lines += ["    if (!x.has_value) return \"None\";", "    if (til_tostr_depth > 16) return \"...\";",
                      "    til_tostr_depth++;", "    til_buf_add(&b, \"Some(\");",
                      "    " + self.append_quoted("&b", "x.value", inner, node), "    til_buf_add(&b, \")\");"]
        elif t.name == 'result':
            ok, err = t.args
            lines += ["    if (til_tostr_depth > 16) return \"...\";", "    til_tostr_depth++;",
                      "    if (x.is_ok) {", "        til_buf_add(&b, \"Ok(\");",
                      "        " + (self.append_quoted("&b", "x.value", ok, node) if ok is not NIL
                                    else "til_buf_add(&b, \"None\");"),
                      "    } else {", "        til_buf_add(&b, \"Err(\");",
                      "        " + (self.append_quoted("&b", "x.error", err, node) if err is not NIL
                                    else "til_buf_add(&b, \"None\");"),
                      "    }", "    til_buf_add(&b, \")\");"]
        else:
            self.unsupported(f"printing a value of type {show(t)}", node)
        lines += ["    til_tostr_depth--;", "    r = til_buf_done(&b);", "    return r;", "}"]
        self.helpers[name] = proto + " " + "\n".join(lines)
        return name

    @staticmethod
    def mixed_cmp(a: str, ta, b: str, tb, op: str) -> Optional[str]:
        """Exact int-vs-float comparison (C would round the int to a double)."""
        if {ta.name, tb.name} != {'int', 'float'}:
            return None
        small = lambda c: c.lstrip('-').isdigit() and abs(int(c)) <= (1 << 53)   # noqa: E731
        if (ta is INT and small(a)) or (tb is INT and small(b)):
            return None
        fn = "til_cmp_if" if ta is INT else "til_cmp_fi"
        test = {'==': "== 0", '!=': "!= 0", '<': "== -1", '<=': "<= 0", '>': "== 1", '>=': ">= 0"}[op]
        if op in ('<=', '>='):
            return f"({{ int _c = {fn}({a}, {b}); _c != 2 && _c {test.split(' ')[0]} {test.split(' ')[1]}; }})"
        return f"({fn}({a}, {b}) {test})"

    def eq_code(self, a: str, ta, b: str, tb, node=None) -> str:
        """C boolean for `a == b` with the VMs' rules (structural equality)."""
        num = ('int', 'float', 'unk')
        if ta.name in num and tb.name in num:
            mixed = self.mixed_cmp(a, ta, b, tb, '==')
            return mixed if mixed else f"({a} == {b})"
        if ta is BOOL and tb is BOOL:
            return f"({a} == {b})"
        if ta is NIL or tb is NIL:
            other, oc, nc = (tb, b, a) if ta is NIL else (ta, a, b)
            if other is NIL:
                return f"((void)({a}), (void)({b}), true)"
            if is_ptr(other):
                return f"((void)({nc}), ({oc}) == NULL)"
            if other.name == 'option':
                return f"((void)({nc}), !({oc}).has_value)"
            return f"((void)({a}), (void)({b}), false)"
        if ta is STR and tb is STR:
            return f"til_str_eqn({a}, {b})"
        if ta.name != tb.name:
            return f"((void)({a}), (void)({b}), false)"
        return f"{self.eq_helper(ta, tb, node)}({a}, {b})"

    def eq_helper(self, ta, tb, node=None) -> str:
        if canon(ta) != canon(tb):
            if self.ct(ta, node) != self.ct(tb, node):
                self.unsupported(f"comparing {show(ta)} with {show(tb)}", node)
        t = ta
        name = f"til_eq_{canon(t)}"
        if name in self.helpers:
            return name
        c = self.ct(t, node, "a compared value")
        proto = f"static bool {name}({c} a, {c} b)"
        self.helper_protos[name] = proto + ";"
        self.helpers[name] = proto + " { return false; }"
        if t.name == 'array':
            e = t.args[0]
            self.check_slot_type(e, node)
            body = ("{ int64_t i; if (a == b) return true; if (!a || !b || a->len != b->len) return false; "
                    "for (i = 0; i < a->len; i++) if (!" +
                    self.eq_code(self.slot_get(e, "a->data[i]"), e, self.slot_get(e, "b->data[i]"), e, node) +
                    ") return false; return true; }")
        elif t.name == 'map':
            k, v = t.args
            kind = self.key_kind(k, node)
            body = ("{ int64_t i; if (a == b) return true; if (!a || !b || a->len != b->len) return false; "
                    "for (i = 0; i < a->len; i++) { int64_t j = til_map_find(b, a->keys[i], " + str(kind) +
                    "); if (j < 0 || !" +
                    self.eq_code(self.slot_get(v, "a->vals[i]"), v, self.slot_get(v, "b->vals[j]"), v, node) +
                    ") return false; } return true; }")
        elif is_struct(t):
            info = self.ti.structs[nominal_name(t)]
            parts = [self.eq_code(f"a->{f}", info.frozen[f], f"b->{f}", info.frozen[f], node) for f in info.order]
            body = ("{ if (a == b) return true; if (!a || !b) return false; return " +
                    (" && ".join(parts) or "true") + "; }")
        elif is_adt(t):
            info = self.ti.adts[nominal_name(t)]
            cases = []
            for tag, payload in enumerate(info.frozen.values()):
                parts = [self.eq_code(self.slot_get(pt, f"a->f[{i}]"), pt, self.slot_get(pt, f"b->f[{i}]"), pt, node)
                         for i, pt in enumerate(payload)]
                cases.append(f"case {tag}: return {' && '.join(parts) or 'true'};")
            body = ("{ if (a == b) return true; if (!a || !b || a->tag != b->tag) return false; switch (a->tag) { " +
                    " ".join(cases) + " default: return true; } }")
        elif t.name == 'option':
            inner = t.args[0]
            body = ("{ if (a.has_value != b.has_value) return false; if (!a.has_value) return true; return " +
                    self.eq_code("a.value", inner, "b.value", inner, node) + "; }")
        elif t.name == 'result':
            ok, err = t.args
            body = ("{ if (a.is_ok != b.is_ok) return false; return a.is_ok ? " +
                    self.eq_code("a.value", ok, "b.value", ok, node) + " : " +
                    self.eq_code("a.error", err, "b.error", err, node) + "; }")
        else:
            self.unsupported(f"comparing values of type {show(t)}", node)
        self.helpers[name] = proto + " " + body
        return name

    def truthy(self, node) -> str:
        code = self.x(node)
        return self.truthy_code(code, self.T(node))

    def truthy_code(self, code: str, t) -> str:
        if t is BOOL:
            return code
        if t is INT or t is UNK:
            return f"({code} != 0)"
        if t is FLOAT:
            return f"({code} != 0.0)"
        if t is STR:
            return f"til_truthy_s({code})"
        if t.name == 'array':
            return f"til_truthy_arr({code})"
        if t.name == 'map':
            return f"til_truthy_map({code})"
        if t.name == 'option':
            return f"({code}).has_value"
        if t is NIL:
            return f"((void)({code}), false)"
        if is_struct(t) or is_adt(t):
            return f"({code} != NULL)"
        if t.name == 'result':
            return f"((void)({code}), true)"
        self.ct(t, None, "a condition")
        return f"({code} != 0)"

    # ── scopes ──────────────────────────────────────────────────────────────
    def push_scope(self):
        self.scopes.append({})
        self.cnames_used.append(set())

    def pop_scope(self):
        self.scopes.pop()
        self.cnames_used.pop()

    def declare_var(self, name: str, t, node=None) -> str:
        base = c_local(name) if name != 'error' else 'error'
        cname = base
        used = set().union(*self.cnames_used) if self.cnames_used else set()
        if cname in self.cnames_used[-1] or cname in ('self', '_result') or (cname in used and False):
            cname = f"{base}__{self.fresh_id()}"
        self.cnames_used[-1].add(cname)
        self.scopes[-1][name] = (cname, t)
        return cname

    def lookup_var(self, name: str):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    # ── driver ──────────────────────────────────────────────────────────────
    def generate(self, program) -> str:
        self.module_names = set(getattr(program, 'module_names', set()) or ())
        self._uses_timeout = any(isinstance(n, til.TimeoutExpr) for st in program.statements for n in walk(st))
        self._globals = []
        self._type_aliases = {}
        self._trait_defs = {}
        for st in program.statements:
            if isinstance(st, til.StructDef):
                self.structs[st.name] = st
            elif isinstance(st, til.FuncDef):
                self.functions[st.name] = st
            elif isinstance(st, til.EnumDef):
                if not any(v.fields for v in st.variants):
                    self.enums[st.name] = st
            elif isinstance(st, til.ImplBlock):
                self.impl_methods.setdefault(st.type_name, []).extend(st.methods)
            elif isinstance(st, til.VarDecl):
                self._globals.append(st)
            elif isinstance(st, til.TypeAlias):
                self._type_aliases[st.name] = st
            elif isinstance(st, til.TraitDef):
                self._trait_defs[st.name] = st
            elif isinstance(st, til.CapabilityType):
                if not hasattr(self, '_capabilities'):
                    self._capabilities = {}
                self._capabilities[st.name] = st
        self.build_instances()

        # function bodies first: they decide which helper types are needed
        functions = []
        for ikey, inst in list(self.insts.items()):
            try:
                functions.append((ikey, self.function_text(inst)))
            except NativeUnsupported:
                if ikey in self.reachable:
                    raise
                # dead code the native backend cannot express: leave it out
        kept = {ikey for ikey, _ in functions}
        for ikey in self.reachable:
            if ikey not in kept and ikey in self.insts:
                self.unsupported(f"'{ikey[0]}' cannot be compiled")
        global_lines = self.global_decls()
        struct_lines = self.struct_defs()
        self.no_main_inits = []
        if ('main', ()) not in self.insts:
            # no main(): the program is its global initialisers
            self.output, self.indent = [], 0
            self.scopes, self.cnames_used, self._recover_stack = [], [], []
            self.global_inits()
            self.no_main_inits = list(self.output)

        self.output = []
        self.emit_header()
        self.emit_types()
        self.emit_enums()
        self.emit_helpers()
        self.emit_optresult_methods()
        self.emit_raw(TYPED_RUNTIME)
        self.emit_raw("// Forward Declarations")
        for name in self.ti.structs:
            c = self.mangle_name(name)
            self.emit_raw(f"typedef struct {c} {c};")
        for name in self.ti.adts:
            c = self.adt_c(name)
            self.emit_raw(f"typedef struct {c} {c};")
        for line in self.adt_defs():
            self.emit_raw(line)
        for text in self.typedefs.values():
            self.emit_raw(text)
        self.emit_type_aliases_typed()
        for line in struct_lines:
            self.emit_raw(line)
        if self.helper_protos:
            self.emit_raw("// Typed helpers")
            for proto in self.helper_protos.values():
                self.emit_raw(proto)
        for line in global_lines:
            self.emit_raw(line)
        self.emit_raw("// Function Prototypes")
        for ikey, (proto, _text, comment) in functions:
            self.emit_raw(f"{proto};{comment}")
        self.emit_raw("")
        if self.lambda_defs:
            self.emit_raw("// Lambda Functions")
            for text in self.lambda_defs:
                self.emit_raw(text)
        for text in self.helpers.values():
            self.emit_raw(text)
        self.emit_capabilities()
        self.emit_raw("// TIL Functions")
        for ikey, (_proto, text, _comment) in functions:
            self.emit_raw(text)
        self.emit_raw("// Main Entry Point")
        self.emit_raw("int main(int argc, char** argv) {")
        self.emit_raw("    til_argc = argc;")
        self.emit_raw("    til_argv = argv;")
        if ('main', ()) in self.insts:
            self.emit_raw("    til_main();")
        else:
            self.emit_raw("    (void)argc; (void)argv;")
            for line in self.no_main_inits:
                self.emit_raw("    " + line)
        self.emit_raw("    return 0;")
        self.emit_raw("}")
        return '\n'.join(self.output)

    def emit_type_aliases_typed(self):
        if not self._type_aliases:
            return
        self.emit_raw("// Type Aliases")
        for name, alias in self._type_aliases.items():
            t = self.ti.ann(alias.type) if alias.type is not None else INT
            try:
                c = self.ct(t if t is not None else INT)
            except NativeUnsupported:
                continue
            self.emit_raw(f"typedef {c} {self.mangle_name(name)};")
        self.emit_raw("")

    def build_instances(self):
        ti = self.ti
        queue: List[Instance] = []
        counts: Dict[str, int] = {}

        def want(key, params):
            ikey = (key, tuple(canon(p) for p in params))
            if ikey not in self.insts:
                counts[key] = counts.get(key, 0) + 1
                if counts[key] > self.MAX_INSTANCES:
                    self.unsupported(f"'{key}' is used with more than {self.MAX_INSTANCES} type combinations")
                inst = ti.specialize(key, params)
                inst.ikey = ikey
                self.insts[ikey] = inst
                queue.append(inst)
            return ikey

        roots = []
        for key in ti.funcs:
            if not ti.is_generic(key):
                roots.append(want(key, ti.scheme_params(key)))
        self.ginst = ti.specialize_globals()
        self.ginst.ikey = ('<globals>', ())
        edges: Dict[Any, Set] = {}
        work = [self.ginst] + queue
        queue.clear()
        while work:
            inst = work.pop()
            targets = set()
            for key, params in inst.ctx.calls.values():
                if key in ti.funcs:
                    targets.add(want(key, [deep(p, UNK) for p in params]))
            edges[inst.ikey] = targets
            work.extend(queue)
            queue.clear()
        reach = set()
        stack = [('main', ()), ('<globals>', ())]
        while stack:
            k = stack.pop()
            if k in reach:
                continue
            reach.add(k)
            stack.extend(edges.get(k, ()))
        self.reachable = {k for k in reach if k in self.insts}
        for ikey, inst in self.insts.items():
            key = ikey[0]
            info = ti.funcs[key]
            if info.owner is None:
                base = f"til_{self.mangle_name(info.name)}"
            else:
                base = f"{self.mangle_name(info.owner)}_{self.mangle_name(info.name)}"
            if ti.is_generic(key):
                # explicit type parameters name the instance (max_of__int); parameters
                # without annotation (each its own type variable) are added after them
                ps = [canon(t) for t in getattr(inst, 'tparams', {}).values()]
                modes = getattr(inst, 'modes', [None] * len(inst.params))
                ps += [canon(p) for p, prm, mode in zip(inst.params, info.fdef.params, modes)
                       if prm.name != 'self' and mode is None]
                base += "__" + ("_".join(ps) if ps else "0")
                while base in self.inst_names.values():
                    base += "_"
            self.inst_names[ikey] = base

    def global_decls(self) -> List[str]:
        lines = []
        if not self._globals:
            return lines
        lines.append("// Global Variables")
        self.inst = self.ginst
        self.scopes, self.cnames_used = [], []
        for decl in self._globals:
            g = self.ti.globals[decl.name]
            t = g.frozen
            c = self.ct(t, decl, f"global '{decl.name}'")
            cname = c_global(decl.name)
            if decl.value is not None and not decl.deferred and \
                    isinstance(decl.value, (til.IntLit, til.FloatLit, til.StringLit, til.BoolLit, til.CharLit)):
                lines.append(f"static {c} {cname} = {self.x(decl.value)};")
            else:
                lines.append(f"static {c} {cname};")
        lines.append("")
        return lines

    def struct_defs(self) -> List[str]:
        lines = []
        if not self.ti.structs:
            return lines
        lines.append("// Struct Definitions")
        for name, info in self.ti.structs.items():
            c = self.mangle_name(name)
            fields = []
            for fname in info.order:
                ft = info.frozen[fname]
                try:
                    fields.append((fname, self.ct(ft)))
                except NativeUnsupported:
                    fields.append((fname, "int64_t /* dynamic: not usable natively */"))
            lines.append(f"struct {c} {{")
            for fname, fc in fields:
                lines.append(f"    {fc.split(' /*')[0]} {fname};")
            if not fields:
                lines.append("    char _empty;")
            lines.append("};")
            params = ", ".join(f"{fc.split(' /*')[0]} _{fname}" for fname, fc in fields) or "void"
            lines.append(f"static {c}* {c}_create({params}) {{")
            lines.append(f"    {c}* _self = ({c}*)til_alloc_obj(sizeof({c}));")
            for fname, _fc in fields:
                lines.append(f"    _self->{fname} = _{fname};")
            lines.append("    return _self;")
            lines.append("}")
            lines.append("")
        return lines

    # ── functions ───────────────────────────────────────────────────────────
    def function_text(self, inst: Instance):
        info = inst.info
        fdef = info.fdef
        self.inst = inst
        self.output = []
        self.indent = 0
        self.scopes, self.cnames_used = [], []
        self.lambdas = {}
        self._recover_stack = []
        self._loop_depth = 0
        if inst.conflicts:
            node, text = inst.conflicts[0]
            self.unsupported(f"conflicting types ({text})", node)
        is_main = info.key == 'main'
        self.cur_ret = NIL if is_main else inst.ret
        ret_c = "void" if is_main else self.ret_ct(inst.ret)
        self.push_scope()
        params = []
        for p, t in zip(fdef.params, inst.params):
            if p.name == 'self' and info.owner:
                params.append(f"{self.mangle_name(info.owner)}* self")
                self.scopes[-1]['self'] = ('self', t)
                self.cnames_used[-1].add('self')
                continue
            ctype = self.ct(t, fdef, f"parameter '{p.name}'")
            params.append(f"{ctype} {self.declare_var(p.name, t)}")
        name = self.inst_names[inst.ikey]
        attrs = []
        if fdef.level == 0:
            attrs.append("__attribute__((always_inline)) inline")
        elif fdef.level == 1:
            attrs.append("inline")
        if fdef.effects and 'pure' in fdef.effects and inst.ret in (INT, FLOAT, BOOL) \
                and not fdef.requires and not fdef.ensures:
            attrs.append("__attribute__((pure))")
        for attr in fdef.attributes:
            if attr == "noinline":
                attrs.append("__attribute__((noinline))")
            elif attr == "hot":
                attrs.append("__attribute__((hot))")
            elif attr == "cold":
                attrs.append("__attribute__((cold))")
            elif attr.startswith("energy:"):
                opt = {"low": "__attribute__((optimize(\"Os\")))",
                       "high": "__attribute__((optimize(\"O3\")))"}.get(attr[len("energy:"):].strip())
                if opt:
                    attrs.append(opt)
        static = "static " if info.owner else ""
        proto = f"{static}{ret_c} {name}({', '.join(params) or 'void'})"
        level_names = {0: "Hardware", 1: "Systems", 2: "Safe", 3: "Script", 4: "Formal"}
        comment = f"  // Level {fdef.level}: {level_names.get(fdef.level, '?')}" if not info.owner else ""
        head = f"{' '.join(attrs) + ' ' if attrs else ''}{proto} {{"
        self.emit_raw(f"// {info.key}: {inst.signature()}" if self.ti.is_generic(info.key) else
                      f"// Level {fdef.level}: {level_names.get(fdef.level, '?')}")
        self.emit_raw(head)
        self.indent = 1
        if self._uses_timeout and not is_main:
            self.emit("til_check_deadline();")
        if is_main:
            self.global_inits()
        for k, req in enumerate(fdef.requires):
            text = fdef.requires_text[k] if k < len(fdef.requires_text) else ""
            msg = f"contract violation: {fdef.name} requires {text}".strip()
            self.emit(f"if (!{self.truthy(req)}) til_panic({til.c_string_literal(msg)});")
        self.ensures = fdef.ensures or []
        self.ensures_text = fdef.ensures_text or []
        self.fname = fdef.name
        self.propagates = isinstance(fdef.ret_type, (til.OptionType, til.ResultType))
        self.stmts(fdef.body)
        if not is_main and inst.ret is not NIL and falls_through(fdef.body):
            self.emit(f"return {self.zero(inst.ret)};")
        self.indent = 0
        self.emit_raw("}")
        self.emit_raw("")
        self.pop_scope()
        return proto, '\n'.join(self.output), comment

    def global_inits(self):
        """Program start: initialise globals whose values are not literals (before main's body)."""
        saved = self.inst
        self.inst = self.ginst
        for decl in self._globals:
            if decl.value is None or decl.deferred:
                continue
            if isinstance(decl.value, (til.IntLit, til.FloatLit, til.StringLit, til.BoolLit, til.CharLit)):
                continue
            g = self.ti.globals[decl.name]
            code = self.coerce(self.x(decl.value), self.T(decl.value), g.frozen, decl)
            self.emit(f"{c_global(decl.name)} = {code};")
        self.inst = saved

    # ── statements ──────────────────────────────────────────────────────────
    def stmts(self, body):
        self.push_scope()
        for st in (body.statements if isinstance(body, til.Block) else [body]):
            self.stmt(st)
        self.pop_scope()

    def braced(self, body):
        self.indent += 1
        self.stmts(body)
        self.indent -= 1

    def stmt(self, n):
        m = getattr(self, 'g_' + type(n).__name__, None)
        if m is not None:
            m(n)
        elif isinstance(n, (til.FuncDef, til.StructDef, til.EnumDef, til.ImplBlock, til.TraitDef,
                            til.TypeAlias, til.Import, til.CapabilityType)):
            pass
        else:
            code = self.x(n)
            if code:
                self.emit(f"{code};")

    def g_ExprStmt(self, n):
        e = n.expr
        if isinstance(e, til.Call) and isinstance(e.func, til.Identifier) and \
                e.func.name in ('print', 'println') and self.lookup_var(e.func.name) is None:
            self.print_call(e)
            return
        if isinstance(e, (til.RetryExpr, til.TimeoutExpr)):
            self.emit(self.guarded(e, want_value=False) + ";")
            return
        if isinstance(e, til.MatchExpr):
            self.match_stmt(e)
            return
        if isinstance(e, til.IntentBlock):
            self.g_IntentBlock(e)
            return
        code = self.x(e, statement=True)
        if code:
            self.emit(f"{code};")

    def g_Block(self, n):
        self.emit("{")
        self.braced(n)
        self.emit("}")

    def g_VarDecl(self, n):
        t = self.inst.decl(n)
        if isinstance(n.value, til.Lambda):
            self.bind_lambda(n.name, n.value)
            return
        c = self.ct(t, n, f"variable '{n.name}'")
        if n.value is not None:
            code = self.coerce(self.x(n.value), self.T(n.value), t, n)
        else:
            code = self.zero(t)
        cname = self.declare_var(n.name, t, n)
        self.emit(f"{c} {cname} = {code};")

    def g_GlobalInit(self, n):
        decl = n.decl
        g = self.ti.globals[decl.name]
        code = self.coerce(self.x(decl.value), self.T(decl.value), g.frozen, n)
        self.emit(f"{c_global(decl.name)} = {code};")

    def place(self, target):
        """(C lvalue, type) of an identifier target."""
        found = self.lookup_var(target.name)
        if found is not None:
            if isinstance(found[0], dict):
                self.unsupported("assigning to a variable that holds a lambda", target)
            return found[0], found[1]
        g = self.ti.globals.get(target.name)
        if g is not None:
            return c_global(target.name), g.frozen
        self.unsupported(f"assignment to unknown name '{target.name}'", target)

    def g_Assignment(self, n):
        target, op = n.target, n.op
        vt = self.T(n.value)
        if isinstance(target, til.Identifier):
            lv, t = self.place(target)
            if op == '=':
                self.emit(f"{lv} = {self.coerce(self.x(n.value), vt, t, n)};")
            else:
                res = self.arith(op[:-1], lv, t, self.x(n.value), vt, self.T(target) if False else None, n)
                self.emit(f"{lv} = {self.coerce(res[0], res[1], t, n)};")
            return
        if isinstance(target, til.Index):
            ot = self.T(target.obj)
            k = self.fresh_id()
            if ot.name == 'array':
                e = ot.args[0]
                self.check_slot_type(e, n)
                self.emit("{")
                self.indent += 1
                self.emit(f"TIL_Slot* _p{k} = til_arr_at({self.x(target.obj)}, {self.int_code(target.index)});")
                if op == '=':
                    val = self.coerce(self.x(n.value), vt, e, n)
                else:
                    res = self.arith(op[:-1], self.slot_get(e, f"*_p{k}"), e, self.x(n.value), vt, None, n)
                    val = self.coerce(res[0], res[1], e, n)
                self.emit(f"*_p{k} = {self.slot_make(e, val)};")
                self.indent -= 1
                self.emit("}")
            elif ot.name == 'map':
                kt, v = ot.args
                self.check_slot_type(kt, n, "a map key")
                self.check_slot_type(v, n, "a map value")
                kind = self.key_kind(kt, n)
                self.emit("{")
                self.indent += 1
                self.emit(f"TIL_Map* _m{k} = {self.x(target.obj)};")
                self.emit(f"TIL_Slot _k{k} = {self.slot_make(kt, self.coerce(self.x(target.index), self.T(target.index), kt, n))};")
                if op == '=':
                    val = self.coerce(self.x(n.value), vt, v, n)
                else:
                    cur = self.slot_get(v, f"til_map_get(_m{k}, _k{k}, {kind})")
                    res = self.arith(op[:-1], cur, v, self.x(n.value), vt, None, n)
                    val = self.coerce(res[0], res[1], v, n)
                self.emit(f"til_map_set(_m{k}, _k{k}, {self.slot_make(v, val)}, {kind});")
                self.indent -= 1
                self.emit("}")
            elif ot is STR:
                self.emit(f"(void)({self.x(target.obj)}); til_panic(\"strings are immutable\");")
            else:
                self.unsupported(f"assignment into a value of type {show(ot)}", n)
            return
        if isinstance(target, til.Attribute):
            ot = self.T(target.obj)
            if not is_struct(ot):
                self.unsupported(f"field assignment on a value of type {show(ot)}", n)
            info = self.ti.structs[nominal_name(ot)]
            ft = info.frozen.get(target.attr)
            if ft is None:
                self.unsupported(f"unknown field '{target.attr}'", n)
            self.ct(ft, n, f"field '{target.attr}'")
            k = self.fresh_id()
            sc = self.mangle_name(nominal_name(ot))
            obj = self.x(target.obj)
            self.emit("{")
            self.indent += 1
            self.emit(f"{sc}* _o{k} = ({sc}*)til_nn({obj}, {til.c_string_literal(target.attr)});")
            if op == '=':
                val = self.coerce(self.x(n.value), vt, ft, n)
            else:
                res = self.arith(op[:-1], f"_o{k}->{target.attr}", ft, self.x(n.value), vt, None, n)
                val = self.coerce(res[0], res[1], ft, n)
            self.emit(f"_o{k}->{target.attr} = {val};")
            self.indent -= 1
            self.emit("}")
            return
        self.unsupported("this assignment target", n)

    def g_If(self, n):
        first = True
        for cond, body in [(n.condition, n.then_body)] + list(n.elifs):
            c = self.truthy(cond)
            self.emit(f"{'if' if first else '} else if'} ({c}) {{")
            first = False
            self.braced(body)
        if n.else_body is not None:
            self.emit("} else {")
            self.braced(n.else_body)
        self.emit("}")

    def loop_body(self, body):
        self.indent += 1
        if self._uses_timeout:
            self.emit("til_check_deadline();")
        self._loop_depth += 1
        self.stmts(body)
        self._loop_depth -= 1
        self.indent -= 1

    def g_While(self, n):
        # the condition is re-evaluated (with its temporaries) on every iteration
        self.emit(f"while ({self.truthy(n.condition)}) {{")
        self.loop_body(n.body)
        self.emit("}")

    def g_Loop(self, n):
        self.emit("while (1) {")
        self.loop_body(n.body)
        self.emit("}")

    def g_For(self, n):
        vt = self.inst.decl(n)
        k = self.fresh_id()
        it = n.iter
        self.push_scope()
        if isinstance(it, til.Range):
            if vt not in (INT, FLOAT, UNK):
                self.unsupported("a range with non-numeric bounds", n)
            c = self.ct(vt)
            start = self.coerce(self.x(it.start), self.T(it.start), vt, n)
            end = self.x(it.end)
            self.emit("{")
            self.indent += 1
            self.emit(f"{self.ct(self.T(it.end)) if self.T(it.end) in (INT, FLOAT) else 'int64_t'} _end{k} = {end};")
            cv = self.declare_var(n.var, vt, n)
            op = "<=" if it.inclusive else "<"
            self.emit(f"for ({c} {cv} = {start}; {cv} {op} _end{k}; {cv}++) {{")
            self.loop_body(n.body)
            self.emit("}")
            self.indent -= 1
            self.emit("}")
            self.pop_scope()
            return
        tt = self.T(it)
        src = self.x(it)
        self.emit("{")
        self.indent += 1
        if tt.name == 'array' or tt.name == 'map':
            if tt.name == 'array':
                tmp = f"_arr_{self.mangle_name(n.var)}" if isinstance(it, til.ArrayLit) else f"_it{k}"
                self.emit(f"TIL_Arr* {tmp} = {src};")
                e = tt.args[0]
            else:
                tmp = f"_it{k}"
                self.check_slot_type(tt.args[0], n, "a map key")
                self.emit(f"TIL_Arr* {tmp} = til_map_keys({src});")
                e = tt.args[0]
            self.check_slot_type(e, n)
            self.emit(f"for (int64_t _i{k} = 0; _i{k} < til_arr_len({tmp}); _i{k}++) {{")
            self.indent += 1
            cv = self.declare_var(n.var, vt, n)
            self.emit(f"{self.ct(vt, n)} {cv} = {self.coerce(self.slot_get(e, f'{tmp}->data[_i{k}]'), e, vt, n)};")
            self.indent -= 1
        elif tt is STR:
            self.emit(f"const char* _s{k} = {src};")
            self.emit(f"(void)til_str_len(_s{k});")
            self.emit(f"for (int64_t _i{k} = 0; _s{k}[_i{k}]; _i{k}++) {{")
            self.indent += 1
            cv = self.declare_var(n.var, vt, n)
            self.emit(f"int64_t {cv} = (int64_t)(unsigned char)_s{k}[_i{k}];")
            self.indent -= 1
        else:
            self.unsupported(f"iterating over a value of type {show(tt)}", n)
        self.loop_body(n.body)
        self.emit("}")
        self.indent -= 1
        self.emit("}")
        self.pop_scope()

    def g_Return(self, n):
        if self._recover_stack:
            self.emit(f"til_recover_top = {self._recover_stack[0][0]}.prev;")
        if n.value is None or self.inst.key == 'main':
            if n.value is not None:
                self.emit(f"(void)({self.x(n.value)});")
            if self.cur_ret is NIL or self.inst.key == 'main':
                self.emit("return;")
            else:
                self.emit(f"return {self.zero(self.cur_ret)};")
            return
        val = self.coerce(self.x(n.value), self.T(n.value), self.cur_ret, n)
        if self.cur_ret is NIL:
            self.emit(f"(void)({val});")
            self.emit("return;")
            return
        if self.ensures:
            self.emit(f"{{ {self.ct(self.cur_ret)} _result = {val};")
            self.push_scope()
            self.scopes[-1]['result'] = ('_result', self.cur_ret)
            for k, cond in enumerate(self.ensures):
                text = self.ensures_text[k] if k < len(self.ensures_text) else ""
                msg = f"contract violation: {self.fname} ensures {text}".strip()
                self.emit(f"  if (!{self.truthy(cond)}) til_panic({til.c_string_literal(msg)});")
            self.pop_scope()
            self.emit("  return _result; }")
        else:
            self.emit(f"return {val};")

    def g_Break(self, n):
        self._leave_guarded_regions_in_loop()
        self.emit("break;")

    def g_Continue(self, n):
        self._leave_guarded_regions_in_loop()
        self.emit("continue;")

    def g_MatchExpr(self, n):
        self.match_stmt(n)

    def g_AssertStmt(self, n, kind="Assertion failed"):
        msg = f"{kind}: {n.message}" if n.message else kind
        self.emit(f"if (!{self.truthy(n.condition)}) til_panic({til.c_string_literal(msg)});")

    def g_InvariantStmt(self, n):
        self.g_AssertStmt(n, "Invariant violated")

    def g_IntentBlock(self, n, value_type=None):
        """Errors inside carry the intent: "intent '<goal>' failed: <error>"."""
        k = self.fresh_id()
        rec = f"_irec{k}"
        self.emit(til.c_comment(f"INTENT: {n.description}"))
        self.emit("{")
        self.indent += 1
        self.emit(f"TIL_Recover {rec};")
        self.emit(f"{rec}.prev = til_recover_top; {rec}.deadline = 0; til_recover_top = &{rec};")
        self.emit(f"if (setjmp({rec}.env) == 0) {{")
        self.indent += 1
        self._recover_stack.append((rec, self._loop_depth))
        if isinstance(n.body, til.Block):
            self.stmts(n.body)
        else:
            code = self.x(n.body, statement=True)
            if code:
                self.emit(f"{code};")
        self._recover_stack.pop()
        self.emit(f"til_recover_top = {rec}.prev;")
        self.indent -= 1
        self.emit("} else {")
        self.indent += 1
        self.emit(f"til_recover_top = {rec}.prev;")
        prefix = c_string_opaque(f"intent '{n.description}' failed: ")
        self.emit(f"til_panic(til_str_concat({prefix}, til_panic_msg));")
        self.indent -= 1
        self.emit("}")
        self.indent -= 1
        self.emit("}")

    # ── expression helpers ──────────────────────────────────────────────────
    PURE_NATIVES = {'len', 'str', 'int', 'float', 'bool', 'abs', 'sqrt', 'pow', 'sin', 'cos', 'tan',
                    'log', 'exp', 'floor', 'ceil', 'round', 'min', 'max', 'chr', 'ord', 'range',
                    'type_of', 'format', 'sum', 'str_bytes'}
    PURE_METHODS = {'len', 'contains', 'starts_with', 'ends_with', 'trim', 'to_upper', 'to_lower',
                    'replace', 'slice', 'find', 'index_of', 'count', 'eq', 'split', 'repeat',
                    'is_empty', 'get', 'join', 'first', 'last', 'has', 'keys', 'values', 'get_or',
                    'is_some', 'is_none', 'is_ok', 'is_err', 'unwrap_or', 'error', 'to_string', 'copy'}

    def capture(self, fn) -> List[str]:
        saved_out, saved_indent = self.output, self.indent
        self.output, self.indent = [], 1
        try:
            fn()
            return self.output
        finally:
            self.output, self.indent = saved_out, saved_indent

    @staticmethod
    def sx(lines: List[str], value: Optional[str]) -> str:
        """A GNU C statement expression: statements, then a value."""
        inner = "\n".join(lines)
        tail = f" {value};" if value else ""
        return f"({{\n{inner}{tail} }})"

    def effectful(self, node) -> bool:
        for n in walk(node):
            if isinstance(n, til.Call):
                f = n.func
                if isinstance(f, til.Identifier) and self.lookup_var(f.name) is None and (
                        f.name in self.PURE_NATIVES or f.name in self.ti.structs or f.name in ('Some', 'Ok', 'Err')):
                    continue
                if isinstance(f, til.Attribute) and f.attr in self.PURE_METHODS:
                    rt = self.T(f.obj)
                    if isinstance(rt, TCon) and not is_struct(rt):
                        continue
                return True
            if isinstance(n, (til.RetryExpr, til.TimeoutExpr, til.PerformEffect, til.IntentBlock,
                              til.NullCheck, til.Assignment)):
                return True
        return False

    def seq(self, nodes, gen):
        """Generate operands left to right like the VM: operands up to the last
        one with side effects are evaluated into temporaries first (C leaves the
        order of operand evaluation unspecified)."""
        last = -1
        for i, nd in enumerate(nodes):
            if nd is not None and self.effectful(nd):
                last = i
        prelude, codes = [], []
        for i, nd in enumerate(nodes):
            code, t = gen(i, nd)
            if i <= last and last >= 0 and not (isinstance(nd, (til.IntLit, til.FloatLit, til.StringLit,
                                                                  til.BoolLit, til.CharLit))):
                tmp = f"_e{self.fresh_id()}"
                prelude.append(f"{self.ct(t, nd)} {tmp} = {code};")
                codes.append(tmp)
            else:
                codes.append(code)
        return prelude, codes

    def wrap(self, prelude, value: str) -> str:
        return self.sx(prelude, value) if prelude else value

    def int_code(self, node) -> str:
        t = self.T(node)
        if t not in (INT, UNK, BOOL):
            self.unsupported(f"an index or count of type {show(t)}", node)
        return self.x(node)

    def vm_type_name(self, t) -> str:
        if is_struct(t):
            return nominal_name(t)
        return {'nil': 'nil', 'fn': 'fn'}.get(t.name, t.name)

    # ── expressions ─────────────────────────────────────────────────────────
    def x(self, n, statement=False) -> str:
        if isinstance(n, til.Call):
            return self.e_Call(n, statement)
        m = getattr(self, 'e_' + type(n).__name__, None)
        if m is None:
            self.unsupported(f"{type(n).__name__} expressions", n)
        return m(n)

    def e_IntLit(self, n):
        v = til.wrap_int64(n.value)
        if v == -(1 << 63):
            return "(-9223372036854775807LL - 1)"
        return str(v) if -(1 << 31) < v < (1 << 31) else f"{v}LL"

    def e_FloatLit(self, n):
        r = repr(float(n.value))
        if r in ('inf', '-inf', 'nan'):
            return {'inf': 'INFINITY', '-inf': '(-INFINITY)', 'nan': 'NAN'}[r]
        return r

    def e_StringLit(self, n):
        if '\0' in n.value:
            self.unsupported("a string containing a NUL byte", n)
        return til.c_string_literal(n.value)

    def e_BoolLit(self, n):
        return "true" if n.value else "false"

    def e_CharLit(self, n):
        ch = n.value
        if len(ch) == 1 and 32 <= ord(ch) < 127 and ch not in "'\\":
            return f"((int64_t)'{ch}')"
        return f"((int64_t){ord(ch) if ch else 0})"

    def e_DurationLit(self, n):
        mult = {"ms": 1, "s": 1000, "m": 60000, "h": 3600000}.get(n.unit, 1)
        return str(int(n.value * mult))

    def e_FStringLit(self, n):
        parts = [p for p in n.parts]
        exprs = [p[1] if p[0] != 'str' else None for p in parts]

        def gen(i, node):
            part = parts[i]
            if part[0] == 'str':
                return self.e_StringLit(part[1]), STR
            t = self.T(node)
            code = self.x(node)
            if len(part) > 2:
                return self.fmt_spec(code, t, part[2], node), STR
            return self.to_s(code, t, node), STR
        prelude, codes = self.seq(exprs, gen)
        if not codes:
            return '""'
        result = codes[0]
        for c in codes[1:]:
            result = f"til_str_concat({result}, {c})"
        return self.wrap(prelude, result)

    def fmt_spec(self, code: str, t, spec: str, node) -> str:
        k = self.fresh_id()
        lit = til.c_string_literal(spec)
        if t is INT or t is UNK:
            return self.sx([f"int64_t _f{k} = {code};"],
                           f"til_fmt_spec(0, _f{k}, 0.0, til_int_to_str(_f{k}), {lit})")
        if t is FLOAT:
            return self.sx([f"double _f{k} = {code};"],
                           f"til_fmt_spec(1, 0, _f{k}, til_float_str(_f{k}), {lit})")
        return f"til_fmt_spec(2, 0, 0.0, {self.to_s(code, t, node)}, {lit})"

    def e_Identifier(self, n):
        name = n.name
        if name == 'NULL':
            t = self.T(n)
            if t.name == 'option':
                return f"{self.option_ctor(t, n)}()"
            return "NULL"
        found = self.lookup_var(name)
        if found is not None:
            if isinstance(found[0], dict):
                self.unsupported(f"the lambda '{name}' used as a value", n)
            return found[0]
        if name in self.ti.globals:
            return c_global(name)
        if name in self.ti.funcs:
            self.unsupported(f"the function '{name}' used as a value", n)
        self.unsupported(f"unknown name '{name}'", n)

    def folded(self, n) -> Optional[str]:
        """The bytecode compiler folds constant operator trees with const_eval
        (Python semantics: `true == 1` is true there); use the same value."""
        if not hasattr(self, '_inline_consts'):
            env = {}
            for name, g in self.ti.globals.items():
                d = g.decl
                if not d.mutable and isinstance(d.value, (til.IntLit, til.FloatLit, til.StringLit,
                                                          til.BoolLit, til.CharLit)):
                    env[name] = ord(d.value.value) if isinstance(d.value, til.CharLit) else d.value.value
            self._inline_consts = env
        if not self.fold_differs(n):
            return None             # run-time semantics give the same value: generate normally
        try:
            v = til.const_eval(n, self._inline_consts)
        except (til.NotConstant, OverflowError, ValueError, ZeroDivisionError, TypeError):
            return None
        want = BOOL if isinstance(v, bool) else INT if isinstance(v, int) else \
            FLOAT if isinstance(v, float) else STR if isinstance(v, str) else None
        if want is None:
            return None
        if self.T(n) is not want:
            # e.g. a local that shadows a global constant (the VM still folds with the constant)
            self.unsupported("an expression the bytecode compiler folds with a different type", n)
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return self.e_IntLit(til.IntLit(value=v))
        if isinstance(v, float):
            return self.e_FloatLit(til.FloatLit(value=v))
        if '\0' in v:
            self.unsupported("a string containing a NUL byte", n)
        return til.c_string_literal(v)

    def fold_differs(self, n) -> bool:
        """Does const_eval disagree with the run-time rules somewhere in n?
        (bool == number is true when folded, false at run time; a local that
        shadows an inlined constant is still folded with the constant)."""
        env = self._inline_consts
        for x in walk(n):
            if isinstance(x, til.Identifier) and x.name in env and self.lookup_var(x.name) is not None:
                return True
            if isinstance(x, til.BinaryOp) and x.op in ('==', '!='):
                try:
                    a, b = til.const_eval(x.left, env), til.const_eval(x.right, env)
                except (til.NotConstant, OverflowError, ValueError, ZeroDivisionError, TypeError):
                    continue
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
                        isinstance(a, bool) != isinstance(b, bool):
                    return True
        return False

    def e_BinaryOp(self, n):
        op = n.op
        f = self.folded(n)
        if f is not None:
            return f
        if op in ('and', 'or'):
            return f"({self.truthy(n.left)} {'&&' if op == 'and' else '||'} {self.truthy(n.right)})"
        lt, rt = self.T(n.left), self.T(n.right)
        prelude, (a, b) = self.seq([n.left, n.right], lambda i, nd: (self.x(nd), self.T(nd)))
        if op in ('==', '!='):
            eq = self.eq_code(a, lt, b, rt, n)
            return self.wrap(prelude, eq if op == '==' else f"(!{eq})")
        if op in CMP_OPS:
            num = ('int', 'float', 'unk')
            if lt.name in num and rt.name in num:
                mixed = self.mixed_cmp(a, lt, b, rt, op)
                return self.wrap(prelude, mixed if mixed else f"({a} {op} {b})")
            if lt is STR and rt is STR:
                return self.wrap(prelude, f"(til_str_cmp({a}, {b}) {op} 0)")
            self.unsupported(f"comparing {show(lt)} with {show(rt)}", n)
        if op in ARITH_OPS:
            return self.wrap(prelude, self.arith(op, a, lt, b, rt, self.T(n), n)[0])
        if op in BIT_OPS:
            if lt.name not in ('int', 'unk') or rt.name not in ('int', 'unk'):
                self.unsupported(f"bitwise {op} on {show(lt)} and {show(rt)}", n)
            if op == '<<':
                return self.wrap(prelude, f"til_shl({a}, {b})")
            if op == '>>':
                return self.wrap(prelude, f"til_shr({a}, {b})")
            return self.wrap(prelude, f"({a} {op} {b})")
        self.unsupported(f"operator {op}", n)

    def arith(self, op, a, ta, b, tb, rt, node):
        num = ('int', 'float', 'unk')
        if ta.name in num and tb.name in num:
            if ta is not FLOAT and tb is not FLOAT:
                if op in ('+', '-', '*'):
                    return f"({a} {op} {b})", INT
                return {'/': f"til_div_i({a}, {b})", '%': f"til_mod_i({a}, {b})",
                        '**': f"til_ipow({a}, {b})"}[op], INT
            if op in ('+', '-', '*'):
                return f"({a} {op} {b})", FLOAT
            if op == '/':
                return f"((double)({a}) / (double)({b}))", FLOAT
            if op == '%':
                return f"til_fmod({a}, {b})", FLOAT
            return f"til_fpow({a}, {b})", FLOAT
        if op == '+' and (ta is STR or tb is STR):
            return f"til_str_concat({self.to_s(a, ta, node)}, {self.to_s(b, tb, node)})", STR
        if op == '+' and ta.name == 'array' and tb.name == 'array':
            return f"til_arr_concat({a}, {b})", ta
        if op == '*' and ta is STR and tb.name in ('int', 'unk'):
            return f"til_str_rep({a}, {b})", STR
        if op == '*' and ta.name == 'array' and tb.name in ('int', 'unk'):
            return f"til_arr_repeat({a}, {b})", ta
        self.unsupported(f"{show(ta)} {op} {show(tb)}", node)

    def e_UnaryOp(self, n):
        f = self.folded(n) if n.op in ('-', 'not', '~') else None
        if f is not None:
            return f
        t = self.T(n.operand)
        c = self.x(n.operand)
        if n.op == 'not':
            return f"(!{self.truthy_code(c, t)})"
        if n.op == '-':
            if t in (INT, UNK, FLOAT):
                return f"(-{c})"
            self.unsupported(f"negating a value of type {show(t)}", n)
        if n.op == '~':
            if t in (INT, UNK):
                return f"(~{c})"
            self.unsupported(f"~ on a value of type {show(t)}", n)
        return c

    def e_Index(self, n):
        ot = self.T(n.obj)
        prelude, (obj, idx) = self.seq([n.obj, n.index], lambda i, nd: (self.x(nd), self.T(nd)))
        if ot.name == 'array':
            e = ot.args[0]
            self.check_slot_type(e, n)
            if self.T(n.index) not in (INT, UNK, BOOL):
                self.unsupported("an array index that is not an integer", n)
            if self.inst.info is not None and self.inst.info.fdef.level < 2:
                return self.wrap(prelude, self.slot_get(e, f"({obj})->data[{idx}]"))
            return self.wrap(prelude, self.slot_get(e, f"*til_arr_at({obj}, {idx})"))
        if ot is STR:
            if self.T(n.index) not in (INT, UNK, BOOL):
                self.unsupported("a string index that is not an integer", n)
            return self.wrap(prelude, f"til_str_at({obj}, {idx})")
        if ot.name == 'map':
            k, v = ot.args
            self.check_slot_type(k, n, "a map key")
            self.check_slot_type(v, n, "a map value")
            key = self.slot_make(k, self.coerce(idx, self.T(n.index), k, n))
            return self.wrap(prelude, self.slot_get(v, f"til_map_get({obj}, {key}, {self.key_kind(k, n)})"))
        self.unsupported(f"indexing a value of type {show(ot)}", n)

    def e_Attribute(self, n):
        obj = n.obj
        if isinstance(obj, til.Identifier) and self.lookup_var(obj.name) is None:
            name = obj.name
            if name in self.ti.enums:
                return f"((int64_t){self.mangle_name(name)}_{self.mangle_name(n.attr)})"
            if name in self.ti.adts:
                payload = self.ti.adts[name].frozen.get(n.attr)
                if payload != []:
                    self.unsupported(f"'{name}.{n.attr}' without its values", n)
                return f"{self.adt_ctor(name, n.attr, n)}()"
            if name in self.ti.module_names and name not in self.ti.globals:
                return self.e_Identifier(til.Identifier(name=n.attr, line=n.line, col=n.col))
        ot = self.T(obj)
        if is_struct(ot):
            info = self.ti.structs[nominal_name(ot)]
            ft = info.frozen.get(n.attr)
            if ft is None:
                self.unsupported(f"struct '{info.name}' has no field '{n.attr}'", n)
            self.ct(ft, n, f"field '{n.attr}'")
            if isinstance(obj, til.Identifier) and obj.name == 'self' and self.lookup_var('self') is not None \
                    and self.lookup_var('self')[0] == 'self':
                return f"self->{n.attr}"
            sc = self.mangle_name(info.name)
            return f"(({sc}*)til_nn({self.x(obj)}, {til.c_string_literal(n.attr)}))->{n.attr}"
        if ot.name in ('option', 'result') and n.attr in ('value', 'error'):
            return f"({self.x(obj)}).{n.attr}"
        self.unsupported(f"field '{n.attr}' of a value of type {show(ot)}", n)

    def e_ArrayLit(self, n):
        t = self.T(n)
        if t.name != 'array':
            self.unsupported(f"an array literal of type {show(t)}", n)
        e = t.args[0]
        if not n.elements:
            return "til_arr_new(0)"
        self.check_slot_type(e, n)
        prelude, codes = self.seq(n.elements, lambda i, nd: (
            self.coerce(self.x(nd), self.T(nd), e, nd), e))
        items = ", ".join(self.slot_make(e, c) for c in codes)
        return self.wrap(prelude, f"til_arr_lit({len(codes)}, (TIL_Slot[]){{{items}}})")

    e_TupleLit = e_ArrayLit

    def e_DictLit(self, n):
        t = self.T(n)
        k, v = t.args
        if not n.pairs:
            return "til_map_new()"
        self.check_slot_type(k, n, "a map key")
        self.check_slot_type(v, n, "a map value")
        kind = self.key_kind(k, n)
        nodes = [x for pair in n.pairs for x in pair]
        prelude, codes = self.seq(nodes, lambda i, nd: (
            self.coerce(self.x(nd), self.T(nd), k if i % 2 == 0 else v, nd), k if i % 2 == 0 else v))
        items = ", ".join(self.slot_make(k if i % 2 == 0 else v, c) for i, c in enumerate(codes))
        return self.wrap(prelude, f"til_map_lit({len(n.pairs)}, {kind}, (TIL_Slot[]){{{items}}})")

    def e_Range(self, n):
        prelude, (a, b) = self.seq([n.start, n.end], lambda i, nd: (self.int_code(nd), INT))
        return self.wrap(prelude, f"til_range({a}, {b}, {'true' if n.inclusive else 'false'})")

    def e_StructInit(self, n):
        info = self.ti.structs.get(n.name)
        if info is None:
            self.unsupported(f"unknown struct '{n.name}'", n)
        fields = {f.name: f for f in info.sdef.fields}
        nodes, kinds = [], []
        for fname in info.order:
            if fname in n.fields:
                nodes.append(n.fields[fname])
                kinds.append('given')
            elif fields[fname].default is not None:
                nodes.append(fields[fname].default)
                kinds.append('default')
            else:
                nodes.append(None)
                kinds.append('zero')

        def gen(i, nd):
            ft = info.frozen[info.order[i]]
            self.ct(ft, n, f"field '{info.order[i]}'")
            if nd is None:
                return self.zero(ft), ft
            return self.coerce(self.x(nd), self.T(nd), ft, nd), ft
        prelude, codes = self.seq(nodes, gen)
        return self.wrap(prelude, f"{self.mangle_name(n.name)}_create({', '.join(codes)})")

    def e_IfExpr(self, n):
        t = self.T(n)
        self.ct(t, n, "the value of an if-expression")
        c = self.truthy(n.condition)
        a = self.coerce(self.x(n.then_expr), self.T(n.then_expr), t, n)
        b = (self.coerce(self.x(n.else_expr), self.T(n.else_expr), t, n) if n.else_expr is not None
             else self.coerce("NULL", NIL, t, n))
        return f"({c} ? {a} : {b})"

    def e_Cast(self, n):
        src = self.T(n.expr)
        code = self.x(n.expr)
        target = n.target_type
        if not isinstance(target, til.PrimitiveType):
            return code
        dst = self.T(n)
        if dst is FLOAT:
            if src in (INT, BOOL, UNK):
                return f"((double){code})"
            if src is FLOAT:
                return code
            if src is STR:
                return f"til_str_to_float({code})"
            if src is NIL:
                return "0.0"
        elif dst is INT:
            if src in (INT, UNK):
                return code
            if src is FLOAT:
                return f"til_f2i({code})"
            if src is BOOL:
                return f"((int64_t){code})"
            if src is STR:
                return f"til_str_to_int({code})"
            if src is NIL:
                return "0"
        elif dst is STR:
            return self.to_s(code, src, n)
        elif dst is BOOL:
            return self.truthy_code(code, src)
        self.unsupported(f"converting {show(src)} to {show(dst)}", n)

    def restore_recovery(self) -> str:
        return f"til_recover_top = {self._recover_stack[0][0]}.prev; " if self._recover_stack else ""

    def e_NullCheck(self, n):
        t = self.T(n.expr)
        code = self.x(n.expr)
        k = self.fresh_id()
        if t.name not in ('option', 'result'):
            return code
        c = self.ct(t, n)
        lines = [f"{c} _q{k} = {code};"]
        ret = self.cur_ret
        if t.name == 'option':
            if self.propagates:
                if ret.name != 'option':
                    self.unsupported("`?` on an Option inside a function returning Result", n)
                lines.append(f"if (!_q{k}.has_value) {{ {self.restore_recovery()}return "
                             f"{self.option_ctor(ret, n)}(); }}")
            else:
                lines.append(f"if (!_q{k}.has_value) til_panic(\"? applied to None\");")
            return self.sx(lines, f"_q{k}.value")
        if self.propagates:
            if ret.name != 'result' or self.ct(ret.args[1]) != self.ct(t.args[1]):
                self.unsupported("`?` on a Result whose error type differs from the function's", n)
            lines.append(f"if (!_q{k}.is_ok) {{ {self.restore_recovery()}return "
                         f"til_Err_{self.result_type(ret)}(_q{k}.error); }}")
        else:
            lines.append(f"if (!_q{k}.is_ok) til_panic(til_str_concat(\"? applied to Err: \", "
                         f"{self.to_s(f'_q{k}.error', t.args[1], n)}));")
        return self.sx(lines, f"_q{k}.value")

    def e_ListComprehension(self, n):
        t = self.T(n)
        e = t.args[0]
        self.check_slot_type(e, n)
        k = self.fresh_id()
        arr = f"_comp_{k}"

        def body():
            self.emit(f"TIL_Arr* {arr} = til_arr_new(0);")

            def inner():
                code = self.coerce(self.x(n.expr), self.T(n.expr), e, n)
                if n.condition is not None:
                    self.emit(f"if ({self.truthy(n.condition)}) til_arr_push({arr}, {self.slot_make(e, code)});")
                else:
                    self.emit(f"til_arr_push({arr}, {self.slot_make(e, code)});")
            self.emit_loop(n.var, n.iter, self.inst.decl(n), inner, n)
        return self.sx(self.capture(body), arr)

    def emit_loop(self, var, it, vt, inner, node):
        """for var in it: inner() -- used by comprehensions."""
        k = self.fresh_id()
        self.push_scope()
        self.emit("{")
        self.indent += 1
        if isinstance(it, til.Range):
            c = self.ct(vt, node)
            self.emit(f"int64_t _end{k} = {self.int_code(it.end)};")
            cv = self.declare_var(var, vt, node)
            self.emit(f"for ({c} {cv} = {self.coerce(self.x(it.start), self.T(it.start), vt, node)}; "
                      f"{cv} {'<=' if it.inclusive else '<'} _end{k}; {cv}++) {{")
        else:
            tt = self.T(it)
            src = self.x(it)
            if tt.name in ('array', 'map'):
                e = tt.args[0]
                self.check_slot_type(e, node)
                self.emit(f"TIL_Arr* _it{k} = {'til_map_keys(' + src + ')' if tt.name == 'map' else src};")
                self.emit(f"for (int64_t _i{k} = 0; _i{k} < til_arr_len(_it{k}); _i{k}++) {{")
                cv = self.declare_var(var, vt, node)
                self.emit(f"    {self.ct(vt, node)} {cv} = "
                          f"{self.coerce(self.slot_get(e, f'_it{k}->data[_i{k}]'), e, vt, node)};")
            elif tt is STR:
                self.emit(f"const char* _s{k} = {src};")
                self.emit(f"for (int64_t _i{k} = 0; _s{k}[_i{k}]; _i{k}++) {{")
                cv = self.declare_var(var, vt, node)
                self.emit(f"    int64_t {cv} = (int64_t)(unsigned char)_s{k}[_i{k}];")
            else:
                self.unsupported(f"iterating over a value of type {show(tt)}", node)
        self.indent += 1
        if self._uses_timeout:
            self.emit("til_check_deadline();")
        inner()
        self.indent -= 1
        self.emit("}")
        self.indent -= 1
        self.emit("}")
        self.pop_scope()

    def e_Lambda(self, n):
        self.unsupported("a lambda used as a value (closures as values run on the VM)", n)

    def e_Block(self, n):
        t = self.T(n)
        k = self.fresh_id()
        if t is NIL:
            return self.sx(self.capture(lambda: self.block_into(n, None, None)), "NULL")
        c = self.ct(t, n, "the value of a block")
        return self.sx(self.capture(lambda: (self.emit(f"{c} _bv{k} = {self.none_value(t)};"),
                                             self.block_into(n, f"_bv{k}", t))), f"_bv{k}")

    def e_IntentBlock(self, n):
        t = self.T(n)
        k = self.fresh_id()
        if t is NIL:
            return self.sx(self.capture(lambda: self.g_IntentBlock(n)), "NULL")
        c = self.ct(t, n, "the value of an intent block")

        def body():
            self.emit(f"{c} volatile _iv{k} = {self.none_value(t)};")
            self.g_IntentBlock_value(n, f"_iv{k}", t)
        return self.sx(self.capture(body), f"_iv{k}")

    def g_IntentBlock_value(self, n, var, t):
        k = self.fresh_id()
        rec = f"_irec{k}"
        self.emit(til.c_comment(f"INTENT: {n.description}"))
        self.emit("{")
        self.indent += 1
        self.emit(f"TIL_Recover {rec};")
        self.emit(f"{rec}.prev = til_recover_top; {rec}.deadline = 0; til_recover_top = &{rec};")
        self.emit(f"if (setjmp({rec}.env) == 0) {{")
        self.indent += 1
        self._recover_stack.append((rec, self._loop_depth))
        self.block_into(n.body, var, t)
        self._recover_stack.pop()
        self.emit(f"til_recover_top = {rec}.prev;")
        self.indent -= 1
        self.emit("} else {")
        self.indent += 1
        self.emit(f"til_recover_top = {rec}.prev;")
        prefix = c_string_opaque(f"intent '{n.description}' failed: ")
        self.emit(f"til_panic(til_str_concat({prefix}, til_panic_msg));")
        self.indent -= 1
        self.emit("}")
        self.indent -= 1
        self.emit("}")

    def e_EffectHandler(self, n):
        self.unsupported("`handle` (effect handlers run on the VM)", n)

    def e_PerformEffect(self, n):
        return self.call_user(n, n.args)

    def none_value(self, t) -> str:
        if t.name == 'option':
            return f"{self.option_ctor(t, None)}()"
        if is_ptr(t) or t is NIL:
            return "NULL"
        return self.zero(t)

    def e_RetryExpr(self, n):
        return self.guarded(n, want_value=True)

    e_TimeoutExpr = e_RetryExpr

    def e_MatchExpr(self, n):
        t = self.T(n)
        k = self.fresh_id()
        if t is NIL:
            return self.sx(self.capture(lambda: self.match_into(n, None, None)), "NULL")
        c = self.ct(t, n, "the value of a match")

        def body():
            self.emit(f"{c} _mv{k} = {self.none_value(t)};")
            self.match_into(n, f"_mv{k}", t)
        return self.sx(self.capture(body), f"_mv{k}")

    # ── blocks with values ──────────────────────────────────────────────────
    def block_into(self, body, var, t):
        """Emit a block; its value (last expression) is stored in var (if var)."""
        if not isinstance(body, til.Block):
            if var is None:
                code = self.x(body, statement=True)
                if code:
                    self.emit(f"{code};")
            else:
                self.emit(f"{var} = {self.coerce(self.x(body), self.T(body), t, body)};")
            return
        self.push_scope()
        stmts = body.statements
        for st in stmts[:-1]:
            self.stmt(st)
        if stmts:
            last = stmts[-1]
            if var is not None and isinstance(last, til.ExprStmt):
                self.emit(f"{var} = {self.coerce(self.x(last.expr), self.T(last.expr), t, last)};")
            elif var is not None and isinstance(last, til.MatchExpr):
                self.match_into(last, var, t)
            elif var is not None and isinstance(last, til.If) and last.else_body is not None:
                first = True
                for cond, b in [(last.condition, last.then_body)] + list(last.elifs):
                    self.emit(f"{'if' if first else '} else if'} ({self.truthy(cond)}) {{")
                    first = False
                    self.indent += 1
                    self.block_into(b, var, t)
                    self.indent -= 1
                self.emit("} else {")
                self.indent += 1
                self.block_into(last.else_body, var, t)
                self.indent -= 1
                self.emit("}")
            elif var is not None and isinstance(last, (til.Block, til.IntentBlock)):
                self.emit(f"{var} = {self.coerce(self.x(last), self.T(last), t, last)};")
            else:
                self.stmt(last)
        self.pop_scope()

    # ── retry / timeout ─────────────────────────────────────────────────────
    def guarded(self, n, want_value: bool) -> str:
        """retry(n) / timeout(ms) with recovery points (setjmp/longjmp), as in
        the classic backend; the error message is bound to `error`."""
        k = self.fresh_id()
        is_retry = isinstance(n, til.RetryExpr)
        t = self.T(n) if want_value else None
        if t is NIL:
            want_value, t = False, None
        var = f"_gv{k}" if want_value else None
        rec = f"_rec{k}"

        def body():
            if want_value:
                self.emit(f"{self.ct(t, n, 'the value of retry/timeout')} volatile {var} = {self.none_value(t)};")
            self.emit("{")
            self.indent += 1
            if is_retry:
                self.emit(f"volatile int64_t _lim{k} = {self.int_code(n.count)};")
                self.emit(f"volatile int64_t _cnt{k} = 0;")
            self.emit(f"TIL_Recover {rec};")
            if is_retry:
                self.emit(f"_again{k}:")
                self.emit(f"{rec}.prev = til_recover_top; {rec}.deadline = 0; til_recover_top = &{rec};")
            else:
                dur = self.x(n.duration)
                self.emit(f"{rec}.prev = til_recover_top; {rec}.deadline = til_now_ms() + (double)({dur}); "
                          f"til_recover_top = &{rec};")
            self.emit(f"if (setjmp({rec}.env) == 0) {{")
            self.indent += 1
            self._recover_stack.append((rec, self._loop_depth))
            self.block_into(n.body, var, t)
            self._recover_stack.pop()
            self.emit(f"til_recover_top = {rec}.prev;")
            self.indent -= 1
            self.emit("} else {")
            self.indent += 1
            self.emit(f"til_recover_top = {rec}.prev;")
            if is_retry:
                self.emit(f"if (++_cnt{k} < _lim{k}) goto _again{k};")
            if n.fallback is None:
                self.emit("til_panic(til_panic_msg);")
            else:
                self.push_scope()
                ev = self.declare_var('error', STR, n)
                self.emit(f"const char* {ev} = til_strdup_n(til_panic_msg, strlen(til_panic_msg));")
                self.emit(f"(void){ev};")
                self.block_into(n.fallback, var, t)
                self.pop_scope()
            self.indent -= 1
            self.emit("}")
            self.indent -= 1
            self.emit("}")
        return self.sx(self.capture(body), var)

    # ── pattern matching ────────────────────────────────────────────────────
    def match_stmt(self, n):
        self.match_into(n, None, None)

    def switchable(self, n, arms, subj_t) -> bool:
        if subj_t not in (INT, UNK):
            return False
        seen = set()
        for pat, guard, body in arms:
            if guard is not None:
                return False
            if isinstance(pat, til.Identifier) and pat.name == '_':
                continue
            if isinstance(pat, til.IntLit):
                v = pat.value
            elif isinstance(pat, til.UnaryOp) and pat.op == '-' and isinstance(pat.operand, til.IntLit):
                v = -pat.operand.value
            else:
                return False
            if v in seen:
                return False
            seen.add(v)
        bodies = [b for _p, _g, b in arms]
        return not any(isinstance(x, (til.Break, til.Continue)) for b in bodies for x in walk(b))

    def match_into(self, n, var, t):
        subj_t = self.T(n.value)
        arms = [arm if len(arm) == 3 else (arm[0], None, arm[1]) for arm in n.arms]
        k = self.fresh_id()
        subj = self.x(n.value)
        if var is None and self.switchable(n, arms, subj_t):
            self.emit(f"switch ({subj}) {{")
            self.indent += 1
            for pat, _guard, body in arms:
                if isinstance(pat, til.Identifier):
                    self.emit("default: {")
                else:
                    self.emit(f"case {self.x(pat)}: {{")
                self.indent += 1
                self.block_into(body, None, None)
                self.emit("break;")
                self.indent -= 1
                self.emit("}")
            self.indent -= 1
            self.emit("}")
            return
        self.emit("{")
        self.indent += 1
        sc = self.ct(subj_t, n, "the value matched")
        self.emit(f"{sc} _ms{k} = {subj};")
        for pat, guard, body in arms:
            self.push_scope()
            tests, binds = [], []
            self.pattern(pat, f"_ms{k}", subj_t, tests, binds)
            self.emit(f"if ({' && '.join(tests) if tests else 'true'}) {{")
            self.indent += 1
            for b in binds:
                self.emit(b)
            if guard is not None:
                self.emit(f"if ({self.truthy(guard)}) {{")
                self.indent += 1
            self.block_into(body, var, t)
            self.emit(f"goto _mend{k};")
            if guard is not None:
                self.indent -= 1
                self.emit("}")
            self.indent -= 1
            self.emit("}")
            self.pop_scope()
        if var is not None and self.inst.is_open_match(n) and not is_ptr(t) and t.name != 'option' \
                and t is not NIL:
            self.emit("til_panic(\"match: no arm matched, and the native backend cannot represent "
                      "the resulting None value\");")
        self.indent -= 1
        self.emit(f"_mend{k}: ;")
        self.emit("}")

    def adt_pattern(self, name, variant, args, subj, st, tests, binds, pat):
        info = self.ti.adts[name]
        if not (is_adt(st) and nominal_name(st) == name) or variant not in info.variants:
            tests.append("false")
            return
        tag = list(info.variants).index(variant)
        tests.append(f"({subj}) != NULL && ({subj})->tag == {tag}")
        payload = info.frozen[variant]
        for i, (p, pt) in enumerate(zip(args, payload)):
            self.pattern(p, self.slot_get(pt, f"({subj})->f[{i}]"), pt, tests, binds)

    def pattern(self, pat, subj: str, st, tests: List[str], binds: List[str]):
        if isinstance(pat, til.Identifier):
            name = pat.name
            if name == '_':
                return
            if name == 'NULL':
                tests.append(self.eq_code(subj, st, "NULL", NIL, pat))
                return
            if self.lookup_var(name) is None and name in self.ti.globals:
                tests.append(self.eq_code(subj, st, c_global(name), self.ti.globals[name].frozen, pat))
                return
            bt = self.T(pat)
            cname = self.declare_var(name, bt, pat)
            binds.append(f"{self.ct(bt, pat)} {cname} = {self.coerce(subj, st, bt, pat)}; (void){cname};")
            return
        if isinstance(pat, til.Call):
            f = pat.func
            if isinstance(f, til.Identifier) and f.name in ('Some', 'Ok', 'Err'):
                if f.name == 'Some':
                    if st.name != 'option':
                        tests.append("false")
                        return
                    tests.append(f"({subj}).has_value")
                    inner, it = f"({subj}).value", st.args[0]
                else:
                    if st.name != 'result':
                        tests.append("false")
                        return
                    tests.append(f"({subj}).is_ok" if f.name == 'Ok' else f"!({subj}).is_ok")
                    inner, it = (f"({subj}).value", st.args[0]) if f.name == 'Ok' else (f"({subj}).error", st.args[1])
                for p in pat.args[:1]:
                    self.pattern(p, inner, it, tests, binds)
                return
            if isinstance(f, til.Identifier) and f.name in self.ti.structs and self.lookup_var(f.name) is None:
                info = self.ti.structs[f.name]
                if not (is_struct(st) and nominal_name(st) == f.name):
                    tests.append("false")
                    return
                tests.append(f"({subj}) != NULL")
                for p, fname in zip(pat.args, info.order):
                    self.pattern(p, f"({subj})->{fname}", info.frozen[fname], tests, binds)
                return
            if isinstance(f, til.Attribute) and isinstance(f.obj, til.Identifier) and f.obj.name in self.ti.adts:
                self.adt_pattern(f.obj.name, f.attr, pat.args, subj, st, tests, binds, pat)
                return
        if isinstance(pat, til.Attribute) and isinstance(pat.obj, til.Identifier) and pat.obj.name in self.ti.adts:
            self.adt_pattern(pat.obj.name, pat.attr, [], subj, st, tests, binds, pat)
            return
        if isinstance(pat, (til.TupleLit, til.ArrayLit)):
            if st.name != 'array':
                tests.append("false")
                return
            e = st.args[0]
            self.check_slot_type(e, pat)
            tests.append(f"({subj}) != NULL && ({subj})->len == {len(pat.elements)}")
            for i, p in enumerate(pat.elements):
                self.pattern(p, self.slot_get(e, f"({subj})->data[{i}]"), e, tests, binds)
            return
        if isinstance(pat, til.StructInit) and pat.name in self.ti.structs:
            info = self.ti.structs[pat.name]
            if not (is_struct(st) and nominal_name(st) == pat.name):
                tests.append("false")
                return
            tests.append(f"({subj}) != NULL")
            for fname, p in pat.fields.items():
                ft = info.frozen.get(fname)
                if ft is None:
                    self.unsupported(f"struct '{pat.name}' has no field '{fname}'", pat)
                self.pattern(p, f"({subj})->{fname}", ft, tests, binds)
            return
        if isinstance(pat, til.Range):
            if st not in (INT, FLOAT, UNK):
                tests.append("false")
                return
            tests.append(f"({subj} >= {self.x(pat.start)} && {subj} {'<=' if pat.inclusive else '<'} "
                         f"{self.x(pat.end)})")
            return
        if isinstance(pat, til.BinaryOp) and pat.op == '|':
            alts, stack = [], [pat]
            while stack:
                p = stack.pop()
                if isinstance(p, til.BinaryOp) and p.op == '|':
                    stack.extend([p.right, p.left])
                else:
                    alts.append(p)
            tests.append("(" + " || ".join(self.eq_code(subj, st, self.x(a), self.T(a), a) for a in alts) + ")")
            return
        tests.append(self.eq_code(subj, st, self.x(pat), self.T(pat), pat))

    # ── calls ───────────────────────────────────────────────────────────────
    def e_Call(self, n, statement=False):
        f = n.func
        ti = self.ti
        if isinstance(f, til.Identifier):
            name = f.name
            found = self.lookup_var(name)
            if found is not None:
                if isinstance(found[0], dict):
                    return self.call_lambda(found[0], n.args, n)
                self.unsupported(f"calling the value '{name}' (closures as values run on the VM)", n)
            if name in ('print', 'println'):
                return self.sx(self.capture(lambda: self.print_call(n)), None if statement else "NULL")
            if name in ti.funcs and '.' not in name:
                return self.call_user(n, n.args)
            if name in ti.structs:
                info = ti.structs[name]
                if len(n.args) != len(info.order):
                    self.unsupported(f"'{name}(...)' with a wrong number of fields", n)
                prelude, codes = self.seq(n.args, lambda i, nd: (
                    self.coerce(self.x(nd), self.T(nd), info.frozen[info.order[i]], nd),
                    info.frozen[info.order[i]]))
                return self.wrap(prelude, f"{self.mangle_name(name)}_create({', '.join(codes)})")
            if name in ('Some', 'Ok', 'Err'):
                t = self.T(n)
                arg = n.args[0] if n.args else None
                if t.name == 'option':
                    inner = t.args[0]
                    code = self.coerce(self.x(arg), self.T(arg), inner, n) if arg is not None else self.zero(inner)
                    return f"{self.option_ctor(t, n, some=True)}({code})"
                if t.name == 'result':
                    rname = self.result_type(t, n)
                    slot = t.args[0] if name == 'Ok' else t.args[1]
                    if slot is NIL:
                        code = "0"
                    else:
                        code = self.coerce(self.x(arg), self.T(arg), slot, n) if arg is not None else self.zero(slot)
                    return f"til_{name}_{rname}({code})"
                self.unsupported(f"{name}(...) without a known Option/Result type", n)
            if name in ti.globals:
                self.unsupported(f"calling the global value '{name}'", n)
            if name in til.NATIVE_ID:
                return self.native(name, n, statement)
            self.unsupported(f"call of unknown function '{name}'", n)
        if isinstance(f, til.Attribute):
            obj = f.obj
            if isinstance(obj, til.Identifier) and self.lookup_var(obj.name) is None:
                oname = obj.name
                if oname in ti.structs:
                    return self.call_user(n, n.args)
                if oname in ti.adts:
                    info = ti.adts[oname]
                    payload = info.frozen.get(f.attr)
                    if payload is None or len(payload) != len(n.args):
                        self.unsupported(f"'{oname}.{f.attr}(...)'", n)
                    ctor = self.adt_ctor(oname, f.attr, n)
                    prelude, codes = self.seq(n.args, lambda i, nd: (
                        self.coerce(self.x(nd), self.T(nd), payload[i], nd), payload[i]))
                    return self.wrap(prelude, f"{ctor}({', '.join(codes)})")
                if oname in ti.module_names and oname not in ti.globals:
                    return self.call_user(n, n.args)
                if oname.startswith('__Vec__') and f.attr == 'new':
                    return "til_arr_new(0)"
                if oname.startswith('__HashMap__') and f.attr == 'new':
                    return "til_map_new()"
            return self.method_call(n, statement)
        self.unsupported("calling the result of an expression", n)

    def call_user(self, n, args):
        target = self.inst.call(n)
        if target is None:
            self.unsupported("a call whose target is not known statically", n)
        key, params = target
        ikey = (key, tuple(canon(p) for p in params))
        cname = self.inst_names.get(ikey)
        callee = self.insts.get(ikey)
        if cname is None or callee is None:
            self.unsupported(f"no instance of '{key}' for these argument types", n)
        return self.call_instance(callee, cname, None, None, args, n)

    def call_instance(self, callee: Instance, cname: str, recv_node, recv_code, args, n):
        fdef = callee.info.fdef
        params = list(fdef.params)
        ptypes = list(callee.params)
        nodes: List[Any] = []
        kinds = []
        if recv_node is not None:
            params, ptypes = params[1:], ptypes[1:]
        for i, p in enumerate(params):
            if i < len(args):
                nodes.append(args[i])
                kinds.append('arg')
            elif p.default is not None:
                nodes.append(p.default)
                kinds.append('default')
            else:
                self.unsupported(f"missing argument '{p.name}' for '{callee.key}'", n)
        if len(args) > len(params):
            self.unsupported(f"too many arguments for '{callee.key}'", n)

        def gen(i, nd):
            if kinds[i] == 'default':
                if not literal_default(nd):
                    self.unsupported(f"the default value of '{params[i].name}' (only literals are "
                                     "evaluated by the caller natively)", n)
                saved = self.inst
                self.inst = callee
                try:
                    code, t = self.x(nd), self.T(nd)
                finally:
                    self.inst = saved
            else:
                code, t = self.x(nd), self.T(nd)
            return self.coerce(code, t, ptypes[i], nd), ptypes[i]
        prelude, codes = self.seq(nodes, gen)
        if recv_code is not None:
            codes = [recv_code] + codes
        return self.wrap(prelude, f"{cname}({', '.join(codes)})")

    # ── lambdas bound to names: static C functions, captures passed at the call ──
    def bind_lambda(self, name: str, lam):
        ft = self.T(lam)
        if ft.name != 'fn':
            self.unsupported(f"the lambda '{name}'", lam)
        ret, ptypes = ft.args[0], list(ft.args[1:])
        k = self.fresh_id()
        cname = f"_til_lambda_{k}"
        pnames = [p for p, _t in lam.params]
        caps = []
        for node in walk(lam.body):
            if isinstance(node, til.Identifier) and node.name not in pnames and node.name not in [c[0] for c in caps]:
                found = self.lookup_var(node.name)
                if found is not None:
                    if isinstance(found[0], dict):
                        self.unsupported(f"the lambda '{name}' calls another lambda", lam)
                    caps.append((node.name, found[0], found[1]))
        saved = (self.output, self.indent, self.scopes, self.cnames_used, self._recover_stack, self._loop_depth)
        self.output, self.indent, self.scopes, self.cnames_used = [], 1, [], []
        self._recover_stack, self._loop_depth = [], 0
        try:
            self.push_scope()
            params = []
            for pname, pt in zip(pnames, ptypes):
                params.append(f"{self.ct(pt, lam, f'lambda parameter {pname}')} {self.declare_var(pname, pt, lam)}")
            for til_name, c_name, ct_ in caps:
                if c_name == 'self':
                    params.append(f"{self.ct(ct_, lam)} self")
                    self.scopes[-1]['self'] = ('self', ct_)
                else:
                    self.scopes[-1][til_name] = (c_name, ct_)
                    self.cnames_used[-1].add(c_name)
                    params.append(f"{self.ct(ct_, lam, f'captured variable {til_name}')} {c_name}")
            if isinstance(lam.body, til.Block):
                code = self.e_Block(lam.body)
            else:
                code = self.x(lam.body)
            bt = self.T(lam.body)
            if ret is NIL or bt is NIL:
                text = f"static void {cname}({', '.join(params) or 'void'}) {{ (void)({code}); }}"
            else:
                code = self.coerce(code, bt, ret, lam)
                text = f"static {self.ct(ret, lam)} {cname}({', '.join(params) or 'void'}) {{ return {code}; }}"
            self.pop_scope()
        finally:
            self.output, self.indent, self.scopes, self.cnames_used, self._recover_stack, self._loop_depth = saved
        self.lambda_defs.append(text)
        self.scopes[-1][name] = ({'cname': cname, 'caps': caps, 'ptypes': ptypes, 'ret': ret,
                                  'name': name}, ft)

    def call_lambda(self, info, args, n):
        if len(args) != len(info['ptypes']):
            self.unsupported(f"calling the lambda '{info['name']}' with {len(args)} argument(s)", n)
        for til_name, c_name, _t in info['caps']:
            found = self.lookup_var(til_name)
            if found is None or found[0] != c_name:
                self.unsupported(f"the lambda '{info['name']}' is called where its captured "
                                 f"variable '{til_name}' is not visible", n)
        prelude, codes = self.seq(args, lambda i, nd: (
            self.coerce(self.x(nd), self.T(nd), info['ptypes'][i], nd), info['ptypes'][i]))
        codes += [c for _t, c, _ty in info['caps']]
        call = f"{info['cname']}({', '.join(codes)})"
        if info['ret'] is NIL:
            call = f"({call}, NULL)"
        return self.wrap(prelude, call)

    def callable(self, node, arity: int, n):
        """A function argument of map/filter/...: (call(arg_codes) -> code, param types, ret)."""
        ft = self.T(node)
        if ft.name != 'fn' or len(ft.args) - 1 != arity:
            self.unsupported("a higher-order method with this function argument", n)
        ret, ptypes = ft.args[0], list(ft.args[1:])
        if isinstance(node, til.Lambda):
            def call(codes):
                def body():
                    self.push_scope()
                    for (pname, _pt), pt, code in zip(node.params, ptypes, codes):
                        cv = self.declare_var(pname, pt, node)
                        self.emit(f"{self.ct(pt, node)} {cv} = {code}; (void){cv};")
                    k = self.fresh_id()
                    if ret is NIL:
                        self.block_into(node.body, None, None)
                        self.emit(f"void* _lv{k} = NULL; (void)_lv{k};")
                    else:
                        self.emit(f"{self.ct(ret, node)} _lv{k} = {self.zero(ret)};")
                        self.block_into(node.body, f"_lv{k}", ret)
                    self.pop_scope()
                    return k
                saved_out, saved_indent = self.output, self.indent
                self.output, self.indent = [], 1
                try:
                    k = body()
                    lines = self.output
                finally:
                    self.output, self.indent = saved_out, saved_indent
                return self.sx(lines, f"_lv{k}")
            return call, ptypes, ret
        if isinstance(node, til.Identifier):
            found = self.lookup_var(node.name)
            if found is not None and isinstance(found[0], dict):
                info = found[0]
                return (lambda codes: self.call_lambda_codes(info, codes, n)), ptypes, ret
            if found is None and node.name in self.ti.funcs:
                target = self.inst.call(node)
                if target is None:
                    self.unsupported(f"the function '{node.name}' used as a value", n)
                key, params = target
                ikey = (key, tuple(canon(p) for p in params))
                cname = self.inst_names.get(ikey)
                if cname is None:
                    self.unsupported(f"the function '{node.name}' used as a value", n)
                callee = self.insts[ikey]
                if len(callee.params) != arity:
                    self.unsupported(f"the function '{node.name}' takes {len(callee.params)} argument(s)", n)

                def call_fn(codes):
                    args = [self.coerce(c, pt, cp, n) for c, pt, cp in zip(codes, ptypes, callee.params)]
                    code = f"{cname}({', '.join(args)})"
                    return f"({code}, NULL)" if callee.ret is NIL else code
                return call_fn, ptypes, ret
        self.unsupported("a higher-order method with this function argument", n)

    def call_lambda_codes(self, info, codes, n):
        for til_name, c_name, _t in info['caps']:
            found = self.lookup_var(til_name)
            if found is None or found[0] != c_name:
                self.unsupported(f"the lambda '{info['name']}' is used where its captured variable "
                                 f"'{til_name}' is not visible", n)
        args = [self.coerce(c, pt, pt, n) for c, pt in zip(codes, info['ptypes'])]
        call = f"{info['cname']}({', '.join(args + [c for _t, c, _ty in info['caps']])})"
        return f"({call}, NULL)" if info['ret'] is NIL else call

    # ── print ───────────────────────────────────────────────────────────────
    def print_call(self, n):
        """print(a, b, ...): like the VMs, every argument is evaluated before
        anything is printed (one line per argument)."""
        if not n.args:
            self.emit('til_print_str("");')
            return
        literal = (til.IntLit, til.FloatLit, til.StringLit, til.BoolLit, til.CharLit)
        hoist = len(n.args) > 1
        codes = []
        if hoist:
            self.emit("{")
            self.indent += 1
        for a in n.args:
            t = self.T(a)
            code = self.x(a)
            if hoist and not isinstance(a, literal):
                tmp = f"_pa{self.fresh_id()}"
                self.emit(f"{self.ct(t, a, 'a printed value')} {tmp} = {code};")
                code = tmp
            codes.append((a, t, code))
        for a, t, code in codes:
            if isinstance(a, til.StringLit):
                self.emit(f"til_print_str({code});")
            elif t is STR or t.name == 'strish':
                self.emit(f"til_print_s({code});")
            elif t is INT or t is UNK:
                self.emit(f"til_print_int({code});")
            elif t is FLOAT:
                self.emit(f"til_print_g({code});")
            elif t is BOOL:
                self.emit(f"til_print_bool({code});")
            else:
                self.emit(f"til_print_str({self.to_s(code, t, a)});")
        if hoist:
            self.indent -= 1
            self.emit("}")

    # ── natives ─────────────────────────────────────────────────────────────
    def native(self, name: str, n, statement=False):
        args = n.args
        if name in UNSUPPORTED_NATIVES:
            self.unsupported(f"{name}() (available on the TIL VM)", n)
        types = [self.T(a) for a in args]
        prelude, codes = self.seq(args, lambda i, nd: (self.x(nd), self.T(nd)))
        w = lambda code: self.wrap(prelude, code)          # noqa: E731
        t0 = types[0] if types else UNK
        a0 = codes[0] if codes else ""
        if name == 'len':
            if t0 is STR:
                return w(f"til_str_len({a0})")
            if t0.name == 'array':
                return w(f"til_arr_len({a0})")
            if t0.name == 'map':
                return w(f"(til_map_chk({a0})->len)")
            self.unsupported(f"len() of {show(t0)}", n)
        if name == 'str':
            return w(self.to_s(a0, t0, n))
        if name in ('int', 'float'):
            fake = til.Cast(expr=args[0], target_type=til.PrimitiveType(name), line=n.line, col=n.col)
            self.inst.ctx.types[id(fake)] = INT if name == 'int' else FLOAT
            return w(self.cast_code(a0, t0, INT if name == 'int' else FLOAT, n))
        if name == 'bool':
            return w(self.truthy_code(a0, t0))
        if name == 'abs':
            if t0 in (INT, UNK):
                return w(f"til_abs_i({a0})")
            if t0 is FLOAT:
                return w(f"fabs({a0})")
            self.unsupported(f"abs() of {show(t0)}", n)
        if name in ('sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'floor', 'ceil', 'round'):
            if t0 not in (INT, FLOAT, UNK):
                self.unsupported(f"{name}() of {show(t0)}", n)
            fn = {'round': 'til_round_vm', 'sqrt': 'sqrt', 'log': 'log'}.get(name, name)
            return w(f"{fn}((double)({a0}))")
        if name == 'pow':
            if any(t not in (INT, FLOAT, UNK) for t in types):
                self.unsupported("pow() of non-numbers", n)
            return w(f"til_fpow((double)({codes[0]}), (double)({codes[1]}))")
        if name in ('min', 'max', 'sum'):
            return w(self.fold_native(name, codes, types, n))
        if name == 'input':
            return w(f"til_input({self.to_s(a0, t0, n) if args else 'NULL'})")
        if name == 'chr':
            return w(f"til_chr({self.cast_code(a0, t0, INT, n)})")
        if name == 'ord':
            if t0 is not STR:
                self.unsupported("ord() of a non-string", n)
            return w(f"til_ord({a0})")
        if name == 'range':
            if any(t not in (INT, UNK) for t in types[:2]):
                self.unsupported("range() with non-integer bounds", n)
            if len(codes) == 1:
                return w(f"til_range(0, {codes[0]}, false)")
            incl = self.truthy_code(codes[2], types[2]) if len(codes) > 2 else "false"
            return w(f"til_range({codes[0]}, {codes[1]}, {incl})")
        if name == 'type_of':
            return w(self.type_of(a0, t0, n))
        if name == 'format':
            if types[1] is not STR:
                self.unsupported("format() with a non-string spec", n)
            k = self.fresh_id()
            if t0 in (INT, UNK):
                return w(self.sx([f"int64_t _f{k} = {codes[0]};"],
                                 f"til_fmt_spec(0, _f{k}, 0.0, til_int_to_str(_f{k}), {codes[1]})"))
            if t0 is FLOAT:
                return w(self.sx([f"double _f{k} = {codes[0]};"],
                                 f"til_fmt_spec(1, 0, _f{k}, til_float_str(_f{k}), {codes[1]})"))
            return w(f"til_fmt_spec(2, 0, 0.0, {self.to_s(codes[0], t0, n)}, {codes[1]})")
        if name == 'clock_ms':
            return w("til_clock_ms()")
        if name == 'random':
            if not codes:
                return w("til_random()")
            if types[0] in (INT, UNK) and types[1] in (INT, UNK):
                return w(f"til_random_range({codes[0]}, {codes[1]})")
            self.unsupported("random() with non-integer bounds", n)
        if name == 'seed':
            return w(f"(til_seed({self.cast_code(a0, t0, INT, n)}), NULL)")
        if name == 'sleep_ms':
            return w(f"(til_sleep_ms({self.cast_code(a0, t0, INT, n)}), NULL)")
        if name == 'exit':
            code = self.cast_code(a0, t0, INT, n) if codes else "0"
            return w(f"(fflush(stdout), exit((int)({code})), NULL)")
        if name == 'panic':
            msg = self.to_s(a0, t0, n) if codes else '"panic"'
            return w(f"(til_panic({msg}), NULL)")
        if name in ('read_file', 'write_file', 'file_exists'):
            if t0 is not STR:
                return w(f"(til_panic(\"{name}() expects a path string, got {self.vm_type_name(t0)}\"), "
                         f"{'NULL' if name != 'file_exists' else 'false'})")
            if name == 'read_file':
                return w(f"til_read_file({a0})")
            if name == 'file_exists':
                return w(f"til_file_exists({a0})")
            return w(f"(til_write_file({a0}, {self.to_s(codes[1], types[1], n)}), NULL)")
        if name == 'args':
            return w("til_args()")
        if name == 'str_bytes':
            if t0 is not STR:
                self.unsupported("str_bytes() of a non-string", n)
            return w(f"til_str_bytes({a0})")
        self.unsupported(f"{name}()", n)

    def cast_code(self, code, src, dst, n):
        if dst is INT:
            if src in (INT, UNK):
                return code
            if src is FLOAT:
                return f"til_f2i({code})"
            if src is BOOL:
                return f"((int64_t){code})"
            if src is STR:
                return f"til_str_to_int({code})"
            if src is NIL:
                return "0"
        if dst is FLOAT:
            if src in (INT, BOOL, UNK):
                return f"((double){code})"
            if src is FLOAT:
                return code
            if src is STR:
                return f"til_str_to_float({code})"
            if src is NIL:
                return "0.0"
        self.unsupported(f"converting {show(src)} to {show(dst)}", n)

    def type_of(self, code, t, n):
        if t is INT or t is UNK:
            name = "int"
        elif t is FLOAT:
            name = "float"
        elif t is BOOL:
            name = "bool"
        elif t is NIL:
            return f"((void)({code}), \"nil\")"
        elif t.name == 'option':
            return f"(({code}).has_value ? \"Some\" : \"nil\")"
        elif t.name == 'result':
            return f"(({code}).is_ok ? \"Ok\" : \"Err\")"
        elif is_adt(t):
            return f"(({code}) ? {self.adt_c(nominal_name(t))}_names[({code})->tag] : \"nil\")"
        elif t is STR or t.name in ('array', 'map') or is_struct(t):
            label = {'str': 'str', 'array': 'array', 'map': 'map'}.get(t.name) or nominal_name(t)
            return f"(({code}) ? {til.c_string_literal(label)} : \"nil\")"
        else:
            self.unsupported(f"type_of() of {show(t)}", n)
        return f"((void)({code}), \"{name}\")"

    def fold_native(self, name, codes, types, n):
        """min / max / sum of arguments or of one array."""
        if len(codes) == 1 and types[0].name == 'array':
            e = types[0].args[0]
            return self.fold_array(name, codes[0], e, n)
        if len(codes) == 1:
            t = types[0]
            if name == 'sum':
                if t in (INT, UNK, FLOAT):
                    return codes[0]
                self.unsupported(f"sum() of {show(t)}", n)
            return codes[0]
        if name == 'sum':
            if all(t in (INT, UNK) for t in types):
                return "(" + " + ".join(codes) + ")"
            if all(t in (INT, UNK, FLOAT) for t in types):
                return "(" + " + ".join(f"(double)({c})" for c in codes) + ")"
            self.unsupported("sum() of non-numbers", n)
        first = types[0]
        if not all(t is first for t in types):
            self.unsupported(f"{name}() of mixed types", n)
        fn = {INT: 'i', UNK: 'i', FLOAT: 'f', STR: 's'}.get(first)
        if fn is None:
            self.unsupported(f"{name}() of {show(first)}", n)
        code = codes[0]
        for c in codes[1:]:
            code = f"til_{name}_{fn}({code}, {c})"
        return code

    def fold_array(self, name, code, e, n):
        k = self.fresh_id()
        if name == 'sum':
            if e in (INT, UNK):
                return self.sx([f"TIL_Arr* _a{k} = {code}; int64_t _s{k} = 0;",
                                f"for (int64_t _i{k} = 0; _i{k} < til_arr_len(_a{k}); _i{k}++) "
                                f"_s{k} += _a{k}->data[_i{k}].i;"], f"_s{k}")
            if e is FLOAT:
                return self.sx([f"TIL_Arr* _a{k} = {code}; double _s{k} = 0;",
                                f"for (int64_t _i{k} = 0; _i{k} < til_arr_len(_a{k}); _i{k}++) "
                                f"_s{k} += _a{k}->data[_i{k}].f;"], f"_s{k}")
            self.unsupported(f"sum() of [{show(e)}]", n)
        fn = {INT: 'i', UNK: 'i', FLOAT: 'f', STR: 's'}.get(e)
        if fn is None:
            self.unsupported(f"{name}() of [{show(e)}]", n)
        f = self.slot_field(e)
        c = self.ct(e)
        return self.sx([f"TIL_Arr* _a{k} = {code};",
                        f"if (til_arr_len(_a{k}) == 0) til_panic(\"{name}() of empty sequence\");",
                        f"{c} _b{k} = _a{k}->data[0].{f};",
                        f"for (int64_t _i{k} = 1; _i{k} < _a{k}->len; _i{k}++) "
                        f"_b{k} = til_{name}_{fn}(_b{k}, _a{k}->data[_i{k}].{f});"], f"_b{k}")

    # ── methods ─────────────────────────────────────────────────────────────
    def method_call(self, n, statement=False):
        f = n.func
        meth = f.attr
        obj = f.obj
        rt = self.T(obj)
        args = n.args
        if meth == 'to_string':
            return self.to_s(self.x(obj), rt, n)
        if is_struct(rt):
            target = self.inst.call(n)
            if target is None:
                self.unsupported(f"method '{meth}' of struct '{nominal_name(rt)}'", n)
            key, params = target
            ikey = (key, tuple(canon(p) for p in params))
            callee = self.insts.get(ikey)
            if callee is None:
                self.unsupported(f"no instance of '{key}' for these types", n)
            if isinstance(obj, til.Identifier) and obj.name == 'self' and self.lookup_var('self') == ('self', rt):
                recv = "self"
            else:
                sc = self.mangle_name(nominal_name(rt))
                recv = f"(({sc}*)til_nm({self.x(obj)}, {til.c_string_literal(meth)}))"
            if self.effectful(obj) and any(self.effectful(a) for a in args):
                k = self.fresh_id()
                return self.sx([f"{self.ct(rt)} _r{k} = {recv};"],
                               self.call_instance(callee, self.inst_names[ikey], obj, f"_r{k}", args, n))
            return self.call_instance(callee, self.inst_names[ikey], obj, recv, args, n)
        nodes = [obj] + list(args)
        if meth in ('map', 'filter', 'reduce', 'each', 'any', 'all') and rt.name == 'array':
            return self.hof(meth, n, rt)
        prelude, codes = self.seq(nodes, lambda i, nd: (self.x(nd), self.T(nd)))
        recv, acodes = codes[0], codes[1:]
        atypes = [self.T(a) for a in args]
        if rt is STR:
            code = self.str_method(meth, recv, acodes, atypes, n)
        elif rt.name == 'array':
            code = self.array_method(meth, recv, rt.args[0], acodes, atypes, n)
        elif rt.name == 'map':
            code = self.map_method(meth, recv, rt, acodes, atypes, n)
        elif rt.name in ('option', 'result'):
            code = self.optres_method(meth, recv, rt, acodes, atypes, n)
        elif rt is NIL and meth in ('is_some', 'is_ok'):
            code = f"((void)({recv}), false)"
        elif rt is NIL and meth in ('is_none', 'is_err'):
            code = f"((void)({recv}), true)"
        else:
            self.unsupported(f"method '{meth}' on a value of type {show(rt)}", n)
        return self.wrap(prelude, code)

    def str_method(self, meth, s, a, at, n):
        def need_str(i):
            if i >= len(at) or at[i] is not STR:
                self.unsupported(f"str.{meth} with a non-string argument", n)
        if meth == 'len':
            return f"til_str_len({s})"
        if meth in ('contains', 'starts_with', 'ends_with', 'eq'):
            need_str(0)
            fn = {'contains': 'til_str_contains', 'starts_with': 'til_str_starts_with',
                  'ends_with': 'til_str_ends_with', 'eq': 'til_str_eqn'}[meth]
            return f"{fn}({s}, {a[0]})"
        if meth in ('trim', 'to_upper', 'to_lower'):
            return f"til_str_{meth}({s})"
        if meth == 'replace':
            need_str(0)
            return f"til_str_replace2({s}, {a[0]}, {self.to_s(a[1], at[1], n)})"
        if meth == 'slice':
            k = self.fresh_id()
            hi = a[1] if len(a) > 1 else f"til_str_len(_ss{k})"
            return self.sx([f"const char* _ss{k} = {s};"], f"til_str_slice(_ss{k}, {a[0]}, {hi})")
        if meth in ('find', 'index_of'):
            need_str(0)
            return f"til_str_find({s}, {a[0]})"
        if meth == 'count':
            need_str(0)
            return f"til_str_count({s}, {a[0]})"
        if meth == 'split':
            if not a:
                return f"til_str_split_ws({s})"
            need_str(0)
            return f"til_str_split({s}, {a[0]})"
        if meth == 'repeat':
            return f"til_str_rep({s}, {a[0]})"
        if meth == 'to_int':
            return f"til_str_to_int({s})"
        if meth == 'to_float':
            return f"til_str_to_float({s})"
        if meth == 'is_empty':
            return f"(til_str_len({s}) == 0)"
        if meth == 'get':
            return f"til_str_at({s}, {a[0]})"
        self.unsupported(f"str.{meth}()", n)

    def array_method(self, meth, a, e, args, at, n):
        if meth not in ('len', 'reverse', 'clear', 'is_empty', 'copy', 'slice', 'insert', 'remove', 'pop',
                        'push', 'get', 'set', 'first', 'last', 'join', 'contains', 'index_of', 'count',
                        'sort', 'sum', 'min', 'max'):
            self.unsupported(f"array.{meth}()", n)
        if meth not in ('len', 'reverse', 'clear', 'is_empty', 'copy', 'slice'):
            self.check_slot_type(e, n)
        k = self.fresh_id()
        if meth == 'len':
            return f"til_arr_len({a})"
        if meth == 'push':
            return f"(til_arr_push({a}, {self.slot_make(e, self.coerce(args[0], at[0], e, n))}), NULL)"
        if meth == 'pop':
            return self.slot_get(e, f"til_arr_pop({a})")
        if meth == 'get':
            return self.slot_get(e, f"*til_arr_at({a}, {args[0]})")
        if meth == 'set':
            return f"(*til_arr_at({a}, {args[0]}) = {self.slot_make(e, self.coerce(args[1], at[1], e, n))}, NULL)"
        if meth == 'insert':
            return f"(til_arr_insert({a}, {args[0]}, {self.slot_make(e, self.coerce(args[1], at[1], e, n))}), NULL)"
        if meth == 'remove':
            return self.slot_get(e, f"til_arr_remove({a}, {args[0]})")
        if meth == 'reverse':
            return f"(til_arr_reverse({a}), NULL)"
        if meth == 'clear':
            return f"(til_arr_clear({a}), NULL)"
        if meth == 'is_empty':
            return f"(til_arr_len({a}) == 0)"
        if meth == 'copy':
            return f"til_arr_copy({a})"
        if meth == 'slice':
            hi = args[1] if len(args) > 1 else f"til_arr_len(_sa{k})"
            return self.sx([f"TIL_Arr* _sa{k} = {a};"], f"til_arr_slice(_sa{k}, {args[0]}, {hi})")
        if meth in ('first', 'last'):
            idx = "0" if meth == 'first' else f"_fa{k}->len - 1"
            if is_ptr(e) or e is NIL:
                return self.sx([f"TIL_Arr* _fa{k} = {a};"],
                               f"(til_arr_len(_fa{k}) ? {self.slot_get(e, f'_fa{k}->data[{idx}]')} : NULL)")
            # the VMs give None for an empty array; a native int/float/bool cannot be None
            return self.sx([f"TIL_Arr* _fa{k} = {a};",
                            f"if (til_arr_len(_fa{k}) == 0) til_panic(\"{meth}() of an empty array is None, "
                            f"which the native backend cannot represent as {show(e)}\");"],
                           self.slot_get(e, f"_fa{k}->data[{idx}]"))
        if meth == 'join':
            sep = self.to_s(args[0], at[0], n) if args else '""'
            item = self.to_s(self.slot_get(e, f"_ja{k}->data[_i{k}]"), e, n)
            return self.sx([f"TIL_Arr* _ja{k} = {a}; const char* _js{k} = {sep}; TIL_Buf _jb{k} = {{0, 0, 0}};",
                            f"for (int64_t _i{k} = 0; _i{k} < til_arr_len(_ja{k}); _i{k}++) {{ "
                            f"if (_i{k}) til_buf_add(&_jb{k}, _js{k}); til_buf_add(&_jb{k}, {item}); }}"],
                           f"til_buf_done(&_jb{k})")
        if meth in ('contains', 'index_of', 'count'):
            val_t = at[0]
            eq = self.eq_code(self.slot_get(e, f"_ca{k}->data[_i{k}]"), e, f"_cv{k}", val_t, n)
            lines = [f"TIL_Arr* _ca{k} = {a}; {self.ct(val_t, n)} _cv{k} = {args[0]}; int64_t _cr{k} = "
                     f"{'-1' if meth != 'count' else '0'};",
                     f"for (int64_t _i{k} = 0; _i{k} < til_arr_len(_ca{k}); _i{k}++) if ({eq}) "
                     + (f"_cr{k}++;" if meth == 'count' else f"{{ _cr{k} = _i{k}; break; }}")]
            if meth == 'contains':
                return self.sx(lines, f"(_cr{k} >= 0)")
            return self.sx(lines, f"_cr{k}")
        if meth == 'sort':
            cmp = {INT: 'til_cmp_i', UNK: 'til_cmp_i', FLOAT: 'til_cmp_f', STR: 'til_cmp_s'}.get(e)
            if cmp is None:
                return f"((void)({a}), til_panic(\"sort() needs an array of numbers or of strings\"), NULL)"
            return f"(til_arr_sort({a}, {cmp}), NULL)"
        if meth in ('sum', 'min', 'max'):
            return self.fold_array(meth, a, e, n)
        self.unsupported(f"array.{meth}()", n)

    def hof(self, meth, n, rt):
        """map / filter / reduce / each / any / all on an array, with a lambda or a
        named function: an inline loop (the VMs run the same loop in TIL)."""
        e = rt.args[0]
        self.check_slot_type(e, n)
        args = n.args
        if not args:
            self.unsupported(f"{meth}() without a function", n)
        k = self.fresh_id()
        call, ptypes, ret = self.callable(args[0], 2 if meth == 'reduce' else 1, n)
        lines = [f"TIL_Arr* _ha{k} = {self.x(n.func.obj)};"]
        elem = self.coerce(self.slot_get(e, f"_ha{k}->data[_hi{k}]"), e, ptypes[-1], n)
        loop = f"for (int64_t _hi{k} = 0; _hi{k} < til_arr_len(_ha{k}); _hi{k}++)"
        if meth == 'map':
            out_t = self.T(n).args[0]
            self.check_slot_type(out_t, n)
            lines.append(f"TIL_Arr* _hr{k} = til_arr_new(0);")
            res = self.coerce(call([elem]), ret, out_t, n)
            lines.append(f"{loop} til_arr_push(_hr{k}, {self.slot_make(out_t, res)});")
            return self.sx(lines, f"_hr{k}")
        if meth == 'filter':
            lines.append(f"TIL_Arr* _hr{k} = til_arr_new(0);")
            lines.append(f"{loop} {{ {self.ct(e)} _hx{k} = {self.slot_get(e, f'_ha{k}->data[_hi{k}]')}; "
                         f"if ({self.truthy_code(call([self.coerce(f'_hx{k}', e, ptypes[0], n)]), ret)}) "
                         f"til_arr_push(_hr{k}, {self.slot_make(e, f'_hx{k}')}); }}")
            return self.sx(lines, f"_hr{k}")
        if meth == 'reduce':
            acc_t = self.T(n)
            if len(args) < 2:
                self.unsupported("reduce() without an initial value", n)
            init = self.coerce(self.x(args[1]), self.T(args[1]), acc_t, n)
            lines.insert(0, f"{self.ct(acc_t, n)} _hacc{k} = {init};")
            res = self.coerce(call([self.coerce(f'_hacc{k}', acc_t, ptypes[0], n), elem]), ret, acc_t, n)
            lines.append(f"{loop} _hacc{k} = {res};")
            return self.sx(lines, f"_hacc{k}")
        if meth == 'each':
            lines.append(f"{loop} (void)({call([elem])});")
            return self.sx(lines, "NULL")
        if meth in ('any', 'all'):
            found = "true" if meth == 'any' else "false"
            test = self.truthy_code(call([elem]), ret)
            lines.append(f"bool _hr{k} = {'false' if meth == 'any' else 'true'};")
            cond = test if meth == 'any' else f"!{test}"
            lines.append(f"{loop} if ({cond}) {{ _hr{k} = {found}; break; }}")
            return self.sx(lines, f"_hr{k}")
        self.unsupported(f"array.{meth}()", n)

    def map_method(self, meth, m, rt, args, at, n):
        k_t, v_t = rt.args
        if meth in ('len', 'is_empty'):
            return f"(til_map_chk({m})->len{' == 0' if meth == 'is_empty' else ''})"
        self.check_slot_type(k_t, n, "a map key")
        self.check_slot_type(v_t, n, "a map value")
        kind = self.key_kind(k_t, n)
        key = lambda i: self.slot_make(k_t, self.coerce(args[i], at[i], k_t, n))   # noqa: E731
        if meth == 'set':
            return f"(til_map_set({m}, {key(0)}, {self.slot_make(v_t, self.coerce(args[1], at[1], v_t, n))}, {kind}), NULL)"
        if meth == 'get':
            return self.slot_get(v_t, f"til_map_get({m}, {key(0)}, {kind})")
        if meth == 'get_or':
            k = self.fresh_id()
            dflt = self.coerce(args[1], at[1], v_t, n)
            return self.sx([f"TIL_Map* _gm{k} = {m}; int64_t _gi{k} = til_map_find(_gm{k}, {key(0)}, {kind});"],
                           f"(_gi{k} >= 0 ? {self.slot_get(v_t, f'_gm{k}->vals[_gi{k}]')} : {dflt})")
        if meth in ('has', 'contains'):
            return f"(til_map_find({m}, {key(0)}, {kind}) >= 0)"
        if meth in ('delete', 'remove'):
            return f"(til_map_delete({m}, {key(0)}, {kind}), NULL)"
        if meth == 'keys':
            return f"til_map_keys({m})"
        if meth == 'values':
            return f"til_map_values({m})"
        if meth == 'clear':
            return f"(til_map_clear({m}), NULL)"
        if meth == 'copy':
            return f"til_map_copy({m}, {kind})"
        self.unsupported(f"map.{meth}()", n)

    def optres_method(self, meth, o, rt, args, at, n):
        is_opt = rt.name == 'option'
        t = rt.args[0]
        k = self.fresh_id()
        c = self.ct(rt, n)
        flag = "has_value" if is_opt else "is_ok"
        if meth in ('is_some', 'is_ok'):
            return f"({o}).{flag}"
        if meth in ('is_none', 'is_err'):
            return f"(!({o}).{flag})"
        if meth == 'unwrap':
            if is_opt:
                msg = '"unwrap() called on None"'
            else:
                msg = f"til_str_concat(\"unwrap() called on Err: \", {self.to_s(f'_u{k}.error', rt.args[1], n)})"
            return self.sx([f"{c} _u{k} = {o};", f"if (!_u{k}.{flag}) til_panic({msg});"], f"_u{k}.value")
        if meth == 'expect':
            return self.sx([f"{c} _u{k} = {o};",
                            f"if (!_u{k}.{flag}) til_panic({self.to_s(args[0], at[0], n)});"], f"_u{k}.value")
        if meth == 'unwrap_or':
            return self.sx([f"{c} _u{k} = {o};"],
                           f"(_u{k}.{flag} ? _u{k}.value : {self.coerce(args[0], at[0], t, n)})")
        if meth == 'error' and not is_opt:
            err = rt.args[1]
            if not is_ptr(err):
                self.unsupported("Result.error() of a non-reference error type", n)
            return self.sx([f"{c} _u{k} = {o};"], f"(_u{k}.is_ok ? NULL : _u{k}.error)")
        self.unsupported(f"{'Option' if is_opt else 'Result'}.{meth}()", n)


# ── public API (native backend) ─────────────────────────────────────────────

def generate_c(program, source: str = "", filename: str = "<stdin>", ti: Optional[TypeInfer] = None) -> str:
    """C code for a parsed, import-resolved program, or NativeUnsupported."""
    if ti is None:
        ti = infer_program(program, source, filename)
    return TypedCGen(ti).generate(program)
