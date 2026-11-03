# Effects with resumption (TIL 3.1)

A function *performs* an effect; the nearest `handle ... with` around the call
(dynamically, across any number of calls) decides what the effect means.

```til
greet() -> str
    let name = perform Ask("name?")
    perform Log("hello " + name)
    return name

let r = handle
    greet()
    "done"
with
    Ask(question) => resume("Aigerim")    # continue greet() with "Aigerim"
    Log(msg) =>
        print(msg)
        resume(None)
    Fail(reason) => "failed: " + reason   # no resume: ends the whole handle
print(r)                                   # hello Aigerim / done
```

Effect handlers run on the TIL VM (`til run --vm`, `til build --target vm`, the
C VM `vm/tilvm.c` on every device). `til run` falls back to the VM
automatically; the native C backend reports them as unsupported (see
[Native backend](#native-backend)). Example program: `examples/12_effects.til`;
tests: `tests/test_effects.py`. Design spec: `docs/dev/TIL_3_1_DESIGN.md` §8.

## 1. Syntax

```til
handle                              # block form
    <body>                          # any statements; the last expression is the value
with
    Name(p1, p2) => <expression>    # one-line clause
    Name(p: float) =>               # block clause (typed parameters are converted
        <statements>                #   like function parameters)
    Tick => <expression>            # no parameters: `Tick` or `Tick()`

handle <expression>                 # expression body
with
    Name(x) => ...

handle compute() with Ask() => resume(1)    # one line (also inside parentheses)

perform Name(args)                  # an expression: its value is what resume() passes
perform Name                        # no arguments
resume(v)   resume()   resume       # only inside a clause; resume() == resume(None)
```

* `handle` is an expression: `let x = handle ...`, `return handle ...`, or a
  statement (value discarded).
* A clause head is an identifier and an optional parameter list; parameters may
  carry a type (`x: float` converts like a function parameter). Clauses for
  different effects may appear in any order; **two clauses for one effect in
  one handle are a compile error**.
* `with` and `resume` are keywords since TIL 3.1 (they cannot be variable
  names any more; `perform` and `handle` were already reserved in 3.0).

| English | Kazakh (always on) | Kazakh Latin (`#[lang: kk-latn]`) | Russian (`#[lang: ru]`) |
|---|---|---|---|
| `perform` | `орындау` | `oryndau` | `выполнить` |
| `handle` | `өңдеу` | `ondeu` | `обработать` |
| `with` | `арқылы` | `arqyly` | `с` |
| `resume` | `жалғау` | `zhalgau` | `возобновить` |

`til translate --to kk|kk-latn|ru|en` converts all four and round-trips.

### Compile-time errors

| Program | Error |
|---|---|
| two clauses `A(...)` in one handle | `duplicate clause for effect 'A' in one handle` |
| `resume` outside a clause (also in a lambda inside a clause) | `resume(...) outside an effect clause` |
| a statement directly after `resume(...)` in the same block | `unreachable code after resume(...)` |
| `return` in a clause | `'return' is not allowed in an effect clause` |
| `break` / `continue` that would leave a clause | `'break' cannot leave an effect clause` |
| `perform A(1, 2)` lexically inside a handle whose clause `A` takes 1 parameter | `effect 'A' expects 1 argument(s), got 2` (E-VM) |
| `handle` body without `with` | `Expected 'with' and effect clauses after the body of 'handle'` |

The argument count is checked at compile time only when the handler is
statically known: the perform is in the same function as the handle (not in
a lambda) and the innermost enclosing handle with a clause for the effect is
used (a perform inside a clause skips that clause's own handle). Everything
else is checked at run time.

## 2. Semantics

**Dynamic, innermost, deep.** `perform E(args)` looks for the innermost active
handler that has a clause for `E` - across calls, closures, methods, prelude
functions (`map`, `each`, ...), `retry`/`timeout` bodies. The clause runs with
its parameters bound to the arguments. The handler stays installed for the
whole body (deep handlers): every perform in the body finds it again.

**`resume(v)` - one-shot tail resumption.** `resume(v)` immediately returns `v`
as the value of the `perform` expression and the body continues; the clause is
finished at that point. Statements after `resume` in the clause never run (a
statement directly after it is a compile error; code reachable only through
other paths, e.g. after an `if` that resumes in one branch, is fine). The body
can be resumed at most once per perform.

**No resume - abort.** When a clause finishes without `resume`, its value (the
last expression of the clause; `None` if it ends with a statement) becomes the
value of the whole `handle` expression. Everything between the handle and the
perform is abandoned: call frames, temporaries, inner handlers, and the
recovery points of `retry`/`timeout` blocks entered inside the body.

**Value of a handle.** The value of the body (its last expression, like
`retry`/`timeout` blocks; `None` when the body ends with a statement), unless a
clause aborts it.

**Masking.** While a clause runs, its own handle - *all* of its clauses - and
every handler installed inside that handle's body are invisible. A perform
inside a clause searches outward from the handler outside the owning handle:

```til
handle
    handle
        perform Log("a")            # handled by the inner handle
    with
        Log(m) =>
            perform Log("inner: " + m)   # goes to the OUTER handle
            resume()
with
    Log(m) =>
        print(m)
        resume()
```

Handlers installed *inside* the clause itself are visible to the clause's own
performs. After `resume` the handler stack is exactly as it was at the perform.

**No handler.** If no visible handler has a clause for `E`, perform calls the
top-level function named `E` (the TIL 3.0 meaning of `perform`, so 3.0
programs keep working), also when the handler is masked. Without such a
function the runtime error `unhandled effect 'E'` is raised (catchable with
`retry`/`timeout` like every runtime error).

**Argument count.** A clause with N parameters performed with M != N arguments
raises the runtime error `effect 'E' expects N argument(s), got M` (or the
compile-time error above when statically known). The default function is
called like any function (`E() takes N argument(s), got M`).

**Runtime errors inside a clause** propagate like an error raised at the
perform: the innermost `retry`/`timeout` that is active at the perform catches
it - including one *inside* the handle body around the perform (a clause is a
call made at the perform site). Handler entries installed after that recovery
point are discarded. Example (from `examples/12_effects.til`):

```til
let reading = handle
    retry(3)
        perform Sensor()        # the clause panics twice, then resumes 42
    else
        -1
with
    Sensor() =>
        calls += 1
        if calls < 3
            panic("sensor busy")
        resume(42)
```

**Leaving a handle body** with `return`, `break`, `continue` or `?` removes its
handler. `retry`/`timeout` may be used inside clauses; `resume` from inside
them drops their recovery points.

**Closures.** Clauses are closures: they see the variables in scope at the
`handle` expression (not the variables declared inside its body) and may
assign `var`s (`count += 1` in a clause updates the enclosing function's
variable). A lambda created in a handle body performs through whatever
handlers are active when it is *called*.

**Limits.** At most 32 handler entries are active at a time
(`TILVM_MAX_HANDLERS`; a handle uses one entry while its body runs, a running
clause uses one more); beyond that: runtime error `too many nested handlers`.
At most 255 clauses per handle and 255 arguments per perform. Effect names in
runtime error messages are cut to 64 bytes (both VMs).

### Effect checking (`#[pure]`, `#[effects: ...]`)

`perform E` inside the body of a handle that has a clause for `E` is not an
effect of the enclosing function, and neither is `E` coming from a call inside
such a body. Clause bodies count normally (a `print` in a clause is `io`), and
a perform inside a clause is checked against the handles outside the clause's
own handle (masking). Lambdas written inside a handle body are assumed to run
inside it (the checker is lexical).

```til
#[pure]
total(xs) -> int
    return handle
        xs.map(|x| perform Weight(x)).sum()
    with
        Weight(x) => resume(x * 2)       # pure: Weight is handled here
```

### Native backend

The C backend compiles `perform E(args)` without handlers as a call of the
function `E` (unchanged from 3.0). A program containing `handle`/`resume` (or
performing an effect that has no top-level function) raises
`NativeUnsupported` (a `CompileError`): `til run` prints
`[TIL] native backend cannot compile this program; running it on the TIL VM`
and runs it on the reference VM; `til build` (native) fails with
`the native backend does not support effect handlers ...` and suggests
`til build --target vm`.

## 3. Bytecode (format v2)

### Compilation scheme

```
handle B with E1(...) => C1 ... En(...) => Cn        perform E(a1..am)

    <closure of C1>  PUSH_CONST "E1"                     <a1> ... <am>
    ...                                                  PERFORM "E", default, m
    <closure of Cn>  PUSH_CONST "En"
    HANDLE n, end
    <value of B>
    END_HANDLE
end:
```

* Each clause is a separate function with flag `FLAG_EFFECT_CLAUSE = 16`
  (named `<clause E:LINE>` in tracebacks and `til dis`), `n_params =
  n_required =` number of clause parameters. It is pushed with `PUSH_FUNC`, or
  `MAKE_CLOSURE` when it captures variables (captured locals live in cells,
  like for lambdas). Its last instruction is `RETURN` of the clause value.
* `resume(v)` compiles to `<v> RESUME` (`resume()` pushes nil).
* `break`/`continue` leaving k handle bodies emit `PUSH_NIL`, `END_HANDLE` x k,
  `POP` before the jump. `return` and `?` need nothing: returning drops the
  handler entries of the frame.

### Handler stack

Both VMs keep a stack of handler entries next to the recovery points. Every
entry has the same six fields:

| field | kind 0: active `handle` | kind 1: running clause |
|---|---|---|
| `kind` | 0 | 1 |
| `n_frames` | number of call frames when HANDLE ran (the handle's frame is frame `n_frames - 1`) | number of frames at the PERFORM (the perform site is frame `n_frames - 1`; the clause frame is frame `n_frames`) |
| `sp` | value-stack index of the first (clause, name) pair | 0 |
| `n_try` | number of recovery points when HANDLE ran | number of recovery points at the PERFORM |
| `abort_pc` | absolute code offset after END_HANDLE (`pc_after_HANDLE + rel_end`) | 0 |
| `n` | number of clauses | index of the owning kind-0 entry |

* Python VM: `TILVM.handlers` is a list of 6-element lists
  `[kind, n_frames, sp, n_try, abort_pc, n]`; `TILVM.MAX_HANDLERS = 32`.
* C VM: `tilvm_state.handlers[TILVM_MAX_HANDLERS]` (`tv_handler`: `uint32_t
  abort_pc; uint16_t n_frames, sp, n_try, n; uint8_t kind, _pad[3]` - 16 bytes)
  and `tilvm_state.n_handlers`.
* Recovery points (`TRY`/`TRY_TIMEOUT`) also record the number of handler
  entries: Python `recover` entries are `[handler_pc, n_frames, sp, deadline,
  n_handlers]`, C `tv_try.n_handlers`.
* The clause values and names live on the **value stack** at
  `stack[sp + 2j]` (clause, a closure or function value) and
  `stack[sp + 2j + 1]` (effect name, a string), j = 0..n-1. They are ordinary
  stack values, so the garbage collector marks (and moves) them like every
  other root, and snapshots save them with the stack. Handler entries
  themselves contain only integers.
* Invariants (check them when loading a snapshot): entries are ordered by
  `n_frames` (non-decreasing); a kind-1 entry's owner index is smaller than
  its own index and points at a kind-0 entry; `sp + 2n <= SP`; kind-0
  `n_frames >= 1`.

### Instructions

| op | hex | operands | stack | meaning |
|---|---|---|---|---|
| HANDLE | 76 | `b n`, `s rel_end` | `[.. c1 "E1" .. cn "En"]` unchanged | Push a kind-0 entry: `sp = SP - 2n`, `abort_pc = pc after the instruction + rel_end`, `n_frames`, `n_try` = current counts. Error `too many nested handlers` (recoverable) when the stack of entries is full. |
| END_HANDLE | 77 | - | `[.. pairs(2n) v]` -> `[.. v]` | The body finished: pop the top entry (it must be a kind-0 entry of the current frame, else fatal `END_HANDLE without an active handle`) and remove its pairs, keeping the value on top. |
| PERFORM | 78 | `h name` (string constant), `h default` (function index or `0xFFFF` = none), `b argc` | `[.. a1..am]` -> (after the clause resumes / the default function returns) `[.. v]` | Search the entries from the top: a kind-1 entry with owner `o` continues the search at entry `o - 1`; a kind-0 entry is searched for a pair whose name equals `name` (byte equality; the first matching pair wins). Found: check `n_required <= argc <= n_params` of the clause function (else `effect 'E' expects N argument(s), got M`), push a kind-1 entry (owner = found index; `too many nested handlers` when full) and call the clause with the m arguments (a new frame, no callee slot). Not found: call `default` like `CALL default, argc`, or raise `unhandled effect 'E'` when `default = 0xFFFF`. |
| RESUME | 79 | - | `[.. v]` in the clause frame -> `[.. v]` at the perform site | Find the topmost kind-1 entry; it must belong to the current frame (`n_frames == frames - 1`) and the current function must have flag 16, else fatal `resume outside an effect clause`. Drop it and every entry above it, drop the recovery points above its `n_try`, pop the clause frame like RETURN and push `v` as the value of the PERFORM. |

**RETURN / RETURN_NIL / UNWRAP_OR_RETURN** (when handler entries exist): after
the recovery points of the returning frame are dropped, entries with
`n_frames >= current frame count` (installed by the returning frame) are
dropped. If the returning function has flag 16 the clause finished without
resume: the top entry must be the clause's kind-1 entry (else fatal `effect
clause returned outside its handler`); with `h` = its owner entry, the VM drops
the recovery points above `h.n_try`, truncates the frames to `h.n_frames` and
the value stack to `h.sp`, pushes the return value, continues at `h.abort_pc`
in frame `h.n_frames - 1` and truncates the entries to the owner's index (the
owner is gone too).

**Runtime errors** unwind to the innermost (or the expired timeout's) recovery
point as before and truncate the handler entries to the count saved in it.

**Steps** (`--max-steps`, `tilvm_run_steps` budgets, snapshots): PERFORM counts
one step when it enters a clause or the default function, exactly like CALL.
HANDLE, END_HANDLE, RESUME and aborts are not steps (like RETURN). Both VMs
count identically (tested).

### Snapshots

* Raw C VM snapshots (`tilvm_snapshot`, `tilvm --snapshot-after N FILE`,
  `--resume FILE`) contain the handler entries and `n_handlers` (they are part
  of `tilvm_state` before the value stack) and the new `tv_try` field; a
  snapshot can be taken at any step, also inside a clause. The struct size
  is part of the snapshot header, so builds with a different
  `TILVM_MAX_HANDLERS` (or TIL 3.0 builds) reject each other's snapshots.
* Portable snapshots (`.tilp`, §2 of the design): write `n_handlers` and, per
  entry, the six fields with `abort_pc` made relative to the code section
  (kind 1: 0) and `sp` / `n_frames` / `n_try` as they are; the extra
  `n_handlers` of every recovery point; the clause pairs are ordinary stack
  values (closure objects and constant strings). Without any handler the
  section is empty.
* Python VM state for the time-travel debugger: `handlers` (list of lists) is
  part of the VM state together with `stack`, `frames`, `recover`.

### C VM configuration

| macro | default | meaning |
|---|---|---|
| `TILVM_MAX_HANDLERS` | 32 | handler entries (16 bytes each in RAM) |
| `TILVM_NO_EFFECTS` | off | leave effect handlers out: `HANDLE` raises `effect handlers are not available in this VM build` (recoverable), `PERFORM` still calls the default function or raises `unhandled effect`; no handler array in RAM |

Effects are included in `TILVM_LEAN` builds. Code size of the effect support
(`-Os`, compared with the same build without it): Cortex-M3 int32 about
+1.3 KB, Cortex-M0 lean int16 about +1.8 KB (part of it is GCC changing its
inlining), AVR about +3 KB. `TILVM_NO_EFFECTS` brings the size back to the
3.0 level.

## 4. Design notes

* **Pairs stay on the value stack.** HANDLE does not copy the clauses into
  the entry: they are already on the stack, so HANDLE never allocates (it
  cannot fail for lack of heap), the GC and snapshots need no special roots,
  and an entry is six integers (easy to serialize portably).
* **Tail resumption only.** A clause cannot do work after `resume`, so the VMs
  never copy or keep a continuation: `resume` is a return to the perform site
  and an abort is a jump to the handle's end. Multi-shot or non-tail
  resumption would need first-class continuations (not supported - and a
  statement after `resume` is rejected rather than silently ignored).
* **Errors in clauses propagate from the perform site** (see Semantics): a
  clause behaves like a dynamically bound function called by `perform`, which
  is what TIL 3.0's `perform` was. Handler *effects* are masked, recovery
  points are not.
* **`return` is rejected in clauses** instead of guessing whether it should
  leave the clause or the enclosing function.
* **Statement ends after block expressions.** An expression ending with an
  indented block (`handle`, `match`, `retry`, `timeout`) now also ends its
  statement; before, a following line starting with `if` was read as the
  ternary `x if c else y` (a parser fix made together with this feature).
