# Proving contracts for all inputs (`til prove`)

`til run` checks contracts while a program runs: a broken `#[requires]` or
`#[ensures]` stops it, but only for the inputs you happen to try.
`til prove` checks them **for every possible input** with the
[Z3](https://github.com/Z3Prover/z3) SMT solver, and when a contract can break
it shows an input that breaks it - after running the function with that input
to make sure.

```
pip install z3-solver          # once
til prove examples/13_proofs.til
```

```til
#[ensures: result > x]
inc(x: int) -> int
    return x + 1
```

```
$ til prove inc.til
Proving inc.til with Z3 4.x (64-bit integers, 10000 ms per query)
  inc: counterexample: x = 9223372036854775807 (#[ensures: result > x] fails, line 1)
error[E-PROVE]: 'inc': #[ensures: result > x] fails for x = 9223372036854775807
 --> inc.til:1:1
  1 | #[ensures: result > x]
    | ^
  2 | inc(x: int) -> int
  = hint: with these inputs result = -9223372036854775808; confirmed on the TIL VM (running it fails with: contract violation: inc ensures result > x)
Result: 1 counterexample
```

Ints are 64-bit and wrap around, so `x + 1 > x` is false for the largest int.
Add `#[requires: x < 9223372036854775807]` and `inc` is proved.

## Commands

| command | proves | without Z3 |
|---|---|---|
| `til prove prog.til` | every function with `#[requires]`, `#[ensures]`, `assert` or `invariant`, and every function that calls a function with `#[requires]` | explains `pip install z3-solver`, exit 1 |
| `til check --prove prog.til` | the same, as part of the checks | error `E-PROVE`, exit 1 |
| `til check prog.til` | only `#[level: 4]` functions (level 4 = "formal") | warning on stderr, does not fail |

Only functions defined in the file itself are proved; imported functions are
proved when their own file is. Their contracts are used at call sites either way.

`til prove` options:

* `--timeout MS` - time limit of each solver query (default 10000). One
  condition may take up to three queries (bit-vector, integer, bit-vector with
  another strategy).
* `--function NAME` - prove only this function (also one without contracts;
  `Type.method` or `method`).
* `--allow-unsupported` - functions the prover cannot handle do not make the
  exit code nonzero.
* `-v` - list every proof condition with its result.

Exit code: 0 when everything in scope is proved, 1 otherwise (counterexample,
not proved, unknown, unsupported), 2 for a wrong command line.

Per function the result is one of:

* `proved` - holds for all inputs (with a note when the `#[requires]` can never
  hold: then every call fails and the proof says nothing);
* `counterexample: x = ..., y = ...` - these inputs break the named condition;
  the counterexample has been **run on the reference VM** and fails there exactly
  as reported (same error, same function and line);
* `not proved: ...` - the solver found a candidate, but it relies on an
  abstraction (a loop without a strong enough invariant, a callee known only
  through its `#[ensures]`, ...) and does not fail when run; the hint says what
  it relies on;
* `unknown (timeout)` - Z3 gave up within the time limit;
* `unsupported: <reason>` - the function uses something the prover does not
  model; nothing about it is claimed.

Every result other than `proved` is also reported as an error with code
`E-PROVE` at the line of the condition (the `#[requires]`/`#[ensures]`
attribute, the `assert`, the call, the division).

## What is proved

For each function in scope and **every** combination of `int` and `bool`
parameter values that satisfies its `#[requires]`:

1. each `#[ensures]` holds when the function returns (`result` is the returned
   value, other names are read at the return, like the VM does);
2. each `assert` and `invariant` statement holds;
3. each call satisfies the called function's `#[requires]`;
4. no `/` or `%` divides by zero (also inside contracts);
5. each loop invariant holds when the loop is reached and after every
   iteration.

This is *partial correctness*: it is about runs that reach these points.
Termination, stack overflows and runtime errors the prover does not model
(index out of bounds, errors inside callees it does not see) are not covered.
Integer overflow is not an error in TIL - it wraps around - and is not reported
by itself; it shows up when it breaks a contract.

## The integer model

The prover uses exactly the semantics of both TIL VMs (`TILVM` in
`src/til.py`, `vm/tilvm.c`):

| operation | semantics |
|---|---|
| `+ - *`, unary `-` | 64-bit two's complement, wrap around (`MAX + 1 == MIN`, `-MIN == MIN`) |
| `/` | truncates toward zero like C (`-7 / 2 == -3`); `MIN / -1 == MIN`; divisor 0 is an error |
| `%` | sign of the dividend (`-7 % 2 == -1`, `7 % -2 == 1`); `MIN % -1 == 0`; divisor 0 is an error |
| `<< >>` | shift amount taken modulo 64 (`x << 64 == x`); `>>` is arithmetic |
| `& \| ^ ~` | bitwise on 64 bits |
| `**` | wrapping power; negative exponents give 1, -1 or 0 like the VMs |
| `abs(x)` | `abs(MIN) == MIN` |
| `min`, `max` | on ints, any number of arguments |
| `int(x)`, `x as int` | bool -> 0/1, nil -> 0 |
| `bool(x)`, conditions | ints are true when nonzero, nil is false |
| `==` | values of different types are never equal (`1 == true` is false) |
| comparisons | signed |

Constant subexpressions are folded exactly like the compiler folds them
(`BytecodeCompiler.try_fold`), because that is what the bytecode computes.

Integers are Z3 bit-vectors of 64 bits, so the model is exact. Nonlinear
arithmetic (`i * (i + 1)`, `x % y`) is hard for bit-vectors; for those the
prover also tries mathematical integers. That is only allowed when it proves
that **no** `+ - * **` in the function overflows (with the overflow
conditions as extra proof obligations); then both models agree on every run
and integer arithmetic is used for the rest. This is what makes loop invariants
like `2 * s == i * (i + 1)` provable in milliseconds.

`-` (unary), `abs` and `/` overflow only for `MIN`; the integer model keeps
their exact VM result there.

## How code is handled

* **Parameters**: `int` (or `i64`) and `bool`. Other parameters may be used
  freely, but a contract or a condition that depends on them makes the
  function unsupported. Parameters without a type are treated the same way:
  annotate them.
* **Locals**: `let`, `var`, `const`, assignments (`=`, `+=`, ...), shadowing,
  block scopes.
* **Control flow**: `if`/`elif`/`else`, `x if c else y`, `and`/`or` (the right
  side is only checked when it is evaluated), early `return`, `break`,
  `continue`.
* **Loops**: `invariant` statements at the very top of a loop body are its loop
  invariants. They must hold when the loop is reached and after each iteration
  (also after `continue`); after the loop they hold and the condition is false.
  A loop without invariants is handled soundly: the variables it assigns are
  unknown afterwards (and inside it), so many facts can then not be proved.
  `for i in a..b` / `a..=b` and `for i in range(a, b)` add the implicit bound
  `a <= i <= max(a, b)` automatically (unless the body assigns `i`).
  The VM checks invariant statements only when the body runs; the prover also
  requires them before a loop that does not run at all (standard loop
  invariant rules) - the error says so.
* **Calls to functions with contracts**: the arguments must satisfy the
  callee's `#[requires]`; the result is unknown except for what the callee's
  `#[ensures]` says. This is sound whether or not the callee itself is proved,
  because the VM checks `#[ensures]` before every return. A callee's
  `#[ensures]` that reads a parameter the callee reassigns, or a local of the
  callee, is not used (it does not mean what the caller would think).
  Recursion works the same way (by induction).
* **Calls to functions without contracts**: a function whose body is just
  `return <expression>` is inlined (exactly, including its divisions); any
  other call has an unknown result of the declared return type (`-> int`,
  `-> bool`). Add `#[ensures]` to the callee to prove more.
* **Builtins**: `abs`, `min`, `max`, `int`, `bool` are modeled; `print` /
  `println` have no effect on the proof; other builtins return unknown values.
* **Constants and enums**: `const`/`let` globals with constant values and
  C-like enum values are known; mutable (`var`) globals are unknown values.
* **Static methods** (`Type.f(...)`) and module functions (`m.f(...)`) are
  ordinary calls; methods with `self` are unsupported.
* Strings, floats, arrays, maps and structs may flow through a function (for
  example to `print`), but contracts, assertions and divisors that depend on
  them make the function unsupported. Also unsupported: closures, `match`,
  `retry`/`timeout`, `?`, effects, list comprehensions, method calls to
  methods with `#[requires]`.

## Counterexamples are real

When Z3 finds inputs that break a condition, the prover builds a small
program that calls the function with them and runs it on the reference VM
(output discarded, no input, files/screen/sleep blocked, at most 300000
steps). The result is reported as a `counterexample` only when the run fails
with the predicted error in the predicted place.

When the candidate comes from an abstraction (a loop invariant that is too
weak, a callee's `#[ensures]`) and does not fail when run, the prover tries
edge values (0, 1, -1, MIN, MAX, ...) on the real function; a run that breaks
the condition is a real counterexample. Otherwise the result is `not proved`,
with the candidate and what it relies on:

```
error[E-PROVE]: 'f': cannot prove #[ensures: result >= 1]
  = hint: the solver found a = 0 (result = 0), but running it with these inputs does
    not fail; it relies on the result of 'Point.grow' (only its #[ensures] is known);
    the #[ensures] of the called function may be too weak
```

## Examples

`examples/13_proofs.til` (proved completely; `til run` prints the same
natively and on both VMs): `clamp`, an overflow-free `midpoint`, integer
division with its defining property, a caller (`pages`) proved through the
callee's contract, `sum_to` and `gcd` with loop invariants (`#[level: 4]`, so
plain `til check` proves them), and `square_by_odds` with a `for` loop.

Things the prover found while that example was written:

* `midpoint(a, b) = a + (b - a) / 2` with only `#[requires: a <= b]` breaks
  for `a = -6425787866654150400, b = 8024048470302220032`: `b - a` does not
  fit in 64 bits. The fix is `#[requires: 0 <= a and a <= b]`.
* `#[ensures: result * b + a % b == a]` does not let a caller conclude
  `result >= 0`: multiplication modulo 2^64 is not invertible for even `b`
  (`result = -4611686018427387905, b = 2` satisfies it). A bound
  (`result >= 0 and result <= a`) makes it useful.
* `fact(n) = n * fact(n - 1)` with `#[ensures: result >= 1]` breaks for
  `n = 32`: 32! wraps around to a value below 1.

## Limitations

* Proofs are about the VM semantics with 64-bit ints: the desktop VMs, and
  the native backend as far as it matches them. C VM builds with
  `TILVM_INT_BITS=16` or `32` (calculators, microcontrollers) have narrower
  ints; proofs do not cover them. Types narrower than 64 bits (`i8`, `u8`,
  `i32`, ...) are treated as unknown values because native code narrows them.
* The native C backend compiles `int` arithmetic to plain C, where signed
  overflow and `MIN / -1` are undefined behaviour; a C compiler may then
  disagree with the VMs (for example fold `x + 1 > x` to true). Proved
  properties hold for native builds only when they are compiled with
  wrap-around semantics (`-fwrapv`).
* `bool` parameters are not converted by the VM; the proof assumes callers
  pass bools.
* `#[ensures]` of a function without a return type is checked by the VM only at
  `return <value>`; the prover follows the VM.
* Calls through function values (closures, functions passed as arguments) are
  not checked; passing a function that has `#[requires]` as a value makes the
  caller unsupported.
* Nonlinear bit-vector problems that do not fit the integer model (they need
  wrap-around) can time out; the result is then `unknown`.
* Floats, strings, arrays, maps, structs and methods with `self` are not
  modeled in contracts.

## For tool authors

`src/tilprove.py`:

* `main(argv) -> int` - `til prove`;
* `check_program(program, source, filename, explicit) -> List[str]` - used by
  `til check` (via `run_proofs`): formatted `E-PROVE` errors, empty when
  everything in scope is proved;
* `prove_source(source, filename, explicit=True, only=None, timeout_ms=...)`
  and `prove_program(...)` - one `FunctionReport` per function (`status`,
  `failures` with `inputs`, `extras`, `confirmed`, `outcome`, `depends`).

Tests: `tests/test_prove.py` (skipped when z3-solver is missing, except the
tests of the no-Z3 behaviour). Every counterexample in the tests is run again on
the reference VM and on the C VM. `TIL_TEST_SLOW=1` runs more random programs.
