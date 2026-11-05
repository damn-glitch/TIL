# TIL 3.1 — types: gradual inference, E-TYPE and the typed native backend

TIL is dynamically typed: both VMs check types while the program runs. TIL 3.1
adds a **type inference pass** over the whole program. It is used for two
things:

* **E-TYPE** — compile-time errors for type errors that are *definite*
  (the program can never run that line correctly). It never rejects a valid
  program: anything uncertain stays dynamic.
* **The typed native backend** — `til run` / `til build` generate C from the
  inferred types instead of guessing them from names and initializers, and
  **generic functions are compiled once per type combination**
  (monomorphisation: `til_twice__int`, `til_twice__str`).

```til
twice(x)                     # no annotation: a generic function
    return x + x

max_of<T>(a: T, b: T) -> T   # explicit type parameter (same as #[generic: T])
    if a > b
        return a
    return b

main()
    print(twice(21), twice("ab"), twice(1.5))      # 42  abab  3
    print(max_of(3, 7), max_of("pear", "apple"))   # 7   pear
    var names = []                                 # [str], learned from the push
    names.push("Aigerim")
    print(names.map(|n| n.to_upper()))             # ["AIGERIM"]
```

The native backend compiles this program to C with `til_twice__int`,
`til_twice__str`, `til_twice__float`, `til_max_of__int` and `til_max_of__str`,
typed arrays and an inline loop for `map`, and prints exactly what the VMs print.

The implementation is `src/tiltypes.py`. It is optional: `src/til.py` loads it
like the other TIL 3.1 tool modules and keeps the TIL 3.0 behaviour (no E-TYPE,
classic C backend) when it is missing or when `TIL_NO_TYPES=1` is set.

## 1. The type system

| TIL type | written as | notes |
|---|---|---|
| `int`, `float`, `bool`, `str` | `int`, `float`, `bool`, `str` | `i8`…`u64`, `char` are `int`; `f32`/`f64` are `float` (the VMs use 64-bit ints and doubles) |
| `None` | `None` | fits every reference type (strings, arrays, maps, structs, enums with data) and `Option` |
| arrays | `[T]`, `T[]`, `Vec<T>` | one element type; tuples are arrays (as in the VMs) |
| maps | `HashMap<K, V>`; literals `{k: v}` | keys `int`, `str` or `bool` |
| structs | `Point` | references, like in the VMs |
| enums with data | `Shape` | `Shape.Circle(1.0)`; C-like enums are `int` |
| `Option<T>`, `Result<T, E>` | | `Some`, `None`, `Ok`, `Err`, `?` |
| functions | `fn(int) -> int` | lambdas and function names |
| *dynamic* | — | conflicting types: the value's type is only known at run time |

**Inference.** Annotations give types; every unannotated parameter, local and
return value gets a type variable. Unification (union-find) connects them;
operators, indexing, fields, methods and `for` are *deferred constraints*
solved as soon as the operand types are known, with the VMs' exact rules:
`int / int` is `int` (truncating), `int + float` is `float`, `str + x` is `str`
(x is converted), `str * int` repeats, `[a] + [b]` concatenates, comparisons
need two numbers or two strings, `and`/`or` give `bool`.

Values **flow** one way into variables, array elements, map entries, fields and
parameters: `let i = 1; let mix = [i, "s"]` makes `mix` a `[dynamic]` array but
leaves `i` an `int`. Containers learn their element types from their use
(`var xs = []` followed by `xs.push("a")` is `[str]`); map *lookups* never
change the key type, only stores do.

**Generics.** Functions are generalised per strongly connected component of
the call graph, together with their open constraints (`twice(x)` keeps
"`x + x` decides the result"). Each call instantiates the function afresh, so
`twice(21)` is an `int` and `twice("ab")` a `str`. Unannotated parameters make a
function generic; so do explicit type parameters, written `#[generic: T]` or
`name<T, U>(...)` (new syntax, the bytecode compiler ignores it like any
annotation).

**Gradual.** A conflict is never an error: `var x = 1` followed by `x = "a"` makes
`x` *dynamic*, and so does `[1, "a"]` or `1 if c else "one"`. The program is
valid (it runs on the VMs); it just cannot be compiled to native code with
static types, so `til run` uses the VM for it. One special case keeps a common
pattern native: a `retry`/`timeout` whose body is a number and whose `else` is a
string (`retry(3) compute() else "failed: " + error`) has the type
"*str or int*": such a value can be printed, interpolated and concatenated with
strings (exactly what the VMs print), nothing else.

## 2. E-TYPE

E-TYPE errors are reported only when a type error is **definite**. They are
computed in a separate pass in which every unannotated parameter is *dynamic*,
so an error never depends on how a generic function happens to be called.

| reported | example |
|---|---|
| a value that can never satisfy an annotation | `let x: int = "a"`, `let s: str = 5`, `let p: Point = Circle {…}`, `f(s: str)` called as `f(5)`, `-> int` returning `"x"`, `var n: int = 0` then `n = "s"` |
| unknown field or method of a known struct | `p.z`, `p.nope()`, `p.w = 2` |
| arithmetic between types that never mix | `1 - "a"`, `"a" * "b"`, `true + 1` |

```
error[E-TYPE]: 'x' is int, but the value is the string 'a', which is not a number
 --> demo.til:3:5
  3 |     let x: int = "a"
    |     ^
  = hint: convert the value explicitly (str(x), int(x), float(x)) or change the annotation
```

Not reported (the VMs decide at run time, and tests rely on that):

* anything that depends on an unannotated (generic) parameter;
* code inside `retry`/`timeout` bodies — failing on purpose is how recoverable
  errors are produced;
* numbers converted by the VMs: `let x: float = 1`, `let y: int = 2.5`,
  `let z: int = "12"`, `f(x: int)` called with `"12"` or `true`; a string
  literal that is not a number is an error at declarations, returns and
  defaults, but not as a call argument (the callee's conversion fails at run
  time);
* `None` for references (`let s: str = None`, `var p: Point = None`);
* a number for a `bool` annotation (`f(b: bool)` called as `f(5)`,
  `b: bool = 0`, `-> bool` returning an int): the VMs keep the number as it is
  (no conversion and no error), and programs rely on that — the soundness tests
  of `til prove` pass ints to `bool` parameters on purpose. The native backend
  runs such programs on the VM;
* effects (docs/TIL_EFFECTS.md): a `perform E(...)` may be handled by any
  dynamically enclosing clause for `E`, so when some `handle ... with` in the
  program has a clause `E`, the arguments and the result of `perform E` are
  *dynamic* (the default function's annotations do not apply to them), and so
  is the value of a `handle` with clauses (the body's value, or that of a
  clause that ends without `resume()`). Clause bodies are inferred (they are
  closures and may assign variables of the enclosing function) without
  reporting. A `perform` of an effect that no clause anywhere handles always
  calls the function `E` and is checked like a call;
* comparisons, indexing, iteration, calling a non-function, the arguments of
  builtin natives (`read_file(5)`), methods of builtin types — the VMs report
  these at run time with their own messages.

E-TYPE is checked by `til check`, by every compilation to bytecode
(`til run --vm`, `til build --target vm`, `--cvm`) and by the native backend.
`--no-types` (or `TIL_NO_TYPES=1`) switches the pass off (the option only for
that command: an in-process `til.main()` call restores the environment). The
errors are localized like the other diagnostics (`--lang ru|kk`).

## 3. The typed native backend

### Representation

| TIL | C |
|---|---|
| `int`, `float`, `bool` | `int64_t` (wraps around: `-fwrapv`), `double`, `bool` |
| `str` | `const char*` (`NULL` is `None`) |
| arrays | `TIL_Arr*`: growable, reference semantics, 8-byte slots (`int`, `float`, `str`, references) |
| maps | `TIL_Map*`: insertion-ordered hash map (iteration and printing in insertion order like the VMs) |
| structs | `Point*` on the heap — **reference semantics** like the VMs (`let b = a; b.x = 5` changes `a`) |
| enums with data | tagged struct with payload slots, one constructor per variant |
| `Option<T>` / `Result<T, E>` | small structs passed by value |
| generic functions | one C function per instance: `til_f__int`, `til_f__str`, `til_max_of__float`, methods `Box_add__int` |
| lambdas | inline in `map`/`filter`/`reduce`/`each`/`any`/`all`; bound to a name and called: a static C function that reads the captured variables at the call |
| closures as values | `TIL_Clo*`: function pointer + copies of the captured variables (returned, stored, passed on; named functions too); printed `<fn …>`, compared by identity |

### Identical behaviour

Native output (stdout and exit code) is byte-identical to the reference VM.
In particular the native code reproduces:

* printing: floats as `%g` with `nan`/`inf`/`-0`, arrays/maps/structs/enums/
  `Some(…)`/`Ok(…)`/`Err(…)` exactly like the VMs (strings quoted inside
  containers), f-string format specs (`{x:.2f}`, `{n:x}`, `{s:>8}`, …; `{x:d}`,
  `{x:x}`, `{x:b}` of a float beyond the 64-bit range print the exact integer,
  like Python's `int(x)` on the reference VM);
* `print(a, b, c)` evaluates all arguments before printing any; operands and
  arguments are evaluated left to right (C leaves the order unspecified);
* integer division/modulo truncating toward zero, `division by zero`,
  `2 ** -1 == 0`, wrapping overflow, shifts, `int(x)` of floats, Python-exact
  comparisons between ints and floats (`2**53 + 1 != 2.0**53`), `floor()` and
  `ceil()` that never give `-0` (the reference VM computes `float(math.floor(x))`:
  `ceil(-0.5)` prints `0`);
* the bytecode compiler's constant folding where it differs from run time:
  integer literals beyond 64 bits (`18446744073709551616 > 0` is folded with
  the exact value, a variable holding that literal is wrapped to `0`). Folding
  uses a global constant only where no local, parameter, loop variable or
  lambda parameter of the same name is visible (`area(size: int)` computes
  `size * size` with its parameter even when a global `size` exists), exactly
  like `BytecodeCompiler.names_a_constant`; `true == 1` is `false` folded or
  not;
* where the VMs produce `None` implicitly: `var x` without a value, a
  declaration or struct field that is not given and whose annotation is not a
  primitive type (`var p: Point`, `var xs: [int]`; primitive annotations start
  as `0`, `0.0`, `""`, `false`), a global without a value, a string, array,
  struct … global read by an initialiser before its own line has run, and a
  function that ends (or uses a bare `return`) without returning a value — also
  one declared `-> str`;
* run-time error messages (`index out of bounds: index 5, length 2`,
  `key not found: q`, `invalid integer: "x1"`, `pop() from empty array`,
  `unwrap() called on None`, contract and assertion messages,
  `intent '…' failed: …`), so `error` in a `retry`/`timeout` `else` is the same;
* `retry`/`timeout`/`intent` semantics, contracts (`#[requires]`, `#[ensures]`),
  `?` propagation, program arguments `args()`, `read_file`/`write_file`/
  `file_exists`, `str_bytes`, `input()`, `random()`/`seed()` (same generator).

### What runs on the VM instead

When a program needs something the native backend cannot express, it raises
`NativeUnsupported`; `til run` then runs the program on the VM (with `-v` it
says why) and `til build` reports the reason. Never a silent wrong result.

* values that are *dynamic* (mixed types, see §1), e.g. heterogeneous arrays
  used for dynamic dispatch over trait objects;
* a closure used as a value that captures a variable which changes after the
  closure is created (the VMs share captured variables; a copy would differ),
  or a loop variable;
* a lambda bound to a name that is called (or passed on) where a variable it
  captured is shadowed by another one of the same name;
* a `None` that a native `int`, `float`, `bool` or C-like enum cannot hold:
  `var x` used as an `int` before it is assigned, a global `var n: int`
  without a value, a struct field of an enum type that is not given, a
  function declared `-> int` that can end without returning a value (the VMs
  return `None` there), and a computed `int`/`float`/`bool` global that is read
  before its line has run (`let A = B == NULL` followed by `let B = 5 + 0`;
  also through a function the initialiser calls, a contract, or — in script
  mode — a top-level statement before the line). A `let`/`const` global with a
  literal value is set before anything runs (the VMs inline it, the C code
  initialises it statically); every other global, also a `var` with a literal
  value, is set in order at start-up, natively too;
* a number passed to a `bool` parameter or stored in a `bool` variable (the VMs
  keep the number, a native `bool` would not);
* types that would be infinite, e.g. polymorphic recursion (`nest(v)` calling
  `nest([v])`) or an array pushed into itself: they are *dynamic*;
* `bytes_to_str()` and strings containing NUL bytes (native strings are C
  strings); graphics, keys and `energy_mode()` natives; `handle ... with` /
  `resume` (effect handlers); `perform E(...)` when no function `E` exists
  (the default handler);
* generic functions used with more than 32 type combinations.

A function that is never called (dead code) does not force a fallback.

Very deep recursion overflows the C stack (the process crashes) where the VMs
stop with "stack overflow (recursion too deep)" after 2000 frames.

A few cases are **explicit run-time errors** (a message on stderr, exit
code 1) instead of a fallback, because they depend on values only known while
the program runs:

* the program reaches a `None` that a native `int`/`float`/`bool` cannot hold:
  `first()`/`last()` of an empty `[int]` array, and a `match` used as a value
  whose patterns cover nothing for the input (the VMs give `None`);
* a NUL byte would enter a string (native strings are C strings): `chr(0)`,
  a line read by `input()` that contains a NUL byte, `read_file()` of a file
  with NUL bytes (the VMs handle these strings);
* `sort()` of a float array that contains `nan`: where `nan` ends up depends on
  the sorting algorithm (the reference VM uses the host Python's sort, whose
  details changed between Python versions; the C VM an insertion sort).

They never print a wrong value; the VMs run these programs to the end.

## 4. Coverage before and after

`tools/native_coverage.py` measures which programs compile natively and whether
the native executable prints exactly what the reference VM prints.

**Corpus.** Every TIL program written in `tests/test_compiler.py`,
`tests/test_v3.py` and `tests/test_foundation.py`, plus `examples/**/*.til`. The
test files are parsed with Python's `ast` module (static, no test is run):
every string literal is a candidate, and so is every `+` concatenation of
literals and of names bound to string literals in the same function or at
module level. A candidate is a program when it parses, passes the verifier,
compiles to bytecode and contains at least one call (expected outputs such as
`"3\n2"` contain no call; deliberately invalid programs do not compile).
Programs are de-duplicated by text; a program's id is its SHA-1.

**Measure.** Reference: the Python VM (no arguments, empty stdin, 5,000,000
steps). Native: the same driver as `til run` (`-O0` in the tool), empty stdin,
timeout. Programs the VM cannot finish (step budget, recursion limit) are not
compared. A native program that stops with one of the explicit run-time errors
of §3 (exit code 1, a message naming the native backend, the same output up to
there) is counted as `native-refused`, not as a wrong output.

```
python3 tools/native_coverage.py                       # summary
python3 tools/native_coverage.py --details --reasons    # per program / why
python3 tools/native_coverage.py --compare tools/native_coverage_baseline.json
```

| 233 programs of the TIL 3.0 corpus (214 from the tests, 19 examples), 231 compared | before (TIL 3.0 backend) | after (typed backend) |
|---|---|---|
| compiled natively | 205 (88.7%) | **225 (97.4%)** |
| … output identical to the VM | 203 | **225** |
| … output **different** from the VM | 2 | **0** |
| fall back to the VM | 26 | 6 |

The TIL 3.1 tree adds four examples (237 programs, 235 compared): 226 compiled
natively, all identical to the VM; `11_graphics`, `12_effects` and
`14_energy` use VM-only features (graphics, effect handlers, `energy_mode()`)
and run on the VM, `13_proofs` is native.

The baseline per-program results are in `tools/native_coverage_baseline.json`:
the TIL 3.0 backend (commit `6f4d00b`, before this work) measured with this
tool, so both columns are counted the same way (re-measure it by running the
tool in a checkout of that commit). No program that compiled before
regresses; `--compare` fails on a regression and on any native output that
differs from the VM. The 2 former mismatches were silent wrong results of the
old backend (a `retry` value that is an int or a string, and an `intent` error
message); a third program, which recurses into `main` until the VM's recursion
limit, crashed natively with the old backend and is not compared (VM limit). Newly native: dynamic
arrays, maps, higher-order methods, string methods, Option/Result matching,
enums with data, closures as values, script-mode globals, ranges,
`args()`/files, programs without `main()`.

The 6 remaining fallbacks: heterogeneous arrays (3: dynamic dispatch over two
struct types in `tests/test_v3.py` and `examples/09_v3_features.til`, and
`[1, [2, 3]]`) and `bytes_to_str` (3).

The old backend also compiled programs to wrong code that the corpus did not
exercise, e.g. `print([1, 2, 3])` printed a pointer, `min(1.5, 2.5)` printed `1`,
`abs(-5) / 2` printed `2.5`, `"é"[0]` was negative, string `<` compared
pointers; all of these now match the VMs.

**The whole TIL 3.1 corpus.** Every TIL program in all test files of the
integrated tree (`tests/test_*.py`: all 14 features) plus the examples,
measured the same way (`python3 tools/native_coverage.py tests/test_*.py`;
13 programs that call `sleep()` were left out, because the in-process VM really
sleeps): 826 programs, 807 compared, **580 compiled natively: 578 with output
identical to the VM, 0 different**, and 2 that stop with one of the explicit
run-time errors above (`chr(0)` and `sort()` of nan in `tests/test_types.py`,
status `native-refused`). The 227 fallbacks: effect handlers 85, VM-only
natives (graphics, keys, `energy_mode()`) 58, dynamic values 44, and 40 for
the other reasons listed in §3 (a global read before it is set 9, a `None`
that a native number cannot hold 9, closures capturing a variable that
changes 5, `perform` of an effect without a default function 5, `bytes_to_str`
4, ...). 11 programs hit a VM limit; the 8 programs the VM cannot compile are
the deliberately wrong E-TYPE examples of `tests/test_types.py`. Over the same
corpus (and the self-hosted compiler `selfhost/tilc.til`, 8300 lines,
inferred in about 0.5 s) the inference never
fails and reports E-TYPE only for deliberately wrong programs (those of
`tests/test_types.py` and the type-mismatch test of `tests/test_compiler.py`);
no program of `tests/test_effects*.py` gets one.

## 5. Implementation notes

`src/tiltypes.py`:

* **Phase A** — `TypeInfer.run()`: declarations, call-graph SCCs (callees
  first), per-SCC inference with deferred constraints and wake-up lists,
  generalisation (open constraints become part of the scheme; variables tied to
  globals or struct fields through open constraints stay monomorphic). Only
  the open constraints connected to the function's signature are kept in the
  scheme: the others concern variables private to the body, and copying them
  at every call made schemes grow with the call graph (the 6300-line
  self-hosted compiler took about 20 s instead of 0.4 s). Unification has an occurs check:
  a type that would contain itself is *dynamic*.
* **Phase E** — definite errors: every body again with unannotated parameters
  *dynamic*; generic callees are opaque (their annotated parameters are still
  checked).
* **Phase B** — `TypeInfer.specialize(f, param_types)`: one function instance
  with concrete parameter types; every expression gets a settled type, every
  call its target instance. Constraints that stay open default to *unknown*
  (unconstrained: any representation).
* `TypedCGen` (a `CCodeGenerator` subclass) builds the instance worklist from
  `main`, the global initialisers and every non-generic function, generates
  one C function per instance and the helpers each type needs (printing,
  equality, Option/Result structs).

`src/til.py` hooks: `NativeUnsupported(CompileError)`, `types_module()`,
`type_errors()` (used by `compile_to_bytecode` and `til check`), the typed
backend in `TILCompiler.compile`, the VM fallback in `compile_to_executable`,
`-fwrapv` for native builds (plus, for C from the typed backend built with gcc
or clang, `-Werror=int-conversion -Werror=incompatible-pointer-types`: if the
generated C ever mixed up an integer and a pointer, the build fails and
`til run` uses the VM instead of running wrong code), `--no-types`, and the
`name<T>(...)` syntax in the parser. `TIL_DEBUG_TYPES=1` re-raises internal errors of the typed backend
(normally they make `til run` use the VM). Python code must stay 3.9
compatible (`vermin -t=3.9-`).

Tests: `tests/test_types.py` (inference, E-TYPE, monomorphisation, native ==
VM on formerly unsupported programs, fallbacks, coverage; the full measurement
with `TIL_TEST_SLOW=1`).

### Known differences of the C VM

Not part of the native backend, found while testing it: the C VM compares an
`int` with a `float` after converting the int to a double
(`9007199254740993 == 9007199254740992.0` is `true` there, `false` on the
reference VM and natively), truncates very long fixed-point formats
(`{1e100:.1f}`), gives `-0` for `ceil(-0.5)` and `floor(-0.0)` (the reference VM
and the native code give `0`), and formats a float beyond the 64-bit range with
`{x:d}`/`{x:x}`/`{x:o}`/`{x:b}` after wrapping it to an int64 (the reference VM
prints the exact integer). The reference VM itself stops with an internal error
(`cannot convert float infinity to integer`, not a TIL run-time error that
`retry` could catch) for `{x:d}` of `inf`/`nan`; the C VM prints
`-9223372036854775808`.
