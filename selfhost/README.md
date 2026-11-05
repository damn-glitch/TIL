# tilc — the TIL compiler written in TIL

`selfhost/tilc.til` is a compiler for TIL written in TIL. It compiles a TIL
program to `.tbc` bytecode (version 2) that runs on every TIL VM: the
reference VM in `src/til.py`, the portable C VM `vm/tilvm.c`, and the C VM on
calculators and microcontrollers. Because tilc itself is a `.tbc` program, a
calculator that runs the TIL VM can now compile TIL programs on its own.

tilc is a port of the Python front end and bytecode compiler (`Lexer`,
`Parser`, `SemanticChecker`, `BytecodeCompiler`, `TBCModule.to_bytes` in
`src/til.py`) and produces **the same bytes** as `til build --target vm` for
every program of the test corpus and every example (see *Results*).

## Usage

```sh
# run tilc on the reference (Python) VM
til run --vm selfhost/tilc.til input.til output.tbc

# build tilc once, then run it on any TIL VM
til build --target vm selfhost/tilc.til -o tilc.tbc
vm/tilvm tilc.tbc input.til output.tbc        # tilvm built with a >= 1 MB heap
vm/tilvm output.tbc                           # run the result

# bootstrap: tilc compiles itself
vm/tilvm tilc.tbc selfhost/tilc.til tilc2.tbc     # tilc2.tbc == tilc.tbc
```

* Success: no output, exit code 0, `output.tbc` written.
* Errors: one line `tilc: FILE:LINE: error: MESSAGE`, exit code 1. The
  messages are those of the Python compiler (`unknown name 'x'`,
  `cannot assign twice to immutable variable 'x'`, `'f' expects 1 to 2
  argument(s), got 3`, `Unexpected token: DEDENT`, E-EFFECT / E-CAP /
  E-CONTRACT texts, ...) at the same line; tilc stops at the first error.
* Constructs it cannot compile: `tilc: FILE:LINE: unsupported: WHAT`,
  exit code 1.
* Wrong command line: usage line, exit code 2.

The file is read with `read_file` and written with `write_file`; the program
arguments come from `args()`. A VM build without stdio needs the
`vm->host.read_file` / `write_file` hooks (e.g. the browser playground).

## The language tilc accepts ("TIL Core")

TIL Core is the whole language as compiled by `BytecodeCompiler`, except the
items listed under *Not supported*:

* **Program structure**: functions `name(params) -> T` / `fn name(...)` /
  `pub`, bodies as indented blocks, `{ ... }` or `=> expression`; parameters
  with optional types (`int`/`float` annotations convert arguments, `-> int` /
  `-> float` convert results, exactly like the Python compiler) and default
  values; `let` / `var` / `const` (locals and globals); script mode (top-level
  statements, `main()` synthesis, globals initialised at their position,
  literal `let`/`const` globals inlined); `#[...]` attributes (unknown ones
  are ignored, as in the Python compiler); `type` aliases; `cap` declarations.
* **Values**: int (64-bit; decimal, `0x`, `0b`, `0o`, `_` separators), float
  (exact IEEE 754 conversion of the literal), str with every escape (`\n \t \r
  \0 \\ \" \' \{ \} \a \b \e \xNN \u{...}`), char literals `'a'` (ints), bool,
  `None`, f-strings (with format specs `{x:.2f}`, `{n:>5}`, `{n:x}` ...),
  arrays, maps, tuples, `Vec<T>.new()`, `HashMap<K,V>.new()`.
* **Operators**: all of them with the same precedence and semantics: `+ - * /
  % **`, comparisons (chained: `0 <= i < n`), `and or not`, `& | ^ ~ << >>`,
  ranges `..` / `..=`, ternary `a if c else b`, pipes `x |> f(y)`, casts `as`,
  `?` (Option/Result propagation), `&x` / `*x`; compound assignment `+= -= *=
  /= %=` on variables, indexes and fields. Constant expressions are folded
  exactly like `const_eval` (64-bit wrap-around integers, strings, booleans,
  and float arithmetic rounded as IEEE 754 doubles).
* **Statements**: `if / elif / else if / else` (also one-line bodies),
  `while`, `loop`, `for x in a..b` / `a..=b` / arrays / strings / maps,
  `break`, `continue`, `return`, `assert` / `invariant` with messages,
  `match` (literal, `_`, binding, `None`, ranges, `a | b`, guards, `Some(x)` /
  `Ok(x)` / `Err(e)`, enum variants with data, struct patterns, tuple/array
  patterns; as statement and as expression).
* **Types**: structs (field types, defaults, literals `P { x: 1 }`, positional
  `P(1, 2)`), `impl` blocks (methods with `self`, static functions
  `P.new()`), traits with default methods and `impl Trait for Type`, C-like
  enums (`Color.Red`, explicit values) and algebraic enums (`Shape.Circle(r)`).
* **Functions as values**: lambdas `|x| x + 1` / `|x: int| -> int { ... }`
  with captured variables (cells and upvalues, using the Python compiler's
  capture analysis), function names as values, `map/filter/reduce/each/any/all`
  (compiled from the same prelude source), list comprehensions
  `[e for x in it if cond]`.
* **Recovery and checks**: `retry(n)` / `timeout(ms)` with `else` and
  `error`, `intent "..."` blocks; `#[requires]` / `#[ensures]` (run-time
  checks and the compile-time E-CONTRACT check with constant arguments);
  `#[pure]` / `#[effects: ...]` / `#[cap: ...]` (the SemanticChecker's effect
  inference and capability rules, E-EFFECT / E-CAP).
* **Modules**: `import m`, `import m as x`, `from m import a, b`,
  `from m import *`, packages (`a.b` -> `a/b.til` or `a/b/mod.til`),
  qualified calls `m.f()`, import cycles; each module keeps its own keyword
  languages.
* **Languages**: English and Kazakh keywords always, `#[lang: kk-latn]` and
  `#[lang: ru]` pragmas, localized builtin names (`басып_шығару`, `печать`, ...),
  Unicode identifiers.

### Not supported (reported as `unsupported`)

| construct | why |
|---|---|
| `handle ... with ...`, `perform E(...)` | effects with resumption are being redesigned for TIL 3.1 (§8 of the design, new opcodes); tilc will follow once that lands |
| `import m` when `m` is not next to the importing file nor in the current directory | a TIL program cannot see `TIL_PATH` or where the standard library is installed |

### Known differences from the Python compiler

They never change the behaviour of a valid program unless stated:

* A float constant expression whose value is NaN (`1e400 - 1e400`) is not
  folded (its bit pattern in the Python compiler depends on the CPU); the
  program computes it at run time and prints `nan` like the Python-compiled
  one. Everything else about a NaN is folded as in Python: comparisons and
  truthiness (`(1e400 - 1e400) != 0.0` is `true`), and a NaN argument is
  checked against `#[requires]` at compile time (E-CONTRACT `x=nan`).
* Unicode: identifiers, numbers and `repr()` follow the tables of Unicode
  15.1 (Python 3.13), generated by `selfhost/unicode_tables.py`. The Python
  compiler uses the Unicode database of the Python running it, so on an older
  Python it rejects letters and digits assigned after its Unicode version
  that tilc accepts, and shows them escaped in E-CONTRACT messages (Python
  3.12: 627 code points differ, 3.9: 5,945, all unassigned in that Python).
  Every character older Pythons know is classified the same (checked for
  3.9, 3.11 and 3.12).
* Lone surrogates (`\u{D800}` to `\u{DFFF}`) are an invalid `\u{...}`
  escape for tilc. The Python compiler accepts them in the lexer and fails
  when it writes such a string (`'utf-8' codec can't encode ...`), so both
  reject a program unless the string is folded away (`"\u{D800}" == "a"`).
* Diagnostics: only the first error is reported. When a program has several
  errors of different kinds, tilc may report another one first: it checks
  effects and capabilities right after parsing, but arity and constant
  contracts while it compiles each function (the Python compiler runs all
  SemanticChecker checks, arity first, before compiling). In a fuzz run of
  ~1,800 mutated corpus programs that the Python compiler rejects, tilc
  reported the same first message at the same line for 99.7% of them; every
  difference was of this kind or the missing-module case below. No mutant
  (nor any of ~1,500 truncated corpus programs) crashed tilc, and apart from
  the missing-module case (and the Unicode-version and size limits in this
  list) every program one compiler accepted the other accepted too, with
  identical bytes.
* Limits of nesting (C VM with its default 1000 call frames; the Python
  compiler is limited by Python's recursion limit and gives up earlier,
  `Error: maximum recursion depth exceeded`): parentheses nested 52 deep
  (Python 37), calls 59 (40), lists 52 (37), blocks 237 (234). Beyond that
  tilc stops with the VM's `fatal error: stack overflow (recursion too
  deep)`. Operator chains have no limit in tilc (`1 + 2 + ... + n`, `not not
  ...`, `- - x` are compiled with loops); the Python compiler handles about
  1000 terms.
* Memory: tilc keeps the source text, the declarations, the tree of the
  statement it is compiling and the bytecode generated so far (as bytes). On
  the 64-bit C VM, with its default 256 KB heap, a script or a function of
  about 4,800 one-line statements compiles; on the calculator build (int32,
  float32, 64 KB heap) about 1,000. Each distinct constant
  costs about 100 bytes of heap (64-bit VM). When the heap is too small tilc
  stops with the VM's `runtime error: out of memory`; reading or writing a
  file that fails gives `tilc: cannot read FILE (...)` / `tilc: cannot write
  FILE (...)` with the VM's message, exit code 1.

Quirks of the Python compiler that tilc reproduces on purpose (so that both
give the same result; reported to the integrator):

* constant folding uses a global `let`/`const` even inside a function where a
  local of the same name shadows it;
* inside a lambda, calling a captured function parameter (`|x| f(x)`) is
  `unknown function 'f'` unless `f` was already used as a value in that
  lambda;
* `~true` folds to `-2` and `true == 1` folds to `true`, while the VMs, at run
  time, reject `~` on a bool and say `false`;
* `Vec<int>.new()` and `HashMap<K, V>.new()` are resolved by the
  SemanticChecker like any `.new` method call: when exactly one impl defines
  a `new` (e.g. `P.new(x)`), its arity and contracts are checked against the
  `Vec` call (`'P.new' expects 1 argument(s), got 0`) and it counts as a call
  for effects and capabilities;
* a compile error in an imported module is reported with the main file's name
  and the module's line number.

## How it works

The structure follows `src/til.py` function by function, so that both
compilers emit the same instructions, constants (in the same order) and line
tables.

* **Lexer** (`lex_next`): the token stream of `Lexer.tokenize`, produced one
  token at a time (indentation stack, `INDENT`/`DEDENT`, newlines ignored
  inside brackets, attributes, f-string parts, all escapes, number formats).
  Keyword tables are per source file (pragmas). Sources are checked as UTF-8
  (Python's `UnicodeDecodeError` message) and newlines are translated like
  Python's universal newlines.
* **Unicode**: the lexer classifies characters like `str.isalpha()`,
  `isdigit()`, `isdecimal()`, `isalnum()` (identifiers, numbers of every
  script: `١٢٣` is 123, `1²` is an invalid literal), `str.strip()` and `\s`
  use Unicode white space, `\u{...}` is read like `int(text, 16)`, and
  `repr()` escapes what `str.isprintable()` rejects. The data are two tables
  of runs (12 KB of string constants) between the `unicode tables` markers in
  tilc.til, generated by `selfhost/unicode_tables.py` (`--check` /
  `--write`; a test keeps them in sync).
* **Two passes.** Pass 1 parses the whole file (so syntax errors come first,
  as in Python) but keeps only declarations: function signatures with the
  source position of their body, structs, enums, impls, traits, globals, and
  the positions of script statements. Pass 2 compiles `__start__` and then
  every function in the order of `BytecodeCompiler.pending`, re-reading each
  body from its saved position.
* **Statements are compiled while they are parsed** (`stmt_c`): only
  expressions become trees. In pass 1 the same code runs with `DRY` set
  (parse only). Consecutive script statements share one item (their position
  and count), so a long script costs little more than its source text.
* **Operator chains** (`a + b + c`, `not not x`) are evaluated by
  `chain_values` and compiled by `cbinary` / `cunary` with loops, from the
  innermost operand out, with the same result as the recursive
  `compile_expr` (the outermost node that folds is pushed as a constant).
* **Blocks used as values** (`retry`/`timeout` bodies, their `else`, an `if`
  that is the last statement of such a block): the token stream is *forked*
  (a copy of the lexer state) and the block is skipped; when the expression is
  compiled the fork is replayed. A dry pre-scan on another fork finds the last
  statement (`compile_block_value` semantics).
* **Capture analysis** (`captured_names`) and the **SemanticChecker's call
  graph** are computed while pass 1 parses a function. The call graph is only
  recorded when it can matter: when the main file's tokens (from the lexical
  pass that runs before parsing) contain a `#[pure]` / `#[effects: ...]` /
  `#[cap: ...]` attribute or an `import` / `from` in any keyword language.
  Without such a token no function declares effects or capabilities, so
  E-EFFECT and E-CAP cannot occur.
* **Integers** are 4 limbs of 16 bits: tilc gives the same output on 32- and
  64-bit VMs and never needs an integer constant outside 32 bits itself. A
  literal of 2^63 or more also keeps its exact value, as in Python's
  `const_eval` (comparisons, `/`, `%`, conversion to float and E-CONTRACT
  messages use it; results wrap to 64 bits).
* **Floats**: literals are converted to IEEE 754 bits exactly (big-number
  arithmetic, round half to even, subnormals); constant folding of float
  expressions uses exact rational arithmetic rounded like double operations;
  Python's `repr()` of floats (shortest round trip) is used in E-CONTRACT
  messages. No host float operation is used, so a VM without doubles (float32
  calculators) produces the same bytes.
* **Output**: the code of the function being compiled is a byte array of at
  most ~520 elements; older bytes move to a rope of strings (one byte per
  byte instead of one VM value), and jump operands patched after their bytes
  moved are applied when the function is done. The line table is built as
  bytes too, and the constant table is a rope with its index in 16 maps (a
  map of the C VM holds 32,767 entries). The file is assembled from string
  chunks joined in groups.

## Bootstrap and tests

`tests/test_selfhost.py` (default suite, ~30 s):

* stage 1: the Python compiler builds `tilc.tbc`; stage 2: stage 1 on the C VM
  compiles `tilc.til`; stage 3: stage 2 compiles `tilc.til`. The test requires
  **stage 2 == stage 3 == stage 1**, byte for byte.
* the Python VM running stage 1 produces the same bytes as the C VM (a small
  program by default; all of `tilc.til` with `TIL_TEST_SLOW=1`, ~45 s);
* every example and every program of `tests/selfhost_corpus.txt` (the 262
  sources the test-suite compiles, harvested from `tests/test_compiler.py`,
  `tests/test_v3.py` and `tests/test_foundation.py`) is compiled by stage 2
  and by the Python compiler: the bytes must be identical, or (only for NaN
  folding) both versions must print the same on both VMs; programs the Python
  compiler rejects must be rejected;
* diagnostics, unsupported constructs, imports, the calculator configuration,
  and the C VM fixes (below);
* the findings of the first review: effect and capability checks for every
  spelling of imports and attributes, Unicode lexing (BOM, digits of other
  scripts, white space, `\u{...}`, invalid UTF-8), `repr()` in E-CONTRACT
  messages, NaN and big-literal folding, the `Vec<T>.new()` quirk, functions
  over 64 KB of bytecode, long operator chains, long programs on the
  calculator VM, an unwritable output file; the IDENTICAL programs are also
  run on both VMs; the Unicode tables are checked against `unicodedata`
  (with `TIL_TEST_SLOW=1` also more than 32,767 and 65,535 constants).

### Results

* All 20 programs in `examples/` compile to the same bytes as with the Python
  compiler (including `09_v3_features.til` and `10_verified.til`).
* Corpus: 244 of 262 programs compile to identical bytes; the other 18 are
  invalid programs that both compilers reject (17 with the same message at the
  same line; one imports a module that does not exist next to it).
* Programs outside TIL Core: none in the corpus or the examples.

### Performance and memory

Compiling `tilc.til` itself (6,360 lines, 197 KB of source; 118.9 KB of
bytecode; 2.87 million VM steps — loop iterations and calls). CPU time on the
development machine (a shared x86-64 server):

| compiler / VM | CPU time |
|---|---|
| Python compiler (stage 1) | 0.6 s |
| tilc on the C VM, 64-bit, `-O2`, 1 MB heap | 0.5 s |
| tilc on the C VM, int32 + float32, 1 MB heap | 0.5 s |
| tilc on the reference VM (Python) | 43 s |

All of them produce the same 118,908 bytes.

Self-compilation needs a heap of about 1 MB (992 KB on the 64-bit VM, 704 KB
with int32; the default C VM heap is 256 KB: build `tilvm` with
`-DTILVM_HEAP_SIZE=1048576` or more).

### Calculator demo

tilc running on a calculator-class build of the C VM —
`-DTILVM_INT_BITS=32 -DTILVM_FLOAT=1 -DTILVM_HEAP_SIZE=65536` (32-bit ints,
single-precision floats, 64 KB heap) — compiles programs to the same bytes as
the Python compiler on a PC:

```sh
cc -std=c99 -O2 -DTILVM_INT_BITS=32 -DTILVM_FLOAT=1 -DTILVM_HEAP_SIZE=65536 \
   -o tilvm-calc vm/tilvm.c vm/main.c -lm
./tilvm-calc --stats tilc.tbc examples/07_multilevel.til out.tbc
```

Smallest heap (in steps of 4-8 KB) that compiles each program on the
int32/float32 VM, VM steps, and CPU time on the development machine:

| program | lines | heap needed | steps | CPU time |
|---|---|---|---|---|
| `print(1)` | 1 | 20 KB | 686 | 2 ms |
| `TestCalculator` program (structs, loops, f-strings) | 36 | 24 KB | 18,850 | 4 ms |
| `examples/03_functions.til` | 56 | 28 KB | 15,768 | 4 ms |
| `examples/04_structs.til` | 81 | 36 KB | 32,966 | 6 ms |
| `examples/basic/control_flow.til` | 77 | 36 KB | 22,699 | 5 ms |
| `examples/07_multilevel.til` | 147 | 36 KB | 59,040 | 9 ms |
| `examples/09_v3_features.til` | 112 | 56 KB | 55,359 | 11 ms |

tilc's own data (keyword tables etc.) takes about 11 KB of that heap. A
calculator CPU is roughly 100-1000 times slower than a desktop one, so these
compile in well under a minute to a few seconds there (estimate from the step
counts; not measured on hardware).

## C VM fixes found while writing tilc and by its review

`join`, `copy`, `slice`, array `+`, map `keys`/`values`/`copy`, iterating a
map, `range` and `split` staged every element on the C VM's value stack (8192
slots), so arrays of more than about 8000 elements failed on the C VM with
`stack overflow` while the Python VM handled them (tilc builds large strings
from many chunks). The C VM now allocates these results directly
(`push_empty` in `vm/tilvm.c`); regression test `TestVMLargeArrays`.

`str()`, `join` and the "key not found" message measure a value's text with
`emit_value` before allocating it. The count was a `uint32_t`: an array that
shares big sub-arrays (16 references to 16 references ... to a 64 KB string)
has a text of more than 4 GB, the count wrapped, a small string was allocated
and gigabytes were written into it (heap overflow, found with ASan). The
count now stops just above the string limit (`runtime error: string too
long`); regression test `TestVMHugeText`. (The Python VM has no string
length limit.)
