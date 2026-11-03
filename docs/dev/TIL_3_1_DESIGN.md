# TIL 3.1 — design of the 14 roadmap features

Internal engineering spec. Every feature below is implemented against this
document; where it fixes a name, number, syntax or format, implementations must
follow it exactly so that independently developed parts fit together.

## 0. Ground rules (all features)

* **Two VMs in lockstep.** Every new opcode, native or builtin method exists in
  the reference VM (`class TILVM`, `src/til.py`) *and* the C VM (`vm/tilvm.c`)
  with identical observable behaviour. Differential tests compare them.
* **C VM portability.** `vm/tilvm.c` must compile with
  `-std=c99 -Wall -Wextra -pedantic -Werror` with gcc and clang in these
  configurations: default; `-DTILVM_INT_BITS=32 -DTILVM_FLOAT=1`;
  `-DTILVM_LEAN -DTILVM_INT_BITS=16 -DTILVM_FLOAT=0 -DTILVM_NO_STDIO`;
  `-DTILVM_NO_STDIO -DTILVM_NO_TIME`; `-DTILVM_NO_STDIO -DTILVM_NO_MATH -DTILVM_FLOAT=1`.
  The VM core never calls `malloc`. Host services go through `vm->host` hooks.
* **Tests.** `python3 -m pytest tests/ -q` must stay green. Each feature adds
  its own test file `tests/test_<feature>.py` (do not rewrite existing tests).
  New programs used in tests run on both VMs (and natively when supported) and
  must produce identical output.
* **Bytecode version 2.** The compiler writes version 2; VMs accept 1 and 2.
  New instructions/natives are append-only; numbers never change meaning.
* **No silent caps.** If something is unsupported (a construct the prover
  cannot handle, a builtin missing in `TILVM_LEAN`), report it explicitly.

## 0.1 What the foundation commit provides

* Bytecode v2 (`TBC_VERSION = 2`, `TBC_MIN_VERSION = 1`; C:
  `TILVM_BYTECODE_VERSION 2`, `TILVM_BYTECODE_MIN_VERSION 1`).
* Natives 34–54 registered in `NATIVES` (`src/til.py`), the C enum and
  `NATIVE_NAMES` (`vm/tilvm.c`). **Implemented in both VMs:** `read_file`,
  `write_file`, `file_exists`, `args`, `bytes_to_str`, `str_bytes` (not in
  `TILVM_LEAN`). The screen/keys natives (40–53) and `energy_mode` (54) still
  report `<name>() is not available in this VM build` — their features
  implement them.
* `NATIVE_ARITY` (compile-time min/max check) and `NATIVE_MIN_ARGS` (runtime
  minimum, both VMs: `<name>() is missing arguments`). `range(n)` compiles as
  `range(0, n)`. Keep the C table `NATIVE_MIN_ARGS` in sync (a test checks it).
* Opcodes `HANDLE/END_HANDLE/PERFORM/RESUME` are defined (`Op`, `OP_OPERANDS`,
  disassembler, C enum) but **not implemented** by the VMs yet (§8).
* `TBCFunction.FLAG_ENERGY_LOW = 8`, `FLAG_EFFECT_CLAUSE = 16`; C:
  `FLAG_ENERGY_LOW`, `FLAG_EFFECT_CLAUSE`.
* `tilvm_host` (below) is the field `vm->host`; `tilvm_load` resets it and,
  with stdio, installs file hooks. `vm/main.c` passes the arguments after the
  program file to `host.argc/argv`.
* C VM strings may be up to 16 MB when the heap is larger than 64 KB
  (`TV_STR_MAX`); arrays still hold at most 65535 elements.
* Python VM: `TILVM(module, write=None, read_line=None, max_steps=0, argv=None)`;
  add new keyword parameters after `argv`.
* CLI (`main()` in `src/til.py`):
  * `til run prog.til a b c` / `til exec prog.tbc a b` / `--cvm`: arguments
    after the program file (or after `--`) go to `args()`.
  * `RUN_VALUE_OPTIONS` (`--screen`, `--keys`, `--pause-after`, `--save`,
    `--resume`, `--lang`, `--device`) and `RUN_FLAG_OPTIONS` (`--screen-ansi`,
    `--prove`) are parsed into `opts` and passed to
    `run_bytecode(module, max_steps, argv, opts)`. Add new run options to these
    tables instead of new parsing code.
  * `EXTRA_COMMANDS` maps `pkg`, `prove`, `fit`, `debug` to modules next to
    `til.py` (`src/tilpkg.py`, `src/tilprove.py`, `src/tilfit.py`,
    `src/tildebug.py`), each exposing `main(argv: List[str]) -> int`. Load them
    with `feature_module(name)`; inside the tools `import til` returns the
    running compiler module.
  * `til check` calls `run_proofs()` → `tilprove.check_program(program, source,
    filename, explicit) -> List[str]` (formatted errors; empty = OK).
* Tests: `tests/tilhelpers.py` (`run_vm`, `run_cvm`, `run_both`, `cli`,
  `cvm_binary(profile)`, `expect_compile_error`, `needs_cc`) and
  `tests/test_foundation.py`.
* Python code must run on Python 3.9 (CI matrix 3.9/3.11/3.12): no `match`,
  no `X | Y` annotations evaluated at runtime, no nested same-type quotes in
  f-strings. Check with `vermin -t=3.9- --violations src/`.
* The C VM's default `clock_ms` uses `clock()` (CPU time). Wall-clock based
  features (energy saving, sleeping) must not rely on it while the CPU idles.

## 1. Reserved identifiers (registered in both VMs by the foundation commit)

### Natives (ids 34–54, appended to `NATIVES`)

| id | name | semantics |
|---|---|---|
| 34 | `read_file(path)` | file contents as a (byte) string; panics `cannot read file: <path>` |
| 35 | `write_file(path, data)` | writes the string's bytes; returns nil; panics on failure |
| 36 | `file_exists(path)` | bool |
| 37 | `args()` | array of strings: program arguments after the program file |
| 38 | `bytes_to_str(arr)` | string whose bytes are the ints 0..255 in `arr` |
| 39 | `str_bytes(s)` | array of the string's byte values |
| 40 | `screen_init(w, h)` | open/resize the screen (pixels), cleared to black |
| 41 | `screen_clear(color)` | fill with color |
| 42 | `draw_pixel(x, y, color)` | |
| 43 | `draw_line(x0, y0, x1, y1, color)` | Bresenham |
| 44 | `draw_rect(x, y, w, h, color)` | outline |
| 45 | `fill_rect(x, y, w, h, color)` | |
| 46 | `draw_circle(cx, cy, r, color)` | midpoint circle outline |
| 47 | `draw_text(x, y, text, color)` | built-in 5x7 font, 6 px advance |
| 48 | `get_pixel(x, y)` | int color |
| 49 | `screen_show()` | present the frame (flush to device/terminal) |
| 50 | `key_pressed()` | key code or 0 (non-blocking) |
| 51 | `wait_key()` | blocking key code |
| 52 | `screen_width()` | |
| 53 | `screen_height()` | |
| 54 | `energy_mode(mode)` | 0 performance, 1 balanced, 2 low; returns previous |

Colors are `0xRRGGBB` ints. Key codes: `UP=1 DOWN=2 LEFT=3 RIGHT=4`, ASCII
otherwise (`ENTER=13 ESC=27 SPACE=32`, letters/digits as ASCII).

### Opcodes

| op | hex | operands | meaning |
|---|---|---|---|
| HANDLE | 76 | b n, s rel_end | pushes a handler record for the n (clause closure, effect-name string) pairs on top of the value stack; the pairs **stay on the value stack** under the body (stack effect 0; the record holds their index). `rel_end` = abort target; there the stack is cut back to below the pairs and holds the clause value (depth before the pairs + 1) |
| END_HANDLE | 77 | | body finished: pops the value, the handler record and its 2n pairs, pushes the value again (stack effect −2n) |
| PERFORM | 78 | h name, h default_fn, b argc | perform an effect (see §8); stack like `CALL`: argc arguments → 1 result |
| RESUME | 79 | | inside a clause: return TOS to the `perform` site |
| 0x80–0xBF | | | reserved for superinstructions (§13) |

Exact semantics of the four effect opcodes (handler entries, masking, aborts,
interaction with `TRY`, snapshots): `docs/TIL_EFFECTS.md` §3. Stack-depth tools
(`til fit`, debugger, portable snapshots) must count the 2n pair slots of every
active `handle`.

### Function flags (byte in the function record)

`1` method, `2` `?` propagates, `4` lambda, `8` **energy-low**, `16` **effect clause**.

### Host hooks (C VM, `vm->host`)

```c
typedef struct tilvm_host {
    long (*read_file)(void *ctx, const char *path, char *buf, unsigned long cap); /* size or -1 */
    int  (*write_file)(void *ctx, const char *path, const char *data, unsigned long len);
    void (*gfx_init)(void *ctx, int w, int h);
    void (*gfx_pixel)(void *ctx, int x, int y, unsigned long rgb);
    unsigned long (*gfx_get)(void *ctx, int x, int y);
    void (*gfx_show)(void *ctx);
    int  (*key_pressed)(void *ctx);
    int  (*wait_key)(void *ctx);
    void (*idle)(void *ctx, unsigned long ms);
    int argc;
    const char *const *argv;
} tilvm_host;
```

`NULL` hooks mean "not available" (natives report it). With stdio, `tilvm_load`
installs default `read_file`/`write_file`.

## 1. Self-hosted compiler (`selfhost/`)

`selfhost/tilc.til` is a TIL compiler written in TIL ("TIL Core" subset) that
produces `.tbc` bytecode which runs on both VMs.

* Usage: `til run --vm selfhost/tilc.til input.til output.tbc` (any VM:
  `tilvm tilc.tbc input.til output.tbc`). Uses `args`, `read_file`,
  `write_file`, `bytes_to_str`.
* Supported subset (documented in `selfhost/README.md`): functions (params,
  return), `let/var/const`, globals, int/float/str/bool literals, all
  operators, `if/elif/else`, `while`, `loop`, `for` over ranges and arrays,
  `break/continue/return`, arrays and maps (literals, index, methods), structs
  (+ `impl` methods), f-strings without format specs, `print` and the other
  natives, string methods, `assert`. Anything else → a clear compile error.
* **Bootstrap fixed point** (test): stage1 = Python compiler builds `tilc.tbc`;
  stage2 = stage1 compiles `tilc.til`; stage3 = stage2 compiles `tilc.til`;
  require `stage2 == stage3` byte-for-byte, and programs built by stage2 print
  the same as when built by the Python compiler.

## 2. Portable snapshots (`.tilp`)

Device-independent snapshot: start on a PC, continue on a calculator.

* Format `TILP` v1 (little endian): magic, version, FNV-1a-32 hash of the
  `.tbc` bytes, state (steps u32, rng u64, flags), globals count, frames
  (func u16, pc relative to the code section u32, base u16, extra u8, closure
  object id u32), recovery points (handler pc rel, **remaining** deadline ms,
  frame count, sp), handler records (if effects exist), stack values, then the
  object table (objects numbered in first-visit order from the roots).
* Values: nil, bool, int (i64), float (f64), const string (u16 const index),
  object ref (u32 id). Objects: str, array, map, struct, closure, cell.
* C API: `tilvm_save_portable(vm, write_cb, ctx)`,
  `tilvm_load_portable(vm, read_cb, ctx)` (after `tilvm_load`, before running);
  must work with 16/32/64-bit ints (out-of-range ints → load error) and
  with/without floats.
* Python VM: `TILVM.run_steps(budget)` (pause after N steps — same step
  counting as C), `save_portable() -> bytes`, `load_portable(bytes)`.
* CLI: `til run --vm prog.til --pause-after N --save state.tilp`,
  `til run --vm prog.til --resume state.tilp`; `tilvm --save-portable-after N FILE`,
  `tilvm --resume-portable FILE`.
* Test: Python VM → C VM (int32/float build) and C VM → Python VM; outputs of
  (part1 + part2) equal an uninterrupted run.

## 3. Graphics and keys

Natives 40–53. Shapes and the 5x7 font are rasterized in the VM core (identical
pixels everywhere); hosts only provide `gfx_pixel/get/show` and keys.

* Desktop C VM (`vm/main.c`) and Python VM keep a framebuffer;
  `--screen out.ppm` writes the final frame (P6), `--screen-ansi` renders to
  the terminal with 24-bit colour half blocks; `--keys "UP,UP,ENTER,a"` feeds
  key presses (tests), else keys come from the terminal.
* `til run --vm game.til --screen out.ppm --keys ...`.
* Example `examples/11_graphics.til`; test: identical PPM from both VMs.

## 4. Time-travel debugger

`til debug prog.til` (interactive) / `til debug prog.til --commands "b 5; c; p x; back; p x; q"`.

* Runs on the reference VM with line events. Commands: `b LINE`, `d LINE`,
  `c`, `s` (step into), `n` (step over), `back` (step back one line event),
  `rc` (reverse-continue to the previous breakpoint hit), `p NAME` (local or
  global), `locals`, `bt`, `out`, `q`.
* Snapshots of the whole VM state every K line events; stepping back restores
  the nearest snapshot and replays forward deterministically (recorded input,
  clock and random values; output suppressed during replay).
* Needs local-variable debug info: an optional `.tbc` section appended after
  the line table: `n u16, (func u16, slot u8, name const u16, start_pc u32, end_pc u32)*`
  (omitted with `--strip`; the C VM ignores it).

## 5. Online playground (`website/playground/`)

* `tilvm.wasm` built from the unmodified `vm/tilvm.c` with
  `clang --target=wasm32` (`TILVM_NO_STDIO`, tiny libc shim, `Math.*` imports).
* Compiles in the browser with the self-hosted compiler (`tilc.tbc`), runs
  the result in a second VM instance; output panel; graphics on a `<canvas>`;
  examples; share via URL hash. No CDN dependencies.
* Build script `website/playground/build.sh`; test with Node + Playwright.
* `website/index.html` links to the real playground (replace the fake one).

## 6. Contract proofs (`src/tilprove.py`)

`til prove prog.til` and `til check --prove`: proves `#[requires]` ⇒
`#[ensures]` for every function and that every call satisfies the callee's
`#[requires]`, **for all inputs**, with Z3 (`pip install z3-solver`).

* Integers as 64-bit bit-vectors (exact wrap-around, signed `/` `%` like C).
* Supported: int/bool params, locals, arithmetic, comparisons, logic,
  `if/elif/else`, `return`, calls to functions with contracts (assume their
  `ensures`), `while` loops with `invariant` statements as loop invariants.
* Per function: `proved` / `counterexample: n = -1` / `unsupported: <reason>`.
* Level 4 functions (`#[level: 4]`) are proved during `til check` when Z3 is
  available; without Z3 the command explains how to install it.

## 7. Typed native backend with generics

* `class TypeInfer` (gradual type inference, unification, generics as type
  variables) annotates every expression; definite type errors are reported as
  `E-TYPE` (never on valid programs).
* The C backend uses the inferred types instead of name-based guessing and
  monomorphizes generic functions (`f<int>`, `f<str>` → separate C functions).
* Success = more programs compile natively without falling back to the VM
  (measured before/after over all test programs and examples), no regressions.

## 8. Effects with resumption

```
handle
    let name = perform Ask("name?")
    perform Log("hello " + name)
    "done"
with
    Ask(question) => resume("Aigerim")
    Log(msg) =>
        print(msg)
        resume(None)
    Fail(reason) => "failed: " + reason        # no resume: aborts the handle
```

* `perform E(args)` finds the innermost enclosing handler with a clause for
  `E` (dynamically, across calls). The clause runs with its parameters bound.
  `resume(v)` returns `v` as the value of `perform` and continues the body.
  If the clause finishes without `resume`, its value becomes the value of the
  whole `handle` expression (the body is abandoned).
* Inside a clause, `perform` of the same effect goes to an *outer* handler.
* No handler: a top-level function named `E` is the default handler (keeps
  v3.0 programs working); otherwise runtime error `unhandled effect 'E'`.
* VM: opcodes HANDLE/END_HANDLE/PERFORM/RESUME; clause = closure with flag 16.
  Snapshots (native and portable) include handler records.
* The native C backend may fall back to the VM for programs using `handle`.

## 9. Z80

* Make a 16-bit lean build compile with SDCC (find and work around the
  internal compiler error) and fit below 0xC000; run programs on `sz80`.
* `vm/ports/ti84ce/`: TI-84 Plus CE (eZ80) port (CE toolchain), graphics hooks
  via graphx. Build it if the toolchain can be obtained.
* Update `vm/ports/z80/RESULTS.md` with measurements.

## 10. Memory fit check

`til fit prog.til --device NAME` (and `til build --target vm --device NAME`):

* Device profiles (int bits, float, heap, stack slots, frames, consts, types,
  try depth, lean, code storage): `desktop`, `ti84ce`, `numworks`, `nspire`,
  `casio-cg50`, `cortex-m3-64k`, `atmega1284`, `arduino-mega`, `esp32`,
  `rp2040`, `z80-16`.
* Static: bytecode size, constants/types counts, int/float requirements,
  lean-only builtins, per-function max operand stack (abstract interpretation of
  bytecode), worst-case stack slots and call depth over the call graph
  (recursion reported), allocation sites in loops.
* Dynamic (`--run`): run under the device's limits (C VM high-water marks:
  peak stack, peak heap after GC, max frames) and report.
* Output: table + verdict `fits` / `does not fit: <reasons>`; exit code 0/1.

## 11. Energy saving

* `#[energy: low]` on a function sets flag 8; a top-level `#[energy: low]`
  pragma (before any definition) sets the initial mode for the whole program.
* Modes (`energy_mode()`): 0 performance (never idle), 1 balanced (idle 1 ms
  every 4096 steps), 2 low (idle 1 ms every 512 steps). Inside an energy-low
  function the VM behaves as mode 2.
* Idle = `vm->host.idle(ctx, ms)` (desktop: sleep; calculators: halt until
  interrupt). Reference VM: `time.sleep`.
* Test: a CPU-bound loop under mode 2 uses measurably less CPU time per wall
  second than mode 0; output identical.

## 12. Localized messages (Kazakh, Russian)

* All compiler diagnostics (syntax errors, E-EFFECT, E-CAP, E-CONTRACT,
  E-ARITY, E-TYPE, E-VM, hints, the words "error"/"hint"/"warning") come from a
  message catalog with `en`, `ru`, `kk` texts.
* Language: `--lang en|ru|kk` > `TIL_LANG` env > the program's `#[lang: ...]`
  pragma (`ru` → ru, `kk`/`kk-latn` → kk) > English.
* Runtime errors: values seen by programs (`error` in `else`) stay English;
  the uncaught-error *report* is localized (prefix + known message patterns),
  for both VMs when run through `til`.

## 13. Faster VM (2–3×)

* Targets (desktop, `-O2`): `fib(27)` and a loop/array/string benchmark at least
  2× faster than TIL 3.0 on the C VM, ideally 3×; reference VM not slower.
* Techniques: computed-goto dispatch (`__GNUC__`, fallback switch),
  cached stack pointer, superinstructions emitted by a compiler peephole pass
  (0x80–0xBF; implemented in both VMs), precomputed function table, field
  inline caches. `benchmarks/` with a runner script and recorded results.

## 14. Package manager (`src/tilpkg.py`)

* `til pkg init` → `til.json` `{name, version, dependencies: {name: spec}}`.
* `til pkg add NAME SPEC` where SPEC is a local path, `git+URL[#rev]`, or
  `https://…tar.gz|zip#sha256=…`; `til pkg install` resolves dependencies
  (recursively), fetches into `til_modules/NAME/`, writes `til.lock`
  (resolved source + sha256 of the tree); `til pkg remove`, `til pkg list`,
  `til pkg verify` (hashes). Imports search `til_modules/` (nearest project
  root upwards from the importing file).
* Optional registry: JSON index (`TIL_REGISTRY` env / `til.json` "registry")
  mapping names to sources and versions.
* Tests use local paths and local git repositories (`file://`).
