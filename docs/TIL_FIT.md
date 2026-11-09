# `til fit` — will a program fit a device?

TIL programs run on calculators and microcontrollers through the portable C VM
(`vm/tilvm.c`). Those devices have a few kilobytes of RAM, a fixed value stack,
a fixed call depth, sometimes no floating point and sometimes a *lean* VM build
without some builtins. `til fit` answers, before you copy anything to the
device, whether a program fits — and if not, why.

```bash
til fit game.til --device ti84ce            # static analysis: table + verdict
til fit game.til --device ti84ce --run      # ... and run it under the device's limits
til fit game.til --device z80-16 --json     # machine-readable report
til fit --list-devices                      # the device profiles
til build --target vm --device atmega1284 sensor.til   # fail the build if it does not fit
```

Exit status: **0** fits, **1** does not fit, **2** usage error or the program
cannot be analysed (compile error, unreadable or malformed file).

## Example

```
$ til fit examples/03_functions.til --device atmega1284 --run
til fit: examples/03_functions.til on atmega1284
device: ATmega1284P, 16 KB RAM / 128 KB flash; 8-bit AVR @ 16-20 MHz, verified in simavr
VM build: int32, float32, 5 KB heap, 160 stack slots, 24 frames, full VM

  resource         program needs        measured  device   status
  bytecode         882 B                          55 KB    ok
  VM RAM           15.6 KB                        16 KB    ok  VM struct 9.7 KB + VM tables 3.5 KB + C stack 1.5 KB + bytecode 882 B
  constants        22                             160      ok
  struct types     3                              24       ok
  functions        7                              160      ok
  globals          0                              152      ok
  integers         16-bit                         32-bit   ok
  floats           none                           float32  ok
  builtins         1 native, 0 methods            full     ok
  value stack      11 slots             22        160      ok  +2 per recursion level
  call frames      3                    12        24       ok  +1 per recursion level
  recovery points  0                    0         6        ok
  heap             depends on data      94 B      5 KB     -

warnings:
  - recursion: factorial calls itself (line 26 in factorial); the depth depends on the
    input. Each level needs 2 value-stack slots and 1 call frame: 22 levels fit on this
    device (limited by call frames)
  - recursion: fib calls itself (line 32 in fib); ...

notes:
  - integers are 32-bit on this device: results beyond +/-2147483647 wrap around

run: exit code 0, 319 steps, 0 garbage collections

verdict: fits
```

A program with `handle ... with` gets an *effect handlers* row (open handle blocks
plus running clauses, `TILVM_MAX_HANDLERS`); on `z80-16`, whose VM is built without
them, it does not fit:

```
$ til fit examples/12_effects.til --device z80-16
  ...
  builtins         3 natives, 1 method  lean      FAIL  not in this build: effect handlers
  value stack      28 slots             192       ok
  call frames      4                    32        ok
  recovery points  1                    8         ok
  effect handlers  4                    none      FAIL  TILVM_NO_EFFECTS
  ...
verdict: does not fit: effect handlers not available (TILVM_NO_EFFECTS)
```

*program needs* is the static worst case (recursion counted once), *measured* the
high-water mark of the `--run`, *device* the limit of the device's VM build.
Errors make the verdict `does not fit: <reasons>`; warnings are risks that depend
on the input; notes are information.

## Options

| option | meaning |
|---|---|
| `--device NAME` | a built-in profile (below) or a JSON file (see *Custom devices*) |
| `--run` | build the C VM with the device's limits (needs a C compiler) and run the program; adds measured high-water marks and compares the output with the desktop VM |
| `--json` | machine-readable report (schema below) |
| `--strip` | check the bytecode without line tables, as you would deploy it (`til build --strip`) |
| `--functions` | per-function table: locals, operand-stack depth, frame requirement |
| `--max-steps N` | `--run`: stop after N steps (loop iterations + calls); measurements cover that part |
| `--timeout SEC` | `--run`: give up after SEC seconds |
| `--stdin FILE` | `--run`: text for `input()` (default: none, `input()` returns `""`) |
| `--show-output` | `--run`: print the program's output before the report |
| `-- ARGS` | `--run`: program arguments (`args()`); the static numbers count them too |

The input may be a `.til` source or a compiled `.tbc` file.

## Device profiles

Every profile is a build configuration of the C VM (the `TILVM_*` macros of
`vm/tilvm.h`) plus what the device offers around it.

| device | int | float | heap | stack slots | frames | consts / types | try | handlers | lean | other switches | bytecode storage | RAM for the VM | VM struct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `desktop` | 64 | double | 256 KB | 8192 | 1000 | 4096 / 512 | 32 | 32 | no | — | — | — | 456 KB |
| `ti84ce` | 32 | float | 32 KB | 1024 | 128 | 1024 / 128 | 16 | 32 | no | addr16, screen 320x240 | 65505 B | 96 KB | 48.3 KB |
| `numworks` | 32 | double | 32 KB | 1024 | 200 | 1024 / 128 | 16 | 32 | no | — | 1 MB | 96 KB | 65.6 KB |
| `nspire` | 64 | double | 1 MB | 16384 | 2000 | 8192 / 1024 | 32 | 32 | no | — | 16 MB | 16 MB | 1.4 MB |
| `casio-cg50` | 32 | double | 64 KB | 2048 | 256 | 2048 / 256 | 16 | 32 | no | — | 2 MB | 512 KB | 128 KB |
| `cortex-m3-64k` | 32 | float | 16 KB | 512 | 64 | 512 / 64 | 8 | 32 | no | — | 200 KB | 64 KB | 29.1 KB |
| `atmega1284` | 32 | float | 5 KB | 160 | 24 | 160 / 24 | 6 | 32 | no | font in flash | 55 KB | 16 KB | 9.7 KB |
| `arduino-mega` | 16 | float | 1.5 KB | 96 | 16 | 96 / 16 | 4 | 8 | **yes** | no screen | 208 KB | 8 KB | 3.1 KB |
| `esp32` | 32 | float | 64 KB | 2048 | 256 | 2048 / 256 | 16 | 32 | no | — | 1 MB | 160 KB | 112 KB |
| `rp2040` | 32 | float | 128 KB | 2048 | 256 | 2048 / 256 | 16 | 32 | no | — | 1 MB | 240 KB | 176 KB |
| `z80-16` | 16 | none | 4 KB | 192 | 32 | 192 / 24 | 8 | — | **yes** | no effects, no screen, addr16 | 6126 B | 6 KB | 5.7 KB |

*VM struct* is `sizeof(tilvm)` for the device's compiler (see *The VM's own RAM*).
All profiles except `desktop` and `nspire` build the VM with `TILVM_NO_STDIO`
and, except `ti84ce`, `TILVM_NO_TIME` (the port provides output and, where noted,
a clock). The TIL 3.1 tables of the VM struct follow the header's defaults unless a
profile says otherwise: the function table (`TILVM_MAX_FUNCS` = constants; none in
lean builds), the field/method caches (`TILVM_FIELD_CACHE` = 64; none in lean
builds) and 32 effect handler entries (`TILVM_MAX_HANDLERS`).

### Sources and assumptions

* **Verified profiles.** `cortex-m3-64k`, `atmega1284`, `z80-16` and `ti84ce` use
  exactly the flags of `vm/ports/cortex-m/Makefile`, `vm/ports/avr/Makefile`,
  `vm/ports/z80/Makefile` and `vm/ports/ti84ce/makefile` (switches such as
  `TILVM_ADDR16`, `TILVM_NO_EFFECTS`, `TILVM_NO_GFX`, `TILVM_ROM=__flash` and
  `TILVM_GFX_MAX_W/H` included; a test checks this). The Cortex-M and AVR ports run
  all examples in QEMU / simavr, the Z80 build in the ucsim simulator. Measured with
  arm-none-eabi-gcc 13.2 `-Os` + newlib-nano: VM code 53.2 KB of the 256 KB flash
  (storage = 256 − 56 KB), 80 bytes of other static data; with avr-gcc 7.3 `-Os`:
  VM code 74 KB of the 128 KB flash (storage = 128 − 75 KB, which also leaves the
  flash copy of the VM's tables and of the bytecode room).
* **AVR keeps constant data in RAM** (Harvard architecture, no `PROGMEM` in the
  VM, except the 5x7 font, which `TILVM_ROM=__flash` keeps in flash): the VM's
  string tables and error messages take 3860 bytes of RAM (full build) or 2790
  bytes (lean build), measured as `.data` minus the program with avr-gcc
  (`TIL_TEST_CROSS=1 pytest tests/test_fit.py` links the AVR and Cortex-M ports and
  compares their `.data`/`.bss` with the model). The AVR port also copies the
  bytecode into RAM (`program.h`), so on `atmega1284` and `arduino-mega` the
  bytecode counts against RAM, not only against flash. On the ATmega1284P the TIL
  3.1 VM struct (9.7 KB: function table 1.25 KB, lookup caches 768 B, effect
  handlers 512 B on top of heap and stack) leaves about 1.2 KB for bytecode.
* **`arduino-mega`** (ATmega2560: 8 KB RAM, 256 KB flash): a lean 16-bit build in
  the style of `make tiny`, sized to leave ~0.8 KB of RAM for bytecode: 1.5 KB
  heap, 96 stack slots, 16 frames, 8 effect handler entries, no screen natives
  (`TILVM_NO_GFX`: the board has no display). Floats are single precision
  (avr-gcc's `double` is 32-bit). VM code 44.9 KB (lean, avr-gcc `-Os`).
* **`z80-16`** is the build of `vm/ports/z80` (`TILVM_LEAN`, `TILVM_ADDR16`,
  16-bit integers, no floats, `TILVM_NO_EFFECTS`, `TILVM_NO_GFX`: with handlers and
  screen natives the code would not fit below 0xC000) with the memory map of its
  simulator harness (`main_z80.c`, `RESULTS.md`): the VM state at `0xC000-0xD7FF`
  (6 KB; `sizeof(tilvm)` = 5840 bytes with SDCC), the program at `0xD800-0xEFEF`
  (a u16 length + at most 6126 bytes of bytecode), the C stack in a region of its own
  at `0xF800` (2 KB; not counted). A real TI-83 Plus / TI-84 Plus needs a multi-page
  Flash application (not implemented, see `RESULTS.md`).
* **`ti84ce`**: TI-84 Plus CE (eZ80, ~150 KB of user RAM): the profile of
  `vm/ports/ti84ce/makefile` (heap 32 KB, `TILVM_FLOAT=1`: the CE toolchain's
  `double` is 32-bit; `TILVM_ADDR16`; a 320x240 screen, so `screen_init` accepts
  at most that; a clock and AppVar files). The bytecode is stored in one AppVar (at
  most 65505 bytes) and the port copies it to RAM; 96 KB of RAM for the VM's data,
  the C stack and that copy is an assumption that leaves room for the VM code,
  which also runs from RAM on the CE.
* **`numworks`** (external app, Cortex-M7, 256 KB RAM): 96 KB of RAM for the app and
  a 32 KB heap (the size of the calculator's own Python heap) are assumptions;
  doubles, like the built-in Python.
* **`nspire`** (Ndless, ARM9, 64 MB SDRAM): the desktop build works as is; a 1 MB
  heap and 16 MB of RAM/storage are conservative choices.
* **`casio-cg50`** (gint add-in, SH4A, 8 MB RAM): 512 KB of add-in RAM is an
  assumption. SH4 is modelled as a 32-bit ABI with 8-byte-aligned doubles, which
  overestimates `sizeof(tilvm)` slightly (an upper bound).
* **`esp32`, `rp2040`**: vendor RAM/flash sizes; 160 KB of static DRAM (ESP32) and
  240 KB of SRAM (RP2040) for the VM, and 1 MB of flash for bytecode (read in
  place), are assumptions.
* **C stack reserve** (inside *RAM for the VM*): 1.5 KB on AVR (the deepest VM
  call chain — printing a value nested 16 levels deep — needs about 2.2 KB; a
  typical one about 0.8 KB, measured with `-fstack-usage`), 4 KB on Cortex-M3 and
  the CE, 8–64 KB on the larger devices.
* **Device services**: `screen` (calculators, desktop: the port installs the
  screen/keys hooks), `files` (desktop, nspire, the CE's AppVars),
  `console` for `input()`, `args` (a command line for `args()`: desktop, nspire), and
  `clock` (the port provides `clock_ms`; without it `timeout(ms)` and `clock_ms()`
  count VM steps, 1 ms = 1000 steps — true for the reference ports `cortex-m`, `avr`
  and `z80`).

### Custom devices

`--device board.json` takes a JSON object with any of the fields above, based on
a built-in profile:

```json
{"base": "cortex-m3-64k", "name": "my-board", "heap": 32768, "stack": 1024, "ram": 131072}
```

Fields: `int_bits`, `float`, `heap`, `stack`, `frames`, `consts`, `types`,
`try_depth`, `max_handlers`, `mark_stack`, `max_funcs`, `field_cache`, `lean`,
`addr16`, `no_effects`, `no_gfx`, `gfx_max_w`, `gfx_max_h`, `no_stdio`, `no_time`,
`no_math`, `rom`, `storage`, `ram`, `c_stack`, `vm_static_ram`, `bytecode_in_ram`,
`abi` (`lp64`, `ilp32`, `avr`, `z80`, `ez80`), `clock`, `screen`, `files`, `console`,
`args`, `title`, `cpu`, `source` (the build fields are the `TILVM_*` macros of
`vm/tilvm.h`; `max_funcs` and `field_cache` `null` = the header's default, which
depends on `lean`; `rom` is the device compiler's `TILVM_ROM` qualifier, left out of
the desktop builds of `--run`). Unknown fields, impossible values and wrong JSON types
are errors: numbers must be whole numbers (`16`, not `16.0` or `true`), switches
`true`/`false`, `storage`, `ram`, `max_funcs` and `field_cache` may be `null`.
`float: 2` is refused with the `avr`, `z80` and `ez80` ABIs: their toolchains'
`double` is 32-bit (use `float: 1`); `addr16` needs a heap of at most 65535 bytes.

## What is checked

### Errors (the verdict becomes `does not fit`)

| check | why it matters on the device |
|---|---|
| bytecode size > storage | the program cannot be stored (`--strip` saves the line tables and the debug-info section) |
| bytecode ≥ 64 KB with `TILVM_ADDR16` | `tilvm_load` fails (error -10) |
| VM RAM > device RAM | `sizeof(tilvm)` + VM tables + C stack (+ bytecode on AVR and the CE); the error names the biggest parts of the VM struct |
| constants > `TILVM_MAX_CONSTS`, struct types > `TILVM_MAX_TYPES` | `tilvm_load` fails (error -4 / -7) |
| functions > `TILVM_MAX_FUNCS` (when the build has a function table) | `tilvm_load` fails (error -10) |
| globals > stack slots − 8 | `tilvm_load` fails (error -8) |
| an integer constant wider than the device's integers | `tilvm_load` fails (error -6); the report names the constant and its line |
| float constants on a VM without floats | `tilvm_load` fails (error -3) |
| float conversions (`float()`, `: float` / `-> float`, `as float`) and `sqrt/sin/cos/tan/log/exp/floor/ceil/round/pow` on a VM without floats | runtime error ("built without floating point", "not available") |
| `sin/cos/tan/log/exp` with `TILVM_NO_MATH` | runtime error |
| `handle ... with` on a `TILVM_NO_EFFECTS` build (`z80-16`) | runtime error "effect handlers are not available in this VM build" (`perform` without a handler still calls the function of that name, so that is fine) |
| screen/keys natives on a `TILVM_NO_GFX` build (`z80-16`, `arduino-mega`) | runtime error "... is not available in this VM build" |
| `screen_init(w, h)` with constant sizes beyond `TILVM_GFX_MAX_W/H` (`ti84ce`: 320x240) | runtime error "screen_init() argument 1 must be 1..320" (a computed size is a note) |
| builtins a lean build leaves out (`input chr ord type_of clock_ms random seed sleep_ms read_file write_file file_exists args bytes_to_str str_bytes`, and the methods below) | runtime error ("not available", "no method") |
| `+` on two arrays on a lean build, when one operand is always an array | runtime error ("unsupported operand types for +: array and array") |
| f-string format specs / `format()` on a lean build | the lean VM ignores specs: the output would differ |
| map literals with a repeated constant key (`{"a": 1, "a": 2}`, also `1` and `1.0`) or an integral float key (`2.0`) on a lean build | the lean VM keeps every entry and float keys: `len()` and the output differ |
| value stack, call frames, recovery points, effect handler entries over the limit (worst case) | "stack overflow", "recursion too deep", "too many nested retry/timeout blocks", "too many nested handlers" |
| recursion where not even two levels fit | |
| analysis incomplete (unknown opcode, malformed code) | the stack use cannot be verified |
| `--run`: the run stops with a device limit (out of memory, stack overflow, recursion too deep, too many nested blocks or handlers, range/array/string too large — also when the desktop VM stops there too: then the program needs more than any VM build offers), a build limit the desktop VM does not have, or the VM refuses to load the program | |

Lean builds lack these methods: str `trim repeat replace split to_float`; array
`remove insert reverse sort sum min max copy slice join`; map `values copy
clear`. A method that a lean build lacks only for *some* receivers (`slice` and
`remove` for arrays, `clear` for maps) and methods that may dispatch to your own
struct method of the same name are warnings, not errors (the receiver type is
only known at run time). Lean builds also leave out **array concatenation**
(`[1] + [2]`) and the **map-literal key check** (`map_literal_fixup`: removing
repeated keys, turning `2.0` into `2`, rejecting arrays/maps/structs as keys).
The tables and both differences are checked against a real lean build by
`tests/test_fit.py`.

### Warnings (risks that depend on the input)

* **Recursion** — the depth depends on the input. The report gives the cost of one
  level and how many levels fit: `count calls itself (line 4); each level needs 2
  value-stack slots and 1 call frame: 14 levels fit on this device (limited by call
  frames)`. A *level* is one way around the recursion: `count -> count`,
  `even -> odd -> even` (2 frames), `depth -> .map() -> <lambda> -> depth` (3
  frames: the prelude function and the lambda are frames too). The number is a
  **guarantee**: it assumes every level takes its most expensive way around and the
  topmost level runs everything it can before its recursive call. The tests run the
  predicted number of levels and more on the device builds: it is exact for
  `count`/`factorial`-style recursion, for mutual recursion (`even -> odd -> even`)
  and for `f -> .map() -> f`, and one level low when the deepest level does less than
  the analysis must assume — it returns before pushing the values the other levels
  push for their recursive call (`return n * (n + f(n - 1))`), before entering their
  `retry` block, or before calling the lambda of `.map()`. A recursion in which only
  the outer levels make some expensive call can manage more levels than reported.
  When some cycle of the recursion
  avoids the function it is entered by (`a -> b -> a` and `b -> c -> b`), levels are
  not comparable and the report counts *nested calls* instead, each at the cost of
  the most expensive call.
* **Stack use that grows with the data** — only `args()` puts a value per program
  argument on the value stack before building the array: a warning on devices with a
  command line; the static numbers count the arguments given to `--run`. (TIL 3.0
  VMs also pushed every element for `.split()`, `.join()`, `.keys()`, `.values()`,
  array `.copy()`/`.slice()`, `range(n)` and the keys of a map in `for x in m`; the
  TIL 3.1 VMs build these results in the heap: a constant 1-2 slots, counted
  exactly.)
* **Allocations inside loops** — a loop that keeps what it allocates
  (`push`, `insert`, map item assignment, and `x = x + y` / `x += y` building an
  ever longer string or array, also `s = s * 2`) grows the heap with every iteration;
  loops that only allocate temporaries (f-strings, `"a" + b` that is not kept) are
  listed in a note. `+` and `*` count as allocations only when the value kinds (below)
  show they always give a string or an array.
* **Lean builds**: `+` whose operands may both be arrays when that is decided at run
  time (e.g. `m[0] + m[1]`), and map literals with computed keys (repeated keys,
  `2.0` keys and invalid keys are not checked by the lean VM).
* Little room left (≥ 80 % of a limit), `random()` returning integers on a VM
  without floats, screen/file/input natives on devices without them (but with them in
  the VM build), `timeout`
  and `clock_ms` counting steps on devices without a clock, `x ** y` / `pow()` on a
  `TILVM_NO_MATH` build (integer exponents only: a fractional one gives 0),
  single-precision floats, integer wrap-around on 16/32-bit devices (notes).
* `--run`: the output differs from the desktop VM (integer width, float
  precision, the lean VM), live data close to the heap size, the program failing
  the same way on the desktop VM (its own error, e.g. an index out of bounds — not
  a device limit).

## How the static analysis works

The analysis reads the compiled bytecode (`.tbc`), not the source, so it sees
exactly what the VM executes, including the prelude functions behind
`.map()/.filter()/...`, default arguments, contracts and `retry`.

**The C VM's value stack.** One array of `TILVM_STACK_SIZE` slots holds the
globals, then for every active call its locals and its operand stack. A call
`f(a, b)` leaves `a, b` on the caller's operand stack; they become the callee's
first locals, and `push_frame` allocates the remaining locals and requires 4
more free slots. The analysis models this exactly, including the temporaries
that builtins push inside one instruction (array and map literals keep the new
buffer on the stack; `sum()` pushes two values; `range()`, `.split()`, `.keys()`
and the other builtins that build an array create it empty above their arguments
and fill it in the heap; `args()` pushes every argument). Effect handlers keep
their clauses on this stack too: `HANDLE n` leaves its n (clause, effect name)
pairs on the operand stack under the body, so every open `handle` block counts 2n
slots.

**Per function: abstract interpretation.** Starting at offset 0 with an empty
operand stack, every reachable instruction is visited with the operand-stack
depth, the number of open recovery points and the open `handle` blocks; at jumps
and joins the maximum is kept (a loop whose stack would keep growing is reported,
not followed forever). `FOR_ITER` continues with one more value and exits without
it; `TRY` / `TRY_TIMEOUT` reach their handler with the stack of the `TRY` plus the
error message; `HANDLE n` continues with its pairs on the stack and reaches its end
target (where a clause that finishes without `resume` continues) with the stack
below the pairs plus the clause value; `END_HANDLE` cuts the stack back to the
pairs of the innermost open block and keeps the value; `RETURN`, `RESUME`, `PANIC`
and `HALT` end a path. The stack effect of every plain opcode is one line in
`STACK_EFFECTS` in `src/tilfit.py`:

```python
_O.NEW_ARRAY: E(_arg(0), 1, peak=1, need=2, alloc='array'),   # pops n, pushes 1, ...
```

**Superinstructions** (0x80–0xBF, `docs/TIL_BYTECODE_V2.md`) are one line each in
`SUPERINSTRUCTIONS`: the sequence of plain instructions they stand for. Their
stack effect is that of the sequence, their jump landing and end of flow come from
the compiler's own tables (`til._JUMP_LANDING`, `til._ENDS_FLOW`; a test checks
that the analysis agrees with `til._STACK_EFFECT` for every opcode), and builtins,
casts, constants and the value kinds (below) are followed through the sequence:

```python
_O.INDEX_LL: (lambda a: [(_L, (a[0],)), (_L, (a[1],)), (_O.GET_INDEX, ())], True),
```

The second element says whether the C VM writes the sequence's values to the
stack. Its slow paths do (they push the operands and run the plain handlers);
they are all that `-Os` and `TILVM_LEAN` builds — the device ports — compile. The
compare-and-branch forms (`JUMP_IF_NOT*`), `RETURN_LOCAL` and `FOR_ITER_STORE`
never push their operands. The integer fast paths of `-O2` desktop builds use the
same or fewer slots, so the static numbers stay upper bounds there.

**Over the call graph.** Call edges come from `CALL` (direct), `INVOKE` (struct
methods with that name, the prelude function of a builtin method, and — when a
struct field has that name — any function value), and `CALL_VALUE` (every
function whose address is taken by `PUSH_FUNC` / `MAKE_CLOSURE` in reachable
code, filtered by arity). `xs.map(f)` / `.filter` / `.reduce` / `.each` / `.any` /
`.all` run a prelude function (written in TIL) that calls `f`: the call site gets an
edge to the prelude function and one *through* it to `f` (2 frames deeper, at the
offset where the prelude function calls it) — `f` is the function passed at this
site when the bytecode shows it (a lambda or a named function), else any function
value. So `m.map(|row| row.map(|x| x * 2))` is no recursion. Other indirect calls
are conservative. `PERFORM` is a call too: of the default function (the top-level
function named like the effect) and of the effect clauses (functions with
`FLAG_EFFECT_CLAUSE`, found from the pairs below each `HANDLE`) of the handle blocks
that may be active — the analysis follows, per function, which `handle` blocks can
be active while it runs (those of its callers and those open at the call), searches
the blocks open in the performing function innermost first (a clause there with the
effect's name ends the search, as in the VM) and lets a clause run without its own
handle block (masking). A clause adds one handler entry (the running clause) and one
frame. Clause functions are never taken as function values. The strongly connected
components of this graph are the recursions. Without recursion the worst case of a function is the maximum of its
own needs and, for every call edge, *where the callee's frame starts* + *the
callee's worst case*; the program's worst case is the globals plus the entry
function's. Recursive components are counted once in the totals; the report adds
their cost per level and the number of levels that fit. A level costs the most
expensive simple cycle through the function the recursion is entered by (slots,
frames, recovery points and effect handler entries each summed along the cycle and
maximised on their own); `top` is the most expensive partial level: a simple path
from that function plus what the last function can run before a recursive call
(with its other callees):

```
levels = min( (stack − base − top_need) / slots_per_level + 1,
              (frames − frames_below + 1 − top_frames) / frames_per_level + 1,
              (try − tries_below − top_tries) / tries_per_level + 1,
              (handlers − handlers_below − top_handlers) / handlers_per_level + 1 )
```

(a recursion whose levels each open a `handle` block is limited by
`TILVM_MAX_HANDLERS`: "too many nested handlers").

over every function the recursion is entered by.

**Value kinds.** A second dataflow pass over the same bytecode tracks which kinds
of value (nil, bool, int, float, str, array, map, other) every operand-stack slot
and local may hold at every instruction; parameters get what their call sites
pass, call results what the callees return, globals and struct fields (by name)
what is ever stored in them. It decides the checks where operand types matter:
`+` on arrays in lean builds (an error when one operand is always an array — then
`+` only works as a concatenation —, a warning when both may be arrays), `+` and
`*` that allocate strings or arrays, `for x in m` over a map, and the keys of map
literals. Cells, upvalues, indexing results and loop elements are "any".

**Exactness.** For programs without recursion and without data-dependent
builtins, the static numbers (value stack, call frames, recovery points, effect
handler entries) equal the measured high-water marks of the device VM builds of
`--run` *exactly* when every path runs — `tests/test_fit.py` checks this for all
examples (fused and with `--no-fuse`) and a corpus of programs (closures, structs,
maps, retry/intent, match, Option/`?`, default arguments, comprehensions, globals,
higher-order methods, nested `.map()`, array `+`, nested handlers, aborting and
resuming clauses, clauses with `retry`, generators, default functions). Otherwise
the static numbers are upper bounds (paths that are not taken, conservative
indirect calls, handle blocks that may but need not be active; a `retry` handler
is counted with the error message on the stack). For recursion the number of
levels is a guarantee: exact for the common shapes, otherwise usually one level
low (see *Recursion* above).

**Not checked statically**: how much heap a program needs (it depends on the
data — use `--run`), integer overflow at run time (only constants are checked;
see the wrap-around note and `--run`'s output comparison), and allocations in
functions *called* from a loop (only allocation sites directly in the loop are
reported).

### Unknown opcodes

An opcode without an entry in `STACK_EFFECTS` — a number no instruction uses
(0x8B–0x8F, 0x95–0x9F, 0xC0–0xFF), an opcode added to the compiler but not yet to
`til fit`, or a corrupt file — never crashes the analysis: the function is marked
*incomplete*, the stack rows show `?`, and the verdict is
`does not fit: analysis incomplete (unknown opcode 0xC0 at line 3 in main)`. So is
a superinstruction with a malformed operand (a compare-and-branch whose comparison
byte is not `EQ`..`GE`: the VMs stop with "bad opcode"). Builtins and calls after
it are still collected by a linear sweep as long as the operand layout is known
(`til.OP_OPERANDS`). Adding an opcode is one line: a plain one in
`STACK_EFFECTS`, e.g. `ADD_I8 n` (pops 1, pushes 1)

```python
_O.ADD_I8: E(1, 1),
```

and a superinstruction in `SUPERINSTRUCTIONS` (its sequence). A test fails when
the compiler has an opcode `til fit` does not know, or when the two disagree on a
stack effect or a jump landing.

## `--run`: measuring on the device's VM

`--run` compiles `vm/tilvm.c` + `vm/main.c` with the device's `-D` flags (without
the device compiler's `TILVM_ROM` qualifier) plus `-DTILVM_STATS`,
`-DTILVM_FAST_PATHS=0` (and `-DTILVM_PACK` for 8-bit targets), caches the binary in
`~/.til/cache` (or `$TIL_FIT_CACHE`; keyed by the VM sources, the flags and the
compiler), runs the program and reads the high-water marks. Without a usable home
directory the cache is `<temp>/til-cache-<uid>`, created with mode 0700 and used
only when it belongs to the user and nobody else can write to it (else a new
private temporary directory): cached executables are run, so they never come from
a shared, predictable place. It also runs the program on the desktop VM (the
`desktop` profile, built the same way) and compares output and exit code
(skipped when the program uses `clock_ms`, `sleep_ms`, keys, or `random` on a
device whose random generator differs, and after a device run that hit
`--timeout`).

* **Faithful device builds.** `TILVM_NO_STDIO` / `TILVM_NO_TIME` stay on, so number
  formatting and step-counted timeouts are those of the device; `vm/main.c`
  gives such a core a console (stdout/stderr/stdin). No file hooks are
  installed, as on the devices. `TILVM_LEAN`, `TILVM_ADDR16`, `TILVM_NO_EFFECTS`,
  `TILVM_NO_GFX` and the screen size are the device's; the desktop runner provides
  the screen and keys hooks (a device without a display is a warning of the static
  check). Without the integer fast paths (`TILVM_FAST_PATHS=0`) the interpreter runs
  the code the device ports compile (`-Os`, lean): the superinstructions' slow paths,
  which write the stack slots the static analysis counts.
* **8-bit layouts on the desktop.** On AVR, Z80 and eZ80 a value takes 3 or 5
  bytes (no alignment); on the desktop 4 or 8. `-DTILVM_PACK` packs `tv_val` and
  the heap object headers so heap objects have the device's sizes and measured
  heap use matches the device (checked against avr-gcc / SDCC by the tests; with
  a heap above 64 KB the packed headers also keep 32-bit fields at 2-byte
  alignment well-defined — a UBSan test covers it).
* **Errors** that come from the VM's limits (`out of memory`, `stack overflow`,
  `recursion too deep`, `too many nested retry/timeout blocks`, `too many nested
  handlers`, `range too large`, `array too large`, `string too long`) make the
  verdict `does not fit` — also when the desktop VM stops the same way (the program
  needs more than any VM build offers). Errors from the device's build (`not
  available`, `no method` in a lean build, effect handlers or screen natives left
  out, no floats, no `<math.h>`, a screen larger than the device's) are `does not
  fit` unless the desktop VM fails the same way. Any other error the desktop VM reports too (an index out of
  bounds, a failed `assert`) is the program's own and only a warning. A VM killed
  by a signal does not fit.

### C VM API (`TILVM_STATS`)

```c
#define TILVM_STATS             /* or -DTILVM_STATS */
#include "tilvm.h"

tilvm_stats s;
tilvm_get_stats(&vm, &s);
/* s.peak_stack   value-stack slots ever used (globals + locals + temporaries)
   s.peak_frames  deepest call
   s.peak_try     most nested retry/timeout/intent blocks
   s.peak_heap    arena high-water mark (heap_top), bytes
   s.peak_live    most heap in use right after a collection, bytes
   s.peak_handlers most effect handler entries (open handle blocks + running clauses) */
```

The counters live in `alloc_obj`, `gc`, `push_frame`, `push_try`, `HANDLE` and
`effect_target` (a perform that runs a clause); the stack
high-water mark is found without touching the interpreter loop: `tilvm_load`
paints the free stack slots with an invalid tag and `tilvm_get_stats` finds the
highest slot ever written (this also catches temporaries pushed inside
builtins). Without `TILVM_STATS` the VM is unchanged. `vm/tilvm --stats` prints
a second line in such builds:

```
[tilvm] steps=319 gc=0 heap=136/262144 bytes
[tilvm] peak stack=22/8192 frames=12/1000 try=0/32 heap=136/262144 live=0 handlers=0/32
```

## The VM's own RAM

`sizeof(tilvm)` depends on the profile *and* on the target's ABI (an AVR value is
5 bytes, a Cortex-M value 8 or 16). `src/tilfit.py` contains a layout model of the
structs in `vm/tilvm.h` (`vm_struct_layout`) for five ABIs: `lp64` (desktops),
`ilp32` (32-bit ARM, Xtensa, RISC-V, SH4), `avr`, `z80` (SDCC) and `ez80` (CE
toolchain: 24-bit pointers). The tests compare it with the host compiler for
every profile (and for builds with each of the switches `TILVM_NO_EFFECTS`,
`TILVM_MAX_HANDLERS`, `TILVM_MAX_FUNCS`, `TILVM_FIELD_CACHE`, `TILVM_ADDR16`,
`TILVM_GFX_MAX_W/H`) and, when installed, with avr-gcc, arm-none-eabi-gcc and SDCC.

The model is a table, `VM_STRUCTS`, that mirrors the header struct by struct and
member by member (type, array length, `#if` condition):

```python
_f('n_handlers', 'u16'), _f('_hpad', 'u16'),
_f('handlers', 'tv_handler', 'max_handlers', when=_NO_EFFECTS),
```

**When `vm/tilvm.h` changes, add the member at the same place in `VM_STRUCTS`** (a
new state byte is one line, e.g. `_f('energy_mode', 'u8')`; a new build macro also
gets a device field) — the host test fails until then. A RAM error lists the
biggest parts of the struct (heap, value stack, function table, constant table,
call frames, effect handlers, lookup caches); `--json` has them under
`vm.struct_parts`.

## JSON report

`--json` prints one object:

```json
{
  "program": "game.til", "fits": false, "verdict": "does not fit: ...",
  "errors": ["..."], "reasons": ["short form of each error"], "warnings": ["..."], "notes": ["..."],
  "device": {"name": "z80-16", "int_bits": 16, "float": 0, "heap": 4096, "...": "..."},
  "resources": [{"key": "stack", "resource": "value stack", "needs": "52 slots", "limit": "192",
                 "status": "ok", "measured": "47", "need_value": 52, "limit_value": 192, "note": ""}],
  "static": {"bytecode_bytes": 2952, "constants": 78, "types": 8, "globals": 1, "functions": 18,
             "reachable_functions": 18, "complete": true, "problems": [],
             "stack_slots": 52, "stack_slots_written": 50, "call_frames": 4, "recovery_points": 2,
             "effect_handlers": 3, "handle_blocks": 2, "debug_info": false,
             "recursion": [{"functions": ["fib"], "at": "line 32 in fib", "unit": "levels",
                            "cycle": "fib -> fib", "slots_per_level": 2, "frames_per_level": 1,
                            "tries_per_level": 0, "handlers_per_level": 0, "levels": 22,
                            "limited_by": "call frames"}],
             "uses_float_constants": true, "functions_detail": [{"name": "main", "...": "..."}]},
  "vm": {"struct_bytes": 5840, "value_bytes": 3, "struct_parts": {"heap": 4096, "...": 0},
         "c_flags": ["-DTILVM_INT_BITS=16", "..."]},
  "run": null
}
```

`stack_slots` is what the VM requires (including the free slots it checks for),
`stack_slots_written` what it uses; both are `null` when the analysis is
incomplete. `effect_handlers` is the worst case of handler entries (open handle
blocks + running clauses), `handle_blocks` the number of reachable `handle`
expressions, `debug_info` whether the file has the debug-info section. In
`recursion`, `unit` is `levels` or `calls` (see *Recursion*) and `cycle` the most
expensive level. With `--run`, `run` holds `exit_code`, `stdout`, `errors`,
`error`, `timeout`, `stats` (`steps`, `gc`, `heap`, `peak_stack`, `peak_frames`,
`peak_try`, `peak_heap`, `peak_live`, `peak_handlers`) and `desktop` (`exit_code`,
`error`, `same_output`; absent when the desktop run was skipped).

## The debug-info section

`til build --debug-info` (and `til debug`) append variable names to the bytecode
after the line table (`docs/TIL_DEBUGGER.md`); the VMs ignore them. `til fit`
reads such files, counts the section in the bytecode size (it is stored on the
device like the rest) and notes that `--strip` removes it together with the line
tables; for a `.tbc` input that does not fit the storage, the error says how many
bytes `--strip` would save.

## `til build --target vm --device NAME`

After compiling, the build runs the same static check. When the program fits it
prints a one-line summary (and the warnings) and writes the `.tbc`; when it does
not fit it prints the full report to stderr, writes nothing and exits with 1 —
like a linker whose memory region overflows. `--strip` is taken into account.
`--target c-vm --device NAME` checks the same way (compile the generated C file
with the device's flags; `til fit --json` lists them under `vm.c_flags`).
`--device` with any other target or command, or with a `.tbc` input (use `til fit
prog.tbc --device NAME`), is an error.

## Tests

`tests/test_fit.py`: profiles (the flags of the port builds) and the struct model
(host compiler, every build switch; avr-gcc, arm-none-eabi-gcc and SDCC when
installed), the opcode tables against the compiler's, the analysis on hand-made and
patched bytecode, static bounds against measured high-water marks (exact on the
corpus, fused and unfused, and on effect handlers), recursion levels against the
device builds (also limited by handler entries), `TILVM_NO_EFFECTS` / `TILVM_NO_GFX`
/ screen size / function table / `TILVM_ADDR16` verdicts, the debug-info section,
the value kinds,
verdicts, `--run` (device limits, timeouts, signals, `--stdin`, program
arguments), the lean tables and lean differences against a real lean build, the
C VM macros (strict builds, packed layout incl. a UBSan run with a big packed
heap, console runner), the private VM cache and `til build --device`. Opt-in:
`TIL_TEST_SLOW=1` (every device VM built and run, `--timeout` timing) and
`TIL_TEST_CROSS=1` (linked ports).

## Limitations

* The heap is not bounded statically; `--run` measures it for one input.
* Integer overflow at run time is not detected statically (constants are).
* Allocation warnings cover allocation sites directly inside a loop, not those
  in functions called from it.
* The value kinds are coarse: struct fields are tracked by name across all struct
  types, cells, upvalues, indexing results and loop elements are "any"; checks that
  depend on them say "may" when the kinds are not certain.
* Which `handle` blocks are active is followed per function, not per call path: a
  function called both inside and outside a handle block counts the clauses it may
  run in either case (an upper bound), and a clause is assumed to see the handle
  blocks opened inside its own block's body (the VM masks them).
* Recursion levels are a guarantee, not always the exact maximum (see *Recursion*).
* The profiles marked as assumptions above (TI-84 CE, NumWorks, Casio, ESP32,
  RP2040 RAM shares) have not been measured on hardware; `z80-16` is the simulated
  Z80 of `vm/ports/z80`.
* `--run` runs on the desktop: speed, real clocks and device I/O are not
  emulated (timeouts count steps on devices without a clock, as on the device).
