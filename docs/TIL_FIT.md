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
  bytecode         946 B                          72 KB    ok
  VM RAM           12.7 KB                        16 KB    ok  VM struct 7.2 KB + VM tables 3.1 KB + C stack 1.5 KB + bytecode 946 B
  constants        22                             160      ok
  struct types     3                              24       ok
  globals          0                              152      ok
  integers         16-bit                         32-bit   ok
  floats           none                           float32  ok
  builtins         1 native, 0 methods            full     ok
  value stack      11 slots             23        160      ok  +2 per recursion level
  call frames      3                    12        24       ok  +1 per recursion level
  recovery points  0                    0         6        ok
  heap             depends on data      94 B      5 KB     -

warnings:
  - recursion: factorial calls itself (line 26 in factorial); the depth depends on the
    input. Each level needs 2 value-stack slots and 1 call frame: at most 22 levels fit on
    this device (limited by call frames)
  - recursion: fib calls itself (line 32 in fib); ...

notes:
  - integers are 32-bit on this device: results beyond +/-2147483647 wrap around

run: exit code 0, 319 steps, 0 garbage collections

verdict: fits
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
| `-- ARGS` | `--run`: program arguments (`args()`) |

The input may be a `.til` source or a compiled `.tbc` file.

## Device profiles

Every profile is a build configuration of the C VM (the `TILVM_*` macros of
`vm/tilvm.h`) plus what the device offers around it.

| device | int | float | heap | stack slots | frames | consts / types | try | lean | bytecode storage | RAM for the VM | VM struct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `desktop` | 64 | double | 256 KB | 8192 | 1000 | 4096 / 512 | 32 | no | — | — | 423 KB |
| `ti84ce` | 32 | float | 32 KB | 1024 | 128 | 1024 / 128 | 16 | no | 65505 B | 64 KB | 44.0 KB |
| `numworks` | 32 | double | 32 KB | 1024 | 200 | 1024 / 128 | 16 | no | 1 MB | 96 KB | 56.2 KB |
| `nspire` | 64 | double | 1 MB | 16384 | 2000 | 8192 / 1024 | 32 | no | 16 MB | 16 MB | 1.3 MB |
| `casio-cg50` | 32 | double | 64 KB | 2048 | 256 | 2048 / 256 | 16 | no | 2 MB | 512 KB | 111 KB |
| `cortex-m3-64k` | 32 | float | 16 KB | 512 | 64 | 512 / 64 | 8 | no | 208 KB | 64 KB | 23.8 KB |
| `atmega1284` | 32 | float | 5 KB | 160 | 24 | 160 / 24 | 6 | no | 72 KB | 16 KB | 7.2 KB |
| `arduino-mega` | 16 | float | 1.5 KB | 96 | 16 | 96 / 16 | 4 | **yes** | 212 KB | 8 KB | 3.0 KB |
| `esp32` | 32 | float | 64 KB | 2048 | 256 | 2048 / 256 | 16 | no | 1 MB | 160 KB | 94.7 KB |
| `rp2040` | 32 | float | 128 KB | 2048 | 256 | 2048 / 256 | 16 | no | 1 MB | 240 KB | 159 KB |
| `z80-16` | 16 | none | 4 KB | 192 | 32 | 192 / 24 | 8 | **yes** | 8 KB | 7 KB | 6.2 KB |

*VM struct* is `sizeof(tilvm)` for the device's compiler (see *The VM's own RAM*).
All profiles except `desktop` and `nspire` build the VM with `TILVM_NO_STDIO`
and `TILVM_NO_TIME` (the port provides output and, where noted, a clock).

### Sources and assumptions

* **Verified profiles.** `cortex-m3-64k` and `atmega1284` use exactly the flags of
  `vm/ports/cortex-m/Makefile` and `vm/ports/avr/Makefile` (a test checks this);
  both ports run all examples in QEMU / simavr. Measured with
  arm-none-eabi-gcc 13.2 `-Os` + newlib-nano: VM code 46.9 KB of the 256 KB flash
  (storage = 256 − 48 KB), 80 bytes of other static data; with avr-gcc 7.3 `-Os`:
  VM code 55.5 KB of the 128 KB flash (storage = 128 − 56 KB).
* **AVR keeps constant data in RAM** (Harvard architecture, no `PROGMEM` in the
  VM): the VM's string tables and error messages take 3130 bytes of RAM (full
  build) or 2480 bytes (lean build), measured as `.data` minus the program with
  avr-gcc (`TIL_TEST_CROSS=1 pytest tests/test_fit.py` links the AVR and Cortex-M
  ports and compares their `.data`/`.bss` with the model). The AVR port also
  copies the bytecode into RAM (`program.h`), so on
  `atmega1284` and `arduino-mega` the bytecode counts against RAM, not only
  against flash — on the ATmega1284P that leaves about 4.3 KB for bytecode.
* **`arduino-mega`** (ATmega2560: 8 KB RAM, 256 KB flash): a lean 16-bit build in
  the style of `make tiny`, sized to leave ~1 KB of RAM for bytecode: 1.5 KB
  heap, 96 stack slots, 16 frames. Floats are single precision (avr-gcc's
  `double` is 32-bit). VM code 42 KB (lean, avr-gcc `-Os`).
* **`z80-16`** follows `vm/ports/z80` with 16-bit integers (§9 of the TIL 3.1
  design): VM data at `0xC000-0xDBFF` (7 KB), C stack above the output buffer
  (not counted). The **storage of 8 KB is an assumption** — the 16-bit lean VM is
  expected to leave about 8 KB of the 48 KB code area for bytecode; update it
  when `vm/ports/z80/RESULTS.md` has the measurement.
* **`ti84ce`**: TI-84 Plus CE (eZ80, 154 KB of RAM for programs). Heap ≤ 32 KB and
  `TILVM_FLOAT=1` as recommended by `vm/README.md` (the CE toolchain's `double`
  is 32-bit); the bytecode is stored in one AppVar (at most 65505 bytes); 64 KB of
  RAM for the VM is an assumption that leaves room for the VM code, which also
  runs from RAM on the CE.
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
* **Device services**: `screen` (calculators, desktop), `files` (desktop, nspire),
  `console` for `input()`, and `clock` (the port provides `clock_ms`; without it
  `timeout(ms)` and `clock_ms()` count VM steps, 1 ms = 1000 steps — true for the
  reference ports `cortex-m`, `avr` and `z80`).

### Custom devices

`--device board.json` takes a JSON object with any of the fields above, based on
a built-in profile:

```json
{"base": "cortex-m3-64k", "name": "my-board", "heap": 32768, "stack": 1024, "ram": 131072}
```

Fields: `int_bits`, `float`, `heap`, `stack`, `frames`, `consts`, `types`,
`try_depth`, `mark_stack`, `lean`, `no_stdio`, `no_time`, `no_math`, `storage`,
`ram`, `c_stack`, `vm_static_ram`, `bytecode_in_ram`, `abi` (`lp64`, `ilp32`,
`avr`, `z80`, `ez80`), `clock`, `screen`, `files`, `console`, `title`, `cpu`,
`source`. Unknown fields and impossible values are errors.

## What is checked

### Errors (the verdict becomes `does not fit`)

| check | why it matters on the device |
|---|---|
| bytecode size > storage | the program cannot be stored (`--strip` saves the line tables) |
| VM RAM > device RAM | `sizeof(tilvm)` + VM tables + C stack (+ bytecode on AVR) |
| constants > `TILVM_MAX_CONSTS`, struct types > `TILVM_MAX_TYPES` | `tilvm_load` fails (error -4 / -7) |
| globals > stack slots − 8 | `tilvm_load` fails (error -8) |
| an integer constant wider than the device's integers | `tilvm_load` fails (error -6); the report names the constant and its line |
| float constants on a VM without floats | `tilvm_load` fails (error -3) |
| float conversions (`float()`, `: float` / `-> float`, `as float`) and `sqrt/sin/cos/tan/log/exp/floor/ceil/round/pow` on a VM without floats | runtime error ("built without floating point", "not available") |
| `sin/cos/tan/log/exp` with `TILVM_NO_MATH` | runtime error |
| builtins a lean build leaves out (`input chr ord type_of clock_ms random seed sleep_ms read_file write_file file_exists args bytes_to_str str_bytes`, and the methods below) | runtime error ("not available", "no method") |
| f-string format specs / `format()` on a lean build | the lean VM ignores specs: the output would differ |
| value stack, call frames, recovery points over the limit (worst case) | "stack overflow", "recursion too deep", "too many nested retry/timeout blocks" |
| recursion where not even two levels fit | |
| analysis incomplete (unknown opcode, malformed code) | the stack use cannot be verified |
| `--run`: the run stops with a device error, or the VM refuses to load the program | |

Lean builds lack these methods: str `trim repeat replace split to_float`; array
`remove insert reverse sort sum min max copy slice join`; map `values copy
clear`. A method that a lean build lacks only for *some* receivers (`slice` and
`remove` for arrays, `clear` for maps) and methods that may dispatch to your own
struct method of the same name are warnings, not errors (the receiver type is
only known at run time). The tables are checked against a real lean build by
`tests/test_fit.py` (the comment in `vm/tilvm.h` also lists str `count`, which the
lean build does have).

### Warnings (risks that depend on the input)

* **Recursion** — the depth depends on the input. The report gives the cost of one
  level and how many levels fit: `count calls itself (line 4); each level needs 2
  value-stack slots and 1 call frame: at most 14 levels fit on this device
  (limited by call frames)`. The number is exact for the device's VM (a test
  runs N and N+1 levels on the device build).
* **Stack use that grows with the data** — some builtins put a whole collection
  on the value stack before building the result: `.split()`, `.join()`,
  `.keys()`, `.values()`, array `.copy()`/`.slice()`, `range(n)` with a
  non-constant `n`, and `for x in m` over a map. `range(n)` with constant bounds
  is counted exactly (and `for i in a..b` needs no stack at all).
* **Allocations inside loops** — a loop that keeps what it allocates
  (`push`, `insert`, map item assignment) grows the heap with every iteration;
  loops that only allocate temporaries are listed in a note.
* Little room left (≥ 80 % of a limit), `random()` returning integers on a VM
  without floats, screen/file/input natives on devices without them, `timeout`
  and `clock_ms` counting steps on devices without a clock, single-precision
  floats, integer wrap-around on 16/32-bit devices (notes).
* `--run`: the output differs from the desktop VM (integer width, float
  precision), live data close to the heap size, the program failing on the
  desktop VM too (not a device limit).

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
buffer on the stack; `sum()` pushes two values; `range()` pushes every element).

**Per function: abstract interpretation.** Starting at offset 0 with an empty
operand stack, every reachable instruction is visited with the operand-stack
depth and the number of open recovery points; at jumps and joins the maximum is
kept (a loop whose stack would keep growing is reported, not followed forever).
`FOR_ITER` continues with one more value and exits without it; `TRY` /
`TRY_TIMEOUT` reach their handler with the stack of the `TRY` plus the error
message; `RETURN`, `PANIC` and `HALT` end a path. The stack effect of every
opcode is one line in `STACK_EFFECTS` in `src/tilfit.py`:

```python
_O.NEW_ARRAY: E(_arg(0), 1, peak=1, need=2, alloc='array'),   # pops n, pushes 1, ...
```

**Over the call graph.** Call edges come from `CALL` (direct), `INVOKE` (struct
methods with that name, the prelude function of a builtin method, and — when a
struct field has that name — any function value), and `CALL_VALUE` (every
function whose address is taken by `PUSH_FUNC` / `MAKE_CLOSURE` in reachable
code, filtered by arity). Indirect calls are therefore conservative. The
strongly connected components of this graph are the recursions. Without
recursion the worst case of a function is the maximum of its own needs and, for
every call site, *where the callee's frame starts* + *the callee's worst case*;
the program's worst case is the globals plus the entry function's. Recursive
components are counted once in the totals; the report adds their cost per level
and the number of levels that fit:

```
levels = min( (stack − base − top) / slots_per_level + 1,
              frames − frames_below − top_frames + 2,
              (try − tries_below − top_tries) / tries_per_level + 1 )
```

**Exactness.** For programs without recursion and without data-dependent
builtins, the static numbers equal the measured high-water marks of the C VM
*exactly* when every path runs — `tests/test_fit.py` checks this for all
examples and a corpus of programs (closures, structs, maps, retry/intent,
match, Option/`?`, default arguments, comprehensions, globals, higher-order
methods). Otherwise the static numbers are upper bounds (paths that are not
taken, conservative indirect calls); for recursion the per-level formula is an
upper bound that is at most one slot above the measurement in the tests.

**Not checked statically**: how much heap a program needs (it depends on the
data — use `--run`), integer overflow at run time (only constants are checked;
see the wrap-around note and `--run`'s output comparison), and allocations in
functions *called* from a loop (only allocation sites directly in the loop are
reported).

### Unknown opcodes

An opcode without an entry in `STACK_EFFECTS` — a superinstruction (0x80–0xBF),
the effect opcodes `HANDLE/END_HANDLE/PERFORM/RESUME` (0x76–0x79) until they get
their entries, or a corrupt file — never crashes the analysis: the function is
marked *incomplete*, the stack rows show `?`, and the verdict is
`does not fit: analysis incomplete (unknown opcode 0xB0 at line 3 in main)`.
Builtins and calls after it are still collected by a linear sweep as long as
the operand layout is known (`til.OP_OPERANDS`). Adding an opcode is one line,
e.g. for a superinstruction `ADD_I8 n` (pops 1, pushes 1):

```python
_O.ADD_I8: E(1, 1),
```

## `--run`: measuring on the device's VM

`--run` compiles `vm/tilvm.c` + `vm/main.c` with the device's `-D` flags plus
`-DTILVM_STATS` (and `-DTILVM_PACK` for 8-bit targets), caches the binary in
`~/.til/cache` (or `$TIL_FIT_CACHE`; keyed by the VM sources, the flags and the
compiler), runs the program and reads the high-water marks. It also runs the
program on the normal desktop VM and compares output and exit code (skipped
when the program uses `clock_ms`, `sleep_ms`, keys, or `random` on a device whose
random generator differs).

* **Faithful device builds.** `TILVM_NO_STDIO` / `TILVM_NO_TIME` stay on, so number
  formatting and step-counted timeouts are those of the device; `vm/main.c`
  gives such a core a console (stdout/stderr/stdin). No file hooks are
  installed, as on the devices.
* **8-bit layouts on the desktop.** On AVR, Z80 and eZ80 a value takes 3 or 5
  bytes (no alignment); on the desktop 4 or 8. `-DTILVM_PACK` packs `tv_val` so
  heap objects have the device's sizes and measured heap use matches the device
  (checked against avr-gcc / SDCC by the tests).
* **Errors** that come from the device's limits (`out of memory`, `stack
  overflow`, `recursion too deep`, `too many nested retry/timeout blocks`,
  `range too large`, `not available`, `no method` in a lean build, no floats...)
  make the verdict `does not fit`. An error the desktop VM reports too is the
  program's own and only a warning.

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
   s.peak_live    most heap in use right after a collection, bytes */
```

The counters live in `alloc_obj`, `gc`, `push_frame` and `push_try`; the stack
high-water mark is found without touching the interpreter loop: `tilvm_load`
paints the free stack slots with an invalid tag and `tilvm_get_stats` finds the
highest slot ever written (this also catches temporaries pushed inside
builtins). Without `TILVM_STATS` the VM is unchanged. `vm/tilvm --stats` prints
a second line in such builds:

```
[tilvm] steps=319 gc=0 heap=136/262144 bytes
[tilvm] peak stack=23/8192 frames=12/1000 try=0/32 heap=136/262144 live=0
```

## The VM's own RAM

`sizeof(tilvm)` depends on the profile *and* on the target's ABI (an AVR value is
5 bytes, a Cortex-M value 8 or 16). `src/tilfit.py` contains a layout model of the
structs in `vm/tilvm.h` (`vm_struct_layout`) for five ABIs: `lp64` (desktops),
`ilp32` (32-bit ARM, Xtensa, RISC-V, SH4), `avr`, `z80` (SDCC) and `ez80` (CE
toolchain: 24-bit pointers). The tests compare it with the host compiler for
every profile and, when installed, with avr-gcc, arm-none-eabi-gcc and SDCC.
**When `vm/tilvm.h` changes, update `vm_struct_layout()`** — the host test fails
until then.

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
             "recursion": [{"functions": ["fib"], "at": "line 32 in fib", "slots_per_level": 2,
                            "frames_per_level": 1, "tries_per_level": 0, "levels": 22,
                            "limited_by": "call frames"}],
             "uses_float_constants": true, "functions_detail": [{"name": "main", "...": "..."}]},
  "vm": {"struct_bytes": 6400, "value_bytes": 3, "c_flags": ["-DTILVM_INT_BITS=16", "..."]},
  "run": null
}
```

`stack_slots` is what the VM requires (including the free slots it checks for),
`stack_slots_written` what it uses; both are `null` when the analysis is
incomplete. With `--run`, `run` holds `exit_code`, `stdout`, `errors`, `error`,
`stats` (`steps`, `gc`, `heap`, `peak_stack`, `peak_frames`, `peak_try`,
`peak_heap`, `peak_live`) and `desktop` (`exit_code`, `error`, `same_output`).

## `til build --target vm --device NAME`

After compiling, the build runs the same static check. When the program fits it
prints a one-line summary (and the warnings) and writes the `.tbc`; when it does
not fit it prints the full report to stderr, writes nothing and exits with 1 —
like a linker whose memory region overflows. `--strip` is taken into account.
`--target c-vm --device NAME` checks the same way (compile the generated C file
with the device's flags; `til fit --json` lists them under `vm.c_flags`).
`--device` with any other target or command is an error.

## Tests

`tests/test_fit.py`: profiles and the struct model (host compiler; avr-gcc,
arm-none-eabi-gcc and SDCC when installed), the analysis on hand-made and
patched bytecode, static bounds against measured high-water marks (exact on
the corpus), recursion levels against the device builds, verdicts, `--run`,
the lean tables against a real lean build, the C VM macros (strict builds,
packed layout, console runner) and `til build --device`. Opt-in:
`TIL_TEST_SLOW=1` (every device VM built and run) and `TIL_TEST_CROSS=1`
(linked ports).

## Limitations

* The heap is not bounded statically; `--run` measures it for one input.
* Integer overflow at run time is not detected statically (constants are).
* Allocation warnings cover allocation sites directly inside a loop, not those
  in functions called from it.
* The profiles marked as assumptions above (TI-84 CE, NumWorks, Casio, ESP32,
  RP2040 RAM shares; Z80 storage) have not been measured on hardware.
* `--run` runs on the desktop: speed, real clocks and device I/O are not
  emulated (timeouts count steps on devices without a clock, as on the device).
