# TIL VM — run TIL on calculators, microcontrollers and anything with a C compiler

`tilvm.c` + `tilvm.h` are a complete virtual machine for TIL bytecode in portable
C99:

* **no `malloc`** — the whole VM is one struct; the host decides where it lives
* **bytecode stays in ROM/flash** — string constants are never copied to RAM
* **fixed arena + compacting garbage collector** — no fragmentation on tiny heaps
* **configurable** — 16/32/64-bit integers, no floats / `float` / `double`,
  heap, stack and call-depth sizes
* **pluggable I/O** — `write`, `read_line`, `clock_ms` callbacks
* **snapshots** — pause a running program, save it, resume it after a power cycle
* **portable snapshots** — save a paused program as a device-independent `.tilp`
  stream and continue it on another VM build or the reference VM: start on a PC,
  continue on a calculator ([docs/TIL_SNAPSHOTS.md](../docs/TIL_SNAPSHOTS.md))
* **step budgets** — run N steps at a time (UI loops, cooperative multitasking),
  sandbox limits, real `timeout(ms)` blocks
* **screen and keys** — lines, rectangles, circles and a 5×7 font are drawn
  inside the VM (identical pixels on every host); a device only stores pixels
  and reports keys ([docs/TIL_GRAPHICS.md](../docs/TIL_GRAPHICS.md))

The same program produces byte-identical output on the reference VM
(`til run --vm`), the desktop C VM and the embedded builds below — this is
checked by `tests/test_v3.py` and by simulator runs.

## Quick start (desktop)

```bash
make -C vm                          # builds vm/tilvm
python3 src/til.py build --target vm examples/03_functions.til     # -> 03_functions.tbc
vm/tilvm 03_functions.tbc
```

Or let the compiler produce **one self-contained C file** (VM + program) that
any C99 compiler can build — the easiest way to get a program onto a device:

```bash
python3 src/til.py build --target c-vm game.til        # -> game_vm.c
cc -O2 -o game game_vm.c -lm                           # or your device toolchain
```

## Verified targets

All runs below execute the unmodified `tilvm.c` on a simulated CPU, and the
output of every example is compared with the reference VM.

| Target | CPU / board | Simulator | Profile | Code | RAM | Result |
|---|---|---|---|---|---|---|
| ARM Cortex-M (NumWorks/STM32 class) | Cortex-M3, 64 KB RAM | QEMU `lm3s6965evb` | int32, float32, 16 KB heap | 48.4 KB² | 24 KB | 17/17 programs identical¹ |
| 8-bit AVR (Arduino Mega class) | ATmega1284P, 16 KB RAM | simavr | int32, float32, 5 KB heap | 60.2 KB² | 11.2 KB | 17/17 programs identical |
| Z80 (TI-83/84 class) | Z80, 64 KB address space | ucsim `sz80` | int16, lean, 16-bit offsets, 4 KB heap, no effects / screen / energy saving³ | 47.1 KB (SDCC 4.2), ends at 0xBF31 (below 0xC000) | 5.7 KB + ≤1.4 KB C stack | 10/10 int16-compatible examples + test corpus identical³ — see `ports/z80/RESULTS.md` |
| Desktop | x86-64 | native | int64, double, 256 KB heap | 80 KB | ~400 KB | all 255 + 81 tests |

¹ Single-precision builds print the last digit of some floats differently
(`78.5397` vs `78.5398`), as expected for `float`.
² TIL 3.1 builds with the screen/keys natives (about 2.7 KB of ARM code, 7 KB
of AVR code; `-DTILVM_NO_GFX` leaves them out). Measured with `examples/03_functions.til`.
³ 8 examples use floats and 2 compute values beyond 16 bits (`10!`); they are
rejected at load or wrap, as documented for 16-bit builds. The Z80 build leaves
out effect handlers, the screen/keys natives and energy saving (`-DTILVM_NO_EFFECTS
-DTILVM_NO_GFX -DTILVM_NO_ENERGY`, about 1.5 KB, 7.5 KB and 0.8 KB of Z80 code):
with them the code would not fit below 0xC000.

Reproduce:

```bash
make -C vm/ports/cortex-m PROGRAM=../../../examples/03_functions.til run
python3 vm/ports/avr/run_sim.py examples/03_functions.til
```

## Configuration

Define before including `tilvm.h` (or pass `-D...`):

| Macro | Default | Meaning |
|---|---|---|
| `TILVM_INT_BITS` | 64 | 16, 32 or 64-bit integers (constants that do not fit are rejected at load) |
| `TILVM_FLOAT` | 2 | 0 = no floats (programs using floats are rejected), 1 = `float`, 2 = `double` |
| `TILVM_HEAP_SIZE` | 262144 | arena bytes (≤ 65535 switches to 16-bit references) |
| `TILVM_STACK_SIZE` | 8192 | value-stack slots (globals + locals + temporaries) |
| `TILVM_MAX_FRAMES` | 1000 | call depth |
| `TILVM_MAX_TRY` | 32 | nested `retry`/`timeout`/`intent` blocks |
| `TILVM_MAX_HANDLERS` | 32 | active effect handlers + running clauses (`handle ... with`, 16 bytes each; [docs/TIL_EFFECTS.md](../docs/TIL_EFFECTS.md)) |
| `TILVM_NO_EFFECTS` | off | leave effect handlers out (~1–3 KB of code): `handle` reports "not available", `perform` still calls the default function (the Z80 build uses it) |
| `TILVM_MAX_CONSTS` / `TILVM_MAX_TYPES` | 4096 / 512 | per-program table sizes |
| `TILVM_NO_STDIO` | off | no `<stdio.h>`: the host sets `write` (float formatting never uses stdio: see [Floats](#floats)) |
| `TILVM_NO_TIME` | off | no `<time.h>`: `clock_ms()`/`timeout` count 1 ms per 1000 steps |
| `TILVM_NO_MATH` | off | no `<math.h>`: built-in exact `floor`/`ceil`/`%`, correctly rounded `sqrt`, `pow` with integer exponents; `sin`/`cos`/`tan`/`log`/`exp` and `pow` with a fractional exponent are runtime errors (see [Floats](#floats)) |
| `TILVM_ADDR16` | off | 16-bit address space (Z80, AVR): bytecode offsets and object lengths are 16-bit (needs heap ≤ 65535; programs above 64 KB are rejected with load error -10). Much smaller code on 8-bit CPUs |
| `TILVM_LEAN` | off | smallest code for 8-bit CPUs: the whole language core, without format specs, snapshots, `input/chr/ord/type_of/clock_ms/random/sleep_ms`, str `split/replace/trim/repeat/count/to_float`, array `insert/remove/sort/reverse/slice/copy/join/sum/min/max`, map `values/copy/clear` (calling them reports "not available" / "no method"), array `+` array ("unsupported operand types"), and the map-literal key check (`{"a": 1, "a": 2}` keeps both entries, `2.0` stays a float key). `til fit --device` reports all of them. ~25% smaller: the data opcodes and every superinstruction run in a compact second function (`lean_op`), with switch dispatch and without the integer fast paths (same results and steps). Effect handlers and the screen/keys natives stay in |
| `TILVM_MAX_FUNCS` | `TILVM_MAX_CONSTS` (lean: 0) | function records decoded once at load, 8 bytes each; more functions → load error -10. 0 = read records from the bytecode on every call (no limit, no RAM) |
| `TILVM_FIELD_CACHE` | 64 (lean: 0) | entries of the (struct type, name) → field / method lookup caches (power of two, 6 bytes each, two caches) |
| `TILVM_COMPUTED_GOTO` | 1 with GCC/Clang, except lean and `-Os` | 1 = "labels as values" dispatch (one indirect jump per handler), 0 = portable `switch` |
| `TILVM_FAST_PATHS` | 1, except lean and `-Os` | inline integer fast paths in the interpreter; 0 = smaller code, same results |
| `TILVM_NO_PORTABLE` | off | leave out portable snapshots (`tilvm_save_portable` / `tilvm_load_portable` report "not available"; never in `TILVM_LEAN` builds); saves ~5 KB on ARM |
| `TILVM_NO_ENERGY` | off | leave out energy saving ([docs/TIL_ENERGY.md](../docs/TIL_ENERGY.md)): `#[energy: ...]` programs run at full speed, `energy_mode()` reports "not available in this VM build"; saves ~0.8 KB on the Z80 (the Z80 build uses it) |
| `TILVM_NO_GFX` | off | leave out the screen/keys natives (`screen_init` … `wait_key` report "not available in this VM build"); saves ~2.7 KB on ARM, ~7.5 KB on the Z80 (the Z80 build uses it) |
| `TILVM_GFX_MAX_W` / `TILVM_GFX_MAX_H` | 4096 / 4096 | largest screen `screen_init` accepts: set the display's size on a device (`screen_init(400, 300)` on a 320×240 build is a runtime error, so the screen hooks never get coordinates outside the display) |
| `TILVM_ROM` | empty | storage qualifier for the 480-byte font table, e.g. `__flash` for avr-gcc (keeps it out of RAM) |
| `TILVM_STATS` | off | record high-water marks (value stack, frames, recovery points, heap, effect handler entries): `tilvm_get_stats()`, second `--stats` line; used by `til fit --run` (docs/TIL_FIT.md) |
| `TILVM_PACK` | off | pack values and heap objects like 8-bit CPUs do (GCC/Clang): desktop emulation of AVR / Z80 / eZ80 heap use |

Profiles in the `Makefile`: `make` (desktop), `make small` (32-bit, float,
16 KB heap), `make tiny` (16-bit, no floats, 2 KB heap, lean), `make check`
(AddressSanitizer + UBSan with a 4 KB heap — exercises the compacting GC).

Code size of the whole VM (`arm-none-eabi-gcc -Os`, Cortex-M0, 32-bit ints, no
floats, `TILVM_NO_STDIO`, TIL 3.1 with effects and the screen/keys natives):
**31.4 KB**, or **23.6 KB** with `TILVM_LEAN` (28.7 KB / 21.0 KB with
`TILVM_NO_GFX`; measured `tilvm.o`, text + data).

Speed: the interpreter keeps its registers in C locals, dispatches with computed
goto on GCC/Clang and runs the superinstructions of bytecode v2
([docs/TIL_BYTECODE_V2.md](../docs/TIL_BYTECODE_V2.md)); on the desktop it is
1.7–3x faster than TIL 3.0 (`benchmarks/RESULTS.md`, `python3 benchmarks/run.py`).

## Loading, errors, floats

`tilvm_load()` returns 0 or a negative code: -1 not a `.tbc` file, -2 an
unknown bytecode version, -3 the program uses floats (a build without them),
-4 more constants than `TILVM_MAX_CONSTS`, **-5 a truncated or malformed
file**, -6 an integer constant too large for `TILVM_INT_BITS`, -7 more types
than `TILVM_MAX_TYPES`, -8 more globals than the value stack holds, -10 more
functions than `TILVM_MAX_FUNCS` (or, `TILVM_ADDR16`, a file above 64 KB).
Loading is memory safe for any input: every section is checked against the
file size, and the records against the rest of the file (constant and
function indexes in range, names are string constants, function code inside
the code section, the line table complete), so neither loading nor the error
reports (names, line numbers) read outside the buffer. A file that loaded is
then **trusted like code**: instruction operands (jump offsets, local slots,
constant indexes in instructions) are not checked while the program runs - a
`.tbc` written by `til` is always consistent; a deliberately broken one can
still make the interpreter misbehave. Portable snapshots (`.tilp`) are
checked against the bytecode as well, see docs/TIL_SNAPSHOTS.md.

<a id="floats"></a>**Floats.** Every build prints floats from the exact
binary value: `print`/`str` (`%g`), and the f-string specs `.Nf`, `.Ne`,
`.N`, `%` produce the digits of the float's exact decimal expansion rounded
half to even - the same text as glibc's `printf` and the reference VM, with or
without `TILVM_NO_STDIO`, for any magnitude (`f"{1e100:.2f}"` prints all 101
digits) and precision; a NaN prints `nan` (never `-nan`). `TILVM_FLOAT=1`
builds print the float32 value the same way, so their output equals the
reference VM's output for that float32 value (float constants are rounded
to the nearest float32 when they load). `int(x)` and typed `int` parameters
truncate a float exactly and wrap it to `TILVM_INT_BITS` like integer
arithmetic (the reference VM: 64 bits); `f"{x:d}"` (`x`, `X`, `o`, `b`) of a
float prints its exact integer part, whatever its size (`f"{1e30:d}"` is
`1000000000000000019884624838656`, as on the reference VM); nan/inf are the
runtime error `cannot convert nan/inf to int`. Without `<math.h>` (`TILVM_NO_MATH`),
`floor`, `ceil`, `round` and `%` are exact and `sqrt` is correctly rounded
(the same results as the C library); `x ** y` with an integer `y` is computed
by repeated squaring - exact whenever the result needs no rounding, otherwise
within a few units in the last place of the C library's `pow`; nan / infinite
exponents, 0, 1, inf and negative bases follow C99 `pow`; any other
fractional exponent is the runtime error `pow with a fractional exponent is
not available in this VM build` (never a silently wrong value).

**Timeouts.** `timeout(ms)` takes any non-negative number of milliseconds:
deadlines are kept as a 32-bit clock value plus a count of further 2^30 ms
periods, so `timeout(3000000000)` (or more) waits that long instead of
expiring at once, up to 2^62 ms; `nan` and `inf` never expire (as the
reference VM's float deadlines). A non-number is the runtime error
`timeout expects a number, got <type>` (lean builds: 0 ms).

## API

```c
#include "tilvm.h"

static tilvm vm;                       /* all VM state */

int run(const uint8_t *tbc, uint32_t size) {
    if (tilvm_load(&vm, tbc, size) != 0) return -1;   /* tbc may be in flash */
    vm.write = my_write;               /* void (*)(void *ctx, const char *s, size_t n) */
    vm.read_line = my_read_line;       /* int  (*)(void *ctx, char *buf, size_t cap)  */
    vm.clock_ms = my_clock;            /* uint32_t (*)(void *ctx)                      */
    vm.ctx = my_screen;
    return tilvm_run(&vm);             /* exit code */
}
```

Cooperative scheduling (keep a UI responsive, or interleave several programs):

```c
for (;;) {
    int st = tilvm_run_steps(&vm, 500);       /* 500 loop iterations / calls */
    if (st != TILVM_YIELD) break;             /* TILVM_DONE, TILVM_ERROR, TILVM_EXIT */
    handle_keypad_and_redraw();
}
```

Hibernate (e.g. on low battery or power-off):

```c
uint32_t n = tilvm_snapshot(&vm, buffer, sizeof buffer);   /* after a TILVM_YIELD */
save_to_flash(buffer, n);
/* ... later, after power-on: */
tilvm_load(&vm, tbc, size);
tilvm_restore(&vm, buffer, n);       /* same program + same VM build */
tilvm_run(&vm);                      /* continues where it stopped */
```

Continue on another device (any VM build, or `til exec prog.tbc --resume`):

```c
tilvm_save_portable(&vm, put, ctx);  /* after a TILVM_YIELD: put(ctx, bytes, n) streams the .tilp */
/* ... on the other device, with the same .tbc: */
tilvm_load(&vm, tbc, size);
tilvm_load_portable(&vm, get, ctx);  /* get(ctx, buf, n) reads n bytes; 0 or TILVM_P_* < 0 */
tilvm_run(&vm);
```

The desktop runner exposes the same features:

```bash
vm/tilvm --snapshot-after 300 state.snap prog.tbc   # run 300 steps, save, exit 3
vm/tilvm --resume state.snap prog.tbc               # continue
vm/tilvm --save-portable-after 300 s.tilp prog.tbc  # portable snapshot: continue on any VM build
vm/tilvm --resume-portable s.tilp prog.tbc          # (or: til exec prog.tbc --resume s.tilp)
vm/tilvm --max-steps 100000 prog.tbc                # sandbox
vm/tilvm --stats prog.tbc                           # GC / heap statistics (+ high-water marks with TILVM_STATS)
vm/tilvm --screen out.ppm prog.tbc                  # graphics: final frame as a PPM image
vm/tilvm --screen-ansi prog.tbc                     # graphics: frames in a 24-bit colour terminal
vm/tilvm --keys "UP,UP,ENTER,a,ESC" prog.tbc        # scripted keys (default: the keyboard)
```

Without `--screen`/`--screen-ansi` the runner draws into an offscreen
framebuffer (headless); without `--keys` keys come from the terminal in raw mode
(restored at exit) or, when stdin is not a terminal, from stdin's bytes
(on POSIX `key_pressed()` never waits for them). `clock_ms()`, timeouts and
`sleep_ms()` use the wall clock, and the runner really sleeps when the VM idles
(energy saving, `sleep_ms()`; [docs/TIL_ENERGY.md](../docs/TIL_ENERGY.md)). Ctrl-C ends the program,
writes the `--screen` image, restores the terminal and ends the runner by
SIGINT. See
[docs/TIL_GRAPHICS.md](../docs/TIL_GRAPHICS.md).

## Porting to a device

1. Pick a profile (`TILVM_INT_BITS`, `TILVM_FLOAT`, `TILVM_HEAP_SIZE`, ...).
2. Compile `tilvm.c` with the device toolchain (it only needs `<stdint.h>`,
   `<stddef.h>`, `<string.h>`, `<limits.h>`; stdio/time/math are optional).
3. Provide `write` (and `read_line`, `clock_ms` if available).
4. For a display and keys, set `vm.host.gfx_pixel` / `gfx_get` (and optionally
   `gfx_init`, `gfx_show`, `key_pressed`, `wait_key`) and build with the
   display's size (`-DTILVM_GFX_MAX_W=320 -DTILVM_GFX_MAX_H=240`) — the VM
   does all the drawing and clipping; [docs/TIL_GRAPHICS.md](../docs/TIL_GRAPHICS.md) has a
   complete host and notes for TI-84 Plus CE (graphx), NumWorks (eadk), Casio
   (gint), SDL, Linux framebuffer and browser canvas. Without hooks the screen
   functions report "not available on this device".
5. For battery saving set `vm.host.idle` (sleep / halt the CPU for about `ms`
   milliseconds: `halt`, `__WFI()`, an idle sleep mode) and a wall clock as
   `clock_ms`: the VM idles in the energy modes of `#[energy: ...]` /
   `energy_mode()` and while `sleep_ms()` waits —
   [docs/TIL_ENERGY.md](../docs/TIL_ENERGY.md) has the rule and per-device notes.
6. Embed the program (`til build --target c-vm`, or load a `.tbc` file from the
   device's file system).

Integration notes for common platforms (the core is verified in simulators;
these glue snippets are starting points and have not been run on hardware):

* **TI-84 Plus CE** (eZ80, CE C toolchain): a complete port is in
  `ports/ti84ce/` (programs in AppVars, home-screen console, keys, graphx
  screen hooks, [on] = break); its host code is tested on the desktop, the
  `.8xp` build itself is not verified yet — see `ports/ti84ce/README.md`.
* **TI-83 Plus / TI-84 Plus** (Z80): the 16-bit lean VM fits below 0xC000 and
  runs on a simulated Z80 (`ports/z80/`); on the calculators it needs a
  four-page Flash application (not done, see `ports/z80/RESULTS.md`). Output
  would use the `_PutC`/`_PutS` bcalls.
* **Casio fx-CG50 / fx-9860G** (SH4, gint SDK): `vm.write` → `dtext` into a
  scrolling console buffer + `dupdate`; `clock_ms` from `rtc_ticks`.
* **NumWorks** (Cortex-M7, external apps / `eadk`): `vm.write` →
  `eadk_display_draw_string`; `clock_ms` → `eadk_timing_millis`.
* **TI-Nspire** (ARM9, Ndless): the desktop build with newlib works as is.
* **Arduino / ESP32 / RP2040**: `vm.write` → `Serial.write`; on AVR read the
  bytecode from flash or keep it small (the AVR port copies it to RAM).

## Memory layout

```
tilvm (one struct)
├── host bindings (write/read/clock/ctx)        not saved in snapshots
├── program tables (constant & type offsets)    rebuilt by tilvm_load
├── tilvm_state                                 ← snapshots
│   ├── registers, step counters, RNG
│   ├── call frames, recovery points (retry/timeout/intent)
│   └── value stack (globals, locals)
└── arena (heap)                                ← snapshots (used part only)
    objects: str | array | map | value buffer | struct | closure | cell
    references are byte offsets → the mark-compact GC can slide objects,
    and a snapshot can be restored at any address
```
