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
| Z80 (TI-83/84 class) | Z80, 64 KB address space | ucsim `sz80` | int16, lean, 16-bit offsets, 4 KB heap, no effects / screen³ | 46.9 KB (SDCC 4.2), ends at 0xBE26 (below 0xC000) | 5.7 KB + ≤1.4 KB C stack | 10/10 int16-compatible examples + test corpus identical³ — see `ports/z80/RESULTS.md` |
| Desktop | x86-64 | native | int64, double, 256 KB heap | 80 KB | ~400 KB | all 255 + 81 tests |

¹ Single-precision builds print the last digit of some floats differently
(`78.5397` vs `78.5398`), as expected for `float`.
² TIL 3.1 builds with the screen/keys natives (about 2.7 KB of ARM code, 7 KB
of AVR code; `-DTILVM_NO_GFX` leaves them out). Measured with `examples/03_functions.til`.
³ 8 examples use floats and 2 compute values beyond 16 bits (`10!`); they are
rejected at load or wrap, as documented for 16-bit builds. The Z80 build leaves
out effect handlers and the screen/keys natives (`-DTILVM_NO_EFFECTS
-DTILVM_NO_GFX`, about 1.5 KB and 7.5 KB of Z80 code): with them the code would
not fit below 0xC000.

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
| `TILVM_NO_STDIO` | off | no `<stdio.h>`: built-in number formatting, host sets `write` |
| `TILVM_NO_TIME` | off | no `<time.h>`: `clock_ms()`/`timeout` count 1 ms per 1000 steps |
| `TILVM_NO_MATH` | off | no `<math.h>`: built-in `sqrt`/`floor`/`pow` (integer powers) |
| `TILVM_ADDR16` | off | 16-bit address space (Z80, AVR): bytecode offsets and object lengths are 16-bit (needs heap ≤ 65535; programs above 64 KB are rejected with load error -10). Much smaller code on 8-bit CPUs |
| `TILVM_LEAN` | off | smallest code for 8-bit CPUs: the whole language core, without format specs, snapshots, `input/chr/ord/type_of/clock_ms/random/sleep_ms`, str `split/replace/trim/repeat/count/to_float`, array `insert/remove/sort/reverse/slice/copy/join/sum/min/max`, map `values/copy/clear` (calling them reports "not available" / "no method"), array `+` array ("unsupported operand types"), and the map-literal key check (`{"a": 1, "a": 2}` keeps both entries, `2.0` stays a float key). `til fit --device` reports all of them. ~25% smaller: the data opcodes and every superinstruction run in a compact second function (`lean_op`), with switch dispatch and without the integer fast paths (same results and steps). Effect handlers and the screen/keys natives stay in |
| `TILVM_MAX_FUNCS` | `TILVM_MAX_CONSTS` (lean: 0) | function records decoded once at load, 8 bytes each; more functions → load error -10. 0 = read records from the bytecode on every call (no limit, no RAM) |
| `TILVM_FIELD_CACHE` | 64 (lean: 0) | entries of the (struct type, name) → field / method lookup caches (power of two, 6 bytes each, two caches) |
| `TILVM_COMPUTED_GOTO` | 1 with GCC/Clang, except lean and `-Os` | 1 = "labels as values" dispatch (one indirect jump per handler), 0 = portable `switch` |
| `TILVM_FAST_PATHS` | 1, except lean and `-Os` | inline integer fast paths in the interpreter; 0 = smaller code, same results |
| `TILVM_NO_GFX` | off | leave out the screen/keys natives (`screen_init` … `wait_key` report "not available in this VM build"); saves ~2.7 KB on ARM, ~7.5 KB on the Z80 (the Z80 build uses it) |
| `TILVM_GFX_MAX_W` / `TILVM_GFX_MAX_H` | 4096 / 4096 | largest screen `screen_init` accepts: set the display's size on a device (`screen_init(400, 300)` on a 320×240 build is a runtime error, so the screen hooks never get coordinates outside the display) |
| `TILVM_ROM` | empty | storage qualifier for the 480-byte font table, e.g. `__flash` for avr-gcc (keeps it out of RAM) |
| `TILVM_STATS` | off | record high-water marks (value stack, frames, recovery points, heap): `tilvm_get_stats()`, second `--stats` line; used by `til fit --run` (docs/TIL_FIT.md) |
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

The desktop runner exposes the same features:

```bash
vm/tilvm --snapshot-after 300 state.snap prog.tbc   # run 300 steps, save, exit 3
vm/tilvm --resume state.snap prog.tbc               # continue
vm/tilvm --max-steps 100000 prog.tbc                # sandbox
vm/tilvm --stats prog.tbc                           # GC / heap statistics (+ high-water marks with TILVM_STATS)
vm/tilvm --screen out.ppm prog.tbc                  # graphics: final frame as a PPM image
vm/tilvm --screen-ansi prog.tbc                     # graphics: frames in a 24-bit colour terminal
vm/tilvm --keys "UP,UP,ENTER,a,ESC" prog.tbc        # scripted keys (default: the keyboard)
```

Without `--screen`/`--screen-ansi` the runner draws into an offscreen
framebuffer (headless); without `--keys` keys come from the terminal in raw mode
(restored at exit) or, when stdin is not a terminal, from stdin's bytes
(on POSIX `key_pressed()` never waits for them). Ctrl-C ends the program,
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
5. Embed the program (`til build --target c-vm`, or load a `.tbc` file from the
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
