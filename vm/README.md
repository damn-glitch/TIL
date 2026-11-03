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
| ARM Cortex-M (NumWorks/STM32 class) | Cortex-M3, 64 KB RAM | QEMU `lm3s6965evb` | int32, float32, 16 KB heap | 44.7 KB | 24 KB | 17/17 programs identical¹ |
| 8-bit AVR (Arduino Mega class) | ATmega1284P, 16 KB RAM | simavr | int32, float32, 5 KB heap | 53.8 KB | 10.5 KB | 17/17 programs identical |
| Z80 (TI-83/84 class) | Z80, 64 KB address space | ucsim `sz80` | int32, lean | 70.7 KB (SDCC) | ~7 KB | **does not fit** the 64 KB address space yet — see `ports/z80/RESULTS.md` |
| Desktop | x86-64 | native | int64, double, 256 KB heap | 80 KB | ~400 KB | all 255 + 81 tests |

¹ Single-precision builds print the last digit of some floats differently
(`78.5397` vs `78.5398`), as expected for `float`.

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
| `TILVM_MAX_CONSTS` / `TILVM_MAX_TYPES` | 4096 / 512 | per-program table sizes |
| `TILVM_NO_STDIO` | off | no `<stdio.h>`: built-in number formatting, host sets `write` |
| `TILVM_NO_TIME` | off | no `<time.h>`: `clock_ms()`/`timeout` count 1 ms per 1000 steps |
| `TILVM_NO_MATH` | off | no `<math.h>`: built-in `sqrt`/`floor`/`pow` (integer powers) |
| `TILVM_LEAN` | off | smallest code for 8-bit CPUs: the whole language core, without format specs, snapshots, `input/chr/ord/type_of/clock_ms/random/sleep_ms`, str `split/replace/trim/repeat/count`, array `insert/remove/sort/reverse/slice/copy/join/sum/min/max`, map `values/copy/clear` (calling them reports "not available"). ~20% smaller |
| `TILVM_MAX_FUNCS` | `TILVM_MAX_CONSTS` (lean: 0) | function records decoded once at load, 8 bytes each; more functions → load error -10. 0 = read records from the bytecode on every call (no limit, no RAM) |
| `TILVM_FIELD_CACHE` | 64 (lean: 0) | entries of the (struct type, name) → field / method lookup caches (power of two, 6 bytes each, two caches) |
| `TILVM_COMPUTED_GOTO` | 1 with GCC/Clang, except lean and `-Os` | 1 = "labels as values" dispatch (one indirect jump per handler), 0 = portable `switch` |
| `TILVM_FAST_PATHS` | 1, except lean and `-Os` | inline integer fast paths in the interpreter; 0 = smaller code, same results |

Profiles in the `Makefile`: `make` (desktop), `make small` (32-bit, float,
16 KB heap), `make tiny` (16-bit, no floats, 2 KB heap, lean), `make check`
(AddressSanitizer + UBSan with a 4 KB heap — exercises the compacting GC).

Code size of the whole VM (`arm-none-eabi-gcc -Os`, Cortex-M0, 32-bit ints, no
floats): **27.3 KB**, or **20.8 KB** with `TILVM_LEAN` (TIL 3.1; 3.0: 26.2 / 19.9 KB; measured `tilvm.o`).

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
vm/tilvm --stats prog.tbc                           # GC / heap statistics
```

## Porting to a device

1. Pick a profile (`TILVM_INT_BITS`, `TILVM_FLOAT`, `TILVM_HEAP_SIZE`, ...).
2. Compile `tilvm.c` with the device toolchain (it only needs `<stdint.h>`,
   `<stddef.h>`, `<string.h>`; stdio/time/math are optional).
3. Provide `write` (and `read_line`, `clock_ms` if available).
4. Embed the program (`til build --target c-vm`, or load a `.tbc` file from the
   device's file system).

Integration notes for common platforms (the core is verified in simulators;
these glue snippets are starting points and have not been run on hardware):

* **TI-84 Plus CE** (eZ80, CE C toolchain): `vm.write` → `os_PutStrFull` /
  `os_NewLine`; keep `TILVM_HEAP_SIZE` ≤ 32 KB; `tilvm_run_steps` +
  `os_GetCSC()` between slices lets [ON] stop a program. Use `TILVM_FLOAT=1`.
* **TI-83 Plus / TI-84 Plus** (Z80): not yet — with SDCC the VM is ~70 KB of
  code, more than the Z80's 64 KB address space; it needs a banked Flash
  application or a 16-bit build (see `ports/z80/RESULTS.md`). Output would use
  the `_PutC`/`_PutS` bcalls.
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
