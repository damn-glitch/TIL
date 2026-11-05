# TIL VM on the TI-84 Plus CE

The TI-84 Plus CE (eZ80 at 48 MHz, 24-bit addresses, ~150 KB of user RAM) runs
the full TIL VM — not the lean Z80 build — with 32-bit integers and
single-precision floats. Programs are TIL bytecode stored in AppVars.

```
vm/ports/ti84ce/
├── makefile          CE C toolchain build -> bin/TILVM.8xp
├── src/main.c        host: console, keys, graphx screen, AppVar files, [on] = break
├── src/tilvm_ce.c    compiles ../../tilvm.c into the program
├── tbc2appvar.py     .tbc / .til -> .8xv AppVar (no toolchain needed)
└── desktop/          stand-ins for the CE libraries (tests only, see below)
```

## Build and run

```bash
# CE C toolchain (CEdev) from https://github.com/CE-Programming/toolchain/releases
export PATH=/path/to/CEdev/bin:$PATH
make -C vm/ports/ti84ce                      # -> vm/ports/ti84ce/bin/TILVM.8xp

python3 vm/ports/ti84ce/tbc2appvar.py examples/03_functions.til FUNCS   # -> FUNCS.8xv
```

Send `TILVM.8xp` and the AppVar(s) with TI Connect CE or TiLP, then run
`TILVM` from the [prgm] menu (`Asm(prgmTILVM)` on older OS versions). TILVM
lists every AppVar whose data starts with the bytecode magic `TILB`; with more
than one, choose with the arrow keys and [enter] ([clear] leaves).

## VM profile (makefile `TIL_PROFILE`)

`TILVM_INT_BITS=32`, `TILVM_FLOAT=1` (the CE's `float`), `TILVM_NO_STDIO`,
`TILVM_ADDR16` (AppVars and the 32 KB heap are below 64 KB, so 16-bit offsets),
heap 32 KB, 1024 value-stack slots, 128 call frames, 1024 constants, 128 struct
types. All builtins are available (not `TILVM_LEAN`).

## Host services

| TIL | CE |
|---|---|
| `print` / errors | OS home screen via `outchar()` (wraps and scrolls); non-ASCII UTF-8 shows as `?` |
| `key_pressed()` / `wait_key()` | `os_GetCSC()`: arrows UP=1 DOWN=2 LEFT=3 RIGHT=4, [enter] 13, [clear] 27, [del] 8, digits, `+ - * / ^ ( ) , .`, and the alpha letters a b c d e f g i n s x of [math] [apps] [prgm] [x⁻¹] [sin] [cos] [tan] [x²] [log] [ln] [sto→] |
| screen hooks (`gfx_init/pixel/get/show`) | graphx, 8-bit palette RRRGGGBB; the TIL screen is scaled by an integer factor and centred on 320x240; `get_pixel` returns the palette colour (pure primaries, black and white round-trip exactly) |
| `read_file` / `write_file` / `file_exists` | AppVars: the path is the AppVar name (1-8 letters/digits) |
| `clock_ms()`, `timeout` | `clock()` (32768 Hz) |
| `idle` (energy saving) | `delay(ms)` |
| [on] | stops the running program (`BREAK`) |

Text printed while the graphics screen is active is kept (last 1 KB) and shown
on the home screen after the program ends; the last frame stays visible until a
key is pressed. The bytecode is copied to RAM before it runs, so programs may
create AppVars without moving their own code.

The graphics and energy natives are implemented by the screen/energy features
of TIL 3.1; this port provides the device hooks they call (`vm->host`).

## What was verified

* `src/main.c` + `tilvm.c` compile with `-std=c99 -Wall -Wextra -pedantic
  -Werror` against the stand-in headers in `desktop/` (same function names and
  signatures as CEdev v15.0's `ti/screen.h`, `ti/getcsc.h`, `keypadc.h`,
  `fileioc.h`, `graphx.h`, `sys/timers.h`), and the resulting desktop program
  runs TIL programs from `.8xv` files written by `tbc2appvar.py`: program menu,
  AppVar loading, console output, `read_file`/`write_file` through AppVars —
  output identical to the reference VM (`tests/test_z80.py`).
* The `.8xv` writer is checked against the TI-83 Plus family file format
  (header, entry, checksum) by the same tests.

**Not verified here:** building `TILVM.8xp` with CEdev and running it on a
calculator or in CEmu. The CE toolchain v15.0 (CEdev-Linux) was downloaded, but
running its build was not permitted in the sandbox this port was written in,
and CEmu needs a calculator ROM image, which cannot be distributed. With CEdev
installed, `TIL_TEST_CROSS=1 python3 -m pytest tests/test_z80.py -k ce_program`
builds the program and checks that it is below the 64 KB variable limit.
