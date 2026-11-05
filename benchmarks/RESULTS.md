# TIL VM benchmarks — TIL 3.0 vs TIL 3.1

Baseline: TIL 3.0 as of commit `6f4d00b` (its compiler, its C VM, its reference
VM). New: this branch (superinstructions on, the default). Every program prints
the same output on every engine (checked by `run.py`).

Reproduce:

```bash
python3 benchmarks/run.py --baseline 6f4d00b --vms c-base,c,c-nofuse --repeat 21 --markdown
python3 benchmarks/run.py --baseline 6f4d00b --vms py-base,py,native --repeat 3 --markdown
TIL_BENCH=1 python3 -m pytest tests/test_speed.py -k speedup -s     # the 2x check
```

## Machine and method

* Intel Xeon @ 2.10 GHz, 4 cores (shared virtual machine, other jobs running:
  load average 7–8), Linux 6.18, glibc 2.39
* gcc 13.3.0, `-std=c99 -O2` (the desktop build `til run --cvm` uses);
  Python 3.13.18
* CPU time (user + sys) of each run from `wait4`; for the Python VMs only
  `TILVM.run()` is timed. Runs of all engines are interleaved; the tables show
  medians (C VMs: 21 runs, Python VMs: 3 runs). On this busy machine single
  runs vary by ±10–20%; the instruction counts below are exact.

| benchmark | what it does |
|---|---|
| `fib` | recursive `fib(27)`: calls, compare-and-branch, typed parameters and returns |
| `loops` | nested `for` loops and a `while` loop with integer `* + ^ % & -` |
| `arrays` | `push`, indexing in counted loops, `for x in a`, `sum` |
| `strings` | f-strings, `split`, `to_int`, `find`, `str()`, concatenation |
| `maps` | inserting and looking up string and int keys |
| `structs` | struct fields, field assignment, methods returning new structs |
| `closures` | closures calling closures, captured variables, `map/filter/reduce` |

## C VM (desktop, `-O2`)

| benchmark | TIL 3.0 (s) | TIL 3.1 (s) | 3.1 `--no-fuse` (s) | speedup | speedup without superinstructions |
|---|---:|---:|---:|---:|---:|
| fib | 0.034 | 0.012 | 0.022 | **2.76x** | 1.57x |
| loops | 0.073 | 0.027 | 0.041 | **2.68x** | 1.78x |
| arrays | 0.060 | 0.020 | 0.043 | **2.97x** | 1.37x |
| strings | 0.046 | 0.018 | 0.019 | **2.51x** | 2.44x |
| maps | 0.024 | 0.010 | 0.010 | 2.46x | 2.40x |
| structs | 0.044 | 0.026 | 0.028 | 1.70x | 1.60x |
| closures | 0.036 | 0.022 | 0.026 | 1.65x | 1.40x |

The targets of the design (at least 2x on fib and the loop/array/string
benchmarks) are met; fib and arrays are close to 3x. `TIL_BENCH=1` (best of 9
runs) measured fib 2.75x, loops 2.62x, arrays 3.03x, strings 2.43x.
Calls (structs, closures) gain least: every call still checks arity, frame and
stack limits and counts a step, exactly as before.

Executed machine instructions (valgrind `callgrind`, whole process):

| benchmark | TIL 3.0 | TIL 3.1 | ratio | 3.1 `--no-fuse` |
|---|---:|---:|---:|---:|
| fib | 373.1 M | 157.3 M | 2.37x | 291.1 M |
| loops | 805.2 M | 295.0 M | 2.73x | 478.8 M |
| arrays | 668.6 M | 240.3 M | 2.78x | 428.0 M |
| strings | 466.2 M | 136.8 M | 3.41x | 143.2 M |
| maps | 355.0 M | 112.4 M | 3.16x | 116.3 M |
| structs | 532.6 M | 227.5 M | 2.34x | 265.6 M |
| closures | 450.7 M | 251.6 M | 1.79x | 327.6 M |

Where the time went: computed-goto dispatch and registers in C locals (all
benchmarks); superinstructions (`--no-fuse` column: fib, loops, arrays); the
`memchr`-based `find`, single-allocation f-strings and direct int/string map
keys (strings, maps); the field/method cache (structs).

## Reference VM (Python, `src/til.py`)

| benchmark | TIL 3.0 (s) | TIL 3.1 (s) | speedup |
|---|---:|---:|---:|
| fib | 3.701 | 2.032 | 1.82x |
| loops | 7.047 | 4.621 | 1.52x |
| arrays | 5.378 | 2.984 | 1.80x |
| strings | 0.551 | 0.494 | 1.12x |
| maps | 0.158 | 0.103 | 1.54x |
| structs | 4.912 | 2.310 | 2.13x |
| closures | 4.778 | 3.126 | 1.53x |

Not slower anywhere: fewer dispatches (superinstructions), a dispatch chain in
measured frequency order comparing against local constants, and more integer
fast paths. Python 3.11 and 3.12 run the same code (smoke-tested: all examples,
fused == unfused).

## Native backend (for reference)

`til build` (C via gcc `-O2`): fib 0.002 s, loops 0.004 s. The native backend
cannot compile the other five benchmarks (it falls back to the VM for them).

## Embedded builds

Code size of `tilvm.o` (`.text`, bytes):

| build | TIL 3.0 | TIL 3.1 | change |
|---|---:|---:|---:|
| Cortex-M0, `-Os`, `TILVM_LEAN`, int32, no float | 20 335 | 21 315 | +4.8% |
| Cortex-M0, `-Os`, int32, no float | 26 831 | 27 907 | +4.0% |
| AVR ATmega1284P, `-Os` (port profile) | 55 653 | 57 987 | +4.2% |
| Z80, SDCC 4.2.0, `TILVM_LEAN` int32 (port profile, `_CODE`) | 70 675 | 63 450 | **−10.2%** |

`-Os` builds use the switch dispatch and the general paths only
(`TILVM_COMPUTED_GOTO=0`, `TILVM_FAST_PATHS=0` by default); with `-O2`
(desktop) both are on. RAM: the function table (8 bytes per function, up to
`TILVM_MAX_FUNCS`) and two 64-entry lookup caches (768 bytes) — the Cortex-M
port (`TILVM_MAX_CONSTS=512`) uses 4.9 KB more, the AVR port 2.0 KB more
(13.2 of 16 KB); `TILVM_MAX_FUNCS=0` / `TILVM_FIELD_CACHE=0` remove them.

Speed on a cycle-accurate simulator (simavr, ATmega1284P @ 16 MHz, AVR port
profile, `-Os`): a loop of 20 000 `%`/`+=` iterations plus 12 x `fib(15)` takes
3.78 s of simulation with TIL 3.0 and 2.85 s with TIL 3.1 (1.33x), with the same
output. The AVR and Cortex-M (QEMU) ports print the same as the reference VM for
`03_functions`, `04_structs` and `09_v3_features`.

Z80: SDCC compiles the TIL 3.1 lean build with warnings only, and the code is
smaller than before — SDCC generates much better code for the registers kept in
C locals than for `vm->st.sp` accesses through the VM pointer.

## After merging effects with resumption (feature 8)

The effect opcodes (`HANDLE`/`END_HANDLE`/`PERFORM`/`RESUME`) were ported into
this interpreter; the only cost on code without handlers is one test of
`n_handlers` per return. Same machine and compiler, benchmark bytecode
byte-identical before and after; base = the faster VM before the merge.

| benchmark | instructions before | after | change | CPU time before (ms) | after (ms) |
|---|---:|---:|---:|---:|---:|
| fib | 157.3 M | 163.7 M | +4.0% | 12.4 | 12.2 |
| loops | 295.0 M | 295.0 M | 0.0% | 27.7 | 27.4 |
| arrays | 240.3 M | 239.8 M | −0.2% | 19.0 | 18.6 |
| strings | 136.8 M | 136.2 M | −0.5% | 18.5 | 18.7 |
| maps | 112.4 M | 112.3 M | −0.1% | 9.6 | 10.0 |
| structs | 227.5 M | 229.9 M | +1.1% | 25.0 | 24.2 |
| closures | 251.6 M | 256.6 M | +2.0% | 22.0 | 21.3 |

Instructions: valgrind `callgrind`, gcc 13.3 `-O2`; times: medians of 60
interleaved runs (differences within the ±3% noise of this machine). The extra
instructions on call-heavy code come from register allocation around the
return path, not from the test itself. Inlining the effect helpers into the
interpreter (GCC's default for static functions called once) costs another
0.5–4% of instructions on every benchmark, so they are marked `TV_NOINLINE`. `TIL_BENCH=1` against TIL 3.0
after the merge: fib 2.81x, loops 2.66x, arrays 3.08x, strings 2.43x. An
effects-heavy program (300 000 resumed performs in a fused loop, 100 000
aborted handles around a `for` loop) runs 2.2x faster than on the effects
branch's TIL 3.0-style interpreter.
