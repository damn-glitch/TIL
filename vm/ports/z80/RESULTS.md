# TIL VM on the Z80 — status (TIL 3.1)

**Summary:** the 16-bit lean VM (`TILVM_LEAN`, `TILVM_ADDR16`, 16-bit ints, no
floats) compiles with SDCC 4.2, its code and constants end at **0xBE68** (below
0xC000), and it runs TIL programs on the ucsim Z80 simulator (`sz80`) with output
identical to the reference VM for every example that fits 16-bit integers
without floats. A real TI-83 Plus / TI-84 Plus additionally needs a multi-page
Flash application (see *Real calculators* below); the TI-84 Plus CE has its own
port in `../ti84ce/`.

```bash
make                                    # build/tilvm.ihx (SDCC, ~3 minutes)
python3 run_sim.py ../../../examples/05_control_flow.til     # output + statistics
python3 run_sim.py --size               # memory map and the interpreter's stack frame
TIL_TEST_CROSS=1 python3 -m pytest ../../../tests/test_z80.py   # build + compare (needs sdcc, sz80)
```

## The SDCC internal compiler error

`error 9: FATAL Compiler Internal Error in file 'gen.c' line number '7884' (and
'7476') : Unimplemented` while generating `tilvm_run_steps`. Both sites are in
`genPlus` (z80 code generator). Cause, found with `--dump-i-code` and the SDCC
4.2 sources:

* Global common-subexpression elimination keeps dozens of temporaries alive
  across the whole interpreter loop; they are spilled and the stack frame grows
  to ~285 bytes.
* Spill locations below IX-128 can only be reached through HL ("extended stack"
  operands, `AOP_EXSTK`). During register allocation SDCC costs every spilled
  operand as an IX-relative one (`AOP_STK`, "on-stack for dry run"), so it
  happily puts the other operand of an addition into H/L. At final code
  generation the operand turns out to be extended-stack, needs HL itself, and
  `genPlus` has no code for that combination.
* The 32-bit build had the same oversized frame (288 bytes, 355 extended-stack
  operand uses) and only escaped by luck of the register assignment. SDCC 4.5.0
  (#15242, Debian package) fails on the same function too (`gen.c:8282`).

**Workaround** (`vm/tilvm.c`, SDCC only): `#pragma save / #pragma nogcse /
#pragma restore` around `tilvm_run_steps` — the frame drops to ~70 bytes, every
spill is IX-relative and the bad combination cannot occur. The frame must stay
below 128 bytes; `run_sim.py --size` prints it and `tests/test_z80.py` checks it.
A second SDCC 4.2 bug of the same kind (`gen.c:5733`, `genSend`: a register
argument meets a value whose linear live range spans other branches of a
`?:` of calls) appeared in intermediate versions; the final code avoids it
(lean builds do not compile the int fast paths of the main loop).

## Size

All numbers: SDCC 4.2.0 #13081, `-mz80`, VM profile of the `Makefile` (heap
4 KB, 192 stack slots, 32 frames, 192 constants, 24 types).

| Build | VM code + constants | Below 0xC000? |
|---|---|---|
| TIL 3.0, 32-bit ints, all builtins | 76 KB | no |
| TIL 3.0, 32-bit ints, `TILVM_LEAN` | 70.7 KB | no |
| TIL 3.0, 16-bit ints, `TILVM_LEAN` | — | SDCC internal error |
| 16-bit lean, ICE worked around, byte readers as functions (`--opt-code-size`) | 57,407 B | no |
| + `TILVM_ADDR16` (16-bit offsets/lengths) | 58,314 B¹ | no |
| + data opcodes in `lean_op()` (loop keeps control flow) | 53,741 B | no |
| + shared push/pop tails in `lean_op`, `reset_state` memset | 51,631 B | no |
| + `?` helper, shared error messages, map equality via `map_find`, no 32-bit multiply/divide routines | 49,328 B | no |
| + `--opt-code-speed --max-allocs-per-node 10000` (Makefile) | **47,726 B** | **yes** |
| for comparison: same code and flags, 32-bit ints (compiled only, not run) | 49,106 B | no (about 1 KB over with harness and runtime) |

¹ The rest of the VM shrank by 1.9 KB, but SDCC allocated registers worse in the
huge interpreter function (+2.8 KB) — which is why the data opcodes moved out.

Linked image (`run_sim.py --size`): code 0x0110-0xBE68 = 48,472 bytes (VM
47,726 + harness 274 + SDCC runtime/startup), **408 bytes free** below 0xC000.
Largest functions: `lean_op` 5.0 KB, `call_native` 3.6 KB, `tilvm_run_steps`
3.1 KB, `array_method` 1.9 KB, `tilvm_load` 1.9 KB.

Desktop builds keep their behaviour: with the default profile only `arith`,
`equal`, `report`, `tilvm_load` and `call_method` compile differently (same
semantics: pointer-stepping copy loops, map equality through `map_find`, 16-bit
line search, memset in `reset_state`); the interpreter loop is untouched. Lean
builds run data opcodes through `lean_op`, the same semantics in a compact form
(the tests run the corpus on lean desktop builds against the reference VM).

## RAM

| | bytes |
|---|---|
| VM state (`static tilvm vm`: 4 KB heap, 192 × 3-byte value stack, 32 frames, tables) | 5,812 |
| C stack, typical program | ~250 |
| C stack, deepest case measured (printing / comparing 15-31 levels of nested arrays) | 1,403 |
| program bytecode (examples, with line numbers) | 157-1,704 |

## Speed (simulated clock ticks = Z80 T-states)

| Measurement | T-states |
|---|---|
| `while i < 2000: s = s ^ i; i += 1` per iteration (13 instructions) | 41,628 (3,200 per instruction) |
| `fib(18)` per call (8,362 calls) | 55,517 |
| load + print of a 1-line program | 0.86 M |

At 15 MHz (TI-84 Plus) that is ~360 loop iterations or ~270 calls per second.

## Programs on the simulated Z80

Corpus in `tests/test_z80.py` (operators, strings, arrays + higher-order
methods, maps, structs + methods + enums + `match`, closures + default
arguments + globals, `Result`/`Option` + `?`, `retry` + runtime errors, GC
pressure, deep recursion): all **identical** to the reference VM, including
error messages and exit codes.

| Example | Result | Bytecode | Steps | T-states |
|---|---|---|---|---|
| `01_hello.til` | identical | 261 | 1 | 1,362,083 |
| `05_control_flow.til` | identical | 1,079 | 49 | 8,171,028 |
| `06_enums.til` | identical | 730 | 4 | 3,515,773 |
| `08_imports.til` | identical | 386 | 4 | 1,995,125 |
| `10_verified.til` | identical | 864 | 10 | 4,351,674 |
| `math_utils.til` | identical | 304 | 0 | 1,451,649 |
| `basic/control_flow.til` | identical | 1,361 | 61 | 9,972,680 |
| `basic/hello.til` | identical | 157 | 1 | 928,172 |
| `kazakh/salam.til` | identical | 469 | 12 | 2,755,758 |
| `kazakh/salam_latn.til` | identical | 430 | 7 | 2,348,869 |
| `03_functions.til`, `basic/functions.til` | differ: `10!` = 3628800 wraps to 24320 in 16 bits | | | |
| `02_variables`, `04_structs`, `07_multilevel`, `09_v3_features`, `advanced/structs`, `basic/variables`, `multilevel/multilevel`, `russian/privet` | not run: they use floats (load error -3, reported) | | | |

## What the lean 16-bit build does not have

Explicit errors, never silent: floats (programs using them are rejected at
load), integer constants beyond ±32767 (rejected at load), format specs, file /
argument / screen / key / energy natives, `input chr ord type_of clock_ms random
sleep_ms`, str `split replace trim repeat count`, array `insert remove sort
reverse slice copy join sum min max`, map `values copy clear`, snapshots.
Integer arithmetic wraps at 16 bits (like every `TILVM_INT_BITS` build).
Inherited from `TILVM_LEAN` and not changed here: f-string format specs are
ignored and duplicate keys in map literals are not merged.

## Memory map (simulator, `main_z80.c`)

```
0x0000-0x010F  reset vectors, startup
0x0110-0xBFFF  VM code and constants (ends at 0xBE68)
0xC000-0xD7FF  VM state (5,812 bytes used)
0xD800-0xEFEF  program bytecode (u16 length + .tbc), written by run_sim.py
0xEFF0-0xEFFF  results (output length, exit code, steps, heap, GC count, C stack mark)
0xF000-0xF7FF  program output
0xF800-0xFFFF  C stack
```

## Real calculators (TI-83 Plus / TI-84 Plus)

The simulator gives the program a flat 64 KB. On the calculators, assembly
programs run from RAM at 0x9D95 and code above 0xC000 cannot execute, so a
program is limited to ~8.8 KB; larger code must be a Flash application, whose
16 KB pages are mapped at 0x4000-0x7FFF one at a time (port 6). The VM (47.7 KB:
26.5 KB interpreter, arithmetic, objects and GC; 12.0 KB natives and builtin
methods; 9.2 KB loading, printing and errors) would need **four application
pages** plus cross-page calls, e.g.

* page 0: startup, `tilvm_run_steps`, `lean_op`, call/return, arithmetic, comparison;
* page 1: allocation, GC, strings, arrays, maps, structs, indexing;
* page 2: `call_native` and the builtin methods (`str_method`, `array_method`, ...);
* page 3: loading, value printing, error reports and message tables;
* RAM 0x8000-0xFFFF: VM state (5.8 KB), the program AppVar, OS areas.

Calls between pages 0 and 1 are frequent, so every page switch (writing port 6
through a trampoline, on the order of 100 T-states) would slow the interpreter
noticeably.

SDCC can call between pages (`__banked` functions through a trampoline that
writes port 6), but each page must be a separate translation unit, so
`tilvm.c` would have to be split (or compiled several times with parts
disabled) and its `static` helpers exported — a structural change to the
single-file VM that was not made here. It also needs a signed-app toolchain
(e.g. rabbitsign) and an emulator with a TI-84 Plus ROM (not distributable) to
test, neither available in this environment. **Not implemented**; the TI-84
Plus CE (`../ti84ce/`) has no such limit.

## Keeping it below 0xC000

Only 408 bytes are free. New features in `tilvm.c` should be excluded from
`TILVM_LEAN` builds (as the lean list above), or the Z80 image overflows —
`TIL_TEST_CROSS=1 python3 -m pytest tests/test_z80.py` reports the code end
and the interpreter's frame size. Opcodes added to `tilvm_run_steps` must also
be added to `lean_op` (data opcodes) or stay in the loop for lean builds.
