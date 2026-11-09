# TIL VM on the Z80 — status (TIL 3.1)

**Summary:** the 16-bit lean VM (`TILVM_LEAN`, `TILVM_ADDR16`, 16-bit ints, no
floats; TIL 3.1 with the faster interpreter and superinstructions, without
effect handlers, the screen natives and energy saving) compiles with SDCC 4.2,
its code and constants end at **0xBF31** (below 0xC000), and it runs TIL programs on the
ucsim Z80 simulator (`sz80`) with output identical to the reference VM for every
example that fits 16-bit integers without floats. A real TI-83 Plus / TI-84 Plus additionally needs a multi-page
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

Since the TIL 3.1 merge, lean builds keep only the control flow in
`tilvm_run_steps` (the data opcodes and every superinstruction run in
`lean_op`), so the function is small enough for GCSE: its frame is 55 bytes
with it, and GCSE saves 0.8 KB. The pragma now applies only to builds with the
whole interpreter loop (`TILVM_LEAN` off).

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

### TIL 3.1 merge: faster VM, effects, graphics

The integration branch brought a rewritten interpreter (registers `ip`/`sp`/`bp`
in C locals, superinstructions 0x80-0xBF that the compiler always emits,
computed goto on GCC/Clang), effect handlers and the screen/keys natives. Merged
as is, the Z80 image ended at 0xE3F6, 9.2 KB too large. Steps (VM `_CODE`, same
profile and flags):

| Build | VM `_CODE` | Free below 0xC000 |
|---|---|---|
| this port before the merge (TIL 3.1 z80 branch) | 47,726 B | 408 B |
| merged, `-DTILVM_NO_GFX -DTILVM_NO_EFFECTS` (Makefile) | 57,691 B | −9,206 B |
| + data opcodes and every superinstruction in `lean_op()` | 53,435 B | −4,950 B |
| + `?`, `CAST_RETURN`, `RETURN_LOCAL` in `lean_op`; plain per-step checks instead of the fuel counter; stack sync through small functions; one decoder for the compare-and-branch superinstructions; no bytecode hash (only snapshots use it) | 49,915 B | −1,430 B |
| + pc sync, recovery points and the return path through small functions / the shared `resume` code; one path for "push a local, then run the plain opcode" | 49,253 B | −768 B |
| + `make_array` through `push_empty` (one allocation routine) | 49,064 B | −579 B |
| + GCSE on for the (now small) lean `tilvm_run_steps` | 48,209 B | 276 B |
| + shared cell/upvalue, `MATCH_TYPE`/`IS_ARRAY_LEN`, `FOR_ITER`, field cases in `lean_op` (final) | **48,011 B** | **474 B** |

`--max-allocs-per-node 20000` gave 49,374 B and 40000 gave 48,234 B for the
same source, so the Makefile keeps 10000 (and the ~2.5 minute build). SDCC's
register allocation is chaotic: reordering two cases of `lean_op` once moved
its size by 1.8 KB. When a small change overflows the image, try an equivalent
formulation of the changed code and compare `build/tilvm.lst` per function.

Linked image (`run_sim.py --size`): code 0x0110-0xBE26 = 48,406 bytes (VM
48,011 + harness + SDCC runtime/startup), **474 bytes free** below 0xC000.
Largest functions: `lean_op` 5.6 KB, `call_native` 3.9 KB, `tilvm_run_steps`
3.2 KB, `array_method` 1.9 KB, `tilvm_load` 1.8 KB.

### TIL 3.1 energy saving (feature 11)

Energy saving (`#[energy: ...]`, `energy_mode()`, docs/TIL_ENERGY.md) costs the
lean Z80 image 802 bytes (VM `_CODE` 48,813 B: `energy_point` 287 B,
`tilvm_load` +161 B, `tilvm_run_steps` +147 B, `call_native` +108 B, the error
message 68 B, `reset_state` +31 B): the code would end at 0xC148, 328 bytes
above the limit. The Makefile therefore adds `-DTILVM_NO_ENERGY` (programs run
at full speed; `energy_mode()` reports "not available in this VM build"). What
remains is the idle counter (4 bytes of RAM) and its reset at load: VM
`_CODE` **48,029 B**, code 0x0110-0xBE38 = 48,424 bytes, **456 bytes free**;
VM state 5,840 bytes. A Flash application (see below) would have room for
energy saving, and the TI-84 Plus CE port includes it (`delay()` as idle).

### Memory-safe loading (TIL 3.1 review)

`tilvm_load` checks every section of a `.tbc` against the file size and every
record against the rest of the file (a damaged file is load error -5 instead of
reads outside the buffer; vm/README.md). On the Z80 this costs 153 bytes
(`tilvm_load` -107 B, the record walk `ld_records` 168 B, `str_const` 77 B,
`ld16` 20 B, `push_try` +18 B for deadlines that wrap to 0): code
0x0110-0xBF31 (VM `_CODE` 48,278 B), **207 bytes free**.

Desktop builds keep their behaviour and speed: the interpreter loop of the
default build compiles from the same code as before the merge (the lean-only
paths are `#ifdef TILVM_LEAN`); outside it only `arith`, `equal`, `report`,
`tilvm_load`, `call_method` and `make_array` compile differently, with the same
semantics. Lean builds run the data opcodes through `lean_op`, the same
semantics in a compact form: the tests run the corpus, every superinstruction
and the effect opcodes on lean desktop builds against the default build
(output, errors and step counts) and the reference VM.

## RAM

| | bytes |
|---|---|
| VM state (`static tilvm vm`: 4 KB heap, 192 × 3-byte value stack, 32 frames, tables) | 5,840 |
| C stack, typical program | ~250 |
| C stack, deepest case measured (printing / comparing 15-31 levels of nested arrays) | 1,392 |
| program bytecode (examples, with line numbers) | 156-1,596 |

## Speed (simulated clock ticks = Z80 T-states)

| Measurement | T-states |
|---|---|
| `while i < 2000: s = s ^ i; i += 1` per iteration (6 instructions with superinstructions; 13 before TIL 3.1) | 31,131 (41,628 before the merge) |
| `fib(18)` per call (8,362 calls) | 45,385 (55,517 before the merge) |

At 15 MHz (TI-84 Plus) that is ~480 loop iterations or ~330 calls per second.
Start-up is faster too: lean builds no longer hash the bytecode at load (only
snapshots used the hash), which was most of the time of small programs
(`01_hello.til`: 1.36 M T-states before the merge, 0.40 M now).

## Programs on the simulated Z80

Corpus in `tests/test_z80.py` (operators, strings, arrays + higher-order
methods, maps, structs + methods + enums + `match`, closures + default
arguments + globals, `Result`/`Option` + `?`, `retry` + runtime errors, GC
pressure, deep recursion): all **identical** to the reference VM, including
error messages and exit codes.

| Example | Result | Bytecode | Steps | T-states |
|---|---|---|---|---|
| `01_hello.til` | identical | 258 | 1 | 397,564 |
| `05_control_flow.til` | identical | 1,013 | 49 | 3,595,287 |
| `06_enums.til` | identical | 703 | 4 | 799,855 |
| `08_imports.til` | identical | 350 | 4 | 525,678 |
| `10_verified.til` | identical | 829 | 10 | 1,153,822 |
| `math_utils.til` | identical | 271 | 0 | 322,130 |
| `basic/control_flow.til` | identical | 1,276 | 61 | 4,208,679 |
| `basic/hello.til` | identical | 156 | 1 | 348,146 |
| `kazakh/salam.til` | identical | 438 | 12 | 883,071 |
| `kazakh/salam_latn.til` | identical | 406 | 7 | 684,122 |
| `03_functions.til`, `basic/functions.til` | differ: `10!` = 3628800 wraps to 24320 in 16 bits | | | |
| `02_variables`, `04_structs`, `07_multilevel`, `09_v3_features`, `advanced/structs`, `basic/variables`, `multilevel/multilevel`, `russian/privet` | not run: they use floats (load error -3, reported) | | | |

## What the lean 16-bit build does not have

Explicit errors, never silent: floats (programs using them are rejected at
load), integer constants beyond ±32767 (rejected at load), format specs, file /
argument / screen / key / energy natives, `input chr ord type_of clock_ms random
sleep_ms`, str `split replace trim repeat count`, array `insert remove sort
reverse slice copy join sum min max`, map `values copy clear`, snapshots,
effect handlers (`handle ... with` reports "effect handlers are not available
in this VM build"; `perform E()` still calls a top-level function `E`).
`TILVM_NO_EFFECTS`, `TILVM_NO_GFX` and `TILVM_NO_ENERGY` are Z80 Makefile
choices: with them in, the image would be about 1.5 KB, 7.5 KB and 0.8 KB
larger. (Without energy saving, `#[energy: ...]` programs run at full speed
with the same output.)
Integer arithmetic wraps at 16 bits (like every `TILVM_INT_BITS` build).
Inherited from `TILVM_LEAN` and not changed here: f-string format specs are
ignored and duplicate keys in map literals are not merged.

## Memory map (simulator, `main_z80.c`)

```
0x0000-0x010F  reset vectors, startup
0x0110-0xBFFF  VM code and constants (ends at 0xBF31)
0xC000-0xD7FF  VM state (5,840 bytes used)
0xD800-0xEFEF  program bytecode (u16 length + .tbc), written by run_sim.py
0xEFF0-0xEFFF  results (output length, exit code, steps, heap, GC count, C stack mark)
0xF000-0xF7FF  program output
0xF800-0xFFFF  C stack
```

## Real calculators (TI-83 Plus / TI-84 Plus)

The simulator gives the program a flat 64 KB. On the calculators, assembly
programs run from RAM at 0x9D95 and code above 0xC000 cannot execute, so a
program is limited to ~8.8 KB; larger code must be a Flash application, whose
16 KB pages are mapped at 0x4000-0x7FFF one at a time (port 6). The VM (48.0 KB;
split as measured before the TIL 3.1 merge, at 47.7 KB: 26.5 KB interpreter,
arithmetic, objects and GC; 12.0 KB natives and builtin methods; 9.2 KB
loading, printing and errors) would need **four application pages** plus
cross-page calls, e.g.

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

Only 207 bytes are free. New features in `tilvm.c` should be excluded from
`TILVM_LEAN` builds (as the lean list above) or get a `TILVM_NO_...` switch for
the Z80 Makefile, or the Z80 image overflows — `TIL_TEST_CROSS=1 python3 -m
pytest tests/test_z80.py` reports the code end and the interpreter's frame size.

In lean builds `tilvm_run_steps` keeps only the control flow (jumps, calls,
returns, natives, recovery points, effects) and everything else goes through
`lean_op`: an opcode added to the main loop inside `#ifndef TILVM_LEAN` must
also be added to `lean_op` (a lean build reports "bad opcode" otherwise), and a
new superinstruction can usually be written there as "push the operand, then
run the plain opcode" (`push_local` / `push_again`). `tests/test_z80.py` runs
every superinstruction and the effect opcodes on a lean build.
