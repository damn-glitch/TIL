# TIL bytecode v2: superinstructions and the faster VMs

TIL 3.1 writes bytecode **version 2**. The file layout is exactly the version 1
layout described in [TIL_BYTECODE.md](TIL_BYTECODE.md) (header, constants,
functions, types, methods, prelude, code, line table); version 2 only adds
instructions and natives. Numbers are append-only and never change meaning.

| Added in v2 | Where it is specified |
|---|---|
| natives 34–54 (`read_file` … `energy_mode`) | `docs/dev/TIL_3_1_DESIGN.md` §1 |
| opcodes `0x76–0x79` (`HANDLE`, `END_HANDLE`, `PERFORM`, `RESUME`) | `docs/dev/TIL_3_1_DESIGN.md` §8 |
| opcodes `0x80–0xBF`: **superinstructions** | this document |

Both VMs load versions 1 and 2. A version-1 file never contains the new
opcodes; a TIL 3.0 VM (or any VM without these opcodes) runs programs built
with `--no-fuse`, which emits only the instructions of TIL_BYTECODE.md (plus the
natives / effect opcodes a program really uses).

## 1. Superinstructions

A superinstruction does in one dispatch what a short, frequent sequence of
ordinary instructions does. The compiler's peephole pass (section 2) emits them;
every VM must execute each one **exactly** like the sequence in the "same as"
column:

* same results, same stack effect, same error messages;
* an error inside a superinstruction is reported at the superinstruction's
  first byte — which has the same source line as the failing instruction of the
  sequence, because a sequence never contains a line-table entry after its first
  instruction;
* a jump counts one step (loop back-edge) when it goes backwards, exactly like
  `JUMP` / `JUMP_IF_FALSE`; superinstructions never contain calls, so step
  counts, `--max-steps`, budgets of `tilvm_run_steps()`, timeouts and snapshots
  behave as for the unfused code;
* a superinstruction never needs more value-stack slots than its sequence
  (the integer fast paths of the C VM need none).

Operand types: `b` = u8, `c` = i8, `h` = u16, `s` = i16 (little endian), as in
TIL_BYTECODE.md. `L[a]` is local slot `a` of the current frame; `k` is an
immediate integer; `off` is a jump offset relative to the end of the
instruction. `<cmp>` is one of the comparison opcodes `EQ 0x30`, `NE 0x31`,
`LT 0x32`, `LE 0x33`, `GT 0x34`, `GE 0x35`, stored as an operand byte; any other
value is malformed bytecode (both VMs stop with a fatal "bad opcode" error).

### Locals, fields, indexing, returns

| op | hex | operands | same as | stack |
|---|---|---|---|---|
| `LOAD_LOCAL2` | 80 | b a, b b | `LOAD_LOCAL a; LOAD_LOCAL b` | → L[a], L[b] |
| `LOAD_LOCAL_FIELD` | 81 | b a, h name | `LOAD_LOCAL a; GET_FIELD name` | → L[a].name |
| `INC_LOCAL` | 82 | b a, c k | `LOAD_LOCAL a; PUSH_I8 k; ADD; STORE_LOCAL a` | — (L[a] = L[a] + k) |
| `CAST_LOCAL` | 83 | b a, b kind | `LOAD_LOCAL a; CAST kind; STORE_LOCAL a` | — |
| `CAST_RETURN` | 84 | b kind | `CAST kind; RETURN` | v → (returns cast(v)) |
| `ADD_STORE_LOCAL` | 85 | b a | `ADD; STORE_LOCAL a` | x, y → (L[a] = x + y) |
| `NATIVE_POP` | 86 | b id, b argc | `NATIVE id argc; POP` | args → |
| `INDEX_L` | 87 | b b | `LOAD_LOCAL b; GET_INDEX` | obj → obj[L[b]] |
| `INDEX_LL` | 88 | b a, b b | `LOAD_LOCAL a; LOAD_LOCAL b; GET_INDEX` | → L[a][L[b]] |
| `RETURN_LOCAL` | 89 | b a | `LOAD_LOCAL a; RETURN` | (returns L[a]) |
| `FOR_ITER_STORE` | 8A | b it, b x, s off | `FOR_ITER it off; STORE_LOCAL x` | — (L[x] = next element, or jump) |

`ADD` keeps all of its meanings in these forms (`"a" + 1` concatenates, int
overflow wraps around, `[1] + [2]` concatenates arrays). `CAST` kinds are those
of TIL_BYTECODE.md (0 int, 1 float, 2 str, 3 bool). `NATIVE_POP` with `exit()`
stops the program like `NATIVE` does (the `POP` is never reached).

### Compare and branch

| op | hex | operands | same as |
|---|---|---|---|
| `JUMP_IF_NOT` | 90 | b cmp, s off | `<cmp>; JUMP_IF_FALSE off` |
| `JUMP_IF_NOT_L` | 91 | b cmp, b b, s off | `LOAD_LOCAL b; <cmp>; JUMP_IF_FALSE off` |
| `JUMP_IF_NOT_I` | 92 | b cmp, s k, s off | `PUSH_I8 k` or `PUSH_I16 k`; `<cmp>; JUMP_IF_FALSE off` |
| `JUMP_IF_NOT_LL` | 93 | b cmp, b a, b b, s off | `LOAD_LOCAL a; LOAD_LOCAL b; <cmp>; JUMP_IF_FALSE off` |
| `JUMP_IF_NOT_LI` | 94 | b cmp, b a, s k, s off | `LOAD_LOCAL a; PUSH_I8/PUSH_I16 k; <cmp>; JUMP_IF_FALSE off` |

The left operand is the value below (stack forms) or `L[a]`; the right operand
is the top of the stack, `L[b]` or `k`. Stack effect: `JUMP_IF_NOT` pops two
values, `_L` and `_I` pop one, `_LL` and `_LI` none. The comparison has its
usual meaning: `EQ`/`NE` are structural equality (`1 == 1.0`, arrays, maps,
structs), `LT`..`GE` compare numbers or strings and otherwise fail with
`cannot compare <type> and <type>`; NaN compares false. The jump is taken when
the comparison is **false**; a taken backward jump counts one step.

### Binary operators with an immediate or a local right operand

`0xA0 + i` = `<op>_I k` (operand `s k`, an i16): `PUSH_I8 k` or `PUSH_I16 k`;
`<op>`. `0xB0 + i` = `<op>_L b` (operand `b b`): `LOAD_LOCAL b; <op>`.
Stack effect of both: `x → x <op> right`.

| i | op | `_I` | `_L` | i | op | `_I` | `_L` |
|---|---|---|---|---|---|---|---|
| 0 | `ADD` | A0 | B0 | 8 | `SHL` | A8 | B8 |
| 1 | `SUB` | A1 | B1 | 9 | `SHR` | A9 | B9 |
| 2 | `MUL` | A2 | B2 | 10 | `EQ` | AA | BA |
| 3 | `DIV` | A3 | B3 | 11 | `NE` | AB | BB |
| 4 | `MOD` | A4 | B4 | 12 | `LT` | AC | BC |
| 5 | `BAND` | A5 | B5 | 13 | `LE` | AD | BD |
| 6 | `BOR` | A6 | B6 | 14 | `GT` | AE | BE |
| 7 | `BXOR` | A7 | B7 | 15 | `GE` | AF | BF |

The operator has exactly its usual meaning, errors included (`division by
zero`, `bitwise operators need integers`, `unsupported operand types for -:
str and int`, ...). `POW` has no fused forms. Opcodes `0x8B–0x8F`, `0x95–0x9F`
and `0xC0–0xFF` are not assigned (bad opcode).

## 2. The peephole pass

`fuse_superinstructions()` in `src/til.py` runs on every function after the
bytecode compiler has emitted it (on by default; `til run|build|dis --no-fuse`
and `compile_to_bytecode(..., fuse=False)` turn it off):

1. decode the function, collect the **jump targets** (all instructions with a
   jump offset, including `TRY`/`TRY_TIMEOUT` handlers, `FOR_ITER`,
   `ARG_DEFAULT` and `HANDLE`) and the pcs of the **line table**;
2. a sequence is fused only when none of its instructions *after the first* is
   a jump target or has a line-table entry. Sequences never contain calls,
   `TRY`/`END_TRY` (recovery points), `PANIC` or effect instructions;
3. `JUMP 0` (a jump to the next instruction, left behind by `if` without
   `else`) is removed;
4. among all possible choices, dynamic programming picks the one with the
   fewest instructions (`LOAD_LOCAL2` counts slightly more than one instruction,
   so `LOAD_LOCAL a; ADD_L b` is preferred over `LOAD_LOCAL2 a b; ADD`);
5. the code is laid out again, every jump offset is recomputed and the line
   table is remapped (a removed `JUMP 0` hands its line entry to the next
   instruction unless that one has its own).

The pass returns the map *old pc → new pc* of the function; a later pass that
records pcs during code generation (e.g. local-variable debug info) must map
them through it. Running the pass on already fused code changes nothing.

Measured on the test programs and benchmarks, fused code has 25–40% fewer
instructions and is 10–25% smaller.

## 3. The C VM interpreter (`vm/tilvm.c`)

* **Dispatch.** With `TILVM_COMPUTED_GOTO` (default when `__GNUC__` is defined,
  i.e. GCC and Clang, except `TILVM_LEAN` and `-Os` builds) every handler ends
  with its own indirect jump through a 256-entry label table (GNU "labels as
  values"; the `-Wpedantic` diagnostics for it are switched off around the
  interpreter only, so `-std=c99 -pedantic -Werror` builds stay clean). With
  `-DTILVM_COMPUTED_GOTO=0`, MSVC, SDCC and the lean/size builds a portable
  `switch` runs the same handlers.
* **Registers.** The instruction pointer, the stack pointer and the frame base
  live in C locals. `vm->st.sp` and the frame's `pc` are written back only
  before a helper that works on the stack, before anything that can allocate
  (the collector scans `st.sp` slots) and before returning to the host — the
  saved state is exactly what TIL 3.0 saved, so raw snapshots
  (`tilvm_snapshot` / `tilvm_restore`) keep their format.
* **Fast paths** for integer arithmetic, comparisons, compare-and-branch,
  array indexing, `array.push`, field access and calls; everything else goes
  through the TIL 3.0 helpers. Superinstruction slow paths push their operands
  and run the plain handler, so they cannot differ from the sequence.
* **Function table** (`TILVM_MAX_FUNCS`, default `TILVM_MAX_CONSTS`; `0` in
  `TILVM_LEAN`): function records are decoded once by `tilvm_load` (8 bytes per
  function). A program with more functions is rejected with error `-10`
  ("too many functions for this VM build"); with `TILVM_MAX_FUNCS=0` records are
  read from the bytecode on each call (no limit, no RAM).
* **Lookup caches** (`TILVM_FIELD_CACHE` entries, power of two, default 64; `0`
  in `TILVM_LEAN`): direct-mapped caches *(struct type, name) → field slot* and
  *(struct type, name) → method*. They hold only facts about the loaded
  program, are reset by `tilvm_load` and are not part of snapshots.
* **Steps.** A step is still a loop back-edge or a call. The common case costs
  one counter decrement and two tests: a "fuel" counter says how many steps may
  run before the sandbox limit (`max_steps`) or the budget of
  `tilvm_run_steps()` must be looked at; the slow path then performs exactly the
  checks of TIL 3.0, in the same order (step limit, deadlines every 256 steps,
  budget). Steps, `tick`, yields and timeouts are unchanged. (A feature that
  needs work every N steps, such as idling for energy saving, adds a term to
  `refuel()` and its check to the slow path.)
* **Strings and maps.** `find` / `contains` / `split` / `count` / `replace`
  locate candidates with `memchr`; map lookups compare int and string keys
  directly; f-strings are measured and built with a single allocation;
  `str(int)` formats once. Results are unchanged.

Code size (`arm-none-eabi-gcc -Os -mcpu=cortex-m0 -mthumb -DTILVM_INT_BITS=32
-DTILVM_FLOAT=0`, `tilvm.o` text): `TILVM_LEAN` 20 335 → 21 315 bytes (+4.8%);
full build 26 831 → 30 927 bytes (+15%; with computed goto, `-O2`-style builds,
about 2 KB more). The lean build keeps only the compact slow paths of the
superinstructions.

## 4. The reference VM (`src/til.py`, class `TILVM`)

`TILVM.execute()` implements every superinstruction with the same semantics
(`binop()` and `cmp_values()` are the shared general paths). The dispatch chain
is ordered by measured frequency and compares against opcode constants held in
local variables; calls extend the stack in one step. It is faster than TIL 3.0
on every benchmark (see `benchmarks/RESULTS.md`).

## 5. Checking it

* `tests/test_speed.py`: tables (Python == C, dispatch table), the pass (never
  across jump targets or line entries, idempotent, lines kept, `--no-fuse`),
  fused == unfused on both VMs for every example, a corpus that reaches every
  superinstruction's fast and slow paths (including errors and their lines),
  step counts, `--max-steps`, snapshot/resume at many pause points, timeouts,
  C build variants (switch, no fast paths, no tables, a 2-entry cache, lean) and
  strict builds. With the git history at hand it also checks against TIL 3.0
  itself (commit `6f4d00b`): `--no-fuse` writes byte-identical bytecode, and
  steps, `--max-steps` stops and snapshot pause points equal those of the 3.0
  C VM. `TIL_BENCH=1` adds the timing check (at least 2x on fib / loops /
  arrays / strings, reference VM not slower).
* `benchmarks/run.py`: timings of any commit against any other.
