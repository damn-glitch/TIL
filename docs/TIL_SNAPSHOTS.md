# Portable snapshots (`.tilp`)

A *portable snapshot* is a paused TIL program written as a device-independent
byte stream: start a program on a PC, save it, and continue it on a calculator
or a microcontroller - on any VM build that loads the same `.tbc` file (another
integer width, with or without floats, the reference VM in Python). It is TIL
3.1 feature 2 (`docs/dev/TIL_3_1_DESIGN.md` §2).

Raw snapshots (`tilvm_snapshot` / `tilvm_restore`, `tilvm --snapshot-after`)
remain: they are a memory image of one VM build (fast, but only that build and
heap size can restore them). Portable snapshots describe the program's state
instead of the VM's memory.

## 1. Using it

```
til run --vm prog.til --pause-after 5000 --save state.tilp   # run 5000 steps, save, exit 3
til run --vm prog.til --resume state.tilp                    # continue to the end
til run --cvm prog.til --pause-after 5000 --save state.tilp  # the same on the C VM
til run --cvm prog.til --resume state.tilp
til exec prog.tbc --resume state.tilp                        # a built program (.tbc)
til run --vm prog.til --resume a.tilp --pause-after 100 --save b.tilp   # one more slice

tilvm --save-portable-after 5000 state.tilp prog.tbc         # the C VM runner (vm/main.c)
tilvm --resume-portable state.tilp prog.tbc
```

* A *step* is a loop back-edge or a call (also `perform` entering a clause),
  counted exactly the same way by both VMs, by `--max-steps`, by
  `tilvm_run_steps()` budgets and by `TILVM.run_steps()`. `--pause-after N`
  runs N more steps (counted from the start, or from the resumed point).
* When the program pauses, the state is saved and the command ends with exit
  code **3** and a message on standard error (`til: paused after N steps;
  saved state.tilp (B bytes)`); the program's output so far is flushed first.
  When the program ends before N steps, it ends normally (its own exit code)
  and nothing is saved (a note says so).
* `--pause-after` needs `--save` and vice versa (exit code 2 otherwise).
  `--resume` combines with both (run a further slice and save again).
* The `.tbc` must be byte-for-byte the one the snapshot was made with: the
  snapshot holds an FNV-1a hash of it, and a snapshot of another program (or
  of the same source built with other options, e.g. `--strip`, `--no-fuse`)
  is refused. `til run --vm prog.til` and `til run --cvm prog.til` build the
  same bytes, as does `til build --target vm prog.til`.
* What is *not* in a snapshot: the program's arguments (`args()` - pass them
  again), open input, scripted keys, the screen's pixels (the size is kept;
  the screen is re-created all black on resume, like a raw restore - programs
  that redraw every frame continue unchanged), host settings. `--max-steps`
  counts the program's steps since its start, also those before the pause.
* `--pause-after 0` saves the program before its first step.
* `random()` continues the saved generator state. 16/32-bit C builds use a
  32-bit generator (as without snapshots): numbers drawn after a resume on
  another kind of VM differ from an uninterrupted run there.
* `timeout(ms)` blocks keep the time they had left when the program paused;
  the clock does not run while the program is saved.

## 2. APIs

### C VM (`vm/tilvm.h`)

```c
typedef int (*tilvm_put_fn)(void *ctx, const uint8_t *data, uint32_t len);  /* 0 = ok */
typedef int (*tilvm_get_fn)(void *ctx, uint8_t *buf, uint32_t len);        /* 0 = read len bytes */

uint32_t tilvm_save_portable(tilvm *vm, tilvm_put_fn put, void *ctx); /* bytes written, 0 = error */
int tilvm_load_portable(tilvm *vm, tilvm_get_fn get, void *ctx);      /* 0 or TILVM_P_xxx (< 0) */
```

* **Save** after `tilvm_run_steps()` returned `TILVM_YIELD` (a pause point:
  the interpreter's cached registers are written back there), or right after
  `tilvm_load()` (step 0). A finished or failed program is not saved (0). The
  bytes go to `put` in pieces of at most 64 bytes (string contents: in one
  piece); a `put` that returns non-zero stops the save (0, the VM is
  unchanged). The save uses no `malloc` and does not allocate: object numbers
  live in the heap object headers while it runs (`mark` / `fwd`, and a 16-bit
  hash in `aux` of strings; all are 0 again afterwards, as outside the GC);
  the queue of objects still to write and a hash table of the strings (to
  find strings with the same bytes) live in the free end of the arena. With
  little room the table is left out or fills up, and the queue shrinks to
  the GC's mark stack (`TILVM_MARK_STACK` entries): heap rescans then find
  the objects that did not fit, as the GC's mark phase does after an
  overflow - slower, never wrong. The VM is unchanged afterwards (it can
  continue, or save again: the same bytes).
* **Load** after `tilvm_load()` (same bytecode) and after the host hooks are
  set (the screen is re-created with `host.gfx_init`), before running. It
  replaces the whole program state; then `tilvm_run_steps()` / `tilvm_run()`
  continue the program. Objects are allocated in the arena in table order (no
  collection runs while loading); references are object ids until the end,
  then they are patched using an id table kept at the top of the arena (when
  the objects need that room, a walk over the loaded objects finds them
  instead), so the heap needed is the program's live data, no more. Fields
  are read through `get` a few bytes at a time (string contents in one
  piece), so the snapshot never has to be in RAM as a whole.
  On an error the VM is left reset and failed (`tilvm_run_steps` returns
  `TILVM_ERROR`): load the program again.

| code | macro | meaning |
|---|---|---|
| -1 | `TILVM_P_NOT_TILP` | not a portable snapshot (magic) |
| -2 | `TILVM_P_VERSION` | another format version, or unknown flags |
| -3 | `TILVM_P_PROGRAM` | saved by another program (bytecode hash) |
| -4 | `TILVM_P_CORRUPT` | truncated, damaged (checksum) or malformed data |
| -5 | `TILVM_P_TOO_BIG` | does not fit this build: value stack, frames, recovery points, handler entries, heap, an array longer than 65535 elements, a string longer than the build allows |
| -6 | `TILVM_P_INT_RANGE` | an integer outside this build's `TILVM_INT_BITS` |
| -7 | `TILVM_P_NO_FLOAT` | a float in a build without floats |
| -8 | `TILVM_P_SCREEN` | a screen this build cannot show (`TILVM_NO_GFX`, or larger than `TILVM_GFX_MAX_W/H`) |
| -9 | `TILVM_P_UNAVAILABLE` | `TILVM_LEAN` / `TILVM_NO_PORTABLE` build: portable snapshots are left out |
| -10 | `TILVM_P_NO_EFFECTS` | active effect handlers in a `TILVM_NO_EFFECTS` build |

`TILVM_LEAN` builds (8-bit CPUs, the Z80 port) leave portable snapshots out,
and `-DTILVM_NO_PORTABLE` does so in any build: both functions only report
"not available" (`tilvm_save_portable` returns 0, `tilvm_load_portable`
returns -9; `tilvm` prints `portable snapshots are not available in this VM
build`). Code size: see §7. `-DTILVM_P_SCRATCH=n` (tests only) limits the
save queue and the load id table to n entries and the string table to 4, to
exercise the rescan and walk paths.

A host that keeps the snapshot in memory:

```c
static int put_mem(void *ctx, const uint8_t *p, uint32_t n) { /* append to a buffer */ }
struct src { const uint8_t *p; uint32_t left; };
static int get_mem(void *ctx, uint8_t *buf, uint32_t n) {
    struct src *s = ctx;
    if (n > s->left) return -1;
    memcpy(buf, s->p, n); s->p += n; s->left -= n;
    return 0;
}
...
tilvm_load(&vm, program, size);           /* the same .tbc */
vm.write = my_write;                      /* hooks first */
struct src s = { snapshot, snapshot_size };
if (tilvm_load_portable(&vm, get_mem, &s) != 0) { /* report */ }
tilvm_run(&vm);
```

### Reference VM (`src/til.py`)

```python
vm = til.TILVM(module, ...)          # optional tbc=<the .tbc bytes> (else module.raw / serialize())
status = vm.run_steps(5000)          # TILVM.DONE / ERROR / YIELD / EXIT (0..3, as in C)
data = vm.save_portable()            # bytes (after YIELD, or before the first run)
vm2 = til.TILVM(module, ...)
vm2.load_portable(data)              # raises til.SnapshotError (.code = the C error code)
vm2.run()                            # or run_steps(n) again
```

`run_steps(n)` pauses after exactly n steps (0 = no limit); the output is
flushed before it returns. `vm.exit_code` holds the exit code after DONE /
ERROR / EXIT.

## 3. The format (TILP version 1)

All numbers are little endian; `u8/u16/u32/u64` unsigned, `i64` two's
complement, `f64` IEEE 754 binary64. The file is one sequence of sections:

```
offset size  field
------ ----  -----------------------------------------------------------------
HEADER (10 bytes)
     0    4  magic "TILP"
     4    1  version = 1
     5    1  flags = 0 (reserved: readers reject other values)
     6    4  program hash: FNV-1a-32 of the complete .tbc file
             (h = 2166136261; for each byte b: h = (h ^ b) * 16777619 mod 2^32)
STATE (16 bytes)
    10    4  u32 steps        steps executed so far (mod 2^32)
    14    8  u64 rng          random() generator state (16/32-bit C VMs: the
                              low 32 bits, upper half 0; a 16/32-bit reader
                              takes the low 32 bits, 0 -> the default seed)
    22    2  u16 tick         deadline-check counter, 0..255 (both VMs look at
                              the deadlines every 256 steps while one exists)
    24    1  u8  energy mode  battery saving (design §11): 0 performance, 1 balanced, 2 low
                              performance, 1 balanced, 2 low. Written 0 by
                              this version; readers accept 0..2 (other values
                              are corrupt) and ignore it until §11 lands
    25    1  u8  reserved = 0
GLOBALS COUNT
    26    2  u16 G            must equal the program's global count
FRAMES
          2  u16 F            call frames, 1..; then F records (bottom first):
          2    u16 func       function index
          4    u32 pc         where the frame continues, relative to the start
                              of the code section (the top frame: the pause
                              point; below it: after the call instruction)
          2    u16 base       first local slot; value-stack index (see VALUES)
          1    u8  extra      1 when a callee slot lies below the arguments
                              (CALL_VALUE / a struct field holding a function)
          4    u32 closure    object id of the closure the frame runs, 0 = none
                              (also 0 for a closure without upvalues)
RECOVERY POINTS (retry / timeout / intent blocks)
          2  u16 T            then T records (oldest first):
          4    u32 handler    pc of the recovery code (relative)
          4    u32 deadline   0 = no deadline; else the milliseconds left before
                              the block times out, at least 1 (an expired one
                              is written as 1: it fires at the next check)
          2    u16 frames     frame count when the block started
          2    u16 sp         value-stack depth when the block started
          2    u16 handlers   effect handler entries when the block started
HANDLER RECORDS (effect handlers, docs/TIL_EFFECTS.md §3)
          2  u16 H            then H records (bottom first):
          1    u8  kind       0 active `handle`, 1 running clause
          2    u16 frames     n_frames
          2    u16 sp         kind 0: value-stack index of the first (clause,
                              name) pair; kind 1: 0
          2    u16 n_try      recovery points when installed
          4    u32 abort pc   kind 0: relative pc after END_HANDLE; kind 1: 0
          2    u16 n          kind 0: clause count; kind 1: owner index
VALUES
          G values            the globals, in order
          2  u16 S            value-stack depth (globals not counted)
          S values            the value stack, bottom first (locals of every
                              frame, temporaries, handler pairs)
SCREEN
          2  u16 width        0 = no screen_init yet
          2  u16 height       (both 0 or both 1..4096)
OBJECT TABLE
             objects 1, 2, 3, ... in id order (see 3.2), each:
          1    u8 kind
               ...payload
TRAILER
          4  u32 N            number of objects
          4  u32 checksum     CRC-32 (the one of zlib / PNG: polynomial 0xEDB88320,
                              reflected, initial and final value 0xFFFFFFFF) of
                              every byte before this field
```

Value-stack indexes (`base`, recovery `sp`, handler `sp`, `S`) do not count the
globals: index 0 is the first value after them. (The C VM keeps the globals at
the bottom of its value stack and adds `G`; the reference VM keeps them apart.)

### 3.1 Values

A value is a tag byte and a payload:

| tag | value | payload |
|---|---|---|
| 0 | `None` | - |
| 1 | `false` | - |
| 2 | `true` | - |
| 3 | int | `i64` |
| 4 | float | `f64` (a `float` C build widens exactly) |
| 5 | constant string | `u16` index of a string constant of the program |
| 6 | object reference | `u32` object id, >= 1 |
| 7 | function | `u16` function index (a function value without upvalues) |
| 8 | missing argument | - (a parameter the caller did not pass; its default is computed when the function starts - a pause can happen before that) |

Objects (the payload after the kind byte):

| kind | object | payload |
|---|---|---|
| 1 | string | `u32` length, the bytes |
| 2 | array | `u32` n, n values |
| 3 | map | `u32` n, n entries in insertion order: key value, then value |
| 4 | struct / enum variant / Some / Ok / Err | `u16` type index, `u16` n (= the type's field count), n field values |
| 5 | closure | `u16` function index, `u16` n (= the function's upvalue count), n upvalues (references to cells) |
| 6 | cell (a captured variable) | 1 value |

### 3.2 Object ids: the traversal order

An object gets the next id (1, 2, 3, ...) when the writer writes the first
reference to it; references are written in file order. So the roots are
visited in this order - frame closures (bottom frame first), globals, value
stack (bottom first) - and then the object table itself: writing object k's
payload (values left to right; map entries key before value) numbers the
objects first referenced there. The object table is therefore a breadth-first
traversal from the roots, and object k is written after objects 1..k-1. A
reference never uses an id larger than the number of ids assigned so far + 1
(readers check it); the table ends when every assigned id has been written.

### 3.3 Canonical encodings

Both VMs write the same bytes for the same program state (up to deadlines,
which depend on the clock, and the 32-bit generator of small C builds):

* **Strings are values.** A string whose bytes equal a string constant of
  the program is written as a constant string (tag 5) with the lowest index
  of such a constant (a C VM writes a constant it holds with its own index;
  `til` never writes the same string constant twice). Any other string is a
  string object, and all strings with the same bytes share one object (the
  first visited). Strings in TIL are immutable, so sharing is not
  observable. Readers accept either form for any string.
* **Functions.** A function value without upvalues (C: `TV_FUNC`; reference
  VM: a closure with no captured variables) is tag 7; a closure with
  upvalues is an object. A frame's closure field is 0 when the frame runs no
  closure or one without upvalues.
* Everything else that can be shared - arrays, maps, structs, closures,
  cells - is an object, so shared and cyclic references keep their identity.
* Integers are always 64-bit in the file; a 16/32-bit C VM sign-extends, and
  refuses (`-6`) to load an integer it cannot hold. Floats: a `float` C VM
  writes the exact double of its value and rounds when it loads one; a VM
  without floats refuses (`-7`).

### 3.4 What a reader checks

In this order (both VMs report the first problem with the same code): magic
(-1), version and flags (-2), the program hash (-3), then while reading
(-4 unless noted): energy byte 0..2, tick <= 255,
`G` = the program's globals; the counts of frames / recovery points / handler
entries / stack values against the build's limits (-5; `TILVM_NO_EFFECTS`:
-10 for any handler entry); every function / type / constant index is in
range and constant strings are string constants; pcs lie inside the code
section; frame bases ascend, frame 0 has base 0 and no callee slot, every
frame's locals lie below the next frame's arguments (the top frame's below
`S`); a frame's closure is a closure object of the frame's function; recovery
points and handler records satisfy the invariants of `docs/TIL_EFFECTS.md`
(ordered by frames, a running clause's owner is an earlier `handle`, pairs
below `S`); struct field and closure upvalue counts match the program;
upvalues are cells; map keys are `None`, bools, numbers or strings; object
ids are assigned in order and every id is defined; integers fit the build
(-6), floats exist in it (-7); the screen size is 0x0 or within 1..4096 and
the build's limits (-8); objects fit the heap and the build's string / array
limits (-5); then the object count and the checksum, and (the whole file
being there) no bytes after it. A snapshot is otherwise trusted like
the bytecode itself (a VM does not re-verify that, say, a local the code
treats as a cell holds a cell): the checks catch damage, truncation and
mismatched programs, not deliberately forged files.

### 3.5 Limits

| | reference VM | C VM |
|---|---|---|
| call frames | 2000 | `TILVM_MAX_FRAMES` |
| recovery points | - | `TILVM_MAX_TRY` |
| handler entries | 32 | `TILVM_MAX_HANDLERS` |
| globals + value stack | - | `TILVM_STACK_SIZE` |
| string length | - | 65535 bytes (arena <= 64 KB), else 16 MB |
| array / map | - | 65535 elements / 32767 entries |
| heap | - | the program's live objects must fit `TILVM_HEAP_SIZE` |

## 4. Walkthrough: start on a PC, continue on a calculator

The `.tbc` is the program on both sides, so build it once:

```
$ til build --target vm --strip examples/03_functions.til -o prog.tbc
$ tilvm --save-portable-after 200 state.tilp prog.tbc       # desktop C VM (64-bit ints, doubles)
...first part of the output...
tilvm: paused after 200 steps; saved state.tilp (176 bytes)
```

(`til exec prog.tbc --pause-after 200 --save state.tilp` does the same on the
reference VM - and writes the same 176 bytes.)

**A 32-bit calculator build** (NumWorks / TI-Nspire class: `int32`, `float`):

```
$ cc -O2 -DTILVM_INT_BITS=32 -DTILVM_FLOAT=1 -DTILVM_HEAP_SIZE=16384 \
     -o tilvm32 vm/tilvm.c vm/main.c -lm
$ ./tilvm32 --resume-portable state.tilp prog.tbc
...the rest of the output...
```

**ARM Cortex-M on QEMU** (`vm/ports/cortex-m`, the stand-in for ARM
calculators and microcontrollers: Cortex-M3, int32, float, 16 KB heap, no
stdio, no clock). The port embeds the program and, with `SNAPSHOT=`, the
snapshot; `main_arm.c` then calls `tilvm_load_portable` before running:

```
$ make -C vm/ports/cortex-m clean
$ make -C vm/ports/cortex-m TBC=$PWD/prog.tbc SNAPSHOT=$PWD/state.tilp run
...the rest of the output...
```

(The concatenated output of the two parts equals an uninterrupted run; the
QEMU test in `tests/test_portable.py` checks exactly this.)

On a real device the snapshot comes from a file or a serial link instead:
`tilvm_load_portable` reads it through the `get` callback a few bytes at a
time, so it never has to be in RAM as a whole. To go back to the PC, save on the
device with `tilvm_save_portable` (a `put` that writes to the file / link)
and resume with `til exec prog.tbc --resume state.tilp` or `tilvm
--resume-portable`. `tests/test_portable.py` runs these paths (the QEMU one
with `TIL_TEST_CROSS=1`).

## 5. The playground

The playground (`website/playground/`) runs the unmodified `vm/tilvm.c` as
WebAssembly, so it already has both functions. A "download snapshot /
continue on device" button needs: an exported wrapper that runs
`tilvm_run_steps(vm, n)` and, on `TILVM_YIELD`, calls `tilvm_save_portable`
with a `put` that appends to a JS-visible buffer; the page offers that
buffer as `program.tilp` next to the compiled `program.tbc` (the bytes of the
self-hosted compiler's output - the snapshot names that exact `.tbc`). On the
device, `tilvm --resume-portable program.tilp program.tbc` (or the port's
loader) continues it. The reverse direction ("continue in the browser")
calls `tilvm_load_portable` with a `get` reading an uploaded file before the
first run slice. This is not built yet.

## 6. Pause points and the two VMs

A pause happens at a step, after the step's instruction completed: after a
backward jump (at the loop's head) or after a call entered its function (at
the first instruction; parameters the caller left out are still "missing
argument" values). The C VM writes its register-cached state back there
(`TILVM_YIELD`); the reference VM raises an internal pause from its step
check after the jump target / the new frame is in place. Both VMs keep the
same state there, so neither needs a conversion beyond the index bases:

* `for x in xs` loops keep the iterated array or string and the next index in
  two hidden locals (`FOR_ITER` / `FOR_ITER_STORE`) in both VMs; map loops
  iterate a keys array made by `iter()`.
* Recovery points: the C VM stores an absolute deadline (`clock_ms` of the
  host), the reference VM a monotonic time: both are written as time left.
  Their stack depth and handler count are written as they are (stack
  relative to the globals).
* Handler entries: identical fields; `sp` and `abort_pc` are rebased (stack
  without the globals, pc relative to the code section).
* Superinstructions never end at a step in the middle of the sequence they
  replace (a fused compare-and-branch counts its backward jump like the plain
  `JUMP_IF_FALSE`), so fused and unfused programs have the same pause points.

## 7. Code size, speed and tests

Code size of `vm/tilvm.c` (`arm-none-eabi-gcc -Os -mcpu=cortex-m0 -mthumb
-DTILVM_NO_STDIO -DTILVM_INT_BITS=32`, `tilvm.o` text):

| build | without | with portable snapshots | difference |
|---|---|---|---|
| no floats (`-DTILVM_FLOAT=0`) | 32 198 | 37 386 | +5 188 bytes (+16%) |
| floats (`-DTILVM_FLOAT=1`) | 36 630 | 42 114 | +5 484 bytes |
| `TILVM_LEAN` | 24 158 | 24 166 | +8 bytes (stubs) |

`-DTILVM_NO_PORTABLE` takes the 5 KB out again (32 210 bytes). RAM: nothing
(the save works in the free part of the arena, the load in the arena).

Speed (desktop, `-O2`, 4 MB heap): saving 20 000 distinct strings in an
array (a 309 KB snapshot, written to a file) takes about 7 ms, loading it
about 5 ms; the reference VM saves it in about 20 ms. The C interpreter is
unchanged (snapshots are taken between `tilvm_run_steps` calls); the
reference VM's step check has one comparison more (the pause point).

`tests/test_portable.py`: reference VM -> C VM (desktop, int32/float, small
heap, gcstress) and back at every step, part 1 + part 2 output equal to an
uninterrupted run, byte-identical snapshots from both VMs (deadlines and the
32-bit generator of small builds aside); pauses inside nested calls,
closures, fused loops, retry/timeout (time left), effect handlers (inside a
clause and inside a handled body), graphics (screen re-created), maps
(insertion order), shared and cyclic references, strings over 64 KB (and a
1 MB heap), several hops (reference VM -> C VM -> reference VM -> int32 C VM
-> reference VM), the rescan / walk fallbacks (`TILVM_P_SCRATCH`), a 2 KB
heap, a wrong program, damaged files (the same error code from both VMs),
64-bit integers into 16/32-bit VMs, floats into a VM without them, too large
screens, `TILVM_NO_EFFECTS`, lean builds, the C API from a host program
(save twice, a failing `put`, continue after saving), the command line in
three languages, and the Cortex-M QEMU port (`TIL_TEST_CROSS=1`).
