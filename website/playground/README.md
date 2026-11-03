# TIL Playground (`website/playground/`)

A static page that compiles and runs TIL in the browser. Nothing is sent to a
server: the TIL VM (`vm/tilvm.c`) is compiled to WebAssembly, the self-hosted
compiler (`selfhost/tilc.til`, compiled to `tilc.tbc`) runs inside that VM, and
the compiled program runs in a second VM instance. No CDN, no framework, no
web fonts: the directory works offline from any static web server.

```
website/playground/
  index.html, playground.css, playground.js   the page (editor, output, canvas, examples, sharing)
  tilvm.js                                     JavaScript API of tilvm.wasm (browser and Node)
  wasm_host.c                                  WebAssembly host: VM instances, hooks, pausing
  shim/string.h, shim/math.h, shim/libc.c      freestanding C library for clang --target=wasm32
  build.sh                                     builds everything below (reproducible)
  tilvm.wasm, build-info.json                  build output (committed: the site is static)
  tilc.tbc                                     build output, once selfhost/tilc.til exists
  examples/*.til, examples/index.json          bundled examples
  examples/*.tbc                               fallback: examples precompiled by src/til.py
                                               (only while there is no tilc.tbc)
  test/                                        Playwright spec, Node runner, test fixtures
```

## Building

```
website/playground/build.sh
```

needs `clang` with the wasm32 target and `wasm-ld` (lld), and `python3`. It

1. compiles the **unmodified** `vm/tilvm.c` + `wasm_host.c` + `shim/libc.c` with
   `clang --target=wasm32 -O2 -std=c99 -ffreestanding -nostdlib -mbulk-memory`
   and `-DTILVM_NO_STDIO -DTILVM_INT_BITS=64 -DTILVM_FLOAT=2` (int64 + double),
   an 8 MB heap per VM, 2 VM instances (`TILVM_WASM_HEAP`, `TILVM_WASM_VMS`),
   and checks that the module imports nothing but what `tilvm.js` provides;
2. compiles `selfhost/tilc.til` to `tilc.tbc` with `src/til.py` when that file
   exists (otherwise prints a message and skips it);
3. while there is no `tilc.tbc`: precompiles `examples/*.til` to
   `examples/*.tbc` (development fallback, see below); once `tilc.tbc` is built
   the fallback files are removed;
4. writes `build-info.json` (sha256 of every input and output, clang version,
   configuration). `tests/test_playground.py` fails when the committed build no
   longer matches its sources: after changing `vm/tilvm.c`, `vm/tilvm.h`,
   `selfhost/tilc.til` or this directory, run `build.sh` and commit the outputs.

Options (environment): `OUT=dir` (build elsewhere), `TILVM_C=path/tilvm.c` (a
candidate VM; its `tilvm.h` is taken from the same directory), `TILC_TIL=path`
(a candidate compiler), `EXAMPLES=0`, `WERROR=0`, `CLANG`, `PYTHON`.

The output is reproducible with the same clang version (no paths or dates are
embedded; bytecode does not depend on file paths). The memory of the module is
about 21.5 MB (two VMs with an 8 MB heap each, two 2 MB bytecode buffers, a
1 MB C stack); screen buffers are allocated on demand.

## How the page runs a program

```
          source (editor)
                │  virtual file system {"/main.til": source}
                ▼
  VM instance A: tilc.tbc   args ["/main.til", "/main.tbc"]
                │  /main.tbc (or compile errors: tilc's output, line highlighted)
                ▼
  VM instance B: /main.tbc  output panel, canvas, keys, input line
```

* Both instances run in slices: `vm.run(n)` executes at most `n` steps (loop
  iterations and calls) and returns. `playground.js` runs slices for about 10 ms
  per animation frame (`requestAnimationFrame`, `setTimeout` in background tabs)
  and adapts `n` to the machine, so the page stays responsive and **Stop** takes
  effect immediately. Output is flushed and the screen is presented once per
  frame.
* Files: `read_file`/`write_file`/`file_exists` go to an in-memory file system
  shared by both instances (a fresh one per run); the program can read
  `/main.til` and the files it writes. `args()` of the program is empty.
* Runtime errors are the VM's own reports (`runtime error: ...` and
  `at f (line N)`); the first line number is highlighted in the editor. Compile
  errors are whatever the compiler prints; the line is taken from
  `--> FILE:LINE:COL`, `FILE:LINE: error` or `(line N)`.
* Screen: `screen_init`/`draw_*`/`get_pixel` are rasterized by the VM core,
  which calls the host hooks `gfx_init`/`gfx_pixel`/`gfx_get`/`gfx_show`;
  `wasm_host.c` keeps a back buffer and a front buffer (RGBA) and
  `screen_show()` copies one to the other, so the canvas never shows half-drawn
  frames. The canvas is scaled by an integer factor with `image-rendering:
  pixelated`.
* Keys: while the screen (or the output panel) has the focus, key presses are
  queued for `key_pressed()`/`wait_key()`: arrows → 1-4, Enter 13, Esc 27,
  Backspace 8, Tab 9, other keys their character code. The queue holds 256
  keys.
* `input()` shows an input line under the output; the typed line is echoed
  into the output. Ctrl+D in the input line means end of input (`input()` then
  returns `""`). Lines are delivered like `fgets` with the VM's 256-byte
  buffer (a longer line arrives in pieces, as on the desktop VM).
* Sharing: **Share** puts the source into the URL hash (`#z=` deflate-raw +
  base64url when `CompressionStream` exists, else `#s=` plain base64url) and
  copies the link. `#example=ID` opens a bundled example. The current text is
  also kept in `localStorage`.
* `?compiler=/path/to/compiler.tbc` (same origin only) uses another compiler
  with tilc's interface; the tests use it with a stand-in compiler.

### Development fallback (no `tilc.tbc` yet)

Without `tilc.tbc` the page says so (footer and output) and runs only the
**unmodified** bundled examples, from `examples/*.tbc` precompiled by
`src/til.py`. Edited code gets a clear "no in-browser compiler" message. When
`selfhost/tilc.til` exists, `build.sh` builds `tilc.tbc`, the fallback files
disappear and every program (examples included) is compiled by tilc in the
browser. No change to the page is needed.

## Pausing without blocking the browser

A desktop VM blocks inside `wait_key()`, `input()` and `sleep_ms()`; a browser
must never block. `wasm_host.c` turns these into pauses (status codes of
`wh_run`: `WAIT_KEY`, `WAIT_LINE`, `SLEEP`):

* `wait_key()` / `input()`: when no key/line is queued, the hook moves the
  current frame's saved pc back to the start of the `NATIVE` instruction and
  calls the JavaScript import `suspend`, which throws. The exception unwinds
  the WebAssembly stack back to `tilvm.js`, which restores `__stack_pointer`
  (exported by the linker for this) and returns the pause status. The native
  had not changed anything yet, so when the key/line arrives the instruction
  simply runs again and the hook returns the value. The prompt that
  `input(prompt)` printed is not printed twice (it is swallowed while the
  instruction re-runs).
* `sleep_ms(ms)`: the VM waits by reading the clock hook. Inside a `sleep_ms`
  native the hook lets the wait end at once and asks the VM to stop right after
  the native (`st.halted = 2`, the flag `exit()` uses; `wh_run` clears it and
  reports `SLEEP`); the page then waits `wh_sleep_ms()` real milliseconds. The
  program's clock never goes back. The `idle` hook (energy saving) is handled
  the same way inside `sleep_ms` and ignored elsewhere.

**Requirements on `vm/tilvm.c`** (true for TIL 3.0/3.1; checked by
`tests/test_playground.py`):

1. Before calling a native, `OP_NATIVE` stores the pc after its operands in the
   current frame (`fr->pc = pc;`), and a native calls `read_line`/`wait_key`
   before it changes the stack. If a future VM breaks this, `wait_key()` and
   `input()` stop the program with "`<name>() cannot pause in this VM build`"
   (an explicit error, never a wrong value); `sleep_ms()` then busy-waits.
2. After a native, `st.halted == 2` makes `tilvm_run_steps` return
   `TILVM_EXIT` (needed by `exit()` anyway).
3. The C library the VM may use is limited to `shim/`; `build.sh` fails with
   the name of any missing function (add it to `shim/`).

The host stores the program's bytecode, so these instructions (`NATIVE id
argc`, 3 bytes) are checked before pausing; a superinstruction that fuses a
native call would be detected and reported, not mis-executed.

## Limits and differences from the desktop VM

* **Float formatting**: `TILVM_NO_STDIO` builds (this one, and calculator
  builds) format floats with the VM's own printf-free code
  (`fmt_gp`/`fmt_fixed`/`fmt_exp` in `vm/tilvm.c`), which differs from printf
  for exact and near ties, `-0.0` and huge fixed-point numbers, e.g.
  `print(123456.5)` gives `123457` (desktop `123456`), `f"{0.005:.2f}"` gives
  `0.00` (desktop `0.01`), `print(-0.0)` gives `0`. Common values print the
  same (`tests/test_playground.py::test_float_formatting_matches_printf`
  tracks this as an expected failure).
* Transcendental functions (`sin`, `cos`, `tan`, `log`, `exp`, `pow` with a
  fractional exponent) come from the browser's `Math`; the last digit can
  differ from the desktop C library in rare cases. `sqrt`, `floor`, `ceil`
  are exact WebAssembly instructions, `%` on floats is exact.
* Memory: 8 MB heap per VM (the desktop default is 256 KB), 8192 stack slots,
  1000 call frames, 2 MB of bytecode, 16384 constants.
* `clock_ms()` is wall-clock time since the program started (the desktop VM
  uses CPU time). `timeout()` deadlines keep running while the program waits
  for a key or a line.
* `energy_mode()` saves nothing in a browser tab (idling inside a time slice
  is a no-op); `sleep_ms()` really waits.
* Screen: one screen shared by the VM instances, up to 4096x4096 pixels
  (memory is grown on demand; a larger request stops the program with an
  explicit error).
* Programs cannot read the network or the user's files; `args()` is empty.
* The page needs a browser with WebAssembly (bulk-memory: Chrome 75, Firefox 79,
  Safari 15 or newer) and must be served over http(s) (`fetch` of the `.wasm`
  does not work from `file://`). Background tabs run programs slowly (browsers
  throttle timers).

## Tests

```
python3 -m pytest tests/test_playground.py -v          # everything below
node website/playground/test/playground.spec.mjs       # browser tests on the site as committed
node website/playground/test/tilvm_node.js prog.tbc     # run a program on tilvm.wasm (like vm/main.c)
```

`tests/test_playground.py` builds the playground into a temporary directory
(never into the tree), then

* compares the WebAssembly VM with the desktop C VM (output, errors, exit
  code, step count) on targeted programs and on all examples;
* checks that `input()`, `sleep_ms()` and `wait_key()` return control to
  JavaScript, and that a VM that cannot pause is reported;
* runs the Playwright spec: hello world, the line of a runtime error, Stop on
  an endless loop, `input()`, share links, canvas, keys, compile errors, and
  the whole compiler-VM → program-VM pipeline with a stand-in for `tilc.tbc`
  (`test/mock_programs.json`).

While `vm/tilvm.c` has no screen/key natives, the tests paste
`test/mock_gfx.inc.c` (test only) into a temporary copy of the VM so the
canvas and key handling are still exercised; with the real natives the mock is
not used. The checks of the committed build need nothing; the rest is skipped
without node, Playwright or a wasm-capable clang.
