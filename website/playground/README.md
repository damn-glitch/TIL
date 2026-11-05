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
   configuration, and `"graphics"`: whether `vm/tilvm.c` implements the
   screen/key natives - the page marks the screen examples when it does not).
   `tests/test_playground.py` fails when the committed build no
   longer matches its sources: after changing `vm/tilvm.c`, `vm/tilvm.h`,
   `selfhost/tilc.til`, the build files here, or `src/til.py` in a way that
   changes the bytecode of tilc or the examples, run `build.sh` and commit the
   outputs.

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

* Both instances run in time slices: `vm.run(0, 8)` returns after about 8 ms.
  Inside, `wh_run` runs sub-slices of a few steps (loop iterations and calls)
  and reads the clock after each; the sub-slice length follows the measured
  time per step (doubling while steps are cheap, at most 32 steps; shrinking
  at once, down to one step, when they become costly), so a call stays short
  even when every iteration sorts or copies a big list. Slices are scheduled
  as `MessageChannel` tasks (`setTimeout` in background tabs), so the browser
  renders and handles input between them and **Stop** takes effect within a
  slice. Output is flushed and the screen is presented once per animation
  frame; the output panel keeps the newest 400 000 characters.
  What can still block: one step that is costly by itself (one native on a
  huge value, e.g. sorting a million numbers) runs to its end, and after a
  long run of cheap steps the first costly steps can take up to 32 times
  their own time once before the sub-slices shrink.
* Files: `read_file`/`write_file`/`file_exists` go to an in-memory file system
  shared by both instances (a fresh one per run); the program can read
  `/main.til` and the files it writes. `args()` of the program is empty.
* Runtime errors are the VM's own reports (`runtime error: ...` and
  `at f (line N)`); the first line number is highlighted in the editor. Compile
  errors are whatever the compiler prints; the line is taken only from
  diagnostics about the program: `--> /main.til:LINE:COL` or
  `/main.til:LINE: ...` (`error:`, `unsupported:`, ...). When the compiler
  itself fails (a runtime error inside tilc, e.g. its call depth for a very
  deeply nested expression), the page says "Compiler failed (internal error)"
  and highlights no line: the lines in that trace are tilc's own.
* Screen: `screen_init`/`draw_*`/`get_pixel` are rasterized by the VM core,
  which calls the host hooks `gfx_init`/`gfx_pixel`/`gfx_get`/`gfx_show`;
  `wasm_host.c` keeps a back buffer and a front buffer (RGBA) and
  `screen_show()` copies one to the other, so the canvas never shows half-drawn
  frames. When the program ends (also by an error or Stop) the canvas shows the
  framebuffer as it is, whether or not `screen_show()` was called after the
  last drawing - like the desktop VM's `--screen FILE.ppm`. The canvas is
  scaled by an integer factor with `image-rendering: pixelated`.
* Keys: while the screen (or the output panel) has the focus, key presses are
  queued for `key_pressed()`/`wait_key()`: arrows → 1-4, Enter 13, Esc 27,
  Backspace 8, other keys their character (Unicode code point). Tab and
  Shift+Tab are not sent to the program: they move the focus as usual, so a
  keyboard user can always leave the screen (no keyboard trap) and reach
  **Stop**. Esc goes to the program (games use it to quit); in the editor it
  stops the program. The queue holds 256 keys. Touch screens get on-screen
  keys under the screen (arrows, Space, Enter, Esc).
* `input()` shows an input line under the output; the typed line is echoed
  into the output. Ctrl+D in the input line means end of input, as in a
  terminal: text already typed is delivered first (without a newline), then
  `input()` returns `""`. Lines are delivered like `fgets` with the VM's
  256-byte buffer (a longer line arrives in pieces, as on the desktop VM).
* `sleep_ms(ms)` waits with a timer for the whole milliseconds and short tasks
  for the rest, so sleeps take about as long as on the desktop (not a whole
  animation frame each). Long sleeps behave like the desktop VM's - e.g.
  `sleep_ms(1 << 31)` waits 2^31 ms - but the page stays responsive and Stop
  works.
* Sharing: **Share** puts the source into the URL hash (`#z=` deflate-raw +
  base64url when `CompressionStream` exists, else `#s=` plain base64url) and
  copies the link. `#example=ID` opens a bundled example. The code the user
  last edited is kept in `localStorage` (the draft). Examples and links never
  overwrite it: when one replaces the user's own code, a message offers Undo
  and the examples menu offers "Your previous code" (also after a reload), and
  the draft is only replaced once the user edits the new code.
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
`wh_run`: `WAIT_KEY`, `WAIT_LINE`, `SLEEP`), all with one mechanism:

* When the program would have to wait (no key or line queued yet; the
  busy-wait loop of `sleep_ms`), the hook moves the current frame's saved pc
  back to the start of the native-call instruction and calls the JavaScript
  import `suspend`, which throws. The exception unwinds the WebAssembly stack
  back to `tilvm.js`, which restores `__stack_pointer` (exported by the linker
  for this) and returns the pause status. The native had not changed anything
  yet, so the next `wh_run` executes the same instruction again: now the key
  or line is there, or the sleep is over.
* `input(prompt)` prints its prompt before it reads; the repeated print is
  swallowed while the instruction runs again.
* `sleep_ms(ms)` is recognised by its clock reads: the native computes
  `until = now() + ms` and then reads the clock again at the very same spot
  (same frame pc, step count, stack pointer, frames, recovery points). That
  second read pauses the program; the page waits `wh_sleep_ms()` real
  milliseconds. When the instruction runs again, its first clock read returns
  the original start (so `until` is the original deadline) and later reads at
  least the deadline, so the wait ends at once. The program's clock never goes
  back. Clock reads elsewhere (`timeout()` set-up, deadline checks) can look
  like the first read of a sleep (the frame's saved pc may still point after an
  earlier `sleep_ms` there), but they never read the clock twice at one spot,
  so they never pause.
* The `idle` hook (energy saving, design section 11) never pauses and does
  nothing: at a loop back-edge the frame's saved pc is stale, so the program
  could not resume at the right instruction (tested with a VM that idles at
  back-edges).

**Requirements on `vm/tilvm.c`** (true for TIL 3.0/3.1, also for the
computed-goto VM with superinstructions of design section 13; checked by
`tests/test_playground.py`):

1. Before calling a native, the VM stores the pc after the instruction's
   operands in the current frame (`fr->pc = pc;` / `SAVE_PC()`), and the
   native calls `read_line`/`wait_key`/the clock before it changes the stack.
2. Natives are called by `NATIVE id argc` (0x55) or the superinstruction
   `NATIVE_POP id argc` (0x86), both 3 bytes. A new or renumbered
   native-call superinstruction must be added to `in_native()` in
   `wasm_host.c`.
3. `sleep_ms` keeps reading the clock while it waits (`until = now + ms;
   while (now < until) ...`, with or without `idle` calls in the loop). A
   `sleep_ms` that only calls `idle(ms)` would not wait in the browser
   (`tests/test_playground.py::test_sleep_returns_to_the_event_loop` fails).
4. The C library the VM may use is limited to `shim/`; `build.sh` fails with
   the name of any missing function (add it to `shim/`).

If 1 or 2 does not hold, `wait_key()` and `input()` stop the program with
"`<name>() cannot pause in this VM build`" (an explicit error, never a wrong
value) and `sleep_ms()` busy-waits as on a desktop.

## Limits and differences from the desktop VM

* **Float formatting**: `TILVM_NO_STDIO` builds (this one, and calculator
  builds) format floats with the VM's own printf-free code
  (`fmt_gp`/`fmt_fixed`/`fmt_exp` in `vm/tilvm.c`), which is not exact: it
  differs from printf for exact ties and near-ties, `-0.0`, 17 significant
  digits and huge fixed-point numbers, e.g. `print(123456.5)` gives `123457`
  (desktop `123456`), `print(2877580 / 8.0)` gives `359697` (desktop
  `359698`), `f"{0.005:.2f}"` gives `0.00` (desktop `0.01`),
  `f"{1.0 / 3.0:.17f}"` ends in `...330` (desktop `...331`), `print(-0.0)`
  gives `0`. In a sample of about 8000 ordinary values (`i / 7.0`, `i * 1.1`,
  `n / 8.0`, ...) printed with `%g`, `.1f`, `.2f` and `.3e`, about 0.5%
  printed differently. `tests/test_playground.py::test_float_formatting_matches_printf`
  tracks this as an expected failure.
* **Float parsing** (`float(s)`, `s.to_float()`) is exact, as on the desktop
  and in the reference VM: same syntax (exponents, `inf`, `nan`, `_` between
  digits, ASCII whitespace) and the correctly rounded value, in every build of
  `vm/tilvm.c`.
* Transcendental functions (`sin`, `cos`, `tan`, `log`, `exp`, `pow` with a
  fractional exponent) come from the browser's `Math`. Their special cases
  (infinities, NaN, zeros) are those of C99 - `pow` corrects the two where
  JavaScript differs (`1 ** y` and `(-1) ** inf` are 1) - but the last bit of
  ordinary results often differs from the desktop C library: in a sample of
  1800 calls, about 6% (`pow`, `exp`, `log` most, then `sin`, `cos`, `tan`).
  The default 6-digit printing hides it; `==` and 17-digit output do not.
  `sqrt`, `floor`, `ceil` are exact WebAssembly instructions, `%` on floats is
  exact.
* Memory: 8 MB heap per VM (the desktop default is 256 KB), 8192 stack slots,
  1000 call frames, 2 MB of bytecode, 16384 constants.
* `clock_ms()` is wall-clock time since the program started (the desktop VM
  uses CPU time). `timeout()` deadlines keep running while the program waits
  for a key or a line.
* `energy_mode()` saves nothing in a browser tab (idling inside a time slice
  is a no-op); `sleep_ms()` really waits.
* Responsiveness: see "How the page runs a program" for what can still block
  (one step that is costly by itself).
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
  code, step count) on targeted programs and on all examples, and float
  parsing in all three VMs (and bit for bit against Python's `float()` in
  every C build, halfway points included); `pow` special cases;
* checks that `input()`, `sleep_ms()` (also of 2^31 ms) and `wait_key()`
  return control to JavaScript, that time slices stay short with costly
  loop iterations, that energy-saving idles at back-edges are harmless, that a
  VM that cannot pause is reported, the `--keys` syntax of the Node runner and
  the final framebuffer;
* runs the Playwright spec: hello world, the line of a runtime error, Stop on
  an endless loop and on costly iterations, `input()` and Ctrl+D, share links
  (`#z=`, `#s=`, `#example=`), a refused cross-origin compiler, the theme, the
  editor's indentation, drafts kept when examples or links replace them,
  canvas (also without `screen_show()`), keys and Tab leaving the screen,
  sleep timing, compile errors (with a line, `unsupported:`, a crashing
  compiler), and the whole compiler-VM → program-VM pipeline with a stand-in
  for `tilc.tbc` (`test/mock_programs.json`).

While `vm/tilvm.c` has no screen/key natives, the tests paste
`test/mock_gfx.inc.c` (test only) into a temporary copy of the VM so the
canvas and key handling are still exercised; with the real natives the mock is
not used. The checks of the committed build need nothing; the rest is skipped
without node, Playwright or a wasm-capable clang.
