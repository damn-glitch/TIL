# TIL graphics and keys

TIL 3.1 programs can draw on a screen and read keys — on a desktop terminal, in
an image file, on a calculator or a microcontroller display. Fourteen builtin
functions (bytecode natives 40–53) cover it:

```til
screen_init(96, 64)                 # open a 96 x 64 pixel screen, all black
screen_clear(0x1E3A5F)              # colours are 0xRRGGBB ints
draw_line(0, 63, 95, 0, 0xFFD23F)
draw_rect(10, 10, 30, 20, 0xFFFFFF)
fill_rect(12, 12, 26, 16, 0xE63946)
draw_circle(70, 32, 12, 0xA8DADC)
draw_text(2, 2, "TIL 3.1", 0xFFFFFF)
screen_show()                       # present the frame
print(get_pixel(70, 20))            # 11066076 (0xA8DADC)
```

**The VM draws every pixel itself.** Lines, rectangles, circles and the
built-in 5×7 font are rasterized and clipped inside the virtual machine — the
reference VM (`til run --vm`) and the C VM (`vm/tilvm.c`) use the same
algorithms and produce identical pixels. A device only has to store a pixel
and return one (four small hooks), so a picture looks the same on every host
that can show 24-bit colour. The test-suite compares the PPM images of both
VMs byte for byte (`tests/test_graphics.py`).

Contents: [running](#running-a-graphics-program) ·
[reference](#reference) · [coordinates and clipping](#coordinates-sizes-and-clipping) ·
[colours](#colours) · [drawing rules](#exact-drawing-rules) · [text](#text) ·
[keys](#keys) · [errors](#errors) · [build options and code size](#build-options-and-code-size) ·
[writing a host](#writing-a-host-c-api) · [device notes](#device-integration-notes)

## Running a graphics program

```bash
til run --vm game.til                         # headless: draws into an offscreen framebuffer
til run --vm game.til --screen out.ppm        # save the final frame as an image (PPM, P6)
til run --vm game.til --screen-ansi           # draw every frame in the terminal
til run --vm game.til --keys "UP,UP,ENTER,a"  # scripted key presses (tests, demos)
til run --cvm game.til --screen-ansi          # the same on the C VM
til exec game.tbc --screen out.ppm            # compiled bytecode
vm/tilvm --screen out.ppm --keys "UP,ESC" game.tbc
```

* **Headless by default.** Without a screen option, drawing goes to an
  offscreen framebuffer, `screen_show()` shows nothing and `get_pixel` reads
  the framebuffer — so graphics programs run (and can be tested) in CI.
* `--screen FILE.ppm` writes the framebuffer as it is when the program ends
  (normal end, runtime error or `exit()`), whether or not `screen_show()` was
  called after the last drawing. If the program never called `screen_init`,
  no file is written and a note goes to stderr. A file that cannot be written
  is an error (exit code 2 unless the program already failed).
* `--screen-ansi` draws the screen in the terminal on every `screen_show()`:
  each character cell shows two pixels with the upper half block `▀`
  (foreground = upper pixel, background = lower pixel) in 24-bit colour. The
  first frame after `screen_init` clears the terminal (`ESC[2J`), every frame
  starts at the top-left corner (`ESC[H`). Both VMs write byte-identical
  frames into the program's standard output. It needs a terminal with 24-bit
  colour (most Linux/macOS terminals, Windows Terminal); a 64×48 screen is 64
  columns × 24 rows.
* `til run game.til` (native backend) runs graphics programs on the VM: the
  native C backend has no screen, so the program falls back to the VM, and any
  of `--screen`, `--screen-ansi`, `--keys` selects the VM directly.
* Ctrl-C ends a program; the terminal is restored (exit code 130).

## Reference

| function | result | arguments |
|---|---|---|
| `screen_init(w, h)` | nil | open (or resize) a `w`×`h` screen, all black; `w`, `h` in 1..4096 |
| `screen_clear()` / `screen_clear(color)` | nil | fill the screen (default black) |
| `draw_pixel(x, y, color)` | nil | one pixel; any int coordinates (off-screen: nothing) |
| `draw_line(x0, y0, x1, y1, color)` | nil | Bresenham line, both end points included |
| `draw_rect(x, y, w, h, color)` | nil | 1-pixel outline of the `w`×`h` rectangle at (x, y) |
| `fill_rect(x, y, w, h, color)` | nil | filled rectangle |
| `draw_circle(cx, cy, r, color)` | nil | midpoint circle outline |
| `draw_text(x, y, text, color)` | nil | 5×7 font, (x, y) = top-left corner; `text` may be any value |
| `get_pixel(x, y)` | int | colour of a pixel; 0 outside the screen |
| `screen_show()` | nil | present the frame (terminal / device) |
| `screen_width()`, `screen_height()` | int | size of the screen; 0 before `screen_init` |
| `key_pressed()` | int | a waiting key or 0 — never blocks |
| `wait_key()` | int | waits for a key |

Key codes: `UP=1 DOWN=2 LEFT=3 RIGHT=4`, otherwise the character (`ENTER=13
ESC=27 SPACE=32`, `BACKSPACE=8`, `TAB=9`, letters and digits as ASCII,
non-ASCII characters as their Unicode code point).

All of them are `io` effects: a `#[pure]` function cannot draw or read keys
(`E-EFFECT`). The compiler checks the number of arguments (`E-ARITY`); the VMs
check the values (see [Errors](#errors)).

Drawing needs `screen_init` first; otherwise `screen_clear`, `draw_*`,
`fill_rect`, `get_pixel` and `screen_show` stop with the runtime error
`screen not initialized (call screen_init)`. (There is no automatic screen:
the right size depends on the device.) `screen_width()`, `screen_height()`
and the key functions work without a screen.

## Coordinates, sizes and clipping

* (0, 0) is the top-left pixel; x grows to the right, y downwards.
* Everything is clipped: pixels outside the screen are skipped, so shapes may
  lie partly or completely outside it.
* Coordinates, widths, heights and the radius of `draw_line`, `draw_rect`,
  `fill_rect`, `draw_circle` and `draw_text` must be in **-32768..32767**
  (otherwise a runtime error). This keeps every intermediate value inside 32
  bits on 8- and 16-bit CPUs and bounds the work of one call; the screen is
  at most 4096 pixels wide, so there is plenty of room off-screen.
  `draw_pixel` and `get_pixel` accept any int.
* A rectangle with `w <= 0` or `h <= 0` draws nothing; a circle with `r < 0`
  draws nothing, `r = 0` is one pixel.
* `screen_init` may be called again: the screen gets the new size and is
  cleared to black.

## Colours

Colours are ints `0xRRGGBB` (`0x000000` black … `0xFFFFFF` white). A colour
outside `0..0xFFFFFF` is a runtime error (it is almost always a bug, e.g. an
`0xRRGGBBAA` value). `get_pixel` returns the same form.

**16-bit int builds** (`TILVM_INT_BITS=16`, e.g. the `tiny` profile for 8-bit
CPUs) cannot hold 24-bit values, so there a colour is an **RGB565** value: the
int's 16 bits read as `RRRRRGGGGGGBBBBB`. Values from 0x8000 up are negative
ints, so write them as negative numbers or build them with bit operations:

| colour | 24-bit (32/64-bit builds) | RGB565 (16-bit builds) |
|---|---|---|
| black | `0x000000` | `0` |
| white | `0xFFFFFF` | `-1` (0xFFFF) |
| red | `0xFF0000` | `-2048` (0xF800) |
| green | `0x00FF00` | `2016` (0x07E0) |
| blue | `0x0000FF` | `31` (0x001F) |
| yellow | `0xFFFF00` | `-32` (0xFFE0) |

```til
rgb565(r: int, g: int, b: int) -> int       # r, g, b in 0..255; wraps into 16 bits
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
```

The VM converts RGB565 to `0xRRGGBB` for the host by repeating the high bits
(`r8 = r5 << 3 | r5 >> 2`, `g8 = g6 << 2 | g6 >> 4`, `b8 = b5 << 3 | b5 >> 2`)
and back for `get_pixel` (`r8 >> 3`, ...), so a colour written and read again
is unchanged, and a 16-bit program draws the same picture as a 24-bit
program that uses the expanded colours (tested). Every 16-bit value is a
valid colour there; there is no colour range error.

Hosts with fewer colours (8-bit palettes, monochrome LCDs) convert when they
store a pixel ([device notes](#device-integration-notes)); `get_pixel` then
returns what the device stores. Programs that must run everywhere should keep
their game state in variables and not read it back from the screen.

## Exact drawing rules

The algorithms are part of the language, so any new VM (the WebAssembly
playground, a JavaScript port, ...) can produce the same pixels. Reference
implementations: `gfx_line/gfx_circle/gfx_text` in `vm/tilvm.c` and
`gfx_line_points/gfx_circle_points/gfx_text_points` in `src/til.py`.

**Line** — integer Bresenham for all octants (the error-term formulation that
treats x and y alike):

```
dx = |x1 - x0|,  sx = x0 < x1 ? 1 : -1
dy = -|y1 - y0|, sy = y0 < y1 ? 1 : -1
err = dx + dy
loop:
    plot(x0, y0)
    if x0 == x1 and y0 == y1: stop
    e2 = 2 * err
    if e2 >= dy: err += dy; x0 += sx
    if e2 <= dx: err += dx; y0 += sy
```

The line has `max(|dx|, |dy|) + 1` pixels and is 8-connected. On a tie the
first step is diagonal, so the line from A to B and the line from B to A can
differ in single pixels — both VMs draw exactly the pixels above.

**Circle** — midpoint algorithm, the eight symmetric points of each step:

```
x = 0, y = r, d = 1 - r
while x <= y:
    plot (cx±x, cy±y) and (cx±y, cy±x)
    if d < 0: d += 2x + 3
    else:     d += 2(x - y) + 5; y -= 1
    x += 1
```

**Rectangles** — `fill_rect` sets every pixel with `x <= px < x + w` and
`y <= py < y + h`; `draw_rect` sets the pixels of that rectangle that touch
its border (the top and bottom rows and the left and right columns).

**Clearing** — `screen_init` and `screen_clear()` make every pixel black;
`screen_clear(c)` sets every pixel to `c`.

## Text

`draw_text(x, y, text, color)` draws `text` in the built-in 5×7 pixel font
(a classic dot-matrix LCD style; table `FONT5X7` in both VMs):

* (x, y) is the top-left corner of the first character cell; each character
  is 5 pixels wide and 7 high, and advances 6 pixels (one blank column).
  Only the set pixels are drawn — the background stays as it is.
* `"\n"` starts a new line 8 pixels lower at the original x.
* ASCII 32..126 have glyphs. Every other character — control characters such
  as `"\t"`, and every non-ASCII UTF-8 character (`"ә"`, `"✓"`) — is drawn as
  one replacement glyph (a hollow box), one per character: a byte below 0x80
  or a UTF-8 lead byte starts a character, continuation bytes (0x80–0xBF) are
  skipped.
* `text` may be any value; it is drawn the way `print` shows it
  (`draw_text(0, 0, score, WHITE)`).
* Text is not wrapped; it is clipped at the screen edge. A string of n
  ASCII characters is `6n - 1` pixels wide.

## Keys

`key_pressed()` returns a key that is already waiting, or 0 — use it in game
loops. `wait_key()` waits for the next key. Where keys come from:

**Scripted keys (`--keys`)** — for tests, demos and CI. The list is
comma-separated; each entry is a name (any case) — `UP DOWN LEFT RIGHT ENTER
ESC SPACE TAB BACKSPACE COMMA NONE` — or exactly one character (`a`, `A`, `5`,
`ә`); spaces around entries are ignored, and `--keys ""` is an empty script.

* `key_pressed()` returns the next entry (`NONE` gives 0: "no key this
  frame"); when the script is used up it returns 0.
* `wait_key()` skips `NONE` entries and returns the next key; when the script
  is used up it returns 27 (ESC), so programs that wait for keys always end.
* A bad entry (`--keys "UP,JUMP"`, an empty entry) stops before the program
  runs: `bad --keys entry 'JUMP' ...`, exit code 2.

**The terminal** (no `--keys`, stdin is a terminal) — while the program reads
keys, the terminal is in raw mode (no echo, no Enter needed); it is restored at
exit, on Ctrl-C and while `input()` reads a line. Arrow keys (`ESC [ A..D`,
`ESC O A..D`) give 1..4, a lone ESC gives 27 (after 30 ms without a following
byte), Enter 13, Backspace 8, UTF-8 characters their code point; other escape
sequences (function keys) are ignored. Windows consoles use `_kbhit/_getwch`
(arrows are scan codes 72/80/75/77).

**Standard input that is not a terminal** (a file or a pipe) — its bytes are
decoded the same way, which makes `til run --vm game.til < keys.txt`
reproducible: `key_pressed()` and `wait_key()` both take the next key (they
wait for data from a pipe), a NUL byte is "no key", and at the end of the input
`key_pressed()` returns 0 and `wait_key()` 27.

An interactive example (run it with `--screen-ansi`; it is also run by the
test-suite with scripted keys):

```til
# move a dot with the arrow keys; SPACE changes its colour, ESC quits
const W = 32
const H = 24
const COLORS = [0xFFFFFF, 0xFF4040, 0x40FF40, 0x4080FF]

main()
    screen_init(W, H)
    var x = W / 2
    var y = H / 2
    var c = 0
    var moves = 0
    draw_pixel(x, y, COLORS[c])
    screen_show()
    loop
        let k = wait_key()
        if k == 27
            break
        if k == 1 and y > 0
            y -= 1
        elif k == 2 and y < H - 1
            y += 1
        elif k == 3 and x > 0
            x -= 1
        elif k == 4 and x < W - 1
            x += 1
        elif k == 32
            c = (c + 1) % len(COLORS)
        moves += 1
        draw_pixel(x, y, COLORS[c])
        screen_show()
    print(f"{moves} keys, dot at {x},{y}")
```

```bash
til run --vm dot.til --screen-ansi                        # play
til run --vm dot.til --keys "RIGHT,RIGHT,DOWN,SPACE,LEFT,ESC" --screen dot.ppm
# -> 5 keys, dot at 17,13
```

A real-time game loop polls instead: draw a frame, `screen_show()`,
`key_pressed()` (0 most of the time), `sleep_ms(30)`. With `--keys`, put
`NONE` entries where frames pass without a key.

## Errors

Arguments are checked in order and both VMs report the same messages (the
errors can be caught with `retry`/`else` like any runtime error):

| situation | message |
|---|---|
| drawing before `screen_init` | `screen not initialized (call screen_init)` |
| not an int | `draw_line() argument 3 must be an int, got float` |
| screen size | `screen_init() argument 1 must be 1..4096, got 5000` |
| shape coordinate/size | `draw_rect() argument 1 must be -32768..32767, got 40000` |
| colour (32/64-bit builds) | `fill_rect() argument 5 must be a color 0..0xFFFFFF, got -1` |
| key hook returned a negative code (or > 32767 on 16-bit builds) | `key code out of range` |
| C VM: the host has no screen / keys | `screen_init() is not available on this device` |
| C VM built with `TILVM_NO_GFX` | `screen_init() is not available in this VM build` |

Struct values are named by their type (`got Point`).

## Build options and code size

The graphics natives are part of every C VM build, including `TILVM_LEAN`
(calculators need them most). Options:

* `-DTILVM_NO_GFX` leaves them out; the natives then report `... is not
  available in this VM build`.
* `-DTILVM_ROM=<qualifier>` places the 480-byte font table in program memory
  on Harvard CPUs, e.g. `-DTILVM_ROM=__flash` with avr-gcc in GNU C mode (the
  AVR port does this; otherwise the table is copied to RAM at start-up).

Code size of `vm/tilvm.c` (`TILVM_NO_STDIO`, object file text):

| target | without graphics | with graphics | difference |
|---|---|---|---|
| ARM Cortex-M0, `-Os`, int32, no floats | 25.7 KB | 28.2 KB | +2.5 KB |
| same, `TILVM_LEAN` | 19.5 KB | 22.0 KB | +2.5 KB |
| same, `TILVM_LEAN`, int16 | 19.5 KB | 22.1 KB | +2.6 KB |
| AVR ATmega1284P, `-Os`, int32, float (AVR port flags) | 54.4 KB | 61.3 KB | +6.9 KB |

(The +2.5 KB include the 480-byte font. 8-bit CPUs need more code for the
32-bit arithmetic of the clipping.)

## Writing a host (C API)

A host fills in `vm->host` after `tilvm_load` (see `vm/tilvm.h`). The VM calls
the screen hooks only with on-screen coordinates, so a host never needs to
clip:

| hook | needed for | contract |
|---|---|---|
| `gfx_pixel(ctx, x, y, rgb)` | every screen function | store one pixel (`rgb` = 0xRRGGBB) |
| `gfx_get(ctx, x, y)` | `get_pixel` | return the stored pixel as 0xRRGGBB |
| `gfx_init(ctx, w, h)` | optional | create or resize a `w`×`h` screen, **all black**; without it the VM clears the screen with `gfx_pixel` |
| `gfx_show(ctx)` | optional | present the frame (copy to the LCD, flip buffers, ...); without it `screen_show()` does nothing |
| `key_pressed(ctx)` | `key_pressed` | a key code ≥ 0, 0 = none; must not block |
| `wait_key(ctx)` | optional | block until a key; without it the VM polls `key_pressed` and calls `idle(ctx, 1)` between polls |

A missing required hook makes the function report `... is not available on
this device`. `tilvm_restore` (raw snapshots) keeps the screen *size* of the
saved program and calls `gfx_init` again (the pixels are not part of a
snapshot), so install the hooks before restoring; programs that redraw every
frame continue seamlessly.

A complete host with an in-memory framebuffer — for a device, replace the
array accesses with the display driver:

```c
/* host example (tested by tests/test_graphics.py) */
#include "tilvm.h"

#define MAX_W 320
#define MAX_H 240
static uint8_t screen_rgb[MAX_H][MAX_W][3];
static int screen_w;

static void my_init(void *ctx, int w, int h) {
    int x, y;
    (void)ctx;
    screen_w = w;                                     /* a real host checks w <= MAX_W, h <= MAX_H */
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            screen_rgb[y][x][0] = screen_rgb[y][x][1] = screen_rgb[y][x][2] = 0;
}

static void my_pixel(void *ctx, int x, int y, unsigned long rgb) {
    (void)ctx;
    screen_rgb[y][x][0] = (uint8_t)(rgb >> 16);
    screen_rgb[y][x][1] = (uint8_t)(rgb >> 8);
    screen_rgb[y][x][2] = (uint8_t)rgb;
}

static unsigned long my_get(void *ctx, int x, int y) {
    (void)ctx;
    return ((unsigned long)screen_rgb[y][x][0] << 16) | ((unsigned long)screen_rgb[y][x][1] << 8)
           | screen_rgb[y][x][2];
}

static int my_key(void *ctx) {                       /* poll the keypad here */
    (void)ctx;
    return 0;
}

void install_screen(tilvm *vm) {
    vm->host.gfx_init = my_init;
    vm->host.gfx_pixel = my_pixel;
    vm->host.gfx_get = my_get;
    vm->host.key_pressed = my_key;                   /* wait_key: the VM polls key_pressed */
}
```

`vm/main.c` (the desktop runner) is a full host: framebuffer, PPM output,
terminal frames, scripted, terminal and stdin keys.

## Device integration notes

The VM core is verified in simulators and on the desktop; the snippets below
are starting points for real devices and have not been run on the hardware.
Colour conversion from the VM's `0xRRGGBB`:

```c
#define R8(c) ((unsigned)((c) >> 16) & 0xFFu)
#define G8(c) ((unsigned)((c) >> 8) & 0xFFu)
#define B8(c) ((unsigned)(c) & 0xFFu)
#define TO_RGB565(c) (uint16_t)(((R8(c) >> 3) << 11) | ((G8(c) >> 2) << 5) | (B8(c) >> 3))
#define FROM_RGB565(v) ((((unsigned long)((v) >> 11) * 255 / 31) << 16) | \
                        (((unsigned long)(((v) >> 5) & 63) * 255 / 63) << 8) | \
                        ((unsigned long)((v) & 31) * 255 / 31))
#define TO_RGB332(c) (uint8_t)((R8(c) & 0xE0) | ((G8(c) & 0xE0) >> 3) | (B8(c) >> 6))
#define LUMA(c) ((R8(c) * 77 + G8(c) * 150 + B8(c) * 29) >> 8)        /* 0..255 */
```

**TI-84 Plus CE** (eZ80, CE C toolchain, `graphx`, 320×240): graphx draws in
8-bit palette mode. Load an RGB332 palette once
(`for (i = 0; i < 256; i++) gfx_palette[i] = gfx_RGBTo1555(i & 0xE0, (i << 3) & 0xE0, (i << 6) & 0xC0);`),
then `gfx_pixel` = `gfx_SetColor(TO_RGB332(rgb)); gfx_SetPixel(x, y);`,
`gfx_get` = expand `gfx_GetPixel(x, y)` (RGB332 → 0xRRGGBB, quantized),
`gfx_init` = `gfx_Begin(); gfx_SetDrawBuffer(); gfx_FillScreen(0)` (keep
`w, h` ≤ 320×240), `gfx_show` = `gfx_SwapDraw()`. Keys: `kb_Scan()` and
`kb_Data[7] & kb_Up` … for arrows, `os_GetCSC()` for the others
(`sk_Enter` → 13, `sk_Clear` → 27). Use `TILVM_INT_BITS=32`, `TILVM_FLOAT=1`.
A 24-bit shadow buffer does not fit in RAM (230 KB), so `get_pixel` returns
the quantized colour.

**NumWorks** (Cortex-M7, external apps with `eadk.h`, 320×240):
`gfx_pixel` = `eadk_display_push_rect_uniform((eadk_rect_t){x, y, 1, 1}, TO_RGB565(rgb))`,
`gfx_get` = `eadk_display_pull_rect((eadk_rect_t){x, y, 1, 1}, &c)` then
`FROM_RGB565(c)`, `gfx_init` = `eadk_display_push_rect_uniform(eadk_screen_rect, 0)`.
Drawing goes straight to the LCD, so `gfx_show` can stay NULL. Keys:
`eadk_keyboard_scan()` + `eadk_keyboard_key_down(state, eadk_key_up)` …
(`eadk_key_ok`/`eadk_key_exe` → 13, `eadk_key_back` → 27), remembering the
previous state to report each press once. `eadk_timing_msleep(1)` as `idle`.

**Casio fx-CG50 / Graph 90+E** (SH4, gint, 396×224, RGB565 VRAM):
`gfx_pixel` = `dpixel(x, y, TO_RGB565(rgb))`, `gfx_get` =
`FROM_RGB565(gint_vram[y * DWIDTH + x])`, `gfx_init` = `dclear(C_BLACK)`,
`gfx_show` = `dupdate()`. Keys: `pollevent()` (non-blocking) / `getkey()`
(blocking); `KEY_UP/DOWN/LEFT/RIGHT` → 1..4, `KEY_EXE` → 13, `KEY_EXIT` → 27.
**fx-9860G** (monochrome 128×64): `dpixel(x, y, LUMA(rgb) >= 128 ? C_WHITE : C_BLACK)`;
`gfx_get` returns `0xFFFFFF` or `0`.

**SDL2** (desktops, Raspberry Pi): a `uint32_t fb[w*h]` with
`SDL_PIXELFORMAT_RGB888` needs no conversion (`fb[y*w+x] = rgb`); `gfx_show` =
`SDL_UpdateTexture` + `SDL_RenderCopy` + `SDL_RenderPresent`, and pump
`SDL_PollEvent` there into a key queue (`SDLK_UP` → 1, `SDLK_RETURN` → 13,
`SDLK_ESCAPE` → 27, text input → code points) that `key_pressed` pops.

**Linux framebuffer** (`/dev/fb0`, typically 32 bpp XRGB): `mmap` it and write
`rgb` (or `TO_RGB565(rgb)` for 16 bpp) at `y * line_length + x * bytes_per_pixel`;
read keys from the terminal (as `vm/main.c` does) or `/dev/input/event*`.

**Browser canvas** (WebAssembly build): keep an `ImageData` (RGBA) and write
`R8, G8, B8, 255` in `gfx_pixel`; `gfx_show` = `ctx.putImageData(img, 0, 0)`.
Run the VM in slices (`tilvm_run_steps`) from `requestAnimationFrame` so the
page can paint between frames, and feed `keydown` events (`ArrowUp` → 1,
`Enter` → 13, `Escape` → 27, `event.key` → code point) into a queue for
`key_pressed`. A browser main thread cannot block, so either leave `wait_key`
NULL and run the VM in a Web Worker (the VM polls `key_pressed`; block with
`Atomics.wait` on a shared key queue), or ask programs to poll with
`key_pressed()`.

**Monochrome LCDs** (SSD1306, Nokia 5110, TI-83/84 Plus 96×64): store
`LUMA(rgb) >= 128` as one bit per pixel; return `0xFFFFFF`/`0` from `gfx_get`.
