/*
 * tilvm - command-line runner for TIL bytecode (.tbc)
 *
 *   tilvm program.tbc [args...]           run a program (args() returns the args)
 *   tilvm --max-steps N program.tbc       sandbox: stop after N loop iterations/calls
 *   tilvm --snapshot-after N FILE p.tbc   run N steps, save the paused program to FILE
 *   tilvm --resume FILE program.tbc       continue a saved program
 *   tilvm --save-portable-after N FILE p.tbc  run N steps, save a portable snapshot (.tilp)
 *   tilvm --resume-portable FILE p.tbc    continue a portable snapshot (from any VM build
 *                                         or `til run --vm`; docs/TIL_SNAPSHOTS.md)
 *   tilvm --stats program.tbc             print memory / GC statistics to stderr
 *                                         (+ high-water marks with -DTILVM_STATS,
 *                                         + energy mode and idles if it saved energy)
 *   tilvm --screen out.ppm program.tbc    save the screen (screen_init, draw_*) as a PPM at exit
 *   tilvm --screen-ansi program.tbc       draw the screen in the terminal on every screen_show()
 *   tilvm --keys "UP,UP,ENTER,a" p.tbc    scripted keys (default: the keyboard, or stdin's bytes)
 *
 * Produce bytecode with:  til build --target vm program.til
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L         /* termios, select, isatty: keys from the terminal */
#endif
#include "tilvm.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <conio.h>
#include <io.h>
#include <windows.h>
#else
#include <sys/select.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#endif

static tilvm vm;   /* the whole VM state: one static block, no other allocations */

/* ─── screen and keys: the desktop host of the graphics natives ─────────────
 * The VM rasterizes and clips (docs/TIL_GRAPHICS.md); this host keeps the
 * framebuffer (3 bytes per pixel), writes it as a PPM image at exit (--screen),
 * draws it in a 24-bit colour terminal on every screen_show() (--screen-ansi)
 * and supplies keys: scripted (--keys), from the terminal in raw mode, or from
 * stdin's bytes when stdin is not a terminal. `til run --vm` does the same in
 * src/til.py (TILScreen, KeyScript, TerminalKeys) - byte for byte. */
static unsigned char *fb;
static int fb_w, fb_h, ansi_screen, ansi_frames;

static void gfx_init(void *ctx, int w, int h) {
    unsigned char *p = (unsigned char *)calloc((size_t)w * (size_t)h, 3);
    (void)ctx;
    if (!p) {
        fprintf(stderr, "tilvm: not enough memory for a %dx%d screen\n", w, h);
        exit(2);
    }
    free(fb);
    fb = p;
    fb_w = w;
    fb_h = h;
    ansi_frames = 0;                    /* the next frame clears the terminal */
}

static void gfx_pixel(void *ctx, int x, int y, unsigned long rgb) {
    unsigned char *p = fb + 3 * ((size_t)y * (size_t)fb_w + (size_t)x);
    (void)ctx;
    p[0] = (unsigned char)(rgb >> 16);
    p[1] = (unsigned char)(rgb >> 8);
    p[2] = (unsigned char)rgb;
}

static unsigned long gfx_get(void *ctx, int x, int y) {
    const unsigned char *p = fb + 3 * ((size_t)y * (size_t)fb_w + (size_t)x);
    (void)ctx;
    return ((unsigned long)p[0] << 16) | ((unsigned long)p[1] << 8) | (unsigned long)p[2];
}

/* --screen-ansi: one character cell shows two pixels (U+2580 upper half block:
 * foreground = upper pixel, background = lower pixel); colours only change
 * when they differ from the previous cell. */
static void gfx_show(void *ctx) {
    int x, y;
    if (!ansi_screen) return;
    fputs(ansi_frames++ ? "\033[H" : "\033[2J\033[H", stdout);
    for (y = 0; y < fb_h; y += 2) {
        long fg = -1, bg = -1;
        for (x = 0; x < fb_w; x++) {
            const unsigned char *t = fb + 3 * ((size_t)y * (size_t)fb_w + (size_t)x);
            const unsigned char *b = y + 1 < fb_h ? t + 3 * (size_t)fb_w : NULL;
            long top = (long)gfx_get(ctx, x, y), bot = b ? (long)gfx_get(ctx, x, y + 1) : -2;
            if (top != fg) printf("\033[38;2;%d;%d;%dm", t[0], t[1], t[2]);
            if (bot != bg) {
                if (b) printf("\033[48;2;%d;%d;%dm", b[0], b[1], b[2]);
                else fputs("\033[49m", stdout);
            }
            fg = top;
            bg = bot;
            fputs("\342\226\200", stdout);
        }
        fputs("\033[0m\n", stdout);
    }
    fflush(stdout);
}

/* --screen FILE: the framebuffer as it is when the program ends (binary PPM). */
static int save_screen(const char *path, int rc) {
    FILE *f;
    size_t n = (size_t)fb_w * (size_t)fb_h;
    int ok;
    if (!path) return rc;
    if (!fb) {
        fprintf(stderr, "tilvm: --screen: the program did not call screen_init (no image written)\n");
        return rc;
    }
    f = fopen(path, "wb");
    ok = f != NULL && fprintf(f, "P6\n%d %d\n255\n", fb_w, fb_h) > 0 && fwrite(fb, 3, n, f) == n;
    if (f && fclose(f) != 0) ok = 0;
    if (!ok) {
        fprintf(stderr, "tilvm: cannot write %s\n", path);
        return rc ? rc : 2;
    }
    return rc;
}

/* Keys: 0 = none, UP=1 DOWN=2 LEFT=3 RIGHT=4, ENTER=13, ESC=27, otherwise the
 * character's code point. */
static int *keys;                       /* --keys script (NULL: keyboard / stdin) */
static size_t n_keys, key_pos;
static int key_source = -1;

static const struct { const char *name; int code; } KEY_NAMES[] = {
    {"UP", 1}, {"DOWN", 2}, {"LEFT", 3}, {"RIGHT", 4}, {"ENTER", 13}, {"ESC", 27},
    {"SPACE", 32}, {"TAB", 9}, {"BACKSPACE", 8}, {"COMMA", 44}, {"NONE", 0}
};

/* Code point of a token that is exactly one UTF-8 character, else -1. */
static long utf8_char(const unsigned char *s, size_t n) {
    size_t len, i;
    long c;
    if (!n) return -1;
    c = s[0];
    len = c < 0x80 ? 1 : c < 0xC0 ? 0 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : c < 0xF8 ? 4 : 0;
    if (!len || len != n) return -1;
    if (len > 1) c &= 0x3F >> (len - 1);
    for (i = 1; i < len; i++) {
        if ((s[i] & 0xC0) != 0x80) return -1;
        c = (c << 6) | (s[i] & 0x3F);
    }
    if ((len == 2 && c < 0x80) || (len == 3 && c < 0x800) || (len == 4 && c < 0x10000) ||
        c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF))
        return -1;                       /* overlong, out of range, surrogate */
    return c;
}

static int key_code(const char *s, size_t n) {
    size_t i, j;
    long c = utf8_char((const unsigned char *)s, n);
    if (c >= 0) return (int)c;
    for (i = 0; i < sizeof(KEY_NAMES) / sizeof(KEY_NAMES[0]); i++) {
        const char *name = KEY_NAMES[i].name;
        for (j = 0; j < n && name[j]; j++)
            if ((s[j] >= 'a' && s[j] <= 'z' ? s[j] - 32 : s[j]) != name[j]) break;
        if (j == n && !name[j]) return KEY_NAMES[i].code;
    }
    return -1;
}

/* --keys "UP,UP,ENTER,a,ESC": comma-separated names (any case) or single
 * characters; "" = no keys at all. Returns 0, or -1 after printing an error. */
static int parse_keys(const char *spec) {
    size_t cap = 1;
    const char *p;
    for (p = spec; *p; p++)
        if (*p == ',') cap++;
    free(keys);
    keys = (int *)malloc(cap * sizeof(int));
    n_keys = key_pos = 0;
    if (!keys) return -1;
    for (p = spec; *p == ' ' || *p == '\t'; p++) { }
    if (!*p) return 0;
    for (p = spec;;) {
        const char *end = strchr(p, ','), *a = p, *b;
        int code;
        if (!end) end = p + strlen(p);
        b = end;
        while (a < b && (*a == ' ' || *a == '\t')) a++;
        while (b > a && (b[-1] == ' ' || b[-1] == '\t')) b--;
        code = key_code(a, (size_t)(b - a));
        if (code < 0) {
            fprintf(stderr, "tilvm: bad --keys entry '%.*s' (use UP, DOWN, LEFT, RIGHT, ENTER, ESC, SPACE, "
                            "TAB, BACKSPACE, COMMA, NONE or one character)\n", (int)(b - a), a);
            return -1;
        }
        keys[n_keys++] = code;
        if (!*end) return 0;
        p = end + 1;
    }
}

/* Ctrl-C (POSIX): the program ends after its current step - a native that is
 * running finishes first (sleep_ms() at once: host_idle is cut short) - then
 * main() flushes the output,
 * writes the --screen image, restores the terminal and ends the process by
 * SIGINT, as the reference VM does (`til run --vm`). A second Ctrl-C ends it at
 * once. The hooks below notice the flag; run_slices() covers loops that call
 * no hook. */
static volatile sig_atomic_t interrupted;

static int stop_program(void) {         /* in a hook: end the program like exit(130) */
    vm.st.halted = 2;
    vm.st.exit_code = 130;
    return 0;
}

#ifndef _WIN32
static struct termios saved_tio;
static volatile sig_atomic_t raw_mode;

static void term_restore(void) {
    if (raw_mode) {
        tcsetattr(0, TCSANOW, &saved_tio);
        raw_mode = 0;
    }
}

static void term_signal(int sig) {      /* SIGTERM: restore the terminal, then end */
    term_restore();
    signal(sig, SIG_DFL);
    raise(sig);
}

static void on_sigint(int sig) {
    if (interrupted) term_signal(sig);  /* the second Ctrl-C */
    interrupted = 1;
}

/* Keys arrive without Enter and without echo while the program reads them;
 * the terminal is restored at exit, on Ctrl-C, on SIGTERM and before input(). */
static void term_raw(void) {
    static int hooked;
    struct termios t;
    if (raw_mode || tcgetattr(0, &saved_tio) != 0) return;
    if (!hooked) {                      /* before the terminal changes */
        hooked = 1;
        atexit(term_restore);
        signal(SIGTERM, term_signal);
    }
    t = saved_tio;
    t.c_lflag &= ~(tcflag_t)(ICANON | ECHO);
    t.c_cc[VMIN] = 1;
    t.c_cc[VTIME] = 0;
    raw_mode = 1;                       /* (restoring saved_tio early is harmless) */
    if (tcsetattr(0, TCSANOW, &t) != 0) raw_mode = 0;
}
#else
/* Windows console: arrows come as 0 / 0xE0 followed by a scan code. */
static int win_key(int block) {
    for (;;) {
        int c;
        if (!block && !_kbhit()) return 0;
        c = (int)_getwch();
        if (c == 0 || c == 0xE0) {
            c = (int)_getwch();
            if (c == 72) return 1;
            if (c == 80) return 2;
            if (c == 75) return 3;
            if (c == 77) return 4;
            continue;
        }
        if (c == '\r' || c == '\n') return 13;
        return c;
    }
}
#endif

/* stdin: keys and input() lines are taken from one buffer, so they never
 * overtake each other. key_pressed() only takes bytes that are already there
 * (POSIX: select() without waiting): it never blocks, also with a pipe; a
 * regular file is always ready, so `tilvm game.tbc < keys.txt` is reproducible.
 * (Windows cannot poll a pipe: there the read waits.) */
static unsigned char in_buf[4096];
static size_t in_pos, in_len;

/* Next byte of stdin: -1 = none within `ms` milliseconds (ms < 0: wait) or
 * Ctrl-C, -2 = end of input (or stdin cannot be read). */
static int in_byte(int ms) {
    if (in_pos < in_len) return in_buf[in_pos++];
    for (;;) {
#ifndef _WIN32
        fd_set set;
        struct timeval tv;
        ssize_t n;
        int r;
        if (interrupted) return -1;
        FD_ZERO(&set);
        FD_SET(0, &set);
        tv.tv_sec = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000L;
        r = select(1, &set, NULL, NULL, ms < 0 ? NULL : &tv);
        if (r == 0) return -1;
        if (r < 0 && errno != EINTR) return -2;
        n = r < 0 ? -1 : read(0, in_buf, sizeof in_buf);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) {
            if (ms < 0) continue;              /* waiting: only Ctrl-C stops it */
            return -1;
        }
#else
        int n = _read(0, in_buf, (unsigned)sizeof in_buf);
        (void)ms;
#endif
        if (n <= 0) return -2;
        in_len = (size_t)n;
        in_pos = 1;
        return in_buf[0];
    }
}

/* One key from stdin, `c` being its first byte: ESC [ A..D (or ESC O A..D) ->
 * UP DOWN RIGHT LEFT, UTF-8 -> code point, CR and LF -> ENTER (13), DEL ->
 * BACKSPACE (8). -1 = bytes that are not a key (other escape sequences, broken
 * UTF-8). A lone ESC is one that no '[' or 'O' follows within 30 ms. */
static int decode_key(int c) {
    if (c == 27) {
        int c2 = in_byte(30), c3;
        if (c2 != '[' && c2 != 'O') {
            if (c2 >= 0) in_pos--;              /* a lone ESC: c2 is the next key */
            return 27;
        }
        c3 = in_byte(30);
        if (c3 >= 'A' && c3 <= 'D') return "\1\2\4\3"[c3 - 'A'];
        while (c3 >= 0 && (c3 < 0x40 || c3 > 0x7E)) c3 = in_byte(30);
        return -1;
    }
    if (c >= 0xC0 && c < 0xF8) {
        int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1, i;
        long cp = c & (0x3F >> extra);
        for (i = 0; i < extra; i++) {
            int b = in_byte(30);
            if (b < 0x80 || b > 0xBF) return -1;
            cp = (cp << 6) | (b & 0x3F);
        }
        return (int)cp;
    }
    if (c >= 0x80) return -1;
    if (c == '\r' || c == '\n') return 13;
    return c == 127 ? 8 : c;
}

static int pick_key_source(void) {        /* 0 script, 1 terminal, 2 stdin bytes */
    if (key_source < 0) {
#ifdef _WIN32
        key_source = keys ? 0 : _isatty(_fileno(stdin)) ? 1 : 2;
#else
        key_source = keys ? 0 : isatty(0) ? 1 : 2;
#endif
    }
    return key_source;
}

/* key_pressed(): the next scripted entry (0 when the script is used up), or
 * a key that is already waiting (0 if none; stdin bytes: 0 at the end). */
static int host_key_pressed(void *ctx) {
    int c, k;
    (void)ctx;
    if (interrupted) return stop_program();
    if (pick_key_source() == 0) return key_pos < n_keys ? keys[key_pos++] : 0;
#ifdef _WIN32
    if (key_source == 1) return win_key(0);
#else
    if (key_source == 1) term_raw();
#endif
    c = in_byte(0);
    k = c < 0 ? 0 : decode_key(c);
    if (interrupted) return stop_program();
    return k < 0 ? 0 : k;
}

/* wait_key(): the next key (scripted NONE entries are skipped); 27 (ESC) when
 * the script or the input has ended, so programs can always finish. */
static int host_wait_key(void *ctx) {
    int c, k;
    (void)ctx;
    if (interrupted) return stop_program();
    if (pick_key_source() == 0) {
        while (key_pos < n_keys)
            if (keys[key_pos++]) return keys[key_pos - 1];
        return 27;
    }
    fflush(stdout);
#ifdef _WIN32
    if (key_source == 1) return win_key(1);
#else
    if (key_source == 1) term_raw();
#endif
    for (;;) {
        c = in_byte(-1);
        k = c < 0 ? 0 : decode_key(c);
        if (interrupted) return stop_program();
        if (c == -2) return 27;
        if (k > 0) return k;
    }
}

/* input(): at most cap - 1 bytes, up to and without the newline (NUL bytes are
 * kept, as in the reference VM), from the same buffer as the keys, with the
 * terminal back in normal mode. */
static int read_line_cooked(void *ctx, char *buf, size_t cap) {
    size_t n = 0;
    (void)ctx;
#ifndef _WIN32
    term_restore();
#endif
    fflush(stdout);
    while (n + 1 < cap) {
        int c = in_byte(-1);
        if (c < 0) break;
        buf[n++] = (char)c;
        if (c == '\n') break;
    }
    if (interrupted) {
        stop_program();
        return -1;
    }
    if (!n) return -1;                          /* end of input */
    if (buf[n - 1] == '\n') n--;
    buf[n] = 0;
    return (int)n;
}

/* clock_ms(), timeouts and sleep_ms() use the wall clock (milliseconds since the
 * start): the VM core's default clock() is CPU time, which stands still while
 * the program sleeps or idles. */
#ifdef _WIN32
static uint32_t wall_ms(void) {
#if defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0600
    return (uint32_t)GetTickCount64();
#else
    return (uint32_t)GetTickCount();
#endif
}
#else
static uint32_t wall_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint32_t)((unsigned long)ts.tv_sec * 1000ul + (unsigned long)ts.tv_nsec / 1000000ul);
}
#endif
static uint32_t wall_start;

/* sleep_ms() waits on the clock, idling in between: Ctrl-C ends the program. */
static uint32_t clock_hook(void *ctx) {
    (void)ctx;
    if (interrupted) stop_program();
    return wall_ms() - wall_start;
}

/* Energy saving (docs/TIL_ENERGY.md): the VM idles 1 ms every 512 / 4096 steps
 * in the low / balanced modes, and sleep_ms() idles for the time left. The
 * process really sleeps (no CPU time); Ctrl-C interrupts the sleep. */
static void host_idle(void *ctx, unsigned long ms) {
    (void)ctx;
    if (interrupted) {
        stop_program();
        return;
    }
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    {
        struct timespec ts;
        ts.tv_sec = (time_t)(ms / 1000ul);
        ts.tv_nsec = (long)(ms % 1000ul) * 1000000L;
        nanosleep(&ts, NULL);           /* EINTR on Ctrl-C: the VM re-checks */
    }
#endif
    if (interrupted) stop_program();
}

static void install_screen_and_keys(tilvm *v) {
    v->host.gfx_init = gfx_init;
    v->host.gfx_pixel = gfx_pixel;
    v->host.gfx_get = gfx_get;
    v->host.gfx_show = gfx_show;
    v->host.key_pressed = host_key_pressed;
    v->host.wait_key = host_wait_key;
    v->read_line = read_line_cooked;
    v->host.idle = host_idle;
    if (v->clock_ms) {                  /* (TILVM_NO_TIME builds keep counting steps) */
        wall_start = wall_ms();
        v->clock_ms = clock_hook;
    }
}

/* Run up to `steps` steps (0: to the end) in slices, so that Ctrl-C also stops a
 * loop that calls no host function. Returns a TILVM_ status. */
#define RUN_SLICE 65536u

static int run_slices(uint32_t steps) {
    for (;;) {
        uint32_t n = steps && steps < RUN_SLICE ? steps : RUN_SLICE;
        int st = tilvm_run_steps(&vm, n);
        if (st != TILVM_YIELD || interrupted) return st;
        if (steps && !(steps -= n)) return TILVM_YIELD;
    }
}

static unsigned char *read_file(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    unsigned char *buf;
    long n;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (unsigned char *)malloc(n > 0 ? (size_t)n : 1);
    if (buf && n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); buf = NULL; }
    fclose(f);
    *size = n;
    return buf;
}

static void usage(void) {
    fprintf(stderr, "usage: tilvm [--max-steps N] [--snapshot-after N FILE] [--resume FILE] "
                    "[--save-portable-after N FILE] [--resume-portable FILE] [--stats] "
                    "[--screen FILE.ppm] [--screen-ansi] [--keys LIST] program.tbc [args...]\n");
}

/* Portable snapshots (.tilp): the VM streams them through these callbacks. */
static int put_file(void *ctx, const uint8_t *data, uint32_t len) {
    return fwrite(data, 1, len, (FILE *)ctx) == len ? 0 : -1;
}

static int get_file(void *ctx, uint8_t *buf, uint32_t len) {
    return fread(buf, 1, len, (FILE *)ctx) == len ? 0 : -1;
}

/* the reasons of tilvm_load_portable's error codes (src/til.py: TILP_ERRORS) */
static const char *portable_error(int rc) {
    static const char *const why[] = {
        "not a portable TIL snapshot (.tilp)",
        "unsupported snapshot version",
        "the snapshot was saved by a different program",
        "the snapshot is damaged or incomplete",
        "the program's state does not fit this VM",
        "the snapshot holds an integer that does not fit this VM",
        "the snapshot holds floats, but this VM has no floats",
        "the snapshot's screen is larger than this VM can show",
        "portable snapshots are not available in this VM build",
        "the snapshot uses effect handlers, which this VM build leaves out"
    };
    return rc <= -1 && rc >= -10 ? why[-rc - 1] : "error";
}

#ifdef TILVM_NO_STDIO
/* The VM core was built without stdio (a device profile emulated on the desktop,
 * see `til fit --run`): this runner provides the console. */
static void console_write(void *ctx, const char *data, size_t len) {
    (void)ctx;
    fwrite(data, 1, len, stdout);
}
static void console_write_err(void *ctx, const char *data, size_t len) {
    (void)ctx;
    fflush(stdout);
    fwrite(data, 1, len, stderr);
}
static int console_read_line(void *ctx, char *buf, size_t cap) {
    size_t n;
    (void)ctx;
    fflush(stdout);
    if (!fgets(buf, (int)cap, stdin)) return -1;
    n = strlen(buf);
    if (n && buf[n - 1] == '\n') buf[--n] = 0;
    return (int)n;
}
#endif

int main(int argc, char **argv) {
    const char *path = NULL, *snap_out = NULL, *resume = NULL, *screen_out = NULL;
    const char *psave_out = NULL, *presume = NULL;
    unsigned long snap_after = 0, max_steps = 0, psave_after = 0;
    int stats = 0, i, rc, prog_argc = 0;
    char **prog_argv = NULL;
    long size = 0;
    unsigned char *code;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-steps") && i + 1 < argc) max_steps = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--snapshot-after") && i + 2 < argc) {
            snap_after = strtoul(argv[++i], NULL, 10);
            snap_out = argv[++i];
        } else if (!strcmp(argv[i], "--resume") && i + 1 < argc) resume = argv[++i];
        else if (!strcmp(argv[i], "--save-portable-after") && i + 2 < argc) {
            psave_after = strtoul(argv[++i], NULL, 10);
            psave_out = argv[++i];
        } else if (!strcmp(argv[i], "--resume-portable") && i + 1 < argc) presume = argv[++i];
        else if (!strcmp(argv[i], "--stats")) stats = 1;
        else if (!strcmp(argv[i], "--screen") && i + 1 < argc) screen_out = argv[++i];
        else if (!strcmp(argv[i], "--screen-ansi")) ansi_screen = 1;
        else if (!strcmp(argv[i], "--keys") && i + 1 < argc) { if (parse_keys(argv[++i]) != 0) return 2; }
        else if (!strcmp(argv[i], "--version")) { printf("tilvm %s (bytecode v%d, int%d, heap %ld bytes)\n",
                                                        TILVM_VERSION, TILVM_BYTECODE_VERSION, TILVM_INT_BITS,
                                                        (long)TILVM_HEAP_SIZE); return 0; }
        else if (argv[i][0] == '-') { usage(); return 2; }
        else {
            path = argv[i];                     /* everything after the program is for it */
            prog_argv = argv + i + 1;
            prog_argc = argc - i - 1;
            break;
        }
    }
    if (!path) { usage(); return 2; }
    code = read_file(path, &size);
    if (!code) { fprintf(stderr, "tilvm: cannot read %s\n", path); return 2; }
    rc = tilvm_load(&vm, code, (uint32_t)size);
    if (rc != 0) {
        const char *why = rc == -3 ? ": program uses floats, VM built without them" :
                          rc == -4 ? ": too many constants for this VM build (TILVM_MAX_CONSTS)" :
                          rc == -5 ? ": truncated or malformed file" :
                          rc == -6 ? ": integer constant too large for this VM" :
                          rc == -7 ? ": too many types for this VM build (TILVM_MAX_TYPES)" :
                          rc == -8 ? ": too many globals for the value stack (TILVM_STACK_SIZE)" :
                          rc == -10 ? ": too many functions for this VM build (TILVM_MAX_FUNCS)" : "";
#ifdef TILVM_ADDR16
        if (rc == -10 && size > 0xFFFFL) why = ": program larger than 64 KB (TILVM_ADDR16 build)";
#endif
        fprintf(stderr, "tilvm: cannot load %s (error %d%s)\n", path, rc, why);
        return 2;
    }
    vm.max_steps = (uint32_t)max_steps;
    vm.host.argc = prog_argc;
    vm.host.argv = (const char *const *)prog_argv;
#ifdef TILVM_NO_STDIO
    vm.write = console_write;
    vm.write_err = console_write_err;
    vm.read_line = console_read_line;
#endif
    install_screen_and_keys(&vm);       /* before --resume: it re-creates a saved screen */

    if (resume) {
        long ssize = 0;
        unsigned char *snap = read_file(resume, &ssize);
        if (!snap || tilvm_restore(&vm, snap, (uint32_t)ssize) != 0) {
            fprintf(stderr, "tilvm: cannot restore snapshot %s (different program or VM build?)\n", resume);
            return 2;
        }
        free(snap);
    }
    if (presume) {                      /* after the hooks: a saved screen is re-created */
        FILE *f = fopen(presume, "rb");
        int prc;
        if (!f) { fprintf(stderr, "tilvm: cannot read %s\n", presume); return 2; }
        prc = tilvm_load_portable(&vm, get_file, f);
        if (prc == 0 && fgetc(f) != EOF) prc = TILVM_P_CORRUPT;     /* bytes after the snapshot */
        fclose(f);
        if (prc != 0) {
            fprintf(stderr, "tilvm: cannot resume %s: %s\n", presume, portable_error(prc));
            return 2;
        }
    }

#ifndef _WIN32
    {                                   /* Ctrl-C: see on_sigint (unless it is ignored) */
        struct sigaction sa, old;
        if (sigaction(SIGINT, NULL, &old) == 0 && old.sa_handler != SIG_IGN) {
            memset(&sa, 0, sizeof sa);
            sa.sa_handler = on_sigint;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0;            /* no SA_RESTART: a waiting select() returns */
            sigaction(SIGINT, &sa, NULL);
        }
    }
#endif

    if (psave_out) {                    /* N more steps (0: none), then a portable snapshot */
        rc = psave_after ? run_slices((uint32_t)psave_after) : TILVM_YIELD;
        fflush(stdout);
        if (rc == TILVM_YIELD && !interrupted) {
            FILE *f = fopen(psave_out, "wb");
            uint32_t n = f ? tilvm_save_portable(&vm, put_file, f) : 0;
            if (f && fclose(f) != 0) n = 0;
            if (!n) {
#if defined(TILVM_LEAN) || defined(TILVM_NO_PORTABLE)
                fprintf(stderr, "tilvm: %s\n", portable_error(TILVM_P_UNAVAILABLE));
#else
                fprintf(stderr, "tilvm: cannot write %s\n", psave_out);
#endif
                return 2;
            }
            fprintf(stderr, "tilvm: paused after %lu steps; saved %s (%lu bytes)\n",
                    psave_after, psave_out, (unsigned long)n);
            return save_screen(screen_out, 3);
        }
        if (!interrupted)
            fprintf(stderr, "tilvm: the program ended before the pause point (nothing saved)\n");
    } else if (snap_out) {
        rc = run_slices((uint32_t)snap_after);
        fflush(stdout);
        if (rc == TILVM_YIELD && !interrupted) {
            uint32_t n = tilvm_snapshot_size(&vm);
            unsigned char *buf = (unsigned char *)malloc(n);
            FILE *f = fopen(snap_out, "wb");
            if (!buf || !f || tilvm_snapshot(&vm, buf, n) != n || fwrite(buf, 1, n, f) != n) {
                fprintf(stderr, "tilvm: cannot write snapshot\n");
                return 2;
            }
            fclose(f);
            free(buf);
            fprintf(stderr, "[tilvm] paused after %lu steps; snapshot saved to %s (%lu bytes)\n",
                    snap_after, snap_out, (unsigned long)n);
            return save_screen(screen_out, 3);
        }
    } else {
        rc = run_slices(0);
    }
    rc = interrupted ? 130 : rc == TILVM_EXIT ? vm.st.exit_code : rc == TILVM_DONE ? 0 : 1;
    fflush(stdout);
    if (stats) {
        fprintf(stderr, "[tilvm] steps=%lu gc=%lu heap=%lu/%ld bytes\n",
                (unsigned long)vm.st.steps, (unsigned long)vm.st.gc_count,
                (unsigned long)tilvm_heap_used(&vm), (long)TILVM_HEAP_SIZE);
        if (vm.idle_count || vm.st.energy || (vm.flags & 0x8C))  /* the program saves energy */
            fprintf(stderr, "[tilvm] energy_mode=%u idle=%lu\n", (unsigned)vm.st.energy,
                    (unsigned long)vm.idle_count);
#ifdef TILVM_STATS
        {   /* high-water marks (til fit --run reads this line) */
            tilvm_stats s;
            tilvm_get_stats(&vm, &s);
            fprintf(stderr, "[tilvm] peak stack=%lu/%lu frames=%lu/%lu try=%lu/%lu heap=%lu/%ld live=%lu "
                    "handlers=%lu/%lu\n",
                    (unsigned long)s.peak_stack, (unsigned long)TILVM_STACK_SIZE,
                    (unsigned long)s.peak_frames, (unsigned long)TILVM_MAX_FRAMES,
                    (unsigned long)s.peak_try, (unsigned long)TILVM_MAX_TRY,
                    (unsigned long)s.peak_heap, (long)TILVM_HEAP_SIZE, (unsigned long)s.peak_live,
                    (unsigned long)s.peak_handlers, (unsigned long)TILVM_MAX_HANDLERS);
        }
#endif
    }
    rc = save_screen(screen_out, rc);
    free(code);
    free(fb);
    free(keys);
#ifndef _WIN32
    if (interrupted) {                  /* Ctrl-C: end as an uncaught SIGINT does, so */
        fflush(stdout);                 /* that a shell also stops the script it runs */
        term_signal(SIGINT);
    }
#endif
    return rc;
}
