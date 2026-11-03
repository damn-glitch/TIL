/*
 * TIL playground - WebAssembly host for the portable TIL VM
 * ==========================================================
 *
 * build.sh compiles this file together with the UNMODIFIED vm/tilvm.c
 * (TILVM_NO_STDIO, 64-bit ints, double floats) and the freestanding shim in
 * shim/ into tilvm.wasm. playground/tilvm.js loads it in the browser (and in
 * Node for the tests).
 *
 * What it provides
 *   - WH_MAX_VMS independent VM instances in a static array (no malloc): the
 *     playground runs the self-hosted compiler (tilc.tbc) in one and the
 *     compiled program in another;
 *   - per-instance program arguments (args()) and bytecode buffers;
 *   - host hooks that forward output, files (a virtual file system kept in
 *     JavaScript), keys, input lines and the clock to JavaScript imports;
 *   - one screen (back buffer + front buffer, 0xRRGGBB pixels stored as RGBA
 *     bytes, in memory grown on demand) that the page copies to a <canvas>.
 *
 * Never blocking the browser
 *   JavaScript calls wh_run(id, budget) for a few milliseconds per animation
 *   frame; each call runs at most `budget` steps (loop iterations + calls) and
 *   returns a status. Three natives would block a desktop VM; here they pause:
 *
 *   wait_key(), input(), sleep_ms()  When the program would have to wait (no
 *       key / line queued yet; the busy-wait of sleep_ms), the hook moves the
 *       current frame's pc back to the start of the native-call instruction
 *       (the VM saved the pc after the instruction's operands in the frame
 *       before calling the native) and calls the import `suspend`, which
 *       throws a JavaScript exception. The exception unwinds the WebAssembly
 *       frames back to tilvm.js, which restores __stack_pointer and reports
 *       WH_WAIT_KEY / WH_WAIT_LINE / WH_SLEEP. The native had not changed the
 *       VM state yet (no allocation, stack untouched), so the next wh_run
 *       simply executes the instruction again: now the key or line is there,
 *       or the sleep is over (its clock reads return the original deadline).
 *       input("prompt") prints its prompt before reading: the repeated print
 *       is swallowed (`swallow`) until the hook is reached again.
 *
 *   Pausing checks the bytecode at the saved pc (NATIVE or NATIVE_POP with
 *   the expected native id), so a VM build that does not save the pc before
 *   natives is detected: wait_key() and input() then stop the program with an
 *   explicit error, and sleep_ms() busy-waits as on a desktop.
 *
 * Exports (all ids are instance numbers 0..WH_MAX_VMS-1) - see tilvm.js.
 */
#include "tilvm.h"
#include <string.h>

#ifndef WH_MAX_VMS
#define WH_MAX_VMS 2
#endif
#ifndef WH_CODE_MAX
#define WH_CODE_MAX (2L * 1024L * 1024L)    /* bytecode bytes per instance */
#endif
#ifndef WH_SCREEN_MAX
#define WH_SCREEN_MAX 4096                  /* width and height limit (the VM core allows 1..4096) */
#endif
#define WH_ARGS_MAX 8192                    /* bytes of program arguments */
#define WH_ARGC_MAX 64
#define WH_ERR_MAX 4096                     /* captured error report */

/* status codes of wh_run (0-3 are the TILVM_* codes) */
#define WH_DONE 0
#define WH_ERROR 1
#define WH_YIELD 2
#define WH_EXIT 3
#define WH_WAIT_KEY 4                       /* wait_key(): call wh_run again when a key is queued */
#define WH_WAIT_LINE 5                      /* input(): call wh_run again when a line is queued */
#define WH_SLEEP 6                          /* sleep_ms(): call wh_run again after wh_sleep_ms() ms */
#define WH_HOST_ERROR 7                     /* the playground host stopped the program (message on stderr) */

/* Bytecode format constants (src/til.py: Op, NATIVES; append-only). Both
 * native-call instructions are 3 bytes long: opcode, native id, argc. */
#define WH_OP_NATIVE 0x55
#define WH_OP_NATIVE_POP 0x86               /* superinstruction NATIVE + POP (design section 13) */
#define WH_N_INPUT 18
#define WH_N_SLEEP_MS 27
#define WH_N_WAIT_KEY 51

#define WASM_EXPORT(name) __attribute__((export_name(name)))
#define JS_IMPORT(name) __attribute__((import_module("env"), import_name(name)))

/* ─── JavaScript imports (tilvm.js) ─────────────────────────────────────── */
/* stream 1 = program output, 2 = error reports */
JS_IMPORT("write") void js_write(int vm, int stream, const char *data, unsigned long len);
/* milliseconds (performance.now()) */
JS_IMPORT("now") double js_now(void);
/* copy the next input line (at most cap-1 bytes, no newline) into buf:
 * length, -1 at end of input, -2 when no line is queued yet */
JS_IMPORT("read_line") int js_read_line(int vm, char *buf, int cap);
/* next queued key code, 0 when none */
JS_IMPORT("key") int js_key(int vm);
/* virtual file system: size or -1 (copies at most cap bytes when buf != 0); 0 or -1 */
JS_IMPORT("file_read") long js_file_read(int vm, const char *path, int path_len, char *buf, unsigned long cap);
JS_IMPORT("file_write") int js_file_write(int vm, const char *path, int path_len, const char *data,
                                          unsigned long len);
/* screen events: 0 = opened (a = width, b = height), 1 = frame shown (a = frame number) */
JS_IMPORT("screen") void js_screen(int vm, int event, int a, int b);
/* throws: unwinds out of wh_run (see the header comment) */
JS_IMPORT("suspend") __attribute__((noreturn)) void js_suspend(int vm);

/* ─── instances ─────────────────────────────────────────────────────────── */
/* Where the VM is: two clock reads at the same spot happen only inside one
 * native call (the frame's saved pc alone may be stale between natives). */
typedef struct wh_spot {
    uint32_t pc, steps;
    uint16_t sp, frames, tries, deadlines;
} wh_spot;

typedef struct wh_vm {
    tilvm vm;
    int id;
    int used;
    int loaded;
    int status;             /* status of the last wh_run */
    int host_failed;        /* WH_HOST_ERROR: never resume */
    int swallow;            /* drop program output until input() reads again */
    double t0;              /* js_now() when the program was loaded */
    uint32_t last_ms;       /* the clock never goes back */
    /* sleep_ms() (see h_clock): 0 = none, 1 = first clock read inside a sleep_ms
     * native, 2 = that sleep was paused and its native runs again */
    int sleep_state;
    wh_spot sleep_spot;     /* where the native reads the clock */
    uint32_t sleep_until;   /* its deadline */
    uint32_t sleep_len;     /* its argument (ms) */
    uint32_t sleep_reads;   /* clock reads since it runs again */
    /* program arguments */
    int argc;
    const char *argv[WH_ARGC_MAX + 1];
    char args[WH_ARGS_MAX];
    /* error report (copy of everything written to the error stream) */
    uint32_t err_len;
    char err[WH_ERR_MAX];
    /* the program (must stay valid while it runs) */
    uint32_t code_size;
    uint8_t code[WH_CODE_MAX];
} wh_vm;

static wh_vm vms[WH_MAX_VMS];

/* the screen: one display shared by all instances (the last screen_init owns
 * it). The two pixel buffers live above the static data (__heap_base), in
 * WebAssembly memory grown on demand: a program without graphics costs
 * nothing, a 4096x4096 screen needs 128 MB. */
static struct {
    int open;               /* screen_init() was called */
    int owner;              /* instance id */
    int w, h;
    uint32_t frame;         /* incremented by screen_show() */
    uint32_t *back;         /* drawing target */
    uint32_t *front;        /* last shown frame */
} screen;

extern unsigned char __heap_base;           /* end of the static data (wasm-ld) */

static wh_vm *get_vm(int id) {
    if (id < 0 || id >= WH_MAX_VMS || !vms[id].used) return 0;
    return &vms[id];
}

/* ─── output ────────────────────────────────────────────────────────────── */
static void h_write(void *ctx, const char *data, size_t len) {
    wh_vm *w = (wh_vm *)ctx;
    if (w->swallow || !len) return;
    js_write(w->id, 1, data, (unsigned long)len);
}

static void h_write_err(void *ctx, const char *data, size_t len) {
    wh_vm *w = (wh_vm *)ctx;
    size_t room = WH_ERR_MAX - 1 - w->err_len;
    if (!len) return;
    memcpy(w->err + w->err_len, data, len < room ? len : room);
    w->err_len += (uint32_t)(len < room ? len : room);
    w->err[w->err_len] = 0;
    js_write(w->id, 2, data, (unsigned long)len);
}

static void err_str(wh_vm *w, const char *s) { h_write_err(w, s, strlen(s)); }

/* Stop the program for good with an explicit message and leave the VM. */
static __attribute__((noreturn)) void host_fail(wh_vm *w, const char *a, const char *b) {
    err_str(w, "playground error: ");
    err_str(w, a);
    if (b) err_str(w, b);
    err_str(w, "\n");
    w->host_failed = 1;
    w->status = WH_HOST_ERROR;
    w->vm.st.failed = 1;                    /* the VM refuses to run on */
    js_suspend(w->id);
}

/* ─── pausing inside a native ───────────────────────────────────────────── */
static tv_frame *top_frame(wh_vm *w) {
    return w->vm.st.n_frames ? &w->vm.st.frames[w->vm.st.n_frames - 1] : 0;
}

/* Is the VM executing `NATIVE native_id argc`? (the frame's saved pc points
 * right after the instruction while the native runs) */
static int in_native(wh_vm *w, int native_id) {
    tv_frame *fr = top_frame(w);
    uint32_t pc;
    if (!fr) return 0;
    pc = fr->pc;
    if (pc < w->vm.code_off + 3 || pc > w->vm.code_off + w->vm.code_size) return 0;
    return (w->vm.bc[pc - 3] == WH_OP_NATIVE || w->vm.bc[pc - 3] == WH_OP_NATIVE_POP) &&
           w->vm.bc[pc - 2] == native_id;
}

/* Leave the VM so that the native instruction runs again on the next wh_run. */
static __attribute__((noreturn)) void pause_native(wh_vm *w, int native_id, int status, const char *who) {
    if (!in_native(w, native_id))
        host_fail(w, who, "() cannot pause in this VM build (the VM must save the pc before calling "
                          "a native; see website/playground/README.md)");
    top_frame(w)->pc -= 3;
    w->status = status;
    js_suspend(w->id);
}

/* ─── input: lines and keys ─────────────────────────────────────────────── */
static int h_read_line(void *ctx, char *buf, size_t cap) {
    wh_vm *w = (wh_vm *)ctx;
    int n = js_read_line(w->id, buf, (int)cap);
    if (n == -2) {
        w->swallow = 1;                     /* input(prompt) prints the prompt again */
        pause_native(w, WH_N_INPUT, WH_WAIT_LINE, "input");
    }
    w->swallow = 0;
    if (n >= (int)cap) n = (int)cap - 1;
    return n;
}

static int h_key_pressed(void *ctx) {
    return js_key(((wh_vm *)ctx)->id);
}

static int h_wait_key(void *ctx) {
    wh_vm *w = (wh_vm *)ctx;
    int k = js_key(w->id);
    if (k > 0) return k;
    pause_native(w, WH_N_WAIT_KEY, WH_WAIT_KEY, "wait_key");
}

/* ─── clock and sleep_ms ────────────────────────────────────────────────── */
/* milliseconds since the program was loaded (wall clock) */
static uint32_t real_ms(wh_vm *w) {
    double t = js_now() - w->t0;
    return t > 0 ? (uint32_t)(uint64_t)t : 0;
}

/* the program's clock: the wall clock, but never going back */
static uint32_t wall_ms(wh_vm *w) {
    uint32_t ms = real_ms(w);
    if ((int32_t)(ms - w->last_ms) < 0) ms = w->last_ms;
    w->last_ms = ms;
    return ms;
}

static void get_spot(wh_vm *w, wh_spot *p) {
    tv_frame *fr = top_frame(w);
    p->pc = fr ? fr->pc : 0;
    p->steps = w->vm.st.steps;
    p->sp = w->vm.st.sp;
    p->frames = w->vm.st.n_frames;
    p->tries = w->vm.st.n_try;
    p->deadlines = w->vm.st.n_deadlines;
}

static int same_spot(const wh_spot *a, const wh_spot *b) {
    return a->pc == b->pc && a->steps == b->steps && a->sp == b->sp && a->frames == b->frames &&
           a->tries == b->tries && a->deadlines == b->deadlines;
}

/* the int argument of the sleep_ms native being executed, or -1 */
static long sleep_arg(wh_vm *w) {
    tv_frame *fr = top_frame(w);
    uint8_t argc;
    const tv_val *a;
    if (!in_native(w, WH_N_SLEEP_MS)) return -1;
    argc = w->vm.bc[fr->pc - 1];
    if (!argc || w->vm.st.sp < argc) return -1;
    a = &w->vm.st.stack[w->vm.st.sp - argc];
    if (a->t != TV_INT || a->u.i <= 0) return -1;
    return (long)(uint32_t)a->u.i;             /* the VM computes until = now + (uint32_t)ms */
}

static uint32_t sleep_end(wh_vm *w) {
    if ((int32_t)(w->sleep_until - w->last_ms) > 0) w->last_ms = w->sleep_until;   /* never back */
    return w->sleep_until;
}

/* Leave the VM during the sleep; the native runs again once it is over. */
static __attribute__((noreturn)) void pause_sleep(wh_vm *w) {
    w->sleep_state = 2;
    w->sleep_reads = 0;
    pause_native(w, WH_N_SLEEP_MS, WH_SLEEP, "sleep_ms");
}

/* sleep_ms(ms) runs `until = now() + ms; while (now() < until) {}`. The first
 * clock read at a spot inside a sleep_ms native records the deadline; a second
 * read at the very same spot is the busy-wait: the program pauses (WH_SLEEP)
 * and the page waits. When the instruction runs again, its first read returns
 * the original start (so `until` is the same deadline) and later reads return
 * at least the deadline, so the wait ends at once. Reads elsewhere (timeouts)
 * are never taken for a sleep: they happen at another spot. */
static uint32_t h_clock(void *ctx) {
    wh_vm *w = (wh_vm *)ctx;
    wh_spot here;
    uint32_t now;
    long ms;
    get_spot(w, &here);
    if (w->sleep_state && same_spot(&here, &w->sleep_spot)) {
        if (w->sleep_state == 2) {
            if (!w->sleep_reads++) return w->sleep_until - w->sleep_len;
            return sleep_end(w);
        }
        if ((int32_t)(w->sleep_until - real_ms(w)) > 0) pause_sleep(w);
        w->sleep_state = 0;
        return wall_ms(w);
    }
    w->sleep_state = 0;
    now = wall_ms(w);
    ms = sleep_arg(w);
    if (ms > 0) {
        w->sleep_state = 1;
        w->sleep_spot = here;
        w->sleep_until = now + (uint32_t)ms;
        w->sleep_len = (uint32_t)ms;
    }
    return now;
}

/* Idling inside sleep_ms() (a VM that sleeps through the idle hook) is the
 * same pause. Energy saving elsewhere is a no-op: a browser tab gains
 * nothing from idling inside a time slice. */
static void h_idle(void *ctx, unsigned long ms) {
    wh_vm *w = (wh_vm *)ctx;
    wh_spot here;
    if (!ms) return;
    get_spot(w, &here);
    if (w->sleep_state && same_spot(&here, &w->sleep_spot)) {
        if (w->sleep_state == 1 && (int32_t)(w->sleep_until - real_ms(w)) > 0) pause_sleep(w);
        return;                                 /* paused before: the wait is over */
    }
    if (sleep_arg(w) != (long)ms) return;      /* only an idle(ms) that is the sleep itself */
    w->sleep_spot = here;
    w->sleep_until = wall_ms(w) + (uint32_t)ms;
    w->sleep_len = (uint32_t)ms;
    pause_sleep(w);
}

/* ─── files: the virtual file system lives in JavaScript ───────────────── */
static long h_read_file(void *ctx, const char *path, char *buf, unsigned long cap) {
    wh_vm *w = (wh_vm *)ctx;
    return js_file_read(w->id, path, (int)strlen(path), buf, buf ? cap : 0);
}

static int h_write_file(void *ctx, const char *path, const char *data, unsigned long len) {
    wh_vm *w = (wh_vm *)ctx;
    return js_file_write(w->id, path, (int)strlen(path), data, len);
}

/* ─── screen ────────────────────────────────────────────────────────────── */
static uint32_t to_rgba(unsigned long rgb) {
    /* 0xRRGGBB -> bytes R, G, B, 255 in memory (little-endian uint32) */
    return 0xFF000000u | ((uint32_t)(rgb & 0xFF) << 16) | ((uint32_t)rgb & 0xFF00u) |
           ((uint32_t)(rgb >> 16) & 0xFFu);
}

static unsigned long from_rgba(uint32_t p) {
    return (unsigned long)(((p & 0xFFu) << 16) | (p & 0xFF00u) | ((p >> 16) & 0xFFu));
}

#define WH_STR_(x) #x
#define WH_STR(x) WH_STR_(x)

/* Make room for two width x height buffers above the static data. */
static int screen_memory(unsigned long pixels) {
    unsigned long base = ((unsigned long)&__heap_base + 15ul) & ~15ul;
    unsigned long need = base + pixels * 8ul;                 /* two buffers of 4-byte pixels */
    unsigned long have = (unsigned long)__builtin_wasm_memory_size(0) * 65536ul;
    if (need > have) {
        unsigned long pages = (need - have + 65535ul) / 65536ul;
        if (__builtin_wasm_memory_grow(0, pages) == (unsigned long)-1) return 0;
    }
    screen.back = (uint32_t *)base;
    screen.front = (uint32_t *)base + pixels;
    return 1;
}

static void h_gfx_init(void *ctx, int width, int height) {
    wh_vm *w = (wh_vm *)ctx;
    long i, n;
    if (width < 1 || height < 1 || width > WH_SCREEN_MAX || height > WH_SCREEN_MAX)
        host_fail(w, "screen_init(): the playground screen is limited to ",
                  WH_STR(WH_SCREEN_MAX) "x" WH_STR(WH_SCREEN_MAX) " pixels");
    if (!screen_memory((unsigned long)width * (unsigned long)height))
        host_fail(w, "screen_init(): ", "the browser cannot provide the memory for a screen this large");
    screen.open = 1;
    screen.owner = w->id;
    screen.w = width;
    screen.h = height;
    n = (long)width * height;
    for (i = 0; i < n; i++) screen.back[i] = screen.front[i] = 0xFF000000u;
    screen.frame++;
    js_screen(w->id, 0, width, height);
}

static void h_gfx_pixel(void *ctx, int x, int y, unsigned long rgb) {
    (void)ctx;
    if (!screen.open || x < 0 || y < 0 || x >= screen.w || y >= screen.h) return;
    screen.back[(long)y * screen.w + x] = to_rgba(rgb);
}

static unsigned long h_gfx_get(void *ctx, int x, int y) {
    (void)ctx;
    if (!screen.open || x < 0 || y < 0 || x >= screen.w || y >= screen.h) return 0;
    return from_rgba(screen.back[(long)y * screen.w + x]);
}

static void h_gfx_show(void *ctx) {
    wh_vm *w = (wh_vm *)ctx;
    if (!screen.open) return;
    memcpy(screen.front, screen.back, (size_t)screen.w * (size_t)screen.h * sizeof(uint32_t));
    screen.frame++;
    js_screen(w->id, 1, (int)screen.frame, 0);
}

/* ─── exported API ──────────────────────────────────────────────────────── */
WASM_EXPORT("wh_create") int wh_create(void) {
    int i;
    for (i = 0; i < WH_MAX_VMS; i++) {
        if (!vms[i].used) {
            vms[i].used = 1;
            vms[i].id = i;
            vms[i].loaded = 0;
            vms[i].argc = 0;
            vms[i].argv[0] = 0;
            vms[i].code_size = 0;
            return i;
        }
    }
    return -1;
}

WASM_EXPORT("wh_destroy") void wh_destroy(int id) {
    wh_vm *w = get_vm(id);
    if (!w) return;
    w->used = 0;
    w->loaded = 0;
    if (screen.open && screen.owner == id) screen.open = 0;
}

WASM_EXPORT("wh_code_ptr") uint8_t *wh_code_ptr(int id) {
    wh_vm *w = get_vm(id);
    return w ? w->code : 0;
}

WASM_EXPORT("wh_args_ptr") char *wh_args_ptr(int id) {
    wh_vm *w = get_vm(id);
    return w ? w->args : 0;
}

/* `nbytes` bytes of NUL-terminated strings were written at wh_args_ptr(id).
 * Returns argc, or -1 when there are too many. */
WASM_EXPORT("wh_set_args") int wh_set_args(int id, int nbytes) {
    wh_vm *w = get_vm(id);
    int pos = 0, argc = 0;
    if (!w || nbytes < 0 || nbytes > WH_ARGS_MAX) return -1;
    while (pos < nbytes) {
        if (argc == WH_ARGC_MAX) return -1;
        w->argv[argc++] = w->args + pos;
        while (pos < nbytes && w->args[pos]) pos++;
        if (pos == nbytes) return -1;       /* not terminated */
        pos++;
    }
    w->argv[argc] = 0;
    w->argc = argc;
    w->vm.host.argc = argc;
    w->vm.host.argv = w->argv;
    return argc;
}

/* The program (`size` bytes at wh_code_ptr(id)) is loaded and reset to its
 * first instruction. Returns 0 or a tilvm_load error code (-100: too large). */
WASM_EXPORT("wh_load") int wh_load(int id, uint32_t size) {
    wh_vm *w = get_vm(id);
    int rc;
    if (!w) return -101;
    w->loaded = 0;
    if (screen.open && screen.owner == id) screen.open = 0;   /* a new program starts without a screen */
    if (size > WH_CODE_MAX) return -100;
    w->code_size = size;
    rc = tilvm_load(&w->vm, w->code, size);
    if (rc != 0) return rc;
    w->vm.write = h_write;
    w->vm.write_err = h_write_err;
    w->vm.read_line = h_read_line;
    w->vm.clock_ms = h_clock;
    w->vm.ctx = w;
    w->vm.host.read_file = h_read_file;
    w->vm.host.write_file = h_write_file;
    w->vm.host.gfx_init = h_gfx_init;
    w->vm.host.gfx_pixel = h_gfx_pixel;
    w->vm.host.gfx_get = h_gfx_get;
    w->vm.host.gfx_show = h_gfx_show;
    w->vm.host.key_pressed = h_key_pressed;
    w->vm.host.wait_key = h_wait_key;
    w->vm.host.idle = h_idle;
    w->vm.host.argc = w->argc;
    w->vm.host.argv = w->argv;
    w->loaded = 1;
    w->status = WH_YIELD;
    w->host_failed = 0;
    w->swallow = 0;
    w->sleep_state = 0;
    w->t0 = js_now();
    w->last_ms = 0;
    w->err_len = 0;
    w->err[0] = 0;
    return 0;
}

/* Run at most `budget` steps (0 = no limit). Returns a WH_* status. */
WASM_EXPORT("wh_run") int wh_run(int id, uint32_t budget) {
    wh_vm *w = get_vm(id);
    int rc;
    if (!w || !w->loaded) return WH_HOST_ERROR;
    if (w->host_failed) return w->status = WH_HOST_ERROR;
    w->status = WH_YIELD;
    if (w->sleep_state == 1) w->sleep_state = 0;    /* a paused sleep (2) is resumed now */
    rc = tilvm_run_steps(&w->vm, budget);
    return w->status = rc;
}

WASM_EXPORT("wh_status") int wh_status(int id) {
    wh_vm *w = get_vm(id);
    return w ? w->status : WH_HOST_ERROR;
}

WASM_EXPORT("wh_exit_code") int wh_exit_code(int id) {
    wh_vm *w = get_vm(id);
    if (!w) return 1;
    if (w->status == WH_EXIT) return w->vm.st.exit_code;
    return w->status == WH_DONE ? 0 : 1;
}

/* WH_SLEEP: milliseconds left until the program wakes up (0 = now) */
WASM_EXPORT("wh_sleep_ms") int wh_sleep_ms(int id) {
    wh_vm *w = get_vm(id);
    int32_t left;
    if (!w) return 0;
    left = (int32_t)(w->sleep_until - real_ms(w));
    return left > 0 ? (int)left : 0;
}

WASM_EXPORT("wh_steps") uint32_t wh_steps(int id) {
    wh_vm *w = get_vm(id);
    return w ? w->vm.st.steps : 0;
}

WASM_EXPORT("wh_gc_count") uint32_t wh_gc_count(int id) {
    wh_vm *w = get_vm(id);
    return w ? w->vm.st.gc_count : 0;
}

WASM_EXPORT("wh_heap_used") uint32_t wh_heap_used(int id) {
    wh_vm *w = get_vm(id);
    return w && w->loaded ? tilvm_heap_used(&w->vm) : 0;
}

WASM_EXPORT("wh_error_ptr") const char *wh_error_ptr(int id) {
    wh_vm *w = get_vm(id);
    return w ? w->err : 0;
}

WASM_EXPORT("wh_error_len") uint32_t wh_error_len(int id) {
    wh_vm *w = get_vm(id);
    return w ? w->err_len : 0;
}

WASM_EXPORT("wh_screen_ptr") const uint32_t *wh_screen_ptr(void) { return screen.open ? screen.front : 0; }
WASM_EXPORT("wh_screen_w") int wh_screen_w(void) { return screen.open ? screen.w : 0; }
WASM_EXPORT("wh_screen_h") int wh_screen_h(void) { return screen.open ? screen.h : 0; }
WASM_EXPORT("wh_screen_frame") uint32_t wh_screen_frame(void) { return screen.frame; }
WASM_EXPORT("wh_screen_owner") int wh_screen_owner(void) { return screen.open ? screen.owner : -1; }

WASM_EXPORT("wh_version") const char *wh_version(void) { return TILVM_VERSION; }

/* build configuration, for the page and the tests */
WASM_EXPORT("wh_info") long wh_info(int key) {
    switch (key) {
    case 0: return WH_MAX_VMS;
    case 1: return (long)TILVM_HEAP_SIZE;
    case 2: return TILVM_STACK_SIZE;
    case 3: return TILVM_MAX_FRAMES;
    case 4: return (long)WH_CODE_MAX;
    case 5: return WH_SCREEN_MAX;
    case 6: return WH_SCREEN_MAX;
    case 7: return TILVM_INT_BITS;
    case 8: return TILVM_FLOAT;
    case 9: return TILVM_BYTECODE_VERSION;
    case 10: return TILVM_BYTECODE_MIN_VERSION;
    case 11: return TILVM_MAX_CONSTS;
    case 12: return WH_ARGS_MAX;
    case 13: return WH_ARGC_MAX;
    default: return -1;
    }
}
