/*
 * TIL VM - portable virtual machine for TIL bytecode (.tbc)
 * =========================================================
 * Author: Alisher Beisembekov (TIL), VM design for TIL 3.0
 *
 * One C99 file with:
 *   - no malloc / free: all state lives in one `tilvm` struct the host allocates
 *     (static, global or on the stack) - the program bytecode can stay in ROM/flash
 *   - a fixed arena heap with a mark-compact garbage collector (no fragmentation)
 *   - configurable integer width (16/32/64 bit) and float (none / float / double)
 *   - pluggable output, input and clock (calculator screen, UART, stdout ...)
 *   - snapshot / resume of a running program (hibernate on power-off)
 *   - step budgets (cooperative multitasking, sandboxing) and real timeouts
 *
 * Targets: desktop (gcc/clang/MSVC), calculator-class CPUs (ARM: NumWorks,
 * TI-Nspire; eZ80: TI-84 Plus CE toolchain; SH4: Casio via gint) and
 * microcontrollers (AVR, ARM Cortex-M, ESP32, RP2040). Verified in emulators:
 * ARM Cortex-M3, 8-bit AVR and the Z80 (16-bit lean build, ports/z80/RESULTS.md).
 *
 * Configuration (define before including, or with -D):
 *   TILVM_INT_BITS     16 | 32 | 64           (default 64)
 *   TILVM_FLOAT        0 (no floats) | 1 (float) | 2 (double)   (default 2)
 *   TILVM_HEAP_SIZE    arena bytes             (default 256 KB, small: 2-16 KB)
 *   TILVM_STACK_SIZE   value-stack slots       (default 8192)
 *   TILVM_MAX_FRAMES   call depth              (default 1000)
 *   TILVM_MAX_TRY      nested retry/timeout    (default 32)
 *   TILVM_MAX_HANDLERS active effect handlers + running clauses (default 32)
 *   TILVM_MAX_CONSTS   constants per program   (default 4096)
 *   TILVM_MAX_TYPES    struct types            (default 512)
 *   TILVM_MAX_FUNCS    functions decoded into a table at load (default TILVM_MAX_CONSTS;
 *                      TILVM_LEAN: 0 = no table, records are read from the bytecode).
 *                      A program with more functions is rejected by tilvm_load (-10).
 *   TILVM_FIELD_CACHE  entries of the struct field / method lookup cache (power of two;
 *                      default 64, TILVM_LEAN: 0 = no cache)
 *   TILVM_COMPUTED_GOTO 1 = dispatch with GCC/Clang "labels as values" (default with
 *                      __GNUC__, except TILVM_LEAN and -Os builds), 0 = portable switch
 *   TILVM_FAST_PATHS   1 = inline integer fast paths in the interpreter (default, except
 *                      TILVM_LEAN and -Os builds), 0 = smaller code, same results
 *   TILVM_NO_STDIO     no <stdio.h>: the host must set vm->write
 *   TILVM_NO_TIME      no <time.h>: timeouts count steps (1 ms = 1000 steps)
 *   TILVM_NO_MATH      no <math.h>: sqrt/sin/... use built-in approximations
 *   TILVM_ADDR16       16-bit address space (Z80): 16-bit bytecode offsets and object
 *                      lengths (programs and heap below 64 KB) - much smaller code
 *   TILVM_LEAN         smallest code for 8-bit CPUs: the whole language core, but
 *                      without format specs, snapshots, input/chr/ord/type_of/clock/
 *                      random/sleep, and str split/replace/trim/repeat/count/to_float,
 *                      array insert/remove/sort/reverse/slice/copy/join/sum/min/max,
 *                      map values/copy/clear (they report "not available" / "no
 *                      method"), array + array ("unsupported operand types"), and
 *                      the map-literal key check ({"a": 1, "a": 2} keeps both
 *                      entries, 2.0 stays a float key); `til fit` reports all of them.
 *                      Effect handlers and the screen/keys natives ARE included.
 *   TILVM_NO_EFFECTS   leave out effect handlers (handle/with/resume, ~1-3 KB of
 *                      code, TILVM_MAX_HANDLERS * 16 bytes of RAM): `handle` reports
 *                      "effect handlers are not available in this VM build";
 *                      `perform E()` still calls the top-level function E (TIL 3.0)
 *   TILVM_NO_GFX       leave out the screen/keys natives (screen_init ... wait_key
 *                      report "not available in this VM build")
 *   TILVM_GFX_MAX_W    largest screen the device can show (default 4096 x 4096, the
 *   TILVM_GFX_MAX_H    language limit): a host with a fixed display sets its size;
 *                      screen_init(w, h) beyond it is then a runtime error, and the
 *                      screen hooks never see x >= TILVM_GFX_MAX_W, y >= TILVM_GFX_MAX_H
 *   TILVM_ROM          storage qualifier for the 5x7 font table, e.g. __flash with
 *                      avr-gcc (gnu99) to keep it out of RAM (default: empty)
 *   TILVM_STATS        record high-water marks (tilvm_get_stats; til fit --run)
 *   TILVM_PACK         pack values and heap objects like 8-bit CPUs do (desktop
 *                      emulation of AVR / Z80 / eZ80 builds by til fit --run;
 *                      GCC/Clang hosts only)
 */
#ifndef TILVM_H
#define TILVM_H

#include <stdint.h>
#include <stddef.h>

#define TILVM_VERSION "3.0.0"
#define TILVM_BYTECODE_VERSION 2        /* newest .tbc version this VM loads */
#define TILVM_BYTECODE_MIN_VERSION 1    /* oldest (TIL 3.0) */

#ifndef TILVM_INT_BITS
#define TILVM_INT_BITS 64
#endif
#ifndef TILVM_FLOAT
#define TILVM_FLOAT 2
#endif
#ifndef TILVM_HEAP_SIZE
#define TILVM_HEAP_SIZE (256L * 1024L)
#endif
#ifndef TILVM_STACK_SIZE
#define TILVM_STACK_SIZE 8192
#endif
#ifndef TILVM_MAX_FRAMES
#define TILVM_MAX_FRAMES 1000
#endif
#ifndef TILVM_MAX_TRY
#define TILVM_MAX_TRY 32
#endif
#ifndef TILVM_MAX_HANDLERS
#define TILVM_MAX_HANDLERS 32
#endif
#ifndef TILVM_MAX_CONSTS
#define TILVM_MAX_CONSTS 4096
#endif
#ifndef TILVM_MAX_TYPES
#define TILVM_MAX_TYPES 512
#endif
#ifndef TILVM_MARK_STACK
#define TILVM_MARK_STACK 64
#endif
#ifndef TILVM_MAX_FUNCS
#ifdef TILVM_LEAN
#define TILVM_MAX_FUNCS 0
#else
#define TILVM_MAX_FUNCS TILVM_MAX_CONSTS
#endif
#endif
#ifndef TILVM_FIELD_CACHE
#ifdef TILVM_LEAN
#define TILVM_FIELD_CACHE 0
#else
#define TILVM_FIELD_CACHE 64
#endif
#endif
#ifndef TILVM_GFX_MAX_W
#define TILVM_GFX_MAX_W 4096            /* screen_init: width 1..TILVM_GFX_MAX_W */
#endif
#ifndef TILVM_GFX_MAX_H
#define TILVM_GFX_MAX_H 4096            /* screen_init: height 1..TILVM_GFX_MAX_H */
#endif

#if TILVM_INT_BITS == 16
typedef int16_t tv_int;
typedef uint16_t tv_uint;
#define TV_INT_MAX 32767L
#define TV_INT_MIN (-32767L - 1)
#elif TILVM_INT_BITS == 32
typedef int32_t tv_int;
typedef uint32_t tv_uint;
#define TV_INT_MAX 2147483647L
#define TV_INT_MIN (-2147483647L - 1)
#else
typedef int64_t tv_int;
typedef uint64_t tv_uint;
#define TV_INT_MAX INT64_MAX
#define TV_INT_MIN INT64_MIN
#endif

#if TILVM_FLOAT == 1
typedef float tv_float;
#elif TILVM_FLOAT == 2
typedef double tv_float;
#endif

/* Heap references are byte offsets into the arena (0 = none), so the
 * compacting collector can move objects and snapshots are position-free. */
#if TILVM_HEAP_SIZE <= 65535L
typedef uint16_t tv_ref;
typedef uint16_t tv_size;
#else
typedef uint32_t tv_ref;
typedef uint32_t tv_size;
#endif

/* Offsets into the bytecode (pc). TILVM_ADDR16 (CPUs with a 16-bit address
 * space, e.g. the Z80): programs and the heap are smaller than 64 KB, so offsets
 * and object lengths are 16-bit - much less code on 8-bit CPUs. tilvm_load
 * rejects larger programs (error -10). */
#ifdef TILVM_ADDR16
#if TILVM_HEAP_SIZE > 65535L
#error "TILVM_ADDR16 needs TILVM_HEAP_SIZE <= 65535"
#endif
typedef uint16_t tv_off;
#else
typedef uint32_t tv_off;
#endif

/* value tags */
enum {
    TV_NIL = 0, TV_BOOL, TV_INT, TV_FLOAT, TV_STRK, TV_STR, TV_ARRAY, TV_MAP,
    TV_STRUCT, TV_CLOSURE, TV_FUNC, TV_CELL, TV_UNDEF
};

#if defined(TILVM_PACK) && defined(__GNUC__)
#define TV_PACK __attribute__((packed))   /* 8-bit CPU layout: values without padding */
#else
#define TV_PACK
#endif

typedef struct TV_PACK tv_val {
    uint8_t t;
    union {
        tv_int i;
#if TILVM_FLOAT
        tv_float f;
#endif
        tv_ref r;       /* heap object */
        uint16_t k;     /* constant string index / function index */
    } u;
} tv_val;

typedef struct tv_frame {
    tv_off ret_pc;      /* where the caller continues */
    tv_off pc;          /* saved pc of this frame while it calls others */
    uint16_t func;
    uint16_t base;      /* first local slot on the value stack */
    tv_ref closure;     /* closure being executed (0 = plain function) */
    uint8_t extra;      /* callee slot below the arguments to drop on return */
} tv_frame;

typedef struct tv_try {
    tv_off handler;
    uint32_t deadline;  /* ms (0 = no deadline) */
    uint16_t n_frames;
    uint16_t sp;
    uint16_t n_handlers; /* effect handler entries when the block started */
} tv_try;

/* Effect handler stack entry (TIL 3.1, docs/TIL_EFFECTS.md).
 * kind 0 - an active `handle`: its n (clause, effect-name) value pairs lie on the
 *          value stack at stack[sp .. sp + 2n - 1]; a clause that finishes
 *          without resume() continues at abort_pc in frame n_frames - 1.
 * kind 1 - a running clause of handler #n (the owner): it masks the owner and
 *          every entry above it; n_frames / n_try are the values at the perform. */
typedef struct tv_handler {
    tv_off abort_pc;    /* kind 0: absolute pc after END_HANDLE; kind 1: 0 */
    uint16_t n_frames;  /* frames when installed (kind 1: at the perform) */
    uint16_t sp;        /* kind 0: stack index of the first pair; kind 1: 0 */
    uint16_t n_try;     /* recovery points when installed */
    uint16_t n;         /* kind 0: clause count; kind 1: owner index */
    uint8_t kind;
    uint8_t _pad[3];
} tv_handler;

/* Everything that changes while a program runs. Saved by snapshots. */
typedef struct tilvm_state {
    uint32_t magic;
    uint16_t sp, n_frames, n_try, n_deadlines;
    tv_size heap_top;
    uint8_t halted, exit_code, failed, _pad;
    uint16_t scr_w, scr_h;      /* screen size in pixels (0 = no screen_init yet) */
    uint32_t steps, gc_count, tick;
#if TILVM_INT_BITS == 64
    uint64_t rng;
#else
    uint32_t rng;
#endif
    tv_frame frames[TILVM_MAX_FRAMES];
    tv_try tries[TILVM_MAX_TRY];
    uint16_t n_handlers, _hpad;                 /* effect handlers (TIL 3.1) */
#ifndef TILVM_NO_EFFECTS
    tv_handler handlers[TILVM_MAX_HANDLERS];
#endif
    tv_val stack[TILVM_STACK_SIZE];
} tilvm_state;

/* A function record decoded once at load (TILVM_MAX_FUNCS > 0). */
typedef struct tv_func {
    tv_off code;        /* absolute offset of the first instruction in the bytecode */
    uint8_t nparams, nreq, nlocals, flags;
} tv_func;

/* Struct field / method lookup cache entry: (type, name constant) -> slot or function. */
typedef struct tv_cache {
    uint16_t type, name, value;
} tv_cache;

typedef void (*tilvm_write_fn)(void *ctx, const char *data, size_t len);
typedef int (*tilvm_read_fn)(void *ctx, char *buf, size_t cap);   /* returns length or -1 */
typedef uint32_t (*tilvm_clock_fn)(void *ctx);                    /* milliseconds */

/* Optional device services (TIL 3.1). A NULL hook means "not available on this
 * device": the native that needs it reports "<name>() is not available ...".
 * tilvm_load() resets the struct; with stdio it installs read_file/write_file.
 * All hooks receive vm->ctx. */
typedef struct tilvm_host {
    /* files: read_file returns the file size (or -1) and copies at most `cap`
     * bytes into buf (buf may be NULL to ask for the size); write_file returns
     * 0 on success, -1 on failure */
    long (*read_file)(void *ctx, const char *path, char *buf, unsigned long cap);
    int (*write_file)(void *ctx, const char *path, const char *data, unsigned long len);
    /* screen: pixels are 0xRRGGBB; the VM rasterizes lines, shapes and text and
     * clips them, so gfx_pixel/gfx_get only ever see 0 <= x < w, 0 <= y < h of the
     * last gfx_init(w, h), where w <= TILVM_GFX_MAX_W and h <= TILVM_GFX_MAX_H
     * (screen_init refuses larger screens: build the VM with the display's size).
     * gfx_pixel is required for a screen (gfx_get for get_pixel); gfx_init
     * (create or resize a w x h screen, all black) and gfx_show (present the
     * frame) are optional: without gfx_init the VM clears the screen pixel by
     * pixel. See docs/TIL_GRAPHICS.md. */
    void (*gfx_init)(void *ctx, int w, int h);
    void (*gfx_pixel)(void *ctx, int x, int y, unsigned long rgb);
    unsigned long (*gfx_get)(void *ctx, int x, int y);
    void (*gfx_show)(void *ctx);
    /* keys: 0 = none; UP=1 DOWN=2 LEFT=3 RIGHT=4, otherwise ASCII (ENTER=13, ESC=27)
     * or a Unicode code point. key_pressed must not block; wait_key blocks (when it
     * is NULL the VM polls key_pressed and calls idle(1) between polls). */
    int (*key_pressed)(void *ctx);
    int (*wait_key)(void *ctx);
    /* battery saving: sleep / halt the CPU for about `ms` milliseconds */
    void (*idle)(void *ctx, unsigned long ms);
    /* program arguments (args()) */
    int argc;
    const char *const *argv;
} tilvm_host;

#ifdef TILVM_STATS
/* High-water marks of one run (build with -DTILVM_STATS; used by `til fit --run`) */
typedef struct tilvm_stats {
    uint32_t peak_stack;    /* value-stack slots ever used (globals + locals + temporaries) */
    uint32_t peak_frames;   /* deepest call */
    uint32_t peak_try;      /* most nested retry/timeout/intent blocks */
    uint32_t peak_heap;     /* arena high-water mark (heap_top), bytes */
    uint32_t peak_live;     /* most heap in use right after a collection, bytes */
    uint32_t peak_handlers; /* most effect handler entries (open handle blocks + running clauses) */
} tilvm_stats;
#endif

typedef struct tilvm {
    /* ---- host bindings (set after tilvm_load; not part of snapshots) ---- */
    tilvm_write_fn write;       /* program output */
    tilvm_write_fn write_err;   /* error reports (default: same as write) */
    tilvm_read_fn read_line;    /* input() */
    tilvm_clock_fn clock_ms;    /* clock_ms(), timeout() */
    void *ctx;
    uint32_t max_steps;         /* 0 = unlimited (sandbox) */
    tilvm_host host;            /* files, screen, keys, idle, args (TIL 3.1) */

    /* ---- program (read-only, may live in ROM) ---- */
    const uint8_t *bc;
    uint32_t bc_size;
    tv_off code_off, code_size;
    tv_off funcs_off, methods_off, prelude_off, lines_off;
    uint16_t n_consts, n_funcs, n_types, n_methods, n_globals, entry;
    uint16_t n_lines;
    uint8_t n_prelude, flags;
    tv_off const_off[TILVM_MAX_CONSTS];
    tv_off type_off[TILVM_MAX_TYPES];
    uint32_t bc_hash;                 /* FNV-1a of the bytecode, for snapshots (0 in TILVM_LEAN) */
#if TILVM_MAX_FUNCS > 0
    tv_func funcs[TILVM_MAX_FUNCS];   /* decoded function records (TIL 3.1) */
#endif
#if TILVM_FIELD_CACHE > 0
    tv_cache field_cache[TILVM_FIELD_CACHE];    /* (struct type, field name) -> slot */
    tv_cache method_cache[TILVM_FIELD_CACHE];   /* (struct type, method name) -> function */
#endif

    /* ---- runtime ---- */
    const char *err_msg;        /* last runtime error (C string) */
    tv_val err_val;             /* ... or TIL value */
    char errbuf[128];
    uint16_t mark_sp;
    uint8_t mark_overflow;
    tv_ref mark_stack[TILVM_MARK_STACK];
    tilvm_state st;
    union {                     /* the arena (aligned for values) */
        uint8_t heap[TILVM_HEAP_SIZE];
        tv_val align_v_;
#if TILVM_FLOAT == 2 || TILVM_INT_BITS == 64
        uint64_t align_u_;
#endif
    } mem;
#ifdef TILVM_STATS
    tilvm_stats stats;          /* high-water marks (not part of snapshots) */
#endif
} tilvm;

/* status codes of tilvm_run_steps */
#define TILVM_DONE   0   /* program finished */
#define TILVM_ERROR  1   /* uncaught runtime error (already reported) */
#define TILVM_YIELD  2   /* step budget used up: call again to continue */
#define TILVM_EXIT   3   /* exit(n) was called: see st.exit_code */

/* Load a program. The bytecode buffer must stay valid while the VM runs.
 * Returns 0 on success, a negative error code otherwise. */
int tilvm_load(tilvm *vm, const uint8_t *bytecode, uint32_t size);

/* Run to completion. Returns the program's exit code (0 = success). */
int tilvm_run(tilvm *vm);

/* Run at most `budget` steps (loop iterations + calls); 0 = no limit.
 * Returns TILVM_DONE / TILVM_ERROR / TILVM_YIELD / TILVM_EXIT. */
int tilvm_run_steps(tilvm *vm, uint32_t budget);

/* Snapshots ("hibernate"): save a paused program, resume it later - even after
 * a power cycle - with the same VM build and the same bytecode. */
uint32_t tilvm_snapshot_size(const tilvm *vm);
uint32_t tilvm_snapshot(const tilvm *vm, uint8_t *out, uint32_t cap);   /* bytes written, 0 = error */
int tilvm_restore(tilvm *vm, const uint8_t *data, uint32_t size);      /* after tilvm_load */

/* Memory statistics */
uint32_t tilvm_heap_used(const tilvm *vm);
#ifdef TILVM_STATS
void tilvm_get_stats(const tilvm *vm, tilvm_stats *out);   /* high-water marks so far */
#endif

/* Default host bindings (stdout / stdin / clock) unless TILVM_NO_STDIO */
void tilvm_default_write(void *ctx, const char *data, size_t len);

#endif /* TILVM_H */
