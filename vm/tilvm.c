/*
 * TIL VM - portable virtual machine for TIL bytecode. See tilvm.h.
 *
 * Semantics follow the reference VM in src/til.py (TILVM class) exactly; the
 * test-suite runs every program on both and compares the output byte for byte.
 */
#include "tilvm.h"
#include <string.h>

#ifndef TILVM_NO_STDIO
#include <stdio.h>
#include <stdlib.h>
#endif
#if !defined(TILVM_NO_TIME) && !defined(TILVM_NO_STDIO)
#include <time.h>
#define TV_HAVE_CLOCK 1
#endif
#if TILVM_FLOAT && !defined(TILVM_NO_MATH)
#include <math.h>
#define TV_HAVE_MATH 1
#endif

/* Integer fast paths of the interpreter and of map lookups / int formatting:
 * pure optimizations (the general paths give the same results). Off for
 * TILVM_LEAN and for builds optimized for size (-Os), where they cost the most. */
#ifndef TILVM_FAST_PATHS
#if defined(TILVM_LEAN) || defined(__OPTIMIZE_SIZE__)
#define TILVM_FAST_PATHS 0
#else
#define TILVM_FAST_PATHS 1
#endif
#endif

/* ─── opcodes (must match class Op in src/til.py) ───────────────────────── */
enum {
    OP_NOP = 0x00, OP_PUSH_NIL = 0x01, OP_PUSH_TRUE = 0x02, OP_PUSH_FALSE = 0x03,
    OP_PUSH_I8 = 0x04, OP_PUSH_I16 = 0x05, OP_PUSH_CONST = 0x06, OP_POP = 0x07,
    OP_DUP = 0x08, OP_DUP2 = 0x09, OP_SWAP = 0x0A,
    OP_LOAD_LOCAL = 0x10, OP_STORE_LOCAL = 0x11, OP_LOAD_GLOBAL = 0x12, OP_STORE_GLOBAL = 0x13,
    OP_LOAD_UPVAL = 0x14, OP_STORE_UPVAL = 0x15, OP_LOAD_CELL = 0x16, OP_STORE_CELL = 0x17,
    OP_MAKE_CELL = 0x18, OP_LOAD_UPVAL_RAW = 0x19, OP_ARG_DEFAULT = 0x1A,
    OP_ADD = 0x20, OP_SUB = 0x21, OP_MUL = 0x22, OP_DIV = 0x23, OP_MOD = 0x24, OP_POW = 0x25,
    OP_NEG = 0x26, OP_NOT = 0x27, OP_BAND = 0x28, OP_BOR = 0x29, OP_BXOR = 0x2A, OP_BNOT = 0x2B,
    OP_SHL = 0x2C, OP_SHR = 0x2D,
    OP_EQ = 0x30, OP_NE = 0x31, OP_LT = 0x32, OP_LE = 0x33, OP_GT = 0x34, OP_GE = 0x35,
    OP_JUMP = 0x40, OP_JUMP_IF_FALSE = 0x41, OP_JUMP_IF_TRUE = 0x42,
    OP_CALL = 0x50, OP_CALL_VALUE = 0x51, OP_INVOKE = 0x52, OP_RETURN = 0x53,
    OP_RETURN_NIL = 0x54, OP_NATIVE = 0x55,
    OP_NEW_ARRAY = 0x60, OP_NEW_MAP = 0x61, OP_NEW_STRUCT = 0x62, OP_GET_INDEX = 0x63,
    OP_SET_INDEX = 0x64, OP_GET_FIELD = 0x65, OP_SET_FIELD = 0x66, OP_GET_SLOT = 0x67,
    OP_MAKE_CLOSURE = 0x68, OP_PUSH_FUNC = 0x69, OP_CONCAT = 0x6A, OP_FOR_ITER = 0x6B,
    OP_MATCH_TYPE = 0x6C, OP_CAST = 0x6D, OP_IS_ARRAY_LEN = 0x6E,
    OP_TRY = 0x70, OP_TRY_TIMEOUT = 0x71, OP_END_TRY = 0x72, OP_PANIC = 0x73,
    OP_UNWRAP_OR_RETURN = 0x74, OP_UNWRAP_OR_PANIC = 0x75,
    /* effects with resumption (TIL 3.1, docs/dev/TIL_3_1_DESIGN.md section 8) */
    OP_HANDLE = 0x76, OP_END_HANDLE = 0x77, OP_PERFORM = 0x78, OP_RESUME = 0x79,
    OP_HALT = 0x7F,
    /* 0x80-0xBF: superinstructions emitted by the compiler's peephole pass
     * (TIL 3.1, docs/TIL_BYTECODE_V2.md); each one does exactly what the
     * instruction sequence it replaces does */
    OP_LOAD_LOCAL2 = 0x80, OP_LOAD_LOCAL_FIELD = 0x81, OP_INC_LOCAL = 0x82, OP_CAST_LOCAL = 0x83,
    OP_CAST_RETURN = 0x84, OP_ADD_STORE_LOCAL = 0x85, OP_NATIVE_POP = 0x86, OP_INDEX_L = 0x87,
    OP_INDEX_LL = 0x88, OP_RETURN_LOCAL = 0x89, OP_FOR_ITER_STORE = 0x8A,
    OP_JUMP_IF_NOT = 0x90, OP_JUMP_IF_NOT_L = 0x91, OP_JUMP_IF_NOT_I = 0x92, OP_JUMP_IF_NOT_LL = 0x93,
    OP_JUMP_IF_NOT_LI = 0x94,
    /* <op>_I k = PUSH_I8/PUSH_I16 k; <op> (k: i16)   <op>_L b = LOAD_LOCAL b; <op>   (op: FUSED_BINOP) */
    OP_ADD_I = 0xA0, OP_SUB_I = 0xA1, OP_MUL_I = 0xA2, OP_DIV_I = 0xA3, OP_MOD_I = 0xA4,
    OP_BAND_I = 0xA5, OP_BOR_I = 0xA6, OP_BXOR_I = 0xA7, OP_SHL_I = 0xA8, OP_SHR_I = 0xA9,
    OP_EQ_I = 0xAA, OP_NE_I = 0xAB, OP_LT_I = 0xAC, OP_LE_I = 0xAD, OP_GT_I = 0xAE, OP_GE_I = 0xAF,
    OP_ADD_L = 0xB0, OP_SUB_L = 0xB1, OP_MUL_L = 0xB2, OP_DIV_L = 0xB3, OP_MOD_L = 0xB4,
    OP_BAND_L = 0xB5, OP_BOR_L = 0xB6, OP_BXOR_L = 0xB7, OP_SHL_L = 0xB8, OP_SHR_L = 0xB9,
    OP_EQ_L = 0xBA, OP_NE_L = 0xBB, OP_LT_L = 0xBC, OP_LE_L = 0xBD, OP_GT_L = 0xBE, OP_GE_L = 0xBF
};

/* base operator of the <op>_I / <op>_L superinstructions (opcode & 15) */
static const uint8_t FUSED_BINOP[16] = {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD, OP_BAND, OP_BOR, OP_BXOR, OP_SHL, OP_SHR,
    OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE
};

/* NATIVES (order = NATIVES in src/til.py) */
enum {
    N_PRINT, N_LEN, N_STR, N_INT, N_FLOAT, N_ABS, N_SQRT, N_POW, N_SIN, N_COS,
    N_TAN, N_LOG, N_EXP, N_FLOOR, N_CEIL, N_ROUND, N_MIN, N_MAX, N_INPUT, N_CHR,
    N_ORD, N_RANGE, N_TYPE_OF, N_FORMAT, N_CLOCK_MS, N_RANDOM, N_SEED, N_SLEEP_MS,
    N_EXIT, N_BOOL, N_SUM, N_PANIC, N_ITER, N_PRINTLN,
    /* TIL 3.1 (bytecode v2) */
    N_READ_FILE, N_WRITE_FILE, N_FILE_EXISTS, N_ARGS, N_BYTES_TO_STR, N_STR_BYTES,
    N_SCREEN_INIT, N_SCREEN_CLEAR, N_DRAW_PIXEL, N_DRAW_LINE, N_DRAW_RECT, N_FILL_RECT,
    N_DRAW_CIRCLE, N_DRAW_TEXT, N_GET_PIXEL, N_SCREEN_SHOW, N_KEY_PRESSED, N_WAIT_KEY,
    N_SCREEN_WIDTH, N_SCREEN_HEIGHT, N_ENERGY_MODE
};

/* METHODS (order = METHODS in src/til.py) */
enum {
    M_LEN, M_PUSH, M_POP, M_GET, M_SET, M_CONTAINS, M_STARTS_WITH, M_ENDS_WITH,
    M_TRIM, M_TO_UPPER, M_TO_LOWER, M_REPLACE, M_SLICE, M_FIND, M_EQ, M_SPLIT,
    M_REPEAT, M_TO_INT, M_TO_FLOAT, M_IS_EMPTY, M_INSERT, M_REMOVE, M_INDEX_OF,
    M_REVERSE, M_SORT, M_JOIN, M_FIRST, M_LAST, M_CLEAR, M_SUM, M_MIN, M_MAX,
    M_HAS, M_DELETE, M_KEYS, M_VALUES, M_GET_OR, M_UNWRAP, M_EXPECT, M_UNWRAP_OR,
    M_IS_SOME, M_IS_NONE, M_IS_OK, M_IS_ERR, M_ERROR, M_TO_STRING, M_COPY, M_COUNT,
    M_MAP, M_FILTER, M_REDUCE, M_EACH, M_ANY, M_ALL, M_NONE_ = 0xFF
};

#define TYPE_SOME 0
#define TYPE_OK   1
#define TYPE_ERR  2

#define FLAG_PROPAGATE 2
#define FLAG_ENERGY_LOW 8
#define FLAG_EFFECT_CLAUSE 16

/* Longest string: 64 KB - 1 with 16-bit heap references, else 16 MB */
#if TILVM_HEAP_SIZE <= 65535L
#define TV_STR_MAX 0xFFFFul
#else
#define TV_STR_MAX 0xFFFFFFul
#endif

/* ─── heap objects ──────────────────────────────────────────────────────── */
enum { OBJ_STR = 1, OBJ_ARRAY, OBJ_MAP, OBJ_VALS, OBJ_STRUCT, OBJ_CLOSURE, OBJ_CELL };

typedef struct tv_hdr {
    uint8_t kind;
    uint8_t mark;
    uint16_t aux;       /* struct: type index; closure: upvalue count */
    tv_size size;       /* whole object, bytes */
    tv_ref fwd;         /* forwarding address while compacting */
} tv_hdr;

typedef struct { tv_hdr h; tv_size len; char data[1]; } tv_strobj;
typedef struct { tv_hdr h; tv_size len; tv_ref buf; } tv_arrobj;     /* arrays and maps */
typedef struct { tv_hdr h; tv_val v[1]; } tv_valsobj;               /* capacity = aux */
typedef struct { tv_hdr h; tv_val v[1]; } tv_structobj;             /* fields */
typedef struct { tv_hdr h; uint16_t func; tv_val v[1]; } tv_closobj;  /* upvalues (cells) */
typedef struct { tv_hdr h; tv_val v; } tv_cellobj;

struct tv_align_probe { char c; tv_val v; };
#define TV_ALIGN (offsetof(struct tv_align_probe, v) < 2 ? 2 : offsetof(struct tv_align_probe, v))
#define ALIGN_UP(n) (((n) + TV_ALIGN - 1) / TV_ALIGN * TV_ALIGN)
#define OBJ(vm, r) ((tv_hdr *)((vm)->mem.heap + (r)))
#define STROBJ(vm, r) ((tv_strobj *)((vm)->mem.heap + (r)))
#define ARROBJ(vm, r) ((tv_arrobj *)((vm)->mem.heap + (r)))
#define VALSOBJ(vm, r) ((tv_valsobj *)((vm)->mem.heap + (r)))
#define STRUCTOBJ(vm, r) ((tv_structobj *)((vm)->mem.heap + (r)))
#define CLOSOBJ(vm, r) ((tv_closobj *)((vm)->mem.heap + (r)))
#define CELLOBJ(vm, r) ((tv_cellobj *)((vm)->mem.heap + (r)))
#define VALS(vm, r) (VALSOBJ(vm, r)->v)
#define IS_HEAP(t) ((t) == TV_STR || (t) == TV_ARRAY || (t) == TV_MAP || (t) == TV_STRUCT || \
                    (t) == TV_CLOSURE || (t) == TV_CELL)

#define RD16(p) ((uint16_t)((uint16_t)(p)[0] | ((uint16_t)(p)[1] << 8)))
#define RD32(p) ((uint32_t)RD16(p) | ((uint32_t)RD16((p) + 2) << 16))

#define SP (vm->st.sp)
#define STK (vm->st.stack)
#define TOP (vm->st.stack[vm->st.sp - 1])

/* Names in bytecode-format order (the test-suite checks them against src/til.py) */
static const char *const NATIVE_NAMES[] = {
    "print", "len", "str", "int", "float", "abs", "sqrt", "pow", "sin", "cos",
    "tan", "log", "exp", "floor", "ceil", "round", "min", "max", "input", "chr",
    "ord", "range", "type_of", "format", "clock_ms", "random", "seed", "sleep_ms",
    "exit", "bool", "sum", "panic", "iter", "println",
    "read_file", "write_file", "file_exists", "args", "bytes_to_str", "str_bytes",
    "screen_init", "screen_clear", "draw_pixel", "draw_line", "draw_rect", "fill_rect",
    "draw_circle", "draw_text", "get_pixel", "screen_show", "key_pressed", "wait_key",
    "screen_width", "screen_height", "energy_mode"
};
static const char *const METHOD_NAMES[] = {
    "len", "push", "pop", "get", "set", "contains", "starts_with", "ends_with",
    "trim", "to_upper", "to_lower", "replace", "slice", "find", "eq", "split",
    "repeat", "to_int", "to_float", "is_empty", "insert", "remove", "index_of",
    "reverse", "sort", "join", "first", "last", "clear", "sum", "min", "max",
    "has", "delete", "keys", "values", "get_or", "unwrap", "expect", "unwrap_or",
    "is_some", "is_none", "is_ok", "is_err", "error", "to_string", "copy", "count",
    "map", "filter", "reduce", "each", "any", "all"
};
#define N_NATIVES (sizeof(NATIVE_NAMES) / sizeof(NATIVE_NAMES[0]))
#define N_METHODS (sizeof(METHOD_NAMES) / sizeof(METHOD_NAMES[0]))

/* ─── small helpers ─────────────────────────────────────────────────────── */
static void set_nil(tv_val *v) { v->t = TV_NIL; v->u.i = 0; }
static void set_bool(tv_val *v, int b) { v->t = TV_BOOL; v->u.i = b ? 1 : 0; }
static void set_int(tv_val *v, tv_int i) { v->t = TV_INT; v->u.i = i; }
#if TILVM_FLOAT
static void set_float(tv_val *v, tv_float f) { v->t = TV_FLOAT; v->u.f = f; }
#endif
static void set_ref(tv_val *v, uint8_t t, tv_ref r) { v->t = t; v->u.i = 0; v->u.r = r; }

static int fail(tilvm *vm, const char *msg) {
    vm->err_msg = msg;
    set_nil(&vm->err_val);
    return 0;
}

static char *fmt_uint(char *end, tv_uint n) {     /* writes backwards, returns start */
    *--end = 0;
    do { *--end = (char)('0' + (int)(n % 10)); n /= 10; } while (n);
    return end;
}

static char *fmt_int(char *buf32, tv_int n) {
    char *p = fmt_uint(buf32 + 31, n < 0 ? (tv_uint)0 - (tv_uint)n : (tv_uint)n);
    if (n < 0) *--p = '-';
    return p;
}

/* Build an error message from up to 6 C strings (NULL entries are skipped). */
static int failn(tilvm *vm, const char *p0, const char *p1, const char *p2,
                 const char *p3, const char *p4, const char *p5) {
    const char *parts[6];
    size_t pos = 0, cap = sizeof(vm->errbuf) - 1;
    int k;
    parts[0] = p0; parts[1] = p1; parts[2] = p2; parts[3] = p3; parts[4] = p4; parts[5] = p5;
    for (k = 0; k < 6; k++) {
        const char *q = parts[k];
        if (!q) continue;
        while (*q && pos < cap) vm->errbuf[pos++] = *q++;
    }
    vm->errbuf[pos] = 0;
    return fail(vm, vm->errbuf);
}

static int failf(tilvm *vm, const char *a, tv_int n1, const char *b, tv_int n2, const char *c) {
    /* tiny formatter: a <n1> b <n2> c  (no printf needed) */
    char num[32];
    size_t pos = 0, cap = sizeof(vm->errbuf) - 1;
    const char *parts[5];
    int k;
    parts[0] = a; parts[1] = 0; parts[2] = b; parts[3] = 0; parts[4] = c;
    for (k = 0; k < 5; k++) {
        const char *s = parts[k];
        if (k == 1) s = fmt_int(num, n1);
        if (k == 3) { if (!b) break; s = fmt_int(num, n2); }
        if (!s) continue;
        while (*s && pos < cap) vm->errbuf[pos++] = *s++;
    }
    vm->errbuf[pos] = 0;
    return fail(vm, vm->errbuf);
}

/* ─── host bindings ─────────────────────────────────────────────────────── */
#ifndef TILVM_NO_STDIO
void tilvm_default_write(void *ctx, const char *data, size_t len) {
    (void)ctx;
    fwrite(data, 1, len, stdout);
}
static void default_write_err(void *ctx, const char *data, size_t len) {
    (void)ctx;
    fflush(stdout);
    fwrite(data, 1, len, stderr);
}
static int default_read_line(void *ctx, char *buf, size_t cap) {
    size_t n;
    (void)ctx;
    fflush(stdout);
    if (!fgets(buf, (int)cap, stdin)) return -1;
    n = strlen(buf);
    if (n && buf[n - 1] == '\n') buf[--n] = 0;
    return (int)n;
}
/* files: the size is measured with fseek/ftell; a directory or unreadable file is -1 */
static long default_read_file(void *ctx, const char *path, char *buf, unsigned long cap) {
    FILE *f = fopen(path, "rb");
    long n;
    int c;
    (void)ctx;
    if (!f) return -1;
    c = getc(f);                                   /* fails for directories */
    if (c == EOF && ferror(f)) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    if (buf && cap) {
        size_t want = (unsigned long)n < cap ? (size_t)n : (size_t)cap;
        if (fread(buf, 1, want, f) != want) { fclose(f); return -1; }
    }
    fclose(f);
    return n;
}
static int default_write_file(void *ctx, const char *path, const char *data, unsigned long len) {
    FILE *f = fopen(path, "wb");
    int ok;
    (void)ctx;
    if (!f) return -1;
    ok = fwrite(data, 1, (size_t)len, f) == (size_t)len;
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}
#else
void tilvm_default_write(void *ctx, const char *data, size_t len) { (void)ctx; (void)data; (void)len; }
#endif

#ifdef TV_HAVE_CLOCK
static uint32_t default_clock_ms(void *ctx) {
    (void)ctx;
    return (uint32_t)((double)clock() * 1000.0 / (double)CLOCKS_PER_SEC);
}
#endif

static uint32_t now_ms(tilvm *vm) {
    if (vm->clock_ms) return vm->clock_ms(vm->ctx);
    return vm->st.steps / 1000u;   /* no clock: 1 ms per 1000 steps */
}

static void out(tilvm *vm, const char *s, size_t n) {
    if (vm->write && n) vm->write(vm->ctx, s, n);
}

/* ─── allocation & garbage collection ──────────────────────────────────── */
static void gc(tilvm *vm);

static tv_ref alloc_obj(tilvm *vm, uint8_t kind, uint32_t size) {
    tv_ref r;
    tv_hdr *h;
    size = (uint32_t)ALIGN_UP(size);
    if (size > (uint32_t)TILVM_HEAP_SIZE) { fail(vm, "out of memory"); return 0; }
    if ((uint32_t)vm->st.heap_top + size > (uint32_t)TILVM_HEAP_SIZE) {
        gc(vm);
        if ((uint32_t)vm->st.heap_top + size > (uint32_t)TILVM_HEAP_SIZE) {
            fail(vm, "out of memory");
            return 0;
        }
    }
    r = vm->st.heap_top;
    vm->st.heap_top = (tv_size)(vm->st.heap_top + size);
    memset(vm->mem.heap + r, 0, size);
    h = OBJ(vm, r);
    h->kind = kind;
    h->size = (tv_size)size;
    return r;
}

static void mark_ref(tilvm *vm, tv_ref r) {
    tv_hdr *h;
    if (!r) return;
    h = OBJ(vm, r);
    if (h->mark) return;
    h->mark = 1;
    if (h->kind == OBJ_STR) return;
    if (vm->mark_sp < TILVM_MARK_STACK) vm->mark_stack[vm->mark_sp++] = r;
    else vm->mark_overflow = 1;
}

static void mark_val(tilvm *vm, const tv_val *v) {
    if (IS_HEAP(v->t)) mark_ref(vm, v->u.r);
}

static uint32_t obj_nvals(tilvm *vm, tv_ref r, tv_val **vals) {
    tv_hdr *h = OBJ(vm, r);
    switch (h->kind) {
    case OBJ_VALS: *vals = VALSOBJ(vm, r)->v; return h->aux;
    case OBJ_STRUCT: *vals = STRUCTOBJ(vm, r)->v;
        return (uint32_t)((h->size - offsetof(tv_structobj, v)) / sizeof(tv_val));
    case OBJ_CLOSURE: *vals = CLOSOBJ(vm, r)->v; return h->aux;
    case OBJ_CELL: *vals = &CELLOBJ(vm, r)->v; return 1;
    default: *vals = 0; return 0;
    }
}

static void scan_children(tilvm *vm, tv_ref r) {
    tv_hdr *h = OBJ(vm, r);
    tv_val *vals;
    uint32_t n, i;
    if (h->kind == OBJ_ARRAY || h->kind == OBJ_MAP) {
        mark_ref(vm, ARROBJ(vm, r)->buf);
        return;
    }
    n = obj_nvals(vm, r, &vals);
    for (i = 0; i < n; i++) mark_val(vm, &vals[i]);
}

static tv_ref fwd(tilvm *vm, tv_ref r) { return r ? OBJ(vm, r)->fwd : 0; }

static void fix_val(tilvm *vm, tv_val *v) {
    if (IS_HEAP(v->t)) v->u.r = fwd(vm, v->u.r);
}

/* Mark-compact (Lisp-2): mark, compute new addresses, fix references, slide. */
static void gc(tilvm *vm) {
    tilvm_state *st = &vm->st;
    tv_size start = (tv_size)ALIGN_UP(1), top = st->heap_top, scan, free_at;
    uint32_t i;
    st->gc_count++;
    /* 1. mark */
    vm->mark_sp = 0;
    vm->mark_overflow = 0;
    for (i = 0; i < st->sp; i++) mark_val(vm, &st->stack[i]);
    for (i = 0; i < st->n_frames; i++) mark_ref(vm, st->frames[i].closure);
    mark_val(vm, &vm->err_val);
    for (;;) {
        while (vm->mark_sp) scan_children(vm, vm->mark_stack[--vm->mark_sp]);
        if (!vm->mark_overflow) break;
        vm->mark_overflow = 0;     /* mark stack overflowed: rescan marked objects */
        for (scan = start; scan < top; scan = (tv_size)(scan + OBJ(vm, scan)->size))
            if (OBJ(vm, scan)->mark) scan_children(vm, scan);
    }
    /* 2. forwarding addresses */
    free_at = start;
    for (scan = start; scan < top; scan = (tv_size)(scan + OBJ(vm, scan)->size)) {
        tv_hdr *h = OBJ(vm, scan);
        if (h->mark) { h->fwd = free_at; free_at = (tv_size)(free_at + h->size); }
    }
    /* 3. update references */
    for (i = 0; i < st->sp; i++) fix_val(vm, &st->stack[i]);
    for (i = 0; i < st->n_frames; i++) st->frames[i].closure = fwd(vm, st->frames[i].closure);
    fix_val(vm, &vm->err_val);
    for (scan = start; scan < top; scan = (tv_size)(scan + OBJ(vm, scan)->size)) {
        tv_hdr *h = OBJ(vm, scan);
        tv_val *vals;
        uint32_t n, k;
        if (!h->mark) continue;
        if (h->kind == OBJ_ARRAY || h->kind == OBJ_MAP) {
            ARROBJ(vm, scan)->buf = fwd(vm, ARROBJ(vm, scan)->buf);
            continue;
        }
        n = obj_nvals(vm, scan, &vals);
        for (k = 0; k < n; k++) fix_val(vm, &vals[k]);
    }
    /* 4. slide */
    for (scan = start; scan < top;) {
        tv_hdr *h = OBJ(vm, scan);
        tv_size size = h->size;
        if (h->mark) {
            tv_ref to = h->fwd;
            h->mark = 0;
            if (to != scan) memmove(vm->mem.heap + to, vm->mem.heap + scan, size);
        }
        scan = (tv_size)(scan + size);
    }
    st->heap_top = free_at;
}

uint32_t tilvm_heap_used(const tilvm *vm) { return vm->st.heap_top; }

/* ─── strings ───────────────────────────────────────────────────────────── */
static const uint8_t *const_ptr(tilvm *vm, uint16_t k) { return vm->bc + vm->const_off[k]; }

static int is_str(const tv_val *v) { return v->t == TV_STR || v->t == TV_STRK; }

static const char *str_data(tilvm *vm, const tv_val *v, uint32_t *len) {
    if (v->t == TV_STRK) {
        const uint8_t *p = const_ptr(vm, v->u.k);
        *len = RD16(p);
        return (const char *)p + 2;
    }
    if (v->t == TV_STR) {
        tv_strobj *s = STROBJ(vm, v->u.r);
        *len = s->len;
        return s->data;
    }
    *len = 0;
    return "";
}

/* New heap string of `len` bytes (contents zeroed). May run the GC. */
static tv_ref new_str(tilvm *vm, uint32_t len) {
    tv_ref r = alloc_obj(vm, OBJ_STR, (uint32_t)offsetof(tv_strobj, data) + len + 1);
    if (r) STROBJ(vm, r)->len = (tv_size)len;
    return r;
}

#ifndef TILVM_LEAN
static int make_str(tilvm *vm, tv_val *dst, const char *s, uint32_t len) {
    tv_ref r;
    if (len > TV_STR_MAX) return fail(vm, "string too long");
    r = new_str(vm, len);
    if (!r) return 0;
    memcpy(STROBJ(vm, r)->data, s, len);
    set_ref(dst, TV_STR, r);
    return 1;
}
#endif

static int str_eq(tilvm *vm, const tv_val *a, const tv_val *b) {
    uint32_t la, lb;
    const char *pa = str_data(vm, a, &la), *pb = str_data(vm, b, &lb);
    return la == lb && memcmp(pa, pb, la) == 0;
}

static int str_cmp(tilvm *vm, const tv_val *a, const tv_val *b) {
    uint32_t la, lb, n;
    const char *pa = str_data(vm, a, &la), *pb = str_data(vm, b, &lb);
    int c;
    n = la < lb ? la : lb;
    c = memcmp(pa, pb, n);
    if (c) return c;
    return la < lb ? -1 : la > lb ? 1 : 0;
}

/* ─── type names / metadata ─────────────────────────────────────────────── */
static uint16_t type_name_const(tilvm *vm, uint16_t t) { return RD16(vm->bc + vm->type_off[t]); }
static uint8_t type_nfields(tilvm *vm, uint16_t t) { return vm->bc[vm->type_off[t] + 2]; }
static uint16_t type_field_const(tilvm *vm, uint16_t t, uint8_t i) {
    return RD16(vm->bc + vm->type_off[t] + 3 + 2 * i);
}

static const uint8_t *func_rec(tilvm *vm, uint16_t f) { return vm->bc + vm->funcs_off + 11u * f; }
#define FN_NAME(p) RD16(p)
#define FN_NPARAMS(p) ((p)[2])
#define FN_NREQ(p) ((p)[3])
#define FN_NLOCALS(p) ((p)[4])
#define FN_FLAGS(p) ((p)[6])
#define FN_OFFSET(p) RD32((p) + 7)

/* function records: decoded once by tilvm_load (TILVM_MAX_FUNCS > 0) or read
 * from the bytecode on every call (TILVM_MAX_FUNCS == 0, smallest RAM) */
#if TILVM_MAX_FUNCS > 0
#define F_CODE(vm, f) ((vm)->funcs[f].code)
#define F_NPARAMS(vm, f) ((vm)->funcs[f].nparams)
#define F_NREQ(vm, f) ((vm)->funcs[f].nreq)
#define F_NLOCALS(vm, f) ((vm)->funcs[f].nlocals)
#else
#define F_CODE(vm, f) ((vm)->code_off + FN_OFFSET(func_rec(vm, f)))
#define F_NPARAMS(vm, f) FN_NPARAMS(func_rec(vm, f))
#define F_NREQ(vm, f) FN_NREQ(func_rec(vm, f))
#define F_NLOCALS(vm, f) FN_NLOCALS(func_rec(vm, f))
#endif

/* compile-time check: the lookup cache size is a power of two (or 0) */
typedef char tv_check_cache_size[(TILVM_FIELD_CACHE & (TILVM_FIELD_CACHE - 1)) == 0 ? 1 : -1];
#define CACHE_EMPTY 0xFFFFu

static int const_str_eq(tilvm *vm, uint16_t k, const char *s) {
    const uint8_t *p = const_ptr(vm, k);
    uint16_t n = RD16(p);
    return strlen(s) == n && memcmp(p + 2, s, n) == 0;
}

/* ─── value -> text ─────────────────────────────────────────────────────── */
typedef struct sink {
    tilvm *vm;
    int mode;          /* 0 count, 1 copy into buffer, 2 program output */
    char *dst;
    uint32_t len;
} sink;

static void sink_put(sink *s, const char *p, uint32_t n) {
    if (s->mode == 1) memcpy(s->dst + s->len, p, n);
    else if (s->mode == 2) out(s->vm, p, n);
    s->len += n;
}

static void sink_cstr(sink *s, const char *p) { sink_put(s, p, (uint32_t)strlen(p)); }

#if TILVM_FLOAT
#if !defined(TILVM_NO_STDIO)
static void fmt_g(char *buf, tv_float x) {
    if (x != x) { strcpy(buf, "nan"); return; }
    snprintf(buf, 40, "%g", (double)x);
    if (buf[0] == '-' && buf[1] == 'n') strcpy(buf, "nan");
}
#else
/* printf-free float formatting for freestanding builds (calculators, MCUs). */
static int fmt_special(char *buf, tv_float x) {
    if (x != x) { strcpy(buf, "nan"); return 1; }
    if (x - x != x - x) { strcpy(buf, x < 0 ? "-inf" : "inf"); return 1; }
    return 0;
}

/* normalize |x| to [1, 10): returns the decimal exponent */
static int fnorm(tv_float *x) {
    int e = 0;
    if (*x == 0) return 0;
    while (*x >= 10) { *x /= 10; e++; }
    while (*x < 1) { *x *= 10; e--; }
    return e;
}

/* %.{sig}g */
static void fmt_gp(char *buf, tv_float x, int sig) {
    char digits[24];
    int e, i, nd, pos = 0;
    tv_float m;
    if (fmt_special(buf, x)) return;
    if (sig < 1) sig = 1;
    if (sig > 17) sig = 17;
    if (x < 0) { buf[pos++] = '-'; x = -x; }
    if (x == 0) { strcpy(buf + pos, "0"); return; }
    e = fnorm(&x);
    m = x;
    for (i = 0; i < sig; i++) { int d = (int)m; if (d > 9) d = 9; digits[i] = (char)('0' + d); m = (m - (tv_float)d) * 10; }
    if (m >= 5) {                                   /* round half up */
        for (i = sig - 1; i >= 0; i--) {
            if (digits[i] < '9') { digits[i]++; break; }
            digits[i] = '0';
        }
        if (i < 0) { for (i = sig - 1; i > 0; i--) digits[i] = digits[i - 1]; digits[0] = '1'; e++; }
    }
    nd = sig;
    while (nd > 1 && digits[nd - 1] == '0') nd--;
    if (e < -4 || e >= sig) {
        buf[pos++] = digits[0];
        if (nd > 1) { buf[pos++] = '.'; for (i = 1; i < nd; i++) buf[pos++] = digits[i]; }
        buf[pos++] = 'e';
        buf[pos++] = e < 0 ? '-' : '+';
        if (e < 0) e = -e;
        if (e >= 100) buf[pos++] = (char)('0' + e / 100);
        buf[pos++] = (char)('0' + (e / 10) % 10);
        buf[pos++] = (char)('0' + e % 10);
    } else if (e >= 0) {
        for (i = 0; i <= e; i++) buf[pos++] = i < nd ? digits[i] : '0';
        if (nd > e + 1) { buf[pos++] = '.'; for (i = e + 1; i < nd; i++) buf[pos++] = digits[i]; }
    } else {
        buf[pos++] = '0'; buf[pos++] = '.';
        for (i = 0; i < -e - 1; i++) buf[pos++] = '0';
        for (i = 0; i < nd; i++) buf[pos++] = digits[i];
    }
    buf[pos] = 0;
}

static void fmt_g(char *buf, tv_float x) { fmt_gp(buf, x, 6); }

/* %.{prec}f  (exact ties round to even, like printf) */
static void fmt_fixed(char *buf, tv_float x, int prec) {
    char dig[72];
    int pos = 0, n = 0, i, intlen;
    tv_float ipart, frac, p10 = 1;
    if (fmt_special(buf, x)) return;
    if (prec > 20) prec = 20;
    if (x < 0) { buf[pos++] = '-'; x = -x; }
    ipart = x;
    dig[n++] = '0';                       /* spare leading digit for carries */
    {
        int k = 0;
        while (p10 * 10 <= ipart && k < 40) { p10 *= 10; k++; }
        for (; k >= 0; k--) {
            int d = 0;
            while (ipart >= p10 && d < 9) { ipart -= p10; d++; }
            dig[n++] = (char)('0' + d);
            p10 /= 10;
        }
    }
    intlen = n;
    frac = ipart < 1 ? ipart : 0;
    for (i = 0; i < prec; i++) {
        int d;
        frac *= 10;
        d = (int)frac;
        if (d > 9) d = 9;
        dig[n++] = (char)('0' + d);
        frac -= (tv_float)d;
    }
    if (frac > (tv_float)0.5 || (frac == (tv_float)0.5 && ((dig[n - 1] - '0') & 1))) {
        for (i = n - 1; i >= 0; i--) {
            if (dig[i] < '9') { dig[i]++; break; }
            dig[i] = '0';
        }
    }
    i = dig[0] == '0' ? 1 : 0;            /* drop the spare digit if unused */
    for (; i < intlen; i++) buf[pos++] = dig[i];
    if (prec > 0) {
        buf[pos++] = '.';
        for (i = intlen; i < n; i++) buf[pos++] = dig[i];
    }
    buf[pos] = 0;
}

/* %.{prec}e */
static void fmt_exp(char *buf, tv_float x, int prec) {
    int e, pos = 0, i;
    tv_float m;
    char mant[32];
    if (fmt_special(buf, x)) return;
    if (x < 0) { buf[pos++] = '-'; x = -x; }
    e = fnorm(&x);
    m = x;
    fmt_fixed(mant, m, prec);
    if (mant[0] == '1' && mant[1] == '0') {        /* 9.99 rounded up to 10.0 */
        m /= 10; e++;
        fmt_fixed(mant, m, prec);
    }
    for (i = 0; mant[i]; i++) buf[pos++] = mant[i];
    buf[pos++] = 'e';
    buf[pos++] = e < 0 ? '-' : '+';
    if (e < 0) e = -e;
    if (e >= 100) buf[pos++] = (char)('0' + e / 100);
    buf[pos++] = (char)('0' + (e / 10) % 10);
    buf[pos++] = (char)('0' + e % 10);
    buf[pos] = 0;
}
#endif
#endif

static void emit_value(sink *s, const tv_val *v, int depth, int quote);

static void emit_type_name(sink *s, uint16_t t) {
    uint32_t n;
    tv_val name;
    const char *p;
    name.t = TV_STRK;
    name.u.k = type_name_const(s->vm, t);
    p = str_data(s->vm, &name, &n);
    sink_put(s, p, n);
}

static void emit_const_str(sink *s, uint16_t k) {
    tv_val v;
    uint32_t n;
    const char *p;
    v.t = TV_STRK;
    v.u.k = k;
    p = str_data(s->vm, &v, &n);
    sink_put(s, p, n);
}

static void emit_value(sink *s, const tv_val *v, int depth, int quote) {
    tilvm *vm = s->vm;
    char buf[48];
    uint32_t n, i;
    switch (v->t) {
    case TV_NIL: sink_cstr(s, "None"); return;
    case TV_BOOL: sink_cstr(s, v->u.i ? "true" : "false"); return;
    case TV_INT: sink_cstr(s, fmt_int(buf, v->u.i)); return;
#if TILVM_FLOAT
    case TV_FLOAT: fmt_g(buf, v->u.f); sink_cstr(s, buf); return;
#endif
    case TV_STRK:
    case TV_STR: {
        const char *p = str_data(vm, v, &n);
        if (quote) sink_put(s, "\"", 1);
        sink_put(s, p, n);
        if (quote) sink_put(s, "\"", 1);
        return;
    }
    case TV_FUNC:
    case TV_CLOSURE: {
        uint16_t f = v->t == TV_FUNC ? v->u.k : CLOSOBJ(vm, v->u.r)->func;
        sink_cstr(s, "<fn ");
        emit_const_str(s, FN_NAME(func_rec(vm, f)));
        sink_cstr(s, ">");
        return;
    }
    default: break;
    }
    if (depth > 16) { sink_cstr(s, "..."); return; }
    if (v->t == TV_ARRAY || v->t == TV_MAP) {
        tv_arrobj *a = ARROBJ(vm, v->u.r);
        uint32_t len = a->len;
        tv_ref buf_ref = a->buf;
        sink_cstr(s, v->t == TV_ARRAY ? "[" : "{");
        for (i = 0; i < len; i++) {
            tv_val *vals = VALS(vm, buf_ref);
            if (i) sink_cstr(s, ", ");
            if (v->t == TV_ARRAY) emit_value(s, &vals[i], depth + 1, 1);
            else {
                emit_value(s, &vals[2 * i], depth + 1, 1);
                sink_cstr(s, ": ");
                emit_value(s, &vals[2 * i + 1], depth + 1, 1);
            }
        }
        sink_cstr(s, v->t == TV_ARRAY ? "]" : "}");
        return;
    }
    if (v->t == TV_STRUCT) {
        tv_ref r = v->u.r;
        uint16_t t = OBJ(vm, r)->aux;
        uint8_t nf = type_nfields(vm, t);
        int positional = t <= TYPE_ERR;
        if (nf && !positional) positional = const_str_eq(vm, type_field_const(vm, t, 0), "0");
        emit_type_name(s, t);
        if (!nf) return;
        sink_cstr(s, positional ? "(" : " { ");
        for (i = 0; i < nf; i++) {
            if (i) sink_cstr(s, ", ");
            if (!positional) {
                emit_const_str(s, type_field_const(vm, t, (uint8_t)i));
                sink_cstr(s, ": ");
            }
            emit_value(s, &STRUCTOBJ(vm, r)->v[i], depth + 1, 1);
        }
        sink_cstr(s, positional ? ")" : " }");
        return;
    }
    sink_cstr(s, "?");
}

/* Convert stack slot *slot to a string value in place (may allocate). */
static int to_str_slot(tilvm *vm, uint16_t slot) {
    sink s;
    tv_ref r;
    tv_val *v = &vm->st.stack[slot];
    if (is_str(v)) return 1;
#if TILVM_FAST_PATHS
    if (v->t == TV_INT) {             /* common case: format once, allocate once */
        char num[32];
        const char *p = fmt_int(num, v->u.i);
        uint32_t n = (uint32_t)strlen(p);
        r = new_str(vm, n);
        if (!r) return 0;
        memcpy(STROBJ(vm, r)->data, p, n);
        set_ref(&vm->st.stack[slot], TV_STR, r);
        return 1;
    }
#endif
    s.vm = vm; s.mode = 0; s.dst = 0; s.len = 0;
    emit_value(&s, v, 0, 0);
    if (s.len > TV_STR_MAX) return fail(vm, "string too long");
    r = new_str(vm, s.len);           /* GC may move objects: re-read the slot */
    if (!r) return 0;
    s.mode = 1; s.dst = STROBJ(vm, r)->data; s.len = 0;
    emit_value(&s, &vm->st.stack[slot], 0, 0);
    set_ref(&vm->st.stack[slot], TV_STR, r);
    return 1;
}

static void print_value(tilvm *vm, const tv_val *v) {
    sink s;
    s.vm = vm; s.mode = 2; s.dst = 0; s.len = 0;
    emit_value(&s, v, 0, 0);
}

/* ─── equality / comparison / truthiness ───────────────────────────────── */
static int truthy(tilvm *vm, const tv_val *v) {
    uint32_t n;
    switch (v->t) {
    case TV_NIL: return 0;
    case TV_BOOL: return v->u.i != 0;
    case TV_INT: return v->u.i != 0;
#if TILVM_FLOAT
    case TV_FLOAT: return v->u.f != 0;
#endif
    case TV_STRK: case TV_STR: str_data(vm, v, &n); return n != 0;
    case TV_ARRAY: case TV_MAP: return ARROBJ(vm, v->u.r)->len != 0;
    default: return 1;
    }
}

static int is_num(const tv_val *v) {
#if TILVM_FLOAT
    return v->t == TV_INT || v->t == TV_FLOAT;
#else
    return v->t == TV_INT;
#endif
}

#if TILVM_FLOAT
static tv_float as_float(const tv_val *v) { return v->t == TV_INT ? (tv_float)v->u.i : v->u.f; }
#endif

static int equal(tilvm *vm, const tv_val *a, const tv_val *b, int depth) {
    uint32_t i, n;
    if (is_num(a) && is_num(b)) {
#if TILVM_FLOAT
        if (a->t == TV_FLOAT || b->t == TV_FLOAT) return as_float(a) == as_float(b);
#endif
        return a->u.i == b->u.i;
    }
    if (is_str(a) && is_str(b)) return str_eq(vm, a, b);
    if (a->t != b->t) return 0;
    switch (a->t) {
    case TV_NIL: return 1;
    case TV_BOOL: return a->u.i == b->u.i;
    case TV_FUNC: return a->u.k == b->u.k;
    case TV_CLOSURE: case TV_CELL: return a->u.r == b->u.r;
    default: break;
    }
    if (a->u.r == b->u.r) return 1;
    if (depth > 32) return 0;
    if (a->t == TV_ARRAY || a->t == TV_MAP) {
        tv_arrobj *x = ARROBJ(vm, a->u.r), *y = ARROBJ(vm, b->u.r);
        if (x->len != y->len) return 0;
        n = x->len;
        if (a->t == TV_ARRAY) {
            for (i = 0; i < n; i++)
                if (!equal(vm, &VALS(vm, x->buf)[i], &VALS(vm, y->buf)[i], depth + 1)) return 0;
            return 1;
        }
        for (i = 0; i < n; i++) {     /* maps: same keys, equal values */
            tv_val *k = &VALS(vm, x->buf)[2 * i];
            uint32_t j;
            int found = 0;
            for (j = 0; j < n; j++) {
                if (equal(vm, k, &VALS(vm, y->buf)[2 * j], depth + 1)) {
                    if (!equal(vm, &VALS(vm, x->buf)[2 * i + 1], &VALS(vm, y->buf)[2 * j + 1], depth + 1))
                        return 0;
                    found = 1;
                    break;
                }
            }
            if (!found) return 0;
        }
        return 1;
    }
    if (a->t == TV_STRUCT) {
        tv_hdr *ha = OBJ(vm, a->u.r), *hb = OBJ(vm, b->u.r);
        if (ha->aux != hb->aux) return 0;
        n = (uint32_t)((ha->size - offsetof(tv_structobj, v)) / sizeof(tv_val));
        for (i = 0; i < n; i++)
            if (!equal(vm, &STRUCTOBJ(vm, a->u.r)->v[i], &STRUCTOBJ(vm, b->u.r)->v[i], depth + 1)) return 0;
        return 1;
    }
    return 0;
}

static const char *type_name(tilvm *vm, const tv_val *v) {
    (void)vm;
    switch (v->t) {
    case TV_NIL: return "nil";
    case TV_BOOL: return "bool";
    case TV_INT: return "int";
    case TV_FLOAT: return "float";
    case TV_STRK: case TV_STR: return "str";
    case TV_ARRAY: return "array";
    case TV_MAP: return "map";
    case TV_FUNC: case TV_CLOSURE: return "fn";
    case TV_STRUCT: return "struct";
    default: return "?";
    }
}

static int type_error2(tilvm *vm, const char *what, const tv_val *a, const tv_val *b) {
    size_t pos = 0, cap = sizeof(vm->errbuf) - 1;
    const char *parts[4];
    int k;
    parts[0] = what; parts[1] = type_name(vm, a); parts[2] = " and "; parts[3] = type_name(vm, b);
    for (k = 0; k < 4; k++) {
        const char *s = parts[k];
        while (*s && pos < cap) vm->errbuf[pos++] = *s++;
    }
    vm->errbuf[pos] = 0;
    return fail(vm, vm->errbuf);
}

/* a < b etc. Returns 1 ok (result in *res), 0 error */
static int compare(tilvm *vm, int op, const tv_val *a, const tv_val *b, int *res) {
    int c;
    if (is_num(a) && is_num(b)) {
#if TILVM_FLOAT
        if (a->t == TV_FLOAT || b->t == TV_FLOAT) {
            tv_float x = as_float(a), y = as_float(b);
            *res = op == OP_LT ? x < y : op == OP_LE ? x <= y : op == OP_GT ? x > y : x >= y;
            return 1;
        }
#endif
        c = a->u.i < b->u.i ? -1 : a->u.i > b->u.i ? 1 : 0;
    } else if (is_str(a) && is_str(b)) {
        c = str_cmp(vm, a, b);
    } else {
        return type_error2(vm, "cannot compare ", a, b);
    }
    *res = op == OP_LT ? c < 0 : op == OP_LE ? c <= 0 : op == OP_GT ? c > 0 : c >= 0;
    return 1;
}

/* ─── integer arithmetic with wrap-around ───────────────────────────────── */
static tv_int iwrap_add(tv_int a, tv_int b) { return (tv_int)((tv_uint)a + (tv_uint)b); }
static tv_int iwrap_sub(tv_int a, tv_int b) { return (tv_int)((tv_uint)a - (tv_uint)b); }
static tv_int iwrap_mul(tv_int a, tv_int b) { return (tv_int)((tv_uint)a * (tv_uint)b); }
static tv_int idiv(tv_int a, tv_int b) {
    if (b == -1) return (tv_int)((tv_uint)0 - (tv_uint)a);   /* MIN / -1 wraps */
    return a / b;                                              /* C99: truncates */
}
static tv_int imod(tv_int a, tv_int b) { return b == -1 ? 0 : a % b; }
static tv_int ipow(tv_int base, tv_int e) {
    tv_uint r = 1, b = (tv_uint)base;
    if (e < 0) return base == 1 ? 1 : base == -1 ? ((e & 1) ? -1 : 1) : 0;
    while (e > 0) { if (e & 1) r *= b; b *= b; e >>= 1; }
    return (tv_int)r;
}
static tv_int ishr(tv_int a, unsigned n) {
    if (a >= 0) return (tv_int)((tv_uint)a >> n);
    return (tv_int)~((tv_uint)~a >> n);
}

#if TILVM_FLOAT
#ifdef TV_HAVE_MATH
#define F_SQRT(x) ((tv_float)sqrt((double)(x)))
#define F_POW(x, y) ((tv_float)pow((double)(x), (double)(y)))
#define F_FMOD(x, y) ((tv_float)fmod((double)(x), (double)(y)))
#define F_FLOOR(x) ((tv_float)floor((double)(x)))
#define F_CEIL(x) ((tv_float)ceil((double)(x)))
#else
static tv_float F_FLOOR(tv_float x) {
    tv_float t = (tv_float)(long)x;
    return (t > x) ? t - 1 : t;
}
static tv_float F_CEIL(tv_float x) { return -F_FLOOR(-x); }
static tv_float F_FMOD(tv_float x, tv_float y) {
    tv_float q = x / y;
    q = q < 0 ? -F_FLOOR(-q) : F_FLOOR(q);
    return x - q * y;
}
static tv_float F_SQRT(tv_float x) {
    tv_float g = x > 1 ? x / 2 : 1;
    int i;
    if (x <= 0) return 0;
    for (i = 0; i < 40; i++) g = (g + x / g) / 2;
    return g;
}
static tv_float F_POW(tv_float x, tv_float y) {
    tv_float r = 1;
    long n = (long)y;
    int neg = n < 0;
    if ((tv_float)n != y) return 0;    /* fractional powers need <math.h> */
    if (neg) n = -n;
    while (n) { if (n & 1) r *= x; x *= x; n >>= 1; }
    return neg ? 1 / r : r;
}
#endif
#endif

#ifndef TILVM_LEAN
static int tv_array_concat(tilvm *vm);
#endif

/* binary arithmetic: a = a OP b (a, b are stack slots; may allocate for strings) */
static int arith(tilvm *vm, int op) {
    tv_val *a = &STK[SP - 2], *b = &STK[SP - 1];
    if (a->t == TV_INT && b->t == TV_INT) {
        tv_int x = a->u.i, y = b->u.i, r;
        switch (op) {
        case OP_ADD: r = iwrap_add(x, y); break;
        case OP_SUB: r = iwrap_sub(x, y); break;
        case OP_MUL: r = iwrap_mul(x, y); break;
        case OP_DIV: if (!y) return fail(vm, "division by zero"); r = idiv(x, y); break;
        case OP_MOD: if (!y) return fail(vm, "division by zero"); r = imod(x, y); break;
        default: r = ipow(x, y); break;
        }
        set_int(a, r);
        SP--;
        return 1;
    }
#if TILVM_FLOAT
    if (is_num(a) && is_num(b)) {
        tv_float x = as_float(a), y = as_float(b), r;
        switch (op) {
        case OP_ADD: r = x + y; break;
        case OP_SUB: r = x - y; break;
        case OP_MUL: r = x * y; break;
        case OP_DIV: r = x / y; break;
        case OP_MOD: r = (y == 0) ? (tv_float)0 / (tv_float)0 : F_FMOD(x, y); break;
        default: r = F_POW(x, y); break;
        }
        set_float(a, r);
        SP--;
        return 1;
    }
#endif
    if (op == OP_ADD && (is_str(a) || is_str(b))) {
        uint32_t la, lb;
        tv_ref r;
        if (!to_str_slot(vm, (uint16_t)(SP - 2)) || !to_str_slot(vm, (uint16_t)(SP - 1))) return 0;
        str_data(vm, &STK[SP - 2], &la);
        str_data(vm, &STK[SP - 1], &lb);
        if (la + lb > TV_STR_MAX) return fail(vm, "string too long");
        r = new_str(vm, la + lb);
        if (!r) return 0;
        {
            const char *pa = str_data(vm, &STK[SP - 2], &la);
            const char *pb = str_data(vm, &STK[SP - 1], &lb);
            memcpy(STROBJ(vm, r)->data, pa, la);
            memcpy(STROBJ(vm, r)->data + la, pb, lb);
        }
        set_ref(&STK[SP - 2], TV_STR, r);
        SP--;
        return 1;
    }
    if (op == OP_MUL && is_str(a) && b->t == TV_INT) {
        uint32_t la, k, times = b->u.i > 0 ? (uint32_t)b->u.i : 0;
        tv_ref r;
        str_data(vm, a, &la);
        if (times && la > TV_STR_MAX / times) return fail(vm, "string too long");
        r = new_str(vm, la * times);
        if (!r) return 0;
        {
            const char *pa = str_data(vm, &STK[SP - 2], &la);
            for (k = 0; k < times; k++) memcpy(STROBJ(vm, r)->data + k * la, pa, la);
        }
        set_ref(&STK[SP - 2], TV_STR, r);
        SP--;
        return 1;
    }
#ifndef TILVM_LEAN
    if (op == OP_ADD && a->t == TV_ARRAY && b->t == TV_ARRAY) return tv_array_concat(vm);
#endif
    {
        static const char *const names[] = {"+", "-", "*", "/", "%", "**"};
        size_t pos = 0, cap = sizeof(vm->errbuf) - 1;
        const char *parts[6];
        int k;
        parts[0] = "unsupported operand types for "; parts[1] = names[op - OP_ADD];
        parts[2] = ": "; parts[3] = type_name(vm, a); parts[4] = " and "; parts[5] = type_name(vm, b);
        for (k = 0; k < 6; k++) {
            const char *s = parts[k];
            while (*s && pos < cap) vm->errbuf[pos++] = *s++;
        }
        vm->errbuf[pos] = 0;
        return fail(vm, vm->errbuf);
    }
}

/* ─── arrays and maps ───────────────────────────────────────────────────── */
static tv_ref new_vals(tilvm *vm, uint32_t cap) {
    tv_ref r;
    uint32_t i;
    if (cap > 0xFFFF) { fail(vm, "array too large"); return 0; }
    r = alloc_obj(vm, OBJ_VALS, (uint32_t)offsetof(tv_valsobj, v) + cap * (uint32_t)sizeof(tv_val));
    if (!r) return 0;
    OBJ(vm, r)->aux = (uint16_t)cap;
    for (i = 0; i < cap; i++) set_nil(&VALS(vm, r)[i]);
    return r;
}

/* Replace the top `n` stack values by a new array / map of them. */
static int make_array(tilvm *vm, uint8_t kind, uint32_t n) {
    uint32_t nvals = kind == OBJ_MAP ? 2 * n : n, cap = nvals < 4 ? 4 : nvals, i;
    tv_ref buf, arr;
    if (SP + 2 > TILVM_STACK_SIZE) return fail(vm, "stack overflow");
    buf = new_vals(vm, cap);
    if (!buf) return 0;
    set_ref(&STK[SP], TV_CELL, buf);          /* keep the buffer rooted */
    SP++;
    arr = alloc_obj(vm, kind, sizeof(tv_arrobj));
    SP--;
    if (!arr) return 0;
    buf = STK[SP].u.r;
    for (i = 0; i < nvals; i++) VALS(vm, buf)[i] = STK[SP - nvals + i];
    ARROBJ(vm, arr)->buf = buf;
    ARROBJ(vm, arr)->len = (tv_size)n;
    SP = (uint16_t)(SP - nvals);
    set_ref(&STK[SP], kind == OBJ_MAP ? TV_MAP : TV_ARRAY, arr);
    SP++;
    return 1;
}

/* Ensure array/map in slot has room for `need` values (may move). */
static int reserve(tilvm *vm, uint16_t slot, uint32_t need) {
    tv_ref arr = STK[slot].u.r, buf = ARROBJ(vm, arr)->buf, nb;
    uint32_t cap = OBJ(vm, buf)->aux, ncap, used, i;
    if (need <= cap) return 1;
    ncap = cap * 2;
    if (ncap < need) ncap = need;
    if (ncap > 0xFFFF) ncap = 0xFFFF;
    if (ncap < need) return fail(vm, "array too large");
    nb = new_vals(vm, ncap);
    if (!nb) return 0;
    arr = STK[slot].u.r;                       /* re-read after a possible GC */
    buf = ARROBJ(vm, arr)->buf;
    used = (STK[slot].t == TV_MAP ? 2u : 1u) * ARROBJ(vm, arr)->len;
    for (i = 0; i < used; i++) VALS(vm, nb)[i] = VALS(vm, buf)[i];
    ARROBJ(vm, arr)->buf = nb;
    return 1;
}

static int array_push(tilvm *vm, uint16_t arr_slot, uint16_t val_slot) {
    tv_ref arr;
    uint32_t len = ARROBJ(vm, STK[arr_slot].u.r)->len;
    if (!reserve(vm, arr_slot, len + 1)) return 0;
    arr = STK[arr_slot].u.r;
    VALS(vm, ARROBJ(vm, arr)->buf)[len] = STK[val_slot];
    ARROBJ(vm, arr)->len = (tv_size)(len + 1);
    return 1;
}

#ifndef TILVM_LEAN
static int tv_array_concat(tilvm *vm) {    /* [.., a, b] -> [.., a+b] */
    uint32_t la = ARROBJ(vm, STK[SP - 2].u.r)->len, lb = ARROBJ(vm, STK[SP - 1].u.r)->len, i;
    if (SP + 1 > TILVM_STACK_SIZE) return fail(vm, "stack overflow");
    for (i = 0; i < la + lb; i++) {           /* push copies of all elements */
        tv_val v;
        if (SP >= TILVM_STACK_SIZE - 2) return fail(vm, "stack overflow");
        v = i < la ? VALS(vm, ARROBJ(vm, STK[SP - 2 - i].u.r)->buf)[i]
                   : VALS(vm, ARROBJ(vm, STK[SP - 1 - i].u.r)->buf)[i - la];
        STK[SP++] = v;
    }
    if (!make_array(vm, OBJ_ARRAY, la + lb)) return 0;
    STK[SP - 3] = STK[SP - 1];
    SP -= 2;
    return 1;
}

#endif

/* normalize map keys: 2.0 -> 2 */
static void map_key(tv_val *k) {
#if TILVM_FLOAT
    if (k->t == TV_FLOAT && k->u.f == (tv_float)(tv_int)k->u.f) set_int(k, (tv_int)k->u.f);
#else
    (void)k;
#endif
}

static int map_find(tilvm *vm, tv_ref map, const tv_val *key) {
    tv_arrobj *m = ARROBJ(vm, map);
    uint32_t i, n = m->len;
#if TILVM_FAST_PATHS
    /* the usual key types, compared without the general equal() (same results) */
    const tv_val *v = VALS(vm, m->buf);
    if (key->t == TV_INT) {            /* only ints and floats can equal an int */
        for (i = 0; i < n; i++, v += 2) {
            if (v->t == TV_INT) { if (v->u.i == key->u.i) return (int)i; }
            else if (v->t == TV_FLOAT && equal(vm, v, key, 0)) return (int)i;
        }
        return -1;
    }
    if (is_str(key)) {                 /* only strings can equal a string */
        uint32_t kl, l;
        const char *kp = str_data(vm, key, &kl), *p;
        for (i = 0; i < n; i++, v += 2) {
            if (!is_str(v)) continue;
            p = str_data(vm, v, &l);
            if (l == kl && memcmp(p, kp, l) == 0) return (int)i;
        }
        return -1;
    }
#endif
    for (i = 0; i < n; i++)
        if (equal(vm, &VALS(vm, m->buf)[2 * i], key, 0)) return (int)i;
    return -1;
}

static int check_map_key(tilvm *vm, const tv_val *k) {
    if (k->t == TV_NIL || k->t == TV_BOOL || k->t == TV_INT || is_str(k)) return 1;
#if TILVM_FLOAT
    if (k->t == TV_FLOAT) return 1;
#endif
    return failn(vm, type_name(vm, k), " cannot be a map key", 0, 0, 0, 0);
}

/* map[key] = value with map, key, value in slots (no stack change) */
static int map_set(tilvm *vm, uint16_t map_slot, uint16_t key_slot, uint16_t val_slot) {
    int i;
    uint32_t len;
    map_key(&STK[key_slot]);
    if (!check_map_key(vm, &STK[key_slot])) return 0;
    i = map_find(vm, STK[map_slot].u.r, &STK[key_slot]);
    if (i >= 0) {
        VALS(vm, ARROBJ(vm, STK[map_slot].u.r)->buf)[2 * i + 1] = STK[val_slot];
        return 1;
    }
    len = ARROBJ(vm, STK[map_slot].u.r)->len;
    if (!reserve(vm, map_slot, 2 * (len + 1))) return 0;
    {
        tv_arrobj *m = ARROBJ(vm, STK[map_slot].u.r);
        VALS(vm, m->buf)[2 * len] = STK[key_slot];
        VALS(vm, m->buf)[2 * len + 1] = STK[val_slot];
        m->len = (tv_size)(len + 1);
    }
    return 1;
}

static void map_delete(tilvm *vm, tv_ref map, int i) {
    tv_arrobj *m = ARROBJ(vm, map);
    uint32_t j, n = m->len;
    tv_val *v = VALS(vm, m->buf);
    for (j = (uint32_t)i; j + 1 < n; j++) { v[2 * j] = v[2 * j + 2]; v[2 * j + 1] = v[2 * j + 3]; }
    set_nil(&v[2 * (n - 1)]);
    set_nil(&v[2 * (n - 1) + 1]);
    m->len = (tv_size)(n - 1);
}

static int key_not_found(tilvm *vm, uint16_t key_slot) {
    uint32_t pos = 0, cap = sizeof(vm->errbuf) - 1;
    sink s;
    const char *pre = "key not found: ";
    while (*pre && pos < cap) vm->errbuf[pos++] = *pre++;
    s.vm = vm; s.mode = 0; s.dst = 0; s.len = 0;
    emit_value(&s, &STK[key_slot], 0, 0);
    if (s.len <= cap - pos) {
        s.mode = 1; s.dst = vm->errbuf + pos; s.len = 0;
        emit_value(&s, &STK[key_slot], 0, 0);
        pos += s.len;
    }
    vm->errbuf[pos] = 0;
    return fail(vm, vm->errbuf);
}

static int bounds_error(tilvm *vm, tv_int idx, uint32_t len) {
    return failf(vm, "index out of bounds: index ", idx, ", length ", (tv_int)len, "");
}

/* [.., obj, idx] -> [.., obj[idx]] */
static int get_index(tilvm *vm) {
    tv_val *o = &STK[SP - 2], *ix = &STK[SP - 1];
    if (o->t == TV_ARRAY || is_str(o)) {
        uint32_t len;
        const char *p = 0;
        if (ix->t != TV_INT) return fail(vm, "index must be an integer");
        if (o->t == TV_ARRAY) len = ARROBJ(vm, o->u.r)->len;
        else p = str_data(vm, o, &len);
        if (ix->u.i < 0 || (tv_uint)ix->u.i >= len) return bounds_error(vm, ix->u.i, len);
        if (o->t == TV_ARRAY) *o = VALS(vm, ARROBJ(vm, o->u.r)->buf)[ix->u.i];
        else set_int(o, (tv_int)(uint8_t)p[ix->u.i]);
        SP--;
        return 1;
    }
    if (o->t == TV_MAP) {
        int i;
        map_key(ix);
        i = map_find(vm, o->u.r, ix);
        if (i < 0) return key_not_found(vm, (uint16_t)(SP - 1));
        *o = VALS(vm, ARROBJ(vm, o->u.r)->buf)[2 * i + 1];
        SP--;
        return 1;
    }
    if (o->t == TV_STRUCT && ix->t == TV_INT) {
        tv_hdr *h = OBJ(vm, o->u.r);
        uint32_t n = (uint32_t)((h->size - offsetof(tv_structobj, v)) / sizeof(tv_val));
        if (ix->u.i >= 0 && (tv_uint)ix->u.i < n) {
            *o = STRUCTOBJ(vm, o->u.r)->v[ix->u.i];
            SP--;
            return 1;
        }
    }
    {
        size_t pos = 0;
        const char *s1 = type_name(vm, o), *s2 = " cannot be indexed";
        while (*s1) vm->errbuf[pos++] = *s1++;
        while (*s2) vm->errbuf[pos++] = *s2++;
        vm->errbuf[pos] = 0;
        return fail(vm, vm->errbuf);
    }
}

/* [.., obj, idx, val] -> [..] */
static int set_index(tilvm *vm) {
    tv_val *o = &STK[SP - 3], *ix = &STK[SP - 2];
    if (o->t == TV_ARRAY) {
        uint32_t len = ARROBJ(vm, o->u.r)->len;
        if (ix->t != TV_INT) return fail(vm, "index must be an integer");
        if (ix->u.i < 0 || (tv_uint)ix->u.i >= len) return bounds_error(vm, ix->u.i, len);
        VALS(vm, ARROBJ(vm, o->u.r)->buf)[ix->u.i] = STK[SP - 1];
        SP -= 3;
        return 1;
    }
    if (o->t == TV_MAP) {
        if (!map_set(vm, (uint16_t)(SP - 3), (uint16_t)(SP - 2), (uint16_t)(SP - 1))) return 0;
        SP -= 3;
        return 1;
    }
    if (is_str(o)) return fail(vm, "strings are immutable");
    return failn(vm, type_name(vm, o), " does not support item assignment", 0, 0, 0, 0);
}

/* ─── structs ───────────────────────────────────────────────────────────── */
static int field_slot(tilvm *vm, uint16_t t, uint16_t name_const) {
    uint8_t n = type_nfields(vm, t), i;
    const uint8_t *p = const_ptr(vm, name_const);
    uint16_t len = RD16(p);
    for (i = 0; i < n; i++) {
        uint16_t fc = type_field_const(vm, t, i);
        const uint8_t *q;
        if (fc == name_const) return i;
        q = const_ptr(vm, fc);
        if (RD16(q) == len && memcmp(q + 2, p + 2, len) == 0) return i;
    }
    return -1;
}

/* field_slot() through a small direct-mapped cache: (type, name) -> slot.
 * The probe is a macro so that it is inlined into the interpreter. */
#if TILVM_FIELD_CACHE > 0
#define CACHE_AT(cache, T_, N_) (&(cache)[((uint32_t)(T_) * 31u + (N_)) & (TILVM_FIELD_CACHE - 1)])
static int field_slot_fill(tilvm *vm, uint16_t t, uint16_t name_const) {
    tv_cache *c = CACHE_AT(vm->field_cache, t, name_const);
    int slot = field_slot(vm, t, name_const);
    if (slot >= 0) { c->type = t; c->name = name_const; c->value = (uint16_t)slot; }
    return slot;
}
#define FIELD_SLOT(VM_, T_, N_, OUT_) do {                                     \
        const tv_cache *c_ = CACHE_AT((VM_)->field_cache, T_, N_);             \
        (OUT_) = c_->type == (T_) && c_->name == (N_) ? (int)c_->value          \
                                                      : field_slot_fill(VM_, T_, N_); \
    } while (0)
#else
#define FIELD_SLOT(VM_, T_, N_, OUT_) ((OUT_) = field_slot(VM_, T_, N_))
#endif

static int no_field(tilvm *vm, uint16_t name_const, const tv_val *obj) {
    const uint8_t *p = const_ptr(vm, name_const);
    uint16_t len = RD16(p), pos = 0;
    const char *pre = "no field '", *mid = "' on ", *tn = type_name(vm, obj);
    uint32_t tl = (uint32_t)strlen(tn);
    if (obj->t == TV_STRUCT) {
        tv_val nv;
        nv.t = TV_STRK;
        nv.u.k = type_name_const(vm, OBJ(vm, obj->u.r)->aux);
        tn = str_data(vm, &nv, &tl);
    }
    while (*pre) vm->errbuf[pos++] = *pre++;
    if (len > 50) len = 50;
    memcpy(vm->errbuf + pos, p + 2, len);
    pos = (uint16_t)(pos + len);
    while (*mid) vm->errbuf[pos++] = *mid++;
    if (tl > 50) tl = 50;
    memcpy(vm->errbuf + pos, tn, tl);
    pos = (uint16_t)(pos + tl);
    vm->errbuf[pos] = 0;
    return fail(vm, vm->errbuf);
}

static int new_struct(tilvm *vm, uint16_t t, uint8_t n) {
    tv_ref r = alloc_obj(vm, OBJ_STRUCT, (uint32_t)offsetof(tv_structobj, v) + n * (uint32_t)sizeof(tv_val));
    uint8_t i;
    if (!r) return 0;
    OBJ(vm, r)->aux = t;
    for (i = 0; i < n; i++) STRUCTOBJ(vm, r)->v[i] = STK[SP - n + i];
    SP = (uint16_t)(SP - n);
    set_ref(&STK[SP], TV_STRUCT, r);
    SP++;
    return 1;
}

/* ─── casts ─────────────────────────────────────────────────────────────── */
static int parse_int(const char *p, uint32_t n, tv_int *out_i) {
    uint32_t i = 0;
    int neg = 0, any = 0;
    tv_uint v = 0;
    while (i < n && (p[i] == ' ' || p[i] == '\t' || p[i] == '\n' || p[i] == '\r')) i++;
    while (n > i && (p[n - 1] == ' ' || p[n - 1] == '\t' || p[n - 1] == '\n' || p[n - 1] == '\r')) n--;
    if (i < n && (p[i] == '-' || p[i] == '+')) { neg = p[i] == '-'; i++; }
    for (; i < n; i++) {
        if (p[i] == '_') continue;
        if (p[i] < '0' || p[i] > '9') return 0;
        v = v * 10 + (tv_uint)(p[i] - '0');
        any = 1;
    }
    if (!any) return 0;
    *out_i = neg ? (tv_int)((tv_uint)0 - v) : (tv_int)v;
    return 1;
}

#if TILVM_FLOAT
static int parse_float(const char *p, uint32_t n, tv_float *out_f) {
    char buf[64];
    if (n == 0 || n >= sizeof(buf)) return 0;
    memcpy(buf, p, n);
    buf[n] = 0;
#ifndef TILVM_NO_STDIO
    {
        char *end;
        double d = strtod(buf, &end);
        while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') end++;
        if (*end || end == buf) return 0;
        *out_f = (tv_float)d;
        return 1;
    }
#else
    {
        uint32_t i = 0;
        tv_float v = 0, scale = 1;
        int neg = 0, any = 0, frac = 0;
        while (buf[i] == ' ') i++;
        if (buf[i] == '-' || buf[i] == '+') { neg = buf[i] == '-'; i++; }
        for (; buf[i]; i++) {
            if (buf[i] == '.') { if (frac) return 0; frac = 1; continue; }
            if (buf[i] < '0' || buf[i] > '9') break;
            any = 1;
            if (frac) { scale /= 10; v += (tv_float)(buf[i] - '0') * scale; }
            else v = v * 10 + (tv_float)(buf[i] - '0');
        }
        while (buf[i] == ' ') i++;
        if (!any || buf[i]) return 0;
        *out_f = neg ? -v : v;
        return 1;
    }
#endif
}
#endif

static int invalid_number(tilvm *vm, uint16_t slot, const char *what) {
    uint32_t n, pos = 0, cap = sizeof(vm->errbuf) - 4;
    const char *p = str_data(vm, &STK[slot], &n);
    while (*what && pos < cap) vm->errbuf[pos++] = *what++;
    vm->errbuf[pos++] = '"';
    while (n-- && pos < cap) vm->errbuf[pos++] = *p++;
    vm->errbuf[pos++] = '"';
    vm->errbuf[pos] = 0;
    return fail(vm, vm->errbuf);
}

/* cast stack slot in place: 0 int, 1 float, 2 str, 3 bool */
static int cast_slot(tilvm *vm, uint16_t slot, int kind) {
    tv_val *v = &STK[slot];
    if (kind == 2) return to_str_slot(vm, slot);
    if (kind == 3) { set_bool(v, truthy(vm, v)); return 1; }
    if (kind == 0) {
        switch (v->t) {
        case TV_INT: return 1;
        case TV_BOOL: set_int(v, v->u.i); return 1;
        case TV_NIL: set_int(v, 0); return 1;
#if TILVM_FLOAT
        case TV_FLOAT: {
            tv_float f = v->u.f;
            if (f != f || f - f != f - f) return fail(vm, "cannot convert nan/inf to int");
#if TILVM_INT_BITS == 64 && defined(TV_HAVE_MATH)
            if (f >= (tv_float)9.2e18 || f <= (tv_float)-9.2e18) {
                double m = fmod((double)f, 18446744073709551616.0);
                if (m < 0) m += 18446744073709551616.0;
                set_int(v, (tv_int)(uint64_t)m);
                return 1;
            }
#endif
            set_int(v, (tv_int)f);
            return 1;
        }
#endif
        case TV_STR: case TV_STRK: {
            uint32_t n;
            const char *p = str_data(vm, v, &n);
            tv_int i;
            if (!parse_int(p, n, &i)) return invalid_number(vm, slot, "invalid integer: ");
            set_int(&STK[slot], i);
            return 1;
        }
        default: return fail(vm, "cannot convert value to int");
        }
    }
#if TILVM_FLOAT
    switch (v->t) {
    case TV_FLOAT: return 1;
    case TV_INT: case TV_BOOL: set_float(v, (tv_float)v->u.i); return 1;
    case TV_NIL: set_float(v, 0); return 1;
    case TV_STR: case TV_STRK: {
        uint32_t n;
        const char *p = str_data(vm, v, &n);
        tv_float f;
        if (!parse_float(p, n, &f)) return invalid_number(vm, slot, "invalid number: ");
        set_float(&STK[slot], f);
        return 1;
    }
    default: return fail(vm, "cannot convert value to float");
    }
#else
    return fail(vm, "this TIL VM was built without floating point");
#endif
}

#ifndef TILVM_LEAN
/* ─── f-string format specs ─────────────────────────────────────────────── */
static int format_slot(tilvm *vm, uint16_t vslot, uint16_t spec_slot) {
    /* [<>^][0][width][.prec][fdxXobe%] - same rules as til_format_spec() */
    char body[96], *b = body;
    uint32_t slen, blen, pad, i, width = 0;
    const char *sp = str_data(vm, &STK[spec_slot], &slen);
    char align = 0, kind = 0;
    int zero = 0, prec = -1, ok = 1, explicit_align;
    tv_val *v = &STK[vslot];
    uint32_t k = 0;
    if (k < slen && (sp[k] == '<' || sp[k] == '>' || sp[k] == '^')) align = sp[k++];
    if (k < slen && sp[k] == '0') { zero = 1; k++; }
    while (k < slen && sp[k] >= '0' && sp[k] <= '9') width = width * 10 + (uint32_t)(sp[k++] - '0');
    if (k < slen && sp[k] == '.') {
        prec = 0; k++;
        while (k < slen && sp[k] >= '0' && sp[k] <= '9') prec = prec * 10 + (sp[k++] - '0');
    }
    if (k < slen) kind = sp[k++];
    explicit_align = align != 0;
    if (k != slen || (kind && !strchr("fdxXobe%", kind))) ok = 0;
    if (width > 60) width = 60;
    if (prec > 30) prec = 30;
    body[0] = 0;
    if (!ok || !is_num(v)) {
        if (!to_str_slot(vm, vslot)) return 0;
        if (!ok) return 1;
        b = 0;
    } else if (kind == 'f' || kind == 'e' || kind == '%') {
#if TILVM_FLOAT
        tv_float x = as_float(v) * (kind == '%' ? 100 : 1);
#ifndef TILVM_NO_STDIO
        snprintf(body, 80, kind == 'e' ? "%.*e" : "%.*f", prec < 0 ? 6 : prec, (double)x);
#else
        if (kind == 'e') fmt_exp(body, x, prec < 0 ? 6 : prec);
        else fmt_fixed(body, x, prec < 0 ? 6 : prec);
#endif
        if (kind == '%') strcat(body, "%");
#else
        strcpy(body, fmt_int(body + 64, v->u.i));
#endif
    } else if (kind == 'd' || kind == 'x' || kind == 'X' || kind == 'o' || kind == 'b') {
        tv_int n;
        tv_uint u;
        char tmp[80], *q = tmp + 79;
        unsigned base = kind == 'd' ? 10 : kind == 'o' ? 8 : kind == 'b' ? 2 : 16;
#if TILVM_FLOAT
        n = v->t == TV_INT ? v->u.i : (tv_int)v->u.f;
#else
        n = v->u.i;
#endif
        u = n < 0 ? (tv_uint)0 - (tv_uint)n : (tv_uint)n;
        *q = 0;
        do {
            unsigned d = (unsigned)(u % base);
            *--q = (char)(d < 10 ? '0' + d : (kind == 'X' ? 'A' : 'a') + d - 10);
            u /= base;
        } while (u);
        if (n < 0) *--q = '-';
        strcpy(body, q);
    } else {
#if TILVM_FLOAT
        if (v->t == TV_FLOAT && prec >= 0) {
#ifndef TILVM_NO_STDIO
            snprintf(body, 80, "%.*g", prec, (double)v->u.f);
#else
            fmt_gp(body, v->u.f, prec == 0 ? 1 : prec);
#endif
        } else
#endif
        {
            sink s;
            s.vm = vm; s.mode = 1; s.dst = body; s.len = 0;
            emit_value(&s, v, 0, 0);
            body[s.len] = 0;
        }
    }
    {
        /* build the padded result */
        const char *src;
        uint32_t total;
        tv_ref r;
        int numeric = is_num(&STK[vslot]) && b != 0;
        if (b) { src = body; blen = (uint32_t)strlen(body); }
        else src = str_data(vm, &STK[vslot], &blen);
        total = blen < width ? width : blen;
        r = new_str(vm, total);
        if (!r) return 0;
        if (!b) src = str_data(vm, &STK[vslot], &blen);
        pad = total - blen;
        {
            char *d = STROBJ(vm, r)->data;
            if (!align) align = numeric ? '>' : '<';
            if (pad && zero && numeric && !explicit_align) {
                uint32_t sign = (src[0] == '-' || src[0] == '+') ? 1 : 0;
                memcpy(d, src, sign);
                for (i = 0; i < pad; i++) d[sign + i] = '0';
                memcpy(d + sign + pad, src + sign, blen - sign);
            } else {
                uint32_t left = align == '<' ? 0 : align == '>' ? pad : pad / 2;
                for (i = 0; i < left; i++) d[i] = ' ';
                memcpy(d + left, src, blen);
                for (i = left + blen; i < total; i++) d[i] = ' ';
            }
        }
        set_ref(&STK[vslot], TV_STR, r);
    }
    return 1;
}

#endif /* TILVM_LEAN */

/* ─── natives ───────────────────────────────────────────────────────────── */
#if TILVM_FLOAT
static tv_float f_round(tv_float x) {
    if (x != x) return x;
    return x < 0 ? -F_FLOOR(-x + (tv_float)0.5) : F_FLOOR(x + (tv_float)0.5);
}
#endif

#ifndef TILVM_LEAN
static uint32_t rng_next(tilvm *vm) {
#if TILVM_INT_BITS == 64
    uint64_t x = vm->st.rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    vm->st.rng = x;
    return (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 32);
#else
    uint32_t x = vm->st.rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    vm->st.rng = x ? x : 0x9E3779B9u;
    return vm->st.rng;
#endif
}
#endif

/* Minimum argument count of each native (the compiler also checks the maximum).
 * Bytecode from other compilers cannot make a native read past its arguments. */
static const uint8_t NATIVE_MIN_ARGS[] = {
    0, 1, 1, 1, 1, 1, 1, 2, 1, 1,      /* print len str int float abs sqrt pow sin cos */
    1, 1, 1, 1, 1, 1, 1, 1, 0, 1,      /* tan log exp floor ceil round min max input chr */
    1, 2, 1, 2, 0, 0, 1, 1, 0, 1,      /* ord range type_of format clock_ms random seed sleep_ms exit bool */
    1, 0, 1, 0,                        /* sum panic iter println */
    1, 2, 1, 0, 1, 1,                  /* read_file write_file file_exists args bytes_to_str str_bytes */
    2, 0, 3, 5, 5, 5, 4, 4, 2, 0,      /* screen_init screen_clear draw_pixel draw_line draw_rect
                                          fill_rect draw_circle draw_text get_pixel screen_show */
    0, 0, 0, 0, 0                      /* key_pressed wait_key screen_width screen_height energy_mode */
};

/* compile-time check: one entry per native */
typedef char tv_check_native_tables[(sizeof(NATIVE_MIN_ARGS) == N_NATIVES) ? 1 : -1];

#ifndef TILVM_LEAN
/* ─── files, bytes, program arguments (TIL 3.1) ────────────────────────── */
#define TV_PATH_MAX 256

/* Copy a string value into a NUL-terminated C path. */
static int c_path(tilvm *vm, const tv_val *v, char *buf, const char *who) {
    uint32_t n;
    const char *p;
    if (!is_str(v)) return failn(vm, who, "() expects a path string, got ", type_name(vm, v), 0, 0, 0);
    p = str_data(vm, v, &n);
    if (n == 0 || n >= TV_PATH_MAX || memchr(p, 0, n)) return failn(vm, who, "(): invalid path", 0, 0, 0, 0);
    memcpy(buf, p, n);
    buf[n] = 0;
    return 1;
}

/* Push a new array of n nil values. */
static int push_new_array(tilvm *vm, uint32_t n) {
    tv_ref buf, arr;
    if (n > 0xFFFF) return fail(vm, "array too large");
    if (SP + 2 > TILVM_STACK_SIZE) return fail(vm, "stack overflow");
    buf = new_vals(vm, n < 4 ? 4 : n);
    if (!buf) return 0;
    set_ref(&STK[SP], TV_CELL, buf);          /* keep the buffer rooted */
    SP++;
    arr = alloc_obj(vm, OBJ_ARRAY, sizeof(tv_arrobj));
    SP--;
    if (!arr) return 0;
    ARROBJ(vm, arr)->buf = STK[SP].u.r;
    ARROBJ(vm, arr)->len = (tv_size)n;
    set_ref(&STK[SP], TV_ARRAY, arr);
    SP++;
    return 1;
}

static int native_read_file(tilvm *vm, uint16_t base, tv_val *res) {
    char path[TV_PATH_MAX];
    long n, got;
    tv_ref r;
    if (!c_path(vm, &STK[base], path, "read_file")) return 0;
    if (!vm->host.read_file) return fail(vm, "read_file() is not available on this device");
    n = vm->host.read_file(vm->ctx, path, 0, 0);
    if (n < 0) return failn(vm, "cannot read file: ", path, 0, 0, 0, 0);
    if ((unsigned long)n > TV_STR_MAX) return failn(vm, "file too large: ", path, 0, 0, 0, 0);
    r = new_str(vm, (uint32_t)n);
    if (!r) return 0;
    got = vm->host.read_file(vm->ctx, path, STROBJ(vm, r)->data, (unsigned long)n);
    if (got != n) return failn(vm, "cannot read file: ", path, 0, 0, 0, 0);
    set_ref(res, TV_STR, r);
    return 1;
}

static int native_write_file(tilvm *vm, uint16_t base) {
    char path[TV_PATH_MAX];
    uint32_t n;
    const char *d;
    if (!c_path(vm, &STK[base], path, "write_file")) return 0;
    if (!vm->host.write_file) return fail(vm, "write_file() is not available on this device");
    if (!to_str_slot(vm, (uint16_t)(base + 1))) return 0;
    d = str_data(vm, &STK[base + 1], &n);
    if (vm->host.write_file(vm->ctx, path, d, (unsigned long)n) != 0)
        return failn(vm, "cannot write file: ", path, 0, 0, 0, 0);
    return 1;
}
#endif

/* args are STK[base .. base+argc); the result replaces them */
static int call_native(tilvm *vm, uint8_t id, uint8_t argc) {
    uint16_t base = (uint16_t)(SP - argc);
    tv_val *a = &STK[base];
    tv_val res;
    set_nil(&res);
    if (id < sizeof(NATIVE_MIN_ARGS) && argc < NATIVE_MIN_ARGS[id])
        return failn(vm, NATIVE_NAMES[id], "() is missing arguments", 0, 0, 0, 0);
    switch (id) {
    case N_PRINT: case N_PRINTLN: {
        uint8_t i;
        if (!argc) out(vm, "\n", 1);
        for (i = 0; i < argc; i++) { print_value(vm, &STK[base + i]); out(vm, "\n", 1); }
        break;
    }
    case N_ITER:
        if (a->t == TV_MAP) {
            /* keys snapshot: push keys, make array */
            uint32_t n = ARROBJ(vm, a->u.r)->len, i;
            if (SP + n + 2 > TILVM_STACK_SIZE) return fail(vm, "stack overflow");
            for (i = 0; i < n; i++) STK[SP++] = VALS(vm, ARROBJ(vm, STK[base].u.r)->buf)[2 * i];
            if (!make_array(vm, OBJ_ARRAY, n)) return 0;
            STK[base] = STK[SP - 1];
            SP = (uint16_t)(base + 1);
            return 1;
        }
        if (a->t != TV_ARRAY && !is_str(a)) return fail(vm, "value is not iterable");
        res = *a;
        break;
    case N_LEN: {
        uint32_t n;
        if (is_str(a)) str_data(vm, a, &n);
        else if (a->t == TV_ARRAY || a->t == TV_MAP) n = ARROBJ(vm, a->u.r)->len;
        else return failn(vm, "len() of ", type_name(vm, a), 0, 0, 0, 0);
        set_int(&res, (tv_int)n);
        break;
    }
    case N_STR: if (!cast_slot(vm, base, 2)) return 0; res = STK[base]; break;
    case N_INT: if (!cast_slot(vm, base, 0)) return 0; res = STK[base]; break;
    case N_FLOAT: if (!cast_slot(vm, base, 1)) return 0; res = STK[base]; break;
    case N_BOOL: set_bool(&res, truthy(vm, a)); break;
    case N_ABS:
        if (a->t == TV_INT) set_int(&res, a->u.i < 0 ? (tv_int)((tv_uint)0 - (tv_uint)a->u.i) : a->u.i);
#if TILVM_FLOAT
        else if (a->t == TV_FLOAT) set_float(&res, a->u.f < 0 ? -a->u.f : a->u.f);
#endif
        else return failn(vm, "abs expects a number, got ", type_name(vm, a), 0, 0, 0, 0);
        break;
#if TILVM_FLOAT
    case N_SQRT: case N_SIN: case N_COS: case N_TAN: case N_LOG: case N_EXP:
    case N_FLOOR: case N_CEIL: case N_ROUND: {
        tv_float x;
        if (!is_num(a)) return failn(vm, NATIVE_NAMES[id], " expects a number, got ", type_name(vm, a), 0, 0, 0);
        x = as_float(a);
        switch (id) {
        case N_SQRT: x = x < 0 ? (tv_float)0 / (tv_float)0 : F_SQRT(x); break;
        case N_FLOOR: x = F_FLOOR(x); break;
        case N_CEIL: x = F_CEIL(x); break;
        case N_ROUND: x = f_round(x); break;
#ifdef TV_HAVE_MATH
        case N_SIN: x = (tv_float)sin((double)x); break;
        case N_COS: x = (tv_float)cos((double)x); break;
        case N_TAN: x = (tv_float)tan((double)x); break;
        case N_LOG: x = x > 0 ? (tv_float)log((double)x) : (x == 0 ? -(tv_float)1 / (tv_float)0 : (tv_float)0 / (tv_float)0); break;
        case N_EXP: x = (tv_float)exp((double)x); break;
#else
        default: return fail(vm, "sin/cos/tan/log/exp need a build with <math.h>");
#endif
        }
        set_float(&res, x);
        break;
    }
    case N_POW:
        if (!is_num(a) || !is_num(a + 1)) return fail(vm, "pow expects numbers");
        set_float(&res, F_POW(as_float(a), as_float(a + 1)));
        break;
#endif
    case N_MIN: case N_MAX: case N_SUM: {
        /* over the arguments, or over a single array argument */
        tv_val *vals = a;
        uint32_t n = argc, i;
        int r;
        tv_val best;
        if (argc == 1 && a->t == TV_ARRAY) {
            n = ARROBJ(vm, a->u.r)->len;
            vals = VALS(vm, ARROBJ(vm, a->u.r)->buf);
        }
        if (id == N_SUM) {
            /* push 0, then add each element (arith works on stack slots) */
            if (SP + 3 > TILVM_STACK_SIZE) return fail(vm, "stack overflow");
            set_int(&STK[SP++], 0);
            for (i = 0; i < n; i++) {
                if (argc == 1 && STK[base].t == TV_ARRAY)
                    STK[SP++] = VALS(vm, ARROBJ(vm, STK[base].u.r)->buf)[i];
                else
                    STK[SP++] = STK[base + i];
                if (!is_num(&STK[SP - 1])) return fail(vm, "sum() needs numbers");
                if (!arith(vm, OP_ADD)) return 0;
            }
            res = STK[SP - 1];
            break;
        }
        if (!n) return fail(vm, id == N_MIN ? "min() of empty sequence" : "max() of empty sequence");
        best = vals[0];
        for (i = 1; i < n; i++) {
            if (!compare(vm, id == N_MIN ? OP_LT : OP_GT, &vals[i], &best, &r)) return 0;
            if (r) best = vals[i];
        }
        res = best;
        break;
    }
#ifndef TILVM_LEAN
    case N_INPUT: {
        char buf[256];
        int n = -1;
        if (argc) print_value(vm, a);
        if (vm->read_line) n = vm->read_line(vm->ctx, buf, sizeof(buf));
        if (!make_str(vm, &res, buf, n < 0 ? 0 : (uint32_t)n)) return 0;
        break;
    }
    case N_CHR: {
        char buf[4];
        uint32_t c, n;
        if (a->t != TV_INT || a->u.i < 0) return fail(vm, "chr() argument out of range");
#if TILVM_INT_BITS > 16
        if (a->u.i > 0x10FFFF) return fail(vm, "chr() argument out of range");
#endif
        c = (uint32_t)a->u.i;
        if (c < 0x80) { buf[0] = (char)c; n = 1; }
        else if (c < 0x800) { buf[0] = (char)(0xC0 | (c >> 6)); buf[1] = (char)(0x80 | (c & 0x3F)); n = 2; }
        else if (c < 0x10000) { buf[0] = (char)(0xE0 | (c >> 12)); buf[1] = (char)(0x80 | ((c >> 6) & 0x3F));
                                buf[2] = (char)(0x80 | (c & 0x3F)); n = 3; }
        else { buf[0] = (char)(0xF0 | (c >> 18)); buf[1] = (char)(0x80 | ((c >> 12) & 0x3F));
               buf[2] = (char)(0x80 | ((c >> 6) & 0x3F)); buf[3] = (char)(0x80 | (c & 0x3F)); n = 4; }
        if (!make_str(vm, &res, buf, n)) return 0;
        break;
    }
    case N_ORD: {
        uint32_t n, c;
        const uint8_t *p;
        if (!is_str(a)) return fail(vm, "ord() expects a non-empty string");
        p = (const uint8_t *)str_data(vm, a, &n);
        if (!n) return fail(vm, "ord() expects a non-empty string");
        c = p[0];
        if (c >= 0xF0 && n >= 4) c = ((c & 7u) << 18) | ((p[1] & 63u) << 12) | ((p[2] & 63u) << 6) | (p[3] & 63u);
        else if (c >= 0xE0 && n >= 3) c = ((c & 15u) << 12) | ((p[1] & 63u) << 6) | (p[2] & 63u);
        else if (c >= 0xC0 && n >= 2) c = ((c & 31u) << 6) | (p[1] & 63u);
        set_int(&res, (tv_int)c);
        break;
    }
#endif
    case N_RANGE: {
        tv_int lo, hi, i;
        uint32_t n;
        if (a->t != TV_INT || a[1].t != TV_INT) return fail(vm, "range bounds must be integers");
        lo = a->u.i;
        hi = a[1].u.i;
        if (argc > 2 && truthy(vm, &a[2])) hi = iwrap_add(hi, 1);
        n = hi > lo ? (uint32_t)(hi - lo) : 0;
        if (n > 0xFFFF || base + n + 2 > TILVM_STACK_SIZE) return fail(vm, "range too large");
        SP = base;
        for (i = 0; i < (tv_int)n; i++) set_int(&STK[SP++], lo + i);
        return make_array(vm, OBJ_ARRAY, n);
    }
#ifndef TILVM_LEAN
    case N_TYPE_OF: {
        const char *s = type_name(vm, a);
        if (a->t == TV_STRUCT) {
            res.t = TV_STRK;
            res.u.k = type_name_const(vm, OBJ(vm, a->u.r)->aux);
        } else if (!make_str(vm, &res, s, (uint32_t)strlen(s))) return 0;
        break;
    }
#endif
    case N_FORMAT:
#ifndef TILVM_LEAN
        if (!format_slot(vm, base, (uint16_t)(base + 1))) return 0;
#else
        if (!to_str_slot(vm, base)) return 0;       /* lean build: specs ignored */
#endif
        res = STK[base];
        break;
#ifndef TILVM_LEAN
    case N_CLOCK_MS: set_int(&res, (tv_int)now_ms(vm)); break;
    case N_RANDOM:
        if (argc >= 2) {
            tv_int lo = a->u.i, hi = a[1].u.i;
            tv_uint span;
            if (a->t != TV_INT || a[1].t != TV_INT || hi < lo) return fail(vm, "random(lo, hi) needs lo <= hi");
            span = (tv_uint)(hi - lo) + 1;
            set_int(&res, span ? (tv_int)((tv_uint)lo + (tv_uint)rng_next(vm) % span) : (tv_int)rng_next(vm));
        } else {
#if TILVM_FLOAT
            set_float(&res, (tv_float)rng_next(vm) / (tv_float)4294967296.0);
#else
            set_int(&res, (tv_int)(rng_next(vm) & 0x7FFF));
#endif
        }
        break;
    case N_SEED:
        if (a->t != TV_INT) return fail(vm, "seed() needs an integer");
        vm->st.rng = a->u.i ? (uint32_t)a->u.i : 0x9E3779B9u;
        break;
    case N_SLEEP_MS: {
        uint32_t until;
        if (a->t != TV_INT) return fail(vm, "sleep_ms() needs an integer");
        until = now_ms(vm) + (uint32_t)(a->u.i > 0 ? a->u.i : 0);
        if (vm->clock_ms) while ((int32_t)(now_ms(vm) - until) < 0) { }
        break;
    }
#endif
#ifndef TILVM_LEAN
    case N_READ_FILE:
        if (!native_read_file(vm, base, &res)) return 0;
        break;
    case N_WRITE_FILE:
        if (!native_write_file(vm, base)) return 0;
        break;
    case N_FILE_EXISTS: {
        char path[TV_PATH_MAX];
        if (!c_path(vm, a, path, "file_exists")) return 0;
        set_bool(&res, vm->host.read_file && vm->host.read_file(vm->ctx, path, 0, 0) >= 0);
        break;
    }
    case N_ARGS: {
        int i, n = vm->host.argv ? vm->host.argc : 0;
        if (n < 0) n = 0;
        if ((uint32_t)SP + (uint32_t)n + 2 > TILVM_STACK_SIZE) return fail(vm, "stack overflow");
        SP = base;
        for (i = 0; i < n; i++) {
            const char *s = vm->host.argv[i] ? vm->host.argv[i] : "";
            if (!make_str(vm, &STK[SP], s, (uint32_t)strlen(s))) return 0;
            SP++;
        }
        return make_array(vm, OBJ_ARRAY, (uint32_t)n);
    }
    case N_BYTES_TO_STR: {
        uint32_t n, i;
        tv_ref r;
        if (a->t != TV_ARRAY) return fail(vm, "bytes_to_str() expects an array of ints 0..255");
        n = ARROBJ(vm, a->u.r)->len;
        for (i = 0; i < n; i++) {
            const tv_val *e = &VALS(vm, ARROBJ(vm, a->u.r)->buf)[i];
            if (e->t != TV_INT || e->u.i < 0 || e->u.i > 255)
                return fail(vm, "bytes_to_str() expects an array of ints 0..255");
        }
        r = new_str(vm, n);                       /* may move the array: re-read it */
        if (!r) return 0;
        for (i = 0; i < n; i++)
            STROBJ(vm, r)->data[i] = (char)(unsigned char)VALS(vm, ARROBJ(vm, STK[base].u.r)->buf)[i].u.i;
        set_ref(&res, TV_STR, r);
        break;
    }
    case N_STR_BYTES: {
        uint32_t n, i;
        const uint8_t *p;
        tv_ref buf;
        if (!is_str(a)) return failn(vm, "str_bytes() expects a string, got ", type_name(vm, a), 0, 0, 0, 0);
        str_data(vm, a, &n);
        if (!push_new_array(vm, n)) return 0;   /* GC may move the string: re-read it */
        p = (const uint8_t *)str_data(vm, &STK[base], &n);
        buf = ARROBJ(vm, STK[SP - 1].u.r)->buf;
        for (i = 0; i < n; i++) set_int(&VALS(vm, buf)[i], (tv_int)p[i]);
        STK[base] = STK[SP - 1];
        SP = (uint16_t)(base + 1);
        return 1;
    }
#endif
    case N_EXIT:
        vm->st.exit_code = (uint8_t)(argc && a->t == TV_INT ? a->u.i : 0);
        vm->st.halted = 2;
        break;
    case N_PANIC:
        if (argc) {
            if (!to_str_slot(vm, base)) return 0;
            vm->err_val = STK[base];
            vm->err_msg = "";
        } else {
            set_nil(&vm->err_val);
            vm->err_msg = "panic";
        }
        return 0;
    default:
        return failn(vm, id < N_NATIVES ? NATIVE_NAMES[id] : "?", "() is not available in this VM build",
                     0, 0, 0, 0);
    }
    STK[base] = res;
    SP = (uint16_t)(base + 1);
    return 1;
}

/* ─── builtin methods ───────────────────────────────────────────────────── */
static int find_bytes(const char *h, uint32_t hl, const char *n, uint32_t nl, uint32_t from) {
    uint32_t i;
    if (nl == 0) return (int)from;
#ifndef TILVM_LEAN
    /* memchr jumps to the next possible start (same result as the plain loop) */
    for (i = from; i + nl <= hl; i++) {
        const char *p = (const char *)memchr(h + i, (unsigned char)n[0], hl - nl + 1 - i);
        uint32_t j;
        if (!p) return -1;
        i = (uint32_t)(p - h);
        for (j = 1; j < nl && p[j] == n[j]; j++) { }
        if (j == nl) return (int)i;
    }
#else
    for (i = from; i + nl <= hl; i++)
        if (memcmp(h + i, n, nl) == 0) return (int)i;
#endif
    return -1;
}

static int str_method(tilvm *vm, uint8_t mid, uint16_t base, uint8_t argc, tv_val *res) {
    uint32_t n, m = 0;
    const char *s = str_data(vm, &STK[base], &n);
    tv_val *a1 = &STK[base + 1];
    switch (mid) {
    case M_LEN: set_int(res, (tv_int)n); return 1;
    case M_IS_EMPTY: set_bool(res, n == 0); return 1;
    case M_CONTAINS: case M_STARTS_WITH: case M_ENDS_WITH: case M_FIND: case M_INDEX_OF:
    case M_COUNT: case M_EQ: {
        const char *t;
        if (!argc || !is_str(a1)) return failn(vm, "str.", METHOD_NAMES[mid], " expects a string argument", 0, 0, 0);
        t = str_data(vm, a1, &m);
        if (mid == M_CONTAINS) set_bool(res, find_bytes(s, n, t, m, 0) >= 0);
        else if (mid == M_STARTS_WITH) set_bool(res, m <= n && memcmp(s, t, m) == 0);
        else if (mid == M_ENDS_WITH) set_bool(res, m <= n && memcmp(s + n - m, t, m) == 0);
        else if (mid == M_EQ) set_bool(res, m == n && memcmp(s, t, m) == 0);
        else if (mid == M_COUNT) {
            tv_int c = 0;
            int at = 0;
            if (m) while ((at = find_bytes(s, n, t, m, (uint32_t)at)) >= 0) { c++; at += (int)m; }
            set_int(res, c);
        } else set_int(res, find_bytes(s, n, t, m, 0));
        return 1;
    }
    case M_GET:
        if (!argc || a1->t != TV_INT) return fail(vm, "index must be an integer");
        if (a1->u.i < 0 || (tv_uint)a1->u.i >= n) return bounds_error(vm, a1->u.i, n);
        set_int(res, (tv_int)(uint8_t)s[a1->u.i]);
        return 1;
#ifndef TILVM_LEAN
    case M_TRIM: case M_REPEAT:
#endif
    case M_TO_UPPER: case M_TO_LOWER: case M_SLICE: {
        uint32_t lo = 0, hi = n, i, times = 1;
        tv_ref r;
        if (mid == M_TRIM) {
            while (lo < hi && (s[lo] == ' ' || s[lo] == '\t' || s[lo] == '\r' || s[lo] == '\n')) lo++;
            while (hi > lo && (s[hi - 1] == ' ' || s[hi - 1] == '\t' || s[hi - 1] == '\r' || s[hi - 1] == '\n')) hi--;
        } else if (mid == M_SLICE) {
            tv_int x = argc && a1->t == TV_INT ? a1->u.i : 0;
            tv_int y = argc > 1 && STK[base + 2].t == TV_INT ? STK[base + 2].u.i : (tv_int)n;
            if (x < 0) x = 0;
            if (y > (tv_int)n) y = (tv_int)n;
            if (x > (tv_int)n) x = (tv_int)n;
            if (y < 0) y = 0;
            lo = (uint32_t)x;
            hi = (uint32_t)(y > x ? y : x);
        } else if (mid == M_REPEAT) {
            if (!argc || a1->t != TV_INT) return fail(vm, "repeat() needs an integer");
            times = a1->u.i > 0 ? (uint32_t)a1->u.i : 0;
            if (times && n > TV_STR_MAX / times) return fail(vm, "string too long");
        }
        r = new_str(vm, (hi - lo) * times);
        if (!r) return 0;
        s = str_data(vm, &STK[base], &n);
        for (i = 0; i < times; i++) memcpy(STROBJ(vm, r)->data + i * (hi - lo), s + lo, hi - lo);
        if (mid == M_TO_UPPER || mid == M_TO_LOWER) {
            char *d = STROBJ(vm, r)->data;
            for (i = 0; i < hi - lo; i++) {
                if (mid == M_TO_UPPER && d[i] >= 'a' && d[i] <= 'z') d[i] = (char)(d[i] - 32);
                if (mid == M_TO_LOWER && d[i] >= 'A' && d[i] <= 'Z') d[i] = (char)(d[i] + 32);
            }
        }
        set_ref(res, TV_STR, r);
        return 1;
    }
#ifndef TILVM_LEAN
    case M_REPLACE: {
        const char *t, *u;
        uint32_t ul, count = 0, i, w;
        int at;
        tv_ref r;
        if (argc < 2 || !is_str(a1)) return fail(vm, "replace() needs two strings");
        if (!to_str_slot(vm, (uint16_t)(base + 2))) return 0;
        s = str_data(vm, &STK[base], &n);
        t = str_data(vm, &STK[base + 1], &m);
        if (!m) { *res = STK[base]; return 1; }
        str_data(vm, &STK[base + 2], &ul);
        at = 0;
        while ((at = find_bytes(s, n, t, m, (uint32_t)at)) >= 0) { count++; at += (int)m; }
        if (ul > m && (uint32_t)count > (TV_STR_MAX - n) / (ul - m)) return fail(vm, "string too long");
        r = new_str(vm, n + count * ul - count * m);
        if (!r) return 0;
        s = str_data(vm, &STK[base], &n);
        t = str_data(vm, &STK[base + 1], &m);
        u = str_data(vm, &STK[base + 2], &ul);
        for (i = 0, w = 0; i < n;) {
            if (i + m <= n && memcmp(s + i, t, m) == 0) {
                memcpy(STROBJ(vm, r)->data + w, u, ul); w += ul; i += m;
            } else STROBJ(vm, r)->data[w++] = s[i++];
        }
        set_ref(res, TV_STR, r);
        return 1;
    }
    case M_SPLIT: {
        /* split(sep) / split() on whitespace; pieces are pushed then gathered */
        uint32_t start = 0, i, pieces = 0;
        int ws = !argc;
        const char *t = 0;
        if (argc && !is_str(a1)) return fail(vm, "split() needs a string separator");
        if (argc) { t = str_data(vm, a1, &m); if (!m) ws = 2; }
        i = 0;
        for (;;) {
            uint32_t pend, next;
            int at;
            s = str_data(vm, &STK[base], &n);
            if (argc) t = str_data(vm, &STK[base + 1], &m);
            if (ws == 1) {
                while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
                if (i >= n) break;
                start = i;
                while (i < n && !(s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
                pend = i;
                next = i;
            } else if (ws == 2) {
                if (i >= n) break;
                start = i; pend = i + 1; next = i + 1;
            } else {
                at = find_bytes(s, n, t, m, start);
                pend = at < 0 ? n : (uint32_t)at;
                next = at < 0 ? n + 1 : (uint32_t)at + m;
            }
            if (SP + 3 > TILVM_STACK_SIZE) return fail(vm, "stack overflow");
            {
                tv_ref r = new_str(vm, pend - start);
                if (!r) return 0;
                s = str_data(vm, &STK[base], &n);
                memcpy(STROBJ(vm, r)->data, s + start, pend - start);
                set_ref(&STK[SP++], TV_STR, r);
                pieces++;
            }
            if (ws == 0) { if (next > n) break; start = next; }
            else i = next;
        }
        if (!make_array(vm, OBJ_ARRAY, pieces)) return 0;
        *res = STK[--SP];
        return 1;
    }
#endif
    case M_TO_INT: if (!cast_slot(vm, base, 0)) return 0; *res = STK[base]; return 1;
#ifndef TILVM_LEAN
    case M_TO_FLOAT: if (!cast_slot(vm, base, 1)) return 0; *res = STK[base]; return 1;
#endif
    default: return failn(vm, "no method '", mid < N_METHODS ? METHOD_NAMES[mid] : "?", "' on str", 0, 0, 0);
    }
}

#ifndef TILVM_LEAN
static void sort_array(tilvm *vm, tv_ref arr, int *err) {
    /* stable insertion sort (small, no recursion, no malloc) */
    tv_arrobj *a = ARROBJ(vm, arr);
    uint32_t n = a->len, i, j;
    tv_val *v = VALS(vm, a->buf), key;
    int lt = 0;
    for (i = 0; i < n; i++) {
        if (!(is_num(&v[i]) && is_num(&v[0])) && !(is_str(&v[i]) && is_str(&v[0]))) {
            *err = 1;
            return;
        }
    }
    for (i = 1; i < n; i++) {
        key = v[i];
        j = i;
        while (j > 0) {
            compare(vm, OP_LT, &key, &v[j - 1], &lt);
            if (!lt) break;
            v[j] = v[j - 1];
            j--;
        }
        v[j] = key;
    }
}

#endif

static int array_method(tilvm *vm, uint8_t mid, uint16_t base, uint8_t argc, tv_val *res) {
    tv_arrobj *a = ARROBJ(vm, STK[base].u.r);
    uint32_t n = a->len, i;
    tv_val *v = VALS(vm, a->buf), *a1 = &STK[base + 1];
    switch (mid) {
    case M_LEN: set_int(res, (tv_int)n); return 1;
    case M_IS_EMPTY: set_bool(res, n == 0); return 1;
    case M_PUSH:
        if (!argc) return fail(vm, "push() needs a value");
        return array_push(vm, base, (uint16_t)(base + 1));
    case M_POP:
        if (!n) return fail(vm, "pop() from empty array");
        *res = v[n - 1];
        set_nil(&v[n - 1]);
        a->len = (tv_size)(n - 1);
        return 1;
#ifndef TILVM_LEAN
    case M_REMOVE: case M_INSERT:
#endif
    case M_GET: case M_SET: {
        tv_int k;
        if (!argc || a1->t != TV_INT) return fail(vm, "index must be an integer");
        k = a1->u.i;
        if (mid == M_INSERT) {
            if (k < 0 || (tv_uint)k > n) return failf(vm, "insert index out of bounds: ", k, 0, 0, "");
            if (argc < 2) return fail(vm, "insert() needs an index and a value");
            if (!reserve(vm, base, n + 1)) return 0;
            a = ARROBJ(vm, STK[base].u.r);
            v = VALS(vm, a->buf);
            for (i = n; i > (uint32_t)k; i--) v[i] = v[i - 1];
            v[k] = STK[base + 2];
            a->len = (tv_size)(n + 1);
            return 1;
        }
        if (k < 0 || (tv_uint)k >= n) {
            if (mid == M_REMOVE) return failf(vm, "remove index out of bounds: ", k, 0, 0, "");
            return bounds_error(vm, k, n);
        }
        if (mid == M_GET) { *res = v[k]; return 1; }
        if (mid == M_SET) { if (argc < 2) return fail(vm, "set() needs an index and a value"); v[k] = STK[base + 2]; return 1; }
        *res = v[k];
        for (i = (uint32_t)k; i + 1 < n; i++) v[i] = v[i + 1];
        set_nil(&v[n - 1]);
        a->len = (tv_size)(n - 1);
        return 1;
    }
    case M_CONTAINS: case M_INDEX_OF: case M_COUNT: {
        tv_int c = 0;
        if (!argc) return fail(vm, "method needs a value");
        for (i = 0; i < n; i++) {
            if (equal(vm, &VALS(vm, ARROBJ(vm, STK[base].u.r)->buf)[i], &STK[base + 1], 0)) {
                if (mid == M_CONTAINS) { set_bool(res, 1); return 1; }
                if (mid == M_INDEX_OF) { set_int(res, (tv_int)i); return 1; }
                c++;
            }
        }
        if (mid == M_CONTAINS) set_bool(res, 0);
        else if (mid == M_INDEX_OF) set_int(res, -1);
        else set_int(res, c);
        return 1;
    }
#ifndef TILVM_LEAN
    case M_REVERSE:
        for (i = 0; i < n / 2; i++) { tv_val t; t = v[i]; v[i] = v[n - 1 - i]; v[n - 1 - i] = t; }
        return 1;
    case M_SORT: {
        int err = 0;
        if (n) sort_array(vm, STK[base].u.r, &err);
        if (err) return fail(vm, "sort() needs an array of numbers or of strings");
        return 1;
    }
#endif
    case M_FIRST: if (n) *res = v[0]; return 1;
    case M_LAST: if (n) *res = v[n - 1]; return 1;
    case M_CLEAR:
        for (i = 0; i < n; i++) set_nil(&v[i]);
        a->len = 0;
        return 1;
#ifndef TILVM_LEAN
    case M_SUM: case M_MIN: case M_MAX:
        SP = (uint16_t)(base + 1);
        if (!call_native(vm, mid == M_SUM ? N_SUM : mid == M_MIN ? N_MIN : N_MAX, 1)) return 0;
        *res = STK[base];
        return 1;
    case M_COPY: case M_SLICE: {
        tv_int lo = 0, hi = (tv_int)n;
        if (mid == M_SLICE) {
            lo = argc && a1->t == TV_INT ? a1->u.i : 0;
            hi = argc > 1 && STK[base + 2].t == TV_INT ? STK[base + 2].u.i : (tv_int)n;
            if (lo < 0) lo = 0;
            if (hi > (tv_int)n) hi = (tv_int)n;
            if (lo > hi) lo = hi;
        }
        if (SP + (uint32_t)(hi - lo) + 2 > TILVM_STACK_SIZE) return fail(vm, "stack overflow");
        for (i = (uint32_t)lo; i < (uint32_t)hi; i++) STK[SP++] = VALS(vm, ARROBJ(vm, STK[base].u.r)->buf)[i];
        if (!make_array(vm, OBJ_ARRAY, (uint32_t)(hi - lo))) return 0;
        *res = STK[--SP];
        return 1;
    }
    case M_JOIN: {
        /* convert every element to a string on the stack, then concatenate */
        uint32_t total = 0, sl = 0, k;
        tv_ref r;
        if (SP + n + 3 > TILVM_STACK_SIZE) return fail(vm, "stack overflow");
        if (argc) { if (!to_str_slot(vm, (uint16_t)(base + 1))) return 0; str_data(vm, &STK[base + 1], &sl); }
        for (i = 0; i < n; i++) {
            STK[SP++] = VALS(vm, ARROBJ(vm, STK[base].u.r)->buf)[i];
            if (!to_str_slot(vm, (uint16_t)(SP - 1))) return 0;
            { uint32_t l; str_data(vm, &STK[SP - 1], &l); total += l; }
        }
        if (n) total += sl * (n - 1);
        if (total > TV_STR_MAX) return fail(vm, "string too long");
        r = new_str(vm, total);
        if (!r) return 0;
        for (i = 0, k = 0; i < n; i++) {
            uint32_t l;
            const char *p;
            if (i && sl) { p = str_data(vm, &STK[base + 1], &l); memcpy(STROBJ(vm, r)->data + k, p, l); k += l; }
            p = str_data(vm, &STK[SP - n + i], &l);
            memcpy(STROBJ(vm, r)->data + k, p, l);
            k += l;
        }
        SP = (uint16_t)(SP - n);
        set_ref(res, TV_STR, r);
        return 1;
    }
#endif
    default: return failn(vm, "no method '", mid < N_METHODS ? METHOD_NAMES[mid] : "?", "' on array", 0, 0, 0);
    }
}

static int map_method(tilvm *vm, uint8_t mid, uint16_t base, uint8_t argc, tv_val *res) {
    tv_ref m = STK[base].u.r;
    uint32_t n = ARROBJ(vm, m)->len, i;
    int k;
    switch (mid) {
    case M_LEN: set_int(res, (tv_int)n); return 1;
    case M_IS_EMPTY: set_bool(res, n == 0); return 1;
    case M_SET:
        if (argc < 2) return fail(vm, "set() needs a key and a value");
        return map_set(vm, base, (uint16_t)(base + 1), (uint16_t)(base + 2));
    case M_GET: case M_GET_OR: case M_HAS: case M_CONTAINS: case M_DELETE: case M_REMOVE:
        if (!argc) return fail(vm, "map method needs a key");
        map_key(&STK[base + 1]);
        k = map_find(vm, m, &STK[base + 1]);
        if (mid == M_HAS || mid == M_CONTAINS) { set_bool(res, k >= 0); return 1; }
        if (mid == M_DELETE || mid == M_REMOVE) { if (k >= 0) map_delete(vm, m, k); return 1; }
        if (k >= 0) { *res = VALS(vm, ARROBJ(vm, m)->buf)[2 * k + 1]; return 1; }
        if (mid == M_GET_OR) { *res = STK[base + 2]; return 1; }
        return key_not_found(vm, (uint16_t)(base + 1));
#ifndef TILVM_LEAN
    case M_VALUES: case M_COPY:
#endif
    case M_KEYS: {
        uint32_t per = mid == M_COPY ? 2 : 1;
        if (SP + per * n + 2 > TILVM_STACK_SIZE) return fail(vm, "stack overflow");
        for (i = 0; i < n; i++) {
            tv_val *v = VALS(vm, ARROBJ(vm, STK[base].u.r)->buf);
            if (mid != M_VALUES) STK[SP++] = v[2 * i];
            if (mid != M_KEYS) STK[SP++] = v[2 * i + 1];
        }
        if (!make_array(vm, mid == M_COPY ? OBJ_MAP : OBJ_ARRAY, n)) return 0;
        *res = STK[--SP];
        return 1;
    }
#ifndef TILVM_LEAN
    case M_CLEAR: {
        tv_val *v = VALS(vm, ARROBJ(vm, m)->buf);
        for (i = 0; i < 2 * n; i++) set_nil(&v[i]);
        ARROBJ(vm, m)->len = 0;
        return 1;
    }
#endif
    default: return failn(vm, "no method '", mid < N_METHODS ? METHOD_NAMES[mid] : "?", "' on map", 0, 0, 0);
    }
}

static int variant_method(tilvm *vm, uint8_t mid, uint16_t base, uint8_t argc, tv_val *res) {
    tv_val *r = &STK[base];
    int kind = r->t == TV_NIL ? -1 : (int)OBJ(vm, r->u.r)->aux;
    int ok = kind == TYPE_SOME || kind == TYPE_OK;
    switch (mid) {
    case M_IS_SOME: case M_IS_OK: set_bool(res, ok); return 1;
    case M_IS_NONE: case M_IS_ERR: set_bool(res, !ok); return 1;
    case M_UNWRAP:
        if (ok) { *res = STRUCTOBJ(vm, r->u.r)->v[0]; return 1; }
        if (kind == -1) return fail(vm, "unwrap() called on None");
        STK[base] = STRUCTOBJ(vm, r->u.r)->v[0];
        if (!to_str_slot(vm, base)) return 0;
        vm->err_val = STK[base];
        vm->err_msg = "unwrap() called on Err: ";
        return 0;
    case M_EXPECT:
        if (ok) { *res = STRUCTOBJ(vm, r->u.r)->v[0]; return 1; }
        if (!argc) return fail(vm, "expect() failed");
        if (!to_str_slot(vm, (uint16_t)(base + 1))) return 0;
        vm->err_val = STK[base + 1];
        vm->err_msg = "";
        return 0;
    case M_UNWRAP_OR:
        if (ok) *res = STRUCTOBJ(vm, r->u.r)->v[0];
        else if (argc) *res = STK[base + 1];
        return 1;
    case M_ERROR:
        if (kind == TYPE_ERR) *res = STRUCTOBJ(vm, r->u.r)->v[0];
        return 1;
    default: return failn(vm, "no method '", mid < N_METHODS ? METHOD_NAMES[mid] : "?", "' on ",
                          r->t == TV_NIL ? "nil" : kind == TYPE_SOME ? "Some" : kind == TYPE_OK ? "Ok" : "Err", 0, 0);
    }
}

/* [.., recv, args] -> [.., result] */
static int call_method(tilvm *vm, uint8_t mid, uint8_t argc) {
    uint16_t base = (uint16_t)(SP - argc - 1);
    tv_val res, *r = &STK[base];
    int ok;
    set_nil(&res);
    if (mid == M_TO_STRING) {
        if (!to_str_slot(vm, base)) return 0;
        SP = (uint16_t)(base + 1);
        return 1;
    }
    if (is_str(r)) ok = str_method(vm, mid, base, argc, &res);
    else if (r->t == TV_ARRAY) ok = array_method(vm, mid, base, argc, &res);
    else if (r->t == TV_MAP) ok = map_method(vm, mid, base, argc, &res);
    else if (r->t == TV_NIL || (r->t == TV_STRUCT && OBJ(vm, r->u.r)->aux <= TYPE_ERR))
        ok = variant_method(vm, mid, base, argc, &res);
    else ok = failn(vm, "no method '", mid < N_METHODS ? METHOD_NAMES[mid] : "?", "' on ", type_name(vm, r), 0, 0);
    if (!ok) return 0;
    STK[base] = res;
    SP = (uint16_t)(base + 1);
    return 1;
}

/* ─── loading ───────────────────────────────────────────────────────────── */
static uint32_t fnv1a(const uint8_t *p, uint32_t n) {
    uint32_t h = 2166136261u, i;
    for (i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static void reset_state(tilvm *vm) {
    tilvm_state *st = &vm->st;
    const uint8_t *f;
    uint16_t i;
    st->magic = 0x54494C53u;   /* "TILS" */
    st->sp = 0;
    st->n_frames = 0;
    st->n_try = 0;
    st->n_deadlines = 0;
    st->heap_top = (tv_size)ALIGN_UP(1);
    st->halted = 0;
    st->exit_code = 0;
    st->failed = 0;
    st->steps = 0;
    st->gc_count = 0;
    st->tick = 0;
    st->rng = 0x9E3779B9u;
#if TILVM_INT_BITS == 64
    st->rng = 0x9E3779B97F4A7C15ULL;
#endif
    for (i = 0; i < vm->n_globals; i++) set_nil(&st->stack[st->sp++]);
    f = func_rec(vm, vm->entry);
    st->frames[0].func = vm->entry;
    st->frames[0].base = st->sp;
    st->frames[0].pc = vm->code_off + FN_OFFSET(f);
    st->frames[0].ret_pc = 0;
    st->frames[0].closure = 0;
    st->frames[0].extra = 0;
    st->n_frames = 1;
    for (i = 0; i < FN_NLOCALS(f); i++) set_nil(&st->stack[st->sp++]);
    set_nil(&vm->err_val);
    vm->err_msg = 0;
}

int tilvm_load(tilvm *vm, const uint8_t *bc, uint32_t size) {
    uint32_t p = 6, i;
    uint16_t n;
    if (size < 16 || bc[0] != 'T' || bc[1] != 'I' || bc[2] != 'L' || bc[3] != 'B') return -1;
    if (bc[4] < TILVM_BYTECODE_MIN_VERSION || bc[4] > TILVM_BYTECODE_VERSION) return -2;
    vm->bc = bc;
    vm->bc_size = size;
    vm->flags = bc[5];
#if !TILVM_FLOAT
    if (vm->flags & 1) return -3;          /* program uses floats */
#endif
    vm->n_globals = RD16(bc + p); p += 2;
    vm->entry = RD16(bc + p); p += 2;
    n = RD16(bc + p); p += 2;
    if (n > TILVM_MAX_CONSTS) return -4;
    vm->n_consts = n;
    for (i = 0; i < n; i++) {
        uint8_t tag;
        if (p >= size) return -5;
        tag = bc[p++];
        vm->const_off[i] = p;
        if (tag == 1) {
#if TILVM_INT_BITS < 64
            /* the constant must fit this VM's integer width */
            uint32_t lo = RD32(bc + p), hi = RD32(bc + p + 4);
            int32_t v = (int32_t)lo;
            if (hi != (v < 0 ? 0xFFFFFFFFu : 0u)) return -6;
#if TILVM_INT_BITS == 16
            if (v < -32768 || v > 32767) return -6;
#endif
#endif
            p += 8;
        } else if (tag == 2) p += 8;
        else if (tag == 3) p += 2u + RD16(bc + p);
        else return -5;
    }
    vm->n_funcs = RD16(bc + p); p += 2;
    vm->funcs_off = p;
#if TILVM_MAX_FUNCS > 0
    if (vm->n_funcs > TILVM_MAX_FUNCS) return -10;
#endif
    p += 11u * vm->n_funcs;
    n = RD16(bc + p); p += 2;
    if (n > TILVM_MAX_TYPES) return -7;
    vm->n_types = n;
    for (i = 0; i < n; i++) {
        vm->type_off[i] = p;
        p += 3u + 2u * bc[p + 2];
    }
    vm->n_methods = RD16(bc + p); p += 2;
    vm->methods_off = p;
    p += 6u * vm->n_methods;
    vm->n_prelude = bc[p++];
    vm->prelude_off = p;
    p += 3u * vm->n_prelude;
    vm->code_size = RD32(bc + p); p += 4;
    vm->code_off = p;
    p += vm->code_size;
    if (p + 2 > size) return -5;
    vm->n_lines = RD16(bc + p); p += 2;
    vm->lines_off = p;
    if (vm->n_globals + 8 > TILVM_STACK_SIZE) return -8;
#if TILVM_MAX_FUNCS > 0
    for (i = 0; i < vm->n_funcs; i++) {       /* decode the function records once */
        const uint8_t *f = func_rec(vm, (uint16_t)i);
        vm->funcs[i].code = vm->code_off + FN_OFFSET(f);
        vm->funcs[i].nparams = FN_NPARAMS(f);
        vm->funcs[i].nreq = FN_NREQ(f);
        vm->funcs[i].nlocals = FN_NLOCALS(f);
        vm->funcs[i].flags = FN_FLAGS(f);
    }
#endif
#if TILVM_FIELD_CACHE > 0
    for (i = 0; i < TILVM_FIELD_CACHE; i++) {
        vm->field_cache[i].type = CACHE_EMPTY;
        vm->method_cache[i].type = CACHE_EMPTY;
    }
#endif
    vm->bc_hash = fnv1a(bc, size);
#ifndef TILVM_NO_STDIO
    vm->write = tilvm_default_write;
    vm->write_err = default_write_err;
    vm->read_line = default_read_line;
#else
    vm->write = 0;
    vm->write_err = 0;
    vm->read_line = 0;
#endif
#ifdef TV_HAVE_CLOCK
    vm->clock_ms = default_clock_ms;
#else
    vm->clock_ms = 0;
#endif
    vm->ctx = 0;
    vm->max_steps = 0;
    memset(&vm->host, 0, sizeof(vm->host));
#ifndef TILVM_NO_STDIO
    vm->host.read_file = default_read_file;
    vm->host.write_file = default_write_file;
#endif
    reset_state(vm);
    return 0;
}

/* ─── errors ────────────────────────────────────────────────────────────── */
static uint16_t line_for(tilvm *vm, uint32_t pc) {
    uint32_t lo = 0, hi = vm->n_lines, best = 0;
    pc -= vm->code_off;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        const uint8_t *e = vm->bc + vm->lines_off + 6u * mid;
        if (RD32(e) <= pc) { best = RD16(e + 4); lo = mid + 1; }
        else hi = mid;
    }
    return (uint16_t)best;
}

static void err_out(tilvm *vm, const char *s, size_t n) {
    tilvm_write_fn w = vm->write_err ? vm->write_err : vm->write;
    if (w && n) w(vm->ctx, s, n);
}

static void err_cstr(tilvm *vm, const char *s) { err_out(vm, s, strlen(s)); }

static void report(tilvm *vm, uint32_t pc, const char *prefix) {
    char num[32];
    uint16_t i, line;
    err_cstr(vm, prefix);
    if (vm->err_msg) err_cstr(vm, vm->err_msg);
    if (is_str(&vm->err_val)) {
        uint32_t n;
        const char *p = str_data(vm, &vm->err_val, &n);
        err_out(vm, p, n);
    }
    err_cstr(vm, "\n");
    for (i = vm->st.n_frames; i > 0 && vm->st.n_frames - i < 8; i--) {
        tv_frame *fr = &vm->st.frames[i - 1];
        uint32_t at = (i == vm->st.n_frames) ? pc : fr->pc - 1;
        uint32_t nlen;
        tv_val name;
        const char *p;
        name.t = TV_STRK;
        name.u.k = FN_NAME(func_rec(vm, fr->func));
        p = str_data(vm, &name, &nlen);
        err_cstr(vm, "  at ");
        err_out(vm, p, nlen);
        line = line_for(vm, at);
        if (line) {
            err_cstr(vm, " (line ");
            err_cstr(vm, fmt_int(num, (tv_int)line));
            err_cstr(vm, ")");
        }
        err_cstr(vm, "\n");
    }
}

/* Unwind to the innermost recovery point (or `target`), push the error text. */
static int recover(tilvm *vm, int target) {
    tilvm_state *st = &vm->st;
    tv_try *t;
    tv_ref r;
    if (!st->n_try) return 0;
    if (target >= 0) {
        while (st->n_try > (uint16_t)(target + 1)) {
            if (st->tries[st->n_try - 1].deadline) st->n_deadlines--;
            st->n_try--;
        }
    }
    t = &st->tries[--st->n_try];
    if (t->deadline) st->n_deadlines--;
    st->n_frames = t->n_frames;
    st->sp = t->sp;
    st->frames[st->n_frames - 1].pc = t->handler;
    /* push the message as a string */
    if (vm->err_val.t != TV_NIL && is_str(&vm->err_val) && (!vm->err_msg || !*vm->err_msg)) {
        st->stack[st->sp++] = vm->err_val;
    } else {
        uint32_t ml = vm->err_msg ? (uint32_t)strlen(vm->err_msg) : 0, vl = 0;
        if (is_str(&vm->err_val)) str_data(vm, &vm->err_val, &vl);
        st->stack[st->sp++] = vm->err_val;          /* keep it rooted during allocation */
        r = new_str(vm, ml + vl);
        if (!r) { set_nil(&st->stack[st->sp - 1]); }
        else {
            const char *p;
            memcpy(STROBJ(vm, r)->data, vm->err_msg, ml);
            if (vl) { p = str_data(vm, &st->stack[st->sp - 1], &vl); memcpy(STROBJ(vm, r)->data + ml, p, vl); }
            set_ref(&st->stack[st->sp - 1], TV_STR, r);
        }
    }
    vm->err_msg = 0;
    set_nil(&vm->err_val);
    return 1;
}

/* constant k (int / float / string kept in ROM) -> *dst */
static void load_const(tilvm *vm, uint16_t k, tv_val *dst) {
    const uint8_t *p = vm->bc + vm->const_off[k];
    uint8_t tag = p[-1];
    if (tag == 3) { dst->t = TV_STRK; dst->u.i = 0; dst->u.k = k; }
    else if (tag == 1) {
#if TILVM_INT_BITS == 64
        set_int(dst, (tv_int)((uint64_t)RD32(p) | ((uint64_t)RD32(p + 4) << 32)));
#else
        set_int(dst, (tv_int)(int32_t)RD32(p));
#endif
    } else {
#if TILVM_FLOAT == 2 && !defined(__SDCC)
        uint64_t bits = (uint64_t)RD32(p) | ((uint64_t)RD32(p + 4) << 32);
        double d;
        memcpy(&d, &bits, sizeof d);
        set_float(dst, (tv_float)d);
#elif TILVM_FLOAT
        /* single precision (or SDCC, where double is 32-bit): decode IEEE754 */
        uint32_t hi = RD32(p + 4), lo = RD32(p);
        int e = (int)((hi >> 20) & 0x7FF) - 1023;
        tv_float m = (tv_float)1 + (tv_float)(hi & 0xFFFFF) / (tv_float)1048576.0
                     + (tv_float)lo / (tv_float)4503599627370496.0;
        if (((hi >> 20) & 0x7FF) == 0) m = 0;
        if (e > 200) m = (tv_float)1e30 * (tv_float)1e30;   /* inf in float */
        else if (e < -200) m = 0;
        else {
            while (e > 0) { m *= 2; e--; }
            while (e < 0) { m /= 2; e++; }
        }
        set_float(dst, (hi >> 31) ? -m : m);
#else
        set_nil(dst);
#endif
    }
}

static int no_method_error(tilvm *vm, uint16_t name_c, const tv_val *recv) {
    char nm[64], tn[64];
    uint32_t len;
    tv_val nv;
    const char *np;
    nv.t = TV_STRK;
    nv.u.k = name_c;
    np = str_data(vm, &nv, &len);
    if (len > 63) len = 63;
    memcpy(nm, np, len);
    nm[len] = 0;
    if (recv->t == TV_STRUCT) {
        nv.u.k = type_name_const(vm, OBJ(vm, recv->u.r)->aux);
        np = str_data(vm, &nv, &len);
        if (len > 63) len = 63;
        memcpy(tn, np, len);
        tn[len] = 0;
    } else strcpy(tn, type_name(vm, recv));
    return failn(vm, "no method '", nm, "' on ", tn, 0, 0);
}

static uint16_t find_method(tilvm *vm, uint16_t t, uint16_t name_c, int *found) {
    /* binary search of the (type, name) sorted method table */
    uint16_t lo = 0, hi = vm->n_methods;
    while (lo < hi) {
        uint16_t mid = (uint16_t)((lo + hi) / 2);
        const uint8_t *e = vm->bc + vm->methods_off + 6u * mid;
        uint16_t et = RD16(e), en = RD16(e + 2);
        if (et == t && en == name_c) { *found = 1; return RD16(e + 4); }
        if (et < t || (et == t && en < name_c)) lo = (uint16_t)(mid + 1);
        else hi = mid;
    }
    *found = 0;
    return 0;
}

/* find_method() through a cache: (type, name) -> function (CACHE_EMPTY = no method) */
static uint16_t find_method_cached(tilvm *vm, uint16_t t, uint16_t name_c, int *found) {
#if TILVM_FIELD_CACHE > 0
    tv_cache *c = CACHE_AT(vm->method_cache, t, name_c);
    uint16_t f;
    if (c->type == t && c->name == name_c) { *found = c->value != CACHE_EMPTY; return c->value; }
    f = find_method(vm, t, name_c, found);
    c->type = t;
    c->name = name_c;
    c->value = *found ? f : (uint16_t)CACHE_EMPTY;
    return f;
#else
    return find_method(vm, t, name_c, found);
#endif
}

/* Decode INVOKE (operands at *pc) for the general cases: methods of user structs,
 * functions held in struct fields, prelude functions and builtin methods. Returns 1
 * when a bytecode function must be entered (outputs set), 2 when a builtin method
 * already left its result on the stack, 0 on error. */
static int prepare_invoke(tilvm *vm, uint32_t *pc, uint16_t *fidx, uint8_t *argc,
                          uint8_t *extra, tv_ref *clos) {
    const uint8_t *code = vm->bc;
    {
        uint16_t name_c = RD16(code + *pc), k;
        uint8_t mid = code[*pc + 2];
        tv_val *recv;
        int found = 0;
        *argc = code[*pc + 3];
        *pc += 4;
        recv = &STK[SP - *argc - 1];
        if (recv->t == TV_STRUCT) {
            uint16_t t = OBJ(vm, recv->u.r)->aux;
            *fidx = find_method_cached(vm, t, name_c, &found);
            if (found) { (*argc)++; return 1; }       /* the receiver becomes self */
            if (t > TYPE_ERR) {
                int slot = field_slot(vm, t, name_c);
                if (slot >= 0) {                       /* field holding a function */
                    tv_val fv;
                    fv = STRUCTOBJ(vm, recv->u.r)->v[slot];
                    *extra = 1;
                    if (fv.t == TV_FUNC) { *fidx = fv.u.k; return 1; }
                    if (fv.t == TV_CLOSURE) { *clos = fv.u.r; *fidx = CLOSOBJ(vm, *clos)->func; return 1; }
                    return no_field(vm, name_c, recv);
                }
            }
        }
        for (k = 0; k < vm->n_prelude && recv->t == TV_ARRAY; k++) {
            const uint8_t *e = vm->bc + vm->prelude_off + 3u * k;
            if (e[0] == mid) { *fidx = RD16(e + 1); (*argc)++; return 1; }
        }
        if (mid == M_NONE_) return no_method_error(vm, name_c, recv);
        if (!call_method(vm, mid, *argc)) return 0;
        return 2;
    }
}

/* "f() takes N argument(s), got M" (a recoverable error) */
static int arity_error(tilvm *vm, uint16_t fidx, uint8_t argc) {
    const uint8_t *f = func_rec(vm, fidx);
    char name[48], n1[32], n2[32];
    uint32_t nl;
    tv_val nv;
    const char *np;
    nv.t = TV_STRK;
    nv.u.k = FN_NAME(f);
    np = str_data(vm, &nv, &nl);
    if (nl > sizeof(name) - 1) nl = sizeof(name) - 1;
    memcpy(name, np, nl);
    name[nl] = 0;
    return failn(vm, name, "() takes ", fmt_int(n1, (tv_int)FN_NPARAMS(f)), " argument(s), got ",
                 fmt_int(n2, (tv_int)argc), 0);
}

#ifndef TILVM_LEAN
/* {k: v, ...} literal on top of the stack: normalise keys, later duplicates win */
static int map_literal_fixup(tilvm *vm) {
    tv_ref m = TOP.u.r;
    uint32_t i, j, len = ARROBJ(vm, m)->len;
    tv_val *v = VALS(vm, ARROBJ(vm, m)->buf);
    for (i = 0; i < len; i++) { map_key(&v[2 * i]); if (!check_map_key(vm, &v[2 * i])) return 0; }
    for (i = 0; i < len; i++) {
        for (j = i + 1; j < len; j++) {
            if (equal(vm, &v[2 * i], &v[2 * j], 0)) {
                uint32_t k;
                v[2 * i + 1] = v[2 * j + 1];
                for (k = j; k + 1 < len; k++) { v[2 * k] = v[2 * k + 2]; v[2 * k + 1] = v[2 * k + 3]; }
                len--;
                set_nil(&v[2 * len]);
                set_nil(&v[2 * len + 1]);
                j--;
            }
        }
    }
    ARROBJ(vm, m)->len = (tv_size)len;
    return 1;
}

#endif

/* replace the top n values by one string (f-strings) */
static int concat_values(tilvm *vm, uint8_t n) {
    uint8_t i;
    uint32_t total = 0, k = 0, l;
    uint16_t first = (uint16_t)(SP - n);
    tv_ref r;
#ifndef TILVM_LEAN
    /* measure the text of every part, then one allocation (the parts stay on the
     * stack, so the collector keeps and updates them) and one copy */
    sink s;
    char num[32];
    for (i = 0; i < n; i++) {
        const tv_val *v = &STK[first + i];
        if (is_str(v)) str_data(vm, v, &l);
        else if (v->t == TV_INT) l = (uint32_t)strlen(fmt_int(num, v->u.i));
        else { s.vm = vm; s.mode = 0; s.dst = 0; s.len = 0; emit_value(&s, v, 0, 0); l = s.len; }
        if (l > TV_STR_MAX || (total += l) > TV_STR_MAX) return fail(vm, "string too long");
    }
    r = new_str(vm, total);
    if (!r) return 0;
    for (i = 0; i < n; i++) {
        const tv_val *v = &STK[first + i];
        if (is_str(v)) {
            const char *p = str_data(vm, v, &l);
            memcpy(STROBJ(vm, r)->data + k, p, l);
        } else if (v->t == TV_INT) {
            const char *p = fmt_int(num, v->u.i);
            l = (uint32_t)strlen(p);
            memcpy(STROBJ(vm, r)->data + k, p, l);
        } else {
            s.vm = vm; s.mode = 1; s.dst = STROBJ(vm, r)->data + k; s.len = 0;
            emit_value(&s, v, 0, 0);
            l = s.len;
        }
        k += l;
    }
#else
    for (i = 0; i < n; i++) {
        if (!to_str_slot(vm, (uint16_t)(first + i))) return 0;
        str_data(vm, &STK[first + i], &l);
        total += l;
    }
    if (total > TV_STR_MAX) return fail(vm, "string too long");
    r = new_str(vm, total);
    if (!r) return 0;
    for (i = 0; i < n; i++) {
        const char *p = str_data(vm, &STK[first + i], &l);
        memcpy(STROBJ(vm, r)->data + k, p, l);
        k += l;
    }
#endif
    SP = first;
    set_ref(&STK[SP++], TV_STR, r);
    return 1;
}

/* retry / timeout / intent recovery point */
static int push_try(tilvm *vm, uint8_t op, uint32_t handler) {
    tilvm_state *st = &vm->st;
    tv_try *t;
    uint32_t deadline = 0;
    if (op == OP_TRY_TIMEOUT) {
        tv_val *ms = &STK[--SP];
        tv_int v = ms->t == TV_INT ? ms->u.i : 0;
#if TILVM_FLOAT
        if (ms->t == TV_FLOAT) v = (tv_int)ms->u.f;
#endif
        deadline = now_ms(vm) + (uint32_t)(v > 0 ? v : 0) + 1;
    }
    if (st->n_try >= TILVM_MAX_TRY) return fail(vm, "too many nested retry/timeout blocks");
    t = &st->tries[st->n_try++];
    t->handler = handler;
    t->deadline = deadline;
    t->n_frames = st->n_frames;
    t->sp = SP;
    if (deadline) st->n_deadlines++;
    return 1;
}

/* index of the outermost recovery point whose deadline has passed, or -1 */
static int expired_try(tilvm *vm) {
    tilvm_state *st = &vm->st;
    uint32_t now = now_ms(vm);
    uint16_t i;
    for (i = 0; i < st->n_try; i++)
        if (st->tries[i].deadline && (int32_t)(now - st->tries[i].deadline) >= 0) return i;
    return -1;
}

/* ─── the interpreter ───────────────────────────────────────────────────── */
/*
 * Registers: ip = next bytecode byte, sp = first free value-stack slot,
 * bp = locals of the current frame, fr = current frame, op_ip = start of the
 * current instruction (error reports). They live in C locals; the VM state
 * (st->sp, fr->pc) is brought up to date before a helper that works on the
 * stack (HELPER, SYNC_SP / RELOAD_SP), before anything that may allocate (the
 * collector scans the st->sp stack slots) and before returning to the host.
 *
 * Dispatch: with TILVM_COMPUTED_GOTO (GCC/Clang "labels as values") every
 * handler ends with its own indirect jump through dispatch[], otherwise one
 * portable switch; both come from the same CASE() handlers. To add an opcode:
 * write `CASE(OP_X) ... NEXT;` and put &&L_OP_X at its number in dispatch[].
 *
 * Steps (loop back-edges + calls) are counted at `backedge`. `fuel` counts down
 * to the next step at which the sandbox limit (vm->max_steps) or the budget of
 * tilvm_run_steps() must be looked at, so the common case is one decrement and
 * two tests; slow_backedge does exactly what TIL 3.0 did at every step.
 *
 * Superinstructions (0x80-0xBF) behave exactly like the sequences they replace
 * (same results, errors, error lines and steps); their slow paths push the
 * operands and run the plain handler. Without TILVM_FAST_PATHS (TILVM_LEAN,
 * -Os) only those slow paths are compiled.
 */
#ifndef TILVM_COMPUTED_GOTO
#if defined(__GNUC__) && !defined(TILVM_LEAN) && !defined(__SDCC) && !defined(__OPTIMIZE_SIZE__)
#define TILVM_COMPUTED_GOTO 1      /* GCC/Clang, not when optimizing for size (-Os) */
#else
#define TILVM_COMPUTED_GOTO 0
#endif
#endif

#if TILVM_COMPUTED_GOTO
#define CASE(x) L_##x:
#define NEXT do { op_ip = ip; op = *ip++; goto *dispatch[op]; } while (0)
#define DISPATCH_OP() goto *dispatch[op]
#else
#define CASE(x) case x:
#define NEXT goto next
#define DISPATCH_OP() goto dispatch_op
#endif
#define SYNC_SP() (st->sp = (uint16_t)(sp - stk))
#define RELOAD_SP() (sp = stk + st->sp)
#define SAVE_PC() (fr->pc = (uint32_t)(ip - code))
#define TOPSLOT(n) ((uint16_t)(sp - stk - (n)))          /* stack index of sp[-n] */
#define RDS16(p) ((int16_t)RD16(p))
#define CHECK_STACK(n) do { if (stk_end - sp < (n)) goto stack_overflow; } while (0)
/* run a helper that works on vm->st.sp; panic when it fails */
#define HELPER(call) do { int ok_; SYNC_SP(); ok_ = (call); RELOAD_SP(); if (!ok_) goto panic; } while (0)
#define REFUEL() (fuel_set = fuel = refuel(vm, budget))

/* steps that may run before the slow path of `backedge` must check the sandbox
 * limit or the budget (budget 0 = none) */
static uint32_t refuel(const tilvm *vm, uint32_t budget) {
    uint32_t fuel = budget ? budget : 0xFFFFFFFFu;
    if (vm->max_steps) {
        uint32_t left = vm->st.steps < vm->max_steps ? vm->max_steps - vm->st.steps : 0;
        if (left != 0xFFFFFFFFu) left++;
        if (left < fuel) fuel = left;
    }
    return fuel;
}

/* integer comparison x <c> y for c in OP_EQ..OP_GE without branches:
 * bit (sign(x - y) + 1) of the mask */
static const uint8_t CMP_MASK[6] = { 2, 5, 1, 3, 4, 6 };
#define IS_CMP(c) ((unsigned)(c) - OP_EQ < 6u)
#define INT_CMP(c, x, y) ((CMP_MASK[(c) - OP_EQ] >> (((x) > (y)) - ((x) < (y)) + 1)) & 1)

/* a <c> b for any values (fused compare-and-branch): 1 / 0, -1 error, -2 bad operand */
static int cmp_values(tilvm *vm, unsigned c, const tv_val *a, const tv_val *b) {
    int r;
    if (c == OP_EQ || c == OP_NE) {
        r = equal(vm, a, b, 0);
        return c == OP_EQ ? r : !r;
    }
    if (!IS_CMP(c)) { fail(vm, "bad opcode"); return -2; }
    if (!compare(vm, (int)c, a, b, &r)) return -1;
    return r;
}

#if TILVM_FAST_PATHS
/* does the program replace builtin array method `mid` by a prelude function? */
static int prelude_has(tilvm *vm, uint8_t mid) {
    uint8_t k;
    for (k = 0; k < vm->n_prelude; k++)
        if (vm->bc[vm->prelude_off + 3u * k] == mid) return 1;
    return 0;
}
#endif

#if TILVM_COMPUTED_GOTO && defined(__GNUC__)
/* labels as values are a GNU extension: allowed here even under -pedantic */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

int tilvm_run_steps(tilvm *vm, uint32_t budget) {
    const uint8_t *const code = vm->bc;
    tilvm_state *const st = &vm->st;
    tv_val *const stk = st->stack;
    tv_val *const stk_end = stk + TILVM_STACK_SIZE;
    const uint8_t *ip, *op_ip;
    tv_val *sp, *bp;
    tv_frame *fr;
    tv_val result, imm;
    const tv_val *cmp_a = 0, *cmp_b = 0;
    unsigned op, cmp = 0, cmp_pop = 0;
    int target = -1, r;
    uint32_t fuel, fuel_set;
    uint16_t fidx = 0, name_c = 0;
    uint8_t argc = 0, extra = 0;
    tv_ref clos = 0;
#if TILVM_COMPUTED_GOTO
    static const void *const dispatch[256] = {
        /* 0x00 */ &&L_OP_NOP, &&L_OP_PUSH_NIL, &&L_OP_PUSH_TRUE, &&L_OP_PUSH_FALSE, &&L_OP_PUSH_I8, &&L_OP_PUSH_I16, &&L_OP_PUSH_CONST, &&L_OP_POP,
        /* 0x08 */ &&L_OP_DUP, &&L_OP_DUP2, &&L_OP_SWAP, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0x10 */ &&L_OP_LOAD_LOCAL, &&L_OP_STORE_LOCAL, &&L_OP_LOAD_GLOBAL, &&L_OP_STORE_GLOBAL, &&L_OP_LOAD_UPVAL, &&L_OP_STORE_UPVAL, &&L_OP_LOAD_CELL, &&L_OP_STORE_CELL,
        /* 0x18 */ &&L_OP_MAKE_CELL, &&L_OP_LOAD_UPVAL_RAW, &&L_OP_ARG_DEFAULT, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0x20 */ &&L_OP_ADD, &&L_OP_SUB, &&L_OP_MUL, &&L_OP_DIV, &&L_OP_MOD, &&L_OP_POW, &&L_OP_NEG, &&L_OP_NOT,
        /* 0x28 */ &&L_OP_BAND, &&L_OP_BOR, &&L_OP_BXOR, &&L_OP_BNOT, &&L_OP_SHL, &&L_OP_SHR, &&L_BAD, &&L_BAD,
        /* 0x30 */ &&L_OP_EQ, &&L_OP_NE, &&L_OP_LT, &&L_OP_LE, &&L_OP_GT, &&L_OP_GE, &&L_BAD, &&L_BAD,
        /* 0x38 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0x40 */ &&L_OP_JUMP, &&L_OP_JUMP_IF_FALSE, &&L_OP_JUMP_IF_TRUE, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0x48 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0x50 */ &&L_OP_CALL, &&L_OP_CALL_VALUE, &&L_OP_INVOKE, &&L_OP_RETURN, &&L_OP_RETURN_NIL, &&L_OP_NATIVE, &&L_BAD, &&L_BAD,
        /* 0x58 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0x60 */ &&L_OP_NEW_ARRAY, &&L_OP_NEW_MAP, &&L_OP_NEW_STRUCT, &&L_OP_GET_INDEX, &&L_OP_SET_INDEX, &&L_OP_GET_FIELD, &&L_OP_SET_FIELD, &&L_OP_GET_SLOT,
        /* 0x68 */ &&L_OP_MAKE_CLOSURE, &&L_OP_PUSH_FUNC, &&L_OP_CONCAT, &&L_OP_FOR_ITER, &&L_OP_MATCH_TYPE, &&L_OP_CAST, &&L_OP_IS_ARRAY_LEN, &&L_BAD,
        /* 0x70 */ &&L_OP_TRY, &&L_OP_TRY_TIMEOUT, &&L_OP_END_TRY, &&L_OP_PANIC, &&L_OP_UNWRAP_OR_RETURN, &&L_OP_UNWRAP_OR_PANIC, &&L_BAD, &&L_BAD,
        /* 0x78 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_OP_HALT,
        /* 0x80 */ &&L_OP_LOAD_LOCAL2, &&L_OP_LOAD_LOCAL_FIELD, &&L_OP_INC_LOCAL, &&L_OP_CAST_LOCAL, &&L_OP_CAST_RETURN, &&L_OP_ADD_STORE_LOCAL, &&L_OP_NATIVE_POP, &&L_OP_INDEX_L,
        /* 0x88 */ &&L_OP_INDEX_LL, &&L_OP_RETURN_LOCAL, &&L_OP_FOR_ITER_STORE, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0x90 */ &&L_OP_JUMP_IF_NOT, &&L_OP_JUMP_IF_NOT_L, &&L_OP_JUMP_IF_NOT_I, &&L_OP_JUMP_IF_NOT_LL, &&L_OP_JUMP_IF_NOT_LI, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0x98 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0xA0 */ &&L_OP_ADD_I, &&L_OP_SUB_I, &&L_OP_MUL_I, &&L_OP_DIV_I, &&L_OP_MOD_I, &&L_OP_BAND_I, &&L_OP_BOR_I, &&L_OP_BXOR_I,
        /* 0xA8 */ &&L_OP_SHL_I, &&L_OP_SHR_I, &&L_OP_EQ_I, &&L_OP_NE_I, &&L_OP_LT_I, &&L_OP_LE_I, &&L_OP_GT_I, &&L_OP_GE_I,
        /* 0xB0 */ &&L_OP_ADD_L, &&L_OP_SUB_L, &&L_OP_MUL_L, &&L_OP_DIV_L, &&L_OP_MOD_L, &&L_OP_BAND_L, &&L_OP_BOR_L, &&L_OP_BXOR_L,
        /* 0xB8 */ &&L_OP_SHL_L, &&L_OP_SHR_L, &&L_OP_EQ_L, &&L_OP_NE_L, &&L_OP_LT_L, &&L_OP_LE_L, &&L_OP_GT_L, &&L_OP_GE_L,
        /* 0xC0 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0xC8 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0xD0 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0xD8 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0xE0 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0xE8 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0xF0 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
        /* 0xF8 */ &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD,
    };
#endif

    if (st->halted == 1) return TILVM_DONE;
    if (st->halted == 2) return TILVM_EXIT;
    if (st->failed) return TILVM_ERROR;
    set_nil(&result);
    set_nil(&imm);
    REFUEL();

resume:
    fr = &st->frames[st->n_frames - 1];
    ip = code + fr->pc;
    op_ip = ip;
    bp = stk + fr->base;
    RELOAD_SP();
#if TILVM_COMPUTED_GOTO
    NEXT;
#else
next:
    op_ip = ip;
    op = *ip++;
dispatch_op:
    switch (op) {
#endif

    /* ── constants and stack ── */
    CASE(OP_NOP) NEXT;
    CASE(OP_PUSH_NIL) CHECK_STACK(1); set_nil(sp); sp++; NEXT;
    CASE(OP_PUSH_TRUE) CHECK_STACK(1); set_bool(sp, 1); sp++; NEXT;
    CASE(OP_PUSH_FALSE) CHECK_STACK(1); set_bool(sp, 0); sp++; NEXT;
    CASE(OP_PUSH_I8) CHECK_STACK(1); set_int(sp, (tv_int)(int8_t)*ip); sp++; ip++; NEXT;
    CASE(OP_PUSH_I16) CHECK_STACK(1); set_int(sp, (tv_int)RDS16(ip)); sp++; ip += 2; NEXT;
    CASE(OP_PUSH_CONST) CHECK_STACK(1); load_const(vm, RD16(ip), sp); sp++; ip += 2; NEXT;
    CASE(OP_POP) sp--; NEXT;
    CASE(OP_DUP) CHECK_STACK(1); sp[0] = sp[-1]; sp++; NEXT;
    CASE(OP_DUP2) CHECK_STACK(2); sp[0] = sp[-2]; sp[1] = sp[-1]; sp += 2; NEXT;
    CASE(OP_SWAP) result = sp[-1]; sp[-1] = sp[-2]; sp[-2] = result; NEXT;

    /* ── variables ── */
    CASE(OP_LOAD_LOCAL) CHECK_STACK(1); *sp++ = bp[*ip++]; NEXT;
    CASE(OP_STORE_LOCAL) bp[*ip++] = *--sp; NEXT;
    CASE(OP_LOAD_GLOBAL) CHECK_STACK(1); *sp++ = stk[RD16(ip)]; ip += 2; NEXT;
    CASE(OP_STORE_GLOBAL) stk[RD16(ip)] = *--sp; ip += 2; NEXT;
    CASE(OP_LOAD_CELL) CHECK_STACK(1); *sp++ = CELLOBJ(vm, bp[*ip++].u.r)->v; NEXT;
    CASE(OP_STORE_CELL) CELLOBJ(vm, bp[*ip++].u.r)->v = *--sp; NEXT;
    CASE(OP_MAKE_CELL) {
        uint8_t slot = *ip++;
        tv_ref cell;
        SYNC_SP();
        cell = alloc_obj(vm, OBJ_CELL, sizeof(tv_cellobj));
        if (!cell) goto panic;
        CELLOBJ(vm, cell)->v = bp[slot];
        set_ref(&bp[slot], TV_CELL, cell);
        NEXT;
    }
    CASE(OP_LOAD_UPVAL) CHECK_STACK(1); *sp++ = CELLOBJ(vm, CLOSOBJ(vm, fr->closure)->v[*ip++].u.r)->v; NEXT;
    CASE(OP_STORE_UPVAL) CELLOBJ(vm, CLOSOBJ(vm, fr->closure)->v[*ip++].u.r)->v = *--sp; NEXT;
    CASE(OP_LOAD_UPVAL_RAW) CHECK_STACK(1); *sp++ = CLOSOBJ(vm, fr->closure)->v[*ip++]; NEXT;
    CASE(OP_ARG_DEFAULT) {
        uint8_t slot = ip[0];
        int16_t rel = RDS16(ip + 1);
        ip += 3;
        if (bp[slot].t != TV_UNDEF) ip += rel;
        NEXT;
    }

    /* ── arithmetic ── */
#if TILVM_FAST_PATHS
    CASE(OP_ADD)
        if (sp[-2].t == TV_INT && sp[-1].t == TV_INT) { sp[-2].u.i = iwrap_add(sp[-2].u.i, sp[-1].u.i); sp--; NEXT; }
        goto arith_op;
    CASE(OP_SUB)
        if (sp[-2].t == TV_INT && sp[-1].t == TV_INT) { sp[-2].u.i = iwrap_sub(sp[-2].u.i, sp[-1].u.i); sp--; NEXT; }
        goto arith_op;
    CASE(OP_MUL)
        if (sp[-2].t == TV_INT && sp[-1].t == TV_INT) { sp[-2].u.i = iwrap_mul(sp[-2].u.i, sp[-1].u.i); sp--; NEXT; }
        goto arith_op;
#else
    CASE(OP_ADD) CASE(OP_SUB) CASE(OP_MUL)
        if (sp[-2].t == TV_INT && sp[-1].t == TV_INT) {
            tv_int x = sp[-2].u.i, y = sp[-1].u.i;
            sp[-2].u.i = op == OP_ADD ? iwrap_add(x, y) : op == OP_SUB ? iwrap_sub(x, y) : iwrap_mul(x, y);
            sp--;
            NEXT;
        }
        goto arith_op;
#endif
#if TILVM_FAST_PATHS
    CASE(OP_DIV)
        if (sp[-2].t == TV_INT && sp[-1].t == TV_INT && sp[-1].u.i != 0) {
            sp[-2].u.i = idiv(sp[-2].u.i, sp[-1].u.i); sp--; NEXT;
        }
        goto arith_op;
    CASE(OP_MOD)
        if (sp[-2].t == TV_INT && sp[-1].t == TV_INT && sp[-1].u.i != 0) {
            sp[-2].u.i = imod(sp[-2].u.i, sp[-1].u.i); sp--; NEXT;
        }
        goto arith_op;
    CASE(OP_POW)
#else
    CASE(OP_DIV) CASE(OP_MOD) CASE(OP_POW)
#endif
    arith_op:
        HELPER(arith(vm, (int)op));
        NEXT;
    CASE(OP_NEG)
        if (sp[-1].t == TV_INT) sp[-1].u.i = (tv_int)((tv_uint)0 - (tv_uint)sp[-1].u.i);
#if TILVM_FLOAT
        else if (sp[-1].t == TV_FLOAT) sp[-1].u.f = -sp[-1].u.f;
#endif
        else { fail(vm, "cannot negate a non-number"); goto panic; }
        NEXT;
    CASE(OP_NOT) set_bool(&sp[-1], !truthy(vm, &sp[-1])); NEXT;
    CASE(OP_BNOT)
        if (sp[-1].t != TV_INT) { fail(vm, "~ needs an integer"); goto panic; }
        sp[-1].u.i = ~sp[-1].u.i;
        NEXT;
    CASE(OP_BAND) CASE(OP_BOR) CASE(OP_BXOR) CASE(OP_SHL) CASE(OP_SHR) {
        tv_int x = sp[-2].u.i, y = sp[-1].u.i, v;
        if (sp[-2].t != TV_INT || sp[-1].t != TV_INT) { fail(vm, "bitwise operators need integers"); goto panic; }
        switch (op) {
        case OP_BAND: v = x & y; break;
        case OP_BOR: v = x | y; break;
        case OP_BXOR: v = x ^ y; break;
        case OP_SHL: v = (tv_int)((tv_uint)x << ((unsigned)y & (TILVM_INT_BITS - 1))); break;
        default: v = ishr(x, (unsigned)y & (TILVM_INT_BITS - 1)); break;
        }
        set_int(&sp[-2], v);
        sp--;
        NEXT;
    }
    CASE(OP_EQ) CASE(OP_NE) {
#if TILVM_FAST_PATHS
        int e = sp[-2].t == TV_INT && sp[-1].t == TV_INT ? sp[-2].u.i == sp[-1].u.i
                                                         : equal(vm, &sp[-2], &sp[-1], 0);
#else
        int e = equal(vm, &sp[-2], &sp[-1], 0);
#endif
        set_bool(&sp[-2], op == OP_EQ ? e : !e);
        sp--;
        NEXT;
    }
    CASE(OP_LT) CASE(OP_LE) CASE(OP_GT) CASE(OP_GE)
        if (sp[-2].t == TV_INT && sp[-1].t == TV_INT) r = INT_CMP(op, sp[-2].u.i, sp[-1].u.i);
        else if (!compare(vm, (int)op, &sp[-2], &sp[-1], &r)) goto panic;
        set_bool(&sp[-2], r);
        sp--;
        NEXT;

    /* ── jumps ── */
    CASE(OP_JUMP) {
        int16_t rel = RDS16(ip);
        ip += 2 + rel;
        if (rel < 0) goto backedge;
        NEXT;
    }
    CASE(OP_JUMP_IF_FALSE) CASE(OP_JUMP_IF_TRUE) {
        int16_t rel = RDS16(ip);
        tv_val *v = --sp;
        int t = v->t == TV_BOOL ? (int)v->u.i : truthy(vm, v);
        ip += 2;
        if (t == (op == OP_JUMP_IF_TRUE)) {
            ip += rel;
            if (rel < 0) goto backedge;
        }
        NEXT;
    }

    /* ── calls ── */
    CASE(OP_CALL)
        fidx = RD16(ip);
        argc = ip[2];
        ip += 3;
        extra = 0;
        clos = 0;
        goto call_fn;
    CASE(OP_CALL_VALUE) {
        const tv_val *callee;
        argc = *ip++;
        callee = &sp[-(int)argc - 1];
        extra = 1;
        clos = 0;
        if (callee->t == TV_FUNC) fidx = callee->u.k;
        else if (callee->t == TV_CLOSURE) { clos = callee->u.r; fidx = CLOSOBJ(vm, clos)->func; }
        else { failn(vm, type_name(vm, callee), " is not callable", 0, 0, 0, 0); goto panic; }
        goto call_fn;
    }
    CASE(OP_INVOKE) {
        uint32_t pc;
#if TILVM_FAST_PATHS
        const tv_val *recv;
        name_c = RD16(ip);
        argc = ip[3];
        recv = &sp[-(int)argc - 1];
        if (recv->t == TV_STRUCT) {                  /* user method: cached lookup */
            uint16_t t = OBJ(vm, recv->u.r)->aux;
            int found;
#if TILVM_FIELD_CACHE > 0
            const tv_cache *c = CACHE_AT(vm->method_cache, t, name_c);
            if (c->type == t && c->name == name_c) { fidx = c->value; found = fidx != CACHE_EMPTY; }
            else
#endif
                fidx = find_method_cached(vm, t, name_c, &found);
            if (found) { ip += 4; argc++; extra = 0; clos = 0; goto call_fn; }
        } else if (recv->t == TV_ARRAY && ip[2] == M_PUSH && argc == 1 && !prelude_has(vm, M_PUSH)) {
            tv_arrobj *a = ARROBJ(vm, recv->u.r);  /* array.push(x): no general method call */
            ip += 4;
            if (a->len < OBJ(vm, a->buf)->aux) {     /* room in the buffer */
                VALS(vm, a->buf)[a->len] = sp[-1];
                a->len = (tv_size)(a->len + 1);
            } else {
                HELPER(array_push(vm, TOPSLOT(2), TOPSLOT(1)));
            }
            sp--;
            set_nil(&sp[-1]);
            NEXT;
        }
#endif
        pc = (uint32_t)(ip - code);
        extra = 0;
        clos = 0;
        SAVE_PC();
        SYNC_SP();
        r = prepare_invoke(vm, &pc, &fidx, &argc, &extra, &clos);
        RELOAD_SP();
        ip = code + pc;
        if (r == 0) goto panic;
        if (r == 2) NEXT;                            /* builtin method: result pushed */
        goto call_fn;
    }
call_fn:   /* enter function fidx: argc arguments on the stack (+ `extra` callee slot below) */
    {
        uint8_t np = F_NPARAMS(vm, fidx), nl = F_NLOCALS(vm, fidx), i;
        tv_val *nb;
        SAVE_PC();                                   /* the caller continues after the call */
        if (argc > np || argc < F_NREQ(vm, fidx)) { arity_error(vm, fidx, argc); goto panic; }
        if (st->n_frames >= TILVM_MAX_FRAMES) { fail(vm, "stack overflow (recursion too deep)"); st->failed = 1; goto fatal; }
        if ((sp - stk) + (nl - argc) + 4 > TILVM_STACK_SIZE) { fail(vm, "stack overflow"); st->failed = 1; goto fatal; }
        nb = sp - argc;
        for (i = argc; i < np; i++) { sp->t = TV_UNDEF; sp->u.i = 0; sp++; }
        for (i = np; i < nl; i++) { set_nil(sp); sp++; }
        fr = &st->frames[st->n_frames++];
        fr->func = fidx;
        fr->base = (uint16_t)(nb - stk);
        fr->closure = clos;
        fr->extra = extra;
        fr->ret_pc = 0;
        fr->pc = F_CODE(vm, fidx);
        ip = code + fr->pc;
        bp = nb;
        goto backedge;
    }
    CASE(OP_RETURN) result = *--sp; goto do_return;
    CASE(OP_RETURN_NIL) set_nil(&result);
do_return:
    while (st->n_try && st->tries[st->n_try - 1].n_frames >= st->n_frames) {
        if (st->tries[st->n_try - 1].deadline) st->n_deadlines--;
        st->n_try--;
    }
    sp = stk + fr->base - fr->extra;
    *sp++ = result;
    if (!--st->n_frames) { st->halted = 1; SYNC_SP(); return TILVM_DONE; }
    fr = &st->frames[st->n_frames - 1];
    ip = code + fr->pc;
    bp = stk + fr->base;
    NEXT;
    CASE(OP_UNWRAP_OR_RETURN) CASE(OP_UNWRAP_OR_PANIC) {
        tv_val *v = &sp[-1];
        int ok = v->t == TV_STRUCT && (OBJ(vm, v->u.r)->aux == TYPE_SOME || OBJ(vm, v->u.r)->aux == TYPE_OK);
        int bad = v->t == TV_NIL || (v->t == TV_STRUCT && OBJ(vm, v->u.r)->aux == TYPE_ERR);
        if (ok) { *v = STRUCTOBJ(vm, v->u.r)->v[0]; NEXT; }
        if (!bad) NEXT;
        if (op == OP_UNWRAP_OR_PANIC) {
            if (v->t == TV_NIL) { fail(vm, "? applied to None"); goto panic; }
            sp[-1] = STRUCTOBJ(vm, v->u.r)->v[0];
            HELPER(to_str_slot(vm, TOPSLOT(1)));
            vm->err_val = sp[-1];
            vm->err_msg = "? applied to Err: ";
            goto panic;
        }
        result = *--sp;
        goto do_return;
    }
    CASE(OP_NATIVE) CASE(OP_NATIVE_POP) {
        uint8_t id = ip[0], n = ip[1];
        ip += 2;
        SAVE_PC();
        HELPER(call_native(vm, id, n));
        if (st->halted == 2) return TILVM_EXIT;
        if (op == OP_NATIVE_POP) sp--;
        NEXT;
    }

    /* ── objects ── */
    CASE(OP_NEW_ARRAY) CASE(OP_NEW_MAP) {
        uint16_t n = RD16(ip);
        ip += 2;
        CHECK_STACK(2);
        HELPER(make_array(vm, op == OP_NEW_MAP ? OBJ_MAP : OBJ_ARRAY, n));
#ifndef TILVM_LEAN
        if (op == OP_NEW_MAP && n) HELPER(map_literal_fixup(vm));
#endif
        NEXT;
    }
    CASE(OP_NEW_STRUCT) {
        uint16_t t = RD16(ip);
        uint8_t n = ip[2];
        ip += 3;
        HELPER(new_struct(vm, t, n));
        NEXT;
    }
    CASE(OP_GET_INDEX)
    get_index:
#if TILVM_FAST_PATHS
        if (sp[-2].t == TV_ARRAY && sp[-1].t == TV_INT) {
            tv_arrobj *a = ARROBJ(vm, sp[-2].u.r);
            if ((tv_uint)sp[-1].u.i < a->len) { sp[-2] = VALS(vm, a->buf)[sp[-1].u.i]; sp--; NEXT; }
        }
#endif
        HELPER(get_index(vm));
        NEXT;
    CASE(OP_SET_INDEX)
#if TILVM_FAST_PATHS
        if (sp[-3].t == TV_ARRAY && sp[-2].t == TV_INT) {
            tv_arrobj *a = ARROBJ(vm, sp[-3].u.r);
            if ((tv_uint)sp[-2].u.i < a->len) { VALS(vm, a->buf)[sp[-2].u.i] = sp[-1]; sp -= 3; NEXT; }
        }
#endif
        HELPER(set_index(vm));
        NEXT;
    CASE(OP_GET_FIELD)
        name_c = RD16(ip);
        ip += 2;
    get_field:
        if (sp[-1].t == TV_STRUCT) {
            int slot;
            FIELD_SLOT(vm, OBJ(vm, sp[-1].u.r)->aux, name_c, slot);
            if (slot >= 0) { sp[-1] = STRUCTOBJ(vm, sp[-1].u.r)->v[slot]; NEXT; }
        }
        no_field(vm, name_c, &sp[-1]);
        goto panic;
    CASE(OP_SET_FIELD) {
        tv_val *o = &sp[-2];
        name_c = RD16(ip);
        ip += 2;
        if (o->t == TV_STRUCT) {
            int slot;
            FIELD_SLOT(vm, OBJ(vm, o->u.r)->aux, name_c, slot);
            if (slot >= 0) { STRUCTOBJ(vm, o->u.r)->v[slot] = sp[-1]; sp -= 2; NEXT; }
        }
        no_field(vm, name_c, o);
        goto panic;
    }
    CASE(OP_GET_SLOT) {
        uint8_t i = *ip++;
        tv_val *v = &sp[-1];
        if (v->t == TV_STRUCT && i < (OBJ(vm, v->u.r)->size - offsetof(tv_structobj, v)) / sizeof(tv_val))
            *v = STRUCTOBJ(vm, v->u.r)->v[i];
        else if (v->t == TV_ARRAY && i < ARROBJ(vm, v->u.r)->len)
            *v = VALS(vm, ARROBJ(vm, v->u.r)->buf)[i];
        else { fail(vm, "pattern slot out of range"); goto panic; }
        NEXT;
    }
    CASE(OP_MAKE_CLOSURE) {
        uint16_t f = RD16(ip);
        uint8_t n = ip[2], i;
        tv_ref cr;
        ip += 3;
        SYNC_SP();
        cr = alloc_obj(vm, OBJ_CLOSURE, (uint32_t)offsetof(tv_closobj, v) + n * (uint32_t)sizeof(tv_val));
        if (!cr) goto panic;
        OBJ(vm, cr)->aux = n;
        CLOSOBJ(vm, cr)->func = f;
        for (i = 0; i < n; i++) CLOSOBJ(vm, cr)->v[i] = sp[(int)i - (int)n];
        sp -= n;
        set_ref(sp, TV_CLOSURE, cr);
        sp++;
        NEXT;
    }
    CASE(OP_PUSH_FUNC) CHECK_STACK(1); sp->t = TV_FUNC; sp->u.i = 0; sp->u.k = RD16(ip); sp++; ip += 2; NEXT;
    CASE(OP_CONCAT) {
        uint8_t n = *ip++;
        HELPER(concat_values(vm, n));
        NEXT;
    }
    CASE(OP_FOR_ITER) CASE(OP_FOR_ITER_STORE) {
        /* FOR_ITER it off [; STORE_LOCAL x]: next element of the array / string in
         * local `it` (index in it + 1), or jump when done */
        tv_val *c = &bp[ip[0]], *ix = c + 1, *dst;
        uint32_t len = 0;
        const char *p = 0;
        int16_t rel;
        if (op == OP_FOR_ITER) {
            rel = RDS16(ip + 1);
            ip += 3;
            dst = sp;
        } else {
            rel = RDS16(ip + 2);
            dst = &bp[ip[1]];
            ip += 4;
        }
        if (c->t == TV_ARRAY) len = ARROBJ(vm, c->u.r)->len;
        else p = str_data(vm, c, &len);
        if ((tv_uint)ix->u.i < len) {
            if (op == OP_FOR_ITER) CHECK_STACK(1);
            if (c->t == TV_ARRAY) *dst = VALS(vm, ARROBJ(vm, c->u.r)->buf)[ix->u.i];
            else set_int(dst, (tv_int)(uint8_t)p[ix->u.i]);
            if (op == OP_FOR_ITER) sp++;
            ix->u.i++;
        } else {
            ip += rel;
        }
        NEXT;
    }
    CASE(OP_MATCH_TYPE) {
        uint16_t t = RD16(ip);
        ip += 2;
        set_bool(&sp[-1], sp[-1].t == TV_STRUCT && OBJ(vm, sp[-1].u.r)->aux == t);
        NEXT;
    }
    CASE(OP_IS_ARRAY_LEN) {
        uint16_t n = RD16(ip);
        ip += 2;
        set_bool(&sp[-1], sp[-1].t == TV_ARRAY && ARROBJ(vm, sp[-1].u.r)->len == n);
        NEXT;
    }
    CASE(OP_CAST) {
        uint8_t kind = *ip++;
        HELPER(cast_slot(vm, TOPSLOT(1), kind));
        NEXT;
    }

    /* ── errors and recovery points ── */
    CASE(OP_TRY) CASE(OP_TRY_TIMEOUT) {
        int16_t rel = RDS16(ip);
        ip += 2;
        HELPER(push_try(vm, (uint8_t)op, (uint32_t)(ip - code) + (uint32_t)(int32_t)rel));
        NEXT;
    }
    CASE(OP_END_TRY)
        if (st->n_try) {
            if (st->tries[st->n_try - 1].deadline) st->n_deadlines--;
            st->n_try--;
        }
        NEXT;
    CASE(OP_PANIC)
        HELPER(to_str_slot(vm, TOPSLOT(1)));
        vm->err_val = *--sp;
        vm->err_msg = "";
        goto panic;
    CASE(OP_HALT)
        st->halted = 1;
        SAVE_PC();
        SYNC_SP();
        return TILVM_DONE;

    /* ── superinstructions (docs/TIL_BYTECODE_V2.md) ── */
    CASE(OP_LOAD_LOCAL2)                     /* LOAD_LOCAL a; LOAD_LOCAL b */
        CHECK_STACK(2);
        sp[0] = bp[ip[0]];
        sp[1] = bp[ip[1]];
        sp += 2;
        ip += 2;
        NEXT;
    CASE(OP_LOAD_LOCAL_FIELD)                /* LOAD_LOCAL a; GET_FIELD name */
        CHECK_STACK(1);
        *sp++ = bp[ip[0]];
        name_c = RD16(ip + 1);
        ip += 3;
        goto get_field;
    CASE(OP_INC_LOCAL) {                     /* LOAD_LOCAL a; PUSH_I8 k; ADD; STORE_LOCAL a */
        tv_val *v = &bp[ip[0]];
#if TILVM_FAST_PATHS
        if (v->t == TV_INT) { v->u.i = iwrap_add(v->u.i, (tv_int)(int8_t)ip[1]); ip += 2; NEXT; }
#endif
        CHECK_STACK(2);
        sp[0] = *v;
        set_int(&sp[1], (tv_int)(int8_t)ip[1]);
        sp += 2;
        HELPER(arith(vm, OP_ADD));
        *v = *--sp;
        ip += 2;
        NEXT;
    }
    CASE(OP_CAST_LOCAL) {                    /* LOAD_LOCAL a; CAST kind; STORE_LOCAL a */
        tv_val *v = &bp[ip[0]];
        uint8_t kind = ip[1];
        ip += 2;
#if TILVM_FAST_PATHS
        if (kind == 0 && v->t == TV_INT) NEXT;
#if TILVM_FLOAT
        if (kind == 1 && v->t == TV_FLOAT) NEXT;
#endif
#endif
        CHECK_STACK(1);
        *sp++ = *v;
        HELPER(cast_slot(vm, TOPSLOT(1), kind));
        *v = *--sp;
        NEXT;
    }
    CASE(OP_CAST_RETURN) {                   /* CAST kind; RETURN */
        uint8_t kind = *ip++;
#if TILVM_FAST_PATHS
        if (!(kind == 0 && sp[-1].t == TV_INT))
#endif
            HELPER(cast_slot(vm, TOPSLOT(1), kind));
        result = *--sp;
        goto do_return;
    }
    CASE(OP_RETURN_LOCAL)                    /* LOAD_LOCAL a; RETURN */
        result = bp[*ip++];
        goto do_return;
    CASE(OP_ADD_STORE_LOCAL)                 /* ADD; STORE_LOCAL a */
#if TILVM_FAST_PATHS
        if (sp[-2].t == TV_INT && sp[-1].t == TV_INT) {
            set_int(&bp[*ip++], iwrap_add(sp[-2].u.i, sp[-1].u.i));
            sp -= 2;
            NEXT;
        }
#endif
        HELPER(arith(vm, OP_ADD));
        bp[*ip++] = *--sp;
        NEXT;
    CASE(OP_INDEX_L)                         /* LOAD_LOCAL b; GET_INDEX */
#if TILVM_FAST_PATHS
        if (sp[-1].t == TV_ARRAY && bp[*ip].t == TV_INT) {
            tv_arrobj *a = ARROBJ(vm, sp[-1].u.r);
            if ((tv_uint)bp[*ip].u.i < a->len) { sp[-1] = VALS(vm, a->buf)[bp[*ip].u.i]; ip++; NEXT; }
        }
#endif
        CHECK_STACK(1);
        *sp++ = bp[*ip++];
        goto get_index;
    CASE(OP_INDEX_LL)                        /* LOAD_LOCAL a; LOAD_LOCAL b; GET_INDEX */
#if TILVM_FAST_PATHS
        if (bp[ip[0]].t == TV_ARRAY && bp[ip[1]].t == TV_INT && sp < stk_end) {
            tv_arrobj *a = ARROBJ(vm, bp[ip[0]].u.r);
            if ((tv_uint)bp[ip[1]].u.i < a->len) { *sp++ = VALS(vm, a->buf)[bp[ip[1]].u.i]; ip += 2; NEXT; }
        }
#endif
        CHECK_STACK(2);
        sp[0] = bp[ip[0]];
        sp[1] = bp[ip[1]];
        sp += 2;
        ip += 2;
        goto get_index;

    /* <cmp> + JUMP_IF_FALSE: jump when the comparison is false */
    CASE(OP_JUMP_IF_NOT)                     /* <cmp>; JUMP_IF_FALSE */
        cmp_a = &sp[-2]; cmp_b = &sp[-1]; cmp_pop = 2; cmp = ip[0]; ip += 1;
        goto cmp_branch;
    CASE(OP_JUMP_IF_NOT_L)                   /* LOAD_LOCAL b; <cmp>; JUMP_IF_FALSE */
        cmp_a = &sp[-1]; cmp_b = &bp[ip[1]]; cmp_pop = 1; cmp = ip[0]; ip += 2;
        goto cmp_branch;
    CASE(OP_JUMP_IF_NOT_I)                   /* PUSH_I8/I16 k; <cmp>; JUMP_IF_FALSE */
        set_int(&imm, (tv_int)RDS16(ip + 1));
        cmp_a = &sp[-1]; cmp_b = &imm; cmp_pop = 1; cmp = ip[0]; ip += 3;
        goto cmp_branch;
    CASE(OP_JUMP_IF_NOT_LL)                  /* LOAD_LOCAL a; LOAD_LOCAL b; <cmp>; JUMP_IF_FALSE */
        cmp_a = &bp[ip[1]]; cmp_b = &bp[ip[2]]; cmp_pop = 0; cmp = ip[0];
#if TILVM_FAST_PATHS
        if (cmp_a->t == TV_INT && cmp_b->t == TV_INT && IS_CMP(cmp)) {
            int16_t rel = RDS16(ip + 3);
            ip += 5;
            if (!INT_CMP(cmp, cmp_a->u.i, cmp_b->u.i)) { ip += rel; if (rel < 0) goto backedge; }
            NEXT;
        }
#endif
        ip += 3;
        goto cmp_branch;
    CASE(OP_JUMP_IF_NOT_LI)                  /* LOAD_LOCAL a; PUSH_I8/I16 k; <cmp>; JUMP_IF_FALSE */
        cmp_a = &bp[ip[1]]; cmp = ip[0];
#if TILVM_FAST_PATHS
        if (cmp_a->t == TV_INT && IS_CMP(cmp)) {
            int16_t rel = RDS16(ip + 4);
            tv_int k = (tv_int)RDS16(ip + 2);
            ip += 6;
            if (!INT_CMP(cmp, cmp_a->u.i, k)) { ip += rel; if (rel < 0) goto backedge; }
            NEXT;
        }
#endif
        set_int(&imm, (tv_int)RDS16(ip + 2));
        cmp_b = &imm; cmp_pop = 0; ip += 4;
cmp_branch:   /* ip at the i16 offset */
    {
        int16_t rel = RDS16(ip);
        ip += 2;
        if (cmp_a->t == TV_INT && cmp_b->t == TV_INT && IS_CMP(cmp)) r = INT_CMP(cmp, cmp_a->u.i, cmp_b->u.i);
        else {
            r = cmp_values(vm, cmp, cmp_a, cmp_b);
            if (r == -2) { st->failed = 1; goto fatal; }
            if (r < 0) goto panic;
        }
        sp -= cmp_pop;
        if (!r) { ip += rel; if (rel < 0) goto backedge; }
        NEXT;
    }

    /* <op>_I k = PUSH_I8/PUSH_I16 k; <op> (k: i16)     <op>_L b = LOAD_LOCAL b; <op> */
#if TILVM_FAST_PATHS
#define FAST_I(expr) \
        if (sp[-1].t == TV_INT) { tv_int x = sp[-1].u.i, k = (tv_int)RDS16(ip); ip += 2; sp[-1].u.i = (expr); NEXT; } \
        goto fused_i;
#define FAST_I_BOOL(expr) \
        if (sp[-1].t == TV_INT) { tv_int x = sp[-1].u.i, k = (tv_int)RDS16(ip); ip += 2; set_bool(&sp[-1], expr); NEXT; } \
        goto fused_i;
#define FAST_L(expr) \
        if (sp[-1].t == TV_INT && bp[*ip].t == TV_INT) { tv_int x = sp[-1].u.i, k = bp[*ip].u.i; ip++; sp[-1].u.i = (expr); NEXT; } \
        goto fused_l;
#define FAST_L_BOOL(expr) \
        if (sp[-1].t == TV_INT && bp[*ip].t == TV_INT) { tv_int x = sp[-1].u.i, k = bp[*ip].u.i; ip++; set_bool(&sp[-1], expr); NEXT; } \
        goto fused_l;
    CASE(OP_ADD_I) FAST_I(iwrap_add(x, k))
    CASE(OP_SUB_I) FAST_I(iwrap_sub(x, k))
    CASE(OP_MUL_I) FAST_I(iwrap_mul(x, k))
    CASE(OP_BAND_I) FAST_I(x & k)
    CASE(OP_BOR_I) FAST_I(x | k)
    CASE(OP_BXOR_I) FAST_I(x ^ k)
    CASE(OP_EQ_I) FAST_I_BOOL(x == k)
    CASE(OP_NE_I) FAST_I_BOOL(x != k)
    CASE(OP_LT_I) FAST_I_BOOL(x < k)
    CASE(OP_LE_I) FAST_I_BOOL(x <= k)
    CASE(OP_GT_I) FAST_I_BOOL(x > k)
    CASE(OP_GE_I) FAST_I_BOOL(x >= k)
    CASE(OP_DIV_I) if (RDS16(ip) != 0) { FAST_I(idiv(x, k)) } goto fused_i;
    CASE(OP_MOD_I) if (RDS16(ip) != 0) { FAST_I(imod(x, k)) } goto fused_i;
    CASE(OP_ADD_L) FAST_L(iwrap_add(x, k))
    CASE(OP_SUB_L) FAST_L(iwrap_sub(x, k))
    CASE(OP_MUL_L) FAST_L(iwrap_mul(x, k))
    CASE(OP_BAND_L) FAST_L(x & k)
    CASE(OP_BOR_L) FAST_L(x | k)
    CASE(OP_BXOR_L) FAST_L(x ^ k)
    CASE(OP_EQ_L) FAST_L_BOOL(x == k)
    CASE(OP_NE_L) FAST_L_BOOL(x != k)
    CASE(OP_LT_L) FAST_L_BOOL(x < k)
    CASE(OP_LE_L) FAST_L_BOOL(x <= k)
    CASE(OP_GT_L) FAST_L_BOOL(x > k)
    CASE(OP_GE_L) FAST_L_BOOL(x >= k)
    CASE(OP_DIV_L) if (bp[*ip].t == TV_INT && bp[*ip].u.i != 0) { FAST_L(idiv(x, k)) } goto fused_l;
    CASE(OP_MOD_L) if (bp[*ip].t == TV_INT && bp[*ip].u.i != 0) { FAST_L(imod(x, k)) } goto fused_l;
#undef FAST_I
#undef FAST_I_BOOL
#undef FAST_L
#undef FAST_L_BOOL
    CASE(OP_SHL_I) CASE(OP_SHR_I)
    fused_i:
#else
    CASE(OP_ADD_I) CASE(OP_SUB_I) CASE(OP_MUL_I) CASE(OP_DIV_I) CASE(OP_MOD_I) CASE(OP_BAND_I)
    CASE(OP_BOR_I) CASE(OP_BXOR_I) CASE(OP_SHL_I) CASE(OP_SHR_I) CASE(OP_EQ_I) CASE(OP_NE_I)
    CASE(OP_LT_I) CASE(OP_LE_I) CASE(OP_GT_I) CASE(OP_GE_I)
#endif
        CHECK_STACK(1);
        set_int(sp, (tv_int)RDS16(ip));
        sp++;
        ip += 2;
        op = FUSED_BINOP[op & 15];
        DISPATCH_OP();
#if TILVM_FAST_PATHS
    CASE(OP_SHL_L) CASE(OP_SHR_L)
    fused_l:
#else
    CASE(OP_ADD_L) CASE(OP_SUB_L) CASE(OP_MUL_L) CASE(OP_DIV_L) CASE(OP_MOD_L) CASE(OP_BAND_L)
    CASE(OP_BOR_L) CASE(OP_BXOR_L) CASE(OP_SHL_L) CASE(OP_SHR_L) CASE(OP_EQ_L) CASE(OP_NE_L)
    CASE(OP_LT_L) CASE(OP_LE_L) CASE(OP_GT_L) CASE(OP_GE_L)
#endif
        CHECK_STACK(1);
        *sp++ = bp[*ip++];
        op = FUSED_BINOP[op & 15];
        DISPATCH_OP();

#if TILVM_COMPUTED_GOTO
    L_BAD:
#else
    default:
#endif
        fail(vm, "bad opcode");
        st->failed = 1;
        goto fatal;
#if !TILVM_COMPUTED_GOTO
    }
#endif

backedge:
    /* loop back-edges and calls: one step */
    st->steps++;
    if (--fuel != 0 && !st->n_deadlines) NEXT;
    /* sandbox limit, deadlines and the budget, exactly as TIL 3.0 checked them at
     * every step (the steps since the last REFUEL all passed these checks) */
    if (budget) budget -= fuel_set - fuel - 1;
    if (vm->max_steps && st->steps > vm->max_steps) {
        fail(vm, "step limit exceeded");
        st->failed = 1;
        goto fatal;
    }
    if (st->n_deadlines && ++st->tick >= 256) {
        st->tick = 0;
        target = expired_try(vm);
        if (target >= 0) { REFUEL(); fail(vm, "timeout"); goto panic; }
    }
    if (budget && !--budget) {
        SAVE_PC();
        SYNC_SP();
        return TILVM_YIELD;
    }
    REFUEL();
    NEXT;

stack_overflow:
    fail(vm, "stack overflow");
panic:
    SAVE_PC();
    SYNC_SP();
    if (recover(vm, target)) { target = -1; goto resume; }
    report(vm, (uint32_t)(op_ip - code), "runtime error: ");
    st->failed = 1;
    return TILVM_ERROR;
fatal:
    SAVE_PC();
    SYNC_SP();
    report(vm, (uint32_t)(op_ip - code), "fatal error: ");
    return TILVM_ERROR;
}

#if TILVM_COMPUTED_GOTO && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

int tilvm_run(tilvm *vm) {
    int rc = tilvm_run_steps(vm, 0);
    if (rc == TILVM_EXIT) return vm->st.exit_code;
    return rc == TILVM_DONE ? 0 : 1;
}

/* ─── snapshots ─────────────────────────────────────────────────────────── */
#ifdef TILVM_LEAN
uint32_t tilvm_snapshot_size(const tilvm *vm) { (void)vm; return 0; }
uint32_t tilvm_snapshot(const tilvm *vm, uint8_t *outp, uint32_t cap) { (void)vm; (void)outp; (void)cap; return 0; }
int tilvm_restore(tilvm *vm, const uint8_t *data, uint32_t size) { (void)vm; (void)data; (void)size; return -9; }
#else
#define SNAP_HDR 16

uint32_t tilvm_snapshot_size(const tilvm *vm) {
    return SNAP_HDR + (uint32_t)offsetof(tilvm_state, stack) + vm->st.sp * (uint32_t)sizeof(tv_val)
           + vm->st.heap_top;
}

static void wr32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

uint32_t tilvm_snapshot(const tilvm *vm, uint8_t *outp, uint32_t cap) {
    uint32_t need = tilvm_snapshot_size(vm), fixed = (uint32_t)offsetof(tilvm_state, stack);
    uint32_t stack_bytes = vm->st.sp * (uint32_t)sizeof(tv_val);
    if (cap < need) return 0;
    memcpy(outp, "TILS", 4);
    wr32(outp + 4, vm->bc_hash);
    wr32(outp + 8, (uint32_t)sizeof(tilvm_state) ^ ((uint32_t)TILVM_HEAP_SIZE << 8));
    wr32(outp + 12, need);
    memcpy(outp + SNAP_HDR, &vm->st, fixed);
    memcpy(outp + SNAP_HDR + fixed, vm->st.stack, stack_bytes);
    memcpy(outp + SNAP_HDR + fixed + stack_bytes, vm->mem.heap, vm->st.heap_top);
    return need;
}

int tilvm_restore(tilvm *vm, const uint8_t *data, uint32_t size) {
    uint32_t fixed = (uint32_t)offsetof(tilvm_state, stack), stack_bytes;
    const tilvm_state *st;
    if (size < SNAP_HDR + fixed || memcmp(data, "TILS", 4) != 0) return -1;
    if (RD32(data + 4) != vm->bc_hash) return -2;               /* different program */
    if (RD32(data + 8) != ((uint32_t)sizeof(tilvm_state) ^ ((uint32_t)TILVM_HEAP_SIZE << 8))) return -3;
    if (RD32(data + 12) != size) return -4;
    st = (const tilvm_state *)(const void *)(data + SNAP_HDR);
    memcpy(&vm->st, data + SNAP_HDR, fixed);
    (void)st;
    stack_bytes = vm->st.sp * (uint32_t)sizeof(tv_val);
    if (SNAP_HDR + fixed + stack_bytes + vm->st.heap_top != size) return -4;
    memcpy(vm->st.stack, data + SNAP_HDR + fixed, stack_bytes);
    memcpy(vm->mem.heap, data + SNAP_HDR + fixed + stack_bytes, vm->st.heap_top);
    return 0;
}
#endif /* TILVM_LEAN */
