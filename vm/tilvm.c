/*
 * TIL VM - portable virtual machine for TIL bytecode. See tilvm.h.
 *
 * Semantics follow the reference VM in src/til.py (TILVM class) exactly; the
 * test-suite runs every program on both and compares the output byte for byte.
 */
#include "tilvm.h"
#include <string.h>
#include <limits.h>

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
#if TILVM_FLOAT
#include <float.h>
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

/* vm->flags: the .tbc header flags, and bit 7 = a function is #[energy: low] */
#define TBC_ENERGY_SHIFT 2      /* header bits 2-3: initial energy mode */
#define VMF_ENERGY_LOW 0x80

/* Longest string: 64 KB - 1 with 16-bit heap references, else 16 MB */
#if TILVM_HEAP_SIZE <= 65535L
#define TV_STR_MAX 0xFFFFul
#else
#define TV_STR_MAX 0xFFFFFFul
#endif

/* Length / index of a string, array or map (see TILVM_ADDR16 in tilvm.h).
 * Sums and products of lengths are computed in uint32_t before they are checked. */
#ifdef TILVM_ADDR16
typedef uint16_t tv_len;
#else
typedef uint32_t tv_len;
#endif

/* ─── heap objects ──────────────────────────────────────────────────────── */
enum { OBJ_STR = 1, OBJ_ARRAY, OBJ_MAP, OBJ_VALS, OBJ_STRUCT, OBJ_CLOSURE, OBJ_CELL };

/* TV_PACK (TILVM_PACK, desktop emulation of 8-bit layouts): objects are 2-byte aligned
 * there, so headers with 32-bit sizes (heap > 64 KB) must be packed too. */
typedef struct TV_PACK tv_hdr {
    uint8_t kind;
    uint8_t mark;
    uint16_t aux;       /* struct: type index; closure: upvalue count */
    tv_size size;       /* whole object, bytes */
    tv_ref fwd;         /* forwarding address while compacting */
} tv_hdr;

typedef struct TV_PACK { tv_hdr h; tv_size len; char data[1]; } tv_strobj;
typedef struct TV_PACK { tv_hdr h; tv_size len; tv_ref buf; } tv_arrobj;     /* arrays and maps */
typedef struct TV_PACK { tv_hdr h; tv_val v[1]; } tv_valsobj;               /* capacity = aux */
typedef struct TV_PACK { tv_hdr h; tv_val v[1]; } tv_structobj;             /* fields */
typedef struct TV_PACK { tv_hdr h; uint16_t func; tv_val v[1]; } tv_closobj;  /* upvalues (cells) */
typedef struct TV_PACK { tv_hdr h; tv_val v; } tv_cellobj;

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

#ifdef TILVM_LEAN
/* 8-bit CPUs: one call instead of ~40 bytes of inline shifts per use */
static uint16_t rd16(const uint8_t *p) { return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)rd16(p) | ((uint32_t)rd16(p + 2) << 16); }
#define RD16(p) rd16(p)
#define RD32(p) rd32(p)
#else
#define RD16(p) ((uint16_t)((uint16_t)(p)[0] | ((uint16_t)(p)[1] << 8)))
#define RD32(p) ((uint32_t)RD16(p) | ((uint32_t)RD16((p) + 2) << 16))
#endif

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

/* messages used in many places: a single copy in ROM on 8-bit targets */
static const char E_STACK[] = "stack overflow";
static const char E_STR_LONG[] = "string too long";
static const char E_INDEX_INT[] = "index must be an integer";
static const char E_NO_METHOD[] = "no method '";

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

/* ─── high-water marks (TILVM_STATS, til fit --run) ─────────────────────── */
#ifdef TILVM_STATS
#define TV_PAINT 0xA5u          /* tag of value-stack slots never written */
#define TV_STAT_MAX(field, v) \
    do { if ((uint32_t)(v) > vm->stats.field) vm->stats.field = (uint32_t)(v); } while (0)
#else
#define TV_STAT_MAX(field, v) ((void)0)
#endif

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
    TV_STAT_MAX(peak_heap, vm->st.heap_top);
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

static tv_len obj_nvals(tilvm *vm, tv_ref r, tv_val **vals) {
    tv_hdr *h = OBJ(vm, r);
    switch (h->kind) {
    case OBJ_VALS: *vals = VALSOBJ(vm, r)->v; return h->aux;
    case OBJ_STRUCT: *vals = STRUCTOBJ(vm, r)->v;
        return (tv_len)((h->size - offsetof(tv_structobj, v)) / sizeof(tv_val));
    case OBJ_CLOSURE: *vals = CLOSOBJ(vm, r)->v; return h->aux;
    case OBJ_CELL: *vals = &CELLOBJ(vm, r)->v; return 1;
    default: *vals = 0; return 0;
    }
}

static void scan_children(tilvm *vm, tv_ref r) {
    tv_hdr *h = OBJ(vm, r);
    tv_val *vals;
    tv_len n, i;
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
    tv_len i;
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
        tv_len n, k;
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
    TV_STAT_MAX(peak_live, free_at);
}

uint32_t tilvm_heap_used(const tilvm *vm) { return vm->st.heap_top; }

#ifdef TILVM_STATS
void tilvm_get_stats(const tilvm *vm, tilvm_stats *out) {
    uint32_t k = TILVM_STACK_SIZE;
    *out = vm->stats;
    /* tilvm_load painted the free stack slots: the highest slot ever written is
     * the high-water mark (this also sees temporaries pushed inside builtins) */
    while (k > vm->st.sp && vm->st.stack[k - 1].t == TV_PAINT) k--;
    out->peak_stack = k;
    if (vm->st.n_frames > out->peak_frames) out->peak_frames = vm->st.n_frames;
    if (vm->st.n_try > out->peak_try) out->peak_try = vm->st.n_try;
    if (vm->st.heap_top > out->peak_heap) out->peak_heap = vm->st.heap_top;
    if (vm->st.n_handlers > out->peak_handlers) out->peak_handlers = vm->st.n_handlers;
}
#endif

/* ─── strings ───────────────────────────────────────────────────────────── */
static const uint8_t *const_ptr(tilvm *vm, uint16_t k) { return vm->bc + vm->const_off[k]; }

static int is_str(const tv_val *v) { return v->t == TV_STR || v->t == TV_STRK; }

static const char *str_data(tilvm *vm, const tv_val *v, tv_len *len) {
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
    if (len > TV_STR_MAX) return fail(vm, E_STR_LONG);
    r = new_str(vm, len);
    if (!r) return 0;
    memcpy(STROBJ(vm, r)->data, s, len);
    set_ref(dst, TV_STR, r);
    return 1;
}
#endif

static int str_eq(tilvm *vm, const tv_val *a, const tv_val *b) {
    tv_len la, lb;
    const char *pa = str_data(vm, a, &la), *pb = str_data(vm, b, &lb);
    return la == lb && memcmp(pa, pb, la) == 0;
}

static int str_cmp(tilvm *vm, const tv_val *a, const tv_val *b) {
    tv_len la, lb, n;
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
#define F_FLAGS(vm, f) ((vm)->funcs[f].flags)
#else
#define F_CODE(vm, f) ((vm)->code_off + FN_OFFSET(func_rec(vm, f)))
#define F_NPARAMS(vm, f) FN_NPARAMS(func_rec(vm, f))
#define F_NREQ(vm, f) FN_NREQ(func_rec(vm, f))
#define F_NLOCALS(vm, f) FN_NLOCALS(func_rec(vm, f))
#define F_FLAGS(vm, f) FN_FLAGS(func_rec(vm, f))
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

static void sink_put(sink *s, const char *p, tv_len n) {
    if (s->mode == 0) {
        /* measuring: the text of a value can exceed 4 GB (arrays sharing big
           sub-arrays), so the count stops just above TV_STR_MAX instead of
           wrapping (callers fail with "string too long") */
        s->len = (s->len > TV_STR_MAX || n > TV_STR_MAX - s->len) ? (uint32_t)(TV_STR_MAX + 1) : s->len + n;
        return;
    }
    if (s->mode == 1) memcpy(s->dst + s->len, p, n);
    else if (s->mode == 2) out(s->vm, p, n);
    s->len += n;
}

static void sink_cstr(sink *s, const char *p) { sink_put(s, p, (tv_len)strlen(p)); }

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

#ifndef TILVM_LEAN      /* format specs %.Nf / %.Ne (lean builds ignore specs) */
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
#endif

static void emit_value(sink *s, const tv_val *v, int depth, int quote);

static void emit_type_name(sink *s, uint16_t t) {
    tv_len n;
    tv_val name;
    const char *p;
    name.t = TV_STRK;
    name.u.k = type_name_const(s->vm, t);
    p = str_data(s->vm, &name, &n);
    sink_put(s, p, n);
}

static void emit_const_str(sink *s, uint16_t k) {
    tv_val v;
    tv_len n;
    const char *p;
    v.t = TV_STRK;
    v.u.k = k;
    p = str_data(s->vm, &v, &n);
    sink_put(s, p, n);
}

static void emit_value(sink *s, const tv_val *v, int depth, int quote) {
    tilvm *vm = s->vm;
    char buf[48];
    tv_len n, i;
    if (s->mode == 0 && s->len > TV_STR_MAX) return;   /* already too long: stop the walk */
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
        tv_len len = a->len;
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
    if (s.len > TV_STR_MAX) return fail(vm, E_STR_LONG);
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
    tv_len n;
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

static int map_find(tilvm *vm, tv_ref map, const tv_val *key);

static int equal(tilvm *vm, const tv_val *a, const tv_val *b, int depth) {
    tv_len i, n;
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
        for (i = 0; i < n; i++) {     /* maps: same keys (scalars), equal values */
            int j = map_find(vm, b->u.r, &VALS(vm, x->buf)[2 * i]);
            if (j < 0 || !equal(vm, &VALS(vm, x->buf)[2 * i + 1], &VALS(vm, y->buf)[2 * j + 1], depth + 1))
                return 0;
        }
        return 1;
    }
    if (a->t == TV_STRUCT) {
        tv_hdr *ha = OBJ(vm, a->u.r), *hb = OBJ(vm, b->u.r);
        if (ha->aux != hb->aux) return 0;
        n = (tv_len)((ha->size - offsetof(tv_structobj, v)) / sizeof(tv_val));
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
        tv_len la, lb;
        tv_ref r;
        if (!to_str_slot(vm, (uint16_t)(SP - 2)) || !to_str_slot(vm, (uint16_t)(SP - 1))) return 0;
        str_data(vm, &STK[SP - 2], &la);
        str_data(vm, &STK[SP - 1], &lb);
        if ((uint32_t)la + lb > TV_STR_MAX) return fail(vm, E_STR_LONG);
        r = new_str(vm, (uint32_t)la + lb);
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
        tv_len la;
        uint32_t k, times = b->u.i > 0 ? (uint32_t)b->u.i : 0;
        tv_ref r;
        str_data(vm, a, &la);
#ifdef TILVM_ADDR16     /* 16-bit lengths: no 32-bit multiply / divide routines */
        if (times > TV_STR_MAX ? la != 0 : times && la > (tv_len)0xFFFFu / (tv_len)times)
            return fail(vm, E_STR_LONG);
        r = new_str(vm, (tv_len)(la * (tv_len)times));
#else
        if (times && la > TV_STR_MAX / times) return fail(vm, E_STR_LONG);
        r = new_str(vm, la * times);
#endif
        if (!r) return 0;
        {
            const char *pa = str_data(vm, &STK[SP - 2], &la);
            char *d = STROBJ(vm, r)->data;
            for (k = 0; k < times; k++, d += la) memcpy(d, pa, la);
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
#ifdef TILVM_ADDR16     /* larger than the heap anyway; the product then fits 16 bits */
    if (cap > TILVM_HEAP_SIZE / sizeof(tv_val)) { fail(vm, "out of memory"); return 0; }
    r = alloc_obj(vm, OBJ_VALS, (uint32_t)offsetof(tv_valsobj, v) + (uint32_t)((tv_len)cap * sizeof(tv_val)));
#else
    r = alloc_obj(vm, OBJ_VALS, (uint32_t)offsetof(tv_valsobj, v) + cap * (uint32_t)sizeof(tv_val));
#endif
    if (!r) return 0;
    OBJ(vm, r)->aux = (uint16_t)cap;
    for (i = 0; i < cap; i++) set_nil(&VALS(vm, r)[i]);
    return r;
}

/* Push a new array (or map) with room for `nvals` values and length `len`;
 * the caller fills the values. Results built this way never stage their
 * elements on the value stack, so they work up to the 65535-element limit
 * whatever TILVM_STACK_SIZE is. */
static int push_empty(tilvm *vm, uint8_t kind, uint32_t nvals, uint32_t len) {
    tv_ref buf, arr;
    if (SP + 2 > TILVM_STACK_SIZE) return fail(vm, E_STACK);
    buf = new_vals(vm, nvals < 4 ? 4 : nvals);
    if (!buf) return 0;
    set_ref(&STK[SP], TV_CELL, buf);          /* keep the buffer rooted */
    SP++;
    arr = alloc_obj(vm, kind, sizeof(tv_arrobj));
    SP--;
    if (!arr) return 0;
    ARROBJ(vm, arr)->buf = STK[SP].u.r;
    ARROBJ(vm, arr)->len = (tv_size)len;
    set_ref(&STK[SP], kind == OBJ_MAP ? TV_MAP : TV_ARRAY, arr);
    SP++;
    return 1;
}

/* Replace the top `n` stack values by a new array / map of them. */
static int make_array(tilvm *vm, uint8_t kind, uint32_t n) {
    uint32_t nvals = kind == OBJ_MAP ? 2 * n : n;
    uint16_t k;
    if (!push_empty(vm, kind, nvals, n)) return 0;   /* above the values: they stay rooted */
    k = (uint16_t)nvals;                             /* <= 0xFFFF (new_vals checked it) */
    memcpy(VALS(vm, ARROBJ(vm, STK[SP - 1].u.r)->buf), &STK[SP - 1 - k], k * sizeof(tv_val));
    SP = (uint16_t)(SP - k);
    STK[SP - 1] = STK[SP + k - 1];
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
/* [.., a, b] -> [.., a+b].  The elements are copied heap to heap: no value-stack
 * slots beyond a and b, whatever the lengths (til fit relies on this). */
static int tv_array_concat(tilvm *vm) {
    uint32_t la = ARROBJ(vm, STK[SP - 2].u.r)->len, lb = ARROBJ(vm, STK[SP - 1].u.r)->len, i;
    tv_ref buf, arr;
    buf = new_vals(vm, la + lb < 4 ? 4 : la + lb);     /* a and b stay rooted on the stack */
    if (!buf) return 0;
    {
        tv_val *dst = VALS(vm, buf);
        const tv_val *src = VALS(vm, ARROBJ(vm, STK[SP - 2].u.r)->buf);
        for (i = 0; i < la; i++) dst[i] = src[i];
        src = VALS(vm, ARROBJ(vm, STK[SP - 1].u.r)->buf);
        for (i = 0; i < lb; i++) dst[la + i] = src[i];
    }
    set_ref(&STK[SP - 1], TV_CELL, buf);      /* b is copied: the buffer now roots the elements */
    arr = alloc_obj(vm, OBJ_ARRAY, sizeof(tv_arrobj));
    if (!arr) return 0;
    ARROBJ(vm, arr)->buf = STK[SP - 1].u.r;  /* re-read after a possible collection */
    ARROBJ(vm, arr)->len = (tv_size)(la + lb);
    set_ref(&STK[SP - 2], TV_ARRAY, arr);
    SP--;
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
    tv_len i, n = m->len;
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
        tv_len kl, l;
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
    tv_len j, n = m->len;
    tv_val *v = VALS(vm, m->buf);
    for (j = (tv_len)i; j + 1 < n; j++) { v[2 * j] = v[2 * j + 2]; v[2 * j + 1] = v[2 * j + 3]; }
    set_nil(&v[2 * (n - 1)]);
    set_nil(&v[2 * (n - 1) + 1]);
    m->len = (tv_size)(n - 1);
}

static int key_not_found(tilvm *vm, uint16_t key_slot) {
    tv_len pos = 0, cap = sizeof(vm->errbuf) - 1;
    sink s;
    const char *pre = "key not found: ";
    while (*pre && pos < cap) vm->errbuf[pos++] = *pre++;
    s.vm = vm; s.mode = 0; s.dst = 0; s.len = 0;
    emit_value(&s, &STK[key_slot], 0, 0);
    if (s.len <= (uint32_t)(cap - pos)) {
        s.mode = 1; s.dst = vm->errbuf + pos; s.len = 0;
        emit_value(&s, &STK[key_slot], 0, 0);
        pos += s.len;
    }
    vm->errbuf[pos] = 0;
    return fail(vm, vm->errbuf);
}

static int bounds_error(tilvm *vm, tv_int idx, tv_len len) {
    return failf(vm, "index out of bounds: index ", idx, ", length ", (tv_int)len, "");
}

/* [.., obj, idx] -> [.., obj[idx]] */
static int get_index(tilvm *vm) {
    tv_val *o = &STK[SP - 2], *ix = &STK[SP - 1];
    if (o->t == TV_ARRAY || is_str(o)) {
        tv_len len;
        const char *p = 0;
        if (ix->t != TV_INT) return fail(vm, E_INDEX_INT);
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
        tv_len n = (tv_len)((h->size - offsetof(tv_structobj, v)) / sizeof(tv_val));
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
        tv_len len = ARROBJ(vm, o->u.r)->len;
        if (ix->t != TV_INT) return fail(vm, E_INDEX_INT);
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
    tv_len tl = (tv_len)strlen(tn);
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
    tv_ref r = alloc_obj(vm, OBJ_STRUCT, (uint32_t)offsetof(tv_structobj, v) + (uint32_t)(n * sizeof(tv_val)));
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
static int parse_int(const char *p, tv_len n, tv_int *out_i) {
    tv_len i = 0;
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
/* float(str) - the syntax of the reference VM (Python's float() of the bytes):
 * ASCII whitespace around [+-] (digits [. [digits]] | . digits) [(e|E) [+-]
 * digits], or [+-] inf / infinity / nan in any case, '_' only between two
 * digits. The value is the correctly rounded (ties to even) value of the
 * decimal, computed with exact integer arithmetic, so every build gives the
 * same float, without the C library. The first TV_FDIG significant digits are
 * kept and any further nonzero digit counts as one more digit 1; this rounds
 * like the whole string because no halfway point between two floats has more
 * than 768 (double) or 113 (float) significant digits. A smaller
 * TILVM_FLOAT_DIGITS saves stack (about 2 bytes per digit) on tiny devices
 * at the price of exactness for longer inputs. */
#if TILVM_FLOAT == 1
#define TV_F_MANT FLT_MANT_DIG
#define TV_F_EMIN (FLT_MIN_EXP - FLT_MANT_DIG)    /* exponent of the smallest subnormal */
#define TV_F_EMAX FLT_MAX_EXP                     /* floats are < 2^TV_F_EMAX */
#define TV_F_HUGE FLT_MAX
typedef uint32_t tv_fmant;
#else
#define TV_F_MANT DBL_MANT_DIG
#define TV_F_EMIN (DBL_MIN_EXP - DBL_MANT_DIG)
#define TV_F_EMAX DBL_MAX_EXP
#define TV_F_HUGE DBL_MAX
typedef uint64_t tv_fmant;
#endif
#ifdef TILVM_FLOAT_DIGITS
#define TV_FDIG TILVM_FLOAT_DIGITS
#elif TILVM_FLOAT == 1
#define TV_FDIG 120
#else
#define TV_FDIG 770
#endif
/* 16-bit limbs for 10^(TV_FDIG + 1 - smallest decimal exponent) * 2^(mantissa + 2) */
#define TV_BIG_LIMBS \
    (((TV_FDIG + 3 + (1L - TV_F_EMIN) * 30103L / 100000L) * 3322L / 1000L + TV_F_MANT + 40) / 16)

typedef struct tv_dec {
    uint16_t nd;                /* significant digits, the first is not 0 */
    uint8_t d[TV_FDIG + 1];     /* their values */
    long e;                     /* value = (d as an integer) * 10^e */
} tv_dec;

typedef struct tv_big { uint16_t n, w[TV_BIG_LIMBS]; } tv_big;   /* little-endian */

static int f_space(char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

static int f_word(const char *p, uint32_t n, const char *w) {   /* w: lower case */
    uint32_t i;
    for (i = 0; i < n; i++)
        if (!w[i] || (p[i] | 0x20) != w[i]) return 0;
    return !w[n];
}

/* digits with '_' only between two digits: part 0 integer, 1 fraction (into dec), 2 exponent (ex) */
static uint32_t f_digits(const char *p, uint32_t i, uint32_t n, tv_dec *dec, int part, int *any, int *sticky,
                         long *ex) {
    int prev = 0;
    for (; i < n; i++) {
        char c = p[i];
        if (c == '_' && prev && i + 1 < n && p[i + 1] >= '0' && p[i + 1] <= '9') { prev = 0; continue; }
        if (c < '0' || c > '9') break;
        prev = *any = 1;
        if (part == 2) {                                  /* exponent */
            if (*ex < 100000L) *ex = *ex * 10 + (c - '0');
        } else if (dec->nd < TV_FDIG && (dec->nd || c != '0')) {
            dec->d[dec->nd++] = (uint8_t)(c - '0');
            if (part) dec->e--;
        } else if (!dec->nd) {                            /* a leading zero */
            if (part) dec->e--;
        } else {                                          /* beyond TV_FDIG digits */
            if (c != '0') *sticky = 1;
            if (!part) dec->e++;
        }
    }
    return i;
}

/* 0 = not a number, 1 = finite (dec), 2 = inf, 3 = nan */
static int scan_float(const char *p, uint32_t n, int *neg, tv_dec *dec) {
    uint32_t i = 0;
    int any = 0, sticky = 0;
    long ex = 0;
    while (i < n && f_space(p[i])) i++;
    while (n > i && f_space(p[n - 1])) n--;
    *neg = 0;
    if (i < n && (p[i] == '+' || p[i] == '-')) *neg = p[i++] == '-';
    if (f_word(p + i, n - i, "inf") || f_word(p + i, n - i, "infinity")) return 2;
    if (f_word(p + i, n - i, "nan")) return 3;
    dec->nd = 0;
    dec->e = 0;
    i = f_digits(p, i, n, dec, 0, &any, &sticky, &ex);
    if (i < n && p[i] == '.') i = f_digits(p, i + 1, n, dec, 1, &any, &sticky, &ex);
    if (!any) return 0;
    if (i < n && (p[i] == 'e' || p[i] == 'E')) {
        int eneg = 0, edig = 0;
        if (++i < n && (p[i] == '+' || p[i] == '-')) eneg = p[i++] == '-';
        i = f_digits(p, i, n, dec, 2, &edig, &sticky, &ex);
        if (!edig) return 0;
        if (eneg) ex = -ex;
    }
    if (i != n) return 0;
    if (sticky) { dec->d[dec->nd++] = 1; dec->e--; }
    dec->e += ex;
    return 1;
}

static int big_bits(const tv_big *b) {
    int bits = b->n ? (b->n - 1) * 16 : 0;
    uint16_t top = b->n ? b->w[b->n - 1] : 0;
    while (top) { bits++; top = (uint16_t)(top >> 1); }
    return bits;
}

static void big_muladd(tv_big *b, uint32_t m, uint32_t c) {   /* b = b * m + c (m, c < 65536) */
    int i;
    for (i = 0; i < b->n; i++) {
        uint32_t t = (uint32_t)b->w[i] * m + c;
        b->w[i] = (uint16_t)t;
        c = t >> 16;
    }
    if (c) b->w[b->n++] = (uint16_t)c;
}

static void big_pow10(tv_big *b, long k) {                   /* b *= 10^k */
    for (; k >= 4; k -= 4) big_muladd(b, 10000, 0);
    for (; k > 0; k--) big_muladd(b, 10, 0);
}

static void big_shl(tv_big *b, long s) {                     /* b <<= s */
    int limbs = (int)(s / 16), bits = (int)(s % 16), i;
    if (!b->n) return;
    if (bits) {
        uint32_t c = 0;
        for (i = 0; i < b->n; i++) {
            uint32_t t = ((uint32_t)b->w[i] << bits) | c;
            b->w[i] = (uint16_t)t;
            c = t >> 16;
        }
        if (c) b->w[b->n++] = (uint16_t)c;
    }
    if (limbs) {
        for (i = b->n - 1; i >= 0; i--) b->w[i + limbs] = b->w[i];
        for (i = 0; i < limbs; i++) b->w[i] = 0;
        b->n = (uint16_t)(b->n + limbs);
    }
}

static void big_shr1(tv_big *b) {
    int i;
    for (i = 0; i < b->n; i++)
        b->w[i] = (uint16_t)((b->w[i] >> 1) | (i + 1 < b->n ? (b->w[i + 1] & 1) << 15 : 0));
    while (b->n && !b->w[b->n - 1]) b->n--;
}

static int big_cmp(const tv_big *a, const tv_big *b) {
    int i;
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (i = a->n - 1; i >= 0; i--)
        if (a->w[i] != b->w[i]) return a->w[i] < b->w[i] ? -1 : 1;
    return 0;
}

static void big_sub(tv_big *a, const tv_big *b) {           /* a -= b (a >= b) */
    int i;
    uint32_t borrow = 0;
    for (i = 0; i < a->n; i++) {
        uint32_t t = (uint32_t)(i < b->n ? b->w[i] : 0) + borrow;
        borrow = a->w[i] < t;
        a->w[i] = (uint16_t)(a->w[i] - t);
    }
    while (a->n && !a->w[a->n - 1]) a->n--;
}

/* the float nearest to the decimal (ties to even) */
static tv_float dec_to_float(const tv_dec *dec) {
    const long lo = -(((1L - TV_F_EMIN) * 30103L + 99999L) / 100000L);  /* 10^lo < half the smallest float */
    const long hi = (long)TV_F_EMAX * 30103L / 100000L + 1;             /* 10^(hi) > the largest float */
    long mag = (long)dec->nd + dec->e, s, e2;
    tv_big num, den;
    tv_fmant q = 0, m, half, low;
    int i, qbits = 0, extra, sticky;
    tv_float x;
    if (!dec->nd || mag <= lo) return 0;
    if (mag > hi) { x = TV_F_HUGE; return x + x; }           /* inf */
    num.n = 0;
    for (i = 0; i < dec->nd; i++) big_muladd(&num, 10, dec->d[i]);
    den.n = 1;
    den.w[0] = 1;
    if (dec->e > 0) big_pow10(&num, dec->e);
    else big_pow10(&den, -dec->e);
    /* q = floor(num / (den * 2^s)) has TV_F_MANT + 1 or + 2 bits */
    s = (long)big_bits(&num) - big_bits(&den) - (TV_F_MANT + 1);
    if (s > 0) big_shl(&den, s);
    else big_shl(&num, -s);
    big_shl(&den, TV_F_MANT + 1);
    for (i = 0; i <= TV_F_MANT + 1; i++) {
        q <<= 1;
        if (big_cmp(&num, &den) >= 0) { big_sub(&num, &den); q |= 1; }
        big_shr1(&den);
    }
    sticky = num.n != 0;
    for (m = q; m; m >>= 1) qbits++;
    extra = qbits - TV_F_MANT;                               /* bits to round off */
    if (s + extra < TV_F_EMIN) extra = (int)(TV_F_EMIN - s);    /* subnormal */
    if (extra > qbits) return 0;
    half = (tv_fmant)1 << (extra - 1);
    low = q & ((half << 1) - 1);
    m = q >> extra;
    if (low > half || (low == half && (sticky || (m & 1)))) m++;
    e2 = s + extra;
    for (qbits = 0, q = m; q; q >>= 1) qbits++;
    if (qbits + e2 > TV_F_EMAX) { x = TV_F_HUGE; return x + x; }
    x = (tv_float)m;                                         /* exact; scaling by 2^e2 is exact too */
    for (; e2 >= 16; e2 -= 16) x *= (tv_float)65536.0;
    for (; e2 > 0; e2--) x *= 2;
    for (; e2 <= -16; e2 += 16) x *= (tv_float)1.52587890625e-05;
    for (; e2 < 0; e2++) x *= (tv_float)0.5;
    return x;
}

static int parse_float(const char *p, uint32_t n, tv_float *out_f) {
    tv_dec dec;
    int neg, kind = scan_float(p, n, &neg, &dec);
    tv_float x;
    if (!kind) return 0;
    if (kind == 1) x = dec_to_float(&dec);
    else {
        x = TV_F_HUGE;
        x = x + x;                                           /* inf */
        if (kind == 3) x = x - x;                            /* nan */
    }
    *out_f = neg ? -x : x;
    return 1;
}
#endif

static int invalid_number(tilvm *vm, uint16_t slot, const char *what) {
    tv_len n, pos = 0, cap = sizeof(vm->errbuf) - 4;
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
            tv_len n;
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
        tv_len n;
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
    tv_len slen, blen;
    uint32_t pad, i, width = 0;
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
        if (b) { src = body; blen = (tv_len)strlen(body); }
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
    tv_len n;
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
    if (SP + 2 > TILVM_STACK_SIZE) return fail(vm, E_STACK);
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
    tv_len n;
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

#ifndef TILVM_NO_GFX
/* ─── screen and keys (TIL 3.1, docs/TIL_GRAPHICS.md) ──────────────────── */
/* The VM rasterizes shapes and text and clips them to the screen, so every
 * host shows identical pixels; hosts only store pixels (vm->host.gfx_*).
 * TILVM.gfx_native in src/til.py implements the same algorithms, limits and
 * messages. */
#ifndef TILVM_ROM
#define TILVM_ROM
#endif

/* screen_init: 1..TILVM_GFX_MAX_W x 1..TILVM_GFX_MAX_H pixels (tilvm.h); 4096 per
 * side is the language limit, a device with a fixed display sets its size */
#if TILVM_GFX_MAX_W < 1 || TILVM_GFX_MAX_W > 4096 || TILVM_GFX_MAX_H < 1 || TILVM_GFX_MAX_H > 4096
#error "TILVM_GFX_MAX_W and TILVM_GFX_MAX_H must be 1..4096"
#endif
#define GFX_COORD_LIM 32768L        /* shape coordinates and sizes: -32768..32767 */

/* 5x7 font: ASCII 32..126, then the replacement glyph used for every other
 * character. 5 columns per glyph, bit 0 = top row. Same table as FONT5X7 in
 * src/til.py (the test-suite compares them). */
static const TILVM_ROM uint8_t FONT5X7[96 * 5] = {
    0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x4F,0x00,0x00, 0x00,0x07,0x00,0x07,0x00, 0x14,0x7F,0x14,0x7F,0x14, /* sp ! " # */
    0x24,0x2A,0x7F,0x2A,0x12, 0x23,0x13,0x08,0x64,0x62, 0x36,0x49,0x55,0x22,0x50, 0x00,0x05,0x03,0x00,0x00, /* $ % & ' */
    0x00,0x1C,0x22,0x41,0x00, 0x00,0x41,0x22,0x1C,0x00, 0x14,0x08,0x3E,0x08,0x14, 0x08,0x08,0x3E,0x08,0x08, /* ( ) * + */
    0x00,0x50,0x30,0x00,0x00, 0x08,0x08,0x08,0x08,0x08, 0x00,0x60,0x60,0x00,0x00, 0x20,0x10,0x08,0x04,0x02, /* , - . / */
    0x3E,0x51,0x49,0x45,0x3E, 0x00,0x42,0x7F,0x40,0x00, 0x42,0x61,0x51,0x49,0x46, 0x21,0x41,0x45,0x4B,0x31, /* 0 1 2 3 */
    0x18,0x14,0x12,0x7F,0x10, 0x27,0x45,0x45,0x45,0x39, 0x3C,0x4A,0x49,0x49,0x30, 0x01,0x71,0x09,0x05,0x03, /* 4 5 6 7 */
    0x36,0x49,0x49,0x49,0x36, 0x06,0x49,0x49,0x29,0x1E, 0x00,0x36,0x36,0x00,0x00, 0x00,0x56,0x36,0x00,0x00, /* 8 9 : ; */
    0x08,0x14,0x22,0x41,0x00, 0x14,0x14,0x14,0x14,0x14, 0x00,0x41,0x22,0x14,0x08, 0x02,0x01,0x51,0x09,0x06, /* < = > ? */
    0x32,0x49,0x79,0x41,0x3E, 0x7E,0x11,0x11,0x11,0x7E, 0x7F,0x49,0x49,0x49,0x36, 0x3E,0x41,0x41,0x41,0x22, /* @ A B C */
    0x7F,0x41,0x41,0x22,0x1C, 0x7F,0x49,0x49,0x49,0x41, 0x7F,0x09,0x09,0x09,0x01, 0x3E,0x41,0x49,0x49,0x7A, /* D E F G */
    0x7F,0x08,0x08,0x08,0x7F, 0x00,0x41,0x7F,0x41,0x00, 0x20,0x40,0x41,0x3F,0x01, 0x7F,0x08,0x14,0x22,0x41, /* H I J K */
    0x7F,0x40,0x40,0x40,0x40, 0x7F,0x02,0x0C,0x02,0x7F, 0x7F,0x04,0x08,0x10,0x7F, 0x3E,0x41,0x41,0x41,0x3E, /* L M N O */
    0x7F,0x09,0x09,0x09,0x06, 0x3E,0x41,0x51,0x21,0x5E, 0x7F,0x09,0x19,0x29,0x46, 0x46,0x49,0x49,0x49,0x31, /* P Q R S */
    0x01,0x01,0x7F,0x01,0x01, 0x3F,0x40,0x40,0x40,0x3F, 0x1F,0x20,0x40,0x20,0x1F, 0x3F,0x40,0x38,0x40,0x3F, /* T U V W */
    0x63,0x14,0x08,0x14,0x63, 0x07,0x08,0x70,0x08,0x07, 0x61,0x51,0x49,0x45,0x43, 0x00,0x7F,0x41,0x41,0x00, /* X Y Z [ */
    0x02,0x04,0x08,0x10,0x20, 0x00,0x41,0x41,0x7F,0x00, 0x04,0x02,0x01,0x02,0x04, 0x40,0x40,0x40,0x40,0x40, /* bs ] ^ _ */
    0x00,0x01,0x02,0x04,0x00, 0x20,0x54,0x54,0x54,0x78, 0x7F,0x48,0x44,0x44,0x38, 0x38,0x44,0x44,0x44,0x20, /* ` a b c */
    0x38,0x44,0x44,0x48,0x7F, 0x38,0x54,0x54,0x54,0x18, 0x08,0x7E,0x09,0x01,0x02, 0x08,0x54,0x54,0x54,0x3C, /* d e f g */
    0x7F,0x08,0x04,0x04,0x78, 0x00,0x44,0x7D,0x40,0x00, 0x20,0x40,0x44,0x3D,0x00, 0x7F,0x10,0x28,0x44,0x00, /* h i j k */
    0x00,0x41,0x7F,0x40,0x00, 0x7C,0x04,0x18,0x04,0x78, 0x7C,0x08,0x04,0x04,0x78, 0x38,0x44,0x44,0x44,0x38, /* l m n o */
    0x7C,0x14,0x14,0x14,0x08, 0x08,0x14,0x14,0x18,0x7C, 0x7C,0x08,0x04,0x04,0x08, 0x48,0x54,0x54,0x54,0x20, /* p q r s */
    0x04,0x3F,0x44,0x40,0x20, 0x3C,0x40,0x40,0x20,0x7C, 0x1C,0x20,0x40,0x20,0x1C, 0x3C,0x40,0x30,0x40,0x3C, /* t u v w */
    0x44,0x28,0x10,0x28,0x44, 0x0C,0x50,0x50,0x50,0x3C, 0x44,0x64,0x54,0x4C,0x44, 0x00,0x08,0x36,0x41,0x00, /* x y z { */
    0x00,0x00,0x7F,0x00,0x00, 0x00,0x41,0x36,0x08,0x00, 0x08,0x04,0x08,0x10,0x08, 0x7F,0x41,0x41,0x41,0x7F  /* | } ~ replacement */
};

/* Argument kinds of screen_init .. screen_show: i = any int, c = coordinate or
 * size -32768..32767, s = screen size 1..4096, k = color, t = any value (text). */
static const char *const GFX_KINDS[] = {
    "ss", "k", "iik", "cccck", "cccck", "cccck", "ccck", "cctk", "ii", ""
};

static void gfx_plot(tilvm *vm, long x, long y, unsigned long rgb) {
    if (x >= 0 && y >= 0 && x < (long)vm->st.scr_w && y < (long)vm->st.scr_h)
        vm->host.gfx_pixel(vm->ctx, (int)x, (int)y, rgb);
}

/* Fill [x0, x1) x [y0, y1), clipped to the screen. */
static void gfx_fill(tilvm *vm, long x0, long y0, long x1, long y1, unsigned long rgb) {
    long x;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (long)vm->st.scr_w) x1 = (long)vm->st.scr_w;
    if (y1 > (long)vm->st.scr_h) y1 = (long)vm->st.scr_h;
    for (; y0 < y1; y0++)
        for (x = x0; x < x1; x++) vm->host.gfx_pixel(vm->ctx, (int)x, (int)y0, rgb);
}

/* Bresenham line, all octants, both end points included. */
static void gfx_line(tilvm *vm, long x0, long y0, long x1, long y1, unsigned long rgb) {
    long dx = x1 > x0 ? x1 - x0 : x0 - x1, dy = y1 > y0 ? y0 - y1 : y1 - y0;   /* dy = -|y1 - y0| */
    long sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy, e2;
    for (;;) {
        gfx_plot(vm, x0, y0, rgb);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

/* Midpoint circle outline (r < 0: nothing, r = 0: one pixel). */
static void gfx_circle(tilvm *vm, long cx, long cy, long r, unsigned long rgb) {
    long x = 0, y = r, d = 1 - r;
    while (x <= y) {
        gfx_plot(vm, cx + x, cy + y, rgb);
        gfx_plot(vm, cx - x, cy + y, rgb);
        gfx_plot(vm, cx + x, cy - y, rgb);
        gfx_plot(vm, cx - x, cy - y, rgb);
        gfx_plot(vm, cx + y, cy + x, rgb);
        gfx_plot(vm, cx - y, cy + x, rgb);
        gfx_plot(vm, cx + y, cy - x, rgb);
        gfx_plot(vm, cx - y, cy - x, rgb);
        if (d < 0) d += 2 * x + 3;
        else { d += 2 * (x - y) + 5; y--; }
        x++;
    }
}

/* Text: one 5x7 glyph per character, 6 px advance, "\n" starts a new line 8 px
 * lower. A character is a byte below 0x80 or a UTF-8 lead byte (continuation
 * bytes 0x80-0xBF are skipped); everything outside ASCII 32..126 is drawn as
 * the replacement glyph. */
static void gfx_text(tilvm *vm, long x, long y, const uint8_t *s, uint32_t n, unsigned long rgb) {
    long cx = x, cy = y;
    uint32_t i;
    for (i = 0; i < n; i++) {
        unsigned c = s[i];
        const TILVM_ROM uint8_t *g;
        int col, row;
        if (c == '\n') { cx = x; cy += 8; continue; }
        if (c >= 0x80 && c < 0xC0) continue;
        g = FONT5X7 + 5u * (c >= 32 && c <= 126 ? c - 32 : 95);
        if (cx < (long)vm->st.scr_w && cx > -5 && cy < (long)vm->st.scr_h && cy > -7)
            for (col = 0; col < 5; col++)
                for (row = 0; row < 7; row++)
                    if ((g[col] >> row) & 1) gfx_plot(vm, cx + col, cy + row, rgb);
        cx += 6;
    }
}

/* Color argument -> 0xRRGGBB for the host. 16-bit int builds use RGB565
 * values (the int's 16 bits; e.g. red 0xF800 is -2048). */
static unsigned long gfx_rgb(const tv_val *v) {
#if TILVM_INT_BITS == 16
    unsigned long c = (uint16_t)v->u.i, r = c >> 11, g = (c >> 5) & 63u, b = c & 31u;
    return (((r << 3) | (r >> 2)) << 16) | (((g << 2) | (g >> 4)) << 8) | ((b << 3) | (b >> 2));
#else
    return (unsigned long)v->u.i;
#endif
}

/* Host pixel -> get_pixel() value (RGB565 on 16-bit int builds). */
static tv_int gfx_value(unsigned long rgb) {
#if TILVM_INT_BITS == 16
    long c = (long)((((rgb >> 19) & 31u) << 11) | (((rgb >> 10) & 63u) << 5) | ((rgb >> 3) & 31u));
    return (tv_int)(c > 32767L ? c - 65536L : c);
#else
    return (tv_int)(rgb & 0xFFFFFFul);
#endif
}

static int gfx_arg_error(tilvm *vm, uint8_t id, uint8_t k, const char *what, const char *got) {
    char num[2];
    num[0] = (char)('1' + k);
    num[1] = 0;
    return failn(vm, NATIVE_NAMES[id], "() argument ", num, what, got, 0);
}

/* "... must be an int, got <type>". A struct is named by its type, like the
 * reference VM: the name stays the program's constant string (err_val, printed
 * after err_msg), so it is never cut, whatever its length. */
static int gfx_int_error(tilvm *vm, uint8_t id, uint8_t k, const tv_val *v) {
    gfx_arg_error(vm, id, k, " must be an int, got ", v->t == TV_STRUCT ? 0 : type_name(vm, v));
    if (v->t == TV_STRUCT) {
        vm->err_val.t = TV_STRK;
        vm->err_val.u.i = 0;
        vm->err_val.u.k = type_name_const(vm, OBJ(vm, v->u.r)->aux);
    }
    return 0;
}

static int gfx_check(tilvm *vm, uint8_t id, uint16_t base, uint8_t argc) {
    const char *kinds = GFX_KINDS[id - N_SCREEN_INIT];
    char buf[32], lim[32];
    uint8_t k;
    for (k = 0; kinds[k] && k < argc; k++) {
        const tv_val *v = &STK[base + k];
        tv_int i;
        if (kinds[k] == 't') continue;
        if (v->t != TV_INT) return gfx_int_error(vm, id, k, v);
        i = v->u.i;
        if (kinds[k] == 's' && (i < 1 || i > (k ? TILVM_GFX_MAX_H : TILVM_GFX_MAX_W)))   /* screen_init */
            return failn(vm, k ? "screen_init() argument 2 must be 1.." : "screen_init() argument 1 must be 1..",
                         fmt_int(lim, (tv_int)(k ? TILVM_GFX_MAX_H : TILVM_GFX_MAX_W)), ", got ",
                         fmt_int(buf, i), 0, 0);
#if TILVM_INT_BITS > 16
        if (kinds[k] == 'c' && (i < -GFX_COORD_LIM || i >= GFX_COORD_LIM))
            return gfx_arg_error(vm, id, k, " must be -32768..32767, got ", fmt_int(buf, i));
        if (kinds[k] == 'k' && (i < 0 || i > 0xFFFFFFL))
            return gfx_arg_error(vm, id, k, " must be a color 0..0xFFFFFF, got ", fmt_int(buf, i));
#endif
    }
    return 1;
}

/* Create the host screen (all black): screen_init, and after a snapshot
 * restore (snapshots keep the size, not the pixels). */
static void gfx_open(tilvm *vm) {
    if (vm->host.gfx_init) vm->host.gfx_init(vm->ctx, (int)vm->st.scr_w, (int)vm->st.scr_h);
    else gfx_fill(vm, 0, 0, (long)vm->st.scr_w, (long)vm->st.scr_h, 0);
}

static int gfx_key(tilvm *vm, uint8_t id, tv_val *res) {
    int k;
    if (id == N_WAIT_KEY && vm->host.wait_key) k = vm->host.wait_key(vm->ctx);
    else if (!vm->host.key_pressed)
        return failn(vm, NATIVE_NAMES[id], "() is not available on this device", 0, 0, 0, 0);
    else {
        k = vm->host.key_pressed(vm->ctx);
        while (id == N_WAIT_KEY && k == 0 && vm->st.halted != 2) {   /* no wait_key hook: poll */
            if (vm->host.idle) vm->host.idle(vm->ctx, 1);
            k = vm->host.key_pressed(vm->ctx);
        }
    }
    if (k < 0) return fail(vm, "key code out of range");
#if TILVM_INT_BITS == 16 && INT_MAX > 32767
    if (k > 32767) return fail(vm, "key code out of range");
#endif
    set_int(res, (tv_int)k);
    return 1;
}

static int native_gfx(tilvm *vm, uint8_t id, uint16_t base, uint8_t argc, tv_val *res) {
    tv_val *a = &STK[base];
    long w = (long)vm->st.scr_w, h = (long)vm->st.scr_h;
    switch (id) {
    case N_SCREEN_WIDTH: set_int(res, (tv_int)w); return 1;
    case N_SCREEN_HEIGHT: set_int(res, (tv_int)h); return 1;
    case N_KEY_PRESSED: case N_WAIT_KEY: return gfx_key(vm, id, res);
    default: break;
    }
    if (!vm->host.gfx_pixel || (id == N_GET_PIXEL && !vm->host.gfx_get))
        return failn(vm, NATIVE_NAMES[id], "() is not available on this device", 0, 0, 0, 0);
    if (id != N_SCREEN_INIT && !w) return fail(vm, "screen not initialized (call screen_init)");
    if (!gfx_check(vm, id, base, argc)) return 0;
    switch (id) {
    case N_SCREEN_INIT:
        vm->st.scr_w = (uint16_t)a[0].u.i;
        vm->st.scr_h = (uint16_t)a[1].u.i;
        gfx_open(vm);
        break;
    case N_SCREEN_CLEAR: gfx_fill(vm, 0, 0, w, h, argc ? gfx_rgb(a) : 0); break;
    case N_DRAW_PIXEL:
        if (a[0].u.i >= 0 && a[0].u.i < (tv_int)w && a[1].u.i >= 0 && a[1].u.i < (tv_int)h)
            vm->host.gfx_pixel(vm->ctx, (int)a[0].u.i, (int)a[1].u.i, gfx_rgb(a + 2));
        break;
    case N_GET_PIXEL:
        if (a[0].u.i >= 0 && a[0].u.i < (tv_int)w && a[1].u.i >= 0 && a[1].u.i < (tv_int)h)
            set_int(res, gfx_value(vm->host.gfx_get(vm->ctx, (int)a[0].u.i, (int)a[1].u.i)));
        else
            set_int(res, 0);
        break;
    case N_DRAW_LINE:
        gfx_line(vm, (long)a[0].u.i, (long)a[1].u.i, (long)a[2].u.i, (long)a[3].u.i, gfx_rgb(a + 4));
        break;
    case N_DRAW_RECT: case N_FILL_RECT: {
        long x = (long)a[0].u.i, y = (long)a[1].u.i, rw = (long)a[2].u.i, rh = (long)a[3].u.i;
        unsigned long c = gfx_rgb(a + 4);
        if (rw <= 0 || rh <= 0) break;                    /* empty */
        if (id == N_FILL_RECT) { gfx_fill(vm, x, y, x + rw, y + rh, c); break; }
        gfx_fill(vm, x, y, x + rw, y + 1, c);                     /* top */
        gfx_fill(vm, x, y + rh - 1, x + rw, y + rh, c);           /* bottom */
        gfx_fill(vm, x, y + 1, x + 1, y + rh - 1, c);             /* left */
        gfx_fill(vm, x + rw - 1, y + 1, x + rw, y + rh - 1, c);   /* right */
        break;
    }
    case N_DRAW_CIRCLE:
        gfx_circle(vm, (long)a[0].u.i, (long)a[1].u.i, (long)a[2].u.i, gfx_rgb(a + 3));
        break;
    case N_DRAW_TEXT: {
        tv_len n;
        const char *s;
        if (!to_str_slot(vm, (uint16_t)(base + 2))) return 0;     /* any value, like print */
        s = str_data(vm, &STK[base + 2], &n);
        gfx_text(vm, (long)a[0].u.i, (long)a[1].u.i, (const uint8_t *)s, n, gfx_rgb(a + 3));
        break;
    }
    case N_SCREEN_SHOW:
        if (vm->host.gfx_show) vm->host.gfx_show(vm->ctx);
        break;
    default: break;
    }
    return 1;
}
#endif /* TILVM_NO_GFX */

/* ─── energy saving (TIL 3.1, docs/TIL_ENERGY.md) ─────────────────────────
 * Modes (st.energy): 0 performance (never idle), 1 balanced (idle 1 ms at
 * every step that is a multiple of 4096), 2 low (every multiple of 512).
 * While a frame of an #[energy: low] function is active the VM behaves as
 * mode 2. The interpreter calls energy_point() only at steps that are
 * multiples of 512 while ENERGY_ON (refuel() makes the fuel counter end
 * there; energy_mode() refuels): nothing is added per instruction. */
#ifndef TILVM_NO_ENERGY
#define ENERGY_ON(vm) ((vm)->st.energy || ((vm)->flags & VMF_ENERGY_LOW))

static void energy_point(tilvm *vm) {
    tilvm_state *st = &vm->st;
    unsigned m = st->energy;
    uint16_t i;
    if (m != 2 && (vm->flags & VMF_ENERGY_LOW))
        for (i = st->n_frames; i > 0; i--)
            if (F_FLAGS(vm, st->frames[i - 1].func) & FLAG_ENERGY_LOW) { m = 2; break; }
    if (m == 2 || (m == 1 && !((uint16_t)st->steps & 4095u))) {
        vm->idle_count++;
        if (vm->host.idle) vm->host.idle(vm->ctx, 1);
    }
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
#ifndef TILVM_NO_GFX
    case N_SCREEN_INIT: case N_SCREEN_CLEAR: case N_DRAW_PIXEL: case N_DRAW_LINE: case N_DRAW_RECT:
    case N_FILL_RECT: case N_DRAW_CIRCLE: case N_DRAW_TEXT: case N_GET_PIXEL: case N_SCREEN_SHOW:
    case N_KEY_PRESSED: case N_WAIT_KEY: case N_SCREEN_WIDTH: case N_SCREEN_HEIGHT:
        if (!native_gfx(vm, id, base, argc, &res)) return 0;
        break;
#endif
    case N_PRINT: case N_PRINTLN: {
        uint8_t i;
        if (!argc) out(vm, "\n", 1);
        for (i = 0; i < argc; i++) { print_value(vm, &STK[base + i]); out(vm, "\n", 1); }
        break;
    }
    case N_ITER:
        if (a->t == TV_MAP) {
            /* keys snapshot */
            tv_len n = ARROBJ(vm, a->u.r)->len, i;
            tv_val *d;
            if (!push_empty(vm, OBJ_ARRAY, n, n)) return 0;
            d = VALS(vm, ARROBJ(vm, STK[SP - 1].u.r)->buf);
            for (i = 0; i < n; i++) d[i] = VALS(vm, ARROBJ(vm, STK[base].u.r)->buf)[2 * i];
            STK[base] = STK[SP - 1];
            SP = (uint16_t)(base + 1);
            return 1;
        }
        if (a->t != TV_ARRAY && !is_str(a)) return fail(vm, "value is not iterable");
        res = *a;
        break;
    case N_LEN: {
        tv_len n;
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
        tv_len n = argc, i;
        int r;
        tv_val best;
        if (argc == 1 && a->t == TV_ARRAY) {
            n = ARROBJ(vm, a->u.r)->len;
            vals = VALS(vm, ARROBJ(vm, a->u.r)->buf);
        }
        if (id == N_SUM) {
            /* push 0, then add each element (arith works on stack slots) */
            if (SP + 3 > TILVM_STACK_SIZE) return fail(vm, E_STACK);
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
        tv_len n;
        uint32_t c;
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
        if (n > 0xFFFF) return fail(vm, "range too large");
        SP = base;
        if (!push_empty(vm, OBJ_ARRAY, n, n)) return 0;
        {
            tv_val *d = VALS(vm, ARROBJ(vm, STK[SP - 1].u.r)->buf);
            for (i = 0; i < (tv_int)n; i++) set_int(&d[i], lo + i);
        }
        return 1;
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
        uint32_t ms, until, now;
        if (a->t != TV_INT) return fail(vm, "sleep_ms() needs an integer");
        ms = (uint32_t)(a->u.i > 0 ? a->u.i : 0);
        if (vm->clock_ms) {
            /* wait on the clock (re-read after every idle: hosts may watch the reads, see
             * website/playground/wasm_host.c); idle instead of spinning when the host can */
            until = now_ms(vm) + ms;
            while ((int32_t)((now = now_ms(vm)) - until) < 0 && vm->st.halted != 2)
                if (vm->host.idle) vm->host.idle(vm->ctx, (unsigned long)(until - now));
        } else if (vm->host.idle && ms) {
            vm->host.idle(vm->ctx, ms);         /* no clock: trust the idle hook */
        }
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
        if ((uint32_t)SP + (uint32_t)n + 2 > TILVM_STACK_SIZE) return fail(vm, E_STACK);
        SP = base;
        for (i = 0; i < n; i++) {
            const char *s = vm->host.argv[i] ? vm->host.argv[i] : "";
            if (!make_str(vm, &STK[SP], s, (uint32_t)strlen(s))) return 0;
            SP++;
        }
        return make_array(vm, OBJ_ARRAY, (uint32_t)n);
    }
    case N_BYTES_TO_STR: {
        tv_len n, i;
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
        tv_len n, i;
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
#ifndef TILVM_NO_ENERGY
    case N_ENERGY_MODE:             /* energy_mode() -> mode; energy_mode(m) -> previous mode */
        set_int(&res, (tv_int)vm->st.energy);
        if (argc) {
            if (a->t != TV_INT || a->u.i < 0 || a->u.i > 2)
                return fail(vm, "energy_mode() needs a mode 0 (performance), 1 (balanced) or 2 (low)");
            vm->st.energy = (uint8_t)a->u.i;
        }
        break;
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
static int find_bytes(const char *h, tv_len hl, const char *n, tv_len nl, tv_len from) {
    tv_len i;
    if (nl == 0) return (int)from;
#ifndef TILVM_LEAN
    /* memchr jumps to the next possible start (same result as the plain loop) */
    for (i = from; (uint32_t)i + nl <= hl; i++) {
        const char *p = (const char *)memchr(h + i, (unsigned char)n[0], (size_t)(hl - nl + 1 - i));
        tv_len j;
        if (!p) return -1;
        i = (tv_len)(p - h);
        for (j = 1; j < nl && p[j] == n[j]; j++) { }
        if (j == nl) return (int)i;
    }
#else
    for (i = from; (uint32_t)i + nl <= hl; i++)
        if (memcmp(h + i, n, nl) == 0) return (int)i;
#endif
    return -1;
}

static int str_method(tilvm *vm, uint8_t mid, uint16_t base, uint8_t argc, tv_val *res) {
    tv_len n, m = 0;
    const char *s = str_data(vm, &STK[base], &n);
    tv_val *a1 = &STK[base + 1];
    switch (mid) {
    case M_LEN: set_int(res, (tv_int)n); return 1;
    case M_IS_EMPTY: set_bool(res, n == 0); return 1;
    case M_CONTAINS: case M_STARTS_WITH: case M_ENDS_WITH: case M_FIND: case M_INDEX_OF:
#ifndef TILVM_LEAN
    case M_COUNT:
#endif
    case M_EQ: {
        const char *t;
        if (!argc || !is_str(a1)) return failn(vm, "str.", METHOD_NAMES[mid], " expects a string argument", 0, 0, 0);
        t = str_data(vm, a1, &m);
        if (mid == M_CONTAINS) set_bool(res, find_bytes(s, n, t, m, 0) >= 0);
        else if (mid == M_STARTS_WITH) set_bool(res, m <= n && memcmp(s, t, m) == 0);
        else if (mid == M_ENDS_WITH) set_bool(res, m <= n && memcmp(s + n - m, t, m) == 0);
        else if (mid == M_EQ) set_bool(res, m == n && memcmp(s, t, m) == 0);
#ifndef TILVM_LEAN
        else if (mid == M_COUNT) {
            tv_int c = 0;
            int at = 0;
            if (m) while ((at = find_bytes(s, n, t, m, (tv_len)at)) >= 0) { c++; at += (int)m; }
            set_int(res, c);
        }
#endif
        else set_int(res, find_bytes(s, n, t, m, 0));
        return 1;
    }
    case M_GET:
        if (!argc || a1->t != TV_INT) return fail(vm, E_INDEX_INT);
        if (a1->u.i < 0 || (tv_uint)a1->u.i >= n) return bounds_error(vm, a1->u.i, n);
        set_int(res, (tv_int)(uint8_t)s[a1->u.i]);
        return 1;
#ifndef TILVM_LEAN
    case M_TRIM: case M_REPEAT:
#endif
    case M_TO_UPPER: case M_TO_LOWER: case M_SLICE: {
        tv_len lo = 0, hi = n;
        uint32_t i, times = 1;
        tv_ref r;
#ifndef TILVM_LEAN
        if (mid == M_TRIM) {
            while (lo < hi && (s[lo] == ' ' || s[lo] == '\t' || s[lo] == '\r' || s[lo] == '\n')) lo++;
            while (hi > lo && (s[hi - 1] == ' ' || s[hi - 1] == '\t' || s[hi - 1] == '\r' || s[hi - 1] == '\n')) hi--;
        } else
#endif
        if (mid == M_SLICE) {
            tv_int x = argc && a1->t == TV_INT ? a1->u.i : 0;
            tv_int y = argc > 1 && STK[base + 2].t == TV_INT ? STK[base + 2].u.i : (tv_int)n;
            if (x < 0) x = 0;
            if (y > (tv_int)n) y = (tv_int)n;
            if (x > (tv_int)n) x = (tv_int)n;
            if (y < 0) y = 0;
            lo = (tv_len)x;
            hi = (tv_len)(y > x ? y : x);
        }
#ifndef TILVM_LEAN
        else if (mid == M_REPEAT) {
            if (!argc || a1->t != TV_INT) return fail(vm, "repeat() needs an integer");
            times = a1->u.i > 0 ? (uint32_t)a1->u.i : 0;
            if (times && n > TV_STR_MAX / times) return fail(vm, E_STR_LONG);
        }
        r = new_str(vm, (uint32_t)(hi - lo) * times);
#else
        r = new_str(vm, (uint32_t)(hi - lo));
#endif
        if (!r) return 0;
        s = str_data(vm, &STK[base], &n);
        {
            char *d = STROBJ(vm, r)->data;
            for (i = 0; i < times; i++, d += hi - lo) memcpy(d, s + lo, hi - lo);
        }
        if (mid == M_TO_UPPER || mid == M_TO_LOWER) {
            char *d = STROBJ(vm, r)->data;
            for (i = 0; i < (uint32_t)(hi - lo); i++) {
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
        tv_len ul;
        uint32_t count = 0, i, w;
        int at;
        tv_ref r;
        if (argc < 2 || !is_str(a1)) return fail(vm, "replace() needs two strings");
        if (!to_str_slot(vm, (uint16_t)(base + 2))) return 0;
        s = str_data(vm, &STK[base], &n);
        t = str_data(vm, &STK[base + 1], &m);
        if (!m) { *res = STK[base]; return 1; }
        str_data(vm, &STK[base + 2], &ul);
        at = 0;
        while ((at = find_bytes(s, n, t, m, (tv_len)at)) >= 0) { count++; at += (int)m; }
        if (ul > m && (uint32_t)count > (TV_STR_MAX - n) / (ul - m)) return fail(vm, E_STR_LONG);
        r = new_str(vm, (uint32_t)n + count * ul - count * m);
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
        /* split(sep) / split() on whitespace; pieces are appended to the result */
        uint32_t start = 0, i;
        uint16_t arr_slot;
        int ws = !argc;
        const char *t = 0;
        if (argc && !is_str(a1)) return fail(vm, "split() needs a string separator");
        if (argc) { t = str_data(vm, a1, &m); if (!m) ws = 2; }
        if (!push_empty(vm, OBJ_ARRAY, 0, 0)) return 0;
        arr_slot = (uint16_t)(SP - 1);
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
                next = at < 0 ? (uint32_t)n + 1 : (uint32_t)at + m;
            }
            if (SP + 3 > TILVM_STACK_SIZE) return fail(vm, E_STACK);
            {
                tv_ref r = new_str(vm, pend - start);
                if (!r) return 0;
                s = str_data(vm, &STK[base], &n);
                memcpy(STROBJ(vm, r)->data, s + start, pend - start);
                set_ref(&STK[SP++], TV_STR, r);
                if (!array_push(vm, arr_slot, (uint16_t)(SP - 1))) return 0;
                SP--;
            }
            if (ws == 0) { if (next > n) break; start = next; }
            else i = next;
        }
        *res = STK[--SP];
        return 1;
    }
#endif
    case M_TO_INT: if (!cast_slot(vm, base, 0)) return 0; *res = STK[base]; return 1;
#ifndef TILVM_LEAN
    case M_TO_FLOAT: if (!cast_slot(vm, base, 1)) return 0; *res = STK[base]; return 1;
#endif
    default: return failn(vm, E_NO_METHOD, mid < N_METHODS ? METHOD_NAMES[mid] : "?", "' on str", 0, 0, 0);
    }
}

#ifndef TILVM_LEAN
static void sort_array(tilvm *vm, tv_ref arr, int *err) {
    /* stable insertion sort (small, no recursion, no malloc) */
    tv_arrobj *a = ARROBJ(vm, arr);
    tv_len n = a->len, i, j;
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
    tv_len n = a->len, i;
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
        if (!argc || a1->t != TV_INT) return fail(vm, E_INDEX_INT);
        k = a1->u.i;
        if (mid == M_INSERT) {
            if (k < 0 || (tv_uint)k > n) return failf(vm, "insert index out of bounds: ", k, 0, 0, "");
            if (argc < 2) return fail(vm, "insert() needs an index and a value");
            if (!reserve(vm, base, (uint32_t)n + 1)) return 0;
            a = ARROBJ(vm, STK[base].u.r);
            v = VALS(vm, a->buf);
            for (i = n; i > (tv_len)k; i--) v[i] = v[i - 1];
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
        for (i = (tv_len)k; i + 1 < n; i++) v[i] = v[i + 1];
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
        if (!push_empty(vm, OBJ_ARRAY, (uint32_t)(hi - lo), (uint32_t)(hi - lo))) return 0;
        {
            tv_val *d = VALS(vm, ARROBJ(vm, STK[SP - 1].u.r)->buf);
            for (i = (tv_len)lo; i < (tv_len)hi; i++) d[i - (tv_len)lo] = VALS(vm, ARROBJ(vm, STK[base].u.r)->buf)[i];
        }
        *res = STK[--SP];
        return 1;
    }
    case M_JOIN: {
        /* measure the text of every element, allocate once, then write it
           (no temporary strings: works for arrays of any length) */
        tv_len sl = 0, l;
        tv_ref r;
        sink sk;
        if (argc) { if (!to_str_slot(vm, (uint16_t)(base + 1))) return 0; str_data(vm, &STK[base + 1], &sl); }
        sk.vm = vm; sk.mode = 0; sk.dst = 0; sk.len = 0;
        for (i = 0; i < n; i++) {
            if (i) sk.len += sl;
            emit_value(&sk, &VALS(vm, ARROBJ(vm, STK[base].u.r)->buf)[i], 0, 0);
            if (sk.len > TV_STR_MAX) return fail(vm, E_STR_LONG);
        }
        r = new_str(vm, sk.len);                  /* may move the array: re-read it */
        if (!r) return 0;
        sk.mode = 1; sk.dst = STROBJ(vm, r)->data; sk.len = 0;
        for (i = 0; i < n; i++) {
            if (i && sl) {
                const char *sp = str_data(vm, &STK[base + 1], &l);
                sink_put(&sk, sp, l);
            }
            emit_value(&sk, &VALS(vm, ARROBJ(vm, STK[base].u.r)->buf)[i], 0, 0);
        }
        set_ref(res, TV_STR, r);
        return 1;
    }
#endif
    default: return failn(vm, E_NO_METHOD, mid < N_METHODS ? METHOD_NAMES[mid] : "?", "' on array", 0, 0, 0);
    }
}

static int map_method(tilvm *vm, uint8_t mid, uint16_t base, uint8_t argc, tv_val *res) {
    tv_ref m = STK[base].u.r;
    tv_len n = ARROBJ(vm, m)->len, i;
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
        uint32_t per = mid == M_COPY ? 2 : 1, k = 0;
        tv_val *d;
        if (!push_empty(vm, mid == M_COPY ? OBJ_MAP : OBJ_ARRAY, per * n, n)) return 0;
        d = VALS(vm, ARROBJ(vm, STK[SP - 1].u.r)->buf);
        for (i = 0; i < n; i++) {
            tv_val *v = VALS(vm, ARROBJ(vm, STK[base].u.r)->buf);
            if (mid != M_VALUES) d[k++] = v[2 * i];
            if (mid != M_KEYS) d[k++] = v[2 * i + 1];
        }
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
    default: return failn(vm, E_NO_METHOD, mid < N_METHODS ? METHOD_NAMES[mid] : "?", "' on map", 0, 0, 0);
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
    default: return failn(vm, E_NO_METHOD, mid < N_METHODS ? METHOD_NAMES[mid] : "?", "' on ",
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
    else ok = failn(vm, E_NO_METHOD, mid < N_METHODS ? METHOD_NAMES[mid] : "?", "' on ", type_name(vm, r), 0, 0);
    if (!ok) return 0;
    STK[base] = res;
    SP = (uint16_t)(base + 1);
    return 1;
}

/* ─── loading ───────────────────────────────────────────────────────────── */
#ifndef TILVM_LEAN
static uint32_t fnv1a(const uint8_t *p, uint32_t n) {     /* identifies the program in snapshots */
    uint32_t h = 2166136261u, i;
    for (i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}
#endif

static void reset_state(tilvm *vm) {
    tilvm_state *st = &vm->st;
    const uint8_t *f;
    uint16_t i;
    memset(st, 0, offsetof(tilvm_state, frames));     /* counters, flags, sp = 0 */
    st->magic = 0x54494C53u;   /* "TILS" */
#ifndef TILVM_NO_ENERGY
    st->energy = (uint8_t)((vm->flags >> TBC_ENERGY_SHIFT) & 3u);
#endif
    st->heap_top = (tv_size)ALIGN_UP(1);
    st->n_handlers = 0;        /* after frames[]: not covered by the memset */
    st->_hpad = 0;
    st->rng = 0x9E3779B9u;
#if TILVM_INT_BITS == 64
    st->rng = 0x9E3779B97F4A7C15ULL;
#endif
    for (i = 0; i < vm->n_globals; i++) set_nil(&st->stack[st->sp++]);
    f = func_rec(vm, vm->entry);
    st->frames[0].func = vm->entry;
    st->frames[0].base = st->sp;
    st->frames[0].pc = (tv_off)(vm->code_off + FN_OFFSET(f));
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
#ifdef TILVM_ADDR16
    if (size > 0xFFFFu) return -10;        /* offsets are 16-bit */
#endif
    vm->bc = bc;
    vm->bc_size = size;
#ifndef TILVM_NO_ENERGY
    vm->flags = (uint8_t)(bc[5] & 0x7Fu);
    if (((vm->flags >> TBC_ENERGY_SHIFT) & 3u) == 3u) return -5;   /* energy mode 0..2 */
#else
    vm->flags = bc[5];
#endif
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
        vm->funcs[i].code = (tv_off)(vm->code_off + FN_OFFSET(f));
        vm->funcs[i].nparams = FN_NPARAMS(f);
        vm->funcs[i].nreq = FN_NREQ(f);
        vm->funcs[i].nlocals = FN_NLOCALS(f);
        vm->funcs[i].flags = FN_FLAGS(f);
    }
#endif
#ifndef TILVM_NO_ENERGY
    for (i = 0; i < vm->n_funcs; i++)         /* energy_point() looks at the frames then */
        if (F_FLAGS(vm, (uint16_t)i) & FLAG_ENERGY_LOW) vm->flags |= VMF_ENERGY_LOW;
#endif
#if TILVM_FIELD_CACHE > 0
    for (i = 0; i < TILVM_FIELD_CACHE; i++) {
        vm->field_cache[i].type = CACHE_EMPTY;
        vm->method_cache[i].type = CACHE_EMPTY;
    }
#endif
#ifndef TILVM_LEAN
    vm->bc_hash = fnv1a(bc, size);
#else
    vm->bc_hash = 0;           /* only snapshots use it: none in lean builds */
#endif
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
    vm->idle_count = 0;
    memset(&vm->host, 0, sizeof(vm->host));
#ifndef TILVM_NO_STDIO
    vm->host.read_file = default_read_file;
    vm->host.write_file = default_write_file;
#endif
    reset_state(vm);
#ifdef TILVM_STATS
    {   /* paint the free stack slots so tilvm_get_stats() can find the high-water mark */
        uint32_t k;
        for (k = vm->st.sp; k < (uint32_t)TILVM_STACK_SIZE; k++) vm->st.stack[k].t = (uint8_t)TV_PAINT;
        memset(&vm->stats, 0, sizeof(vm->stats));
        vm->stats.peak_frames = vm->st.n_frames;
        vm->stats.peak_heap = vm->st.heap_top;
    }
#endif
    return 0;
}

/* ─── errors ────────────────────────────────────────────────────────────── */
static uint16_t line_for(tilvm *vm, tv_off pc) {
    uint16_t lo = 0, hi = vm->n_lines, best = 0;
    pc = (tv_off)(pc - vm->code_off);
    while (lo < hi) {
        uint16_t mid = (uint16_t)(((uint32_t)lo + hi) / 2);
        const uint8_t *e = vm->bc + vm->lines_off + 6u * mid;
        if (RD32(e) <= pc) { best = RD16(e + 4); lo = (uint16_t)(mid + 1); }
        else hi = mid;
    }
    return (uint16_t)best;
}

static void err_out(tilvm *vm, const char *s, size_t n) {
    tilvm_write_fn w = vm->write_err ? vm->write_err : vm->write;
    if (w && n) w(vm->ctx, s, n);
}

static void err_cstr(tilvm *vm, const char *s) { err_out(vm, s, strlen(s)); }

static void report(tilvm *vm, tv_off pc, const char *prefix) {
    char num[32];
    uint16_t i, line;
    err_cstr(vm, prefix);
    if (vm->err_msg) err_cstr(vm, vm->err_msg);
    if (is_str(&vm->err_val)) {
        tv_len n;
        const char *p = str_data(vm, &vm->err_val, &n);
        err_out(vm, p, n);
    }
    err_cstr(vm, "\n");
    for (i = vm->st.n_frames; i > 0 && vm->st.n_frames - i < 8; i--) {
        tv_frame *fr = &vm->st.frames[i - 1];
        tv_off at = (tv_off)((i == vm->st.n_frames) ? pc : fr->pc - 1);
        tv_len nlen;
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
    st->n_handlers = t->n_handlers;     /* handles / running clauses inside the block */
    st->frames[st->n_frames - 1].pc = t->handler;
    /* push the message as a string */
    if (vm->err_val.t != TV_NIL && is_str(&vm->err_val) && (!vm->err_msg || !*vm->err_msg)) {
        st->stack[st->sp++] = vm->err_val;
    } else {
        tv_len ml = vm->err_msg ? (tv_len)strlen(vm->err_msg) : 0, vl = 0;
        if (is_str(&vm->err_val)) str_data(vm, &vm->err_val, &vl);
        st->stack[st->sp++] = vm->err_val;          /* keep it rooted during allocation */
        r = new_str(vm, (uint32_t)ml + vl);
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
    tv_len len;
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
    return failn(vm, E_NO_METHOD, nm, "' on ", tn, 0, 0);
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
static int prepare_invoke(tilvm *vm, tv_off *pc, uint16_t *fidx, uint8_t *argc,
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
    tv_len nl;
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
    tv_len i, j, len = ARROBJ(vm, m)->len;
    tv_val *v = VALS(vm, ARROBJ(vm, m)->buf);
    for (i = 0; i < len; i++) { map_key(&v[2 * i]); if (!check_map_key(vm, &v[2 * i])) return 0; }
    for (i = 0; i < len; i++) {
        for (j = i + 1; j < len; j++) {
            if (equal(vm, &v[2 * i], &v[2 * j], 0)) {
                tv_len k;
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
    uint32_t total = 0, k = 0;
    uint16_t first = (uint16_t)(SP - n);
    tv_ref r;
#ifndef TILVM_LEAN
    /* measure the text of every part, then one allocation (the parts stay on the
     * stack, so the collector keeps and updates them) and one copy */
    uint32_t l;                 /* a measured length may exceed TV_STR_MAX (tv_len) */
    tv_len sl;
    sink s;
    char num[32];
    for (i = 0; i < n; i++) {
        const tv_val *v = &STK[first + i];
        if (is_str(v)) { str_data(vm, v, &sl); l = sl; }
        else if (v->t == TV_INT) l = (uint32_t)strlen(fmt_int(num, v->u.i));
        else { s.vm = vm; s.mode = 0; s.dst = 0; s.len = 0; emit_value(&s, v, 0, 0); l = s.len; }
        if (l > TV_STR_MAX || (total += l) > TV_STR_MAX) return fail(vm, E_STR_LONG);
    }
    r = new_str(vm, total);
    if (!r) return 0;
    for (i = 0; i < n; i++) {
        const tv_val *v = &STK[first + i];
        if (is_str(v)) {
            const char *p = str_data(vm, v, &sl);
            l = sl;
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
    tv_len l;
    for (i = 0; i < n; i++) {
        if (!to_str_slot(vm, (uint16_t)(first + i))) return 0;
        str_data(vm, &STK[first + i], &l);
        total += l;
    }
    if (total > TV_STR_MAX) return fail(vm, E_STR_LONG);
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
static int push_try(tilvm *vm, uint8_t op, tv_off handler) {
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
    TV_STAT_MAX(peak_try, st->n_try);
    t->handler = handler;
    t->deadline = deadline;
    t->n_frames = st->n_frames;
    t->sp = SP;
    t->n_handlers = st->n_handlers;
    if (deadline) st->n_deadlines++;
    return 1;
}

#ifdef TILVM_LEAN
/* lean builds (less code): drop the innermost recovery point (END_TRY) */
static void pop_try(tilvm_state *st) {
    if (st->n_try) {
        if (st->tries[st->n_try - 1].deadline) st->n_deadlines--;
        st->n_try--;
    }
}

/* drop the recovery points of the returning frame */
static void pop_frame_tries(tilvm_state *st) {
    while (st->n_try && st->tries[st->n_try - 1].n_frames >= st->n_frames) pop_try(st);
}
#endif

/* index of the outermost recovery point whose deadline has passed, or -1 */
static int expired_try(tilvm *vm) {
    tilvm_state *st = &vm->st;
    uint32_t now = now_ms(vm);
    uint16_t i;
    for (i = 0; i < st->n_try; i++)
        if (st->tries[i].deadline && (int32_t)(now - st->tries[i].deadline) >= 0) return i;
    return -1;
}

/* ─── effect handlers with resumption (TIL 3.1, docs/TIL_EFFECTS.md) ────── */
#define EFFECT_NAME_MAX 64     /* effect names in error messages (same cut in src/til.py) */

/* The effect helpers below are called from one place each in the interpreter.
 * Inlined there (GCC does that for static functions called once) they cost the
 * hot opcode handlers registers: up to 4% more executed instructions on the
 * benchmarks. Kept out of line they cost one call per perform and per return
 * while handler entries exist. */
#if defined(__GNUC__)
#define TV_NOINLINE __attribute__((noinline))
#define TV_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define TV_NOINLINE
#define TV_ALWAYS_INLINE
#endif

/* effect name (string constant k) as a C string for error messages */
static const char *effect_name(tilvm *vm, uint16_t k, char *buf) {
    tv_val nv;
    tv_len n;
    const char *p;
    nv.t = TV_STRK;
    nv.u.i = 0;
    nv.u.k = k;
    p = str_data(vm, &nv, &n);
    if (n > EFFECT_NAME_MAX) n = EFFECT_NAME_MAX;
    memcpy(buf, p, n);
    buf[n] = 0;
    return buf;
}

#ifdef TILVM_NO_EFFECTS
/* PERFORM without handler support: the top-level function E (TIL 3.0). */
TV_NOINLINE static int effect_target(tilvm *vm, uint16_t name_c, uint16_t dflt, uint8_t argc,
                                     uint16_t *fidx, tv_ref *clos) {
    char name[EFFECT_NAME_MAX + 1];
    (void)argc;
    *clos = 0;
    if (dflt == 0xFFFF) return failn(vm, "unhandled effect '", effect_name(vm, name_c, name), "'", 0, 0, 0);
    *fidx = dflt;
    return 1;
}
#else
TV_NOINLINE static void pop_tries_to(tilvm *vm, uint16_t n) {
    tilvm_state *st = &vm->st;
    while (st->n_try > n) {
        if (st->tries[st->n_try - 1].deadline) st->n_deadlines--;
        st->n_try--;
    }
}

/* Innermost visible clause for effect `name_c`: returns the handler index and
 * stores the clause value, or -1. A running clause (kind 1) masks its owner and
 * everything above it, so the search continues below the owner. */
static int find_clause(tilvm *vm, uint16_t name_c, tv_val *clause) {
    tilvm_state *st = &vm->st;
    tv_val key;
    int i = (int)st->n_handlers - 1;
    key.t = TV_STRK;
    key.u.i = 0;
    key.u.k = name_c;
    while (i >= 0) {
        const tv_handler *h = &st->handlers[i];
        uint16_t j;
        if (h->kind == 1) { i = (int)h->n - 1; continue; }
        for (j = 0; j < h->n; j++) {
            const tv_val *nm = &STK[h->sp + 2u * j + 1u];
            if ((nm->t == TV_STRK && nm->u.k == name_c) || (is_str(nm) && str_eq(vm, nm, &key))) {
                *clause = STK[h->sp + 2u * j];
                return i;
            }
        }
        i--;
    }
    return -1;
}

/* RETURN with handler entries present: drop the handles the returning frame
 * installed. Returning from an effect clause (= it finished without resume)
 * aborts the handle that owns it: the clause value becomes the value of the
 * handle expression. Returns 1 after such an abort (frames, recovery points
 * and SP are cut back to the handle; the caller pushes the value and continues
 * at the frame's pc), 0 for a normal return, -1 on a fatal error. The value is
 * not passed here: taking the address of the interpreter's `result` would keep
 * it in memory on every RETURN. */
TV_NOINLINE static int effect_leave(tilvm *vm) {
    tilvm_state *st = &vm->st;
    const tv_handler *m, *h;
    uint16_t owner;
    while (st->n_handlers && st->handlers[st->n_handlers - 1].n_frames >= st->n_frames) st->n_handlers--;
    if (!(F_FLAGS(vm, st->frames[st->n_frames - 1].func) & FLAG_EFFECT_CLAUSE)) return 0;
    m = st->n_handlers ? &st->handlers[st->n_handlers - 1] : 0;
    if (!m || m->kind != 1 || m->n_frames + 1 != st->n_frames) {
        fail(vm, "effect clause returned outside its handler");
        return -1;
    }
    owner = m->n;
    h = &st->handlers[owner];
    pop_tries_to(vm, h->n_try);
    st->n_frames = h->n_frames;
    SP = h->sp;
    st->frames[st->n_frames - 1].pc = h->abort_pc;
    st->n_handlers = owner;
    return 1;
}

/* PERFORM: choose the function to enter (a clause, or the default handler).
 * Returns 1 (fidx / clos set; for a clause the running-clause entry is pushed),
 * 0 on a runtime error. */
TV_NOINLINE static int effect_target(tilvm *vm, uint16_t name_c, uint16_t dflt, uint8_t argc,
                                     uint16_t *fidx, tv_ref *clos) {
    tilvm_state *st = &vm->st;
    char name[EFFECT_NAME_MAX + 1], n1[32], n2[32];
    tv_val clause;
    tv_handler *m;
    uint8_t np;
    int idx = st->n_handlers ? find_clause(vm, name_c, &clause) : -1;
    *clos = 0;
    if (idx < 0) {
        if (dflt == 0xFFFF) return failn(vm, "unhandled effect '", effect_name(vm, name_c, name), "'", 0, 0, 0);
        *fidx = dflt;                       /* the top-level function E (TIL 3.0) */
        return 1;
    }
    if (clause.t == TV_CLOSURE) { *clos = clause.u.r; *fidx = CLOSOBJ(vm, *clos)->func; }
    else if (clause.t == TV_FUNC) *fidx = clause.u.k;
    else return failn(vm, "effect '", effect_name(vm, name_c, name), "' has no callable clause", 0, 0, 0);
    np = F_NPARAMS(vm, *fidx);
    if (argc > np || argc < F_NREQ(vm, *fidx))
        return failn(vm, "effect '", effect_name(vm, name_c, name), "' expects ",
                     fmt_int(n1, (tv_int)np), " argument(s), got ", fmt_int(n2, (tv_int)argc));
    if (st->n_handlers >= TILVM_MAX_HANDLERS) return fail(vm, "too many nested handlers");
    m = &st->handlers[st->n_handlers++];
    TV_STAT_MAX(peak_handlers, st->n_handlers);
    m->kind = 1;
    m->n_frames = st->n_frames;
    m->sp = 0;
    m->n_try = st->n_try;
    m->n = (uint16_t)idx;
    m->abort_pc = 0;
    m->_pad[0] = m->_pad[1] = m->_pad[2] = 0;
    return 1;
}
#endif /* TILVM_NO_EFFECTS */

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
 * operands and run the plain handler. Without TILVM_FAST_PATHS (-Os) only those
 * slow paths are compiled. TILVM_LEAN builds compile only the control flow here
 * and run the data opcodes and every superinstruction in lean_op() (above).
 */
#ifdef TILVM_LEAN
#undef TILVM_COMPUTED_GOTO
#define TILVM_COMPUTED_GOTO 0      /* lean: switch dispatch, the data opcodes in lean_op() */
#endif
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
#ifdef TILVM_LEAN
/* lean builds: a call instead of a pointer division / multiplication (values are
 * 3 or 5 bytes on 8-bit CPUs) at every use */
static uint16_t lean_slot(const tv_val *stk, const tv_val *p) { return (uint16_t)(p - stk); }
static tv_val *lean_top(tilvm_state *st) { return st->stack + st->sp; }
static void lean_sync(tilvm_state *st, const tv_val *sp) { st->sp = (uint16_t)(sp - st->stack); }
static void lean_save_pc(tv_frame *fr, const uint8_t *ip, const uint8_t *code) { fr->pc = (tv_off)(ip - code); }
#define SLOT_OF(p) lean_slot(stk, p)
#define SP_INDEX(p) lean_slot(stk, p)
#define RELOAD_SP() (sp = lean_top(st))
#define SYNC_SP() lean_sync(st, sp)
#define SAVE_PC() lean_save_pc(fr, ip, code)
#else
#define SLOT_OF(p) ((uint16_t)(p - stk))                  /* stack index of a slot */
#define SP_INDEX(p) ((p) - stk)                           /* the same for comparisons */
#define RELOAD_SP() (sp = stk + st->sp)
#define SYNC_SP() (st->sp = SLOT_OF(sp))
#define SAVE_PC() (fr->pc = (tv_off)(ip - code))
#endif
#ifdef TILVM_LEAN
#define TOPSLOT(n) ((uint16_t)(SLOT_OF(sp) - (n)))       /* stack index of sp[-n] */
#else
#define TOPSLOT(n) ((uint16_t)(sp - stk - (n)))
#endif
#define RDS16(p) ((int16_t)RD16(p))
#define CHECK_STACK(n) do { if (stk_end - sp < (n)) goto stack_overflow; } while (0)
/* run a helper that works on vm->st.sp; panic when it fails */
#define HELPER(call) do { int ok_; SYNC_SP(); ok_ = (call); RELOAD_SP(); if (!ok_) goto panic; } while (0)
#ifdef TILVM_LEAN
#define REFUEL() ((void)0)         /* lean builds check every step (see backedge) */
#else
#define REFUEL() (fuel_set = fuel = refuel(vm, budget))

/* steps that may run before the slow path of `backedge` must check the sandbox
 * limit or the budget (budget 0 = none), or must look at energy saving (the
 * next step that is a multiple of 512). Always inlined: as a call (GCC stops
 * inlining it once it has the energy term) it costs the hot handlers registers,
 * up to 3% more executed instructions on the benchmarks. */
static TV_ALWAYS_INLINE uint32_t refuel(const tilvm *vm, uint32_t budget) {
    uint32_t fuel = budget ? budget : 0xFFFFFFFFu;
    if (vm->max_steps) {
        uint32_t left = vm->st.steps < vm->max_steps ? vm->max_steps - vm->st.steps : 0;
        if (left != 0xFFFFFFFFu) left++;
        if (left < fuel) fuel = left;
    }
#ifndef TILVM_NO_ENERGY
    if (ENERGY_ON(vm)) {
        uint32_t left = 512u - (vm->st.steps & 511u);
        if (left < fuel) fuel = left;
    }
#endif
    return fuel;
}
#endif

#define IS_CMP(c) ((unsigned)(c) - OP_EQ < 6u)
#ifndef TILVM_LEAN
/* integer comparison x <c> y for c in OP_EQ..OP_GE without branches:
 * bit (sign(x - y) + 1) of the mask */
static const uint8_t CMP_MASK[6] = { 2, 5, 1, 3, 4, 6 };
#define INT_CMP(c, x, y) ((CMP_MASK[(c) - OP_EQ] >> (((x) > (y)) - ((x) < (y)) + 1)) & 1)
#endif

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

#ifdef TILVM_LEAN
/* ─── lean builds: the data opcodes in a function of their own ─────────────
 * TILVM_LEAN (8-bit CPUs): the stack and data opcodes - every superinstruction
 * included - run in lean_op(), and tilvm_run_steps keeps the control flow
 * (jumps, calls, returns, recovery points, effects). SDCC allocates registers
 * far better in two medium-sized functions than in one huge one, and the
 * shared push/pop tails save more: the Z80 image is several KB smaller. Desktop
 * builds keep the single loop (one dispatch per instruction).
 * Same semantics as the CASE() handlers of tilvm_run_steps - keep both in sync
 * (the tests run every program on lean builds too). lean_op works on
 * vm->st.sp; *ipp points at the operands, and after LEAN_NEXT / LEAN_BACK at
 * the next instruction. A superinstruction runs as the sequence it replaces
 * (re-dispatching the plain opcodes), the compare-and-branch ones without
 * pushing their operands, like tilvm_run_steps. */
enum { LEAN_PANIC, LEAN_NEXT, LEAN_BACK, LEAN_RETURN, LEAN_BAD };
/* LEAN_BACK: a jump backwards was taken (one step); LEAN_RETURN: return *res */

static int lean_op(tilvm *vm, unsigned op, const uint8_t **ipp, tv_val *bp, const tv_frame *fr,
                   tv_val *res) {
    const uint8_t *ip = *ipp;
    tv_val v, imm, *sp, *a, *b;
    unsigned cmp, npop;
    int r;
again:
    sp = &STK[SP];                              /* sp[-1] is the top of the stack */
    switch (op) {
    /* constants and stack */
    case OP_NOP: break;
    case OP_PUSH_NIL: set_nil(&v); goto push;
    case OP_PUSH_TRUE: case OP_PUSH_FALSE: set_bool(&v, op == OP_PUSH_TRUE); goto push;
    case OP_PUSH_I8: set_int(&v, (tv_int)(int8_t)*ip++); goto push;
    case OP_PUSH_I16: set_int(&v, (tv_int)RDS16(ip)); ip += 2; goto push;
    case OP_PUSH_CONST: load_const(vm, RD16(ip), &v); ip += 2; goto push;
    case OP_POP: goto pop;
    case OP_DUP: v = sp[-1]; goto push;
    case OP_DUP2:
        if (SP + 2 > TILVM_STACK_SIZE) goto overflow;
        sp[0] = sp[-2];
        sp[1] = sp[-1];
        SP += 2;
        break;
    case OP_SWAP: v = sp[-1]; sp[-1] = sp[-2]; sp[-2] = v; break;
    /* variables */
    case OP_LOAD_LOCAL: v = bp[*ip++]; goto push;
    case OP_STORE_LOCAL: bp[*ip++] = sp[-1]; goto pop;
    case OP_LOAD_GLOBAL: v = STK[RD16(ip)]; ip += 2; goto push;
    case OP_STORE_GLOBAL: STK[RD16(ip)] = sp[-1]; ip += 2; goto pop;
    case OP_LOAD_CELL: case OP_STORE_CELL:                  /* the cell in local x */
    case OP_LOAD_UPVAL: case OP_STORE_UPVAL: case OP_LOAD_UPVAL_RAW: {   /* captured value x */
        a = op >= OP_LOAD_CELL && op <= OP_STORE_CELL ? &bp[*ip] : &CLOSOBJ(vm, fr->closure)->v[*ip];
        ip++;
        if (op != OP_LOAD_UPVAL_RAW) a = &CELLOBJ(vm, a->u.r)->v;
        if (op == OP_STORE_CELL || op == OP_STORE_UPVAL) { *a = sp[-1]; goto pop; }
        v = *a;
        goto push;
    }
    case OP_MAKE_CELL: {
        tv_ref cell = alloc_obj(vm, OBJ_CELL, sizeof(tv_cellobj));
        if (!cell) goto panic;
        CELLOBJ(vm, cell)->v = bp[*ip];
        set_ref(&bp[*ip], TV_CELL, cell);
        ip++;
        break;
    }
    case OP_ARG_DEFAULT:
        if (bp[ip[0]].t != TV_UNDEF) ip += RDS16(ip + 1);
        ip += 3;
        break;
    /* arithmetic (arith() has the integer paths) */
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: case OP_POW:
        if (!arith(vm, (int)op)) goto panic;
        break;
    case OP_NEG:
        if (sp[-1].t == TV_INT) sp[-1].u.i = (tv_int)((tv_uint)0 - (tv_uint)sp[-1].u.i);
#if TILVM_FLOAT
        else if (sp[-1].t == TV_FLOAT) sp[-1].u.f = -sp[-1].u.f;
#endif
        else { fail(vm, "cannot negate a non-number"); goto panic; }
        break;
    case OP_NOT: set_bool(&sp[-1], !truthy(vm, &sp[-1])); break;
    case OP_BNOT:
        if (sp[-1].t != TV_INT) { fail(vm, "~ needs an integer"); goto panic; }
        sp[-1].u.i = ~sp[-1].u.i;
        break;
    case OP_BAND: case OP_BOR: case OP_BXOR: case OP_SHL: case OP_SHR: {
        tv_int x = sp[-2].u.i, y = sp[-1].u.i;
        if (sp[-2].t != TV_INT || sp[-1].t != TV_INT) { fail(vm, "bitwise operators need integers"); goto panic; }
        if (op == OP_BAND) x &= y;
        else if (op == OP_BOR) x |= y;
        else if (op == OP_BXOR) x ^= y;
        else if (op == OP_SHL) x = (tv_int)((tv_uint)x << ((unsigned)y & (TILVM_INT_BITS - 1)));
        else x = ishr(x, (unsigned)y & (TILVM_INT_BITS - 1));
        sp[-2].u.i = x;
        goto pop;
    }
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
        r = cmp_values(vm, op, &sp[-2], &sp[-1]);
        if (r < 0) goto panic;
        set_bool(&sp[-2], r);
        goto pop;
    /* objects */
    case OP_NEW_ARRAY: case OP_NEW_MAP:        /* make_array checks the 2 stack slots */
        if (!make_array(vm, op == OP_NEW_MAP ? OBJ_MAP : OBJ_ARRAY, RD16(ip))) goto panic;
        ip += 2;
        break;
    case OP_NEW_STRUCT:
        if (!new_struct(vm, RD16(ip), ip[2])) goto panic;
        ip += 3;
        break;
    case OP_GET_INDEX: if (!get_index(vm)) goto panic; break;
    case OP_SET_INDEX: if (!set_index(vm)) goto panic; break;
    case OP_GET_FIELD: case OP_SET_FIELD: {
        uint16_t name_c = RD16(ip);
        int slot = -1;
        a = &sp[-1];
        if (op == OP_SET_FIELD) a--;
        ip += 2;
        if (a->t == TV_STRUCT) FIELD_SLOT(vm, OBJ(vm, a->u.r)->aux, name_c, slot);
        if (slot < 0) { no_field(vm, name_c, a); goto panic; }
        if (op == OP_GET_FIELD) { *a = STRUCTOBJ(vm, a->u.r)->v[slot]; break; }
        STRUCTOBJ(vm, a->u.r)->v[slot] = sp[-1];
        SP -= 2;
        break;
    }
    case OP_GET_SLOT: {
        uint8_t i = *ip++;
        a = &sp[-1];
        if (a->t == TV_STRUCT && i < (OBJ(vm, a->u.r)->size - offsetof(tv_structobj, v)) / sizeof(tv_val))
            *a = STRUCTOBJ(vm, a->u.r)->v[i];
        else if (a->t == TV_ARRAY && i < ARROBJ(vm, a->u.r)->len)
            *a = VALS(vm, ARROBJ(vm, a->u.r)->buf)[i];
        else { fail(vm, "pattern slot out of range"); goto panic; }
        break;
    }
    case OP_MAKE_CLOSURE: {
        uint8_t n = ip[2], i;
        tv_ref cr = alloc_obj(vm, OBJ_CLOSURE, (uint32_t)offsetof(tv_closobj, v) + (uint32_t)(n * sizeof(tv_val)));
        if (!cr) goto panic;
        OBJ(vm, cr)->aux = n;
        CLOSOBJ(vm, cr)->func = RD16(ip);
        ip += 3;
        sp -= n;
        for (i = 0; i < n; i++) CLOSOBJ(vm, cr)->v[i] = sp[i];
        set_ref(sp, TV_CLOSURE, cr);
        SP = (uint16_t)(SP - n + 1);
        break;
    }
    case OP_PUSH_FUNC: v.t = TV_FUNC; v.u.i = 0; v.u.k = RD16(ip); ip += 2; goto push;
    case OP_CONCAT: if (!concat_values(vm, *ip++)) goto panic; break;
    case OP_FOR_ITER: case OP_FOR_ITER_STORE: {
        /* FOR_ITER it off [; STORE_LOCAL x]: next element of the array / string in
         * local `it` (index in it + 1), or jump when done */
        tv_val *c = &bp[*ip++], *ix = c + 1, *dst = sp;
        tv_len len = 0;
        const char *p = 0;
        int16_t rel;
        if (op == OP_FOR_ITER_STORE) dst = &bp[*ip++];
        rel = RDS16(ip);
        ip += 2;
        if (c->t == TV_ARRAY) len = ARROBJ(vm, c->u.r)->len;
        else p = str_data(vm, c, &len);
        if ((tv_uint)ix->u.i >= len) { ip += rel; break; }
        if (op == OP_FOR_ITER && SP >= TILVM_STACK_SIZE) goto overflow;
        if (c->t == TV_ARRAY) *dst = VALS(vm, ARROBJ(vm, c->u.r)->buf)[ix->u.i];
        else set_int(dst, (tv_int)(uint8_t)p[ix->u.i]);
        if (op == OP_FOR_ITER) SP++;
        ix->u.i++;
        break;
    }
    case OP_MATCH_TYPE: case OP_IS_ARRAY_LEN: {
        uint16_t k = RD16(ip);
        a = &sp[-1];
        ip += 2;
        if (op == OP_MATCH_TYPE) r = a->t == TV_STRUCT && OBJ(vm, a->u.r)->aux == k;
        else r = a->t == TV_ARRAY && ARROBJ(vm, a->u.r)->len == k;
        set_bool(a, r);
        break;
    }
    case OP_CAST: if (!cast_slot(vm, (uint16_t)(SP - 1), *ip++)) goto panic; break;
    /* returns: the value goes to *res */
    case OP_UNWRAP_OR_RETURN: case OP_UNWRAP_OR_PANIC: {     /* the ? operator */
        uint16_t kind = sp[-1].t == TV_STRUCT ? OBJ(vm, sp[-1].u.r)->aux : 0xFFFFu;
        if (kind == TYPE_SOME || kind == TYPE_OK) { sp[-1] = STRUCTOBJ(vm, sp[-1].u.r)->v[0]; break; }
        if (sp[-1].t != TV_NIL && kind != TYPE_ERR) break;           /* not an Option / Result */
        if (op == OP_UNWRAP_OR_RETURN) goto ret;
        if (sp[-1].t == TV_NIL) { fail(vm, "? applied to None"); goto panic; }
        sp[-1] = STRUCTOBJ(vm, sp[-1].u.r)->v[0];
        if (!to_str_slot(vm, (uint16_t)(SP - 1))) goto panic;
        vm->err_val = STK[SP - 1];
        vm->err_msg = "? applied to Err: ";
        goto panic;
    }
    case OP_CAST_RETURN:                        /* CAST kind; RETURN */
        if (!cast_slot(vm, (uint16_t)(SP - 1), *ip)) goto panic;
    ret:
        *res = STK[--SP];
        return LEAN_RETURN;
    case OP_RETURN_LOCAL:                       /* LOAD_LOCAL a; RETURN */
        *res = bp[*ip];
        return LEAN_RETURN;
    /* superinstructions (docs/TIL_BYTECODE_V2.md) */
    /* LOAD_LOCAL x, then the rest of the sequence */
    case OP_LOAD_LOCAL2: op = OP_LOAD_LOCAL; goto push_local;     /* LOAD_LOCAL a; LOAD_LOCAL b */
    case OP_INDEX_LL: op = OP_INDEX_L; goto push_local;           /* LOAD_LOCAL a; LOAD_LOCAL b; GET_INDEX */
    case OP_INDEX_L: op = OP_GET_INDEX; goto push_local;          /* LOAD_LOCAL b; GET_INDEX */
    case OP_LOAD_LOCAL_FIELD: op = OP_GET_FIELD; goto push_local; /* LOAD_LOCAL a; GET_FIELD name */
    case OP_ADD_STORE_LOCAL:                    /* ADD; STORE_LOCAL a */
        if (!arith(vm, OP_ADD)) goto panic;
        op = OP_STORE_LOCAL;
        goto again;
    case OP_INC_LOCAL:                          /* LOAD_LOCAL a; PUSH_I8 k; ADD; STORE_LOCAL a */
        if (SP + 2 > TILVM_STACK_SIZE) goto overflow;
        sp[0] = bp[ip[0]];
        set_int(&sp[1], (tv_int)(int8_t)ip[1]);
        SP += 2;
        if (!arith(vm, OP_ADD)) goto panic;
        bp[ip[0]] = sp[0];
        ip += 2;
        goto pop;
    case OP_CAST_LOCAL:                         /* LOAD_LOCAL a; CAST kind; STORE_LOCAL a */
        if (SP >= TILVM_STACK_SIZE) goto overflow;
        sp[0] = bp[ip[0]];
        SP++;
        if (!cast_slot(vm, (uint16_t)(SP - 1), ip[1])) goto panic;
        bp[ip[0]] = sp[0];
        ip += 2;
        goto pop;
    /* <cmp> + JUMP_IF_FALSE: jump when the comparison is false. Operands: cmp, then
     *   JUMP_IF_NOT     -          (a, b on the stack)    LOAD_LOCAL b: JUMP_IF_NOT_L  b
     *   JUMP_IF_NOT_I   k16        (b = k)                LOAD_LOCAL a + b: _LL  a b,  _LI  a k16
     * then rel; the operands are not pushed */
    case OP_JUMP_IF_NOT: case OP_JUMP_IF_NOT_L: case OP_JUMP_IF_NOT_I:
    case OP_JUMP_IF_NOT_LL: case OP_JUMP_IF_NOT_LI: {
        int16_t rel;
        cmp = *ip++;
        if (op >= OP_JUMP_IF_NOT_LL) { a = &bp[*ip++]; npop = 0; op -= 2; }   /* -> _L / _I */
        else { npop = op == OP_JUMP_IF_NOT ? 2 : 1; a = sp - npop; }
        b = a + 1;                              /* JUMP_IF_NOT */
        if (op == OP_JUMP_IF_NOT_L) b = &bp[*ip++];
        else if (op == OP_JUMP_IF_NOT_I) { set_int(&imm, (tv_int)RDS16(ip)); ip += 2; b = &imm; }
        rel = RDS16(ip);
        r = cmp_values(vm, cmp, a, b);
        if (r == -2) return LEAN_BAD;
        if (r < 0) goto panic;
        SP = (uint16_t)(SP - npop);
        ip += 2;
        if (r) break;
        ip += rel;
        if (rel >= 0) break;
        *ipp = ip;
        return LEAN_BACK;
    }
    default:
        /* <op>_I k = PUSH_I8/PUSH_I16 k; <op>      <op>_L b = LOAD_LOCAL b; <op> */
        if (op < OP_ADD_I || op > OP_GE_L) return LEAN_BAD;
        if (op >= OP_ADD_L) { op = FUSED_BINOP[op & 15]; goto push_local; }
        set_int(&v, (tv_int)RDS16(ip));
        ip += 2;
        op = FUSED_BINOP[op & 15];
        goto push_again;
    }
    *ipp = ip;
    return LEAN_NEXT;
push:                                           /* push v (the cases above leave SP alone) */
    if (SP >= TILVM_STACK_SIZE) goto overflow;
    *sp = v;
    SP++;
    *ipp = ip;
    return LEAN_NEXT;
push_local:                                     /* push local *ip, then run `op` */
    v = bp[*ip++];
push_again:                                     /* push v, then run `op` */
    if (SP >= TILVM_STACK_SIZE) goto overflow;
    *sp = v;
    SP++;
    goto again;
pop:
    SP--;
    *ipp = ip;
    return LEAN_NEXT;
overflow:
    fail(vm, E_STACK);
panic:
    return LEAN_PANIC;
}
#endif /* TILVM_LEAN */

/* SDCC (Z80): in a huge interpreter loop, global CSE keeps dozens of temporaries
 * alive across the whole function, so the stack frame grows beyond the reach of
 * IX (-128..127). SDCC 4.2's register allocator costs such "extended stack"
 * operands as if they were IX-relative, and code generation then fails in
 * genPlus ("Unimplemented", gen.c:7476/7884). Lean builds keep only the control
 * flow here (the data opcodes are in lean_op), so the frame stays small with
 * GCSE (55 bytes on the Z80) and GCSE saves ~0.8 KB; builds with the whole loop
 * turn GCSE off for this function. Keep the frame below 128 bytes when adding
 * opcodes (tests/test_z80.py checks it). */
#if defined(__SDCC) && !defined(TILVM_LEAN)
#pragma save
#pragma nogcse
#endif
int tilvm_run_steps(tilvm *vm, uint32_t budget) {
    const uint8_t *const code = vm->bc;
    tilvm_state *const st = &vm->st;
    tv_val *const stk = st->stack;
#ifndef TILVM_LEAN
    tv_val *const stk_end = stk + TILVM_STACK_SIZE;
#endif
    const uint8_t *ip, *op_ip;
    tv_val *sp, *bp;
    tv_frame *fr;
#ifndef TILVM_LEAN
    tv_val result, imm;
    const tv_val *cmp_a = 0, *cmp_b = 0;
    unsigned op, cmp = 0, cmp_pop = 0;
#else
    tv_val result;
    unsigned op;
#endif
    int target = -1, r;
#ifndef TILVM_LEAN
    uint32_t fuel, fuel_set;
#endif
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
        /* 0x70 */ &&L_OP_TRY, &&L_OP_TRY_TIMEOUT, &&L_OP_END_TRY, &&L_OP_PANIC, &&L_OP_UNWRAP_OR_RETURN, &&L_OP_UNWRAP_OR_PANIC, &&L_OP_HANDLE, &&L_OP_END_HANDLE,
        /* 0x78 */ &&L_OP_PERFORM, &&L_OP_RESUME, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_BAD, &&L_OP_HALT,
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
#ifndef TILVM_LEAN
    set_nil(&imm);
#endif
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
#ifndef TILVM_LEAN
dispatch_op:
#endif
    switch (op) {
#endif

#ifndef TILVM_LEAN      /* lean builds run the data opcodes in lean_op() */
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
#endif /* TILVM_LEAN */

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
        tv_off pc;
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
        pc = (tv_off)(ip - code);
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
#if TILVM_MAX_FUNCS > 0
#define CF_NREQ() F_NREQ(vm, fidx)
#define CF_CODE() F_CODE(vm, fidx)
        uint8_t np = F_NPARAMS(vm, fidx), nl = F_NLOCALS(vm, fidx), i;
#else
#define CF_NREQ() FN_NREQ(fn)
#define CF_CODE() ((tv_off)(vm->code_off + FN_OFFSET(fn)))
        const uint8_t *fn = func_rec(vm, fidx);    /* no table: read the record once */
        uint8_t np = FN_NPARAMS(fn), nl = FN_NLOCALS(fn), i;
#endif
        tv_val *nb;
        SAVE_PC();                                   /* the caller continues after the call */
        if (argc > np || argc < CF_NREQ()) { arity_error(vm, fidx, argc); goto panic; }
        if (st->n_frames >= TILVM_MAX_FRAMES) { fail(vm, "stack overflow (recursion too deep)"); st->failed = 1; goto fatal; }
        if (SP_INDEX(sp) + (nl - argc) + 4 > TILVM_STACK_SIZE) { fail(vm, E_STACK); st->failed = 1; goto fatal; }
        nb = sp - argc;
        for (i = argc; i < np; i++) { sp->t = TV_UNDEF; sp->u.i = 0; sp++; }
        for (i = np; i < nl; i++) { set_nil(sp); sp++; }
        fr = &st->frames[st->n_frames++];
        TV_STAT_MAX(peak_frames, st->n_frames);
        fr->func = fidx;
        fr->base = (uint16_t)(nb - stk);
        fr->closure = clos;
        fr->extra = extra;
        fr->ret_pc = 0;
        fr->pc = CF_CODE();
#undef CF_NREQ
#undef CF_CODE
        ip = code + fr->pc;
        bp = nb;
        goto backedge;
    }
    CASE(OP_RETURN) result = *--sp; goto do_return;
    CASE(OP_RETURN_NIL) set_nil(&result);
do_return:
#ifdef TILVM_LEAN
    pop_frame_tries(st);
#else
    while (st->n_try && st->tries[st->n_try - 1].n_frames >= st->n_frames) {
        if (st->tries[st->n_try - 1].deadline) st->n_deadlines--;
        st->n_try--;
    }
#endif
#ifndef TILVM_NO_EFFECTS
    if (st->n_handlers) {
        /* drop the handles this frame installed; an effect clause that returns
         * (finished without resume) aborts its handle: continue after it */
        r = effect_leave(vm);
        if (r) {
            if (r < 0) { st->failed = 1; goto fatal; }
            fr = &st->frames[st->n_frames - 1];
            ip = code + fr->pc;
            bp = stk + fr->base;
            RELOAD_SP();
            *sp++ = result;                  /* the value of the handle expression */
            NEXT;
        }
    }
#endif
#ifdef TILVM_LEAN
    st->sp = (uint16_t)(fr->base - fr->extra);   /* lean: less code, the caller's registers at resume */
    st->stack[st->sp++] = result;
    if (!--st->n_frames) { st->halted = 1; return TILVM_DONE; }
    goto resume;
#else
    sp = stk + fr->base - fr->extra;
    *sp++ = result;
    if (!--st->n_frames) { st->halted = 1; SYNC_SP(); return TILVM_DONE; }
    fr = &st->frames[st->n_frames - 1];
    ip = code + fr->pc;
    bp = stk + fr->base;
    NEXT;
#endif
#ifndef TILVM_LEAN      /* lean builds: lean_op() */
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
#endif
    CASE(OP_NATIVE) CASE(OP_NATIVE_POP) {
        uint8_t id = ip[0], n = ip[1];
        ip += 2;
        SAVE_PC();
        HELPER(call_native(vm, id, n));
        if (st->halted == 2) return TILVM_EXIT;
        if (op == OP_NATIVE_POP) sp--;
#if !defined(TILVM_LEAN) && !defined(TILVM_NO_ENERGY)
        if (id == N_ENERGY_MODE) goto energy_changed;
#endif
        NEXT;
    }

#ifndef TILVM_LEAN
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
        tv_len len = 0;
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
#endif /* TILVM_LEAN */

    /* ── errors and recovery points ── */
    CASE(OP_TRY) CASE(OP_TRY_TIMEOUT) {
        int16_t rel = RDS16(ip);
        ip += 2;
        HELPER(push_try(vm, (uint8_t)op, (tv_off)((uint32_t)(ip - code) + (uint32_t)(int32_t)rel)));
        NEXT;
    }
    CASE(OP_END_TRY)
#ifdef TILVM_LEAN
        pop_try(st);
#else
        if (st->n_try) {
            if (st->tries[st->n_try - 1].deadline) st->n_deadlines--;
            st->n_try--;
        }
#endif
        NEXT;
    CASE(OP_PANIC)
        HELPER(to_str_slot(vm, TOPSLOT(1)));
        vm->err_val = *--sp;
        vm->err_msg = "";
        goto panic;
    /* ── effects with resumption (docs/TIL_EFFECTS.md) ── */
    CASE(OP_PERFORM) {                       /* PERFORM name, default, argc: a call */
        uint16_t dflt = RD16(ip + 2);
        name_c = RD16(ip);
        argc = ip[4];
        ip += 5;
        extra = 0;
        /* a clause (the running-clause entry is pushed) or the default function */
        if (!effect_target(vm, name_c, dflt, argc, &fidx, &clos)) goto panic;
        goto call_fn;
    }
#ifdef TILVM_NO_EFFECTS
    CASE(OP_HANDLE)
        fail(vm, "effect handlers are not available in this VM build");
        goto panic;
#else
    CASE(OP_HANDLE) {                        /* HANDLE n, rel_end: the n pairs stay on the stack */
        uint8_t n = ip[0];
        int16_t rel = RDS16(ip + 1);
        tv_handler *h;
        ip += 3;
        if (st->n_handlers >= TILVM_MAX_HANDLERS) { fail(vm, "too many nested handlers"); goto panic; }
        h = &st->handlers[st->n_handlers++];
        TV_STAT_MAX(peak_handlers, st->n_handlers);
        h->kind = 0;
        h->n_frames = st->n_frames;
        h->sp = (uint16_t)(TOPSLOT(0) - 2u * n);
        h->n_try = st->n_try;
        h->n = n;
        h->abort_pc = (tv_off)((uint32_t)(ip - code) + (uint32_t)(int32_t)rel);
        h->_pad[0] = h->_pad[1] = h->_pad[2] = 0;
        NEXT;
    }
    CASE(OP_END_HANDLE) {                    /* drop the record and its pairs, keep the value */
        const tv_handler *h = st->n_handlers ? &st->handlers[st->n_handlers - 1] : 0;
        if (!h || h->kind != 0 || h->n_frames != st->n_frames) {
            fail(vm, "END_HANDLE without an active handle");
            st->failed = 1;
            goto fatal;
        }
        st->n_handlers--;
        stk[h->sp] = sp[-1];
        sp = stk + h->sp + 1;
        NEXT;
    }
    CASE(OP_RESUME) {                        /* in a clause: TOS -> value of the perform */
        int k = (int)st->n_handlers - 1;
        result = *--sp;
        while (k >= 0 && st->handlers[k].kind != 1) k--;
        if (k < 0 || st->handlers[k].n_frames + 1 != st->n_frames ||
            !(F_FLAGS(vm, fr->func) & FLAG_EFFECT_CLAUSE)) {
            fail(vm, "resume outside an effect clause");
            st->failed = 1;
            goto fatal;
        }
        pop_tries_to(vm, st->handlers[k].n_try);
        st->n_handlers = (uint16_t)k;        /* exactly as at the perform */
        sp = stk + fr->base - fr->extra;
        *sp++ = result;
        st->n_frames--;
        fr = &st->frames[st->n_frames - 1];
        ip = code + fr->pc;
        bp = stk + fr->base;
        NEXT;
    }
#endif /* TILVM_NO_EFFECTS */
    CASE(OP_HALT)
        st->halted = 1;
        SAVE_PC();
        SYNC_SP();
        return TILVM_DONE;

    /* ── superinstructions (docs/TIL_BYTECODE_V2.md) ── */
#ifndef TILVM_LEAN      /* lean builds: lean_op() */
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
#endif /* TILVM_LEAN */

#ifdef TILVM_NO_EFFECTS
    CASE(OP_END_HANDLE) CASE(OP_RESUME)      /* never reached: HANDLE reports the build */
#endif
#if TILVM_COMPUTED_GOTO
    L_BAD:
#else
    default:
#endif
#ifdef TILVM_LEAN
        SYNC_SP();
        r = lean_op(vm, op, &ip, bp, fr, &result);
        RELOAD_SP();
        if (r == LEAN_NEXT) NEXT;
        if (r == LEAN_BACK) goto backedge;
        if (r == LEAN_RETURN) goto do_return;
        if (r == LEAN_PANIC) goto panic;
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
#ifdef TILVM_LEAN
    /* lean builds: the checks of every step without the fuel counter (less code) */
    if (vm->max_steps && st->steps > vm->max_steps) {
        fail(vm, "step limit exceeded");
        st->failed = 1;
        goto fatal;
    }
#ifndef TILVM_NO_ENERGY
    if (!((uint16_t)st->steps & 511u) && ENERGY_ON(vm)) {
        energy_point(vm);
        if (st->halted == 2) goto host_exit;
    }
#endif
    if (st->n_deadlines && ++st->tick >= 256) {
        st->tick = 0;
        target = expired_try(vm);
        if (target >= 0) { fail(vm, "timeout"); goto panic; }
    }
    if (budget && !--budget) {
        SAVE_PC();
        SYNC_SP();
        return TILVM_YIELD;
    }
    NEXT;
#else
    if (--fuel != 0 && !st->n_deadlines) NEXT;
    /* sandbox limit, deadlines and the budget, exactly as TIL 3.0 checked them at
     * every step (the steps since the last REFUEL all passed these checks) */
    if (budget) budget -= fuel_set - fuel - 1;
    if (vm->max_steps && st->steps > vm->max_steps) {
        fail(vm, "step limit exceeded");
        st->failed = 1;
        goto fatal;
    }
#ifndef TILVM_NO_ENERGY
    if (!(st->steps & 511u) && ENERGY_ON(vm)) {
        energy_point(vm);
        if (st->halted == 2) goto host_exit;    /* the idle hook ended the program */
    }
#endif
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
#endif

#if !defined(TILVM_LEAN) && !defined(TILVM_NO_ENERGY)
energy_changed:     /* energy_mode() set the mode: refuel for the next energy point */
    if (budget) budget -= fuel_set - fuel;   /* (the steps run since the last REFUEL) */
    REFUEL();
    NEXT;
#endif
#ifndef TILVM_NO_ENERGY
host_exit:
    SAVE_PC();
    SYNC_SP();
    return TILVM_EXIT;
#endif
#ifndef TILVM_LEAN
stack_overflow:
    fail(vm, E_STACK);
#endif
panic:
    SAVE_PC();
    SYNC_SP();
    if (recover(vm, target)) { target = -1; goto resume; }
    report(vm, (tv_off)(op_ip - code), "runtime error: ");
    st->failed = 1;
    return TILVM_ERROR;
fatal:
    SAVE_PC();
    SYNC_SP();
    report(vm, (tv_off)(op_ip - code), "fatal error: ");
    return TILVM_ERROR;
}
#if defined(__SDCC) && !defined(TILVM_LEAN)
#pragma restore
#endif

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
#ifndef TILVM_NO_GFX
    /* a screen this build cannot show (another TILVM_GFX_MAX_W/H) is a different
     * build; the half-restored program must not run with it */
    if ((vm->st.scr_w == 0) != (vm->st.scr_h == 0) || vm->st.scr_w > TILVM_GFX_MAX_W ||
        vm->st.scr_h > TILVM_GFX_MAX_H) {
        vm->st.scr_w = vm->st.scr_h = 0;
        vm->st.failed = 1;
        return -3;
    }
    if (vm->st.scr_w && vm->host.gfx_pixel) gfx_open(vm);   /* same size, pixels cleared */
#endif
    return 0;
}
#endif /* TILVM_LEAN */

/* ─── portable snapshots (.tilp, docs/TIL_SNAPSHOTS.md) ───────────────────
 * The byte layout, the object numbering (breadth-first from the roots, in
 * file order) and the canonical encodings are specified in the document; the
 * reference VM (TILVM.save_portable / load_portable in src/til.py) writes and
 * checks exactly the same.
 *
 * Saving numbers the objects in their headers: mark 1 = numbered (fwd = id),
 * 2 = a string with the same bytes as numbered string fwd, 3 = a string equal
 * to string constant fwd; strings also keep a 16-bit hash of their bytes in
 * aux. Afterwards mark and the strings' aux are 0 again (as outside the GC).
 * The queue of numbered objects not yet written is a ring in the free end of
 * the arena (or the GC's mark stack); objects that did not fit into it are
 * found again by a heap scan, as the GC's mark phase does after an overflow.
 *
 * Loading allocates the objects in table order and stores references as
 * TV_PEND values (u.r = object id) until every object exists; the id -> object
 * table grows down from the end of the arena (dropped when the objects need
 * that room: then a walk over the loaded objects finds them). */
#if defined(TILVM_LEAN) || defined(TILVM_NO_PORTABLE)
uint32_t tilvm_save_portable(tilvm *vm, tilvm_put_fn put, void *ctx) { (void)vm; (void)put; (void)ctx; return 0; }
int tilvm_load_portable(tilvm *vm, tilvm_get_fn get, void *ctx) { (void)vm; (void)get; (void)ctx; return TILVM_P_UNAVAILABLE; }
#else
enum { TP_NIL, TP_FALSE, TP_TRUE, TP_INT, TP_FLOAT, TP_CONST, TP_REF, TP_FUNC, TP_UNDEF };
#define TV_PEND 13                  /* loading: a reference to object u.r (an id) */
#define FN_NUPVALS(p) ((p)[5])
/* object kind (OBJ_STR ... OBJ_CELL) -> value tag, and the .tilp object kind */
static const uint8_t TP_TAG[8] = { 0, TV_STR, TV_ARRAY, TV_MAP, 0, TV_STRUCT, TV_CLOSURE, TV_CELL };
static const uint8_t TP_KIND[8] = { 0, 1, 2, 3, 0, 4, 5, 6 };

static uint32_t crc32_add(uint32_t c, const uint8_t *p, uint32_t n) {   /* = zlib's crc32 */
    unsigned k;
    c = ~c;
    while (n--) {
        c ^= *p++;
        for (k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

#if TILVM_FLOAT && !(TILVM_FLOAT == 2 && DBL_MANT_DIG == 53)
/* binary32 floats (TILVM_FLOAT=1; also AVR's 32-bit double) <-> the binary64
 * bits of the file (hi, lo words), without a 64-bit double */
typedef char tv_check_float32[sizeof(tv_float) == 4 ? 1 : -1];
static void f32_bits(tv_float f, uint32_t *hi, uint32_t *lo) {
    uint32_t b, e, m;
    memcpy(&b, &f, 4);
    e = (b >> 23) & 0xFFu;
    m = b & 0x7FFFFFu;
    if (e == 0xFFu) e = 0x7FFu;                       /* inf / nan */
    else if (e) e += 1023u - 127u;
    else if (m) {                                     /* subnormal: normalize */
        e = 1023u - 126u;
        while (!(m & 0x800000u)) { m <<= 1; e--; }
        m &= 0x7FFFFFu;
    }
    *hi = (b & 0x80000000u) | (e << 20) | (m >> 3);
    *lo = m << 29;
}

static tv_float f32_from_bits(uint32_t hi, uint32_t lo) {   /* round to nearest even */
    uint32_t sign = hi & 0x80000000u, m, rest, b;
    int32_t e = (int32_t)((hi >> 20) & 0x7FFu);
    tv_float f;
    if (e == 0x7FF) b = sign | 0x7F800000u | (((hi & 0xFFFFFu) | lo) ? 0x400000u : 0u);
    else if (e == 0) b = sign;                        /* zero / double subnormal: far below */
    else {
        e += 127 - 1023;
        m = 0x800000u | ((hi & 0xFFFFFu) << 3) | (lo >> 29);   /* 24 bits */
        rest = lo & 0x1FFFFFFFu;                               /* 29 bits below them */
        if (e <= 0) {                                 /* a float subnormal */
            uint32_t s = (uint32_t)(1 - e), lost, half;
            if (s > 25) b = sign;
            else {
                lost = m & ((1u << s) - 1u);
                half = 1u << (s - 1u);
                m >>= s;
                if (lost > half || (lost == half && (rest || (m & 1u)))) m++;
                b = sign | m;
            }
        } else {
            if (rest > 0x10000000u || (rest == 0x10000000u && (m & 1u))) m++;
            if (m == 0x1000000u) { m >>= 1; e++; }
            b = e >= 0xFF ? sign | 0x7F800000u : sign | ((uint32_t)e << 23) | (m & 0x7FFFFFu);
        }
    }
    memcpy(&f, &b, 4);
    return f;
}
#endif

/* ─── saving ─── */
typedef struct tp_out {
    tilvm *vm;
    tilvm_put_fn put;
    void *ctx;
    uint32_t total, crc;
    int err;
    uint32_t n_ids, done, stored, qcap;    /* ids given; written; queued (done+1 .. stored) */
    tv_ref *q;                              /* ring: q[id % qcap] */
    tv_ref *ht;                             /* numbered strings by content (0 = empty slot) */
    uint32_t hmask, hn;                     /* size - 1 (0: no table), entries */
    int hfull;                              /* entries stopped: misses need a heap scan */
    uint16_t len;
    uint8_t buf[64];
} tp_out;

static void tp_put(tp_out *w, const uint8_t *p, uint32_t n) {
    if (w->err || !n) return;
    w->crc = crc32_add(w->crc, p, n);
    if (w->put(w->ctx, p, n) != 0) w->err = 1;
    w->total += n;
}

static void tp_flush(tp_out *w) {
    tp_put(w, w->buf, w->len);
    w->len = 0;
}

static void tp_bytes(tp_out *w, const uint8_t *p, uint32_t n) {
    if (n > sizeof(w->buf) - w->len) {
        tp_flush(w);
        if (n > sizeof(w->buf)) { tp_put(w, p, n); return; }
    }
    memcpy(w->buf + w->len, p, n);
    w->len = (uint16_t)(w->len + n);
}

static void tp_num(tp_out *w, uint32_t v, unsigned n) {      /* n <= 4 bytes, little endian */
    uint8_t b[4];
    unsigned i;
    for (i = 0; i < n; i++) { b[i] = (uint8_t)v; v >>= 8; }
    tp_bytes(w, b, n);
}

/* the records of the format: field sizes in bytes (0 ends) */
static const uint8_t TP_STATE[] = { 4, 4, 4, 2, 1, 1, 2, 2, 0 };  /* steps, rng lo/hi, tick, energy, 0, G, F */
static const uint8_t TP_FRAME[] = { 2, 4, 2, 1, 4, 0 };
static const uint8_t TP_TRY[] = { 4, 4, 2, 2, 2, 0 };
#ifndef TILVM_NO_EFFECTS
static const uint8_t TP_HANDLER[] = { 1, 2, 2, 2, 4, 2, 0 };
#endif

static void tp_rec_out(tp_out *w, const uint8_t *f, const uint32_t *v) {
    while (*f) tp_num(w, *v++, *f++);
}

/* first reference to heap object r: give it an id (a string equal to a constant
 * or to a numbered string gets theirs) */
/* is numbered string o (mark 1, aux = its hash) equal to s? */
static int tp_same_str(tilvm *vm, tv_ref o, const tv_strobj *s, uint16_t hash) {
    const tv_hdr *h = OBJ(vm, o);
    return h->kind == OBJ_STR && h->mark == 1 && h->aux == hash && STROBJ(vm, o)->len == s->len &&
           memcmp(STROBJ(vm, o)->data, s->data, s->len) == 0;
}

static void tp_number(tp_out *w, tv_ref r) {
    tilvm *vm = w->vm;
    tv_hdr *h = OBJ(vm, r);
    if (h->kind == OBJ_STR) {
        const tv_strobj *s = STROBJ(vm, r);
        uint32_t hash = 2166136261u, i = 0;
        tv_ref o = 0;
        tv_size scan;
        uint16_t k;
        for (k = 0; k < vm->n_consts; k++) {         /* a string constant with these bytes */
            const uint8_t *p = const_ptr(vm, k);
            if (p[-1] == 3 && RD16(p) == s->len && memcmp(p + 2, s->data, s->len) == 0) {
                h->mark = 3;
                h->fwd = (tv_ref)k;
                return;
            }
        }
        for (; i < s->len; i++) hash = (hash ^ (uint8_t)s->data[i]) * 16777619u;
        h->aux = (uint16_t)(hash ^ (hash >> 16));
        if (w->hmask) {                              /* a numbered string with these bytes */
            for (i = hash & w->hmask; (o = w->ht[i]) != 0; i = (i + 1) & w->hmask)
                if (tp_same_str(vm, o, s, h->aux)) break;
            if (!o && !w->hfull) {                   /* new: remember it */
                if (++w->hn > w->hmask - w->hmask / 4) w->hfull = 1;
                else w->ht[i] = r;
            }
        }
        if (!o && (!w->hmask || w->hfull))           /* no table (or a full one): look at all */
            for (scan = (tv_size)ALIGN_UP(1); scan < vm->st.heap_top; scan = (tv_size)(scan + OBJ(vm, scan)->size))
                if (tp_same_str(vm, scan, s, h->aux)) { o = scan; break; }
        if (o) {
            h->mark = 2;
            h->fwd = OBJ(vm, o)->fwd;
            return;
        }
    }
    h->mark = 1;
    h->fwd = (tv_ref)++w->n_ids;
    if (w->n_ids == w->stored + 1 && w->n_ids - w->done <= w->qcap) {
        w->q[w->n_ids % w->qcap] = r;
        w->stored = w->n_ids;
    }
}

/* object id of r for a frame's closure field (0: none, or no upvalues) */
static uint32_t tp_closure_id(tp_out *w, tv_ref r) {
    if (!r || !OBJ(w->vm, r)->aux) return 0;
    if (!OBJ(w->vm, r)->mark) tp_number(w, r);
    return OBJ(w->vm, r)->fwd;
}

static void tp_value(tp_out *w, const tv_val *v) {
    tilvm *vm = w->vm;
    uint8_t b[3];
    switch (v->t) {
    case TV_NIL: b[0] = TP_NIL; break;
    case TV_BOOL: b[0] = v->u.i ? TP_TRUE : TP_FALSE; break;
    case TV_INT:
        b[0] = TP_INT;
        tp_bytes(w, b, 1);
#if TILVM_INT_BITS == 64
        tp_num(w, (uint32_t)v->u.i, 4);
        tp_num(w, (uint32_t)((uint64_t)v->u.i >> 32), 4);
#else
        tp_num(w, (uint32_t)(int32_t)v->u.i, 4);
        tp_num(w, v->u.i < 0 ? 0xFFFFFFFFu : 0u, 4);
#endif
        return;
#if TILVM_FLOAT
    case TV_FLOAT: {
        uint32_t hi, lo;
#if TILVM_FLOAT == 2 && DBL_MANT_DIG == 53
        uint64_t bits;
        memcpy(&bits, &v->u.f, 8);
        lo = (uint32_t)bits;
        hi = (uint32_t)(bits >> 32);
#else
        f32_bits(v->u.f, &hi, &lo);
#endif
        b[0] = TP_FLOAT;
        tp_bytes(w, b, 1);
        tp_num(w, lo, 4);
        tp_num(w, hi, 4);
        return;
    }
#endif
    case TV_STRK: b[0] = TP_CONST; b[1] = (uint8_t)v->u.k; b[2] = (uint8_t)(v->u.k >> 8); tp_bytes(w, b, 3); return;
    case TV_FUNC: b[0] = TP_FUNC; b[1] = (uint8_t)v->u.k; b[2] = (uint8_t)(v->u.k >> 8); tp_bytes(w, b, 3); return;
    case TV_UNDEF: b[0] = TP_UNDEF; break;
    case TV_STR: case TV_ARRAY: case TV_MAP: case TV_STRUCT: case TV_CLOSURE: case TV_CELL: {
        tv_hdr *h = OBJ(vm, v->u.r);
        if (h->kind > OBJ_CELL || TP_TAG[h->kind] != v->t) { w->err = 1; return; }
        if (h->kind == OBJ_CLOSURE && !h->aux) {      /* no upvalues: a function value */
            tv_val f;
            f.t = TV_FUNC;
            f.u.i = 0;
            f.u.k = CLOSOBJ(vm, v->u.r)->func;
            tp_value(w, &f);
            return;
        }
        if (!h->mark) tp_number(w, v->u.r);
        if (h->mark == 3) {
            b[0] = TP_CONST; b[1] = (uint8_t)h->fwd; b[2] = (uint8_t)(h->fwd >> 8);
            tp_bytes(w, b, 3);
            return;
        }
        b[0] = TP_REF;
        tp_bytes(w, b, 1);
        tp_num(w, (uint32_t)h->fwd, 4);
        return;
    }
    default: w->err = 1; return;        /* not a value a paused program holds */
    }
    tp_bytes(w, b, 1);
}

static void tp_values(tp_out *w, const tv_val *v, uint32_t n) {
    while (n-- && !w->err) tp_value(w, v++);
}

uint32_t tilvm_save_portable(tilvm *vm, tilvm_put_fn put, void *ctx) {
    tilvm_state *st = &vm->st;
    tp_out w;
    uint16_t i, g = vm->n_globals;
    tv_size scan;
    uint32_t v[8];
    uint8_t crc[4];
    if (st->halted || st->failed || !st->n_frames || st->sp < g) return 0;
    memset(&w, 0, sizeof w);
    w.vm = vm;
    w.put = put;
    w.ctx = ctx;
    {   /* the queue (and a hash table of the strings, when there is room for both) in
         * the free end of the arena; when it is tiny, the queue is the GC's mark stack */
        uint32_t at = ((uint32_t)st->heap_top + sizeof(tv_ref) - 1u) / sizeof(tv_ref) * sizeof(tv_ref);
        uint32_t room = at < (uint32_t)TILVM_HEAP_SIZE ? ((uint32_t)TILVM_HEAP_SIZE - at) / sizeof(tv_ref) : 0;
        tv_ref *free_end = (tv_ref *)(void *)(vm->mem.heap + at);
        if (room >= 4u * TILVM_MARK_STACK && room >= 64u) {
            uint32_t n = 32;
            while (n * 4u <= room) n *= 2;           /* the table: a quarter to a half of the room */
#ifdef TILVM_P_SCRATCH
            n = 4;                                   /* tests: a table that fills up (then scans) */
#endif
            w.ht = free_end;
            w.hmask = n - 1u;
            memset(w.ht, 0, n * sizeof(tv_ref));
            free_end += n;
            room -= n;
        }
        if (room > TILVM_MARK_STACK) { w.q = free_end; w.qcap = room; }
        else { w.q = vm->mark_stack; w.qcap = TILVM_MARK_STACK; }
#ifdef TILVM_P_SCRATCH         /* tests: a queue of this many entries (heap rescans) */
        if (w.qcap > TILVM_P_SCRATCH) w.qcap = TILVM_P_SCRATCH;
#endif
    }
    tp_bytes(&w, (const uint8_t *)"TILP\1\0", 6);
    tp_num(&w, vm->bc_hash, 4);
    v[0] = st->steps;
#if TILVM_INT_BITS == 64
    v[1] = (uint32_t)st->rng;
    v[2] = (uint32_t)(st->rng >> 32);
#else
    v[1] = st->rng;
    v[2] = 0;
#endif
    v[3] = st->tick;
    v[4] = st->energy;                      /* energy mode (design §11) */
    v[5] = 0;                               /* reserved */
    v[6] = g;
    v[7] = st->n_frames;
    tp_rec_out(&w, TP_STATE, v);
    for (i = 0; i < st->n_frames; i++) {
        const tv_frame *f = &st->frames[i];
        v[0] = f->func;
        v[1] = (uint32_t)(f->pc - vm->code_off);
        v[2] = (uint32_t)(f->base - g);
        v[3] = f->extra;
        v[4] = tp_closure_id(&w, f->closure);
        tp_rec_out(&w, TP_FRAME, v);
    }
    tp_num(&w, st->n_try, 2);
    for (i = 0; i < st->n_try; i++) {
        const tv_try *t = &st->tries[i];
        v[0] = (uint32_t)(t->handler - vm->code_off);
        v[1] = 0;
        if (t->deadline) {                  /* time left; at least 1 (0 = no deadline) */
            int32_t d = (int32_t)(t->deadline - now_ms(vm) - 1u);
            v[1] = d < 1 ? 1u : (uint32_t)d;
        }
        v[2] = t->n_frames;
        v[3] = (uint32_t)(t->sp - g);
        v[4] = t->n_handlers;
        tp_rec_out(&w, TP_TRY, v);
    }
    tp_num(&w, st->n_handlers, 2);
#ifndef TILVM_NO_EFFECTS
    for (i = 0; i < st->n_handlers; i++) {
        const tv_handler *h = &st->handlers[i];
        v[0] = h->kind;
        v[1] = h->n_frames;
        v[2] = h->kind ? 0u : (uint32_t)(h->sp - g);
        v[3] = h->n_try;
        v[4] = h->kind ? 0u : (uint32_t)(h->abort_pc - vm->code_off);
        v[5] = h->n;
        tp_rec_out(&w, TP_HANDLER, v);
    }
#endif
    tp_values(&w, st->stack, g);
    tp_num(&w, (uint32_t)(st->sp - g), 2);
    tp_values(&w, st->stack + g, (uint32_t)(st->sp - g));
    tp_num(&w, st->scr_w, 2);
    tp_num(&w, st->scr_h, 2);
    while (w.done < w.n_ids && !w.err) {    /* the object table, in id order */
        uint32_t id = w.done + 1, n;
        tv_ref r;
        tv_hdr *h;
        if (id > w.stored) {                /* not queued: find ids id .. id+qcap-1 again */
            uint32_t hi = w.n_ids - w.done < w.qcap ? w.n_ids : w.done + w.qcap;
            for (scan = (tv_size)ALIGN_UP(1); scan < st->heap_top; scan = (tv_size)(scan + OBJ(vm, scan)->size)) {
                h = OBJ(vm, scan);
                if (h->mark == 1 && h->fwd >= id && h->fwd <= hi) w.q[h->fwd % w.qcap] = scan;
            }
            w.stored = hi;
        }
        r = w.q[id % w.qcap];
        w.done = id;
        h = OBJ(vm, r);
        tp_num(&w, TP_KIND[h->kind], 1);
        switch (h->kind) {
        case OBJ_STR:
            tp_num(&w, STROBJ(vm, r)->len, 4);
            tp_bytes(&w, (const uint8_t *)STROBJ(vm, r)->data, STROBJ(vm, r)->len);
            break;
        case OBJ_ARRAY: case OBJ_MAP:
            n = ARROBJ(vm, r)->len;
            tp_num(&w, n, 4);
            tp_values(&w, VALS(vm, ARROBJ(vm, r)->buf), h->kind == OBJ_MAP ? 2 * n : n);
            break;
        case OBJ_STRUCT:
            n = (uint32_t)((h->size - offsetof(tv_structobj, v)) / sizeof(tv_val));
            tp_num(&w, h->aux, 2);
            tp_num(&w, n, 2);
            tp_values(&w, STRUCTOBJ(vm, r)->v, n);
            break;
        case OBJ_CLOSURE:
            tp_num(&w, CLOSOBJ(vm, r)->func, 2);
            tp_num(&w, h->aux, 2);
            tp_values(&w, CLOSOBJ(vm, r)->v, h->aux);
            break;
        default:
            tp_value(&w, &CELLOBJ(vm, r)->v);
            break;
        }
    }
    tp_num(&w, w.n_ids, 4);
    tp_flush(&w);
    for (i = 0; i < 4; i++) crc[i] = (uint8_t)(w.crc >> (8 * i));
    tp_put(&w, crc, 4);
    /* the headers as they were */
    for (scan = (tv_size)ALIGN_UP(1); scan < st->heap_top; scan = (tv_size)(scan + OBJ(vm, scan)->size)) {
        tv_hdr *h = OBJ(vm, scan);
        h->mark = 0;
        if (h->kind == OBJ_STR) h->aux = 0;
    }
    return w.err ? 0 : w.total;
}

/* ─── loading ─── */
typedef struct tp_in {
    tilvm *vm;
    tilvm_get_fn get;
    void *ctx;
    uint32_t crc, n_ids;                    /* ids referenced so far */
    int err;
    int tab_on;                             /* the id -> object table at the arena's end */
    uint32_t tab_n, cur_id;                 /* table entries; walk position (cur_id: cur_ref) */
    tv_ref cur_ref;
} tp_in;

static void tp_fail(tp_in *r, int code) {
    if (!r->err) r->err = code;
}

static uint32_t tp_rd(tp_in *r, unsigned n) {           /* n <= 4 bytes, little endian */
    uint8_t b[4];
    uint32_t v = 0;
    if (r->err) return 0;
    if (r->get(r->ctx, b, n) != 0) { tp_fail(r, TILVM_P_CORRUPT); return 0; }
    r->crc = crc32_add(r->crc, b, n);
    while (n--) v = (v << 8) | b[n];
    return v;
}

static void tp_rec_in(tp_in *r, const uint8_t *f, uint32_t *v) {
    while (*f) *v++ = tp_rd(r, *f++);
}

static void tp_get_value(tp_in *r, tv_val *dst) {
    tilvm *vm = r->vm;
    uint32_t tag = tp_rd(r, 1), lo, hi, k;
    set_nil(dst);
    switch (tag) {
    case TP_NIL: return;
    case TP_FALSE: case TP_TRUE: set_bool(dst, tag == TP_TRUE); return;
    case TP_INT:
        lo = tp_rd(r, 4);
        hi = tp_rd(r, 4);
#if TILVM_INT_BITS == 64
        set_int(dst, (tv_int)(((uint64_t)hi << 32) | lo));
#else
        if (hi != ((lo & 0x80000000u) ? 0xFFFFFFFFu : 0u)) { tp_fail(r, TILVM_P_INT_RANGE); return; }
#if TILVM_INT_BITS == 16
        if (lo + 0x8000u > 0xFFFFu) { tp_fail(r, TILVM_P_INT_RANGE); return; }
#endif
        set_int(dst, (tv_int)(int32_t)lo);
#endif
        return;
    case TP_FLOAT:
        lo = tp_rd(r, 4);
        hi = tp_rd(r, 4);
#if !TILVM_FLOAT
        (void)lo; (void)hi;
        tp_fail(r, TILVM_P_NO_FLOAT);
#elif TILVM_FLOAT == 2 && DBL_MANT_DIG == 53
        {
            uint64_t bits = ((uint64_t)hi << 32) | lo;
            tv_float f;
            memcpy(&f, &bits, 8);
            set_float(dst, f);
        }
#else
        set_float(dst, f32_from_bits(hi, lo));
#endif
        return;
    case TP_CONST: case TP_FUNC:
        k = tp_rd(r, 2);
        if (tag == TP_CONST ? k >= vm->n_consts || vm->bc[vm->const_off[k] - 1] != 3 : k >= vm->n_funcs) break;
        dst->t = tag == TP_CONST ? TV_STRK : TV_FUNC;
        dst->u.k = (uint16_t)k;
        return;
    case TP_REF:
        k = tp_rd(r, 4);
        if (r->err) return;
        if (!k || k > r->n_ids + 1) break;          /* ids come in order */
        if (k > (uint32_t)TILVM_HEAP_SIZE / 8u) { tp_fail(r, TILVM_P_TOO_BIG); return; }   /* objects >= 8 bytes */
        if (k > r->n_ids) r->n_ids = k;
        dst->t = TV_PEND;
        dst->u.r = (tv_ref)k;
        return;
    case TP_UNDEF: dst->t = TV_UNDEF; return;
    default: break;
    }
    tp_fail(r, TILVM_P_CORRUPT);
}

/* a new object (no GC while loading: references are ids until the end) */
static tv_ref tp_alloc(tp_in *r, uint8_t kind, uint32_t size) {
    tilvm *vm = r->vm;
    uint32_t top = vm->st.heap_top, limit;
    tv_ref ref;
    if (r->err) return 0;
    size = (uint32_t)ALIGN_UP(size);
    limit = r->tab_on ? (uint32_t)TILVM_HEAP_SIZE - r->tab_n * (uint32_t)sizeof(tv_ref) : (uint32_t)TILVM_HEAP_SIZE;
    if (size > (uint32_t)TILVM_HEAP_SIZE - top) { tp_fail(r, TILVM_P_TOO_BIG); return 0; }
    if (top + size > limit) r->tab_on = 0;            /* the objects need the table's room */
    ref = (tv_ref)top;
    vm->st.heap_top = (tv_size)(top + size);
    memset(vm->mem.heap + ref, 0, size);
    OBJ(vm, ref)->kind = kind;
    OBJ(vm, ref)->size = (tv_size)size;
    return ref;
}

static tv_ref tp_lookup(tp_in *r, uint32_t id) {        /* object `id` (1 ..) */
    tilvm *vm = r->vm;
    tv_ref ref;
    if (r->tab_on) {
        memcpy(&ref, vm->mem.heap + TILVM_HEAP_SIZE - id * sizeof(tv_ref), sizeof ref);
        return ref;
    }
    if (!r->cur_id || id < r->cur_id) { r->cur_id = 1; r->cur_ref = (tv_ref)ALIGN_UP(1); }
    while (r->cur_id < id) {                            /* objects lie in id order (+ buffers) */
        do r->cur_ref = (tv_ref)(r->cur_ref + OBJ(vm, r->cur_ref)->size);
        while (OBJ(vm, r->cur_ref)->kind == OBJ_VALS);
        r->cur_id++;
    }
    return r->cur_ref;
}

static void tp_patch(tp_in *r, tv_val *v, uint32_t n) {
    for (; n--; v++)
        if (v->t == TV_PEND) {
            tv_ref o = tp_lookup(r, v->u.r);
            v->t = TP_TAG[OBJ(r->vm, o)->kind];
            v->u.r = o;
        }
}

int tilvm_load_portable(tilvm *vm, tilvm_get_fn get, void *ctx) {
    tilvm_state *st = &vm->st;
    tp_in r;
    uint32_t state[8], v[6], n, i, k, depth, w, h, n_read = 0, crc;
    uint16_t g = vm->n_globals, nf, nt, nh;
    tv_size scan;
    memset(&r, 0, sizeof r);
    r.vm = vm;
    r.get = get;
    r.ctx = ctx;
    r.tab_on = 1;
    reset_state(vm);
    st->sp = 0;
    if (tp_rd(&r, 4) != 0x504C4954u) { tp_fail(&r, TILVM_P_NOT_TILP); goto fail; }     /* "TILP" */
    if (tp_rd(&r, 2) != 1u) tp_fail(&r, TILVM_P_VERSION);                              /* version 1, flags 0 */
    if (!r.err && tp_rd(&r, 4) != vm->bc_hash) tp_fail(&r, TILVM_P_PROGRAM);
    tp_rec_in(&r, TP_STATE, state);         /* steps, rng, tick, energy mode, 0, G, F */
    if (state[4] > 2 || state[3] > 255 || state[6] != g) tp_fail(&r, TILVM_P_CORRUPT);
    /* frames (closure: the object id until the end), recovery points, handler records */
    nf = (uint16_t)state[7];
    if (!nf) tp_fail(&r, TILVM_P_CORRUPT);
    if (nf > TILVM_MAX_FRAMES) tp_fail(&r, TILVM_P_TOO_BIG);
    for (i = 0; i < nf && !r.err; i++) {
        tv_frame *f = &st->frames[i];
        tp_rec_in(&r, TP_FRAME, v);
        f->func = (uint16_t)v[0];
        f->pc = (tv_off)(vm->code_off + v[1]);
        f->base = (uint16_t)v[2];
        f->extra = (uint8_t)v[3];
        f->ret_pc = 0;
        if (v[0] >= vm->n_funcs || v[1] >= vm->code_size || v[3] > 1) tp_fail(&r, TILVM_P_CORRUPT);
        k = v[4];
        if (k > r.n_ids + 1) tp_fail(&r, TILVM_P_CORRUPT);                  /* ids come in order */
        else if (k > (uint32_t)TILVM_HEAP_SIZE / 8u) tp_fail(&r, TILVM_P_TOO_BIG);
        else if (k > r.n_ids) r.n_ids = k;
        f->closure = (tv_ref)k;
    }
    nt = (uint16_t)tp_rd(&r, 2);
    if (nt > TILVM_MAX_TRY) tp_fail(&r, TILVM_P_TOO_BIG);
    for (i = 0; i < nt && !r.err; i++) {
        tv_try *t = &st->tries[i];
        tp_rec_in(&r, TP_TRY, v);
        t->handler = (tv_off)(vm->code_off + v[0]);
        t->deadline = v[1];                            /* time left (0 = none) until the end */
        t->n_frames = (uint16_t)v[2];
        t->sp = (uint16_t)v[3];
        t->n_handlers = (uint16_t)v[4];
        if (v[0] >= vm->code_size || !v[2] || v[2] > nf || (i && v[2] < t[-1].n_frames))
            tp_fail(&r, TILVM_P_CORRUPT);
    }
    nh = (uint16_t)tp_rd(&r, 2);
#ifdef TILVM_NO_EFFECTS
    if (nh) tp_fail(&r, TILVM_P_NO_EFFECTS);
#else
    if (nh > TILVM_MAX_HANDLERS) tp_fail(&r, TILVM_P_TOO_BIG);
    for (i = 0; i < nh && !r.err; i++) {
        tv_handler *e = &st->handlers[i];
        tp_rec_in(&r, TP_HANDLER, v);
        e->kind = (uint8_t)v[0];
        e->n_frames = (uint16_t)v[1];
        e->sp = (uint16_t)v[2];
        e->n_try = (uint16_t)v[3];
        k = v[4];
        e->n = (uint16_t)v[5];
        e->abort_pc = (tv_off)(e->kind ? 0u : vm->code_off + k);
        e->_pad[0] = e->_pad[1] = e->_pad[2] = 0;
        if (v[0] > 1 || e->n_frames > nf || e->n_try > nt || (i && e->n_frames < e[-1].n_frames) ||
            (e->kind == 0 && (!e->n_frames || k >= vm->code_size)) ||
            (e->kind == 1 && (!e->n_frames || e->n_frames >= nf || e->n >= i || st->handlers[e->n].kind ||
                              e->sp || k)))
            tp_fail(&r, TILVM_P_CORRUPT);
    }
#endif
    /* values: the globals, then the value stack */
    for (i = 0; i < g && !r.err; i++) tp_get_value(&r, &st->stack[i]);
    depth = tp_rd(&r, 2);
    if ((uint32_t)g + depth > TILVM_STACK_SIZE) tp_fail(&r, TILVM_P_TOO_BIG);
    for (i = 0; i < depth && !r.err; i++) tp_get_value(&r, &st->stack[g + i]);
    if (r.err) goto fail;
    st->sp = (uint16_t)(g + depth);
    for (i = 0; i < nf; i++) {          /* locals below the next frame's arguments */
        tv_frame *f = &st->frames[i];
        uint32_t top = i + 1 < nf ? (uint32_t)f[1].base - f[1].extra : depth;
        if (f->base < f->extra || (!i && f->base) || f->base + (uint32_t)F_NLOCALS(vm, f->func) > top ||
            (i + 1 < nf && f[1].base < f[1].extra))
            tp_fail(&r, TILVM_P_CORRUPT);
    }
    for (i = 0; i < nt; i++)
        if (st->tries[i].sp > depth || st->tries[i].n_handlers > nh) tp_fail(&r, TILVM_P_CORRUPT);
#ifndef TILVM_NO_EFFECTS
    for (i = 0; i < nh; i++)
        if (!st->handlers[i].kind && st->handlers[i].sp + 2u * st->handlers[i].n > depth) tp_fail(&r, TILVM_P_CORRUPT);
#endif
    w = tp_rd(&r, 2);
    h = tp_rd(&r, 2);
    if ((w == 0) != (h == 0) || w > 4096 || h > 4096) tp_fail(&r, TILVM_P_CORRUPT);
#ifdef TILVM_NO_GFX
    if (w) tp_fail(&r, TILVM_P_SCREEN);
#else
    if (w > TILVM_GFX_MAX_W || h > TILVM_GFX_MAX_H) tp_fail(&r, TILVM_P_SCREEN);
#endif
    /* the object table */
    while (n_read < r.n_ids && !r.err) {
        uint32_t kind = tp_rd(&r, 1);
        tv_ref o = 0;
        tv_val *vals = 0;
        n = 0;
        switch (kind) {
        case 1:                                         /* string */
            n = tp_rd(&r, 4);
            if (n > TV_STR_MAX) { tp_fail(&r, TILVM_P_TOO_BIG); break; }
            o = tp_alloc(&r, OBJ_STR, (uint32_t)offsetof(tv_strobj, data) + n + 1);
            if (!o) break;
            STROBJ(vm, o)->len = (tv_size)n;
            if (n && get(ctx, (uint8_t *)STROBJ(vm, o)->data, n) != 0) tp_fail(&r, TILVM_P_CORRUPT);
            else r.crc = crc32_add(r.crc, (const uint8_t *)STROBJ(vm, o)->data, n);
            n = 0;
            break;
        case 2: case 3: {                               /* array, map */
            uint32_t len = tp_rd(&r, 4), cap;
            tv_ref buf;
            n = kind == 3 ? 2 * len : len;
            if (len > 0xFFFFu || n > 0xFFFFu) { tp_fail(&r, TILVM_P_TOO_BIG); break; }
            o = tp_alloc(&r, kind == 3 ? OBJ_MAP : OBJ_ARRAY, sizeof(tv_arrobj));
            cap = n < 4 ? 4 : n;
            buf = tp_alloc(&r, OBJ_VALS, (uint32_t)offsetof(tv_valsobj, v) + cap * (uint32_t)sizeof(tv_val));
            if (!buf) break;
            OBJ(vm, buf)->aux = (uint16_t)cap;
            for (k = 0; k < cap; k++) set_nil(&VALS(vm, buf)[k]);
            ARROBJ(vm, o)->buf = buf;
            ARROBJ(vm, o)->len = (tv_size)len;
            vals = VALS(vm, buf);
            break;
        }
        case 4: case 5: {                               /* struct, closure */
            uint32_t a = tp_rd(&r, 2);
            n = tp_rd(&r, 2);
            if (r.err) break;
            if (kind == 4 ? a >= vm->n_types || n != type_nfields(vm, (uint16_t)a)
                          : a >= vm->n_funcs || n != FN_NUPVALS(func_rec(vm, (uint16_t)a))) {
                tp_fail(&r, TILVM_P_CORRUPT);
                break;
            }
            o = tp_alloc(&r, kind == 4 ? OBJ_STRUCT : OBJ_CLOSURE,
                         (uint32_t)(kind == 4 ? offsetof(tv_structobj, v) : offsetof(tv_closobj, v)) +
                         n * (uint32_t)sizeof(tv_val));
            if (!o) break;
            if (kind == 4) { OBJ(vm, o)->aux = (uint16_t)a; vals = STRUCTOBJ(vm, o)->v; }
            else { OBJ(vm, o)->aux = (uint16_t)n; CLOSOBJ(vm, o)->func = (uint16_t)a; vals = CLOSOBJ(vm, o)->v; }
            break;
        }
        case 6:                                         /* cell */
            o = tp_alloc(&r, OBJ_CELL, sizeof(tv_cellobj));
            if (o) vals = &CELLOBJ(vm, o)->v;
            n = 1;
            break;
        default:
            tp_fail(&r, TILVM_P_CORRUPT);
        }
        for (k = 0; k < n && !r.err; k++) tp_get_value(&r, &vals[k]);
        if (r.err) break;
        n_read++;
        if (r.tab_on) {                                 /* id -> object, at the arena's end */
#ifdef TILVM_P_SCRATCH         /* tests: an id table of this many entries (then walks) */
            if (n_read > TILVM_P_SCRATCH) r.tab_on = 0;
            else
#endif
            if (n_read * (uint32_t)sizeof(tv_ref) <= (uint32_t)TILVM_HEAP_SIZE - st->heap_top) {
                memcpy(vm->mem.heap + TILVM_HEAP_SIZE - n_read * sizeof(tv_ref), &o, sizeof o);
                r.tab_n = n_read;
            } else r.tab_on = 0;
        }
    }
    if (tp_rd(&r, 4) != r.n_ids) tp_fail(&r, TILVM_P_CORRUPT);
    crc = r.crc;
    if (tp_rd(&r, 4) != crc) tp_fail(&r, TILVM_P_CORRUPT);
    if (r.err) goto fail;
    /* references: ids -> objects; check what the program relies on */
    tp_patch(&r, st->stack, st->sp);
    for (i = 0; i < nf; i++) {
        tv_frame *f = &st->frames[i];
        f->base = (uint16_t)(f->base + g);
        if (f->closure) {
            f->closure = tp_lookup(&r, f->closure);
            if (OBJ(vm, f->closure)->kind != OBJ_CLOSURE || CLOSOBJ(vm, f->closure)->func != f->func)
                tp_fail(&r, TILVM_P_CORRUPT);
        }
    }
    for (scan = (tv_size)ALIGN_UP(1); scan < st->heap_top; scan = (tv_size)(scan + OBJ(vm, scan)->size)) {
        tv_hdr *o = OBJ(vm, scan);
        tv_val *vals;
        n = obj_nvals(vm, scan, &vals);
        if (o->kind == OBJ_MAP) {                       /* its buffer: keys are scalars or strings */
            tv_ref buf = ARROBJ(vm, scan)->buf;
            tv_val *kv = VALS(vm, buf);
            tp_patch(&r, kv, OBJ(vm, buf)->aux);
            for (k = 0; k < ARROBJ(vm, scan)->len; k++) {
                uint8_t t = kv[2 * k].t;
                if (t != TV_NIL && t != TV_BOOL && t != TV_INT && t != TV_FLOAT && !is_str(&kv[2 * k]))
                    tp_fail(&r, TILVM_P_CORRUPT);
            }
        }
        tp_patch(&r, vals, n);
        if (o->kind == OBJ_CLOSURE)
            for (k = 0; k < n; k++)
                if (vals[k].t != TV_CELL) tp_fail(&r, TILVM_P_CORRUPT);
    }
    if (r.err) goto fail;
    /* the state */
    st->n_frames = nf;
    st->n_try = nt;
    st->n_handlers = nh;
    st->steps = state[0];
#if TILVM_INT_BITS == 64
    st->rng = ((uint64_t)state[2] << 32) | state[1];
#else
    st->rng = state[1];
#endif
    if (!st->rng) st->rng = 0x9E3779B9u;
    st->tick = state[3];
    st->energy = (uint8_t)state[4];         /* energy mode (design §11) */
    st->n_deadlines = 0;
    for (i = 0; i < nt; i++) {
        tv_try *t = &st->tries[i];
        t->sp = (uint16_t)(t->sp + g);
        if (t->deadline) { t->deadline = now_ms(vm) + t->deadline + 1u; st->n_deadlines++; }
    }
#ifndef TILVM_NO_EFFECTS
    for (i = 0; i < nh; i++)
        if (!st->handlers[i].kind) st->handlers[i].sp = (uint16_t)(st->handlers[i].sp + g);
#endif
    st->scr_w = (uint16_t)w;
    st->scr_h = (uint16_t)h;
#ifndef TILVM_NO_GFX
    if (w && vm->host.gfx_pixel) gfx_open(vm);         /* the size is kept, the pixels are not */
#endif
    TV_STAT_MAX(peak_heap, st->heap_top);
    return 0;
fail:
    reset_state(vm);
    st->failed = 1;
    return r.err;
}
#endif /* TILVM_LEAN, TILVM_NO_PORTABLE */
