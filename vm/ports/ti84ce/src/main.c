/*
 * TIL VM for the TI-84 Plus CE (eZ80, CE C toolchain "CEdev").
 *
 * Runs TIL bytecode stored in an AppVar (make one with tbc2appvar.py). The
 * program lists every AppVar whose data starts with the bytecode magic "TILB";
 * with several, pick one with the arrow keys and [enter]. [on] stops a running
 * program, [clear] leaves the menu.
 *
 *   console   print() -> OS home screen via outchar() (wraps and scrolls)
 *   keys      key_pressed()/wait_key() <- os_GetCSC(), mapped to TIL key codes
 *   screen    graphics hooks -> graphx (320 x 240, 8-bit palette RRRGGGBB),
 *             the TIL screen is scaled by an integer factor and centred
 *   files     read_file()/write_file() <- AppVars (name = path, 1-8 characters)
 *   time      clock_ms() <- clock() (32768 Hz timer), idle -> delay()
 *
 * The bytecode is copied to RAM (malloc), so creating AppVars from the program
 * cannot move it. See README.md.
 */
#include <stdint.h>
#include <stdio.h>          /* outchar() only - the VM itself is built without stdio */
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ti/screen.h>
#include <ti/getcsc.h>
#include <sys/timers.h>
#include <keypadc.h>
#include <fileioc.h>
#include <graphx.h>

#include "tilvm.h"

#define MAX_PROGRAMS 16
#define TEXT_KEEP 1024          /* output kept while the graphics screen is shown */

static tilvm vm;

/* ─── console ────────────────────────────────────────────────────────────── */
static uint8_t gfx_active;
static char held[TEXT_KEEP];    /* text printed while graphx owns the LCD */
static unsigned held_len;

/* outchar(): OS _PutC (wraps, scrolls) or _NewLine. UTF-8 is shown as one '?'
 * per character: the OS font is not Unicode. */
static void con_write(const char *data, size_t len) {
    while (len--) {
        unsigned char c = (unsigned char)*data++;
        if (c >= 0x80 && c < 0xC0) continue;           /* continuation byte */
        outchar(c >= 0x80 || (c < 0x20 && c != '\n') ? '?' : (char)c);
    }
}

static void vm_write(void *ctx, const char *data, size_t len) {
    (void)ctx;
    if (!gfx_active) { con_write(data, len); return; }
    while (len--) {                                     /* keep the tail for later */
        if (held_len == TEXT_KEEP) { memmove(held, held + 1, TEXT_KEEP - 1); held_len--; }
        held[held_len++] = *data++;
    }
}

static void con_str(const char *s) { con_write(s, strlen(s)); }

/* ─── keys ───────────────────────────────────────────────────────────────── */
/* os_GetCSC scan code -> TIL key code (UP=1 DOWN=2 LEFT=3 RIGHT=4, ENTER=13,
 * ESC=27 for [clear], BACKSPACE=8 for [del], digits and operators as ASCII,
 * letters as printed above the keys in alpha mode, lower case) */
static const uint8_t KEYMAP[57] = {
    /* 00 */ 0, 2, 3, 4, 1, 0, 0, 0, 0, 13, '+', '-', '*', '/', '^', 27,
    /* 10 */ 0, '-', '3', '6', '9', ')', 'g', 0, 0, '.', '2', '5', '8', '(', 'f', 'c',
    /* 20 */ 0, '0', '1', '4', '7', ',', 'e', 'b', 0, 0, 'x', 's', 'n', 'i', 'd', 'a',
    /* 30 */ 0, 0, 0, 0, 0, 0, 0, 0, 8
};
/* sk_Sin=0x26 'e', sk_Cos=0x1E 'f', sk_Tan=0x16 'g', sk_Power=0x0E '^' (h with alpha),
 * sk_Square=0x2D 'i', sk_Recip=0x2E 'd', sk_Math=0x2F 'a', sk_Apps=0x27 'b',
 * sk_Prgm=0x1F 'c', sk_Store=0x2A 'x', sk_Ln=0x2B 's', sk_Log=0x2C 'n' */

static int map_key(uint8_t sc) { return sc < sizeof KEYMAP ? KEYMAP[sc] : 0; }

static int ce_key_pressed(void *ctx) {
    (void)ctx;
    return map_key(os_GetCSC());
}

static int ce_wait_key(void *ctx) {
    int k;
    (void)ctx;
    do {
        if (kb_On) return 27;                           /* [on] also ends a wait */
        k = map_key(os_GetCSC());
    } while (!k);
    return k;
}

/* ─── screen (graphx, 8-bit palette RRRGGGBB) ────────────────────────────── */
static uint16_t palette[256];
static uint8_t scale = 1;
static int org_x, org_y;

static unsigned expand3(unsigned v) { return (v << 5) | (v << 2) | (v >> 1); }

static void ce_gfx_init(void *ctx, int w, int h) {
    unsigned i, sx, sy;
    (void)ctx;
    if (!gfx_active) {
        for (i = 0; i < 256; i++)
            palette[i] = gfx_RGBTo1555(expand3(i >> 5), expand3((i >> 2) & 7), (i & 3) * 0x55);
        gfx_Begin();
        gfx_SetPalette(palette, sizeof palette, 0);
        gfx_SetDrawBuffer();
        gfx_active = 1;
    }
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    sx = 320u / (unsigned)w;
    sy = 240u / (unsigned)h;
    scale = (uint8_t)(sx < sy ? sx : sy);
    if (!scale) scale = 1;
    org_x = (320 - w * scale) / 2;
    org_y = (240 - h * scale) / 2;
    if (org_x < 0) org_x = 0;
    if (org_y < 0) org_y = 0;
    gfx_FillScreen(0);
}

static uint8_t rgb_index(unsigned long rgb) {
    return (uint8_t)(((rgb >> 16) & 0xE0) | ((rgb >> 11) & 0x1C) | ((rgb >> 6) & 0x03));
}

static void ce_gfx_pixel(void *ctx, int x, int y, unsigned long rgb) {
    int px = org_x + x * scale, py = org_y + y * scale;
    (void)ctx;
    if (px < 0 || py < 0 || px + scale > 320 || py + scale > 240) return;
    gfx_SetColor(rgb_index(rgb));
    if (scale == 1) gfx_SetPixel((uint24_t)px, (uint8_t)py);
    else gfx_FillRectangle_NoClip((uint24_t)px, (uint8_t)py, scale, scale);
}

static unsigned long ce_gfx_get(void *ctx, int x, int y) {
    int px = org_x + x * scale, py = org_y + y * scale;
    unsigned i;
    (void)ctx;
    if (px < 0 || py < 0 || px >= 320 || py >= 240) return 0;
    i = gfx_GetPixel((uint24_t)px, (uint8_t)py);
    return ((unsigned long)expand3(i >> 5) << 16) | ((unsigned long)expand3((i >> 2) & 7) << 8) | ((i & 3) * 0x55);
}

static void ce_gfx_show(void *ctx) {
    (void)ctx;
    gfx_BlitBuffer();
}

static void gfx_finish(void) {
    if (!gfx_active) return;
    gfx_BlitBuffer();
    while (!os_GetCSC() && !kb_On) { }                  /* show the last frame */
    gfx_End();
    gfx_active = 0;
    os_ClrHome();
    con_write(held, held_len);
    held_len = 0;
}

/* ─── files = AppVars ────────────────────────────────────────────────────── */
static int appvar_name(const char *path) {
    size_t n = strlen(path), i;
    if (n < 1 || n > 8) return 0;
    for (i = 0; i < n; i++) {
        char c = path[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) return 0;
    }
    return !(path[0] >= '0' && path[0] <= '9');
}

static long ce_read_file(void *ctx, const char *path, char *buf, unsigned long cap) {
    uint8_t h;
    long size;
    (void)ctx;
    if (!appvar_name(path) || !(h = ti_Open(path, "r"))) return -1;
    size = (long)ti_GetSize(h);
    if (buf && cap) {
        size_t want = (unsigned long)size < cap ? (size_t)size : (size_t)cap;
        if (ti_Read(buf, 1, want, h) != want) size = -1;
    }
    ti_Close(h);
    return size;
}

static int ce_write_file(void *ctx, const char *path, const char *data, unsigned long len) {
    uint8_t h;
    int ok;
    (void)ctx;
    if (!appvar_name(path) || len > 65000ul || !(h = ti_Open(path, "w"))) return -1;
    ok = ti_Write(data, 1, (size_t)len, h) == (size_t)len;
    ti_Close(h);
    return ok ? 0 : -1;
}

/* ─── time ───────────────────────────────────────────────────────────────── */
static uint32_t ce_clock_ms(void *ctx) {
    unsigned long t = (unsigned long)clock();
    (void)ctx;
    return (uint32_t)((t / CLOCKS_PER_SEC) * 1000ul + (t % CLOCKS_PER_SEC) * 1000ul / CLOCKS_PER_SEC);
}

static void ce_idle(void *ctx, unsigned long ms) {
    (void)ctx;
    delay((uint16_t)(ms > 1000 ? 1000 : ms));
}

/* ─── program selection and run ──────────────────────────────────────────── */
static char names[MAX_PROGRAMS][9];

static int find_programs(void) {
    void *pos = NULL;
    char *name;
    int n = 0;
    while (n < MAX_PROGRAMS && (name = ti_Detect(&pos, "TILB")) != NULL) {
        strncpy(names[n], name, 8);
        names[n][8] = 0;
        n++;
    }
    return n;
}

static int choose(int n) {
    int sel = 0, i;
    uint8_t k;
    for (;;) {
        os_ClrHome();
        con_str("TIL programs:\n");
        for (i = 0; i < n && i < 9; i++) {
            con_str(i == sel ? "> " : "  ");
            con_str(names[i]);
            con_str("\n");
        }
        do { k = os_GetCSC(); } while (!k);
        if (k == sk_Clear) return -1;
        if (k == sk_Enter) return sel;
        if (k == sk_Up && sel > 0) sel--;
        if (k == sk_Down && sel < n - 1 && sel < 8) sel++;
    }
}

static uint8_t *load(const char *name, uint32_t *size) {
    uint8_t h = ti_Open(name, "r");
    uint8_t *buf = NULL;
    if (!h) return NULL;
    *size = ti_GetSize(h);
    if (*size) buf = (uint8_t *)malloc(*size);
    if (buf && ti_Read(buf, 1, *size, h) != *size) { free(buf); buf = NULL; }
    ti_Close(h);
    return buf;
}

static int run(const char *name) {
    uint32_t size = 0;
    uint8_t *code = load(name, &size);
    int rc;
    os_ClrHome();
    if (!code) { con_str("cannot read AppVar "); con_str(name); con_str("\n"); return 1; }
    rc = tilvm_load(&vm, code, size);
    if (rc != 0) {
        con_str(rc == -3 ? "program uses floats\n" : rc == -6 ? "integer constant too large\n"
                : rc == -10 ? "program larger than 64 KB\n" : "not a TIL program for this VM\n");
        free(code);
        return 1;
    }
    vm.write = vm_write;
    vm.write_err = vm_write;
    vm.clock_ms = ce_clock_ms;
    vm.host.read_file = ce_read_file;
    vm.host.write_file = ce_write_file;
    vm.host.gfx_init = ce_gfx_init;
    vm.host.gfx_pixel = ce_gfx_pixel;
    vm.host.gfx_get = ce_gfx_get;
    vm.host.gfx_show = ce_gfx_show;
    vm.host.key_pressed = ce_key_pressed;
    vm.host.wait_key = ce_wait_key;
    vm.host.idle = ce_idle;
    kb_EnableOnLatch();
    kb_ClearOnLatch();
    while ((rc = tilvm_run_steps(&vm, 2000)) == TILVM_YIELD) {
        if (kb_On) {                                     /* [on]: stop the program */
            kb_ClearOnLatch();
            vm_write(0, "\nBREAK\n", 7);
            rc = TILVM_ERROR;
            break;
        }
    }
    gfx_finish();
    free(code);
    return rc == TILVM_EXIT ? vm.st.exit_code : rc == TILVM_DONE ? 0 : 1;
}

int main(void) {
    int n, pick;
    os_ClrHome();
    n = find_programs();
    if (n == 0) {
        con_str("No TIL programs.\nSend an AppVar made\nwith tbc2appvar.py\n");
    } else {
        pick = n == 1 ? 0 : choose(n);
        if (pick >= 0) run(names[pick]);
    }
    con_str("\n[press a key]");
    while (!os_GetCSC()) { }
    kb_DisableOnLatch();
    os_ClrHome();
    return 0;
}
