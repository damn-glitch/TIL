/*
 * Desktop stand-ins for the CE toolchain libraries used by ../src/main.c, so
 * the TI-84 Plus CE host code can be compiled and run by the test-suite without
 * the CE toolchain or an emulator. Behaviour of the real OS is approximated:
 *
 *   outchar        -> stdout; os_ClrHome writes a form feed (\f)
 *   os_GetCSC      <- scan codes from $TILCE_KEYS ("0x04,9,..."), then [clear]
 *   AppVars        <- files $TILCE_DIR/NAME.8xv in the real TI file format
 *                     (as written by tbc2appvar.py); "w" writes such a file
 *   graphx         -> 320x240 palette-index framebuffer; $TILCE_PPM gets the
 *                     last frame shown with gfx_BlitBuffer (P6, palette colours)
 *
 * Not part of the calculator build.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ti/screen.h"
#include "ti/getcsc.h"
#include "sys/timers.h"
#include "keypadc.h"
#include "fileioc.h"
#include "graphx.h"

int ce_shim_on_key;

/* ─── console and keys ───────────────────────────────────────────────────── */
void outchar(char c) { putchar(c); }
void os_ClrHome(void) { putchar('\f'); }
void delay(uint16_t msec) { (void)msec; }

uint8_t os_GetCSC(void) {
    static const char *p;
    static int started;
    char *end;
    unsigned long v;
    if (!started) { p = getenv("TILCE_KEYS"); started = 1; }
    if (!p || !*p) return sk_Clear;
    v = strtoul(p, &end, 0);
    p = *end == ',' ? end + 1 : end;
    return (uint8_t)v;
}

/* ─── AppVars as .8xv files ──────────────────────────────────────────────── */
#define MAX_OPEN 4
static struct { int used, writing; char name[9]; unsigned char *data; size_t size, pos; } files[MAX_OPEN + 1];

static const char *dir(void) { const char *d = getenv("TILCE_DIR"); return d ? d : "."; }

static unsigned char *read_8xv(const char *name, size_t *size) {
    char path[512];
    FILE *f;
    unsigned char *blob, *data;
    long n;
    snprintf(path, sizeof path, "%s/%s.8xv", dir(), name);
    if (!(f = fopen(path, "rb"))) return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    blob = malloc(n > 0 ? (size_t)n : 1);
    if (!blob || n < 74 || fread(blob, 1, (size_t)n, f) != (size_t)n || memcmp(blob, "**TI83F*", 8) != 0) {
        fclose(f);
        free(blob);
        return NULL;
    }
    fclose(f);
    *size = (size_t)(blob[72] | (blob[73] << 8));              /* AppVar size word */
    if (74 + *size > (size_t)n) { free(blob); return NULL; }
    data = malloc(*size ? *size : 1);
    memcpy(data, blob + 74, *size);
    free(blob);
    return data;
}

static void write_8xv(const char *name, const unsigned char *data, size_t size) {
    char path[512];
    unsigned char hdr[74];
    unsigned sum = 0;
    size_t i, entry = 17 + 2 + size;
    FILE *f;
    memset(hdr, 0, sizeof hdr);
    memcpy(hdr, "**TI83F*\x1a\x0a\x00", 11);
    hdr[53] = (unsigned char)entry; hdr[54] = (unsigned char)(entry >> 8);
    hdr[55] = 13;
    hdr[57] = (unsigned char)(size + 2); hdr[58] = (unsigned char)((size + 2) >> 8);
    hdr[59] = 0x15;
    strncpy((char *)hdr + 60, name, 8);
    hdr[70] = (unsigned char)(size + 2); hdr[71] = (unsigned char)((size + 2) >> 8);
    hdr[72] = (unsigned char)size; hdr[73] = (unsigned char)(size >> 8);
    for (i = 55; i < 74; i++) sum += hdr[i];
    for (i = 0; i < size; i++) sum += data[i];
    snprintf(path, sizeof path, "%s/%s.8xv", dir(), name);
    if (!(f = fopen(path, "wb"))) return;
    fwrite(hdr, 1, sizeof hdr, f);
    fwrite(data, 1, size, f);
    fputc((int)(sum & 255), f);
    fputc((int)((sum >> 8) & 255), f);
    fclose(f);
}

uint8_t ti_Open(const char *name, const char *mode) {
    int h;
    for (h = 1; h <= MAX_OPEN && files[h].used; h++) { }
    if (h > MAX_OPEN || strlen(name) > 8) return 0;
    memset(&files[h], 0, sizeof files[h]);
    strcpy(files[h].name, name);
    if (mode[0] == 'w') files[h].writing = 1;
    else if (!(files[h].data = read_8xv(name, &files[h].size))) return 0;
    files[h].used = 1;
    return (uint8_t)h;
}

int ti_Close(uint8_t h) {
    if (files[h].writing) write_8xv(files[h].name, files[h].data, files[h].size);
    free(files[h].data);
    memset(&files[h], 0, sizeof files[h]);
    return 1;
}

size_t ti_Read(void *data, size_t size, size_t count, uint8_t h) {
    size_t n = size * count, left = files[h].size - files[h].pos;
    if (n > left) n = left;
    memcpy(data, files[h].data + files[h].pos, n);
    files[h].pos += n;
    return size ? n / size : 0;
}

size_t ti_Write(const void *data, size_t size, size_t count, uint8_t h) {
    size_t n = size * count;
    files[h].data = realloc(files[h].data, files[h].size + n + 1);
    memcpy(files[h].data + files[h].size, data, n);
    files[h].size += n;
    return count;
}

uint16_t ti_GetSize(uint8_t h) { return (uint16_t)files[h].size; }

char *ti_Detect(void **vat_ptr, const char *detect) {
    static char names[64][9];
    static int n = -1;
    static char out[9];
    size_t idx = (size_t)*vat_ptr;
    if (n < 0) {                                           /* sorted list of matching AppVars */
        DIR *d = opendir(dir());
        struct dirent *e;
        n = 0;
        while (d && (e = readdir(d)) != NULL && n < 64) {
            size_t len = strlen(e->d_name), sz;
            unsigned char *data;
            char name[9];
            if (len < 5 || len > 12 || strcmp(e->d_name + len - 4, ".8xv") != 0) continue;
            memcpy(name, e->d_name, len - 4);
            name[len - 4] = 0;
            if (!(data = read_8xv(name, &sz))) continue;
            if (sz >= strlen(detect) && memcmp(data, detect, strlen(detect)) == 0) strcpy(names[n++], name);
            free(data);
        }
        if (d) closedir(d);
        qsort(names, (size_t)n, sizeof names[0], (int (*)(const void *, const void *))strcmp);
    }
    if ((int)idx >= n) return NULL;
    strcpy(out, names[idx]);
    *vat_ptr = (void *)(idx + 1);
    return out;
}

/* ─── graphx ─────────────────────────────────────────────────────────────── */
static uint8_t fb[240][320], color;
static uint16_t pal[256];

void gfx_Begin(void) { memset(fb, 0, sizeof fb); }
void gfx_SetPalette(const void *palette, uint24_t size, uint8_t offset) {
    memcpy(pal + offset, palette, size);
}
void gfx_FillScreen(uint8_t index) { memset(fb, index, sizeof fb); }
uint8_t gfx_SetColor(uint8_t index) { uint8_t old = color; color = index; return old; }
void gfx_SetPixel(uint24_t x, uint8_t y) { if (x < 320 && y < 240) fb[y][x] = color; }
uint8_t gfx_GetPixel(uint24_t x, uint8_t y) { return x < 320 && y < 240 ? fb[y][x] : 0; }
void gfx_FillRectangle_NoClip(uint24_t x, uint8_t y, uint24_t w, uint8_t h) {
    uint24_t i, j;
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++) gfx_SetPixel(x + i, (uint8_t)(y + j));
}
void gfx_SetDrawBuffer(void) { }
void gfx_BlitBuffer(void) {
    const char *path = getenv("TILCE_PPM");
    FILE *f;
    int x, y;
    if (!path || !(f = fopen(path, "wb"))) return;
    fprintf(f, "P6\n320 240\n255\n");
    for (y = 0; y < 240; y++)
        for (x = 0; x < 320; x++) {
            uint16_t c = pal[fb[y][x]];                     /* 1555 -> 888 */
            fputc(((c >> 10) & 31) << 3, f);
            fputc(((c >> 5) & 31) << 3, f);
            fputc((c & 31) << 3, f);
        }
    fclose(f);
}
void gfx_End(void) { }
