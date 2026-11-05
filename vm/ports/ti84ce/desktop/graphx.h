/* Desktop stand-in for the CE toolchain header (tests only, see ce_shim.c). */
#ifndef CE_SHIM_GRAPHX_H
#define CE_SHIM_GRAPHX_H
#include <stdint.h>
typedef uint32_t uint24_t;   /* <stdint.h> on the calculator */
#define gfx_RGBTo1555(r,g,b) \
((uint16_t)(((((r) & 255) >> 3) << 10) | ((((g) & 255) >> 3) << 5) | (((b) & 255) >> 3)))
void gfx_Begin(void);
void gfx_End(void);
void gfx_SetPalette(const void *palette, uint24_t size, uint8_t offset);
void gfx_FillScreen(uint8_t index);
uint8_t gfx_SetColor(uint8_t index);
void gfx_SetPixel(uint24_t x, uint8_t y);
uint8_t gfx_GetPixel(uint24_t x, uint8_t y);
void gfx_FillRectangle_NoClip(uint24_t x, uint8_t y, uint24_t width, uint8_t height);
void gfx_SetDrawBuffer(void);
void gfx_BlitBuffer(void);
#endif
