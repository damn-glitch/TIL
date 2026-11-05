/* Desktop stand-in for the CE toolchain header (tests only, see ce_shim.c). */
#ifndef CE_SHIM_TI_GETCSC_H
#define CE_SHIM_TI_GETCSC_H
#include <stdint.h>
#define sk_Down  0x01
#define sk_Left  0x02
#define sk_Right 0x03
#define sk_Up    0x04
#define sk_Enter 0x09
#define sk_Clear 0x0F
uint8_t os_GetCSC(void);
#endif
