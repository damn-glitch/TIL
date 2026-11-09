/* Desktop stand-in for the CE toolchain header (tests only, see ce_shim.c). */
#ifndef CE_SHIM_TI_SCREEN_H
#define CE_SHIM_TI_SCREEN_H
void os_ClrHome(void);
void outchar(char c);        /* <stdio.h> on the calculator */
#endif
