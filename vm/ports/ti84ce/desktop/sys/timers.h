/* Desktop stand-in for the CE toolchain header (tests only, see ce_shim.c). */
#ifndef CE_SHIM_SYS_TIMERS_H
#define CE_SHIM_SYS_TIMERS_H
#include <stdint.h>
void delay(uint16_t msec);
#endif
