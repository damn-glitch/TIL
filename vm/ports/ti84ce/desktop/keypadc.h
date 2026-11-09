/* Desktop stand-in for the CE toolchain header (tests only, see ce_shim.c). */
#ifndef CE_SHIM_KEYPADC_H
#define CE_SHIM_KEYPADC_H
extern int ce_shim_on_key;
#define kb_On (ce_shim_on_key)
#define kb_EnableOnLatch() ((void)0)
#define kb_DisableOnLatch() ((void)0)
#define kb_ClearOnLatch() ((void)(ce_shim_on_key = 0))
#endif
