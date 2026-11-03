/* Desktop stand-in for the CE toolchain header (tests only, see ce_shim.c). */
#ifndef CE_SHIM_FILEIOC_H
#define CE_SHIM_FILEIOC_H
#include <stddef.h>
#include <stdint.h>
uint8_t ti_Open(const char *name, const char *mode);
int ti_Close(uint8_t handle);
size_t ti_Read(void *data, size_t size, size_t count, uint8_t handle);
size_t ti_Write(const void *data, size_t size, size_t count, uint8_t handle);
uint16_t ti_GetSize(uint8_t handle);
char *ti_Detect(void **vat_ptr, const char *detect_string);
#endif
