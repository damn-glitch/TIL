/*
 * TIL VM on a Z80 (the CPU family of the TI-83 / TI-84 graphing calculators).
 *
 * This harness is used to verify the VM on a simulated Z80 (ucsim `sz80`,
 * shipped with SDCC): program output is written into a RAM buffer that the
 * test script reads back after the CPU halts. On a real calculator, replace
 * z80_write() with the OS text routine (e.g. _PutC / _PutS on the TI-84).
 */
#include "tilvm.h"
#include "program.h"   /* generated: til_program[] (the .tbc bytes, kept in ROM) */

#define OUT_MAX 0x1800
volatile __at(0x9FFC) unsigned int out_len;
volatile __at(0x9FFE) unsigned char exit_code;
volatile __at(0xA000) char out_buf[OUT_MAX];

static void z80_write(void *ctx, const char *data, size_t len) {
    (void)ctx;
    while (len--) {
        if (out_len < OUT_MAX) out_buf[out_len++] = *data;
        data++;
    }
}

static tilvm vm;

int main(void) {
    int rc;
    out_len = 0;
    exit_code = 0xEE;
    rc = tilvm_load(&vm, til_program, (uint32_t)sizeof(til_program));
    if (rc != 0) {
        z80_write(0, "TIL VM: load error\n", 19);
        exit_code = 0xFD;
        return 1;
    }
    vm.write = z80_write;
    vm.write_err = z80_write;
    exit_code = (unsigned char)tilvm_run(&vm);
    return 0;
}
