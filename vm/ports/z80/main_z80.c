/*
 * TIL VM on a Z80 (the CPU family of the TI-83 Plus / TI-84 Plus calculators).
 *
 * Test harness for the ucsim Z80 simulator (`sz80`, shipped with SDCC): the
 * bytecode is loaded into RAM like a calculator program (run_sim.py adds it to
 * the image at 0xD800), output goes to a RAM buffer that run_sim.py reads back
 * after the CPU halts. On a real calculator, z80_write() becomes the OS text
 * routine (_PutC / _PutS on the TI-84 Plus) and the bytecode comes from an
 * AppVar - see RESULTS.md.
 *
 * Memory map (64 KB, flat):
 *   0x0000-0x01FF  reset vectors, startup (crt0)
 *   0x0200-0xBFFF  VM code and constants            (must end below 0xC000)
 *   0xC000-0xD7FF  VM state: `static tilvm vm` (heap, value stack, frames)
 *   0xD800-0xEFEF  program: u16 length + .tbc bytes (written by run_sim.py)
 *   0xEFF0-0xEFFF  results (struct results below)
 *   0xF000-0xF7FF  program output
 *   0xF800-0xFFFF  C stack (grows down from 0xFFFF)
 */
#include "tilvm.h"

#define PROG_ADDR 0xD800
#define RES_ADDR  0xEFF0
#define OUT_ADDR  0xF000
#define OUT_MAX   0x0800
#define STACK_LO  0xF800
#define STACK_PAINT_END 0xFFC0   /* main() and the startup code use less than 64 bytes */

struct results {               /* read back by run_sim.py (little endian) */
    uint16_t out_len;          /* 0xEFF0 */
    uint8_t exit_code;         /* 0xEFF2: program exit code, 0xEE = did not finish */
    uint8_t load_error;        /* 0xEFF3: -tilvm_load() */
    uint32_t steps;            /* 0xEFF4: VM steps (loop iterations + calls) */
    uint16_t heap_used;        /* 0xEFF8 */
    uint16_t gc_count;         /* 0xEFFA */
    uint16_t stack_low;        /* 0xEFFC: lowest address the C stack reached */
};

#define RES ((volatile struct results *)RES_ADDR)

static void z80_write(void *ctx, const char *data, size_t len) {
    volatile char *out = (volatile char *)OUT_ADDR;
    uint16_t n = RES->out_len;
    (void)ctx;
    while (len--) {
        if (n < OUT_MAX) out[n++] = *data;
        data++;
    }
    RES->out_len = n;
}

static tilvm vm;

int main(void) {
    const uint8_t *prog = (const uint8_t *)PROG_ADDR;
    unsigned char *p;
    int rc;
    /* paint the C stack below main's frame to measure its peak use afterwards */
    for (p = (unsigned char *)STACK_LO; p < (unsigned char *)STACK_PAINT_END; p++) *p = 0xA5;
    RES->out_len = 0;
    RES->exit_code = 0xEE;
    rc = tilvm_load(&vm, prog + 2, (uint32_t)(prog[0] | (prog[1] << 8)));
    RES->load_error = (uint8_t)(0 - rc);
    if (rc != 0) {
        z80_write(0, "TIL VM: cannot load the program\n", 32);
        RES->exit_code = 0xFD;
        RES->stack_low = STACK_PAINT_END;
        return 1;
    }
    vm.write = z80_write;
    vm.write_err = z80_write;
    rc = tilvm_run(&vm);
    RES->exit_code = (uint8_t)rc;
    RES->steps = vm.st.steps;
    RES->heap_used = (uint16_t)tilvm_heap_used(&vm);
    RES->gc_count = (uint16_t)vm.st.gc_count;
    for (p = (unsigned char *)STACK_LO; *p == 0xA5; p++) { }
    RES->stack_low = (uint16_t)p;
    return 0;
}
