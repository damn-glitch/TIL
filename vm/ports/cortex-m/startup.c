/* Minimal Cortex-M startup: vector table, .data/.bss init, semihosting I/O.
 * Used to run the TIL VM on QEMU's lm3s6965evb board (Cortex-M3, 64 KB RAM),
 * a stand-in for ARM calculators (NumWorks) and microcontrollers (STM32). */
#include <stdint.h>
#include <string.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
int main(void);

static int semihost(int op, void *arg) {
    register int r0 __asm__("r0") = op;
    register void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

void sh_write(const char *data, unsigned len) {
    char chunk[65];
    while (len) {
        unsigned n = len > 64 ? 64 : len, i, k = 0;
        for (i = 0; i < n; i++) if (data[i]) chunk[k++] = data[i];
        chunk[k] = 0;
        semihost(0x04, chunk);          /* SYS_WRITE0 */
        data += n;
        len -= n;
    }
}

void sh_exit(int code) {
    uint32_t args[2] = {0x20026u, (uint32_t)code};   /* ADP_Stopped_ApplicationExit */
    semihost(0x20, args);                              /* SYS_EXIT_EXTENDED */
    for (;;) { }
}

void Reset_Handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss; ) *dst++ = 0;
    sh_exit(main());
}

void Default_Handler(void) { sh_write("fault\n", 6); sh_exit(99); }

__attribute__((section(".isr_vector"), used))
void (*const vectors[16])(void) = {
    (void (*)(void))&_estack, Reset_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler, Default_Handler, 0, 0, 0, 0,
    Default_Handler, Default_Handler, 0, Default_Handler, Default_Handler,
};
