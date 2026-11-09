/* TIL VM on an 8-bit AVR (ATmega1284P: 16 KB RAM, 128 KB flash) - the class of
 * chip in Arduino boards. Output goes to UART0 (115200 baud on hardware; the
 * simavr simulator prints it to the console). The bytecode is copied to RAM by
 * the C runtime here; a production port would read it from flash. */
#include <avr/io.h>
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include "tilvm.h"
#include "program.h"

static void uart_put(char c) {
    while (!(UCSR0A & (1 << UDRE0))) { }
    UDR0 = c;
}

static void avr_write(void *ctx, const char *data, size_t len) {
    (void)ctx;
    while (len--) uart_put(*data++);
}

static tilvm vm;

int main(void) {
    UBRR0 = 8;                       /* 115200 baud @ 16 MHz */
    UCSR0B = (1 << TXEN0);
    if (tilvm_load(&vm, til_program, (uint32_t)sizeof(til_program)) != 0) {
        avr_write(0, "TIL VM: load error\n", 19);
    } else {
        vm.write = avr_write;
        vm.write_err = avr_write;
        tilvm_run(&vm);
    }
    avr_write(0, "\n<<TIL-EOT>>\n", 14);   /* end marker for the test harness */
    cli();
    sleep_cpu();                     /* simavr stops here */
    return 0;
}
