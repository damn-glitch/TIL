/* TIL VM on ARM Cortex-M. Output goes through semihosting here; on a real
 * device route vm.write to the screen / UART (see ../README.md). */
#include "tilvm.h"
#include "program.h"

void sh_write(const char *data, unsigned len);

static void arm_write(void *ctx, const char *data, size_t len) { (void)ctx; sh_write(data, (unsigned)len); }

static tilvm vm;

int main(void) {
    if (tilvm_load(&vm, til_program, (uint32_t)sizeof(til_program)) != 0) {
        sh_write("TIL VM: load error\n", 19);
        return 2;
    }
    vm.write = arm_write;
    vm.write_err = arm_write;
    return tilvm_run(&vm);
}
