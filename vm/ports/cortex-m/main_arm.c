/* TIL VM on ARM Cortex-M. Output goes through semihosting here; on a real
 * device route vm.write to the screen / UART (see ../README.md). */
#include "tilvm.h"
#include "program.h"

void sh_write(const char *data, unsigned len);

static void arm_write(void *ctx, const char *data, size_t len) { (void)ctx; sh_write(data, (unsigned)len); }

static tilvm vm;

#ifdef TIL_SNAPSHOT
/* Continue a program paused on another VM (make SNAPSHOT=state.tilp): the
 * snapshot is read through a callback - from flash here, from a file or a
 * serial link on a real device (docs/TIL_SNAPSHOTS.md). */
struct snap_src { const unsigned char *p; unsigned long left; };

static int snap_get(void *ctx, uint8_t *buf, uint32_t n) {
    struct snap_src *s = (struct snap_src *)ctx;
    uint32_t i;
    if (n > s->left) return -1;
    for (i = 0; i < n; i++) buf[i] = s->p[i];
    s->p += n;
    s->left -= n;
    return 0;
}
#endif

int main(void) {
    if (tilvm_load(&vm, til_program, (uint32_t)sizeof(til_program)) != 0) {
        sh_write("TIL VM: load error\n", 19);
        return 2;
    }
    vm.write = arm_write;
    vm.write_err = arm_write;
#ifdef TIL_SNAPSHOT
    {
        struct snap_src src;
        src.p = til_snapshot;
        src.left = til_snapshot_size;
        if (tilvm_load_portable(&vm, snap_get, &src) != 0 || src.left) {
            sh_write("TIL VM: cannot resume the snapshot\n", 35);
            return 2;
        }
    }
#endif
    return tilvm_run(&vm);
}
