/*
 * tilvm - command-line runner for TIL bytecode (.tbc)
 *
 *   tilvm program.tbc [args...]           run a program (args() returns the args)
 *   tilvm --max-steps N program.tbc       sandbox: stop after N loop iterations/calls
 *   tilvm --snapshot-after N FILE p.tbc   run N steps, save the paused program to FILE
 *   tilvm --resume FILE program.tbc       continue a saved program
 *   tilvm --stats program.tbc             print memory / GC statistics to stderr
 *
 * Produce bytecode with:  til build --target vm program.til
 */
#include "tilvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static tilvm vm;   /* the whole VM state: one static block, no other allocations */

static unsigned char *read_file(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    unsigned char *buf;
    long n;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (unsigned char *)malloc(n > 0 ? (size_t)n : 1);
    if (buf && n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); buf = NULL; }
    fclose(f);
    *size = n;
    return buf;
}

static void usage(void) {
    fprintf(stderr, "usage: tilvm [--max-steps N] [--snapshot-after N FILE] [--resume FILE] [--stats] "
                    "program.tbc [args...]\n");
}

int main(int argc, char **argv) {
    const char *path = NULL, *snap_out = NULL, *resume = NULL;
    unsigned long snap_after = 0, max_steps = 0;
    int stats = 0, i, rc, prog_argc = 0;
    char **prog_argv = NULL;
    long size = 0;
    unsigned char *code;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-steps") && i + 1 < argc) max_steps = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--snapshot-after") && i + 2 < argc) {
            snap_after = strtoul(argv[++i], NULL, 10);
            snap_out = argv[++i];
        } else if (!strcmp(argv[i], "--resume") && i + 1 < argc) resume = argv[++i];
        else if (!strcmp(argv[i], "--stats")) stats = 1;
        else if (!strcmp(argv[i], "--version")) { printf("tilvm %s (bytecode v%d, int%d, heap %ld bytes)\n",
                                                        TILVM_VERSION, TILVM_BYTECODE_VERSION, TILVM_INT_BITS,
                                                        (long)TILVM_HEAP_SIZE); return 0; }
        else if (argv[i][0] == '-') { usage(); return 2; }
        else {
            path = argv[i];                     /* everything after the program is for it */
            prog_argv = argv + i + 1;
            prog_argc = argc - i - 1;
            break;
        }
    }
    if (!path) { usage(); return 2; }
    code = read_file(path, &size);
    if (!code) { fprintf(stderr, "tilvm: cannot read %s\n", path); return 2; }
    rc = tilvm_load(&vm, code, (uint32_t)size);
    if (rc != 0) {
        fprintf(stderr, "tilvm: cannot load %s (error %d%s)\n", path, rc,
                rc == -3 ? ": program uses floats, VM built without them" :
                rc == -6 ? ": integer constant too large for this VM" : "");
        return 2;
    }
    vm.max_steps = (uint32_t)max_steps;
    vm.host.argc = prog_argc;
    vm.host.argv = (const char *const *)prog_argv;

    if (resume) {
        long ssize = 0;
        unsigned char *snap = read_file(resume, &ssize);
        if (!snap || tilvm_restore(&vm, snap, (uint32_t)ssize) != 0) {
            fprintf(stderr, "tilvm: cannot restore snapshot %s (different program or VM build?)\n", resume);
            return 2;
        }
        free(snap);
    }

    if (snap_out) {
        rc = tilvm_run_steps(&vm, (uint32_t)snap_after);
        fflush(stdout);
        if (rc == TILVM_YIELD) {
            uint32_t n = tilvm_snapshot_size(&vm);
            unsigned char *buf = (unsigned char *)malloc(n);
            FILE *f = fopen(snap_out, "wb");
            if (!buf || !f || tilvm_snapshot(&vm, buf, n) != n || fwrite(buf, 1, n, f) != n) {
                fprintf(stderr, "tilvm: cannot write snapshot\n");
                return 2;
            }
            fclose(f);
            free(buf);
            fprintf(stderr, "[tilvm] paused after %lu steps; snapshot saved to %s (%lu bytes)\n",
                    snap_after, snap_out, (unsigned long)n);
            return 3;
        }
        rc = rc == TILVM_EXIT ? vm.st.exit_code : rc == TILVM_DONE ? 0 : 1;
    } else {
        rc = tilvm_run(&vm);
    }
    fflush(stdout);
    if (stats) {
        fprintf(stderr, "[tilvm] steps=%lu gc=%lu heap=%lu/%ld bytes\n",
                (unsigned long)vm.st.steps, (unsigned long)vm.st.gc_count,
                (unsigned long)tilvm_heap_used(&vm), (long)TILVM_HEAP_SIZE);
    }
    free(code);
    return rc;
}
