# TIL VM for the Z80 (TI-83/84 class) with SDCC.  Usage:
#   make            build build/tilvm.rel (the VM; SDCC takes ~5 minutes)
#   python3 run_sim.py program.til     compile + run a TIL program on a simulated Z80
SDCC    ?= sdcc
DEFS     = -DTILVM_LEAN -DTILVM_INT_BITS=32 -DTILVM_FLOAT=0 -DTILVM_NO_STDIO -DTILVM_NO_TIME \
           -DTILVM_HEAP_SIZE=4096 -DTILVM_STACK_SIZE=192 -DTILVM_MAX_FRAMES=32 \
           -DTILVM_MAX_CONSTS=192 -DTILVM_MAX_TYPES=24 -DTILVM_MAX_TRY=8 -DTILVM_MARK_STACK=16
CFLAGS   = -mz80 --std-c99 --opt-code-size $(DEFS) -I../..

all: build/tilvm.rel

build/tilvm.rel: ../../tilvm.c ../../tilvm.h
	mkdir -p build
	$(SDCC) $(CFLAGS) -c ../../tilvm.c -o build/tilvm.rel

clean:
	rm -rf build

.PHONY: all clean
