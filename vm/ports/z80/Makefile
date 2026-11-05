# TIL VM for the Z80 (TI-83 Plus / TI-84 Plus class) with SDCC: 16-bit lean build,
# without effect handlers, the screen/keys natives and energy saving (TILVM_NO_EFFECTS,
# TILVM_NO_GFX, TILVM_NO_ENERGY): with them the code would not fit below 0xC000.
# Programs using them get an explicit "not available in this VM build" error
# (`perform E()` still calls a function E; #[energy: ...] programs run at full speed).
#
#   make                              build/tilvm.ihx: VM + test harness (~2 minutes)
#   python3 run_sim.py program.til    compile a TIL program and run it on the simulated Z80
#   make size                         code / data sizes and the interpreter's stack frame
#
# Memory map: see main_z80.c - the VM code and constants must end below 0xC000.
SDCC    ?= sdcc
PROFILE  = -DTILVM_LEAN -DTILVM_ADDR16 -DTILVM_INT_BITS=16 -DTILVM_FLOAT=0 -DTILVM_NO_STDIO \
           -DTILVM_NO_TIME -DTILVM_HEAP_SIZE=4096 -DTILVM_STACK_SIZE=192 -DTILVM_MAX_FRAMES=32 \
           -DTILVM_MAX_CONSTS=192 -DTILVM_MAX_TYPES=24 -DTILVM_MAX_TRY=8 -DTILVM_MARK_STACK=16 \
           -DTILVM_NO_GFX -DTILVM_NO_EFFECTS -DTILVM_NO_ENERGY
CFLAGS   = -mz80 --std-c99 --opt-code-speed --max-allocs-per-node 10000 $(PROFILE) -I../..
LDFLAGS  = -mz80 --code-loc 0x0110 --data-loc 0xC000

all: build/tilvm.ihx

build/tilvm.rel: ../../tilvm.c ../../tilvm.h
	mkdir -p build
	$(SDCC) $(CFLAGS) -c ../../tilvm.c -o build/tilvm.rel

build/main_z80.rel: main_z80.c ../../tilvm.h
	mkdir -p build
	$(SDCC) $(CFLAGS) -c main_z80.c -o build/main_z80.rel

build/tilvm.ihx: build/main_z80.rel build/tilvm.rel
	$(SDCC) $(LDFLAGS) build/main_z80.rel build/tilvm.rel -o build/tilvm.ihx

size: build/tilvm.ihx
	python3 run_sim.py --size

clean:
	rm -rf build

.PHONY: all size clean
