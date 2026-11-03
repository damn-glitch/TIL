# TIL VM - build the portable bytecode VM
#
#   make            host VM (64-bit ints, double floats, 256 KB heap)
#   make small      calculator-class profile: 32-bit ints, float, 16 KB heap
#   make tiny       microcontroller profile: 16-bit ints, no floats, 2 KB heap
#   make check      sanitizer build (ASan + UBSan) with a 4 KB heap (GC stress)
#   make z80        cross-compile for the Z80 (TI-83/84 CPU) with SDCC
#   make cortex-m   cross-compile for ARM Cortex-M (NumWorks / STM32 class)
#   make avr        cross-compile for AVR (Arduino Mega class)
#   make sizes      print code size per target

CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pedantic
SRC      = tilvm.c main.c

all: tilvm

tilvm: tilvm.c tilvm.h main.c
	$(CC) $(CFLAGS) -o $@ $(SRC) -lm

small: tilvm.c tilvm.h main.c
	$(CC) $(CFLAGS) -DTILVM_INT_BITS=32 -DTILVM_FLOAT=1 -DTILVM_HEAP_SIZE=16384 \
	  -DTILVM_STACK_SIZE=512 -DTILVM_MAX_FRAMES=64 -DTILVM_MAX_CONSTS=512 -DTILVM_MAX_TYPES=64 \
	  -o tilvm-small $(SRC) -lm

tiny: tilvm.c tilvm.h main.c
	$(CC) $(CFLAGS) -DTILVM_INT_BITS=16 -DTILVM_FLOAT=0 -DTILVM_HEAP_SIZE=2048 \
	  -DTILVM_STACK_SIZE=128 -DTILVM_MAX_FRAMES=24 -DTILVM_MAX_CONSTS=128 -DTILVM_MAX_TYPES=16 \
	  -DTILVM_MAX_TRY=4 -o tilvm-tiny $(SRC) -lm

check: tilvm.c tilvm.h main.c
	$(CC) -std=c99 -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
	  -DTILVM_HEAP_SIZE=4096 -o tilvm-check $(SRC) -lm

# ---- cross compilation (see ports/ for device integration) ----------------
z80:
	$(MAKE) -C ports/z80

cortex-m:
	$(MAKE) -C ports/cortex-m

avr:
	$(MAKE) -C ports/avr

sizes: z80 cortex-m avr
	@cat ports/*/size.txt

clean:
	rm -f tilvm tilvm-small tilvm-tiny tilvm-check *.o
	-$(MAKE) -C ports/z80 clean
	-$(MAKE) -C ports/cortex-m clean
	-$(MAKE) -C ports/avr clean

.PHONY: all small tiny check z80 cortex-m avr sizes clean
