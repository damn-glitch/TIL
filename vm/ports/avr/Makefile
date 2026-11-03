# TIL VM for 8-bit AVR (Arduino Mega / ATmega1284P class).
#   make PROGRAM=../../../examples/03_functions.til     build tilvm.elf
#   make run                                            run on simavr
CC      = avr-gcc
MCU    ?= atmega1284p
PROGRAM ?= ../../../examples/01_hello.til
DEFS    = -DTILVM_INT_BITS=32 -DTILVM_FLOAT=1 -DTILVM_NO_STDIO -DTILVM_NO_TIME \
          -DTILVM_HEAP_SIZE=5120 -DTILVM_STACK_SIZE=160 -DTILVM_MAX_FRAMES=24 \
          -DTILVM_MAX_CONSTS=160 -DTILVM_MAX_TYPES=24 -DTILVM_MAX_TRY=6 -DTILVM_MARK_STACK=16 \
          -DTILVM_ROM=__flash
CFLAGS  = -mmcu=$(MCU) -Os -std=gnu99 -Wall -ffunction-sections -fdata-sections $(DEFS) -I../.. -Ibuild

all: build/tilvm.elf

build/program.h: $(PROGRAM)
	mkdir -p build
	python3 -c "import sys; sys.path.insert(0,'../../../src'); import til; \
m=til.compile_to_bytecode(open('$(PROGRAM)',encoding='utf-8').read(),'$(PROGRAM)'); \
open('build/program.h','w').write(til.bytes_to_c_array('til_program', m.serialize(strip=True)))"

build/tilvm.elf: build/program.h ../../tilvm.c ../../tilvm.h main_avr.c
	$(CC) $(CFLAGS) -Wl,--gc-sections -o $@ main_avr.c ../../tilvm.c -lm
	avr-size $@ | tee build/size.txt

run: build/tilvm.elf
	simavr -m $(MCU) -f 16000000 build/tilvm.elf

clean:
	rm -rf build

.PHONY: all run clean
