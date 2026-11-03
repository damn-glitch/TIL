# TIL VM for ARM Cortex-M (NumWorks / STM32 / RP2040 class).
#   make PROGRAM=../../../examples/03_functions.til   build tilvm.elf with that program
#   make run                                           run it on QEMU (lm3s6965evb)
CC      = arm-none-eabi-gcc
SIZE    = arm-none-eabi-size
PROGRAM ?= ../../../examples/01_hello.til
DEFS    = -DTILVM_INT_BITS=32 -DTILVM_FLOAT=1 -DTILVM_NO_STDIO -DTILVM_NO_TIME \
          -DTILVM_HEAP_SIZE=16384 -DTILVM_STACK_SIZE=512 -DTILVM_MAX_FRAMES=64 \
          -DTILVM_MAX_CONSTS=512 -DTILVM_MAX_TYPES=64 -DTILVM_MAX_TRY=8
CFLAGS  = -mcpu=cortex-m3 -mthumb -Os -std=c99 -Wall -ffunction-sections -fdata-sections $(DEFS) \
          -I../.. -Ibuild
LDFLAGS = -T lm3s6965.ld -nostartfiles --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections

all: build/tilvm.elf

build/program.h: $(PROGRAM)
	mkdir -p build
	python3 -c "import sys; sys.path.insert(0,'../../../src'); import til; \
m=til.compile_to_bytecode(open('$(PROGRAM)',encoding='utf-8').read(),'$(PROGRAM)'); \
open('build/program.h','w').write(til.bytes_to_c_array('til_program', m.serialize(strip=True)))"

build/tilvm.elf: build/program.h ../../tilvm.c ../../tilvm.h main_arm.c startup.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ main_arm.c startup.c ../../tilvm.c -lm
	$(SIZE) $@ | tee build/size.txt

run: build/tilvm.elf
	qemu-system-arm -M lm3s6965evb -nographic -semihosting-config enable=on,target=native \
	  -kernel build/tilvm.elf

clean:
	rm -rf build

.PHONY: all run clean
