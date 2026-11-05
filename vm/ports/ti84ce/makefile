# TIL VM for the TI-84 Plus CE (eZ80) with the CE C toolchain (CEdev,
# https://github.com/CE-Programming/toolchain; written against v15.0).
#
#   make                                     -> bin/TILVM.8xp
#   python3 tbc2appvar.py prog.tbc PROG      -> PROG.8xv (TIL bytecode as an AppVar)
#
# Send TILVM.8xp and the AppVar(s) to the calculator (TI Connect CE / tilp) and
# run TILVM from the [prgm] menu. See README.md.

NAME = TILVM
DESCRIPTION = "TIL bytecode VM"
COMPRESSED = NO
ARCHIVED = NO

# 32-bit ints, single-precision floats, 32 KB heap; programs (AppVars) and the
# heap are below 64 KB, so the VM uses 16-bit offsets (TILVM_ADDR16); the screen
# is 320 x 240, so screen_init() accepts at most that (larger is a runtime error)
TIL_PROFILE = -DTILVM_INT_BITS=32 -DTILVM_FLOAT=1 -DTILVM_NO_STDIO -DTILVM_ADDR16 \
              -DTILVM_HEAP_SIZE=32768 -DTILVM_STACK_SIZE=1024 -DTILVM_MAX_FRAMES=128 \
              -DTILVM_MAX_CONSTS=1024 -DTILVM_MAX_TYPES=128 -DTILVM_MAX_TRY=16 -DTILVM_MARK_STACK=32 \
              -DTILVM_GFX_MAX_W=320 -DTILVM_GFX_MAX_H=240

CFLAGS = -Wall -Wextra -Oz $(TIL_PROFILE) -I../..
CXXFLAGS = -Wall -Wextra -Oz

include $(shell cedev-config --makefile)
