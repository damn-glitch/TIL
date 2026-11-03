# TIL VM on the Z80 — status

**Summary:** the VM compiles for the Z80 with SDCC 4.2, but a complete build is
larger than the 64 KB address space of classic Z80 calculators (TI-83 Plus /
TI-84 Plus), so it does not run there yet. ARM Cortex-M and 8-bit AVR builds
run all examples correctly (see `../../README.md`).

## Measurements (SDCC 4.2.0, `-mz80 --opt-code-size`)

| Build | Code size | Fits below 0xC000 (48 KB)? |
|---|---|---|
| 32-bit ints, all builtins | 76 KB | no |
| 32-bit ints, `TILVM_LEAN` | 70.7 KB | no |
| 16-bit ints, `TILVM_LEAN` | — | SDCC internal compiler error* |

For comparison, the same source is **24.5 KB** (19.4 KB lean) for ARM
Cortex-M0 and 53.8 KB for AVR: SDCC expands 32-bit arithmetic and the 5-byte
value copies inline on the Z80.

\* `error 9: FATAL Compiler Internal Error in file 'gen.c' line number '7884' :
Unimplemented` while generating `tilvm_run_steps` with `TILVM_INT_BITS=16`.
It does not reproduce with isolated snippets of the same operations.

## Ways forward

1. **TI-84 Plus CE** (eZ80, 24-bit addresses, 154 KB RAM): no 64 KB limit; use the
   CE C/C++ toolchain (LLVM). Recommended TI target.
2. **Banked Flash application** on the TI-83+/84+: place the builtins
   (`call_native`, `str_method`, `array_method`, ...) on separate 16 KB pages.
3. **16-bit build** once the SDCC bug is fixed or worked around (expected to
   roughly halve the code size), or with another Z80 compiler (z88dk).

## Reproduce

```bash
make                                   # build/tilvm.rel (~5 minutes)
python3 run_sim.py ../../../examples/03_functions.til
# -> "code ends at 0x11644: does not fit below the VM data at 0xC000"
```

`run_sim.py` builds the program, checks the memory map and runs it on the
ucsim Z80 simulator (`sz80`), reading the program output back from RAM.
