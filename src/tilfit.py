#!/usr/bin/env python3
"""
til fit - will a TIL program fit a device?  (TIL 3.1, docs/TIL_FIT.md)

    til fit prog.til --device NAME [--run] [--json] [--strip] [--functions]
                     [--max-steps N] [--timeout SEC] [--show-output] [-- ARGS...]
    til fit --list-devices [--json]
    til build --target vm --device NAME prog.til      (the same check after building)

Static analysis of the compiled bytecode against a device profile: bytecode
size, constant and type tables, integer width, floats, builtins that lean VM
builds leave out, the operand-stack depth of every function (abstract
interpretation of the bytecode), worst-case value-stack slots, call frames and
recovery points over the call graph (recursion is reported with the number of
levels that fit), stack use that grows with data, allocations inside loops,
and the RAM of the C VM itself.  --run builds the C VM with the device's limits
(plus TILVM_STATS) and reports the measured high-water marks.

Exit status: 0 fits, 1 does not fit, 2 usage error / the program cannot be
analysed (compile error, unreadable or malformed file).
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import til

# Struct layout of the C VM per target ABI: (size, alignment) of each C type.
# Used by vm_struct_layout() below (sizeof(tilvm) for a device profile).
ABIS: Dict[str, Dict[str, Tuple[int, int]]] = {
    # x86-64 / ARM64 desktops
    'lp64': dict(u8=(1, 1), u16=(2, 2), u32=(4, 4), u64=(8, 8), float=(4, 4), double=(8, 8),
                 ptr=(8, 8), int=(4, 4)),
    # 32-bit ARM (AAPCS), Xtensa, RISC-V; SH4 is modelled with it (8-byte doubles: an upper bound)
    'ilp32': dict(u8=(1, 1), u16=(2, 2), u32=(4, 4), u64=(8, 8), float=(4, 4), double=(8, 8),
                  ptr=(4, 4), int=(4, 4)),
    # 8-bit AVR (avr-gcc: no alignment, 16-bit int and pointers, double = float)
    'avr': dict(u8=(1, 1), u16=(2, 1), u32=(4, 1), u64=(8, 1), float=(4, 1), double=(4, 1),
                ptr=(2, 1), int=(2, 1)),
    # Z80 (SDCC): like AVR
    'z80': dict(u8=(1, 1), u16=(2, 1), u32=(4, 1), u64=(8, 1), float=(4, 1), double=(4, 1),
                ptr=(2, 1), int=(2, 1)),
    # eZ80 (TI-84 Plus CE toolchain): 24-bit int and pointers, no alignment, double = float
    'ez80': dict(u8=(1, 1), u16=(2, 1), u32=(4, 1), u64=(8, 1), float=(4, 1), double=(4, 1),
                 ptr=(3, 1), int=(3, 1)),
}
PACKED_ABIS = ('avr', 'z80', 'ez80')     # emulated on the desktop with -DTILVM_PACK


# ═══════════════════════════════════════════════════════════════════════════════
#                               DEVICE PROFILES
# ═══════════════════════════════════════════════════════════════════════════════
# Every profile is a C VM build configuration (the TILVM_* macros of vm/tilvm.h)
# plus what the device offers around it.  Sources and assumptions for each
# number are in docs/TIL_FIT.md.

KB = 1024
MB = 1024 * 1024

DEVICE_KEYS = {
    # C VM build (vm/tilvm.h)
    'int_bits': "TILVM_INT_BITS: 16, 32 or 64",
    'float': "TILVM_FLOAT: 0 none, 1 float (32-bit), 2 double",
    'heap': "TILVM_HEAP_SIZE: arena bytes",
    'stack': "TILVM_STACK_SIZE: value-stack slots (globals + locals + temporaries)",
    'frames': "TILVM_MAX_FRAMES: call depth",
    'consts': "TILVM_MAX_CONSTS: constants per program",
    'types': "TILVM_MAX_TYPES: struct types per program (3 are built in)",
    'try_depth': "TILVM_MAX_TRY: nested retry/timeout/intent blocks",
    'mark_stack': "TILVM_MARK_STACK: GC mark stack entries",
    'lean': "TILVM_LEAN: builtins of the lean build only",
    'no_stdio': "TILVM_NO_STDIO", 'no_time': "TILVM_NO_TIME", 'no_math': "TILVM_NO_MATH",
    # the device around the VM
    'storage': "bytes available for one program's bytecode (None = no practical limit)",
    'ram': "RAM for the VM: its struct, static data, C stack (and the bytecode when it is "
           "copied to RAM); None = no practical limit",
    'c_stack': "C stack reserved for the VM's own functions inside `ram`",
    'vm_static_ram': "RAM taken by the VM's tables and strings (AVR keeps them in RAM)",
    'bytecode_in_ram': "the port copies the bytecode into RAM (AVR: Harvard architecture)",
    'abi': "struct layout model: lp64, ilp32, avr, z80, ez80",
    'clock': "the port provides a millisecond clock (else timeouts count steps)",
    'screen': "display and keys (screen_*/draw_*/key natives)",
    'files': "a file system hook (read_file / write_file / file_exists)",
    'console': "text input for input()",
    'title': "", 'cpu': "", 'source': "where the numbers come from",
}

_DESKTOP_BUILD = dict(int_bits=64, float=2, heap=256 * KB, stack=8192, frames=1000, consts=4096,
                      types=512, try_depth=32, mark_stack=64, lean=False, no_stdio=False,
                      no_time=False, no_math=False)
_BASE = dict(_DESKTOP_BUILD, storage=None, ram=None, c_stack=0, vm_static_ram=0,
             bytecode_in_ram=False, abi='lp64', clock=True, screen=False, files=False,
             console=False, title='', cpu='', source='')


class Device:
    """A device profile: the C VM build configuration and what the device offers."""

    def __init__(self, name: str, **values):
        unknown = set(values) - set(DEVICE_KEYS)
        if unknown:
            raise ValueError(f"unknown device field(s): {', '.join(sorted(unknown))}")
        merged = dict(_BASE)
        merged.update(values)
        self.name = name
        for key in DEVICE_KEYS:
            setattr(self, key, merged[key])
        self.validate()

    def validate(self):
        def need(cond, what):
            if not cond:
                raise ValueError(f"device '{self.name}': {what}")
        need(self.int_bits in (16, 32, 64), "int_bits must be 16, 32 or 64")
        need(self.float in (0, 1, 2), "float must be 0, 1 or 2")
        need(isinstance(self.heap, int) and 256 <= self.heap <= 0xFFFFFFFF, "heap must be 256..2^32-1 bytes")
        need(isinstance(self.stack, int) and 16 <= self.stack <= 0xFFFF, "stack must be 16..65535 slots")
        need(isinstance(self.frames, int) and 2 <= self.frames <= 0xFFFF, "frames must be 2..65535")
        need(isinstance(self.consts, int) and 1 <= self.consts <= 0xFFFF, "consts must be 1..65535")
        need(isinstance(self.types, int) and 3 <= self.types <= 0xFFFF, "types must be 3..65535")
        need(isinstance(self.try_depth, int) and 1 <= self.try_depth <= 0xFFFF, "try_depth must be 1..65535")
        need(isinstance(self.mark_stack, int) and self.mark_stack >= 1, "mark_stack must be >= 1")
        need(self.abi in ABIS, f"abi must be one of {', '.join(ABIS)}")
        for key in ('storage', 'ram'):
            v = getattr(self, key)
            need(v is None or (isinstance(v, int) and v > 0), f"{key} must be a positive number of bytes or null")
        for key in ('c_stack', 'vm_static_ram'):
            need(isinstance(getattr(self, key), int) and getattr(self, key) >= 0, f"{key} must be >= 0")

    def as_dict(self) -> Dict[str, Any]:
        d = {'name': self.name}
        for key in DEVICE_KEYS:
            d[key] = getattr(self, key)
        return d

    def c_flags(self) -> List[str]:
        """-D flags of the C VM build for this device (TILVM_STATS/TILVM_PACK not included)."""
        flags = [f"-DTILVM_INT_BITS={self.int_bits}", f"-DTILVM_FLOAT={self.float}",
                 f"-DTILVM_HEAP_SIZE={self.heap}", f"-DTILVM_STACK_SIZE={self.stack}",
                 f"-DTILVM_MAX_FRAMES={self.frames}", f"-DTILVM_MAX_CONSTS={self.consts}",
                 f"-DTILVM_MAX_TYPES={self.types}", f"-DTILVM_MAX_TRY={self.try_depth}",
                 f"-DTILVM_MARK_STACK={self.mark_stack}"]
        for key, macro in (('lean', 'TILVM_LEAN'), ('no_stdio', 'TILVM_NO_STDIO'),
                           ('no_time', 'TILVM_NO_TIME'), ('no_math', 'TILVM_NO_MATH')):
            if getattr(self, key):
                flags.append(f"-D{macro}")
        return flags

    def summary(self) -> str:
        fl = {0: "no floats", 1: "float32", 2: "float64"}[self.float]
        return (f"int{self.int_bits}, {fl}, {human(self.heap)} heap, {self.stack} stack slots, "
                f"{self.frames} frames, {'lean' if self.lean else 'full'} VM")


DEVICES: Dict[str, Device] = {}


def _device(name: str, **values):
    DEVICES[name] = Device(name, **values)


_device('desktop', title="Desktop PC, the default C VM build", cpu="x86-64 / ARM64",
        clock=True, screen=True, files=True, console=True,
        source="vm/Makefile `make` (the defaults of vm/tilvm.h)")
_device('ti84ce', title="TI-84 Plus CE graphing calculator",
        cpu="eZ80 @ 48 MHz, 256 KB RAM (154 KB for programs), 3 MB flash",
        int_bits=32, float=1, heap=32 * KB, stack=1024, frames=128, consts=1024, types=128,
        try_depth=16, no_stdio=True, no_time=True, storage=65505, ram=64 * KB, c_stack=4 * KB,
        abi='ez80', clock=True, screen=True, console=True,
        source="vendor specs; vm/README.md (heap <= 32 KB, TILVM_FLOAT=1: the CE toolchain's "
               "double is 32-bit); one AppVar holds at most 65505 bytes")
_device('numworks', title="NumWorks graphing calculator (external app)",
        cpu="ARM Cortex-M7 @ 216 MHz, 256 KB RAM, 8 MB flash",
        int_bits=32, float=2, heap=32 * KB, stack=1024, frames=200, consts=1024, types=128,
        try_depth=16, no_stdio=True, no_time=True, storage=1 * MB, ram=96 * KB, c_stack=8 * KB,
        abi='ilp32', clock=True, screen=True, console=True,
        source="vendor specs; assumed 96 KB of RAM for an external app (heap = the 32 KB "
               "of the built-in Python)")
_device('nspire', title="TI-Nspire CX / CX II (Ndless)",
        cpu="ARM926EJ-S @ 132-396 MHz, 64 MB SDRAM, 100 MB flash",
        int_bits=64, float=2, heap=1 * MB, stack=16384, frames=2000, consts=8192, types=1024,
        try_depth=32, storage=16 * MB, ram=16 * MB, c_stack=64 * KB, abi='ilp32',
        clock=True, screen=True, files=True, console=True,
        source="vendor specs; vm/README.md (the desktop build with newlib works as is)")
_device('casio-cg50', title="Casio fx-CG50 / Graph 90+E (gint add-in)",
        cpu="SH4A @ 59 MHz, 8 MB RAM, 16 MB flash",
        int_bits=32, float=2, heap=64 * KB, stack=2048, frames=256, consts=2048, types=256,
        try_depth=16, no_stdio=True, no_time=True, storage=2 * MB, ram=512 * KB, c_stack=16 * KB,
        abi='ilp32', clock=True, screen=True, console=True,
        source="vendor specs; assumed 512 KB of add-in RAM; SH4 modelled as a 32-bit ABI "
               "with 8-byte doubles (an upper bound)")
_device('cortex-m3-64k', title="ARM Cortex-M3 board, 64 KB RAM / 256 KB flash",
        cpu="Cortex-M3, verified in QEMU (lm3s6965evb)",
        int_bits=32, float=1, heap=16 * KB, stack=512, frames=64, consts=512, types=64,
        try_depth=8, no_stdio=True, no_time=True, storage=256 * KB - 48 * KB, ram=64 * KB,
        c_stack=4 * KB, vm_static_ram=128, abi='ilp32', clock=False,
        source="vm/ports/cortex-m/Makefile (verified profile); VM code 46.9 KB of the "
               "256 KB flash (arm-none-eabi-gcc -Os, newlib-nano)")
_device('atmega1284', title="ATmega1284P, 16 KB RAM / 128 KB flash",
        cpu="8-bit AVR @ 16-20 MHz, verified in simavr",
        int_bits=32, float=1, heap=5120, stack=160, frames=24, consts=160, types=24, try_depth=6,
        mark_stack=16, no_stdio=True, no_time=True, storage=128 * KB - 56 * KB, ram=16 * KB,
        c_stack=1536, vm_static_ram=3130, bytecode_in_ram=True, abi='avr', clock=False,
        source="vm/ports/avr/Makefile (verified profile); avr-gcc -Os: VM code 55.5 KB, "
               "VM strings/tables in RAM 3130 bytes; the port copies the bytecode to RAM")
_device('arduino-mega', title="Arduino Mega 2560 (ATmega2560), 8 KB RAM / 256 KB flash",
        cpu="8-bit AVR @ 16 MHz",
        int_bits=16, float=1, heap=1536, stack=96, frames=16, consts=96, types=16, try_depth=4,
        mark_stack=16, lean=True, no_stdio=True, no_time=True, storage=256 * KB - 44 * KB,
        ram=8 * KB, c_stack=1536, vm_static_ram=2480, bytecode_in_ram=True, abi='avr', clock=False,
        source="lean 16-bit build in the style of `make tiny`, sized for 8 KB; avr-gcc -Os: "
               "VM code 42 KB, VM strings/tables in RAM 2480 bytes; bytecode copied to RAM")
_device('esp32', title="ESP32 (ESP32-WROOM-32), 520 KB SRAM / 4 MB flash",
        cpu="Xtensa LX6 dual-core @ 240 MHz",
        int_bits=32, float=1, heap=64 * KB, stack=2048, frames=256, consts=2048, types=256,
        try_depth=16, no_stdio=True, no_time=True, storage=1 * MB, ram=160 * KB, c_stack=8 * KB,
        abi='ilp32', clock=True, console=True,
        source="vendor specs; assumed 160 KB of static DRAM for the VM and 1 MB of the app "
               "partition for bytecode (read from flash in place)")
_device('rp2040', title="Raspberry Pi Pico (RP2040), 264 KB SRAM / 2 MB flash",
        cpu="dual Cortex-M0+ @ 133 MHz",
        int_bits=32, float=1, heap=128 * KB, stack=2048, frames=256, consts=2048, types=256,
        try_depth=16, no_stdio=True, no_time=True, storage=1 * MB, ram=240 * KB, c_stack=8 * KB,
        abi='ilp32', clock=True, console=True,
        source="vendor specs; assumed 240 KB of SRAM for the VM and 1 MB of flash for bytecode")
_device('z80-16', title="Z80 calculator class (TI-83 Plus / TI-84 Plus), 16-bit lean build",
        cpu="Z80 @ 6-15 MHz, 64 KB address space",
        int_bits=16, float=0, heap=4096, stack=192, frames=32, consts=192, types=24, try_depth=8,
        mark_stack=16, lean=True, no_stdio=True, no_time=True, storage=8 * KB, ram=7 * KB,
        c_stack=0, abi='z80', clock=False,
        source="vm/ports/z80 (16-bit lean build of TIL 3.1 §9): VM data at 0xC000-0xDBFF; "
               "storage assumes the VM leaves 8 KB of the 48 KB code area for bytecode")


def load_device(spec: str) -> Device:
    """A built-in profile name, or a JSON file {"base": "cortex-m3-64k", "heap": 32768, ...}."""
    if spec in DEVICES:
        return DEVICES[spec]
    if spec.endswith('.json') or os.path.isfile(spec):
        try:
            with open(spec, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise ValueError(f"cannot read device file {spec}: {e}")
        if not isinstance(data, dict):
            raise ValueError(f"device file {spec}: expected a JSON object")
        base_name = data.pop('base', 'desktop')
        if base_name not in DEVICES:
            raise ValueError(f"device file {spec}: unknown base device '{base_name}'")
        name = data.pop('name', os.path.splitext(os.path.basename(spec))[0])
        values = DEVICES[base_name].as_dict()
        values.pop('name')
        values['title'] = f"custom ({spec}, based on {base_name})"
        values.update(data)
        return Device(str(name), **values)
    raise KeyError(spec)


# ═══════════════════════════════════════════════════════════════════════════════
#                     RAM OF THE C VM (sizeof(tilvm) per target ABI)
# ═══════════════════════════════════════════════════════════════════════════════
# A layout model of the structs in vm/tilvm.h: (size, alignment) per C type and
# ABI, and the ordinary C rules for struct and union layout.  tests/test_fit.py
# compares it with the C compilers (host; and avr-gcc, arm-none-eabi-gcc, sdcc
# when installed) - when tilvm.h changes, update vm_struct_layout() below.



def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _struct(*members: Tuple[int, int]) -> Tuple[int, int]:
    off, al = 0, 1
    for size, align in members:
        off = _align(off, align) + size
        al = max(al, align)
    return _align(off, al), al


def _union(*members: Tuple[int, int]) -> Tuple[int, int]:
    al = max(a for _s, a in members)
    return _align(max(s for s, _a in members), al), al


def _array(t: Tuple[int, int], n: int) -> Tuple[int, int]:
    return t[0] * n, t[1]


def vm_struct_layout(dev: Device) -> Dict[str, int]:
    """sizeof() of the C VM structs for this device (mirror of vm/tilvm.h)."""
    a = ABIS[dev.abi]
    u8, u16, u32, u64, ptr, cint = a['u8'], a['u16'], a['u32'], a['u64'], a['ptr'], a['int']
    tv_int = {16: u16, 32: u32, 64: u64}[dev.int_bits]
    tv_ref = u16 if dev.heap <= 65535 else u32          # tv_ref and tv_size
    members = [tv_int, tv_ref, u16]
    if dev.float == 1:
        members.append(a['float'])
    elif dev.float == 2:
        members.append(a['double'])
    tv_val = _struct(u8, _union(*members))
    tv_frame = _struct(u32, u32, u16, u16, tv_ref, u8)
    tv_try = _struct(u32, u32, u16, u16)
    state = _struct(u32, u16, u16, u16, u16, tv_ref, u8, u8, u8, u8, u32, u32, u32,
                    u64 if dev.int_bits == 64 else u32,
                    _array(tv_frame, dev.frames), _array(tv_try, dev.try_depth), _array(tv_val, dev.stack))
    host = _struct(*([ptr] * 9 + [cint, ptr]))
    arena = [_array(u8, dev.heap), tv_val] + ([u64] if dev.float == 2 or dev.int_bits == 64 else [])
    vm = _struct(ptr, ptr, ptr, ptr, ptr, u32, host,                    # host bindings
                 ptr, u32, u32, u32, u32, u32, u32, u32,                 # program
                 u16, u16, u16, u16, u16, u16, u16, u8, u8,
                 _array(u32, dev.consts), _array(u32, dev.types), u32,
                 ptr, tv_val, _array(u8, 128), u16, u8, _array(tv_ref, dev.mark_stack),   # runtime
                 state, _union(*arena))
    return {'tilvm': vm[0], 'tilvm_state': state[0], 'tv_val': tv_val[0], 'tv_frame': tv_frame[0],
            'tv_try': tv_try[0], 'stack_bytes': tv_val[0] * dev.stack, 'heap': dev.heap}


def vm_struct_size(dev: Device) -> int:
    return vm_struct_layout(dev)['tilvm']


def human(n: Optional[int]) -> str:
    if n is None:
        return "-"
    if n < 1024:
        return f"{n} B"
    if n < MB:
        v = n / 1024
        return f"{v:.0f} KB" if v >= 100 or v == int(v) else f"{v:.1f} KB"
    v = n / MB
    return f"{v:.0f} MB" if v == int(v) else f"{v:.1f} MB"


# ═══════════════════════════════════════════════════════════════════════════════
#                        BYTECODE: STACK EFFECT OF EVERY OPCODE
# ═══════════════════════════════════════════════════════════════════════════════
# Operand layouts come from til.OP_OPERANDS (the compiler's table).  Adding an
# opcode = one line in STACK_EFFECTS.  An opcode without an entry makes the
# analysis of its function "incomplete" (reported, never guessed).

# control flow after an instruction
NEXT, JUMP, BRANCH, ITER, DEFAULT, TRY, STOP = range(7)
# what an instruction does besides moving values
K_CALL, K_CALL_VALUE, K_INVOKE, K_NATIVE, K_FUNC_REF = range(1, 6)


class Effect:
    """pops/pushes: operand-stack values consumed/produced (int or f(operands)).
    peak/need: transient slots used / checked above the depth *before* the
    instruction (C VM internals, e.g. make_array keeps the new buffer on the
    stack).  flow: NEXT JUMP BRANCH ITER (FOR_ITER: +1 when it continues, jumps
    without) DEFAULT (ARG_DEFAULT) TRY (handler entered with the stack of the TRY
    + the error message) STOP.  tryd: change of the recovery-point depth."""
    __slots__ = ('pops', 'pushes', 'flow', 'peak', 'need', 'kind', 'alloc', 'tryd')

    def __init__(self, pops=0, pushes=0, flow=NEXT, peak=0, need=0, kind=0, alloc=None, tryd=0):
        self.pops, self.pushes, self.flow = pops, pushes, flow
        self.peak, self.need, self.kind, self.alloc, self.tryd = peak, need, kind, alloc, tryd


def _arg(i: int, mul: int = 1, add: int = 0) -> Callable[[Tuple[int, ...]], int]:
    return lambda a: a[i] * mul + add


_O = til.Op
E = Effect
STACK_EFFECTS: Dict[int, Effect] = {
    _O.NOP: E(),
    _O.PUSH_NIL: E(0, 1), _O.PUSH_TRUE: E(0, 1), _O.PUSH_FALSE: E(0, 1),
    _O.PUSH_I8: E(0, 1), _O.PUSH_I16: E(0, 1), _O.PUSH_CONST: E(0, 1),
    _O.POP: E(1, 0), _O.DUP: E(1, 2), _O.DUP2: E(2, 4), _O.SWAP: E(2, 2),
    _O.LOAD_LOCAL: E(0, 1), _O.STORE_LOCAL: E(1, 0), _O.LOAD_GLOBAL: E(0, 1), _O.STORE_GLOBAL: E(1, 0),
    _O.LOAD_UPVAL: E(0, 1), _O.STORE_UPVAL: E(1, 0), _O.LOAD_CELL: E(0, 1), _O.STORE_CELL: E(1, 0),
    _O.MAKE_CELL: E(alloc='cell'), _O.LOAD_UPVAL_RAW: E(0, 1), _O.ARG_DEFAULT: E(flow=DEFAULT),
    _O.ADD: E(2, 1), _O.SUB: E(2, 1), _O.MUL: E(2, 1), _O.DIV: E(2, 1), _O.MOD: E(2, 1),
    _O.POW: E(2, 1), _O.NEG: E(1, 1), _O.NOT: E(1, 1), _O.BAND: E(2, 1), _O.BOR: E(2, 1),
    _O.BXOR: E(2, 1), _O.BNOT: E(1, 1), _O.SHL: E(2, 1), _O.SHR: E(2, 1),
    _O.EQ: E(2, 1), _O.NE: E(2, 1), _O.LT: E(2, 1), _O.LE: E(2, 1), _O.GT: E(2, 1), _O.GE: E(2, 1),
    _O.JUMP: E(flow=JUMP), _O.JUMP_IF_FALSE: E(1, 0, flow=BRANCH), _O.JUMP_IF_TRUE: E(1, 0, flow=BRANCH),
    _O.CALL: E(_arg(1), 1, kind=K_CALL),                       # h func, b argc
    _O.CALL_VALUE: E(_arg(0, add=1), 1, kind=K_CALL_VALUE),    # callee below the arguments
    _O.INVOKE: E(_arg(2, add=1), 1, kind=K_INVOKE),            # h name, b method, b argc (+ receiver)
    _O.RETURN: E(1, 0, flow=STOP), _O.RETURN_NIL: E(flow=STOP),
    _O.NATIVE: E(_arg(1), 1, kind=K_NATIVE),                   # b native, b argc
    _O.NEW_ARRAY: E(_arg(0), 1, peak=1, need=2, alloc='array'),
    _O.NEW_MAP: E(_arg(0, mul=2), 1, peak=1, need=2, alloc='map'),
    _O.NEW_STRUCT: E(_arg(1), 1, alloc='struct'),
    _O.GET_INDEX: E(2, 1), _O.SET_INDEX: E(3, 0), _O.GET_FIELD: E(1, 1), _O.SET_FIELD: E(2, 0),
    _O.GET_SLOT: E(1, 1),
    _O.MAKE_CLOSURE: E(_arg(1), 1, kind=K_FUNC_REF, alloc='closure'),
    _O.PUSH_FUNC: E(0, 1, kind=K_FUNC_REF),
    _O.CONCAT: E(_arg(0), 1, alloc='string'),
    _O.FOR_ITER: E(0, 1, flow=ITER),
    _O.MATCH_TYPE: E(1, 1), _O.CAST: E(1, 1), _O.IS_ARRAY_LEN: E(1, 1),
    _O.TRY: E(flow=TRY), _O.TRY_TIMEOUT: E(1, 0, flow=TRY), _O.END_TRY: E(tryd=-1),
    _O.PANIC: E(1, 0, flow=STOP), _O.UNWRAP_OR_RETURN: E(1, 1), _O.UNWRAP_OR_PANIC: E(1, 1),
    _O.HALT: E(flow=STOP),
    # TIL 3.1 opcodes from parallel branches are added here, one line each.
}
del E

# Transient stack use of builtins in the C VM (slots written / checked above the
# depth before the instruction, with empty collections; data-dependent ones are
# reported separately).  Keys: native or method name.
NATIVE_TRANSIENT = {'sum': (2, 3), 'str_bytes': (1, 2), 'args': (1, 2)}
METHOD_TRANSIENT = {'sum': (2, 3), 'split': (2, 3), 'copy': (1, 2), 'slice': (1, 2), 'join': (0, 3),
                    'keys': (1, 2), 'values': (1, 2)}
DATA_DEPENDENT_METHODS = {
    'split': "puts every piece on the stack", 'join': "puts every element on the stack",
    'keys': "puts every key on the stack", 'values': "puts every value on the stack",
    'copy': "puts every element on the stack", 'slice': "puts the elements of an array slice on the stack",
}
ALLOC_NATIVES = {'str': 'string', 'format': 'string', 'input': 'string', 'chr': 'string',
                 'type_of': 'string', 'range': 'array', 'read_file': 'string', 'args': 'array',
                 'bytes_to_str': 'string', 'str_bytes': 'array'}
ALLOC_METHODS = {'push': 'array push', 'insert': 'array insert', 'set': 'map insert',
                 'split': 'strings', 'replace': 'string', 'trim': 'string', 'repeat': 'string',
                 'to_upper': 'string', 'to_lower': 'string', 'slice': 'slice', 'copy': 'copy',
                 'keys': 'array', 'values': 'array', 'join': 'string', 'to_string': 'string'}
GROWING_ALLOCS = ('array push', 'array insert', 'map insert')

# ── builtins and TILVM_LEAN (vm/tilvm.c, #ifndef TILVM_LEAN) ───────────────────
LEAN_MISSING_NATIVES = ('input', 'chr', 'ord', 'type_of', 'clock_ms', 'random', 'seed', 'sleep_ms',
                        'read_file', 'write_file', 'file_exists', 'args', 'bytes_to_str', 'str_bytes')
# builtin methods per receiver in a full build, and the ones a lean build lacks
METHOD_RECEIVERS = {
    'str': ('len', 'is_empty', 'contains', 'starts_with', 'ends_with', 'find', 'index_of', 'count',
            'eq', 'get', 'trim', 'repeat', 'to_upper', 'to_lower', 'slice', 'replace', 'split',
            'to_int', 'to_float', 'to_string'),
    'array': ('len', 'is_empty', 'push', 'pop', 'remove', 'insert', 'get', 'set', 'contains',
              'index_of', 'count', 'reverse', 'sort', 'first', 'last', 'clear', 'sum', 'min', 'max',
              'copy', 'slice', 'join', 'to_string', 'map', 'filter', 'reduce', 'each', 'any', 'all'),
    'map': ('len', 'is_empty', 'set', 'get', 'get_or', 'has', 'contains', 'delete', 'remove',
            'values', 'copy', 'keys', 'clear', 'to_string'),
    'Option/Result': ('is_some', 'is_ok', 'is_none', 'is_err', 'unwrap', 'expect', 'unwrap_or',
                      'error', 'to_string'),
}
LEAN_MISSING_METHODS = {
    'str': ('trim', 'repeat', 'replace', 'split', 'to_float'),
    'array': ('remove', 'insert', 'reverse', 'sort', 'sum', 'min', 'max', 'copy', 'slice', 'join'),
    'map': ('values', 'copy', 'clear'),
}
FLOAT_NATIVES = ('sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'floor', 'ceil', 'round', 'pow')
MATH_NATIVES = ('sin', 'cos', 'tan', 'log', 'exp')
SCREEN_NATIVES = tuple(til.NATIVES[40:54])
FILE_NATIVES = ('read_file', 'write_file', 'file_exists')
CLOCK_NATIVES = ('clock_ms', 'sleep_ms')
# --run does not compare the output with the desktop VM when these are used
NONDETERMINISTIC_NATIVES = ('clock_ms', 'random', 'sleep_ms', 'key_pressed', 'wait_key')


def lean_method_status(name: str) -> Tuple[str, List[str]]:
    """('ok' | 'missing' | 'partly', receivers that lack it) for a builtin method in TILVM_LEAN."""
    has = [r for r, ms in METHOD_RECEIVERS.items() if name in ms]
    lacks = [r for r in has if name in LEAN_MISSING_METHODS.get(r, ())]
    if not lacks:
        return 'ok', []
    return ('missing' if len(lacks) == len(has) else 'partly'), lacks


# ═══════════════════════════════════════════════════════════════════════════════
#                     ANALYSIS OF ONE FUNCTION (abstract interpretation)
# ═══════════════════════════════════════════════════════════════════════════════

MAX_REVISITS = 64       # a join point whose stack keeps growing is a loop that never balances


class CallSite:
    __slots__ = ('pc', 'op', 'args', 'depth', 'tries', 'targets')

    def __init__(self, pc, op, args, depth, tries):
        self.pc, self.op, self.args, self.depth, self.tries = pc, op, args, depth, tries
        self.targets: List[Tuple[int, int]] = []     # (callee, values consumed by its frame)


class FuncFacts:
    """What the analysis knows about one bytecode function."""

    def __init__(self, index: int, f, start: int, end: int):
        self.index, self.name = index, f.name
        self.n_params, self.n_required, self.n_locals, self.flags = f.n_params, f.n_required, f.n_locals, f.flags
        self.start, self.end = start, end
        self.complete = True
        self.problems: List[Tuple[int, str]] = []        # (pc, text)
        self.max_occ = 0         # operand-stack slots written (above the locals)
        self.max_need = 0        # ... and checked by the C VM (CHECK_STACK, make_array, ...)
        self.max_try = 0         # recovery points opened inside this function
        self.calls: List[CallSite] = []
        self.func_refs: Set[int] = set()
        self.natives: List[Tuple[int, str, int]] = []           # (pc, name, argc)
        self.methods: List[Tuple[int, str, int, int]] = []      # (pc, name, method id, argc)
        self.allocs: List[Tuple[int, str]] = []
        self.loops: List[Tuple[int, int]] = []                  # (first pc, back-edge pc)
        self.const_uses: Dict[int, List[int]] = {}
        self.float_casts: List[int] = []
        self.formats: List[Tuple[int, Optional[str]]] = []
        self.data_dependent: List[Tuple[int, str]] = []
        self.timeouts: List[int] = []
        self.ops: Set[int] = set()
        self.range_consts: Dict[int, int] = {}                  # pc of NATIVE range -> length
        self.reachable = 0

    def fail(self, pc: int, text: str):
        self.complete = False
        if len(self.problems) < 8:
            self.problems.append((pc, text))


def _decode(code: bytes, pc: int, end: int):
    """(op, operands, next pc) or (op, None, reason)."""
    op = code[pc]
    if op not in til.OP_NAMES:
        return op, None, f"unknown opcode 0x{op:02X}"
    p = pc + 1
    args = []
    for kind in til.OP_OPERANDS.get(op, ''):
        size = 1 if kind in 'bc' else 2
        if p + size > end:
            return op, None, f"{til.OP_NAMES[op]} is cut off at the end of the function"
        if kind == 'b':
            v = code[p]
        elif kind == 'c':
            v = code[p] - 256 if code[p] > 127 else code[p]
        else:
            v = code[p] | (code[p + 1] << 8)
            if kind == 's' and v >= 0x8000:
                v -= 0x10000
        args.append(v)
        p += size
    return op, tuple(args), p


def _value(x, args) -> int:
    return x(args) if callable(x) else x


def _flow_target(args, nxt: int) -> int:
    """Jump target: the last operand is relative to the end of the instruction."""
    return nxt + args[-1]


def analyze_function(an: 'Analysis', fidx: int) -> FuncFacts:
    m, code = an.module, an.module.code
    start, end = an.ranges[fidx]
    ff = FuncFacts(fidx, m.funcs[fidx], start, end)
    if start >= end:
        ff.fail(start, "the function has no code")
        return ff
    states: Dict[int, Tuple[int, int]] = {}
    decoded: Dict[int, Tuple[int, Tuple[int, ...], int]] = {}
    visits: Dict[int, int] = {}
    work = [(start, 0, 0)]
    while work:
        pc, d, t = work.pop()
        old = states.get(pc)
        if old is not None:
            if d <= old[0] and t <= old[1]:
                continue
            n = visits.get(pc, 0) + 1
            visits[pc] = n
            if n > MAX_REVISITS:
                ff.fail(pc, "the operand stack grows on every pass through a loop")
                continue
            d, t = max(d, old[0]), max(t, old[1])
        states[pc] = (d, t)
        if pc not in decoded:
            op, args, nxt = _decode(code, pc, end)
            if args is None:
                ff.fail(pc, nxt)
                continue
            decoded[pc] = (op, args, nxt)
        op, args, nxt = decoded[pc]
        eff = STACK_EFFECTS.get(op)
        if eff is None:
            ff.fail(pc, f"no stack-effect entry for opcode {til.OP_NAMES[op]} (0x{op:02X})")
            continue
        pops = _value(eff.pops, args)
        if d < pops:
            ff.fail(pc, f"{til.OP_NAMES[op]} pops {pops} value(s) from an operand stack of {d}")
            continue
        d2 = d - pops + _value(eff.pushes, args)
        t2 = max(t + eff.tryd, 0)
        flow = eff.flow
        if flow == NEXT:
            succ = [(nxt, d2, t2)]
        elif flow == JUMP:
            succ = [(_flow_target(args, nxt), d2, t2)]
        elif flow in (BRANCH, DEFAULT):
            succ = [(nxt, d2, t2), (_flow_target(args, nxt), d2, t2)]
        elif flow == ITER:
            succ = [(nxt, d2, t2), (_flow_target(args, nxt), d, t2)]
        elif flow == TRY:
            succ = [(nxt, d2, t2 + 1), (_flow_target(args, nxt), d2 + 1, t2)]
        else:
            succ = []
        for s in succ:
            if not start <= s[0] < end:
                ff.fail(pc, "control leaves the function" if s[0] == end else "jump outside the function")
                continue
            work.append(s)

    # an incomplete function: also decode it linearly (as far as the operand layouts
    # are known) so builtins, calls and conversions after the problem are still seen
    swept: Set[int] = set()
    if not ff.complete:
        pc = start
        while pc < end:
            if pc not in decoded:
                op, args, nxt = _decode(code, pc, end)
                if args is None:
                    break
                decoded[pc] = (op, args, nxt)
            if pc not in states:
                swept.add(pc)
            pc = decoded[pc][2]

    # second pass over the fixpoint: peaks, call sites, builtins, allocations
    prev_of = {nxt: pc for pc, (_op, _a, nxt) in decoded.items()}
    targets = set()
    for pc, (op, args, nxt) in decoded.items():
        eff = STACK_EFFECTS.get(op)
        if eff is not None and eff.flow in (JUMP, BRANCH, ITER, DEFAULT, TRY):
            targets.add(_flow_target(args, nxt))
    has_maps = an.has_maps
    for pc in sorted(decoded):
        op, args, nxt = decoded[pc]
        eff = STACK_EFFECTS.get(op)
        if eff is None or (pc not in states and pc not in swept):
            continue
        d, t = states.get(pc, (0, 0))
        pops = _value(eff.pops, args)
        if pc in swept:
            d = pops                  # depth unknown: the facts below are still collected
        if d < pops:
            continue
        ff.reachable += 1
        ff.ops.add(op)
        d2 = d - pops + _value(eff.pushes, args)
        peak, need = eff.peak, eff.need
        if eff.kind == K_NATIVE:
            name = til.NATIVES[args[0]] if args[0] < len(til.NATIVES) else f"native#{args[0]}"
            ff.natives.append((pc, name, args[1]))
            peak, need = NATIVE_TRANSIENT.get(name, (0, 0))
            if name == 'range':
                n = _const_range(an, decoded, prev_of, targets, pc)
                if n is None:
                    peak, need = 1 - args[1], 2 - args[1]
                    ff.data_dependent.append((pc, "range(...) puts every value on the stack before "
                                                  "building the array"))
                else:
                    ff.range_consts[pc] = n
                    peak, need = n + 1 - args[1], n + 2 - args[1]
            elif name == 'iter' and has_maps:
                peak, need = 1, 2
                ff.data_dependent.append((pc, "a for-loop over a map first puts every key on the stack"))
            elif name == 'format':
                ff.formats.append((pc, _const_string(an, decoded, prev_of, pc)))
            if name in ALLOC_NATIVES:
                ff.allocs.append((pc, ALLOC_NATIVES[name]))
        elif eff.kind == K_INVOKE:
            name = an.const_text(args[0])
            ff.methods.append((pc, name, args[1], args[2]))
            if args[1] != til.NO_METHOD:
                peak, need = METHOD_TRANSIENT.get(name, (0, 0))
                if name in DATA_DEPENDENT_METHODS:
                    ff.data_dependent.append((pc, f".{name}() {DATA_DEPENDENT_METHODS[name]}"))
                if name in ALLOC_METHODS and (name != 'set' or has_maps):
                    ff.allocs.append((pc, ALLOC_METHODS[name]))
            ff.calls.append(CallSite(pc, op, args, d, t))
        elif eff.kind in (K_CALL, K_CALL_VALUE):
            ff.calls.append(CallSite(pc, op, args, d, t))
        elif eff.kind == K_FUNC_REF:
            ff.func_refs.add(args[0])
        if op == _O.PUSH_CONST:
            ff.const_uses.setdefault(args[0], []).append(pc)
        elif op == _O.CAST and args[0] == til.CAST_KINDS['float']:
            ff.float_casts.append(pc)
        elif op == _O.TRY_TIMEOUT:
            ff.timeouts.append(pc)
        elif op == _O.SET_INDEX and has_maps:
            ff.allocs.append((pc, 'map insert'))
        if eff.alloc:
            ff.allocs.append((pc, eff.alloc))
        occ = max(d, d2, d + peak)
        ff.max_occ = max(ff.max_occ, occ)
        ff.max_need = max(ff.max_need, occ, d + need)
        ff.max_try = max(ff.max_try, t + (1 if eff.flow == TRY else 0))
        if eff.flow in (JUMP, BRANCH):
            target = _flow_target(args, nxt)
            if target <= pc:
                ff.loops.append((target, pc))
    return ff


def _push_value(an: 'Analysis', ins) -> Any:
    """The constant a push instruction pushes (None when it is not a constant push)."""
    op, args, _nxt = ins
    if op in (_O.PUSH_I8, _O.PUSH_I16):
        return args[0]
    if op == _O.PUSH_TRUE:
        return True
    if op == _O.PUSH_FALSE:
        return False
    if op == _O.PUSH_CONST and args[0] < len(an.module.consts):
        return an.module.consts[args[0]]
    return None


def _const_range(an, decoded, prev_of, targets, pc) -> Optional[int]:
    """Length of range(lo, hi, inclusive) when the compiler pushed three constants."""
    op, args, _ = decoded[pc]
    if args[1] != 3:
        return None
    vals, p = [], pc
    for _ in range(3):
        if p in targets:
            return None
        p = prev_of.get(p)
        if p is None:
            return None
        vals.append(_push_value(an, decoded[p]))
    incl, hi, lo = vals
    if not (isinstance(lo, int) and isinstance(hi, int) and not isinstance(lo, bool)
            and not isinstance(hi, bool) and isinstance(incl, bool)):
        return None
    return max(0, hi + (1 if incl else 0) - lo)


def _const_string(an, decoded, prev_of, pc) -> Optional[str]:
    p = prev_of.get(pc)
    if p is None:
        return None
    v = _push_value(an, decoded[p])
    return v.decode('utf-8', 'replace') if isinstance(v, bytes) else None


# ═══════════════════════════════════════════════════════════════════════════════
#                 ANALYSIS OF THE WHOLE PROGRAM (call graph, worst cases)
# ═══════════════════════════════════════════════════════════════════════════════

class Recursion:
    """A recursive part of the call graph (a strongly connected component)."""

    def __init__(self, members: List[int]):
        self.members = members
        self.slots_per_level = 0     # value-stack slots added by one more level
        self.tries_per_level = 0     # recovery points added by one more level
        self.call_pc: Optional[Tuple[int, int]] = None      # (function, pc) of a recursive call
        self.base = 0                # value-stack slots in use below its first frame
        self.frames_below = 1        # call depth of its first frame
        self.tries_below = 0
        self.top_need = 0            # slots the topmost level needs above its base (with callees)
        self.top_occ = 0             # ... and writes
        self.top_frames = 1
        self.top_tries = 0


class Analysis:
    """Static facts about a bytecode program, independent of any device."""

    def __init__(self, data: bytes, filename: str = "<program>"):
        self.data = bytes(data)
        self.filename = filename
        self.size = len(self.data)
        try:
            self.module = m = til.TBCModule.from_bytes(self.data)
        except Exception as e:      # malformed file: struct.error, IndexError, ValueError ...
            raise ValueError(f"cannot read the bytecode: {e}")
        if not m.funcs or not 0 <= m.entry < len(m.funcs):
            raise ValueError("cannot read the bytecode: no entry function")
        self.has_lines = bool(self.data[5] & 2)
        self.n_globals = len(m.globals)
        self.uses_float_consts = m.uses_float
        order = sorted(range(len(m.funcs)), key=lambda i: (m.funcs[i].offset, i))
        self.ranges: Dict[int, Tuple[int, int]] = {}
        for k, i in enumerate(order):
            start = m.funcs[i].offset
            end = m.funcs[order[k + 1]].offset if k + 1 < len(order) else len(m.code)
            self.ranges[i] = (min(start, len(m.code)), min(max(end, start), len(m.code)))
        self._line_pcs = [pc for pc, _l in m.lines]
        self._line_nums = [line for _pc, line in m.lines]
        self.methods_by_name: Dict[str, List[int]] = {}
        for _t, name_k, fidx in m.methods:
            if 0 <= fidx < len(m.funcs):
                self.methods_by_name.setdefault(self.const_text(name_k), []).append(fidx)
        m.prelude = {mid: f for mid, f in m.prelude.items() if 0 <= f < len(m.funcs)}
        self.field_names = {fld for _n, fields in m.types[til.TYPE_ERR + 1:] for fld in fields}
        self._facts: Dict[int, FuncFacts] = {}
        self.has_maps = self._program_has_maps()
        self._resolve()

    # ── helpers ─────────────────────────────────────────────────────────────
    def const_text(self, k: int) -> str:
        c = self.module.consts[k] if 0 <= k < len(self.module.consts) else None
        return c.decode('utf-8', 'replace') if isinstance(c, bytes) else f"#{k}"

    def facts(self, fidx: int) -> FuncFacts:
        ff = self._facts.get(fidx)
        if ff is None:
            ff = self._facts[fidx] = analyze_function(self, fidx)
        return ff

    def line_of(self, pc: int) -> int:
        import bisect
        i = bisect.bisect_right(self._line_pcs, pc) - 1
        return self._line_nums[i] if i >= 0 else 0

    def where(self, fidx: int, pc: int) -> str:
        """'line 12 in main' or, without line tables, 'main+26' (offset in the function)."""
        name = self.module.funcs[fidx].name
        if name.startswith('__prelude_'):            # .map()/.filter()/... are written in TIL
            return f"in .{name[len('__prelude_'):]}()"
        line = self.line_of(pc) if self.has_lines else 0
        if fidx == self.module.entry:
            return f"line {line} (global initialization)" if line else "the global initialization"
        if line:
            return f"line {line} in {name}"
        return f"{name}+{pc - self.ranges[fidx][0]}"

    def _program_has_maps(self) -> bool:
        """Maps exist only through NEW_MAP (literals, HashMap.new()) - decode linearly."""
        code = self.module.code
        for fidx, (start, end) in self.ranges.items():
            pc = start
            while pc < end:
                op, args, nxt = _decode(code, pc, end)
                if args is None:
                    return True          # cannot tell: assume maps
                if op == _O.NEW_MAP:
                    return True
                pc = nxt
        return False

    def arity_ok(self, fidx: int, argc: int) -> bool:
        f = self.module.funcs[fidx]
        return f.n_required <= argc <= f.n_params

    def call_targets(self, site: CallSite, address_taken: Set[int]) -> List[Tuple[int, int]]:
        """Possible callees of a call site and how many stack values become the
        callee's parameters (the callee's frame starts that far below the top)."""
        m = self.module
        op, args = site.op, site.args
        if op == _O.CALL:
            return [(args[0], args[1])] if args[0] < len(m.funcs) else []
        if op == _O.CALL_VALUE:     # any function whose address is taken (closures, fn values)
            return [(g, args[0]) for g in sorted(address_taken) if self.arity_ok(g, args[0])]
        name, mid, argc = self.const_text(args[0]), args[1], args[2]
        out = [(g, argc + 1) for g in self.methods_by_name.get(name, ()) if self.arity_ok(g, argc + 1)]
        if mid in m.prelude and self.arity_ok(m.prelude[mid], argc + 1):
            out.append((m.prelude[mid], argc + 1))          # map/filter/... written in TIL
        if name in self.field_names:                          # struct field holding a function
            out += [(g, argc) for g in sorted(address_taken) if self.arity_ok(g, argc)]
        return out

    # ── reachability, call graph, worst cases ────────────────────────────────
    def _resolve(self):
        m = self.module
        entry = m.entry
        address_taken: Set[int] = set()
        while True:
            reach = {entry}
            work = [entry]
            while work:
                f = work.pop()
                for site in self.facts(f).calls:
                    for g, _c in self.call_targets(site, address_taken):
                        if g not in reach:
                            reach.add(g)
                            work.append(g)
            refs = set()
            for f in reach:
                refs |= {g for g in self.facts(f).func_refs if 0 <= g < len(m.funcs)}
            if refs <= address_taken:
                break
            address_taken |= refs
        self.reachable = sorted(reach)
        self.address_taken = address_taken
        for f in self.reachable:
            for site in self.facts(f).calls:
                site.targets = self.call_targets(site, address_taken)
        self.complete = all(self.facts(f).complete for f in self.reachable)
        self._worst_cases()

    def _sccs(self) -> List[List[int]]:
        """Tarjan's algorithm (iterative); components come callees-first."""
        index: Dict[int, int] = {}
        low: Dict[int, int] = {}
        on_stack: Set[int] = set()
        stack: List[int] = []
        out: List[List[int]] = []
        counter = 0
        succ = {f: sorted({g for s in self.facts(f).calls for g, _c in s.targets}) for f in self.reachable}
        for root in self.reachable:
            if root in index:
                continue
            index[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            work = [(root, iter(succ[root]))]
            while work:
                v, it = work[-1]
                advanced = False
                for w in it:
                    if w not in index:
                        index[w] = low[w] = counter
                        counter += 1
                        stack.append(w)
                        on_stack.add(w)
                        work.append((w, iter(succ[w])))
                        advanced = True
                        break
                    if w in on_stack:
                        low[v] = min(low[v], index[w])
                if advanced:
                    continue
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    out.append(sorted(comp))
        return out

    def _worst_cases(self):
        """Per function (relative to its frame base, recursion cut after one level):
        slots written (occ) and needed (need), call depth, recovery points.  Then
        the program totals and, for recursion, the cost of every further level."""
        entry = self.module.entry
        self.sccs = self._sccs()
        comp_of = {f: i for i, comp in enumerate(self.sccs) for f in comp}
        occ: Dict[int, Optional[int]] = {}
        need: Dict[int, Optional[int]] = {}
        frames: Dict[int, Optional[int]] = {}
        tries: Dict[int, Optional[int]] = {}
        rec_of: Dict[int, Recursion] = {}
        self.recursions: List[Recursion] = []
        for ci, comp in enumerate(self.sccs):            # callees before callers
            rec = None
            for f in comp:
                for s in self.facts(f).calls:
                    for g, c in s.targets:
                        if comp_of.get(g) != ci:
                            continue
                        rec = rec or Recursion(comp)
                        ff = self.facts(f)
                        rel = ff.n_locals + s.depth - c
                        if rec.call_pc is None or rel > rec.slots_per_level:
                            rec.call_pc = (f, s.pc)
                        rec.slots_per_level = max(rec.slots_per_level, rel)
                        rec.tries_per_level = max(rec.tries_per_level, s.tries)
            for f in comp:
                ff = self.facts(f)
                occ[f] = need[f] = frames[f] = tries[f] = None
                if not ff.complete:
                    continue
                o = ff.n_locals + ff.max_occ
                n = ff.n_locals + ff.max_need
                if f != entry:
                    n = max(n, ff.n_locals + 4)          # push_frame keeps 4 slots free
                fr, tr = 1, ff.max_try
                unknown = False
                for s in ff.calls:
                    for g, c in s.targets:
                        if comp_of.get(g) == ci:         # recursive call: per-level cost
                            continue
                        if need.get(g) is None:
                            unknown = True
                            continue
                        rel = ff.n_locals + s.depth - c
                        o = max(o, rel + occ[g])
                        n = max(n, rel + need[g])
                        fr = max(fr, 1 + frames[g])
                        tr = max(tr, s.tries + tries[g])
                if not unknown:
                    occ[f], need[f], frames[f], tries[f] = o, n, fr, tr
            if rec is not None:
                rec_of[ci] = rec
                if all(need.get(f) is not None for f in comp):
                    rec.top_need = max(need[f] for f in comp)
                    rec.top_occ = max(occ[f] for f in comp)
                    rec.top_frames = max(frames[f] for f in comp)
                    rec.top_tries = max(tries[f] for f in comp)
                    self.recursions.append(rec)
        self.func_occ, self.func_need, self.func_frames, self.func_tries = occ, need, frames, tries
        g0 = self.n_globals
        self.slots_occ = None if occ.get(entry) is None else g0 + occ[entry]
        self.slots_need = None if need.get(entry) is None else g0 + need[entry]
        self.frames = frames.get(entry)
        self.tries = tries.get(entry)
        # where each recursion starts: longest call paths from the entry (recursion cut)
        base = {entry: g0}
        depth = {entry: 1}
        tdepth = {entry: 0}
        for ci in reversed(range(len(self.sccs))):     # callers before callees
            comp = self.sccs[ci]
            rec = rec_of.get(ci)
            entered = [f for f in comp if f in base]
            if rec is not None and entered:
                rec.base = max(base[f] for f in entered)
                rec.frames_below = max(depth[f] for f in entered)
                rec.tries_below = max(tdepth[f] for f in entered)
                for f in comp:                         # reached only from inside: one level deeper
                    if f not in base:
                        base[f] = rec.base + rec.slots_per_level
                        depth[f] = rec.frames_below + 1
                        tdepth[f] = rec.tries_below + rec.tries_per_level
            for f in comp:
                if f not in base:
                    continue
                ff = self.facts(f)
                for s in ff.calls:
                    for g, c in s.targets:
                        if comp_of.get(g) == ci:
                            continue
                        rel = ff.n_locals + s.depth - c
                        base[g] = max(base.get(g, 0), base[f] + rel)
                        depth[g] = max(depth.get(g, 0), depth[f] + 1)
                        tdepth[g] = max(tdepth.get(g, 0), tdepth[f] + s.tries)

    # ── summaries ───────────────────────────────────────────────────────────
    def reachable_facts(self) -> List[FuncFacts]:
        return [self.facts(f) for f in self.reachable]

    def problems(self) -> List[str]:
        out = []
        for ff in self.reachable_facts():
            for pc, text in ff.problems:
                out.append(f"{text} at {self.where(ff.index, pc)}")
        return out

    def int_constants(self) -> List[Tuple[int, int]]:
        return [(k, c) for k, c in enumerate(self.module.consts)
                if isinstance(c, int) and not isinstance(c, bool)]

    def uses(self, k: int) -> List[Tuple[int, int]]:
        """(function, pc) of the reachable PUSH_CONST k instructions."""
        return [(ff.index, pc) for ff in self.reachable_facts() for pc in ff.const_uses.get(k, ())]

    def worst_path(self, table: Optional[Dict[int, Optional[int]]] = None) -> List[int]:
        """The call chain behind a worst case (value-stack need by default; recursion cut)."""
        table = self.func_need if table is None else table
        by_slots = table is self.func_need
        path, f, seen = [self.module.entry], self.module.entry, set()
        while f not in seen:
            seen.add(f)
            ff = self.facts(f)
            best, best_val = None, -1
            for s in ff.calls:
                for g, c in s.targets:
                    v = table.get(g)
                    if g in seen or v is None:
                        continue
                    v += (ff.n_locals + s.depth - c) if by_slots else 0
                    if v > best_val:
                        best, best_val = g, v
            if best is None or (by_slots and best_val < ff.n_locals + ff.max_need):
                break
            path.append(best)
            f = best
        return path

    def display_name(self, fidx: int) -> str:
        name = self.module.funcs[fidx].name
        if fidx == self.module.entry:
            return "(start)"
        if name.startswith('__prelude_'):
            return f".{name[len('__prelude_'):]}()"
        return name


# ═══════════════════════════════════════════════════════════════════════════════
#                              CHECK AGAINST A DEVICE
# ═══════════════════════════════════════════════════════════════════════════════

class Finding:
    """An error (does not fit) or a warning; `reason` is the short form for the verdict."""
    __slots__ = ('text', 'reason')

    def __init__(self, text: str, reason: str = ""):
        self.text, self.reason = text, reason or text


class Report:
    def __init__(self, an: Analysis, dev: Device):
        self.an, self.dev = an, dev
        self.rows: List[Dict[str, Any]] = []
        self.errors: List[Finding] = []
        self.warnings: List[Finding] = []
        self.notes: List[str] = []
        self.recursion: List[Dict[str, Any]] = []
        self.run: Optional[Dict[str, Any]] = None
        self.vm_layout = vm_struct_layout(dev)

    def row(self, key: str, label: str, needs: str, limit: str, status: str, note: str = "",
            need_value: Optional[int] = None, limit_value: Optional[int] = None):
        self.rows.append({'key': key, 'resource': label, 'needs': needs, 'limit': limit, 'status': status,
                          'note': note, 'measured': None, 'need_value': need_value,
                          'limit_value': limit_value})

    def get_row(self, key: str) -> Optional[Dict[str, Any]]:
        return next((r for r in self.rows if r['key'] == key), None)

    def error(self, text: str, reason: str = ""):
        self.errors.append(Finding(text, reason))

    def warn(self, text: str):
        self.warnings.append(Finding(text))

    @property
    def fits(self) -> bool:
        return not self.errors

    def verdict(self) -> str:
        if self.fits:
            return "fits"
        reasons = []
        for e in self.errors:
            if e.reason not in reasons:
                reasons.append(e.reason)
        return "does not fit: " + "; ".join(reasons)


def _places(an: Analysis, sites: List[Tuple[int, int]], limit: int = 3) -> str:
    out = []
    for f, pc in sites:
        w = an.where(f, pc)
        if w not in out:
            out.append(w)
    more = len(out) - limit
    return ", ".join(out[:limit]) + (f" and {more} more" if more > 0 else "")


def _int_bits_needed(v: int) -> int:
    if -32768 <= v <= 32767:
        return 16
    if -2 ** 31 <= v < 2 ** 31:
        return 32
    return 64


def recursion_levels(rec: Recursion, dev: Device) -> Tuple[int, str]:
    """How many levels of a recursion fit the device, and which limit is hit first."""
    limits = [(dev.frames - rec.frames_below - rec.top_frames + 2, "call frames")]
    if rec.slots_per_level > 0:
        limits.append(((dev.stack - rec.base - rec.top_need) // rec.slots_per_level + 1, "value stack"))
    if rec.tries_per_level > 0:
        limits.append(((dev.try_depth - rec.tries_below - rec.top_tries) // rec.tries_per_level + 1,
                       "recovery points"))
    return min(limits)


def check(an: Analysis, dev: Device, strip_saving: int = 0) -> Report:
    """Compare the program's needs with a device profile."""
    r = Report(an, dev)
    m = an.module
    facts = an.reachable_facts()
    natives: Dict[str, List[Tuple[int, int]]] = {}
    methods: Dict[str, List[Tuple[int, int]]] = {}
    for ff in facts:
        for pc, name, _argc in ff.natives:
            natives.setdefault(name, []).append((ff.index, pc))
        for pc, name, mid, _argc in ff.methods:
            if mid != til.NO_METHOD:
                methods.setdefault(name, []).append((ff.index, pc))

    # ── storage and RAM ─────────────────────────────────────────────────────
    ok = dev.storage is None or an.size <= dev.storage
    r.row('bytecode', "bytecode", human(an.size), human(dev.storage), "ok" if ok else "FAIL",
          need_value=an.size, limit_value=dev.storage)
    if not ok:
        hint = f" (--strip would save {strip_saving} bytes)" if strip_saving else ""
        r.error(f"the bytecode is {an.size} bytes; the device keeps at most {dev.storage} bytes "
                f"for one program{hint}", f"bytecode {human(an.size)} > {human(dev.storage)} of storage")
    vm_bytes = r.vm_layout['tilvm']
    ram = vm_bytes + dev.vm_static_ram + dev.c_stack + (an.size if dev.bytecode_in_ram else 0)
    parts = [f"VM struct {human(vm_bytes)}"]
    if dev.vm_static_ram:
        parts.append(f"VM tables {human(dev.vm_static_ram)}")
    if dev.c_stack:
        parts.append(f"C stack {human(dev.c_stack)}")
    if dev.bytecode_in_ram:
        parts.append(f"bytecode {human(an.size)}")
    ok = dev.ram is None or ram <= dev.ram
    r.row('ram', "VM RAM", human(ram), human(dev.ram), "ok" if ok else "FAIL", " + ".join(parts),
          need_value=ram, limit_value=dev.ram)
    if not ok:
        where = " (this port copies the bytecode into RAM)" if dev.bytecode_in_ram else ""
        r.error(f"the VM needs {ram} bytes of RAM ({', '.join(parts)}){where}; the device has {dev.ram}",
                f"VM RAM {human(ram)} > {human(dev.ram)}")

    # ── tables checked when the VM loads the program ────────────────────────
    for key, label, n, limit, code in (('consts', "constants", len(m.consts), dev.consts, -4),
                                       ('types', "struct types", len(m.types), dev.types, -7)):
        ok = n <= limit
        r.row(key, label, str(n), str(limit), "ok" if ok else "FAIL", need_value=n, limit_value=limit)
        if not ok:
            r.error(f"the program has {n} {label}; the device's VM holds at most {limit} "
                    f"(tilvm_load fails with error {code})", f"{n} {label} > {limit}")
    limit = dev.stack - 8
    ok = an.n_globals <= limit
    r.row('globals', "globals", str(an.n_globals), str(limit), "ok" if ok else "FAIL",
          need_value=an.n_globals, limit_value=limit)
    if not ok:
        r.error(f"{an.n_globals} global variables leave no room on a {dev.stack}-slot value stack "
                f"(tilvm_load fails with error -8)", f"{an.n_globals} globals > {limit}")

    # ── integers ────────────────────────────────────────────────────────────
    ints = an.int_constants()
    bits = max([16] + [_int_bits_needed(c) for _k, c in ints])
    big = [(k, c) for k, c in ints if _int_bits_needed(c) > dev.int_bits]
    r.row('ints', "integers", f"{bits}-bit", f"{dev.int_bits}-bit", "ok" if not big else "FAIL")
    for k, c in big[:3]:
        uses = an.uses(k)
        at = f" ({_places(an, uses)})" if uses else ""
        r.error(f"the integer constant {c}{at} does not fit {dev.int_bits}-bit integers; the device's VM "
                f"refuses to load the program (error -6)",
                f"integer constant {c} needs {_int_bits_needed(c)}-bit integers")
    if len(big) > 3:
        r.error(f"... and {len(big) - 3} more integer constants too large for {dev.int_bits}-bit integers",
                f"{len(big)} integer constants too large")
    arith = {_O.MUL, _O.POW, _O.SHL} | ({_O.ADD, _O.SUB} if dev.int_bits == 16 else set())
    if dev.int_bits < 64 and any(ff.ops & arith for ff in facts):
        top = (1 << (dev.int_bits - 1)) - 1
        r.notes.append(f"integers are {dev.int_bits}-bit on this device: results beyond +/-{top} wrap around")

    # ── floats ──────────────────────────────────────────────────────────────
    float_k = [k for k, c in enumerate(m.consts) if isinstance(c, float)]
    casts = [(ff.index, pc) for ff in facts for pc in ff.float_casts]
    fl_natives = [n for n in FLOAT_NATIVES + ('float',) if n in natives]
    if 'to_float' in methods:
        fl_natives.append('to_float')
    uses_float = bool(an.uses_float_consts or casts or fl_natives)
    kinds = [k for k, on in (("constants", an.uses_float_consts), ("conversions", casts),
                             ("math", fl_natives)) if on]
    used = ", ".join(kinds) if kinds else "none"
    has = {0: "none", 1: "float32", 2: "float64"}[dev.float]
    fail = dev.float == 0 and uses_float
    r.row('floats', "floats", used, has, "FAIL" if fail else "ok")
    if dev.float == 0:
        if an.uses_float_consts:
            sites = [s for k in float_k for s in an.uses(k)]
            at = f" ({_places(an, sites)})" if sites else ""
            r.error(f"the program has float constants{at}; a VM built without floats refuses to load it "
                    f"(error -3)", "float constants (the device has no floats)")
        if casts:
            r.error(f"values are converted to float at {_places(an, casts)} (float(), `: float` / "
                    f"`-> float` annotations, `as float`); a VM without floats stops there",
                    "float conversions (the device has no floats)")
        if fl_natives:
            sites = [s for n in fl_natives for s in natives.get(n, methods.get(n, []))]
            listed = ", ".join(n + "()" for n in fl_natives)
            verb = "needs" if len(fl_natives) == 1 else "need"
            r.error(f"{listed} {verb} floats ({_places(an, sites)}); a VM without floats stops there",
                    f"{listed} {verb} floats")
        if 'random' in natives and not dev.lean:
            r.warn("random() returns an integer 0..32767 instead of a float on a VM without floats")
    elif dev.float == 1 and uses_float:
        r.notes.append("floats are single precision (float32) on this device: about 7 significant digits")
    if dev.no_math:
        names = [n for n in MATH_NATIVES if n in natives]
        if names:
            r.error(f"{', '.join(n + '()' for n in names)} need <math.h>; this device's VM is built with "
                    f"TILVM_NO_MATH", f"{', '.join(n + '()' for n in names)} (no math library)")

    # ── builtins left out of lean builds ────────────────────────────────────
    missing, partly = [], []
    if dev.lean:
        for name in LEAN_MISSING_NATIVES:
            if name in natives:
                missing.append((f"{name}()", natives[name]))
        for name in sorted(methods):
            status, lacks = lean_method_status(name)
            if status == 'ok':
                continue
            user = name in an.methods_by_name or name in an.field_names
            if status == 'missing' and not user:
                missing.append((f".{name}()", methods[name]))
            else:
                partly.append((name, lacks, methods[name], user))
    label = (f"{len(natives)} native{'s' if len(natives) != 1 else ''}, "
             f"{len(methods)} method{'s' if len(methods) != 1 else ''}") + ("" if an.complete else " (incomplete)")
    r.row('builtins', "builtins", label, "lean" if dev.lean else "full",
          "FAIL" if missing else ("warn" if partly else "ok"),
          note=("missing: " + ", ".join(n for n, _s in missing)) if missing else "")
    if missing:
        r.error("the lean VM of this device leaves out " +
                "; ".join(f"{n} ({_places(an, s, 2)})" for n, s in missing) +
                " - calling them stops the program", "not in the lean VM: " + ", ".join(n for n, _s in missing))
    for name, lacks, sites, user in partly:
        extra = " (fine when it is your own struct's method)" if user else ""
        r.warn(f".{name}() ({_places(an, sites, 2)}) is not available for {' or '.join(lacks)} values "
               f"in the lean VM{extra}")
    if dev.lean and 'format' in natives:
        specs = [ff_spec for ff in facts for ff_spec in ff.formats]
        shown = ", ".join(sorted({repr(s) for _pc, s in specs if s is not None})[:3])
        sites = natives['format']
        r.error(f"format specs ({shown or 'format()'} at {_places(an, sites, 2)}) are ignored by the lean VM: "
                f"values print unformatted", "format specs (ignored by the lean VM)")

    unknown = sorted(n for n in natives if n.startswith('native#'))
    if unknown:
        r.warn(f"the bytecode calls unknown builtins ({', '.join(unknown)}): every VM reports "
               f"\"?() is not available in this VM build\"")

    # ── device services ─────────────────────────────────────────────────────
    scr = [n for n in SCREEN_NATIVES if n in natives]
    if scr and not dev.screen:
        r.warn(f"{', '.join(n + '()' for n in scr[:4])} need a screen; this device has none (the VM "
               f"reports \"not available\" unless the port adds display hooks)")
    if not dev.lean:
        fs = [n for n in FILE_NATIVES if n in natives]
        if fs and not dev.files:
            r.warn(f"{', '.join(n + '()' for n in fs)} need a file system hook on this device "
                   f"({_places(an, [s for n in fs for s in natives[n]], 2)})")
        if 'input' in natives and not dev.console:
            r.warn("input() returns an empty string: the device has no text input")
    if not dev.clock:
        if any(ff.timeouts for ff in facts):
            r.warn("timeout(ms) counts VM steps on this device (no clock: 1 ms = 1000 steps)")
        cl = [n for n in CLOCK_NATIVES if n in natives and not dev.lean]
        if cl:
            r.warn(f"{', '.join(n + '()' for n in cl)} count VM steps on this device (no clock)")

    # ── value stack, call depth, recovery points ────────────────────────────
    if not an.complete:
        probs = an.problems() or ["a function could not be analysed"]
        for key, label, limit in (('stack', "value stack", dev.stack), ('frames', "call frames", dev.frames),
                                  ('try', "recovery points", dev.try_depth)):
            r.row(key, label, "?", str(limit), "?")
        r.error("cannot verify the stack use: analysis incomplete: " + "; ".join(probs[:3]),
                "analysis incomplete (" + probs[0] + ")")
    else:
        rec_note = ""
        if an.recursions:
            per = max(rc.slots_per_level for rc in an.recursions)
            rec_note = f"+{per} per recursion level" if per else "recursion"
        chains = {'stack': an.worst_path(), 'frames': an.worst_path(an.func_frames)}
        for key, label, need, limit, unit, what in (
                ('stack', "value stack", an.slots_need, dev.stack, "slots", "value-stack slots"),
                ('frames', "call frames", an.frames, dev.frames, "", "call frames"),
                ('try', "recovery points", an.tries, dev.try_depth, "", "nested retry/timeout/intent blocks")):
            ok = need <= limit
            note = rec_note if key == 'stack' else ("+1 per recursion level" if key == 'frames' and an.recursions
                                                    else "")
            r.row(key, label, f"{need} {unit}".strip(), str(limit), "ok" if ok else "FAIL", note,
                  need_value=need, limit_value=limit)
            path = [an.display_name(f) for f in chains.get(key, [])]
            if len(path) > 8:
                path = path[:3] + [f"... {len(path) - 6} more ..."] + path[-3:]
            chain = f" (deepest call chain: {' -> '.join(path)})" if len(path) > 2 else ""
            if not ok:
                r.error(f"the worst case needs {need} {what}{chain}; the device has {limit}",
                        f"{label} {need} > {limit}")
            elif need > 0.8 * limit and need > 4:
                r.warn(f"the worst case uses {need} of the {limit} {what}: little room left")
        for rec in an.recursions:
            levels, limiter = recursion_levels(rec, dev)
            names = [an.display_name(f) for f in rec.members]
            who = (f"{names[0]} calls itself" if len(names) == 1 else
                   f"{', '.join(names)} call each other")
            at = an.where(*rec.call_pc) if rec.call_pc else names[0]
            cost = (f"{rec.slots_per_level} value-stack slot{'s' if rec.slots_per_level != 1 else ''} and "
                    if rec.slots_per_level else "")
            r.recursion.append({'functions': names, 'at': at, 'slots_per_level': rec.slots_per_level,
                                'frames_per_level': 1, 'tries_per_level': rec.tries_per_level,
                                'levels': max(levels, 0), 'limited_by': limiter})
            if levels < 2:
                r.error(f"recursion: {who} ({at}) - not even two levels fit the {limiter} of this device",
                        f"recursion in {names[0]} does not fit")
            else:
                r.warn(f"recursion: {who} ({at}); the depth depends on the input. Each level needs "
                       f"{cost}1 call frame: at most {levels} levels fit on this device "
                       f"(limited by {limiter})")
        dd: Dict[str, List[Tuple[int, int]]] = {}
        for ff in facts:
            for pc, text in ff.data_dependent:
                dd.setdefault(text, []).append((ff.index, pc))
        for text, sites in dd.items():
            r.warn(f"stack use grows with the data: {text} ({_places(an, sites, 2)}): one free slot per "
                   f"element of the {dev.stack} on this device")
        for ff in facts:
            for pc, n in ff.range_consts.items():
                if n > dev.stack // 2:
                    r.warn(f"range() of {n} values at {an.where(ff.index, pc)} needs {n + 2} free stack slots; "
                           f"a counting loop (for i in a..b) needs none")
    r.row('heap', "heap", "depends on data", human(dev.heap), "-", need_value=None, limit_value=dev.heap)

    # ── allocations inside loops ────────────────────────────────────────────
    loops_out = []
    for ff in facts:
        if ff.name.startswith('__prelude_'):    # .map()/.filter()/...: bounded by the input array
            continue
        groups: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
        for pc, kind in ff.allocs:
            inner = [lp for lp in ff.loops if lp[0] <= pc <= lp[1]]
            if inner:
                lp = min(inner, key=lambda x: x[1] - x[0])
                groups.setdefault(lp, []).append((pc, kind))
        for (head, _back), sites in sorted(groups.items()):
            kinds: List[str] = []
            first: Dict[str, int] = {}
            for pc, kind in sites:
                if kind not in kinds:
                    kinds.append(kind)
                    first[kind] = pc
            grows = [f"{k} at {an.where(ff.index, first[k]).split(' in ')[0]}" for k in kinds
                     if k in GROWING_ALLOCS]
            loops_out.append((ff.index, head, kinds, grows))
    growing = [x for x in loops_out if x[3]]
    temporary = [x for x in loops_out if not x[3]]
    for f, head, kinds, grows in growing[:6]:
        r.warn(f"the loop at {an.where(f, head)} keeps allocating ({', '.join(grows)}): heap use grows "
               f"with the number of iterations ({human(dev.heap)} heap on this device)")
    if len(growing) > 6:
        r.warn(f"... and {len(growing) - 6} more loops whose heap use grows with the iterations")
    if temporary:
        places = ", ".join(an.where(f, head) for f, head, _k, _g in temporary[:3])
        more = f" and {len(temporary) - 3} more" if len(temporary) > 3 else ""
        kinds = sorted({k for _f, _h, ks, _g in temporary for k in ks})
        r.notes.append(f"{len(temporary)} loop{'s' if len(temporary) != 1 else ''} allocate temporary "
                       f"objects on every pass ({', '.join(kinds)}) at {places}{more}; the garbage "
                       f"collector reclaims them")
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#                 --run: THE C VM BUILT WITH THE DEVICE'S LIMITS
# ═══════════════════════════════════════════════════════════════════════════════

LOAD_ERRORS = {
    -1: "not a TIL bytecode file", -2: "unsupported bytecode version",
    -3: "the program uses floats and this VM has none", -4: "too many constants for this VM",
    -5: "malformed bytecode", -6: "an integer constant is too large for this VM",
    -7: "too many struct types for this VM", -8: "too many globals for the value stack",
}
# runtime errors that come from the device's limits or build (not from the program's logic)
DEVICE_ERRORS = ("out of memory", "stack overflow", "too many nested retry/timeout blocks",
                 "range too large", "array too large", "string too long", "is not available",
                 "built without floating point", "need a build with <math.h>", "no method '")


def _cache_dir() -> str:
    cache = os.environ.get('TIL_FIT_CACHE') or os.path.join(os.path.expanduser('~'), '.til', 'cache')
    try:
        os.makedirs(cache, exist_ok=True)
        return cache
    except OSError:
        return tempfile.gettempdir()


VM_OPT = '-O2'      # optimization of the device VM builds (part of the cache key)


def device_vm_flags(dev: Device) -> List[str]:
    return dev.c_flags() + ['-DTILVM_STATS'] + (['-DTILVM_PACK'] if dev.abi in PACKED_ABIS else [])


def build_device_vm(dev: Device, verbose: bool = False) -> Tuple[Optional[str], str]:
    """Build (once, cached by source + flags) the desktop C VM with the device's
    limits and TILVM_STATS.  Returns (executable, error text)."""
    vm_dir = til.find_vm_sources()
    cc = til.TILCompiler().find_c_compiler()
    if vm_dir is None:
        return None, "cannot find the C VM sources (vm/tilvm.c); set TIL_HOME"
    if cc is None:
        return None, "--run needs a C compiler (gcc, clang or cc)"
    sources = [os.path.join(vm_dir, n) for n in ('tilvm.c', 'tilvm.h', 'main.c')]
    flags = device_vm_flags(dev)
    digest = hashlib.sha256((cc + ' ' + VM_OPT + ' ' + ' '.join(flags)).encode())
    for path in sources:
        with open(path, 'rb') as f:
            digest.update(f.read())
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', dev.name)
    exe = os.path.join(_cache_dir(), f"tilvm-fit-{safe}-{digest.hexdigest()[:16]}" +
                       ('.exe' if sys.platform == 'win32' else ''))
    if not os.path.exists(exe):
        tmp = f"{exe}.{os.getpid()}.tmp"
        cmd = [cc, '-std=c99', VM_OPT] + flags + ['-o', tmp, sources[0], sources[2], '-lm']
        if verbose:
            print("[til fit] building the device VM: " + " ".join(cmd), file=sys.stderr)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return None, "building the device VM failed:\n" + res.stderr
        os.replace(tmp, exe)
    return exe, ""


def _exec_vm(exe: str, tbc: str, prog_args: List[str], max_steps: int, stdin_data: bytes,
             timeout: Optional[float], stats: bool) -> Dict[str, Any]:
    cmd = [exe] + (['--stats'] if stats else []) + (['--max-steps', str(max_steps)] if max_steps else [])
    cmd += [tbc] + list(prog_args)
    res: Dict[str, Any] = {'exit_code': None, 'stdout': '', 'errors': '', 'stats': {}, 'timeout': False}
    try:
        p = subprocess.run(cmd, input=stdin_data, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        res['timeout'] = True
        return res
    res['exit_code'] = p.returncode
    res['stdout'] = p.stdout.decode('utf-8', 'replace')
    other = []
    for line in p.stderr.decode('utf-8', 'replace').splitlines():
        if line.startswith('[tilvm] '):
            for key, value in re.findall(r'(\w+)=(\d+)', line):
                res['stats'][('peak_' + key) if line.startswith('[tilvm] peak') else key] = int(value)
        else:
            other.append(line)
    res['errors'] = "\n".join(other)
    mm = re.search(r'cannot load .* \(error (-\d+)', res['errors'])
    if mm:
        res['load_error'] = int(mm.group(1))
    mm = re.search(r'^(?:runtime|fatal) error: (.*)$', res['errors'], re.M)
    res['error'] = mm.group(1) if mm else None
    return res


def run_on_device(dev: Device, data: bytes, prog_args: List[str], max_steps: int = 0,
                  timeout: Optional[float] = None, stdin_data: bytes = b"", compare: bool = True,
                  verbose: bool = False) -> Dict[str, Any]:
    """Run the program on the device-configured C VM and (when `compare`) on the
    desktop C VM.  Raises RuntimeError when no VM can be built."""
    exe, err = build_device_vm(dev, verbose)
    if exe is None:
        raise RuntimeError(err)
    desktop = til._cvm_binary(verbose) if compare else None
    fd, tbc = tempfile.mkstemp(suffix='.tbc')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        out = _exec_vm(exe, tbc, prog_args, max_steps, stdin_data, timeout, True)
        out['desktop'] = (_exec_vm(desktop, tbc, prog_args, max_steps, stdin_data, timeout, False)
                          if desktop else None)
    finally:
        os.unlink(tbc)
    return out


def apply_run(r: Report, run: Dict[str, Any], compared: bool, why_not: str = ""):
    """Add the measurements of a --run to the report."""
    dev, an = r.dev, r.an
    r.run = run
    st = run.get('stats', {})
    if run.get('timeout'):
        r.warn("the run did not finish within the --timeout: no measurements")
        return
    if 'load_error' in run:
        code = run['load_error']
        r.error(f"the device's VM refuses to load the program: {LOAD_ERRORS.get(code, 'error')} "
                f"(error {code})", f"the VM refuses to load it ({LOAD_ERRORS.get(code, code)})")
        return
    for key, stat in (('stack', 'peak_stack'), ('frames', 'peak_frames'), ('try', 'peak_try')):
        row = r.get_row(key)
        if row is not None and stat in st:
            row['measured'] = str(st[stat])
    heap = r.get_row('heap')
    if heap is not None and 'peak_heap' in st:
        live = st.get('peak_live', 0)
        heap['measured'] = human(st['peak_heap']) + (f" (live {human(live)})" if st.get('gc') else "")
    err = run.get('error')
    desk = run.get('desktop')
    code = run.get('exit_code')
    if code is not None and code < 0 and err is None:
        err = f"the VM was killed by signal {-code}"
    stopped = err is not None and err.startswith('step limit exceeded')
    if stopped:
        r.notes.append("the run stopped at --max-steps; the measurements cover that part only")
        err = None
    if err is not None:
        desk_err = desk.get('error') if desk else None
        if desk is not None and desk_err == err:
            r.warn(f"the program stops with \"{err}\" on the desktop VM too (not a device limit)")
        elif any(p in err for p in DEVICE_ERRORS + ("killed by signal",)) or \
                (desk is not None and desk_err is None):
            r.error(f"the run on the device's VM stops with \"{err}\"", f"run: {err}")
        else:
            r.warn(f"the run stops with \"{err}\"")
    elif desk is not None and compared and not stopped and desk.get('error') is None:
        if desk['stdout'] != run['stdout'] or desk['exit_code'] != run['exit_code']:
            a, b = run['stdout'].splitlines(), desk['stdout'].splitlines()
            i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            dline = repr(a[i]) if i < len(a) else "(end)"
            wline = repr(b[i]) if i < len(b) else "(end)"
            r.warn(f"the output differs from the desktop VM from line {i + 1} on: {dline} instead of "
                   f"{wline} (integer width or float precision?)")
    if not compared and why_not:
        r.notes.append(f"output not compared with the desktop VM ({why_not})")
    for key, stat, limit, label in (('stack', 'peak_stack', dev.stack, "value stack"),
                                    ('frames', 'peak_frames', dev.frames, "call frames")):
        v = st.get(stat)
        if v is not None and err is None and v > 0.8 * limit and v > 4:
            r.warn(f"the run used {v} of the {limit} {label}{' slots' if key == 'stack' else ''}: "
                   f"little room left")
    live = st.get('peak_live')
    if live and live > 0.75 * dev.heap:
        r.warn(f"live data reached {human(live)} of the {human(dev.heap)} heap after a collection: "
               f"a little more data will exhaust it")
    # self-check: without recursion and data-dependent builtins the static numbers are upper bounds
    if an.complete and not an.recursions and err is None and \
            not any(ff.data_dependent for ff in an.reachable_facts()):
        for stat, bound in (('peak_stack', an.slots_occ), ('peak_frames', an.frames), ('peak_try', an.tries)):
            if bound is not None and st.get(stat, 0) > bound:
                r.warn(f"measured {stat[5:]} ({st[stat]}) exceeds the static bound ({bound}): please "
                       f"report this (til fit analysis bug)")


# ═══════════════════════════════════════════════════════════════════════════════
#                                    OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def _wrap(text: str, width: int = 92, indent: str = "    ") -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width - len(indent):
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return [("  - " if i == 0 else indent) + ln for i, ln in enumerate(lines)]


def format_report(r: Report, functions: bool = False) -> str:
    an, dev = r.an, r.dev
    out = [f"til fit: {an.filename} on {dev.name}",
           f"device: {dev.title}" + (f"; {dev.cpu}" if dev.cpu else ""),
           f"VM build: {dev.summary()}", ""]
    measured = r.run is not None
    cols = ["resource", "program needs", "device", "status"]
    if measured:
        cols.insert(2, "measured")
    table = []
    for row in r.rows:
        cells = [row['resource'], row['needs']]
        if measured:
            cells.append(row['measured'] or "")
        cells += [row['limit'], row['status'] + (f"  {row['note']}" if row['note'] else "")]
        table.append(cells)
    widths = [max(len(c[i]) for c in [cols] + table) for i in range(len(cols) - 1)]
    for cells in [cols] + table:
        out.append("  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)) + "  " + cells[-1])
    if functions:
        out += ["", "functions (reachable; slots relative to the frame base, recursion cut):",
                "  function                         locals  operands  frame need  calls"]
        for ff in sorted(an.reachable_facts(), key=lambda x: -(an.func_need.get(x.index) or 0)):
            need = an.func_need.get(ff.index)
            out.append(f"  {ff.name[:32]:<32} {ff.n_locals:>6}  {ff.max_occ:>8}  "
                       f"{'?' if need is None else need:>10}  {len(ff.calls):>5}"
                       + ("" if ff.complete else "  (incomplete)"))
    for title, items in (("errors", [e.text for e in r.errors]), ("warnings", [w.text for w in r.warnings]),
                         ("notes", r.notes)):
        if items:
            out += ["", title + ":"]
            for text in items:
                out += _wrap(text)
    if r.run is not None and r.run.get('stats'):
        st = r.run['stats']
        out += ["", f"run: exit code {r.run.get('exit_code')}, {st.get('steps', 0)} steps, "
                    f"{st.get('gc', 0)} garbage collections"]
    out += ["", "verdict: " + r.verdict()]
    return "\n".join(out)


def report_json(r: Report) -> Dict[str, Any]:
    an = r.an
    funcs = []
    for ff in an.reachable_facts():
        funcs.append({'name': ff.name, 'index': ff.index, 'locals': ff.n_locals, 'operand_stack': ff.max_occ,
                      'operand_need': ff.max_need, 'frame_need': an.func_need.get(ff.index),
                      'calls': len(ff.calls), 'complete': ff.complete,
                      'problems': [t for _pc, t in ff.problems]})
    run = None
    if r.run is not None:
        run = {k: v for k, v in r.run.items() if k not in ('desktop',)}
        if r.run.get('desktop') is not None:
            d = r.run['desktop']
            run['desktop'] = {'exit_code': d.get('exit_code'), 'error': d.get('error'),
                              'same_output': d.get('stdout') == r.run.get('stdout')}
    return {
        'program': an.filename, 'device': r.dev.as_dict(), 'fits': r.fits, 'verdict': r.verdict(),
        'errors': [e.text for e in r.errors], 'reasons': [e.reason for e in r.errors],
        'warnings': [w.text for w in r.warnings], 'notes': r.notes,
        'resources': [{k: v for k, v in row.items()} for row in r.rows],
        'static': {
            'bytecode_bytes': an.size, 'constants': len(an.module.consts), 'types': len(an.module.types),
            'globals': an.n_globals, 'functions': len(an.module.funcs), 'reachable_functions': len(an.reachable),
            'complete': an.complete, 'problems': an.problems(),
            'stack_slots': an.slots_need, 'stack_slots_written': an.slots_occ, 'call_frames': an.frames,
            'recovery_points': an.tries, 'recursion': r.recursion,
            'uses_float_constants': an.uses_float_consts, 'functions_detail': funcs,
        },
        'vm': {'struct_bytes': r.vm_layout['tilvm'], 'value_bytes': r.vm_layout['tv_val'],
               'c_flags': r.dev.c_flags()},
        'run': run,
    }


def format_devices() -> str:
    hdr = ("device", "int", "float", "heap", "stack", "frames", "lean", "storage", "RAM", "VM struct",
           "description")
    rows = []
    for d in DEVICES.values():
        rows.append((d.name, str(d.int_bits), {0: "-", 1: "f32", 2: "f64"}[d.float], human(d.heap), str(d.stack),
                     str(d.frames), "yes" if d.lean else "no", human(d.storage), human(d.ram),
                     human(vm_struct_size(d)), d.title))
    widths = [max(len(r[i]) for r in [hdr] + rows) for i in range(len(hdr) - 1)]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[-1] for r in [hdr] + rows)


# ═══════════════════════════════════════════════════════════════════════════════
#                                  ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

USAGE = """\
usage: til fit PROGRAM --device NAME [options] [-- ARGS...]
       til fit --list-devices [--json]

Will a TIL program (.til or .tbc) fit a device?  Static analysis of the
bytecode against the device's C VM build; --run also runs it under those limits.

options:
  --device NAME      a built-in profile (see --list-devices) or a JSON file
                     {"base": "cortex-m3-64k", "heap": 32768, ...}
  --run              run on the C VM built with the device's limits (needs a C
                     compiler) and report the measured high-water marks
  --json             machine-readable report
  --strip            check the stripped bytecode (no line tables), as deployed
  --functions        per-function stack table
  --max-steps N      --run: stop after N steps (loop iterations + calls)
  --timeout SEC      --run: give up after SEC seconds
  --stdin FILE       --run: text for input() (default: none)
  --show-output      --run: print the program's output before the report
  -v, --verbose      show how the device VM is built
  -- ARGS            --run: program arguments (args())

exit status: 0 fits, 1 does not fit, 2 usage error or the program cannot be analysed
"""


def load_program(path: str, strip: bool = False) -> Tuple[bytes, int]:
    """Bytecode of a .til (compiled) or .tbc file, and how many bytes --strip saves."""
    if path.endswith('.tbc'):
        with open(path, 'rb') as f:
            data = f.read()
        if strip and len(data) > 5 and data[5] & 2:
            try:
                data = til.TBCModule.from_bytes(data).serialize(strip=True)
            except Exception as e:
                raise ValueError(f"cannot read the bytecode: {e}")
        return data, 0
    with open(path, encoding='utf-8') as f:
        source = f.read()
    module = til.compile_to_bytecode(source, path)
    full = module.serialize()
    stripped = module.serialize(strip=True)
    return (stripped if strip else full), (0 if strip else len(full) - len(stripped))


def check_build(data: bytes, device: str, filename: str) -> bool:
    """`til build --target vm --device NAME`: check the built bytecode; print the
    result; False = does not fit (the build fails)."""
    try:
        dev = load_device(device)
    except KeyError:
        print(f"Error: unknown device '{device}' (til fit --list-devices)", file=sys.stderr)
        return False
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False
    try:
        an = Analysis(data, filename)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False
    r = check(an, dev)
    if not r.fits:
        print(format_report(r), file=sys.stderr)
        return False
    def row(key):
        x = r.get_row(key)
        return f"{x['needs']} of {x['limit']}" if x else "?"
    print(f"Fits {dev.name}: bytecode {row('bytecode')}, value stack {row('stack')}, "
          f"call frames {row('frames')}, VM RAM {row('ram')}")
    for w in r.warnings:
        print("\n".join(ln.replace("  - ", "", 1) if k == 0 else ln
                        for k, ln in enumerate(_wrap("warning: " + w.text, indent="  "))))
    return True


def main(argv: List[str]) -> int:
    device, program = None, None
    want_run = want_json = strip = functions = show_output = verbose = list_devices = False
    max_steps, timeout, stdin_path = 0, None, None
    prog_args: List[str] = []
    i = 0
    try:
        while i < len(argv):
            a = argv[i]
            value = None
            if a.startswith('--') and '=' in a:
                a, value = a.split('=', 1)
            def take():
                nonlocal i
                if value is not None:
                    return value
                if i + 1 >= len(argv):
                    raise ValueError(f"{a} needs a value")
                i += 1
                return argv[i]
            if a == '--':
                prog_args = argv[i + 1:]
                break
            if a in ('-h', '--help'):
                print(USAGE)
                return 0
            if a == '--list-devices':
                list_devices = True
            elif a == '--device':
                device = take()
            elif a == '--run':
                want_run = True
            elif a == '--json':
                want_json = True
            elif a == '--strip':
                strip = True
            elif a == '--functions':
                functions = True
            elif a == '--show-output':
                show_output = True
            elif a in ('-v', '--verbose'):
                verbose = True
            elif a == '--max-steps':
                max_steps = int(take())
            elif a == '--timeout':
                timeout = float(take())
            elif a == '--stdin':
                stdin_path = take()
            elif a.startswith('-'):
                raise ValueError(f"unknown option {a}")
            elif program is None:
                program = a
            else:
                prog_args.append(a)
            i += 1
    except ValueError as e:
        print(f"til fit: {e}\n\n{USAGE}", file=sys.stderr)
        return 2

    if list_devices:
        if want_json:
            print(json.dumps([dict(d.as_dict(), vm_struct_bytes=vm_struct_size(d)) for d in DEVICES.values()],
                             indent=2))
        else:
            print(format_devices())
        return 0
    if program is None or device is None:
        print("til fit: " + ("no program given" if program is None else "--device NAME is required "
                             "(til fit --list-devices)") + "\n\n" + USAGE, file=sys.stderr)
        return 2
    try:
        dev = load_device(device)
    except KeyError:
        print(f"til fit: unknown device '{device}'; known: {', '.join(DEVICES)}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"til fit: {e}", file=sys.stderr)
        return 2
    if not os.path.exists(program):
        print(f"til fit: file not found: {program}", file=sys.stderr)
        return 2
    try:
        data, saving = load_program(program, strip)
        an = Analysis(data, program)
    except (til.CompileError, SyntaxError) as e:
        print(str(e), file=sys.stderr)
        return 2
    except (OSError, ValueError, UnicodeDecodeError) as e:
        print(f"til fit: {e}", file=sys.stderr)
        return 2
    report = check(an, dev, saving)
    if want_run:
        stdin_data = b""
        if stdin_path:
            try:
                with open(stdin_path, 'rb') as f:
                    stdin_data = f.read()
            except OSError as e:
                print(f"til fit: {e}", file=sys.stderr)
                return 2
        used = {name for ff in an.reachable_facts() for _pc, name, _a in ff.natives}
        why_not = ""
        odd = sorted(used & set(NONDETERMINISTIC_NATIVES))
        if 'random' in odd and dev.int_bits == 64:
            odd.remove('random')        # the same generator as the desktop VM
        if odd:
            why_not = "the program uses " + ", ".join(n + "()" for n in odd)
        try:
            run = run_on_device(dev, data, prog_args, max_steps, timeout, stdin_data, compare=True,
                                verbose=verbose)
        except RuntimeError as e:
            print(f"til fit --run: {e}", file=sys.stderr)
            return 2
        apply_run(report, run, compared=not why_not, why_not=why_not)
        if show_output and not want_json:
            sys.stdout.write(run.get('stdout', ''))
            if run.get('stdout') and not run['stdout'].endswith('\n'):
                sys.stdout.write('\n')
            print("--- end of program output ---")
    if want_json:
        print(json.dumps(report_json(report), indent=2, default=str))
    else:
        print(format_report(report, functions))
    return 0 if report.fits else 1


if __name__ == '__main__':
    sys.modules.setdefault('tilfit', sys.modules[__name__])
    sys.exit(main(sys.argv[1:]))
