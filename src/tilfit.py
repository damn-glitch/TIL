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
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

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
    'max_handlers': "TILVM_MAX_HANDLERS: effect handler entries (active handle blocks + running clauses)",
    'mark_stack': "TILVM_MARK_STACK: GC mark stack entries",
    'max_funcs': "TILVM_MAX_FUNCS: function records decoded at load (null = the header's default: "
                 "consts, 0 in lean builds; 0 = no table)",
    'field_cache': "TILVM_FIELD_CACHE: field/method lookup cache entries, a power of two (null = the "
                   "header's default: 64, 0 in lean builds)",
    'lean': "TILVM_LEAN: builtins of the lean build only",
    'addr16': "TILVM_ADDR16: 16-bit bytecode offsets and object lengths (programs below 64 KB)",
    'no_effects': "TILVM_NO_EFFECTS: no effect handlers (handle ... with ... reports 'not available')",
    'no_gfx': "TILVM_NO_GFX: no screen/keys natives",
    'gfx_max_w': "TILVM_GFX_MAX_W: widest screen screen_init() accepts (pixels)",
    'gfx_max_h': "TILVM_GFX_MAX_H: tallest screen screen_init() accepts (pixels)",
    'no_stdio': "TILVM_NO_STDIO", 'no_time': "TILVM_NO_TIME", 'no_math': "TILVM_NO_MATH",
    'rom': "TILVM_ROM: storage qualifier for the VM's font table with the device's compiler "
           "(e.g. __flash for avr-gcc; not used by the desktop builds of --run)",
    # the device around the VM
    'storage': "bytes available for one program's bytecode (None = no practical limit)",
    'ram': "RAM for the VM: its struct, static data, C stack (and the bytecode when it is "
           "copied to RAM); None = no practical limit",
    'c_stack': "C stack reserved for the VM's own functions inside `ram`",
    'vm_static_ram': "RAM taken by the VM's tables and strings (AVR keeps them in RAM)",
    'bytecode_in_ram': "the port copies the bytecode into RAM (AVR: Harvard architecture)",
    'abi': "struct layout model: lp64, ilp32, avr, z80, ez80",
    'clock': "the port provides a millisecond clock (else timeouts count steps)",
    'screen': "display and keys (the port installs the screen/keys hooks)",
    'files': "a file system hook (read_file / write_file / file_exists)",
    'console': "text input for input()",
    'args': "the port passes program arguments to args() (a command line)",
    'title': "", 'cpu': "", 'source': "where the numbers come from",
}

BOOL_KEYS = ('lean', 'addr16', 'no_effects', 'no_gfx', 'no_stdio', 'no_time', 'no_math', 'bytecode_in_ram',
             'clock', 'screen', 'files', 'console', 'args')
STR_KEYS = ('rom', 'abi', 'title', 'cpu', 'source')
NULLABLE_KEYS = ('storage', 'ram', 'max_funcs', 'field_cache')
GFX_LIMIT = 4096                  # the language's largest screen (TILVM_GFX_MAX_W/H default)

_DESKTOP_BUILD = dict(int_bits=64, float=2, heap=256 * KB, stack=8192, frames=1000, consts=4096,
                      types=512, try_depth=32, max_handlers=32, mark_stack=64, max_funcs=None,
                      field_cache=None, lean=False, addr16=False, no_effects=False, no_gfx=False,
                      gfx_max_w=GFX_LIMIT, gfx_max_h=GFX_LIMIT, no_stdio=False, no_time=False,
                      no_math=False, rom='')
_BASE = dict(_DESKTOP_BUILD, storage=None, ram=None, c_stack=0, vm_static_ram=0,
             bytecode_in_ram=False, abi='lp64', clock=True, screen=False, files=False,
             console=False, args=False, title='', cpu='', source='')
# (key, macro) of the switches that are -DMACRO without a value
_SWITCHES = (('lean', 'TILVM_LEAN'), ('addr16', 'TILVM_ADDR16'), ('no_stdio', 'TILVM_NO_STDIO'),
             ('no_time', 'TILVM_NO_TIME'), ('no_math', 'TILVM_NO_MATH'), ('no_effects', 'TILVM_NO_EFFECTS'),
             ('no_gfx', 'TILVM_NO_GFX'))


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
        for key in DEVICE_KEYS:              # JSON types: true is not 1, 16.0 is not 16
            v = getattr(self, key)
            if key in BOOL_KEYS:
                need(isinstance(v, bool), f"{key} must be true or false")
            elif key in STR_KEYS:
                need(isinstance(v, str), f"{key} must be a string")
            elif not (key in NULLABLE_KEYS and v is None):
                need(isinstance(v, int) and not isinstance(v, bool), f"{key} must be a whole number")
        need(self.int_bits in (16, 32, 64), "int_bits must be 16, 32 or 64")
        need(self.float in (0, 1, 2), "float must be 0, 1 or 2")
        need(not (self.float == 2 and self.abi in PACKED_ABIS),
             f"float 2 (double) is 32-bit with the {self.abi} toolchain: use float 1")
        need(256 <= self.heap <= 0xFFFFFFFF, "heap must be 256..2^32-1 bytes")
        need(not (self.addr16 and self.heap > 0xFFFF), "addr16 needs a heap of at most 65535 bytes")
        need(16 <= self.stack <= 0xFFFF, "stack must be 16..65535 slots")
        need(2 <= self.frames <= 0xFFFF, "frames must be 2..65535")
        need(1 <= self.consts <= 0xFFFF, "consts must be 1..65535")
        need(3 <= self.types <= 0xFFFF, "types must be 3..65535")
        need(1 <= self.try_depth <= 0xFFFF, "try_depth must be 1..65535")
        need(1 <= self.max_handlers <= 0xFFFF, "max_handlers must be 1..65535")
        need(self.mark_stack >= 1, "mark_stack must be >= 1")
        need(self.max_funcs is None or 0 <= self.max_funcs <= 0xFFFF, "max_funcs must be 0..65535 or null")
        need(self.field_cache is None or (0 <= self.field_cache <= 0x8000 and
                                          self.field_cache & (self.field_cache - 1) == 0),
             "field_cache must be 0 or a power of two up to 32768, or null")
        for key in ('gfx_max_w', 'gfx_max_h'):
            need(1 <= getattr(self, key) <= GFX_LIMIT, f"{key} must be 1..{GFX_LIMIT}")
        need(re.fullmatch(r'([A-Za-z_]\w*)?', self.rom) is not None, "rom must be a C identifier or empty")
        need(self.abi in ABIS, f"abi must be one of {', '.join(ABIS)}")
        for key in ('storage', 'ram'):
            v = getattr(self, key)
            need(v is None or v > 0, f"{key} must be a positive number of bytes or null")
        for key in ('c_stack', 'vm_static_ram'):
            need(getattr(self, key) >= 0, f"{key} must be >= 0")

    @property
    def funcs_table(self) -> int:
        """TILVM_MAX_FUNCS of the build (0 = no table: records are read from the bytecode)."""
        if self.max_funcs is not None:
            return self.max_funcs
        return 0 if self.lean else self.consts

    @property
    def cache_entries(self) -> int:
        """TILVM_FIELD_CACHE of the build."""
        if self.field_cache is not None:
            return self.field_cache
        return 0 if self.lean else 64

    def as_dict(self) -> Dict[str, Any]:
        d = {'name': self.name}
        for key in DEVICE_KEYS:
            d[key] = getattr(self, key)
        return d

    def c_flags(self) -> List[str]:
        """-D flags of the device's C VM build (TILVM_STATS/TILVM_PACK not included;
        device_vm_flags() gives the flags of the desktop emulation that --run builds)."""
        flags = [f"-DTILVM_INT_BITS={self.int_bits}", f"-DTILVM_FLOAT={self.float}",
                 f"-DTILVM_HEAP_SIZE={self.heap}", f"-DTILVM_STACK_SIZE={self.stack}",
                 f"-DTILVM_MAX_FRAMES={self.frames}", f"-DTILVM_MAX_CONSTS={self.consts}",
                 f"-DTILVM_MAX_TYPES={self.types}", f"-DTILVM_MAX_TRY={self.try_depth}",
                 f"-DTILVM_MAX_HANDLERS={self.max_handlers}", f"-DTILVM_MARK_STACK={self.mark_stack}"]
        if self.max_funcs is not None:
            flags.append(f"-DTILVM_MAX_FUNCS={self.max_funcs}")
        if self.field_cache is not None:
            flags.append(f"-DTILVM_FIELD_CACHE={self.field_cache}")
        if self.gfx_max_w != GFX_LIMIT or self.gfx_max_h != GFX_LIMIT:
            flags += [f"-DTILVM_GFX_MAX_W={self.gfx_max_w}", f"-DTILVM_GFX_MAX_H={self.gfx_max_h}"]
        for key, macro in _SWITCHES:
            if getattr(self, key):
                flags.append(f"-D{macro}")
        if self.rom:
            flags.append(f"-DTILVM_ROM={self.rom}")
        return flags

    def summary(self) -> str:
        fl = {0: "no floats", 1: "float32", 2: "float64"}[self.float]
        extra = [x for x, on in (("no effect handlers", self.no_effects), ("no screen natives", self.no_gfx))
                 if on]
        return (f"int{self.int_bits}, {fl}, {human(self.heap)} heap, {self.stack} stack slots, "
                f"{self.frames} frames, {'lean' if self.lean else 'full'} VM" +
                "".join(", " + x for x in extra))


DEVICES: Dict[str, Device] = {}


def _device(name: str, **values):
    DEVICES[name] = Device(name, **values)


_device('desktop', title="Desktop PC, the default C VM build", cpu="x86-64 / ARM64",
        clock=True, screen=True, files=True, console=True, args=True,
        source="vm/Makefile `make` (the defaults of vm/tilvm.h)")
_device('ti84ce', title="TI-84 Plus CE graphing calculator",
        cpu="eZ80 @ 48 MHz, 256 KB RAM (~150 KB for programs), 3 MB flash",
        int_bits=32, float=1, heap=32 * KB, stack=1024, frames=128, consts=1024, types=128,
        try_depth=16, mark_stack=32, addr16=True, gfx_max_w=320, gfx_max_h=240, no_stdio=True,
        storage=65505, ram=96 * KB, c_stack=4 * KB, bytecode_in_ram=True, abi='ez80', clock=True,
        screen=True, files=True, console=True,
        source="vm/ports/ti84ce/makefile (TIL_PROFILE; the CE toolchain's double is 32-bit); one "
               "AppVar holds at most 65505 bytes and the port copies it to RAM; 96 KB of the "
               "user RAM for the VM's data, C stack and bytecode is an assumption")
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
        clock=True, screen=True, files=True, console=True, args=True,
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
        try_depth=8, no_stdio=True, no_time=True, storage=256 * KB - 56 * KB, ram=64 * KB,
        c_stack=4 * KB, vm_static_ram=128, abi='ilp32', clock=False,
        source="vm/ports/cortex-m/Makefile (verified profile); VM code 53.2 KB of the "
               "256 KB flash (arm-none-eabi-gcc -Os, newlib-nano)")
_device('atmega1284', title="ATmega1284P, 16 KB RAM / 128 KB flash",
        cpu="8-bit AVR @ 16-20 MHz, verified in simavr",
        int_bits=32, float=1, heap=5120, stack=160, frames=24, consts=160, types=24, try_depth=6,
        mark_stack=16, rom='__flash', no_stdio=True, no_time=True, storage=128 * KB - 73 * KB,
        ram=16 * KB, c_stack=1536, vm_static_ram=3630, bytecode_in_ram=True, abi='avr', clock=False,
        source="vm/ports/avr/Makefile (verified profile); avr-gcc -Os: VM code 69.1 KB, VM "
               "strings/tables in RAM 3630 bytes (the font in flash); the port copies the bytecode "
               "to RAM")
_device('arduino-mega', title="Arduino Mega 2560 (ATmega2560), 8 KB RAM / 256 KB flash",
        cpu="8-bit AVR @ 16 MHz",
        int_bits=16, float=1, heap=1536, stack=96, frames=16, consts=96, types=16, try_depth=4,
        max_handlers=8, mark_stack=16, lean=True, no_gfx=True, rom='__flash', no_stdio=True,
        no_time=True, storage=256 * KB - 48 * KB, ram=8 * KB, c_stack=1536, vm_static_ram=2680,
        bytecode_in_ram=True, abi='avr', clock=False,
        source="lean 16-bit build in the style of `make tiny`, sized for 8 KB (8 effect handler "
               "entries, no screen natives: the board has no display); avr-gcc -Os: VM code "
               "44.9 KB, VM strings/tables in RAM 2680 bytes; bytecode copied to RAM")
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
        cpu="Z80 @ 6-15 MHz, 64 KB address space, verified in the ucsim simulator",
        int_bits=16, float=0, heap=4096, stack=192, frames=32, consts=192, types=24, try_depth=8,
        mark_stack=16, lean=True, addr16=True, no_effects=True, no_gfx=True, no_stdio=True,
        no_time=True, storage=0xEFF0 - 0xD802, ram=0xD800 - 0xC000, c_stack=0, abi='z80',
        clock=False,
        source="vm/ports/z80/Makefile and RESULTS.md: VM state at 0xC000-0xD7FF, the program "
               "at 0xD800-0xEFEF (u16 length + bytecode), the C stack in its own 2 KB at "
               "0xF800 (not counted); no effect handlers and no screen natives (they would "
               "not fit below 0xC000)")


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
        if not isinstance(base_name, str) or base_name not in DEVICES:
            raise ValueError(f"device file {spec}: unknown base device {base_name!r}")
        name = data.pop('name', os.path.splitext(os.path.basename(spec))[0])
        if not isinstance(name, str) or not name:
            raise ValueError(f"device file {spec}: name must be a string")
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
# when installed).  VM_STRUCTS below mirrors the header member by member: when
# vm/tilvm.h gets a field, add one line at the same place.


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


def _f(name: str, ctype: str, count: Any = 1, when: Optional[Callable[['Device'], bool]] = None):
    """A struct member: C type (an ABI type, a typedef below or an earlier struct), array
    length (a number, a Device attribute name or a function of the device) and, for
    members inside #if/#ifdef, when the member exists."""
    return name, ctype, count, when


def _typedefs(dev: 'Device') -> Dict[str, Tuple[int, int]]:
    """The scalar typedefs of vm/tilvm.h for a device's build."""
    a = ABIS[dev.abi]
    t = dict(a)
    t['char'] = a['u8']
    t['tv_int'] = {16: a['u16'], 32: a['u32'], 64: a['u64']}[dev.int_bits]
    t['tv_float'] = a['float'] if dev.float == 1 else a['double']
    t['tv_ref'] = t['tv_size'] = a['u16'] if dev.heap <= 65535 else a['u32']
    t['tv_off'] = a['u16'] if dev.addr16 else a['u32']
    t['rng'] = a['u64'] if dev.int_bits == 64 else a['u32']
    return t


_NO_EFFECTS = lambda d: not d.no_effects              # noqa: E731  #ifndef TILVM_NO_EFFECTS
_FUNCS = lambda d: d.funcs_table > 0                   # noqa: E731  #if TILVM_MAX_FUNCS > 0
_CACHE = lambda d: d.cache_entries > 0                 # noqa: E731  #if TILVM_FIELD_CACHE > 0

# vm/tilvm.h, in the order of the header (structs before their users).  'union' members
# overlap; TV_PACK (TILVM_PACK) is not modelled: the 8-bit ABIs have no padding anyway.
VM_STRUCTS: List[Tuple[str, str, List[tuple]]] = [
    ('tv_val.u', 'union', [_f('i', 'tv_int'), _f('f', 'tv_float', when=lambda d: d.float != 0),
                           _f('r', 'tv_ref'), _f('k', 'u16')]),
    ('tv_val', 'struct', [_f('t', 'u8'), _f('u', 'tv_val.u')]),
    ('tv_frame', 'struct', [_f('ret_pc', 'tv_off'), _f('pc', 'tv_off'), _f('func', 'u16'), _f('base', 'u16'),
                            _f('closure', 'tv_ref'), _f('extra', 'u8')]),
    ('tv_try', 'struct', [_f('handler', 'tv_off'), _f('deadline', 'u32'), _f('n_frames', 'u16'),
                          _f('sp', 'u16'), _f('n_handlers', 'u16')]),
    ('tv_handler', 'struct', [_f('abort_pc', 'tv_off'), _f('n_frames', 'u16'), _f('sp', 'u16'),
                              _f('n_try', 'u16'), _f('n', 'u16'), _f('kind', 'u8'), _f('_pad', 'u8', 3)]),
    ('tilvm_state', 'struct', [
        _f('magic', 'u32'),
        _f('sp', 'u16'), _f('n_frames', 'u16'), _f('n_try', 'u16'), _f('n_deadlines', 'u16'),
        _f('heap_top', 'tv_size'),
        _f('halted', 'u8'), _f('exit_code', 'u8'), _f('failed', 'u8'), _f('_pad', 'u8'),
        _f('scr_w', 'u16'), _f('scr_h', 'u16'),
        _f('steps', 'u32'), _f('gc_count', 'u32'), _f('tick', 'u32'),
        _f('rng', 'rng'),
        _f('frames', 'tv_frame', 'frames'),
        _f('tries', 'tv_try', 'try_depth'),
        _f('n_handlers', 'u16'), _f('_hpad', 'u16'),
        _f('handlers', 'tv_handler', 'max_handlers', when=_NO_EFFECTS),
        _f('stack', 'tv_val', 'stack')]),
    ('tv_func', 'struct', [_f('code', 'tv_off'), _f('nparams', 'u8'), _f('nreq', 'u8'), _f('nlocals', 'u8'),
                           _f('flags', 'u8')]),
    ('tv_cache', 'struct', [_f('type', 'u16'), _f('name', 'u16'), _f('value', 'u16')]),
    ('tilvm_host', 'struct', [
        _f('read_file', 'ptr'), _f('write_file', 'ptr'), _f('gfx_init', 'ptr'), _f('gfx_pixel', 'ptr'),
        _f('gfx_get', 'ptr'), _f('gfx_show', 'ptr'), _f('key_pressed', 'ptr'), _f('wait_key', 'ptr'),
        _f('idle', 'ptr'), _f('argc', 'int'), _f('argv', 'ptr')]),
    ('tilvm.mem', 'union', [_f('heap', 'u8', 'heap'), _f('align_v_', 'tv_val'),
                            _f('align_u_', 'u64', when=lambda d: d.float == 2 or d.int_bits == 64)]),
    ('tilvm', 'struct', [
        # host bindings
        _f('write', 'ptr'), _f('write_err', 'ptr'), _f('read_line', 'ptr'), _f('clock_ms', 'ptr'),
        _f('ctx', 'ptr'), _f('max_steps', 'u32'), _f('host', 'tilvm_host'),
        # program
        _f('bc', 'ptr'), _f('bc_size', 'u32'),
        _f('code_off', 'tv_off'), _f('code_size', 'tv_off'),
        _f('funcs_off', 'tv_off'), _f('methods_off', 'tv_off'), _f('prelude_off', 'tv_off'),
        _f('lines_off', 'tv_off'),
        _f('n_consts', 'u16'), _f('n_funcs', 'u16'), _f('n_types', 'u16'), _f('n_methods', 'u16'),
        _f('n_globals', 'u16'), _f('entry', 'u16'), _f('n_lines', 'u16'),
        _f('n_prelude', 'u8'), _f('flags', 'u8'),
        _f('const_off', 'tv_off', 'consts'), _f('type_off', 'tv_off', 'types'), _f('bc_hash', 'u32'),
        _f('funcs', 'tv_func', 'funcs_table', when=_FUNCS),
        _f('field_cache', 'tv_cache', 'cache_entries', when=_CACHE),
        _f('method_cache', 'tv_cache', 'cache_entries', when=_CACHE),
        # runtime
        _f('err_msg', 'ptr'), _f('err_val', 'tv_val'), _f('errbuf', 'char', 128),
        _f('mark_sp', 'u16'), _f('mark_overflow', 'u8'), _f('mark_stack', 'tv_ref', 'mark_stack'),
        _f('st', 'tilvm_state'), _f('mem', 'tilvm.mem')]),
        # (tilvm_stats follows with TILVM_STATS: not part of the device builds)
]
# the parts of the VM struct that the report lists (member path, label)
VM_PARTS = (('mem', "heap"), ('st.stack', "value stack"), ('funcs', "function table"),
            ('const_off', "constant table"), ('st.frames', "call frames"), ('st.handlers', "effect handlers"),
            ('field_cache', "lookup caches"), ('method_cache', "lookup caches"))


def vm_struct_layout(dev: Device) -> Dict[str, Any]:
    """sizeof() of the C VM structs for this device (mirror of vm/tilvm.h): one entry per
    struct name, plus 'stack_bytes', 'heap' and 'parts' (bytes of the big members)."""
    types = _typedefs(dev)
    member_sizes: Dict[str, int] = {}
    for sname, kind, members in VM_STRUCTS:
        laid = []
        for mname, ctype, count, when in members:
            if when is not None and not when(dev):
                continue
            n = count(dev) if callable(count) else getattr(dev, count) if isinstance(count, str) else count
            t = _array(types[ctype], n)
            laid.append(t)
            member_sizes[f"{sname}.{mname}"] = t[0]
        types[sname] = _struct(*laid) if kind == 'struct' else _union(*laid)
    out: Dict[str, Any] = {name: types[name][0] for name, _k, _m in VM_STRUCTS if '.' not in name}
    out['stack_bytes'] = out['tv_val'] * dev.stack
    out['heap'] = dev.heap
    parts: Dict[str, int] = {}
    for path, label in VM_PARTS:
        key = 'tilvm.' + path if not path.startswith('st.') else 'tilvm_state.' + path[3:]
        if member_sizes.get(key):
            parts[label] = parts.get(label, 0) + member_sizes[key]
    out['parts'] = parts
    return out


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
# Operand layouts come from til.OP_OPERANDS (the compiler's table).  Adding a
# plain opcode = one line in STACK_EFFECTS; adding a superinstruction = one line
# in SUPERINSTRUCTIONS (the sequence it stands for).  An opcode without an entry
# makes the analysis of its function "incomplete" (reported, never guessed).

# control flow after an instruction
NEXT, JUMP, BRANCH, ITER, DEFAULT, TRY, STOP, HANDLE = range(8)
# what an instruction does besides moving values
K_CALL, K_CALL_VALUE, K_INVOKE, K_NATIVE, K_FUNC_REF, K_PERFORM = range(1, 7)
CALL_KINDS = (K_CALL, K_CALL_VALUE, K_INVOKE, K_PERFORM)
NO_DEFAULT = 0xFFFF            # PERFORM: no default function


class Effect:
    """pops/pushes: operand-stack values consumed/produced (int or f(operands)).
    peak/need: transient slots used / checked above the depth *before* the
    instruction (C VM internals, e.g. make_array keeps the new buffer on the
    stack).  flow: NEXT JUMP BRANCH ITER (FOR_ITER: +1 when it continues, jumps
    without) DEFAULT (ARG_DEFAULT) TRY (handler entered with the stack of the TRY
    + the error message) STOP HANDLE (HANDLE n: the n clause pairs stay on the
    stack; its target - where an aborting clause continues - gets the stack below
    the pairs + the clause value).  land: a superinstruction that may jump lands
    with the stack at depth + land (til._JUMP_LANDING).  tryd: change of the
    recovery-point depth.  ends: END_HANDLE (drops the pairs of the innermost
    handle: its pops depend on the handle)."""
    __slots__ = ('pops', 'pushes', 'flow', 'peak', 'need', 'kind', 'alloc', 'tryd', 'land', 'ends')

    def __init__(self, pops=0, pushes=0, flow=NEXT, peak=0, need=0, kind=0, alloc=None, tryd=0, land=None,
                 ends=False):
        self.pops, self.pushes, self.flow = pops, pushes, flow
        self.peak, self.need, self.kind, self.alloc, self.tryd = peak, need, kind, alloc, tryd
        self.land, self.ends = land, ends

    def jumps(self) -> bool:
        """The instruction has a jump target (its last operand)."""
        return self.flow in (JUMP, BRANCH, ITER, DEFAULT, TRY, HANDLE) or self.land is not None


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
    # effects (docs/TIL_EFFECTS.md): the clause pairs stay on the stack under the body
    _O.HANDLE: E(flow=HANDLE),                                 # b n, s rel_end
    _O.END_HANDLE: E(ends=True),                               # [pairs(2n) v] -> [v]
    _O.PERFORM: E(_arg(2), 1, kind=K_PERFORM),                 # h name, h default, b argc: a call
    _O.RESUME: E(1, 0, flow=STOP),                             # returns TOS to the perform site
    _O.HALT: E(flow=STOP),
}
del E


def _cmp(c: int) -> int:
    if c not in til.CMP_OPS:
        raise ValueError(f"bad comparison operand 0x{c:02X}")
    return c


# Superinstructions (docs/TIL_BYTECODE_V2.md): the sequence of plain instructions each
# one stands for, and whether the C VM writes the sequence's values to the stack.  The
# VM's slow paths (all of them in -Os and TILVM_LEAN builds, which have no integer fast
# paths; `til fit --run` builds the same way) push the operands and run the plain
# handlers, so they use exactly the stack of the sequence; the compare-and-branch
# forms, RETURN_LOCAL and FOR_ITER_STORE never push their operands.  Net effect and
# jump landing must agree with the compiler's tables (til._STACK_EFFECT,
# til._JUMP_LANDING, til._ENDS_FLOW; tests/test_fit.py checks it).
_L, _I = _O.LOAD_LOCAL, _O.PUSH_I16
SUPERINSTRUCTIONS: Dict[int, Tuple[Callable[[Tuple[int, ...]], List[Tuple[int, Tuple[int, ...]]]], bool]] = {
    _O.LOAD_LOCAL2: (lambda a: [(_L, (a[0],)), (_L, (a[1],))], True),
    _O.LOAD_LOCAL_FIELD: (lambda a: [(_L, (a[0],)), (_O.GET_FIELD, (a[1],))], True),
    _O.INC_LOCAL: (lambda a: [(_L, (a[0],)), (_O.PUSH_I8, (a[1],)), (_O.ADD, ()), (_O.STORE_LOCAL, (a[0],))], True),
    _O.CAST_LOCAL: (lambda a: [(_L, (a[0],)), (_O.CAST, (a[1],)), (_O.STORE_LOCAL, (a[0],))], True),
    _O.CAST_RETURN: (lambda a: [(_O.CAST, (a[0],)), (_O.RETURN, ())], True),
    _O.ADD_STORE_LOCAL: (lambda a: [(_O.ADD, ()), (_O.STORE_LOCAL, (a[0],))], True),
    _O.NATIVE_POP: (lambda a: [(_O.NATIVE, (a[0], a[1])), (_O.POP, ())], True),
    _O.INDEX_L: (lambda a: [(_L, (a[0],)), (_O.GET_INDEX, ())], True),
    _O.INDEX_LL: (lambda a: [(_L, (a[0],)), (_L, (a[1],)), (_O.GET_INDEX, ())], True),
    _O.RETURN_LOCAL: (lambda a: [(_L, (a[0],)), (_O.RETURN, ())], False),
    _O.FOR_ITER_STORE: (lambda a: [(_O.FOR_ITER, (a[0], a[2])), (_O.STORE_LOCAL, (a[1],))], False),
    _O.JUMP_IF_NOT: (lambda a: [(_cmp(a[0]), ()), (_O.JUMP_IF_FALSE, (a[1],))], False),
    _O.JUMP_IF_NOT_L: (lambda a: [(_L, (a[1],)), (_cmp(a[0]), ()), (_O.JUMP_IF_FALSE, (a[2],))], False),
    _O.JUMP_IF_NOT_I: (lambda a: [(_I, (a[1],)), (_cmp(a[0]), ()), (_O.JUMP_IF_FALSE, (a[2],))], False),
    _O.JUMP_IF_NOT_LL: (lambda a: [(_L, (a[1],)), (_L, (a[2],)), (_cmp(a[0]), ()),
                                   (_O.JUMP_IF_FALSE, (a[3],))], False),
    _O.JUMP_IF_NOT_LI: (lambda a: [(_L, (a[1],)), (_I, (a[2],)), (_cmp(a[0]), ()),
                                   (_O.JUMP_IF_FALSE, (a[3],))], False),
}
for _k, _base in enumerate(til.FUSED_BINOPS):
    SUPERINSTRUCTIONS[0xA0 + _k] = (lambda a, b=_base: [(_I, (a[0],)), (b, ())], True)    # <op>_I k
    SUPERINSTRUCTIONS[0xB0 + _k] = (lambda a, b=_base: [(_L, (a[0],)), (b, ())], True)    # <op>_L b
del _k, _base


def expand(op: int, args: Tuple[int, ...]) -> Optional[List[Tuple[int, Tuple[int, ...]]]]:
    """The plain instructions an instruction stands for: itself, or a superinstruction's
    sequence (None when its operands are malformed, e.g. a bad comparison)."""
    s = SUPERINSTRUCTIONS.get(op)
    if s is None:
        return [(op, args)]
    try:
        seq = s[0](args)
    except (ValueError, IndexError):
        return None
    return seq if all(mop in STACK_EFFECTS for mop, _a in seq) else None


def _seq_effect(seq: List[Tuple[int, Tuple[int, ...]]]) -> Tuple[int, int]:
    """(pops, pushes) of a straight sequence of plain instructions."""
    d = low = 0
    for op, args in seq:
        e = STACK_EFFECTS[op]
        d -= _value(e.pops, args)
        low = min(low, d)
        d += _value(e.pushes, args)
    return -low, d - low


def _super_effect(op: int) -> Effect:
    """The Effect of a superinstruction: pops/pushes of its sequence, the jump landing
    and end of flow from the compiler's tables."""
    def part(k):
        def f(args):
            seq = expand(op, args)
            return _seq_effect(seq)[k] if seq is not None else 0
        return f
    land = getattr(til, '_JUMP_LANDING', {}).get(op)
    ends = op in getattr(til, '_ENDS_FLOW', ())
    return Effect(part(0), part(1), flow=STOP if ends else NEXT, land=land)


for _op in SUPERINSTRUCTIONS:
    STACK_EFFECTS[_op] = _super_effect(_op)
del _op

# Transient stack use of builtins in the C VM (slots written / checked above the
# depth before the instruction).  Builtins that build an array or a map create it
# empty above their arguments (push_empty: 1 slot written, 2 checked) and fill it in
# the heap; only args() puts a value per program argument on the stack.  Keys: native
# or method name (a method: the most of its receivers).
NATIVE_TRANSIENT = {'sum': (2, 3), 'str_bytes': (1, 2), 'args': (1, 2)}
METHOD_TRANSIENT = {'sum': (2, 3), 'split': (2, 4), 'copy': (1, 2), 'slice': (1, 2), 'keys': (1, 2),
                    'values': (1, 2)}
DATA_DEPENDENT_METHODS: Dict[str, str] = {}      # none since TIL 3.1 (split/join/keys/... build in the heap)
ALLOC_NATIVES = {'str': 'string', 'format': 'string', 'input': 'string', 'chr': 'string',
                 'type_of': 'string', 'range': 'array', 'read_file': 'string', 'args': 'array',
                 'bytes_to_str': 'string', 'str_bytes': 'array'}
ALLOC_METHODS = {'push': 'array push', 'insert': 'array insert', 'set': 'map insert',
                 'split': 'strings', 'replace': 'string', 'trim': 'string', 'repeat': 'string',
                 'to_upper': 'string', 'to_lower': 'string', 'slice': 'slice', 'copy': 'copy',
                 'keys': 'array', 'values': 'array', 'join': 'string', 'to_string': 'string'}
GROWING_ALLOCS = ('array push', 'array insert', 'map insert', 'string built up with +',
                  'array built up with +', 'string built up with *')

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
    'str': ('count', 'trim', 'repeat', 'replace', 'split', 'to_float'),
    'array': ('remove', 'insert', 'reverse', 'sort', 'sum', 'min', 'max', 'copy', 'slice', 'join'),
    'map': ('values', 'copy', 'clear'),
}
FLOAT_NATIVES = ('sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'floor', 'ceil', 'round', 'pow')
MATH_NATIVES = ('sin', 'cos', 'tan', 'log', 'exp')
SCREEN_NATIVES = tuple(til.NATIVES[til.NATIVE_ID['screen_init']:til.NATIVE_ID['screen_height'] + 1])
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
FLAG_EFFECT_CLAUSE = getattr(til.TBCFunction, 'FLAG_EFFECT_CLAUSE', 16)


class CallSite:
    __slots__ = ('pc', 'op', 'args', 'depth', 'tries', 'hands', 'open_handles', 'targets', 'via', 'edges',
                 'callback', 'fn_arg')

    def __init__(self, pc, op, args, depth, tries, hands):
        self.pc, self.op, self.args, self.depth, self.tries, self.hands = pc, op, args, depth, tries, hands
        self.open_handles: Tuple[int, ...] = ()      # pcs of the HANDLEs of this function open here
        self.targets: List[Tuple[int, int]] = []     # (callee, values consumed by its frame)
        self.via: List[Tuple[int, int, int]] = []    # (callback, prelude function calling it, its args)
        self.edges: List['Edge'] = []                # every frame this call can push (Analysis._edges)
        self.callback = False      # the call of the function argument inside .map()/.filter()/...
        self.fn_arg: Optional[int] = None   # the function passed as .map(f)'s f, when the code shows it


class Edge:
    """A call in the call graph: `callee`'s frame starts `rel` value-stack slots above the
    caller's frame base, `frames` frames, `tries` recovery points and `hands` effect
    handler entries deeper.  A call of .map()/.filter()/... with a function argument gives
    two edges: to the prelude function, and through it (`via`) to the function argument
    (2 frames deeper).  A perform that runs an effect clause adds the clause's entry."""
    __slots__ = ('callee', 'rel', 'frames', 'tries', 'hands', 'via', 'site')

    def __init__(self, callee: int, rel: int, frames: int, tries: int, hands: int, via: Optional[int],
                 site: CallSite):
        self.callee, self.rel, self.frames, self.tries, self.hands = callee, rel, frames, tries, hands
        self.via, self.site = via, site

    @property
    def vec(self) -> Tuple[int, int, int, int]:
        return self.rel, self.frames, self.tries, self.hands


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
        self.max_hand = 0        # effect handler entries opened inside this function (handle blocks)
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
        self.ops: Set[int] = set()                              # opcodes, superinstructions expanded
        self.screen_sizes: List[Tuple[int, Optional[int], Optional[int]]] = []   # screen_init (pc, w, h)
        # effects: HANDLE sites (pc, [(effect name or None, clause function or None)]), the
        # operand-stack depth below the pairs of the handle an END_HANDLE closes
        self.handles: List[Tuple[int, List[Tuple[Optional[str], Optional[int]]]]] = []
        self.handle_base: Dict[int, int] = {}
        self.reachable = 0
        # the abstract interpretation's results per reachable instruction
        self.code: Dict[int, Tuple[int, Tuple[int, ...], int]] = {}   # pc -> (op, operands, next pc)
        self.depth_at: Dict[int, int] = {}                      # operand-stack depth before it
        self.succ: Dict[int, List[int]] = {}                    # successors inside the function
        self.cost: Dict[int, Tuple[int, int, int, int]] = {}    # (slots written, needed, tries, handlers)
        self.prev_of: Dict[int, int] = {}                       # straight-line predecessor
        self.jump_targets: Set[int] = set()
        self.map_iters: Dict[int, int] = {}      # pc of NATIVE iter that may get a map -> depth after

    @property
    def is_clause(self) -> bool:
        return bool(self.flags & FLAG_EFFECT_CLAUSE)

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


Handles = Tuple[Tuple[int, int], ...]
State = Tuple[int, int, Handles]     # (operand-stack depth, recovery points, open handle blocks)


def _transfer(pc: int, op: int, eff: Effect, args, nxt: int, d: int, t: int, hs: Handles):
    """One instruction from the state (d, t, hs): (pops, depth after, successors as
    (pc, d, t, hs)) or a problem text.  hs: per open handle block of this function, the
    operand-stack depth below its clause pairs (END_HANDLE cuts the stack back to it) and
    the pc of its HANDLE."""
    name = til.OP_NAMES.get(op, f"0x{op:02X}")
    if eff.ends:                                   # END_HANDLE: [pairs v] -> [v]
        if not hs:
            return "END_HANDLE without an open handle block"
        base = hs[-1][0]
        if d < base + 1:
            return f"END_HANDLE with {d - base} values above the handle's pairs"
        pops, d2, hs2 = d - base, base + 1, hs[:-1]
    else:
        pops = _value(eff.pops, args)
        if d < pops:
            return f"{name} pops {pops} value(s) from an operand stack of {d}"
        d2, hs2 = d - pops + _value(eff.pushes, args), hs
    t2 = max(t + eff.tryd, 0)
    flow = eff.flow
    if flow == HANDLE:
        n = args[0]
        if d < 2 * n:
            return f"HANDLE {n} needs {2 * n} clause values on an operand stack of {d}"
        base = d - 2 * n
        return pops, d2, [(nxt, d2, t2, hs + ((base, pc),)), (_flow_target(args, nxt), base + 1, t2, hs)]
    succ: List[Tuple[int, int, int, Tuple[int, ...]]] = []
    if flow in (NEXT, BRANCH, DEFAULT, ITER):
        succ.append((nxt, d2, t2, hs2))
    elif flow == TRY:
        succ.append((nxt, d2, t2 + 1, hs2))
    if flow in (JUMP, BRANCH, DEFAULT):
        succ.append((_flow_target(args, nxt), d2, t2, hs2))
    elif flow == ITER:
        succ.append((_flow_target(args, nxt), d, t2, hs2))
    elif flow == TRY:
        succ.append((_flow_target(args, nxt), d2 + 1, t2, hs2))
    if eff.land is not None:                       # a superinstruction that may jump
        succ.append((_flow_target(args, nxt), d + eff.land, t2, hs2))
    return pops, d2, succ


def analyze_function(an: 'Analysis', fidx: int) -> FuncFacts:
    m, code = an.module, an.module.code
    start, end = an.ranges[fidx]
    ff = FuncFacts(fidx, m.funcs[fidx], start, end)
    if start >= end:
        ff.fail(start, "the function has no code")
        return ff
    states: Dict[int, State] = {}
    decoded: Dict[int, Tuple[int, Tuple[int, ...], int]] = {}
    micro: Dict[int, List[Tuple[int, Tuple[int, ...]]]] = {}
    succs: Dict[int, List[int]] = {}
    visits: Dict[int, int] = {}
    work: List[Tuple[int, int, int, Handles]] = [(start, 0, 0, ())]
    while work:
        pc, d, t, hs = work.pop()
        old = states.get(pc)
        if old is not None:
            if [h[1] for h in old[2]] != [h[1] for h in hs]:
                ff.fail(pc, "handle blocks are nested differently on two paths to this instruction")
                continue
            if d <= old[0] and t <= old[1] and all(a[0] <= b[0] for a, b in zip(hs, old[2])):
                continue
            n = visits.get(pc, 0) + 1
            visits[pc] = n
            if n > MAX_REVISITS:
                ff.fail(pc, "the operand stack grows on every pass through a loop")
                continue
            d, t = max(d, old[0]), max(t, old[1])
            hs = tuple((max(a[0], b[0]), a[1]) for a, b in zip(hs, old[2]))
        states[pc] = (d, t, hs)
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
        if pc not in micro:
            seq = expand(op, args)
            if seq is None:
                ff.fail(pc, f"{til.OP_NAMES[op]} has a malformed operand (the VMs stop with \"bad opcode\")")
                continue
            micro[pc] = seq
        res = _transfer(pc, op, eff, args, nxt, d, t, hs)
        if isinstance(res, str):
            ff.fail(pc, res)
            continue
        succ = res[2]
        succs[pc] = [s[0] for s in succ if start <= s[0] < end]
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
        if eff is not None and eff.jumps():
            targets.add(_flow_target(args, nxt))
    has_maps = an.has_maps
    ff.prev_of, ff.jump_targets = prev_of, targets
    prelude = fidx in an.prelude_functions
    for pc in sorted(decoded):
        op, args, nxt = decoded[pc]
        eff = STACK_EFFECTS.get(op)
        if eff is None or (pc not in states and pc not in swept):
            continue
        seq = micro.get(pc) or expand(op, args)
        if seq is None:
            continue
        if pc in states:
            d, t, hs = states[pc]
            res = _transfer(pc, op, eff, args, nxt, d, t, hs)
            if isinstance(res, str):
                continue
            pops, d2 = res[0], res[1]
        else:                         # depth unknown: the facts below are still collected
            if eff.ends:
                continue
            t, hs = 0, ()
            pops = _value(eff.pops, args)
            d = pops
            d2 = d - pops + _value(eff.pushes, args)
        ff.reachable += 1
        ff.ops.add(op)
        occ = need = max(d, d2)
        writes = SUPERINSTRUCTIONS.get(op, (None, True))[1]
        md = d
        for mop, margs in seq:                   # the plain instructions it stands for
            meff = STACK_EFFECTS[mop]
            ff.ops.add(mop)
            md2 = md if eff.ends else md - _value(meff.pops, margs) + _value(meff.pushes, margs)
            peak, mneed = meff.peak, meff.need
            if mop == _O.NATIVE:
                peak, mneed = _native_facts(an, ff, decoded, prev_of, targets, pc, margs, md, md2, pc in states)
            elif mop == _O.PUSH_CONST:
                ff.const_uses.setdefault(margs[0], []).append(pc)
            elif mop == _O.CAST and margs[0] == til.CAST_KINDS['float']:
                ff.float_casts.append(pc)
            elif mop == _O.TRY_TIMEOUT:
                ff.timeouts.append(pc)
            elif mop == _O.SET_INDEX and has_maps:
                ff.allocs.append((pc, 'map insert'))
            if meff.alloc:
                ff.allocs.append((pc, meff.alloc))
            if mop == _O.INVOKE:
                name = an.const_text(margs[0])
                ff.methods.append((pc, name, margs[1], margs[2]))
                if margs[1] != til.NO_METHOD:
                    peak, mneed = METHOD_TRANSIENT.get(name, (0, 0))
                    if name in DATA_DEPENDENT_METHODS:
                        ff.data_dependent.append((pc, f".{name}() {DATA_DEPENDENT_METHODS[name]}"))
                    if name in ALLOC_METHODS and (name != 'set' or has_maps):
                        ff.allocs.append((pc, ALLOC_METHODS[name]))
            if writes:
                occ = max(occ, md, md2, md + peak)
                need = max(need, md + mneed)
            md = md2
        need = max(need, occ)
        if eff.kind in CALL_KINDS:
            site = CallSite(pc, op, args, d, t, len(hs))
            site.open_handles = tuple(h[1] for h in hs)
            if pc in states:
                if op == _O.INVOKE and args[2] >= 1:     # xs.map(f) / xs.reduce(f, init): f is the 1st argument
                    site.fn_arg = _function_pushed(decoded, prev_of, targets, pc, args[2] - 1)
                elif op == _O.CALL and args[1] >= 2:     # __prelude_map(xs, f) called directly
                    site.fn_arg = _function_pushed(decoded, prev_of, targets, pc, args[1] - 2)
                elif op == _O.CALL_VALUE and prelude:
                    # __prelude_map(a, f) & co. call their parameter f: the callers' edges cover it
                    q = _producer(decoded, prev_of, targets, pc, args[0])
                    site.callback = q is not None and q[1] == _O.LOAD_LOCAL and q[2][0] == 1
            ff.calls.append(site)
        elif eff.kind == K_FUNC_REF:
            ff.func_refs.add(args[0])
        if eff.flow == HANDLE and pc in states:
            ff.handles.append((pc, _clause_pairs(an, decoded, prev_of, targets, pc, args[0])))
        if eff.ends and pc in states:
            ff.handle_base[pc] = hs[-1][0]
        tries = t + (1 if eff.flow == TRY else 0)
        hands = len(hs) + (1 if eff.flow == HANDLE else 0)
        ff.max_occ = max(ff.max_occ, occ)
        ff.max_need = max(ff.max_need, need)
        ff.max_try = max(ff.max_try, tries)
        ff.max_hand = max(ff.max_hand, hands)
        if pc in states:
            ff.code[pc] = (op, args, nxt)
            ff.depth_at[pc] = d
            ff.cost[pc] = (occ, need, tries, hands)
            ff.succ[pc] = succs.get(pc, [])
        if eff.flow in (JUMP, BRANCH) or eff.land is not None:
            target = _flow_target(args, nxt)
            if target <= pc:
                ff.loops.append((target, pc))
    return ff


def _native_facts(an: 'Analysis', ff: FuncFacts, decoded, prev_of, targets, pc: int, margs, d: int, d2: int,
                  known: bool) -> Tuple[int, int]:
    """Record a NATIVE call (at pc, the operand-stack depth d before it, d2 after it);
    returns its transient (peak, need) above d."""
    name = til.NATIVES[margs[0]] if margs[0] < len(til.NATIVES) else f"native#{margs[0]}"
    argc = margs[1]
    ff.natives.append((pc, name, argc))
    peak, need = NATIVE_TRANSIENT.get(name, (0, 0))
    if name == 'args':        # one string per program argument, then the array
        peak, need = peak + an.n_args, need + an.n_args
    if name == 'range':                 # the new array replaces the arguments
        peak, need = 1 - argc, 2 - argc
    elif name == 'iter' and an.has_maps:    # a loop over a map iterates over a new array of the keys
        peak, need = 1, 2
        ff.map_iters[pc] = max(d, d2)
    elif name == 'format':
        ff.formats.append((pc, _const_string(an, decoded, prev_of, targets, pc) if known else None))
    elif name == 'screen_init' and argc == 2:
        w = _push_value(an, _producer(decoded, prev_of, targets, pc, 1)) if known else None
        h = _push_value(an, _producer(decoded, prev_of, targets, pc, 0)) if known else None
        ff.screen_sizes.append((pc, w if isinstance(w, int) and not isinstance(w, bool) else None,
                                h if isinstance(h, int) and not isinstance(h, bool) else None))
    if name in ALLOC_NATIVES:
        ff.allocs.append((pc, ALLOC_NATIVES[name]))
    return peak, need


def _clause_pairs(an: 'Analysis', decoded, prev_of, targets, pc: int, n: int) -> List[Tuple[Optional[str], Optional[int]]]:
    """(effect name, clause function) of each pair below HANDLE n: the compiler pushes
    the clause (PUSH_FUNC / MAKE_CLOSURE) and then its name (PUSH_CONST)."""
    pairs: List[Tuple[Optional[str], Optional[int]]] = []
    for j in range(n):
        pos = 2 * (n - 1 - j)
        name_q = _producer(decoded, prev_of, targets, pc, pos)
        fn_q = _producer(decoded, prev_of, targets, pc, pos + 1)
        name = an.const_text(name_q[2][0]) if name_q is not None and name_q[1] == _O.PUSH_CONST else None
        fn = fn_q[2][0] if fn_q is not None and fn_q[1] in (_O.PUSH_FUNC, _O.MAKE_CLOSURE) else None
        pairs.append((name, fn if fn is not None and 0 <= fn < len(an.module.funcs) else None))
    return pairs


def _function_pushed(decoded, prev_of, targets, pc: int, pos: int) -> Optional[int]:
    """The function a PUSH_FUNC / MAKE_CLOSURE put `pos` slots below the top before `pc`."""
    q = _producer(decoded, prev_of, targets, pc, pos)
    if q is not None and q[1] in (_O.PUSH_FUNC, _O.MAKE_CLOSURE):
        return q[2][0]
    return None


def _producer(decoded, prev_of, targets, pc: int, pos: int,
              upto: Optional[int] = None) -> Optional[Tuple[int, int, Tuple[int, ...]]]:
    """(pc, op, operands) of the plain instruction - or the part of a superinstruction -
    that pushed the value `pos` slots below the top of the operand stack just before
    `pc` (before the `upto`-th instruction of the sequence of the superinstruction at
    pc, when given), found by walking straight-line code backwards (None when a jump
    target or the start of the function is crossed: several producers possible)."""
    p = pc
    if upto:
        op, args, _nxt = decoded[pc]
        seq = expand(op, args) or []
        for mop, margs in reversed(seq[:upto]):
            meff = STACK_EFFECTS[mop]
            pushes = _value(meff.pushes, margs)
            if pos < pushes:
                return pc, mop, margs
            pos += _value(meff.pops, margs) - pushes
    for _ in range(256):
        if p in targets:
            return None
        q = prev_of.get(p)
        if q is None or q not in decoded:
            return None
        op, args, _nxt = decoded[q]
        eff = STACK_EFFECTS.get(op)
        seq = expand(op, args) if eff is not None else None
        if eff is None or seq is None or eff.flow in (JUMP, STOP) or eff.ends:
            return None
        for mop, margs in reversed(seq):
            meff = STACK_EFFECTS[mop]
            pushes = _value(meff.pushes, margs)
            if pos < pushes:
                return q, mop, margs
            pos += _value(meff.pops, margs) - pushes
        p = q
    return None


def _push_value(an: 'Analysis', ins) -> Any:
    """The constant a push instruction ((pc, op, operands) of _producer) pushes (None when
    it is not a constant push)."""
    if ins is None:
        return None
    _pc, op, args = ins
    if op in (_O.PUSH_I8, _O.PUSH_I16):
        return args[0]
    if op == _O.PUSH_TRUE:
        return True
    if op == _O.PUSH_FALSE:
        return False
    if op == _O.PUSH_CONST and args[0] < len(an.module.consts):
        return an.module.consts[args[0]]
    return None


def _const_string(an, decoded, prev_of, targets, pc) -> Optional[str]:
    v = _push_value(an, _producer(decoded, prev_of, targets, pc, 0))
    return v.decode('utf-8', 'replace') if isinstance(v, bytes) else None


# ═══════════════════════════════════════════════════════════════════════════════
#                 ANALYSIS OF THE WHOLE PROGRAM (call graph, worst cases)
# ═══════════════════════════════════════════════════════════════════════════════
# Costs along the call graph are vectors (value-stack slots, call frames, recovery
# points, effect handler entries); each component is summed along a path and
# maximised on its own.

MAX_PATH_STEPS = 20000   # simple paths enumerated per recursion before falling back to per-call costs
Vec = Tuple[int, int, int, int]
ZERO: Vec = (0, 0, 0, 0)


def _vadd(a: Vec, b: Vec) -> Vec:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]


def _vmax(a: Optional[Vec], b: Vec) -> Vec:
    return b if a is None else (max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


class Recursion:
    """A recursive part of the call graph (a strongly connected component) and what one
    more level of it costs.

    A *level* is one way around the recursion back to the function it is entered by:
    count -> count, even -> odd -> even, depth -> .map() -> <lambda> -> depth.  Its cost
    is the most expensive simple cycle (slots, frames, recovery points and effect handler
    entries each maximised on their own), so the number of levels computed from it always
    fits.  When a cycle avoids the entry function (nested recursion), `unit` is 'calls':
    every call inside the recursion is counted with the most expensive one."""

    def __init__(self, members: List[int], intra: List[Tuple[int, Edge]]):
        self.members = members
        self.intra = intra                # (caller, call) inside the recursion
        self.unit = 'levels'
        self.slots_per_level = 0          # value-stack slots added by one more level
        self.frames_per_level = 1         # call frames added by one more level
        self.tries_per_level = 0          # recovery points added by one more level
        self.hands_per_level = 0          # effect handler entries added by one more level
        self.cycle: List[Tuple[int, Edge]] = []          # the most expensive level (for the report)
        caller, best = max(intra, key=lambda x: (x[1].rel, x[1].frames))
        self.call_pc: Optional[Tuple[int, int]] = (caller, best.site.pc)   # a recursive call
        # member as the topmost frame: (slots written, slots needed, frames, tries, handlers)
        self.tops: Dict[int, Tuple[int, int, int, int, int]] = {}
        self.entries: List[Dict[str, int]] = []   # one per function the recursion is entered by
        # the entry with the deepest start (kept as attributes for the report and the tests)
        self.base = 0                # value-stack slots in use below its first frame
        self.frames_below = 1        # call depth of its first frame
        self.tries_below = 0
        self.hands_below = 0
        self.top_need = 0            # slots the topmost (partial) level needs above the base
        self.top_occ = 0             # ... and writes
        self.top_frames = 1
        self.top_tries = 0
        self.top_hands = 0

    @property
    def per_level(self) -> Vec:
        return self.slots_per_level, self.frames_per_level, self.tries_per_level, self.hands_per_level

    def measure(self, entries: List[Tuple[int, Vec]]) -> Dict[int, Vec]:
        """Per-level costs and, for every (function, position) the recursion is entered by
        (position = value-stack base, frame depth, recovery points, handler entries), what
        its topmost partial level needs.  Returns the deepest position of each member
        within one level."""
        out: Dict[int, List[Edge]] = {f: [] for f in self.members}
        for f, e in self.intra:
            out[f].append(e)
        paths: Dict[int, Any] = {}
        if all(_acyclic_without(self.members, out, f) for f, _pos in entries):
            for f, _pos in entries:
                p = _simple_paths(f, out)
                if p is None:
                    break
                paths[f] = p
        inner: Dict[int, Vec] = {}

        def deeper(f, pos):
            inner[f] = _vmax(inner.get(f), pos)

        def entry(f, pos, top):
            b, d, t, h = pos
            self.entries.append(dict(function=f, base=b, frames_below=d, tries_below=t, hands_below=h,
                                     top_occ=top[0], top_need=top[1], top_frames=top[2], top_tries=top[3],
                                     top_hands=top[4]))

        if len(paths) == len(entries):
            self.unit = 'levels'
            cyc: Optional[Vec] = None
            best = -1
            for f, (_reach, c, edges) in paths.items():
                if c[0] > best:
                    best, self.cycle = c[0], edges
                cyc = _vmax(cyc, c)
            assert cyc is not None
            self.slots_per_level, self.frames_per_level, self.tries_per_level, self.hands_per_level = cyc
            for f, pos in entries:
                reach = paths[f][0]
                entry(f, pos, self._top(reach))
                for x, p in reach.items():
                    deeper(x, _vadd(pos, p))
        else:
            self.unit = 'calls'
            per = ZERO
            for _f, e in self.intra:
                per = _vmax(per, e.vec)
            self.slots_per_level, self.frames_per_level, self.tries_per_level, self.hands_per_level = per
            top = tuple(max(self.tops[x][k] for x in self.members) for k in range(5))
            for f, pos in entries:
                entry(f, pos, top)
                for x in self.members:
                    deeper(x, pos if x == f else _vadd(pos, per))
        worst = max(self.entries, key=lambda en: (en['base'] + en['top_need'], en['frames_below']))
        for key in ('base', 'frames_below', 'tries_below', 'hands_below', 'top_need', 'top_occ', 'top_frames',
                    'top_tries', 'top_hands'):
            setattr(self, key, worst[key])
        return inner

    def _top(self, reach: Dict[int, Vec]) -> Tuple[int, int, int, int, int]:
        """The most expensive partial level: a simple path to a member plus what that member
        runs as the topmost frame."""
        return (max(reach[x][0] + self.tops[x][0] for x in reach),
                max(reach[x][0] + self.tops[x][1] for x in reach),
                max(reach[x][1] + self.tops[x][2] for x in reach),
                max(reach[x][2] + self.tops[x][3] for x in reach),
                max(reach[x][3] + self.tops[x][4] for x in reach))


def _acyclic_without(members: List[int], out: Dict[int, List[Edge]], removed: int) -> bool:
    """True when every cycle of the recursion passes through `removed`."""
    color = {f: 0 for f in members if f != removed}       # 0 new, 1 on the path, 2 done
    for root in color:
        if color[root]:
            continue
        color[root] = 1
        stack = [(root, iter([e.callee for e in out[root]]))]
        while stack:
            v, it = stack[-1]
            w = next(it, None)
            if w is None:
                color[v] = 2
                stack.pop()
            elif w != removed and color[w] == 1:
                return False
            elif w != removed and color[w] == 0:
                color[w] = 1
                stack.append((w, iter([e.callee for e in out[w]])))
    return True


def _simple_paths(start: int, out: Dict[int, List[Edge]]):
    """Simple paths from `start` inside a recursion: per member the most expensive way to
    reach it, and the most expensive cycle back to `start` (each cost component maximised
    on its own) with the edges of the most slot-expensive one.  None when there are more
    than MAX_PATH_STEPS paths."""
    reach: Dict[int, Vec] = {start: ZERO}
    cyc: Optional[Vec] = None
    cyc_edges: List[Tuple[int, Edge]] = []
    best_slots = -1
    steps = 0
    stack: List[Tuple[int, Vec, Set[int], List[Tuple[int, Edge]]]] = [(start, ZERO, {start}, [])]
    while stack:
        v, cost, on_path, edges = stack.pop()
        for e in out[v]:
            steps += 1
            if steps > MAX_PATH_STEPS:
                return None
            c = _vadd(cost, e.vec)
            if e.callee == start:
                cyc = _vmax(cyc, c)
                if c[0] > best_slots:
                    best_slots, cyc_edges = c[0], edges + [(v, e)]
                continue
            if e.callee in on_path:
                continue
            reach[e.callee] = _vmax(reach.get(e.callee), c)
            stack.append((e.callee, c, on_path | {e.callee}, edges + [(v, e)]))
    if cyc is None:
        return None
    return reach, cyc, cyc_edges


class Analysis:
    """Static facts about a bytecode program, independent of any device."""

    def __init__(self, data: bytes, filename: str = "<program>", n_args: int = 0):
        self.data = bytes(data)
        self.filename = filename
        self.size = len(self.data)
        self.n_args = n_args            # program arguments: args() puts each on the value stack
        try:
            self.module = m = til.TBCModule.from_bytes(self.data)
        except Exception as e:      # malformed file: struct.error, IndexError, ValueError ...
            raise ValueError(f"cannot read the bytecode: {e}")
        if not m.funcs or not 0 <= m.entry < len(m.funcs):
            raise ValueError("cannot read the bytecode: no entry function")
        self.has_lines = bool(self.data[5] & 2)
        self.has_debug_info = bool(getattr(m, 'debug_info', False))   # the optional section after the lines
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
        self.prelude_functions = set(m.prelude.values())     # .map()/.filter()/... written in TIL
        self.field_names = {fld for _n, fields in m.types[til.TYPE_ERR + 1:] for fld in fields}
        self.clause_functions = {i for i, f in enumerate(m.funcs) if f.flags & FLAG_EFFECT_CLAUSE}
        # per reachable function: the `handle` blocks ((function, pc of its HANDLE)) that may be
        # active while it runs - a perform can only run the clauses of those
        self.handler_ctx: Dict[int, FrozenSet[Tuple[int, int]]] = {}
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

    def site_pairs(self, handle: Tuple[int, int]) -> List[Tuple[Optional[str], Optional[int]]]:
        """(effect name, clause function) pairs of the handle block at (function, pc)."""
        f, pc = handle
        return next((pairs for hpc, pairs in self.facts(f).handles if hpc == pc), [])

    def active_handles(self, f: int, site: CallSite) -> FrozenSet[Tuple[int, int]]:
        """The handle blocks that may be active at a call site of f."""
        return self.handler_ctx.get(f, frozenset()) | {(f, h) for h in site.open_handles}

    def perform_targets(self, f: int, site: CallSite) -> List[Tuple[int, int, Optional[Tuple[int, int]]]]:
        """What `perform E(...)` at a site of f may run: (function, values its frame takes,
        the handle block whose clause it is - None for the default function E and for
        clauses the bytecode does not show)."""
        m = self.module
        name, dflt, argc = self.const_text(site.args[0]), site.args[1], site.args[2]
        out: List[Tuple[int, int, Optional[Tuple[int, int]]]] = []

        def search(handle) -> bool:
            """Add the clauses of a handle block that may handle the effect; True when
            one surely does (the search stops there)."""
            sure = False
            for nm, g in self.site_pairs(handle):
                if g is None:            # a clause the bytecode does not show: any clause function
                    out.extend((c, argc, None) for c in sorted(self.clause_functions) if self.arity_ok(c, argc))
                elif nm is None or nm == name:
                    if self.arity_ok(g, argc):
                        out.append((g, argc, handle))
                    sure = sure or nm == name
            return sure

        # the handle blocks open in f itself are searched innermost first, as the VM does;
        # those of the callers may be active in any combination
        lexical = [(f, h) for h in reversed(site.open_handles)]
        if not any(search(handle) for handle in lexical):
            for handle in sorted(self.handler_ctx.get(f, frozenset()) - set(lexical)):
                search(handle)
            if dflt != NO_DEFAULT and dflt < len(m.funcs):
                out.append((dflt, argc, None))
        seen: Set[Tuple[int, Optional[Tuple[int, int]]]] = set()
        uniq = []
        for g, c, h in out:
            if (g, h) not in seen:
                seen.add((g, h))
                uniq.append((g, c, h))
        return uniq

    def call_targets(self, site: CallSite, address_taken: Set[int], caller: int = -1) -> List[Tuple[int, int]]:
        """Possible callees of a call site and how many stack values become the
        callee's parameters (the callee's frame starts that far below the top)."""
        m = self.module
        op, args = site.op, site.args
        if site.callback:           # f(x) inside .map()/...: the edges of the callers cover it
            return []
        if op == _O.CALL:
            return [(args[0], args[1])] if args[0] < len(m.funcs) else []
        if op == _O.CALL_VALUE:     # any function whose address is taken (closures, fn values)
            return [(g, args[0]) for g in sorted(address_taken) if self.arity_ok(g, args[0])]
        if op == _O.PERFORM:        # a clause of an active handle, or the default function
            out: List[Tuple[int, int]] = []
            for g, c, _h in self.perform_targets(caller, site):
                if (g, c) not in out:
                    out.append((g, c))
            return out
        name, mid, argc = self.const_text(args[0]), args[1], args[2]
        out = [(g, argc + 1) for g in self.methods_by_name.get(name, ()) if self.arity_ok(g, argc + 1)]
        if mid in m.prelude and self.arity_ok(m.prelude[mid], argc + 1):
            out.append((m.prelude[mid], argc + 1))          # map/filter/... written in TIL
        if name in self.field_names:                          # struct field holding a function
            out += [(g, argc) for g in sorted(address_taken) if self.arity_ok(g, argc)]
        return out

    def callbacks(self, site: CallSite, address_taken: Set[int]) -> List[Tuple[int, int, int]]:
        """(function, prelude function, values its frame takes) triples: .map(f) /
        .filter(f) / ... called here runs a prelude function, which calls f - the function
        passed at this site when the bytecode shows it (a lambda or a named function),
        else any function value."""
        m = self.module
        out: List[Tuple[int, int, int]] = []
        if site.op == _O.PERFORM:
            return out
        for p, c in self.call_targets(site, address_taken):
            if p not in self.prelude_functions:
                continue
            argc = site.args[2] if site.op == _O.INVOKE else site.args[-1] if site.op == _O.CALL else site.args[0]
            known = site.fn_arg if (site.op == _O.INVOKE and c == argc + 1) or site.op == _O.CALL else None
            for cb in self.facts(p).calls:
                if not cb.callback:
                    continue
                cands = sorted(address_taken) if known is None else [known] if 0 <= known < len(m.funcs) else []
                out += [(g, p, c) for g in cands if self.arity_ok(g, cb.args[0]) and (g, p, c) not in out]
        return out

    def _edges(self, ff: FuncFacts, site: CallSite) -> List[Edge]:
        out = []
        for g, c in site.targets:
            clause = 1 if site.op == _O.PERFORM and g in self.clause_functions else 0   # its running entry
            out.append(Edge(g, ff.n_locals + site.depth - c, 1, site.tries, site.hands + clause, None, site))
        for g, p, c in site.via:
            below = ff.n_locals + site.depth - c                     # the prelude function's frame
            pf = self.facts(p)
            for cb in pf.calls:
                if cb.callback:
                    out.append(Edge(g, below + pf.n_locals + cb.depth - cb.args[0], 2,
                                    site.tries + cb.tries, site.hands + cb.hands, p, site))
        return out

    # ── reachability, call graph, worst cases ────────────────────────────────
    def _resolve(self):
        m = self.module
        entry = m.entry
        address_taken: Set[int] = set()
        while True:
            # reachable functions and the handle blocks that may be active in each: a call
            # passes the caller's ones and those open at the call; a clause runs without the
            # handle block it belongs to (masking; blocks opened inside that one are kept:
            # an over-approximation)
            self.handler_ctx = ctx = {entry: frozenset()}
            reach = {entry}
            work = [entry]
            while work:
                f = work.pop()
                for site in self.facts(f).calls:
                    active = self.active_handles(f, site)
                    if site.op == _O.PERFORM:
                        callees = [(g, active - {h} if h else active) for g, _c, h in self.perform_targets(f, site)]
                    else:
                        callees = [(g, active) for g, _c in self.call_targets(site, address_taken, f)]
                        callees += [(g, active) for g, _p, _c in self.callbacks(site, address_taken)]
                    for g, gctx in callees:
                        old = ctx.get(g)
                        if old is None or not gctx <= old:
                            ctx[g] = gctx if old is None else old | gctx
                            reach.add(g)
                            work.append(g)
            # functions used as values (effect clauses are not: only perform runs them)
            refs = set()
            for f in reach:
                refs |= {g for g in self.facts(f).func_refs
                         if 0 <= g < len(m.funcs) and g not in self.clause_functions}
            if refs <= address_taken:
                break
            address_taken |= refs
        self.reachable = sorted(reach)
        self.address_taken = address_taken
        for f in self.reachable:
            for site in self.facts(f).calls:
                site.targets = self.call_targets(site, address_taken, f)
                site.via = self.callbacks(site, address_taken)
        for f in self.reachable:
            ff = self.facts(f)
            for site in ff.calls:
                site.edges = self._edges(ff, site)
        self.complete = all(self.facts(f).complete for f in self.reachable)
        self.kinds = infer_kinds(self)
        if self.kinds is not None:
            self._loops_over_arrays()
            self._arith_allocs()
        self._worst_cases()

    def handle_sites(self) -> List[Tuple[int, int]]:
        """(function, pc) of every reachable `handle` block."""
        return [(f, pc) for f in self.reachable for pc, _p in self.facts(f).handles]

    def _arith_allocs(self):
        """+ and * that give a string or an array allocate it; `s = s + x` in a loop builds
        an ever longer value (the old one becomes garbage, the live data grows)."""
        for (f, pc), (_a, _b, r) in sorted(self.kinds.binop.items()):
            if not r or r & ~K_HEAP:           # may give a number: not an allocation for sure
                continue
            ff = self.facts(f)
            sym = '+' if self.kinds.binop_at[(f, pc)][0] == _O.ADD else '*'
            if self.accumulates(ff, pc, self.kinds.binop_at[(f, pc)][1]):
                kind = f"{'array' if r & K_ARRAY else 'string'} built up with {sym}"
            else:
                kind = 'string' if r == K_STR else 'array' if r == K_ARRAY else 'string or array'
            ff.allocs.append((pc, kind))

    @staticmethod
    def accumulates(ff: FuncFacts, pc: int, k: int) -> bool:
        """The result of the binary operator (the k-th plain instruction of the instruction
        at pc) is stored into a variable it read (x = x + y, x += y)."""
        op, args, nxt = ff.code[pc]
        seq = expand(op, args) or []
        if k + 1 < len(seq):
            store = seq[k + 1]                   # ADD_STORE_LOCAL a, INC_LOCAL a
        elif nxt in ff.code and k + 1 == len(seq):
            nseq = expand(*ff.code[nxt][:2]) or []
            if not nseq or nxt in ff.jump_targets:
                return False
            store = nseq[0]
        else:
            return False
        store_op, store_args = store
        load = {_O.STORE_LOCAL: _O.LOAD_LOCAL, _O.STORE_GLOBAL: _O.LOAD_GLOBAL, _O.STORE_CELL: _O.LOAD_CELL,
                _O.STORE_UPVAL: _O.LOAD_UPVAL, _O.SET_FIELD: _O.GET_FIELD}.get(store_op)
        if load is None:
            return False
        for pos in (0, 1):
            q = _producer(ff.code, ff.prev_of, ff.jump_targets, pc, pos, upto=k)
            if q is not None and q[1] == load and q[2][0] == store_args[0]:
                return True
        return False

    def _loops_over_arrays(self):
        """for x in a: iter() puts the keys on the stack only when `a` is a map."""
        for f in self.reachable:
            ff = self.facts(f)
            plain = [pc for pc in ff.map_iters if not self.kinds.iter_arg.get((f, pc), K_ANY) & K_MAP]
            if not plain:
                continue
            for pc in plain:
                d = ff.map_iters.pop(pc)
                ff.cost[pc] = (d, d) + ff.cost[pc][2:]
            ff.data_dependent = [x for x in ff.data_dependent if x[0] not in plain]
            ff.max_occ = max(c[0] for c in ff.cost.values())
            ff.max_need = max(c[1] for c in ff.cost.values())

    def _sccs(self) -> List[List[int]]:
        """Tarjan's algorithm (iterative); components come callees-first."""
        index: Dict[int, int] = {}
        low: Dict[int, int] = {}
        on_stack: Set[int] = set()
        stack: List[int] = []
        out: List[List[int]] = []
        counter = 0
        succ = {f: sorted({e.callee for s in self.facts(f).calls for e in s.edges}) for f in self.reachable}
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

    def _own_costs(self, f: int, pcs: Optional[Set[int]] = None) -> Tuple[int, int, int, int, int]:
        """(slots written, slots needed, frames, tries, handlers) of f's own code (at the
        instructions `pcs`, default all), relative to its frame base."""
        ff = self.facts(f)
        costs = list(ff.cost.values()) if pcs is None else [ff.cost[pc] for pc in pcs]
        o = ff.n_locals + max([c[0] for c in costs] or [0])
        n = ff.n_locals + max([c[1] for c in costs] or [0])
        if f != self.module.entry:
            n = max(n, ff.n_locals + 4)          # push_frame keeps 4 slots free
        return (o, n, 1, max([c[2] for c in costs] or [0]), max([c[3] for c in costs] or [0]))

    def _worst_cases(self):
        """Per function (relative to its frame base, recursion cut after one level):
        slots written (occ) and needed (need), call depth, recovery points, handler
        entries.  Then the program totals and, for recursion, the cost of every further
        level."""
        entry = self.module.entry
        self.sccs = self._sccs()
        comp_of = {f: i for i, comp in enumerate(self.sccs) for f in comp}
        worst: Dict[int, Optional[Tuple[int, int, int, int, int]]] = {}
        rec_of: Dict[int, Recursion] = {}
        self.recursions: List[Recursion] = []
        for ci, comp in enumerate(self.sccs):            # callees before callers
            for f in comp:
                ff = self.facts(f)
                worst[f] = None
                if not ff.complete:
                    continue
                w = self._with_callees(f, ci, comp_of, worst, self._own_costs(f), ff.calls)
                if w is not None:
                    worst[f] = w
            intra = [(f, e) for f in comp for s in self.facts(f).calls for e in s.edges
                     if comp_of.get(e.callee) == ci]
            if intra and all(worst.get(f) is not None for f in comp):
                rec = Recursion(comp, intra)
                rec.tops = {f: self._top_costs(f, ci, comp_of, worst) for f in comp}
                rec_of[ci] = rec
                self.recursions.append(rec)
        self.comp_of = comp_of
        self.func_occ = {f: None if w is None else w[0] for f, w in worst.items()}
        self.func_need = {f: None if w is None else w[1] for f, w in worst.items()}
        self.func_frames = {f: None if w is None else w[2] for f, w in worst.items()}
        self.func_tries = {f: None if w is None else w[3] for f, w in worst.items()}
        self.func_hands = {f: None if w is None else w[4] for f, w in worst.items()}
        g0 = self.n_globals
        top = worst.get(entry)
        self.slots_occ = None if top is None else g0 + top[0]
        self.slots_need = None if top is None else g0 + top[1]
        self.frames = None if top is None else top[2]
        self.tries = None if top is None else top[3]
        self.hands = None if top is None else top[4]
        # where each recursion starts: longest call paths from the entry (recursion cut)
        pos: Dict[int, Vec] = {entry: (g0, 1, 0, 0)}
        for ci in reversed(range(len(self.sccs))):     # callers before callees
            comp = self.sccs[ci]
            rec = rec_of.get(ci)
            entered = [f for f in comp if f in pos]
            if rec is not None and entered:
                inner = rec.measure([(f, pos[f]) for f in entered])
                for f, p in inner.items():               # the deepest place within one level
                    pos[f] = _vmax(pos.get(f), p)
            for f in comp:
                if f not in pos:
                    continue
                for s in self.facts(f).calls:
                    for e in s.edges:
                        if comp_of.get(e.callee) != ci:
                            pos[e.callee] = _vmax(pos.get(e.callee), _vadd(pos[f], e.vec))
        self.recursions = [rec for rec in self.recursions if rec.entries]

    @staticmethod
    def _with_callees(f: int, ci: int, comp_of, worst, own: Tuple[int, int, int, int, int],
                      calls: List[CallSite], pcs: Optional[Set[int]] = None):
        """own costs combined with every call (outside the recursion ci) at the instructions
        `pcs` (default all): None when a callee's worst case is unknown."""
        o, n, fr, tr, hd = own
        for s in calls:
            if pcs is not None and s.pc not in pcs:
                continue
            for e in s.edges:
                g = e.callee
                if comp_of.get(g) == ci:               # recursive call: per-level cost
                    continue
                w = worst.get(g)
                if w is None:
                    if pcs is None:
                        return None
                    continue
                o, n = max(o, e.rel + w[0]), max(n, e.rel + w[1])
                fr, tr, hd = max(fr, e.frames + w[2]), max(tr, e.tries + w[3]), max(hd, e.hands + w[4])
        return o, n, fr, tr, hd

    def _top_costs(self, f: int, ci: int, comp_of, worst) -> Tuple[int, int, int, int, int]:
        """(slots written, slots needed, frames, recovery points, handler entries) of `f` as
        the topmost frame of its recursion: what it can run before a recursive call, with
        its other callees."""
        ff = self.facts(f)
        stop = {s.pc for s in ff.calls if s.edges and all(comp_of.get(e.callee) == ci for e in s.edges)}
        seen: Set[int] = set()
        work = [ff.start]
        while work:
            pc = work.pop()
            if pc in seen or pc not in ff.cost:
                continue
            seen.add(pc)
            if pc not in stop:
                work.extend(ff.succ.get(pc, ()))
        calls = [s for s in ff.calls if s.pc not in stop]
        res = self._with_callees(f, ci, comp_of, worst, self._own_costs(f, seen), calls, seen)
        assert res is not None
        return res

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

    def worst_path(self, table: Optional[Dict[int, Optional[int]]] = None) -> List[str]:
        """The call chain behind a worst case (value-stack need by default; recursion cut),
        as display names."""
        table = self.func_need if table is None else table
        by_slots = table is self.func_need
        f = self.module.entry
        path, seen = [self.display_name(f)], set()
        while f not in seen:
            seen.add(f)
            ff = self.facts(f)
            best, best_val = None, -1
            for s in ff.calls:
                for e in s.edges:
                    v = table.get(e.callee)
                    if e.callee in seen or v is None:
                        continue
                    v += e.rel if by_slots else e.frames
                    if v > best_val:
                        best, best_val = e, v
            if best is None or (by_slots and best_val < ff.n_locals + ff.max_need):
                break
            if best.via is not None:
                path.append(self.display_name(best.via))
            path.append(self.display_name(best.callee))
            f = best.callee
        return path

    def cycle_text(self, rec: Recursion) -> str:
        """'even -> odd -> even' (the most expensive level of a recursion)."""
        if not rec.cycle:
            return ""
        names = [self.display_name(rec.cycle[0][0])]
        for _caller, e in rec.cycle:
            if e.via is not None:
                names.append(self.display_name(e.via))
            names.append(self.display_name(e.callee))
        return " -> ".join(names)

    def display_name(self, fidx: int) -> str:
        name = self.module.funcs[fidx].name
        if fidx == self.module.entry:
            return "(start)"
        if name.startswith('__prelude_'):
            return f".{name[len('__prelude_'):]}()"
        return name


# ═══════════════════════════════════════════════════════════════════════════════
#               VALUE KINDS (what may reach +, *, map literals, for-loops)
# ═══════════════════════════════════════════════════════════════════════════════
# A dataflow analysis over the same bytecode: the kinds of value (int, str, array,
# ...) every operand-stack slot, local, global, parameter and return value may hold.
# Locals and globals are flow-insensitive; parameters get what the call sites pass,
# call results what the callees return (the call graph above).  It decides checks
# where operand types matter: array + (missing in lean builds), allocations by + and
# *, map-literal keys, and which for-loops may iterate over a map.

K_NIL, K_BOOL, K_INT, K_FLOAT, K_STR, K_ARRAY, K_MAP, K_OTHER = 1, 2, 4, 8, 16, 32, 64, 128
K_NUM = K_INT | K_FLOAT
K_ANY = 255
K_HEAP = K_STR | K_ARRAY          # results of + and * that allocate

NATIVE_KINDS = {
    'print': K_NIL, 'println': K_NIL, 'len': K_INT, 'str': K_STR, 'int': K_INT, 'float': K_FLOAT,
    'abs': K_NUM, 'sqrt': K_FLOAT, 'pow': K_NUM, 'sin': K_FLOAT, 'cos': K_FLOAT, 'tan': K_FLOAT,
    'log': K_FLOAT, 'exp': K_FLOAT, 'floor': K_NUM, 'ceil': K_NUM, 'round': K_NUM, 'input': K_STR,
    'chr': K_STR, 'ord': K_INT, 'range': K_ARRAY, 'type_of': K_STR, 'format': K_STR, 'clock_ms': K_INT,
    'random': K_NUM, 'seed': K_NIL, 'sleep_ms': K_NIL, 'bool': K_BOOL, 'iter': K_ARRAY | K_STR, 'sum': K_NUM,
    'read_file': K_STR, 'write_file': K_NIL, 'file_exists': K_BOOL, 'args': K_ARRAY, 'bytes_to_str': K_STR,
    'str_bytes': K_ARRAY, 'get_pixel': K_INT, 'key_pressed': K_INT, 'wait_key': K_INT, 'screen_width': K_INT,
    'screen_height': K_INT, 'energy_mode': K_INT,
}
NATIVE_KINDS.update({n: K_NIL for n in SCREEN_NATIVES if n not in NATIVE_KINDS})
METHOD_KINDS = {
    'len': K_INT, 'index_of': K_INT, 'count': K_INT, 'to_int': K_INT, 'to_float': K_FLOAT,
    'is_empty': K_BOOL, 'contains': K_BOOL, 'starts_with': K_BOOL, 'ends_with': K_BOOL, 'has': K_BOOL,
    'eq': K_BOOL, 'is_some': K_BOOL, 'is_none': K_BOOL, 'is_ok': K_BOOL, 'is_err': K_BOOL,
    'to_upper': K_STR, 'to_lower': K_STR, 'trim': K_STR, 'repeat': K_STR, 'replace': K_STR,
    'join': K_STR, 'to_string': K_STR, 'split': K_ARRAY, 'keys': K_ARRAY, 'values': K_ARRAY,
    'slice': K_STR | K_ARRAY, 'copy': K_ARRAY | K_MAP,
    # written in TIL (the prelude): their results are the prelude functions' return kinds
    'map': 0, 'filter': 0, 'reduce': 0, 'each': 0, 'any': 0, 'all': 0,
}
CAST_RESULT = {til.CAST_KINDS['int']: K_INT, til.CAST_KINDS['float']: K_FLOAT, til.CAST_KINDS['str']: K_STR,
               til.CAST_KINDS['bool']: K_BOOL}


def kind_names(k: int) -> str:
    names = [n for bit, n in ((K_NIL, "nil"), (K_BOOL, "bool"), (K_INT, "int"), (K_FLOAT, "float"),
                              (K_STR, "str"), (K_ARRAY, "array"), (K_MAP, "map"), (K_OTHER, "other"))
             if k & bit]
    return "any" if k == K_ANY else "/".join(names) or "none"


def arith_kinds(op: int, a: int, b: int) -> int:
    """What a + b, a * b, ... may give (the C VM's arith(); errors give nothing)."""
    r = 0
    if a & K_INT and b & K_INT:
        r |= K_INT
    if a & K_NUM and b & K_NUM and (a | b) & K_FLOAT:
        r |= K_FLOAT
    if op == _O.ADD:
        if a and b and (a | b) & K_STR:
            r |= K_STR                  # str + anything converts the other side
        if a & K_ARRAY and b & K_ARRAY:
            r |= K_ARRAY
    elif op == _O.MUL and a & K_STR and b & K_INT:
        r |= K_STR
    return r


class _KindsUnknown(Exception):
    pass


class KindFacts:
    """Results of the value-kind analysis (keys: (function, pc))."""

    def __init__(self):
        self.binop: Dict[Tuple[int, int], Tuple[int, int, int]] = {}   # ADD/MUL: (left, right, result)
        self.binop_at: Dict[Tuple[int, int], Tuple[int, int]] = {}     # ... (ADD or MUL, index in the sequence)
        self.map_keys: Dict[Tuple[int, int], List[int]] = {}            # NEW_MAP n > 0: kinds of the keys
        self.iter_arg: Dict[Tuple[int, int], int] = {}                  # NATIVE iter: its argument
        self.params: Dict[int, List[int]] = {}
        self.returns: Dict[int, int] = {}
        self.globals: List[int] = []


class _KindState:
    def __init__(self, an: 'Analysis'):
        m = an.module
        self.params = {f: [0] * m.funcs[f].n_params for f in an.reachable}
        self.rets = {f: 0 for f in an.reachable}
        self.glob = [K_NIL] * an.n_globals
        self.fields: Dict[str, int] = {}       # struct fields by name (any struct type)
        self.changed = False

    def param(self, g: int, values: List[int], first: int = 0):
        ps = self.params.get(g)
        if ps is None:
            return
        for i, k in enumerate(values):
            if first + i < len(ps) and k & ~ps[first + i]:
                ps[first + i] |= k
                self.changed = True

    def ret(self, f: int, k: int):
        if k & ~self.rets[f]:
            self.rets[f] |= k
            self.changed = True

    def field(self, name: str, k: int):
        old = self.fields.get(name, 0)
        if k & ~old:
            self.fields[name] = old | k
            self.changed = True

    def store_global(self, g: int, k: int):
        if 0 <= g < len(self.glob) and k & ~self.glob[g]:
            self.glob[g] |= k
            self.changed = True


def infer_kinds(an: 'Analysis', max_rounds: int = 200) -> Optional[KindFacts]:
    """The value kinds of a completely analysed program (None when it is incomplete or
    the analysis does not settle: every check then assumes any kind)."""
    if not an.complete:
        return None
    callbacks: Dict[int, Set[int]] = {}        # prelude function -> functions passed to it
    for f in an.reachable:
        for s in an.facts(f).calls:
            for g, p, _c in s.via:
                callbacks.setdefault(p, set()).add(g)
    st = _KindState(an)
    out = KindFacts()
    try:
        for _round in range(max_rounds):
            st.changed = False
            for f in an.reachable:
                _function_kinds(an, f, st, out, callbacks)
            if not st.changed:
                out.params, out.returns, out.globals = st.params, st.rets, st.glob
                return out
    except _KindsUnknown:
        pass
    return None


def _function_kinds(an: 'Analysis', f: int, st: _KindState, out: KindFacts, callbacks: Dict[int, Set[int]]):
    """One pass over a function (locals flow-sensitive: a store replaces the kinds).
    Superinstructions are followed through the plain instructions they stand for."""
    ff = an.facts(f)
    sites = {s.pc: s for s in ff.calls}
    params = st.params[f]
    entry_locals = tuple(params) + (K_NIL,) * max(0, ff.n_locals - len(params))
    states: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {ff.start: ((), entry_locals)}
    work = [ff.start]
    steps = 0
    while work:
        steps += 1
        if steps > 64 * len(ff.code) + 64:
            raise _KindsUnknown()
        pc = work.pop()
        stack, loc = states[pc]
        stk = list(stack)
        op, args, nxt = ff.code[pc]
        eff = STACK_EFFECTS[op]
        jump: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None   # the state at the jump target
        if eff.ends:                                  # END_HANDLE: [pairs v] -> [v]
            base = ff.handle_base.get(pc)
            if base is None or len(stk) < base + 1:
                raise _KindsUnknown()
            stk = stk[:base] + stk[-1:]
        elif eff.flow == HANDLE:                      # abort: below the pairs + the clause value
            n2 = 2 * args[0]
            if len(stk) < n2:
                raise _KindsUnknown()
            jump = (tuple(stk[:len(stk) - n2]) + (K_ANY,), loc)
        else:
            seq = expand(op, args)
            if seq is None:
                raise _KindsUnknown()
            for k, (mop, margs) in enumerate(seq):
                meff = STACK_EFFECTS[mop]
                if meff.flow == ITER:                 # FOR_ITER jumps before pushing
                    jump = (tuple(stk), loc)
                loc = _micro_kinds(an, f, pc, k, mop, margs, stk, loc, st, out, sites.get(pc), callbacks)
                if meff.flow in (JUMP, BRANCH, DEFAULT):
                    jump = (tuple(stk), loc)
                elif meff.flow == TRY:
                    jump = (tuple(stk) + (K_ANY,), loc)
        after = tuple(stk)
        succ: List[Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = []
        if eff.flow not in (JUMP, STOP):
            succ.append((nxt, after, loc))
        if jump is not None and eff.jumps():
            succ.append((_flow_target(args, nxt),) + jump)
        for q, s2, l2 in succ:
            if q not in ff.code:
                continue
            old = states.get(q)
            if old is None:
                states[q] = (s2, l2)
                work.append(q)
                continue
            if len(old[0]) != len(s2) or len(old[1]) != len(l2):
                raise _KindsUnknown()
            joined = (tuple(a | b for a, b in zip(old[0], s2)), tuple(a | b for a, b in zip(old[1], l2)))
            if joined != old:
                states[q] = joined
                work.append(q)


def _micro_kinds(an: 'Analysis', f: int, pc: int, k: int, op: int, args, stk: List[int], loc: Tuple[int, ...],
                 st: _KindState, out: KindFacts, site: Optional[CallSite], callbacks) -> Tuple[int, ...]:
    """The value kinds over one plain instruction (the k-th of the instruction at pc):
    updates the operand stack `stk` in place, returns the locals."""
    consts, types = an.module.consts, an.module.types
    eff = STACK_EFFECTS[op]
    pops = _value(eff.pops, args)
    if len(stk) < pops:
        raise _KindsUnknown()
    popped = stk[len(stk) - pops:] if pops else []
    del stk[len(stk) - pops:]
    push: List[int] = []
    if op in (_O.PUSH_I8, _O.PUSH_I16):
        push = [K_INT]
    elif op in (_O.PUSH_TRUE, _O.PUSH_FALSE):
        push = [K_BOOL]
    elif op == _O.PUSH_NIL:
        push = [K_NIL]
    elif op == _O.PUSH_CONST:
        c = consts[args[0]] if args[0] < len(consts) else None
        push = [K_STR if isinstance(c, bytes) else K_FLOAT if isinstance(c, float) else
                K_BOOL if isinstance(c, bool) else K_INT if isinstance(c, int) else K_ANY]
    elif op in (_O.DUP, _O.DUP2):
        push = popped * 2
    elif op == _O.SWAP:
        push = popped[::-1]
    elif op == _O.LOAD_LOCAL:
        push = [loc[args[0]] if args[0] < len(loc) else K_ANY]
    elif op == _O.STORE_LOCAL:
        if args[0] < len(loc):
            loc = loc[:args[0]] + (popped[0],) + loc[args[0] + 1:]
    elif op == _O.LOAD_GLOBAL:
        push = [st.glob[args[0]] if args[0] < len(st.glob) else K_ANY]
    elif op == _O.STORE_GLOBAL:
        st.store_global(args[0], popped[0])
    elif _O.ADD <= op <= _O.POW:
        r = arith_kinds(op, popped[0], popped[1])
        if op in (_O.ADD, _O.MUL):
            old = out.binop.get((f, pc))
            out.binop[(f, pc)] = (popped[0], popped[1], r) if old is None else \
                (old[0] | popped[0], old[1] | popped[1], old[2] | r)
            out.binop_at[(f, pc)] = (op, k)
        push = [r]
    elif op == _O.NEG:
        push = [popped[0] & K_NUM]
    elif op in (_O.NOT, _O.EQ, _O.NE, _O.LT, _O.LE, _O.GT, _O.GE, _O.MATCH_TYPE, _O.IS_ARRAY_LEN):
        push = [K_BOOL]
    elif op in (_O.BAND, _O.BOR, _O.BXOR, _O.BNOT, _O.SHL, _O.SHR):
        push = [K_INT | K_BOOL]
    elif op == _O.CAST:
        push = [CAST_RESULT.get(args[0], K_ANY)]
    elif op == _O.NEW_ARRAY:
        push = [K_ARRAY]
    elif op == _O.NEW_MAP:
        if args[0]:
            keys = popped[0::2]
            old = out.map_keys.get((f, pc))
            out.map_keys[(f, pc)] = keys if old is None else [a | b for a, b in zip(old, keys)]
        push = [K_MAP]
    elif op == _O.NEW_STRUCT:           # structs come only from NEW_STRUCT: fields by name
        names = types[args[0]][1] if args[0] < len(types) else []
        for i, name in enumerate(names):
            st.field(name, popped[i] if i < len(popped) else K_NIL)
        push = [K_OTHER]
    elif op == _O.SET_FIELD:
        st.field(an.const_text(args[0]), popped[1])
    elif op == _O.GET_FIELD:
        push = [st.fields.get(an.const_text(args[0]), 0)]
    elif op in (_O.MAKE_CLOSURE, _O.PUSH_FUNC):
        push = [K_OTHER]
    elif op == _O.CONCAT:
        push = [K_STR]
    elif op == _O.NATIVE:
        name = til.NATIVES[args[0]] if args[0] < len(til.NATIVES) else ""
        if name == 'iter':
            out.iter_arg[(f, pc)] = out.iter_arg.get((f, pc), 0) | (popped[0] if popped else K_ANY)
        push = [NATIVE_KINDS.get(name, K_ANY)]
    elif op in (_O.CALL, _O.CALL_VALUE, _O.INVOKE, _O.PERFORM):
        push = [_call_kinds(an, f, site, popped, st, callbacks)]
    elif op in (_O.RETURN, _O.RESUME):          # a clause's resume(v) is the value of the perform
        st.ret(f, popped[0])
    elif op == _O.RETURN_NIL:
        st.ret(f, K_NIL)
    elif op == _O.UNWRAP_OR_RETURN:
        st.ret(f, K_NIL | K_OTHER)              # None / Err(...) returned as it is
        push = [K_ANY]
    else:                                       # cells, upvalues, fields, items, ...
        push = [K_ANY] * _value(eff.pushes, args)
    if len(push) != _value(eff.pushes, args):
        raise _KindsUnknown()
    stk.extend(push)
    return loc


def _call_kinds(an: 'Analysis', f: int, site: Optional[CallSite], popped: List[int], st: _KindState,
                callbacks: Dict[int, Set[int]]) -> int:
    """Pass argument kinds to the callees of a call; the kinds its result may have."""
    if site is None:
        return K_ANY
    result, known = 0, False
    if site.op == _O.CALL_VALUE and site.callback:
        for g in sorted(callbacks.get(f, ())):           # f(x) inside .map()/...
            if an.arity_ok(g, site.args[0]):
                st.param(g, popped[1:])
                result |= st.rets.get(g, K_ANY)
                known = True
        return result if known else K_ANY
    for g, c in site.targets:
        known = True
        st.param(g, popped[len(popped) - c:])          # the callee's parameters are the last c values
        result |= st.rets.get(g, K_ANY)
    if site.op == _O.INVOKE and site.args[1] != til.NO_METHOD:
        result |= METHOD_KINDS.get(an.const_text(site.args[0]), K_ANY)
        known = True
    return result if known else K_ANY


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


def _count(n: int, noun: str) -> str:
    """'1 call frame', '3 call frames'."""
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _int_bits_needed(v: int) -> int:
    if -32768 <= v <= 32767:
        return 16
    if -2 ** 31 <= v < 2 ** 31:
        return 32
    return 64


def recursion_levels(rec: Recursion, dev: Device) -> Tuple[int, str]:
    """How many levels of a recursion surely fit the device (every level taking its most
    expensive way round), and which limit is hit first.  With k levels the device needs
        base + (k - 1) * slots_per_level + top_need             value-stack slots
        frames_below - 1 + (k - 1) * frames_per_level + top_frames   call frames
        tries_below + (k - 1) * tries_per_level + top_tries     recovery points
        hands_below + (k - 1) * hands_per_level + top_hands     effect handler entries
    for every function the recursion is entered by."""
    best: Optional[Tuple[int, str]] = None
    for en in rec.entries:
        limits = [((dev.frames - en['frames_below'] + 1 - en['top_frames']) // rec.frames_per_level + 1,
                   "call frames")]
        if rec.slots_per_level > 0:
            limits.append(((dev.stack - en['base'] - en['top_need']) // rec.slots_per_level + 1, "value stack"))
        if rec.tries_per_level > 0:
            limits.append(((dev.try_depth - en['tries_below'] - en['top_tries']) // rec.tries_per_level + 1,
                           "recovery points"))
        if rec.hands_per_level > 0 and not dev.no_effects:
            limits.append(((dev.max_handlers - en['hands_below'] - en['top_hands']) // rec.hands_per_level + 1,
                           "effect handlers"))
        cur = min(limits)
        if best is None or cur < best:
            best = cur
    assert best is not None
    return best


def lean_array_concat(an: Analysis) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """+ that joins two arrays (tv_array_concat is not in TILVM_LEAN): (sure, maybe).
    Sure: one operand is always an array, so the + only works as a concatenation."""
    sure: List[Tuple[int, int]] = []
    maybe: List[Tuple[int, int]] = []
    if an.kinds is None:
        return sure, maybe
    for (f, pc), (a, b, _r) in sorted(an.kinds.binop.items()):
        if an.kinds.binop_at[(f, pc)][0] == _O.ADD and a & K_ARRAY and b & K_ARRAY:
            (sure if K_ARRAY in (a, b) else maybe).append((f, pc))
    return sure, maybe


def _key_value(an: Analysis, ins) -> Optional[Tuple[Any, ...]]:
    """A constant map key as the full VM compares it (2.0 == 2), or None (ins: a
    (pc, op, operands) of _producer)."""
    if ins[1] == _O.PUSH_NIL:
        return ('nil',)
    v = _push_value(an, ins)
    if isinstance(v, bool):
        return ('bool', v)
    if isinstance(v, (int, float)):
        return ('num', v)
    if isinstance(v, bytes):
        return ('str', v)
    return None


def lean_map_literals(an: Analysis) -> List[Tuple[bool, str, str]]:
    """Map literals that a lean VM builds differently: it leaves out map_literal_fixup,
    so repeated keys stay (len() and printing differ), 2.0 stays a float key, and keys
    the full VM rejects (arrays, maps, structs) are accepted.  (sure, text, reason)."""
    out: List[Tuple[bool, str, str]] = []
    if an.kinds is None:
        return out
    for (f, pc), kinds in sorted(an.kinds.map_keys.items()):
        ff = an.facts(f)
        n = ff.code[pc][1][0]
        keys = []
        for i in range(n):
            q = _producer(ff.code, ff.prev_of, ff.jump_targets, pc, 2 * (n - 1 - i) + 1)
            keys.append(_key_value(an, q) if q is not None else None)
        where = an.where(f, pc)
        known = [k for k in keys if k is not None]
        repeated = sorted({_show_key(k) for k in known if known.count(k) > 1})
        floats = sorted({_show_key(k) for k in known if k[0] == 'num' and isinstance(k[1], float)
                         and k[1] == int(k[1])})
        if repeated:
            out.append((True, f"the map literal at {where} repeats the key {', '.join(repeated)}: the lean VM keeps "
                              f"every entry (the full VM keeps the last one), so len() and the output differ",
                        "repeated keys in a map literal (the lean VM keeps them)"))
        if floats:
            out.append((True, f"the map literal at {where} has the key {', '.join(floats)}: the lean VM keeps it a "
                              f"float key (the full VM turns it into an integer), so arithmetic on the key "
                              f"(3.0 / 2 is 1.5, 3 / 2 is 1) and the output may differ",
                        "float keys in a map literal (the lean VM keeps them)"))
        computed = [kinds[i] for i in range(n) if keys[i] is None]
        risks = []
        if computed and n > 1:
            risks.append("if two keys are equal it keeps both entries")
        if any(k & K_FLOAT for k in computed):
            risks.append("it keeps float keys like 2.0 as floats")
        if any(k & (K_ARRAY | K_MAP | K_OTHER) for k in computed):
            risks.append("it accepts arrays, maps and structs as keys (the full VM stops)")
        if risks:
            out.append((False, f"the map literal at {where} has computed keys; unlike the full VM the lean VM does not "
                               f"check them: {'; '.join(risks)}", ""))
    return out


def _show_key(k: Tuple[Any, ...]) -> str:
    if k[0] == 'str':
        return '"' + k[1].decode('utf-8', 'replace') + '"'
    if k[0] == 'nil':
        return "nil"
    return str(k[1]).lower() if k[0] == 'bool' else repr(k[1])


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
    hint = f" (--strip would save {strip_saving} bytes)" if strip_saving else ""
    if not ok:
        r.error(f"the bytecode is {an.size} bytes; the device keeps at most {dev.storage} bytes "
                f"for one program{hint}", f"bytecode {human(an.size)} > {human(dev.storage)} of storage")
    if dev.addr16 and an.size > 0xFFFF:
        r.error(f"the bytecode is {an.size} bytes; this device's VM uses 16-bit offsets (TILVM_ADDR16) and "
                f"loads programs below 64 KB only (tilvm_load fails with error -10){hint}",
                "bytecode larger than 64 KB (TILVM_ADDR16)")
    if an.has_debug_info:
        r.notes.append("the bytecode has a debug-info section (variable names for til debug) that the "
                       "device's VM does not use; --strip removes it with the line tables")
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
        big = sorted(r.vm_layout['parts'].items(), key=lambda kv: -kv[1])[:5]
        inside = "; the VM struct holds " + ", ".join(f"{label} {human(n)}" for label, n in big) if big else ""
        r.error(f"the VM needs {ram} bytes of RAM ({', '.join(parts)}){where}; the device has "
                f"{dev.ram}{inside}", f"VM RAM {human(ram)} > {human(dev.ram)}")

    # ── tables checked when the VM loads the program ────────────────────────
    for key, label, n, limit, code in (('consts', "constants", len(m.consts), dev.consts, -4),
                                       ('types', "struct types", len(m.types), dev.types, -7)):
        ok = n <= limit
        r.row(key, label, str(n), str(limit), "ok" if ok else "FAIL", need_value=n, limit_value=limit)
        if not ok:
            r.error(f"the program has {n} {label}; the device's VM holds at most {limit} "
                    f"(tilvm_load fails with error {code})", f"{n} {label} > {limit}")
    n, limit = len(m.funcs), dev.funcs_table
    ok = not limit or n <= limit
    r.row('funcs', "functions", str(n), str(limit) if limit else "no table", "ok" if ok else "FAIL",
          "" if limit else "TILVM_MAX_FUNCS=0: records read from the bytecode", need_value=n,
          limit_value=limit or None)
    if not ok:
        r.error(f"the program has {n} functions (prelude and lambdas included); the device's VM decodes at "
                f"most {limit} (TILVM_MAX_FUNCS; tilvm_load fails with error -10)", f"{n} functions > {limit}")
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
            listed = ", ".join(n + "()" for n in names)
            r.error(f"{listed} {'needs' if len(names) == 1 else 'need'} <math.h>; this device's VM is built "
                    f"with TILVM_NO_MATH", f"{listed} (no math library)")
        if dev.float and ('pow' in natives or any(_O.POW in ff.ops for ff in facts)):
            r.warn("x ** y and pow() compute integer powers only on this device (TILVM_NO_MATH): a "
                   "fractional exponent gives 0")

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
    concat_sure, concat_maybe = lean_array_concat(an) if dev.lean else ([], [])
    if concat_sure:
        missing.append(("array + array", concat_sure))
    label = (f"{len(natives)} native{'s' if len(natives) != 1 else ''}, "
             f"{len(methods)} method{'s' if len(methods) != 1 else ''}") + ("" if an.complete else " (incomplete)")
    off = [x for x, on in (("effect handlers", dev.no_effects and an.handle_sites()),
                           ("screen/keys natives", dev.no_gfx and any(n in natives for n in SCREEN_NATIVES)))
           if on]
    note = "; ".join(x for x in (("missing: " + ", ".join(n for n, _s in missing)) if missing else "",
                                 ("not in this build: " + ", ".join(off)) if off else "") if x)
    r.row('builtins', "builtins", label, "lean" if dev.lean else "full",
          "FAIL" if missing or off else ("warn" if partly else "ok"), note=note)
    if missing:
        r.error("the lean VM of this device leaves out " +
                "; ".join(f"{n} ({_places(an, s, 2)})" for n, s in missing) +
                " - using them stops the program", "not in the lean VM: " + ", ".join(n for n, _s in missing))
    if concat_maybe:
        r.warn(f"+ at {_places(an, concat_maybe, 3)} may join two arrays (the operand types depend on the "
               f"data); the lean VM has no array concatenation: it stops with \"unsupported operand types "
               f"for +: array and array\"")
    if dev.lean:
        for sure, text, reason in lean_map_literals(an):
            if sure:
                r.error(text, reason)
            else:
                r.warn(text)
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

    # ── effect handlers, screen and keys (build switches), device services ──
    handles = an.handle_sites()
    if handles and dev.no_effects:
        r.error(f"`handle ... with` ({_places(an, handles, 2)}) needs effect handlers; this device's VM is "
                f"built without them (TILVM_NO_EFFECTS): the program stops with \"effect handlers are not "
                f"available in this VM build\" (a `perform` without a handler still calls the function of "
                f"that name)", "effect handlers not available (TILVM_NO_EFFECTS)")
    scr = [n for n in SCREEN_NATIVES if n in natives]
    if scr and dev.no_gfx:
        listed = ", ".join(n + "()" for n in scr[:4]) + (", ..." if len(scr) > 4 else "")
        r.error(f"{listed} ({_places(an, [s for n in scr for s in natives[n]], 2)}): this device's VM is built "
                f"without the screen and keys natives (TILVM_NO_GFX): the program stops with \"... is not "
                f"available in this VM build\"", "screen natives not available (TILVM_NO_GFX)")
    elif scr and not dev.screen:
        r.warn(f"{', '.join(n + '()' for n in scr[:4])} need a screen; this device has none (the VM "
               f"reports \"not available\" unless the port adds display hooks)")
    if 'screen_init' in natives and not dev.no_gfx:
        computed = []
        for ff in facts:
            for pc, w, h in ff.screen_sizes:
                if (w is not None and not 1 <= w <= dev.gfx_max_w) or (h is not None and not 1 <= h <= dev.gfx_max_h):
                    shown = f"screen_init({'?' if w is None else w}, {'?' if h is None else h})"
                    r.error(f"{shown} at {an.where(ff.index, pc)}: this device's VM accepts screens of 1..{dev.gfx_max_w} "
                            f"x 1..{dev.gfx_max_h} pixels (TILVM_GFX_MAX_W/H); larger is a runtime error",
                            f"{shown} > {dev.gfx_max_w}x{dev.gfx_max_h} screen")
                elif w is None or h is None:
                    computed.append((ff.index, pc))
        if computed and (dev.gfx_max_w < GFX_LIMIT or dev.gfx_max_h < GFX_LIMIT):
            r.notes.append(f"screen_init() at {_places(an, computed, 2)} gets a computed size: at most "
                           f"{dev.gfx_max_w}x{dev.gfx_max_h} pixels on this device")
    if not dev.lean:
        fs = [n for n in FILE_NATIVES if n in natives]
        if fs and not dev.files:
            r.warn(f"{', '.join(n + '()' for n in fs)} need a file system hook on this device "
                   f"({_places(an, [s for n in fs for s in natives[n]], 2)})")
        if 'input' in natives and not dev.console:
            r.warn("input() returns an empty string: the device has no text input")
        if 'args' in natives:
            if dev.args:
                r.warn(f"args() puts every program argument on the value stack before building the array: "
                       f"one free slot per argument ({_places(an, natives['args'], 2)}; the numbers above count "
                       f"{_count(an.n_args, 'argument')})")
            else:
                r.notes.append("args() returns an empty array on this device (no command line)")
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
        if handles:
            r.row('handlers', "effect handlers", "?", "none" if dev.no_effects else str(dev.max_handlers), "?")
        r.error("cannot verify the stack use: analysis incomplete: " + "; ".join(probs[:3]),
                "analysis incomplete (" + probs[0] + ")")
    else:
        notes = {'stack': "", 'frames': "", 'try': "", 'handlers': ""}
        if an.recursions:
            per = max(rc.slots_per_level for rc in an.recursions)
            notes['stack'] = f"+{per} per recursion level" if per else "recursion"
            notes['frames'] = f"+{max(rc.frames_per_level for rc in an.recursions)} per recursion level"
            per = max(rc.tries_per_level for rc in an.recursions)
            notes['try'] = f"+{per} per recursion level" if per else ""
            per = max(rc.hands_per_level for rc in an.recursions)
            notes['handlers'] = f"+{per} per recursion level" if per else ""
        chains = {'stack': an.worst_path(), 'frames': an.worst_path(an.func_frames)}
        resources = [('stack', "value stack", an.slots_need, dev.stack, "slots", "value-stack slots"),
                     ('frames', "call frames", an.frames, dev.frames, "", "call frames"),
                     ('try', "recovery points", an.tries, dev.try_depth, "", "nested retry/timeout/intent blocks")]
        if handles and not dev.no_effects:
            resources.append(('handlers', "effect handlers", an.hands, dev.max_handlers, "",
                              "effect handler entries (open handle blocks + running clauses)"))
        for key, label, need, limit, unit, what in resources:
            ok = need <= limit
            r.row(key, label, f"{need} {unit}".strip(), str(limit), "ok" if ok else "FAIL", notes[key],
                  need_value=need, limit_value=limit)
            path = chains.get(key, [])
            if len(path) > 8:
                path = path[:3] + [f"... {len(path) - 6} more ..."] + path[-3:]
            chain = f" (deepest call chain: {' -> '.join(path)})" if len(path) > 2 else ""
            if not ok:
                r.error(f"the worst case needs {need} {what}{chain}; the device has {limit}",
                        f"{label} {need} > {limit}")
            elif need > 0.8 * limit and need > 4:
                r.warn(f"the worst case uses {need} of the {limit} {what}: little room left")
        if handles and dev.no_effects:
            r.row('handlers', "effect handlers", str(an.hands), "none", "FAIL", "TILVM_NO_EFFECTS",
                  need_value=an.hands, limit_value=0)
        for rec in an.recursions:
            levels, limiter = recursion_levels(rec, dev)
            names = [an.display_name(f) for f in rec.members]
            who = (f"{names[0]} calls itself" if len(names) == 1 else
                   f"{', '.join(names)} call each other")
            at = an.where(*rec.call_pc) if rec.call_pc else names[0]
            cycle = an.cycle_text(rec)
            costs = [_count(rec.slots_per_level, "value-stack slot") if rec.slots_per_level else "",
                     _count(rec.frames_per_level, "call frame"),
                     _count(rec.tries_per_level, "recovery point") if rec.tries_per_level else "",
                     _count(rec.hands_per_level, "effect handler entry").replace("entrys", "entries")
                     if rec.hands_per_level else ""]
            costs = [c for c in costs if c]
            cost = ", ".join(costs[:-1]) + " and " + costs[-1] if len(costs) > 1 else costs[0]
            r.recursion.append({'functions': names, 'at': at, 'unit': rec.unit, 'cycle': cycle,
                                'slots_per_level': rec.slots_per_level, 'frames_per_level': rec.frames_per_level,
                                'tries_per_level': rec.tries_per_level, 'handlers_per_level': rec.hands_per_level,
                                'levels': max(levels, 0), 'limited_by': limiter})
            if levels < 2:
                r.error(f"recursion: {who} ({at}) - not even two {rec.unit} fit the {limiter} of this device",
                        f"recursion in {names[0]} does not fit")
            elif rec.unit == 'calls':
                r.warn(f"recursion: {who} ({at}) in more than one cycle; the depth depends on the input. "
                       f"Each call needs at most {cost}: {levels} nested calls fit on this device "
                       f"(limited by {limiter})")
            else:
                shown = f" ({cycle})" if len(rec.cycle) > 1 or any(e.via is not None for _c, e in rec.cycle) else ""
                r.warn(f"recursion: {who} ({at}); the depth depends on the input. Each level{shown} needs "
                       f"{cost}: {levels} levels fit on this device (limited by {limiter})")
        dd: Dict[str, List[Tuple[int, int]]] = {}
        for ff in facts:
            for pc, text in ff.data_dependent:
                dd.setdefault(text, []).append((ff.index, pc))
        for text, sites in dd.items():
            r.warn(f"stack use grows with the data: {text} ({_places(an, sites, 2)}): one free slot per "
                   f"element of the {dev.stack} on this device")
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
        verb = "allocates" if len(temporary) == 1 else "allocate"
        r.notes.append(f"{_count(len(temporary), 'loop')} {verb} temporary objects on every pass "
                       f"({', '.join(kinds)}) at {places}{more}; the garbage collector reclaims them")
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#                 --run: THE C VM BUILT WITH THE DEVICE'S LIMITS
# ═══════════════════════════════════════════════════════════════════════════════

LOAD_ERRORS = {
    -1: "not a TIL bytecode file", -2: "unsupported bytecode version",
    -3: "the program uses floats and this VM has none", -4: "too many constants for this VM",
    -5: "malformed bytecode", -6: "an integer constant is too large for this VM",
    -7: "too many struct types for this VM", -8: "too many globals for the value stack",
    -10: "too many functions for this VM build (TILVM_MAX_FUNCS) or a program above 64 KB (TILVM_ADDR16)",
}
# runtime errors that come from the VM's limits: the program needs more than the device's
# VM offers (also when the desktop VM runs out too - then it needs more than any VM offers)
RESOURCE_ERRORS = ("out of memory", "stack overflow", "too many nested retry/timeout blocks",
                   "too many nested handlers", "range too large", "array too large", "string too long")
# runtime errors that come from the device's VM build (lean, no floats, no math library, no
# effect handlers or screen natives, a smaller screen)
BUILD_ERRORS = ("is not available", "are not available", "built without floating point",
                "need a build with <math.h>", "no method '")
DEVICE_ERRORS = RESOURCE_ERRORS + BUILD_ERRORS


def _cache_dir() -> str:
    """$TIL_FIT_CACHE, else a directory only this user can write to (til.user_cache_dir):
    the cached device VMs are executed, so never a shared, predictable place."""
    cache = os.environ.get('TIL_FIT_CACHE')
    if cache:
        try:
            os.makedirs(cache, exist_ok=True)
            return cache
        except OSError:
            pass
    return til.user_cache_dir()


VM_OPT = '-O2'      # optimization of the device VM builds (part of the cache key)


def device_vm_flags(dev: Device) -> List[str]:
    """The device's build emulated on the desktop: its flags (without the device compiler's
    TILVM_ROM qualifier), the high-water marks (TILVM_STATS), the 8-bit value layout
    (TILVM_PACK) and no integer fast paths - like the -Os and lean builds of the device
    ports, which run the superinstructions' slow paths: they write the most stack slots,
    and the static analysis counts exactly those."""
    flags = [f for f in dev.c_flags() if not f.startswith('-DTILVM_ROM=')]
    return flags + ['-DTILVM_STATS', '-DTILVM_FAST_PATHS=0'] + (['-DTILVM_PACK'] if dev.abi in PACKED_ABIS else [])


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
    desktop = build_device_vm(DEVICES['desktop'], verbose)[0] if compare else None
    fd, tbc = tempfile.mkstemp(suffix='.tbc')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        out = _exec_vm(exe, tbc, prog_args, max_steps, stdin_data, timeout, True)
        # nothing to compare after a device run that timed out (and it would take as long again)
        out['desktop'] = (_exec_vm(desktop, tbc, prog_args, max_steps, stdin_data, timeout, False)
                          if desktop and not out['timeout'] else None)
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
    for key, stat in (('stack', 'peak_stack'), ('frames', 'peak_frames'), ('try', 'peak_try'),
                      ('handlers', 'peak_handlers')):
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
        same = desk is not None and desk_err == err
        if any(p in err for p in RESOURCE_ERRORS + ("killed by signal",)):
            too = (" (the desktop VM stops there too: the program needs more than any VM build offers)"
                   if same else "")
            r.error(f"the run on the device's VM stops with \"{err}\"{too}", f"run: {err}")
        elif same:
            r.warn(f"the program stops with \"{err}\" on the desktop VM too (the program's own error, "
                   f"not a device limit)")
        elif any(p in err for p in BUILD_ERRORS) or (desk is not None and desk_err is None):
            r.error(f"the run on the device's VM stops with \"{err}\"", f"run: {err}")
        else:
            r.warn(f"the run stops with \"{err}\"")
    elif desk is not None and compared and not stopped and desk.get('error') is None:
        if desk['stdout'] != run['stdout'] or desk['exit_code'] != run['exit_code']:
            a, b = run['stdout'].splitlines(), desk['stdout'].splitlines()
            i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            dline = repr(a[i]) if i < len(a) else "(end)"
            wline = repr(b[i]) if i < len(b) else "(end)"
            causes = [c for c, on in (("integer width", dev.int_bits < 64), ("float precision", dev.float == 1),
                                      ("the lean VM", dev.lean)) if on]
            why = (", ".join(causes[:-1]) + " or " * (len(causes) > 1) + causes[-1] + "?") if causes \
                else "please report this"
            r.warn(f"the output differs from the desktop VM from line {i + 1} on: {dline} instead of "
                   f"{wline} ({why})")
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
        for stat, bound in (('peak_stack', an.slots_occ), ('peak_frames', an.frames), ('peak_try', an.tries),
                            ('peak_handlers', an.hands)):
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
            'recovery_points': an.tries, 'effect_handlers': an.hands, 'handle_blocks': len(an.handle_sites()),
            'debug_info': an.has_debug_info, 'recursion': r.recursion,
            'uses_float_constants': an.uses_float_consts, 'functions_detail': funcs,
        },
        'vm': {'struct_bytes': r.vm_layout['tilvm'], 'value_bytes': r.vm_layout['tv_val'],
               'struct_parts': r.vm_layout['parts'], 'c_flags': r.dev.c_flags()},
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
        try:
            stripped = til.TBCModule.from_bytes(data).serialize(strip=True) if len(data) > 5 else data
        except Exception:
            stripped = data            # malformed: Analysis reports it
        if strip:
            return (stripped if len(stripped) < len(data) else data), 0
        return data, max(0, len(data) - len(stripped))
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
        an = Analysis(data, program, n_args=len(prog_args) if want_run else 0)
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
