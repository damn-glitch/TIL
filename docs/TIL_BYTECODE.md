# TIL Bytecode (`.tbc`) — Format Specification, version 1 (and 2)

> **TIL 3.1 writes version 2.** Version 2 keeps this layout exactly and only
> appends: natives 34–54 (below), opcodes `0x76–0x79` (effect handlers) and
> `0x80–0xBF` (superinstructions), header flag bits 2–3 (initial energy mode),
> function flags 8 (`#[energy: low]`) and 16 (effect clause), and an optional
> debug-info section after the line table. Both VMs load versions 1 and 2.
> Details: [TIL_BYTECODE_V2.md](TIL_BYTECODE_V2.md).

TIL 3.0 has two backends:

| Backend | Command | Output | Runs on |
|---|---|---|---|
| **native** | `til build app.til` | C → executable (gcc/clang) | desktops/servers |
| **TIL VM** | `til build --target vm app.til` | `app.tbc` bytecode | any device with a TIL VM |

There are two VM implementations that must behave identically (the test-suite
checks this byte-for-byte on every program):

* **Reference VM** — class `TILVM` in `src/til.py` (pure Python; used by
  `til run --vm`, the REPL and when no C compiler is installed).
* **Portable VM** — `vm/tilvm.c` + `vm/tilvm.h` (C99, no `malloc`; desktops,
  graphing calculators, microcontrollers — see `vm/README.md`).

`til dis program.til` prints a readable listing of the bytecode.

---

## 1. Value model

Values are dynamically tagged, so every language feature has one meaning on
every device:

| Tag | TIL type | Notes |
|---|---|---|
| `nil` | `None`, `void` results | |
| `bool` | `true` / `false` | |
| `int` | `int`, `char` | 64-bit two's complement, wrap-around (VM builds may choose 32 or 16 bits) |
| `float` | `float` | IEEE double (VM builds may choose single precision or none) |
| `str` | `str` | UTF-8 bytes; constants are read straight from the bytecode (ROM) |
| `array` | `[a, b]`, tuples, `Vec<T>` | growable, reference semantics |
| `map` | `{k: v}`, `HashMap<K,V>` | insertion-ordered; keys: str, int, bool, nil |
| `struct` | structs, enum variants with data, `Some`/`Ok`/`Err` | type index + fields |
| `fn` | functions and closures | closures capture variables by reference (cells) |

Integer semantics follow C: `/` truncates toward zero, `%` has the sign of the
dividend, division by zero is a runtime error, `**` on integers is exact
(wrapping). `int op float` promotes to float. `+` with a string operand
concatenates (`"n=" + 42`). `==` is structural for arrays, maps and structs.
Truthiness: `nil`, `false`, `0`, `0.0`, `""`, `[]`, `{}` are false.

Floats print like C `printf("%g")` (6 significant digits).

## 2. File layout (all integers little-endian)

```
magic      "TILB"                      4 bytes
version    u8 = 1 or 2 (TIL 3.1 writes 2)
flags      u8   bit0: program uses floats   bit1: has line table
                bits 2-3 (v2): initial energy mode 0..2 (#[energy: ...] pragma)
n_globals  u16
entry      u16  function index of the entry function (__start__)

n_consts   u16
  const*   tag u8 + payload
           1 = int    i64
           2 = float  f64
           3 = str    u16 length + UTF-8 bytes

n_funcs    u16
  func*    name u16 (const) | n_params u8 | n_required u8 | n_locals u8 |
           n_upvals u8 | flags u8 | code_offset u32          (11 bytes)
           flags: 1 = method (param 0 is self), 2 = `?` propagates, 4 = lambda

n_types    u16      (types 0, 1, 2 are always Some(value), Ok(value), Err(error))
  type*    name u16 | n_fields u8 | field name u16 * n_fields

n_methods  u16      sorted by (type, name) for binary search
  method*  type u16 | name u16 (const) | func u16

n_prelude  u8       builtin array methods implemented in TIL (map, filter, ...)
  prelude* method_id u8 | func u16

code_len   u32
code       bytes

n_lines    u16      (0 when built with --strip)
  line*    pc u32 | line u16
```

A VM rejects a file whose version it does not know, and a VM built without
floats rejects programs whose `flags` say they use floats. A VM built with
32-bit or 16-bit integers rejects integer constants that do not fit.

## 3. Execution model

* **Value stack**: globals occupy the first `n_globals` slots; each call frame's
  locals (parameters first) follow.
* **Frames**: function, saved pc, base slot, running closure.
* **Recovery points** (`TRY`): handler pc, frame depth, stack depth and an
  optional deadline. A runtime error ("panic") unwinds to the innermost
  recovery point, pushes the error message (a string) and jumps to the handler.
  Without one, the VM reports `runtime error: <msg>` with a TIL-level trace
  (`at f (line N)`) and stops with exit code 1.
* **Steps**: loop back-edges and calls count as steps. They drive step budgets
  (`tilvm_run_steps`, cooperative multitasking, snapshots), the sandbox limit
  (`--max-steps`) and deadline checks (every 256 steps).

## 4. Instructions

Operands: `b` u8, `c` i8, `h` u16, `s` i16. Jumps are relative to the end of the
instruction.

| Op | Hex | Operands | Effect |
|---|---|---|---|
| NOP | 00 | | |
| PUSH_NIL / PUSH_TRUE / PUSH_FALSE | 01–03 | | push constant |
| PUSH_I8 / PUSH_I16 | 04 / 05 | c / s | push small int |
| PUSH_CONST | 06 | h | push constant (strings stay in ROM) |
| POP / DUP / DUP2 / SWAP | 07–0A | | stack shuffles |
| LOAD_LOCAL / STORE_LOCAL | 10 / 11 | b | local slot |
| LOAD_GLOBAL / STORE_GLOBAL | 12 / 13 | h | global slot |
| LOAD_UPVAL / STORE_UPVAL | 14 / 15 | b | closure variable (through its cell) |
| LOAD_CELL / STORE_CELL | 16 / 17 | b | local that is captured by a closure |
| MAKE_CELL | 18 | b | box a local into a cell |
| LOAD_UPVAL_RAW | 19 | b | push the cell itself (nested closures) |
| ARG_DEFAULT | 1A | b s | jump unless the argument slot is unset |
| ADD SUB MUL DIV MOD POW | 20–25 | | arithmetic (see §1) |
| NEG NOT BNOT | 26, 27, 2B | | unary |
| BAND BOR BXOR SHL SHR | 28–2A, 2C, 2D | | integer bit operations |
| EQ NE LT LE GT GE | 30–35 | | comparisons → bool |
| JUMP / JUMP_IF_FALSE / JUMP_IF_TRUE | 40–42 | s | (conditional jumps pop) |
| CALL | 50 | h b | call function h with b arguments |
| CALL_VALUE | 51 | b | call the function value below the arguments |
| INVOKE | 52 | h b b | method call: name const, builtin method id (FF = none), argc |
| RETURN / RETURN_NIL | 53 / 54 | | |
| NATIVE | 55 | b b | builtin function id, argc |
| NEW_ARRAY / NEW_MAP | 60 / 61 | h | from n values / n key-value pairs |
| NEW_STRUCT | 62 | h b | type, n fields (declaration order) |
| GET_INDEX / SET_INDEX | 63 / 64 | | `a[i]`, `a[i] = v` (bounds-checked) |
| GET_FIELD / SET_FIELD | 65 / 66 | h | by field-name constant |
| GET_SLOT | 67 | b | positional field / element (pattern matching) |
| MAKE_CLOSURE | 68 | h b | function + n captured cells |
| PUSH_FUNC | 69 | h | function value |
| CONCAT | 6A | b | stringify and join n values (f-strings) |
| FOR_ITER | 6B | b s | iterate the array/string in locals b, b+1 |
| MATCH_TYPE | 6C | h | is the value a struct of type h? |
| CAST | 6D | b | 0 int, 1 float, 2 str, 3 bool |
| IS_ARRAY_LEN | 6E | h | is it an array of length h? |
| TRY | 70 | s | push a recovery point (handler) |
| TRY_TIMEOUT | 71 | s | same, with a deadline (ms popped from the stack) |
| END_TRY | 72 | | pop the recovery point |
| PANIC | 73 | | raise the popped value as an error |
| UNWRAP_OR_RETURN | 74 | | `?`: unwrap Some/Ok, or return None/Err |
| UNWRAP_OR_PANIC | 75 | | `?` outside an Option/Result function |
| HALT | 7F | | end of program |

### Builtin functions (`NATIVE` id)

`print len str int float abs sqrt pow sin cos tan log exp floor ceil round min
max input chr ord range type_of format clock_ms random seed sleep_ms exit bool
sum panic iter println` — ids 0..33 in this order.

Version 2 appends ids 34..54: `read_file write_file file_exists args
bytes_to_str str_bytes screen_init screen_clear draw_pixel draw_line draw_rect
fill_rect draw_circle draw_text get_pixel screen_show key_pressed wait_key
screen_width screen_height energy_mode`. The screen and keys natives (40–53)
are specified in [TIL_GRAPHICS.md](TIL_GRAPHICS.md) — drawing algorithms, the
5×7 font and the error messages are part of the contract (identical pixels on
every VM); `energy_mode` in [TIL_ENERGY.md](TIL_ENERGY.md). Argument counts are
checked at compile time (`NATIVE_ARITY`) and the minimum again at run time.

### Builtin methods (`INVOKE` method id)

`len push pop get set contains starts_with ends_with trim to_upper to_lower
replace slice find eq split repeat to_int to_float is_empty insert remove
index_of reverse sort join first last clear sum min max has delete keys values
get_or unwrap expect unwrap_or is_some is_none is_ok is_err error to_string
copy count map filter reduce each any all` — ids 0..53 in this order.
`map filter reduce each any all` are compiled from TIL source (the prelude), so
the C VM never has to call bytecode from native code.

## 5. How language features compile

| TIL | Bytecode |
|---|---|
| `retry(n) { body } else { fb }` | `TRY h` · body · `END_TRY` · on error: count, `JUMP_IF_TRUE` back while attempts < n, then `fb` with `error` bound (or re-`PANIC`) |
| `timeout(ms) { body } else { fb }` | `TRY_TIMEOUT h` · body · `END_TRY`; the VM checks deadlines at back-edges and calls |
| `intent "goal" { body }` | `TRY` around the body; on error re-raise `intent 'goal' failed: <msg>` |
| `#[requires: c]` | check at function entry → `PANIC "contract violation: f requires c"` |
| `#[ensures: c]` | check before every return, `result` bound to the return value |
| `assert c, "m"` | `JUMP_IF_TRUE` · `PANIC "Assertion failed: m"` |
| `x?` | `UNWRAP_OR_RETURN` (in `-> Option`/`-> Result` functions) |
| `match` | patterns compile to `EQ`/`MATCH_TYPE`/`GET_SLOT`/`IS_ARRAY_LEN` tests + bindings |
| closures | captured locals live in cells (`MAKE_CELL`), closures hold the cells |

## 6. Compatibility rules

* New instructions, natives or methods are appended — existing numbers never
  change. A change of meaning bumps `version`.
* The C VM and the reference VM must produce identical output for every
  program (`tests/test_v3.py`).
