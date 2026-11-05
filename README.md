<div align="center">

# TIL Programming Language

## Mixed Martial Programming

**Simpler than Python. Faster than C. Smarter than all.**

*"Proще Python. Быстрее C. Умнее всех."*

[![CI](https://github.com/damn-glitch/TIL/actions/workflows/ci.yml/badge.svg)](https://github.com/damn-glitch/TIL/actions/workflows/ci.yml)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-orange.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-3.0.0-green.svg)](https://github.com/damn-glitch/TIL/releases)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux%20%7C%20calculators%20%7C%20MCUs-lightgrey.svg)](vm/README.md)
[![Tests](https://img.shields.io/badge/tests-336%20passing-brightgreen.svg)](tests/)

**Author: Alisher Beisembekov | Patent No 66853**

</div>

---

## What is TIL?

TIL (The Intelligent Language) is a compiled programming language with a unique **Multi-Level Architecture**. Instead of forcing one paradigm, TIL lets you pick the right level of abstraction for each function — what we call **Mixed Martial Programming**.

TIL has **two backends**: it compiles to C (native binaries via GCC/Clang), and — new in 3.0 — to compact **bytecode** for the **TIL VM**, a tiny portable virtual machine that runs the same programs on PCs without a C compiler, on **calculator-class ARM CPUs** and on **8-bit microcontrollers** (both verified in emulators). The entire compiler is still a single Python file.

TIL supports **Kazakh** (Cyrillic and Latin) and **Russian** keywords natively, and translates programs between them.

> **New in 3.0:** [TIL VM for calculators and microcontrollers](vm/README.md) · real `retry` / `timeout` / effects / capabilities / contracts · closures, algebraic data types, traits, `?` · Kazakh-Latin and Russian syntax + `til translate` · snapshots ("hibernate") · [full list (RU)](docs/TIL_3_IDEAS_RU.md) · [bytecode spec](docs/TIL_BYTECODE.md)

---

## Quick Start

### Requirements

- Python 3.9+
- GCC or Clang — optional: without a C compiler, programs run on the TIL VM

### Install & Run

```bash
git clone https://github.com/damn-glitch/TIL.git
cd TIL

# Run a program (native; falls back to the TIL VM if no C compiler is installed)
python src/til.py run examples/01_hello.til

# Run on the portable TIL VM (no C compiler needed)
python src/til.py run --vm examples/01_hello.til

# Compile to binary
python src/til.py examples/01_hello.til -o hello
./hello

# Output generated C code
python src/til.py -c examples/01_hello.til
```

### CLI Reference

```bash
python src/til.py <command> [options] <file>

Commands:
  run <file.til>          Compile and run (native, or the VM when needed)
  build <file.til>        --target native (default) | vm (.tbc) | c-vm (one C file) | c
  exec <file.tbc>         Run bytecode on the TIL VM
  dis <file>              Disassemble bytecode
  check <file.til>        Verify effects, capabilities, contracts, names
  translate <file.til>    Translate keywords: --to en | kk | kk-latn | ru
  repl                    Interactive REPL (on the VM)
  debug <file.til>        Time-travel debugger: breakpoints, step, step BACK, reverse-continue
  prove <file.til>        Prove #[requires]/#[ensures] for ALL inputs (Z3)
  fit <file.til>          Will it fit a device?  --device ti84ce|numworks|z80-16|... [--run]
  pkg <cmd>               Packages: init | add NAME SPEC | install | remove | list | verify

Options:
  --vm / --cvm            run on the Python VM / on the C VM
  --max-steps <n>         VM sandbox limit
  --strip                 smallest bytecode (no line table)
  --no-fuse               bytecode without superinstructions
  --device <name>         build: fail unless the program fits the device
  --lang en|ru|kk         message language (also TIL_LANG, or #[lang: ...] in the program)
  --screen <file.ppm>     graphics: save the final screen as a PPM image
  --screen-ansi           graphics: draw the screen in the terminal (24-bit colour)
  --keys <list>           graphics: scripted keys, e.g. "UP,UP,ENTER,a,ESC"
  --pause-after <n> --save <f.tilp>   pause after n steps, save a portable snapshot
  --resume <f.tilp>       continue a saved program (on any TIL VM)
  -- <args...>            arguments for the program (args())
  -o <file>               Output file name
  -c                      Output C code only (same as --target c)
  -O0, -O1, -O2, -O3      Optimization level (default: -O2)
  --keep-c                Keep generated C file
  --no-check              Disable type warnings
  --no-verify             Disable effect/capability/contract verification
  -v, --verbose           Verbose output
  --version               Show version
```

---

## TIL 3.0 — Run Anywhere, Even on a Calculator

```bash
til build --target vm game.til       # -> game.tbc: runs on every TIL VM
til build --target c-vm game.til     # -> game_vm.c: VM + program in ONE C file
cc -O2 -o game game_vm.c -lm         # ...or the TI-84 CE / Casio / NumWorks / Arduino toolchain
```

The TIL VM (`vm/tilvm.c`) is portable C99 with **no malloc**, a **compacting
garbage collector** in a fixed arena, configurable 16/32/64-bit integers and
optional floats, pluggable output, step budgets and **snapshots** (pause a
running program and resume it later, even after a power cycle).

| Verified in simulators | Memory used by VM + program | Result |
|---|---|---|
| ARM Cortex-M3, 64 KB RAM (NumWorks / STM32 class, QEMU) | 44.7 KB flash, 24 KB RAM | all examples identical |
| 8-bit AVR ATmega1284P, 16 KB RAM (Arduino class, simavr) | 53.8 KB flash, 10.5 KB RAM | all examples identical |
| Z80 (TI-83/84 class, ucsim) — TIL 3.1 | 16-bit lean build: code ends at 0xBE38 (below 0xC000), 5.7 KB RAM — [results](vm/ports/z80/RESULTS.md) | all examples without floats or values beyond 16 bits identical |

Every program in the test-suite runs on three engines — native C, the Python
VM and the C VM — and must produce byte-identical output. Details:
[vm/README.md](vm/README.md).

---

## TIL 3.1 — the 14 ideas, done

| # | Idea | How to use it | Docs |
|---|---|---|---|
| 1 | **Self-hosted compiler** — TIL compiled by TIL, so a calculator can compile programs itself | `til run --vm selfhost/tilc.til in.til out.tbc` | [selfhost/README.md](selfhost/README.md) |
| 2 | **Portable snapshots** — start on a PC, continue on a calculator | `til run --vm prog.til --pause-after 5000 --save s.tilp` → `tilvm --resume-portable s.tilp prog.tbc` | [TIL_SNAPSHOTS.md](docs/TIL_SNAPSHOTS.md) |
| 3 | **Graphics and keys** — `draw_pixel`, `draw_line`, `draw_text`, `key_pressed`, ... | `til run --vm examples/11_graphics.til --screen-ansi` | [TIL_GRAPHICS.md](docs/TIL_GRAPHICS.md) |
| 4 | **Time-travel debugger** — step back, reverse-continue | `til debug prog.til` (`b 12`, `c`, `back`, `rc`, `p x`) | [TIL_DEBUGGER.md](docs/TIL_DEBUGGER.md) |
| 5 | **Online playground** — compiler + VM in the browser (WebAssembly) | open `website/playground/` | [playground](website/playground/README.md) |
| 6 | **Contract proofs for all inputs** (Z3, 64-bit exact) | `til prove prog.til`, `til check --prove` | [TIL_PROOFS.md](docs/TIL_PROOFS.md) |
| 7 | **Typed native backend + generics** (inference, `E-TYPE`, monomorphization) | `til run prog.til` | [TIL_TYPES.md](docs/TIL_TYPES.md) |
| 8 | **Effects with resumption** — `handle ... with E(x) => resume(v)` | `examples/12_effects.til` | [TIL_EFFECTS.md](docs/TIL_EFFECTS.md) |
| 9 | **Z80** — 16-bit build fits below 0xC000; TI-84 Plus CE port | `make -C vm/ports/z80` | [RESULTS.md](vm/ports/z80/RESULTS.md) |
| 10 | **Will it fit?** — static + measured memory check per device | `til fit prog.til --device ti84ce` | [TIL_FIT.md](docs/TIL_FIT.md) |
| 11 | **Battery saving** — `#[energy: low]`, `energy_mode()` | `examples/14_energy.til` | [TIL_ENERGY.md](docs/TIL_ENERGY.md) |
| 12 | **Error messages in Kazakh and Russian** | `til run --lang kk prog.til`, `TIL_LANG=ru` | [TIL_I18N.md](docs/TIL_I18N.md) |
| 13 | **2–3× faster VM** — computed goto, superinstructions, caches | `python3 benchmarks/run.py` | [TIL_BYTECODE_V2.md](docs/TIL_BYTECODE_V2.md) |
| 14 | **Package manager** — `til.json`, `til.lock`, git/archive/path/registry sources | `til pkg add utils ../utils && til pkg install` | [TIL_PACKAGES.md](docs/TIL_PACKAGES.md) |

```til
# effects with resumption: the handler answers, the body continues
handle
    let name = perform Ask("name?")
    print("hello " + name)
with
    Ask(question) => resume("Aigerim")
```

```til
# graphics: the VM draws the shapes and the 5x7 font, identical pixels everywhere
screen_init(96, 64)
screen_clear(0x1E3A5F)
draw_circle(70, 32, 12, 0xFFD23F)
draw_text(2, 2, "TIL 3.1", 0xFFFFFF)
screen_show()
```

```bash
til prove examples/13_proofs.til          # proved / counterexample: x = 9223372036854775807
til fit game.til --device z80-16          # fits / does not fit: <reasons>
til run --lang kk broken.til              # қате: ... (Kazakh diagnostics)
til debug prog.til --commands "b 5; c; p x; back; p x; q"
```

Measured (details in each document):

* **Speed** (C VM, desktop, vs TIL 3.0): `fib(27)` 2.8×, loops 2.6×, arrays 3.0×,
  strings 2.4×; the Python reference VM 1.1–2.1× faster.
* **Native backend**: 97.4% of the test-suite and example programs compile
  natively (TIL 3.0: 88.7%), with output identical to the VM for every one.
* **Self-hosting**: `tilc.til` (~8,100 lines of TIL) compiles itself to the same
  bytes as the Python compiler (stage 1 = stage 2 = stage 3) — 0.4 s on the C VM.
* **Devices** (emulated): Cortex-M3 55 KB flash / 30 KB RAM, AVR ATmega1284P
  76 KB flash / 14 KB RAM, Z80 16-bit lean build below 0xC000 / 5.7 KB RAM.
  The TI-84 Plus CE port is written but has not been built with the CE
  toolchain or run on a calculator yet.

Every program in the test-suite runs on the Python VM and the C VM (and
natively where it compiles) and must produce byte-identical output.

---

## The Multi-Level System

TIL's defining feature: **choose your abstraction level per-function**.

| Level | Name | Style | Use Case |
|-------|------|-------|----------|
| **0** | Hardware | SIMD, inline asm | Inner loops, math kernels |
| **1** | Systems | Like C | Performance-critical algorithms |
| **2** | Safe | Like Rust | Application code **(default)** |
| **3** | Script | Like Python | Prototyping, glue code |
| **4** | Formal | Contracts, proofs | Safety-critical systems |

```python
#[level: 0]
fast_add(a: int, b: int) -> int    # Hardware — always inlined
    return a + b

#[level: 1]
sys_work(ptr: int) -> int          # Systems — C-like performance
    return ptr + 1

safe_calc(x: int) -> int           # Level 2 (default) — bounds checked
    let arr = [1, 2, 3, 4, 5]
    return arr[x]                  # runtime bounds check

#[level: 3]
script_mode()                      # Script — maximum ease
    print("Quick prototyping!")
```

All levels compile to the same binary. They interoperate seamlessly.

---

## Language Tour

### Hello World

```python
main()
    print("Hello, World!")
```

### Variables & Constants

```python
main()
    let name = "TIL"          # immutable (default)
    var counter = 0            # mutable
    const MAX = 100            # compile-time constant

    counter = counter + 1      # ok — var is mutable
    # name = "other"           # error — let is immutable
```

### Types

```python
main()
    let x: int = 42            # 64-bit integer
    let pi: float = 3.14159    # double precision
    let greeting: str = "hi"   # string
    let flag: bool = true      # boolean
    let ch = 'A'               # character literal
    let y = x as float         # type casting
```

### Functions

```python
add(a: int, b: int) -> int
    return a + b

greet(name: str, times: int = 1)    # default parameter
    for i in 0..times
        print("Hello, " + name + "!")

main()
    print(add(10, 20))         # 30
    greet("World")             # Hello, World!
    greet("TIL", 3)            # prints 3 times
```

### Structs & Methods

```python
struct Point
    x: float
    y: float

impl Point
    fn new(x: float, y: float) -> Point
        return Point { x: x, y: y }

    fn distance(self) -> float
        return sqrt(self.x * self.x + self.y * self.y)

    fn add(self, other: Point) -> Point
        return Point { x: self.x + other.x, y: self.y + other.y }

main()
    let p1 = Point { x: 3.0, y: 4.0 }
    print(p1.distance())       # 5
    let p2 = Point.new(1.0, 2.0)
    let p3 = p1.add(p2)
```

### Enums & Match

```python
enum Color
    Red
    Green
    Blue

enum HttpStatus
    OK = 200
    NotFound = 404
    Error = 500

main()
    let c = Color.Red
    match c
        Color.Red
            print("It's red!")
        Color.Green
            print("It's green!")
        Color.Blue
            print("It's blue!")
```

### Control Flow

```python
main()
    # If / elif / else
    let x = 42
    if x > 100
        print("big")
    elif x > 10
        print("medium")
    else
        print("small")

    # For loops with ranges
    for i in 0..5
        print(i)               # 0, 1, 2, 3, 4

    # While loops
    var n = 10
    while n > 0
        n = n - 1

    # Loop with break
    var count = 0
    loop
        count = count + 1
        if count >= 5
            break

    # FizzBuzz
    for i in 1..101
        if i % 15 == 0
            print("FizzBuzz")
        elif i % 3 == 0
            print("Fizz")
        elif i % 5 == 0
            print("Buzz")
        else
            print(i)
```

### String Operations

```python
main()
    let s = "Hello, World!"

    print(s.contains("World"))       # true
    print(s.starts_with("Hello"))    # true
    print(s.ends_with("!"))          # true
    print(s.to_upper())              # HELLO, WORLD!
    print(s.to_lower())              # hello, world!
    print(s.trim())                  # Hello, World!
    print(s.replace("World", "TIL")) # Hello, TIL!
    print(s.slice(0, 5))             # Hello
    print(s.len())                   # 13
    print(s.find("World"))           # 7
```

### F-Strings (String Interpolation)

```python
main()
    let name = "World"
    let x = 42
    print(f"Hello {name}!")        # Hello World!
    print(f"Answer: {x}")          # Answer: 42
    print(f"{name} is {x}")        # World is 42
```

### List Comprehensions

```python
main()
    let squares = [x * x for x in 0..5]
    let evens = [x for x in 0..20 if x % 2 == 0]
```

### Lambdas & Ternary

```python
main()
    let double = |x: int| -> int { x * 2 }
    print(double(5))               # 10

    let x = 10
    let result = "big" if x > 5 else "small"
    print(result)                  # big
```

### Option & Result Types

```python
find_item(id: int) -> Option<str>
    if id == 1
        return Some("Found it")
    return None

divide(a: int, b: int) -> Result<int, str>
    if b == 0
        return Err("Division by zero")
    return Ok(a / b)
```

### Global Constants

```python
const PI = 3
const GREETING = "Hello"
const PRIMES = [2, 3, 5, 7, 11]

main()
    print(GREETING)
    print(PI)
```

### Type Aliases & Traits

```python
type Score = int
type Name = str

trait Printable
    fn to_string(self) -> str
        return ""
```

### Module System

```python
# math_utils.til
square(x: int) -> int
    return x * x

cube(x: int) -> int
    return x * x * x
```

```python
# main.til
import math_utils

main()
    print(math_utils.square(5))    # 25
```

---

## Kazakh Language Support

TIL is the first programming language with full Kazakh keyword support. Every English keyword has a Kazakh equivalent. Cyrillic Kazakh is always available; Latin-script Kazakh is enabled with `#[lang: kk-latn]` and Russian with `#[lang: ru]`:

```python
#[lang: kk-latn]
# Kazakh program (Latin script)
negizgi()
    turaqty aty = "Alem"
    turaqty zhasy = 25

    eger zhasy > 18
        basyp_shygaru("Salam, " + aty + "!")
    aitpese
        basyp_shygaru("Salemetsiiz be!")

    ushin i ishinde 1..5
        basyp_shygaru(i)
```

### Full Keyword Mapping

| Kazakh | English | | Kazakh | English |
|--------|---------|---|--------|---------|
| егер | if | | тұрақты | let |
| әйтпесе | else | | айнымалы | var |
| немесе | elif | | тұрақтама | const |
| үшін | for | | құрылым | struct |
| кезінде | while | | санақ | enum |
| цикл | loop | | іске | impl |
| ішінде | in | | қасиет | trait |
| қайтару | return | | сәйкестік | match |
| тоқтату | break | | импорт | import |
| жалғастыру | continue | | бастап | from |
| функция | fn | | тип | type |
| ашық | pub | | ретінде | as |
| өзгермелі | mut | | өзін | self |
| және | and | | емес | not |
| ақиқат | true | | жалған | false |
| әйтпесе егер | elif | | басып_шығару | print |
| негізгі | main | | тексеру | assert |

Russian (`#[lang: ru]`): `если / иначе / иначе если / для … в / пока / цикл / вернуть / пусть / перем / конст / функция / структура / перечисление / реализация / выбор / и / или / не / истина / ложь / печать / главная`.

Translate any program between languages (the round trip is exact):

```bash
til translate --to ru program.til > program_ru.til
til translate --to en program_ru.til
```

Unicode identifiers are automatically mangled to valid C via `_uXXXX` encoding.

### Closures (v2.0)

```python
main()
    let x = 10
    let add_x = |y: int| -> int { y + x }   # captures x
    print(add_x(5))                           # 15
    print(add_x(20))                          # 30
```

### Dynamic Arrays (v2.0)

```python
main()
    let v = Vec<int>.new()
    v.push(10)
    v.push(20)
    v.push(30)
    print(v.len())      # 3
    print(v.get(1))     # 20
    v.pop()
    print(v.len())      # 2
```

### Hash Maps (v2.0)

```python
main()
    let m = HashMap<str, int>.new()
    m.set("alice", 95)
    m.set("bob", 87)
    print(m.get("alice"))    # 95
    print(m.has("bob"))      # true
    print(m.len())           # 2
```

### Pattern Matching with Guards (v2.0)

```python
classify(n: int) -> str
    let r = match n
        0 => "zero"
        n if n > 100 => "huge"
        n if n > 0 => "positive"
        _ => "negative"
    return r

main()
    print(classify(42))     # positive
    print(classify(-5))     # negative
    print(classify(200))    # huge
```

### Design by Contract (v2.0)

```python
#[requires: n > 0]
#[ensures: result >= 1]
factorial(n: int) -> int
    if n == 1
        return 1
    return n * factorial(n - 1)

#[pure]
square(x: int) -> int
    return x * x

main()
    print(factorial(5))    # 120
    print(square(7))       # 49
```

### Interactive REPL (v2.0)

```bash
$ python src/til.py repl
TIL v2.0 REPL - The Intelligent Language
til> 2 + 3
5
til> "hello"
hello
til> exit
Bye!
```

---

## Standard Library

### stdlib/math.til

| Function | Description |
|----------|-------------|
| `abs_val(x)` | Absolute value |
| `max_val(a, b)` | Maximum of two values |
| `min_val(a, b)` | Minimum of two values |
| `clamp(x, lo, hi)` | Clamp value to range |
| `is_even(x)` | Check if even |
| `is_odd(x)` | Check if odd |
| `gcd(a, b)` | Greatest common divisor |
| `factorial(n)` | Factorial (n!) |
| `fibonacci(n)` | Nth Fibonacci number |

### stdlib/strings.til

| Function | Description |
|----------|-------------|
| `repeat(s, n)` | Repeat string n times |
| `is_empty(s)` | Check if string is empty |
| `reverse_str(s)` | Reverse a string |

### Built-in Functions

| Function | Description |
|----------|-------------|
| `print(x)` | Print any type with newline (auto-detects type) |
| `sqrt(x)`, `pow(b, e)`, `abs(x)` | Math basics |
| `sin(x)`, `cos(x)`, `tan(x)` | Trigonometry |
| `log(x)`, `exp(x)` | Logarithm / exponential |
| `floor(x)`, `ceil(x)`, `round(x)` | Rounding |
| `min(a, b)`, `max(a, b)` | Min / max |
| `len(s)` | String or array length |
| `Some(x)`, `Ok(x)`, `Err(msg)` | Option/Result constructors |

---

## Compiler Architecture

```
TIL Source --> Lexer --> Parser --> Imports --> Verifier --+--> CCodeGenerator --> GCC --> Binary
                                                            |
                                                            +--> BytecodeCompiler --> .tbc --> TIL VM (Python / C)
```

The entire compiler is a single Python file: `src/til.py` (~9800 lines). The Verifier checks effects, capabilities, contracts and arities for both backends.

| Stage | What it does |
|-------|-------------|
| **Lexer** | Tokenizes source with indentation tracking (INDENT/DEDENT). Supports Unicode, f-strings, char literals, hex/binary/scientific numbers. |
| **Parser** | Recursive descent, 40+ AST node types. Operator precedence, struct init, pattern matching, comprehensions. |
| **TypeChecker** | Validates immutability, type annotations, return types, trait implementations. |
| **CCodeGenerator** | Translates AST to C. Multi-level optimization, Unicode name mangling, bounds checking, string ops, Option/Result as tagged unions. |
| **GCC Backend** | Compiles C with configurable optimization (-O0 to -O3). |

### Error Messages

TIL provides detailed error messages with source context and hints:

```
Error [E001]: Type mismatch at line 5, col 10
  |
5 |     let x: int = "hello"
  |                  ^^^^^^^ expected int, got str
  |
Hint: Use 'as' for type casting: "hello" as int
```

---

## Examples

The `examples/` directory contains progressive tutorials:

| File | Topic |
|------|-------|
| `01_hello.til` | Hello World, multilingual output |
| `02_variables.til` | Variables, types, casting, mutability |
| `03_functions.til` | Functions, default params, recursion (factorial, fibonacci) |
| `04_structs.til` | Structs, impl blocks, methods (Point, Rectangle, Person) |
| `05_control_flow.til` | If/elif/else, for/while/loop, break/continue, FizzBuzz |
| `06_enums.til` | Enums with values, match expressions |
| `07_multilevel.til` | Multi-level programming across all levels |
| `08_imports.til` | Module imports and reuse |
| `09_v3_features.til` | TIL 3.0: retry, timeout, ADTs, traits, Option/Result |
| `10_verified.til` | Effects, capabilities and contracts checked by the compiler |
| `11_graphics.til` | Screen graphics: shapes, text, pixel checksum (3.1) |
| `12_effects.til` | Effects with resumption: `handle ... with ... resume` (3.1) |
| `13_proofs.til` | Contracts proved for all inputs with `til prove` (3.1) |
| `14_energy.til` | Battery saving: `#[energy: low]`, `energy_mode()` (3.1) |
| `kazakh/salam.til` | Full program written in Kazakh |

Run any example:

```bash
python src/til.py run examples/01_hello.til
```

---

## Testing

**336 tests** covering the full compiler pipeline — `tests/test_compiler.py` (255) and `tests/test_v3.py` (81, every program runs natively, on the Python VM and on the C VM, outputs must be identical):

```bash
pip install pytest
python -m pytest tests/ -v
TIL_TEST_CROSS=1 python -m pytest tests/test_v3.py -k Cross   # QEMU / simavr / ucsim runs
```

| Area | Tests | Coverage |
|------|-------|----------|
| Lexer | 17 | Tokenization, literals, keywords, hex/binary, escapes |
| Parser | 26 | Expressions, statements, structs, enums, comprehensions, traits |
| Code Generation | 25 | C output, bounds checks, lambdas, string ops, globals |
| Type Checker | 4 | Immutability, type annotations, return types |
| Integration | 43 | Full compile-and-run for all features |
| Examples | 13 | All example files compile and produce correct output |
| Kazakh | 8 | Keywords, Unicode mangling, Kazakh program execution |
| Module System | 4 | Import resolution, stdlib loading |
| Option/Result | 5 | Tagged unions, Some/Ok/Err constructors |
| Globals & Aliases | 7 | Global constants (int, string, array), type aliases |
| Char & F-Strings | 12 | Char literals, escape sequences, f-string interpolation |
| Traits & Tuples | 3 | Trait definitions, tuple literals |
| Stdlib | 3 | Math and string module resolution |
| Advanced | 10 | Nested loops, recursion, enums, struct methods |
| **Closures** | 6 | Variable capture, string closures, multi-capture |
| **Dynamic Arrays** | 6 | Vec\<int/float/str\>, push/pop/get/len |
| **Hash Maps** | 5 | HashMap\<str,int/str\>, set/get/has/overwrite |
| **Pattern Matching v2** | 6 | Wildcards, guards, match-as-expression |
| **Effects & Contracts** | 7 | #[pure], #[effects], #[requires], #[ensures] |
| **REPL** | 2 | Version output, help text |
| **v2.0 Integration** | 7 | Cross-feature combinations |
| **Generics** | 2 | Generic annotations, type parameters |
| **Algebraic Effects** | 3 | Pure, IO, multi-effect annotations |
| **Capability Security** | 3 | Cap declarations, function caps, cap+effects |
| **Temporal Types** | 2 | Timeout expressions, timed computation |
| **Gradual Verification** | 5 | Assert pass/fail, invariants, multi-assert |
| **Adaptive Errors** | 2 | Retry blocks, retry expressions |
| **Energy-Aware** | 3 | Low/high energy, mixed energy functions |
| **Cross-Level Contracts** | 2 | Level transitions, level+requires |
| **Distributed Types** | 2 | Channel simulation, message queues |
| **Intent Programming** | 3 | Intent blocks with logic, loops |
| **Revolutionary Integration** | 5 | Multi-annotation, combined features |

### CI/CD

Automated testing on every push via GitHub Actions:

- **Linux** — Python 3.9, 3.11, 3.12 + GCC
- **macOS** — Python 3.11 + Clang
- **Windows** — Python 3.11 + GCC

---

## Project Structure

```
TIL/
├── src/
│   └── til.py               # The compiler: front end, verifier, C backend, bytecode compiler, reference VM
├── vm/
│   ├── tilvm.c, tilvm.h      # Portable C VM (calculators, microcontrollers, desktops)
│   ├── main.c                # tilvm command-line runner
│   └── ports/                # Cortex-M (QEMU), AVR (simavr), Z80 (ucsim) builds
├── examples/
│   ├── 01_hello.til          # Progressive tutorials (01-08)
│   ├── 02_variables.til
│   ├── 03_functions.til
│   ├── 04_structs.til
│   ├── 05_control_flow.til
│   ├── 06_enums.til
│   ├── 07_multilevel.til
│   ├── 08_imports.til
│   ├── math_utils.til        # Importable library module
│   └── kazakh/
│       └── salam.til         # Kazakh language example
├── stdlib/
│   ├── math.til              # Math standard library
│   └── strings.til           # String standard library
├── tests/
│   ├── test_compiler.py      # 255 tests
│   └── test_v3.py            # 81 tests: native == Python VM == C VM
├── .github/
│   └── workflows/
│       └── ci.yml            # CI for Linux, macOS, Windows
└── README.md
```

---

## Feature Summary

### Implemented in v1.5

- Multi-level system (Levels 0-3 with `#[level: N]` annotations)
- Variables (`let`, `var`, `const`) with immutability enforcement
- Types: `int`, `float`, `str`, `bool`, `char`, `void`
- Functions with default parameters and recursion
- Structs with `impl` blocks and methods
- Enums with values and `match` expressions
- Control flow: `if`/`elif`/`else`, `for`/`while`/`loop`, `break`/`continue`
- Arrays with bounds checking (Level 2+)
- String operations (11 built-in methods)
- F-string interpolation (`f"Hello {name}!"`)
- Character literals (`'x'`)
- List comprehensions with filters (`[x*2 for x in 0..10 if x > 3]`)
- Lambda expressions (`|x| x * 2`)
- Ternary expressions (`a if cond else b`)
- Option\<T\> and Result\<T, E\> types
- Global constants and variables
- Type aliases (`type Name = int`)
- Trait definitions with validation
- Tuple literals
- Module system with `import` / `from ... import`
- Standard library (math, strings)
- Kazakh language keywords (30+ keywords)
- Unicode identifier support with name mangling
- Beautiful error messages with source context and hints
- Type checker with immutability and annotation validation
- CI/CD across Linux, macOS, Windows

### New in v2.0

- [x] **Closures** with variable capture (`|x: int| -> int { x + captured_var }`)
- [x] **Dynamic Arrays** — `Vec<int>`, `Vec<float>`, `Vec<str>` with `.push()`, `.pop()`, `.get()`, `.len()`
- [x] **Hash Maps** — `HashMap<str, int>`, `HashMap<str, str>` with `.set()`, `.get()`, `.has()`, `.len()`
- [x] **Pattern Matching** with wildcards (`_`) and guards (`pattern if cond => expr`)
- [x] **Match expressions** as values (`let r = match x ...`)
- [x] **Effect System** — `#[pure]`, `#[effects: io]` annotations
- [x] **Design by Contract** — `#[requires: expr]`, `#[ensures: expr]` with runtime verification
- [x] **Interactive REPL** — `til repl` for live expression evaluation

### Revolutionary Features (v2.0)

- [x] **Generics** — `#[generic: T]` annotations; on the VM every function is generic (dynamic typing)
- [x] **Effect System** — `#[pure]`, `#[effects: io, fs]` — effects are inferred and **checked by the compiler** (3.0)
- [x] **Capability-Based Security** — `#[cap: FileRead]` functions can only be used by functions holding the capability (or `main`) — **compile-time check** (3.0)
- [x] **Temporal Types** — `timeout(ms) { … } else { … }` **really interrupts** long computations (3.0)
- [x] **Gradual Verification** — `assert`, `invariant`, `#[requires]`/`#[ensures]`; violations with constant arguments are **compile errors** (3.0)
- [x] **Adaptive Error Handling** — `retry(n) { … } else { … }` **really re-runs** the body after any runtime error (3.0)
- [x] **Energy-Aware Compilation** — `#[energy: low|medium|high]` optimization hints
- [x] **Cross-Level Contracts** — `#[cross_level: 1 -> 2]` annotations
- [x] **Distributed Types** — Channel-like patterns with `Vec<T>` message queues
- [x] **Intent-Based Programming** — `intent "description" { body }` — failures report the intent (3.0, VM)

### New in v3.0

- [x] **TIL VM** — bytecode backend, Python reference VM, portable C VM (no malloc, compacting GC)
- [x] **Calculators & microcontrollers** — verified on emulated ARM Cortex-M3 and 8-bit AVR (Z80: compiles, does not fit 64 KB yet)
- [x] **Snapshots** — pause a running program and resume it later
- [x] **Closures as values**, `map` / `filter` / `reduce` / `each` / `any` / `all`, pipeline `|>`
- [x] **Algebraic data types** — enum variants with data, destructuring `match`
- [x] **Traits** with default methods and dynamic dispatch
- [x] **Option / Result** with `?` propagation and `match` on `Some/None/Ok/Err`
- [x] **Patterns**: ranges, alternatives `1 | 2`, tuples, struct patterns, guards
- [x] **Maps** `{"k": v}`, string methods, f-string format specs `{x:.2f}`
- [x] **Script mode** — top-level statements are a program
- [x] **Kazakh Latin + Russian** syntax, `til translate`
- [x] 25+ bug fixes (f-strings, string equality, one-line functions, Kazakh `elif`, injection-safe C output, ...)

### Roadmap

- [x] Full pattern matching with destructuring (VM)
- [x] Full trait dispatch (dynamic method resolution) (VM)
- [x] Self-hosted compiler → compile TIL on the calculator itself (3.1)
- [x] Portable snapshots: start a program on a PC, continue it on a calculator (3.1)
- [x] Calculator graphics & keypad API (`draw_pixel`, `key_pressed`, ...) (3.1)
- [x] Contracts proven with an SMT solver (Level 4 "Formal") (3.1)
- [x] Typed AST + monomorphization for the native backend (3.1)
- [x] WebAssembly build of the VM for an in-browser playground (3.1)
- [x] Package manager (3.1)
- [x] Time-travel debugger, effects with resumption, battery saving, Kazakh/Russian messages, 2–3× faster VM, memory-fit check, Z80 16-bit build (3.1)
- [ ] TI-84 Plus CE: build and run on a real calculator; TI-84 Plus Flash application (banked Z80)
- [ ] LSP keyword sync

All ideas, in Russian: [docs/TIL_3_IDEAS_RU.md](docs/TIL_3_IDEAS_RU.md)

---

## Philosophy

> "Mixed Martial Programming — use the right level for each task."

Most languages force you into one paradigm. TIL gives you **five levels** in one language:

- Writing a game engine? Use **Level 0-1** for the hot loop, **Level 2** for game logic.
- Building a web service? Use **Level 2** for core logic, **Level 3** for configuration.
- Proving correctness? Use **Level 4** for safety-critical paths.

All levels compile to the same binary. You pick the right tool for each function.

---

<div align="center">

**Author: Alisher Beisembekov**

Made in Kazakhstan

*"Proще Python. Быстрее C. Умнее всех."*

*Patent No 66853*

</div>
