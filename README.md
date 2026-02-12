<div align="center">

# TIL Programming Language

## Mixed Martial Programming

**Simpler than Python. Faster than C. Smarter than all.**

*"Proще Python. Быстрее C. Умнее всех."*

[![CI](https://github.com/damn-glitch/TIL/actions/workflows/ci.yml/badge.svg)](https://github.com/damn-glitch/TIL/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/damn-glitch/TIL/releases)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/damn-glitch/TIL)
[![Tests](https://img.shields.io/badge/tests-223%20passing-brightgreen.svg)](tests/test_compiler.py)

**Author: Alisher Beisembekov | Patent No 66853**

</div>

---

## What is TIL?

TIL (The Intelligent Language) is a compiled programming language with a unique **Multi-Level Architecture**. Instead of forcing one paradigm, TIL lets you pick the right level of abstraction for each function — what we call **Mixed Martial Programming**.

TIL compiles to C via GCC, producing native binaries. The entire compiler is a single Python file.

TIL also supports **Kazakh language keywords** natively — the first programming language with full Kazakh syntax support.

---

## Quick Start

### Requirements

- Python 3.9+
- GCC (any recent version)

### Install & Run

```bash
git clone https://github.com/damn-glitch/TIL.git
cd TIL

# Run a program
python src/til.py run examples/01_hello.til

# Compile to binary
python src/til.py examples/01_hello.til -o hello
./hello

# Output generated C code
python src/til.py -c examples/01_hello.til
```

### CLI Reference

```bash
python src/til.py [command] [options] <input.til>

Commands:
  run <file>              Compile and run immediately
  build <file>            Compile to executable
  check <file>            Type-check only (no compilation)
  repl                    Interactive REPL

Options:
  -o <file>               Output file name
  -c                      Output C code only (don't compile)
  -O0, -O1, -O2, -O3     Optimization level (default: -O2)
  --keep-c                Keep generated C file
  --no-check              Disable type checking
  -v, --verbose           Verbose output
  --version               Show version
```

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

TIL is the first programming language with full Kazakh keyword support. Every English keyword has a Kazakh equivalent:

```python
# Kazakh program
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
TIL Source --> Lexer --> Parser --> TypeChecker --> CCodeGenerator --> GCC --> Binary
```

The entire compiler is a single Python file: `src/til.py` (~5000 lines).

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
| `kazakh/salam.til` | Full program written in Kazakh |

Run any example:

```bash
python src/til.py run examples/01_hello.til
```

---

## Testing

**223 tests** covering the full compiler pipeline:

```bash
pip install pytest
python -m pytest tests/test_compiler.py -v
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
│   └── til.py               # The compiler (single file, ~5000 lines)
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
│   └── test_compiler.py      # 223 tests
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

### Roadmap to v3.0

- [ ] Generics (`fn max<T>(a: T, b: T) -> T`)
- [ ] Full pattern matching with destructuring
- [ ] Full trait dispatch (dynamic method resolution)
- [ ] LSP server for editor integration
- [ ] Package manager
- [ ] Expanded standard library

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
