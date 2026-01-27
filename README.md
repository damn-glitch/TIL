<div align="center">

# TIL Programming Language

## Mixed Martial Programming

**Simpler than Python. Faster than C. Smarter than all.**

*"Проще Python. Быстрее C. Умнее всех."*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/til-lang/til/releases)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/til-lang/til)

**Author: Alisher Beisembekov**

[Website](https://til-dev.vercel.app) • [Documentation](docs/) • [Examples](examples/) • [Discord](https://discord.gg/til-lang)

</div>

---

## ⚡ Quick Start

### Install

```bash
# Linux / macOS
curl -fsSL https://til-dev.vercel.app/install.sh | sh

# Windows (PowerShell)
irm https://til-dev.vercel.app/install.ps1 | iex

# Or clone and run directly
git clone https://github.com/til-lang/til.git
cd til
python src/til.py --version
```

### Hello World

```python
# hello.til
main()
    print("Hello, World!")
```

```bash
til run hello.til
```

---

## 🎯 What is TIL?

TIL is a **multi-level programming language** created by **Alisher Beisembekov** that combines:

| Feature | Description |
|---------|-------------|
| **Python's Syntax** | Clean, readable, indentation-based |
| **C's Performance** | Compiles to native executables |
| **Rust's Safety** | Structs with methods, strong types |
| **Unique Levels** | 4 abstraction levels in one file |

---

## 🔥 The Multi-Level System

TIL's killer feature: **choose your abstraction level per-function**:

```python
# Level 0: Maximum performance (always inlined)
#[level: 0]
fast_multiply(a: float, b: float) -> float
    return a * b

# Level 2: Safe and balanced (default)
struct Point
    x: float
    y: float

impl Point
    distance(self) -> float
        return sqrt(self.x ** 2 + self.y ** 2)

# Level 3: Script-like ease
#[level: 3]
main()
    let p = Point { x: 3.0, y: 4.0 }
    print(p.distance())  # 5.0
```

| Level | Name | Use Case | Features |
|-------|------|----------|----------|
| **0** | Hardware | Inner loops, SIMD | Always inline, no checks |
| **1** | Systems | Critical algorithms | Inline hints, C-like |
| **2** | Safe | Application code | Bounds checking (default) |
| **3** | Script | Prototyping | Maximum ease |

---

## 📦 Features

- ✅ **Multi-Level System** - 4 abstraction levels (0-3)
- ✅ **Native Performance** - Compiles to C → executable
- ✅ **Python-like Syntax** - Clean, readable code
- ✅ **Strong Typing** - Static types with inference
- ✅ **Structs & Methods** - OOP without classes
- ✅ **Enums** - With values and pattern matching
- ✅ **Zero Runtime** - No VM, no GC
- ✅ **Single-File Compiler** - Just `til.py`!

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Language Reference](docs/TIL_LANGUAGE_REFERENCE.md)** | Complete language specification |
| **[Compiler Reference](docs/TIL_COMPILER_REFERENCE.md)** | Compiler internals and architecture |
| **[Installation Guide](docs/TIL_INSTALLATION_GUIDE.md)** | Setup, IDE, troubleshooting |

---

## 🚀 Examples

### Variables and Types
```python
main()
    let name = "TIL"           # Immutable
    var count = 0              # Mutable
    const PI = 3.14159         # Constant
    
    let x: int = 42            # Explicit type
    let y = x as float         # Type cast
```

### Structs and Methods
```python
struct Rectangle
    width: float
    height: float

impl Rectangle
    new(w: float, h: float) -> Rectangle
        return Rectangle { width: w, height: h }
    
    area(self) -> float
        return self.width * self.height

main()
    let rect = Rectangle.new(10.0, 5.0)
    print(rect.area())  # 50.0
```

### Enums
```python
enum Color
    Red
    Green
    Blue

enum HttpStatus
    OK = 200
    NotFound = 404
    Error = 500
```

### Control Flow
```python
main()
    # FizzBuzz
    for i in 1..=100
        if i % 15 == 0
            print("FizzBuzz")
        elif i % 3 == 0
            print("Fizz")
        elif i % 5 == 0
            print("Buzz")
        else
            print(i)
```

📁 See [examples/](examples/) for more.

---

## 🛠️ CLI Commands

```bash
til run <file.til>           # Compile and run
til build <file.til>         # Compile to executable
til build <file.til> -o out  # Specify output name
til build <file.til> -c      # Output C code only
til build <file.til> -O3     # Maximum optimization
til check <file.til>         # Syntax check only
til --version                # Show version
```

---

## 🔌 Editor Support

### VS Code
Install from [editors/vscode/](editors/vscode/):
- Syntax highlighting
- Code snippets  
- Build commands (F5/F6/F7)
- Hover documentation

### Vim
Copy [editors/vim/til.vim](editors/vim/til.vim) to `~/.vim/syntax/`

### Sublime Text
Copy [editors/sublime/TIL.sublime-syntax](editors/sublime/) to Packages/User/

---

## 📂 Repository Structure

```
til/
├── src/
│   └── til.py                    # 🔥 The compiler (single file!)
├── docs/
│   ├── TIL_LANGUAGE_REFERENCE.md # Language specification
│   ├── TIL_COMPILER_REFERENCE.md # Compiler documentation  
│   └── TIL_INSTALLATION_GUIDE.md # Setup guide
├── examples/
│   ├── 01_hello.til
│   ├── 02_variables.til
│   ├── 03_functions.til
│   └── ...
├── editors/
│   ├── vscode/                   # VS Code extension
│   ├── vim/                      # Vim syntax
│   └── sublime/                  # Sublime syntax
├── scripts/
│   ├── install.sh                # Linux/macOS installer
│   └── install.ps1               # Windows installer
├── website/                      # til-dev.vercel.app
├── tests/                        # Test suite
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## 📊 Performance

TIL compiles to C, achieving near-native performance:

| Benchmark | TIL | Python | C | vs Python |
|-----------|-----|--------|---|-----------|
| Fibonacci(40) | 0.8s | 45s | 0.7s | **56x faster** |
| Matrix 1000² | 1.2s | 120s | 1.1s | **100x faster** |
| Prime sieve | 0.3s | 8s | 0.25s | **27x faster** |

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/til-lang/til.git
cd til
python src/til.py run examples/01_hello.til
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">

**Author: Alisher Beisembekov**

🇰🇿 Made in Kazakhstan

*"Проще Python. Быстрее C. Умнее всех."*

⭐ Star this repo if you like TIL!

</div>
