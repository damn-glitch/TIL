# TIL Installation & Setup Guide

## Complete Installation, Configuration, and Usage Guide v2.0

---

<div align="center">

**Author: Alisher Beisembekov**

*"Проще Python. Быстрее C. Умнее всех."*

*"Simpler than Python. Faster than C. Smarter than all."*

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Requirements](#2-system-requirements)
3. [Quick Installation](#3-quick-installation)
4. [Detailed Installation](#4-detailed-installation)
5. [C Compiler Setup](#5-c-compiler-setup)
6. [Configuration](#6-configuration)
7. [IDE Setup](#7-ide-setup)
8. [First Program](#8-first-program)
9. [Command Reference](#9-command-reference)
10. [Project Structure](#10-project-structure)
11. [Troubleshooting](#11-troubleshooting)
12. [Upgrading](#12-upgrading)
13. [Uninstallation](#13-uninstallation)
14. [Building from Source](#14-building-from-source)
15. [Docker Installation](#15-docker-installation)
16. [Frequently Asked Questions](#16-frequently-asked-questions)

---

## 1. Overview

### 1.1 What You'll Install

The TIL installation includes:

| Component | Description |
|-----------|-------------|
| **TIL Compiler** | The main compiler (`til` or `til.py`) |
| **Standard Library** | Built-in functions and types |
| **VS Code Extension** | Syntax highlighting and IDE features |
| **LSP Server** | Language server for IDE integration |
| **Documentation** | Language and compiler references |

### 1.2 About TIL

TIL is a multi-level programming language created by **Alisher Beisembekov**. It compiles to C code, which is then compiled to native executables using standard C compilers.

**Key Features:**
- Multi-level programming (4 abstraction levels)
- Python-like syntax
- Native performance (compiles to C)
- Single-file compiler

### 1.3 Installation Methods

| Method | Best For |
|--------|----------|
| **Quick Install** | Most users |
| **Manual Install** | Custom setups |
| **From Source** | Contributors |
| **Docker** | Isolated environment |

---

## 2. System Requirements

### 2.1 Supported Operating Systems

| OS | Version | Status |
|----|---------|--------|
| **Windows** | 10, 11 | ✅ Full Support |
| **macOS** | 10.15+ (Catalina+) | ✅ Full Support |
| **Linux** | Ubuntu 20.04+, Debian 11+, Fedora 35+ | ✅ Full Support |
| **WSL** | WSL2 | ✅ Full Support |

### 2.2 Required Software

#### Python (Required)
- **Version**: Python 3.8 or later
- **Why**: The TIL compiler is written in Python

#### C Compiler (Required)
- **GCC** (recommended for Linux/macOS)
- **Clang** (alternative)
- **MSVC** (Windows Visual Studio)
- **MinGW-w64** (recommended for Windows)

### 2.3 Minimum Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 512 MB | 4 GB |
| **Disk** | 50 MB | 500 MB |
| **CPU** | Any x64/ARM64 | Multi-core |

### 2.4 Version Check Commands

```bash
# Check Python version
python --version
# or
python3 --version

# Check GCC version
gcc --version

# Check Clang version
clang --version
```

---

## 3. Quick Installation

### 3.1 Linux / macOS

Open terminal and run:

```bash
curl -fsSL https://til-dev.vercel.app/install.sh | sh
```

This will:
1. Download the TIL compiler
2. Install to `~/.til/bin`
3. Add to your PATH
4. Verify installation

### 3.2 Windows (PowerShell)

Open PowerShell as Administrator and run:

```powershell
irm https://til-dev.vercel.app/install.ps1 | iex
```

This will:
1. Download the TIL compiler
2. Install to `%USERPROFILE%\.til\bin`
3. Add to your PATH
4. Verify installation

### 3.3 Verify Installation

After installation, open a **new terminal** and run:

```bash
til --version
```

Expected output:
```
TIL Compiler 2.0.0
Author: Alisher Beisembekov
```

---

## 4. Detailed Installation

### 4.1 Linux Installation

#### Ubuntu / Debian

```bash
# 1. Install prerequisites
sudo apt update
sudo apt install -y python3 python3-pip gcc build-essential

# 2. Download TIL compiler
mkdir -p ~/.til/bin
curl -fsSL https://raw.githubusercontent.com/til-lang/til/main/til.py \
    -o ~/.til/bin/til.py

# 3. Create wrapper script
cat > ~/.til/bin/til << 'EOF'
#!/bin/bash
python3 "$(dirname "$0")/til.py" "$@"
EOF
chmod +x ~/.til/bin/til
chmod +x ~/.til/bin/til.py

# 4. Add to PATH
echo 'export PATH="$HOME/.til/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 5. Verify
til --version
```

#### Fedora / RHEL / CentOS

```bash
# 1. Install prerequisites
sudo dnf install -y python3 python3-pip gcc

# 2. Download and install (same as Ubuntu)
mkdir -p ~/.til/bin
curl -fsSL https://raw.githubusercontent.com/til-lang/til/main/til.py \
    -o ~/.til/bin/til.py

# 3. Create wrapper and add to PATH
cat > ~/.til/bin/til << 'EOF'
#!/bin/bash
python3 "$(dirname "$0")/til.py" "$@"
EOF
chmod +x ~/.til/bin/til ~/.til/bin/til.py

echo 'export PATH="$HOME/.til/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Arch Linux

```bash
# 1. Install prerequisites
sudo pacman -S python gcc

# 2. Download and install (same steps)
# ...
```

### 4.2 macOS Installation

#### Using Homebrew (Recommended)

```bash
# 1. Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install prerequisites
brew install python gcc

# 3. Download TIL compiler
mkdir -p ~/.til/bin
curl -fsSL https://raw.githubusercontent.com/til-lang/til/main/til.py \
    -o ~/.til/bin/til.py

# 4. Create wrapper script
cat > ~/.til/bin/til << 'EOF'
#!/bin/bash
python3 "$(dirname "$0")/til.py" "$@"
EOF
chmod +x ~/.til/bin/til
chmod +x ~/.til/bin/til.py

# 5. Add to PATH (for zsh, default on macOS)
echo 'export PATH="$HOME/.til/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 6. Verify
til --version
```

#### Manual Installation

```bash
# 1. Install Xcode Command Line Tools (includes clang)
xcode-select --install

# 2. Verify Python
python3 --version

# 3. Download and install TIL
mkdir -p ~/.til/bin
curl -fsSL https://raw.githubusercontent.com/til-lang/til/main/til.py \
    -o ~/.til/bin/til.py

# 4. Create wrapper
echo '#!/bin/bash
python3 "$(dirname "$0")/til.py" "$@"' > ~/.til/bin/til
chmod +x ~/.til/bin/til ~/.til/bin/til.py

# 5. Add to PATH
echo 'export PATH="$HOME/.til/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 4.3 Windows Installation

#### Using MinGW-w64 (Recommended)

**Step 1: Install Python**

1. Download Python from https://www.python.org/downloads/
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Click "Install Now"

**Step 2: Install MinGW-w64 (GCC for Windows)**

Option A - Using winget:
```powershell
winget install -e --id mingw-w64.mingw-w64
```

Option B - Manual:
1. Download from https://www.mingw-w64.org/downloads/
2. Install to `C:\mingw64`
3. Add `C:\mingw64\bin` to PATH

**Step 3: Install TIL**

```powershell
# Create directory
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.til\bin"

# Download compiler
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/til-lang/til/main/til.py" `
    -OutFile "$env:USERPROFILE\.til\bin\til.py"

# Create batch wrapper
@"
@echo off
python "%~dp0til.py" %*
"@ | Out-File -FilePath "$env:USERPROFILE\.til\bin\til.bat" -Encoding ASCII

# Add to PATH
$path = [Environment]::GetEnvironmentVariable("Path", "User")
$tilPath = "$env:USERPROFILE\.til\bin"
if ($path -notlike "*$tilPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$tilPath;$path", "User")
}
```

**Step 4: Restart Terminal and Verify**

```powershell
# Open new PowerShell window
til --version
```

#### Using Visual Studio (Alternative)

1. Install Visual Studio with C++ workload
2. Use "Developer Command Prompt for VS"
3. TIL will automatically detect `cl.exe`

### 4.4 WSL Installation

```bash
# In WSL terminal, follow Linux instructions
curl -fsSL https://til-dev.vercel.app/install.sh | sh
```

---

## 5. C Compiler Setup

### 5.1 Why a C Compiler?

TIL compiles to C code, which requires a C compiler to produce executables:

```
TIL (.til) → C Code (.c) → Executable (.exe / binary)
```

### 5.2 Supported C Compilers

| Compiler | Platform | Command |
|----------|----------|---------|
| **GCC** | Linux, macOS, Windows (MinGW) | `gcc` |
| **Clang** | Linux, macOS | `clang` |
| **MSVC** | Windows | `cl` |

### 5.3 GCC Installation

#### Linux
```bash
# Ubuntu/Debian
sudo apt install gcc build-essential

# Fedora
sudo dnf install gcc

# Arch
sudo pacman -S gcc
```

#### macOS
```bash
# Via Xcode (installs clang as gcc)
xcode-select --install

# Or via Homebrew
brew install gcc
```

#### Windows
```powershell
# Via winget
winget install -e --id mingw-w64.mingw-w64

# Via Chocolatey
choco install mingw

# Via Scoop
scoop install gcc
```

### 5.4 Clang Installation

```bash
# Linux (Ubuntu)
sudo apt install clang

# macOS (included with Xcode)
xcode-select --install

# Windows (via LLVM)
winget install -e --id LLVM.LLVM
```

### 5.5 Verifying C Compiler

```bash
# Check GCC
gcc --version

# Check Clang
clang --version

# Windows MSVC (in Developer Command Prompt)
cl
```

### 5.6 Compiler Selection

TIL automatically searches for compilers in this order:

1. `gcc`
2. `clang`
3. `cc`
4. `cl` (Windows)

To specify a compiler:

```bash
# Set environment variable
export TIL_CC=clang

# Or use command line
til build program.til --cc clang
```

---

## 6. Configuration

### 6.1 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TIL_HOME` | TIL installation directory | `~/.til` |
| `TIL_CC` | C compiler to use | Auto-detected |
| `TIL_OPT` | Default optimization level | `2` |

```bash
# Set in ~/.bashrc or ~/.zshrc
export TIL_HOME="$HOME/.til"
export TIL_CC="gcc"
export TIL_OPT="2"
```

### 6.2 Configuration File (Future)

TIL will support a configuration file at `~/.tilconfig`:

```toml
# ~/.tilconfig (planned)
[compiler]
cc = "gcc"
optimization = 2

[editor]
indent_size = 4
theme = "dark"

[lsp]
enabled = true
```

### 6.3 Project Configuration (Future)

Project-level configuration in `til.toml`:

```toml
# til.toml
[project]
name = "my_project"
version = "1.0.0"
author = "Your Name"

[build]
optimization = 3
output_dir = "build"

[dependencies]
# future package manager support
```

---

## 7. IDE Setup

### 7.1 Visual Studio Code (Recommended)

#### Install Extension

**Method 1: Marketplace**
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search "TIL Language"
4. Click Install

**Method 2: Manual**
```bash
# Download extension
curl -fsSL https://github.com/til-lang/til/releases/download/v2.0.0/til-lang-1.0.0.vsix -o til.vsix

# Install
code --install-extension til.vsix
```

#### Configure Settings

Add to VS Code settings (`Ctrl+,`):

```json
{
    "til.compilerPath": "til",
    "til.defaultLevel": 2,
    "til.optimization": "-O2",
    "til.lsp.enabled": true,
    "[til]": {
        "editor.tabSize": 4,
        "editor.insertSpaces": true,
        "editor.formatOnSave": true
    }
}
```

#### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `F5` | Run current file |
| `F6` | Compile |
| `F7` | Compile & Run |
| `F8` | View generated C code |

### 7.2 Sublime Text

Create syntax highlighting file:

```bash
# Create package directory
mkdir -p ~/.config/sublime-text/Packages/TIL

# Download syntax file
curl -fsSL https://raw.githubusercontent.com/til-lang/til/main/editors/sublime/TIL.sublime-syntax \
    -o ~/.config/sublime-text/Packages/TIL/TIL.sublime-syntax
```

### 7.3 Vim / Neovim

Add to `.vimrc` or `init.vim`:

```vim
" TIL syntax highlighting
au BufNewFile,BufRead *.til set filetype=til
au Syntax til runtime! syntax/til.vim

" TIL compiler integration
command! TilRun !til run %
command! TilBuild !til build %
```

Create syntax file at `~/.vim/syntax/til.vim`:

```vim
" TIL syntax highlighting for Vim
" Author: Alisher Beisembekov

if exists("b:current_syntax")
    finish
endif

" Keywords
syn keyword tilKeyword if else elif for while loop in return
syn keyword tilKeyword break continue fn let var const mut
syn keyword tilKeyword struct enum impl trait match type pub as
syn keyword tilKeyword and or not self

" Types
syn keyword tilType int float str bool char void
syn keyword tilType i8 i16 i32 i64 u8 u16 u32 u64 f32 f64

" Booleans
syn keyword tilBoolean true false True False None

" Built-ins
syn keyword tilBuiltin print input sqrt abs pow sin cos tan
syn keyword tilBuiltin log exp floor ceil round min max len

" Numbers
syn match tilNumber "\<\d\+\>"
syn match tilNumber "\<0x[0-9a-fA-F]\+\>"
syn match tilNumber "\<0b[01]\+\>"
syn match tilNumber "\<\d\+\.\d*\>"

" Strings
syn region tilString start='"' end='"' skip='\\"'
syn region tilString start="'" end="'" skip="\\'"

" Comments
syn match tilComment "#.*$"

" Attributes
syn region tilAttribute start="#\[" end="\]"

" Highlighting
hi def link tilKeyword Keyword
hi def link tilType Type
hi def link tilBoolean Boolean
hi def link tilBuiltin Function
hi def link tilNumber Number
hi def link tilString String
hi def link tilComment Comment
hi def link tilAttribute PreProc

let b:current_syntax = "til"
```

### 7.4 JetBrains IDEs (PyCharm, IntelliJ)

1. Install "TextMate Bundles" plugin
2. Download TIL TextMate grammar
3. Import grammar in Settings → Editor → TextMate Bundles

### 7.5 Language Server Protocol (LSP)

TIL includes an LSP server for advanced IDE features.

#### Start LSP Server

```bash
python ~/.til/bin/til_lsp.py
```

#### VS Code Configuration

The VS Code extension automatically starts the LSP server.

#### Other Editors

Configure your editor's LSP client to connect to:

```json
{
    "command": ["python", "~/.til/bin/til_lsp.py"],
    "languageId": "til",
    "rootUri": null
}
```

---

## 8. First Program

### 8.1 Hello World

Create `hello.til`:

```til
# hello.til
# My first TIL program
# Author: [Your Name]

main()
    print("Hello, World!")
```

### 8.2 Compile and Run

```bash
# Method 1: Run directly
til run hello.til

# Method 2: Compile then run
til build hello.til -o hello
./hello          # Linux/macOS
hello.exe        # Windows
```

### 8.3 Expected Output

```
Hello, World!
```

### 8.4 More Examples

#### Calculator

```til
# calculator.til
# Author: Alisher Beisembekov

add(a: int, b: int) -> int
    return a + b

multiply(a: int, b: int) -> int
    return a * b

main()
    let x = 10
    let y = 5
    
    print("Addition:")
    print(add(x, y))
    
    print("Multiplication:")
    print(multiply(x, y))
```

#### Struct Example

```til
# point.til
# Author: Alisher Beisembekov

struct Point
    x: float
    y: float

impl Point
    new(x: float, y: float) -> Point
        return Point { x: x, y: y }
    
    distance(self) -> float
        return sqrt(self.x ** 2 + self.y ** 2)

main()
    let p = Point.new(3.0, 4.0)
    print("Distance from origin:")
    print(p.distance())
```

#### Multi-Level Example

```til
# multilevel.til
# Author: Alisher Beisembekov

# Level 0: Ultra-fast
#[level: 0]
fast_multiply(a: float, b: float) -> float
    return a * b

# Level 2: Safe (default)
compute(values: float[]) -> float
    var sum = 0.0
    for v in values
        sum = sum + fast_multiply(v, v)
    return sum

main()
    let data = [1.0, 2.0, 3.0, 4.0, 5.0]
    print("Sum of squares:")
    print(compute(data))
```

---

## 9. Command Reference

### 9.1 Basic Commands

```bash
til <command> [options] <file>
```

### 9.2 Commands

| Command | Description |
|---------|-------------|
| `run` | Compile and run |
| `build` | Compile to executable |
| `check` | Check syntax only |
| `fmt` | Format code (future) |
| `doc` | Generate documentation (future) |

### 9.3 Options

| Option | Description |
|--------|-------------|
| `-o, --output <file>` | Specify output file |
| `-c, --emit-c` | Output C code only |
| `--keep-c` | Keep intermediate C file |
| `-O0, -O1, -O2, -O3` | Optimization level |
| `-v, --version` | Show version |
| `-h, --help` | Show help |
| `--verbose` | Verbose output |

### 9.4 Examples

```bash
# Run a TIL program
til run program.til

# Compile to executable
til build program.til -o myprogram

# Compile with maximum optimization
til build program.til -O3 -o fast_program

# Generate C code only
til build program.til -c > output.c

# Check syntax without compiling
til check program.til

# Verbose compilation
til build program.til --verbose
```

### 9.5 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Syntax error |
| 2 | Compilation error |
| 3 | Runtime error |
| 4 | File not found |

---

## 10. Project Structure

### 10.1 Recommended Layout

```
my_til_project/
├── src/
│   ├── main.til           # Entry point
│   ├── lib/
│   │   ├── math.til       # Math utilities
│   │   ├── string.til     # String utilities
│   │   └── io.til         # I/O utilities
│   └── models/
│       ├── user.til       # User struct
│       └── config.til     # Configuration
├── tests/
│   ├── test_math.til
│   └── test_string.til
├── examples/
│   └── demo.til
├── docs/
│   └── README.md
├── build/                  # Compiled output
├── README.md
└── til.toml               # Project config (future)
```

### 10.2 Multi-File Projects (Future)

```til
# main.til
import math from "./lib/math.til"
import User from "./models/user.til"

main()
    let user = User.new("Alice", 30)
    let result = math.calculate(10, 20)
```

Currently, TIL supports single-file programs. Multi-file support is planned.

### 10.3 Build Script

Create a build script for complex projects:

```bash
#!/bin/bash
# build.sh

set -e

echo "Building TIL Project..."
echo "Author: Alisher Beisembekov"

# Create build directory
mkdir -p build

# Compile main
til build src/main.til -O2 -o build/app

echo "Build complete: build/app"
```

---

## 11. Troubleshooting

### 11.1 Common Issues

#### "command not found: til"

**Cause**: TIL not in PATH

**Solution**:
```bash
# Linux/macOS
export PATH="$HOME/.til/bin:$PATH"
source ~/.bashrc  # or ~/.zshrc

# Windows
# Restart terminal or run:
$env:Path = "$env:USERPROFILE\.til\bin;$env:Path"
```

#### "No C compiler found"

**Cause**: GCC/Clang not installed or not in PATH

**Solution**:
```bash
# Linux
sudo apt install gcc

# macOS
xcode-select --install

# Windows
winget install -e --id mingw-w64.mingw-w64
```

#### "Python not found"

**Cause**: Python not installed or wrong version

**Solution**:
```bash
# Install Python 3.8+
# Linux
sudo apt install python3

# macOS
brew install python

# Windows
winget install -e --id Python.Python.3.11
```

#### Syntax Error in valid code

**Cause**: Incorrect indentation (tabs vs spaces)

**Solution**:
- Use **4 spaces** for indentation
- Configure editor to use spaces, not tabs
- Run: `til check file.til` to find errors

#### Compilation error from C compiler

**Cause**: Generated C code has issues

**Solution**:
1. View generated C: `til build file.til -c`
2. Check for type mismatches
3. Report bug if TIL code is correct

### 11.2 Debug Mode

```bash
# Run with verbose output
til build program.til --verbose

# Keep C file for inspection
til build program.til --keep-c

# Check generated C code
til build program.til -c
```

### 11.3 Getting Help

1. **Documentation**: https://til-dev.vercel.app/docs
2. **GitHub Issues**: https://github.com/til-lang/til/issues
3. **Discord**: https://discord.gg/til-lang
4. **Stack Overflow**: Tag `til-lang`

### 11.4 Reporting Bugs

Include in bug reports:
- TIL version (`til --version`)
- OS and version
- Minimal code to reproduce
- Full error message
- Expected vs actual behavior

---

## 12. Upgrading

### 12.1 Linux / macOS

```bash
# Method 1: Re-run installer
curl -fsSL https://til-dev.vercel.app/install.sh | sh

# Method 2: Manual update
curl -fsSL https://raw.githubusercontent.com/til-lang/til/main/til.py \
    -o ~/.til/bin/til.py

# Verify
til --version
```

### 12.2 Windows

```powershell
# Method 1: Re-run installer
irm https://til-dev.vercel.app/install.ps1 | iex

# Method 2: Manual update
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/til-lang/til/main/til.py" `
    -OutFile "$env:USERPROFILE\.til\bin\til.py"

# Verify
til --version
```

### 12.3 VS Code Extension

1. Open Extensions view (Ctrl+Shift+X)
2. Find "TIL Language"
3. Click "Update" if available

---

## 13. Uninstallation

### 13.1 Linux / macOS

```bash
# Remove TIL directory
rm -rf ~/.til

# Remove from PATH (edit ~/.bashrc or ~/.zshrc)
# Remove the line: export PATH="$HOME/.til/bin:$PATH"

# Remove VS Code extension
code --uninstall-extension til-lang.til
```

### 13.2 Windows

```powershell
# Remove TIL directory
Remove-Item -Recurse -Force "$env:USERPROFILE\.til"

# Remove from PATH
$path = [Environment]::GetEnvironmentVariable("Path", "User")
$newPath = ($path -split ";" | Where-Object { $_ -notlike "*\.til\bin*" }) -join ";"
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")

# Remove VS Code extension
code --uninstall-extension til-lang.til
```

---

## 14. Building from Source

### 14.1 Clone Repository

```bash
git clone https://github.com/til-lang/til.git
cd til
```

### 14.2 Repository Structure

```
til/
├── src/
│   └── til.py              # Main compiler
├── lsp/
│   └── til_lsp.py          # Language server
├── vscode-extension/       # VS Code extension
├── tests/                  # Test suite
├── examples/               # Example programs
├── docs/                   # Documentation
└── README.md
```

### 14.3 Run from Source

```bash
# Direct execution
python src/til.py run examples/hello.til

# Or add to PATH
export PATH="$PWD/src:$PATH"
til run examples/hello.til
```

### 14.4 Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_lexer.py

# Run with coverage
python -m pytest --cov=src tests/
```

### 14.5 Build VS Code Extension

```bash
cd vscode-extension
npm install
npm run compile
npm run package
# Creates til-lang-X.X.X.vsix
```

---

## 15. Docker Installation

### 15.1 Using Docker

```bash
# Pull image
docker pull tillang/til:latest

# Run TIL program
docker run -v $(pwd):/code tillang/til run /code/program.til

# Interactive shell
docker run -it -v $(pwd):/code tillang/til /bin/bash
```

### 15.2 Dockerfile

```dockerfile
# Dockerfile for TIL
FROM python:3.11-slim

# Install GCC
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Install TIL
RUN mkdir -p /root/.til/bin
COPY til.py /root/.til/bin/
RUN echo '#!/bin/bash\npython3 /root/.til/bin/til.py "$@"' > /root/.til/bin/til \
    && chmod +x /root/.til/bin/til /root/.til/bin/til.py

ENV PATH="/root/.til/bin:${PATH}"

WORKDIR /code

ENTRYPOINT ["til"]
CMD ["--help"]
```

### 15.3 Build Custom Image

```bash
docker build -t my-til .
docker run -v $(pwd):/code my-til run /code/program.til
```

---

## 16. Frequently Asked Questions

### 16.1 General

**Q: Who created TIL?**
A: TIL was created by **Alisher Beisembekov**.

**Q: Is TIL free to use?**
A: Yes, TIL is open source under the MIT License.

**Q: What does TIL stand for?**
A: TIL can stand for "Today I Learned" or "Tactical Intermediate Language". The name reflects its goal of being easy to learn while being powerful.

**Q: What platforms does TIL support?**
A: Windows, macOS, Linux, and any platform with Python 3.8+ and a C compiler.

### 16.2 Technical

**Q: Why does TIL compile to C?**
A: Compiling to C allows TIL to leverage decades of C compiler optimizations and provides easy interoperability with existing C libraries.

**Q: Can I use TIL without a C compiler?**
A: No, a C compiler (GCC, Clang, or MSVC) is required to produce executables.

**Q: How fast is TIL compared to Python/C?**
A: TIL code runs at near C speed (within 10-15%) and is typically 10-100x faster than Python.

**Q: Does TIL support object-oriented programming?**
A: TIL supports structs with methods (similar to Rust), but not traditional classes with inheritance.

**Q: Can I call C functions from TIL?**
A: Direct C interop is planned for future versions. Currently, you can view and modify the generated C code.

### 16.3 Troubleshooting

**Q: Why is `til` not recognized as a command?**
A: Your PATH is not configured correctly. See [Troubleshooting](#11-troubleshooting).

**Q: Why do I get "No C compiler found"?**
A: Install GCC (Linux/macOS) or MinGW-w64 (Windows). See [C Compiler Setup](#5-c-compiler-setup).

**Q: How do I report a bug?**
A: Open an issue at https://github.com/til-lang/til/issues with your OS, TIL version, code, and error message.

---

## Quick Reference Card

### Commands

```bash
til run <file>              # Run program
til build <file>            # Compile to executable
til build <file> -o out     # Specify output name
til build <file> -c         # Output C code
til build <file> -O3        # Maximum optimization
til check <file>            # Syntax check only
til --version               # Show version
til --help                  # Show help
```

### Keyboard Shortcuts (VS Code)

| Key | Action |
|-----|--------|
| F5 | Run |
| F6 | Compile |
| F7 | Compile & Run |
| F8 | View C Code |

### Support

- Website: https://til-dev.vercel.app
- GitHub: https://github.com/til-lang/til
- Discord: https://discord.gg/til-lang
- Author: **Alisher Beisembekov**

---

<div align="center">

**TIL Installation Guide v2.0**

**Author: Alisher Beisembekov**

*"Проще Python. Быстрее C. Умнее всех."*

© 2025 TIL Language. MIT License.

</div>
