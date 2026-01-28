# TIL Language for Visual Studio Code

**TIL: Simpler than Python. Faster than C. Smarter than all.**

*by Alisher Beisembekov*

## Features

- 🎨 **Syntax Highlighting** - Full support for TIL syntax
- ▶️ **Run (F5)** - Run TIL files directly
- 🔨 **Build (F6)** - Compile to executable
- ✓ **Check (F7)** - Syntax checking
- ⚡ **Code Snippets** - Quick templates for common patterns
- 🔴 **Error Highlighting** - Inline error diagnostics
- 📝 **Language Configuration** - Auto-closing brackets, comments, indentation

## Keyboard Shortcuts

| Shortcut | Command |
|----------|---------|
| `F5` | Run current file |
| `F6` | Build current file |
| `F7` | Check syntax |
| `Ctrl+F5` | Run in terminal |

## Right-Click Menu

Right-click in a `.til` file to see:
- TIL: Run File
- TIL: Build File
- TIL: Check Syntax

## Requirements

- **TIL Compiler** - Install from https://til-dev.vercel.app

```bash
# Windows (PowerShell)
irm https://til-dev.vercel.app/install.ps1 | iex

# Linux/macOS
curl -fsSL https://til-dev.vercel.app/install.sh | sh
```

## Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `til.compilerPath` | Path to TIL compiler | `til` |
| `til.optimization` | Optimization level | `-O2` |

## Snippets

| Prefix | Description |
|--------|-------------|
| `main` | Main function |
| `fn` | Function definition |
| `struct` | Struct definition |
| `impl` | Impl block |
| `for` | For loop |
| `if` | If statement |
| `level0` | Level 0 function |

## Example

```til
struct Point
    x: float
    y: float

impl Point
    distance(self) -> float
        return sqrt(self.x ** 2 + self.y ** 2)

main()
    let p = Point { x: 3.0, y: 4.0 }
    print(p.distance())  # 5.0
```

Press **F5** to run!

## Links

- [Website](https://til-dev.vercel.app)
- [GitHub](https://github.com/damn-glitch/TIL)
- [Documentation](https://github.com/damn-glitch/TIL/tree/main/docs)

## Author

**Alisher Beisembekov** 🇰🇿

## License

MIT License
