# TIL Language for Visual Studio Code

**TIL: Simpler than Python. Faster than C. Smarter than all.**

*by Alisher Beisembekov*

## Features

- 🎨 **Syntax Highlighting** - Full support for TIL syntax
- ⚡ **Code Snippets** - Quick templates for common patterns
- 🔨 **Build Commands** - Compile and run with keyboard shortcuts
- 📝 **Language Configuration** - Auto-closing brackets, comments, indentation

## Installation

1. Install from VS Code Marketplace (search "TIL Language")
2. Or download `.vsix` from [GitHub Releases](https://github.com/damn-glitch/TIL/releases)

## Requirements

- **Python 3.8+** - For the TIL compiler
- **GCC/Clang** - For C compilation

Install TIL compiler:
```bash
# Linux/macOS
curl -fsSL https://til-dev.vercel.app/install.sh | sh

# Windows (PowerShell)
irm https://til-dev.vercel.app/install.ps1 | iex
```

## Keyboard Shortcuts

| Shortcut | Command |
|----------|---------|
| `F5` | Run current file |
| `F6` | Build current file |
| `F7` | Build and run |

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

## Links

- [Website](https://til-dev.vercel.app)
- [GitHub](https://github.com/damn-glitch/TIL)
- [Documentation](https://github.com/damn-glitch/TIL/tree/main/docs)

## Author

**Alisher Beisembekov** 🇰🇿

## License

MIT License
