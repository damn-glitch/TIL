# TIL IDE

A full-featured Integrated Development Environment for the TIL programming language.

**Author:** Alisher Beisembekov

## Features

- 🎨 **Syntax Highlighting** - Full TIL language support with Dracula theme
- 📝 **Code Editor** - Line numbers, auto-indent, current line highlight
- 📁 **File Explorer** - Browse and open project files
- 📑 **Tabbed Editing** - Work with multiple files
- ▶️ **Run & Build** - Execute TIL code with F5/F6/F7
- 📊 **Output Console** - See compilation results and program output
- 🌙 **Dark Theme** - Beautiful Dracula-inspired dark theme

## Screenshot

```
┌─────────────────────────────────────────────────────────────────┐
│ File  Edit  Run  Help                    📄 New 📂 Open ▶️ Run  │
├──────────┬──────────────────────────────────────────────────────┤
│ EXPLORER │  main.til                                            │
│          │ ┌─────────────────────────────────────────────────┐  │
│ 📁 src   │ │  1 │ # Hello World in TIL                       │  │
│ 📄 main  │ │  2 │                                             │  │
│          │ │  3 │ main()                                      │  │
│          │ │  4 │     print("Hello, World!")                  │  │
│          │ │  5 │                                             │  │
│          │ └─────────────────────────────────────────────────┘  │
│          ├──────────────────────────────────────────────────────┤
│          │ OUTPUT                                               │
│          │ ▶ Running: main.til                                  │
│          │ ──────────────────────────                           │
│          │ Hello, World!                                        │
│          │ ✓ Execution completed successfully                   │
└──────────┴──────────────────────────────────────────────────────┘
```

## Installation

### Requirements
- Python 3.8+
- PyQt6
- TIL Compiler

### Quick Start

**Windows:**
```batch
cd ide
pip install PyQt6
python til_ide.py
```

**Linux/macOS:**
```bash
cd ide
pip3 install PyQt6
python3 til_ide.py
```

Or use the launcher scripts:
- Windows: Double-click `til-ide.bat`
- Linux/macOS: Run `./til-ide.sh`

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New file |
| `Ctrl+O` | Open file |
| `Ctrl+S` | Save file |
| `Ctrl+Shift+S` | Save as |
| `Ctrl+Shift+O` | Open folder |
| `F5` | Run |
| `F6` | Build |
| `F7` | Check syntax |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+C` | Copy |
| `Ctrl+V` | Paste |
| `Ctrl+X` | Cut |

## Project Structure

```
ide/
├── til_ide.py          # Main IDE application
├── til-ide.bat         # Windows launcher
├── til-ide.sh          # Linux/macOS launcher
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Theme

The IDE uses a Dracula-inspired dark theme with the following colors:

| Element | Color |
|---------|-------|
| Background | #282a36 |
| Foreground | #f8f8f2 |
| Keywords | #ff79c6 (Pink) |
| Types | #8be9fd (Cyan) |
| Functions | #50fa7b (Green) |
| Strings | #f1fa8c (Yellow) |
| Numbers | #bd93f9 (Purple) |
| Comments | #6272a4 (Gray) |

## License

MIT License - See LICENSE file in the root directory.

## Links

- **Website:** https://til-dev.vercel.app
- **GitHub:** https://github.com/damn-glitch/TIL
- **VS Code Extension:** Search "TIL Language" in VS Code
