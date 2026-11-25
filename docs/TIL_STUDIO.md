# TIL Studio

TIL Studio is the IDE for TIL. It runs in your browser, needs only Python
3.9+ (no pip packages, works offline), and uses the real TIL compiler for
every diagnostic, completion and rename — not regular expressions.

![TIL Studio: editor](images/studio-editor-dark.png)

## Start

```bash
til ide                 # opens the current folder
til ide path/to/project # any folder; a til.json makes it a package project
til ide --lang kk       # interface and messages in Kazakh (en, ru, kk)
til ide --port 8800 --no-browser
```

`til ide` prints a URL such as `http://127.0.0.1:41723/?token=…` and opens it
— as a separate app window when Chrome, Edge or Chromium is installed, else
in your default browser. Without installing TIL you can start it from a
checkout with `ide/til-ide.sh` (Linux/macOS) or `ide\til-ide.bat` (Windows).

**On the web**: the same app runs on the website (`website/studio/`) with the
compiler (`tilc`, the TIL compiler written in TIL) and the VM compiled to
WebAssembly — nothing is installed, your files stay in the browser.
Debugging, proofs, devices, packages and native builds need the desktop
version.

## The window

| Area | What is there |
|---|---|
| Title bar | menus, the search box (files and commands), backend picker (Native · VM · C VM), device picker, Run, Debug, theme |
| Activity bar | Explorer, Search, Run & Debug, Problems, Proofs, Devices, Packages, Bytecode, Settings |
| Editor | tabs, split view, breadcrumbs, code lenses (▶ Run · Debug · Prove) above functions |
| Bottom panel | Output (program output and input), Problems, Debug console, Screen |
| Status bar | position, indentation, keyword languages of the file, backend, UI language, errors/warnings, TIL version |

## Editing

* **Three keyword languages.** English, Kazakh and Russian keywords are
  highlighted and completed (`#[lang: ru]` files get Russian keywords first).
  *Translate view* shows the same program side by side in another language;
  *Translate file* rewrites it (undoable).
* **Completion** for keywords, builtins with their signatures, your
  functions, structs, fields and methods after `.`, enum variants, module
  members, attributes inside `#[...]`, and snippets.
* **Errors as you type** from the compiler itself: syntax, effects,
  capabilities, contracts, arity, types (`E-TYPE`), imports — localized into
  the UI language, with the hint below the message. `F8` / `Shift+F8` jumps
  between problems.
* **Hover** shows the inferred type or signature, the comment above the
  definition, contracts and effects; for keywords and builtins, built-in
  documentation with the keyword in all languages.
* **Inline type hints** after `let`/`var` without an annotation.
* **Refactoring**: `F2` rename (scope-correct, across files that import each
  other), `F12` go to definition, `Shift+F12` references, `Ctrl+.` quick
  fixes (`let` → `var`, did-you-mean, add import, effects), `Shift+Alt+F`
  format (a conservative formatter that never changes what the program does).
* Outline (`Ctrl+Shift+O`), workspace symbols (`Ctrl+T`), folding by
  indentation, multi-cursor, search and replace (`Ctrl+F`, `Ctrl+H`),
  search in files (`Ctrl+Shift+F`), quick open (`Ctrl+P`), command palette
  (`Ctrl+Shift+P` / `F1`).

## Running

`Ctrl+F5` (or `Ctrl+Enter`) runs the current file on the backend chosen in the
title bar:

* **Native** — compiled to C and run (falls back to the VM for programs the
  native backend does not cover, exactly like `til run`).
* **VM** — the reference VM: live graphics, keys, input, snapshots.
* **C VM** — the portable C VM (the one that runs on calculators).

Program output appears in **Output**; when the program calls `input()` the
input line at the bottom is focused (`Ctrl+D` ends the input). Runtime errors
link to their line. Programs that use the screen run on the VM and draw into
**Screen** — optionally inside a calculator frame — and receive the keys you
press there (arrows, Enter, Esc, letters). *Snapshot* pauses a VM run and
saves a portable `.tilp` file that you can resume on another TIL VM — even on
a calculator ([TIL_SNAPSHOTS.md](TIL_SNAPSHOTS.md)).

## Debugging with time travel

![TIL Studio: debugger](images/studio-debug-dark.png)

`F5` starts the debugger; click the gutter or press `F9` for a breakpoint
(right click: condition or hit count).

| Key | Action |
|---|---|
| `F5` | start / continue |
| `F10` / `F11` / `Shift+F11` | step over / into / out |
| `Shift+F10` | **step back** |
| `Alt+F5` | **reverse continue** — back to the previous breakpoint hit |
| `Shift+F5` | stop |

The **timeline** in Run & Debug shows every line the program has executed;
drag it to jump to any moment — variables, call stack, output and the screen
follow. Watch expressions and the debug console evaluate TIL expressions
(`a.b`, `xs[i]`, arithmetic, pure builtins) at the current moment. Input that
the program reads is recorded, so going back and forth never asks twice and
output is never printed twice.

## Proofs, devices, packages, bytecode

* **Proofs** — runs `til prove` (Z3) on the file: ✓ proved, ✗ counterexample
  (with the values; *Debug with these inputs* starts the debugger on them),
  or the reason a function is unsupported. Install Z3 with
  `pip install z3-solver`. See [TIL_PROOFS.md](TIL_PROOFS.md).
* **Devices** — will the program fit a TI-84 Plus CE, NumWorks, TI-Nspire,
  Casio, Cortex-M, ATmega, Arduino Mega, ESP32, RP2040 or a Z80 calculator?
  Memory bars (bytecode, VM RAM, stack, frames, heap), warnings linked to the
  lines, and *Run on the device VM* for measured high-water marks.
  See [TIL_FIT.md](TIL_FIT.md).
* **Packages** — the project's `til.json` dependencies: add (path, git,
  archive, registry), install, remove, verify. See [TIL_PACKAGES.md](TIL_PACKAGES.md).
* **Bytecode** — the disassembly of the current file next to the source; the
  cursor in one highlights the other; superinstructions are marked.

## Settings

Theme (dark / light), editor font size, UI language (English, Русский,
Қазақша), keyword language for new files, error lens (messages at the end of
the line), format on save, auto save, keybindings (every command can be
rebound). Settings, open tabs and breakpoints are remembered per project.

## Keyboard shortcuts

| Keys | Command |
|---|---|
| `Ctrl+Shift+P`, `F1` | command palette |
| `Ctrl+P` | go to file |
| `Ctrl+Shift+O` / `Ctrl+T` | symbol in file / in workspace |
| `Ctrl+G` | go to line |
| `Ctrl+S` / `Ctrl+Alt+S` | save / save all |
| `Ctrl+B` / ``Ctrl+` `` | toggle side bar / bottom panel |
| `Ctrl+\` | split editor |
| `Alt+Shift+T` | dark / light theme |
| `Ctrl+=` / `Ctrl+-` / `Ctrl+0` | zoom |
| `Ctrl+K` | all shortcuts |

On macOS `Cmd` replaces `Ctrl`. Shortcuts follow the physical keys, so they
also work with Russian and Kazakh keyboard layouts.

## Security

The server listens on `127.0.0.1` only and requires the random token from the
URL it printed (kept in an HttpOnly cookie after the first load). Requests
from other web sites are refused (Host, Origin and Fetch-Metadata checks), and
every file operation stays inside the project folder (symbolic links that lead
outside are refused). Stop the server with `Ctrl+C`.

## Other editors

The language intelligence and the time-travel debugger are also available as
standard protocols:

* `til lsp` — Language Server Protocol ([TIL_LSP.md](TIL_LSP.md): VS Code,
  Neovim, Helix, Sublime Text, Emacs).
* `til dap` — Debug Adapter Protocol with Step Back / Reverse Continue
  ([TIL_DEBUGGER.md](TIL_DEBUGGER.md#debug-adapter-dap)).
* The VS Code extension in `editors/vscode/` uses both
  ([README](../editors/vscode/README.md)).

## For developers

| Part | File |
|---|---|
| Language services (diagnostics, completion, rename, …) | `src/tills.py` |
| Language server | `src/tillsp.py` |
| Debug adapter / debugging engine | `src/tildap.py`, `src/tildebug.py` |
| Studio server (HTTP + server-sent events) | `src/tilstudio.py` |
| Frontend (CodeMirror 6, plain ES modules) | `ide/studio/src/` → `ide/studio.html` |
| Web version | `website/studio/index.html` (same page) + `website/playground/` runtime |

Rebuild the frontend with `cd ide/studio && npm ci && node build.mjs` (it
writes `ide/studio.html` and `website/studio/index.html`). Tests:
`tests/test_tills.py`, `tests/test_lsp.py`, `tests/test_dap.py`,
`tests/test_studio.py`, `tests/test_studio_ui.py` (Playwright, end to end
against the real server). Design: [dev/TIL_STUDIO_DESIGN.md](dev/TIL_STUDIO_DESIGN.md).
