# Changelog

All notable changes to the TIL extension for VS Code.

## [3.1.0] - 2026-11

The extension becomes TIL Studio's "any editor" face: the TIL language server
and the time-travel debugger in VS Code. The version follows TIL's.

### Added
- **Language client** for `til lsp` (diagnostics from the real compiler,
  completion, hover with inferred types, signature help, definition, references,
  rename, symbols, folding, formatting, inlay hints, quick fixes, semantic
  tokens, code lenses Run / Debug / Prove). Settings `til.path`, `til.python`,
  `til.lsp.enabled`, `til.locale` (en / ru / kk / auto), `til.trace.server`;
  *TIL: Restart Language Server*; a status bar item with the server state and
  TIL's version.
- **Debug type `til`** on `til dap`: F5 without a launch.json, Ctrl+F5 run
  without debugging, launch configuration snippets, breakpoints (conditions, hit
  counts), **Step Back** and **Reverse Continue**, *TIL: Go to Point in Time*,
  the **TIL Time Travel** view (a slider over the recorded history), an input
  box for `input()` (`til/inputRequested` -> `til/stdin`), the **TIL Screen**
  panel for graphics programs (`til/screen`, crisp scaling, save as PNG).
- Commands: *Run File* (native), *Run on the TIL VM*, *Run on the C VM* (in the
  terminal), *Prove Contracts in File* / *Prove Function at Cursor* (results in
  the output channel and as diagnostics), *Fit for Device...* (device list from
  `til fit --list-devices --json`), *Translate Keywords...* (en, kk, kk-latn,
  ru; through the language server's `til.translate`), *Open in TIL Studio*
  (`til ide`), *Build Bytecode (.tbc)*, *Build C VM File (_vm.c)*.
- The TextMate grammar is generated from the compiler's keyword tables
  (`scripts/gen_grammar.py`, which also writes the Vim and Sublime Text syntax
  files): keywords of all four keyword languages (Russian and Latin-script
  Kazakh from their `#[lang: ...]` pragma on), TIL 3.1's contextual effect words,
  builtins and their localized names, attributes, f-strings with
  interpolation and format specs, durations, numbers.
- Indentation rules for TIL's blocks in every keyword language; folding by
  indentation.
- Snippets for TIL 3.1: `handle` / `with` / `resume`, `retry`, `timeout`,
  contracts, `struct` / `impl` / `trait`, `enum`, `match`, a graphics main loop,
  energy modes.

### Changed
- F5 starts the debugger (VS Code's convention); running is Ctrl+F5 or the Run
  commands. F6 builds a native executable, F7 checks the file.
- The extension is bundled with esbuild into `dist/extension.js` (committed:
  it runs from a checkout without npm). Requires VS Code 1.82 or newer.
- Programs run in the integrated terminal (input and keys work); the
  `til.runInTerminal` setting is gone and `til.compilerPath` is deprecated
  (use `til.path`).

## [1.1.0] - 2025-01-28

### Added
- Run (F5), Build (F6), Check (F7), Run in Terminal (Ctrl+F5)
- Error highlighting from the compiler's output
- Editor context menu and Run button
- Settings for the compiler path and optimization level

## [1.0.0] - 2025-01-27

### Added
- Initial release: syntax highlighting, snippets, language configuration,
  multi-level attributes (`#[level: N]`)
