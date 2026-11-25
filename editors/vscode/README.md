# TIL for Visual Studio Code

TIL (The Intelligent Language) in VS Code: the real TIL compiler behind every
squiggle, a time-travel debugger with **Step Back** and **Reverse Continue**,
runs on the native backend, the TIL VM and the C VM, the contract prover,
device fit checks for calculators and microcontrollers, and keyword
translation between English, Kazakh (Cyrillic and Latin) and Russian.

This extension is TIL Studio's "any editor" face: it starts `til lsp` (the
language server) and `til dap` (the debug adapter), the same engines that run
TIL Studio (`til ide`).

*by Alisher Beisembekov*

## Requirements

- **TIL 3.1** with **Python 3.9+**. Install it with
  `curl -fsSL https://til-dev.vercel.app/install.sh | sh` (Linux, macOS) or
  `irm https://til-dev.vercel.app/install.ps1 | iex` (Windows), or use a checkout
  of the TIL repository.
- Optional: a C compiler (native runs and builds, the C VM), `pip install z3-solver`
  (the prover).

The extension finds TIL by itself: the `til.path` setting, else `til` on `PATH`,
else the TIL checkout the extension was loaded from, else a TIL checkout opened as
a workspace folder, else `~/.til/bin/til.py`. A `til.py` is run with Python
(`til.python`, else `python3`; on Windows `python` or `py -3`). The status bar
item (bottom right, in `.til` files) shows the server state and TIL's version;
click it for actions.

## Features

**Language server** (`til lsp`; every answer comes from the compiler, in English,
Russian or Kazakh, see `til.locale`):

- Diagnostics while you type: syntax errors, effects (E-EFFECT), capabilities
  (E-CAP), contracts (E-CONTRACT), arity (E-ARITY), the bytecode compiler (E-VM),
  imports (E-IMPORT) and types (E-TYPE), with hints.
- Completion of keywords in the file's keyword language, builtins, names in
  scope, fields and methods after `.`, enum variants, module members, attributes
  inside `#[...]`, snippets.
- Hover with inferred types, doc comments, contracts and the keyword in every
  language (`if` = `егер` = `eger` = `если`); signature help.
- Go to definition, find references, highlight, rename (scope-correct, across
  files that import each other), document and workspace symbols, folding.
- Formatting that never changes meaning, inlay hints (inferred types), semantic
  highlighting.
- Quick fixes: `let` assigned twice, "did you mean", missing imports, effects
  and capabilities; refactoring "Translate file to ...".
- Code lenses: **Run** and **Debug** above `main`, **Prove** above functions with
  `#[requires]` / `#[ensures]`.

**Running.** *TIL: Run File* (native backend; programs that use the screen run on
the VM), *Run on the TIL VM* and *Run on the C VM* run the file in the integrated
terminal, so `input()` and the keys of graphics programs work; graphics
programs draw their screen in the terminal (`--screen-ansi`, see
`til.run.screenAnsi`). The editor's Run menu (the play button) has all three
and Debug.

**Debugging with time travel** (the `til` debug type, `til dap`):

- **F5** debugs the current file (no `launch.json` needed); **Ctrl+F5** runs it
  without debugging (output in the Debug Console).
- Breakpoints with conditions and hit counts, Step Over / Into / Out, the call
  stack, variables (arrays, maps, structs, closures expand), watches, hovers,
  stopping at uncaught runtime errors.
- **Step Back** and **Reverse Continue** on the debug toolbar: the program is not
  re-run, the debugger restores the earlier state exactly and replays the input
  it read, so nothing is printed twice.
- **TIL: Go to Point in Time** (Ctrl+Alt+G, or the history button of the debug
  toolbar): jump to any recorded line event (a number, or `-10` / `+5` relative
  to now).
- The **TIL Time Travel** view in the Run and Debug side bar: a slider over
  every recorded line event, with start / reverse continue / step back / step /
  continue / end buttons.
- **Input**: when the program calls `input()`, an input box asks for the line
  (Escape: type it in the Debug Console instead; the stop button of the box
  ends the input). *TIL: Send Input to the Program* sends a line at any time.
  The `stdin` launch attribute gives the whole input in advance.
- **Screen**: graphics programs (`screen_init`, `draw_*`, `screen_show`) show
  their framebuffer in a **TIL Screen** panel, scaled with crisp pixels (zoom,
  save as PNG). After stepping back the panel shows that moment's frame. Keys
  come from the `keys` launch attribute.

**Prover.** *TIL: Prove Contracts in File* runs `til prove` (Z3): every
`#[ensures]`, `assert` and loop `invariant` is checked for all inputs, every call
against the callee's `#[requires]`. Results go to the TIL output channel;
counterexamples are errors on the failing line, unproved conditions warnings,
unsupported constructs information. *Prove Function at Cursor* and the
**Prove** code lens prove one function.

**Device fit.** *TIL: Fit for Device...* picks a device from `til fit
--list-devices` (TI-84 Plus CE, NumWorks, Casio, Cortex-M, AVR, ESP32, RP2040,
Z80 ...) and shows the report: bytecode, RAM, stack, integer width, missing
builtins. Problems appear on the lines they name. The play button next to a
device also runs the program on that device's C VM build (needs a C compiler).

**Translate keywords.** *TIL: Translate Keywords...* opens the file with its
keywords (and builtin names) in English, Қазақша, Qazaqsha (latyn) or Русский in
a new editor, next to the original.

**Build.** *Build Bytecode (.tbc)* (`til build --target vm`, for any TIL VM),
*Build C VM File (_vm.c)* (one C file with the VM and the program, for
calculators and microcontrollers), *Build Native Executable* (F6), *Check File*
(F7).

**TIL Studio.** *TIL: Open in TIL Studio* starts `til ide` on the workspace
folder (or the folder you right-click in the Explorer): the full TIL IDE in
the browser.

**Syntax highlighting** for the keywords of every keyword language (generated
from the compiler's tables; Russian and Latin-script Kazakh keywords from a
`#[lang: ru]` / `#[lang: kk-latn]` line on), builtins, attributes `#[...]`,
f-strings with `{interpolation}` and format specs, numbers and durations;
indentation rules for TIL's blocks; **snippets** for TIL 3.1: `handle` / `with`
/ `resume`, `retry`, `timeout`, `fnc` (`#[requires]` / `#[ensures]`), `struct`,
`impl`, `trait`, `enum`, `match`, `screen` (a graphics main loop), `energy`.

## Commands

| Command | |
|---|---|
| TIL: Run File · Run on the TIL VM · Run on the C VM | run in the terminal |
| TIL: Debug File | start the debugger (also F5) |
| TIL: Prove Contracts in File · Prove Function at Cursor | `til prove` |
| TIL: Fit for Device... | `til fit` |
| TIL: Translate Keywords... | to en / kk / kk-latn / ru, in a new editor |
| TIL: Build Bytecode (.tbc) · Build C VM File (_vm.c) · Build Native Executable | `til build` |
| TIL: Check File | `til check` |
| TIL: Open in TIL Studio | `til ide FOLDER` |
| TIL: Go to Point in Time... · Send Input to the Program... · Show Program Screen | while debugging |
| TIL: Restart Language Server · Show Output · Language Server Status | |

## Keyboard shortcuts

| Key | |
|---|---|
| F5 | start debugging (the current file without a launch.json) |
| Ctrl+F5 | run without debugging |
| F9 · F10 · F11 · Shift+F11 | breakpoint · step over · step into · step out (VS Code's) |
| Ctrl+Alt+G (Cmd+Alt+G) | go to a point in time (while stopped) |
| F6 | build a native executable |
| F7 | check the file |

Step Back and Reverse Continue are on the debug toolbar (bind
`workbench.action.debug.stepBack` / `workbench.action.debug.reverseContinue` to
keys of your choice).

## Settings

| Setting | Default | |
|---|---|---|
| `til.path` | `""` | the `til` launcher, `src/til.py` of a checkout, or the checkout folder (a relative path starts at the workspace folder; `~` is the home folder); empty: found automatically (see Requirements) |
| `til.python` | `""` | the Python that runs `til.py` (`python3`; Windows `python` / `py -3`); may include arguments |
| `til.lsp.enabled` | `true` | start the language server |
| `til.locale` | `auto` | `en` / `ru` / `kk`: language of diagnostics, hovers and debugger messages (`initializationOptions.locale`, and `--lang` for the tools); `auto`: VS Code's display language (the command-line tools then follow TIL's own rules: `TIL_LANG`, the file's `#[lang: ...]`) |
| `til.trace.server` | `off` | `messages` / `verbose`: log the LSP traffic in *TIL Language Server* |
| `til.run.screenAnsi` | `true` | show the screen of graphics programs in the terminal |
| `til.prove.timeout` | `10000` | solver time limit per proof condition (ms) |
| `til.debug.showScreen` | `true` | open the TIL Screen panel when a debugged program draws |
| `til.optimization` | `-O2` | optimization level of native builds |
| `til.compilerPath` | `til` | deprecated: use `til.path` |

In Restricted Mode (untrusted workspaces) the workspace values of `til.path` and
`til.python` are ignored and no TIL checkout is taken from the workspace.

## Debugging: launch.json

F5 needs no configuration. For more, *Run and Debug → create a launch.json
file → TIL*, or add a configuration with *Add Configuration...*:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "til",
      "request": "launch",
      "name": "Debug the current TIL file",
      "program": "${file}",
      "cwd": "${workspaceFolder}",
      "args": [],
      "stopOnEntry": false
    },
    {
      "type": "til",
      "request": "launch",
      "name": "Debug the game with input and keys",
      "program": "${workspaceFolder}/game.til",
      "stdin": "Aigerim\n42\n",
      "keys": "RIGHT,RIGHT,ENTER,ESC",
      "stopAtEnd": true,
      "lang": "kk"
    }
  ]
}
```

| Attribute | |
|---|---|
| `program` | the `.til` file (or `.tbc` bytecode) |
| `args` | arguments for `args()` (a list, or one string) |
| `cwd` | where relative file paths of the program start |
| `stdin` | the program's whole input (without it: the input box) |
| `keys` | scripted key presses: `"UP,UP,ENTER,a,ESC"` or a list of key codes |
| `lang` | `en` / `ru` / `kk`: the debugger's messages (default: `til.locale`) |
| `stopOnEntry` | stop at the first line |
| `stopAtEnd` | stop at the end instead of ending the session, to step back from there |
| `noFuse` | compile without superinstructions |
| `snapshotEvery` | line events between time-travel snapshots |

The debugger (`til dap`) is described in the repository's
`docs/TIL_DEBUGGER.md`; the language server in `docs/TIL_LSP.md`, with setups
for Neovim, Helix, Sublime Text, Emacs and Zed.

## Installing

**From a .vsix.** In a TIL checkout:

```bash
cd editors/vscode
npx @vscode/vsce package          # -> til-lang-3.1.0.vsix (uses the committed dist/extension.js)
code --install-extension til-lang-3.1.0.vsix
```

or *Extensions → ... → Install from VSIX...*.

**From the repository folder, without packaging.** The built extension is
committed (`dist/extension.js`), so VS Code can load the folder directly:

```bash
code --extensionDevelopmentPath=/path/to/TIL/editors/vscode /path/to/your/project
```

Loaded from the checkout, the extension uses that checkout's `src/til.py` when
no `til` is on `PATH`.

## Development

```bash
cd editors/vscode
npm ci                                  # esbuild, vscode-languageclient, test tools
node scripts/build.mjs                  # src/ -> dist/extension.js (+ ThirdPartyNotices.txt)
node scripts/build.mjs --check          # is the committed bundle up to date?
python3 scripts/gen_grammar.py          # syntaxes/, language-configuration.json, ../vim, ../sublime
npm run typecheck                       # the sources against the VS Code 1.82 API
python3 -m pytest ../../tests/test_vscode_ext.py -v
```

`syntaxes/til.tmLanguage.json`, `language-configuration.json`,
`editors/vim/til.vim` and `editors/sublime/TIL.sublime-syntax` are generated from
the keyword tables of `src/til.py`: change the compiler, then run the script.

## License

Business Source License 1.1, the license of the TIL repository (see `LICENSE`).
`dist/extension.js` bundles `vscode-languageclient` and its dependencies (MIT /
ISC): see `ThirdPartyNotices.txt`.
