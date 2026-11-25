# TIL Studio — design of the TIL IDE

Internal engineering spec. Every part below is built against this document;
where it fixes a name, a message shape, a file or an endpoint, implementations
follow it exactly so that independently developed parts fit together.

TIL Studio replaces the old PyQt6 IDE (`ide/til_ide.py`: regex highlighting,
no language intelligence, needs PyQt6). It is one IDE with three faces:

| Face | Start | Backend |
|---|---|---|
| **Desktop** (full power) | `til ide [FOLDER]` → opens the browser at `http://127.0.0.1:PORT/` (app window when Chrome/Edge is available) | `src/tilstudio.py`: real files, native/VM/C-VM runs, debugger, prover, fit, packages |
| **Web** | `website/studio/` (static hosting) | in the browser: `tilvm.wasm` + `tilc.tbc` (the playground runtime), files in browser storage |
| **Any editor** | `til lsp` / `til dap` | Language Server Protocol + Debug Adapter Protocol: VS Code, Neovim, Helix, Zed, Emacs, Sublime |

## 0. Ground rules

* **Zero install.** Python 3.9+ standard library only on the server side (no
  pip packages; z3 / a C compiler are optional and detected). The frontend is
  ONE self-contained file `ide/studio.html` (all JS/CSS inline, no CDN, works
  offline). It is built from `ide/studio/` sources with esbuild (dev-time only;
  `ide/studio/package.json` + lockfile committed, `node_modules` not), and the
  built file is committed so users never need node.
* **One source of truth.** Keywords (all languages: en, kk, kk-latn, ru),
  builtins (`NATIVES`, `NATIVE_ARITY`), methods (`METHODS`) come from
  `src/til.py`; the frontend embeds a generated `tables.json`
  (`ide/studio/gen_tables.py`) and the server also serves it at
  `/api/info` (a test checks they agree).
* **Real compiler, not regexes.** Every diagnostic, completion, hover, rename
  … comes from `src/til.py`'s lexer/parser/checkers (plus `src/tiltypes.py`
  for types) through `src/tills.py`.
* **Localized.** The UI exists in English, Russian and Kazakh (switchable,
  default from the browser language); diagnostics use the same language
  through `src/tilmsg.py` (`localize`).
* **Security** (desktop face): the server binds to 127.0.0.1 only, requires a
  random per-launch token (`?token=` in the opened URL, then an HttpOnly
  SameSite=Strict cookie; API calls also accept `X-TIL-Token`), checks the
  `Host` header (`127.0.0.1:PORT` / `localhost:PORT`) and, for state-changing
  requests, the `Origin` header; every path is confined to the project root
  (`realpath` check, no symlink escapes). Requests without the token → 403.
* **Python 3.9 compatible** (`vermin -t=3.9- --violations`), tests stay green
  (`python3 -m pytest tests/ -q`), new tests per part (below).
* Commit messages end with the session trailer lines; no model names anywhere.

## 1. Files

```
src/tills.py          language services (analysis core, pure functions over til)
src/tillsp.py         LSP server (stdio JSON-RPC; in-process handle() for the studio)   `til lsp`
src/tildap.py         DAP server  (stdio; in-process handle() + event callback)          `til dap`
src/tilstudio.py      studio HTTP server: app, files, LSP/DAP bridges, runs, tools       `til ide`
ide/studio.html       the built single-file frontend (committed)
ide/studio/           frontend sources: src/*.js, styles/*.css, i18n/*.json, gen_tables.py,
                      build.mjs, package.json, package-lock.json, test/*.spec.mjs
ide/README.md         user guide summary (full guide: docs/TIL_STUDIO.md)
website/studio/       web face: index.html (= ide/studio.html) + uses ../playground/tilvm.{js,wasm}, tilc.tbc
editors/vscode/       VS Code extension: LSP client, `til` debug type (DAP), commands
```

`EXTRA_COMMANDS` in `src/til.py` gains `ide` → `tilstudio`, `lsp` → `tillsp`,
`dap` → `tildap` (each module exposes `main(argv) -> int`).

## 2. Language services (`src/tills.py`)

Positions are **LSP positions**: 0-based line, character offset in **UTF-16
code units** (what JS strings and LSP use by default). `tills` converts
to/from the compiler's 1-based line / column (code points) exactly — TIL
sources are often Cyrillic, and astral characters (emoji in strings) must not
shift ranges. Helper `PositionMap(text)` does both directions.

```python
class Workspace:
    def __init__(self, root: Optional[str] = None, locale: str = 'en')
    def open(self, uri: str, text: str, version: int = 0) -> None
    def change(self, uri: str, text: str, version: int) -> None      # full text
    def close(self, uri: str) -> None
    def diagnostics(self, uri) -> List[dict]          # LSP Diagnostic dicts
    def completion(self, uri, line, character) -> List[dict]          # CompletionItem
    def resolve_completion(self, item: dict) -> dict
    def hover(self, uri, line, character) -> Optional[dict]          # Hover (markdown)
    def signature_help(self, uri, line, character) -> Optional[dict]
    def definition(self, uri, line, character) -> List[dict]          # Location
    def references(self, uri, line, character, include_declaration=True) -> List[dict]
    def prepare_rename(self, uri, line, character) -> Optional[dict]
    def rename(self, uri, line, character, new_name) -> Optional[dict]   # WorkspaceEdit
    def document_symbols(self, uri) -> List[dict]     # DocumentSymbol (hierarchical)
    def workspace_symbols(self, query) -> List[dict]
    def folding_ranges(self, uri) -> List[dict]
    def formatting(self, uri, tab_size=4) -> List[dict]   # TextEdit (conservative, see below)
    def inlay_hints(self, uri, rng) -> List[dict]     # inferred types of let/var without annotation, return types
    def code_actions(self, uri, rng, diagnostics) -> List[dict]
    def code_lenses(self, uri) -> List[dict]          # "▶ Run", "Debug", "Prove", per function / main
    def semantic_tokens(self, uri) -> dict            # {'data': [...]} full document (legend in tills.TOKEN_TYPES)
```

* **Diagnostics** come from the real pipeline, collected structurally (not by
  parsing formatted text): lexer/parser `SyntaxError` (line/col), the
  SemanticChecker's list (E-EFFECT, E-CAP, E-CONTRACT, E-ARITY), the
  BytecodeCompiler (E-VM, first error), E-IMPORT, E-TYPE (`tiltypes`, when
  available), type warnings. `code` = the error code (e.g. `E-VM`), `source`
  = `"til"`, `message` localized, hint in `data.hint` and appended to the
  message on a new line. Range: the token at the reported position (whole
  identifier/word), never an empty range. Imports are resolved relative to
  the document (packages through `til_modules/` as the compiler does); a
  diagnostic located in an imported file is reported on that file's URI and,
  on the import line of the opened file, as an Information diagnostic.
  Analysis is incremental-enough: re-run on change with the newest text only
  (the studio debounces 250 ms), cache per (uri, version), and must handle
  *broken* code gracefully (completion and outline still work from the last
  good parse plus a token scan).
* **Completion**: keywords of the document's keyword language(s) (a
  `#[lang: ru]` file gets Russian keywords first; Kazakh always available),
  builtins with signatures (`print(...)`, arity from `NATIVE_ARITY`, docs),
  user functions / structs / enums / constants / globals / locals in scope,
  struct fields and methods after `.` (from inferred types when known, else all
  methods of user structs + builtin methods by receiver kind), enum variants
  after `Enum.`, module members after `module.`, import names after `import`
  / `from`, attribute names inside `#[` (`pure`, `effects`, `cap`, `requires`,
  `ensures`, `level`, `energy`, `lang` …), snippets (function, struct, impl,
  match, for, while, retry/timeout, handle/with). Each item: `label`, `kind`,
  `detail` (signature/type), `documentation` (markdown, localized), optional
  `insertText` + `insertTextFormat: 2` (snippet).
* **Hover**: for an identifier: its kind and signature/type (inferred by
  `tiltypes` when available: `x: int`, `f(a: int, b) -> str`), the doc
  comment above its definition (`# ...` lines), contracts (`#[requires]`)
  and effects; for builtins and keywords: built-in documentation (en/ru/kk),
  including the keyword's translations (`if` = `егер` = `eger` = `если`).
* **Rename / references / definition** are scope-correct (locals, params,
  globals, struct fields, methods, enum variants; across the files of the
  workspace that import each other). Rename refuses keywords, builtins and
  names that would clash (returns an error message).
* **Formatting** is conservative and never changes meaning: normalize
  indentation to the indent unit (spaces), strip trailing whitespace, one final
  newline, collapse >2 blank lines, single space around binary operators /
  after commas **outside strings and comments** — implemented on the token
  stream and verified by re-tokenizing (identical token sequence) or it
  returns no edits.
* **Code actions**: quick fixes derived from hints: `let` assigned twice →
  "Change to `var`"; unknown name with a close match → "Did you mean …"; missing
  import of a known module → add import; E-EFFECT on `#[pure]` → remove
  `#[pure]` / add `#[effects: io]`; "Translate file to …" (en/kk/kk-latn/ru via
  `translate_source`) as a refactor action.
* **Code lenses** (above `main` / script start and each function with
  contracts): `Run`, `Debug`, `Prove` with commands `til.run`, `til.debug`,
  `til.prove` and arguments `{uri, function}`.

Tests: `tests/test_tills.py` (unit, many programs incl. Kazakh/Russian
keywords, UTF-16 positions with emoji, broken code) and `tests/test_lsp.py`
(protocol over a real `til lsp` subprocess).

## 3. LSP server (`src/tillsp.py`)

* `til lsp` — stdio, `Content-Length` framing, JSON-RPC 2.0. Options:
  `--tcp PORT` (127.0.0.1), `--log FILE`.
* Implements: `initialize` (capabilities below, `positionEncoding: utf-16`,
  `serverInfo {name: "til-lsp", version}`; `locale` → message language),
  `initialized`, `shutdown`, `exit`, `textDocument/didOpen|didChange|didSave|didClose`
  (full sync), `publishDiagnostics` (debounced 200 ms on change),
  `completion` (+ `completionItem/resolve`), `hover`, `signatureHelp` (trigger
  `(`, `,`), `definition`, `references`, `documentHighlight`,
  `prepareRename`, `rename`, `documentSymbol`, `workspace/symbol`,
  `foldingRange`, `formatting`, `inlayHint`, `codeAction`, `codeLens`,
  `semanticTokens/full`, `workspace/executeCommand` (`til.translate`,
  `til.run`-style commands return `null` — clients handle running),
  `$/cancelRequest` (ignored safely). Unknown requests → MethodNotFound.
* In-process API for the studio:

```python
class LspServer:
    def __init__(self, root: Optional[str] = None)
    def handle(self, message: dict) -> List[dict]   # responses AND notifications to send now
    def poll(self) -> List[dict]                     # notifications that became due (debounced diagnostics)
```

## 4. Debug adapter (`src/tildap.py`)

Built on `src/tildebug.py`'s `DebugVM` (refactor it so the debugger logic is
usable programmatically; keep `til debug` working and its tests green).

* `til dap` — stdio DAP (`Content-Length` framing).
* Capabilities: `supportsConfigurationDoneRequest`, `supportsStepBack`
  (**time travel: VS Code shows Step Back / Reverse Continue**),
  `supportsEvaluateForHovers`, `supportsConditionalBreakpoints`,
  `supportsHitConditionalBreakpoints` (optional), `supportsTerminateRequest`,
  `supportsLoadedSourcesRequest: false`.
* Requests: `initialize`, `launch` (`program` path, `args`, `stopOnEntry`,
  `noDebug`, `cwd`, `stdin` (pre-recorded input text), `keys` (key script),
  `lang`), `setBreakpoints` (lines + conditions), `setExceptionBreakpoints`
  (filter `runtime-error`: stop on uncaught runtime errors), `configurationDone`,
  `threads` (one: `{id: 1, name: "main"}`), `stackTrace`, `scopes` (`Locals`,
  `Globals`), `variables` (arrays/maps/structs/closures expandable,
  `variablesReference` stable while stopped), `continue`, `next`, `stepIn`,
  `stepOut`, `stepBack`, `reverseContinue`, `pause`, `evaluate` (names,
  `a.b`, `a[0]`; contexts `hover`/`watch`/`repl`), `terminate`, `disconnect`.
* Events: `initialized`, `stopped` (`reason`: `entry`/`breakpoint`/`step`/
  `pause`/`exception`; `allThreadsStopped: true`), `continued`, `output`
  (`category: stdout|stderr`), `terminated`, `exited` (`exitCode`).
* Custom (prefixed `til/`): request `til/timeline` → `{index, total}` (line
  events so far / recorded), request `til/goto` `{index}` → jump to any
  recorded line event (the studio's timeline scrubber), request `til/stdin`
  `{text}` (interactive input; the program waits in `input()` and the adapter
  emits event `til/inputRequested`), event `til/screen` `{w, h, rgb_b64}` on
  `screen_show()` while debugging graphics programs.
* In-process API for the studio:

```python
class DapSession:
    def __init__(self, emit: Callable[[dict], None], root: Optional[str] = None)
    def handle(self, message: dict) -> List[dict]   # responses (events go through emit, from any thread)
    def close(self) -> None
```

Tests: `tests/test_dap.py` (real `til dap` subprocess: launch, breakpoints,
next/stepIn/stepOut, stepBack/reverseContinue restore exactly the earlier
state, variables expansion, evaluate, output not duplicated after going back,
exceptions filter, input, terminate).

## 5. Studio server (`src/tilstudio.py`)

`til ide [FOLDER] [--port N] [--no-browser] [--lang en|ru|kk]`. FOLDER
defaults to the current directory (a project root: may contain `til.json`).
Uses `http.server.ThreadingHTTPServer`. Prints the URL with the token. Opens
the browser (`webbrowser`; tries Chrome/Edge `--app=URL` for an app window
first, unless `--no-browser`). Serves `ide/studio.html` (found next to the
installed files like `find_vm_sources`: repo layout `../ide/studio.html` or
`~/.til/ide/studio.html`).

All API responses are JSON (`application/json; charset=utf-8`); errors:
`{"error": "message"}` with 4xx/5xx. Paths in the API are **POSIX paths
relative to the root**.

| Method & path | Body / query | Result |
|---|---|---|
| `GET /` | `?token=` | the app (sets the cookie) |
| `GET /api/info` | | `{version, root, rootName, platform, lang, capabilities: {native, cvm, z3, debugger: true, pkg: true}, devices: [{name, title}], examples: [{name, path}], tables: {...same as tables.json}}` |
| `GET /api/tree` | `?path=` (dir, default root) | `{entries: [{name, path, type: "dir"|"file", size, mtime}]}` (sorted dirs first; hides `.git`, `__pycache__`, `til_modules` content collapsed) |
| `GET /api/file` | `?path=` | `{path, text, mtime, size}` (UTF-8; binary → 415) |
| `PUT /api/file` | `{path, text, mtime?}` | `{mtime}`; 409 if `mtime` given and the file changed on disk |
| `POST /api/file/create` | `{path, type: "file"|"dir", text?}` | `{path}` |
| `POST /api/file/rename` | `{path, to}` | `{path}` |
| `POST /api/file/delete` | `{path}` | `{}` (deletes; the UI asks for confirmation first) |
| `POST /api/search` | `{query, regex?, case?, glob?}` | `{matches: [{path, line, col, text}], truncated}` (max 2000) |
| `POST /api/lsp` | one JSON-RPC message | `{messages: [...]}` (responses + notifications) |
| `POST /api/run` | `{path? , source?, backend: "native"|"vm"|"cvm", args: [], stdin?: str, lang?, maxSteps?, keys?}` | `{runId}` |
| `POST /api/run/{id}/stdin` | `{text}` | `{}` |
| `POST /api/run/{id}/key` | `{code}` | `{}` |
| `POST /api/run/{id}/stop` | | `{}` |
| `POST /api/run/{id}/snapshot` | | `{path}` (vm backend: pause + save `.tilp` next to the program) |
| `POST /api/dap/start` | | `{sessionId}` |
| `POST /api/dap/{id}` | one DAP message | `{messages: [...]}` |
| `POST /api/tools/prove` | `{path, function?}` | `{available: bool, results: [{function, line, status: "proved"|"counterexample"|"unsupported"|"unknown"|"not proved", detail, counterexample?: {...}}]}` |
| `POST /api/tools/fit` | `{path, device, run?: bool}` | the `til fit --json` report |
| `POST /api/tools/dis` | `{path, fuse?: true}` | `{text, functions: [{name, start, end}], lines: [[pc, line]], instructions: [{pc, func, op, args, note, line}]}` |
| `POST /api/tools/translate` | `{path? , source?, to}` | `{text}` |
| `POST /api/tools/build` | `{path, target: "vm"|"c-vm"|"native", device?}` | `{ok, output, artifact}` |
| `POST /api/pkg` | `{argv: [...]}` (`init`, `add`, `install`, `remove`, `list`, `verify`) | `{rc, stdout, stderr, manifest}` |
| `GET /api/events` | SSE stream | events below |

SSE events (`event:` name, `data:` JSON): `lsp` (one LSP message),
`run` `{runId, kind: "stdout"|"stderr"|"status"|"exit"|"screen"|"input", text?, code?, w?, h?, rgb?}`
(`screen`: base64 RGB24 frame on every `screen_show()`, coalesced to ≤ 30 fps;
`input`: the program waits for a line), `dap` `{sessionId, message}`, `fs`
`{paths: [...]}` (files changed on disk: polling every 1 s on open files and
the tree), `ping` every 15 s.

Runs: `vm` runs in-process on the reference VM in a thread (live screen, keys,
stdin); `cvm` and `native` run subprocesses (`til run --cvm|native`) with
stdout/stderr streamed (line-buffered → chunks ≤ 50 ms), stdin from
`/stdin`, killed on stop; programs that use the screen run on `vm`
automatically. Output larger than 4 MB per run is cut with a notice. At most
4 concurrent runs.

Tests: `tests/test_studio.py` — start the server on a free port in a temp
project, token/Host/Origin/path-traversal checks, files API, search, LSP
bridge (diagnostics arrive as SSE), runs on all backends with streamed
output, stdin, stop, graphics frames, DAP bridge round trip, tools endpoints.

## 6. Frontend (`ide/studio/` → `ide/studio.html`)

Built with CodeMirror 6 (MIT; license text kept in the bundle header) and
esbuild. Vanilla JS modules otherwise (no framework). Target: evergreen
browsers (Chrome, Edge, Firefox, Safari latest 2 versions).

**Layout** (VS-Code-like but calmer):
title bar (logo, project name, command palette search box, run/debug buttons,
backend picker native/VM/C VM, device picker, theme toggle) · activity bar
(Explorer, Search, Run & Debug, Problems, Proofs, Devices, Packages,
Bytecode, Settings) · side panel · editor area (tabs, split right, breadcrumbs,
minimap-free) · bottom panel (Output/Terminal, Problems, Debug console,
Screen) · status bar (Ln/Col, indent, keyword language, UI language, backend,
errors/warnings count, TIL version).

**Editor** (CodeMirror): TIL language (stream parser over the generated
keyword tables — all keyword languages, f-strings, attributes `#[...]`,
numbers, durations `5s`, comments), indentation-based folding, smart
auto-indent (indent after block openers, dedent on `else/elif/with/…`),
bracket matching/closing, multiple cursors, search/replace, go-to-line,
autocompletion (LSP), lint (LSP diagnostics with hover), hover tooltips (LSP
markdown rendered safely — no raw HTML), signature help, inlay hints, code
lenses (Run/Debug/Prove above functions), F12 definition, Shift+F12
references (peek list), F2 rename, Shift+Alt+F format, Ctrl+. code actions,
breakpoints in the gutter (click), current debug line highlight, error lens
(diagnostic text at end of line, toggle).

**Panels**: Explorer (tree, new/rename/delete, drag-free), Search (workspace
text search + replace in files), Run & Debug (configs: backend, args, stdin
script, keys; Debug toolbar: Continue, Step Over/Into/Out, **Step Back,
Reverse Continue**, Stop; Variables (expandable), Watch, Call Stack,
Breakpoints; **timeline scrubber** — a slider over all recorded line events
with the current position; dragging jumps there (`til/goto`)), Problems
(all files), Proofs (per function: ✓ proved / ✗ counterexample with values
and a "Debug with these inputs" button / unsupported reason; gutter marks),
Devices (pick a device: memory bars static vs measured, verdict, warnings
list linked to lines; "Run on device VM" = `fit --run`), Packages (til.json
dependencies list, add/remove/install/verify), Bytecode (disassembly of the
current file, synchronized with the cursor: hovering a line highlights its
instructions and vice versa; fused superinstructions marked), Screen (the
program's framebuffer, scaled crisp, keyboard focus sends keys; a
calculator-style frame option), Settings (theme, font size, UI language,
keyword language for new files, error lens, format on save, auto save).

**Command palette** (Ctrl+Shift+P / F1) with every command; quick open
(Ctrl+P) with fuzzy file search; Ctrl+Shift+F search; Ctrl+` bottom panel;
F5 debug / Ctrl+F5 run; F9 breakpoint; F10/F11/Shift+F11 step; Shift+F10
step back (configurable keymap).

**Translate view**: "View in Kazakh/Russian/English" opens a read-only
side-by-side translation of the current file (`/api/tools/translate`), and
"Translate file" replaces it (undoable).

**Design**: dark theme (default) and light theme, both WCAG AA contrast. Brand
palette from the website: accent `#00d4ff`, green `#4ade80`, yellow
`#fbbf24`, red `#f87171`, purple `#a78bfa`, orange `#fb923c`; dark
backgrounds `#0b0d12` / `#11141b` / `#171b24` (subtle blue-black), borders
`#232836`; light: `#ffffff` / `#f6f7f9` / `#eef0f4`, text `#1b1f2a`. Syntax
colors harmonized with the palette (keywords purple, functions cyan, strings
green, numbers orange, types yellow, comments muted gray-blue italic,
attributes `#[...]` pink-purple). UI font: system UI stack 13 px; code font:
`"JetBrains Mono", "Cascadia Code", "Fira Code", Menlo, Consolas, monospace`
14 px with ligatures off by default. Inline SVG icons (stroke 1.5, 16 px).
Rounded 6 px corners, 1 px borders, no heavy shadows, 120 ms transitions,
focus rings visible. Empty states and a Welcome page (recent folders,
examples gallery with previews, "Learn TIL" cards in the UI language,
keyboard shortcuts). Responsive down to 1024 px wide; on narrow widths the
side panel overlays.

**i18n**: `ide/studio/i18n/{en,ru,kk}.json` — every UI string; tests check
the three files have identical keys and no empty values.

**Persistence** (localStorage, try/catch): open tabs and cursor positions per
project, layout sizes, theme, UI language, settings, breakpoints.

**Web face**: if `/api/info` is missing (static hosting), the same app uses a
browser backend: files in IndexedDB (a starter project with examples), runs
via `../playground/tilvm.js` (wasm VM, compile with `tilc.tbc` in VM A, run in
VM B, screen and keys), diagnostics from tilc's compile errors (debounced),
completion/hover from the static tables plus a symbol scan of the document;
features that need the desktop server (debugger, prover, fit, packages,
native) show a clear "Available in TIL Studio desktop: `til ide`" card.

Tests: `ide/studio/test/studio.spec.mjs` (Playwright + real server in a temp
project: open, edit with completion, diagnostics appear, quick fix, rename,
run each backend, input, graphics screen + keys, debug with breakpoint, step
back, timeline jump, proofs panel, device panel, bytecode sync, theme and UI
language switch, command palette, persistence across reload; web face against
a static server) + screenshots saved to `ide/studio/test/screenshots/`
(not committed) for visual review; `tests/test_studio_ui.py` runs it when node
+ Playwright are available.

## 7. VS Code extension (`editors/vscode/`)

Version 3.1.0: language client for `til lsp` (bundled `vscode-languageclient`
via esbuild into `dist/extension.js`; `node_modules` not committed), debug
type `til` using `til dap` (launch config snippets; Step Back enabled), commands
Run / Run on VM / Run on C VM / Prove / Fit for device / Translate / Open in
TIL Studio, updated TextMate grammar (all keyword languages generated from
`til.py`), snippets. Packaged with `vsce` if available (the `.vsix` is not
committed).

## 8. Installers and docs

* `install.sh` / `install.ps1` (+ `website/` copies) fetch `src/tills.py`,
  `src/tillsp.py`, `src/tildap.py`, `src/tilstudio.py` and `ide/studio.html`
  (into `~/.til/ide/`).
* `docs/TIL_STUDIO.md`: user guide with screenshots (PNG in `docs/images/`,
  taken by the Playwright test at 1440×900, dark + light), editor setup for
  VS Code / Neovim / Helix / Sublime with `til lsp` and `til dap`.
* README: a "TIL Studio" section near the top.
