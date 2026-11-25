# TIL language server (`til lsp`)

`til lsp` is the TIL language server: diagnostics, completion, hover,
go to definition, references, rename, outline, formatting and more in any
editor that speaks the [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)
— VS Code, Neovim, Helix, Zed, Sublime Text, Emacs. TIL Studio (`til ide`)
uses the same server.

Every answer comes from the compiler itself (`src/til.py`: its lexer,
parser, import resolution, effect / capability / contract verifier and
bytecode compiler, plus `src/tiltypes.py` for types), so the editor reports
exactly what `til check` / `til run` would — in English, Russian or Kazakh.

```bash
til lsp                     # stdio (what editors start)
til lsp --tcp 2087          # one client on 127.0.0.1:2087
til lsp --log /tmp/til-lsp.log
python3 path/to/TIL/src/til.py lsp      # without the `til` wrapper (Windows, a checkout)
```

`lsp/til_lsp.py` (the old server's path) still works: it starts `til lsp`.

Contents: [features](#features) · [VS Code](#vs-code) · [Neovim](#neovim) ·
[Helix](#helix) · [Sublime Text](#sublime-text) · [Emacs](#emacs) ·
[Zed and others](#zed-and-other-editors) · [message language](#message-language) ·
[details](#details) · [limitations](#limitations)

## Features

| LSP feature | What TIL does |
|---|---|
| diagnostics | syntax errors, E-EFFECT / E-CAP / E-CONTRACT / E-ARITY (verifier), E-VM (compiler), E-IMPORT (missing modules, one namespace), E-TYPE (type inference); errors inside imported files are shown on those files and noted on the import line; recomputed 200 ms after the last keystroke |
| completion | keywords of the file's language (`#[lang: ru]` → Russian first; Kazakh is always on), builtins with signatures, locals / parameters / functions / structs / enums / constants in scope, fields and methods after `.` (from the inferred type), enum variants after `Enum.`, module members after `module.`, module names after `import` / `from`, attributes inside `#[...]`, snippets |
| hover | kind and signature with inferred types (`let d: float`, `fact(n: int) -> int`), the `#` comment above a definition, contracts and effects; builtins and keywords with documentation and the keyword in every language (`if` = `егер` = `eger` = `если`) |
| signature help | on `(` and `,`, for functions, methods, struct constructors and builtins |
| definition / references / document highlight | scope-correct (locals, parameters, globals, fields, methods, variants), across the files that import each other |
| rename | the same occurrences; refuses keywords, builtins and names that would clash (the editor shows why) |
| document / workspace symbols | functions, structs (+ fields), enums (+ variants), traits, `impl` blocks (+ methods), constants |
| folding | indented blocks, comment blocks, multi-line brackets, import groups |
| formatting | indentation to 4 spaces (the editor's tab size), trailing whitespace, one final newline, at most two blank lines, one space around binary operators and after commas — never inside strings or comments; the result is checked to lex to the very same tokens, otherwise nothing is changed |
| inlay hints | the inferred type of `let` / `var` / `for` variables without an annotation, and the inferred return type of functions without `->` |
| code actions | `let` assigned twice → change to `var`; *did you mean …* → replace; unknown name defined in another file → add the import; `#[pure]` function with effects → remove `#[pure]` / declare `#[effects: io]`; missing effect → add it to `#[effects: ...]`; missing capability → add `#[cap: X]`; **Translate file to** English / Қазақша / Qazaqsha (latyn) / Русский |
| code lenses | `▶ Run` and `Debug` above `main` (or the first line of a script), `Prove` above each function with `#[requires]` / `#[ensures]` |
| semantic tokens | keywords of every language, functions, methods, parameters, variables, fields, structs, enums, variants, modules, types, attributes, strings, numbers, comments |

The code lenses carry the commands `til.run`, `til.debug` and `til.prove`
with the argument `{"uri": ..., "function": ...}`; the editor runs them (the
VS Code extension and TIL Studio do; elsewhere they are just labels — use
`til run`, `til debug`, `til prove` in a terminal).

## VS Code

The TIL extension (`editors/vscode`) starts `til lsp` for `.til` files — install
it and nothing else is needed. To use the server from another LSP client
extension, point it at the command `til lsp` for the language id `til`.

## Neovim

Neovim 0.11 or newer:

```lua
vim.filetype.add({ extension = { til = 'til' } })
vim.lsp.config('til', {
  cmd = { 'til', 'lsp' },
  filetypes = { 'til' },
  root_markers = { 'til.json', '.git' },
})
vim.lsp.enable('til')
```

Neovim 0.8 – 0.10:

```lua
vim.filetype.add({ extension = { til = 'til' } })
vim.api.nvim_create_autocmd('FileType', {
  pattern = 'til',
  callback = function(args)
    local root = vim.fs.find({ 'til.json', '.git' }, { upward = true, path = vim.api.nvim_buf_get_name(args.buf) })[1]
    vim.lsp.start({
      name = 'til-lsp',
      cmd = { 'til', 'lsp' },
      root_dir = root and vim.fs.dirname(root) or vim.fn.getcwd(),
    })
  end,
})
```

Syntax highlighting: `editors/vim/til.vim` (copy it to `~/.config/nvim/syntax/`);
with semantic tokens on (the default) the server colours names by what they are.

## Helix

`~/.config/helix/languages.toml`:

```toml
[language-server.til-lsp]
command = "til"
args = ["lsp"]

[[language]]
name = "til"
scope = "source.til"
file-types = ["til"]
roots = ["til.json"]
comment-token = "#"
indent = { tab-width = 4, unit = "    " }
language-servers = ["til-lsp"]
```

## Sublime Text

Install the `LSP` package and the TIL syntax (`editors/sublime/TIL.sublime-syntax`,
scope `source.til`), then in *Preferences → Package Settings → LSP → Settings*:

```json
{
  "clients": {
    "til": {
      "enabled": true,
      "command": ["til", "lsp"],
      "selector": "source.til"
    }
  }
}
```

## Emacs

Eglot (built into Emacs 29):

```elisp
(define-derived-mode til-mode prog-mode "TIL"
  (setq-local comment-start "# "))
(add-to-list 'auto-mode-alist '("\\.til\\'" . til-mode))
(with-eval-after-load 'eglot
  (add-to-list 'eglot-server-programs '(til-mode "til" "lsp")))
(add-hook 'til-mode-hook #'eglot-ensure)
```

lsp-mode:

```elisp
(with-eval-after-load 'lsp-mode
  (add-to-list 'lsp-language-id-configuration '(til-mode . "til"))
  (lsp-register-client
   (make-lsp-client :new-connection (lsp-stdio-connection '("til" "lsp"))
                    :activation-fn (lsp-activate-on "til")
                    :server-id 'til-lsp)))
(add-hook 'til-mode-hook #'lsp-deferred)
```

## Zed and other editors

Any client that can start a language server over stdio works: the command is
`til lsp`, the language id `til`, the file extension `.til`. For a client that
connects over TCP start `til lsp --tcp PORT` first.

## Message language

Diagnostics, hovers, completion documentation and action titles follow the
editor's display language (the `locale` of the LSP `initialize` request, e.g.
VS Code's *Configure Display Language*): `ru` → Russian, `kk` → Kazakh, anything
else → English. `til lsp --lang kk` or `TIL_LANG=ru` force a language;
`initializationOptions: {"locale": "ru"}` sets it per client, and the setting
`til.locale` (workspace/didChangeConfiguration) changes it at run time.

## Details

* **Positions** are UTF-16 code units (`positionEncoding: "utf-16"`), as LSP
  and JavaScript count them: Cyrillic letters and emoji inside strings never
  shift a range.
* **Text sync** is full (`change: 1`); `\r\n` files are fine.
* **Imports** are resolved as the compiler does (next to the file, `TIL_PATH`,
  `til_modules/` packages, the standard library). Like `til run`, imported
  files are read **from disk**: a file that imports `geo.til` sees changes to
  `geo.til` once it is saved. Open files are always analysed from the editor's text.
* **Broken code**: when a file does not parse, the server analyses it with the
  broken lines blanked out (the first syntax error is reported), so outline,
  hover, completion and definition keep working; if even that fails, the last
  version that parsed is used for the outline and completion.
* **Formatting** only changes whitespace between tokens and the indentation of
  lines; it does nothing when the file does not lex.
* **Commands**: `workspace/executeCommand` accepts `til.translate`
  (`[{"uri": ..., "to": "en"|"kk"|"kk-latn"|"ru"}]` → `{"text": ...}`) and returns
  `null` for `til.run` / `til.debug` / `til.prove`. They are not listed in
  `executeCommandProvider`, so a client's own commands of these names are not
  replaced.
* **Semantic token legend** (`semanticTokensProvider.legend`):
  types `namespace type struct enum interface typeParameter parameter variable
  property enumMember function method keyword comment string number operator
  decorator`; modifiers `declaration readonly defaultLibrary static modification`.

## Limitations

* Members accessed on a value whose type is unknown (no annotation, nothing to
  infer it from) resolve only when exactly one struct in the program has a
  member of that name; otherwise references and rename leave them out.
* Rename works on names defined in the workspace folder; names from the
  standard library or `til_modules/` packages are not renamed.
* Effect names (`perform Ask(...)`, `Ask(x) =>`) are linked only when a
  top-level function of that name exists (its default handler).
* Types come from `src/tiltypes.py`; without it (or with `TIL_NO_TYPES=1`)
  hovers show the written annotations only and E-TYPE is not reported.
