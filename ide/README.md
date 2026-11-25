# TIL Studio — the IDE for TIL

**Author:** Alisher Beisembekov

TIL Studio opens in your browser and needs nothing but Python 3.9+ (no
pip packages, no internet). It uses the real TIL compiler for every
diagnostic and completion, runs programs natively, on the TIL VM and on the
C VM, and debugs them with **step back** and a **timeline** you can drag
through the whole run.

```bash
til ide                 # the current folder
til ide my-project      # a project folder (it may contain til.json)
til ide --lang ru       # UI and messages in Russian (en, ru, kk)
```

or, from this folder without installing: `./til-ide.sh [FOLDER]` (Linux/macOS)
or `til-ide.bat [FOLDER]` (Windows).

The command prints a local URL with a one-time token and opens it (as an app
window when Chrome or Edge is installed). Only programs on your computer can
reach it, and only with that token.

![TIL Studio](../docs/images/studio-editor-dark.png)

What you get:

* **Editor** — highlighting for English, Kazakh and Russian keywords,
  completion with signatures, hover with inferred types, inline type hints,
  errors as you type (localized), quick fixes, rename across files, go to
  definition, outline, formatting, folding, multi-cursor, split view.
* **Run** — native / TIL VM / C VM, program input, live graphics screen in a
  calculator frame with keys, stop, portable snapshots.
* **Debug** — breakpoints (with conditions), step over/into/out, step back,
  reverse continue, a timeline scrubber, variables, watch, call stack.
* **Proofs** — contracts proved for all inputs with Z3; counterexamples you
  can debug with one click.
* **Devices** — will the program fit a TI-84 Plus CE, NumWorks, Z80,
  Arduino…? Memory bars, warnings linked to lines, run on the device's VM.
* **Packages**, **Bytecode** view synced with the source, **Translate view**
  (the same program in Kazakh / Russian / English).
* Dark and light themes; the whole UI in English, Russian and Kazakh.

The same app also runs on the website without installing anything
(`website/studio/`: the compiler and VM as WebAssembly in the browser), and
the editor features are available in VS Code, Neovim, Helix, Sublime and
Emacs through `til lsp` and `til dap`.

Full guide: [docs/TIL_STUDIO.md](../docs/TIL_STUDIO.md). The frontend sources
are in `studio/src`; `cd studio && npm ci && node build.mjs` rebuilds
`studio.html` (the built file is committed, so users never need Node).
