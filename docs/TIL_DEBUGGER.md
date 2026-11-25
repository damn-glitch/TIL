# TIL time-travel debugger

`til debug` runs a TIL program step by step on the reference VM: breakpoints,
stepping into, over and out of functions, printing variables and the call
stack — and stepping **backwards**. `back` undoes one line, `rc`
(reverse-continue) goes back to the previous breakpoint hit. Going back is
exact: the program is not re-run with fresh input, it replays what it read the
first time, and nothing it printed is printed twice.

```bash
til debug prog.til                                    # interactive, prompt "(til) "
til debug prog.til --commands "b 5; c; p x; back; p x; q"
til debug game.til --keys "UP,UP,ENTER" -- level1     # scripted keys, args() for the program
til dap                                               # the same debugger for editors (DAP)
```

The same engine drives editors and TIL Studio: `til dap` is a [Debug Adapter
Protocol](#debug-adapter-dap) server, so VS Code, Neovim, Helix and the studio
get breakpoints, Step Back and Reverse Continue.

Contents: [example](#an-example-session) · [commands](#commands) ·
[options](#options) · [line events](#line-events) · [time travel](#time-travel) ·
[variables](#variables) · [expressions](#expressions) ·
[debug adapter (DAP)](#debug-adapter-dap) · [debug info in the bytecode](#the-debug-info-section) ·
[performance](#performance) · [messages](#messages) · [limitations](#limitations) ·
[implementation notes](#implementation-notes)

## An example session

`avg.til` averages groups of numbers; the third group is empty:

```til
var calls = 0

average(xs) -> int
    calls += 1
    var total = 0
    for x in xs
        total += x
    return total / len(xs)

main()
    let groups = [[3, 4, 5], [10, 20], []]
    for g in groups
        print(average(g))
```

`til debug avg.til --commands "b 8; c; p total; p xs; c; p calls; c; bt; p xs; back; back; p total; rc; p total; p xs[1]; locals; out; q"`
prints (program output is interleaved: `4` and `15`):

```text
__start__ at line 1
    1 | var calls = 0
(til) b 8
Breakpoint 1 at line 8
(til) c
Breakpoint 1, average at line 8
    8 |     return total / len(xs)
(til) p total
total = 12
(til) p xs
xs = [3, 4, 5]
(til) c
4
Breakpoint 1, average at line 8
    8 |     return total / len(xs)
(til) p calls
calls = 2
(til) c
15
Breakpoint 1, average at line 8
    8 |     return total / len(xs)
(til) bt
  at average(xs=[]) (line 8)
  at main() (line 13)
  at __start__() (line 1)
(til) p xs
xs = []
(til) back
average at line 6
    6 |     for x in xs
(til) back
average at line 5
    5 |     var total = 0
(til) p total
No variable 'total' here
(til) rc
Breakpoint 1, average at line 8
    8 |     return total / len(xs)
(til) p total
total = 30
(til) p xs[1]
xs[1] = 20
(til) locals
xs = [10, 20]
total = 30
(til) out
main at line 13
   13 |         print(average(g))
average returned 15
(til) q
```

`back` from the third call went to line 6 (the `for` over the empty array) and
line 5 (before `total` exists); `rc` jumped back to the breakpoint hit of the
*second* call, where `total` is 30 again; `out` finished that call — without
printing `4` or `15` a second time.

## Commands

| command | does |
|---|---|
| `b LINE` | set a breakpoint (on the next line with code if LINE has none); `b` alone lists them |
| `d LINE` | delete the breakpoint at LINE; `d` alone deletes all |
| `c` | continue to the next breakpoint (or the end) |
| `s` | step to the next line event anywhere (into calls) |
| `n` | step to the next line event in this function or a caller (over calls) |
| `out` | run until the current function returns: stops in the caller, in the middle of the line of the call, and shows the returned value |
| `back` | step back one line event |
| `rc` | reverse-continue: back to the previous breakpoint hit (none: back to the start) |
| `p NAME` | print a local, an upvalue (a captured variable of a closure or effect clause) or a global; also `NAME.field`, `NAME[3]`, `NAME["key"]`, `NAME[i]` and chains of them, and any [expression](#expressions) (`p total / len(xs)`) |
| `locals` | the local variables live at this point (innermost declaration of each name), then the upvalues |
| `bt` | the call stack, innermost first, with the parameters of every frame |
| `q` | quit |
| `help` | the command list |

Aliases: `break`, `delete`, `continue`, `step`, `next`, `finish`, `reverse-step`,
`reverse-continue`, `print`, `backtrace`/`where`, `quit`/`exit`. An unknown
command prints a one-line summary of the commands. In interactive mode an empty
line repeats the last command and Ctrl-C stops the running program at its next
line event.

`p` formats values like `print()`: strings inside containers are quoted
(`["a", "b"]`), a string on its own is not (`name = Aigerim`). A value longer
than 4000 characters is cut, and the cut says how much was left out.

Stops are shown as `FUNC at line N` (`Breakpoint K, FUNC at line N` when the line
has a breakpoint) followed by the source line. After the end of the program
(`Program exited with code 0`, `Program ended with an error (exit code 1)` — the
error report itself is printed once, on standard error, as `til run` prints
it), `p`, `locals` and `bt` show the final state (after an error: the state where
it happened), and `back`/`rc` travel back into the run.

## Options

| option | |
|---|---|
| `--commands "CMDS"` | run commands separated by `;` (or newlines); each is echoed as `(til) CMD` before its output, so a session is a deterministic transcript. When the script ends the debugger quits. |
| `--input FILE` | lines for the program's `input()` (default: standard input; in interactive mode the program and the prompt share it) |
| `--keys LIST` | scripted key presses for `key_pressed()`/`wait_key()` (as for `til run`); without it there are none: `key_pressed()` is 0 and `wait_key()` is 27 (ESC) |
| `--screen FILE.ppm` | when the session ends, write the screen as a PPM image |
| `--screen-ansi` | draw `screen_show()` frames in the terminal (part of the program output) |
| `--snapshot-every K` | copy the VM state every K line events (default 100; the interval grows by itself, see below) |
| `--no-fuse` | debug bytecode without superinstructions |
| `--lang en\|ru\|kk` | language of the debugger's messages |
| `-- ARGS` | arguments for `args()` (words after the program that are not options also go there) |

`til debug prog.tbc` debugs a compiled file. Built with `til build --target vm
--debug-info` it has variable names; without them `p $3` prints slot 3 of the
current frame and globals are `g0`, `g1`, ... If `prog.til` lies next to
`prog.tbc`, its lines are shown. A file built with `--strip` has no line table
and therefore no line events (the debugger says so and runs it to the end).

## Line events

A **line event** happens when the program is about to execute the first
instruction of a source line: when execution reaches the start of a line-table
entry whose line differs from the line this *frame* was on. Concretely:

* entering a function: its first line;
* every new line in the same frame, including the jump back to a loop header
  (`while`/`for` lines get an event per iteration);
* returning from a call continues the caller's line — no event until the
  caller reaches its next line;
* the implicit return at the end of a function without `return` is reported on
  the function's own line (`main at line 10` / `10 | main()`), like a closing
  brace in C;
* code of the prelude (the TIL source of `map`, `filter`, `reduce`, `each`,
  `any`, `all`) has no line events: `s` steps from `xs.map(|x| ...)` straight
  into the lambda.

Breakpoints are line numbers of the program; they are hit at the line events of
that line. `n` stops at the next line event of the current frame or of one of
its callers (by frame identity, so a recursive call of the same function is
stepped over); `out` stops when the current frame returns to its caller (or at
a line event of a caller, if the frame is left by an error or an effect clause
that aborts its handle). `n` and `out` also stop at breakpoints on the way.

## Time travel

The debugger numbers the line events of the run (0, 1, 2, ...) and remembers
the line of each. Every K events (default 100) it copies the complete VM state
(a *snapshot*). `back` goes to event *current − 1*, `rc` to the last event before
the current point whose line has a breakpoint; both restore the newest snapshot
at or before the target and run forward to it. This is exact because the run is
deterministic:

* **Recorded inputs.** Everything the program gets from outside is recorded the
  first time and played back on every later run over the same events:
  `input()` lines, `clock_ms()`, the clock that `timeout` blocks check, `sleep_ms()`
  (it does not sleep again), `read_file()`, `file_exists()`, `write_file()` (not
  written again) and key presses (`key_pressed()`, `wait_key()`), including the
  errors they raised. `random()` needs no recording: its generator state is part
  of the snapshot. `args()` is fixed for the session.
* **Output once.** The debugger counts the bytes of output a run has produced;
  only bytes past the furthest point ever printed are printed. Going back and
  continuing never repeats output, also not an `input()` prompt or an error
  report.
* **Clock.** `clock_ms()` and timeouts see a virtual clock that runs only while
  the program runs live (past everything recorded): time spent at the prompt or
  replaying history does not count, so a `timeout(100)` block does not expire
  because you looked at it for a minute.
* **Forward after back.** After `back`/`rc` the VM state is real: `c`, `s`, `n`
  and `out` simply continue. Up to the furthest point reached before, the
  program replays the recording; after it, it runs live again (new input is
  read, new output printed).

**What a snapshot holds** (`DebugVM.STATE_OBJECTS`/`STATE_NUMBERS`): the value
stack, the frames, the globals, the effect handler records (with the clause
pairs on the stack), the recovery points of `retry`/`timeout` (with their
deadlines), the screen (framebuffer, size, ANSI frame count), the step counter,
the random generator, and the debugger's position (event number, the line each
frame is on, the position in the input recording and in the output). The key
source and standard input are outside streams like a terminal: what the program
read from them is in the recording, and after the recorded part they continue
where they are.

**Snapshot interval.** Copies are made with a dedicated copier (`copy_state`:
keeps shared arrays/cells shared, no recursion limit for long linked
structures, 3–6× faster than `copy.deepcopy` on the states of the `benchmarks/`
programs). The debugger keeps at most 256 snapshots and at most about 20 million
copied values; beyond that it doubles K and drops every other snapshot, and it
also doubles K when copying takes longer than running between two copies. So
memory stays bounded and `back` replays at most K events (K is 6400–25600
after the multi-million-event benchmarks, about 10–50 ms of replay). The line
history itself grows by one list entry per line event.

## Variables

Names come from the [debug-info section](#the-debug-info-section), which `til
debug` asks the compiler for. A local is visible from just after its first
store (`let x = f()` — not on its own line yet, from the next line on) to the end
of its block; a later `let x` in the same or an inner block hides the outer one.
Parameters are visible in the whole function. Captured variables (cells) are
shown by value. In a lambda or effect clause the upvalues (captured variables of
the enclosing function) are visible by name. Globals are visible everywhere;
constants that the compiler inlines (`let X = 5` at the top level) have no slot,
but `p X` and expressions still know their value (the compiler's table).

## Expressions

`p`, the editors' hover and watch expressions, the debug console and breakpoint
conditions evaluate the same expressions, with the VM's own operations (so `7 /
2` is `3`, `"a" + 1` is `"a1"` and `"a" - 1` fails, as in the program):

| | examples |
|---|---|
| variables | locals, upvalues, globals, inlined constants, `$3` (slot 3 without debug info) |
| fields and indexes | `p.x`, `xs[i]`, `m["key"]`, `m.key` (a map key), `opt.value` (Some / Ok), `res.error` (Err), `shape[0]` (an enum value's fields), `Color.Red`, `Shape.Empty` |
| literals | `42`, `1.5`, `"text"`, `'c'`, `true`, `false`, `None`, `[1, 2]`, `{"a": 1}`, `f"{x}-{y}"`, `Some(x)`, `Ok(x)`, `Err(x)` |
| operators | `+ - * / % **`, comparisons (also chained: `0 <= i < n`), `and or not`, `& \| ^ ~ << >>`, `x if c else y`, `x as float` |
| calls | builtins without side effects: `len str int float bool abs sqrt pow sin cos tan log exp floor ceil round min max sum chr ord range type_of format str_bytes bytes_to_str args screen_width screen_height get_pixel`; builtin methods that do not change their receiver: `len contains starts_with ends_with trim to_upper to_lower replace slice find index_of count eq split repeat to_int to_float is_empty get join first last sum min max has keys values get_or unwrap expect unwrap_or is_some is_none is_ok is_err error to_string copy` |

Nothing an expression does can change the program or the world: calls of the
program's own functions and methods, methods that change their receiver (`push`,
`sort`, ...), `print`, `input`, `random`, files and keys, assignments, lambdas
and list comprehensions are refused (`Cannot evaluate '...': only variables,
fields, ...`); `xs * n` and `range()` may not build more than a million elements.
Keywords of the program's languages work (`и`/`және` for `and` in a `#[lang:
ru]` / Kazakh program).

## Debug adapter (DAP)

`til dap` speaks the [Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/)
over standard input and output (`Content-Length` framing; `--log FILE` writes
every message to FILE). It is this debugger: the program runs on the reference
VM with line events, snapshots and the recorded inputs, so the editor's **Step
Back** and **Reverse Continue** buttons work (`supportsStepBack`), and output is
printed once however often you go back.

**VS Code** (`.vscode/launch.json`; the debug type `til` comes with the TIL
extension in `editors/vscode/`, which starts `til dap`):

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
      "name": "Debug the game with scripted keys and input",
      "program": "${workspaceFolder}/game.til",
      "keys": "RIGHT,RIGHT,ENTER,ESC",
      "stdin": "Aigerim\n42\n",
      "stopAtEnd": true,
      "lang": "kk"
    }
  ]
}
```

**Neovim** ([nvim-dap](https://github.com/mfussenegger/nvim-dap)):

```lua
local dap = require('dap')
dap.adapters.til = { type = 'executable', command = 'til', args = { 'dap' } }
dap.configurations.til = {
  {
    type = 'til', request = 'launch', name = 'Debug TIL file',
    program = '${file}', cwd = '${workspaceFolder}', stopOnEntry = false,
  },
}
-- reverse debugging: dap.step_back(), dap.reverse_continue()
```

(`vim.filetype.add({ extension = { til = 'til' } })` if Neovim does not know
`.til` files yet.) **Helix** (`languages.toml`):

```toml
[[language]]
name = "til"
scope = "source.til"
file-types = ["til"]

[language.debugger]
name = "til"
transport = "stdio"
command = "til"
args = ["dap"]

[[language.debugger.templates]]
name = "launch"
request = "launch"
completion = [ { name = "program", completion = "filename" } ]
args = { program = "{0}" }
```

**Launch arguments**

| argument | |
|---|---|
| `program` | the `.til` (compiled with debug info) or `.tbc` file; relative paths start at the session's root (TIL Studio's project) or the adapter's working directory |
| `args` | arguments for `args()` (a list of strings) |
| `cwd` | where the program's relative file paths start (`read_file`, `write_file`, `file_exists`); default: the root / the adapter's working directory |
| `stdin` | the program's whole input (text); without it input is interactive: see `til/stdin` |
| `keys` | scripted key presses, as `--keys` (`"UP,UP,ENTER"`) or a list of key codes |
| `stopOnEntry` | stop at the first line event (reason `entry`) |
| `noDebug` | just run (no stops; output, input and screen as when debugging) |
| `stopAtEnd` | stop at the end of the program (reason `exit`) instead of ending the session, so you can still go back in time |
| `lang` | `en`, `ru` or `kk`: the language of the adapter's messages (default: the client's `locale`) |
| `source` | the program's text instead of the file (an unsaved editor buffer; `program` names it) |
| `noFuse`, `snapshotEvery` | as `--no-fuse` and `--snapshot-every` |

**Requests.** `initialize`, `launch`, `setBreakpoints` (conditions and hit
conditions), `setExceptionBreakpoints`, `breakpointLocations`,
`configurationDone`, `threads` (one thread, `main`), `stackTrace` (paged),
`scopes`, `variables` (paged), `evaluate`, `exceptionInfo`, `continue`, `next`,
`stepIn`, `stepOut`, `stepBack`, `reverseContinue`, `pause`, `terminate`,
`disconnect`, `loadedSources`; `attach`, `setVariable`, `restart` and function
breakpoints are not supported. Events: `initialized` (after the launch
response), `stopped` (`entry`, `breakpoint` with `hitBreakpointIds`, `step`,
`pause`, `goto`, `exception`, `exit`; `allThreadsStopped: true`), `continued`,
`output` (`stdout`, `stderr` for the error report, `console` for the adapter's
notes), `breakpoint`, `exited` (`exitCode`) followed by `terminated`.

* **Breakpoints** on a line without code move to the next line with code
  (`verified: true` and the actual `line` in the response; no code at or after
  it: `verified: false` with a message). A breakpoint belongs to its file: code
  imported from another module stops only at that module's breakpoints, and its
  stack frames name that file. A **condition** is an [expression](#expressions)
  (a condition that cannot be evaluated stops, and the console says why); a **hit
  condition** counts the line events of that line from the start of the program
  (also after going back in time): `N` (the Nth time), `== N`, `>= N`, `> N`,
  `<= N`, `< N`, `% N` (every Nth time).
* **Stepping.** `next` / `stepIn` / `stepOut` are `n` / `s` / `out`. `stepBack`
  is the reverse of `next`: back to the previous line event of this function or
  of a caller, stepping over the calls made in between (at a function's first
  line: back to the line that called it); from the end of the program it goes to
  the last line that ran. `reverseContinue` goes back to the previous breakpoint
  hit, with conditions and hit counts (none: to the start of the program). After
  `stepOut` the Locals scope shows the returned value (`f() returned`).
* **Variables.** Scopes `Locals` (the frame's live locals and upvalues) and
  `Globals`; arrays (`indexedVariables`, paged), maps, structs, closures (their
  captured variables), Some/Ok/Err and enum values with fields expand;
  `evaluateName` gives the path of each part (`m["nested"]["a"][1]`). Strings are
  shown quoted with escapes. `variablesReference` and frame ids are valid while
  the program is stopped.
* **Runtime errors.** The exception filter `runtime-error` (on by default) stops
  where an uncaught error ends the program (reason `exception`, `exceptionInfo`
  has the message and the traceback); the state is the one where it happened, so
  you can inspect it and step back. Continuing ends the session (`exited` with
  code 1). Without the filter the program just ends. The error report is an
  `output` event (`stderr`), once.
* **Input.** With launch `stdin` the program reads that text. Otherwise, when it
  calls `input()`, the adapter sends the event `til/inputRequested` and waits for
  the request `til/stdin {text, eof?}` (several lines at once are fine; `eof: true`
  ends the input); in VS Code type the line in the debug console (an `evaluate`
  with context `repl` while the program waits for input is its next line).
  Going back in time replays what was read: the program is not asked again.
* **Custom requests and events** (for TIL Studio's timeline): `til/timeline` →
  `{index, total, complete}` (line events before the current point, line events
  recorded, whether the history reaches the end of the program); `til/goto
  {index}` jumps to any recorded line event (`continued`, then `stopped` with
  reason `goto`; `index = total` once `complete`: to the end, reason `exit` or
  `exception`); event `til/screen {w,
  h, rgb_b64}` (the framebuffer as base64 RGB24) on `screen_show()` - at most 30
  frames a second while the program runs - and at every stop and at the end when
  the framebuffer changed, so after going back the screen shows that moment.

**In-process API** (TIL Studio, `src/tilstudio.py`):

```python
from tildap import DapSession
session = DapSession(emit, root)        # emit(event: dict): called from any thread, one at a time
responses = session.handle(message)    # one DAP message -> the response(s) to send now
with session.hold_events():            # optional: events caused by the request come after
    send(session.handle(message))      #   what you send inside the block
session.close()
```

The program runs on a worker thread; `handle()` does not wait for it (only
`disconnect` and `close()` wait up to 5 seconds for it to end), and may be
called from several threads (requests are processed one at a time). `seq`
numbers of responses and events come from one counter.

## The debug-info section

An optional section of the `.tbc` format, right after the line table (pcs are
absolute offsets in the code section, like the line table):

```
n u16, then n records:  func u16, slot u8, name u16, start_pc u32, end_pc u32   (13 bytes each)
```

`name` is the index of a string constant. Three kinds of record:

| record | `func` | `slot` | `start_pc` / `end_pc` |
|---|---|---|---|
| local variable | the function's index | its local slot | live while `start_pc <= pc < end_pc` |
| global | `0xFFFF` (`DEBUG_GLOBAL`) | 0 | both: the global's index |
| upvalue | `0xFFFE` (`DEBUG_UPVAL`) | the upvalue's index | both: the index of the function (lambda / clause) whose closure holds it |

**Globals.** The global index is the u16 operand of `LOAD_GLOBAL`; the u8 `slot`
field cannot hold indices above 255, so a global record keeps the index in the
two pc fields (both equal) and marks itself with the pseudo function index
`0xFFFF` — a value no function can have (`PERFORM` already uses `0xFFFF` as
"no function"). A program may have up to 65535 globals with names. Upvalues use
`0xFFFE` the same way (their index fits the u8 slot field, like `n_upvals` of a
function record). Locals never need more than the u8 slot (the compiler allows
250 locals per function). With debug info a module may have at most 65533
functions and 65535 records (otherwise the build fails with a clear error).

Records are written in a fixed order (locals by function and start pc, then
upvalues, then globals), so reading a file and writing it again gives the same
bytes. The names are appended to the constant table when the module is written
(code already compiled keeps its constant indices).

**Who writes it.** Only on request: `til debug` (always) and `til build --target
vm|c-vm --debug-info` (also `til dis --debug-info`, which lists the names and
ranges). `--strip` omits it together with the line table: `til build --strip
--debug-info` is byte for byte `til build --strip`. Default output is unchanged
— the self-hosted compiler's byte-identity tests compare `tilc` with the Python
compiler, and `tilc` does not write the section. The section is the compiler's
answer to the question "what is this slot called"; superinstructions
(`fuse_superinstructions`) move code, so the compiler remaps the ranges through
the pass's old→new pc map (a range that ends inside a fused instruction lasts to
its end: `y` stays visible at `RETURN_LOCAL y`, which replaced `LOAD_LOCAL y;
RETURN`).

**Who reads it.** `TBCModule.from_bytes` reads it when there are bytes after the
line table (`module.debug_info`, `TBCFunction.locals`/`upval_names`,
`module.globals` get the names; a truncated or inconsistent section is a load
error). The C VM ignores it: `tilvm_load` stops after the line table, so files
with the section run unchanged on every C VM build (tested with the host and
int32 builds). Files without it — older files, `tilc` output — load as before.
The names add constants: a C VM build with a small `TILVM_MAX_CONSTS` may need
the file without debug info.

## Performance

* **Normal runs pay nothing.** `til run`/`til exec` use `TILVM.execute()`
  unchanged; it only gained a comment (`# @line-event`). The debugger compiles
  its own copy of `execute()` from that source with the check
  `if pc in dbg_lines: dbg_event(pc)` inserted at the comment
  (`tildebug.debug_execute_code`). `python3 benchmarks/run.py --baseline <commit
  before the debugger> --vms py-base,py --repeat 5` (CPU seconds, medians):

  | benchmark | before (s) | after (s) |
  |---|---:|---:|
  | arrays | 2.934 | 2.945 |
  | closures | 3.017 | 3.004 |
  | fib | 1.924 | 1.938 |
  | loops | 4.456 | 4.450 |
  | maps | 0.100 | 0.100 |
  | strings | 0.469 | 0.467 |
  | structs | 2.306 | 2.244 |

  (0.99×–1.03×: noise of a shared machine; the executed code is the same.)
  Compiling is unchanged too: without `debug_info` the compiler only tests a flag
  when it declares a local.
* **Under the debugger** (no breakpoints, `c` to the end, snapshots on), the
  benchmarks run 1.2–1.8× slower than on the normal VM: fib 1.43×, loops 1.60×,
  arrays 1.78×, strings 1.22×, maps 1.60×, structs 1.52×, closures 1.49×
  (1.3 million line events for fib(27); snapshot copying is under 10% of that).

## Messages

The debugger's messages go through TIL's message catalog (`src/tilmsg.py`,
command-line entries): `--lang ru|kk`, `TIL_LANG` or the program's
`#[lang: ...]` pragma choose Russian or Kazakh, as for compiler diagnostics.
Values, source lines and command names are not translated; `bt` uses the same
`at f (line N)` lines as error reports, which the catalog already translates.

## Limitations

* Runs on the reference (Python) VM only; the C VM is not instrumented.
* Line numbers have no file in the bytecode: `til debug`'s `b 5` stops at line 5
  of every file of the program, and its stops show the main file's source line.
  The debug adapter knows the file of each function (from the compiler: named
  functions and methods by the file that defines them, lambdas and effect
  clauses by the function they are written in), so its breakpoints and stack
  frames are per file; a `.tbc` program is one file. A hit condition counts the
  line events of its line number in all files.
* The debug adapter runs one program per session (one thread, `main`); keys come
  from `keys` only (no live keyboard), `setVariable` is not offered (changing a
  value would break the recorded history), and `pause` while the program waits
  in `input()` takes effect once the line has arrived.
* A function name with a space (`<clause Ask:7>`) keeps the location message
  English in `ru`/`kk`.
* Keys come from `--keys` only (no terminal keys: the terminal is the debugger's
  prompt). Interactive programs that read `input()` share standard input with
  the prompt (use `--input FILE` to separate them).
* Re-running history is deterministic as long as the program depends only on
  the recorded inputs; anything else outside the VM (another process changing a
  file between a `write_file` and a `read_file` of the original run) is replayed
  as it was recorded. If a replay ever differed from the history, the debugger
  says so (`the replay differs from the recorded run at line event N`).
* `--commands` scripts split on `;` and newlines, so a `p m["a;b"]` cannot be
  written in a script.

## Implementation notes

* `src/tildebug.py`: `debug_execute_code()` (the instrumented copy of
  `execute()`), `DebugVM` (a `TILVM`: line-event hook, `resume(stop)`,
  snapshots, `restore`/`travel`, `nd()` record/replay, overrides of
  `call_native`, `gfx_native`, `path_arg` and `check_budget`; `request_pause()`
  and `kill()` from other threads), `copy_state`, `DebugSession` (the engine,
  below), `Session` (the commands of `til debug`, on a DebugSession),
  `compile_program` / `load_program` (the program with the compiler's
  `ProgramInfo`: the file of every function, inlined constants, enum values),
  `main(argv)` (routed as `til debug` through `EXTRA_COMMANDS`).
* `DebugSession(module, source, path, args=, stdin=, keys=, screen=, out=, err=,
  on_screen=, every=, cwd=, lang=, info=)` or `DebugSession.load(path, source=None)`:
  breakpoints (`resolve_line`, `add_breakpoint(line, file, condition,
  hit_condition)`, `set_file_breakpoints`, `remove_breakpoint`), run methods that
  return a `Stop` (`start`, `cont`, `step_in`, `step_over`, `step_out`,
  `step_back` = `back`, `step_back_over` = DAP's `stepBack`, `reverse_continue`,
  `goto(index)`, `run_to_end`; `interrupt()`, `kill()` from any thread), queries
  (`stack()`, `locals(depth)`, `params(depth)`, `globals()`, `lookup`,
  `evaluate(expr, depth)`, `timeline()`, `output`, `finished`), values
  (`preview`, `children`, `describe`; module functions `preview`, `value_type`,
  `value_kind`, `value_children`, `child_counts`).
* Reverse "step over" needs the depth of every line event: `DebugVM.ev_meta`
  (2 × depth + 1 for a frame's first line event) grows with the history, like
  `ev_lines`. Reverse-continue checks breakpoints without conditions on the
  history alone; with conditions it replays the history backwards one snapshot
  interval at a time and evaluates them there.
* `src/tildap.py`: `DapSession` (requests -> engine calls; the program on a
  worker thread; variables references and frame ids per stop; output sent in
  chunks of at most 30 ms), `StdioServer` and `main(argv)` (`til dap`).
* The `# @line-event` comment in `TILVM.execute()` must stay at the top of the
  dispatch loop, after `op_pc = pc` and before the opcode is read; everything
  the loop keeps in local variables must be derivable from `self.frames[-1]` at
  that point (it is: `pc`, `base`, `closure`, `frame`). The debugger pauses by
  raising an exception there and resumes by calling `execute()` again.
  `tests/test_debugger.py::TestInternals` checks the marker.
* A new native whose result comes from outside the program (time, files,
  devices, network) must be added to `tildebug._RECORDED` (and handled in
  `DebugVM.call_native`); new VM state that a run changes must be added to
  `DebugVM.STATE_OBJECTS` or `STATE_NUMBERS`.
* Compiler side: `BytecodeCompiler(debug_info=True)` (`compile_to_bytecode(...,
  debug_info=True)`) records `[slot, name, start, end]` per local in
  `TBCFunction.locals` (from `declare_local`/`store_new_local`/`end_scope`) and
  the upvalue names; `_remap_live_ranges` moves them after the superinstruction
  pass; `TBCModule.debug_records()`/`read_debug_records()` write and read the
  section.
