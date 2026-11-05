#!/usr/bin/env python3
"""
til debug - the time-travel debugger of TIL 3.1 (docs/TIL_DEBUGGER.md).

    til debug prog.til                                  # interactive, prompt "(til) "
    til debug prog.til --commands "b 5; c; p x; back; p x; q"
    til debug prog.til --keys "UP,ENTER" -- arg1 arg2   # keys and args() for the program

The program runs on the reference VM (class DebugVM below, a TILVM). It stops
at *line events* ("about to execute source line L"); between stops it runs at
the VM's normal pace. Every K line events (default 100) the debugger copies the
whole VM state; `back` and `rc` restore the nearest copy and run forward again
to the line event they want. Everything that could differ on the second run is
recorded the first time and played back: input() lines, clock_ms(), the clock
of timeout blocks, sleep_ms(), read_file()/file_exists()/write_file() and key
presses. Program output that was printed once is never printed again.

How the line events cost normal runs nothing: TILVM.execute() (src/til.py) has
a marker comment `# @line-event` at the top of its dispatch loop. This module
compiles a copy of execute() with a line-event check put at the marker
(`if pc in dbg_lines: dbg_event(pc)`), and only DebugVM runs that copy. The
fast loop of `til run` / `til exec` is untouched (and keeps every opcode in
sync with the debugger automatically).

Variable names come from the debug-info section of the bytecode (the compiler
writes it when asked: BytecodeCompiler(debug_info=True), `til build
--debug-info`): locals with live pc ranges, upvalues of closures and globals.
"""

import copy
import inspect
import os
import re
import signal
import sys
import textwrap
import time
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import til

USAGE = """\
usage: til debug PROGRAM.til|PROGRAM.tbc [options] [-- program arguments]

  --commands "CMDS"     run a script of commands separated by ';' (each one is
                        echoed as "(til) CMD"); without it: interactive
  --input FILE          lines for the program's input() (default: standard input)
  --keys LIST           scripted key presses for key_pressed()/wait_key()
  --screen FILE.ppm     at the end of the session: write the screen as a PPM image
  --snapshot-every K    copy the VM state every K line events (default 100)
  --no-fuse             debug bytecode without superinstructions
  -- ARGS               arguments for args()

commands: b LINE, d LINE, c, s, n, out, back, rc, p NAME, locals, bt, q (help: details)"""

SNAPSHOT_EVERY = 100            # default K: line events between two state copies
MAX_SNAPSHOTS = 256             # more copies than this: keep every other one (K doubles)
SNAPSHOT_SLOTS = 20_000_000     # values held by all copies together before thinning them
MAX_SHOWN = 4000                # characters of one printed value


def _(text: str) -> str:
    """A debugger message in the language of TIL's messages (src/tilmsg.py)."""
    return til.localize(text)


# ═══════════════════════════════════════════════════════════════════════════
# the debug copy of TILVM.execute()
# ═══════════════════════════════════════════════════════════════════════════

_MARKER = re.compile(r'^([ \t]*)# @line-event\b')
_EXECUTE_CODE: List[Any] = []


def debug_execute_code():
    """TILVM.execute() with the line-event check at its `# @line-event` marker,
    compiled once (a code object; DebugVM makes a function of it per VM)."""
    if _EXECUTE_CODE:
        return _EXECUTE_CODE[0]
    try:
        src = textwrap.dedent(inspect.getsource(til.TILVM.execute))
    except (OSError, TypeError) as e:
        raise RuntimeError(f"the source of TILVM.execute() is not available ({e})")
    lines = src.split('\n')
    marks = [i for i, ln in enumerate(lines) if _MARKER.match(ln)]
    if len(marks) != 1 or not lines[0].lstrip().startswith('def execute(self'):
        raise RuntimeError("TILVM.execute() has no `# @line-event` marker (src/til.py and "
                           "src/tildebug.py do not belong together)")
    indent = _MARKER.match(lines[marks[0]]).group(1)
    lines[marks[0]:marks[0] + 1] = [indent + "if pc in dbg_lines:", indent + "    dbg_event(pc)"]
    body = re.match(r'[ \t]*', lines[1]).group(0) or '    '
    lines[1:1] = [body + "dbg_lines = self.dbg_lines", body + "dbg_event = self.dbg_line_event"]
    code = compile('\n'.join(lines), '<til debug: TILVM.execute with line events>', 'exec')
    _EXECUTE_CODE.append(code)
    return code


class _Pause(Exception):
    """Raised by the line-event hook to stop the program at an instruction boundary."""


class VirtualClock:
    """Milliseconds that advance only while the program runs live (not while the
    debugger waits for a command or re-runs history)."""

    def __init__(self):
        self.base = time.monotonic() * 1000.0
        self.running_since: Optional[float] = None

    def run(self):
        if self.running_since is None:
            self.running_since = time.monotonic() * 1000.0

    def halt(self):
        if self.running_since is not None:
            self.base += time.monotonic() * 1000.0 - self.running_since
            self.running_since = None

    def now_ms(self) -> float:
        if self.running_since is None:
            return self.base
        return self.base + time.monotonic() * 1000.0 - self.running_since


class _TimeShim:
    """`_time` for the debug copy of execute(): timeout blocks read the recorded clock."""

    def __init__(self, vm: 'DebugVM'):
        self.vm = vm

    def monotonic(self) -> float:
        return self.vm.vclock() / 1000.0


# ═══════════════════════════════════════════════════════════════════════════
# copying the VM state
# ═══════════════════════════════════════════════════════════════════════════

_IMMUTABLE = frozenset((int, float, bytes, bool, type(None), str))


def copy_state(root, memo: Optional[Dict[int, Any]] = None) -> Tuple[Any, int]:
    """A deep copy of VM values (lists, maps, structs, closures, cells, the
    screen) that keeps sharing (two references to one array stay one array) and
    needs no recursion (long linked structures are fine). Returns (copy, number
    of values copied). 3-6x faster than copy.deepcopy on the states of the
    benchmarks/ programs."""
    imm = _IMMUTABLE
    if memo is None:
        memo = {}
    memo.setdefault(id(til._UNDEF), til._UNDEF)
    todo: List[Tuple[Any, Any]] = []
    VMStruct, VMClosure, VMCell = til.VMStruct, til.VMClosure, til.VMCell
    size = 0

    def get(v):
        c = memo.get(id(v))
        if c is not None:
            return c
        t = type(v)
        if t is list:
            c = []
        elif t is dict:
            c = {}
        elif t is VMStruct:
            c = VMStruct(v.t, None)
        elif t is VMClosure:
            c = VMClosure(v.fn, ())
        elif t is VMCell:
            c = VMCell(None)
        elif t is bytearray:
            c = bytearray(v)
            memo[id(v)] = c
            return c
        elif t is til.TILScreen:
            c = copy.copy(v)
            c.fb = bytearray(v.fb)
            memo[id(v)] = c
            return c
        else:
            c = copy.deepcopy(v)
            memo[id(v)] = c
            return c
        memo[id(v)] = c
        todo.append((v, c))
        return c

    res = root if type(root) in imm else get(root)
    while todo:
        v, c = todo.pop()
        t = type(v)
        if t is list:
            size += len(v)
            c.extend([x if type(x) in imm else get(x) for x in v])
        elif t is dict:
            size += len(v)
            for k, x in v.items():
                c[k] = x if type(x) in imm else get(x)
        elif t is VMStruct:
            c.f = get(v.f)
        elif t is VMClosure:
            c.up = tuple(get(u) for u in v.up)
        else:
            x = v.v
            c.v = x if type(x) in imm else get(x)
    return res, size


# ═══════════════════════════════════════════════════════════════════════════
# the debugging VM
# ═══════════════════════════════════════════════════════════════════════════

# natives whose results come from outside the program: recorded, then played back
_RECORDED = ('input', 'clock_ms', 'sleep_ms', 'read_file', 'file_exists', 'write_file',
             'key_pressed', 'wait_key')


class DebugVM(til.TILVM):
    """A TILVM with line events, state snapshots and record/replay of inputs.

    State that a snapshot copies (and restores): STATE_OBJECTS + STATE_NUMBERS
    of the VM - the value stack, frames, globals, effect handler records,
    recovery points (retry/timeout, with their deadlines), the screen's
    framebuffer, the step counter, the random generator, the energy mode and
    idle count - and the debugger's position: the line-event index, the frame
    line memory, the position in the input recording and in the program's
    output. The key source and stdin are outside streams: what the program read
    from them is in the recording, and after the recorded part they continue
    where they are. Energy saving counts its idles but does not sleep here (a
    debugging session saves no battery, and history re-runs stay fast)."""

    STATE_OBJECTS = ('stack', 'frames', 'globals', 'handlers', 'recover', 'screen', 'shadow')
    STATE_NUMBERS = ('deadlines', 'tick', 'steps', 'rng', 'energy', 'idle_count', 'ev', 'nd_pos',
                     'out_pos')

    def __init__(self, module: 'til.TBCModule', out: Callable[[bytes], None],
                 read_line: Callable[[], bytes], argv=None, screen=None, keys=None,
                 every: int = SNAPSHOT_EVERY, on_report: Optional[Callable[[], None]] = None):
        super().__init__(module, write=self._program_write, read_line=read_line, argv=argv,
                         screen=screen, keys=keys)
        self.out = out
        self.on_report = on_report
        # line-entry pc -> line (None: the return point `out` waits for, see return_stop)
        self.dbg_lines: Dict[int, Optional[int]] = dict(module.lines)
        # the prelude (array map/filter/... written in TIL) has line numbers of its own
        # source: no line events there (stepping goes through to the lambda)
        self.hidden_funcs = set(module.prelude.values())
        for fidx in self.hidden_funcs:
            f = module.funcs[fidx]
            for off, _line in f.lines:
                self.dbg_lines.pop(f.offset + off, None)
        self.ret_stop: Optional[Tuple[list, int]] = None    # (caller frame, its return pc)
        self.pseudo_pc: Optional[int] = None
        g = dict(vars(til))
        g['_time'] = _TimeShim(self)
        exec(debug_execute_code(), g)
        self._execute = types.MethodType(g['execute'], self)
        self.clock = VirtualClock()
        self.recorded = {til.NATIVE_ID[n] for n in _RECORDED}
        self.nd_log: List[Tuple[str, bool, Any]] = []    # (kind, ok, value or panic message)
        self.nd_pos = 0
        self.ev = -1                    # index of the line event the program is at
        self.ev_lines: List[int] = []   # line of every line event seen (the history)
        self.shadow: List[list] = []    # [frame, last line] per frame: what is a new line
        self.out_pos = 0                # bytes of output this run has produced
        self.out_printed = 0            # bytes of output printed (only these are new)
        self.snaps: Dict[int, tuple] = {}
        self.every = max(1, int(every))
        self.snap_slots = 0
        self.snap_time = 0.0            # seconds spent copying states
        self.last_snap_at = time.perf_counter()
        self.stop: Callable[[int, int, int, list], bool] = lambda ev, line, depth, top: True
        self.dbg_skip = False           # the next hook call is the event we stopped at
        self.paused = False             # stopped (between two instructions: state is consistent)
        self.at_event = False           # ... exactly at line event `ev` (else later, mid-line)
        self.interrupted = False        # Ctrl-C: stop at the next line event
        self.finished: Optional[Tuple[int, str]] = None   # (exit code, 'exit' | 'error')
        self.reported = False           # the error report of the end was printed
        self.diverged: Optional[int] = None
        entry = module.funcs[module.entry]
        self.frames = [[module.entry, entry.offset, 0, None, 0]]
        self.stack = [None] * entry.n_locals

    # ── output: what was printed once is not printed again ─────────────────
    def _program_write(self, data: bytes):
        start = self.out_pos
        end = self.out_pos = start + len(data)
        if end > self.out_printed:
            skip = self.out_printed - start
            self.out(data[skip:] if skip > 0 else data)
            self.out_printed = end

    def report(self, msg: bytes, pc: int):
        if self.reported:
            return                      # the same end again (history is deterministic)
        self.reported = True
        if self.on_report:
            self.on_report()
        super().report(msg, pc)

    # ── recorded inputs ─────────────────────────────────────────────────────
    def nd(self, kind: str, live: Callable[[], Any]):
        """The result of something outside the program: played back from the
        recording while there is one, else obtained live and recorded."""
        pos = self.nd_pos
        log = self.nd_log
        if pos < len(log):
            k, ok, value = log[pos]
            if k != kind and self.diverged is None:
                self.diverged = self.ev
            self.nd_pos = pos + 1
            if ok:
                return value
            raise til.TILPanic(value)
        try:
            value = live()
        except til.TILPanic as p:
            log.append((kind, False, p.msg))
            self.nd_pos = pos + 1
            raise
        log.append((kind, True, value))
        self.nd_pos = pos + 1
        return value

    def vclock(self) -> float:
        return self.nd('clock', self.clock.now_ms)

    def call_native(self, nid: int, args: list):
        if nid not in self.recorded:
            return til.TILVM.call_native(self, nid, args)
        name = til.NATIVES[nid]
        if len(args) < til.NATIVE_MIN_ARGS[nid]:
            self.panic(f"{name}() is missing arguments")
        if name == 'input':
            if args:
                self.write(self.to_str(args[0]))
            return self.nd('input', self.read_line)
        if name == 'clock_ms':
            return int(self.vclock())
        if name == 'sleep_ms':
            ms = max(0, self.num(args[0], 'sleep_ms'))
            return self.nd('sleep', lambda: time.sleep(ms / 1000.0))
        if name in ('read_file', 'file_exists', 'write_file'):
            self.path_arg(name, args[0])            # its errors come from the program
        return self.nd(name, lambda: til.TILVM.call_native(self, nid, args))

    def idle(self, ms: int):
        pass                            # energy saving: counted (energy_point), no sleep

    def check_budget(self, pc: int):
        self.steps += 1
        if self._slow:                  # a step limit or energy saving (TILVM.check_budget)
            steps = self.steps
            if self._max_steps and steps > self._max_steps:
                raise til.TILFatal(f"step limit exceeded ({self._max_steps})")
            if not steps & 511 and (self._energy or self.low_funcs):
                self.energy_point()
        if self.deadlines:
            self.tick += 1
            if self.tick >= self.CHECK_INTERVAL:
                self.tick = 0
                now = self.vclock()
                for i, rf in enumerate(self.recover):
                    if rf[3] and now > rf[3]:
                        raise til.TILPanic(b"timeout", target=i)

    # ── line events ─────────────────────────────────────────────────────────
    def dbg_line_event(self, pc: int):
        """Called by the debug execute() before every instruction that starts a
        line-table entry. It is a line event unless the frame is still on that line."""
        frames = self.frames
        top = frames[-1]
        top[1] = pc
        if self.dbg_skip:
            self.dbg_skip = False           # resuming at the event we stopped at
            return
        line = self.dbg_lines[pc]
        if line is None:                    # only the return point of `out`
            if self.ret_stop is not None and top is self.ret_stop[0]:
                self.ret_stop = None
                raise _Pause(False)
            return
        sh = self.shadow
        n = len(frames)
        k = len(sh)
        if k > n:
            del sh[n:]
            k = n
        while k and sh[k - 1][0] is not frames[k - 1]:
            sh.pop()
            k -= 1
        if k == n:
            entry = sh[-1]
            if entry[1] == line:
                if self.ret_stop is not None and top is self.ret_stop[0] and pc == self.ret_stop[1]:
                    self.ret_stop = None
                    raise _Pause(False)     # `out` returned to the middle of this line
                return
            entry[1] = line
        else:
            while k < n - 1:
                sh.append([frames[k], 0])
                k += 1
            sh.append([top, line])
        ev = self.ev = self.ev + 1
        hist = self.ev_lines
        if ev < len(hist):
            if hist[ev] != line and self.diverged is None:
                self.diverged = ev
        else:
            hist.append(line)
            self.clock.run()                # past the history: the program runs live
        if ev % self.every == 0 and ev not in self.snaps:
            self.take_snapshot()
        if self.stop(ev, line, n, top) or self.interrupted:
            raise _Pause(True)

    # ── running ─────────────────────────────────────────────────────────────
    def resume(self, stop: Callable[[int, int, int, list], bool]) -> Optional[int]:
        """Run until stop(event, line, depth, frame) is true at a line event (-> None)
        or the program ends (-> its exit code; self.finished says how)."""
        if self.finished is not None:
            return self.finished[0]
        self.stop = stop
        self.interrupted = False
        self.dbg_skip = self.paused and self.at_event
        self.paused = False
        if self.ev + 1 >= len(self.ev_lines):
            self.clock.run()                # at the newest point of the history: live
        try:
            return self._drive()
        except _Pause as p:
            self.paused = True
            self.at_event = p.args[0]
            return None
        finally:
            self.clock.halt()

    def return_stop(self, caller: Optional[list]):
        """`out`: also stop when the program gets back to `caller` (at its saved
        return pc, in the middle of the line of the call); None: no such stop."""
        if self.pseudo_pc is not None:
            del self.dbg_lines[self.pseudo_pc]
            self.pseudo_pc = None
        self.ret_stop = None
        if caller is not None:
            self.ret_stop = (caller, caller[1])
            if caller[1] not in self.dbg_lines:
                self.dbg_lines[caller[1]] = None
                self.pseudo_pc = caller[1]

    def position(self) -> int:
        """Line events strictly before the point where the program is."""
        return self.ev if self.paused and self.at_event else self.ev + 1

    def line_at(self, frame, pc: int) -> int:
        """Source line of `pc` in the function of `frame` (0: none)."""
        if frame[0] in self.hidden_funcs:
            return 0
        f = self.m.funcs[frame[0]]
        rel, best = pc - f.offset, 0
        for off, line in f.lines:
            if off > rel:
                break
            best = line
        return best

    def _drive(self) -> int:
        """TILVM._run() with the debug execute()."""
        try:
            while True:
                try:
                    self._execute()
                    return self._end(0, 'exit')
                except til.TILPanic as p:
                    if not self.recover_from(p):
                        self.report(b"runtime error: " + p.msg, p.pc)
                        return self._end(1, 'error')
        except til.TILFatal as f:
            self.report(b"fatal error: " + f.msg, f.pc)
            return self._end(1, 'error')
        except til.TILExit as e:
            return self._end(e.code, 'exit')
        except RecursionError:
            self.report(b"fatal error: native recursion limit", -1)
            return self._end(1, 'error')

    def _end(self, code: int, kind: str) -> int:
        self.finished = (code, kind)
        return code

    # ── snapshots ───────────────────────────────────────────────────────────
    def take_snapshot(self):
        t0 = time.perf_counter()
        objs, size = copy_state([getattr(self, a) for a in self.STATE_OBJECTS])
        self.snaps[self.ev] = (objs, tuple(getattr(self, a) for a in self.STATE_NUMBERS), size)
        self.snap_slots += size
        t1 = time.perf_counter()
        self.snap_time += t1 - t0
        # copying should not dominate: copy less often when it takes longer than running
        if t1 - t0 > t0 - self.last_snap_at:
            self.every *= 2
        self.last_snap_at = t1
        while len(self.snaps) > MAX_SNAPSHOTS or (self.snap_slots > SNAPSHOT_SLOTS and len(self.snaps) > 2):
            self.every *= 2
            for k in [k for k in self.snaps if k % self.every]:
                self.snap_slots -= self.snaps.pop(k)[2]
            if len(self.snaps) <= 1:
                break

    def restore(self, ev: int):
        """Go back to the state of the newest snapshot at or before line event ev;
        returns the event it restored."""
        at = max(k for k in self.snaps if k <= ev)
        objs, numbers, _size = self.snaps[at]
        objs, _size = copy_state(objs)      # the snapshot itself stays as it is
        for name, value in zip(self.STATE_OBJECTS, objs):
            setattr(self, name, value)
        for name, value in zip(self.STATE_NUMBERS, numbers):
            setattr(self, name, value)
        self.paused = self.at_event = True
        self.finished = None
        return at

    def travel(self, target: int) -> Optional[int]:
        """Stop at line event `target` of the history (restore + run forward)."""
        if self.restore(target) == target:
            return None
        return self.resume(lambda ev, line, depth, top: ev >= target)


# ═══════════════════════════════════════════════════════════════════════════
# the command line of the debugger
# ═══════════════════════════════════════════════════════════════════════════

COMMANDS = [
    ('b LINE', "set a breakpoint (b alone: list them)"),
    ('d LINE', "delete a breakpoint (d alone: delete all)"),
    ('c', "continue to the next breakpoint"),
    ('s', "step to the next line, into calls"),
    ('n', "step to the next line, over calls"),
    ('out', "run until the current function returns"),
    ('back', "step back one line"),
    ('rc', "reverse-continue to the previous breakpoint"),
    ('p NAME', "print a variable (also NAME.field, NAME[i])"),
    ('locals', "print the local variables"),
    ('bt', "show the call stack"),
    ('q', "quit"),
]
SHORT_HELP = "commands: b LINE, d LINE, c, s, n, out, back, rc, p NAME, locals, bt, q (help: details)"

ALIASES = {
    'break': 'b', 'delete': 'd', 'continue': 'c', 'step': 's', 'next': 'n', 'finish': 'out',
    'reverse-step': 'back', 'reverse-continue': 'rc', 'print': 'p', 'backtrace': 'bt',
    'where': 'bt', 'quit': 'q', 'exit': 'q', 'h': 'help', '?': 'help', 'info': 'locals',
}

_PATH = re.compile(r'\s*(\$?[^\W\d]\w*|\$\d+)((?:\s*(?:\.\s*[^\W\d]\w*|\[[^\]]*\]))*)\s*\Z')
_STEP = re.compile(r'\s*(?:\.\s*([^\W\d]\w*)|\[([^\]]*)\])')


class DebugError(Exception):
    """A command that cannot be done (the message is shown, the session goes on)."""


class Session:
    def __init__(self, vm: DebugVM, source_lines: Optional[List[str]], write: Callable[[bytes], None],
                 flush: Callable[[], None]):
        self.vm = vm
        self.source = source_lines
        self._write = write
        self.partial = False        # the program's output so far ends inside a line
        self.flush = flush
        self.breakpoints: Dict[int, int] = {}      # line -> breakpoint number
        self.next_bp = 1
        self.code_lines = sorted({line for line in vm.dbg_lines.values() if line})

    # ── output ──────────────────────────────────────────────────────────────
    def say(self, text: str):
        """A debugger message (localized)."""
        self.raw(_(text))

    def raw(self, text: str):
        self.end_program_line()
        self._write(text.encode('utf-8', 'replace') + b"\n")

    def program_output(self, data: bytes):
        if data:
            self._write(data)
            self.partial = not data.endswith(b"\n")

    def end_program_line(self):
        """Debugger text starts on a line of its own (after `input("name? ")`, say)."""
        if self.partial:
            self._write(b"\n")
            self.partial = False

    # ── where the program is ────────────────────────────────────────────────
    def func_name(self, frame) -> str:
        return self.vm.m.funcs[frame[0]].name

    def frame_line(self, depth: int) -> int:
        """Line where frame `depth` is: the current line event for the top frame,
        else the line of the call (or of the instruction that ended the program)."""
        vm = self.vm
        frame = vm.frames[depth]
        if depth == len(vm.frames) - 1 and vm.paused and vm.at_event and 0 <= vm.ev < len(vm.ev_lines):
            return vm.ev_lines[vm.ev]
        return vm.line_at(frame, frame[1] - 1)

    def current_line(self) -> int:
        return self.frame_line(len(self.vm.frames) - 1) if self.vm.frames else 0

    def show_stop(self, result: Optional[int], returned: Optional[str] = None):
        vm = self.vm
        if vm.diverged is not None:
            self.say(f"Warning: the replay differs from the recorded run at line event {vm.diverged}")
            vm.diverged = None
        if result is not None:
            code, kind = vm.finished
            if kind == 'error':
                self.say(f"Program ended with an error (exit code {code})")
            else:
                self.say(f"Program exited with code {code}")
            return
        if vm.interrupted:
            vm.interrupted = False
            self.say("Interrupted")
        line = self.current_line()
        func = self.func_name(vm.frames[-1])
        bp = self.breakpoints.get(line) if vm.at_event else None
        if bp is not None:
            self.say(f"Breakpoint {bp}, {func} at line {line}")
        elif line:
            self.say(f"{func} at line {line}")
        else:
            self.say(f"{func} (no source line)")
        if self.source and 0 < line <= len(self.source):
            self.raw(f"{line:5d} | {self.source[line - 1].rstrip()}")
        if returned is not None and not vm.at_event and vm.stack:
            self.say(f"{returned} returned {self.short(vm.stack[-1])}")

    # ── commands ────────────────────────────────────────────────────────────
    def command(self, text: str) -> bool:
        """Do one command; False = quit."""
        words = text.split(None, 1)
        if not words:
            return True
        cmd = ALIASES.get(words[0], words[0])
        arg = words[1].strip() if len(words) > 1 else ""
        try:
            if cmd == 'q':
                return False
            handler = getattr(self, 'cmd_' + cmd, None) if cmd.isidentifier() else None
            if handler is None:
                self.say(f"Unknown command '{words[0]}'")
                self.say(SHORT_HELP)
            else:
                handler(arg)
        except DebugError as e:
            self.say(str(e))
        return True

    def cmd_help(self, arg: str):
        self.say("Commands:")
        for name, text in COMMANDS:
            self.raw(f"  {name:<9}{_(text)}")

    def line_arg(self, arg: str, cmd: str) -> int:
        try:
            line = int(arg)
        except ValueError:
            raise DebugError(f"Usage: {cmd} LINE")
        return line

    def cmd_b(self, arg: str):
        if not arg:
            if not self.breakpoints:
                self.say("No breakpoints")
            for line, n in sorted(self.breakpoints.items(), key=lambda kv: kv[1]):
                self.say(f"Breakpoint {n} at line {line}")
            return
        asked = self.line_arg(arg, 'b')
        later = [ln for ln in self.code_lines if ln >= asked]
        if not later:
            raise DebugError(f"No code at or after line {asked}")
        line = later[0]
        if line not in self.breakpoints:
            self.breakpoints[line] = self.next_bp
            self.next_bp += 1
        n = self.breakpoints[line]
        if line == asked:
            self.say(f"Breakpoint {n} at line {line}")
        else:
            self.say(f"Breakpoint {n} at line {line} (line {asked} has no code)")

    def cmd_d(self, arg: str):
        if not arg:
            self.breakpoints.clear()
            self.say("Deleted all breakpoints")
            return
        line = self.line_arg(arg, 'd')
        if line not in self.breakpoints:
            raise DebugError(f"No breakpoint at line {line}")
        n = self.breakpoints.pop(line)
        self.say(f"Deleted breakpoint {n} at line {line}")

    def forward(self, stop, returned: Optional[str] = None):
        vm = self.vm
        if vm.finished is not None:
            raise DebugError("The program has finished (back and rc go back in time, q quits)")
        self.show_stop(vm.resume(stop), returned)

    def cmd_c(self, arg: str):
        bps = self.breakpoints
        self.forward(lambda ev, line, depth, top: line in bps)

    def cmd_s(self, arg: str):
        self.forward(lambda ev, line, depth, top: True)

    def cmd_n(self, arg: str):
        bps = self.breakpoints
        frames = list(self.vm.frames)
        d0 = len(frames)
        self.forward(lambda ev, line, depth, top: line in bps or (depth <= d0 and frames[depth - 1] is top))

    def cmd_out(self, arg: str):
        vm = self.vm
        bps = self.breakpoints
        frames = list(vm.frames)
        d0 = len(frames)
        if vm.finished is None:
            vm.return_stop(frames[-2] if d0 >= 2 else None)
        try:
            self.forward(lambda ev, line, depth, top: line in bps or (depth < d0 and frames[depth - 1] is top),
                         self.func_name(frames[-1]) if frames else None)
        finally:
            vm.return_stop(None)

    def cmd_back(self, arg: str):
        target = self.vm.position() - 1
        if target < 0:
            raise DebugError("Already at the start of the program")
        self.show_stop(self.vm.travel(target))

    def cmd_rc(self, arg: str):
        vm = self.vm
        cur = vm.position()
        hist = vm.ev_lines
        bps = self.breakpoints
        for i in range(cur - 1, -1, -1):
            if hist[i] in bps:
                self.show_stop(vm.travel(i))
                return
        if cur <= 0:
            raise DebugError("Already at the start of the program")
        self.say("No earlier breakpoint hit: back to the start of the program")
        self.show_stop(vm.travel(0))

    # ── variables ───────────────────────────────────────────────────────────
    def frame_vars(self, frame, at_call: bool = False) -> List[Tuple[str, Any]]:
        """(name, value) of the locals live in `frame` (innermost declaration of
        each name), then its upvalues."""
        vm = self.vm
        f = vm.m.funcs[frame[0]]
        rel = frame[1] - f.offset - (1 if at_call else 0)
        live: Dict[str, list] = {}
        for rec in f.locals:
            slot, name, start, end = rec
            if start <= rel < end and (name not in live or start >= live[name][2]):
                live[name] = rec
        out = []
        for rec in sorted(live.values(), key=lambda r: r[0]):
            out.append((rec[1], self.slot_value(frame, rec[0])))
        closure = frame[3]
        if closure is not None:
            for k, name in enumerate(f.upval_names):
                if k < len(closure.up) and name not in live:
                    out.append((name, closure.up[k].v))
        return out

    def mid_line(self) -> bool:
        """The top frame's pc is just after an instruction (a call `out` came back
        from, or the one that ended the program), not at a line event."""
        return not (self.vm.paused and self.vm.at_event)

    def slot_value(self, frame, slot: int):
        v = self.vm.stack[frame[2] + slot]
        return v.v if type(v) is til.VMCell else v

    def lookup(self, name: str):
        vm = self.vm
        if vm.frames:
            frame = vm.frames[-1]
            for n, v in self.frame_vars(frame, self.mid_line()):
                if n == name:
                    return v
            if name.startswith('$') and name[1:].isdigit():     # a slot, without debug info
                slot = int(name[1:])
                if slot < vm.m.funcs[frame[0]].n_locals and frame[2] + slot < len(vm.stack):
                    return self.slot_value(frame, slot)
        if name in vm.m.globals:
            return vm.globals[vm.m.globals.index(name)]
        raise DebugError(f"No variable '{name}' here")

    def show(self, v, quote: bool = False) -> str:
        if v is til._UNDEF:
            return "<no value>"
        text = self.vm.to_str(v, 0, quote).decode('utf-8', 'replace')
        if len(text) > MAX_SHOWN:
            text = text[:MAX_SHOWN] + _(f"... ({len(text) - MAX_SHOWN} more characters)")
        return text

    def evaluate(self, expr: str):
        m = _PATH.match(expr)
        if not m:
            raise DebugError(f"Cannot evaluate '{expr.strip()}': use NAME, NAME.field or NAME[index]")
        vm = self.vm
        v = self.lookup(m.group(1))
        for field, index in _STEP.findall(m.group(2)):
            try:
                if field:
                    if type(v) is til.VMStruct and field.encode() in vm.field_names[v.t]:
                        v = v.f[vm.field_names[v.t].index(field.encode())]
                    elif type(v) is dict and field.encode() in v:
                        v = v[field.encode()]
                    else:
                        raise til.TILPanic(f"no field '{field}' on {vm.type_name(v).decode()}")
                else:
                    key = index.strip()
                    if re.fullmatch(r'-?\d+', key):
                        k: Any = int(key)
                    elif len(key) >= 2 and key[0] == key[-1] and key[0] in '"\'':
                        k = key[1:-1].encode('utf-8')
                    else:
                        k = self.evaluate(key)
                    v = vm.index_get(v, k)
            except til.TILPanic as p:
                raise DebugError(f"Cannot evaluate '{expr.strip()}': {p.msg.decode('utf-8', 'replace')}")
        return v

    def cmd_p(self, arg: str):
        if not arg:
            raise DebugError("Usage: p NAME")
        self.raw(f"{arg} = {self.show(self.evaluate(arg))}")

    def cmd_locals(self, arg: str):
        vm = self.vm
        pairs = self.frame_vars(vm.frames[-1], self.mid_line()) if vm.frames else []
        if not pairs:
            self.say("No local variables")
        for name, v in pairs:
            self.raw(f"{name} = {self.show(v)}")

    def cmd_bt(self, arg: str):
        vm = self.vm
        for depth in range(len(vm.frames) - 1, -1, -1):
            frame = vm.frames[depth]
            f = vm.m.funcs[frame[0]]
            line = self.frame_line(depth)
            names: Dict[int, str] = {}
            for slot, name, _start, _end in f.locals:
                if slot < f.n_params:
                    names.setdefault(slot, name)
            shown = ", ".join(f"{names[s]}={self.short(self.slot_value(frame, s))}" for s in sorted(names))
            self.say(f"  at {f.name}({shown})" + (f" (line {line})" if line else ""))

    def short(self, v) -> str:
        text = self.show(v, quote=True).replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r')
        return text if len(text) <= 60 else text[:57] + "..."


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def load_module(path: str, fuse: bool = True,
                pragma_language: bool = False) -> Tuple['til.TBCModule', Optional[List[str]]]:
    """The program as the debugger runs it: compiled with debug info (a .til) or
    read from a .tbc; always through the bytes of a .tbc file, so the debugger
    sees exactly what the format holds. pragma_language: the program's
    #[lang: ...] chooses the message language (main() only)."""
    if path.endswith('.tbc'):
        with open(path, 'rb') as f:
            data = f.read()
        source = None
        base = os.path.splitext(path)[0] + '.til'
        if os.path.exists(base):
            with open(base, encoding='utf-8', errors='replace') as f:
                source = f.read().split('\n')
        return til.TBCModule.from_bytes(data), source
    with open(path, encoding='utf-8') as f:
        text = f.read()
    if pragma_language:
        til.use_program_message_language(text)
    module = til.compile_to_bytecode(text, path, True, fuse, debug_info=True)
    return til.TBCModule.from_bytes(module.serialize()), text.split('\n')


class _Lines:
    """Lines of a binary stream (standard input or a file) for commands and input()."""

    def __init__(self, stream):
        self.stream = stream

    def readline(self) -> Optional[bytes]:
        """The next line without its newline; None at the end."""
        try:
            line = self.stream.readline()
        except (OSError, ValueError):
            return None
        if not line:
            return None
        return line[:-1] if line.endswith(b"\n") else line


def parse_args(argv: List[str]) -> Dict[str, Any]:
    opts: Dict[str, Any] = {'program': None, 'commands': None, 'input': None, 'keys': None,
                            'screen': None, 'every': SNAPSHOT_EVERY, 'fuse': True, 'args': [],
                            'screen_ansi': False}
    values = {'--commands': 'commands', '--input': 'input', '--keys': 'keys', '--screen': 'screen',
              '--snapshot-every': 'every'}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--':
            opts['args'].extend(argv[i + 1:])
            break
        key, value = a, None
        if a.startswith('--') and '=' in a:
            key, value = a.split('=', 1)
        if key in values:
            if value is None:
                if i + 1 >= len(argv):
                    raise ValueError(f"{key} needs a value")
                i += 1
                value = argv[i]
            opts[values[key]] = value
        elif a in ('-h', '--help'):
            opts['help'] = True
        elif a == '--no-fuse':
            opts['fuse'] = False
        elif a == '--screen-ansi':
            opts['screen_ansi'] = True
        elif a.startswith('-') and a != '-':
            raise ValueError(f"unknown option {a}")
        elif opts['program'] is None:
            opts['program'] = a
        else:
            opts['args'].append(a)
        i += 1
    try:
        opts['every'] = int(opts['every'])
        if opts['every'] < 1:
            raise ValueError
    except ValueError:
        raise ValueError("--snapshot-every needs a positive number")
    return opts


def _isatty(stream) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def split_script(script: str) -> List[str]:
    return [c.strip() for c in re.split(r'[;\n]', script) if c.strip()]


@til.scoped_message_language          # the language chosen here lasts for this call only
def main(argv: List[str]) -> int:
    try:
        return _main(argv)
    except BrokenPipeError:             # `til debug ... | head`
        try:
            sys.stdout = open(os.devnull, 'w')
        except OSError:
            pass
        return 0


def _main(argv: List[str]) -> int:
    if any(a == '--lang' or a.startswith('--lang=') for a in argv[:argv.index('--') if '--' in argv else None]):
        err = til.set_message_language(argv)        # called directly (til.py strips --lang)
        if err:
            print(err, file=sys.stderr)
            return 2
        argv = til.without_lang_option(argv)
    try:
        opts = parse_args(argv)
    except ValueError as e:
        print(_(f"Error: {e}"), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if opts.get('help'):
        print(USAGE)
        return 0
    path = opts['program']
    if path is None:
        print(_("Error: no program given"), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if not os.path.exists(path):
        print(_(f"Error: File not found: {path}"), file=sys.stderr)
        return 2
    try:
        module, source = load_module(path, opts['fuse'], pragma_language=True)
        screen = til.TILScreen(ansi=opts['screen_ansi'])
        keys = til.KeyScript(til.parse_key_script(opts['keys'] or ''))
        debug_execute_code()
    except SyntaxError as e:
        print(_(f"Syntax Error: {e}"), file=sys.stderr)
        return 1
    except til.CompileError as e:
        print(_(str(e)), file=sys.stderr)
        return 1
    except (ValueError, OSError, UnicodeDecodeError, RuntimeError) as e:
        print(_(f"Error: {e}"), file=sys.stderr)
        return 1

    stdout = getattr(sys.stdout, 'buffer', None)

    def write(data: bytes):
        if stdout is not None:
            stdout.write(data)
        else:
            sys.stdout.write(data.decode('utf-8', 'replace'))

    def flush():
        try:
            sys.stdout.flush()
        except (OSError, ValueError):
            pass
    stdin = _Lines(getattr(sys.stdin, 'buffer', sys.stdin))
    program_input = stdin
    input_file = None
    if opts['input']:
        try:
            input_file = open(opts['input'], 'rb')
        except OSError as e:
            print(_(f"Error: {e}"), file=sys.stderr)
            return 2
        program_input = _Lines(input_file)

    def read_line() -> bytes:
        flush()
        line = program_input.readline()
        return b"" if line is None else line

    sessions: List[Session] = []
    vm = DebugVM(module, lambda data: sessions[0].program_output(data), read_line, argv=opts['args'],
                 screen=screen, keys=keys, every=opts['every'], on_report=flush)
    session = Session(vm, source, write, flush)
    sessions.append(session)
    script = split_script(opts['commands']) if opts['commands'] is not None else None
    interactive = script is None
    old_sigint = None
    if interactive:
        def on_sigint(signum, frame):
            vm.interrupted = True
        try:
            old_sigint = signal.signal(signal.SIGINT, on_sigint)
        except (ValueError, OSError, AttributeError):
            old_sigint = None
        session.say("TIL time-travel debugger (type help for the commands)")
    tty = interactive and input_file is None and _isatty(sys.stdin)
    if tty:
        try:
            import readline  # noqa: F401  (line editing and history for input())
        except ImportError:
            pass

    def read_command() -> Optional[str]:
        if tty:                         # the terminal: input() edits the line
            session.end_program_line()
            flush()
            try:
                return input("(til) ")
            except EOFError:
                return None
        session.end_program_line()
        write(b"(til) ")
        flush()
        got = stdin.readline()
        return None if got is None else got.decode('utf-8', 'replace')
    try:
        if not vm.m.lines:
            session.say("The program has no line table (built with --strip): it has no line events")
        session.show_stop(vm.resume(lambda ev, line, depth, top: True))
        last = ""
        while True:
            if script is not None:
                if not script:
                    break
                cmd = script.pop(0)
                session.raw(f"(til) {cmd}")
            else:
                got = read_command()
                if got is None:
                    write(b"\n")
                    break
                cmd = got.strip() or last          # an empty line repeats the last command
                last = cmd
            if not session.command(cmd):
                break
            flush()
    except KeyboardInterrupt:
        write(b"\n")
        return 130
    finally:
        flush()
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)
        if input_file is not None:
            input_file.close()
    if opts['screen']:
        rc = til.save_screen(vm.screen, opts['screen'], 0)
        if rc:
            return rc
    return 0


if __name__ == '__main__':
    sys.modules.setdefault('tildebug', sys.modules[__name__])
    sys.exit(main(sys.argv[1:]))
