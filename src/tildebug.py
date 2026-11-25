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

Three layers:

* DebugVM       the VM: line events, snapshots, record/replay, output once;
* DebugSession  the debugging engine, for programs: breakpoints (conditions,
                hit counts), stepping forwards and backwards, goto(event), the
                call stack, variables, structured values, expressions. `til
                debug` (Session) and the debug adapter `til dap` (src/tildap.py)
                are built on it;
* Session       the command line of `til debug`.
"""

import bisect
import copy
import inspect
import os
import re
import signal
import sys
import textwrap
import threading
import time
import types
from array import array
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
PREVIEW = 200                   # characters of a value in a variables view (DebugSession.preview)
MAX_OUTPUT_KEPT = 16 << 20      # bytes of program output a DebugSession keeps (DebugSession.output)
MAX_EVAL_ITEMS = 1_000_000      # elements an evaluated expression may build (`[0] * n`, range())


def _(text: str) -> str:
    """A debugger message in the language of TIL's messages (src/tilmsg.py)."""
    return til.localize(text)


class DebugError(Exception):
    """A command that cannot be done (the message is shown, the session goes on)."""


class Killed(Exception):
    """The program was ended from outside (DebugSession.kill(): a debug adapter's
    terminate / disconnect). Raised out of the run methods."""


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
    line memory, the position in the input recording, in the program's output
    and in its screen_show() calls. The key source and stdin are outside
    streams: what the program read from them is in the recording, and after the
    recorded part they continue where they are. Energy saving counts its idles
    but does not sleep here (a debugging session saves no battery, and history
    re-runs stay fast).

    For hosts (DebugSession, the debug adapter): `err` receives the error report
    of the end (else it goes to stderr, as `til run` prints it), `on_screen` is
    called at every screen_show() past the furthest point of the run, `cwd`
    is where relative file paths start, `lang` the language of the report;
    kill() and request_pause() may be called from any thread."""

    STATE_OBJECTS = ('stack', 'frames', 'globals', 'handlers', 'recover', 'screen', 'shadow')
    STATE_NUMBERS = ('deadlines', 'tick', 'steps', 'rng', 'energy', 'idle_count', 'ev', 'nd_pos',
                     'out_pos', 'shows')

    def __init__(self, module: 'til.TBCModule', out: Callable[[bytes], None],
                 read_line: Callable[[], bytes], argv=None, screen=None, keys=None,
                 every: int = SNAPSHOT_EVERY, on_report: Optional[Callable[[], None]] = None,
                 err: Optional[Callable[[bytes], None]] = None, cwd: Optional[str] = None,
                 on_screen: Optional[Callable[['til.TILScreen'], None]] = None,
                 lang: Optional[str] = None):
        super().__init__(module, write=self._program_write, read_line=read_line, argv=argv,
                         screen=screen, keys=keys)
        self.out = out
        self.on_report = on_report
        self.err = err
        self.cwd = cwd
        self.on_screen = on_screen
        self.lang = lang
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
        # per line event: 2 * frame depth + 1 if it is the first line event of its frame
        # (reverse "step over" finds the line events of a frame and of its callers)
        self.ev_meta = array('I')
        self.shadow: List[list] = []    # [frame, last line] per frame: what is a new line
        self.out_pos = 0                # bytes of output this run has produced
        self.out_printed = 0            # bytes of output printed (only these are new)
        self.shows = 0                  # screen_show() calls this run has made
        self.shows_seen = 0             # ... the most ever made (only later ones are new)
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
        self.pause_pending = False      # request_pause() before resume() started
        self.killed = False             # kill(): end the run (raises Killed)
        self.kill_event = threading.Event()
        self.finished: Optional[Tuple[int, str]] = None   # (exit code, 'exit' | 'error')
        self.reported = False           # the error report of the end was printed
        self.end_error: Optional[Tuple[bytes, int]] = None   # (message, pc) of that error
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
        self.end_error = (msg, pc)
        if self.on_report:
            self.on_report()
        if self.err is None:
            super().report(msg, pc)
            return
        text = msg + b"\n" + (self.traceback(pc) + b"\n" if pc >= 0 else b"")
        self.err(til.localize_report_bytes(text, self.lang))

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
            ms = args[0]
            if type(ms) is not int:             # the checks of TILVM.call_native
                self.panic("sleep_ms() needs an integer")
            ms = min(max(0, ms), til.SLEEP_MAX_MS)
            return self.nd('sleep', lambda: self.live_sleep(ms / 1000.0))
        if name in ('read_file', 'file_exists', 'write_file'):
            self.path_arg(name, args[0])            # its errors come from the program
        return self.nd(name, lambda: til.TILVM.call_native(self, nid, args))

    def live_sleep(self, seconds: float):
        """sleep_ms() of the live run; kill() wakes it (and ends the run)."""
        end = time.monotonic() + seconds
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return None
            if self.kill_event.wait(left):
                raise Killed()

    def path_arg(self, who: str, v) -> str:
        path = til.TILVM.path_arg(self, who, v)
        if self.cwd and not os.path.isabs(path):
            path = os.path.join(self.cwd, path)
        return path

    def gfx_native(self, name: str, args: list):
        r = til.TILVM.gfx_native(self, name, args)
        if name == 'screen_show':
            self.shows += 1
            if self.shows > self.shows_seen:
                self.shows_seen = self.shows
                if self.on_screen is not None:
                    self.on_screen(self.screen)
        return r

    def idle(self, ms: int):
        pass                            # energy saving: counted (energy_point), no sleep

    def check_budget(self, pc: int):
        self.steps += 1
        if self._slow:                  # a step limit or energy saving (TILVM.check_budget); kill()
            if self.killed:
                raise Killed()
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

    # ── control from other threads ──────────────────────────────────────────
    def request_pause(self):
        """Stop at the next line event (also when the run has not started yet)."""
        self.pause_pending = True
        self.interrupted = True

    def kill(self):
        """End the run: the run methods raise Killed (at the next line event, loop
        back-edge or call, or in sleep_ms())."""
        self.killed = True
        self._slow = True               # check_budget looks at `killed`
        self.interrupted = True
        self.kill_event.set()

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
            first = 0
        else:
            while k < n - 1:
                sh.append([frames[k], 0])
                k += 1
            sh.append([top, line])
            first = 1
        ev = self.ev = self.ev + 1
        hist = self.ev_lines
        if ev < len(hist):
            if hist[ev] != line and self.diverged is None:
                self.diverged = ev
        else:
            hist.append(line)
            self.ev_meta.append(n + n + first)
            self.clock.run()                # past the history: the program runs live
        if ev % self.every == 0 and ev not in self.snaps:
            self.take_snapshot()
        if self.stop(ev, line, n, top) or self.interrupted:
            raise _Pause(True)

    # ── running ─────────────────────────────────────────────────────────────
    def resume(self, stop: Callable[[int, int, int, list], bool]) -> Optional[int]:
        """Run until stop(event, line, depth, frame) is true at a line event (-> None)
        or the program ends (-> its exit code; self.finished says how)."""
        if self.killed:
            raise Killed()
        if self.finished is not None:
            return self.finished[0]
        self.stop = stop
        self.interrupted = self.pause_pending
        self.dbg_skip = self.paused and self.at_event
        self.paused = False
        if self.ev + 1 >= len(self.ev_lines):
            self.clock.run()                # at the newest point of the history: live
        try:
            return self._drive()
        except _Pause as p:
            self.paused = True
            self.at_event = p.args[0]
            self.pause_pending = False
            if self.killed:
                raise Killed()
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
# values: previews and children (for variables views)
# ═══════════════════════════════════════════════════════════════════════════

def _escape(text: str) -> str:
    """Control characters of a string value as escapes (a preview stays one line)."""
    if text.isprintable() and '\\' not in text:
        return text
    out = []
    for ch in text:
        o = ord(ch)
        if ch == '\\':
            out.append('\\\\')
        elif ch == '\n':
            out.append('\\n')
        elif ch == '\r':
            out.append('\\r')
        elif ch == '\t':
            out.append('\\t')
        elif o < 32 or o == 127:
            out.append('\\x%02x' % o)
        else:
            out.append(ch)
    return ''.join(out)


def preview(vm: 'til.TILVM', v, limit: int = PREVIEW) -> str:
    """A value as print() writes it, strings quoted (also at the top) with their
    control characters escaped; at most `limit` characters (then "..."). Big
    values cost only what is shown."""
    out: List[str] = []
    room = [limit + 1]
    UNDEF = til._UNDEF

    def put(s: str):
        out.append(s)
        room[0] -= len(s)

    def walk(v, depth: int):
        if room[0] <= 0:
            return
        if v is None:
            put("None")
            return
        if v is True:
            put("true")
            return
        if v is False:
            put("false")
            return
        tv = type(v)
        if tv is int:
            put(str(v))
            return
        if tv is float:
            put(til.fmt_g(v).decode('ascii', 'replace'))
            return
        if tv is bytes:
            put('"' + _escape(v[:4 * room[0] + 4].decode('utf-8', 'replace')) + '"')
            return
        if v is UNDEF:
            put("<no value>")
            return
        if tv is til._MapBool:
            put("true" if v.v else "false")
            return
        if tv is til.VMClosure:
            put("<fn " + vm.m.funcs[v.fn].name + ">")
            return
        if depth > 16:
            put("...")
            return
        if tv is list:
            put("[")
            for i, x in enumerate(v):
                if room[0] <= 0:
                    break
                if i:
                    put(", ")
                walk(x, depth + 1)
            put("]")
        elif tv is dict:
            put("{")
            for i, (k, x) in enumerate(v.items()):
                if room[0] <= 0:
                    break
                if i:
                    put(", ")
                walk(k, depth + 1)
                put(": ")
                walk(x, depth + 1)
            put("}")
        elif tv is til.VMStruct:
            name = vm.type_names[v.t]
            fields = vm.field_names[v.t]
            if v.t <= til.TYPE_ERR or (fields and fields[0] == b"0"):
                if not fields:
                    put(name)
                    return
                put(name + "(")
                for i, x in enumerate(v.f):
                    if i:
                        put(", ")
                    walk(x, depth + 1)
                put(")")
            elif not fields:
                put(name)
            else:
                put(name + " { ")
                for i, (fn, x) in enumerate(zip(fields, v.f)):
                    if i:
                        put(", ")
                    put(fn.decode('utf-8', 'replace') + ": ")
                    walk(x, depth + 1)
                put(" }")
        else:
            put("?")

    walk(v, 0)
    text = "".join(out)
    return text if len(text) <= limit else text[:limit] + "..."


def value_type(vm: 'til.TILVM', v) -> str:
    """The type of a value as type_of() names it ('int', 'str', 'array', 'Point', ...)."""
    if v is til._UNDEF:
        return ""
    if type(v) is til._MapBool:
        return "bool"
    return vm.type_name(v).decode('utf-8', 'replace')


def value_kind(vm: 'til.TILVM', v) -> str:
    """What a value is, for a variables view: 'none', 'bool', 'int', 'float',
    'string', 'array', 'map', 'struct', 'variant' (an enum value with fields),
    'option' (Some), 'result' (Ok / Err), 'closure' or 'undefined'."""
    if v is None:
        return 'none'
    tv = type(v)
    if tv is bool or tv is til._MapBool:
        return 'bool'
    if tv is int:
        return 'int'
    if tv is float:
        return 'float'
    if tv is bytes:
        return 'string'
    if tv is list:
        return 'array'
    if tv is dict:
        return 'map'
    if tv is til.VMClosure:
        return 'closure'
    if tv is til.VMStruct:
        if v.t == til.TYPE_SOME:
            return 'option'
        if v.t in (til.TYPE_OK, til.TYPE_ERR):
            return 'result'
        fields = vm.field_names[v.t]
        return 'variant' if (not fields or fields[0] == b"0") and '.' in vm.type_names[v.t] else 'struct'
    return 'undefined'


def _til_literal(v) -> Optional[str]:
    """A TIL literal for a map key (an evaluateName `m["k"]`), None if there is none."""
    if v is None:
        return "None"
    if v is True or v is False:
        return "true" if v else "false"
    if type(v) is int:
        return str(v)
    if type(v) is float:
        text = til.fmt_g(v).decode('ascii', 'replace')
        return text if re.fullmatch(r'-?\d+(\.\d+)?(e[-+]?\d+)?', text) else None
    if type(v) is bytes:
        try:
            s = v.decode('utf-8')
        except UnicodeDecodeError:
            return None
        out = []
        for ch in s:
            if ch in '\\"':
                out.append('\\' + ch)
            elif ch == '\n':
                out.append('\\n')
            elif ch == '\t':
                out.append('\\t')
            elif ch == '\r':
                out.append('\\r')
            elif ord(ch) < 32:
                out.append('\\x%02x' % ord(ch))
            else:
                out.append(ch)
        return '"' + ''.join(out) + '"'
    return None


def value_children(vm: 'til.TILVM', v, start: int = 0,
                   count: Optional[int] = None) -> List[Tuple[str, Any, Optional[str]]]:
    """The parts of a structured value: (name, value, evaluate suffix) for the
    elements of an array (`[0]`), the entries of a map (`"key"` -> `["key"]`),
    the fields of a struct (`x` -> `.x`), Some/Ok/Err (`value` / `error`), an
    enum value with fields (`0`, `1`, ... -> `[0]`) and the captured variables
    of a closure (no suffix). `start`/`count` page through arrays and maps."""
    tv = type(v)
    stop = None if count is None or count <= 0 else start + count
    if tv is list:
        end = len(v) if stop is None else min(len(v), stop)
        return [(f"[{i}]", v[i], f"[{i}]") for i in range(max(start, 0), end)]
    if tv is dict:
        out = []
        for i, (k, x) in enumerate(v.items()):
            if i < start:
                continue
            if stop is not None and i >= stop:
                break
            key = k.v if type(k) is til._MapBool else k
            lit = _til_literal(key)
            out.append((preview(vm, key, 60), x, None if lit is None else "[" + lit + "]"))
        return out
    if tv is til.VMStruct:
        fields = vm.field_names[v.t]
        if fields and fields[0] == b"0":
            return [(str(i), x, f"[{i}]") for i, x in enumerate(v.f)]
        return [(fn.decode('utf-8', 'replace'), x, "." + fn.decode('utf-8', 'replace'))
                for fn, x in zip(fields, v.f)]
    if tv is til.VMClosure:
        names = vm.m.funcs[v.fn].upval_names
        out = []
        for i, cell in enumerate(v.up):
            value = cell.v if type(cell) is til.VMCell else cell
            out.append((names[i] if i < len(names) else f"up{i}", value, None))
        return out
    return []


def child_counts(vm: 'til.TILVM', v) -> Tuple[int, int]:
    """(indexed, named) parts of a value (both 0: nothing to expand)."""
    tv = type(v)
    if tv is list:
        return len(v), 0
    if tv is dict:
        return 0, len(v)
    if tv is til.VMStruct:
        return 0, len(v.f)
    if tv is til.VMClosure:
        return 0, len(v.up)
    return 0, 0


# ═══════════════════════════════════════════════════════════════════════════
# expressions: what `p`, watch, hover and breakpoint conditions evaluate
# ═══════════════════════════════════════════════════════════════════════════

# a variable and its fields / indexes: x, x.f, x[3], x["k"], x[i], $3 (a slot)
_PATH = re.compile(r'\s*(\$?[^\W\d]\w*|\$\d+)((?:\s*(?:\.\s*[^\W\d]\w*|\[[^\]]*\]))*)\s*\Z')
_STEP = re.compile(r'\s*(?:\.\s*([^\W\d]\w*)|\[([^\]]*)\])')

# builtins without side effects (no output, input, time, files, keys, randomness)
PURE_NATIVES = frozenset((
    'len', 'str', 'int', 'float', 'bool', 'abs', 'sqrt', 'pow', 'sin', 'cos', 'tan', 'log', 'exp',
    'floor', 'ceil', 'round', 'min', 'max', 'sum', 'chr', 'ord', 'range', 'type_of', 'format',
    'str_bytes', 'bytes_to_str', 'args', 'screen_width', 'screen_height', 'get_pixel'))
# builtin methods that do not change their receiver
PURE_METHODS = frozenset((
    'len', 'contains', 'starts_with', 'ends_with', 'trim', 'to_upper', 'to_lower', 'replace',
    'slice', 'find', 'index_of', 'count', 'eq', 'split', 'repeat', 'to_int', 'to_float', 'is_empty',
    'get', 'join', 'first', 'last', 'sum', 'min', 'max', 'has', 'keys', 'values', 'get_or',
    'unwrap', 'expect', 'unwrap_or', 'is_some', 'is_none', 'is_ok', 'is_err', 'error',
    'to_string', 'copy'))
UNSUPPORTED = ("only variables, fields, indexes, literals, operators and calls of builtins "
               "without side effects can be evaluated")
_ANSI = re.compile(r'\x1b\[[0-9;]*m')


class _Expression:
    """An expression prepared for evaluation (DebugSession.prepare)."""
    __slots__ = ('text', 'path', 'node')

    def __init__(self, text: str, path=None, node=None):
        self.text, self.path, self.node = text, path, node


class _Evaluator:
    """Evaluates an expression tree of TIL's parser with the VM's own operations
    (vm.binop, vm.index_get, vm.call_native, ...), so values and errors are
    those of the program; nothing is executed that could change the program's
    state or the world (no calls of program functions, no assignments)."""

    def __init__(self, session: 'DebugSession', ctx, text: str):
        self.s = session
        self.vm = session.vm
        self.ctx = ctx
        self.text = text

    def fail(self, reason: str):
        raise DebugError(f"Cannot evaluate '{self.text}': {reason}")

    def run(self, node):
        try:
            return self.ev(node)
        except til.TILPanic as p:
            self.fail(p.msg.decode('utf-8', 'replace'))
        except RecursionError:
            self.fail(UNSUPPORTED)
        except (TypeError, ValueError, OverflowError, IndexError, KeyError, AttributeError,
                ZeroDivisionError, MemoryError):
            self.fail("invalid operation")

    def ev(self, node):
        t = type(node)
        vm = self.vm
        if t is til.IntLit:
            return til.wrap_int64(node.value)
        if t is til.FloatLit:
            return float(node.value)
        if t is til.StringLit:
            v = node.value
            return v.encode('utf-8') if isinstance(v, str) else bytes(v)
        if t is til.CharLit:
            return ord(node.value)
        if t is til.BoolLit:
            return bool(node.value)
        if t is til.Identifier:
            if node.name == 'NULL':
                return None
            return self.s._lookup(node.name, self.ctx)
        if t is til.BinaryOp:
            op = node.op
            if op == 'and':
                return vm.truthy(self.ev(node.left)) and vm.truthy(self.ev(node.right))
            if op == 'or':
                return vm.truthy(self.ev(node.left)) or vm.truthy(self.ev(node.right))
            code = til.BytecodeCompiler.BINOPS.get(op)
            if code is None:
                self.fail(UNSUPPORTED)
            a = self.ev(node.left)
            b = self.ev(node.right)
            if code == til.Op.MUL and type(b) is int and type(a) in (list, bytes) and \
                    len(a) * max(b, 0) > MAX_EVAL_ITEMS:
                self.fail("the value would be too large")
            return vm.binop(code, a, b)
        if t is til.UnaryOp:
            v = self.ev(node.operand)
            if node.op == '-':
                if type(v) is int:
                    return til.wrap_int64(-v)
                if type(v) is float:
                    return -v
                vm.panic(f"cannot negate {vm.type_name(v).decode()}")
            if node.op == 'not':
                return not vm.truthy(v)
            if node.op == '~':
                if type(v) is not int:
                    vm.panic("~ needs an integer")
                return ~v
            return v                                # & / &mut / *: references already
        if t is til.Index:
            return vm.index_get(self.ev(node.obj), self.ev(node.index))
        if t is til.Attribute:
            return self.attribute(node)
        if t is til.ArrayLit or t is til.TupleLit:
            return [self.ev(e) for e in node.elements]
        if t is til.DictLit:
            d = {}
            for k, v in node.pairs:
                d[vm.map_key(self.ev(k))] = self.ev(v)
            return d
        if t is til.IfExpr:
            return self.ev(node.then_expr) if vm.truthy(self.ev(node.condition)) else self.ev(node.else_expr)
        if t is til.FStringLit:
            parts = []
            for part in node.parts:
                if part[0] == 'str':
                    parts.append(self.ev(part[1]))
                else:
                    v = self.ev(part[1])
                    if len(part) > 2:
                        spec = part[2].encode('utf-8') if isinstance(part[2], str) else part[2]
                        parts.append(til.til_format_spec(v, spec, vm.to_str))
                    else:
                        parts.append(vm.to_str(v))
            return b"".join(parts)
        if t is til.Cast and isinstance(node.target_type, til.PrimitiveType):
            name = node.target_type.name
            kind = ('float' if name in ('float', 'f32', 'f64') else 'str' if name == 'str' else
                    'bool' if name == 'bool' else 'int')
            return vm.cast(til.CAST_KINDS[kind], self.ev(node.expr))
        if t is til.Call:
            return self.call(node)
        self.fail(UNSUPPORTED)

    def attribute(self, node):
        vm = self.vm
        obj = node.obj
        if type(obj) is til.Identifier and obj.name != 'NULL' and not self.s._is_local(obj.name, self.ctx):
            members = self.s.info.enums.get(obj.name)
            if members is not None and obj.name not in vm.m.globals:
                if node.attr in members:
                    return members[node.attr]
                vm.panic(f"enum '{obj.name}' has no variant '{node.attr}'")
        return self.s._field(self.ev(obj), node.attr)

    def call(self, node):
        vm = self.vm
        f = node.func
        if type(f) is til.Identifier:
            name = f.name
            if self.s._is_local(name, self.ctx) or name in vm.m.globals:
                self.fail(UNSUPPORTED)              # a closure in a variable: program code
            if name in ('Some', 'Ok', 'Err') and len(node.args) == 1:
                kind = {'Some': til.TYPE_SOME, 'Ok': til.TYPE_OK, 'Err': til.TYPE_ERR}[name]
                return til.VMStruct(kind, [self.ev(node.args[0])])
            if name not in PURE_NATIVES:
                self.fail(UNSUPPORTED)
            lo, hi = til.NATIVE_ARITY[name]
            args = [self.ev(a) for a in node.args]
            if len(args) < lo or (hi is not None and len(args) > hi):
                vm.panic(f"{name}(): invalid arguments")
            if name == 'range':
                if len(args) == 1:
                    args = [0, args[0]]
                if all(type(a) is int for a in args[:2]) and args[1] - args[0] > MAX_EVAL_ITEMS:
                    self.fail("the value would be too large")
            return vm.call_native(til.NATIVE_ID[name], args)
        if type(f) is til.Attribute and f.attr in PURE_METHODS:
            if type(f.obj) is til.Identifier and f.obj.name in self.s.info.enums:
                self.fail(UNSUPPORTED)
            recv = self.ev(f.obj)
            if type(recv) is til.VMStruct and self.s._user_member(recv, f.attr):
                self.fail(UNSUPPORTED)              # a method of the program (or a closure field)
            args = [self.ev(a) for a in node.args]
            if f.attr == 'repeat' and type(recv) is bytes and args and type(args[0]) is int and \
                    len(recv) * max(args[0], 0) > til.STR_MAX:
                vm.panic("string too long")
            return vm.call_method(til.METHOD_ID[f.attr], recv, args)
        self.fail(UNSUPPORTED)


# ═══════════════════════════════════════════════════════════════════════════
# the engine
# ═══════════════════════════════════════════════════════════════════════════

_HIT = re.compile(r'\s*(==|=|>=|>|<=|<|%)?\s*(\d+)\s*\Z')


def parse_hit_condition(text: str) -> Tuple[str, int]:
    """A hit condition: 'N' (the Nth time), '== N', '>= N', '> N', '<= N', '< N',
    '% N' (every Nth time) -> (operator, N). Raises DebugError."""
    m = _HIT.match(text or "")
    if not m or (m.group(1) == '%' and int(m.group(2)) == 0):
        raise DebugError(f"Invalid hit condition '{(text or '').strip()}': use N, == N, >= N, > N, "
                         f"<= N, < N or % N")
    op = m.group(1) or '=='
    return ('==' if op == '=' else op), int(m.group(2))


def hit_condition_met(hit: Tuple[str, int], count: int) -> bool:
    op, n = hit
    if op == '==':
        return count == n
    if op == '>=':
        return count >= n
    if op == '>':
        return count > n
    if op == '<=':
        return count <= n
    if op == '<':
        return count < n
    return count % n == 0


class Breakpoint:
    """A breakpoint of a DebugSession: a line with code, of one file of the
    program (`file`) or of any file (file None: what `til debug`'s `b LINE`
    sets), an optional condition (an expression, see DebugSession.evaluate)
    and hit condition (see parse_hit_condition). `tag` is free for the host."""
    __slots__ = ('id', 'line', 'file', 'file_key', 'condition', 'hit_condition', 'cond', 'hit', 'tag')

    def __init__(self, id: int, line: int, file: Optional[str], file_key: Optional[str]):
        self.id = id
        self.line = line
        self.file = file
        self.file_key = file_key
        self.condition: Optional[str] = None
        self.hit_condition: Optional[str] = None
        self.cond: Optional[_Expression] = None
        self.hit: Optional[Tuple[str, int]] = None
        self.tag: Any = None

    def __repr__(self):
        return f"Breakpoint({self.id}, line {self.line}" + (f", {self.file}" if self.file else "") + ")"


class Stop:
    """Where a run method of DebugSession stopped.

    reason      'entry', 'step', 'breakpoint', 'pause', 'goto', 'exception' (the
                program ended with a runtime error; the state is the one where it
                happened) or 'exit' (it ended normally / with exit())
    event       the timeline position: line events before this point (DebugSession.timeline)
    line, function, file, depth   where (line 0: no source line; file None: the main file)
    at_event    exactly at the line event `event` (else in the middle of a line: after
                step_out, or at the end)
    breakpoint  a breakpoint on this line (any reason), fired: the one that stopped it
    returned    step_out: (function, value it returned)
    finished, exit_code, error   the end of the program (error: the runtime error message)
    interrupted interrupt() stopped it; diverged: a replay differed from the history at
                this line event; note: a message for the user; condition_error: (breakpoint,
                message) when a condition could not be evaluated (the breakpoint stopped)"""
    __slots__ = ('reason', 'event', 'total', 'line', 'function', 'file', 'depth', 'at_event',
                 'breakpoint', 'fired', 'returned', 'finished', 'exit_code', 'error', 'interrupted',
                 'diverged', 'note', 'condition_error')

    def __init__(self, reason: str):
        self.reason = reason
        self.event = self.total = self.line = self.depth = 0
        self.function = ""
        self.file: Optional[str] = None
        self.at_event = False
        self.breakpoint: Optional[Breakpoint] = None
        self.fired: Optional[Breakpoint] = None
        self.returned: Optional[Tuple[str, Any]] = None
        self.finished = False
        self.exit_code: Optional[int] = None
        self.error: Optional[str] = None
        self.interrupted = False
        self.diverged: Optional[int] = None
        self.note: Optional[str] = None
        self.condition_error: Optional[Tuple[Breakpoint, str]] = None

    def __repr__(self):
        return f"Stop({self.reason}, {self.function} line {self.line}, event {self.event})"


class FrameInfo:
    """One frame of the call stack: depth 0 is the innermost; `index` its place in
    vm.frames; `function` the function's index; line 0 = no source line; file
    None = the main file; hidden = the prelude (map/filter/... written in TIL)."""
    __slots__ = ('depth', 'index', 'function', 'name', 'line', 'file', 'hidden')

    def __init__(self, depth, index, function, name, line, file, hidden):
        self.depth, self.index, self.function, self.name = depth, index, function, name
        self.line, self.file, self.hidden = line, file, hidden

    def __repr__(self):
        return f"FrameInfo({self.depth}: {self.name} line {self.line})"


class ProgramInfo:
    """What the compiler knows about a program besides its bytecode: the source
    file of every function (absolute paths; None = the main file), the
    constants it inlined (`let X = 5`), enum values, the keyword languages."""
    __slots__ = ('path', 'func_files', 'constants', 'enums', 'languages')

    def __init__(self, path: Optional[str] = None, func_files: Optional[List[Optional[str]]] = None,
                 constants: Optional[Dict[str, Any]] = None, enums: Optional[Dict[str, Dict[str, Any]]] = None,
                 languages: Optional[List[str]] = None):
        self.path = path
        self.func_files = func_files
        self.constants = constants or {}
        self.enums = enums or {}
        self.languages = languages


def _file_key(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


class _LineFeed:
    """input() lines from a text (str or bytes); b"" at the end."""

    def __init__(self, text):
        data = text.encode('utf-8') if isinstance(text, str) else bytes(text)
        self.lines = data.split(b"\n")
        if self.lines and self.lines[-1] == b"":
            self.lines.pop()
        self.pos = 0

    def read(self) -> bytes:
        if self.pos < len(self.lines):
            self.pos += 1
            return self.lines[self.pos - 1]
        return b""


class DebugSession:
    """The debugging engine: one program on a DebugVM, its breakpoints, and the
    run methods - start(), cont(), step_in(), step_over(), step_out(),
    step_back(), step_back_over(), reverse_continue(), goto(), run_to_end() -
    which block until the program stops and return a Stop (they raise
    DebugError when they cannot run, Killed after kill()). State queries
    (stack, locals, globals, evaluate, timeline, ...) read the state where the
    program stopped. interrupt() and kill() may be called from another thread
    while a run method runs; everything else belongs to one thread at a time.

    `stdin`: input() lines (a str/bytes text, or a function returning the next
    line as bytes; default: none); `keys`: a --keys script or key codes;
    `out(bytes)`: the program's output, each byte once; `err(bytes)`: the error
    report of the end (default: stderr); `on_screen(screen)`: screen_show()."""

    def __init__(self, module: 'til.TBCModule', source=None, path: Optional[str] = None, args=(),
                 stdin=None, keys=None, screen=None, out: Optional[Callable[[bytes], None]] = None,
                 err: Optional[Callable[[bytes], None]] = None, on_report: Optional[Callable[[], None]] = None,
                 on_screen=None, every: int = SNAPSHOT_EVERY, cwd: Optional[str] = None,
                 lang: Optional[str] = None, info: Optional[ProgramInfo] = None):
        self.module = module
        self.source: Optional[List[str]] = source.split('\n') if isinstance(source, str) else source
        if info is None:
            info = ProgramInfo(os.path.abspath(path) if path and not path.startswith('<') else path)
        self.info = info
        self.path = info.path               # the main source file (absolute; None / "<...>": none)
        self.output = bytearray()               # the program's output so far (at most MAX_OUTPUT_KEPT)
        self.output_cut = False                 # ... older output was dropped
        self._out = out
        if stdin is None:
            read_line: Callable[[], bytes] = lambda: b""
        elif callable(stdin):
            read_line = stdin
        else:
            read_line = _LineFeed(stdin).read
        self.vm = DebugVM(module, self._program_out, read_line, argv=list(args), screen=screen, keys=keys,
                          every=every, on_report=on_report, err=err, cwd=cwd, on_screen=on_screen, lang=lang)
        files = self.info.func_files
        if not files or len(files) != len(module.funcs):
            files = [None] * len(module.funcs)
        main_key = _file_key(self.path) if self.path and not self.path.startswith('<') else None
        # file of each function (None: the main file) and its comparison key
        self.func_files: List[Optional[str]] = [None if f is None or (main_key and _file_key(f) == main_key)
                                                else f for f in files]
        self._func_keys = [main_key if f is None else _file_key(f) for f in self.func_files]
        self.main_key = main_key
        self.code_lines = sorted({line for line in self.vm.dbg_lines.values() if line})
        self._file_lines: Optional[Dict[Optional[str], List[int]]] = None
        self.breakpoints: Dict[int, Tuple[Breakpoint, ...]] = {}     # line -> breakpoints on it
        self.next_bp = 1
        self.started = False
        self.end_seen = False                   # the history reaches the end of the program
        self._why: Optional[str] = None         # set by the stop tests: 'step' / 'breakpoint'
        self._fired: Optional[Breakpoint] = None
        self._cond_error: Optional[Tuple[Breakpoint, str]] = None
        self._hits: Dict[int, Tuple[int, int]] = {}  # line -> (line event, line events at it so far)
        self._user_members: Optional[set] = None
        self.last_stop: Optional[Stop] = None

    @classmethod
    def load(cls, path: str, source: Optional[str] = None, fuse: bool = True, **kwargs) -> 'DebugSession':
        """A session for a program file (.til: compiled with debug info; .tbc: read),
        or for `source` (TIL text; `path` names it and its imports start there)."""
        module, lines, info = load_program(path, source, fuse)
        return cls(module, lines, path, info=info, **kwargs)

    # ── output ──────────────────────────────────────────────────────────────
    def _program_out(self, data: bytes):
        out = self.output
        out += data
        if len(out) > MAX_OUTPUT_KEPT:
            del out[:len(out) - MAX_OUTPUT_KEPT]
            self.output_cut = True
        if self._out is not None:
            self._out(data)

    # ── control ─────────────────────────────────────────────────────────────
    def interrupt(self):
        """Stop the running program at its next line event (any thread)."""
        self.vm.request_pause()

    def kill(self):
        """End the program: a run method that runs (or the next one) raises Killed."""
        self.vm.kill()

    @property
    def finished(self) -> bool:
        return self.vm.finished is not None

    # ── files and lines ─────────────────────────────────────────────────────
    def file_key(self, file: Optional[str]) -> Optional[str]:
        """The key that names `file` in this session (None: the main file)."""
        if file is None:
            return self.main_key
        return _file_key(file)

    def files(self) -> List[str]:
        """The source files of the program (absolute paths; the main file first)."""
        out = [self.path] if self.path and not self.path.startswith('<') else []
        for f in self.func_files:
            if f is not None and f not in out:
                out.append(f)
        return out

    def has_file(self, file: str) -> bool:
        key = _file_key(file)
        return key == self.main_key or key in self._func_keys

    def _lines_by_file(self) -> Dict[Optional[str], List[int]]:
        """File key -> its lines with code (built once)."""
        if self._file_lines is None:
            per: Dict[Optional[str], set] = {}
            hidden = self.vm.hidden_funcs
            for i, f in enumerate(self.module.funcs):
                if i not in hidden:
                    per.setdefault(self._func_keys[i], set()).update(line for _off, line in f.lines if line)
            self._file_lines = {k: sorted(v) for k, v in per.items()}
        return self._file_lines

    def code_lines_of(self, file: Optional[str] = None) -> List[int]:
        """The lines with code: of the whole program (file None) or of one of its files."""
        if file is None:
            return self.code_lines
        return self._lines_by_file().get(_file_key(file), [])

    def resolve_line(self, line: int, file: Optional[str] = None) -> Optional[int]:
        """The first line with code at or after `line` (None: there is none)."""
        lines = self.code_lines_of(file)
        i = bisect.bisect_left(lines, line)
        return lines[i] if i < len(lines) else None

    def _line_files(self, line: int) -> List[Optional[str]]:
        """Keys of the files that have code at `line`."""
        out = []
        for key, lines in self._lines_by_file().items():
            i = bisect.bisect_left(lines, line)
            if i < len(lines) and lines[i] == line:
                out.append(key)
        return out

    # ── breakpoints ─────────────────────────────────────────────────────────
    def all_breakpoints(self) -> List[Breakpoint]:
        return sorted((bp for group in self.breakpoints.values() for bp in group), key=lambda b: b.id)

    def breakpoint_at(self, line: int, file_key: Optional[str] = None) -> Optional[Breakpoint]:
        """The breakpoint that a line event at `line` of the file `file_key` meets."""
        for bp in self.breakpoints.get(line, ()):
            if bp.file_key is None or bp.file_key == file_key:
                return bp
        return None

    def find_breakpoint(self, line: int, file: Optional[str] = None) -> Optional[Breakpoint]:
        """The breakpoint set exactly at `line` of `file` (None: the any-file one)."""
        key = None if file is None else _file_key(file)
        for bp in self.breakpoints.get(line, ()):
            if bp.file_key == key:
                return bp
        return None

    def _conditions(self, condition: Optional[str], hit_condition: Optional[str]):
        """(prepared condition, hit condition) of a breakpoint; DebugError if unusable."""
        cond = self.prepare(condition) if condition and condition.strip() else None
        hit = parse_hit_condition(hit_condition) if hit_condition and hit_condition.strip() else None
        return cond, hit

    @staticmethod
    def _apply(bp: Breakpoint, condition, hit_condition, cond, hit):
        bp.condition = condition if cond is not None else None
        bp.hit_condition = hit_condition if hit is not None else None
        bp.cond, bp.hit = cond, hit

    def add_breakpoint(self, line: int, file: Optional[str] = None, condition: Optional[str] = None,
                       hit_condition: Optional[str] = None) -> Breakpoint:
        """A breakpoint at `line` (give a line with code: resolve_line) of `file`
        (None: of every file, like `til debug`'s `b LINE`). The same place again
        gives the breakpoint that is there (with the new conditions). Raises
        DebugError for a condition or hit condition that cannot be used."""
        cond, hit = self._conditions(condition, hit_condition)
        bp = self.find_breakpoint(line, file)
        if bp is None:
            bp = Breakpoint(self.next_bp, line, None if file is None else os.path.abspath(file),
                            None if file is None else _file_key(file))
            self.next_bp += 1
            self._apply(bp, condition, hit_condition, cond, hit)
            groups = dict(self.breakpoints)        # replaced, never changed: a run may be reading it
            groups[line] = groups.get(line, ()) + (bp,)
            self.breakpoints = groups
        else:
            self._apply(bp, condition, hit_condition, cond, hit)
        return bp

    def remove_breakpoint(self, bp: Breakpoint):
        groups = dict(self.breakpoints)
        rest = tuple(b for b in groups.get(bp.line, ()) if b is not bp)
        if rest:
            groups[bp.line] = rest
        else:
            groups.pop(bp.line, None)
        self.breakpoints = groups

    def clear_breakpoints(self):
        self.breakpoints = {}

    def set_file_breakpoints(self, file: str, specs) -> List[Any]:
        """Replace the breakpoints of `file` (a debug adapter's setBreakpoints):
        specs = [(line, condition, hit condition)]. Returns, per spec, its
        Breakpoint (at the first line with code at or after `line`) or the
        DebugError that says why there is none."""
        key = _file_key(file)
        old = {bp.line: bp for group in self.breakpoints.values() for bp in group if bp.file_key == key}
        new: Dict[int, Breakpoint] = {}
        results: List[Any] = []
        for line, condition, hit_condition in specs:
            actual = self.resolve_line(line, file)
            if actual is None:
                results.append(DebugError(f"No code at or after line {line}"))
                continue
            if actual in new:
                results.append(new[actual])
                continue
            try:
                cond, hit = self._conditions(condition, hit_condition)
            except DebugError as e:
                results.append(e)
                continue
            bp = old.get(actual)
            if bp is None:
                bp = Breakpoint(self.next_bp, actual, os.path.abspath(file), key)
                self.next_bp += 1
            self._apply(bp, condition, hit_condition, cond, hit)
            new[actual] = bp
            results.append(bp)
        groups: Dict[int, Tuple[Breakpoint, ...]] = {}
        for line, group in self.breakpoints.items():
            rest = tuple(b for b in group if b.file_key != key)
            if rest:
                groups[line] = rest
        for line, bp in new.items():
            groups[line] = groups.get(line, ()) + (bp,)
        self.breakpoints = groups
        return results

    def _count_hit(self, line: int, ev: int) -> int:
        """Line events at `line` from the start of the program up to event `ev`."""
        c = self._hits.get(line)
        hist = self.vm.ev_lines
        if c is not None and c[0] == ev:
            return c[1]
        if c is not None and c[0] < ev:
            n = c[1] + hist[c[0] + 1:ev + 1].count(line)
        else:
            n = hist[:ev + 1].count(line)
        self._hits[line] = (ev, n)
        return n

    def _meets(self, bp: Breakpoint, ev: int, line: int, top) -> bool:
        if bp.file_key is not None and bp.file_key != self._func_keys[top[0]]:
            return False
        if bp.hit is not None and not hit_condition_met(bp.hit, self._count_hit(line, ev)):
            return False
        if bp.cond is not None:
            try:
                return bool(self.vm.truthy(self._run_prepared(bp.cond, (top, False))))
            except DebugError as e:
                self._cond_error = (bp, str(e))     # stop: the user sees why
            except Exception as e:                  # noqa: BLE001 - a condition never breaks the run
                self._cond_error = (bp, f"Cannot evaluate '{bp.condition}': {e}")
        return True

    def _fires(self, ev: int, line: int, top) -> bool:
        """At a line event: does a breakpoint stop the program here?"""
        for bp in self.breakpoints.get(line, ()):
            if self._meets(bp, ev, line, top):
                self._why = 'breakpoint'
                self._fired = bp
                return True
        return False

    # ── running ─────────────────────────────────────────────────────────────
    def _begin(self):
        self._why = self._fired = self._cond_error = None
        self._hits = {}

    def _run(self, stop, why: str) -> Stop:
        if self.vm.finished is not None:
            raise DebugError("The program has finished (back and rc go back in time, q quits)")
        self.started = True
        self._begin()
        return self._stopped(self.vm.resume(stop), why)

    def start(self) -> Stop:
        """Run to the first line event (reason 'entry'; a program without line
        events runs to its end)."""
        return self._run(lambda ev, line, depth, top: True, 'entry')

    def cont(self) -> Stop:
        """Continue to the next breakpoint (or the end). From the very start (no
        start() before), a breakpoint on the first line event stops too."""
        bps = self
        fires = self._fires

        def stop(ev, line, depth, top):
            return line in bps.breakpoints and fires(ev, line, top)
        return self._run(stop, 'breakpoint')

    def run_to_end(self) -> Stop:
        """Run to the end without stopping (no breakpoints; interrupt() still stops)."""
        return self._run(lambda ev, line, depth, top: False, 'pause')

    def step_in(self) -> Stop:
        """To the next line event, anywhere (into calls)."""
        def stop(ev, line, depth, top):
            self._why = 'step'
            return True
        return self._run(stop, 'step')

    def step_over(self) -> Stop:
        """To the next line event of this frame or of a caller (over calls; also
        stops at breakpoints on the way)."""
        frames = list(self.vm.frames)
        d0 = len(frames)
        bps = self
        fires = self._fires

        def stop(ev, line, depth, top):
            if depth <= d0 and frames[depth - 1] is top:
                self._why = 'step'
                return True
            return line in bps.breakpoints and fires(ev, line, top)
        return self._run(stop, 'step')

    def step_out(self) -> Stop:
        """Until the current frame returns: stops in the caller, in the middle of the
        line of the call (Stop.returned: the value), or at a breakpoint on the way."""
        vm = self.vm
        if vm.finished is not None:
            raise DebugError("The program has finished (back and rc go back in time, q quits)")
        frames = list(vm.frames)
        d0 = len(frames)
        bps = self
        fires = self._fires
        vm.return_stop(frames[-2] if d0 >= 2 else None)

        def stop(ev, line, depth, top):
            if depth < d0 and frames[depth - 1] is top:
                self._why = 'step'
                return True
            return line in bps.breakpoints and fires(ev, line, top)
        try:
            st = self._run(stop, 'step')
        finally:
            vm.return_stop(None)
        if not st.finished and not vm.at_event and vm.stack and frames:
            st.returned = (vm.m.funcs[frames[-1][0]].name, vm.stack[-1])
        return st

    def _travel(self, target: int, why: str) -> Stop:
        self.started = True
        self._begin()
        return self._stopped(self.vm.travel(target), why)

    def can_step_back(self) -> bool:
        return self.vm.position() > 0

    def step_back(self) -> Stop:
        """Back one line event (anywhere: `til debug`'s `back`)."""
        target = self.vm.position() - 1
        if target < 0:
            raise DebugError("Already at the start of the program")
        return self._travel(target, 'step')

    def back_over_target(self) -> Optional[int]:
        """The line event step_back_over() goes to (None: none before this point)."""
        vm = self.vm
        pos = vm.position()
        if pos <= 0:
            return None
        if vm.finished is not None and vm.finished[1] == 'exit':
            return pos - 1                  # from the end: the last line that ran
        meta = vm.ev_meta
        d0 = len(vm.frames)
        limit = d0
        if vm.paused and vm.at_event and 0 <= vm.ev < len(meta) and meta[vm.ev] == d0 + d0 + 1:
            limit = d0 - 1                  # the first line of this frame: back to the line of the call
        for e in range(min(pos, len(meta)) - 1, -1, -1):
            if meta[e] >> 1 <= limit:
                return e
        return None

    def step_back_over(self) -> Stop:
        """Back to the previous line event of this frame or of a caller - the
        reverse of step_over() (and of step_in() at a function's first line): the
        calls made in between are stepped over. From the end of the program: its
        last line event."""
        target = self.back_over_target()
        if target is None:
            raise DebugError("Already at the start of the program")
        return self._travel(target, 'step')

    def _needs_state(self, bp: Breakpoint) -> bool:
        """Whether checking `bp` at an earlier line event needs the program state there."""
        if bp.cond is not None:
            return True
        return bp.file_key is not None and len(self._line_files(bp.line)) > 1

    def _scan_segment(self, s: int, end: int) -> Optional[int]:
        """Replay line events s..end; the last one where a breakpoint fires."""
        vm = self.vm
        hist = vm.ev_lines
        found = None
        vm.restore(s)
        if hist[s] in self.breakpoints and self._fires(s, hist[s], vm.frames[-1]):
            found = s
        if end > s:
            bps = self
            fires = self._fires

            def stop(ev, line, depth, top):
                nonlocal found
                if line in bps.breakpoints and fires(ev, line, top):
                    found = ev
                return ev >= end
            vm.resume(stop)
        return found

    def _last_hit_before(self, cur: int) -> Optional[int]:
        """The last line event before `cur` where a breakpoint would have stopped."""
        vm = self.vm
        hist = vm.ev_lines
        groups = self.breakpoints
        if not groups or cur <= 0:
            return None
        self._begin()
        if not any(self._needs_state(bp) for group in groups.values() for bp in group):
            counts: Dict[int, int] = {}
            for i in range(cur - 1, -1, -1):
                line = hist[i]
                group = groups.get(line)
                if group is None:
                    continue
                if line not in counts:
                    counts[line] = hist[:i + 1].count(line)
                n = counts[line]
                counts[line] = n - 1
                files = None
                for bp in group:
                    if bp.file_key is not None:
                        if files is None:
                            files = self._line_files(line)
                        if bp.file_key not in files:
                            continue
                    if bp.hit is not None and not hit_condition_met(bp.hit, n):
                        continue
                    return i
            return None
        end = cur - 1
        while end >= 0:
            s = max(k for k in vm.snaps if k <= end)
            if any(hist[i] in groups for i in range(s, end + 1)):
                found = self._scan_segment(s, end)
                if found is not None:
                    return found
            end = s - 1
        return None

    def reverse_continue(self) -> Stop:
        """Back to the last breakpoint hit before this point (conditions and hit
        counts as forwards); none: back to the start of the program (Stop.note)."""
        vm = self.vm
        cur = vm.position()
        found = self._last_hit_before(cur)
        if found is not None:
            st = self._travel(found, 'breakpoint')
            if st.fired is None and st.at_event:
                st.fired = st.breakpoint
            return st
        if cur <= 0:
            raise DebugError("Already at the start of the program")
        st = self._travel(0, 'step')
        st.note = "No earlier breakpoint hit: back to the start of the program"
        return st

    def goto(self, index: int) -> Stop:
        """To line event `index` of the history (0 .. timeline total - 1), or to
        the end of the program (index = total, once the history reaches it)."""
        vm = self.vm
        total = len(vm.ev_lines)
        if index == total and self.end_seen and vm.snaps:
            self.started = True
            self._begin()
            vm.restore(max(vm.snaps))
            return self._stopped(vm.resume(lambda ev, line, depth, top: False), 'goto')
        if not 0 <= index < total:
            raise DebugError(f"No line event {index} in the history (0 to {total - 1})")
        return self._travel(index, 'goto')

    def _stopped(self, result: Optional[int], why: str) -> Stop:
        vm = self.vm
        if vm.killed:
            raise Killed()
        if result is not None:
            self.end_seen = True
            code, kind = vm.finished
            st = Stop('exception' if kind == 'error' else 'exit')
            st.finished = True
            st.exit_code = code
            if kind == 'error' and vm.end_error is not None:
                st.error = vm.end_error[0].decode('utf-8', 'replace')
        else:
            st = Stop(self._why or ('pause' if vm.interrupted else why))
            st.interrupted = vm.interrupted
            vm.interrupted = False
            st.at_event = vm.at_event
            if st.reason == 'breakpoint':
                st.fired = self._fired
        st.diverged, vm.diverged = vm.diverged, None
        st.condition_error = self._cond_error
        st.event = vm.position()
        st.total = len(vm.ev_lines)
        if vm.frames:
            top = vm.frames[-1]
            st.depth = 0
            st.line = self.frame_line(len(vm.frames) - 1)
            st.function = vm.m.funcs[top[0]].name
            st.file = self.func_files[top[0]]
            if st.at_event:
                st.breakpoint = self.breakpoint_at(st.line, self._func_keys[top[0]])
        self.last_stop = st
        return st

    # ── where the program is ────────────────────────────────────────────────
    def timeline(self) -> Tuple[int, int]:
        """(index, total): line events before the current point, line events recorded."""
        return self.vm.position(), len(self.vm.ev_lines)

    def frame_line(self, index: int) -> int:
        """Line where vm.frames[index] is: the current line event for the top frame,
        else the line of the call (or of the instruction that ended the program)."""
        vm = self.vm
        frame = vm.frames[index]
        if index == len(vm.frames) - 1 and vm.paused and vm.at_event and 0 <= vm.ev < len(vm.ev_lines):
            return vm.ev_lines[vm.ev]
        return vm.line_at(frame, frame[1] - 1)

    def current_line(self) -> int:
        return self.frame_line(len(self.vm.frames) - 1) if self.vm.frames else 0

    def stack(self) -> List[FrameInfo]:
        """The call stack, innermost frame first."""
        vm = self.vm
        n = len(vm.frames)
        out = []
        for depth in range(n):
            i = n - 1 - depth
            fidx = vm.frames[i][0]
            out.append(FrameInfo(depth, i, fidx, vm.m.funcs[fidx].name, self.frame_line(i),
                                 self.func_files[fidx], fidx in vm.hidden_funcs))
        return out

    def _ctx(self, depth: int = 0):
        """(frame, at_call) of frame `depth` (0: innermost): at_call = the frame's pc
        is just after an instruction (a call, `out` came back from, the end)."""
        vm = self.vm
        n = len(vm.frames)
        if not n:
            return None
        if not 0 <= depth < n:
            raise DebugError(f"No frame {depth}")
        return vm.frames[n - 1 - depth], depth > 0 or not (vm.paused and vm.at_event)

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

    def slot_value(self, frame, slot: int):
        v = self.vm.stack[frame[2] + slot]
        return v.v if type(v) is til.VMCell else v

    def locals(self, depth: int = 0) -> List[Tuple[str, Any]]:
        """The variables of frame `depth` (0: innermost): live locals, then upvalues."""
        ctx = self._ctx(depth)
        return self.frame_vars(*ctx) if ctx else []

    def params(self, depth: int = 0) -> List[Tuple[str, Any]]:
        """(name, value) of the parameters of frame `depth` (what `bt` shows)."""
        vm = self.vm
        frame = vm.frames[len(vm.frames) - 1 - depth]
        f = vm.m.funcs[frame[0]]
        names: Dict[int, str] = {}
        for slot, name, _start, _end in f.locals:
            if slot < f.n_params:
                names.setdefault(slot, name)
        return [(names[s], self.slot_value(frame, s)) for s in sorted(names)]

    def globals(self) -> List[Tuple[str, Any]]:
        """(name, value) of the program's globals (constants the compiler inlined are
        not variables: see ProgramInfo.constants)."""
        vm = self.vm
        return [(name, vm.globals[i]) for i, name in enumerate(vm.m.globals) if i < len(vm.globals)]

    def _is_local(self, name: str, ctx) -> bool:
        return ctx is not None and any(n == name for n, _v in self.frame_vars(*ctx))

    def _lookup(self, name: str, ctx):
        vm = self.vm
        if ctx is not None:
            frame, at_call = ctx
            for n, v in self.frame_vars(frame, at_call):
                if n == name:
                    return v
            if name.startswith('$') and name[1:].isdigit():     # a slot, without debug info
                slot = int(name[1:])
                if slot < vm.m.funcs[frame[0]].n_locals and frame[2] + slot < len(vm.stack):
                    return self.slot_value(frame, slot)
        if name in vm.m.globals:
            return vm.globals[vm.m.globals.index(name)]
        if name in self.info.constants:
            return self.info.constants[name]
        raise DebugError(f"No variable '{name}' here")

    def lookup(self, name: str, depth: int = 0):
        """A variable: local or upvalue of frame `depth`, `$N` (slot N, for bytecode
        without names), a global, or a constant the compiler inlined."""
        return self._lookup(name, self._ctx(depth))

    def _user_member(self, v, name: str) -> bool:
        """`name` is a method of the program for this struct (or a closure field)."""
        vm = self.vm
        if self._user_members is None:
            members = set()
            for t, nc, _f in vm.m.methods:
                members.add((t, vm.consts[nc]))
            self._user_members = members
        key = name.encode('utf-8')
        if (v.t, key) in self._user_members:
            return True
        return v.t > til.TYPE_ERR and key in vm.field_names[v.t]

    def _field(self, v, field: str):
        vm = self.vm
        key = field.encode('utf-8')
        if type(v) is til.VMStruct and key in vm.field_names[v.t]:
            return v.f[vm.field_names[v.t].index(key)]
        if type(v) is dict and key in v:
            return v[key]
        raise til.TILPanic(f"no field '{field}' on {vm.type_name(v).decode()}")

    def _eval_path(self, expr: str, m, ctx):
        vm = self.vm
        steps = _STEP.findall(m.group(2))
        name = m.group(1)
        try:
            v = self._lookup(name, ctx)
        except DebugError:
            members = self.info.enums.get(name)
            if members is None or not steps or not steps[0][0]:
                raise
            if steps[0][0] not in members:                  # Enum.Variant
                raise DebugError(f"Cannot evaluate '{expr.strip()}': enum '{name}' has no variant "
                                 f"'{steps[0][0]}'")
            v = members[steps[0][0]]
            steps = steps[1:]
        for field, index in steps:
            try:
                if field:
                    v = self._field(v, field)
                else:
                    key = index.strip()
                    if re.fullmatch(r'-?\d+', key):
                        k: Any = int(key)
                    elif len(key) >= 2 and key[0] == key[-1] and key[0] in '"\'':
                        k = key[1:-1].encode('utf-8')
                    else:
                        k = self._evaluate(key, ctx)
                    v = vm.index_get(v, k)
            except til.TILPanic as p:
                raise DebugError(f"Cannot evaluate '{expr.strip()}': {p.msg.decode('utf-8', 'replace')}")
        return v

    def prepare(self, expr: str) -> _Expression:
        """Parse an expression once (for a breakpoint condition); DebugError if it
        cannot be evaluated."""
        text = expr.strip()
        m = _PATH.match(expr)
        if m and (m.group(1).startswith('$') or self._is_name(m.group(1))):
            return _Expression(text, path=m)
        try:
            tokens = til.Lexer(text, "<expression>", languages=self.info.languages).tokenize()
            parser = til.Parser(tokens, "<expression>", languages=self.info.languages)
            node = parser.parse_expression()
            rest = [t for t in tokens[parser.pos:]
                    if t.type not in (til.TokenType.NEWLINE, til.TokenType.EOF, til.TokenType.DEDENT)]
        except SyntaxError as e:
            first = _ANSI.sub('', str(e)).strip().split('\n')[0]
            first = re.sub(r'^error(\[[^\]]*\])?:\s*', '', first)
            raise DebugError(f"Cannot evaluate '{text}': {first}")
        except RecursionError:
            raise DebugError(f"Cannot evaluate '{text}': {UNSUPPORTED}")
        if rest:
            raise DebugError(f"Cannot evaluate '{text}': {UNSUPPORTED}")
        return _Expression(text, node=node)

    def _is_name(self, word: str) -> bool:
        """`word` is a name of the program's languages (not None, true, a keyword)."""
        if word == 'None':
            return False
        try:
            tokens = til.Lexer(word, "<expression>", languages=self.info.languages).tokenize()
        except SyntaxError:
            return False
        return bool(tokens) and tokens[0].type == til.TokenType.IDENT and tokens[0].value == word

    def _run_prepared(self, e: _Expression, ctx):
        if e.path is not None:
            return self._eval_path(e.text, e.path, ctx)
        return _Evaluator(self, ctx, e.text).run(e.node)

    def _evaluate(self, expr: str, ctx):
        return self._run_prepared(self.prepare(expr), ctx)

    def evaluate(self, expr: str, depth: int = 0):
        """The value of an expression in frame `depth` (0: innermost). Supported:
        variables (see lookup), `a.b` (struct fields, map keys, Some/Ok `.value`,
        Err `.error`, `Enum.Variant`), `a[i]`, literals (numbers, strings, true,
        false, None, arrays, maps, f-strings), the operators of TIL (arithmetic,
        comparisons, and/or/not, bitwise, `x if c else y`, `as`), Some/Ok/Err(x)
        and calls of builtins and builtin methods without side effects
        (PURE_NATIVES, PURE_METHODS). Not: calls of the program's functions or
        methods, assignments, lambdas. Raises DebugError (the message says why)."""
        return self._evaluate(expr, self._ctx(depth))

    # ── values ──────────────────────────────────────────────────────────────
    def preview(self, v, limit: int = PREVIEW) -> str:
        return preview(self.vm, v, limit)

    def children(self, v, start: int = 0, count: Optional[int] = None):
        return value_children(self.vm, v, start, count)

    def describe(self, v, limit: int = PREVIEW) -> Dict[str, Any]:
        """{'type', 'kind', 'value' (preview), 'indexed', 'named'} of a value."""
        indexed, named = child_counts(self.vm, v)
        return {'type': value_type(self.vm, v), 'kind': value_kind(self.vm, v),
                'value': preview(self.vm, v, limit), 'indexed': indexed, 'named': named}


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


class Session:
    """The command line of `til debug`, on a DebugSession (`engine`)."""

    def __init__(self, engine: DebugSession, write: Callable[[bytes], None], flush: Callable[[], None]):
        self.engine = engine
        self.vm = engine.vm
        self.source = engine.source
        self._write = write
        self.partial = False        # the program's output so far ends inside a line
        self.flush = flush

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

    def show_stop(self, stop: Stop):
        if stop.note:
            self.say(stop.note)
        if stop.diverged is not None:
            self.say(f"Warning: the replay differs from the recorded run at line event {stop.diverged}")
        if stop.finished:
            if stop.reason == 'exception':
                self.say(f"Program ended with an error (exit code {stop.exit_code})")
            else:
                self.say(f"Program exited with code {stop.exit_code}")
            return
        if stop.interrupted:
            self.say("Interrupted")
        line, func = stop.line, stop.function
        if stop.breakpoint is not None:
            self.say(f"Breakpoint {stop.breakpoint.id}, {func} at line {line}")
        elif line:
            self.say(f"{func} at line {line}")
        else:
            self.say(f"{func} (no source line)")
        if self.source and 0 < line <= len(self.source):
            self.raw(f"{line:5d} | {self.source[line - 1].rstrip()}")
        if stop.returned is not None:
            self.say(f"{stop.returned[0]} returned {self.short(stop.returned[1])}")

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
        engine = self.engine
        if not arg:
            bps = engine.all_breakpoints()
            if not bps:
                self.say("No breakpoints")
            for bp in bps:
                self.say(f"Breakpoint {bp.id} at line {bp.line}")
            return
        asked = self.line_arg(arg, 'b')
        line = engine.resolve_line(asked)
        if line is None:
            raise DebugError(f"No code at or after line {asked}")
        n = engine.add_breakpoint(line).id
        if line == asked:
            self.say(f"Breakpoint {n} at line {line}")
        else:
            self.say(f"Breakpoint {n} at line {line} (line {asked} has no code)")

    def cmd_d(self, arg: str):
        engine = self.engine
        if not arg:
            engine.clear_breakpoints()
            self.say("Deleted all breakpoints")
            return
        line = self.line_arg(arg, 'd')
        bp = engine.find_breakpoint(line)
        if bp is None:
            raise DebugError(f"No breakpoint at line {line}")
        engine.remove_breakpoint(bp)
        self.say(f"Deleted breakpoint {bp.id} at line {line}")

    def cmd_c(self, arg: str):
        self.show_stop(self.engine.cont())

    def cmd_s(self, arg: str):
        self.show_stop(self.engine.step_in())

    def cmd_n(self, arg: str):
        self.show_stop(self.engine.step_over())

    def cmd_out(self, arg: str):
        self.show_stop(self.engine.step_out())

    def cmd_back(self, arg: str):
        self.show_stop(self.engine.step_back())

    def cmd_rc(self, arg: str):
        self.show_stop(self.engine.reverse_continue())

    # ── variables ───────────────────────────────────────────────────────────
    def show(self, v, quote: bool = False) -> str:
        if v is til._UNDEF:
            return "<no value>"
        text = self.vm.to_str(v, 0, quote).decode('utf-8', 'replace')
        if len(text) > MAX_SHOWN:
            text = text[:MAX_SHOWN] + _(f"... ({len(text) - MAX_SHOWN} more characters)")
        return text

    def cmd_p(self, arg: str):
        if not arg:
            raise DebugError("Usage: p NAME")
        self.raw(f"{arg} = {self.show(self.engine.evaluate(arg))}")

    def cmd_locals(self, arg: str):
        pairs = self.engine.locals(0) if self.vm.frames else []
        if not pairs:
            self.say("No local variables")
        for name, v in pairs:
            self.raw(f"{name} = {self.show(v)}")

    def cmd_bt(self, arg: str):
        engine = self.engine
        for fr in engine.stack():
            shown = ", ".join(f"{name}={self.short(v)}" for name, v in engine.params(fr.depth))
            self.say(f"  at {fr.name}({shown})" + (f" (line {fr.line})" if fr.line else ""))

    def short(self, v) -> str:
        text = self.show(v, quote=True).replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r')
        return text if len(text) <= 60 else text[:57] + "..."


# ═══════════════════════════════════════════════════════════════════════════
# loading programs
# ═══════════════════════════════════════════════════════════════════════════

def _imported_definitions(path: str, modules: List[str]) -> Dict[str, str]:
    """Functions and methods (`Type.method`) defined in the files a program
    imports -> that file (absolute path), found as the compiler finds them."""
    resolver = til.ModuleResolver()
    names: Dict[str, str] = {}
    seen = {til.module_file_key(path)}
    todo = [(m, path) for m in modules]
    while todo:
        mod, frm = todo.pop(0)
        try:
            found = resolver.resolve(mod, frm)
        except (OSError, ValueError):
            found = None
        if not found:
            continue
        key = til.module_file_key(found)
        if key in seen:
            continue
        seen.add(key)
        try:
            with open(found, encoding='utf-8') as f:
                tree = til.parse_source(f.read(), found)
        except (OSError, SyntaxError, UnicodeDecodeError, til.CompileError):
            continue
        full = os.path.abspath(found)
        for s in tree.statements:
            if isinstance(s, til.FuncDef) and s.name != 'main':
                names.setdefault(s.name, full)
            elif isinstance(s, til.ImplBlock):
                for meth in s.methods:
                    names.setdefault(f"{s.type_name}.{meth.name}", full)
            elif isinstance(s, til.Import):
                todo.append((s.module, found))
    return names


def compile_program(text: str, path: str, fuse: bool = True) -> Tuple['til.TBCModule', ProgramInfo]:
    """til.compile_to_bytecode(text, path, True, fuse, debug_info=True) - the
    same bytes (tests/test_dap.py checks it) - plus what the debugger learns
    from the compiler: the file of each function (code a program imports has
    the line numbers of its own file), the constants it inlined, enum values."""
    tree = til.parse_source(text, path)
    imports = [s.module for s in tree.statements if isinstance(s, til.Import)]
    program = til.resolve_imports(tree, path)
    errors = til.SemanticChecker(text, path).check(program)
    if errors:
        raise til.CompileError("\n\n".join(errors))
    compiler = til.BytecodeCompiler(text, path, fuse=fuse, debug_info=True)
    roots: Dict[int, int] = {}          # function -> the function whose compilation made it (lambdas)
    compile_function = compiler.compile_function

    def tracked(fidx, fdef, owner):
        before = len(compiler.m.funcs)
        compile_function(fidx, fdef, owner)
        for k in range(before, len(compiler.m.funcs)):
            roots.setdefault(k, fidx)
    compiler.compile_function = tracked
    module = compiler.compile_program(program)
    errors = til.type_errors(program, text, path)
    if errors:
        raise til.CompileError("\n\n".join(errors))
    module.link()
    info = ProgramInfo(os.path.abspath(path) if not path.startswith('<') else path,
                       languages=til.detect_languages(text))
    if imports:
        names = _imported_definitions(path, imports)
        info.func_files = [names.get(module.funcs[roots.get(i, i)].name) for i in range(len(module.funcs))]
    for name, value in getattr(compiler, 'inline_consts', {}).items():
        info.constants[name] = value.encode('utf-8') if isinstance(value, str) else value
    for name, values in getattr(compiler, 'enum_values', {}).items():
        info.enums[name] = dict(values)
    for name, variants in getattr(compiler, 'adt_variants', {}).items():
        info.enums.setdefault(name, {}).update(
            {v: til.VMStruct(tidx, []) for v, (tidx, arity) in variants.items() if arity == 0})
    return module, info


def load_program(path: str, source: Optional[str] = None, fuse: bool = True,
                 pragma_language: bool = False) -> Tuple['til.TBCModule', Optional[List[str]], ProgramInfo]:
    """The program as the debugger runs it: (module, source lines, ProgramInfo).
    A .til (or `source`, named `path`) is compiled with debug info, a .tbc is
    read (its source lines from the .til next to it); always through the bytes
    of a .tbc file, so the debugger sees exactly what the format holds.
    pragma_language: the program's #[lang: ...] chooses the message language
    (`til debug`'s main() only)."""
    if source is None and path.endswith('.tbc'):
        with open(path, 'rb') as f:
            data = f.read()
        lines = None
        base = os.path.splitext(path)[0] + '.til'
        if os.path.exists(base):
            with open(base, encoding='utf-8', errors='replace') as f:
                lines = f.read().split('\n')
        return til.TBCModule.from_bytes(data), lines, ProgramInfo(os.path.abspath(base if lines else path))
    if source is None:
        with open(path, encoding='utf-8') as f:
            source = f.read()
    if pragma_language:
        til.use_program_message_language(source)
    module, info = compile_program(source, path, fuse)
    return til.TBCModule.from_bytes(module.serialize()), source.split('\n'), info


def load_module(path: str, fuse: bool = True,
                pragma_language: bool = False) -> Tuple['til.TBCModule', Optional[List[str]]]:
    """(module, source lines) of load_program()."""
    module, lines, _info = load_program(path, None, fuse, pragma_language)
    return module, lines


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

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
        module, source, info = load_program(path, None, opts['fuse'], pragma_language=True)
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
    engine = DebugSession(module, source, path, args=opts['args'], stdin=read_line, screen=screen, keys=keys,
                          out=lambda data: sessions[0].program_output(data), every=opts['every'],
                          on_report=flush, info=info)
    vm = engine.vm
    session = Session(engine, write, flush)
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
        session.show_stop(engine.start())
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
