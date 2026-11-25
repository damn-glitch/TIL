#!/usr/bin/env python3
"""
til dap - the Debug Adapter Protocol server of TIL (docs/TIL_DEBUGGER.md,
"Debug adapter (DAP)"; docs/dev/TIL_STUDIO_DESIGN.md section 4).

    til dap                     # stdio, Content-Length framing (VS Code, Neovim, Helix, ...)
    til dap --log FILE          # also write every message to FILE

The debugger is TIL's time-travel debugger (src/tildebug.py, DebugSession):
breakpoints (conditions, hit counts), step in / over / out, step back and
reverse continue (supportsStepBack), the call stack, variables (arrays, maps,
structs, closures, Option / Result expandable), evaluate (hover, watch, repl)
and stops on runtime errors (exception filter `runtime-error`). Custom
requests: `til/timeline` -> {index, total}, `til/goto` {index}, `til/stdin`
{text} (with the event `til/inputRequested`); event `til/screen` {w, h,
rgb_b64} on screen_show().

In-process (TIL Studio, src/tilstudio.py):

    session = DapSession(emit, root)       # emit(event dict): from any thread, one at a time
    responses = session.handle(message)    # one DAP request -> its response(s)
    session.close()

The program runs on a worker thread; handle() does not wait for it (run
requests answer at once, the `stopped` / `terminated` events follow through
emit; only disconnect and close() wait up to 5 seconds for it to end).
Messages that are not requests (responses to reverse requests, events) are
ignored.
"""

import base64
import codecs
import collections
import contextlib
import json
import os
import queue
import shlex
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import til
import tildebug

USAGE = """\
usage: til dap [--log FILE]

The TIL debug adapter (Debug Adapter Protocol over stdin/stdout) for editors:
VS Code, Neovim (nvim-dap), Helix, Emacs (dap-mode), Sublime, ... Launch
arguments: program, args, cwd, stdin, keys, stopOnEntry, noDebug, lang (see
docs/TIL_DEBUGGER.md)."""

THREAD_ID = 1
FRAME_INTERVAL = 1 / 30         # til/screen: at most 30 frames a second while the program runs
OUTPUT_DELAY = 0.03             # program output is sent in chunks: at most this late (seconds)
OUTPUT_CHUNK = 32768            # ... and at most this many characters at once
PENDING_IDS = 100000            # ids of breakpoints set before launch (breakpoint events follow)
EXCEPTION_FILTER = 'runtime-error'


class DapError(Exception):
    """A request that fails: its response has success false and this (localized) message."""


def capabilities(lang: Optional[str] = None) -> Dict[str, Any]:
    """The adapter's Capabilities (the body of the initialize response)."""
    return {
        'supportsConfigurationDoneRequest': True,
        'supportsStepBack': True,
        'supportsEvaluateForHovers': True,
        'supportsConditionalBreakpoints': True,
        'supportsHitConditionalBreakpoints': True,
        'supportsTerminateRequest': True,
        'supportTerminateDebuggee': True,
        'supportsLoadedSourcesRequest': False,
        'supportsExceptionInfoRequest': True,
        'supportsBreakpointLocationsRequest': True,
        'supportsDelayedStackTraceLoading': True,
        'supportsClipboardContext': True,
        'supportsSetVariable': False,
        'supportsRestartRequest': False,
        'supportsFunctionBreakpoints': False,
        'supportsGotoTargetsRequest': False,
        'supportsStepInTargetsRequest': False,
        'supportsCompletionsRequest': False,
        'supportsModulesRequest': False,
        'exceptionBreakpointFilters': [{
            'filter': EXCEPTION_FILTER,
            'label': til.localize("Runtime errors", lang),
            'description': til.localize("Stop where the program ends with a runtime error", lang),
            'default': True,
        }],
    }


class DapSession:
    """One debug session (one program) of the Debug Adapter Protocol.

    handle(request) returns the response(s) to send now; events (initialized,
    stopped, continued, output, exited, terminated, breakpoint, til/...) go
    through emit(event), from the thread that has them (the worker thread that
    runs the program, a timer for output) - never two at a time, in order. An
    event caused by a request is emitted when handle() returns (not before,
    unless another thread's handle() ends first); wrap handle() and sending its
    responses in `with session.hold_events():` to deliver such events after the
    responses (the stdio server does). emit must not block for long and must
    not call back into the session. `seq` numbers of responses and events come
    from one counter. Thread-safe: handle() may be called from any thread (one
    request at a time is processed)."""

    _HANDLERS = {
        'initialize': '_initialize', 'launch': '_launch', 'attach': '_attach',
        'setBreakpoints': '_set_breakpoints', 'setExceptionBreakpoints': '_set_exception_breakpoints',
        'setFunctionBreakpoints': '_set_function_breakpoints',
        'breakpointLocations': '_breakpoint_locations', 'configurationDone': '_configuration_done',
        'threads': '_threads', 'stackTrace': '_stack_trace', 'scopes': '_scopes',
        'variables': '_variables', 'evaluate': '_evaluate', 'exceptionInfo': '_exception_info',
        'continue': '_continue', 'next': '_next', 'stepIn': '_step_in', 'stepOut': '_step_out',
        'stepBack': '_step_back', 'reverseContinue': '_reverse_continue', 'pause': '_pause',
        'terminate': '_terminate', 'disconnect': '_disconnect', 'cancel': '_cancel',
        'loadedSources': '_loaded_sources', 'source': '_source_request',
        'til/timeline': '_til_timeline', 'til/goto': '_til_goto', 'til/stdin': '_til_stdin',
    }

    def __init__(self, emit: Callable[[dict], None], root: Optional[str] = None):
        self._emit = emit
        self.root = os.path.abspath(root) if root else None
        self._lock = threading.RLock()          # the session's state; one request at a time
        self._emit_lock = threading.RLock()     # event order, seq numbers
        self._out_lock = threading.Lock()       # the program output buffer
        self._holds = 0                         # threads inside hold_events()
        self._held: List[dict] = []
        self._local = threading.local()
        self._seq = 0
        self.lang: Optional[str] = None         # message language (launch `lang`, initialize `locale`)
        self.lines_start_at_1 = True
        self.columns_start_at_1 = True
        self.state = 'new'      # new -> launched -> running <-> stopped -> terminated (-> closed)
        self.engine: Optional[tildebug.DebugSession] = None
        self.program: Optional[str] = None
        self.filters = {EXCEPTION_FILTER}
        self.stop_on_entry = self.no_debug = self.stop_at_end = False
        self._stop: Optional[tildebug.Stop] = None
        self._commands: 'queue.Queue' = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._killed = False
        self._closing = False
        self._input: 'collections.deque' = collections.deque()
        self._input_eof = False
        self._input_cond = threading.Condition()
        self.waiting_for_input = False
        self._input_hint = False
        self._refs: Dict[int, tuple] = {}       # variablesReference / frame id -> (key, value, evaluate name)
        self._ref_of: Dict[Any, int] = {}
        self._next_ref = 1
        self._pending: Dict[str, tuple] = {}    # breakpoints set before launch: file key -> (path, specs, ids)
        self._next_pending = PENDING_IDS
        self._out_buf: List[str] = []
        self._out_size = 0
        self._out_timer: Optional[threading.Timer] = None
        self._decode_out = codecs.getincrementaldecoder('utf-8')('replace')
        self._decode_err = codecs.getincrementaldecoder('utf-8')('replace')
        self._frame: Optional[tuple] = None     # the last til/screen frame sent (w, h, bytes)
        self._frame_time = 0.0

    # ── messages ────────────────────────────────────────────────────────────
    def _(self, text: str) -> str:
        return til.localize(text, self.lang)

    def _fail(self, text: str):
        raise DapError(self._(text))

    @contextlib.contextmanager
    def hold_events(self):
        """Events emitted while the block runs (by any thread) are delivered when it
        ends - after whatever the block sent itself. Blocks nest in a thread (the
        outermost one delivers); when blocks of several threads overlap, the end of
        each thread's outermost block delivers what is held (nothing waits for
        long)."""
        local = self._local
        depth = getattr(local, 'depth', 0)
        local.depth = depth + 1
        if not depth:
            with self._emit_lock:
                self._holds += 1
        try:
            yield
        finally:
            local.depth = depth
            if not depth:
                with self._emit_lock:
                    self._holds -= 1
                    held, self._held = self._held, []
                    for msg in held:
                        self._deliver(msg)

    def _deliver(self, msg: dict):
        self._seq += 1
        msg['seq'] = self._seq
        try:
            self._emit(msg)
        except Exception:               # noqa: BLE001 - a broken client must not stop the program
            pass

    def _event(self, name: str, body: Optional[dict] = None, now: bool = False):
        """Emit an event (held back inside hold_events(); now: delivered at once,
        after what is held - a diagnostic that must come before a failed response)."""
        msg: Dict[str, Any] = {'type': 'event', 'event': name}
        if body is not None:
            msg['body'] = body
        with self._emit_lock:
            if self._holds and not now:
                self._held.append(msg)
                return
            held, self._held = self._held, []
            for m in held:
                self._deliver(m)
            self._deliver(msg)

    def _response(self, request: dict, success: bool, body: Optional[dict] = None,
                  message: Optional[str] = None) -> dict:
        seq = request.get('seq', 0)
        resp: Dict[str, Any] = {'type': 'response', 'request_seq': seq if isinstance(seq, int) else 0,
                                'success': success, 'command': request.get('command', '')}
        if message is not None:
            resp['message'] = message
        if body is not None:
            resp['body'] = body
        with self._emit_lock:
            self._seq += 1
            resp['seq'] = self._seq
        return resp

    def notice(self, text: str):
        """A message of the adapter for the user (an `output` event, category console)."""
        self._flush_output()
        self._event('output', {'category': 'console', 'output': self._(text) + "\n"})

    # ── requests ────────────────────────────────────────────────────────────
    def handle(self, message: dict) -> List[dict]:
        """One message from the client -> the responses to send now (events go
        through emit)."""
        with self.hold_events():
            with self._lock:
                responses = self._dispatch(message)
            if self._closing:
                self._join(5.0)
        return responses

    def _dispatch(self, message) -> List[dict]:
        if not isinstance(message, dict) or message.get('type') != 'request':
            return []
        command = message.get('command')
        if not isinstance(command, str) or not command:
            return [self._response(message, False, message=self._("The request has no command"))]
        name = self._HANDLERS.get(command)
        if name is None:
            return [self._response(message, False, message=self._(f"Unknown request '{command}'"))]
        args = message.get('arguments')
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return [self._response(message, False,
                                   message=self._(f"The arguments of '{command}' must be an object"))]
        try:
            body = getattr(self, name)(args)
        except DapError as e:
            return [self._response(message, False, message=str(e))]
        except Exception as e:          # noqa: BLE001 - a bug here must not end the session
            return [self._response(message, False,
                                   message=self._(f"Internal error: {type(e).__name__}: {e}"))]
        return [self._response(message, True, body)]

    def close(self):
        """End the session: the program is stopped, the worker thread ends."""
        with self._lock:
            self._closing = True
            self._kill()
            if self.state != 'closed':
                self.state = 'closed'
        self._join(5.0)

    # ── arguments ───────────────────────────────────────────────────────────
    def _str_arg(self, args: dict, name: str, required: bool = False) -> Optional[str]:
        v = args.get(name)
        if v is None:
            if required:
                self._fail(f"The argument '{name}' is missing")
            return None
        if not isinstance(v, str):
            self._fail(f"The argument '{name}' must be a string")
        return v

    def _int_arg(self, args: dict, name: str, required: bool = False) -> Optional[int]:
        v = args.get(name)
        if v is None:
            if required:
                self._fail(f"The argument '{name}' is missing")
            return None
        if type(v) is not int:
            self._fail(f"The argument '{name}' must be an integer")
        return v

    def _bool_arg(self, args: dict, name: str) -> bool:
        v = args.get(name, False)
        if v is None:
            return False
        if type(v) is not bool:
            self._fail(f"The argument '{name}' must be true or false")
        return v

    def _line_in(self, line: int) -> int:
        return line if self.lines_start_at_1 else line + 1

    def _line_out(self, line: int) -> int:
        return line if self.lines_start_at_1 or not line else line - 1

    def _column_out(self) -> int:
        return 1 if self.columns_start_at_1 else 0

    def _path(self, path: str, base: Optional[str] = None) -> str:
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.join(base or self.root or os.getcwd(), path)
        return os.path.abspath(path)

    def _source(self, file: Optional[str]) -> Optional[dict]:
        """The DAP Source of a file of the program (None: the main file)."""
        if file is None:
            file = self.engine.path if self.engine else self.program
        if not file or file.startswith('<'):
            return None
        name = os.path.basename(file)
        if self.root:
            try:
                rel = os.path.relpath(file, self.root)
            except ValueError:              # another drive (Windows)
                rel = '..'
            if not rel.startswith('..') and not os.path.isabs(rel):
                name = rel.replace(os.sep, '/')
        return {'name': name, 'path': file}

    # ── the state ───────────────────────────────────────────────────────────
    def _need_engine(self):
        if self.engine is None:
            self._fail("No program has been launched")

    def _need_stopped(self):
        self._need_engine()
        if self.state == 'stopped':
            return
        if self.state == 'running':
            self._fail("The program is running")
        if self.state == 'launched':
            self._fail("The program has not started yet (configurationDone starts it)")
        self._fail("The program has ended")

    def _clear_refs(self):
        self._refs.clear()
        self._ref_of.clear()

    def _ref(self, key, value=None, evaluate_name: Optional[str] = None) -> int:
        r = self._ref_of.get(key)
        if r is None:
            r = self._next_ref
            self._next_ref += 1
            self._ref_of[key] = r
            self._refs[r] = (key, value, evaluate_name)
        return r

    def _resume(self, command: str, arg=None):
        """Hand a run command to the worker (the program is stopped)."""
        self._need_stopped()
        self.state = 'running'
        self._clear_refs()
        self._commands.put((command, arg))

    # ── initialize, launch, configuration ───────────────────────────────────
    def _initialize(self, args: dict) -> dict:
        self.lines_start_at_1 = args.get('linesStartAt1', True) is not False
        self.columns_start_at_1 = args.get('columnsStartAt1', True) is not False
        locale = args.get('locale')
        if self.lang is None and isinstance(locale, str):
            self.lang = til.message_language_code(locale)
        return capabilities(self.lang)

    def _attach(self, args: dict):
        self._fail("attach is not supported: the debug adapter launches the program (use launch)")

    def _launch(self, args: dict) -> dict:
        if self.state != 'new':
            self._fail("The program has already been launched")
        lang = self._str_arg(args, 'lang')
        if lang:
            code = til.message_language_code(lang)
            if code is None:
                self._fail(f"unknown message language '{lang}' (use: en, ru, kk)")
            self.lang = code
        program = self._str_arg(args, 'program')
        source = self._str_arg(args, 'source')
        if not program and source is None:
            self._fail("launch needs a 'program' (the path of a .til or .tbc file)")
        cwd_arg = self._str_arg(args, 'cwd')
        cwd = self._path(cwd_arg) if cwd_arg else (self.root or os.getcwd())
        path = self._path(program, self.root or cwd) if program else os.path.join(cwd, "<source>.til")
        argv = args.get('args', [])
        if isinstance(argv, str):
            try:
                argv = shlex.split(argv)
            except ValueError:
                argv = None
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            self._fail("The argument 'args' must be a list of strings")
        stdin = self._str_arg(args, 'stdin')
        keys = args.get('keys')
        if isinstance(keys, list) and all(type(k) is int for k in keys):
            key_codes = list(keys)
        elif isinstance(keys, list) and all(isinstance(k, str) for k in keys):
            keys = ",".join(keys)
            key_codes = None
        elif keys is None or isinstance(keys, str):
            key_codes = None
        else:
            self._fail("The argument 'keys' must be a key script (\"UP,UP,ENTER\") or a list of key codes")
        if key_codes is None:
            try:
                key_codes = til.parse_key_script(keys or "")
            except ValueError as e:
                self._fail(str(e))
        self.stop_on_entry = self._bool_arg(args, 'stopOnEntry')
        self.no_debug = self._bool_arg(args, 'noDebug')
        self.stop_at_end = self._bool_arg(args, 'stopAtEnd')
        fuse = not self._bool_arg(args, 'noFuse')
        every = self._int_arg(args, 'snapshotEvery')
        if every is not None and every < 1:
            self._fail("The argument 'snapshotEvery' must be a positive integer")
        if source is None and not os.path.isfile(path):
            self._fail(f"File not found: {path}")
        try:
            module, lines, info = tildebug.load_program(path, source, fuse)
            tildebug.debug_execute_code()
        except SyntaxError as e:
            self._compile_failed(f"Syntax Error: {e}")
        except til.CompileError as e:
            self._compile_failed(str(e))
        except (OSError, ValueError, UnicodeDecodeError, RuntimeError) as e:
            self._fail(f"Error: {e}")
        if stdin is not None:
            self._input.extend(stdin.encode('utf-8').split(b"\n"))
            if stdin.endswith("\n"):
                self._input.pop()
            elif stdin == "":
                self._input.clear()
            self._input_eof = True
        self.program = path
        self.engine = tildebug.DebugSession(
            module, lines, path, args=argv, stdin=self._read_input, keys=key_codes, screen=til.TILScreen(),
            out=self._program_out, err=self._program_err, on_screen=self._on_screen,
            every=10 ** 9 if self.no_debug else (every or tildebug.SNAPSHOT_EVERY), cwd=cwd,
            lang=self.lang, info=info)
        self.state = 'launched'
        if not module.lines:
            self.notice("The program has no line table (built with --strip): it has no line events")
        for key, (file, specs, ids) in list(self._pending.items()):
            for item in self._install(file, specs, ids):
                self._event('breakpoint', {'reason': 'changed', 'breakpoint': item})
        self._pending.clear()
        self._event('initialized')
        return {}

    def _compile_failed(self, text: str):
        """A program that does not compile: the whole diagnostic as output, its first
        line as the message of the failed launch."""
        text = self._(text)
        self._event('output', {'category': 'stderr', 'output': text.rstrip("\n") + "\n"}, now=True)
        first = next((ln for ln in tildebug._ANSI.sub('', text).split("\n") if ln.strip()), text)
        raise DapError(first.strip())

    def _configuration_done(self, args: dict) -> dict:
        if self.state == 'new':
            self._fail("No program has been launched")
        if self.state != 'launched':
            return {}
        self.state = 'running'
        first = 'run' if self.no_debug or not self.stop_on_entry else 'entry'
        self._worker = threading.Thread(target=self._work, args=(first,), name="til-dap-program",
                                        daemon=True)
        self._worker.start()
        return {}

    # ── breakpoints ─────────────────────────────────────────────────────────
    def _breakpoint_specs(self, args: dict) -> List[tuple]:
        specs = args.get('breakpoints')
        if specs is None and isinstance(args.get('lines'), list):
            specs = [{'line': ln} for ln in args['lines']]
        if specs is None:
            specs = []
        if not isinstance(specs, list):
            self._fail("The argument 'breakpoints' must be a list")
        out = []
        for bp in specs:
            if not isinstance(bp, dict) or type(bp.get('line')) is not int:
                self._fail("The argument 'breakpoints' must be a list")
            cond = bp.get('condition')
            hit = bp.get('hitCondition')
            out.append((self._line_in(bp['line']), cond if isinstance(cond, str) else None,
                        hit if isinstance(hit, str) else None))
        return out

    def _install(self, file: str, specs: List[tuple], ids: Optional[List[int]] = None) -> List[dict]:
        engine = self.engine
        source = self._source(file)
        if not engine.has_file(file):
            text = self._("This file is not part of the program being debugged")
            return [dict({'verified': False, 'line': self._line_out(line), 'message': text, 'source': source},
                         **({'id': ids[i]} if ids else {})) for i, (line, _c, _h) in enumerate(specs)]
        out = []
        for i, result in enumerate(engine.set_file_breakpoints(file, specs)):
            if isinstance(result, tildebug.DebugError):
                item = {'verified': False, 'line': self._line_out(specs[i][0]), 'message': self._(str(result))}
                if ids:
                    item['id'] = ids[i]
            else:
                if ids and result.tag is None:
                    result.tag = ids[i]
                item = {'id': result.tag if result.tag is not None else result.id, 'verified': True,
                        'line': self._line_out(result.line)}
            item['source'] = source
            out.append(item)
        return out

    def _set_breakpoints(self, args: dict) -> dict:
        src = args.get('source')
        if not isinstance(src, dict) or not isinstance(src.get('path'), str):
            self._fail("The argument 'source' must be an object with a 'path'")
        specs = self._breakpoint_specs(args)
        file = self._path(src['path'])
        if self.engine is None:
            ids = list(range(self._next_pending, self._next_pending + len(specs)))
            self._next_pending += len(specs)
            self._pending[tildebug._file_key(file)] = (file, specs, ids)
            text = self._("The program has not been launched yet")
            return {'breakpoints': [{'id': ids[i], 'verified': False, 'line': self._line_out(line), 'message': text}
                                    for i, (line, _c, _h) in enumerate(specs)]}
        return {'breakpoints': self._install(file, specs)}

    def _set_exception_breakpoints(self, args: dict) -> dict:
        filters = args.get('filters') or []
        options = args.get('filterOptions') or []
        if not isinstance(filters, list) or not isinstance(options, list):
            self._fail("The argument 'filters' must be a list")
        names = [f for f in filters if isinstance(f, str)]
        names += [o.get('filterId') for o in options if isinstance(o, dict)]
        self.filters = {f for f in names if f == EXCEPTION_FILTER}
        out = []
        for f in names:
            if f == EXCEPTION_FILTER:
                out.append({'verified': True})
            else:
                out.append({'verified': False, 'message': self._(f"Unknown exception filter '{f}'")})
        return {'breakpoints': out}

    def _set_function_breakpoints(self, args: dict) -> dict:
        bps = args.get('breakpoints') or []
        text = self._("Function breakpoints are not supported: set a breakpoint on a line")
        return {'breakpoints': [{'verified': False, 'message': text} for _ in bps if isinstance(bps, list)]}

    def _breakpoint_locations(self, args: dict) -> dict:
        src = args.get('source')
        line = self._int_arg(args, 'line', required=True)
        end = self._int_arg(args, 'endLine') or line
        if self.engine is None or not isinstance(src, dict) or not isinstance(src.get('path'), str):
            return {'breakpoints': []}
        lines = self.engine.code_lines_of(self._path(src['path']))
        lo, hi = self._line_in(line), self._line_in(end)
        return {'breakpoints': [{'line': self._line_out(ln)} for ln in lines if lo <= ln <= hi]}

    # ── threads, stack, variables ───────────────────────────────────────────
    def _threads(self, args: dict) -> dict:
        if self.engine is None:
            return {'threads': []}
        return {'threads': [{'id': THREAD_ID, 'name': 'main'}]}

    def _depth_of(self, frame_id) -> int:
        if frame_id is None:
            return 0
        entry = self._refs.get(frame_id) if type(frame_id) is int else None
        if entry is None or entry[0][0] != 'frame':
            self._fail(f"Invalid frameId {frame_id} (frames are valid only while the program is stopped)")
        return entry[0][1]

    def _stack_trace(self, args: dict) -> dict:
        self._need_stopped()
        frames = self.engine.stack()
        start = self._int_arg(args, 'startFrame') or 0
        levels = self._int_arg(args, 'levels') or 0
        chosen = frames[start:start + levels] if levels > 0 else frames[start:]
        out = []
        for fr in chosen:
            item: Dict[str, Any] = {'id': self._ref(('frame', fr.depth)), 'name': fr.name,
                                    'line': self._line_out(fr.line), 'column': self._column_out()}
            source = None if fr.hidden or not fr.line else self._source(fr.file)
            if source is not None:
                item['source'] = source
            else:
                item['line'] = 0
                item['column'] = 0
            if fr.hidden:
                item['presentationHint'] = 'subtle'
            out.append(item)
        return {'stackFrames': out, 'totalFrames': len(frames)}

    def _scopes(self, args: dict) -> dict:
        self._need_stopped()
        depth = self._depth_of(args.get('frameId'))
        engine = self.engine
        n_locals = len(engine.locals(depth)) + (1 if depth == 0 and self._returned() else 0)
        return {'scopes': [
            {'name': 'Locals', 'presentationHint': 'locals', 'variablesReference': self._ref(('locals', depth)),
             'namedVariables': n_locals, 'expensive': False},
            {'name': 'Globals', 'presentationHint': 'globals', 'variablesReference': self._ref(('globals',)),
             'namedVariables': len(engine.globals()), 'expensive': False},
        ]}

    def _returned(self):
        stop = self._stop
        return stop.returned if stop is not None and not stop.finished else None

    def _variable(self, name: str, value, evaluate_name: Optional[str] = None,
                  limit: int = tildebug.PREVIEW) -> dict:
        vm = self.engine.vm
        indexed, named = tildebug.child_counts(vm, value)
        var: Dict[str, Any] = {'name': name, 'value': tildebug.preview(vm, value, limit),
                               'type': tildebug.value_type(vm, value), 'variablesReference': 0}
        if indexed or named:
            var['variablesReference'] = self._ref(('value', id(value)), value, evaluate_name)
            if indexed:
                var['indexedVariables'] = indexed
            if named:
                var['namedVariables'] = named
        if evaluate_name:
            var['evaluateName'] = evaluate_name
        return var

    def _variables(self, args: dict) -> dict:
        self._need_stopped()
        ref = args.get('variablesReference')
        entry = self._refs.get(ref) if type(ref) is int else None
        if entry is None or entry[0][0] not in ('locals', 'globals', 'value'):
            self._fail(f"Invalid variablesReference {ref} (variables are valid only while the program "
                       f"is stopped)")
        key, value, evaluate_name = entry
        engine = self.engine
        start = self._int_arg(args, 'start') or 0
        count = self._int_arg(args, 'count') or 0
        flt = args.get('filter')
        out = []
        if key[0] == 'locals':
            if key[1] == 0 and self._returned() is not None:
                func, v = self._returned()
                var = self._variable(self._(f"{func}() returned"), v)
                var['presentationHint'] = {'kind': 'virtual', 'attributes': ['readOnly']}
                out.append(var)
            out += [self._variable(name, v, name) for name, v in engine.locals(key[1])]
        elif key[0] == 'globals':
            out = [self._variable(name, v, name) for name, v in engine.globals()]
        else:
            is_array = type(value) is list
            if (flt == 'named' and is_array) or (flt == 'indexed' and not is_array):
                return {'variables': []}
            for name, v, suffix in engine.children(value, start, count or None):
                out.append(self._variable(name, v, evaluate_name + suffix if evaluate_name and suffix else None))
            return {'variables': out}
        if start or count:
            out = out[start:start + count] if count else out[start:]
        return {'variables': out}

    def _evaluate(self, args: dict) -> dict:
        expr = self._str_arg(args, 'expression', required=True)
        context = args.get('context') or 'repl'
        if self.state == 'running' and context == 'repl' and self.waiting_for_input:
            self._feed([expr.encode('utf-8')], False)       # the debug console as the program's stdin
            return {'result': '', 'variablesReference': 0}
        self._need_stopped()
        depth = self._depth_of(args.get('frameId'))
        if not expr.strip():
            self._fail("The argument 'expression' is missing")
        try:
            value = self.engine.evaluate(expr, depth)
        except tildebug.DebugError as e:
            raise DapError(self._(str(e)))
        limit = tildebug.MAX_SHOWN if context in ('repl', 'clipboard') else tildebug.PREVIEW
        var = self._variable(expr.strip(), value, expr.strip(), limit)
        body = {'result': var['value'], 'type': var['type'], 'variablesReference': var['variablesReference']}
        for k in ('indexedVariables', 'namedVariables'):
            if k in var:
                body[k] = var[k]
        return body

    def _exception_info(self, args: dict) -> dict:
        self._need_stopped()
        stop = self._stop
        if stop is None or stop.reason != 'exception':
            self._fail("The program is not stopped at an exception")
        text = self._error_text(stop)
        vm = self.engine.vm
        trace = ""
        if vm.end_error is not None and vm.end_error[1] >= 0:
            trace = til.localize_report_bytes(vm.traceback(vm.end_error[1]), self.lang).decode('utf-8', 'replace')
        return {'exceptionId': EXCEPTION_FILTER, 'description': text, 'breakMode': 'unhandled',
                'details': {'message': text, 'typeName': EXCEPTION_FILTER, 'stackTrace': trace}}

    def _error_text(self, stop: tildebug.Stop) -> str:
        if not stop.error:
            return ""
        return til.localize_report_bytes(stop.error.encode('utf-8'), self.lang).decode('utf-8', 'replace')

    # ── running ─────────────────────────────────────────────────────────────
    def _continue(self, args: dict) -> dict:
        self._resume('continue')
        return {'allThreadsContinued': True}

    def _next(self, args: dict) -> dict:
        self._resume('next')
        return {}

    def _step_in(self, args: dict) -> dict:
        self._resume('stepIn')
        return {}

    def _step_out(self, args: dict) -> dict:
        self._resume('stepOut')
        return {}

    def _step_back(self, args: dict) -> dict:
        self._need_stopped()
        if self.engine.back_over_target() is None:
            self._fail("Already at the start of the program")
        self._resume('stepBack')
        return {}

    def _reverse_continue(self, args: dict) -> dict:
        self._need_stopped()
        if not self.engine.can_step_back():
            self._fail("Already at the start of the program")
        self._resume('reverseContinue')
        return {}

    def _pause(self, args: dict) -> dict:
        if self.state == 'running' and self.engine is not None:
            self.engine.interrupt()
        return {}

    def _til_goto(self, args: dict) -> dict:
        index = self._int_arg(args, 'index', required=True)
        self._need_stopped()
        engine = self.engine
        total = len(engine.vm.ev_lines)
        if not (0 <= index < total or (index == total and engine.end_seen and engine.vm.snaps)):
            self._fail(f"No line event {index} in the history (0 to {total - 1})")
        # a custom request: tell the client the program runs (before the worker can stop it)
        self._event('continued', {'threadId': THREAD_ID, 'allThreadsContinued': True})
        self._resume('goto', index)
        return {}

    def _til_timeline(self, args: dict) -> dict:
        if self.engine is None:
            return {'index': 0, 'total': 0, 'complete': False}
        index, total = self.engine.timeline()
        return {'index': index, 'total': total, 'complete': self.engine.end_seen}

    def _til_stdin(self, args: dict) -> dict:
        text = self._str_arg(args, 'text')
        eof = self._bool_arg(args, 'eof')
        lines: List[bytes] = []
        if text is not None:
            data = text[:-1] if text.endswith("\n") else text
            lines = [ln.encode('utf-8') for ln in data.split("\n")]
        self._feed(lines, eof)
        return {}

    def _feed(self, lines: List[bytes], eof: bool):
        with self._input_cond:
            self._input.extend(lines)
            if eof:
                self._input_eof = True
            self._input_cond.notify_all()

    def _terminate(self, args: dict) -> dict:
        if self.state in ('running', 'stopped'):
            self._kill()
        elif self.state in ('new', 'launched'):
            self.state = 'terminated'
            self._event('terminated', {})
        return {}

    def _disconnect(self, args: dict) -> dict:
        self._closing = True
        self._kill()
        if self.state in ('new', 'launched'):
            self.state = 'terminated'
        return {}

    def _cancel(self, args: dict) -> dict:
        return {}

    def _loaded_sources(self, args: dict) -> dict:
        if self.engine is None:
            return {'sources': []}
        return {'sources': [s for s in (self._source(f) for f in self.engine.files()) if s]}

    def _source_request(self, args: dict):
        self._fail("Source requests are not supported: the stack frames name their files")

    def _kill(self):
        self._killed = True
        with self._input_cond:
            self._input_cond.notify_all()
        if self.engine is not None:
            self.engine.kill()
        self._commands.put(('kill', None))

    def _join(self, timeout: float):
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout)

    # ── the worker thread: runs the program ─────────────────────────────────
    def _work(self, first: str):
        command, arg = first, None
        try:
            while True:
                try:
                    stop = self._execute(command, arg)
                except tildebug.DebugError as e:
                    self.notice(str(e))         # could not run after all: stay where it was
                    stop = self._stop
                if self._report(stop):
                    return
                command, arg = self._commands.get()
                if command == 'kill':
                    raise tildebug.Killed()
        except tildebug.Killed:
            self._ended(None)
        except Exception as e:          # noqa: BLE001 - say what broke, end the session cleanly
            self.notice(f"Internal error: {type(e).__name__}: {e}")
            self._ended(None)

    def _execute(self, command: str, arg) -> Optional[tildebug.Stop]:
        """Run one command of the client; None: the program has ended (the session ends)."""
        e = self.engine
        if self._killed:
            raise tildebug.Killed()
        if command == 'entry':
            return e.start()
        if command == 'run':
            return e.run_to_end() if self.no_debug else e.cont()
        if e.finished and command in ('continue', 'next', 'stepIn', 'stepOut'):
            return None                 # going on from the stop at the end
        if command == 'continue':
            return e.cont()
        if command == 'next':
            return e.step_over()
        if command == 'stepIn':
            return e.step_in()
        if command == 'stepOut':
            return e.step_out()
        if command == 'stepBack':
            return e.step_back_over()
        if command == 'reverseContinue':
            return e.reverse_continue()
        if command == 'goto':
            return e.goto(arg)
        raise ValueError(f"unknown command {command}")

    def _report(self, stop: Optional[tildebug.Stop]) -> bool:
        """The events of a stop; True when the session has ended."""
        engine = self.engine
        self._flush_output()
        if engine.vm.screen.w:
            self._send_screen(engine.vm.screen)     # the screen as it is now (also at the end)
        if stop is None:
            self._ended(engine.vm.finished[0] if engine.vm.finished else None)
            return True
        if stop.diverged is not None:
            self.notice(f"Warning: the replay differs from the recorded run at line event {stop.diverged}")
        if stop.note:
            self.notice(stop.note)
        if stop.condition_error is not None:
            self.notice(stop.condition_error[1])
        if stop.finished and (self.no_debug or not (
                (stop.reason == 'exception' and EXCEPTION_FILTER in self.filters) or self.stop_at_end)):
            self._ended(stop.exit_code)
            return True
        with self._lock:
            if self._killed:
                raise tildebug.Killed()
            self._stop = stop
            self.state = 'stopped'
            self._clear_refs()
            engine.vm.pause_pending = False
        body: Dict[str, Any] = {'reason': stop.reason, 'threadId': THREAD_ID, 'allThreadsStopped': True}
        if stop.reason == 'breakpoint' and stop.fired is not None:
            body['hitBreakpointIds'] = [stop.fired.tag if stop.fired.tag is not None else stop.fired.id]
        if stop.reason == 'exception':
            body['description'] = self._("Paused on exception")
            body['text'] = self._error_text(stop)
        elif stop.reason == 'exit':
            body['description'] = self._(f"Program exited with code {stop.exit_code}")
        self._event('stopped', body)
        return False

    def _ended(self, code: Optional[int]):
        with self._lock:
            if self.state != 'closed':
                self.state = 'terminated'
        self._flush_output()
        if code is not None:
            self._event('exited', {'exitCode': code})
        self._event('terminated', {})

    # ── what the program does: input, output, screen ───────────────────────
    def _read_input(self) -> bytes:
        """input() of the program (worker thread): the next line given with launch
        `stdin`, til/stdin or the debug console; waits for one (til/inputRequested)."""
        self._flush_output()
        asked = False
        while True:
            with self._input_cond:
                if self._killed:
                    raise tildebug.Killed()
                if self._input:
                    self.waiting_for_input = False
                    return self._input.popleft()
                if self._input_eof:
                    self.waiting_for_input = False
                    return b""
                if asked:
                    self._input_cond.wait()
                    continue
                self.waiting_for_input = True
            asked = True
            self._event('til/inputRequested', {})
            if not self._input_hint:
                self._input_hint = True
                self.notice("The program is waiting for input: send it with til/stdin, or type it in "
                            "the debug console")

    def _program_out(self, data: bytes):
        text = self._decode_out.decode(data)
        if not text:
            return
        with self._out_lock:
            self._out_buf.append(text)
            self._out_size += len(text)
            full = self._out_size >= OUTPUT_CHUNK
            if not full and self._out_timer is None:
                self._out_timer = threading.Timer(OUTPUT_DELAY, self._flush_output)
                self._out_timer.daemon = True
                self._out_timer.start()
        if full:
            self._flush_output()

    def _flush_output(self):
        with self._out_lock:
            if self._out_timer is not None:
                self._out_timer.cancel()
                self._out_timer = None
            if self._out_buf:
                text = "".join(self._out_buf)
                self._out_buf = []
                self._out_size = 0
                self._event('output', {'category': 'stdout', 'output': text})

    def _program_err(self, data: bytes):
        self._flush_output()
        text = self._decode_err.decode(data, final=True)
        if text:
            self._event('output', {'category': 'stderr', 'output': text})

    def _on_screen(self, screen):
        if time.monotonic() - self._frame_time >= FRAME_INTERVAL:
            self._send_screen(screen)

    def _send_screen(self, screen):
        frame = (screen.w, screen.h, bytes(screen.fb))
        if frame == self._frame:
            return
        self._flush_output()
        self._frame = frame
        self._frame_time = time.monotonic()
        self._event('til/screen', {'w': screen.w, 'h': screen.h,
                                   'rgb_b64': base64.b64encode(frame[2]).decode('ascii')})


# ═══════════════════════════════════════════════════════════════════════════
# the stdio server
# ═══════════════════════════════════════════════════════════════════════════

class StdioServer:
    """DAP over two binary streams with Content-Length framing."""

    def __init__(self, inp, out, log=None):
        self.inp = inp
        self.out = out
        self.log = log
        self.wlock = threading.Lock()

    def _log(self, direction: str, data: bytes):
        if self.log is not None:
            try:
                self.log.write(f"{time.strftime('%H:%M:%S')} {direction} ".encode('utf-8') + data + b"\n")
                self.log.flush()
            except (OSError, ValueError):
                pass

    def send(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        with self.wlock:
            try:
                self.out.write(b"Content-Length: %d\r\n\r\n" % len(data) + data)
                self.out.flush()
            except (OSError, ValueError):
                pass
            self._log("->", data)

    def read(self):
        """The next message: a dict (or any JSON value), str (a problem with the
        frame, to report) or None (end of input)."""
        length = None
        headers = False
        while True:
            line = self.inp.readline()
            if not line:
                return None
            text = line.strip()
            if not text:
                if length is not None:
                    break
                if headers:
                    return "a message without a Content-Length header"
                continue                    # blank lines between messages
            headers = True
            name, sep, value = text.partition(b":")
            if not sep:
                return f"not a header: {text[:60].decode('utf-8', 'replace')}"
            if name.strip().lower() == b"content-length":
                try:
                    length = int(value.strip())
                    if length < 0:
                        raise ValueError
                except ValueError:
                    return f"bad Content-Length header: {value.strip()[:40].decode('latin-1')}"
        body = self.inp.read(length)
        if body is None or len(body) < length:
            return None
        self._log("<-", body)
        try:
            return json.loads(body.decode('utf-8'))
        except (UnicodeDecodeError, ValueError) as e:
            return f"invalid JSON ({e})"

    def run(self) -> int:
        session = DapSession(self.send)
        try:
            while True:
                msg = self.read()
                if msg is None:
                    break
                if isinstance(msg, str):
                    session.notice(f"Cannot read a message: {msg}")
                    continue
                with session.hold_events():
                    for response in session.handle(msg):
                        self.send(response)
                if isinstance(msg, dict) and msg.get('type') == 'request' and msg.get('command') == 'disconnect':
                    break
        finally:
            session.close()
        return 0


@til.scoped_message_language
def main(argv: List[str]) -> int:
    if any(a == '--lang' or a.startswith('--lang=') for a in argv):
        err = til.set_message_language(argv)        # called directly (til.py strips --lang)
        if err:
            print(err, file=sys.stderr)
            return 2
        argv = til.without_lang_option(argv)
    log_path = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('-h', '--help'):
            print(USAGE)
            return 0
        if a == '--log' and i + 1 < len(argv):
            log_path = argv[i + 1]
            i += 2
            continue
        if a.startswith('--log='):
            log_path = a.split('=', 1)[1]
            i += 1
            continue
        print(til.localize(f"Error: unknown option {a}"), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    inp = getattr(sys.stdin, 'buffer', None)
    out = getattr(sys.stdout, 'buffer', None)
    if inp is None or out is None:
        print(til.localize("Error: til dap needs binary standard input and output"), file=sys.stderr)
        return 2
    if os.name == 'nt':                 # no newline translation on the protocol streams
        try:
            import msvcrt
            msvcrt.setmode(inp.fileno(), os.O_BINARY)
            msvcrt.setmode(out.fileno(), os.O_BINARY)
        except (ImportError, OSError, ValueError, AttributeError):
            pass
    log = None
    if log_path:
        try:
            log = open(log_path, 'ab')
        except OSError as e:
            print(til.localize(f"Error: {e}"), file=sys.stderr)
            return 2
    saved = sys.stdout
    sys.stdout = sys.stderr             # a stray print() must not break the protocol
    try:
        return StdioServer(inp, out, log).run()
    except KeyboardInterrupt:
        return 130
    finally:
        sys.stdout = saved
        if log is not None:
            log.close()


if __name__ == '__main__':
    sys.modules.setdefault('tildap', sys.modules[__name__])
    sys.exit(main(sys.argv[1:]))
