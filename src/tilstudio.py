#!/usr/bin/env python3
"""
TIL Studio - the local server of the TIL IDE: `til ide [FOLDER] [--port N]
[--no-browser] [--lang en|ru|kk]` (docs/dev/TIL_STUDIO_DESIGN.md, section 5).

A `http.server.ThreadingHTTPServer` on 127.0.0.1 serves the single-file
frontend (ide/studio.html) and a JSON API over the project folder:

* files (tree, read, write with conflict detection, create, rename, delete,
  search), every path confined to the project root (realpath, no symlink
  escapes, no absolute paths, no `..`); the TIL examples are readable as
  `@examples/...`;
* the language server (src/tillsp.py, in-process `LspServer(root)`): one per
  studio, requests answered in the POST reply, notifications also pushed on
  the event stream, debounced diagnostics by a ticker calling poll();
* the debug adapter (src/tildap.py, `DapSession(emit, root)`): sessions by id,
  events on the event stream;
* runs: `vm` in-process on the reference VM (live output, input(), screen
  frames, keys, portable snapshots), `cvm` and `native` as subprocesses
  (`til exec --cvm` / the program built by the native backend) with streamed
  output; at most 4 at a time, 4 MB of output each;
* tools: prove (tilprove, in a worker process), fit (tilfit), structured
  disassembly, translate, build, packages (tilpkg).

Security: a random token per launch (`?token=` in the printed URL, then an
HttpOnly SameSite=Strict cookie; API clients may send `X-TIL-Token`), the
`Host` header must name 127.0.0.1:PORT or localhost:PORT, a request with a
foreign `Origin` (or a cross-site `Sec-Fetch-Site`) is refused, no CORS.

Server-sent events (`GET /api/events`, one stream per client tab):
    lsp   one LSP message (notification or request from the server)
    run   {runId, kind, ...}  kind: stdout | stderr {text}; status {state: starting |
          compiling | running | snapshot, backend, pid?, path?, steps?}; input {} (the
          program waits for a line); screen {w, h, rgb (base64 RGB24)}; exit {code,
          reason: exited | stopped | failed | paused, seconds}
    dap   {sessionId, message}
    fs    {paths: [...]}      files and folders changed on disk
    ping  {time}              at connect and every 15 s
"""

import base64
import bisect
import codecs
import collections
import contextlib
import errno
import fnmatch
import hmac
import http.cookies
import http.server
import importlib
import importlib.util
import json
import os
import queue
import re
import secrets
import shutil
import socket
import socketserver
import stat as _stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple

import til

# ── limits and timings ──────────────────────────────────────────────────────
MAX_RUNS = 4                          # concurrent runs
OUTPUT_LIMIT = 4 * 1024 * 1024        # program output forwarded per run (stdout + stderr)
FLUSH_AGE = 0.04                      # output is sent at most this late (chunks <= 50 ms)
PUMP_SECONDS = 0.01                   # how often pending output / frames are looked at
FRAME_FPS = 30                        # screen frames per second at most
FRAME_BYTES_PER_SECOND = 48 * 1024 * 1024   # big screens: fewer frames per second
PING_SECONDS = 15.0
WATCH_SECONDS = 1.0
WATCH_LIMIT = 20000                   # entries looked at per file-watch poll
LSP_TICK_SECONDS = 0.05
SEARCH_LIMIT = 2000
SEARCH_SECONDS = 10.0
SEARCH_MAX_FILE = 4 * 1024 * 1024
OPEN_LIMIT = 16 * 1024 * 1024         # largest file GET /api/file returns
BODY_LIMIT = 64 * 1024 * 1024         # largest request body
TREE_LIMIT = 10000                    # entries per folder listing
CLIENT_QUEUE_LIMIT = 64 * 1024 * 1024  # undelivered event bytes before a client is dropped
SNAPSHOT_WAIT = 3.0                   # seconds a snapshot waits for the program to reach a step
PROVE_SECONDS = 600                   # the prover worker's time limit
FIT_RUN_SECONDS = 60                  # fit --run's time limit
BUILD_SECONDS = 600
MAX_DAP_SESSIONS = 8
FINISHED_RUNS_KEPT = 16
MTIME_TOLERANCE = 0.001               # PUT /api/file: mtime compared to the millisecond

HIDDEN_NAMES = frozenset(('.git', '.hg', '.svn', '__pycache__', '.DS_Store'))
COLLAPSED_NAMES = frozenset((getattr(til, 'PACKAGE_DIR', 'til_modules'), 'node_modules'))
TEMP_SUFFIX = '.til-studio-tmp'       # atomic saves write NAME.<random>.til-studio-tmp first
EXAMPLES = '@examples'                # read-only virtual folder: the TIL distribution's examples
BACKENDS = ('native', 'vm', 'cvm')
PKG_COMMANDS = ('init', 'add', 'install', 'remove', 'rm', 'list', 'ls', 'verify')
BUILD_TARGETS = ('vm', 'c-vm', 'native')
TRANSLATE_LANGUAGES = tuple(til.LANGUAGES)
_SLEEP_NATIVE = til.NATIVE_ID['sleep_ms']
_GFX_NATIVES = range(til.GFX_FIRST_NATIVE, til.GFX_LAST_NATIVE + 1)
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_MB = 1024 * 1024


def _t(text: str, lang: Optional[str]) -> str:
    """A user-facing message in `lang` (src/tilmsg.py; English without a catalog)."""
    return til.localize(text, lang or 'en')


def _plain(text: str) -> str:
    """Diagnostics without terminal colour codes."""
    return _ANSI_RE.sub('', text)


class _ApiError(Exception):
    """An API request that fails: HTTP status + the (localized) text of `{"error": ...}`."""

    def __init__(self, status: int, text: str):
        super().__init__(text)
        self.status = status
        self.text = text


class _Stopped(BaseException):
    """Raised in a run's own thread when the run is stopped. A BaseException:
    the VM turns ordinary exceptions raised inside builtins into TIL errors."""


def _json_safe(value: Any) -> Any:
    """JSON-ready data: no NaN/Infinity (null), tuples and sets as lists, other objects as text."""
    if isinstance(value, float):
        return value if value == value and value not in (float('inf'), float('-inf')) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    return str(value)


def _dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)


def _utf16_len(text: str) -> int:
    return len(text.encode('utf-16-le')) // 2


def _js_value(v: Any) -> Any:
    """A counterexample value for JavaScript: integers beyond 2**53 as exact decimal text."""
    if isinstance(v, bool) or v is None or isinstance(v, (str, float)):
        return _json_safe(v)
    if isinstance(v, int):
        return v if -(2 ** 53 - 1) <= v <= 2 ** 53 - 1 else str(v)
    return _json_safe(v)


def _til_script() -> str:
    return os.path.abspath(til.__file__)


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# ── output of library code in request / run threads ────────────────────────
# Some code prints (til pkg, warnings of the import resolver). While a thread
# captures, sys.stdout / sys.stderr route its writes to that thread's sinks;
# every other thread writes to the original streams as before.

_capture_lock = threading.Lock()
_capture_sinks: Dict[int, Tuple[Callable[[str], None], Callable[[str], None]]] = {}
_capture_state: Dict[str, Any] = {'refs': 0, 'saved': None}


class _BinarySink:
    def __init__(self, sink: Callable[[str], None]):
        self._sink = sink

    def write(self, data) -> int:
        self._sink(bytes(data).decode('utf-8', 'replace'))
        return len(data)

    def flush(self):
        pass


class _StreamRouter:
    """sys.stdout / sys.stderr while some thread captures its output."""

    def __init__(self, original, index: int):
        self._original = original
        self._index = index

    def _sink(self):
        sinks = _capture_sinks.get(threading.get_ident())
        return sinks[self._index] if sinks is not None else None

    def write(self, text):
        sink = self._sink()
        if sink is not None:
            sink(text if isinstance(text, str) else bytes(text).decode('utf-8', 'replace'))
            return len(text)
        if self._original is None:
            return len(text)
        return self._original.write(text)

    def flush(self):
        if self._sink() is None and self._original is not None:
            try:
                self._original.flush()
            except (OSError, ValueError):
                pass

    def isatty(self) -> bool:
        if self._sink() is not None or self._original is None:
            return False
        try:
            return self._original.isatty()
        except (OSError, ValueError):
            return False

    @property
    def buffer(self):
        sink = self._sink()
        if sink is not None:
            return _BinarySink(sink)
        return getattr(self._original, 'buffer', None)

    @property
    def encoding(self):
        return getattr(self._original, 'encoding', None) or 'utf-8'

    def __getattr__(self, name):
        return getattr(self._original, name)


@contextlib.contextmanager
def capture_output(out_sink: Callable[[str], None], err_sink: Callable[[str], None]):
    """Route this thread's sys.stdout / sys.stderr writes to the sinks."""
    tid = threading.get_ident()
    with _capture_lock:
        if _capture_state['refs'] == 0:
            saved = (sys.stdout, sys.stderr)
            _capture_state['saved'] = saved
            sys.stdout = _StreamRouter(saved[0], 0)
            sys.stderr = _StreamRouter(saved[1], 1)
        _capture_state['refs'] += 1
        outer = _capture_sinks.get(tid)
        _capture_sinks[tid] = (out_sink, err_sink)
    try:
        yield
    finally:
        with _capture_lock:
            if outer is None:
                _capture_sinks.pop(tid, None)
            else:
                _capture_sinks[tid] = outer
            _capture_state['refs'] -= 1
            if _capture_state['refs'] == 0:
                saved = _capture_state['saved']
                _capture_state['saved'] = None
                if isinstance(sys.stdout, _StreamRouter):
                    sys.stdout = saved[0]
                if isinstance(sys.stderr, _StreamRouter):
                    sys.stderr = saved[1]


class _Collector:
    """Text written to a capture sink."""

    def __init__(self):
        self.parts: List[str] = []
        self.lock = threading.Lock()

    def __call__(self, text: str):
        with self.lock:
            self.parts.append(text)

    def text(self) -> str:
        with self.lock:
            return "".join(self.parts)


# ── server-sent events ──────────────────────────────────────────────────────

class _Client:
    """One event stream (a browser tab): undelivered frames, newest screen
    frame per run only (a slow client skips frames, never output)."""

    def __init__(self, sock):
        self.sock = sock
        self.cond = threading.Condition()
        self.items: collections.deque = collections.deque()   # [key, frame bytes]
        self.keyed: Dict[Any, list] = {}
        self.size = 0
        self.closed = False


class EventHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.clients: List[_Client] = []

    def subscribe(self, sock=None) -> _Client:
        client = _Client(sock)
        with self.lock:
            self.clients.append(client)
        return client

    def unsubscribe(self, client: _Client):
        with self.lock:
            if client in self.clients:
                self.clients.remove(client)
        with client.cond:
            client.closed = True
            client.items.clear()
            client.keyed.clear()
            client.cond.notify_all()

    def publish(self, event: str, data: Any, key: Any = None):
        """Send an event to every stream. With `key`, an undelivered event of
        the same key is replaced (screen frames)."""
        frame = ("event: %s\ndata: %s\n\n" % (event, _dumps(data))).encode('utf-8')
        with self.lock:
            clients = list(self.clients)
        for c in clients:
            with c.cond:
                if c.closed:
                    continue
                if key is not None and key in c.keyed:
                    item = c.keyed[key]
                    c.size += len(frame) - len(item[1])
                    item[1] = frame
                    continue
                if c.size + len(frame) > CLIENT_QUEUE_LIMIT:
                    c.closed = True               # a client that does not read: dropped (it reconnects)
                    c.cond.notify_all()
                    continue
                item = [key, frame]
                c.items.append(item)
                if key is not None:
                    c.keyed[key] = item
                c.size += len(frame)
                c.cond.notify_all()

    def take(self, client: _Client, timeout: float) -> Optional[bytes]:
        """Frames waiting for `client` (b"" after `timeout`), None once it is closed."""
        with client.cond:
            if not client.items and not client.closed:
                client.cond.wait(timeout)
            if client.closed:
                return None
            out = []
            while client.items:
                key, frame = client.items.popleft()
                if key is not None:
                    client.keyed.pop(key, None)
                out.append(frame)
            client.size = 0
            return b"".join(out)

    def close_all(self):
        with self.lock:
            clients = list(self.clients)
        for c in clients:
            with c.cond:
                c.closed = True
                c.cond.notify_all()
            if c.sock is not None:
                try:
                    c.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


# ── project paths ───────────────────────────────────────────────────────────

def _examples_dir() -> Optional[str]:
    for home in til.til_home_dirs():
        d = os.path.join(home, 'examples')
        if os.path.isdir(d):
            return os.path.realpath(d)
    return None


def _inside(path: str, folder: str) -> bool:
    a, b = os.path.normcase(path), os.path.normcase(folder)
    return a == b or a.startswith(b.rstrip(os.sep) + os.sep)


class _Project:
    """The project root and the confinement of API paths to it."""

    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        self.examples = _examples_dir()

    def rel(self, abs_path: str) -> Optional[str]:
        """POSIX path relative to the root (None outside it)."""
        real = os.path.realpath(abs_path)
        if not _inside(real, self.root):
            return None
        rel = os.path.relpath(real, self.root)
        return '' if rel == '.' else rel.replace(os.sep, '/')

    def resolve(self, path: Any, lang: Optional[str], write: bool = False, follow: bool = True,
                allow_root: bool = True) -> Tuple[str, str, bool]:
        """API path -> (absolute path, normalized POSIX path, read-only example).
        Refuses absolute paths, `..`, NUL and paths whose real location (symlinks
        resolved) is outside the root. follow=False: the last component itself
        (a link is renamed or deleted, never what it points to)."""
        if path is None:
            path = ''
        if not isinstance(path, str):
            raise _ApiError(400, _t("missing or invalid field 'path'", lang))
        shown = path
        if '\x00' in path:
            raise _ApiError(400, _t(f"invalid path: {shown!r}", lang))
        if os.name == 'nt' and ('\\' in path or ':' in path):
            raise _ApiError(403, _t(f"path outside the project: {shown}", lang))
        if path.startswith('/') or os.path.isabs(path):
            raise _ApiError(403, _t(f"path outside the project: {shown}", lang))
        parts = [p for p in path.split('/') if p not in ('', '.')]
        if '..' in parts:
            raise _ApiError(403, _t(f"path outside the project: {shown}", lang))
        base, virtual = self.root, False
        if parts and parts[0] == EXAMPLES:
            if self.examples is None:
                raise _ApiError(404, _t(f"no such file or folder: {shown}", lang))
            if write:
                raise _ApiError(403, _t(f"the examples are read-only: {shown}", lang))
            base, virtual, parts = self.examples, True, parts[1:]
        norm = '/'.join(([EXAMPLES] if virtual else []) + parts)
        if not parts:
            if not allow_root and not follow:
                raise _ApiError(400, _t("the project folder itself cannot be renamed or deleted", lang))
            if not allow_root:
                raise _ApiError(400, _t(f"not a file: {norm or '.'}", lang))
            return base, norm, virtual
        joined = os.path.join(base, *parts)
        if follow:
            real = os.path.realpath(joined)
        else:
            parent = os.path.realpath(os.path.dirname(joined))
            if not _inside(parent, base):
                raise _ApiError(403, _t(f"path outside the project: {shown}", lang))
            real = os.path.join(parent, parts[-1])
        if not _inside(real, base):
            raise _ApiError(403, _t(f"path outside the project: {shown}", lang))
        return real, norm, virtual


def _entry_sort_key(e: Dict[str, Any]):
    return (e['type'] != 'dir', e['name'].casefold(), e['name'])


# ── runs ────────────────────────────────────────────────────────────────────

class _PauseRequest:
    def __init__(self, stop_after: bool):
        self.stop_after = stop_after
        self.done = threading.Event()
        self.taken = False
        self.path: Optional[str] = None
        self.size = 0
        self.steps = 0
        self.error: Optional[str] = None


class _RunInput:
    """input() of an in-process VM run: the run's pre-recorded stdin, then
    lines from POST /api/run/{id}/stdin. While the program waits for a line
    the event `input` is sent (once per line)."""

    def __init__(self, run: '_Run', initial: bytes):
        self.run = run
        self.cond = threading.Condition()
        self.buf = bytearray(initial)
        self.eof = False

    def feed(self, data: bytes, eof: bool = False):
        with self.cond:
            self.buf.extend(data)
            self.eof = self.eof or eof
            self.cond.notify_all()

    def take(self) -> Tuple[bytes, bool]:
        """What is buffered (a process run takes it over)."""
        with self.cond:
            data, eof = bytes(self.buf), self.eof
            self.buf.clear()
            return data, eof

    def wake(self):
        with self.cond:
            self.cond.notify_all()

    def read_line(self) -> bytes:
        announced = False
        while True:
            with self.cond:
                if self.run.stopping:
                    raise _Stopped()
                i = self.buf.find(b"\n")
                if i >= 0:
                    line = bytes(self.buf[:i])
                    del self.buf[:i + 1]
                    return line
                if self.eof:
                    line = bytes(self.buf)
                    self.buf.clear()
                    return line
                if announced:
                    self.cond.wait(0.5)
                    continue
            announced = True
            self.run.emit('input')          # (outside the condition: emit takes the run's lock)


class _RunKeys:
    """Keys of an in-process VM run: the run's key script, then POST
    /api/run/{id}/key. key_pressed() takes the next key (0: none), wait_key()
    waits for one (NONE entries of the script are skipped)."""

    def __init__(self, run: '_Run', codes: List[int]):
        self.run = run
        self.cond = threading.Condition()
        self.codes = collections.deque(codes)

    def push(self, code: int):
        with self.cond:
            self.codes.append(code)
            self.cond.notify_all()

    def wake(self):
        with self.cond:
            self.cond.notify_all()

    def pressed(self) -> int:
        with self.cond:
            if self.run.stopping:
                raise _Stopped()
            return self.codes.popleft() if self.codes else 0

    def wait(self) -> int:
        with self.cond:
            while True:
                if self.run.stopping:
                    raise _Stopped()
                while self.codes:
                    k = self.codes.popleft()
                    if k:
                        return k
                self.cond.wait(0.5)

    def close(self):
        pass


class _StudioScreen(til.TILScreen):
    """The VM's framebuffer: every screen_show() hands a copy to the run, which
    sends it as a `screen` event (at most FRAME_FPS per second, the newest)."""

    def __init__(self, run: '_Run'):
        super().__init__()
        self.studio_run = run

    def show(self, write):
        self.studio_run.frame(self.w, self.h, bytes(self.fb))


class _StudioVM(til.TILVM):
    """The reference VM of a `vm` run: output, errors, input, keys and sleeps
    go through the run (stoppable at any step or wait)."""

    def __init__(self, module, run: '_Run'):
        super().__init__(module, write=run.vm_write, read_line=run.input.read_line,
                         max_steps=run.max_steps, argv=list(run.args), screen=_StudioScreen(run),
                         keys=run.keys)
        self.studio_run = run

    def report(self, msg, pc):          # (the parameter names of TILVM.report)
        run = self.studio_run
        if run.stopping:
            return
        text = msg + b"\n" + (self.traceback(pc) + b"\n" if pc >= 0 else b"")
        run.out('stderr', til.localize_report_bytes(text, run.lang))

    def call_native(self, nid, args):
        if nid == _SLEEP_NATIVE and args and type(args[0]) is int:     # (errors: TILVM's own checks)
            self.studio_run.sleep(min(max(0, args[0]), til.SLEEP_MAX_MS) / 1000.0)
            return None
        return til.TILVM.call_native(self, nid, args)

    def idle(self, ms):
        self.studio_run.sleep(ms / 1000.0)

    def path_arg(self, who, v):
        path = til.TILVM.path_arg(self, who, v)
        return path if os.path.isabs(path) else os.path.join(self.studio_run.cwd, path)


def _uses_screen(module) -> bool:
    """Does the bytecode call a screen or key builtin (ids GFX_FIRST..GFX_LAST)?"""
    code = module.code
    operands = til.OP_OPERANDS
    native_ops = (til.Op.NATIVE, til.Op.NATIVE_POP)
    ends = sorted(f.offset for f in module.funcs) + [len(code)]
    for start, end in zip(ends, ends[1:]):
        pc = start
        while pc < end:
            op = code[pc]
            if op in native_ops and pc + 1 < len(code) and code[pc + 1] in _GFX_NATIVES:
                return True
            pc += 1 + sum(1 if k in 'bc' else 2 for k in operands.get(op, ''))
    return False


def _kill_process(proc: subprocess.Popen):
    """Kill a run's process and everything it started (its process group)."""
    if proc.poll() is not None and os.name != 'posix':
        return
    try:
        if os.name == 'posix':
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _group_kwargs() -> Dict[str, Any]:
    if os.name == 'posix':
        return {'start_new_session': True}
    return {'creationflags': getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)}


class _StdinWriter(threading.Thread):
    """Feeds a process's stdin from a queue (POST /stdin never blocks)."""

    _QUIT = object()

    def __init__(self, pipe, name: str):
        super().__init__(name=name, daemon=True)
        self.pipe = pipe
        self.q: queue.Queue = queue.Queue()

    def put(self, data: bytes):
        if data:
            self.q.put(data)

    def close_input(self):
        self.q.put(None)

    def quit(self):
        self.q.put(self._QUIT)

    def run(self):
        try:
            while True:
                item = self.q.get()
                if item is None or item is self._QUIT:
                    break
                view = memoryview(item)
                while view:
                    n = self.pipe.write(view)
                    view = view[n or len(view):]
                self.pipe.flush()
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.pipe.close()
            except OSError:
                pass


class _Run:
    """One program run: its thread, its output (coalesced into chunks of at
    most ~50 ms, cut at OUTPUT_LIMIT), screen frames, input and keys."""

    def __init__(self, studio: 'StudioServer', run_id: str, spec: Dict[str, Any]):
        self.studio = studio
        self.id = run_id
        self.backend = spec['backend']
        self.actual = spec['backend']
        self.path: Optional[str] = spec['path']          # absolute path of the program file
        self.rel: Optional[str] = spec['rel']            # its API path
        self.virtual = spec['virtual']
        self.source: str = spec['source']
        self.filename: str = spec['filename']
        self.args: List[str] = spec['args']
        self.lang: str = spec['lang']
        self.max_steps: int = spec['max_steps']
        self.cwd: str = spec['cwd']
        self.lock = threading.RLock()
        self.pending: List[bytes] = []
        self.pending_kind = ''
        self.pending_size = 0
        self.pending_since = 0.0
        self.decoders: Dict[str, Any] = {}
        self.out_total = 0
        self.capped = False
        self.frame_data: Optional[Tuple[int, int, bytes]] = None
        self.next_frame = 0.0
        self.stopping = False
        self.stop_event = threading.Event()
        self.done = threading.Event()
        self.exit_code: Optional[int] = None
        self.vm: Optional[_StudioVM] = None
        self.proc: Optional[subprocess.Popen] = None
        self.writer: Optional[_StdinWriter] = None
        self.pause_req: Optional[_PauseRequest] = None
        # given `stdin`: the program's whole input (end of input after it, as `til run < FILE`
        # and the debugger's launch); else lines from /stdin. Given `keys`: a key script, as
        # `til run --keys` (wait_key() returns ESC when it ends); else keys from /key.
        self.whole_input = spec['stdin'] is not None
        self.input = _RunInput(self, spec['stdin'] or b"")
        if self.whole_input:
            self.input.eof = True
        self.scripted_keys = spec['keys'] is not None
        self.keys = til.KeyScript(spec['keys']) if self.scripted_keys else _RunKeys(self, [])
        self.started = time.monotonic()
        self.finished_at = 0.0
        self.helpers: List[threading.Thread] = []
        self.thread = threading.Thread(target=self._main, name=f"til-run-{run_id}", daemon=True)

    # ── events and output ───────────────────────────────────────────────────
    def _publish(self, kind: str, **fields):
        data = {'runId': self.id, 'kind': kind}
        data.update(fields)
        self.studio.hub.publish('run', data, key=('screen', self.id) if kind == 'screen' else None)

    def emit(self, kind: str, **fields):
        """A run event, after the output written before it."""
        with self.lock:
            self._flush()
            self._publish(kind, **fields)

    def _flush(self, final: bool = False):
        if self.pending:
            data = b"".join(self.pending)
            kind = self.pending_kind
            self.pending, self.pending_size = [], 0
            dec = self.decoders.get(kind)
            if dec is None:
                dec = self.decoders[kind] = codecs.getincrementaldecoder('utf-8')('replace')
            text = dec.decode(data)
            if text:
                self._publish(kind, text=text)
        if final:
            for kind, dec in self.decoders.items():
                text = dec.decode(b"", True)
                if text:
                    self._publish(kind, text=text)

    def out(self, kind: str, data: bytes):
        """Program output (`stdout` / `stderr`)."""
        if not data:
            return
        with self.lock:
            if self.capped or self.done.is_set():
                return
            room = OUTPUT_LIMIT - self.out_total
            cut = len(data) > room
            if cut:
                data = data[:room]
            self.out_total += len(data)
            now = time.monotonic()
            if data:
                if self.pending and self.pending_kind != kind:
                    self._flush()
                if not self.pending:
                    self.pending_since = now
                    self.pending_kind = kind
                self.pending.append(data)
                self.pending_size += len(data)
            if cut:
                self.capped = True
                self._flush(final=True)
                notice = _t(f"[TIL] output limit reached ({OUTPUT_LIMIT // _MB} MB): the rest of the "
                            f"output is not shown", self.lang)
                self._publish('stderr', text="\n" + notice + "\n")
            elif self.pending_size >= 256 * 1024 or now - self.pending_since >= FLUSH_AGE:
                self._flush()

    def note(self, text: str):
        """A message of the studio in the program's error output (never cut)."""
        with self.lock:
            self._flush()
            self._publish('stderr', text=text.rstrip('\n') + '\n')

    def vm_write(self, data: bytes):
        self.out('stdout', data)

    def frame(self, w: int, h: int, fb: bytes):
        with self.lock:
            self.frame_data = (w, h, fb)
            now = time.monotonic()
            if now >= self.next_frame:
                self._send_frame(now)

    def _send_frame(self, now: float):
        w, h, fb = self.frame_data
        self.frame_data = None
        self._flush()
        self.next_frame = now + max(1.0 / FRAME_FPS, len(fb) / FRAME_BYTES_PER_SECOND)
        self._publish('screen', w=w, h=h, rgb=base64.b64encode(fb).decode('ascii'))

    def tick(self, now: float):
        """The pump: output older than FLUSH_AGE, a frame that is due."""
        with self.lock:
            if self.pending and now - self.pending_since >= FLUSH_AGE:
                self._flush()
            if self.frame_data is not None and now >= self.next_frame:
                self._send_frame(now)

    def sleep(self, seconds: float):
        if self.stop_event.wait(min(max(seconds, 0.0), threading.TIMEOUT_MAX)):
            raise _Stopped()

    # ── control (request threads) ───────────────────────────────────────────
    def send_stdin(self, data: bytes, eof: bool, lang: Optional[str]):
        with self.lock:
            if self.done.is_set():
                raise _ApiError(409, _t("the run has ended", lang))
            if self.whole_input:
                raise _ApiError(409, _t("this run was given its whole input (stdin)", lang))
            if self.writer is not None:
                self.writer.put(data)
                if eof:
                    self.writer.close_input()
            else:
                self.input.feed(data, eof)

    def press(self, code: int, lang: Optional[str]):
        with self.lock:
            if self.done.is_set():
                raise _ApiError(409, _t("the run has ended", lang))
            if self.proc is not None:
                raise _ApiError(409, _t("keys go to programs that run on the VM backend", lang))
            if self.scripted_keys:
                raise _ApiError(409, _t("this run plays a key script (keys)", lang))
            self.keys.push(code)

    def stop(self):
        with self.lock:
            if self.done.is_set():
                return
            self.stopping = True
            vm, proc = self.vm, self.proc
        self.stop_event.set()
        self.input.wake()
        if isinstance(self.keys, _RunKeys):
            self.keys.wake()
        if vm is not None:
            vm.max_steps = 1           # the next step ends the program (report() stays quiet)
        if proc is not None:
            _kill_process(proc)

    def snapshot(self, stop_after: bool, lang: Optional[str]) -> Dict[str, Any]:
        """Pause the VM after its current step, save a portable snapshot (.tilp)."""
        with self.lock:
            if self.done.is_set():
                raise _ApiError(409, _t("the run has ended", lang))
            if self.actual != 'vm' or self.vm is None:
                raise _ApiError(409, _t("snapshots need the vm backend", lang))
            if self.pause_req is not None:
                raise _ApiError(409, _t("a snapshot is being saved", lang))
            req = self.pause_req = _PauseRequest(stop_after)
            vm = self.vm
        deadline = time.monotonic() + SNAPSHOT_WAIT
        while not req.done.wait(0.002):
            if self.done.is_set() or time.monotonic() > deadline:
                break
            steps = vm.steps
            if vm.pause_at <= steps:
                vm.pause_at = steps + 1    # the VM checks it at its next step
        with self.lock:
            if not req.done.is_set() and not req.taken:
                self.pause_req = None
                if self.done.is_set():
                    raise _ApiError(409, _t("the run has ended", lang))
                raise _ApiError(409, _t("the program is waiting (input, keys or sleep): take the "
                                        "snapshot while it runs", lang))
        req.done.wait()
        if req.error is not None:
            raise _ApiError(500, _t(f"cannot save the snapshot: {req.error}", lang))
        return {'path': req.path, 'size': req.size, 'steps': req.steps}

    # ── the run's thread ────────────────────────────────────────────────────
    def _main(self):
        code, reason = 1, 'failed'
        try:
            code, reason = self._execute()
        except _Stopped:
            code, reason = 130, 'stopped'
        except Exception as e:      # a bug in the studio: the run fails, the server lives on
            self.studio.log_exception(f"run {self.id}")
            self.note(_t(f"[TIL] internal error: {type(e).__name__}: {e}", self.lang))
            code, reason = 1, 'failed'
        finally:
            if self.stopping and reason != 'paused':
                code, reason = 130, 'stopped'
            self._finish(code, reason)

    def _finish(self, code: int, reason: str):
        try:
            with self.lock:
                delay = self.next_frame - time.monotonic() if self.frame_data is not None else 0.0
            if delay > 0:                   # the last frame keeps to the frame rate too
                time.sleep(min(delay, 1.0 / FRAME_FPS))
            with self.lock:
                self._flush(final=True)
                if self.frame_data is not None:
                    self._send_frame(time.monotonic())
                self.finished_at = time.monotonic()
                self._publish('exit', code=code, reason=reason,
                              seconds=round(self.finished_at - self.started, 3))
        except Exception:
            self.studio.log_exception(f"run {self.id}")
        finally:
            with self.lock:                 # whatever happened: the run no longer counts as active
                self.exit_code = code
                self.finished_at = self.finished_at or time.monotonic()
                self.vm = None
                self.done.set()
            self.studio.run_finished(self)

    def _capture(self):
        sink = (lambda text: self.out('stderr', _plain(text).encode('utf-8', 'replace')))
        return capture_output(sink, sink)

    def _compile(self):
        text = ""
        with self._capture():
            try:
                return til.compile_to_bytecode(self.source, self.filename)
            except SyntaxError as e:
                text = til.localize(f"Syntax Error: {e}", self.lang)
            except til.CompileError as e:
                text = til.localize(str(e), self.lang)
            except RecursionError:
                text = til.localize("Error: the program is nested too deeply to compile", self.lang)
            except (ValueError, OSError) as e:
                text = til.localize(f"Error: {e}", self.lang)
        self.out('stderr', (_plain(text).rstrip('\n') + '\n').encode('utf-8'))
        return None

    def _execute(self) -> Tuple[int, str]:
        self.emit('status', state='starting', backend=self.backend)
        module = self._compile()
        if module is None:
            return 1, 'failed'
        if self.stopping:
            raise _Stopped()
        backend = self.backend
        if backend != 'vm' and _uses_screen(module):
            self.note(_t("[TIL] the program uses the screen: running it on the TIL VM", self.lang))
            backend = 'vm'
        if backend == 'cvm':
            return self._run_cvm(module)
        if backend == 'native':
            return self._run_native(module)
        return self._run_vm(module)

    def _run_vm(self, module) -> Tuple[int, str]:
        vm = _StudioVM(module, self)
        with self.lock:
            self.actual = 'vm'
            self.vm = vm
        if self.stopping:
            raise _Stopped()
        self.emit('status', state='running', backend='vm')
        while True:
            status = vm.run_steps(0)
            if status != vm.YIELD:
                break
            if self._service_pause():
                return 3, 'paused'
        if self.stopping:
            raise _Stopped()
        return (vm.exit_code if vm.exit_code is not None else 1), 'exited'

    def _service_pause(self) -> bool:
        """The VM paused at a step: save the snapshot that was asked for (True:
        end the run now)."""
        with self.lock:
            req = self.pause_req
            if req is None:
                return False                  # a pause nobody waits for any more: go on
            req.taken = True
        rel = None
        try:
            data = self.vm.save_portable()
            req.steps = self.vm.steps
            target, rel = self._snapshot_target()
            with open(target, 'xb') as f:
                f.write(data)
            req.path, req.size = rel, len(data)
        except til.SnapshotError as e:
            req.error = str(e)
        except OSError as e:
            req.error = e.strerror or str(e)
        finally:
            with self.lock:
                self.pause_req = None
            req.done.set()
        if req.path is not None:
            self.emit('status', state='snapshot', path=req.path, steps=req.steps)
            if req.stop_after:
                self.note(til.localize(f"til: paused after {req.steps} steps; saved {req.path} ({req.size} bytes)",
                                       self.lang))
        return req.stop_after and req.path is not None

    def _snapshot_target(self) -> Tuple[str, str]:
        """NAME.tilp next to the program (NAME-2.tilp, ... if taken); for an
        unsaved buffer or an example: in the project root."""
        project = self.studio.project
        if self.path and not self.virtual:
            folder = os.path.dirname(self.path)
            stem = os.path.splitext(os.path.basename(self.path))[0]
        else:
            folder = project.root
            stem = os.path.splitext(os.path.basename(self.path or 'untitled.til'))[0] or 'untitled'
        k = 1
        while True:
            name = f"{stem}.tilp" if k == 1 else f"{stem}-{k}.tilp"
            target = os.path.join(folder, name)
            if not os.path.lexists(target):
                rel = project.rel(target)
                return target, rel if rel is not None else name
            k += 1

    def _run_cvm(self, module) -> Tuple[int, str]:
        tmp = tempfile.mkdtemp(prefix='til-studio-')
        try:
            tbc = os.path.join(tmp, 'program.tbc')
            with open(tbc, 'wb') as f:
                f.write(module.serialize())
            cmd = [sys.executable, _til_script(), 'exec', tbc, '--cvm', '--lang', self.lang]
            if self.max_steps:
                cmd += ['--max-steps', str(self.max_steps)]
            return self._run_process(cmd + ['--'] + self.args, 'cvm')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _run_native(self, module) -> Tuple[int, str]:
        compiler = til.TILCompiler()
        if compiler.find_c_compiler() is None:
            self.note(til.localize("[TIL] No C compiler found - running on the TIL VM", self.lang))
            return self._run_vm(module)
        tmp = tempfile.mkdtemp(prefix='til-studio-')
        try:
            exe = os.path.join(tmp, 'program' + ('.exe' if sys.platform == 'win32' else ''))
            self.emit('status', state='compiling', backend='native')
            text = ""
            with self._capture():
                try:
                    ok = compiler.compile_to_executable(self.source, exe, self.filename, quiet=True)
                except SyntaxError as e:
                    ok, text = None, til.localize(f"Syntax Error: {e}", self.lang)
                except til.CompileError as e:
                    ok, text = None, til.localize(str(e), self.lang)
            if ok is None:
                self.out('stderr', (_plain(text).rstrip('\n') + '\n').encode('utf-8'))
                return 1, 'failed'
            if self.stopping:
                raise _Stopped()
            if not ok:
                self.note(til.localize("[TIL] native backend cannot compile this program; running it on the TIL VM",
                                       self.lang))
                return self._run_vm(module)
            return self._run_process([exe] + self.args, 'native')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _run_process(self, cmd: List[str], backend: str) -> Tuple[int, str]:
        """Run a subprocess: stdout through a pseudo-terminal on POSIX (the C
        stdio of the program is then line-buffered), stderr through a pipe,
        stdin from the run; its own process group (stop kills all of it)."""
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        master = slave = None
        if os.name == 'posix':
            try:
                master, slave = os.openpty()
                import tty
                tty.setraw(slave)           # no echo, no \n -> \r\n
            except (OSError, ImportError):
                for fd in (master, slave):
                    if fd is not None:
                        os.close(fd)
                master = slave = None
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=slave if slave is not None else subprocess.PIPE,
                                    stderr=subprocess.PIPE, cwd=self.cwd, env=env, bufsize=0,
                                    close_fds=True, **_group_kwargs())
        except OSError as e:
            if master is not None:
                os.close(master)
            self.note(_t(f"[TIL] cannot start the program: {e.strerror or e}", self.lang))
            return 1, 'failed'
        finally:
            if slave is not None:
                os.close(slave)
        out_fd = master if master is not None else proc.stdout.fileno()
        err_fd = proc.stderr.fileno()
        writer = _StdinWriter(proc.stdin, f"til-run-{self.id}-stdin")
        with self.lock:
            self.actual = backend
            self.proc = proc
            data, eof = self.input.take()
            writer.put(data)
            if eof:
                writer.close_input()
            self.writer = writer
        writer.start()
        readers = [threading.Thread(target=self._read_fd, args=(out_fd, 'stdout'),
                                    name=f"til-run-{self.id}-out", daemon=True),
                   threading.Thread(target=self._read_fd, args=(err_fd, 'stderr'),
                                    name=f"til-run-{self.id}-err", daemon=True)]
        for t in readers:
            t.start()
        self.helpers = readers + [writer]
        if self.stopping:
            _kill_process(proc)
        else:
            self.emit('status', state='running', backend=backend, pid=proc.pid)
        rc = proc.wait()
        for t in readers:
            t.join(0.5)
        if any(t.is_alive() for t in readers):
            _kill_process(proc)             # what the program left running holds its output open
            for t in readers:
                t.join(5.0)
        writer.quit()
        writer.join(2.0)
        with self.lock:
            self.writer = None
        for fd in ([master] if master is not None else []):
            try:
                os.close(fd)
            except OSError:
                pass
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if self.stopping:
            raise _Stopped()
        return (128 - rc if rc < 0 else rc), 'exited'

    def _read_fd(self, fd: int, kind: str):
        while True:
            try:
                data = os.read(fd, 65536)
            except OSError as e:          # EIO: the pseudo-terminal's other side is closed
                if e.errno == errno.EINTR:
                    continue
                break
            except ValueError:
                break
            if not data:
                break
            self.out(kind, data)


# ── language server and debug adapter bridges ──────────────────────────────

def _file_uri_path(uri: Any) -> Optional[str]:
    if not isinstance(uri, str) or not uri.startswith('file:'):
        return None
    try:
        import urllib.request
        return urllib.request.url2pathname(urllib.parse.urlsplit(uri).path)
    except (ValueError, OSError):
        return None


class _LspBridge:
    """One in-process LspServer (src/tillsp.py) shared by every tab."""

    def __init__(self, studio: 'StudioServer'):
        self.studio = studio
        self.lock = threading.Lock()
        self.server = None
        self.init_result: Optional[Any] = None
        self.init_locale: Any = None
        self.initialized = False
        self.failed_polls = 0
        self.wake = threading.Event()           # a message arrived: the ticker looks again

    def _instance(self, lang: Optional[str]):
        if self.server is None:
            module = self.studio.module('tillsp', lang)
            self.server = module.LspServer(self.studio.root)
        return self.server

    def _publish(self, messages: List[Any]):
        for m in messages:
            if isinstance(m, dict) and 'method' in m:
                self.studio.hub.publish('lsp', m)

    def handle(self, message: Dict[str, Any], lang: Optional[str]) -> List[Any]:
        method = message.get('method')
        mid = message.get('id')
        if method == 'shutdown' and mid is not None:     # the server is shared: a tab cannot end it
            return [{'jsonrpc': '2.0', 'id': mid, 'result': None}]
        if method == 'exit':
            return []
        if method in ('textDocument/didOpen', 'textDocument/didClose'):
            params = message.get('params') if isinstance(message.get('params'), dict) else {}
            doc = params.get('textDocument') if isinstance(params.get('textDocument'), dict) else {}
            path = _file_uri_path(doc.get('uri'))
            if path is not None:
                self.studio.watcher.document(path, method.endswith('didOpen'))
        with self.lock:
            server = self._instance(lang)
            if method == 'initialized' and self.initialized:
                return []
            if method == 'initialize' and self.init_result is not None and mid is not None:
                params = message.get('params') if isinstance(message.get('params'), dict) else {}
                if params.get('locale') == self.init_locale:
                    return [{'jsonrpc': '2.0', 'id': mid, 'result': self.init_result}]
            try:
                out = list(server.handle(message) or [])
            except Exception as e:
                self.studio.log_exception("language server")
                if mid is None or method is None:
                    out = [{'jsonrpc': '2.0', 'method': 'window/logMessage',
                            'params': {'type': 1, 'message': f"{type(e).__name__}: {e}"}}]
                else:
                    out = [{'jsonrpc': '2.0', 'id': mid,
                            'error': {'code': -32603, 'message': f"{type(e).__name__}: {e}"}}]
            if method == 'initialized':
                self.initialized = True
            if method == 'initialize' and mid is not None:
                params = message.get('params') if isinstance(message.get('params'), dict) else {}
                for i, m in enumerate(out):
                    if isinstance(m, dict) and m.get('id') == mid and 'method' not in m:
                        if 'result' in m and m.get('error') is None:
                            if self.init_result is None or params.get('locale') != self.init_locale:
                                self.init_result, self.init_locale = m['result'], params.get('locale')
                        elif self.init_result is not None:    # it refuses a second initialize
                            out[i] = {'jsonrpc': '2.0', 'id': mid, 'result': self.init_result}
        self.wake.set()
        self._publish(out)
        return out

    def delay(self) -> Optional[float]:
        """Seconds until poll() is due: the server's next_due() (tillsp), else
        LSP_TICK_SECONDS; None = nothing pending (wait for the next message)."""
        server = self.server
        if server is None:
            return None
        next_due = getattr(server, 'next_due', None)
        if next_due is None:
            return LSP_TICK_SECONDS
        try:
            due = next_due()
        except Exception:
            return LSP_TICK_SECONDS
        return None if due is None else max(0.005, float(due))

    def poll(self):
        with self.lock:
            if self.server is None:
                return
            try:
                out = list(self.server.poll() or [])
                self.failed_polls = 0
            except Exception:
                self.failed_polls += 1
                if self.failed_polls == 1:
                    self.studio.log_exception("language server poll")
                return
        self._publish(out)


class _DapBridge:
    """Debug sessions (src/tildap.py DapSession) by id; events -> `dap` events."""

    def __init__(self, studio: 'StudioServer'):
        self.studio = studio
        self.lock = threading.Lock()
        self.sessions: Dict[str, Any] = {}
        self.used: Dict[str, float] = {}

    def start(self, lang: Optional[str]) -> str:
        module = self.studio.module('tildap', lang)
        sid = secrets.token_hex(4)
        hub = self.studio.hub

        def emit(payload):
            hub.publish('dap', {'sessionId': sid, 'message': payload})
        session = module.DapSession(emit, self.studio.root)
        evicted = []
        with self.lock:
            self.sessions[sid] = session
            self.used[sid] = time.monotonic()
            while len(self.sessions) > MAX_DAP_SESSIONS:
                oldest = min(self.used, key=self.used.get)
                evicted.append(self.sessions.pop(oldest))
                del self.used[oldest]
        for s in evicted:
            self._close(s)
        return sid

    def handle(self, sid: str, message: Dict[str, Any], lang: Optional[str],
               reply: Callable[[List[Any]], None]):
        """One DAP message: `reply` gets the responses. With tildap's hold_events(),
        the events the message causes go out after the reply (as over stdio)."""
        with self.lock:
            session = self.sessions.get(sid)
            if session is not None:
                self.used[sid] = time.monotonic()
        if session is None:
            raise _ApiError(404, _t(f"no such debug session: {sid}", lang))
        hold = getattr(session, 'hold_events', None)
        with (hold() if callable(hold) else contextlib.nullcontext()):
            try:
                out = list(session.handle(message) or [])
            except Exception as e:
                self.studio.log_exception("debug adapter")
                out = [{'type': 'response', 'request_seq': message.get('seq', 0), 'success': False,
                        'command': message.get('command', ''), 'message': f"{type(e).__name__}: {e}"}]
            reply(out)
        if message.get('type') == 'request' and message.get('command') == 'disconnect':
            with self.lock:
                self.sessions.pop(sid, None)
                self.used.pop(sid, None)
            self._close(session)

    def _close(self, session):
        try:
            session.close()
        except Exception:
            self.studio.log_exception("debug adapter close")

    def close_all(self):
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
            self.used.clear()
        for s in sessions:
            self._close(s)


# ── file watching ───────────────────────────────────────────────────────────

class _Watcher:
    """Polls the tree and the open files every WATCH_SECONDS; changes -> `fs`."""

    def __init__(self, studio: 'StudioServer'):
        self.studio = studio
        self.lock = threading.Lock()
        self.snapshot: Optional[Dict[str, tuple]] = None
        self.documents: Dict[str, int] = {}             # open in the editor (LSP), refcount
        self.read: 'collections.OrderedDict[str, None]' = collections.OrderedDict()   # GET /api/file
        self.expected: Dict[str, tuple] = {}            # written by the studio itself: no echo

    def document(self, abs_path: str, opened: bool):
        rel = self.studio.project.rel(abs_path)
        if not rel:
            return
        with self.lock:
            n = self.documents.get(rel, 0) + (1 if opened else -1)
            if n > 0:
                self.documents[rel] = n
            else:
                self.documents.pop(rel, None)

    def opened(self, rel: str):
        with self.lock:
            self.read[rel] = None
            self.read.move_to_end(rel)
            while len(self.read) > 256:
                self.read.popitem(last=False)

    def wrote(self, rel: str):
        """The studio wrote this file: the next poll does not report it."""
        try:
            st = os.stat(os.path.join(self.studio.project.root, *rel.split('/')))
        except OSError:
            return
        with self.lock:
            self.expected[rel] = ('f', st.st_mtime_ns, st.st_size)

    def scan(self) -> Dict[str, tuple]:
        root = self.studio.project.root
        out: Dict[str, tuple] = {}
        stack = ['']
        count = 0
        while stack and count < WATCH_LIMIT:
            rel_dir = stack.pop()
            try:
                with os.scandir(os.path.join(root, *rel_dir.split('/')) if rel_dir else root) as it:
                    entries = list(it)
            except OSError:
                continue
            for e in entries:
                name = e.name
                if name in HIDDEN_NAMES or name.endswith(TEMP_SUFFIX):
                    continue
                rel = f"{rel_dir}/{name}" if rel_dir else name
                count += 1
                try:
                    if e.is_dir(follow_symlinks=False):
                        out[rel] = ('d',)
                        if name not in COLLAPSED_NAMES:
                            stack.append(rel)
                        continue
                    st = e.stat()                  # a link: what it points to
                except OSError:
                    continue
                out[rel] = ('d',) if _stat.S_ISDIR(st.st_mode) else ('f', st.st_mtime_ns, st.st_size)
        with self.lock:
            extra = [r for r in list(self.documents) + list(self.read) if r not in out]
        for rel in extra:
            try:
                st = os.stat(os.path.join(root, *rel.split('/')))
            except OSError:
                continue
            out[rel] = ('d',) if os.path.isdir(os.path.join(root, *rel.split('/'))) else \
                ('f', st.st_mtime_ns, st.st_size)
        return out

    def poll(self):
        current = self.scan()
        with self.lock:
            previous = self.snapshot
            self.snapshot = current
            if previous is None:
                return
            changed = set()
            for rel, sig in current.items():
                old = previous.get(rel)
                if old != sig:
                    if self.expected.get(rel) == sig:
                        continue
                    changed.add(rel)
            for rel in previous:
                if rel not in current:
                    changed.add(rel)
            for rel in list(self.expected):
                if current.get(rel) == self.expected[rel] or rel not in current:
                    del self.expected[rel]
        if changed:
            self.studio.hub.publish('fs', {'paths': sorted(changed)})

    def loop(self, stop: threading.Event):
        while True:
            try:
                self.poll()
            except Exception:
                self.studio.log_exception("file watcher")
            if stop.wait(WATCH_SECONDS):
                return


# ── the language tables (/api/info) ─────────────────────────────────────────

def _builtin_tables() -> Dict[str, Any]:
    """The tables of ide/studio/gen_tables.py, for installations without it."""
    tt = lambda t: t.name                                       # noqa: E731
    keywords = {'en': {w: tt(t) for w, t in til.ENGLISH_KEYWORDS.items()}}
    aliases = {}
    for code, (table, alias) in til.LANGUAGES.items():
        if code != 'en':
            keywords[code] = {w: tt(t) for w, t in table.items()}
            aliases[code] = dict(alias)
    types = list(til.TYPE_MAP.keys())
    for extra in ('Option', 'Result', 'Some', 'None', 'Ok', 'Err', 'Self'):
        if extra not in types:
            types.append(extra)
    return {
        'version': til.TIL_VERSION,
        'languages': list(til.LANGUAGES.keys()),
        'default_languages': list(getattr(til, 'DEFAULT_LANGUAGES', ('en',))),
        'keywords': keywords,
        'contextual': {c: {w: tt(t) for w, t in tbl.items()}
                       for c, tbl in getattr(til, 'CONTEXTUAL_KEYWORDS', {}).items()},
        'aliases': aliases,
        'true_words': sorted(getattr(til, 'TRUE_WORDS', ())),
        'natives': list(til.NATIVES),
        'native_arity': {n: [lo, hi] for n, (lo, hi) in til.NATIVE_ARITY.items()},
        'methods': list(til.METHODS),
        'types': types,
        'attributes': ['pure', 'effects', 'requires', 'ensures', 'cap', 'level', 'energy', 'lang'],
        'energy_modes': list(getattr(til, 'ENERGY_MODES', {}).keys()),
        'energy_levels': list(getattr(til, 'FUNCTION_ENERGY_LEVELS', ())),
        'key_names': dict(til.KEY_NAMES),
    }


def studio_tables() -> Dict[str, Any]:
    """The dict of ide/studio/tables.json: built by ide/studio/gen_tables.py
    (build_tables()) when the sources are installed, else read from a
    tables.json next to the frontend, else built here."""
    for home in til.til_home_dirs():
        gen = os.path.join(home, 'ide', 'studio', 'gen_tables.py')
        if os.path.isfile(gen):
            try:
                spec = importlib.util.spec_from_file_location('til_studio_gen_tables', gen)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.build_tables()
            except Exception:
                pass
        for path in (os.path.join(home, 'ide', 'studio', 'tables.json'), os.path.join(home, 'ide', 'tables.json')):
            if os.path.isfile(path):
                try:
                    with open(path, encoding='utf-8') as f:
                        return json.load(f)
                except (OSError, ValueError):
                    pass
    return _builtin_tables()


def find_frontend() -> Optional[str]:
    """ide/studio.html next to the installed files (repo layout or ~/.til/ide/)."""
    for home in til.til_home_dirs():
        path = os.path.join(home, 'ide', 'studio.html')
        if os.path.isfile(path):
            return path
    return None


_FALLBACK_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TIL Studio</title>
<style>
:root { color-scheme: dark light; }
body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0b0d12; color: #e6e8ee;
       font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 36rem; padding: 2rem; border: 1px solid #232836; border-radius: 6px; background: #11141b; }
h1 { margin: 0 0 .5rem; font-size: 1.25rem; color: #00d4ff; }
code { font-family: "JetBrains Mono", Menlo, Consolas, monospace; color: #4ade80; }
</style></head>
<body><main><h1>TIL Studio</h1>
<p>%(message)s</p>
<p><code>%(root)s</code></p>
</main></body></html>
"""


# ── the server ──────────────────────────────────────────────────────────────

class _HTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != 'nt'       # Windows: SO_REUSEADDR lets two servers share a port
    request_queue_size = 64

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)  # (no getfqdn(): a reverse DNS lookup can be slow)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = host, port

    def process_request_thread(self, request, client_address):
        studio = self.studio
        with studio.conn_lock:
            studio.connections[threading.current_thread()] = request
        try:
            super().process_request_thread(request, client_address)
        finally:
            with studio.conn_lock:
                studio.connections.pop(threading.current_thread(), None)


class StudioServer:
    """The TIL Studio server for one project folder.

        studio = StudioServer('/path/to/project')    # binds 127.0.0.1, port 0 = a free port
        studio.start()                               # in background threads (tests), or
        studio.serve_forever()                       # in this thread (til ide)
        print(studio.url)                            # http://127.0.0.1:PORT/?token=...
        studio.stop()

    `frontend`: path of the app page; None = ide/studio.html of the installation,
    False = the built-in page. `modules`: replacements for the tool modules
    ({'tillsp': ..., 'tildap': ...}; tests)."""

    def __init__(self, root: str, port: int = 0, lang: Optional[str] = None, frontend: Any = None,
                 modules: Optional[Dict[str, Any]] = None, log=None):
        if not os.path.isdir(root):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), root)
        self.project = _Project(root)
        self.root = self.project.root
        self.lang = til.message_language_code(lang) if lang else None
        self.token = secrets.token_urlsafe(24)
        self.frontend = frontend
        self.modules = dict(modules or {})
        self.log_stream = log if log is not None else sys.stderr
        self.hub = EventHub()
        self.runs: Dict[str, _Run] = {}
        self.runs_lock = threading.Lock()
        self.pump_wake = threading.Event()
        self.lsp = _LspBridge(self)
        self.dap = _DapBridge(self)
        self.watcher = _Watcher(self)
        self.stopping = threading.Event()
        self.conn_lock = threading.Lock()
        self.connections: Dict[threading.Thread, Any] = {}
        self.workers_lock = threading.Lock()
        self.workers: set = set()
        self.pkg_lock = threading.Lock()
        self.info_cache: Optional[Dict[str, Any]] = None
        self.info_lock = threading.Lock()
        self.threads: List[threading.Thread] = []
        self.serve_lock = threading.Lock()
        self.served = threading.Event()
        self.httpd = _HTTPServer(('127.0.0.1', port), _Handler)
        self.httpd.studio = self
        self.port = self.httpd.server_address[1]

    # ── lifecycle ───────────────────────────────────────────────────────────
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    @property
    def url(self) -> str:
        return f"{self.base_url}?token={self.token}"

    @property
    def cookie_name(self) -> str:
        return f"til_studio_{self.port}"     # cookies are not per port: one name per studio

    def _start_background(self):
        for name, target in (('til-studio-pump', self._pump_loop), ('til-studio-lsp', self._lsp_loop),
                             ('til-studio-watch', lambda: self.watcher.loop(self.stopping))):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self.threads.append(t)

    def start(self) -> 'StudioServer':
        """Serve in background threads."""
        self._start_background()
        t = threading.Thread(target=self._serve, name='til-studio-http', daemon=True)
        t.start()
        self.threads.append(t)
        return self

    def serve_forever(self):
        """Serve in this thread until stop() (or KeyboardInterrupt)."""
        self._start_background()
        self._serve()

    def _serve(self):
        with self.serve_lock:
            if self.stopping.is_set():
                return
            self.served.set()
        self.httpd.serve_forever(poll_interval=0.1)

    def stop(self, timeout: float = 5.0):
        """Stop everything: runs (their processes killed), debug sessions, tool
        workers, event streams, the HTTP server and the background threads."""
        with self.serve_lock:
            if self.stopping.is_set():
                return
            self.stopping.set()
            served = self.served.is_set()
        self.pump_wake.set()
        self.lsp.wake.set()
        with self.runs_lock:
            runs = list(self.runs.values())
        for r in runs:
            r.stop()
        self.dap.close_all()
        with self.workers_lock:
            workers = list(self.workers)
        for p in workers:
            _kill_process(p)
        self.hub.close_all()
        if served:
            self.httpd.shutdown()
        self.httpd.server_close()
        with self.conn_lock:
            conns = list(self.connections.items())
        for _t_, sock in conns:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        deadline = time.monotonic() + timeout
        for r in runs:
            r.thread.join(max(0.0, deadline - time.monotonic()))
            for h in r.helpers:
                h.join(max(0.0, deadline - time.monotonic()))
        for t in self.threads:
            if t is not threading.current_thread():
                t.join(max(0.0, deadline - time.monotonic()))
        for t, _sock in conns:
            if t is not threading.current_thread():
                t.join(max(0.0, deadline - time.monotonic()))

    def log(self, text: str):
        stream = self.log_stream
        if stream is None:
            return
        try:
            stream.write(text if text.endswith('\n') else text + '\n')
            stream.flush()
        except (OSError, ValueError):
            pass

    def log_exception(self, where: str):
        self.log(f"til ide: {where}: " + traceback.format_exc())

    def module(self, name: str, lang: Optional[str]):
        """A tool module (tillsp, tildap, tilfit, tilpkg, ...)."""
        if name in self.modules:
            if self.modules[name] is None:
                raise _ApiError(503, _t(f"not available in this installation ({name}.py is missing)", lang))
            return self.modules[name]
        try:
            return til.feature_module(name)
        except ImportError as e:
            if getattr(e, 'name', None) != name:
                raise
            raise _ApiError(503, _t(f"not available in this installation ({name}.py is missing)", lang))

    def has_module(self, name: str) -> bool:
        if name in self.modules:
            return self.modules[name] is not None
        return os.path.isfile(os.path.join(os.path.dirname(_til_script()), name + '.py')) or _find_spec(name)

    def app_page(self, lang: str) -> bytes:
        """ide/studio.html (read at every request: a rebuilt frontend needs no restart),
        else a page saying that the frontend is not built."""
        path = self.frontend if self.frontend not in (None, False, '') else None
        if self.frontend is None:
            path = find_frontend()
        if path:
            try:
                with open(path, 'rb') as f:
                    return f.read()
            except OSError:
                pass
        text = _t("The TIL Studio frontend is not built: ide/studio.html is missing. Build it with "
                  "`npm run build` in ide/studio/, or install it into ~/.til/ide/.", lang)
        return (_FALLBACK_PAGE % {'message': _html(text), 'root': _html(self.root)}).encode('utf-8')

    # ── background loops ────────────────────────────────────────────────────
    def _pump_loop(self):
        while not self.stopping.is_set():
            with self.runs_lock:
                active = [r for r in self.runs.values() if not r.done.is_set()]
            if not active:
                self.pump_wake.wait(1.0)
                self.pump_wake.clear()
                continue
            now = time.monotonic()
            for r in active:
                try:
                    r.tick(now)
                except Exception:
                    self.log_exception("output pump")
            self.stopping.wait(PUMP_SECONDS)

    def _lsp_loop(self):
        """Debounced diagnostics: poll() when the language server says they are due."""
        lsp = self.lsp
        while not self.stopping.is_set():
            delay = lsp.delay()
            lsp.wake.wait(1.0 if delay is None else min(delay, 1.0))
            lsp.wake.clear()
            if self.stopping.is_set():
                break
            lsp.poll()

    # ── runs ────────────────────────────────────────────────────────────────
    def run_finished(self, run: _Run):
        with self.runs_lock:
            done = sorted((r for r in self.runs.values() if r.done.is_set()), key=lambda r: r.finished_at)
            for r in done[:-FINISHED_RUNS_KEPT]:
                self.runs.pop(r.id, None)

    def get_run(self, run_id: str, lang: Optional[str]) -> _Run:
        with self.runs_lock:
            run = self.runs.get(run_id)
        if run is None:
            raise _ApiError(404, _t(f"no such run: {run_id}", lang))
        return run

    # ── API: info, files, search ────────────────────────────────────────────
    def api_info(self, lang: Optional[str]) -> Dict[str, Any]:
        with self.info_lock:
            if self.info_cache is None:
                devices = []
                try:
                    fit = self.module('tilfit', lang)
                    devices = [{'name': d.name, 'title': d.title} for d in fit.DEVICES.values()]
                except _ApiError:
                    pass
                cc = til.TILCompiler().find_c_compiler() is not None
                self.info_cache = {
                    'version': til.TIL_VERSION,
                    'capabilities': {
                        'native': cc,
                        'cvm': cc and til.find_vm_sources() is not None,
                        'z3': _find_spec('z3'),
                        'debugger': self.has_module('tildap'),
                        'pkg': self.has_module('tilpkg'),
                        'lsp': self.has_module('tillsp'),
                        'prover': self.has_module('tilprove'),
                        'fit': self.has_module('tilfit'),
                    },
                    'devices': devices,
                    'tables': studio_tables(),
                }
            info = dict(self.info_cache)
        info.update({
            'root': self.root,
            'rootName': os.path.basename(self.root.rstrip(os.sep)) or self.root,
            'rootUri': _path_uri(self.root),
            'platform': sys.platform,
            'lang': self.lang,
            'examples': self._examples(),
            'limits': {'runs': MAX_RUNS, 'output': OUTPUT_LIMIT, 'search': SEARCH_LIMIT},
        })
        return info

    def _examples(self) -> List[Dict[str, str]]:
        base = self.project.examples
        if base is None:
            return []
        found = []
        for folder, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in HIDDEN_NAMES and not d.startswith('.'))
            for name in sorted(files):
                if name.endswith('.til'):
                    abs_path = os.path.join(folder, name)
                    rel = os.path.relpath(abs_path, base).replace(os.sep, '/')
                    in_root = self.project.rel(abs_path)
                    found.append({'name': rel[:-4], 'path': in_root if in_root else f"{EXAMPLES}/{rel}"})
                    if len(found) >= 200:
                        return found
        return found

    def api_tree(self, path: Any, lang: Optional[str]) -> Dict[str, Any]:
        abs_dir, rel_dir, virtual = self.project.resolve(path, lang)
        if not os.path.exists(abs_dir):
            raise _ApiError(404, _t(f"no such file or folder: {rel_dir}", lang))
        if not os.path.isdir(abs_dir):
            raise _ApiError(400, _t(f"not a folder: {rel_dir}", lang))
        base = self.project.examples if virtual else self.root
        entries = []
        truncated = False
        try:
            with os.scandir(abs_dir) as it:
                items = sorted(it, key=lambda e: e.name)
        except OSError as e:
            raise _ApiError(403, _t(f"cannot read {rel_dir}: {e.strerror or e}", lang))
        for e in items:
            if e.name in HIDDEN_NAMES or e.name.endswith(TEMP_SUFFIX):
                continue
            try:
                if e.is_symlink() and not _inside(os.path.realpath(e.path), base):
                    continue                    # it points outside the project: not shown
                st = e.stat()
                is_dir = e.is_dir()
            except OSError:
                continue
            if not is_dir and not _stat.S_ISREG(st.st_mode):
                continue                        # FIFOs, sockets, devices: nothing an editor opens
            if len(entries) >= TREE_LIMIT:
                truncated = True
                break
            entry = {'name': e.name, 'path': f"{rel_dir}/{e.name}" if rel_dir else e.name,
                     'type': 'dir' if is_dir else 'file', 'size': 0 if is_dir else st.st_size,
                     'mtime': st.st_mtime}
            if is_dir and e.name in COLLAPSED_NAMES:
                entry['collapsed'] = True
            if e.is_symlink():
                entry['link'] = True
            if virtual:
                entry['readonly'] = True
            entries.append(entry)
        entries.sort(key=_entry_sort_key)
        out: Dict[str, Any] = {'path': rel_dir, 'entries': entries}
        if truncated:
            out['truncated'] = True
        return out

    def api_file_get(self, path: Any, lang: Optional[str]) -> Dict[str, Any]:
        abs_path, rel, virtual = self.project.resolve(path, lang, allow_root=False)
        try:
            st = os.stat(abs_path)
        except OSError:
            raise _ApiError(404, _t(f"no such file or folder: {rel}", lang))
        if os.path.isdir(abs_path):
            raise _ApiError(400, _t(f"not a file: {rel}", lang))
        if not _stat.S_ISREG(st.st_mode):
            raise _ApiError(415, _t(f"not a text file: {rel}", lang))     # a FIFO would block the read
        if st.st_size > OPEN_LIMIT:
            raise _ApiError(413, _t(f"the file is too large to open ({st.st_size // _MB} MB): {rel}", lang))
        try:
            with open(abs_path, 'rb') as f:
                data = f.read(OPEN_LIMIT + 1)
                st = os.fstat(f.fileno())
        except OSError as e:
            raise _ApiError(403, _t(f"cannot read {rel}: {e.strerror or e}", lang))
        if b'\x00' in data:
            raise _ApiError(415, _t(f"not a text file: {rel}", lang))
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            raise _ApiError(415, _t(f"not a UTF-8 text file: {rel}", lang))
        if not virtual:
            self.watcher.opened(rel)
        out = {'path': rel, 'text': text, 'mtime': st.st_mtime, 'size': st.st_size}
        if virtual:
            out['readonly'] = True
        return out

    def api_file_put(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        abs_path, rel, _v = self.project.resolve(body.get('path'), lang, write=True, allow_root=False)
        text = body.get('text')
        if not isinstance(text, str):
            raise _ApiError(400, _t("missing or invalid field 'text'", lang))
        expect = body.get('mtime')
        if expect is not None and (isinstance(expect, bool) or not isinstance(expect, (int, float))):
            raise _ApiError(400, _t("missing or invalid field 'mtime'", lang))
        if os.path.isdir(abs_path):
            raise _ApiError(400, _t(f"not a file: {rel}", lang))
        mode = None
        try:
            st = os.stat(abs_path)
            mode = st.st_mode & 0o7777
            if expect is not None and abs(st.st_mtime - float(expect)) > MTIME_TOLERANCE:
                raise _ApiError(409, _t(f"the file changed on disk: {rel}", lang))
        except FileNotFoundError:
            if expect is not None:
                raise _ApiError(409, _t(f"the file changed on disk: {rel}", lang))
        folder = os.path.dirname(abs_path)
        if not os.path.isdir(folder):
            raise _ApiError(404, _t(f"no such file or folder: {rel}", lang))
        data = text.encode('utf-8', 'replace')
        try:
            _atomic_write(abs_path, data, mode)
            mtime = os.stat(abs_path).st_mtime
        except OSError as e:
            raise _ApiError(403, _t(f"cannot write {rel}: {e.strerror or e}", lang))
        self.watcher.wrote(rel)
        return {'path': rel, 'mtime': mtime, 'size': len(data)}

    def api_file_create(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        abs_path, rel, _v = self.project.resolve(body.get('path'), lang, write=True, follow=False,
                                                 allow_root=False)
        kind = body.get('type', 'file')
        if kind not in ('file', 'dir'):
            raise _ApiError(400, _t("missing or invalid field 'type'", lang))
        text = body.get('text', '')
        if not isinstance(text, str):
            raise _ApiError(400, _t("missing or invalid field 'text'", lang))
        if os.path.lexists(abs_path):
            raise _ApiError(409, _t(f"already exists: {rel}", lang))
        try:
            if kind == 'dir':
                os.makedirs(abs_path)
            else:
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                self.project.resolve(rel, lang, write=True)          # the folders made are inside the root
                with open(abs_path, 'xb') as f:
                    f.write(text.encode('utf-8', 'replace'))
        except FileExistsError:
            raise _ApiError(409, _t(f"already exists: {rel}", lang))
        except OSError as e:
            raise _ApiError(403, _t(f"cannot write {rel}: {e.strerror or e}", lang))
        out: Dict[str, Any] = {'path': rel}
        if kind == 'file':
            out['mtime'] = os.stat(abs_path).st_mtime
        return out

    def api_file_rename(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        src, rel, _v = self.project.resolve(body.get('path'), lang, write=True, follow=False, allow_root=False)
        to = body.get('to')
        if not isinstance(to, str):
            raise _ApiError(400, _t("missing or invalid field 'to'", lang))
        dst, to_rel, _v2 = self.project.resolve(to, lang, write=True, follow=False, allow_root=False)
        if not os.path.lexists(src):
            raise _ApiError(404, _t(f"no such file or folder: {rel}", lang))
        if os.path.lexists(dst) and os.path.normcase(src) != os.path.normcase(dst):
            raise _ApiError(409, _t(f"already exists: {to_rel}", lang))
        if _inside(dst, src) and os.path.normcase(dst) != os.path.normcase(src):
            raise _ApiError(400, _t(f"cannot move a folder into itself: {to_rel}", lang))
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            self.project.resolve(posix_dirname(to_rel), lang, write=True)
            os.rename(src, dst)
        except _ApiError:
            raise
        except OSError as e:
            raise _ApiError(403, _t(f"cannot rename {rel}: {e.strerror or e}", lang))
        return {'path': to_rel}

    def api_file_delete(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        target, rel, _v = self.project.resolve(body.get('path'), lang, write=True, follow=False, allow_root=False)
        if not os.path.lexists(target):
            raise _ApiError(404, _t(f"no such file or folder: {rel}", lang))
        try:
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)
        except OSError as e:
            raise _ApiError(403, _t(f"cannot delete {rel}: {e.strerror or e}", lang))
        return {}

    def api_search(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        query = body.get('query')
        if not isinstance(query, str):
            raise _ApiError(400, _t("missing or invalid field 'query'", lang))
        globs_spec = body.get('glob') or ''
        if not isinstance(globs_spec, str):
            raise _ApiError(400, _t("missing or invalid field 'glob'", lang))
        if not query:
            return {'matches': [], 'truncated': False}
        flags = 0 if body.get('case') else re.IGNORECASE
        try:
            pattern = re.compile(query if body.get('regex') else re.escape(query), flags)
        except (re.error, RecursionError, OverflowError) as e:
            raise _ApiError(400, _t(f"invalid regular expression: {e}", lang))
        globs = [g.strip() for g in globs_spec.split(',') if g.strip()]
        matches: List[Dict[str, Any]] = []
        deadline = time.monotonic() + SEARCH_SECONDS
        truncated = False
        for rel, abs_path in self._walk_files():
            if globs and not any(fnmatch.fnmatch(rel if '/' in g else rel.rsplit('/', 1)[-1], g) for g in globs):
                continue
            try:
                if os.path.getsize(abs_path) > SEARCH_MAX_FILE:
                    continue
                with open(abs_path, 'rb') as f:
                    data = f.read()
            except OSError:
                continue
            if b'\x00' in data[:8192]:
                continue
            text = data.decode('utf-8', 'replace')
            for lineno, line in enumerate(text.split('\n'), 1):
                line = line.rstrip('\r')
                for m in pattern.finditer(line):
                    if m.end() == m.start():
                        continue
                    matches.append({'path': rel, 'line': lineno, 'col': _utf16_len(line[:m.start()]) + 1,
                                    'length': _utf16_len(m.group(0)), 'text': line[:1000]})
                    if len(matches) >= SEARCH_LIMIT:
                        return {'matches': matches, 'truncated': True}
                if time.monotonic() > deadline:
                    truncated = True
                    break
            if truncated:
                break
        return {'matches': matches, 'truncated': truncated}

    def _walk_files(self):
        root = self.root
        stack = ['']
        while stack:
            rel_dir = stack.pop()
            folder = os.path.join(root, *rel_dir.split('/')) if rel_dir else root
            try:
                with os.scandir(folder) as it:
                    entries = sorted(it, key=lambda e: e.name)
            except OSError:
                continue
            subdirs = []
            for e in entries:
                if e.name in HIDDEN_NAMES or e.name.endswith(TEMP_SUFFIX):
                    continue
                rel = f"{rel_dir}/{e.name}" if rel_dir else e.name
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.name not in COLLAPSED_NAMES:
                            subdirs.append(rel)
                        continue
                    if e.is_symlink() and not _inside(os.path.realpath(e.path), root):
                        continue
                    if e.is_file():
                        yield rel, e.path
                except OSError:
                    continue
            stack.extend(reversed(subdirs))

    # ── API: runs ───────────────────────────────────────────────────────────
    def _program(self, body: Dict[str, Any], lang: Optional[str], need_path: bool = False):
        """(absolute path or None, API path or None, read-only, source, filename) of
        a request's program: `source` (an unsaved buffer) and/or `path`."""
        path, source = body.get('path'), body.get('source')
        if source is not None and not isinstance(source, str):
            raise _ApiError(400, _t("missing or invalid field 'source'", lang))
        if path is None and (source is None or need_path):
            raise _ApiError(400, _t("missing or invalid field 'path'", lang))
        abs_path = rel = None
        virtual = False
        if path is not None:
            abs_path, rel, virtual = self.project.resolve(path, lang, allow_root=False)
            if source is None or need_path:
                if not os.path.isfile(abs_path):
                    raise _ApiError(404, _t(f"no such file or folder: {rel}", lang))
            if source is None:
                try:
                    with open(abs_path, 'r', encoding='utf-8', newline='') as f:
                        source = f.read()
                except UnicodeDecodeError:
                    raise _ApiError(415, _t(f"not a UTF-8 text file: {rel}", lang))
                except OSError as e:
                    raise _ApiError(403, _t(f"cannot read {rel}: {e.strerror or e}", lang))
        filename = abs_path or os.path.join(self.root, 'untitled.til')
        return abs_path, rel, virtual, source, filename

    def api_run(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        backend = body.get('backend') or 'vm'
        if backend not in BACKENDS:
            raise _ApiError(400, _t(f"unknown backend '{backend}' (use native, vm or cvm)", lang))
        abs_path, rel, virtual, source, filename = self._program(body, lang)
        args = body.get('args') or []
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise _ApiError(400, _t("missing or invalid field 'args'", lang))
        stdin = body.get('stdin')
        if stdin is not None and not isinstance(stdin, str):
            raise _ApiError(400, _t("missing or invalid field 'stdin'", lang))
        max_steps = body.get('maxSteps') or 0
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise _ApiError(400, _t("missing or invalid field 'maxSteps'", lang))
        keys = body.get('keys')
        codes: Optional[List[int]] = None
        if keys is not None:
            try:
                if isinstance(keys, str):
                    codes = til.parse_key_script(keys)
                elif isinstance(keys, list):
                    codes = [_key_code(k) for k in keys]
                else:
                    raise ValueError(keys)
            except (ValueError, TypeError):
                raise _ApiError(400, _t("missing or invalid field 'keys'", lang))
        cwd = self.root
        if body.get('cwd') is not None:
            cwd, cwd_rel, cwd_virtual = self.project.resolve(body.get('cwd'), lang)
            if cwd_virtual or not os.path.isdir(cwd):
                raise _ApiError(400, _t("missing or invalid field 'cwd'", lang))
        run_lang = (til.message_language_code(body.get('lang')) if isinstance(body.get('lang'), str) else None) \
            or self.lang or til.pragma_message_language(source) or lang or 'en'
        spec = {'backend': backend, 'path': abs_path, 'rel': rel, 'virtual': virtual, 'source': source,
                'filename': filename, 'args': args,
                'stdin': None if stdin is None else stdin.encode('utf-8', 'replace'),
                'lang': run_lang, 'max_steps': max_steps, 'keys': codes, 'cwd': cwd}
        with self.runs_lock:
            if self.stopping.is_set():
                raise _ApiError(503, _t("TIL Studio is stopping", lang))
            if sum(1 for r in self.runs.values() if not r.done.is_set()) >= MAX_RUNS:
                raise _ApiError(429, _t(f"too many runs at once (at most {MAX_RUNS}): stop one first", lang))
            run_id = secrets.token_hex(4)
            while run_id in self.runs:
                run_id = secrets.token_hex(4)
            run = _Run(self, run_id, spec)
            self.runs[run_id] = run
        run.thread.start()
        self.pump_wake.set()
        return {'runId': run_id}

    # ── API: tools ──────────────────────────────────────────────────────────
    def _tool_capture(self):
        out = _Collector()
        return out, capture_output(out, out)

    def api_prove(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        abs_path, rel, _v, source, filename = self._program(body, lang)
        function = body.get('function')
        if function is not None and not isinstance(function, str):
            raise _ApiError(400, _t("missing or invalid field 'function'", lang))
        if not self.has_module('tilprove'):
            raise _ApiError(503, _t("not available in this installation (tilprove.py is missing)", lang))
        if not _find_spec('z3'):
            return {'available': False, 'results': [], 'hint': 'pip install z3-solver'}
        timeout_ms = body.get('timeout')
        if timeout_ms is not None and (isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int)
                                       or timeout_ms < 1):
            raise _ApiError(400, _t("missing or invalid field 'timeout'", lang))
        request = {'path': filename, 'rel': rel or 'untitled.til', 'source': source, 'function': function,
                   'lang': lang or 'en', 'timeout_ms': timeout_ms}
        return self._worker('prove', request, PROVE_SECONDS, lang)

    def _worker(self, kind: str, request: Dict[str, Any], seconds: int, lang: Optional[str]) -> Dict[str, Any]:
        """Run `tilstudio.py --studio-worker KIND` (JSON in, JSON out)."""
        cmd = [sys.executable, os.path.abspath(__file__), '--studio-worker', kind]
        env = dict(os.environ)
        env['PYTHONIOENCODING'] = 'utf-8'
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    cwd=self.root, env=env, **_group_kwargs())
        except OSError as e:
            raise _ApiError(500, _t(f"cannot start the prover: {e.strerror or e}", lang))
        with self.workers_lock:
            self.workers.add(proc)
        try:
            out, err = proc.communicate(json.dumps(request).encode('utf-8'), timeout=seconds)
        except subprocess.TimeoutExpired:
            _kill_process(proc)
            proc.communicate()
            raise _ApiError(504, _t(f"the prover did not finish in {seconds} s", lang))
        finally:
            with self.workers_lock:
                self.workers.discard(proc)
        if self.stopping.is_set():
            raise _ApiError(503, _t("TIL Studio is stopping", lang))
        try:
            result = json.loads(out.decode('utf-8'))
            if not isinstance(result, dict):
                raise ValueError(result)
            return result
        except ValueError:
            tail = err.decode('utf-8', 'replace').strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
            raise _ApiError(500, _t(f"the prover failed: {tail[0]}", lang))

    def _device(self, fit, device: Any, lang: Optional[str]):
        if not isinstance(device, str) or not device:
            raise _ApiError(400, _t("missing or invalid field 'device'", lang))
        spec = device
        if device not in fit.DEVICES:
            if not device.endswith('.json'):
                raise _ApiError(400, _t(f"unknown device '{device}'", lang))
            spec, _rel, _v = self.project.resolve(device, lang, allow_root=False)
        try:
            return fit.load_device(spec)
        except KeyError:
            raise _ApiError(400, _t(f"unknown device '{device}'", lang))
        except (ValueError, OSError) as e:
            raise _ApiError(400, _t(f"invalid device '{device}': {e}", lang))

    def api_fit(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        fit = self.module('tilfit', lang)
        abs_path, rel, _v, source, filename = self._program(body, lang)
        dev = self._device(fit, body.get('device'), lang)
        max_steps = body.get('maxSteps') or 0
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise _ApiError(400, _t("missing or invalid field 'maxSteps'", lang))
        log, capture = self._tool_capture()
        with capture:
            try:
                module = til.compile_to_bytecode(source, filename)
                data = module.serialize()
                saving = max(0, len(data) - len(module.serialize(strip=True)))
                an = fit.Analysis(data, rel or 'untitled.til')
            except SyntaxError as e:
                raise _ApiError(422, _plain(til.localize(f"Syntax Error: {e}", lang)))
            except til.CompileError as e:
                raise _ApiError(422, _plain(til.localize(str(e), lang)))
            except ValueError as e:
                raise _ApiError(422, _plain(til.localize(f"Error: {e}", lang)))
            report = fit.check(an, dev, saving)
            run_error = None
            if body.get('run'):
                used = {name for ff in an.reachable_facts() for _pc, name, _a in ff.natives}
                odd = sorted(used & set(fit.NONDETERMINISTIC_NATIVES))
                if 'random' in odd and dev.int_bits == 64:
                    odd.remove('random')
                why_not = ("the program uses " + ", ".join(n + "()" for n in odd)) if odd else ""
                try:
                    run = fit.run_on_device(dev, data, [], max_steps, FIT_RUN_SECONDS, b"", compare=True)
                    fit.apply_run(report, run, compared=not why_not, why_not=why_not)
                except RuntimeError as e:
                    run_error = str(e)
            result = _json_safe(fit.report_json(report))
        if run_error is not None:
            result['runError'] = _plain(run_error)
        return result

    def api_dis(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        abs_path, rel, _v, source, filename = self._program(body, lang)
        fuse = body.get('fuse', True) is not False
        log, capture = self._tool_capture()
        with capture:
            try:
                module = til.compile_to_bytecode(source, filename, True, fuse, bool(body.get('debugInfo')))
            except SyntaxError as e:
                raise _ApiError(422, _plain(til.localize(f"Syntax Error: {e}", lang)))
            except til.CompileError as e:
                raise _ApiError(422, _plain(til.localize(str(e), lang)))
            except ValueError as e:
                raise _ApiError(422, _plain(til.localize(f"Error: {e}", lang)))
            text = til.disassemble(module)
        return structured_disassembly(module, text)

    def api_translate(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        to = body.get('to')
        if to not in TRANSLATE_LANGUAGES:
            raise _ApiError(400, _t(f"unknown language '{to}' (use: {', '.join(TRANSLATE_LANGUAGES)})", lang))
        abs_path, rel, _v, source, filename = self._program(body, lang)
        log, capture = self._tool_capture()
        with capture:
            try:
                return {'text': til.translate_source(source, to, filename), 'to': to}
            except SyntaxError as e:
                raise _ApiError(422, _plain(til.localize(f"Syntax Error: {e}", lang)))
            except til.CompileError as e:
                raise _ApiError(422, _plain(til.localize(str(e), lang)))

    def api_build(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        abs_path, rel, virtual, _source, _filename = self._program({'path': body.get('path')}, lang,
                                                                   need_path=True)
        target = body.get('target') or 'vm'
        if target not in BUILD_TARGETS:
            raise _ApiError(400, _t(f"unknown target '{target}' (use vm, c-vm or native)", lang))
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        name = {'vm': stem + '.tbc', 'c-vm': stem + '_vm.c'}.get(
            target, stem + ('.exe' if sys.platform == 'win32' else ''))
        folder = self.root if virtual else os.path.dirname(abs_path)
        out_path = os.path.join(folder, name)
        if os.path.normcase(out_path) == os.path.normcase(abs_path):
            out_path = os.path.join(folder, stem + '.out')
        cmd = [sys.executable, _til_script(), 'build', abs_path, '--target', target, '-o', out_path,
               '--lang', lang or 'en']
        device = body.get('device')
        if device:
            fit = self.module('tilfit', lang)
            self._device(fit, device, lang)          # a known device or a .json file in the project
            cmd += ['--device', device if device in fit.DEVICES else
                    self.project.resolve(device, lang, allow_root=False)[0]]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self.root,
                                    **_group_kwargs())
        except OSError as e:
            raise _ApiError(500, _t(f"cannot start the build: {e.strerror or e}", lang))
        with self.workers_lock:
            self.workers.add(proc)
        try:
            out, _ = proc.communicate(timeout=BUILD_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process(proc)
            out, _ = proc.communicate()
        finally:
            with self.workers_lock:
                self.workers.discard(proc)
        ok = proc.returncode == 0 and os.path.exists(out_path)
        artifact = self.project.rel(out_path) if ok else None
        return {'ok': ok, 'rc': proc.returncode, 'output': _plain(out.decode('utf-8', 'replace')),
                'artifact': artifact}

    def api_pkg(self, body: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
        argv = body.get('argv')
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise _ApiError(400, _t("missing or invalid field 'argv'", lang))
        if argv[0] not in PKG_COMMANDS:
            raise _ApiError(400, _t(f"unknown package command '{argv[0]}' (use init, add, install, remove, "
                                    f"list, verify)", lang))
        if any(a == '-C' or a.startswith('-C') or a.startswith('--directory') for a in argv):
            raise _ApiError(400, _t("missing or invalid field 'argv'", lang))
        manifest_path = os.path.join(self.root, 'til.json')
        pkg = self.module('tilpkg', lang)
        out, err = _Collector(), _Collector()
        if argv[0] != 'init' and not os.path.isfile(manifest_path):
            err(_t("til pkg: error: no til.json in the project folder (create one with `til pkg init`)",
                   lang) + "\n")
            rc = 1
        else:
            with self.pkg_lock, capture_output(out, err):
                try:
                    rc = pkg.main(['-C', self.root] + argv)
                except SystemExit as e:
                    rc = e.code if isinstance(e.code, int) else 1
        manifest = None
        try:
            with open(manifest_path, encoding='utf-8-sig') as f:
                manifest = json.load(f)
        except (OSError, ValueError, RecursionError):
            pass
        return {'rc': rc, 'stdout': out.text(), 'stderr': _plain(err.text()), 'manifest': manifest}


def posix_dirname(rel: str) -> str:
    return rel.rsplit('/', 1)[0] if '/' in rel else ''


def _path_uri(path: str) -> str:
    try:
        import pathlib
        return pathlib.Path(path).as_uri()
    except ValueError:
        return 'file://' + urllib.parse.quote(path.replace(os.sep, '/'))


def _key_code(k: Any) -> int:
    """A key for POST /key: a code (int) or a --keys name / one character."""
    if isinstance(k, bool):
        raise ValueError(k)
    if isinstance(k, int):
        if not 0 <= k <= 0x10FFFF:
            raise ValueError(k)
        return k
    if isinstance(k, str) and k:
        codes = til.parse_key_script(k)
        if len(codes) == 1:
            return codes[0]
    raise ValueError(k)


def _atomic_write(path: str, data: bytes, mode: Optional[int]):
    """Write a file through a temporary file in the same folder (a crash never
    leaves half a file); keeps the permission bits."""
    folder, name = os.path.split(path)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix='.' + name + '.', suffix=TEMP_SUFFIX)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        try:
            os.chmod(tmp, mode if mode is not None else 0o644)
        except OSError:
            pass
        try:
            os.replace(tmp, path)
        except PermissionError:
            if os.name != 'nt':
                raise
            with open(path, 'wb') as f:      # Windows: the file is open elsewhere; write it in place
                f.write(data)
            os.unlink(tmp)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def structured_disassembly(module, text: str) -> Dict[str, Any]:
    """`til dis` as data: {text, functions: [{name, index, start, end}], lines:
    [[pc, line]], instructions: [{pc, offset, func, op, args, note, line, fused}]}.
    pc: absolute code offset (as in the line table); offset: within the function
    (as `text` shows it); line: the source line in effect; note: the comment of
    the listing (a constant, a call target, a jump target, ...)."""
    code = module.code
    funcs = module.funcs
    bounds = sorted((f.offset, i) for i, f in enumerate(funcs))
    ends = {idx: (bounds[k + 1][0] if k + 1 < len(bounds) else len(code)) for k, (_o, idx) in enumerate(bounds)}
    line_at = dict(module.lines)
    lines = sorted(module.lines)
    starts = [pc for pc, _l in lines]

    def line_for(pc: int) -> int:
        i = bisect.bisect_right(starts, pc) - 1
        return lines[i][1] if i >= 0 else 0
    listing = [ln for ln in text.split('\n') if re.match(r'^  \d{4,}  ', ln)]
    instructions = []
    functions = []
    k = 0
    for i, f in enumerate(funcs):
        start, end = f.offset, ends[i]
        functions.append({'name': f.name, 'index': i, 'start': start, 'end': end, 'params': f.n_params,
                          'locals': f.n_locals})
        pc = start
        while pc < end:
            op = code[pc]
            name = til.OP_NAMES.get(op, f"0x{op:02x}")
            p = pc + 1
            args = []
            for kind in til.OP_OPERANDS.get(op, ''):
                if kind == 'b':
                    args.append(code[p])
                    p += 1
                elif kind == 'c':
                    args.append(code[p] - 256 if code[p] > 127 else code[p])
                    p += 1
                elif kind == 'h':
                    args.append(code[p] | (code[p + 1] << 8))
                    p += 2
                else:
                    v = code[p] | (code[p + 1] << 8)
                    args.append(v - 0x10000 if v >= 0x8000 else v)
                    p += 2
            note = ""
            if k < len(listing):
                prefix = f"  {pc - start:04d}  {name:<18}{' '.join(str(v) for v in args):<12}"
                row = listing[k]
                if row.startswith(prefix.rstrip()):
                    rest = row[len(prefix):]
                    if pc in line_at:
                        rest = re.sub(r'\s*@ line \d+$', '', rest)
                    rest = rest.strip()
                    note = rest[2:] if rest.startswith('; ') else ''
            k += 1
            instructions.append({'pc': pc, 'offset': pc - start, 'func': f.name, 'op': name, 'args': args,
                                 'note': note, 'line': line_for(pc), 'fused': op >= 0x80,
                                 'lineStart': pc in line_at})
            pc = p
    return {'text': text, 'functions': functions, 'lines': [[pc, ln] for pc, ln in module.lines],
            'instructions': instructions}


# ── HTTP ────────────────────────────────────────────────────────────────────

_ROUTES: List[Tuple[str, Any, str]] = []
_SENT = object()                # a route that sent its reply itself


def _route(method: str, pattern: str):
    def deco(func):
        _ROUTES.append((method, re.compile('^' + pattern + '$'), func.__name__))
        return func
    return deco


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'          # one request per connection (event streams end by closing)
    server_version = 'TILStudio/' + til.TIL_VERSION
    sys_version = ''
    timeout = 60                            # seconds a socket read or write may block

    def log_message(self, format, *args):  # noqa: A002 (the base class's signature)
        pass

    def send_error(self, code, *args, **kwargs):
        """Errors of the HTTP layer itself (malformed requests): JSON too."""
        try:
            text = args[0] if args and args[0] else http.HTTPStatus(code).phrase
        except ValueError:
            text = str(code)
        self.close_connection = True
        try:
            self._send_json(code, {'error': str(text)})
        except OSError:
            pass

    # every method goes through _dispatch
    def do_GET(self):
        self._dispatch('GET')

    def do_POST(self):
        self._dispatch('POST')

    def do_PUT(self):
        self._dispatch('PUT')

    def do_DELETE(self):
        self._dispatch('DELETE')

    def do_PATCH(self):
        self._dispatch('PATCH')

    def do_OPTIONS(self):
        self._dispatch('OPTIONS')

    def do_HEAD(self):
        self._dispatch('HEAD')

    # ── responses ───────────────────────────────────────────────────────────
    def _send(self, status: int, body: bytes, ctype: Optional[str], headers: Tuple[Tuple[str, str], ...] = ()):
        self.send_response(status)
        if ctype:
            self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _send_json(self, status: int, data: Any):
        self._send(status, _dumps(data).encode('utf-8'), 'application/json; charset=utf-8')

    # ── checks ──────────────────────────────────────────────────────────────
    def _lang(self) -> str:
        studio = self.server.studio
        header = self.headers.get('X-TIL-Lang')
        lang = til.message_language_code(header) if header else None
        if lang is None:
            lang = studio.lang
        if lang is None:
            best = []
            for part in (self.headers.get('Accept-Language') or '').split(','):
                tag, _sep, params = part.strip().partition(';')
                code = til.message_language_code(tag.strip())
                q = 1.0
                m = re.search(r'q=([0-9.]+)', params)
                if m:
                    try:
                        q = float(m.group(1))
                    except ValueError:
                        q = 0.0
                if code and q > 0:
                    best.append((-q, len(best), code))
            lang = min(best)[2] if best else None
        return lang or 'en'

    def _host_ok(self) -> bool:
        port = self.server.studio.port
        host = (self.headers.get('Host') or '').strip().lower()
        return host in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _origin_ok(self, method: str, route: str) -> bool:
        port = self.server.studio.port
        origin = self.headers.get('Origin')
        if origin is not None and origin.strip().lower() not in (f"http://127.0.0.1:{port}",
                                                                 f"http://localhost:{port}"):
            return False
        site = (self.headers.get('Sec-Fetch-Site') or '').strip().lower()
        if site in ('cross-site', 'same-site') and not (method == 'GET' and route in ('/', '/index.html')):
            return False
        return True

    def _cookie_token(self) -> Optional[str]:
        raw = self.headers.get('Cookie')
        if not raw:
            return None
        try:
            cookie = http.cookies.SimpleCookie()
            cookie.load(raw)
        except http.cookies.CookieError:
            return None
        morsel = cookie.get(self.server.studio.cookie_name)
        return morsel.value if morsel is not None else None

    @staticmethod
    def _same(a: Optional[str], b: str) -> bool:
        return bool(a) and hmac.compare_digest(a.encode('utf-8', 'replace'), b.encode('utf-8'))

    def _token_ok(self, query: Dict[str, List[str]]) -> bool:
        token = self.server.studio.token
        return (self._same(self.headers.get('X-TIL-Token'), token)
                or self._same((query.get('token') or [None])[0], token)
                or self._same(self._cookie_token(), token))

    def _body(self, lang: str) -> Dict[str, Any]:
        if self.headers.get('Transfer-Encoding'):
            self.close_connection = True
            raise _ApiError(411, _t("the request needs a Content-Length", lang))
        length = self.headers.get('Content-Length')
        if length is None:
            return {}
        try:
            n = int(length)
        except ValueError:
            raise _ApiError(400, _t("the request needs a JSON object", lang))
        if n < 0:
            raise _ApiError(400, _t("the request needs a JSON object", lang))
        if n > BODY_LIMIT:
            self.close_connection = True
            raise _ApiError(413, _t(f"the request is too large (more than {BODY_LIMIT // _MB} MB)", lang))
        data = self.rfile.read(n) if n else b""
        if len(data) != n:
            raise _ApiError(400, _t("the request needs a JSON object", lang))
        if not data.strip():
            return {}
        try:
            value = json.loads(data.decode('utf-8'))
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise _ApiError(400, _t("the request needs a JSON object", lang))
        if not isinstance(value, dict):
            raise _ApiError(400, _t("the request needs a JSON object", lang))
        return value

    # ── dispatch ────────────────────────────────────────────────────────────
    def _dispatch(self, method: str):
        studio = self.server.studio
        lang = 'en'
        try:
            lang = self._lang()
            url = urllib.parse.urlsplit(self.path)
            route = url.path
            query = urllib.parse.parse_qs(url.query, keep_blank_values=True)
            if not self._host_ok():
                raise _ApiError(403, _t(f"requests must be sent to 127.0.0.1:{studio.port} or "
                                        f"localhost:{studio.port}", lang))
            if not self._origin_ok(method, route):
                raise _ApiError(403, _t("cross-origin requests are refused", lang))
            if route in ('/', '/index.html') and method in ('GET', 'HEAD'):
                return self._app(query, lang)
            if not self._token_ok(query):
                raise _ApiError(403, _t("missing or wrong token: open the address that `til ide` printed",
                                        lang))
            if route == '/favicon.ico' and method == 'GET':
                return self._send(204, b"", None)
            if route == '/api/events':
                if method != 'GET':
                    raise _ApiError(405, _t(f"method {method} is not allowed here", lang))
                return self._events()
            known = False
            for m, pattern, name in _ROUTES:
                match = pattern.match(route)
                if match is None:
                    continue
                known = True
                if m != method:
                    continue
                body = self._body(lang) if method in ('POST', 'PUT') else {}
                result = getattr(self, name)(studio, body, query, lang, *match.groups())
                if result is not _SENT:
                    self._send_json(200, result)
                return
            if known:
                raise _ApiError(405, _t(f"method {method} is not allowed here", lang))
            raise _ApiError(404, _t(f"unknown address: {route}", lang))
        except _ApiError as e:
            try:
                self._send_json(e.status, {'error': e.text})
            except OSError:
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
            self.close_connection = True
        except Exception as e:      # never a traceback to the client
            studio.log_exception(f"{method} {self.path.split('?', 1)[0]}")
            try:
                self._send_json(500, {'error': _t(f"internal error: {type(e).__name__}: {e}", lang)})
            except OSError:
                self.close_connection = True

    def _app(self, query: Dict[str, List[str]], lang: str):
        studio = self.server.studio
        token = (query.get('token') or [None])[0]
        headers = [('Content-Security-Policy', "frame-ancestors 'none'; object-src 'none'; base-uri 'none'")]
        if self._same(token, studio.token):
            headers.append(('Set-Cookie', f"{studio.cookie_name}={studio.token}; Path=/; HttpOnly; "
                                          f"SameSite=Strict"))
        elif not self._same(self._cookie_token(), studio.token):
            text = _t("missing or wrong token: open the address that `til ide` printed", lang)
            page = _FALLBACK_PAGE % {'message': _html(text), 'root': ''}
            return self._send(403, page.encode('utf-8'), 'text/html; charset=utf-8', tuple(headers))
        self._send(200, studio.app_page(lang), 'text/html; charset=utf-8', tuple(headers))

    def _events(self):
        """GET /api/events: the event stream of one tab (until it disconnects)."""
        studio = self.server.studio
        self.close_connection = True
        client = studio.hub.subscribe(self.connection)
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Accel-Buffering', 'no')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(b"retry: 2000\n\n" + _ping_frame())
            self.wfile.flush()
            next_ping = time.monotonic() + PING_SECONDS
            last_check = time.monotonic()
            while not studio.stopping.is_set():
                now = time.monotonic()
                data = studio.hub.take(client, max(0.05, min(1.0, next_ping - now)))
                if data is None:
                    break
                now = time.monotonic()
                if now >= next_ping:
                    data += _ping_frame()
                    next_ping = now + PING_SECONDS
                if data:
                    self.wfile.write(data)
                    self.wfile.flush()
                if now - last_check >= 1.0:
                    last_check = now
                    if _peer_closed(self.connection):
                        break
        except (OSError, ValueError):
            pass
        finally:
            studio.hub.unsubscribe(client)

    # ── API routes: (self, studio, body, query, lang, *groups) -> JSON ──────
    @_route('GET', '/api/info')
    def r_info(self, studio, body, query, lang):
        return studio.api_info(lang)

    @_route('GET', '/api/tree')
    def r_tree(self, studio, body, query, lang):
        return studio.api_tree((query.get('path') or [''])[0], lang)

    @_route('GET', '/api/file')
    def r_file_get(self, studio, body, query, lang):
        return studio.api_file_get((query.get('path') or [None])[0], lang)

    @_route('PUT', '/api/file')
    def r_file_put(self, studio, body, query, lang):
        return studio.api_file_put(body, lang)

    @_route('POST', '/api/file/create')
    def r_file_create(self, studio, body, query, lang):
        return studio.api_file_create(body, lang)

    @_route('POST', '/api/file/rename')
    def r_file_rename(self, studio, body, query, lang):
        return studio.api_file_rename(body, lang)

    @_route('POST', '/api/file/delete')
    def r_file_delete(self, studio, body, query, lang):
        return studio.api_file_delete(body, lang)

    @_route('POST', '/api/search')
    def r_search(self, studio, body, query, lang):
        return studio.api_search(body, lang)

    @_route('POST', '/api/lsp')
    def r_lsp(self, studio, body, query, lang):
        return {'messages': studio.lsp.handle(body, lang)}

    @_route('POST', '/api/run')
    def r_run(self, studio, body, query, lang):
        return studio.api_run(body, lang)

    @_route('POST', r'/api/run/([A-Za-z0-9_-]+)/stdin')
    def r_run_stdin(self, studio, body, query, lang, run_id):
        text = body.get('text', '')
        if not isinstance(text, str):
            raise _ApiError(400, _t("missing or invalid field 'text'", lang))
        if text and not text.endswith('\n'):
            text += '\n'                         # every /stdin text is a complete line
        studio.get_run(run_id, lang).send_stdin(text.encode('utf-8', 'replace'), bool(body.get('eof')), lang)
        return {}

    @_route('POST', r'/api/run/([A-Za-z0-9_-]+)/key')
    def r_run_key(self, studio, body, query, lang, run_id):
        try:
            code = _key_code(body.get('code'))
        except (ValueError, TypeError):
            raise _ApiError(400, _t("missing or invalid field 'code'", lang))
        studio.get_run(run_id, lang).press(code, lang)
        return {}

    @_route('POST', r'/api/run/([A-Za-z0-9_-]+)/stop')
    def r_run_stop(self, studio, body, query, lang, run_id):
        studio.get_run(run_id, lang).stop()
        return {}

    @_route('POST', r'/api/run/([A-Za-z0-9_-]+)/snapshot')
    def r_run_snapshot(self, studio, body, query, lang, run_id):
        return studio.get_run(run_id, lang).snapshot(bool(body.get('stop')), lang)

    @_route('POST', '/api/dap/start')
    def r_dap_start(self, studio, body, query, lang):
        return {'sessionId': studio.dap.start(lang)}

    @_route('POST', r'/api/dap/([A-Za-z0-9_-]+)')
    def r_dap(self, studio, body, query, lang, sid):
        studio.dap.handle(sid, body, lang, lambda out: self._send_json(200, {'messages': out}))
        return _SENT

    @_route('POST', '/api/tools/prove')
    def r_prove(self, studio, body, query, lang):
        return studio.api_prove(body, lang)

    @_route('POST', '/api/tools/fit')
    def r_fit(self, studio, body, query, lang):
        return studio.api_fit(body, lang)

    @_route('POST', '/api/tools/dis')
    def r_dis(self, studio, body, query, lang):
        return studio.api_dis(body, lang)

    @_route('POST', '/api/tools/translate')
    def r_translate(self, studio, body, query, lang):
        return studio.api_translate(body, lang)

    @_route('POST', '/api/tools/build')
    def r_build(self, studio, body, query, lang):
        return studio.api_build(body, lang)

    @_route('POST', '/api/pkg')
    def r_pkg(self, studio, body, query, lang):
        return studio.api_pkg(body, lang)


def _ping_frame() -> bytes:
    return ("event: ping\ndata: %s\n\n" % json.dumps({'time': int(time.time() * 1000)})).encode('utf-8')


def _peer_closed(sock) -> bool:
    """Has the browser closed the event stream (readable with nothing to read)?"""
    try:
        import select
        readable, _w, _x = select.select([sock], [], [], 0)
        if not readable:
            return False
        return sock.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


def _html(text: str) -> str:
    return (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            .replace('"', '&quot;').replace('`', '&#96;'))





# ── the browser ─────────────────────────────────────────────────────────────

def _chromium_browsers() -> List[str]:
    """Chrome / Edge / Chromium executables (they open `--app=URL` as an app window)."""
    found = []
    if sys.platform == 'darwin':
        for app, exe in (('Google Chrome', 'Google Chrome'), ('Microsoft Edge', 'Microsoft Edge'),
                         ('Chromium', 'Chromium'), ('Brave Browser', 'Brave Browser')):
            for base in ('/Applications', os.path.expanduser('~/Applications')):
                path = os.path.join(base, app + '.app', 'Contents', 'MacOS', exe)
                if os.path.isfile(path):
                    found.append(path)
    elif os.name == 'nt':
        bases = [os.environ.get(v) for v in ('PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA')]
        for base in [b for b in bases if b]:
            for rel in (('Google', 'Chrome', 'Application', 'chrome.exe'),
                        ('Microsoft', 'Edge', 'Application', 'msedge.exe'),
                        ('Chromium', 'Application', 'chrome.exe')):
                path = os.path.join(base, *rel)
                if os.path.isfile(path):
                    found.append(path)
    else:
        for name in ('google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser',
                     'microsoft-edge', 'microsoft-edge-stable', 'brave-browser'):
            path = shutil.which(name)
            if path:
                found.append(path)
    return found


def open_browser(url: str) -> bool:
    """An app window of Chrome / Edge / Chromium (`--app=URL`), else the default
    browser. False when nothing could be opened (no display)."""
    if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        return False
    for exe in _chromium_browsers():
        try:
            kwargs: Dict[str, Any] = {'stdin': subprocess.DEVNULL, 'stdout': subprocess.DEVNULL,
                                      'stderr': subprocess.DEVNULL}
            if os.name == 'posix':
                kwargs['start_new_session'] = True
            else:
                kwargs['creationflags'] = getattr(subprocess, 'DETACHED_PROCESS', 0)
            proc = subprocess.Popen([exe, f"--app={url}", '--new-window'], **kwargs)
        except OSError:
            continue

        def check(p=proc):
            try:
                rc = p.wait(3.0)
            except subprocess.TimeoutExpired:
                return                       # still running: the window is open
            if rc != 0:
                _default_browser(url)
        threading.Thread(target=check, name='til-studio-browser', daemon=True).start()
        return True
    return _default_browser(url)


def _default_browser(url: str) -> bool:
    try:
        import webbrowser
        return webbrowser.open(url, new=1)
    except Exception:
        return False


# ── command line ────────────────────────────────────────────────────────────

USAGE = """usage: til ide [FOLDER] [--port N] [--no-browser] [--lang en|ru|kk]

TIL Studio: the TIL IDE in your browser, for the project in FOLDER (default:
the current folder). The server listens on 127.0.0.1 only and prints the
address to open (with this launch's access token).

options:
  --port N        port (default 0: a free one)
  --no-browser    do not open a browser window
  --lang L        language of messages and of the UI default: en, ru, kk
  -h, --help      this help
"""


def _say(text: str, stream=None):
    stream = stream or sys.stdout
    try:
        stream.write(text + "\n")
        stream.flush()
    except (OSError, ValueError, AttributeError):
        pass


@til.scoped_message_language
def main(argv: List[str]) -> int:
    """`til ide`. Messages are in the language of TIL's messages (--lang / TIL_LANG)."""
    folder, port, browser, lang = None, 0, True, None
    i = 0
    while i < len(argv):
        a = argv[i]
        value = None
        if a.startswith('--') and '=' in a:
            a, value = a.split('=', 1)
        if a in ('-h', '--help'):
            _say(USAGE.rstrip('\n'))
            return 0
        if a in ('--port', '--lang'):
            if value is None:
                if i + 1 >= len(argv):
                    _say(_t(f"til ide: {a} needs a value", til.MESSAGE_LANG) + "\n\n" + USAGE.rstrip('\n'),
                         sys.stderr)
                    return 2
                i += 1
                value = argv[i]
            if a == '--port':
                try:
                    port = int(value)
                    if not 0 <= port <= 65535:
                        raise ValueError(value)
                except ValueError:
                    _say(_t(f"til ide: --port needs a number 0..65535, got '{value}'", til.MESSAGE_LANG),
                         sys.stderr)
                    return 2
            else:
                lang = til.message_language_code(value)
                if lang is None:
                    _say(_t(f"til ide: unknown message language '{value}' (use: en, ru, kk)", til.MESSAGE_LANG),
                         sys.stderr)
                    return 2
        elif a == '--no-browser':
            browser = False
        elif a.startswith('-') and a != '-':
            _say(_t(f"til ide: unknown option {a}", til.MESSAGE_LANG) + "\n\n" + USAGE.rstrip('\n'), sys.stderr)
            return 2
        elif folder is None:
            folder = a
        else:
            _say(_t(f"til ide: unknown option {a}", til.MESSAGE_LANG) + "\n\n" + USAGE.rstrip('\n'), sys.stderr)
            return 2
        i += 1
    if lang is None and getattr(til, '_MESSAGE_LANG_FROM_USER', False):
        lang = til.MESSAGE_LANG            # `til ide --lang ru` / TIL_LANG (til.py took --lang)
    say_lang = lang or til.MESSAGE_LANG
    root = os.path.abspath(folder or os.getcwd())
    if not os.path.isdir(root):
        _say(_t(f"til ide: no such folder: {root}", say_lang), sys.stderr)
        return 1
    try:
        studio = StudioServer(root, port=port, lang=lang)
    except OSError as e:
        _say(_t(f"til ide: cannot listen on 127.0.0.1:{port} ({e.strerror or e})", say_lang), sys.stderr)
        return 1
    try:
        os.chdir(studio.root)               # programs and imports see the project folder as the current one
    except OSError:
        pass
    _say(_t(f"TIL Studio {til.TIL_VERSION}: project {studio.root}", say_lang))
    _say("  " + studio.url)
    _say(_t("press Ctrl+C to stop", say_lang))
    if browser and not open_browser(studio.url):
        _say(_t("open the address above in a browser", say_lang))
    signals = _StopSignals()
    try:
        studio.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        signals.during_cleanup()            # a second Ctrl+C does not cut the cleanup short
        try:
            studio.stop()
        finally:
            signals.restore()
    return 0


class _StopSignals:
    """SIGTERM ends the studio like Ctrl+C (runs and their processes are cleaned
    up); while the studio stops, SIGINT and SIGTERM are ignored."""

    def __init__(self):
        self.saved: Dict[int, Any] = {}
        try:
            import signal

            def on_term(signum, frame):
                raise KeyboardInterrupt
            self.saved[signal.SIGTERM] = signal.signal(signal.SIGTERM, on_term)
            self.saved[signal.SIGINT] = signal.getsignal(signal.SIGINT)
        except (ValueError, OSError, AttributeError):      # not the main thread
            self.saved = {}

    def during_cleanup(self):
        import signal
        for signum in self.saved:
            try:
                signal.signal(signum, signal.SIG_IGN)
            except (ValueError, OSError):
                pass

    def restore(self):
        import signal
        for signum, handler in self.saved.items():
            try:
                signal.signal(signum, handler if handler is not None else signal.SIG_DFL)
            except (ValueError, OSError, TypeError):
                pass


# ── the prover worker (`tilstudio.py --studio-worker prove`) ────────────────

def _proof_report(prover, p, rep, lang: str) -> Dict[str, Any]:
    summary = prover.summary_line(rep)
    prefix = rep.title + ": "
    out: Dict[str, Any] = {
        'function': rep.key, 'title': rep.title, 'line': rep.line, 'col': rep.col, 'status': rep.status,
        'detail': summary[len(prefix):] if summary.startswith(prefix) else summary,
        'obligations': rep.n_obligations, 'seconds': round(rep.seconds, 3),
        'errors': [_plain(til.localize(e, lang)) for e in prover.format_errors(rep, p.reporter, p.timeout)],
    }
    if rep.status == 'unsupported':
        out['reason'] = rep.reason
    failures = []
    for f in rep.failures:
        item: Dict[str, Any] = {'line': f.ob.line, 'col': f.ob.col, 'status': f.status, 'what': f.ob.what,
                                'inputs': {k: _js_value(v) for k, v in f.inputs},
                                'confirmed': f.confirmed, 'outcome': f.outcome}
        try:
            if f.order:
                item['args'] = [_js_value(v) for v in f.arguments()]
        except Exception:
            pass
        failures.append(item)
    out['failures'] = failures
    ce = rep.counterexample
    if ce is not None:
        out['counterexample'] = {k: _js_value(v) for k, v in ce.items()}
    return out


def _prove_request(request: Dict[str, Any]) -> Dict[str, Any]:
    prover = til.feature_module('tilprove')
    if not prover.load_z3():
        return {'available': False, 'results': [], 'hint': prover.INSTALL_HINT}
    lang = request.get('lang') or 'en'
    filename, source, rel = request['path'], request['source'], request.get('rel') or request['path']
    only = request.get('function') or None
    try:
        program = til.resolve_imports(til.parse_source(source, filename), filename)
        errors = til.SemanticChecker(source, filename).check(program)
        if errors:
            return {'available': True, 'results': [],
                    'error': _plain("\n\n".join(til.localize(e, lang) for e in errors))}
        til.BytecodeCompiler(source, filename).compile_program(program)
    except SyntaxError as e:
        return {'available': True, 'results': [], 'error': _plain(til.localize(f"Syntax Error: {e}", lang))}
    except til.CompileError as e:
        return {'available': True, 'results': [], 'error': _plain(til.localize(str(e), lang))}
    timeout = request.get('timeout_ms') or prover.DEFAULT_TIMEOUT_MS
    p = prover.Prover(program, source, filename, timeout)
    if only is not None and not p.targets(False, only):
        return {'available': True, 'results': [],
                'error': til.localize(f"til prove: no function named '{only}' in {rel}", lang)}
    results = [_proof_report(prover, p, p.prove_function(key), lang) for key in p.targets(True, only)]
    return {'available': True, 'z3': prover.z3_version(), 'results': results}


def _studio_worker(kind: str) -> int:
    request = json.loads(sys.stdin.buffer.read().decode('utf-8'))
    answer = sys.stdout
    sys.stdout = sys.stderr                  # what the tools print is not the answer
    try:
        if kind != 'prove':
            raise ValueError(kind)
        result = _prove_request(request)
    except Exception as e:
        result = {'available': True, 'results': [], 'error': f"{type(e).__name__}: {e}"}
    finally:
        sys.stdout = answer
    answer.write(_dumps(_json_safe(result)))
    answer.flush()
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--studio-worker':
        sys.exit(_studio_worker(sys.argv[2]))
    sys.exit(main(sys.argv[1:]))
