#!/usr/bin/env python3
"""
til lsp -- the TIL language server (Language Server Protocol 3.17).

    til lsp                    # stdio: Content-Length framing, JSON-RPC 2.0
    til lsp --tcp 2087         # one client on 127.0.0.1:2087
    til lsp --log lsp.log      # log the traffic and errors

Every feature comes from src/tills.py (the compiler's own lexer, parser and
checkers). Editor setup: docs/TIL_LSP.md.

In-process use (TIL Studio's server):

    server = LspServer(root)
    out = server.handle(message)     # responses and notifications to send now
    out += server.poll()             # notifications that became due (debounced diagnostics)

The server never stops on bad input: malformed JSON and invalid messages get
JSON-RPC errors, an exception inside a handler becomes an InternalError
response for that request.
"""

import json
import os
import queue
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set

if 'til' in sys.modules:            # til.py loaded us (also as __main__): use that module
    til = sys.modules['til']
else:                               # pragma: no cover - direct import (tests, lsp/til_lsp.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import til                      # noqa: E402
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
import tills                        # noqa: E402

SERVER_NAME = "til-lsp"
DEBOUNCE_SECONDS = 0.2

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
SERVER_NOT_INITIALIZED = -32002
REQUEST_FAILED = -32803

RUN_COMMANDS = ('til.run', 'til.debug', 'til.prove')     # the client runs these: the result is null


class _BadParams(Exception):
    pass


class _Failed(Exception):
    """A request that cannot be carried out (RequestFailed, with the reason)."""


def capabilities() -> Dict[str, Any]:
    return {
        'positionEncoding': 'utf-16',
        'textDocumentSync': {'openClose': True, 'change': 1, 'save': {'includeText': False}},
        'completionProvider': {'triggerCharacters': ['.', '[', ':'], 'resolveProvider': True},
        'hoverProvider': True,
        'signatureHelpProvider': {'triggerCharacters': ['(', ','], 'retriggerCharacters': [',']},
        'definitionProvider': True,
        'referencesProvider': True,
        'documentHighlightProvider': True,
        'renameProvider': {'prepareProvider': True},
        'documentSymbolProvider': True,
        'workspaceSymbolProvider': True,
        'foldingRangeProvider': True,
        'documentFormattingProvider': True,
        'inlayHintProvider': True,
        'codeActionProvider': {'codeActionKinds': ['quickfix', 'refactor.rewrite']},
        'codeLensProvider': {'resolveProvider': False},
        'semanticTokensProvider': {
            'legend': {'tokenTypes': list(tills.TOKEN_TYPES), 'tokenModifiers': list(tills.TOKEN_MODIFIERS)},
            'full': True, 'range': False},
        # til.translate and the til.run / til.debug / til.prove commands of the code
        # lenses are accepted by workspace/executeCommand but not listed here, so that
        # clients which register listed commands (VS Code) keep their own commands
        'executeCommandProvider': {'commands': []},
        'workspace': {'workspaceFolders': {'supported': True, 'changeNotifications': True}},
    }


def _notification(method: str, params: Any) -> dict:
    return {'jsonrpc': '2.0', 'method': method, 'params': params}


def _response(id_, result: Any) -> dict:
    return {'jsonrpc': '2.0', 'id': id_, 'result': result}


def _error(id_, code: int, text: str) -> dict:
    return {'jsonrpc': '2.0', 'id': id_, 'error': {'code': code, 'message': text}}


class LspServer:
    """The language server as a message handler (no I/O)."""

    def __init__(self, root: Optional[str] = None, locale: Optional[str] = None,
                 clock=time.monotonic, debounce: float = DEBOUNCE_SECONDS):
        self.root = os.path.abspath(root) if root else None
        self.forced_locale = locale             # --lang / TIL_LANG: beats the client's locale
        self.ws = tills.Workspace(self.root, locale or 'en')
        self.clock = clock
        self.debounce = debounce
        self.initialized = False
        self.shutdown_requested = False
        self.exit_code: Optional[int] = None
        self.pending: Dict[str, float] = {}     # uri -> when its diagnostics are due
        self.foreign: Dict[str, Dict[str, List[dict]]] = {}   # file -> {source uri: diagnostics}
        self.foreign_sources: Dict[str, Set[str]] = {}
        self.lock = threading.RLock()
        self.log = None                         # a callable(str), or None

    # ── entry points ────────────────────────────────────────────────────────
    def handle(self, message: Any) -> List[dict]:
        with self.lock:
            try:
                return self._handle(message)
            except Exception as e:                      # never crash on one message
                self._log_exception(e)
                if isinstance(message, dict) and 'id' in message and 'method' in message:
                    return [_error(message.get('id'), INTERNAL_ERROR, f"{type(e).__name__}: {e}")]
                return []

    def poll(self) -> List[dict]:
        """Diagnostics whose debounce time has passed."""
        with self.lock:
            now = self.clock()
            due = [u for u, t in self.pending.items() if t <= now]
            out: List[dict] = []
            for uri in due:
                self.pending.pop(uri, None)
                try:
                    out += self._publish(uri)
                except Exception as e:
                    self._log_exception(e)
            return out

    def next_due(self) -> Optional[float]:
        """Seconds until poll() has something to send (None: nothing pending)."""
        with self.lock:
            if not self.pending:
                return None
            return max(0.0, min(self.pending.values()) - self.clock())

    # ── dispatch ────────────────────────────────────────────────────────────
    def _handle(self, msg: Any) -> List[dict]:
        if not isinstance(msg, dict):
            return [_error(None, INVALID_REQUEST, "a JSON-RPC message must be an object")]
        method = msg.get('method')
        is_request = 'id' in msg
        id_ = msg.get('id')
        if method is None:
            if 'result' in msg or 'error' in msg:
                return []                               # a response to us (we send no requests)
            return [_error(id_ if is_request else None, INVALID_REQUEST, "missing 'method'")]
        if not isinstance(method, str) or (is_request and not isinstance(id_, (int, str)) and id_ is not None):
            return [_error(id_ if isinstance(id_, (int, str)) else None, INVALID_REQUEST, "invalid request")]
        params = msg.get('params')
        if params is None:
            params = {}
        if method == 'exit':
            self.exit_code = 0 if self.shutdown_requested else 1
            return []
        if not self.initialized and method != 'initialize':
            if is_request:
                return [_error(id_, SERVER_NOT_INITIALIZED, "the server is not initialized")]
            return []
        if self.shutdown_requested and is_request:
            return [_error(id_, INVALID_REQUEST, "the server is shutting down")]
        if is_request:
            handler = self.REQUESTS.get(method)
            if handler is None:
                return [_error(id_, METHOD_NOT_FOUND, f"method not found: {method}")]
            out: List[dict] = []
            try:
                result = handler(self, params, out)
            except _BadParams as e:
                return [_error(id_, INVALID_PARAMS, str(e))]
            except (tills.RenameError, _Failed) as e:
                return [_error(id_, REQUEST_FAILED, str(e))]
            return [_response(id_, result)] + out
        handler = self.NOTIFICATIONS.get(method)
        if handler is None:
            return []                                   # unknown notifications ($/...) are ignored
        out = []
        try:
            handler(self, params, out)
        except _BadParams as e:
            self._log(f"ignored {method}: {e}")
        return out

    # ── parameters ──────────────────────────────────────────────────────────
    @staticmethod
    def _obj(params, key: str) -> dict:
        v = params.get(key) if isinstance(params, dict) else None
        if not isinstance(v, dict):
            raise _BadParams(f"missing '{key}'")
        return v

    def _uri(self, params) -> str:
        td = self._obj(params, 'textDocument')
        uri = td.get('uri')
        if not isinstance(uri, str) or not uri:
            raise _BadParams("missing 'textDocument.uri'")
        return uri

    def _pos(self, params):
        p = self._obj(params, 'position')
        line, ch = p.get('line'), p.get('character')
        if not isinstance(line, int) or not isinstance(ch, int) or isinstance(line, bool) or line < 0 or ch < 0:
            raise _BadParams("invalid 'position'")
        return line, ch

    @staticmethod
    def _range(params, key: str = 'range') -> Optional[dict]:
        r = params.get(key) if isinstance(params, dict) else None
        if r is None:
            return None
        try:
            for k in ('start', 'end'):
                if not isinstance(r[k]['line'], int) or not isinstance(r[k]['character'], int):
                    raise TypeError
        except (KeyError, TypeError):
            raise _BadParams(f"invalid '{key}'")
        return r

    # ── lifecycle ───────────────────────────────────────────────────────────
    def on_initialize(self, params, out):
        root = None
        if isinstance(params, dict):
            if params.get('rootUri'):
                root = tills.uri_to_path(params['rootUri'])
            elif params.get('rootPath'):
                root = params['rootPath']
            folders = params.get('workspaceFolders')
            if not root and isinstance(folders, list) and folders and isinstance(folders[0], dict):
                root = tills.uri_to_path(folders[0].get('uri', ''))
        locale = self.forced_locale
        if locale is None and isinstance(params, dict):
            opts = params.get('initializationOptions')
            if isinstance(opts, dict) and isinstance(opts.get('locale'), str):
                locale = opts['locale']
            elif isinstance(params.get('locale'), str):
                locale = params['locale']
        if root and (self.root is None or not self.ws.docs):
            self.root = os.path.abspath(root)
            self.ws = tills.Workspace(self.root, locale or self.ws.locale)
        elif locale:
            self.ws.set_locale(locale)
        self.initialized = True
        return {'capabilities': capabilities(), 'serverInfo': {'name': SERVER_NAME, 'version': til.TIL_VERSION}}

    def on_initialized(self, params, out):
        pass

    def on_shutdown(self, params, out):
        self.shutdown_requested = True
        self.pending.clear()
        return None

    # ── documents ───────────────────────────────────────────────────────────
    def on_did_open(self, params, out):
        td = self._obj(params, 'textDocument')
        uri, text = td.get('uri'), td.get('text')
        if not isinstance(uri, str) or not isinstance(text, str):
            raise _BadParams("didOpen needs textDocument.uri and .text")
        version = td.get('version') if isinstance(td.get('version'), int) else 0
        self.ws.open(uri, text, version)
        self.pending.pop(uri, None)
        out += self._publish(uri)
        self._schedule_dependents(uri)

    def on_did_change(self, params, out):
        uri = self._uri(params)
        td = params['textDocument']
        version = td.get('version') if isinstance(td.get('version'), int) else None
        changes = params.get('contentChanges')
        if not isinstance(changes, list):
            raise _BadParams("missing 'contentChanges'")
        doc = self.ws.docs.get(uri)
        text = doc.raw if doc is not None else ""
        for ch in changes:
            if not isinstance(ch, dict) or not isinstance(ch.get('text'), str):
                continue
            if isinstance(ch.get('range'), dict):            # incremental (we ask for full, but accept it)
                pm = tills.PositionMap(text)
                r = ch['range']
                try:
                    a = pm.offset(r['start']['line'], r['start']['character'])
                    b = pm.offset(r['end']['line'], r['end']['character'])
                except (KeyError, TypeError):
                    continue
                text = text[:a] + ch['text'] + text[b:]
            else:
                text = ch['text']
        if version is None:
            version = (doc.version + 1) if doc is not None else 0
        self.ws.change(uri, text, version)
        self.pending[uri] = self.clock() + self.debounce

    def on_did_save(self, params, out):
        uri = self._uri(params)
        if isinstance(params.get('text'), str):
            doc = self.ws.docs.get(uri)
            self.ws.change(uri, params['text'], (doc.version + 1) if doc else 0)
        self.ws.saved(uri)
        if uri in self.ws.docs:
            self.pending[uri] = self.clock()
        self._schedule_dependents(uri)

    def on_did_close(self, params, out):
        uri = self._uri(params)
        self.ws.close(uri)
        self.pending.pop(uri, None)
        out.append(_notification('textDocument/publishDiagnostics', {'uri': uri, 'diagnostics': []}))
        for target in self.foreign_sources.pop(uri, set()):
            per = self.foreign.get(target, {})
            per.pop(uri, None)
            if not self._is_open(target):
                out.append(_notification('textDocument/publishDiagnostics',
                                         {'uri': target, 'diagnostics': self._merged(target)}))

    def on_watched_files(self, params, out):
        paths = []
        for ch in (params.get('changes') or []) if isinstance(params, dict) else []:
            if isinstance(ch, dict) and isinstance(ch.get('uri'), str):
                p = tills.uri_to_path(ch['uri'])
                if p:
                    paths.append(p)
        self.ws.files_changed(paths)
        now = self.clock()
        for uri in self.ws.docs:
            self.pending[uri] = now + self.debounce

    def on_configuration(self, params, out):
        settings = params.get('settings') if isinstance(params, dict) else None
        til_settings = settings.get('til') if isinstance(settings, dict) else None
        if isinstance(til_settings, dict) and isinstance(til_settings.get('locale'), str) and \
                self.forced_locale is None:
            self.ws.set_locale(til_settings['locale'])
            now = self.clock()
            for uri in self.ws.docs:
                self.pending[uri] = now

    def _schedule_dependents(self, uri: str, all_open: bool = False):
        """Open files that import `uri` (directly or not): their diagnostics may
        change (the compiler reads imported files from disk: after a save)."""
        doc = self.ws.docs.get(uri)
        path = doc.path if doc is not None else tills.uri_to_path(uri)
        if not path:
            return
        key = tills.path_key(path)
        due = self.clock() + self.debounce
        for other in list(self.ws.docs.values()):
            if other.uri == uri or not other.path:
                continue
            try:
                closure = self.ws._closure(other.path)
            except Exception:
                continue
            if all_open or any(tills.path_key(p) == key for p in closure):
                self.pending[other.uri] = due

    # ── diagnostics ─────────────────────────────────────────────────────────
    def _is_open(self, uri: str) -> bool:
        if uri in self.ws.docs:
            return True
        p = tills.uri_to_path(uri)
        if not p:
            return False
        k = tills.path_key(p)
        return any(d.path and tills.path_key(d.path) == k for d in self.ws.docs.values())

    def _merged(self, target: str) -> List[dict]:
        seen, out = set(), []
        for diags in self.foreign.get(target, {}).values():
            for d in diags:
                key = json.dumps([d.get('range'), d.get('code'), d.get('message')], sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        return out

    def _publish(self, uri: str) -> List[dict]:
        doc = self.ws.docs.get(uri)
        if doc is None:
            return []
        by_uri = self.ws.diagnostics_by_uri(uri)
        own = by_uri.pop(doc.uri, [])
        out = [_notification('textDocument/publishDiagnostics',
                             {'uri': uri, 'version': doc.version, 'diagnostics': own})]
        old = self.foreign_sources.get(uri, set())
        new = set(by_uri)
        for target in new:
            self.foreign.setdefault(target, {})[uri] = by_uri[target]
        for target in old - new:
            self.foreign.get(target, {}).pop(uri, None)
        self.foreign_sources[uri] = new
        for target in sorted(new | old):
            if self._is_open(target):
                continue
            out.append(_notification('textDocument/publishDiagnostics',
                                     {'uri': target, 'diagnostics': self._merged(target)}))
        return out

    # ── language features ───────────────────────────────────────────────────
    def on_completion(self, params, out):
        uri = self._uri(params)
        line, ch = self._pos(params)
        return {'isIncomplete': False, 'items': self.ws.completion(uri, line, ch)}

    def on_resolve(self, params, out):
        if not isinstance(params, dict) or 'label' not in params:
            raise _BadParams("completionItem/resolve needs a CompletionItem")
        return self.ws.resolve_completion(params)

    def on_hover(self, params, out):
        return self.ws.hover(self._uri(params), *self._pos(params))

    def on_signature(self, params, out):
        return self.ws.signature_help(self._uri(params), *self._pos(params))

    def on_definition(self, params, out):
        return self.ws.definition(self._uri(params), *self._pos(params))

    def on_references(self, params, out):
        ctx = params.get('context') if isinstance(params.get('context'), dict) else {}
        return self.ws.references(self._uri(params), *self._pos(params),
                                  include_declaration=bool(ctx.get('includeDeclaration', True)))

    def on_highlight(self, params, out):
        return self.ws.document_highlights(self._uri(params), *self._pos(params))

    def on_prepare_rename(self, params, out):
        return self.ws.prepare_rename(self._uri(params), *self._pos(params))

    def on_rename(self, params, out):
        new_name = params.get('newName')
        if not isinstance(new_name, str):
            raise _BadParams("missing 'newName'")
        return self.ws.rename(self._uri(params), *self._pos(params), new_name)

    def on_document_symbol(self, params, out):
        return self.ws.document_symbols(self._uri(params))

    def on_workspace_symbol(self, params, out):
        q = params.get('query', '') if isinstance(params, dict) else ''
        return self.ws.workspace_symbols(q if isinstance(q, str) else '')

    def on_folding(self, params, out):
        return self.ws.folding_ranges(self._uri(params))

    def on_formatting(self, params, out):
        opts = params.get('options') if isinstance(params.get('options'), dict) else {}
        tab = opts.get('tabSize', 4)
        return self.ws.formatting(self._uri(params), tab if isinstance(tab, int) and not isinstance(tab, bool)
                                  else 4)

    def on_inlay(self, params, out):
        return self.ws.inlay_hints(self._uri(params), self._range(params))

    def on_code_action(self, params, out):
        ctx = params.get('context') if isinstance(params.get('context'), dict) else {}
        diags = ctx.get('diagnostics') if isinstance(ctx.get('diagnostics'), list) else []
        actions = self.ws.code_actions(self._uri(params), self._range(params), diags)
        only = ctx.get('only')
        if isinstance(only, list) and only:
            actions = [a for a in actions if any(str(a.get('kind', '')).startswith(k) for k in only
                                                 if isinstance(k, str))]
        return actions

    def on_code_lens(self, params, out):
        return self.ws.code_lenses(self._uri(params))

    def on_semantic_tokens(self, params, out):
        return self.ws.semantic_tokens(self._uri(params))

    def on_execute(self, params, out):
        command = params.get('command') if isinstance(params, dict) else None
        args = params.get('arguments') if isinstance(params.get('arguments'), list) else []
        if command in RUN_COMMANDS:
            return None                                  # the client runs / debugs / proves
        if command == 'til.translate':
            uri, target = None, None
            if args and isinstance(args[0], dict):
                uri, target = args[0].get('uri'), args[0].get('to')
            elif len(args) >= 2:
                uri, target = args[0], args[1]
            if not isinstance(uri, str) or target not in tills.KEYWORD_LANGS:
                raise _BadParams("til.translate needs {uri, to: en|kk|kk-latn|ru}")
            doc = self.ws._doc(uri)
            if doc is None:
                raise _BadParams(f"unknown document: {uri}")
            try:
                return {'text': til.translate_source(doc.text, target, doc.path or "<lsp>")}
            except (SyntaxError, til.CompileError) as e:
                raise _Failed(str(e).split('\n')[0])
        raise _BadParams(f"unknown command: {command}")

    def on_ignore(self, params, out):
        return None

    REQUESTS = {
        'initialize': on_initialize,
        'shutdown': on_shutdown,
        'textDocument/completion': on_completion,
        'completionItem/resolve': on_resolve,
        'textDocument/hover': on_hover,
        'textDocument/signatureHelp': on_signature,
        'textDocument/definition': on_definition,
        'textDocument/references': on_references,
        'textDocument/documentHighlight': on_highlight,
        'textDocument/prepareRename': on_prepare_rename,
        'textDocument/rename': on_rename,
        'textDocument/documentSymbol': on_document_symbol,
        'workspace/symbol': on_workspace_symbol,
        'textDocument/foldingRange': on_folding,
        'textDocument/formatting': on_formatting,
        'textDocument/inlayHint': on_inlay,
        'textDocument/codeAction': on_code_action,
        'textDocument/codeLens': on_code_lens,
        'textDocument/semanticTokens/full': on_semantic_tokens,
        'workspace/executeCommand': on_execute,
    }
    NOTIFICATIONS = {
        'initialized': on_initialized,
        'textDocument/didOpen': on_did_open,
        'textDocument/didChange': on_did_change,
        'textDocument/didSave': on_did_save,
        'textDocument/didClose': on_did_close,
        'workspace/didChangeWatchedFiles': on_watched_files,
        'workspace/didChangeConfiguration': on_configuration,
        '$/cancelRequest': on_ignore,
        '$/setTrace': on_ignore,
    }

    # ── logging ─────────────────────────────────────────────────────────────
    def _log(self, text: str):
        if self.log is not None:
            try:
                self.log(text)
            except Exception:
                pass

    def _log_exception(self, e: Exception):
        if self.log is not None:
            import traceback
            self._log("internal error: " + ''.join(traceback.format_exception(type(e), e, e.__traceback__)))


# ═══════════════════════════════════════════════════════════════════════════════
#                         TRANSPORT: Content-Length framing
# ═══════════════════════════════════════════════════════════════════════════════

MAX_MESSAGE = 64 * 1024 * 1024
_EOF = object()


class FdReader:
    """readline() / read(n) straight from a file descriptor. The reader thread
    blocks in os.read, never inside sys.stdin's buffer, so the interpreter can
    shut down while it waits (a thread blocked in a BufferedReader holds its
    lock, and Python aborts at exit)."""

    def __init__(self, fd: int):
        self.fd = fd
        self.buf = b''
        self.eof = False

    def _fill(self) -> bool:
        if self.eof:
            return False
        try:
            chunk = os.read(self.fd, 65536)
        except OSError:
            chunk = b''
        if not chunk:
            self.eof = True
            return False
        self.buf += chunk
        return True

    def readline(self, limit: int = -1) -> bytes:
        while b'\n' not in self.buf and (limit < 0 or len(self.buf) < limit):
            if not self._fill():
                break
        k = self.buf.find(b'\n')
        end = k + 1 if k >= 0 else len(self.buf)
        if limit >= 0:
            end = min(end, limit)
        line, self.buf = self.buf[:end], self.buf[end:]
        return line

    def read(self, n: int) -> bytes:
        while len(self.buf) < n:
            if not self._fill():
                break
        data, self.buf = self.buf[:n], self.buf[n:]
        return data


def _binary_stdio():
    if os.name == 'nt':                                  # pragma: no cover - Windows
        try:
            import msvcrt
            msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
            msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        except (ImportError, OSError, ValueError):
            pass


class _Malformed:
    def __init__(self, why: str):
        self.why = why


def read_message(stream) -> Any:
    """One message from a binary stream: a dict / list / value, a _Malformed,
    or _EOF."""
    length = None
    saw_header = False
    while True:
        line = stream.readline(65537)
        if not line:
            return _EOF
        if line in (b'\r\n', b'\n'):
            if saw_header:
                break
            continue                                     # stray blank lines between messages
        saw_header = True
        try:
            name, _, value = line.decode('ascii', 'replace').partition(':')
        except Exception:
            continue
        if name.strip().lower() == 'content-length':
            try:
                length = int(value.strip())
            except ValueError:
                length = None
    if length is None or length < 0 or length > MAX_MESSAGE:
        return _Malformed("missing or invalid Content-Length")
    body = b''
    while len(body) < length:
        chunk = stream.read(length - len(body))
        if not chunk:
            return _EOF
        body += chunk
    try:
        return json.loads(body.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as e:
        return _Malformed(f"parse error: {e}")


def write_message(stream, message: dict):
    data = json.dumps(message, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    stream.write(b'Content-Length: %d\r\n\r\n' % len(data) + data)
    stream.flush()


def serve(server: LspServer, rfile, wfile, log=None) -> int:
    """Run the server over two binary streams until `exit` (or end of input)."""
    inbox: 'queue.Queue[Any]' = queue.Queue()

    def reader():
        while True:
            try:
                msg = read_message(rfile)
            except Exception as e:                      # a closed / broken stream
                if log:
                    log(f"read error: {e}")
                msg = _EOF
            inbox.put(msg)
            if msg is _EOF:
                return

    threading.Thread(target=reader, name='til-lsp-reader', daemon=True).start()
    while True:
        timeout = server.next_due()
        try:
            item = inbox.get(timeout=timeout) if timeout is not None else inbox.get()
        except queue.Empty:
            item = None
        if item is _EOF:
            break
        if item is None:
            out = server.poll()
        elif isinstance(item, _Malformed):
            out = [_error(None, PARSE_ERROR, item.why)]
        elif isinstance(item, list):                    # a JSON-RPC batch
            out = []
            for m in item:
                out += server.handle(m)
            if not item:
                out = [_error(None, INVALID_REQUEST, "empty batch")]
        else:
            if log:
                log("<- " + json.dumps(item, ensure_ascii=False)[:2000])
            out = server.handle(item)
        for m in out:
            if log:
                log("-> " + json.dumps(m, ensure_ascii=False)[:2000])
            try:
                write_message(wfile, m)
            except (OSError, ValueError):
                return 1
        if server.exit_code is not None:
            return server.exit_code
    return 0 if server.shutdown_requested else 1


USAGE = """\
usage: til lsp [--tcp PORT] [--log FILE] [--lang en|ru|kk]
  the TIL language server: LSP over stdio (or one TCP client on 127.0.0.1:PORT)
  editor setup: docs/TIL_LSP.md"""


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    port = None
    log_path = None
    lang_arg = None            # `til lsp --lang X` is taken by til.py; direct starts pass it here
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('-h', '--help'):
            print(USAGE)
            return 0
        if a == '--stdio':
            i += 1
            continue
        if a in ('--tcp', '--port') and i + 1 < len(argv):
            try:
                port = int(argv[i + 1])
            except ValueError:
                print(USAGE, file=sys.stderr)
                return 2
            i += 2
            continue
        if a.startswith('--tcp='):
            try:
                port = int(a.split('=', 1)[1])
            except ValueError:
                print(USAGE, file=sys.stderr)
                return 2
            i += 1
            continue
        if a == '--log' and i + 1 < len(argv):
            log_path = argv[i + 1]
            i += 2
            continue
        if a == '--lang' and i + 1 < len(argv) and til.message_language_code(argv[i + 1]):
            lang_arg = til.message_language_code(argv[i + 1])
            i += 2
            continue
        if a.startswith('--log='):
            log_path = a.split('=', 1)[1]
            i += 1
            continue
        if a.startswith('--clientProcessId'):
            i += 2 if '=' not in a else 1
            continue
        print(USAGE, file=sys.stderr)
        return 2
    log = None
    log_file = None
    if log_path:
        try:
            log_file = open(log_path, 'a', encoding='utf-8')
        except OSError as e:
            print(til.localize(f"til lsp: cannot open the log file: {e}"), file=sys.stderr)
            return 2
        lock = threading.Lock()

        def log(text: str):
            with lock:
                log_file.write(time.strftime('%H:%M:%S ') + text + '\n')
                log_file.flush()
    forced = lang_arg
    if forced is not None:
        pass
    elif getattr(til, '_MESSAGE_LANG_FROM_USER', False) and getattr(til, 'MESSAGE_LANG', None):
        forced = til.MESSAGE_LANG
    elif os.environ.get('TIL_LANG') and til.message_language_code(os.environ.get('TIL_LANG')):
        forced = til.message_language_code(os.environ.get('TIL_LANG'))
    server = LspServer(None, forced)
    server.log = log
    try:
        if port is not None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
            sock.listen(1)
            if log:
                log(f"listening on 127.0.0.1:{sock.getsockname()[1]}")
            conn, _addr = sock.accept()
            sock.close()
            rfile = conn.makefile('rb')
            wfile = conn.makefile('wb')
            try:
                return serve(server, rfile, wfile, log)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
        _binary_stdio()
        rfile = FdReader(sys.stdin.fileno())
        wfile = sys.stdout.buffer
        # anything printed while the server runs goes to stderr, never into the protocol
        saved = sys.stdout
        sys.stdout = sys.stderr
        try:
            return serve(server, rfile, wfile, log)
        finally:
            sys.stdout = saved
    except KeyboardInterrupt:
        return 1
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == '__main__':
    sys.exit(main())
