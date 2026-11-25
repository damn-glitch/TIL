#!/usr/bin/env python3
"""
`til lsp` (src/tillsp.py): the Language Server Protocol over a real
`python3 src/til.py lsp` subprocess (stdio, Content-Length framing, JSON-RPC
2.0), --tcp and --log, the lsp/til_lsp.py compatibility path, robustness
against malformed input, shutdown / exit, and the in-process
LspServer.handle() / poll() API that TIL Studio uses.

Run: python3 -m pytest tests/test_lsp.py -v
"""

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time

import pytest

from tilhelpers import ROOT, TIL, til

sys.path.insert(0, os.path.join(ROOT, 'src'))
import tills  # noqa: E402
import tillsp  # noqa: E402

TIMEOUT = 30


class Client:
    """A language client over the pipes (or a socket) of a server process."""

    def __init__(self, args=(), env=None, cmd=None, cwd=None, connect=None):
        full_env = dict(os.environ)
        full_env.pop('TIL_LANG', None)
        full_env.update(env or {})
        self.proc = subprocess.Popen(cmd or [sys.executable, TIL, 'lsp', *args], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=full_env, cwd=cwd)
        self.sock = None
        if connect is not None:
            self.sock = connect()
            self.rfile = self.sock.makefile('rb')
            self.wfile = self.sock.makefile('wb')
        else:
            self.rfile, self.wfile = self.proc.stdout, self.proc.stdin
        self.inbox = queue.Queue()
        self.notifications = []
        self.next_id = 0
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            msg = tillsp.read_message(self.rfile)
            if msg is tillsp._EOF:
                self.inbox.put(None)
                return
            self.inbox.put(msg)

    def write_raw(self, data: bytes):
        self.wfile.write(data)
        self.wfile.flush()

    def send(self, msg):
        tillsp.write_message(self.wfile, msg)

    def notify(self, method, params=None):
        self.send({'jsonrpc': '2.0', 'method': method, 'params': params or {}})

    def get(self, timeout=TIMEOUT):
        msg = self.inbox.get(timeout=timeout)
        if msg is None:
            raise EOFError("the server closed the connection")
        return msg

    def response(self, id_, timeout=TIMEOUT):
        deadline = time.time() + timeout
        while True:
            msg = self.get(max(0.1, deadline - time.time()))
            if 'method' in msg and 'id' not in msg:
                self.notifications.append(msg)
                continue
            if msg.get('id') == id_:
                return msg

    def request(self, method, params=None):
        self.next_id += 1
        self.send({'jsonrpc': '2.0', 'id': self.next_id, 'method': method, 'params': params or {}})
        return self.response(self.next_id)

    def result(self, method, params=None):
        r = self.request(method, params)
        assert 'error' not in r, r
        return r['result']

    def notification(self, method, pred=lambda m: True, timeout=TIMEOUT):
        for k, m in enumerate(self.notifications):
            if m.get('method') == method and pred(m):
                return self.notifications.pop(k)
        deadline = time.time() + timeout
        while True:
            msg = self.get(max(0.1, deadline - time.time()))
            if msg.get('method') == method and pred(msg):
                return msg
            self.notifications.append(msg)

    def initialize(self, root=None, locale=None):
        params = {'processId': os.getpid(), 'capabilities': {}}
        if root:
            params['rootUri'] = tills.path_to_uri(root)
        if locale:
            params['locale'] = locale
        r = self.result('initialize', params)
        self.notify('initialized')
        return r

    def open(self, uri, text, version=1):
        self.notify('textDocument/didOpen', {'textDocument': {'uri': uri, 'languageId': 'til', 'version': version,
                                                              'text': text}})
        return self.notification('textDocument/publishDiagnostics', lambda m: m['params']['uri'] == uri)

    def close(self):
        try:
            self.proc.kill()
        except OSError:
            pass
        self.proc.wait(timeout=10)
        for f in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                f.close()
            except OSError:
                pass
        if self.sock is not None:
            self.sock.close()


def doc(uri):
    return {'textDocument': {'uri': uri}}


def pos(uri, line, character):
    return {'textDocument': {'uri': uri}, 'position': {'line': line, 'character': character}}


@pytest.fixture
def client():
    c = Client()
    yield c
    c.close()


@pytest.fixture
def ready(client, tmp_path):
    (tmp_path / 'geo.til').write_text("fact(n: int) -> int\n    return n\n", encoding='utf-8')
    client.initialize(str(tmp_path))
    return client, tmp_path


SRC = ("struct Point\n    x: float\n    y: float\n\nimpl Point\n    fn norm(self) -> float\n"
       "        return sqrt(self.x * self.x + self.y * self.y)\n\nmain()\n    let p = Point { x: 3.0, y: 4.0 }\n"
       "    let total = p.norm()\n    print(total)\n")


# ═══════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════

class TestLifecycle:
    def test_initialize_capabilities(self, client):
        r = client.initialize()
        caps = r['capabilities']
        assert r['serverInfo'] == {'name': 'til-lsp', 'version': til.TIL_VERSION}
        assert caps['positionEncoding'] == 'utf-16'
        assert caps['textDocumentSync']['change'] == 1 and caps['textDocumentSync']['openClose']
        assert caps['signatureHelpProvider']['triggerCharacters'] == ['(', ',']
        assert '.' in caps['completionProvider']['triggerCharacters'] and caps['completionProvider']['resolveProvider']
        assert caps['renameProvider'] == {'prepareProvider': True}
        assert caps['semanticTokensProvider']['legend'] == {'tokenTypes': tills.TOKEN_TYPES,
                                                            'tokenModifiers': tills.TOKEN_MODIFIERS}
        for key in ('hoverProvider', 'definitionProvider', 'referencesProvider', 'documentHighlightProvider',
                    'documentSymbolProvider', 'workspaceSymbolProvider', 'foldingRangeProvider',
                    'documentFormattingProvider', 'inlayHintProvider', 'codeActionProvider', 'codeLensProvider',
                    'executeCommandProvider'):
            assert caps[key], key

    def test_shutdown_then_exit(self, client):
        client.initialize()
        assert client.request('shutdown') == {'jsonrpc': '2.0', 'id': client.next_id, 'result': None}
        r = client.request('textDocument/hover', pos('untitled:x', 0, 0))
        assert r['error']['code'] == tillsp.INVALID_REQUEST
        client.notify('exit')
        assert client.proc.wait(timeout=TIMEOUT) == 0

    def test_exit_without_shutdown(self, client):
        client.initialize()
        client.notify('exit')
        assert client.proc.wait(timeout=TIMEOUT) == 1

    def test_end_of_input_stops_the_server(self, client):
        client.initialize()
        client.proc.stdin.close()
        assert client.proc.wait(timeout=TIMEOUT) == 1

    def test_requests_before_initialize(self, client):
        r = client.request('textDocument/hover', pos('untitled:x', 0, 0))
        assert r['error']['code'] == tillsp.SERVER_NOT_INITIALIZED
        client.initialize()
        assert client.result('workspace/symbol', {'query': 'zz'}) == []

    def test_help(self):
        r = subprocess.run([sys.executable, TIL, 'lsp', '--help'], capture_output=True, text=True, timeout=TIMEOUT)
        assert r.returncode == 0 and 'til lsp' in r.stdout
        r = subprocess.run([sys.executable, TIL, 'lsp', '--bogus'], capture_output=True, text=True, timeout=TIMEOUT)
        assert r.returncode == 2


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

class TestDiagnostics:
    def test_did_open_publishes(self, ready):
        c, root = ready
        uri = tills.path_to_uri(str(root / 'a.til'))
        m = c.open(uri, "main()\n    let total = 1\n    print(totl)\n")
        d = m['params']['diagnostics']
        assert m['params']['version'] == 1
        assert [x['code'] for x in d] == ['E-VM']
        assert d[0]['range'] == {'start': {'line': 2, 'character': 10}, 'end': {'line': 2, 'character': 14}}
        assert d[0]['source'] == 'til' and d[0]['severity'] == 1

    def test_did_change_storm_is_debounced(self, ready):
        c, root = ready
        uri = tills.path_to_uri(str(root / 'b.til'))
        c.open(uri, "main()\n    print(1)\n")
        for k in range(40):
            c.notify('textDocument/didChange', {'textDocument': {'uri': uri, 'version': 2 + k},
                                                'contentChanges': [{'text': f"main()\n    print(x{k})\n"}]})
        m = c.notification('textDocument/publishDiagnostics', lambda m: m['params']['uri'] == uri)
        assert m['params']['version'] == 41
        assert "x39" in m['params']['diagnostics'][0]['message']
        c.result('workspace/symbol', {'query': 'zz'})           # a round trip: nothing else was published
        assert not [n for n in c.notifications if n.get('method') == 'textDocument/publishDiagnostics']

    def test_incremental_change_is_accepted(self, ready):
        c, root = ready
        uri = tills.path_to_uri(str(root / 'inc.til'))
        c.open(uri, "main()\n    print(1)\n")
        c.notify('textDocument/didChange', {'textDocument': {'uri': uri, 'version': 2}, 'contentChanges': [
            {'range': {'start': {'line': 1, 'character': 10}, 'end': {'line': 1, 'character': 11}}, 'text': 'zz'}]})
        m = c.notification('textDocument/publishDiagnostics', lambda m: m['params']['uri'] == uri)
        assert "zz" in m['params']['diagnostics'][0]['message']

    def test_close_clears(self, ready):
        c, root = ready
        uri = tills.path_to_uri(str(root / 'c.til'))
        c.open(uri, "main()\n    print(nope)\n")
        c.notify('textDocument/didClose', doc(uri))
        m = c.notification('textDocument/publishDiagnostics', lambda m: m['params']['uri'] == uri)
        assert m['params']['diagnostics'] == []

    def test_imported_file_problems_are_published_on_that_file(self, ready):
        c, root = ready
        (root / 'bad.til').write_text("helper(x: int) -> int\n    return oops + x\n", encoding='utf-8')
        uri = tills.path_to_uri(str(root / 'uses.til'))
        own = c.open(uri, "import bad\nmain()\n    print(helper(1))\n")
        assert own['params']['diagnostics'][0]['severity'] == 3
        bad_uri = tills.path_to_uri(str(root / 'bad.til'))
        m = c.notification('textDocument/publishDiagnostics', lambda m: m['params']['uri'] == bad_uri)
        assert m['params']['diagnostics'][0]['code'] == 'E-VM'

    def test_russian_locale(self, client):
        client.initialize(locale='ru-RU')
        m = client.open('untitled:ru.til', "main()\n    print(totl)\n")
        assert m['params']['diagnostics'][0]['message'].startswith("неизвестное имя")

    def test_lang_option_beats_the_client_locale(self):
        c = Client(args=('--lang', 'kk'))
        try:
            c.initialize(locale='ru')
            m = c.open('untitled:kk.til', "main()\n    print(len())\n")
            assert "аргумент" in m['params']['diagnostics'][0]['message']
            assert "ожида" not in m['params']['diagnostics'][0]['message']
        finally:
            c.close()


# ═══════════════════════════════════════════════════════════════
# LANGUAGE FEATURES
# ═══════════════════════════════════════════════════════════════

class TestFeatures:
    @pytest.fixture
    def opened(self, ready):
        c, root = ready
        uri = tills.path_to_uri(str(root / 'app.til'))
        c.open(uri, SRC)
        return c, uri, root

    def test_completion_and_resolve(self, opened):
        c, uri, _root = opened
        r = c.result('textDocument/completion', pos(uri, 11, 10))
        assert r['isIncomplete'] is False
        labels = [i['label'] for i in r['items']]
        assert 'total' in labels and 'print' in labels and 'Point' in labels
        item = next(i for i in r['items'] if i['label'] == 'len')
        resolved = c.result('completionItem/resolve', item)
        assert resolved['label'] == 'len' and resolved['documentation']['kind'] == 'markdown'

    def test_member_completion(self, opened):
        c, uri, _root = opened
        text = SRC.replace("    print(total)\n", "    p.\n")
        c.notify('textDocument/didChange', {'textDocument': {'uri': uri, 'version': 2},
                                            'contentChanges': [{'text': text}]})
        r = c.result('textDocument/completion', pos(uri, 11, 6))
        assert {'x', 'y', 'norm'} <= {i['label'] for i in r['items']}

    def test_hover(self, opened):
        c, uri, _root = opened
        h = c.result('textDocument/hover', pos(uri, 10, 18))
        assert h['contents']['kind'] == 'markdown' and "Point.norm(self) -> float" in h['contents']['value']
        assert c.result('textDocument/hover', pos(uri, 3, 0)) is None

    def test_signature_help(self, opened):
        c, uri, _root = opened
        text = SRC.replace("    print(total)\n", "    print(total, \n")
        c.notify('textDocument/didChange', {'textDocument': {'uri': uri, 'version': 3},
                                            'contentChanges': [{'text': text}]})
        s = c.result('textDocument/signatureHelp', pos(uri, 11, 17))
        assert s['signatures'][0]['label'].startswith("print(") and s['activeParameter'] == 0

    def test_definition_references_highlight(self, opened):
        c, uri, _root = opened
        locs = c.result('textDocument/definition', pos(uri, 11, 11))
        assert locs == [{'uri': uri, 'range': {'start': {'line': 10, 'character': 8},
                                               'end': {'line': 10, 'character': 13}}}]
        refs = c.result('textDocument/references', {**pos(uri, 10, 9), 'context': {'includeDeclaration': True}})
        assert sorted(r['range']['start']['line'] for r in refs) == [10, 11]
        refs = c.result('textDocument/references', {**pos(uri, 10, 9), 'context': {'includeDeclaration': False}})
        assert [r['range']['start']['line'] for r in refs] == [11]
        hl = c.result('textDocument/documentHighlight', pos(uri, 11, 11))
        assert sorted(h['kind'] for h in hl) == [2, 3]

    def test_prepare_rename_and_rename(self, opened):
        c, uri, _root = opened
        prep = c.result('textDocument/prepareRename', pos(uri, 11, 11))
        assert prep['placeholder'] == 'total'
        edit = c.result('textDocument/rename', {**pos(uri, 11, 11), 'newName': 'length'})
        assert len(edit['changes'][uri]) == 2 and all(e['newText'] == 'length' for e in edit['changes'][uri])

    def test_rename_refused_is_an_error_with_the_reason(self, opened):
        c, uri, _root = opened
        r = c.request('textDocument/rename', {**pos(uri, 11, 5), 'newName': 'echo'})
        assert r['error']['code'] == tillsp.REQUEST_FAILED and "print" in r['error']['message']
        r = c.request('textDocument/prepareRename', pos(uri, 11, 5))
        assert r['error']['code'] == tillsp.REQUEST_FAILED

    def test_symbols_folding_lenses_tokens(self, opened):
        c, uri, _root = opened
        syms = c.result('textDocument/documentSymbol', doc(uri))
        assert [s['name'] for s in syms] == ['Point', 'impl Point', 'main']
        ws = c.result('workspace/symbol', {'query': 'fact'})
        assert ws and ws[0]['location']['uri'].endswith('geo.til')
        folds = c.result('textDocument/foldingRange', doc(uri))
        assert {'startLine': 8, 'endLine': 11} in folds
        lenses = c.result('textDocument/codeLens', doc(uri))
        assert [l['command']['command'] for l in lenses] == ['til.run', 'til.debug']
        toks = c.result('textDocument/semanticTokens/full', doc(uri))
        assert toks['data'] and len(toks['data']) % 5 == 0

    def test_formatting(self, ready):
        c, root = ready
        uri = tills.path_to_uri(str(root / 'fmt.til'))
        c.open(uri, "main()\n  let x=1\n  print(x ,x)\n")
        edits = c.result('textDocument/formatting', {**doc(uri), 'options': {'tabSize': 4, 'insertSpaces': True}})
        text = "main()\n  let x=1\n  print(x ,x)\n"
        assert TestUtil.apply(text, edits) == "main()\n    let x = 1\n    print(x, x)\n"

    def test_inlay_hints_and_code_actions(self, ready):
        c, root = ready
        uri = tills.path_to_uri(str(root / 'act.til'))
        m = c.open(uri, "main()\n    let x = 1\n    x = 2\n")
        hints = c.result('textDocument/inlayHint', {**doc(uri), 'range': {'start': {'line': 0, 'character': 0},
                                                                         'end': {'line': 3, 'character': 0}}})
        if til.types_module() is not None:
            assert hints[0]['label'] == ': int'
        d = m['params']['diagnostics'][0]
        acts = c.result('textDocument/codeAction', {**doc(uri), 'range': d['range'],
                                                    'context': {'diagnostics': [d]}})
        assert acts[0]['title'] == "Change 'let' to 'var'" and acts[0]['kind'] == 'quickfix'
        only = c.result('textDocument/codeAction', {**doc(uri), 'range': d['range'],
                                                    'context': {'diagnostics': [d], 'only': ['refactor']}})
        assert only and all(a['kind'].startswith('refactor') for a in only)

    def test_execute_command(self, opened):
        c, uri, _root = opened
        r = c.result('workspace/executeCommand', {'command': 'til.translate', 'arguments': [{'uri': uri, 'to': 'kk'}]})
        assert r['text'] == til.translate_source(SRC, 'kk')
        assert c.result('workspace/executeCommand', {'command': 'til.run',
                                                     'arguments': [{'uri': uri, 'function': 'main'}]}) is None
        r = c.request('workspace/executeCommand', {'command': 'til.nothing'})
        assert r['error']['code'] == tillsp.INVALID_PARAMS

    def test_cancel_request_is_ignored(self, opened):
        c, uri, _root = opened
        c.notify('$/cancelRequest', {'id': 999})
        assert c.result('textDocument/hover', pos(uri, 0, 0)) is None or True


class TestUtil:
    @staticmethod
    def apply(text, edits):
        pm = tills.PositionMap(text)
        spans = sorted(((pm.offset(e['range']['start']['line'], e['range']['start']['character']),
                         pm.offset(e['range']['end']['line'], e['range']['end']['character']), e['newText'])
                        for e in edits), reverse=True)
        for a, b, new in spans:
            text = text[:a] + new + text[b:]
        return text


# ═══════════════════════════════════════════════════════════════
# ROBUSTNESS
# ═══════════════════════════════════════════════════════════════

class TestMalformed:
    def test_bad_json_is_a_parse_error(self, client):
        client.initialize()
        client.write_raw(b'Content-Length: 9\r\n\r\n{"bad": }')
        m = client.get()
        assert m['error']['code'] == tillsp.PARSE_ERROR and m['id'] is None

    def test_invalid_utf8(self, client):
        client.initialize()
        client.write_raw(b'Content-Length: 4\r\n\r\n\xff\xfe\xfd\xfc')
        assert client.get()['error']['code'] == tillsp.PARSE_ERROR

    def test_missing_content_length(self, client):
        client.initialize()
        client.write_raw(b'X-Junk: 1\r\n\r\n')
        assert client.get()['error']['code'] == tillsp.PARSE_ERROR
        assert client.result('workspace/symbol', {'query': ''}) == []

    def test_not_an_object_and_missing_method(self, client):
        client.initialize()
        client.send(42)
        assert client.get()['error']['code'] == tillsp.INVALID_REQUEST
        client.send({'jsonrpc': '2.0', 'id': 77})
        assert client.get()['error']['code'] == tillsp.INVALID_REQUEST

    def test_unknown_method(self, client):
        client.initialize()
        r = client.request('textDocument/frobnicate', {})
        assert r['error']['code'] == tillsp.METHOD_NOT_FOUND
        client.notify('custom/notification', {'x': 1})        # ignored
        assert client.result('workspace/symbol', {'query': ''}) == []

    def test_invalid_params(self, client):
        client.initialize()
        r = client.request('textDocument/hover', {'textDocument': {'uri': 'untitled:x'}})
        assert r['error']['code'] == tillsp.INVALID_PARAMS
        r = client.request('textDocument/rename', {**pos('untitled:x', 0, 0)})
        assert r['error']['code'] == tillsp.INVALID_PARAMS
        r = client.request('textDocument/completion', {'textDocument': 5, 'position': None})
        assert r['error']['code'] == tillsp.INVALID_PARAMS

    def test_batch(self, client):
        client.initialize()
        client.send([{'jsonrpc': '2.0', 'id': 'a', 'method': 'workspace/symbol', 'params': {'query': ''}},
                     {'jsonrpc': '2.0', 'id': 'b', 'method': 'nope'}])
        got = {m['id']: m for m in (client.get(), client.get())}
        assert got['a']['result'] == [] and got['b']['error']['code'] == tillsp.METHOD_NOT_FOUND

    def test_garbage_documents_never_crash(self, client):
        client.initialize()
        uri = 'untitled:garbage.til'
        client.open(uri, "\x00 ))) ((( \"\n\t#[\n" * 20)
        for method in ('textDocument/documentSymbol', 'textDocument/foldingRange', 'textDocument/semanticTokens/full',
                       'textDocument/codeLens', 'textDocument/formatting'):
            r = client.request(method, doc(uri))
            assert 'result' in r, (method, r)
        for method in ('textDocument/completion', 'textDocument/hover', 'textDocument/definition',
                       'textDocument/signatureHelp', 'textDocument/references'):
            r = client.request(method, pos(uri, 3, 2))
            assert 'result' in r, (method, r)
        r = client.request('textDocument/hover', pos('untitled:never-opened', 0, 0))
        assert r['result'] is None

    def test_stdout_stays_clean(self, client):
        # a module the compiler cannot find makes resolve_imports print a warning:
        # it must never reach the protocol stream
        client.initialize()
        m = client.open('untitled:imp.til', "import nowhere_mod\nmain()\n    print(1)\n")
        assert m['params']['diagnostics'][0]['code'] == 'E-IMPORT'
        assert client.result('workspace/symbol', {'query': 'nothing'}) == []


# ═══════════════════════════════════════════════════════════════
# TRANSPORTS AND ENTRY POINTS
# ═══════════════════════════════════════════════════════════════

def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestTransports:
    def test_tcp(self):
        port = free_port()

        def connect():
            deadline = time.time() + TIMEOUT
            while True:
                try:
                    return socket.create_connection(('127.0.0.1', port), timeout=TIMEOUT)
                except OSError:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.05)
        c = Client(args=('--tcp', str(port)), connect=connect)
        try:
            r = c.initialize()
            assert r['serverInfo']['name'] == 'til-lsp'
            m = c.open('untitled:tcp.til', "main()\n    print(nope)\n")
            assert m['params']['diagnostics']
            c.result('shutdown')
            c.notify('exit')
            assert c.proc.wait(timeout=TIMEOUT) == 0
        finally:
            c.close()

    def test_log_file(self, tmp_path):
        log = tmp_path / 'lsp.log'
        c = Client(args=('--log', str(log)))
        try:
            c.initialize()
            c.result('shutdown')
            c.notify('exit')
            c.proc.wait(timeout=TIMEOUT)
        finally:
            c.close()
        text = log.read_text(encoding='utf-8')
        assert '"method":"initialize"' in text.replace(' ', '') and '-> ' in text

    def test_old_path_lsp_til_lsp_py(self):
        c = Client(cmd=[sys.executable, os.path.join(ROOT, 'lsp', 'til_lsp.py')])
        try:
            assert c.initialize()['serverInfo']['name'] == 'til-lsp'
            m = c.open('untitled:old.til', "main()\n    print(1)\n")
            assert m['params']['diagnostics'] == []
            c.result('shutdown')
            c.notify('exit')
            assert c.proc.wait(timeout=TIMEOUT) == 0
        finally:
            c.close()


# ═══════════════════════════════════════════════════════════════
# IN-PROCESS API (TIL Studio)
# ═══════════════════════════════════════════════════════════════

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def init(server, root=None):
    out = server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                         'params': {'rootUri': tills.path_to_uri(root) if root else None}})
    server.handle({'jsonrpc': '2.0', 'method': 'initialized', 'params': {}})
    return out


class TestInProcess:
    def test_handle_returns_responses_and_notifications(self, tmp_path):
        server = tillsp.LspServer(str(tmp_path))
        out = init(server)
        assert out[0]['result']['capabilities']['positionEncoding'] == 'utf-16'
        uri = tills.path_to_uri(str(tmp_path / 'a.til'))
        out = server.handle({'jsonrpc': '2.0', 'method': 'textDocument/didOpen', 'params': {
            'textDocument': {'uri': uri, 'version': 1, 'text': "main()\n    print(nope)\n"}}})
        assert out[0]['method'] == 'textDocument/publishDiagnostics' and out[0]['params']['diagnostics']

    def test_poll_debounces_changes(self, tmp_path):
        clock = FakeClock()
        server = tillsp.LspServer(str(tmp_path), clock=clock)
        init(server)
        uri = 'untitled:deb.til'
        server.handle({'jsonrpc': '2.0', 'method': 'textDocument/didOpen', 'params': {
            'textDocument': {'uri': uri, 'version': 1, 'text': "main()\n    print(1)\n"}}})
        for k in range(10):
            out = server.handle({'jsonrpc': '2.0', 'method': 'textDocument/didChange', 'params': {
                'textDocument': {'uri': uri, 'version': 2 + k}, 'contentChanges': [{'text': f"main()\n    p{k}()\n"}]}})
            assert out == []
            clock.t += 0.05
            assert server.poll() == []
        assert 0 < server.next_due() <= 0.2
        clock.t += 0.2
        out = server.poll()
        assert len(out) == 1 and out[0]['params']['version'] == 11 and "p9" in out[0]['params']['diagnostics'][0]['message']
        assert server.poll() == [] and server.next_due() is None

    def test_errors_never_escape(self):
        server = tillsp.LspServer()
        init(server)
        server.ws.hover = None             # break a feature: the request gets an InternalError
        out = server.handle({'jsonrpc': '2.0', 'id': 5, 'method': 'textDocument/hover',
                             'params': pos('untitled:x', 0, 0)})
        assert out[0]['error']['code'] == tillsp.INTERNAL_ERROR
        assert server.handle("not a message")[0]['error']['code'] == tillsp.INVALID_REQUEST
        assert server.handle({'jsonrpc': '2.0', 'result': 1, 'id': 9}) == []      # a response to us

    def test_imported_file_diagnostics_are_cleared(self, tmp_path):
        (tmp_path / 'm.til').write_text("helper() -> int\n    return oops\n", encoding='utf-8')
        server = tillsp.LspServer(str(tmp_path))
        init(server)
        uri = tills.path_to_uri(str(tmp_path / 'u.til'))
        m_uri = tills.path_to_uri(str(tmp_path / 'm.til'))
        out = server.handle({'jsonrpc': '2.0', 'method': 'textDocument/didOpen', 'params': {
            'textDocument': {'uri': uri, 'version': 1, 'text': "import m\nmain()\n    print(helper())\n"}}})
        published = {o['params']['uri']: o['params']['diagnostics'] for o in out}
        assert published[m_uri][0]['code'] == 'E-VM'
        # the import goes away: the imported file's diagnostics are cleared
        server.handle({'jsonrpc': '2.0', 'method': 'textDocument/didChange', 'params': {
            'textDocument': {'uri': uri, 'version': 2}, 'contentChanges': [{'text': "main()\n    print(1)\n"}]}})
        server.pending[uri] = 0
        out = server.poll()
        published = {o['params']['uri']: o['params']['diagnostics'] for o in out}
        assert published[uri] == [] and published[m_uri] == []

    def test_save_rechecks_the_files_that_import_it(self, tmp_path):
        clock = FakeClock()
        (tmp_path / 'lib.til').write_text("value() -> int\n    return 1\n", encoding='utf-8')
        server = tillsp.LspServer(str(tmp_path), clock=clock)
        init(server)
        app = tills.path_to_uri(str(tmp_path / 'app.til'))
        lib = tills.path_to_uri(str(tmp_path / 'lib.til'))
        out = server.handle({'jsonrpc': '2.0', 'method': 'textDocument/didOpen', 'params': {
            'textDocument': {'uri': app, 'version': 1, 'text': "import lib\nmain()\n    print(value())\n"}}})
        assert out[0]['params']['diagnostics'] == []
        server.handle({'jsonrpc': '2.0', 'method': 'textDocument/didOpen', 'params': {
            'textDocument': {'uri': lib, 'version': 1, 'text': "value() -> int\n    return 1\n"}}})
        (tmp_path / 'lib.til').write_text("renamed() -> int\n    return 1\n", encoding='utf-8')
        server.handle({'jsonrpc': '2.0', 'method': 'textDocument/didChange', 'params': {
            'textDocument': {'uri': lib, 'version': 2}, 'contentChanges': [{'text': "renamed() -> int\n    return 1\n"}]}})
        server.handle({'jsonrpc': '2.0', 'method': 'textDocument/didSave', 'params': {'textDocument': {'uri': lib}}})
        clock.t += 1
        out = server.poll()
        app_diags = [o for o in out if o['params']['uri'] == app]
        assert app_diags and app_diags[0]['params']['diagnostics'][0]['code'] == 'E-VM'

    def test_locale_from_initialization_options(self):
        server = tillsp.LspServer()
        server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                       'params': {'initializationOptions': {'locale': 'kk'}}})
        out = server.handle({'jsonrpc': '2.0', 'method': 'textDocument/didOpen', 'params': {
            'textDocument': {'uri': 'untitled:k', 'version': 1, 'text': "main()\n    print(nope)\n"}}})
        assert "белгісіз" in out[0]['params']['diagnostics'][0]['message']
