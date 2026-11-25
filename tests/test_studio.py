#!/usr/bin/env python3
"""
TIL Studio server (src/tilstudio.py, docs/dev/TIL_STUDIO_DESIGN.md section 5).

A real server on a free port of 127.0.0.1 over a temporary project:

* security: the token (header, query, cookie), the Host and Origin checks,
  paths confined to the project (`..`, absolute paths, symlinks to outside);
* files: tree, read, write with conflict detection, create, rename, delete,
  search, file-change events;
* the LSP bridge (diagnostics arrive as events) and the DAP bridge, with
  stand-in modules that implement the documented in-process interfaces
  (LspServer(root).handle/poll, DapSession(emit, root).handle/close); the real
  src/tillsp.py and src/tildap.py too when they are installed;
* runs on the vm, cvm and native backends: streamed output in order, stdin,
  stop, exit codes, screen frames and keys, the output limit, the limit of
  concurrent runs, portable snapshots;
* tools: dis, translate, fit, build, pkg, prove (with z3 only);
* shutdown: no threads or processes are left behind; `til ide` itself.

Run: python3 -m pytest tests/test_studio.py -v
"""

import base64
import http.client
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from tilhelpers import CC, ROOT, SRC, TIL, needs_cc, til

import tilstudio  # noqa: E402  (src/ is on sys.path through tilhelpers)

# ═══════════════════════════════════════════════════════════════════════════
# stand-ins for src/tillsp.py and src/tildap.py (the documented interfaces)
# ═══════════════════════════════════════════════════════════════════════════

STANDIN_LSP = r'''
import time

class LspServer:
    instances = []

    def __init__(self, root=None):
        self.root = root
        self.docs = {}
        self.due = {}
        self.inits = 0
        self.methods = []
        LspServer.instances.append(self)

    def handle(self, message):
        m, mid = message.get('method'), message.get('id')
        self.methods.append(m)
        if m == 'initialize':
            self.inits += 1
            if self.inits > 1:
                return [{'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32600, 'message': 'initialized twice'}}]
            return [{'jsonrpc': '2.0', 'id': mid, 'result': {
                'capabilities': {'hoverProvider': True, 'positionEncoding': 'utf-16'},
                'serverInfo': {'name': 'til-lsp', 'version': 'standin'}}}]
        if m in ('textDocument/didOpen', 'textDocument/didChange'):
            p = message['params']
            uri = p['textDocument']['uri']
            text = p['textDocument']['text'] if m.endswith('didOpen') else p['contentChanges'][-1]['text']
            self.docs[uri] = text
            self.due[uri] = time.monotonic() + 0.2          # debounced, like tillsp
            return [{'jsonrpc': '2.0', 'method': 'window/logMessage',
                     'params': {'type': 3, 'message': 'opened ' + uri}}]
        if m == 'textDocument/didClose':
            self.docs.pop(message['params']['textDocument']['uri'], None)
            return []
        if m == 'textDocument/hover':
            return [{'jsonrpc': '2.0', 'id': mid, 'result': {'contents': {'kind': 'markdown', 'value': '`x: int`'}}}]
        if m == 'boom':
            raise RuntimeError('boom')
        if mid is not None and m is not None:
            return [{'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32601, 'message': 'MethodNotFound'}}]
        return []

    def poll(self):
        out, now = [], time.monotonic()
        for uri, t in list(self.due.items()):
            if now >= t:
                del self.due[uri]
                diags = []
                for i, line in enumerate(self.docs.get(uri, '').split('\n')):
                    if 'oops' in line:
                        c = line.index('oops')
                        diags.append({'range': {'start': {'line': i, 'character': c},
                                                'end': {'line': i, 'character': c + 4}},
                                      'severity': 1, 'code': 'E-VM', 'source': 'til', 'message': "unknown name 'oops'"})
                out.append({'jsonrpc': '2.0', 'method': 'textDocument/publishDiagnostics',
                            'params': {'uri': uri, 'diagnostics': diags}})
        return out
'''

STANDIN_DAP = r'''
import threading

class DapSession:
    instances = []

    def __init__(self, emit, root=None):
        self.emit = emit
        self.root = root
        self.closed = False
        self.seq = 0
        self.lock = threading.Lock()
        DapSession.instances.append(self)

    def _next(self):
        with self.lock:
            self.seq += 1
            return self.seq

    def event(self, name, body=None):
        self.emit({'seq': self._next(), 'type': 'event', 'event': name, 'body': body or {}})

    def response(self, req, body=None):
        return {'seq': self._next(), 'type': 'response', 'request_seq': req.get('seq', 0), 'success': True,
                'command': req.get('command'), 'body': body or {}}

    def handle(self, message):
        cmd = message.get('command')
        if cmd == 'initialize':
            threading.Timer(0.05, self.event, ('initialized',)).start()      # events from another thread
            return [self.response(message, {'supportsStepBack': True, 'supportsConfigurationDoneRequest': True})]
        if cmd == 'launch':
            def program():
                self.event('output', {'category': 'stdout', 'output': 'hello from ' + message['arguments']['program'] + '\n'})
                self.event('terminated')
            threading.Thread(target=program).start()
            return [self.response(message)]
        if cmd == 'boom':
            raise RuntimeError('dap boom')
        return [self.response(message)]

    def close(self):
        self.closed = True
'''


def load_standin(name: str, source: str, folder: str):
    path = os.path.join(folder, f"standin_{name}.py")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(f"standin_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════════════════════
# programs of the test project
# ═══════════════════════════════════════════════════════════════════════════

PROGRAMS = {
    'hello.til': 'main()\n    print("Hello, Studio!")\n',
    'greet.til': ('main()\n    print("start")\n    let name = input("Name? ")\n    print("Hello " + name)\n'
                  '    let more = input()\n    print("then " + more)\n'),
    'count.til': 'main()\n    for i in 0..3000\n        print(i)\n',
    'slow.til': ('main()\n    for i in 0..4\n        print(i)\n        sleep_ms(150)\n'
                 '    let name = input("Name? ")\n    print("Hello " + name)\n    exit(3)\n'),
    'loop.til': 'main()\n    var x = 0\n    while true\n        x = x + 1\n',
    'exit3.til': 'main()\n    print("bye")\n    exit(3)\n',
    'broken.til': 'main()\n    print(oops)\n',
    'crash.til': 'main()\n    let a = [1, 2]\n    print("before")\n    print(a[5])\n',
    'args.til': 'main()\n    print(args())\n',
    'sleepy.til': 'main()\n    print("zz")\n    sleep_ms(100000)\n',
    'big.til': 'main()\n    let line = "x" * 1023\n    for i in 0..5000\n        print(line)\n    print("done")\n',
    'gfx.til': ('main()\n    screen_init(4, 2)\n    screen_clear(0)\n    draw_pixel(0, 0, 0xFF0000)\n'
                '    screen_show()\n    let k = wait_key()\n    print(f"key {k}")\n'
                '    fill_rect(0, 0, 4, 2, 0x00FF00)\n    screen_show()\n'),
    'poll.til': ('main()\n    screen_init(2, 2)\n    var n = 0\n    while n < 3\n        let k = key_pressed()\n'
                 '        if k != 0\n            n = n + 1\n            print(k)\n        sleep_ms(5)\n'),
    'frames.til': 'main()\n    screen_init(8, 8)\n    for i in 0..300\n        screen_clear(i)\n        screen_show()\n',
    'long.til': ('main()\n    var total = 0\n    var i = 0\n    while true\n        total = total + i\n'
                 '        i = i + 1\n        if i % 20000 == 0\n            print(i)\n'),
    'mixed.til': 'main()\n    let mixed = [1, "a", 2.5]\n    print(mixed)\n',
    'proofs.til': ('#[requires: x >= 0 and x < 1000000]\n#[ensures: result >= x]\ndouble(x: int) -> int\n'
                   '    return x + x\n\n'
                   '#[ensures: result > 0]\nbad(x: int) -> int\n    return x\n\nmain()\n    print(double(2))\n'),
    'sub/inner.til': 'main()\n    print("inner")\n',
    'sub/data.txt': 'Alpha beta\nGAMMA alpha\n😀 alpha\n',
}


@pytest.fixture(scope='module')
def project(tmp_path_factory):
    base = str(tmp_path_factory.mktemp('studio'))
    root = os.path.join(base, 'proj')
    outside = os.path.join(base, 'outside')
    os.makedirs(outside)
    with open(os.path.join(outside, 'secret.txt'), 'w') as f:
        f.write('top secret\n')
    for rel, text in PROGRAMS.items():
        path = os.path.join(root, *rel.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    os.makedirs(os.path.join(root, '.git'))
    with open(os.path.join(root, '.git', 'HEAD'), 'w') as f:
        f.write('ref: refs/heads/main\n')
    os.makedirs(os.path.join(root, '__pycache__'))
    os.makedirs(os.path.join(root, 'til_modules', 'pkg'))
    with open(os.path.join(root, 'til_modules', 'pkg', 'pkg.til'), 'w') as f:
        f.write('hello() -> str\n    return "pkg alpha"\n')
    with open(os.path.join(root, 'image.bin'), 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00')
    with open(os.path.join(root, 'latin1.txt'), 'wb') as f:
        f.write('caf\xe9\n'.encode('latin-1'))
    try:
        os.symlink(outside, os.path.join(root, 'link_out'))
        os.symlink(os.path.join(outside, 'secret.txt'), os.path.join(root, 'secret_link.txt'))
        os.symlink(os.path.join(root, 'sub'), os.path.join(root, 'link_in'))
    except (OSError, NotImplementedError):
        pass
    return root


@pytest.fixture(scope='module')
def standins(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp('standins'))
    return {'tillsp': load_standin('tillsp', STANDIN_LSP, folder),
            'tildap': load_standin('tildap', STANDIN_DAP, folder)}


# ═══════════════════════════════════════════════════════════════════════════
# a small client
# ═══════════════════════════════════════════════════════════════════════════

class Response:
    def __init__(self, status, data, headers):
        self.status, self.data, self.headers = status, data, headers

    def json(self):
        return json.loads(self.data.decode('utf-8'))

    def __repr__(self):
        return f"<{self.status} {self.data[:200]!r}>"


class Api:
    def __init__(self, server: tilstudio.StudioServer):
        self.server = server
        self.port = server.port
        self.token = server.token

    def request(self, method, path, body=None, headers=None, token=True, raw=None, host=None, timeout=120):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        h = {}
        if token:
            h['X-TIL-Token'] = self.token
        if host is not None:
            h['Host'] = host
        h.update(headers or {})
        data = raw if raw is not None else (json.dumps(body).encode('utf-8') if body is not None else None)
        if data is not None:
            h.setdefault('Content-Type', 'application/json')
        conn.request(method, path, body=data, headers=h)
        r = conn.getresponse()
        out = Response(r.status, r.read(), r.getheaders())
        conn.close()
        return out

    def get(self, path, **kw):
        return self.request('GET', path, **kw)

    def post(self, path, body=None, **kw):
        return self.request('POST', path, body if body is not None else {}, **kw)

    def put(self, path, body, **kw):
        return self.request('PUT', path, body, **kw)

    def ok(self, method, path, body=None):
        r = self.request(method, path, body)
        assert r.status == 200, r
        return r.json()

    def events(self) -> 'Events':
        return Events(self)

    def run(self, events: 'Events', timeout=90, **body):
        r = self.post('/api/run', body)
        assert r.status == 200, r
        rid = r.json()['runId']
        return rid, events.until_exit(rid, timeout)


class Events:
    """One event stream (GET /api/events) read by a thread."""

    def __init__(self, api: Api, token=True):
        self.conn = http.client.HTTPConnection('127.0.0.1', api.port, timeout=300)
        self.conn.request('GET', '/api/events', headers={'X-TIL-Token': api.token} if token else {})
        self.sock = self.conn.sock              # (the response takes the socket over)
        self.response = self.conn.getresponse()
        self.status = self.response.status
        self.items = []
        self.cond = threading.Condition()
        self.ended = False
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        if self.status == 200:
            self.wait(lambda name, data: name == 'ping', 10)

    def _read(self):
        name, data = None, []
        fp = self.response.fp                   # (close() clears response.fp)
        try:
            while fp is not None:
                line = fp.readline()
                if not line:
                    break
                line = line.decode('utf-8').rstrip('\n')
                if line.startswith('event:'):
                    name = line[6:].strip()
                elif line.startswith('data:'):
                    data.append(line[5:].strip())
                elif line == '':
                    if name is not None:
                        with self.cond:
                            self.items.append((time.monotonic(), name, json.loads('\n'.join(data))))
                            self.cond.notify_all()
                    name, data = None, []
        except (OSError, ValueError, AttributeError):
            pass
        finally:
            with self.cond:
                self.ended = True
                self.cond.notify_all()

    def wait(self, pred, timeout=10.0, start=0):
        """The first event (name, data) at index >= start that matches."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                for _t, name, data in self.items[start:]:
                    if pred(name, data):
                        return data
                left = deadline - time.monotonic()
                if left <= 0 or self.ended:
                    raise AssertionError(f"no matching event within {timeout} s; got "
                                         f"{[(n, str(d)[:120]) for _t, n, d in self.items[-15:]]}")
                self.cond.wait(left)

    def run_events(self, rid):
        with self.cond:
            return [d for _t, n, d in self.items if n == 'run' and d.get('runId') == rid]

    def timed(self, rid):
        with self.cond:
            return [(t, d) for t, n, d in self.items if n == 'run' and d.get('runId') == rid]

    def until_exit(self, rid, timeout=90):
        self.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d['kind'] == 'exit', timeout)
        return self.run_events(rid)

    def mark(self) -> int:
        with self.cond:
            return len(self.items)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.response.close()
        self.conn.close()
        self.thread.join(5)


def text_of(evs, kind='stdout'):
    return "".join(e.get('text', '') for e in evs if e['kind'] == kind)


def exit_of(evs):
    return [e for e in evs if e['kind'] == 'exit'][-1]


@pytest.fixture(scope='module')
def studio(project, standins):
    server = tilstudio.StudioServer(project, frontend=False, modules=standins, log=open(os.devnull, 'w'))
    server.start()
    api = Api(server)
    yield api
    server.stop()


@pytest.fixture
def events(studio):
    ev = studio.events()
    yield ev
    ev.close()


def needs_symlinks(project):
    if not os.path.islink(os.path.join(project, 'link_out')):
        pytest.skip("no symlinks on this system")


# ═══════════════════════════════════════════════════════════════════════════
# security
# ═══════════════════════════════════════════════════════════════════════════

class TestSecurity:
    def test_token_required(self, studio):
        for method, path in (('GET', '/api/info'), ('GET', '/api/tree'), ('GET', '/api/file?path=hello.til'),
                             ('GET', '/api/events'), ('POST', '/api/run'), ('POST', '/api/search'),
                             ('POST', '/api/lsp'), ('GET', '/')):
            r = studio.request(method, path, {} if method == 'POST' else None, token=False)
            assert r.status == 403, (path, r)
        r = studio.get('/api/info', token=False, headers={'X-TIL-Token': 'wrong'})
        assert r.status == 403 and 'token' in r.json()['error']
        r = studio.get('/api/info', token=False, headers={'Cookie': f"{studio.server.cookie_name}=wrong"})
        assert r.status == 403
        assert studio.get('/api/info').status == 200
        assert studio.get(f'/api/info?token={studio.token}', token=False).status == 200

    def test_app_sets_the_cookie(self, studio):
        r = studio.get(f'/?token={studio.token}', token=False)
        assert r.status == 200 and b'TIL Studio' in r.data
        headers = dict((k.lower(), v) for k, v in r.headers)
        assert headers['content-type'].startswith('text/html')
        cookie = headers['set-cookie']
        assert cookie.startswith(f"{studio.server.cookie_name}={studio.token}")
        assert 'HttpOnly' in cookie and 'SameSite=Strict' in cookie
        assert "frame-ancestors 'none'" in headers['content-security-policy']
        jar = cookie.split(';')[0]
        assert studio.get('/api/info', token=False, headers={'Cookie': jar}).status == 200
        assert studio.get('/', token=False, headers={'Cookie': jar}).status == 200
        assert studio.get('/?token=nope', token=False).status == 403

    def test_host_header(self, studio):
        for host in ('evil.example', f'evil.example:{studio.port}', f'127.0.0.1:{studio.port + 1}',
                     f'127.0.0.1.evil.example:{studio.port}', ''):
            r = studio.get('/api/info', host=host)
            assert r.status == 403, (host, r)
        assert studio.get('/api/info', host=f'localhost:{studio.port}').status == 200
        assert studio.get('/api/info', host=f'LOCALHOST:{studio.port}').status == 200
        # no Host header at all
        conn = http.client.HTTPConnection('127.0.0.1', studio.port, timeout=10)
        conn.putrequest('GET', '/api/info', skip_host=True)
        conn.putheader('X-TIL-Token', studio.token)
        conn.endheaders()
        assert conn.getresponse().status == 403
        conn.close()

    def test_origin(self, studio):
        body = {'query': 'alpha'}
        for origin in ('http://evil.example', f'http://127.0.0.1:{studio.port + 1}', 'null',
                       f'https://127.0.0.1:{studio.port}'):
            r = studio.post('/api/search', body, headers={'Origin': origin})
            assert r.status == 403 and 'origin' in r.json()['error'], (origin, r)
        r = studio.post('/api/search', body, headers={'Sec-Fetch-Site': 'cross-site'})
        assert r.status == 403
        r = studio.post('/api/search', body, headers={'Sec-Fetch-Site': 'same-site'})
        assert r.status == 403
        for origin in (f'http://127.0.0.1:{studio.port}', f'http://localhost:{studio.port}'):
            assert studio.post('/api/search', body, headers={'Origin': origin}).status == 200
        assert studio.post('/api/search', body, headers={'Sec-Fetch-Site': 'same-origin'}).status == 200
        # a cross-origin read of a GET endpoint is refused too
        assert studio.get('/api/info', headers={'Origin': 'http://evil.example'}).status == 403
        # no CORS headers, ever
        r = studio.request('OPTIONS', '/api/run', headers={'Origin': 'http://evil.example',
                                                           'Access-Control-Request-Method': 'POST'})
        assert r.status == 403
        assert not any(k.lower().startswith('access-control') for k, _v in r.headers)

    @pytest.mark.parametrize('path', ['../outside/secret.txt', 'sub/../../outside/secret.txt', '..', '/etc/passwd',
                                      '//etc/passwd', 'sub/../..', '@examples/../../outside/secret.txt'])
    def test_traversal_refused(self, studio, path):
        q = urllib_quote(path)
        r = studio.get(f'/api/file?path={q}')
        assert r.status == 403, r
        assert studio.get(f'/api/tree?path={q}').status == 403
        assert studio.put('/api/file', {'path': path, 'text': 'x'}).status == 403
        assert studio.post('/api/file/create', {'path': path + '/new.txt', 'type': 'file'}).status == 403
        assert studio.post('/api/file/delete', {'path': path}).status in (400, 403)
        assert studio.post('/api/file/rename', {'path': 'hello.til', 'to': path + '/x.til'}).status == 403
        assert studio.post('/api/run', {'path': path}).status == 403

    def test_nul_and_bad_types(self, studio):
        assert studio.get('/api/file?path=hello.til%00.txt').status == 400
        assert studio.put('/api/file', {'path': 5, 'text': 'x'}).status == 400
        assert studio.put('/api/file', {'path': 'x.til', 'text': None}).status == 400

    def test_symlink_escape(self, studio, project):
        needs_symlinks(project)
        for path in ('link_out/secret.txt', 'secret_link.txt', 'link_out'):
            assert studio.get(f'/api/file?path={path}').status == 403
        assert studio.get('/api/tree?path=link_out').status == 403
        assert studio.put('/api/file', {'path': 'link_out/new.txt', 'text': 'pwned'}).status == 403
        assert studio.post('/api/file/create', {'path': 'link_out/new.til', 'type': 'file'}).status == 403
        assert studio.post('/api/file/rename', {'path': 'hello.til', 'to': 'link_out/hello.til'}).status == 403
        assert not os.path.exists(os.path.join(os.path.dirname(project), 'outside', 'new.txt'))
        names = [e['name'] for e in studio.ok('GET', '/api/tree')['entries']]
        assert 'link_out' not in names and 'secret_link.txt' not in names
        # a link that stays inside the project works
        assert 'link_in' in names
        assert studio.ok('GET', '/api/file?path=link_in/inner.til')['text'] == PROGRAMS['sub/inner.til']

    def test_root_cannot_be_deleted(self, studio, project):
        for path in ('', '.', './'):
            assert studio.post('/api/file/delete', {'path': path}).status == 400
            assert studio.post('/api/file/rename', {'path': path, 'to': 'elsewhere'}).status == 400
        assert os.path.isdir(project)

    def test_errors_are_json_without_traceback(self, studio, monkeypatch):
        r = studio.request('POST', '/api/search', raw=b'{not json')
        assert r.status == 400 and 'error' in r.json()
        r = studio.request('POST', '/api/search', raw=b'[1, 2]')
        assert r.status == 400
        assert studio.get('/api/nothing').status == 404
        assert studio.request('DELETE', '/api/info').status == 405

        def boom(body, lang):
            raise RuntimeError("kaboom")
        monkeypatch.setattr(studio.server, 'api_search', boom)
        r = studio.post('/api/search', {'query': 'x'})
        assert r.status == 500
        error = r.json()['error']
        assert 'kaboom' in error and 'Traceback' not in error and 'File "' not in error

    def test_request_limits(self, studio):
        conn = http.client.HTTPConnection('127.0.0.1', studio.port, timeout=10)
        conn.putrequest('POST', '/api/search')
        conn.putheader('X-TIL-Token', studio.token)
        conn.putheader('Content-Length', str(tilstudio.BODY_LIMIT + 1))
        conn.endheaders()
        r = conn.getresponse()
        assert r.status == 413 and 'error' in json.loads(r.read())
        conn.close()
        r = studio.request('HEAD', '/api/info')
        assert r.status == 405 and r.data == b''
        conn = http.client.HTTPConnection('127.0.0.1', studio.port, timeout=10)
        conn.putrequest('GET', '/api/info')
        conn.putheader('X-TIL-Token', studio.token)
        conn.putheader('Transfer-Encoding', 'chunked')
        conn.endheaders()
        conn.send(b'0\r\n\r\n')
        assert conn.getresponse().status == 200          # (a GET body is not read)
        conn.close()
        # a malformed request line: a JSON error too
        with socket.create_connection(('127.0.0.1', studio.port), timeout=10) as sock:
            sock.sendall(b'GARBAGE\r\n\r\n')
            data = sock.recv(65536)
        assert b'"error"' in data and b'Traceback' not in data     # (Python 3.9: an HTTP/0.9 reply, no status line)

    def test_messages_in_the_request_language(self, studio):
        r = studio.get('/api/file?path=../x', headers={'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8'})
        assert r.status == 403 and 'проект' in r.json()['error']
        r = studio.get('/api/file?path=../x', headers={'X-TIL-Lang': 'kk'})
        assert r.status == 403 and 'жоба' in r.json()['error']
        r = studio.get('/api/file?path=../x', headers={'Accept-Language': 'de-DE'})
        assert r.json()['error'].startswith('path outside the project')


def urllib_quote(path):
    import urllib.parse
    return urllib.parse.quote(path, safe='')


# ═══════════════════════════════════════════════════════════════════════════
# info, files, search, file events
# ═══════════════════════════════════════════════════════════════════════════

class TestFiles:
    def test_info(self, studio, project):
        info = studio.ok('GET', '/api/info')
        assert info['version'] == til.TIL_VERSION
        assert info['root'] == os.path.realpath(project)
        assert info['rootName'] == 'proj'
        assert info['platform'] == sys.platform
        caps = info['capabilities']
        assert caps['debugger'] is True and caps['pkg'] is True
        assert caps['native'] == (CC is not None)
        assert set(caps) >= {'native', 'cvm', 'z3', 'debugger', 'pkg'}
        names = [d['name'] for d in info['devices']]
        assert 'desktop' in names and all(d['title'] for d in info['devices'])
        assert info['tables']['natives'] == list(til.NATIVES)
        assert info['tables']['methods'] == list(til.METHODS)
        ex = {e['name']: e['path'] for e in info['examples']}
        assert ex['01_hello'] == '@examples/01_hello.til'
        r = studio.ok('GET', '/api/file?path=@examples/01_hello.til')
        assert 'Hello, World!' in r['text'] and r['readonly'] is True
        assert studio.put('/api/file', {'path': '@examples/01_hello.til', 'text': 'x'}).status == 403
        assert studio.post('/api/file/delete', {'path': '@examples/01_hello.til'}).status == 403
        tree = studio.ok('GET', '/api/tree?path=@examples')
        assert any(e['path'] == '@examples/01_hello.til' for e in tree['entries'])

    def test_tables_agree_with_the_frontend(self, studio):
        """/api/info serves the dict of ide/studio/tables.json (built by gen_tables.py)."""
        tables = studio.ok('GET', '/api/info')['tables']
        gen = os.path.join(ROOT, 'ide', 'studio', 'gen_tables.py')
        built = os.path.join(ROOT, 'ide', 'studio', 'tables.json')
        if os.path.isfile(gen):
            spec = importlib.util.spec_from_file_location('gen_tables_for_test', gen)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert tables == json.loads(json.dumps(module.build_tables()))
        elif os.path.isfile(built):
            with open(built, encoding='utf-8') as f:
                assert tables == json.load(f)
        assert tables['native_arity']['print'] == [0, None]
        assert tables['keywords']['ru']['если'] == 'IF' and tables['key_names']['ENTER'] == 13

    def test_tree(self, studio):
        tree = studio.ok('GET', '/api/tree')
        entries = tree['entries']
        names = [e['name'] for e in entries]
        assert '.git' not in names and '__pycache__' not in names
        kinds = [e['type'] for e in entries]
        assert kinds == sorted(kinds, key=lambda k: k != 'dir')          # folders first
        files = [e['name'] for e in entries if e['type'] == 'file']
        assert files == sorted(files, key=str.casefold)
        mods = next(e for e in entries if e['name'] == 'til_modules')
        assert mods['type'] == 'dir' and mods['collapsed'] is True
        hello = next(e for e in entries if e['name'] == 'hello.til')
        assert hello == {'name': 'hello.til', 'path': 'hello.til', 'type': 'file',
                         'size': len(PROGRAMS['hello.til']), 'mtime': hello['mtime']}
        sub = studio.ok('GET', '/api/tree?path=sub')['entries']
        assert {e['path'] for e in sub} == {'sub/inner.til', 'sub/data.txt'}
        assert studio.get('/api/tree?path=nope').status == 404
        assert studio.get('/api/tree?path=hello.til').status == 400

    def test_read(self, studio):
        r = studio.ok('GET', '/api/file?path=hello.til')
        assert r['path'] == 'hello.til' and r['text'] == PROGRAMS['hello.til']
        assert r['size'] == len(PROGRAMS['hello.til']) and isinstance(r['mtime'], float)
        assert studio.ok('GET', '/api/file?path=./sub//inner.til')['path'] == 'sub/inner.til'
        assert studio.get('/api/file?path=image.bin').status == 415
        assert studio.get('/api/file?path=latin1.txt').status == 415
        assert studio.get('/api/file?path=sub').status == 400
        assert studio.get('/api/file?path=missing.til').status == 404
        assert studio.get('/api/file').status == 400

    def test_create_write_conflict_rename_delete(self, studio, project):
        r = studio.ok('POST', '/api/file/create', {'path': 'work/deep/a.til', 'type': 'file', 'text': 'v1\n'})
        assert r['path'] == 'work/deep/a.til'
        assert studio.post('/api/file/create', {'path': 'work/deep/a.til', 'type': 'file'}).status == 409
        got = studio.ok('GET', '/api/file?path=work/deep/a.til')
        assert got['text'] == 'v1\n'
        # write with the mtime that was read
        w = studio.ok('PUT', '/api/file', {'path': 'work/deep/a.til', 'text': 'v2 ✓\r\n', 'mtime': got['mtime']})
        with open(os.path.join(project, 'work', 'deep', 'a.til'), 'rb') as f:
            assert f.read() == 'v2 ✓\r\n'.encode('utf-8')            # exactly the text: no newline translation
        # the old mtime is stale now: 409
        time.sleep(0.01)
        r = studio.put('/api/file', {'path': 'work/deep/a.til', 'text': 'v3', 'mtime': got['mtime'] - 5})
        assert r.status == 409 and 'changed' in r.json()['error']
        # changed on disk behind the studio's back
        path = os.path.join(project, 'work', 'deep', 'a.til')
        os.utime(path, (w['mtime'] + 10, w['mtime'] + 10))
        assert studio.put('/api/file', {'path': 'work/deep/a.til', 'text': 'v3', 'mtime': w['mtime']}).status == 409
        # without mtime: written; a new file too
        assert studio.put('/api/file', {'path': 'work/deep/a.til', 'text': 'v4'}).status == 200
        assert studio.put('/api/file', {'path': 'work/new.til', 'text': 'n'}).status == 200
        assert studio.put('/api/file', {'path': 'work/gone/new.til', 'text': 'n'}).status == 404
        assert studio.put('/api/file', {'path': 'work/x.til', 'text': 'n', 'mtime': 1.0}).status == 409
        # permissions are kept by the atomic save
        if os.name == 'posix':
            os.chmod(path, 0o755)
            studio.ok('PUT', '/api/file', {'path': 'work/deep/a.til', 'text': 'v5'})
            assert os.stat(path).st_mode & 0o777 == 0o755
        assert not [n for n in os.listdir(os.path.dirname(path)) if n.endswith(tilstudio.TEMP_SUFFIX)]
        # folders, rename, delete
        assert studio.ok('POST', '/api/file/create', {'path': 'work/empty', 'type': 'dir'}) == {'path': 'work/empty'}
        assert studio.post('/api/file/create', {'path': 'work/empty', 'type': 'dir'}).status == 409
        assert studio.ok('POST', '/api/file/rename', {'path': 'work/deep/a.til', 'to': 'work/b.til'}) == \
            {'path': 'work/b.til'}
        assert studio.get('/api/file?path=work/deep/a.til').status == 404
        assert studio.ok('GET', '/api/file?path=work/b.til')['text'] == 'v5' if os.name == 'posix' else True
        assert studio.post('/api/file/rename', {'path': 'work/b.til', 'to': 'work/new.til'}).status == 409
        assert studio.post('/api/file/rename', {'path': 'work/nothing', 'to': 'work/x'}).status == 404
        assert studio.post('/api/file/rename', {'path': 'work', 'to': 'work/empty/work'}).status == 400
        assert studio.ok('POST', '/api/file/delete', {'path': 'work/b.til'}) == {}
        assert studio.post('/api/file/delete', {'path': 'work/b.til'}).status == 404
        assert studio.ok('POST', '/api/file/delete', {'path': 'work'}) == {}
        assert not os.path.exists(os.path.join(project, 'work'))

    def test_special_files(self, studio, project):
        if not hasattr(os, 'mkfifo'):
            pytest.skip("no FIFOs here")
        fifo = os.path.join(project, 'pipe.fifo')
        os.mkfifo(fifo)
        try:
            t0 = time.monotonic()
            assert studio.get('/api/file?path=pipe.fifo').status == 415      # never blocks on the read
            assert time.monotonic() - t0 < 5
            assert 'pipe.fifo' not in [e['name'] for e in studio.ok('GET', '/api/tree')['entries']]
            assert studio.ok('POST', '/api/search', {'query': 'zzz'})['matches'] == []
            assert studio.post('/api/run', {'path': 'pipe.fifo'}).status == 404
        finally:
            os.unlink(fifo)

    def test_delete_and_rename_links_not_targets(self, studio, project):
        needs_symlinks(project)
        outside = os.path.join(os.path.dirname(project), 'outside')
        os.symlink(outside, os.path.join(project, 'tmp_link'))
        assert studio.ok('POST', '/api/file/rename', {'path': 'tmp_link', 'to': 'tmp_link2'}) == {'path': 'tmp_link2'}
        assert studio.ok('POST', '/api/file/delete', {'path': 'tmp_link2'}) == {}
        assert os.path.isfile(os.path.join(outside, 'secret.txt'))       # the target is untouched

    def test_search(self, studio):
        r = studio.ok('POST', '/api/search', {'query': 'alpha'})
        found = {(m['path'], m['line'], m['col']) for m in r['matches']}
        assert ('sub/data.txt', 1, 1) in found and ('sub/data.txt', 2, 7) in found
        assert ('sub/data.txt', 3, 4) in found                     # after an emoji: UTF-16 columns
        assert not any(m['path'].startswith('til_modules') for m in r['matches'])
        assert r['truncated'] is False
        m = next(m for m in r['matches'] if m['line'] == 3)
        assert m['text'] == '😀 alpha' and m['length'] == 5
        cs = studio.ok('POST', '/api/search', {'query': 'alpha', 'case': True})['matches']
        assert {(m['line'], m['col']) for m in cs if m['path'] == 'sub/data.txt'} == {(2, 7), (3, 4)}
        rx = studio.ok('POST', '/api/search', {'query': r'GAM+A\s', 'regex': True, 'case': True})['matches']
        assert [(m['path'], m['line']) for m in rx] == [('sub/data.txt', 2)]
        g = studio.ok('POST', '/api/search', {'query': 'print', 'glob': '*.til'})['matches']
        assert g and all(m['path'].endswith('.til') for m in g)
        g = studio.ok('POST', '/api/search', {'query': 'print', 'glob': 'sub/*.til'})['matches']
        assert {m['path'] for m in g} == {'sub/inner.til'}
        assert studio.post('/api/search', {'query': '(', 'regex': True}).status == 400
        assert studio.ok('POST', '/api/search', {'query': ''}) == {'matches': [], 'truncated': False}

    def test_search_is_capped(self, studio, project):
        with open(os.path.join(project, 'many.txt'), 'w') as f:
            f.write('needle ' * 2500 + '\n')
        try:
            r = studio.ok('POST', '/api/search', {'query': 'needle'})
            assert len(r['matches']) == 2000 and r['truncated'] is True
        finally:
            os.unlink(os.path.join(project, 'many.txt'))

    def test_fs_events(self, studio, project, events):
        path = os.path.join(project, 'external.til')
        try:
            mark = events.mark()
            with open(path, 'w') as f:
                f.write('main()\n    print(1)\n')
            data = events.wait(lambda n, d: n == 'fs' and 'external.til' in d['paths'], 5, mark)
            assert data['paths'] == sorted(data['paths'])
            # a file opened in the editor (GET) and changed on disk
            studio.ok('GET', '/api/file?path=external.til')
            mark = events.mark()
            time.sleep(0.05)
            with open(path, 'a') as f:
                f.write('# more\n')
            events.wait(lambda n, d: n == 'fs' and 'external.til' in d['paths'], 5, mark)
            # the studio's own save is not echoed back
            mark = events.mark()
            studio.ok('PUT', '/api/file', {'path': 'external.til', 'text': 'main()\n    print(2)\n'})
            time.sleep(2.5)
            with events.cond:
                echoed = [d for _t, n, d in events.items[mark:] if n == 'fs' and 'external.til' in d['paths']]
            assert echoed == []
            mark = events.mark()
            os.unlink(path)
            events.wait(lambda n, d: n == 'fs' and 'external.til' in d['paths'], 5, mark)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
# the LSP bridge
# ═══════════════════════════════════════════════════════════════════════════

def lsp(studio, method, params=None, mid=None):
    msg = {'jsonrpc': '2.0', 'method': method, 'params': params or {}}
    if mid is not None:
        msg['id'] = mid
    r = studio.post('/api/lsp', msg)
    assert r.status == 200, r
    return r.json()['messages']


class TestLsp:
    def test_diagnostics_arrive_as_events(self, studio, standins, project, events):
        out = lsp(studio, 'initialize', {'processId': None, 'rootUri': None, 'locale': 'en'}, 1)
        assert out[0]['id'] == 1 and out[0]['result']['serverInfo']['name'] == 'til-lsp'
        assert lsp(studio, 'initialized') == []
        uri = 'file://' + os.path.join(os.path.realpath(project), 'broken.til')
        mark = events.mark()
        out = lsp(studio, 'textDocument/didOpen', {'textDocument': {
            'uri': uri, 'languageId': 'til', 'version': 1, 'text': 'main()\n    print(oops)\n'}})
        # notifications of handle(): in the reply and on the stream
        assert out and out[0]['method'] == 'window/logMessage'
        events.wait(lambda n, d: n == 'lsp' and d.get('method') == 'window/logMessage', 5, mark)
        # the debounced diagnostics: the ticker calls poll()
        diag = events.wait(lambda n, d: n == 'lsp' and d.get('method') == 'textDocument/publishDiagnostics'
                           and d['params']['uri'] == uri, 5, mark)
        d = diag['params']['diagnostics']
        assert len(d) == 1 and d[0]['code'] == 'E-VM' and d[0]['range']['start'] == {'line': 1, 'character': 10}
        hover = lsp(studio, 'textDocument/hover', {'textDocument': {'uri': uri},
                                                   'position': {'line': 1, 'character': 11}}, 2)
        assert hover == [{'jsonrpc': '2.0', 'id': 2, 'result': {'contents': {'kind': 'markdown',
                                                                            'value': '`x: int`'}}}]
        # responses are not events
        with events.cond:
            assert not [x for _t, n, x in events.items if n == 'lsp' and x.get('id') == 2]
        lsp(studio, 'textDocument/didClose', {'textDocument': {'uri': uri}})

    def test_one_shared_server(self, studio, standins):
        out = lsp(studio, 'initialize', {'processId': None, 'locale': 'en'}, 7)
        assert out[0]['id'] == 7 and 'result' in out[0]
        out = lsp(studio, 'initialize', {'processId': None, 'locale': 'ru'}, 8)
        assert out[0]['id'] == 8 and 'result' in out[0]          # refused by the server: the first answer
        server = standins['tillsp'].LspServer.instances[-1]
        assert len(standins['tillsp'].LspServer.instances) == 1
        assert server.root == os.path.realpath(studio.server.root)
        # a tab's shutdown / exit does not end the shared server
        assert lsp(studio, 'shutdown', None, 9) == [{'jsonrpc': '2.0', 'id': 9, 'result': None}]
        assert lsp(studio, 'exit') == []
        assert 'shutdown' not in server.methods and 'exit' not in server.methods

    def test_errors(self, studio):
        out = lsp(studio, 'boom', {}, 11)
        assert out[0]['id'] == 11 and out[0]['error']['code'] == -32603 and 'boom' in out[0]['error']['message']
        out = lsp(studio, 'boom')
        assert out[0]['method'] == 'window/logMessage'
        out = lsp(studio, 'textDocument/unknown', {}, 12)
        assert out[0]['error']['code'] == -32601

    def test_without_tillsp(self, project):
        server = tilstudio.StudioServer(project, frontend=False, modules={'tillsp': None, 'tildap': None},
                                        log=open(os.devnull, 'w')).start()
        try:
            api = Api(server)
            r = api.post('/api/lsp', {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}})
            assert r.status == 503 and 'tillsp.py' in r.json()['error']
            assert api.post('/api/dap/start').status == 503
            assert api.ok('GET', '/api/info')['capabilities']['debugger'] is False
        finally:
            server.stop()


# ═══════════════════════════════════════════════════════════════════════════
# the DAP bridge
# ═══════════════════════════════════════════════════════════════════════════

class TestDap:
    def test_round_trip(self, studio, standins, events):
        sid = studio.ok('POST', '/api/dap/start')['sessionId']
        assert isinstance(sid, str) and sid
        session = standins['tildap'].DapSession.instances[-1]
        assert session.root == os.path.realpath(studio.server.root)
        mark = events.mark()
        out = studio.ok('POST', f'/api/dap/{sid}', {'seq': 1, 'type': 'request', 'command': 'initialize',
                                                    'arguments': {'adapterID': 'til'}})['messages']
        assert out[0]['command'] == 'initialize' and out[0]['body']['supportsStepBack'] is True
        ev = events.wait(lambda n, d: n == 'dap' and d['sessionId'] == sid, 5, mark)
        assert ev['message']['event'] == 'initialized'
        studio.ok('POST', f'/api/dap/{sid}', {'seq': 2, 'type': 'request', 'command': 'launch',
                                              'arguments': {'program': 'hello.til'}})
        ev = events.wait(lambda n, d: n == 'dap' and d['sessionId'] == sid and d['message']['event'] == 'output',
                         5, mark)
        assert ev['message']['body']['output'] == 'hello from hello.til\n'
        events.wait(lambda n, d: n == 'dap' and d['message'].get('event') == 'terminated', 5, mark)
        # an exception in the adapter: an error response, not a 500
        out = studio.ok('POST', f'/api/dap/{sid}', {'seq': 3, 'type': 'request', 'command': 'boom'})['messages']
        assert out[0]['success'] is False and 'dap boom' in out[0]['message']
        out = studio.ok('POST', f'/api/dap/{sid}', {'seq': 4, 'type': 'request', 'command': 'disconnect'})
        assert out['messages'][0]['command'] == 'disconnect'
        assert session.closed is True
        assert studio.post(f'/api/dap/{sid}', {'seq': 5, 'type': 'request', 'command': 'threads'}).status == 404

    def test_sessions_are_separate(self, studio, standins):
        a = studio.ok('POST', '/api/dap/start')['sessionId']
        b = studio.ok('POST', '/api/dap/start')['sessionId']
        assert a != b
        for sid in (a, b):
            studio.ok('POST', f'/api/dap/{sid}', {'seq': 1, 'type': 'request', 'command': 'disconnect'})
        assert studio.post('/api/dap/nope', {'seq': 1, 'type': 'request', 'command': 'threads'}).status == 404


# ═══════════════════════════════════════════════════════════════════════════
# runs
# ═══════════════════════════════════════════════════════════════════════════

BACKENDS = ['vm', pytest.param('cvm', marks=needs_cc), pytest.param('native', marks=needs_cc)]


class TestRuns:
    @pytest.mark.parametrize('backend', BACKENDS)
    def test_output_in_order_and_exit(self, studio, events, backend):
        rid, evs = studio.run(events, path='count.til', backend=backend)
        assert text_of(evs) == "".join(f"{i}\n" for i in range(3000))
        ex = exit_of(evs)
        assert ex['code'] == 0 and ex['reason'] == 'exited' and 'text' not in ex
        assert any(e['kind'] == 'status' and e['state'] == 'running' and e['backend'] == backend for e in evs)
        assert evs[-1]['kind'] == 'exit'
        # `text` is the program's output only (the UI prints it); states are machine words
        assert all('text' not in e for e in evs if e['kind'] in ('status', 'exit', 'input', 'screen'))
        rid, evs = studio.run(events, path='exit3.til', backend=backend)
        assert text_of(evs) == "bye\n" and exit_of(evs)['code'] == 3

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_streaming_and_stdin(self, studio, events, backend):
        r = studio.post('/api/run', {'path': 'slow.til', 'backend': backend})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and 'Name? ' in d.get('text', ''), 60)
        timed = [(t, d) for t, d in events.timed(rid) if d['kind'] == 'stdout']
        # four lines 150 ms apart arrived as they were printed (not all at the end)
        firsts = [t for t, d in timed if d['text'].strip() in ('0', '3')]
        assert len(firsts) == 2 and firsts[1] - firsts[0] > 0.3
        assert studio.ok('POST', f'/api/run/{rid}/stdin', {'text': 'Ann'}) == {}
        evs = events.until_exit(rid, 30)
        assert text_of(evs) == "0\n1\n2\n3\nName? Hello Ann\n"
        assert exit_of(evs)['code'] == 3
        assert studio.post(f'/api/run/{rid}/stdin', {'text': 'late'}).status == 409

    def test_vm_input_event(self, studio, events):
        r = studio.post('/api/run', {'path': 'greet.til', 'backend': 'vm'})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d['kind'] == 'input', 30)
        before = events.run_events(rid)
        assert text_of(before) == "start\nName? "
        kinds = [e['kind'] for e in before]
        assert kinds.index('input') > max(i for i, k in enumerate(kinds) if k == 'stdout')   # after the prompt
        mark = events.mark()
        studio.ok('POST', f'/api/run/{rid}/stdin', {'text': 'Bob'})
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d['kind'] == 'input', 30, mark)
        studio.ok('POST', f'/api/run/{rid}/stdin', {'text': 'more\n'})
        evs = events.until_exit(rid, 30)
        assert text_of(evs) == "start\nName? Hello Bob\nthen more\n" and exit_of(evs)['code'] == 0
        # end of input: input() returns ""
        r = studio.post('/api/run', {'path': 'greet.til', 'backend': 'vm'})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d['kind'] == 'input', 30)
        studio.ok('POST', f'/api/run/{rid}/stdin', {'text': '', 'eof': True})
        evs = events.until_exit(rid, 30)
        assert text_of(evs) == "start\nName? Hello \nthen \n"

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_given_stdin_is_the_whole_input(self, studio, events, backend):
        """As `til run < FILE` and the debugger's launch `stdin`: then the end of input."""
        rid, evs = studio.run(events, path='greet.til', backend=backend, stdin='Bob\n')
        assert text_of(evs) == "start\nName? Hello Bob\nthen \n" and exit_of(evs)['code'] == 0
        assert not [e for e in evs if e['kind'] == 'input']
        src = 'main()\n    print(input())\n    sleep_ms(100000)\n'
        r = studio.post('/api/run', {'source': src, 'backend': backend, 'stdin': 'x'})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d.get('text') == 'x\n', 60)
        r = studio.post(f'/api/run/{rid}/stdin', {'text': 'more'})
        assert r.status == 409 and 'whole input' in r.json()['error']
        studio.ok('POST', f'/api/run/{rid}/stop')
        events.until_exit(rid, 10)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_stop_infinite_loop(self, studio, events, backend):
        r = studio.post('/api/run', {'path': 'loop.til', 'backend': backend})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d.get('state') == 'running', 60)
        time.sleep(0.3)
        t0 = time.monotonic()
        assert studio.ok('POST', f'/api/run/{rid}/stop') == {}
        evs = events.until_exit(rid, 5)
        assert time.monotonic() - t0 < 1.0
        ex = exit_of(evs)
        assert ex['code'] == 130 and ex['reason'] == 'stopped'
        assert text_of(evs, 'stderr') == ''            # no "step limit" report
        assert studio.ok('POST', f'/api/run/{rid}/stop') == {}       # stopping again is harmless

    @pytest.mark.parametrize('program, stdin', [('sleepy.til', None), ('greet.til', None), ('gfx.til', None)])
    def test_stop_while_waiting(self, studio, events, program, stdin):
        r = studio.post('/api/run', {'path': program, 'backend': 'vm'})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d['kind'] in ('stdout', 'screen'), 30)
        time.sleep(0.2)
        t0 = time.monotonic()
        studio.ok('POST', f'/api/run/{rid}/stop')
        evs = events.until_exit(rid, 5)
        assert time.monotonic() - t0 < 1.0 and exit_of(evs)['reason'] == 'stopped'

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_errors(self, studio, events, backend):
        rid, evs = studio.run(events, path='broken.til', backend=backend)
        assert "unknown name 'oops'" in text_of(evs, 'stderr') and '\x1b[' not in text_of(evs, 'stderr')
        assert exit_of(evs) ['code'] == 1 and exit_of(evs)['reason'] == 'failed'
        rid, evs = studio.run(events, path='crash.til', backend=backend)
        assert text_of(evs) == "before\n" and exit_of(evs)['code'] == 1
        if backend != 'native':
            assert 'runtime error: index out of bounds' in text_of(evs, 'stderr')

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_working_directory_is_the_project(self, studio, events, backend):
        src = 'main()\n    print(read_file("sub/data.txt").split("\\n")[0])\n'
        rid, evs = studio.run(events, source=src, backend=backend)
        assert text_of(evs) == "Alpha beta\n", evs
        rid, evs = studio.run(events, source=src, backend=backend, cwd='sub')
        assert 'cannot read file' in text_of(evs, 'stderr') or exit_of(evs)['code'] != 0

    def test_internal_error_ends_the_run_cleanly(self, studio, events, monkeypatch):
        def boom(module):
            raise RuntimeError("studio bug")
        monkeypatch.setattr(tilstudio, '_uses_screen', boom)
        rid, evs = studio.run(events, path='hello.til', backend='native')
        assert '[TIL] internal error: RuntimeError: studio bug' in text_of(evs, 'stderr')
        assert 'Traceback' not in text_of(evs, 'stderr')
        assert exit_of(evs)['code'] == 1 and exit_of(evs)['reason'] == 'failed'
        monkeypatch.undo()
        rid, evs = studio.run(events, path='hello.til', backend='vm')     # the server lives on
        assert text_of(evs) == "Hello, Studio!\n"

    def test_messages_in_the_run_language(self, studio, events):
        rid, evs = studio.run(events, path='crash.til', backend='vm', lang='ru')
        assert 'ошибка выполнения' in text_of(evs, 'stderr')

    def test_source_args_and_steps(self, studio, events):
        rid, evs = studio.run(events, source='main()\n    print("unsaved")\n', backend='vm')
        assert text_of(evs) == "unsaved\n"
        rid, evs = studio.run(events, path='args.til', backend='vm', args=['a', 'b c'])
        assert text_of(evs) == '["a", "b c"]\n'
        rid, evs = studio.run(events, path='loop.til', backend='vm', maxSteps=1000)
        assert 'step limit exceeded (1000)' in text_of(evs, 'stderr') and exit_of(evs)['code'] == 1
        assert studio.post('/api/run', {'path': 'hello.til', 'backend': 'gpu'}).status == 400
        assert studio.post('/api/run', {'path': 'nothing.til'}).status == 404
        assert studio.post('/api/run', {}).status == 400
        assert studio.post('/api/run', {'path': 'hello.til', 'args': 'x'}).status == 400
        assert studio.post('/api/run/nope/stop').status == 404

    @needs_cc
    def test_cvm_args(self, studio, events):
        rid, evs = studio.run(events, path='args.til', backend='cvm', args=['x y'])
        assert text_of(evs) == '["x y"]\n'
        rid, evs = studio.run(events, path='loop.til', backend='cvm', maxSteps=1000)
        assert 'step limit exceeded' in text_of(evs, 'stderr') and exit_of(evs)['code'] == 1

    @needs_cc
    def test_native_falls_back_to_the_vm(self, studio, events):
        rid, evs = studio.run(events, path='mixed.til', backend='native')
        assert text_of(evs) == '[1, "a", 2.5]\n'
        assert 'running it on the TIL VM' in text_of(evs, 'stderr')
        assert any(e['kind'] == 'status' and e.get('backend') == 'vm' and e['state'] == 'running' for e in evs)

    @pytest.mark.parametrize('backend', ['vm', 'native', 'cvm'])
    def test_screen_frames_and_keys(self, studio, events, backend):
        r = studio.post('/api/run', {'path': 'gfx.til', 'backend': backend})
        rid = r.json()['runId']
        first = events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d['kind'] == 'screen', 30)
        assert (first['w'], first['h']) == (4, 2)
        rgb = base64.b64decode(first['rgb'])
        assert len(rgb) == 4 * 2 * 3 and rgb[:3] == b'\xff\x00\x00' and rgb[3:6] == b'\x00\x00\x00'
        studio.ok('POST', f'/api/run/{rid}/key', {'code': 'ENTER'})
        evs = events.until_exit(rid, 30)
        assert text_of(evs) == "key 13\n" and exit_of(evs)['code'] == 0
        frames = [e for e in evs if e['kind'] == 'screen']
        assert base64.b64decode(frames[-1]['rgb']) == b'\x00\xff\x00' * 8
        if backend != 'vm':                    # programs that use the screen run on the VM
            assert 'uses the screen' in text_of(evs, 'stderr')
            assert any(e['kind'] == 'status' and e.get('backend') == 'vm' and e['state'] == 'running' for e in evs)

    def test_key_pressed_and_key_scripts(self, studio, events):
        r = studio.post('/api/run', {'path': 'poll.til', 'backend': 'vm'})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d.get('state') == 'running', 30)
        for code in ('a', 'UP', 32):
            studio.ok('POST', f'/api/run/{rid}/key', {'code': code})
        evs = events.until_exit(rid, 30)
        assert text_of(evs) == "97\n1\n32\n"
        assert studio.post(f'/api/run/{rid}/key', {'code': 'a'}).status == 409
        rid, evs = studio.run(events, path='poll.til', backend='vm', keys='NONE,x,NONE,y,DOWN')
        assert text_of(evs) == "120\n121\n2\n"
        assert studio.post('/api/run', {'path': 'poll.til', 'keys': 'NOPE,LEFTISH'}).status == 400
        # a key script is the run's keys (as `til run --keys`): wait_key() gets ESC when it ends
        rid, evs = studio.run(events, path='gfx.til', backend='vm', keys='NONE')
        assert text_of(evs) == "key 27\n"
        rid, evs = studio.run(events, path='gfx.til', backend='vm', keys=[ord('q')])
        assert text_of(evs) == "key 113\n"
        r = studio.post('/api/run', {'path': 'poll.til', 'backend': 'vm', 'keys': 'NONE'})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d.get('state') == 'running', 30)
        r = studio.post(f'/api/run/{rid}/key', {'code': 'a'})
        assert r.status == 409 and 'key script' in r.json()['error']
        studio.ok('POST', f'/api/run/{rid}/stop')
        events.until_exit(rid, 10)

    def test_frames_are_coalesced(self, studio, events, monkeypatch):
        sent = []                         # when the server sends frames (receiving can bunch up)
        publish = studio.server.hub.publish

        def recording(event, data, key=None):
            if event == 'run' and data.get('kind') == 'screen':
                sent.append(time.monotonic())
            publish(event, data, key)
        monkeypatch.setattr(studio.server.hub, 'publish', recording)
        rid, evs = studio.run(events, path='frames.til', backend='vm')
        frames = [e for e in evs if e['kind'] == 'screen']
        assert 1 <= len(frames) <= len(sent) <= (sent[-1] - sent[0]) * 30 + 1
        assert base64.b64decode(frames[-1]['rgb'])[:3] == bytes((0, 1, 43))      # the last one: 299 = 0x00012B
        assert all(b - a >= 1 / 30 - 0.002 for a, b in zip(sent, sent[1:]))       # at most 30 per second

    def test_output_limit(self, studio, events):
        rid, evs = studio.run(events, path='big.til', backend='vm')
        out = text_of(evs)
        assert len(out.encode('utf-8')) == tilstudio.OUTPUT_LIMIT
        assert 'done' not in out and 'output limit reached (4 MB)' in text_of(evs, 'stderr')
        assert exit_of(evs)['code'] == 0
        chunks = [e for e in evs if e['kind'] == 'stdout']
        assert 1 < len(chunks) < 5000                                        # coalesced, not one per print

    def test_concurrent_runs_limit(self, studio, events):
        ids = []
        try:
            for _ in range(tilstudio.MAX_RUNS):
                r = studio.post('/api/run', {'path': 'loop.til', 'backend': 'vm'})
                assert r.status == 200, r
                ids.append(r.json()['runId'])
            r = studio.post('/api/run', {'path': 'hello.til', 'backend': 'vm'})
            assert r.status == 429 and 'at most 4' in r.json()['error']
        finally:
            for rid in ids:
                studio.post(f'/api/run/{rid}/stop')
        for rid in ids:
            assert exit_of(events.until_exit(rid, 10))['reason'] == 'stopped'
        rid, evs = studio.run(events, path='hello.til', backend='vm')
        assert text_of(evs) == "Hello, Studio!\n"

    def test_snapshot(self, studio, events, project):
        r = studio.post('/api/run', {'path': 'long.til', 'backend': 'vm'})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d['kind'] == 'stdout', 60)
        snap = studio.ok('POST', f'/api/run/{rid}/snapshot')
        assert snap['path'] == 'long.tilp' and snap['size'] > 0 and snap['steps'] > 0
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d.get('state') == 'snapshot', 10)
        snap2 = studio.ok('POST', f'/api/run/{rid}/snapshot', {'stop': True})
        assert snap2['path'] == 'long-2.tilp' and snap2['steps'] >= snap['steps']
        evs = events.until_exit(rid, 10)
        assert exit_of(evs)['code'] == 3 and exit_of(evs)['reason'] == 'paused'
        assert f"paused after {snap2['steps']} steps; saved long-2.tilp" in text_of(evs, 'stderr')
        # the snapshot continues on a VM that loads the same program
        with open(os.path.join(project, 'long.tilp'), 'rb') as f:
            data = f.read()
        assert data[:4] == b'TILP'
        module = til.compile_to_bytecode(PROGRAMS['long.til'], 'long.til')
        out = []
        vm = til.TILVM(module, write=out.append)
        vm.load_portable(data)
        assert vm.steps == snap['steps']
        assert vm.run_steps(50000) == vm.YIELD and out
        # not while it waits for input; not on other backends; not after the end
        r = studio.post('/api/run', {'path': 'greet.til', 'backend': 'vm'})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d['kind'] == 'input', 30)
        t0 = time.monotonic()
        r = studio.post(f'/api/run/{rid}/snapshot')
        assert r.status == 409 and time.monotonic() - t0 < tilstudio.SNAPSHOT_WAIT + 2
        studio.ok('POST', f'/api/run/{rid}/stop')
        events.until_exit(rid, 10)
        assert studio.post(f'/api/run/{rid}/snapshot').status == 409
        for name in ('long.tilp', 'long-2.tilp'):
            os.unlink(os.path.join(project, name))

    @needs_cc
    def test_snapshot_needs_the_vm(self, studio, events):
        r = studio.post('/api/run', {'path': 'loop.til', 'backend': 'cvm'})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d.get('state') == 'running', 60)
        r = studio.post(f'/api/run/{rid}/snapshot')
        assert r.status == 409 and 'vm' in r.json()['error']
        assert studio.post(f'/api/run/{rid}/key', {'code': 13}).status == 409
        studio.ok('POST', f'/api/run/{rid}/stop')
        events.until_exit(rid, 10)

    def test_snapshot_of_an_unsaved_buffer(self, studio, events, project):
        r = studio.post('/api/run', {'source': PROGRAMS['long.til'], 'backend': 'vm'})
        rid = r.json()['runId']
        events.wait(lambda n, d: n == 'run' and d.get('runId') == rid and d['kind'] == 'stdout', 60)
        snap = studio.ok('POST', f'/api/run/{rid}/snapshot', {'stop': True})
        assert snap['path'] == 'untitled.tilp' and os.path.isfile(os.path.join(project, 'untitled.tilp'))
        events.until_exit(rid, 10)
        os.unlink(os.path.join(project, 'untitled.tilp'))


# ═══════════════════════════════════════════════════════════════════════════
# tools
# ═══════════════════════════════════════════════════════════════════════════

class TestTools:
    def test_dis(self, studio):
        r = studio.ok('POST', '/api/tools/dis', {'path': 'count.til'})
        module = til.compile_to_bytecode(PROGRAMS['count.til'], 'count.til')
        assert r['text'] == til.disassemble(module)
        assert r['lines'] == [list(x) for x in module.lines]
        names = [f['name'] for f in r['functions']]
        assert names == [f.name for f in module.funcs]
        main = r['functions'][names.index('main')]
        ins = r['instructions']
        assert ins[0]['pc'] == main['start'] and all(main['start'] <= i['pc'] < main['end'] for i in ins
                                                     if i['func'] == 'main')
        printing = [i for i in ins if i['op'] in ('NATIVE', 'NATIVE_POP')]
        assert printing and printing[0]['note'] == 'print()' and printing[0]['line'] == 3
        assert any(i['fused'] for i in ins) and all(i['fused'] == (i['op'] not in til.Op.__dict__ or
                                                                   getattr(til.Op, i['op']) >= 0x80) for i in ins)
        jumps = [i for i in ins if i['op'] == 'JUMP']
        assert jumps and jumps[0]['note'].startswith('-> ') and jumps[0]['args'][0] < 0
        plain = studio.ok('POST', '/api/tools/dis', {'path': 'count.til', 'fuse': False})
        assert not any(i['fused'] for i in plain['instructions'])
        r = studio.post('/api/tools/dis', {'path': 'broken.til'})
        assert r.status == 422 and "unknown name 'oops'" in r.json()['error']

    def test_translate(self, studio):
        r = studio.ok('POST', '/api/tools/translate', {'path': 'hello.til', 'to': 'ru'})
        assert r['text'].startswith('#[lang: ru]')
        back = studio.ok('POST', '/api/tools/translate', {'source': r['text'], 'to': 'en'})
        assert back['text'] == PROGRAMS['hello.til']
        r = studio.ok('POST', '/api/tools/translate', {'source': 'main()\n    if true\n        print(1)\n', 'to': 'kk'})
        assert 'егер' in r['text'] and 'ақиқат' in r['text']
        assert studio.post('/api/tools/translate', {'path': 'hello.til', 'to': 'fr'}).status == 400

    def test_fit(self, studio):
        r = studio.ok('POST', '/api/tools/fit', {'path': 'hello.til', 'device': 'desktop'})
        assert r['fits'] is True and r['verdict'] == 'fits' and r['device']['name'] == 'desktop'
        assert r['program'] == 'hello.til' and r['resources']
        r = studio.ok('POST', '/api/tools/fit', {'path': 'gfx.til', 'device': 'z80-16'})
        assert set(r) >= {'fits', 'verdict', 'errors', 'warnings', 'static', 'vm'}
        assert studio.post('/api/tools/fit', {'path': 'hello.til', 'device': 'toaster'}).status == 400
        assert studio.post('/api/tools/fit', {'path': 'hello.til', 'device': '/etc/passwd'}).status == 400
        assert studio.post('/api/tools/fit', {'path': 'broken.til', 'device': 'desktop'}).status == 422

    @needs_cc
    def test_fit_run(self, studio):
        r = studio.ok('POST', '/api/tools/fit', {'path': 'count.til', 'device': 'desktop', 'run': True})
        assert r['run'] is not None and r['run']['exit_code'] == 0

    def test_build(self, studio, project):
        r = studio.ok('POST', '/api/tools/build', {'path': 'hello.til', 'target': 'vm'})
        assert r['ok'] is True and r['artifact'] == 'hello.tbc' and 'Built' in r['output']
        with open(os.path.join(project, 'hello.tbc'), 'rb') as f:
            assert f.read()[:4] == b'TILB'
        r = studio.ok('POST', '/api/tools/build', {'path': 'sub/inner.til', 'target': 'c-vm'})
        assert r['ok'] is True and r['artifact'] == 'sub/inner_vm.c'
        r = studio.ok('POST', '/api/tools/build', {'path': 'broken.til', 'target': 'vm'})
        assert r['ok'] is False and r['artifact'] is None and "oops" in r['output']
        assert studio.post('/api/tools/build', {'path': 'hello.til', 'target': 'gpu'}).status == 400
        for name in ('hello.tbc', 'sub/inner_vm.c'):
            os.unlink(os.path.join(project, name))

    @needs_cc
    def test_build_native(self, studio, project):
        r = studio.ok('POST', '/api/tools/build', {'path': 'hello.til', 'target': 'native'})
        exe = r['artifact']
        assert r['ok'] is True and exe in ('hello', 'hello.exe')
        out = subprocess.run([os.path.join(project, exe)], capture_output=True, text=True)
        assert out.stdout == "Hello, Studio!\n"
        os.unlink(os.path.join(project, exe))

    def test_pkg(self, studio, project):
        r = studio.ok('POST', '/api/pkg', {'argv': ['list']})
        assert r['rc'] == 1 and 'til.json' in r['stderr'] and r['manifest'] is None
        r = studio.ok('POST', '/api/pkg', {'argv': ['init', 'demo']})
        assert r['rc'] == 0 and 'created' in r['stdout']
        assert r['manifest'] == {'name': 'demo', 'version': '0.1.0', 'dependencies': {}}
        lib = os.path.join(os.path.dirname(project), 'mylib')
        os.makedirs(lib, exist_ok=True)
        with open(os.path.join(lib, 'mylib.til'), 'w') as f:
            f.write('greet() -> str\n    return "hi from mylib"\n')
        r = studio.ok('POST', '/api/pkg', {'argv': ['add', 'mylib', '../mylib']})
        assert r['rc'] == 0, r
        assert r['manifest']['dependencies'] == {'mylib': '../mylib'}
        r = studio.ok('POST', '/api/pkg', {'argv': ['list']})
        assert r['rc'] == 0 and 'mylib' in r['stdout']
        assert studio.ok('POST', '/api/pkg', {'argv': ['verify']})['rc'] == 0
        assert studio.ok('POST', '/api/pkg', {'argv': ['remove', 'mylib']})['manifest']['dependencies'] == {}
        assert studio.post('/api/pkg', {'argv': ['publish']}).status == 400
        assert studio.post('/api/pkg', {'argv': ['list', '-C', '/']}).status == 400
        assert studio.post('/api/pkg', {'argv': []}).status == 400
        for name in ('til.json', 'til.lock'):
            if os.path.exists(os.path.join(project, name)):
                os.unlink(os.path.join(project, name))

    def test_prove(self, studio):
        r = studio.ok('POST', '/api/tools/prove', {'path': 'proofs.til'})
        if not tilstudio._find_spec('z3'):
            assert r == {'available': False, 'results': [], 'hint': 'pip install z3-solver'}
            pytest.skip("z3 is not installed")
        assert r['available'] is True
        by = {x['function']: x for x in r['results']}
        assert by['double']['status'] == 'proved' and by['double']['line'] == 3
        bad = by['bad']
        # x <= 0; INT_MIN does not fit a JavaScript number: exact decimal text then
        assert bad['status'] == 'counterexample' and int(bad['counterexample']['x']) <= 0
        assert bad['errors'] and 'E-PROVE' in bad['errors'][0] and bad['failures'][0]['args']
        only = studio.ok('POST', '/api/tools/prove', {'path': 'proofs.til', 'function': 'double'})
        assert [x['function'] for x in only['results']] == ['double']
        r = studio.ok('POST', '/api/tools/prove', {'path': 'proofs.til', 'function': 'nothing'})
        assert r['results'] == [] and 'nothing' in r['error']
        r = studio.ok('POST', '/api/tools/prove', {'path': 'broken.til'})
        assert r['results'] == [] and 'oops' in r['error']


# ═══════════════════════════════════════════════════════════════════════════
# the real language server / debug adapter (when installed)
# ═══════════════════════════════════════════════════════════════════════════

HAS_TILLSP = os.path.isfile(os.path.join(SRC, 'tillsp.py'))
HAS_TILDAP = os.path.isfile(os.path.join(SRC, 'tildap.py'))


@pytest.fixture(scope='module')
def real_studio(project):
    server = tilstudio.StudioServer(project, frontend=False, log=open(os.devnull, 'w')).start()
    yield Api(server)
    server.stop()


@pytest.mark.skipif(not HAS_TILLSP, reason="src/tillsp.py is not installed")
def test_real_language_server(real_studio, project):
    events = real_studio.events()
    try:
        out = lsp(real_studio, 'initialize', {'processId': None, 'rootUri': tilstudio._path_uri(project),
                                              'capabilities': {}, 'locale': 'en'}, 1)
        assert out[0]['id'] == 1 and 'capabilities' in out[0]['result']
        lsp(real_studio, 'initialized')
        uri = tilstudio._path_uri(os.path.join(os.path.realpath(project), 'broken.til'))
        lsp(real_studio, 'textDocument/didOpen', {'textDocument': {
            'uri': uri, 'languageId': 'til', 'version': 1, 'text': PROGRAMS['broken.til']}})
        diag = events.wait(lambda n, d: n == 'lsp' and d.get('method') == 'textDocument/publishDiagnostics'
                           and d['params']['uri'] == uri and d['params']['diagnostics'], 15)
        assert any('oops' in d['message'] for d in diag['params']['diagnostics'])
        # a change: its diagnostics are debounced, poll() (the ticker) sends them on the stream
        mark = events.mark()
        out = lsp(real_studio, 'textDocument/didChange', {
            'textDocument': {'uri': uri, 'version': 2},
            'contentChanges': [{'text': 'main()\n    print("fixed")\n    print(nope)\n'}]})
        assert not [m for m in out if m.get('method') == 'textDocument/publishDiagnostics']
        diag = events.wait(lambda n, d: n == 'lsp' and d.get('method') == 'textDocument/publishDiagnostics'
                           and d['params']['uri'] == uri, 10, mark)
        assert [d['range']['start']['line'] for d in diag['params']['diagnostics']] == [2]
        assert 'nope' in diag['params']['diagnostics'][0]['message']
        out = lsp(real_studio, 'textDocument/hover', {'textDocument': {'uri': uri},
                                                      'position': {'line': 1, 'character': 5}}, 2)
        assert out[0]['id'] == 2 and 'result' in out[0]
        lsp(real_studio, 'textDocument/didClose', {'textDocument': {'uri': uri}})
    finally:
        events.close()


@pytest.mark.skipif(not HAS_TILDAP, reason="src/tildap.py is not installed")
def test_real_debug_adapter(real_studio, project, monkeypatch):
    events = real_studio.events()
    api = real_studio
    try:
        sid = api.ok('POST', '/api/dap/start')['sessionId']

        def dap(seq, command, arguments=None):
            msg = {'seq': seq, 'type': 'request', 'command': command, 'arguments': arguments or {}}
            out = api.ok('POST', f'/api/dap/{sid}', msg)['messages']
            assert out and out[0]['type'] == 'response' and out[0]['command'] == command, out
            return out[0]

        def event(name, mark, timeout=20):
            return events.wait(lambda n, d: n == 'dap' and d['sessionId'] == sid
                               and d['message'].get('event') == name, timeout, mark)['message']
        r = dap(1, 'initialize', {'adapterID': 'til', 'linesStartAt1': True, 'columnsStartAt1': True})
        assert r['success'] and r['body']['supportsStepBack'] is True
        # the events a request causes go out after its reply (hold_events)
        published = []
        publish = api.server.hub.publish

        def recording(name, data, key=None):
            published.append((name, data))
            publish(name, data, key)
        monkeypatch.setattr(api.server.hub, 'publish', recording)
        seen = []
        mark = events.mark()
        launch = {'seq': 2, 'type': 'request', 'command': 'launch',
                  'arguments': {'program': 'greet.til', 'stopOnEntry': True}}
        api.server.dap.handle(sid, launch, 'en', lambda out: seen.append(
            (out[0]['success'], [d['message'].get('event') for n, d in published if n == 'dap'])))
        assert seen == [(True, [])]
        assert 'initialized' in [d['message'].get('event') for n, d in published if n == 'dap']
        monkeypatch.undo()
        event('initialized', mark)
        dap(3, 'configurationDone')
        stopped = event('stopped', mark)
        assert stopped['body']['reason'] == 'entry' and stopped['body']['threadId'] == 1
        assert dap(4, 'threads')['body']['threads'][0]['id'] == 1
        frames = dap(5, 'stackTrace', {'threadId': 1})['body']['stackFrames']
        assert frames[0]['source']['name'] == 'greet.til' and frames[0]['line'] >= 1
        mark = events.mark()
        dap(6, 'continue', {'threadId': 1})
        event('til/inputRequested', mark)                 # without launch `stdin`: interactive input
        mark = events.mark()
        dap(7, 'til/stdin', {'text': 'Zoe\n'})
        event('til/inputRequested', mark)
        dap(8, 'til/stdin', {'text': 'more\n'})
        event('terminated', mark)
        with events.cond:
            out = "".join(d['message']['body'].get('output', '') for _t, n, d in events.items
                          if n == 'dap' and d['sessionId'] == sid and d['message'].get('event') == 'output')
        assert 'Hello Zoe' in out and 'then more' in out
        dap(9, 'disconnect')
        assert api.post(f'/api/dap/{sid}', {'seq': 10, 'type': 'request', 'command': 'threads'}).status == 404
    finally:
        events.close()


# ═══════════════════════════════════════════════════════════════════════════
# shutdown and the command line
# ═══════════════════════════════════════════════════════════════════════════

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                    # a zombie (not reaped yet) is not running
        with open(f'/proc/{pid}/stat') as f:
            return f.read().split(')')[-1].split()[0] != 'Z'
    except OSError:
        return True


def test_shutdown_cleans_up(project, standins):
    before = set(threading.enumerate())
    server = tilstudio.StudioServer(project, frontend=False, modules=standins, log=open(os.devnull, 'w')).start()
    api = Api(server)
    ev = api.events()
    second = api.events()
    rid_vm = api.post('/api/run', {'path': 'loop.til', 'backend': 'vm'}).json()['runId']
    rid_in = api.post('/api/run', {'path': 'greet.til', 'backend': 'vm'}).json()['runId']
    pids = []
    if CC:
        rid_p = api.post('/api/run', {'path': 'loop.til', 'backend': 'cvm'}).json()['runId']
        st = ev.wait(lambda n, d: n == 'run' and d.get('runId') == rid_p and d.get('state') == 'running', 60)
        pids.append(st['pid'])
    ev.wait(lambda n, d: n == 'run' and d.get('runId') == rid_in and d['kind'] == 'input', 30)
    api.ok('POST', '/api/dap/start')
    t0 = time.monotonic()
    server.stop()
    assert time.monotonic() - t0 < 10
    ev.thread.join(5)
    second.thread.join(5)
    assert ev.ended and second.ended                     # the event streams were closed
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        left = [t for t in threading.enumerate() if t not in before and t.is_alive()
                and t not in (ev.thread, second.thread)]
        if not left:
            break
        time.sleep(0.05)
    assert left == [], [t.name for t in left]
    for pid in pids:
        assert not _alive(pid)
    assert all(s.closed for s in standins['tildap'].DapSession.instances)
    with pytest.raises(OSError):
        socket.create_connection(('127.0.0.1', api.port), timeout=2).close()


def test_command_line(project, tmp_path):
    env = dict(os.environ)
    env.pop('TIL_LANG', None)
    proc = subprocess.Popen([sys.executable, TIL, 'ide', project, '--port', '0', '--no-browser'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        url = None
        lines = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and url is None:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            m = re.search(r'(http://127\.0\.0\.1:(\d+)/\?token=([\w-]+))', line)
            if m:
                url = m
        assert url is not None, "".join(lines) + proc.stderr.read()
        assert 'TIL Studio' in lines[0] and os.path.realpath(project) in lines[0]
        port, token = int(url.group(2)), url.group(3)
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        conn.request('GET', '/api/info', headers={'X-TIL-Token': token})
        r = conn.getresponse()
        assert r.status == 200 and json.loads(r.read())['root'] == os.path.realpath(project)
        conn.close()
        proc.send_signal(signal.SIGINT if os.name == 'posix' else signal.CTRL_C_EVENT)
        assert proc.wait(15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()
    r = subprocess.run([sys.executable, TIL, 'ide', str(tmp_path / 'missing'), '--no-browser'],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1 and 'no such folder' in r.stderr
    r = subprocess.run([sys.executable, TIL, 'ide', '--lang', 'ru', str(tmp_path / 'missing'), '--no-browser'],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1 and 'папк' in r.stderr
    r = subprocess.run([sys.executable, TIL, 'ide', '--port', 'x'], capture_output=True, text=True, env=env,
                       timeout=60)
    assert r.returncode == 2 and '--port' in r.stderr
    r = subprocess.run([sys.executable, TIL, 'ide', '--help'], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and 'usage: til ide' in r.stdout


def test_sigterm_stops_runs(project, tmp_path):
    if os.name != 'posix':
        pytest.skip("POSIX signals")
    proc = subprocess.Popen([sys.executable, os.path.join(SRC, 'tilstudio.py'), project, '--no-browser'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        m = None
        while m is None:
            line = proc.stdout.readline()
            assert line, proc.stderr.read()
            m = re.search(r'http://127\.0\.0\.1:(\d+)/\?token=([\w-]+)', line)
        port, token = int(m.group(1)), m.group(2)
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
        backend = 'cvm' if CC else 'vm'
        conn.request('POST', '/api/run', body=json.dumps({'path': 'loop.til', 'backend': backend}),
                     headers={'X-TIL-Token': token})
        assert conn.getresponse().status == 200
        time.sleep(1.5 if CC else 0.3)
        children = []
        if shutil.which('pgrep'):
            children = subprocess.run(['pgrep', '-P', str(proc.pid)], capture_output=True,
                                      text=True).stdout.split()
        proc.terminate()
        assert proc.wait(15) == 0
        time.sleep(0.2)
        assert not [pid for pid in children if _alive(int(pid))]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()
