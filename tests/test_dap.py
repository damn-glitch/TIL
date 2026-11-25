#!/usr/bin/env python3
"""
TIL Studio, part 4 (docs/dev/TIL_STUDIO_DESIGN.md): the debug adapter `til dap`
(src/tildap.py) and the debugging engine it is built on (tildebug.DebugSession).

* the protocol over a real `python3 src/til.py dap` subprocess: initialize /
  launch / configurationDone, seq numbers and message order, breakpoints (moved
  to the next line with code, conditions, hit counts), stop reasons, next /
  stepIn / stepOut, stepBack and reverseContinue restoring exactly the earlier
  state (locals, stack, output), til/goto to any line event, til/timeline,
  variables of nested values, evaluate (hover / watch / repl), runtime-error
  stops, input (til/stdin, til/inputRequested, the debug console, launch
  stdin), effects and retry, graphics (til/screen), pause, terminate,
  disconnect, malformed requests, imported files;
* DapSession in-process (events from the worker thread, concurrent requests,
  close);
* the engine: compile_program == compile_to_bytecode, conditions and hit
  counts backwards and forwards with tiny snapshot intervals, goto, evaluate,
  values, kill from another thread.

Run: python3 -m pytest tests/test_dap.py -v
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from tilhelpers import ROOT, SRC, TIL, cli, run_vm, til

sys.path.insert(0, SRC)
import tildap  # noqa: E402
import tildebug  # noqa: E402

TMP = tempfile.mkdtemp(prefix='til-dap-')
TIMEOUT = 60


def write(name: str, text: str) -> str:
    path = os.path.join(TMP, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


LOOP = """\
var total = 0
let names = ["a", "b"]

add(a, b)
    let s = a + b
    return s

main()
    for i in 0..4
        let sq = i * i
        total = add(total, sq)
        print(total)
    print("done")
"""

FACT = """\
fact(n: int) -> int
    if n <= 1
        return 1
    return n * fact(n - 1)

main()
    let r = fact(4)
    print(r)
"""

NESTED = """\
struct Point
    x: int
    y: int

enum Shape
    Circle(int)
    Rect(int, int)
    Empty

make_adder(k)
    return |x| x + k

main()
    let p = Point { x: 1, y: 2 }
    let pts = [Point { x: 3, y: 4 }, Point { x: 5, y: 6 }]
    let m = {"name": "Aigerim", "scores": [90, 85], "nested": {"a": [1, [2, 3]]}, 7: true}
    let s = Some(42)
    let none = None
    let ok = Ok("fine")
    let err = Err("bad")
    let shape = Shape.Rect(3, 4)
    let add = make_adder(10)
    let big = range(0, 250)
    let text = "line1\\nline2"
    print(len(pts))
"""

ERR = """\
divide(a, b)
    return a / b

main()
    for x in [4, 0]
        print(divide(8, x))
"""

INPUT = """\
main()
    let name = input("name? ")
    let age = input("age? ")
    print("hi " + name + " " + age)
"""

FOREVER = "main()\n    var n = 0\n    while true\n        n += 1\n"

EFFECTS = """\
greet() -> str
    let name = perform Ask("name?")
    perform Log("hello " + name)
    return name

main()
    var count = 0
    let r = handle
        greet()
        "done"
    with
        Ask(question) => resume("Aigerim")
        Log(msg) =>
            count += 1
            print(msg)
            resume(None)
    print(r)
    print(count)
"""

RETRY = """\
flaky(n: int) -> int
    if n < 3
        panic("not yet")
    return n * 10

main()
    var tries = 0
    let v = retry(5)
        tries += 1
        flaky(tries)
    else
        -1
    let k = 2
    let scaled = [1, 2, 3].map(|x| x * k + v)
    print(scaled)
"""

GFX = """\
main()
    screen_init(8, 6)
    screen_clear(0x102030)
    var x = 0
    while true
        let k = wait_key()
        if k == 27
            break
        if k == 4
            x += 1
        draw_pixel(x, 2, 0xFFFFFF)
        screen_show()
    print(get_pixel(2, 2))
"""


# ═══════════════════════════════════════════════════════════════════════════
# a small DAP client
# ═══════════════════════════════════════════════════════════════════════════

class DapClient:
    """Talks to a `til dap` subprocess. Every message received is kept in
    arrival order; event() hands out the events of one name one by one."""

    def __init__(self, *args, env=None, cwd=None):
        full = dict(os.environ)
        full.pop('TIL_LANG', None)
        if env:
            full.update(env)
        self.proc = subprocess.Popen([sys.executable, TIL, 'dap', *args], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=full, cwd=cwd)
        self.seq = 0
        self.messages = []
        self.taken = set()
        self.closed = False
        self.cond = threading.Condition()
        self.stderr = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self.proc.stderr:
            self.stderr.append(line)

    def _read(self):
        f = self.proc.stdout
        try:
            while True:
                headers = {}
                line = f.readline()
                if not line:
                    break
                while line.strip():
                    key, _, value = line.decode('ascii').partition(':')
                    headers[key.strip().lower()] = value.strip()
                    line = f.readline()
                if not headers:
                    continue
                body = f.read(int(headers['content-length']))
                msg = json.loads(body.decode('utf-8'))
                with self.cond:
                    self.messages.append(msg)
                    self.cond.notify_all()
        finally:
            with self.cond:
                self.closed = True
                self.cond.notify_all()

    def raw(self, data: bytes):
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def send(self, command: str, arguments=None, **extra) -> int:
        self.seq += 1
        msg = {'seq': self.seq, 'type': 'request', 'command': command}
        if arguments is not None:
            msg['arguments'] = arguments
        msg.update(extra)
        data = json.dumps(msg).encode('utf-8')
        self.raw(b"Content-Length: %d\r\n\r\n" % len(data) + data)
        return self.seq

    def _wait(self, pred, timeout=TIMEOUT):
        end = time.time() + timeout
        with self.cond:
            while True:
                for i, m in enumerate(self.messages):
                    if i not in self.taken and pred(m):
                        self.taken.add(i)
                        return m
                left = end - time.time()
                if left <= 0 or self.closed:
                    raise AssertionError(f"no such message; last ones: {self.messages[-6:]}\n"
                                         f"stderr: {b''.join(self.stderr)[-2000:]!r}")
                self.cond.wait(min(left, 0.5))

    def request(self, command: str, timeout=TIMEOUT, **arguments) -> dict:
        seq = self.send(command, arguments)
        return self._wait(lambda m: m.get('type') == 'response' and m.get('request_seq') == seq, timeout)

    def ok(self, command: str, **arguments) -> dict:
        r = self.request(command, **arguments)
        assert r['success'], r
        assert r['command'] == command
        return r.get('body', {})

    def fail(self, command: str, **arguments) -> str:
        r = self.request(command, **arguments)
        assert not r['success'], r
        assert r['message']
        return r['message']

    def event(self, name: str, timeout=TIMEOUT) -> dict:
        return self._wait(lambda m: m.get('type') == 'event' and m.get('event') == name, timeout)

    def events(self, name: str):
        with self.cond:
            return [m for m in self.messages if m.get('type') == 'event' and m.get('event') == name]

    def output(self, category='stdout') -> str:
        return "".join(m['body']['output'] for m in self.events('output') if m['body']['category'] == category)

    def wait_output(self, text: str, category='console'):
        end = time.time() + TIMEOUT
        while text not in self.output(category):
            assert time.time() < end, f"no output {text!r}: {self.output(category)!r}"
            time.sleep(0.01)

    def start(self, path, breakpoints=None, filters=None, init=None, **launch):
        options = {'clientID': 'pytest', 'adapterID': 'til', 'linesStartAt1': True, 'columnsStartAt1': True}
        options.update(init or {})
        self.capabilities = self.ok('initialize', **options)
        self.ok('launch', program=path, **launch)
        self.event('initialized')
        result = None
        if breakpoints is not None:
            result = self.ok('setBreakpoints', source={'path': path},
                             breakpoints=[{'line': b} if isinstance(b, int) else b for b in breakpoints])
        if filters is not None:
            self.ok('setExceptionBreakpoints', filters=filters)
        self.ok('configurationDone')
        return result

    def stopped(self) -> dict:
        return self.event('stopped')['body']

    def frames(self):
        return self.ok('stackTrace', threadId=1)['stackFrames']

    def stack(self):
        return [(f['name'], f['line']) for f in self.frames()]

    def scope(self, name='Locals', frame=0):
        frames = self.frames()
        scopes = self.ok('scopes', frameId=frames[frame]['id'])['scopes']
        ref = next(s['variablesReference'] for s in scopes if s['name'] == name)
        return self.variables(ref)

    def variables(self, ref, **kw):
        return {v['name']: v for v in self.ok('variables', variablesReference=ref, **kw)['variables']}

    def locals(self, frame=0):
        return {k: v['value'] for k, v in self.scope('Locals', frame).items()}

    def state(self):
        """Everything that must come back exactly after going back in time."""
        return self.stack(), self.locals(), len(self.output()), self.ok('til/timeline')['index']

    def evaluate(self, expression, context='watch', frame=0):
        return self.ok('evaluate', expression=expression, context=context,
                       frameId=self.frames()[frame]['id'])

    def close(self):
        if self.proc.poll() is None:
            try:
                self.request('disconnect', timeout=20)
            except (AssertionError, OSError, ValueError):
                pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        return self.proc.returncode


@pytest.fixture
def dap():
    clients = []

    def make(*args, **kw):
        c = DapClient(*args, **kw)
        clients.append(c)
        return c
    yield make
    for c in clients:
        c.close()


def index_of(client, pred):
    with client.cond:
        return next(i for i, m in enumerate(client.messages) if pred(m))


# ═══════════════════════════════════════════════════════════════════════════
# the protocol
# ═══════════════════════════════════════════════════════════════════════════

class TestProtocol:
    def test_initialize_and_launch_sequence(self, dap):
        c = dap()
        path = write("seq.til", LOOP)
        caps = c.ok('initialize', clientID='pytest', adapterID='til')
        for cap in ('supportsConfigurationDoneRequest', 'supportsStepBack', 'supportsEvaluateForHovers',
                    'supportsConditionalBreakpoints', 'supportsHitConditionalBreakpoints',
                    'supportsTerminateRequest'):
            assert caps[cap] is True, cap
        assert caps['supportsLoadedSourcesRequest'] is False
        assert caps['exceptionBreakpointFilters'][0]['filter'] == 'runtime-error'
        c.ok('launch', program=path, stopOnEntry=True)
        c.event('initialized')
        c.ok('configurationDone')
        assert c.stopped() == {'reason': 'entry', 'threadId': 1, 'allThreadsStopped': True}
        assert c.ok('threads') == {'threads': [{'id': 1, 'name': 'main'}]}
        assert c.stack() == [('__start__', 1)]
        frame = c.frames()[0]
        assert frame['source'] == {'name': 'seq.til', 'path': path} and frame['column'] == 1
        # seq: one counter, strictly increasing in the order of the stream; responses name
        # their request; the launch response comes before `initialized`, configurationDone's
        # before `stopped`
        seqs = [m['seq'] for m in c.messages]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        assert all(m['request_seq'] <= c.seq for m in c.messages if m['type'] == 'response')
        launch = index_of(c, lambda m: m.get('command') == 'launch')
        assert launch < index_of(c, lambda m: m.get('event') == 'initialized')
        assert index_of(c, lambda m: m.get('command') == 'configurationDone') < \
            index_of(c, lambda m: m.get('event') == 'stopped')

    def test_breakpoints_move_to_code_and_stop(self, dap):
        c = dap()
        path = write("bps.til", LOOP)
        bps = c.start(path, breakpoints=[3, 7, 11, 99])['breakpoints']
        assert [(b['verified'], b['line']) for b in bps] == [(True, 4), (True, 8), (True, 11), (False, 99)]
        assert "No code at or after line 99" in bps[3]['message']
        assert bps[0]['source']['path'] == path
        ids = [b['id'] for b in bps[:3]]
        assert len(set(ids)) == 3
        st = c.stopped()                        # line 4 never runs (add returns), 8 is main's end
        assert st['reason'] == 'breakpoint' and st['hitBreakpointIds'] == [ids[2]]
        assert c.stack()[0] == ('main', 11) and c.locals() == {'i': '0', 'sq': '0'}
        # setBreakpoints replaces the file's breakpoints; the same line keeps its id
        again = c.ok('setBreakpoints', source={'path': path}, breakpoints=[{'line': 11}])['breakpoints']
        assert again == [{'id': ids[2], 'verified': True, 'line': 11, 'source': bps[2]['source']}]
        c.ok('continue', threadId=1)
        st = c.stopped()
        assert st['hitBreakpointIds'] == [ids[2]] and c.locals() == {'i': '1', 'sq': '1'}
        assert c.ok('breakpointLocations', source={'path': path}, line=9, endLine=12)['breakpoints'] == \
            [{'line': 9}, {'line': 10}, {'line': 11}, {'line': 12}]
        c.ok('setBreakpoints', source={'path': path}, breakpoints=[])
        c.ok('continue', threadId=1)
        assert c.event('exited')['body'] == {'exitCode': 0}
        c.event('terminated')
        assert index_of(c, lambda m: m.get('event') == 'exited') < index_of(c, lambda m: m.get('event') == 'terminated')
        assert c.output() == run_vm(LOOP)[1]

    def test_responses_come_before_their_events(self, dap):
        c = dap()
        c.start(write("order.til", LOOP), stopOnEntry=True)
        c.stopped()
        for command in ('next', 'stepIn', 'next', 'stepBack', 'stepIn', 'stepOut'):
            seq = c.send(command, {'threadId': 1})
            c.event('stopped')
            resp = index_of(c, lambda m: m.get('type') == 'response' and m.get('request_seq') == seq)
            last_stop = max(i for i, m in enumerate(c.messages) if m.get('event') == 'stopped')
            assert resp < last_stop, command

    def test_next_step_in_step_out(self, dap):
        c = dap()
        c.start(write("steps.til", LOOP), breakpoints=[11])
        c.stopped()
        lines = []
        for command in ('stepIn', 'stepIn', 'stepIn', 'next', 'next', 'next'):
            c.ok(command, threadId=1)
            assert c.stopped()['reason'] in ('step', 'breakpoint')
            lines.append(c.stack()[0])
        assert lines == [('add', 5), ('add', 6), ('main', 12), ('main', 9), ('main', 10), ('main', 11)]
        c.ok('stepIn', threadId=1)
        c.stopped()
        assert c.stack() == [('add', 5), ('main', 11), ('__start__', 2)]
        c.ok('stepOut', threadId=1)
        assert c.stopped()['reason'] == 'step'
        assert c.stack()[0] == ('main', 11)
        # the value the call returned: a read-only pseudo variable in Locals
        scope = c.scope()
        assert scope['add() returned']['value'] == '1'
        assert scope['add() returned']['presentationHint']['kind'] == 'virtual'
        assert c.locals()['i'] == '1'

    def test_step_back_and_reverse_continue_restore_the_state(self, dap):
        c = dap()
        c.start(write("back.til", LOOP), breakpoints=[11])
        c.stopped()
        history = [c.state()]
        for command in ('next', 'next', 'stepIn', 'next', 'next', 'next', 'next', 'stepIn'):
            c.ok(command, threadId=1)
            c.stopped()
            history.append(c.state())
        # stepBack is the reverse of next (and of stepIn at a function's first line)
        for expected in reversed(history[:-1]):
            c.ok('stepBack', threadId=1)
            assert c.stopped()['reason'] == 'step'
            stack, local, _out, index = c.state()
            assert (stack, local, index) == expected[:2] + expected[3:]
        # reverseContinue: back to the previous breakpoint hit
        printed = len(c.output())
        c.ok('continue', threadId=1)
        c.stopped()
        c.ok('continue', threadId=1)
        c.stopped()
        at_third = c.state()
        assert at_third[1] == {'i': '2', 'sq': '4'}
        c.ok('reverseContinue', threadId=1)
        st = c.stopped()
        assert st['reason'] == 'breakpoint'
        assert c.locals() == {'i': '1', 'sq': '1'}
        c.ok('continue', threadId=1)
        c.stopped()
        assert c.state()[:2] == at_third[:2]
        assert len(c.output()) >= printed

    def test_output_exactly_once_after_time_travel(self, dap):
        c = dap()
        c.start(write("once.til", LOOP), breakpoints=[12])
        for _ in range(3):
            c.stopped()
            c.ok('continue', threadId=1)
        c.stopped()
        for command in ('stepBack',) * 6 + ('reverseContinue', 'reverseContinue', 'next', 'stepIn'):
            c.ok(command, threadId=1)
            c.stopped()
        c.ok('til/goto', index=0)
        c.stopped()
        c.ok('setBreakpoints', source={'path': os.path.join(TMP, "once.til")}, breakpoints=[])
        c.ok('continue', threadId=1)
        c.event('terminated')
        assert c.output() == run_vm(LOOP)[1] == "0\n1\n5\n14\ndone\n"

    def test_goto_any_line_event(self, dap):
        c = dap()
        path = write("goto.til", FACT)
        c.start(path, stopOnEntry=True)
        c.stopped()
        seen = []
        while True:
            index = c.ok('til/timeline')['index']
            seen.append((index, c.stack(), c.locals()))
            c.ok('stepIn', threadId=1)
            if c.event('stopped', timeout=TIMEOUT)['body']['reason'] != 'step':
                break
            if c.stack()[0] == ('main', 6):         # the implicit return of main: the last line
                index = c.ok('til/timeline')['index']
                seen.append((index, c.stack(), c.locals()))
                break
        timeline = c.ok('til/timeline')
        assert timeline['total'] == len(seen) and timeline['index'] == len(seen) - 1
        assert [s[0] for s in seen] == list(range(len(seen)))
        for target in (5, 0, len(seen) - 1, 3, 7, 2):
            c.ok('til/goto', index=target)
            assert c.event('continued')['body'] == {'threadId': 1, 'allThreadsContinued': True}
            assert c.stopped()['reason'] == 'goto'
            assert (c.ok('til/timeline')['index'], c.stack(), c.locals()) == seen[target]
        assert "No line event 99" in c.fail('til/goto', index=99)
        assert c.fail('til/goto', index='x')
        assert c.output() == "24\n"                # printed once, while stepping to the end
        c.ok('continue', threadId=1)
        c.event('terminated')
        assert c.output() == "24\n"

    def test_variables_of_nested_values(self, dap):
        c = dap()
        c.start(write("nested.til", NESTED), breakpoints=[25])
        c.stopped()
        scope = c.scope()
        assert scope['p']['value'] == 'Point { x: 1, y: 2 }' and scope['p']['type'] == 'Point'
        p = c.variables(scope['p']['variablesReference'])
        assert {k: (v['value'], v['evaluateName']) for k, v in p.items()} == {'x': ('1', 'p.x'), 'y': ('2', 'p.y')}
        pts = c.variables(scope['pts']['variablesReference'])
        assert scope['pts']['indexedVariables'] == 2
        second = c.variables(pts['[1]']['variablesReference'])
        assert second['y']['evaluateName'] == 'pts[1].y' and second['y']['value'] == '6'
        m = c.variables(scope['m']['variablesReference'])
        assert list(m) == ['"name"', '"scores"', '"nested"', '7']
        assert m['"name"']['value'] == '"Aigerim"' and m['7']['value'] == 'true'
        nested = c.variables(m['"nested"']['variablesReference'])
        a = c.variables(nested['"a"']['variablesReference'])
        inner = c.variables(a['[1]']['variablesReference'])
        assert inner['[1]']['value'] == '3' and inner['[1]']['evaluateName'] == 'm["nested"]["a"][1][1]'
        assert c.evaluate(inner['[1]']['evaluateName'])['result'] == '3'
        s = c.variables(scope['s']['variablesReference'])
        assert scope['s']['value'] == 'Some(42)' and s['value']['value'] == '42'
        assert scope['none'] == {'name': 'none', 'value': 'None', 'type': 'nil', 'variablesReference': 0,
                                 'evaluateName': 'none'}
        assert c.variables(scope['err']['variablesReference'])['error']['value'] == '"bad"'
        assert c.variables(scope['ok']['variablesReference'])['value']['value'] == '"fine"'
        shape = c.variables(scope['shape']['variablesReference'])
        assert scope['shape']['value'] == 'Shape.Rect(3, 4)'
        assert (shape['0']['value'], shape['1']['evaluateName']) == ('3', 'shape[1]')
        add = c.variables(scope['add']['variablesReference'])
        assert scope['add']['type'] == 'fn' and add['k']['value'] == '10'
        assert scope['text']['value'] == '"line1\\nline2"'
        # big arrays page (VS Code asks for 100 at a time)
        big = scope['big']
        assert big['indexedVariables'] == 250 and big['value'].endswith("...")
        page = c.ok('variables', variablesReference=big['variablesReference'], filter='indexed',
                    start=200, count=5)['variables']
        assert [v['name'] for v in page] == ['[200]', '[201]', '[202]', '[203]', '[204]']
        assert c.ok('variables', variablesReference=big['variablesReference'], filter='named')['variables'] == []
        glob = c.scope('Globals')
        assert glob == {}
        # references end when the program goes on
        ref = scope['p']['variablesReference']
        c.ok('next', threadId=1)
        c.stopped()
        assert "variablesReference" in c.fail('variables', variablesReference=ref)

    def test_evaluate(self, dap):
        c = dap()
        c.start(write("eval.til", LOOP), breakpoints=[5])
        c.stopped()
        c.ok('continue', threadId=1)
        c.stopped()                                     # add(1, 1): a = 1, b = 1 (second call)
        assert c.locals() == {'a': '0', 'b': '1'}
        assert c.evaluate('a + b * 10')['result'] == '10'
        assert c.evaluate('names', context='hover')['result'] == '["a", "b"]'
        names = c.evaluate('names')
        assert names['variablesReference'] > 0 and names['indexedVariables'] == 2
        assert c.evaluate('names[1] == "b" and total >= 0')['result'] == 'true'
        assert c.evaluate('len(names) + 1')['result'] == '3'
        assert c.evaluate('i', frame=1)['result'] == '1'       # the caller's frame
        assert c.evaluate('f"{a}-{b}"', context='repl')['result'] == '"0-1"'
        r = c.request('evaluate', expression='nope', context='hover', frameId=c.frames()[0]['id'])
        assert not r['success'] and r['message'] == "No variable 'nope' here"
        assert "Cannot evaluate 'add(1, 2)'" in c.fail('evaluate', expression='add(1, 2)', context='repl')
        assert "Cannot evaluate 'names[7]'" in c.fail('evaluate', expression='names[7]')
        assert c.fail('evaluate', expression='a = 3')
        assert c.fail('evaluate', context='repl')
        # nothing an expression does reaches the program
        assert c.locals() == {'a': '0', 'b': '1'}

    def test_conditional_and_hit_count_breakpoints(self, dap):
        c = dap()
        path = write("cond.til", LOOP)
        bps = c.start(path, breakpoints=[{'line': 11, 'condition': 'i == 2'}, {'line': 12, 'hitCondition': '% 2'},
                                         {'line': 10, 'condition': 'i +'}, {'line': 6, 'hitCondition': 'often'}])
        bps = bps['breakpoints']
        assert [b['verified'] for b in bps] == [True, True, False, False]
        assert "Cannot evaluate 'i +'" in bps[2]['message'] and "Invalid hit condition 'often'" in bps[3]['message']
        stops = []
        while True:
            m = c._wait(lambda m: m.get('event') in ('stopped', 'terminated'))
            if m['event'] == 'terminated':
                break
            stops.append((c.stack()[0][1], c.locals()['i'], m['body']['hitBreakpointIds'][0]))
            c.ok('continue', threadId=1)
        # line 12 on its 2nd and 4th time; line 11 when i == 2
        assert stops == [(12, '1', bps[1]['id']), (11, '2', bps[0]['id']), (12, '3', bps[1]['id'])]

    def test_condition_that_fails_stops_and_says_why(self, dap):
        c = dap()
        c.start(write("condfail.til", LOOP), breakpoints=[{'line': 11, 'condition': 'missing > 1'}])
        assert c.stopped()['reason'] == 'breakpoint'
        assert "No variable 'missing' here" in c.output('console')

    def test_reverse_continue_with_a_condition(self, dap):
        c = dap()
        path = write("rcond.til", "main()\n    var t = 0\n    for i in 0..300\n        t += i\n    print(t)\n")
        c.start(path, breakpoints=[{'line': 4, 'condition': 'i % 50 == 7'}], snapshotEvery=16, stopAtEnd=True)
        seen = []
        while True:
            st = c.stopped()
            if st['reason'] == 'exit':
                break
            seen.append(c.locals()['i'])
            c.ok('continue', threadId=1)
        assert seen == ['7', '57', '107', '157', '207', '257']
        back = []
        for _ in range(6):
            c.ok('reverseContinue', threadId=1)
            assert c.stopped()['reason'] == 'breakpoint'
            back.append(c.locals()['i'])
        assert back == list(reversed(seen))
        c.ok('reverseContinue', threadId=1)
        c.stopped()
        assert "No earlier breakpoint hit" in c.output('console')
        assert c.ok('til/timeline')['index'] == 0

    def test_runtime_error_stops_with_the_filter(self, dap):
        c = dap()
        c.start(write("err.til", ERR), filters=['runtime-error'])
        st = c.stopped()
        assert st['reason'] == 'exception' and st['text'] == "runtime error: division by zero"
        assert st['description'] == "Paused on exception"
        info = c.ok('exceptionInfo', threadId=1)
        assert info['exceptionId'] == 'runtime-error' and info['breakMode'] == 'unhandled'
        assert "at divide (line 2)" in info['details']['stackTrace']
        assert c.stack()[:2] == [('divide', 2), ('main', 6)]
        assert c.locals() == {'a': '8', 'b': '0'}
        assert c.output('stderr').count("runtime error: division by zero") == 1
        c.ok('stepBack', threadId=1)                   # the start of the line that failed
        c.stopped()
        assert c.stack()[0] == ('divide', 2) and c.ok('til/timeline')['index'] < c.ok('til/timeline')['total']
        c.ok('continue', threadId=1)
        assert c.stopped()['reason'] == 'exception'     # the same error again (not reported again)
        c.ok('continue', threadId=1)
        assert c.event('exited')['body'] == {'exitCode': 1}
        c.event('terminated')
        assert c.output() == "2\n" and c.output('stderr').count("division by zero") == 1

    def test_runtime_error_without_the_filter(self, dap):
        c = dap()
        c.start(write("err2.til", ERR), filters=[])
        assert c.event('exited')['body'] == {'exitCode': 1}
        c.event('terminated')
        assert not c.events('stopped')
        assert "runtime error: division by zero\n  at divide (line 2)" in c.output('stderr')

    def test_input_through_til_stdin(self, dap):
        c = dap()
        path = write("input.til", INPUT)
        c.start(path, breakpoints=[4])
        c.event('til/inputRequested')
        assert c.output() == "name? "
        c.wait_output("waiting for input")
        c.ok('til/stdin', text="Aigerim\n")
        c.event('til/inputRequested')
        assert c.ok('evaluate', expression="30", context='repl') == {'result': '', 'variablesReference': 0}
        c.stopped()
        assert c.locals() == {'name': '"Aigerim"', 'age': '"30"'}
        # back over the input() calls: replayed, not asked again
        for _ in range(2):
            c.ok('stepBack', threadId=1)
            c.stopped()
        assert c.ok('til/timeline')['index'] == 0
        assert c.fail('stepBack', threadId=1) == "Already at the start of the program"
        c.ok('setBreakpoints', source={'path': path}, breakpoints=[])
        c.ok('continue', threadId=1)
        c.event('terminated')
        assert len(c.events('til/inputRequested')) == 2
        assert c.output() == "name? age? hi Aigerim 30\n"

    def test_launch_stdin_is_the_whole_input(self, dap):
        c = dap()
        c.start(write("input2.til", INPUT), stdin="Ruslan\n42\n")
        c.event('terminated')
        assert c.output() == "name? age? hi Ruslan 42\n"
        assert not c.events('til/inputRequested')
        c2 = dap()
        c2.start(write("input3.til", INPUT), stdin="")
        c2.event('terminated')
        assert c2.output() == "name? age? hi  \n"

    def test_effects(self, dap):
        c = dap()
        path = write("effects.til", EFFECTS)
        c.start(path, breakpoints=[14])
        st = c.stopped()
        assert st['reason'] == 'breakpoint'
        assert c.stack() == [('<clause Log:13>', 14), ('greet', 3), ('main', 9), ('__start__', 0)]
        assert c.locals() == {'msg': '"hello Aigerim"', 'count': '0'}
        c.ok('next', threadId=1)
        c.stopped()
        c.ok('next', threadId=1)
        c.stopped()
        assert c.locals()['count'] == '1' and c.output() == "hello Aigerim\n"
        before = c.state()
        c.ok('stepBack', threadId=1)
        c.stopped()
        c.ok('next', threadId=1)
        c.stopped()
        assert c.state() == before
        c.ok('reverseContinue', threadId=1)
        c.stopped()
        assert c.locals() == {'msg': '"hello Aigerim"', 'count': '0'}
        c.ok('stepBack', threadId=1)                    # back through the perform
        c.stopped()
        assert c.stack()[0] == ('greet', 3)
        c.ok('setBreakpoints', source={'path': path}, breakpoints=[])
        c.ok('continue', threadId=1)
        c.event('terminated')
        assert c.output() == run_vm(EFFECTS)[1] == "hello Aigerim\ndone\n1\n"

    def test_retry_and_closures(self, dap):
        c = dap()
        path = write("retry.til", RETRY)
        c.start(path, breakpoints=[{'line': 2, 'condition': 'n == 3'}, 14])
        c.stopped()
        assert c.locals() == {'n': '3'} and c.stack()[1] == ('main', 10)
        c.ok('continue', threadId=1)
        c.stopped()
        c.ok('stepIn', threadId=1)                      # through the prelude's map into the lambda
        c.stopped()
        frames = c.frames()
        assert frames[0]['name'] == '<lambda:14>' and frames[1]['name'] == '__prelude_map'
        assert frames[1]['presentationHint'] == 'subtle' and 'source' not in frames[1]
        assert c.locals() == {'x': '1', 'k': '2', 'v': '30'}
        assert c.evaluate('x * k + v')['result'] == '32'
        c.ok('setBreakpoints', source={'path': path}, breakpoints=[])
        c.ok('continue', threadId=1)
        c.event('terminated')
        assert c.output() == "[32, 34, 36]\n"

    def test_graphics_screen_events(self, dap):
        path = write("gfx.til", GFX)
        ppm = os.path.join(TMP, "gfx.ppm")
        r = cli('run', '--vm', path, '--keys', 'RIGHT,RIGHT,a,RIGHT,ESC', '--screen', ppm)
        assert r.returncode == 0, r.stderr
        with open(ppm, 'rb') as f:
            final = f.read().split(b"\n", 3)[3]
        c = dap()
        c.start(path, keys="RIGHT,RIGHT,a,RIGHT,ESC", breakpoints=[13])
        c.stopped()
        frames = c.events('til/screen')
        assert frames and all((f['body']['w'], f['body']['h']) == (8, 6) for f in frames)
        assert base64.b64decode(frames[-1]['body']['rgb_b64']) == final
        # going back in time: the screen as it was then (before the last draw_pixel)
        c.ok('setBreakpoints', source={'path': path}, breakpoints=[{'line': 11}, {'line': 13}])
        c.ok('reverseContinue', threadId=1)
        c.stopped()
        assert c.stack()[0] == ('main', 11) and c.locals()['x'] == '3'
        shown = base64.b64decode(c.events('til/screen')[-1]['body']['rgb_b64'])
        assert shown != final and len(shown) == 8 * 6 * 3
        c.ok('setBreakpoints', source={'path': path}, breakpoints=[])
        c.ok('continue', threadId=1)
        c.event('terminated')
        assert base64.b64decode(c.events('til/screen')[-1]['body']['rgb_b64']) == final
        assert c.output() == r.stdout

    def test_pause_and_terminate_a_running_program(self, dap):
        c = dap()
        c.start(write("forever.til", FOREVER))
        time.sleep(0.3)
        c.ok('pause', threadId=1)
        assert c.stopped()['reason'] == 'pause'
        n = int(c.evaluate('n')['result'])
        assert n > 0 and c.stack()[0][0] == 'main'
        c.ok('continue', threadId=1)
        time.sleep(0.2)
        c.ok('terminate')
        c.event('terminated')
        assert not c.events('exited')
        assert c.fail('next', threadId=1) == "The program has ended"
        c.ok('disconnect')
        assert c.proc.wait(20) == 0

    def test_terminate_while_waiting_for_input(self, dap):
        c = dap()
        c.start(write("wait.til", INPUT))
        c.event('til/inputRequested')
        c.ok('terminate')
        c.event('terminated')
        c.ok('disconnect')
        assert c.proc.wait(20) == 0

    def test_disconnect_while_running_ends_the_adapter(self, dap):
        c = dap()
        c.start(write("forever2.til", FOREVER))
        time.sleep(0.2)
        c.ok('disconnect', terminateDebuggee=True)
        assert c.proc.wait(20) == 0

    def test_malformed_requests(self, dap):
        c = dap()
        assert c.fail('frobnicate') == "Unknown request 'frobnicate'"
        assert c.fail('stackTrace', threadId=1) == "No program has been launched"
        assert c.fail('configurationDone') == "No program has been launched"
        r = c.request('evaluate', expression=None)
        assert not r['success']
        seq = c.send('launch', [1, 2])                  # arguments that are not an object
        r = c._wait(lambda m: m.get('request_seq') == seq)
        assert not r['success'] and "must be an object" in r['message']
        c.raw(b"Content-Length: 9\r\n\r\n{not json")
        c.event('output')
        assert "Cannot read a message: invalid JSON" in c.output('console')
        assert c.fail('launch', program=os.path.join(TMP, "missing.til")).startswith("File not found")
        assert c.fail('launch', program=write("x.til", LOOP), args=5) == \
            "The argument 'args' must be a list of strings"
        assert "bad --keys entry" in c.fail('launch', program=write("x.til", LOOP), keys="UP,NOPE")
        msg = c.fail('launch', program=write("syntax.til", "main()\n    let = 3\n"))
        assert "error" in msg.lower()
        assert "syntax.til" in c.output('stderr')
        assert c.fail('attach').startswith("attach is not supported")
        # the session still works
        c.ok('initialize')
        c.ok('launch', program=write("x.til", LOOP))
        c.event('initialized')
        assert c.fail('launch', program=write("x.til", LOOP)) == "The program has already been launched"
        assert c.fail('next', threadId=1).startswith("The program has not started yet")
        assert c.fail('setBreakpoints', breakpoints=[{'line': 3}]).startswith("The argument 'source'")
        assert c.fail('setBreakpoints', source={'path': os.path.join(TMP, "x.til")}, breakpoints=[{'line': 'a'}])
        other = c.ok('setBreakpoints', source={'path': os.path.join(TMP, "other.til")}, breakpoints=[{'line': 1}])
        assert other['breakpoints'][0]['verified'] is False
        c.ok('configurationDone')
        c.event('terminated')
        assert c.fail('variables', variablesReference=12345)

    def test_stop_at_end_then_travel_back(self, dap):
        c = dap()
        path = write("end.til", LOOP)
        c.start(path, stopAtEnd=True)
        st = c.stopped()
        assert st['reason'] == 'exit' and st['description'] == "Program exited with code 0"
        timeline = c.ok('til/timeline')
        assert timeline['complete'] and timeline['index'] == timeline['total']
        c.ok('stepBack', threadId=1)                    # the last line that ran: main's implicit return
        c.stopped()
        assert c.stack()[0] == ('main', 8)
        c.ok('til/goto', index=timeline['total'])       # and to the end again
        assert c.stopped()['reason'] == 'exit'
        c.ok('continue', threadId=1)
        c.event('terminated')
        assert c.output() == "0\n1\n5\n14\ndone\n"

    def test_no_debug_ignores_breakpoints(self, dap):
        c = dap()
        c.start(write("nodebug.til", LOOP), breakpoints=[11], noDebug=True, stopOnEntry=True)
        assert c.event('exited')['body'] == {'exitCode': 0}
        c.event('terminated')
        assert not c.events('stopped') and c.output() == "0\n1\n5\n14\ndone\n"

    def test_args_cwd_and_files(self, dap):
        folder = tempfile.mkdtemp(prefix='til-dap-cwd-', dir=TMP)
        with open(os.path.join(folder, "data.txt"), 'w') as f:
            f.write("from the file")
        prog = write("files.til", 'main()\n    print(args())\n    print(read_file("data.txt"))\n'
                                  '    write_file("out.txt", "written")\n')
        c = dap()
        c.start(prog, args=["x", "y z"], cwd=folder)
        c.event('terminated')
        assert c.output() == '["x", "y z"]\nfrom the file\n'
        with open(os.path.join(folder, "out.txt")) as f:
            assert f.read() == "written"

    def test_imported_files(self, dap):
        folder = tempfile.mkdtemp(prefix='til-dap-mod-', dir=TMP)
        mod = os.path.join(folder, "helpers.til")
        with open(mod, 'w', encoding='utf-8') as f:
            f.write("# helpers\n\ntwice(x)\n    let y = x * 2\n    return y\n")
        main = os.path.join(folder, "main.til")
        with open(main, 'w', encoding='utf-8') as f:
            f.write("import helpers\n\nmain()\n    let a = twice(3)\n    print(a)\n")
        c = dap()
        c.ok('initialize')
        c.ok('launch', program=main)
        c.event('initialized')
        r = c.ok('setBreakpoints', source={'path': mod}, breakpoints=[{'line': 4}])['breakpoints']
        assert r[0]['verified'] and r[0]['line'] == 4 and r[0]['source']['path'] == mod
        r = c.ok('setBreakpoints', source={'path': main}, breakpoints=[{'line': 5}])['breakpoints']
        assert r[0]['verified'] and r[0]['line'] == 5
        c.ok('configurationDone')
        c.stopped()
        frames = c.frames()
        assert (frames[0]['name'], frames[0]['line'], frames[0]['source']['path']) == ('twice', 4, mod)
        assert (frames[1]['name'], frames[1]['line'], frames[1]['source']['path']) == ('main', 4, main)
        assert c.locals() == {'x': '3'}
        c.ok('continue', threadId=1)                    # line 5 of main, not of helpers.til
        c.stopped()
        assert c.stack()[0] == ('main', 5) and c.locals() == {'a': '6'}
        sources = c.ok('loadedSources')['sources']
        assert {s['path'] for s in sources} == {main, mod}

    def test_messages_in_the_program_language(self, dap):
        c = dap()
        c.start(write("ru.til", LOOP), breakpoints=[11], lang='ru')
        c.stopped()
        msg = c.fail('evaluate', expression='nope', frameId=c.frames()[0]['id'])
        assert msg == "Здесь нет переменной 'nope'"
        c2 = dap()
        c2.ok('initialize', locale='kk-KZ')
        assert c2.fail('stackTrace', threadId=1) == "Бағдарлама іске қосылмаған"

    def test_lines_start_at_zero(self, dap):
        c = dap()
        path = write("zero.til", LOOP)
        bps = c.start(path, breakpoints=[10], init={'linesStartAt1': False})['breakpoints']
        assert bps[0]['line'] == 10
        c.stopped()
        assert c.stack()[0] == ('main', 10)                 # line 11 of the file, 0-based
        assert c.locals() == {'i': '0', 'sq': '0'}


# ═══════════════════════════════════════════════════════════════════════════
# in-process: the API of TIL Studio
# ═══════════════════════════════════════════════════════════════════════════

class Collector:
    def __init__(self):
        self.events = []
        self.cond = threading.Condition()
        self.threads = set()

    def __call__(self, event):
        with self.cond:
            self.events.append(event)
            self.threads.add(threading.current_thread().name)
            self.cond.notify_all()

    def wait(self, name, timeout=TIMEOUT, start=0):
        end = time.time() + timeout
        with self.cond:
            while True:
                for e in self.events[start:]:
                    if e['event'] == name:
                        return e
                left = end - time.time()
                assert left > 0, f"no {name} event in {self.events}"
                self.cond.wait(left)


class TestInProcess:
    def request(self, session, command, **arguments):
        self.seq = getattr(self, 'seq', 0) + 1
        responses = session.handle({'seq': self.seq, 'type': 'request', 'command': command,
                                    'arguments': arguments})
        assert len(responses) == 1 and responses[0]['request_seq'] == self.seq
        return responses[0]

    def test_session_api(self):
        events = Collector()
        root = tempfile.mkdtemp(prefix='til-dap-root-', dir=TMP)
        with open(os.path.join(root, "prog.til"), 'w', encoding='utf-8') as f:
            f.write(LOOP)
        s = tildap.DapSession(events, root)
        assert self.request(s, 'initialize')['body']['supportsStepBack']
        assert self.request(s, 'launch', program="prog.til")['success']      # relative to the root
        events.wait('initialized')
        r = self.request(s, 'setBreakpoints', source={'path': "prog.til"}, breakpoints=[{'line': 11}])
        assert r['body']['breakpoints'][0]['source'] == {'name': 'prog.til', 'path': os.path.join(root, "prog.til")}
        self.request(s, 'configurationDone')
        events.wait('stopped')
        assert any(name == "til-dap-program" for name in events.threads)
        top = self.request(s, 'stackTrace', threadId=1)['body']['stackFrames'][0]
        assert (top['name'], top['line']) == ('main', 11)
        # requests from several threads at once (the studio's HTTP server is threaded)
        results = []

        def poll():
            for _ in range(20):
                results.append(self_request(s, 'til/timeline')['success'])

        def self_request(session, command):
            return session.handle({'seq': 1, 'type': 'request', 'command': command})[0]
        threads = [threading.Thread(target=poll) for _ in range(4)]
        for t in threads:
            t.start()
        n = len(events.events)
        self.request(s, 'continue', threadId=1)
        for t in threads:
            t.join()
        assert all(results) and len(results) == 80
        events.wait('stopped', start=n)
        top = self.request(s, 'stackTrace', threadId=1)['body']['stackFrames'][0]
        assert self.request(s, 'evaluate', expression='i', frameId=top['id'])['body']['result'] == '1'
        s.close()
        events.wait('terminated')
        assert not s._worker.is_alive()
        assert self.request(s, 'til/timeline')['body']['total'] > 0

    def test_events_are_not_held_back_by_busy_clients(self):
        """Requests that keep arriving from several threads (a threaded HTTP server,
        a timeline that polls) never keep the events back."""
        events = Collector()
        s = tildap.DapSession(events)
        path = write("busy.til", "main()\n    var t = 0\n    for i in 0..20000\n        t += i\n    print(t)\n")
        s.handle({'seq': 1, 'type': 'request', 'command': 'initialize', 'arguments': {}})
        s.handle({'seq': 2, 'type': 'request', 'command': 'launch', 'arguments': {'program': path}})
        s.handle({'seq': 3, 'type': 'request', 'command': 'setBreakpoints',
                  'arguments': {'source': {'path': path}, 'breakpoints': [{'line': 5}]}})
        done = threading.Event()

        def poll():
            while not done.is_set():
                with s.hold_events():
                    s.handle({'seq': 9, 'type': 'request', 'command': 'til/timeline'})
                    time.sleep(0.002)
        threads = [threading.Thread(target=poll) for _ in range(3)]
        for t in threads:
            t.start()
        try:
            s.handle({'seq': 4, 'type': 'request', 'command': 'configurationDone', 'arguments': {}})
            assert events.wait('stopped')['body']['reason'] == 'breakpoint'
        finally:
            done.set()
            for t in threads:
                t.join()
        s.close()

    def test_events_wait_for_hold_events(self):
        sent = []
        s = tildap.DapSession(sent.append)
        path = write("hold.til", LOOP)
        s.handle({'seq': 1, 'type': 'request', 'command': 'initialize', 'arguments': {}})
        with s.hold_events():
            responses = s.handle({'seq': 2, 'type': 'request', 'command': 'launch',
                                  'arguments': {'program': path, 'stopOnEntry': True}})
            sent.extend(responses)
            assert [m.get('event') for m in sent] == [None]     # initialized waits for the block
        assert [m.get('command') or m.get('event') for m in sent] == ['launch', 'initialized']
        assert sent[0]['seq'] < sent[1]['seq']
        s.handle({'seq': 3, 'type': 'request', 'command': 'configurationDone', 'arguments': {}})
        deadline = time.time() + TIMEOUT
        while not any(m.get('event') == 'stopped' for m in sent) and time.time() < deadline:
            time.sleep(0.01)
        assert any(m.get('event') == 'stopped' for m in sent)
        assert s.handle({'type': 'event', 'event': 'whatever'}) == []
        assert s.handle("garbage") == []
        s.close()


# ═══════════════════════════════════════════════════════════════════════════
# the engine (tildebug.DebugSession)
# ═══════════════════════════════════════════════════════════════════════════

class TestEngine:
    def test_compile_program_gives_the_bytes_of_compile_to_bytecode(self):
        names = ['01_hello.til', '04_structs.til', '06_enums.til', '08_imports.til', '12_effects.til']
        for name in names:
            path = os.path.join(ROOT, 'examples', name)
            with open(path, encoding='utf-8') as f:
                src = f.read()
            for fuse in (True, False):
                module, info = tildebug.compile_program(src, path, fuse)
                assert module.serialize() == til.compile_to_bytecode(src, path, True, fuse, debug_info=True).serialize()
        _m, info = tildebug.compile_program(open(os.path.join(ROOT, 'examples', '08_imports.til')).read(),
                                            os.path.join(ROOT, 'examples', '08_imports.til'))
        assert os.path.join(ROOT, 'examples', 'math_utils.til') in info.func_files

    def test_stops_and_state(self):
        out = []
        s = tildebug.DebugSession.load(write("engine.til", LOOP), out=out.append)
        st = s.start()
        assert (st.reason, st.function, st.line, st.event) == ('entry', '__start__', 1, 0)
        bp = s.add_breakpoint(11, condition='sq > 3')
        st = s.cont()
        assert st.reason == 'breakpoint' and st.fired is bp and s.locals() == [('i', 2), ('sq', 4)]
        assert b"".join(out) == b"0\n1\n"
        assert [(f.name, f.line) for f in s.stack()] == [('main', 11), ('__start__', 2)]
        assert s.globals() == [('total', 1), ('names', [b"a", b"b"])]
        assert s.evaluate('sq * 2 + total') == 9 and s.evaluate('names[0]') == b"a"
        st = s.step_in()
        assert (st.function, st.line, st.reason) == ('add', 5, 'step') and s.params() == [('a', 1), ('b', 4)]
        st = s.step_out()
        assert st.returned == ('add', 5) and not st.at_event and s.stack()[0].line == 11
        st = s.step_back_over()                         # back to the start of line 11
        assert st.at_event and st.line == 11 and st.event == 16
        assert s.timeline() == (16, len(s.vm.ev_lines))
        st = s.goto(3)
        assert (st.reason, st.event, st.line) == ('goto', 3, 10)
        s.remove_breakpoint(bp)
        st = s.cont()
        assert st.finished and st.reason == 'exit' and st.exit_code == 0 and s.end_seen
        assert bytes(s.output) == b"0\n1\n5\n14\ndone\n"
        with pytest.raises(tildebug.DebugError, match="has finished"):
            s.step_over()

    def test_hit_counts_and_conditions_backwards_with_tiny_snapshot_intervals(self):
        src = "main()\n    var t = 0\n    for i in 0..400\n        t += i\n        let d = t % 7\n    print(t)\n"
        for every in (1, 7, 100):
            s = tildebug.DebugSession.load(write("hits.til", src), every=every)
            s.add_breakpoint(5, condition='t % 7 == 3')
            s.add_breakpoint(4, hit_condition='% 97')
            forward = []
            while True:
                st = s.cont()
                if st.finished:
                    break
                forward.append((st.line, s.lookup('i')))
            assert len(forward) > 10
            backward = []
            for _ in forward:
                st = s.reverse_continue()
                assert st.reason == 'breakpoint'
                backward.append((st.line, s.lookup('i')))
            assert backward == list(reversed(forward)), every
            st = s.reverse_continue()
            assert st.note and st.event == 0

    def test_step_back_over_mirrors_step_over_and_step_in(self):
        s = tildebug.DebugSession.load(write("mirror.til", FACT), every=3)
        s.start()
        states = []
        for step in ('in', 'in', 'in', 'in', 'in', 'over', 'in', 'in'):
            states.append(([(f.name, f.line) for f in s.stack()], s.locals(), s.timeline()[0]))
            st = s.step_in() if step == 'in' else s.step_over()
            assert not st.finished
        for expected in reversed(states):
            s.step_back_over()
            assert ([(f.name, f.line) for f in s.stack()], s.locals(), s.timeline()[0]) == expected
        with pytest.raises(tildebug.DebugError, match="Already at the start"):
            s.step_back_over()

    def test_evaluate_supported_syntax(self):
        s = tildebug.DebugSession.load(write("evalsyntax.til", NESTED))
        s.add_breakpoint(25)
        s.cont()
        cases = {
            'p.x': 1, 'pts[1].y': 6, 'm["nested"]["a"][1][0]': 2, 'm.scores': [90, 85], 'm[7]': True,
            's.value': 42, 'err.error': b"bad", 'shape[1]': 4, 'p.x * 10 + p.y': 12, 'len(big) > 200': True,
            '-p.x': -1, 'not s.is_some()': False, '"ab" * 2': b"abab", '2 ** 10': 1024, '7 / 2': 3,
            's.unwrap_or(0) + 1': 43, 'none.is_none()': True, 'None': None, '[1, 2][1]': 2,
            'f"{p.x}:{p.y}"': b"1:2", '"x" if p.x > 0 else "y"': b"x", 'range(0, 3)': [0, 1, 2],
            'p.x as float': 1.0, 'Some(1).value': 1, 'text.split("1")': [b"line", b"\nline2"],
            'type_of(m)': b"map", '0 <= p.x < 5': True, 'pts.len()': 2, 'true': True,
        }
        for expr, expected in cases.items():
            assert s.evaluate(expr) == expected, expr
        assert s.preview(s.evaluate('Shape.Empty')) == "Shape.Empty"
        for bad, why in [('add(1)', "only variables"), ('pts.push(1)', "only variables"), ('x = 1', "only variables"),
                         ('[x for x in big]', "only variables"), ('q', "No variable 'q' here"),
                         ('pts[9]', "index out of bounds"), ('p.z', "no field 'z' on Point"),
                         ('Shape.Nope', "has no variant 'Nope'"), ('1 +', "Unexpected token"),
                         ('[0] * 100000000', "too large"), ('print(1)', "only variables")]:
            with pytest.raises(tildebug.DebugError, match=why):
                s.evaluate(bad)
        assert dict(s.locals())['p'].f == [1, 2]       # nothing changed the program's values

    def test_values(self):
        s = tildebug.DebugSession.load(write("values.til", NESTED))
        s.add_breakpoint(25)
        s.cont()
        values = dict(s.locals())
        kinds = {name: tildebug.value_kind(s.vm, v) for name, v in values.items()}
        assert kinds == {'p': 'struct', 'pts': 'array', 'm': 'map', 's': 'option', 'none': 'none', 'ok': 'result',
                         'err': 'result', 'shape': 'variant', 'add': 'closure', 'big': 'array', 'text': 'string'}
        assert s.describe(values['m'])['named'] == 4 and s.describe(values['big'])['indexed'] == 250
        assert [c[:1] + c[2:] for c in s.children(values['m'])] == [('"name"', '["name"]'), ('"scores"', '["scores"]'),
                                                                   ('"nested"', '["nested"]'), ('7', '[7]')]
        assert len(s.preview(values['big'], 50)) == 53 and s.preview(values['big'], 50).endswith("...")
        assert s.children(values['big'], 248, 10) == [('[248]', 248, '[248]'), ('[249]', 249, '[249]')]

    def test_kill_from_another_thread(self):
        s = tildebug.DebugSession.load(write("kill.til", FOREVER))
        threading.Timer(0.2, s.kill).start()
        with pytest.raises(tildebug.Killed):
            s.cont()
        with pytest.raises(tildebug.Killed):
            s.step_in()

    def test_interrupt_from_another_thread(self):
        s = tildebug.DebugSession.load(write("intr.til", FOREVER))
        threading.Timer(0.2, s.interrupt).start()
        st = s.cont()
        assert st.reason == 'pause' and st.interrupted and s.lookup('n') > 0

    def test_load_source_text(self):
        s = tildebug.DebugSession.load(os.path.join(TMP, "unsaved.til"), source=FACT, stdin="")
        s.add_breakpoint(4)
        st = s.cont()
        assert st.line == 4 and s.lookup('n') == 4 and s.source[3].strip() == "return n * fact(n - 1)"
