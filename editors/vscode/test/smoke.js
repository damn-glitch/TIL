#!/usr/bin/env node
// Smoke test of the extension in plain node (tests/test_vscode_ext.py runs it):
//   1. activate the bundled extension (dist/extension.js) against test/fake-vscode.js
//      and report what it registered (commands, debug type, views);
//   2. the server-launch logic: the command line computed from the settings,
//      PATH and the extension's location, in several situations;
//   3. protocol: start the computed `... lsp` and `... dap` and talk to them
//      (LSP initialize / diagnostics / til.translate / shutdown; DAP initialize /
//      launch / breakpoint / step back / til/timeline / til/goto / til/stdin).
//
//   node test/smoke.js REPO [--python PY] [--bundle FILE]
// Prints one JSON object with everything it saw.
'use strict';
const Module = require('module');
const cp = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const argv = process.argv.slice(2);
const repo = path.resolve(argv[0] || path.join(__dirname, '..', '..', '..'));
const opt = (name, def) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : def; };
const python = opt('--python', process.platform === 'win32' ? 'python' : 'python3');
const bundle = path.resolve(opt('--bundle', path.join(__dirname, '..', 'dist', 'extension.js')));
const extDir = path.join(repo, 'editors', 'vscode');
const tilPy = path.join(repo, 'src', 'til.py');

// `require('vscode')` -> the fake
const fakePath = path.join(__dirname, 'fake-vscode.js');
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
    if (request === 'vscode') return fakePath;
    return origResolve.call(this, request, ...rest);
};
const fake = require(fakePath).__fake;
const ext = require(bundle);

const result = { bundle, activation: {}, client: {}, resolve: {}, quoting: {}, lsp: {}, dap: {} };
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'til-vscode-smoke-'));

function setSettings(s) {
    fake.state.settings = { 'til.lsp.enabled': false, ...s };
}

// ── 1. activation ──────────────────────────────────────────────────────────
function activation() {
    setSettings({ 'til.path': tilPy, 'til.python': python });
    const context = {
        subscriptions: [], extensionPath: extDir, extensionUri: { fsPath: extDir },
        workspaceState: { get() { return undefined; }, update() { return Promise.resolve(); } },
        globalState: { get() { return undefined; }, update() { return Promise.resolve(); } },
    };
    const api = ext.activate(context);
    const out = result.activation;
    out.commands = [...fake.calls.commands.keys()].sort();
    out.debugTypes = [...fake.calls.debugFactories.keys()];
    out.debugProviders = fake.calls.debugProviders.map((p) => p.type);
    out.trackers = [...fake.calls.trackers.keys()];
    out.webviewViews = [...fake.calls.webviewViews.keys()];
    out.subscriptions = context.subscriptions.length;
    out.api = Object.keys(api || {}).sort();
    // the debug adapter the factory starts
    const factory = fake.calls.debugFactories.get('til');
    const session = { configuration: { cwd: tmp }, workspaceFolder: undefined, name: 'test' };
    const desc = factory.createDebugAdapterDescriptor(session);
    out.adapter = { command: desc.command, args: desc.args, cwd: desc.options && desc.options.cwd };
    // F5 without launch.json in a .til editor
    const provider = fake.calls.debugProviders.find((p) => p.provider.resolveDebugConfiguration).provider;
    fake.state.activeTextEditor = { document: { languageId: 'til', uri: { fsPath: path.join(tmp, 'a.til'), scheme: 'file' } } };
    out.f5 = provider.resolveDebugConfiguration(undefined, {});
    setSettings({ 'til.path': tilPy, 'til.python': python, 'til.locale': 'kk' });
    out.f5kk = provider.resolveDebugConfiguration(undefined, { type: 'til', request: 'launch', name: 'x', program: 'a.til' });
    fake.state.activeTextEditor = undefined;
    const dynamic = fake.calls.debugProviders.find((p) => p.provider.provideDebugConfigurations);
    out.dynamicProvider = !!dynamic;
    ext.deactivate();
}

// ── 1b. the real language client (bundled vscode-languageclient) starts til lsp ──
async function languageClient() {
    for (const m of Object.values(fake.calls)) if (m.clear) m.clear();
    fake.calls.debugProviders.length = 0;
    fake.calls.statusItems.length = 0;
    setSettings({ 'til.path': tilPy, 'til.python': python, 'til.lsp.enabled': true });
    const context = {
        subscriptions: [], extensionPath: extDir, extensionUri: { fsPath: extDir },
        workspaceState: { get() { return undefined; }, update() { return Promise.resolve(); } },
        globalState: { get() { return undefined; }, update() { return Promise.resolve(); } },
    };
    ext.activate(context);
    const log = () => (fake.calls.outputs.get('TIL') || { text: '' }).text;
    const end = Date.now() + 30000;
    while (Date.now() < end && !/language server running|did not start/.test(log())) {
        await new Promise((r) => setTimeout(r, 100));
    }
    const out = result.client = {};
    out.log = log().split('\n').filter((l) => /language server/.test(l));
    out.status = fake.calls.statusItems.map((i) => i.text);
    const t0 = Date.now();
    await ext.deactivate();
    out.stopMs = Date.now() - t0;
}

// ── 2. the command line ────────────────────────────────────────────────────
function resolveScenarios() {
    const r = result.resolve;
    const savedPath = process.env.PATH;
    const emptyBin = path.join(tmp, 'empty-bin');
    fs.mkdirSync(emptyBin);
    const pyDir = path.dirname(cp.execFileSync(python, ['-c', 'import sys; print(sys.executable)'], { encoding: 'utf8' }).trim());
    try {
        // a. til.path names til.py; til.python chosen
        setSettings({ 'til.path': tilPy, 'til.python': python });
        r.settingPy = ext.lspServerCommand(extDir);
        // b. `til` on PATH (a launcher script)
        const bin = path.join(tmp, 'bin');
        fs.mkdirSync(bin);
        const launcherFile = path.join(bin, process.platform === 'win32' ? 'til.cmd' : 'til');
        fs.writeFileSync(launcherFile, process.platform === 'win32' ? `@"${python}" "${tilPy}" %*\r\n`
            : `#!/bin/sh\nexec "${python}" "${tilPy}" "$@"\n`);
        fs.chmodSync(launcherFile, 0o755);
        process.env.PATH = [bin, pyDir, emptyBin].join(path.delimiter);
        setSettings({});
        r.onPath = ext.lspServerCommand(extDir);
        // c. no til on PATH: the checkout the extension runs from
        process.env.PATH = [pyDir, emptyBin].join(path.delimiter);
        r.fromExtension = ext.lspServerCommand(extDir);
        // d. til.path names the checkout folder
        setSettings({ 'til.path': repo, 'til.python': python });
        r.settingFolder = ext.lspServerCommand(path.join(tmp, 'elsewhere'));
        // e. til.path that does not exist
        setSettings({ 'til.path': path.join(tmp, 'nope', 'til') });
        r.settingMissing = ext.lspServerCommand(extDir);
        // f. a TIL checkout as a workspace folder: used when trusted, not in Restricted Mode
        setSettings({ 'til.python': python });
        fake.state.workspaceFolders = [{ uri: { fsPath: repo, scheme: 'file' }, name: 'TIL', index: 0 }];
        const home = os.homedir;
        os.homedir = () => path.join(tmp, 'home');                   // no ~/.til here
        fake.state.isTrusted = true;
        r.workspaceTrusted = ext.lspServerCommand(path.join(tmp, 'elsewhere'));
        fake.state.isTrusted = false;
        r.workspaceUntrusted = ext.lspServerCommand(path.join(tmp, 'elsewhere'));
        fake.state.isTrusted = true;
        fake.state.workspaceFolders = [];
        // g. the installer's ~/.til/bin/til.py (not on PATH)
        fs.mkdirSync(path.join(tmp, 'home', '.til', 'bin'), { recursive: true });
        fs.writeFileSync(path.join(tmp, 'home', '.til', 'bin', 'til.py'), '# placeholder\n');
        r.home = ext.lspServerCommand(path.join(tmp, 'elsewhere'));
        os.homedir = home;
        // h. the deprecated til.compilerPath still works
        setSettings({ 'til.compilerPath': tilPy, 'til.python': python });
        r.legacy = ext.lspServerCommand(path.join(tmp, 'elsewhere'));
        // i. a til.python that does not exist: one clear problem, nothing is started
        setSettings({ 'til.path': tilPy, 'til.python': 'no-such-python-3' });
        const ctx = new ext.Context({ subscriptions: [], extensionPath: extDir });
        r.missingPython = { til: ctx.til(), problem: ctx.problem() };
    } finally {
        process.env.PATH = savedPath;
    }
    // i. Windows, with an injected file system (installer layout: til.bat next to til.py)
    const L = ext.launcher;
    const files = new Set(['C:\\Users\\a\\.til\\bin\\til.bat', 'C:\\Users\\a\\.til\\bin\\til.py',
        'C:\\Python312\\python.exe', 'D:\\tools\\til.cmd', 'C:\\Windows\\py.exe']);
    const winFs = { isFile: (p) => files.has(p), isDir: () => false, isExecutable: () => true, realpath: (p) => p };
    const winEnv = { Path: 'C:\\Python312;C:\\Users\\a\\.til\\bin;C:\\Windows', PATHEXT: '.COM;.EXE;.BAT;.CMD', ComSpec: 'C:\\Windows\\system32\\cmd.exe' };
    r.windowsBat = L.resolveTil({ platform: 'win32', env: winEnv, fs: winFs, home: 'C:\\Users\\a' });
    r.windowsCmdOnly = L.resolveTil({ platform: 'win32', env: { ...winEnv, Path: 'C:\\Python312;D:\\tools' }, fs: winFs });
    r.windowsPySetting = L.resolveTil({ platform: 'win32', env: winEnv, fs: winFs, tilPath: 'C:\\Users\\a\\.til\\bin\\til.py', python: 'py -3' });
    // quoting of terminal command lines
    const argvQ = ['/usr/bin/python3', '/home/a b/til.py', 'run', "it's.til"];
    result.quoting = {
        posix: L.commandLine(argvQ, L.shellKind('/bin/zsh', 'linux')),
        pwsh: L.commandLine(['C:\\Program Files\\Python\\python.exe', 'C:\\til\\til.py', 'run', "C:\\a b\\it's.til"],
            L.shellKind('C:\\Program Files\\PowerShell\\7\\pwsh.exe', 'win32')),
        cmd: L.commandLine(['C:\\Python\\python.exe', 'C:\\a b\\til.py', 'run', 'x.til'], L.shellKind('C:\\Windows\\System32\\cmd.exe', 'win32')),
        defaultWin: L.shellKind(undefined, 'win32'),
    };
}

// ── 3. protocol ────────────────────────────────────────────────────────────
class Peer {
    constructor(command, args, cwd) {
        this.proc = cp.spawn(command, args, { cwd, env: { ...process.env, PYTHONIOENCODING: 'utf-8' }, stdio: ['pipe', 'pipe', 'pipe'] });
        this.buf = Buffer.alloc(0);
        this.messages = [];
        this.waiters = [];
        this.stderr = '';
        this.proc.stdout.on('data', (d) => this.data(d));
        this.proc.stderr.on('data', (d) => { this.stderr += d; });
        this.exited = new Promise((res) => this.proc.on('exit', (code) => res(code)));
    }

    data(d) {
        this.buf = Buffer.concat([this.buf, d]);
        for (;;) {
            const end = this.buf.indexOf('\r\n\r\n');
            if (end < 0) return;
            const m = /Content-Length: *(\d+)/i.exec(this.buf.slice(0, end).toString('ascii'));
            const len = m ? parseInt(m[1], 10) : 0;
            if (this.buf.length < end + 4 + len) return;
            const msg = JSON.parse(this.buf.slice(end + 4, end + 4 + len).toString('utf8'));
            this.buf = this.buf.slice(end + 4 + len);
            this.messages.push(msg);
            this.waiters = this.waiters.filter((w) => !w(msg));
        }
    }

    send(msg) {
        const body = Buffer.from(JSON.stringify(msg), 'utf8');
        this.proc.stdin.write(`Content-Length: ${body.length}\r\n\r\n`);
        this.proc.stdin.write(body);
    }

    /** The first message (already received or coming) that satisfies pred. */
    wait(pred, what, ms = 30000) {
        const seen = this.messages.find(pred);
        if (seen) return Promise.resolve(seen);
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error(`timeout waiting for ${what}; stderr: ${this.stderr.slice(-500)}`)), ms);
            this.waiters.push((msg) => {
                if (!pred(msg)) return false;
                clearTimeout(timer);
                resolve(msg);
                return true;
            });
        });
    }
}

async function lsp(cmd) {
    const out = result.lsp;
    const peer = new Peer(cmd.command, cmd.args, tmp);
    const file = path.join(tmp, 'prog.til');
    const text = 'main()\n    let x = 1\n    print(y)\n';
    fs.writeFileSync(file, text);
    const uri = 'file://' + (process.platform === 'win32' ? '/' + file.replace(/\\/g, '/') : file);
    peer.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: {
        processId: process.pid, rootUri: null, capabilities: {}, initializationOptions: { locale: 'ru' } } });
    const init = await peer.wait((m) => m.id === 1, 'initialize');
    out.serverInfo = init.result && init.result.serverInfo;
    out.positionEncoding = init.result && init.result.capabilities.positionEncoding;
    out.codeLensProvider = !!(init.result && init.result.capabilities.codeLensProvider);
    peer.send({ jsonrpc: '2.0', method: 'initialized', params: {} });
    peer.send({ jsonrpc: '2.0', method: 'textDocument/didOpen', params: { textDocument: { uri, languageId: 'til', version: 1, text } } });
    const diags = await peer.wait((m) => m.method === 'textDocument/publishDiagnostics' && m.params.uri === uri, 'diagnostics');
    out.diagnostics = diags.params.diagnostics.map((d) => ({ code: d.code, line: d.range.start.line, message: d.message }));
    peer.send({ jsonrpc: '2.0', id: 2, method: 'textDocument/codeLens', params: { textDocument: { uri } } });
    const lenses = await peer.wait((m) => m.id === 2, 'codeLens');
    out.lensCommands = (lenses.result || []).map((l) => l.command && l.command.command);
    out.lensArguments = (lenses.result || []).map((l) => l.command && l.command.arguments);
    peer.send({ jsonrpc: '2.0', id: 3, method: 'workspace/executeCommand', params: { command: 'til.translate', arguments: [{ uri, to: 'ru' }] } });
    const tr = await peer.wait((m) => m.id === 3, 'til.translate');
    out.translated = tr.result && tr.result.text;
    peer.send({ jsonrpc: '2.0', id: 4, method: 'workspace/executeCommand', params: { command: 'til.run', arguments: [{ uri, function: 'main' }] } });
    out.runCommandResult = (await peer.wait((m) => m.id === 4, 'til.run')).result;
    // til.locale changed in VS Code: the extension sends workspace/didChangeConfiguration
    peer.send({ jsonrpc: '2.0', method: 'workspace/didChangeConfiguration', params: { settings: { til: { locale: 'kk' } } } });
    peer.send({ jsonrpc: '2.0', id: 5, method: 'textDocument/hover', params: { textDocument: { uri }, position: { line: 2, character: 5 } } });
    const hover = await peer.wait((m) => m.id === 5, 'hover');
    out.hoverKk = hover.result && JSON.stringify(hover.result.contents).slice(0, 400);
    peer.send({ jsonrpc: '2.0', id: 6, method: 'shutdown' });
    await peer.wait((m) => m.id === 6, 'shutdown');
    peer.send({ jsonrpc: '2.0', method: 'exit' });
    out.exitCode = await peer.exited;
}

async function dap(cmd) {
    const out = result.dap;
    const peer = new Peer(cmd.command, cmd.args, tmp);
    let seq = 0;
    const request = (command, args) => {
        const s = ++seq;
        peer.send({ seq: s, type: 'request', command, arguments: args });
        return peer.wait((m) => m.type === 'response' && m.request_seq === s, command);
    };
    const event = (name, after = 0) => peer.wait((m) => m.type === 'event' && m.event === name && m.seq > after, name);
    const file = path.join(tmp, 'debug.til');
    fs.writeFileSync(file, 'main()\n    var total = 0\n    for i in 0..3\n        total += i\n    let name = input()\n    print(f"{name}: {total}")\n');
    const init = await request('initialize', { clientID: 'vscode', adapterID: 'til', linesStartAt1: true, columnsStartAt1: true, locale: 'en' });
    out.supportsStepBack = !!(init.body && init.body.supportsStepBack);
    out.supportsConfigurationDone = !!(init.body && init.body.supportsConfigurationDoneRequest);
    const launch = await request('launch', { program: file, cwd: tmp, stopOnEntry: false, stopAtEnd: true });
    out.launch = launch.success;
    await event('initialized');
    const bp = await request('setBreakpoints', { source: { path: file }, breakpoints: [{ line: 4 }] });
    out.breakpoints = (bp.body.breakpoints || []).map((b) => b.verified);
    await request('configurationDone', {});
    const stop1 = await event('stopped');
    out.firstStop = stop1.body.reason;
    await request('next', {});
    const stop2 = await event('stopped', stop1.seq);
    const back = await request('stepBack', { threadId: 1 });
    out.stepBack = back.success;
    const stop3 = await event('stopped', stop2.seq);
    const tl = await request('til/timeline', {});
    out.timeline = tl.body;
    const gotoRes = await request('til/goto', { index: 0 });
    out.goto = gotoRes.success;
    const stop4 = await event('stopped', stop3.seq);
    out.gotoReason = stop4.body.reason;
    // run on: input() asks the client (til/inputRequested), the client sends til/stdin
    await request('setBreakpoints', { source: { path: file }, breakpoints: [] });
    await request('continue', { threadId: 1 });
    await event('til/inputRequested');
    out.inputRequested = true;
    const stdin = await request('til/stdin', { text: 'Aigerim\n' });
    out.stdin = stdin.success;
    await peer.wait((m) => m.type === 'event' && m.event === 'output' && /Aigerim: 3/.test(m.body.output || ''), 'program output');
    out.output = 'Aigerim: 3';
    const end = await event('stopped', stop4.seq);
    out.endReason = end.body.reason;
    await request('disconnect', { terminateDebuggee: true });
    out.exitCode = await peer.exited;
}

async function main() {
    try {
        activation();
        await languageClient();
        resolveScenarios();
        const cmd = ext.lspServerCommand(extDir);
        await lsp((() => { setSettings({ 'til.path': tilPy, 'til.python': python }); return ext.lspServerCommand(extDir); })());
        setSettings({ 'til.path': tilPy, 'til.python': python });
        await dap(ext.dapCommand(extDir));
        result.ok = !!cmd;
    } catch (e) {
        result.error = String(e && e.stack || e);
    } finally {
        fs.rmSync(tmp, { recursive: true, force: true });
    }
    process.stdout.write(JSON.stringify(result) + '\n');
    process.exit(0);
}

main();
