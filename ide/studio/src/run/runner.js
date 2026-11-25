// Runs: POST /api/run (or the browser runtime), output streamed as `run`
// events {runId, kind, ...}: stdout/stderr {text}; status {state: starting |
// compiling | running | snapshot, backend (the one really used), pid?, path?,
// steps?}; input {}; screen {w, h, rgb}; exit {code, reason: exited | stopped
// | failed | paused, seconds}. The events carry no text for people: every
// message is the UI's own, in the UI language. Stdin from the output panel,
// keys from the screen, Stop.
import { t, getLocale } from '../core/i18n.js';
import { fmtMs } from '../core/dom.js';
import { basename } from '../core/paths.js';
import { toast } from '../ui/widgets.js';

/* "a 'b c' d" -> ['a', 'b c', 'd'] */
export function splitArgs(text) {
    const out = [];
    const re = /"((?:\\.|[^"\\])*)"|'([^']*)'|(\S+)/g;
    let m;
    while ((m = re.exec(String(text || '')))) out.push(m[1] !== undefined ? m[1].replace(/\\(.)/g, '$1') : m[2] !== undefined ? m[2] : m[3]);
    return out;
}

/* KeyboardEvent -> TIL key code (UP=1 DOWN=2 LEFT=3 RIGHT=4, Enter 13, Esc 27, Backspace 8, else the character) */
export function keyCodeOf(e) {
    const named = { ArrowUp: 1, ArrowDown: 2, ArrowLeft: 3, ArrowRight: 4, Enter: 13, Escape: 27, Backspace: 8, Delete: 127, ' ': 32 };
    if (named[e.key] !== undefined) return named[e.key];
    if (e.key && e.key.length >= 1) {
        const cp = e.key.codePointAt(0);
        if (String.fromCodePoint(cp) === e.key && cp >= 32) return cp;
    }
    return 0;
}

export class Runner {
    constructor(app) {
        this.app = app;
        this.current = null;
        app.backend.on('run', (ev) => this.onEvent(ev));
        app.backend.on('screen-rgba', (ev) => {
            if (this.current && String(this.current.id) === String(ev.runId)) app.screen.frameRGBA(ev.w, ev.h, ev.rgba);
        });
    }

    get running() { return Boolean(this.current && !this.current.done); }

    get currentPath() { return this.current ? this.current.path : null; }

    async run(doc, opts = {}) {
        const app = this.app;
        if (this.running) await this.stop({ quiet: true });
        if (doc.kind === 'file' && doc.dirty) {
            const ok = await app.saveDoc(doc);
            if (!ok) return null;
        }
        const cfg = app.views.debug.config();
        const backend = opts.backend || app.effectiveBackend();
        const payload = { backend, args: splitArgs(opts.args !== undefined ? opts.args : cfg.args), lang: getLocale() };
        const stamp = () => new Date().toLocaleTimeString(getLocale(), { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        if (doc.kind === 'file') payload.path = doc.path; else payload.source = doc.current().doc.toString();
        const stdin = opts.stdin !== undefined ? opts.stdin : cfg.stdin;
        if (stdin) payload.stdin = stdin.endsWith('\n') ? stdin : stdin + '\n';
        const keys = opts.keys !== undefined ? opts.keys : cfg.keys;
        if (keys) payload.keys = keys;
        if (opts.maxSteps) payload.maxSteps = opts.maxSteps;
        const out = app.output;
        out.clear();
        const time = stamp();
        out.header(doc.name, `${t(`backend.${backend}`)} · ${time}`);
        app.layout.showPanel('output');
        app.screen.reset();
        const run = {
            id: null, path: doc.kind === 'file' ? doc.path : null, doc, backend, actual: backend, time,
            scripted: { stdin: Boolean(payload.stdin), keys: Boolean(payload.keys) },
            started: performance.now(), done: false, early: [],
        };
        this.current = run;
        app.statusbar.setRun({ label: t('run.starting'), kind: 'run' });
        this.updateButtons();
        try {
            run.id = await app.backend.run(payload);
        } catch (e) {
            run.done = true;
            this.current = run;
            if (e && e.notAvailable) { app.showDesktopOnly(e.feature); out.footer({ ok: false, text: t('run.notAvailable') }); }
            else out.footer({ ok: false, text: t('run.cannotStart', { error: e && e.message ? e.message : String(e) }) });
            app.statusbar.setRun(null);
            this.updateButtons();
            return null;
        }
        if (run.id === undefined || run.id === null) {
            run.done = true;
            out.footer({ ok: false, text: t('run.cannotStart', { error: 'no runId' }) });
            app.statusbar.setRun(null);
            this.updateButtons();
            return null;
        }
        app.statusbar.setRun({ label: t('run.running', { name: doc.name }), kind: 'run' });
        // with an input script the program reads the script, not the panel
        if (run.scripted.stdin) { out.detachInput(); out.note(t('run.stdinFromScript')); } else out.attachInput((text) => this.sendStdin(text), () => this.sendEof());
        for (const ev of run.early.splice(0)) this.onEvent(ev);
        return run;
    }

    onEvent(ev) {
        if (!ev || ev.runId === undefined) return;
        const run = this.current;
        if (!run) return;
        if (run.id === null && !run.done) { run.early.push(ev); return; }
        if (String(ev.runId) !== String(run.id)) return;
        const out = this.app.output;
        switch (ev.kind) {
        case 'stdout': out.write(ev.text || '', 1); break;
        case 'stderr': out.write(ev.text || '', 2); break;
        case 'status': this.onStatus(run, ev); break;
        case 'input':
            this.app.statusbar.setRun({ label: t('run.waitingInput'), kind: 'wait' });
            out.waitForInput();
            break;
        case 'screen': this.app.screen.frameFromRun(ev); break;
        case 'exit': this.finish(ev); break;
        default: break;
        }
    }

    onStatus(run, ev) {
        const app = this.app;
        if (ev.backend && ev.backend !== run.actual) {
            // e.g. a program that uses the screen runs on the VM whatever was asked
            run.actual = ev.backend;
            app.output.setHeaderDetail(`${t(`backend.${ev.backend}`)} · ${run.time}`);
        }
        const name = run.doc.name;
        const web = app.web;        // the page compiles with tilc and runs on the WebAssembly VM
        const label = {
            starting: t('run.state.starting', { name }),
            compiling: web ? t('web.compiling') : t('run.state.compiling', { name }),
            running: web ? t('web.running') : t('run.running', { name }),
            snapshot: t('run.snapshotSaved', { path: ev.path || '' }),
        }[ev.state];
        if (ev.state === 'snapshot') { app.statusbar.flash(label); app.views.explorer.refresh(); return; }
        if (label) app.statusbar.setRun({ label, kind: 'run' });
    }

    finish(ev) {
        const run = this.current;
        if (!run || run.done) return;
        run.done = true;
        const seconds = typeof ev.seconds === 'number' && Number.isFinite(ev.seconds) ? ev.seconds : null;
        const time = fmtMs(seconds !== null ? seconds * 1000 : performance.now() - run.started);
        const out = this.app.output;
        out.flush();
        out.detachInput();
        const code = ev.code;
        const reason = ev.reason || (run.stopped ? 'stopped' : code === 0 || code === undefined || code === null ? 'exited' : 'exited');
        if (reason === 'stopped' || run.stopped) out.footer({ ok: false, kind: 'stopped', text: t('run.stopped', { time }) });
        else if (reason === 'paused') out.footer({ ok: true, kind: 'paused', text: t('run.paused', { time }) });
        else if (reason === 'failed') out.footer({ ok: false, text: t('run.failed', { time }) });
        else if (code === 0 || code === undefined || code === null) out.footer({ ok: true, text: t('run.finished', { time }) });
        else out.footer({ ok: false, text: t('run.exited', { code, time }) });
        this.app.statusbar.setRun(null);
        this.app.screen.ended();
        this.updateButtons();
    }

    async stop({ quiet = false } = {}) {
        const run = this.current;
        if (!run || run.done) return;
        run.stopped = true;
        if (run.id === null) { run.done = true; this.updateButtons(); return; }
        try {
            await this.app.backend.runStop(run.id);
        } catch (e) {
            if (!quiet) toast(t('run.cannotStop'), { kind: 'error', detail: e.message });
        }
        // the exit event follows; if the server is gone, end here
        setTimeout(() => { if (this.current === run && !run.done) this.finish({ code: -1 }); }, 2500);
    }

    async sendStdin(text) {
        const run = this.current;
        if (!run || run.done || run.id === null) return;
        this.app.statusbar.setRun({ label: t('run.running', { name: run.doc.name }), kind: 'run' });
        try {
            await this.app.backend.runStdin(run.id, text);
        } catch (e) {
            // 409: the run has ended or reads an input script (the server says which)
            if (e && e.status === 409) this.app.statusbar.flash(e.message);
            else toast(t('run.stdinFailed'), { kind: 'error', detail: e.message });
        }
    }

    async sendEof() {
        const run = this.current;
        if (!run || run.done || run.id === null) return;
        try {
            await this.app.backend.runEof(run.id);
        } catch (e) { /* the program has ended or its input is scripted */ }
    }

    sendKey(code) {
        const run = this.current;
        if (!code || !run || run.done || run.id === null) return false;
        if (run.scripted.keys) { this.keyNote(run, t('run.keysScripted')); return true; }
        // 409: a key script, a program not on the VM, or the run has ended (the server says which)
        this.app.backend.runKey(run.id, code).catch((e) => { if (e && e.status === 409) this.keyNote(run, e.message); });
        return true;
    }

    keyNote(run, text) {
        if (run.keyNoted) return;
        run.keyNoted = true;
        this.app.statusbar.flash(text);
    }

    forwardKey(e) {
        if (!this.running || e.ctrlKey || e.metaKey || e.altKey || e.key === 'Tab') return;
        const code = keyCodeOf(e);
        if (code && this.sendKey(code)) e.preventDefault();
    }

    async snapshot() {
        const run = this.current;
        // the backend the program really runs on (status events), not the one asked for
        if (!run || run.done || run.actual !== 'vm') { toast(t('run.snapshotNeedsVm'), { kind: 'info' }); return; }
        try {
            const r = await this.app.backend.runSnapshot(run.id);
            toast(r && r.steps !== undefined ? t('run.snapshotSavedSteps', { path: r.path || '', steps: r.steps }) : t('run.snapshotSaved', { path: (r && r.path) || '' }), { kind: 'success' });
            this.app.views.explorer.refresh();
        } catch (e) {
            if (e && e.notAvailable) { this.app.showDesktopOnly(e.feature); return; }
            toast(t('run.snapshotFailed'), { kind: 'error', detail: e.message });
        }
    }

    updateButtons() {
        const L = this.app.layout;
        const busy = this.running || (this.app.debugger && this.app.debugger.active);
        L.stopBtn.hidden = !busy;
        L.runBtn.hidden = false;
    }
}

export { basename };
