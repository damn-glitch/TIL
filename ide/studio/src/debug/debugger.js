// The debug session: DAP over the studio server (launch, breakpoints,
// stepping, time travel with stepBack / reverseContinue, the timeline
// scrubber through til/timeline + til/goto, variables, watch, evaluate,
// stdin through til/stdin, the screen through til/screen).
import { h } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, getLocale } from '../core/i18n.js';
import { basename, dirname } from '../core/paths.js';
import { DapClient } from '../dap/dap.js';
import { toast, tip } from '../ui/widgets.js';
import { setDebugLine } from '../editor/gutters.js';
import { splitArgs } from '../run/runner.js';

export class Debugger {
    constructor(app) {
        this.app = app;
        this.client = null;
        this.active = false;
        this.stopped = false;
        this.frames = [];
        this.frameIndex = 0;
        this.scopes = [];
        this.timeline = null;
        this.location = null;           // {path, line}
        this.watches = Array.isArray(app.project.watch) ? app.project.watch : [];
        this.watchValues = new Map();
        this.target = null;             // {docPath, program, source}
        this.bar = null;
        this.gotoTimer = 0;
        this.pendingGoto = null;
        this.busy = false;
    }

    /* ── start / stop ── */
    async start(doc, opts = {}) {
        const app = this.app;
        if (this.active) await this.stop();
        if (app.runner.running) await app.runner.stop({ quiet: true });
        if (doc.kind === 'file' && doc.dirty) {
            const ok = await app.saveDoc(doc);
            if (!ok) return;
        }
        const cfg = app.views.debug.config();
        // the program the adapter runs: the file, or its text through `source`
        // (documents that are not files yet, and "Debug with these inputs")
        // a file of the project is debugged by its path; anything else (a new
        // buffer, a read-only TIL example "@examples/...") by its text
        const inProject = doc.kind === 'file' && !doc.path.startsWith('@');
        const target = { docPath: doc.path, program: app.absPath(inProject ? doc.path : doc.name), source: undefined };
        if (!inProject) target.source = doc.current().doc.toString();
        if (opts.call) {
            target.source = harnessText(doc.current().doc.toString(), opts.call);
            this.breakAtFunction(doc, opts.call.function);
        } else if (opts.function && opts.function !== 'main') {
            this.breakAtFunction(doc, opts.function);
        }
        this.target = target;
        this.lastOpts = opts;
        this.client = new DapClient(app.backend);
        const client = this.client;
        this.wire(client);
        this.active = true;
        this.stopped = false;
        this.programPath = doc.path;
        this.timeline = null;
        app.output.clear();
        app.output.header(opts.call ? t('debug.headerCall', { fn: opts.call.function }) : t('debug.header', { name: doc.name }), t('backend.vm') + ' · ' + t('debug.timeTravel'));
        app.layout.showSide('debug', { focus: false });
        app.layout.showPanel('output');
        app.screen.reset();
        app.statusbar.setDebug(t('debug.starting'));
        this.showBar();
        app.runner.updateButtons();
        app.views.debug.render();
        try {
            await client.start();
            const caps = await client.request('initialize', {
                clientID: 'til-studio', clientName: 'TIL Studio', adapterID: 'til', locale: getLocale(),
                linesStartAt1: true, columnsStartAt1: true, pathFormat: 'path', supportsVariableType: true,
                supportsVariablePaging: false, supportsRunInTerminalRequest: false, supportsProgressReporting: false,
            });
            client.capabilities = caps || {};
            const initialized = new Promise((resolve) => {
                const off = client.on('initialized', () => { off(); resolve(true); });
                setTimeout(() => resolve(false), 4000);
            });
            const launchArgs = {
                program: target.program, cwd: app.absPath(dirname(inProject ? doc.path : '')) || app.info.root,
                args: splitArgs(cfg.args), stopOnEntry: Boolean(opts.stopOnEntry !== undefined ? opts.stopOnEntry : cfg.stopOnEntry),
                noDebug: false, lang: getLocale(),
            };
            if (target.source !== undefined) launchArgs.source = target.source;
            if (cfg.stdin) launchArgs.stdin = cfg.stdin.endsWith('\n') ? cfg.stdin : cfg.stdin + '\n';
            if (cfg.keys) launchArgs.keys = cfg.keys;
            const launchP = client.request('launch', launchArgs);
            launchP.catch(() => {});
            await initialized;
            await this.sendAllBreakpoints();
            try { await client.request('setExceptionBreakpoints', { filters: ['runtime-error'] }); } catch (e) { /* optional */ }
            if (client.capabilities.supportsConfigurationDoneRequest !== false) await client.request('configurationDone', {});
            await launchP;
            if (this.active && !this.stopped) app.statusbar.setDebug(t('debug.running'));
            app.output.attachInput((text) => this.sendStdin(text), () => this.sendStdin('', true));
        } catch (e) {
            if (e && e.notAvailable) app.showDesktopOnly(e.feature);
            else toast(t('debug.cannotStart'), { kind: 'error', detail: e && e.message });
            this.end();
        }
    }

    /* a breakpoint on the first line of a function's body ('f' or 'Type.method') */
    breakAtFunction(doc, fn) {
        if (!fn) return;
        const name = String(fn).split('.').pop().replace(/[^\p{L}\p{N}_]/gu, '');
        const lines = doc.current().doc.toString().split('\n');
        const defRe = new RegExp(`^(\\s*)(?:pub\\s+)?(?:fn\\s+|функция\\s+|funktsiya\\s+)?${name}\\s*\\(`, 'u');
        const def = lines.findIndex((l) => defRe.test(l));
        if (def < 0) return;
        const indent = lines[def].length - lines[def].trimStart().length;
        let body = -1;
        for (let i = def + 1; i < lines.length; i++) {
            const l = lines[i];
            if (!l.trim() || /^\s*#(?!\[)/.test(l)) continue;
            if (l.length - l.trimStart().length > indent) body = i;
            break;
        }
        if (body < 0) return;
        const list = (this.app.breakpoints.get(doc.path) || []).slice();
        if (!list.some((b) => b.line === body + 1)) {
            list.push({ line: body + 1, condition: '', enabled: true });
            this.app.setBreakpoints(doc.path, list);
        }
    }

    wire(client) {
        client.on('stopped', (b) => this.onStopped(b));
        client.on('continued', () => { this.stopped = false; this.clearLine(); this.app.statusbar.setDebug(t('debug.running')); this.app.views.debug.render(); this.updateBar(); });
        client.on('output', (b) => this.onOutput(b));
        client.on('terminated', () => this.end());
        client.on('exited', (b) => { this.exitCode = b.exitCode; });
        client.on('til/inputRequested', () => { this.app.output.waitForInput(); this.app.statusbar.setDebug(t('run.waitingInput')); });
        client.on('til/screen', (b) => this.app.screen.frameFromDap(b));
        client.on('breakpoint', (b) => this.onBreakpointEvent(b));
    }

    async stop() {
        const client = this.client;
        if (!client) { this.end(); return; }
        try {
            if (client.capabilities.supportsTerminateRequest) await client.request('terminate', {}, 4000);
            await client.request('disconnect', { terminateDebuggee: true }, 4000);
        } catch (e) { /* the session may be gone */ }
        this.end();
    }

    end() {
        if (!this.active) return;
        this.active = false;
        this.stopped = false;
        const code = this.exitCode;
        if (this.client) this.client.close();
        this.client = null;
        this.frames = [];
        this.scopes = [];
        this.location = null;
        this.clearLine();
        this.hideBar();
        this.app.output.detachInput();
        this.app.output.footer({ ok: code === undefined || code === 0, text: code !== undefined && code !== 0 ? t('debug.endedCode', { code }) : t('debug.ended') });
        this.exitCode = undefined;
        this.app.statusbar.setDebug(null);
        this.app.views.debug.render();
        this.app.runner.updateButtons();
        this.app.layout.setDot('debug', false);
        this.target = null;
    }

    /* ── breakpoints ── */
    sourceFor(path) {
        if (this.target && path === this.target.docPath) return { path: this.target.program, name: basename(this.target.program) };
        return { path: this.app.absPath(path), name: path };
    }

    async sendBreakpoints(path) {
        if (!this.client) return;
        const list = (this.app.breakpoints.get(path) || []).filter((b) => b.enabled !== false);
        let res;
        try {
            res = await this.client.request('setBreakpoints', {
                source: this.sourceFor(path),
                breakpoints: list.map((b) => {
                    const o = { line: b.line };
                    if (b.condition) o.condition = b.condition;
                    if (b.hitCondition) o.hitCondition = b.hitCondition;
                    return o;
                }),
                lines: list.map((b) => b.line),
                sourceModified: false,
            });
        } catch (e) { return; }
        const answered = res && Array.isArray(res.breakpoints) ? res.breakpoints : null;
        if (answered) {
            const doc = this.app.docs.get(path);
            const all = this.app.breakpoints.get(path) || [];
            let k = 0;
            const merged = all.map((b) => {
                if (b.enabled === false) return b;
                const a = answered[k++];
                return a ? Object.assign({}, b, { verified: a.verified !== false, line: a.line || b.line }) : b;
            });
            if (doc) doc.dispatch({ effects: (await import('../editor/gutters.js')).bpEffect.of({ type: 'verify', list: merged }) });
        }
    }

    async sendAllBreakpoints() {
        for (const path of this.app.breakpoints.keys()) await this.sendBreakpoints(path);
    }

    breakpointsChanged(path) {
        if (this.active) this.sendBreakpoints(path);
    }

    onBreakpointEvent() { /* verification updates: re-sent with the next setBreakpoints */ }

    /* ── stops ── */
    async onStopped(body) {
        this.stopped = true;
        this.stopReason = body.reason || 'step';
        this.stopText = body.text || body.description || '';
        this.frameIndex = 0;
        const client = this.client;
        if (!client) return;
        this.app.layout.setDot('debug', true);
        try {
            const st = await client.request('stackTrace', { threadId: body.threadId || 1, startFrame: 0, levels: 100 });
            this.frames = (st && st.stackFrames) || [];
        } catch (e) { this.frames = []; }
        await this.loadTimeline();
        await this.selectFrame(0, { reveal: true });
        const label = this.stopReason === 'exception' ? t('debug.stoppedException') : t('debug.paused');
        this.app.statusbar.setDebug(label);
        if (this.stopReason === 'exception') {
            if (client.capabilities.supportsExceptionInfoRequest) {
                try {
                    const info = await client.request('exceptionInfo', { threadId: body.threadId || 1 }, 5000);
                    this.stopText = (info && (info.description || (info.details && info.details.message) || info.exceptionId)) || this.stopText;
                } catch (e) { /* keep the stop text */ }
            }
            if (this.stopText) this.app.output.note(this.stopText, 'err');
        }
        this.updateBar();
    }

    async loadTimeline() {
        if (!this.client) return;
        try {
            const tl = await this.client.request('til/timeline', {}, 8000);
            if (tl && typeof tl.total === 'number') this.timeline = { index: tl.index || 0, total: tl.total, complete: tl.complete !== false };
        } catch (e) { this.timeline = null; }
    }

    framePath(frame) {
        const src = frame && frame.source;
        if (!src) return null;
        if (this.target && src.path && samePath(src.path, this.target.program)) return this.target.docPath;
        // Source.path is absolute; Source.name is the root-relative POSIX path
        return this.app.relPath(src.path || '') || (src.name ? String(src.name).replace(/\\/g, '/') : null);
    }

    async selectFrame(i, { reveal = false } = {}) {
        this.frameIndex = i;
        const frame = this.frames[i];
        this.scopes = [];
        if (frame && this.client) {
            try {
                const sc = await this.client.request('scopes', { frameId: frame.id });
                this.scopes = (sc && sc.scopes) || [];
            } catch (e) { this.scopes = []; }
        }
        for (const s of this.scopes) {
            s.children = null;
            if (s.variablesReference && !s.expensive && /local|globa/i.test(s.name || '') && (s === this.scopes[0] || /local/i.test(s.name))) {
                try { s.children = await this.variables(s.variablesReference); } catch (e) { s.children = []; }
            }
        }
        await this.evaluateWatches();
        const path = frame ? this.framePath(frame) : null;
        this.location = path && frame ? { path, line: frame.line } : null;
        this.app.views.debug.render();
        this.showLine();
        if (reveal && this.location) {
            await this.app.openFile(this.location.path, { line: this.location.line, focus: false, flash: false });
            this.showLine();
        }
    }

    showLine() {
        const loc = this.location;
        for (const doc of this.app.docs.all()) {
            const want = loc && doc.path === loc.path ? { line: loc.line, top: this.frameIndex === 0 } : null;
            const cur = doc.current().field(debugLineFieldRef(), false);
            if (JSON.stringify(cur) !== JSON.stringify(want)) doc.dispatch({ effects: setDebugLine.of(want) });
        }
    }

    clearLine() {
        for (const doc of this.app.docs.all()) doc.dispatch({ effects: setDebugLine.of(null) });
    }

    refreshEditors() { if (this.active && this.stopped) this.showLine(); }

    async variables(ref) {
        if (!this.client || !ref) return [];
        const r = await this.client.request('variables', { variablesReference: ref });
        return (r && r.variables) || [];
    }

    async evaluate(expression, context = 'repl') {
        if (!this.client) throw new Error(t('debug.notRunning'));
        const frame = this.frames[this.frameIndex];
        return this.client.request('evaluate', { expression, context, frameId: frame ? frame.id : undefined });
    }

    async evaluateWatches() {
        this.watchValues = new Map();
        if (!this.stopped) return;
        for (const w of this.watches) {
            try {
                const r = await this.evaluate(w, 'watch');
                this.watchValues.set(w, { value: r.result, type: r.type, ref: r.variablesReference });
            } catch (e) {
                this.watchValues.set(w, { error: e.message });
            }
        }
    }

    addWatch(expr) {
        if (!expr || this.watches.includes(expr)) return;
        this.watches.push(expr);
        this.app.saveLayout();
        this.evaluateWatches().then(() => this.app.views.debug.render());
    }

    removeWatch(expr) {
        this.watches = this.watches.filter((w) => w !== expr);
        this.watchValues.delete(expr);
        this.app.saveLayout();
        this.app.views.debug.render();
    }

    /* ── stepping ── */
    async step(command, extra = {}) {
        if (!this.client || this.busy) return;
        if (command !== 'pause' && !this.stopped) return;
        if ((command === 'stepBack' || command === 'reverseContinue') && this.client.capabilities.supportsStepBack === false) {
            toast(t('debug.noStepBack'), { kind: 'info' });
            return;
        }
        this.busy = true;
        try {
            if (command !== 'pause') { this.stopped = false; this.updateBar(); }
            await this.client.request(command, Object.assign({ threadId: 1 }, extra));
        } catch (e) {
            this.stopped = true;
            toast(t('debug.stepFailed'), { kind: 'warning', detail: e.message });
        } finally {
            this.busy = false;
            this.updateBar();
        }
    }

    continue() { return this.step('continue'); }
    next() { return this.step('next'); }
    stepIn() { return this.step('stepIn'); }
    stepOut() { return this.step('stepOut'); }
    stepBack() { return this.step('stepBack'); }
    reverseContinue() { return this.step('reverseContinue'); }
    pause() { return this.step('pause'); }

    async restart() {
        const path = this.programPath;
        const opts = this.lastOpts || {};
        await this.stop();
        const doc = this.app.docs.get(path) || await this.app.docs.open(path);
        return this.start(doc, opts);
    }

    /* the timeline scrubber: jump to a recorded line event (throttled while dragging) */
    goto(index, { final = false } = {}) {
        if (!this.client || !this.timeline) return;
        this.pendingGoto = Math.max(0, Math.min(this.timeline.total - 1, Math.round(index)));
        if (this.gotoTimer && !final) return;
        const fire = async () => {
            this.gotoTimer = 0;
            const target = this.pendingGoto;
            this.pendingGoto = null;
            if (target === null) return;
            this.timeline.index = target;
            try {
                await this.client.request('til/goto', { index: target });
            } catch (e) {
                toast(t('debug.gotoFailed'), { kind: 'warning', detail: e.message });
            }
        };
        if (final) { clearTimeout(this.gotoTimer); fire(); return; }
        this.gotoTimer = setTimeout(fire, 70);
    }

    async sendStdin(text, eof = false) {
        if (!this.client) return;
        try { await this.client.request('til/stdin', eof ? { text, eof: true } : { text }); } catch (e) { toast(t('run.stdinFailed'), { kind: 'error', detail: e.message }); }
        this.app.statusbar.setDebug(t('debug.running'));
    }

    onOutput(b) {
        const cat = b.category || 'console';
        const text = b.output || '';
        if (cat === 'stdout') this.app.output.write(text, 1);
        else if (cat === 'stderr') this.app.output.write(text, 2);
        else if (cat !== 'telemetry') this.app.debugConsole.log(text, cat);
    }

    /* ── the floating toolbar ── */
    showBar() {
        if (this.bar) return;
        const btn = (name, cls, key, keys, run) => {
            const b = h('button', { class: `icon-btn ${cls}`, type: 'button' }, icon(name));
            tip(b, key, keys);
            b.addEventListener('click', run);
            return b;
        };
        this.barButtons = {
            cont: btn('continue', 'db-continue', 'debug.continue', 'F5', () => (this.stopped ? this.continue() : this.pause())),
            over: btn('stepOver', 'db-over', 'debug.stepOver', 'F10', () => this.next()),
            into: btn('stepInto', 'db-into', 'debug.stepInto', 'F11', () => this.stepIn()),
            out: btn('stepOut', 'db-out', 'debug.stepOut', 'Shift+F11', () => this.stepOut()),
            back: btn('stepBack', 'db-back', 'debug.stepBack', 'Shift+F10', () => this.stepBack()),
            rev: btn('reverseContinue', 'db-reverse', 'debug.reverseContinue', 'Shift+F5', () => this.reverseContinue()),
            restart: btn('restart', 'db-restart', 'debug.restart', 'Mod+Shift+F5', () => this.restart()),
            stop: btn('stop', 'db-stop', 'debug.stop', 'Shift+F5', () => this.stop()),
        };
        this.barPos = h('span', { class: 'db-pos' });
        const b = this.barButtons;
        this.bar = h('div', { class: 'debugbar', role: 'toolbar', 'aria-label': t('debug.toolbar') },
            h('span', { class: 'db-grip', 'aria-hidden': 'true' }),
            b.cont, b.over, b.into, b.out, h('span', { class: 'db-sep' }), b.back, b.rev, h('span', { class: 'db-sep' }), this.barPos, b.restart, b.stop);
        this.dragBar();
        this.app.layout.editorArea.appendChild(this.bar);
        this.updateBar();
    }

    dragBar() {
        const grip = this.bar.querySelector('.db-grip');
        grip.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            grip.setPointerCapture(e.pointerId);
            const startX = e.clientX, base = this.bar.offsetLeft;
            const move = (ev) => { this.bar.style.left = `${base + ev.clientX - startX}px`; this.bar.style.transform = 'none'; };
            const up = () => { grip.removeEventListener('pointermove', move); grip.removeEventListener('pointerup', up); };
            grip.addEventListener('pointermove', move);
            grip.addEventListener('pointerup', up);
        });
    }

    hideBar() {
        if (this.bar) this.bar.remove();
        this.bar = null;
    }

    updateBar() {
        if (!this.bar) return;
        const b = this.barButtons;
        b.cont.replaceChildren(icon(this.stopped ? 'continue' : 'pause'));
        tip(b.cont, this.stopped ? 'debug.continue' : 'debug.pause', this.stopped ? 'F5' : 'F6');
        for (const k of ['over', 'into', 'out', 'back', 'rev']) b[k].disabled = !this.stopped;
        const back = this.client && this.client.capabilities.supportsStepBack !== false;
        b.back.hidden = !back;
        b.rev.hidden = !back;
        this.barPos.textContent = this.timeline && this.stopped ? t('debug.position', { index: this.timeline.index + 1, total: this.timeline.total }) : '';
        this.barPos.hidden = !this.barPos.textContent;
    }
}

import { debugLineField } from '../editor/gutters.js';
function debugLineFieldRef() { return debugLineField; }

function samePath(a, b) {
    const n = (p) => String(p || '').replace(/\\/g, '/').replace(/\/+$/, '');
    const x = n(a), y = n(b);
    return /^[A-Za-z]:/.test(x) ? x.toLowerCase() === y.toLowerCase() : x === y;
}

/* "Debug with these inputs": the program with a call of the function appended
 * as top-level code (it runs before main(), and exit(0) skips main); the
 * lines of the program keep their numbers */
export function harnessText(src, call) {
    const args = call.args.map(formatValue).join(', ');
    return `${src.replace(/\s*$/, '')}\n\n# TIL Studio: ${t('debug.harnessNote')}\nprint(${call.function}(${args}))\nexit(0)\n`;
}

export function formatValue(v) {
    if (typeof v === 'string') return JSON.stringify(v);
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (v === null || v === undefined) return '0';
    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(v);
    if (typeof v === 'bigint') return v.toString();
    if (Array.isArray(v)) return `[${v.map(formatValue).join(', ')}]`;
    return String(v);
}
