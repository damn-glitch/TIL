// @ts-check
'use strict';
// Two webviews of the debugger: the program's screen (event til/screen: the
// framebuffer as base64 RGB24, drawn on a canvas and scaled with crisp pixels)
// and the time-travel view in the Run and Debug side bar (til/timeline +
// til/goto: a slider over every recorded line event).

const vscode = require('vscode');
const crypto = require('crypto');

function nonce() {
    return crypto.randomBytes(16).toString('base64').replace(/[^A-Za-z0-9]/g, '');
}

function page(webview, title, n, style, body, script) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} data:; style-src 'nonce-${n}'; script-src 'nonce-${n}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${title}</title>
<style nonce="${n}">
html, body { margin: 0; padding: 0; }
body { color: var(--vscode-foreground); font-family: var(--vscode-font-family); font-size: var(--vscode-font-size); }
button { color: var(--vscode-button-secondaryForeground, var(--vscode-foreground)); background: var(--vscode-button-secondaryBackground, transparent);
  border: 1px solid var(--vscode-button-border, transparent); border-radius: 4px; padding: 2px 8px; cursor: pointer; font: inherit; }
button:hover:not(:disabled) { background: var(--vscode-button-secondaryHoverBackground, var(--vscode-toolbar-hoverBackground)); }
button:disabled { opacity: 0.45; cursor: default; }
button:focus-visible, input:focus-visible, select:focus-visible { outline: 1px solid var(--vscode-focusBorder); outline-offset: 1px; }
select, input[type=number] { color: var(--vscode-input-foreground); background: var(--vscode-input-background);
  border: 1px solid var(--vscode-input-border, transparent); border-radius: 2px; font: inherit; padding: 1px 4px; }
.muted { opacity: 0.75; }
${style}
</style>
</head>
<body>
${body}
<script nonce="${n}">
const vscode = acquireVsCodeApi();
${script}
</script>
</body>
</html>`;
}

// ── the program's screen ────────────────────────────────────────────────────

const SCREEN_STYLE = `
html, body { height: 100%; }
body { display: flex; flex-direction: column; background: var(--vscode-editor-background); }
header { display: flex; gap: 8px; align-items: center; padding: 4px 8px; border-bottom: 1px solid var(--vscode-panel-border, transparent); }
#info { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
main { flex: 1; display: flex; align-items: center; justify-content: center; overflow: auto; min-height: 0; }
canvas { image-rendering: pixelated; image-rendering: crisp-edges; background: #000; outline: 1px solid var(--vscode-panel-border, #3338); }
#empty { padding: 16px; text-align: center; }
`;

const SCREEN_BODY = `
<header>
  <span id="info" class="muted">Waiting for the program to draw (screen_show)</span>
  <label class="muted" for="zoom">Zoom</label>
  <select id="zoom"><option value="fit">Fit</option><option value="1">1x</option><option value="2">2x</option>
    <option value="3">3x</option><option value="4">4x</option><option value="6">6x</option><option value="8">8x</option></select>
  <button id="save" disabled title="Save the frame as a PNG image">Save PNG</button>
</header>
<main><canvas id="screen" width="1" height="1" hidden></canvas><div id="empty" class="muted">No frame yet: the screen appears when a debugged program calls screen_show().</div></main>
`;

const SCREEN_SCRIPT = `
const canvas = document.getElementById('screen');
const g = canvas.getContext('2d');
const info = document.getElementById('info');
const zoom = document.getElementById('zoom');
const empty = document.getElementById('empty');
const save = document.getElementById('save');
const main = document.querySelector('main');
let frame = null;
let count = 0;
let ended = false;
const state = vscode.getState() || {};
if (state.zoom) zoom.value = state.zoom;

function draw(w, h, b64) {
  const bin = atob(b64);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const img = g.createImageData(w, h);
  const d = img.data;
  const n = Math.min(bin.length / 3, w * h);
  for (let i = 0, j = 0, k = 0; k < n; k++, i += 3, j += 4) {
    d[j] = bin.charCodeAt(i); d[j + 1] = bin.charCodeAt(i + 1); d[j + 2] = bin.charCodeAt(i + 2); d[j + 3] = 255;
  }
  g.putImageData(img, 0, 0);
}

function layout() {
  if (!frame) return;
  const box = main.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  let s;
  if (zoom.value === 'fit') {
    // the largest whole number of device pixels per program pixel (crisp), at least 1
    const fit = Math.min((box.width - 8) * dpr / frame.w, (box.height - 8) * dpr / frame.h);
    s = fit >= 1 ? Math.floor(fit) / dpr : Math.max(fit, 0.05) / dpr;
  } else {
    s = Number(zoom.value);
  }
  canvas.style.width = (frame.w * s) + 'px';
  canvas.style.height = (frame.h * s) + 'px';
}

function label() {
  if (!frame) return;
  info.textContent = (frame.session ? frame.session + ' \\u00b7 ' : '') + frame.w + ' \\u00d7 ' + frame.h +
    ' \\u00b7 frame ' + count + (ended ? ' \\u00b7 ended' : '');
}

window.addEventListener('message', (ev) => {
  const m = ev.data || {};
  if (m.type === 'frame') {
    if (m.reset) count = 0;
    ended = false;
    frame = { w: m.w, h: m.h, session: m.session };
    count++;
    draw(m.w, m.h, m.rgb);
    canvas.hidden = false;
    empty.hidden = true;
    save.disabled = false;
    layout();
    label();
  } else if (m.type === 'ended') {
    ended = true;
    label();
  }
});
zoom.addEventListener('change', () => { vscode.setState({ zoom: zoom.value }); layout(); });
save.addEventListener('click', () => vscode.postMessage({ type: 'savePng', data: canvas.toDataURL('image/png') }));
new ResizeObserver(layout).observe(main);
vscode.postMessage({ type: 'ready' });
`;

class ScreenPanel {
    /** @param {import('./extension').Context} ctx */
    constructor(ctx) {
        this.ctx = ctx;
        /** @type {vscode.WebviewPanel | undefined} */
        this.panel = undefined;
        /** @type {{w: number, h: number, rgb: string, session: string} | null} */
        this.last = null;
        /** sessions whose screen panel the user closed: not reopened by their frames */
        this.closedFor = new Set();
        this.currentSession = '';
    }

    /** A frame (event til/screen) of a session. */
    frame(session, body) {
        if (!body || typeof body.rgb_b64 !== 'string' || !(body.w > 0) || !(body.h > 0)) return;
        const reset = this.currentSession !== session.id;
        this.currentSession = session.id;
        this.last = { w: body.w, h: body.h, rgb: body.rgb_b64, session: session.name };
        if (!this.panel) {
            if (this.closedFor.has(session.id)) return;
            if (!vscode.workspace.getConfiguration('til').get('debug.showScreen', true)) return;
            this.open(false, session.id);
        }
        this.post({ type: 'frame', ...this.last, reset });
    }

    ended(session) {
        if (this.currentSession === session.id) this.post({ type: 'ended' });
        this.closedFor.delete(session.id);
    }

    /** Open (or reveal) the panel. */
    open(focus, sessionId) {
        if (this.panel) {
            this.panel.reveal(undefined, !focus);
            return;
        }
        const panel = vscode.window.createWebviewPanel('til.screen', 'TIL Screen',
            { viewColumn: vscode.ViewColumn.Beside, preserveFocus: !focus },
            { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [] });
        panel.iconPath = vscode.Uri.joinPath(this.ctx.extension.extensionUri, 'images', 'til-logo.png');
        panel.webview.html = page(panel.webview, 'TIL Screen', nonce(), SCREEN_STYLE, SCREEN_BODY, SCREEN_SCRIPT);
        panel.webview.onDidReceiveMessage((m) => this.message(m));
        panel.onDidDispose(() => {
            if (this.panel === panel) this.panel = undefined;
            const active = vscode.debug.activeDebugSession;
            if (sessionId || (active && active.type === 'til')) this.closedFor.add(sessionId || active.id);
        });
        this.panel = panel;
    }

    post(msg) {
        if (this.panel) this.panel.webview.postMessage(msg).then(undefined, () => { /* closed */ });
    }

    async message(m) {
        if (!m) return;
        if (m.type === 'ready' && this.last) this.post({ type: 'frame', ...this.last, reset: true });
        if (m.type === 'savePng' && typeof m.data === 'string' && m.data.startsWith('data:image/png;base64,')) {
            const target = await vscode.window.showSaveDialog({ filters: { 'PNG image': ['png'] }, saveLabel: 'Save Frame' });
            if (!target) return;
            await vscode.workspace.fs.writeFile(target, Buffer.from(m.data.slice('data:image/png;base64,'.length), 'base64'));
            vscode.window.setStatusBarMessage(`TIL: saved ${target.fsPath}`, 4000);
        }
    }

    /** The command "TIL: Show Program Screen". */
    show() {
        const active = vscode.debug.activeDebugSession;
        if (active) this.closedFor.delete(active.id);
        this.open(true, active ? active.id : '');
        if (this.last) this.post({ type: 'frame', ...this.last, reset: false });
    }

    dispose() {
        if (this.panel) this.panel.dispose();
    }
}

// ── time travel ─────────────────────────────────────────────────────────────

const ICONS = {
    start: '<path d="M3 3h2v10H3zM13 3v10L6 8z"/>',
    reverse: '<path d="M8 3v10L2 8zM14 3v10L8 8z"/>',
    back: '<path d="M11 3v10L4 8z"/>',
    step: '<path d="M5 3v10l7-5z"/>',
    forward: '<path d="M2 3v10l6-5zM8 3v10l6-5z"/>',
    end: '<path d="M11 3h2v10h-2zM3 3v10l7-5z"/>',
};

function icon(name) {
    return `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">${ICONS[name]}</svg>`;
}

const TIMELINE_STYLE = `
body { padding: 8px 12px; }
.row { display: flex; gap: 6px; align-items: center; margin: 6px 0; flex-wrap: wrap; }
.buttons button { display: inline-flex; align-items: center; padding: 3px 6px; }
#slider { width: 100%; margin: 6px 0; accent-color: var(--vscode-focusBorder); }
#pos { font-variant-numeric: tabular-nums; flex: 1; }
#index { width: 7em; }
p { margin: 6px 0; }
`;

const TIMELINE_BODY = `
<div id="idle" class="muted"><p>Debug a TIL program (F5) to travel in time: every line the program ran is recorded.</p>
<p>Step Back and Reverse Continue are on the debug toolbar; this slider jumps to any recorded moment.</p></div>
<div id="ui" hidden>
  <div class="row"><span id="pos">0 / 0</span><span id="state" class="muted"></span></div>
  <input id="slider" type="range" min="0" max="0" value="0" step="1" aria-label="Point in time (line event)">
  <div class="row buttons">
    <button data-goto="start" title="Go to the start of the program">${icon('start')}</button>
    <button data-cmd="workbench.action.debug.reverseContinue" title="Reverse Continue (to the previous breakpoint)">${icon('reverse')}</button>
    <button data-cmd="workbench.action.debug.stepBack" title="Step Back">${icon('back')}</button>
    <button data-cmd="workbench.action.debug.stepOver" title="Step Over">${icon('step')}</button>
    <button data-cmd="workbench.action.debug.continue" title="Continue">${icon('forward')}</button>
    <button data-goto="end" title="Go to the latest recorded moment">${icon('end')}</button>
  </div>
  <div class="row"><label for="index" class="muted">Line event</label><input id="index" type="number" min="0" value="0"><button id="go">Go</button></div>
  <p id="note" class="muted"></p>
</div>
`;

const TIMELINE_SCRIPT = `
const $ = (id) => document.getElementById(id);
let st = { active: false };
function maxIndex() { return st.complete ? st.total : Math.max(0, st.total - 1); }
function render() {
  $('idle').hidden = !!st.active;
  $('ui').hidden = !st.active;
  if (!st.active) return;
  const max = maxIndex();
  const slider = $('slider');
  slider.max = String(max);
  if (!slider.matches(':active')) slider.value = String(Math.min(st.index, max));
  $('index').max = String(max);
  if (document.activeElement !== $('index')) $('index').value = String(st.index);
  $('pos').textContent = 'line event ' + st.index + ' of ' + st.total + (st.complete ? ' (to the end)' : '');
  $('state').textContent = st.stopped ? 'paused' : (st.ended ? 'ended' : 'running');
  const stopped = !!st.stopped;
  for (const b of document.querySelectorAll('button, #slider, #index')) b.disabled = !stopped;
  $('note').textContent = st.session ? st.session : '';
}
function go(index) {
  index = Math.max(0, Math.min(maxIndex(), Math.round(Number(index))));
  if (!Number.isFinite(index)) return;
  vscode.postMessage({ type: 'goto', index });
}
window.addEventListener('message', (ev) => { st = ev.data || { active: false }; render(); });
$('slider').addEventListener('input', () => { $('pos').textContent = 'line event ' + $('slider').value + ' of ' + st.total; });
$('slider').addEventListener('change', () => go($('slider').value));
$('go').addEventListener('click', () => go($('index').value));
$('index').addEventListener('keydown', (e) => { if (e.key === 'Enter') go($('index').value); });
for (const b of document.querySelectorAll('button[data-cmd]')) b.addEventListener('click', () => vscode.postMessage({ type: 'command', id: b.dataset.cmd }));
for (const b of document.querySelectorAll('button[data-goto]')) b.addEventListener('click', () => go(b.dataset.goto === 'start' ? 0 : maxIndex()));
vscode.postMessage({ type: 'ready' });
`;

const ALLOWED_COMMANDS = new Set(['workbench.action.debug.reverseContinue', 'workbench.action.debug.stepBack',
    'workbench.action.debug.stepOver', 'workbench.action.debug.continue']);

/** The "TIL Time Travel" view (a webview view in the Run and Debug side bar). */
class TimelineView {
    /** @param {import('./extension').Context} ctx  @param {{isStopped(s: vscode.DebugSession): boolean, goto(s: vscode.DebugSession, i: number): Promise<void>}} debug */
    constructor(ctx, debug) {
        this.ctx = ctx;
        this.debug = debug;
        /** @type {vscode.WebviewView | undefined} */
        this.view = undefined;
        this.state = { active: false };
        this.pending = null;
    }

    /** @param {vscode.WebviewView} view */
    resolveWebviewView(view) {
        this.view = view;
        view.webview.options = { enableScripts: true, localResourceRoots: [] };
        view.webview.html = page(view.webview, 'TIL Time Travel', nonce(), TIMELINE_STYLE, TIMELINE_BODY, TIMELINE_SCRIPT);
        view.webview.onDidReceiveMessage((m) => this.message(m));
        view.onDidDispose(() => { if (this.view === view) this.view = undefined; });
        view.onDidChangeVisibility(() => { if (view.visible) this.refresh(); });
    }

    async message(m) {
        if (!m) return;
        if (m.type === 'ready') {
            this.post();
            this.refresh();
        }
        const session = vscode.debug.activeDebugSession;
        if (m.type === 'goto' && session && session.type === 'til' && Number.isInteger(m.index)) {
            await this.debug.goto(session, m.index);
        }
        if (m.type === 'command' && ALLOWED_COMMANDS.has(m.id)) {
            vscode.commands.executeCommand(m.id).then(undefined, () => { /* not available now */ });
        }
    }

    post() {
        if (this.view) this.view.webview.postMessage(this.state).then(undefined, () => { /* hidden */ });
    }

    /** Ask the active session where it is in time (til/timeline) and show it. */
    refresh(session = vscode.debug.activeDebugSession) {
        if (!session || session.type !== 'til') {
            this.state = { active: false };
            this.post();
            return;
        }
        const stopped = this.debug.isStopped(session);
        if (!stopped) {
            this.state = { ...this.state, active: true, stopped: false, session: session.name };
            this.post();
            return;
        }
        const ask = session.customRequest('til/timeline').then((t) => {
            if (this.pending !== ask) return;
            this.state = { active: true, stopped: this.debug.isStopped(session), session: session.name,
                index: Number(t && t.index) || 0, total: Number(t && t.total) || 0, complete: !!(t && t.complete) };
            this.post();
        }, () => { /* the session ended meanwhile */ });
        this.pending = ask;
    }

    ended(session) {
        const active = vscode.debug.activeDebugSession;
        if (!active || active.id === session.id) {
            this.state = this.state.active ? { ...this.state, stopped: false, ended: true } : { active: false };
            this.post();
        }
    }
}

/** The pages' parts (style, body, script): the tests run the scripts against a fake DOM. */
const PARTS = {
    screen: { style: SCREEN_STYLE, body: SCREEN_BODY, script: SCREEN_SCRIPT },
    timeline: { style: TIMELINE_STYLE, body: TIMELINE_BODY, script: TIMELINE_SCRIPT },
};

module.exports = { ScreenPanel, TimelineView, page, nonce, PARTS };
