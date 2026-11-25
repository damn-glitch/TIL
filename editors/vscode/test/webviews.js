#!/usr/bin/env node
// The scripts of the two webviews (src/webviews.js), run in node against a
// minimal DOM: the screen panel decodes til/screen frames (base64 RGB24) into
// canvas pixels and scales them by whole device pixels; the time-travel view
// shows til/timeline and asks for til/goto. Prints a JSON report.
'use strict';
const Module = require('module');
const path = require('path');
const vm = require('vm');

const fakePath = path.join(__dirname, 'fake-vscode.js');
const orig = Module._resolveFilename;
Module._resolveFilename = function (r, ...rest) { return r === 'vscode' ? fakePath : orig.call(this, r, ...rest); };
const { PARTS, page } = require(path.join(__dirname, '..', 'src', 'webviews.js'));

class Element {
    constructor(id, tag = 'div') {
        this.id = id; this.tagName = tag; this.hidden = false; this.disabled = false; this.value = '';
        this.textContent = ''; this.style = {}; this.dataset = {}; this.listeners = {}; this.max = ''; this.min = '';
        this.width = 1; this.height = 1;
    }
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
    fire(type, ev = {}) { for (const fn of this.listeners[type] || []) fn(ev); }
    matches() { return false; }
    getBoundingClientRect() { return { width: 400, height: 300 }; }
}

function runPage(kind, ids, buttons, dpr) {
    const els = new Map(ids.map((id) => [id, new Element(id, id === 'screen' ? 'canvas' : 'div')]));
    const posted = [];
    let image = null;
    const ctx2d = {
        createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
        putImageData: (img) => { image = img; },
    };
    const canvas = els.get('screen');
    if (canvas) { canvas.getContext = () => ctx2d; canvas.toDataURL = () => 'data:image/png;base64,AAAA'; }
    const main = new Element('main');
    const winListeners = {};
    const document = {
        getElementById: (id) => els.get(id),
        querySelector: (sel) => (sel === 'main' ? main : null),
        querySelectorAll: (sel) => {
            if (sel === 'button, #slider, #index') return [...buttons, els.get('slider'), els.get('index')].filter(Boolean);
            if (sel === 'button[data-cmd]') return buttons.filter((b) => b.dataset.cmd);
            if (sel === 'button[data-goto]') return buttons.filter((b) => b.dataset.goto);
            return [];
        },
        activeElement: null,
    };
    let state;
    const sandbox = {
        document, atob: (b) => Buffer.from(b, 'base64').toString('latin1'),
        acquireVsCodeApi: () => ({ postMessage: (m) => posted.push(m), getState: () => state, setState: (s) => { state = s; } }),
        window: { addEventListener: (t, fn) => { winListeners[t] = fn; }, devicePixelRatio: dpr },
        ResizeObserver: class { observe() {} }, Number, Math, String, console,
    };
    const html = page({ cspSource: 'vscode-resource:' }, 'x', 'NONCE', PARTS[kind].style, PARTS[kind].body, PARTS[kind].script);
    const script = /<script nonce="NONCE">([\s\S]*)<\/script>/.exec(html)[1];
    vm.runInNewContext(script, sandbox);
    return { els, posted, image: () => image, send: (data) => winListeners.message({ data }), html, document };
}

const out = {};
// ── the screen ──
{
    const ids = ['screen', 'info', 'zoom', 'empty', 'save'];
    const p = runPage('screen', ids, [], 2);
    p.els.get('zoom').value = 'fit';
    const w = 4, h = 2;
    const rgb = Buffer.alloc(w * h * 3);
    for (let i = 0; i < w * h; i++) { rgb[i * 3] = i * 10; rgb[i * 3 + 1] = 200; rgb[i * 3 + 2] = 255 - i; }
    p.send({ type: 'frame', w, h, rgb: rgb.toString('base64'), session: 'game', reset: true });
    const img = p.image();
    const canvas = p.els.get('screen');
    out.screen = {
        ready: p.posted[0],
        canvas: [canvas.width, canvas.height],
        pixel5: img ? Array.from(img.data.slice(5 * 4, 5 * 4 + 4)) : null,
        cssWidth: canvas.style.width, cssHeight: canvas.style.height,
        hidden: canvas.hidden, emptyHidden: p.els.get('empty').hidden, saveDisabled: p.els.get('save').disabled,
        info: p.els.get('info').textContent,
        csp: /Content-Security-Policy" content="default-src 'none'; img-src vscode-resource: data:; style-src 'nonce-NONCE'; script-src 'nonce-NONCE';"/.test(p.html),
    };
    p.els.get('zoom').value = '3';
    p.els.get('zoom').fire('change');
    out.screen.zoom3 = canvas.style.width;
    p.send({ type: 'ended' });
    out.screen.ended = p.els.get('info').textContent;
    p.els.get('save').fire('click');
    out.screen.save = p.posted[p.posted.length - 1];
}
// ── the timeline ──
{
    const ids = ['idle', 'ui', 'pos', 'state', 'slider', 'index', 'go', 'note'];
    const mk = (dataset) => { const b = new Element('b', 'button'); b.dataset = dataset; return b; };
    const buttons = [mk({ goto: 'start' }), mk({ cmd: 'workbench.action.debug.reverseContinue' }), mk({ cmd: 'workbench.action.debug.stepBack' }),
        mk({ cmd: 'workbench.action.debug.stepOver' }), mk({ cmd: 'workbench.action.debug.continue' }), mk({ goto: 'end' }), mk({})];
    const p = runPage('timeline', ids, buttons, 1);
    out.timeline = { ready: p.posted[0], idleShown: !p.els.get('idle').hidden };
    p.send({ active: true, stopped: true, session: 'loop.til', index: 7, total: 20, complete: false });
    const slider = p.els.get('slider');
    out.timeline.slider = { max: slider.max, value: slider.value, disabled: slider.disabled };
    out.timeline.pos = p.els.get('pos').textContent;
    out.timeline.uiShown = !p.els.get('ui').hidden;
    slider.value = '3';
    slider.fire('change');
    buttons[0].fire('click');
    buttons[5].fire('click');
    buttons[2].fire('click');
    p.els.get('index').value = '99';
    p.els.get('go').fire('click');
    out.timeline.posted = p.posted.slice(1);
    p.send({ active: true, stopped: true, index: 20, total: 20, complete: true });
    out.timeline.completeMax = slider.max;
    p.send({ active: true, stopped: false, index: 20, total: 20, complete: true });
    out.timeline.runningDisabled = slider.disabled && buttons.every((b) => b.disabled);
    out.timeline.state = p.els.get('state').textContent;
    p.send({ active: false });
    out.timeline.idleAgain = !p.els.get('idle').hidden;
}
process.stdout.write(JSON.stringify(out) + '\n');
