// Screen: the program's framebuffer, scaled crisp by an integer factor.
// With the focus on it, keys go to the program (Tab still leaves). Optional
// calculator bezel with clickable keys. Frames: base64 RGB24 (run events,
// til/screen of the debugger) or RGBA (the browser runtime).
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, L } from '../core/i18n.js';
import { tip } from '../ui/widgets.js';
import { keyCodeOf } from '../run/runner.js';

function b64ToBytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
}

export class ScreenView {
    constructor(app) {
        this.app = app;
        this.canvas = h('canvas', { class: 'screen-canvas', width: 160, height: 120, tabindex: '0', 'aria-label': t('screen.canvasLabel') });
        this.canvas.addEventListener('keydown', (e) => this.onKey(e));
        this.canvas.addEventListener('pointerdown', () => this.canvas.focus({ preventScroll: true }));
        this.canvas.addEventListener('focus', () => this.frameEl.classList.add('focused'));
        this.canvas.addEventListener('blur', () => this.frameEl.classList.remove('focused'));
        this.info = h('div', { class: 'screen-info' });
        this.hint = h('div', { class: 'screen-hint' });
        this.keypad = this.buildKeypad();
        this.frameEl = h('div', { class: 'screen-frame' },
            h('div', { class: 'bezel-top' }, h('span', { class: 'bezel-brand' }, 'TIL', h('b', {}, '·84')), h('span', { class: 'bezel-model' }, 'GRAPHING VM')),
            h('div', { class: 'screen-glass' }, this.canvas),
            this.keypad);
        this.emptyEl = h('div', { class: 'screen-empty' }, icon('screen'), L('screen.empty', null, 'p'));
        this.stage = h('div', { class: 'screen-stage' }, this.frameEl, this.emptyEl);
        this.el = h('div', { class: 'screen-view' }, this.stage, h('div', { class: 'screen-bar' }, this.info, this.hint));
        this.size = null;
        this.frames = 0;
        this.applyBezel();
        this.reset();
        if (typeof ResizeObserver === 'function') new ResizeObserver(() => this.fit()).observe(this.stage);
        app.settings.on('change:calculatorBezel', () => { this.applyBezel(); this.fit(); });
        app.settings.on('change:screenScale', () => this.fit());
    }

    actions() {
        const bezel = h('button', { class: 'icon-btn small', type: 'button', 'aria-pressed': String(Boolean(this.app.settings.get('calculatorBezel'))) }, icon('calculator'));
        tip(bezel, 'screen.bezel');
        bezel.addEventListener('click', () => {
            const on = !this.app.settings.get('calculatorBezel');
            this.app.settings.set('calculatorBezel', on);
            bezel.setAttribute('aria-pressed', String(on));
        });
        const scale = h('select', { class: 'select small-select', 'aria-label': t('screen.scale') },
            ...[0, 1, 2, 3, 4, 6].map((s) => h('option', { value: String(s), selected: this.app.settings.get('screenScale') === s }, s ? `${s}×` : t('screen.fit'))));
        scale.addEventListener('change', () => this.app.settings.set('screenScale', Number(scale.value)));
        const save = h('button', { class: 'icon-btn small', type: 'button' }, icon('camera'));
        tip(save, 'screen.savePng');
        save.addEventListener('click', () => this.savePng());
        return [scale, bezel, save];
    }

    buildKeypad() {
        const key = (label, code, cls, aria) => {
            const b = h('button', { class: `pad-key ${cls || ''}`, type: 'button', tabindex: '-1', 'aria-label': aria || label, dataset: { code: String(code) } }, label);
            b.addEventListener('pointerdown', (e) => { e.preventDefault(); this.send(code); b.classList.add('down'); });
            b.addEventListener('pointerup', () => b.classList.remove('down'));
            b.addEventListener('pointerleave', () => b.classList.remove('down'));
            return b;
        };
        return h('div', { class: 'keypad' },
            h('div', { class: 'pad-row' }, key('Esc', 27, 'fn'), key('Space', 32, 'fn wide'), key('Enter', 13, 'fn accent')),
            h('div', { class: 'dpad' }, key('▲', 1, 'up', t('screen.keyUp')), key('◀', 3, 'left', t('screen.keyLeft')), key('▶', 4, 'right', t('screen.keyRight')), key('▼', 2, 'down', t('screen.keyDown'))));
    }

    applyBezel() {
        const on = Boolean(this.app.settings.get('calculatorBezel'));
        this.frameEl.classList.toggle('bezel', on);
        // the calculator body needs room: grow a short bottom panel
        const lay = this.app.layout;
        if (on && lay && lay.panel && lay.panel.offsetHeight && lay.panel.offsetHeight < 420) {
            lay.setSizes({ panel: 420 });
            this.app.editorsResized();
            if (this.app.saveLayout) this.app.saveLayout();
        }
    }

    focus() { if (this.size) this.canvas.focus(); }

    reset() {
        this.size = null;
        this.frames = 0;
        this.frameEl.hidden = true;
        this.emptyEl.hidden = false;
        this.info.textContent = '';
        this.hint.textContent = '';
    }

    ended() {
        if (this.size) this.hint.textContent = t('screen.ended');
    }

    open(w, hgt) {
        const first = !this.size;
        if (this.canvas.width !== w || this.canvas.height !== hgt) { this.canvas.width = w; this.canvas.height = hgt; }
        this.size = [w, hgt];
        this.frameEl.hidden = false;
        this.emptyEl.hidden = true;
        this.fit();
        if (first) {
            this.app.layout.showPanel('screen');
            this.roomFor(w, hgt);
            this.hint.textContent = t('screen.keysHint');
            requestAnimationFrame(() => this.canvas.focus({ preventScroll: true }));
        }
    }

    /* a short bottom panel grows (once per run, up to about half the window)
       so a small screen gets at least twice its size */
    roomFor(w, hgt) {
        const lay = this.app.layout;
        if (!lay || !lay.panel || !lay.panel.offsetHeight || this.app.settings.get('screenScale')) return;
        const around = lay.panel.offsetHeight - this.stage.clientHeight;
        const bezel = this.frameEl.classList.contains('bezel');
        const chrome = bezel && this.frameEl.offsetHeight ? 28 + this.frameEl.offsetHeight - this.canvas.offsetHeight : 24;
        const need = Math.ceil(around + chrome + 2 * hgt);
        if (need > lay.panel.offsetHeight && need <= window.innerHeight * 0.55 && 2 * w <= this.stage.clientWidth - 32) {
            lay.setSizes({ panel: need });
            this.app.editorsResized();
            requestAnimationFrame(() => this.fit());
        }
    }

    frameFromRun(ev) {
        if (!ev.w || !ev.h || !ev.rgb) return;
        this.frameRGB(ev.w, ev.h, b64ToBytes(ev.rgb));
    }

    frameFromDap(body) {
        const rgb = body.rgb_b64 || body.rgb;
        if (!body.w || !body.h || !rgb) return;
        this.frameRGB(body.w, body.h, b64ToBytes(rgb));
    }

    frameRGB(w, hgt, rgb) {
        if (rgb.length < w * hgt * 3) return;
        this.open(w, hgt);
        const img = new ImageData(w, hgt);
        const d = img.data;
        for (let i = 0, j = 0; i < w * hgt; i++, j += 3) {
            const k = i * 4;
            d[k] = rgb[j]; d[k + 1] = rgb[j + 1]; d[k + 2] = rgb[j + 2]; d[k + 3] = 255;
        }
        this.canvas.getContext('2d').putImageData(img, 0, 0);
        this.countFrame();
    }

    frameRGBA(w, hgt, pixels) {
        this.open(w, hgt);
        this.canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pixels), w, hgt), 0, 0);
        this.countFrame();
    }

    countFrame() {
        this.frames++;
        this.canvas.dataset.frame = String(this.frames);
        this.info.textContent = t('screen.info', { w: this.size[0], h: this.size[1], n: this.frames });
    }

    fit() {
        if (!this.size) return;
        const [w, hgt] = this.size;
        const want = this.app.settings.get('screenScale');
        const bezel = this.frameEl.classList.contains('bezel');
        // the calculator body around the glass: measured (it is additive in height)
        let chromeH = 24;
        if (bezel) chromeH = this.frameEl.offsetHeight ? 28 + this.frameEl.offsetHeight - this.canvas.offsetHeight : 210;
        const availW = Math.max(64, this.stage.clientWidth - (bezel ? 90 : 32));
        const availH = Math.max(48, this.stage.clientHeight - chromeH);
        let scale = want || Math.min(availW / w, availH / hgt);
        if (scale >= 1) scale = Math.floor(scale);
        else if (bezel) scale = 1;            // the stage scrolls rather than shrinking a small screen
        scale = Math.max(scale, 0.25);
        this.canvas.style.width = `${Math.round(w * scale)}px`;
        this.canvas.style.height = `${Math.round(hgt * scale)}px`;
    }

    onKey(e) {
        if (e.ctrlKey || e.metaKey || e.altKey || e.key === 'Tab') return;
        const code = keyCodeOf(e);
        if (!code) return;
        if (this.send(code)) e.preventDefault();
    }

    send(code) {
        const runner = this.app.runner;
        if (runner.running) return runner.sendKey(code);
        if (this.app.debugger.active) { this.app.statusbar.flash(t('screen.debugKeys')); return true; }
        return false;
    }

    savePng() {
        if (!this.size) return;
        const a = h('a', { href: this.canvas.toDataURL('image/png'), download: 'til-screen.png' });
        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    relocalize() {
        this.canvas.setAttribute('aria-label', t('screen.canvasLabel'));
        if (this.size) this.info.textContent = t('screen.info', { w: this.size[0], h: this.size[1], n: this.frames });
    }
}

export { clear };
