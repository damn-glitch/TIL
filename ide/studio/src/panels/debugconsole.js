// Debug console: messages of the debugger and a REPL that evaluates
// expressions in the selected stack frame (DAP evaluate, context "repl").
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, L } from '../core/i18n.js';
import { tip } from '../ui/widgets.js';

export class DebugConsoleView {
    constructor(app) {
        this.app = app;
        this.log_ = h('div', { class: 'dc-log', role: 'log', 'aria-live': 'polite', tabindex: '0' });
        this.input = h('input', { class: 'term-input', type: 'text', spellcheck: 'false', autocomplete: 'off', placeholder: t('console.placeholder'), 'aria-label': t('console.placeholder') });
        this.form = h('form', { class: 'term-input-row dc-input' }, h('span', { class: 'term-prompt' }, '»'), this.input);
        this.form.addEventListener('submit', (e) => { e.preventDefault(); this.evaluate(); });
        this.input.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowUp') { e.preventDefault(); this.hist(-1); }
            if (e.key === 'ArrowDown') { e.preventDefault(); this.hist(1); }
        });
        this.history = [];
        this.pos = 0;
        this.el = h('div', { class: 'console-view term' }, this.log_, this.form);
        this.empty();
    }

    actions() {
        const c = h('button', { class: 'icon-btn small', type: 'button' }, icon('clear'));
        tip(c, 'output.clear');
        c.addEventListener('click', () => this.empty());
        return [c];
    }

    focus() { this.input.focus(); }

    empty() {
        clear(this.log_);
        this.log_.appendChild(h('div', { class: 'term-empty' }, icon('console'), L('console.empty')));
        this.isEmpty = true;
    }

    line(el) {
        if (this.isEmpty) { clear(this.log_); this.isEmpty = false; }
        this.log_.appendChild(el);
        this.log_.scrollTop = this.log_.scrollHeight;
    }

    log(text, category = 'console') {
        if (!text) return;
        this.line(h('div', { class: `dc-line dc-${category}` }, text.replace(/\n$/, '')));
    }

    async evaluate() {
        const expr = this.input.value.trim();
        if (!expr) return;
        this.input.value = '';
        this.history.push(expr);
        this.pos = this.history.length;
        this.line(h('div', { class: 'dc-line dc-in' }, h('span', { class: 'dc-chev' }, '»'), expr));
        const dbg = this.app.debugger;
        if (!dbg.active || !dbg.stopped) { this.line(h('div', { class: 'dc-line dc-err' }, t('console.notStopped'))); return; }
        try {
            const r = await dbg.evaluate(expr, 'repl');
            this.line(h('div', { class: 'dc-line dc-out' }, h('span', { class: 'dc-chev' }, '«'), r.result, r.type ? h('span', { class: 'var-type' }, r.type) : null));
        } catch (e) {
            this.line(h('div', { class: 'dc-line dc-err' }, e.message));
        }
    }

    hist(d) {
        if (!this.history.length) return;
        this.pos = Math.max(0, Math.min(this.history.length, this.pos + d));
        this.input.value = this.history[this.pos] || '';
    }

    relocalize() {
        this.input.placeholder = t('console.placeholder');
        if (this.isEmpty) this.empty();
    }
}
