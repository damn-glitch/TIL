// Proofs: `til prove` per function (POST /api/tools/prove): proved,
// counterexample (with the values and "Debug with these inputs"), not proved,
// unknown, unsupported (with the reason); marks in the editor gutter.
// Integers beyond 2^53 arrive as decimal text and stay exact.
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, tn, L } from '../core/i18n.js';
import { basename } from '../core/paths.js';
import { tip, toast } from '../ui/widgets.js';
import { setProofMarks } from '../editor/gutters.js';
import { desktopCard } from './debug.js';
import { formatValue } from '../debug/debugger.js';

const STATUS = {
    proved: { icon: 'checkCircle', cls: 'ok' },
    counterexample: { icon: 'crossCircle', cls: 'bad' },
    'not proved': { icon: 'questionCircle', cls: 'warn' },
    unknown: { icon: 'questionCircle', cls: 'warn' },
    unsupported: { icon: 'banCircle', cls: 'muted' },
};

export class ProofsView {
    constructor(app) {
        this.app = app;
        this.results = new Map();      // path -> {available, results, time}
        this.busy = false;
        this.proveBtn = h('button', { class: 'btn primary small', type: 'button' }, icon('proofs'), L('proofs.proveFile'));
        this.proveBtn.addEventListener('click', () => this.prove());
        this.fileLabel = h('div', { class: 'proofs-file' });
        this.body = h('div', { class: 'side-body pad proofs-body' });
        this.titleEl = L('activity.proofs');
        this.titleEl.className = 'side-title';
        this.el = h('div', { class: 'side-view' }, h('div', { class: 'side-head' }, this.titleEl), h('div', { class: 'proofs-top' }, this.fileLabel, this.proveBtn), this.body);
        this.render();
    }

    onShow() { this.render(); }

    focus() { this.proveBtn.focus(); }

    targetPath() {
        const doc = this.app.editors.activeDoc();
        return doc && doc.isTil && doc.kind === 'file' ? doc.path : null;
    }

    activeChanged() { this.render(); this.applyMarks(); }

    async prove(pathArg, fn) {
        if (this.app.web) { this.app.showDesktopOnly('prover'); return; }
        const path = pathArg || this.targetPath();
        if (!path) { toast(t('proofs.noFile'), { kind: 'info' }); return; }
        const doc = this.app.docs.get(path);
        if (doc && doc.dirty) { const ok = await this.app.saveDoc(doc); if (!ok) return; }
        this.busy = path;
        this.render();
        let res;
        try {
            const body = { path };
            if (fn) body.function = fn;
            res = await this.app.backend.prove(body);
        } catch (e) {
            this.busy = false;
            this.render();
            if (e && e.notAvailable) { this.app.showDesktopOnly(e.feature); return; }
            toast(t('proofs.failed'), { kind: 'error', detail: e.message });
            return;
        }
        this.busy = false;
        const prev = this.results.get(path);
        let results = Array.isArray(res.results) ? res.results : [];
        if (fn && prev && prev.results) results = prev.results.filter((r) => r.function !== fn).concat(results).sort((a, b) => (a.line || 0) - (b.line || 0));
        this.results.set(path, {
            available: res.available !== false, results, time: Date.now(),
            hint: typeof res.hint === 'string' && res.hint ? res.hint : 'pip install z3-solver',
            error: typeof res.error === 'string' ? res.error : '',
        });
        this.render();
        this.applyMarks();
        const bad = results.filter((r) => r.status === 'counterexample').length;
        if (res.available !== false) this.app.statusbar.flash(bad ? tn('proofs.foundCounter', bad) : t('proofs.done'));
    }

    applyMarks() {
        for (const [path, r] of this.results) {
            const doc = this.app.docs.get(path);
            if (!doc) continue;
            const marks = (r.results || []).filter((x) => x.line).map((x) => ({ line: x.line, status: x.status, title: `${x.function}: ${t(`proofs.status.${x.status.replace(' ', '_')}`)}` }));
            doc.dispatch({ effects: setProofMarks.of(marks) });
        }
    }

    render() {
        clear(this.body);
        const path = this.targetPath();
        this.fileLabel.textContent = path ? basename(path) : t('proofs.openFile');
        this.proveBtn.disabled = !path || Boolean(this.busy) || this.app.web;
        if (this.app.web) { this.body.appendChild(desktopCard(this.app, 'prover')); return; }
        if (this.busy) {
            this.body.appendChild(h('div', { class: 'loading-row' }, h('span', { class: 'spinner' }), t('proofs.proving', { name: basename(this.busy) })));
            return;
        }
        const r = path ? this.results.get(path) : null;
        if (!r) {
            this.body.appendChild(h('div', { class: 'empty-state' }, icon('proofs', { className: 'empty-icon' }), L('proofs.intro', null, 'p'),
                h('pre', { class: 'proofs-sample' }, '#[requires: b != 0]\n#[ensures: result * b <= a]\ndivide(a: int, b: int) -> int')));
            return;
        }
        if (!r.available) {
            const copy = h('button', { class: 'btn small', type: 'button' }, icon('copy'), L('desktop.copy'));
            copy.addEventListener('click', () => this.app.copyText(r.hint));
            this.body.appendChild(h('div', { class: 'note-card warn' }, icon('warning'), h('div', {}, L('proofs.noZ3', null, 'p'), h('div', { class: 'dc-cmd' }, h('code', {}, r.hint), copy))));
            return;
        }
        if (r.error && !r.results.length) {
            // the file does not compile (or names no such function): the prover's message
            this.body.appendChild(h('div', { class: 'note-card bad' }, icon('error'), h('div', {}, L('proofs.compileError', null, 'p'), h('pre', { class: 'proof-errors' }, r.error))));
            return;
        }
        if (!r.results.length) {
            this.body.appendChild(h('div', { class: 'empty-state' }, icon('info', { className: 'empty-icon' }), L('proofs.nothing', null, 'p')));
            return;
        }
        const counts = {};
        for (const x of r.results) counts[x.status] = (counts[x.status] || 0) + 1;
        const chips = h('div', { class: 'proof-summary' });
        for (const st of ['proved', 'counterexample', 'not proved', 'unknown', 'unsupported']) {
            if (!counts[st]) continue;
            const cls = { proved: 'green', counterexample: 'red', unsupported: '', 'not proved': 'yellow', unknown: 'yellow' }[st];
            chips.appendChild(h('span', { class: `chip ${cls}` }, `${counts[st]} ${t(`proofs.status.${st.replace(' ', '_')}`)}`));
        }
        this.body.appendChild(chips);
        const doc = this.app.docs.get(path);
        const text = doc ? doc.current().doc.toString() : '';
        for (const x of r.results) this.body.appendChild(this.card(path, x, text));
    }

    card(path, x, text = '') {
        const st = STATUS[x.status] || STATUS.unknown;
        const head = h('div', { class: 'proof-head' },
            icon(st.icon, { className: `proof-icon ${st.cls}` }),
            h('span', { class: 'proof-fn' }, x.title || x.function),
            h('span', { class: `proof-status ${st.cls}` }, t(`proofs.status.${String(x.status).replace(' ', '_')}`)),
            x.line ? h('button', { class: 'proof-line', type: 'button', on: { click: () => this.app.openFile(path, { line: x.line }) } }, t('proofs.line', { line: x.line })) : null);
        const card = h('div', { class: `proof-card ${st.cls}` }, head);
        if (x.detail) card.appendChild(h('div', { class: 'proof-detail' }, String(x.detail)));
        else if (x.reason) card.appendChild(h('div', { class: 'proof-detail' }, String(x.reason)));
        const failures = Array.isArray(x.failures) ? x.failures.filter((f) => f && f.what) : [];
        if (failures.length && x.status !== 'proved') {
            const list = h('ul', { class: 'proof-failures', 'aria-label': t('proofs.failures') });
            for (const f of failures.slice(0, 6)) {
                const item = h('li', {}, f.line ? h('button', { class: 'pf-line', type: 'button', on: { click: () => this.app.openFile(path, { line: f.line, col: f.col }) } }, t('proofs.line', { line: f.line })) : null, h('code', {}, String(f.what)));
                list.appendChild(item);
            }
            card.appendChild(list);
        }
        if (Array.isArray(x.errors) && x.errors.length) card.appendChild(h('pre', { class: 'proof-errors' }, x.errors.join('\n')));
        if (x.counterexample && typeof x.counterexample === 'object') {
            const types = new Map((x.function ? paramsOf(text, x.function) : []).map((p) => [p.name, p.type]));
            const table = h('div', { class: 'cex' });
            for (const [k, v] of Object.entries(x.counterexample)) {
                table.append(h('span', { class: 'cex-name' }, k), h('code', { class: 'cex-value' }, formatValue(typedValue(v, types.get(k)))));
            }
            card.appendChild(h('div', { class: 'cex-title' }, t('proofs.counterexample')));
            card.appendChild(table);
            const btn = h('button', { class: 'btn small', type: 'button' }, icon('bug'), L('proofs.debugInputs'));
            tip(btn, 'proofs.debugInputsTip');
            btn.disabled = !this.app.caps.debugger;
            btn.addEventListener('click', () => this.debugWith(path, x));
            const rerun = h('button', { class: 'btn ghost small', type: 'button' }, icon('refresh'), L('proofs.recheck'));
            rerun.addEventListener('click', () => this.prove(path, x.function));
            card.appendChild(h('div', { class: 'proof-actions' }, btn, rerun));
        }
        return card;
    }

    async debugWith(path, x) {
        const doc = this.app.docs.get(path) || await this.app.docs.open(path);
        const text = doc.current().doc.toString();
        const params = paramsOf(text, x.function);
        const ce = x.counterexample || {};
        const args = params.map((p) => (Object.prototype.hasOwnProperty.call(ce, p.name) ? typedValue(ce[p.name], p.type) : defaultFor(p.type)));
        await this.app.debugger.start(doc, { call: { function: x.function, args } });
    }

    relocalize() { this.render(); }
}

/* the parameters of a function definition: [{name, type}] */
export function paramsOf(text, fn) {
    const re = new RegExp(`^\\s*(?:pub\\s+)?(?:fn\\s+|функция\\s+|funktsiya\\s+)?${fn.replace(/[^\p{L}\p{N}_]/gu, '')}\\s*\\(([^)]*)\\)`, 'mu');
    const m = re.exec(text);
    if (!m) return [];
    return m[1].split(',').map((s) => s.trim()).filter(Boolean).map((s) => {
        const [name, type] = s.split(':').map((x) => x.trim());
        return { name: name.replace(/=.*$/, '').trim(), type: (type || '').replace(/=.*$/, '').trim() };
    });
}

/* a counterexample value for a parameter of this type: an integer sent as
   decimal text (beyond 2^53) becomes an exact BigInt */
export function typedValue(v, type) {
    if (typeof v === 'string' && /^-?\d+$/.test(v) && /^(int|uint|[iu](8|16|32|64))$/.test(String(type || '').trim())) {
        try { return BigInt(v); } catch (e) { return v; }
    }
    return v;
}

function defaultFor(type) {
    if (/^(float|f32|f64)$/.test(type)) return 0.0;
    if (type === 'str') return '';
    if (type === 'bool') return false;
    if (/^\[/.test(type)) return [];
    return 0;
}
