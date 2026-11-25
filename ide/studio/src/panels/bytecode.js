// Bytecode: the disassembly of the current file (POST /api/tools/dis),
// synchronized with the editor: the cursor's line highlights its
// instructions; hovering an instruction highlights its source line; fused
// superinstructions are marked.
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, L } from '../core/i18n.js';
import { basename } from '../core/paths.js';
import { tip, toast } from '../ui/widgets.js';
import { setLinkedLines } from '../editor/gutters.js';
import { TABLES } from '../editor/til_lang.js';
import { desktopCard } from './debug.js';

const FUSED = new Set(TABLES.superinstructions || []);

function opClass(op) {
    if (/^JUMP|^LOOP|^FOR_ITER/.test(op)) return 'op-jump';
    if (/^CALL|^RETURN|^INVOKE|^NATIVE/.test(op)) return 'op-call';
    if (/^LOAD|^PUSH|^CONST/.test(op)) return 'op-load';
    if (/^STORE|^SET/.test(op)) return 'op-store';
    if (/^(ADD|SUB|MUL|DIV|MOD|NEG|INC|EQ|NE|LT|LE|GT|GE|AND|OR|NOT|BAND|BOR|BXOR|SHL|SHR|POW)/.test(op)) return 'op-arith';
    return '';
}

export class BytecodeView {
    constructor(app) {
        this.app = app;
        this.data = null;
        this.path = null;
        this.stale = false;
        this.busy = false;
        this.rowsByLine = new Map();
        this.fuse = h('input', { type: 'checkbox', checked: app.settings.get('fuseBytecode') !== false });
        this.fuse.addEventListener('change', () => { app.settings.set('fuseBytecode', this.fuse.checked); this.load(); });
        const refresh = h('button', { class: 'icon-btn small', type: 'button' }, icon('refresh'));
        tip(refresh, 'bytecode.refresh');
        refresh.addEventListener('click', () => this.load());
        this.summary = h('div', { class: 'bc-summary' });
        this.list = h('div', { class: 'bc-list', role: 'listbox', 'aria-label': t('activity.bytecode'), tabindex: '0' });
        this.list.addEventListener('mouseleave', () => this.link([]));
        this.banner = h('div', { class: 'bc-banner', hidden: true }, icon('info'), L('bytecode.stale'));
        this.titleEl = L('activity.bytecode');
        this.titleEl.className = 'side-title';
        this.body = h('div', { class: 'side-body bc-body' }, this.summary, this.banner, this.list);
        this.el = h('div', { class: 'side-view' }, h('div', { class: 'side-head' }, this.titleEl,
            h('div', { class: 'side-actions' }, h('label', { class: 'switch small-switch' }, this.fuse, h('span', { class: 'track' }), L('bytecode.fuse')), refresh)), this.body);
        app.docs.on('saved', (doc) => { if (doc.path === this.path && this.visible()) this.load(); });
        this.render();
    }

    visible() { return this.app.layout.activeSide === 'bytecode' && this.app.layout.sideOpen(); }

    onShow() {
        const doc = this.app.editors.activeDoc();
        if (doc && doc.isTil && doc.kind === 'file' && (doc.path !== this.path || !this.data)) this.load();
        else this.render();
    }

    focus() { this.list.focus(); }

    activeChanged(doc) {
        if (!this.visible()) return;
        if (doc && doc.isTil && doc.kind === 'file' && doc.path !== this.path) this.load();
    }

    markStale() {
        if (!this.data || this.stale) return;
        this.stale = true;
        this.banner.hidden = false;
    }

    async load() {
        if (this.app.web) { this.render(); return; }
        const doc = this.app.editors.activeDoc();
        if (!doc || !doc.isTil || doc.kind !== 'file') { this.data = null; this.path = null; this.render(); return; }
        this.busy = true;
        this.path = doc.path;
        this.render();
        try {
            this.data = await this.app.backend.dis({ path: doc.path, fuse: this.fuse.checked });
            this.stale = doc.dirty;
        } catch (e) {
            this.data = null;
            if (e && e.notAvailable) this.app.showDesktopOnly(e.feature);
            else this.error = e.message;
        }
        this.busy = false;
        this.render();
        const view = this.app.editors.activeView();
        if (view) this.syncCursor(view);
    }

    render() {
        clear(this.list);
        this.rowsByLine.clear();
        this.banner.hidden = !this.stale;
        if (this.app.web) { this.summary.textContent = ''; this.list.appendChild(desktopCard(this.app, 'bytecode')); return; }
        if (this.busy) { this.summary.textContent = ''; this.list.appendChild(h('div', { class: 'loading-row' }, h('span', { class: 'spinner' }), t('bytecode.loading'))); return; }
        if (!this.data) {
            this.summary.textContent = '';
            this.list.appendChild(h('div', { class: 'empty-state' }, icon('bytecode', { className: 'empty-icon' }), h('p', {}, this.error ? t('bytecode.failed', { error: this.error }) : t('bytecode.intro'))));
            this.error = null;
            return;
        }
        const d = this.data;
        const first = String(d.text || '').split('\n').find((l) => l.startsWith(';'));
        this.summary.textContent = first ? first.replace(/^;\s*/, '') : basename(this.path || '');
        const funcs = Array.isArray(d.functions) ? d.functions : [];
        const ins = Array.isArray(d.instructions) ? d.instructions : [];
        const frag = document.createDocumentFragment();
        let lastFunc = null;
        let n = 0;
        for (const i of ins) {
            if (i.func !== lastFunc) {
                lastFunc = i.func;
                const f = funcs.find((x) => x.name === i.func || x.index === i.func) || { name: i.func };
                frag.appendChild(h('div', { class: 'bc-func', role: 'presentation' }, icon('symbolFunction'), h('span', {}, `fn ${f.name !== undefined ? f.name : i.func}`),
                    f.start !== undefined ? h('span', { class: 'muted' }, `${f.start}–${f.end}`) : null));
            }
            // the server says which instructions are fused superinstructions; else by name
            const fused = typeof i.fused === 'boolean' ? i.fused : FUSED.has(i.op);
            const args = Array.isArray(i.args) ? i.args.join(' ') : i.args !== undefined && i.args !== null ? String(i.args) : '';
            const row = h('div', { class: ['bc-row', fused && 'fused', i.lineStart && 'line-start'], role: 'option', 'aria-selected': 'false', dataset: { line: String(i.line || 0), idx: String(n++) } },
                // the offset within the function, as `til dis` prints it (pc: the absolute address)
                h('span', { class: 'bc-pc', title: i.pc !== undefined ? t('bytecode.pcTip', { pc: i.pc }) : null }, String(i.offset !== undefined ? i.offset : i.pc).padStart(4, '0')),
                h('span', { class: `bc-op ${opClass(i.op)}` }, fused ? icon('zap', { className: 'bc-fused' }) : null, i.op),
                h('span', { class: 'bc-args' }, args),
                i.note ? h('span', { class: 'bc-note' }, i.note) : h('span', { class: 'bc-note' }),
                i.line ? h('span', { class: 'bc-line' }, String(i.line)) : h('span', { class: 'bc-line' }));
            if (fused) row.title = t('bytecode.fusedTip');
            if (i.line) {
                if (!this.rowsByLine.has(i.line)) this.rowsByLine.set(i.line, []);
                this.rowsByLine.get(i.line).push(row);
                row.addEventListener('mouseenter', () => this.link([i.line]));
                row.addEventListener('click', () => this.app.openFile(this.path, { line: i.line, focus: false }));
            }
            frag.appendChild(row);
        }
        this.list.appendChild(frag);
    }

    /* highlight source lines in the editor (bytecode -> editor) */
    link(lines) {
        const doc = this.path ? this.app.docs.get(this.path) : null;
        if (!doc) return;
        doc.dispatch({ effects: setLinkedLines.of(lines) });
        for (const r of this.list.querySelectorAll('.bc-row.hover')) r.classList.remove('hover');
        for (const l of lines) for (const r of this.rowsByLine.get(l) || []) r.classList.add('hover');
    }

    /* editor -> bytecode */
    syncCursor(view) {
        if (!this.data || !view || !view._studioDoc || view._studioDoc.path !== this.path) return;
        const line = view.state.doc.lineAt(view.state.selection.main.head).number;
        if (line === this.cursorLine) return;
        this.cursorLine = line;
        for (const r of this.list.querySelectorAll('.bc-row.current')) { r.classList.remove('current'); r.setAttribute('aria-selected', 'false'); }
        const rows = this.rowsByLine.get(line) || [];
        for (const r of rows) { r.classList.add('current'); r.setAttribute('aria-selected', 'true'); }
        if (rows.length && this.visible()) {
            const r = rows[0];
            const top = r.offsetTop - this.list.offsetTop, bottom = top + r.offsetHeight;
            const sc = this.body;
            if (top < sc.scrollTop || bottom > sc.scrollTop + sc.clientHeight) sc.scrollTo({ top: Math.max(0, top - sc.clientHeight / 3), behavior: 'smooth' });
        }
    }

    relocalize() { this.render(); }
}

export { toast };
