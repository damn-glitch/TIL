// The editor area: one or two groups (split right), each with tabs (dirty
// markers, close buttons, context menu), breadcrumbs and one CodeMirror view
// that shows the active document (or a special view: welcome page, translation).
import { EditorView } from '@codemirror/view';
import { EditorSelection } from '@codemirror/state';
import { h, clear, focusSoon } from '../core/dom.js';
import { icon, fileIcon, logoMark } from '../core/icons.js';
import { t, L } from '../core/i18n.js';
import { formatKeys } from '../core/commands.js';
import { dirname } from '../core/paths.js';
import { openMenu, tip } from './widgets.js';
import { createView } from '../editor/editor.js';
import { flashEffect } from '../editor/gutters.js';

let flashSeq = 0;

class EditorGroup {
    constructor(area, index) {
        this.area = area;
        this.app = area.app;
        this.index = index;
        this.tabs = [];               // {key, kind: 'doc'|'special', doc, special}
        this.active = null;
        this.view = null;
        this.memory = new Map();      // path -> {scrollTop, selection}
        this.tabsEl = h('div', { class: 'tabs', role: 'tablist', 'aria-label': t('editor.tabs') });
        this.tabsEl.addEventListener('keydown', (e) => this.tabKeys(e));
        this.tabsEl.addEventListener('wheel', (e) => { if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { this.tabsEl.scrollLeft += e.deltaY; e.preventDefault(); } }, { passive: false });
        this.actionsEl = h('div', { class: 'tabbar-actions' });
        this.crumbsEl = h('div', { class: 'crumbs', 'aria-label': t('editor.breadcrumbs') });
        this.host = h('div', { class: 'editor-host' });
        this.specialHost = h('div', { class: 'special-host', hidden: true });
        this.emptyEl = this.buildEmpty();
        this.el = h('div', { class: 'editor-group', dataset: { group: String(index) } },
            h('div', { class: 'tabbar' }, this.tabsEl, this.actionsEl), this.crumbsEl, this.host, this.specialHost, this.emptyEl);
        this.el.addEventListener('focusin', () => this.area.setFocused(this));
        this.el.addEventListener('mousedown', () => this.area.setFocused(this));
        this.buildActions();
        this.updateEmpty();
    }

    buildEmpty() {
        const rows = [
            ['palette.showCommands', 'workbench.commandPalette'], ['palette.quickOpen', 'workbench.quickOpen'],
            ['cmd.runFile', 'run.run'], ['cmd.debugFile', 'debug.start'], ['cmd.togglePanel', 'workbench.togglePanel'],
        ];
        const grid = h('div', { class: 'shortcuts' });
        for (const [label, id] of rows) {
            grid.append(L(label), h('span', {}, h('kbd', {}, formatKeys((this.app.commands.keysFor(id)[0]) || ''))));
        }
        this.emptyGrid = grid;
        return h('div', { class: 'editor-empty' }, logoMark(84), grid);
    }

    refreshEmpty() {
        const old = this.emptyEl;
        this.emptyEl = this.buildEmpty();
        old.replaceWith(this.emptyEl);
        this.updateEmpty();
    }

    buildActions() {
        const run = h('button', { class: 'icon-btn small', type: 'button' }, icon('play'));
        tip(run, 'editor.runFile', 'Mod+F5');
        run.addEventListener('click', () => this.app.commands.run('run.run'));
        const split = h('button', { class: 'icon-btn small', type: 'button' }, icon('split'));
        tip(split, 'editor.splitRight', 'Mod+\\');
        split.addEventListener('click', () => this.area.splitRight(this));
        const more = h('button', { class: 'icon-btn small', type: 'button', 'aria-haspopup': 'menu' }, icon('more'));
        tip(more, 'editor.more');
        more.addEventListener('click', () => this.app.editorMoreMenu(more.getBoundingClientRect()));
        this.runBtn = run;
        this.actionsEl.append(run, split, more);
    }

    current() { return this.tabs.find((x) => x.key === this.active) || null; }

    activeDoc() { const c = this.current(); return c && c.kind === 'doc' ? c.doc : null; }

    ensureView(doc) {
        if (!this.view) {
            this.view = createView(this.host, doc.current());
            this.view._studioDoc = doc;
            this.view.dom.addEventListener('contextmenu', (e) => this.app.editorContextMenu(this, e));
            this.view.dom.addEventListener('mousedown', (e) => {
                if ((e.ctrlKey || e.metaKey) && e.button === 0) {
                    const pos = this.view.posAtCoords({ x: e.clientX, y: e.clientY });
                    if (pos !== null) {
                        e.preventDefault();
                        this.view.dispatch({ selection: { anchor: pos } });
                        this.app.commands.run('editor.definition');
                    }
                }
            }, true);
            doc.views.add(this.view);
            return true;
        }
        return false;
    }

    showDoc(doc) {
        const created = this.ensureView(doc);
        const view = this.view;
        const prev = view._studioDoc;
        if (!created && prev !== doc) {
            if (prev) {
                this.memory.set(prev.path, { scrollTop: view.scrollDOM.scrollTop, scrollLeft: view.scrollDOM.scrollLeft });
                prev.state = view.state;
                prev.views.delete(view);
            }
            view.setState(doc.current());
            view._studioDoc = doc;
            doc.views.add(view);
            const mem = this.memory.get(doc.path);
            if (mem) requestAnimationFrame(() => { view.scrollDOM.scrollTop = mem.scrollTop; view.scrollDOM.scrollLeft = mem.scrollLeft; });
        }
        this.host.hidden = false;
        this.specialHost.hidden = true;
        this.crumbsEl.hidden = false;
        this.runBtn.hidden = !doc.isTil;
    }

    add(entry, { activate = true, focus = true, index } = {}) {
        if (!this.tabs.some((x) => x.key === entry.key)) {
            if (index !== undefined) this.tabs.splice(index, 0, entry);
            else {
                const at = this.tabs.findIndex((x) => x.key === this.active);
                this.tabs.splice(at >= 0 ? at + 1 : this.tabs.length, 0, entry);
            }
        }
        if (activate) this.activate(entry.key, { focus });
        else this.renderTabs();
    }

    activate(key, { focus = true } = {}) {
        const entry = this.tabs.find((x) => x.key === key);
        if (!entry) return;
        const old = this.current();
        if (old && old.kind === 'special' && old.special.onHide && old !== entry) old.special.onHide();
        this.active = key;
        if (entry.kind === 'doc') {
            this.showDoc(entry.doc);
            if (focus) focusSoon(() => this.view && this.view.focus(), this.el);
        } else {
            this.host.hidden = true;
            this.crumbsEl.hidden = true;
            this.specialHost.hidden = false;
            this.runBtn.hidden = true;
            if (entry.special.el.parentNode !== this.specialHost) {
                clear(this.specialHost).appendChild(entry.special.el);
            }
            if (entry.special.onShow) entry.special.onShow();
            if (focus && entry.special.focus) focusSoon(() => entry.special.focus(), this.el);
        }
        this.renderTabs();
        this.updateEmpty();
        this.area.changed(this);
    }

    async close(key, { force = false } = {}) {
        const i = this.tabs.findIndex((x) => x.key === key);
        if (i < 0) return true;
        const entry = this.tabs[i];
        if (entry.kind === 'doc' && !force) {
            const doc = entry.doc;
            const elsewhere = this.area.groups.some((g) => g !== this && g.tabs.some((x) => x.key === key));
            if (doc.dirty && !elsewhere) {
                const choice = await this.app.askSave(doc);
                if (choice === 'cancel') return false;
                if (choice === 'save') { const ok = await this.app.saveDoc(doc); if (!ok) return false; }
            }
        }
        this.tabs.splice(i, 1);
        if (entry.kind === 'doc') {
            const doc = entry.doc;
            if (this.view && this.view._studioDoc === doc) {
                doc.state = this.view.state;
                doc.views.delete(this.view);
                this.view._studioDoc = null;
            }
            this.memory.delete(doc.path);
            const elsewhere = this.area.groups.some((g) => g !== this && g.tabs.some((x) => x.key === key));
            if (!elsewhere) this.app.docs.close(doc);
        } else if (entry.special.dispose) entry.special.dispose();
        if (this.active === key) {
            const next = this.tabs[Math.min(i, this.tabs.length - 1)];
            this.active = null;
            if (next) this.activate(next.key);
            else {
                if (this.view) { this.view.destroy(); this.view = null; }
                clear(this.host);
                this.host.hidden = true;
                this.specialHost.hidden = true;
                this.crumbsEl.hidden = true;
            }
        }
        this.renderTabs();
        this.updateEmpty();
        this.area.changed(this);
        if (!this.tabs.length && this.area.groups.length > 1) this.area.removeGroup(this);
        return true;
    }

    updateEmpty() {
        const empty = !this.tabs.length;
        this.emptyEl.hidden = !empty;
        if (empty) { this.host.hidden = true; this.crumbsEl.hidden = true; this.specialHost.hidden = true; }
        this.actionsEl.hidden = empty;
    }

    tabTitle(entry) {
        if (entry.kind === 'doc') return entry.doc.name;
        return entry.special.title();
    }

    renderTabs() {
        clear(this.tabsEl);
        const names = new Map();
        for (const e of this.tabs) { const n = this.tabTitle(e); names.set(n, (names.get(n) || 0) + 1); }
        for (const entry of this.tabs) {
            const sel = entry.key === this.active;
            const title = this.tabTitle(entry);
            const doc = entry.kind === 'doc' ? entry.doc : null;
            const diag = doc ? this.app.diagnosticSummary(doc.uri) : null;
            const nameEl = h('span', { class: ['tab-name', diag && diag.errors && 'has-errors', diag && !diag.errors && diag.warnings && 'has-warnings'] }, title);
            const tabEl = h('div', {
                class: ['tab', doc && doc.dirty && 'dirty', doc && doc.readOnly && 'readonly'],
                role: 'tab', 'aria-selected': String(sel), tabindex: sel ? '0' : '-1', dataset: { key: entry.key },
                title: doc ? (doc.kind === 'file' ? doc.path : doc.name) : title,
            },
            icon(doc ? fileIcon(doc.name) : entry.special.iconName || 'file', { className: doc && doc.isTil ? 'til-icon' : '' }),
            nameEl,
            doc && doc.readOnly ? icon('lock', { className: 'tab-lock', title: t('file.readOnly') }) : null,
            names.get(title) > 1 && doc && doc.kind === 'file' ? h('span', { class: 'tab-desc' }, dirname(doc.path) || '/') : null);
            const close = h('button', { class: 'tab-close', type: 'button', tabindex: '-1', 'aria-label': t('editor.closeTab', { name: title }) }, icon('close'));
            close.addEventListener('mousedown', (e) => e.stopPropagation());
            close.addEventListener('click', (e) => { e.stopPropagation(); this.close(entry.key); });
            tabEl.appendChild(close);
            tabEl.addEventListener('mousedown', (e) => {
                if (e.button === 1) { e.preventDefault(); this.close(entry.key); return; }
                if (e.button === 0) this.activate(entry.key, { focus: false });
            });
            tabEl.addEventListener('click', () => { if (entry.kind === 'doc' && this.view) this.view.focus(); });
            tabEl.addEventListener('contextmenu', (e) => { e.preventDefault(); this.tabMenu(entry, e); });
            this.tabsEl.appendChild(tabEl);
            if (sel) requestAnimationFrame(() => tabEl.scrollIntoView({ block: 'nearest', inline: 'nearest' }));
        }
    }

    tabKeys(e) {
        const tabs = [...this.tabsEl.querySelectorAll('.tab')];
        const i = tabs.indexOf(document.activeElement);
        if (i < 0) return;
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
            e.preventDefault();
            const j = (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
            this.activate(tabs[j].dataset.key, { focus: false });
            this.tabsEl.querySelector(`.tab[data-key="${CSS.escape(tabs[j].dataset.key)}"]`).focus();
        } else if (e.key === 'Delete') {
            e.preventDefault();
            this.close(tabs[i].dataset.key);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            this.activate(tabs[i].dataset.key, { focus: true });
        }
    }

    tabMenu(entry, e) {
        const i = this.tabs.indexOf(entry);
        const doc = entry.kind === 'doc' ? entry.doc : null;
        openMenu([
            { label: t('editor.close'), keys: 'Mod+W', run: () => this.close(entry.key) },
            { label: t('editor.closeOthers'), run: () => this.closeMany(this.tabs.filter((x) => x !== entry)) },
            { label: t('editor.closeRight'), disabled: i === this.tabs.length - 1, run: () => this.closeMany(this.tabs.slice(i + 1)) },
            { label: t('editor.closeSaved'), run: () => this.closeMany(this.tabs.filter((x) => x.kind !== 'doc' || !x.doc.dirty)) },
            { label: t('editor.closeAll'), run: () => this.closeMany(this.tabs.slice()) },
            { separator: true },
            { label: t('editor.splitRight'), icon: 'split', run: () => { this.activate(entry.key); this.area.splitRight(this); } },
            doc && doc.kind === 'file' ? { label: t('explorer.copyPath'), icon: 'copy', run: () => this.app.copyText(doc.path) } : null,
            doc && doc.kind === 'file' ? { label: t('explorer.reveal'), icon: 'files', run: () => this.app.revealInExplorer(doc.path) } : null,
        ], { x: e.clientX, y: e.clientY });
    }

    async closeMany(entries) {
        for (const e of entries.slice()) {
            const ok = await this.close(e.key);
            if (!ok) break;
        }
    }

    focus() {
        const c = this.current();
        if (!c) return;
        if (c.kind === 'doc' && this.view) this.view.focus();
        else if (c.kind === 'special' && c.special.focus) c.special.focus();
    }
}

export class EditorArea {
    constructor(app, container) {
        this.app = app;
        this.container = container;
        this.groups = [];
        this.focused = null;
        this.addGroup();
    }

    addGroup() {
        const g = new EditorGroup(this, this.groups.length);
        this.groups.push(g);
        this.container.appendChild(g.el);
        if (!this.focused) this.focused = g;
        return g;
    }

    removeGroup(g) {
        if (this.groups.length < 2) return;
        if (g.view) { const d = g.view._studioDoc; if (d) d.views.delete(g.view); g.view.destroy(); g.view = null; }
        g.el.remove();
        this.groups = this.groups.filter((x) => x !== g);
        this.groups.forEach((x, i) => { x.index = i; x.el.dataset.group = String(i); });
        this.setFocused(this.groups[0]);
        this.app.editorsResized();
        this.app.saveLayout();
    }

    setFocused(g) {
        if (this.focused === g) return;
        this.focused = g;
        for (const x of this.groups) x.el.classList.toggle('focused', x === g);
        this.app.onActiveEditorChanged();
    }

    group() { return this.focused || this.groups[0]; }

    activeDoc() { return this.group().activeDoc(); }

    activeView() {
        const g = this.group();
        const d = g.activeDoc();
        return d && g.view && g.view._studioDoc === d ? g.view : null;
    }

    /* every open document, in tab order */
    openDocs() {
        const out = [];
        for (const g of this.groups) for (const e of g.tabs) if (e.kind === 'doc' && !out.includes(e.doc)) out.push(e.doc);
        return out;
    }

    findTab(key) {
        for (const g of this.groups) if (g.tabs.some((x) => x.key === key)) return g;
        return null;
    }

    /* show a document; opts: {group, line, col, selection: {from, to}, focus, flash} */
    show(doc, opts = {}) {
        const g = opts.group || this.findTab(doc.path) || this.group();
        g.add({ key: doc.path, kind: 'doc', doc }, { focus: opts.focus !== false });
        this.setFocused(g);
        const view = g.view;
        if (view && (opts.line || opts.selection)) {
            const state = view.state;
            let anchor, head;
            if (opts.selection) { anchor = Math.min(opts.selection.from, state.doc.length); head = Math.min(opts.selection.to, state.doc.length); }
            else {
                const line = state.doc.line(Math.max(1, Math.min(state.doc.lines, opts.line)));
                anchor = head = Math.min(line.to, line.from + Math.max(0, (opts.col || 1) - 1));
            }
            const effects = [EditorView.scrollIntoView(EditorSelection.range(anchor, head), { y: 'center' })];
            if (opts.flash !== false) effects.push(flashEffect.of({ line: state.doc.lineAt(head).number, seq: ++flashSeq }));
            view.dispatch({ selection: EditorSelection.single(anchor, head), effects });
        }
        return g;
    }

    showSpecial(key, special, opts = {}) {
        const g = opts.group || this.findTab(key) || this.group();
        g.add({ key, kind: 'special', special }, { focus: opts.focus !== false });
        this.setFocused(g);
        return g;
    }

    splitRight(from) {
        const src = from || this.group();
        const cur = src.current();
        let target = this.groups.find((g) => g !== src);
        if (!target) {
            if (this.groups.length >= 2) return;
            target = this.addGroup();
        }
        if (cur && cur.kind === 'doc') this.show(cur.doc, { group: target });
        this.setFocused(target);
        this.app.editorsResized();
        this.app.saveLayout();
    }

    changed(g) {
        if (g === this.focused) this.app.onActiveEditorChanged();
        this.app.saveLayout();
    }

    renderAllTabs() {
        for (const g of this.groups) g.renderTabs();
    }

    async closeAll() {
        for (const g of this.groups.slice()) {
            for (const e of g.tabs.slice()) {
                if (!(await g.close(e.key))) return false;
            }
        }
        return true;
    }

    layoutState() {
        return {
            groups: this.groups.map((g) => ({
                tabs: g.tabs.filter((e) => e.kind === 'doc' ? e.doc.kind === 'file' : e.special.persist).map((e) => e.kind === 'doc' ? { path: e.doc.path, sel: selOf(g, e.doc) } : { special: e.key }),
                active: g.active,
            })),
            focused: this.groups.indexOf(this.focused),
        };
    }
}

function selOf(g, doc) {
    const st = g.view && g.view._studioDoc === doc ? g.view.state : doc.state;
    const r = st.selection.main;
    const line = st.doc.lineAt(r.head);
    return { line: line.number, col: r.head - line.from + 1 };
}
