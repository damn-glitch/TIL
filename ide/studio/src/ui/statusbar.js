// The status bar (connection, problems, run state | cursor, indentation,
// keyword languages, UI language, backend, version, notifications) and the
// breadcrumbs of the editor groups.
import { h, clear } from '../core/dom.js';
import { icon, fileIcon } from '../core/icons.js';
import { t, getLocale, LOCALE_NAMES } from '../core/i18n.js';
import { languagesOf } from '../editor/til_lang.js';
import { notifications, tip } from './widgets.js';

export class StatusBar {
    constructor(app) {
        this.app = app;
        const L = app.layout;
        this.conn = L.statusItem({ side: 'left', className: 'sb-conn', tipKey: 'status.connectionTip' });
        this.problems = L.statusItem({ side: 'left', className: 'sb-problems', onClick: () => app.layout.showPanel('problems'), tipKey: 'status.problemsTip' });
        this.runState = L.statusItem({ side: 'left', className: 'sb-run' });
        this.debugState = L.statusItem({ side: 'left', className: 'sb-debug', onClick: () => app.layout.showSide('debug') });
        this.message = L.statusItem({ side: 'left', className: 'sb-msg' });
        this.cursor = L.statusItem({ side: 'right', className: 'sb-cursor', onClick: () => app.commands.run('editor.gotoLine'), tipKey: 'status.gotoLine' });
        this.indent = L.statusItem({ side: 'right', className: 'sb-indent', tipKey: 'status.indentTip' });
        this.kwlang = L.statusItem({ side: 'right', className: 'sb-kwlang', onClick: (e) => this.kwMenu(e), tipKey: 'status.kwTip' });
        this.backend = L.statusItem({ side: 'right', className: 'sb-backend', onClick: (e) => app.pickBackend(e.currentTarget.getBoundingClientRect()), tipKey: 'titlebar.backendTip' });
        this.uiLang = L.statusItem({ side: 'right', className: 'sb-uilang', onClick: (e) => app.pickLocale(e.currentTarget.getBoundingClientRect()), tipKey: 'status.uiLangTip' });
        this.version = L.statusItem({ side: 'right', className: 'sb-version', onClick: () => app.commands.run('help.about'), tipKey: 'status.versionTip' });
        this.bell = L.statusItem({ side: 'right', className: 'sb-bell', onClick: (e) => this.showNotifications(e.currentTarget), tipKey: 'notifications.title' });
        this.runState.hidden = true;
        this.debugState.hidden = true;
        this.connected = true;
        this.setConnection(true);
    }

    setConnection(on) {
        this.connected = on;
        const web = this.app.web;
        clear(this.conn).append(h('span', { class: ['dot', web ? 'web' : on ? '' : 'off'] }),
            h('span', {}, web ? t('status.web') : on ? t('status.desktop') : t('status.reconnecting')));
    }

    update() {
        const { errors, warnings } = this.app.diagnosticTotals();
        clear(this.problems).append(icon('error', { className: 'icon-error' }), h('span', { class: 'n' }, String(errors)),
            icon('warning', { className: 'icon-warning' }), h('span', { class: 'n' }, String(warnings)));
        const doc = this.app.editors ? this.app.editors.activeDoc() : null;
        this.indent.textContent = t('status.spaces', { n: 4 });
        this.indent.hidden = !doc;
        if (doc && doc.isTil) {
            const langs = languagesOf(doc.current().doc.sliceString(0, 4000));
            this.kwlang.textContent = langs.map((l) => t(`kwlang.short.${l}`)).join(' + ');
            this.kwlang.hidden = false;
        } else this.kwlang.hidden = true;
        const names = { native: t('backend.native'), vm: t('backend.vm'), cvm: t('backend.cvm') };
        clear(this.backend).append(icon('layers'), h('span', {}, names[this.app.effectiveBackend()]));
        clear(this.uiLang).append(icon('globe'), h('span', {}, LOCALE_NAMES[getLocale()]));
        this.version.textContent = `TIL ${this.app.info.version || this.app.tables.version}`;
        this.updateBell();
        this.setConnection(this.connected);
    }

    updateBell() {
        const n = notifications.unread;
        clear(this.bell).append(icon('bell'));
        if (n) this.bell.appendChild(h('span', { class: 'badge accent' }, String(Math.min(n, 99))));
    }

    updateCursor(view) {
        if (!view) { this.cursor.hidden = true; return; }
        const sel = view.state.selection.main;
        const line = view.state.doc.lineAt(sel.head);
        const n = view.state.selection.ranges.length;
        const selLen = sel.to - sel.from;
        let text = t('status.cursor', { line: line.number, col: sel.head - line.from + 1 });
        if (n > 1) text += ` · ${t('status.cursors', { n })}`;
        else if (selLen) text += ` · ${t('status.selected', { n: selLen })}`;
        this.cursor.textContent = text;
        this.cursor.hidden = false;
    }

    setRun(state) {
        // state: null | {label, kind: 'run'|'compile'|'wait'}
        if (!state) { this.runState.hidden = true; this.app.layout.root.dataset.running = 'false'; return; }
        this.runState.hidden = false;
        this.app.layout.root.dataset.running = 'true';
        clear(this.runState).append(state.kind === 'wait' ? icon('keyboard') : h('span', { class: 'spinner' }), h('span', {}, state.label));
    }

    setDebug(label) {
        if (!label) { this.debugState.hidden = true; this.app.layout.root.dataset.debugging = 'false'; return; }
        this.debugState.hidden = false;
        this.app.layout.root.dataset.debugging = 'true';
        clear(this.debugState).append(icon('bug'), h('span', {}, label));
    }

    flash(text) {
        this.message.textContent = text;
        this.message.hidden = false;
        clearTimeout(this._flash);
        this._flash = setTimeout(() => { this.message.hidden = true; }, 2600);
    }

    kwMenu(e) {
        const doc = this.app.editors.activeDoc();
        if (!doc || !doc.isTil) return;
        this.app.commands.run('translate.pick');
        void e;
    }

    showNotifications(anchor) {
        const old = document.querySelector('.notif-pop');
        if (old) { old.remove(); return; }
        notifications.unread = 0;
        this.updateBell();
        const list = notifications.list;
        const pop = h('div', { class: 'notif-pop', role: 'dialog', 'aria-label': t('notifications.title') },
            h('div', { class: 'np-head' }, h('span', {}, t('notifications.title')), h('span', { class: 'spacer' }),
                h('button', { class: 'btn ghost small', type: 'button', on: { click: () => { notifications.list.length = 0; pop.remove(); } } }, t('notifications.clear'))));
        if (!list.length) pop.appendChild(h('div', { class: 'np-empty' }, t('notifications.none')));
        for (const n of list) {
            const kindIcon = { info: 'info', success: 'checkCircle', warning: 'warning', error: 'error' }[n.kind] || 'info';
            pop.appendChild(h('div', { class: 'np-item' }, icon(kindIcon, { className: `sev-${n.kind === 'success' ? 'info' : n.kind}` }),
                h('div', {}, n.message, n.detail ? h('div', { class: 'muted' }, n.detail.slice(0, 200)) : null),
                h('span', { class: 'np-time' }, new Date(n.time).toLocaleTimeString(getLocale(), { hour: '2-digit', minute: '2-digit' }))));
        }
        document.body.appendChild(pop);
        const r = anchor.getBoundingClientRect();
        pop.style.right = `${Math.max(8, window.innerWidth - r.right)}px`;
        pop.style.bottom = `${window.innerHeight - r.top + 6}px`;
        const close = (ev) => { if (!pop.contains(ev.target) && ev.target !== anchor && !anchor.contains(ev.target)) { pop.remove(); document.removeEventListener('mousedown', close, true); } };
        setTimeout(() => document.addEventListener('mousedown', close, true), 0);
    }
}

/* ── breadcrumbs: folders › file › symbol at the cursor ── */
const symbolCache = new Map();     // uri -> {version, symbols}

export async function documentSymbols(app, doc) {
    if (!doc || !doc.isTil || !app.lsp || !app.lsp.ready || !app.lsp.has('documentSymbolProvider')) return [];
    const version = app.lsp.version(doc.uri) + ':' + doc.current().doc.length;
    const cached = symbolCache.get(doc.uri);
    if (cached && cached.version === version) return cached.symbols;
    let symbols = [];
    try { symbols = await app.lsp.request('textDocument/documentSymbol', { textDocument: { uri: doc.uri } }) || []; } catch (e) { symbols = []; }
    symbolCache.set(doc.uri, { version, symbols });
    return symbols;
}

function symbolPath(symbols, line) {
    const out = [];
    let level = symbols;
    while (level && level.length) {
        const s = level.find((x) => {
            const r = x.range || (x.location && x.location.range);
            return r && r.start.line <= line && line <= r.end.line;
        });
        if (!s) break;
        out.push(s);
        level = s.children;
    }
    return out;
}

const SYMBOL_ICON = { 12: 'symbolFunction', 6: 'symbolFunction', 9: 'symbolFunction', 23: 'symbolStruct', 5: 'symbolStruct', 10: 'symbolEnum', 22: 'symbolEnum', 14: 'symbolConst', 13: 'variable', 8: 'symbolField', 7: 'symbolField', 11: 'symbolStruct' };
export function symbolIcon(kind) { return SYMBOL_ICON[kind] || 'symbolKeyword'; }

export async function renderBreadcrumbs(app, group, doc) {
    const el = group.crumbsEl;
    if (!doc) { clear(el); return; }
    const parts = doc.kind === 'file' ? doc.path.split('/') : [doc.name];
    const frag = document.createDocumentFragment();
    parts.forEach((p, i) => {
        if (i) frag.appendChild(icon('chevronRight', { className: 'crumb-sep' }));
        const last = i === parts.length - 1;
        const b = h('button', { class: 'crumb', type: 'button' }, icon(last ? fileIcon(p) : 'folder'), p);
        const dirPath = parts.slice(0, i + 1).join('/');
        b.addEventListener('click', () => (last ? app.commands.run('workbench.gotoSymbol') : app.revealInExplorer(dirPath)));
        frag.appendChild(b);
    });
    const view = group.view;
    if (view && view._studioDoc === doc && doc.isTil) {
        const line = view.state.doc.lineAt(view.state.selection.main.head).number - 1;
        const syms = symbolPath(await documentSymbols(app, doc), line);
        for (const s of syms) {
            frag.appendChild(icon('chevronRight', { className: 'crumb-sep' }));
            const b = h('button', { class: 'crumb symbol', type: 'button' }, icon(symbolIcon(s.kind)), s.name);
            b.addEventListener('click', () => app.commands.run('workbench.gotoSymbol'));
            frag.appendChild(b);
        }
    }
    if (doc.readOnly && doc.kind === 'file') {
        // a TIL example: read-only; a copy in the project can be changed
        const copy = h('button', { class: 'crumb-readonly', type: 'button' }, icon('lock'), h('span', {}, t('file.readOnly')), h('span', { class: 'crumb-action' }, t('file.saveCopy')));
        tip(copy, 'file.readOnlyHint');
        copy.addEventListener('click', () => app.saveAs(doc));
        frag.appendChild(copy);
    }
    clear(el).appendChild(frag);
}
