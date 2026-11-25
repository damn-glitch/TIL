// Language features that act on the editor: go to definition (F12),
// references (Shift+F12, a peek list), rename (F2, inline box), format
// (Shift+Alt+F), code actions (Ctrl+.), and workspace edits across files.
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, tn } from '../core/i18n.js';
import { basename } from '../core/paths.js';
import { toast, openMenu } from '../ui/widgets.js';
import { toPosition, toOffset, rangeToOffsets, textEditsToChanges, diagnosticsField } from './cm_lsp.js';
import { highlightInto } from '../editor/til_lang.js';

export class LspActions {
    constructor(app) { this.app = app; this.peekEl = null; }

    ready(cap) {
        const c = this.app.lsp;
        return c && c.ready && (!cap || c.has(cap));
    }

    target() {
        const view = this.app.editors.activeView();
        const doc = this.app.editors.activeDoc();
        if (!view || !doc || !doc.isTil) return null;
        return { view, doc, pos: view.state.selection.main.head };
    }

    async definition() {
        const tg = this.target();
        if (!tg || !this.ready('definitionProvider')) return;
        let res;
        try {
            res = await this.app.lsp.request('textDocument/definition', { textDocument: { uri: tg.doc.uri }, position: toPosition(tg.view.state.doc, tg.pos) });
        } catch (e) { res = null; }
        const locs = normalizeLocations(res);
        if (!locs.length) { this.app.statusbar.flash(t('lsp.noDefinition')); return; }
        if (locs.length > 1) { this.showPeek(tg, locs, t('lsp.definitions')); return; }
        await this.goto(locs[0]);
    }

    async goto(loc, opts = {}) {
        const path = this.app.pathOf(loc.uri);
        if (path === null) { toast(t('lsp.outsideProject', { uri: loc.uri }), { kind: 'info' }); return; }
        await this.app.openLocation(path, loc.range, opts);
    }

    async references() {
        const tg = this.target();
        if (!tg || !this.ready('referencesProvider')) return;
        let res;
        try {
            res = await this.app.lsp.request('textDocument/references', {
                textDocument: { uri: tg.doc.uri }, position: toPosition(tg.view.state.doc, tg.pos), context: { includeDeclaration: true },
            });
        } catch (e) { res = null; }
        const locs = normalizeLocations(res);
        if (!locs.length) { this.app.statusbar.flash(t('lsp.noReferences')); return; }
        const word = tg.view.state.wordAt(tg.pos);
        this.showPeek(tg, locs, word ? tg.view.state.sliceDoc(word.from, word.to) : '');
    }

    closePeek() {
        if (this.peekEl) { this.peekEl.remove(); this.peekEl = null; }
    }

    async showPeek(tg, locs, title) {
        this.closePeek();
        const items = [];
        const byFile = new Map();
        for (const l of locs) {
            if (!byFile.has(l.uri)) byFile.set(l.uri, []);
            byFile.get(l.uri).push(l);
        }
        const list = h('div', { class: 'peek-list', role: 'listbox', tabindex: '-1' });
        for (const [uri, ls] of byFile) {
            const path = this.app.pathOf(uri);
            const text = await this.textOf(uri, path);
            list.appendChild(h('div', { class: 'peek-file' }, path || uri));
            ls.sort((a, b) => a.range.start.line - b.range.start.line || a.range.start.character - b.range.start.character);
            for (const l of ls) {
                const lineText = text ? text.split('\n')[l.range.start.line] || '' : '';
                const code = h('code');
                const s = l.range.start.character, e = l.range.end.line === l.range.start.line ? l.range.end.character : lineText.length;
                const lead = lineText.length - lineText.trimStart().length;
                code.append(lineText.slice(lead, s), h('mark', {}, lineText.slice(s, e)), lineText.slice(e));
                const row = h('div', { class: 'peek-item', role: 'option', 'aria-selected': 'false' }, h('span', { class: 'ln' }, String(l.range.start.line + 1)), code);
                row.addEventListener('click', () => { this.closePeek(); this.goto(l); });
                list.appendChild(row);
                items.push({ row, loc: l });
            }
        }
        const close = h('button', { class: 'icon-btn small', type: 'button', 'aria-label': t('dialog.close') }, icon('close'));
        close.addEventListener('click', () => { this.closePeek(); tg.view.focus(); });
        const el = h('div', { class: 'peek', role: 'dialog', 'aria-label': t('lsp.references') },
            h('div', { class: 'peek-head' }, icon('link'), h('b', {}, title), h('span', {}, tn('lsp.refCount', locs.length)), h('span', { class: 'spacer' }), close), list);
        document.body.appendChild(el);
        this.peekEl = el;
        const coords = tg.view.coordsAtPos(tg.pos) || { left: 200, bottom: 200, top: 200 };
        const r = tg.view.dom.getBoundingClientRect();
        el.style.left = `${Math.max(r.left + 40, Math.min(coords.left - 40, window.innerWidth - el.offsetWidth - 16))}px`;
        let top = coords.bottom + 6;
        if (top + el.offsetHeight > window.innerHeight - 30) top = Math.max(50, coords.top - el.offsetHeight - 6);
        el.style.top = `${top}px`;
        let index = 0;
        const mark = () => items.forEach((it, i) => { it.row.setAttribute('aria-selected', String(i === index)); if (i === index) it.row.scrollIntoView({ block: 'nearest' }); });
        mark();
        list.focus();
        el.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown') { e.preventDefault(); index = Math.min(items.length - 1, index + 1); mark(); }
            else if (e.key === 'ArrowUp') { e.preventDefault(); index = Math.max(0, index - 1); mark(); }
            else if (e.key === 'Enter') { e.preventDefault(); const it = items[index]; this.closePeek(); if (it) this.goto(it.loc); }
            else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); this.closePeek(); tg.view.focus(); }
        });
        const outside = (e) => { if (this.peekEl && !this.peekEl.contains(e.target)) { this.closePeek(); document.removeEventListener('mousedown', outside, true); } };
        setTimeout(() => document.addEventListener('mousedown', outside, true), 0);
    }

    async textOf(uri, path) {
        for (const d of this.app.docs.all()) if (d.uri === uri) return d.current().doc.toString();
        if (path === null) return '';
        try { return (await this.app.backend.readFile(path)).text; } catch (e) { return ''; }
    }

    async rename() {
        const tg = this.target();
        if (!tg || tg.doc.readOnly || !this.ready('renameProvider')) return;
        const { view, doc } = tg;
        let range = null, placeholder = '';
        const word = view.state.wordAt(tg.pos);
        const prepare = this.app.lsp.capabilities.renameProvider && this.app.lsp.capabilities.renameProvider.prepareProvider;
        if (prepare) {
            let res;
            try {
                res = await this.app.lsp.request('textDocument/prepareRename', { textDocument: { uri: doc.uri }, position: toPosition(view.state.doc, tg.pos) });
            } catch (e) {
                toast(e.message || t('lsp.cannotRename'), { kind: 'warning' });
                return;
            }
            if (!res) { this.app.statusbar.flash(t('lsp.cannotRename')); return; }
            if (res.range) { range = rangeToOffsets(view.state.doc, res.range); placeholder = res.placeholder || view.state.sliceDoc(range.from, range.to); }
            else if (res.start) { range = rangeToOffsets(view.state.doc, res); placeholder = view.state.sliceDoc(range.from, range.to); }
            else if (res.defaultBehavior && word) { range = word; placeholder = view.state.sliceDoc(word.from, word.to); }
        } else if (word) { range = word; placeholder = view.state.sliceDoc(word.from, word.to); }
        if (!range) return;
        const coords = view.coordsAtPos(range.from);
        const input = h('input', { class: 'input', type: 'text', value: placeholder, spellcheck: 'false', 'aria-label': t('cmd.rename') });
        const err = h('div', { class: 'error', role: 'alert' });
        const box = h('div', { class: 'rename-box' }, input, h('div', { class: 'hint' }, t('lsp.renameHint')), err);
        document.body.appendChild(box);
        box.style.left = `${coords ? coords.left - 6 : 100}px`;
        box.style.top = `${coords ? coords.bottom + 4 : 100}px`;
        input.focus();
        input.select();
        let done = false;
        const finish = async (commit) => {
            if (done) return;
            const name = input.value.trim();
            if (commit) {
                if (!/^[\p{L}_][\p{L}\p{N}_]*$/u.test(name)) { err.textContent = t('lsp.badName'); return; }
                if (name === placeholder) { commit = false; }
            }
            done = true;
            box.remove();
            view.focus();
            if (!commit) return;
            let edit;
            try {
                edit = await this.app.lsp.request('textDocument/rename', { textDocument: { uri: doc.uri }, position: toPosition(view.state.doc, range.from), newName: name });
            } catch (e) {
                toast(e.message || t('lsp.cannotRename'), { kind: 'warning' });
                return;
            }
            if (!edit) { this.app.statusbar.flash(t('lsp.cannotRename')); return; }
            const n = await this.applyWorkspaceEdit(edit);
            this.app.statusbar.flash(tn('lsp.renamed', n, { name }));
        };
        input.addEventListener('keydown', (e) => {
            e.stopPropagation();
            if (e.key === 'Enter') { e.preventDefault(); finish(true); }
            if (e.key === 'Escape') { e.preventDefault(); finish(false); }
        });
        input.addEventListener('blur', () => setTimeout(() => finish(false), 120));
    }

    /* apply a WorkspaceEdit: open documents through their editors (undoable), other files on disk */
    async applyWorkspaceEdit(edit) {
        const changes = new Map();
        if (edit && edit.changes) for (const [uri, edits] of Object.entries(edit.changes)) changes.set(uri, edits);
        if (edit && Array.isArray(edit.documentChanges)) {
            for (const dc of edit.documentChanges) {
                if (dc.textDocument && Array.isArray(dc.edits)) changes.set(dc.textDocument.uri, (changes.get(dc.textDocument.uri) || []).concat(dc.edits));
            }
        }
        let count = 0;
        for (const [uri, edits] of changes) {
            count += edits.length;
            const doc = this.app.docs.all().find((d) => d.uri === uri);
            if (doc && doc.readOnly) { this.app.statusbar.flash(t('file.readOnlyHint')); continue; }
            if (doc) {
                const ch = textEditsToChanges(doc.current().doc, edits);
                if (ch.length) doc.dispatch({ changes: ch, userEvent: 'input.rename' });
                continue;
            }
            const path = this.app.pathOf(uri);
            if (path === null || path.startsWith('@')) continue;
            try {
                const file = await this.app.backend.readFile(path);
                const { Text } = await import('@codemirror/state');
                const text = Text.of(file.text.replace(/\r\n?/g, '\n').split('\n'));
                const ch = textEditsToChanges(text, edits);
                let out = text.toString();
                for (const c of ch.slice().reverse()) out = out.slice(0, c.from) + c.insert + out.slice(c.to);
                await this.app.backend.writeFile(path, out, file.mtime);
            } catch (e) {
                toast(t('file.cannotSave', { name: basename(path) }), { kind: 'error', detail: e.message });
            }
        }
        return count;
    }

    async format(docArg) {
        const doc = docArg || this.app.editors.activeDoc();
        if (!doc || !doc.isTil || doc.readOnly || !this.ready('documentFormattingProvider')) return false;
        let edits;
        try {
            edits = await this.app.lsp.request('textDocument/formatting', { textDocument: { uri: doc.uri }, options: { tabSize: 4, insertSpaces: true, trimTrailingWhitespace: true, insertFinalNewline: true } });
        } catch (e) { edits = null; }
        if (!edits || !edits.length) { if (!docArg) this.app.statusbar.flash(t('lsp.formatNoChanges')); return false; }
        const ch = textEditsToChanges(doc.current().doc, edits);
        doc.dispatch({ changes: ch, userEvent: 'input.format' });
        if (!docArg) this.app.statusbar.flash(t('lsp.formatted'));
        return true;
    }

    async codeActions() {
        const tg = this.target();
        if (!tg || !this.ready('codeActionProvider')) return;
        if (tg.doc.readOnly) { this.app.statusbar.flash(t('file.readOnlyHint')); return; }
        const { view, doc } = tg;
        // diagnostics of an older text carry old ranges: a fix built from them
        // would edit the wrong characters. Wait (briefly) for the current ones;
        // if they do not come, let the server find the problems itself.
        this.app.lsp.flush(doc.uri);
        const fresh = await this.app.freshDiagnostics(doc);
        const sel = view.state.selection.main;
        const line = view.state.doc.lineAt(sel.head);
        const from = sel.empty ? line.from : sel.from, to = sel.empty ? line.to : sel.to;
        const diags = fresh ? (view.state.field(diagnosticsField, false) || []).filter((d) => d.to >= from && d.from <= to).map((d) => d.raw) : [];
        let actions;
        try {
            actions = await this.app.lsp.request('textDocument/codeAction', {
                textDocument: { uri: doc.uri },
                range: { start: toPosition(view.state.doc, from), end: toPosition(view.state.doc, to) },
                context: { diagnostics: diags, triggerKind: 1 },
            });
        } catch (e) { actions = null; }
        actions = (actions || []).filter((a) => a && a.title && !a.disabled);
        const coords = view.coordsAtPos(sel.head) || { left: 200, bottom: 200 };
        if (!actions.length) {
            openMenu([{ label: t('lsp.noActions'), disabled: true }], { x: coords.left, y: coords.bottom });
            return;
        }
        const quick = actions.filter((a) => !a.kind || /^quickfix/.test(a.kind));
        const other = actions.filter((a) => a.kind && !/^quickfix/.test(a.kind));
        const item = (a) => ({ label: a.title, icon: /^quickfix/.test(a.kind || 'quickfix') ? 'lightbulb' : 'sparkle', run: () => this.runAction(a) });
        openMenu([
            ...quick.map(item),
            quick.length && other.length ? { separator: true } : null,
            ...other.map(item),
        ], { x: coords.left, y: coords.bottom + 2 }, { label: t('cmd.codeActions') });
    }

    async runAction(a) {
        let act = a;
        if (!a.edit && !a.command && a.data !== undefined && this.app.lsp.capabilities.codeActionProvider && this.app.lsp.capabilities.codeActionProvider.resolveProvider) {
            try { act = await this.app.lsp.request('codeAction/resolve', a); } catch (e) { act = a; }
        }
        if (act.edit) await this.applyWorkspaceEdit(act.edit);
        const cmd = act.command && typeof act.command === 'object' ? act.command : (typeof act.command === 'string' ? act : null);
        if (cmd && cmd.command) {
            if (/til\.(run|debug|prove)$/.test(cmd.command)) { await this.app.executeLensCommand(cmd); return; }
            if (/til\.translate$/.test(cmd.command)) {
                const arg = (cmd.arguments || [])[0] || {};
                const lang = arg.to || arg.lang || (typeof arg === 'string' ? arg : null);
                const path = arg.uri ? this.app.pathOf(arg.uri) : null;
                if (lang) { await this.app.translateInPlace(path, lang); return; }
            }
            try { await this.app.lsp.request('workspace/executeCommand', { command: cmd.command, arguments: cmd.arguments || [] }); } catch (e) {
                toast(t('command.failed', { name: cmd.title || cmd.command }), { kind: 'error', detail: e.message });
            }
        }
    }

    /* symbols for the "@" palette */
    async symbolItems() {
        const doc = this.app.editors.activeDoc();
        if (!doc || !doc.isTil) return [];
        const { documentSymbols, symbolIcon } = await import('../ui/statusbar.js');
        const syms = await documentSymbols(this.app, doc);
        const out = [];
        const walk = (list, prefix) => {
            for (const s of list || []) {
                const r = s.selectionRange || s.range || (s.location && s.location.range);
                out.push({ label: s.name, description: prefix ? `${prefix} · ${s.detail || ''}` : s.detail || '', icon: symbolIcon(s.kind), iconClass: s.kind === 12 || s.kind === 6 ? 'fn' : 'type', value: { doc, range: r } });
                walk(s.children, s.name);
            }
        };
        walk(syms, '');
        return out;
    }

    async workspaceSymbolItems(query) {
        if (!this.ready('workspaceSymbolProvider')) return [];
        let list;
        try { list = await this.app.lsp.request('workspace/symbol', { query }); } catch (e) { list = null; }
        const { symbolIcon } = await import('../ui/statusbar.js');
        const items = (list || []).map((s) => ({
            label: s.name, description: s.location ? this.app.pathOf(s.location.uri) || s.location.uri : '', icon: symbolIcon(s.kind),
            value: { loc: s.location },
        }));
        items.noFilter = true;
        return items;
    }
}

export function normalizeLocations(res) {
    if (!res) return [];
    const arr = Array.isArray(res) ? res : [res];
    return arr.map((l) => (l.targetUri ? { uri: l.targetUri, range: l.targetSelectionRange || l.targetRange } : l)).filter((l) => l && l.uri && l.range);
}

export { toOffset, clear, highlightInto };
