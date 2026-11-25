// Open documents: one EditorState per file (shown by the views of zero, one
// or two editor groups), saved text, modification time, LSP sync, and
// external changes on disk.
import { Annotation, Transaction } from '@codemirror/state';
import { EditorView } from '@codemirror/view';
import { createEditorState, settingsEffects } from './editor.js';
import { bpEffect, breakpointList, breakpointField } from './gutters.js';
import { diagnosticsSpec } from '../lsp/cm_lsp.js';
import { basename, isTil } from '../core/paths.js';
import { Emitter } from '../core/events.js';

export const syncAnnotation = Annotation.define();

let untitledSeq = 0;

export class Doc {
    constructor(mgr, { path, uri, text, mtime, kind = 'file', readOnly = false, name }) {
        this.mgr = mgr;
        this.path = path;
        this.uri = uri;
        this.kind = kind;            // file | untitled | example
        this.readOnly = readOnly;
        this.name = name || basename(path);
        this.mtime = mtime;
        this.language = isTil(this.name) || kind !== 'file' && /\.til$/i.test(this.name) ? 'til' : 'plain';
        this.views = new Set();
        this.lspOpen = false;
        this.ctx = { uri, path, client: mgr.app.lsp, app: mgr.app };
        this.state = createEditorState({
            text, ctx: this.ctx, readOnly, language: this.language, settings: mgr.app.settings.values,
            listeners: [mgr.listener],
        });
        this.savedDoc = this.state.doc;
        this.diskConflict = false;
    }

    get text() { return this.state.doc.toString(); }

    get dirty() { return this.kind === 'untitled' ? this.state.doc.length > 0 && !this.state.doc.eq(this.savedDoc) : !this.state.doc.eq(this.savedDoc); }

    get isTil() { return this.language === 'til'; }

    /* the newest state: the one of the focused view showing it, else the stored one */
    current() {
        for (const v of this.views) if (v.hasFocus) return v.state;
        const first = this.views.values().next();
        return first.done ? this.state : first.value.state;
    }

    dispatch(spec) {
        if (this.views.size) {
            let first = true;
            for (const v of this.views) {
                v.dispatch(Object.assign({}, spec, first ? {} : { annotations: syncAnnotation.of(true) }));
                if (first) this.state = v.state;
                first = false;
            }
        } else {
            this.state = this.state.update(spec).state;
        }
    }

    markSaved(mtime) {
        this.state = this.current();
        this.savedDoc = this.state.doc;
        if (mtime !== undefined) this.mtime = mtime;
        this.diskConflict = false;
        this.mgr.emit('dirty', this);
    }

    replaceText(text, { keepHistory = true, saved = false } = {}) {
        const cur = this.current();
        if (cur.doc.toString() !== text) {
            const spec = { changes: { from: 0, to: cur.doc.length, insert: text } };
            if (!keepHistory) spec.annotations = Transaction.addToHistory.of(false);
            this.dispatch(spec);
        }
        if (saved) { this.state = this.current(); this.savedDoc = this.state.doc; }
        this.mgr.emit('dirty', this);
    }
}

export class DocumentManager extends Emitter {
    constructor(app) {
        super();
        this.app = app;
        this.docs = new Map();         // path -> Doc
        this.loading = new Map();      // path -> Promise<Doc>
        // one update listener shared by every editor state
        this.listener = viewListener(this);
    }

    get(path) { return this.docs.get(path) || null; }

    all() { return [...this.docs.values()]; }

    dirtyDocs() { return this.all().filter((d) => d.dirty); }

    async open(path) {
        if (this.docs.has(path)) return this.docs.get(path);
        if (this.loading.has(path)) return this.loading.get(path);
        const p = (async () => {
            const file = await this.app.backend.readFile(path);
            // the TIL examples ("@examples/...") and other files the server marks read-only
            const readOnly = Boolean(file.readonly) || path.startsWith('@');
            const doc = new Doc(this, { path, uri: this.app.uriOf(path), text: normalizeNewlines(file.text), mtime: file.mtime, readOnly });
            this.docs.set(path, doc);
            this.attach(doc);
            return doc;
        })();
        this.loading.set(path, p);
        try { return await p; } finally { this.loading.delete(path); }
    }

    /* a document that is not (yet) a file of the project: examples, new files */
    openVirtual({ text, name, kind = 'untitled', readOnly = false }) {
        const path = `${kind}:${++untitledSeq}/${name || 'untitled.til'}`;
        const doc = new Doc(this, { path, uri: `${kind}:///${untitledSeq}/${encodeURIComponent(name || 'untitled.til')}`, text, kind, readOnly, name: name || 'untitled.til' });
        doc.savedDoc = kind === 'example' ? doc.state.doc : doc.state.doc;
        this.docs.set(path, doc);
        this.attach(doc);
        return doc;
    }

    attach(doc) {
        if (doc.isTil && this.app.lsp && doc.kind !== 'example') {
            this.app.lsp.open(doc.uri, doc.state.doc.toString());
            doc.lspOpen = true;
        }
        const bps = this.app.breakpoints.get(doc.path);
        if (bps && bps.length) doc.dispatch({ effects: bpEffect.of({ type: 'set', list: bps }) });
        const diags = this.app.diagnostics.get(doc.uri);
        if (diags && diags.length) this.applyDiagnostics(doc, diags);
        this.emit('open', doc);
    }

    applyDiagnostics(doc, lspDiagnostics) {
        doc.dispatch(diagnosticsSpec(doc.current(), lspDiagnostics));
    }

    applySettings(settings) {
        const effects = settingsEffects(settings);
        for (const doc of this.docs.values()) doc.dispatch({ effects });
    }

    async save(doc, { force = false } = {}) {
        if (doc.kind !== 'file') return false;
        if (this.app.settings.get('formatOnSave') && doc.isTil) {
            try { await this.app.formatDoc(doc); } catch (e) { /* keep the text */ }
        }
        const text = doc.current().doc.toString();
        let res;
        try {
            res = await this.app.backend.writeFile(doc.path, text, force ? undefined : doc.mtime);
        } catch (e) {
            if (e && e.status === 409) {
                const choice = await this.app.askConflict(doc);
                if (choice === 'overwrite') return this.save(doc, { force: true });
                if (choice === 'reload') { await this.reload(doc); return false; }
                return false;
            }
            throw e;
        }
        doc.markSaved(res && res.mtime !== undefined ? res.mtime : doc.mtime);
        if (doc.lspOpen) this.app.lsp.saved(doc.uri, text);
        this.emit('saved', doc);
        return true;
    }

    async reload(doc) {
        const file = await this.app.backend.readFile(doc.path);
        doc.replaceText(normalizeNewlines(file.text), { saved: true });
        doc.mtime = file.mtime;
        doc.diskConflict = false;
        this.emit('dirty', doc);
    }

    close(doc) {
        if (doc.views.size) return;
        if (doc.lspOpen) this.app.lsp.close(doc.uri);
        this.docs.delete(doc.path);
        this.emit('close', doc);
    }

    /* the file was renamed (explorer): move the document with it */
    renamed(oldPath, newPath) {
        for (const [p, doc] of [...this.docs]) {
            if (p === oldPath || p.startsWith(oldPath + '/')) {
                const np = newPath + p.slice(oldPath.length);
                this.docs.delete(p);
                const text = doc.current().doc.toString();
                if (doc.lspOpen) this.app.lsp.close(doc.uri);
                doc.path = np;
                doc.name = basename(np);
                doc.uri = this.app.uriOf(np);
                doc.ctx.uri = doc.uri;
                doc.ctx.path = np;
                this.docs.set(np, doc);
                if (doc.lspOpen) this.app.lsp.open(doc.uri, text);
                this.emit('renamed', doc, p);
            }
        }
    }

    deleted(path) {
        for (const [p, doc] of [...this.docs]) {
            if (p === path || p.startsWith(path + '/')) this.emit('deleted', doc);
        }
    }

    /* files changed on disk (SSE fs events): reload clean documents, flag dirty ones */
    async externalChange(paths) {
        for (const p of paths) {
            const doc = this.docs.get(p);
            if (!doc || doc.kind !== 'file') continue;
            let file;
            try { file = await this.app.backend.readFile(p); } catch (e) { continue; }
            const text = normalizeNewlines(file.text);
            if (text === doc.savedDoc.toString()) { doc.mtime = file.mtime; continue; }
            if (!doc.dirty) {
                doc.replaceText(text, { saved: true });
                doc.mtime = file.mtime;
                this.emit('reloaded', doc);
            } else {
                doc.diskConflict = true;
                this.emit('conflict', doc);
            }
        }
    }

    breakpointsOf(doc) { return breakpointList(doc.current()); }
}

export function normalizeNewlines(text) {
    return String(text || '').replace(/\r\n?/g, '\n');
}

/* an update listener for every editor state: dirty flags, LSP sync, split-view sync, cursor */
function viewListener(mgr) {
    return EditorView.updateListener.of((update) => {
        const view = update.view;
        const doc = view._studioDoc;
        if (!doc) return;
        if (update.docChanged) {
            const synced = update.transactions.some((tr) => tr.annotation(syncAnnotation));
            if (!synced) {
                for (const other of doc.views) {
                    if (other === view) continue;
                    other.dispatch({ changes: update.changes, annotations: syncAnnotation.of(true) });
                }
                doc.state = view.state;
                if (doc.lspOpen) mgr.app.lsp.change(doc.uri, () => doc.current().doc.toString());
                mgr.emit('change', doc, update);
            }
            mgr.emit('dirty', doc);
        } else if (!doc.views.size || view.hasFocus) {
            doc.state = view.state;
        }
        const bpChanged = update.transactions.some((tr) => tr.effects.some((e) => e.is(bpEffect) && e.value.type !== 'verify')) ||
            (update.docChanged && update.startState.field(breakpointField, false) && update.startState.field(breakpointField).size > 0);
        if (bpChanged) {
            const forwarded = update.transactions.some((tr) => tr.annotation(syncAnnotation));
            if (!forwarded) {
                for (const tr of update.transactions) {
                    const effects = tr.effects.filter((e) => e.is(bpEffect));
                    if (!effects.length) continue;
                    for (const other of doc.views) if (other !== view) other.dispatch({ effects, annotations: syncAnnotation.of(true) });
                }
            }
            mgr.emit('breakpoints', doc, breakpointList(view.state));
        }
        if (update.selectionSet || update.docChanged || update.focusChanged) mgr.emit('cursor', doc, view, update);
    });
}
