// CodeMirror features from the language server: diagnostics (+ error lens),
// completion, hover, signature help, inlay hints, code lenses and document
// highlights. Positions: LSP uses 0-based lines and UTF-16 offsets, which are
// exactly JavaScript string offsets inside a CodeMirror line.
import { Facet, StateField, StateEffect, RangeSetBuilder } from '@codemirror/state';
import { EditorView, Decoration, WidgetType, ViewPlugin, hoverTooltip, showTooltip, keymap } from '@codemirror/view';
import { setDiagnostics } from '@codemirror/lint';
import { snippet } from '@codemirror/autocomplete';
import { renderMarkdown, hoverToMarkdown } from '../core/markdown.js';
import { highlightInto } from '../editor/til_lang.js';
import { t } from '../core/i18n.js';
import { h } from '../core/dom.js';

/* per-document context: {uri, path, client, app} */
export const docContext = Facet.define({ combine: (v) => v[0] || null });

/* ── positions ─────────────────────────────────────────────────────── */
export function toOffset(doc, p) {
    if (!p) return 0;
    const line = Math.max(0, Math.min(doc.lines - 1, p.line | 0));
    const l = doc.line(line + 1);
    return Math.min(l.to, l.from + Math.max(0, p.character | 0));
}

export function toPosition(doc, offset) {
    const l = doc.lineAt(Math.max(0, Math.min(doc.length, offset)));
    return { line: l.number - 1, character: offset - l.from };
}

export function rangeToOffsets(doc, range) {
    const from = toOffset(doc, range.start);
    const to = Math.max(from, toOffset(doc, range.end));
    return { from, to };
}

/* ── diagnostics ───────────────────────────────────────────────────── */
export const setLspDiagnostics = StateEffect.define();

/* the document's diagnostics as {from, to, severity, message, code, source, data, raw} */
export const diagnosticsField = StateField.define({
    create() { return []; },
    update(value, tr) {
        for (const e of tr.effects) if (e.is(setLspDiagnostics)) return e.value;
        if (!tr.docChanged || !value.length) return value;
        return value.map((d) => {
            const from = tr.changes.mapPos(d.from, 1);
            const to = Math.max(from, tr.changes.mapPos(d.to, -1));
            return Object.assign({}, d, { from, to });
        });
    },
});

const SEVERITY = { 1: 'error', 2: 'warning', 3: 'info', 4: 'hint' };

/* transaction spec that shows these LSP diagnostics (squiggles, hover, error lens) */
export function diagnosticsSpec(state, lspDiagnostics) {
    const doc = state.doc;
    const list = (lspDiagnostics || []).map((d) => {
        let { from, to } = rangeToOffsets(doc, d.range || { start: { line: 0, character: 0 }, end: { line: 0, character: 0 } });
        if (to <= from) {                              // never an empty range: the word or the rest of the line
            const line = doc.lineAt(from);
            const word = /^[\p{L}\p{N}_]+/u.exec(line.text.slice(from - line.from));
            to = word ? from + word[0].length : Math.min(line.to, from + 1);
            if (to <= from && from > line.from) from -= 1;
        }
        return {
            from, to, severity: SEVERITY[d.severity] || 'error', message: String(d.message || ''),
            code: d.code !== undefined && d.code !== null ? String(typeof d.code === 'object' ? d.code.value : d.code) : '',
            source: d.source || '', data: d.data, raw: d,
        };
    }).sort((a, b) => a.from - b.from || sevRank(a.severity) - sevRank(b.severity));
    const cm = list.map((d) => ({
        from: d.from, to: d.to, severity: d.severity === 'hint' ? 'info' : d.severity,
        message: d.message, source: [d.source, d.code].filter(Boolean).join(' '),
        markClass: d.severity === 'hint' ? 'cm-lintRange-hint' : undefined,
    }));
    const spec = setDiagnostics(state, cm);
    const effects = [].concat(spec.effects || [], setLspDiagnostics.of(list));
    return { effects };
}

export function sevRank(s) { return { error: 0, warning: 1, info: 2, hint: 3 }[s] ?? 4; }

/* error lens: the most severe diagnostic of each line, at the end of the line */
export const errorLensEnabled = Facet.define({ combine: (v) => (v.length ? v[v.length - 1] : true) });

class LensWidget extends WidgetType {
    constructor(d, more) { super(); this.d = d; this.more = more; }
    eq(o) { return o.d.message === this.d.message && o.d.severity === this.d.severity && o.more === this.more; }
    toDOM() {
        const first = this.d.message.split('\n')[0];
        const el = h('span', { class: `cm-errorLens cm-errorLens-${this.d.severity}`, 'aria-hidden': 'true' },
            first + (this.more ? `  (+${this.more})` : ''));
        return el;
    }
    ignoreEvent() { return true; }
}

export const errorLensField = StateField.define({
    create(state) { return buildLens(state); },
    update(value, tr) {
        if (tr.effects.some((e) => e.is(setLspDiagnostics)) || tr.startState.facet(errorLensEnabled) !== tr.state.facet(errorLensEnabled)) return buildLens(tr.state);
        return tr.docChanged ? value.map(tr.changes) : value;
    },
    provide: (f) => EditorView.decorations.from(f),
});

function buildLens(state) {
    if (!state.facet(errorLensEnabled)) return Decoration.none;
    const diags = state.field(diagnosticsField, false) || [];
    const byLine = new Map();
    for (const d of diags) {
        if (d.severity === 'hint') continue;
        const line = state.doc.lineAt(Math.min(d.from, state.doc.length));
        const cur = byLine.get(line.number);
        if (!cur) byLine.set(line.number, { d, n: 1, to: line.to });
        else {
            cur.n++;
            if (sevRank(d.severity) < sevRank(cur.d.severity)) cur.d = d;
        }
    }
    const b = new RangeSetBuilder();
    for (const n of [...byLine.keys()].sort((a, c) => a - c)) {
        const e = byLine.get(n);
        b.add(e.to, e.to, Decoration.widget({ widget: new LensWidget(e.d, e.n - 1), side: 1 }));
    }
    return b.finish();
}

/* ── completion ────────────────────────────────────────────────────── */
const KIND = {
    1: 'text', 2: 'method', 3: 'function', 4: 'function', 5: 'property', 6: 'variable', 7: 'class', 8: 'interface',
    9: 'namespace', 10: 'property', 11: 'constant', 12: 'variable', 13: 'enum', 14: 'keyword', 15: 'text', 16: 'constant',
    17: 'text', 18: 'text', 19: 'namespace', 20: 'enum', 21: 'constant', 22: 'class', 23: 'keyword', 24: 'keyword', 25: 'type',
};

function lspSnippetToCm(text) {
    return String(text)
        .replace(/\$\{(\d+)\|([^|}]*)(?:,[^|}]*)*\|\}/g, '${$1:$2}')        // choices: the first one
        .replace(/\$(\d+)/g, '${$1}')
        .replace(/\\\$/g, '$');
}

export async function completionSource(context) {
    const ctx = context.state.facet(docContext);
    if (!ctx || !ctx.client || !ctx.client.ready || !ctx.client.has('completionProvider')) return null;
    const word = context.matchBefore(/[\p{L}\p{N}_]*/u);
    const before = context.state.sliceDoc(Math.max(0, context.pos - 1), context.pos);
    const triggers = ctx.client.triggers('completionProvider');
    const triggered = triggers.includes(before);
    if (!context.explicit && !triggered && (!word || word.from === word.to)) return null;
    // inside a comment or a plain string: no completion (attributes #[...] still complete)
    const lineText = context.state.doc.lineAt(context.pos);
    const head = lineText.text.slice(0, context.pos - lineText.from);
    if (/(^|[^#])#(?!\[)/.test(head.replace(/"[^"]*"|'[^']*'/g, '""'))) return null;
    let result;
    try {
        result = await ctx.client.request('textDocument/completion', {
            textDocument: { uri: ctx.uri },
            position: toPosition(context.state.doc, context.pos),
            context: triggered && !context.explicit ? { triggerKind: 2, triggerCharacter: before } : { triggerKind: 1 },
        });
    } catch (e) {
        return null;
    }
    if (context.aborted) return null;
    const items = Array.isArray(result) ? result : (result && result.items) || [];
    if (!items.length) return null;
    const from = word ? word.from : context.pos;
    const ordered = items.map((item, i) => ({ item, i })).sort((a, b) => {
        const sa = a.item.sortText || a.item.label || '', sb = b.item.sortText || b.item.label || '';
        return sa < sb ? -1 : sa > sb ? 1 : a.i - b.i;
    });
    const options = ordered.map(({ item }, i) => toCompletion(item, i, ctx, context.state.doc));
    return {
        from,
        options,
        validFor: /^[\p{L}\p{N}_]*$/u,
    };
}

function toCompletion(item, index, ctx, doc) {
    const label = String(item.label || '');
    const kind = KIND[item.kind] || 'text';
    const c = {
        label,
        type: kind,
        detail: item.detail || (item.labelDetails && (item.labelDetails.detail || item.labelDetails.description)) || undefined,
        boost: item.preselect ? 50 : -Math.min(99, index / 10),
    };
    if (item.filterText && item.filterText !== label) c.displayLabel = label;
    const insert = item.textEdit ? item.textEdit.newText : (item.insertText || label);
    const isSnippet = item.insertTextFormat === 2;
    c.apply = (view, completion, from, to) => {
        let f = from, tEnd = to;
        if (item.textEdit && item.textEdit.range) {
            const r = rangeToOffsets(view.state.doc, item.textEdit.range);
            f = Math.min(r.from, from);
            tEnd = Math.max(to, view.state.selection.main.head);
        } else if (item.textEdit && item.textEdit.insert) {
            const r = rangeToOffsets(view.state.doc, item.textEdit.replace || item.textEdit.insert);
            f = Math.min(r.from, from);
        }
        if (isSnippet) snippet(lspSnippetToCm(insert))(view, completion, f, tEnd);
        else view.dispatch({ changes: { from: f, to: tEnd, insert }, selection: { anchor: f + insert.length }, userEvent: 'input.complete' });
        if (Array.isArray(item.additionalTextEdits) && item.additionalTextEdits.length) applyTextEdits(view, item.additionalTextEdits);
    };
    c.info = async () => {
        let it = item;
        if (!it.documentation && ctx.client.capabilities.completionProvider && ctx.client.capabilities.completionProvider.resolveProvider) {
            try { it = Object.assign({}, item, await ctx.client.request('completionItem/resolve', item)); } catch (e) { it = item; }
        }
        const md = typeof it.documentation === 'string' ? it.documentation : it.documentation ? hoverToMarkdown(it.documentation) : '';
        if (!md && !it.detail) return null;
        const box = h('div', { class: 'cm-lsp-doc' });
        if (it.detail) box.appendChild(highlightInto(h('pre', { class: 'cm-lsp-sig' }), it.detail));
        if (md) box.appendChild(renderMarkdown(md, { highlight: highlightInto }));
        return box;
    };
    return c;
}

/* ── text edits ────────────────────────────────────────────────────── */
export function textEditsToChanges(doc, edits) {
    return (edits || []).map((e) => {
        const r = rangeToOffsets(doc, e.range);
        return { from: r.from, to: r.to, insert: e.newText || '' };
    }).sort((a, b) => a.from - b.from || a.to - b.to);
}

export function applyTextEdits(view, edits, userEvent = 'input') {
    const changes = textEditsToChanges(view.state.doc, edits);
    if (!changes.length) return false;
    view.dispatch({ changes, userEvent });
    return true;
}

/* ── hover ─────────────────────────────────────────────────────────── */
export const lspHover = hoverTooltip(async (view, pos) => {
    const ctx = view.state.facet(docContext);
    if (!ctx || !ctx.client || !ctx.client.ready || !ctx.client.has('hoverProvider')) return null;
    const line = view.state.doc.lineAt(pos);
    const ch = line.text[pos - line.from] || '';
    const prev = line.text[pos - line.from - 1] || '';
    if (!/[\p{L}\p{N}_]/u.test(ch) && !/[\p{L}\p{N}_]/u.test(prev)) return null;
    let result;
    try {
        result = await ctx.client.request('textDocument/hover', { textDocument: { uri: ctx.uri }, position: toPosition(view.state.doc, pos) });
    } catch (e) {
        return null;
    }
    if (!result || !result.contents) return null;
    const md = hoverToMarkdown(result.contents);
    if (!md.trim()) return null;
    let from = pos, to = pos;
    if (result.range) ({ from, to } = rangeToOffsets(view.state.doc, result.range));
    else {
        const w = view.state.wordAt(pos);
        if (w) { from = w.from; to = w.to; }
    }
    return {
        pos: from, end: to, above: true,
        create() {
            const dom = h('div', { class: 'cm-lsp-hover' });
            dom.appendChild(renderMarkdown(md, { highlight: highlightInto }));
            return { dom };
        },
    };
}, { hoverTime: 320 });

/* ── signature help ────────────────────────────────────────────────── */
const setSignature = StateEffect.define();

const signatureField = StateField.define({
    create() { return null; },
    update(value, tr) {
        for (const e of tr.effects) if (e.is(setSignature)) return e.value;
        if (value && tr.docChanged) return Object.assign({}, value, { pos: tr.changes.mapPos(value.pos) });
        if (value && tr.selection) {
            const head = tr.state.selection.main.head;
            const line = tr.state.doc.lineAt(head);
            if (head < value.pos || line.from > value.pos) return null;
        }
        return value;
    },
    provide: (f) => showTooltip.compute([f], (state) => {
        const v = state.field(f);
        if (!v) return null;
        return { pos: v.pos, above: true, strictSide: true, arrow: false, create: () => ({ dom: renderSignature(v.help) }) };
    }),
});

function renderSignature(help) {
    const sigs = help.signatures || [];
    const active = Math.min(sigs.length - 1, Math.max(0, help.activeSignature || 0));
    const sig = sigs[active];
    const dom = h('div', { class: 'cm-lsp-signature' });
    if (!sig) return dom;
    const label = String(sig.label || '');
    const pIndex = sig.activeParameter !== undefined && sig.activeParameter !== null ? sig.activeParameter : help.activeParameter || 0;
    const params = sig.parameters || [];
    const code = h('div', { class: 'cm-lsp-sig-label' });
    const p = params[pIndex];
    let range = null;
    if (p) {
        if (Array.isArray(p.label)) range = p.label;
        else if (typeof p.label === 'string') {
            const at = label.indexOf(p.label, label.indexOf('('));
            if (at >= 0) range = [at, at + p.label.length];
        }
    }
    if (range) {
        code.append(label.slice(0, range[0]), h('strong', { class: 'cm-lsp-sig-active' }, label.slice(range[0], range[1])), label.slice(range[1]));
    } else code.textContent = label;
    dom.appendChild(code);
    const docText = (p && p.documentation) || sig.documentation;
    if (docText) dom.appendChild(renderMarkdown(typeof docText === 'string' ? docText : hoverToMarkdown(docText)));
    if (sigs.length > 1) dom.appendChild(h('div', { class: 'cm-lsp-sig-count' }, `${active + 1}/${sigs.length}`));
    return dom;
}

const signaturePlugin = ViewPlugin.fromClass(class {
    constructor(view) { this.view = view; this.timer = 0; this.seq = 0; }
    update(u) {
        if (!u.docChanged && !u.selectionSet) return;
        const ctx = u.state.facet(docContext);
        if (!ctx || !ctx.client || !ctx.client.ready || !ctx.client.has('signatureHelpProvider')) return;
        const open = u.state.field(signatureField, false);
        let typed = '';
        for (const tr of u.transactions) {
            if (!tr.isUserEvent('input.type')) continue;
            tr.changes.iterChanges((fa, ta, fb, tb, ins) => { typed += ins.toString(); });
        }
        const triggers = ctx.client.triggers('signatureHelpProvider').concat(['(', ',']);
        const last = typed.slice(-1);
        if (last === ')' && open) { this.close(); return; }
        if (!open && !(last && triggers.includes(last))) return;
        clearTimeout(this.timer);
        this.timer = setTimeout(() => this.fetch(open ? 3 : 2, last), open ? 120 : 30);
    }
    close() {
        if (this.view.state.field(signatureField, false)) this.view.dispatch({ effects: setSignature.of(null) });
    }
    async fetch(kind, ch) {
        const view = this.view;
        const ctx = view.state.facet(docContext);
        const seq = ++this.seq;
        const head = view.state.selection.main.head;
        let help = null;
        try {
            help = await ctx.client.request('textDocument/signatureHelp', {
                textDocument: { uri: ctx.uri }, position: toPosition(view.state.doc, head),
                context: { triggerKind: kind === 2 ? 2 : 3, triggerCharacter: kind === 2 ? ch : undefined, isRetrigger: kind === 3 },
            });
        } catch (e) { help = null; }
        if (seq !== this.seq || !view.dom.isConnected) return;
        if (!help || !help.signatures || !help.signatures.length) { this.close(); return; }
        const line = view.state.doc.lineAt(view.state.selection.main.head);
        const open = openParen(line.text, view.state.selection.main.head - line.from);
        view.dispatch({ effects: setSignature.of({ pos: open >= 0 ? line.from + open : view.state.selection.main.head, help }) });
    }
    destroy() { clearTimeout(this.timer); }
});

function openParen(text, upto) {
    let depth = 0;
    for (let i = upto - 1; i >= 0; i--) {
        const c = text[i];
        if (c === ')') depth++;
        else if (c === '(') { if (depth === 0) return i; depth--; }
    }
    return -1;
}

const signatureKeys = keymap.of([{
    key: 'Escape',
    run: (view) => {
        if (!view.state.field(signatureField, false)) return false;
        view.dispatch({ effects: setSignature.of(null) });
        return true;
    },
}]);

/* ── inlay hints and code lenses (state fields fed by a fetching plugin) ── */
export const setInlayHints = StateEffect.define();
export const setCodeLenses = StateEffect.define();
export const setHighlights = StateEffect.define();
export const lspFeatureToggles = Facet.define({ combine: (v) => Object.assign({ inlayHints: true, codeLens: true }, ...v) });

class HintWidget extends WidgetType {
    constructor(label, kind, padL, padR) { super(); this.label = label; this.kind = kind; this.padL = padL; this.padR = padR; }
    eq(o) { return o.label === this.label && o.kind === this.kind; }
    toDOM() {
        return h('span', { class: `cm-inlayHint ${this.kind === 2 ? 'cm-inlayHint-param' : 'cm-inlayHint-type'}${this.padL ? ' pl' : ''}${this.padR ? ' pr' : ''}` }, this.label);
    }
    ignoreEvent() { return true; }
}

const inlayField = StateField.define({
    create() { return Decoration.none; },
    update(value, tr) {
        for (const e of tr.effects) if (e.is(setInlayHints)) return e.value;
        return tr.docChanged ? value.map(tr.changes) : value;
    },
    provide: (f) => EditorView.decorations.from(f),
});

class LensBlock extends WidgetType {
    constructor(lenses, indent) { super(); this.lenses = lenses; this.indent = indent; }
    eq(o) { return o.indent === this.indent && JSON.stringify(o.lenses.map((l) => l.command && l.command.title)) === JSON.stringify(this.lenses.map((l) => l.command && l.command.title)); }
    toDOM(view) {
        const ctx = view.state.facet(docContext);
        const dom = h('div', { class: 'cm-codeLens', style: { paddingLeft: `${this.indent}ch` } });
        this.lenses.forEach((lens, i) => {
            if (!lens.command) return;
            if (i) dom.appendChild(h('span', { class: 'cm-codeLens-sep', 'aria-hidden': 'true' }, '·'));
            const title = String(lens.command.title || '').replace(/^[▶▷►]\s*/, '');
            const cmd = lens.command.command || '';
            const icon = /run/.test(cmd) ? '▶ ' : /debug/.test(cmd) ? '⬢ ' : /prove/.test(cmd) ? '✓ ' : '';
            const btn = h('button', { type: 'button', class: 'cm-codeLens-btn', dataset: { command: cmd } }, icon + title);
            btn.addEventListener('mousedown', (e) => e.preventDefault());
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                if (ctx && ctx.app) ctx.app.executeLensCommand(lens.command, ctx.path);
            });
            dom.appendChild(btn);
        });
        return dom;
    }
    ignoreEvent() { return true; }
    get estimatedHeight() { return 18; }
}

const lensField = StateField.define({
    create() { return Decoration.none; },
    update(value, tr) {
        for (const e of tr.effects) if (e.is(setCodeLenses)) return e.value;
        return tr.docChanged ? value.map(tr.changes) : value;
    },
    provide: (f) => EditorView.decorations.from(f),
});

const highlightMark = { 1: Decoration.mark({ class: 'cm-docHighlight' }), 2: Decoration.mark({ class: 'cm-docHighlight cm-docHighlight-read' }),
    3: Decoration.mark({ class: 'cm-docHighlight cm-docHighlight-write' }) };

const highlightField = StateField.define({
    create() { return Decoration.none; },
    update(value, tr) {
        for (const e of tr.effects) if (e.is(setHighlights)) return e.value;
        if (tr.docChanged) return Decoration.none;
        return value;
    },
    provide: (f) => EditorView.decorations.from(f),
});

const fetchPlugin = ViewPlugin.fromClass(class {
    constructor(view) {
        this.view = view;
        this.hintTimer = 0;
        this.lensTimer = 0;
        this.hlTimer = 0;
        this.seq = { hint: 0, lens: 0, hl: 0 };
        this.schedule(true, true, 60);
    }
    update(u) {
        if (u.docChanged) this.schedule(true, true, 450);
        else if (u.viewportChanged) this.schedule(true, false, 200);
        if (u.selectionSet && !u.docChanged) {
            clearTimeout(this.hlTimer);
            this.hlTimer = setTimeout(() => this.highlights(), 350);
        }
        if (u.startState.facet(lspFeatureToggles) !== u.state.facet(lspFeatureToggles)) this.schedule(true, true, 0);
    }
    schedule(hints, lenses, ms) {
        if (hints) { clearTimeout(this.hintTimer); this.hintTimer = setTimeout(() => this.hints(), ms); }
        if (lenses) { clearTimeout(this.lensTimer); this.lensTimer = setTimeout(() => this.lenses(), ms + 150); }
    }
    ctx() {
        const ctx = this.view.state.facet(docContext);
        return ctx && ctx.client && ctx.client.ready ? ctx : null;
    }
    async hints() {
        const ctx = this.ctx();
        const view = this.view;
        if (!ctx || !view.dom.isConnected) return;
        const on = view.state.facet(lspFeatureToggles).inlayHints;
        if (!on || !ctx.client.has('inlayHintProvider')) {
            if (view.state.field(inlayField, false).size) view.dispatch({ effects: setInlayHints.of(Decoration.none) });
            return;
        }
        const seq = ++this.seq.hint;
        const doc = view.state.doc;
        const { from, to } = view.viewport;
        const range = { start: toPosition(doc, doc.lineAt(from).from), end: toPosition(doc, doc.lineAt(Math.min(doc.length, to)).to) };
        let list;
        try { list = await ctx.client.request('textDocument/inlayHint', { textDocument: { uri: ctx.uri }, range }); } catch (e) { list = null; }
        if (seq !== this.seq.hint || view.state.doc !== doc || !view.dom.isConnected) return;
        const b = [];
        for (const hint of list || []) {
            const pos = toOffset(doc, hint.position);
            const label = typeof hint.label === 'string' ? hint.label : (hint.label || []).map((p) => p.value).join('');
            if (!label) continue;
            b.push(Decoration.widget({ widget: new HintWidget(label, hint.kind, hint.paddingLeft, hint.paddingRight), side: hint.kind === 2 ? -1 : 1 }).range(pos));
        }
        view.dispatch({ effects: setInlayHints.of(Decoration.set(b, true)) });
    }
    async lenses() {
        const ctx = this.ctx();
        const view = this.view;
        if (!ctx || !view.dom.isConnected) return;
        const on = view.state.facet(lspFeatureToggles).codeLens;
        if (!on || !ctx.client.has('codeLensProvider')) {
            if (view.state.field(lensField, false).size) view.dispatch({ effects: setCodeLenses.of(Decoration.none) });
            return;
        }
        const seq = ++this.seq.lens;
        const doc = view.state.doc;
        let list;
        try { list = await ctx.client.request('textDocument/codeLens', { textDocument: { uri: ctx.uri } }); } catch (e) { list = null; }
        if (seq !== this.seq.lens || view.state.doc !== doc || !view.dom.isConnected) return;
        const byLine = new Map();
        for (const lens of list || []) {
            if (!lens || !lens.range) continue;
            const n = Math.max(0, Math.min(doc.lines - 1, lens.range.start.line));
            if (!byLine.has(n)) byLine.set(n, []);
            byLine.get(n).push(lens);
        }
        const b = [];
        for (const n of [...byLine.keys()].sort((a, c) => a - c)) {
            const line = doc.line(n + 1);
            // above the attributes of a function (#[requires] ...), like an editor shows them
            let at = line;
            while (at.number > 1 && /^\s*#\[/.test(doc.line(at.number - 1).text)) at = doc.line(at.number - 1);
            const indent = (/^\s*/.exec(line.text) || [''])[0].length;
            b.push(Decoration.widget({ widget: new LensBlock(byLine.get(n), indent), block: true, side: -1 }).range(at.from));
        }
        view.dispatch({ effects: setCodeLenses.of(Decoration.set(b, true)) });
    }
    async highlights() {
        const ctx = this.ctx();
        const view = this.view;
        if (!ctx || !ctx.client.has('documentHighlightProvider') || !view.dom.isConnected) return;
        const sel = view.state.selection.main;
        const word = view.state.wordAt(sel.head);
        if (!sel.empty || !word) {
            if (view.state.field(highlightField, false).size) view.dispatch({ effects: setHighlights.of(Decoration.none) });
            return;
        }
        const seq = ++this.seq.hl;
        const doc = view.state.doc;
        let list;
        try { list = await ctx.client.request('textDocument/documentHighlight', { textDocument: { uri: ctx.uri }, position: toPosition(doc, sel.head) }); } catch (e) { list = null; }
        if (seq !== this.seq.hl || view.state.doc !== doc || !view.dom.isConnected) return;
        const ranges = (list || []).map((hl) => {
            const r = rangeToOffsets(doc, hl.range);
            return r.to > r.from ? (highlightMark[hl.kind] || highlightMark[1]).range(r.from, r.to) : null;
        }).filter(Boolean);
        if (ranges.length < 2) ranges.length = 0;
        view.dispatch({ effects: setHighlights.of(Decoration.set(ranges, true)) });
    }
    destroy() { clearTimeout(this.hintTimer); clearTimeout(this.lensTimer); clearTimeout(this.hlTimer); }
});

/* every LSP-driven extension of an editor state */
export function lspExtensions() {
    return [diagnosticsField, errorLensField, lspHover, signatureField, signaturePlugin, signatureKeys, inlayField, lensField, highlightField, fetchPlugin];
}

export function translateKey(key) { return t(key); }
