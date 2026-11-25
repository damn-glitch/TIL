// Editor decorations owned by the studio: breakpoints (gutter, click to
// toggle), the current debug line, proof marks, bytecode-linked lines, a
// flash for "go to" jumps, and indentation guides (TIL blocks are indented).
import { StateField, StateEffect, RangeSet, RangeSetBuilder } from '@codemirror/state';
import { EditorView, Decoration, GutterMarker, gutter, ViewPlugin } from '@codemirror/view';
import { docContext } from '../lsp/cm_lsp.js';
import { icon } from '../core/icons.js';

/* ── breakpoints ── */
export const bpEffect = StateEffect.define();

class BpMarker extends GutterMarker {
    constructor(bp) { super(); this.bp = bp; }
    eq(o) { return o.bp.condition === this.bp.condition && o.bp.enabled === this.bp.enabled && o.bp.verified === this.bp.verified && o.bp.hitCondition === this.bp.hitCondition; }
    toDOM() {
        const cls = ['bp-dot'];
        if (this.bp.enabled === false) cls.push('disabled');
        else if (this.bp.verified === false) cls.push('unverified');
        if (this.bp.condition || this.bp.hitCondition) cls.push('conditional');
        const el = document.createElement('div');
        el.className = cls.join(' ');
        el.title = this.bp.condition ? `if ${this.bp.condition}` : '';
        return el;
    }
}

class GhostMarker extends GutterMarker {
    eq() { return true; }
    toDOM() { const el = document.createElement('div'); el.className = 'bp-ghost'; return el; }
}
const ghost = new GhostMarker();

class ArrowMarker extends GutterMarker {
    eq() { return true; }
    toDOM() { const el = document.createElement('div'); el.className = 'debug-arrow'; el.appendChild(icon('debugArrow')); return el; }
}
const arrow = new ArrowMarker();

function bpRange(state, line, bp) {
    const l = state.doc.line(Math.max(1, Math.min(state.doc.lines, line)));
    return new BpMarker(bp).range(l.from);
}

export const breakpointField = StateField.define({
    create() { return RangeSet.empty; },
    update(set, tr) {
        set = set.map(tr.changes);
        for (const e of tr.effects) {
            if (!e.is(bpEffect)) continue;
            const v = e.value;
            if (v.type === 'set') {
                const ranges = [];
                const seen = new Set();
                for (const bp of v.list) {
                    if (bp.line < 1 || bp.line > tr.state.doc.lines || seen.has(bp.line)) continue;
                    seen.add(bp.line);
                    ranges.push(bpRange(tr.state, bp.line, bp));
                }
                ranges.sort((a, b) => a.from - b.from);
                set = RangeSet.of(ranges);
            } else if (v.type === 'toggle') {
                const line = tr.state.doc.line(v.line);
                let had = false;
                set.between(line.from, line.from, () => { had = true; });
                if (had) set = set.update({ filter: (from) => from !== line.from });
                else set = set.update({ add: [bpRange(tr.state, v.line, { enabled: true, condition: v.condition || '' })] });
            } else if (v.type === 'update') {
                const line = tr.state.doc.line(v.line);
                let cur = null;
                set.between(line.from, line.from, (f, t, m) => { cur = m.bp; });
                if (cur) {
                    set = set.update({ filter: (from) => from !== line.from, add: [bpRange(tr.state, v.line, Object.assign({}, cur, v.props))] });
                }
            } else if (v.type === 'verify') {
                // the adapter's answer: verified flags (and moved lines) of every breakpoint
                const ranges = [];
                const seen = new Set();
                for (const bp of v.list) {
                    if (bp.line < 1 || bp.line > tr.state.doc.lines || seen.has(bp.line)) continue;
                    seen.add(bp.line);
                    ranges.push(bpRange(tr.state, bp.line, bp));
                }
                ranges.sort((a, b) => a.from - b.from);
                set = RangeSet.of(ranges);
            }
        }
        if (tr.docChanged) {
            // two breakpoints that end up on one line after a deletion: keep one
            const seen = new Set();
            set = set.update({ filter: (from) => { const n = tr.state.doc.lineAt(from).number; if (seen.has(n)) return false; seen.add(n); return true; } });
        }
        return set;
    },
});

/* [{line, condition, hitCondition, enabled, verified}] */
export function breakpointList(state) {
    const out = [];
    const set = state.field(breakpointField, false);
    if (!set) return out;
    const it = set.iter();
    while (it.value) {
        out.push(Object.assign({}, it.value.bp, { line: state.doc.lineAt(it.from).number }));
        it.next();
    }
    return out;
}

export const setDebugLine = StateEffect.define();

export const debugLineField = StateField.define({
    create() { return null; },
    update(v, tr) {
        for (const e of tr.effects) if (e.is(setDebugLine)) return e.value;
        return v;
    },
    provide: (f) => EditorView.decorations.compute([f], (state) => {
        const v = state.field(f);
        if (!v || v.line < 1 || v.line > state.doc.lines) return Decoration.none;
        const l = state.doc.line(v.line);
        return Decoration.set([Decoration.line({ class: v.top === false ? 'cm-debugLine cm-debugLine-top' : 'cm-debugLine' }).range(l.from)]);
    }),
});

const bpGutter = gutter({
    class: 'cm-bp-gutter',
    markers: (view) => view.state.field(breakpointField),
    lineMarker(view, line, others) {
        const dbg = view.state.field(debugLineField, false);
        if (dbg && dbg.line >= 1 && dbg.line <= view.state.doc.lines && view.state.doc.line(dbg.line).from === line.from) return arrow;
        if (others.length) return null;
        return ghost;
    },
    lineMarkerChange: (u) => u.startState.field(debugLineField, false) !== u.state.field(debugLineField, false),
    initialSpacer: () => ghost,
    domEventHandlers: {
        mousedown(view, line, event) {
            if (event.button !== 0) return false;
            const n = view.state.doc.lineAt(line.from).number;
            view.dispatch({ effects: bpEffect.of({ type: 'toggle', line: n }) });
            return true;
        },
        contextmenu(view, line, event) {
            const ctx = view.state.facet(docContext);
            if (!ctx || !ctx.app) return false;
            event.preventDefault();
            ctx.app.breakpointMenu(view, view.state.doc.lineAt(line.from).number, event);
            return true;
        },
    },
});

/* ── proof marks ── */
export const setProofMarks = StateEffect.define();

class ProofMarker extends GutterMarker {
    constructor(status, title) { super(); this.status = status; this.title = title; }
    eq(o) { return o.status === this.status && o.title === this.title; }
    toDOM() {
        const name = { proved: 'checkCircle', counterexample: 'crossCircle', unsupported: 'banCircle' }[this.status] || 'questionCircle';
        const el = icon(name, { className: `proof-mark ${this.status.replace(/\s+/g, '-')}` });
        const wrap = document.createElement('div');
        wrap.title = this.title || this.status;
        wrap.appendChild(el);
        return wrap;
    }
}

export const proofField = StateField.define({
    create() { return RangeSet.empty; },
    update(set, tr) {
        for (const e of tr.effects) {
            if (!e.is(setProofMarks)) continue;
            const ranges = [];
            const seen = new Set();
            for (const m of e.value) {
                if (m.line < 1 || m.line > tr.state.doc.lines || seen.has(m.line)) continue;
                seen.add(m.line);
                ranges.push(new ProofMarker(m.status, m.title).range(tr.state.doc.line(m.line).from));
            }
            ranges.sort((a, b) => a.from - b.from);
            return RangeSet.of(ranges);
        }
        return set.map(tr.changes);
    },
});

const proofGutter = gutter({ class: 'cm-proof-gutter', markers: (view) => view.state.field(proofField) });

/* ── lines linked from other views (bytecode hover), flash after a jump ── */
export const setLinkedLines = StateEffect.define();
export const linkedLinesField = StateField.define({
    create() { return []; },
    update(v, tr) {
        for (const e of tr.effects) if (e.is(setLinkedLines)) return e.value;
        return tr.docChanged ? [] : v;
    },
    provide: (f) => EditorView.decorations.compute([f], (state) => {
        const lines = [...new Set(state.field(f))].filter((n) => n >= 1 && n <= state.doc.lines).sort((a, b) => a - b);
        return Decoration.set(lines.map((n) => Decoration.line({ class: 'cm-bytecodeLine' }).range(state.doc.line(n).from)));
    }),
});

export const flashEffect = StateEffect.define();
export const flashField = StateField.define({
    create() { return null; },
    update(v, tr) {
        for (const e of tr.effects) if (e.is(flashEffect)) return e.value;
        return tr.docChanged ? null : v;
    },
    provide: (f) => EditorView.decorations.compute([f], (state) => {
        const v = state.field(f);
        if (!v || v.line < 1 || v.line > state.doc.lines) return Decoration.none;
        return Decoration.set([Decoration.line({ class: 'cm-flashLine', attributes: { 'data-flash': String(v.seq) } }).range(state.doc.line(v.line).from)]);
    }),
});

/* ── indentation guides ── */
function indentWidth(text) {
    let n = 0;
    for (const c of text) { if (c === ' ') n++; else if (c === '\t') n += 4; else break; }
    return n;
}

const guidePlugin = ViewPlugin.fromClass(class {
    constructor(view) { this.decorations = this.build(view); }
    update(u) { if (u.docChanged || u.viewportChanged) this.decorations = this.build(u.view); }
    build(view) {
        const b = new RangeSetBuilder();
        const doc = view.state.doc;
        for (const { from, to } of view.visibleRanges) {
            let n = doc.lineAt(from).number;
            const last = doc.lineAt(to).number;
            for (; n <= last; n++) {
                const line = doc.line(n);
                let width;
                if (line.text.trim()) width = indentWidth(line.text);
                else {
                    // blank line inside a block: the smaller indentation of its neighbours
                    let prev = 0, next = 0;
                    for (let k = n - 1; k >= 1 && k > n - 50; k--) { const t = doc.line(k).text; if (t.trim()) { prev = indentWidth(t); break; } }
                    for (let k = n + 1; k <= doc.lines && k < n + 50; k++) { const t = doc.line(k).text; if (t.trim()) { next = indentWidth(t); break; } }
                    width = Math.min(prev, next);
                }
                const levels = Math.floor(width / 4);
                if (levels > 0) b.add(line.from, line.from, Decoration.line({ class: 'cm-ig', attributes: { style: `--ig:${levels}` } }));
            }
        }
        return b.finish();
    }
}, { decorations: (v) => v.decorations });

export function studioGutters() {
    return [breakpointField, debugLineField, bpGutter, proofField, proofGutter, linkedLinesField, flashField, guidePlugin];
}
