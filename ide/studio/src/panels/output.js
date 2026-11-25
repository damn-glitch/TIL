// Output: the program's terminal. ANSI colours, a header and an exit line
// per run, clickable source locations, a stdin line (Enter sends a line,
// Ctrl+D ends the input), and a cap on the kept output.
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, L } from '../core/i18n.js';
import { AnsiParser } from '../core/ansi.js';
import { tip } from '../ui/widgets.js';

const MAX_CHARS = 400000;
const CHUNK = 8192;
const LOC_RE = /(-->\s*)?((?:[\w.\-/]|[^\x00-\x7F])+\.til):(\d+)(?::(\d+))?|\((?:line|строка|жол)\s+(\d+)\)/g;

export class OutputView {
    constructor(app) {
        this.app = app;
        this.out = h('div', { class: 'term-out', role: 'log', 'aria-live': 'polite', 'aria-label': t('panel.output'), tabindex: '0' });
        this.prompt = h('span', { class: 'term-prompt', 'aria-hidden': 'true' }, '›');
        this.input = h('input', { class: 'term-input', type: 'text', spellcheck: 'false', autocomplete: 'off', 'aria-label': t('output.stdinLabel') });
        this.inputHint = h('span', { class: 'term-input-hint' });
        this.inputRow = h('form', { class: 'term-input-row', hidden: true }, this.prompt, this.input, this.inputHint);
        this.inputRow.addEventListener('submit', (e) => { e.preventDefault(); this.submit(); });
        this.input.addEventListener('keydown', (e) => {
            if (e.ctrlKey && (e.key === 'd' || e.key === 'D')) { e.preventDefault(); this.eof(); }
            else if (e.key === 'ArrowUp') { e.preventDefault(); this.history(-1); }
            else if (e.key === 'ArrowDown') { e.preventDefault(); this.history(1); }
        });
        this.el = h('div', { class: 'output-view term' }, this.out, this.inputRow);
        this.out.addEventListener('click', (e) => {
            const a = e.target.closest('.term-loc');
            if (a) this.app.openFile(a.dataset.path, { line: Number(a.dataset.line), col: Number(a.dataset.col || 1) });
        });
        this.out.addEventListener('keydown', (e) => this.app.runner.forwardKey(e));
        this.parsers = { 1: new AnsiParser(), 2: new AnsiParser() };
        this.chars = 0;
        this.chunk = null;
        this.chunkLen = 0;
        this.pending = [];
        this.raf = 0;
        this.sink = null;           // (text) => send stdin
        this.hist = [];
        this.histPos = 0;
        this.trimmedNote = false;
        this.renderEmpty();
    }

    actions() {
        const clearBtn = h('button', { class: 'icon-btn small', type: 'button' }, icon('clear'));
        tip(clearBtn, 'output.clear');
        clearBtn.addEventListener('click', () => this.clear());
        const wrap = h('button', { class: 'icon-btn small', type: 'button', 'aria-pressed': String(this.el.classList.contains('wrap')) }, icon('wrap'));
        tip(wrap, 'output.wrap');
        wrap.addEventListener('click', () => { const on = this.el.classList.toggle('wrap'); wrap.setAttribute('aria-pressed', String(on)); });
        const copy = h('button', { class: 'icon-btn small', type: 'button' }, icon('copy'));
        tip(copy, 'output.copy');
        copy.addEventListener('click', () => this.app.copyText(this.out.innerText));
        return [wrap, copy, clearBtn];
    }

    focus() { (this.inputRow.hidden ? this.out : this.input).focus(); }

    renderEmpty() {
        clear(this.out);
        this.out.appendChild(h('div', { class: 'term-empty' }, icon('terminal'), L('output.empty')));
        this.empty = true;
    }

    clear() {
        this.pending = [];
        this.chars = 0;
        this.chunk = null;
        this.chunkLen = 0;
        this.parsers = { 1: new AnsiParser(), 2: new AnsiParser() };
        this.trimmedNote = false;
        this.renderEmpty();
    }

    ensureStarted() {
        if (this.empty) { clear(this.out); this.empty = false; }
    }

    header(label, detail) {
        this.flush();
        this.ensureStarted();
        this.chunk = null;
        this.headDetail = h('span', { class: 'term-head-detail' }, detail || '');
        this.out.appendChild(h('div', { class: 'term-head' }, icon('play'), h('span', { class: 'term-head-label' }, label), this.headDetail));
    }

    /* the detail of the last header (the backend the program really runs on) */
    setHeaderDetail(detail) { if (this.headDetail) this.headDetail.textContent = detail; }

    footer({ ok, text, kind }) {
        this.flush();
        this.ensureStarted();
        this.chunk = null;
        this.out.appendChild(h('div', { class: `term-foot ${kind || (ok ? 'ok' : 'bad')}` }, icon(kind === 'stopped' ? 'stop' : kind === 'paused' ? 'pause' : ok ? 'checkCircle' : 'error'), h('span', {}, text)));
        this.scrollToEnd(true);
    }

    note(text, cls = 'note') {
        this.flush();
        this.ensureStarted();
        this.chunk = null;
        this.out.appendChild(h('div', { class: `term-line ${cls}` }, text));
        this.scrollToEnd();
    }

    /* program output; stream 1 = stdout, 2 = stderr, 'echo' = typed input */
    write(text, stream = 1) {
        if (!text) return;
        this.pending.push([text, stream]);
        if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.flush(); });
    }

    flush() {
        if (this.raf) { cancelAnimationFrame(this.raf); this.raf = 0; }
        const items = this.pending;
        if (!items.length) return;
        this.pending = [];
        this.ensureStarted();
        const atBottom = this.out.scrollHeight - this.out.scrollTop - this.out.clientHeight < 40;
        for (const [text, stream] of items) {
            if (stream === 'echo') { this.append([{ text, style: '' }], 'term-echo'); continue; }
            const parser = this.parsers[stream === 2 ? 2 : 1];
            this.append(parser.feed(text), stream === 2 ? 'term-err' : '');
        }
        if (this.chars > MAX_CHARS) this.trim();
        if (atBottom) this.scrollToEnd(true);
    }

    append(segments, cls) {
        for (const seg of segments) {
            let text = seg.text;
            while (text.length) {
                if (!this.chunk || this.chunkLen >= CHUNK) {
                    this.chunk = h('div', { class: 'term-chunk' });
                    this.out.appendChild(this.chunk);
                    this.chunkLen = 0;
                }
                const piece = text.length > CHUNK ? text.slice(0, CHUNK) : text;
                this.chunk.appendChild(this.span(piece, cls, seg.style));
                this.chunkLen += piece.length;
                this.chars += piece.length;
                text = text.slice(piece.length);
            }
        }
    }

    span(text, cls, style) {
        const s = h('span', { class: cls || undefined });
        if (style) s.style.cssText = style;
        // source locations become links
        LOC_RE.lastIndex = 0;
        let last = 0, m;
        while ((m = LOC_RE.exec(text))) {
            let path, line, col;
            if (m[2]) { path = this.resolvePath(m[2]); line = m[3]; col = m[4]; }
            else { path = this.app.runner.currentPath; line = m[5]; }
            if (!path) continue;
            if (m.index > last) s.appendChild(document.createTextNode(text.slice(last, m.index)));
            s.appendChild(h('a', { class: 'term-loc', href: '#', dataset: { path, line, col: col || '1' }, on: { click: (e) => e.preventDefault() } }, m[0]));
            last = LOC_RE.lastIndex;
        }
        if (last < text.length) s.appendChild(document.createTextNode(text.slice(last)));
        return s;
    }

    resolvePath(p) {
        const rel = this.app.relPath(p) || p.replace(/^\.\//, '');
        if (/^[A-Za-z]:|^\//.test(rel)) return null;
        return rel;
    }

    trim() {
        while (this.chars > MAX_CHARS * 0.75 && this.out.firstChild && this.out.firstChild !== this.chunk) {
            this.chars -= this.out.firstChild.textContent.length;
            this.out.removeChild(this.out.firstChild);
        }
        if (!this.trimmedNote) {
            this.trimmedNote = true;
            this.out.insertBefore(h('div', { class: 'term-line note' }, t('output.trimmed')), this.out.firstChild);
        }
    }

    scrollToEnd(force) {
        if (force) this.out.scrollTop = this.out.scrollHeight;
    }

    /* stdin: sink(text) sends one line (with "\n"); eofSink() ends the input */
    attachInput(sink, eofSink) {
        this.sink = sink;
        this.eofSink = eofSink;
        this.inputRow.hidden = false;
        this.inputRow.classList.remove('waiting');
        this.input.placeholder = t('output.stdinPlaceholder');
        this.inputHint.textContent = '';
    }

    waitForInput() {
        this.inputRow.hidden = false;
        this.inputRow.classList.add('waiting');
        this.inputHint.textContent = t('output.waiting');
        this.app.layout.showPanel('output');
        this.input.focus({ preventScroll: true });
        if (document.activeElement !== this.input) requestAnimationFrame(() => this.input.focus({ preventScroll: true }));
    }

    detachInput() {
        this.sink = null;
        this.eofSink = null;
        this.inputRow.hidden = true;
        this.inputRow.classList.remove('waiting');
    }

    submit() {
        if (!this.sink) return;
        const text = this.input.value;
        this.input.value = '';
        if (text) { this.hist.push(text); if (this.hist.length > 100) this.hist.shift(); }
        this.histPos = this.hist.length;
        this.write(text + '\n', 'echo');
        this.flush();
        this.inputRow.classList.remove('waiting');
        this.inputHint.textContent = '';
        this.sink(text + '\n');
    }

    eof() {
        if (!this.eofSink) return;
        const text = this.input.value;
        this.input.value = '';
        if (text) { this.write(text, 'echo'); this.sink(text); }
        this.note(t('output.eof'));
        this.eofSink();
        this.inputRow.classList.remove('waiting');
    }

    history(dir) {
        if (!this.hist.length) return;
        this.histPos = Math.max(0, Math.min(this.hist.length, this.histPos + dir));
        this.input.value = this.hist[this.histPos] || '';
    }

    relocalize() {
        this.input.setAttribute('aria-label', t('output.stdinLabel'));
        if (this.empty) this.renderEmpty();
    }
}
