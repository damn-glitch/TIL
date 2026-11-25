// Search: text search in the workspace (POST /api/search: regex, case,
// glob), results grouped by file, and replace in files (with confirmation).
import { h, clear, debounce } from '../core/dom.js';
import { icon, fileIcon } from '../core/icons.js';
import { t, tn, L } from '../core/i18n.js';
import { basename, dirname } from '../core/paths.js';
import { tip, confirmDialog, toast } from '../ui/widgets.js';

export class SearchView {
    constructor(app) {
        this.app = app;
        this.opts = { regex: false, case: false };
        this.query = h('input', { class: 'input search-input', type: 'text', placeholder: t('search.placeholder'), spellcheck: 'false', 'aria-label': t('search.placeholder') });
        this.replace = h('input', { class: 'input', type: 'text', placeholder: t('search.replacePlaceholder'), spellcheck: 'false', 'aria-label': t('search.replacePlaceholder') });
        this.glob = h('input', { class: 'input', type: 'text', placeholder: t('search.globPlaceholder'), spellcheck: 'false', 'aria-label': t('search.globPlaceholder') });
        this.caseBtn = this.toggle('caseSensitive', 'search.matchCase', 'case');
        this.regexBtn = this.toggle('regex', 'search.useRegex', 'regex');
        this.replaceAllBtn = h('button', { class: 'icon-btn small', type: 'button' }, icon('replaceAll'));
        tip(this.replaceAllBtn, 'search.replaceAll');
        this.replaceAllBtn.addEventListener('click', () => this.replaceAll());
        this.showReplace = false;
        this.replaceToggle = h('button', { class: 'icon-btn small replace-toggle', type: 'button', 'aria-expanded': 'false' }, icon('chevronRight'));
        tip(this.replaceToggle, 'search.toggleReplace');
        this.replaceToggle.addEventListener('click', () => this.setReplace(!this.showReplace));
        this.replaceRow = h('div', { class: 'search-row', hidden: true }, this.replace, this.replaceAllBtn);
        this.summary = h('div', { class: 'search-summary' });
        this.results = h('div', { class: 'search-results', role: 'tree', 'aria-label': t('search.results') });
        const form = h('div', { class: 'search-form' },
            h('div', { class: 'search-main' }, this.replaceToggle,
                h('div', { class: 'search-fields' },
                    h('div', { class: 'search-row with-toggles' }, this.query, h('div', { class: 'search-toggles' }, this.caseBtn, this.regexBtn)),
                    this.replaceRow)),
            h('div', { class: 'search-row glob' }, this.glob));
        this.titleEl = L('activity.search');
        this.titleEl.className = 'side-title';
        this.el = h('div', { class: 'side-view' }, h('div', { class: 'side-head' }, this.titleEl), form, this.summary, h('div', { class: 'side-body' }, this.results));
        const run = debounce(() => this.search(), 300);
        this.query.addEventListener('input', run);
        this.glob.addEventListener('input', run);
        this.query.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); run.flush(); } });
        this.replace.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); this.replaceAll(); } });
        this.matches = [];
        this.seq = 0;
        this.collapsed = new Set();
    }

    toggle(name, key, opt) {
        const b = h('button', { class: 'icon-btn small', type: 'button', 'aria-pressed': 'false' }, icon(name));
        tip(b, key);
        b.addEventListener('click', () => {
            this.opts[opt] = !this.opts[opt];
            b.setAttribute('aria-pressed', String(this.opts[opt]));
            this.search();
        });
        return b;
    }

    setReplace(on) {
        this.showReplace = on;
        this.replaceRow.hidden = !on;
        this.replaceToggle.setAttribute('aria-expanded', String(on));
        this.replaceToggle.replaceChildren(icon(on ? 'chevronDown' : 'chevronRight'));
    }

    focus(text) {
        if (typeof text === 'string' && text) { this.query.value = text; this.search(); }
        this.query.focus();
        this.query.select();
    }

    async search() {
        const query = this.query.value;
        const seq = ++this.seq;
        if (!query) { this.matches = []; this.render(); return; }
        if (this.opts.regex) {
            try { new RegExp(query); } catch (e) { this.summary.textContent = t('search.badRegex'); clear(this.results); return; }
        }
        this.summary.textContent = t('search.searching');
        let res;
        try {
            const body = { query, regex: this.opts.regex, case: this.opts.case };
            if (this.glob.value.trim()) body.glob = this.glob.value.trim();
            res = await this.app.backend.search(body);
        } catch (e) {
            if (seq !== this.seq) return;
            this.summary.textContent = t('search.failed');
            toast(t('search.failed'), { kind: 'error', detail: e.message });
            return;
        }
        if (seq !== this.seq) return;
        this.matches = res.matches;
        this.truncated = res.truncated;
        this.render();
    }

    render() {
        clear(this.results);
        const byFile = new Map();
        for (const m of this.matches) {
            if (!byFile.has(m.path)) byFile.set(m.path, []);
            byFile.get(m.path).push(m);
        }
        if (!this.query.value) { this.summary.textContent = ''; return; }
        if (!this.matches.length) { this.summary.textContent = t('search.noResults'); return; }
        this.summary.textContent = tn('search.summary', this.matches.length, { files: byFile.size }) + (this.truncated ? ` ${t('search.truncated')}` : '');
        const re = this.matcher();
        for (const [path, list] of byFile) {
            const open = !this.collapsed.has(path);
            const head = h('div', { class: 'sr-file', role: 'treeitem', 'aria-expanded': String(open), tabindex: '0' },
                icon(open ? 'chevronDown' : 'chevronRight', { className: 'chev' }), icon(fileIcon(path), { className: 'til-icon' }),
                h('span', { class: 'sr-name' }, basename(path)), h('span', { class: 'sr-dir' }, dirname(path)), h('span', { class: 'badge' }, String(list.length)));
            head.addEventListener('click', () => { if (open) this.collapsed.add(path); else this.collapsed.delete(path); this.render(); });
            this.results.appendChild(head);
            if (!open) continue;
            for (const m of list) {
                const text = String(m.text || '');
                const row = h('div', { class: 'sr-match', role: 'treeitem', tabindex: '-1', title: `${path}:${m.line}` });
                row.appendChild(this.highlight(text, m.col, re, m.length));
                row.appendChild(h('span', { class: 'sr-line' }, String(m.line)));
                row.addEventListener('click', () => this.app.openFile(path, { line: m.line, col: m.col }));
                row.addEventListener('keydown', (e) => { if (e.key === 'Enter') this.app.openFile(path, { line: m.line, col: m.col }); });
                this.results.appendChild(row);
            }
        }
    }

    matcher() {
        const q = this.query.value;
        try {
            return new RegExp(this.opts.regex ? q : q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), this.opts.case ? 'g' : 'gi');
        } catch (e) { return null; }
    }

    /* col: 1-based, in UTF-16 units (= JS string offsets); length: of the match */
    highlight(text, col, re, length) {
        const el = h('code', { class: 'sr-text' });
        const trimmed = text.replace(/^\s+/, '');
        const lead = text.length - trimmed.length;
        let start = Math.max(0, (col || 1) - 1 - lead), len = 0;
        if (typeof length === 'number' && length >= 0) len = length;
        else if (re) {
            re.lastIndex = Math.max(0, (col || 1) - 1);
            const m = re.exec(text);
            if (m && m.index >= 0) { start = m.index - lead; len = m[0].length; }
        }
        const cut = start > 40 ? start - 30 : 0;
        el.append(cut ? '…' + trimmed.slice(cut, start) : trimmed.slice(0, start), h('mark', {}, trimmed.slice(start, start + len)), trimmed.slice(start + len, start + len + 160));
        return el;
    }

    async replaceAll() {
        const re = this.matcher();
        if (!re || !this.matches.length) return;
        const files = [...new Set(this.matches.map((m) => m.path))];
        const ok = await confirmDialog({
            title: t('search.replaceTitle'),
            message: tn('search.replaceMsg', this.matches.length, { files: files.length, text: this.replace.value }),
            confirm: t('search.replaceAll'),
        });
        if (!ok) return;
        const replacement = this.replace.value;
        let changed = 0;
        for (const path of files) {
            const doc = this.app.docs.get(path);
            try {
                if (doc) {
                    const text = doc.current().doc.toString();
                    const out = text.replace(re, this.opts.regex ? replacement : () => replacement);
                    if (out !== text) { doc.replaceText(out); changed++; await this.app.saveDoc(doc); }
                } else {
                    const file = await this.app.backend.readFile(path);
                    const out = file.text.replace(re, this.opts.regex ? replacement : () => replacement);
                    if (out !== file.text) { await this.app.backend.writeFile(path, out, file.mtime); changed++; }
                }
            } catch (e) {
                toast(t('file.cannotSave', { name: basename(path) }), { kind: 'error', detail: e.message });
            }
        }
        this.app.statusbar.flash(tn('search.replaced', changed));
        this.search();
    }

    relocalize() {
        this.query.placeholder = t('search.placeholder');
        this.replace.placeholder = t('search.replacePlaceholder');
        this.glob.placeholder = t('search.globPlaceholder');
        this.render();
    }
}
