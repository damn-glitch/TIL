// Problems: the diagnostics of every file, grouped by file, with a filter;
// a click opens the location. Shown as a side view and as a bottom panel tab.
import { h, clear } from '../core/dom.js';
import { icon, fileIcon } from '../core/icons.js';
import { t, tn, L } from '../core/i18n.js';
import { basename, dirname } from '../core/paths.js';
import { tip } from '../ui/widgets.js';

const SEV = { 1: 'error', 2: 'warning', 3: 'info', 4: 'hint' };

export class ProblemsView {
    constructor(app, { side }) {
        this.app = app;
        this.side = side;
        this.filter = '';
        this.collapsed = new Set();
        this.filterInput = h('input', { class: 'input small-input', type: 'search', placeholder: t('problems.filter'), 'aria-label': t('problems.filter') });
        this.filterInput.addEventListener('input', () => { this.filter = this.filterInput.value.trim().toLowerCase(); this.render(); });
        this.list = h('div', { class: 'problems-list', role: 'tree', 'aria-label': t('panel.problems') });
        this.list.addEventListener('keydown', (e) => this.keys(e));
        if (side) {
            this.titleEl = L('activity.problems');
            this.titleEl.className = 'side-title';
            this.el = h('div', { class: 'side-view problems-view' },
                h('div', { class: 'side-head' }, this.titleEl),
                h('div', { class: 'problems-filter' }, this.filterInput),
                h('div', { class: 'side-body' }, this.list));
        } else {
            this.el = h('div', { class: 'problems-view bottom' }, h('div', { class: 'problems-scroll' }, this.list));
        }
        this.render();
    }

    actions() {
        const collapse = h('button', { class: 'icon-btn small', type: 'button' }, icon('collapseAll'));
        tip(collapse, 'problems.collapse');
        collapse.addEventListener('click', () => {
            for (const uri of this.app.diagnostics.keys()) this.collapsed.add(uri);
            this.render();
        });
        const wrap = h('div', { class: 'panel-filter' }, this.filterInput);
        return [wrap, collapse];
    }

    focus() { (this.list.querySelector('[tabindex="0"]') || this.filterInput).focus(); }

    visible() {
        const L = this.app.layout;
        return this.side ? L.activeSide === 'problems' && L.sideOpen() : L.activePanel === 'problems' && L.panelOpen();
    }

    renderIfVisible() {
        if (this.visible()) this.render(); else this.stale = true;
    }

    onShow() { if (this.stale !== false) this.render(); }

    render() {
        this.stale = false;
        clear(this.list);
        const entries = [...this.app.diagnostics.entries()]
            .map(([uri, list]) => ({ uri, path: this.app.pathOf(uri), list: list.filter((d) => this.matches(d, uri)) }))
            .filter((e) => e.list.length)
            .sort((a, b) => (a.path || a.uri).localeCompare(b.path || b.uri));
        if (!entries.length) {
            this.list.appendChild(h('div', { class: 'empty-state' }, icon('checkCircle', { className: 'empty-icon ok' }),
                h('p', {}, this.filter ? t('problems.noMatch') : t('problems.none'))));
            return;
        }
        let first = true;
        for (const e of entries) {
            const name = e.path !== null ? basename(e.path) : e.uri;
            const open = !this.collapsed.has(e.uri);
            const head = h('div', { class: 'prob-file', role: 'treeitem', 'aria-expanded': String(open), tabindex: first ? '0' : '-1' },
                icon(open ? 'chevronDown' : 'chevronRight', { className: 'chev' }),
                icon(fileIcon(name), { className: 'til-icon' }),
                h('span', { class: 'prob-name' }, name),
                e.path ? h('span', { class: 'prob-dir' }, dirname(e.path)) : null,
                h('span', { class: 'badge' }, String(e.list.length)));
            first = false;
            head.addEventListener('click', () => { if (open) this.collapsed.add(e.uri); else this.collapsed.delete(e.uri); this.render(); });
            this.list.appendChild(head);
            if (!open) continue;
            const sorted = e.list.slice().sort((a, b) => (a.range.start.line - b.range.start.line) || ((a.severity || 1) - (b.severity || 1)));
            if (sorted.length > 500) {
                this.list.appendChild(h('div', { class: 'prob-row muted' }, t('problems.more', { n: sorted.length - 500 })));
                sorted.length = 500;
            }
            for (const d of sorted) {
                const sev = SEV[d.severity || 1] || 'error';
                const msg = String(d.message || '').split('\n');
                const row = h('div', { class: `prob-row sev-${sev}`, role: 'treeitem', tabindex: '-1' },
                    icon(sev === 'hint' ? 'hint' : sev, { className: `sev-${sev}` }),
                    h('span', { class: 'prob-msg' }, msg[0], msg.length > 1 ? h('span', { class: 'prob-hint' }, msg.slice(1).join(' ')) : null),
                    d.code ? h('span', { class: 'prob-code' }, String(typeof d.code === 'object' ? d.code.value : d.code)) : null,
                    h('span', { class: 'prob-loc' }, t('problems.loc', { line: d.range.start.line + 1, col: d.range.start.character + 1 })));
                row.addEventListener('click', () => this.open(e, d));
                row.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); this.open(e, d); } });
                this.list.appendChild(row);
            }
        }
    }

    matches(d, uri) {
        if (!this.filter) return true;
        const path = this.app.pathOf(uri) || uri;
        return `${d.message} ${d.code || ''} ${path}`.toLowerCase().includes(this.filter);
    }

    open(entry, d) {
        if (entry.path === null) return;
        this.app.openLocation(entry.path, d.range);
    }

    keys(e) {
        const rows = [...this.list.querySelectorAll('[role="treeitem"]')];
        const i = rows.indexOf(document.activeElement);
        if (i < 0) return;
        let j = -1;
        if (e.key === 'ArrowDown') j = Math.min(rows.length - 1, i + 1);
        else if (e.key === 'ArrowUp') j = Math.max(0, i - 1);
        else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); rows[i].click(); return; }
        if (j >= 0) {
            e.preventDefault();
            rows.forEach((r) => { r.tabIndex = -1; });
            rows[j].tabIndex = 0;
            rows[j].focus();
        }
    }

    summary() {
        const { errors, warnings } = this.app.diagnosticTotals();
        return tn('problems.count', errors + warnings);
    }

    relocalize() {
        this.filterInput.placeholder = t('problems.filter');
        this.render();
    }
}
