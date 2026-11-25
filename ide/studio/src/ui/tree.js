// A keyboard-accessible tree (ARIA tree pattern) used by the explorer, the
// variables view and the outline: arrows move/expand/collapse, Enter
// activates, Home/End, type-ahead; rows render through a callback.
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';

export class TreeView {
    constructor({ label, getChildren, render, key, expandable, onActivate, onContextMenu, onKey, indent = 12, base = 8, className = '' }) {
        this.opts = { getChildren, render, key, expandable, onActivate, onContextMenu, onKey, indent, base };
        this.el = h('div', { class: `tree ${className}`, role: 'tree', tabindex: '0', 'aria-label': label });
        this.roots = [];
        this.expanded = new Set();
        this.children = new Map();      // key -> nodes (loaded)
        this.loading = new Map();
        this.selected = null;
        this.focusedKey = null;
        this.rows = [];                 // visible: {node, level, key, parentKey}
        this.el.addEventListener('keydown', (e) => this.keydown(e));
        this.el.addEventListener('focus', () => { if (!this.focusedKey && this.rows.length) this.setFocus(this.rows[0].key); else this.markFocus(); });
        this.typed = '';
        this.typedTimer = 0;
    }

    setRoots(nodes) {
        this.roots = nodes;
        this.render();
    }

    isExpanded(k) { return this.expanded.has(k); }

    async loadChildren(node) {
        const k = this.opts.key(node);
        if (this.children.has(k)) return this.children.get(k);
        if (this.loading.has(k)) return this.loading.get(k);
        const p = Promise.resolve(this.opts.getChildren(node)).then((list) => {
            this.children.set(k, list || []);
            this.loading.delete(k);
            return list || [];
        }, (e) => { this.loading.delete(k); this.children.set(k, []); throw e; });
        this.loading.set(k, p);
        return p;
    }

    async expand(node, { render = true } = {}) {
        const k = this.opts.key(node);
        this.expanded.add(k);
        try { await this.loadChildren(node); } catch (e) { /* shown as empty */ }
        if (render) this.render();
    }

    collapse(node) {
        this.expanded.delete(this.opts.key(node));
        this.render();
    }

    async toggle(node) {
        if (this.expanded.has(this.opts.key(node))) this.collapse(node);
        else await this.expand(node);
    }

    /* forget loaded children (all, or of one node) and reload expanded ones */
    async refresh(nodeKey) {
        if (nodeKey === undefined) this.children.clear();
        else this.children.delete(nodeKey);
        await this.reloadExpanded(this.roots);
        this.render();
    }

    async reloadExpanded(nodes) {
        for (const n of nodes) {
            const k = this.opts.key(n);
            if (this.expanded.has(k) && this.opts.expandable(n)) {
                let kids = [];
                try { kids = await this.loadChildren(n); } catch (e) { kids = []; }
                await this.reloadExpanded(kids);
            }
        }
    }

    render() {
        const prevScroll = this.el.scrollTop;
        clear(this.el);
        this.rows = [];
        const walk = (nodes, level, parentKey) => {
            nodes.forEach((node, i) => {
                const k = this.opts.key(node);
                this.rows.push({ node, level, key: k, parentKey, setsize: nodes.length, pos: i + 1 });
                if (this.opts.expandable(node) && this.expanded.has(k) && this.children.has(k)) walk(this.children.get(k), level + 1, k);
            });
        };
        walk(this.roots, 1, null);
        const frag = document.createDocumentFragment();
        for (const r of this.rows) frag.appendChild(this.renderRow(r));
        this.el.appendChild(frag);
        this.el.scrollTop = prevScroll;
        if (this.focusedKey && !this.rows.some((r) => r.key === this.focusedKey)) this.focusedKey = this.rows.length ? this.rows[0].key : null;
        this.markFocus();
    }

    renderRow(r) {
        const exp = this.opts.expandable(r.node);
        const open = exp && this.expanded.has(r.key);
        const row = h('div', {
            class: ['tree-row', r.key === this.selected && 'selected'],
            role: 'treeitem',
            'aria-level': String(r.level),
            'aria-setsize': String(r.setsize),
            'aria-posinset': String(r.pos),
            'aria-expanded': exp ? String(open) : undefined,
            'aria-selected': String(r.key === this.selected),
            id: this.rowId(r.key),
            dataset: { key: r.key },
            style: { paddingLeft: `${this.opts.base + (r.level - 1) * this.opts.indent}px` },
        });
        for (let l = 1; l < r.level; l++) row.appendChild(h('span', { class: 'tree-guide', style: { left: `${this.opts.base + (l - 1) * this.opts.indent + 7}px` } }));
        const twist = h('span', { class: ['tree-twist', open && 'open'] }, exp ? icon('chevronRight') : null);
        row.appendChild(twist);
        const content = h('span', { class: 'tree-content' });
        this.opts.render(r.node, content, { open, level: r.level });
        row.appendChild(content);
        row.addEventListener('mousedown', (e) => { if (e.button === 0 && !e.target.closest('input')) e.preventDefault(); });
        row.addEventListener('click', (e) => {
            if (e.target.closest('input, .tree-action')) return;
            this.select(r.key);
            this.setFocus(r.key);
            this.el.focus({ preventScroll: true });
            if (exp && (e.target.closest('.tree-twist') || !this.opts.onActivate || this.opts.activateToggles !== false)) this.toggle(r.node);
            if (this.opts.onActivate) this.opts.onActivate(r.node, e);
        });
        row.addEventListener('contextmenu', (e) => {
            if (!this.opts.onContextMenu) return;
            e.preventDefault();
            this.select(r.key);
            this.setFocus(r.key);
            this.opts.onContextMenu(r.node, e);
        });
        return row;
    }

    rowId(key) { return 'tr-' + String(key).replace(/[^\w-]/g, (c) => '_' + c.charCodeAt(0).toString(16)); }

    select(key) {
        this.selected = key;
        for (const el of this.el.querySelectorAll('.tree-row')) {
            const on = el.dataset.key === key;
            el.classList.toggle('selected', on);
            el.setAttribute('aria-selected', String(on));
        }
    }

    setFocus(key) {
        this.focusedKey = key;
        this.markFocus();
    }

    markFocus() {
        for (const el of this.el.querySelectorAll('.tree-row.focused')) el.classList.remove('focused');
        if (!this.focusedKey) { this.el.removeAttribute('aria-activedescendant'); return; }
        const el = this.el.querySelector(`#${CSS.escape(this.rowId(this.focusedKey))}`);
        if (el) {
            el.classList.add('focused');
            this.el.setAttribute('aria-activedescendant', el.id);
            const top = el.offsetTop, bottom = top + el.offsetHeight;
            if (top < this.el.scrollTop) this.el.scrollTop = top;
            else if (bottom > this.el.scrollTop + this.el.clientHeight) this.el.scrollTop = bottom - this.el.clientHeight;
        }
    }

    rowOf(key) { return this.rows.find((r) => r.key === key) || null; }

    keydown(e) {
        if (e.target.closest && e.target.closest('input')) return;
        const i = this.rows.findIndex((r) => r.key === this.focusedKey);
        const cur = this.rows[i];
        const move = (j) => { if (this.rows[j]) { this.setFocus(this.rows[j].key); this.select(this.rows[j].key); } };
        if (this.opts.onKey && cur && this.opts.onKey(e, cur.node)) return;
        switch (e.key) {
        case 'ArrowDown': e.preventDefault(); move(Math.min(this.rows.length - 1, i + 1)); break;
        case 'ArrowUp': e.preventDefault(); move(Math.max(0, i - 1)); break;
        case 'Home': e.preventDefault(); move(0); break;
        case 'End': e.preventDefault(); move(this.rows.length - 1); break;
        case 'ArrowRight':
            e.preventDefault();
            if (cur && this.opts.expandable(cur.node)) {
                if (!this.expanded.has(cur.key)) this.expand(cur.node);
                else move(i + 1);
            }
            break;
        case 'ArrowLeft':
            e.preventDefault();
            if (cur) {
                if (this.opts.expandable(cur.node) && this.expanded.has(cur.key)) this.collapse(cur.node);
                else if (cur.parentKey !== null) move(this.rows.findIndex((r) => r.key === cur.parentKey));
            }
            break;
        case 'Enter':
        case ' ':
            e.preventDefault();
            if (cur) {
                this.select(cur.key);
                if (this.opts.expandable(cur.node) && (e.key === ' ' || !this.opts.onActivate)) this.toggle(cur.node);
                else if (this.opts.onActivate) { if (this.opts.expandable(cur.node)) this.toggle(cur.node); this.opts.onActivate(cur.node, e); }
            }
            break;
        default:
            if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey && /\S/.test(e.key)) {
                clearTimeout(this.typedTimer);
                this.typed += e.key.toLowerCase();
                this.typedTimer = setTimeout(() => { this.typed = ''; }, 700);
                const n = this.rows.length;
                for (let k = 0; k < n; k++) {
                    const r = this.rows[(i + (this.typed.length === 1 ? 1 : 0) + k) % n];
                    const text = (r.node.name || r.node.label || '').toLowerCase();
                    if (text.startsWith(this.typed)) { this.setFocus(r.key); this.select(r.key); break; }
                }
            }
        }
    }
}
