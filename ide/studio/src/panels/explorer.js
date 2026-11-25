// Explorer: the project tree (new file / folder, rename, delete with
// confirmation, copy path, run), synced with the active editor and with
// changes on disk.
import { h } from '../core/dom.js';
import { icon, fileIcon } from '../core/icons.js';
import { t, L, setText } from '../core/i18n.js';
import { TreeView } from '../ui/tree.js';
import { openMenu, tip, confirmDialog, toast } from '../ui/widgets.js';
import { compareNames, dirname, join, basename, isTil } from '../core/paths.js';

const HIDDEN = /^(\.git|__pycache__|\.DS_Store|node_modules)$|^\.studio-debug-/;

export class ExplorerView {
    constructor(app) {
        this.app = app;
        this.titleEl = h('span', { class: 'side-title' });
        const btn = (name, key, run, keys) => {
            const b = h('button', { class: 'icon-btn small', type: 'button' }, icon(name));
            tip(b, key, keys);
            b.addEventListener('click', run);
            return b;
        };
        this.actions = h('div', { class: 'side-actions' },
            btn('newFile', 'explorer.newFile', () => this.startCreate('file')),
            btn('newFolder', 'explorer.newFolder', () => this.startCreate('dir')),
            btn('refresh', 'explorer.refresh', () => this.refresh()),
            btn('collapseAll', 'explorer.collapse', () => this.collapseAll()));
        this.tree = new TreeView({
            label: t('activity.explorer'),
            className: 'explorer-tree',
            key: (n) => n.key || n.path,
            expandable: (n) => n.type === 'dir',
            getChildren: (n) => this.list(n.path),
            render: (n, el, info) => this.renderNode(n, el, info),
            onActivate: (n, e) => { if (n.type === 'file') this.app.openFile(n.path, { focus: e.type === 'keydown' || e.detail > 1 }); },
            onContextMenu: (n, e) => this.menu(n, e),
            onKey: (e, n) => {
                if (e.key === 'F2') { e.preventDefault(); this.startRename(n); return true; }
                if (e.key === 'Delete') { e.preventDefault(); this.remove(n); return true; }
                return false;
            },
        });
        this.tree.opts.activateToggles = true;
        this.empty = h('div', { class: 'empty-state small', hidden: true },
            L('explorer.empty', null, 'p'),
            h('button', { class: 'btn small', type: 'button', on: { click: () => this.startCreate('file') } }, icon('newFile'), L('explorer.newFile')));
        this.body = h('div', { class: 'side-body explorer-body' }, this.tree.el, this.empty);
        this.body.addEventListener('contextmenu', (e) => {
            if (e.target.closest('.tree-row')) return;
            e.preventDefault();
            this.menu({ type: 'dir', path: '', name: '', root: true }, e);
        });
        this.el = h('div', { class: 'side-view' }, h('div', { class: 'side-head' }, this.titleEl, this.actions), this.body);
        this.pending = null;      // inline input: {mode, parent, node}
        this.setTitle();
    }

    setTitle() {
        this.titleEl.textContent = this.app.info ? this.app.info.rootName : t('activity.explorer');
    }

    focus() { this.tree.el.focus(); }

    async start() {
        await this.loadRoot();
    }

    async list(path) {
        let entries;
        try {
            entries = await this.app.backend.tree(path);
        } catch (e) {
            toast(t('explorer.cannotList', { path: path || '/' }), { kind: 'error', detail: e.message });
            return [];
        }
        const nodes = entries.filter((e) => e && e.name && !HIDDEN.test(e.name)).map((e) => ({
            name: e.name, path: e.path || join(path, e.name), type: e.type === 'dir' ? 'dir' : 'file', size: e.size, mtime: e.mtime,
            collapsed: Boolean(e.collapsed), link: Boolean(e.link), readonly: Boolean(e.readonly),
        }));
        nodes.sort((a, b) => (a.type === b.type ? compareNames(a.name, b.name) : a.type === 'dir' ? -1 : 1));
        if (this.pending && this.pending.mode !== 'rename' && this.pending.parent === path) {
            nodes.unshift({ key: '__input__', type: 'input', name: '', path: join(path, '__new__'), parent: path });
        }
        return nodes;
    }

    async loadRoot() {
        const roots = await this.list('');
        this.tree.children.clear();
        this.tree.setRoots(roots);
        this.empty.hidden = roots.length > 0;
        this.app.fileIndex.invalidate();
    }

    async refresh(dir) {
        // while a name is being typed in the tree, redrawing it would replace
        // the box: catch up once the box is done
        if (this.pending) { this.deferred = true; return; }
        if (dir === undefined || dir === '') {
            const roots = await this.list('');
            this.tree.roots = roots;
            await this.tree.refresh();
            this.empty.hidden = roots.length > 0;
        } else {
            await this.tree.refresh(dir);
        }
        this.app.fileIndex.invalidate();
        this.markActive();
    }

    collapseAll() {
        this.tree.expanded.clear();
        this.tree.render();
    }

    renderNode(n, el, { open }) {
        if (n.type === 'input' || (this.pending && this.pending.mode === 'rename' && this.pending.node && this.pending.node.path === n.path)) {
            const mode = n.type === 'input' ? this.pending.mode : 'rename';
            const pend = this.pending;
            const again = pend.value !== undefined;     // the tree was redrawn while typing
            const input = h('input', {
                class: 'tree-input', type: 'text', spellcheck: 'false', autocomplete: 'off',
                value: again ? pend.value : mode === 'rename' ? n.name : '', 'aria-label': t(mode === 'rename' ? 'explorer.rename' : mode === 'dir' ? 'explorer.newFolder' : 'explorer.newFile'),
                placeholder: mode === 'dir' ? t('explorer.folderName') : mode === 'rename' ? '' : t('explorer.fileName'),
            });
            const err = h('div', { class: 'tree-input-error', role: 'alert' });
            el.append(icon(mode === 'dir' ? 'folder' : fileIcon(input.value || 'a.til')), h('span', { class: 'tree-input-wrap' }, input, err));
            let done = false;
            const finish = async (commit) => {
                if (done) return;
                const name = input.value.trim();
                if (commit) {
                    const msg = this.validateName(name, mode === 'rename' ? dirname(n.path) : this.pending.parent, mode === 'rename' ? n.name : null);
                    if (msg) { err.textContent = msg; input.focus(); return; }
                }
                done = true;
                const pend = this.pending;
                this.pending = null;
                if (commit && name && !(mode === 'rename' && name === n.name)) {
                    if (mode === 'rename') await this.app.renamePath(n.path, join(dirname(n.path), name));
                    else await this.app.createPath(join(pend.parent, name), mode === 'dir' ? 'dir' : 'file');
                }
                const all = this.deferred;
                this.deferred = false;
                await this.refresh(all ? undefined : mode === 'rename' ? dirname(n.path) : pend.parent);
                this.tree.el.focus({ preventScroll: true });
            };
            input.addEventListener('keydown', (e) => {
                e.stopPropagation();
                if (e.key === 'Enter') { e.preventDefault(); finish(true); }
                else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
            });
            input.addEventListener('input', () => { err.textContent = ''; pend.value = input.value; });
            // leaving the box commits it; a box replaced by a redraw does not
            input.addEventListener('blur', () => setTimeout(() => { if (input.isConnected) finish(Boolean(input.value.trim())); }, 100));
            // startCreate/startRename focus and select the first box; a redrawn
            // one gets the focus back with the text typed so far
            if (again) {
                requestAnimationFrame(() => {
                    const a = document.activeElement;
                    if (input.isConnected && (!a || a === document.body || this.tree.el.contains(a))) {
                        input.focus();
                        input.setSelectionRange(input.value.length, input.value.length);
                    }
                });
            }
            return;
        }
        const isDir = n.type === 'dir';
        const ic = icon(isDir ? (open ? 'folderOpen' : 'folder') : fileIcon(n.name), { className: isDir ? 'dir-icon' : isTil(n.name) ? 'til-icon' : 'file-icon' });
        const label = h('span', { class: 'tree-label' }, n.name);
        el.append(ic, label);
        // dependency folders (til_modules), symbolic links, read-only entries
        const notes = [];
        if (n.collapsed) { el.classList.add('dim'); notes.push(t('explorer.collapsedDir')); }
        if (n.link) { el.appendChild(icon('link', { className: 'tree-flag' })); notes.push(t('explorer.link')); }
        if (n.readonly) { el.appendChild(icon('lock', { className: 'tree-flag' })); notes.push(t('file.readOnly')); }
        if (notes.length) el.title = `${n.path} · ${notes.join(' · ')}`;
        const doc = this.app.docs.get(n.path);
        if (doc && doc.dirty) el.appendChild(h('span', { class: 'tree-dirty', title: t('explorer.unsaved') }));
        const diag = !isDir ? this.app.diagnosticSummary(this.app.uriOf(n.path)) : null;
        if (diag && (diag.errors || diag.warnings)) {
            label.classList.add(diag.errors ? 'has-errors' : 'has-warnings');
            el.appendChild(h('span', { class: `tree-count ${diag.errors ? 'error' : 'warning'}` }, String(diag.errors || diag.warnings)));
        }
    }

    validateName(name, parent, current) {
        if (!name) return t('explorer.nameRequired');
        if (/[\\/:*?"<>|]/.test(name) || name === '.' || name === '..') return t('explorer.nameInvalid');
        if (name === current) return '';
        const siblings = parent === '' ? this.tree.roots : this.tree.children.get(parent) || [];
        if (siblings.some((s) => s.type !== 'input' && s.name.toLowerCase() === name.toLowerCase())) return t('explorer.nameExists', { name });
        return '';
    }

    /* the folder new files go to: the selected folder, the folder of the selected file, or the root */
    targetDir() {
        const sel = this.tree.selected ? this.tree.rowOf(this.tree.selected) : null;
        if (!sel) return '';
        return sel.node.type === 'dir' ? sel.node.path : dirname(sel.node.path);
    }

    async startCreate(mode, parent) {
        this.app.layout.showSide('explorer', { focus: false });
        const dir = parent !== undefined ? parent : this.targetDir();
        this.pending = { mode, parent: dir };
        if (dir) {
            const node = this.findNode(dir) || { type: 'dir', path: dir, name: basename(dir) };
            this.tree.expanded.add(dir);
            this.tree.children.delete(dir);
            await this.tree.loadChildren(node);
            this.tree.render();
        } else {
            this.tree.roots = await this.list('');
            this.tree.render();
        }
        this.empty.hidden = true;
        this.focusInput();
        if (this.pending && this.pending.value === undefined) this.pending.value = '';
    }

    focusInput() {
        const input = this.tree.el.querySelector('.tree-input');
        if (input) { input.focus(); input.scrollIntoView({ block: 'nearest' }); }
    }

    startRename(n) {
        if (!n || n.type === 'input' || n.root) return;
        this.pending = { mode: 'rename', node: n };
        this.tree.render();
        const input = this.tree.el.querySelector('.tree-input');
        if (input) {
            this.pending.value = input.value;
            input.focus();
            const dot = n.name.lastIndexOf('.');
            input.setSelectionRange(0, dot > 0 && n.type === 'file' ? dot : n.name.length);
        }
    }

    findNode(path) {
        if (path === '') return null;
        const look = (nodes) => {
            for (const n of nodes) {
                if (n.path === path) return n;
                if (n.type === 'dir' && this.tree.children.has(n.path)) {
                    const f = look(this.tree.children.get(n.path));
                    if (f) return f;
                }
            }
            return null;
        };
        return look(this.tree.roots);
    }

    async remove(n) {
        if (!n || n.root || n.type === 'input') return;
        const ok = await confirmDialog({
            title: t(n.type === 'dir' ? 'explorer.deleteFolderTitle' : 'explorer.deleteFileTitle'),
            message: t(n.type === 'dir' ? 'explorer.deleteFolderMsg' : 'explorer.deleteFileMsg', { name: n.name }),
            detail: t('explorer.deleteDetail'),
            confirm: t('explorer.delete'),
            danger: true,
        });
        if (!ok) return;
        await this.app.deletePath(n.path);
        await this.refresh(dirname(n.path));
    }

    menu(n, e) {
        const isFile = n.type === 'file';
        const dir = n.type === 'dir' ? n.path : dirname(n.path);
        openMenu([
            isFile && isTil(n.name) ? { label: t('cmd.runFile'), icon: 'play', run: () => this.app.runFile(n.path) } : null,
            isFile && isTil(n.name) && this.app.caps.debugger ? { label: t('cmd.debugFile'), icon: 'bug', run: () => this.app.debugFile(n.path) } : null,
            isFile && isTil(n.name) ? { label: t('explorer.translate'), icon: 'translate', submenu: this.app.translateMenu(n.path) } : null,
            isFile && isTil(n.name) ? { separator: true } : null,
            isFile ? { label: t('explorer.open'), run: () => this.app.openFile(n.path) } : null,
            isFile ? { label: t('editor.openToSide'), icon: 'split', run: () => this.app.openFile(n.path, { side: true }) } : null,
            { label: t('explorer.newFile'), icon: 'newFile', run: () => this.startCreate('file', dir) },
            { label: t('explorer.newFolder'), icon: 'newFolder', run: () => this.startCreate('dir', dir) },
            n.root ? null : { separator: true },
            n.root ? null : { label: t('explorer.rename'), icon: 'edit', keys: 'F2', run: () => this.startRename(n) },
            n.root ? null : { label: t('explorer.delete'), icon: 'trash', keys: 'Delete', danger: true, run: () => this.remove(n) },
            { separator: true },
            { label: t('explorer.copyPath'), icon: 'copy', run: () => this.app.copyText(n.path || '.') },
            n.root ? { label: t('explorer.refresh'), icon: 'refresh', run: () => this.refresh() } : null,
        ], { x: e.clientX, y: e.clientY });
    }

    /* select the active editor's file (expanding its folders) */
    async reveal(path, { focus = false } = {}) {
        if (!path || path.includes(':') || path.startsWith('@') || this.pending) return;
        const parts = path.split('/');
        let cur = '';
        for (let i = 0; i < parts.length - 1; i++) {
            cur = cur ? `${cur}/${parts[i]}` : parts[i];
            if (!this.tree.expanded.has(cur)) {
                this.tree.expanded.add(cur);
                try { await this.tree.loadChildren({ type: 'dir', path: cur }); } catch (e) { /* */ }
            }
        }
        this.tree.render();
        this.tree.select(path);
        this.tree.setFocus(path);
        if (focus) this.tree.el.focus();
    }

    markActive() {
        const doc = this.app.editors.activeDoc();
        if (doc && doc.kind === 'file') {
            this.tree.select(doc.path);
            this.tree.focusedKey = doc.path;
            this.tree.markFocus();
        }
    }

    rerender() {
        if (this.pending) { this.deferred = true; return; }
        this.tree.render();
    }
}

export function setTextOf(el, key) { setText(el, key); }
