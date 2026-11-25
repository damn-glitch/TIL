// TIL Studio: the application object that wires the backend (desktop server
// or browser), the language server client, documents, editors, panels,
// commands and the persisted state together.
import { h, debounce, isMac } from './core/dom.js';
import { icon } from './core/icons.js';
import { t, getLocale, setLocale, onLocale, applyI18n, LOCALES, LOCALE_NAMES, detectLocale } from './core/i18n.js';
import { settings, loadProject, saveProject, load, save } from './core/store.js';
import { Commands, formatKeys } from './core/commands.js';
import { rootToUri, pathToUri, uriToPath, basename, dirname, isTil, join } from './core/paths.js';
import { LspClient } from './lsp/lsp.js';
import { DocumentManager } from './editor/documents.js';
import { bpEffect, setDebugLine } from './editor/gutters.js';
import { Layout } from './ui/layout.js';
import { EditorArea } from './ui/editor_area.js';
import { installTips, refreshTips, toast, confirmDialog, customDialog, openMenu, closeMenus, notifications } from './ui/widgets.js';
import { closeQuickPick } from './ui/palette.js';
import { ExplorerView } from './panels/explorer.js';
import { SearchView } from './panels/search.js';
import { DebugView } from './panels/debug.js';
import { ProblemsView } from './panels/problems.js';
import { ProofsView } from './panels/proofs.js';
import { DevicesView } from './panels/devices.js';
import { PackagesView } from './panels/packages.js';
import { BytecodeView } from './panels/bytecode.js';
import { SettingsView } from './panels/settings.js';
import { OutputView } from './panels/output.js';
import { ScreenView } from './panels/screen.js';
import { DebugConsoleView } from './panels/debugconsole.js';
import { Runner } from './run/runner.js';
import { Debugger } from './debug/debugger.js';
import { registerCommands, buildMenubar } from './app_commands.js';
import { LspActions } from './lsp/actions.js';
import { StatusBar } from './ui/statusbar.js';
import { WelcomePage } from './views/welcome.js';
import { TranslateView } from './views/translate.js';
import { TABLES } from './editor/til_lang.js';

export class FileIndex {
    constructor(app) { this.app = app; this.files = null; this.loading = null; }
    invalidate() { this.files = null; }
    async all() {
        if (this.files) return this.files;
        if (this.loading) return this.loading;
        this.loading = (async () => {
            const out = [];
            const walk = async (dir, depth) => {
                if (depth > 12 || out.length > 8000) return;
                let entries = [];
                try { entries = await this.app.backend.tree(dir); } catch (e) { return; }
                for (const e of entries) {
                    if (!e || !e.name || /^(\.git|__pycache__|node_modules)$|^\.studio-debug-/.test(e.name)) continue;
                    const p = e.path || join(dir, e.name);
                    if (e.type === 'dir') { if (e.name !== 'til_modules') await walk(p, depth + 1); } else out.push(p);
                }
            };
            await walk('', 0);
            this.files = out;
            this.loading = null;
            return out;
        })();
        return this.loading;
    }
}

export class App {
    constructor({ backend, info }) {
        this.backend = backend;
        this.info = info;
        this.caps = info.capabilities;
        this.web = backend.kind === 'web';
        this.settings = settings;
        this.commands = new Commands();
        this.rootUri = info.rootUri || backend.rootUri || rootToUri(info.root || '/');
        this.projectKey = info.projectKey || info.root || info.rootName || 'default';
        this.project = loadProject(this.projectKey);
        this.breakpoints = new Map(Object.entries(this.project.breakpoints || {}));
        this.diagnostics = new Map();
        this.fileIndex = new FileIndex(this);
        this.saveLayout = debounce(() => this.persist(), 400);
        this.tables = TABLES;
    }

    /* ── start-up ─────────────────────────────────────────────────── */
    async start(mountEl) {
        const s = this.settings;
        this.applyTheme();
        setLocale(s.get('uiLang') || this.info.lang && LOCALES.includes(this.info.lang) && this.info.lang || detectLocale());
        if (getLocale() === 'en') document.documentElement.lang = 'en';

        this.lsp = new LspClient(this.backend, { rootUri: this.rootUri, rootName: this.info.rootName, locale: getLocale(), version: this.info.version });
        this.docs = new DocumentManager(this);
        this.layout = new Layout(this);
        this.editors = new EditorArea(this, this.layout.editorArea);
        this.lspActions = new LspActions(this);
        this.runner = new Runner(this);
        this.debugger = new Debugger(this);

        /* side views */
        this.views = {
            explorer: new ExplorerView(this),
            search: new SearchView(this),
            debug: new DebugView(this),
            problems: new ProblemsView(this, { side: true }),
            proofs: new ProofsView(this),
            devices: new DevicesView(this),
            packages: new PackagesView(this),
            bytecode: new BytecodeView(this),
            settings: new SettingsView(this),
        };
        const L = this.layout;
        L.addActivity('explorer', { iconName: 'files', titleKey: 'activity.explorer', keys: 'Mod+Shift+E', view: this.views.explorer });
        L.addActivity('search', { iconName: 'search', titleKey: 'activity.search', keys: 'Mod+Shift+F', view: this.views.search });
        L.addActivity('debug', { iconName: 'bug', titleKey: 'activity.debug', keys: 'Mod+Shift+D', view: this.views.debug });
        L.addActivity('problems', { iconName: 'problems', titleKey: 'activity.problems', keys: 'Mod+Shift+M', view: this.views.problems });
        L.addActivity('proofs', { iconName: 'proofs', titleKey: 'activity.proofs', view: this.views.proofs });
        L.addActivity('devices', { iconName: 'devices', titleKey: 'activity.devices', view: this.views.devices });
        L.addActivity('packages', { iconName: 'packages', titleKey: 'activity.packages', view: this.views.packages });
        L.addActivity('bytecode', { iconName: 'bytecode', titleKey: 'activity.bytecode', view: this.views.bytecode });
        L.addActivity('settings', { iconName: 'settings', titleKey: 'activity.settings', keys: 'Mod+,', bottom: true, view: this.views.settings });

        /* bottom panels */
        this.output = new OutputView(this);
        this.bottomProblems = new ProblemsView(this, { side: false });
        this.debugConsole = new DebugConsoleView(this);
        this.screen = new ScreenView(this);
        L.addPanel('output', { titleKey: 'panel.output', view: this.output });
        L.addPanel('problems', { titleKey: 'panel.problems', view: this.bottomProblems });
        L.addPanel('console', { titleKey: 'panel.debugConsole', view: this.debugConsole });
        L.addPanel('screen', { titleKey: 'panel.screen', view: this.screen });

        this.statusbar = new StatusBar(this);
        registerCommands(this);
        buildMenubar(this);
        this.installKeys();
        installTips();
        L.mount(mountEl);
        L.projectLabel.textContent = this.info.rootName;
        L.searchKbd.textContent = formatKeys(this.commands.keysFor('workbench.quickOpen')[0] || 'Mod+P');
        document.title = `${this.info.rootName} — TIL Studio`;

        this.wireEvents();
        this.restoreLayout();
        this.updateTitlebar();
        this.statusbar.update();

        // the explorer and the language server load in parallel; the UI is usable at once
        const explorerP = this.views.explorer.start();
        // an installation without the language server (capabilities.lsp false) edits without it
        const lspP = this.caps.lsp === false ? Promise.resolve() : this.lsp.start().catch((e) => {
            this.lspError = e;
            toast(t('lsp.failed'), { kind: 'warning', detail: e && e.message });
        });
        this.remember();
        await explorerP;
        await this.restoreTabs();
        if (!this.editors.openDocs().length && this.settings.get('welcomeOnStart')) this.showWelcome({ focus: false });
        lspP.then(() => this.statusbar.update());
        onLocale(() => this.onLocaleChanged());
        window.addEventListener('beforeunload', (e) => {
            this.persist();
            if (this.docs.dirtyDocs().length && !this.web) { e.preventDefault(); e.returnValue = ''; }
        });
        // a page that goes away (reload, close) stops its program: the server
        // runs at most 4 at a time, and nobody could stop an orphan
        window.addEventListener('pagehide', () => {
            const run = this.runner.current;
            if (run && !run.done && run.id !== null && this.backend.runStopOnUnload) this.backend.runStopOnUnload(run.id);
            // and its debug session (else the server closes it a minute after the page is gone)
            const dap = this.debugger && this.debugger.active ? this.debugger.client : null;
            if (dap && dap.sessionId && !dap.closed && this.backend.dapDisconnectOnUnload) this.backend.dapDisconnectOnUnload(dap.sessionId, dap.seq++);
        });
        window.addEventListener('resize', () => this.editorsResized());
        this.ready = true;
        window.__studioReadyMs = Math.round(performance.now());
        document.body.dataset.ready = 'true';
        if (window.__studioReadyCallbacks) for (const cb of window.__studioReadyCallbacks) cb(this);
        return this;
    }

    wireEvents() {
        this.lsp.on('diagnostics', (p) => this.onDiagnostics(p));
        this.lsp.on('message', (p) => {
            if (!p || !p.message) return;
            toast(p.message, { kind: p.type === 1 ? 'error' : p.type === 2 ? 'warning' : 'info' });
        });
        this.lsp.on('applyEdit', (edit, done) => { this.lspActions.applyWorkspaceEdit(edit).then(() => done(true), () => done(false)); });
        this.docs.on('dirty', (doc) => this.onDirty(doc));
        this.docs.on('cursor', (doc, view) => this.onCursor(doc, view));
        this.docs.on('change', (doc) => this.onDocChange(doc));
        this.docs.on('breakpoints', (doc, list) => this.onBreakpoints(doc, list));
        this.docs.on('conflict', (doc) => toast(t('file.changedOnDisk', { name: doc.name }), {
            kind: 'warning', actions: [
                { label: t('file.reload'), run: () => this.docs.reload(doc) },
                { label: t('file.keepMine'), run: () => { doc.diskConflict = false; } },
            ],
        }));
        this.docs.on('reloaded', (doc) => this.editors.renderAllTabs());
        this.backend.on('fs', (ev) => this.onFsChanged(ev));
        this.backend.on('connection', (on) => this.statusbar.setConnection(on));
        this.backend.on('auth', () => {
            if (this.authWarned) return;
            this.authWarned = true;
            toast(t('server.forbidden'), { kind: 'error', timeout: 0 });
        });
        this.settings.on('change', (key, value) => this.onSetting(key, value));
        this.commands.on('error', (e, id) => {
            if (e && e.notAvailable) { this.showDesktopOnly(e.feature); return; }
            console.error(`[studio] command ${id} failed`, e);
            toast(t('command.failed', { name: this.commands.title(id) }), { kind: 'error', detail: e && e.message });
        });
        notifications.on('change', () => this.statusbar.updateBell());
    }

    installKeys() {
        window.addEventListener('keydown', (e) => {
            if (e.defaultPrevented || e.isComposing) return;
            const id = this.commands.match(e);
            if (!id) return;
            const cmd = this.commands.get(id);
            // inside text fields only commands with a modifier or function key run
            const inField = e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) && !e.target.closest('.cm-editor');
            if (inField && !(e.ctrlKey || e.metaKey || e.altKey || /^F\d+$/.test(e.key)) && !cmd.inFields) return;
            e.preventDefault();
            e.stopPropagation();
            this.commands.run(id);
        }, true);
    }

    /* ── persistence ─────────────────────────────────────────────── */
    persist() {
        if (!this.layout) return;
        const st = this.project;
        st.layout = {
            side: this.layout.activeSide, sideOpen: this.layout.sideOpen(), panel: this.layout.activePanel,
            panelOpen: this.layout.panelOpen(), sizes: this.layout.sizes(),
        };
        st.editors = this.editors.layoutState();
        st.breakpoints = Object.fromEntries([...this.breakpoints].filter(([p, v]) => v && v.length && !p.includes(':')));
        st.watch = this.debugger ? this.debugger.watches : st.watch;
        st.runConfig = this.views && this.views.debug ? this.views.debug.config() : st.runConfig;
        saveProject(this.projectKey, st);
    }

    restoreLayout() {
        const l = this.project.layout || {};
        if (l.sizes) this.layout.setSizes(l.sizes);
        const side = l.side && this.layout.sideViews.has(l.side) ? l.side : 'explorer';
        this.layout.showSide(side, { focus: false });
        if (l.sideOpen === false) this.layout.hideSide();
        const panel = l.panel && this.layout.panels.has(l.panel) ? l.panel : 'output';
        this.layout.showPanel(panel);
        if (l.panelOpen !== true) this.layout.setPanelOpen(false);
    }

    async restoreTabs() {
        const e = this.project.editors;
        if (!e || !Array.isArray(e.groups)) return;
        for (let gi = 0; gi < e.groups.length && gi < 2; gi++) {
            const g = e.groups[gi];
            let group = this.editors.groups[gi];
            if (!group) { if (!g.tabs || !g.tabs.length) continue; group = this.editors.addGroup(); }
            for (const tab of g.tabs || []) {
                if (tab.special === 'welcome') { this.showWelcome({ group, focus: false }); continue; }
                if (!tab.path) continue;
                try {
                    const doc = await this.docs.open(tab.path);
                    this.editors.show(doc, { group, focus: false, line: tab.sel && tab.sel.line, col: tab.sel && tab.sel.col, flash: false });
                } catch (err) { /* the file is gone */ }
            }
            if (g.active && group.tabs.some((x) => x.key === g.active)) group.activate(g.active, { focus: false });
        }
        const fg = this.editors.groups[e.focused] || this.editors.groups[0];
        this.editors.setFocused(fg);
        if (!this.editors.groups[0].tabs.length && this.editors.groups.length > 1) this.editors.removeGroup(this.editors.groups[0]);
    }

    /* recent projects (welcome page) */
    remember() {
        const list = load('recent', []).filter((r) => r.key !== this.projectKey);
        list.unshift({ key: this.projectKey, name: this.info.rootName, root: this.info.root, web: this.web, time: Date.now() });
        save('recent', list.slice(0, 8));
    }

    /* ── settings, theme, language ──────────────────────────────── */
    effectiveTheme() {
        const th = this.settings.get('theme');
        if (th === 'light' || th === 'dark') return th;
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    applyTheme() {
        document.documentElement.dataset.theme = this.effectiveTheme();
        document.documentElement.style.setProperty('--fs-code', `${this.settings.get('fontSize')}px`);
        if (this.layout) this.updateTitlebar();
    }

    toggleTheme() {
        this.settings.set('theme', this.effectiveTheme() === 'dark' ? 'light' : 'dark');
    }

    onSetting(key, value) {
        if (key === 'theme' || key === 'fontSize') {
            this.applyTheme();
            this.editorsResized();
        }
        if (key === 'uiLang') setLocale(value || detectLocale());
        if (['errorLens', 'inlayHints', 'codeLens', 'wordWrap'].includes(key)) this.docs.applySettings(this.settings.values);
        if (key === 'backend' || key === 'device') this.updateTitlebar();
        if (this.views && this.views.settings) this.views.settings.sync();
        this.statusbar.update();
    }

    onLocaleChanged() {
        refreshTips(document);
        applyI18n(document);
        this.layout.menubar.replaceChildren();
        buildMenubar(this);
        this.layout.fitTitlebar();          // menu labels have another length now
        this.editors.renderAllTabs();
        for (const g of this.editors.groups) {
            g.refreshEmpty();
            // views in editor tabs (welcome page, translations, about)
            for (const e of g.tabs) if (e.kind === 'special' && e.special.relocalize) e.special.relocalize();
        }
        for (const v of Object.values(this.views)) if (v.relocalize) v.relocalize();
        for (const v of [this.output, this.bottomProblems, this.debugConsole, this.screen]) if (v.relocalize) v.relocalize();
        this.statusbar.update();
        this.updateTitlebar();
        if (this.lsp) {
            this.lsp.locale = getLocale();
            if (this.lsp.ready) {
                this.lsp.notify('workspace/didChangeConfiguration', { settings: { til: { locale: getLocale() } } });
                this.docs.applySettings(this.settings.values);
            }
        }
    }

    updateTitlebar() {
        const L = this.layout;
        const backend = this.settings.get('backend');
        const names = { native: t('backend.native'), vm: t('backend.vm'), cvm: t('backend.cvm') };
        L.setPickerLabel(L.backendPicker, names[this.effectiveBackend()] || names.vm, backend === 'native' ? 'zap' : backend === 'cvm' ? 'calculator' : 'layers');
        const devs = this.info.devices || [];
        L.devicePicker.hidden = this.web || !devs.length;
        const dev = devs.find((d) => d.name === this.settings.get('device')) || devs[0];
        L.setPickerLabel(L.devicePicker, dev ? dev.name : '');
        L.themeBtn.replaceChildren(icon(this.effectiveTheme() === 'dark' ? 'sun' : 'moon'));
        L.debugBtn.hidden = !this.caps.debugger;
    }

    /* the backend a run uses: what is chosen, unless this face does not have it */
    effectiveBackend() {
        const b = this.settings.get('backend');
        if (b === 'native' && !this.caps.native) return 'vm';
        if (b === 'cvm' && !this.caps.cvm) return 'vm';
        return b || 'vm';
    }

    backendAvailable(b) {
        if (b === 'vm') return true;
        if (b === 'native') return Boolean(this.caps.native);
        if (b === 'cvm') return Boolean(this.caps.cvm);
        return false;
    }

    pickBackend(at) {
        const cur = this.effectiveBackend();
        openMenu([
            { header: t('backend.title') },
            ...['native', 'vm', 'cvm'].map((b) => ({
                label: t(`backend.${b}`), checked: cur === b, disabled: !this.backendAvailable(b),
                run: () => this.settings.set('backend', b),
            })),
            { separator: true },
            { label: t('backend.explain'), disabled: true },
        ], at, { align: 'end' });
    }

    pickDevice(at) {
        const devs = this.info.devices || [];
        openMenu([{ header: t('devices.pick') }, ...devs.map((d) => ({
            label: `${d.name} — ${d.title || ''}`, checked: d.name === this.settings.get('device'),
            run: () => { this.settings.set('device', d.name); this.layout.showSide('devices'); },
        }))], at, { align: 'end' });
    }

    pickLocale(at) {
        openMenu(LOCALES.map((l) => ({ label: LOCALE_NAMES[l], checked: getLocale() === l, run: () => this.settings.set('uiLang', l) })), at, { align: 'end' });
    }

    /* ── paths ──────────────────────────────────────────────────── */
    uriOf(path) { return pathToUri(this.rootUri, path); }

    pathOf(uri) {
        if (!uri) return null;
        const p = uriToPath(this.rootUri, uri);
        if (p !== null) return p;
        for (const d of this.docs.all()) if (d.uri === uri) return d.path;
        return null;
    }

    /* absolute path for the debug adapter (DAP uses file system paths) */
    absPath(path) {
        const root = String(this.info.root || '');
        if (!root) return path;
        const sep = root.includes('\\') && !root.includes('/') ? '\\' : '/';
        return root.replace(/[\\/]+$/, '') + sep + path.split('/').join(sep);
    }

    relPath(abs) {
        if (!abs) return null;
        const root = String(this.info.root || '').replace(/\\/g, '/').replace(/\/+$/, '');
        const p = String(abs).replace(/\\/g, '/');
        const win = /^[A-Za-z]:/.test(root);
        const a = win ? root.toLowerCase() : root, b = win ? p.toLowerCase() : p;
        if (root && b.startsWith(a + '/')) return p.slice(root.length + 1);
        if (!p.startsWith('/') && !/^[A-Za-z]:/.test(p)) return p.replace(/^\.\//, '');
        return null;
    }

    /* ── files ──────────────────────────────────────────────────── */
    async openFile(path, opts = {}) {
        let doc;
        try {
            doc = await this.docs.open(path);
        } catch (e) {
            if (e && e.status === 415) toast(t('file.binary', { name: basename(path) }), { kind: 'warning' });
            else toast(t('file.cannotOpen', { name: basename(path) }), { kind: 'error', detail: e && e.message });
            return null;
        }
        let group = opts.group;
        if (opts.side) {
            if (this.editors.groups.length < 2) this.editors.addGroup();
            group = this.editors.groups.find((g) => g !== this.editors.group()) || this.editors.groups[1];
            this.editorsResized();
        }
        this.editors.show(doc, Object.assign({}, opts, { group }));
        return doc;
    }

    async openLocation(path, range, opts = {}) {
        const doc = await this.openFile(path, opts);
        if (!doc || !range) return doc;
        const view = this.editors.activeView();
        if (!view) return doc;
        const { rangeToOffsets } = await import('./lsp/cm_lsp.js');
        const r = rangeToOffsets(view.state.doc, range);
        this.editors.show(doc, { selection: { from: r.from, to: r.to }, focus: opts.focus !== false });
        return doc;
    }

    async saveDoc(doc) {
        if (!doc) return false;
        if (doc.kind === 'example' || doc.kind === 'untitled' || doc.readOnly) return this.saveAs(doc);
        try {
            const ok = await this.docs.save(doc);
            if (ok) this.statusbar.flash(t('file.saved', { name: doc.name }));
            return ok;
        } catch (e) {
            toast(t('file.cannotSave', { name: doc.name }), { kind: 'error', detail: e && e.message });
            return false;
        }
    }

    async saveAs(doc) {
        const { promptDialog } = await import('./ui/widgets.js');
        const suggested = doc.kind === 'example' || doc.readOnly ? `examples/${doc.name}` : doc.name;
        const path = await promptDialog({
            title: t('file.saveAsTitle'), message: t('file.saveAsMsg'), value: suggested, confirm: t('file.save'),
            validate: (v) => (/^\/|\.\.|\\|:/.test(v) ? t('explorer.nameInvalid') : ''),
            select: [suggested.lastIndexOf('/') + 1, suggested.lastIndexOf('.') > 0 ? suggested.lastIndexOf('.') : suggested.length],
        });
        if (!path) return false;
        const text = doc.current().doc.toString();
        try {
            const dir = dirname(path);
            if (dir) await this.ensureDir(dir);
            await this.backend.createFile(path, 'file', text);
        } catch (e) {
            if (e && e.status === 409) {
                const ok = await confirmDialog({ title: t('file.existsTitle'), message: t('file.existsMsg', { name: path }), confirm: t('file.overwrite'), danger: true });
                if (!ok) return false;
                await this.backend.writeFile(path, text);
            } else {
                toast(t('file.cannotSave', { name: path }), { kind: 'error', detail: e && e.message });
                return false;
            }
        }
        await this.views.explorer.refresh();
        const g = this.editors.findTab(doc.path);
        const newDoc = await this.docs.open(path);
        if (g) {
            const idx = g.tabs.findIndex((x) => x.key === doc.path);
            await g.close(doc.path, { force: true });
            g.add({ key: newDoc.path, kind: 'doc', doc: newDoc }, { index: Math.max(0, idx) });
        } else this.editors.show(newDoc);
        return true;
    }

    async ensureDir(dir) {
        try { await this.backend.createFile(dir, 'dir'); } catch (e) { /* exists */ }
    }

    async saveAll() {
        for (const doc of this.docs.dirtyDocs()) if (doc.kind === 'file') await this.saveDoc(doc);
    }

    /* 'save' | 'discard' | 'cancel' */
    async askSave(doc) {
        const r = await confirmDialog({
            title: t('file.unsavedTitle'), message: t('file.unsavedMsg', { name: doc.name }),
            confirm: t('file.save'), cancel: t('dialog.cancel'), third: t('file.dontSave'),
        });
        if (r === true) return 'save';
        if (r === 'third') return 'discard';
        return 'cancel';
    }

    async askConflict(doc) {
        const v = await customDialog({
            title: t('file.conflictTitle'),
            body: h('div', { class: 'dialog-body' }, h('p', {}, t('file.conflictMsg', { name: doc.name }))),
            buttons: [
                { label: t('dialog.cancel'), value: 'cancel' },
                { label: t('file.reload'), value: 'reload' },
                { label: t('file.overwrite'), value: 'overwrite', danger: true },
            ],
        });
        return v || 'cancel';
    }

    async createPath(path, type) {
        try {
            await this.backend.createFile(path, type, type === 'file' ? this.newFileTemplate(path) : undefined);
        } catch (e) {
            toast(t('explorer.cannotCreate', { name: path }), { kind: 'error', detail: e && e.message });
            return;
        }
        this.fileIndex.invalidate();
        if (type === 'file') this.openFile(path);
    }

    newFileTemplate(path) {
        if (!isTil(path)) return '';
        const lang = this.settings.get('kwLang');
        const name = basename(path).replace(/\.til$/i, '');
        const tpl = {
            en: `# ${name}.til\n\nmain()\n    print("Hello from ${name}!")\n`,
            kk: `# ${name}.til\n\nнегізгі()\n    басып_шығару("Сәлем, ${name}!")\n`,
            'kk-latn': `#[lang: kk-latn]\n# ${name}.til\n\nnegizgi()\n    basyp_shygaru("Salem, ${name}!")\n`,
            ru: `#[lang: ru]\n# ${name}.til\n\nглавная()\n    печать("Привет, ${name}!")\n`,
        };
        return tpl[lang] || tpl.en;
    }

    async renamePath(from, to) {
        try {
            await this.backend.renamePath(from, to);
        } catch (e) {
            toast(t('explorer.cannotRename', { name: from }), { kind: 'error', detail: e && e.message });
            return false;
        }
        this.docs.renamed(from, to);
        const bps = this.breakpoints.get(from);
        if (bps) { this.breakpoints.delete(from); this.breakpoints.set(to, bps); }
        for (const g of this.editors.groups) {
            for (const tab of g.tabs) if (tab.kind === 'doc') tab.key = tab.doc.path;
            if (g.active === from) g.active = to;
        }
        this.editors.renderAllTabs();
        this.fileIndex.invalidate();
        this.onActiveEditorChanged();
        return true;
    }

    async deletePath(path) {
        try {
            await this.backend.deletePath(path);
        } catch (e) {
            toast(t('explorer.cannotDelete', { name: path }), { kind: 'error', detail: e && e.message });
            return false;
        }
        for (const g of this.editors.groups.slice()) {
            for (const tab of g.tabs.slice()) {
                if (tab.kind === 'doc' && (tab.doc.path === path || tab.doc.path.startsWith(path + '/'))) await g.close(tab.key, { force: true });
            }
        }
        this.breakpoints.delete(path);
        this.fileIndex.invalidate();
        return true;
    }

    async newUntitled() {
        const lang = this.settings.get('kwLang');
        const doc = this.docs.openVirtual({ text: this.newFileTemplate('untitled.til').replace('untitled', 'main'), name: 'untitled.til', kind: 'untitled' });
        doc.savedDoc = doc.state.doc.slice(0, 0);
        this.editors.show(doc);
        void lang;
        return doc;
    }

    async openExample(ex) {
        const doc = this.docs.openVirtual({ text: ex.source, name: ex.file, kind: 'example' });
        this.editors.show(doc);
        return doc;
    }

    onFsChanged(ev) {
        const paths = ev && Array.isArray(ev.paths) ? ev.paths : [];
        if (!paths.length) return;
        this.docs.externalChange(paths);
        const dirs = new Set(paths.map((p) => dirname(p)));
        if (dirs.size > 3 || dirs.has('')) this.views.explorer.refresh();
        else for (const d of dirs) this.views.explorer.refresh(d);
        this.fileIndex.invalidate();
        if (paths.some((p) => /(^|\/)til\.json$/.test(p))) this.views.packages.refresh();
    }

    /* ── editor events ──────────────────────────────────────────── */
    onDirty(doc) {
        const dirty = doc.dirty;
        if (dirty !== doc._shownDirty) {
            doc._shownDirty = dirty;
            this.editors.renderAllTabs();
            this.views.explorer.rerender();
        }
        if (this.settings.get('autoSave') === 'delay' && doc.kind === 'file' && doc.dirty) {
            clearTimeout(doc._autoSave);
            doc._autoSave = setTimeout(() => { if (doc.dirty) this.saveDoc(doc); }, 1000);
        }
    }

    onDocChange(doc) {
        if (this.views.bytecode && doc === this.editors.activeDoc()) this.views.bytecode.markStale();
    }

    onCursor(doc, view, update) {
        if (doc !== this.editors.activeDoc()) return;
        this.statusbar.updateCursor(view);
        this.crumbsLater();
        if (this.views.bytecode) this.views.bytecode.syncCursor(view);
        if (update && update.focusChanged && !view.hasFocus && this.settings.get('autoSave') === 'focus' && doc.dirty && doc.kind === 'file') this.saveDoc(doc);
    }

    crumbsLater() {
        clearTimeout(this._crumbTimer);
        this._crumbTimer = setTimeout(() => this.renderCrumbs(), 150);
    }

    async renderCrumbs() {
        const g = this.editors.group();
        const doc = g.activeDoc();
        const { renderBreadcrumbs } = await import('./ui/statusbar.js');
        renderBreadcrumbs(this, g, doc);
    }

    onActiveEditorChanged() {
        const doc = this.editors.activeDoc();
        const view = this.editors.activeView();
        this.statusbar.updateCursor(view);
        this.statusbar.update();
        this.renderCrumbs();
        this.views.explorer.markActive();
        if (this.views.bytecode) this.views.bytecode.activeChanged(doc);
        if (this.views.proofs) this.views.proofs.activeChanged(doc);
        this.debugger.refreshEditors();
    }

    editorsResized() {
        for (const g of this.editors ? this.editors.groups : []) if (g.view) g.view.requestMeasure();
        if (this.screen) this.screen.fit();
    }

    onBreakpoints(doc, list) {
        this.breakpoints.set(doc.path, list.map((b) => ({ line: b.line, condition: b.condition || '', hitCondition: b.hitCondition || '', enabled: b.enabled !== false })));
        this.views.debug.renderBreakpoints();
        this.debugger.breakpointsChanged(doc.path);
        this.saveLayout();
    }

    setBreakpoints(path, list) {
        this.breakpoints.set(path, list);
        const doc = this.docs.get(path);
        if (doc) doc.dispatch({ effects: bpEffect.of({ type: 'set', list }) });
        this.views.debug.renderBreakpoints();
        this.debugger.breakpointsChanged(path);
        this.saveLayout();
    }

    toggleBreakpoint() {
        const view = this.editors.activeView();
        const doc = this.editors.activeDoc();
        if (!view || !doc) return;
        const line = view.state.doc.lineAt(view.state.selection.main.head).number;
        view.dispatch({ effects: bpEffect.of({ type: 'toggle', line }) });
    }

    breakpointMenu(view, line, e) {
        const doc = view._studioDoc;
        const list = this.docs.breakpointsOf(doc);
        const bp = list.find((b) => b.line === line);
        openMenu(bp ? [
            { label: t('bp.edit'), icon: 'edit', run: () => this.editBreakpoint(doc, line) },
            { label: t('bp.editHits'), icon: 'edit', run: () => this.editBreakpoint(doc, line, false, 'hitCondition') },
            { label: bp.enabled === false ? t('bp.enable') : t('bp.disable'), run: () => view.dispatch({ effects: bpEffect.of({ type: 'update', line, props: { enabled: bp.enabled === false } }) }) },
            { label: t('bp.remove'), icon: 'trash', danger: true, run: () => view.dispatch({ effects: bpEffect.of({ type: 'toggle', line }) }) },
        ] : [
            { label: t('bp.add'), icon: 'breakpoint', run: () => view.dispatch({ effects: bpEffect.of({ type: 'toggle', line }) }) },
            { label: t('bp.addConditional'), icon: 'edit', run: () => this.editBreakpoint(doc, line, true) },
        ], { x: e.clientX, y: e.clientY });
    }

    async editBreakpoint(doc, line, create = false, field = 'condition') {
        const { promptDialog } = await import('./ui/widgets.js');
        const list = this.docs.breakpointsOf(doc);
        const bp = list.find((b) => b.line === line);
        const hits = field === 'hitCondition';
        const value = await promptDialog({
            title: t(hits ? 'bp.hitsTitle' : 'bp.conditionTitle', { line }), message: t(hits ? 'bp.hitsMsg' : 'bp.conditionMsg'),
            value: bp ? bp[field] || '' : '', placeholder: hits ? '>= 3' : 'i == 3', allowEmpty: true,
        });
        if (value === null) return;
        if (!bp && create) doc.dispatch({ effects: bpEffect.of({ type: 'toggle', line, condition: hits ? '' : value }) });
        else doc.dispatch({ effects: bpEffect.of({ type: 'update', line, props: { [field]: value } }) });
    }

    /* ── diagnostics ────────────────────────────────────────────── */
    onDiagnostics(params) {
        const uri = params.uri;
        const list = Array.isArray(params.diagnostics) ? params.diagnostics : [];
        if (list.length) this.diagnostics.set(uri, list); else this.diagnostics.delete(uri);
        const versioned = params.version !== undefined && params.version !== null;
        for (const doc of this.docs.all()) {
            if (doc.uri !== uri) continue;
            if (versioned && doc.lspOpen && params.version < this.lsp.version(uri) - 1) continue;
            if (versioned) doc.diagVersion = params.version;
            this.docs.applyDiagnostics(doc, list);
        }
        if (versioned && this.diagWaiters) {
            for (const w of this.diagWaiters.filter((x) => x.uri === uri && params.version >= x.version)) w.resolve(true);
        }
        this.diagnosticsChanged();
    }

    /* resolves true once the diagnostics shown are those of the document's
       current text (servers that publish versions), false after `ms` */
    freshDiagnostics(doc, ms = 800) {
        const want = this.lsp.version(doc.uri);
        if (doc.diagVersion === undefined || doc.diagVersion >= want) return Promise.resolve(true);
        if (!this.diagWaiters) this.diagWaiters = [];
        return new Promise((resolve) => {
            const w = { uri: doc.uri, version: want, resolve: null };
            const done = (v) => { this.diagWaiters = this.diagWaiters.filter((x) => x !== w); clearTimeout(timer); resolve(v); };
            w.resolve = done;
            const timer = setTimeout(() => done(false), ms);
            this.diagWaiters.push(w);
        });
    }

    diagnosticsChanged() {
        clearTimeout(this._diagTimer);
        this._diagTimer = setTimeout(() => {
            this.views.problems.renderIfVisible();
            this.bottomProblems.renderIfVisible();
            this.statusbar.update();
            const sig = [...this.diagnostics.keys()].sort().map((u) => { const x = this.diagnosticSummary(u); return `${u}:${x.errors}:${x.warnings}`; }).join('|');
            if (sig !== this._diagSig) {
                this._diagSig = sig;
                this.editors.renderAllTabs();
                this.views.explorer.rerender();
            }
            const { errors, warnings } = this.diagnosticTotals();
            this.layout.setBadge('problems', errors + warnings, errors ? 'error' : '');
            this.layout.setPanelBadge('problems', errors + warnings, errors ? 'error' : '');
        }, 30);
    }

    diagnosticSummary(uri) {
        const list = this.diagnostics.get(uri);
        if (!list) return null;
        let errors = 0, warnings = 0;
        for (const d of list) { if ((d.severity || 1) === 1) errors++; else if (d.severity === 2) warnings++; }
        return { errors, warnings };
    }

    diagnosticTotals() {
        let errors = 0, warnings = 0, infos = 0;
        for (const list of this.diagnostics.values()) {
            for (const d of list) {
                const s = d.severity || 1;
                if (s === 1) errors++; else if (s === 2) warnings++; else infos++;
            }
        }
        return { errors, warnings, infos };
    }

    /* ── run, debug, tools ──────────────────────────────────────── */
    activeTilDoc() {
        const doc = this.editors.activeDoc();
        if (doc && doc.isTil) return doc;
        return this.editors.openDocs().find((d) => d.isTil) || null;
    }

    async runFile(path, opts = {}) {
        let doc = path ? this.docs.get(path) || await this.docs.open(path).catch(() => null) : this.activeTilDoc();
        if (!doc) { toast(t('run.noFile'), { kind: 'info' }); return; }
        return this.runner.run(doc, opts);
    }

    async debugFile(path, opts = {}) {
        if (!this.caps.debugger) { this.showDesktopOnly('debugger'); return; }
        let doc = path ? this.docs.get(path) || await this.docs.open(path).catch(() => null) : this.activeTilDoc();
        if (!doc) { toast(t('run.noFile'), { kind: 'info' }); return; }
        return this.debugger.start(doc, opts);
    }

    async executeLensCommand(command, path) {
        const name = command.command || '';
        const arg = Array.isArray(command.arguments) && command.arguments[0] || {};
        const p = (arg.uri && this.pathOf(arg.uri)) || path;
        if (/til\.run$/.test(name)) return this.runFile(p);
        if (/til\.debug$/.test(name)) return this.debugFile(p, { function: arg.function || null });
        if (/til\.prove$/.test(name)) { this.layout.showSide('proofs'); return this.views.proofs.prove(p, arg.function); }
        try {
            await this.lsp.request('workspace/executeCommand', { command: name, arguments: command.arguments || [] });
        } catch (e) {
            toast(t('command.failed', { name: command.title || name }), { kind: 'error', detail: e.message });
        }
        return null;
    }

    async formatDoc(doc) { return this.lspActions.format(doc); }

    showWelcome(opts = {}) {
        if (!this.welcome) this.welcome = new WelcomePage(this);
        this.editors.showSpecial('welcome', this.welcome, opts);
    }

    async openTranslation(path, lang) {
        const key = `translate:${path}:${lang}`;
        const view = new TranslateView(this, path, lang);
        this.editors.showSpecial(key, view);
        await view.load();
    }

    translateMenu(path) {
        return ['en', 'kk', 'kk-latn', 'ru'].map((lang) => ({
            label: t(`kwlang.${lang}`),
            submenu: [
                { label: t('translate.view'), icon: 'split', run: () => this.openTranslation(path, lang) },
                { label: t('translate.replace'), icon: 'translate', run: () => this.translateInPlace(path, lang) },
            ],
        }));
    }

    async translateInPlace(path, lang) {
        const doc = path ? await this.docs.open(path) : this.activeTilDoc();
        if (!doc) return;
        if (doc.readOnly) { toast(t('file.readOnlyHint'), { kind: 'info' }); return; }
        let res;
        try {
            res = await this.backend.translate({ path: doc.kind === 'file' ? doc.path : undefined, source: doc.current().doc.toString(), to: lang });
        } catch (e) {
            if (e && e.notAvailable) { this.showDesktopOnly(e.feature); return; }
            toast(t('translate.failed'), { kind: 'error', detail: e && e.message });
            return;
        }
        if (res && typeof res.text === 'string') {
            doc.replaceText(res.text);
            toast(t('translate.done', { lang: t(`kwlang.${lang}`) }), { kind: 'success', actions: [{ label: t('edit.undo'), run: () => this.undoIn(doc) }] });
        }
    }

    async undoIn(doc) {
        const { undo } = await import('@codemirror/commands');
        for (const v of doc.views) { undo(v); break; }
    }

    showDesktopOnly(feature) {
        if (this.web) {
            import('./views/desktop_card.js').then(({ desktopOnlyDialog }) => desktopOnlyDialog(this, feature));
        } else toast(t('feature.unavailable', { feature: t(`feature.${feature}`) }), { kind: 'warning' });
    }

    /* ── small helpers ──────────────────────────────────────────── */
    async copyText(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.statusbar.flash(t('clipboard.copied'));
        } catch (e) {
            const ta = h('textarea', { style: 'position:fixed;opacity:0' });
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            try { document.execCommand('copy'); this.statusbar.flash(t('clipboard.copied')); } catch (err) { /* */ }
            ta.remove();
        }
    }

    revealInExplorer(path) {
        this.layout.showSide('explorer', { focus: false });
        this.views.explorer.reveal(path, { focus: true });
    }

    editorContextMenu(group, e) {
        const view = group.view;
        if (!view) return;
        e.preventDefault();
        const doc = view._studioDoc;
        const lsp = this.lsp && this.lsp.ready;
        const til = doc && doc.isTil;
        const sel = !view.state.selection.main.empty;
        openMenu([
            til ? { label: t('cmd.definition'), keys: 'F12', disabled: !lsp, run: () => this.commands.run('editor.definition') } : null,
            til ? { label: t('cmd.references'), keys: 'Shift+F12', disabled: !lsp, run: () => this.commands.run('editor.references') } : null,
            til ? { label: t('cmd.rename'), keys: 'F2', disabled: !lsp || (doc && doc.readOnly), run: () => this.commands.run('editor.rename') } : null,
            til ? { label: t('cmd.codeActions'), keys: 'Mod+.', disabled: !lsp, run: () => this.commands.run('editor.codeActions') } : null,
            til ? { label: t('cmd.format'), keys: 'Shift+Alt+F', disabled: !lsp || (doc && doc.readOnly), run: () => this.commands.run('editor.format') } : null,
            til ? { separator: true } : null,
            til ? { label: t('cmd.runFile'), icon: 'play', keys: 'Mod+F5', run: () => this.commands.run('run.run') } : null,
            til ? { label: t('cmd.toggleBreakpoint'), icon: 'breakpoint', keys: 'F9', run: () => this.toggleBreakpoint() } : null,
            til ? { label: t('explorer.translate'), icon: 'translate', submenu: this.translateMenu(doc.kind === 'file' ? doc.path : null) } : null,
            { separator: true },
            { label: t('edit.cut'), keys: 'Mod+X', disabled: !sel || (doc && doc.readOnly), run: () => this.clipboardAction(view, 'cut') },
            { label: t('edit.copy'), keys: 'Mod+C', disabled: !sel, run: () => this.clipboardAction(view, 'copy') },
            { label: t('edit.paste'), keys: 'Mod+V', disabled: doc && doc.readOnly, run: () => this.clipboardAction(view, 'paste') },
            { separator: true },
            { label: t('palette.showCommands'), keys: 'Mod+Shift+P', run: () => this.commands.run('workbench.commandPalette') },
        ], { x: e.clientX, y: e.clientY });
    }

    async clipboardAction(view, what) {
        const r = view.state.selection.main;
        if (what === 'copy' || what === 'cut') {
            const text = view.state.sliceDoc(r.from, r.to);
            await this.copyText(text);
            if (what === 'cut') view.dispatch({ changes: { from: r.from, to: r.to }, userEvent: 'delete.cut' });
        } else {
            try {
                const text = await navigator.clipboard.readText();
                view.dispatch(view.state.replaceSelection(text));
            } catch (e) { toast(t('clipboard.pasteBlocked'), { kind: 'info' }); }
        }
        view.focus();
    }

    editorMoreMenu(rect) {
        const doc = this.editors.activeDoc();
        openMenu([
            doc && doc.isTil ? { label: t('explorer.translate'), icon: 'translate', submenu: this.translateMenu(doc.kind === 'file' ? doc.path : null) } : null,
            { label: t('cmd.toggleErrorLens'), checked: this.settings.get('errorLens'), run: () => this.settings.set('errorLens', !this.settings.get('errorLens')) },
            { label: t('cmd.toggleInlayHints'), checked: this.settings.get('inlayHints'), run: () => this.settings.set('inlayHints', !this.settings.get('inlayHints')) },
            { label: t('cmd.toggleWordWrap'), checked: this.settings.get('wordWrap'), run: () => this.settings.set('wordWrap', !this.settings.get('wordWrap')) },
            { separator: true },
            { label: t('editor.closeAll'), run: () => this.editors.closeAll() },
        ], rect, { align: 'end' });
    }

    setDebugLine(path, line) {
        for (const doc of this.docs.all()) {
            const want = doc.path === path ? { line } : null;
            doc.dispatch({ effects: setDebugLine.of(want) });
        }
    }

    platformKey() { return isMac() ? 'mac' : 'other'; }
}
