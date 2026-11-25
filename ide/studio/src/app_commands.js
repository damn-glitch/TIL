// Every command of TIL Studio (with its default key binding) and the menu bar.
import { undo, redo, selectAll, toggleComment } from '@codemirror/commands';
import { openSearchPanel, replaceAll } from '@codemirror/search';
import { foldAll, unfoldAll } from '@codemirror/language';
import { EditorSelection } from '@codemirror/state';
import { EditorView } from '@codemirror/view';
import { h } from './core/dom.js';
import { icon, fileIcon, logoMark } from './core/icons.js';
import { t, LOCALES, LOCALE_NAMES } from './core/i18n.js';
import { basename, dirname, isTil } from './core/paths.js';
import { quickPick } from './ui/palette.js';
import { openMenu, customDialog, toast } from './ui/widgets.js';
import { flashEffect } from './editor/gutters.js';

export function registerCommands(app) {
    const C = app.commands;
    const reg = (id, title, run, opts = {}) => C.register(Object.assign({ id, title, run }, opts));
    const view = () => app.editors.activeView();
    const onView = (fn) => () => { const v = view(); if (v) { fn(v); v.focus(); } };
    const dbg = app.debugger;
    const debugging = () => dbg.active;
    const stopped = () => dbg.active && dbg.stopped;
    const hasDebugger = () => Boolean(app.caps.debugger);

    /* palette and navigation */
    reg('workbench.commandPalette', 'palette.showCommands', () => openPalette(app, '>'), { keys: ['Mod+Shift+P', 'F1'], category: 'cat.view' });
    reg('workbench.quickOpen', 'palette.quickOpen', () => openPalette(app, ''), { keys: 'Mod+P', category: 'cat.go' });
    reg('workbench.gotoSymbol', 'palette.gotoSymbol', () => openPalette(app, '@'), { keys: 'Mod+Shift+O', category: 'cat.go' });
    reg('workbench.workspaceSymbol', 'palette.workspaceSymbol', () => openPalette(app, '#'), { keys: 'Mod+T', category: 'cat.go' });
    reg('editor.gotoLine', 'palette.gotoLine', () => openPalette(app, ':'), { keys: 'Mod+G', category: 'cat.go', when: () => Boolean(view()) });

    /* views */
    for (const [id, key, keys] of [['explorer', 'activity.explorer', 'Mod+Shift+E'], ['search', 'activity.search', 'Mod+Shift+F'], ['debug', 'activity.debug', 'Mod+Shift+D'],
        ['problems', 'activity.problems', 'Mod+Shift+M'], ['proofs', 'activity.proofs'], ['devices', 'activity.devices'], ['packages', 'activity.packages'],
        ['bytecode', 'activity.bytecode'], ['settings', 'activity.settings', 'Mod+,']]) {
        reg(`workbench.show.${id}`, key, () => {
            app.layout.showSide(id);
            if (id === 'search') {
                const v = view();
                const sel = v ? v.state.sliceDoc(v.state.selection.main.from, v.state.selection.main.to) : '';
                app.views.search.focus(sel && !sel.includes('\n') ? sel : undefined);
            }
        }, { keys, category: 'cat.view' });
    }
    reg('workbench.toggleSidebar', 'cmd.toggleSidebar', () => (app.layout.sideOpen() ? app.layout.hideSide() : app.layout.showSide(app.layout.activeSide || 'explorer')), { keys: 'Mod+B', category: 'cat.view' });
    reg('workbench.togglePanel', 'cmd.togglePanel', () => app.layout.togglePanel(), { keys: 'Ctrl+`', category: 'cat.view' });
    reg('workbench.maximizePanel', 'panel.maximize', () => app.layout.togglePanelMax(), { category: 'cat.view' });
    for (const [id, key] of [['output', 'panel.output'], ['problems', 'panel.problems'], ['console', 'panel.debugConsole'], ['screen', 'panel.screen']]) {
        reg(`workbench.panel.${id}`, key, () => app.layout.showPanel(id, { focus: true }), { category: 'cat.view' });
    }
    reg('workbench.welcome', 'welcome.title', () => app.showWelcome(), { category: 'cat.help' });
    reg('view.toggleTheme', 'cmd.toggleTheme', () => app.toggleTheme(), { keys: 'Alt+Shift+T', category: 'cat.view' });
    reg('view.zoomIn', 'cmd.zoomIn', () => app.settings.set('fontSize', Math.min(28, app.settings.get('fontSize') + 1)), { keys: 'Mod+=', category: 'cat.view' });
    reg('view.zoomOut', 'cmd.zoomOut', () => app.settings.set('fontSize', Math.max(10, app.settings.get('fontSize') - 1)), { keys: 'Mod+-', category: 'cat.view' });
    reg('view.zoomReset', 'cmd.zoomReset', () => app.settings.set('fontSize', 14), { keys: 'Mod+0', category: 'cat.view' });
    for (const l of LOCALES) reg(`view.uiLang.${l}`, () => t('cmd.uiLanguage', { lang: LOCALE_NAMES[l] }), () => app.settings.set('uiLang', l), { category: 'cat.view' });

    /* files */
    reg('file.newUntitled', 'cmd.newFile', () => app.newUntitled(), { keys: 'Alt+N', category: 'cat.file' });
    reg('file.newFile', 'explorer.newFile', () => app.views.explorer.startCreate('file'), { category: 'cat.file' });
    reg('file.newFolder', 'explorer.newFolder', () => app.views.explorer.startCreate('dir'), { category: 'cat.file' });
    reg('file.save', 'file.save', () => app.saveDoc(app.editors.activeDoc()), { keys: 'Mod+S', category: 'cat.file', when: () => Boolean(app.editors.activeDoc()), inFields: true });
    reg('file.saveAll', 'file.saveAll', () => app.saveAll(), { keys: 'Mod+Alt+S', category: 'cat.file' });
    reg('file.revert', 'file.revert', () => { const d = app.editors.activeDoc(); if (d && d.kind === 'file') app.docs.reload(d); }, { category: 'cat.file', when: () => Boolean(app.editors.activeDoc()) });
    reg('file.close', 'editor.close', () => { const g = app.editors.group(); if (g.active) g.close(g.active); }, { keys: 'Alt+W', category: 'cat.file' });
    reg('file.closeAll', 'editor.closeAll', () => app.editors.closeAll(), { category: 'cat.file' });
    reg('file.reveal', 'explorer.reveal', () => { const d = app.editors.activeDoc(); if (d && d.kind === 'file') app.revealInExplorer(d.path); }, { category: 'cat.file', when: () => Boolean(app.editors.activeDoc()) });
    reg('file.copyPath', 'explorer.copyPath', () => { const d = app.editors.activeDoc(); if (d) app.copyText(d.path); }, { category: 'cat.file', when: () => Boolean(app.editors.activeDoc()) });

    /* edit */
    reg('edit.undo', 'edit.undo', onView((v) => undo(v)), { category: 'cat.edit', palette: true });
    reg('edit.redo', 'edit.redo', onView((v) => redo(v)), { category: 'cat.edit' });
    reg('edit.selectAll', 'edit.selectAll', onView((v) => selectAll(v)), { category: 'cat.edit' });
    reg('edit.find', 'edit.find', onView((v) => openSearchPanel(v)), { category: 'cat.edit' });
    reg('edit.replace', 'edit.replace', onView((v) => openSearchPanel(v)), { keys: 'Mod+H', category: 'cat.edit', when: () => Boolean(view()) });
    reg('edit.toggleComment', 'edit.toggleComment', onView((v) => toggleComment(v)), { category: 'cat.edit' });
    reg('edit.foldAll', 'edit.foldAll', onView((v) => foldAll(v)), { category: 'cat.edit' });
    reg('edit.unfoldAll', 'edit.unfoldAll', onView((v) => unfoldAll(v)), { category: 'cat.edit' });
    void replaceAll;

    /* language features: only while the editor (not a tree, a list or a field) has the focus */
    const editorFocused = () => {
        const a = document.activeElement;
        return !a || a === document.body || Boolean(a.closest && a.closest('.cm-editor, .menu, .quickpick'));
    };
    const til = () => { const d = app.editors.activeDoc(); return Boolean(d && d.isTil && view()) && editorFocused(); };
    reg('editor.definition', 'cmd.definition', () => app.lspActions.definition(), { keys: 'F12', category: 'cat.go', when: til });
    reg('editor.references', 'cmd.references', () => app.lspActions.references(), { keys: 'Shift+F12', category: 'cat.go', when: til });
    reg('editor.rename', 'cmd.rename', () => app.lspActions.rename(), { keys: 'F2', category: 'cat.edit', when: til });
    reg('editor.format', 'cmd.format', () => app.lspActions.format(), { keys: 'Shift+Alt+F', category: 'cat.edit', when: til });
    reg('editor.codeActions', 'cmd.codeActions', () => app.lspActions.codeActions(), { keys: 'Mod+.', category: 'cat.edit', when: til });
    reg('editor.splitRight', 'editor.splitRight', () => app.editors.splitRight(), { keys: 'Mod+\\', category: 'cat.view' });
    reg('editor.toggleErrorLens', 'cmd.toggleErrorLens', () => app.settings.set('errorLens', !app.settings.get('errorLens')), { category: 'cat.view' });
    reg('editor.toggleInlayHints', 'cmd.toggleInlayHints', () => app.settings.set('inlayHints', !app.settings.get('inlayHints')), { category: 'cat.view' });
    reg('editor.toggleWordWrap', 'cmd.toggleWordWrap', () => app.settings.set('wordWrap', !app.settings.get('wordWrap')), { keys: 'Alt+Z', category: 'cat.view' });
    reg('editor.nextProblem', 'cmd.nextProblem', () => gotoProblem(app, 1), { keys: 'F8', category: 'cat.go', when: () => Boolean(view()) });
    reg('editor.prevProblem', 'cmd.prevProblem', () => gotoProblem(app, -1), { keys: 'Shift+F8', category: 'cat.go', when: () => Boolean(view()) });

    /* run */
    reg('run.run', 'cmd.runFile', () => app.runFile(), { keys: ['Mod+F5', 'Mod+Enter'], category: 'cat.run' });
    for (const b of ['native', 'vm', 'cvm']) {
        reg(`run.run.${b}`, () => t('cmd.runOn', { backend: t(`backend.${b}`) }), () => app.runFile(null, { backend: b }), { category: 'cat.run', when: () => app.backendAvailable(b) });
    }
    reg('run.stop', 'cmd.stop', () => (debugging() ? dbg.stop() : app.runner.stop()), { keys: 'Shift+F5', category: 'cat.run', when: () => app.runner.running || debugging() });
    reg('run.snapshot', 'cmd.snapshot', () => app.runner.snapshot(), { category: 'cat.run', when: () => app.runner.running && !app.web });
    reg('run.pickBackend', 'cmd.pickBackend', () => app.pickBackend(app.layout.backendPicker.getBoundingClientRect()), { category: 'cat.run' });

    /* debug */
    reg('debug.start', 'cmd.debugFile', () => (stopped() ? dbg.continue() : debugging() ? null : app.debugFile()), { keys: 'F5', category: 'cat.debug', when: hasDebugger });
    reg('debug.continue', 'debug.continue', () => dbg.continue(), { category: 'cat.debug', when: stopped });
    reg('debug.pause', 'debug.pause', () => dbg.pause(), { keys: 'F6', category: 'cat.debug', when: () => debugging() && !dbg.stopped });
    reg('debug.stepOver', 'debug.stepOver', () => dbg.next(), { keys: 'F10', category: 'cat.debug', when: stopped });
    reg('debug.stepInto', 'debug.stepInto', () => dbg.stepIn(), { keys: 'F11', category: 'cat.debug', when: stopped });
    reg('debug.stepOut', 'debug.stepOut', () => dbg.stepOut(), { keys: 'Shift+F11', category: 'cat.debug', when: stopped });
    reg('debug.stepBack', 'debug.stepBack', () => dbg.stepBack(), { keys: 'Shift+F10', category: 'cat.debug', when: stopped });
    reg('debug.reverseContinue', 'debug.reverseContinue', () => dbg.reverseContinue(), { keys: 'Alt+F5', category: 'cat.debug', when: stopped });
    reg('debug.restart', 'debug.restart', () => dbg.restart(), { keys: 'Mod+Shift+F5', category: 'cat.debug', when: debugging });
    reg('debug.stop', 'debug.stop', () => dbg.stop(), { category: 'cat.debug', when: debugging });
    reg('debug.toggleBreakpoint', 'cmd.toggleBreakpoint', () => app.toggleBreakpoint(), { keys: 'F9', category: 'cat.debug', when: () => Boolean(view()) && editorFocused() });
    reg('debug.removeAllBreakpoints', 'debug.removeAllBps', () => app.views.debug.removeAllBreakpoints(), { category: 'cat.debug' });

    /* tools */
    reg('tools.prove', 'cmd.prove', () => { app.layout.showSide('proofs'); app.views.proofs.prove(); }, { category: 'cat.tools' });
    reg('tools.fit', 'cmd.fit', () => { app.layout.showSide('devices'); app.views.devices.check(); }, { category: 'cat.tools' });
    reg('tools.bytecode', 'cmd.bytecode', () => { app.layout.showSide('bytecode'); app.views.bytecode.load(); }, { category: 'cat.tools' });
    for (const target of ['vm', 'c-vm', 'native']) {
        reg(`tools.build.${target}`, () => t('cmd.build', { target: t(`build.${target}`) }), () => build(app, target), { category: 'cat.tools', when: () => !app.web });
    }
    for (const lang of ['en', 'kk', 'kk-latn', 'ru']) {
        reg(`translate.view.${lang}`, () => t('cmd.viewIn', { lang: t(`kwlang.${lang}`) }), () => { const d = app.activeTilDoc(); if (d) app.openTranslation(d.kind === 'file' ? d.path : d.path, lang); }, { category: 'cat.translate' });
        reg(`translate.replace.${lang}`, () => t('cmd.translateTo', { lang: t(`kwlang.${lang}`) }), () => app.translateInPlace(null, lang), { category: 'cat.translate' });
    }
    reg('translate.pick', 'cmd.translatePick', () => {
        const r = app.statusbar.kwlang.getBoundingClientRect();
        const d = app.activeTilDoc();
        if (d) openMenu(app.translateMenu(d.kind === 'file' ? d.path : null), r, { align: 'end' });
    }, { category: 'cat.translate' });

    /* help */
    reg('help.about', 'cmd.about', () => about(app), { category: 'cat.help' });
    reg('help.shortcuts', 'cmd.shortcuts', () => shortcuts(app), { keys: 'Mod+K', category: 'cat.help' });
    reg('help.docs', 'cmd.docs', () => window.open('https://github.com/damn-glitch/TIL/tree/main/docs', '_blank', 'noopener'), { category: 'cat.help' });

    /* title bar buttons */
    const L = app.layout;
    L.runBtn.addEventListener('click', () => C.run('run.run'));
    L.debugBtn.addEventListener('click', () => C.run('debug.start'));
    L.stopBtn.addEventListener('click', () => C.run('run.stop'));
    L.themeBtn.addEventListener('click', () => C.run('view.toggleTheme'));
    L.backendPicker.addEventListener('click', () => app.pickBackend(L.backendPicker.getBoundingClientRect()));
    L.devicePicker.addEventListener('click', () => app.pickDevice(L.devicePicker.getBoundingClientRect()));
    void EditorSelection;
}

/* ── the menu bar ── */
export function buildMenubar(app) {
    const C = app.commands;
    const item = (id, extra = {}) => {
        const c = C.get(id);
        if (!c) return null;
        return Object.assign({ label: C.title(id).replace(/^[^:]+: /, ''), keys: C.keysFor(id)[0], disabled: !C.enabled(id), run: () => C.run(id) }, extra);
    };
    const menus = [
        ['menu.file', () => [
            item('file.newUntitled', { icon: 'newFile' }), item('file.newFile'), item('file.newFolder', { icon: 'newFolder' }),
            { separator: true },
            item('workbench.quickOpen', { icon: 'search' }),
            { separator: true },
            item('file.save'), item('file.saveAll'), item('file.revert'),
            { separator: true },
            item('file.close'), item('file.closeAll'),
            { separator: true },
            item('workbench.show.settings', { icon: 'settings' }),
        ]],
        ['menu.edit', () => [
            item('edit.undo'), item('edit.redo'), { separator: true },
            item('edit.find', { keys: 'Mod+F' }), item('edit.replace'), item('workbench.show.search', { label: t('edit.findInFiles') }),
            { separator: true },
            item('edit.toggleComment', { keys: 'Mod+/' }), item('editor.format'), item('editor.rename'), item('editor.codeActions'),
            { separator: true },
            item('edit.selectAll', { keys: 'Mod+A' }), item('edit.foldAll'), item('edit.unfoldAll'),
        ]],
        ['menu.view', () => [
            item('workbench.commandPalette', { icon: 'keyboard' }),
            { separator: true },
            item('workbench.show.explorer', { icon: 'files' }), item('workbench.show.search', { icon: 'search' }), item('workbench.show.debug', { icon: 'bug' }),
            item('workbench.show.problems', { icon: 'problems' }), item('workbench.show.proofs', { icon: 'proofs' }), item('workbench.show.devices', { icon: 'devices' }),
            item('workbench.show.packages', { icon: 'packages' }), item('workbench.show.bytecode', { icon: 'bytecode' }),
            { separator: true },
            item('workbench.toggleSidebar'), item('workbench.togglePanel'), item('editor.splitRight', { icon: 'split' }),
            { separator: true },
            item('view.toggleTheme', { icon: app.effectiveTheme() === 'dark' ? 'sun' : 'moon' }),
            { label: t('settings.uiLang'), icon: 'globe', submenu: LOCALES.map((l) => item(`view.uiLang.${l}`, { label: LOCALE_NAMES[l] })) },
            { label: t('explorer.translate'), icon: 'translate', submenu: ['en', 'kk', 'kk-latn', 'ru'].map((l) => item(`translate.view.${l}`)) },
            { separator: true },
            item('editor.toggleErrorLens', { checked: app.settings.get('errorLens') }), item('editor.toggleInlayHints', { checked: app.settings.get('inlayHints') }),
            item('editor.toggleWordWrap', { checked: app.settings.get('wordWrap') }),
            { separator: true },
            item('view.zoomIn'), item('view.zoomOut'), item('view.zoomReset'),
        ]],
        ['menu.run', () => [
            item('run.run', { icon: 'play' }), item('debug.start', { icon: 'bug' }), item('run.stop', { icon: 'stop' }),
            { label: t('cmd.runWith'), icon: 'layers', submenu: ['native', 'vm', 'cvm'].map((b) => item(`run.run.${b}`)) },
            { separator: true },
            item('debug.stepOver'), item('debug.stepInto'), item('debug.stepOut'), item('debug.stepBack'), item('debug.reverseContinue'),
            { separator: true },
            item('debug.toggleBreakpoint', { icon: 'breakpoint' }), item('debug.removeAllBreakpoints'),
            { separator: true },
            item('tools.prove', { icon: 'proofs' }), item('tools.fit', { icon: 'devices' }), item('tools.bytecode', { icon: 'bytecode' }),
            app.web ? null : { label: t('cmd.buildMenu'), icon: 'download', submenu: ['vm', 'c-vm', 'native'].map((x) => item(`tools.build.${x}`)) },
            item('run.snapshot', { icon: 'camera' }),
        ]],
        ['menu.help', () => [
            item('workbench.welcome', { icon: 'home' }), item('help.shortcuts', { icon: 'keyboard' }), item('help.docs', { icon: 'book' }),
            { separator: true },
            item('help.about', { icon: 'info' }),
        ]],
    ];
    const bar = app.layout.menubar;
    const buttons = [];
    let openIndex = -1;
    const more = app.layout.menuMore;
    if (more && !more._wired) {
        more._wired = true;
        more.addEventListener('click', () => {
            more.setAttribute('aria-expanded', 'true');
            const all = app.layout.menuItems();
            const menu = openMenu(all, more.getBoundingClientRect(), { label: t('menu.bar'), onClose: () => more.setAttribute('aria-expanded', 'false') });
            if (menu.focusFirst) menu.focusFirst();
        });
    }
    app.layout.menuItems = () => menus.map(([key, items]) => ({ label: t(key), submenu: items().filter(Boolean) }));
    const open = (i) => {
        const [key, items] = menus[i];
        const btn = buttons[i];
        btn.setAttribute('aria-expanded', 'true');
        openIndex = i;
        const menu = openMenu(items(), btn.getBoundingClientRect(), {
            label: t(key),
            onClose: () => { btn.setAttribute('aria-expanded', 'false'); if (openIndex === i) openIndex = -1; },
        });
        menu.el.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' && !e.target.closest('[aria-haspopup="menu"]')) { e.preventDefault(); menu.close(false); open((i + 1) % menus.length); }
            else if (e.key === 'ArrowLeft') { e.preventDefault(); menu.close(false); open((i - 1 + menus.length) % menus.length); }
        });
        if (menu.focusFirst) menu.focusFirst();
    };
    menus.forEach(([key], i) => {
        const b = h('button', { type: 'button', role: 'menuitem', 'aria-haspopup': 'menu', 'aria-expanded': 'false', tabindex: i ? '-1' : '0' }, t(key));
        b.addEventListener('mousedown', (e) => { e.preventDefault(); if (openIndex === i) { document.querySelector('.menu') && document.querySelector('.menu').remove(); openIndex = -1; b.setAttribute('aria-expanded', 'false'); } else open(i); });
        b.addEventListener('mouseenter', () => { if (openIndex >= 0 && openIndex !== i) open(i); });
        b.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') { e.preventDefault(); open(i); }
            if (e.key === 'ArrowRight') { e.preventDefault(); buttons[(i + 1) % buttons.length].focus(); }
            if (e.key === 'ArrowLeft') { e.preventDefault(); buttons[(i - 1 + buttons.length) % buttons.length].focus(); }
        });
        buttons.push(b);
        bar.appendChild(b);
    });
}

/* ── the palette ── */
export function openPalette(app, prefix) {
    const C = app.commands;
    const commandItems = () => {
        const recent = C.recent.filter((id) => C.get(id) && C.enabled(id));
        const items = [];
        for (const id of recent) items.push({ label: C.title(id), keys: C.keysFor(id)[0], value: { cmd: id }, group: t('palette.recent'), boost: 40 });
        for (const c of C.list.values()) {
            if (c.palette === false || recent.includes(c.id) || !C.enabled(c.id)) continue;
            items.push({ label: C.title(c.id), keys: C.keysFor(c.id)[0], value: { cmd: c.id }, group: recent.length ? t('palette.other') : undefined });
        }
        return items;
    };
    const fileItems = async () => {
        const open = app.editors.openDocs().filter((d) => d.kind === 'file');
        const files = await app.fileIndex.all();
        const items = [];
        for (const d of open) items.push({ label: d.name, description: dirname(d.path), icon: fileIcon(d.name), iconClass: isTil(d.name) ? 'til' : '', value: { path: d.path }, group: t('palette.openEditors'), filterText: d.path, boost: 30 });
        for (const p of files) {
            if (open.some((d) => d.path === p)) continue;
            items.push({ label: basename(p), description: dirname(p), icon: fileIcon(p), iconClass: isTil(p) ? 'til' : '', value: { path: p }, filterText: p });
        }
        return items;
    };
    const lineMode = (q) => {
        const v = app.editors.activeView();
        if (!v) return [];
        const n = parseInt(q, 10);
        const total = v.state.doc.lines;
        const items = [{ label: Number.isFinite(n) ? t('palette.gotoLineN', { line: n, total }) : t('palette.gotoLineHint', { total }), value: { line: n }, icon: 'arrowRight' }];
        items.noFilter = true;
        return items;
    };
    lineMode.accept = (q) => {
        const [l, c] = String(q).split(/[:,]/).map((x) => parseInt(x, 10));
        if (Number.isFinite(l)) gotoLine(app, l, c);
    };
    quickPick({
        placeholder: prefix === '>' ? t('palette.commandsPlaceholder') : prefix === ':' ? t('palette.linePlaceholder') : prefix === '@' ? t('palette.symbolPlaceholder') : t('palette.filesPlaceholder'),
        value: prefix,
        ariaLabel: t('palette.title'),
        modes: {
            '': fileItems,
            '>': commandItems,
            ':': lineMode,
            '@': () => app.lspActions.symbolItems(),
            '#': (q) => app.lspActions.workspaceSymbolItems(q),
            '?': () => {
                const items = [['', 'palette.helpFiles'], ['>', 'palette.helpCommands'], [':', 'palette.helpLine'], ['@', 'palette.helpSymbol'], ['#', 'palette.helpWorkspace']]
                    .map(([p, k]) => ({ label: `${p || '…'}  ${t(k)}`, value: { prefix: p } }));
                items.noFilter = true;
                return items;
            },
        },
        onAccept: (it, { prefix: p }) => {
            const v = it.value || {};
            if (v.cmd) C.run(v.cmd);
            else if (v.path) app.openFile(v.path);
            else if (v.line !== undefined) gotoLine(app, v.line);
            else if (v.prefix !== undefined) openPalette(app, v.prefix);
            else if (v.doc && v.range) app.openLocation(v.doc.path, v.range);
            else if (v.loc) app.lspActions.goto(v.loc);
            void p;
        },
    });
}

function gotoLine(app, line, col) {
    const v = app.editors.activeView();
    const d = app.editors.activeDoc();
    if (!v || !d || !Number.isFinite(line)) return;
    app.editors.show(d, { line: Math.max(1, Math.min(v.state.doc.lines, line)), col: col || 1 });
}

function gotoProblem(app, dir) {
    const v = app.editors.activeView();
    const d = app.editors.activeDoc();
    if (!v || !d) return;
    const list = (app.diagnostics.get(d.uri) || []).map((x) => x.range).sort((a, b) => a.start.line - b.start.line || a.start.character - b.start.character);
    if (!list.length) { app.statusbar.flash(t('problems.none')); return; }
    const head = v.state.selection.main.head;
    const cur = v.state.doc.lineAt(head);
    const pos = { line: cur.number - 1, character: head - cur.from };
    const cmp = (r) => r.start.line - pos.line || r.start.character - pos.character;
    let target = dir > 0 ? list.find((r) => cmp(r) > 0) || list[0] : [...list].reverse().find((r) => cmp(r) < 0) || list[list.length - 1];
    app.openLocation(d.path, target);
}

async function build(app, target) {
    const d = app.activeTilDoc();
    if (!d || d.kind !== 'file') { toast(t('run.noFile'), { kind: 'info' }); return; }
    if (d.dirty) await app.saveDoc(d);
    app.output.clear();
    app.output.header(t('cmd.build', { target: t(`build.${target}`) }), basename(d.path));
    app.layout.showPanel('output');
    try {
        const body = { path: d.path, target };
        if (target !== 'native') body.device = app.settings.get('device');
        const r = await app.backend.build(body);
        if (r.output) app.output.write(String(r.output).endsWith('\n') ? r.output : r.output + '\n', r.ok ? 1 : 2);
        app.output.footer({ ok: Boolean(r.ok), text: r.ok ? t('build.done', { artifact: r.artifact || '' }) : t('build.failed') });
        if (r.ok) app.views.explorer.refresh();
    } catch (e) {
        if (e && e.notAvailable) { app.showDesktopOnly(e.feature); return; }
        app.output.footer({ ok: false, text: t('build.failed') + ': ' + e.message });
    }
}

function about(app) {
    const info = app.info;
    const caps = info.capabilities || {};
    const rows = [
        [t('about.version'), info.version || app.tables.version],
        [t('about.face'), app.web ? t('status.web') : t('status.desktop')],
        [t('about.project'), info.root || info.rootName],
        [t('about.platform'), info.platform || navigator.platform],
        [t('about.backends'), ['vm', caps.native && 'native', caps.cvm && 'cvm'].filter(Boolean).map((b) => t(`backend.${b}`)).join(', ')],
        [t('about.tools'), [caps.debugger && t('feature.debugger'), caps.z3 && t('feature.prover'), caps.pkg && t('feature.packages')].filter(Boolean).join(', ') || '—'],
        [t('about.lsp'), app.lsp && app.lsp.serverInfo ? `${app.lsp.serverInfo.name} ${app.lsp.serverInfo.version || ''}` : app.lsp && app.lsp.ready ? '✓' : '—'],
    ];
    const grid = h('div', { class: 'about-grid' });
    for (const [k, v] of rows) grid.append(h('span', { class: 'muted' }, k), h('span', {}, String(v || '—')));
    customDialog({
        title: 'TIL Studio',
        body: h('div', { class: 'dialog-body about' }, h('div', { class: 'about-head' }, logoMark(44), h('div', {}, h('div', { class: 'about-name' }, 'TIL Studio'), h('div', { class: 'muted' }, t('about.tagline')))), grid,
            h('p', { class: 'muted small' }, t('about.license'))),
        buttons: [{ label: t('dialog.close'), value: true, primary: true }],
    });
}

function shortcuts(app) {
    const C = app.commands;
    const ids = ['workbench.commandPalette', 'workbench.quickOpen', 'workbench.gotoSymbol', 'editor.gotoLine', 'run.run', 'debug.start', 'run.stop', 'debug.toggleBreakpoint',
        'debug.stepOver', 'debug.stepInto', 'debug.stepOut', 'debug.stepBack', 'debug.reverseContinue', 'editor.definition', 'editor.references', 'editor.rename',
        'editor.format', 'editor.codeActions', 'workbench.togglePanel', 'workbench.show.search', 'file.save', 'editor.splitRight'];
    const grid = h('div', { class: 'shortcut-grid' });
    for (const id of ids) {
        const k = C.keysFor(id)[0];
        if (!k) continue;
        grid.append(h('span', {}, C.title(id).replace(/^[^:]+: /, '')), h('span', { class: 'keys' }, h('kbd', {}, formatK(k))));
    }
    customDialog({ title: t('cmd.shortcuts'), body: h('div', { class: 'dialog-body' }, grid), buttons: [{ label: t('dialog.close'), value: true, primary: true }], wide: true });
}

import { formatKeys as formatK } from './core/commands.js';
export { icon, EditorView, flashEffect };
