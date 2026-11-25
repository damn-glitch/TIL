// The shell: title bar, activity bar + side panel, editor area, bottom panel,
// status bar, and the sashes between them (sizes are kept per project).
import { h, clear, focusSoon } from '../core/dom.js';
import { icon, logoMark } from '../core/icons.js';
import { L, t, setText, bindAttr } from '../core/i18n.js';
import { tip } from './widgets.js';

export class Layout {
    constructor(app) {
        this.app = app;
        this.sideViews = new Map();
        this.panels = new Map();
        this.activeSide = null;
        this.activePanel = null;
        this.build();
    }

    build() {
        const app = this.app;
        this.root = h('div', { class: 'studio', 'data-sidebar': 'open', 'data-panel': 'open' });

        /* title bar */
        this.menubar = h('nav', { class: 'menubar', role: 'menubar', 'aria-label': t('menu.bar') });
        // when the menus do not fit (narrow windows, long labels): one button with all of them
        this.menuMore = h('button', { class: 'icon-btn tb-menu-more', type: 'button', 'aria-haspopup': 'menu' }, icon('menu'));
        bindAttr(this.menuMore, 'aria-label', 'menu.bar');
        tip(this.menuMore, 'menu.bar');
        this.projectLabel = h('span', { class: 'tb-project' });
        this.searchHint = L('titlebar.search');
        this.searchHint.className = 'tb-search-hint';
        this.searchKbd = h('kbd', {}, '');
        this.searchBtn = h('button', { class: 'tb-search', type: 'button' }, icon('search'), this.projectLabel, this.searchHint, this.searchKbd);
        tip(this.searchBtn, 'titlebar.searchTip');
        this.backendPicker = this.picker('vm', 'titlebar.backendTip');
        this.devicePicker = this.picker('devices', 'titlebar.deviceTip');
        this.runBtn = h('button', { class: 'btn primary tb-run', type: 'button' }, icon('play'), L('titlebar.run'));
        tip(this.runBtn, 'titlebar.runTip', 'Mod+F5');
        this.debugBtn = h('button', { class: 'icon-btn tb-debug', type: 'button' }, icon('bug'));
        tip(this.debugBtn, 'titlebar.debugTip', 'F5');
        this.stopBtn = h('button', { class: 'btn danger tb-stop', type: 'button', hidden: true }, icon('stop'), L('titlebar.stop'));
        tip(this.stopBtn, 'titlebar.stopTip', 'Shift+F5');
        this.themeBtn = h('button', { class: 'icon-btn tb-theme', type: 'button' }, icon('moon'));
        tip(this.themeBtn, 'titlebar.themeTip');
        this.titlebar = h('header', { class: 'titlebar', role: 'banner' },
            h('div', { class: 'tb-left' },
                h('div', { class: 'tb-brand' }, logoMark(22), h('span', { class: 'tb-name' }, 'TIL ', h('b', {}, 'Studio'))),
                this.menuMore, this.menubar),
            h('div', { class: 'tb-center' }, this.searchBtn),
            h('div', { class: 'tb-right' },
                h('div', { class: 'tb-pickers' }, this.backendPicker, this.devicePicker),
                h('span', { class: 'tb-sep' }),
                this.stopBtn, this.runBtn, this.debugBtn,
                h('span', { class: 'tb-sep' }),
                this.themeBtn));

        /* workbench */
        this.activity = h('nav', { class: 'activitybar', role: 'tablist', 'aria-orientation': 'vertical', 'aria-label': t('activity.label') });
        this.activityTop = h('div', { class: 'ab-group', style: 'display:contents' });
        this.activityBottom = h('div', { class: 'ab-group', style: 'display:contents' });
        this.activity.append(this.activityTop, h('div', { class: 'spacer' }), this.activityBottom);
        this.activity.addEventListener('keydown', (e) => this.activityKeys(e));
        this.sidebar = h('aside', { class: 'sidebar', 'aria-label': t('activity.sidePanel') });
        this.sideSash = h('div', { class: 'sash sash-v sash-side', role: 'separator', 'aria-orientation': 'vertical', tabindex: '0', 'aria-label': t('layout.resizeSide') });
        this.editorArea = h('section', { class: 'editor-area', 'aria-label': t('layout.editors') });
        this.panelSash = h('div', { class: 'sash sash-h sash-panel', role: 'separator', 'aria-orientation': 'horizontal', tabindex: '0', 'aria-label': t('layout.resizePanel') });
        this.panelTabs = h('div', { class: 'panel-tabs', role: 'tablist', 'aria-label': t('panel.label') });
        this.panelTabs.addEventListener('keydown', (e) => this.panelKeys(e));
        this.panelActions = h('div', { class: 'panel-actions' });
        this.panelBody = h('div', { class: 'panel-body' });
        const maxBtn = h('button', { class: 'icon-btn small', type: 'button' }, icon('maximize'));
        tip(maxBtn, 'panel.maximize');
        maxBtn.addEventListener('click', () => this.togglePanelMax());
        this.panelMaxBtn = maxBtn;
        const closeBtn = h('button', { class: 'icon-btn small', type: 'button' }, icon('close'));
        tip(closeBtn, 'panel.close', 'Ctrl+`');
        closeBtn.addEventListener('click', () => this.setPanelOpen(false));
        this.panel = h('section', { class: 'panel', 'aria-label': t('panel.label') },
            h('div', { class: 'panel-head' }, this.panelTabs, this.panelActions, h('span', { class: 'tb-sep' }), maxBtn, closeBtn),
            this.panelBody);
        this.main = h('main', { class: 'main' }, this.editorArea, this.panelSash, this.panel);
        this.workbench = h('div', { class: 'workbench' }, this.activity, this.sidebar, this.sideSash, this.main);

        /* status bar */
        this.statusLeft = h('div', { class: 'sb-left' });
        this.statusRight = h('div', { class: 'sb-right' });
        this.statusbar = h('footer', { class: 'statusbar', role: 'contentinfo' }, this.statusLeft, this.statusRight);

        this.root.append(this.titlebar, this.workbench, this.statusbar);
        this.installSashes();
        if (typeof ResizeObserver === 'function') new ResizeObserver(() => this.fitTitlebar()).observe(this.titlebar.querySelector('.tb-left'));
        // narrow windows: the side bar floats over the editor, so working in
        // the editor or the panel puts it away
        this.main.addEventListener('pointerdown', () => {
            if (this.sideOpen() && window.matchMedia && window.matchMedia('(max-width: 1023.98px)').matches) this.hideSide();
        });
        this.searchBtn.addEventListener('click', () => app.commands.run('workbench.quickOpen'));
    }

    /* the menu bar folds into one button when its menus do not fit */
    fitTitlebar() {
        const tb = this.titlebar;
        tb.classList.remove('compact');
        if (this.menubar.scrollWidth > this.menubar.clientWidth + 1) tb.classList.add('compact');
    }

    picker(iconName, tipKey) {
        const b = h('button', { class: 'tb-picker', type: 'button', 'aria-haspopup': 'menu' },
            icon(iconName), h('span', { class: 'tb-picker-label' }), icon('chevronDown', { className: 'chev' }));
        tip(b, tipKey);
        return b;
    }

    setPickerLabel(btn, text, iconName) {
        btn.querySelector('.tb-picker-label').textContent = text;
        if (iconName) {
            const old = btn.querySelector('.icon:not(.chev)');
            old.replaceWith(icon(iconName));
        }
    }

    mount(parent) {
        parent.classList.remove('app-boot');
        clear(parent).appendChild(this.root);
    }

    /* ── activity bar + side panel ── */
    addActivity(id, { iconName, titleKey, keys, bottom = false, view }) {
        const btn = h('button', { class: 'ab-btn', type: 'button', role: 'tab', 'aria-selected': 'false', tabindex: '-1', dataset: { view: id }, 'aria-controls': `side-${id}` }, icon(iconName));
        tip(btn, titleKey, keys, 'right');
        btn.addEventListener('click', () => this.toggleSide(id));
        (bottom ? this.activityBottom : this.activityTop).appendChild(btn);
        if (view) {
            view.el.id = `side-${id}`;
            view.el.setAttribute('role', 'tabpanel');
            view.el.hidden = true;
            this.sidebar.appendChild(view.el);
            this.sideViews.set(id, { btn, view });
        }
        return btn;
    }

    activityButtons() { return [...this.activity.querySelectorAll('.ab-btn')]; }

    activityKeys(e) {
        const btns = this.activityButtons();
        const i = btns.indexOf(document.activeElement);
        if (i < 0) return;
        let j = -1;
        if (e.key === 'ArrowDown') j = (i + 1) % btns.length;
        else if (e.key === 'ArrowUp') j = (i - 1 + btns.length) % btns.length;
        else if (e.key === 'Home') j = 0;
        else if (e.key === 'End') j = btns.length - 1;
        if (j >= 0) { e.preventDefault(); btns[j].focus(); }
    }

    setBadge(id, n, kind = '') {
        const entry = this.sideViews.get(id);
        if (!entry) return;
        let b = entry.btn.querySelector('.badge');
        if (!n) { if (b) b.remove(); return; }
        if (!b) { b = h('span', { class: 'badge' }); entry.btn.appendChild(b); }
        b.className = `badge ${kind}`;
        b.textContent = n > 99 ? '99+' : String(n);
    }

    setDot(id, on) {
        const entry = this.sideViews.get(id);
        if (!entry) return;
        let d = entry.btn.querySelector('.ab-dot');
        if (!on) { if (d) d.remove(); return; }
        if (!d) entry.btn.appendChild(h('span', { class: 'ab-dot' }));
    }

    sideOpen() { return this.root.dataset.sidebar !== 'closed'; }

    showSide(id, { focus = true } = {}) {
        const entry = this.sideViews.get(id);
        if (!entry) return;
        for (const [vid, e] of this.sideViews) {
            const on = vid === id;
            e.view.el.hidden = !on;
            e.btn.setAttribute('aria-selected', String(on));
            e.btn.tabIndex = on ? 0 : -1;
        }
        this.activeSide = id;
        this.root.dataset.sidebar = 'open';
        if (entry.view.onShow) entry.view.onShow();
        if (focus && entry.view.focus) focusSoon(() => entry.view.focus(), entry.view.el);
        this.app.saveLayout();
    }

    toggleSide(id) {
        if (this.activeSide === id && this.sideOpen()) this.hideSide();
        else this.showSide(id);
    }

    hideSide() {
        this.root.dataset.sidebar = 'closed';
        for (const e of this.sideViews.values()) e.btn.setAttribute('aria-selected', 'false');
        this.app.saveLayout();
    }

    /* ── bottom panel ── */
    addPanel(id, { titleKey, view }) {
        const tab = h('button', { class: 'panel-tab', type: 'button', role: 'tab', 'aria-selected': 'false', tabindex: '-1', dataset: { panel: id }, 'aria-controls': `panel-${id}` }, L(titleKey));
        tab.addEventListener('click', () => this.showPanel(id));
        this.panelTabs.appendChild(tab);
        view.el.id = `panel-${id}`;
        view.el.classList.add('panel-view');
        view.el.setAttribute('role', 'tabpanel');
        view.el.hidden = true;
        this.panelBody.appendChild(view.el);
        this.panels.set(id, { tab, view });
    }

    panelKeys(e) {
        const tabs = [...this.panelTabs.children];
        const i = tabs.indexOf(document.activeElement);
        if (i < 0) return;
        let j = -1;
        if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
        else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
        if (j >= 0) { e.preventDefault(); tabs[j].focus(); this.showPanel(tabs[j].dataset.panel); }
    }

    setPanelBadge(id, n, kind = '') {
        const entry = this.panels.get(id);
        if (!entry) return;
        let b = entry.tab.querySelector('.badge');
        if (!n) { if (b) b.remove(); return; }
        if (!b) { b = h('span', { class: 'badge' }); entry.tab.appendChild(b); }
        b.className = `badge ${kind}`;
        b.textContent = n > 999 ? '999+' : String(n);
    }

    showPanel(id, { focus = false } = {}) {
        const entry = this.panels.get(id);
        if (!entry) return;
        for (const [pid, e] of this.panels) {
            const on = pid === id;
            e.view.el.hidden = !on;
            e.tab.setAttribute('aria-selected', String(on));
            e.tab.tabIndex = on ? 0 : -1;
        }
        this.activePanel = id;
        clear(this.panelActions);
        if (entry.view.actions) for (const a of entry.view.actions()) this.panelActions.appendChild(a);
        this.setPanelOpen(true);
        if (entry.view.onShow) entry.view.onShow();
        if (focus && entry.view.focus) focusSoon(() => entry.view.focus(), entry.view.el);
        this.app.saveLayout();
    }

    panelOpen() { return this.root.dataset.panel !== 'closed'; }

    setPanelOpen(open) {
        if (open) { if (this.root.dataset.panel === 'closed') this.root.dataset.panel = 'open'; }
        else this.root.dataset.panel = 'closed';
        this.app.saveLayout();
        this.app.editorsResized();
    }

    togglePanel() {
        if (this.panelOpen()) this.setPanelOpen(false);
        else this.showPanel(this.activePanel || 'output', { focus: true });
    }

    togglePanelMax() {
        const max = this.root.dataset.panel === 'max';
        this.root.dataset.panel = max ? 'open' : 'max';
        this.panelMaxBtn.replaceChildren(icon(max ? 'maximize' : 'restore'));
        tip(this.panelMaxBtn, max ? 'panel.maximize' : 'panel.restore');
        this.app.editorsResized();
    }

    /* ── sizes ── */
    setSizes({ side, panel } = {}) {
        if (side) this.root.style.setProperty('--sidebar-w', `${Math.max(180, Math.min(side, window.innerWidth * 0.7))}px`);
        if (panel) this.root.style.setProperty('--panel-h', `${Math.max(120, Math.min(panel, window.innerHeight - 180))}px`);
    }

    sizes() {
        return { side: this.sidebar.offsetWidth || 280, panel: this.panel.offsetHeight || 260 };
    }

    installSashes() {
        const drag = (sash, axis, onMove) => {
            sash.addEventListener('pointerdown', (e) => {
                if (e.button !== 0) return;
                e.preventDefault();
                sash.setPointerCapture(e.pointerId);
                sash.classList.add('dragging');
                document.body.classList.add('resizing', axis === 'x' ? 'resizing-col' : 'resizing-row');
                const start = axis === 'x' ? e.clientX : e.clientY;
                const base = axis === 'x' ? this.sidebar.offsetWidth : this.panel.offsetHeight;
                const move = (ev) => onMove(base, (axis === 'x' ? ev.clientX : ev.clientY) - start);
                const up = () => {
                    sash.classList.remove('dragging');
                    document.body.classList.remove('resizing', 'resizing-col', 'resizing-row');
                    sash.removeEventListener('pointermove', move);
                    sash.removeEventListener('pointerup', up);
                    sash.removeEventListener('pointercancel', up);
                    this.app.saveLayout();
                    this.app.editorsResized();
                };
                sash.addEventListener('pointermove', move);
                sash.addEventListener('pointerup', up);
                sash.addEventListener('pointercancel', up);
            });
        };
        drag(this.sideSash, 'x', (base, d) => this.setSizes({ side: base + d }));
        drag(this.panelSash, 'y', (base, d) => { this.setSizes({ panel: base - d }); this.app.editorsResized(); });
        this.sideSash.addEventListener('keydown', (e) => {
            const step = e.shiftKey ? 40 : 10;
            if (e.key === 'ArrowLeft') { e.preventDefault(); this.setSizes({ side: this.sidebar.offsetWidth - step }); this.app.saveLayout(); }
            if (e.key === 'ArrowRight') { e.preventDefault(); this.setSizes({ side: this.sidebar.offsetWidth + step }); this.app.saveLayout(); }
        });
        this.panelSash.addEventListener('keydown', (e) => {
            const step = e.shiftKey ? 40 : 10;
            if (e.key === 'ArrowUp') { e.preventDefault(); this.setSizes({ panel: this.panel.offsetHeight + step }); this.app.saveLayout(); this.app.editorsResized(); }
            if (e.key === 'ArrowDown') { e.preventDefault(); this.setSizes({ panel: this.panel.offsetHeight - step }); this.app.saveLayout(); this.app.editorsResized(); }
        });
    }

    /* status bar item: a button (with onClick) or a span */
    statusItem({ side = 'left', className = '', onClick, tipKey, order }) {
        const el = h(onClick ? 'button' : 'span', { class: `sb-item ${className}`, type: onClick ? 'button' : undefined });
        if (onClick) el.addEventListener('click', onClick);
        if (tipKey) tip(el, tipKey, null, 'above');
        if (order !== undefined) el.style.order = String(order);
        (side === 'left' ? this.statusLeft : this.statusRight).appendChild(el);
        return el;
    }
}

export function setLabel(el, key, params) { return setText(el, key, params); }
