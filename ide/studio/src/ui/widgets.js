// Overlays: context menus (ARIA menu), modal dialogs (confirm / prompt),
// toasts with a notification history, and hover tips.
import { h, placeNear, uniqueId, focusSoon } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, onLocale } from '../core/i18n.js';
import { formatKeys } from '../core/commands.js';
import { Emitter } from '../core/events.js';

let layer = null;
function overlayLayer() {
    if (!layer) {
        layer = h('div', { class: 'overlay-layer' });
        document.body.appendChild(layer);
    }
    return layer;
}

/* ── context menus ─────────────────────────────────────────────────── */
let openMenus = [];

export function closeMenus() {
    for (const m of openMenus.slice().reverse()) m.close(false);
    openMenus = [];
}

/**
 * items: [{label, icon, keys, run, disabled, checked, danger, separator, submenu, id}]
 * at: a DOMRect or {x, y}
 */
export function openMenu(items, at, { onClose, parent = null, align = 'start', label = '' } = {}) {
    if (!parent) closeMenus();
    const restore = document.activeElement;
    const el = h('div', { class: 'menu', role: 'menu', tabindex: '-1', 'aria-label': label || undefined });
    const rows = [];
    for (const it of items) {
        if (!it) continue;
        if (it.separator) { el.appendChild(h('div', { class: 'menu-sep', role: 'separator' })); continue; }
        if (it.header) { el.appendChild(h('div', { class: 'menu-header', role: 'presentation' }, it.header)); continue; }
        const row = h('div', {
            class: ['menu-item', it.danger && 'danger', it.disabled && 'disabled'],
            role: it.checked !== undefined ? 'menuitemcheckbox' : 'menuitem',
            tabindex: '-1',
            'aria-disabled': it.disabled ? 'true' : undefined,
            'aria-checked': it.checked !== undefined ? String(Boolean(it.checked)) : undefined,
            'aria-haspopup': it.submenu ? 'menu' : undefined,
            dataset: it.id ? { id: it.id } : undefined,
        },
        h('span', { class: 'menu-icon' }, it.checked ? icon('check') : it.icon ? icon(it.icon) : null),
        h('span', { class: 'menu-label' }, it.label),
        it.keys ? h('span', { class: 'menu-keys' }, formatKeys(it.keys)) : null,
        it.submenu ? icon('chevronRight', { className: 'menu-sub' }) : null);
        row._item = it;
        rows.push(row);
        row.addEventListener('mouseenter', () => focusRow(rows.indexOf(row), false));
        row.addEventListener('click', (e) => { e.stopPropagation(); activate(row); });
        el.appendChild(row);
    }
    overlayLayer().appendChild(el);
    const rect = at instanceof DOMRect || (at && at.width !== undefined) ? at : new DOMRect(at.x, at.y, 0, 0);
    if (parent) {
        el.style.left = '0px';
        el.style.top = '0px';
        const w = el.offsetWidth;
        let x = rect.right + 2;
        if (x + w > window.innerWidth - 8) x = rect.left - w - 2;
        el.style.left = `${Math.max(8, x)}px`;
        el.style.top = `${Math.max(8, Math.min(rect.top - 4, window.innerHeight - el.offsetHeight - 8))}px`;
    } else placeNear(el, rect, { align });
    let index = -1;
    let sub = null;
    const menu = {
        el,
        close(returnFocus = true) {
            if (sub) sub.close(false);
            el.remove();
            openMenus = openMenus.filter((m) => m !== menu);
            document.removeEventListener('mousedown', outside, true);
            if (returnFocus && restore && restore.focus && !parent) restore.focus({ preventScroll: true });
            if (onClose) onClose();
        },
    };
    function focusRow(i, openSub = false) {
        if (!rows.length) return;
        index = (i + rows.length) % rows.length;
        rows.forEach((r, k) => r.classList.toggle('active', k === index));
        rows[index].focus({ preventScroll: true });
        if (sub && sub.owner !== rows[index]) { sub.close(false); sub = null; }
        if (openSub && rows[index]._item.submenu) openSubmenu(rows[index]);
    }
    function openSubmenu(row) {
        if (sub && sub.owner === row) return;
        if (sub) sub.close(false);
        sub = openMenu(row._item.submenu, row.getBoundingClientRect(), { parent: menu, onClose: () => { sub = null; } });
        sub.owner = row;
    }
    function activate(row) {
        const it = row._item;
        if (it.disabled) return;
        if (it.submenu) { openSubmenu(row); sub && sub.focusFirst(); return; }
        closeMenus();
        if (parent) parent.close(false);
        if (restore && restore.focus) restore.focus({ preventScroll: true });
        setTimeout(() => it.run && it.run(), 0);
    }
    menu.focusFirst = () => focusRow(0);
    el.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown') { e.preventDefault(); focusRow(index + 1); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); focusRow(index - 1); }
        else if (e.key === 'Home') { e.preventDefault(); focusRow(0); }
        else if (e.key === 'End') { e.preventDefault(); focusRow(rows.length - 1); }
        else if (e.key === 'ArrowRight') { e.preventDefault(); if (rows[index] && rows[index]._item.submenu) { openSubmenu(rows[index]); sub && sub.focusFirst(); } }
        else if (e.key === 'ArrowLeft') { e.preventDefault(); if (parent) menu.close(false), parent.refocus(); }
        else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); if (rows[index]) activate(rows[index]); }
        else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); menu.close(true); }
        else if (e.key === 'Tab') { e.preventDefault(); closeMenus(); }
        else if (e.key.length === 1 && /\S/.test(e.key)) {
            const ch = e.key.toLowerCase();
            const n = rows.length;
            for (let k = 1; k <= n; k++) {
                const r = rows[(index + k) % n];
                if (r.textContent.trim().toLowerCase().startsWith(ch)) { focusRow((index + k) % n); break; }
            }
        }
    });
    menu.refocus = () => focusRow(index < 0 ? 0 : index);
    function outside(e) {
        if (!openMenus.some((m) => m.el.contains(e.target))) closeMenus();
    }
    setTimeout(() => document.addEventListener('mousedown', outside, true), 0);
    openMenus.push(menu);
    // focus the menu now (keys typed right after opening it go to it) and
    // after the next frame in case the opening click moved the focus
    const grab = () => {
        if (!el.isConnected || el.contains(document.activeElement)) return;
        if (index >= 0 && rows[index]) rows[index].focus({ preventScroll: true });
        else { el.focus({ preventScroll: true }); if (parent) focusRow(0); }
    };
    grab();
    requestAnimationFrame(grab);
    return menu;
}

/* ── modal dialogs ─────────────────────────────────────────────────── */
function modal({ title, body, buttons, onKey, labelId, initialFocus, wide = false }) {
    const restore = document.activeElement;
    const scrim = h('div', { class: 'scrim' });
    const box = h('div', { class: ['dialog', wide && 'wide'], role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': labelId });
    box.appendChild(h('h2', { class: 'dialog-title', id: labelId }, title));
    if (body) box.appendChild(body);
    const foot = h('div', { class: 'dialog-buttons' });
    for (const b of buttons) foot.appendChild(b);
    box.appendChild(foot);
    scrim.appendChild(box);
    overlayLayer().appendChild(scrim);
    const focusables = () => [...box.querySelectorAll('button, input, select, textarea, [tabindex]:not([tabindex="-1"])')].filter((x) => !x.disabled);
    box.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            const f = focusables();
            if (!f.length) return;
            const first = f[0], last = f[f.length - 1];
            if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
            else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
        }
        if (onKey) onKey(e);
    });
    focusSoon(() => (initialFocus || focusables()[0] || box).focus(), box);
    return {
        scrim,
        close() {
            scrim.classList.add('closing');
            setTimeout(() => scrim.remove(), 90);
            if (restore && restore.focus && restore.isConnected) restore.focus({ preventScroll: true });
        },
    };
}

export function confirmDialog({ title, message, detail, confirm, cancel, danger = false, third = null }) {
    return new Promise((resolve) => {
        const id = uniqueId('dlg');
        const ok = h('button', { class: ['btn', danger ? 'danger' : 'primary'], type: 'button' }, confirm || t('dialog.ok'));
        const no = h('button', { class: 'btn', type: 'button' }, cancel || t('dialog.cancel'));
        const buttons = [];
        let thirdBtn = null;
        if (third) { thirdBtn = h('button', { class: 'btn ghost', type: 'button' }, third); buttons.push(thirdBtn); }
        buttons.push(no, ok);
        const body = h('div', { class: 'dialog-body' }, h('p', {}, message), detail ? h('p', { class: 'muted small' }, detail) : null);
        let done = false;
        const finish = (v) => { if (done) return; done = true; m.close(); resolve(v); };
        const m = modal({
            title, body, buttons, labelId: id, initialFocus: danger ? no : ok,
            onKey: (e) => {
                if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(false); }
            },
        });
        ok.addEventListener('click', () => finish(true));
        no.addEventListener('click', () => finish(false));
        if (thirdBtn) thirdBtn.addEventListener('click', () => finish('third'));
        m.scrim.addEventListener('mousedown', (e) => { if (e.target === m.scrim) finish(false); });
    });
}

export function promptDialog({ title, message, value = '', placeholder = '', confirm, validate, select = null, allowEmpty = false }) {
    return new Promise((resolve) => {
        const id = uniqueId('dlg');
        const input = h('input', { class: 'input', type: 'text', value, placeholder, spellcheck: 'false', autocomplete: 'off', 'aria-label': title });
        const err = h('div', { class: 'dialog-error', role: 'alert' });
        const ok = h('button', { class: 'btn primary', type: 'button' }, confirm || t('dialog.ok'));
        const no = h('button', { class: 'btn', type: 'button' }, t('dialog.cancel'));
        const body = h('div', { class: 'dialog-body' }, message ? h('p', {}, message) : null, input, err);
        let done = false;
        const check = () => {
            const msg = validate ? validate(input.value) : '';
            err.textContent = msg || '';
            ok.disabled = Boolean(msg) || (!allowEmpty && !input.value.trim());
            return !msg && (allowEmpty || input.value.trim());
        };
        const finish = (v) => { if (done) return; done = true; m.close(); resolve(v); };
        const m = modal({
            title, body, buttons: [no, ok], labelId: id, initialFocus: input,
            onKey: (e) => {
                if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(null); }
                if (e.key === 'Enter' && document.activeElement === input) { e.preventDefault(); if (check()) finish(input.value.trim()); }
            },
        });
        input.addEventListener('input', check);
        ok.addEventListener('click', () => { if (check()) finish(input.value.trim()); });
        no.addEventListener('click', () => finish(null));
        m.scrim.addEventListener('mousedown', (e) => { if (e.target === m.scrim) finish(null); });
        // the dialog is in the document already: focus and select right away,
        // so text typed at once is neither lost nor selected over
        input.focus();
        if (select) input.setSelectionRange(select[0], select[1]); else input.select();
        check();
    });
}

/* a dialog with free content; resolves when one of the buttons is chosen */
export function customDialog({ title, body, buttons, wide = false }) {
    return new Promise((resolve) => {
        const id = uniqueId('dlg');
        let done = false;
        const finish = (v) => { if (done) return; done = true; m.close(); resolve(v); };
        const els = buttons.map((b) => {
            const el = h('button', { class: ['btn', b.primary && 'primary', b.danger && 'danger'], type: 'button' }, b.label);
            el.addEventListener('click', () => finish(b.value));
            return el;
        });
        const m = modal({
            title, body, buttons: els, labelId: id, wide,
            onKey: (e) => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(null); } },
        });
        m.scrim.addEventListener('mousedown', (e) => { if (e.target === m.scrim) finish(null); });
    });
}

/* ── toasts and notifications ──────────────────────────────────────── */
export const notifications = new Emitter();
notifications.list = [];
notifications.unread = 0;

let toastBox = null;
const KIND_ICON = { info: 'info', success: 'checkCircle', warning: 'warning', error: 'error' };

export function toast(message, { kind = 'info', actions = [], timeout, detail = '' } = {}) {
    if (!toastBox) {
        toastBox = h('div', { class: 'toasts', role: 'region', 'aria-live': 'polite', 'aria-label': t('notifications.title') });
        document.body.appendChild(toastBox);
        onLocale(() => toastBox.setAttribute('aria-label', t('notifications.title')));
    }
    const entry = { message, kind, detail, time: Date.now() };
    notifications.list.unshift(entry);
    notifications.list.length = Math.min(notifications.list.length, 50);
    notifications.unread++;
    notifications.emit('change');
    const el = h('div', { class: ['toast', `toast-${kind}`], role: kind === 'error' ? 'alert' : 'status' },
        icon(KIND_ICON[kind] || 'info', { className: 'toast-icon' }),
        h('div', { class: 'toast-text' }, h('div', { class: 'toast-msg' }, message), detail ? h('div', { class: 'toast-detail' }, detail) : null),
    );
    const btns = h('div', { class: 'toast-actions' });
    for (const a of actions) {
        const b = h('button', { class: 'btn small', type: 'button' }, a.label);
        b.addEventListener('click', () => { dismiss(); a.run(); });
        btns.appendChild(b);
    }
    const close = h('button', { class: 'icon-btn small toast-close', type: 'button', 'aria-label': t('dialog.close') }, icon('close'));
    close.addEventListener('click', () => dismiss());
    if (actions.length) el.appendChild(btns);
    el.appendChild(close);
    toastBox.appendChild(el);
    while (toastBox.children.length > 4) toastBox.firstChild.remove();
    const ms = timeout !== undefined ? timeout : kind === 'error' ? 9000 : actions.length ? 10000 : 4200;
    let timer = ms ? setTimeout(() => dismiss(), ms) : 0;
    el.addEventListener('mouseenter', () => clearTimeout(timer));
    el.addEventListener('mouseleave', () => { if (ms) timer = setTimeout(() => dismiss(), 2500); });
    function dismiss() {
        clearTimeout(timer);
        if (!el.isConnected) return;
        el.classList.add('leaving');
        setTimeout(() => el.remove(), 160);
    }
    return { dismiss, el };
}

/* ── hover tips for [data-tip] elements ────────────────────────────── */
export function installTips() {
    let tip = null, timer = 0, target = null;
    const hide = () => { clearTimeout(timer); timer = 0; if (tip) { tip.remove(); tip = null; } target = null; };
    document.addEventListener('mouseover', (e) => {
        const el = e.target.closest && e.target.closest('[data-tip]');
        if (el === target) return;
        hide();
        if (!el) return;
        target = el;
        timer = setTimeout(() => {
            if (!el.isConnected) return;
            const text = el.dataset.tip;
            if (!text) return;
            tip = h('div', { class: 'tip', role: 'tooltip' }, text);
            if (el.dataset.tipKeys) tip.appendChild(h('span', { class: 'tip-keys' }, formatKeys(el.dataset.tipKeys)));
            overlayLayer().appendChild(tip);
            const r = el.getBoundingClientRect();
            const side = el.dataset.tipSide || 'below';
            if (side === 'right') {
                tip.style.left = `${Math.round(r.right + 8)}px`;
                tip.style.top = `${Math.round(r.top + r.height / 2 - tip.offsetHeight / 2)}px`;
            } else placeNear(tip, r, { gap: 6, prefer: side === 'above' ? 'above' : 'below' });
        }, 480);
    });
    document.addEventListener('mousedown', hide, true);
    document.addEventListener('keydown', hide, true);
    window.addEventListener('blur', hide);
}

/* set a hover tip (and the accessible name) that follows the UI language */
export function tip(el, key, keys, side) {
    const apply = () => {
        const text = t(key);
        el.dataset.tip = text;
        el.setAttribute('aria-label', text);
    };
    apply();
    if (keys) el.dataset.tipKeys = keys;
    if (side) el.dataset.tipSide = side;
    el.dataset.tipKey = key;
    return el;
}

export function refreshTips(root = document) {
    for (const el of root.querySelectorAll('[data-tip-key]')) {
        const text = t(el.dataset.tipKey);
        el.dataset.tip = text;
        el.setAttribute('aria-label', text);
    }
}
