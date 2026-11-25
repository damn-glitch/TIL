// Commands and key bindings. Every action of the UI is a command (shown in
// the command palette with its key binding); bindings are configurable
// (Settings → Keyboard shortcuts, kept in localStorage).
//
// Key strings: "Mod+Shift+P" (Mod = Cmd on macOS, Ctrl elsewhere), "F5",
// "Shift+F11", "Ctrl+`", "Alt+Shift+F". Letter keys are matched by their
// physical position (event.code), so shortcuts work with Russian and Kazakh
// keyboard layouts too.
import { Emitter } from './events.js';
import { isMac } from './dom.js';
import { t } from './i18n.js';
import { load, save } from './store.js';

const MAC = isMac();

export class Commands extends Emitter {
    constructor() {
        super();
        this.list = new Map();
        this.overrides = load('keymap', {});
        this.recent = load('recent-commands', []);
        this.byKey = new Map();
    }

    register(cmd) {
        this.list.set(cmd.id, Object.assign({ palette: true }, cmd));
        this.rebuild();
    }

    get(id) { return this.list.get(id); }

    keysFor(id) {
        if (Object.prototype.hasOwnProperty.call(this.overrides, id)) return this.overrides[id] ? [this.overrides[id]] : [];
        const c = this.list.get(id);
        if (!c || !c.keys) return [];
        return Array.isArray(c.keys) ? c.keys : [c.keys];
    }

    setBinding(id, keys) {
        const c = this.list.get(id);
        const defaults = c && c.keys ? (Array.isArray(c.keys) ? c.keys : [c.keys]) : [];
        if (keys === null) delete this.overrides[id];
        else if (defaults.length === 1 && defaults[0] === keys) delete this.overrides[id];
        else this.overrides[id] = keys;
        save('keymap', this.overrides);
        this.rebuild();
        this.emit('keymap');
    }

    resetBindings() {
        this.overrides = {};
        save('keymap', this.overrides);
        this.rebuild();
        this.emit('keymap');
    }

    rebuild() {
        this.byKey.clear();
        for (const id of this.list.keys()) {
            for (const k of this.keysFor(id)) {
                const norm = normalizeKeys(k);
                if (!this.byKey.has(norm)) this.byKey.set(norm, []);
                this.byKey.get(norm).push(id);
            }
        }
    }

    title(id) {
        const c = this.list.get(id);
        if (!c) return id;
        const title = typeof c.title === 'function' ? c.title() : t(c.title);
        const cat = c.category ? t(c.category) : '';
        return cat && cat !== title ? `${cat}: ${title}` : title;
    }

    enabled(id) {
        const c = this.list.get(id);
        return Boolean(c) && (!c.when || c.when());
    }

    async run(id, ...args) {
        const c = this.list.get(id);
        if (!c) return undefined;
        if (c.when && !c.when()) return undefined;
        this.recent = [id, ...this.recent.filter((x) => x !== id)].slice(0, 12);
        save('recent-commands', this.recent);
        try {
            return await c.run(...args);
        } catch (e) {
            this.emit('error', e, id);
            return undefined;
        }
    }

    /* the command bound to a keyboard event (that is enabled now), or null */
    match(e) {
        const key = eventKeys(e);
        if (!key) return null;
        const ids = this.byKey.get(key);
        if (!ids) return null;
        for (const id of ids) if (this.enabled(id)) return id;
        return null;
    }
}

const CODE_KEYS = {
    Backquote: '`', Minus: '-', Equal: '=', BracketLeft: '[', BracketRight: ']', Backslash: '\\',
    Semicolon: ';', Quote: "'", Comma: ',', Period: '.', Slash: '/', Space: 'Space',
};

/* keyboard event -> normalized key string ("Mod+Shift+P"), or '' for a bare modifier */
export function eventKeys(e) {
    let key;
    if (/^Key[A-Z]$/.test(e.code)) key = e.code.slice(3);
    else if (/^Digit\d$/.test(e.code)) key = e.code.slice(5);
    else if (CODE_KEYS[e.code]) key = CODE_KEYS[e.code];
    else if (/^(Control|Shift|Alt|Meta|OS)/.test(e.key)) return '';
    else key = e.key === ' ' ? 'Space' : (e.key && e.key.length === 1 ? e.key.toUpperCase() : e.key);
    if (!key) return '';
    const mods = [];
    const mod = MAC ? e.metaKey : e.ctrlKey;
    const ctrl = MAC ? e.ctrlKey : false;
    const meta = MAC ? false : e.metaKey;
    if (mod) mods.push('Mod');
    if (ctrl) mods.push('Ctrl');
    if (meta) mods.push('Meta');
    if (e.altKey) mods.push('Alt');
    if (e.shiftKey) mods.push('Shift');
    return [...mods, key].join('+');
}

/* "ctrl+shift+p" / "Mod+Shift+P" -> canonical "Mod+Shift+P" */
export function normalizeKeys(text) {
    const parts = String(text).split('+').map((p) => p.trim()).filter(Boolean);
    if (!parts.length) return '';
    let key = parts.pop();
    const set = new Set(parts.map((p) => p.toLowerCase()));
    const mods = [];
    const hasCtrl = set.has('ctrl') || set.has('control');
    const hasMeta = set.has('meta') || set.has('cmd') || set.has('command');
    if (set.has('mod') || (!MAC && hasCtrl) || (MAC && hasMeta)) mods.push('Mod');
    if (MAC && hasCtrl) mods.push('Ctrl');
    if (!MAC && hasMeta) mods.push('Meta');
    if (set.has('alt') || set.has('option')) mods.push('Alt');
    if (set.has('shift')) mods.push('Shift');
    if (key.length === 1) key = key.toUpperCase();
    else if (/^f\d+$/i.test(key)) key = key.toUpperCase();
    else if (key.toLowerCase() === 'esc') key = 'Escape';
    else if (key.toLowerCase() === 'space') key = 'Space';
    return [...mods, key].join('+');
}

/* "Mod+Shift+P" -> "Ctrl+Shift+P" or "⌘⇧P" for display */
export function formatKeys(text) {
    if (!text) return '';
    const parts = normalizeKeys(text).split('+');
    const key = parts.pop();
    const names = { ArrowUp: '↑', ArrowDown: '↓', ArrowLeft: '←', ArrowRight: '→', Escape: 'Esc', Space: 'Space', Enter: '↵' };
    const k = names[key] || key;
    if (MAC) {
        const sym = { Mod: '⌘', Ctrl: '⌃', Alt: '⌥', Shift: '⇧', Meta: '⌘' };
        return parts.map((p) => sym[p] || p).join('') + k;
    }
    return [...parts.map((p) => (p === 'Mod' ? 'Ctrl' : p)), k].join('+');
}
