// Persistence in localStorage (every access in try/catch: storage can be
// disabled or full). Settings are global; the rest is kept per project.
import { Emitter } from './events.js';

const PREFIX = 'til-studio:';

export function load(key, fallback) {
    try {
        const raw = localStorage.getItem(PREFIX + key);
        if (raw === null || raw === undefined) return fallback;
        return JSON.parse(raw);
    } catch (e) {
        return fallback;
    }
}

export function save(key, value) {
    try {
        localStorage.setItem(PREFIX + key, JSON.stringify(value));
        return true;
    } catch (e) {
        return false;
    }
}

export function remove(key) {
    try { localStorage.removeItem(PREFIX + key); } catch (e) { /* storage disabled */ }
}

export const DEFAULT_SETTINGS = {
    theme: 'dark',            // dark | light | system
    fontSize: 14,
    uiLang: '',               // '' = from the browser
    kwLang: 'en',             // keyword language for new files: en | kk | kk-latn | ru
    errorLens: true,
    formatOnSave: false,
    autoSave: 'off',          // off | delay | focus
    inlayHints: true,
    codeLens: true,
    wordWrap: false,
    backend: 'vm',            // native | vm | cvm
    device: 'ti84ce',
    welcomeOnStart: true,
    calculatorBezel: false,
    screenScale: 0,           // 0 = fit
    fuseBytecode: true,
};

class Settings extends Emitter {
    constructor() {
        super();
        this.values = Object.assign({}, DEFAULT_SETTINGS, load('settings', {}));
    }

    get(key) { return this.values[key]; }

    set(key, value) {
        if (this.values[key] === value) return;
        this.values[key] = value;
        save('settings', this.values);
        this.emit('change', key, value);
        this.emit(`change:${key}`, value);
    }
}

export const settings = new Settings();

/* per-project state: tabs, layout, breakpoints, watch expressions, run config */
export function projectKey(root) {
    return 'project:' + String(root || 'default');
}

export function loadProject(root) {
    return load(projectKey(root), {});
}

export function saveProject(root, state) {
    return save(projectKey(root), state);
}
