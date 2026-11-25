// A tiny event emitter.
export class Emitter {
    constructor() { this._h = new Map(); }

    on(name, fn) {
        if (!this._h.has(name)) this._h.set(name, new Set());
        this._h.get(name).add(fn);
        return () => this.off(name, fn);
    }

    off(name, fn) {
        const s = this._h.get(name);
        if (s) s.delete(fn);
    }

    emit(name, ...args) {
        const s = this._h.get(name);
        if (!s) return;
        for (const fn of [...s]) {
            try { fn(...args); } catch (e) { console.error(`[studio] ${name} handler failed`, e); }
        }
    }
}
