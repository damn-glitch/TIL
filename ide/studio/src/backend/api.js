// The desktop backend: the HTTP + SSE API of the studio server
// (src/tilstudio.py, docs/dev/TIL_STUDIO_DESIGN.md section 5).
//
// fetch wrappers for every endpoint plus ONE EventSource on /api/events,
// whose events (lsp, run, dap, fs, ping) are re-emitted by this object.
// Paths are POSIX paths relative to the project root (the TIL examples are
// the read-only "@examples/..."). Unknown fields in responses are ignored;
// missing ones get defaults. Every request says the UI language
// (X-TIL-Lang), so the server's error messages come in it.
import { Emitter } from '../core/events.js';
import { getLocale } from '../core/i18n.js';

export class ApiError extends Error {
    constructor(status, message, body) {
        super(message || `HTTP ${status}`);
        this.status = status;
        this.body = body;
    }
}

export class NotAvailable extends Error {
    constructor(feature) {
        super(`not available: ${feature}`);
        this.feature = feature;
        this.notAvailable = true;
    }
}

const TOKEN_KEY = 'til-studio-token';

/* the per-launch token: ?token= of the opened URL (then kept for this tab) */
export function takeToken() {
    let token = null;
    try {
        const url = new URL(location.href);
        token = url.searchParams.get('token');
        if (token) {
            try { sessionStorage.setItem(TOKEN_KEY, token); } catch (e) { /* storage disabled */ }
            url.searchParams.delete('token');
            history.replaceState(history.state, '', url.pathname + (url.search || '') + url.hash);
        } else {
            try { token = sessionStorage.getItem(TOKEN_KEY); } catch (e) { token = null; }
        }
    } catch (e) {
        token = null;
    }
    return token;
}

export class DesktopBackend extends Emitter {
    constructor({ token = null, base = '' } = {}) {
        super();
        this.kind = 'desktop';
        this.token = token;
        this.base = base;
        this.info = null;
        this.source = null;
        this.connected = false;
    }

    url(path, query) {
        const u = new URL(this.base + path.replace(/^\//, ''), location.href);
        if (query) for (const [k, v] of Object.entries(query)) if (v !== undefined && v !== null) u.searchParams.set(k, String(v));
        return u.href;
    }

    async request(method, path, body, query) {
        const headers = { Accept: 'application/json', 'X-TIL-Lang': getLocale() };
        if (this.token) headers['X-TIL-Token'] = this.token;
        const init = { method, headers, credentials: 'same-origin', cache: 'no-store' };
        if (body !== undefined) {
            headers['Content-Type'] = 'application/json';
            init.body = JSON.stringify(body);
        }
        let resp;
        try {
            resp = await fetch(this.url(path, query), init);
        } catch (e) {
            this.setConnected(false);
            throw new ApiError(0, e && e.message ? e.message : 'network error');
        }
        let data = null;
        const type = resp.headers.get('content-type') || '';
        if (type.includes('json')) {
            try { data = await resp.json(); } catch (e) { data = null; }
        } else {
            try { data = { text: await resp.text() }; } catch (e) { data = null; }
        }
        if (!resp.ok) {
            if (resp.status === 403) this.emit('auth', resp.status);
            const msg = data && typeof data.error === 'string' ? data.error : `HTTP ${resp.status}`;
            throw new ApiError(resp.status, msg, data);
        }
        return data || {};
    }

    /* GET /api/info; resolves null when this is not the desktop server */
    async probe() {
        try {
            const headers = { Accept: 'application/json', 'X-TIL-Lang': getLocale() };
            if (this.token) headers['X-TIL-Token'] = this.token;
            const resp = await fetch(this.url('api/info'), { headers, credentials: 'same-origin', cache: 'no-store' });
            const type = resp.headers.get('content-type') || '';
            if (resp.status === 403) return { forbidden: true };
            if (!resp.ok || !type.includes('json')) return null;
            const info = await resp.json();
            if (!info || typeof info !== 'object' || !('capabilities' in info || 'version' in info)) return null;
            return info;
        } catch (e) {
            return null;
        }
    }

    async connect() {
        const info = await this.probe();
        if (!info || info.forbidden) return info;
        this.info = normalizeInfo(info);
        this.openEvents();
        return this.info;
    }

    setConnected(v) {
        if (this.connected === v) return;
        this.connected = v;
        this.emit('connection', v);
    }

    openEvents() {
        if (typeof EventSource !== 'function') return;
        const es = new EventSource(this.url('api/events', this.token ? { token: this.token } : null), { withCredentials: true });
        this.source = es;
        es.onopen = () => this.setConnected(true);
        es.onerror = () => this.setConnected(false);
        const relay = (name) => es.addEventListener(name, (ev) => {
            let data = null;
            try { data = ev.data ? JSON.parse(ev.data) : null; } catch (e) { return; }
            this.setConnected(true);
            this.emit(name, data);
        });
        for (const name of ['lsp', 'run', 'dap', 'fs', 'ping']) relay(name);
    }

    close() {
        if (this.source) this.source.close();
        this.source = null;
    }

    /* ── files ── */
    async tree(path = '') {
        const r = await this.request('GET', 'api/tree', undefined, path ? { path } : null);
        return Array.isArray(r.entries) ? r.entries : [];
    }

    async readFile(path) {
        const r = await this.request('GET', 'api/file', undefined, { path });
        return { path: r.path || path, text: typeof r.text === 'string' ? r.text : '', mtime: r.mtime, size: r.size, readonly: Boolean(r.readonly) };
    }

    async writeFile(path, text, mtime) {
        const body = { path, text };
        if (mtime !== undefined && mtime !== null) body.mtime = mtime;
        return this.request('PUT', 'api/file', body);
    }

    createFile(path, type = 'file', text) {
        const body = { path, type };
        if (text !== undefined) body.text = text;
        return this.request('POST', 'api/file/create', body);
    }

    renamePath(path, to) { return this.request('POST', 'api/file/rename', { path, to }); }

    deletePath(path) { return this.request('POST', 'api/file/delete', { path }); }

    async search(opts) {
        const r = await this.request('POST', 'api/search', opts);
        return { matches: Array.isArray(r.matches) ? r.matches : [], truncated: Boolean(r.truncated) };
    }

    /* ── language server: one JSON-RPC message in, the messages to deliver now out ── */
    async lsp(message) {
        const r = await this.request('POST', 'api/lsp', message);
        return Array.isArray(r.messages) ? r.messages : [];
    }

    /* ── runs ── */
    async run(opts) {
        const r = await this.request('POST', 'api/run', opts);
        return r.runId;
    }

    /* one line of input (409: the run reads an input script) */
    runStdin(id, text) { return this.request('POST', `api/run/${encodeURIComponent(id)}/stdin`, { text }); }

    /* end of input (Ctrl+D) */
    runEof(id) { return this.request('POST', `api/run/${encodeURIComponent(id)}/stdin`, { text: '', eof: true }); }

    /* a key code (409: the run follows a key script) */
    runKey(id, code) { return this.request('POST', `api/run/${encodeURIComponent(id)}/key`, { code }); }

    runStop(id) { return this.request('POST', `api/run/${encodeURIComponent(id)}/stop`, {}); }

    /* Stop while the page unloads: a keepalive request outlives the page */
    runStopOnUnload(id) {
        const headers = { 'Content-Type': 'application/json', 'X-TIL-Lang': getLocale() };
        if (this.token) headers['X-TIL-Token'] = this.token;
        try {
            fetch(this.url(`api/run/${encodeURIComponent(id)}/stop`), { method: 'POST', headers, body: '{}', credentials: 'same-origin', keepalive: true }).catch(() => {});
        } catch (e) { /* the page is going away anyway */ }
    }

    /* {path, size, steps}; stop: end the run after saving */
    runSnapshot(id, stop = false) { return this.request('POST', `api/run/${encodeURIComponent(id)}/snapshot`, stop ? { stop: true } : {}); }

    /* ── debugger ── */
    async dapStart() {
        const r = await this.request('POST', 'api/dap/start', {});
        return r.sessionId;
    }

    async dap(id, message) {
        const r = await this.request('POST', `api/dap/${encodeURIComponent(id)}`, message);
        return Array.isArray(r.messages) ? r.messages : [];
    }

    /* ── tools ── */
    prove(opts) { return this.request('POST', 'api/tools/prove', opts); }

    fit(opts) { return this.request('POST', 'api/tools/fit', opts); }

    dis(opts) { return this.request('POST', 'api/tools/dis', opts); }

    translate(opts) { return this.request('POST', 'api/tools/translate', opts); }

    build(opts) { return this.request('POST', 'api/tools/build', opts); }

    pkg(argv) { return this.request('POST', 'api/pkg', { argv }); }
}

/* defaults for every field the UI reads (the server may send more or fewer) */
export function normalizeInfo(info) {
    // lsp, prover and fit are assumed when a server does not list them
    const caps = Object.assign({ native: false, cvm: false, z3: false, debugger: false, pkg: false, lsp: true, prover: true, fit: true }, info.capabilities || {});
    return {
        version: info.version || '',
        root: info.root || '',
        rootUri: typeof info.rootUri === 'string' ? info.rootUri : '',
        limits: info.limits && typeof info.limits === 'object' ? info.limits : {},
        rootName: info.rootName || basenameOf(info.root) || 'project',
        platform: info.platform || '',
        lang: info.lang || '',
        capabilities: caps,
        devices: Array.isArray(info.devices) ? info.devices.filter((d) => d && d.name) : [],
        examples: Array.isArray(info.examples) ? info.examples.filter((e) => e && (e.path || e.name)) : [],
        tables: info.tables && typeof info.tables === 'object' ? info.tables : null,
        raw: info,
    };
}

function basenameOf(p) {
    const s = String(p || '').replace(/[\\/]+$/, '');
    const i = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
    return i >= 0 ? s.slice(i + 1) : s;
}
