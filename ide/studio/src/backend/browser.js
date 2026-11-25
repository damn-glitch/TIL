// The browser backend of the web face (website/studio/): the same interface
// as the desktop backend (api.js), implemented in the page. Files live in
// IndexedDB (a starter project with the examples), programs compile with
// tilc.tbc and run on tilvm.wasm (../playground/), diagnostics come from
// tilc's compile errors, the language server runs in the page (weblsp.js).
// Debugger, prover, fit, packages, bytecode and native builds need the
// desktop server: they throw NotAvailable (the UI shows a card).
import EXAMPLES from 'virtual:examples';
import { Emitter } from '../core/events.js';
import { NotAvailable, ApiError } from './api.js';
import { WebLanguageServer } from './weblsp.js';
import { WebRuntime } from './webrt.js';
import { translateSource } from './translate.js';
import { TABLES } from './lang_scan.js';
import { t, getLocale } from '../core/i18n.js';
import { basename, dirname, normalize, join } from '../core/paths.js';

const DB_NAME = 'til-studio-web';
const ROOT_URI = 'file:///til-web';

function openDb() {
    return new Promise((resolve, reject) => {
        if (typeof indexedDB === 'undefined') { reject(new Error('IndexedDB is not available')); return; }
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains('files')) db.createObjectStore('files', { keyPath: 'path' });
            if (!db.objectStoreNames.contains('meta')) db.createObjectStore('meta', { keyPath: 'key' });
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error || new Error('cannot open IndexedDB'));
    });
}

function tx(db, store, mode, fn) {
    return new Promise((resolve, reject) => {
        const tr = db.transaction(store, mode);
        const st = tr.objectStore(store);
        let result;
        Promise.resolve(fn(st)).then((r) => { result = r; });
        tr.oncomplete = () => resolve(result);
        tr.onerror = () => reject(tr.error);
        tr.onabort = () => reject(tr.error || new Error('transaction aborted'));
    });
}

function reqP(req) {
    return new Promise((resolve, reject) => { req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error); });
}

/* a memory store when IndexedDB is not usable (private windows of some browsers) */
class MemoryStore {
    constructor() { this.files = new Map(); this.meta = new Map(); }
    async all() { return [...this.files.values()]; }
    async get(path) { return this.files.get(path) || null; }
    async put(entry) { this.files.set(entry.path, entry); }
    async delete(path) { this.files.delete(path); }
    async getMeta(key) { return this.meta.get(key); }
    async putMeta(key, value) { this.meta.set(key, value); }
}

class DbStore {
    constructor(db) { this.db = db; }
    all() { return tx(this.db, 'files', 'readonly', (st) => reqP(st.getAll())); }
    get(path) { return tx(this.db, 'files', 'readonly', (st) => reqP(st.get(path))).then((x) => x || null); }
    put(entry) { return tx(this.db, 'files', 'readwrite', (st) => { st.put(entry); }); }
    delete(path) { return tx(this.db, 'files', 'readwrite', (st) => { st.delete(path); }); }
    getMeta(key) { return tx(this.db, 'meta', 'readonly', (st) => reqP(st.get(key))).then((x) => (x ? x.value : undefined)); }
    putMeta(key, value) { return tx(this.db, 'meta', 'readwrite', (st) => { st.put({ key, value }); }); }
}

const STARTER_MAIN = `# Welcome to TIL Studio in your browser!
# Programs compile with the self-hosted TIL compiler and run on the TIL VM,
# both as WebAssembly: nothing is sent to a server.
#
#   Ctrl+F5 (or the Run button)   run this file
#   Ctrl+Space                    completion
#   examples/                     the examples (graphics, input, Kazakh, Russian ...)

greet(name: str) -> str
    return f"Hello, {name}!"

main()
    print(greet("TIL"))
    let squares = [1, 2, 3, 4, 5].map(|x| x * x)
    print(f"squares: {squares}")
    for lang in ["English", "Қазақша", "Русский"]
        print(f"  write TIL in {lang}")
`;

export class BrowserBackend extends Emitter {
    constructor({ runtimeBase } = {}) {
        super();
        this.kind = 'web';
        this.rootUri = ROOT_URI;
        // the runtime next to the page (website/playground/). Never taken from the
        // page's URL: a link could make the page run another site's script.
        this.runtime = new WebRuntime(new URL(runtimeBase || '../playground/', location.href).href);
        this.cache = new Map();       // path -> {path, type, text, mtime}
        this.runs = new Map();
        this.runSeq = 0;
    }

    async connect() {
        try { this.store = new DbStore(await openDb()); } catch (e) { this.store = new MemoryStore(); this.memoryOnly = true; }
        let entries = [];
        try { entries = await this.store.all(); } catch (e) { this.store = new MemoryStore(); this.memoryOnly = true; entries = []; }
        const seeded = await this.store.getMeta('seeded').catch(() => undefined);
        if (!seeded && !entries.length) {
            entries = this.starter();
            for (const e of entries) await this.store.put(e);
            await this.store.putMeta('seeded', 1).catch(() => {});
        }
        for (const e of entries) this.cache.set(e.path, e);
        this.server = new WebLanguageServer({
            emit: (m) => this.emit('lsp', m),
            diagnose: (uri, text) => this.diagnose(uri, text),
            files: () => this.tilFiles(),
            t,
            name: 'til-web-lsp',
        });
        this.runtime.init().then(() => this.emit('connection', true), (e) => { this.runtimeError = e; this.emit('connection', false); });
        this.info = {
            version: TABLES.version, root: '', rootName: 'til-web-project', platform: 'web', lang: getLocale(),
            capabilities: { native: false, cvm: false, z3: false, debugger: false, pkg: false, web: true },
            devices: [],
            examples: EXAMPLES.map((e) => ({ name: e.id, path: `examples/${e.file}` })),
            tables: TABLES,
            projectKey: 'web:default',
            raw: {},
        };
        return this.info;
    }

    starter() {
        const now = Date.now();
        const out = [{ path: 'main.til', type: 'file', text: STARTER_MAIN, mtime: now }, { path: 'examples', type: 'dir', mtime: now }];
        for (const e of EXAMPLES) out.push({ path: `examples/${e.file}`, type: 'file', text: e.source, mtime: now });
        return out;
    }

    tilFiles() {
        const m = new Map();
        for (const e of this.cache.values()) if (e.type === 'file' && /\.til$/.test(e.path)) m.set(this.uri(e.path), e.text);
        return m;
    }

    uri(path) { return `${ROOT_URI}/${path.split('/').map(encodeURIComponent).join('/')}`; }

    pathOfUri(uri) {
        if (!String(uri).startsWith(ROOT_URI + '/')) return null;
        try { return decodeURIComponent(uri.slice(ROOT_URI.length + 1)); } catch (e) { return null; }
    }

    /* ── files ── */
    async tree(path = '') {
        const dir = normalize(path);
        const children = new Map();
        for (const e of this.cache.values()) {
            if (e.path === dir) continue;
            const parent = dirname(e.path);
            if (parent === dir) children.set(e.path, { name: basename(e.path), path: e.path, type: e.type, size: e.type === 'file' ? (e.text || '').length : undefined, mtime: e.mtime });
            else if (dir === '' ? true : parent.startsWith(dir + '/')) {
                // an implicit directory between dir and the file
                const rest = dir ? e.path.slice(dir.length + 1) : e.path;
                const first = rest.split('/')[0];
                const p = dir ? `${dir}/${first}` : first;
                if (!children.has(p) && rest.includes('/')) children.set(p, { name: first, path: p, type: 'dir', mtime: e.mtime });
            }
        }
        return [...children.values()];
    }

    async readFile(path) {
        const e = this.cache.get(normalize(path));
        if (!e || e.type !== 'file') throw new ApiError(404, `no such file: ${path}`);
        return { path: e.path, text: e.text || '', mtime: e.mtime, size: (e.text || '').length };
    }

    async writeFile(path, text, mtime) {
        const p = normalize(path);
        const cur = this.cache.get(p);
        if (cur && mtime !== undefined && mtime !== null && cur.mtime !== mtime) throw new ApiError(409, 'the file changed on disk');
        const entry = { path: p, type: 'file', text, mtime: Math.max(Date.now(), cur ? cur.mtime + 1 : 0) };
        this.cache.set(p, entry);
        await this.store.put(entry);
        if (this.runtime.fs) this.runtime.writeFile('/' + p, text);
        return { mtime: entry.mtime };
    }

    async createFile(path, type = 'file', text = '') {
        const p = normalize(path);
        if (!p) throw new ApiError(400, 'bad path');
        if (this.cache.has(p)) throw new ApiError(409, `${p} exists`);
        const entry = { path: p, type, mtime: Date.now() };
        if (type === 'file') entry.text = text || '';
        this.cache.set(p, entry);
        await this.store.put(entry);
        if (type === 'file' && this.runtime.fs) this.runtime.writeFile('/' + p, entry.text);
        return { path: p };
    }

    async renamePath(path, to) {
        const from = normalize(path), dest = normalize(to);
        if (this.cache.has(dest)) throw new ApiError(409, `${dest} exists`);
        const moved = [...this.cache.values()].filter((e) => e.path === from || e.path.startsWith(from + '/'));
        if (!moved.length) throw new ApiError(404, `no such file: ${from}`);
        for (const e of moved) {
            await this.store.delete(e.path);
            this.cache.delete(e.path);
            const n = Object.assign({}, e, { path: dest + e.path.slice(from.length), mtime: Date.now() });
            this.cache.set(n.path, n);
            await this.store.put(n);
        }
        return { path: dest };
    }

    async deletePath(path) {
        const p = normalize(path);
        for (const e of [...this.cache.values()]) {
            if (e.path === p || e.path.startsWith(p + '/')) { this.cache.delete(e.path); await this.store.delete(e.path); }
        }
        return {};
    }

    async search({ query, regex, case: cs, glob }) {
        let re;
        try { re = new RegExp(regex ? query : query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), cs ? 'g' : 'gi'); } catch (e) { throw new ApiError(400, e.message); }
        const g = glob ? globRe(glob) : null;
        const matches = [];
        for (const e of [...this.cache.values()].sort((a, b) => a.path.localeCompare(b.path))) {
            if (e.type !== 'file' || (g && !g.test(e.path) && !g.test(basename(e.path)))) continue;
            const lines = (e.text || '').split('\n');
            for (let i = 0; i < lines.length; i++) {
                re.lastIndex = 0;
                const m = re.exec(lines[i]);
                if (m) matches.push({ path: e.path, line: i + 1, col: m.index + 1, text: lines[i].slice(0, 400) });
                if (matches.length >= 2000) return { matches, truncated: true };
            }
        }
        return { matches, truncated: false };
    }

    /* ── language server (in the page) ── */
    lsp(message) { return this.server.handle(message); }

    async syncFs() {
        await this.runtime.init();
        for (const e of this.cache.values()) if (e.type === 'file') this.runtime.writeFile('/' + e.path, e.text || '');
    }

    async diagnose(uri, text) {
        const path = this.pathOfUri(uri) || `.studio/${uri.replace(/[^\w.-]+/g, '_')}.til`;
        try { await this.syncFs(); } catch (e) { return []; }
        this.runtime.writeFile('/' + path, text);
        let res;
        try { res = await this.runtime.compile('/' + path); } catch (e) { return []; }
        if (res.ok) return [];
        return compileDiagnostics(res, path, text);
    }

    /* ── runs ── */
    async run(opts) {
        if (opts.backend && opts.backend !== 'vm') throw new NotAvailable(opts.backend === 'native' ? 'native' : 'cvm');
        const id = `w${++this.runSeq}`;
        const ev = (kind, extra) => this.emit('run', Object.assign({ runId: id, kind }, extra));
        const t0 = performance.now();
        // the same events as the studio server: no text for people in status/exit
        const exit = (code, reason) => ev('exit', { code, reason, seconds: Math.round(performance.now() - t0) / 1000 });
        const state = { id, handle: null, stopped: false };
        this.runs.set(id, state);
        setTimeout(async () => {
            try {
                await this.syncFs();
            } catch (e) {
                ev('stderr', { text: `${t('web.runtimeFailed')}: ${e.message}\n` });
                exit(1, 'failed');
                return;
            }
            let path = opts.path ? normalize(opts.path) : '.studio/untitled.til';
            if (!opts.path) this.runtime.writeFile('/' + path, opts.source || '');
            ev('status', { state: 'compiling', backend: 'vm' });
            const res = await this.runtime.compile('/' + path);
            if (state.stopped) { exit(130, 'stopped'); return; }
            if (!res.ok) {
                ev('stderr', { text: res.output || 'compile error\n' });
                exit(1, 'failed');
                return;
            }
            ev('status', { state: 'running', backend: 'vm' });
            const written = new Set();
            const fs = this.runtime.fs;
            const origWrite = fs.write;
            fs.write = (p, bytes) => { if (!p.startsWith('/.studio/')) written.add(p); return origWrite.call(fs, p, bytes); };
            state.handle = this.runtime.run(res.tbc, {
                onOutput: (text, stream) => ev(stream === 2 ? 'stderr' : 'stdout', { text }),
                onScreen: (w, h, rgba) => this.emit('screen-rgba', { runId: id, w, h, rgba }),
                onInput: () => ev('input', {}),
                onExit: async (code, text) => {
                    fs.write = origWrite;
                    this.runs.delete(id);
                    const changed = [];
                    for (const p of written) {
                        const rel = p.replace(/^\/+/, '');
                        const bytes = fs.read(p);
                        if (!bytes) continue;
                        try { await this.writeFile(rel, new TextDecoder().decode(bytes)); changed.push(rel); } catch (e) { /* */ }
                    }
                    if (changed.length) this.emit('fs', { paths: changed });
                    if (text) ev('stderr', { text: text.endsWith('\n') ? text : `${text}\n` });
                    if (state.stopped || code === -1) exit(130, 'stopped');
                    else exit(code, text ? 'failed' : 'exited');
                },
            }, { stdin: opts.stdin, keys: opts.keys, args: opts.args });
        }, 0);
        return id;
    }

    async runStdin(id, text) { const r = this.runs.get(id); if (r && r.handle) r.handle.pushLine(text); return {}; }

    async runEof(id) { const r = this.runs.get(id); if (r && r.handle) r.handle.eof(); return {}; }

    async runKey(id, code) { const r = this.runs.get(id); if (r && r.handle) r.handle.pushKey(code); return {}; }

    async runStop(id) {
        const r = this.runs.get(id);
        if (!r) return {};
        r.stopped = true;
        if (r.handle) r.handle.stop();
        return {};
    }

    async runSnapshot() { throw new NotAvailable('snapshot'); }

    /* ── tools: translation works in the page; the rest needs the desktop ── */
    async translate({ path, source, to }) {
        let text = source;
        if (text === undefined && path) text = (await this.readFile(path)).text;
        return { text: translateSource(text || '', to) };
    }

    async dapStart() { throw new NotAvailable('debugger'); }
    async dap() { throw new NotAvailable('debugger'); }
    async prove() { throw new NotAvailable('prover'); }
    async fit() { throw new NotAvailable('fit'); }
    async dis() { throw new NotAvailable('bytecode'); }
    async build() { throw new NotAvailable('build'); }
    async pkg() { throw new NotAvailable('packages'); }

    async request(method, path, body) {
        if (/\/stdin$/.test(path) && body && body.eof) { const id = path.split('/')[2]; return this.runEof(decodeURIComponent(id)); }
        throw new NotAvailable('server');
    }

    close() {}
}

function globRe(glob) {
    const parts = glob.split(',').map((g) => g.trim()).filter(Boolean).map((g) => {
        let s = '';
        for (let i = 0; i < g.length; i++) {
            const c = g[i];
            if (c === '*' && g[i + 1] === '*') { s += '.*'; i++; if (g[i + 1] === '/') i++; }
            else if (c === '*') s += '[^/]*';
            else if (c === '?') s += '[^/]';
            else s += c.replace(/[.+^${}()|[\]\\]/g, '\\$&');
        }
        return s;
    });
    return new RegExp(`^(?:${parts.join('|')})$`);
}

/* tilc's output -> LSP diagnostics for the document at `path` */
export function compileDiagnostics(res, path, text) {
    const lines = text.split('\n');
    const out = res.output || '';
    const re = /tilc:\s*(\/?[^:\n]+):(\d+):\s*(error|unsupported):\s*([^\n]*)/;
    const m = re.exec(out);
    if (!m) {
        const msg = out.trim().split('\n')[0] || 'compile error';
        return [{ range: { start: { line: 0, character: 0 }, end: { line: 0, character: (lines[0] || '').length || 1 } }, severity: 1, source: 'tilc', message: res.internal ? `${msg} (compiler failure)` : msg }];
    }
    const file = m[1].replace(/^\/+/, '');
    let line = Math.max(0, parseInt(m[2], 10) - 1);
    let message = m[4].trim();
    const kind = m[3];
    if (file !== path) {
        // an error in an imported module: on the import line of this document
        const imp = lines.findIndex((l) => new RegExp(`\\b${basename(file).replace(/\.til$/, '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`).test(l) && /^\s*(import|from|импорт|из|бастап)\b/u.test(l));
        line = imp >= 0 ? imp : 0;
        message = `${file}:${m[2]}: ${message}`;
    }
    const lt = lines[line] || '';
    let start = lt.length - lt.trimStart().length, end = lt.length;
    const q = /'([^']+)'/.exec(message);
    if (q && file === path) {
        const at = lt.search(new RegExp(`(?<![\\p{L}\\p{N}_])${q[1].replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\p{L}\\p{N}_])`, 'u'));
        if (at >= 0) { start = at; end = at + q[1].length; }
    }
    if (end <= start) end = start + 1;
    const diag = { range: { start: { line, character: start }, end: { line, character: end } }, severity: kind === 'unsupported' ? 2 : 1, source: 'tilc', code: kind === 'unsupported' ? 'unsupported' : 'E-VM', message };
    if (/unknown function '([^']+)'/.test(message)) diag.data = { hint: 'did you mean' };
    return [diag];
}

export { join };
