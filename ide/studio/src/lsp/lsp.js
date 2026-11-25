// A small JSON-RPC client for the TIL language server, reached through the
// backend (POST /api/lsp: one message in, the messages to deliver now out;
// later notifications such as debounced diagnostics arrive as SSE `lsp`
// events). Full document sync; changes are debounced and flushed before
// every request, so a request always sees the newest text.
import { Emitter } from '../core/events.js';

const CHANGE_DEBOUNCE_MS = 250;
const REQUEST_TIMEOUT_MS = 20000;

export class LspClient extends Emitter {
    constructor(backend, { rootUri, rootName, locale, version }) {
        super();
        this.backend = backend;
        this.rootUri = rootUri;
        this.rootName = rootName;
        this.locale = locale;
        this.clientVersion = version || '';
        this.nextId = 1;
        this.pending = new Map();
        this.capabilities = {};
        this.ready = false;
        this.docs = new Map();      // uri -> {version, text, timer, open}
        this.failed = false;
        this.restarting = null;
        backend.on('lsp', (msg) => this.dispatch(msg));
    }

    async start() {
        const params = {
            processId: null,
            clientInfo: { name: 'TIL Studio', version: this.clientVersion },
            locale: this.locale,
            rootUri: this.rootUri,
            workspaceFolders: [{ uri: this.rootUri, name: this.rootName }],
            initializationOptions: { client: 'til-studio' },
            capabilities: {
                general: { positionEncodings: ['utf-16'] },
                textDocument: {
                    synchronization: { didSave: true, dynamicRegistration: false },
                    completion: {
                        completionItem: { snippetSupport: true, documentationFormat: ['markdown', 'plaintext'], labelDetailsSupport: true },
                        contextSupport: true,
                    },
                    hover: { contentFormat: ['markdown', 'plaintext'] },
                    signatureHelp: { signatureInformation: { documentationFormat: ['markdown', 'plaintext'], parameterInformation: { labelOffsetSupport: true } } },
                    definition: { linkSupport: false },
                    references: {},
                    documentHighlight: {},
                    documentSymbol: { hierarchicalDocumentSymbolSupport: true },
                    formatting: {},
                    codeAction: { codeActionLiteralSupport: { codeActionKind: { valueSet: ['quickfix', 'refactor', 'refactor.rewrite', 'source'] } } },
                    codeLens: {},
                    rename: { prepareSupport: true },
                    foldingRange: { lineFoldingOnly: true },
                    publishDiagnostics: { relatedInformation: true, versionSupport: true },
                    inlayHint: {},
                    semanticTokens: { requests: { full: true }, tokenTypes: [], tokenModifiers: [], formats: ['relative'] },
                },
                workspace: { workspaceFolders: true, symbol: {}, executeCommand: {}, applyEdit: true, configuration: false },
                window: { showMessage: {} },
            },
        };
        try {
            const result = await this.request('initialize', params, { noFlush: true, allowBeforeReady: true });
            this.capabilities = (result && result.capabilities) || {};
            this.serverInfo = (result && result.serverInfo) || null;
        } catch (e) {
            // a server that refuses a second initialize (page reload): assume the
            // features of the spec; requests that fail later just give no result
            if (e && e.code === -32600 || (e && /initiali[sz]ed/i.test(e.message || ''))) {
                this.capabilities = DEFAULT_CAPABILITIES;
            } else {
                this.failed = true;
                this.emit('failed', e);
                throw e;
            }
        }
        this.ready = true;
        this.notify('initialized', {});
        for (const [uri, d] of this.docs) if (d.open) this.sendOpen(uri, d);
        this.emit('ready', this.capabilities);
        return this.capabilities;
    }

    /* initialize again and open the documents again */
    reinitialize() {
        if (this.restarting) return this.restarting;
        this.ready = false;
        this.restarting = this.start().catch(() => {}).finally(() => { this.restarting = null; });
        return this.restarting;
    }

    has(name) {
        const v = this.capabilities[name];
        return v !== undefined && v !== null && v !== false;
    }

    /* the trigger characters of completion/signature help */
    triggers(name) {
        const p = this.capabilities[name];
        if (!p || typeof p !== 'object') return [];
        return p.triggerCharacters || [];
    }

    async send(message) {
        let messages;
        try {
            messages = await this.backend.lsp(message);
        } catch (e) {
            if (message.id !== undefined && this.pending.has(message.id)) {
                const p = this.pending.get(message.id);
                this.pending.delete(message.id);
                clearTimeout(p.timer);
                p.reject(e);
            }
            return;
        }
        for (const m of messages) this.dispatch(m);
    }

    request(method, params, opts = {}) {
        if (!this.ready && !opts.allowBeforeReady) return Promise.reject(new Error('language server not ready'));
        if (!opts.noFlush) this.flushAll();
        const id = this.nextId++;
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                if (this.pending.has(id)) {
                    this.pending.delete(id);
                    reject(new Error(`${method}: no answer`));
                    this.notify('$/cancelRequest', { id });
                }
            }, opts.timeout || REQUEST_TIMEOUT_MS);
            this.pending.set(id, { resolve, reject, timer, method });
            this.send({ jsonrpc: '2.0', id, method, params });
        });
    }

    notify(method, params) {
        this.send({ jsonrpc: '2.0', method, params });
    }

    dispatch(msg) {
        if (!msg || typeof msg !== 'object') return;
        if (msg.id !== undefined && msg.id !== null && !msg.method) {
            const p = this.pending.get(msg.id);
            if (!p) return;
            this.pending.delete(msg.id);
            clearTimeout(p.timer);
            if (msg.error) {
                const err = new Error(msg.error.message || 'language server error');
                err.code = msg.error.code;
                err.data = msg.error.data;
                // ServerNotInitialized after we were ready: the server let this page's
                // language server go (the page was away); start a new one with our documents
                if (err.code === -32002 && this.ready && p.method !== 'initialize') this.reinitialize();
                p.reject(err);
            } else p.resolve(msg.result === undefined ? null : msg.result);
            return;
        }
        if (!msg.method) return;
        if (msg.id !== undefined && msg.id !== null) {         // a request from the server
            this.answerServer(msg);
            return;
        }
        switch (msg.method) {
        case 'textDocument/publishDiagnostics':
            if (msg.params) this.emit('diagnostics', msg.params);
            break;
        case 'window/showMessage':
            if (msg.params) this.emit('message', msg.params);
            break;
        case 'window/logMessage':
            if (msg.params) this.emit('log', msg.params);
            break;
        default:
            this.emit('notification', msg);
        }
    }

    async answerServer(msg) {
        let result = null, error = null;
        try {
            switch (msg.method) {
            case 'workspace/applyEdit': {
                const ok = await new Promise((resolve) => {
                    if (!this.emit || !this._h.has('applyEdit')) { resolve(false); return; }
                    this.emit('applyEdit', msg.params && msg.params.edit, resolve);
                });
                result = { applied: Boolean(ok) };
                break;
            }
            case 'workspace/configuration':
                result = ((msg.params && msg.params.items) || []).map(() => null);
                break;
            case 'client/registerCapability':
            case 'client/unregisterCapability':
            case 'window/workDoneProgress/create':
                result = null;
                break;
            case 'window/showMessageRequest':
                this.emit('message', msg.params || {});
                result = null;
                break;
            default:
                error = { code: -32601, message: `method not found: ${msg.method}` };
            }
        } catch (e) {
            error = { code: -32603, message: String(e && e.message ? e.message : e) };
        }
        const reply = { jsonrpc: '2.0', id: msg.id };
        if (error) reply.error = error; else reply.result = result;
        this.send(reply);
    }

    /* ── documents ── */
    open(uri, text) {
        const d = { version: 1, text, timer: 0, open: true, dirty: false };
        this.docs.set(uri, d);
        if (this.ready) this.sendOpen(uri, d);
    }

    sendOpen(uri, d) {
        if (typeof d.text === 'function') d.text = d.text();
        this.notify('textDocument/didOpen', { textDocument: { uri, languageId: 'til', version: d.version, text: d.text } });
    }

    isOpen(uri) { return this.docs.has(uri); }

    /* text: the new full text, or a function that returns it (evaluated when the change is sent) */
    change(uri, text) {
        const d = this.docs.get(uri);
        if (!d) return;
        if (typeof text === 'string' && d.text === text) return;
        d.text = text;
        d.dirty = true;
        clearTimeout(d.timer);
        d.timer = setTimeout(() => this.flush(uri), CHANGE_DEBOUNCE_MS);
    }

    flush(uri) {
        const d = this.docs.get(uri);
        if (!d || !d.dirty) return;
        clearTimeout(d.timer);
        d.dirty = false;
        d.version++;
        if (typeof d.text === 'function') d.text = d.text();
        if (this.ready) {
            this.notify('textDocument/didChange', { textDocument: { uri, version: d.version }, contentChanges: [{ text: d.text }] });
        }
    }

    flushAll() {
        for (const uri of this.docs.keys()) this.flush(uri);
    }

    version(uri) {
        const d = this.docs.get(uri);
        return d ? d.version : 0;
    }

    saved(uri, text) {
        this.flush(uri);
        if (this.ready && this.docs.has(uri)) this.notify('textDocument/didSave', { textDocument: { uri }, text });
    }

    close(uri) {
        const d = this.docs.get(uri);
        if (!d) return;
        clearTimeout(d.timer);
        this.docs.delete(uri);
        if (this.ready) this.notify('textDocument/didClose', { textDocument: { uri } });
    }

    rename(oldUri, newUri, text) {
        this.close(oldUri);
        this.open(newUri, text);
    }
}

/* what the spec promises (section 3), for servers that do not answer initialize */
const DEFAULT_CAPABILITIES = {
    textDocumentSync: 1,
    completionProvider: { triggerCharacters: ['.', '['], resolveProvider: true },
    hoverProvider: true,
    signatureHelpProvider: { triggerCharacters: ['(', ','] },
    definitionProvider: true,
    referencesProvider: true,
    documentHighlightProvider: true,
    renameProvider: { prepareProvider: true },
    documentSymbolProvider: true,
    workspaceSymbolProvider: true,
    foldingRangeProvider: true,
    documentFormattingProvider: true,
    inlayHintProvider: true,
    codeActionProvider: true,
    codeLensProvider: {},
};
