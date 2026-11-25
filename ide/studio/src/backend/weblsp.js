// A small language server that runs inside the page (the web face) or in
// the test mock: LSP messages in, LSP messages out. Diagnostics come from a
// pluggable diagnose(uri, text) (tilc in WebAssembly for the web face);
// completion, hover, signature help, symbols, definition, references,
// rename, formatting and quick fixes come from the static tables plus a
// symbol scan of the documents (lang_scan.js).
import { scan, occurrences, wordAt, functionAt, levenshtein, codeOnly, KEYWORD_TYPE, ALIAS_OF, TABLES } from './lang_scan.js';
import { BUILTIN_PARAMS, builtinSignature, METHOD_RECEIVERS, METHOD_PARAMS } from './builtins.js';

const KIND = { Text: 1, Method: 2, Function: 3, Field: 5, Variable: 6, Class: 7, Interface: 8, Module: 9, Property: 10, Enum: 13, Keyword: 14, Snippet: 15, Constant: 21, Struct: 22, EnumMember: 20, TypeParameter: 25 };
const SYM = { function: 12, method: 6, struct: 23, enum: 10, trait: 11, impl: 5, const: 14, global: 13, field: 8, variant: 22 };

const SNIPPETS = [
    ['main', 'main()\n    ${0}', 'main()'],
    ['fn', '${1:name}(${2:x}: ${3:int}) -> ${4:int}\n    ${0:return x}', 'function'],
    ['struct', 'struct ${1:Name}\n    ${2:field}: ${3:int}', 'struct'],
    ['enum', 'enum ${1:Name}\n    ${2:First}\n    ${3:Second}', 'enum'],
    ['impl', 'impl ${1:Name}\n    ${2:method}(self) -> ${3:int}\n        ${0:return 0}', 'impl'],
    ['if', 'if ${1:condition}\n    ${0}', 'if'],
    ['ifelse', 'if ${1:condition}\n    ${2}\nelse\n    ${0}', 'if / else'],
    ['for', 'for ${1:i} in ${2:0..10}\n    ${0}', 'for loop'],
    ['while', 'while ${1:condition}\n    ${0}', 'while loop'],
    ['match', 'match ${1:value}\n    ${2:pattern} => ${0}', 'match'],
    ['retry', 'retry(${1:3})\n    ${2:body}\nelse\n    ${0:fallback}', 'retry / else'],
    ['timeout', 'timeout(${1:100})\n    ${2:body}\nelse\n    ${0:fallback}', 'timeout / else'],
    ['handle', 'handle\n    ${1:body}\nwith\n    ${2:Effect}(${3:x}) =>\n        resume(${0})', 'handle / with'],
    ['requires', '#[requires: ${1:x >= 0}]\n#[ensures: ${0:result >= 0}]', 'contract'],
];

function pos(line, character) { return { line, character }; }
function range(l1, c1, l2, c2) { return { start: pos(l1, c1), end: pos(l2, c2) }; }

export class WebLanguageServer {
    /**
     * opts: {emit(msg), diagnose(uri, text) -> Promise<Diagnostic[]>, files() -> Map(uri -> text),
     *        t(key, params) (UI strings for docs), desktop (extra features of the mock: inlay hints, Debug/Prove lenses)}
     */
    constructor(opts) {
        this.opts = opts;
        this.docs = new Map();
        this.timers = new Map();
        this.cache = new Map();
    }

    t(key, params) { return this.opts.t ? this.opts.t(key, params) : key; }

    info(uri) {
        const d = this.docs.get(uri);
        if (!d) return null;
        const c = this.cache.get(uri);
        if (c && c.version === d.version) return c.info;
        const info = scan(d.text);
        info.langs = languagesOf(d.text);
        this.cache.set(uri, { version: d.version, info });
        return info;
    }

    /* every scanned document: open ones and the project's other files */
    allInfos() {
        const out = [];
        for (const uri of this.docs.keys()) out.push([uri, this.info(uri)]);
        if (this.opts.files) {
            for (const [uri, text] of this.opts.files()) {
                if (this.docs.has(uri)) continue;
                const key = `file:${uri}`;
                const c = this.cache.get(key);
                if (c && c.text === text) { out.push([uri, c.info]); continue; }
                const info = scan(text);
                this.cache.set(key, { text, info });
                out.push([uri, info]);
            }
        }
        return out;
    }

    /* one message in -> the messages to send now */
    async handle(msg) {
        if (!msg || typeof msg !== 'object') return [];
        const { id, method, params } = msg;
        if (method === undefined) return [];          // a response from the client
        if (method === 'textDocument/didOpen' && this.opts.diagnosticsInReply && params && params.textDocument) {
            // like til lsp: the diagnostics of an opened document come with the reply
            const td = params.textDocument;
            this.docs.set(td.uri, { text: td.text || '', version: td.version || 0 });
            let diags = [];
            try { diags = this.opts.diagnose ? await this.opts.diagnose(td.uri, td.text || '', this.info(td.uri)) : []; } catch (e) { diags = []; }
            return [{ jsonrpc: '2.0', method: 'textDocument/publishDiagnostics', params: { uri: td.uri, version: td.version || 0, diagnostics: diags } }];
        }
        try {
            const result = await this.dispatch(method, params || {});
            if (id === undefined || id === null) return [];
            if (result === NOT_FOUND) return [{ jsonrpc: '2.0', id, error: { code: -32601, message: `method not found: ${method}` } }];
            return [{ jsonrpc: '2.0', id, result: result === undefined ? null : result }];
        } catch (e) {
            if (id === undefined || id === null) return [];
            return [{ jsonrpc: '2.0', id, error: { code: e.code || -32603, message: e.message || String(e) } }];
        }
    }

    async dispatch(method, p) {
        switch (method) {
        case 'initialize': return this.initialize(p);
        case 'initialized': case '$/cancelRequest': case 'exit': case 'workspace/didChangeConfiguration': return undefined;
        case 'shutdown': return null;
        case 'textDocument/didOpen': this.open(p.textDocument); return undefined;
        case 'textDocument/didChange': this.change(p); return undefined;
        case 'textDocument/didSave': return undefined;
        case 'textDocument/didClose': this.docs.delete(p.textDocument.uri); this.opts.emit({ jsonrpc: '2.0', method: 'textDocument/publishDiagnostics', params: { uri: p.textDocument.uri, diagnostics: [] } }); return undefined;
        case 'textDocument/completion': return this.completion(p);
        case 'completionItem/resolve': return p;
        case 'textDocument/hover': return this.hover(p);
        case 'textDocument/signatureHelp': return this.signatureHelp(p);
        case 'textDocument/definition': return this.definition(p);
        case 'textDocument/references': return this.references(p);
        case 'textDocument/documentHighlight': return this.highlights(p);
        case 'textDocument/prepareRename': return this.prepareRename(p);
        case 'textDocument/rename': return this.rename(p);
        case 'textDocument/documentSymbol': return this.documentSymbols(p);
        case 'workspace/symbol': return this.workspaceSymbols(p);
        case 'textDocument/formatting': return this.formatting(p);
        case 'textDocument/codeAction': return this.codeActions(p);
        case 'textDocument/codeLens': return this.codeLenses(p);
        case 'textDocument/inlayHint': return this.opts.desktop ? this.inlayHints(p) : [];
        case 'textDocument/foldingRange': return this.folding(p);
        case 'workspace/executeCommand': return null;
        default: return NOT_FOUND;
        }
    }

    initialize() {
        const caps = {
            positionEncoding: 'utf-16',
            textDocumentSync: { openClose: true, change: 1, save: { includeText: false } },
            completionProvider: { triggerCharacters: ['.', '['], resolveProvider: false },
            hoverProvider: true,
            signatureHelpProvider: { triggerCharacters: ['(', ','] },
            definitionProvider: true,
            referencesProvider: true,
            documentHighlightProvider: true,
            renameProvider: { prepareProvider: true },
            documentSymbolProvider: true,
            workspaceSymbolProvider: true,
            documentFormattingProvider: true,
            codeActionProvider: { codeActionKinds: ['quickfix', 'refactor.rewrite'] },
            codeLensProvider: { resolveProvider: false },
            foldingRangeProvider: true,
            executeCommandProvider: { commands: ['til.run', 'til.debug', 'til.prove', 'til.translate'] },
        };
        if (this.opts.desktop) caps.inlayHintProvider = true;
        return { capabilities: caps, serverInfo: { name: this.opts.name || 'til-web-lsp', version: TABLES.version } };
    }

    open(td) {
        this.docs.set(td.uri, { text: td.text || '', version: td.version || 0 });
        this.schedule(td.uri, 30);
    }

    change(p) {
        const d = this.docs.get(p.textDocument.uri);
        if (!d) return;
        const ch = p.contentChanges || [];
        if (ch.length) d.text = ch[ch.length - 1].text;
        d.version = p.textDocument.version || d.version + 1;
        this.schedule(p.textDocument.uri, 200);
    }

    schedule(uri, ms) {
        clearTimeout(this.timers.get(uri));
        this.timers.set(uri, setTimeout(() => this.publish(uri), ms));
    }

    async publish(uri) {
        const d = this.docs.get(uri);
        if (!d || !this.opts.diagnose) return;
        const version = d.version;
        let diags = [];
        try { diags = await this.opts.diagnose(uri, d.text, this.info(uri)); } catch (e) { diags = []; }
        const cur = this.docs.get(uri);
        if (!cur || cur.version !== version) return;
        this.opts.emit({ jsonrpc: '2.0', method: 'textDocument/publishDiagnostics', params: { uri, version, diagnostics: diags } });
    }

    lines(uri) { const d = this.docs.get(uri); return d ? d.text.split('\n') : []; }

    /* ── completion ── */
    completion(p) {
        const uri = p.textDocument.uri;
        const info = this.info(uri);
        if (!info) return [];
        const lines = this.lines(uri);
        const { line, character } = p.position;
        const text = lines[line] || '';
        const before = text.slice(0, character);
        const items = [];
        const seen = new Set();
        const add = (it) => { if (seen.has(it.label)) return; seen.add(it.label); items.push(it); };
        // inside #[ ... ]: attribute names
        if (/#\[\s*[\p{L}_]*$/u.test(before)) {
            for (const a of TABLES.attributes) add({ label: a, kind: KIND.Property, detail: '#[' + a + (['pure', 'inline', 'noinline', 'hot', 'cold'].includes(a) ? ']' : ': …]'), documentation: { kind: 'markdown', value: this.t(`attr.${a}`) }, insertText: ['pure', 'inline', 'noinline', 'hot', 'cold'].includes(a) ? a : `${a}: ` });
            return { isIncomplete: false, items };
        }
        const dot = /([\p{L}_][\p{L}\p{N}_]*)\s*\.\s*[\p{L}_]*$/u.exec(before);
        if (dot) {
            const recv = dot[1];
            const enumSym = info.symbols.find((s) => s.kind === 'enum' && s.name === recv);
            if (enumSym) {
                for (const v of enumSym.children) add({ label: v.name, kind: KIND.EnumMember, detail: v.detail });
                return { isIncomplete: false, items };
            }
            const structSym = info.symbols.find((s) => (s.kind === 'struct' || s.kind === 'impl') && s.typeName === recv);
            for (const [, inf] of this.allInfos()) {
                for (const s of inf.symbols) {
                    if (s.kind === 'method' && (!structSym || s.parent === structSym.typeName)) add({ label: s.name, kind: KIND.Method, detail: s.detail, documentation: s.doc ? { kind: 'markdown', value: s.doc } : undefined, insertText: s.params.length > (s.params[0] && /^(self|өзін|ozin|сам)$/.test(s.params[0].name) ? 1 : 0) ? `${s.name}($1)` : `${s.name}()`, insertTextFormat: 2 });
                    if (s.kind === 'field') add({ label: s.name, kind: KIND.Field, detail: s.detail });
                }
            }
            for (const m of TABLES.methods) {
                const recvKinds = Object.entries(METHOD_RECEIVERS).filter(([, ms]) => ms.includes(m)).map(([k]) => k);
                const params = METHOD_PARAMS[m] || [];
                add({ label: m, kind: KIND.Method, detail: `.${m}(${params.join(', ')})${recvKinds.length ? '  · ' + recvKinds.join(', ') : ''}`, documentation: { kind: 'markdown', value: this.t(`method.${m}`) }, insertText: params.length ? `${m}($1)` : `${m}()`, insertTextFormat: 2 });
            }
            return { isIncomplete: false, items };
        }
        const langs = info.langs || ['en', 'kk'];
        const localLangs = langs.filter((l) => l !== 'en' && l !== 'kk');
        const fnHere = functionAt(info, line);
        if (fnHere) {
            for (const l of info.locals) {
                if (l.fn !== fnHere || l.line > line) continue;
                add({ label: l.name, kind: KIND.Variable, detail: l.kind === 'param' ? `${l.name}${l.type ? ': ' + l.type : ''}` : `${l.kind} ${l.name}`, sortText: '0' + l.name });
            }
        }
        for (const [u, inf] of this.allInfos()) {
            const own = u === uri;
            for (const s of inf.symbols) {
                if (s.kind === 'function') add({ label: s.name, kind: KIND.Function, detail: s.detail, documentation: s.doc ? { kind: 'markdown', value: s.doc } : undefined, insertText: s.params.length ? `${s.name}($1)` : `${s.name}()`, insertTextFormat: 2, sortText: (own ? '1' : '3') + s.name });
                else if (s.kind === 'struct') add({ label: s.name, kind: KIND.Struct, detail: `struct ${s.name}`, sortText: '2' + s.name });
                else if (s.kind === 'enum') add({ label: s.name, kind: KIND.Enum, detail: `enum ${s.name}`, sortText: '2' + s.name });
                else if (s.kind === 'trait') add({ label: s.name, kind: KIND.Interface, detail: `trait ${s.name}`, sortText: '2' + s.name });
                else if (s.kind === 'const' || s.kind === 'global') add({ label: s.name, kind: s.kind === 'const' ? KIND.Constant : KIND.Variable, detail: s.detail, sortText: '2' + s.name });
            }
        }
        for (const name of TABLES.natives) {
            add({ label: name, kind: KIND.Function, detail: builtinSignature(name), documentation: { kind: 'markdown', value: this.t(`builtin.${name}`) }, insertText: (BUILTIN_PARAMS[name] || []).length ? `${name}($1)` : `${name}()`, insertTextFormat: 2, sortText: '4' + name });
        }
        for (const l of langs) {
            for (const [w, canon] of Object.entries(TABLES.aliases[l] || {})) {
                if (canon === 'main') continue;
                add({ label: w, kind: KIND.Function, detail: `${builtinSignature(canon).replace(canon, w)}  = ${canon}`, documentation: { kind: 'markdown', value: this.t(`builtin.${canon}`) }, insertText: `${w}($1)`, insertTextFormat: 2, sortText: '4' + w });
            }
        }
        // keywords: the file's own languages first
        const kwLangs = [...localLangs, 'en', 'kk'].filter((l, i, a) => a.indexOf(l) === i);
        kwLangs.forEach((l, rank) => {
            for (const [w, ty] of Object.entries(TABLES.keywords[l] || {})) {
                if (ty === 'BOOL' && /^[TF]/.test(w)) continue;
                add({ label: w, kind: KIND.Keyword, detail: l === 'en' ? keywordDetail(ty) : `${keywordDetail(ty)} · ${l}`, sortText: `${5 + rank}${w}` });
            }
            for (const [w, ty] of Object.entries((TABLES.contextual || {})[l] || {})) add({ label: w, kind: KIND.Keyword, detail: `${keywordDetail(ty)} · ${l}`, sortText: `${5 + rank}${w}` });
        });
        for (const ty of TABLES.types) add({ label: ty, kind: KIND.TypeParameter, detail: 'type', sortText: '6' + ty });
        for (const [label, body, detail] of SNIPPETS) {
            items.push({ label, kind: KIND.Snippet, detail: `${detail} · snippet`, insertText: body, insertTextFormat: 2, sortText: '9' + label, filterText: label });
        }
        return { isIncomplete: false, items };
    }

    /* ── hover ── */
    hover(p) {
        const uri = p.textDocument.uri;
        const info = this.info(uri);
        if (!info) return null;
        const lines = this.lines(uri);
        const w = wordAt(lines, p.position.line, p.position.character);
        if (!w) return null;
        const r = range(p.position.line, w.start, p.position.line, w.end);
        const word = w.word;
        const md = (value) => ({ contents: { kind: 'markdown', value }, range: r });
        const kwTy = KEYWORD_TYPE.get(word);
        const lineText = lines[p.position.line] || '';
        const inCode = codeOnly(lineText).slice(w.start, w.end) === word;
        if (!inCode) return null;
        // user symbols first (locals, functions, types)
        const fnHere = functionAt(info, p.position.line);
        const local = info.locals.filter((l) => l.name === word && l.fn === fnHere && l.line <= p.position.line).pop();
        if (local && !kwTy) {
            const head = local.kind === 'param' ? this.t('lsp.hoverParam') : this.t('lsp.hoverLocal');
            const ty = local.type || guessType(local.value);
            return md('```til\n' + `${local.kind === 'param' ? '' : local.kind + ' '}${word}${ty ? ': ' + ty : ''}` + '\n```\n*' + head + '*');
        }
        for (const [, inf] of this.allInfos()) {
            const s = inf.symbols.find((x) => x.name === word && ['function', 'method', 'struct', 'enum', 'trait', 'const', 'global', 'field', 'variant'].includes(x.kind));
            if (!s) continue;
            let sig = s.detail;
            if (s.kind === 'struct') sig = `struct ${s.name}\n` + s.children.map((f) => `    ${f.detail}`).join('\n');
            if (s.kind === 'enum') sig = `enum ${s.name}\n` + s.children.map((f) => `    ${f.detail}`).join('\n');
            if (s.kind === 'method' && s.parent) sig = `${s.parent}.${s.detail}`;
            let value = '```til\n' + sig + '\n```';
            const contracts = (s.attrs || []).filter((a) => /#\[\s*(requires|ensures|pure|effects|cap|energy|level)/.test(a));
            if (contracts.length) value += '\n```til\n' + contracts.join('\n') + '\n```';
            if (s.doc) value += '\n\n' + s.doc;
            return md(value);
        }
        if (kwTy) {
            const spellings = [];
            for (const l of TABLES.languages) {
                const words = Object.entries(Object.assign({}, TABLES.keywords[l] || {}, (TABLES.contextual || {})[l] || {})).filter(([, ty]) => ty === kwTy).map(([x]) => x);
                if (words.length) spellings.push(`${words.map((x) => '`' + x + '`').join(' / ')} (${l})`);
            }
            return md(`**${this.t('lsp.keyword')}** \`${word}\` — ${keywordDetail(kwTy)}\n\n${spellings.join(' = ')}`);
        }
        const canon = ALIAS_OF.get(word) || word;
        if (TABLES.natives.includes(canon)) {
            const sig = builtinSignature(canon);
            let value = '```til\n' + (canon !== word ? sig.replace(canon, word) : sig) + '\n```\n' + this.t(`builtin.${canon}`);
            if (canon !== word) value += `\n\n${this.t('lsp.aliasOf', { name: canon })}`;
            return md(value);
        }
        if (TABLES.methods.includes(word) && /\.\s*$/.test(lineText.slice(0, w.start))) {
            return md('```til\n.' + word + '(' + (METHOD_PARAMS[word] || []).join(', ') + ')\n```\n' + this.t(`method.${word}`));
        }
        if (TABLES.types.includes(word)) return md(`**${this.t('lsp.type')}** \`${word}\``);
        return null;
    }

    /* ── signature help ── */
    signatureHelp(p) {
        const uri = p.textDocument.uri;
        const info = this.info(uri);
        if (!info) return null;
        const lines = this.lines(uri);
        const text = codeOnly(lines[p.position.line] || '').slice(0, p.position.character);
        let depth = 0, comma = 0, i = text.length - 1;
        for (; i >= 0; i--) {
            const c = text[i];
            if (c === ')' || c === ']' || c === '}') depth++;
            else if (c === '(' || c === '[' || c === '{') { if (depth === 0) break; depth--; }
            else if (c === ',' && depth === 0) comma++;
        }
        if (i < 0 || text[i] !== '(') return null;
        const m = /([\p{L}_][\p{L}\p{N}_]*)\s*$/u.exec(text.slice(0, i));
        if (!m) return null;
        const name = m[1];
        const canon = ALIAS_OF.get(name) || name;
        let label, params, doc;
        let user = null;
        for (const [, inf] of this.allInfos()) { user = inf.symbols.find((s) => (s.kind === 'function' || s.kind === 'method') && s.name === name); if (user) break; }
        if (user) {
            const ps = user.params.filter((x, k) => !(k === 0 && /^(self|өзін|ozin|сам)$/.test(x.name)));
            params = ps.map((x) => (x.type ? `${x.name}: ${x.type}` : x.name));
            label = `${name}(${params.join(', ')})${user.ret ? ' -> ' + user.ret : ''}`;
            doc = user.doc;
        } else if (TABLES.natives.includes(canon)) {
            params = BUILTIN_PARAMS[canon] || [];
            label = builtinSignature(canon).replace(canon, name);
            doc = this.t(`builtin.${canon}`);
        } else if (TABLES.methods.includes(name)) {
            params = METHOD_PARAMS[name] || [];
            label = `.${name}(${params.join(', ')})`;
            doc = this.t(`method.${name}`);
        } else return null;
        const active = params.length ? Math.min(comma, params.length - 1) : 0;
        return {
            signatures: [{ label, documentation: doc ? { kind: 'markdown', value: doc } : undefined, parameters: params.map((x) => ({ label: x })) }],
            activeSignature: 0,
            activeParameter: params.some((x) => x.startsWith('...')) && comma >= params.length ? params.length - 1 : active,
        };
    }

    /* ── navigation ── */
    resolveSymbol(uri, position) {
        const info = this.info(uri);
        if (!info) return null;
        const lines = this.lines(uri);
        const w = wordAt(lines, position.line, position.character);
        if (!w || KEYWORD_TYPE.has(w.word)) return null;
        const fnHere = functionAt(info, position.line);
        const local = info.locals.filter((l) => l.name === w.word && l.fn === fnHere).sort((a, b) => a.line - b.line)[0];
        if (local) return { w, scope: 'local', fn: fnHere, def: { uri, line: local.line, col: local.col } };
        for (const [u, inf] of [[uri, info], ...this.allInfos().filter(([x]) => x !== uri)]) {
            const s = inf.symbols.find((x) => x.name === w.word && x.kind !== 'impl');
            if (s) return { w, scope: 'global', def: { uri: u, line: s.line, col: s.col }, sym: s };
        }
        return { w, scope: 'unknown' };
    }

    definition(p) {
        const r = this.resolveSymbol(p.textDocument.uri, p.position);
        if (!r || !r.def) return null;
        return [{ uri: r.def.uri, range: range(r.def.line, r.def.col, r.def.line, r.def.col + r.w.word.length) }];
    }

    references(p) {
        const r = this.resolveSymbol(p.textDocument.uri, p.position);
        if (!r || r.scope === 'unknown' && !r.w) return [];
        const out = [];
        const word = r.w.word;
        if (r.scope === 'local') {
            const lines = this.lines(p.textDocument.uri);
            for (const o of occurrences(lines, word, r.fn ? r.fn.line : 0, r.fn ? r.fn.endLine : lines.length - 1)) out.push({ uri: p.textDocument.uri, range: range(o.line, o.col, o.line, o.col + word.length) });
            return out;
        }
        const seenUris = new Set();
        for (const [u, inf] of this.allInfos()) {
            seenUris.add(u);
            for (const o of occurrences(inf.lines, word)) out.push({ uri: u, range: range(o.line, o.col, o.line, o.col + word.length) });
        }
        if (p.context && p.context.includeDeclaration === false && r.def) return out.filter((x) => !(x.uri === r.def.uri && x.range.start.line === r.def.line));
        return out;
    }

    highlights(p) {
        const r = this.resolveSymbol(p.textDocument.uri, p.position);
        if (!r) return [];
        const lines = this.lines(p.textDocument.uri);
        const word = r.w.word;
        const from = r.scope === 'local' && r.fn ? r.fn.line : 0;
        const to = r.scope === 'local' && r.fn ? r.fn.endLine : lines.length - 1;
        return occurrences(lines, word, from, to).map((o) => ({ range: range(o.line, o.col, o.line, o.col + word.length), kind: 1 }));
    }

    prepareRename(p) {
        const lines = this.lines(p.textDocument.uri);
        const w = wordAt(lines, p.position.line, p.position.character);
        if (!w) return null;
        if (KEYWORD_TYPE.has(w.word)) throw rpcError(this.t('lsp.renameKeyword', { name: w.word }));
        const canon = ALIAS_OF.get(w.word) || w.word;
        const r = this.resolveSymbol(p.textDocument.uri, p.position);
        if ((!r || r.scope === 'unknown') && (TABLES.natives.includes(canon) || TABLES.methods.includes(canon))) throw rpcError(this.t('lsp.renameBuiltin', { name: w.word }));
        if (!r || r.scope === 'unknown') throw rpcError(this.t('lsp.renameUnknown', { name: w.word }));
        return { range: range(p.position.line, w.start, p.position.line, w.end), placeholder: w.word };
    }

    rename(p) {
        this.prepareRename(p);
        const newName = p.newName;
        if (KEYWORD_TYPE.has(newName) || TABLES.natives.includes(newName)) throw rpcError(this.t('lsp.renameClash', { name: newName }));
        const refs = this.references(Object.assign({}, p, { context: { includeDeclaration: true } }));
        const changes = {};
        for (const r of refs) (changes[r.uri] = changes[r.uri] || []).push({ range: r.range, newText: newName });
        return { changes };
    }

    documentSymbols(p) {
        const info = this.info(p.textDocument.uri);
        if (!info) return [];
        const conv = (s) => {
            const end = s.endLine !== undefined ? s.endLine : s.line;
            const lines = info.lines;
            const o = {
                name: s.name, kind: SYM[s.kind] || 13, detail: s.kind === 'function' || s.kind === 'method' ? s.detail.replace(s.name, '').trim() : s.kind === 'field' ? s.type || '' : '',
                range: range(s.line, 0, end, (lines[end] || '').length), selectionRange: range(s.line, Math.max(0, s.col), s.line, Math.max(0, s.col) + s.name.length),
            };
            if (s.children && s.children.length) o.children = s.children.map(conv);
            return o;
        };
        return info.top.map(conv);
    }

    workspaceSymbols(p) {
        const q = String(p.query || '').toLowerCase();
        const out = [];
        for (const [uri, inf] of this.allInfos()) {
            for (const s of inf.symbols) {
                if (!['function', 'method', 'struct', 'enum', 'trait', 'const'].includes(s.kind)) continue;
                if (q && !s.name.toLowerCase().includes(q)) continue;
                out.push({ name: s.name, kind: SYM[s.kind], location: { uri, range: range(s.line, s.col, s.line, s.col + s.name.length) }, containerName: s.parent || undefined });
            }
        }
        return out.slice(0, 500);
    }

    folding(p) {
        const lines = this.lines(p.textDocument.uri);
        const out = [];
        const ind = (s) => s.length - s.trimStart().length;
        for (let i = 0; i < lines.length; i++) {
            if (!lines[i].trim()) continue;
            let end = -1;
            for (let k = i + 1; k < lines.length; k++) {
                if (!lines[k].trim()) continue;
                if (ind(lines[k]) <= ind(lines[i])) break;
                end = k;
            }
            if (end > i) out.push({ startLine: i, endLine: end });
        }
        return out;
    }

    /* conservative formatting: trailing whitespace, tabs in indentation, blank runs, final newline */
    formatting(p) {
        const d = this.docs.get(p.textDocument.uri);
        if (!d) return [];
        const src = d.text;
        const lines = src.split('\n');
        const out = [];
        let blanks = 0;
        for (const l of lines) {
            let s = l.replace(/[ \t]+$/, '');
            s = s.replace(/^[ \t]+/, (m) => m.replace(/\t/g, '    '));
            if (!s) { blanks++; if (blanks > 2) continue; } else blanks = 0;
            out.push(s);
        }
        while (out.length && out[out.length - 1] === '') out.pop();
        const text = out.join('\n') + '\n';
        if (text === src) return [];
        return [{ range: range(0, 0, lines.length, 0), newText: text }];
    }

    codeActions(p) {
        const uri = p.textDocument.uri;
        const info = this.info(uri);
        if (!info) return [];
        const lines = this.lines(uri);
        const actions = [];
        for (const d of (p.context && p.context.diagnostics) || []) {
            const msg = String(d.message || '');
            const unknown = /unknown (?:function|name|variable)\s+'([^']+)'/.exec(msg) || /(?:неизвестн\S*|белгісіз)\s+\S*\s*'([^']+)'/i.exec(msg);
            if (unknown) {
                const bad = unknown[1];
                const cands = new Set([...TABLES.natives, ...info.symbols.filter((s) => s.kind === 'function' || s.kind === 'const' || s.kind === 'global').map((s) => s.name), ...info.locals.map((l) => l.name)]);
                let best = null, bestD = 3;
                for (const c of cands) { const dd = levenshtein(bad, c); if (dd < bestD && c !== bad) { best = c; bestD = dd; } }
                const hint = /did you mean '([^']+)'/.exec(msg) || /'([^']+)'\?/.exec(msg.split('\n').slice(1).join(' '));
                if (hint) best = hint[1];
                if (best) {
                    actions.push({ title: this.t('lsp.fixDidYouMean', { name: best }), kind: 'quickfix', diagnostics: [d], isPreferred: true, edit: { changes: { [uri]: [{ range: d.range, newText: best }] } } });
                }
            }
            const twice = /assign twice to immutable variable '([^']+)'/.exec(msg) || /immutable[^']*'([^']+)'/.exec(msg);
            if (twice) {
                const name = twice[1];
                const decl = info.locals.find((l) => l.name === name && l.kind === 'let') || info.symbols.find((s) => s.name === name && s.kind === 'global' && !s.mutable);
                if (decl) {
                    const text = lines[decl.line];
                    const m = /^(\s*)(\S+)/.exec(text);
                    const letWord = m[2];
                    const varWord = varFor(letWord);
                    actions.push({
                        title: this.t('lsp.fixLetToVar', { name, let: letWord, var: varWord }), kind: 'quickfix', diagnostics: [d], isPreferred: true,
                        edit: { changes: { [uri]: [{ range: range(decl.line, m[1].length, decl.line, m[1].length + letWord.length), newText: varWord }] } },
                    });
                }
            }
        }
        for (const lang of ['en', 'kk', 'kk-latn', 'ru']) {
            actions.push({ title: this.t('lsp.translateTo', { lang: this.t(`kwlang.${lang}`) }), kind: 'refactor.rewrite', command: { title: 'translate', command: 'til.translate', arguments: [{ uri, to: lang }] } });
        }
        return actions;
    }

    codeLenses(p) {
        const info = this.info(p.textDocument.uri);
        if (!info) return [];
        const out = [];
        const uri = p.textDocument.uri;
        const main = info.symbols.find((s) => s.kind === 'function' && (s.canonical === 'main' || s.name === 'main'));
        if (main) {
            out.push({ range: range(main.line, 0, main.line, 0), command: { title: this.t('lens.run'), command: 'til.run', arguments: [{ uri, function: 'main' }] } });
            if (this.opts.desktop) out.push({ range: range(main.line, 0, main.line, 0), command: { title: this.t('lens.debug'), command: 'til.debug', arguments: [{ uri, function: 'main' }] } });
        } else if (info.lines.some((l) => l.trim() && !l.trim().startsWith('#'))) {
            const first = info.lines.findIndex((l) => l.trim() && !l.trim().startsWith('#'));
            out.push({ range: range(first, 0, first, 0), command: { title: this.t('lens.run'), command: 'til.run', arguments: [{ uri }] } });
        }
        if (this.opts.desktop) {
            for (const s of info.symbols) {
                if (s.kind !== 'function' || !(s.attrs || []).some((a) => /requires|ensures/.test(a))) continue;
                out.push({ range: range(s.line, 0, s.line, 0), command: { title: this.t('lens.prove'), command: 'til.prove', arguments: [{ uri, function: s.name }] } });
                out.push({ range: range(s.line, 0, s.line, 0), command: { title: this.t('lens.debug'), command: 'til.debug', arguments: [{ uri, function: s.name }] } });
            }
        }
        return out;
    }

    inlayHints(p) {
        const info = this.info(p.textDocument.uri);
        if (!info) return [];
        const out = [];
        for (const l of info.locals) {
            if ((l.kind !== 'let' && l.kind !== 'var') || l.type) continue;
            const ty = guessType(l.value);
            if (!ty) continue;
            if (l.line < p.range.start.line || l.line > p.range.end.line) continue;
            out.push({ position: pos(l.line, l.col + l.name.length), label: `: ${ty}`, kind: 1, paddingLeft: false });
        }
        for (const s of info.symbols) {
            if (s.kind !== 'function' || s.ret || s.canonical === 'main' || s.line < p.range.start.line || s.line > p.range.end.line) continue;
            const closing = (info.lines[s.line] || '').lastIndexOf(')');
            if (closing >= 0) out.push({ position: pos(s.line, closing + 1), label: ' -> ' + (s.name.startsWith('is_') ? 'bool' : 'int'), kind: 1 });
        }
        return out;
    }
}

const NOT_FOUND = Symbol('not found');

function rpcError(message) { const e = new Error(message); e.code = -32803; return e; }

export function languagesOf(text) {
    const langs = [...TABLES.default_languages];
    const re = /^[ \t]*#\[\s*lang\s*:\s*([^\]]*)\]/gm;
    let m;
    while ((m = re.exec(text))) for (const c of m[1].split(',')) { const x = c.trim().toLowerCase(); if (TABLES.languages.includes(x) && !langs.includes(x)) langs.push(x); }
    return langs;
}

function keywordDetail(ty) {
    return String(ty).toLowerCase().replace(/_/g, ' ');
}

export function guessType(value) {
    const v = String(value || '').trim();
    if (!v) return '';
    if (/^-?\d+$/.test(v) || /^0[xbo][\da-f_]+$/i.test(v)) return 'int';
    if (/^-?\d+\.\d+([eE][+-]?\d+)?$/.test(v)) return 'float';
    if (/^f?["']/.test(v)) return 'str';
    if (/^(true|false|ақиқат|жалған|истина|ложь|aqiqat|zhalgan)$/.test(v)) return 'bool';
    if (/^\[\s*-?\d/.test(v)) return '[int]';
    if (/^\[\s*["']/.test(v)) return '[str]';
    if (/^\[/.test(v)) return '[?]';
    if (/^\{/.test(v)) return 'map';
    if (/^(len|ord|int|clock_ms|key_pressed|wait_key|get_pixel)\(/.test(v)) return 'int';
    if (/^(str|input|chr|format)\(/.test(v)) return 'str';
    if (/^(sqrt|sin|cos|float)\(/.test(v)) return 'float';
    const s = /^([\p{Lu}][\p{L}\p{N}_]*)\s*(\{|\.new\(|\()/u.exec(v);
    if (s) return s[1];
    return '';
}

function varFor(letWord) {
    const ty = KEYWORD_TYPE.get(letWord);
    if (ty !== 'LET') return 'var';
    for (const l of TABLES.languages) {
        const tbl = TABLES.keywords[l] || {};
        if (tbl[letWord] === 'LET') {
            const v = Object.entries(tbl).find(([, t2]) => t2 === 'VAR');
            if (v) return v[0];
        }
    }
    return 'var';
}
