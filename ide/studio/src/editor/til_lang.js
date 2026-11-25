// The TIL language for CodeMirror 6: a stream parser over the generated
// keyword tables (every keyword language: English, Kazakh in Cyrillic and in
// Latin, Russian), f-strings with highlighted interpolations, attributes
// #[...], numbers and durations, comments; indentation-based folding and
// smart indentation (indent after block openers, dedent on else/with/...).
import { StreamLanguage, LanguageSupport, foldService, indentService, indentUnit } from '@codemirror/language';
import { tags, tagHighlighter, highlightTree } from '@lezer/highlight';
import TABLES from '../../tables.json';

export { TABLES };

/* ── word tables ─────────────────────────────────────────────────────── */
const CONTROL = new Set(['IF', 'ELSE', 'ELIF', 'FOR', 'WHILE', 'LOOP', 'RETURN', 'BREAK', 'CONTINUE', 'MATCH',
    'RETRY', 'FALLBACK', 'TIMEOUT', 'PERFORM', 'HANDLE', 'WITH', 'RESUME', 'ASSERT', 'INVARIANT', 'INTENT']);
const DEFINE = new Set(['FN', 'LET', 'VAR', 'CONST', 'STRUCT', 'ENUM', 'IMPL', 'TRAIT', 'TYPE', 'IMPORT', 'FROM', 'AS', 'CAP']);
const MODIFIER = new Set(['MUT', 'PUB']);
const OPWORD = new Set(['AND', 'OR', 'NOT', 'IN']);

/* block openers: a line whose first word has one of these token types opens an indented block */
const BLOCK_TYPES = new Set(['IF', 'ELIF', 'ELSE', 'WHILE', 'FOR', 'LOOP', 'MATCH', 'STRUCT', 'ENUM', 'IMPL', 'TRAIT',
    'FN', 'RETRY', 'TIMEOUT', 'INTENT', 'HANDLE', 'WITH']);
/* words that close a block and continue the statement above (dedented to it) */
const DEDENT_TYPES = new Set(['ELSE', 'ELIF', 'WITH']);

function buildWordTables(tables) {
    const byLang = {};
    for (const lang of tables.languages) {
        const m = new Map(Object.entries(tables.keywords[lang] || {}));
        byLang[lang] = m;
    }
    const contextual = {};
    for (const [lang, tbl] of Object.entries(tables.contextual || {})) contextual[lang] = new Map(Object.entries(tbl));
    const aliases = new Map();
    for (const tbl of Object.values(tables.aliases || {})) for (const [w, canon] of Object.entries(tbl)) aliases.set(w, canon);
    // every spelling of each token type (for indentation rules, independent of the file's languages)
    const allWords = new Map();
    for (const m of Object.values(byLang)) for (const [w, ty] of m) allWords.set(w, ty);
    for (const m of Object.values(contextual)) for (const [w, ty] of m) allWords.set(w, ty);
    return {
        byLang, contextual, aliases, allWords,
        natives: new Set(tables.natives),
        methods: new Set(tables.methods),
        types: new Set(tables.types),
        attributes: new Set(tables.attributes),
        entry: new Set(['main', ...[...aliases].filter(([, c]) => c === 'main').map(([w]) => w)]),
    };
}

export const WORDS = buildWordTables(TABLES);

/* languages of a source file: defaults + every #[lang: ...] pragma (like til.detect_languages) */
export function languagesOf(text) {
    const langs = [...TABLES.default_languages];
    const re = /^[ \t]*#\[\s*lang\s*:\s*([^\]]*)\]/gm;
    let m;
    while ((m = re.exec(text))) {
        for (const code of m[1].split(',')) {
            const c = code.trim().toLowerCase();
            if (TABLES.languages.includes(c) && !langs.includes(c)) langs.push(c);
        }
    }
    return langs;
}

/* the keyword table (word -> TokenType name) of a set of languages */
function keywordsFor(langs) {
    const m = new Map(WORDS.byLang.en);
    for (const l of langs) for (const [w, ty] of WORDS.byLang[l] || []) m.set(w, ty);
    return m;
}

const KW_CACHE = new Map();
function keywordMap(langs) {
    const key = langs.join(',');
    if (!KW_CACHE.has(key)) KW_CACHE.set(key, keywordsFor(langs));
    return KW_CACHE.get(key);
}

const IDENT_START = /[\p{L}_]/u;
const IDENT_REST = /[\p{L}\p{N}_]/u;
const UNITS = /^(ms|s|m|h)(?![\p{L}\p{N}_])/u;

function kwClass(ty) {
    if (ty === 'BOOL') return 'bool';
    if (ty === 'SELF') return 'self';
    if (CONTROL.has(ty)) return 'kw';
    if (DEFINE.has(ty)) return 'kwDef';
    if (MODIFIER.has(ty)) return 'kwMod';
    if (OPWORD.has(ty)) return 'kwOp';
    return 'kw';
}

function isDefinitionLine(rest, name, indent) {
    const m = /^([\p{L}_][\p{L}\p{N}_]*)\s*\(([^()]*)\)\s*(->[^#]*)?\s*(#.*)?$/u.exec(rest);
    if (!m || m[1] !== name) return false;
    if (m[3]) return true;
    const params = m[2].trim();
    if (/["']/.test(params)) return false;
    return /^[\p{L}_][\p{L}\p{N}_]*\s*:/u.test(params) || WORDS.entry.has(name) || /^(self|өзін|ozin|сам)\b/.test(params) ||
        (params === '' && indent === 0);
}

/* ── the stream parser ───────────────────────────────────────────────── */
// state.ctx: a stack of contexts within the line: 'f' (inside f-string text,
// with .q quote), 'e' (inside an f-string {expression}, .depth braces),
// 'a' (inside #[...], .depth brackets, .name read)
const tilParser = {
    name: 'til',
    startState() {
        return { langs: [...TABLES.default_languages], kw: keywordMap(TABLES.default_languages), ctx: [], lineStart: true, afterDot: false, defNext: false };
    },
    copyState(s) {
        return { langs: s.langs, kw: s.kw, ctx: s.ctx.map((c) => Object.assign({}, c)), lineStart: s.lineStart, afterDot: s.afterDot, defNext: s.defNext };
    },
    token(stream, state) {
        if (stream.sol()) {
            state.ctx = [];          // strings and attributes never continue on the next line
            state.lineStart = true;
            state.afterDot = false;
            state.defNext = false;
        }
        const top = state.ctx[state.ctx.length - 1];
        if (top && (top.k === 'f' || top.k === 's')) return stringText(stream, state, top, top.k === 'f');
        if (stream.eatSpace()) return null;
        const ch = stream.peek();
        const atLineStart = state.lineStart;
        state.lineStart = false;

        // attribute / comment
        if (ch === '#') {
            if (stream.match('#[')) {
                const prag = /^\s*lang\s*:\s*([^\]]*)\]/.exec(stream.string.slice(stream.pos));
                if (prag) {
                    let changed = false;
                    for (const code of prag[1].split(',')) {
                        const c = code.trim().toLowerCase();
                        if (TABLES.languages.includes(c) && !state.langs.includes(c)) { state.langs = [...state.langs, c]; changed = true; }
                    }
                    if (changed) state.kw = keywordMap(state.langs);
                }
                state.ctx.push({ k: 'a', depth: 1, name: false });
                return 'attrBracket';
            }
            stream.skipToEnd();
            return 'comment';
        }
        const inAttr = top && top.k === 'a';
        if (inAttr) {
            if (!top.name) {
                const m = stream.match(/^[\p{L}_][\p{L}\p{N}_]*/u);
                if (m) { top.name = true; top.lang = m[0] === 'lang'; return 'attrName'; }
            }
            if (ch === '[') { top.depth++; stream.next(); return 'bracket'; }
            if (ch === ']') {
                top.depth--;
                stream.next();
                if (top.depth === 0) { state.ctx.pop(); return 'attrBracket'; }
                return 'bracket';
            }
            if (top.lang && stream.match(/^[a-z-]+/)) return 'atom';
        }
        const inExpr = top && top.k === 'e';
        if (inExpr) {
            if (ch === '{') { top.depth++; stream.next(); return 'bracket'; }
            if (ch === '}') {
                top.depth--;
                stream.next();
                if (top.depth === 0) { state.ctx.pop(); return 'interp'; }
                return 'bracket';
            }
            if (ch === ':' && top.depth === 1 && stream.match(/^:[<>^]?\d*(?:\.\d+)?[fdxXobe%]?(?=\})/)) return 'fmt';
            if (ch === top.q) {      // quote inside an interpolation ends the string (malformed): recover
                state.ctx.pop();
                const f = state.ctx[state.ctx.length - 1];
                if (f && f.k === 'f') { state.ctx.pop(); stream.next(); return 'str'; }
            }
        }

        // strings
        if ((ch === 'f' || ch === 'F') && /^[fF]["']/.test(stream.string.slice(stream.pos, stream.pos + 2))) {
            stream.next();
            const q = stream.next();
            state.ctx.push({ k: 'f', q });
            return 'str';
        }
        if (ch === '"' || ch === "'") {
            stream.next();
            state.ctx.push({ k: 's', q: ch });
            return 'str';
        }

        // numbers and durations
        if (/\d/.test(ch)) {
            if (stream.match(/^0[xX][0-9a-fA-F_]+/) || stream.match(/^0[bB][01_]+/) || stream.match(/^0[oO][0-7_]+/)) return 'num';
            stream.match(/^\d[\d_]*(\.\d[\d_]*)?([eE][+-]?\d+)?/);
            if (stream.match(UNITS, false)) { stream.match(UNITS); return 'dur'; }
            return 'num';
        }

        // identifiers and keywords
        if (IDENT_START.test(ch)) {
            let word = '';
            while (!stream.eol() && IDENT_REST.test(stream.peek())) word += stream.next();
            const afterDot = state.afterDot;
            state.afterDot = false;
            const rest = stream.string.slice(stream.pos);
            const call = /^\s*\(/.test(rest);
            if (afterDot) {
                if (/^\p{Lu}/u.test(word)) return 'type';
                return call ? 'method' : 'prop';
            }
            const ty = state.kw.get(word);
            if (ty) {
                if (ty === 'FN') state.defNext = true;
                return kwClass(ty);
            }
            // contextual keywords: `with` alone on its line, resume(...), localized perform/handle
            for (const lang of state.langs) {
                const ctxTy = WORDS.contextual[lang] && WORDS.contextual[lang].get(word);
                if (!ctxTy) continue;
                if (ctxTy === 'WITH' && atLineStart && /^\s*(#.*)?$/.test(rest)) return 'kw';
                if (ctxTy === 'RESUME' && call) return 'kw';
                if (ctxTy === 'PERFORM' && /^\s+[\p{L}_]/u.test(rest)) return 'kw';
                if (ctxTy === 'HANDLE' && (/^\s*(#.*)?$/.test(rest) || /^\s+[\p{L}_\d"']/u.test(rest))) return 'kw';
            }
            if (state.defNext) { state.defNext = false; return 'fnDef'; }
            if (atLineStart && call && isDefinitionLine(stream.string.slice(stream.pos - word.length).trimEnd(), word, stream.indentation())) return 'fnDef';
            if (WORDS.types.has(word) && !call) return 'type';
            const canon = WORDS.aliases.get(word);
            if (WORDS.natives.has(word) || (canon && WORDS.natives.has(canon)) || (canon === 'main')) return call ? 'builtin' : (canon ? 'builtin' : 'var');
            if (WORDS.types.has(word)) return 'type';
            if (inAttr && word === 'result') return 'self';
            if (/^\p{Lu}[\p{Lu}\p{N}_]+$/u.test(word)) return 'const';
            if (call) return /^\p{Lu}/u.test(word) ? 'type' : 'fn';
            if (/^\p{Lu}/u.test(word)) return 'type';
            return 'var';
        }

        // operators and punctuation
        state.afterDot = false;
        if (stream.match('..=') || stream.match('..')) return 'op';
        if (ch === '.') { stream.next(); state.afterDot = true; return 'punc'; }
        if (stream.match(/^(->|=>|\|>|\*\*|==|!=|<=|>=|<<|>>|\+=|-=|\*=|\/=|%=)/)) return 'op';
        if (/[+\-*/%=<>!&|^~?@]/.test(ch)) { stream.next(); return 'op'; }
        if (/[()[\]{}]/.test(ch)) { stream.next(); return 'bracket'; }
        if (/[,:;]/.test(ch)) { stream.next(); return 'punc'; }
        stream.next();
        return 'invalid';
    },
    indent() { return null; },
    languageData: {
        commentTokens: { line: '#' },
        closeBrackets: { brackets: ['(', '[', '{', '"', "'"] },
        indentOnInput: null,      // set below (needs the dedent words)
        wordChars: '_',
    },
    tokenTable: {
        kw: tags.controlKeyword, kwDef: tags.definitionKeyword, kwMod: tags.modifier, kwOp: tags.operatorKeyword,
        bool: tags.bool, self: tags.self, num: tags.number, dur: tags.unit, str: tags.string, esc: tags.escape,
        interp: tags.special(tags.brace), fmt: tags.special(tags.string), comment: tags.lineComment,
        attrBracket: tags.annotation, attrName: tags.attributeName, fnDef: tags.function(tags.definition(tags.variableName)),
        fn: tags.function(tags.variableName), method: tags.function(tags.propertyName), builtin: tags.standard(tags.variableName),
        type: tags.typeName, const: tags.constant(tags.variableName), prop: tags.propertyName, var: tags.variableName,
        op: tags.operator, punc: tags.punctuation, bracket: tags.bracket, atom: tags.atom, invalid: tags.invalid,
    },
};

function readEscape(stream) {
    // at a backslash
    stream.next();
    if (stream.match(/^x[0-9a-fA-F]{2}/) || stream.match(/^u\{[0-9a-fA-F]+\}/)) return;
    if (!stream.eol()) stream.next();
}

function stringText(stream, state, top, fmt) {
    const start = stream.pos;
    while (!stream.eol()) {
        const c = stream.peek();
        if (c === '\\') {
            if (stream.pos > start) return 'str';
            readEscape(stream);
            return 'esc';
        }
        if (fmt && c === '{') {
            if (stream.pos > start) return 'str';
            if (stream.string[stream.pos + 1] === '{') { stream.pos += 2; return 'esc'; }
            stream.next();
            state.ctx.push({ k: 'e', depth: 1, q: top.q });
            return 'interp';
        }
        if (fmt && c === '}' && stream.string[stream.pos + 1] === '}') {
            if (stream.pos > start) return 'str';
            stream.pos += 2;
            return 'esc';
        }
        stream.next();
        if (c === top.q) { state.ctx.pop(); return 'str'; }
    }
    state.ctx.pop();
    return 'str';
}

/* ── indentation and folding ─────────────────────────────────────────── */
function stripLine(line) {
    // drop strings and the comment of a line (for the block-opener check)
    let out = '';
    for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '#' && line[i + 1] !== '[') break;
        if (c === '"' || c === "'") {
            const q = c;
            i++;
            while (i < line.length && line[i] !== q) { if (line[i] === '\\') i++; i++; }
            out += '""';
            continue;
        }
        out += c;
    }
    return out.trimEnd();
}

function firstWord(text) {
    const m = /^\s*([\p{L}_][\p{L}\p{N}_]*)/u.exec(text);
    return m ? m[1] : '';
}

/* does this line open an indented block? (TIL blocks have no ':' marker) */
export function opensBlock(lineText) {
    const t = stripLine(lineText).trim();
    if (!t || t.startsWith('#')) return false;
    const w = firstWord(t);
    const ty = WORDS.allWords.get(w);
    if (ty && BLOCK_TYPES.has(ty)) {
        if (ty === 'WITH' && t !== w) return false;
        return true;
    }
    if (/=>\s*$/.test(t)) return true;                                   // match arm with a block
    const words = t.match(/[\p{L}_][\p{L}\p{N}_]*/gu) || [];
    const last = words[words.length - 1];
    const lastTy = last ? WORDS.allWords.get(last) : null;
    if (lastTy === 'HANDLE' || lastTy === 'LOOP' || lastTy === 'ELSE') return /[\p{L}_]$/u.test(t);
    if (words.some((x) => WORDS.allWords.get(x) === 'MATCH') && !/[)\]}]$/.test(t.replace(/\bmatch\b.*$/, ''))) return true;
    if (/\b[\p{L}_]+\s*\([^()]*\)\s*$/u.test(t)) {
        const m = /([\p{L}_][\p{L}\p{N}_]*)\s*\([^()]*\)\s*$/u.exec(t);
        const nty = m && WORDS.allWords.get(m[1]);
        if (nty === 'RETRY' || nty === 'TIMEOUT') return true;
    }
    const def = /^([\p{L}_][\p{L}\p{N}_]*)\s*\((.*)\)\s*(->.*)?$/u.exec(t);
    if (def && !WORDS.allWords.has(def[1])) {
        return Boolean(def[3]) || def[2].includes(':') || WORDS.entry.has(def[1]) || /^(self|өзін|ozin|сам)\b/.test(def[2].trim()) ||
            (lineText.search(/\S/) === 0 && def[2].trim() === '');
    }
    return false;
}

const DEDENT_WORDS = [...WORDS.allWords].filter(([, ty]) => DEDENT_TYPES.has(ty)).map(([w]) => w);
const DEDENT_RE = new RegExp(`^\\s*(${DEDENT_WORDS.map((w) => w.replace(/[-]/g, '\\-')).join('|')})(?![\\p{L}\\p{N}_])`, 'u');

function indentOf(text) {
    let n = 0;
    for (const c of text) {
        if (c === ' ') n++;
        else if (c === '\t') n += 4;
        else break;
    }
    return n;
}

/* the indentation for a new line at `pos`, or for the line at pos when it starts with else/with */
function tilIndent(context, pos) {
    const doc = context.state.doc;
    const line = doc.lineAt(pos);
    const unit = context.unit || 4;
    const lineText = line.text;
    // the line we are indenting starts with a dedent word: align with its opener
    if (DEDENT_RE.test(lineText) && line.number > 1) {
        const word = firstWord(lineText);
        const ty = WORDS.allWords.get(word);
        let cur = Infinity;
        for (let n = line.number - 1; n >= 1 && n > line.number - 400; n--) {
            const t = doc.line(n).text;
            if (!t.trim() || /^\s*#(?!\[)/.test(t)) continue;
            const ind = indentOf(t);
            if (ind >= cur) continue;
            cur = ind;
            const w = firstWord(t);
            const wty = WORDS.allWords.get(w);
            if (ty === 'WITH' && (wty === 'HANDLE' || /(^|=\s*)\S*\s*$/.test(t) && /[\p{L}_]+\s*$/u.test(t) && WORDS.allWords.get(t.trim().split(/\s+/).pop()) === 'HANDLE')) return ind;
            if (ty !== 'WITH' && (wty === 'IF' || wty === 'ELIF' || wty === 'ELSE' || wty === 'FOR' || wty === 'WHILE' || /\b(retry|timeout|қайталау|повторить|таймаут|qaitalau)\s*\(/u.test(t))) return ind;
            if (ind === 0) return 0;
        }
        return Math.max(0, indentOf(lineText) - unit);
    }
    // a new line: look at the previous non-blank line
    let n = line.number - (pos > line.from || line.text.trim() === '' ? 0 : 1);
    if (context.simulatedBreak !== undefined && context.simulatedBreak !== null) n = doc.lineAt(context.simulatedBreak).number;
    for (; n >= 1; n--) {
        const l = doc.line(n);
        let text = l.text;
        if (context.simulatedBreak !== undefined && context.simulatedBreak !== null && l.from <= context.simulatedBreak && context.simulatedBreak <= l.to) {
            text = text.slice(0, context.simulatedBreak - l.from);
        }
        if (!text.trim()) continue;
        const ind = indentOf(text);
        if (opensBlock(text)) return ind + unit;
        const t = stripLine(text).trim();
        if (/^(return|break|continue|қайтару|тоқтату|жалғастыру|вернуть|прервать|продолжить|qaitaru|toqtatu|zhalgastyru)\b/u.test(t) && !/[([{,]$/.test(t)) return Math.max(0, ind - unit);
        return ind;
    }
    return 0;
}

/* fold a line whose following lines are indented deeper */
function tilFold(state, from, to) {
    const line = state.doc.lineAt(from);
    if (!line.text.trim() || /^\s*#(?!\[)/.test(line.text)) return null;
    const base = indentOf(line.text);
    let end = null;
    for (let n = line.number + 1; n <= state.doc.lines; n++) {
        const l = state.doc.line(n);
        if (!l.text.trim()) continue;
        if (indentOf(l.text) <= base) break;
        end = l;
    }
    if (!end) return null;
    return { from: line.to, to: end.to };
}

export const tilStreamLanguage = StreamLanguage.define(Object.assign({}, tilParser, {
    languageData: Object.assign({}, tilParser.languageData, { indentOnInput: DEDENT_RE }),
}));

export function tilLanguage() {
    return new LanguageSupport(tilStreamLanguage, [
        indentUnit.of('    '),
        indentService.of(tilIndent),
        foldService.of(tilFold),
    ]);
}

/* ── highlighting: CSS classes, shared by the editor and static previews ── */
export const tilHighlighter = tagHighlighter([
    { tag: tags.controlKeyword, class: 't-kw' },
    { tag: tags.definitionKeyword, class: 't-kw t-kw-def' },
    { tag: tags.operatorKeyword, class: 't-kw t-kw-op' },
    { tag: tags.modifier, class: 't-kw t-kw-mod' },
    { tag: tags.keyword, class: 't-kw' },
    { tag: tags.bool, class: 't-bool' },
    { tag: tags.self, class: 't-self' },
    { tag: tags.unit, class: 't-num t-unit' },
    { tag: tags.number, class: 't-num' },
    { tag: tags.special(tags.brace), class: 't-interp' },
    { tag: tags.special(tags.string), class: 't-fmt' },
    { tag: tags.escape, class: 't-esc' },
    { tag: tags.string, class: 't-str' },
    { tag: tags.comment, class: 't-comment' },
    { tag: tags.annotation, class: 't-attr' },
    { tag: tags.attributeName, class: 't-attr t-attr-name' },
    { tag: tags.function(tags.definition(tags.variableName)), class: 't-fn t-def' },
    { tag: tags.function(tags.propertyName), class: 't-fn t-method' },
    { tag: tags.function(tags.variableName), class: 't-fn' },
    { tag: tags.standard(tags.variableName), class: 't-builtin' },
    { tag: tags.typeName, class: 't-type' },
    { tag: tags.constant(tags.variableName), class: 't-const' },
    { tag: tags.propertyName, class: 't-prop' },
    { tag: tags.variableName, class: 't-var' },
    { tag: tags.operator, class: 't-op' },
    { tag: tags.punctuation, class: 't-punc' },
    { tag: tags.bracket, class: 't-bracket' },
    { tag: tags.atom, class: 't-atom' },
    { tag: tags.invalid, class: 't-invalid' },
]);

/* highlight TIL source into an element (welcome previews, hover code blocks) */
export function highlightInto(el, text) {
    el.textContent = '';
    const tree = tilStreamLanguage.parser.parse(text);
    let pos = 0;
    const put = (to, cls) => {
        if (to <= pos) return;
        const s = text.slice(pos, to);
        if (cls) {
            const span = document.createElement('span');
            span.className = cls;
            span.textContent = s;
            el.appendChild(span);
        } else el.appendChild(document.createTextNode(s));
        pos = to;
    };
    highlightTree(tree, tilHighlighter, (from, to, cls) => { put(from, ''); put(to, cls); });
    put(text.length, '');
    return el;
}
