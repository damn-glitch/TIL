// A light symbol scan of TIL source (no compiler needed): functions with
// parameters and return types, structs with fields, enums with variants,
// traits, impls with methods, constants, globals and locals; identifier
// occurrences outside strings and comments. Used by the web face's language
// server and by the test mock server. Keyword spellings of every language
// come from the generated tables.
import TABLES from '../../tables.json' with { type: 'json' };

const ALL = new Map();
for (const tbl of Object.values(TABLES.keywords)) for (const [w, ty] of Object.entries(tbl)) ALL.set(w, ty);
for (const tbl of Object.values(TABLES.contextual || {})) for (const [w, ty] of Object.entries(tbl)) if (!ALL.has(w)) ALL.set(w, ty);
const ALIASES = new Map();
for (const tbl of Object.values(TABLES.aliases || {})) for (const [w, c] of Object.entries(tbl)) ALIASES.set(w, c);

export const KEYWORD_TYPE = ALL;
export const ALIAS_OF = ALIASES;

const IDENT = /[\p{L}_][\p{L}\p{N}_]*/gu;
const isType = (w, ty) => ALL.get(w) === ty;

export function indentOf(text) {
    let n = 0;
    for (const c of text) { if (c === ' ') n++; else if (c === '\t') n += 4; else break; }
    return n;
}

/* the line with strings and the comment blanked out (same length) */
export function codeOnly(line, keepInterpolations = true) {
    let out = '';
    for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '#' && line[i + 1] !== '[') { out += ' '.repeat(line.length - i); break; }
        if (c === '"' || c === "'") {
            const q = c;
            let j = i + 1;
            // f-string interpolations keep their code
            const fmt = keepInterpolations && i > 0 && (line[i - 1] === 'f' || line[i - 1] === 'F');
            let piece = q;
            while (j < line.length && line[j] !== q) {
                if (line[j] === '\\') { piece += '  '; j += 2; continue; }
                if (fmt && line[j] === '{' && line[j + 1] !== '{') {
                    let depth = 1, k = j + 1;
                    piece += '{';
                    while (k < line.length && depth > 0 && line[k] !== q) {
                        if (line[k] === '{') depth++;
                        else if (line[k] === '}') { depth--; if (!depth) break; }
                        piece += line[k];
                        k++;
                    }
                    piece += k < line.length && line[k] === '}' ? '}' : '';
                    j = k + 1;
                    continue;
                }
                piece += ' ';
                j++;
            }
            out += piece + (j < line.length ? q : '');
            i = j;
            continue;
        }
        out += c;
    }
    return out;
}

function parseParams(text) {
    const out = [];
    let depth = 0, cur = '';
    for (const c of text) {
        if (c === '(' || c === '[' || c === '<' || c === '{') depth++;
        if (c === ')' || c === ']' || c === '>' || c === '}') depth--;
        if (c === ',' && depth === 0) { out.push(cur); cur = ''; } else cur += c;
    }
    if (cur.trim()) out.push(cur);
    return out.map((p) => {
        const m = /^\s*(?:mut\s+)?([\p{L}_][\p{L}\p{N}_]*)\s*(?::\s*([^=]+?))?\s*(?:=\s*(.+))?\s*$/u.exec(p);
        return m ? { name: m[1], type: (m[2] || '').trim(), default: m[3] ? m[3].trim() : undefined } : null;
    }).filter(Boolean);
}

/**
 * scan(text) -> {
 *   symbols: [{name, kind: function|method|struct|enum|trait|const|global|field|variant|impl, line, col, endLine,
 *              params, ret, detail, doc, parent, children}],
 *   functions: Map name -> symbol, locals: [{name, line, col, fn}], langs }
 */
export function scan(text) {
    const lines = text.split('\n');
    const symbols = [];
    const top = [];
    const locals = [];
    let container = null;          // {sym, indent} of struct/enum/impl/trait
    let fn = null;                 // {sym, indent}
    const docAbove = (i) => {
        const doc = [];
        for (let k = i - 1; k >= 0; k--) {
            const l = lines[k].trim();
            if (l.startsWith('#[')) continue;
            if (!l.startsWith('#')) break;
            doc.unshift(l.replace(/^#\s?/, ''));
        }
        return doc.join('\n').trim();
    };
    const attrsAbove = (i) => {
        const out = [];
        for (let k = i - 1; k >= 0; k--) {
            const l = lines[k].trim();
            if (!l.startsWith('#[')) break;
            out.unshift(l);
        }
        return out;
    };
    for (let i = 0; i < lines.length; i++) {
        const raw = lines[i];
        if (!raw.trim()) continue;
        const code = codeOnly(raw);
        const t = code.trim();
        if (!t || t.startsWith('#')) continue;
        const ind = indentOf(raw);
        if (fn && ind <= fn.indent) { fn.sym.endLine = i - 1; fn = null; }
        if (container && ind <= container.indent) { container.sym.endLine = i - 1; container = null; }
        const words = t.match(IDENT) || [];
        const w0 = words[0] || '';
        const col0 = raw.length - raw.trimStart().length;
        // struct / enum / trait / impl
        if (isType(w0, 'STRUCT') || isType(w0, 'ENUM') || isType(w0, 'TRAIT') || isType(w0, 'IMPL') || (isType(w0, 'PUB') && words[1] && (isType(words[1], 'STRUCT') || isType(words[1], 'ENUM')))) {
            const off = isType(w0, 'PUB') ? 1 : 0;
            const kw = words[off];
            const ty = ALL.get(kw);
            const name = words[off + 1] || '';
            const kind = ty === 'STRUCT' ? 'struct' : ty === 'ENUM' ? 'enum' : ty === 'TRAIT' ? 'trait' : 'impl';
            let label = name;
            if (kind === 'impl') {
                const forIdx = words.findIndex((x, k) => k > off + 1 && isType(x, 'FOR'));
                label = forIdx > 0 ? `${name} for ${words[forIdx + 1] || ''}` : name;
            }
            const sym = { name: label, typeName: kind === 'impl' ? (words[words.findIndex((x, k) => k > off + 1 && isType(x, 'FOR')) + 1] || name) : name, kind, line: i, col: raw.indexOf(name, col0), endLine: i, children: [], doc: docAbove(i), detail: t };
            symbols.push(sym);
            top.push(sym);
            container = { sym, indent: ind };
            if (/\{.*\}/.test(t) && kind === 'struct') container = null;
            continue;
        }
        if (container && ind > container.indent && !fn) {
            const c = container.sym;
            if (c.kind === 'struct') {
                const m = /^([\p{L}_][\p{L}\p{N}_]*)\s*:\s*(.+)$/u.exec(t);
                if (m) {
                    const f = { name: m[1], kind: 'field', line: i, col: raw.indexOf(m[1]), detail: `${m[1]}: ${m[2].trim()}`, type: m[2].trim(), parent: c.name };
                    c.children.push(f);
                    symbols.push(f);
                    continue;
                }
            } else if (c.kind === 'enum') {
                const m = /^([\p{L}_][\p{L}\p{N}_]*)\s*(\(([^)]*)\))?/u.exec(t);
                if (m) {
                    const v = { name: m[1], kind: 'variant', line: i, col: raw.indexOf(m[1]), detail: m[0], parent: c.name };
                    c.children.push(v);
                    symbols.push(v);
                    continue;
                }
            }
        }
        // function definitions: [pub] [fn] name(params) [-> ret]
        let off = 0;
        if (isType(words[0], 'PUB')) off = 1;
        if (isType(words[off], 'FN')) off++;
        const defRe = /^(?:[\p{L}_][\p{L}\p{N}_]*\s+){0,2}?([\p{L}_][\p{L}\p{N}_]*)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*(?:->\s*(.+?))?\s*$/u;
        const m = defRe.exec(t);
        const nameWord = words[off];
        if (m && m[1] === nameWord && !ALL.has(nameWord) && (off > 0 || looksLikeDef(t, m, ind, container, lines, i))) {
            const params = parseParams(m[2]);
            const ret = (m[3] || '').trim();
            const inImpl = container && (container.sym.kind === 'impl' || container.sym.kind === 'trait');
            const canon = ALIASES.get(nameWord) || nameWord;
            const sym = {
                name: nameWord, canonical: canon, kind: inImpl ? 'method' : 'function', line: i, col: raw.indexOf(nameWord, col0), endLine: i,
                params, ret, doc: docAbove(i), attrs: attrsAbove(i), parent: inImpl ? container.sym.typeName : null,
                detail: `${nameWord}(${params.map((p) => (p.type ? `${p.name}: ${p.type}` : p.name)).join(', ')})${ret ? ` -> ${ret}` : ''}`,
            };
            symbols.push(sym);
            if (inImpl) container.sym.children.push(sym); else top.push(sym);
            fn = { sym, indent: ind };
            for (const p of params) locals.push({ name: p.name, line: i, col: raw.indexOf(p.name, raw.indexOf('(')), fn: sym, kind: 'param', type: p.type });
            continue;
        }
        // const NAME = ... / let|var at top level (globals) or inside functions (locals)
        if (isType(w0, 'CONST') || isType(w0, 'LET') || isType(w0, 'VAR')) {
            const name = words[1] && isType(words[1], 'MUT') ? words[2] : words[1];
            if (!name) continue;
            const mm = new RegExp(`^\\S+\\s+(?:\\S+\\s+)?${escapeRe(name)}\\s*(?::\\s*([^=]+?))?\\s*=\\s*(.+)$`, 'u').exec(t);
            const valueText = mm ? mm[2] : '';
            const typeText = mm && mm[1] ? mm[1].trim() : '';
            const entry = { name, line: i, col: raw.indexOf(name, col0 + w0.length), kind: isType(w0, 'CONST') ? 'const' : isType(w0, 'VAR') ? 'var' : 'let', value: valueText, type: typeText };
            if (fn && ind > fn.indent) { locals.push(Object.assign(entry, { fn: fn.sym })); continue; }
            const sym = { name, kind: isType(w0, 'CONST') ? 'const' : 'global', line: i, col: entry.col, detail: t, doc: docAbove(i), mutable: isType(w0, 'VAR'), value: valueText, type: typeText };
            symbols.push(sym);
            top.push(sym);
            continue;
        }
        // for x in ... (a local)
        if (isType(w0, 'FOR') && words[1] && fn) {
            locals.push({ name: words[1], line: i, col: raw.indexOf(words[1], col0 + w0.length), fn: fn.sym, kind: 'loop' });
        }
    }
    if (fn) fn.sym.endLine = lines.length - 1;
    if (container) container.sym.endLine = lines.length - 1;
    const functions = new Map();
    for (const s of symbols) if (s.kind === 'function') { functions.set(s.name, s); if (s.canonical) functions.set(s.canonical, s); }
    return { symbols, top, locals, functions, lines };
}

function looksLikeDef(t, m, ind, container, lines, i) {
    if (m[3]) return true;
    const params = m[2].trim();
    if (/["']/.test(params)) return false;
    if (/^[\p{L}_][\p{L}\p{N}_]*\s*:/u.test(params) || /^(self|өзін|ozin|сам)\b/.test(params)) return true;
    if (container && (container.sym.kind === 'impl' || container.sym.kind === 'trait')) return true;
    // `name()` at column 0 followed by an indented line
    if (ind === 0) {
        for (let k = i + 1; k < lines.length; k++) {
            if (!lines[k].trim()) continue;
            return indentOf(lines[k]) > 0;
        }
    }
    return false;
}

export function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

/* identifier occurrences of `name` in code (not in strings/comments): [{line, col}] */
export function occurrences(lines, name, fromLine = 0, toLine = lines.length - 1) {
    const out = [];
    const re = new RegExp(`(?<![\\p{L}\\p{N}_])${escapeRe(name)}(?![\\p{L}\\p{N}_])`, 'gu');
    for (let i = Math.max(0, fromLine); i <= Math.min(lines.length - 1, toLine); i++) {
        const code = codeOnly(lines[i]);
        // attributes #[requires: x > 0] count as code
        re.lastIndex = 0;
        let m;
        while ((m = re.exec(code))) out.push({ line: i, col: m.index });
    }
    return out;
}

/* the identifier at (line, character), or null: {word, start, end} */
export function wordAt(lines, line, character) {
    const text = lines[line] || '';
    IDENT.lastIndex = 0;
    let m;
    while ((m = IDENT.exec(text))) {
        if (m.index <= character && character <= m.index + m[0].length) return { word: m[0], start: m.index, end: m.index + m[0].length };
    }
    return null;
}

/* the function whose body contains a line */
export function functionAt(info, line) {
    let best = null;
    for (const s of info.symbols) {
        if ((s.kind === 'function' || s.kind === 'method') && s.line <= line && line <= (s.endLine ?? s.line)) best = s;
    }
    return best;
}

export function levenshtein(a, b) {
    if (a === b) return 0;
    const m = a.length, n = b.length;
    if (!m) return n;
    if (!n) return m;
    let prev = Array.from({ length: n + 1 }, (_, j) => j);
    for (let i = 1; i <= m; i++) {
        const cur = [i];
        for (let j = 1; j <= n; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
        prev = cur;
    }
    return prev[n];
}

export { TABLES };
