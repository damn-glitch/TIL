// Keyword translation in JavaScript (the web face and the test mock): the
// same rules as til.translate_source in src/til.py - keywords, booleans and
// localized builtin names change; identifiers, strings, comments and
// attributes stay; "else if" <-> "elif"; #[lang: ...] pragmas are rewritten.
import TABLES from '../../tables.json' with { type: 'json' };
import { codeOnly } from './lang_scan.js';

function languagesOf(text) {
    const langs = [...TABLES.default_languages];
    const re = /^[ \t]*#\[\s*lang\s*:\s*([^\]]*)\]/gm;
    let m;
    while ((m = re.exec(text))) for (const c of m[1].split(',')) { const x = c.trim().toLowerCase(); if (TABLES.languages.includes(x) && !langs.includes(x)) langs.push(x); }
    return langs;
}

export function translateSource(source, target) {
    if (!TABLES.languages.includes(target)) throw new Error(`unknown language '${target}'`);
    const langs = languagesOf(source);
    const kw = new Map(Object.entries(TABLES.keywords.en));
    const aliases = new Map();
    for (const l of langs) {
        if (l === 'en') continue;
        for (const [w, ty] of Object.entries(TABLES.keywords[l] || {})) kw.set(w, ty);
        for (const [w, c] of Object.entries(TABLES.aliases[l] || {})) aliases.set(w, c);
    }
    const ctx = new Map();
    for (const l of langs) for (const [w, ty] of Object.entries((TABLES.contextual || {})[l] || {})) ctx.set(w, ty);
    const english = TABLES.keywords.en;
    const wordFor = new Map();
    for (const [w, ty] of Object.entries(english)) if (!wordFor.has(ty)) wordFor.set(ty, w);
    const targetTable = target === 'en' ? english : (TABLES.keywords[target] || {});
    if (target !== 'en') {
        for (const [w, ty] of Object.entries(targetTable)) {
            if (!wordFor.has(ty) || Object.prototype.hasOwnProperty.call(english, wordFor.get(ty))) wordFor.set(ty, w);
        }
    }
    for (const [w, ty] of Object.entries((TABLES.contextual || {})[target] || {})) wordFor.set(ty, w);
    const trueWords = new Set(TABLES.true_words);
    let trueWord = 'true', falseWord = 'false';
    if (target !== 'en') {
        for (const [w, ty] of Object.entries(targetTable)) {
            if (ty !== 'BOOL') continue;
            if (trueWords.has(w) && trueWord === 'true') trueWord = w;
            if (!trueWords.has(w) && falseWord === 'false') falseWord = w;
        }
    }
    const canonToAlias = new Map();
    for (const [w, c] of Object.entries(TABLES.aliases[target] || {})) if (!canonToAlias.has(c)) canonToAlias.set(c, w);

    const lines = source.split('\n');
    const out = [];
    for (const line of lines) {
        if (/^[ \t]*#\[\s*lang\s*:/.test(line)) { out.push(null); continue; }
        const code = codeOnly(line, false).replace(/#\[[^\]]*\]/g, (m) => ' '.repeat(m.length));
        const re = /[\p{L}_][\p{L}\p{N}_]*/gu;
        const toks = [];
        let m;
        while ((m = re.exec(code))) toks.push({ text: m[0], start: m.index, end: m.index + m[0].length });
        const reps = [];
        for (let k = 0; k < toks.length; k++) {
            const tk = toks[k];
            // f"..." / b"...": the prefix is not a word
            if ((tk.text === 'f' || tk.text === 'F') && /["']/.test(line[tk.end] || '')) continue;
            let ty = kw.get(tk.text);
            const rest = code.slice(tk.end);
            if (!ty && ctx.has(tk.text)) {
                const c = ctx.get(tk.text);
                const lineStart = !code.slice(0, tk.start).trim();
                if ((c === 'WITH' && lineStart && !rest.trim()) || (c === 'RESUME' && /^\s*\(/.test(rest)) ||
                    (c === 'PERFORM' && /^\s+[\p{L}_]/u.test(rest)) || (c === 'HANDLE' && (!rest.trim() || /^\s+[\p{L}_\d"']/u.test(rest)))) ty = c;
            }
            if (ty) {
                const next = toks[k + 1];
                if (target === 'en' && ty === 'ELSE' && next && kw.get(next.text) === 'IF') {
                    reps.push([tk.start, next.end, 'elif']);
                    k++;
                    continue;
                }
                let neu;
                if (ty === 'BOOL') neu = trueWords.has(tk.text) ? trueWord : falseWord;
                else if (ty === 'ELIF' && target !== 'en') neu = `${wordFor.get('ELSE')} ${wordFor.get('IF')}`;
                else neu = wordFor.get(ty) || tk.text;
                if (neu !== tk.text) reps.push([tk.start, tk.end, neu]);
                continue;
            }
            // identifiers: localized builtin names
            if (/^\s*\./.test(code.slice(0, tk.start).slice(-1)) || code[tk.start - 1] === '.') continue;
            const canon = aliases.get(tk.text) || tk.text;
            let neu = tk.text;
            if (target === 'en') neu = canon;
            else if (canonToAlias.has(canon) && (tk.text === canon || aliases.has(tk.text))) neu = canonToAlias.get(canon);
            if (neu !== tk.text) reps.push([tk.start, tk.end, neu]);
        }
        let s = line;
        for (const [a, b, r] of reps.sort((x, y) => y[0] - x[0])) s = s.slice(0, a) + r + s.slice(b);
        out.push(s);
    }
    let text = out.filter((l) => l !== null).join('\n').replace(/^\n+/, '');
    if (target === 'ru' || target === 'kk-latn') text = `#[lang: ${target}]\n` + text;
    return text;
}
