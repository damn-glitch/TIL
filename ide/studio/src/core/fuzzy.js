// Fuzzy matching for quick open and the command palette: subsequence match
// with a score that prefers word starts, consecutive runs and short targets.

export function fuzzyMatch(query, target) {
    if (!query) return { score: 0, positions: [] };
    const q = query.toLowerCase();
    const t = target.toLowerCase();
    // exact substring: best, and cheap
    const sub = t.indexOf(q);
    if (sub >= 0) {
        const positions = [];
        for (let i = 0; i < q.length; i++) positions.push(sub + i);
        let score = 1000 - sub - (t.length - q.length) * 0.5;
        if (sub === 0 || isBoundary(target, sub)) score += 200;
        return { score, positions };
    }
    const positions = [];
    let score = 0, ti = 0, prev = -2;
    for (let qi = 0; qi < q.length; qi++) {
        const ch = q[qi];
        if (ch === ' ') continue;
        let found = -1;
        // prefer a word boundary match ahead
        for (let k = ti; k < t.length; k++) {
            if (t[k] === ch && isBoundary(target, k)) { found = k; break; }
        }
        if (found < 0) found = t.indexOf(ch, ti);
        if (found < 0) return null;
        score += found === prev + 1 ? 30 : 0;
        score += isBoundary(target, found) ? 25 : 0;
        score -= Math.min(found - ti, 20);
        positions.push(found);
        prev = found;
        ti = found + 1;
    }
    return { score: score - target.length * 0.2, positions };
}

function isBoundary(s, i) {
    if (i === 0) return true;
    const p = s[i - 1], c = s[i];
    return p === '/' || p === '_' || p === '-' || p === '.' || p === ' ' || (p.toLowerCase() === p && c.toUpperCase() === c && c.toLowerCase() !== c);
}

/* build a fragment with the matched characters wrapped in <mark> */
export function highlighted(text, positions) {
    const frag = document.createDocumentFragment();
    if (!positions || !positions.length) { frag.appendChild(document.createTextNode(text)); return frag; }
    const set = new Set(positions);
    let run = '', inMark = false;
    const flush = () => {
        if (!run) return;
        if (inMark) { const m = document.createElement('mark'); m.textContent = run; frag.appendChild(m); } else frag.appendChild(document.createTextNode(run));
        run = '';
    };
    for (let i = 0; i < text.length; i++) {
        const m = set.has(i);
        if (m !== inMark) { flush(); inMark = m; }
        run += text[i];
    }
    flush();
    return frag;
}
