// POSIX paths relative to the project root (what the API uses) and file:// URIs (what LSP uses).

export function basename(p) {
    const s = String(p || '').replace(/\/+$/, '');
    const i = s.lastIndexOf('/');
    return i >= 0 ? s.slice(i + 1) : s;
}

export function dirname(p) {
    const s = String(p || '').replace(/\/+$/, '');
    const i = s.lastIndexOf('/');
    return i > 0 ? s.slice(0, i) : '';
}

export function join(...parts) {
    return normalize(parts.filter((x) => x !== '' && x !== null && x !== undefined).join('/'));
}

export function normalize(p) {
    const out = [];
    for (const seg of String(p || '').replace(/\\/g, '/').split('/')) {
        if (!seg || seg === '.') continue;
        if (seg === '..') { out.pop(); continue; }
        out.push(seg);
    }
    return out.join('/');
}

export function extname(p) {
    const b = basename(p);
    const i = b.lastIndexOf('.');
    return i > 0 ? b.slice(i) : '';
}

export function isTil(p) { return /\.til$/i.test(p || ''); }

/* "C:\\Users\\x" or "/home/x" -> "file:///C:/Users/x" / "file:///home/x" */
export function rootToUri(root) {
    let r = String(root || '/').replace(/\\/g, '/');
    if (/^[A-Za-z]:/.test(r)) r = '/' + r;
    if (!r.startsWith('/')) r = '/' + r;
    r = r.replace(/\/+$/, '');
    return 'file://' + r.split('/').map((seg, i) => (i === 1 && /^[A-Za-z]:$/.test(seg) ? seg : encodeURIComponent(seg))).join('/');
}

export function pathToUri(rootUri, rel) {
    const segs = normalize(rel).split('/').filter(Boolean).map(encodeURIComponent);
    return rootUri.replace(/\/+$/, '') + '/' + segs.join('/');
}

function decodeUriPath(uri) {
    let s = String(uri || '');
    if (s.startsWith('file://')) s = s.slice(7);
    try { s = decodeURIComponent(s); } catch (e) { /* keep as is */ }
    return s;
}

/* a URI of a file inside the project -> its relative path; null when outside */
export function uriToPath(rootUri, uri) {
    const root = decodeUriPath(rootUri).replace(/\/+$/, '');
    const full = decodeUriPath(uri);
    const winLike = /^\/[A-Za-z]:/.test(root);
    const a = winLike ? root.toLowerCase() : root;
    const b = winLike ? full.toLowerCase() : full;
    if (b === a) return '';
    if (b.startsWith(a + '/')) return full.slice(root.length + 1);
    return null;
}

/* compare paths for sorting: dirs first, then natural, case-insensitive */
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
export function compareNames(a, b) { return collator.compare(a, b); }
