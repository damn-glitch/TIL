// A small, safe Markdown renderer for hover texts and documentation: builds
// DOM nodes (never innerHTML), so raw HTML in the text shows as text.
// Supports paragraphs, # headings, - lists, ``` code blocks (highlighted by
// the given function), `code`, **bold**, *italic*, [links](https://...), ---.

export function renderMarkdown(text, { highlight } = {}) {
    const root = document.createElement('div');
    root.className = 'md';
    const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
    let i = 0;
    let para = [];
    let list = null;
    const flushPara = () => {
        if (para.length) {
            const p = document.createElement('p');
            inline(p, para.join('\n'));
            root.appendChild(p);
            para = [];
        }
    };
    const flushList = () => { if (list) { root.appendChild(list); list = null; } };
    while (i < lines.length) {
        const line = lines[i];
        const fence = /^\s*```\s*([\w-]*)\s*$/.exec(line);
        if (fence) {
            flushPara(); flushList();
            const lang = fence[1];
            const body = [];
            i++;
            while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
            i++;
            const pre = document.createElement('pre');
            const code = document.createElement('code');
            const src = body.join('\n');
            if (highlight && (!lang || lang === 'til')) highlight(code, src);
            else code.textContent = src;
            pre.appendChild(code);
            root.appendChild(pre);
            continue;
        }
        if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) { flushPara(); flushList(); root.appendChild(document.createElement('hr')); i++; continue; }
        const head = /^(#{1,4})\s+(.*)$/.exec(line);
        if (head) {
            flushPara(); flushList();
            const el = document.createElement('h' + Math.min(6, head[1].length + 2));
            inline(el, head[2]);
            root.appendChild(el);
            i++;
            continue;
        }
        const item = /^\s*(?:[-*+]|\d+\.)\s+(.*)$/.exec(line);
        if (item) {
            flushPara();
            if (!list) list = document.createElement(/^\s*\d/.test(line) ? 'ol' : 'ul');
            const li = document.createElement('li');
            inline(li, item[1]);
            list.appendChild(li);
            i++;
            continue;
        }
        if (!line.trim()) { flushPara(); flushList(); i++; continue; }
        flushList();
        para.push(line);
        i++;
    }
    flushPara(); flushList();
    return root;
}

const INLINE = /(`+)([\s\S]*?)\1|\*\*([^*]+)\*\*|__([^_]+)__|\*([^*\s][^*]*)\*|_([^_\s][^_]*)_|\[([^\]]+)\]\(([^)\s]+)\)|\\([\\`*_[\]()#+\-.!])|(\n)/g;

function inline(el, text) {
    // a regex of its own per call: bold and italic recurse, and a shared
    // lastIndex would send the outer loop back to the start forever
    const re = new RegExp(INLINE.source, 'g');
    let last = 0;
    let m;
    while ((m = re.exec(text))) {
        if (m.index > last) el.appendChild(document.createTextNode(text.slice(last, m.index)));
        if (m[1]) { const c = document.createElement('code'); c.textContent = m[2].replace(/^ (.*) $/, '$1'); el.appendChild(c); }
        else if (m[3] || m[4]) { const b = document.createElement('strong'); inline(b, m[3] || m[4]); el.appendChild(b); }
        else if (m[5] || m[6]) { const it = document.createElement('em'); inline(it, m[5] || m[6]); el.appendChild(it); }
        else if (m[7]) {
            const href = m[8];
            if (/^https?:\/\//i.test(href)) {
                const a = document.createElement('a');
                a.href = href;
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
                a.textContent = m[7];
                el.appendChild(a);
            } else el.appendChild(document.createTextNode(m[7]));
        } else if (m[9]) el.appendChild(document.createTextNode(m[9]));
        else if (m[10]) el.appendChild(document.createElement('br'));
        last = re.lastIndex;
    }
    if (last < text.length) el.appendChild(document.createTextNode(text.slice(last)));
}

/* LSP hover contents (MarkupContent | MarkedString | MarkedString[]) -> markdown text */
export function hoverToMarkdown(contents) {
    if (!contents) return '';
    if (typeof contents === 'string') return contents;
    if (Array.isArray(contents)) return contents.map(hoverToMarkdown).filter(Boolean).join('\n\n---\n\n');
    if (contents.kind === 'plaintext') return String(contents.value || '').replace(/([\\`*_[\]])/g, '\\$1');
    if (contents.kind === 'markdown') return String(contents.value || '');
    if (contents.language !== undefined) return '```' + (contents.language || '') + '\n' + (contents.value || '') + '\n```';
    if (typeof contents.value === 'string') return contents.value;
    return '';
}
