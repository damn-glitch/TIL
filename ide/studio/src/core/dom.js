// Small DOM helpers: h() builds elements, plus timing utilities.

export function h(tag, props, ...children) {
    const el = tag instanceof Element ? tag : document.createElement(tag);
    if (props) {
        for (const key in props) {
            const v = props[key];
            if (v === undefined || v === null || v === false) continue;
            if (key === 'class' || key === 'className') el.className = Array.isArray(v) ? v.filter(Boolean).join(' ') : v;
            else if (key === 'style') {
                if (typeof v === 'string') el.style.cssText = v;
                else for (const s in v) el.style.setProperty(s.startsWith('--') ? s : s.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase()), v[s]);
            } else if (key === 'dataset') Object.assign(el.dataset, v);
            else if (key === 'on') for (const ev in v) el.addEventListener(ev, v[ev]);
            else if (key === 'text') el.textContent = v;
            else if (key === 'ref') v(el);
            else if (key in el && typeof v !== 'string' && key !== 'list') el[key] = v;
            else if (v === true) el.setAttribute(key, '');
            else el.setAttribute(key, v);
        }
    }
    append(el, children);
    return el;
}

function append(el, children) {
    for (const c of children) {
        if (c === null || c === undefined || c === false) continue;
        if (Array.isArray(c)) append(el, c);
        else el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
    }
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function clear(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
    return el;
}

export function debounce(fn, ms) {
    let t = 0;
    const d = (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
    d.cancel = () => clearTimeout(t);
    d.flush = (...args) => { clearTimeout(t); fn(...args); };
    return d;
}

export function throttle(fn, ms) {
    let last = 0, timer = 0, pending = null;
    return (...args) => {
        const now = performance.now();
        pending = args;
        if (now - last >= ms) {
            last = now;
            fn(...pending);
            pending = null;
        } else if (!timer) {
            timer = setTimeout(() => {
                timer = 0;
                last = performance.now();
                if (pending) { fn(...pending); pending = null; }
            }, ms - (now - last));
        }
    };
}

export function nextFrame() {
    return new Promise((r) => requestAnimationFrame(() => r()));
}

/* move the focus now (the target is visible already) and, when that did not
   stick (a target that renders a frame later), once more after the next
   frame, unless the focus went somewhere else meanwhile: fast typing right
   after opening something always lands in it, and nothing steals the focus
   back later */
export function focusSoon(doFocus, container) {
    const before = document.activeElement;
    doFocus();
    requestAnimationFrame(() => {
        const now = document.activeElement;
        if (container && container.isConnected && container.contains(now)) return;
        if (!now || now === before || now === document.body) doFocus();
    });
}

let uid = 0;
export const uniqueId = (prefix = 'id') => `${prefix}-${++uid}`;

export function isMac() {
    return /Mac|iPhone|iPad/.test(navigator.platform || '') || /Mac OS X/.test(navigator.userAgent || '');
}

/* format a size in bytes */
export function fmtBytes(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return '';
    if (n < 1024) return `${n} B`;
    if (n < 1048576) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
    return `${(n / 1048576).toFixed(1)} MB`;
}

export function fmtMs(ms) {
    if (ms < 1000) return `${Math.max(0, Math.round(ms))} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(2)} s`;
    const m = Math.floor(ms / 60000);
    return `${m}:${String(Math.floor((ms % 60000) / 1000)).padStart(2, '0')} min`;
}

/* run a click handler on Enter/Space too (for elements that are not buttons) */
export function onActivate(el, fn) {
    el.addEventListener('click', fn);
    el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(e); }
    });
}

/* place a floating element near an anchor rect, inside the viewport */
export function placeNear(el, rect, { gap = 4, align = 'start', prefer = 'below' } = {}) {
    const vw = window.innerWidth, vh = window.innerHeight;
    el.style.left = '0px';
    el.style.top = '0px';
    const w = el.offsetWidth, ht = el.offsetHeight;
    let x = align === 'end' ? rect.right - w : rect.left;
    let y = prefer === 'above' ? rect.top - ht - gap : rect.bottom + gap;
    if (prefer !== 'above' && y + ht > vh - 8 && rect.top - ht - gap > 8) y = rect.top - ht - gap;
    if (prefer === 'above' && y < 8) y = rect.bottom + gap;
    x = Math.max(8, Math.min(x, vw - w - 8));
    y = Math.max(8, Math.min(y, vh - ht - 8));
    el.style.left = `${Math.round(x)}px`;
    el.style.top = `${Math.round(y)}px`;
}
