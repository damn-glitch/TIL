// ANSI SGR colours for the output terminal. Stateful: a chunk may end inside
// an escape sequence, and colours carry over to the next chunk.

const BASIC = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'];

function xterm256(n) {
    if (n < 8) return `var(--ansi-${BASIC[n]})`;
    if (n < 16) return `var(--ansi-bright-${BASIC[n - 8]})`;
    if (n < 232) {
        const v = n - 16, r = Math.floor(v / 36), g = Math.floor((v % 36) / 6), b = v % 6;
        const k = (x) => (x ? 55 + x * 40 : 0);
        return `rgb(${k(r)},${k(g)},${k(b)})`;
    }
    const g = 8 + (n - 232) * 10;
    return `rgb(${g},${g},${g})`;
}

export class AnsiParser {
    constructor() { this.reset(); this.carry = ''; }

    reset() { this.fg = null; this.bg = null; this.bold = false; this.dim = false; this.italic = false; this.underline = false; this.inverse = false; }

    style() {
        let fg = this.fg, bg = this.bg;
        if (this.inverse) { [fg, bg] = [bg || 'var(--term-bg)', fg || 'var(--term-fg)']; }
        const parts = [];
        if (fg) parts.push(`color:${fg}`);
        if (bg) parts.push(`background:${bg}`);
        if (this.bold) parts.push('font-weight:600');
        if (this.dim) parts.push('opacity:.7');
        if (this.italic) parts.push('font-style:italic');
        if (this.underline) parts.push('text-decoration:underline');
        return parts.join(';');
    }

    sgr(params) {
        const p = params.length ? params : [0];
        for (let i = 0; i < p.length; i++) {
            const n = p[i];
            if (n === 0) this.reset();
            else if (n === 1) this.bold = true;
            else if (n === 2) this.dim = true;
            else if (n === 3) this.italic = true;
            else if (n === 4) this.underline = true;
            else if (n === 7) this.inverse = true;
            else if (n === 22) { this.bold = false; this.dim = false; }
            else if (n === 23) this.italic = false;
            else if (n === 24) this.underline = false;
            else if (n === 27) this.inverse = false;
            else if (n >= 30 && n <= 37) this.fg = `var(--ansi-${BASIC[n - 30]})`;
            else if (n >= 90 && n <= 97) this.fg = `var(--ansi-bright-${BASIC[n - 90]})`;
            else if (n >= 40 && n <= 47) this.bg = `var(--ansi-${BASIC[n - 40]})`;
            else if (n >= 100 && n <= 107) this.bg = `var(--ansi-bright-${BASIC[n - 100]})`;
            else if (n === 39) this.fg = null;
            else if (n === 49) this.bg = null;
            else if ((n === 38 || n === 48) && p[i + 1] === 5 && p[i + 2] !== undefined) {
                const col = xterm256(p[i + 2]);
                if (n === 38) this.fg = col; else this.bg = col;
                i += 2;
            } else if ((n === 38 || n === 48) && p[i + 1] === 2 && p[i + 4] !== undefined) {
                const col = `rgb(${p[i + 2]},${p[i + 3]},${p[i + 4]})`;
                if (n === 38) this.fg = col; else this.bg = col;
                i += 4;
            }
        }
    }

    /* text -> [{text, style}] */
    feed(text) {
        let s = this.carry + text;
        this.carry = '';
        const out = [];
        let i = 0, start = 0;
        const push = (end) => {
            if (end > start) out.push({ text: s.slice(start, end), style: this.style() });
        };
        while (i < s.length) {
            const ch = s.charCodeAt(i);
            if (ch === 27) {
                if (i + 1 >= s.length) { push(i); this.carry = s.slice(i); return out; }
                if (s[i + 1] === '[') {
                    let j = i + 2;
                    while (j < s.length && !/[@-~]/.test(s[j])) j++;
                    if (j >= s.length) {
                        if (j - i > 64) { i = j; continue; }      // garbage: drop
                        push(i);
                        this.carry = s.slice(i);
                        return out;
                    }
                    push(i);
                    if (s[j] === 'm') {
                        const params = s.slice(i + 2, j).split(';').filter((x) => x !== '').map((x) => parseInt(x, 10) || 0);
                        this.sgr(params);
                    }
                    i = j + 1;
                    start = i;
                    continue;
                }
                // other escapes (ESC c, ...): skip ESC and the next character
                push(i);
                i += 2;
                start = i;
                continue;
            }
            if (ch === 13 && s.charCodeAt(i + 1) !== 10) {   // bare \r: drop it
                push(i);
                i++;
                start = i;
                continue;
            }
            i++;
        }
        push(s.length);
        return out;
    }
}
