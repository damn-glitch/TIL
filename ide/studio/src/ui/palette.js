// The quick pick: command palette (">"), quick open (files), go to line
// (":"), symbols in the file ("@") and in the workspace ("#"). Also used for
// simple pickers (backend, device, language).
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { fuzzyMatch, highlighted } from '../core/fuzzy.js';
import { formatKeys } from '../core/commands.js';
import { t } from '../core/i18n.js';

let current = null;

export function closeQuickPick() {
    if (current) current.close();
}

/**
 * provider(query) -> [{label, description, detail, icon, keys, value, iconClass, group}] (or a Promise)
 * modes: {prefix: provider} for prefix switching ("" is the default)
 */
export function quickPick({ placeholder, value = '', modes, onAccept, onChange, matchOn = 'label', ariaLabel, emptyText }) {
    closeQuickPick();
    const restore = document.activeElement;
    const box = h('div', { class: 'quickpick', role: 'dialog', 'aria-label': ariaLabel || placeholder });
    const input = h('input', {
        class: 'qp-input', type: 'text', spellcheck: 'false', autocomplete: 'off', placeholder,
        role: 'combobox', 'aria-expanded': 'true', 'aria-controls': 'qp-list', 'aria-autocomplete': 'list',
    });
    input.value = value;
    const list = h('div', { class: 'qp-list', id: 'qp-list', role: 'listbox' });
    const empty = h('div', { class: 'qp-empty' });
    box.append(h('div', { class: 'qp-head' }, icon('search', { className: 'qp-search-icon' }), input), list, empty);
    const scrim = h('div', { class: 'qp-scrim' });
    document.body.append(scrim, box);
    let items = [];
    let index = 0;
    let seq = 0;
    let closed = false;

    function modeFor(text) {
        const prefixes = Object.keys(modes).filter((p) => p).sort((a, b) => b.length - a.length);
        for (const p of prefixes) if (text.startsWith(p)) return { prefix: p, query: text.slice(p.length).trimStart() };
        return { prefix: '', query: text };
    }

    async function refresh() {
        const mine = ++seq;
        const { prefix, query } = modeFor(input.value);
        const provider = modes[prefix];
        let raw = provider ? await provider(query) : [];
        if (mine !== seq || closed) return;
        if (!Array.isArray(raw)) raw = [];
        let ranked;
        if (raw.ranked) ranked = raw.map((it) => ({ it, m: null }));
        else if (query && !raw.noFilter) {
            ranked = [];
            for (const it of raw) {
                const target = matchOn === 'label+description' ? `${it.label} ${it.description || ''}` : it.filterText || it.label;
                const m = fuzzyMatch(query, target);
                if (m) ranked.push({ it, m: matchOn === 'label' || !it.filterText ? m : null, score: m.score + (it.boost || 0) });
            }
            ranked.sort((a, b) => b.score - a.score);
        } else ranked = raw.map((it) => ({ it, m: null }));
        items = ranked.slice(0, 300);
        index = 0;
        render();
        if (onChange) onChange(input.value);
    }

    function render() {
        clear(list);
        let lastGroup = null;
        items.forEach(({ it, m }, i) => {
            if (it.group && it.group !== lastGroup) {
                list.appendChild(h('div', { class: 'qp-group', role: 'presentation' }, it.group));
                lastGroup = it.group;
            }
            const row = h('div', { class: 'qp-item', role: 'option', id: `qp-opt-${i}`, 'aria-selected': String(i === index) },
                it.icon ? icon(it.icon, { className: `qp-icon ${it.iconClass || ''}` }) : h('span', { class: 'qp-icon-space' }),
                h('span', { class: 'qp-label' }, m && m.positions && m.positions.every((p) => p < it.label.length) ? highlighted(it.label, m.positions) : it.label),
                it.description ? h('span', { class: 'qp-desc' }, it.description) : null,
                it.keys ? h('span', { class: 'qp-keys' }, ...formatKeys(it.keys).split(/(?<=\+)|(?=\+)/).filter((x) => x !== '+').map((k) => h('kbd', {}, k))) : null,
            );
            row.addEventListener('mousedown', (e) => e.preventDefault());
            row.addEventListener('click', () => accept(i));
            row.addEventListener('mousemove', () => { if (index !== i) { index = i; mark(); } });
            list.appendChild(row);
        });
        empty.textContent = items.length ? '' : (emptyText || t('palette.noResults'));
        empty.hidden = Boolean(items.length);
        mark();
    }

    function mark() {
        const rows = list.querySelectorAll('.qp-item');
        rows.forEach((r, i) => r.setAttribute('aria-selected', String(i === index)));
        const cur = rows[index];
        if (cur) {
            input.setAttribute('aria-activedescendant', cur.id);
            const top = cur.offsetTop, bottom = top + cur.offsetHeight;
            if (top < list.scrollTop) list.scrollTop = top - 4;
            else if (bottom > list.scrollTop + list.clientHeight) list.scrollTop = bottom - list.clientHeight + 4;
        }
    }

    function accept(i) {
        const entry = items[i];
        if (!entry) return;
        const { prefix, query } = modeFor(input.value);
        close(false);
        onAccept(entry.it, { prefix, query });
    }

    function close(returnFocus = true) {
        if (closed) return;
        closed = true;
        box.remove();
        scrim.remove();
        current = null;
        if (returnFocus && restore && restore.focus && restore.isConnected) restore.focus({ preventScroll: true });
    }

    input.addEventListener('input', () => refresh());
    input.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown') { e.preventDefault(); if (items.length) { index = (index + 1) % items.length; mark(); } }
        else if (e.key === 'ArrowUp') { e.preventDefault(); if (items.length) { index = (index - 1 + items.length) % items.length; mark(); } }
        else if (e.key === 'PageDown') { e.preventDefault(); index = Math.min(items.length - 1, index + 10); mark(); }
        else if (e.key === 'PageUp') { e.preventDefault(); index = Math.max(0, index - 10); mark(); }
        else if (e.key === 'Enter') { e.preventDefault(); const { prefix } = modeFor(input.value); if (prefix === ':' && modes[':'].accept) { close(false); modes[':'].accept(input.value.slice(1)); } else accept(index); }
        else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(true); }
        else if (e.key === 'Tab') { e.preventDefault(); }
    });
    scrim.addEventListener('mousedown', () => close(true));
    current = { close, input, refresh, box };
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    refresh();
    return current;
}
