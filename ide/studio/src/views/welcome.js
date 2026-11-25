// The Welcome page: start actions, recent projects, "Learn TIL" cards, the
// examples gallery (highlighted previews) and the main keyboard shortcuts.
import EXAMPLES from 'virtual:examples';
import { h, clear } from '../core/dom.js';
import { icon, logoMark } from '../core/icons.js';
import { t, L } from '../core/i18n.js';
import { formatKeys } from '../core/commands.js';
import { load } from '../core/store.js';
import { highlightInto } from '../editor/til_lang.js';

export { EXAMPLES };

const LEARN = [
    { id: 'hello', icon: 'sparkle', key: 'learn.hello', color: 'cyan' },
    { id: 'kazakh', icon: 'translate', key: 'learn.languages', color: 'purple' },
    { id: 'structs', icon: 'symbolStruct', key: 'learn.types', color: 'yellow' },
    { id: 'effects', icon: 'zap', key: 'learn.effects', color: 'orange' },
    { id: 'proofs', icon: 'proofs', key: 'learn.proofs', color: 'green' },
    { id: 'animation', icon: 'screen', key: 'learn.graphics', color: 'pink' },
];

const FILTERS = [
    ['all', 'welcome.filterAll', () => true],
    ['basics', 'welcome.filterBasics', (e) => e.tags.some((x) => ['basics', 'types', 'language', 'module', 'input'].includes(x))],
    ['languages', 'welcome.filterLanguages', (e) => e.tags.some((x) => ['kk', 'ru', 'kk-latn'].includes(x))],
    ['screen', 'welcome.filterScreen', (e) => e.tags.includes('screen')],
    ['verified', 'welcome.filterVerified', (e) => e.tags.some((x) => ['contracts', 'proofs', 'effects', 'energy'].includes(x))],
];

export class WelcomePage {
    constructor(app) {
        this.app = app;
        this.iconName = 'home';
        this.persist = true;
        this.filter = 'all';
        this.el = h('div', { class: 'welcome', tabindex: '-1' });
        this.render();
    }

    title() { return t('welcome.tab'); }

    focus() { const b = this.el.querySelector('.wl-hero .btn'); if (b) b.focus({ preventScroll: true }); }

    onShow() { this.renderRecent(); }

    relocalize() { this.render(); }

    render() {
        const app = this.app;
        clear(this.el);
        const K = (id) => formatKeys(app.commands.keysFor(id)[0] || '');
        const action = (iconName, key, id, keysId) => {
            const b = h('button', { class: 'wl-action', type: 'button' }, icon(iconName), h('span', { class: 'wl-action-label' }, t(key)), keysId ? h('kbd', {}, K(keysId)) : null);
            b.addEventListener('click', () => app.commands.run(id));
            return b;
        };
        const newBtn = h('button', { class: 'btn primary large', type: 'button' }, icon('newFile'), t('welcome.newFile'));
        newBtn.addEventListener('click', () => app.commands.run(app.web ? 'file.newFile' : 'file.newFile'));
        const tourBtn = h('button', { class: 'btn large', type: 'button' }, icon('sparkle'), t('welcome.openTour'));
        tourBtn.addEventListener('click', () => this.openExample('tour'));
        const palBtn = h('button', { class: 'btn ghost large', type: 'button' }, icon('keyboard'), t('palette.showCommands'), h('kbd', {}, K('workbench.commandPalette')));
        palBtn.addEventListener('click', () => app.commands.run('workbench.commandPalette'));

        const hero = h('div', { class: 'wl-hero' },
            h('div', { class: 'wl-logo' }, logoMark(64)),
            h('div', { class: 'wl-hero-text' },
                h('h1', {}, 'TIL ', h('span', {}, 'Studio')),
                h('p', { class: 'wl-tagline' }, t('welcome.tagline')),
                h('div', { class: 'wl-badges' },
                    h('span', { class: 'chip accent' }, `TIL ${app.info.version || app.tables.version}`),
                    h('span', { class: 'chip' }, app.web ? t('status.web') : t('status.desktop')),
                    h('span', { class: 'chip' }, 'English · Қазақша · Русский'))),
            h('div', { class: 'wl-hero-actions' }, newBtn, tourBtn, palBtn));

        const start = h('section', { class: 'wl-card wl-start' }, h('h2', {}, t('welcome.start')),
            action('search', 'palette.quickOpen', 'workbench.quickOpen', 'workbench.quickOpen'),
            action('play', 'cmd.runFile', 'run.run', 'run.run'),
            app.caps.debugger ? action('bug', 'cmd.debugFile', 'debug.start', 'debug.start') : null,
            action('translate', 'cmd.translatePick', 'translate.pick'),
            action('keyboard', 'cmd.shortcuts', 'help.shortcuts', 'help.shortcuts'),
            action('book', 'cmd.docs', 'help.docs'),
            action('settings', 'activity.settings', 'workbench.show.settings', 'workbench.show.settings'));
        this.recentEl = h('div', { class: 'wl-recent-list' });
        const recent = h('section', { class: 'wl-card wl-recent' }, h('h2', {}, t('welcome.recent')), this.recentEl);
        this.renderRecent();

        const learnGrid = h('div', { class: 'wl-learn-grid' });
        for (const c of LEARN) {
            const card = h('button', { class: `wl-learn ${c.color}`, type: 'button' },
                h('span', { class: 'wl-learn-icon' }, icon(c.icon)),
                h('span', { class: 'wl-learn-title' }, t(`${c.key}.title`)),
                h('span', { class: 'wl-learn-text' }, t(`${c.key}.text`)));
            card.addEventListener('click', () => this.openExample(c.id));
            learnGrid.appendChild(card);
        }
        const learn = h('section', { class: 'wl-card wl-learn-card' }, h('h2', {}, t('welcome.learn')), learnGrid);

        const chips = h('div', { class: 'wl-filters', role: 'tablist', 'aria-label': t('welcome.examples') });
        this.galleryEl = h('div', { class: 'wl-gallery' });
        for (const [id, key] of FILTERS) {
            const b = h('button', { class: 'wl-filter', type: 'button', role: 'tab', 'aria-selected': String(this.filter === id) }, t(key));
            b.addEventListener('click', () => { this.filter = id; for (const x of chips.children) x.setAttribute('aria-selected', String(x === b)); this.renderGallery(); });
            chips.appendChild(b);
        }
        const examples = h('section', { class: 'wl-examples' }, h('div', { class: 'wl-section-head' }, h('h2', {}, t('welcome.examples')), chips), this.galleryEl);
        this.renderGallery();

        const keys = h('div', { class: 'wl-keys-grid' });
        for (const id of ['workbench.commandPalette', 'workbench.quickOpen', 'run.run', 'debug.start', 'debug.toggleBreakpoint', 'debug.stepOver', 'debug.stepBack', 'editor.definition', 'editor.rename', 'editor.codeActions', 'workbench.togglePanel', 'editor.format']) {
            if (!app.commands.get(id)) continue;
            const k = app.commands.keysFor(id)[0];
            if (!k) continue;
            keys.append(h('div', { class: 'wl-key' }, h('span', {}, app.commands.title(id).replace(/^[^:]+: /, '')), h('kbd', {}, formatKeys(k))));
        }
        const keysSec = h('section', { class: 'wl-shortcuts' }, h('h2', {}, t('welcome.shortcuts')), keys);

        const onStart = h('input', { type: 'checkbox', checked: Boolean(app.settings.get('welcomeOnStart')) });
        onStart.addEventListener('change', () => app.settings.set('welcomeOnStart', onStart.checked));
        const foot = h('footer', { class: 'wl-foot' }, h('label', { class: 'switch' }, onStart, h('span', { class: 'track' }), t('welcome.showOnStart')),
            h('span', { class: 'muted' }, t('welcome.footer')));

        this.el.appendChild(h('div', { class: 'wl-inner' }, hero, h('div', { class: 'wl-grid' }, start, recent, learn), examples, keysSec, foot));
    }

    renderRecent() {
        if (!this.recentEl) return;
        clear(this.recentEl);
        const list = load('recent', []).slice(0, 5);
        const app = this.app;
        if (!list.length) { this.recentEl.appendChild(h('p', { class: 'muted' }, t('welcome.noRecent'))); return; }
        for (const r of list) {
            const current = r.key === app.projectKey;
            const cmd = r.web ? '' : `til ide ${/\s/.test(r.root || '') ? JSON.stringify(r.root) : r.root || ''}`;
            const row = h('div', { class: ['wl-recent-item', current && 'current'] },
                icon(r.web ? 'globe' : 'folder', { className: 'wl-recent-icon' }),
                h('div', { class: 'wl-recent-text' }, h('div', { class: 'wl-recent-name' }, r.name || r.root, current ? h('span', { class: 'chip accent' }, t('welcome.current')) : null),
                    h('div', { class: 'wl-recent-path' }, r.web ? t('welcome.browserProject') : r.root || '')));
            if (!current && cmd) {
                const copy = h('button', { class: 'icon-btn small', type: 'button', title: cmd, 'aria-label': t('welcome.copyOpen', { cmd }) }, icon('copy'));
                copy.addEventListener('click', () => { app.copyText(cmd); });
                row.appendChild(copy);
            }
            this.recentEl.appendChild(row);
        }
    }

    renderGallery() {
        clear(this.galleryEl);
        const pred = (FILTERS.find((f) => f[0] === this.filter) || FILTERS[0])[2];
        for (const ex of EXAMPLES.filter(pred)) {
            if (ex.id === 'math_utils') continue;
            const preview = h('pre', { class: 'wl-preview', 'aria-hidden': 'true' });
            highlightInto(preview, previewLines(ex.source));
            const card = h('button', { class: 'wl-example', type: 'button' },
                preview,
                h('div', { class: 'wl-ex-meta' },
                    h('div', { class: 'wl-ex-title' }, t(`example.${ex.id}.title`)),
                    h('div', { class: 'wl-ex-desc' }, t(`example.${ex.id}.desc`)),
                    h('div', { class: 'wl-ex-tags' }, ...ex.tags.slice(0, 3).map((tag) => h('span', { class: `chip ${tagColor(tag)}` }, t(`tag.${tag}`))))));
            card.addEventListener('click', () => this.openExample(ex.id));
            this.galleryEl.appendChild(card);
        }
    }

    async openExample(id) {
        const app = this.app;
        const ex = EXAMPLES.find((e) => e.id === id);
        if (!ex) return;
        // a copy in the project (web face: the starter project; desktop: the server's examples)
        const candidates = [];
        if (app.web) candidates.push(`examples/${ex.file}`);
        for (const e of app.info.examples || []) {
            const name = String(e.name || '').replace(/\.til$/, '');
            const file = String(e.path || '').split('/').pop();
            if ((file === ex.file || name === ex.id) && e.path && !/^\/|^[A-Za-z]:/.test(e.path)) candidates.push(e.path);
        }
        for (const p of candidates) {
            const doc = await app.openFile(p).catch(() => null);
            if (doc) return;
        }
        await app.openExample(ex);
    }
}

function previewLines(src) {
    const lines = src.split('\n');
    // skip the leading comment block, keep a short, representative piece
    let i = 0;
    while (i < lines.length - 1 && (/^\s*#(?!\[)/.test(lines[i]) || !lines[i].trim())) i++;
    return lines.slice(i, i + 9).join('\n');
}

function tagColor(tag) {
    return { screen: 'purple', keys: 'purple', kk: 'accent', ru: 'accent', 'kk-latn': 'accent', proofs: 'green', contracts: 'green', effects: 'orange', energy: 'yellow', input: 'yellow' }[tag] || '';
}
