// Settings: theme, font size, UI language, keyword language for new files,
// editor options, and the keyboard shortcuts (click a binding, press keys).
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, L, LOCALES, LOCALE_NAMES, getLocale } from '../core/i18n.js';
import { formatKeys, eventKeys } from '../core/commands.js';
import { tip, confirmDialog } from '../ui/widgets.js';
import { DEFAULT_SETTINGS } from '../core/store.js';

export class SettingsView {
    constructor(app) {
        this.app = app;
        this.body = h('div', { class: 'side-body pad settings-body' });
        this.titleEl = L('activity.settings');
        this.titleEl.className = 'side-title';
        this.el = h('div', { class: 'side-view' }, h('div', { class: 'side-head' }, this.titleEl), this.body);
        this.filter = '';
        this.build();
        app.commands.on('keymap', () => this.renderKeys());
    }

    focus() { const f = this.body.querySelector('button, input, select'); if (f) f.focus(); }

    build() {
        const s = this.app.settings;
        clear(this.body);
        const group = (titleKey, ...rows) => h('div', { class: 'set-group' }, h('div', { class: 'set-group-title' }, t(titleKey)), ...rows);
        const row = (labelKey, control, hintKey) => h('div', { class: ['set-row', !control.classList.contains('switch') && 'stack'] },
            h('div', { class: 'set-label' }, h('div', {}, t(labelKey)), hintKey ? h('div', { class: 'set-hint' }, t(hintKey)) : null), control);
        const sw = (key) => {
            const box = h('input', { type: 'checkbox', checked: Boolean(s.get(key)), 'aria-label': t(`settings.${key}`) });
            box.addEventListener('change', () => s.set(key, box.checked));
            this.controls[key] = { sync: () => { box.checked = Boolean(s.get(key)); } };
            return h('label', { class: 'switch' }, box, h('span', { class: 'track' }));
        };
        const select = (key, options) => {
            const sel = h('select', { class: 'select', 'aria-label': t(`settings.${key}`) }, ...options.map(([v, label]) => h('option', { value: v }, label)));
            sel.value = String(s.get(key));
            sel.addEventListener('change', () => s.set(key, sel.value));
            this.controls[key] = { sync: () => { sel.value = String(s.get(key)); } };
            return sel;
        };
        this.controls = {};
        const themeSeg = h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': t('settings.theme') });
        const renderTheme = () => {
            clear(themeSeg);
            for (const [v, k, ic] of [['dark', 'settings.themeDark', 'moon'], ['light', 'settings.themeLight', 'sun'], ['system', 'settings.themeSystem', 'screen']]) {
                const b = h('button', { type: 'button', role: 'radio', 'aria-checked': String(s.get('theme') === v) }, t(k));
                b.addEventListener('click', () => s.set('theme', v));
                themeSeg.appendChild(b);
                void ic;
            }
        };
        renderTheme();
        this.controls.theme = { sync: renderTheme };
        const size = h('input', { class: 'range', type: 'range', min: '11', max: '22', step: '1', value: String(s.get('fontSize')), 'aria-label': t('settings.fontSize') });
        const sizeVal = h('span', { class: 'range-value' }, `${s.get('fontSize')} px`);
        size.addEventListener('input', () => { sizeVal.textContent = `${size.value} px`; s.set('fontSize', Number(size.value)); });
        this.controls.fontSize = { sync: () => { size.value = String(s.get('fontSize')); sizeVal.textContent = `${s.get('fontSize')} px`; } };
        const uiLang = h('select', { class: 'select', 'aria-label': t('settings.uiLang') }, ...LOCALES.map((l) => h('option', { value: l }, LOCALE_NAMES[l])));
        uiLang.value = getLocale();
        uiLang.addEventListener('change', () => s.set('uiLang', uiLang.value));
        this.controls.uiLang = { sync: () => { uiLang.value = getLocale(); } };

        this.keysList = h('div', { class: 'keys-list' });
        this.keysFilter = h('input', { class: 'input small-input', type: 'search', placeholder: t('settings.keysFilter'), 'aria-label': t('settings.keysFilter') });
        this.keysFilter.addEventListener('input', () => { this.filter = this.keysFilter.value.toLowerCase(); this.renderKeys(); });
        const reset = h('button', { class: 'btn ghost small', type: 'button' }, icon('restart'), t('settings.keysReset'));
        reset.addEventListener('click', async () => {
            if (await confirmDialog({ title: t('settings.keysReset'), message: t('settings.keysResetMsg'), confirm: t('settings.keysReset') })) this.app.commands.resetBindings();
        });
        const resetAll = h('button', { class: 'btn ghost small', type: 'button' }, icon('restart'), t('settings.resetAll'));
        resetAll.addEventListener('click', async () => {
            if (!(await confirmDialog({ title: t('settings.resetAll'), message: t('settings.resetAllMsg'), confirm: t('settings.resetAll'), danger: true }))) return;
            for (const [k, v] of Object.entries(DEFAULT_SETTINGS)) s.set(k, v);
        });

        this.body.append(
            group('settings.appearance',
                row('settings.theme', themeSeg),
                row('settings.fontSize', h('div', { class: 'range-row' }, size, sizeVal)),
                row('settings.uiLang', uiLang, 'settings.uiLangHint')),
            group('settings.editor',
                row('settings.kwLang', select('kwLang', [['en', t('kwlang.en')], ['kk', t('kwlang.kk')], ['kk-latn', t('kwlang.kk-latn')], ['ru', t('kwlang.ru')]]), 'settings.kwLangHint'),
                row('settings.errorLens', sw('errorLens'), 'settings.errorLensHint'),
                row('settings.inlayHints', sw('inlayHints')),
                row('settings.codeLens', sw('codeLens')),
                row('settings.wordWrap', sw('wordWrap')),
                row('settings.formatOnSave', sw('formatOnSave')),
                row('settings.autoSave', select('autoSave', [['off', t('settings.autoSaveOff')], ['delay', t('settings.autoSaveDelay')], ['focus', t('settings.autoSaveFocus')]]))),
            group('settings.workbench',
                row('settings.welcomeOnStart', sw('welcomeOnStart')),
                row('settings.calculatorBezel', sw('calculatorBezel'), 'settings.calculatorBezelHint')),
            group('settings.keys', h('div', { class: 'row' }, this.keysFilter, reset), this.keysList),
            h('div', { class: 'set-footer' }, resetAll));
        this.renderKeys();
    }

    sync() {
        for (const c of Object.values(this.controls || {})) c.sync();
    }

    renderKeys() {
        if (!this.keysList) return;
        clear(this.keysList);
        const cmds = [...this.app.commands.list.values()].filter((c) => c.palette !== false && c.id !== 'settings.recordKey');
        cmds.sort((a, b) => this.app.commands.title(a.id).localeCompare(this.app.commands.title(b.id)));
        for (const c of cmds) {
            const title = this.app.commands.title(c.id);
            const keys = this.app.commands.keysFor(c.id);
            if (this.filter && !`${title} ${keys.join(' ')}`.toLowerCase().includes(this.filter)) continue;
            const btn = h('button', { class: 'key-btn', type: 'button', 'aria-label': t('settings.keysEdit', { name: title }) },
                keys.length ? formatKeys(keys[0]) : h('span', { class: 'muted' }, '—'));
            tip(btn, 'settings.keysEditTip');
            btn.addEventListener('click', () => this.record(c.id, btn));
            this.keysList.appendChild(h('div', { class: 'key-row' }, h('span', { class: 'key-cmd' }, title), btn));
        }
    }

    record(id, btn) {
        btn.classList.add('recording');
        btn.textContent = t('settings.keysPress');
        const onKey = (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.key === 'Escape') { done(); return; }
            if (e.key === 'Backspace' || e.key === 'Delete') { this.app.commands.setBinding(id, ''); done(); return; }
            const k = eventKeys(e);
            if (!k || /^(Mod|Ctrl|Alt|Shift|Meta)(\+(Mod|Ctrl|Alt|Shift|Meta))*$/.test(k)) return;
            this.app.commands.setBinding(id, k);
            done();
        };
        const done = () => {
            window.removeEventListener('keydown', onKey, true);
            btn.classList.remove('recording');
            this.renderKeys();
        };
        window.addEventListener('keydown', onKey, true);
        btn.addEventListener('blur', done, { once: true });
    }

    relocalize() { this.build(); }
}
