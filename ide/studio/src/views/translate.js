// The translate view: the current file and its translation side by side
// (POST /api/tools/translate), read-only, scrolled together; "Translate file"
// replaces the file's text (undoable).
import { EditorState } from '@codemirror/state';
import { EditorView, lineNumbers } from '@codemirror/view';
import { syntaxHighlighting } from '@codemirror/language';
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, L, bindAttr } from '../core/i18n.js';
import { basename } from '../core/paths.js';
import { tilLanguage, tilHighlighter } from '../editor/til_lang.js';
import { studioTheme } from '../editor/theme.js';

const LANGS = ['en', 'kk', 'kk-latn', 'ru'];

function readOnlyView(parent, text) {
    return new EditorView({
        parent,
        state: EditorState.create({
            doc: text,
            extensions: [tilLanguage(), syntaxHighlighting(tilHighlighter), studioTheme, lineNumbers(), EditorState.readOnly.of(true), EditorView.editable.of(false)],
        }),
    });
}

export class TranslateView {
    constructor(app, path, lang) {
        this.app = app;
        this.path = path;
        this.lang = lang;
        this.iconName = 'translate';
        this.left = h('div', { class: 'tr-pane' });
        this.right = h('div', { class: 'tr-pane' });
        this.leftHead = h('div', { class: 'tr-pane-head' });
        this.rightHead = h('div', { class: 'tr-pane-head' });
        this.seg = bindAttr(h('div', { class: 'segmented', role: 'radiogroup' }), 'aria-label', 'translate.to');
        this.applyBtn = h('button', { class: 'btn primary small', type: 'button' }, icon('translate'), L('translate.replace'));
        this.applyBtn.addEventListener('click', () => this.apply());
        const refresh = h('button', { class: 'icon-btn small', type: 'button' }, icon('refresh'));
        bindAttr(refresh, 'aria-label', 'translate.refresh');
        bindAttr(refresh, 'title', 'translate.refresh');
        refresh.addEventListener('click', () => this.load());
        this.status = h('span', { class: 'tr-status muted' });
        this.el = h('div', { class: 'translate-view' },
            h('div', { class: 'tr-bar' }, icon('translate', { className: 'tr-icon' }), h('span', { class: 'tr-title' }, L('translate.title', { name: basename(path || '') })),
                this.seg, this.status, h('span', { class: 'spacer' }), refresh, this.applyBtn),
            h('div', { class: 'tr-split' }, h('div', { class: 'tr-col' }, this.leftHead, this.left), h('div', { class: 'tr-col' }, this.rightHead, this.right)));
        this.renderSeg();
        this.syncing = false;
    }

    title() { return `${basename(this.path || '')} → ${t(`kwlang.short.${this.lang}`)}`; }

    renderSeg() {
        clear(this.seg);
        for (const l of LANGS) {
            const b = h('button', { type: 'button', role: 'radio', 'aria-checked': String(l === this.lang) }, t(`kwlang.${l}`));
            b.addEventListener('click', () => { this.lang = l; this.renderSeg(); this.load(); this.app.editors.renderAllTabs(); });
            this.seg.appendChild(b);
        }
    }

    async load() {
        const doc = this.path ? (this.app.docs.get(this.path) || await this.app.docs.open(this.path).catch(() => null)) : this.app.activeTilDoc();
        if (!doc) return;
        this.doc = doc;
        this.applyBtn.disabled = Boolean(doc.readOnly);      // a TIL example: view only
        const source = doc.current().doc.toString();
        this.heads();
        this.status.textContent = t('translate.working');
        let text = '';
        try {
            const res = await this.app.backend.translate({ path: doc.kind === 'file' ? doc.path : undefined, source, to: this.lang });
            text = res && typeof res.text === 'string' ? res.text : '';
            this.status.textContent = '';
        } catch (e) {
            if (e && e.notAvailable) { this.app.showDesktopOnly(e.feature); this.status.textContent = ''; return; }
            this.status.textContent = t('translate.failed');
            text = '';
        }
        this.translated = text;
        this.mount(source, text);
    }

    mount(source, text) {
        if (this.lv) this.lv.destroy();
        if (this.rv) this.rv.destroy();
        clear(this.left);
        clear(this.right);
        this.lv = readOnlyView(this.left, source);
        this.rv = readOnlyView(this.right, text);
        const link = (a, b) => a.scrollDOM.addEventListener('scroll', () => {
            if (this.syncing) return;
            this.syncing = true;
            b.scrollDOM.scrollTop = a.scrollDOM.scrollTop;
            requestAnimationFrame(() => { this.syncing = false; });
        });
        link(this.lv, this.rv);
        link(this.rv, this.lv);
    }

    heads() {
        if (!this.doc) return;
        this.leftHead.textContent = `${this.doc.name} · ${t('translate.original')}`;
        this.rightHead.textContent = `${t(`kwlang.${this.lang}`)} · ${t('translate.preview')}`;
    }

    relocalize() {
        this.renderSeg();
        this.heads();
    }

    async apply() {
        if (!this.doc || !this.translated || this.doc.readOnly) return;
        this.doc.replaceText(this.translated);
        this.app.editors.show(this.doc);
        this.app.statusbar.flash(t('translate.done', { lang: t(`kwlang.${this.lang}`) }));
    }

    focus() { this.applyBtn.focus(); }

    dispose() {
        if (this.lv) this.lv.destroy();
        if (this.rv) this.rv.destroy();
    }

    onShow() {
        requestAnimationFrame(() => { if (this.lv) this.lv.requestMeasure(); if (this.rv) this.rv.requestMeasure(); });
    }
}
