// UI strings in English, Russian and Kazakh (i18n/*.json), switchable live.
//
//   t('menu.file')                    -> "File" / "Файл" / "Файл"
//   t('run.exited', {code: 0})        -> "{code}" placeholders
//   tn('problems.count', 3)           -> plural forms: key.one / key.few / key.many / key.other
//   L('menu.file')                    -> a <span> whose text follows the language
//   bindAttr(el, 'title', 'run.tip')  -> an attribute that follows the language
import en from '../../i18n/en.json';
import ru from '../../i18n/ru.json';
import kk from '../../i18n/kk.json';
import { Emitter } from './events.js';

export const LOCALES = ['en', 'ru', 'kk'];
export const LOCALE_NAMES = { en: 'English', ru: 'Русский', kk: 'Қазақша' };
const CATALOGS = { en, ru, kk };
const events = new Emitter();
let locale = 'en';
let rules = new Intl.PluralRules('en');

export function detectLocale() {
    const langs = navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language || 'en'];
    for (const l of langs) {
        const code = String(l).toLowerCase().split('-')[0];
        if (CATALOGS[code]) return code;
    }
    return 'en';
}

export function getLocale() { return locale; }

export function setLocale(next) {
    if (!CATALOGS[next] || next === locale) return;
    locale = next;
    try { rules = new Intl.PluralRules(next); } catch (e) { rules = new Intl.PluralRules('en'); }
    document.documentElement.lang = next;
    applyI18n(document);
    events.emit('change', next);
}

export function onLocale(fn) { return events.on('change', fn); }

function fill(text, params) {
    if (!params) return text;
    return text.replace(/\{(\w+)\}/g, (m, k) => (params[k] !== undefined && params[k] !== null ? String(params[k]) : m));
}

export function has(key) { return Object.prototype.hasOwnProperty.call(CATALOGS.en, key); }

export function t(key, params) {
    const cat = CATALOGS[locale];
    let text = cat[key];
    if (text === undefined) text = CATALOGS.en[key];
    if (text === undefined) {
        if (typeof STUDIO_DEV !== 'undefined' && STUDIO_DEV) console.warn('[i18n] missing key', key);
        return key;
    }
    return fill(text, params);
}

export function tn(key, n, params) {
    let cat = 'other';
    try { cat = rules.select(n); } catch (e) { /* keep other */ }
    const full = `${key}.${cat}`;
    const p = Object.assign({ n }, params);
    if (CATALOGS[locale][full] !== undefined || CATALOGS.en[full] !== undefined) return t(full, p);
    return t(`${key}.other`, p);
}

/* a span whose text follows the UI language */
export function L(key, params, tag = 'span') {
    const el = document.createElement(tag);
    setText(el, key, params);
    return el;
}

export function setText(el, key, params) {
    el.dataset.i18n = key;
    if (params) el.dataset.i18nP = JSON.stringify(params); else delete el.dataset.i18nP;
    el.textContent = t(key, params);
    return el;
}

/* an attribute (title, aria-label, placeholder) that follows the UI language */
export function bindAttr(el, attr, key, params) {
    el.dataset['i18nA' + attr.replace(/(^|-)(\w)/g, (m, a, c) => c.toUpperCase())] = params ? `${key}|${JSON.stringify(params)}` : key;
    el.setAttribute(attr, t(key, params));
    return el;
}

const ATTRS = ['title', 'aria-label', 'placeholder', 'aria-description'];

export function applyI18n(root) {
    for (const el of root.querySelectorAll('[data-i18n]')) {
        let params = null;
        if (el.dataset.i18nP) { try { params = JSON.parse(el.dataset.i18nP); } catch (e) { params = null; } }
        el.textContent = t(el.dataset.i18n, params);
    }
    for (const attr of ATTRS) {
        const prop = 'i18nA' + attr.replace(/(^|-)(\w)/g, (m, a, c) => c.toUpperCase());
        const sel = '[data-i18n-a-' + attr + ']';
        for (const el of root.querySelectorAll(sel)) {
            const raw = el.dataset[prop];
            const bar = raw.indexOf('|');
            let key = raw, params = null;
            if (bar > 0) { key = raw.slice(0, bar); try { params = JSON.parse(raw.slice(bar + 1)); } catch (e) { params = null; } }
            el.setAttribute(attr, t(key, params));
        }
    }
}

export function catalog(lang) { return CATALOGS[lang] || CATALOGS.en; }
