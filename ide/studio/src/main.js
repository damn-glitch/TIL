// TIL Studio entry point: find the backend (the desktop server's /api/info,
// else the browser backend of the web face) and start the app.
import { DesktopBackend, takeToken } from './backend/api.js';
import { App } from './app.js';
import { h } from './core/dom.js';
import { logoMark } from './core/icons.js';
import { t, setLocale, detectLocale } from './core/i18n.js';
import { settings } from './core/store.js';

async function boot() {
    const mount = document.getElementById('app');
    setLocale(settings.get('uiLang') || detectLocale());
    mount.appendChild(h('div', { class: 'boot' }, logoMark(56), h('div', { class: 'boot-bar' }, h('span'))));
    const token = takeToken();
    const desktop = new DesktopBackend({ token });
    const info = location.protocol === 'file:' ? null : await desktop.connect();
    let backend, appInfo;
    if (info && info.forbidden) {
        mount.replaceChildren(h('div', { class: 'boot-error' }, logoMark(48), h('h1', {}, t('boot.forbiddenTitle')), h('p', {}, t('boot.forbiddenText')), h('code', {}, 'til ide')));
        return;
    }
    if (info) {
        backend = desktop;
        appInfo = info;
    } else {
        const { BrowserBackend } = await import('./backend/browser.js');
        backend = new BrowserBackend();
        try {
            appInfo = await backend.connect();
        } catch (e) {
            mount.replaceChildren(h('div', { class: 'boot-error' }, logoMark(48), h('h1', {}, t('boot.webFailedTitle')), h('p', {}, String(e && e.message ? e.message : e))));
            return;
        }
    }
    const app = new App({ backend, info: appInfo });
    window.tilStudio = app;
    try {
        await app.start(mount);
    } catch (e) {
        console.error(e);
        mount.replaceChildren(h('div', { class: 'boot-error' }, logoMark(48), h('h1', {}, t('boot.failedTitle')), h('pre', {}, String(e && e.stack ? e.stack : e))));
    }
}

window.addEventListener('error', (e) => { if (window.__studioErrors) window.__studioErrors.push(String(e.message)); });
window.__studioErrors = [];
boot();
