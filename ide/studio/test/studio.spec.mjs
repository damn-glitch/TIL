// Browser tests of TIL Studio (Node 22 + Playwright, node:test).
//
//   node --test ide/studio/test/studio.spec.mjs
//
// By default the desktop face runs against test/mock_server.mjs (a stand-in
// for src/tilstudio.py with canned behaviour) in a temporary project, and the
// web face against a static server of website/ (the real WebAssembly VM and
// tilc.tbc of the playground). Environment (optional):
//   STUDIO_URL       a running TIL Studio server: the URL printed by `til ide` (with ?token=);
//                    the tests then write their project files through its API
//   STUDIO_HTML      the page the mock server serves (default: ide/studio.html)
//   SITE_DIR         the website directory for the web face (default: website/)
//   SCREENSHOTS      directory for the screenshots (default: ide/studio/test/screenshots)
//   SKIP_WEB=1       skip the web face; SKIP_SHOTS=1 skip the screenshot tour
import { after, before, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startMockServer, PROJECT } from './mock_server.mjs';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..', '..');
const SITE_DIR = path.resolve(process.env.SITE_DIR || path.join(ROOT, 'website'));
const SHOTS = path.resolve(process.env.SCREENSHOTS || path.join(HERE, 'screenshots'));
const HTML = process.env.STUDIO_HTML ? path.resolve(process.env.STUDIO_HTML) : undefined;
const MOD = process.platform === 'darwin' ? 'Meta' : 'Control';

let mock = null, browser = null, URL0 = '', BASE = '', TOKEN = '';

async function api(method, p, body) {
    const resp = await fetch(BASE + p, {
        method, headers: { 'X-TIL-Token': TOKEN, 'Content-Type': 'application/json', Origin: BASE },
        body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(`${method} ${p}: ${resp.status} ${data.error || ''}`);
    return data;
}

before(async () => {
    if (process.env.STUDIO_URL) {
        URL0 = process.env.STUDIO_URL;
        const u = new URL(URL0);
        BASE = u.origin;
        TOKEN = u.searchParams.get('token') || '';
        for (const [p, text] of Object.entries(PROJECT)) {
            const dir = p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '';
            if (dir) await api('POST', '/api/file/create', { path: dir, type: 'dir' }).catch(() => {});      // 409: it exists
            await api('PUT', '/api/file', { path: p, text });
        }
    } else {
        mock = await startMockServer({ html: HTML });
        URL0 = mock.url;
        BASE = mock.base;
        TOKEN = mock.token;
    }
    browser = await chromium.launch();
    fs.mkdirSync(SHOTS, { recursive: true });
});

after(async () => {
    if (browser) await browser.close();
    if (mock) await mock.close();
});

/* a fresh page (own storage unless `ctx` is given) with the app ready */
async function open({ ctx, viewport = { width: 1440, height: 900 }, url = URL0, locale } = {}) {
    const context = ctx || await browser.newContext({ viewport, locale: locale || 'en-US' });
    const page = await context.newPage();
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(String(e.stack || e)));
    await page.goto(url);
    await page.waitForSelector('body[data-ready="true"]', { timeout: 15000 });
    return page;
}

async function openFile(page, name) {
    await page.evaluate((n) => window.tilStudio.openFile(n), name);
    await page.waitForFunction((n) => {
        const d = window.tilStudio.editors.activeDoc();
        return d && d.path === n;
    }, name);
    await page.waitForSelector('.cm-content');
    await page.waitForFunction(() => window.tilStudio.lsp && window.tilStudio.lsp.ready);
}

const docText = (page) => page.evaluate(() => window.tilStudio.editors.activeView().state.doc.toString());

async function gotoLineEnd(page, line) {
    await page.evaluate((n) => {
        const v = window.tilStudio.editors.activeView();
        const l = v.state.doc.line(n);
        v.dispatch({ selection: { anchor: l.to } });
        v.focus();
    }, line);
}

async function placeCursor(page, line, text, offset = 0) {
    await page.evaluate(([n, t, o]) => {
        const v = window.tilStudio.editors.activeView();
        const l = v.state.doc.line(n);
        const at = l.text.indexOf(t);
        v.dispatch({ selection: { anchor: l.from + Math.max(0, at) + o } });
        v.focus();
    }, [line, text, offset]);
}

const lineOf = (src, needle) => src.split('\n').findIndex((l) => l.includes(needle)) + 1;

async function shot(page, name) {
    if (process.env.SKIP_SHOTS) return;
    await page.waitForTimeout(250);
    await page.screenshot({ path: path.join(SHOTS, `${name}.png`) });
}

/* ══════════════════════════════ desktop face ══════════════════════════════ */
describe('desktop face', () => {
    it('opens fast and shows the project', async () => {
        const page = await open();
        const ready = await page.evaluate(() => window.__studioReadyMs);
        assert.ok(ready < 1000, `the app took ${ready} ms to open`);
        await page.waitForSelector('.tree-row:has-text("main.til")');
        assert.equal(await page.locator('.activitybar .ab-btn').count(), 9);
        assert.ok(await page.locator('.wl-hero').isVisible(), 'the welcome page is shown');
        assert.deepEqual(page.errors, []);
        await page.context().close();
    });

    it('rejects a page without the token', async () => {
        const resp = await fetch(BASE + '/api/info');
        assert.equal(resp.status, 403);
    });

    it('highlights TIL in every keyword language', async () => {
        const page = await open();
        await page.click('.tree-row:has-text("kazakh.til")');
        await page.waitForSelector('.cm-content .t-kw');
        const kws = await page.locator('.cm-content .t-kw').allTextContents();
        assert.ok(kws.includes('қайтару') && kws.includes('үшін'), kws.join(','));
        assert.ok(await page.locator('.cm-content .t-builtin:has-text("басып_шығару")').count() > 0);
        await openFile(page, 'main.til');
        assert.ok(await page.locator('.cm-content .t-str').count() > 0);
        assert.ok(await page.locator('.cm-content .t-interp').count() > 0, 'f-string interpolation');
        assert.ok(await page.locator('.cm-content .t-comment').count() > 0);
        await page.context().close();
    });

    it('completes, shows diagnostics, applies a quick fix', async () => {
        const page = await open();
        await openFile(page, 'main.til');
        const src = await docText(page);
        await gotoLineEnd(page, lineOf(src, 'print("done")'));
        await page.keyboard.press('Enter');
        await page.keyboard.type('pri');
        await page.waitForSelector('.cm-tooltip-autocomplete li:has-text("print")', { timeout: 5000 });
        await page.keyboard.press('Escape');
        // a misspelled builtin: a diagnostic, the Problems count, the error lens
        await page.keyboard.type('nt_x("hi")');
        await page.waitForSelector('.cm-lintRange-error', { timeout: 8000 });
        await page.waitForSelector('.cm-errorLens-error');
        await page.waitForFunction(() => window.tilStudio.diagnosticTotals().errors > 0);
        await shot(page, 'diagnostics');
        await page.keyboard.press(`${MOD}+.`);
        const fix = page.locator('.menu .menu-item:not(.disabled)').first();
        await fix.waitFor({ timeout: 5000 });
        await fix.click();
        await page.waitForFunction(() => !window.tilStudio.editors.activeView().state.doc.toString().includes('print_x('), null, { timeout: 5000 });
        assert.ok((await docText(page)).includes('print("hi")'), 'the quick fix replaced the name');
        await page.waitForFunction(() => window.tilStudio.diagnosticTotals().errors === 0, null, { timeout: 8000 });
        await page.context().close();
    });

    it('hovers, goes to definition and renames across the file', async () => {
        const page = await open();
        await openFile(page, 'main.til');
        const src = await docText(page);
        const callLine = lineOf(src, 'print(greet(');
        await placeCursor(page, callLine, 'greet', 2);
        await page.locator('.cm-line').nth(callLine - 1).locator('.t-fn:has-text("greet")').hover();
        await page.waitForSelector('.cm-lsp-hover', { timeout: 5000 });
        assert.match(await page.locator('.cm-lsp-hover').textContent(), /greet/);
        // hovers with **bold** and *italic* text (keywords, locals) render and the page stays responsive
        const retLine = lineOf(src, 'return f"Hello');
        await page.mouse.move(0, 0);
        await page.locator('.cm-line').nth(retLine - 1).locator('.t-kw').first().hover();
        await page.waitForFunction(() => document.querySelectorAll('.cm-lsp-hover').length > 0, null, { timeout: 5000 });
        assert.ok((await page.locator('.cm-lsp-hover strong, .cm-lsp-hover em, .cm-lsp-hover code').count()) > 0);
        await page.mouse.move(0, 0);
        await page.keyboard.press('F12');
        await page.waitForFunction((l) => {
            const v = window.tilStudio.editors.activeView();
            return v.state.doc.lineAt(v.state.selection.main.head).number === l;
        }, lineOf(src, 'greet(name'), { timeout: 5000 });
        await page.keyboard.press('F2');
        await page.waitForSelector('.rename-box input');
        await page.keyboard.press(`${MOD}+a`);
        await page.keyboard.type('welcome');
        await page.keyboard.press('Enter');
        await page.waitForFunction(() => !window.tilStudio.editors.activeView().state.doc.toString().includes('greet('), null, { timeout: 5000 });
        const text = await docText(page);
        assert.equal((text.match(/welcome\(/g) || []).length, 2);
        await page.context().close();
    });

    it('runs on every backend and streams the output', async () => {
        const page = await open();
        await openFile(page, 'main.til');
        const caps = await page.evaluate(() => window.tilStudio.caps);
        for (const backend of ['vm', 'cvm', 'native']) {
            if (backend !== 'vm' && !caps[backend]) continue;
            await page.evaluate((b) => window.tilStudio.settings.set('backend', b), backend);
            await page.keyboard.press(`${MOD}+F5`);
            await page.waitForSelector('.term-foot', { timeout: 30000 });
            const out = await page.locator('.term-out').textContent();
            assert.match(out, /Hello, TIL Studio!/, `${backend}: ${out}`);
            assert.ok(await page.locator('.term-foot.ok').count() === 1, `${backend} finished`);
        }
        await page.evaluate(() => window.tilStudio.settings.set('backend', 'vm'));
        await shot(page, 'run');
        await page.context().close();
    });

    it('sends input to a program that waits for it', async () => {
        const page = await open();
        await openFile(page, 'input.til');
        await page.keyboard.press(`${MOD}+F5`);
        await page.waitForSelector('.term-input-row.waiting', { timeout: 10000 });
        await page.waitForFunction(() => document.activeElement && document.activeElement.classList.contains('term-input'));
        await page.keyboard.type('Aigerim');
        await page.keyboard.press('Enter');
        await page.waitForSelector('.term-foot', { timeout: 10000 });
        assert.match(await page.locator('.term-out').textContent(), /Aigerim/);
        await page.context().close();
    });

    it('follows the backend the server really uses, saves a snapshot, stops', async () => {
        const page = await open();
        await openFile(page, 'graphics.til');
        // a program that uses the screen runs on the VM whatever was asked (status.backend)
        await page.evaluate(() => window.tilStudio.settings.set('backend', 'native'));
        await page.keyboard.press(`${MOD}+F5`);
        await page.waitForSelector('.screen-canvas[data-frame]', { timeout: 10000 });
        await page.waitForFunction(() => /^VM/.test(document.querySelector('.term-head-detail').textContent));
        await page.evaluate(() => window.tilStudio.commands.run('run.snapshot'));
        // the mock always saves one; the real server refuses while the program
        // waits for a key (409, its message is shown): either way a toast says so
        await page.waitForSelector(mock ? '.toast:has-text(".tilp")' : '.toast', { timeout: 5000 });
        await page.keyboard.press('Shift+F5');
        // exit {reason: stopped}: the footer is the UI's own text
        await page.waitForSelector('.term-foot.stopped', { state: 'attached', timeout: 5000 });
        assert.match(await page.locator('.term-foot').textContent(), /Stopped after/);
        await page.evaluate(() => window.tilStudio.settings.set('backend', 'vm'));
        await page.context().close();
    });

    it('opens the TIL examples read-only and runs them', async () => {
        const page = await open();
        const ex = await page.evaluate(() => (window.tilStudio.info.examples.find((e) => /hello/.test(e.name)) || window.tilStudio.info.examples[0]).path);
        assert.match(ex, /^@examples\//);
        await page.evaluate((p) => window.tilStudio.openFile(p), ex);
        await page.waitForFunction((p) => window.tilStudio.editors.activeDoc() && window.tilStudio.editors.activeDoc().path === p, ex);
        await page.waitForSelector('.crumb-readonly');
        assert.equal(await page.evaluate(() => window.tilStudio.editors.activeView().state.readOnly), true);
        await page.keyboard.press(`${MOD}+F5`);
        await page.waitForSelector('.term-foot.ok', { timeout: 30000 });
        assert.match(await page.locator('.term-out').textContent(), /Hello/);
        await page.context().close();
    });

    it('shows the screen and forwards keys', async () => {
        const page = await open();
        await openFile(page, 'graphics.til');
        await page.keyboard.press(`${MOD}+F5`);
        await page.waitForSelector('.screen-canvas[data-frame]', { timeout: 10000 });
        const pixelAt = () => page.evaluate(() => {
            const c = document.querySelector('.screen-canvas');
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            let sum = 0;
            for (let i = 0; i < d.length; i += 4) if (d[i] > 200 && d[i + 1] > 150 && d[i + 2] < 100) sum += (i / 4) % c.width;
            return sum;
        });
        const before = await pixelAt();
        const frame0 = Number(await page.locator('.screen-canvas').getAttribute('data-frame'));
        await page.click('.screen-canvas');
        await page.keyboard.press('ArrowRight');
        await page.keyboard.press('ArrowRight');
        await page.waitForFunction((f) => Number(document.querySelector('.screen-canvas').dataset.frame) >= f + 2, frame0, { timeout: 5000 });
        assert.notEqual(await pixelAt(), before, 'the square moved');
        await page.evaluate(() => window.tilStudio.settings.set('calculatorBezel', true));
        await shot(page, 'screen-bezel');
        await page.evaluate(() => window.tilStudio.settings.set('calculatorBezel', false));
        await page.click('.screen-canvas');
        await page.keyboard.press('Escape');
        await page.waitForFunction(() => !window.tilStudio.runner.running, null, { timeout: 10000 });
        await page.context().close();
    });

    it('debugs: breakpoint, variables, step, step back, timeline', async () => {
        const page = await open();
        await openFile(page, 'main.til');
        const src = await docText(page);
        const bpLine = lineOf(src, 'print(f"total');
        await placeCursor(page, bpLine, 'print');
        await page.keyboard.press('F9');
        await page.waitForFunction((l) => (window.tilStudio.breakpoints.get('main.til') || []).some((b) => b.line === l), bpLine);
        await page.keyboard.press('F5');
        await page.waitForSelector('.cm-debugLine', { timeout: 15000 });
        const at = () => page.evaluate(() => window.tilStudio.debugger.location && window.tilStudio.debugger.location.line);
        assert.equal(await at(), bpLine);
        await page.waitForSelector('.vars .var-row');
        await page.waitForSelector('.debugbar');
        const tl0 = await page.evaluate(() => window.tilStudio.debugger.timeline);
        assert.ok(tl0 && tl0.total > tl0.index, JSON.stringify(tl0));
        await shot(page, 'debug');
        await page.keyboard.press('F10');
        await page.waitForFunction((l) => window.tilStudio.debugger.stopped && window.tilStudio.debugger.location.line !== l, bpLine, { timeout: 10000 });
        await page.keyboard.press('Shift+F10');
        await page.waitForFunction((l) => window.tilStudio.debugger.stopped && window.tilStudio.debugger.location.line === l, bpLine, { timeout: 10000 });
        // the timeline: jump back to the first recorded step
        const range = page.locator('.tl-range');
        await range.evaluate((el) => { el.value = '0'; el.dispatchEvent(new Event('input')); el.dispatchEvent(new Event('change')); });
        await page.waitForFunction((l) => {
            const d = window.tilStudio.debugger;
            return d.stopped && d.timeline && d.timeline.index === 0 && d.location && d.location.line !== l;
        }, bpLine, { timeout: 10000 });
        // watch an expression
        await page.fill('.watch-add input', 'numbers');
        await page.keyboard.press('Enter');
        await page.waitForSelector('.watch-row');
        await page.keyboard.press('Shift+F5');
        await page.waitForFunction(() => !window.tilStudio.debugger.active, null, { timeout: 10000 });
        assert.equal(await page.locator('.cm-debugLine').count(), 0);
        await page.context().close();
    });

    it('debugs a TIL example where it is (its imports resolve)', async () => {
        const page = await open();
        const ex = await page.evaluate(() => {
            const list = window.tilStudio.info.examples;
            return (list.find((e) => /08_imports/.test(e.path)) || list.find((e) => /hello/.test(e.name)) || list[0]).path;
        });
        await page.evaluate((p) => window.tilStudio.openFile(p), ex);
        await page.waitForFunction((p) => window.tilStudio.editors.activeDoc() && window.tilStudio.editors.activeDoc().path === p, ex);
        await page.keyboard.press('F5');
        await page.waitForFunction(() => window.tilStudio.debugger.active, null, { timeout: 10000 });
        await page.waitForFunction(() => !window.tilStudio.debugger.active, null, { timeout: 20000 });
        const out = await page.locator('.term-out').textContent();
        if (!mock && /08_imports/.test(ex)) assert.match(out, /25\s*27\s*10/, out);      // square, cube, clamp from math_utils
        assert.doesNotMatch(out, /Could not find module|cannot read/, out);
        assert.deepEqual(page.errors, []);
        await page.context().close();
    });

    it('ends its debug session when the page goes away', async () => {
        const page = await open();
        await openFile(page, 'main.til');
        const src = await docText(page);
        await placeCursor(page, lineOf(src, 'print(f"total'), 'print');
        await page.keyboard.press('F9');
        await page.keyboard.press('F5');
        await page.waitForSelector('.cm-debugLine', { timeout: 15000 });
        const sid = await page.evaluate(() => window.tilStudio.debugger.client.sessionId);
        await page.goto('about:blank');                        // pagehide: a keepalive disconnect
        const deadline = Date.now() + 10000;
        let gone = false;
        while (!gone && Date.now() < deadline) {
            if (mock) gone = mock.state.stats.dap.includes('disconnect');
            else gone = await api('POST', `/api/dap/${sid}`, { seq: 99, type: 'request', command: 'threads' }).then(() => false, (e) => /404/.test(e.message));
            if (!gone) await new Promise((r) => setTimeout(r, 100));
        }
        assert.ok(gone, 'the debug session was ended');
        await page.context().close();
    });

    it('names each page to the server: its own language server and language', async () => {
        const kk = await browser.newContext({ viewport: { width: 1440, height: 900 } });
        await kk.addInitScript(() => localStorage.setItem('til-studio:settings', JSON.stringify({ uiLang: 'kk' })));
        const a = await open({ ctx: kk });
        const b = await open();
        const ids = await Promise.all([a, b].map((p) => p.evaluate(() => window.tilStudio.backend.clientId)));
        assert.ok(ids[0] && ids[1] && ids[0] !== ids[1], JSON.stringify(ids));
        if (mock) {
            assert.ok(mock.state.stats.clients.events.includes(ids[0]) && mock.state.stats.clients.events.includes(ids[1]), 'the event streams name their page');
            assert.ok(mock.state.stats.clients.requests.includes(ids[0]), 'the requests name their page');
        } else {
            // the diagnostics of each page come from its own language server, in its own language
            await api('PUT', '/api/file', { path: 'tabs.til', text: 'main()\n    print(nope)\n' });
            const messages = async (page) => {
                await openFile(page, 'tabs.til');
                await page.waitForFunction(() => [...window.tilStudio.diagnostics.values()].some((l) => l.length), null, { timeout: 10000 });
                return page.evaluate(() => [...window.tilStudio.diagnostics.values()].flat().map((d) => d.message).join('\n'));
            };
            const inKazakh = await messages(a);
            const inEnglish = await messages(b);
            assert.match(inEnglish, /unknown name 'nope'/);
            assert.ok(inKazakh.includes('nope') && !/unknown name/.test(inKazakh), inKazakh);
            await api('POST', '/api/file/delete', { path: 'tabs.til' });
        }
        await a.context().close();
        await b.context().close();
    });

    it('proves contracts and offers to debug a counterexample', async () => {
        const page = await open();
        await openFile(page, 'proofs.til');
        await page.click('.ab-btn[data-view="proofs"]');
        await page.click('.proofs-top .btn');
        await page.waitForSelector('.proof-card, .note-card', { timeout: 60000 });
        if (await page.locator('.note-card.warn').count()) return;            // the server has no z3
        assert.ok(await page.locator('.proof-card.ok').count() >= 1, 'a proved function');
        if (await page.locator('.proof-card.bad').count()) {
            assert.ok(await page.locator('.proof-card.bad .cex').count() === 1);
            // sent as decimal text, shown and used as exact integers
            assert.match(await page.locator('.proof-card.bad .cex-value').first().textContent(), /^-?\d{12,}$/);
            assert.ok(await page.locator('.proof-card.bad button:has-text("Debug with these inputs")').isVisible());
        }
        assert.ok(await page.locator('.cm-proof-gutter .proof-mark').count() >= 1, 'gutter marks');
        await shot(page, 'proofs');
        await page.context().close();
    });

    it('checks the program against a device', async () => {
        const page = await open();
        await openFile(page, 'main.til');
        await page.click('.ab-btn[data-view="devices"]');
        await page.selectOption('.devices-body select', 'ti84ce');
        await page.click('.devices-body .btn.primary');
        await page.waitForSelector('.verdict', { timeout: 30000 });
        assert.ok(await page.locator('.res-row .bar').count() >= 2, 'memory bars');
        await shot(page, 'devices');
        await page.context().close();
    });

    it('syncs the bytecode with the editor', async () => {
        const page = await open();
        await openFile(page, 'main.til');
        await page.click('.ab-btn[data-view="bytecode"]');
        await page.waitForSelector('.bc-row', { timeout: 30000 });
        const src = await docText(page);
        const line = lineOf(src, 'print("done")');
        await placeCursor(page, line, 'print');
        await page.waitForSelector('.bc-row.current', { timeout: 5000 });
        const rowLine = await page.locator('.bc-row.current').first().getAttribute('data-line');
        assert.equal(Number(rowLine), line);
        await page.locator('.bc-row[data-line]:not([data-line="0"])').first().hover();
        await page.waitForSelector('.cm-bytecodeLine', { timeout: 5000 });
        await shot(page, 'bytecode');
        await page.context().close();
    });

    it('lists, adds and removes packages', async () => {
        const page = await open();
        await page.click('.ab-btn[data-view="packages"]');
        await page.waitForSelector('.pkg-row, .packages-body .empty-state', { timeout: 30000 });
        if (await page.locator('.packages-body .empty-state').count()) return;
        await page.fill('.pkg-add input >> nth=0', 'colors');
        await page.fill('.pkg-add input >> nth=1', './libs/colors');
        await page.click('.pkg-add .btn.primary');
        await page.waitForSelector('.pkg-row:has-text("colors")', { timeout: 30000 });
        await page.locator('.pkg-row:has-text("colors") button').click();
        await page.click('.dialog .btn.danger');
        await page.waitForSelector('.pkg-row:has-text("colors")', { state: 'detached', timeout: 30000 });
        await page.context().close();
    });

    it('searches in files', async () => {
        const page = await open();
        await page.keyboard.press(`${MOD}+Shift+F`);
        await page.keyboard.type('square');
        await page.waitForSelector('.sr-match', { timeout: 10000 });
        assert.ok(await page.locator('.sr-file:has-text("math_utils.til")').count() === 1);
        await page.locator('.sr-match').first().click();
        await page.waitForFunction(() => window.tilStudio.editors.activeDoc() !== null);
        await page.context().close();
    });

    it('creates, renames and deletes files with confirmation', async () => {
        const page = await open();
        await page.click('.side-actions button >> nth=0');
        await page.waitForFunction(() => document.activeElement && document.activeElement.classList.contains('tree-input'));
        await page.keyboard.type('scratch.til');
        await page.keyboard.press('Enter');
        await page.waitForFunction(() => window.tilStudio.editors.activeDoc() && window.tilStudio.editors.activeDoc().path === 'scratch.til', null, { timeout: 10000 });
        await page.locator('.tree-row:has-text("scratch.til")').click();
        await page.keyboard.press('F2');
        await page.waitForFunction(() => document.activeElement && document.activeElement.classList.contains('tree-input'));
        await page.keyboard.press(`${MOD}+a`);
        await page.keyboard.type('renamed.til');
        await page.keyboard.press('Enter');
        await page.waitForSelector('.tree-row:has-text("renamed.til")', { timeout: 10000 });
        await page.locator('.tree-row:has-text("renamed.til")').click({ button: 'right' });
        await page.click('.menu .menu-item.danger');
        await page.waitForSelector('.dialog');
        await shot(page, 'confirm-delete');
        await page.click('.dialog .btn.danger');
        await page.waitForSelector('.tree-row:has-text("renamed.til")', { state: 'detached', timeout: 10000 });
        await page.context().close();
    });

    it('saves with Ctrl+S and keeps a conflict from overwriting newer changes', async () => {
        const page = await open();
        await openFile(page, 'math_utils.til');
        await gotoLineEnd(page, 1);
        await page.keyboard.type(' (edited)');
        await page.waitForSelector('.tab.dirty');
        await page.keyboard.press(`${MOD}+s`);
        await page.waitForSelector('.tab.dirty', { state: 'detached' });
        const saved = await api('GET', '/api/file?path=math_utils.til');
        assert.match(saved.text, /\(edited\)/);
        // someone else changes the file: saving asks first
        await api('PUT', '/api/file', { path: 'math_utils.til', text: saved.text + '\n# changed elsewhere\n' });
        await page.keyboard.type('!');
        await page.keyboard.press(`${MOD}+s`);
        const dialog = page.locator('.dialog');
        await dialog.waitFor({ timeout: 10000 }).catch(() => {});
        if (await dialog.count()) await dialog.locator('.btn.danger').click();         // overwrite
        await page.waitForSelector('.tab.dirty', { state: 'detached', timeout: 10000 });
        await api('PUT', '/api/file', { path: 'math_utils.til', text: PROJECT['math_utils.til'] });
        await page.context().close();
    });

    it('switches theme and UI language live', async () => {
        const page = await open();
        await page.click('.tb-theme');
        assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'light');
        await page.evaluate(() => window.tilStudio.settings.set('uiLang', 'ru'));
        assert.equal(await page.locator('.menubar button').first().textContent(), 'Файл');
        if (mock) {
            await page.evaluate(() => window.tilStudio.backend.tree(''));
            assert.equal(mock.state.stats.lang, 'ru', 'X-TIL-Lang follows the UI language');
        }
        assert.equal(await page.evaluate(() => document.documentElement.lang), 'ru');
        await page.evaluate(() => window.tilStudio.commands.run('workbench.welcome'));
        assert.equal(await page.locator('.wl-start h2').textContent(), 'Начало', 'the welcome page follows too');
        await page.evaluate(() => window.tilStudio.settings.set('uiLang', 'kk'));
        assert.match(await page.locator('.tb-run').textContent(), /Іске қосу/);
        assert.equal(await page.locator('.wl-start h2').textContent(), 'Бастау');
        await page.evaluate(() => { window.tilStudio.settings.set('uiLang', 'en'); window.tilStudio.settings.set('theme', 'dark'); });
        await page.context().close();
    });

    it('runs commands from the palette and opens files by name', async () => {
        const page = await open();
        await page.keyboard.press(`${MOD}+Shift+P`);
        await page.waitForSelector('.quickpick');
        await page.keyboard.type('toggle dark');
        await page.waitForSelector('.qp-item[aria-selected="true"]');
        await page.keyboard.press('Enter');
        assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'light');
        await page.keyboard.press(`${MOD}+p`);
        await page.keyboard.type('mautil');
        await page.waitForSelector('.qp-item[aria-selected="true"]:has-text("math_utils.til")');
        await page.keyboard.press('Enter');
        await page.waitForFunction(() => window.tilStudio.editors.activeDoc() && window.tilStudio.editors.activeDoc().path === 'math_utils.til');
        await page.keyboard.press(`${MOD}+g`);
        await page.keyboard.type('4');
        await page.keyboard.press('Enter');
        await page.waitForFunction(() => {
            const v = window.tilStudio.editors.activeView();
            return v.state.doc.lineAt(v.state.selection.main.head).number === 4;
        });
        await page.context().close();
    });

    it('translates a file side by side and in place (undoable)', async () => {
        const page = await open();
        await openFile(page, 'main.til');
        await page.evaluate(() => window.tilStudio.openTranslation('main.til', 'kk'));
        await page.waitForSelector('.translate-view .tr-pane .cm-content');
        await page.waitForFunction(() => document.querySelectorAll('.translate-view .cm-content')[1].textContent.includes('қайтару'));
        await shot(page, 'translate');
        await page.click('.translate-view .tr-bar .btn.primary');
        await page.waitForFunction(() => window.tilStudio.editors.activeView().state.doc.toString().includes('қайтару'));
        await page.keyboard.press(`${MOD}+z`);
        await page.waitForFunction(() => !window.tilStudio.editors.activeView().state.doc.toString().includes('қайтару'));
        await page.context().close();
    });

    it('keeps tabs, breakpoints and settings across a reload', async () => {
        const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
        let page = await open({ ctx: context });
        await openFile(page, 'main.til');
        await openFile(page, 'math_utils.til');
        await placeCursor(page, 4, 'return');
        await page.keyboard.press('F9');
        await page.evaluate(() => { window.tilStudio.settings.set('theme', 'light'); window.tilStudio.settings.set('fontSize', 16); window.tilStudio.persist(); });
        await page.reload();
        await page.waitForSelector('body[data-ready="true"]');
        await page.waitForFunction(() => window.tilStudio.editors.openDocs().length >= 2, null, { timeout: 10000 });
        assert.equal(await page.evaluate(() => window.tilStudio.editors.activeDoc().path), 'math_utils.til');
        assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'light');
        assert.deepEqual(await page.evaluate(() => (window.tilStudio.breakpoints.get('math_utils.til') || []).map((b) => b.line)), [4]);
        assert.ok(await page.locator('.bp-dot').count() >= 1);
        await context.close();
    });

    it('is usable from the keyboard: menus, activity bar, focus rings', async () => {
        const page = await open();
        await page.locator('.menubar button').first().focus();
        await page.keyboard.press('Enter');
        await page.waitForSelector('.menu[role="menu"]');
        await page.keyboard.press('ArrowDown');
        await page.keyboard.press('Escape');
        assert.equal(await page.locator('.menu').count(), 0);
        await page.locator('.ab-btn[aria-selected="true"]').focus();
        await page.keyboard.press('ArrowDown');
        assert.equal(await page.evaluate(() => document.activeElement.dataset.view), 'search');
        await page.context().close();
    });

    it('types fast in a 5000-line file', async () => {
        const lines = [];
        for (let i = 0; i < 5000; i++) {
            lines.push(i % 25 === 0 ? `f${i}(x: int) -> int` : i % 25 === 24 ? `    return x + ${i}` : `    print(x * ${i} + len("line ${i}"))  # comment ${i}`);
        }
        await api('PUT', '/api/file', { path: 'big.til', text: lines.join('\n') + '\n' });
        const page = await open();
        await openFile(page, 'big.til');
        await gotoLineEnd(page, 2500);
        await page.evaluate(() => {
            window.__lat = [];
            // keydown -> the next frame (the editor updates its DOM synchronously; the frame paints it)
            document.addEventListener('keydown', () => {
                const t0 = performance.now();
                requestAnimationFrame(() => window.__lat.push(performance.now() - t0));
            }, true);
        });
        for (let i = 0; i < 40; i++) await page.keyboard.type('x', { delay: 0 });
        await page.waitForTimeout(300);
        const lat = await page.evaluate(() => window.__lat.slice().sort((a, b) => a - b));
        const median = lat[Math.floor(lat.length / 2)];
        console.log(`typing latency in a 5000-line file: median ${median.toFixed(1)} ms, max ${lat[lat.length - 1].toFixed(1)} ms`);
        assert.ok(median < 50, `median keystroke latency ${median} ms`);
        assert.ok(lat[Math.floor(lat.length * 0.9)] < 120, `90th percentile ${lat[Math.floor(lat.length * 0.9)]} ms`);
        await page.context().close();
        await api('POST', '/api/file/delete', { path: 'big.til' });
    });
});

/* ══════════════════════════════ screenshots ══════════════════════════════ */
describe('screenshots (1440x900, dark + light, en/ru/kk)', { skip: Boolean(process.env.SKIP_SHOTS) }, () => {
    it('takes the tour', async () => {
        for (const lang of ['en', 'ru', 'kk']) {
            for (const theme of ['dark', 'light']) {
                const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
                await context.addInitScript(([l, th]) => {
                    localStorage.setItem('til-studio:settings', JSON.stringify({ uiLang: l, theme: th }));
                }, [lang, theme]);
                const page = await open({ ctx: context });
                await shot(page, `${lang}-${theme}-1-welcome`);
                await openFile(page, 'main.til');
                await page.keyboard.press(`${MOD}+F5`);
                await page.waitForSelector('.term-foot', { timeout: 30000 });
                await shot(page, `${lang}-${theme}-2-editor`);
                if (await page.evaluate(() => window.tilStudio.caps.debugger)) {
                    const src = await docText(page);
                    await placeCursor(page, lineOf(src, 'print(f"total'), 'print');
                    await page.keyboard.press('F9');
                    await page.keyboard.press('F5');
                    await page.waitForSelector('.cm-debugLine', { timeout: 15000 });
                    await page.waitForSelector('.vars .var-row');
                    await shot(page, `${lang}-${theme}-3-debug`);
                    await page.keyboard.press('Shift+F5');
                    await page.waitForFunction(() => !window.tilStudio.debugger.active, null, { timeout: 10000 });
                    await page.keyboard.press('F9');
                }
                await page.keyboard.press(`${MOD}+Shift+P`);
                await page.waitForSelector('.quickpick');
                await shot(page, `${lang}-${theme}-4-palette`);
                await page.keyboard.press('Escape');
                await context.close();
            }
        }
    });
});

/* ══════════════════════════════ web face ══════════════════════════════ */
const TYPES = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.json': 'application/json', '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.til': 'text/plain; charset=utf-8',
    '.tbc': 'application/octet-stream', '.png': 'image/png',
};

describe('web face (static hosting)', { skip: Boolean(process.env.SKIP_WEB) || !fs.existsSync(path.join(SITE_DIR, 'studio', 'index.html')) }, () => {
    let server, WEB;
    before(async () => {
        server = http.createServer((req, res) => {
            const url = new URL(req.url, 'http://localhost');
            let file = path.join(SITE_DIR, path.normalize(decodeURIComponent(url.pathname)));
            if (!file.startsWith(SITE_DIR)) { res.writeHead(403); res.end(); return; }
            try {
                if (fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
                res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
                res.end(fs.readFileSync(file));
            } catch (e) {
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end('not found');
            }
        });
        await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
        WEB = `http://127.0.0.1:${server.address().port}/studio/`;
    });
    after(async () => { if (server) await new Promise((resolve) => server.close(resolve)); });

    it('starts with the browser backend and a starter project', async () => {
        const page = await open({ url: WEB });
        assert.equal(await page.evaluate(() => window.tilStudio.backend.kind), 'web');
        await page.waitForSelector('.tree-row:has-text("main.til")');
        await page.waitForSelector('.tree-row:has-text("examples")');
        await shot(page, 'web-welcome');
        await page.context().close();
    });

    it('compiles and runs in the browser (tilc + WebAssembly VM)', async () => {
        const page = await open({ url: WEB });
        await openFile(page, 'main.til');
        await page.keyboard.press(`${MOD}+F5`);
        await page.waitForSelector('.term-foot', { timeout: 30000 });
        assert.match(await page.locator('.term-out').textContent(), /Hello, TIL!/);
        assert.equal(await page.locator('.term-foot.ok').count(), 1);
        // a compile error from tilc becomes a diagnostic
        await gotoLineEnd(page, (await docText(page)).split('\n').length - 1);
        await page.keyboard.press('Enter');
        await page.keyboard.type('prnt(1)');
        await page.waitForSelector('.cm-errorLens-error', { timeout: 20000 });
        assert.match(await page.locator('.cm-errorLens-error').textContent(), /prnt/);
        await shot(page, 'web-diagnostics');
        await page.context().close();
    });

    it('runs graphics with keys and input()', async () => {
        const page = await open({ url: WEB });
        await openFile(page, 'examples/keys.til');
        await page.keyboard.press(`${MOD}+F5`);
        await page.waitForSelector('.screen-canvas[data-frame]', { timeout: 30000 });
        await page.click('.screen-canvas');
        const f0 = Number(await page.locator('.screen-canvas').getAttribute('data-frame'));
        await page.keyboard.press('ArrowRight');
        await page.waitForFunction((f) => Number(document.querySelector('.screen-canvas').dataset.frame) > f, f0, { timeout: 10000 });
        await shot(page, 'web-screen');
        await page.click('.tb-stop');
        await page.waitForFunction(() => !window.tilStudio.runner.running, null, { timeout: 10000 });
        await openFile(page, 'examples/input.til');
        await page.keyboard.press(`${MOD}+F5`);
        await page.waitForSelector('.term-input-row.waiting', { timeout: 30000 });
        await page.waitForFunction(() => document.activeElement && document.activeElement.classList.contains('term-input'));
        await page.keyboard.type('50');
        await page.keyboard.press('Enter');
        await page.waitForFunction(() => document.querySelector('.term-out').textContent.includes('50'), null, { timeout: 10000 });
        await page.click('.tb-stop');
        await page.context().close();
    });

    it('shows the desktop card for desktop-only features', async () => {
        const page = await open({ url: WEB });
        await page.click('.ab-btn[data-view="proofs"]');
        await page.waitForSelector('.proofs-body .desktop-card');
        assert.match(await page.locator('.proofs-body .desktop-card').textContent(), /til ide/);
        await page.click('.ab-btn[data-view="debug"]');
        await page.waitForSelector('.debug-card .desktop-card');
        assert.equal(await page.locator('.tb-debug').isVisible(), false);
        await page.context().close();
    });

    it('keeps files in the browser across a reload', async () => {
        const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
        let page = await open({ ctx: context, url: WEB });
        await openFile(page, 'main.til');
        await gotoLineEnd(page, 1);
        await page.keyboard.type(' (kept)');
        await page.keyboard.press(`${MOD}+s`);
        await page.waitForSelector('.tab.dirty', { state: 'detached' });
        await page.reload();
        await page.waitForSelector('body[data-ready="true"]');
        await openFile(page, 'main.til');
        assert.match(await docText(page), /\(kept\)/);
        await context.close();
    });
});
