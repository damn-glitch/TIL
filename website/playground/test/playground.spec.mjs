// Browser tests for the TIL playground (Node 22 + Playwright, node:test).
//
//   node website/playground/test/playground.spec.mjs
//
// Serves the site with a small static server on a random port and drives
// /playground/ in headless Chromium. Environment (all optional; set by
// tests/test_playground.py):
//   SITE_DIR      directory served at / (default: website/ of this checkout)
//   TEST_ASSETS   directory served at /__test__/; if it holds mock-tilc.tbc, the
//                 compile pipeline is also tested with ?compiler=/__test__/mock-tilc.tbc
//                 (a stand-in for tilc.tbc that knows the programs in mock_programs.json)
//   GRAPHICS      "real" or "mock": the VM implements the screen/key natives, so
//                 the graphics tests must not be skipped
import { after, before, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SITE_DIR = path.resolve(process.env.SITE_DIR || path.join(HERE, '..', '..'));
const TEST_ASSETS = process.env.TEST_ASSETS ? path.resolve(process.env.TEST_ASSETS) : null;
const GRAPHICS = process.env.GRAPHICS || '';
const PROGRAMS = JSON.parse(fs.readFileSync(path.join(HERE, 'mock_programs.json'), 'utf8'));
const program = (name) => PROGRAMS.find((p) => p.name === name);
const HAS_MOCK = Boolean(TEST_ASSETS && fs.existsSync(path.join(TEST_ASSETS, 'mock-tilc.tbc')));
const FINAL = ['done', 'error', 'compile-error', 'no-compiler', 'stopped'];

/* ───────────────────────── static server ───────────────────────── */
const TYPES = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript',
    '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.wasm': 'application/wasm',
    '.svg': 'image/svg+xml', '.png': 'image/png', '.til': 'text/plain; charset=utf-8',
    '.tbc': 'application/octet-stream', '.sh': 'text/plain', '.ps1': 'text/plain',
};

function serve(req, res) {
    const url = new URL(req.url, 'http://localhost');
    let root = SITE_DIR, rel = decodeURIComponent(url.pathname);
    if (rel.startsWith('/__test__/')) {
        if (!TEST_ASSETS) { res.writeHead(404); res.end(); return; }
        root = TEST_ASSETS;
        rel = rel.slice('/__test__'.length);
    }
    let file = path.join(root, path.normalize(rel));
    if (!file.startsWith(root)) { res.writeHead(403); res.end(); return; }
    try {
        if (fs.statSync(file).isDirectory()) {
            if (!rel.endsWith('/')) { res.writeHead(301, { Location: url.pathname + '/' }); res.end(); return; }
            file = path.join(file, 'index.html');
        }
        const data = fs.readFileSync(file);
        res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
            'Cache-Control': 'no-store' });
        res.end(data);
    } catch (e) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('not found');
    }
}

let server, browser, BASE;

before(async () => {
    server = http.createServer(serve);
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    BASE = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch();
});

after(async () => {
    if (browser) await browser.close();
    if (server) await new Promise((resolve) => server.close(resolve));
});

/* ───────────────────────── page helpers ───────────────────────── */
async function openPage(query = '', hash = '') {
    const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
    page.errors = [];
    page.on('pageerror', (e) => page.errors.push(e.message));
    await page.goto(`${BASE}/playground/${query}${hash}`);
    await page.waitForFunction(() => document.body.dataset.state === 'idle', null, { timeout: 20000 });
    return page;
}

const state = (page) => page.evaluate(() => document.body.dataset.state);
const output = (page) => page.textContent('#output');

async function waitState(page, states, timeout = 15000) {
    await page.waitForFunction((s) => s.includes(document.body.dataset.state), states, { timeout });
    return state(page);
}

async function setSource(page, text) {
    await page.fill('#code', text);
}

async function loadExample(page, id) {
    await page.selectOption('#examples', id);
    await page.waitForFunction((i) => document.getElementById('examples').value === i &&
        document.getElementById('code').value.length > 0, id);
}

/* run the current editor text; resolves with the state it settles in */
async function run(page, settle = FINAL, timeout = 15000) {
    await page.click('#run');
    return waitState(page, settle, timeout);
}

async function nonBlackPixels(page) {
    return page.evaluate(() => {
        const c = document.getElementById('screen');
        if (c.closest('[hidden]')) return -1;
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let n = 0;
        for (let i = 0; i < d.length; i += 4) if (d[i] || d[i + 1] || d[i + 2]) n++;
        return n;
    });
}

const notAvailable = (text) => /is not available (in this VM build|on this device)/.test(text);

/* ───────────────────────── checks shared by all modes ───────────────────────── */
async function checkInput(page) {
    await waitState(page, ['waiting-line']);
    assert.equal(await page.isVisible('#stdin-text'), true, 'the input line is shown');
    await page.fill('#stdin-text', 'Aigerim');
    await page.press('#stdin-text', 'Enter');
}

async function checkStop(page, marker) {
    await page.click('#run');
    await page.waitForFunction((m) => document.getElementById('output').textContent.includes(m), marker,
        { timeout: 15000 });
    await page.waitForTimeout(400);
    assert.equal(await state(page), 'running', 'still running');
    const t0 = Date.now();
    // the page answers while the program runs
    assert.equal(await page.evaluate(() => 6 * 7), 42);
    await page.click('#stop');
    assert.equal(await waitState(page, ['stopped'], 5000), 'stopped');
    assert.ok(Date.now() - t0 < 5000, 'Stop reacts quickly');
    const steps = await page.textContent('#stats');
    await page.waitForTimeout(300);
    assert.equal(await page.textContent('#stats'), steps, 'no more steps after Stop');
}

async function checkKeys(page) {
    const canvas = page.locator('#screen');
    await canvas.click();
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('a');
    await page.keyboard.press('Escape');
    return waitState(page, FINAL);
}

/* ───────────────────────── the site as built ───────────────────────── */
describe('playground as built', () => {
    let page, mode;

    before(async () => {
        page = await openPage();
        mode = await page.evaluate(() => document.body.dataset.compiler);
    });

    after(async () => { if (page) await page.close(); });

    it('loads the VM and reports how it compiles', async () => {
        assert.ok(['tilc', 'precompiled'].includes(mode), `compiler mode ${mode}`);
        const info = await page.textContent('#vm-info');
        assert.match(info, /TIL VM \d+\.\d+\.\d+, WebAssembly/);
        if (mode === 'tilc') assert.match(await page.textContent('#compiler-info'), /tilc\.tbc/);
        else assert.match(await page.textContent('#compiler-info'), /precompiled/);
    });

    it('runs hello world', async () => {
        if (mode === 'tilc') {
            await setSource(page, program('hello').source);
            assert.equal(await run(page), 'done', await output(page));
            assert.equal(await output(page), program('hello').expect.output);
        } else {
            await loadExample(page, 'hello');
            assert.equal(await run(page), 'done', await output(page));
            assert.match(await output(page), /^Hello, World!\nСәлем, әлем!\n/);
        }
        assert.match(await page.textContent('#state'), /Finished/);
    });

    it('shows the line of a runtime error', async () => {
        let line;
        if (mode === 'tilc') {
            await setSource(page, program('runtime_error').source);
            line = 3;
        } else {
            await loadExample(page, 'errors');
            line = 5;
        }
        assert.equal(await run(page), 'error');
        const text = await output(page);
        assert.match(text, /runtime error: /);
        assert.match(text, new RegExp(`\\(line ${line}\\)`));
        assert.equal(await page.evaluate(() => document.body.dataset.errorLine), String(line));
        assert.match(await page.textContent('#state'), new RegExp(`line ${line}`));
        assert.equal(await page.isVisible('#error-line'), true);
        assert.equal(await page.textContent('#gutter .err'), String(line));
    });

    it('shows compile errors with their line (in-browser compiler only)', async (t) => {
        if (mode !== 'tilc') {
            t.skip('no in-browser compiler in this build');
            return;
        }
        await setSource(page, program('compile_error').source);
        assert.equal(await run(page), 'compile-error');
        assert.match(await output(page), /\S/);
        assert.equal(await page.evaluate(() => document.body.dataset.errorLine), '2');
        assert.match(await page.textContent('#state'), /Compile error \(line 2\)/);
    });

    it('stops an endless program', async () => {
        if (mode === 'tilc') {
            await setSource(page, program('infinite').source);
            await checkStop(page, 'spinning');
        } else {
            await loadExample(page, 'forever');
            await checkStop(page, 'new record');
        }
    });

    it('pauses input() until a line is typed', async () => {
        if (mode === 'tilc') {
            await setSource(page, program('input').source);
            await page.click('#run');
            await checkInput(page);
            await waitState(page, ['waiting-line']);
            await page.fill('#stdin-text', 'more');
            await page.press('#stdin-text', 'Enter');
            assert.equal(await waitState(page, FINAL), 'done');
            assert.equal(await output(page), 'Name? Aigerim\nHi, Aigerim!\nmore\n[more]\n');
        } else {
            await loadExample(page, 'input');
            await page.click('#run');
            await checkInput(page);
            await page.waitForFunction(() => document.body.dataset.state === 'waiting-line' &&
                document.getElementById('output').textContent.includes('Your guess'));
            await page.fill('#stdin-text', '');
            await page.press('#stdin-text', 'Enter');
            assert.equal(await waitState(page, FINAL), 'done');
            const text = await output(page);
            assert.match(text, /^What is your name\? Aigerim\nNice to meet you, Aigerim!\n/);
            assert.match(text, /The number was \d+\. Bye, Aigerim!\n$/);
            assert.equal(text.split('What is your name?').length, 2, 'the prompt is printed once');
        }
    });

    it('round-trips code through a share link', async () => {
        const source = '# shared: сәлем, привет ✓\nprint("round trip")\n';
        await setSource(page, source);
        await page.evaluate(() => { navigator.clipboard.writeText = () => Promise.resolve(); });
        await page.click('#share');
        await page.waitForFunction(() => location.hash.length > 3);
        const url = await page.evaluate(() => location.href);
        assert.match(url, /#[zs]=[A-Za-z0-9_-]+$/);
        const other = await browser.newPage();
        try {
            await other.goto(url);
            await other.waitForFunction(() => document.body.dataset.state === 'idle');
            assert.equal(await other.inputValue('#code'), source);
        } finally {
            await other.close();
        }
    });

    it('draws on the canvas', async (t) => {
        if (mode === 'tilc') await setSource(page, program('graphics').source);
        else await loadExample(page, 'graphics');
        const st = await run(page);
        const text = await output(page);
        if (st === 'error' && notAvailable(text) && !GRAPHICS) {
            t.skip('this VM build has no screen natives yet');
            return;
        }
        assert.equal(st, 'done', text);
        assert.ok(await nonBlackPixels(page) > 100, 'the canvas shows the picture');
        const box = await page.locator('#screen').boundingBox();
        assert.ok(box.width >= 160, 'the screen is scaled up');
    });

    it('sends keys to wait_key()', async (t) => {
        if (mode === 'tilc') await setSource(page, program('keys').source);
        else await loadExample(page, 'keys');
        await page.click('#run');
        const st = await waitState(page, ['waiting-key', ...FINAL]);
        if (st === 'error' && notAvailable(await output(page)) && !GRAPHICS) {
            t.skip('this VM build has no screen/key natives yet');
            return;
        }
        assert.equal(st, 'waiting-key', await output(page));
        assert.equal(await checkKeys(page), 'done');
        const text = await output(page);
        if (mode === 'tilc') assert.equal(text, 'key 4\nkey 97\nbye\n');
        else assert.match(text, /key code 97\nbye after 1 moves\n$/);
    });

    it('raised no page errors', () => {
        assert.deepEqual(page.errors, []);
    });
});

/* ───────────────────── the compile pipeline with a mock tilc ───────────────────── */
describe('compile pipeline (mock tilc.tbc)', { skip: HAS_MOCK ? false : 'TEST_ASSETS/mock-tilc.tbc not provided' }, () => {
    let page;

    before(async () => {
        page = await openPage('?compiler=/__test__/mock-tilc.tbc');
    });

    after(async () => { if (page) await page.close(); });

    it('uses the compiler from ?compiler=', async () => {
        assert.equal(await page.evaluate(() => document.body.dataset.compiler), 'custom');
        assert.match(await page.textContent('#compiler-info'), /mock-tilc\.tbc/);
    });

    for (const prog of PROGRAMS) {
        it(`runs "${prog.name}" through compiler VM -> program VM`, async (t) => {
            const exp = prog.expect;
            await setSource(page, prog.source);
            if (prog.name === 'infinite') { await checkStop(page, 'spinning'); return; }
            if (prog.name === 'input') {
                await page.click('#run');
                await checkInput(page);
                await waitState(page, ['waiting-line']);
                await page.fill('#stdin-text', 'x');
                await page.press('#stdin-text', 'Enter');
                assert.equal(await waitState(page, FINAL), 'done');
                assert.equal(await output(page), 'Name? Aigerim\nHi, Aigerim!\nx\n[x]\n');
                return;
            }
            if (prog.name === 'keys') {
                await page.click('#run');
                const st = await waitState(page, ['waiting-key', ...FINAL]);
                if (st === 'error' && notAvailable(await output(page)) && !GRAPHICS) {
                    t.skip('no key natives in this VM build');
                    return;
                }
                assert.equal(st, 'waiting-key', await output(page));
                assert.equal(await checkKeys(page), 'done');
                assert.equal(await output(page), 'key 4\nkey 97\nbye\n');
                return;
            }
            const t0 = Date.now();
            const st = await run(page);
            const elapsed = Date.now() - t0;
            const text = await output(page);
            if (prog.name === 'sleep') assert.ok(elapsed >= 300, `sleep_ms(300) really waits (${elapsed} ms)`);
            if (exp.graphics && st === 'error' && notAvailable(text) && !GRAPHICS) {
                t.skip('no screen natives in this VM build');
                return;
            }
            assert.equal(st, exp.state, text);
            if (exp.output !== undefined) assert.equal(text, exp.output);
            for (const frag of exp.output_contains || []) assert.ok(text.includes(frag), `${frag} in ${text}`);
            if (exp.error_line) {
                assert.equal(await page.evaluate(() => document.body.dataset.errorLine), String(exp.error_line));
            }
            if (exp.status_contains) assert.ok((await page.textContent('#state')).includes(exp.status_contains));
            if (exp.graphics) assert.ok(await nonBlackPixels(page) > 50, 'the canvas shows the picture');
            if (prog.name === 'compile_error') {
                assert.match(text, /--> \/main\.til:2:\d+/);
                assert.match(await page.textContent('#state'), /Compile error \(line 2\)/);
            }
        });
    }

    it('reports a program the compiler rejects', async () => {
        await setSource(page, 'print("a program the mock compiler does not know")\n');
        assert.equal(await run(page), 'compile-error');
        assert.match(await output(page), /mock compiler: unknown program/);
    });

    it('raised no page errors', () => {
        assert.deepEqual(page.errors, []);
    });
});

/* ───────────────────── page behaviour without a compiler ───────────────────── */
describe('without an in-browser compiler', () => {
    it('refuses edited code with a clear message (precompiled mode only)', async (t) => {
        const page = await openPage();
        try {
            if (await page.evaluate(() => document.body.dataset.compiler) !== 'precompiled') {
                t.skip('this build has tilc.tbc');
                return;
            }
            await setSource(page, 'print("my own code")\n');
            assert.equal(await run(page), 'no-compiler');
            assert.match(await output(page), /no in-browser compiler/);
        } finally {
            await page.close();
        }
    });

    it('opens an example from #example=', async () => {
        const page = await openPage('', '#example=kazakh');
        try {
            assert.match(await page.inputValue('#code'), /қайтару/);
            assert.equal(await page.inputValue('#examples'), 'kazakh');
        } finally {
            await page.close();
        }
    });
});
