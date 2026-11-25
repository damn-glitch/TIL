#!/usr/bin/env node
// build.mjs - bundle the TIL Studio frontend into ONE self-contained HTML file.
//
//   node build.mjs                    writes ../studio.html and ../../website/studio/index.html
//   node build.mjs --out FILE         writes only FILE (tests/test_studio_ui.py: freshness check)
//   node build.mjs --check            exit 1 when the committed files differ from a fresh build
//   node build.mjs --dev              not minified (readable stack traces), same layout
//   node build.mjs --watch            rebuild on every change (development)
//
// Everything is inlined: the JavaScript (src/main.js + CodeMirror 6), the CSS
// (styles/*.css in name order), the UI strings (i18n/*.json), the language
// tables (tables.json, from gen_tables.py) and the bundled examples. No CDN,
// no web fonts, works offline. The header comment carries the licenses of
// every third-party package that ends up in the bundle (found through
// esbuild's metafile), so the file can be redistributed as it is.
import * as esbuild from 'esbuild';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const DEFAULT_OUTS = [path.join(ROOT, 'ide', 'studio.html'), path.join(ROOT, 'website', 'studio', 'index.html')];

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const value = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : null; };
const DEV = flag('--dev');

/* ── bundled examples: a virtual module `virtual:examples` ───────────────── */
// [file in the repository, id, tags]; titles and descriptions are UI strings
// (i18n keys example.<id>.title / .desc).
const EXAMPLES = [
    ['website/playground/examples/hello.til', 'hello', ['basics']],
    ['website/playground/examples/basics.til', 'basics', ['basics']],
    ['examples/03_functions.til', 'functions', ['basics']],
    ['examples/04_structs.til', 'structs', ['types']],
    ['examples/06_enums.til', 'enums', ['types']],
    ['examples/09_v3_features.til', 'tour', ['language']],
    ['website/playground/examples/kazakh.til', 'kazakh', ['kk']],
    ['website/playground/examples/russian.til', 'russian', ['ru']],
    ['examples/kazakh/salam_latn.til', 'kazakh_latin', ['kk-latn']],
    ['website/playground/examples/input.til', 'input', ['input']],
    ['website/playground/examples/graphics.til', 'graphics', ['screen']],
    ['website/playground/examples/animation.til', 'animation', ['screen']],
    ['website/playground/examples/keys.til', 'keys', ['screen', 'keys']],
    ['examples/10_verified.til', 'verified', ['contracts']],
    ['examples/12_effects.til', 'effects', ['effects']],
    ['examples/13_proofs.til', 'proofs', ['proofs']],
    ['examples/14_energy.til', 'energy', ['energy']],
    ['examples/math_utils.til', 'math_utils', ['module']],
    ['examples/08_imports.til', 'imports', ['module']],
];

const examplesPlugin = {
    name: 'til-examples',
    setup(build) {
        build.onResolve({ filter: /^virtual:examples$/ }, () => ({ path: 'examples', namespace: 'til-examples' }));
        build.onLoad({ filter: /.*/, namespace: 'til-examples' }, () => {
            const list = EXAMPLES.map(([file, id, tags]) => ({
                id, tags, file: path.basename(file),
                source: fs.readFileSync(path.join(ROOT, file), 'utf8').replace(/\r\n?/g, '\n'),
            }));
            return { contents: `export default ${JSON.stringify(list)};`, loader: 'js',
                watchFiles: EXAMPLES.map(([file]) => path.join(ROOT, file)) };
        });
    },
};

/* ── licenses of the bundled packages ───────────────────────────────────── */
function packageOf(input) {
    const m = /node_modules\/((?:@[^/]+\/)?[^/]+)\//.exec(input.replace(/\\/g, '/'));
    return m ? m[1] : null;
}

function licenseHeader(metafile) {
    const pkgs = new Set();
    for (const input of Object.keys(metafile.inputs)) {
        const p = packageOf(input);
        if (p) pkgs.add(p);
    }
    const byText = new Map();
    for (const p of [...pkgs].sort()) {
        const dir = path.join(HERE, 'node_modules', p);
        let version = '?';
        try { version = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8')).version; } catch (e) { /* */ }
        let text = '';
        for (const name of ['LICENSE', 'LICENSE.md', 'license', 'LICENSE.txt']) {
            try { text = fs.readFileSync(path.join(dir, name), 'utf8').trim(); break; } catch (e) { /* */ }
        }
        if (!text) throw new Error(`no license file for the bundled package ${p}`);
        const key = text.replace(/Copyright \(C\)[^\n]*\n/gi, '').trim();
        if (!byText.has(key)) byText.set(key, { names: [], copyrights: new Set(), text: key });
        const entry = byText.get(key);
        entry.names.push(`${p}@${version}`);
        for (const line of text.split('\n')) if (/^Copyright \(C\)/i.test(line)) entry.copyrights.add(line.trim());
    }
    const parts = [
        'TIL Studio - the IDE of the TIL programming language (https://github.com/damn-glitch/TIL).',
        'Built from ide/studio/ (node ide/studio/build.mjs); do not edit this file by hand.',
        'TIL Studio itself is part of TIL and licensed like it (see LICENSE in the TIL repository).',
        '',
        'This file contains third-party code, bundled with esbuild:',
    ];
    for (const entry of byText.values()) {
        parts.push('', entry.names.join(', '), ...entry.copyrights, '', entry.text);
    }
    return '<!--\n' + parts.join('\n').replace(/--/g, '- -') + '\n-->\n';
}

/* ── the page ──────────────────────────────────────────────────────────── */
const FAVICON = 'data:image/svg+xml,' + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#11141b"/>' +
    '<rect x="7" y="8" width="18" height="3.4" rx="1.7" fill="#e6e9f0"/><rect x="14.3" y="8" width="3.4" height="15" rx="1.7" fill="#e6e9f0"/>' +
    '<circle cx="16" cy="26" r="2.1" fill="#00d4ff"/></svg>');

// theme before the first paint (the bundle runs at the end of <body>)
const EARLY = `(function(){try{var s=JSON.parse(localStorage.getItem('til-studio:settings')||'{}');var t=s.theme;` +
    `if(t!=='light'&&t!=='dark'){t=window.matchMedia&&matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'}` +
    `document.documentElement.dataset.theme=t;if(s.uiLang)document.documentElement.lang=s.uiLang}catch(e){}})();`;

function page({ css, js, header }) {
    const safeJs = js.replace(/<\/script/gi, '<\\/script').replace(/<!--/g, '<\\!--');
    const safeCss = css.replace(/<\/style/gi, '<\\/style');
    return `<!DOCTYPE html>\n${header}<html lang="en" data-theme="dark">\n<head>\n<meta charset="utf-8">\n` +
        `<meta name="viewport" content="width=device-width, initial-scale=1">\n` +
        `<meta name="color-scheme" content="dark light">\n<title>TIL Studio</title>\n` +
        `<meta name="description" content="TIL Studio: write, run, debug and prove TIL programs.">\n` +
        `<link rel="icon" href="${FAVICON}">\n<script>${EARLY}</script>\n<style>${safeCss}</style>\n</head>\n` +
        `<body>\n<div id="app" class="app-boot"></div>\n` +
        `<noscript><p style="font:14px system-ui;padding:24px">TIL Studio needs JavaScript.</p></noscript>\n` +
        `<script>${safeJs}</script>\n</body>\n</html>\n`;
}

async function buildCss() {
    const dir = path.join(HERE, 'styles');
    const files = fs.readdirSync(dir).filter((f) => f.endsWith('.css')).sort();
    const css = files.map((f) => fs.readFileSync(path.join(dir, f), 'utf8')).join('\n');
    const out = await esbuild.transform(css, { loader: 'css', minify: !DEV, target: ['chrome100', 'firefox100', 'safari15'] });
    return out.code;
}

async function buildOnce() {
    const result = await esbuild.build({
        entryPoints: [path.join(HERE, 'src', 'main.js')],
        bundle: true,
        write: false,
        format: 'iife',
        minify: !DEV,
        legalComments: 'none',
        charset: 'utf8',
        target: ['es2020', 'chrome100', 'firefox100', 'safari15'],
        metafile: true,
        logLevel: 'warning',
        plugins: [examplesPlugin],
        define: { STUDIO_DEV: DEV ? 'true' : 'false' },
    });
    const js = result.outputFiles[0].text;
    const css = await buildCss();
    return { html: page({ css, js, header: licenseHeader(result.metafile) }), metafile: result.metafile };
}

function writeOut(file, html) {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, html);
}

async function main() {
    const out = value('--out');
    const targets = out ? [path.resolve(out)] : DEFAULT_OUTS;
    if (flag('--check')) {
        const { html } = await buildOnce();
        let stale = 0;
        for (const file of DEFAULT_OUTS) {
            let cur = '';
            try { cur = fs.readFileSync(file, 'utf8'); } catch (e) { /* missing */ }
            if (cur !== html) { console.error(`out of date: ${path.relative(ROOT, file)}`); stale++; }
        }
        process.exit(stale ? 1 : 0);
    }
    const run = async () => {
        const t0 = Date.now();
        const { html } = await buildOnce();
        for (const file of targets) writeOut(file, html);
        const kb = (Buffer.byteLength(html) / 1024).toFixed(1);
        console.log(`built ${targets.map((f) => path.relative(ROOT, f)).join(', ')}: ${kb} KB in ${Date.now() - t0} ms`);
    };
    await run();
    if (flag('--watch')) {
        let timer = 0;
        const again = () => { clearTimeout(timer); timer = setTimeout(() => run().catch((e) => console.error(e.message)), 80); };
        for (const d of ['src', 'styles', 'i18n']) fs.watch(path.join(HERE, d), { recursive: true }, again);
        fs.watch(path.join(HERE, 'tables.json'), again);
        console.log('watching for changes…');
    }
}

main().catch((e) => { console.error(e.stack || e.message); process.exit(1); });
