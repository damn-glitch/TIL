#!/usr/bin/env node
// Bundle the extension: src/extension.js + vscode-languageclient -> dist/extension.js
// (one CommonJS file, `vscode` external), and the license notices of the bundled
// packages -> ThirdPartyNotices.txt. Both are committed, so the extension runs
// from a checkout without npm.
//
//   npm ci && node scripts/build.mjs          build
//   node scripts/build.mjs --check            exit 1 if the committed files are out of date
//   node scripts/build.mjs --release          build + copy the repository's LICENSE (for vsce)
import { build } from 'esbuild';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ext = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pkg = JSON.parse(fs.readFileSync(path.join(ext, 'package.json'), 'utf8'));
const args = new Set(process.argv.slice(2));
const OUT = path.join(ext, 'dist', 'extension.js');
const NOTICES = path.join(ext, 'ThirdPartyNotices.txt');

async function bundle(outfile) {
    const result = await build({
        absWorkingDir: ext,
        entryPoints: ['src/extension.js'],
        outfile,
        bundle: true,
        platform: 'node',
        format: 'cjs',
        target: 'node18',
        external: ['vscode'],
        charset: 'utf8',
        legalComments: 'none',
        minifyWhitespace: true,
        minifySyntax: true,
        keepNames: true,
        metafile: true,
        logLevel: 'warning',
        banner: {
            js: `/* TIL for VS Code ${pkg.version} - built from src/ by scripts/build.mjs (esbuild). ` +
                'Bundles vscode-languageclient and its dependencies (MIT / ISC): see ThirdPartyNotices.txt. */',
        },
    });
    return result.metafile;
}

/** The npm packages that ended up in the bundle (from esbuild's metafile). */
function bundledPackages(metafile) {
    const names = new Set();
    for (const input of Object.keys(metafile.inputs)) {
        const m = /(?:^|\/)node_modules\/((?:@[^/]+\/)?[^/]+)\//.exec(input.replace(/\\/g, '/'));
        if (m) names.add(m[1]);
    }
    return [...names].sort();
}

function licenseText(dir) {
    const file = fs.readdirSync(dir).find((f) => /^licen[cs]e(\.(md|txt))?$/i.test(f));
    return file ? fs.readFileSync(path.join(dir, file), 'utf8').replace(/\r\n/g, '\n').trim() : '';
}

function notices(packages) {
    const parts = [
        'TIL for VS Code: third-party software in dist/extension.js',
        '',
        'dist/extension.js bundles the following npm packages. Their license notices follow.',
        '',
    ];
    for (const name of packages) {
        const dir = path.join(ext, 'node_modules', name);
        const p = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'));
        parts.push('-'.repeat(78));
        parts.push(`${p.name} ${p.version} (${p.license || 'see below'})`);
        if (p.repository) parts.push(typeof p.repository === 'string' ? p.repository : p.repository.url || '');
        parts.push('');
        parts.push(licenseText(dir) || `License: ${p.license}`);
        parts.push('');
    }
    return parts.join('\n').replace(/\n+$/, '\n');
}

function copyLicense() {
    const license = path.join(ext, '..', '..', 'LICENSE');
    if (fs.existsSync(license)) fs.copyFileSync(license, path.join(ext, 'LICENSE'));
}

async function main() {
    if (!fs.existsSync(path.join(ext, 'node_modules', 'vscode-languageclient'))) {
        if (args.has('--release') && fs.existsSync(OUT)) {
            // packaging a checkout without `npm ci`: the committed bundle is used as it is
            copyLicense();
            console.log('build: no node_modules - packaging the committed dist/extension.js');
            return;
        }
        console.error('build: node_modules is missing: run `npm ci` in editors/vscode first');
        process.exit(2);
    }
    if (args.has('--check')) {
        const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'til-vscode-build-'));
        try {
            const out = path.join(tmp, 'extension.js');
            const meta = await bundle(out);
            const stale = [];
            if (!fs.existsSync(OUT) || !fs.readFileSync(out).equals(fs.readFileSync(OUT))) stale.push('dist/extension.js');
            const text = notices(bundledPackages(meta));
            if (!fs.existsSync(NOTICES) || fs.readFileSync(NOTICES, 'utf8') !== text) stale.push('ThirdPartyNotices.txt');
            if (stale.length) {
                console.error(`out of date (run node scripts/build.mjs): ${stale.join(', ')}`);
                process.exit(1);
            }
            console.log('dist/extension.js and ThirdPartyNotices.txt are up to date');
        } finally {
            fs.rmSync(tmp, { recursive: true, force: true });
        }
        return;
    }
    fs.mkdirSync(path.dirname(OUT), { recursive: true });
    const meta = await bundle(OUT);
    fs.writeFileSync(NOTICES, notices(bundledPackages(meta)));
    const size = fs.statSync(OUT).size;
    console.log(`dist/extension.js: ${(size / 1024).toFixed(1)} KB; ThirdPartyNotices.txt: ${bundledPackages(meta).join(', ')}`);
    if (args.has('--release')) copyLicense();
}

main().catch((e) => {
    console.error(e && e.stack || e);
    process.exit(1);
});
