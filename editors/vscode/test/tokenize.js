#!/usr/bin/env node
// Tokenize TIL sources with the extension's TextMate grammar, exactly as
// VS Code does (vscode-textmate + vscode-oniguruma, dev dependencies).
//
//   node test/tokenize.js < request.json
//   request: {"grammar": "path/to/til.tmLanguage.json", "samples": {"name": "source text", ...},
//             "regexes": ["...", ...]}            (optional: regexes that must compile)
//   answer:  {"samples": {"name": [[{"text", "scopes": [...]}, ...] per line]}, "regexes": [errors]}
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
let textmate, oniguruma;
try {
    textmate = require(path.join(root, 'node_modules', 'vscode-textmate'));
    oniguruma = require(path.join(root, 'node_modules', 'vscode-oniguruma'));
} catch (e) {
    process.stdout.write(JSON.stringify({ missing: String(e && e.message) }) + '\n');
    process.exit(3);
}

async function main() {
    const request = JSON.parse(fs.readFileSync(0, 'utf8'));
    const wasm = fs.readFileSync(path.join(root, 'node_modules', 'vscode-oniguruma', 'release', 'onig.wasm'));
    await oniguruma.loadWASM(wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength));
    const onigLib = Promise.resolve({
        createOnigScanner: (patterns) => new oniguruma.OnigScanner(patterns),
        createOnigString: (s) => new oniguruma.OnigString(s),
    });
    const grammarPath = request.grammar;
    const registry = new textmate.Registry({
        onigLib,
        loadGrammar: async (scopeName) => {
            if (scopeName !== 'source.til') return null;
            return textmate.parseRawGrammar(fs.readFileSync(grammarPath, 'utf8'), grammarPath);
        },
    });
    const grammar = await registry.loadGrammar('source.til');
    const out = { samples: {}, regexes: [] };
    for (const [name, text] of Object.entries(request.samples || {})) {
        let state = textmate.INITIAL;
        const lines = [];
        for (const line of text.split(/\r?\n/)) {
            const r = grammar.tokenizeLine(line, state);
            lines.push(r.tokens.map((t) => ({
                text: line.substring(t.startIndex, t.endIndex),
                scopes: t.scopes,
            })));
            state = r.ruleStack;
        }
        out.samples[name] = lines;
    }
    for (const re of request.regexes || []) {
        try {
            const scanner = new oniguruma.OnigScanner([re]);
            scanner.dispose();
        } catch (e) {
            out.regexes.push({ regex: re, error: String(e && e.message) });
        }
    }
    process.stdout.write(JSON.stringify(out) + '\n');
}

main().catch((e) => {
    process.stderr.write(String(e && e.stack || e) + '\n');
    process.exit(1);
});
