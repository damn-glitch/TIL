#!/usr/bin/env node
// mock_server.mjs - a stand-in for src/tilstudio.py (docs/dev/TIL_STUDIO_DESIGN.md
// section 5) for developing and testing the frontend without Python: the same
// endpoints, token / Host / Origin checks and SSE events, with plausible canned
// behaviour - real files in a project folder, a language server (the web
// face's, plus fake compiler diagnostics, inlay hints and Debug/Prove lenses),
// fake runs that stream output / ask for input / draw frames, a fake debug
// adapter with a recorded timeline, and canned prove / fit / dis / build / pkg.
// The payloads follow src/tilstudio.py: run events status {state, backend},
// exit {code, reason, seconds} (no text for people), 409 for scripted input,
// 429 beyond 4 runs, read-only "@examples/...", mtimes in float seconds,
// search matches with 1-based line/col and length, dis offsets/fused/lineStart,
// prove results with exact big integers as decimal text.
//
//   node ide/studio/test/mock_server.mjs [--port N] [--root DIR] [--token T]
//
// No dependencies beyond Node 22 (the language server modules of the frontend
// are plain ES modules).
import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { WebLanguageServer } from '../src/backend/weblsp.js';
import { scan, occurrences, codeOnly, KEYWORD_TYPE, ALIAS_OF, TABLES, levenshtein, functionAt } from '../src/backend/lang_scan.js';
import { translateSource } from '../src/backend/translate.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const STUDIO = path.resolve(HERE, '..');
const DEFAULT_HTML = path.resolve(STUDIO, '..', 'studio.html');
const CATALOGS = Object.fromEntries(['en', 'ru', 'kk'].map((l) => [l, JSON.parse(fs.readFileSync(path.join(STUDIO, 'i18n', `${l}.json`), 'utf8'))]));
let mockLocale = 'en';
const tr = (key, params) => {
    let s = (CATALOGS[mockLocale] || CATALOGS.en)[key] || CATALOGS.en[key] || key;
    if (params) s = s.replace(/\{(\w+)\}/g, (m, k) => (params[k] !== undefined ? String(params[k]) : m));
    return s;
};

/* what the demo programs print (the mock does not run TIL) */
const CANNED = {
    'main.til': ['Hello, TIL Studio!', 'total = 31', 'square of 7 = 49', 'step 0', 'step 1', 'step 2', 'done'],
    'proofs.til': ['10', '5', '55'],
    'kazakh.til': ['Сәлем, Әлем!', '7 15'],
    'examples/hello.til': ['Hello, World!'],
    'examples/structs.til': ['5'],
};

export const DEVICES = [
    ['desktop', 'Desktop PC, the default C VM build'], ['ti84ce', 'TI-84 Plus CE graphing calculator'],
    ['numworks', 'NumWorks graphing calculator (external app)'], ['nspire', 'TI-Nspire CX / CX II (Ndless)'],
    ['casio-cg50', 'Casio fx-CG50 / Graph 90+E (gint add-in)'], ['cortex-m3-64k', 'ARM Cortex-M3 board, 64 KB RAM / 256 KB flash'],
    ['atmega1284', 'ATmega1284P, 16 KB RAM / 128 KB flash'], ['arduino-mega', 'Arduino Mega 2560 (ATmega2560), 8 KB RAM / 256 KB flash'],
    ['esp32', 'ESP32 (ESP32-WROOM-32), 520 KB SRAM / 4 MB flash'], ['rp2040', 'Raspberry Pi Pico (RP2040), 264 KB SRAM / 2 MB flash'],
    ['z80-16', 'Z80 calculator class (TI-83 Plus / TI-84 Plus), 16-bit lean build'],
].map(([name, title]) => ({ name, title }));

/* ── the demo project ───────────────────────────────────────────────── */
export const PROJECT = {
    'main.til': `# main.til - the TIL Studio demo project
import math_utils

# greet someone by name
greet(name: str) -> str
    return f"Hello, {name}!"

total(values: [int]) -> int
    var sum = 0
    for v in values
        sum += v
    return sum

main()
    print(greet("TIL Studio"))
    let numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"total = {total(numbers)}")
    print(f"square of 7 = {square(7)}")
    for i in 0..3
        print(f"step {i}")
    print("done")
`,
    'math_utils.til': `# math helpers, imported by main.til

square(x: int) -> int
    return x * x

cube(x: int) -> int
    return x * x * x
`,
    'graphics.til': `# a square you move with the arrow keys
const W = 96
const H = 64

main()
    screen_init(W, H)
    var x = 40
    var y = 24
    loop
        screen_clear(0x1E3A5F)
        fill_rect(x, y, 16, 16, 0xFBBF24)
        draw_text(2, 2, "TIL", 0xFFFFFF)
        screen_show()
        let k = wait_key()
        if k == 27
            break
        elif k == 3
            x -= 4
        elif k == 4
            x += 4
`,
    'input.til': `# input(): a small conversation
main()
    print("What is your name?")
    let name = input("> ")
    print(f"Nice to meet you, {name}!")
`,
    'proofs.til': `# contracts that til prove checks for every input

#[requires: lo <= hi]
#[ensures: lo <= result and result <= hi]
clamp(x: int, lo: int, hi: int) -> int
    if x < lo
        return lo
    elif x > hi
        return hi
    return x

#[requires: a <= b]
#[ensures: a <= result and result <= b]
midpoint(a: int, b: int) -> int
    return (a + b) / 2

#[requires: n >= 0]
#[ensures: result >= 0]
sum_to(n: int) -> int
    var s = 0
    var i = 0
    while i < n
        i += 1
        s += i
    return s

main()
    print(clamp(15, 0, 10))
    print(midpoint(2, 8))
    print(sum_to(10))
`,
    'kazakh.til': `# Қазақша бағдарлама
қосу(а: int, б: int) -> int
    қайтару а + б

негізгі()
    басып_шығару("Сәлем, Әлем!")
    айнымалы қосынды = 0
    үшін и ішінде 1..=5
        қосынды += и
    басып_шығару(қосу(3, 4), қосынды)
`,
    'til.json': `{\n  "name": "demo",\n  "version": "0.1.0",\n  "dependencies": {\n    "geo": "./libs/geo"\n  }\n}\n`,
    // two local packages (path dependencies work with the real `til pkg` too)
    'libs/geo/til.json': `{\n  "name": "geo",\n  "version": "1.2.0"\n}\n`,
    'libs/geo/geo.til': '# geo - distances on a grid\ndistance2(x: int, y: int) -> int\n    return x * x + y * y\n',
    'libs/colors/til.json': `{\n  "name": "colors",\n  "version": "1.0.0"\n}\n`,
    'libs/colors/colors.til': '# colors - RGB constants for the screen\nconst RED = 0xFF0000\nconst GREEN = 0x00FF00\n',
    'README.md': '# Demo project\n\nOpen `main.til` and press Ctrl+F5.\n',
    'examples/hello.til': '# Hello, World\nmain()\n    print("Hello, World!")\n',
    'examples/structs.til': `struct Point
    x: float
    y: float

impl Point
    new(x: float, y: float) -> Point
        return Point { x: x, y: y }

    length(self) -> float
        return sqrt(self.x * self.x + self.y * self.y)

main()
    let p = Point.new(3.0, 4.0)
    print(p.length())
`,
};

/* ── helpers ────────────────────────────────────────────────────────── */
function json(res, status, data) {
    const body = JSON.stringify(data);
    res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
    res.end(body);
}

function readBody(req) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        req.on('data', (c) => chunks.push(c));
        req.on('end', () => {
            const text = Buffer.concat(chunks).toString('utf8');
            if (!text) { resolve({}); return; }
            try { resolve(JSON.parse(text)); } catch (e) { reject(Object.assign(new Error('bad JSON'), { status: 400 })); }
        });
        req.on('error', reject);
    });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function posix(p) { return String(p || '').replace(/\\/g, '/').replace(/^\/+/, '').replace(/\/+$/, ''); }

/* ── fake compiler diagnostics ──────────────────────────────────────── */
function fakeDiagnostics(text) {
    const info = scan(text);
    const lines = info.lines;
    const diags = [];
    const known = new Set([...TABLES.natives, ...ALIAS_OF.keys(), ...TABLES.types, 'Some', 'Ok', 'Err', 'None', 'resume', 'print']);
    for (const s of info.symbols) known.add(s.name);
    for (const s of info.symbols) if (s.kind === 'impl' || s.kind === 'struct') known.add(s.typeName);
    // imported modules of the demo project
    for (const s of scan(PROJECT['math_utils.til']).symbols) known.add(s.name);
    const hint = (name) => {
        let best = null, bd = 3;
        for (const c of [...TABLES.natives, ...info.symbols.filter((s) => s.kind === 'function').map((s) => s.name)]) {
            const d = levenshtein(name, c);
            if (d < bd) { bd = d; best = c; }
        }
        return best;
    };
    for (let i = 0; i < lines.length; i++) {
        const code = codeOnly(lines[i]);
        if (/^\s*#/.test(lines[i]) && !/^\s*#\[/.test(lines[i])) continue;
        const re = /(?<![\p{L}\p{N}_.])([\p{L}_][\p{L}\p{N}_]*)\s*\(/gu;
        let m;
        while ((m = re.exec(code))) {
            const name = m[1];
            if (known.has(name) || KEYWORD_TYPE.has(name)) continue;
            const fnAt = functionAt(info, i);
            if (fnAt && fnAt.line === i) continue;
            if (info.locals.some((l) => l.name === name)) continue;
            const h = hint(name);
            diags.push({
                range: { start: { line: i, character: m.index }, end: { line: i, character: m.index + name.length } },
                severity: 1, code: 'E-VM', source: 'til',
                message: `unknown function '${name}'` + (h ? `\ndid you mean '${h}'?` : ''),
                data: h ? { en: `unknown function '${name}'`, hint: `did you mean '${h}'?`, enHint: `did you mean '${h}'?` } : { en: `unknown function '${name}'` },
            });
        }
    }
    // a let assigned twice
    for (const l of info.locals) {
        if (l.kind !== 'let') continue;
        const fn = l.fn;
        for (const o of occurrences(lines, l.name, l.line + 1, fn ? fn.endLine : lines.length - 1)) {
            const after = codeOnly(lines[o.line]).slice(o.col + l.name.length);
            const before = codeOnly(lines[o.line]).slice(0, o.col);
            if (/^\s*([+\-*/%]?=)(?!=)/.test(after) && !before.trim()) {
                diags.push({
                    range: { start: { line: o.line, character: o.col }, end: { line: o.line, character: o.col + l.name.length } },
                    severity: 1, code: 'E-VM', source: 'til',
                    message: `cannot assign twice to immutable variable '${l.name}'\ndeclare it with 'var' to make it mutable`,
                    data: { en: `cannot assign twice to immutable variable '${l.name}'`, hint: "declare it with 'var' to make it mutable", enHint: "declare it with 'var' to make it mutable" },
                });
            }
        }
        // never used
        const uses = occurrences(lines, l.name, l.line, fn ? fn.endLine : lines.length - 1).filter((o) => !(o.line === l.line && o.col === l.col));
        if (!uses.length && !l.name.startsWith('_')) {
            diags.push({
                range: { start: { line: l.line, character: l.col }, end: { line: l.line, character: l.col + l.name.length } },
                severity: 2, code: 'W-UNUSED', source: 'til', message: `variable '${l.name}' is never used`, data: { en: `variable '${l.name}' is never used` },
            });
        }
    }
    return diags;
}

/* ── fake runs ──────────────────────────────────────────────────────── */
function printedTexts(source) {
    const out = [];
    const lines = source.split('\n');
    const printRe = /\b(print|println|печать|вывод|басып_шығару|шығару|basyp_shygaru)\s*\((.*)\)\s*$/u;
    lines.forEach((l, i) => {
        if (/^\s*#/.test(l)) return;
        const m = printRe.exec(l.trim());
        if (m) {
            let arg = m[2].trim();
            const lit = /^f?"((?:\\.|[^"\\])*)"$/.exec(arg) || /^f?'((?:\\.|[^'\\])*)'$/.exec(arg);
            if (lit) arg = lit[1].replace(/\{([^}]*)\}/g, (x, e) => (/^\w+$/.test(e) ? `<${e}>` : '…')).replace(/\\n/g, '\n');
            out.push({ line: i + 1, text: arg });
        }
        if (/\binput\s*\(|\bввод\s*\(|\bенгізу\s*\(/.test(l)) out.push({ line: i + 1, input: true });
        if (/\bpanic\s*\(/.test(l)) out.push({ line: i + 1, panic: true });
    });
    return out;
}

/* the canned output of a project file spread over its print lines (a print
   inside a loop takes the extra lines), so the debugger prints what a run does */
function cannedPrints(rel, text, prints) {
    const canned = CANNED[rel];
    const shown = prints.filter((p) => p.text !== undefined);
    if (!canned || text !== PROJECT[rel] || !shown.length || shown.length > canned.length) return prints;
    const lines = text.split('\n');
    const indent = (l) => l.length - l.trimStart().length;
    const inLoop = (n) => {
        const own = indent(lines[n - 1]);
        for (let i = n - 2, lim = own; i >= 0 && lim > 0; i--) {
            if (!lines[i].trim()) continue;
            if (indent(lines[i]) < lim) { if (/^\s*(for|while|loop)\b/.test(lines[i])) return true; lim = indent(lines[i]); }
        }
        return false;
    };
    let extra = canned.length - shown.length, k = 0;
    for (const p of shown) {
        const take = extra > 0 && inLoop(p.line) ? 1 + extra : 1;
        if (take > 1) extra = 0;
        p.text = canned.slice(k, k + take).join('\n');
        k += take;
    }
    return prints;
}

/* ── the fake debug adapter ─────────────────────────────────────────── */
function buildTrace(info) {
    const lines = info.lines;
    const ind = (s) => s.length - s.trimStart().length;
    const isCode = (i) => lines[i] !== undefined && lines[i].trim() && !/^\s*#/.test(lines[i]);
    const fns = new Map(info.symbols.filter((s) => s.kind === 'function').map((s) => [s.name, s]));
    const trace = [];
    const body = (fn) => {
        const out = [];
        for (let i = fn.line + 1; i <= (fn.endLine ?? fn.line); i++) if (isCode(i)) out.push(i);
        return out;
    };
    const walk = (lineList, fnName, depth, stackLines) => {
        for (let k = 0; k < lineList.length && trace.length < 600; k++) {
            const i = lineList[k];
            const text = codeOnly(lines[i]);
            const w = (text.trim().match(/^[\p{L}_]+/u) || [''])[0];
            const ty = KEYWORD_TYPE.get(w);
            trace.push({ line: i + 1, fn: fnName, stack: stackLines });
            // calls into user functions
            if (depth < 3) {
                const callRe = /(?<![\p{L}\p{N}_.])([\p{L}_][\p{L}\p{N}_]*)\s*\(/gu;
                let m;
                while ((m = callRe.exec(text))) {
                    const f = fns.get(m[1]);
                    if (f && f.name !== fnName && f.canonical !== 'main') walk(body(f), f.name, depth + 1, [...stackLines, { fn: fnName, line: i + 1 }]);
                }
            }
            if (ty === 'FOR' || ty === 'WHILE' || ty === 'LOOP') {
                const inner = [];
                let j = k + 1;
                while (j < lineList.length && ind(lines[lineList[j]]) > ind(lines[i])) { inner.push(lineList[j]); j++; }
                for (let rep = 0; rep < 3; rep++) {
                    walk(inner, fnName, depth, stackLines);
                    if (rep < 2) trace.push({ line: i + 1, fn: fnName, stack: stackLines });
                }
                k = j - 1;
            }
        }
    };
    const main = info.symbols.find((s) => s.kind === 'function' && (s.canonical === 'main' || s.name === 'main'));
    if (main) walk(body(main), main.name, 0, []);
    else {
        const top = [];
        for (let i = 0; i < lines.length; i++) if (isCode(i) && ind(lines[i]) === 0 && !info.symbols.some((s) => s.line === i)) top.push(i);
        walk(top, '<top>', 0, []);
    }
    return trace;
}

class FakeDap {
    constructor(server, id) {
        this.server = server;
        this.id = id;
        this.seq = 1;
        this.bps = new Map();
        this.index = 0;
        this.trace = [];
        this.state = 'init';
        this.printed = 0;
    }

    event(name, body) { return { seq: this.seq++, type: 'event', event: name, body }; }

    send(msg) { this.server.broadcast('dap', { sessionId: this.id, message: msg }); }

    respond(req, body, success = true, message) {
        const r = { seq: this.seq++, type: 'response', request_seq: req.seq, command: req.command, success };
        if (body !== undefined) r.body = body;
        if (message) r.message = message;
        return r;
    }

    async handle(req) {
        const a = req.arguments || {};
        const out = [];
        switch (req.command) {
        case 'initialize':
            out.push(this.respond(req, {
                supportsConfigurationDoneRequest: true, supportsStepBack: true, supportsEvaluateForHovers: true,
                supportsConditionalBreakpoints: true, supportsHitConditionalBreakpoints: true, supportsTerminateRequest: true,
                supportsLoadedSourcesRequest: false, supportsExceptionInfoRequest: true, exceptionBreakpointFilters: [{ filter: 'runtime-error', label: 'Runtime errors', default: true }],
            }));
            break;
        case 'launch': {
            const program = String(a.program || '');
            const rel = this.server.rel(program);
            let text = typeof a.source === 'string' ? a.source : '';
            if (typeof a.source !== 'string') {
                // a TIL example by its "@examples/..." path (the real server hands the adapter the file)
                try { text = this.server.virtual(rel) ? await this.server.read(rel) : await fsp.readFile(this.server.abs(rel), 'utf8'); } catch (e) {
                    out.push(this.respond(req, undefined, false, `cannot read ${program}`));
                    break;
                }
            }
            this.program = program;
            this.rel = rel;
            this.info = scan(text);
            this.trace = buildTrace(this.info);
            this.prints = cannedPrints(rel, text, printedTexts(text));
            this.stopOnEntry = Boolean(a.stopOnEntry);
            out.push(this.respond(req, {}));
            setTimeout(() => this.send(this.event('initialized', {})), 10);
            break;
        }
        case 'setBreakpoints': {
            const p = this.server.rel(a.source && a.source.path || '');
            const list = (a.breakpoints || (a.lines || []).map((line) => ({ line }))).map((b) => ({ line: b.line, condition: b.condition }));
            this.bps.set(p, list);
            out.push(this.respond(req, { breakpoints: list.map((b) => ({ verified: true, line: b.line })) }));
            break;
        }
        case 'setExceptionBreakpoints': out.push(this.respond(req, {})); break;
        case 'configurationDone':
            out.push(this.respond(req, {}));
            this.state = 'running';
            setTimeout(() => this.runForward(this.stopOnEntry ? 'entry' : null, -1), 30);
            break;
        case 'threads': out.push(this.respond(req, { threads: [{ id: 1, name: 'main' }] })); break;
        case 'stackTrace': out.push(this.respond(req, { stackFrames: this.frames(), totalFrames: this.frames().length })); break;
        case 'scopes': out.push(this.respond(req, { scopes: [{ name: 'Locals', variablesReference: 1, expensive: false }, { name: 'Globals', variablesReference: 2, expensive: false }] })); break;
        case 'variables': out.push(this.respond(req, { variables: this.variables(a.variablesReference) })); break;
        case 'evaluate': {
            const v = this.lookup(String(a.expression || '').trim());
            if (v) out.push(this.respond(req, { result: v.value, type: v.type, variablesReference: v.variablesReference || 0 }));
            else out.push(this.respond(req, undefined, false, `unknown name '${a.expression}'`));
            break;
        }
        case 'continue': out.push(this.respond(req, { allThreadsContinued: true })); setTimeout(() => this.runForward(null, this.index), 20); break;
        case 'next': case 'stepIn': out.push(this.respond(req, {})); setTimeout(() => this.stepTo(this.index + 1, 'step'), 15); break;
        case 'stepOut': {
            out.push(this.respond(req, {}));
            const fn = (this.trace[this.index] || {}).fn;
            let j = this.index + 1;
            while (j < this.trace.length && this.trace[j].fn === fn) j++;
            setTimeout(() => this.stepTo(j, 'step'), 15);
            break;
        }
        case 'stepBack': out.push(this.respond(req, {})); setTimeout(() => this.stepTo(Math.max(0, this.index - 1), 'step', true), 15); break;
        case 'reverseContinue': {
            out.push(this.respond(req, {}));
            let j = this.index - 1;
            while (j > 0 && !this.isBreak(j)) j--;
            setTimeout(() => this.stepTo(Math.max(0, j), j > 0 ? 'breakpoint' : 'entry', true), 15);
            break;
        }
        case 'pause': out.push(this.respond(req, {})); setTimeout(() => this.stop('pause'), 10); break;
        case 'til/timeline': out.push(this.respond(req, { index: this.index, total: this.trace.length, complete: true })); break;
        case 'exceptionInfo': out.push(this.respond(req, { exceptionId: 'runtime-error', description: 'runtime error: panic', breakMode: 'always' })); break;
        case 'til/goto': {
            const idx = Math.max(0, Math.min(this.trace.length - 1, Number(a.index) || 0));
            out.push(this.respond(req, {}));
            setTimeout(() => { this.send(this.event('continued', { threadId: 1, allThreadsContinued: true })); this.stepTo(idx, 'goto', idx < this.index); }, 10);
            break;
        }
        case 'til/stdin': out.push(this.respond(req, {})); if (this.waitingInput) { this.waitingInput = false; this.send(this.event('output', { category: 'stdout', output: `Nice to meet you, ${String(a.text || '').trim()}!\n` })); } break;
        case 'terminate': case 'disconnect':
            out.push(this.respond(req, {}));
            if (this.state !== 'ended') { this.state = 'ended'; setTimeout(() => { this.send(this.event('terminated', {})); this.send(this.event('exited', { exitCode: 0 })); }, 5); }
            break;
        default:
            out.push(this.respond(req, undefined, false, `unknown request ${req.command}`));
        }
        return out;
    }

    isBreak(j) {
        const list = this.bps.get(this.rel) || [];
        const t = this.trace[j];
        return t && list.some((b) => b.line === t.line);
    }

    emitOutputsUpTo(j) {
        // program output for the print lines passed (never twice after going back)
        const maxLine = this.trace[j] ? this.trace[j].line : Infinity;
        while (this.printed < this.prints.length && this.prints[this.printed].line < maxLine && this.printedTrace(j)) {
            const p = this.prints[this.printed++];
            if (p.text !== undefined) this.send(this.event('output', { category: 'stdout', output: p.text + '\n' }));
            if (p.input) { this.waitingInput = true; this.send(this.event('til/inputRequested', {})); }
        }
    }

    printedTrace() { return true; }

    runForward(reason, from) {
        if (this.state === 'ended') return;
        for (let j = from + 1; j < this.trace.length; j++) {
            if ((reason === 'entry' && j === 0) || this.isBreak(j)) { this.stepTo(j, reason || 'breakpoint'); return; }
        }
        this.emitOutputsUpTo(this.trace.length);
        this.index = Math.max(0, this.trace.length - 1);
        this.state = 'ended';
        this.send(this.event('terminated', {}));
        this.send(this.event('exited', { exitCode: 0 }));
    }

    stepTo(j, reason, back = false) {
        if (this.state === 'ended') return;
        if (j >= this.trace.length) { this.runForward(null, this.trace.length); return; }
        this.index = j;
        if (!back) this.emitOutputsUpTo(j);
        this.stop(reason);
    }

    stop(reason) {
        this.state = 'stopped';
        this.send(this.event('stopped', { reason, threadId: 1, allThreadsStopped: true }));
    }

    frames() {
        const t = this.trace[this.index] || { line: 1, fn: 'main', stack: [] };
        const src = { path: this.program, name: this.rel };
        const frames = [{ id: 1000, name: t.fn, line: t.line, column: 1, source: src }];
        t.stack.slice().reverse().forEach((s, k) => frames.push({ id: 1001 + k, name: s.fn, line: s.line, column: 1, source: src }));
        return frames;
    }

    locals() {
        const t = this.trace[this.index] || {};
        const out = [];
        for (const l of this.info.locals) {
            if (!l.fn || l.fn.name !== t.fn || l.line + 1 > t.line) continue;
            if (l.line + 1 === t.line && l.kind !== 'param') continue;
            out.push(l);
        }
        return out;
    }

    valueOf(l) {
        const v = String(l.value || '');
        const n = (this.index % 7) + 1;
        if (l.kind === 'param') return l.type === 'str' ? { value: '"TIL Studio"', type: 'str' } : l.type && l.type.startsWith('[') ? { value: '[3, 1, 4, 1, 5, …]', type: l.type, variablesReference: 10 } : { value: String(n + 5), type: l.type || 'int' };
        if (l.kind === 'loop') return { value: String(this.index % 3), type: 'int' };
        if (/^\[/.test(v)) return { value: v.length > 40 ? v.slice(0, 38) + '…]' : v, type: '[int]', variablesReference: 10 };
        if (/^f?"/.test(v)) return { value: v.replace(/^f/, ''), type: 'str' };
        if (/^\d+$/.test(v)) return { value: String(Number(v) + (l.kind === 'var' ? this.index : 0)), type: 'int' };
        if (/^[A-Z]\w*\s*\{|\.new\(/.test(v)) return { value: v.split(/[{(.]/)[0].trim() + ' {…}', type: v.split(/[{(.]/)[0].trim(), variablesReference: 11 };
        return { value: String(n * 3), type: 'int' };
    }

    variables(ref) {
        if (ref === 1) return this.locals().map((l) => Object.assign({ name: l.name }, this.valueOf(l))).map((v) => Object.assign({ variablesReference: 0 }, v));
        if (ref === 2) return this.info.symbols.filter((s) => s.kind === 'const' || s.kind === 'global').map((s) => ({ name: s.name, value: s.value || '0', type: /^0x|^\d/.test(s.value || '') ? 'int' : 'str', variablesReference: 0 }));
        if (ref === 10) return [3, 1, 4, 1, 5, 9, 2, 6].map((x, i) => ({ name: `[${i}]`, value: String(x), type: 'int', variablesReference: 0 }));
        if (ref === 11) return [{ name: 'x', value: '3.0', type: 'float', variablesReference: 0 }, { name: 'y', value: '4.0', type: 'float', variablesReference: 0 }];
        return [];
    }

    lookup(expr) {
        const m = /^([\p{L}_][\p{L}\p{N}_]*)(?:\[(\d+)\])?$/u.exec(expr);
        if (!m) return /^[\d\s+\-*/()]+$/.test(expr) ? { value: String(Function(`return (${expr})`)()), type: 'int' } : null;
        const all = this.variables(1).concat(this.variables(2));
        const v = all.find((x) => x.name === m[1]);
        if (!v) return null;
        if (m[2] !== undefined && v.variablesReference) {
            const el = this.variables(v.variablesReference)[Number(m[2])];
            return el || null;
        }
        return v;
    }
}

/* the TIL distribution's examples: the read-only virtual folder "@examples" */
export const EXAMPLES = {
    'hello.til': '# hello.til - the classic\nprint("Hello, World!")\n',
    'structs.til': PROJECT['examples/structs.til'],
};
const MAX_RUNS = 4;
const FUSED_OPS = new Set(TABLES.superinstructions || []);
const KEY_NAMES = { UP: 1, DOWN: 2, LEFT: 3, RIGHT: 4, ENTER: 13, ESC: 27, ESCAPE: 27, SPACE: 32, BACKSPACE: 8, TAB: 9 };

/* ── the server ─────────────────────────────────────────────────────── */
export async function startMockServer({ root, port = 0, token = crypto.randomBytes(12).toString('hex'), html = DEFAULT_HTML, seed = true, latency = 0 } = {}) {
    if (!root) root = await fsp.mkdtemp(path.join(os.tmpdir(), 'til-studio-mock-'));
    root = path.resolve(root);
    if (seed) {
        for (const [p, text] of Object.entries(PROJECT)) {
            const f = path.join(root, p);
            if (fs.existsSync(f)) continue;
            await fsp.mkdir(path.dirname(f), { recursive: true });
            await fsp.writeFile(f, text);
        }
    }
    const clients = new Set();
    const runs = new Map();
    const daps = new Map();
    let runSeq = 0, dapSeq = 0;
    const stats = { requests: [], lsp: [], dap: [], runs: [], lang: '', clients: { requests: [], events: [] } };
    const state = { root, token, stats, clients, runs, daps };

    const S = {
        broadcast(name, data) {
            const msg = `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
            for (const c of clients) c.write(msg);
        },
        rel(p) {
            const s = String(p || '').replace(/\\/g, '/');
            const r = root.replace(/\\/g, '/');
            if (s.startsWith(r + '/')) return s.slice(r.length + 1);
            return posix(s);
        },
        virtual(rel) { return /^@examples(\/|$)/.test(String(rel || '')); },
        /* the text of a project file or of an example */
        async read(rel) {
            if (S.virtual(rel)) {
                const name = String(rel).replace(/^@examples\/?/, '');
                if (!Object.prototype.hasOwnProperty.call(EXAMPLES, name)) throw Object.assign(new Error(`no such file or folder: ${rel}`), { status: 404 });
                return EXAMPLES[name];
            }
            return fsp.readFile(S.abs(rel), 'utf8');
        },
        abs(rel) {
            if (S.virtual(rel)) throw Object.assign(new Error(`read-only: ${rel}`), { status: 403 });
            const full = path.resolve(root, posix(rel));
            if (full !== root && !full.startsWith(root + path.sep)) throw Object.assign(new Error('path outside the project'), { status: 403 });
            return full;
        },
    };
    state.server = S;

    const lsp = new WebLanguageServer({
        emit: (m) => S.broadcast('lsp', m),
        diagnose: async (uri, text) => fakeDiagnostics(text),
        files: () => {
            const m = new Map();
            for (const f of ['main.til', 'math_utils.til']) {
                try { m.set(`file://${root}/${f}`, fs.readFileSync(path.join(root, f), 'utf8')); } catch (e) { /* */ }
            }
            return m;
        },
        t: tr,
        desktop: true,
        diagnosticsInReply: true,
        name: 'til-lsp (mock)',
    });

    async function tree(rel) {
        if (S.virtual(rel)) {
            return Object.entries(EXAMPLES).map(([name, text]) => ({ name, path: `@examples/${name}`, type: 'file', size: Buffer.byteLength(text), mtime: 1700000000.5, readonly: true }));
        }
        const dir = S.abs(rel || '');
        const names = await fsp.readdir(dir, { withFileTypes: true });
        const entries = [];
        for (const d of names) {
            if (d.name === '.git' || d.name === '__pycache__') continue;
            const st = await fsp.stat(path.join(dir, d.name));
            const entry = { name: d.name, path: posix(path.posix.join(posix(rel || ''), d.name)), type: st.isDirectory() ? 'dir' : 'file', size: st.isDirectory() ? 0 : st.size, mtime: st.mtimeMs / 1000 };
            if (st.isDirectory() && (d.name === 'til_modules' || d.name === 'node_modules')) entry.collapsed = true;
            if (d.isSymbolicLink()) entry.link = true;
            entries.push(entry);
        }
        entries.sort((a, b) => (a.type === b.type ? a.name.localeCompare(b.name) : a.type === 'dir' ? -1 : 1));
        return entries;
    }

    async function walkFiles(dir, out = []) {
        for (const d of await fsp.readdir(dir, { withFileTypes: true })) {
            if (d.name === '.git' || d.name === 'node_modules') continue;
            const f = path.join(dir, d.name);
            if (d.isDirectory()) await walkFiles(f, out); else out.push(f);
        }
        return out;
    }

    function startRun(opts) {
        const id = `r${++runSeq}`;
        stats.runs.push(opts);
        const backend = opts.backend || 'vm';
        const run = {
            id, stopped: false, done: false, stdin: [], eof: false, waiting: null, t0: Date.now(), backend, actual: backend,
            scriptedStdin: typeof opts.stdin === 'string', scriptedKeys: opts.keys !== undefined && opts.keys !== null,
        };
        runs.set(id, run);
        const ev = (kind, extra) => S.broadcast('run', Object.assign({ runId: id, kind }, extra || {}));
        // exit {code, reason: exited | stopped | failed | paused, seconds}; no text for people
        run.exit = (code, reason) => {
            if (run.done) return;
            run.done = true;
            ev('exit', { code, reason, seconds: Math.round(Date.now() - run.t0) / 1000 });
        };
        run.ev = ev;
        (async () => {
            let source = opts.source || '';
            if (opts.path) {
                try { source = await S.read(opts.path); } catch (e) { ev('stderr', { text: `til: cannot read ${opts.path}\n` }); run.exit(1, 'failed'); return; }
            }
            if (run.scriptedStdin) run.stdin.push(...String(opts.stdin).split('\n').filter((x, i, a) => i < a.length - 1 || x));
            ev('status', { state: 'starting', backend });
            await sleep(30);
            const diags = fakeDiagnostics(source).filter((d) => d.severity === 1);
            if (diags.length) {
                // compile errors: stderr text, then exit {code: 1, reason: failed}
                const d = diags[0];
                ev('stderr', { text: `error[${d.code}]: ${d.message.split('\n')[0]}\n --> ${opts.path || 'untitled.til'}:${d.range.start.line + 1}:${d.range.start.character + 1}\n` });
                run.exit(1, 'failed');
                return;
            }
            const screen = /\bscreen_init\s*\(/.test(source);
            if (screen && backend !== 'vm') ev('stderr', { text: '[TIL] the program uses the screen: running it on the TIL VM\n' });
            run.actual = screen ? 'vm' : backend;
            if (run.actual === 'native') { ev('status', { state: 'compiling', backend: 'native' }); await sleep(100); }
            if (run.stopped) return;
            ev('status', Object.assign({ state: 'running', backend: run.actual }, run.actual === 'vm' ? {} : { pid: 40000 + runSeq }));
            if (screen) { await graphicsRun(run, ev, source); return; }
            const rel = opts.path ? posix(opts.path) : null;
            const canned = rel && CANNED[rel];
            if (canned && source === PROJECT[rel]) {
                for (const line of canned) { if (run.stopped) return; ev('stdout', { text: line + '\n' }); await sleep(12); }
                if (run.actual === 'cvm') ev('stdout', { text: '\x1b[2m[C VM] 1 284 steps, heap 2.1 KB\x1b[0m\n' });
                run.exit(0, 'exited');
                return;
            }
            for (const p of printedTexts(source)) {
                if (run.stopped) return;
                if (p.panic) { ev('stderr', { text: `runtime error: panic\n  at main (line ${p.line})\n` }); run.exit(1, 'exited'); return; }
                if (p.input) {
                    if (!run.stdin.length && !run.eof && !run.scriptedStdin) {
                        ev('input', {});
                        await new Promise((resolve) => { run.waiting = resolve; });
                        if (run.stopped) return;
                    }
                    const line = String(run.stdin.shift() || '').replace(/\n$/, '');
                    run.lastInput = line;
                    continue;
                }
                let text = p.text;
                if (run.lastInput !== undefined) text = text.replace(/<name>/, run.lastInput);
                ev('stdout', { text: text + '\n' });
                await sleep(12);
            }
            if (run.actual === 'cvm') ev('stdout', { text: '\x1b[2m[C VM] 1 284 steps, heap 2.1 KB\x1b[0m\n' });
            run.exit(0, 'exited');
        })().catch((e) => { ev('stderr', { text: `[TIL] internal error: ${e.message}\n` }); run.exit(1, 'failed'); });
        return id;
    }

    async function graphicsRun(run, ev, source) {
        const m = /screen_init\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)/.exec(source);
        const val = (x, d) => { if (/^\d+$/.test(x)) return Number(x); const c = new RegExp(`const\\s+${x}\\s*=\\s*(\\d+)`).exec(source); return c ? Number(c[1]) : d; };
        const w = m ? val(m[1], 96) : 96, h = m ? val(m[2], 64) : 64;
        ev('stdout', { text: `screen ${w}x${h}\n` });
        let x = Math.floor(w / 2) - 8, y = Math.floor(h / 2) - 8;
        const frame = () => {
            const buf = Buffer.alloc(w * h * 3);
            for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
                const k = (j * w + i) * 3;
                const inSq = i >= x && i < x + 16 && j >= y && j < y + 16;
                buf[k] = inSq ? 0xfb : 0x1e; buf[k + 1] = inSq ? 0xbf : 0x3a; buf[k + 2] = inSq ? 0x24 : 0x5f;
            }
            return buf.toString('base64');
        };
        run.onKey = (code) => {
            if (code === 3) x -= 4; else if (code === 4) x += 4; else if (code === 1) y -= 4; else if (code === 2) y += 4;
            else if (code === 27) { ev('screen', { w, h, rgb: frame() }); run.exit(0, 'exited'); return; }
            x = Math.max(0, Math.min(w - 16, x)); y = Math.max(0, Math.min(h - 16, y));
            ev('screen', { w, h, rgb: frame() });
        };
        ev('screen', { w, h, rgb: frame() });
        // the program waits for keys until Esc or Stop
    }

    async function handleApi(req, res, url) {
        const p = url.pathname;
        const method = req.method;
        stats.requests.push(`${method} ${p}`);
        if (req.headers['x-til-lang']) stats.lang = req.headers['x-til-lang'];      // the language of error messages
        // the page's id (X-TIL-Client; ?client= on the event stream): the real server keeps
        // a language server, debug sessions and runs per page
        if (req.headers['x-til-client']) stats.clients.requests.push(req.headers['x-til-client']);
        if (latency) await sleep(latency);
        if (p === '/api/events' && method === 'GET') {
            stats.clients.events.push(url.searchParams.get('client') || '');
            res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive' });
            // ping {time} at connect and every 15 s
            const pingMsg = () => `event: ping\ndata: ${JSON.stringify({ time: Date.now() / 1000 })}\n\n`;
            res.write(pingMsg());
            clients.add(res);
            const ping = setInterval(() => res.write(pingMsg()), 15000);
            req.on('close', () => { clearInterval(ping); clients.delete(res); });
            return;
        }
        if (p === '/api/info' && method === 'GET') {
            json(res, 200, {
                version: TABLES.version, root, rootName: path.basename(root), rootUri: `file://${root.split(path.sep).map(encodeURIComponent).join('/')}`,
                platform: process.platform, lang: null,
                capabilities: { native: true, cvm: true, z3: true, debugger: true, pkg: true, lsp: true, prover: true, fit: true },
                devices: DEVICES, examples: Object.keys(EXAMPLES).map((name) => ({ name: name.replace(/\.til$/, ''), path: `@examples/${name}` })),
                tables: TABLES, limits: { runs: MAX_RUNS, output: 4194304, search: 2000 },
            });
            return;
        }
        if (p === '/api/tree' && method === 'GET') { const rel = url.searchParams.get('path') || ''; json(res, 200, { path: rel, entries: await tree(rel) }); return; }
        if (p === '/api/file' && method === 'GET') {
            const rel = url.searchParams.get('path');
            if (S.virtual(rel)) {
                const text = await S.read(rel);
                json(res, 200, { path: rel, text, mtime: 1700000000.5, size: Buffer.byteLength(text), readonly: true });
                return;
            }
            const f = S.abs(rel);
            if (!fs.existsSync(f)) { json(res, 404, { error: `no such file or folder: ${rel}` }); return; }
            const buf = await fsp.readFile(f);
            if (buf.includes(0)) { json(res, 415, { error: `not a text file: ${rel}` }); return; }
            const st = await fsp.stat(f);
            json(res, 200, { path: posix(rel), text: buf.toString('utf8'), mtime: st.mtimeMs / 1000, size: st.size });
            return;
        }
        if (method !== 'GET') {
            const origin = req.headers.origin;
            const host = req.headers.host;
            if (origin && origin !== `http://${host}`) { json(res, 403, { error: 'bad Origin' }); return; }
        }
        const body = method === 'GET' ? {} : await readBody(req);
        if (p === '/api/file' && method === 'PUT') {
            const f = S.abs(body.path);
            if (body.mtime !== undefined && body.mtime !== null) {
                // 409 when the file changed on disk or was deleted since it was read
                if (!fs.existsSync(f)) { json(res, 409, { error: `the file changed on disk: ${body.path}` }); return; }
                const st = await fsp.stat(f);
                if (Math.abs(st.mtimeMs / 1000 - body.mtime) > 0.0005) { json(res, 409, { error: `the file changed on disk: ${body.path}` }); return; }
            }
            if (!fs.existsSync(path.dirname(f))) { json(res, 404, { error: `no such file or folder: ${body.path}` }); return; }
            await fsp.writeFile(f, body.text);
            const st = await fsp.stat(f);
            json(res, 200, { path: posix(body.path), mtime: st.mtimeMs / 1000, size: Buffer.byteLength(body.text) });
            return;
        }
        if (p === '/api/file/create') {
            const f = S.abs(body.path);
            if (fs.existsSync(f)) { json(res, 409, { error: `${body.path} exists` }); return; }
            if (body.type === 'dir') await fsp.mkdir(f, { recursive: true });
            else { await fsp.mkdir(path.dirname(f), { recursive: true }); await fsp.writeFile(f, body.text || ''); }
            json(res, 200, { path: posix(body.path) });
            S.broadcast('fs', { paths: [posix(body.path)] });
            return;
        }
        if (p === '/api/file/rename') {
            const to = S.abs(body.to);
            if (fs.existsSync(to)) { json(res, 409, { error: `${body.to} exists` }); return; }
            await fsp.mkdir(path.dirname(to), { recursive: true });
            await fsp.rename(S.abs(body.path), to);
            json(res, 200, { path: posix(body.to) });
            return;
        }
        if (p === '/api/file/delete') { await fsp.rm(S.abs(body.path), { recursive: true, force: true }); json(res, 200, {}); return; }
        if (p === '/api/search') {
            const files = await walkFiles(root);
            const matches = [];
            let re;
            try { re = new RegExp(body.regex ? body.query : String(body.query).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), body.case ? 'g' : 'gi'); } catch (e) { json(res, 400, { error: e.message }); return; }
            for (const f of files.sort()) {
                const rel = posix(path.relative(root, f));
                if (body.glob && !new RegExp('^' + body.glob.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*\*/g, '§').replace(/\*/g, '[^/]*').replace(/§/g, '.*') + '$').test(rel) && !rel.endsWith(body.glob.replace('*', ''))) continue;
                const text = await fsp.readFile(f, 'utf8');
                if (text.includes('\0')) continue;
                // every match: line and col 1-based (col in UTF-16 units), length
                text.split('\n').forEach((l, i) => {
                    for (const m of l.matchAll(re)) {
                        if (!m[0].length || matches.length >= 2000) break;
                        matches.push({ path: rel, line: i + 1, col: m.index + 1, length: m[0].length, text: l.slice(0, 1000) });
                    }
                });
            }
            json(res, 200, { matches, truncated: matches.length >= 2000 });
            return;
        }
        if (p === '/api/lsp') {
            stats.lsp.push(body.method || 'response');
            if (body.method === 'initialize' && body.params && CATALOGS[body.params.locale]) mockLocale = body.params.locale;
            if (body.method === 'workspace/didChangeConfiguration') {
                const loc = body.params && body.params.settings && body.params.settings.til && body.params.settings.til.locale;
                if (CATALOGS[loc]) mockLocale = loc;
            }
            json(res, 200, { messages: await lsp.handle(body) });
            return;
        }
        if (p === '/api/run') {
            if ([...runs.values()].filter((r) => !r.done).length >= MAX_RUNS) { json(res, 429, { error: `too many runs at once (at most ${MAX_RUNS}): stop one first` }); return; }
            json(res, 200, { runId: startRun(body) });
            return;
        }
        let m = /^\/api\/run\/([^/]+)\/(stdin|key|stop|snapshot)$/.exec(p);
        if (m) {
            const run = runs.get(decodeURIComponent(m[1]));
            if (!run) { json(res, 404, { error: 'no such run' }); return; }
            const wake = () => { if (run.waiting) { const w = run.waiting; run.waiting = null; w(); } };
            if (m[2] === 'stop') {
                if (!run.done) { run.stopped = true; wake(); setTimeout(() => run.exit(130, 'stopped'), 20); }
                json(res, 200, {});
                return;
            }
            if (run.done) { json(res, 409, { error: 'the run has ended' }); return; }
            if (m[2] === 'stdin') {
                if (run.scriptedStdin) { json(res, 409, { error: 'this run was given its whole input (stdin)' }); return; }
                if (typeof body.text !== 'string') { json(res, 400, { error: "missing or invalid field 'text'" }); return; }
                if (body.text) run.stdin.push(body.text.replace(/\n$/, ''));
                if (body.eof) run.eof = true;
                wake();
            } else if (m[2] === 'key') {
                const k = body.code;
                const code = typeof k === 'number' ? k : typeof k === 'string' ? (KEY_NAMES[k.toUpperCase()] || (k.length === 1 ? k.codePointAt(0) : NaN)) : NaN;
                if (!Number.isInteger(code)) { json(res, 400, { error: "missing or invalid field 'code'" }); return; }
                if (run.actual !== 'vm') { json(res, 409, { error: 'keys go to programs that run on the VM backend' }); return; }
                if (run.scriptedKeys) { json(res, 409, { error: 'this run plays a key script (keys)' }); return; }
                if (run.onKey) run.onKey(code);
            } else if (m[2] === 'snapshot') {
                if (run.actual !== 'vm') { json(res, 409, { error: 'snapshots need the vm backend' }); return; }
                await fsp.writeFile(path.join(root, 'main.tilp'), 'TILP mock snapshot\n');
                const snap = { path: 'main.tilp', size: 19, steps: 1284 };
                run.ev('status', { state: 'snapshot', path: snap.path, steps: snap.steps });
                if (body.stop) { run.stopped = true; wake(); setTimeout(() => run.exit(3, 'paused'), 10); }
                json(res, 200, snap);
                return;
            }
            json(res, 200, {});
            return;
        }
        if (p === '/api/dap/start') { const id = `d${++dapSeq}`; daps.set(id, new FakeDap(S, id)); json(res, 200, { sessionId: id }); return; }
        m = /^\/api\/dap\/([^/]+)$/.exec(p);
        if (m) {
            const d = daps.get(m[1]);
            if (!d) { json(res, 404, { error: 'no such debug session' }); return; }
            stats.dap.push(body.command);
            json(res, 200, { messages: await d.handle(body) });
            return;
        }
        if (p === '/api/tools/prove') { json(res, 200, await prove(body)); return; }
        if (p === '/api/tools/fit') { json(res, 200, await fit(body)); return; }
        if (p === '/api/tools/dis') { json(res, 200, await dis(body)); return; }
        if (p === '/api/tools/translate') {
            const text = body.source !== undefined ? body.source : await S.read(body.path);
            json(res, 200, { text: translateSource(text, body.to), to: body.to });
            return;
        }
        if (p === '/api/tools/build') {
            const name = path.basename(body.path).replace(/\.til$/, body.target === 'vm' ? '.tbc' : body.target === 'c-vm' ? '.c' : '');
            await fsp.mkdir(path.join(root, 'build'), { recursive: true });
            await fsp.writeFile(path.join(root, 'build', name || 'a.out'), 'mock build output\n');
            json(res, 200, { ok: true, rc: 0, output: `built build/${name} (${body.target}${body.device ? ', fits ' + body.device : ''})`, artifact: `build/${name}` });
            return;
        }
        if (p === '/api/pkg') { json(res, 200, await pkg(body.argv || [])); return; }
        json(res, 404, { error: `no such endpoint: ${method} ${p}` });
    }

    async function prove(body) {
        const text = body.source !== undefined ? body.source : await S.read(body.path);
        const info = scan(text);
        const results = [];
        for (const s of info.symbols) {
            if (s.kind !== 'function' || !(s.attrs || []).some((a) => /requires|ensures/.test(a))) continue;
            if (body.function && s.name !== body.function) continue;
            const bodyText = info.lines.slice(s.line + 1, (s.endLine ?? s.line) + 1).join('\n');
            if (/\(\s*\w+\s*\+\s*\w+\s*\)\s*\/\s*2/.test(bodyText)) {
                results.push({
                    function: s.name, title: s.name, line: s.line + 1, col: 1, status: 'counterexample', detail: '#[ensures] fails: (a + b) wraps around for large inputs',
                    obligations: 2, seconds: 0.05, errors: [],
                    failures: [{ line: s.line, col: 11, status: 'counterexample', what: 'a <= result', inputs: { a: '4611686018427387904', b: '4611686018427387904' }, confirmed: true, outcome: 'result = -4611686018427387904' }],
                    counterexample: { a: '4611686018427387904', b: '4611686018427387904' },
                });
            } else if (/\b(while|loop)\b/.test(bodyText) && !/invariant/.test(bodyText)) {
                results.push({ function: s.name, title: s.name, line: s.line + 1, col: 1, status: 'unsupported', detail: '', reason: 'a while loop needs an `invariant` to be proved', obligations: 0, seconds: 0, errors: [], failures: [] });
            } else {
                results.push({ function: s.name, title: s.name, line: s.line + 1, col: 1, status: 'proved', detail: `${(s.attrs || []).length} obligations proved in 0.04 s`, obligations: (s.attrs || []).length, seconds: 0.04, errors: [], failures: [] });
            }
        }
        await sleep(150);
        return { available: true, z3: '4.13.0', results };
    }

    async function fit(body) {
        const text = body.source !== undefined ? body.source : await S.read(body.path);
        const dev = DEVICES.find((d) => d.name === body.device) || DEVICES[1];
        const small = ['arduino-mega', 'z80-16', 'atmega1284'].includes(dev.name);
        const bc = 400 + text.length * 2;
        const limits = { ram: small ? 8192 : 98304, bc: small ? 6144 : 65505, stack: small ? 96 : 1024, frames: small ? 16 : 128 };
        const ramNeed = small ? 9400 : 53910 + bc;
        const rows = [
            { key: 'bytecode', resource: 'bytecode', needs: `${(bc / 1024).toFixed(1)} KB`, limit: `${(limits.bc / 1024).toFixed(1)} KB`, status: bc <= limits.bc ? 'ok' : 'FAIL', note: '', measured: null, need_value: bc, limit_value: limits.bc },
            { key: 'ram', resource: 'VM RAM', needs: `${(ramNeed / 1024).toFixed(1)} KB`, limit: `${limits.ram / 1024} KB`, status: ramNeed <= limits.ram ? 'ok' : 'FAIL', note: 'VM struct + C stack + bytecode', measured: null, need_value: ramNeed, limit_value: limits.ram },
            { key: 'consts', resource: 'constants', needs: '24', limit: '1024', status: 'ok', note: '', measured: null, need_value: 24, limit_value: 1024 },
            { key: 'ints', resource: 'integers', needs: '32-bit', limit: dev.name === 'z80-16' ? '16-bit' : '32-bit', status: dev.name === 'z80-16' ? 'FAIL' : 'ok', note: '', measured: null, need_value: null, limit_value: null },
            { key: 'stack', resource: 'value stack', needs: '22 slots', limit: String(limits.stack), status: 'ok', note: '', measured: body.run ? '18' : null, need_value: 22, limit_value: limits.stack },
            { key: 'frames', resource: 'call frames', needs: '3', limit: String(limits.frames), status: 'ok', note: '', measured: body.run ? '3' : null, need_value: 3, limit_value: limits.frames },
            { key: 'heap', resource: 'heap', needs: 'depends on data', limit: small ? '1.5 KB' : '32 KB', status: '-', note: '', measured: body.run ? '1.2 KB' : null, need_value: null, limit_value: small ? 1536 : 32768 },
        ];
        const errors = rows.filter((r) => r.status === 'FAIL').map((r) => `${r.resource}: needs ${r.needs}, the device has ${r.limit}`);
        const fits = !errors.length;
        await sleep(120);
        const lineOfLoop = text.split('\n').findIndex((l) => /\bfor\b|\bwhile\b|\bloop\b/.test(l)) + 1;
        return {
            program: body.path, device: { name: dev.name, title: dev.title, cpu: small ? 'AVR @ 16 MHz' : 'eZ80 @ 48 MHz', ram: limits.ram, storage: limits.bc, int_bits: dev.name === 'z80-16' ? 16 : 32, float: 1 },
            fits, verdict: fits ? 'fits' : `does not fit: ${errors.map((e) => e.split(':')[0]).join('; ')}`,
            errors, reasons: errors, warnings: lineOfLoop > 0 ? [`the loop at line ${lineOfLoop} in main keeps allocating: heap use grows with the number of iterations`] : [],
            notes: ['integers are 32-bit on this device: results beyond +/-2147483647 wrap around'], resources: rows,
            static: { bytecode_bytes: bc, constants: 24, functions: 4 },
            run: body.run ? { exit_code: 0, stdout: 'Hello, TIL Studio!\ntotal = 31\n', errors: '', stats: { steps: 1284, gc: 0, heap: 1200, peak_stack: 18, peak_frames: 3, peak_heap: 1229 }, timeout: false, error: null, desktop: { exit_code: 0, error: null, same_output: true } } : null,
        };
    }

    async function dis(body) {
        const text = body.source !== undefined ? body.source : await S.read(body.path);
        const info = scan(text);
        const fuse = body.fuse !== false;
        const instructions = [];
        const functions = [];
        const lines = [];
        let pc = 0;
        const fnsInOrder = info.symbols.filter((s) => s.kind === 'function' || s.kind === 'method').sort((a, b) => a.line - b.line);
        fnsInOrder.forEach((fn, fi) => {
            const start = pc;
            let localPc = 0;
            const add = (op, args, note, line) => {
                const lineStart = !lines.length || lines[lines.length - 1][1] !== line;
                instructions.push({ pc, offset: localPc, func: fn.name, op, args, note: note || '', line, fused: FUSED_OPS.has(op), lineStart });
                if (lineStart) lines.push([pc, line]);
                const size = 1 + args.length * (op.startsWith('JUMP') ? 2 : 1);
                localPc += size;
                pc += size;
            };
            for (let i = fn.line + 1; i <= (fn.endLine ?? fn.line); i++) {
                const l = info.lines[i];
                if (!l || !l.trim() || /^\s*#/.test(l)) continue;
                const c = codeOnly(l).trim();
                const ln = i + 1;
                if (/^(print|печать|басып_шығару)\(/u.test(c)) { add('PUSH_CONST', [instructions.length % 7], '"…"', ln); add(fuse ? 'NATIVE_POP' : 'NATIVE', [0, 1], 'print()', ln); if (!fuse) add('POP', [], '', ln); }
                else if (/^(let|var|айнымалы|тұрақты)\s/u.test(c)) { add('PUSH_I8', [3], '', ln); add('STORE_LOCAL', [1], '', ln); }
                else if (/^(return|қайтару)\b/u.test(c)) { add(fuse ? 'RETURN_LOCAL' : 'LOAD_LOCAL', [0], '', ln); if (!fuse) add('RETURN', [], '', ln); }
                else if (/^(if|elif|егер)\b/u.test(c)) { if (fuse) add('JUMP_IF_NOT_LI', [51, 0, 1, 4], 'LE -> 0014', ln); else { add('LOAD_LOCAL', [0], '', ln); add('PUSH_I8', [1], '', ln); add('LE', [], '', ln); add('JUMP_IF_FALSE', [4], '-> 0014', ln); } }
                else if (/^(for|үшін)\b/u.test(c)) { add('LOAD_LOCAL', [1], '', ln); add(fuse ? 'FOR_ITER_STORE' : 'FOR_ITER', [2, 12], '-> 0032', ln); }
                else if (/\+=/.test(c)) { add(fuse ? 'ADD_STORE_LOCAL' : 'ADD', [2], '', ln); }
                else { add('LOAD_LOCAL', [0], '', ln); add('CALL', [fi, 1], c.split('(')[0].slice(0, 20), ln); }
            }
            add('RETURN_NIL', [], '', fn.line + 1);
            functions.push({ name: fn.name, index: fi, start, end: pc, params: 1, locals: 2 });
        });
        const header = `; TIL bytecode v2: ${functions.length} functions, 24 constants, 0 globals, ${pc} bytes of code`;
        await sleep(60);
        return { text: header + '\n', functions, lines, instructions };
    }

    async function pkg(argv) {
        const manifestPath = path.join(root, 'til.json');
        let manifest = null;
        try { manifest = JSON.parse(await fsp.readFile(manifestPath, 'utf8')); } catch (e) { manifest = null; }
        const save = async () => fsp.writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
        const cmd = argv[0];
        if (cmd === 'init') {
            if (manifest) return { rc: 1, stdout: '', stderr: 'til pkg: error: til.json exists\n', manifest };
            manifest = { name: path.basename(root).replace(/\W/g, '_'), version: '0.1.0', dependencies: {} };
            await save();
            return { rc: 0, stdout: 'created til.json\n', stderr: '', manifest };
        }
        if (!manifest) return { rc: 1, stdout: '', stderr: 'til pkg: error: no til.json in the project folder (create one with `til pkg init`)\n', manifest: null };
        manifest.dependencies = manifest.dependencies || {};
        if (cmd === 'add') {
            const [name, spec] = [argv[1], argv[2] || '^1.0.0'];
            manifest.dependencies[name] = spec;
            await save();
            return { rc: 0, stdout: `added ${name} ${spec}\ninstalled ${name} 1.0.0\n`, stderr: '', manifest };
        }
        if (cmd === 'remove') {
            if (!manifest.dependencies[argv[1]]) return { rc: 1, stdout: '', stderr: `til pkg: error: ${argv[1]} is not a dependency\n`, manifest };
            delete manifest.dependencies[argv[1]];
            await save();
            return { rc: 0, stdout: `removed ${argv[1]}\n`, stderr: '', manifest };
        }
        if (cmd === 'install') return { rc: 0, stdout: Object.keys(manifest.dependencies).map((n) => `installed ${n} 1.0.0`).join('\n') + '\nwrote til.lock\n', stderr: '', manifest };
        if (cmd === 'verify') return { rc: 0, stdout: Object.keys(manifest.dependencies).map((n) => `ok       ${n} 1.0.0`).join('\n') + '\n', stderr: '', manifest };
        if (cmd === 'list') {
            const deps = Object.entries(manifest.dependencies);
            return { rc: 0, stdout: `${manifest.name} ${manifest.version} (${root})\n` + deps.map(([n, s], i) => `${i === deps.length - 1 ? '`--' : '|--'} ${n} 1.0.0  ${s}`).join('\n') + '\n', stderr: '', manifest };
        }
        return { rc: 2, stdout: '', stderr: `til pkg: error: unknown command ${cmd}\n`, manifest };
    }

    function authorized(req, url) {
        if (url.searchParams.get('token') === token) return true;
        if (req.headers['x-til-token'] === token) return true;
        const cookie = req.headers.cookie || '';
        return cookie.split(/;\s*/).some((c) => c === `til_studio_token=${token}`);
    }

    const server = http.createServer(async (req, res) => {
        const url = new URL(req.url, 'http://localhost');
        try {
            const host = req.headers.host || '';
            const okHost = new RegExp(`^(127\\.0\\.0\\.1|localhost):${server.address().port}$`).test(host);
            if (!okHost) { json(res, 403, { error: 'bad Host' }); return; }
            if (!authorized(req, url)) {
                if (url.pathname.startsWith('/api/')) json(res, 403, { error: 'missing or wrong token' });
                else { res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' }); res.end('TIL Studio: open the link printed by `til ide`'); }
                return;
            }
            if (url.pathname.startsWith('/api/')) { await handleApi(req, res, url); return; }
            if (url.pathname === '/' || url.pathname === '/index.html') {
                const headers = { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' };
                if (url.searchParams.get('token') === token) headers['Set-Cookie'] = `til_studio_token=${token}; HttpOnly; SameSite=Strict; Path=/`;
                res.writeHead(200, headers);
                res.end(fs.readFileSync(html));
                return;
            }
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.end('not found');
        } catch (e) {
            if (!res.headersSent) json(res, e.status || (e.code === 'ENOENT' ? 404 : 500), { error: e.message });
        }
    });
    await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve));
    const actualPort = server.address().port;
    return {
        url: `http://127.0.0.1:${actualPort}/?token=${token}`,
        base: `http://127.0.0.1:${actualPort}`,
        port: actualPort, token, root, state, broadcast: S.broadcast,
        async touch(rel, text) {
            await fsp.writeFile(path.join(root, rel), text);
            S.broadcast('fs', { paths: [rel] });
        },
        close: () => new Promise((resolve) => { for (const c of clients) c.end(); server.close(() => resolve()); }),
    };
}

/* ── command line ── */
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const argv = process.argv.slice(2);
    const opt = (n) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : undefined; };
    startMockServer({ port: Number(opt('--port') || 8765), root: opt('--root'), token: opt('--token') || 'dev' }).then((s) => {
        console.log(`TIL Studio mock server: ${s.url}\nproject: ${s.root}`);
    });
}
