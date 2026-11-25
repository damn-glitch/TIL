// @ts-check
'use strict';
// Reading what the TIL command-line tools print: compiler diagnostics
// (`til check`, `til build`, `til prove`), the prover's summary and the
// `til fit --json` report. Pure functions (no `vscode`), tested in node.

// The words of a diagnostic header in every message language (src/tilmsg.py FRAME).
const ERROR_WORDS = ['error', 'ошибка', 'қате'];
const WARNING_WORDS = ['warning', 'предупреждение', 'ескерту'];
const NOTE_WORDS = ['note', 'примечание', 'ескертпе'];
const HINT_WORDS = ['hint', 'подсказка', 'кеңес'];
const SYNTAX_WORDS = ['Syntax Error', 'Синтаксическая ошибка', 'Синтаксистік қате'];

const ANSI = /\x1b\[[0-9;?]*[ -/]*[@-~]/g;

const HEADER = new RegExp(
    '^(?:(?:' + SYNTAX_WORDS.join('|') + '): )?(' + [...ERROR_WORDS, ...WARNING_WORDS, ...NOTE_WORDS].join('|') +
    ')(?:\\[([A-Za-z]+-[A-Za-z0-9-]+)\\])?: (.*)$');
const LOCATION = /^\s*--> (.+):(\d+):(\d+)\s*$/;
const HINT = new RegExp('^\\s*= (?:' + HINT_WORDS.join('|') + '): (.*)$');

/** Strip terminal colours. */
function stripAnsi(text) {
    return String(text || '').replace(ANSI, '');
}

/**
 * @typedef {object} ToolDiagnostic
 * @property {'error' | 'warning' | 'information'} severity
 * @property {string} code       e.g. E-PROVE ('' when there is none)
 * @property {string} message
 * @property {string} file       as printed (may be relative to the tool's working directory)
 * @property {number} line       1-based
 * @property {number} col        1-based
 * @property {string} hint
 */

/**
 * The diagnostics in the output of a TIL tool, in the ErrorReporter format:
 *   error[E-PROVE]: 'add': #[ensures: result > a] fails for a = 1
 *    --> prog.til:2:1
 *     2 | #[ensures: result > a]
 *       | ^
 *     = hint: ...
 * @param {string} text
 * @returns {ToolDiagnostic[]}
 */
function parseToolDiagnostics(text) {
    const lines = stripAnsi(text).split(/\r?\n/);
    /** @type {ToolDiagnostic[]} */
    const out = [];
    /** @type {ToolDiagnostic | null} */
    let cur = null;
    for (const line of lines) {
        const h = HEADER.exec(line);
        if (h) {
            const word = h[1];
            cur = {
                severity: ERROR_WORDS.includes(word) ? 'error' : WARNING_WORDS.includes(word) ? 'warning' : 'information',
                code: h[2] || '', message: h[3].trim(), file: '', line: 0, col: 0, hint: '',
            };
            out.push(cur);
            continue;
        }
        if (!cur) continue;
        const loc = LOCATION.exec(line);
        if (loc && !cur.file) {
            cur.file = loc[1];
            cur.line = parseInt(loc[2], 10);
            cur.col = parseInt(loc[3], 10);
            continue;
        }
        const hint = HINT.exec(line);
        if (hint) {
            cur.hint = hint[1].trim();
            cur = null;                 // the hint ends a diagnostic
        }
    }
    return out.filter((d) => d.line > 0);
}

/**
 * @typedef {object} ProofResult
 * @property {string} name       function name (`__start__` for the global initializers)
 * @property {string} status     proved | counterexample | not proved | unknown | unsupported
 * @property {string} detail     the rest of the summary line
 * @property {number} line       the line the summary names (0 if none)
 */

/**
 * The summary `til prove` prints on standard output (always English):
 *   Proving prog.til with Z3 4.13 (64-bit integers, 10000 ms per query)
 *     clamp: proved (3 conditions)
 *     add: counterexample: a = 1, b = 0 (#[ensures: result > a] fails, line 2)
 *   Result: 1 proved, 1 counterexample
 * @param {string} text
 * @returns {{functions: ProofResult[], result: string | null, header: string | null, nothing: boolean}}
 */
function parseProveSummary(text) {
    /** @type {ProofResult[]} */
    const functions = [];
    let result = null;
    let header = null;
    let nothing = false;
    for (const raw of stripAnsi(text).split(/\r?\n/)) {
        const line = raw.replace(/\s+$/, '');
        if (/^Proving /.test(line)) { header = line; continue; }
        if (/^Nothing to prove/.test(line)) { nothing = true; continue; }
        const r = /^Result: (.*)$/.exec(line);
        if (r) { result = r[1]; continue; }
        const m = /^ {2}(\S.*?): (proved|counterexample|not proved|unknown|unsupported)\b(.*)$/.exec(line);
        if (m) {
            const where = /\bline (\d+)\)?\s*(?:;.*)?$/.exec(m[3]);
            const name = m[1].replace(/ \(global initializers\)$/, '');
            functions.push({ name, status: m[2], detail: m[3].replace(/^[:\s]+/, ''), line: where ? parseInt(where[1], 10) : 0 });
        }
    }
    return { functions, result, header, nothing };
}

/** Severity of a proof diagnostic, from the status of its function. */
function proofSeverity(status) {
    if (status === 'counterexample') return 'error';
    if (status === 'unsupported') return 'information';
    return 'warning';
}

/** The function a prover diagnostic is about: the first quoted name of its message. */
function proofFunction(message) {
    const m = /'([^']+)'/.exec(message || '');
    return m ? m[1] : '';
}

// ── til fit ─────────────────────────────────────────────────────────────────

/** The line numbers a fit message mentions ("line 31 in checksum, line 32 in ..."). */
function mentionedLines(text) {
    const out = [];
    const re = /\blines? (\d+)/g;
    let m;
    while ((m = re.exec(String(text || '')))) out.push(parseInt(m[1], 10));
    return out;
}

/**
 * Diagnostics for a `til fit --json` report: errors, warnings, notes, each at
 * the first line it mentions (else at line 1).
 * @param {any} report
 * @returns {{severity: 'error' | 'warning' | 'information', message: string, line: number}[]}
 */
function fitDiagnostics(report) {
    const out = [];
    const add = (severity, items) => {
        for (const text of Array.isArray(items) ? items : []) {
            const lines = mentionedLines(text);
            out.push({ severity, message: String(text), line: lines.length ? lines[0] : 1 });
        }
    };
    if (report && typeof report === 'object') {
        add('error', report.errors);
        add('warning', report.warnings);
        add('information', report.notes);
    }
    return out;
}

function pad(text, width) {
    text = String(text);
    return text.length >= width ? text : text + ' '.repeat(width - text.length);
}

/**
 * A plain-text rendering of a `til fit --json` report (for the output channel).
 * @param {any} report
 * @returns {string}
 */
function fitReportText(report) {
    const r = report || {};
    const dev = r.device || {};
    const out = [];
    out.push(`til fit: ${r.program || ''} on ${dev.name || '?'}`);
    const about = [dev.title, dev.cpu].filter(Boolean).join('; ');
    if (about) out.push(`device: ${about}`);
    out.push('');
    const rows = Array.isArray(r.resources) ? r.resources : [];
    if (rows.length) {
        const w = [16, 20, 12, 8];
        out.push('  ' + pad('resource', w[0]) + ' ' + pad('program needs', w[1]) + ' ' + pad('device', w[2]) + ' ' +
            pad('status', w[3]));
        for (const row of rows) {
            const measured = row.measured !== null && row.measured !== undefined && row.measured !== ''
                ? ` (measured ${row.measured})` : '';
            out.push('  ' + pad(row.resource || row.key || '', w[0]) + ' ' + pad(String(row.needs ?? '') + measured, w[1]) +
                ' ' + pad(row.limit ?? '', w[2]) + ' ' + pad(row.status || '', w[3]) + (row.note ? '  ' + row.note : ''));
        }
        out.push('');
    }
    const section = (title, items) => {
        if (Array.isArray(items) && items.length) {
            out.push(title + ':');
            for (const s of items) out.push('  - ' + s);
            out.push('');
        }
    };
    section('errors', r.errors);
    section('warnings', r.warnings);
    section('notes', r.notes);
    if (r.run && typeof r.run === 'object') {
        const parts = Object.entries(r.run).filter(([, v]) => v !== null && typeof v !== 'object')
            .map(([k, v]) => `${k} ${v}`);
        if (parts.length) out.push('run: ' + parts.join(', '));
    }
    out.push('verdict: ' + (r.verdict || (r.fits ? 'fits' : 'does not fit')));
    return out.join('\n') + '\n';
}

/** The file a `til build` wrote ("Built: game.tbc (1111 bytes ...)"). */
function builtArtifact(text) {
    const m = /^Built: (.+?)(?: \(.*\))?\s*$/m.exec(stripAnsi(text));
    return m ? m[1] : null;
}

module.exports = {
    stripAnsi, parseToolDiagnostics, parseProveSummary, proofSeverity, proofFunction, mentionedLines,
    fitDiagnostics, fitReportText, builtArtifact,
};
