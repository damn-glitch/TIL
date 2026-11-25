// @ts-check
'use strict';
// Helpers shared by the commands and the debugger: which file a command is
// about, saving it, running a TIL tool and reading its output.

const vscode = require('vscode');
const cp = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const launcher = require('./launcher');

/**
 * The file (and function) a command is about. Commands are called from the
 * command palette (no argument: the active editor), editor title / explorer
 * menus (a Uri), code lenses of the language server ({uri, function}, uri a
 * string) and other extensions (a path or a URI string).
 * @param {any} arg
 * @returns {{uri: vscode.Uri | undefined, fn: string | null, extra: any}}
 */
function targetFrom(arg) {
    if (Array.isArray(arg)) arg = arg[0];
    if (arg && typeof arg === 'object' && typeof arg.scheme === 'string' && typeof arg.fsPath === 'string') {
        return { uri: /** @type {vscode.Uri} */ (arg), fn: null, extra: {} };
    }
    if (arg && typeof arg === 'object' && arg.uri !== undefined) {
        const u = arg.uri;
        const uri = typeof u === 'string' ? parseUri(u) : (u && typeof u.scheme === 'string' ? u : undefined);
        return { uri, fn: typeof arg.function === 'string' ? arg.function : null, extra: arg };
    }
    if (typeof arg === 'string' && arg) return { uri: parseUri(arg), fn: null, extra: {} };
    const ed = vscode.window.activeTextEditor;
    return { uri: ed ? ed.document.uri : undefined, fn: null, extra: {} };
}

function parseUri(s) {
    return /^[a-z][a-z0-9+.-]+:/i.test(s) && !/^[a-z]:[\\/]/i.test(s) ? vscode.Uri.parse(s) : vscode.Uri.file(s);
}

/** The open document of a Uri (if it is open). */
function openDocument(uri) {
    const key = uri.toString();
    return vscode.workspace.textDocuments.find((d) => d.uri.toString() === key);
}

/** The workspace folder of a file, else its directory. */
function folderOf(uri) {
    const folder = uri.scheme === 'file' ? vscode.workspace.getWorkspaceFolder(uri) : undefined;
    if (folder && folder.uri.scheme === 'file') return folder.uri.fsPath;
    if (uri.scheme === 'file') return path.dirname(uri.fsPath);
    const first = (vscode.workspace.workspaceFolders || []).find((f) => f.uri.scheme === 'file');
    return first ? first.uri.fsPath : os.homedir();
}

let untitledCounter = 0;

/**
 * The file on disk to give a tool: saves a modified editor first; an untitled
 * editor is written to a temporary file. undefined when there is no file.
 * @param {vscode.Uri | undefined} uri
 * @returns {Promise<string | undefined>}
 */
async function fileOnDisk(uri) {
    if (!uri) {
        vscode.window.showWarningMessage('TIL: open a .til file first.');
        return undefined;
    }
    const doc = openDocument(uri);
    if (uri.scheme === 'untitled' || (doc && doc.isUntitled)) {
        const text = doc ? doc.getText() : '';
        const dir = path.join(os.tmpdir(), 'til-vscode');
        fs.mkdirSync(dir, { recursive: true });
        const name = (path.basename(uri.path).replace(/[^\w.-]+/g, '_') || 'untitled') + '-' + (++untitledCounter) + '.til';
        const file = path.join(dir, name);
        fs.writeFileSync(file, text, 'utf8');
        return file;
    }
    if (uri.scheme !== 'file') {
        vscode.window.showWarningMessage(`TIL: ${uri.toString()} is not a file on disk.`);
        return undefined;
    }
    if (doc && doc.isDirty) await doc.save();
    return uri.fsPath;
}

/** Does a program draw on the screen (so a terminal run should show it)? */
function usesScreen(text) {
    return /\bscreen_init\s*\(/.test(text || '');
}

/** The text of a file: the editor's when it is open, else the file on disk. */
function textOf(uri) {
    const doc = openDocument(uri);
    if (doc) return doc.getText();
    try { return fs.readFileSync(uri.fsPath, 'utf8'); } catch (e) { return ''; }
}

/**
 * @typedef {{code: number | null, signal: string | null, stdout: string, stderr: string, error?: Error, cancelled?: boolean}} ToolResult
 */

/**
 * Run a TIL command (`til <rest...>`) and collect its output.
 * @param {import('./extension').Context} ctx
 * @param {string[]} rest
 * @param {{cwd?: string, token?: vscode.CancellationToken, timeoutMs?: number, input?: string}} [opts]
 * @returns {Promise<ToolResult>}
 */
function execTil(ctx, rest, opts = {}) {
    const til = ctx.til();
    const problem = ctx.problem();
    if (problem) {
        return Promise.resolve({ code: null, signal: null, stdout: '', stderr: '', error: new Error(problem) });
    }
    const full = [...ctx.langArgs(), ...rest];
    const argv = launcher.tilArgv(til, full);
    ctx.log(`$ ${launcher.describe(til, full)}` + (opts.cwd ? `   (in ${opts.cwd})` : ''));
    return new Promise((resolve) => {
        let stdout = '';
        let stderr = '';
        let done = false;
        let cancelled = false;
        /** @type {NodeJS.Timeout | null} */
        let timer = null;
        /** @type {vscode.Disposable | null} */
        let sub = null;
        const finish = (/** @type {ToolResult} */ r) => {
            if (done) return;
            done = true;
            if (timer) clearTimeout(timer);
            if (sub) sub.dispose();
            resolve(r);
        };
        /** @type {cp.ChildProcess} */
        let child;
        try {
            child = cp.spawn(argv.command, argv.args, { cwd: opts.cwd, env: ctx.processEnv(false), windowsHide: true });
        } catch (e) {
            finish({ code: null, signal: null, stdout, stderr, error: /** @type {Error} */ (e) });
            return;
        }
        const kill = () => { try { child.kill(); } catch (e) { /* already gone */ } };
        if (opts.timeoutMs) timer = setTimeout(kill, opts.timeoutMs);
        if (opts.token) sub = opts.token.onCancellationRequested(() => { cancelled = true; kill(); });
        child.stdout.setEncoding('utf8');
        child.stderr.setEncoding('utf8');
        child.stdout.on('data', (d) => { stdout += d; });
        child.stderr.on('data', (d) => { stderr += d; });
        child.on('error', (e) => finish({ code: null, signal: null, stdout, stderr, error: e }));
        child.on('close', (code, signal) => finish({ code, signal, stdout, stderr, cancelled }));
        child.stdin.on('error', () => { /* the tool exited before reading its input */ });
        child.stdin.end(opts.input !== undefined ? opts.input : undefined);
    });
}

/** Show that TIL could not be started, with the way to fix it. */
async function reportStartFailure(ctx, error) {
    const til = ctx.til();
    const problem = ctx.problem();
    const what = problem || `TIL could not be started (${launcher.describe(til)}): ${error && error.message || error}`;
    const pick = await vscode.window.showErrorMessage(`TIL: ${what}`, 'Open Settings', 'Install TIL');
    if (pick === 'Open Settings') vscode.commands.executeCommand('workbench.action.openSettings', 'til.path');
    if (pick === 'Install TIL') vscode.env.openExternal(vscode.Uri.parse('https://til-dev.vercel.app'));
}

/** A diagnostic range: the word at (line, col) (1-based), at least one character. */
function rangeAt(doc, line, col) {
    const l = Math.max(0, Math.min((doc ? doc.lineCount : line) - 1, line - 1));
    const c = Math.max(0, col - 1);
    if (doc) {
        const word = doc.getWordRangeAtPosition(new vscode.Position(l, c));
        if (word) return word;
        const text = doc.lineAt(l).text;
        if (c >= text.length) return new vscode.Range(l, Math.max(0, text.length - 1), l, Math.max(1, text.length));
    }
    return new vscode.Range(l, c, l, c + 1);
}

const SEVERITY = {
    error: vscode.DiagnosticSeverity.Error,
    warning: vscode.DiagnosticSeverity.Warning,
    information: vscode.DiagnosticSeverity.Information,
};

/** The Uri of a file a tool names (relative to its working directory). */
function fileUri(file, cwd) {
    return vscode.Uri.file(path.isAbsolute(file) ? file : path.resolve(cwd || '.', file));
}

module.exports = {
    targetFrom, parseUri, openDocument, folderOf, fileOnDisk, usesScreen, textOf, execTil, reportStartFailure,
    rangeAt, SEVERITY, fileUri,
};
