// @ts-check
'use strict';
// The TIL extension for VS Code (TIL Studio's "any editor" face,
// docs/dev/TIL_STUDIO_DESIGN.md section 7): a language client for `til lsp`,
// the `til` debug type on `til dap` (time travel), and commands for TIL's
// tools. Built with esbuild into dist/extension.js (scripts/build.mjs).

const vscode = require('vscode');
const os = require('os');
const launcher = require('./launcher');
const { TilClient } = require('./client');
const { Commands } = require('./commands');
const { TilDebug } = require('./debug');

const LANGS = ['en', 'ru', 'kk'];

/**
 * The TIL command line for the current settings (til.path, til.python, the
 * deprecated til.compilerPath), PATH and the extension's location.
 * Restricted Mode: workspace values of til.path / til.python are not used
 * (restrictedConfigurations) and no TIL checkout is taken from the workspace.
 * @param {string | undefined} extensionPath
 * @returns {launcher.TilCommand}
 */
function tilCommandFromSettings(extensionPath) {
    const cfg = vscode.workspace.getConfiguration('til');
    const trusted = vscode.workspace.isTrusted !== false;
    const folders = trusted
        ? (vscode.workspace.workspaceFolders || []).filter((f) => f.uri.scheme === 'file').map((f) => f.uri.fsPath)
        : [];
    return launcher.resolveTil({
        tilPath: cfg.get('path', ''),
        python: cfg.get('python', ''),
        legacyPath: cfg.get('compilerPath', 'til'),
        extensionPath,
        workspaceFolders: folders,
        platform: process.platform,
        env: process.env,
        home: os.homedir(),
    });
}

/** The command line that starts the language server (`til lsp`). */
function lspServerCommand(extensionPath) {
    const til = tilCommandFromSettings(extensionPath);
    return { ...launcher.tilArgv(til, ['lsp']), til };
}

/** The command line of the debug adapter (`til dap`). */
function dapCommand(extensionPath) {
    const til = tilCommandFromSettings(extensionPath);
    return { ...launcher.tilArgv(til, ['dap']), til };
}

/** What the parts of the extension share. */
class Context {
    /** @param {vscode.ExtensionContext} extension */
    constructor(extension) {
        this.extension = extension;
        this.output = vscode.window.createOutputChannel('TIL');
        extension.subscriptions.push(this.output);
        /** @type {launcher.TilCommand | null} */
        this.cached = null;
        /** @type {TilClient | null} */
        this.client = null;
        this.warnedMissing = false;
    }

    /** The TIL command line (cached until the settings change). */
    til() {
        if (!this.cached) {
            this.cached = tilCommandFromSettings(this.extension.extensionPath);
            this.log(this.cached.found ? `TIL: ${launcher.describe(this.cached)} (${this.cached.source})`
                : `TIL not found: ${this.cached.problem}`);
            if (this.cached.found && this.cached.tilPy && this.cached.pythonFound === false) {
                this.log(`no Python 3 found on PATH for ${this.cached.tilPy}: set til.python`);
            }
        }
        return this.cached;
    }

    refresh() {
        this.cached = null;
        this.warnedMissing = false;
    }

    /** Why TIL cannot be started with the current settings (null: it can). */
    problem() {
        const til = this.til();
        if (!til.found) return til.problem || 'TIL was not found';
        if (til.tilPy && til.pythonFound === false) {
            return `Python 3 was not found to run ${til.tilPy}: install Python 3.9+ or set til.python`;
        }
        return null;
    }

    /** til.locale when it names a language (en, ru, kk), else null (auto). */
    explicitLang() {
        const v = String(vscode.workspace.getConfiguration('til').get('locale', 'auto') || 'auto');
        return LANGS.includes(v) ? v : null;
    }

    /** The locale for the language server: til.locale, or VS Code's display language. */
    locale() {
        return this.explicitLang() || vscode.env.language || 'en';
    }

    /** `--lang X` for the command-line tools when til.locale names a language. */
    langArgs() {
        const lang = this.explicitLang();
        return lang ? ['--lang', lang] : [];
    }

    /** The environment of a TIL process (UTF-8 I/O; a chosen til.locale beats TIL_LANG). */
    processEnv(forServer) {
        /** @type {{[k: string]: string | undefined}} */
        const env = { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' };
        if (forServer && this.explicitLang()) delete env.TIL_LANG;
        return env;
    }

    /** Additions to the environment of the debug adapter (merged by VS Code). */
    adapterEnv() {
        return { PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' };
    }

    /** The environment of a new terminal (the same UTF-8 setting). */
    terminalEnv() {
        return { PYTHONIOENCODING: 'utf-8' };
    }

    log(text) {
        this.output.appendLine(text);
    }

    /** Say once that TIL is missing (later attempts just log). */
    warnMissingOnce() {
        if (this.warnedMissing) return;
        this.warnedMissing = true;
        vscode.window.showWarningMessage(`TIL: ${this.problem() || 'TIL was not found.'} (settings: til.path, til.python)`,
            'Open Settings', 'Install TIL').then((pick) => {
            if (pick === 'Open Settings') vscode.commands.executeCommand('workbench.action.openSettings', 'til.path');
            if (pick === 'Install TIL') vscode.env.openExternal(vscode.Uri.parse('https://til-dev.vercel.app'));
        });
    }
}

/** @type {Context | null} */
let current = null;

/** @param {vscode.ExtensionContext} extension */
function activate(extension) {
    const ctx = new Context(extension);
    current = ctx;
    const client = new TilClient(ctx);
    ctx.client = client;
    new Commands(ctx).register();
    new TilDebug(ctx).register();
    extension.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration((e) => {
            const launch = ['til.path', 'til.python', 'til.compilerPath'].some((k) => e.affectsConfiguration(k));
            if (launch) ctx.refresh();
            if (e.affectsConfiguration('til.lsp.enabled')) {
                if (client.enabled) client.start();
                else client.stop();
            } else if (launch && client.enabled) {
                client.restart();
            } else if (e.affectsConfiguration('til.locale')) {
                client.sendLocale();
            }
        }),
        vscode.workspace.onDidChangeWorkspaceFolders(() => ctx.refresh()),
        vscode.workspace.onDidGrantWorkspaceTrust(() => {
            ctx.refresh();
            if (client.enabled) client.restart();
        }),
        { dispose: () => { client.dispose(); } },
    );
    if (client.enabled) client.start();
    else client.setState('disabled');
    return { tilCommand: () => ctx.til(), lspServerCommand: () => lspServerCommand(extension.extensionPath) };
}

function deactivate() {
    const ctx = current;
    current = null;
    if (ctx && ctx.client) return ctx.client.dispose();
    return undefined;
}

module.exports = { activate, deactivate, tilCommandFromSettings, lspServerCommand, dapCommand, launcher, Context };
