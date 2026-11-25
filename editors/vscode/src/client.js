// @ts-check
'use strict';
// The language client for `til lsp` and the status bar item that shows the
// server's state and TIL's version.

const vscode = require('vscode');
const launcher = require('./launcher');
const util = require('./util');

/** @typedef {'disabled' | 'starting' | 'running' | 'stopped' | 'failed' | 'missing'} ServerState */

class TilClient {
    /** @param {import('./extension').Context} ctx */
    constructor(ctx) {
        this.ctx = ctx;
        /** @type {any} the LanguageClient (vscode-languageclient/node), when started */
        this.client = null;
        /** @type {ServerState} */
        this.state = 'disabled';
        this.version = '';
        this.detail = '';
        this.serverOutput = vscode.window.createOutputChannel('TIL Language Server');
        this.status = vscode.window.createStatusBarItem('til.status', vscode.StatusBarAlignment.Right, 100);
        this.status.name = 'TIL';
        this.status.command = 'til.statusMenu';
        /** @type {Promise<void>} serializes start / stop */
        this.queue = Promise.resolve();
        ctx.extension.subscriptions.push(this.serverOutput, this.status,
            vscode.window.onDidChangeActiveTextEditor(() => this.updateStatus()));
        this.updateStatus();
    }

    get enabled() {
        return Boolean(vscode.workspace.getConfiguration('til').get('lsp.enabled', true));
    }

    isRunning() {
        return this.state === 'running' && this.client !== null;
    }

    /** Run start/stop one after another. */
    enqueue(job) {
        this.queue = this.queue.then(job, job).catch((e) => this.ctx.log(`language client: ${e && e.stack || e}`));
        return this.queue;
    }

    start() {
        return this.enqueue(() => this._start());
    }

    stop() {
        return this.enqueue(() => this._stop('disabled'));
    }

    restart() {
        return this.enqueue(async () => {
            await this._stop('stopped');
            await this._start();
        });
    }

    async _start() {
        if (this.client) return;
        if (!this.enabled) {
            this.setState('disabled');
            return;
        }
        const til = this.ctx.til();
        const problem = this.ctx.problem();
        if (problem) {
            this.detail = problem;
            this.setState('missing');
            this.ctx.log(`language server not started: ${problem}`);
            this.ctx.warnMissingOnce();
            return;
        }
        // loaded on first use (bundled): activation stays cheap when the server is off
        const lc = require('vscode-languageclient/node');
        const argv = launcher.tilArgv(til, ['lsp']);
        const folder = (vscode.workspace.workspaceFolders || []).find((f) => f.uri.scheme === 'file');
        /** @type {import('vscode-languageclient/node').ServerOptions} */
        const serverOptions = {
            command: argv.command,
            args: argv.args,
            options: { cwd: folder ? folder.uri.fsPath : undefined, env: this.ctx.processEnv(true) },
        };
        /** @type {import('vscode-languageclient/node').LanguageClientOptions} */
        const clientOptions = {
            documentSelector: [{ scheme: 'file', language: 'til' }, { scheme: 'untitled', language: 'til' }],
            initializationOptions: { locale: this.ctx.locale() },
            synchronize: { fileEvents: vscode.workspace.createFileSystemWatcher('**/*.til') },
            outputChannel: this.serverOutput,
            revealOutputChannelOn: lc.RevealOutputChannelOn.Never,
            initializationFailedHandler: (error) => {
                this.ctx.log(`the language server failed to initialize: ${error && error.message || error}`);
                return false;
            },
        };
        this.detail = launcher.describe(til, ['lsp']);
        this.version = '';
        this.setState('starting');
        this.ctx.log(`starting the language server: ${this.detail}`);
        let client = null;
        try {
            client = new lc.LanguageClient('til', 'TIL Language Server', serverOptions, clientOptions);
            this.client = client;
            const started = client;
            client.onDidChangeState((e) => {
                if (this.client !== started) return;
                if (e.newState === lc.State.Running) this.setState('running');
                else if (e.newState === lc.State.Starting) this.setState('starting');
                else if (e.newState === lc.State.Stopped && this.state !== 'disabled') this.setState('stopped');
            });
            await client.start();
            const info = client.initializeResult && client.initializeResult.serverInfo;
            this.version = info && info.version ? info.version : '';
            this.setState('running');
            this.ctx.log(`language server running: ${info ? `${info.name} ${info.version || ''}` : 'til lsp'}`);
        } catch (e) {
            this.ctx.log(`the language server did not start: ${e && e.message || e}`);
            if (this.client === client) this.client = null;
            if (client) {
                try { await client.dispose(); } catch (err) { /* it never ran */ }
            }
            this.setState('failed');
            util.reportStartFailure(this.ctx, e);
        }
    }

    async _stop(state) {
        const client = this.client;
        this.client = null;
        this.setState(state);
        if (client) {
            try {
                await client.stop(2000);
            } catch (e) {
                this.ctx.log(`stopping the language server: ${e && e.message || e}`);
            }
            try { await client.dispose(); } catch (e) { /* stopped */ }
        }
    }

    /** til.locale changed: tell the running server (workspace/didChangeConfiguration). */
    sendLocale() {
        if (!this.isRunning()) return;
        this.client.sendNotification('workspace/didChangeConfiguration',
            { settings: { til: { locale: this.ctx.locale() } } }).catch(() => { /* stopped meanwhile */ });
        this.updateStatus();
    }

    /**
     * workspace/executeCommand on the server (e.g. til.translate).
     * @returns {Promise<any>}
     */
    executeCommand(command, args) {
        if (!this.isRunning()) return Promise.reject(new Error('the TIL language server is not running'));
        return this.client.sendRequest('workspace/executeCommand', { command, arguments: args });
    }

    /** @param {ServerState} state */
    setState(state) {
        this.state = state;
        this.updateStatus();
    }

    updateStatus() {
        const item = this.status;
        const ed = vscode.window.activeTextEditor;
        const tilEditor = !!ed && ed.document.languageId === 'til';
        const v = this.version ? ` ${this.version}` : '';
        const table = {
            disabled: ['$(circle-slash) TIL', 'the language server is off (til.lsp.enabled)', undefined],
            starting: ['$(sync~spin) TIL', 'the language server is starting', undefined],
            running: [`$(check) TIL${v}`, 'the language server is running', undefined],
            stopped: ['$(warning) TIL', 'the language server stopped', 'statusBarItem.warningBackground'],
            failed: ['$(error) TIL', 'the language server could not be started', 'statusBarItem.errorBackground'],
            missing: ['$(error) TIL', 'TIL was not found', 'statusBarItem.errorBackground'],
        }[this.state];
        item.text = table[0];
        item.backgroundColor = table[2] ? new vscode.ThemeColor(table[2]) : undefined;
        const md = new vscode.MarkdownString(undefined, true);
        md.appendMarkdown(`**TIL${v}**: ${table[1]}\n\n`);
        if (this.detail) md.appendMarkdown('`' + this.detail.replace(/`/g, "'") + '`\n\n');
        md.appendMarkdown(`Messages: ${this.ctx.locale()} · click for actions`);
        item.tooltip = md;
        if (tilEditor || this.state === 'failed' || this.state === 'missing') item.show();
        else item.hide();
    }

    async dispose() {
        await this._stop('disabled');
    }
}

module.exports = { TilClient };
