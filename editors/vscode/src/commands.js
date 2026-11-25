// @ts-check
'use strict';
// The commands of the extension that run TIL's tools: run (native, TIL VM,
// C VM) in a terminal, prove, fit for a device, translate keywords, build,
// check, open TIL Studio, and the language-server commands.

const vscode = require('vscode');
const path = require('path');
const launcher = require('./launcher');
const reports = require('./reports');
const util = require('./util');

const TRANSLATIONS = [
    { to: 'en', label: 'English', description: 'if, for, while, print ...' },
    { to: 'kk', label: 'Қазақша', description: 'егер, үшін, кезінде, басып_шығару ...' },
    { to: 'kk-latn', label: 'Qazaqsha (latyn)', description: 'eger, ushin, kezinde ... (#[lang: kk-latn])' },
    { to: 'ru', label: 'Русский', description: 'если, для, пока, печать ... (#[lang: ru])' },
];

/**
 * A notification whose buttons act later: the command does not wait for a click.
 * @param {'info' | 'warning' | 'error'} kind
 * @param {string} message
 * @param {{[label: string]: () => any}} [actions]
 */
function notify(kind, message, actions = {}) {
    const show = kind === 'error' ? vscode.window.showErrorMessage
        : kind === 'warning' ? vscode.window.showWarningMessage : vscode.window.showInformationMessage;
    show(message, ...Object.keys(actions)).then((pick) => {
        if (pick && actions[pick]) actions[pick]();
    });
}

class Commands {
    /** @param {import('./extension').Context} ctx */
    constructor(ctx) {
        this.ctx = ctx;
        this.proofs = vscode.languages.createDiagnosticCollection('til-prove');
        this.fits = vscode.languages.createDiagnosticCollection('til-fit');
        this.checks = vscode.languages.createDiagnosticCollection('til-check');
        /** uri -> function -> diagnostics (a single-function proof replaces only its own) */
        this.proofDiags = new Map();
        /** @type {any[] | null} */
        this.devices = null;
        ctx.extension.subscriptions.push(this.proofs, this.fits, this.checks,
            vscode.workspace.onDidChangeConfiguration((e) => {
                if (e.affectsConfiguration('til')) this.devices = null;     // another TIL may list other devices
            }),
            vscode.workspace.onDidChangeTextDocument((e) => {
                // results of a tool describe the text it saw: drop them on an edit
                if (e.contentChanges.length && e.document.languageId === 'til') {
                    this.fits.delete(e.document.uri);
                    this.checks.delete(e.document.uri);
                }
            }));
    }

    register() {
        const r = (id, fn) => this.ctx.extension.subscriptions.push(vscode.commands.registerCommand(id, fn));
        r('til.run', (arg) => this.run(arg, 'native'));
        r('til.runVM', (arg) => this.run(arg, 'vm'));
        r('til.runCVM', (arg) => this.run(arg, 'cvm'));
        r('til.prove', (arg) => this.prove(arg, false));
        r('til.proveFunction', (arg) => this.prove(arg, true));
        r('til.fit', (arg) => this.fit(arg));
        r('til.translate', (arg) => this.translate(arg));
        r('til.openStudio', (arg) => this.openStudio(arg));
        r('til.buildBytecode', (arg) => this.build(arg, 'vm'));
        r('til.buildCVM', (arg) => this.build(arg, 'c-vm'));
        r('til.build', (arg) => this.build(arg, 'native'));
        r('til.check', (arg) => this.check(arg));
        r('til.restartServer', () => this.restartServer());
        r('til.showOutput', () => this.ctx.output.show(true));
        r('til.statusMenu', () => this.statusMenu());
    }

    // ── terminals ───────────────────────────────────────────────────────────

    /**
     * Type a TIL command into a terminal (reused by name).
     * @param {string[]} rest  the til command and its arguments
     * @param {string} cwd
     * @param {string} name
     */
    terminal(rest, cwd, name) {
        const til = this.ctx.til();
        if (this.ctx.problem()) {
            util.reportStartFailure(this.ctx, null);
            return undefined;
        }
        let term = vscode.window.terminals.find((t) => t.name === name && t.exitStatus === undefined);
        if (!term) term = vscode.window.createTerminal({ name, cwd, env: this.ctx.terminalEnv() });
        const shell = /** @type {any} */ (term.creationOptions).shellPath || vscode.env.shell;
        const kind = launcher.shellKind(shell, process.platform);
        const argv = launcher.tilArgv(til, [...this.ctx.langArgs(), ...rest]);
        term.show(true);
        term.sendText(launcher.commandLine([argv.command, ...argv.args], kind), true);
        return term;
    }

    /**
     * Run the file in a terminal: `til run [--vm | --cvm] FILE` (input, keys and
     * the --screen-ansi screen work there).
     * @param {any} arg
     * @param {'native' | 'vm' | 'cvm'} backend
     */
    async run(arg, backend) {
        const { uri } = util.targetFrom(arg);
        const file = await util.fileOnDisk(uri);
        if (!file || !uri) return;
        const rest = ['run'];
        if (backend === 'vm') rest.push('--vm');
        if (backend === 'cvm') rest.push('--cvm');
        const cfg = vscode.workspace.getConfiguration('til');
        if (cfg.get('run.screenAnsi', true) && util.usesScreen(util.textOf(uri))) rest.push('--screen-ansi');
        rest.push(file);
        const term = this.terminal(rest, uri.scheme === 'file' ? util.folderOf(uri) : path.dirname(file), 'TIL');
        return term ? { terminal: term.name, args: rest } : undefined;
    }

    // ── til prove ───────────────────────────────────────────────────────────

    /** The function around the cursor (prover name: `Struct.method` for methods). */
    async functionAtCursor(editor) {
        const pos = editor.selection.active;
        try {
            /** @type {vscode.DocumentSymbol[] | undefined} */
            const symbols = await vscode.commands.executeCommand('vscode.executeDocumentSymbolProvider', editor.document.uri);
            let best = null;
            const walk = (list, parent) => {
                for (const s of list || []) {
                    if (!s.range || !s.range.contains(pos)) continue;
                    if (s.kind === vscode.SymbolKind.Function || s.kind === vscode.SymbolKind.Method) {
                        const owner = parent ? String(parent.name).trim().split(/\s+/).pop() : '';
                        best = owner && s.kind === vscode.SymbolKind.Method ? `${owner}.${s.name}` : s.name;
                    }
                    walk(s.children, s);
                }
            };
            if (Array.isArray(symbols) && symbols.length && symbols[0].range) walk(symbols, null);
            if (best) return best;
        } catch (e) { /* no symbol provider: scan the text */ }
        // without the language server: the nearest definition line above the cursor
        const doc = editor.document;
        const indent = (/^\s*/.exec(doc.lineAt(pos.line).text) || [''])[0].length;
        const def = /^(\s*)(?:pub\s+)?(?:fn\s+)?([\p{L}_][\p{L}\p{N}_]*)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?:->.*)?$/u;
        for (let l = pos.line; l >= 0; l--) {
            const m = def.exec(doc.lineAt(l).text);
            if (m && (l === pos.line || m[1].length < indent)) return m[2];
        }
        return null;
    }

    /**
     * `til prove FILE [--function NAME]`: results in the output channel, problems as diagnostics.
     * @param {any} arg
     * @param {boolean} atCursor
     */
    async prove(arg, atCursor) {
        const target = util.targetFrom(arg);
        let fn = target.fn;
        if (atCursor && !fn) {
            const ed = vscode.window.activeTextEditor;
            if (ed && target.uri && ed.document.uri.toString() === target.uri.toString()) fn = await this.functionAtCursor(ed);
            if (!fn) {
                vscode.window.showWarningMessage('TIL: put the cursor inside a function to prove it.');
                return;
            }
        }
        const file = await util.fileOnDisk(target.uri);
        if (!file || !target.uri) return;
        const cwd = path.dirname(file);
        const timeout = vscode.workspace.getConfiguration('til').get('prove.timeout', 10000);
        const rest = ['prove', file, '--timeout', String(timeout)];
        if (fn) rest.push('--function', fn);
        const what = fn ? `'${fn}'` : path.basename(file);
        const res = await vscode.window.withProgress(
            { location: vscode.ProgressLocation.Notification, title: `TIL: proving ${what}`, cancellable: true },
            (_p, token) => util.execTil(this.ctx, rest, { cwd, token }));
        const out = this.ctx.output;
        out.appendLine(res.stdout.trimEnd());
        if (res.stderr.trim()) out.appendLine(res.stderr.trimEnd());
        if (res.cancelled) {
            out.appendLine('(cancelled)');
            return;
        }
        if (res.error) {
            util.reportStartFailure(this.ctx, res.error);
            return;
        }
        const summary = reports.parseProveSummary(res.stdout);
        const diags = reports.parseToolDiagnostics(res.stderr);
        if (!summary.header) {
            // no proof ran: Z3 missing, a compile error, an unknown function
            const first = reports.stripAnsi(res.stderr).split('\n').find((l) => l.trim()) || `til prove exited with ${res.code}`;
            if (/z3|Z3/.test(res.stderr) && /install/i.test(res.stderr)) {
                notify('error', 'TIL: til prove needs the Z3 solver (pip install z3-solver).', {
                    'Copy Command': () => vscode.env.clipboard.writeText('pip install z3-solver'),
                    'Show Output': () => out.show(true),
                });
            } else {
                this.setCheckDiagnostics(diags, cwd, target.uri);
                notify('error', `TIL prove: ${first}`, { 'Show Output': () => out.show(true) });
            }
            return { ok: false, code: res.code, functions: [], problem: first };
        }
        const status = new Map(summary.functions.map((f) => [f.name, f.status]));
        /** @type {Map<string, Map<string, vscode.Diagnostic[]>>} */
        const byFile = new Map();
        const key = target.uri.toString();
        const previous = this.proofDiags.get(key) || new Map();
        if (fn) previous.delete(fn);
        else previous.clear();
        for (const d of diags) {
            const uri = d.file ? util.fileUri(d.file, cwd) : target.uri;
            const doc = util.openDocument(uri);
            const name = reports.proofFunction(d.message);
            const sev = d.severity === 'error' ? reports.proofSeverity(status.get(name) || 'counterexample') : d.severity;
            const diag = new vscode.Diagnostic(util.rangeAt(doc, d.line, d.col),
                d.hint ? `${d.message}\n${d.hint}` : d.message, util.SEVERITY[sev]);
            diag.code = d.code || 'E-PROVE';
            diag.source = 'til prove';
            const fileKey = uri.toString() === key ? key : uri.toString();
            if (!byFile.has(fileKey)) byFile.set(fileKey, new Map());
            const m = byFile.get(fileKey);
            if (!m.has(name)) m.set(name, []);
            m.get(name).push(diag);
        }
        const mine = byFile.get(key) || new Map();
        for (const [name, list] of mine) previous.set(name, list);
        this.proofDiags.set(key, previous);
        this.proofs.set(target.uri, [...previous.values()].flat());
        for (const [fileKey, m] of byFile) {
            if (fileKey !== key) this.proofs.set(vscode.Uri.parse(fileKey), [...m.values()].flat());
        }
        const text = summary.nothing ? 'nothing to prove (no contracts, assert or invariant)'
            : (summary.result || summary.functions.map((f) => `${f.name}: ${f.status}`).join(', '));
        const ok = res.code === 0;
        notify(ok ? 'info' : 'warning', `TIL prove ${what}: ${text}`, { 'Show Output': () => out.show(true) });
        return { ok, code: res.code, functions: summary.functions, result: summary.result, diagnostics: diags.length };
    }

    /** Diagnostics of a compile error a tool printed (when the language server does not show them). */
    setCheckDiagnostics(diags, cwd, fallbackUri) {
        /** @type {Map<string, vscode.Diagnostic[]>} */
        const byFile = new Map();
        for (const d of diags) {
            const uri = d.file ? util.fileUri(d.file, cwd) : fallbackUri;
            const diag = new vscode.Diagnostic(util.rangeAt(util.openDocument(uri), d.line, d.col),
                d.hint ? `${d.message}\n${d.hint}` : d.message, util.SEVERITY[d.severity]);
            if (d.code) diag.code = d.code;
            diag.source = 'til';
            const k = uri.toString();
            if (!byFile.has(k)) byFile.set(k, []);
            byFile.get(k).push(diag);
        }
        if (fallbackUri) this.checks.delete(fallbackUri);
        if (this.ctx.client && this.ctx.client.isRunning()) return byFile.size;   // the server reports them
        for (const [k, list] of byFile) this.checks.set(vscode.Uri.parse(k), list);
        return byFile.size;
    }

    // ── til fit ─────────────────────────────────────────────────────────────

    async listDevices(cwd) {
        if (this.devices) return this.devices;
        const res = await util.execTil(this.ctx, ['fit', '--list-devices', '--json'], { cwd, timeoutMs: 60000 });
        if (res.error) {
            util.reportStartFailure(this.ctx, res.error);
            return null;
        }
        try {
            const list = JSON.parse(res.stdout);
            if (Array.isArray(list)) this.devices = list;
            return this.devices;
        } catch (e) {
            this.ctx.output.appendLine(res.stdout + res.stderr);
            vscode.window.showErrorMessage('TIL: `til fit --list-devices --json` did not give a device list (an older TIL?).');
            return null;
        }
    }

    /** Pick a device and show `til fit FILE --device NAME --json`. */
    async fit(arg) {
        const target = util.targetFrom(arg);
        const file = await util.fileOnDisk(target.uri);
        if (!file || !target.uri) return;
        const cwd = path.dirname(file);
        const given = target.extra && typeof target.extra.device === 'string' ? target.extra.device : null;
        const picked = given ? { item: { label: given }, run: !!target.extra.run } : await this.pickDevice(file, cwd);
        if (!picked || !picked.item) return undefined;
        const device = picked.item.label;
        await this.ctx.extension.workspaceState.update('til.fit.device', device);
        const rest = ['fit', file, '--device', device, '--json'];
        if (picked.run) rest.push('--run', '--timeout', '60');
        const res = await vscode.window.withProgress(
            { location: vscode.ProgressLocation.Notification, title: `TIL: fitting ${path.basename(file)} on ${device}`, cancellable: true },
            (_p, token) => util.execTil(this.ctx, rest, { cwd, token }));
        if (res.cancelled) return undefined;
        if (res.error) {
            util.reportStartFailure(this.ctx, res.error);
            return undefined;
        }
        return this.showFit(res, device, target.uri, cwd);
    }

    /** The device quick pick (the play button of an item: also run the program). */
    async pickDevice(file, cwd) {
        const devices = await this.listDevices(cwd);
        if (!devices) return undefined;
        const last = this.ctx.extension.workspaceState.get('til.fit.device');
        const runButton = { iconPath: new vscode.ThemeIcon('play'), tooltip: 'Analyse and run on the device\'s C VM build (needs a C compiler)' };
        const items = devices.map((d) => ({
            label: String(d.name),
            description: d.title || '',
            detail: [d.cpu, d.int_bits ? `int${d.int_bits}` : '', d.heap ? `heap ${Math.round(d.heap / 1024)} KB` : '',
                d.screen === false ? 'no screen' : ''].filter(Boolean).join(' · '),
            buttons: [runButton],
            device: d,
        }));
        items.sort((a, b) => (a.label === last ? -1 : b.label === last ? 1 : 0));
        const picked = await new Promise((resolve) => {
            const qp = vscode.window.createQuickPick();
            qp.title = `TIL: will ${path.basename(file)} fit a device?`;
            qp.placeholder = 'Pick a device (the ▶ button also runs the program under the device\'s limits)';
            qp.items = items;
            qp.matchOnDescription = true;
            qp.matchOnDetail = true;
            let result;
            qp.onDidAccept(() => { result = { item: qp.selectedItems[0], run: false }; qp.hide(); });
            qp.onDidTriggerItemButton((e) => { result = { item: e.item, run: true }; qp.hide(); });
            qp.onDidHide(() => { qp.dispose(); resolve(result); });
            qp.show();
        });
        return picked;
    }

    /** The report of `til fit --json`: output channel, diagnostics, a notification. */
    showFit(res, device, uri, cwd) {
        let report;
        try {
            report = JSON.parse(res.stdout);
        } catch (e) {
            this.ctx.output.appendLine(res.stdout + res.stderr);
            this.setCheckDiagnostics(reports.parseToolDiagnostics(res.stderr), cwd, uri);
            const first = reports.stripAnsi(res.stderr).split('\n').find((l) => l.trim()) || `til fit exited with ${res.code}`;
            notify('error', `TIL fit: ${first}`, { 'Show Output': () => this.ctx.output.show(true) });
            return undefined;
        }
        if (res.stderr.trim()) this.ctx.output.appendLine(res.stderr.trimEnd());
        this.ctx.output.appendLine(reports.fitReportText(report));
        const doc = util.openDocument(uri);
        this.fits.set(uri, reports.fitDiagnostics(report).map((d) => {
            const diag = new vscode.Diagnostic(util.rangeAt(doc, d.line, 1), `${device}: ${d.message}`, util.SEVERITY[d.severity]);
            diag.source = 'til fit';
            return diag;
        }));
        const verdict = report.verdict || (report.fits ? 'fits' : 'does not fit');
        notify(report.fits ? 'info' : 'warning', `TIL fit (${device}): ${verdict}`,
            { 'Show Report': () => this.ctx.output.show(true) });
        return report;
    }

    // ── translate ───────────────────────────────────────────────────────────

    /** Translate the keywords of a file into another keyword language (a new editor). */
    async translate(arg) {
        const target = util.targetFrom(arg);
        const uri = target.uri;
        if (!uri) {
            vscode.window.showWarningMessage('TIL: open a .til file first.');
            return;
        }
        let to = target.extra && typeof target.extra.to === 'string' ? target.extra.to : null;
        if (!to) {
            const pick = await vscode.window.showQuickPick(TRANSLATIONS.map((t) => ({ ...t })),
                { title: 'TIL: translate the keywords to', placeHolder: 'Keyword language' });
            if (!pick) return;
            to = pick.to;
        }
        let text = null;
        const client = this.ctx.client;
        if (client && client.isRunning() && util.openDocument(uri)) {
            try {
                const res = await client.executeCommand('til.translate', [{ uri: uri.toString(), to }]);
                text = res && typeof res.text === 'string' ? res.text : null;
            } catch (e) {
                vscode.window.showErrorMessage(`TIL: cannot translate: ${e && e.message || e}`);
                return;
            }
        }
        if (text === null) {
            const file = await util.fileOnDisk(uri);
            if (!file) return;
            const res = await util.execTil(this.ctx, ['translate', '--to', to, file], { cwd: path.dirname(file) });
            if (res.error || res.code !== 0) {
                if (res.error) util.reportStartFailure(this.ctx, res.error);
                else vscode.window.showErrorMessage(`TIL: cannot translate: ${reports.stripAnsi(res.stderr).trim().split('\n')[0]}`);
                return;
            }
            text = res.stdout;
        }
        const doc = await vscode.workspace.openTextDocument({ language: 'til', content: text });
        await vscode.window.showTextDocument(doc, { viewColumn: vscode.ViewColumn.Beside, preview: false });
        return { uri: doc.uri.toString(), to };
    }

    // ── TIL Studio ──────────────────────────────────────────────────────────

    /** `til ide FOLDER` in a terminal (the studio opens in the browser). */
    async openStudio(arg) {
        let folder = null;
        const { uri } = util.targetFrom(arg);
        if (uri && uri.scheme === 'file') {
            const stat = await vscode.workspace.fs.stat(uri).then((s) => s, () => null);
            if (stat && stat.type & vscode.FileType.Directory) folder = uri.fsPath;
            else folder = util.folderOf(uri);
        }
        if (!folder) {
            const first = (vscode.workspace.workspaceFolders || []).find((f) => f.uri.scheme === 'file');
            if (first) folder = first.uri.fsPath;
        }
        if (!folder) {
            const picked = await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectFiles: false, openLabel: 'Open in TIL Studio' });
            if (!picked || !picked.length) return;
            folder = picked[0].fsPath;
        }
        const old = vscode.window.terminals.find((t) => t.name === 'TIL Studio' && t.exitStatus === undefined);
        if (old) old.dispose();                       // one studio server at a time
        const term = this.terminal(['ide', folder], folder, 'TIL Studio');
        return term ? { terminal: term.name, folder } : undefined;
    }

    // ── build / check ───────────────────────────────────────────────────────

    /** @param {'vm' | 'c-vm' | 'native'} kind */
    async build(arg, kind) {
        const { uri } = util.targetFrom(arg);
        const file = await util.fileOnDisk(uri);
        if (!file || !uri) return;
        const cwd = path.dirname(file);
        const rest = ['build', file];
        if (kind !== 'native') rest.push('--target', kind);
        else rest.push(String(vscode.workspace.getConfiguration('til').get('optimization', '-O2')));
        const title = { vm: 'bytecode', 'c-vm': 'C VM file', native: 'native executable' }[kind];
        const res = await vscode.window.withProgress(
            { location: vscode.ProgressLocation.Notification, title: `TIL: building ${title}`, cancellable: true },
            (_p, token) => util.execTil(this.ctx, rest, { cwd, token }));
        if (res.cancelled) return;
        if (res.error) {
            util.reportStartFailure(this.ctx, res.error);
            return;
        }
        const out = this.ctx.output;
        if (res.stdout.trim()) out.appendLine(res.stdout.trimEnd());
        if (res.stderr.trim()) out.appendLine(res.stderr.trimEnd());
        this.setCheckDiagnostics(reports.parseToolDiagnostics(res.stderr), cwd, uri);
        const artifact = reports.builtArtifact(res.stdout);
        const full = artifact ? (path.isAbsolute(artifact) ? artifact : path.join(cwd, artifact)) : null;
        if (res.code === 0 && full) {
            notify('info', `TIL: built ${path.basename(full)}`, {
                Reveal: () => vscode.commands.executeCommand('revealInExplorer', vscode.Uri.file(full)),
                'Show Output': () => out.show(true),
            });
        } else if (res.code === 0) {
            notify('info', `TIL: ${reports.stripAnsi(res.stdout).trim().split('\n').pop() || 'built'}`);
        } else {
            const first = reports.stripAnsi(res.stderr || res.stdout).trim().split('\n')[0] || `til build exited with ${res.code}`;
            notify('error', `TIL build: ${first}`, { 'Show Output': () => out.show(true) });
        }
        return { ok: res.code === 0, code: res.code, artifact: full };
    }

    async check(arg) {
        const { uri } = util.targetFrom(arg);
        const file = await util.fileOnDisk(uri);
        if (!file || !uri) return;
        const cwd = path.dirname(file);
        const res = await util.execTil(this.ctx, ['check', file], { cwd, timeoutMs: 120000 });
        if (res.error) {
            util.reportStartFailure(this.ctx, res.error);
            return;
        }
        const out = this.ctx.output;
        if (res.stdout.trim()) out.appendLine(res.stdout.trimEnd());
        if (res.stderr.trim()) out.appendLine(res.stderr.trimEnd());
        const diags = reports.parseToolDiagnostics(res.stderr);
        this.setCheckDiagnostics(diags, cwd, uri);
        if (res.code === 0) {
            vscode.window.setStatusBarMessage(`$(check) TIL: ${path.basename(file)} is OK`, 4000);
        } else {
            const n = diags.length;
            notify('warning', `TIL check: ${n ? `${n} problem${n === 1 ? '' : 's'}` : 'failed'} in ${path.basename(file)}`,
                { 'Show Output': () => out.show(true) });
        }
        return { ok: res.code === 0, code: res.code, problems: diags.length };
    }

    // ── language server ─────────────────────────────────────────────────────

    async restartServer() {
        this.ctx.refresh();
        const cfg = vscode.workspace.getConfiguration('til');
        if (!cfg.get('lsp.enabled', true)) {
            notify('info', 'TIL: the language server is off (til.lsp.enabled).',
                { 'Turn On': () => cfg.update('lsp.enabled', true, vscode.ConfigurationTarget.Global) });
            return;
        }
        await this.ctx.client.restart();
    }

    async statusMenu() {
        const client = this.ctx.client;
        const til = this.ctx.til();
        const on = vscode.workspace.getConfiguration('til').get('lsp.enabled', true);
        const items = [
            { label: '$(refresh) Restart Language Server', id: 'restart' },
            { label: '$(output) Show Language Server Output', id: 'serverOutput' },
            { label: '$(output) Show TIL Output', id: 'output' },
            { label: on ? '$(circle-slash) Turn Off the Language Server' : '$(check) Turn On the Language Server', id: 'toggle' },
            { label: '$(settings-gear) Open TIL Settings', id: 'settings' },
            { label: '$(browser) Open in TIL Studio', id: 'studio' },
        ];
        const pick = await vscode.window.showQuickPick(items, {
            title: `TIL${client.version ? ' ' + client.version : ''}: ${client.state}`,
            placeHolder: til.found ? launcher.describe(til) : (til.problem || 'TIL was not found'),
        });
        if (!pick) return;
        if (pick.id === 'restart') vscode.commands.executeCommand('til.restartServer');
        if (pick.id === 'serverOutput') client.serverOutput.show(true);
        if (pick.id === 'output') this.ctx.output.show(true);
        if (pick.id === 'toggle') {
            await vscode.workspace.getConfiguration('til').update('lsp.enabled', !on, vscode.ConfigurationTarget.Global);
        }
        if (pick.id === 'settings') vscode.commands.executeCommand('workbench.action.openSettings', '@ext:alisher-beisembekov.til-lang');
        if (pick.id === 'studio') vscode.commands.executeCommand('til.openStudio');
    }
}

module.exports = { Commands, TRANSLATIONS };
