// @ts-check
'use strict';
// The `til` debug type: `til dap` as the debug adapter (time travel: VS Code
// shows Step Back and Reverse Continue because the adapter supports stepBack),
// launch configurations, and TIL's custom DAP messages: til/inputRequested ->
// an input box -> til/stdin; til/screen -> the screen panel; til/timeline and
// til/goto -> "TIL: Go to Point in Time" and the time-travel view.

const vscode = require('vscode');
const os = require('os');
const path = require('path');
const launcher = require('./launcher');
const util = require('./util');
const { ScreenPanel, TimelineView } = require('./webviews');

/** The launch attributes of the debug type (docs/TIL_DEBUGGER.md, "Launch arguments"). */
const LAUNCH_ATTRIBUTES = ['program', 'args', 'cwd', 'stdin', 'keys', 'lang', 'stopOnEntry', 'stopAtEnd', 'noFuse',
    'snapshotEvery', 'source'];

/** The configuration for F5 in a .til editor without a launch.json. */
function defaultConfiguration(folder) {
    return {
        type: 'til',
        request: 'launch',
        name: 'Debug the current TIL file',
        program: '${file}',
        cwd: folder ? '${workspaceFolder}' : '${fileDirname}',
        stopOnEntry: false,
    };
}

/** The program is waiting for a line of input: an input box (the Debug Console works too). */
class InputPrompt {
    constructor() {
        /** @type {vscode.InputBox | undefined} */
        this.box = undefined;
        /** @type {vscode.DebugSession | undefined} */
        this.session = undefined;
    }

    /** @param {vscode.DebugSession} session */
    ask(session) {
        if (this.box && this.session && this.session.id === session.id) {
            this.box.show();
            return;
        }
        this.close();
        const box = vscode.window.createInputBox();
        box.title = `${session.name}: the program is waiting for input`;
        box.prompt = 'Enter sends the line to the program (or type it in the Debug Console).';
        box.placeholder = 'input()';
        box.ignoreFocusOut = true;
        const eof = { iconPath: new vscode.ThemeIcon('debug-stop'), tooltip: 'End of input (input() returns "")' };
        box.buttons = [eof];
        box.onDidAccept(() => {
            const text = box.value;
            this.close();
            this.send(session, { text: text + '\n' });
        });
        box.onDidTriggerButton(() => {
            this.close();
            this.send(session, { eof: true });
        });
        box.onDidHide(() => {
            if (this.box === box) {
                this.box = undefined;
                this.session = undefined;
            }
            box.dispose();
        });
        this.box = box;
        this.session = session;
        box.show();
    }

    send(session, body) {
        session.customRequest('til/stdin', body).then(undefined, (e) => {
            vscode.window.showErrorMessage(`TIL: cannot send the input: ${e && e.message || e}`);
        });
    }

    /** The input went another way (the Debug Console) or the session ended. */
    closeFor(session) {
        if (this.session && this.session.id === session.id) this.close();
    }

    close() {
        const box = this.box;
        this.box = undefined;
        this.session = undefined;
        if (box) box.hide();
    }
}

class TilDebug {
    /** @param {import('./extension').Context} ctx */
    constructor(ctx) {
        this.ctx = ctx;
        this.input = new InputPrompt();
        this.screen = new ScreenPanel(ctx);
        /** ids of the til sessions that are stopped now */
        this.stoppedSessions = new Set();
        this.timeline = new TimelineView(ctx, {
            isStopped: (s) => this.stoppedSessions.has(s.id),
            goto: (s, i) => this.goto(s, i),
        });
    }

    register() {
        const subs = this.ctx.extension.subscriptions;
        const self = this;
        subs.push(
            vscode.debug.registerDebugAdapterDescriptorFactory('til', {
                createDebugAdapterDescriptor: (session) => self.descriptor(session),
            }),
            vscode.debug.registerDebugConfigurationProvider('til', {
                resolveDebugConfiguration: (folder, config) => self.resolve(folder, config),
                resolveDebugConfigurationWithSubstitutedVariables: (folder, config) => self.resolveSubstituted(folder, config),
            }),
            vscode.debug.registerDebugConfigurationProvider('til', {
                provideDebugConfigurations: (folder) => {
                    const ed = vscode.window.activeTextEditor;
                    if (!ed || ed.document.languageId !== 'til') return [];
                    return [{ ...defaultConfiguration(folder), name: `Debug ${path.basename(ed.document.uri.fsPath)}`,
                        program: ed.document.uri.scheme === 'file' ? ed.document.uri.fsPath : '${file}' }];
                },
            }, vscode.DebugConfigurationProviderTriggerKind.Dynamic),
            vscode.debug.registerDebugAdapterTrackerFactory('til', {
                createDebugAdapterTracker: (session) => self.tracker(session),
            }),
            vscode.debug.onDidReceiveDebugSessionCustomEvent((e) => self.customEvent(e)),
            vscode.debug.onDidTerminateDebugSession((s) => self.ended(s)),
            vscode.debug.onDidChangeActiveDebugSession((s) => self.timeline.refresh(s)),
            vscode.window.registerWebviewViewProvider('til.timeline', this.timeline),
            vscode.commands.registerCommand('til.debug', (arg) => self.debugFile(arg)),
            vscode.commands.registerCommand('til.gotoTime', () => self.gotoTime()),
            vscode.commands.registerCommand('til.sendInput', () => self.sendInput()),
            vscode.commands.registerCommand('til.showScreen', () => self.screen.show()),
            { dispose: () => self.screen.dispose() },
        );
    }

    // ── starting a session ──────────────────────────────────────────────────

    /** `til dap` for a session. @param {vscode.DebugSession} session */
    descriptor(session) {
        const til = this.ctx.til();
        const problem = this.ctx.problem();
        if (problem) {
            util.reportStartFailure(this.ctx, null);
            throw new Error(problem);
        }
        const argv = launcher.tilArgv(til, ['dap']);
        /** @type {any} */
        const conf = session.configuration || {};
        const folder = session.workspaceFolder && session.workspaceFolder.uri.scheme === 'file'
            ? session.workspaceFolder.uri.fsPath : undefined;
        const cwd = typeof conf.cwd === 'string' && conf.cwd ? conf.cwd : folder;
        this.ctx.log(`debug adapter: ${launcher.describe(til, ['dap'])}` + (cwd ? `   (in ${cwd})` : ''));
        return new vscode.DebugAdapterExecutable(argv.command, argv.args, { cwd, env: this.ctx.adapterEnv() });
    }

    /** Fill in a launch configuration (F5 without launch.json, the debug language). */
    resolve(folder, config) {
        if (!config.type && !config.request && !config.name) {
            const ed = vscode.window.activeTextEditor;
            if (!ed || ed.document.languageId !== 'til') {
                vscode.window.showInformationMessage('TIL: open a .til file to debug it (or add a launch configuration).');
                return undefined;
            }
            config = defaultConfiguration(folder);
        }
        config.type = 'til';
        config.request = config.request || 'launch';
        if (config.request !== 'launch') {
            vscode.window.showErrorMessage('TIL: the TIL debugger launches programs (use "request": "launch").');
            return undefined;
        }
        if (!config.program && typeof config.source !== 'string') {
            const ed = vscode.window.activeTextEditor;
            if (ed && ed.document.languageId === 'til') config.program = '${file}';
            else {
                vscode.window.showErrorMessage('TIL: the launch configuration needs a "program" (a .til file).');
                return undefined;
            }
        }
        const lang = this.ctx.explicitLang();
        if (config.lang === undefined && lang) config.lang = lang;
        return config;
    }

    /** After ${file} & co. are substituted: unsaved editors are debugged from their text. */
    resolveSubstituted(folder, config) {
        const program = typeof config.program === 'string' ? config.program : '';
        const doc = vscode.workspace.textDocuments.find((d) => d.languageId === 'til' &&
            (d.uri.fsPath === program || (d.isUntitled && d.uri.path === program)));
        if (doc && doc.isUntitled && typeof config.source !== 'string') {
            config.source = doc.getText();
            config.program = path.join(os.tmpdir(), 'til-vscode', `${path.basename(doc.uri.path) || 'untitled'}.til`);
        }
        if (!config.cwd) {
            config.cwd = folder && folder.uri.scheme === 'file' ? folder.uri.fsPath
                : (path.isAbsolute(config.program || '') ? path.dirname(config.program) : undefined);
        }
        return config;
    }

    /** The command "TIL: Debug File" (and the Debug code lens). */
    async debugFile(arg) {
        const { uri } = util.targetFrom(arg);
        if (!uri) {
            vscode.window.showWarningMessage('TIL: open a .til file first.');
            return;
        }
        const doc = util.openDocument(uri);
        if (doc && doc.isDirty && !doc.isUntitled) await doc.save();
        const folder = uri.scheme === 'file' ? vscode.workspace.getWorkspaceFolder(uri) : undefined;
        /** @type {vscode.DebugConfiguration} */
        const config = {
            type: 'til', request: 'launch', name: `Debug ${path.basename(uri.path)}`,
            program: uri.scheme === 'file' ? uri.fsPath : uri.path, cwd: util.folderOf(uri), stopOnEntry: false,
        };
        if (doc && doc.isUntitled) {
            config.source = doc.getText();
            config.program = path.join(os.tmpdir(), 'til-vscode', `${path.basename(uri.path) || 'untitled'}.til`);
        }
        await vscode.debug.startDebugging(folder, config);
    }

    // ── what the adapter says ───────────────────────────────────────────────

    /** @param {vscode.DebugSession} session */
    tracker(session) {
        return {
            onWillReceiveMessage: (m) => {
                // typed in the Debug Console while the program waits: that is its input
                if (m && m.type === 'request' && m.command === 'evaluate' && m.arguments && m.arguments.context === 'repl') {
                    this.input.closeFor(session);
                }
            },
            onDidSendMessage: (m) => {
                if (!m || m.type !== 'event') return;
                if (m.event === 'stopped') {
                    this.stoppedSessions.add(session.id);
                    this.timeline.refresh(session);
                } else if (m.event === 'continued') {
                    this.stoppedSessions.delete(session.id);
                    this.timeline.refresh(session);
                } else if (m.event === 'terminated' || m.event === 'exited') {
                    this.stoppedSessions.delete(session.id);
                    this.input.closeFor(session);
                }
            },
        };
    }

    /** @param {vscode.DebugSessionCustomEvent} e */
    customEvent(e) {
        if (!e.session || e.session.type !== 'til') return;
        if (e.event === 'til/inputRequested') this.input.ask(e.session);
        else if (e.event === 'til/screen') this.screen.frame(e.session, e.body);
    }

    /** @param {vscode.DebugSession} session */
    ended(session) {
        if (session.type !== 'til') return;
        this.stoppedSessions.delete(session.id);
        this.input.closeFor(session);
        this.screen.ended(session);
        this.timeline.ended(session);
    }

    // ── time travel ─────────────────────────────────────────────────────────

    activeSession() {
        const s = vscode.debug.activeDebugSession;
        if (s && s.type === 'til') return s;
        vscode.window.showInformationMessage('TIL: no TIL debug session is active.');
        return undefined;
    }

    async goto(session, index) {
        try {
            await session.customRequest('til/goto', { index });
        } catch (e) {
            vscode.window.showErrorMessage(`TIL: ${e && e.message || e}`);
        }
    }

    /** "TIL: Go to Point in Time": til/timeline, a line-event number, til/goto. */
    async gotoTime() {
        const session = this.activeSession();
        if (!session) return;
        let t;
        try {
            t = await session.customRequest('til/timeline');
        } catch (e) {
            vscode.window.showErrorMessage(`TIL: ${e && e.message || e}`);
            return;
        }
        const total = Number(t && t.total) || 0;
        const index = Number(t && t.index) || 0;
        const max = t && t.complete ? total : Math.max(0, total - 1);
        const parse = (v) => {
            const s = String(v).trim();
            if (/^[+-]\d+$/.test(s)) return index + parseInt(s, 10);
            if (/^\d+$/.test(s)) return parseInt(s, 10);
            return NaN;
        };
        const value = await vscode.window.showInputBox({
            title: 'TIL: go to a point in time',
            prompt: `Line event 0 to ${max} (now ${index}; ${total} recorded${t && t.complete ? ', to the end of the program' : ''}). ` +
                '+N / -N: relative to now.',
            value: String(index),
            validateInput: (v) => {
                const n = parse(v);
                return Number.isInteger(n) && n >= 0 && n <= max ? null : `A line event from 0 to ${max}, or +N / -N`;
            },
        });
        if (value === undefined) return;
        await this.goto(session, parse(value));
    }

    /** "TIL: Send Input to the Program". */
    async sendInput() {
        const session = this.activeSession();
        if (!session) return;
        const value = await vscode.window.showInputBox({
            title: `${session.name}: input`, prompt: 'A line for the program\'s input() (sent with til/stdin)',
        });
        if (value === undefined) return;
        this.input.closeFor(session);
        this.input.send(session, { text: value + '\n' });
    }
}

module.exports = { TilDebug, LAUNCH_ATTRIBUTES, defaultConfiguration };
