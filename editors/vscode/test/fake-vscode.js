// A stand-in for the `vscode` module, enough to load and activate the bundled
// extension in plain node (test/smoke.js): settings, workspace folders and
// trust come from `fake.state`; registrations (commands, debug types, views)
// are recorded in `fake.calls`; everything else is a recursive stub.
'use strict';

const state = {
    settings: {},             // 'til.path' -> value
    workspaceFolders: [],     // [{uri: {fsPath, scheme: 'file'}, name, index}]
    isTrusted: true,
    activeTextEditor: undefined,
    textDocuments: [],
    language: 'en',
    shell: '/bin/bash',
};
const calls = { commands: new Map(), debugFactories: new Map(), debugProviders: [], webviewViews: new Map(), trackers: new Map(),
    outputs: new Map(), statusItems: [] };

function stub(name) {
    const fn = function () { /* stub */ };
    const props = new Map();                 // properties set by the code under test
    return new Proxy(fn, {
        get(target, prop) {
            if (prop === 'then') return undefined;                       // not a promise
            if (prop === Symbol.toPrimitive) return () => `[stub ${name}]`;
            if (prop === 'prototype') return target.prototype;
            if (!props.has(prop)) props.set(prop, stub(`${name}.${String(prop)}`));
            return props.get(prop);
        },
        set(target, prop, value) { props.set(prop, value); return true; },
        apply() { return stub(`${name}()`); },
        construct() { return stub(`new ${name}`); },
    });
}

const disposable = () => ({ dispose() { /* nothing */ } });

class Uri {
    constructor(scheme, fsPath) { this.scheme = scheme; this.fsPath = fsPath; this.path = fsPath; }
    static file(p) { return new Uri('file', p); }
    static parse(s) {
        const m = /^file:\/\/(.*)$/.exec(s);
        return m ? new Uri('file', decodeURIComponent(m[1])) : new Uri(s.split(':')[0], s.replace(/^[^:]+:/, ''));
    }
    toString() { return this.scheme === 'file' ? `file://${this.fsPath}` : `${this.scheme}:${this.path}`; }
}

class DebugAdapterExecutable {
    constructor(command, args, options) { this.command = command; this.args = args; this.options = options; }
}

const base = stub('vscode');
const overrides = {
    version: '1.82.0',
    Uri,
    DebugAdapterExecutable,
    workspace: new Proxy(stub('vscode.workspace'), {
        get(target, prop) {
            if (prop === 'getConfiguration') {
                return (section) => ({
                    get(key, def) {
                        const k = section ? `${section}.${key}` : key;
                        return Object.prototype.hasOwnProperty.call(state.settings, k) ? state.settings[k] : def;
                    },
                    inspect() { return {}; },
                    update() { return Promise.resolve(); },
                    has(key) { return Object.prototype.hasOwnProperty.call(state.settings, `${section}.${key}`); },
                });
            }
            if (prop === 'workspaceFolders') return state.workspaceFolders;
            if (prop === 'isTrusted') return state.isTrusted;
            if (prop === 'textDocuments') return state.textDocuments;
            if (prop === 'getWorkspaceFolder') return () => state.workspaceFolders[0];
            if (typeof prop === 'string' && prop.startsWith('onDid')) return () => disposable();
            return target[prop];
        },
    }),
    window: new Proxy(stub('vscode.window'), {
        get(target, prop) {
            if (prop === 'activeTextEditor') return state.activeTextEditor;
            if (prop === 'terminals') return [];
            if (prop === 'createOutputChannel') {
                return (name) => {
                    const channel = {
                        name, text: '',
                        append(t) { this.text += t; }, appendLine(t) { this.text += t + '\n'; },
                        replace(t) { this.text = t; }, clear() { this.text = ''; }, show() {}, hide() {}, dispose() {},
                    };
                    calls.outputs.set(name, channel);
                    return channel;
                };
            }
            if (prop === 'createStatusBarItem') {
                return () => {
                    const item = { text: '', tooltip: undefined, visible: false, show() { this.visible = true; },
                        hide() { this.visible = false; }, dispose() {} };
                    calls.statusItems.push(item);
                    return item;
                };
            }
            if (prop === 'registerWebviewViewProvider') {
                return (id, provider) => { calls.webviewViews.set(id, provider); return disposable(); };
            }
            if (typeof prop === 'string' && prop.startsWith('onDid')) return () => disposable();
            return target[prop];
        },
    }),
    commands: new Proxy(stub('vscode.commands'), {
        get(target, prop) {
            if (prop === 'registerCommand') {
                return (id, fn) => {
                    if (calls.commands.has(id)) throw new Error(`command registered twice: ${id}`);
                    calls.commands.set(id, fn);
                    return disposable();
                };
            }
            if (prop === 'executeCommand') return () => Promise.resolve(undefined);
            return target[prop];
        },
    }),
    debug: new Proxy(stub('vscode.debug'), {
        get(target, prop) {
            if (prop === 'registerDebugAdapterDescriptorFactory') {
                return (type, factory) => { calls.debugFactories.set(type, factory); return disposable(); };
            }
            if (prop === 'registerDebugConfigurationProvider') {
                return (type, provider, kind) => { calls.debugProviders.push({ type, provider, kind }); return disposable(); };
            }
            if (prop === 'registerDebugAdapterTrackerFactory') {
                return (type, factory) => { calls.trackers.set(type, factory); return disposable(); };
            }
            if (prop === 'activeDebugSession') return undefined;
            if (typeof prop === 'string' && prop.startsWith('onDid')) return () => disposable();
            return target[prop];
        },
    }),
    env: new Proxy(stub('vscode.env'), {
        get(target, prop) {
            if (prop === 'language') return state.language;
            if (prop === 'shell') return state.shell;
            return target[prop];
        },
    }),
};

const vscode = new Proxy(base, {
    get(target, prop) {
        if (Object.prototype.hasOwnProperty.call(overrides, prop)) return overrides[prop];
        return target[prop];
    },
});

module.exports = vscode;
module.exports.__fake = { state, calls };
