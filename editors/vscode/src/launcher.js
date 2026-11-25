// @ts-check
'use strict';
// Where is TIL? The command line that starts `til lsp`, `til dap` and the
// other tools, worked out from the settings, PATH and the extension's own
// location. Pure functions (no `vscode` import): file checks and the
// environment are parameters, so the tests run this in plain node.

const fs = require('fs');
const path = require('path');

/** @typedef {{isFile(p: string): boolean, isDir(p: string): boolean, isExecutable(p: string): boolean, realpath(p: string): string}} FileSystem */

/** @type {FileSystem} */
const realFs = {
    isFile(p) {
        try { return fs.statSync(p).isFile(); } catch (e) { return false; }
    },
    isDir(p) {
        try { return fs.statSync(p).isDirectory(); } catch (e) { return false; }
    },
    isExecutable(p) {
        try { fs.accessSync(p, fs.constants.X_OK); return true; } catch (e) { return false; }
    },
    realpath(p) {
        try { return fs.realpathSync(p); } catch (e) { return p; }
    },
};

/** path.win32 or path.posix, for the platform the command will run on */
function pathFor(platform) {
    return platform === 'win32' ? path.win32 : path.posix;
}

/**
 * The first `name` on PATH (Windows: with the extensions of PATHEXT).
 * @param {string} name
 * @param {{[k: string]: string | undefined}} env
 * @param {string} platform
 * @param {FileSystem} fsx
 * @returns {string | null}
 */
function which(name, env, platform, fsx) {
    const win = platform === 'win32';
    const P = pathFor(platform);
    const pathVar = (win ? (env.Path || env.PATH || env.path) : env.PATH) || '';
    const exts = win && !P.extname(name)
        ? (env.PATHEXT || '.COM;.EXE;.BAT;.CMD').split(';').filter(Boolean).map((e) => e.toLowerCase())
        : [''];
    for (let dir of pathVar.split(win ? ';' : ':')) {
        dir = dir.trim().replace(/^"(.*)"$/, '$1');
        if (!dir) continue;
        for (const ext of exts) {
            const candidate = P.join(dir, name + ext);
            if (fsx.isFile(candidate) && (win || fsx.isExecutable(candidate))) return candidate;
        }
    }
    return null;
}

/** Split a setting like "py -3" or "/usr/bin/python3" into a command and arguments. */
function splitCommand(value, fsx) {
    const v = value.trim();
    if (!v || fsx.isFile(v) || !/\s/.test(v)) return { command: v, args: [] };
    const parts = v.match(/"[^"]*"|'[^']*'|\S+/g) || [];
    const unq = parts.map((p) => p.replace(/^(["'])(.*)\1$/, '$2'));
    return { command: unq[0], args: unq.slice(1) };
}

function expandHome(p, home) {
    if (home && (p === '~' || p.startsWith('~/') || p.startsWith('~\\'))) return home + p.slice(1);
    return p;
}

/**
 * The Python that runs til.py.
 * @param {ResolveOptions} o
 * @param {FileSystem} fsx
 * @returns {{command: string, args: string[], found: boolean}}
 */
function resolvePython(o, fsx) {
    const env = o.env || {};
    const setting = expandHome((o.python || '').trim(), o.home);
    if (setting) {
        const s = splitCommand(setting, fsx);
        const found = /[\\/]/.test(s.command) ? fsx.isFile(s.command) : !!which(s.command, env, o.platform, fsx);
        return { command: s.command, args: s.args, found };
    }
    const candidates = o.platform === 'win32'
        ? [['python', []], ['py', ['-3']], ['python3', []]]
        : [['python3', []], ['python', []]];
    for (const [name, args] of candidates) {
        const found = which(/** @type {string} */ (name), env, o.platform, fsx);
        if (found) return { command: found, args: /** @type {string[]} */ (args), found: true };
    }
    return { command: o.platform === 'win32' ? 'python' : 'python3', args: [], found: false };
}

/**
 * @typedef {object} ResolveOptions
 * @property {string} [tilPath]        the setting til.path
 * @property {string} [python]         the setting til.python
 * @property {string} [legacyPath]     the deprecated setting til.compilerPath
 * @property {string} [extensionPath]  where the extension is installed
 * @property {string[]} [workspaceFolders]  folders searched for a TIL checkout (trusted workspaces only)
 * @property {string} platform         process.platform of the machine that runs til
 * @property {{[k: string]: string | undefined}} [env]   its environment (PATH, PATHEXT, ComSpec)
 * @property {string} [home]           its home directory
 * @property {FileSystem} [fs]
 */

/**
 * @typedef {object} TilCommand
 * @property {string} command    the program to start
 * @property {string[]} args     arguments before the til command (`lsp`, `run`, ...)
 * @property {boolean} found     a launcher / til.py was found (else command is a guess)
 * @property {string} source     setting | path | extension | workspace | home | missing
 * @property {string} [tilPy]    the til.py that runs (when known)
 * @property {string} [launcher] the launcher that runs (when it is not til.py)
 * @property {boolean} [pythonFound]  false: no Python was found for tilPy
 * @property {string} [problem]  why nothing was found / the setting is wrong
 */

/**
 * The command line of TIL (`<command> <args...> <til command> ...`).
 * Order: the til.path setting; `til` on PATH; the checkout this extension runs
 * from (editors/vscode inside a TIL repository); a TIL checkout opened as a
 * workspace folder; the default install location ~/.til/bin/til.py.
 * @param {ResolveOptions} o
 * @returns {TilCommand}
 */
function resolveTil(o) {
    const fsx = o.fs || realFs;
    const env = o.env || {};
    const win = o.platform === 'win32';
    const P = pathFor(o.platform);

    /** @returns {TilCommand} */
    const fromTilPy = (tilPy, source) => {
        const py = resolvePython(o, fsx);
        return { command: py.command, args: [...py.args, tilPy], found: true, source, tilPy, pythonFound: py.found };
    };
    /** @returns {TilCommand} */
    const fromLauncher = (launcher, source) => {
        if (/\.py$/i.test(launcher)) return fromTilPy(launcher, source);
        if (win && /\.(bat|cmd)$/i.test(launcher)) {
            // the installer puts til.py next to til.bat: run it directly (a .bat
            // needs a shell, which the debug adapter and the language client do not use)
            const next = P.join(P.dirname(launcher), 'til.py');
            if (fsx.isFile(next)) return fromTilPy(next, source);
            return { command: env.ComSpec || 'cmd.exe', args: ['/d', '/c', launcher], found: true, source, launcher };
        }
        if (win && /\.ps1$/i.test(launcher)) {
            const next = P.join(P.dirname(launcher), 'til.py');
            if (fsx.isFile(next)) return fromTilPy(next, source);
            return { command: 'powershell.exe', args: ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', launcher],
                found: true, source, launcher };
        }
        return { command: launcher, args: [], found: true, source, launcher };
    };
    /** @returns {TilCommand} */
    const missing = (problem, source = 'missing') => ({ command: 'til', args: [], found: false, source, problem });

    // 1. the setting (til.path; the old til.compilerPath when it is not the default)
    let setting = expandHome((o.tilPath || '').trim(), o.home);
    if (!setting) {
        const legacy = (o.legacyPath || '').trim();
        if (legacy && legacy !== 'til') setting = expandHome(legacy, o.home);
    }
    if (setting) {
        if (!P.isAbsolute(setting) && /[\\/]/.test(setting) && o.workspaceFolders && o.workspaceFolders[0]) {
            setting = P.join(o.workspaceFolders[0], setting);       // relative to the workspace
        }
        if (fsx.isDir(setting)) {
            for (const c of [P.join(setting, 'src', 'til.py'), P.join(setting, 'til.py'), P.join(setting, 'bin', 'til.py')]) {
                if (fsx.isFile(c)) return fromTilPy(c, 'setting');
            }
            return missing(`til.path is a folder without til.py: ${setting}`, 'setting');
        }
        if (fsx.isFile(setting)) return fromLauncher(setting, 'setting');
        if (!/[\\/]/.test(setting)) {                                 // a command name: look it up
            const found = which(setting, env, o.platform, fsx);
            if (found) return fromLauncher(found, 'setting');
        }
        return missing(`til.path does not exist: ${setting}`, 'setting');
    }
    // 2. `til` on PATH
    const onPath = which('til', env, o.platform, fsx);
    if (onPath) return fromLauncher(onPath, 'path');
    // 3. the TIL checkout this extension runs from (<repo>/editors/vscode)
    if (o.extensionPath) {
        const ext = fsx.realpath(o.extensionPath);
        const repoTil = P.join(ext, '..', '..', 'src', 'til.py');
        if (fsx.isFile(repoTil)) return fromTilPy(repoTil, 'extension');
    }
    // 4. a TIL checkout opened as a workspace folder
    for (const folder of o.workspaceFolders || []) {
        const c = P.join(folder, 'src', 'til.py');
        if (fsx.isFile(c) && fsx.isFile(P.join(folder, 'src', 'tillsp.py'))) return fromTilPy(c, 'workspace');
    }
    // 5. the installer's location (not on PATH in a session started from the desktop)
    if (o.home) {
        const c = P.join(o.home, '.til', 'bin', 'til.py');
        if (fsx.isFile(c)) return fromTilPy(c, 'home');
    }
    return missing('TIL was not found: no `til` on PATH and the til.path setting is empty');
}

/**
 * The argv of a TIL command (`lsp`, `dap`, `run file.til`, ...).
 * @param {TilCommand} til
 * @param {string[]} rest
 * @returns {{command: string, args: string[]}}
 */
function tilArgv(til, rest) {
    return { command: til.command, args: [...til.args, ...rest] };
}

/** A short, readable form of a command line (for status bars and logs). */
function describe(til, rest = []) {
    return [til.command, ...til.args, ...rest].map((a) => (/[\s"']/.test(a) ? JSON.stringify(a) : a)).join(' ');
}

// ── command lines for terminals ─────────────────────────────────────────────

/**
 * Which quoting the shell of a terminal needs.
 * @param {string | undefined} shellPath  vscode.env.shell (or a terminal's shell)
 * @param {string} platform
 * @returns {'posix' | 'pwsh' | 'cmd'}
 */
function shellKind(shellPath, platform) {
    const base = (shellPath || '').toLowerCase().replace(/^.*[\\/]/, '').replace(/\.exe$/, '');
    if (base === 'pwsh' || base === 'powershell') return 'pwsh';
    if (base === 'cmd') return 'cmd';
    if (!base) return platform === 'win32' ? 'pwsh' : 'posix';
    return 'posix';                       // bash, zsh, fish, sh, Git Bash, nu ...
}

/** Quote one argument for a shell. */
function quoteArg(arg, kind) {
    if (arg === '') return kind === 'cmd' ? '""' : "''";
    if (kind === 'pwsh') return /^[\w+=:,./\\-]+$/.test(arg) ? arg : `'${arg.replace(/'/g, "''")}'`;
    if (kind === 'cmd') return /^[\w+=:,./\\-]+$/.test(arg) ? arg : `"${arg.replace(/"/g, '""')}"`;
    return /^[\w@%+=:,./-]+$/.test(arg) ? arg : `'${arg.replace(/'/g, "'\\''")}'`;
}

/**
 * A command line to type into a terminal.
 * @param {string[]} argv
 * @param {'posix' | 'pwsh' | 'cmd'} kind
 */
function commandLine(argv, kind) {
    const q = argv.map((a) => quoteArg(a, kind));
    // PowerShell runs a quoted program name only with the call operator
    return kind === 'pwsh' ? `& ${q.join(' ')}` : q.join(' ');
}

module.exports = {
    realFs, which, resolvePython, resolveTil, tilArgv, describe, shellKind, quoteArg, commandLine, splitCommand,
};
