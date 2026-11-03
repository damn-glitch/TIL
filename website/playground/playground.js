/*
 * TIL Playground - editor, compile pipeline and VM runner.
 *
 * Pipeline (see README.md):
 *   1. VM instance A runs the self-hosted compiler tilc.tbc with the virtual
 *      file system {"/main.til": source} and args ["/main.til", "/main.tbc"].
 *   2. VM instance B loads /main.tbc and runs it in slices (a few milliseconds
 *      per animation frame), so the page stays responsive; wait_key(),
 *      input() and sleep_ms() pause the program without blocking the page.
 * Without tilc.tbc (development builds) the bundled examples run from
 * bytecode precompiled by the desktop compiler (examples/*.tbc).
 */
(function () {
    'use strict';

    const { Runtime, STATUS, keyCode, memoryFS } = window.TILWasm;
    const $ = (id) => document.getElementById(id);

    const ui = {
        code: $('code'), gutter: $('gutter'), errorLine: $('error-line'),
        run: $('run'), stop: $('stop'), share: $('share'), theme: $('theme'), clear: $('clear'),
        examples: $('examples'), output: $('output'), stdin: $('stdin'), stdinText: $('stdin-text'),
        state: $('state'), stats: $('stats'), screenWrap: $('screen-wrap'), screen: $('screen'),
        compilerInfo: $('compiler-info'), vmInfo: $('vm-info'), toast: $('toast'),
    };

    const DRAFT_KEY = 'til-playground-draft';
    const THEME_KEY = 'til-playground-theme';
    const SLICE_START = 20000;          /* steps per wh_run call (adapted to the machine) */
    const SLICE_MIN = 1000;
    const SLICE_MAX = 4000000;
    const FRAME_MS = 10;                /* VM time per animation frame */
    const OUTPUT_MAX = 400000;          /* characters kept in the output panel */
    const OUTPUT_CHUNK = 8192;          /* characters per output block */
    const FRAME_OUTPUT = 100000;        /* stop a frame's slices early after this much output */
    const INDENT = '    ';

    const app = {
        runtime: null,
        compilerVM: null,
        programVM: null,
        compiler: { kind: 'loading', bytes: null, url: '' },
        examples: [],
        currentExample: '',
        run: null,
        errorLineNo: 0,
        lineCount: 0,
        screenFrame: -1,
        screenSize: null,
        outChars: 0,
        trimmed: false,
        pending: [],
        pendingChars: 0,
        chunk: null,
        chunkLen: 0,
        chunkEndsLine: true,
    };

    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    /* ───────────────────────────── state and messages ──────────────────── */
    function setState(state, text) {
        document.body.dataset.state = state;
        if (text !== undefined) ui.state.textContent = text;
        const active = ['compiling', 'running', 'sleeping', 'waiting-key', 'waiting-line'].includes(state);
        ui.stop.disabled = !active;
        ui.run.disabled = !app.runtime;
    }

    let toastTimer = 0;
    function toast(text) {
        ui.toast.textContent = text;
        ui.toast.hidden = false;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { ui.toast.hidden = true; }, 2600);
    }

    /* ───────────────────────────── output panel ────────────────────────── */
    /* Output goes into block-level chunks of about OUTPUT_CHUNK characters
     * (split after a newline), so appending only lays out the newest chunk and
     * trimming removes whole old chunks. */
    function appendOutput(text, cls) {
        if (!text) return;
        app.pending.push([text, cls || '']);
        app.pendingChars += text.length;
    }

    function newChunk() {
        const div = document.createElement('div');
        ui.output.appendChild(div);
        app.chunk = div;
        app.chunkLen = 0;
    }

    function addToChunk(text, cls) {
        const chunk = app.chunk;
        const last = chunk.lastChild;
        const lastCls = last ? (last.nodeType === 3 ? '' : last.className) : null;
        if (last && lastCls === cls) {
            (last.nodeType === 3 ? last : last.firstChild).appendData(text);
        } else if (cls) {
            const span = document.createElement('span');
            span.className = cls;
            span.textContent = text;
            chunk.appendChild(span);
        } else {
            chunk.appendChild(document.createTextNode(text));
        }
        app.chunkLen += text.length;
        app.outChars += text.length;
    }

    function appendToDom(text, cls) {
        while (text.length) {
            if (!app.chunk || (app.chunkLen >= OUTPUT_CHUNK && app.chunkEndsLine)) newChunk();
            let piece = text;
            if (app.chunkLen + text.length > OUTPUT_CHUNK) {
                const nl = text.indexOf('\n', Math.max(0, OUTPUT_CHUNK - app.chunkLen - 1));
                if (nl >= 0) piece = text.slice(0, nl + 1);
            }
            addToChunk(piece, cls);
            app.chunkEndsLine = piece.endsWith('\n');
            text = text.slice(piece.length);
        }
    }

    function flushOutput() {
        let items = app.pending;
        if (!items.length) return;
        app.pending = [];
        app.pendingChars = 0;
        const total = items.reduce((n, it) => n + it[0].length, 0);
        if (total > OUTPUT_MAX) {               /* only the newest output can stay anyway */
            const kept = [];
            let n = 0;
            for (let i = items.length - 1; i >= 0 && n < OUTPUT_MAX; i--) {
                const [text, cls] = items[i];
                const piece = text.length > OUTPUT_MAX - n ? text.slice(text.length - (OUTPUT_MAX - n)) : text;
                kept.push([piece, cls]);
                n += piece.length;
            }
            items = kept.reverse();
            clearOutput();
            noteTrimmed();
        }
        const out = ui.output;
        const atBottom = out.scrollHeight - out.scrollTop - out.clientHeight < 48;
        for (const [text, cls] of items) appendToDom(text, cls);
        if (app.outChars > OUTPUT_MAX) trimOutput();
        if (atBottom) out.scrollTop = out.scrollHeight;
    }

    function trimOutput() {
        const out = ui.output;
        while (app.outChars > OUTPUT_MAX * 0.75 && out.firstChild && out.firstChild !== app.chunk) {
            app.outChars -= out.firstChild.textContent.length;
            out.removeChild(out.firstChild);
        }
        noteTrimmed();
    }

    function noteTrimmed() {
        if (!app.trimmed) {
            app.trimmed = true;
            toast('Long output: only the newest lines are kept');
        }
    }

    function clearOutput() {
        app.pending = [];
        app.pendingChars = 0;
        ui.output.textContent = '';
        app.outChars = 0;
        app.chunk = null;
        app.chunkLen = 0;
        app.chunkEndsLine = true;
    }

    function note(text) {
        const last = app.pending.length ? app.pending[app.pending.length - 1][0]
            : (ui.output.lastChild ? ui.output.lastChild.textContent : '\n');
        appendOutput((last.endsWith('\n') ? '' : '\n') + text + '\n', 'o-note');
    }

    function fmtMs(ms) {
        return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
    }

    function fmtBytes(n) {
        return n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;
    }

    /* ───────────────────────────── editor ──────────────────────────────── */
    function lineHeight() {
        return parseFloat(getComputedStyle(ui.code).lineHeight) || 20;
    }

    function renderGutter() {
        const n = ui.code.value.split('\n').length;
        app.lineCount = n;
        const parts = [];
        for (let i = 1; i <= n; i++) {
            parts.push(i === app.errorLineNo ? `<span class="err">${i}</span>` : String(i));
        }
        ui.gutter.innerHTML = parts.join('\n') + '\n';
        ui.gutter.scrollTop = ui.code.scrollTop;
    }

    function updateGutter() {
        const n = ui.code.value.split('\n').length;
        if (n !== app.lineCount) renderGutter();
    }

    function positionErrorLine() {
        if (!app.errorLineNo) { ui.errorLine.hidden = true; return; }
        const pad = parseFloat(getComputedStyle(ui.code).paddingTop) || 0;
        ui.errorLine.hidden = false;
        ui.errorLine.style.top = `${pad + (app.errorLineNo - 1) * lineHeight() - ui.code.scrollTop}px`;
    }

    function markErrorLine(n) {
        app.errorLineNo = n > 0 && n <= app.lineCount ? n : 0;
        renderGutter();
        positionErrorLine();
        if (app.errorLineNo) {
            const lh = lineHeight();
            const top = (app.errorLineNo - 1) * lh;
            if (top < ui.code.scrollTop || top > ui.code.scrollTop + ui.code.clientHeight - 2 * lh) {
                ui.code.scrollTop = Math.max(0, top - ui.code.clientHeight / 3);
            }
            document.body.dataset.errorLine = String(app.errorLineNo);
        } else {
            delete document.body.dataset.errorLine;
        }
    }

    function clearErrorLine() {
        if (app.errorLineNo) markErrorLine(0);
    }

    /* Insert text at the selection, keeping the browser's undo history. */
    function insertText(text) {
        const ta = ui.code;
        ta.focus();
        let done = false;
        try { done = document.execCommand('insertText', false, text); } catch (e) { done = false; }
        if (!done) {
            ta.setRangeText(text, ta.selectionStart, ta.selectionEnd, 'end');
            ta.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }

    function setCode(text) {
        const ta = ui.code;
        ta.value = text;
        ta.scrollTop = 0;
        ta.setSelectionRange(0, 0);
        clearErrorLine();
        renderGutter();
    }

    const BLOCK_WORDS = new Set([
        'if', 'elif', 'else', 'while', 'for', 'loop', 'match', 'struct', 'enum', 'impl', 'trait', 'fn',
        'retry', 'timeout', 'intent', 'handle', 'with',
        'егер', 'әйтпесе', 'үшін', 'кезінде', 'цикл', 'функция', 'құрылым', 'санақ', 'іске', 'қасиет',
        'сәйкестік', 'қайталау', 'ниет',
        'eger', 'aitpese', 'ushin', 'kezinde', 'tsikl', 'funktsiya', 'qurylym', 'sanaq', 'iske', 'qasiet',
        'saikestik', 'qaitalau', 'niet',
        'если', 'иначе', 'для', 'пока', 'структура', 'перечисление', 'реализация', 'типаж', 'выбор',
        'повторить', 'таймаут', 'намерение',
    ]);
    const ENTRY_NAMES = new Set(['main', 'негізгі', 'negizgi', 'главная']);

    /* Does this line open an indented block? (TIL blocks have no ':' marker) */
    function opensBlock(line) {
        const t = line.trim();
        if (!t || t.startsWith('#')) return false;
        const first = t.split(/[\s(]/)[0];
        if (BLOCK_WORDS.has(first)) return true;
        if (/=>\s*$/.test(t)) return true;                       /* match arm with a block */
        if (/\b(retry|timeout)\s*\([^)]*\)\s*$/.test(t)) return true;
        const def = /^([\p{L}_][\p{L}\p{N}_]*)\s*\((.*)\)\s*(->.*)?$/u.exec(t);
        if (def) return Boolean(def[3]) || def[2].includes(':') || ENTRY_NAMES.has(def[1]) ||
            /^self\b/.test(def[2].trim());
        return false;
    }

    function selectedLines() {
        const ta = ui.code, v = ta.value;
        const start = v.lastIndexOf('\n', ta.selectionStart - 1) + 1;
        let end = ta.selectionEnd;
        if (end > ta.selectionStart && v[end - 1] === '\n') end--;
        const endNl = v.indexOf('\n', end);
        return { start, end: endNl < 0 ? v.length : endNl };
    }

    function indentSelection(outdent) {
        const ta = ui.code;
        const { start, end } = selectedLines();
        const lines = ta.value.slice(start, end).split('\n');
        const changed = lines.map((l) => {
            if (!outdent) return l.length ? INDENT + l : l;
            const m = /^ {1,4}|^\t/.exec(l);
            return m ? l.slice(m[0].length) : l;
        }).join('\n');
        ta.setSelectionRange(start, end);
        insertText(changed);
        ta.setSelectionRange(start, start + changed.length);
    }

    function onEditorKey(e) {
        const ta = ui.code;
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            runProgram();
            return;
        }
        if (e.key === 'Escape') {
            if (app.run && !app.run.done && !app.run.stopped) { e.preventDefault(); stopProgram(); }
            else ta.blur();
            return;
        }
        if (e.key === 'Tab' && !e.ctrlKey && !e.metaKey && !e.altKey) {
            e.preventDefault();
            const multi = ta.value.slice(ta.selectionStart, ta.selectionEnd).includes('\n');
            if (e.shiftKey || multi) indentSelection(e.shiftKey);
            else insertText(INDENT);
            return;
        }
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey && !e.isComposing) {
            e.preventDefault();
            const v = ta.value;
            const lineStart = v.lastIndexOf('\n', ta.selectionStart - 1) + 1;
            const line = v.slice(lineStart, ta.selectionStart);
            let indent = /^[ \t]*/.exec(line)[0];
            if (opensBlock(line)) indent += INDENT;
            insertText('\n' + indent);
            return;
        }
        if (e.key === 'Backspace' && ta.selectionStart === ta.selectionEnd && !e.ctrlKey && !e.metaKey) {
            const v = ta.value, pos = ta.selectionStart;
            const lineStart = v.lastIndexOf('\n', pos - 1) + 1;
            const before = v.slice(lineStart, pos);
            if (before.length && /^ +$/.test(before)) {
                e.preventDefault();
                const remove = before.length % 4 || 4;
                ta.setSelectionRange(pos - remove, pos);
                insertText('');
            }
        }
    }

    let draftTimer = 0;
    function onEditorInput() {
        updateGutter();
        if (app.errorLineNo) clearErrorLine();
        clearTimeout(draftTimer);
        draftTimer = setTimeout(() => {
            try { localStorage.setItem(DRAFT_KEY, ui.code.value); } catch (e) { /* storage disabled */ }
        }, 300);
        if (location.hash) history.replaceState(null, '', location.pathname + location.search);
        if (app.currentExample) {
            app.currentExample = '';
            ui.examples.value = '';
        }
    }

    /* ───────────────────────────── theme ───────────────────────────────── */
    function effectiveTheme() {
        const t = document.documentElement.dataset.theme;
        if (t === 'light' || t === 'dark') return t;
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    function toggleTheme() {
        const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
        document.documentElement.dataset.theme = next;
        try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage disabled */ }
    }

    /* ───────────────────────────── examples ────────────────────────────── */
    async function fetchBytes(url) {
        const resp = await fetch(url, { cache: 'no-cache' });
        if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
        return new Uint8Array(await resp.arrayBuffer());
    }

    async function loadExampleIndex() {
        try {
            const resp = await fetch('examples/index.json', { cache: 'no-cache' });
            app.examples = await resp.json();
        } catch (e) {
            app.examples = [];
            appendOutput(`cannot load the example list: ${e.message}\n`, 'o-err');
        }
        for (const ex of app.examples) {
            const opt = document.createElement('option');
            opt.value = ex.id;
            opt.textContent = ex.title;
            ui.examples.appendChild(opt);
        }
    }

    async function exampleSource(ex) {
        if (ex.source === undefined) ex.source = decoder.decode(await fetchBytes(`examples/${ex.id}.til`));
        return ex.source;
    }

    async function loadExample(id) {
        const ex = app.examples.find((e) => e.id === id);
        if (!ex) { toast(`no example "${id}"`); return false; }
        try {
            setCode(await exampleSource(ex));
        } catch (e) {
            toast(`cannot load the example: ${e.message}`);
            return false;
        }
        app.currentExample = id;
        ui.examples.value = id;
        try { localStorage.setItem(DRAFT_KEY, ui.code.value); } catch (e) { /* storage disabled */ }
        return true;
    }

    const normalize = (s) => s.replace(/\r\n?/g, '\n').replace(/[ \t]+$/gm, '').replace(/\n+$/, '');

    /* Development fallback: bytecode precompiled by src/til.py for an unmodified example. */
    async function precompiledFor(source) {
        const want = normalize(source);
        for (const ex of app.examples) {
            let src;
            try { src = await exampleSource(ex); } catch (e) { continue; }
            if (normalize(src) === want) {
                if (!ex.tbc) ex.tbc = await fetchBytes(`examples/${ex.id}.tbc`);
                return ex.tbc;
            }
        }
        return null;
    }

    /* ───────────────────────────── share links ─────────────────────────── */
    function b64url(bytes) {
        let s = '';
        for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
        return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    }

    function fromB64url(text) {
        let t = text.replace(/-/g, '+').replace(/_/g, '/');
        while (t.length % 4) t += '=';
        const bin = atob(t);
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out;
    }

    async function transform(bytes, stream) {
        const piped = new Blob([bytes]).stream().pipeThrough(stream);
        return new Uint8Array(await new Response(piped).arrayBuffer());
    }

    async function shareHash(source) {
        const bytes = encoder.encode(source);
        if (typeof CompressionStream === 'function') {
            try { return 'z=' + b64url(await transform(bytes, new CompressionStream('deflate-raw'))); } catch (e) {
                /* no deflate-raw: plain text below */
            }
        }
        return 's=' + b64url(bytes);
    }

    async function sourceFromHash(hash) {
        const params = new URLSearchParams(hash.replace(/^#/, ''));
        if (params.has('z')) {
            if (typeof DecompressionStream !== 'function') throw new Error('this browser cannot open compressed links');
            return decoder.decode(await transform(fromB64url(params.get('z')), new DecompressionStream('deflate-raw')));
        }
        if (params.has('s')) return decoder.decode(fromB64url(params.get('s')));
        if (params.has('example')) return { example: params.get('example') };
        return null;
    }

    async function shareLink() {
        const hash = await shareHash(ui.code.value);
        const url = `${location.origin}${location.pathname}#${hash}`;
        history.replaceState(null, '', `#${hash}`);
        ui.share.dataset.url = url;
        try {
            await navigator.clipboard.writeText(url);
            toast('Link copied to the clipboard');
        } catch (e) {
            toast('Link ready: copy it from the address bar');
        }
    }

    async function loadFromHash() {
        if (!location.hash || location.hash.length < 3) return false;
        try {
            const r = await sourceFromHash(location.hash);
            if (r === null) return false;
            if (typeof r === 'object') return loadExample(r.example);
            setCode(r);
            app.currentExample = '';
            ui.examples.value = '';
            return true;
        } catch (e) {
            toast(`cannot open the shared code: ${e.message}`);
            return false;
        }
    }

    /* ───────────────────────────── screen ──────────────────────────────── */
    function openScreen(w, h) {
        ui.screenWrap.hidden = false;
        if (ui.screen.width !== w || ui.screen.height !== h) {
            ui.screen.width = w;
            ui.screen.height = h;
        }
        app.screenSize = [w, h];
        app.screenFrame = -1;
        fitScreen();
        if (document.activeElement !== ui.screen) ui.screen.focus({ preventScroll: true });
    }

    function fitScreen() {
        if (!app.screenSize || ui.screenWrap.hidden) return;
        const [w, h] = app.screenSize;
        const maxW = Math.max(64, ui.screenWrap.clientWidth - 20);
        const maxH = Math.max(96, window.innerHeight * 0.5);
        let scale = Math.min(maxW / w, maxH / h);
        if (scale >= 1) scale = Math.floor(scale);           /* crisp integer scaling */
        ui.screen.style.width = `${Math.max(1, Math.round(w * scale))}px`;
        ui.screen.style.height = `${Math.max(1, Math.round(h * scale))}px`;
    }

    function closeScreen() {
        ui.screenWrap.hidden = true;
        app.screenSize = null;
        app.screenFrame = -1;
    }

    function presentScreen() {
        if (!app.runtime) return;
        const s = app.runtime.screen();
        if (!s || s.owner !== app.programVM.id) return;
        if (!app.screenSize || app.screenSize[0] !== s.width || app.screenSize[1] !== s.height) openScreen(s.width, s.height);
        if (s.frame === app.screenFrame) return;
        app.screenFrame = s.frame;
        ui.screen.getContext('2d').putImageData(new ImageData(s.pixels, s.width, s.height), 0, 0);
        ui.screen.dataset.frame = String(s.frame);
    }

    /* ───────────────────────────── running ─────────────────────────────── */
    /* The source line of the first diagnostic in a compiler or VM report:
     *   " --> /main.til:3:9"              (src/til.py style)
     *   "tilc: /main.til:3: error: ..."   (FILE:LINE[:COL]: error|warning)
     *   "  at main (line 3)"              (runtime error traces) */
    function errorLineFrom(text) {
        const m = /-->\s*[^\n]*?:(\d+):\d+/.exec(text) ||
            /[^\s:]+:(\d+)(?::\d+)?:\s*(?:error|warning)\b/i.exec(text) ||
            /\(line (\d+)\)/.exec(text) || /\bline (\d+)/i.exec(text);
        return m ? parseInt(m[1], 10) : 0;
    }

    class Run {
        constructor(source) {
            this.source = source;
            this.stopped = false;
            this.done = false;
            this.phase = 'compile';
            this.vm = null;
            this.raf = 0;
            this.timer = 0;
            this.slice = SLICE_START;
            this.sleepUntil = 0;
            this.started = performance.now();
            this.runStarted = 0;
            this.compileMs = 0;
        }

        async start() {
            let tbc;
            if (app.compiler.kind === 'tilc' || app.compiler.kind === 'custom') {
                setState('compiling', 'Compiling with tilc…');
                tbc = await this.compile();
            } else {
                setState('compiling', 'Loading the precompiled example…');
                try {
                    tbc = await precompiledFor(this.source);
                } catch (e) {
                    appendOutput(`cannot load the precompiled example: ${e.message}\n`, 'o-err');
                    tbc = null;
                }
                if (this.stopped) return;
                if (!tbc) { this.noCompiler(); return; }
            }
            if (!tbc || this.stopped) return;
            this.startProgram(tbc);
        }

        noCompiler() {
            appendOutput('This playground build has no in-browser compiler yet, so only the bundled ' +
                'examples can run, unchanged (from bytecode precompiled by the desktop compiler).\n' +
                'The self-hosted compiler tilc.tbc is added by website/playground/build.sh once ' +
                'selfhost/tilc.til exists.\n\nTo run your own code now, install TIL and run:  til run main.til\n',
            'o-err');
            this.finish('no-compiler', 'Cannot compile: no in-browser compiler in this build');
        }

        compile() {
            return new Promise((resolve) => {
                const fs = memoryFS();
                fs.write('/main.til', encoder.encode(this.source));
                app.runtime.fs = fs;
                const vm = app.compilerVM;
                const log = [];
                vm.onOutput = (text, stream) => log.push([text, stream]);
                vm.onScreen = null;
                try {
                    vm.load(app.compiler.bytes);
                    vm.setArgs(['/main.til', '/main.tbc']);
                } catch (e) {
                    appendOutput(`cannot start the compiler: ${e.message}\n`, 'o-err');
                    this.finish('compile-error', 'Cannot start the compiler');
                    resolve(null);
                    return;
                }
                this.resolveCompile = resolve;
                this.drive(vm, (st) => {
                    this.resolveCompile = null;
                    this.compileMs = performance.now() - this.started;
                    const ok = st === STATUS.DONE || (st === STATUS.EXIT && vm.exitCode === 0);
                    const tbc = fs.read('/main.tbc');
                    if (ok && tbc && tbc.length) { resolve(tbc); return; }
                    const text = log.map((x) => x[0]).join('');
                    for (const [t, stream] of log) appendOutput(t, stream === 2 ? 'o-err' : 'o-compile');
                    if (ok) note('the compiler finished without writing /main.tbc');
                    const line = errorLineFrom(text);
                    markErrorLine(line);
                    this.finish('compile-error', line ? `Compile error (line ${line})` : 'Compile error');
                    resolve(null);
                });
            });
        }

        startProgram(tbc) {
            const vm = app.programVM;
            vm.onOutput = (text, stream) => appendOutput(text, stream === 2 ? 'o-err' : '');
            vm.onScreen = (ev, a, b) => { if (ev === 'open') openScreen(a, b); };
            try {
                vm.load(tbc);
                vm.setArgs([]);
            } catch (e) {
                appendOutput(`cannot load the program: ${e.message}\n`, 'o-err');
                this.finish('error', 'Cannot load the program');
                return;
            }
            this.phase = 'run';
            this.slice = SLICE_START;
            this.runStarted = performance.now();
            setState('running', 'Running…');
            this.drive(vm, (st) => this.programFinished(st));
        }

        programFinished(st) {
            const vm = this.vm;
            const ms = fmtMs(performance.now() - this.runStarted);
            if (st === STATUS.DONE || st === STATUS.EXIT) {
                const code = vm.exitCode;
                this.finish('done', code === 0 ? `Finished in ${ms}` : `Exited with code ${code} after ${ms}`);
                return;
            }
            const line = errorLineFrom(vm.errorText);
            markErrorLine(line);
            if (st === STATUS.ERROR) this.finish('error', line ? `Runtime error at line ${line}` : 'Runtime error');
            else this.finish('error', 'Stopped by the playground (see the output)');
        }

        finish(state, text) {
            this.done = true;
            this.cancelTimers();
            hideStdin();
            flushOutput();
            presentScreen();
            updateStats(this);
            setState(state, text);
        }

        drive(vm, onFinish) {
            this.vm = vm;
            this.onFinish = onFinish;
            this.schedule(0);
        }

        schedule(delay) {
            if (this.stopped || this.done) return;
            if (delay > 0) {
                this.timer = setTimeout(() => { this.timer = 0; this.schedule(0); }, delay);
            } else if (document.hidden) {
                this.timer = setTimeout(() => { this.timer = 0; this.step(); }, 0);
            } else {
                this.raf = requestAnimationFrame(() => { this.raf = 0; this.step(); });
            }
        }

        step() {
            if (this.stopped || this.done) return;
            const vm = this.vm;
            const start = performance.now();
            if (this.sleepUntil) {
                const left = this.sleepUntil - start;
                if (left > 1) { this.schedule(left > 30 ? left - 12 : 0); return; }
                this.sleepUntil = 0;
                setState(this.phase === 'run' ? 'running' : 'compiling', this.phase === 'run' ? 'Running…' : 'Compiling with tilc…');
            }
            let st;
            for (;;) {
                const t = performance.now();
                st = vm.run(this.slice);
                const dt = performance.now() - t;
                if (st !== STATUS.YIELD) break;
                if (dt < 2 && this.slice < SLICE_MAX) this.slice *= 2;
                else if (dt > 8 && this.slice > SLICE_MIN) this.slice = Math.max(SLICE_MIN, this.slice >> 1);
                if (performance.now() - start >= FRAME_MS || app.pendingChars > FRAME_OUTPUT) break;
            }
            flushOutput();
            if (this.phase === 'run') presentScreen();
            updateStats(this);
            switch (st) {
            case STATUS.YIELD:
                this.schedule(0);
                return;
            case STATUS.SLEEP: {
                const ms = vm.sleepMs;
                if (ms <= 0) { this.schedule(0); return; }
                this.sleepUntil = performance.now() + ms;
                if (ms > 50) setState('sleeping', `Sleeping (sleep_ms ${ms})…`);
                this.schedule(ms > 30 ? ms - 12 : 0);
                return;
            }
            case STATUS.WAIT_KEY:
                if (vm.keys.length) { this.schedule(0); return; }
                setState('waiting-key', finePointer()
                    ? 'Waiting for a key: click the screen (or the output) and press a key'
                    : 'Waiting for a key: use the keys under the screen');
                focusKeyTarget();
                return;
            case STATUS.WAIT_LINE:
                setState('waiting-line', 'Waiting for input: type a line and press Enter');
                showStdin();
                return;
            default: {
                /* this VM finished; the Run itself ends in finish() (the
                 * compiler VM finishing starts the program VM instead) */
                this.cancelTimers();
                const cb = this.onFinish;
                this.onFinish = null;
                cb(st);
            }
            }
        }

        resume() {
            if (this.stopped || this.done) return;
            const s = document.body.dataset.state;
            if (s === 'waiting-key' || s === 'waiting-line') {
                setState(this.phase === 'run' ? 'running' : 'compiling', this.phase === 'run' ? 'Running…' : 'Compiling with tilc…');
                this.schedule(0);
            }
        }

        cancelTimers() {
            if (this.raf) cancelAnimationFrame(this.raf);
            if (this.timer) clearTimeout(this.timer);
            this.raf = 0;
            this.timer = 0;
        }

        stop() {
            if (this.done || this.stopped) return;
            this.stopped = true;
            this.cancelTimers();
            hideStdin();
            if (this.resolveCompile) { const r = this.resolveCompile; this.resolveCompile = null; r(null); }
            flushOutput();
            presentScreen();
            updateStats(this);
        }
    }

    function updateStats(run) {
        if (!run || !run.vm) { ui.stats.textContent = ''; return; }
        const vm = run.vm;
        const parts = [];
        if (run.phase === 'run') {
            const steps = vm.steps;
            parts.push(`${steps.toLocaleString('en-US')} ${steps === 1 ? 'step' : 'steps'}`);
            parts.push(`heap ${fmtBytes(vm.heapUsed)}`);
            if (run.compileMs) parts.push(`compiled in ${fmtMs(run.compileMs)}`);
        }
        ui.stats.textContent = parts.join(' · ');
    }

    function finePointer() {
        return !window.matchMedia || window.matchMedia('(hover: hover) and (pointer: fine)').matches;
    }

    function focusKeyTarget() {
        const target = ui.screenWrap.hidden ? ui.output : ui.screen;
        if (document.activeElement !== target && document.activeElement !== ui.stdinText) {
            target.focus({ preventScroll: true });
        }
    }

    function showStdin() {
        ui.stdin.hidden = false;
        ui.stdinText.focus({ preventScroll: true });
    }

    function hideStdin() {
        ui.stdin.hidden = true;
    }

    async function runProgram() {
        if (!app.runtime) return;
        if (app.run) app.run.stop();
        flushOutput();
        clearOutput();
        app.trimmed = false;
        clearErrorLine();
        closeScreen();
        if (app.runtime.broken) await resetRuntime();
        const run = new Run(ui.code.value);
        app.run = run;
        try {
            await run.start();
        } catch (e) {
            appendOutput(`internal error: ${e && e.message ? e.message : e}\n`, 'o-err');
            run.finish('error', 'Internal error');
        }
    }

    function stopProgram() {
        const run = app.run;
        if (!run || run.done || run.stopped) return;
        run.stop();
        note('stopped');
        flushOutput();
        setState('stopped', 'Stopped');
    }

    /* After a WebAssembly trap the module state is unknown: start a fresh one. */
    async function resetRuntime() {
        app.runtime = await Runtime.create('tilvm.wasm');
        app.compilerVM = app.runtime.createVM();
        app.programVM = app.runtime.createVM();
    }

    function sendKey(k) {
        const run = app.run;
        if (!k || !run || run.done || run.stopped || run.phase !== 'run') return false;
        run.vm.pushKey(k);
        if (document.body.dataset.state === 'waiting-key') run.resume();
        return true;
    }

    function onProgramKey(e) {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        if (sendKey(keyCode(e.key))) e.preventDefault();
    }

    function onKeybar(e) {
        const button = e.target.closest('button[data-key]');
        if (!button) return;
        e.preventDefault();                     /* keep the focus where it is */
        sendKey(parseInt(button.dataset.key, 10));
    }

    function onStdinSubmit(e) {
        e.preventDefault();
        const run = app.run;
        if (!run || run.done || run.stopped) return;
        const text = ui.stdinText.value;
        ui.stdinText.value = '';
        appendOutput(text + '\n', 'o-echo');
        run.vm.pushLine(text);
        hideStdin();
        flushOutput();
        run.resume();
    }

    function onStdinKey(e) {
        if (e.ctrlKey && (e.key === 'd' || e.key === 'D')) {
            e.preventDefault();
            const run = app.run;
            if (!run || run.done || run.stopped) return;
            run.vm.closeInput();
            note('end of input');
            hideStdin();
            run.resume();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            stopProgram();
        }
    }

    /* ───────────────────────────── start-up ────────────────────────────── */
    async function detectCompiler() {
        const custom = new URLSearchParams(location.search).get('compiler');
        if (custom) {
            const url = new URL(custom, location.href);
            if (url.origin !== location.origin) throw new Error('?compiler= must be a same-origin path');
            app.compiler = { kind: 'custom', bytes: await fetchBytes(url.href), url: url.pathname };
            return;
        }
        let info = null;
        try {
            const resp = await fetch('build-info.json', { cache: 'no-cache' });
            if (resp.ok) info = await resp.json();
        } catch (e) {
            info = null;
        }
        if (!info || info.compiler === 'tilc') {
            try {
                app.compiler = { kind: 'tilc', bytes: await fetchBytes('tilc.tbc'), url: 'tilc.tbc' };
                return;
            } catch (e) {
                /* not built yet */
            }
        }
        app.compiler = { kind: 'precompiled', bytes: null, url: '' };
    }

    function describeSetup() {
        const c = app.compiler;
        document.body.dataset.compiler = c.kind;
        if (c.kind === 'tilc') {
            ui.compilerInfo.textContent = `Compiler: tilc.tbc, the self-hosted TIL compiler, running in your browser (${fmtBytes(c.bytes.length)})`;
        } else if (c.kind === 'custom') {
            ui.compilerInfo.textContent = `Compiler: ${c.url} (custom, ${fmtBytes(c.bytes.length)})`;
        } else {
            ui.compilerInfo.textContent = 'Compiler: not in this build yet - the bundled examples run precompiled';
        }
        const i = app.runtime.info;
        ui.vmInfo.textContent = `VM: TIL VM ${i.version}, WebAssembly, ${i.intBits}-bit ints, ${fmtBytes(i.heapBytes)} heap`;
    }

    async function init() {
        setState('loading', 'Loading the TIL VM…');
        ui.code.addEventListener('keydown', onEditorKey);
        ui.code.addEventListener('input', onEditorInput);
        ui.code.addEventListener('scroll', () => {
            ui.gutter.scrollTop = ui.code.scrollTop;
            positionErrorLine();
        });
        ui.run.addEventListener('click', runProgram);
        ui.stop.addEventListener('click', stopProgram);
        ui.share.addEventListener('click', () => { shareLink().catch((e) => toast(e.message)); });
        ui.theme.addEventListener('click', toggleTheme);
        ui.clear.addEventListener('click', () => { clearOutput(); });
        ui.examples.addEventListener('change', () => { if (ui.examples.value) loadExample(ui.examples.value); });
        ui.screen.addEventListener('keydown', onProgramKey);
        ui.output.addEventListener('keydown', onProgramKey);
        ui.screen.addEventListener('focus', () => ui.screenWrap.classList.add('focused'));
        ui.screen.addEventListener('blur', () => ui.screenWrap.classList.remove('focused'));
        ui.screen.addEventListener('pointerdown', () => ui.screen.focus({ preventScroll: true }));
        $('keybar').addEventListener('pointerdown', onKeybar);
        ui.stdin.addEventListener('submit', onStdinSubmit);
        ui.stdinText.addEventListener('keydown', onStdinKey);
        window.addEventListener('resize', () => { fitScreen(); positionErrorLine(); });
        window.addEventListener('hashchange', () => { loadFromHash(); });

        const runtimeP = Runtime.create('tilvm.wasm');
        const compilerP = detectCompiler();
        runtimeP.catch(() => {});
        compilerP.catch(() => {});
        await loadExampleIndex();
        if (!(await loadFromHash())) {
            let draft = null;
            try { draft = localStorage.getItem(DRAFT_KEY); } catch (e) { draft = null; }
            if (draft) setCode(draft);
            else if (!(await loadExample('hello'))) setCode('print("Hello, World!")\n');
        }
        try {
            app.runtime = await runtimeP;
        } catch (e) {
            appendOutput(`cannot load the TIL VM (tilvm.wasm): ${e.message}\n`, 'o-err');
            flushOutput();
            setState('error', 'Cannot load the TIL VM');
            return;
        }
        app.compilerVM = app.runtime.createVM();
        app.programVM = app.runtime.createVM();
        try {
            await compilerP;
        } catch (e) {
            appendOutput(`cannot load the compiler: ${e.message}\n`, 'o-err');
            app.compiler = { kind: 'precompiled', bytes: null, url: '' };
        }
        describeSetup();
        flushOutput();
        setState('idle', 'Ready');
    }

    window.addEventListener('error', (e) => {
        appendOutput(`internal error: ${e.message}\n`, 'o-err');
        flushOutput();
    });

    /* for the tests and for debugging from the console */
    window.tilPlayground = { app, run: runProgram, stop: stopProgram, setCode, shareHash, sourceFromHash };

    init();
}());
