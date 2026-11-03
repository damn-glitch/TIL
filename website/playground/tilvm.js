/*
 * tilvm.js - the TIL VM (tilvm.wasm) for JavaScript, in the browser and in Node.
 *
 *   const rt = await TILWasm.Runtime.create('tilvm.wasm');   // or bytes in Node
 *   const vm = rt.createVM();
 *   vm.onOutput = (text, stream) => ...;      // stream 1 = output, 2 = errors
 *   vm.load(bytecode);                        // Uint8Array of a .tbc file
 *   vm.setArgs(['a', 'b']);                   // args()
 *   let st = vm.run(20000);                   // at most 20000 steps, never blocks
 *   // st: DONE, ERROR, EXIT (vm.exitCode), YIELD (call run again),
 *   //     WAIT_KEY (vm.pushKey(code), then run), WAIT_LINE (vm.pushLine(text)),
 *   //     SLEEP (run again after vm.sleepMs ms), HOST_ERROR
 *
 * The virtual file system (rt.fs: read(path) -> Uint8Array|null,
 * write(path, bytes) -> bool) is shared by all VMs of a runtime; the default is
 * an in-memory Map. See wasm_host.c for how pausing works.
 */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.TILWasm = factory();
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    const STATUS = Object.freeze({
        DONE: 0, ERROR: 1, YIELD: 2, EXIT: 3, WAIT_KEY: 4, WAIT_LINE: 5, SLEEP: 6, HOST_ERROR: 7,
    });
    const STATUS_NAMES = ['done', 'error', 'yield', 'exit', 'wait-key', 'wait-line', 'sleep', 'host-error'];

    const LOAD_ERRORS = {
        '-1': 'not a TIL bytecode file',
        '-2': 'unsupported bytecode version (this VM loads versions 1-2)',
        '-3': 'the program uses floats, but this VM was built without them',
        '-4': 'too many constants for this VM',
        '-5': 'the bytecode is truncated or corrupt',
        '-6': 'an integer constant is too large for this VM',
        '-7': 'too many struct types for this VM',
        '-8': 'too many globals for this VM',
        '-100': 'the bytecode is too large for the playground',
        '-101': 'no such VM instance',
    };

    /* Browser KeyboardEvent.key -> TIL key code (UP=1 DOWN=2 LEFT=3 RIGHT=4, ASCII otherwise) */
    const NAMED_KEYS = {
        ArrowUp: 1, ArrowDown: 2, ArrowLeft: 3, ArrowRight: 4,
        Enter: 13, Escape: 27, Backspace: 8, Tab: 9, Delete: 127, ' ': 32, Spacebar: 32,
    };

    function keyCode(key) {
        if (Object.prototype.hasOwnProperty.call(NAMED_KEYS, key)) return NAMED_KEYS[key];
        if (typeof key === 'string' && key.length >= 1) {
            const cp = key.codePointAt(0);
            if (String.fromCodePoint(cp) === key && cp >= 32) return cp;
        }
        return 0;
    }

    /* "UP,UP,ENTER,a" (the --keys syntax of the desktop VMs) -> key codes */
    const KEY_WORDS = { UP: 1, DOWN: 2, LEFT: 3, RIGHT: 4, ENTER: 13, ESC: 27, ESCAPE: 27, SPACE: 32,
        TAB: 9, BACKSPACE: 8 };

    function parseKeys(text) {
        const out = [];
        for (const word of String(text).split(',')) {
            if (word === '') continue;
            const up = word.toUpperCase();
            if (Object.prototype.hasOwnProperty.call(KEY_WORDS, up) && word.length > 1) out.push(KEY_WORDS[up]);
            else if (/^\d+$/.test(word) && word.length > 1) out.push(parseInt(word, 10));
            else out.push(word.codePointAt(0));
        }
        return out;
    }

    const SUSPEND = { tilvmSuspend: true };      /* thrown by the `suspend` import */
    const encoder = new TextEncoder();
    const now = (typeof performance !== 'undefined' && performance.now)
        ? () => performance.now() : () => Date.now();

    function memoryFS() {
        const files = new Map();
        return {
            files,
            read(path) { return files.has(path) ? files.get(path) : null; },
            write(path, bytes) { files.set(path, bytes); return true; },
        };
    }

    class Runtime {
        constructor() {
            this.exports = null;
            this.memory = null;
            this.vms = [];
            this.fs = memoryFS();
            this.broken = null;              /* message after a WebAssembly trap */
            this._u8 = null;
        }

        /* source: URL of tilvm.wasm (browser) or its bytes (ArrayBuffer / Uint8Array) */
        static async create(source) {
            const rt = new Runtime();
            const imports = rt._imports();
            let result;
            if (typeof source === 'string') {
                if (typeof WebAssembly.instantiateStreaming === 'function') {
                    try {
                        result = await WebAssembly.instantiateStreaming(fetch(source), imports);
                    } catch (e) {
                        result = null;       /* e.g. served without application/wasm */
                    }
                }
                if (!result) {
                    const resp = await fetch(source);
                    if (!resp.ok) throw new Error(`cannot load ${source}: HTTP ${resp.status}`);
                    result = await WebAssembly.instantiate(await resp.arrayBuffer(), imports);
                }
            } else {
                result = await WebAssembly.instantiate(source, imports);
            }
            rt.exports = result.instance.exports;
            rt.memory = rt.exports.memory;
            return rt;
        }

        get info() {
            const e = this.exports;
            return {
                version: this.cstr(e.wh_version()),
                vms: e.wh_info(0), heapBytes: e.wh_info(1), stackSlots: e.wh_info(2),
                maxFrames: e.wh_info(3), codeBytes: e.wh_info(4),
                screenMaxWidth: e.wh_info(5), screenMaxHeight: e.wh_info(6),
                intBits: e.wh_info(7), floats: e.wh_info(8),
                bytecodeVersion: e.wh_info(9), bytecodeMinVersion: e.wh_info(10),
            };
        }

        u8() {
            if (!this._u8 || this._u8.buffer !== this.memory.buffer) this._u8 = new Uint8Array(this.memory.buffer);
            return this._u8;
        }

        cstr(ptr, len) {
            const u8 = this.u8();
            let end = ptr;
            if (len === undefined) { while (u8[end]) end++; } else end = ptr + len;
            return new TextDecoder().decode(u8.subarray(ptr, end));
        }

        createVM() {
            const id = this.exports.wh_create();
            if (id < 0) throw new Error('no free VM instance');
            const vm = new VM(this, id);
            this.vms[id] = vm;
            return vm;
        }

        /* the screen (shared by all VMs): {width, height, frame, owner, pixels: RGBA bytes} or null */
        screen() {
            const e = this.exports;
            const w = e.wh_screen_w(), h = e.wh_screen_h();
            if (!w || !h) return null;
            return {
                width: w, height: h, frame: e.wh_screen_frame(), owner: e.wh_screen_owner(),
                pixels: new Uint8ClampedArray(this.memory.buffer, e.wh_screen_ptr(), w * h * 4),
            };
        }

        _imports() {
            const rt = this;
            const vmOf = (id) => rt.vms[id];
            return {
                Math,
                env: {
                    fmod: (x, y) => x % y,
                    now,
                    write(id, stream, ptr, len) {
                        const vm = vmOf(id);
                        if (vm) vm._emit(stream, rt.u8().subarray(ptr, ptr + len));
                    },
                    read_line(id, buf, cap) {
                        const vm = vmOf(id);
                        return vm ? vm._readLine(buf, cap) : -1;
                    },
                    key(id) {
                        const vm = vmOf(id);
                        if (!vm) return 0;
                        vm.keyPolls++;
                        return vm.keys.length ? vm.keys.shift() : 0;
                    },
                    file_read(id, pathPtr, pathLen, buf, cap) {
                        const data = rt.fs.read(rt.cstr(pathPtr, pathLen));
                        if (!data) return -1;
                        if (buf && cap) rt.u8().set(data.subarray(0, Math.min(cap, data.length)), buf);
                        return data.length;
                    },
                    file_write(id, pathPtr, pathLen, dataPtr, len) {
                        const ok = rt.fs.write(rt.cstr(pathPtr, pathLen), rt.u8().slice(dataPtr, dataPtr + len));
                        return ok ? 0 : -1;
                    },
                    screen(id, event, a, b) {
                        const vm = vmOf(id);
                        if (vm && vm.onScreen) vm.onScreen(event === 0 ? 'open' : 'show', a, b);
                    },
                    suspend() { throw SUSPEND; },
                },
            };
        }
    }

    class VM {
        constructor(rt, id) {
            this.rt = rt;
            this.id = id;
            this.onOutput = null;            /* (text, stream) */
            this.onScreen = null;            /* ('open', w, h) | ('show', frame) */
            this.status = STATUS.YIELD;
            this.loaded = false;
            this._reset();
        }

        _reset() {
            this.keys = [];
            this.keyPolls = 0;
            this.input = new Uint8Array(0);  /* unread input bytes (lines end with \n) */
            this.inputClosed = false;
            this.decoders = { 1: new TextDecoder(), 2: new TextDecoder() };
            this.errorText = '';
        }

        load(bytes) {
            const e = this.rt.exports;
            if (this.rt.broken) throw new Error(this.rt.broken);
            const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
            if (data.length > e.wh_info(4)) throw new Error(LOAD_ERRORS['-100']);
            this.rt.u8().set(data, e.wh_code_ptr(this.id));
            this._reset();
            const rc = e.wh_load(this.id, data.length);
            this.loaded = rc === 0;
            if (rc !== 0) throw new Error(LOAD_ERRORS[String(rc)] || `cannot load the program (error ${rc})`);
            this.status = STATUS.YIELD;
        }

        setArgs(list) {
            const e = this.rt.exports;
            const parts = list.map((s) => encoder.encode(String(s)));
            const total = parts.reduce((n, p) => n + p.length + 1, 0);
            if (total > e.wh_info(12) || parts.length > e.wh_info(13)) throw new Error('too many program arguments');
            if (parts.some((p) => p.includes(0))) throw new Error('program arguments cannot contain NUL');
            const u8 = this.rt.u8();
            let pos = e.wh_args_ptr(this.id);
            for (const p of parts) { u8.set(p, pos); pos += p.length; u8[pos++] = 0; }
            if (e.wh_set_args(this.id, total) < 0) throw new Error('too many program arguments');
        }

        /* Run at most `budget` steps; returns a STATUS code. Never blocks. */
        run(budget) {
            const rt = this.rt, e = rt.exports;
            if (rt.broken) return (this.status = STATUS.HOST_ERROR);
            const sp = e.__stack_pointer.value;
            let st;
            try {
                st = e.wh_run(this.id, budget >>> 0);
            } catch (err) {
                e.__stack_pointer.value = sp;           /* the C stack was unwound by the exception */
                if (err !== SUSPEND) {
                    rt.broken = `internal VM error: ${err && err.message ? err.message : err}`;
                    this._emitText(2, rt.broken + '\n');
                    return (this.status = STATUS.HOST_ERROR);
                }
                st = e.wh_status(this.id);
            }
            this._flush();
            return (this.status = st);
        }

        get statusName() { return STATUS_NAMES[this.status] || String(this.status); }
        get exitCode() { return this.rt.exports.wh_exit_code(this.id); }
        get sleepMs() { return this.rt.exports.wh_sleep_ms(this.id); }
        get steps() { return this.rt.exports.wh_steps(this.id) >>> 0; }
        get heapUsed() { return this.rt.exports.wh_heap_used(this.id) >>> 0; }
        get gcCount() { return this.rt.exports.wh_gc_count(this.id) >>> 0; }
        get finished() {
            return this.status === STATUS.DONE || this.status === STATUS.ERROR ||
                this.status === STATUS.EXIT || this.status === STATUS.HOST_ERROR;
        }

        /* queue a key for key_pressed()/wait_key() (the oldest are dropped beyond 256) */
        pushKey(code) {
            if (!(code > 0)) return;
            if (this.keys.length >= 256) this.keys.shift();
            this.keys.push(code);
        }

        /* queue one line of input (without the newline) */
        pushLine(text) { this.pushInput(encoder.encode(String(text) + '\n')); }

        /* queue raw input bytes (lines end with \n) */
        pushInput(bytes) {
            const merged = new Uint8Array(this.input.length + bytes.length);
            merged.set(this.input, 0);
            merged.set(bytes, this.input.length);
            this.input = merged;
        }

        /* end of input: input() then returns "" (like a closed stdin) */
        closeInput() { this.inputClosed = true; }

        destroy() {
            this.rt.exports.wh_destroy(this.id);
            this.rt.vms[this.id] = undefined;
        }

        /* fgets(buf, cap) on the queued input, newline removed (vm/tilvm.c default_read_line) */
        _readLine(buf, cap) {
            const data = this.input;
            if (!data.length) return this.inputClosed ? -1 : -2;
            const limit = Math.min(data.length, cap - 1);
            let n = 0;
            while (n < limit && data[n] !== 10) n++;
            const newline = n < data.length && data[n] === 10 && n < cap - 1;
            if (!newline && n < cap - 1 && !this.inputClosed) return -2;     /* incomplete line */
            this.rt.u8().set(data.subarray(0, n), buf);
            this.input = data.subarray(n + (newline ? 1 : 0));
            return n;
        }

        _emit(stream, bytes) {
            const text = this.decoders[stream === 2 ? 2 : 1].decode(bytes, { stream: true });
            this._emitText(stream, text);
        }

        _emitText(stream, text) {
            if (!text) return;
            if (stream === 2) this.errorText += text;
            if (this.onOutput) this.onOutput(text, stream);
        }

        _flush() {
            for (const s of [1, 2]) {
                const text = this.decoders[s].decode(new Uint8Array(0), { stream: true });
                this._emitText(s, text);
            }
        }
    }

    return { Runtime, VM, STATUS, STATUS_NAMES, keyCode, parseKeys, memoryFS };
}));
