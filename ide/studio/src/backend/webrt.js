// The TIL runtime of the web face: ../playground/tilvm.js + tilvm.wasm and
// the self-hosted compiler tilc.tbc. VM instance A compiles (tilc), VM
// instance B runs the program, in time slices so the page stays responsive;
// wait_key(), input() and sleep_ms() pause without blocking (see
// website/playground/README.md).

const SLICE_MS = 8;
const MAX_TIMER_MS = 1 << 30;

const taskQueue = [];
let channel = null;
function postTask(fn) {
    if (!channel) {
        channel = new MessageChannel();
        channel.port1.onmessage = () => { const f = taskQueue.shift(); if (f) f(); };
    }
    taskQueue.push(fn);
    channel.port2.postMessage(0);
}

function loadScript(src) {
    return new Promise((resolve, reject) => {
        if (window.TILWasm) { resolve(window.TILWasm); return; }
        const s = document.createElement('script');
        s.src = src;
        s.async = true;
        s.onload = () => (window.TILWasm ? resolve(window.TILWasm) : reject(new Error(`${src} did not define TILWasm`)));
        s.onerror = () => reject(new Error(`cannot load ${src}`));
        document.head.appendChild(s);
    });
}

export class WebRuntime {
    constructor(base) {
        this.base = base;
        this.rt = null;
        this.api = null;
        this.compilerBytes = null;
        this.queue = Promise.resolve();
        this.ready = null;
    }

    init() {
        if (this.ready) return this.ready;
        this.ready = (async () => {
            this.api = await loadScript(this.base + 'tilvm.js');
            const [rt, tbc] = await Promise.all([
                this.api.Runtime.create(this.base + 'tilvm.wasm'),
                fetch(this.base + 'tilc.tbc', { cache: 'force-cache' }).then((r) => {
                    if (!r.ok) throw new Error(`tilc.tbc: HTTP ${r.status}`);
                    return r.arrayBuffer();
                }),
            ]);
            this.rt = rt;
            this.compilerBytes = new Uint8Array(tbc);
            this.fs = this.api.memoryFS();
            rt.fs = this.fs;
            this.vmA = rt.createVM();
            this.vmB = rt.createVM();
            return this;
        })();
        return this.ready;
    }

    async reset() {
        // after a WebAssembly trap the module state is unknown: a fresh one
        const rt = await this.api.Runtime.create(this.base + 'tilvm.wasm');
        this.rt = rt;
        rt.fs = this.fs;
        this.vmA = rt.createVM();
        this.vmB = rt.createVM();
    }

    get info() { return this.rt ? this.rt.info : null; }

    writeFile(path, text) { this.fs.write(path, new TextEncoder().encode(text)); }

    /* compile /path with tilc (serialized: one compile at a time) -> {ok, tbc, output} */
    compile(path) {
        const job = this.queue.then(() => this._compile(path));
        this.queue = job.catch(() => {});
        return job;
    }

    async _compile(path) {
        await this.init();
        if (this.rt.broken) await this.reset();
        const out = `/.studio/${path.replace(/[^\w.-]+/g, '_')}.tbc`;
        this.fs.files.delete(out);
        const vm = this.vmA;
        const log = [];
        vm.onOutput = (text) => log.push(text);
        vm.onScreen = null;
        vm.load(this.compilerBytes);
        vm.setArgs([path, out]);
        const st = await drive(this.api, vm, { onOutput: null });
        const output = log.join('');
        const ok = (st === this.api.STATUS.DONE || (st === this.api.STATUS.EXIT && vm.exitCode === 0));
        const tbc = this.fs.read(out);
        if (ok && tbc && tbc.length) return { ok: true, tbc, output };
        return { ok: false, output, internal: st === this.api.STATUS.ERROR || st === this.api.STATUS.HOST_ERROR };
    }

    /**
     * Run compiled bytecode on VM B. handlers: onOutput(text, stream), onScreen(w, h, rgba), onInput(), onExit(code, text)
     * opts: {stdin, keys, args}. Returns {stop(), pushLine(text), pushKey(code), eof()}.
     */
    run(tbc, handlers, opts = {}) {
        const api = this.api;
        const vm = this.vmB;
        const rt = this.rt;
        let stopped = false, done = false, timer = 0, sleepUntil = 0, raf = 0, frame = -1;
        vm.onOutput = (text, stream) => handlers.onOutput(text, stream);
        vm.onScreen = () => {};
        try {
            vm.load(tbc);
            vm.setArgs(opts.args || []);
        } catch (e) {
            handlers.onExit(1, `cannot load the program: ${e.message}`);
            return { stop() {}, pushLine() {}, pushKey() {}, eof() {} };
        }
        if (opts.stdin) vm.pushInput(new TextEncoder().encode(opts.stdin));
        if (opts.keys) {
            try { vm.setKeyScript(api.parseKeys(opts.keys)); } catch (e) { handlers.onOutput(`${e.message}\n`, 2); }
        }
        const present = () => {
            raf = 0;
            const s = rt.screen();
            if (!s || s.owner !== vm.id || s.frame === frame) return;
            frame = s.frame;
            handlers.onScreen(s.width, s.height, s.pixels.slice());
        };
        const requestPresent = () => { if (!raf) raf = requestAnimationFrame(present); };
        const finish = (code, text) => {
            if (done) return;
            done = true;
            clearTimeout(timer);
            try { vm.showFinalScreen(); } catch (e) { /* */ }
            frame = -1;
            present();
            handlers.onExit(code, text);
        };
        const schedule = () => {
            if (stopped || done) return;
            if (document.hidden) timer = setTimeout(step, 0); else postTask(step);
        };
        const sleepWait = () => {
            if (stopped || done) return;
            const left = sleepUntil - performance.now();
            if (left <= 0) { sleepUntil = 0; schedule(); }
            else if (left < 1) postTask(sleepWait);
            else timer = setTimeout(() => postTask(sleepWait), Math.min(left, MAX_TIMER_MS));
        };
        let waiting = null;
        const step = () => {
            if (stopped || done || sleepUntil) return;
            const st = vm.run(0, SLICE_MS);
            requestPresent();
            switch (st) {
            case api.STATUS.YIELD: schedule(); return;
            case api.STATUS.SLEEP: {
                const ms = vm.sleepMs;
                if (ms <= 0) { schedule(); return; }
                sleepUntil = performance.now() + ms;
                sleepWait();
                return;
            }
            case api.STATUS.WAIT_KEY:
                if (vm.keys.length) { schedule(); return; }
                waiting = 'key';
                return;
            case api.STATUS.WAIT_LINE:
                waiting = 'line';
                handlers.onInput();
                return;
            case api.STATUS.DONE: finish(0); return;
            case api.STATUS.EXIT: finish(vm.exitCode); return;
            default: finish(1, rt.broken || ''); return;
            }
        };
        const resume = () => { if (waiting && !stopped && !done) { waiting = null; schedule(); } };
        schedule();
        return {
            stop() { if (done) return; stopped = true; clearTimeout(timer); finish(-1, ''); },
            pushLine(text) { vm.pushInput(new TextEncoder().encode(text)); if (waiting === 'line') resume(); },
            pushKey(code) { vm.pushKey(code); if (waiting === 'key') resume(); },
            eof() { vm.closeInput(); resume(); },
            get done() { return done; },
        };
    }
}

/* run a VM to its end in time slices (the compiler) */
function drive(api, vm) {
    return new Promise((resolve) => {
        const step = () => {
            const st = vm.run(0, SLICE_MS);
            if (st === api.STATUS.YIELD || st === api.STATUS.SLEEP) { postTask(step); return; }
            if (st === api.STATUS.WAIT_KEY || st === api.STATUS.WAIT_LINE) { resolve(api.STATUS.ERROR); return; }
            resolve(st);
        };
        postTask(step);
    });
}
