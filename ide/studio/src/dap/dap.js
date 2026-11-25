// A Debug Adapter Protocol client for the TIL debug adapter, reached through
// the backend (POST /api/dap/start, POST /api/dap/{id}: one DAP message in,
// the messages to deliver now out; events also arrive as SSE `dap` events
// {sessionId, message}). Custom requests of the TIL adapter: til/timeline,
// til/goto, til/stdin; custom events: til/inputRequested, til/screen.
import { Emitter } from '../core/events.js';

const TIMEOUT_MS = 30000;

export class DapClient extends Emitter {
    constructor(backend) {
        super();
        this.backend = backend;
        this.sessionId = null;
        this.seq = 1;
        this.pending = new Map();
        this.capabilities = {};
        this.closed = false;
        this.seenEvents = new Set();
        this.unsubscribe = backend.on('dap', (ev) => {
            if (!ev || ev.sessionId === undefined || String(ev.sessionId) !== String(this.sessionId)) return;
            this.dispatch(ev.message);
        });
    }

    async start() {
        this.sessionId = await this.backend.dapStart();
        if (this.sessionId === undefined || this.sessionId === null) throw new Error('the debugger did not start');
        return this.sessionId;
    }

    request(command, args, timeout = TIMEOUT_MS) {
        if (this.closed) return Promise.reject(new Error('debug session ended'));
        const seq = this.seq++;
        const msg = { seq, type: 'request', command };
        if (args !== undefined) msg.arguments = args;
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                if (this.pending.has(seq)) { this.pending.delete(seq); reject(new Error(`${command}: no answer`)); }
            }, timeout);
            this.pending.set(seq, { resolve, reject, timer, command });
            this.backend.dap(this.sessionId, msg).then((messages) => {
                for (const m of messages) this.dispatch(m);
            }, (e) => {
                if (this.pending.has(seq)) {
                    this.pending.delete(seq);
                    clearTimeout(timer);
                    reject(e);
                }
            });
        });
    }

    dispatch(m) {
        if (!m || typeof m !== 'object') return;
        if (m.type === 'response') {
            const p = this.pending.get(m.request_seq);
            if (!p) return;
            this.pending.delete(m.request_seq);
            clearTimeout(p.timer);
            if (m.success === false) {
                const err = new Error(m.message || (m.body && m.body.error && m.body.error.format) || `${p.command} failed`);
                err.body = m.body;
                p.reject(err);
            } else p.resolve(m.body || {});
            return;
        }
        if (m.type === 'event') {
            // the same event can come in a POST answer and on the SSE stream: once
            if (m.seq !== undefined && m.seq !== null) {
                const key = `${m.event}:${m.seq}`;
                if (this.seenEvents.has(key)) return;
                this.seenEvents.add(key);
                if (this.seenEvents.size > 4000) this.seenEvents = new Set([...this.seenEvents].slice(-1000));
            }
            this.emit('event', m.event, m.body || {});
            this.emit(m.event, m.body || {});
            return;
        }
        if (m.type === 'request') {         // reverse requests (runInTerminal ...): not supported
            this.backend.dap(this.sessionId, { seq: this.seq++, type: 'response', request_seq: m.seq, command: m.command, success: false, message: 'not supported' }).catch(() => {});
        }
    }

    close() {
        if (this.closed) return;
        this.closed = true;
        if (this.unsubscribe) this.unsubscribe();
        for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(new Error('debug session ended')); }
        this.pending.clear();
    }
}
