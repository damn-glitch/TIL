// Devices: will the program fit a calculator / microcontroller? (POST
// /api/tools/fit = the `til fit --json` report): the verdict, memory bars
// (static need vs limit, measured use with "Run on device VM"), warnings
// linked to their lines.
import { h, clear, fmtBytes } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, L } from '../core/i18n.js';
import { basename } from '../core/paths.js';
import { toast } from '../ui/widgets.js';
import { desktopCard } from './debug.js';

function parseHuman(s) {
    if (s === null || s === undefined) return null;
    if (typeof s === 'number') return s;
    const m = /^\s*([\d.]+)\s*(B|KB|MB|GB)?\s*$/i.exec(String(s));
    if (!m) return null;
    const mult = { B: 1, KB: 1024, MB: 1048576, GB: 1073741824 }[(m[2] || 'B').toUpperCase()];
    return Math.round(parseFloat(m[1]) * mult);
}

export class DevicesView {
    constructor(app) {
        this.app = app;
        this.report = null;
        this.busy = false;
        this.select = h('select', { class: 'select', 'aria-label': t('devices.pick') });
        this.select.addEventListener('change', () => { app.settings.set('device', this.select.value); this.report = null; this.render(); });
        this.runBox = h('input', { type: 'checkbox' });
        this.checkBtn = h('button', { class: 'btn primary small', type: 'button' }, icon('target'), L('devices.check'));
        this.checkBtn.addEventListener('click', () => this.check());
        this.info = h('div', { class: 'device-info' });
        this.result = h('div', { class: 'fit-result' });
        this.controls = h('div', { class: 'device-controls' },
            h('div', { class: 'field' }, L('devices.device', null, 'label'), this.select),
            this.info,
            h('div', { class: 'row wrap' }, h('label', { class: 'switch' }, this.runBox, h('span', { class: 'track' }), L('devices.runOnVm')), h('span', { class: 'spacer' }), this.checkBtn));
        this.body = h('div', { class: 'side-body pad devices-body' }, this.controls, this.result);
        this.titleEl = L('activity.devices');
        this.titleEl.className = 'side-title';
        this.el = h('div', { class: 'side-view' }, h('div', { class: 'side-head' }, this.titleEl), this.body);
        app.settings.on('change:device', () => { this.syncSelect(); });
        this.fill();
        this.render();
    }

    onShow() { this.render(); }

    focus() { this.select.focus(); }

    fill() {
        clear(this.select);
        for (const d of this.app.info.devices || []) this.select.appendChild(h('option', { value: d.name }, `${d.name} — ${d.title || ''}`));
        this.syncSelect();
    }

    syncSelect() {
        const devs = this.app.info.devices || [];
        const cur = this.app.settings.get('device');
        this.select.value = devs.some((d) => d.name === cur) ? cur : devs.length ? devs[0].name : '';
        this.renderInfo();
    }

    device() {
        const devs = this.app.info.devices || [];
        return devs.find((d) => d.name === this.select.value) || devs[0] || null;
    }

    renderInfo() {
        clear(this.info);
        const d = this.device();
        const rd = this.report && this.report.device && this.report.device.name === (d && d.name) ? this.report.device : null;
        if (!d) return;
        this.info.append(icon(/ti84|casio|nspire|numworks|z80/.test(d.name) ? 'calculator' : 'devices', { className: 'device-icon' }),
            h('div', {}, h('div', { class: 'device-title' }, d.title || d.name),
                rd ? h('div', { class: 'device-meta' }, [rd.cpu, rd.ram ? `RAM ${fmtBytes(rd.ram)}` : '', rd.storage ? t('devices.storage', { size: fmtBytes(rd.storage) }) : '',
                    `${rd.int_bits}-bit int`].filter(Boolean).join(' · ')) : null));
    }

    targetPath() {
        const doc = this.app.editors.activeDoc();
        return doc && doc.isTil && doc.kind === 'file' ? doc.path : null;
    }

    async check() {
        if (this.app.web) { this.app.showDesktopOnly('fit'); return; }
        const path = this.targetPath();
        const d = this.device();
        if (!path || !d) { toast(t('devices.noFile'), { kind: 'info' }); return; }
        const doc = this.app.docs.get(path);
        if (doc && doc.dirty) { const ok = await this.app.saveDoc(doc); if (!ok) return; }
        this.busy = true;
        this.render();
        try {
            this.report = await this.app.backend.fit({ path, device: d.name, run: this.runBox.checked });
            this.report._path = path;
        } catch (e) {
            this.report = null;
            if (e && e.notAvailable) this.app.showDesktopOnly(e.feature);
            else toast(t('devices.failed'), { kind: 'error', detail: e.message });
        }
        this.busy = false;
        this.renderInfo();
        this.render();
    }

    render() {
        clear(this.result);
        this.checkBtn.disabled = this.busy || this.app.web || !this.targetPath();
        if (this.app.web) { this.result.appendChild(desktopCard(this.app, 'fit')); return; }
        if (this.busy) { this.result.appendChild(h('div', { class: 'loading-row' }, h('span', { class: 'spinner' }), t('devices.checking'))); return; }
        const r = this.report;
        if (!r) {
            this.result.appendChild(h('div', { class: 'empty-state' }, icon('devices', { className: 'empty-icon' }), L('devices.intro', null, 'p')));
            return;
        }
        const fits = Boolean(r.fits);
        const verdict = String(r.verdict || (fits ? 'fits' : 'does not fit'));
        const tight = fits && (r.resources || []).some((x) => x.need_value && x.limit_value && x.need_value / x.limit_value > 0.8);
        this.result.appendChild(h('div', { class: `verdict ${fits ? (tight ? 'tight' : 'ok') : 'bad'}` },
            icon(fits ? (tight ? 'warning' : 'checkCircle') : 'crossCircle'),
            h('div', {}, h('div', { class: 'verdict-title' }, fits ? (tight ? t('devices.tight') : t('devices.fits')) : t('devices.notFits')),
                h('div', { class: 'verdict-sub' }, `${basename(r._path || r.program || '')} → ${r.device && r.device.name}` + (fits ? '' : ` · ${verdict.replace(/^does not fit:\s*/, '')}`)))));
        const bars = h('div', { class: 'res-list' });
        for (const row of r.resources || []) bars.appendChild(this.resourceRow(row));
        this.result.append(h('div', { class: 'res-title' }, t('devices.resources')), bars);
        const lists = [['errors', 'error', r.errors], ['warnings', 'warning', r.warnings], ['notes', 'info', r.notes]];
        for (const [key, sev, items] of lists) {
            if (!items || !items.length) continue;
            const ul = h('div', { class: `fit-msgs ${sev}` });
            for (const text of items) {
                const m = /\bline (\d+)/.exec(text);
                const row = h('div', { class: 'fit-msg' }, icon(sev, { className: `sev-${sev}` }), h('span', {}, text));
                if (m) {
                    row.classList.add('link');
                    row.tabIndex = 0;
                    const go = () => this.app.openFile(r._path, { line: Number(m[1]) });
                    row.addEventListener('click', go);
                    row.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
                }
                ul.appendChild(row);
            }
            this.result.append(h('div', { class: 'res-title' }, t(`devices.${key}`)), ul);
        }
        if (r.run) this.result.appendChild(this.runBlock(r.run));
    }

    resourceRow(row) {
        const need = row.need_value, limit = row.limit_value;
        const measured = parseHuman(row.measured);
        const status = String(row.status || '').toLowerCase();
        const cls = status === 'fail' ? 'bad' : status === 'warn' ? 'warn' : status === 'ok' ? 'ok' : 'muted';
        const el = h('div', { class: `res-row ${cls}` });
        el.appendChild(h('div', { class: 'res-head' },
            h('span', { class: 'res-name' }, row.resource || row.key),
            h('span', { class: 'res-nums' }, row.measured ? t('devices.measured', { value: row.measured }) + ' · ' : '', `${row.needs} / ${row.limit}`)));
        if (limit && (need || measured)) {
            const pct = need ? Math.min(100, (need / limit) * 100) : 0;
            const mpct = measured ? Math.min(100, (measured / limit) * 100) : null;
            const bar = h('div', { class: 'bar', role: 'meter', 'aria-valuemin': '0', 'aria-valuemax': String(limit), 'aria-valuenow': String(need || measured || 0), 'aria-label': row.resource },
                h('div', { class: 'bar-fill', style: { width: `${Math.max(pct, need ? 1.5 : 0)}%` } }),
                mpct !== null ? h('div', { class: 'bar-measured', style: { left: `${mpct}%` }, title: t('devices.measured', { value: row.measured }) }) : null);
            if (pct > 80 && cls === 'ok') bar.classList.add('high');
            el.appendChild(bar);
        } else {
            el.appendChild(h('div', { class: 'res-status' }, h('span', { class: `chip ${cls === 'ok' ? 'green' : cls === 'bad' ? 'red' : cls === 'warn' ? 'yellow' : ''}` }, status === '-' || status === '?' ? t('devices.dataDependent') : row.status)));
        }
        if (row.note) el.appendChild(h('div', { class: 'res-note' }, row.note));
        return el;
    }

    runBlock(run) {
        const box = h('div', { class: 'fit-run' });
        const ok = !run.error && (run.exit_code === 0 || run.exit_code === undefined);
        box.appendChild(h('div', { class: 'res-title' }, t('devices.runTitle')));
        box.appendChild(h('div', { class: `run-line ${ok ? 'ok' : 'bad'}` }, icon(ok ? 'checkCircle' : 'error'),
            h('span', {}, run.error ? String(run.error) : t('devices.exit', { code: run.exit_code })),
            run.desktop ? h('span', { class: `chip ${run.desktop.same_output ? 'green' : 'yellow'}` }, run.desktop.same_output ? t('devices.sameOutput') : t('devices.differentOutput')) : null));
        if (run.stats) {
            const s = run.stats;
            const grid = h('div', { class: 'stats-grid' });
            const add = (k, v) => grid.append(h('span', { class: 'muted' }, t(`devices.stat.${k}`)), h('span', { class: 'mono' }, v));
            if (s.steps !== undefined) add('steps', Number(s.steps).toLocaleString());
            if (s.peak_heap !== undefined) add('heap', fmtBytes(s.peak_heap));
            if (s.peak_stack !== undefined) add('stack', String(s.peak_stack));
            if (s.peak_frames !== undefined) add('frames', String(s.peak_frames));
            if (s.gc !== undefined) add('gc', String(s.gc));
            box.appendChild(grid);
        }
        if (run.stdout) {
            const det = h('details', { class: 'fit-output' }, h('summary', {}, t('devices.output')), h('pre', {}, String(run.stdout).slice(0, 20000)));
            box.appendChild(det);
        }
        return box;
    }

    relocalize() { this.fill(); this.render(); }
}
