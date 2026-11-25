// Packages: the dependencies of til.json (POST /api/pkg with the argv of
// `til pkg`): list, add, remove, install, verify, init; the output of the
// last command.
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, L } from '../core/i18n.js';
import { tip, confirmDialog, toast } from '../ui/widgets.js';
import { desktopCard } from './debug.js';

export class PackagesView {
    constructor(app) {
        this.app = app;
        this.manifest = undefined;     // undefined: not loaded; null: no til.json
        this.last = null;              // {argv, rc, stdout, stderr}
        this.busy = false;
        this.nameInput = h('input', { class: 'input mono', type: 'text', placeholder: t('packages.namePlaceholder'), spellcheck: 'false', 'aria-label': t('packages.name') });
        this.specInput = h('input', { class: 'input mono', type: 'text', placeholder: t('packages.specPlaceholder'), spellcheck: 'false', 'aria-label': t('packages.spec') });
        this.addBtn = h('button', { class: 'btn primary small', type: 'button' }, icon('plus'), L('packages.add'));
        this.addBtn.addEventListener('click', () => this.add());
        for (const i of [this.nameInput, this.specInput]) i.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); this.add(); } });
        const refresh = h('button', { class: 'icon-btn small', type: 'button' }, icon('refresh'));
        tip(refresh, 'packages.refresh');
        refresh.addEventListener('click', () => this.refresh());
        this.body = h('div', { class: 'side-body pad packages-body' });
        this.titleEl = L('activity.packages');
        this.titleEl.className = 'side-title';
        this.el = h('div', { class: 'side-view' }, h('div', { class: 'side-head' }, this.titleEl, h('div', { class: 'side-actions' }, refresh)), this.body);
        this.render();
    }

    onShow() { if (this.manifest === undefined) this.refresh(); else this.render(); }

    focus() { this.nameInput.focus(); }

    async exec(argv, { quiet = false } = {}) {
        if (this.app.web) { this.app.showDesktopOnly('packages'); return null; }
        this.busy = argv[0];
        this.render();
        let r;
        try {
            r = await this.app.backend.pkg(argv);
        } catch (e) {
            this.busy = false;
            if (e && e.notAvailable) { this.app.showDesktopOnly(e.feature); return null; }
            toast(t('packages.failed'), { kind: 'error', detail: e.message });
            this.render();
            return null;
        }
        this.busy = false;
        if (r && 'manifest' in r) this.manifest = r.manifest || null;
        // a quiet refresh of a project without til.json is not a failure to show
        if (!quiet || (r && r.rc && r.manifest !== null)) this.last = { argv, rc: r.rc, stdout: r.stdout || '', stderr: r.stderr || '' };
        this.render();
        if (r && r.rc && !quiet) toast(t('packages.commandFailed', { cmd: `til pkg ${argv.join(' ')}` }), { kind: 'error', detail: (r.stderr || r.stdout || '').trim().slice(0, 400) });
        return r;
    }

    async refresh() { await this.exec(['list'], { quiet: true }); }

    async add() {
        const name = this.nameInput.value.trim();
        const spec = this.specInput.value.trim();
        if (!name) { this.nameInput.focus(); return; }
        const argv = spec ? ['add', name, spec] : ['add', name];
        const r = await this.exec(argv);
        if (r && !r.rc) { this.nameInput.value = ''; this.specInput.value = ''; this.app.views.explorer.refresh(); this.app.statusbar.flash(t('packages.added', { name })); }
    }

    async remove(name) {
        const ok = await confirmDialog({ title: t('packages.removeTitle'), message: t('packages.removeMsg', { name }), confirm: t('packages.remove'), danger: true });
        if (!ok) return;
        const r = await this.exec(['remove', name]);
        if (r && !r.rc) this.app.views.explorer.refresh();
    }

    render() {
        clear(this.body);
        if (this.app.web) { this.body.appendChild(desktopCard(this.app, 'packages')); return; }
        if (this.manifest === undefined && !this.busy) {
            this.body.appendChild(h('div', { class: 'loading-row' }, h('span', { class: 'spinner' }), t('packages.loading')));
            return;
        }
        if (this.manifest === null) {
            const init = h('button', { class: 'btn primary', type: 'button' }, icon('packages'), L('packages.init'));
            init.addEventListener('click', async () => { await this.exec(['init']); this.app.views.explorer.refresh(); });
            this.body.appendChild(h('div', { class: 'empty-state' }, icon('packages', { className: 'empty-icon' }), L('packages.noManifest', null, 'p'), init));
            this.renderLast();
            return;
        }
        const m = this.manifest || {};
        this.body.appendChild(h('div', { class: 'pkg-project' }, icon('packages', { className: 'pkg-icon' }),
            h('div', {}, h('div', { class: 'pkg-name' }, m.name || this.app.info.rootName), h('div', { class: 'pkg-version muted' }, m.version ? `v${m.version}` : '')),
            h('span', { class: 'spacer' }),
            this.cmdBtn('install', 'download', 'packages.install'),
            this.cmdBtn('verify', 'checkCircle', 'packages.verify')));
        const deps = Object.entries(m.dependencies || {});
        this.body.appendChild(h('div', { class: 'res-title' }, t('packages.dependencies'), h('span', { class: 'badge' }, String(deps.length))));
        const list = h('div', { class: 'pkg-list' });
        if (!deps.length) list.appendChild(h('div', { class: 'muted small pad' }, t('packages.none')));
        for (const [name, spec] of deps) {
            const rm = h('button', { class: 'icon-btn small', type: 'button' }, icon('trash'));
            tip(rm, 'packages.remove');
            rm.addEventListener('click', () => this.remove(name));
            list.appendChild(h('div', { class: 'pkg-row' }, icon('packages', { className: 'dep-icon' }),
                h('div', { class: 'pkg-dep' }, h('span', { class: 'pkg-dep-name' }, name), h('code', { class: 'pkg-dep-spec' }, typeof spec === 'string' ? spec : JSON.stringify(spec))), rm));
        }
        this.body.appendChild(list);
        this.body.appendChild(h('div', { class: 'pkg-add' }, h('div', { class: 'res-title' }, t('packages.addTitle')),
            h('div', { class: 'col' }, this.nameInput, this.specInput, h('div', { class: 'row' }, L('packages.specHint', null, 'span'), h('span', { class: 'spacer' }), this.addBtn))));
        this.addBtn.disabled = Boolean(this.busy);
        this.renderLast();
    }

    cmdBtn(cmd, iconName, key) {
        const b = h('button', { class: 'btn small', type: 'button', disabled: Boolean(this.busy) }, this.busy === cmd ? h('span', { class: 'spinner' }) : icon(iconName), L(key));
        b.addEventListener('click', () => this.exec([cmd]));
        return b;
    }

    renderLast() {
        if (this.busy && this.busy !== 'list') this.body.appendChild(h('div', { class: 'loading-row' }, h('span', { class: 'spinner' }), `til pkg ${this.busy}…`));
        const l = this.last;
        if (!l) return;
        const text = (l.stdout + (l.stderr ? (l.stdout ? '\n' : '') + l.stderr : '')).trim();
        this.body.appendChild(h('div', { class: `pkg-log ${l.rc ? 'bad' : 'ok'}` },
            h('div', { class: 'pkg-log-head' }, icon(l.rc ? 'error' : 'checkCircle'), h('code', {}, `til pkg ${l.argv.join(' ')}`), h('span', { class: 'muted' }, l.rc ? t('packages.exitCode', { code: l.rc }) : '')),
            text ? h('pre', {}, text.slice(0, 8000)) : null));
    }

    relocalize() {
        this.nameInput.placeholder = t('packages.namePlaceholder');
        this.specInput.placeholder = t('packages.specPlaceholder');
        this.render();
    }
}
