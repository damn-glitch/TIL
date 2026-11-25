// Run & Debug: the run configuration (backend, arguments, stdin script,
// keys), the debug toolbar, the timeline scrubber, Variables, Watch, Call
// Stack and Breakpoints.
import { h, clear } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { t, L } from '../core/i18n.js';
import { basename } from '../core/paths.js';
import { tip } from '../ui/widgets.js';
import { load, save } from '../core/store.js';

function section(id, titleKey, { collapsed = false, actions = [] } = {}) {
    const body = h('div', { class: 'section-body' });
    const head = h('button', { class: 'section-head', type: 'button', 'aria-expanded': String(!collapsed) },
        icon('chevronDown', { className: 'chev' }), L(titleKey), h('span', { class: 'section-badge' }), h('span', { class: 'section-actions' }, ...actions));
    const el = h('div', { class: 'section', dataset: { section: id, collapsed: String(collapsed) } }, head, body);
    head.addEventListener('click', (e) => {
        if (e.target.closest('.section-actions')) return;
        const c = el.dataset.collapsed !== 'true';
        el.dataset.collapsed = String(c);
        head.setAttribute('aria-expanded', String(!c));
        const st = load('sections', {});
        st[id] = c;
        save('sections', st);
    });
    const st = load('sections', {});
    if (st[id] !== undefined) { el.dataset.collapsed = String(st[id]); head.setAttribute('aria-expanded', String(!st[id])); }
    return { el, body, head, badge: head.querySelector('.section-badge') };
}

function actionBtn(name, key, run) {
    const b = h('button', { class: 'icon-btn small tree-action', type: 'button' }, icon(name));
    tip(b, key);
    b.addEventListener('click', (e) => { e.stopPropagation(); run(); });
    return b;
}

export class DebugView {
    constructor(app) {
        this.app = app;
        this.expandedVars = new Set();
        this.varChildren = new Map();
        const cfg = Object.assign({ args: '', stdin: '', keys: '', stopOnEntry: false }, app.project.runConfig || {});
        this.cfg = cfg;

        /* configuration */
        this.argsInput = h('input', { class: 'input mono', type: 'text', value: cfg.args, placeholder: t('debug.argsPlaceholder'), spellcheck: 'false' });
        this.stdinInput = h('textarea', { class: 'textarea mono', rows: '3', placeholder: t('debug.stdinPlaceholder'), spellcheck: 'false' });
        this.stdinInput.value = cfg.stdin;
        this.keysInput = h('input', { class: 'input mono', type: 'text', value: cfg.keys, placeholder: 'UP,UP,ENTER,a', spellcheck: 'false' });
        this.entryBox = h('input', { type: 'checkbox', checked: cfg.stopOnEntry });
        for (const el of [this.argsInput, this.stdinInput, this.keysInput, this.entryBox]) el.addEventListener('change', () => this.saveConfig());
        this.backendSeg = h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': t('backend.title') });
        this.runBtn = h('button', { class: 'btn primary', type: 'button' }, icon('play'), L('debug.runBtn'));
        this.runBtn.addEventListener('click', () => app.runFile());
        this.debugBtn = h('button', { class: 'btn', type: 'button' }, icon('bug'), L('debug.debugBtn'));
        this.debugBtn.addEventListener('click', () => app.debugFile());
        const field = (labelKey, input, hintKey) => h('div', { class: 'field' }, L(labelKey, null, 'label'), input, hintKey ? L(hintKey, null, 'div') : null);
        const fHint = (el) => { el.querySelectorAll('div[data-i18n]').forEach((d) => d.classList.add('field-hint')); return el; };
        this.configBody = h('div', { class: 'run-config' },
            h('div', { class: 'field' }, L('debug.backend', null, 'label'), this.backendSeg),
            fHint(field('debug.args', this.argsInput)),
            fHint(field('debug.stdin', this.stdinInput, 'debug.stdinHint')),
            fHint(field('debug.keys', this.keysInput, 'debug.keysHint')),
            h('label', { class: 'switch' }, this.entryBox, h('span', { class: 'track' }), L('debug.stopOnEntry')),
            h('div', { class: 'run-buttons' }, this.runBtn, this.debugBtn));
        this.secConfig = section('config', 'debug.configuration');
        this.secConfig.body.appendChild(this.configBody);

        /* session */
        this.timelineEl = h('div', { class: 'timeline' });
        this.secTimeline = section('timeline', 'debug.timeline');
        this.secTimeline.body.appendChild(this.timelineEl);
        this.varsEl = h('div', { class: 'vars', role: 'tree', 'aria-label': t('debug.variables') });
        this.secVars = section('variables', 'debug.variables');
        this.secVars.body.appendChild(this.varsEl);
        this.watchEl = h('div', { class: 'vars watch' });
        this.watchInput = h('input', { class: 'input small-input mono', type: 'text', placeholder: t('debug.addWatch'), spellcheck: 'false' });
        this.watchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); const v = this.watchInput.value.trim(); if (v) { app.debugger.addWatch(v); this.watchInput.value = ''; } }
        });
        this.secWatch = section('watch', 'debug.watch', { actions: [actionBtn('plus', 'debug.addWatchTip', () => this.watchInput.focus())] });
        this.secWatch.body.append(this.watchEl, h('div', { class: 'watch-add' }, this.watchInput));
        this.stackEl = h('div', { class: 'stack', role: 'listbox', 'aria-label': t('debug.callStack') });
        this.secStack = section('stack', 'debug.callStack');
        this.secStack.body.appendChild(this.stackEl);
        this.bpEl = h('div', { class: 'bps' });
        this.secBps = section('breakpoints', 'debug.breakpoints', { actions: [actionBtn('trash', 'debug.removeAllBps', () => this.removeAllBreakpoints())] });
        this.secBps.body.appendChild(this.bpEl);

        this.cardEl = h('div', { class: 'debug-card' });
        this.body = h('div', { class: 'side-body debug-body' }, this.secConfig.el, this.cardEl, this.secTimeline.el, this.secVars.el, this.secWatch.el, this.secStack.el, this.secBps.el);
        this.titleEl = L('activity.debug');
        this.titleEl.className = 'side-title';
        this.el = h('div', { class: 'side-view' }, h('div', { class: 'side-head' }, this.titleEl), this.body);
        this.renderBackends();
        app.settings.on('change:backend', () => this.renderBackends());
        this.render();
    }

    config() {
        return {
            args: this.argsInput ? this.argsInput.value : this.cfg.args,
            stdin: this.stdinInput ? this.stdinInput.value : this.cfg.stdin,
            keys: this.keysInput ? this.keysInput.value.trim() : this.cfg.keys,
            stopOnEntry: this.entryBox ? this.entryBox.checked : this.cfg.stopOnEntry,
        };
    }

    saveConfig() { this.app.saveLayout(); }

    focus() { (this.app.debugger.active ? this.varsEl : this.runBtn).focus(); }

    renderBackends() {
        clear(this.backendSeg);
        const cur = this.app.effectiveBackend();
        for (const b of ['native', 'vm', 'cvm']) {
            const ok = this.app.backendAvailable(b);
            const btn = h('button', { type: 'button', role: 'radio', 'aria-checked': String(cur === b), disabled: !ok, title: ok ? '' : t('backend.notInstalled') }, t(`backend.${b}`));
            btn.addEventListener('click', () => this.app.settings.set('backend', b));
            this.backendSeg.appendChild(btn);
        }
    }

    render() {
        const dbg = this.app.debugger;
        const active = dbg.active;
        const canDebug = this.app.caps.debugger;
        this.debugBtn.hidden = !canDebug;
        clear(this.cardEl);
        if (!canDebug) {
            this.cardEl.appendChild(this.app.web ? desktopCard(this.app, 'debugger') : h('div', { class: 'note-card' }, icon('info'), L('debug.unavailable', null, 'p')));
        }
        for (const s of [this.secTimeline, this.secVars, this.secWatch, this.secStack]) s.el.hidden = !active && s !== this.secWatch;
        if (active !== this.wasActive) {
            this.wasActive = active;
            const sec = this.secConfig;
            sec.el.dataset.collapsed = String(active);
            sec.head.setAttribute('aria-expanded', String(!active));
        }
        this.secWatch.el.hidden = !canDebug;
        this.secBps.el.hidden = !canDebug;
        this.renderTimeline();
        this.renderVariables();
        this.renderWatch();
        this.renderStack();
        this.renderBreakpoints();
    }

    renderTimeline() {
        const dbg = this.app.debugger;
        clear(this.timelineEl);
        const tl = dbg.timeline;
        if (!dbg.active) return;
        if (!tl || !tl.total) {
            this.timelineEl.appendChild(h('div', { class: 'muted small pad' }, dbg.stopped ? t('debug.noTimeline') : t('debug.timelineRunning')));
            return;
        }
        const range = h('input', { class: 'tl-range', type: 'range', min: '0', max: String(Math.max(0, tl.total - 1)), step: '1', value: String(tl.index), 'aria-label': t('debug.timeline'), disabled: !dbg.stopped });
        range.style.setProperty('--pos', `${tl.total > 1 ? (tl.index / (tl.total - 1)) * 100 : 100}%`);
        const label = h('div', { class: 'tl-label' },
            h('span', { class: 'tl-index' }, t('debug.event', { index: tl.index + 1, total: tl.total }) + (tl.complete === false ? '+' : '')),
            dbg.location ? h('span', { class: 'tl-line' }, t('debug.lineOf', { line: dbg.location.line, file: basename(dbg.location.path) })) : null);
        const updateLabel = () => {
            const v = Number(range.value);
            range.style.setProperty('--pos', `${tl.total > 1 ? (v / (tl.total - 1)) * 100 : 100}%`);
            label.querySelector('.tl-index').textContent = t('debug.event', { index: v + 1, total: tl.total });
        };
        range.addEventListener('input', () => { updateLabel(); dbg.goto(Number(range.value)); });
        range.addEventListener('change', () => dbg.goto(Number(range.value), { final: true }));
        const start = h('button', { class: 'icon-btn small', type: 'button', disabled: !dbg.stopped }, icon('chevronLeft'));
        tip(start, 'debug.timelineStart');
        start.addEventListener('click', () => dbg.goto(0, { final: true }));
        const back = h('button', { class: 'icon-btn small', type: 'button', disabled: !dbg.stopped }, icon('stepBack'));
        tip(back, 'debug.stepBack', 'Shift+F10');
        back.addEventListener('click', () => dbg.stepBack());
        const fwd = h('button', { class: 'icon-btn small', type: 'button', disabled: !dbg.stopped }, icon('stepOver'));
        tip(fwd, 'debug.stepOver', 'F10');
        fwd.addEventListener('click', () => dbg.next());
        const end = h('button', { class: 'icon-btn small', type: 'button', disabled: !dbg.stopped }, icon('chevronRight'));
        tip(end, 'debug.timelineEnd');
        end.addEventListener('click', () => dbg.goto(tl.total - 1, { final: true }));
        this.timelineEl.append(
            h('div', { class: 'tl-track' }, range),
            h('div', { class: 'tl-controls' }, start, back, label, fwd, end));
    }

    renderVariables() {
        const dbg = this.app.debugger;
        clear(this.varsEl);
        if (!dbg.active) return;
        if (!dbg.stopped) { this.varsEl.appendChild(h('div', { class: 'muted small pad' }, t('debug.runningHint'))); return; }
        for (const scope of dbg.scopes) {
            const key = `scope:${scope.name}:${scope.variablesReference}`;
            const open = scope.children !== null && !this.expandedVars.has(`closed:${key}`);
            const head = h('div', { class: 'var-row scope', role: 'treeitem', 'aria-expanded': String(Boolean(open)), tabindex: '-1' },
                icon(open ? 'chevronDown' : 'chevronRight', { className: 'chev' }), h('span', { class: 'scope-name' }, scope.name));
            head.addEventListener('click', async () => {
                if (open) { this.expandedVars.add(`closed:${key}`); scope.children = scope.children || []; }
                else {
                    this.expandedVars.delete(`closed:${key}`);
                    if (scope.children === null) { try { scope.children = await dbg.variables(scope.variablesReference); } catch (e) { scope.children = []; } }
                }
                this.renderVariables();
            });
            this.varsEl.appendChild(head);
            if (open && scope.children) {
                if (!scope.children.length) this.varsEl.appendChild(h('div', { class: 'var-row empty', style: { paddingLeft: '30px' } }, t('debug.noVariables')));
                for (const v of scope.children) this.renderVar(v, 1, `${key}/`);
            }
        }
    }

    renderVar(v, depth, prefix, container = this.varsEl) {
        const key = `${prefix}${v.name}`;
        const exp = Boolean(v.variablesReference);
        const open = exp && this.expandedVars.has(key);
        const row = h('div', { class: 'var-row', role: 'treeitem', 'aria-expanded': exp ? String(open) : undefined, tabindex: '-1', style: { paddingLeft: `${14 + depth * 12}px` } },
            exp ? icon(open ? 'chevronDown' : 'chevronRight', { className: 'chev' }) : h('span', { class: 'chev-space' }),
            h('span', { class: 'var-name' }, v.name),
            h('span', { class: 'var-eq' }, '='),
            h('span', { class: `var-value ${valueClass(v)}`, title: v.value }, v.value),
            v.type ? h('span', { class: 'var-type' }, v.type) : null);
        container.appendChild(row);
        if (exp) {
            row.addEventListener('click', async () => {
                if (open) this.expandedVars.delete(key);
                else {
                    this.expandedVars.add(key);
                    if (!this.varChildren.has(v.variablesReference)) {
                        try { this.varChildren.set(v.variablesReference, await this.app.debugger.variables(v.variablesReference)); } catch (e) { this.varChildren.set(v.variablesReference, []); }
                    }
                }
                this.renderVariables();
                this.renderWatch();
            });
            if (open) {
                const kids = this.varChildren.get(v.variablesReference);
                if (kids) for (const c of kids) this.renderVar(c, depth + 1, `${key}/`, container);
                else {
                    this.app.debugger.variables(v.variablesReference).then((list) => { this.varChildren.set(v.variablesReference, list); this.renderVariables(); this.renderWatch(); }, () => {});
                }
            }
        }
    }

    renderWatch() {
        const dbg = this.app.debugger;
        clear(this.watchEl);
        if (!dbg.watches.length) { this.watchEl.appendChild(h('div', { class: 'muted small pad' }, t('debug.noWatch'))); return; }
        for (const w of dbg.watches) {
            const val = dbg.watchValues.get(w);
            const rm = actionBtn('close', 'debug.removeWatch', () => dbg.removeWatch(w));
            let value;
            if (!dbg.stopped || !val) value = h('span', { class: 'var-value muted' }, dbg.active ? '…' : t('debug.notAvailable'));
            else if (val.error) value = h('span', { class: 'var-value err' }, val.error);
            else value = h('span', { class: 'var-value' }, val.value);
            const row = h('div', { class: 'var-row watch-row', style: { paddingLeft: '14px' } }, h('span', { class: 'chev-space' }), h('span', { class: 'var-name' }, w), h('span', { class: 'var-eq' }, '='), value, rm);
            this.watchEl.appendChild(row);
            if (val && val.ref && this.expandedVars.has(`watch:${w}`)) {
                const kids = this.varChildren.get(val.ref);
                if (kids) for (const c of kids) this.renderVar(c, 1, `watch:${w}/`, this.watchEl);
            }
        }
    }

    renderStack() {
        const dbg = this.app.debugger;
        clear(this.stackEl);
        if (!dbg.active) return;
        if (!dbg.stopped) { this.stackEl.appendChild(h('div', { class: 'muted small pad' }, t('debug.runningHint'))); return; }
        dbg.frames.forEach((f, i) => {
            const path = dbg.framePath(f);
            const row = h('div', { class: ['stack-row', i === dbg.frameIndex && 'selected'], role: 'option', 'aria-selected': String(i === dbg.frameIndex), tabindex: '0' },
                icon(i === 0 ? 'debugArrow' : 'symbolFunction', { className: i === 0 ? 'frame-top' : 'frame-icon' }),
                h('span', { class: 'frame-name' }, f.name),
                h('span', { class: 'frame-loc' }, path ? `${basename(path)}:${f.line}` : f.line ? `:${f.line}` : ''));
            const go = async () => {
                await dbg.selectFrame(i);
                if (path) this.app.openFile(path, { line: f.line, focus: false });
            };
            row.addEventListener('click', go);
            row.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
            this.stackEl.appendChild(row);
        });
        if (dbg.stopReason) this.secStack.badge.textContent = t(`debug.reason.${dbg.stopReason}`) !== `debug.reason.${dbg.stopReason}` ? t(`debug.reason.${dbg.stopReason}`) : dbg.stopReason;
    }

    renderBreakpoints() {
        clear(this.bpEl);
        const all = [];
        for (const [path, list] of this.app.breakpoints) for (const b of list) all.push({ path, b });
        this.secBps.badge.textContent = all.length ? String(all.length) : '';
        if (!all.length) { this.bpEl.appendChild(h('div', { class: 'muted small pad' }, t('debug.noBreakpoints'))); return; }
        all.sort((x, y) => x.path.localeCompare(y.path) || x.b.line - y.b.line);
        for (const { path, b } of all) {
            const box = h('input', { type: 'checkbox', checked: b.enabled !== false, 'aria-label': t('bp.enabledFor', { file: basename(path), line: b.line }) });
            box.addEventListener('change', () => {
                const list = (this.app.breakpoints.get(path) || []).map((x) => (x.line === b.line ? Object.assign({}, x, { enabled: box.checked }) : x));
                this.app.setBreakpoints(path, list);
            });
            const rm = actionBtn('close', 'bp.remove', () => {
                this.app.setBreakpoints(path, (this.app.breakpoints.get(path) || []).filter((x) => x.line !== b.line));
            });
            const row = h('div', { class: 'bp-row' }, box, h('span', { class: ['bp-dot-mini', b.enabled === false && 'off'] }),
                h('span', { class: 'bp-file' }, basename(path)), h('span', { class: 'bp-line' }, String(b.line)),
                b.condition ? h('code', { class: 'bp-cond' }, b.condition) : null,
                b.hitCondition ? h('code', { class: 'bp-cond' }, `# ${b.hitCondition}`) : null, rm);
            row.addEventListener('click', (e) => { if (e.target === box || e.target.closest('.tree-action')) return; this.app.openFile(path, { line: b.line }); });
            this.bpEl.appendChild(row);
        }
    }

    removeAllBreakpoints() {
        for (const path of [...this.app.breakpoints.keys()]) this.app.setBreakpoints(path, []);
    }

    relocalize() {
        this.argsInput.placeholder = t('debug.argsPlaceholder');
        this.stdinInput.placeholder = t('debug.stdinPlaceholder');
        this.watchInput.placeholder = t('debug.addWatch');
        this.renderBackends();
        this.render();
    }
}

function valueClass(v) {
    const s = String(v.value || '');
    const ty = String(v.type || '');
    if (/^".*"$|^'.*'$/.test(s) || ty === 'str') return 'v-str';
    if (/^-?\d/.test(s) || /^(int|float)$/.test(ty)) return 'v-num';
    if (/^(true|false)$/.test(s) || ty === 'bool') return 'v-bool';
    if (v.variablesReference) return 'v-obj';
    return '';
}

/* the card shown in the web face for features of the desktop server */
export function desktopCard(app, feature) {
    const cmd = 'til ide';
    const copy = h('button', { class: 'btn small', type: 'button' }, icon('copy'), L('desktop.copy'));
    copy.addEventListener('click', () => app.copyText(cmd));
    return h('div', { class: 'desktop-card', dataset: { feature } },
        h('div', { class: 'dc-icon' }, icon('lock')),
        h('div', { class: 'dc-title' }, t('desktop.title', { feature: t(`feature.${feature}`) })),
        h('p', { class: 'dc-text' }, t('desktop.text')),
        h('div', { class: 'dc-cmd' }, h('code', {}, cmd), copy),
        h('a', { class: 'dc-link', href: 'https://github.com/damn-glitch/TIL#installation', target: '_blank', rel: 'noopener' }, t('desktop.install'), icon('external')));
}
