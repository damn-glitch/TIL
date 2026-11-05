#!/usr/bin/env node
/*
 * tilvm_node.js - run a .tbc program on tilvm.wasm in Node, like the desktop
 * `tilvm` runner (used by tests/test_playground.py to compare the browser VM
 * with the desktop C VM).
 *
 *   node tilvm_node.js [options] program.tbc [args...]
 *     --wasm FILE      tilvm.wasm to use (default: ../tilvm.wasm)
 *     --keys LIST      keys for key_pressed()/wait_key(), the desktop VM's syntax:
 *                      "UP,DOWN,ENTER,a,COMMA,NONE" (wait_key() returns 27 = Esc
 *                      once the list is used up, as on the desktop)
 *     --budget N       steps per wh_run call (default 20000)
 *     --screen FILE    write the framebuffer as it is at the end as a binary PPM
 *                      (P6), whether or not screen_show() was called (desktop VM)
 *     --stats          print status/steps/slices to stderr at the end
 *
 * stdin feeds input(); files are the real file system (like the desktop VM);
 * sleep_ms() really waits. The exit code is the program's (0, 1 on errors,
 * n after exit(n)); without --keys, a wait_key() stops with exit code 1.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const TILWasm = require(path.join(__dirname, '..', 'tilvm.js'));

function usage() {
    process.stderr.write('usage: node tilvm_node.js [--wasm F] [--keys LIST] [--budget N] [--screen F] ' +
        '[--stats] program.tbc [args...]\n');
    process.exit(2);
}

async function main() {
    const argv = process.argv.slice(2);
    let wasm = path.join(__dirname, '..', 'tilvm.wasm');
    let keys = null, budget = 20000, screenOut = null, stats = false, prog = null, progArgs = [];
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--wasm' && i + 1 < argv.length) wasm = argv[++i];
        else if (a === '--keys' && i + 1 < argv.length) {
            try {
                keys = TILWasm.parseKeys(argv[++i]);
            } catch (e) {
                process.stderr.write(`tilvm_node: ${e.message}\n`);
                process.exit(2);
            }
        }
        else if (a === '--budget' && i + 1 < argv.length) budget = parseInt(argv[++i], 10);
        else if (a === '--screen' && i + 1 < argv.length) screenOut = argv[++i];
        else if (a === '--stats') stats = true;
        else if (a.startsWith('--')) usage();
        else { prog = a; progArgs = argv.slice(i + 1); break; }
    }
    if (!prog) usage();

    const rt = await TILWasm.Runtime.create(fs.readFileSync(wasm));
    rt.fs = {
        read(p) { try { return new Uint8Array(fs.readFileSync(p)); } catch (e) { return null; } },
        write(p, bytes) { try { fs.writeFileSync(p, bytes); return true; } catch (e) { return false; } },
    };
    const vm = rt.createVM();
    const out = [], err = [];
    vm.onOutput = (text, stream) => (stream === 2 ? err : out).push(text);
    const flush = () => {
        if (out.length) { process.stdout.write(out.join('')); out.length = 0; }
        if (err.length) { process.stderr.write(err.join('')); err.length = 0; }
    };
    try {
        vm.load(fs.readFileSync(prog));
    } catch (e) {
        process.stderr.write(`tilvm_node: cannot load ${prog}: ${e.message}\n`);
        process.exit(2);
    }
    vm.setArgs(progArgs);
    let stdinData = null;
    if (keys) vm.setKeyScript(keys);

    let slices = 0, st;
    for (;;) {
        st = vm.run(budget);
        slices++;
        flush();
        if (st === TILWasm.STATUS.YIELD) continue;
        if (st === TILWasm.STATUS.SLEEP) {
            const ms = vm.sleepMs;
            if (ms > 0) await new Promise((r) => setTimeout(r, ms));
            continue;
        }
        if (st === TILWasm.STATUS.WAIT_LINE) {
            if (stdinData === null) {
                try { stdinData = fs.readFileSync(0); } catch (e) { stdinData = Buffer.alloc(0); }
                vm.pushInput(new Uint8Array(stdinData));
                vm.closeInput();
                continue;
            }
            process.stderr.write('tilvm_node: input() is waiting but stdin is exhausted\n');
            process.exit(1);
        }
        if (st === TILWasm.STATUS.WAIT_KEY) {
            process.stderr.write('tilvm_node: wait_key(): no more keys\n');
            st = TILWasm.STATUS.HOST_ERROR;
        }
        break;
    }
    flush();
    if (screenOut) {
        vm.showFinalScreen();
        const s = rt.screen();
        if (s) {
            const rgb = Buffer.alloc(s.width * s.height * 3);
            for (let i = 0, j = 0; i < s.pixels.length; i += 4) {
                rgb[j++] = s.pixels[i]; rgb[j++] = s.pixels[i + 1]; rgb[j++] = s.pixels[i + 2];
            }
            fs.writeFileSync(screenOut, Buffer.concat([Buffer.from(`P6\n${s.width} ${s.height}\n255\n`), rgb]));
        }
    }
    if (stats) {
        process.stderr.write(`[tilvm_node] status=${TILWasm.STATUS_NAMES[st]} steps=${vm.steps} ` +
            `slices=${slices} gc=${vm.gcCount} heap=${vm.heapUsed}\n`);
    }
    process.exitCode = st === TILWasm.STATUS.HOST_ERROR ? 1 : vm.exitCode;
}

main().catch((e) => { process.stderr.write(`tilvm_node: ${e && e.stack ? e.stack : e}\n`); process.exit(2); });
