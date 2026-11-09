# TIL energy saving

TIL 3.1 programs can save battery on the TIL VM: every few hundred steps the
virtual machine gives the CPU away for a millisecond — a desktop process
sleeps, a calculator halts until its next interrupt, a microcontroller waits in
a low-power mode. Results never depend on it: the same program prints the same
output in every mode, only the energy it uses (and the wall time of CPU-bound
loops) changes.

```til
#[energy: balanced]                 # the program's mode (a pragma before every definition)

#[energy: low]                      # this function always runs as in mode low
sync_log(entries) -> int
    ...

main()
    let saved = energy_mode(0)      # a burst at full speed; returns the previous mode
    render_frame()
    energy_mode(saved)
```

Contents: [modes](#modes) · [the pragma](#the-pragma-energy-mode-of-the-program) ·
[functions](#energy-low-functions) · [energy_mode()](#energy_mode) ·
[the exact rule](#the-exact-rule) · [bytecode and VM state](#bytecode-and-vm-state) ·
[hosts: implementing idle](#hosts-implementing-idle) ·
[build options and costs](#build-options-and-costs) · [other tools](#other-tools)

Example: [`examples/14_energy.til`](../examples/14_energy.til). Tests:
`tests/test_energy.py`.

## Modes

| mode | name | the VM idles |
|---|---|---|
| 0 | `performance` | never (the default) |
| 1 | `balanced` | 1 ms every 4096 steps |
| 2 | `low` | 1 ms every 512 steps |

A *step* is a loop back-edge or a function call — the unit of `--max-steps`.
The C VM runs some tens of millions of steps per second on a desktop (about
30 million on the machine of the measurements below), the reference VM a few
hundred thousand, an 8-bit CPU a few thousand. On a desktop, 512 steps take
about 17 µs: a CPU-bound loop in mode `low` then uses a few per cent of a core
and runs about 60 times longer, in mode `balanced` about 8 times longer
(measured with `tilvm`: 4 % and 12 % CPU per wall second). On a slow CPU,
where 512 steps take many milliseconds, the idles are a small share of a busy
loop's time (more where the device's idle lasts until its next interrupt);
there most of the saving comes from waiting: `sleep_ms()` and a polling
`wait_key()` idle instead of spinning, in every mode. What the VM does while
it idles is up to the device (see [hosts](#hosts-implementing-idle)).

## The pragma: energy mode of the program

`#[energy: low]`, `#[energy: balanced]` or `#[energy: performance]` at the top
level, **before every definition** of the file (functions, structs, enums,
imports, globals...), sets the mode the program starts in. It is written into
the bytecode header, so `til build --target vm` keeps it.

```til
#[energy: low]

main()
    print("I run in mode low")
```

Where `#[energy: low]` could also mark a function, the rule is: **directly
above a function definition** (only other attributes and comment lines in
between) it marks that function; **followed by an empty line**, by a
statement or by another kind of item (a struct, an import...), it is the
pragma. `balanced` and `performance` never mark a
function, so `#[energy: balanced]` directly above the first function is the
pragma too.

```til
#[energy: low]              # the pragma (an empty line follows)

#[energy: low]              # marks fetch()
fetch() -> int
    ...
```

* Compile errors: an unknown mode (`unknown energy mode 'turbo' (use low,
  balanced or performance)`), a pragma after a definition, a second pragma
  (`the program's energy mode is set twice`).
* Only the main file's pragma counts. A module's pragma is checked like any
  other, but importing the module does not change the program's mode (as a
  module's `main()` is not imported either); its `#[energy: low]` functions do
  stay energy-low.

## Energy-low functions

`#[energy: low]` directly above a function sets its flag `energy-low`
(`TBCFunction.FLAG_ENERGY_LOW` = 8 in the function record). **While any frame
of an energy-low function is active** — the function itself, everything it
calls, an effect clause it performs into — the VM behaves as in mode 2,
whatever the program's mode. A lambda created inside such a function is not
energy-low itself: called later from elsewhere, it runs in the program's mode.
Methods (`impl`) and `pub` / `fn` functions can be marked the same way.

`#[energy: medium]` and `#[energy: high]` (TIL 3.0) are accepted on functions
and only steer the native backend's optimization; they mark nothing for the
VMs. Any other value on a function is an error (`unknown energy level 'x' for a
function (use #[energy: low])`).

## energy_mode()

| call | result |
|---|---|
| `energy_mode()` | the program's mode (0, 1 or 2) |
| `energy_mode(m)` | sets the mode to `m` and returns the previous one |

`m` must be the integer 0, 1 or 2; anything else (3, -1, `2.0`, `"2"`, `true`)
is the runtime error `energy_mode() needs a mode 0 (performance), 1
(balanced) or 2 (low)` on both VMs (a `retry` catches it). `energy_mode()`
reports the program's mode, not the effective one: inside an energy-low
function it still returns what the program set, and `energy_mode(m)` there
changes the program's mode for later (the function keeps running as mode 2).
So `let saved = energy_mode(0) ... energy_mode(saved)` always restores the
mode.

## The exact rule

Both VMs (the reference VM `class TILVM` in `src/til.py` and the C VM
`vm/tilvm.c`) count steps identically, and they idle at exactly the same
steps:

> at step *s* (the step counter after it is incremented), the effective mode
> is 2 if any active frame belongs to an energy-low function, else the
> program's mode; the VM idles once (1 ms) if the effective mode is 2 and *s*
> is a multiple of 512, or it is 1 and *s* is a multiple of 4096.

A call's step is counted after the callee's frame is pushed (it is inside the
callee); a return is not a step. Within a step, the order is: step limit
(`--max-steps`: the failing step does not idle), **idle**, timeout deadlines,
the budget of `tilvm_run_steps()`. Mode changes take effect at the next step.
The step counter wraps at 2^32 in the C VM; 2^32 is a multiple of 4096, so the
rule is the same.

The idles of a run are counted: `TILVM.idle_count` in the reference VM, and
`tilvm --stats` prints them (only for programs that save energy):

```
[tilvm] steps=69628 gc=0 heap=544/262144 bytes
[tilvm] energy_mode=2 idle=44
```

`energy_mode=` is the program's mode at the end. `tests/test_energy.py`
compares the counts of both VMs (and of the C VM's lean and 32-bit builds)
with a model of the rule computed from a step trace.

## Bytecode and VM state

* **Header**: bits 2–3 of the flags byte (byte 5: bit 0 floats, bit 1 line
  table) hold the initial mode, 0–2; the rest of the layout is unchanged. A
  file with the value 3 is rejected (`bad bytecode header (energy mode 3)`;
  the C VM: `tilvm_load` error −5). `til dis` prints `; energy mode: low (2,
  #[energy: low])` and `[energy-low]` after a marked function.
* **Program state**: the mode is one byte — `vm->st.energy` in the C VM's
  `tilvm_state` (it took the former padding byte after `failed`, so the
  struct did not change size) and the attribute `TILVM.energy` in the
  reference VM. Raw snapshots (`tilvm_snapshot`, `tilvm --snapshot-after`)
  keep it; portable snapshots (`.tilp`) store it in their state header (byte
  24, docs/TIL_SNAPSHOTS.md), so a program paused in low mode on a PC keeps
  saving energy on the calculator. The idle counter (`vm->idle_count`, `TILVM.idle_count`) is a
  statistic of one run, not program state.
* `vm->flags` (the header byte) gets bit 7 in the C VM when some function is
  energy-low (computed by `tilvm_load`).

## Hosts: implementing idle

The C VM calls the host hook

```c
void (*idle)(void *ctx, unsigned long ms);     /* vm->host.idle */
```

* with `ms = 1` at the idle steps of the rule above;
* from `sleep_ms(n)`: with the time that is left, re-reading `clock_ms`
  after every call until the deadline has passed (a host that returns early
  is called again; a host may watch these clock reads — the playground does);
  without a clock (`TILVM_NO_TIME` and no `vm->clock_ms`) it calls
  `idle(n)` once;
* from `wait_key()` when the host has no `wait_key` hook: between polls of
  `key_pressed`.

`NULL` (the default after `tilvm_load`) means "never idle": the program then
runs at full speed and `sleep_ms` waits on the clock as in TIL 3.0. The hook
may return early (an interrupt, a key press) and may sleep longer than asked;
it may end the program as `exit(n)` does (`vm->st.exit_code = n;
vm->st.halted = 2;`), which `vm/main.c` does on Ctrl-C.

**Use a wall clock.** Timeouts and `sleep_ms()` read `vm->clock_ms`. The C
VM's built-in clock is the C library's `clock()` — CPU time on most systems,
which stands still while the CPU sleeps: with it, a `timeout(100)` around a
loop in mode low would wait much longer than 100 ms. A host that idles should
install a clock that keeps counting (a timer tick, an RTC, `CLOCK_MONOTONIC`).

Examples (sketches; the exact calls depend on the toolchain and the OS):

| device | `idle(ctx, ms)` | clock |
|---|---|---|
| desktop (`vm/main.c`) | `nanosleep()` (POSIX) / `Sleep(ms)` (Windows) | `clock_gettime(CLOCK_MONOTONIC)` / `GetTickCount64()` |
| Z80 calculators (TI-83 Plus / TI-84 Plus) | `ei` + `halt`: the CPU stops until the next interrupt — with TI-OS's timer interrupt that is several milliseconds, so one idle lasts longer than asked; the TI-84 Plus crystal timers give shorter waits | a counter incremented in the interrupt handler |
| eZ80 (TI-84 Plus CE) | `delay(ms)` of the CE toolchain (`<sys/timers.h>`), as `vm/ports/ti84ce/src/main.c` does | `clock()` (32768 Hz timer) |
| NumWorks (external app, Cortex-M7) | `eadk_timing_msleep(ms)` | `eadk_timing_millis()` |
| ARM Cortex-M (STM32, RP2040, ...) | `__WFI()` in a loop until a 1 kHz SysTick counter has advanced `ms` ticks (deeper sleep modes for longer waits) | the SysTick counter |
| AVR (Arduino) | `set_sleep_mode(SLEEP_MODE_IDLE); sleep_mode();` woken by a 1 kHz timer interrupt | `millis()` or the timer's counter |
| ESP32 (FreeRTOS) | `vTaskDelay(pdMS_TO_TICKS(ms))` with a 1000 Hz tick (`CONFIG_FREERTOS_HZ`; at 100 Hz, 1 ms is 0 ticks); the idle task can then enter light sleep | `esp_timer_get_time() / 1000` |
| web playground (`website/playground/wasm_host.c`) | nothing: a browser time slice cannot sleep, and pausing at a back-edge is impossible (the frame's saved pc is stale there); `sleep_ms()` pauses through its clock reads instead | `performance.now()` |

A Cortex-M host, for example:

```c
static volatile uint32_t ticks;                 /* SysTick_Handler: ticks++ (1 kHz) */
static uint32_t board_clock(void *ctx) { (void)ctx; return ticks; }
static void board_idle(void *ctx, unsigned long ms) {
    uint32_t start = ticks;
    (void)ctx;
    while (ticks - start < ms) __WFI();         /* sleep until the next interrupt */
}
...
vm.clock_ms = board_clock;
vm.host.idle = board_idle;
```

On a TI-84 Plus the same with the OS interrupt (`ei` + `halt` in a small
assembly routine). The desktop runner's hook (`host_idle` in `vm/main.c`)
checks Ctrl-C before and after it sleeps.

## Build options and costs

* **No cost per instruction.** The C VM's interpreter counts steps with a
  "fuel" counter (`docs/TIL_BYTECODE_V2.md`): only when it runs out does the
  slow path look at the step limit, deadlines and budgets. While energy saving
  can be active (mode 1 or 2, or the program has an energy-low function),
  `refuel()` also makes the counter end at the next multiple of 512 steps,
  and the slow path calls `energy_point()` there; `energy_mode(m)` refuels.
  Mode 0 programs without energy-low functions run the loop of before (one
  test of the native's number after a native call is the only addition). The
  reference VM's per-step check tests one flag, as it tested the step limit
  before; only with a step limit or energy saving on does it look further
  (`steps & 511`). `TILVM.energy` and `TILVM.max_steps` are properties that
  keep the flag right when they are assigned. Benchmarks: see below.
* `TILVM_LEAN` builds (which check every step anyway) include energy saving.
* `-DTILVM_NO_ENERGY` leaves it out: programs run at full speed whatever their
  mode, the header bits are ignored, and `energy_mode()` reports
  `energy_mode() is not available in this VM build`.
* The Z80 port (`vm/ports/z80`, TI-83 Plus / TI-84 Plus class) builds with
  `-DTILVM_NO_ENERGY`: energy saving costs its lean SDCC image 802 bytes, and
  only 474 were free below 0xC000 (`vm/ports/z80/RESULTS.md`). The TI-84 Plus
  CE port (`vm/ports/ti84ce`) includes it.

Costs measured for this feature against the integration branch before it
(commit `6f3daaa`; benchmark bytecode byte-identical; mode 0, the paths every
existing program takes). Instructions: valgrind callgrind, C VM `-O2`; times:
`benchmarks/run.py --baseline 6f3daaa --repeat 11`, medians of CPU seconds
(Xeon @ 2.1 GHz, gcc 13.3, Python 3.13; differences of a few per cent are
noise on this machine).

| benchmark | instructions, gcc 13 | instructions, clang 18 | C VM time before → after | Python VM time before → after |
|---|---:|---:|---:|---:|
| fib | −0.8% | −2.6% | 0.012 → 0.012 s | 1.98 → 1.96 s |
| loops | 0.0% | −5.2% | 0.027 → 0.027 s | 4.47 → 4.45 s |
| arrays | +0.5% | −3.2% | 0.019 → 0.019 s | 2.95 → 2.98 s |
| strings | +0.4% | −0.8% | 0.018 → 0.019 s | 0.50 → 0.48 s |
| maps | +0.4% | −0.4% | 0.011 → 0.010 s | 0.102 → 0.102 s |
| structs | −0.1% | −4.5% | 0.025 → 0.025 s | 2.29 → 2.21 s |
| closures | −0.4% | −0.8% | 0.022 → 0.022 s | 3.06 → 3.07 s |

Forcing `refuel()` inline matters: with the energy term GCC 13 stopped
inlining it, which cost 1.2–3.2% more instructions on five of the seven
benchmarks.

## Other tools

* **Native backend** (`til run` / `til build` without `--vm`): native code
  never idles. `#[energy: low]` on a function keeps its TIL 3.0 meaning there
  (optimize for size), the pragma is ignored, and a program that calls
  `energy_mode()` is run on the VM (`til run` falls back, as for the other
  VM-only natives).
* **Debugger** (`til debug`): counts idles exactly like the VM but does not
  sleep (a debugging session saves no battery, and re-running history must be
  fast); the mode and the idle count are part of its state snapshots
  (`DebugVM.STATE_NUMBERS`), so `back` restores them.
* **Self-hosted compiler** (`selfhost/tilc.til`): the same placement rules,
  errors, header bits and function flags, byte for byte.
* **`til fit`**: energy saving needs no memory beyond the 4-byte idle counter
  in `struct tilvm` (part of `til fit`'s model); programs fit or not as before.
