#!/usr/bin/env bash
# Build the TIL online playground (website/playground/).
#
#   website/playground/build.sh            build into website/playground/
#   OUT=/tmp/site/playground build.sh      build into another directory
#
# Produces (all committed, because the website is static):
#   tilvm.wasm        vm/tilvm.c (unmodified) + wasm_host.c + shim/ for the browser
#   tilc.tbc          the self-hosted compiler selfhost/tilc.til, compiled by
#                     src/til.py (skipped with a message while it does not exist)
#   examples/*.tbc    the bundled examples, precompiled by src/til.py - only
#                     while there is no tilc.tbc (development fallback); they
#                     are removed once tilc.tbc is built
#   build-info.json   inputs (sha256), toolchain and configuration of the build,
#                     and whether the VM has the screen/key natives ("graphics")
#
# Needs: clang with the wasm32 target and wasm-ld (lld), python3.
# Settings (environment): CLANG, PYTHON, TILVM_WASM_HEAP (bytes per VM,
# default 8 MB), TILVM_WASM_VMS (instances, default 2), WERROR=0 to allow
# warnings, TILVM_C (a different tilvm.c to build, e.g. a candidate VM; the
# default is vm/tilvm.c; its tilvm.h is taken from the same directory),
# TILC_TIL (a different self-hosted compiler source; default selfhost/tilc.til),
# EXAMPLES=0 (skip the precompiled examples, e.g. for test builds).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$HERE}"
CLANG="${CLANG:-clang}"
PYTHON="${PYTHON:-python3}"
HEAP="${TILVM_WASM_HEAP:-8388608}"
NVMS="${TILVM_WASM_VMS:-2}"
WERROR="${WERROR:-1}"
TILVM_C="${TILVM_C:-$ROOT/vm/tilvm.c}"
TILC_TIL="${TILC_TIL:-$ROOT/selfhost/tilc.til}"

mkdir -p "$OUT" "$OUT/examples"

say() { printf '[playground] %s\n' "$*"; }

if ! command -v "$CLANG" >/dev/null 2>&1; then
    echo "build.sh: $CLANG not found (clang with the wasm32 target is required)" >&2
    exit 1
fi

# ---------------------------------------------------------------- tilvm.wasm
DEFINES=(
    -DTILVM_NO_STDIO
    -DTILVM_INT_BITS=64
    -DTILVM_FLOAT=2
    "-DTILVM_HEAP_SIZE=$HEAP"
    -DTILVM_MAX_CONSTS=16384
    -DTILVM_MAX_TYPES=1024
    "-DWH_MAX_VMS=$NVMS"
)
CFLAGS=(
    --target=wasm32 -O2 -std=c99 -ffreestanding -nostdlib
    -fno-exceptions -mbulk-memory
    -Wall -Wextra -pedantic
    -I"$HERE/shim" -I"$(dirname "$TILVM_C")" -I"$ROOT/vm"
)
if [ "$WERROR" != "0" ]; then CFLAGS+=(-Werror); fi
LDFLAGS=(
    -Wl,--no-entry
    -Wl,--export=__stack_pointer      # tilvm.js restores it after a pause (see wasm_host.c)
    -Wl,--stack-first                 # a C stack overflow traps instead of corrupting data
    -Wl,-z,stack-size=1048576
    -Wl,--strip-debug
)
SOURCES=("$TILVM_C" "$HERE/wasm_host.c" "$HERE/shim/libc.c")

say "compiling tilvm.wasm (heap $HEAP bytes x $NVMS VMs)"
"$CLANG" "${CFLAGS[@]}" "${DEFINES[@]}" "${LDFLAGS[@]}" -o "$OUT/tilvm.wasm.tmp" "${SOURCES[@]}"
mv "$OUT/tilvm.wasm.tmp" "$OUT/tilvm.wasm"
chmod 644 "$OUT/tilvm.wasm"

# The module may only import what tilvm.js provides (catches a newer
# vm/tilvm.c that needs a C library function the shim does not have yet).
"$PYTHON" - "$OUT/tilvm.wasm" <<'PYEOF'
import sys

ALLOWED = {
    "env": {"write", "now", "read_line", "key", "file_read", "file_write", "screen", "suspend", "fmod"},
    "Math": {"pow", "exp", "expm1", "log", "log10", "log2", "log1p", "sin", "cos", "tan", "asin",
             "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cbrt", "hypot"},
}


def leb(data, pos):
    result = shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return result, pos


def name(data, pos):
    n, pos = leb(data, pos)
    return data[pos:pos + n].decode("utf-8"), pos + n


data = open(sys.argv[1], "rb").read()
assert data[:8] == b"\0asm\x01\0\0\0", "not a wasm module"
pos = 8
bad = []
while pos < len(data):
    sid = data[pos]
    size, pos = leb(data, pos + 1)
    end = pos + size
    if sid == 2:                       # import section
        count, p = leb(data, pos)
        for _ in range(count):
            mod, p = name(data, p)
            field, p = name(data, p)
            kind = data[p]
            p += 1
            if kind == 0:
                _, p = leb(data, p)
            else:
                bad.append(f"{mod}.{field} (non-function import kind {kind})")
                break
            if field not in ALLOWED.get(mod, ()):
                bad.append(f"{mod}.{field}")
    pos = end
if bad:
    sys.exit("tilvm.wasm imports functions tilvm.js does not provide: " + ", ".join(bad) +
             "\n(add them to shim/ or to tilvm.js)")
PYEOF
say "tilvm.wasm: $(wc -c < "$OUT/tilvm.wasm") bytes"

# ------------------------------------------------------------------ tilc.tbc
TILC_SRC="$TILC_TIL"
if [ -f "$TILC_SRC" ]; then
    say "compiling $TILC_SRC -> tilc.tbc"
    "$PYTHON" "$ROOT/src/til.py" build --target vm "$TILC_SRC" -o "$OUT/tilc.tbc" >/dev/null
    say "tilc.tbc: $(wc -c < "$OUT/tilc.tbc") bytes"
else
    say "selfhost/tilc.til not found: skipping tilc.tbc (the page runs the precompiled examples instead)"
    rm -f "$OUT/tilc.tbc"
fi

# ---------------------------------------------------------- examples/*.tbc
if [ -f "$OUT/tilc.tbc" ]; then
    say "tilc.tbc present: the browser compiles everything; removing precompiled examples"
    rm -f "$OUT"/examples/*.tbc
elif [ "${EXAMPLES:-1}" = "0" ]; then
    say "EXAMPLES=0: not precompiling the examples"
else
    count="$("$PYTHON" - "$ROOT/src" "$HERE/examples" "$OUT/examples" <<'PYEOF'
import os
import sys

sys.path.insert(0, sys.argv[1])
import til  # noqa: E402

src_dir, out_dir = sys.argv[2], sys.argv[3]
names = sorted(f[:-4] for f in os.listdir(src_dir) if f.endswith(".til"))
for f in os.listdir(out_dir):                  # drop bytecode of removed examples
    if f.endswith(".tbc") and f[:-4] not in names:
        os.remove(os.path.join(out_dir, f))
for name in names:
    path = os.path.join(src_dir, name + ".til")
    with open(path, encoding="utf-8") as f:
        data = til.compile_to_bytecode(f.read(), path).serialize()
    with open(os.path.join(out_dir, name + ".tbc"), "wb") as f:
        f.write(data)
print(len(names))
PYEOF
)"
    say "precompiled $count examples into examples/*.tbc (fallback until tilc.tbc exists)"
fi

# --------------------------------------------------------- build-info.json
"$PYTHON" - "$HERE" "$ROOT" "$OUT" "$CLANG" "$HEAP" "$NVMS" "$TILVM_C" "$TILC_TIL" <<'PYEOF'
import hashlib
import json
import os
import re
import subprocess
import sys

here, root, out, clang, heap, nvms, vm_c, tilc = sys.argv[1:9]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


vm_h = os.path.join(os.path.dirname(vm_c), "tilvm.h")
if not os.path.exists(vm_h):
    vm_h = os.path.join(root, "vm", "tilvm.h")
inputs = {"vm/tilvm.c": sha(vm_c), "vm/tilvm.h": sha(vm_h)}
for rel in ("website/playground/wasm_host.c",
            "website/playground/shim/libc.c", "website/playground/shim/string.h",
            "website/playground/shim/math.h", "website/playground/build.sh"):
    inputs[rel] = sha(os.path.join(root, rel))
if os.path.exists(tilc):
    inputs["selfhost/tilc.til"] = sha(tilc)
outputs = {"tilvm.wasm": sha(os.path.join(out, "tilvm.wasm"))}
if os.path.exists(os.path.join(out, "tilc.tbc")):
    outputs["tilc.tbc"] = sha(os.path.join(out, "tilc.tbc"))
version = subprocess.run([clang, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
with open(vm_c, encoding="utf-8") as f:
    graphics = re.search(r"case\s+N_SCREEN_INIT\s*:", f.read()) is not None   # natives 40-53 implemented
info = {
    "inputs": inputs,
    "outputs": outputs,
    "clang": version,
    "heap_bytes_per_vm": int(heap),
    "vms": int(nvms),
    "compiler": "tilc" if "tilc.tbc" in outputs else "precompiled-examples",
    "graphics": graphics,
}
if os.path.realpath(vm_c) != os.path.realpath(os.path.join(root, "vm", "tilvm.c")):
    info["vm_source"] = vm_c
if os.path.exists(tilc) and os.path.realpath(tilc) != os.path.realpath(os.path.join(root, "selfhost", "tilc.til")):
    info["tilc_source"] = tilc
with open(os.path.join(out, "build-info.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, sort_keys=True)
    f.write("\n")
PYEOF
say "done ($OUT)"
