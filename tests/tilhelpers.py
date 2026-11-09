"""
Shared helpers for the TIL test-suite (used by the TIL 3.1 feature tests).

    from tilhelpers import til, run_vm, run_cvm, run_both, cli, needs_cc

Not a test module itself (pytest only collects test_*.py).
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import til  # noqa: E402

TIL = os.path.join(SRC, 'til.py')
VM_DIR = os.path.join(ROOT, 'vm')
CC = shutil.which('gcc') or shutil.which('clang') or shutil.which('cc')
needs_cc = pytest.mark.skipif(CC is None, reason="no C compiler")

# C VM build profiles (same flags as tests/test_v3.py, plus int32)
PROFILES = {
    "host": [],
    "tiny": ['-DTILVM_LEAN', '-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0', '-DTILVM_HEAP_SIZE=2048',
             '-DTILVM_STACK_SIZE=128', '-DTILVM_MAX_FRAMES=24', '-DTILVM_MAX_CONSTS=128',
             '-DTILVM_MAX_TYPES=16', '-DTILVM_MAX_TRY=4'],
    "small": ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1', '-DTILVM_HEAP_SIZE=16384',
              '-DTILVM_STACK_SIZE=512', '-DTILVM_MAX_FRAMES=64'],
    "int32": ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1'],
    "gcstress": ['-DTILVM_HEAP_SIZE=12000'],
}

_CVM = {}


def compile_vm(src: str, filename: str = "<test>") -> 'til.TBCModule':
    """TIL source -> bytecode module (raises til.CompileError / SyntaxError)."""
    return til.compile_to_bytecode(src, filename)


def tbc_bytes(src: str, filename: str = "<test>") -> bytes:
    return compile_vm(src, filename).serialize()


def run_vm(src, max_steps: int = 0, argv=None, **vm_kwargs):
    """Run on the reference (Python) VM. `src` is TIL source or .tbc bytes.
    Returns (exit code, stdout text)."""
    data = src if isinstance(src, (bytes, bytearray)) else tbc_bytes(src)
    module = til.TBCModule.from_bytes(bytes(data))
    out = []
    rc = til.TILVM(module, write=out.append, max_steps=max_steps, argv=argv, **vm_kwargs).run()
    return rc, b"".join(out).decode('utf-8', 'replace')


def cvm_binary(profile: str = "host", extra_flags=()) -> str:
    """Build the C VM (vm/tilvm.c + vm/main.c) once per session and profile,
    with the strict flags every build must pass."""
    key = (profile, tuple(extra_flags))
    if key in _CVM:
        return _CVM[key]
    if CC is None:
        pytest.skip("no C compiler")
    exe = os.path.join(tempfile.mkdtemp(prefix='tilvm-'), 'tilvm-' + profile)
    cmd = [CC, '-std=c99', '-O2', '-Wall', '-Wextra', '-pedantic', '-Werror'] + PROFILES[profile] + \
        list(extra_flags) + ['-o', exe, os.path.join(VM_DIR, 'tilvm.c'), os.path.join(VM_DIR, 'main.c'), '-lm']
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"C VM build failed ({profile}):\n{r.stderr}"
    _CVM[key] = exe
    return exe


def run_cvm(src, profile: str = "host", vm_args=(), prog_args=(), timeout: int = 60, cwd=None):
    """Run on the C VM: tilvm [vm_args] prog.tbc [prog_args]. `src` is TIL source
    or .tbc bytes. Returns (exit code, stdout, stderr)."""
    data = src if isinstance(src, (bytes, bytearray)) else tbc_bytes(src)
    fd, path = tempfile.mkstemp(suffix='.tbc')
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    try:
        r = subprocess.run([cvm_binary(profile), *vm_args, path, *prog_args], capture_output=True,
                           timeout=timeout, cwd=cwd)
        return r.returncode, r.stdout.decode('utf-8', 'replace'), r.stderr.decode('utf-8', 'replace')
    finally:
        os.unlink(path)


def run_both(src, prog_args=(), profile: str = "host", expect_rc: int = 0) -> str:
    """Run on the Python VM and the C VM; require the same exit code and output."""
    rc, out = run_vm(src, argv=list(prog_args))
    assert rc == expect_rc, f"Python VM exit {rc}\n{out}"
    if CC:
        crc, cout, cerr = run_cvm(src, profile=profile, prog_args=prog_args)
        assert crc == expect_rc, f"C VM exit {crc}\n{cerr}"
        assert cout == out, f"C VM differs from the Python VM:\n{cout!r}\n{out!r}"
    return out


def cli(*args, cwd=None, input_text=None, env=None, timeout: int = 120):
    """Run `python3 src/til.py ARGS...`; returns the CompletedProcess (text mode)."""
    full_env = None
    if env:
        full_env = dict(os.environ)
        full_env.update(env)
    return subprocess.run([sys.executable, TIL, *args], capture_output=True, text=True,
                          cwd=cwd, input=input_text, timeout=timeout, env=full_env)


def expect_compile_error(src: str, *fragments: str) -> str:
    with pytest.raises((til.CompileError, SyntaxError)) as info:
        compile_vm(src)
    text = str(info.value)
    for frag in fragments:
        assert frag in text, f"{frag!r} not in:\n{text}"
    return text
