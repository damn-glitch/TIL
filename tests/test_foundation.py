#!/usr/bin/env python3
"""
TIL 3.1 foundation: bytecode v2, new natives (files, bytes, program
arguments), reserved opcodes, native arity checks, CLI program arguments.
Everything runs on both VMs and must behave identically.

Run: python3 -m pytest tests/test_foundation.py -v
"""

import os
import re
import shutil
import subprocess

import pytest

from tilhelpers import (CC, ROOT, VM_DIR, cli, compile_vm, cvm_binary, expect_compile_error, needs_cc,
                        run_both, run_cvm, run_vm, tbc_bytes, til)

C_SOURCE = open(os.path.join(VM_DIR, 'tilvm.c'), encoding='utf-8').read()


# ═══════════════════════════════════════════════════════════════
# BYTECODE FORMAT
# ═══════════════════════════════════════════════════════════════

class TestBytecodeVersion:
    def test_compiler_writes_version_2(self):
        data = tbc_bytes('print("hi")')
        assert data[:4] == b"TILB" and data[4] == 2 == til.TBC_VERSION

    @pytest.mark.parametrize("version", [1, 2])
    def test_versions_1_and_2_load_on_both_vms(self, version):
        data = bytearray(tbc_bytes('print(6 * 7)'))
        data[4] = version
        rc, out = run_vm(bytes(data))
        assert (rc, out) == (0, "42\n")
        if CC:
            crc, cout, cerr = run_cvm(bytes(data))
            assert (crc, cout) == (0, "42\n"), cerr

    def test_future_version_rejected(self):
        data = bytearray(tbc_bytes('print(1)'))
        data[4] = 3
        with pytest.raises(ValueError, match="unsupported bytecode version 3"):
            til.TBCModule.from_bytes(bytes(data))
        if CC:
            crc, _out, err = run_cvm(bytes(data))
            assert crc == 2 and "error -2" in err

    def test_reserved_opcodes(self):
        assert (til.Op.HANDLE, til.Op.END_HANDLE, til.Op.PERFORM, til.Op.RESUME) == (0x76, 0x77, 0x78, 0x79)
        assert til.OP_OPERANDS[til.Op.HANDLE] == 'bs' and til.OP_OPERANDS[til.Op.PERFORM] == 'hhb'
        for name, value in (('HANDLE', 0x76), ('END_HANDLE', 0x77), ('PERFORM', 0x78), ('RESUME', 0x79)):
            assert re.search(rf'OP_{name} = 0x{value:02X}\b', C_SOURCE), name

    def test_function_flags(self):
        assert til.TBCFunction.FLAG_ENERGY_LOW == 8 and til.TBCFunction.FLAG_EFFECT_CLAUSE == 16
        assert '#define FLAG_ENERGY_LOW 8' in C_SOURCE and '#define FLAG_EFFECT_CLAUSE 16' in C_SOURCE


class TestNativeTables:
    def test_names_match_c_vm(self):
        names = re.search(r'NATIVE_NAMES\[\] = \{(.*?)\};', C_SOURCE, re.S).group(1)
        assert re.findall(r'"(\w+)"', names) == til.NATIVES

    def test_new_natives_are_appended(self):
        assert til.NATIVES[33] == 'println'
        assert til.NATIVES[34:40] == ['read_file', 'write_file', 'file_exists', 'args',
                                      'bytes_to_str', 'str_bytes']
        assert til.NATIVES[40] == 'screen_init' and til.NATIVES[54] == 'energy_mode'
        assert len(til.NATIVES) == 55

    def test_min_args_match_c_vm(self):
        table = re.search(r'NATIVE_MIN_ARGS\[\] = \{(.*?)\};', C_SOURCE, re.S).group(1)
        table = re.sub(r'/\*.*?\*/', '', table, flags=re.S)
        assert [int(x) for x in re.findall(r'\d+', table)] == til.NATIVE_MIN_ARGS
        assert set(til.NATIVE_ARITY) == set(til.NATIVES)


# ═══════════════════════════════════════════════════════════════
# NEW NATIVES (both VMs)
# ═══════════════════════════════════════════════════════════════

FILES_PROGRAM = '''
let a = args()
let p = a[0]
write_file(p, "line 1\\nсәлем")
print(file_exists(p))
let t = read_file(p)
print(t)
print(len(t))
write_file(p, 42)
print(read_file(p) + "!")
print(file_exists(p + ".missing"))
'''


class TestFiles:
    def test_write_read_exists(self, tmp_path):
        out = run_both(FILES_PROGRAM, prog_args=[str(tmp_path / "f.txt")])
        assert out == "true\nline 1\nсәлем\n17\n42!\nfalse\n"

    def test_binary_bytes_survive(self, tmp_path):
        target = tmp_path / "bin.dat"
        src = '''
let p = args()[0]
var bytes = []
for i in 0..256
    bytes.push(i)
write_file(p, bytes_to_str(bytes))
let back = str_bytes(read_file(p))
print(len(back))
print(back[0], back[127], back[128], back[255])
'''
        assert run_both(src, prog_args=[str(target)]) == "256\n0\n127\n128\n255\n"
        assert target.read_bytes() == bytes(range(256))

    def test_errors_are_catchable_and_identical(self, tmp_path):
        missing = str(tmp_path / "nope" / "x.txt")
        src = '''
let p = args()[0]
let r = retry(1)
    read_file(p)
else
    error
print(r)
let w = retry(1)
    write_file(p, "x")
else
    error
print(w)
let d = retry(1)
    read_file(args()[1])
else
    "dir: " + error
print(d)
print(file_exists(args()[1]))
let e = retry(1)
    read_file("")
else
    error
print(e)
let k = retry(1)
    read_file(5)
else
    error
print(k)
'''
        out = run_both(src, prog_args=[missing, str(tmp_path)])
        assert out == (f"cannot read file: {missing}\ncannot write file: {missing}\n"
                       f"dir: cannot read file: {tmp_path}\nfalse\n"
                       "read_file(): invalid path\nread_file() expects a path string, got int\n")


class TestBytesAndArgs:
    def test_bytes_round_trip(self):
        src = '''
print(str_bytes("AZ"))
print(str_bytes("ә"))
print(bytes_to_str([72, 105, 33]))
print(bytes_to_str(str_bytes("сәлем, TIL")))
print(len(str_bytes("")))
let e = retry(1)
    bytes_to_str([1, 256])
else
    error
print(e)
let f = retry(1)
    bytes_to_str("no")
else
    error
print(f)
'''
        assert run_both(src) == ("[65, 90]\n[211, 153]\nHi!\nсәлем, TIL\n0\n"
                                 "bytes_to_str() expects an array of ints 0..255\n"
                                 "bytes_to_str() expects an array of ints 0..255\n")

    def test_args(self):
        src = 'let a = args()\nprint(len(a))\nfor x in a\n    print("[" + x + "]")\n'
        assert run_both(src, prog_args=["one", "two words", "сәлем", "--flag"]) == \
            "4\n[one]\n[two words]\n[сәлем]\n[--flag]\n"
        assert run_both(src) == "0\n"

    def test_large_string_from_file(self, tmp_path):
        # 70 KB: more than the old 64 KB string limit of the C VM (heap: 256 KB)
        big = tmp_path / "big.txt"
        big.write_bytes(b"0123456789" * 7000)
        src = 'let t = read_file(args()[0])\nprint(len(t))\nprint(t[69999])\nprint(len(t + "!"))\n'
        assert run_both(src, prog_args=[str(big)]) == "70000\n57\n70001\n"


# ═══════════════════════════════════════════════════════════════
# ARITY CHECKS
# ═══════════════════════════════════════════════════════════════

class TestNativeArity:
    @pytest.mark.parametrize("call, fragment", [
        ("len()", "'len' takes 1 argument(s), got 0"),
        ("pow(2)", "'pow' takes 2 argument(s), got 1"),
        ("args(1)", "'args' takes 0 argument(s), got 1"),
        ("write_file(\"x\")", "'write_file' takes 2 argument(s), got 1"),
        ("min()", "'min' takes at least 1 argument(s), got 0"),
        ("input(1, 2)", "'input' takes 0 to 1 argument(s), got 2"),
    ])
    def test_compile_time(self, call, fragment):
        expect_compile_error(f"print({call})", fragment)

    def test_range_with_one_argument(self):
        assert run_both("print(range(4))\nfor i in range(2)\n    print(i)\n", ) == "[0, 1, 2, 3]\n0\n1\n"

    def _patched(self, native: str, new_argc: int, new_id=None) -> bytes:
        m = compile_vm(f'print({native}("ab"))')
        nid = til.NATIVE_ID[native]
        for f in m.funcs:
            at = f.code.find(bytes([til.Op.NATIVE, nid, 1]))
            if at >= 0:
                f.code[at + 2] = new_argc
                if new_id is not None:
                    f.code[at + 1] = new_id
                break
        else:
            raise AssertionError("NATIVE instruction not found")
        return m.serialize()

    def _run_tbc_cli(self, data: bytes, tmp_path):
        path = tmp_path / "p.tbc"
        path.write_bytes(data)
        r = cli("exec", str(path))
        return r.returncode, r.stderr

    def test_runtime_minimum_both_vms(self, tmp_path):
        data = self._patched('len', 0)
        rc, err = self._run_tbc_cli(data, tmp_path)
        assert rc == 1 and "len() is missing arguments" in err
        if CC:
            crc, _o, cerr = run_cvm(data)
            assert crc == 1 and "len() is missing arguments" in cerr

    def test_unknown_native_both_vms(self, tmp_path):
        data = self._patched('len', 1, new_id=200)
        rc, err = self._run_tbc_cli(data, tmp_path)
        assert rc == 1 and "?() is not available in this VM build" in err
        if CC:
            crc, _o, cerr = run_cvm(data)
            assert crc == 1 and "?() is not available in this VM build" in cerr


class TestEffects:
    def test_file_natives_are_io(self):
        src = '#[pure]\nload(p: str) -> str\n    return read_file(p)\n\nmain()\n    print(load("x"))\n'
        text = expect_compile_error(src, "E-EFFECT")
        assert "io" in text

    def test_bytes_natives_are_pure(self):
        src = '#[pure]\ncodes(s: str) -> str\n    return bytes_to_str(str_bytes(s))\n\nmain()\n    print(codes("ok"))\n'
        assert run_both(src) == "ok\n"


# ═══════════════════════════════════════════════════════════════
# COMMAND LINE
# ═══════════════════════════════════════════════════════════════

ARGS_PROGRAM = 'for x in args()\n    print("<" + x + ">")\n'


class TestCli:
    def _write(self, tmp_path, text=ARGS_PROGRAM, name="a.til"):
        p = tmp_path / name
        p.write_text(text, encoding='utf-8')
        return str(p)

    def test_run_vm_passes_program_arguments(self, tmp_path):
        r = cli("run", "--vm", self._write(tmp_path), "one", "-x", "two words")
        assert r.returncode == 0, r.stderr
        assert r.stdout == "<one>\n<-x>\n<two words>\n"

    def test_double_dash(self, tmp_path):
        r = cli("run", "--vm", self._write(tmp_path), "a", "--", "--vm", "-o", "b")
        assert r.stdout == "<a>\n<--vm>\n<-o>\n<b>\n", r.stderr

    def test_til_options_after_the_file_still_work(self, tmp_path):
        r = cli("run", self._write(tmp_path, 'print("vm")'), "--vm")
        assert r.returncode == 0 and r.stdout == "vm\n" and "native" not in r.stderr

    @needs_cc
    def test_cvm_and_native_fallback_pass_arguments(self, tmp_path):
        prog = self._write(tmp_path)
        r = cli("run", "--cvm", prog, "x", "y")
        assert r.returncode == 0 and r.stdout == "<x>\n<y>\n", r.stderr
        r = cli("run", prog, "z")             # native backend falls back to the VM
        assert r.returncode == 0 and r.stdout == "<z>\n", r.stderr

    def test_exec_bytecode_with_arguments(self, tmp_path):
        prog = self._write(tmp_path)
        out = str(tmp_path / "a.tbc")
        r = cli("build", "--target", "vm", prog, "-o", out)
        assert r.returncode == 0, r.stderr
        r = cli("exec", out, "q", "w")
        assert r.returncode == 0 and r.stdout == "<q>\n<w>\n", r.stderr

    def test_value_options_are_parsed(self, tmp_path, monkeypatch):
        seen = {}

        def fake_run(module, max_steps=0, argv=None, opts=None):
            seen['argv'], seen['opts'] = argv, dict(opts or {})
            return 0
        monkeypatch.setattr(til, 'run_bytecode', fake_run)
        monkeypatch.setattr(til.sys, 'argv', ['til', 'run', '--vm', '--lang', 'kk', '--screen=o.ppm',
                                              '--screen-ansi', self._write(tmp_path), 'p1'])
        assert til.main() == 0
        assert seen['argv'] == ['p1']
        assert seen['opts'] == {'lang': 'kk', 'screen': 'o.ppm', 'screen_ansi': True}

    def test_missing_feature_module_is_reported(self, monkeypatch, capsys):
        monkeypatch.setitem(til.EXTRA_COMMANDS, 'zzz', ('til_no_such_tool', 'test'))
        assert til.run_extra_command('zzz', []) == 1
        assert "til_no_such_tool.py is missing" in capsys.readouterr().err


# ═══════════════════════════════════════════════════════════════
# STRICT PORTABLE BUILDS
# ═══════════════════════════════════════════════════════════════

CONFIGS = [
    [],
    ['-DTILVM_INT_BITS=32', '-DTILVM_FLOAT=1'],
    ['-DTILVM_LEAN', '-DTILVM_INT_BITS=16', '-DTILVM_FLOAT=0', '-DTILVM_NO_STDIO'],
    ['-DTILVM_NO_STDIO', '-DTILVM_NO_TIME'],
    ['-DTILVM_NO_STDIO', '-DTILVM_NO_MATH', '-DTILVM_FLOAT=1'],
]


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("flags", CONFIGS, ids=["default", "int32", "lean16", "nostdio", "nomath"])
def test_strict_c99_builds(compiler, flags, tmp_path):
    if not shutil.which(compiler):
        pytest.skip(f"{compiler} not installed")
    r = subprocess.run([compiler, '-std=c99', '-Wall', '-Wextra', '-pedantic', '-Werror', *flags,
                        '-c', os.path.join(VM_DIR, 'tilvm.c'), '-o', str(tmp_path / 'tilvm.o')],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@needs_cc
def test_int32_build_runs_file_natives(tmp_path):
    out = tmp_path / "o.txt"
    crc, cout, cerr = run_cvm(FILES_PROGRAM, profile="int32", prog_args=[str(out)])
    assert crc == 0, cerr
    assert cout == "true\nline 1\nсәлем\n17\n42!\nfalse\n"


def test_cvm_binary_helper_builds():
    if CC:
        assert os.path.exists(cvm_binary())
    assert os.path.isdir(ROOT)
