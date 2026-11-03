#!/usr/bin/env python3
"""
TIL 3.1 package manager: `til pkg` (src/tilpkg.py) and `import NAME` of
installed packages (til.py: find_package_module).

Never uses the network: local directories, git repositories created in a
temporary directory (git+file://), archives through file:// URLs, a
registry index file, and a fake urlopen for the https:// code path.

Run: python3 -m pytest tests/test_pkg.py -v
"""

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.error
import zipfile

import pytest

from tilhelpers import cli, needs_cc, run_both, til

import tilpkg  # noqa: E402  (src/ is on sys.path via tilhelpers)

needs_git = pytest.mark.skipif(shutil.which('git') is None, reason="git is not installed")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def write(path, text):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def read(path):
    with open(os.fspath(path), encoding='utf-8') as f:
        return f.read()


def make_pkg(directory, name, version=None, deps=None, body=None, main=None, manifest=True):
    """A package directory with til.json and NAME.til defining NAME_id()."""
    data = {"name": name}
    if version:
        data["version"] = version
    if deps:
        data["dependencies"] = deps
    if main:
        data["main"] = main
    if manifest:
        write(os.path.join(directory, 'til.json'), json.dumps(data))
    if body is None:
        body = f'{name}_id() -> str\n    return "{name} {version or "-"}"\n'
    write(os.path.join(directory, main or f'{name}.til'), body)
    return directory


def make_project(directory, deps=None, name="app", **extra):
    data = {"name": name, "version": "0.1.0", "dependencies": deps or {}}
    data.update(extra)
    write(os.path.join(directory, 'til.json'), json.dumps(data))
    return directory


def lock_of(project):
    return json.loads(read(os.path.join(project, 'til.lock')))


def manifest_of(project):
    return json.loads(read(os.path.join(project, 'til.json')))


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def tilpkg_run(capsys):
    """tilpkg_run(project_dir, *args) -> (exit code, stdout, stderr), in-process."""
    def run(project, *args):
        capsys.readouterr()
        rc = tilpkg.main(['-C', os.fspath(project), *args])
        out, err = capsys.readouterr()
        return rc, out, err
    return run


GIT = ['git', '-c', 'user.name=TIL Tests', '-c', 'user.email=tests@example.com', '-c', 'commit.gpgsign=false',
       '-c', 'tag.gpgsign=false', '-c', 'init.defaultBranch=main', '-c', 'core.autocrlf=false']


class GitRepo:
    def __init__(self, path):
        self.path = path
        os.makedirs(path)
        self.git('init', '-q')

    def git(self, *args):
        r = subprocess.run(GIT + list(args), cwd=self.path, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()

    def commit(self, files, message="change"):
        for rel, text in files.items():
            write(os.path.join(self.path, rel), text)
        self.git('add', '-A')
        self.git('commit', '-q', '-m', message)
        return self.git('rev-parse', 'HEAD')

    def spec(self, rev=None):
        return 'git+' + _uri(self.path) + (f'#{rev}' if rev else '')


def _uri(path):
    from pathlib import Path
    return Path(os.path.abspath(path)).as_uri()


def colors_release(version, value):
    return {'til.json': json.dumps({"name": "colors", "version": version}),
            'colors.til': f'red() -> int\n    return {value}\n'}


def make_tar(path, entries):
    """entries: (name, kind, payload) with kind file|dir|sym|hard."""
    with tarfile.open(path, 'w:gz') as tf:
        for name, kind, payload in entries:
            info = tarfile.TarInfo(name)
            if kind == 'file':
                data = payload.encode('utf-8')
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            else:
                info.type = {'dir': tarfile.DIRTYPE, 'sym': tarfile.SYMTYPE, 'hard': tarfile.LNKTYPE}[kind]
                info.linkname = payload or ''
                tf.addfile(info)
    return path


def make_zip(path, entries):
    with zipfile.ZipFile(path, 'w') as zf:
        for name, kind, payload in entries:
            info = zipfile.ZipInfo(name)
            if kind == 'sym':
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            elif kind == 'file':
                info.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(info, payload or '')
    return path


def archive_spec(path):
    return f"{_uri(path)}#sha256={sha256_file(path)}"


# ═══════════════════════════════════════════════════════════════
# NAMES, VERSIONS, REQUIREMENTS, SPECS
# ═══════════════════════════════════════════════════════════════

class TestNames:
    @pytest.mark.parametrize("name", ["geo", "_x", "abc123", "Geo_2", "геометрия", "sandar"])
    def test_valid(self, name):
        assert tilpkg.check_name(name) == name

    @pytest.mark.parametrize("name", ["", "../x", "..", ".", "a/b", "a\\b", "a.b", "my-lib", "1abc", "a b",
                                      "if", "import", "CON", "nul", "x" * 65, "geo\n"])
    def test_invalid(self, name):
        with pytest.raises(tilpkg.PkgError):
            tilpkg.check_name(name)


class TestVersions:
    def test_semver_precedence(self):
        order = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2",
                 "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0", "1.0.1", "1.2.0", "1.10.0", "2.0.0"]
        versions = [tilpkg.Version.parse(v) for v in order]
        assert sorted(reversed(versions)) == versions
        assert tilpkg.Version.parse("1.0.0+build.5") == tilpkg.Version.parse("1.0.0")

    @pytest.mark.parametrize("text", ["1", "1.2", "01.2.3", "1.2.3-", "v1.2.3", "1.2.3.4", "", "x"])
    def test_invalid_versions(self, text):
        with pytest.raises(tilpkg.PkgError):
            tilpkg.Version.parse(text)

    @pytest.mark.parametrize("req, yes, no", [
        ("^1.2", ["1.2.0", "1.9.9"], ["1.1.9", "2.0.0", "1.3.0-beta"]),
        ("^1.2.3", ["1.2.3", "1.5.0"], ["1.2.2", "2.0.0"]),
        ("^0.2.3", ["0.2.3", "0.2.9"], ["0.3.0", "0.2.2"]),
        ("^0.0.3", ["0.0.3"], ["0.0.4", "0.0.2"]),
        ("^0", ["0.0.1", "0.9.9"], ["1.0.0"]),
        ("~1.2.3", ["1.2.3", "1.2.9"], ["1.3.0", "1.2.2"]),
        ("~1.2", ["1.2.0", "1.2.9"], ["1.3.0"]),
        ("~1", ["1.0.0", "1.9.0"], ["2.0.0"]),
        ("1.x", ["1.0.0", "1.99.1"], ["2.0.0", "0.9.0"]),
        ("1.2.*", ["1.2.0", "1.2.7"], ["1.3.0"]),
        ("1.2", ["1.2.0", "1.2.4"], ["1.3.0"]),
        ("*", ["0.0.1", "9.9.9"], ["1.0.0-rc.1"]),
        ("", ["1.0.0"], []),
        ("1.2.3", ["1.2.3", "1.2.3+meta"], ["1.2.4", "1.2.3-rc.1"]),
        ("=1.2.3", ["1.2.3"], ["1.2.4"]),
        (">=1.2 <2", ["1.2.0", "1.9.9"], ["1.1.0", "2.0.0", "2.0.0-rc.1"]),
        (">=1.2, <2", ["1.5.0"], ["2.1.0"]),
        (">= 1.2.0", ["1.2.0", "3.0.0"], ["1.1.9"]),
        (">1.2", ["1.3.0"], ["1.2.9"]),
        ("<=1.2", ["1.2.9"], ["1.3.0"]),
        ("<1.2", ["1.1.9"], ["1.2.0"]),
        ("^1.0 || ^3.0", ["1.4.0", "3.1.0"], ["2.0.0"]),
        ("^2.1.0-beta", ["2.1.0-beta", "2.1.0-beta.2", "2.1.0", "2.5.0"], ["2.1.0-alpha", "2.2.0-beta", "3.0.0"]),
    ])
    def test_requirements(self, req, yes, no):
        r = tilpkg.Requirement(req)
        for v in yes:
            assert r.matches(tilpkg.Version.parse(v)), (req, v)
        for v in no:
            assert not r.matches(tilpkg.Version.parse(v)), (req, v)

    @pytest.mark.parametrize("req", ["1.x.3", ">", "<*", "1.2 - 2.0", "abc", "^x.1", ">=1.2.3.4", "1.2-beta"])
    def test_invalid_requirements(self, req):
        with pytest.raises(tilpkg.PkgError, match="invalid version requirement"):
            tilpkg.Requirement(req)


class TestSpecs:
    def parse(self, raw, base="/proj"):
        return tilpkg.parse_spec(raw, base, "dep", "til.json")

    def test_kinds(self, tmp_path):
        base = str(tmp_path)
        assert self.parse("./libs/dep", base).kind == 'path'
        assert self.parse("./libs/dep", base).path == os.path.realpath(os.path.join(base, 'libs', 'dep'))
        assert self.parse("../dep", base).kind == 'path'
        assert self.parse(base, base).path == os.path.realpath(base)
        git = self.parse("git+https://example.com/r.git#v1.2.0")
        assert (git.kind, git.url, git.rev) == ('git', "https://example.com/r.git", "v1.2.0")
        assert self.parse("git+file:///srv/repo").rev is None
        assert self.parse("git+git@example.com:user/repo.git#main").url == "git@example.com:user/repo.git"
        local_git = self.parse("git+../repo", base)
        assert local_git.path == os.path.realpath(os.path.join(base, '..', 'repo'))
        arc = self.parse("https://example.com/p-1.0.tar.gz#sha256=" + "AB" * 32)
        assert (arc.kind, arc.fmt, arc.sha256) == ('archive', 'tar', "ab" * 32)
        assert self.parse("file:///tmp/p.tgz").fmt == 'tar'
        assert self.parse("file:///tmp/p.zip#sha256=" + "0" * 64).fmt == 'zip'
        assert self.parse("^1.2").kind == 'registry'
        assert self.parse("dep@~1.2").req.text == "~1.2"
        assert self.parse("*").kind == 'registry'

    @pytest.mark.parametrize("raw, message", [
        ("", "needs a spec"),
        ("http://example.com/p.tar.gz#sha256=" + "0" * 64, "https://"),
        ("https://example.com/p.rar", "must end in"),
        ("file:///tmp/dir", "must end in"),
        ("https://example.com/p.zip#md5=abc", "unsupported URL fragment"),
        ("file://server/share/p.zip", "local file"),
        ("git+ext::sh -c touch% /tmp/pwned", "invalid git URL"),
        ("git+--upload-pack=touch /tmp/pwned", "invalid git URL"),
        ("git+-oProxyCommand=x", "invalid git URL"),
        ("git+https://example.com/r.git#--upload-pack=x", "invalid git revision"),
        ("git+https://example.com/r.git#a..b", "invalid git revision"),
        ("git+svn://example.com/r", "unsupported git URL scheme"),
        ("other@^1.0", "registry name must be the dependency name"),
        ("latest", "is not a path"),
    ])
    def test_invalid(self, raw, message):
        with pytest.raises(tilpkg.PkgError, match=message.replace('+', r'\+').replace('.', r'\.')):
            self.parse(raw)

    def test_path_dependencies_need_a_local_declarer(self):
        with pytest.raises(tilpkg.PkgError, match="local path"):
            tilpkg.parse_spec("../x", None, "x", "til.json of package 'remote'")
        with pytest.raises(tilpkg.PkgError, match="local path"):
            tilpkg.parse_spec("git+../repo", None, "x", "til.json of package 'remote'")
        assert tilpkg.parse_spec("git+file:///srv/repo", None, "x", "w").kind == 'git'


# ═══════════════════════════════════════════════════════════════
# TREE HASH
# ═══════════════════════════════════════════════════════════════

class TestTreeHash:
    def test_definition(self, tmp_path):
        """sha256 over `<sha256 of file>  <path>\\n` lines sorted by path bytes (sha256sum format)."""
        root = tmp_path / 'p'
        files = {'b.til': 'b\n', 'a/z.til': 'z', 'a/B.til': '', 'Ä.til': 'unicode name'}
        for rel, text in files.items():
            write(root / rel, text)
        lines = sorted((rel.encode('utf-8'), hashlib.sha256(text.encode('utf-8')).hexdigest())
                       for rel, text in files.items())
        expected = hashlib.sha256(b"".join(h.encode() + b"  " + rel + b"\n" for rel, h in lines)).hexdigest()
        assert tilpkg.tree_hash(str(root)) == expected

    def test_ignores_vcs_and_til_modules_but_not_content(self, tmp_path):
        root = tmp_path / 'p'
        write(root / 'p.til', 'x')
        h = tilpkg.tree_hash(str(root))
        write(root / '.git' / 'HEAD', 'ref')
        write(root / 'til_modules' / 'q' / 'q.til', 'q')
        write(root / 'sub' / '.git', 'gitdir: x')
        os.makedirs(root / 'empty' / 'dirs')
        assert tilpkg.tree_hash(str(root)) == h
        write(root / 'p.til', 'y')
        assert tilpkg.tree_hash(str(root)) != h
        write(root / 'p.til', 'x')
        write(root / 'new.til', '')
        assert tilpkg.tree_hash(str(root)) != h

    @pytest.mark.skipif(not hasattr(os, 'symlink') or sys.platform == 'win32', reason="needs symlinks")
    def test_symlinks_refused(self, tmp_path):
        root = tmp_path / 'p'
        write(root / 'p.til', 'x')
        os.symlink('/etc/passwd', root / 'passwd')
        with pytest.raises(tilpkg.PkgError, match="symbolic link"):
            tilpkg.tree_hash(str(root))


# ═══════════════════════════════════════════════════════════════
# LOCAL PACKAGES: init, add, install, remove, list, verify
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def libs(tmp_path):
    """libs/geo 1.2.0 -> units 0.3.0 (path dependency) ; libs/extra 2.0.0."""
    d = tmp_path / 'libs'
    make_pkg(d / 'units', 'units', '0.3.0',
             body='const SCALE = 10\nscaled(x: int) -> int\n    return x * SCALE\n')
    make_pkg(d / 'geo', 'geo', '1.2.0', deps={'units': '../units'},
             body='import units\narea(w: int, h: int) -> int\n    return scaled(w) * h\n')
    make_pkg(d / 'extra', 'extra', '2.0.0')
    return d


class TestInit:
    def test_init(self, tmp_path, tilpkg_run):
        proj = tmp_path / 'my_app'
        os.makedirs(proj)
        rc, out, _err = tilpkg_run(proj, 'init')
        assert rc == 0 and "my_app" in out
        assert manifest_of(proj) == {"name": "my_app", "version": "0.1.0", "dependencies": {}}
        assert read(proj / 'til.json').endswith('}\n')
        rc, _out, err = tilpkg_run(proj, 'init')
        assert rc == 1 and "already exists" in err

    def test_init_name(self, tmp_path, tilpkg_run):
        proj = tmp_path / 'my-app'
        os.makedirs(proj)
        rc, _out, err = tilpkg_run(proj, 'init', '../evil')
        assert rc == 1 and "invalid package name" in err and not (proj / 'til.json').exists()
        assert tilpkg_run(proj, 'init')[0] == 0
        assert manifest_of(proj)["name"] == "my_app"        # derived from the directory name
        other = tmp_path / 'other'
        os.makedirs(other)
        assert tilpkg_run(other, 'init', 'kitap')[0] == 0
        assert manifest_of(other)["name"] == "kitap"


class TestLocalPackages:
    def test_add_installs_transitive_dependencies(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app')
        rc, out, err = tilpkg_run(proj, 'add', 'geo', '../libs/geo')
        assert rc == 0, err
        assert "+ geo 1.2.0 (../libs/geo)" in out and "+ units 0.3.0 (../libs/units)" in out
        assert manifest_of(proj)["dependencies"] == {"geo": "../libs/geo"}
        assert sorted(os.listdir(proj / 'til_modules')) == ['geo', 'units']
        assert read(proj / 'til_modules' / 'geo' / 'geo.til') == read(libs / 'geo' / 'geo.til')
        lock = lock_of(proj)
        assert list(lock) == ["lockVersion", "packages"] and lock["lockVersion"] == 1
        assert list(lock["packages"]) == ["geo", "units"]
        geo = lock["packages"]["geo"]
        assert list(geo) == ["spec", "resolved", "version", "sha256", "dependencies"]
        assert geo["spec"] == "../libs/geo" and geo["resolved"] == "../libs/geo" and geo["version"] == "1.2.0"
        assert geo["dependencies"] == {"units": "../units"}
        assert geo["sha256"] == tilpkg.tree_hash(str(proj / 'til_modules' / 'geo'))
        units = lock["packages"]["units"]
        assert units["spec"] == "../units" and units["resolved"] == "../libs/units"

    def test_lock_is_deterministic_and_install_idempotent(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo", "extra": "../libs/extra"})
        assert tilpkg_run(proj, 'install')[0] == 0
        first = read(proj / 'til.lock')
        rc, out, _err = tilpkg_run(proj, 'install')
        assert rc == 0 and "up to date" in out and "written" not in out
        assert read(proj / 'til.lock') == first
        shutil.rmtree(proj / 'til_modules')
        os.remove(proj / 'til.lock')
        assert tilpkg_run(proj, 'install')[0] == 0
        assert read(proj / 'til.lock') == first          # byte-identical when re-resolved
        # with til_modules/ intact, install needs no source at all
        shutil.rmtree(libs)
        rc, out, err = tilpkg_run(proj, 'install')
        assert rc == 0 and "up to date" in out, err

    def test_remove(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo", "extra": "../libs/extra"})
        assert tilpkg_run(proj, 'install')[0] == 0
        rc, out, _err = tilpkg_run(proj, 'remove', 'extra')
        assert rc == 0 and "- extra" in out
        assert manifest_of(proj)["dependencies"] == {"geo": "../libs/geo"}
        assert sorted(os.listdir(proj / 'til_modules')) == ['geo', 'units']
        assert sorted(lock_of(proj)["packages"]) == ['geo', 'units']
        rc, _out, err = tilpkg_run(proj, 'remove', 'extra')
        assert rc == 1 and "not a dependency" in err
        # a direct dependency that is also needed by another package stays installed
        assert tilpkg_run(proj, 'add', 'units', '../libs/units')[0] == 0
        rc, out, _err = tilpkg_run(proj, 'remove', 'units')
        assert rc == 0 and "units stays installed: required by geo" in out
        assert sorted(os.listdir(proj / 'til_modules')) == ['geo', 'units']
        assert tilpkg_run(proj, 'rm', 'geo')[0] == 0
        assert not (proj / 'til_modules').exists() and lock_of(proj)["packages"] == {}

    def test_list(self, tmp_path, libs, tilpkg_run):
        make_pkg(libs / 'both', 'both', '1.0.0', deps={'units': '../units'})
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo", "both": "../libs/both"})
        assert tilpkg_run(proj, 'install')[0] == 0
        rc, out, _err = tilpkg_run(proj, 'list')
        assert rc == 0
        lines = out.splitlines()
        assert lines[0].startswith("app 0.1.0 (")
        assert lines[1:] == ["|-- both 1.0.0  ../libs/both",
                             "|   `-- units 0.3.0  ../units  [../libs/units]",
                             "`-- geo 1.2.0  ../libs/geo",
                             "    `-- units 0.3.0  ../units  [../libs/units]"]
        # not installed yet / missing from til_modules
        data = manifest_of(proj)
        data["dependencies"]["extra"] = "../libs/extra"
        write(proj / 'til.json', json.dumps(data))
        shutil.rmtree(proj / 'til_modules' / 'geo')
        out = tilpkg_run(proj, 'ls')[1]
        assert "extra  ../libs/extra  (not installed: run `til pkg install`)" in out
        assert "geo 1.2.0  ../libs/geo  (missing from til_modules/" in out

    def test_list_marks_repeated_subtrees(self, tmp_path, libs, tilpkg_run):
        make_pkg(libs / 'top', 'top', '1.0.0', deps={'geo': '../geo'})
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo", "top": "../libs/top"})
        assert tilpkg_run(proj, 'install')[0] == 0
        out = tilpkg_run(proj, 'list')[1]
        assert "|-- geo 1.2.0  ../libs/geo\n|   `-- units" in out
        assert "    `-- geo 1.2.0  ../geo  [../libs/geo]  (*)" in out

    def test_verify_detects_tampering(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo"})
        assert tilpkg_run(proj, 'install')[0] == 0
        rc, out, _err = tilpkg_run(proj, 'verify')
        assert rc == 0 and "ok       geo 1.2.0" in out and "verified 2 packages" in out
        geo_til = proj / 'til_modules' / 'geo' / 'geo.til'
        write(geo_til, read(geo_til) + "evil() -> int\n    return 666\n")
        rc, out, err = tilpkg_run(proj, 'verify')
        assert rc == 1 and "MODIFIED geo" in out and "ok       units" in out and "1 problem" in err
        write(proj / 'til_modules' / 'units' / 'extra.til', 'x')           # an added file
        shutil.rmtree(proj / 'til_modules' / 'geo')
        write(proj / 'til_modules' / 'intruder' / 'intruder.til', 'x')
        rc, out, _err = tilpkg_run(proj, 'verify')
        assert rc == 1
        assert "MISSING  geo" in out and "MODIFIED units" in out and "EXTRA    til_modules/intruder" in out
        # install restores exactly the locked trees
        rc, out, _err = tilpkg_run(proj, 'install')
        assert rc == 0 and "+ geo" in out and "~ units" in out and "- intruder" in out
        assert tilpkg_run(proj, 'verify')[0] == 0

    def test_verify_detects_outdated_lock(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo"})
        assert tilpkg_run(proj, 'verify')[0] == 1                     # no til.lock yet
        assert tilpkg_run(proj, 'install')[0] == 0
        data = manifest_of(proj)
        data["dependencies"]["extra"] = "../libs/extra"
        write(proj / 'til.json', json.dumps(data))
        rc, out, _err = tilpkg_run(proj, 'verify')
        assert rc == 1 and "OUTDATED extra" in out
        empty = make_project(tmp_path / 'empty')
        assert tilpkg_run(empty, 'verify')[0] == 0                    # nothing to verify

    def test_lock_reproduction_of_a_changed_directory(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo"})
        assert tilpkg_run(proj, 'install')[0] == 0
        original = read(libs / 'geo' / 'geo.til')
        write(libs / 'geo' / 'geo.til', original + "added() -> int\n    return 1\n")
        # til_modules/ still holds the locked files: they are kept
        rc, out, _err = tilpkg_run(proj, 'install')
        assert rc == 0 and "up to date" in out
        assert read(proj / 'til_modules' / 'geo' / 'geo.til') == original
        # from scratch: the changed source no longer matches til.lock -> hash error, nothing installed
        shutil.rmtree(proj / 'til_modules')
        lock_before = read(proj / 'til.lock')
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 1 and "do not match til.lock" in err and "install --update geo" in err
        assert not (proj / 'til_modules').exists() and read(proj / 'til.lock') == lock_before
        # --update accepts the new contents
        rc, out, _err = tilpkg_run(proj, 'install', '--update', 'geo')
        assert rc == 0 and "added() -> int" in read(proj / 'til_modules' / 'geo' / 'geo.til')
        assert lock_of(proj)["packages"]["geo"]["sha256"] != json.loads(lock_before)["packages"]["geo"]["sha256"]
        rc, _out, err = tilpkg_run(proj, 'install', '--update', 'nonexistent')
        assert rc == 1 and "not a dependency" in err

    def test_frozen(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo"})
        rc, _out, err = tilpkg_run(proj, 'install', '--frozen')
        assert rc == 1 and "does not exist" in err
        assert tilpkg_run(proj, 'install')[0] == 0
        shutil.rmtree(proj / 'til_modules')
        compact = json.dumps(lock_of(proj))                      # same content, other formatting
        write(proj / 'til.lock', compact)
        assert tilpkg_run(proj, 'install', '--frozen')[0] == 0
        assert (proj / 'til_modules' / 'units').is_dir()
        assert read(proj / 'til.lock') == compact                # --frozen never writes til.lock
        data = manifest_of(proj)
        data["dependencies"]["extra"] = "../libs/extra"
        write(proj / 'til.json', json.dumps(data))
        rc, _out, err = tilpkg_run(proj, 'install', '--frozen')
        assert rc == 1 and "out of date" in err and not (proj / 'til_modules' / 'extra').exists()

    def test_conflicting_sources(self, tmp_path, tilpkg_run):
        libs = tmp_path / 'libs'
        make_pkg(libs / 'c_one', 'c', '1.0.0')
        make_pkg(libs / 'c_two', 'c', '1.0.0')
        make_pkg(libs / 'a', 'a', deps={'c': '../c_one'})
        make_pkg(libs / 'b', 'b', deps={'c': '../c_two'})
        proj = make_project(tmp_path / 'app', {"a": "../libs/a"})
        assert tilpkg_run(proj, 'install')[0] == 0
        before = (read(proj / 'til.json'), read(proj / 'til.lock'), sorted(os.listdir(proj / 'til_modules')))
        rc, _out, err = tilpkg_run(proj, 'add', 'b', '../libs/b')
        assert rc == 1
        assert "conflicting requirements for package 'c'" in err
        assert "a requires c = ../c_one (= ../libs/c_one)" in err
        assert "b requires c = ../c_two (= ../libs/c_two)" in err
        assert (read(proj / 'til.json'), read(proj / 'til.lock'), sorted(os.listdir(proj / 'til_modules'))) == before

    def test_same_package_twice_is_fine(self, tmp_path, libs, tilpkg_run):
        make_pkg(libs / 'other', 'other', deps={'units': '../units'})
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo", "other": "../libs/other",
                                               "units": "../libs/units"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 0, err
        assert sorted(lock_of(proj)["packages"]) == ['geo', 'other', 'units']
        assert lock_of(proj)["packages"]["units"]["spec"] == "../libs/units"     # the project's own spec

    def test_cycles(self, tmp_path, tilpkg_run):
        libs = tmp_path / 'libs'
        make_pkg(libs / 'x', 'x', deps={'y': '../y'})
        make_pkg(libs / 'y', 'y', deps={'x': '../x'})
        proj = make_project(tmp_path / 'app', {"x": "../libs/x"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 1 and "dependency cycle: x -> y -> x" in err
        assert not (proj / 'til_modules').exists() and not (proj / 'til.lock').exists()
        make_pkg(libs / 'back', 'back', deps={'app': '../../app'})
        proj2 = make_project(tmp_path / 'app2', {"back": "../libs/back"}, name="app")
        rc, _out, err = tilpkg_run(proj2, 'install')
        assert rc == 1 and "dependency cycle: app -> back -> app" in err

    def test_package_name_must_match(self, tmp_path, tilpkg_run):
        make_pkg(tmp_path / 'libs' / 'geometry', 'geometry')
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geometry"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 1 and "is named 'geometry'" in err

    def test_package_without_manifest(self, tmp_path, tilpkg_run):
        make_pkg(tmp_path / 'libs' / 'plain', 'plain', manifest=False)
        proj = make_project(tmp_path / 'app', {"plain": "../libs/plain"})
        rc, out, err = tilpkg_run(proj, 'install')
        assert rc == 0 and "+ plain (../libs/plain)" in out, err
        assert lock_of(proj)["packages"]["plain"]["version"] is None

    def test_manifests_with_a_byte_order_mark(self, tmp_path, libs, tilpkg_run):
        with open(libs / 'extra' / 'til.json', 'w', encoding='utf-8-sig') as f:
            json.dump({"name": "extra", "version": "2.0.0"}, f)
        proj = tmp_path / 'app'
        os.makedirs(proj)
        with open(proj / 'til.json', 'w', encoding='utf-8-sig') as f:
            json.dump({"name": "app", "dependencies": {"extra": "../libs/extra"}}, f)
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 0, err
        assert lock_of(proj)["packages"]["extra"]["version"] == "2.0.0"
        write(proj / 'main.til', 'import extra\nprint(extra_id())\n')
        assert run_file_both(read(proj / 'main.til'), str(proj / 'main.til')) == "extra 2.0.0\n"

    @pytest.mark.skipif(not hasattr(os, 'symlink') or sys.platform == 'win32', reason="needs symlinks")
    def test_symlinked_package_directory(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app', {"extra": "../libs/extra"})
        assert tilpkg_run(proj, 'install')[0] == 0
        shutil.rmtree(proj / 'til_modules' / 'extra')
        os.symlink(os.path.abspath(libs / 'extra'), proj / 'til_modules' / 'extra')
        rc, out, _err = tilpkg_run(proj, 'verify')
        assert rc == 1 and "MODIFIED extra" in out and "symbolic link" in out
        rc, out, _err = tilpkg_run(proj, 'install')
        assert rc == 0 and "~ extra" in out
        assert not os.path.islink(proj / 'til_modules' / 'extra') and (libs / 'extra' / 'extra.til').exists()
        assert tilpkg_run(proj, 'verify')[0] == 0

    def test_unnamed_project_in_a_directory_named_like_a_dependency(self, tmp_path, libs, tilpkg_run):
        proj = tmp_path / 'geo'
        write(proj / 'til.json', json.dumps({"dependencies": {"geo": "../libs/geo"}}))
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 0, err
        assert sorted(lock_of(proj)["packages"]) == ['geo', 'units']
        assert tilpkg_run(proj, 'list')[1].splitlines()[0].startswith("geo (")
        assert tilpkg_run(proj, 'add', 'extra', '../libs/extra')[0] == 0

    def test_main_file_checks(self, tmp_path, tilpkg_run):
        libs = tmp_path / 'libs'
        make_pkg(libs / 'nomain', 'nomain')
        write(libs / 'nomain' / 'til.json', json.dumps({"name": "nomain", "main": "src/missing.til"}))
        make_pkg(libs / 'noentry', 'noentry', body="", main="src/util.til")
        write(libs / 'noentry' / 'til.json', json.dumps({"name": "noentry"}))
        write(libs / 'escape' / 'til.json', json.dumps({"name": "escape", "main": "../outside.til"}))
        proj = make_project(tmp_path / 'app', {"nomain": "../libs/nomain"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 1 and "src/missing.til does not exist" in err
        proj = make_project(tmp_path / 'app2', {"noentry": "../libs/noentry"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 0 and "warning: package 'noentry' has no" in err
        proj = make_project(tmp_path / 'app3', {"escape": "../libs/escape"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 1 and '"main" must be a relative path' in err

    def test_unsafe_names_rejected(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app')
        for bad in ('../evil', 'a/b', '..', 'x.y', 'C:\\evil'):
            rc, _out, err = tilpkg_run(proj, 'add', bad, '../libs/geo')
            assert rc == 1 and "invalid package name" in err, bad
        evil = make_project(tmp_path / 'evil', {"../../escape": "../libs/geo"})
        rc, _out, err = tilpkg_run(evil, 'install')
        assert rc == 1 and "invalid package name" in err
        make_pkg(libs / 'sneaky', 'sneaky', deps={'../x': '../geo'})
        proj2 = make_project(tmp_path / 'app2', {"sneaky": "../libs/sneaky"})
        rc, _out, err = tilpkg_run(proj2, 'install')
        assert rc == 1 and "invalid package name" in err
        assert not os.path.exists(tmp_path / 'escape') and not os.path.exists(tmp_path / 'x')

    def test_self_dependency_and_missing_directory(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app')
        rc, _out, err = tilpkg_run(proj, 'add', 'app', '.')
        assert rc == 1 and "cannot depend on itself" in err
        rc, _out, err = tilpkg_run(proj, 'add', 'me', '.')
        assert rc == 1 and "project itself" in err
        rc, _out, err = tilpkg_run(proj, 'add', 'nope', '../libs/nope')
        assert rc == 1 and "directory not found" in err
        assert manifest_of(proj)["dependencies"] == {}

    @pytest.mark.skipif(not hasattr(os, 'symlink') or sys.platform == 'win32', reason="needs symlinks")
    def test_symlink_in_local_package_rejected(self, tmp_path, libs, tilpkg_run):
        os.symlink(os.path.abspath(tmp_path / 'app'), libs / 'geo' / 'link')
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 1 and "symbolic link" in err

    def test_no_project(self, tmp_path, tilpkg_run):
        rc, _out, err = tilpkg_run(tmp_path, 'install')
        assert rc == 1 and "no til.json" in err and "til pkg init" in err

    def test_found_from_a_subdirectory(self, tmp_path, libs, tilpkg_run):
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo"})
        os.makedirs(proj / 'src' / 'deep')
        rc, _out, err = tilpkg_run(proj / 'src' / 'deep', 'install')
        assert rc == 0, err
        assert (proj / 'til_modules' / 'geo').is_dir() and not (proj / 'src' / 'deep' / 'til_modules').exists()


# ═══════════════════════════════════════════════════════════════
# GIT DEPENDENCIES (local repositories, git+file://)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def colors_repo(tmp_path):
    repo = GitRepo(str(tmp_path / 'colors'))
    repo.v1 = repo.commit(colors_release("1.0.0", 1), "v1")
    repo.git('tag', 'v1.0.0')
    repo.v2 = repo.commit(colors_release("2.0.0", 2), "v2")
    repo.git('tag', '-a', 'v2.0.0', '-m', 'annotated')
    return repo


@needs_git
class TestGit:
    def test_tag_and_commit(self, tmp_path, colors_repo, tilpkg_run):
        proj = make_project(tmp_path / 'app')
        rc, out, err = tilpkg_run(proj, 'add', 'colors', colors_repo.spec('v1.0.0'))
        assert rc == 0, err
        installed = proj / 'til_modules' / 'colors'
        assert "return 1" in read(installed / 'colors.til')
        assert not (installed / '.git').exists()
        entry = lock_of(proj)["packages"]["colors"]
        assert entry["spec"] == colors_repo.spec('v1.0.0')
        assert entry["resolved"] == colors_repo.spec(colors_repo.v1) and entry["version"] == "1.0.0"
        assert f"+ colors 1.0.0 ({colors_repo.spec(colors_repo.v1)})" in out
        # an annotated tag, a full commit SHA, an abbreviated SHA
        for rev, value in (('v2.0.0', 2), (colors_repo.v1, 1), (colors_repo.v2[:10], 2)):
            rc, out, err = tilpkg_run(proj, 'add', 'colors', colors_repo.spec(rev))
            assert rc == 0, err
            assert f"return {value}" in read(installed / 'colors.til')
            assert lock_of(proj)["packages"]["colors"]["resolved"].endswith(
                colors_repo.v1 if value == 1 else colors_repo.v2)
        assert tilpkg_run(proj, 'verify')[0] == 0

    def test_lock_pins_the_commit(self, tmp_path, colors_repo, tilpkg_run):
        proj = make_project(tmp_path / 'app', {"colors": colors_repo.spec()})         # remote HEAD
        assert tilpkg_run(proj, 'install')[0] == 0
        assert "return 2" in read(proj / 'til_modules' / 'colors' / 'colors.til')
        v3 = colors_repo.commit(colors_release("3.0.0", 3), "v3")                     # the branch moves on
        shutil.rmtree(proj / 'til_modules')
        rc, out, err = tilpkg_run(proj, 'install')
        assert rc == 0, err
        assert "return 2" in read(proj / 'til_modules' / 'colors' / 'colors.til')      # locked commit
        assert lock_of(proj)["packages"]["colors"]["resolved"].endswith('#' + colors_repo.v2)
        rc, out, err = tilpkg_run(proj, 'install', '--update')
        assert rc == 0 and "~ colors 2.0.0 -> 3.0.0" in out, err
        assert "return 3" in read(proj / 'til_modules' / 'colors' / 'colors.til')
        assert lock_of(proj)["packages"]["colors"]["resolved"].endswith('#' + v3)

    def test_branch(self, tmp_path, colors_repo, tilpkg_run):
        colors_repo.git('checkout', '-q', '-b', 'dev')
        colors_repo.commit(colors_release("2.1.0-dev", 21), "dev")
        colors_repo.git('checkout', '-q', 'main')
        proj = make_project(tmp_path / 'app', {"colors": colors_repo.spec('dev')})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 0, err
        assert "return 21" in read(proj / 'til_modules' / 'colors' / 'colors.til')
        assert lock_of(proj)["packages"]["colors"]["version"] == "2.1.0-dev"

    def test_bad_revision_and_repository(self, tmp_path, colors_repo, tilpkg_run):
        proj = make_project(tmp_path / 'app')
        rc, _out, err = tilpkg_run(proj, 'add', 'colors', colors_repo.spec('v9.9.9'))
        assert rc == 1 and "no commit, tag or branch 'v9.9.9'" in err
        rc, _out, err = tilpkg_run(proj, 'add', 'colors', 'git+' + _uri(tmp_path / 'missing'))
        assert rc == 1 and "git clone failed" in err
        rc, _out, err = tilpkg_run(proj, 'add', 'colors', colors_repo.spec('--upload-pack=touch'))
        assert rc == 1 and "invalid git revision" in err
        assert manifest_of(proj)["dependencies"] == {} and not (proj / 'til_modules').exists()

    def test_git_environment_does_not_leak(self, tmp_path, colors_repo, tilpkg_run, monkeypatch):
        # e.g. `til pkg install` run from a git hook, where git exports these
        monkeypatch.setenv('GIT_DIR', str(tmp_path / 'elsewhere' / '.git'))
        monkeypatch.setenv('GIT_INDEX_FILE', str(tmp_path / 'elsewhere' / 'index'))
        monkeypatch.setenv('GIT_WORK_TREE', str(tmp_path / 'elsewhere'))
        proj = make_project(tmp_path / 'app', {"colors": colors_repo.spec('v1.0.0')})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 0, err
        assert "return 1" in read(proj / 'til_modules' / 'colors' / 'colors.til')
        assert not (tmp_path / 'elsewhere').exists()

    def test_local_repository_path(self, tmp_path, colors_repo, tilpkg_run):
        proj = make_project(tmp_path / 'app', {"colors": "git+../colors#v1.0.0"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 0, err
        assert lock_of(proj)["packages"]["colors"]["resolved"] == f"git+../colors#{colors_repo.v1}"
        shutil.rmtree(proj / 'til_modules')
        assert tilpkg_run(proj, 'install')[0] == 0                                    # from the lock

    def test_git_package_cannot_use_path_dependencies(self, tmp_path, tilpkg_run):
        repo = GitRepo(str(tmp_path / 'remote'))
        repo.commit({'til.json': json.dumps({"name": "remote", "dependencies": {"x": "../x"}}),
                     'remote.til': 'r() -> int\n    return 0\n'})
        proj = make_project(tmp_path / 'app', {"remote": repo.spec()})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 1 and "til.json of package 'remote'" in err and "local path" in err

    def test_git_and_path_conflict(self, tmp_path, colors_repo, tilpkg_run):
        make_pkg(tmp_path / 'libs' / 'user', 'user', deps={'colors': colors_repo.spec('v1.0.0')})
        proj = make_project(tmp_path / 'app', {"colors": colors_repo.spec('v2.0.0'), "user": "../libs/user"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 1 and "conflicting requirements for package 'colors'" in err
        # the same commit through another name of the revision is not a conflict
        proj2 = make_project(tmp_path / 'app2', {"colors": colors_repo.spec(colors_repo.v1),
                                                 "user": "../libs/user"})
        rc, _out, err = tilpkg_run(proj2, 'install')
        assert rc == 0, err


# ═══════════════════════════════════════════════════════════════
# ARCHIVES (file:// URLs)
# ═══════════════════════════════════════════════════════════════

MATHX = [('mathx-1.0/til.json', 'file', '{"name": "mathx", "version": "1.0.0"}'),
         ('mathx-1.0/mathx.til', 'file', 'twice(x: int) -> int\n    return 2 * x\n'),
         ('mathx-1.0/docs/', 'dir', None)]


class TestArchives:
    def test_tar_gz_and_zip(self, tmp_path, tilpkg_run):
        tgz = make_tar(str(tmp_path / 'mathx-1.0.tar.gz'), MATHX)
        zipped = make_zip(str(tmp_path / 'ziplib.zip'),
                          [('ziplib.til', 'file', 'z() -> int\n    return 7\n'),
                           ('__MACOSX/._ziplib.til', 'file', 'junk')])
        proj = make_project(tmp_path / 'app')
        rc, out, err = tilpkg_run(proj, 'add', 'mathx', archive_spec(tgz))
        assert rc == 0, err
        assert "+ mathx 1.0.0" in out
        assert sorted(os.listdir(proj / 'til_modules' / 'mathx')) == ['mathx.til', 'til.json']  # top dir stripped
        rc, _out, err = tilpkg_run(proj, 'add', 'ziplib', archive_spec(zipped))
        assert rc == 0, err
        assert os.listdir(proj / 'til_modules' / 'ziplib') == ['ziplib.til']
        entry = lock_of(proj)["packages"]["ziplib"]
        assert entry["resolved"] == archive_spec(zipped) == entry["spec"]
        assert tilpkg_run(proj, 'verify')[0] == 0
        tgz2 = make_tar(str(tmp_path / 'p.tgz'), [('p.til', 'file', 'p() -> int\n    return 1\n')])
        assert tilpkg_run(proj, 'add', 'p', archive_spec(tgz2))[0] == 0

    def test_wrong_hash_rejected_before_extracting(self, tmp_path, tilpkg_run, monkeypatch):
        tgz = make_tar(str(tmp_path / 'mathx-1.0.tar.gz'), MATHX)
        proj = make_project(tmp_path / 'app')
        extracted = []
        monkeypatch.setattr(tilpkg, 'extract_archive', lambda *a: extracted.append(a))
        rc, _out, err = tilpkg_run(proj, 'add', 'mathx', f"{_uri(tgz)}#sha256={'0' * 64}")
        assert rc == 1 and "sha256 mismatch" in err and sha256_file(tgz) in err
        assert extracted == [] and manifest_of(proj)["dependencies"] == {}
        assert not (proj / 'til_modules').exists()
        rc, _out, err = tilpkg_run(proj, 'add', 'mathx', _uri(tgz))
        assert rc == 1 and f"{_uri(tgz)}#sha256={sha256_file(tgz)}" in err             # tells the hash
        assert extracted == []

    def test_lock_reproduction_of_a_changed_archive(self, tmp_path, tilpkg_run):
        tgz = make_tar(str(tmp_path / 'mathx-1.0.tar.gz'), MATHX)
        proj = make_project(tmp_path / 'app', {"mathx": archive_spec(tgz)})
        assert tilpkg_run(proj, 'install')[0] == 0
        make_tar(tgz, MATHX[:1] + [('mathx-1.0/mathx.til', 'file', 'twice(x: int) -> int\n    return 3 * x\n')])
        rc, out, _err = tilpkg_run(proj, 'install')
        assert rc == 0 and "up to date" in out                       # installed copy matches the lock
        shutil.rmtree(proj / 'til_modules')
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 1 and "sha256 mismatch" in err and not (proj / 'til_modules').exists()

    def test_https_downloads(self, tmp_path, tilpkg_run, monkeypatch):
        tgz = make_tar(str(tmp_path / 'mathx-1.0.tar.gz'), MATHX)
        with open(tgz, 'rb') as f:
            data = f.read()
        index = json.dumps({"packages": {"mathx": {
            "1.0.0": f"https://pkgs.example/mathx-1.0.tar.gz#sha256={sha256_file(tgz)}"}}}).encode()
        seen = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        def fake_urlopen(request, timeout=None):
            seen.append((request.full_url, request.get_header('User-agent'), timeout))
            if 'missing' in request.full_url:
                raise urllib.error.HTTPError(request.full_url, 404, 'Not Found', {}, None)
            return Response(index if request.full_url.endswith('.json') else data)

        monkeypatch.setattr(tilpkg.urllib.request, 'urlopen', fake_urlopen)
        proj = make_project(tmp_path / 'app')
        url = "https://pkgs.example/mathx-1.0.tar.gz"
        rc, _out, err = tilpkg_run(proj, 'add', 'mathx', f"{url}#sha256={sha256_file(tgz)}")
        assert rc == 0, err
        assert seen[-1][0] == url and seen[-1][1].startswith('til-pkg/') and seen[-1][2]
        rc, _out, err = tilpkg_run(proj, 'add', 'gone', f"https://pkgs.example/missing.tgz#sha256={'0' * 64}")
        assert rc == 1 and "HTTP 404" in err
        monkeypatch.setenv('TIL_REGISTRY', "https://pkgs.example/index.json")
        proj2 = make_project(tmp_path / 'app2', {"mathx": "^1.0"})
        rc, _out, err = tilpkg_run(proj2, 'install')
        assert rc == 0, err
        assert lock_of(proj2)["packages"]["mathx"]["resolved"] == f"{url}#sha256={sha256_file(tgz)}"
        rc, _out, err = tilpkg_run(proj, 'add', 'plain', 'http://pkgs.example/plain.tgz')
        assert rc == 1 and "https://" in err

    @pytest.mark.parametrize("entries, message", [
        ([('../evil.til', 'file', 'x')], "escape"),
        ([('pkg/../../evil.til', 'file', 'x')], "escape"),
        ([('{tmp}/abs_evil.til', 'file', 'x')], "absolute"),
        ([('C:\\evil.til', 'file', 'x')], "absolute"),
        ([('pkg/a.til', 'file', 'x'), ('pkg/link', 'sym', '/etc/passwd')], "symbolic link"),
        ([('pkg/link', 'sym', '../../..'), ('pkg/link/evil.til', 'file', 'x')], "symbolic link"),
        ([('pkg/a.til', 'file', 'x'), ('pkg/hard', 'hard', '../../evil.til')], "hard link"),
        ([('pkg/a.til', 'file', 'x'), ('pkg/a.til', 'file', 'y')], "duplicate"),
        ([('pkg/x:y.til', 'file', 'x')], "':'"),
    ])
    def test_unsafe_tar_rejected(self, tmp_path, tilpkg_run, entries, message):
        entries = [(name.replace('{tmp}', tmp_path.as_posix()), kind, payload) for name, kind, payload in entries]
        tgz = make_tar(str(tmp_path / 'bad.tar.gz'), entries)
        proj = make_project(tmp_path / 'app')
        rc, _out, err = tilpkg_run(proj, 'add', 'bad', archive_spec(tgz))
        assert rc == 1 and message in err, err
        assert not (tmp_path / 'evil.til').exists() and not (tmp_path / 'abs_evil.til').exists()
        assert not (proj / 'til_modules').exists() and manifest_of(proj)["dependencies"] == {}

    @pytest.mark.parametrize("entries, message", [
        ([('../evil.til', 'file', 'x')], "escape"),
        ([('/evil.til', 'file', 'x')], "absolute"),
        ([('pkg/..\\..\\evil.til', 'file', 'x')], "escape"),
        ([('pkg/a.til', 'file', 'x'), ('pkg/link', 'sym', '/etc/passwd')], "symbolic link"),
    ])
    def test_unsafe_zip_rejected(self, tmp_path, tilpkg_run, entries, message):
        archive = make_zip(str(tmp_path / 'bad.zip'), entries)
        proj = make_project(tmp_path / 'app')
        rc, _out, err = tilpkg_run(proj, 'add', 'bad', archive_spec(archive))
        assert rc == 1 and message in err, err
        assert not (tmp_path / 'evil.til').exists() and not (proj / 'til_modules').exists()

    def test_extract_archive_directly(self, tmp_path):
        tgz = make_tar(str(tmp_path / 'bad.tar.gz'), [('a/../../evil.til', 'file', 'x')])
        os.makedirs(tmp_path / 'out')
        with pytest.raises(tilpkg.PkgError, match="escape"):
            tilpkg.extract_archive(tgz, 'tar', str(tmp_path / 'out'), 'bad.tar.gz')
        assert not (tmp_path / 'evil.til').exists()
        corrupt = tmp_path / 'corrupt.zip'
        write(corrupt, 'not a zip')
        with pytest.raises(tilpkg.PkgError, match="cannot extract"):
            tilpkg.extract_archive(str(corrupt), 'zip', str(tmp_path / 'out'), 'corrupt.zip')


# ═══════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════

GEO_VERSIONS = ["1.0.0", "1.2.0", "1.2.5", "1.5.0", "2.0.0", "2.1.0-beta.1"]


@pytest.fixture
def registry(tmp_path):
    """reg/index.json: geo (6 versions, local directories) and needs_geo2 (requires geo ^2)."""
    reg = tmp_path / 'reg'
    for v in GEO_VERSIONS:
        make_pkg(reg / f'geo-{v}', 'geo', v, body=f'geo_version() -> str\n    return "{v}"\n')
    make_pkg(reg / 'needs_geo2', 'needs_geo2', '1.0.0', deps={'geo': '^2.0'})
    make_pkg(reg / 'needs_geo12', 'needs_geo12', '1.0.0', deps={'geo': '~1.2'})
    index = {"packages": {"geo": {v: f"./geo-{v}" for v in GEO_VERSIONS},
                          "needs_geo2": {"1.0.0": {"source": "./needs_geo2"}},
                          "needs_geo12": {"1.0.0": "./needs_geo12"}}}
    write(reg / 'index.json', json.dumps(index))
    return reg


def installed_geo(proj):
    return lock_of(proj)["packages"]["geo"]["version"]


class TestRegistry:
    @pytest.mark.parametrize("req, expected", [
        ("^1.2", "1.5.0"), ("~1.2", "1.2.5"), ("1.x", "1.5.0"), ("*", "2.0.0"), ("1.0.0", "1.0.0"),
        (">=1.2 <1.5", "1.2.5"), ("^2.1.0-beta", "2.1.0-beta.1"), ("geo@~1.0", "1.0.0"),
    ])
    def test_version_requirements(self, tmp_path, registry, tilpkg_run, monkeypatch, req, expected):
        monkeypatch.setenv('TIL_REGISTRY', str(registry / 'index.json'))
        proj = make_project(tmp_path / 'app')
        rc, _out, err = tilpkg_run(proj, 'add', 'geo', req)
        assert rc == 0, err
        assert installed_geo(proj) == expected
        assert f'"{expected}"' in read(proj / 'til_modules' / 'geo' / 'geo.til')
        entry = lock_of(proj)["packages"]["geo"]
        assert entry["spec"] == req and entry["resolved"] == f"../reg/geo-{expected}"

    def test_add_forms_and_manifest_registry(self, tmp_path, registry, tilpkg_run, monkeypatch):
        monkeypatch.delenv('TIL_REGISTRY', raising=False)
        proj = make_project(tmp_path / 'app', registry="../reg")         # directory -> index.json
        rc, out, err = tilpkg_run(proj, 'add', 'geo')
        assert rc == 0, err
        assert manifest_of(proj)["dependencies"] == {"geo": "^2.0.0"}    # latest release, not the beta
        assert "added geo = ^2.0.0" in out and installed_geo(proj) == "2.0.0"
        assert tilpkg_run(proj, 'add', 'geo@~1.2')[0] == 0
        assert manifest_of(proj)["dependencies"] == {"geo": "~1.2"} and installed_geo(proj) == "1.2.5"
        # TIL_REGISTRY wins over til.json "registry"
        other = tmp_path / 'other_reg'
        make_pkg(other / 'geo9', 'geo', '9.0.0')
        write(other / 'index.json', json.dumps({"geo": {"9.0.0": str(other / 'geo9')}}))     # bare form
        monkeypatch.setenv('TIL_REGISTRY', str(other / 'index.json'))
        assert tilpkg_run(proj, 'install', '--update')[0] == 1                # ~1.2 is not in that registry
        assert tilpkg_run(proj, 'add', 'geo', '^9')[0] == 0 and installed_geo(proj) == "9.0.0"

    def test_no_registry(self, tmp_path, tilpkg_run, monkeypatch):
        monkeypatch.delenv('TIL_REGISTRY', raising=False)
        proj = make_project(tmp_path / 'app')
        rc, _out, err = tilpkg_run(proj, 'add', 'geo', '^1.0')
        assert rc == 1 and "no registry is configured" in err
        rc, _out, err = tilpkg_run(proj, 'add', 'geo')
        assert rc == 1 and "needs a registry" in err
        assert manifest_of(proj)["dependencies"] == {}

    def test_unknown_package_and_unsatisfiable(self, tmp_path, registry, tilpkg_run, monkeypatch):
        monkeypatch.setenv('TIL_REGISTRY', str(registry / 'index.json'))
        proj = make_project(tmp_path / 'app')
        rc, _out, err = tilpkg_run(proj, 'add', 'nothere', '^1')
        assert rc == 1 and "not in the registry" in err
        rc, _out, err = tilpkg_run(proj, 'add', 'geo', '^3.0')
        assert rc == 1 and "no version of 'geo'" in err and "available: 1.0.0, 1.2.0" in err

    def test_constraints_from_several_packages(self, tmp_path, registry, tilpkg_run, monkeypatch):
        monkeypatch.setenv('TIL_REGISTRY', str(registry / 'index.json'))
        # the project picks ^1.0 -> 1.5.0 first; needs_geo12 requires ~1.2: re-resolved to 1.2.5
        proj = make_project(tmp_path / 'app', {"geo": "^1.0", "needs_geo12": "1.0.0"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 0, err
        assert installed_geo(proj) == "1.2.5"
        # ^1.0 and ^2.0 cannot both hold
        proj2 = make_project(tmp_path / 'app2', {"geo": "^1.0", "needs_geo2": "*"})
        rc, _out, err = tilpkg_run(proj2, 'install')
        assert rc == 1 and "no version of 'geo' in the registry satisfies all requirements" in err
        assert "app requires geo ^1.0" in err and "needs_geo2 requires geo ^2.0" in err
        # a version requirement is also satisfied by a package from another source with that version
        make_pkg(tmp_path / 'libs' / 'geo', 'geo', '2.3.0')
        proj3 = make_project(tmp_path / 'app3', {"geo": "../libs/geo", "needs_geo2": "*"})
        rc, _out, err = tilpkg_run(proj3, 'install')
        assert rc == 0, err
        assert lock_of(proj3)["packages"]["geo"]["resolved"] == "../libs/geo"

    def test_lock_keeps_the_registry_version(self, tmp_path, registry, tilpkg_run, monkeypatch):
        monkeypatch.setenv('TIL_REGISTRY', str(registry / 'index.json'))
        proj = make_project(tmp_path / 'app', {"geo": "^1.2"})
        assert tilpkg_run(proj, 'install')[0] == 0 and installed_geo(proj) == "1.5.0"
        make_pkg(registry / 'geo-1.9.0', 'geo', '1.9.0')
        index = json.loads(read(registry / 'index.json'))
        index["packages"]["geo"]["1.9.0"] = "./geo-1.9.0"
        write(registry / 'index.json', json.dumps(index))
        shutil.rmtree(proj / 'til_modules')
        assert tilpkg_run(proj, 'install')[0] == 0 and installed_geo(proj) == "1.5.0"
        monkeypatch.setenv('TIL_REGISTRY', str(tmp_path / 'gone.json'))       # the lock needs no registry
        shutil.rmtree(proj / 'til_modules')
        assert tilpkg_run(proj, 'install')[0] == 0
        monkeypatch.setenv('TIL_REGISTRY', str(registry / 'index.json'))
        rc, out, _err = tilpkg_run(proj, 'install', '--update')
        assert rc == 0 and "~ geo 1.5.0 -> 1.9.0" in out and installed_geo(proj) == "1.9.0"

    def test_registry_version_must_match_the_package(self, tmp_path, registry, tilpkg_run, monkeypatch):
        index = json.loads(read(registry / 'index.json'))
        index["packages"]["geo"]["3.0.0"] = "./geo-2.0.0"                        # wrong directory
        write(registry / 'index.json', json.dumps(index))
        monkeypatch.setenv('TIL_REGISTRY', str(registry / 'index.json'))
        proj = make_project(tmp_path / 'app', {"geo": "^3"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 1 and "lists geo 3.0.0" in err and "says version 2.0.0" in err

    @needs_git
    def test_registry_with_git_and_archive_sources(self, tmp_path, colors_repo, tilpkg_run, monkeypatch):
        tgz = make_tar(str(tmp_path / 'mathx-1.0.tar.gz'), MATHX)
        index = {"packages": {"colors": {"1.0.0": colors_repo.spec('v1.0.0'), "2.0.0": colors_repo.spec('v2.0.0')},
                              "mathx": {"1.0.0": archive_spec(tgz)}}}
        write(tmp_path / 'index.json', json.dumps(index))
        monkeypatch.setenv('TIL_REGISTRY', _uri(tmp_path / 'index.json'))      # file:// URL
        proj = make_project(tmp_path / 'app', {"colors": "^1", "mathx": "1.0.0"})
        rc, _out, err = tilpkg_run(proj, 'install')
        assert rc == 0, err
        lock = lock_of(proj)["packages"]
        assert lock["colors"]["resolved"] == colors_repo.spec(colors_repo.v1) and lock["colors"]["version"] == "1.0.0"
        assert lock["mathx"]["resolved"] == archive_spec(tgz)


# ═══════════════════════════════════════════════════════════════
# IMPORTS: programs that use packages (both VMs + native)
# ═══════════════════════════════════════════════════════════════

MAIN = '''import geo
import geo.shapes.square
from units import scaled
main()
    let r = Rect { w: 2, h: 3 }
    print(area(r))
    print(geo.area(r))
    print(square.side(4))
    print(scaled(5))
    print(f"{SCALE} {describe()}")
'''


@pytest.fixture
def app_with_packages(tmp_path):
    """app/ (til.json + src/main.til) depending on geo (main = src/geo.til) -> units."""
    libs = tmp_path / 'libs'
    make_pkg(libs / 'units', 'units', '0.3.0',
             body='const SCALE = 10\nscaled(x: int) -> int\n    return x * SCALE\n')
    make_pkg(libs / 'geo', 'geo', '1.0.0', deps={'units': '../units'}, main='src/geo.til',
             body='import units\nimport helpers\nstruct Rect\n    w: int\n    h: int\n'
                  'area(r: Rect) -> int\n    return scaled(r.w) * r.h\n')
    write(libs / 'geo' / 'src' / 'helpers.til', 'describe() -> str\n    return "geo helpers"\n')
    write(libs / 'geo' / 'src' / 'shapes' / 'square.til', 'side(a: int) -> int\n    return a * a\n')
    proj = make_project(tmp_path / 'app', {"geo": "../libs/geo"})
    write(proj / 'src' / 'main.til', MAIN)
    assert tilpkg.main(['-C', str(proj), 'install']) == 0
    return proj


EXPECTED = "60\n60\n16\n50\n10 geo helpers\n"


class TestImports:
    def test_runs_on_both_vms(self, app_with_packages):
        main = str(app_with_packages / 'src' / 'main.til')
        module = til.compile_to_bytecode(read(main), main)
        assert run_both(module.serialize()) == EXPECTED

    @needs_cc
    def test_runs_natively(self, app_with_packages, tmp_path):
        main = str(app_with_packages / 'src' / 'main.til')
        exe = str(tmp_path / ('native.exe' if sys.platform == 'win32' else 'native'))
        r = cli('build', main, '-o', exe)
        assert r.returncode == 0, r.stderr
        out = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        assert (out.returncode, out.stdout) == (0, EXPECTED)

    def test_cli_run_from_another_directory(self, app_with_packages, tmp_path):
        # the importing file's location decides, not the current directory
        r = cli('run', '--vm', str(app_with_packages / 'src' / 'main.til'), cwd=str(tmp_path))
        assert (r.returncode, r.stdout) == (0, EXPECTED), r.stderr

    @pytest.mark.parametrize("files, manifest, expected", [
        ({'geo.til': 'v() -> str\n    return "NAME.til"\n',
          'lib.til': 'v() -> str\n    return "lib.til"\n'}, None, "NAME.til"),
        ({'lib.til': 'v() -> str\n    return "lib.til"\n',
          'mod.til': 'v() -> str\n    return "mod.til"\n'}, None, "lib.til"),
        ({'mod.til': 'v() -> str\n    return "mod.til"\n'}, None, "mod.til"),
        ({'geo.til': 'v() -> str\n    return "NAME.til"\n',
          'src/entry.til': 'v() -> str\n    return "main"\n'}, {"name": "geo", "main": "src/entry.til"}, "main"),
    ])
    def test_entry_file(self, tmp_path, files, manifest, expected):
        pkg = tmp_path / 'proj' / 'til_modules' / 'geo'
        for rel, text in files.items():
            write(pkg / rel, text)
        if manifest:
            write(pkg / 'til.json', json.dumps(manifest))
        main = tmp_path / 'proj' / 'main.til'
        write(main, 'import geo\nprint(v())\n')
        module = til.compile_to_bytecode(read(main), str(main))
        assert run_both(module.serialize()) == expected + "\n"

    def test_missing_main_is_not_replaced(self, tmp_path):
        pkg = tmp_path / 'proj' / 'til_modules' / 'geo'
        write(pkg / 'geo.til', 'v() -> int\n    return 1\n')
        write(pkg / 'til.json', json.dumps({"main": "gone.til"}))
        assert til.find_package_module('geo', str(tmp_path / 'proj' / 'main.til')) is None
        write(pkg / 'til.json', json.dumps({"main": "../../escape.til"}))
        write(tmp_path / 'escape.til', 'v() -> int\n    return 1\n')
        assert til.find_package_module('geo', str(tmp_path / 'proj' / 'main.til')) is None

    def test_local_modules_come_first(self, tmp_path):
        write(tmp_path / 'proj' / 'til_modules' / 'geo' / 'geo.til', 'v() -> str\n    return "package"\n')
        write(tmp_path / 'proj' / 'geo.til', 'v() -> str\n    return "local"\n')
        main = tmp_path / 'proj' / 'main.til'
        write(main, 'import geo\nprint(v())\n')
        assert run_both(til.compile_to_bytecode(read(main), str(main)).serialize()) == "local\n"
        os.remove(tmp_path / 'proj' / 'geo.til')
        assert run_both(til.compile_to_bytecode(read(main), str(main)).serialize()) == "package\n"

    def test_nearest_til_modules_wins(self, tmp_path):
        write(tmp_path / 'til_modules' / 'geo' / 'geo.til', 'v() -> str\n    return "outer"\n')
        write(tmp_path / 'til_modules' / 'only_outer' / 'only_outer.til', 'w() -> str\n    return "w"\n')
        write(tmp_path / 'inner' / 'til_modules' / 'geo' / 'geo.til', 'v() -> str\n    return "inner"\n')
        main = tmp_path / 'inner' / 'a' / 'b' / 'main.til'
        write(main, 'import geo\nimport only_outer\nprint(v() + w())\n')
        assert run_file_both(read(main), str(main)) == "innerw\n"
        assert til.find_package_module('geo', str(tmp_path / 'x.til')) == \
            str(tmp_path / 'til_modules' / 'geo' / 'geo.til')

    def test_submodules(self, tmp_path):
        pkg = tmp_path / 'til_modules' / 'kit'
        write(pkg / 'kit.til', 'k() -> int\n    return 1\n')
        write(pkg / 'text' / 'mod.til', 't() -> int\n    return 2\n')
        write(pkg / 'deep' / 'er.til', 'd() -> int\n    return 3\n')
        main = tmp_path / 'main.til'
        write(main, 'import kit.text\nimport kit.deep.er\nprint(text.t() + er.d())\n')
        assert run_file_both(read(main), str(main)) == "5\n"
        assert til.find_package_module('kit.nothing', str(main)) is None

    def test_not_installed_hint(self, tmp_path):
        proj = make_project(tmp_path / 'app', {"geo": "../libs/geo"})
        write(proj / 'main.til', 'import geo\nprint(1)\n')
        r = cli('run', '--vm', str(proj / 'main.til'))
        assert "hint: 'geo' is a dependency in" in r.stderr and "til pkg install" in r.stderr
        assert "Could not find module 'geo'" in r.stderr


def run_file_both(source, filename):
    """Compile with a real file name (so imports resolve) and run on both VMs."""
    return run_both(til.compile_to_bytecode(source, filename).serialize())


# ═══════════════════════════════════════════════════════════════
# CLI ROUTING (`til pkg` through til.py)
# ═══════════════════════════════════════════════════════════════

class TestCli:
    def test_routed_through_til(self, tmp_path, libs):
        proj = tmp_path / 'app'
        os.makedirs(proj)
        assert cli('pkg', 'init', 'app', cwd=str(proj)).returncode == 0
        r = cli('pkg', 'add', 'geo', '../libs/geo', cwd=str(proj))
        assert r.returncode == 0 and "+ geo 1.2.0" in r.stdout, r.stderr
        r = cli('pkg', 'list', cwd=str(proj))
        assert r.returncode == 0 and "`-- geo 1.2.0  ../libs/geo" in r.stdout
        r = cli('pkg', '-C', str(proj), 'verify')
        assert r.returncode == 0 and "verified 2 packages" in r.stdout

    def test_help_and_usage_errors(self, tmp_path):
        r = cli('pkg', 'help')
        assert r.returncode == 0 and "git+URL[#REV]" in r.stdout and "install" in r.stdout
        assert cli('pkg').returncode == 0
        r = cli('pkg', 'frobnicate')
        assert r.returncode == 2 and "invalid choice" in r.stderr
        r = cli('pkg', 'install', 'geo', cwd=str(tmp_path))
        assert r.returncode == 1
        assert tilpkg.main(['-C', str(tmp_path / 'missing'), 'list']) == 1
