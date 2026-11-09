#!/usr/bin/env python3
"""
til pkg - the TIL package manager (TIL 3.1).

    til pkg init [NAME]                        create til.json
    til pkg add NAME [SPEC]                    add a dependency and install it
    til pkg install [--update [NAME...]] [--frozen]
    til pkg remove NAME                        drop a dependency
    til pkg list                               dependency tree
    til pkg verify                             re-hash til_modules/ against til.lock

Dependency specs (til.json "dependencies": {"NAME": SPEC}):

    ./dir  ../dir  /abs/dir             local directory (relative to that til.json)
    git+URL[#REV]                       git repository; REV = tag, branch or commit
    https://host/p.tar.gz#sha256=HEX    archive: .tar.gz, .tgz or .zip (also file://)
    ^1.2  ~1.2.3  1.x  *  1.2.3  >=1 <2 version requirement, resolved by the registry
                                        (TIL_REGISTRY or til.json "registry")

Packages are installed flat into til_modules/NAME/ next to til.json; til.lock
records for each one the spec, the resolved source (git commits as SHAs), the
version and the sha256 of the installed files ("tree hash", see tree_hash()).
The compiler finds them with `import NAME` (til.py: find_package_module).

User guide: docs/TIL_PACKAGES.md. til.py dispatches `til pkg ...` here
(EXTRA_COMMANDS); `import til` returns the running compiler module.
"""

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import til

MANIFEST = 'til.json'
LOCKFILE = 'til.lock'
MODULES = getattr(til, 'PACKAGE_DIR', 'til_modules')
LOCK_VERSION = 1

# never part of a package (at any depth): version control data, installed packages
SKIPPED_NAMES = frozenset(('.git', '.hg', '.svn', MODULES))

# explicit limits (exceeding one is an error, never a silent truncation)
MAX_PACKAGE_BYTES = 256 * 1024 * 1024
MAX_PACKAGE_FILES = 100000
MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_RESOLVE_PASSES = 50
DOWNLOAD_TIMEOUT = 60

_WINDOWS_RESERVED = frozenset(['con', 'prn', 'aux', 'nul'] + [f'com{i}' for i in range(1, 10)] +
                              [f'lpt{i}' for i in range(1, 10)])


def _(text: str) -> str:
    """A diagnostic in the language of TIL's messages (--lang / TIL_LANG,
    docs/TIL_I18N.md); what `til pkg` prints on standard output stays English."""
    return til.localize(text)


class PkgError(Exception):
    """A user-facing error: printed as `til pkg: error: ...`, exit code 1."""


# ═══════════════════════════════════════════════════════════════
# NAMES AND VERSIONS
# ═══════════════════════════════════════════════════════════════

def check_name(name: Any, where: str = "") -> str:
    """A package name is what programs `import`: a TIL identifier (letters of
    any script, digits, '_'), so it can never contain '/', '\\' or '..'."""
    prefix = f"{where}: " if where else ""
    if not isinstance(name, str) or not name:
        raise PkgError(f"{prefix}package names must be non-empty strings")
    ok = ((name[0].isalpha() or name[0] == '_') and all(c.isalnum() or c == '_' for c in name)
          and unicodedata.normalize('NFC', name) == name and len(name) <= 64)
    if not ok:
        raise PkgError(f"{prefix}invalid package name '{name}': a package name is the name you `import`, "
                       f"so it must be an identifier (letters, digits, '_'; at most 64 characters) - "
                       f"no '/', '\\', '.', '-' or spaces")
    if name in getattr(til, 'KEYWORDS', {}):
        raise PkgError(f"{prefix}invalid package name '{name}': it is a TIL keyword")
    if name.casefold() == MODULES.casefold():
        raise PkgError(f"{prefix}invalid package name '{name}': '{MODULES}' is reserved (it is the directory "
                       f"packages are installed in)")
    if name.lower() in _WINDOWS_RESERVED:
        raise PkgError(f"{prefix}invalid package name '{name}': reserved file name on Windows")
    return name


_SEMVER_RE = re.compile(r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)'
                        r'(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?'
                        r'(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$')


class Version:
    """A semantic version MAJOR.MINOR.PATCH[-pre][+build] (semver.org 2.0
    precedence; build metadata is ignored when comparing)."""

    __slots__ = ('major', 'minor', 'patch', 'pre', 'text')

    def __init__(self, major: int, minor: int, patch: int, pre: Tuple[str, ...] = (), text: str = ""):
        self.major, self.minor, self.patch, self.pre = major, minor, patch, pre
        self.text = text or f"{major}.{minor}.{patch}" + ("-" + ".".join(pre) if pre else "")

    @classmethod
    def parse(cls, text: Any, where: str = "") -> 'Version':
        m = _SEMVER_RE.match(text.strip()) if isinstance(text, str) else None
        if not m:
            prefix = f"{where}: " if where else ""
            raise PkgError(f"{prefix}invalid version {text!r} (expected MAJOR.MINOR.PATCH, e.g. \"1.2.0\")")
        pre = tuple(m.group(4).split('.')) if m.group(4) else ()
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)), pre, text.strip())

    def key(self) -> tuple:
        if not self.pre:
            pre_key: tuple = (1,)
        else:
            pre_key = (0,) + tuple((0, int(p), '') if p.isdigit() else (1, 0, p) for p in self.pre)
        return (self.major, self.minor, self.patch, pre_key)

    def __eq__(self, other):
        return isinstance(other, Version) and self.key() == other.key()

    def __lt__(self, other: 'Version') -> bool:
        return self.key() < other.key()

    def __hash__(self):
        return hash(self.key())

    def __str__(self):
        return self.text

    __repr__ = __str__


_PARTIAL_RE = re.compile(r'^v?(\d+|[xX*])(?:\.(\d+|[xX*]))?(?:\.(\d+|[xX*]))?'
                         r'(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z.-]+)?$')


def _comparators(token: str, whole: str) -> List[Tuple[str, Version]]:
    """One comparator token of a requirement -> primitive (op, version) bounds."""
    bad = PkgError(f"invalid version requirement '{whole}'")
    m = re.match(r'^(>=|<=|>|<|=|\^|~)?(.*)$', token)
    op, rest = m.group(1) or '', m.group(2)
    if rest in ('', '*', 'x', 'X'):
        if op in ('', '=', '>=', '<=', '^', '~'):
            return []                                   # any version
        raise bad
    pm = _PARTIAL_RE.match(rest)
    if not pm:
        raise bad
    nums: List[int] = []
    for part in pm.group(1, 2, 3):
        if part is None or part in ('x', 'X', '*'):
            break
        nums.append(int(part))
    later = [p for p in pm.group(1, 2, 3)[len(nums):] if p is not None]
    if any(p not in ('x', 'X', '*') for p in later):
        raise bad                                       # 1.x.3
    pre = tuple(pm.group(4).split('.')) if pm.group(4) else ()
    if pre and len(nums) < 3:
        raise bad
    n = len(nums)
    x, y, z = (nums + [0, 0, 0])[:3]
    if n == 0:
        return [] if op in ('', '=', '>=', '<=', '^', '~') else _raise(bad)
    low = Version(x, y, z, pre)
    if n == 1:
        nxt = Version(x + 1, 0, 0)
    elif n == 2:
        nxt = Version(x, y + 1, 0)
    else:
        nxt = None
    if op in ('', '='):
        return [('=', low)] if n == 3 else [('>=', low), ('<', nxt)]
    if op == '^':
        if x > 0 or n == 1:
            upper = Version(x + 1, 0, 0)
        elif y > 0 or n == 2:
            upper = Version(0, y + 1, 0)
        else:
            upper = Version(0, 0, z + 1)
        return [('>=', low), ('<', upper)]
    if op == '~':
        return [('>=', low), ('<', Version(x + 1, 0, 0) if n == 1 else Version(x, y + 1, 0))]
    if op == '>=':
        return [('>=', low)]
    if op == '<':
        return [('<', low)]
    if op == '>':
        return [('>', low)] if n == 3 else [('>=', nxt)]
    # '<='
    return [('<=', low)] if n == 3 else [('<', nxt)]


def _raise(err: Exception):
    raise err


class Requirement:
    """A version requirement (npm-style): `^1.2`, `~1.2.3`, `1.x`, `1.2`, `*`,
    exact `1.2.3`, comparators `>=1.2 <2` (space or comma = and), `a || b`.
    Pre-releases only match when a comparator names a pre-release of the
    same MAJOR.MINOR.PATCH. Hyphen ranges are not supported (an error)."""

    def __init__(self, text: str):
        self.text = text.strip() or '*'
        self.alternatives: List[List[Tuple[str, Version]]] = []
        for alt in self.text.split('||'):
            tokens = [t for t in re.split(r'[\s,]+', alt.strip()) if t]
            merged: List[str] = []
            i = 0
            while i < len(tokens):              # ">= 1.2" -> ">=1.2"
                if tokens[i] in ('>=', '<=', '>', '<', '=', '^', '~') and i + 1 < len(tokens):
                    merged.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            comps: List[Tuple[str, Version]] = []
            for token in merged or ['*']:
                comps.extend(_comparators(token, self.text))
            self.alternatives.append(comps)

    def matches(self, v: Version) -> bool:
        k = v.key()
        for comps in self.alternatives:
            ok = True
            for op, bound in comps:
                b = bound.key()
                if not ((op == '=' and k == b) or (op == '>=' and k >= b) or (op == '>' and k > b)
                        or (op == '<' and k < b) or (op == '<=' and k <= b)):
                    ok = False
                    break
            if not ok:
                continue
            if not v.pre or any(b.pre and (b.major, b.minor, b.patch) == (v.major, v.minor, v.patch)
                                for _op, b in comps):
                return True
        return False

    def __str__(self):
        return self.text


# ═══════════════════════════════════════════════════════════════
# DEPENDENCY SPECS
# ═══════════════════════════════════════════════════════════════

class Spec:
    """A parsed dependency spec. kind: 'path' | 'git' | 'archive' | 'registry'."""

    def __init__(self, kind: str, raw: str, path: Optional[str] = None, url: Optional[str] = None,
                 rev: Optional[str] = None, sha256: Optional[str] = None, fmt: Optional[str] = None,
                 req: Optional[Requirement] = None, anchor: str = 'project', written: Optional[str] = None):
        self.kind = kind
        self.raw = raw              # as written in til.json / til.lock
        # path: absolute directory; git: absolute path of a local repository. The real path (symbolic
        # links resolved): it identifies the directory. `written`: the same directory as it was written
        # (absolute, normalized, links kept): what til.lock records and relative paths start from.
        self.path = path
        self.written = written or path
        self.url = url              # git: URL (or the absolute local path); archive: URL without '#...'
        self.rev = rev              # git: tag / branch / commit (None = the remote HEAD)
        self.sha256 = sha256        # archive: expected sha256 of the archive file
        self.fmt = fmt              # archive: 'tar' (.tar.gz/.tgz) | 'zip'
        self.req = req              # registry: version requirement
        # local paths (path, local git repository): how til.lock writes them - 'project' (relative to
        # the project root), 'absolute' or 'home' (~/...) - so that a lock stays valid wherever the
        # project is checked out. A relative path inherits the anchor of the til.json declaring it.
        self.anchor = anchor


_GIT_REV_RE = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_./+@-]*$')
_SHA_RE = re.compile(r'^[0-9a-f]{40}$|^[0-9a-f]{64}$')
_HEX64_RE = re.compile(r'^[0-9a-f]{64}$')


def _looks_like_path(text: str) -> bool:
    return (text in ('.', '..', '~') or text.startswith(('.', '/', '\\', '~/', '~\\'))
            or '/' in text or '\\' in text or re.match(r'^[A-Za-z]:', text) is not None)


def _path_anchor(text: str, inherited: str) -> str:
    if text == '~' or text.startswith(('~/', '~\\')):
        return 'home'
    if text.startswith('~') or os.path.isabs(text) or text.startswith(('/', '\\')) \
            or re.match(r'^[A-Za-z]:', text):
        return 'absolute'
    return inherited


def parse_spec(raw: Any, base_dir: Optional[str], name: str, where: str, anchor: str = 'project') -> Spec:
    """Parse a dependency spec. `base_dir` = directory that relative paths are
    relative to (the directory of the declaring til.json); None when the
    declaring package is not a local directory: path dependencies are then an
    error (they would point into the installer's file system). `anchor`: how
    the declaring package's own path was written (see Spec.anchor)."""
    if not isinstance(raw, str) or not raw.strip():
        raise PkgError(f"{where}: dependency '{name}' needs a spec string (a path, git+URL, archive URL "
                       f"or version requirement)")
    text = raw.strip()
    if any(ord(c) < 32 for c in text):
        raise PkgError(f"{where}: dependency '{name}': control characters in the spec")
    if text.startswith('git+'):
        return _parse_git(raw, text[4:], base_dir, name, where, anchor)
    lower = text.lower()
    if lower.startswith(('https://', 'http://', 'file://')):
        return _parse_archive(raw, text, name, where)
    if _looks_like_path(text):
        if base_dir is None:
            raise PkgError(f"{where}: dependency '{name}' = '{text}' is a local path; only your project and "
                           f"packages that are local directories may have path dependencies")
        written = os.path.abspath(os.path.join(base_dir, os.path.expanduser(text)))
        return Spec('path', raw, path=os.path.realpath(written), anchor=_path_anchor(text, anchor),
                    written=written)
    req_text = text
    if '@' in text:
        alias, _, req_text = text.partition('@')
        if alias != name:
            raise PkgError(f"{where}: dependency '{name}' = '{text}': the registry name must be the "
                           f"dependency name (installing a package under another name is not supported)")
    try:
        req = Requirement(req_text)
    except PkgError:
        hint = ""
        if base_dir is not None and os.path.isdir(os.path.join(base_dir, text)):
            hint = f" (for the directory write ./{text})"
        raise PkgError(f"{where}: dependency '{name}' = '{text}' is not a path, git+URL, archive URL or "
                       f"version requirement{hint}") from None
    return Spec('registry', raw, req=req)


def _is_local_git_path(url: str) -> bool:
    """git+URL: a local repository path (not scheme://... and not scp-like host:path)."""
    return '://' not in url and (re.match(r'^[A-Za-z]:[\\/]', url) is not None
                                 or url.startswith(('/', '\\', '.', '~')) or ':' not in url)


def _parse_git(raw: str, rest: str, base_dir: Optional[str], name: str, where: str,
               anchor: str = 'project') -> Spec:
    url, sep, rev = rest.partition('#')
    bad = f"{where}: dependency '{name}' = '{raw}'"
    if not url or url.startswith('-') or '::' in url or any(c.isspace() for c in url):
        raise PkgError(f"{bad}: invalid git URL")
    if sep and (not _GIT_REV_RE.match(rev) or '..' in rev or rev.endswith(('.', '/', '.lock'))):
        raise PkgError(f"{bad}: invalid git revision '{rev}'")
    path = written = None
    if '://' in url:
        scheme = url.split('://', 1)[0].lower()
        if scheme not in ('https', 'http', 'ssh', 'git', 'file'):
            raise PkgError(f"{bad}: unsupported git URL scheme '{scheme}' (use https, ssh, git or file)")
    elif _is_local_git_path(url):
        if base_dir is None:
            raise PkgError(f"{bad}: a git repository given as a local path is only allowed in your project "
                           f"and in packages that are local directories (use git+file:///...)")
        anchor = _path_anchor(url, anchor)
        written = os.path.abspath(os.path.join(base_dir, os.path.expanduser(url)))
        path = os.path.realpath(written)
        url = path
    # else: scp-like [user@]host:path (ssh)
    return Spec('git', raw, url=url, rev=rev or None, path=path, anchor=anchor, written=written)


def _parse_archive(raw: str, text: str, name: str, where: str) -> Spec:
    url, sep, frag = text.partition('#')
    bad = f"{where}: dependency '{name}' = '{raw}'"
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme == 'http':
        raise PkgError(f"{bad}: archives must be fetched over https:// (or file:// for a local archive)")
    low = parsed.path.lower()
    fmt = 'tar' if low.endswith(('.tar.gz', '.tgz')) else 'zip' if low.endswith('.zip') else None
    if fmt is None:
        hint = (" (a local directory is a plain path; a git repository is git+file:///...)"
                if scheme == 'file' else "")
        raise PkgError(f"{bad}: an archive URL must end in .tar.gz, .tgz or .zip{hint}")
    if scheme == 'file' and parsed.netloc not in ('', 'localhost'):
        raise PkgError(f"{bad}: file:// URLs must name a local file (file:///path/to/archive)")
    sha = None
    if sep:
        m = re.match(r'^sha256=([0-9a-fA-F]{64})$', frag)
        if not m:
            raise PkgError(f"{bad}: unsupported URL fragment '#{frag}' (expected #sha256=<64 hex digits>)")
        sha = m.group(1).lower()
    return Spec('archive', raw, url=url, sha256=sha, fmt=fmt)


def display_path(path: str, root: str) -> str:
    """A path as written into til.lock / shown to users: relative to the
    project root when possible ('./x', '../x'), always with '/'."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:              # another drive (Windows)
        return path.replace(os.sep, '/')
    rel = rel.replace(os.sep, '/')
    if not os.path.isabs(rel) and not rel.startswith('.'):
        rel = './' + rel
    return rel


def lock_source_path(path: str, root: str, anchor: str) -> str:
    """A local directory as til.lock records it ("resolved"), in the form it was
    declared in, so the lock means the same directory wherever the project is
    checked out: relative to the project root, an absolute path, or ~/...
    `path` is the directory as written (Spec.written): symbolic links on the
    way (~/libs -> /data/libs) are kept, never replaced by their targets."""
    if anchor == 'home':
        home = os.path.abspath(os.path.expanduser('~'))
        try:
            rel = os.path.relpath(path, home)
        except ValueError:
            rel = os.pardir
        if rel == os.curdir:
            return '~'
        if rel != os.pardir and not rel.startswith(os.pardir + os.sep) and not os.path.isabs(rel):
            return '~/' + rel.replace(os.sep, '/')
        anchor = 'absolute'
    if anchor == 'absolute':
        return path.replace(os.sep, '/')
    return display_path(path, root)


# ═══════════════════════════════════════════════════════════════
# FILES: JSON, MANIFESTS, LOCK FILES
# ═══════════════════════════════════════════════════════════════

def read_json(path: str, what: str) -> Any:
    try:
        with open(path, encoding='utf-8-sig') as f:        # tolerate a BOM (Windows editors)
            return json.load(f)
    except ValueError as e:
        raise PkgError(f"{path}: invalid {what} (not valid JSON: {e})") from None
    except RecursionError:              # json nests a Python call per [ or {
        raise PkgError(f"{path}: invalid {what} (JSON nested too deeply)") from None
    except OSError as e:
        raise PkgError(f"cannot read {path}: {e.strerror or e}") from None


def json_text(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _new_name(directory: str, prefix: str) -> str:
    return os.path.join(directory, prefix + os.urandom(6).hex())


def make_dir(parent: str, prefix: str) -> str:
    """A new directory with a unique name in `parent`. Unlike tempfile.mkdtemp
    (always 0700) its permissions follow the umask, like any directory the
    user creates: installed packages must stay readable for other users (a
    Docker image installed as root and run as another user)."""
    while True:
        path = _new_name(parent, prefix)
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            continue


def write_text_atomic(path: str, text: str):
    """Replace `path` atomically. A new file gets the usual permissions (0666
    minus the umask); an existing file keeps its permissions."""
    d = os.path.dirname(os.path.abspath(path))
    try:
        mode: Optional[int] = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        mode = None
    while True:
        tmp = _new_name(d, '.' + os.path.basename(path) + '.')
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o666)
            break
        except FileExistsError:
            continue
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def validate_manifest(data: Any, where: str) -> Dict[str, Any]:
    """Check a til.json; `where` names it in error messages."""
    if not isinstance(data, dict):
        raise PkgError(f"{where}: must contain a JSON object")
    if 'name' in data:
        check_name(data['name'], f"{where}: \"name\"")
    if 'version' in data:
        Version.parse(data['version'], f"{where}: \"version\"")
    deps = data.get('dependencies', {})
    if not isinstance(deps, dict):
        raise PkgError(f"{where}: \"dependencies\" must be an object {{\"name\": \"spec\"}}")
    for dep, spec in deps.items():
        check_name(dep, f"{where}: dependency")
        if not isinstance(spec, str):
            raise PkgError(f"{where}: dependency '{dep}': the spec must be a string")
    if 'main' in data:
        main = data['main']
        parts = main.replace('\\', '/').split('/') if isinstance(main, str) else []
        if (not isinstance(main, str) or not main.endswith('.til') or os.path.isabs(main)
                or main.startswith(('/', '\\')) or re.match(r'^[A-Za-z]:', main) or '..' in parts):
            raise PkgError(f"{where}: \"main\" must be a relative path to a .til file inside the package")
    if 'registry' in data and not isinstance(data['registry'], str):
        raise PkgError(f"{where}: \"registry\" must be a string (a path or URL of a registry index)")
    return data


def load_manifest(directory: str, where: Optional[str] = None,
                  required: bool = True) -> Optional[Dict[str, Any]]:
    path = os.path.join(directory, MANIFEST)
    if not os.path.isfile(path):
        if required:
            raise PkgError(f"{path} not found")
        return None
    return validate_manifest(read_json(path, MANIFEST), where or path)


def find_root(start: str) -> str:
    """The project root: the nearest directory with a til.json, from `start` up."""
    d = os.path.realpath(start)
    while True:
        if os.path.isfile(os.path.join(d, MANIFEST)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise PkgError(f"no {MANIFEST} in {os.path.realpath(start)} or any parent directory "
                           f"(create one with `til pkg init`)")
        d = parent


def read_lock(path: str) -> Dict[str, Any]:
    data = read_json(path, LOCKFILE)
    if not isinstance(data, dict) or not isinstance(data.get('packages'), dict):
        raise PkgError(f"{path} is not a til.lock file (delete it, or run `til pkg install --update`)")
    if data.get('lockVersion') != LOCK_VERSION:
        raise PkgError(f"{path}: unsupported lockVersion {data.get('lockVersion')!r} "
                       f"(this til writes version {LOCK_VERSION})")
    for name, e in data['packages'].items():
        check_name(name, path)
        ok = (isinstance(e, dict) and isinstance(e.get('spec'), str) and isinstance(e.get('resolved'), str)
              and isinstance(e.get('sha256'), str) and _HEX64_RE.match(e['sha256']) is not None
              and (e.get('version') is None or isinstance(e.get('version'), str))
              and isinstance(e.get('dependencies', {}), dict)
              and all(isinstance(k, str) and isinstance(v, str) for k, v in e.get('dependencies', {}).items()))
        if not ok:
            raise PkgError(f"{path}: malformed entry for package '{name}' "
                           f"(delete til.lock or run `til pkg install --update`)")
    return data


def lock_data(packages: Dict[str, 'Package']) -> Dict[str, Any]:
    """til.lock contents: packages sorted by name, fields in a fixed order."""
    entries: Dict[str, Any] = {}
    for name in sorted(packages):
        p = packages[name]
        entries[name] = {
            "spec": p.spec.raw,
            "resolved": p.resolved,
            "version": p.version,
            "sha256": p.sha256,
            "dependencies": {d: p.deps[d] for d in sorted(p.deps)},
        }
    return {"lockVersion": LOCK_VERSION, "packages": entries}


# ═══════════════════════════════════════════════════════════════
# PACKAGE TREES: FILE LIST, TREE HASH, COPYING
# ═══════════════════════════════════════════════════════════════

def _path_key(rel: str) -> bytes:
    return rel.encode('utf-8', 'surrogateescape')


def package_files(root: str, installed: bool = False) -> List[str]:
    """The files of the package in `root`: '/'-separated relative paths, sorted
    by their UTF-8 bytes. .git, .hg, .svn and til_modules are skipped at any
    depth; symbolic links and special files are refused (a package is plain
    files and directories, so it can never point outside itself).
    installed=True: `root` is an installed copy in til_modules/, which never
    contains those directories - finding one is an error (it would be invisible
    to the hash, and a nested til_modules/ could shadow packages)."""
    out: List[str] = []

    def listing(d: str) -> List[os.DirEntry]:
        try:
            with os.scandir(d) as it:
                return sorted(it, key=lambda e: e.name)
        except OSError as e:
            raise PkgError(f"cannot read {d}: {e.strerror or e}") from None

    # depth first in name order, with an explicit stack instead of recursion: no
    # directory depth (the OS allows thousands of levels) can exhaust Python's stack
    stack: List[Tuple[str, List[os.DirEntry], int]] = [('', listing(root), 0)]
    while stack:
        prefix, entries, i = stack[-1]
        if i == len(entries):
            stack.pop()
            continue
        stack[-1] = (prefix, entries, i + 1)
        entry = entries[i]
        rel = prefix + entry.name
        if entry.name in SKIPPED_NAMES:
            if installed:
                raise PkgError(f"{root}: '{rel}' was added after installation "
                               f"(til pkg never installs {entry.name} directories)")
            continue
        if '\n' in entry.name or '\r' in entry.name:
            raise PkgError(f"{root}: file name with a line break: {rel!r}")
        if entry.is_symlink() or (os.name == 'nt' and _is_link(entry.path)):
            raise PkgError(f"{root}: '{rel}' is a symbolic link (packages may only contain "
                           f"plain files and directories)")
        if entry.is_dir(follow_symlinks=False):
            stack.append((rel + '/', listing(entry.path), 0))
        elif entry.is_file(follow_symlinks=False):
            out.append(rel)
            if len(out) > MAX_PACKAGE_FILES:
                raise PkgError(f"{root}: more than {MAX_PACKAGE_FILES} files in one package")
        else:
            raise PkgError(f"{root}: '{rel}' is not a regular file or directory")
    out.sort(key=_path_key)
    return out


def make_dirs(path: str):
    """os.makedirs(path, exist_ok=True) without recursion: os.makedirs calls
    itself once per missing parent, so a tree ~1000 levels deep (valid on
    Linux and macOS) exhausted Python's recursion limit."""
    missing: List[str] = []
    p = path
    while not os.path.isdir(p):
        missing.append(p)
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    for d in reversed(missing):
        try:
            os.mkdir(d)
        except FileExistsError:
            if not os.path.isdir(d):
                raise


def _hash_line(file_hex: str, rel: str) -> bytes:
    return f"{file_hex}  {rel}\n".encode('utf-8', 'surrogateescape')


def tree_hash(root: str, installed: bool = False) -> str:
    """sha256 of a package tree: the sha256 of the text made of one line
    '<sha256 of the file contents, hex>  <relative path>\\n' per file (the
    format of `sha256sum`), files sorted by the UTF-8 bytes of their '/'-separated
    paths. Only paths and contents count (no timestamps, permissions or empty
    directories), so the hash is the same on every OS. `installed`: see
    package_files()."""
    h = hashlib.sha256()
    for rel in package_files(root, installed):
        fh = hashlib.sha256()
        with open(os.path.join(root, *rel.split('/')), 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                fh.update(chunk)
        h.update(_hash_line(fh.hexdigest(), rel))
    return h.hexdigest()


def copy_package(src: str, dest: str) -> str:
    """Copy the package files of `src` into the empty directory `dest`;
    returns the tree hash of what was copied."""
    h = hashlib.sha256()
    total = 0
    for rel in package_files(src):
        target = os.path.join(dest, *rel.split('/'))
        make_dirs(os.path.dirname(target))
        fh = hashlib.sha256()
        try:
            with open(os.path.join(src, *rel.split('/')), 'rb') as fi, open(target, 'xb') as fo:
                for chunk in iter(lambda: fi.read(1 << 20), b''):
                    total += len(chunk)
                    if total > MAX_PACKAGE_BYTES:
                        raise PkgError(f"{src}: package larger than {size_text(MAX_PACKAGE_BYTES)}")
                    fh.update(chunk)
                    fo.write(chunk)
        except FileExistsError:
            raise PkgError(f"{src}: two files differ only in case ('{rel}'): they cannot be installed "
                           f"on this file system") from None
        h.update(_hash_line(fh.hexdigest(), rel))
    return h.hexdigest()


def size_text(n: int) -> str:
    return f"{n >> 20} MB" if n >= 1 << 20 else f"{n} bytes"


_NAME_SURROGATE = 0x20000000        # Windows: reparse tags of links (symlinks, junctions, mount points)


def _is_link(path: str) -> bool:
    """A symbolic link, or (Windows) a junction or other name-surrogate
    reparse point. til pkg removes such entries themselves, never what they
    point to."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, 'st_reparse_tag', 0) & _NAME_SURROGATE)


def _unlink_link(path: str):
    try:
        os.unlink(path)
    except OSError:
        if os.name != 'nt':
            raise
        os.rmdir(path)                  # a directory symlink or junction: removes the link, not its target


def _unlink_file(path: str):
    try:
        os.unlink(path)
    except PermissionError:
        if os.name != 'nt':
            raise
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)       # read-only file (Windows)
        os.unlink(path)


def rmtree(path: str):
    """Remove a directory tree (or a single file or link). Links - symbolic
    links and Windows junctions, at the top or anywhere inside - are removed
    themselves and never followed, so nothing outside `path` is touched."""
    if _is_link(path):
        _unlink_link(path)
        return
    if not os.path.isdir(path):
        if os.path.lexists(path):
            _unlink_file(path)
        return
    stack = [(path, False)]             # iterative: no recursion limit on deep trees
    while stack:
        d, emptied = stack.pop()
        if emptied:
            os.rmdir(d)
            continue
        stack.append((d, True))
        with os.scandir(d) as it:
            entries = list(it)
        for entry in entries:
            if _is_link(entry.path):
                _unlink_link(entry.path)
            elif entry.is_dir(follow_symlinks=False):
                stack.append((entry.path, False))
            else:
                _unlink_file(entry.path)


@contextlib.contextmanager
def work_directory():
    """A private temporary directory for one command, like
    tempfile.TemporaryDirectory - but removed with rmtree(): shutil.rmtree
    recurses once per directory level before Python 3.13, so a deep
    package tree fetched into it raised RecursionError when it was cleaned up."""
    d = tempfile.mkdtemp(prefix='tilpkg-')
    try:
        yield d
    finally:
        try:
            rmtree(d)
        except OSError:
            pass                        # only scratch files are left behind; never hide the real outcome


def modules_dir(root: str) -> str:
    """<root>/til_modules, which til pkg manages completely (it deletes what it
    did not install there). It must be a real directory inside the project -
    never a symbolic link or junction (a til_modules link committed to a
    repository would otherwise make `til pkg install` delete the files it
    points to) - or not exist yet."""
    modules = os.path.join(root, MODULES)
    if _is_link(modules):
        try:
            target = os.readlink(modules)
        except OSError:
            target = '?'
        raise PkgError(f"{modules} is a symbolic link (-> {target}); til pkg manages {MODULES}/ itself and "
                       f"would delete files there, so it refuses to work through a link: remove the link "
                       f"(not the directory it points to) and run `til pkg install`")
    if os.path.lexists(modules):
        if not os.path.isdir(modules):
            raise PkgError(f"{modules} is not a directory: remove it and run `til pkg install`")
        real = os.path.realpath(modules)
        if os.path.normcase(real) != os.path.normcase(os.path.join(os.path.realpath(root), MODULES)):
            raise PkgError(f"{modules} resolves to {real}, outside the project; til pkg refuses to manage it")
    return modules


# ═══════════════════════════════════════════════════════════════
# ARCHIVES: DOWNLOAD, VERIFY, SAFE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def file_url_path(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc not in ('', 'localhost'):
        raise PkgError(f"{url}: file:// URLs must name a local file")
    return urllib.request.url2pathname(parsed.path)


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to https:// URLs. urllib's default handler also
    follows https -> http (and ftp), which would fetch the registry index -
    which has no hash, its integrity rests on TLS - over plain HTTP."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme.lower() != 'https':
            fp.close()
            raise PkgError(f"cannot download {req.full_url}: it redirects to {newurl} "
                           f"(only https:// is allowed)")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _urlopen(request, timeout=None):
    """urllib.request.urlopen(), but every redirect must stay on https://."""
    return urllib.request.build_opener(_HttpsOnlyRedirects()).open(request, timeout=timeout)


def download(url: str, dest: str, limit: Optional[int] = None) -> str:
    """Fetch `url` (https:// or file://) into the file `dest`; returns its sha256.
    At most `limit` bytes (default MAX_DOWNLOAD_BYTES)."""
    limit = MAX_DOWNLOAD_BYTES if limit is None else limit
    h = hashlib.sha256()
    total = 0

    def pump(src):
        nonlocal total
        with open(dest, 'wb') as out:
            for chunk in iter(lambda: src.read(1 << 20), b''):
                total += len(chunk)
                if total > limit:
                    raise PkgError(f"{url}: download larger than {size_text(limit)}")
                h.update(chunk)
                out.write(chunk)

    if url.lower().startswith('file://'):
        path = file_url_path(url)
        try:
            with open(path, 'rb') as src:
                pump(src)
        except OSError as e:
            raise PkgError(f"cannot read {url}: {e.strerror or e}") from None
    elif url.lower().startswith('https://'):
        request = urllib.request.Request(url, headers={'User-Agent': f"til-pkg/{til.TIL_VERSION}"})
        try:
            with _urlopen(request, timeout=DOWNLOAD_TIMEOUT) as src:
                final = getattr(src, 'geturl', lambda: url)()
                if urllib.parse.urlparse(final).scheme.lower() != 'https':      # defence in depth
                    raise PkgError(f"cannot download {url}: it was redirected to {final} "
                                   f"(only https:// is allowed)")
                pump(src)
        except urllib.error.HTTPError as e:
            raise PkgError(f"cannot download {url}: HTTP {e.code} {e.reason}") from None
        except (urllib.error.URLError, OSError) as e:
            raise PkgError(f"cannot download {url}: {getattr(e, 'reason', e)}") from None
    else:
        raise PkgError(f"cannot download {url}: only https:// and file:// URLs are supported")
    return h.hexdigest()


def safe_member_path(name: str, archive: str) -> str:
    """Validate an archive member name; returns the normalized '/'-separated
    relative path ('' for the archive root). Absolute paths, drive letters,
    '..' and characters that are unsafe on some OS are refused."""
    n = name.replace('\\', '/')
    if any(ord(c) < 32 for c in n):
        raise PkgError(f"{archive}: unsafe entry name {name!r} (control characters)")
    if n.startswith('/') or re.match(r'^[A-Za-z]:', n):
        raise PkgError(f"{archive}: unsafe entry '{name}' (absolute path)")
    parts = [p for p in n.split('/') if p not in ('', '.')]
    if '..' in parts:
        raise PkgError(f"{archive}: unsafe entry '{name}' ('..' would escape the package directory)")
    if any(':' in p for p in parts):
        raise PkgError(f"{archive}: unsafe entry '{name}' (':' in a file name)")
    return '/'.join(parts)


class _Extractor:
    """Writes archive members below `dest` (which must be new and empty) without
    ever following a link: we only create directories and new regular files."""

    def __init__(self, dest: str, archive: str):
        self.dest = os.path.realpath(dest)
        self.archive = archive
        self.seen: Set[str] = set()
        self.total = 0
        self.count = 0

    def target(self, rel: str) -> str:
        path = os.path.join(self.dest, *rel.split('/'))
        real = os.path.realpath(path)
        if not real.startswith(self.dest + os.sep):
            raise PkgError(f"{self.archive}: entry '{rel}' escapes the package directory")
        return path

    def directory(self, rel: str):
        if rel:
            path = self.target(rel)
            if os.path.isfile(path):
                raise PkgError(f"{self.archive}: '{rel}' is both a file and a directory")
            make_dirs(path)

    def file(self, rel: str, src):
        if not rel:
            raise PkgError(f"{self.archive}: file entry without a name")
        if rel in self.seen:
            raise PkgError(f"{self.archive}: duplicate entry '{rel}'")
        self.seen.add(rel)
        self.count += 1
        if self.count > MAX_PACKAGE_FILES:
            raise PkgError(f"{self.archive}: more than {MAX_PACKAGE_FILES} files")
        path = self.target(rel)
        parent = os.path.dirname(path)
        if os.path.isfile(parent):
            raise PkgError(f"{self.archive}: '{rel}' is inside a file")
        make_dirs(parent)
        try:
            with open(path, 'xb') as out:
                for chunk in iter(lambda: src.read(1 << 20), b''):
                    self.total += len(chunk)
                    if self.total > MAX_PACKAGE_BYTES:
                        raise PkgError(f"{self.archive}: more than {size_text(MAX_PACKAGE_BYTES)} of files")
                    out.write(chunk)
        except (FileExistsError, IsADirectoryError):
            raise PkgError(f"{self.archive}: entry '{rel}' collides with another entry") from None


def _check_gzip_trailer(stream, label: str):
    """tarfile stops reading at the tar end-of-archive marker, so the gzip
    trailer (CRC-32 and length of the data) would never be checked and a
    damaged .tar.gz could be installed with corrupted files. Reading the rest
    of the stream makes gzip verify it (BadGzipFile, an OSError, on mismatch)."""
    rest = 0
    for chunk in iter(lambda: stream.read(1 << 20), b''):
        rest += len(chunk)
        if rest > MAX_PACKAGE_BYTES:
            raise PkgError(f"{label}: more than {size_text(MAX_PACKAGE_BYTES)} of data after the end of the archive")


def extract_archive(path: str, fmt: str, dest: str, label: str):
    """Safely extract a .tar.gz/.tgz ('tar') or .zip archive into `dest`.
    Only regular files and directories are accepted: symbolic links, hard
    links, devices and FIFOs are refused, like absolute names and '..'.
    Damaged archives are an error: zip members are checked against their
    CRC-32, a .tar.gz against the gzip trailer."""
    ex = _Extractor(dest, label)
    try:
        if fmt == 'tar':
            with tarfile.open(path, 'r:gz') as tf:
                for m in tf:
                    rel = safe_member_path(m.name, label)
                    if m.isdir():
                        ex.directory(rel)
                    elif m.issym() or m.islnk():
                        raise PkgError(f"{label}: entry '{m.name}' is a {'symbolic' if m.issym() else 'hard'} "
                                       f"link -> '{m.linkname}' (links are not allowed in packages)")
                    elif m.isreg():
                        src = tf.extractfile(m)
                        ex.file(rel, src)
                    else:
                        raise PkgError(f"{label}: entry '{m.name}' is not a regular file or directory")
                _check_gzip_trailer(tf.fileobj, label)
        else:
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    if info.filename.replace('\\', '/').split('/')[0] == '__MACOSX':
                        continue        # macOS resource-fork metadata, never package content
                    rel = safe_member_path(info.filename, label)
                    mode = info.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise PkgError(f"{label}: entry '{info.filename}' is a symbolic link "
                                       f"(links are not allowed in packages)")
                    if info.is_dir() or stat.S_ISDIR(mode) or info.filename.endswith('\\'):
                        ex.directory(rel)
                        continue
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG):    # many tools store no type bits
                        raise PkgError(f"{label}: entry '{info.filename}' is not a regular file")
                    with zf.open(info) as src:
                        ex.file(rel, src)
    except PkgError:
        raise
    except (tarfile.TarError, zipfile.BadZipFile, zlib.error, EOFError, gzip.BadGzipFile) as e:
        raise PkgError(f"cannot extract {label}: the archive is damaged or not a valid "
                       f"{'.tar.gz' if fmt == 'tar' else '.zip'} file ({e})") from None
    except (OSError, RuntimeError, ValueError) as e:
        raise PkgError(f"cannot extract {label}: {e}") from None


def single_top_dir(d: str) -> str:
    """Archives that hold exactly one top-level directory (as GitHub's do)
    contain the package in that directory."""
    entries = os.listdir(d)
    if len(entries) == 1 and os.path.isdir(os.path.join(d, entries[0])) \
            and not os.path.islink(os.path.join(d, entries[0])):
        return os.path.join(d, entries[0])
    return d


# ═══════════════════════════════════════════════════════════════
# GIT
# ═══════════════════════════════════════════════════════════════

_GIT_ENV_DROP = frozenset(('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_OBJECT_DIRECTORY',
                           'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_COMMON_DIR', 'GIT_NAMESPACE', 'GIT_PREFIX'))


_GIT_CONFIG = ['-c', 'core.autocrlf=false', '-c', 'protocol.ext.allow=never']


def _git_env() -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV_DROP}   # e.g. set inside git hooks
    env['GIT_TERMINAL_PROMPT'] = '0'
    return env


def _git_subcommand(args: List[str]) -> str:
    i = 0
    while i < len(args) and args[i] in ('-C', '--git-dir', '--work-tree'):
        i += 2
    return args[i] if i < len(args) else 'git'


def _git_failed(what: str, args: List[str], stderr: str) -> PkgError:
    msg = stderr.strip()
    return PkgError(f"{what}: git {_git_subcommand(args)} failed" + (f":\n{msg}" if msg else ""))


def git(args: List[str], what: str, check: bool = True, binary: bool = False) -> subprocess.CompletedProcess:
    """Run git without a shell; credentials are never prompted for and the
    `ext::` transport is disabled. binary=True: stdout/stderr as bytes."""
    cmd = ['git'] + _GIT_CONFIG + args
    text = {} if binary else {'text': True, 'encoding': 'utf-8', 'errors': 'replace'}
    try:
        r = subprocess.run(cmd, capture_output=True, env=_git_env(), **text)
    except FileNotFoundError:
        raise PkgError("git is not installed (it is needed for git+ dependencies)") from None
    if check and r.returncode != 0:
        err = r.stderr or r.stdout
        raise _git_failed(what, args, err.decode('utf-8', 'replace') if binary else err)
    return r


def _git_path(raw: bytes, label: str) -> Optional[str]:
    """A path from a git tree -> the '/'-separated relative path to install,
    or None for paths that are never part of a package (.git, .hg, .svn,
    til_modules at any depth). The rules are those of archives, so a package
    installs identically on every OS: no absolute paths or '..', no '\\', ':' or
    control characters, UTF-8 names."""
    try:
        rel = raw.decode('utf-8')
    except UnicodeDecodeError:
        raise PkgError(f"{label}: the file name {raw!r} is not UTF-8") from None
    parts = rel.split('/')
    if rel.startswith('/') or any(p in ('', '.', '..') for p in parts):
        raise PkgError(f"{label}: unsafe path '{rel}' in the repository")
    if any(ord(c) < 32 for c in rel) or '\\' in rel or ':' in rel:
        raise PkgError(f"{label}: the file name {rel!r} cannot be installed on every OS "
                       f"(control characters, '\\' or ':')")
    if any(p in SKIPPED_NAMES for p in parts):
        return None
    return rel


class _BlobReader:
    """Exactly `size` bytes of `git cat-file --batch` output, as a file."""

    def __init__(self, stream, size: int, label: str):
        self.stream, self.left, self.label = stream, size, label

    def read(self, n: int = -1) -> bytes:
        if self.left <= 0:
            return b''
        chunk = self.stream.read(self.left if n is None or n < 0 else min(n, self.left))
        if not chunk:
            raise PkgError(f"{self.label}: git cat-file output ended early")
        self.left -= len(chunk)
        return chunk


def git_export(clone: str, sha: str, dest: str, label: str) -> List[str]:
    """Write the files of commit `sha` of the repository `clone` into the empty
    directory `dest` exactly as committed (the blob bytes). There is no
    checkout: no line-ending conversion (core.eol, core.autocrlf, `text`
    attributes), no smudge filters (git LFS), no `ident` expansion - all of
    which depend on the installing machine's git configuration, so the tree
    hash would too. Symbolic links are refused, as in archives. Returns the
    paths of submodules (not fetched)."""
    r = git(['-C', clone, 'ls-tree', '-r', '-z', '--full-tree', sha], label, binary=True)
    blobs: List[Tuple[str, str]] = []
    submodules: List[str] = []
    for record in r.stdout.split(b'\0'):
        if not record:
            continue
        meta, _tab, raw_path = record.partition(b'\t')
        fields = meta.decode('ascii', 'replace').split(' ')
        if len(fields) != 3:
            raise PkgError(f"{label}: unexpected git ls-tree output {record!r}")
        mode, kind, obj = fields
        rel = _git_path(raw_path, label)
        if rel is None:
            continue
        if kind == 'commit':
            submodules.append(rel)
        elif kind == 'blob' and mode == '120000':
            raise PkgError(f"{label}: '{rel}' is a symbolic link (packages may only contain plain files "
                           f"and directories)")
        elif kind == 'blob' and mode.startswith('100'):
            blobs.append((obj, rel))
        else:
            raise PkgError(f"{label}: '{rel}' is not a regular file (git mode {mode})")
    ex = _Extractor(dest, label)
    if not blobs:
        return submodules
    with tempfile.TemporaryFile() as listing, tempfile.TemporaryFile() as errors:
        listing.write(b''.join(obj.encode('ascii') + b'\n' for obj, _rel in blobs))
        listing.seek(0)
        args = ['-C', clone, 'cat-file', '--batch']
        try:
            proc = subprocess.Popen(['git'] + _GIT_CONFIG + args, stdin=listing, stdout=subprocess.PIPE,
                                    stderr=errors, env=_git_env())
        except FileNotFoundError:
            raise PkgError("git is not installed (it is needed for git+ dependencies)") from None
        out = proc.stdout
        assert out is not None
        try:
            for obj, rel in blobs:
                header = out.readline().split()
                if len(header) != 3 or header[0] != obj.encode('ascii') or header[1] != b'blob':
                    raise PkgError(f"{label}: cannot read '{rel}' from the repository "
                                   f"(git cat-file: {b' '.join(header).decode('utf-8', 'replace')!r})")
                ex.file(rel, _BlobReader(out, int(header[2]), label))
                if out.read(1) != b'\n':
                    raise PkgError(f"{label}: unexpected git cat-file output after '{rel}'")
        finally:
            out.close()
            rc = proc.wait()
        if rc != 0:
            errors.seek(0)
            raise _git_failed(label, args, errors.read().decode('utf-8', 'replace'))
    return submodules


# ═══════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════

class Registry:
    """A JSON index: {"packages": {NAME: {VERSION: SOURCE}}} (the "packages"
    wrapper is optional). SOURCE is a path (relative to the index file), a
    git+URL#rev or an archive URL#sha256=..., or {"source": SOURCE}."""

    def __init__(self, location: str, base_dir: Optional[str], packages: Dict[str, Any]):
        self.location = location
        self.base_dir = base_dir
        self.packages = packages

    @classmethod
    def load(cls, location: str, relative_to: str, workdir: str) -> 'Registry':
        low = location.lower()
        if low.startswith('https://'):
            path = os.path.join(tempfile.mkdtemp(prefix='index-', dir=workdir), 'index.json')
            download(location, path, MAX_INDEX_BYTES)
            base_dir = None
        elif low.startswith('http://'):
            raise PkgError(f"registry {location}: use https:// (or a local file)")
        else:
            path = file_url_path(location) if low.startswith('file://') else \
                os.path.join(relative_to, os.path.expanduser(location))
            path = os.path.abspath(path)            # links kept: til.lock records sources as written
            if os.path.isdir(path):
                path = os.path.join(path, 'index.json')
            if not os.path.isfile(path):
                raise PkgError(f"registry index not found: {path}")
            base_dir = os.path.dirname(path)
        data = read_json(path, 'registry index')
        if not isinstance(data, dict):
            raise PkgError(f"registry {location}: the index must be a JSON object")
        packages = data['packages'] if isinstance(data.get('packages'), dict) else data
        return cls(location, base_dir, packages)

    def versions(self, name: str) -> List[Tuple[Version, str]]:
        entry = self.packages.get(name)
        if entry is None:
            raise PkgError(f"package '{name}' is not in the registry {self.location}")
        if not isinstance(entry, dict) or not entry:
            raise PkgError(f"registry {self.location}: '{name}' must map versions to sources")
        out = []
        for vtext, src in entry.items():
            v = Version.parse(vtext, f"registry {self.location}: package '{name}'")
            if isinstance(src, dict):
                src = src.get('source')
            if not isinstance(src, str) or not src.strip():
                raise PkgError(f"registry {self.location}: {name} {vtext}: the source must be a string")
            out.append((v, src))
        return out


def latest_version(versions: List[Tuple[Version, str]]) -> Version:
    releases = [v for v, _src in versions if not v.pre] or [v for v, _src in versions]
    return max(releases, key=Version.key)


# ═══════════════════════════════════════════════════════════════
# RESOLUTION
# ═══════════════════════════════════════════════════════════════

class Package:
    """One resolved package (a node of the dependency graph)."""

    def __init__(self, name: str, spec: Spec, requirer: str):
        self.name = name
        self.spec = spec                        # the spec it was first required with
        self.requirers: List[Tuple[str, Spec]] = [(requirer, spec)]
        self.source: Optional[Spec] = None      # concrete source (path / git / archive)
        self.resolved = ""                      # til.lock "resolved"
        self.version: Optional[str] = None
        self.sha256 = ""                        # tree hash of the files to install
        self.dir = ""                           # directory holding exactly those files
        self.base_dir: Optional[str] = None     # its path dependencies are relative to this (local only)
        self.anchor = 'project'                 # Spec.anchor its relative path dependencies inherit
        self.deps: Dict[str, str] = {}          # its til.json dependencies
        self.git_sha: Optional[str] = None
        self.archive_sha: Optional[str] = None
        self.from_lock = False                  # reproduced from a til.lock entry


class _NeedConstraint(Exception):
    """A registry requirement the version picked so far does not satisfy:
    the resolver restarts with it as an extra constraint."""

    def __init__(self, name: str, spec: Spec, requirer: str, message: str):
        super().__init__(message)
        self.name, self.spec, self.requirer, self.message = name, spec, requirer, message


class Fetcher:
    """Fetches sources into a private work directory (cached per run)."""

    def __init__(self, root: str, workdir: str):
        self.root = root
        self.workdir = workdir
        self._paths: Dict[str, Tuple[str, str]] = {}
        self._clones: Dict[str, str] = {}
        self._revs: Dict[Tuple[str, Optional[str]], str] = {}
        self._checkouts: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._archives: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
        self.warnings: List[str] = []

    def new_dir(self, prefix: str) -> str:
        return tempfile.mkdtemp(prefix=prefix, dir=self.workdir)

    def fetch(self, source: Spec, name: str) -> Tuple[str, str, str]:
        """-> (directory with the package files, tree hash, resolved source)."""
        if source.kind == 'path':
            return self._path(source, name)
        if source.kind == 'git':
            return self._git(source, name)
        if source.kind == 'archive':
            return self._archive(source, name)
        raise PkgError(f"package '{name}': cannot fetch a registry requirement directly")

    # -- local directories --------------------------------------------------
    def _path(self, source: Spec, name: str) -> Tuple[str, str, str]:
        p = source.path or ''
        if not os.path.isdir(p):
            raise PkgError(f"package '{name}': directory not found: {p} (paths are relative to the "
                           f"directory of the til.json that declares them)")
        modules = os.path.join(self.root, MODULES)
        if p == self.root or (p + os.sep).startswith(modules + os.sep):
            raise PkgError(f"package '{name}' = {source.raw}: a dependency cannot be the project itself "
                           f"or live inside {MODULES}/")
        if p not in self._paths:
            dest = self.new_dir('path-')
            self._paths[p] = (dest, copy_package(p, dest))
        dest, h = self._paths[p]
        return dest, h, lock_source_path(source.written or p, self.root, source.anchor)

    # -- git ----------------------------------------------------------------
    def _clone(self, url: str) -> str:
        if url not in self._clones:
            d = self.new_dir('clone-')
            git(['clone', '--quiet', '--no-checkout', '--', url, d], f"git {url}")
            self._clones[url] = d
        return self._clones[url]

    def git_commit(self, source: Spec) -> str:
        """Resolve the spec's revision (default: the remote HEAD) to a commit SHA."""
        key = (source.url or '', source.rev)
        if key not in self._revs:
            clone = self._clone(source.url or '')
            rev = source.rev
            candidates = [rev] if rev else ['HEAD']
            if rev and not _SHA_RE.match(rev):
                candidates.append('refs/remotes/origin/' + rev)       # a branch other than the default
            sha = None
            for cand in candidates:
                r = git(['-C', clone, 'rev-parse', '--verify', '--quiet', '--end-of-options', cand + '^{commit}'],
                        f"git {source.url}", check=False)
                if r.returncode == 0 and r.stdout.strip():
                    sha = r.stdout.strip()
                    break
            if sha is None or not _SHA_RE.match(sha):
                raise PkgError(f"git {source.url}: no commit, tag or branch '{rev or 'HEAD'}'")
            self._revs[key] = sha
        return self._revs[key]

    def _git(self, source: Spec, name: str) -> Tuple[str, str, str]:
        url = source.url or ''
        sha = self.git_commit(source)
        if source.rev and _SHA_RE.match(source.rev) and sha != source.rev:
            warning = (f"package '{name}': {source.rev} ({source.raw}) is the SHA of an annotated tag, not of a "
                       f"commit; it names the commit {sha}, and since til.lock records commits, every install "
                       f"checks it in the repository - pin the commit SHA to install without the network")
            if warning not in self.warnings:
                self.warnings.append(warning)
        if (url, sha) not in self._checkouts:
            work = self.new_dir('git-')
            submodules = git_export(self._clone(url), sha, work, f"git {url} ({sha[:12]})")
            if submodules:
                self.warnings.append(f"package '{name}': git submodules are not fetched ({url}: "
                                     f"{', '.join(submodules)})")
            self._checkouts[(url, sha)] = (work, tree_hash(work))
        work, h = self._checkouts[(url, sha)]
        where = lock_source_path(source.written or source.path, self.root, source.anchor) if source.path else url
        return work, h, f"git+{where}#{sha}"

    # -- archives -----------------------------------------------------------
    def _archive(self, source: Spec, name: str) -> Tuple[str, str, str]:
        url = source.url or ''
        key = (url, source.sha256 or '')
        if key not in self._archives:
            data = os.path.join(self.new_dir('download-'), 'archive')
            actual = download(url, data)
            if source.sha256 is None:
                raise PkgError(f"package '{name}': archive dependencies must pin the archive's sha256; "
                               f"its current hash is:\n  {url}#sha256={actual}")
            if actual != source.sha256:
                raise PkgError(f"package '{name}': sha256 mismatch for {url}\n  expected {source.sha256}\n"
                               f"  got      {actual}\nthe archive was not extracted")
            out = self.new_dir('archive-')
            extract_archive(data, source.fmt or 'tar', out, url)
            os.unlink(data)
            top = single_top_dir(out)
            self._archives[key] = (top, tree_hash(top), f"{url}#sha256={actual}")
        return self._archives[key]


class Resolver:
    """Resolves the dependency graph of a project (breadth first, dependencies
    of one til.json in name order, so your own declarations come first).

    * One copy per package name (til_modules/ is flat): a second requirement
      must be satisfied by the package already chosen - the same directory,
      the same git commit, the same archive hash, or (registry requirements)
      a version in the range. Otherwise it is a conflict.
    * Two registry requirements that the first pick does not satisfy restart
      the resolution with both as constraints (highest version satisfying all).
    * Cycles are an error.
    * Packages in til.lock whose spec is unchanged are reproduced exactly
      (same resolved source, same tree hash) unless they are being updated -
      but til.lock never overrides an integrity pin of a til.json (an
      archive's #sha256=, a full commit SHA): an entry that contradicts one
      is not used (a warning; with --frozen an error).
    * frozen (--frozen): every package must be reproduced from til.lock; one
      that is not in it (or not as required) is an error before anything is
      fetched for it."""

    def __init__(self, root: str, manifest: Dict[str, Any], lock: Optional[Dict[str, Any]],
                 unlocked: Optional[Set[str]], fetcher: Fetcher, cwd: Optional[str] = None,
                 frozen: bool = False):
        self.root = root
        self.manifest = manifest
        self.root_name: Optional[str] = manifest.get('name')     # None: nothing can depend on the project
        self.root_label = self.root_name or os.path.basename(root) or 'project'     # in messages
        self.lock: Dict[str, Any] = (lock or {}).get('packages', {})
        self.unlocked = unlocked                 # None: ignore the whole lock
        self.fetcher = fetcher
        self.cwd = cwd or os.getcwd()            # a relative TIL_REGISTRY is relative to this (-C DIR)
        self.frozen = frozen
        self.constraints: Dict[str, List[Tuple[Spec, str]]] = {}
        self._registry: Optional[Registry] = None
        self._installed: Dict[str, Optional[str]] = {}

    def resolve(self) -> Dict[str, Package]:
        if self.frozen:
            self._check_frozen()
        for _ in range(MAX_RESOLVE_PASSES):
            try:
                return self._pass()
            except _NeedConstraint as nc:
                known = self.constraints.setdefault(nc.name, [])
                if any(s.raw == nc.spec.raw and who == nc.requirer for s, who in known):
                    raise PkgError(nc.message) from None
                known.append((nc.spec, nc.requirer))
        raise PkgError("could not find a consistent set of registry versions "
                       f"(gave up after {MAX_RESOLVE_PASSES} attempts)")

    def _pass(self) -> Dict[str, Package]:
        packages: Dict[str, Package] = {}
        root_deps = self.manifest.get('dependencies') or {}
        root_key = self.root_name or '<project>'                 # never a package name
        edges: Dict[str, List[str]] = {root_key: sorted(root_deps)}
        queue = deque([(self.root_label, self.root, root_deps, os.path.join(self.root, MANIFEST), 'project')])
        while queue:
            requirer, base_dir, deps, where, anchor = queue.popleft()
            for name in sorted(deps):
                if name == self.root_name:
                    continue                                  # reported as a cycle below
                spec = parse_spec(deps[name], base_dir, name, where, anchor)
                pkg = packages.get(name)
                if pkg is not None:
                    self._check_compatible(pkg, spec, requirer)
                    pkg.requirers.append((requirer, spec))
                    continue
                pkg = self._obtain(name, spec, requirer)
                packages[name] = pkg
                edges[name] = sorted(pkg.deps)
                queue.append((name, pkg.base_dir, pkg.deps, f"{MANIFEST} of package '{name}'", pkg.anchor))
        self._check_cycles(edges, root_key)
        folded: Dict[str, str] = {}
        for name in packages:
            other = folded.setdefault(name.casefold(), name)
            if other != name:
                raise PkgError(f"packages '{other}' and '{name}' differ only in case; they cannot both be "
                               f"installed in {MODULES}/")
        return packages

    def _check_cycles(self, edges: Dict[str, List[str]], root_key: str):
        """Depth-first search for a cycle; iterative, so a very long dependency
        chain cannot exhaust Python's recursion limit."""
        state: Dict[str, int] = {root_key: 1}
        path: List[str] = [root_key]
        todo: List[Tuple[str, int]] = [(root_key, 0)]          # (node, index of its next edge)
        while todo:
            n, i = todo[-1]
            out = edges.get(n, [])
            if i == len(out):
                todo.pop()
                path.pop()
                state[n] = 2
                continue
            todo[-1] = (n, i + 1)
            m = out[i]
            if state.get(m) == 1:
                cycle = path[path.index(m):] + [m]
                raise PkgError("dependency cycle: " + " -> ".join(cycle))
            if m not in state:
                state[m] = 1
                path.append(m)
                todo.append((m, 0))

    # -- --frozen -------------------------------------------------------------
    def _frozen_error(self, problem: str, suspicious: bool = False) -> PkgError:
        if suspicious:
            return PkgError(f"--frozen: {LOCKFILE} contradicts {MANIFEST}: {problem}\n(til pkg never writes such "
                            f"an entry - was {LOCKFILE} edited? `til pkg install` resolves the package from "
                            f"{MANIFEST} and rewrites {LOCKFILE}; --frozen never does)")
        return PkgError(f"--frozen: {LOCKFILE} is out of date with {MANIFEST}: {problem}\n"
                        f"(`til pkg install` resolves it again and writes {LOCKFILE}; --frozen never does)")

    def _check_frozen(self):
        """--frozen, before anything is fetched: til.lock must have every direct
        dependency of til.json, as til.json requires it, and nothing that is no
        longer required. (Indirect dependencies are checked as the resolution
        reaches them, also before they are fetched - _usable_lock_entry.)"""
        deps = self.manifest.get('dependencies') or {}
        for name in sorted(deps):
            if name == self.root_name:
                continue
            spec = parse_spec(deps[name], self.root, name, os.path.join(self.root, MANIFEST), 'project')
            _found, problem = self._lock_entry(name, spec, self.root_label, ask_git=False)
            if problem is not None:
                raise self._frozen_error(*problem)
        needed: Set[str] = set()
        todo = [n for n in deps if n in self.lock]
        while todo:
            n = todo.pop()
            if n not in needed:
                needed.add(n)
                todo.extend(d for d in self.lock[n].get('dependencies', {}) if d in self.lock)
        extra = sorted(set(self.lock) - needed)
        if extra:       # (one sentence per number: the messages are translated, docs/TIL_I18N.md)
            raise self._frozen_error(
                f"{extra[0]} is in {LOCKFILE}, but nothing in {MANIFEST} requires it any more" if len(extra) == 1
                else f"{', '.join(extra)} are in {LOCKFILE}, but nothing in {MANIFEST} requires them any more")

    # -- obtaining a package --------------------------------------------------
    def _obtain(self, name: str, spec: Spec, requirer: str) -> Package:
        locked = self._usable_lock_entry(name, spec, requirer)
        pkg = Package(name, spec, requirer)
        if locked is not None:
            entry, source = locked
            pkg.from_lock = True
            pkg.source = source
            pkg.resolved = entry['resolved']
            pkg.version = entry.get('version')
            installed = self._installed_copy(name, entry['sha256'])
            if installed:
                pkg.dir, pkg.sha256 = installed, entry['sha256']
            else:
                pkg.dir, pkg.sha256, _resolved = self.fetcher.fetch(source, name)
                if pkg.sha256 != entry['sha256']:
                    raise PkgError(
                        f"package '{name}': the files from {entry['resolved']} do not match til.lock\n"
                        f"  til.lock sha256 {entry['sha256']}\n  fetched  sha256 {pkg.sha256}\n"
                        f"the source changed after til.lock was written; run "
                        f"`til pkg install --update {name}` to accept the new contents")
        else:
            if spec.kind == 'registry':
                version, source = self._pick_registry_version(name, spec, requirer)
                pkg.version = version.text
            else:
                source = spec
            pkg.source = source
            pkg.dir, pkg.sha256, pkg.resolved = self.fetcher.fetch(source, name)
        if pkg.source.kind == 'git':
            pkg.git_sha = pkg.resolved.rsplit('#', 1)[1]
        elif pkg.source.kind == 'archive':
            pkg.archive_sha = pkg.resolved.rsplit('#sha256=', 1)[1]
        if pkg.source.kind == 'path' and spec.kind == 'path':
            # relative dependencies start from the directory as til.json wrote it (Spec.written)
            pkg.base_dir, pkg.anchor = spec.written or pkg.source.path, spec.anchor
        self._read_package_manifest(pkg, requirer)
        return pkg

    def _usable_lock_entry(self, name: str, spec: Spec, requirer: str) -> Optional[Tuple[Dict[str, Any], Spec]]:
        """The til.lock entry (and its resolved source) that reproduces the
        requirement `spec`, or None: then the package is resolved from `spec`
        (--frozen: an error instead). An entry that contradicts an integrity
        pin of `spec` is reported with a warning - til.lock was edited."""
        if self.unlocked is None or name in self.unlocked:
            return None
        found, problem = self._lock_entry(name, spec, requirer)
        if found is not None:
            return found
        assert problem is not None
        message, suspicious = problem
        if self.frozen:
            raise self._frozen_error(message, suspicious)
        if suspicious:
            self.fetcher.warnings.append(f"the {LOCKFILE} entry for '{name}' is not used: {message} (til pkg "
                                         f"never writes such an entry - was {LOCKFILE} edited?); '{name}' is "
                                         f"resolved again")
        return None

    def _lock_entry(self, name: str, spec: Spec, requirer: str, ask_git: bool = True) \
            -> Tuple[Optional[Tuple[Dict[str, Any], Spec]], Optional[Tuple[str, bool]]]:
        """-> ((entry, resolved source), None) when til.lock's entry for `name`
        reproduces the requirement `spec`, else (None, (why, suspicious)).
        suspicious: til pkg never writes such an entry (til.lock was edited).
        Integrity pins win over til.lock: an archive spec's #sha256= (and URL)
        and a full commit SHA must be exactly what the entry resolves to.
        ask_git=False: never contact a repository (a full SHA that differs from
        the locked commit then counts as reproducible - it may be an annotated
        tag object pointing to that commit; the resolution checks it)."""
        entry = self.lock.get(name)
        if entry is None:
            return None, (f"{requirer} requires {name} = {spec.raw}, which is not in {LOCKFILE}", False)
        if entry.get('spec') != spec.raw:
            return None, (f"{requirer} requires {name} = {spec.raw}, {LOCKFILE} has {name} = {entry.get('spec')}",
                          False)
        source = parse_spec(entry['resolved'], self.root, name, os.path.join(self.root, LOCKFILE))
        if source.kind == 'registry':
            raise PkgError(f"{os.path.join(self.root, LOCKFILE)}: package '{name}': \"resolved\" must be a "
                           f"path, git+URL or archive URL")
        here = f"{LOCKFILE} resolves {name} to {entry['resolved']}"
        if spec.kind != 'registry' and source.kind != spec.kind:
            return None, (f"{here}, which is not the source {spec.raw}", True)
        if source.kind == 'git' and not (source.rev and _SHA_RE.match(source.rev)):
            return None, (f"{here}, which does not name a commit by its full SHA", True)
        if spec.kind == 'path' and source.path != spec.path:
            return None, (f"{here}, but {spec.raw} now names {display_path(spec.written or '', self.root)}",
                          False)                # the same text now names another directory
        if spec.kind == 'git':
            if source.url != spec.url:
                return None, (f"{here}, which is not the repository of {spec.raw}", spec.path is None)
            if spec.rev and _SHA_RE.match(spec.rev) and source.rev != spec.rev and ask_git:
                # a full commit SHA in til.json is an integrity pin; the only other commit it can name is
                # the one an annotated tag object points to (`git rev-parse v1.0` prints the tag's SHA)
                if self.fetcher.git_commit(spec) != source.rev:
                    return None, (f"{here}, but {requirer} pins the commit {spec.rev}", True)
        if spec.kind == 'archive':
            if spec.sha256 is None:
                return None, (f"{spec.raw} does not pin the archive's sha256", False)
            if source.url != spec.url or source.sha256 != spec.sha256:
                return None, (f"{here}, but {requirer} pins {spec.raw}", True)
        if spec.kind == 'registry':
            try:
                version = Version.parse(entry.get('version'))
            except PkgError:
                return None, (f"{LOCKFILE} has no valid version for {name}", True)
            for s, who in [(spec, requirer)] + self.constraints.get(name, []):
                if not s.req.matches(version):
                    return None, (f"{who} requires {name} {s.req}, {LOCKFILE} has {name} {version}", False)
        return (entry, source), None

    def _installed_copy(self, name: str, sha256: str) -> Optional[str]:
        """til_modules/NAME when it already holds exactly the locked files."""
        target = os.path.join(self.root, MODULES, name)
        if name not in self._installed:
            h = None
            if os.path.isdir(target) and not _is_link(target):
                try:
                    h = tree_hash(target, installed=True)
                except (PkgError, OSError):
                    h = None
            self._installed[name] = h
        return target if self._installed[name] == sha256 else None

    def _registry_index(self, name: str, spec: Spec) -> Registry:
        if self._registry is None:
            env = os.environ.get('TIL_REGISTRY', '').strip()
            location = env or self.manifest.get('registry')
            if not location:
                raise PkgError(f"dependency '{name}' = '{spec.raw}' is a version requirement, but no registry "
                               f"is configured: set TIL_REGISTRY or \"registry\" in {MANIFEST} (a path or URL "
                               f"of a registry index), or give a path, git+URL or archive URL")
            self._registry = Registry.load(location, self.cwd if env else self.root, self.fetcher.workdir)
        return self._registry

    def _pick_registry_version(self, name: str, spec: Spec, requirer: str) -> Tuple[Version, Spec]:
        registry = self._registry_index(name, spec)
        versions = registry.versions(name)
        reqs = [(spec, requirer)] + [c for c in self.constraints.get(name, []) if c[0].raw != spec.raw
                                     or c[1] != requirer]
        fits = [(v, src) for v, src in versions if all(s.req.matches(v) for s, _who in reqs)]
        if not fits:
            lines = [f"no version of '{name}' in the registry satisfies "
                     + ("all requirements:" if len(reqs) > 1 else f"'{spec.req}' (required by {requirer})")]
            if len(reqs) > 1:
                lines += [f"  {who} requires {name} {s.req}" for s, who in reqs]
            lines.append("  available: " + ", ".join(str(v) for v in sorted(v for v, _src in versions)))
            raise PkgError("\n".join(lines))
        version, src = max(fits, key=lambda vs: vs[0].key())
        source = parse_spec(src, registry.base_dir, name, f"registry {registry.location} ({name} {version})")
        if source.kind == 'registry':
            raise PkgError(f"registry {registry.location}: {name} {version}: the source must be a path, "
                           f"git+URL or archive URL, not '{src}'")
        return version, source

    def _read_package_manifest(self, pkg: Package, requirer: str):
        label = f"{MANIFEST} of package '{pkg.name}' ({pkg.resolved})"
        manifest = load_manifest(pkg.dir, label, required=False) or {}
        declared = manifest.get('name')
        if declared is not None and declared != pkg.name:
            raise PkgError(f"{requirer} requires '{pkg.name}', but the package at {pkg.resolved} is named "
                           f"'{declared}' (installing a package under another name is not supported)")
        mv = manifest.get('version')
        if pkg.spec.kind == 'registry':
            if mv is not None and Version.parse(mv) != Version.parse(pkg.version or ''):
                raise PkgError(f"the registry lists {pkg.name} {pkg.version} at {pkg.resolved}, but its "
                               f"{MANIFEST} says version {mv}")
        else:
            pkg.version = mv
        pkg.deps = dict(manifest.get('dependencies') or {})
        main = manifest.get('main')
        if main is not None and not os.path.isfile(os.path.join(pkg.dir, *main.replace('\\', '/').split('/'))):
            raise PkgError(f"package '{pkg.name}' ({pkg.resolved}): its \"main\" file {main} does not exist")
        if til.package_entry_file(pkg.dir, pkg.name) is None:
            warning = (f"package '{pkg.name}' has no \"main\", {pkg.name}.til, lib.til or mod.til: "
                       f"`import {pkg.name}` will not find it (only `import {pkg.name}.<module>`)")
            if warning not in self.fetcher.warnings:
                self.fetcher.warnings.append(warning)

    # -- a second requirement for a package -----------------------------------
    def _check_compatible(self, pkg: Package, spec: Spec, requirer: str):
        same_text = spec.raw == pkg.spec.raw
        if same_text and (spec.kind in ('registry', 'archive') or (spec.kind == 'git' and spec.path is None)):
            return          # the very same declaration (relative paths depend on where they are written)
        source = pkg.source
        assert source is not None
        if spec.kind == 'registry':
            if pkg.version is not None and spec.req.matches(Version.parse(pkg.version)):
                return
            message = self._conflict(pkg, spec, requirer)
            if pkg.spec.kind == 'registry':
                raise _NeedConstraint(pkg.name, spec, requirer, message)
            raise PkgError(message)
        same = False
        if spec.kind == 'path':
            same = source.kind == 'path' and source.path == spec.path
        elif spec.kind == 'git':
            same = source.kind == 'git' and source.url == spec.url and self._same_commit(pkg, spec, requirer)
        elif spec.kind == 'archive':
            same = source.kind == 'archive' and spec.sha256 is not None and spec.sha256 == pkg.archive_sha
        if not same:
            raise PkgError(self._conflict(pkg, spec, requirer))

    def _same_commit(self, pkg: Package, spec: Spec, requirer: str) -> bool:
        """Does the git requirement `spec` (same repository) name the commit
        chosen for `pkg`? Answered without the network when possible:
        * a full commit SHA equal to that commit (otherwise it is resolved in
          the repository: it may be an annotated tag object's SHA - what `git
          rev-parse v1.0` prints - which names the commit the tag points to;
          an integrity pin is never taken on til.lock's word);
        * when `pkg` was reproduced from til.lock and til.lock records this very
          requirement (`requirer` depends on it with the same text), it was
          checked when the lock was written: like the package itself, a
          branch or HEAD requirement stays pinned to the locked commit until
          `til pkg install --update` - an upstream push changes nothing and a
          repeated install works offline.
        Otherwise the revision is resolved in the repository."""
        if spec.rev and _SHA_RE.match(spec.rev):
            return spec.rev == pkg.git_sha or self.fetcher.git_commit(spec) == pkg.git_sha
        if pkg.from_lock:
            recorded = (self.lock.get(requirer) or {}).get('dependencies', {}).get(pkg.name)
            if recorded == spec.raw:
                return True
        return self.fetcher.git_commit(spec) == pkg.git_sha

    def _conflict(self, pkg: Package, spec: Spec, requirer: str) -> str:
        def show(s: Spec) -> str:
            if s.kind == 'path' and s.written is not None and display_path(s.written, self.root) != s.raw:
                return f"{s.raw} (= {display_path(s.written, self.root)})"
            return s.raw
        lines = [f"conflicting requirements for package '{pkg.name}':"]
        lines += [f"  {who} requires {pkg.name} = {show(s)}" for who, s in pkg.requirers]
        lines.append(f"  {requirer} requires {pkg.name} = {show(spec)}")
        version = f" {pkg.version}" if pkg.version else ""
        lines.append(f"the first resolved to {pkg.name}{version} from {pkg.resolved}; {MODULES}/ holds one copy "
                     f"of each package, so these requirements must agree")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# INSTALLING
# ═══════════════════════════════════════════════════════════════

def _is_scratch(entry: str) -> bool:
    return entry.startswith('.') and ('.new-' in entry or '.old-' in entry)


def apply_packages(root: str, packages: Dict[str, Package]) -> List[Tuple[str, str]]:
    """Make til_modules/ hold exactly `packages`. Each package directory is
    copied to a scratch directory inside til_modules/, checked against its
    tree hash and then renamed into place. Returns [(change, name)] with
    change = 'added' | 'updated' | 'removed'."""
    modules = modules_dir(root)                 # never through a link (it deletes what it did not install)
    changes: List[Tuple[str, str]] = []
    if packages and not os.path.isdir(modules):
        os.mkdir(modules)
    for name in sorted(packages):
        pkg = packages[name]
        target = os.path.join(modules, name)
        if os.path.abspath(pkg.dir) == os.path.abspath(target):
            continue                            # already installed, verified against til.lock
        exists = os.path.lexists(target)
        if exists and os.path.isdir(target) and not _is_link(target):
            try:
                if tree_hash(target, installed=True) == pkg.sha256:
                    continue
            except (PkgError, OSError):
                pass
        scratch = make_dir(modules, f".{name}.new-")
        try:
            if copy_package(pkg.dir, scratch) != pkg.sha256:
                raise PkgError(f"package '{name}': its files changed while it was being installed")
            old = None
            if exists:
                old = make_dir(modules, f".{name}.old-")
                os.rmdir(old)
                os.rename(target, old)
            os.rename(scratch, target)
            scratch = ''
            if old:
                rmtree(old)
        finally:
            if scratch:
                rmtree(scratch)
        changes.append(('updated' if exists else 'added', name))
    if os.path.isdir(modules):
        for entry in sorted(os.listdir(modules)):
            if entry in packages or (entry.startswith('.') and not _is_scratch(entry)):
                continue
            rmtree(os.path.join(modules, entry))
            if not _is_scratch(entry):
                changes.append(('removed', entry))
        if not os.listdir(modules):
            os.rmdir(modules)
    return changes


def install_project(root: str, manifest: Dict[str, Any], unlocked: Optional[Set[str]],
                    frozen: bool = False, before_apply=None, cwd: Optional[str] = None) -> Dict[str, Package]:
    """Resolve, fetch and install the dependencies of the project in `root`;
    write til.lock. `unlocked`: None re-resolves everything (--update),
    otherwise the names to re-resolve (the rest is reproduced from til.lock).
    `before_apply()` runs after a successful resolution, before anything in
    the project changes (add/remove write til.json there). `cwd`: the
    directory the command runs in (-C), for a relative TIL_REGISTRY."""
    modules_dir(root)                           # refuse a til_modules link before doing anything
    lock_path = os.path.join(root, LOCKFILE)
    old_lock: Optional[Dict[str, Any]] = None
    if os.path.exists(lock_path):
        try:
            old_lock = read_lock(lock_path)
        except PkgError:
            if unlocked is not None:
                raise
    if frozen and old_lock is None:
        raise PkgError(f"--frozen: {lock_path} does not exist (run `til pkg install` first)")
    with work_directory() as workdir:
        fetcher = Fetcher(root, workdir)
        resolver = Resolver(root, manifest, old_lock if unlocked is not None else None, unlocked, fetcher, cwd,
                            frozen=frozen)
        packages = resolver.resolve()
        unknown = sorted((unlocked or set()) - set(packages))
        if unknown:
            raise PkgError(f"not a dependency of this project: {', '.join(unknown)}")
        new_lock = lock_data(packages)
        if frozen and new_lock != old_lock:
            raise PkgError(f"--frozen: {LOCKFILE} is out of date with {MANIFEST} (run `til pkg install`)")
        for warning in fetcher.warnings:
            print(_(f"til pkg: warning: {warning}"), file=sys.stderr)
        if before_apply is not None:
            before_apply()
        changes = apply_packages(root, packages)
    text = json_text(new_lock)
    old_bytes = None
    try:
        with open(lock_path, 'rb') as f:        # bytes: the old til.lock may be anything (UTF-16, garbage)
            old_bytes = f.read()
    except FileNotFoundError:
        pass
    lock_written = text.encode('utf-8') != old_bytes and not frozen      # --frozen never writes til.lock
    if lock_written:
        write_text_atomic(lock_path, text)
    _report(packages, changes, (old_lock or {}).get('packages', {}), lock_written)
    return packages


def _report(packages: Dict[str, Package], changes: List[Tuple[str, str]], old: Dict[str, Any],
            lock_written: bool):
    marks = {'added': '+', 'updated': '~', 'removed': '-'}
    for change, name in changes:
        if change == 'removed':
            print(f"- {name}")
            continue
        pkg = packages[name]
        version = f" {pkg.version}" if pkg.version else ""
        was = (old.get(name) or {}).get('version')
        if change == 'updated' and was and was != pkg.version:
            version = f" {was} -> {pkg.version}"
        print(f"{marks[change]} {name}{version} ({pkg.resolved})")
    counts = {c: sum(1 for ch, _n in changes if ch == c) for c in ('added', 'updated', 'removed')}
    n = len(packages)
    summary = f"{n} package{'s' if n != 1 else ''} in {MODULES}/"
    if any(counts.values()):
        summary += " (" + ", ".join(f"{v} {k}" for k, v in counts.items() if v) + ")"
    else:
        summary += ", up to date"
    if lock_written:
        summary += f"; {LOCKFILE} written"
    print(summary)


# ═══════════════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════════════

class UsageError(PkgError):
    """A command-line usage error: exit code 2, like argparse's own errors."""


def installed_package_owner(directory: str) -> Optional[str]:
    """If `directory` is PROJECT/til_modules or anything below it (installed
    packages) - where PROJECT has a til.json or til.lock - the PROJECT
    directory, else None."""
    d = os.path.realpath(directory)
    while True:
        parent = os.path.dirname(d)
        if os.path.basename(d) == MODULES and (os.path.isfile(os.path.join(parent, MANIFEST))
                                                or os.path.isfile(os.path.join(parent, LOCKFILE))):
            return parent
        if parent == d:
            return None
        d = parent


def _refuse_installed_copy(directory: str):
    owner = installed_package_owner(directory)
    if owner is not None:
        raise PkgError(f"{directory} is inside an installed package ({os.path.join(owner, MODULES)}/); "
                       f"til pkg does not change installed copies (their own dependencies are installed "
                       f"flat in {MODULES}/ of the project) - run it in {owner}")


def _project(start: str) -> Tuple[str, Dict[str, Any]]:
    # the directory the command runs in first: inside PROJECT/til_modules/NAME/ without a til.json of
    # its own, find_root() would climb to PROJECT and the command would change the project from there
    _refuse_installed_copy(start)
    root = find_root(start)
    _refuse_installed_copy(root)
    return root, load_manifest(root) or {}


def _spec_from_cwd(raw: str, start: str, root: str) -> str:
    """`til pkg add NAME ../x` in a subdirectory (or with -C DIR): like in any
    shell command, a relative path is relative to where the command runs.
    til.json paths are relative to til.json, so such a path is rewritten
    relative to the project root. Absolute and ~/ paths stay as written."""
    here = os.path.realpath(start)
    if os.path.normcase(here) == os.path.normcase(root):
        return raw
    text = raw.strip()
    if text.startswith('git+'):
        url, sep, rev = text[4:].partition('#')
        if url and _is_local_git_path(url) and _path_anchor(url, 'project') == 'project':
            return 'git+' + display_path(os.path.abspath(os.path.join(here, url)), root) + sep + rev
        return raw
    if text.lower().startswith(('https://', 'http://', 'file://')) or not _looks_like_path(text):
        return raw
    if _path_anchor(text, 'project') != 'project':
        return raw
    return display_path(os.path.abspath(os.path.join(here, text)), root)


def _write_manifest(root: str, manifest: Dict[str, Any]):
    write_text_atomic(os.path.join(root, MANIFEST), json_text(manifest))


def _with_dependencies(manifest: Dict[str, Any], deps: Dict[str, str]) -> Dict[str, Any]:
    new = dict(manifest)
    new['dependencies'] = {k: deps[k] for k in sorted(deps)}
    return new


def cmd_init(args, start: str) -> int:
    directory = os.path.realpath(start)
    path = os.path.join(directory, MANIFEST)
    if os.path.exists(path):
        raise PkgError(f"{path} already exists")
    _refuse_installed_copy(directory)
    name = args.name
    if name is None:
        name = re.sub(r'\W', '_', os.path.basename(directory)) or 'project'
        if name[0].isdigit():
            name = '_' + name
        try:
            check_name(name)
        except PkgError:
            raise PkgError(f"cannot derive a package name from the directory name "
                           f"'{os.path.basename(directory)}': run `til pkg init NAME`") from None
    check_name(name)
    _write_manifest(directory, {"name": name, "version": "0.1.0", "dependencies": {}})
    print(f"created {path} (package '{name}' 0.1.0)")
    return 0


def cmd_add(args, start: str) -> int:
    root, manifest = _project(start)
    name, raw = args.name, args.spec
    if raw is None and '@' in name:
        name, _, raw = name.partition('@')
    check_name(name)
    if name == manifest.get('name'):
        raise PkgError(f"'{name}' is this project: a package cannot depend on itself")
    if raw is None:
        env = os.environ.get('TIL_REGISTRY', '').strip()
        location = env or manifest.get('registry')
        if not location:
            raise PkgError(f"`til pkg add {name}` without a SPEC needs a registry: set TIL_REGISTRY or "
                           f"\"registry\" in {MANIFEST}, or give a path, git+URL or archive URL")
        with work_directory() as workdir:
            registry = Registry.load(location, os.path.abspath(start) if env else root, workdir)
            raw = f"^{latest_version(registry.versions(name))}"
    raw = _spec_from_cwd(raw, start, root)
    parse_spec(raw, root, name, f"til pkg add {name}")           # validate before touching anything
    deps = dict(manifest.get('dependencies') or {})
    previous = deps.get(name)
    deps[name] = raw
    new_manifest = _with_dependencies(manifest, deps)

    def save():
        _write_manifest(root, new_manifest)
        if previous is None:
            print(f"added {name} = {raw} to {MANIFEST}")
        elif previous != raw:
            print(f"changed {name} from {previous} to {raw} in {MANIFEST}")
        else:
            print(f"{name} = {raw} is already in {MANIFEST} (resolved again)")

    install_project(root, new_manifest, {name}, before_apply=save, cwd=os.path.abspath(start))
    return 0


def cmd_install(args, start: str) -> int:
    root, manifest = _project(start)
    if args.names and not args.update:
        raise UsageError("package names are only allowed with --update (`til pkg install --update NAME ...`)")
    if args.update and args.frozen:
        raise UsageError("--update and --frozen cannot be combined")
    for n in args.names:
        check_name(n)
    unlocked: Optional[Set[str]] = set()
    if args.update:
        unlocked = set(args.names) if args.names else None
    install_project(root, manifest, unlocked, frozen=args.frozen, cwd=os.path.abspath(start))
    return 0


def cmd_remove(args, start: str) -> int:
    root, manifest = _project(start)
    name = args.name
    deps = dict(manifest.get('dependencies') or {})
    if name not in deps:
        raise PkgError(f"'{name}' is not a dependency in {os.path.join(root, MANIFEST)}")
    del deps[name]
    new_manifest = _with_dependencies(manifest, deps)

    def save():
        _write_manifest(root, new_manifest)
        print(f"removed {name} from {MANIFEST}")

    packages = install_project(root, new_manifest, set(), before_apply=save, cwd=os.path.abspath(start))
    if name in packages:
        users = sorted({who for who, _s in packages[name].requirers})
        print(f"{name} stays installed: required by {', '.join(users)}")
    return 0


def cmd_list(args, start: str) -> int:
    root, manifest = _project(start)
    lock_path = os.path.join(root, LOCKFILE)
    entries = read_lock(lock_path)['packages'] if os.path.exists(lock_path) else {}
    modules = modules_dir(root)
    root_name = manifest.get('name') or os.path.basename(root)
    head = root_name + (f" {manifest['version']}" if manifest.get('version') else "")
    print(f"{head} ({root})")
    deps = manifest.get('dependencies') or {}
    if not deps:
        print("(no dependencies)")
        return 0
    expanded: Set[str] = set()
    names = sorted(deps)
    # (name, spec, prefix, last, direct), printed depth first with an explicit stack (no recursion limit)
    todo = [(name, deps[name], '', i == len(names) - 1, True) for i, name in enumerate(names)][::-1]
    while todo:
        name, spec, prefix, last, direct = todo.pop()
        entry = entries.get(name)
        line = prefix + ('`-- ' if last else '|-- ') + name
        if entry is not None and entry.get('version'):
            line += ' ' + entry['version']
        line += f"  {spec}"
        if entry is not None and entry['resolved'] != spec:
            line += f"  [{entry['resolved']}]"
        children = sorted((entry or {}).get('dependencies', {}))
        if entry is None:
            line += "  (not installed: run `til pkg install`)"
        elif direct and entry['spec'] != spec:
            line += f"  ({LOCKFILE} has {entry['spec']}: run `til pkg install`)"
            children = []
        elif not os.path.isdir(os.path.join(modules, name)):
            line += f"  (missing from {MODULES}/: run `til pkg install`)"
        elif name in expanded and children:
            line += "  (*)"
            children = []
        print(line)
        expanded.add(name)
        sub = prefix + ('    ' if last else '|   ')
        todo.extend([(child, entry['dependencies'][child], sub, i == len(children) - 1, False)
                     for i, child in enumerate(children)][::-1])
    return 0


def pin_mismatches(root: str, manifest: Dict[str, Any], entries: Dict[str, Any]) -> List[str]:
    """Requirements - in til.json and, for every locked package, in its own
    til.json as til.lock records it - whose integrity pin til.lock does not
    honour: an archive's #sha256= (for the package's own spec also its URL),
    a full commit SHA (also the repository, for the package's own spec).
    til pkg never writes such a til.lock; it was edited. A full SHA that
    differs from the locked commit is resolved in the repository (it may be
    an annotated tag object's SHA); only that needs the network."""
    label = manifest.get('name') or os.path.basename(root) or 'project'
    requirements = [(label, n, s) for n, s in sorted((manifest.get('dependencies') or {}).items())]
    for requirer in sorted(entries):
        requirements += [(requirer, n, s) for n, s in sorted(entries[requirer].get('dependencies', {}).items())]
    lock_path = os.path.join(root, LOCKFILE)
    out: List[str] = []
    with work_directory() as workdir:
        fetcher = Fetcher(root, workdir)
        for requirer, name, text in requirements:
            entry = entries.get(name)
            if entry is None:
                continue                                # reported as OUTDATED
            try:
                spec = parse_spec(text, root, name, MANIFEST)
            except PkgError:
                continue                                # `til pkg install` reports it
            own = text == entry['spec']
            pinned_sha = spec.kind == 'git' and spec.rev is not None and _SHA_RE.match(spec.rev) is not None
            if not ((spec.kind == 'archive' and spec.sha256) or pinned_sha):
                continue
            bad = f"MISMATCH {name}: {requirer} requires {name} = {text}, but {LOCKFILE} resolves it to " \
                  f"{entry['resolved']}"
            try:
                source = parse_spec(entry['resolved'], root, name, lock_path)
            except PkgError as e:
                out.append(f"{bad} ({e})")
                continue
            if spec.kind == 'archive':
                if source.kind != 'archive' or source.sha256 != spec.sha256 or (own and source.url != spec.url):
                    out.append(bad)
                continue
            if source.kind != 'git' or (own and spec.path is None and source.url != spec.url):
                out.append(bad)
            elif source.rev != spec.rev:
                try:
                    commit = fetcher.git_commit(Spec('git', text, url=source.url, rev=spec.rev))
                except PkgError as e:
                    out.append(f"{bad} (and {spec.rev} could not be resolved in the repository to check "
                               f"whether it is an annotated tag pointing to that commit: {e})")
                    continue
                if commit != source.rev:
                    out.append(bad)
    return out


def cmd_verify(args, start: str) -> int:
    root, manifest = _project(start)
    lock_path = os.path.join(root, LOCKFILE)
    deps = manifest.get('dependencies') or {}
    if not os.path.exists(lock_path):
        if not deps:
            print("nothing to verify: no dependencies")
            return 0
        raise PkgError(f"no {LOCKFILE} in {root} (run `til pkg install`)")
    entries = read_lock(lock_path)['packages']
    modules = modules_dir(root)
    problems = 0
    for name in sorted(deps):
        entry = entries.get(name)
        if entry is None or entry['spec'] != deps[name]:
            locked = f"has {entry['spec']}" if entry else "does not have it"
            print(f"OUTDATED {name}: {MANIFEST} requires {deps[name]}, {LOCKFILE} {locked}")
            problems += 1
    for line in pin_mismatches(root, manifest, entries):
        print(line)
        problems += 1
    needed: Set[str] = set()                    # what til.json requires, directly or through til.lock
    todo = [n for n in deps if n in entries]
    while todo:
        n = todo.pop()
        if n not in needed:
            needed.add(n)
            todo.extend(d for d in entries[n].get('dependencies', {}) if d in entries)
    for name in sorted(set(entries) - needed):
        print(f"OUTDATED {name}: in {LOCKFILE}, but nothing in {MANIFEST} requires it any more")
        problems += 1
    for name in sorted(entries):
        entry = entries[name]
        for dep in sorted(entry.get('dependencies', {})):
            if dep not in entries:
                print(f"OUTDATED {name}: its dependency '{dep}' is not in {LOCKFILE}")
                problems += 1
        target = os.path.join(modules, name)
        if _is_link(target) or (os.path.lexists(target) and not os.path.isdir(target)):
            print(f"MODIFIED {name}: {MODULES}/{name} is not a directory installed by til pkg "
                  f"({'a symbolic link' if _is_link(target) else 'a file'})")
            problems += 1
            continue
        if not os.path.isdir(target):
            print(f"MISSING  {name}: {MODULES}/{name} does not exist")
            problems += 1
            continue
        try:
            actual = tree_hash(target, installed=True)
        except (PkgError, OSError) as e:
            print(f"MODIFIED {name}: {e}")
            problems += 1
            continue
        if actual != entry['sha256']:
            print(f"MODIFIED {name}: sha256 {actual}, {LOCKFILE} has {entry['sha256']}")
            problems += 1
        else:
            version = f" {entry['version']}" if entry.get('version') else ""
            print(f"ok       {name}{version}")
    if os.path.isdir(modules):
        for entry_name in sorted(os.listdir(modules)):
            if entry_name not in entries and not entry_name.startswith('.'):
                print(f"EXTRA    {MODULES}/{entry_name} is not in {LOCKFILE}")
                problems += 1
    if problems:
        print(_(f"til pkg verify: {problems} problem{'s' if problems != 1 else ''} "
                f"(`til pkg install` restores {MODULES}/ from {LOCKFILE})"), file=sys.stderr)
        return 1
    print(f"verified {len(entries)} package{'s' if len(entries) != 1 else ''}: {MODULES}/ matches {LOCKFILE}")
    return 0


DESCRIPTION = """\
TIL package manager: dependencies in til.json, installed into til_modules/,
pinned in til.lock (resolved sources + sha256 of every installed tree).

dependency SPECs:
  ./dir  ../dir  /abs/dir              local directory (relative to til.json)
  git+URL[#REV]                        git repository at a tag, branch or commit
  https://.../pkg.tar.gz#sha256=HEX    archive (.tar.gz .tgz .zip; also file://)
  ^1.2  ~1.2.3  1.x  *  1.2.3          version requirement (needs a registry:
                                       TIL_REGISTRY or til.json "registry")
"""

EPILOG = "Programs use packages with `import NAME`. Guide: docs/TIL_PACKAGES.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='til pkg', description=DESCRIPTION, epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-C', dest='directory', metavar='DIR',
                        help="run as if started in DIR (relative paths and TIL_REGISTRY too)")
    parser.add_argument('--version', action='version', version=f"til pkg (TIL {til.TIL_VERSION})")
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')
    p = sub.add_parser('init', help="create til.json (name, version 0.1.0, no dependencies)")
    p.add_argument('name', nargs='?', help="package name (default: the directory name)")
    p = sub.add_parser('add', help="add a dependency to til.json and install it")
    p.add_argument('name', help="package name (NAME@REQUIREMENT for the registry)")
    p.add_argument('spec', nargs='?', help="path, git+URL[#rev], archive URL#sha256=..., or version "
                                           "requirement (default: ^latest from the registry)")
    p = sub.add_parser('install', help="install the dependencies (as locked in til.lock)")
    p.add_argument('--update', action='store_true', help="re-resolve (all, or the NAMEs) ignoring til.lock")
    p.add_argument('--frozen', action='store_true', help="fail if til.lock is missing or out of date")
    p.add_argument('names', nargs='*', metavar='NAME', help="with --update: only these packages")
    p = sub.add_parser('remove', aliases=['rm'], help="remove a dependency")
    p.add_argument('name')
    sub.add_parser('list', aliases=['ls'], help="show the dependency tree")
    sub.add_parser('verify', help="check til_modules/ against the hashes in til.lock")
    sub.add_parser('help', help="show this help")
    return parser


COMMANDS = {'init': cmd_init, 'add': cmd_add, 'install': cmd_install, 'remove': cmd_remove,
            'rm': cmd_remove, 'list': cmd_list, 'ls': cmd_list, 'verify': cmd_verify}


def main(argv: List[str]) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:                     # usage errors / --help
        return e.code if isinstance(e.code, int) else 2
    if args.command in (None, 'help'):
        parser.print_help()
        return 0
    start = args.directory or os.getcwd()
    if not os.path.isdir(start):
        print(_(f"til pkg: error: no such directory: {start}"), file=sys.stderr)
        return 1
    try:
        return COMMANDS[args.command](args, start)
    except UsageError as e:
        print(_(f"til pkg: error: {e}"), file=sys.stderr)
        return 2
    except PkgError as e:
        print(_(f"til pkg: error: {e}"), file=sys.stderr)
        return 1
    except OSError as e:                        # (the system's own text: English)
        print(_(f"til pkg: error: {e}"), file=sys.stderr)
        return 1
    except RecursionError:                      # last resort: walks are iterative, JSON depth is caught
        print(_("til pkg: error: input nested too deeply to process (Python recursion limit)"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(_("til pkg: interrupted"), file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
