# TIL Packages — `til pkg`

TIL 3.1 has a package manager. A project lists its dependencies in `til.json`;
`til pkg install` fetches them into `til_modules/` and pins exactly what was
installed in `til.lock`; programs use them with an ordinary `import`. It works
for every backend — native, the Python VM and the C VM — because packages are
plain TIL source that the compiler merges at build time.

```
$ til pkg init                                # til.json
$ til pkg add geo ../libs/geo                 # a local directory
$ til pkg add colors git+https://example.com/colors.git#v1.2.0
$ til pkg add mathx https://example.com/mathx-1.0.tar.gz#sha256=9f86d0...
$ til pkg list
myapp 0.1.0 (/home/me/myapp)
|-- colors 1.2.0  git+https://example.com/colors.git#v1.2.0  [git+https://example.com/colors.git#5e1f...]
|-- geo 1.2.0  ../libs/geo
|   `-- units 0.3.0  ../units  [../libs/units]
`-- mathx 1.0.0  https://example.com/mathx-1.0.tar.gz#sha256=9f86d0...
$ til run main.til                            # import geo, import colors, ...
```

Nothing is installed globally and nothing runs at install time: a package is
only files.

---

## 1. Commands

| Command | What it does |
|---|---|
| `til pkg init [NAME]` | Create `til.json` in the current directory: `{"name", "version": "0.1.0", "dependencies": {}}`. `NAME` defaults to the directory name (made an identifier: `my-app` → `my_app`). Refuses to overwrite an existing `til.json`. |
| `til pkg add NAME SPEC` | Record `NAME: SPEC` in `til.json` and install it (re-resolving that package; the rest stays as locked). Replaces an existing spec for `NAME`. A relative path is relative to the directory the command runs in, as in any shell command, and is stored relative to the project root: `til pkg add geo ../vendor/geo` in `proj/src/` records `"./vendor/geo"`. |
| `til pkg add NAME@REQ` / `til pkg add NAME` | Registry forms: `add geo@^1.2`; without a spec the newest release in the registry is recorded as `^X.Y.Z`. |
| `til pkg install` | Install everything in `til.json` (recursively). Packages already in `til.lock` with an unchanged spec are reproduced exactly; new or changed ones are resolved (and so is an entry that contradicts an integrity pin of `til.json`, §4). Writes `til.lock`. |
| `til pkg install --update [NAME...]` | Ignore `til.lock` and resolve again (everything, or only the named packages). |
| `til pkg install --frozen` | For CI: install exactly `til.lock`; fail if it is missing or does not match `til.json` — checked before anything is fetched for a package. Never writes `til.lock`. |
| `til pkg remove NAME` (`rm`) | Drop `NAME` from `til.json`, remove packages nothing needs any more. |
| `til pkg list` (`ls`) | Dependency tree from `til.lock` (see below). |
| `til pkg verify` | Re-hash every directory in `til_modules/` and compare with `til.lock`; also check that `til.lock` matches `til.json`, including its integrity pins (§4). Exit code 1 on any difference. |
| `til pkg help` | Usage (also `til pkg --help`, `til pkg COMMAND -h`). |

All commands except `init` find the project by looking for `til.json` in the
current directory and then in its parents, so they work from any subdirectory.
`-C DIR` runs as if started in `DIR` (`til pkg -C ../app list`): the project is
searched from `DIR`, and relative paths on the command line and a relative
`TIL_REGISTRY` are relative to `DIR`.

`til pkg` refuses to run inside an installed package (`PROJECT/til_modules/NAME/`
or below, whether or not the package has a `til.json` of its own): installed
copies are only ever replaced as a whole, and their own dependencies live in
`PROJECT/til_modules/`.

Exit codes: `0` success, `1` error (message on stderr, prefixed `til pkg:
error:`), `2` command-line usage error (unknown command or option,
`install NAME` without `--update`, `--update` with `--frozen`). When
resolution or fetching fails, nothing in the project is changed — not
`til.json`, not `til.lock`, not `til_modules/`.

### `til pkg list`

```
app 0.1.0 (/path/to/app)
|-- both 1.0.0  ../libs/both
|   `-- units 0.3.0  ../units  [../libs/units]
`-- geo 1.2.0  ../libs/geo
    `-- units 0.3.0  ../units  [../libs/units]
```

Each line is `name version  spec` where *spec* is what the parent declared;
`[...]` is the resolved source when it differs (a commit SHA, a path relative
to the project). A subtree printed earlier is marked `(*)`. Problems are shown
inline: ``(not installed: run `til pkg install`)``, ``(missing from
til_modules/: ...)``, ``(til.lock has ...: run `til pkg install`)``.

### `til pkg verify`

```
ok       geo 1.2.0
MODIFIED units: sha256 0be2..., til.lock has 8155...
MODIFIED shapes: /app/til_modules/shapes: 'til_modules' was added after installation (til pkg never installs til_modules directories)
MISSING  colors: til_modules/colors does not exist
EXTRA    til_modules/intruder is not in til.lock
OUTDATED extra: til.json requires ../libs/extra, til.lock does not have it
OUTDATED old: in til.lock, but nothing in til.json requires it any more
MISMATCH mathx: app requires mathx = https://example.com/mathx-1.0.tar.gz#sha256=9f86..., but til.lock resolves it to https://example.org/x.tar.gz#sha256=0be2...
```

`til pkg install` repairs all of these: it restores every package from the
locked source and removes extra directories. (`OUTDATED` means `til.lock`
must be rewritten, which `install --frozen` refuses to do.) `MISMATCH` means
that `til.lock` contradicts an integrity pin of a `til.json` (§4) — `til pkg`
never writes such a lock, so it was edited; `install` resolves that package
from `til.json` again, `install --frozen` refuses.

---

## 2. `til.json`

```json
{
  "name": "myapp",
  "version": "0.1.0",
  "dependencies": {
    "geo": "../libs/geo",
    "colors": "git+https://example.com/colors.git#v1.2.0",
    "mathx": "https://example.com/mathx-1.0.tar.gz#sha256=9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "json": "^1.2"
  },
  "main": "src/myapp.til",
  "registry": "https://example.com/til-index.json"
}
```

| Field | Meaning |
|---|---|
| `name` | Package name — the name programs `import`. Must be an identifier: letters (any script, e.g. `геометрия`), digits and `_`, not starting with a digit, at most 64 characters, not a TIL keyword or a Windows device name (`con`, `nul`, `com1`...), and not `til_modules` in any letter case (that is the directory packages are installed in). So a name can never contain `/`, `\`, `.` or `..`. |
| `version` | Semantic version `MAJOR.MINOR.PATCH[-pre][+build]`. |
| `dependencies` | `{"name": "spec"}`; see §3. |
| `main` | (packages) The file `import NAME` loads: a relative path inside the package ending in `.til` (no `..`; `/` and `\` both separate directories, on every OS). Default: `NAME.til`, then `lib.til`, then `mod.til`. |
| `registry` | (projects) Path or URL of a registry index (§5). Relative paths are relative to this `til.json`. |

All fields are optional for a *dependency*: a directory or repository without
`til.json` is a package named after the dependency, without version or
dependencies. If a dependency has a `til.json` with a `name`, the name must be
the one it is required as (installing under another name is not supported).

`til pkg add`/`remove` rewrite `til.json` with 2-space indentation and
dependencies sorted by name; other fields are kept as they are.

---

## 3. Dependency specs

| Spec | Kind |
|---|---|
| `./libs/geo`, `../geo`, `/abs/geo`, `~/geo` | **Local directory.** Relative to the directory of the `til.json` that declares it — for a package that is itself a local directory, that directory as it was written: `..` goes back up the written path, not up from the target of a symbolic link. Use `/` (also on Windows) so projects stay portable. `til.lock` records the directory in the form it was written — relative to the project root, absolute, or `~/...` (a relative path declared by such a package inherits its form), with symbolic links on the way kept (`~/libs/geo` stays `~/libs/geo` when `~/libs` is a link) — so the lock stays valid wherever the project is checked out. |
| `git+URL` / `git+URL#REV` | **Git repository.** `URL` is any git URL: `https://`, `ssh://`, `git://`, `file://`, `user@host:path`, or a local repository path (relative paths as for directories). `REV` is a tag, a branch (also non-default branches) or a full or abbreviated commit SHA; without `#REV` the remote's default branch (`HEAD`). The resolved **commit SHA** is recorded in `til.lock`. A full SHA is an integrity pin (§4); it may also be an annotated tag object's SHA (what `git rev-parse v1.0` prints), which names the commit the tag points to — since `til.lock` records that commit, every install then checks the tag in the repository (a warning says so; pin the commit SHA to install offline). The package is the files of that commit **exactly as committed** (see "Git packages" below). |
| `https://host/p.tar.gz#sha256=HEX` | **Archive**: `.tar.gz`, `.tgz` or `.zip`, also `file:///path/p.zip#sha256=...`. The sha256 of the archive file is **required** and is checked *before* anything is extracted — also when `til.lock` names the archive (§4). Leave it out and the error message prints the current hash for you to paste. Plain `http://` is refused. If the archive holds exactly one top-level directory (as GitHub release archives do), that directory is the package. |
| `^1.2`, `~1.2.3`, `1.x`, `1.2`, `*`, `1.2.3`, `>=1.2 <2`, `^1 \|\| ^3`, `NAME@^1.2` | **Version requirement**, resolved through the registry (§5). |

Version requirements follow npm:

| Requirement | Means |
|---|---|
| `^1.2.3` / `^1.2` / `^1` | `>=1.2.3 <2.0.0` / `>=1.2.0 <2.0.0` / `>=1.0.0 <2.0.0` |
| `^0.2.3`, `^0.0.3` | `>=0.2.3 <0.3.0`, `>=0.0.3 <0.0.4` (the first non-zero part is fixed) |
| `~1.2.3` / `~1.2` / `~1` | `>=1.2.3 <1.3.0` / `>=1.2.0 <1.3.0` / `>=1.0.0 <2.0.0` |
| `1.x`, `1.*`, `1` / `1.2.x`, `1.2` | `>=1.0.0 <2.0.0` / `>=1.2.0 <1.3.0` |
| `*`, `x`, empty | any version |
| `1.2.3`, `=1.2.3` | exactly that version (build metadata ignored) |
| `>=1.2 <2`, `>=1.2, <2` | all comparators (`<`, `<=`, `>`, `>=`, `=`) must hold; partial versions as in npm (`>1.2` = `>=1.3.0`, `<=1.2` = `<1.3.0`) |
| `A \|\| B` | either |

Pre-releases (`2.1.0-beta.1`) are chosen only when a comparator names a
pre-release of the same `MAJOR.MINOR.PATCH` (`^2.1.0-beta`); `*` and `^2` never
pick one. Hyphen ranges (`1.2 - 2.0`) are not supported and are an error.

**Where path dependencies are allowed.** Your project and packages that are
themselves local directories may declare path dependencies (relative to their
own `til.json`). Packages fetched from git, an archive or a registry may not —
such a path would point into whatever machine installs the package — and are
rejected with an explicit error. They can use git, archive and registry
dependencies.

**Git packages.** `til pkg` never runs `git checkout`: it reads the blobs of
the commit (`git ls-tree`, `git cat-file`) and writes the files itself. So
nothing that depends on the installing machine's git configuration changes
them — no line-ending conversion (`core.eol`, `core.autocrlf`, `text` /
`eol` attributes in `.gitattributes`), no smudge filters (Git LFS files stay
pointer files), no `ident` expansion — and the tree hash in `til.lock` is the
same on Linux, macOS and Windows. File modes are not kept (a package is TIL
source; the hash ignores modes anyway). File names follow the rules for
archives (§7): no `..`, no `\`, `:` or control characters, UTF-8. A symbolic
link in the repository is an error, a submodule is skipped with a warning.

---

## 4. `til_modules/` and `til.lock`

Packages are installed **flat**: every package, direct or transitive, gets one
directory `til_modules/NAME/` next to `til.json`. Consequently each package
name has exactly one version in a project. Installed copies never contain
`.git`, `.hg`, `.svn` or nested `til_modules/` directories; `verify` reports
one that appears later as a modification and `install` replaces that copy.

`til_modules/` belongs to `til pkg`: it removes whatever it did not install
there. It must be a real directory inside the project. If `til_modules` is a
symbolic link (or a Windows junction) — for example one committed to a
repository you cloned — every command refuses to run instead of deleting the
files the link points to; remove the link (not its target). Files are created
with the usual permissions (0666/0777 minus the umask), so other users can
read installed packages; `til.json` keeps the permissions it had.

Commit `til.json` and `til.lock`; add `til_modules/` to `.gitignore`.

### Resolution

Dependencies are resolved breadth first, the dependencies of each `til.json`
in name order, so **your project's own declarations are seen first**.

* A package required a second time must be satisfied by the copy already
  chosen: the same directory, the same git commit (a tag and its SHA are the
  same commit), the same archive hash — or, for a version requirement, a
  version in range. A version requirement is satisfied by a package from any
  source that has a matching `version`, so declaring `geo` as a local
  directory in your project also satisfies a dependency's `geo: ^1.2` when
  the directory's version is in range.
* A second git requirement is checked without the network when it can be: a
  full commit SHA equal to the chosen commit is accepted directly (another
  full SHA is resolved in the repository — it may be an annotated tag object
  pointing to that commit), and a requirement that `til.lock`
  already records for that package (the requiring package's `dependencies`)
  is pinned by the lock like the package itself — a branch or `HEAD`
  requirement does not follow upstream pushes until `til pkg install
  --update`. Any other git requirement is resolved in the repository.
* Otherwise it is a **conflict** and nothing is installed:

  ```
  til pkg: error: conflicting requirements for package 'c':
    a requires c = ../c_one (= ../libs/c_one)
    b requires c = ../c_two (= ../libs/c_two)
  the first resolved to c 1.0.0 from ../libs/c_one; til_modules/ holds one copy of each package, so these requirements must agree
  ```

* Two registry requirements are combined: if the version picked for the first
  one does not satisfy a later one, the resolver starts again and picks the
  highest version satisfying both (`^1.0` and `~1.2` → `1.2.x`). When no
  version satisfies all of them, the error lists every requirement and the
  available versions. (This is not a full backtracking solver: a constraint
  learned in one attempt is kept even if the package that imposed it is no
  longer part of the graph afterwards.)
* **Cycles** are an error: `dependency cycle: x -> y -> x` (also a package
  that depends on your project's own name).
* Two packages whose names differ only in case are an error (they would
  collide on Windows and macOS).

### `til.lock`

```json
{
  "lockVersion": 1,
  "packages": {
    "colors": {
      "spec": "git+https://example.com/colors.git#v1.2.0",
      "resolved": "git+https://example.com/colors.git#5e1f0c2a9d3b7e8f6a4c1d2e3f4a5b6c7d8e9f01",
      "version": "1.2.0",
      "sha256": "c662b7dec6bed51b60019abe2f2e4211a851849d5700222ca1d274aa14af42ab",
      "dependencies": {}
    },
    "units": {
      "spec": "../units",
      "resolved": "../libs/units",
      "version": "0.3.0",
      "sha256": "ab05e249a2daec366e63c8e6581c5a5164a08109dfe67d4e9b803a4b34fdb993",
      "dependencies": {}
    }
  }
}
```

Packages are sorted by name and every entry has the fields in this order, so
the file is byte-for-byte reproducible (UTF-8, `\n`, 2-space indentation).

| Field | Meaning |
|---|---|
| `spec` | The spec the package was first required with (your project's, if you declare it). |
| `resolved` | The concrete source: a directory (relative to the project — `./x`, `../x` — or, when it was declared that way, absolute or `~/x`), `git+URL#<full commit SHA>`, or `URL#sha256=<archive hash>`. For registry packages, the source the index named for the chosen version. |
| `version` | The package's version (`null` if it has none). |
| `sha256` | The **tree hash** of the installed files (below). |
| `dependencies` | The package's own `til.json` dependencies. |

**Reproducing.** `til pkg install` with a `til.lock` reuses every entry whose
spec is unchanged: the same resolved source (git: the same commit even if the
branch or tag moved; registry: the same version even if newer ones were
published; the registry is not even read), and the result must have the same
tree hash. If `til_modules/NAME` already holds exactly those files it is kept
as it is, so a repeated install needs neither the network nor the sources
(this includes git packages required by several packages, see Resolution).
When a source changed under the lock — a directory was edited, an archive
replaced at the same URL — the install stops with a hash error that names the
package and the remedy (`til pkg install --update NAME`). Local directories
are *copied*: after editing one, run `til pkg install --update NAME`.

**Integrity pins win over `til.lock`.** An archive's `#sha256=` and a full
commit SHA — in your `til.json` or in a package's — are never taken on
`til.lock`'s word: a lock entry that resolves such a requirement to another
archive (URL or hash), repository or commit is not used. `til pkg` never
writes such an entry, so it means `til.lock` was edited (lock diffs are rarely
reviewed): `install` warns and resolves the package from `til.json` again,
`install --frozen` fails, and `verify` reports `MISMATCH`. Tags, branches,
abbreviated SHAs and version requirements are not pins — for them, `til.lock`
is the pin.

### Tree hash

`sha256` is the sha256 of the text made of one line per file

```
<sha256 of the file contents, hex>  <path relative to the package, '/'-separated>\n
```

with the files sorted by the UTF-8 bytes of their paths — exactly the output
format of `sha256sum`. Only paths and contents count (not timestamps,
permissions or empty directories), so it is the same on every OS. To check a
package by hand:

```
cd til_modules/geo
find . -type f | sed 's|^\./||' | LC_ALL=C sort |
    while IFS= read -r f; do sha256sum -- "$f"; done | sha256sum
```

A package consists of regular files and directories only: symbolic links,
devices and similar entries are refused (in local directories, archives and
git repositories), so an installed package can never point outside itself.

---

## 5. Registry

A registry is a JSON index that maps names to versions to sources:

```json
{
  "packages": {
    "geo": {
      "1.0.0": "git+https://example.com/geo.git#v1.0.0",
      "1.2.0": "https://example.com/geo-1.2.0.tar.gz#sha256=...",
      "1.5.0": {"source": "./geo-1.5.0"}
    }
  }
}
```

* The `"packages"` wrapper is optional (`{"geo": {...}}` works too) — use it if
  you have a package called `packages`.
* A source is a git spec, an archive URL with `#sha256=`, or (only in an index
  that is a local file) a directory relative to the index file. A source given
  as `{"source": ...}` is the same.
* If the package has a `til.json` version, it must equal the index version.
* The index is found through the `TIL_REGISTRY` environment variable (which
  wins) or `"registry"` in the project's `til.json`; a path (a directory means
  `DIR/index.json`), a `file://` URL or an `https://` URL. A relative
  `TIL_REGISTRY` is relative to the directory the command runs in (`-C DIR`).
  Only the project's registry is used, also for the dependencies of
  dependencies. It is read only when a version requirement has to be resolved.

`til pkg add NAME` (no spec) records `^` + the newest release in the index.

---

## 6. Importing packages

```
import geo                      # til_modules/geo: "main", else geo.til, lib.til, mod.til
import geo as g                 # g.area(...)
from geo import area, Rect
import geo.shapes.square        # til_modules/geo/shapes/square.til (or .../square/mod.til)
```

**In your program** (any file that is not inside `til_modules/`):

* `import NAME` first searches the importing file's directory, the current
  directory and `$TIL_PATH`. Then, if `NAME` is a dependency in your project's
  `til.json`, it is taken from the installed package — only from there: if it
  is not installed, the import fails with a hint. Any other `NAME` is looked
  up in the standard library and then in installed packages. So a local
  `geo.til` next to your program shadows the package `geo`, and a package you
  declare comes before the standard library: `til pkg add math ../my-math`
  replaces the standard `math` module in your program (not in packages that do
  not declare `math` themselves; they keep the standard one).
* Packages are looked up in `til_modules/` of the importing file's directory,
  then of its parent, grandparent, ... (nearest first), **up to the nearest
  project root** — the first directory with a `til.json` (or `til.lock`) —
  and never above it, so an unrelated `til_modules/` in a parent directory
  (say `/tmp/til_modules`) is never used by a project. Programs in
  subdirectories of a project (`src/`, `tests/`) find the project's packages,
  wherever `til` is run from. (A script that belongs to no project searches
  up to the file-system root.)

**In a package** (a file in `PROJECT/til_modules/PKG/`, where `PROJECT` has a
`til.json` or `til.lock`):

* `import NAME` searches the importing file's directory (the package's own
  modules: `import helper` finds its `helper.til`); then a dependency declared
  in the package's `til.json` comes from **`PROJECT/til_modules/` only** — the
  flat directory that `til.lock` describes; any other `NAME` from the standard
  library, then `PROJECT/til_modules/`. The current directory and `$TIL_PATH`
  are not searched, and a `til_modules/` inside an installed package is never
  used. So a package always gets exactly the dependencies `til.lock` pins,
  wherever `til` runs and whatever modules your project (or `$TIL_HOME/stdlib`,
  `~/.til/stdlib`) has: a `units.til` of your own does not replace the package
  `units` that `geo` depends on.

**Both:**

* Entry file of `import NAME`: `til.json` `"main"` if present (if that file
  does not exist the import fails — it never falls back), else `NAME.til`,
  `lib.til`, `mod.til`.
* `import NAME.a.b` loads `a/b.til` or `a/b/mod.til`, looked up in the
  package root and then in the directory of the entry file (so a package with
  `"main": "src/geo.til"` can keep sub-modules in `src/`). Qualified calls use
  the last component: `b.f()`.
* If an import is not found, the compiler adds a hint when it can tell why:
  ``hint: 'geo' is a dependency in /path/til.json but is not installed - run `til pkg install` in /path``,
  or ``hint: the package 'geo' (/path/til_modules/geo) has no module 'shapes' (shapes.til or shapes/mod.til)``.

Imported definitions are merged into the program at compile time, so a program
that uses packages builds and runs the same way natively (`til run`,
`til build`), on the Python VM (`til run --vm`) and on the C VM (`til run
--cvm`, `til build --target vm` / `c-vm` — the `.tbc` file contains the
package code, the device needs no `til_modules/`).

**One namespace.** TIL has no module-private names yet: everything a
program brings in — its own definitions, its modules, every package, directly
or through other packages — shares one namespace, and merged code sees all of
it. `import NAME` brings in every top-level definition of `NAME`; `from NAME
import a, b` brings in `a`, `b` and every definition of `NAME` (and of what
it imports) that they use, transitively — helper functions, constants,
structs and their methods. A top-level name (function, `const`/`let`/`var`,
struct, enum, trait, type alias, capability) or a method of a type that two
different files bring in is a compile error that names both files, on every
backend and in `til check`:

```
error[E-IMPORT]: const 'SCALE' is defined in two files: til_modules/a/a.til:1 and til_modules/b/b.til:1
 --> til_modules/b/b.til:1:1
```

So code of one file never silently uses another file's definition. Two
packages that both define the same name — even an internal `helper()` —
cannot both be imported with `import NAME`; package authors should prefix
internal names with the package name (`geo_helper`). Names that a `from`
import does not bring in never clash: `from math import gcd` next to your own
`factorial()` is fine.

---

## 7. Security

* Package names are identifiers, so they cannot contain path separators or
  `..`; names from `til.json` files of dependencies and from `til.lock` are
  validated the same way before they are used as directory names.
* `til_modules/` is never used through a symbolic link or junction (§4):
  `til pkg` deletes what it did not install there, and a link committed to a
  repository would otherwise make `til pkg install` delete the files it points
  to. Links inside `til_modules/` are removed themselves, never followed.
* Archives: the sha256 is verified before extraction. Extraction writes only
  new regular files and directories below a fresh directory; absolute paths,
  drive letters, `..`, `:` in names, symbolic links, hard links, devices,
  duplicate entries and entries colliding with each other are refused (Python's
  `tarfile`/`zipfile` extraction functions are not used). A damaged archive is
  an error: zip members are checked against their CRC-32, a `.tar.gz` against
  the gzip trailer (CRC-32 and length of the whole stream). Size limits:
  256 MB per download and per package, 100 000 files — exceeding them is an
  error, never a silent truncation.
* Git: always run without a shell; `--` separates options from the user's
  URL; revisions must look like revisions (`[A-Za-z0-9_][A-Za-z0-9_./+@-]*`, no
  `..`) and are passed after `--end-of-options` (git ≥ 2.24); the `ext::`
  transport is disabled and URLs containing `::` are refused; credential
  prompts are disabled (`GIT_TERMINAL_PROMPT=0`); `GIT_DIR`,
  `GIT_WORK_TREE`, `GIT_INDEX_FILE` and similar variables of the caller (for
  example inside git hooks) are not passed on. Files are written from the
  committed blobs by `til pkg` itself (§3, "Git packages"): no checkout, so no
  filter command configured on the machine runs, and the tree hash does not
  depend on any git setting.
* Downloads only over `https://` (or `file://`), and every redirect must stay
  on `https://` (the registry index has no hash of its own; its integrity
  rests on TLS).
* Nothing a package contains is executed by `til pkg`.

---

## 8. Limits and non-goals

* One version per package name in a project (flat `til_modules/`); there is
  no nesting of conflicting versions.
* Installing a package under a different name (aliases) is not supported.
* All packages share the program's single namespace (§6, "One namespace").
* Git submodules are not fetched (a warning names them); Git LFS files are
  installed as the pointer files that are committed, never as their content.
  Repositories with symbolic links, or file names with `\`, `:` or control
  characters, cannot be installed (they would not install the same way on
  every OS).
* Path dependencies inside non-local packages are rejected (§3).
* No publishing command and no hosted registry: a registry is any JSON index
  you serve (or keep in a directory).
* A local registry index referring to directories writes those directories
  into `til.lock` relative to the project — fine on one machine, but for a
  team the index should name git or archive sources.
* Requires git ≥ 2.24 for git dependencies; nothing else beyond Python.
