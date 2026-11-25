#!/usr/bin/env python3
"""
TIL language server -- compatibility entry point.

The language server is now part of the compiler: `til lsp` (src/tillsp.py,
built on the compiler's own lexer, parser and checkers in src/tills.py).
This file only starts it, so editors configured with
`python lsp/til_lsp.py` (or `~/.til/bin/til_lsp.py`) keep working.

    python lsp/til_lsp.py [--tcp PORT] [--log FILE]     # == til lsp ...

Editor setup: docs/TIL_LSP.md

Author: Alisher Beisembekov
"""

import os
import sys


def _server_dir():
    """The directory with til.py and tillsp.py: the repository (lsp/../src),
    next to this file, $TIL_HOME/bin or ~/.til/bin."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, os.pardir, 'src'), here]
    home = os.environ.get('TIL_HOME')
    if home:
        candidates += [os.path.join(home, 'bin'), home]
    candidates.append(os.path.join(os.path.expanduser('~'), '.til', 'bin'))
    for d in candidates:
        if os.path.isfile(os.path.join(d, 'tillsp.py')) and os.path.isfile(os.path.join(d, 'til.py')):
            return os.path.abspath(d)
    return None


def main():
    d = _server_dir()
    if d is None:
        sys.stderr.write("til_lsp.py: cannot find tillsp.py and til.py (looked in ../src, next to this file, "
                         "$TIL_HOME/bin and ~/.til/bin); run `til lsp` instead\n")
        return 1
    if d not in sys.path:
        sys.path.insert(0, d)
    import til  # noqa: F401  (tillsp and tills use this module)
    import tillsp
    return tillsp.main(sys.argv[1:])


if __name__ == '__main__':
    sys.exit(main())
