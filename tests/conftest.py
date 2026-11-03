"""pytest configuration shared by the TIL test-suite."""

import os

# TIL_LANG chooses the language of TIL's diagnostics (docs/TIL_I18N.md). The
# tests expect English unless they set the language themselves, so a value
# exported in the developer's shell must not leak into them.
os.environ.pop('TIL_LANG', None)
