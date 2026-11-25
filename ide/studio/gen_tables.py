#!/usr/bin/env python3
"""
gen_tables.py - the language tables of TIL Studio, generated from src/til.py.

    python3 ide/studio/gen_tables.py            writes ide/studio/tables.json
    python3 ide/studio/gen_tables.py --check    exit 1 when tables.json is out of date
    python3 ide/studio/gen_tables.py --stdout   print the JSON

One source of truth (docs/dev/TIL_STUDIO_DESIGN.md section 0): the keywords of
every keyword language, the builtin functions (NATIVES, NATIVE_ARITY), the
builtin methods (METHODS), the types and the attribute names all come from the
compiler. The frontend embeds tables.json; the studio server serves the same
dict as `tables` in GET /api/info (build_tables() below can be imported).

Shape (every key is always present):
    version            "3.1.0"
    languages          ["en", "kk", "kk-latn", "ru"]
    default_languages  languages that are on without a #[lang: ...] pragma
    keywords           {lang: {word: TOKEN_TYPE_NAME}}   (TokenType names, e.g. "IF")
    contextual         {lang: {word: TOKEN_TYPE_NAME}}   (with/resume, localized perform/handle)
    aliases            {lang: {word: canonical builtin}} (басып_шығару -> print)
    true_words         words whose BOOL value is true
    natives            NATIVES in bytecode order
    native_arity       {name: [min, max or null]}
    methods            METHODS in bytecode order
    types              builtin type names (TYPE_MAP) + Option, Result, Some, Ok, Err
    attributes         attribute names understood inside #[...]
    energy_modes       program energy modes (#[energy: ...] before every definition)
    energy_levels      function energy levels (#[energy: low] above a function)
    key_names          --keys names -> key codes
    superinstructions  opcode names of the fused superinstructions (0x80-0xBF)
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'tables.json')


def _load_til():
    src = os.path.join(ROOT, 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    import til  # noqa: E402  (the compiler is the source of truth)
    return til


def _attribute_names(til_source):
    """Attribute names the compiler interprets: `attr == "pure"` and
    `attr.startswith("requires:")` in src/til.py, plus the two pragmas."""
    names = set(re.findall(r'\battr\s*==\s*["\']([a-z_]+)["\']', til_source))
    names |= set(re.findall(r'\battr\.startswith\(\s*["\']([a-z_]+):', til_source))
    names |= {'level', 'lang', 'energy', 'requires', 'ensures', 'pure', 'effects', 'cap'}
    names.discard('generic')            # written by the parser itself, not by users
    order = ['pure', 'effects', 'requires', 'ensures', 'cap', 'level', 'energy', 'lang',
             'inline', 'noinline', 'hot', 'cold', 'cross_level']
    return [n for n in order if n in names] + sorted(names - set(order))


def build_tables():
    til = _load_til()
    with open(til.__file__, encoding='utf-8') as f:
        til_source = f.read()
    tt = lambda t: t.name                                   # noqa: E731
    keywords = {'en': {w: tt(t) for w, t in til.ENGLISH_KEYWORDS.items()}}
    aliases = {}
    for code, (table, alias) in til.LANGUAGES.items():
        if code == 'en':
            continue
        keywords[code] = {w: tt(t) for w, t in table.items()}
        aliases[code] = dict(alias)
    contextual = {code: {w: tt(t) for w, t in table.items()}
                  for code, table in til.CONTEXTUAL_KEYWORDS.items()}
    types = list(til.TYPE_MAP.keys())
    for extra in ('Option', 'Result', 'Some', 'None', 'Ok', 'Err', 'Self'):
        if extra not in types:
            types.append(extra)
    return {
        'version': til.TIL_VERSION,
        'languages': list(til.LANGUAGES.keys()),
        'default_languages': list(til.DEFAULT_LANGUAGES),
        'keywords': keywords,
        'contextual': contextual,
        'aliases': aliases,
        'true_words': sorted(til.TRUE_WORDS),
        'natives': list(til.NATIVES),
        'native_arity': {n: [lo, hi] for n, (lo, hi) in til.NATIVE_ARITY.items()},
        'methods': list(til.METHODS),
        'types': types,
        'attributes': _attribute_names(til_source),
        'energy_modes': list(til.ENERGY_MODES.keys()),
        'energy_levels': list(til.FUNCTION_ENERGY_LEVELS),
        'key_names': dict(til.KEY_NAMES),
        'superinstructions': [name for name, value in sorted(
            ((k, v) for k, v in vars(til.Op).items() if not k.startswith('_') and isinstance(v, int) and v >= 0x80),
            key=lambda kv: kv[1])],
    }


def render(tables):
    return json.dumps(tables, ensure_ascii=False, indent=1, sort_keys=False) + '\n'


def main(argv):
    text = render(build_tables())
    if '--stdout' in argv:
        sys.stdout.write(text)
        return 0
    if '--check' in argv:
        try:
            with open(OUT, encoding='utf-8') as f:
                current = f.read()
        except OSError:
            current = ''
        if current != text:
            sys.stderr.write('ide/studio/tables.json is out of date: run python3 ide/studio/gen_tables.py\n')
            return 1
        return 0
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print(f'wrote {os.path.relpath(OUT, ROOT)} ({len(text)} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
