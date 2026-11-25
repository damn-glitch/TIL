#!/usr/bin/env python3
"""Generate TIL's editor syntax definitions from the compiler's own tables.

    python3 editors/vscode/scripts/gen_grammar.py           # (re)write the files
    python3 editors/vscode/scripts/gen_grammar.py --check   # exit 1 if one is out of date

One source of truth: the keywords of every keyword language (src/til.py:
ENGLISH_KEYWORDS, LANGUAGES kk / kk-latn / ru, CONTEXTUAL_KEYWORDS), the
localized builtin names (LANGUAGES aliases), the builtins (NATIVES,
BUILTIN_TYPES), the primitive types (TYPE_MAP) and the attribute names
(src/tills.py ATTRIBUTE_DOCS). Written files:

    editors/vscode/syntaxes/til.tmLanguage.json   VS Code, and every TextMate-grammar editor
    editors/vscode/language-configuration.json    VS Code: comments, brackets, indentation rules
    editors/vim/til.vim                           Vim / Neovim
    editors/sublime/TIL.sublime-syntax            Sublime Text

Like the compiler, Russian and Latin-script Kazakh keywords are only keywords
in files with a `#[lang: ru]` / `#[lang: kk-latn]` pragma (TextMate and
Sublime switch to them from the pragma line on; Vim when the file is loaded),
and the TIL 3.1 effect words `with`, `resume` and the localized `perform` /
`handle` only where they can be keywords (docs/TIL_EFFECTS.md).

Python 3.9+, standard library only.
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)                                   # editors/vscode
EDITORS = os.path.dirname(EXT)                                # editors
REPO = os.path.dirname(EDITORS)
sys.path.insert(0, os.path.join(REPO, 'src'))

import til  # noqa: E402

try:
    import tills  # noqa: E402   (attribute names; tills imports til)
    ATTRIBUTES = sorted(tills.ATTRIBUTE_DOCS)
except Exception:  # pragma: no cover - tills is part of the same checkout
    ATTRIBUTES = sorted(['pure', 'effects', 'cap', 'requires', 'ensures', 'level', 'energy', 'lang'])

TT = til.TokenType
GENERATOR = "editors/vscode/scripts/gen_grammar.py"

TM_PATH = os.path.join(EXT, 'syntaxes', 'til.tmLanguage.json')
LANGCONF_PATH = os.path.join(EXT, 'language-configuration.json')
VIM_PATH = os.path.join(EDITORS, 'vim', 'til.vim')
SUBLIME_PATH = os.path.join(EDITORS, 'sublime', 'TIL.sublime-syntax')

LANGS = ('en', 'kk', 'kk-latn', 'ru')
OPT_IN = ('ru', 'kk-latn')                 # keyword languages a #[lang: ...] pragma enables
assert tuple(til.DEFAULT_LANGUAGES) == ('en', 'kk'), "the default keyword languages changed"

# Highlighting groups of the keyword token types:
# (key, TextMate / Sublime scope, Vim group, token types)
GROUPS: List[Tuple[str, str, str, Tuple[str, ...]]] = [
    ('conditional', 'keyword.control.conditional.til', 'tilConditional', ('IF', 'ELSE', 'ELIF', 'MATCH')),
    ('loop', 'keyword.control.loop.til', 'tilRepeat', ('FOR', 'WHILE', 'LOOP', 'IN')),
    ('flow', 'keyword.control.flow.til', 'tilStatement', ('RETURN', 'BREAK', 'CONTINUE')),
    ('recovery', 'keyword.control.recovery.til', 'tilException', ('RETRY', 'TIMEOUT', 'FALLBACK')),
    ('effect', 'keyword.control.effect.til', 'tilEffect', ('PERFORM', 'HANDLE', 'WITH', 'RESUME')),
    ('contract', 'keyword.other.contract.til', 'tilContract', ('ASSERT', 'INVARIANT', 'INTENT')),
    ('import', 'keyword.control.import.til', 'tilInclude', ('IMPORT', 'FROM', 'AS')),
    ('declaration', 'storage.type.til', 'tilStorage',
     ('FN', 'LET', 'VAR', 'CONST', 'STRUCT', 'ENUM', 'IMPL', 'TRAIT', 'TYPE', 'CAP')),
    ('modifier', 'storage.modifier.til', 'tilModifier', ('MUT', 'PUB')),
    ('logical', 'keyword.operator.logical.til', 'tilLogical', ('AND', 'OR', 'NOT')),
    ('self', 'variable.language.self.til', 'tilSelf', ('SELF',)),
    ('boolean', 'constant.language.boolean.til', 'tilBoolean', ('BOOL',)),
]
GROUP_OF = {tt: g for g in GROUPS for tt in g[3]}

# A declaration keyword and the scope of the name after it.
DECL_NAMES = (('STRUCT', 'entity.name.type.struct.til'), ('ENUM', 'entity.name.type.enum.til'),
              ('TRAIT', 'entity.name.type.trait.til'), ('TYPE', 'entity.name.type.alias.til'),
              ('CAP', 'entity.name.type.capability.til'), ('FN', 'entity.name.function.til'))
# Lines that open an indented block start with one of these (indentation rules).
BLOCK_STARTERS = ('IF', 'ELSE', 'ELIF', 'FOR', 'WHILE', 'LOOP', 'MATCH', 'STRUCT', 'ENUM', 'IMPL', 'TRAIT',
                  'HANDLE', 'RETRY', 'TIMEOUT', 'INTENT', 'FN')
# ... or end with one of these after `=` / `return` (`let r = retry(3)`, `return match s`).
BLOCK_EXPRESSIONS = ('MATCH', 'HANDLE', 'RETRY', 'TIMEOUT')
GENERIC_TYPES = ('Option', 'Result')      # the parser's generic type names (parse_type)
TODO_WORDS = ('TODO', 'FIXME', 'XXX', 'NOTE', 'HACK')


# ═══════════════════════════════════════════════════════════════════════════
#                                  TABLES
# ═══════════════════════════════════════════════════════════════════════════

class Tables:
    """Everything the grammars need, read from til.py / tills.py."""

    def __init__(self):
        # lang -> token type name -> sorted words (hard keywords)
        self.keywords: Dict[str, Dict[str, List[str]]] = {}
        # lang -> token type name -> words of the contextual keywords
        self.contextual: Dict[str, Dict[str, List[str]]] = {}
        # lang -> sorted localized builtin names
        self.aliases: Dict[str, List[str]] = {}
        tables = {'en': til.ENGLISH_KEYWORDS}
        tables.update({lang: til.LANGUAGES[lang][0] for lang in LANGS if lang != 'en'})
        for lang in LANGS:
            self.keywords[lang] = self._by_type(tables[lang], f"keywords of '{lang}'")
            self.contextual[lang] = self._by_type(til.CONTEXTUAL_KEYWORDS.get(lang, {}),
                                                  f"contextual keywords of '{lang}'")
            self.aliases[lang] = sorted(til.LANGUAGES[lang][1]) if lang != 'en' else []
        for lang in LANGS:
            for name in self.contextual[lang]:
                if name not in ('WITH', 'RESUME', 'PERFORM', 'HANDLE'):
                    raise SystemExit(f"{GENERATOR}: unknown contextual keyword type {name}")
        self.natives = list(til.NATIVES)
        self.casts = sorted(n for n in self.natives if n in til.TYPE_MAP)       # int(x), str(x) ...
        self.builtins = sorted(n for n in self.natives if n not in til.TYPE_MAP)
        self.constructors = sorted(name for name, _fields in til.BUILTIN_TYPES)  # Some Ok Err
        self.types = sorted(til.TYPE_MAP)
        self.attributes = list(ATTRIBUTES)

    @staticmethod
    def _by_type(table, what: str) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for word, tt in table.items():
            if tt.name not in GROUP_OF:
                raise SystemExit(f"{GENERATOR}: no highlighting group for the keyword type {tt.name} "
                                 f"({what}: '{word}'); add it to GROUPS")
            out.setdefault(tt.name, []).append(word)
        return {k: sorted(set(v)) for k, v in sorted(out.items())}

    def words(self, lang: str, types=None) -> List[str]:
        """Hard keywords of a language (of some token types)."""
        out = []
        for name, words in self.keywords[lang].items():
            if types is None or name in types:
                out.extend(words)
        return sorted(set(out))

    def group_words(self, lang: str, group) -> List[str]:
        return self.words(lang, group[3])

    def all_words(self, langs, types) -> List[str]:
        out = set()
        for lang in langs:
            out.update(self.words(lang, types))
            for name, words in self.contextual[lang].items():
                if name in types:
                    out.update(words)
        return sorted(out)


# ═══════════════════════════════════════════════════════════════════════════
#                       REGULAR EXPRESSIONS (Oniguruma)
# ═══════════════════════════════════════════════════════════════════════════

WORD = r'[\p{L}\p{N}_]'                  # an identifier character: str.isalnum() or '_'
IDENT = r'[\p{L}_][\p{L}\p{N}_]*'
NB = r'(?<![\p{L}\p{N}_])'               # no identifier character before
NE = r'(?![\p{L}\p{N}_])'                # ... or after


def alt(words) -> str:
    """An alternation, longest first (so no word is cut by a shorter one)."""
    return '|'.join(re.escape(w) for w in sorted(set(words), key=lambda w: (-len(w), w)))


def word_re(words) -> str:
    return NB + '(?:' + alt(words) + ')' + NE


# where a contextual effect word is a keyword (approximation of Parser.contextual_start)
CONTEXT_LOOKAHEAD = {
    # `with` after the body of a handle: alone on its line, or `with Name(x) => ...`
    'WITH': r'(?=[ \t]*(?:#.*)?$|[ \t]+' + IDENT + r'[ \t]*(?:\(|=>))',
    # resume(v) inside an effect clause
    'RESUME': r'(?=[ \t]*\()',
    # a localized perform: the effect name follows on the same line
    'PERFORM': r'(?=[ \t]+[\p{L}_])',
    # a localized handle: an indented body follows, or an expression on the same line
    'HANDLE': r'(?=[ \t]*(?:#.*)?$|[ \t]+[\p{L}\p{N}_"\'])',
}


def pragma_has(code: str) -> str:
    """Lookahead: a `#[lang: ...]` pragma (at the current position) that names `code`.
    Codes are split at commas, stripped and lower-cased, as detect_languages() does."""
    return (r'(?=#\[[ \t]*lang[ \t]*:(?:[^\],\n]*,)*[ \t]*(?i:' + re.escape(code) +
            r')[ \t]*[,\]])')


PRAGMA_LINE = r'^[ \t]*'
PRAGMA_BODY = r'(#\[)[ \t]*(lang)[ \t]*(:)([^\]\n]*)(\])'


# ═══════════════════════════════════════════════════════════════════════════
#                               TEXTMATE (VS Code)
# ═══════════════════════════════════════════════════════════════════════════

def tm_keyword_patterns(t: Tables, lang: str) -> List[dict]:
    pats: List[dict] = []
    kws = t.keywords[lang]
    for tt, scope in DECL_NAMES:
        words = kws.get(tt)
        if words:
            pats.append({'match': NB + '(' + alt(words) + r')[ \t]+(' + IDENT + ')' + NE,
                         'captures': {'1': {'name': 'storage.type.til'}, '2': {'name': scope}}})
    for tt, words in t.contextual[lang].items():
        pats.append({'name': 'keyword.control.effect.til',
                     'match': NB + '(?:' + alt(words) + ')' + NE + CONTEXT_LOOKAHEAD[tt]})
    for g in GROUPS:
        words = t.group_words(lang, g)
        if words:
            pats.append({'name': g[1], 'match': word_re(words)})
    return pats


def tm_pragma(begin_lookaheads: List[str], context: str) -> dict:
    """A rule that starts at a `#[lang: ...]` pragma and lasts to the end of the file."""
    return {
        'begin': PRAGMA_LINE + ''.join(begin_lookaheads) + PRAGMA_BODY,
        'beginCaptures': {
            '1': {'name': 'meta.attribute.til punctuation.definition.attribute.begin.til'},
            '2': {'name': 'meta.attribute.til entity.other.attribute-name.til'},
            '3': {'name': 'meta.attribute.til punctuation.separator.key-value.til'},
            '4': {'name': 'meta.attribute.til support.constant.language-code.til'},
            '5': {'name': 'meta.attribute.til punctuation.definition.attribute.end.til'},
        },
        'end': '(?!)',          # never: the pragma holds for the rest of the file
        'patterns': [{'include': '#' + context}],
    }


def tm_string(quote: str, name: str) -> dict:
    return {
        'name': name,
        'begin': quote,
        'beginCaptures': {'0': {'name': 'punctuation.definition.string.begin.til'}},
        'end': '(' + quote + ')|$',
        'endCaptures': {'1': {'name': 'punctuation.definition.string.end.til'}},
        'patterns': [{'include': '#escape'}],
    }


def tm_fstring(quote: str, name: str) -> dict:
    interp_end = r'(\})|(?=' + quote + ')|$'
    return {
        'name': name,
        'begin': '(f)(' + quote + ')',
        'beginCaptures': {'1': {'name': 'storage.type.string.til'},
                          '2': {'name': 'punctuation.definition.string.begin.til'}},
        'end': '(' + quote + ')|$',
        'endCaptures': {'1': {'name': 'punctuation.definition.string.end.til'}},
        'patterns': [
            {'name': 'constant.character.escape.til', 'match': r'\{\{|\}\}'},
            {'include': '#escape'},
            {
                'name': 'meta.interpolation.til',
                'begin': r'\{',
                'beginCaptures': {'0': {'name': 'punctuation.section.interpolation.begin.til'}},
                'end': interp_end,
                'endCaptures': {'1': {'name': 'punctuation.section.interpolation.end.til'}},
                'contentName': 'meta.embedded.line.til',
                'patterns': [
                    {'match': r'(:)([<>^]?\d*(?:\.\d+)?[fdxXobe%]?)(?=\})',
                     'captures': {'1': {'name': 'punctuation.separator.format.til'},
                                  '2': {'name': 'constant.other.format-spec.til'}}},
                    {'begin': r'\{', 'end': interp_end, 'patterns': [{'include': '#expression'}]},
                    {'include': '#expression'},
                ],
            },
        ],
    }


def tm_grammar(t: Tables) -> dict:
    repo: Dict[str, dict] = {}

    # ── keyword languages ───────────────────────────────────────────────────
    for lang in LANGS:
        repo['keywords-' + lang] = {'patterns': tm_keyword_patterns(t, lang)}
        if t.aliases[lang]:
            repo['aliases-' + lang] = {'patterns': [
                {'name': 'support.function.builtin.til', 'match': word_re(t.aliases[lang])}]}
    repo['pragma-ru'] = tm_pragma([pragma_has('ru')], 'context-ru')
    repo['pragma-kk-latn'] = tm_pragma([pragma_has('kk-latn')], 'context-kk-latn')
    repo['pragma-ru-kk-latn'] = tm_pragma([pragma_has('ru'), pragma_has('kk-latn')], 'context-ru-kk-latn')
    repo['context-ru'] = {'patterns': [{'include': '#pragma-kk-latn-from-ru'},
                                       {'include': '#keywords-ru'}, {'include': '#aliases-ru'},
                                       {'include': '#base'}]}
    repo['pragma-kk-latn-from-ru'] = tm_pragma([pragma_has('kk-latn')], 'context-ru-kk-latn')
    repo['context-kk-latn'] = {'patterns': [{'include': '#pragma-ru-from-kk-latn'},
                                            {'include': '#keywords-kk-latn'}, {'include': '#aliases-kk-latn'},
                                            {'include': '#base'}]}
    repo['pragma-ru-from-kk-latn'] = tm_pragma([pragma_has('ru')], 'context-ru-kk-latn')
    repo['context-ru-kk-latn'] = {'patterns': [{'include': '#keywords-ru'}, {'include': '#aliases-ru'},
                                               {'include': '#keywords-kk-latn'},
                                               {'include': '#aliases-kk-latn'}, {'include': '#base'}]}

    # ── tokens ──────────────────────────────────────────────────────────────
    repo['comment'] = {'patterns': [{
        'name': 'comment.line.number-sign.til',
        'match': r'(#)(?!\[).*$',
        'captures': {'1': {'name': 'punctuation.definition.comment.til'}}}]}
    repo['attribute'] = {'patterns': [{
        'name': 'meta.attribute.til',
        'begin': r'(#\[)[ \t]*(?:(' + alt(t.attributes) + ')' + NE + '|(' + IDENT + '))?',
        'beginCaptures': {'1': {'name': 'punctuation.definition.attribute.begin.til'},
                          '2': {'name': 'entity.other.attribute-name.til'},
                          '3': {'name': 'entity.other.attribute-name.unknown.til'}},
        'end': r'(\])|$',
        'endCaptures': {'1': {'name': 'punctuation.definition.attribute.end.til'}},
        'patterns': [{'include': '#attribute-content'}]}]}
    repo['attribute-content'] = {'patterns': [
        {'begin': r'\[', 'end': r'(\])|$', 'patterns': [{'include': '#attribute-content'}]},
        {'name': 'punctuation.separator.key-value.til', 'match': ':'},
        {'include': '#expression'}]}
    repo['escape'] = {'patterns': [{
        'name': 'constant.character.escape.til',
        'match': r'\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]+\}|.)'}]}
    repo['string'] = {'patterns': [
        tm_fstring('"', 'string.interpolated.double.til'),
        tm_fstring("'", 'string.interpolated.single.til'),
        tm_string('"', 'string.quoted.double.til'),
        tm_string("'", 'string.quoted.single.til')]}
    repo['number'] = {'patterns': [
        {'name': 'constant.numeric.duration.til',
         'match': NB + r'(\d[\d_]*(?:\.\d[\d_]*)?)(ms|s|m|h)' + NE,
         'captures': {'2': {'name': 'keyword.other.unit.til'}}},
        {'name': 'constant.numeric.hex.til', 'match': NB + r'0[xX][0-9A-Fa-f_]+' + NE},
        {'name': 'constant.numeric.binary.til', 'match': NB + r'0[bB][01_]+' + NE},
        {'name': 'constant.numeric.octal.til', 'match': NB + r'0[oO][0-7_]+' + NE},
        {'name': 'constant.numeric.float.til',
         'match': NB + r'\d[\d_]*(?:\.\d[\d_]*(?:[eE][+-]?\d+)?|[eE][+-]?\d+)' + NE},
        {'name': 'constant.numeric.integer.til', 'match': NB + r'\d[\d_]*' + NE}]}
    repo['builtin'] = {'patterns': [
        {'name': 'support.function.builtin.til', 'match': word_re(t.builtins)},
        {'name': 'support.function.builtin.til', 'match': NB + '(?:' + alt(t.casts) + r')(?=[ \t]*\()'},
        {'name': 'support.function.builtin.constructor.til', 'match': word_re(t.constructors)},
        {'name': 'constant.language.none.til', 'match': word_re(['None'])}]}
    repo['type'] = {'patterns': [
        {'name': 'support.type.primitive.til', 'match': word_re(t.types)},
        {'name': 'support.type.til', 'match': word_re(GENERIC_TYPES)}]}
    repo['member'] = {'patterns': [
        {'match': r'(?<!\.)(\.)[ \t]*(\p{Lu}[\p{L}\p{N}_]*)' + NE,
         'captures': {'1': {'name': 'punctuation.accessor.til'},
                      '2': {'name': 'variable.other.enummember.til'}}},
        {'match': r'(?<!\.)(\.)[ \t]*(' + IDENT + r')(?=[ \t]*\()',
         'captures': {'1': {'name': 'punctuation.accessor.til'},
                      '2': {'name': 'entity.name.function.member.til'}}},
        {'match': r'(?<!\.)(\.)[ \t]*(' + IDENT + ')',
         'captures': {'1': {'name': 'punctuation.accessor.til'},
                      '2': {'name': 'variable.other.property.til'}}}]}
    repo['name'] = {'patterns': [
        {'name': 'variable.other.constant.til', 'match': NB + r'\p{Lu}[\p{Lu}\p{N}_]+' + NE},
        {'name': 'entity.name.type.til', 'match': NB + r'\p{Lu}[\p{L}\p{N}_]*' + NE},
        {'name': 'entity.name.function.til', 'match': NB + IDENT + r'(?=[ \t]*\()'}]}
    repo['operator'] = {'patterns': [
        {'name': 'keyword.operator.arrow.til', 'match': r'->|=>'},
        {'name': 'keyword.operator.pipeline.til', 'match': r'\|>'},
        {'name': 'keyword.operator.range.til', 'match': r'\.\.=?'},
        {'name': 'keyword.operator.comparison.til', 'match': r'==|!=|<=|>='},
        {'name': 'keyword.operator.assignment.compound.til', 'match': r'\+=|-=|\*=|/=|%='},
        {'name': 'keyword.operator.bitwise.shift.til', 'match': r'<<|>>'},
        {'name': 'keyword.operator.arithmetic.til', 'match': r'\*\*|[-+*/%]'},
        {'name': 'keyword.operator.comparison.til', 'match': r'[<>]'},
        {'name': 'keyword.operator.assignment.til', 'match': r'='},
        {'name': 'keyword.operator.bitwise.til', 'match': r'[&|^~]'},
        {'name': 'keyword.operator.question.til', 'match': r'\?'}]}
    repo['punctuation'] = {'patterns': [
        {'name': 'punctuation.separator.comma.til', 'match': ','},
        {'name': 'punctuation.separator.colon.til', 'match': ':'},
        {'name': 'punctuation.terminator.statement.til', 'match': ';'}]}
    repo['expression'] = {'patterns': [{'include': '#' + k} for k in (
        'keywords-en', 'keywords-kk', 'aliases-kk', 'string', 'number', 'builtin', 'type', 'member',
        'name', 'operator', 'punctuation')]}
    repo['base'] = {'patterns': [{'include': '#comment'}, {'include': '#attribute'},
                                 {'include': '#expression'}]}
    return {
        '$schema': 'https://raw.githubusercontent.com/martinring/tmlanguage/master/tmlanguage.json',
        'information_for_contributors': [
            f"Generated by {GENERATOR} from the keyword tables of src/til.py - do not edit by hand.",
            "Russian / Latin-script Kazakh keywords are keywords from a #[lang: ru] / #[lang: kk-latn] "
            "pragma line on."],
        'name': 'TIL',
        'scopeName': 'source.til',
        'fileTypes': ['til'],
        'patterns': [{'include': '#pragma-ru-kk-latn'}, {'include': '#pragma-ru'},
                     {'include': '#pragma-kk-latn'}, {'include': '#base'}],
        'repository': dict(sorted(repo.items())),
    }


# ═══════════════════════════════════════════════════════════════════════════
#                 LANGUAGE CONFIGURATION (VS Code, JavaScript regexes)
# ═══════════════════════════════════════════════════════════════════════════
# JavaScript regexes without the `u` flag: \w and \b are ASCII only, so word
# ends are spelled out (Cyrillic keywords).

JS_END = r'(?=[\s(:]|$)'                 # a keyword ends: space, '(' or ':' or the end of the line


def js_alt(words) -> str:
    return '(?:' + alt(words) + ')'


def language_configuration(t: Tables) -> dict:
    every = list(LANGS)
    starters = t.all_words(every, set(BLOCK_STARTERS) - {'HANDLE'})
    handle = t.all_words(every, {'HANDLE'})
    expr = t.all_words(every, set(BLOCK_EXPRESSIONS))
    ret = t.all_words(every, {'RETURN'})
    pub = t.all_words(every, {'PUB'})
    with_words = t.all_words(every, {'WITH'})
    dedent = t.all_words(every, {'ELSE', 'ELIF'})
    selfs = t.all_words(every, {'SELF'})
    jump = t.all_words(every, {'BREAK', 'CONTINUE'})
    name = r"[^\s\d()\[\]{}<>,.:;=+\-*/%!&|^~?#\"'][^\s()\[\]{}<>,.:;=+\-*/%!&|^~?#\"']*"
    tail = r'\s*(?:#.*)?$'
    params = r'\((?:[^()]|\([^()]*\))*\)'
    increase = '|'.join([
        # if / for / while / struct / impl / handle ... (any keyword language)
        r'^\s*(?:' + js_alt(pub) + r'\s+)?' + js_alt(starters) + JS_END + r'.*$',
        r'^\s*' + js_alt(handle) + r'(?:\s+.*)?$',
        r'^\s*' + js_alt(with_words) + tail,
        # let x = handle / retry(3) / timeout(20) / match v ; return match s
        r'(?:=|(?:^|\s)' + js_alt(ret) + r')\s*' + js_alt(expr) + JS_END + r'.*$',
        # an effect clause or a match arm with an indented body: `Ask(q) =>`
        r'=>' + tail,
        # function definitions: `name(a: int) -> int`, `name()`, `method(self, ...)`
        r'^(?:' + js_alt(pub) + r'\s+)?' + name + r'\s*(?:<[^<>]*>)?\s*' + params + r'\s*->[^#]*' + tail,
        r'^(?:' + js_alt(pub) + r'\s+)?' + name + r'\s*(?:<[^<>]*>)?\s*\(\s*\)' + tail,
        r'^(?:' + js_alt(pub) + r'\s+)?' + name + r'\s*(?:<[^<>]*>)?\s*\(\s*' + name + r'\s*:[^()]*\)' + tail,
        r'^\s+' + name + r'\s*(?:<[^<>]*>)?\s*\(\s*' + js_alt(selfs) + r'\s*(?:[,:)][^#]*)?' + tail,
    ])
    decrease = '|'.join([
        r'^\s*' + js_alt(dedent) + JS_END,
        r'^\s*' + js_alt(with_words) + tail,
    ])
    return {
        '//': f"Generated by {GENERATOR} from the keyword tables of src/til.py - do not edit by hand.",
        'comments': {'lineComment': '#'},
        'brackets': [['{', '}'], ['[', ']'], ['(', ')']],
        'autoClosingPairs': [
            {'open': '{', 'close': '}'},
            {'open': '[', 'close': ']'},
            {'open': '(', 'close': ')'},
            {'open': '"', 'close': '"', 'notIn': ['string', 'comment']},
            {'open': "'", 'close': "'", 'notIn': ['string', 'comment']},
        ],
        'autoCloseBefore': ";:.,=}])> \n\t",
        'surroundingPairs': [['{', '}'], ['[', ']'], ['(', ')'], ['"', '"'], ["'", "'"]],
        'indentationRules': {
            'increaseIndentPattern': increase,
            'decreaseIndentPattern': decrease,
        },
        'onEnterRules': [
            {
                # after break / continue the block goes on one level up
                'beforeText': r'^\s*' + js_alt(jump) + tail,
                'action': {'indent': 'outdent'},
            },
            {
                # ... and after return, unless it opens a block (`return match s`)
                'beforeText': r'^\s*' + js_alt(ret) + r'(?:\s+(?!' + js_alt(expr) + JS_END + r')[^#]*)?' + tail,
                'action': {'indent': 'outdent'},
            },
        ],
        'folding': {
            'offSide': True,
            'markers': {'start': r'^\s*#\s*region\b', 'end': r'^\s*#\s*endregion\b'},
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#                                    VIM
# ═══════════════════════════════════════════════════════════════════════════

VIM_LINKS = [
    ('tilConditional', 'Conditional'), ('tilRepeat', 'Repeat'), ('tilStatement', 'Statement'),
    ('tilException', 'Exception'), ('tilEffect', 'Keyword'), ('tilContract', 'Keyword'),
    ('tilInclude', 'Include'), ('tilStorage', 'StorageClass'), ('tilModifier', 'StorageClass'),
    ('tilLogical', 'Operator'), ('tilSelf', 'Special'), ('tilBoolean', 'Boolean'), ('tilNone', 'Constant'),
    ('tilBuiltin', 'Function'), ('tilConstructor', 'Function'), ('tilType', 'Type'), ('tilTypeName', 'Type'),
    ('tilDeclType', 'Type'), ('tilDeclFunction', 'Function'), ('tilConstantName', 'Constant'),
    ('tilFunction', 'Function'), ('tilNumber', 'Number'), ('tilFloat', 'Float'), ('tilDuration', 'Number'),
    ('tilString', 'String'), ('tilFString', 'String'), ('tilFStringDelim', 'String'),
    ('tilInterpDelim', 'Special'), ('tilFormatSpec', 'Special'), ('tilBraceEscape', 'SpecialChar'),
    ('tilEscape', 'SpecialChar'), ('tilComment', 'Comment'), ('tilTodo', 'Todo'),
    ('tilAttribute', 'PreProc'), ('tilAttrDelim', 'PreProc'), ('tilAttrName', 'PreProc'),
    ('tilOperator', 'Operator'),
]


def vim_alt(words) -> str:
    return r'\%(' + r'\|'.join(sorted(set(words), key=lambda w: (-len(w), w))) + r'\)'


def vim_quote(pattern: str) -> str:
    """A Vim single-quoted string."""
    return "'" + pattern.replace("'", "''") + "'"


def vim_keywords(t: Tables, lang: str, indent: str) -> List[str]:
    out = []
    for g in GROUPS:
        words = t.group_words(lang, g)
        if words:
            # `syn keyword` lines of at most ~10 words
            for i in range(0, len(words), 10):
                out.append(f"{indent}syn keyword {g[2]} " + ' '.join(words[i:i + 10]))
    for tt, scope in DECL_NAMES:
        words = t.keywords[lang].get(tt)
        if words:
            group = 'tilDeclFunction' if tt == 'FN' else 'tilDeclType'
            pat = r'\%(\<' + vim_alt(words) + r'\s\+\)\@40<=\<\k\+'
            out.append(f"{indent}syn match {group} {vim_quote(pat)}")
    for tt, words in t.contextual[lang].items():
        w = r'\<' + vim_alt(words) + r'\>'
        if tt == 'WITH':
            pats = [w + r'\ze\s*\%(#.*\)\=$', w + r'\ze\s\+\k\+\s*\%((\|=>\)']
        elif tt == 'RESUME':
            pats = [w + r'\ze\s*(']
        elif tt == 'PERFORM':
            pats = [w + r'\ze\s\+\k']
        else:   # HANDLE
            pats = [w + r'\ze\s*\%(#.*\)\=$', w + r'\ze\s\+\%(\k\|["' + "'" + r']\)']
        for p in pats:
            out.append(f"{indent}syn match tilEffect {vim_quote(p)}")
    if t.aliases[lang]:
        out.append(f"{indent}syn keyword tilBuiltin " + ' '.join(t.aliases[lang]))
    return out


def vim_pragma_test(code: str) -> str:
    return r'\m\c^\s*#\[\s*lang\s*:\%([^\],]*,\)*\s*' + code + r'\s*[,\]]'


def vim_syntax(t: Tables) -> str:
    L: List[str] = []
    add = L.append
    add('" Vim syntax file')
    add('" Language:   TIL (The Intelligent Language) ' + til.TIL_VERSION)
    add('" Maintainer: Alisher Beisembekov')
    add('" Generated by ' + GENERATOR + ' from the keyword tables of src/til.py.')
    add('" Do not edit by hand: change src/til.py and run that script.')
    add('"')
    add('" Install: copy this file to ~/.vim/syntax/til.vim (Neovim: ~/.config/nvim/syntax/)')
    add('" and add to your vimrc:  autocmd BufRead,BufNewFile *.til setfiletype til')
    add('" Russian and Latin-script Kazakh keywords are highlighted in files with a')
    add('" #[lang: ru] / #[lang: kk-latn] pragma (checked when the file is loaded: :syntax on again')
    add('" after adding one). Language server: `til lsp` (docs/TIL_LSP.md).')
    add('')
    add("if exists('b:current_syntax')")
    add('  finish')
    add('endif')
    add('')
    add('let s:cpo_save = &cpo')
    add('set cpo&vim')
    add('')
    add('syn case match')
    add('syn sync minlines=50')
    add('')
    add('" Names: user functions, types (CamelCase), constants (ALL_CAPS)')
    add(r"syn match tilFunction '\<\k\+\ze\s*('")
    add(r"syn match tilTypeName '\<\u\k*\>'")
    add(r"syn match tilConstantName '\<\u[[:upper:][:digit:]_]\+\>'")
    add('')
    add('" Keywords: English and Kazakh (always on)')
    for lang in ('en', 'kk'):
        L.extend(vim_keywords(t, lang, ''))
    add('')
    for code, lang in (('ru', 'ru'), ('kk-latn', 'kk-latn')):
        add(f'" #[lang: {code}]: {"Russian" if code == "ru" else "Latin-script Kazakh"} keywords '
            '(the compiler enables them per file)')
        add(f"if search({vim_quote(vim_pragma_test(code))}, 'nw')")
        L.extend(vim_keywords(t, lang, '  '))
        add('endif')
        add('')
    add('" Builtins (NATIVES), constructors, types (TYPE_MAP)')
    builtins = t.builtins
    for i in range(0, len(builtins), 10):
        add('syn keyword tilBuiltin ' + ' '.join(builtins[i:i + 10]))
    add('syn keyword tilConstructor ' + ' '.join(t.constructors))
    add('syn keyword tilNone None')
    types_pat = r'\<' + vim_alt(t.types + list(GENERIC_TYPES)) + r'\>'
    casts_pat = r'\<' + vim_alt(t.casts) + r'\ze\s*('
    add('syn match tilType ' + vim_quote(types_pat))
    add('syn match tilBuiltin ' + vim_quote(casts_pat))
    add('')
    add('" Numbers: integers, floats, hex / binary / octal, durations (5s, 100ms)')
    add(r"syn match tilNumber '\<\d[0-9_]*\>'")
    add(r"syn match tilNumber '\<0[xX][0-9a-fA-F_]\+\>'")
    add(r"syn match tilNumber '\<0[bB][01_]\+\>'")
    add(r"syn match tilNumber '\<0[oO][0-7_]\+\>'")
    add(r"syn match tilFloat '\<\d[0-9_]*\.\d[0-9_]*\%([eE][+-]\=\d\+\)\=\>'")
    add(r"syn match tilFloat '\<\d[0-9_]*[eE][+-]\=\d\+\>'")
    add(r"syn match tilDuration '\<\d[0-9_]*\%(\.\d[0-9_]*\)\=\%(ms\|s\|m\|h\)\>'")
    add('')
    add('" Operators')
    add(r"syn match tilOperator '->\|=>\||>\|\.\.=\=\|[-+*/%=<>!&|^~?]'")
    add('')
    add('" Strings, f-strings with {interpolation} and {value:.2f} format specs')
    add(r"syn match tilEscape '\\\%(x\x\x\|u{\x\+}\|.\)' contained")
    add(r"syn region tilString start=+'+ skip=+\\\\\|\\'+ end=+'+ end=+$+ oneline contains=tilEscape")
    add(r'syn region tilString start=+"+ skip=+\\\\\|\\"+ end=+"+ end=+$+ oneline contains=tilEscape')
    add(r"syn region tilFString matchgroup=tilFStringDelim start=+\<f'+ skip=+\\\\\|\\'+ end=+'+ end=+$+ "
        r"oneline contains=tilEscape,tilInterp,tilBraceEscape")
    add(r'syn region tilFString matchgroup=tilFStringDelim start=+\<f"+ skip=+\\\\\|\\"+ end=+"+ end=+$+ '
        r'oneline contains=tilEscape,tilInterp,tilBraceEscape')
    add(r"syn region tilInterp matchgroup=tilInterpDelim start='{' end='}' contained oneline "
        r"contains=@tilExpr,tilFormatSpec")
    add(r"syn match tilFormatSpec ':[<>^]\=\d*\%(\.\d\+\)\=[fdxXobe%]\=\ze}' contained")
    add(r"syn match tilBraceEscape '{{\|}}' contained")
    add('')
    add('" Comments and attributes #[...]')
    add(r"syn keyword tilTodo " + ' '.join(TODO_WORDS) + " contained")
    add(r"syn match tilComment '#\[\@!.*$' contains=tilTodo,@Spell")
    add(r"syn region tilAttribute matchgroup=tilAttrDelim start='#\[' end='\]' end='$' oneline "
        r"contains=tilAttrName,tilAttrNested,@tilExpr")
    add(r"syn region tilAttrNested start='\[' end='\]' end='$' contained oneline transparent "
        r"contains=tilAttrNested,@tilExpr")
    add(r"syn match tilAttrName '\%(#\[\s*\)\@<=\h\w*' contained")
    add('')
    groups = sorted({g[2] for g in GROUPS} | {'tilBuiltin', 'tilConstructor', 'tilNone', 'tilType', 'tilNumber',
                                              'tilFloat', 'tilDuration', 'tilString', 'tilFString',
                                              'tilOperator', 'tilFunction', 'tilTypeName', 'tilConstantName'})
    add('syn cluster tilExpr contains=' + ','.join(groups))
    add('')
    for group, target in VIM_LINKS:
        add(f'hi def link {group} {target}')
    add('')
    add("let b:current_syntax = 'til'")
    add('')
    add('let &cpo = s:cpo_save')
    add('unlet s:cpo_save')
    return '\n'.join(L) + '\n'


# ═══════════════════════════════════════════════════════════════════════════
#                                  SUBLIME
# ═══════════════════════════════════════════════════════════════════════════

def sublime_keywords(t: Tables, lang: str) -> List[dict]:
    rules: List[dict] = []
    for tt, scope in DECL_NAMES:
        words = t.keywords[lang].get(tt)
        if words:
            rules.append({'match': NB + '(' + alt(words) + r')[ \t]+(' + IDENT + ')' + NE,
                          'captures': {1: 'storage.type.til', 2: scope}})
    for tt, words in t.contextual[lang].items():
        rules.append({'match': NB + '(?:' + alt(words) + ')' + NE + CONTEXT_LOOKAHEAD[tt],
                      'scope': 'keyword.control.effect.til'})
    for g in GROUPS:
        words = t.group_words(lang, g)
        if words:
            rules.append({'match': word_re(words), 'scope': g[1]})
    if t.aliases[lang]:
        rules.append({'match': word_re(t.aliases[lang]), 'scope': 'support.function.builtin.til'})
    return rules


def sublime_pragma(lookaheads: List[str], target: str) -> dict:
    return {'match': PRAGMA_LINE + ''.join(lookaheads) + PRAGMA_BODY,
            'captures': {1: 'meta.attribute.til punctuation.definition.attribute.begin.til',
                         2: 'meta.attribute.til entity.other.attribute-name.til',
                         3: 'meta.attribute.til punctuation.separator.key-value.til',
                         4: 'meta.attribute.til support.constant.language-code.til',
                         5: 'meta.attribute.til punctuation.definition.attribute.end.til'},
            'set': target}


def sublime_syntax(t: Tables) -> dict:
    ctx: Dict[str, list] = {}
    both = [pragma_has('ru'), pragma_has('kk-latn')]
    ctx['main'] = [sublime_pragma(both, 'main-ru-kk-latn'),
                   sublime_pragma([pragma_has('ru')], 'main-ru'),
                   sublime_pragma([pragma_has('kk-latn')], 'main-kk-latn'),
                   {'include': 'base'}]
    ctx['main-ru'] = [sublime_pragma([pragma_has('kk-latn')], 'main-ru-kk-latn'),
                      {'include': 'keywords-ru'}, {'include': 'base'}]
    ctx['main-kk-latn'] = [sublime_pragma([pragma_has('ru')], 'main-ru-kk-latn'),
                           {'include': 'keywords-kk-latn'}, {'include': 'base'}]
    ctx['main-ru-kk-latn'] = [{'include': 'keywords-ru'}, {'include': 'keywords-kk-latn'},
                              {'include': 'base'}]
    for lang in LANGS:
        ctx['keywords-' + lang] = sublime_keywords(t, lang)
    ctx['base'] = [{'include': 'comment'}, {'include': 'attribute'}, {'include': 'expression'}]
    ctx['expression'] = [{'include': 'keywords-en'}, {'include': 'keywords-kk'}, {'include': 'string'},
                         {'include': 'number'}, {'include': 'builtin'}, {'include': 'member'},
                         {'include': 'name'}, {'include': 'operator'}]
    ctx['comment'] = [{'match': r'(#)(?!\[).*$', 'scope': 'comment.line.number-sign.til',
                       'captures': {1: 'punctuation.definition.comment.til'}}]
    ctx['attribute'] = [{'match': r'(#\[)[ \t]*(?:(' + alt(t.attributes) + ')' + NE + '|(' + IDENT + '))?',
                         'captures': {1: 'punctuation.definition.attribute.begin.til',
                                      2: 'entity.other.attribute-name.til',
                                      3: 'entity.other.attribute-name.unknown.til'},
                         'push': 'attribute-body'}]
    ctx['attribute-body'] = [{'meta_scope': 'meta.attribute.til'},
                             {'match': r'\]', 'scope': 'punctuation.definition.attribute.end.til', 'pop': True},
                             {'match': r'$', 'pop': True},
                             {'match': r'\[', 'push': 'attribute-bracket'},
                             {'match': ':', 'scope': 'punctuation.separator.key-value.til'},
                             {'include': 'expression'}]
    ctx['attribute-bracket'] = [{'match': r'\]', 'pop': True}, {'match': r'$', 'pop': True},
                                {'match': r'\[', 'push': 'attribute-bracket'}, {'include': 'expression'}]
    ctx['string'] = []
    for q, kind in (('"', 'double'), ("'", 'single')):
        ctx['string'].append({'match': '(f)(' + q + ')',
                              'captures': {1: 'storage.type.string.til',
                                           2: 'punctuation.definition.string.begin.til'},
                              'push': 'fstring-' + kind})
    for q, kind in (('"', 'double'), ("'", 'single')):
        ctx['string'].append({'match': q, 'scope': 'punctuation.definition.string.begin.til',
                              'push': 'string-' + kind})
    escape = {'match': r'\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]+\}|.)', 'scope': 'constant.character.escape.til'}
    for q, kind in (('"', 'double'), ("'", 'single')):
        ctx['string-' + kind] = [{'meta_scope': f'string.quoted.{kind}.til'},
                                 {'match': q, 'scope': 'punctuation.definition.string.end.til', 'pop': True},
                                 {'match': '$', 'pop': True}, escape]
        ctx['fstring-' + kind] = [{'meta_scope': f'string.interpolated.{kind}.til'},
                                  {'match': q, 'scope': 'punctuation.definition.string.end.til', 'pop': True},
                                  {'match': '$', 'pop': True},
                                  {'match': r'\{\{|\}\}', 'scope': 'constant.character.escape.til'},
                                  escape,
                                  {'match': r'\{', 'scope': 'punctuation.section.interpolation.begin.til',
                                   'push': 'interpolation-' + kind}]
        ctx['interpolation-' + kind] = [
            {'clear_scopes': 1},
            {'meta_scope': 'meta.interpolation.til'},
            {'match': r'\}', 'scope': 'punctuation.section.interpolation.end.til', 'pop': True},
            {'match': '(?=' + q + ')|$', 'pop': True},
            {'match': r'(:)([<>^]?\d*(?:\.\d+)?[fdxXobe%]?)(?=\})',
             'captures': {1: 'punctuation.separator.format.til', 2: 'constant.other.format-spec.til'}},
            {'include': 'expression'}]
    ctx['number'] = [
        {'match': NB + r'(\d[\d_]*(?:\.\d[\d_]*)?)(ms|s|m|h)' + NE, 'scope': 'constant.numeric.duration.til',
         'captures': {2: 'keyword.other.unit.til'}},
        {'match': NB + r'0[xX][0-9A-Fa-f_]+' + NE, 'scope': 'constant.numeric.hex.til'},
        {'match': NB + r'0[bB][01_]+' + NE, 'scope': 'constant.numeric.binary.til'},
        {'match': NB + r'0[oO][0-7_]+' + NE, 'scope': 'constant.numeric.octal.til'},
        {'match': NB + r'\d[\d_]*(?:\.\d[\d_]*(?:[eE][+-]?\d+)?|[eE][+-]?\d+)' + NE,
         'scope': 'constant.numeric.float.til'},
        {'match': NB + r'\d[\d_]*' + NE, 'scope': 'constant.numeric.integer.til'}]
    ctx['builtin'] = [
        {'match': word_re(t.builtins), 'scope': 'support.function.builtin.til'},
        {'match': NB + '(?:' + alt(t.casts) + r')(?=[ \t]*\()', 'scope': 'support.function.builtin.til'},
        {'match': word_re(t.constructors), 'scope': 'support.function.builtin.constructor.til'},
        {'match': word_re(['None']), 'scope': 'constant.language.none.til'},
        {'match': word_re(t.types), 'scope': 'support.type.primitive.til'},
        {'match': word_re(GENERIC_TYPES), 'scope': 'support.type.til'}]
    ctx['member'] = [
        {'match': r'(?<!\.)(\.)[ \t]*(\p{Lu}[\p{L}\p{N}_]*)' + NE,
         'captures': {1: 'punctuation.accessor.til', 2: 'variable.other.enummember.til'}},
        {'match': r'(?<!\.)(\.)[ \t]*(' + IDENT + r')(?=[ \t]*\()',
         'captures': {1: 'punctuation.accessor.til', 2: 'entity.name.function.member.til'}},
        {'match': r'(?<!\.)(\.)[ \t]*(' + IDENT + ')',
         'captures': {1: 'punctuation.accessor.til', 2: 'variable.other.property.til'}}]
    ctx['name'] = [
        {'match': NB + r'\p{Lu}[\p{Lu}\p{N}_]+' + NE, 'scope': 'variable.other.constant.til'},
        {'match': NB + r'\p{Lu}[\p{L}\p{N}_]*' + NE, 'scope': 'entity.name.type.til'},
        {'match': NB + IDENT + r'(?=[ \t]*\()', 'scope': 'variable.function.til'}]
    ctx['operator'] = [
        {'match': r'->|=>', 'scope': 'keyword.operator.arrow.til'},
        {'match': r'\|>', 'scope': 'keyword.operator.pipeline.til'},
        {'match': r'\.\.=?', 'scope': 'keyword.operator.range.til'},
        {'match': r'==|!=|<=|>=|<|>', 'scope': 'keyword.operator.comparison.til'},
        {'match': r'\+=|-=|\*=|/=|%=|=', 'scope': 'keyword.operator.assignment.til'},
        {'match': r'\*\*|[-+*/%]', 'scope': 'keyword.operator.arithmetic.til'},
        {'match': r'<<|>>|[&|^~]', 'scope': 'keyword.operator.bitwise.til'}]
    return {
        'name': 'TIL',
        'file_extensions': ['til'],
        'scope': 'source.til',
        'contexts': ctx,
    }


def yaml_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def yaml_dump(value, indent: int = 0) -> List[str]:
    """A tiny YAML writer for the sublime-syntax shape: maps, lists, strings, ints, bools."""
    pad = '  ' * indent
    out: List[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k)
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{key}:")
                out.extend(yaml_dump(v, indent + 1))
            else:
                out.append(f"{pad}{key}: {yaml_scalar(v)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                lines = yaml_dump(item, indent + 1)
                first = lines[0].lstrip()
                out.append(f"{pad}- {first}")
                out.extend(lines[1:])
            else:
                out.append(f"{pad}- {yaml_scalar(item)}")
    return out


def yaml_scalar(v) -> str:
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return '[' + ', '.join(yaml_scalar(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{}'
    return yaml_str(str(v))


def sublime_text(t: Tables) -> str:
    syntax = sublime_syntax(t)
    head = ['%YAML 1.2', '---',
            f'# TIL syntax for Sublime Text - generated by {GENERATOR}',
            '# from the keyword tables of src/til.py; do not edit by hand.',
            '# Russian / Latin-script Kazakh keywords: from a #[lang: ru] / #[lang: kk-latn] pragma line on.',
            '# Language server: the LSP package with `til lsp` (docs/TIL_LSP.md).']
    body = []
    for key in ('name', 'file_extensions', 'scope'):
        body.append(f"{key}: {yaml_scalar(syntax[key])}")
    body.append('contexts:')
    body.extend(yaml_dump(syntax['contexts'], 1))
    return '\n'.join(head + body) + '\n'


def sublime_regexes(t: Tables) -> List[str]:
    """Every regex of the Sublime syntax (for the tests: they must compile with Oniguruma)."""
    out = []
    for rules in sublime_syntax(t)['contexts'].values():
        for r in rules:
            if 'match' in r:
                out.append(r['match'])
    return out


# ═══════════════════════════════════════════════════════════════════════════
#                                    MAIN
# ═══════════════════════════════════════════════════════════════════════════

def outputs(t: Optional[Tables] = None) -> Dict[str, str]:
    t = t or Tables()
    return {
        TM_PATH: json.dumps(tm_grammar(t), indent=2, ensure_ascii=False) + '\n',
        LANGCONF_PATH: json.dumps(language_configuration(t), indent=2, ensure_ascii=False) + '\n',
        VIM_PATH: vim_syntax(t),
        SUBLIME_PATH: sublime_text(t),
    }


def tables_json(t: Optional[Tables] = None) -> dict:
    """The tables the grammars were generated from (for the tests)."""
    t = t or Tables()
    return {
        'keywords': t.keywords, 'contextual': t.contextual, 'aliases': t.aliases,
        'builtins': t.builtins, 'casts': t.casts, 'constructors': t.constructors, 'types': t.types,
        'attributes': t.attributes, 'groups': {g[0]: {'scope': g[1], 'vim': g[2], 'types': list(g[3])}
                                               for g in GROUPS},
        'sublime_regexes': sublime_regexes(t),
    }


def main(argv: List[str]) -> int:
    if '--tables' in argv:
        sys.stdout.write(json.dumps(tables_json(), ensure_ascii=False, indent=1) + '\n')
        return 0
    check = '--check' in argv
    stale = []
    for path, text in outputs().items():
        try:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                old = f.read()
        except OSError:
            old = None
        if old == text:
            continue
        rel = os.path.relpath(path, REPO).replace(os.sep, '/')
        if check:
            stale.append(rel)
            continue
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        print(f"wrote {rel}")
    if stale:
        print("out of date (run python3 " + GENERATOR + "): " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
