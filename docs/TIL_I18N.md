# Localized messages (Kazakh, Russian)

TIL 3.1 prints its diagnostics in **English**, **Russian** or **Kazakh**:
syntax errors, the verifier's `E-EFFECT` / `E-CAP` / `E-CONTRACT` / `E-ARITY`
errors, `E-VM` (bytecode compiler) errors, hints, warnings, and the uncaught
runtime-error reports of both VMs (the reference VM and the C VM) when a
program runs through `til`.

```
$ til run --vm --lang kk prog.til
қате[E-VM]: өзгермейтін 'x' айнымалысына екінші рет мән меншіктеуге болмайды
 --> prog.til:3:5
  2 |     let x = 1
  3 |     x = 2
    |     ^
  4 |     print(x)
  = кеңес: өзгермелі болуы үшін оны 'var' арқылы жариялаңыз

$ TIL_LANG=ru til run --cvm divide.til
start
ошибка выполнения: деление на ноль
  в f (строка 2)
  в main (строка 5)
```

## Choosing the language

The first of these that is set wins:

1. `--lang en|ru|kk` on the command line (`til run`, `build`, `check`, `exec`,
   `dis`, `translate`, `repl`, also `--lang=ru`). An unknown value is an error.
2. The `TIL_LANG` environment variable. Locale-style values work too
   (`ru_RU.UTF-8`, `kk_KZ`, `kk-latn`, `kz`); an empty value means "not set";
   an unknown value prints a warning and is ignored.
3. The program's own `#[lang: ...]` pragma: `#[lang: ru]` gives Russian
   messages, `#[lang: kk]` and `#[lang: kk-latn]` give Kazakh (always in
   Cyrillic), `#[lang: en]` English. The first language named counts.
4. English.

So a program written with Russian keywords reports its errors in Russian,
unless the user asked for something else.

## What is translated and what is not

Translated:

* every compiler diagnostic: the headline message, the hint, the frame words
  (`error` / `warning` / `hint` / `note`, `Syntax Error`, `Error:`, `Warning:`),
  the native backend's type warnings (`Warning: Line 3: ...`) and the command
  line's own messages (file not found, unknown option, fallbacks to the VM);
* uncaught runtime-error reports of the reference VM (`til run --vm`,
  `til exec`, the REPL) and of the C VM (`til run --cvm`): the prefix
  (`runtime error:` / `fatal error:`), known messages and the traceback lines
  (`  at f (line 3)` becomes `  в f (строка 3)` / `  f ішінде (3-жол)`).

Kept as they are:

* **values that programs see**: the `error` of a `retry` / `timeout` `else`
  branch, a message re-raised with `panic(error)` inside the program, `intent`
  failure texts caught by `retry`. Programs behave identically whatever the
  message language is; only the final report is translated.
* the source excerpt under a diagnostic (`  3 |     x = 2`), file names,
  identifiers, numbers, quoted code and everything inside a placeholder;
* error codes (`E-VM`, `E-EFFECT`, ...), so they stay searchable;
* messages that are not in the catalog yet: they stay English inside a
  translated frame (`ошибка[E-TYPE]: some new message`);
* standard output (`OK: prog.til`, `Built: ...`) and `til --help`;
* runtime errors printed by **native executables** (`til run` with a C
  compiler, `til build`): they come from the generated C runtime, whose
  messages are not catalogued yet. Their compile errors are translated.
* the C VM itself prints English (no catalog in the VM core: it runs on
  calculators and microcontrollers without `malloc`); `til` translates its
  reports on the host. A device without a host shows English.

Without `src/tilmsg.py` (for example an installation that only copied
`til.py`) every message stays English; with an explicit `--lang` the compiler
says so once (`note: message translations are not installed ...`).

## How it works

Diagnostics are written in English where they are raised (`src/til.py`,
`vm/tilvm.c`). Nothing in the compiler or the VMs depends on the language.
The command line translates text at **output time**:

* `til.localize(text)` (and `localize_bytes` for VM reports) translate a whole
  diagnostic into the language chosen by `main()` (`til.MESSAGE_LANG`);
  `til.localize(text, 'kk')` names the language explicitly. Library use of the
  compiler and the VMs (tests, embedding) is unaffected: `MESSAGE_LANG` is
  only set by `main()`.
* `TILVM.report` localizes the uncaught-error report; `til run --cvm` reads the
  C VM's standard error line by line (as it arrives) and translates it.
* The catalog `src/tilmsg.py` is loaded lazily with `feature_module('tilmsg')`.
  A failure inside a translation never hides a diagnostic: the English text
  is printed instead.

`tilmsg.localize(text, lang)` works line by line. It recognises the frame of
each line, translates the frame words and looks the message up in the
catalog:

| line | example |
|---|---|
| header | `error[E-VM]: msg`, `warning: msg`, `Error: msg`, `Warning: msg`, `note: msg` |
| syntax error | `Syntax Error: error: msg` (becomes `Синтаксическая ошибка: msg`) |
| hint | `  = hint: msg`, `hint: msg`, `  = note: msg` |
| runtime report | `runtime error: msg`, `fatal error: msg` |
| traceback | `  at main (line 7)` |
| location, source | ` --> file:3:5`, `  3 | code`, `    |   ^` (never changed) |
| anything else | the whole line is looked up as a message |

ANSI colours (on a terminal) and Windows line endings are preserved.

### The catalog

`MESSAGES` in `src/tilmsg.py` is a list of `(English, Russian, Kazakh)`
templates:

```python
("'{func}' expects {lo} to {hi} argument(s), got {got}",
 "'{func}' ожидает аргументов: от {lo} до {hi}, передано: {got}",
 "'{func}' {lo}–{hi} аргумент күтеді, бірақ {got} берілді"),
```

* `{name}` matches any text, `{name:int}` an integer, `{name:msg}` a nested
  message that is translated too (`intent '{intent}' failed: {error:msg}`);
  `{{` / `}}` are literal braces. A name used twice must match the same text.
* Every translation uses exactly the placeholders of the English template;
  their text is copied unchanged.
* Lookup: exact (placeholder-free) entries first, then the template with the
  most literal text wins (`'f' expects 1 to 2 argument(s)` uses the "to"
  entry, not the generic one; `value is not iterable` beats
  `{type} is not iterable`). A template that starts with a placeholder only
  matches when that placeholder is a single word, so `{type} cannot be
  indexed` never translates a free-text panic message that happens to end the
  same way.
* `UNTRANSLATED` lists the few messages that are English on purpose (the
  default `panic` message is the builtin's name).
* `FRAME` holds the frame words of each language.
* `tilmsg.check_catalog()` reports malformed templates, placeholder
  mismatches, duplicates and untranslated entries (a test runs it).

## Adding a message

1. Write the diagnostic in English where it is raised, as usual
   (`self.error(node, f"...")`, `self.panic(...)`, `fail(vm, "...")`, ...).
2. Run the coverage tool:

   ```
   python3 tools/i18n_coverage.py              # summary and missing messages (exit 1 if any)
   python3 tools/i18n_coverage.py --skeleton   # entries to paste into MESSAGES
   python3 tools/i18n_coverage.py --list       # every extracted template
   python3 tools/i18n_coverage.py --unused     # entries no current message needs
   ```

   It extracts message templates statically from `src/til.py`, the feature
   modules next to it (`src/til*.py`) and `vm/tilvm.c` (see its docstring for
   the exact rules) and checks each one against the catalog with the same
   lookup `til` uses. `tests/test_i18n.py` fails while a message of the
   enforced categories has no entry; plain `print(..., file=sys.stderr)`
   messages are reported but not enforced. Entries may exist ahead of the
   code (the effects and graphics messages were written against their
   branches before those were merged); `--unused` lists them, they do no harm.
3. Add the entry to `MESSAGES` (in the section of its subsystem) with the
   Russian and Kazakh text, following the conventions below. Rename the `x1`,
   `x2` placeholders of the skeleton to meaningful names.
4. If a message is internal and should stay English, add it to
   `UNTRANSLATED` with a comment saying why.
5. `python3 -m pytest tests/test_i18n.py`.

A message whose wording depends on a variable holding English words needs one
entry per wording. The tool expands local variables and literal-only
parameters (`what="name"` / `"function"`) by itself; for other cases
(`f"Invalid {name} literal"` with `name` from a loop) add the specific entries
by hand and keep a generic fallback.

## Adding a language

1. Add its code to `LANGUAGES` and `COLUMNS` in `src/tilmsg.py` and to
   `MESSAGE_LANGUAGES` in `src/til.py` (and to `message_language_code` if it
   has aliases).
2. Add its `FRAME` words (same keys as the other languages).
3. Add its text as a new last column to every `MESSAGES` entry.
   `check_catalog()` lists every entry that still lacks it.
4. Extend the glossary below and the tests' expectations.

## Translation conventions

* A translation starts with a lowercase letter: it follows `ошибка:` /
  `қате:` (`ошибка: неожиданный символ`). Frame words keep the capitalisation
  of the English ones (`Error:` → `Ошибка:`, `error:` → `ошибка:`).
* Identifiers, keywords, code, numbers, quotes, error codes and placeholders
  are never translated or re-quoted: `'{name}'` stays `'{name}'`.
* **Russian** avoids number agreement with a placeholder: "ожидает
  аргументов: 2, передано: 3" instead of "2 аргумента / 5 аргументов".
* **Kazakh** never attaches a suffix to a placeholder, because the ending
  depends on the sound of the word it follows: the suffix goes on the noun
  after it (`'{name}' функциясы`, `{type} типін`), or the value follows a
  colon (`белгісіз атау: '{name}'`). Numerals take a singular noun ("2
  аргумент"); ranges are written `{lo}–{hi}`; ordinals with a hyphen
  (`3-жол`). Use `және` rather than `мен/бен/пен` after a placeholder.
* Prefer the words of TIL's own Kazakh and Russian keywords where they exist
  (`құрылым` struct, `санақ` enum, `ниет` intent, `тексеру` assert;
  `намерение`, `проверка`).

### Glossary

| English | Русский | Қазақша |
|---|---|---|
| error | ошибка | қате |
| warning | предупреждение | ескерту |
| hint | подсказка | кеңес |
| note | примечание | ескертпе |
| syntax error | синтаксическая ошибка | синтаксистік қате |
| runtime error | ошибка выполнения | орындау қатесі |
| fatal error | фатальная ошибка | қатерлі қате |
| function | функция | функция |
| argument / parameter | аргумент / параметр | аргумент / параметр |
| variable | переменная | айнымалы |
| immutable / mutable | неизменяемая / изменяемая | өзгермейтін / өзгермелі |
| constant | константа | тұрақты |
| global variable | глобальная переменная | ғаламдық айнымалы |
| assign | присвоить значение | мән меншіктеу |
| effect | эффект | әсер |
| pure function | чистая функция | таза функция |
| capability | полномочие | өкілеттік |
| contract violation | нарушение контракта | келісімшарт бұзылды |
| requires / ensures | требует / гарантирует | талабы / кепілдігі |
| assertion failed | проверка не пройдена | тексеру сәтсіз аяқталды |
| invariant | инвариант | инвариант |
| intent | намерение | ниет |
| struct / field | структура / поле | құрылым / өріс |
| enum / variant | перечисление / вариант | санақ / нұсқа |
| method | метод | әдіс |
| module | модуль | модуль |
| type / value | тип / значение | тип / мән |
| string | строка | жол |
| array | массив | массив |
| map | словарь | сөздік |
| key / index | ключ / индекс | кілт / индекс |
| out of bounds | вне границ | шектен тыс |
| pattern | образец | үлгі |
| handler | обработчик | өңдеуші |
| literal / token | литерал / токен | литерал / токен |
| indented block | блок с отступом | шегініспен жазылған блок |
| bytecode | байт-код | байт-код |
| stack overflow | переполнение стека | стек толып кетті |
| division by zero | деление на ноль | нөлге бөлу |
| step limit | лимит шагов | қадам шегі |
| file path | путь к файлу | файл жолы |
| line (in a file) | строка | жол (`3-жол`) |
| not available | недоступна | қолжетімсіз |
| VM build | сборка VM | VM құрастырмасы |
| compile time | этап компиляции | компиляция кезі |

## For feature modules

Tools that run under `til` (`til prove`, `til fit`, `til debug`, ...) can
localize their own diagnostics:

```python
import til                                   # the running compiler
print(til.localize(f"error[E-FIT]: {msg}"), file=sys.stderr)
```

`main()` sets `til.MESSAGE_LANG` from `--lang` / `TIL_LANG` before it
dispatches to a tool; `til.message_language(cli_value, source)` applies the
full precedence (including a program's pragma) for tools that parse their own
options. Diagnostics formatted with `til.ErrorReporter` get their frame
translated automatically; add the messages to the catalog for the rest.
