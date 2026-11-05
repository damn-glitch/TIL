# Localized messages (Kazakh, Russian)

TIL 3.1 prints its diagnostics in **English**, **Russian** or **Kazakh**:
syntax errors, the verifier's `E-EFFECT` / `E-CAP` / `E-CONTRACT` / `E-ARITY`
/ `E-IMPORT` errors, `E-VM` (bytecode compiler) errors, the typed backend's
`E-TYPE` errors and `native backend: ...` refusals, the prover's `E-PROVE`
errors, hints, warnings, the uncaught runtime-error reports of both VMs (the
reference VM and the C VM) when a program runs through `til`, and the
diagnostics of `til prove`, `til pkg`, `til fit` and `til debug`.

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
   `dis`, `translate`, `repl`, `prove`, `pkg`, `fit`, `debug`, also
   `--lang=ru`). An unknown value is an error.
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
  (`  at f (line 3)` becomes `  в f (строка 3)` / `  f ішінде (3-жол)`);
* the C VM runner's own messages (`tilvm: bad --keys entry ...`) and the
  graphics and snapshot messages of `til` (`til: --screen: ...`, `til: cannot
  resume x.tilp: the snapshot is damaged or incomplete (truncated)`): the
  program name stays, the message after it is translated;
* the tools: `til prove` (its `E-PROVE` errors with the obligation, the
  reason a function is unsupported and the outcome of a counterexample run,
  its option errors), `til pkg` (`til pkg: error: ...`, `til pkg: warning:
  ...`), `til fit` (`til fit: ...`, device files, `til build --device`) and
  `til debug` (every message of the debugger). `til prove` and `til fit`
  also follow the program's `#[lang: ...]` pragma, as `til check` does.

Kept as they are:

* **values that programs see**: the `error` of a `retry` / `timeout` `else`
  branch, a message re-raised with `panic(error)` inside the program, `intent`
  failure texts caught by `retry`. Programs behave identically whatever the
  message language is; only the final report is translated.
* the source excerpt under a diagnostic (`  3 |     x = 2`), file names,
  identifiers, numbers, quoted code and everything inside a placeholder;
* error codes (`E-VM`, `E-EFFECT`, ...), so they stay searchable;
* messages that are not in the catalog yet: they stay English inside a
  translated frame (`ошибка[E-VM]: some new message`);
* **a program's own message** in its error report (`panic(...)`, `expect(...)`,
  an `Err` value): it is translated only when it is, as a whole, one of the
  messages the VMs report (`panic("division by zero")`). Free text never is:
  `panic("Expected 3 items, got 2")` and `panic("cannot convert negative
  balance to int")` stay English, because the report is matched against the
  run-time messages only and their placeholders take a single word or a type
  name (see [The catalog](#the-catalog)). The lines after the first one of a
  multi-line message are copied unchanged; only traceback lines are
  translated there (a line of the message that looks exactly like one,
  `  at f (line 3)`, cannot be told apart and is translated too);
* a message longer than 10,000 characters (`tilmsg.MAX_LOOKUP`) is not looked
  up and stays English (its frame is translated), and `intent` failures nested
  more than 32 deep (`MAX_NESTING`) keep their inner messages English;
* standard output (`OK: prog.til`, `Built: ...`) and `til --help`; so are
  the reports the tools print there (`til prove`'s `f: proved` lines and
  `Result:`, `til pkg`'s `+ name 1.0.0` and `verify` lines, `til fit`'s
  report and `--json`; also the fit report `til build --device` prints when
  a program does not fit), their usage texts and the messages of `til pkg`'s
  argument parser (argparse);
* text that is not TIL's: operating-system errors (`[Errno 2] No such file
  or directory`), a compiler's or git's own output, Z3's reasons;
* locations inside a message: file names, URLs, `til.json`, `git URL (sha)`,
  `til pkg add NAME`; code such as `str.upper()` in `native backend:
  str.upper() (line 3)`;
* the descriptions of the unknowns a failed proof relies on (`... it relies
  on 'i' at the start of an iteration of the loop at line 5; ...`);
* runtime errors printed by **native executables** (`til run` with a C
  compiler, `til build`): they come from the generated C runtime, whose
  messages are not catalogued yet. Their compile errors are translated.
* the C VM itself prints English (no catalog in the VM core: it runs on
  calculators and microcontrollers without `malloc`); `til` translates its
  reports on the host. A device without a host shows English.

Without `src/tilmsg.py` (for example an installation that only copied
`til.py`), or with one that cannot be loaded (a syntax error, a missing comma
between two entries, a malformed template), every message stays English. When
a translation was asked for (`--lang`, `TIL_LANG` or the program's pragma),
`til` says so once, right before the first diagnostic it prints:

```
note: message translations are not installed (tilmsg.py is missing); messages stay in English
note: message translations could not be loaded (tilmsg.py, line 73: TypeError: 'tuple' object is not callable); messages stay in English
```

A run without diagnostics prints no note.

## How it works

Diagnostics are written in English where they are raised (`src/til.py`,
`vm/tilvm.c`). Nothing in the compiler or the VMs depends on the language.
The command line translates text at **output time**:

* `til.localize(text)` translates a compiler or command-line diagnostic into
  the language chosen by `main()` (`til.MESSAGE_LANG`); `til.localize(text,
  'kk')` names the language explicitly. Library use of the compiler and the
  VMs (tests, embedding) is unaffected: `MESSAGE_LANG` is only set by
  `main()`.
* `til.localize_report_bytes(data)` translates a VM's uncaught-error report
  (`tilmsg.localize_report`). `TILVM.report` uses it after it has flushed the
  program's output; `til run --cvm` reads the C VM's standard error as it
  arrives (`tilmsg.ReportLocalizer`: lines before the report one by one, the
  rest of the report at once) and gives the same text as the reference VM.
* The catalog `src/tilmsg.py` is loaded lazily with `feature_module('tilmsg')`.
  A failure inside a translation never hides a diagnostic: the English text
  is printed instead.
* Every step takes linear time: no regular expression can backtrack over a
  line, templates are matched without backtracking, and the program's own
  text in a report is copied, not looked up. A program cannot slow its own
  report down (`--max-steps` does not bound the report, so this matters).

`tilmsg.localize(text, lang)` works line by line. It recognises the frame of
each line, translates the frame words and looks the message up in the
catalog:

| line | example |
|---|---|
| header | `error[E-VM]: msg`, `warning: msg`, `Error: msg`, `Warning: msg`, `note: msg` |
| syntax error | `Syntax Error: error: msg` (becomes `Синтаксическая ошибка: msg`) |
| hint | `  = hint: msg`, `hint: msg`, `  = note: msg` |
| runtime report | `runtime error: msg`, `fatal error: msg` (msg: run-time messages only) |
| program name | `til: msg`, `tilvm: msg`, `til pkg: error: msg`, `til fit --run: msg` (the name and up to three words after `til` are kept; the rest is framed again) |
| traceback | `  at main (line 7)` |
| location, source | ` --> file:3:5`, `  3 | code`, `    |   ^` (never changed) |
| anything else | the whole line is looked up as a message |

`tilmsg.localize_report(text, lang)` handles what a VM writes to standard
error: before the report, `tilvm: ...` lines are translated and other lines
copied; the report's first line (`runtime error: msg`) is translated with
`RUNTIME_MESSAGES` only; after it, traceback lines are translated and every
other line, the rest of the program's message, is copied unchanged.

ANSI colours (on a terminal) and Windows line endings are preserved.

### The catalog

`src/tilmsg.py` holds `(English, Russian, Kazakh)` templates in three lists:

* `COMPILER_MESSAGES`: syntax errors, the verifier, the bytecode compiler,
  the native backend's warnings, hints;
* `RUNTIME_MESSAGES`: everything the VMs can report at run time (both VMs'
  errors, contracts, asserts and intents compiled into programs, effects,
  graphics). A program's error report is matched against these only;
* `COMMAND_LINE_MESSAGES`: the messages of `til` and of the C VM's runner.

`MESSAGES` is the three lists together (compiler diagnostics are looked up in
all of them).

```python
("'{func}' expects {lo} to {hi} argument(s), got {got}",
 "'{func}' ожидает аргументов: от {lo} до {hi}, передано: {got}",
 "'{func}' {lo}–{hi} аргумент күтеді, бірақ {got} берілді"),
```

A placeholder says what text it stands for (the kind is written in the English
template only):

| placeholder | matches |
|---|---|
| `{name}` | one word: no blanks (`str`, `abs`, `<lambda:2>`, `-1`) |
| `'{name}'` (between matching `'`, `"` or `` ` ``) | any text on one line (`'load the class list'`) |
| `{name:text}` | any text: a condition, a path, a key, a user's message |
| `{name:int}` | an integer |
| `{name:type}` | a type as the VMs name it: `nil`, `bool`, `int`, `float`, `str`, `array`, `map`, `fn`, `struct`, or a capitalized name (`Point`, `Shape.Circle`, `Нүкте`); also as the typed backend writes it (`E-TYPE`): `[int]`, `{str: int}`, `Option<int>`, `Result<int, str>`, `fn(int) -> str`, `str or int`, `?3`, `None`, `dynamic` (checked by a small parser, in linear time) |
| `{name:builtin}` | the name of a builtin function (`[a-z_][a-z0-9_]*`) |
| `{name:msg}` | a nested message, translated too (`intent '{intent}' failed: {error:msg}`) |

`{{` / `}}` are literal braces. A name used twice must match the same text.
Every translation uses exactly the placeholders of the English template; their
text is copied unchanged. Choose the narrowest kind that fits every real
value: that is what keeps a program's own sentences out (`cannot compare
{left:type} and {right:type}` translates `cannot compare str and int`, not
`cannot compare apples and oranges`). Two placeholders must not touch. A
run-time template must not start with a `text` or `msg` placeholder (it would
take a program's own sentence with that ending); a compiler or command-line
template may, when distinctive text follows it (at least 30 characters of
literal text in a compiler template, 12 in a command-line one), e.g. the
parser's `{message:msg} ('{word}' is a reserved word and cannot be used as a
name)`, which translates any "Expected ..." message with that note appended,
or `til pkg`'s `{where:msg}: dependency '{name}' = '{spec}': invalid git URL`
(the location is looked up too: `registry URL (name 1.0.0)` is translated, a
path stays as it is). Such a template must not end with text that a longer
message of another entry also ends with: give that message its own entry.

Lookup:

* exact (placeholder-free) entries first, then the template with the most
  literal text wins (`'f' expects 1 to 2 argument(s)` uses the "to" entry,
  not the generic one; `value is not iterable` beats
  `{type:type} is not iterable`);
* matching never backtracks: a placeholder ends where the text after it in
  the template first occurs (the template's last text ends the message). A
  value that contains that text does not match, and the message stays
  English. This keeps every lookup linear.

`UNTRANSLATED` lists the messages that are English on purpose, each with
its reason: the default `panic` message is the builtin's name; code, file
names and keywords that appear as a nested part of a translated message
(`str.{method}()`, `til.json`, `let`, `#[ensures: ...]`, `git URL`); the notes
about a missing catalog; two internal consistency errors of `til debug`.
`FRAME` holds the frame words of each language.
`tilmsg.check_catalog()` reports malformed templates, placeholder mismatches,
bad kinds, duplicates and untranslated entries (a test runs it).

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
   modules next to it (`src/til*.py`), `vm/tilvm.c` and the C VM's runner
   `vm/main.c` (see its docstring for the exact rules) and checks each one
   against the catalog with the same lookup `til` uses (run-time messages
   against `RUNTIME_MESSAGES` only). It follows messages through helper
   functions (`_(text)`, `self.say(text)`, `self.unsupported(what)`), through
   the exception classes of the tools (`PkgError`, `Unsupported`, the
   `ValueError` that `til fit` prints) and through `localize()` /
   `til.localize()`. `tests/test_i18n.py` fails while a message of the
   enforced categories has no entry (everything but `cli`, plain
   `print(..., file=sys.stderr)` without `localize()`, and `internal`, an
   exception text nothing prints). Some entries cover text the tool cannot
   follow (a message built with `"\n".join(lines)`, a reason returned in a
   tuple, the help lines of `til debug`); `--unused` lists them with the
   entries written for one wording of a message that the tool sees as a
   placeholder (`function 'f' is defined in two files: ...`): they are used
   at run time.
3. Add the entry to the list that `--skeleton` names (`COMPILER_MESSAGES`,
   `RUNTIME_MESSAGES` or `COMMAND_LINE_MESSAGES`), in the section of its
   subsystem, with the Russian and Kazakh text, following the conventions
   below. Rename the `x1`, `x2` placeholders of the skeleton to meaningful
   names and give each the narrowest kind that fits (`{type:type}`,
   `{n:int}`, `{cond:text}`, ...).
4. If a message is internal and should stay English, add it to
   `UNTRANSLATED` with a comment saying why.
5. `python3 -m pytest tests/test_i18n.py`.

A message whose wording depends on a variable holding English words needs one
entry per wording. The tool expands local variables and literal-only
parameters (`what="name"` / `"function"`) by itself; for other cases
(`f"Invalid {name} literal"` with `name` from a loop) add the specific entries
by hand and keep a generic fallback. Prefer one sentence per case in the code
(`f"{x} is in ..." if n == 1 else f"{xs} are in ..."`) to several independent
`{'is' if n == 1 else 'are'}` choices: the tool expands every combination.

A message that quotes another one (`cannot prove 'f': unsupported:
{reason}`, `native backend: {reason} (line 3)`, `til pkg`'s `{where}: ...`)
takes it as a `{x:msg}` placeholder, and the quoted messages get entries of
their own; code, file names and keywords among them go to `UNTRANSLATED`.

## Adding a language

1. Add its code to `LANGUAGES` and `COLUMNS` in `src/tilmsg.py` and to
   `MESSAGE_LANGUAGES` in `src/til.py` (and to `message_language_code` if it
   has aliases).
2. Add its `FRAME` words (same keys as the other languages).
3. Add its text as a new last column to every entry of the three lists.
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
| native backend | нативный бэкенд | нативті бэкенд |
| type inference | вывод типов | типтерді шығару |
| closure / lambda | замыкание / лямбда | тұйықтама / лямбда |
| captured variable | захваченная переменная | сыртқы аймақтан алынған айнымалы |
| prover / solver | модуль доказательства / решатель | дәлелдеу модулі / шешуші |
| global initializers | инициализаторы глобальных переменных | ғаламдық айнымалылардың бастапқы мәндері |
| snapshot (`.tilp`) | снимок | күй көшірмесі |
| device | устройство | құрылғы |
| breakpoint | точка останова | тоқтау нүктесі |
| package / dependency | пакет / зависимость | пакет / тәуелділік |
| registry | реестр | тізілім |
| version requirement | требование к версии | нұсқаға талап |
| archive | архив | архив |
| symbolic link | символическая ссылка | символдық сілтеме |

## For feature modules

Tools that run under `til` (`til prove`, `til pkg`, `til fit`, `til debug`)
localize their own diagnostics (standard error) with a small helper, and the
coverage tool follows it:

```python
import til                                   # the running compiler

def _(text: str) -> str:
    return til.localize(text)

print(_(f"til fit: unknown device '{device}'; known: {names}"), file=sys.stderr)
```

The first words of such a line are the tool's name (`til pkg: error: ...`):
the catalog keeps them and translates the rest. A tool that reads a program
calls `til.use_program_message_language(source)` (the program's `#[lang:
...]` pragma, unless `--lang` or `TIL_LANG` chose a language) inside a
`main()` decorated with `@til.scoped_message_language`, so the choice does
not outlive the call.

`main()` sets `til.MESSAGE_LANG` from `--lang` / `TIL_LANG` before it
dispatches to a tool and removes `--lang X` / `--lang=X` (before any `--`)
from the tool's arguments, so `til pkg --lang ru list` works with any option
parser. `til.message_language(cli_value, source)` applies the full precedence
(including a program's pragma) for a tool that reads a program. The reference
VM's error reports are localized by themselves (`til debug`); a tool that runs
the C VM uses `til.run_localized(cmd)`, or `til.localize_report_bytes(data)`
on what it captured. Text printed with plain `print(..., file=sys.stderr)`
stays English until it goes through `til.localize()` and its messages are in
the catalog.
