#!/usr/bin/env python3
"""
TIL message catalog (TIL 3.1): compiler and VM diagnostics in English,
Russian and Kazakh.

The compiler (src/til.py) and both VMs write their diagnostics in English.
The `til` command translates them at output time:

    import tilmsg
    tilmsg.localize(text, 'ru')          # a compiler / command-line diagnostic, any number of lines
    tilmsg.localize_report(text, 'kk')   # what a VM writes to stderr (uncaught-error report)
    tilmsg.translate(message, 'kk')      # one message, or None if it is not in the catalog

Only reports are translated. Values that programs see (the `error` of a
retry/timeout `else` branch, a panic message caught by `retry`) stay English.

A catalog entry is an English template with its Russian and Kazakh versions,
in one of three lists: COMPILER_MESSAGES, RUNTIME_MESSAGES (what the VMs can
report; a program's error report is only matched against these) and
COMMAND_LINE_MESSAGES. Placeholders: `{name}` a word (any text between
quotes: '{name}'), `{name:text}` any text, `{name:int}` an integer,
`{name:msg}` a nested message that is translated too, `{name:builtin}` the
name of a builtin function; `{{` and `}}` are literal braces. Text inside
placeholders (identifiers, numbers, code, quotes) is copied unchanged. A
message without an entry stays English, but the frame around it
(`error[E-VM]:`, `= hint:`, `runtime error:`, `  at f (line 3)`) is still
translated. Matching takes linear time (no backtracking), so a diagnostic
that quotes program data never slows `til` down.

docs/TIL_I18N.md explains how to add a message or a language;
tools/i18n_coverage.py lists the diagnostics that have no entry yet.
"""

import re
from typing import Dict, List, Optional, Tuple, Union

LANGUAGES = ('en', 'ru', 'kk')
COLUMNS = ('en', 'ru', 'kk')            # order of the texts in a MESSAGES entry

# ═══════════════════════════════════════════════════════════════════════════
# FRAME: the words around messages
# ═══════════════════════════════════════════════════════════════════════════

FRAME: Dict[str, Dict[str, str]] = {
    'ru': {
        'error': 'ошибка', 'warning': 'предупреждение', 'note': 'примечание', 'hint': 'подсказка',
        'Error': 'Ошибка', 'Warning': 'Предупреждение', 'Note': 'Примечание',
        'Syntax Error': 'Синтаксическая ошибка',
        'runtime error': 'ошибка выполнения', 'fatal error': 'фатальная ошибка',
        'at': '{indent}в {func}', 'at_line': '{indent}в {func} (строка {line})',
    },
    'kk': {
        'error': 'қате', 'warning': 'ескерту', 'note': 'ескертпе', 'hint': 'кеңес',
        'Error': 'Қате', 'Warning': 'Ескерту', 'Note': 'Ескертпе',
        'Syntax Error': 'Синтаксистік қате',
        'runtime error': 'орындау қатесі', 'fatal error': 'қатерлі қате',
        'at': '{indent}{func} ішінде', 'at_line': '{indent}{func} ішінде ({line}-жол)',
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# MESSAGES: (English, Russian, Kazakh), in three lists by who prints them.
# Conventions: translations start with a lowercase letter (they follow
# "ошибка: ..." / "қате: ..."); Russian avoids number agreement with
# placeholders ("аргументов: 2"); Kazakh never attaches a suffix to a
# placeholder (the suffix goes on the following noun: "'f' функциясы").
# A placeholder is a word unless it is quoted or has a kind (see KINDS below):
# mark free text `{x:text}` so that only real diagnostics match.
# ═══════════════════════════════════════════════════════════════════════════

COMPILER_MESSAGES: List[Tuple[str, str, str]] = [
    # ── lexer (syntax errors) ──────────────────────────────────────────────
    ("Unexpected character: '{ch}'",
     "неожиданный символ: '{ch}'",
     "күтілмеген таңба: '{ch}'"),
    ("Unterminated attribute",
     "незакрытый атрибут",
     "атрибут жабылмаған"),
    ("Unterminated string",
     "незакрытый строковый литерал",
     "жол литералы жабылмаған"),
    ("Invalid \\x escape: expected two hex digits",
     "неверная escape-последовательность \\x: ожидались две шестнадцатеричные цифры",
     "жарамсыз \\x escape-тізбегі: екі он алтылық цифр күтілді"),
    ("Invalid \\u{{...}} escape",
     "неверная escape-последовательность \\u{{...}}",
     "жарамсыз \\u{{...}} escape-тізбегі"),
    ("Unterminated '{{' in f-string",
     "незакрытая '{{' в f-строке",
     "f-жолдағы '{{' жабылмаған"),
    ("Unterminated f-string",
     "незакрытая f-строка",
     "f-жол жабылмаған"),
    ("Invalid hexadecimal literal: '{text}'",
     "неверный шестнадцатеричный литерал: '{text}'",
     "жарамсыз он алтылық литерал: '{text}'"),
    ("Invalid binary literal: '{text}'",
     "неверный двоичный литерал: '{text}'",
     "жарамсыз екілік литерал: '{text}'"),
    ("Invalid octal literal: '{text}'",
     "неверный восьмеричный литерал: '{text}'",
     "жарамсыз сегіздік литерал: '{text}'"),
    ("Invalid {kind} literal: '{text}'",
     "неверный литерал ({kind}): '{text}'",
     "жарамсыз литерал ({kind}): '{text}'"),
    ("Invalid numeric literal: '{text}'",
     "неверный числовой литерал: '{text}'",
     "жарамсыз сандық литерал: '{text}'"),

    # ── parser (syntax errors) ─────────────────────────────────────────────
    ("Expected {expected}, got {got}",
     "ожидалось {expected}, получено {got}",
     "{expected} күтілді, бірақ {got} кездесті"),
    ("Expected struct, enum, fn, or trait after 'pub'",
     "после 'pub' ожидалось struct, enum, fn или trait",
     "'pub' сөзінен кейін struct, enum, fn немесе trait күтілді"),
    ("Expected function body",
     "ожидалось тело функции",
     "функция денесі күтілді"),
    ("Expected method definition in impl block",
     "в блоке impl ожидалось определение метода",
     "impl блогында әдіс анықтамасы күтілді"),
    ("Expected method definition in trait",
     "в trait ожидалось определение метода",
     "trait ішінде әдіс анықтамасы күтілді"),
    ("Unexpected token: {token}",
     "неожиданный токен: {token}",
     "күтілмеген токен: {token}"),
    # generic functions and type inference (written against the til31/typed branch)
    ("Expected a type parameter name",
     "ожидалось имя параметра типа",
     "тип параметрінің атауы күтілді"),
    ("Expected '>' after the type parameters",
     "после параметров типа ожидалась '>'",
     "тип параметрлерінен кейін '>' күтілді"),
    ("check the spelling, or add the field / method to the struct",
     "проверьте написание или добавьте поле / метод в структуру",
     "жазылуын тексеріңіз немесе өрісті / әдісті құрылымға қосыңыз"),
    ("convert the value explicitly (str(x), int(x), float(x)) or change the annotation",
     "преобразуйте значение явно (str(x), int(x), float(x)) или измените аннотацию типа",
     "мәнді анық түрлендіріңіз (str(x), int(x), float(x)) немесе тип аннотациясын өзгертіңіз"),
    # appended by the parser of til31/effects when a keyword stands where a name is expected
    ("{message:msg} ('{word}' is a reserved word and cannot be used as a name)",
     "{message} ('{word}' — зарезервированное слово, его нельзя использовать как имя)",
     "{message} ('{word}' — резервтелген сөз, оны атау ретінде қолдануға болмайды)"),

    # ── hints for syntax errors (ERROR_HINTS) ──────────────────────────────
    ("Variable or function name expected here",
     "здесь ожидается имя переменной или функции",
     "мұнда айнымалының немесе функцияның атауы күтіледі"),
    ("Missing closing parenthesis ')'",
     "не хватает закрывающей круглой скобки ')'",
     "жабатын ')' жақшасы жетіспейді"),
    ("Missing closing bracket ']'",
     "не хватает закрывающей квадратной скобки ']'",
     "жабатын ']' жақшасы жетіспейді"),
    ("Missing closing brace '}}'",
     "не хватает закрывающей фигурной скобки '}}'",
     "жабатын '}}' жақшасы жетіспейді"),
    ("Expected an indented block (use 4 spaces)",
     "ожидался блок с отступом (используйте 4 пробела)",
     "шегініспен жазылған блок күтілді (4 бос орын қолданыңыз)"),
    ("Unexpected end of line. Check for missing operators or parentheses",
     "неожиданный конец строки. Проверьте, не пропущены ли операторы или скобки",
     "жол күтпеген жерде аяқталды. Оператор немесе жақша түсіп қалмағанын тексеріңіз"),
    ("This character is not valid in TIL",
     "этот символ недопустим в TIL",
     "бұл таңбаны TIL тілінде қолдануға болмайды"),

    # ── effects, capabilities, contracts, arity (E-EFFECT, E-CAP, ...) ─────
    ("pure function '{func}' performs effect '{effect}'",
     "чистая функция '{func}' выполняет эффект '{effect}'",
     "'{func}' таза функциясы '{effect}' әсерін тудырады"),
    ("pure function '{func}' performs effect '{effect}' (through the call to '{via}')",
     "чистая функция '{func}' выполняет эффект '{effect}' (через вызов '{via}')",
     "'{func}' таза функциясы '{effect}' әсерін тудырады ('{via}' шақыруы арқылы)"),
    ("function '{func}' performs effect '{effect}', but declares only #[effects: {allowed:text}]",
     "функция '{func}' выполняет эффект '{effect}', но объявляет только #[effects: {allowed}]",
     "'{func}' функциясы '{effect}' әсерін тудырады, бірақ тек #[effects: {allowed}] жарияланған"),
    ("function '{func}' performs effect '{effect}' (through the call to '{via}'), "
     "but declares only #[effects: {allowed:text}]",
     "функция '{func}' выполняет эффект '{effect}' (через вызов '{via}'), "
     "но объявляет только #[effects: {allowed}]",
     "'{func}' функциясы '{effect}' әсерін тудырады ('{via}' шақыруы арқылы), "
     "бірақ тек #[effects: {allowed}] жарияланған"),
    ("remove #[pure] or declare #[effects: {effect}]",
     "уберите #[pure] или объявите #[effects: {effect}]",
     "#[pure] белгісін алып тастаңыз немесе #[effects: {effect}] деп жариялаңыз"),
    ("add '{effect}' to the #[effects: ...] list",
     "добавьте '{effect}' в список #[effects: ...]",
     "'{effect}' әсерін #[effects: ...] тізіміне қосыңыз"),
    ("'{func}' uses '{callee}', which requires capability '{cap}', but '{func}' does not hold it",
     "'{func}' использует функцию '{callee}', которой требуется полномочие '{cap}', но у '{func}' его нет",
     "'{func}' функциясы '{callee}' функциясын пайдаланады, оған '{cap}' өкілеттігі керек, "
     "бірақ '{func}' функциясында бұл өкілеттік жоқ"),
    ("add #[cap: {cap}] to '{func}' (only main holds every capability)",
     "добавьте #[cap: {cap}] к '{func}' (все полномочия есть только у main)",
     "'{func}' функциясына #[cap: {cap}] қосыңыз (барлық өкілеттік тек main функциясында бар)"),
    ("'{func}' expects {lo} to {hi} argument(s), got {got}",
     "'{func}' ожидает аргументов: от {lo} до {hi}, передано: {got}",
     "'{func}' {lo}–{hi} аргумент күтеді, бірақ {got} берілді"),
    ("'{func}' expects {count} argument(s), got {got}",
     "'{func}' ожидает аргументов: {count}, передано: {got}",
     "'{func}' {count} аргумент күтеді, бірақ {got} берілді"),
    ("contract violation: call to '{func}' ({args:text}) breaks #[requires: {cond:text}]",
     "нарушение контракта: вызов '{func}' ({args}) не удовлетворяет #[requires: {cond}]",
     "келісімшарт бұзылды: '{func}' шақыруы ({args}) #[requires: {cond}] шартын бұзады"),
    ("proved false at compile time; fix the arguments",
     "на этапе компиляции доказано, что условие ложно; исправьте аргументы",
     "шарттың жалған екені компиляция кезінде дәлелденді; аргументтерді түзетіңіз"),

    # ── contract proofs (E-PROVE, src/tilprove.py: til prove, til check) ────
    # The obligation (`{what}`), the reason a function is unsupported and the
    # outcome of a counterexample run are messages of their own (below).
    ("cannot prove '{func}': unsupported: {reason:msg}",
     "нельзя доказать '{func}': не поддерживается: {reason}",
     "'{func}' функциясын дәлелдеу мүмкін емес: қолдау жоқ: {reason}"),
    ("cannot prove '{func}' (the global initializers): unsupported: {reason:msg}",
     "нельзя доказать '{func}' (инициализаторы глобальных переменных): не поддерживается: {reason}",
     "'{func}' (ғаламдық айнымалылардың бастапқы мәндері) дәлелдеу мүмкін емес: қолдау жоқ: {reason}"),
    ("the prover models int and bool values, if/elif/else, loops with `invariant`, and calls; "
     "see docs/TIL_PROOFS.md (til prove --allow-unsupported skips such functions)",
     "модуль доказательства моделирует значения int и bool, if/elif/else, циклы с `invariant` и вызовы; "
     "см. docs/TIL_PROOFS.md (til prove --allow-unsupported пропускает такие функции)",
     "дәлелдеу модулі int және bool мәндерін, if/elif/else, `invariant` бар циклдерді және шақыруларды "
     "модельдейді; docs/TIL_PROOFS.md қараңыз (til prove --allow-unsupported мұндай функцияларды "
     "өткізіп жібереді)"),
    ("'{func}': {what:msg} fails for {inputs:text}",
     "'{func}': {what} нарушается при {inputs}",
     "'{func}': {inputs} кезінде {what} орындалмайды"),
    ("'{func}': {what:msg} fails for no inputs (it fails on every run)",
     "'{func}': {what} нарушается при любых входных данных (функция не проходит ни при одном запуске)",
     "'{func}': {what} кез келген кіріс деректерімен орындалмайды (әр іске қосуда қате шығады)"),
    ("'{func}' (the global initializers): {what:msg} fails for no inputs (it fails on every run)",
     "'{func}' (инициализаторы глобальных переменных): {what} нарушается при любых входных данных "
     "(не проходит ни при одном запуске)",
     "'{func}' (ғаламдық айнымалылардың бастапқы мәндері): {what} кез келген кіріс деректерімен "
     "орындалмайды (әр іске қосуда қате шығады)"),
    ("'{func}': cannot prove {what:msg}",
     "'{func}': не удаётся доказать {what}",
     "'{func}': {what} дәлелденбеді"),
    ("'{func}': could not prove {what:msg} (unknown: timeout, {ms:int} ms per query)",
     "'{func}': не удалось доказать {what} (неизвестно: истекло время, {ms} мс на запрос)",
     "'{func}': {what} дәлелденбеді (белгісіз: уақыт бітті, бір сұрауға {ms} мс)"),
    ("'{func}': could not prove {what:msg} (unknown: {why:text}, {ms:int} ms per query)",
     "'{func}': не удалось доказать {what} (неизвестно: {why}, {ms} мс на запрос)",
     "'{func}': {what} дәлелденбеді (белгісіз: {why}, бір сұрауға {ms} мс)"),
    ("'{func}' (the global initializers): {what:msg} fails for {inputs:text}",
     "'{func}' (инициализаторы глобальных переменных): {what} нарушается при {inputs}",
     "'{func}' (ғаламдық айнымалылардың бастапқы мәндері): {inputs} кезінде {what} орындалмайды"),
    ("'{func}' (the global initializers): cannot prove {what:msg}",
     "'{func}' (инициализаторы глобальных переменных): не удаётся доказать {what}",
     "'{func}' (ғаламдық айнымалылардың бастапқы мәндері): {what} дәлелденбеді"),
    ("'{func}' (the global initializers): could not prove {what:msg} (unknown: timeout, {ms:int} ms per query)",
     "'{func}' (инициализаторы глобальных переменных): не удалось доказать {what} (неизвестно: истекло "
     "время, {ms} мс на запрос)",
     "'{func}' (ғаламдық айнымалылардың бастапқы мәндері): {what} дәлелденбеді (белгісіз: уақыт бітті, "
     "бір сұрауға {ms} мс)"),
    ("'{func}' (the global initializers): could not prove {what:msg} (unknown: {why:text}, {ms:int} ms "
     "per query)",
     "'{func}' (инициализаторы глобальных переменных): не удалось доказать {what} (неизвестно: {why}, "
     "{ms} мс на запрос)",
     "'{func}' (ғаламдық айнымалылардың бастапқы мәндері): {what} дәлелденбеді (белгісіз: {why}, бір "
     "сұрауға {ms} мс)"),
    ("internal error in the prover ({kind}: {detail:text})",
     "внутренняя ошибка модуля доказательства ({kind}: {detail})",
     "дәлелдеу модулінің ішкі қатесі ({kind}: {detail})"),
    ("til check --prove needs the Z3 SMT solver, which is not installed",
     "для til check --prove нужен SMT-решатель Z3, но он не установлен",
     "til check --prove үшін Z3 SMT шешушісі керек, бірақ ол орнатылмаған"),
    # what a proof obligation is about: "'f': division by zero fails for b = 0"
    ("#[requires: {cond:text}] of '{func}'",
     "#[requires: {cond}] функции '{func}'",
     "'{func}' функциясының #[requires: {cond}] шарты"),
    ("#[requires: {cond:text}] of '{func}' (called from '{caller}')",
     "#[requires: {cond}] функции '{func}' (вызов из '{caller}')",
     "'{func}' функциясының #[requires: {cond}] шарты ('{caller}' ішінен шақырылғанда)"),
    ("argument '{name}' of '{func}' is a bool (the proof of '{func}' assumes it)",
     "аргумент '{name}' функции '{func}' имеет тип bool (доказательство '{func}' исходит из этого)",
     "'{func}' функциясының '{name}' аргументі bool типті ('{func}' дәлелдеуі осыған сүйенеді)"),
    ("the bounds of the for loop",
     "границы цикла for",
     "for циклінің шекаралары"),
    ("loop invariant `{cond:text}` when the loop is reached",
     "инвариант цикла `{cond}` при входе в цикл",
     "циклге кірген кездегі `{cond}` цикл инварианты"),
    ("loop invariant `{cond:text}` after an iteration",
     "инвариант цикла `{cond}` после итерации",
     "итерациядан кейінгі `{cond}` цикл инварианты"),
    ("integer overflow",
     "целочисленное переполнение",
     "бүтін санның асып кетуі"),
    ("modulo by zero",
     "остаток от деления на ноль",
     "нөлге бөлгендегі қалдық"),
    ("division by zero in #[requires: {cond:text}] of '{func}'",
     "деление на ноль в #[requires: {cond}] функции '{func}'",
     "'{func}' функциясының #[requires: {cond}] шартындағы нөлге бөлу"),
    ("modulo by zero in #[requires: {cond:text}] of '{func}'",
     "остаток от деления на ноль в #[requires: {cond}] функции '{func}'",
     "'{func}' функциясының #[requires: {cond}] шартындағы нөлге бөлгендегі қалдық"),
    ("{what:msg} (in the default value of parameter '{param}')",
     "{what} (в значении по умолчанию параметра '{param}')",
     "{what} ('{param}' параметрінің әдепкі мәнінде)"),
    ("{what:msg} (in the default value of field '{field}' of '{struct}')",
     "{what} (в значении по умолчанию поля '{field}' структуры '{struct}')",
     "{what} ('{struct}' құрылымының '{field}' өрісінің әдепкі мәнінде)"),
    ("{what:msg} (in the default value of field '{field}' of '{struct}', line {line:int})",
     "{what} (в значении по умолчанию поля '{field}' структуры '{struct}', строка {line})",
     "{what} ('{struct}' құрылымының '{field}' өрісінің әдепкі мәнінде, {line}-жол)"),
    ("the contract of '{func}' calls '{func}' itself (it would never finish)",
     "контракт '{func}' вызывает саму функцию '{func}' (это никогда бы не завершилось)",
     "'{func}' келісімшарты '{func}' функциясының өзін шақырады (бұл ешқашан аяқталмас еді)"),
    # why a function is unsupported
    ("'break'/'continue' outside of a loop",
     "'break'/'continue' вне цикла",
     "'break'/'continue' циклден тыс"),
    ("'match' is not supported",
     "'match' не поддерживается",
     "'match' қолдау көрсетілмейді"),
    ("'{name}' here is a local that shadows the constant '{name}' (the prover does not model such "
     "shadowing; rename the local)",
     "'{name}' здесь — локальная переменная, которая скрывает константу '{name}' (доказатель не "
     "моделирует такое скрытие; переименуйте локальную переменную)",
     "'{name}' мұнда — '{name}' тұрақтысын жасыратын жергілікті айнымалы (дәлелдеуіш мұндай жасыруды "
     "модельдемейді; жергілікті айнымалының атын өзгертіңіз)"),
    ("call passes too few arguments",
     "в вызове слишком мало аргументов",
     "шақыруда аргументтер тым аз"),
    ("division by {what:msg}: the prover cannot rule out division by zero",
     "деление на значение ({what}): модуль доказательства не может исключить деление на ноль",
     "мәнге бөлу ({what}): дәлелдеу модулі нөлге бөлуді жоққа шығара алмайды"),
    ("the condition depends on {what:msg}, which the prover does not model",
     "условие зависит от значения, которое модуль доказательства не моделирует: {what}",
     "шарт дәлелдеу модулі модельдемейтін мәнге тәуелді: {what}"),
    ("function '{name}' (which has #[requires]) is used as a value; calls through function values are "
     "not checked",
     "функция '{name}' (у неё есть #[requires]) используется как значение; вызовы через значения-функции "
     "не проверяются",
     "'{name}' функциясы (оның #[requires] шарты бар) мән ретінде қолданылады; функция-мәндер арқылы "
     "шақырулар тексерілмейді"),
    ("method call '.{name}(...)': methods with #[requires] are not supported",
     "вызов метода '.{name}(...)': методы с #[requires] не поддерживаются",
     "'.{name}(...)' әдісін шақыру: #[requires] бар әдістерге қолдау көрсетілмейді"),
    ("methods with 'self' are not supported (the receiver is a struct)",
     "методы с 'self' не поддерживаются (получатель — структура)",
     "'self' бар әдістерге қолдау көрсетілмейді (қабылдаушы — құрылым)"),
    ("nested definitions are not supported",
     "вложенные определения не поддерживаются",
     "ішкі анықтамаларға қолдау көрсетілмейді"),
    ("operator '{op}' is not supported",
     "оператор '{op}' не поддерживается",
     "'{op}' операторына қолдау көрсетілмейді"),
    ("this assignment target is not supported",
     "такая цель присваивания не поддерживается",
     "мұндай меншіктеу нысанасына қолдау көрсетілмейді"),
    ("variable '{name}' changes its type inside the loop",
     "переменная '{name}' меняет тип внутри цикла",
     "'{name}' айнымалысы цикл ішінде типін өзгертеді"),
    # ("{construct} are not supported": one entry per construct, the generic one keeps it English)
    ("{construct} are not supported",
     "{construct}: не поддерживаются",
     "{construct}: қолдау көрсетілмейді"),
    ("{construct} is not supported",
     "{construct} не поддерживается",
     "{construct} қолдау көрсетілмейді"),
    ("closures (lambdas) are not supported",
     "замыкания (лямбды) не поддерживаются",
     "тұйықтамаларға (лямбдаларға) қолдау көрсетілмейді"),
    ("'match' are not supported",
     "'match' не поддерживается",
     "'match' қолдау көрсетілмейді"),
    ("'?' (Option/Result) are not supported",
     "'?' (Option/Result) не поддерживается",
     "'?' (Option/Result) қолдау көрсетілмейді"),
    ("retry blocks are not supported",
     "блоки retry не поддерживаются",
     "retry блоктарына қолдау көрсетілмейді"),
    ("timeout blocks are not supported",
     "блоки timeout не поддерживаются",
     "timeout блоктарына қолдау көрсетілмейді"),
    ("'intent' blocks are not supported",
     "блоки 'intent' не поддерживаются",
     "'intent' блоктарына қолдау көрсетілмейді"),
    ("list comprehensions are not supported",
     "генераторы списков не поддерживаются",
     "тізім генераторларына қолдау көрсетілмейді"),
    ("effect handlers are not supported",
     "обработчики эффектов не поддерживаются",
     "әсер өңдеушілеріне қолдау көрсетілмейді"),
    ("'perform' are not supported",
     "'perform' не поддерживается",
     "'perform' қолдау көрсетілмейді"),
    ("block expressions are not supported",
     "блочные выражения не поддерживаются",
     "блоктық өрнектерге қолдау көрсетілмейді"),
    ("channels are not supported",
     "каналы не поддерживаются",
     "арналарға қолдау көрсетілмейді"),
    ("'fallback' are not supported",
     "'fallback' не поддерживается",
     "'fallback' қолдау көрсетілмейді"),
    # values the prover does not model ("the condition depends on a string, ...")
    ("an unknown value",
     "неизвестное значение",
     "белгісіз мән"),
    ("a collection",
     "коллекция",
     "жиынтық"),
    ("a conditional expression with values of different types",
     "условное выражение со значениями разных типов",
     "мәндері әртүрлі типті шартты өрнек"),
    ("a constant",
     "константа",
     "тұрақты"),
    ("a default value",
     "значение по умолчанию",
     "әдепкі мән"),
    ("a field",
     "поле",
     "өріс"),
    ("a float",
     "число с плавающей точкой",
     "нақты сан"),
    ("a string",
     "строка",
     "жол"),
    ("a map",
     "словарь",
     "сөздік"),
    ("an array",
     "массив",
     "бір массив"),
    ("a range",
     "диапазон",
     "ауқым"),
    ("an element",
     "элемент",
     "бір элемент"),
    ("an element of an array, string or map",
     "элемент массива, строки или словаря",
     "массивтің, жолдың немесе сөздіктің элементі"),
    ("an element of the loop at line {line:int}",
     "элемент цикла в строке {line}",
     "{line}-жолдағы цикл элементі"),
    ("a value of type {type}",
     "значение типа {type}",
     "{type} типті мән"),
    ("a {name} value",
     "значение {name}",
     "{name} мәні"),
    ("a result of type float",
     "результат типа float",
     "float типті нәтиже"),
    ("a result of type {type} (narrower than 64 bits in native code)",
     "результат типа {type} (в нативном коде он уже 64 бит)",
     "{type} типті нәтиже (нативті кодта ол 64 биттен тар)"),
    ("a value cast to {type} (narrower than 64 bits in native code)",
     "значение, приведённое к {type} (в нативном коде он уже 64 бит)",
     "{type} типіне келтірілген мән (нативті кодта ол 64 биттен тар)"),
    ("'{op}' applied to a {sort}",
     "'{op}', применённый к значению {sort}",
     "{sort} мәніне қолданылған '{op}'"),
    ("'{op}' applied to a {left} and a {right} (a run-time type error)",
     "'{op}', применённый к значениям {left} и {right} (ошибка типов во время выполнения)",
     "{left} және {right} мәндеріне қолданылған '{op}' (орындау кезіндегі тип қатесі)"),
    ("parameter '{name}' (it has no type; annotate it as int or bool)",
     "параметр '{name}' (у него нет типа; укажите int или bool)",
     "'{name}' параметрі (оның типі жоқ; int немесе bool деп көрсетіңіз)"),
    ("parameter '{name}' (when it is left out it holds its default value, which is not a bool)",
     "параметр '{name}' (когда он не передан, в нём значение по умолчанию, а оно не bool)",
     "'{name}' параметрі (берілмегенде оның әдепкі мәні болады, ал ол bool емес)"),
    ("parameter '{name}' before it is set",
     "параметр '{name}' до того, как он задан",
     "мәні берілмей тұрған '{name}' параметрі"),
    ("parameter '{name}' of type float",
     "параметр '{name}' типа float",
     "float типті '{name}' параметрі"),
    ("parameter '{name}' of type {type}",
     "параметр '{name}' типа {type}",
     "{type} типті '{name}' параметрі"),
    ("parameter '{name}' of type {type} (narrower than 64 bits in native code)",
     "параметр '{name}' типа {type} (в нативном коде он уже 64 бит)",
     "{type} типті '{name}' параметрі (нативті кодта ол 64 биттен тар)"),
    ("variable '{name}' (it holds different types on different paths)",
     "переменная '{name}' (на разных путях выполнения в ней значения разных типов)",
     "'{name}' айнымалысы (әртүрлі орындалу жолдарында онда әртүрлі типті мәндер болады)"),
    ("variable '{name}' of type float",
     "переменная '{name}' типа float",
     "float типті '{name}' айнымалысы"),
    ("variable '{name}' of type {type} (narrower than 64 bits in native code)",
     "переменная '{name}' типа {type} (в нативном коде она уже 64 бит)",
     "{type} типті '{name}' айнымалысы (нативті кодта ол 64 биттен тар)"),
    ("the field '{name}'",
     "поле '{name}'",
     "'{name}' өрісі"),
    ("the function value '{name}'",
     "значение-функция '{name}'",
     "'{name}' функция-мәні"),
    ("the type '{name}'",
     "тип '{name}'",
     "'{name}' типі"),
    ("the global '{name}' (computed at run time)",
     "глобальная переменная '{name}' (вычисляется во время выполнения)",
     "'{name}' ғаламдық айнымалысы (орындау кезінде есептеледі)"),
    ("the global '{name}' (the script initializes it)",
     "глобальная переменная '{name}' (её инициализирует сценарий)",
     "'{name}' ғаламдық айнымалысы (оған бастапқы мәнді сценарий береді)"),
    ("the global '{name}' (it is initialized after functions have started to run - the initializer of "
     "'{other}' runs them - and they can read it while it is nil)",
     "глобальная переменная '{name}' (она инициализируется уже после того, как начали выполняться "
     "функции — их запускает инициализатор '{other}', — и они могут прочитать её, пока она равна nil)",
     "'{name}' ғаламдық айнымалысы (оған бастапқы мән функциялар орындала бастағаннан кейін беріледі — "
     "оларды '{other}' айнымалысының бастапқы мәні іске қосады — және олар оны nil кезінде оқи алады)"),
    ("the mutable global '{name}'",
     "изменяемая глобальная переменная '{name}'",
     "өзгермелі '{name}' ғаламдық айнымалысы"),
    ("the result of '{func}'",
     "результат '{func}'",
     "'{func}' нәтижесі"),
    ("the result of '{func}' (it can end without returning a value, and then the result is nil)",
     "результат '{func}' (функция может завершиться без возврата значения, и тогда результат — nil)",
     "'{func}' нәтижесі (функция мән қайтармай аяқталуы мүмкін, сонда нәтиже — nil)"),
    ("the result of '{func}' (it can return a `bool` parameter unchanged, and this call does not pass a "
     "bool there)",
     "результат '{func}' (функция может вернуть параметр `bool` без изменений, а этот вызов передаёт "
     "туда не bool)",
     "'{func}' нәтижесі (функция `bool` параметрін өзгертпей қайтаруы мүмкін, ал бұл шақыру оған bool "
     "бермейді)"),
    ("the result of '{func}' (its `-> bool` returns are not all booleans)",
     "результат '{func}' (не все её возвраты `-> bool` — логические значения)",
     "'{func}' нәтижесі (оның `-> bool` қайтарымдарының бәрі логикалық мән емес)"),
    ("the result of '{func}' of type float",
     "результат '{func}' типа float",
     "'{func}' нәтижесі, float типті"),
    ("the result of '{func}' of type {type} (narrower than 64 bits in native code)",
     "результат '{func}' типа {type} (в нативном коде он уже 64 бит)",
     "'{func}' нәтижесі, {type} типті (нативті кодта ол 64 биттен тар)"),
    ("the result of .{method}()",
     "результат .{method}()",
     ".{method}() нәтижесі"),
    ("the result of {name}()",
     "результат {name}()",
     "{name}() нәтижесі"),
    ("the result of calling a function value",
     "результат вызова значения-функции",
     "функция-мәнді шақыру нәтижесі"),
    ("the result of calling the global '{name}'",
     "результат вызова глобальной переменной '{name}'",
     "'{name}' ғаламдық айнымалысын шақыру нәтижесі"),
    ("the result of calling the value '{name}'",
     "результат вызова значения '{name}'",
     "'{name}' мәнін шақыру нәтижесі"),
    # hints of E-PROVE errors
    ("the solver found {inputs:text} ({extras:text}), but {outcome:msg}; it relies on {deps:text}; {advice:msg}",
     "решатель нашёл {inputs} ({extras}), но {outcome}; это опирается на {deps}; {advice}",
     "шешуші {inputs} ({extras}) мәндерін тапты, бірақ {outcome}; бұл {deps} мәндеріне сүйенеді; {advice}"),
    ("the solver found {inputs:text} ({extras:text}), but {outcome:msg}; {advice:msg}",
     "решатель нашёл {inputs} ({extras}), но {outcome}; {advice}",
     "шешуші {inputs} ({extras}) мәндерін тапты, бірақ {outcome}; {advice}"),
    ("the solver found {inputs:text}, but {outcome:msg}; it relies on {deps:text}; {advice:msg}",
     "решатель нашёл {inputs}, но {outcome}; это опирается на {deps}; {advice}",
     "шешуші {inputs} мәндерін тапты, бірақ {outcome}; бұл {deps} мәндеріне сүйенеді; {advice}"),
    ("the solver found {inputs:text}, but {outcome:msg}; {advice:msg}",
     "решатель нашёл {inputs}, но {outcome}; {advice}",
     "шешуші {inputs} мәндерін тапты, бірақ {outcome}; {advice}"),
    ("the loop invariant (`invariant ...` first in the loop body) may be missing or too weak",
     "инварианта цикла (`invariant ...` в начале тела цикла) может не хватать, или он слишком слаб",
     "цикл инварианты (цикл денесінің басындағы `invariant ...`) жетіспеуі немесе тым әлсіз болуы мүмкін"),
    ("the #[ensures] of the called function may be too weak",
     "#[ensures] вызываемой функции может быть слишком слабым",
     "шақырылатын функцияның #[ensures] шарты тым әлсіз болуы мүмкін"),
    ("add #[ensures] to the called function: without it its result is unknown",
     "добавьте #[ensures] вызываемой функции: без него её результат неизвестен",
     "шақырылатын функцияға #[ensures] қосыңыз: онсыз оның нәтижесі белгісіз"),
    ("it depends on values or operations the prover does not model",
     "это зависит от значений или операций, которые модуль доказательства не моделирует",
     "бұл дәлелдеу модулі модельдемейтін мәндерге немесе операцияларға тәуелді"),
    ("add #[requires] bounds or a stronger contract",
     "добавьте границы в #[requires] или более сильный контракт",
     "#[requires] арқылы шектер немесе күштірек келісімшарт қосыңыз"),
    ("raise --timeout, add bounds with #[requires: ...], or split the computation",
     "увеличьте --timeout, добавьте границы через #[requires: ...] или разбейте вычисление на части",
     "--timeout мәнін арттырыңыз, #[requires: ...] арқылы шектер қосыңыз немесе есептеуді бөліңіз"),
    ("raise --memory (or TIL_PROVE_MEMORY_MB), add bounds with #[requires: ...], or split the computation",
     "увеличьте --memory (или TIL_PROVE_MEMORY_MB), добавьте границы через #[requires: ...] или разбейте "
     "вычисление на части",
     "--memory (немесе TIL_PROVE_MEMORY_MB) мәнін арттырыңыз, #[requires: ...] арқылы шектер қосыңыз "
     "немесе есептеуді бөліңіз"),
    ("with these inputs {extras:text}; confirmed on the TIL VM ({outcome:msg})",
     "при этих входных данных {extras}; подтверждено на TIL VM ({outcome})",
     "осы кіріс деректерімен {extras}; TIL VM-де расталды ({outcome})"),
    ("with these inputs {extras:text}; not run on the VM: {outcome:msg}",
     "при этих входных данных {extras}; на VM не запускалось: {outcome}",
     "осы кіріс деректерімен {extras}; VM-де іске қосылмады: {outcome}"),
    ("with these inputs {extras:text}; not run on the VM",
     "при этих входных данных {extras}; на VM не запускалось",
     "осы кіріс деректерімен {extras}; VM-де іске қосылмады"),
    ("confirmed on the TIL VM ({outcome:msg})",
     "подтверждено на TIL VM ({outcome})",
     "TIL VM-де расталды ({outcome})"),
    ("not run on the VM: {outcome:msg}",
     "на VM не запускалось: {outcome}",
     "VM-де іске қосылмады: {outcome}"),
    ("not run on the VM",
     "на VM не запускалось",
     "VM-де іске қосылмады"),
    # what happened when a counterexample was run on the reference VM
    ("this is not a real run",
     "это не настоящий запуск",
     "бұл нақты іске қосу емес"),
    ("running it fails with: {error:msg}",
     "при запуске возникает ошибка: {error}",
     "іске қосқанда қате шығады: {error}"),
    ("running it fails with: {error:msg} (found by running edge values; the solver's candidate relied on "
     "an abstraction)",
     "при запуске возникает ошибка: {error} (найдено перебором граничных значений; кандидат решателя "
     "опирался на абстракцию)",
     "іске қосқанда қате шығады: {error} (шекаралық мәндерді іске қосып табылды; шешуші ұсынған нұсқа "
     "абстракцияға сүйенген)"),
    ("running it with these inputs does not fail",
     "запуск с этими входными данными завершается без ошибки",
     "осы кіріс деректерімен іске қосу қатесіз аяқталады"),
    ("running it with these inputs did not finish within {steps:int} steps",
     "запуск с этими входными данными не завершился за {steps} шагов",
     "осы кіріс деректерімен іске қосу {steps} қадам ішінде аяқталмады"),
    ("running it with these inputs fails differently: {error:msg}",
     "запуск с этими входными данными завершается другой ошибкой: {error}",
     "осы кіріс деректерімен іске қосу басқа қатемен аяқталады: {error}"),
    ("running it with these inputs fails differently: {error:msg} (in '{func}', line {line:int})",
     "запуск с этими входными данными завершается другой ошибкой: {error} (в '{func}', строка {line})",
     "осы кіріс деректерімен іске қосу басқа қатемен аяқталады: {error} ('{func}' ішінде, {line}-жол)"),
    ("it was not run: the program fails before, while the global initializers run: {error:msg}",
     "запуска не было: программа завершается ошибкой раньше, во время инициализации глобальных "
     "переменных: {error}",
     "іске қосылмады: бағдарлама одан бұрын, ғаламдық айнымалыларға бастапқы мән берілгенде қатемен "
     "аяқталады: {error}"),
    ("it was not run: the program fails before, while the global initializers run: {error:msg} "
     "(in '{func}', line {line:int})",
     "запуска не было: программа завершается ошибкой раньше, во время инициализации глобальных "
     "переменных: {error} (в '{func}', строка {line})",
     "іске қосылмады: бағдарлама одан бұрын, ғаламдық айнымалыларға бастапқы мән берілгенде қатемен "
     "аяқталады: {error} ('{func}' ішінде, {line}-жол)"),
    ("it was not run to the end: the run needs more than {mb:int} MB",
     "запуск не доведён до конца: ему нужно больше {mb} МБ",
     "іске қосу соңына дейін жеткізілмеді: оған {mb} МБ-тан көп жад керек"),
    ("it could not be run ({kind}: {detail:text})",
     "запустить не удалось ({kind}: {detail})",
     "іске қосу мүмкін болмады ({kind}: {detail})"),
    ("the call passes an int or nil, which the VM does not convert (not a run-time error)",
     "вызов передаёт int или nil, которые VM не преобразует (это не ошибка выполнения)",
     "шақыру int немесе nil береді, оларды VM түрлендірмейді (бұл орындау қатесі емес)"),
    ("it cannot be run: the prover has no value for the parameter {names:text}",
     "запустить нельзя: у модуля доказательства нет значения для параметра {names}",
     "іске қосу мүмкін емес: дәлелдеу модулінде параметрдің мәні жоқ: {names}"),
    ("it cannot be run: the prover has no value for the parameters {names:text}",
     "запустить нельзя: у модуля доказательства нет значений для параметров {names}",
     "іске қосу мүмкін емес: дәлелдеу модулінде параметрлердің мәндері жоқ: {names}"),
    ("it was not run: it reads the global {names:text}, which the script initializes and a run of the "
     "function alone cannot",
     "запуска не было: функция читает глобальную переменную {names}, которую инициализирует сценарий, "
     "а при запуске одной функции этого не происходит",
     "іске қосылмады: функция {names} ғаламдық айнымалысын оқиды, оған бастапқы мәнді сценарий береді, "
     "ал функцияны жеке іске қосқанда бұл болмайды"),
    ("it was not run: it reads the globals {names:text}, which the script initializes them and a run of "
     "the function alone cannot",
     "запуска не было: функция читает глобальные переменные {names}, которые инициализирует сценарий, "
     "а при запуске одной функции этого не происходит",
     "іске қосылмады: функция {names} ғаламдық айнымалыларын оқиды, оларға бастапқы мәнді сценарий "
     "береді, ал функцияны жеке іске қосқанда бұл болмайды"),

    # ── effects with resumption: handle / with / resume (TIL 3.1) ──────────
    # (written against the til31/effects branch, so its diagnostics are
    # translated as soon as it is merged; run-time ones: RUNTIME_MESSAGES)
    ("Expected 'with' and effect clauses after the body of 'handle'",
     "после тела 'handle' ожидались 'with' и ветви эффектов",
     "'handle' денесінен кейін 'with' және әсер тармақтары күтілді"),
    ("'handle ... with' needs at least one effect clause: Name(params) => ...",
     "в 'handle ... with' нужна хотя бы одна ветвь эффекта: Name(params) => ...",
     "'handle ... with' ішінде кемінде бір әсер тармағы болуы керек: Name(params) => ..."),
    ("Expected an effect clause: Name(params) => ...",
     "ожидалась ветвь эффекта: Name(params) => ...",
     "әсер тармағы күтілді: Name(params) => ..."),
    ("Expected a parameter name in the effect clause",
     "в ветви эффекта ожидалось имя параметра",
     "әсер тармағында параметр атауы күтілді"),
    ("duplicate parameter '{name}' in the clause for effect '{effect}'",
     "повторяющийся параметр '{name}' в ветви эффекта '{effect}'",
     "'{effect}' әсерінің тармағында '{name}' параметрі қайталанған"),
    ("Expected ')' after the clause parameters",
     "после параметров ветви ожидалась ')'",
     "тармақ параметрлерінен кейін ')' күтілді"),
    ("duplicate clause for effect '{effect}' in one handle",
     "повторная ветвь для эффекта '{effect}' в одном handle",
     "бір handle ішінде '{effect}' әсерінің тармағы қайталанған"),
    ("each effect may have only one clause per 'handle ... with'",
     "у каждого эффекта может быть только одна ветвь в одном 'handle ... with'",
     "бір 'handle ... with' ішінде әр әсердің тек бір тармағы болады"),
    ("Expected '=>' after the effect clause head",
     "после заголовка ветви эффекта ожидалось '=>'",
     "әсер тармағының басынан кейін '=>' күтілді"),
    ("Expected an expression or an indented block after '=>'",
     "после '=>' ожидалось выражение или блок с отступом",
     "'=>' белгісінен кейін өрнек немесе шегініспен жазылған блок күтілді"),
    ("unreachable code after resume(...)",
     "недостижимый код после resume(...)",
     "resume(...) кейінгі код ешқашан орындалмайды"),
    ("resume ends the clause and continues the handled body at once (tail resumption); "
     "it does not return here, so move this code before resume "
     "(write resume(v) as the clause's last statement)",
     "resume сразу завершает ветвь и продолжает обрабатываемое тело (хвостовое возобновление); "
     "сюда управление не возвращается, поэтому перенесите этот код до resume "
     "(resume(v) должен быть последней инструкцией ветви)",
     "resume тармақты бірден аяқтап, өңделетін денені жалғастырады (соңғы жалғастыру); "
     "басқару бұл жерге оралмайды, сондықтан бұл кодты resume алдына жылжытыңыз "
     "(resume(v) тармақтың соңғы нұсқауы болсын)"),
    ("Expected the effect name after 'perform'",
     "после 'perform' ожидалось имя эффекта",
     "'perform' сөзінен кейін әсер атауы күтілді"),
    ("'{word}' is reserved inside an effect clause",
     "'{word}' — зарезервированное слово внутри ветви эффекта",
     "әсер тармағының ішінде '{word}' сөзі резервтелген"),
    ("there it always means resume(...); choose another name",
     "там это слово всегда означает resume(...); выберите другое имя",
     "онда бұл сөз әрқашан resume(...) дегенді білдіреді; басқа атау таңдаңыз"),
    ("there it always means resume(...); use another variable name",
     "там это слово всегда означает resume(...); используйте другое имя переменной",
     "онда бұл сөз әрқашан resume(...) дегенді білдіреді; айнымалыға басқа атау беріңіз"),
    ("unknown function '{name}': resume(...) outside an effect clause",
     "неизвестная функция '{name}': resume(...) вне ветви эффекта",
     "'{name}' функциясы белгісіз: resume(...) әсер тармағынан тыс"),
    ("`perform {callee}(...)` calls the function '{callee}' when no handler is active; {hint:msg}",
     "`perform {callee}(...)` вызывает функцию '{callee}', когда нет активного обработчика; {hint}",
     "белсенді өңдеуші болмаса, `perform {callee}(...)` '{callee}' функциясын шақырады; {hint}"),
    ("resume(...) outside an effect clause",
     "resume(...) вне ветви эффекта",
     "resume(...) әсер тармағынан тыс"),
    ("resume continues the body of a 'handle' and may only be used in one of its 'with' clauses "
     "(not in a nested lambda)",
     "resume продолжает тело 'handle' и допустим только в одной из его ветвей 'with' "
     "(не во вложенной лямбде)",
     "resume 'handle' денесін жалғастырады және тек оның 'with' тармақтарында қолданылады "
     "(ішкі лямбдада емес)"),
    ("resume takes at most one value: resume(v)",
     "resume принимает не больше одного значения: resume(v)",
     "resume ең көбі бір мән қабылдайды: resume(v)"),
    ("'return' is not allowed in an effect clause",
     "'return' недопустим в ветви эффекта",
     "әсер тармағында 'return' қолдануға болмайды"),
    ("use resume(v) to continue the handled body, or end the clause with a value: "
     "that value becomes the value of the whole handle",
     "используйте resume(v), чтобы продолжить обрабатываемое тело, или завершите ветвь значением: "
     "оно станет значением всего handle",
     "өңделетін денені жалғастыру үшін resume(v) қолданыңыз немесе тармақты мәнмен аяқтаңыз: "
     "ол бүкіл handle өрнегінің мәні болады"),
    ("'break' cannot leave an effect clause",
     "'break' не может выйти из ветви эффекта",
     "'break' арқылы әсер тармағынан шығуға болмайды"),
    ("'continue' cannot leave an effect clause",
     "'continue' не может выйти из ветви эффекта",
     "'continue' арқылы әсер тармағынан шығуға болмайды"),
    ("finish the clause with a value (or resume) instead",
     "вместо этого завершите ветвь значением (или вызовите resume)",
     "оның орнына тармақты мәнмен аяқтаңыз (немесе resume қолданыңыз)"),
    ("too many effect clauses in one handle (max 255)",
     "слишком много ветвей эффектов в одном handle (максимум 255)",
     "бір handle ішінде әсер тармақтары тым көп (ең көбі 255)"),
    ("the clause of the enclosing handle takes a different number of parameters",
     "ветвь объемлющего handle принимает другое число параметров",
     "сыртқы handle тармағы параметрлердің басқа санын қабылдайды"),


    ("the match arms are never reached: resume(...) ends the clause",
     "ветви match никогда не выполняются: resume(...) завершает ветвь эффекта",
     "match тармақтары ешқашан орындалмайды: resume(...) әсер тармағын аяқтайды"),

    # ── one namespace for a program and its imports (E-IMPORT) ─────────────
    # "{kind} '{name}'": one entry per kind (the word order and the gender
    # depend on it); the generic one keeps the kind word as it is
    ("function '{name}' is defined in two files: {there:text} and {here:text}",
     "функция '{name}' определена в двух файлах: {there} и {here}",
     "'{name}' функциясы екі файлда анықталған: {there} және {here}"),
    ("struct '{name}' is defined in two files: {there:text} and {here:text}",
     "структура '{name}' определена в двух файлах: {there} и {here}",
     "'{name}' құрылымы екі файлда анықталған: {there} және {here}"),
    ("enum '{name}' is defined in two files: {there:text} and {here:text}",
     "перечисление '{name}' определено в двух файлах: {there} и {here}",
     "'{name}' санағы екі файлда анықталған: {there} және {here}"),
    ("trait '{name}' is defined in two files: {there:text} and {here:text}",
     "трейт '{name}' определён в двух файлах: {there} и {here}",
     "'{name}' трейті екі файлда анықталған: {there} және {here}"),
    ("type alias '{name}' is defined in two files: {there:text} and {here:text}",
     "псевдоним типа '{name}' определён в двух файлах: {there} и {here}",
     "'{name}' тип бүркеншік аты екі файлда анықталған: {there} және {here}"),
    ("capability '{name}' is defined in two files: {there:text} and {here:text}",
     "полномочие '{name}' определено в двух файлах: {there} и {here}",
     "'{name}' өкілеттігі екі файлда анықталған: {there} және {here}"),
    ("const '{name}' is defined in two files: {there:text} and {here:text}",
     "константа '{name}' определена в двух файлах: {there} и {here}",
     "'{name}' тұрақтысы екі файлда анықталған: {there} және {here}"),
    ("let '{name}' is defined in two files: {there:text} and {here:text}",
     "переменная '{name}' (let) определена в двух файлах: {there} и {here}",
     "'{name}' айнымалысы (let) екі файлда анықталған: {there} және {here}"),
    ("var '{name}' is defined in two files: {there:text} and {here:text}",
     "переменная '{name}' (var) определена в двух файлах: {there} и {here}",
     "'{name}' айнымалысы (var) екі файлда анықталған: {there} және {here}"),
    ("definition '{name}' is defined in two files: {there:text} and {here:text}",
     "определение '{name}' есть в двух файлах: {there} и {here}",
     "'{name}' анықтамасы екі файлда бар: {there} және {here}"),
    ("{kind} '{name}' is defined in two files: {there:text} and {here:text}",
     "{kind} '{name}': определено в двух файлах: {there} и {here}",
     "{kind} '{name}': екі файлда анықталған: {there} және {here}"),
    ("'{name}' is defined in two files: {kind:msg} at {there:text}, {other:msg} at {here:text}",
     "'{name}' определено в двух файлах: {kind} — {there}, {other} — {here}",
     "'{name}' екі файлда анықталған: {kind} — {there}, {other} — {here}"),
    # the kinds in that message (let, var and const are keywords: UNTRANSLATED)
    ("function",
     "функция",
     "функция анықтамасы"),
    ("struct",
     "структура",
     "құрылым"),
    ("enum",
     "перечисление",
     "санақ"),
    ("trait",
     "трейт",
     "трейт анықтамасы"),
    ("type alias",
     "псевдоним типа",
     "тип бүркеншік аты"),
    ("capability",
     "полномочие",
     "өкілеттік"),
    ("definition",
     "определение",
     "анықтама"),
    ("a program and everything it imports (modules, packages, also indirectly) share one namespace - "
     "TIL has no module-private names: rename one of the two (packages should prefix their internal "
     "names with the package name)",
     "у программы и всего, что она импортирует (модули, пакеты, в том числе косвенно), одно общее "
     "пространство имён — в TIL нет имён, скрытых внутри модуля: переименуйте одно из двух (пакетам "
     "стоит начинать свои внутренние имена с имени пакета)",
     "бағдарлама мен ол импорттайтынның бәрі (модульдер, пакеттер, жанама түрде де) бір атаулар "
     "кеңістігін бөліседі — TIL-де модуль ішінде жасырын атаулар жоқ: екеуінің бірінің атын өзгертіңіз "
     "(пакеттер өздерінің ішкі атауларын пакет атауынан бастағаны жөн)"),
    ("... and 1 more name defined in two files",
     "... и ещё одно имя, определённое в двух файлах",
     "... және екі файлда анықталған тағы бір атау"),
    ("... and {count:int} more names defined in two files",
     "... и ещё имена, определённые в двух файлах (их число: {count})",
     "... және екі файлда анықталған тағы {count} атау"),

    # ── type inference of the typed backend (E-TYPE, src/tiltypes.py) ──────
    # types are written as the typed backend shows them: [int], {str: int},
    # Option<int>, fn(int) -> str, str or int (the 'type' kind takes them)
    ("cannot compare {left:type} and {right:type} with {op}",
     "нельзя сравнить {left} и {right} с помощью {op}",
     "{left} және {right} мәндерін {op} арқылы салыстыруға болмайды"),
    ("bitwise operator {op} needs integers, got {left:type} and {right:type}",
     "побитовому оператору {op} нужны целые числа, получено: {left} и {right}",
     "{op} биттік операторына бүтін сандар керек, бірақ {left} және {right} берілді"),
    ("cannot apply unary {op} to {type:type}",
     "нельзя применить унарный оператор {op} к {type}",
     "{type} типіне унарлық {op} операторын қолдануға болмайды"),
    ("array index must be an integer, got {type:type}",
     "индекс массива должен быть целым числом, получено: {type}",
     "массив индексі бүтін сан болуы керек, бірақ {type} берілді"),
    ("this function takes {count:int} argument(s), got {got:int}",
     "эта функция принимает аргументов: {count}, передано: {got}",
     "бұл функция {count} аргумент қабылдайды, бірақ {got} берілді"),
    ("struct '{name}' has no method '{method}'",
     "у структуры '{name}' нет метода '{method}'",
     "'{name}' құрылымында '{method}' әдісі жоқ"),
    # "<what> is <declared>, but the value is <value>": one entry per <what>
    ("'{name}' is {declared:type}, but the value is {value:type}",
     "'{name}' имеет тип {declared}, но значение имеет тип {value}",
     "'{name}' типі — {declared}, бірақ мәні {value} типті"),
    ("'{name}' is {declared:type}, but the value is the string {text:text}, which is not a number",
     "'{name}' имеет тип {declared}, но значение — строка {text}, которая не является числом",
     "'{name}' типі — {declared}, бірақ мәні {text} жолы, ол сан емес"),
    ("argument '{name}' of '{func}' is {declared:type}, but the value is {value:type}",
     "аргумент '{name}' функции '{func}' имеет тип {declared}, но значение имеет тип {value}",
     "'{func}' функциясының '{name}' аргументінің типі — {declared}, бірақ мәні {value} типті"),
    ("argument '{name}' of '{func}' is {declared:type}, but the value is the string {text:text}, "
     "which is not a number",
     "аргумент '{name}' функции '{func}' имеет тип {declared}, но значение — строка {text}, "
     "которая не является числом",
     "'{func}' функциясының '{name}' аргументінің типі — {declared}, бірақ мәні {text} жолы, ол сан емес"),
    ("the default of '{name}' is {declared:type}, but the value is {value:type}",
     "значение по умолчанию '{name}' должно иметь тип {declared}, но имеет тип {value}",
     "'{name}' әдепкі мәнінің типі {declared} болуы керек, бірақ ол {value} типті"),
    ("the default of '{name}' is {declared:type}, but the value is the string {text:text}, "
     "which is not a number",
     "значение по умолчанию '{name}' должно иметь тип {declared}, но это строка {text}, "
     "которая не является числом",
     "'{name}' әдепкі мәнінің типі {declared} болуы керек, бірақ ол {text} жолы, ол сан емес"),
    ("the return value of '{func}' is {declared:type}, but the value is {value:type}",
     "возвращаемое значение '{func}' имеет тип {declared}, но значение имеет тип {value}",
     "'{func}' қайтаратын мәннің типі — {declared}, бірақ мәні {value} типті"),
    ("the return value of '{func}' is {declared:type}, but the value is the string {text:text}, "
     "which is not a number",
     "возвращаемое значение '{func}' имеет тип {declared}, но значение — строка {text}, "
     "которая не является числом",
     "'{func}' қайтаратын мәннің типі — {declared}, бірақ мәні {text} жолы, ол сан емес"),
    ("field '{field}' of '{struct}' is {declared:type}, but the value is {value:type}",
     "поле '{field}' структуры '{struct}' имеет тип {declared}, но значение имеет тип {value}",
     "'{struct}' құрылымының '{field}' өрісінің типі — {declared}, бірақ мәні {value} типті"),
    ("field '{field}' of '{struct}' is {declared:type}, but the value is the string {text:text}, "
     "which is not a number",
     "поле '{field}' структуры '{struct}' имеет тип {declared}, но значение — строка {text}, "
     "которая не является числом",
     "'{struct}' құрылымының '{field}' өрісінің типі — {declared}, бірақ мәні {text} жолы, ол сан емес"),

    # ── bytecode compiler (E-VM) ───────────────────────────────────────────
    ("operand out of range ({value}); too many locals/arguments?",
     "операнд вне допустимого диапазона ({value}); слишком много локальных переменных или аргументов?",
     "операнд рұқсат етілген ауқымнан тыс ({value}); жергілікті айнымалылар немесе аргументтер тым көп пе?"),
    ("program too large for the bytecode format",
     "программа слишком велика для формата байт-кода",
     "бағдарлама байт-код пішімі үшін тым үлкен"),
    ("function too large: jump offset exceeds 32 KB",
     "функция слишком велика: смещение перехода превышает 32 КБ",
     "функция тым үлкен: өту ығысуы 32 КБ шегінен асады"),
    ("too many local variables in one function (max 250)",
     "слишком много локальных переменных в одной функции (максимум 250)",
     "бір функцияда жергілікті айнымалылар тым көп (ең көбі 250)"),
    ("function '{func}' is defined twice",
     "функция '{func}' определена дважды",
     "'{func}' функциясы екі рет анықталған"),
    ("impl for unknown struct '{name}'",
     "impl для неизвестной структуры '{name}'",
     "белгісіз '{name}' құрылымына арналған impl"),
    ("impl {trait} for {name} is missing method '{method}'",
     "в impl {trait} for {name} нет метода '{method}'",
     "impl {trait} for {name} ішінде '{method}' әдісі жоқ"),
    ("enum values must be constants",
     "значения перечисления должны быть константами",
     "санақ (enum) мәндері тұрақты болуы керек"),
    ("main() takes no parameters",
     "main() не принимает параметров",
     "main() параметр қабылдамайды"),
    ("definitions are only allowed at the top level",
     "определения допустимы только на верхнем уровне",
     "анықтамаларға тек жоғарғы деңгейде рұқсат етіледі"),
    ("cannot assign twice to immutable variable '{name}'",
     "нельзя повторно присвоить значение неизменяемой переменной '{name}'",
     "өзгермейтін '{name}' айнымалысына екінші рет мән меншіктеуге болмайды"),
    ("cannot assign to immutable captured variable '{name}'",
     "нельзя присвоить значение неизменяемой переменной '{name}', захваченной из внешней области",
     "сыртқы аймақтан алынған өзгермейтін '{name}' айнымалысына мән меншіктеуге болмайды"),
    ("cannot assign to immutable global '{name}'",
     "нельзя присвоить значение неизменяемой глобальной переменной '{name}'",
     "өзгермейтін '{name}' ғаламдық айнымалысына мән меншіктеуге болмайды"),
    ("cannot assign to constant '{name}'",
     "нельзя присвоить значение константе '{name}'",
     "'{name}' тұрақтысына мән меншіктеуге болмайды"),
    ("declare it with 'var' to make it mutable",
     "объявите её через 'var', чтобы она стала изменяемой",
     "өзгермелі болуы үшін оны 'var' арқылы жариялаңыз"),
    ("unknown name '{name}'",
     "неизвестное имя '{name}'",
     "белгісіз атау: '{name}'"),
    ("unknown function '{name}'",
     "неизвестная функция '{name}'",
     "белгісіз функция: '{name}'"),
    ("unknown struct '{name}'",
     "неизвестная структура '{name}'",
     "белгісіз құрылым: '{name}'"),
    ("did you mean '{name}'?",
     "возможно, имелось в виду '{name}'?",
     "'{name}' деп жазғыңыз келді ме?"),
    ("invalid assignment target",
     "недопустимая цель присваивания",
     "меншіктеудің сол жағы жарамсыз"),
    ("'break' outside of a loop",
     "'break' вне цикла",
     "'break' циклден тыс"),
    ("'continue' outside of a loop",
     "'continue' вне цикла",
     "'continue' циклден тыс"),
    ("enum '{name}' has no variant '{variant}'",
     "у перечисления '{name}' нет варианта '{variant}'",
     "'{name}' санағында '{variant}' нұсқасы жоқ"),
    ("pattern expects {count} field(s), got {got}",
     "образец ожидает полей: {count}, получено: {got}",
     "үлгі {count} өріс күтеді, бірақ {got} берілді"),
    ("'{name}' is a type, not a value",
     "'{name}' — это тип, а не значение",
     "'{name}' — мән емес, тип"),
    ("no handler function '{func}' for effect '{effect}'",
     "нет функции-обработчика '{func}' для эффекта '{effect}'",
     "'{effect}' әсері үшін '{func}' өңдеуші функциясы жоқ"),
    ("{construct} is not supported by the VM backend",
     "{construct} не поддерживается бэкендом VM",
     "{construct} VM бэкендінде қолдау көрсетілмейді"),
    ("unsupported operator '{op}'",
     "неподдерживаемый оператор '{op}'",
     "қолдау көрсетілмейтін оператор: '{op}'"),
    ("too many arguments (max 255)",
     "слишком много аргументов (максимум 255)",
     "аргументтер тым көп (ең көбі 255)"),
    ("'{func}' takes at least {lo} argument(s), got {got}",
     "'{func}' принимает аргументов: не меньше {lo}, передано: {got}",
     "'{func}' кемінде {lo} аргумент қабылдайды, бірақ {got} берілді"),
    ("'{func}' takes {lo} to {hi} argument(s), got {got}",
     "'{func}' принимает аргументов: от {lo} до {hi}, передано: {got}",
     "'{func}' {lo}–{hi} аргумент қабылдайды, бірақ {got} берілді"),
    ("'{func}' takes {count} argument(s), got {got}",
     "'{func}' принимает аргументов: {count}, передано: {got}",
     "'{func}' {count} аргумент қабылдайды, бірақ {got} берілді"),
    ("'{name}(...)' needs {count} field value(s), got {got}",
     "'{name}(...)' требует значений полей: {count}, передано: {got}",
     "'{name}(...)' үшін {count} өріс мәні керек, бірақ {got} берілді"),
    ("{name}(...) takes one value",
     "{name}(...) принимает только одно значение",
     "{name}(...) тек бір мән қабылдайды"),
    ("struct '{name}' has no function '{func}'",
     "у структуры '{name}' нет функции '{func}'",
     "'{name}' құрылымында '{func}' функциясы жоқ"),
    ("'{name}.{variant}' takes {count} value(s), got {got}",
     "'{name}.{variant}' принимает значений: {count}, передано: {got}",
     "'{name}.{variant}' {count} мән қабылдайды, бірақ {got} берілді"),
    ("'{name}.{variant}' needs {count} value(s)",
     "'{name}.{variant}' требует значений: {count}",
     "'{name}.{variant}' үшін {count} мән керек"),
    ("module '{module}' has no function '{func}'",
     "в модуле '{module}' нет функции '{func}'",
     "'{module}' модулінде '{func}' функциясы жоқ"),
    ("unknown method '{method}'",
     "неизвестный метод '{method}'",
     "белгісіз әдіс: '{method}'"),
    ("struct '{name}' has no field '{field}'",
     "у структуры '{name}' нет поля '{field}'",
     "'{name}' құрылымында '{field}' өрісі жоқ"),
    ("f-string has too many parts",
     "в f-строке слишком много частей",
     "f-жолда бөліктер тым көп"),

    # ── other compile errors (CompileError, bytecode files) ────────────────
    ("string constant longer than 65535 bytes",
     "строковая константа длиннее 65535 байт",
     "жол тұрақтысы 65535 байттан ұзын"),
    ("not a TIL bytecode file (bad magic)",
     "это не файл байт-кода TIL (неверная сигнатура)",
     "бұл TIL байт-код файлы емес (файл сигнатурасы қате)"),
    ("unsupported bytecode version {version} (this VM loads versions {lo}..{hi})",
     "неподдерживаемая версия байт-кода {version} (эта VM загружает версии {lo}..{hi})",
     "байт-кодтың {version} нұсқасына қолдау жоқ (бұл VM {lo}..{hi} нұсқаларын жүктейді)"),
    ("bad constant tag {tag}",
     "неверный тег константы {tag}",
     "тұрақтының тегі қате: {tag}"),
    # the optional debug-info section (til build --debug-info, til debug)
    ("truncated debug-info section",
     "секция отладочной информации обрезана",
     "жөндеу ақпаратының бөлімі толық емес"),
    ("bad debug-info section (name {index:int} is not a string constant)",
     "неверная секция отладочной информации (имя {index} — не строковая константа)",
     "жөндеу ақпаратының бөлімі қате ({index} атауы жол тұрақтысы емес)"),
    ("bad debug-info section (function {index:int} does not exist)",
     "неверная секция отладочной информации (функции {index} нет)",
     "жөндеу ақпаратының бөлімі қате ({index} функциясы жоқ)"),
    ("program too large for debug info (more than 65535 variables or 65533 functions); "
     "build without --debug-info",
     "программа слишком велика для отладочной информации (больше 65535 переменных или 65533 функций); "
     "соберите без --debug-info",
     "бағдарлама жөндеу ақпараты үшін тым үлкен (65535-тен көп айнымалы немесе 65533-тен көп функция); "
     "--debug-info параметрінсіз құрастырыңыз"),
    ("cannot find the TIL C VM sources (vm/tilvm.c). Set TIL_HOME to the TIL directory.",
     "не найдены исходники C VM для TIL (vm/tilvm.c). Укажите каталог TIL в TIL_HOME.",
     "TIL C VM бастапқы кодтары табылмады (vm/tilvm.c). TIL_HOME айнымалысында TIL каталогын көрсетіңіз."),
    ("unknown language '{lang}' (use: {choices:text})",
     "неизвестный язык '{lang}' (доступны: {choices})",
     "белгісіз тіл: '{lang}' (мүмкін мәндер: {choices})"),
    # energy saving: #[energy: ...] and its .tbc header bits (TIL 3.1, docs/TIL_ENERGY.md)
    ("unknown energy mode '{mode}' (use low, balanced or performance)",
     "неизвестный режим энергосбережения '{mode}' (используйте low, balanced или performance)",
     "белгісіз энергия үнемдеу режимі: '{mode}' (low, balanced немесе performance қолданыңыз)"),
    ("unknown energy level '{level}' for a function (use #[energy: low])",
     "неизвестный уровень энергосбережения функции: '{level}' (используйте #[energy: low])",
     "функцияның белгісіз энергия деңгейі: '{level}' (#[energy: low] қолданыңыз)"),
    ("#[energy: ...] must be directly above a function, or come before every definition "
     "(the program's energy mode)",
     "#[energy: ...] должен стоять прямо над функцией или перед всеми определениями "
     "(режим энергосбережения программы)",
     "#[energy: ...] функцияның тура үстінде немесе барлық анықтамалардан бұрын тұруы керек "
     "(бағдарламаның энергия үнемдеу режимі)"),
    ("the program's energy mode is set twice",
     "режим энергосбережения программы задан дважды",
     "бағдарламаның энергия үнемдеу режимі екі рет берілген"),
    ("bad bytecode header (energy mode {mode:int})",
     "неверный заголовок байт-кода (режим энергосбережения {mode})",
     "байт-код тақырыбы қате (энергия үнемдеу режимі: {mode})"),

    # ── type checker warnings of the native backend ────────────────────────
    ("Line {line:int}: {message:msg}",
     "строка {line}: {message}",
     "{line}-жол: {message}"),
    ("Type mismatch: expected {expected:text}, got {got:text}",
     "несоответствие типов: ожидался {expected}, получен {got}",
     "типтер сәйкес емес: {expected} күтілді, бірақ {got} алынды"),
    ("Cannot assign to immutable variable '{name}'",
     "нельзя присвоить значение неизменяемой переменной '{name}'",
     "өзгермейтін '{name}' айнымалысына мән меншіктеуге болмайды"),
    ("Condition must be bool, got {got:text}",
     "условие должно иметь тип bool, получен {got}",
     "шарт bool типті болуы керек, бірақ {got} алынды"),

    # ── the native backend (til build; src/tiltypes.py, src/til.py) ────────
    # "native backend: <what it cannot compile> in 'f' (line 3)": the frame,
    # then each reason (code such as `str.upper()` stays as it is: UNTRANSLATED)
    ("native backend: {reason:msg} in '{func}' (line {line:int})",
     "нативный бэкенд: {reason} в '{func}' (строка {line})",
     "нативті бэкенд: {reason}, '{func}' ішінде ({line}-жол)"),
    ("native backend: {reason:msg} in '{func}'",
     "нативный бэкенд: {reason} в '{func}'",
     "нативті бэкенд: {reason}, '{func}' ішінде"),
    ("native backend: {reason:msg} (line {line:int})",
     "нативный бэкенд: {reason} (строка {line})",
     "нативті бэкенд: {reason} ({line}-жол)"),
    ("native backend: {reason:msg}",
     "нативный бэкенд: {reason}",
     "нативті бэкенд: {reason}"),
    ("type inference failed ({kind}: {detail:text})",
     "вывод типов не удался ({kind}: {detail})",
     "типтерді шығару сәтсіз аяқталды ({kind}: {detail})"),
    ("internal error ({kind}: {detail:text})",
     "внутренняя ошибка ({kind}: {detail})",
     "ішкі қате ({kind}: {detail})"),
    ("line {line:int}: effect '{effect}' has no default handler function; the TIL VM reports unhandled "
     "effects at run time",
     "строка {line}: у эффекта '{effect}' нет функции-обработчика по умолчанию; TIL VM сообщает о "
     "необработанных эффектах во время выполнения",
     "{line}-жол: '{effect}' әсерінің әдепкі өңдеуші функциясы жоқ; TIL VM өңделмеген әсерлер туралы "
     "орындау кезінде хабарлайды"),
    ("line {line:int}: the native backend does not support effect handlers (handle ... with / resume); "
     "run it on the TIL VM (til run, til run --vm) or build bytecode (til build --target vm)",
     "строка {line}: нативный бэкенд не поддерживает обработчики эффектов (handle ... with / resume); "
     "запустите программу на TIL VM (til run, til run --vm) или соберите байт-код (til build --target vm)",
     "{line}-жол: нативті бэкенд әсер өңдеушілерін қолдамайды (handle ... with / resume); бағдарламаны "
     "TIL VM-де іске қосыңыз (til run, til run --vm) немесе байт-код құрастырыңыз (til build --target vm)"),
    ("line {line:int}: the native backend does not support effect handlers (handle ... with); use the TIL VM",
     "строка {line}: нативный бэкенд не поддерживает обработчики эффектов (handle ... with); используйте TIL VM",
     "{line}-жол: нативті бэкенд әсер өңдеушілерін қолдамайды (handle ... with); TIL VM қолданыңыз"),
    ("line {line:int}: the classic C backend cannot keep the VMs' left-to-right order of evaluation here "
     "(an operand with side effects next to one it could change); run it on the TIL VM, or without --no-types",
     "строка {line}: классический C-бэкенд не может сохранить порядок вычисления слева направо, как в VM "
     "(операнд с побочными эффектами рядом с тем, что он может изменить); запустите программу на TIL VM "
     "или без --no-types",
     "{line}-жол: классикалық C бэкенді VM-дегідей солдан оңға қарай есептеу ретін сақтай алмайды "
     "(жанама әсері бар операнд өзі өзгерте алатын операндтың қасында); бағдарламаны TIL VM-де немесе "
     "--no-types-сіз іске қосыңыз"),
    ("'{func}' can end without returning a value (None), which a native {type:type} cannot hold",
     "'{func}' может завершиться, не вернув значения (None), а нативный тип {type} не может хранить None",
     "'{func}' мән қайтармай аяқталуы мүмкін (None), ал нативті {type} типі None сақтай алмайды"),
    ("'{func}' cannot be compiled",
     "'{func}' не удаётся скомпилировать",
     "'{func}' компиляциялау мүмкін емес"),
    ("'{func}' is used with more than {count:int} type combinations",
     "'{func}' используется со слишком многими комбинациями типов (больше {count})",
     "'{func}' тым көп тип комбинацияларымен қолданылады (шегі: {count})"),
    ("'{name}(...)' with a wrong number of fields",
     "'{name}(...)' с неверным числом полей",
     "өрістер саны қате '{name}(...)'"),
    ("'{name}.{variant}' without its values",
     "'{name}.{variant}' без значений",
     "мәндері жоқ '{name}.{variant}'"),
    ("Result.error() of a non-reference error type",
     "Result.error() для типа ошибки, который не является ссылочным",
     "сілтемелік емес қате типі үшін Result.error()"),
    ("`?` on a Result whose error type differs from the function's",
     "`?` для Result, тип ошибки которого отличается от типа ошибки функции",
     "қате типі функцияның қате типінен өзгеше Result үшін `?`"),
    ("`?` on an Option inside a function returning Result",
     "`?` для Option внутри функции, возвращающей Result",
     "Result қайтаратын функция ішінде Option үшін `?`"),
    ("`handle` (effect handlers run on the VM)",
     "`handle` (обработчики эффектов работают только на VM)",
     "`handle` (әсер өңдеушілері тек VM-де орындалады)"),
    ("a call whose target is not known statically",
     "вызов, цель которого статически неизвестна",
     "нысанасы статикалық түрде белгісіз шақыру"),
    ("a closure capturing '{name}', which changes later (the VMs share captured variables)",
     "замыкание захватывает '{name}', а эта переменная потом меняется (в VM захваченные переменные общие)",
     "тұйықтама '{name}' айнымалысын алады, ал ол кейін өзгереді (VM-дерде алынған айнымалылар ортақ)"),
    ("a closure that uses the lambda '{name}'",
     "замыкание, использующее лямбду '{name}'",
     "'{name}' лямбдасын қолданатын тұйықтама"),
    ("a higher-order method with this function argument",
     "метод высшего порядка с таким аргументом-функцией",
     "мұндай функция-аргументі бар жоғары ретті әдіс"),
    ("a range with non-numeric bounds",
     "диапазон с нечисловыми границами",
     "шекаралары сан емес ауқым"),
    ("a string containing a NUL byte",
     "строка с нулевым байтом (NUL)",
     "NUL байты бар жол"),
    ("a string index that is not an integer",
     "индекс строки, не являющийся целым числом",
     "бүтін сан емес жол индексі"),
    ("an array index that is not an integer",
     "индекс массива, не являющийся целым числом",
     "бүтін сан емес массив индексі"),
    ("a value of type {src:type} is used where {dst:type} is expected",
     "значение типа {src} используется там, где ожидается {dst}",
     "{dst} күтілетін жерде {src} типті мән қолданылған"),
    ("a {type:type} value in a place whose type was never determined",
     "значение типа {type} в месте, тип которого так и не был определён",
     "типі анықталмаған жердегі {type} типті мән"),
    ("an array literal of type {type:type}",
     "литерал массива типа {type}",
     "{type} типті массив литералы"),
    ("an expression the bytecode compiler folds with a different type",
     "выражение, которое компилятор байт-кода заранее вычисляет с другим типом",
     "байт-код компиляторы басқа типпен алдын ала есептейтін өрнек"),
    ("an index or count of type {type:type}",
     "индекс или количество типа {type}",
     "{type} типті индекс немесе сан"),
    ("assigning to a variable that holds a lambda",
     "присваивание переменной, в которой хранится лямбда",
     "лямбда сақталған айнымалыға мән меншіктеу"),
    ("assignment into a value of type {type:type}",
     "присваивание внутрь значения типа {type}",
     "{type} типті мәннің ішіне мән меншіктеу"),
    ("assignment to unknown name '{name}'",
     "присваивание неизвестному имени '{name}'",
     "белгісіз '{name}' атауына мән меншіктеу"),
    ("bitwise {op} on {left:type} and {right:type}",
     "побитовая операция {op} для {left} и {right}",
     "{left} және {right} үшін {op} биттік операциясы"),
    ("call of unknown function '{name}'",
     "вызов неизвестной функции '{name}'",
     "белгісіз '{name}' функциясын шақыру"),
    ("calling a value that is not a known function",
     "вызов значения, которое не является известной функцией",
     "белгілі функция емес мәнді шақыру"),
    ("calling the lambda '{name}' with {count:int} argument(s)",
     "вызов лямбды '{name}' с числом аргументов {count}",
     "'{name}' лямбдасын {count} аргументпен шақыру"),
    ("comparing values of type {type:type}",
     "сравнение значений типа {type}",
     "{type} типті мәндерді салыстыру"),
    ("comparing {left:type} with {right:type}",
     "сравнение {left} с {right}",
     "{left} және {right} типтерін салыстыру"),
    ("conflicting types ({types:text})",
     "конфликтующие типы ({types})",
     "қайшы типтер ({types})"),
    ("converting {src:type} to {dst:type}",
     "преобразование {src} в {dst}",
     "{src} типін {dst} типіне түрлендіру"),
    ("field '{field}' of '{struct}' is not given and starts as None, which a native {type:type} cannot hold",
     "поле '{field}' в '{struct}' не задано и изначально равно None, а нативный тип {type} не может "
     "хранить None",
     "'{struct}' ішіндегі '{field}' өрісі берілмеген және бастапқыда None, ал нативті {type} типі None "
     "сақтай алмайды"),
    ("field '{field}' of a value of type {type:type}",
     "поле '{field}' у значения типа {type}",
     "{type} типті мәннің '{field}' өрісі"),
    ("field assignment on a value of type {type:type}",
     "присваивание полю значения типа {type}",
     "{type} типті мәннің өрісіне мән меншіктеу"),
    ("format() with a non-string spec",
     "format() со спецификацией, не являющейся строкой",
     "спецификациясы жол емес format()"),
    ("global '{name}' is read before its value is set (the VMs see None there, which a native "
     "{type:type} cannot hold)",
     "глобальная переменная '{name}' читается до того, как ей задано значение (VM видят там None, а "
     "нативный тип {type} не может хранить None)",
     "'{name}' ғаламдық айнымалысы мәні берілмей тұрып оқылады (VM-дер онда None көреді, ал нативті "
     "{type} типі None сақтай алмайды)"),
    ("global '{name}' starts as None, which a native {type:type} cannot hold",
     "глобальная переменная '{name}' изначально равна None, а нативный тип {type} не может хранить None",
     "'{name}' ғаламдық айнымалысы бастапқыда None, ал нативті {type} типі None сақтай алмайды"),
    ("variable '{name}' starts as None, which a native {type:type} cannot hold",
     "переменная '{name}' изначально равна None, а нативный тип {type} не может хранить None",
     "'{name}' айнымалысы бастапқыда None, ал нативті {type} типі None сақтай алмайды"),
    ("indexing a value of type {type:type}",
     "индексирование значения типа {type}",
     "{type} типті мәнді индекстеу"),
    ("iterating over a value of type {type:type}",
     "перебор значения типа {type}",
     "{type} типті мән бойынша итерация"),
    ("map keys of type {type:type}",
     "ключи словаря типа {type}",
     "{type} типті сөздік кілттері"),
    ("method '{method}' of struct '{struct}'",
     "метод '{method}' структуры '{struct}'",
     "'{struct}' құрылымының '{method}' әдісі"),
    ("method '{method}' on a value of type {type:type}",
     "метод '{method}' у значения типа {type}",
     "{type} типті мәннің '{method}' әдісі"),
    ("missing argument '{param}' for '{func}'",
     "для '{func}' не передан аргумент '{param}'",
     "'{func}' үшін '{param}' аргументі берілмеген"),
    ("negating a value of type {type:type}",
     "смена знака у значения типа {type}",
     "{type} типті мәннің таңбасын өзгерту"),
    ("no instance of '{func}' for these argument types",
     "нет экземпляра '{func}' для таких типов аргументов",
     "мұндай аргумент типтері үшін '{func}' данасы жоқ"),
    ("no instance of '{func}' for these types",
     "нет экземпляра '{func}' для таких типов",
     "мұндай типтер үшін '{func}' данасы жоқ"),
    ("operator {op}",
     "оператор {op}",
     "{op} операторы"),
    ("ord() of a non-string",
     "ord() от значения, не являющегося строкой",
     "жол емес мән үшін ord()"),
    ("str_bytes() of a non-string",
     "str_bytes() от значения, не являющегося строкой",
     "жол емес мән үшін str_bytes()"),
    ("pow() of non-numbers",
     "pow() от нечисловых значений",
     "сан емес мәндер үшін pow()"),
    ("sum() of non-numbers",
     "sum() от нечисловых значений",
     "сан емес мәндер үшін sum()"),
    ("printing a value of type {type:type}",
     "вывод значения типа {type}",
     "{type} типті мәнді шығару"),
    ("random() with non-integer bounds",
     "random() с нецелыми границами",
     "шекаралары бүтін емес random()"),
    ("range() with non-integer bounds",
     "range() с нецелыми границами",
     "шекаралары бүтін емес range()"),
    ("reduce() without an initial value",
     "reduce() без начального значения",
     "бастапқы мәнсіз reduce()"),
    ("str.{method} with a non-string argument",
     "str.{method} с аргументом, не являющимся строкой",
     "аргументі жол емес str.{method}"),
    ("the default value of '{param}' (only literals are evaluated by the caller natively)",
     "значение по умолчанию параметра '{param}' (в нативном коде вызывающая сторона вычисляет только литералы)",
     "'{param}' параметрінің әдепкі мәні (нативті кодта шақырушы тек литералдарды есептейді)"),
    ("the function '{name}' takes {count:int} argument(s)",
     "функция '{name}' принимает аргументов: {count}",
     "'{name}' функциясы {count} аргумент қабылдайды"),
    ("the function '{name}' used as a value",
     "функция '{name}', использованная как значение",
     "мән ретінде қолданылған '{name}' функциясы"),
    ("the lambda '{name}'",
     "лямбда '{name}'",
     "'{name}' лямбдасы"),
    ("the lambda '{name}' calls another lambda",
     "лямбда '{name}' вызывает другую лямбду",
     "'{name}' лямбдасы басқа лямбданы шақырады"),
    ("the lambda '{name}' is called where its captured variable '{var}' is not visible",
     "лямбда '{name}' вызывается там, где не видна захваченная ею переменная '{var}'",
     "'{name}' лямбдасы өзі алған '{var}' айнымалысы көрінбейтін жерде шақырылады"),
    ("the lambda '{name}' is used where its captured variable '{var}' is not visible",
     "лямбда '{name}' используется там, где не видна захваченная ею переменная '{var}'",
     "'{name}' лямбдасы өзі алған '{var}' айнымалысы көрінбейтін жерде қолданылады"),
    ("this assignment target",
     "такая цель присваивания",
     "мұндай меншіктеу нысанасы"),
    ("this lambda",
     "такая лямбда",
     "мұндай лямбда"),
    ("too many arguments for '{func}'",
     "слишком много аргументов для '{func}'",
     "'{func}' үшін аргументтер тым көп"),
    ("unknown field '{field}'",
     "неизвестное поле '{field}'",
     "белгісіз өріс: '{field}'"),
    ("{kind} expressions",
     "выражения {kind}",
     "{kind} өрнектері"),
    ("{name}() (available on the TIL VM)",
     "{name}() (доступна на TIL VM)",
     "{name}() (TIL VM-де қолжетімді)"),
    ("{name}() of mixed types",
     "{name}() от значений разных типов",
     "әртүрлі типті мәндер үшін {name}()"),
    ("{name}() of [{type:type}]",
     "{name}() от [{type}]",
     "[{type}] үшін {name}()"),
    ("{name}() of {type:type}",
     "{name}() от {type}",
     "{type} үшін {name}()"),
    ("{name}() without a function",
     "{name}() без функции",
     "функциясыз {name}()"),
    ("{name}(...) without a known Option/Result type",
     "{name}(...) без известного типа Option/Result",
     "Option/Result типі белгісіз {name}(...)"),
    ("~ on a value of type {type:type}",
     "~ для значения типа {type}",
     "{type} типті мән үшін ~"),
    ("a Result error has no single static type (it is dynamically typed)",
     "у ошибки Result нет единого статического типа (тип определяется во время выполнения)",
     "Result қатесінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("a Result error has type {type:type}",
     "ошибка Result имеет тип {type}",
     "Result қатесінің типі — {type}"),
    ("a Result value has no single static type (it is dynamically typed)",
     "у значения Result нет единого статического типа (тип определяется во время выполнения)",
     "Result мәнінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("a Result value has type {type:type}",
     "значение Result имеет тип {type}",
     "Result мәнінің типі — {type}"),
    ("a compared value has no single static type (it is dynamically typed)",
     "у сравниваемого значения нет единого статического типа (тип определяется во время выполнения)",
     "салыстырылатын мәннің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("a compared value has type {type:type}",
     "сравниваемое значение имеет тип {type}",
     "салыстырылатын мәннің типі — {type}"),
    ("a condition has no single static type (it is dynamically typed)",
     "у условия нет единого статического типа (тип определяется во время выполнения)",
     "шарттың бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("a condition has type {type:type}",
     "условие имеет тип {type}",
     "шарттың типі — {type}"),
    ("a map key has no single static type (it is dynamically typed)",
     "у ключа словаря нет единого статического типа (тип определяется во время выполнения)",
     "сөздік кілтінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("a map key has type {type:type}",
     "ключ словаря имеет тип {type}",
     "сөздік кілтінің типі — {type}"),
    ("a map key of type {type:type} inside a container",
     "ключ словаря типа {type} внутри контейнера",
     "контейнер ішіндегі {type} типті сөздік кілті"),
    ("a map value has no single static type (it is dynamically typed)",
     "у значения словаря нет единого статического типа (тип определяется во время выполнения)",
     "сөздік мәнінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("a map value has type {type:type}",
     "значение словаря имеет тип {type}",
     "сөздік мәнінің типі — {type}"),
    ("a map value of type {type:type} inside a container",
     "значение словаря типа {type} внутри контейнера",
     "контейнер ішіндегі {type} типті сөздік мәні"),
    ("a payload of {name}.{variant} has no single static type (it is dynamically typed)",
     "у данных варианта {name}.{variant} нет единого статического типа (тип определяется во время выполнения)",
     "{name}.{variant} нұсқасы деректерінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("a payload of {name}.{variant} has type {type:type}",
     "данные варианта {name}.{variant} имеет тип {type}",
     "{name}.{variant} нұсқасы деректерінің типі — {type}"),
    ("a payload of {name}.{variant} of type {type:type} inside a container",
     "данные варианта {name}.{variant} типа {type} внутри контейнера",
     "контейнер ішіндегі {type} типті {name}.{variant} нұсқасының деректері"),
    ("a printed value has no single static type (it is dynamically typed)",
     "у выводимого значения нет единого статического типа (тип определяется во время выполнения)",
     "шығарылатын мәннің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("a printed value has type {type:type}",
     "выводимое значение имеет тип {type}",
     "шығарылатын мәннің типі — {type}"),
    ("a return value has no single static type (it is dynamically typed)",
     "у возвращаемого значения нет единого статического типа (тип определяется во время выполнения)",
     "қайтарылатын мәннің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("a return value has type {type:type}",
     "возвращаемое значение имеет тип {type}",
     "қайтарылатын мәннің типі — {type}"),
    ("a value has no single static type (it is dynamically typed)",
     "у значения нет единого статического типа (тип определяется во время выполнения)",
     "мәннің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("a value has type {type:type}",
     "значение имеет тип {type}",
     "мәннің типі — {type}"),
    ("an Option payload has no single static type (it is dynamically typed)",
     "у содержимого Option нет единого статического типа (тип определяется во время выполнения)",
     "Option ішіндегі мәннің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("an Option payload has type {type:type}",
     "содержимое Option имеет тип {type}",
     "Option ішіндегі мәннің типі — {type}"),
    ("an element has no single static type (it is dynamically typed)",
     "у элемента нет единого статического типа (тип определяется во время выполнения)",
     "элементтің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("an element has type {type:type}",
     "элемент имеет тип {type}",
     "элементтің типі — {type}"),
    ("an element of type {type:type} inside a container",
     "элемент типа {type} внутри контейнера",
     "контейнер ішіндегі {type} типті элемент"),
    ("captured variable '{name}' has no single static type (it is dynamically typed)",
     "у захваченной переменной '{name}' нет единого статического типа (тип определяется во время выполнения)",
     "сыртқы аймақтан алынған '{name}' айнымалысының бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("captured variable '{name}' has type {type:type}",
     "захваченная переменная '{name}' имеет тип {type}",
     "сыртқы аймақтан алынған '{name}' айнымалысының типі — {type}"),
    ("captured variable '{name}' of type {type:type} inside a container",
     "захваченная переменная '{name}' типа {type} внутри контейнера",
     "контейнер ішіндегі {type} типті сыртқы аймақтан алынған '{name}' айнымалысы"),
    ("captured variable {name} has no single static type (it is dynamically typed)",
     "у захваченной переменной {name} нет единого статического типа (тип определяется во время выполнения)",
     "сыртқы аймақтан алынған {name} айнымалысының бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("captured variable {name} has type {type:type}",
     "захваченная переменная {name} имеет тип {type}",
     "сыртқы аймақтан алынған {name} айнымалысының типі — {type}"),
    ("field '{name}' has no single static type (it is dynamically typed)",
     "у поля '{name}' нет единого статического типа (тип определяется во время выполнения)",
     "'{name}' өрісінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("field '{name}' has type {type:type}",
     "поле '{name}' имеет тип {type}",
     "'{name}' өрісінің типі — {type}"),
    ("global '{name}' has no single static type (it is dynamically typed)",
     "у глобальной переменной '{name}' нет единого статического типа (тип определяется во время выполнения)",
     "'{name}' ғаламдық айнымалысының бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("global '{name}' has type {type:type}",
     "глобальная переменная '{name}' имеет тип {type}",
     "'{name}' ғаламдық айнымалысының типі — {type}"),
    ("lambda parameter {name} has no single static type (it is dynamically typed)",
     "у параметра лямбды {name} нет единого статического типа (тип определяется во время выполнения)",
     "лямбданың {name} параметрінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("lambda parameter {name} has type {type:type}",
     "параметр лямбды {name} имеет тип {type}",
     "лямбданың {name} параметрінің типі — {type}"),
    ("parameter '{name}' has no single static type (it is dynamically typed)",
     "у параметра '{name}' нет единого статического типа (тип определяется во время выполнения)",
     "'{name}' параметрінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("parameter '{name}' has type {type:type}",
     "параметр '{name}' имеет тип {type}",
     "'{name}' параметрінің типі — {type}"),
    ("the value matched has no single static type (it is dynamically typed)",
     "у значения, сопоставляемого с образцами, нет единого статического типа (тип определяется во время выполнения)",
     "үлгілермен салыстырылатын мәннің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("the value matched has type {type:type}",
     "значение, сопоставляемое с образцами имеет тип {type}",
     "үлгілермен салыстырылатын мәннің типі — {type}"),
    ("the value of a block has no single static type (it is dynamically typed)",
     "у значения блока нет единого статического типа (тип определяется во время выполнения)",
     "блок мәнінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("the value of a block has type {type:type}",
     "значение блока имеет тип {type}",
     "блок мәнінің типі — {type}"),
    ("the value of a match has no single static type (it is dynamically typed)",
     "у значения match нет единого статического типа (тип определяется во время выполнения)",
     "match мәнінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("the value of a match has type {type:type}",
     "значение match имеет тип {type}",
     "match мәнінің типі — {type}"),
    ("the value of an if-expression has no single static type (it is dynamically typed)",
     "у значения if-выражения нет единого статического типа (тип определяется во время выполнения)",
     "if өрнегі мәнінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("the value of an if-expression has type {type:type}",
     "значение if-выражения имеет тип {type}",
     "if өрнегі мәнінің типі — {type}"),
    ("the value of an intent block has no single static type (it is dynamically typed)",
     "у значения блока intent нет единого статического типа (тип определяется во время выполнения)",
     "intent блогы мәнінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("the value of an intent block has type {type:type}",
     "значение блока intent имеет тип {type}",
     "intent блогы мәнінің типі — {type}"),
    ("the value of retry/timeout has no single static type (it is dynamically typed)",
     "у значения retry/timeout нет единого статического типа (тип определяется во время выполнения)",
     "retry/timeout мәнінің бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("the value of retry/timeout has type {type:type}",
     "значение retry/timeout имеет тип {type}",
     "retry/timeout мәнінің типі — {type}"),
    ("variable '{name}' has no single static type (it is dynamically typed)",
     "у переменной '{name}' нет единого статического типа (тип определяется во время выполнения)",
     "'{name}' айнымалысының бірыңғай статикалық типі жоқ (типі орындау кезінде анықталады)"),
    ("variable '{name}' has type {type:type}",
     "переменная '{name}' имеет тип {type}",
     "'{name}' айнымалысының типі — {type}"),
]

# Messages the VMs can report at run time (also looked up for compiler
# diagnostics). The uncaught-error report of a program is matched against
# these only: a program's own panic text never meets a compiler template.
RUNTIME_MESSAGES: List[Tuple[str, str, str]] = [
    # ── runtime errors (reference VM and C VM) ─────────────────────────────
    ("division by zero",
     "деление на ноль",
     "нөлге бөлу"),
    ("{what:builtin} expects a number, got {type:type}",
     "{what} ожидает число, получено {type}",
     "{what} сан күтеді, бірақ {type} берілді"),
    ("unsupported operand types for {op}: {left:type} and {right:type}",
     "неподдерживаемые типы операндов для {op}: {left} и {right}",
     "{op} операциясы үшін операнд типтеріне қолдау жоқ: {left} және {right}"),
    ("cannot compare {left:type} and {right:type}",
     "нельзя сравнить {left} и {right}",
     "{left} және {right} мәндерін салыстыруға болмайды"),
    ("bitwise operators need integers",
     "побитовым операторам нужны целые числа",
     "биттік операторларға бүтін сандар керек"),
    ("cannot convert nan/inf to int",
     "нельзя преобразовать nan/inf в int",
     "nan/inf мәнін int типіне түрлендіруге болмайды"),
    ("invalid integer: \"{text}\"",
     "неверное целое число: \"{text}\"",
     "жарамсыз бүтін сан: \"{text}\""),
    ("invalid number: \"{text}\"",
     "неверное число: \"{text}\"",
     "жарамсыз сан: \"{text}\""),
    ("cannot convert value to int",
     "нельзя преобразовать значение в int",
     "мәнді int типіне түрлендіруге болмайды"),
    ("cannot convert value to float",
     "нельзя преобразовать значение в float",
     "мәнді float типіне түрлендіруге болмайды"),
    ("cannot convert {type:type} to int",
     "нельзя преобразовать {type} в int",
     "{type} типін int типіне түрлендіруге болмайды"),
    ("cannot convert {type:type} to float",
     "нельзя преобразовать {type} в float",
     "{type} типін float типіне түрлендіруге болмайды"),
    ("this TIL VM was built without floating point",
     "эта TIL VM собрана без поддержки чисел с плавающей точкой",
     "бұл TIL VM жылжымалы нүктелі сандарсыз құрастырылған"),
    ("index must be an integer",
     "индекс должен быть целым числом",
     "индекс бүтін сан болуы керек"),
    ("index out of bounds: index {index:int}, length {length:int}",
     "индекс вне границ: индекс {index}, длина {length}",
     "индекс шектен тыс: индекс {index}, ұзындық {length}"),
    ("key not found: {key:text}",
     "ключ не найден: {key}",
     "кілт табылмады: {key}"),
    ("{type:type} cannot be indexed",
     "{type} не поддерживает индексацию",
     "{type} типін индекстеуге болмайды"),
    ("{type:type} cannot be a map key",
     "{type} не может быть ключом словаря",
     "{type} сөздік кілті бола алмайды"),
    ("strings are immutable",
     "строки неизменяемы",
     "жолдарды өзгертуге болмайды"),
    ("{type:type} does not support item assignment",
     "{type} не поддерживает присваивание элементов",
     "{type} элементтеріне мән меншіктеуге болмайды"),
    ("no field '{field}' on {type:type}",
     "у {type} нет поля '{field}'",
     "{type} ішінде '{field}' өрісі жоқ"),
    ("{name}() is missing arguments",
     "{name}(): не хватает аргументов",
     "{name}() үшін аргументтер жетіспейді"),
    ("value is not iterable",
     "значение не является итерируемым",
     "бұл мән бойынша итерация жасауға болмайды"),
    ("{type:type} is not iterable",
     "{type} не является итерируемым",
     "{type} мәні бойынша итерация жасауға болмайды"),
    ("len() of {type:type}",
     "len() неприменима к {type}",
     "len() функциясын {type} типіне қолдануға болмайды"),
    ("{name}() of empty sequence",
     "{name}() от пустой последовательности",
     "{name}() бос тізбекке қолданылды"),
    ("chr() argument out of range",
     "аргумент chr() вне допустимого диапазона",
     "chr() аргументі рұқсат етілген ауқымнан тыс"),
    ("ord() expects a non-empty string",
     "ord() ожидает непустую строку",
     "ord() бос емес жолды күтеді"),
    ("range bounds must be integers",
     "границы range должны быть целыми числами",
     "range шекаралары бүтін сан болуы керек"),
    ("range too large",
     "диапазон слишком велик",
     "ауқым тым үлкен"),
    ("random(lo, hi) needs lo <= hi",
     "random(lo, hi) требует lo <= hi",
     "random(lo, hi) үшін lo <= hi болуы керек"),
    ("cannot read file: {path:text}",
     "не удаётся прочитать файл: {path}",
     "файлды оқу мүмкін емес: {path}"),
    ("cannot write file: {path:text}",
     "не удаётся записать файл: {path}",
     "файлға жазу мүмкін емес: {path}"),
    ("file too large: {path:text}",
     "файл слишком большой: {path}",
     "файл тым үлкен: {path}"),
    ("{name}() expects a path string, got {type:type}",
     "{name}() ожидает путь к файлу (строку), получено {type}",
     "{name}() файл жолын (str) күтеді, бірақ {type} берілді"),
    ("{name}(): invalid path",
     "{name}(): недопустимый путь",
     "{name}(): файл жолы жарамсыз"),
    ("bytes_to_str() expects an array of ints 0..255",
     "bytes_to_str() ожидает массив целых чисел 0..255",
     "bytes_to_str() 0..255 аралығындағы бүтін сандар массивін күтеді"),
    ("str_bytes() expects a string, got {type:type}",
     "str_bytes() ожидает строку, получено {type}",
     "str_bytes() жол күтеді, бірақ {type} берілді"),
    ("{name}() is not available in this VM build",
     "{name}() недоступна в этой сборке VM",
     "{name}() бұл VM құрастырмасында қолжетімсіз"),
    ("{name}() is not available on this device",
     "{name}() недоступна на этом устройстве",
     "{name}() бұл құрылғыда қолжетімсіз"),
    ("sin/cos/tan/log/exp need a build with <math.h>",
     "sin/cos/tan/log/exp требуют сборки с <math.h>",
     "sin/cos/tan/log/exp үшін <math.h> бар құрастырма керек"),
    ("pow expects numbers",
     "pow ожидает числа",
     "pow сандарды күтеді"),
    ("sum() needs numbers",
     "sum() требует числа",
     "sum() үшін сандар керек"),
    ("{name}() needs an integer",
     "{name}() требует целое число",
     "{name}() үшін бүтін сан керек"),
    ("str.{method} expects a string argument",
     "str.{method} ожидает строковый аргумент",
     "str.{method} жол түріндегі аргументті күтеді"),
    ("replace() needs two strings",
     "replace() требует две строки",
     "replace() үшін екі жол керек"),
    ("split() needs a string separator",
     "split() требует строку-разделитель",
     "split() үшін бөлгіш жол керек"),
    ("string too long",
     "строка слишком длинная",
     "жол тым ұзын"),
    ("array too large",
     "массив слишком большой",
     "массив тым үлкен"),
    ("{name}() needs a value",
     "{name}() требует значение",
     "{name}() үшін мән керек"),
    ("{name}() needs an index and a value",
     "{name}() требует индекс и значение",
     "{name}() үшін индекс пен мән керек"),
    ("{name}() needs a key and a value",
     "{name}() требует ключ и значение",
     "{name}() үшін кілт пен мән керек"),
    ("get_or() needs a key and a default",
     "get_or() требует ключ и значение по умолчанию",
     "get_or() үшін кілт пен әдепкі мән керек"),
    ("unwrap_or() needs a default",
     "unwrap_or() требует значение по умолчанию",
     "unwrap_or() үшін әдепкі мән керек"),
    ("format() expects a spec string, got {type:type}",
     "format() ожидает строку формата, получено {type}",
     "format() формат жолын күтеді, бірақ {type} берілді"),
    ("{name:builtin}(): invalid arguments",
     "{name}(): недопустимые аргументы",
     "{name}(): аргументтер жарамсыз"),
    ("method needs a value",
     "методу нужно значение",
     "әдіске мән керек"),
    ("map method needs a key",
     "методу словаря нужен ключ",
     "сөздік әдісіне кілт керек"),
    ("pop() from empty array",
     "pop() из пустого массива",
     "бос массивке pop() қолданылды"),
    ("insert index out of bounds: {index:int}",
     "индекс вставки вне границ: {index}",
     "кірістіру индексі шектен тыс: {index}"),
    ("remove index out of bounds: {index:int}",
     "индекс удаления вне границ: {index}",
     "жою индексі шектен тыс: {index}"),
    ("sort() needs an array of numbers or of strings",
     "sort() требует массив чисел или строк",
     "sort() үшін сандар немесе жолдар массиві керек"),
    ("unwrap() called on None",
     "unwrap() вызван для None",
     "unwrap() None мәніне шақырылды"),
    ("unwrap() called on Err: {error:text}",
     "unwrap() вызван для Err: {error}",
     "unwrap() Err мәніне шақырылды: {error}"),
    ("expect() failed",
     "expect() завершился неудачей",
     "expect() сәтсіз аяқталды"),
    ("no method '{method}' on {type:type}",
     "у {type} нет метода '{method}'",
     "{type} типінде '{method}' әдісі жоқ"),
    ("field '{field}' is not callable",
     "поле '{field}' нельзя вызвать",
     "'{field}' өрісін шақыруға болмайды"),
    ("{type:type} is not callable",
     "значение типа {type} нельзя вызвать",
     "{type} типті мәнді шақыруға болмайды"),
    ("{func}() takes {count:int} argument(s), got {got:int}",
     "{func}() принимает аргументов: {count}, передано: {got}",
     "{func}() {count} аргумент қабылдайды, бірақ {got} берілді"),
    ("stack overflow (recursion too deep)",
     "переполнение стека (слишком глубокая рекурсия)",
     "стек толып кетті (рекурсия тым терең)"),
    ("stack overflow",
     "переполнение стека",
     "стек толып кетті"),
    ("native recursion limit",
     "превышен внутренний предел рекурсии",
     "ішкі рекурсия шегінен асып кетті"),
    ("out of memory",
     "недостаточно памяти",
     "жад жетіспейді"),
    ("cannot negate a non-number",
     "нельзя применить унарный минус к нечисловому значению",
     "сан емес мәнге унарлық минус қолдануға болмайды"),
    ("cannot negate {type:type}",
     "нельзя применить унарный минус к {type}",
     "{type} типіне унарлық минус қолдануға болмайды"),
    ("~ needs an integer",
     "оператору ~ нужно целое число",
     "~ операторына бүтін сан керек"),
    ("pattern slot out of range",
     "слот образца вне диапазона",
     "үлгі ұяшығы ауқымнан тыс"),
    ("? applied to None",
     "оператор ? применён к None",
     "? операторы None мәніне қолданылды"),
    ("? applied to Err: {error:text}",
     "оператор ? применён к Err: {error}",
     "? операторы Err мәніне қолданылды: {error}"),
    ("step limit exceeded ({limit:int})",
     "превышен лимит шагов ({limit})",
     "қадам шегінен асып кетті ({limit})"),
    ("step limit exceeded",
     "превышен лимит шагов",
     "қадам шегінен асып кетті"),
    ("timeout",
     "истекло время ожидания",
     "күту уақыты бітті"),
    ("too many nested retry/timeout blocks",
     "слишком много вложенных блоков retry/timeout",
     "кірістірілген retry/timeout блоктары тым көп"),
    ("bad opcode 0x{op} at {pc:int}",
     "неверный код операции 0x{op} по адресу {pc}",
     "{pc} адресінде қате операция коды: 0x{op}"),
    ("bad opcode",
     "неверный код операции",
     "қате операция коды"),
    ("bad opcode (comparison 0x{op})",                  # til31/vmcore: compare-and-branch
     "неверный код операции (сравнение 0x{op})",
     "қате операция коды (салыстыру 0x{op})"),

    # ── effects with resumption (TIL 3.1; the til31/effects branch) ────────
    ("unhandled effect '{effect}'",
     "необработанный эффект '{effect}'",
     "өңделмеген әсер: '{effect}'"),
    ("effect '{effect}' expects {count:int} argument(s), got {got:int}",
     "эффект '{effect}' ожидает аргументов: {count}, передано: {got}",
     "'{effect}' әсері {count} аргумент күтеді, бірақ {got} берілді"),
    ("effect '{effect}' has no callable clause",
     "у эффекта '{effect}' нет вызываемой ветви",
     "'{effect}' әсерінің шақырылатын тармағы жоқ"),
    ("effect clause returned outside its handler",
     "ветвь эффекта завершилась вне своего обработчика",
     "әсер тармағы өз өңдеушісінен тыс аяқталды"),
    ("too many nested handlers",
     "слишком много вложенных обработчиков",
     "кірістірілген өңдеушілер тым көп"),
    ("END_HANDLE without an active handle",
     "END_HANDLE без активного handle",
     "белсенді handle жоқ кезде END_HANDLE орындалды"),
    ("resume outside an effect clause",
     "resume вне ветви эффекта",
     "resume әсер тармағынан тыс"),
    ("effect handlers are not available in this VM build",
     "обработчики эффектов недоступны в этой сборке VM",
     "әсер өңдеушілері бұл VM құрастырмасында қолжетімсіз"),

    # ── energy saving (TIL 3.1, docs/TIL_ENERGY.md) ────────────────────────
    ("energy_mode() needs a mode 0 (performance), 1 (balanced) or 2 (low)",
     "energy_mode() ожидает режим 0 (performance), 1 (balanced) или 2 (low)",
     "energy_mode() 0 (performance), 1 (balanced) немесе 2 (low) режимін күтеді"),

    # ── graphics and keys (TIL 3.1; written against the til31/gfx branch) ───
    ("screen not initialized (call screen_init)",
     "экран не инициализирован (вызовите screen_init)",
     "экран іске қосылмаған (screen_init шақырыңыз)"),
    ("key code out of range",
     "код клавиши вне допустимого диапазона",
     "перне коды рұқсат етілген ауқымнан тыс"),
    ("{name}() argument {index:int} must be an int, got {type:type}",
     "{name}(): аргумент {index} должен иметь тип int, получено {type}",
     "{name}(): {index}-аргумент int типті болуы керек, бірақ {type} берілді"),
    ("{name}() argument {index:int} must be 1..{most:int}, got {value:int}",
     "{name}(): аргумент {index} должен быть в диапазоне 1..{most}, получено {value}",
     "{name}(): {index}-аргумент 1..{most} аралығында болуы керек, бірақ {value} берілді"),
    ("{name}() argument {index:int} must be -32768..32767, got {value:int}",
     "{name}(): аргумент {index} должен быть в диапазоне -32768..32767, получено {value}",
     "{name}(): {index}-аргумент -32768..32767 аралығында болуы керек, бірақ {value} берілді"),
    ("{name}() argument {index:int} must be a color 0..0xFFFFFF, got {value:int}",
     "{name}(): аргумент {index} должен быть цветом 0..0xFFFFFF, получено {value}",
     "{name}(): {index}-аргумент 0..0xFFFFFF аралығындағы түс болуы керек, бірақ {value} берілді"),

    # ── runtime errors compiled into programs (asserts, contracts, intents) ─
    ("Assertion failed",
     "проверка не пройдена",
     "тексеру сәтсіз аяқталды"),
    ("Assertion failed: {message:text}",
     "проверка не пройдена: {message}",
     "тексеру сәтсіз аяқталды: {message}"),
    ("Invariant violated",
     "инвариант нарушен",
     "инвариант бұзылды"),
    ("Invariant violated: {message:text}",
     "инвариант нарушен: {message}",
     "инвариант бұзылды: {message}"),
    ("contract violation: {func} requires {cond:text}",
     "нарушение контракта: {func} требует {cond}",
     "келісімшарт бұзылды: {func} талабы орындалмады: {cond}"),
    ("contract violation: {func} ensures {cond:text}",
     "нарушение контракта: {func} гарантирует {cond}",
     "келісімшарт бұзылды: {func} кепілдігі орындалмады: {cond}"),
    ("intent '{intent}' failed: {error:msg}",
     "намерение '{intent}' не выполнено: {error}",
     "'{intent}' ниеті орындалмады: {error}"),
]

# Messages of the command line (`til`, and the C VM's runner `tilvm`).
COMMAND_LINE_MESSAGES: List[Tuple[str, str, str]] = [
    # ── command line ───────────────────────────────────────────────────────
    ("No input file specified",
     "не указан входной файл",
     "кіріс файлы көрсетілмеген"),
    ("--max-steps needs a number >= 0, got '{value:text}'",
     "--max-steps требует число >= 0, получено '{value}'",
     "--max-steps үшін >= 0 сан керек, берілгені: '{value}'"),
    ("File not found: {path:text}",
     "файл не найден: {path}",
     "файл табылмады: {path}"),
    ("unknown option {option:text}",
     "неизвестный параметр {option}",
     "белгісіз параметр: {option}"),
    ("unknown message language '{lang}' (use: en, ru, kk)",
     "неизвестный язык сообщений '{lang}' (доступны: en, ru, kk)",
     "белгісіз хабарлама тілі: '{lang}' (мүмкін мәндер: en, ru, kk)"),
    ("unknown target '{target}' (native, vm, c-vm, c)",
     "неизвестная цель сборки '{target}' (native, vm, c-vm, c)",
     "белгісіз құрастыру мақсаты: '{target}' (native, vm, c-vm, c)"),
    ("Could not find module '{module}'",
     "не удалось найти модуль '{module}'",
     "'{module}' модулі табылмады"),
    ("No C compiler found (tried gcc, clang, cc)",
     "компилятор C не найден (проверены gcc, clang, cc)",
     "C компиляторы табылмады (gcc, clang, cc тексерілді)"),
    ("C compilation failed:",
     "ошибка компиляции C:",
     "C компиляциясы сәтсіз аяқталды:"),
    ("this program may use features only the TIL VM supports; "
     "try `til run --vm` or `til build --target vm`",
     "возможно, программа использует возможности, которые поддерживает только TIL VM; "
     "попробуйте `til run --vm` или `til build --target vm`",
     "бұл бағдарлама тек TIL VM қолдайтын мүмкіндіктерді пайдалануы мүмкін; "
     "`til run --vm` немесе `til build --target vm` командасын қолданып көріңіз"),
    ("C VM build failed:",
     "сборка C VM не удалась:",
     "C VM құрастыру сәтсіз аяқталды:"),
    ("cannot build the C VM (need vm/ sources and a C compiler)",
     "не удаётся собрать C VM (нужны исходники vm/ и компилятор C)",
     "C VM құрастыру мүмкін емес (vm/ бастапқы кодтары мен C компиляторы керек)"),
    ("bad --keys entry '{entry}' (use UP, DOWN, LEFT, RIGHT, ENTER, ESC, SPACE, TAB, BACKSPACE, "
     "COMMA, NONE or one character)",
     "неверный элемент --keys '{entry}' (используйте UP, DOWN, LEFT, RIGHT, ENTER, ESC, SPACE, TAB, "
     "BACKSPACE, COMMA, NONE или один символ)",
     "--keys ішіндегі '{entry}' элементі жарамсыз (UP, DOWN, LEFT, RIGHT, ENTER, ESC, SPACE, TAB, "
     "BACKSPACE, COMMA, NONE немесе бір таңба қолданыңыз)"),
    # the graphics options of `til run` and of the C VM's runner (til31/gfx):
    # printed as "til: ..." / "tilvm: ..." (the program name stays as it is)
    ("--screen: the program did not call screen_init (no image written)",
     "--screen: программа не вызвала screen_init (изображение не записано)",
     "--screen: бағдарлама screen_init шақырмады (сурет жазылмады)"),
    ("cannot write {path:text}",
     "не удаётся записать {path}",
     "жазу мүмкін емес: {path}"),
    ("not enough memory for a {width:int}x{height:int} screen",
     "недостаточно памяти для экрана {width}x{height}",
     "{width}x{height} экраны үшін жад жетіспейді"),
    ("TILScreen: max_w and max_h must be 1..{most:int}",
     "TILScreen: max_w и max_h должны быть в диапазоне 1..{most}",
     "TILScreen: max_w және max_h 1..{most} аралығында болуы керек"),
    # portable snapshots (.tilp, docs/TIL_SNAPSHOTS.md): `til run --pause-after N --save F`,
    # `--resume F` and the C VM's runner (tilvm --save-portable-after / --resume-portable);
    # printed as "til: ..." / "tilvm: ..."
    ("paused after {steps:int} steps; saved {path:text} ({size:int} bytes)",
     "программа приостановлена (шагов: {steps}); сохранено: {path} (байт: {size})",
     "бағдарлама кідіртілді (қадам саны: {steps}); сақталды: {path} (байт саны: {size})"),
    ("the program ended before the pause point (nothing saved)",
     "программа завершилась раньше точки паузы (ничего не сохранено)",
     "бағдарлама кідірту нүктесіне жетпей аяқталды (ештеңе сақталмады)"),
    ("cannot resume {path:text}: {reason:msg}",
     "не удаётся продолжить {path}: {reason}",
     "{path} файлынан жалғастыру мүмкін емес: {reason}"),
    ("cannot save {path:text}: {reason:msg}",
     "не удаётся сохранить {path}: {reason}",
     "{path} файлына сақтау мүмкін емес: {reason}"),
    ("cannot read {path:text}",
     "не удаётся прочитать {path}",
     "оқу мүмкін емес: {path}"),
    ("--pause-after N and --save FILE go together",
     "--pause-after N и --save FILE указываются вместе",
     "--pause-after N және --save FILE бірге көрсетіледі"),
    ("--pause-after needs a number of steps",
     "--pause-after: нужно число шагов",
     "--pause-after үшін қадам саны керек"),
    ("not a portable TIL snapshot (.tilp)",
     "это не переносимый снимок TIL (.tilp)",
     "бұл TIL-дің тасымалды күй көшірмесі (.tilp) емес"),
    ("unsupported snapshot version",
     "неподдерживаемая версия снимка",
     "күй көшірмесінің нұсқасына қолдау жоқ"),
    ("unsupported snapshot version ({detail:msg})",
     "неподдерживаемая версия снимка ({detail})",
     "күй көшірмесінің нұсқасына қолдау жоқ ({detail})"),
    ("the snapshot was saved by a different program",
     "снимок сохранён другой программой",
     "күй көшірмесін басқа бағдарлама сақтаған"),
    ("the snapshot is damaged or incomplete",
     "снимок повреждён или неполон",
     "күй көшірмесі бүлінген немесе толық емес"),
    ("the snapshot is damaged or incomplete ({detail:msg})",
     "снимок повреждён или неполон ({detail})",
     "күй көшірмесі бүлінген немесе толық емес ({detail})"),
    ("the program's state does not fit this VM",
     "состояние программы не помещается в эту VM",
     "бағдарлама күйі бұл VM-ге сыймайды"),
    ("the program's state does not fit this VM ({detail:msg})",
     "состояние программы не помещается в эту VM ({detail})",
     "бағдарлама күйі бұл VM-ге сыймайды ({detail})"),
    ("the snapshot holds an integer that does not fit this VM",
     "в снимке есть целое число, которое не помещается в эту VM",
     "күй көшірмесінде бұл VM-ге сыймайтын бүтін сан бар"),
    ("the snapshot holds floats, but this VM has no floats",
     "в снимке есть числа с плавающей точкой, а эта VM их не поддерживает",
     "күй көшірмесінде нақты сандар бар, бірақ бұл VM оларды қолдамайды"),
    ("the snapshot's screen is larger than this VM can show",
     "экран снимка больше, чем может показать эта VM",
     "күй көшірмесінің экраны бұл VM көрсете алатыннан үлкен"),
    ("the snapshot's screen is larger than this VM can show ({detail:msg})",
     "экран снимка больше, чем может показать эта VM ({detail})",
     "күй көшірмесінің экраны бұл VM көрсете алатыннан үлкен ({detail})"),
    ("portable snapshots are not available in this VM build",
     "переносимые снимки недоступны в этой сборке VM",
     "бұл VM құрастырмасында тасымалды күй көшірмелері қолжетімсіз"),
    ("the snapshot uses effect handlers, which this VM build leaves out",
     "снимок использует обработчики эффектов, которых нет в этой сборке VM",
     "күй көшірмесі әсер өңдеушілерін қолданады, ал бұл VM құрастырмасында олар жоқ"),
    ("the program has ended: there is nothing to save",
     "программа уже завершилась: сохранять нечего",
     "бағдарлама аяқталып қойған: сақтайтын ештеңе жоқ"),
    ("the program's state cannot be written as a snapshot ({detail:msg})",
     "состояние программы нельзя записать в снимок ({detail})",
     "бағдарлама күйін күй көшірмесіне жазу мүмкін емес ({detail})"),
    ("[TIL] No C compiler found - running on the TIL VM",
     "[TIL] компилятор C не найден — программа запускается на TIL VM",
     "[TIL] C компиляторы табылмады — бағдарлама TIL VM-де іске қосылады"),
    ("[TIL] native backend cannot compile this program; running it on the TIL VM",
     "[TIL] нативный бэкенд не может скомпилировать эту программу; она запускается на TIL VM",
     "[TIL] нативті бэкенд бұл бағдарламаны компиляциялай алмайды; ол TIL VM-де іске қосылады"),
    # "til pkg: ...", "til check --prove: ...": the program name stays (tilmsg's frame)
    ("not available in this installation ({module}.py is missing)",
     "недоступно в этой установке (нет файла {module}.py)",
     "бұл орнатылымда қолжетімсіз ({module}.py файлы жоқ)"),
    ("the prover (tilprove.py) is not installed",
     "модуль доказательства (tilprove.py) не установлен",
     "дәлелдеу модулі (tilprove.py) орнатылмаған"),
    # ── til debug (src/tildebug.py; docs/TIL_DEBUGGER.md) ──────────────────
    # Command names and LINE / NAME in usage texts stay as they are typed.
    ("TIL time-travel debugger (type help for the commands)",
     "Отладчик TIL с путешествием во времени (help — список команд)",
     "Уақытта саяхаттайтын TIL жөндеуіші (help — командалар тізімі)"),
    ("no program given",
     "не указана программа",
     "бағдарлама көрсетілмеген"),
    ("{option} needs a value",
     "{option}: нужно значение",
     "{option} үшін мән керек"),
    ("--snapshot-every needs a positive number",
     "--snapshot-every: нужно положительное число",
     "--snapshot-every үшін оң сан керек"),
    ("The program has no line table (built with --strip): it has no line events",
     "В программе нет таблицы строк (собрана с --strip): событий строк не будет",
     "Бағдарламада жолдар кестесі жоқ (--strip арқылы құрастырылған): жол оқиғалары болмайды"),
    ("Breakpoint {n:int}, {func} at line {line:int}",
     "Точка останова {n}, {func}, строка {line}",
     "Тоқтау нүктесі {n}, {func}, {line}-жол"),
    ("{func} at line {line:int}",
     "{func}, строка {line}",
     "{func}, {line}-жол"),
    ("{func} (no source line)",
     "{func} (без строки исходного кода)",
     "{func} (бастапқы код жолы жоқ)"),
    ("{func} returned {value:text}",
     "функция {func} вернула {value}",
     "{func} функциясы қайтарған мән: {value}"),
    ("Program exited with code {code:int}",
     "Программа завершилась с кодом {code}",
     "Бағдарлама {code} кодымен аяқталды"),
    ("Program ended with an error (exit code {code:int})",
     "Программа завершилась с ошибкой (код выхода {code})",
     "Бағдарлама қатемен аяқталды (шығу коды {code})"),
    ("Interrupted",
     "Прервано",
     "Үзілді"),
    ("the replay differs from the recorded run at line event {n:int}",
     "повторное выполнение расходится с записанным на событии строки №{n}",
     "қайта орындау жазылғаннан №{n} жол оқиғасында өзгеше"),
    ("Unknown command '{command}'",
     "Неизвестная команда '{command}'",
     "Белгісіз команда: '{command}'"),
    ("commands: b LINE, d LINE, c, s, n, out, back, rc, p NAME, locals, bt, q (help: details)",
     "команды: b LINE, d LINE, c, s, n, out, back, rc, p NAME, locals, bt, q (подробнее: help)",
     "командалар: b LINE, d LINE, c, s, n, out, back, rc, p NAME, locals, bt, q (толығырақ: help)"),
    ("Commands:",
     "Команды:",
     "Командалар:"),
    ("set a breakpoint (b alone: list them)",
     "поставить точку останова (b без аргумента: список)",
     "тоқтау нүктесін қою (аргументсіз b: тізімі)"),
    ("delete a breakpoint (d alone: delete all)",
     "удалить точку останова (d без аргумента: удалить все)",
     "тоқтау нүктесін жою (аргументсіз d: барлығын жою)"),
    ("continue to the next breakpoint",
     "продолжить до следующей точки останова",
     "келесі тоқтау нүктесіне дейін жалғастыру"),
    ("step to the next line, into calls",
     "шаг на следующую строку, с заходом в вызовы",
     "келесі жолға қадам, шақыруларға кіріп"),
    ("step to the next line, over calls",
     "шаг на следующую строку, без захода в вызовы",
     "келесі жолға қадам, шақыруларға кірмей"),
    ("run until the current function returns",
     "выполнять до возврата из текущей функции",
     "ағымдағы функция қайтқанға дейін орындау"),
    ("step back one line",
     "шаг на одну строку назад",
     "бір жол артқа қадам"),
    ("reverse-continue to the previous breakpoint",
     "выполнять назад до предыдущей точки останова",
     "алдыңғы тоқтау нүктесіне дейін кері орындау"),
    ("print a variable (also NAME.field, NAME[i])",
     "показать переменную (также NAME.field, NAME[i])",
     "айнымалыны көрсету (сондай-ақ NAME.field, NAME[i])"),
    ("print the local variables",
     "показать локальные переменные",
     "жергілікті айнымалыларды көрсету"),
    ("show the call stack",
     "показать стек вызовов",
     "шақырулар стегін көрсету"),
    ("quit",
     "выйти",
     "шығу"),
    ("Usage: {command} LINE",
     "Использование: {command} LINE",
     "Қолданылуы: {command} LINE"),
    ("Usage: p NAME",
     "Использование: p NAME",
     "Қолданылуы: p NAME"),
    ("No breakpoints",
     "Точек останова нет",
     "Тоқтау нүктелері жоқ"),
    ("Breakpoint {n:int} at line {line:int}",
     "Точка останова {n}: строка {line}",
     "Тоқтау нүктесі {n}: {line}-жол"),
    ("Breakpoint {n:int} at line {line:int} (line {asked:int} has no code)",
     "Точка останова {n}: строка {line} (в строке {asked} нет кода)",
     "Тоқтау нүктесі {n}: {line}-жол ({asked}-жолда код жоқ)"),
    ("No code at or after line {line:int}",
     "Нет кода в строке {line} и после неё",
     "{line}-жолда және одан кейін код жоқ"),
    ("Deleted all breakpoints",
     "Все точки останова удалены",
     "Барлық тоқтау нүктелері жойылды"),
    ("No breakpoint at line {line:int}",
     "В строке {line} нет точки останова",
     "{line}-жолда тоқтау нүктесі жоқ"),
    ("Deleted breakpoint {n:int} at line {line:int}",
     "Точка останова {n} (строка {line}) удалена",
     "Тоқтау нүктесі {n} ({line}-жол) жойылды"),
    ("The program has finished (back and rc go back in time, q quits)",
     "Программа завершилась (back и rc возвращают назад во времени, q — выход)",
     "Бағдарлама аяқталды (back пен rc уақытта артқа қайтарады, q — шығу)"),
    ("Already at the start of the program",
     "Уже в начале программы",
     "Бағдарламаның ең басындамыз"),
    ("No earlier breakpoint hit: back to the start of the program",
     "Раньше точки останова не срабатывали: возврат в начало программы",
     "Бұған дейін тоқтау нүктесі іске қосылмаған: бағдарламаның басына оралу"),
    ("No variable '{name}' here",
     "Здесь нет переменной '{name}'",
     "Мұнда '{name}' айнымалысы жоқ"),
    ("No local variables",
     "Локальных переменных нет",
     "Жергілікті айнымалылар жоқ"),
    ("... ({count:int} more characters)",
     "... (ещё символов: {count})",
     "... (тағы {count} таңба)"),
    ("Cannot evaluate '{expr}': use NAME, NAME.field or NAME[index]",
     "Не удаётся вычислить '{expr}': используйте NAME, NAME.field или NAME[index]",
     "'{expr}' есептеу мүмкін емес: NAME, NAME.field немесе NAME[index] қолданыңыз"),
    ("Cannot evaluate '{expr}': {reason:msg}",
     "Не удаётся вычислить '{expr}': {reason}",
     "'{expr}' есептеу мүмкін емес: {reason}"),

    # ── messages of til and of its tools (TIL 3.1 features) ────────────────
    ("Build failed: {path:text} not written",
     "Сборка не удалась: {path} не записан",
     "Құрастыру сәтсіз аяқталды: {path} жазылмады"),
    ("Build failed: nothing written",
     "Сборка не удалась: ничего не записано",
     "Құрастыру сәтсіз аяқталды: ештеңе жазылмады"),
    ("--device needs the fit checker (tilfit.py), which is not installed",
     "для --device нужен модуль проверки размещения (tilfit.py), но он не установлен",
     "--device үшін сыйымдылықты тексеру модулі (tilfit.py) керек, бірақ ол орнатылмаған"),
    ("--device works with `til build --target vm|c-vm` on a .til source; for a report on any program "
     "(also .tbc) use `til fit FILE --device NAME`",
     "--device работает с `til build --target vm|c-vm` для исходника .til; отчёт для любой программы "
     "(и для .tbc) даёт `til fit FILE --device NAME`",
     "--device `til build --target vm|c-vm` командасымен .til бастапқы коды үшін жұмыс істейді; кез "
     "келген бағдарлама (.tbc да) туралы есепті `til fit FILE --device NAME` береді"),
    ("[TIL] type inference skipped (internal error: {kind}: {detail:text})",
     "[TIL] вывод типов пропущен (внутренняя ошибка: {kind}: {detail})",
     "[TIL] типтерді шығару өткізіп жіберілді (ішкі қате: {kind}: {detail})"),
    ("run it on the TIL VM: `til run --vm` or `til build --target vm`",
     "запустите программу на TIL VM: `til run --vm` или `til build --target vm`",
     "бағдарламаны TIL VM-де іске қосыңыз: `til run --vm` немесе `til build --target vm`"),
    # a module that was not found: hints about packages (`import NAME`, docs/TIL_PACKAGES.md)
    ("'{name}' is a dependency in {manifest:text} but is not installed - run `til pkg install` in {dir:text}",
     "'{name}' — зависимость из {manifest}, но она не установлена: выполните `til pkg install` в {dir}",
     "'{name}' — {manifest} ішіндегі тәуелділік, бірақ ол орнатылмаған: {dir} ішінде `til pkg install` "
     "орындаңыз"),
    ("the package '{name}' ({dir:text}) has no entry file: til.json \"main\" if it is set, else "
     "{name}.til, lib.til or mod.til",
     "у пакета '{name}' ({dir}) нет входного файла: это \"main\" из til.json, если он задан, иначе "
     "{name}.til, lib.til или mod.til",
     "'{name}' пакетінің ({dir}) кіру файлы жоқ: ол til.json ішіндегі \"main\" (берілсе), әйтпесе "
     "{name}.til, lib.til немесе mod.til"),
    ("the package '{name}' ({dir:text}) has no module '{module}' ({path:text}.til or {path:text}/mod.til)",
     "в пакете '{name}' ({dir}) нет модуля '{module}' ({path}.til или {path}/mod.til)",
     "'{name}' пакетінде ({dir}) '{module}' модулі жоқ ({path}.til немесе {path}/mod.til)"),
    # portable snapshots: what is wrong with a .tilp file, in "the snapshot is damaged
    # or incomplete (...)" / "the program's state does not fit this VM (...)"
    ("truncated",
     "файл обрывается",
     "файл үзіліп қалған"),
    ("checksum",
     "не совпадает контрольная сумма",
     "бақылау сомасы сәйкес емес"),
    ("bad state",
     "неверное состояние",
     "күйі қате"),
    ("global count",
     "неверное число глобальных переменных",
     "ғаламдық айнымалылар саны қате"),
    ("no frames",
     "нет кадров вызовов",
     "шақыру кадрлары жоқ"),
    ("bad frame",
     "неверный кадр вызова",
     "шақыру кадры қате"),
    ("bad recovery point",
     "неверная точка восстановления",
     "қалпына келтіру нүктесі қате"),
    ("bad handler record",
     "неверная запись обработчика",
     "өңдеуші жазбасы қате"),
    ("bad screen size",
     "неверный размер экрана",
     "экран өлшемі қате"),
    ("bad map key",
     "неверный ключ словаря",
     "сөздік кілті қате"),
    ("object count",
     "неверное число объектов",
     "объектілер саны қате"),
    ("an upvalue is not a cell",
     "захваченная переменная не является ячейкой",
     "алынған айнымалы ұяшық емес"),
    ("a frame's closure is not a closure of its function",
     "замыкание кадра не относится к его функции",
     "кадрдың тұйықтамасы оның функциясына жатпайды"),
    ("bad value tag {tag:int}",
     "неверный тег значения {tag}",
     "мән тегі қате: {tag}"),
    ("bad object kind {kind:int}",
     "неверный вид объекта {kind}",
     "объект түрі қате: {kind}"),
    ("object id {id:int} out of order",
     "идентификатор объекта {id} не по порядку",
     "{id} объект идентификаторы ретсіз"),
    ("constant {index:int} is not a string constant",
     "константа {index} не является строковой",
     "{index} тұрақтысы жол тұрақтысы емес"),
    ("function {index:int} out of range",
     "функция {index} вне допустимого диапазона",
     "{index} функциясы рұқсат етілген ауқымнан тыс"),
    ("bad struct (type {type:int}, {count:int} fields)",
     "неверная структура (тип {type}, полей: {count})",
     "құрылым қате (тип {type}, өріс саны: {count})"),
    ("bad closure (function {index:int}, {count:int} upvalues)",
     "неверное замыкание (функция {index}, захваченных переменных: {count})",
     "тұйықтама қате (функция {index}, алынған айнымалылар саны: {count})"),
    ("version {version:int}, flags {flags:int}",
     "версия {version}, флаги {flags}",
     "нұсқа {version}, жалаушалар {flags}"),
    ("{count:int} call frames, at most {most:int}",
     "кадров вызовов: {count}, максимум {most}",
     "шақыру кадрлары: {count}, ең көбі {most}"),
    ("{count:int} effect handler entries, at most {most:int}",
     "записей обработчиков эффектов: {count}, максимум {most}",
     "әсер өңдеушілерінің жазбалары: {count}, ең көбі {most}"),
    ("cannot save a value of type {type}",
     "нельзя сохранить значение типа {type}",
     "{type} типті мәнді сақтау мүмкін емес"),
    ("load_portable() must come before the program runs",
     "load_portable() нужно вызвать до запуска программы",
     "load_portable() бағдарлама іске қосылмай тұрып шақырылуы керек"),
    # the C VM's runner (vm/main.c), printed as "tilvm: ..."
    ("cannot restore snapshot {path:text} (different program or VM build?)",
     "не удаётся восстановить снимок {path} (другая программа или сборка VM?)",
     "{path} күй көшірмесін қалпына келтіру мүмкін емес (басқа бағдарлама немесе VM құрастырмасы ма?)"),
    ("cannot write snapshot",
     "не удаётся записать снимок",
     "күй көшірмесін жазу мүмкін емес"),
    ("cannot load {path:text} (error {code:int})",
     "не удаётся загрузить {path} (ошибка {code})",
     "{path} жүктеу мүмкін емес (қате {code})"),
    ("cannot load {path:text} (error {code:int}: program uses floats, VM built without them)",
     "не удаётся загрузить {path} (ошибка {code}: программа использует числа с плавающей точкой, а VM "
     "собрана без них)",
     "{path} жүктеу мүмкін емес (қате {code}: бағдарлама нақты сандарды қолданады, ал VM оларсыз "
     "құрастырылған)"),
    ("cannot load {path:text} (error {code:int}: integer constant too large for this VM)",
     "не удаётся загрузить {path} (ошибка {code}: целая константа слишком велика для этой VM)",
     "{path} жүктеу мүмкін емес (қате {code}: бүтін сан тұрақтысы бұл VM үшін тым үлкен)"),
    ("cannot load {path:text} (error {code:int}: too many functions for this VM build (TILVM_MAX_FUNCS))",
     "не удаётся загрузить {path} (ошибка {code}: слишком много функций для этой сборки VM "
     "(TILVM_MAX_FUNCS))",
     "{path} жүктеу мүмкін емес (қате {code}: бұл VM құрастырмасы үшін функциялар тым көп "
     "(TILVM_MAX_FUNCS))"),
    ("cannot load {path:text} (error {code:int}: program larger than 64 KB (TILVM_ADDR16 build))",
     "не удаётся загрузить {path} (ошибка {code}: программа больше 64 КБ (сборка TILVM_ADDR16))",
     "{path} жүктеу мүмкін емес (қате {code}: бағдарлама 64 КБ-тан үлкен (TILVM_ADDR16 құрастырмасы))"),

    # ── til prove (src/tilprove.py; docs/TIL_PROOFS.md) ────────────────────
    # "til prove: ..." (the program name stays); the report on standard output is English
    ("{option} needs a number of milliseconds, got {value:text}",
     "{option}: нужно число миллисекунд, получено {value}",
     "{option} үшін миллисекундтар саны керек, бірақ {value} берілді"),
    ("{option} needs a number of megabytes (at least {least:int}), got {value:text}",
     "{option}: нужно число мегабайт (не меньше {least}), получено {value}",
     "{option} үшін мегабайттар саны керек (кемінде {least}), бірақ {value} берілді"),
    ("no function named '{name}' in {path:text}",
     "в {path} нет функции с именем '{name}'",
     "{path} ішінде '{name}' деп аталатын функция жоқ"),
    ("til prove needs the Z3 SMT solver, which is not installed.",
     "для til prove нужен SMT-решатель Z3, но он не установлен.",
     "til prove үшін Z3 SMT шешушісі керек, бірақ ол орнатылмаған."),
    ("Install it with:  {command:msg}",
     "Установите его командой:  {command}",
     "Оны мына командамен орнатыңыз:  {command}"),
    ("{count:int} #[level: 4] function(s) in {path:text} were not proved: the Z3 SMT solver is not "
     "installed ({command:msg})",
     "функции с #[level: 4] в {path} не доказаны (их число: {count}): SMT-решатель Z3 не установлен "
     "({command})",
     "{path} ішіндегі #[level: 4] функциялары дәлелденбеді (саны: {count}): Z3 SMT шешушісі орнатылмаған "
     "({command})"),

    # ── til fit (src/tilfit.py; docs/TIL_FIT.md) ───────────────────────────
    # "til fit: ..."; the report itself (standard output, --json) is English
    ("--device NAME is required (til fit --list-devices)",
     "нужен параметр --device NAME (til fit --list-devices)",
     "--device NAME параметрі керек (til fit --list-devices)"),
    ("file not found: {path:text}",
     "файл не найден: {path}",
     "файл табылмады: {path}"),
    ("unknown device '{name}'; known: {names:text}",
     "неизвестное устройство '{name}'; известные: {names}",
     "белгісіз құрылғы: '{name}'; белгілілері: {names}"),
    ("unknown device '{name}' (til fit --list-devices)",
     "неизвестное устройство '{name}' (til fit --list-devices)",
     "белгісіз құрылғы: '{name}' (til fit --list-devices)"),
    ("[til fit] building the device VM: {command:text}",
     "[til fit] сборка VM для устройства: {command}",
     "[til fit] құрылғыға арналған VM құрастырылуда: {command}"),
    ("cannot find the C VM sources (vm/tilvm.c); set TIL_HOME",
     "не найдены исходники C VM (vm/tilvm.c); задайте TIL_HOME",
     "C VM бастапқы кодтары табылмады (vm/tilvm.c); TIL_HOME айнымалысын орнатыңыз"),
    ("--run needs a C compiler (gcc, clang or cc)",
     "для --run нужен компилятор C (gcc, clang или cc)",
     "--run үшін C компиляторы керек (gcc, clang немесе cc)"),
    ("building the device VM failed:",
     "сборка VM для устройства не удалась:",
     "құрылғыға арналған VM құрастыру сәтсіз аяқталды:"),
    ("cannot read the bytecode: {reason:msg}",
     "не удаётся прочитать байт-код: {reason}",
     "байт-кодты оқу мүмкін емес: {reason}"),
    ("no entry function",
     "нет входной функции",
     "кіру функциясы жоқ"),
    ("bad comparison operand 0x{op}",
     "неверный операнд сравнения 0x{op}",
     "салыстыру операнды қате: 0x{op}"),
    ("cannot read device file {path:text}: {reason:text}",
     "не удаётся прочитать файл устройства {path}: {reason}",
     "{path} құрылғы файлын оқу мүмкін емес: {reason}"),
    ("device file {path:text}: expected a JSON object",
     "файл устройства {path}: ожидался объект JSON",
     "{path} құрылғы файлы: JSON нысаны күтілді"),
    ("device file {path:text}: name must be a string",
     "файл устройства {path}: name должно быть строкой",
     "{path} құрылғы файлы: name жол болуы керек"),
    ("device file {path:text}: unknown base device {name:text}",
     "файл устройства {path}: неизвестное базовое устройство {name}",
     "{path} құрылғы файлы: белгісіз негізгі құрылғы: {name}"),
    ("unknown device field(s): {fields:text}",
     "неизвестные поля устройства: {fields}",
     "құрылғының белгісіз өрістері: {fields}"),
    ("device '{name}': {problem:msg}",
     "устройство '{name}': {problem}",
     "'{name}' құрылғысы: {problem}"),
    ("{key} must be true or false",
     "{key} должно быть true или false",
     "{key} мәні true немесе false болуы керек"),
    ("{key} must be a string",
     "{key} должно быть строкой",
     "{key} жол болуы керек"),
    ("{key} must be a whole number",
     "{key} должно быть целым числом",
     "{key} бүтін сан болуы керек"),
    ("{key} must be >= 0",
     "{key} должно быть >= 0",
     "{key} мәні >= 0 болуы керек"),
    ("{key} must be 1..{most:int}",
     "{key} должно быть в диапазоне 1..{most}",
     "{key} мәні 1..{most} аралығында болуы керек"),
    ("{key} must be a positive number of bytes or null",
     "{key} должно быть положительным числом байт или null",
     "{key} оң байт саны немесе null болуы керек"),
    ("int_bits must be 16, 32 or 64",
     "int_bits должно быть 16, 32 или 64",
     "int_bits мәні 16, 32 немесе 64 болуы керек"),
    ("float must be 0, 1 or 2",
     "float должно быть 0, 1 или 2",
     "float мәні 0, 1 немесе 2 болуы керек"),
    ("float 2 (double) is 32-bit with the {toolchain} toolchain: use float 1",
     "с набором инструментов {toolchain} float 2 (double) 32-битный: используйте float 1",
     "{toolchain} құралдар жиынтығымен float 2 (double) 32-биттік: float 1 қолданыңыз"),
    ("heap must be 256..2^32-1 bytes",
     "heap должно быть от 256 до 2^32-1 байт",
     "heap мәні 256..2^32-1 байт аралығында болуы керек"),
    ("addr16 needs a heap of at most 65535 bytes",
     "для addr16 куча должна быть не больше 65535 байт",
     "addr16 үшін heap 65535 байттан аспауы керек"),
    ("stack must be 16..65535 slots",
     "stack должно быть от 16 до 65535 ячеек",
     "stack мәні 16..65535 ұяшық аралығында болуы керек"),
    ("frames must be 2..65535",
     "frames должно быть в диапазоне 2..65535",
     "frames мәні 2..65535 аралығында болуы керек"),
    ("consts must be 1..65535",
     "consts должно быть в диапазоне 1..65535",
     "consts мәні 1..65535 аралығында болуы керек"),
    ("types must be 3..65535",
     "types должно быть в диапазоне 3..65535",
     "types мәні 3..65535 аралығында болуы керек"),
    ("try_depth must be 1..65535",
     "try_depth должно быть в диапазоне 1..65535",
     "try_depth мәні 1..65535 аралығында болуы керек"),
    ("max_handlers must be 1..65535",
     "max_handlers должно быть в диапазоне 1..65535",
     "max_handlers мәні 1..65535 аралығында болуы керек"),
    ("mark_stack must be >= 1",
     "mark_stack должно быть >= 1",
     "mark_stack мәні >= 1 болуы керек"),
    ("max_funcs must be 0..65535 or null",
     "max_funcs должно быть в диапазоне 0..65535 или null",
     "max_funcs мәні 0..65535 аралығында немесе null болуы керек"),
    ("field_cache must be 0 or a power of two up to 32768, or null",
     "field_cache должно быть 0, степенью двойки до 32768 или null",
     "field_cache мәні 0, 32768-ге дейінгі екінің дәрежесі немесе null болуы керек"),
    ("rom must be a C identifier or empty",
     "rom должно быть идентификатором C или пустым",
     "rom C идентификаторы немесе бос болуы керек"),
    ("abi must be one of {choices:text}",
     "abi должно быть одним из: {choices}",
     "abi мына мәндердің бірі болуы керек: {choices}"),

    # ── til pkg (src/tilpkg.py; docs/TIL_PACKAGES.md) ──────────────────────
    # printed as "til pkg: error: ..." / "til pkg: warning: ..." (tilmsg's frames);
    # what `til pkg` prints on standard output (+ name, OUTDATED, ...) stays English.
    # Many messages start with where the problem is: a file, a URL, "til.json",
    # "registry URL (name 1.0.0)", "til pkg add NAME": the {where:msg} prefix.
    ("input nested too deeply to process (Python recursion limit)",
     "входные данные вложены слишком глубоко (предел рекурсии Python)",
     "кіріс деректері тым терең кірістірілген (Python рекурсия шегі)"),
    ("no such directory: {path:text}",
     "нет такого каталога: {path}",
     "мұндай каталог жоқ: {path}"),
    ("interrupted",
     "прервано",
     "үзілді"),
    ("{count:int} problem (`til pkg install` restores {dir:text}/ from til.lock)",
     "проблем: {count} (`til pkg install` восстанавливает {dir}/ по til.lock)",
     "{count} мәселе (`til pkg install` {dir}/ каталогын til.lock бойынша қалпына келтіреді)"),
    ("{count:int} problems (`til pkg install` restores {dir:text}/ from til.lock)",
     "проблем: {count} (`til pkg install` восстанавливает {dir}/ по til.lock)",
     "{count} мәселе (`til pkg install` {dir}/ каталогын til.lock бойынша қалпына келтіреді)"),
    # package names
    ("package names must be non-empty strings",
     "имя пакета должно быть непустой строкой",
     "пакет атауы бос емес жол болуы керек"),
    ("{where:msg}: package names must be non-empty strings",
     "{where}: имя пакета должно быть непустой строкой",
     "{where}: пакет атауы бос емес жол болуы керек"),
    ("invalid package name '{name}': a package name is the name you `import`, so it must be an identifier "
     "(letters, digits, '_'; at most 64 characters) - no '/', '\\', '.', '-' or spaces",
     "недопустимое имя пакета '{name}': имя пакета — это то, что пишется в `import`, поэтому оно должно "
     "быть идентификатором (буквы, цифры, '_'; не больше 64 символов) — без '/', '\\', '.', '-' и пробелов",
     "жарамсыз пакет атауы: '{name}': пакет атауы — `import` ішінде жазылатын атау, сондықтан ол "
     "идентификатор болуы керек (әріптер, цифрлар, '_'; ең көбі 64 таңба) — '/', '\\', '.', '-' және "
     "бос орындарсыз"),
    ("{where:msg}: invalid package name '{name}': a package name is the name you `import`, so it must be "
     "an identifier (letters, digits, '_'; at most 64 characters) - no '/', '\\', '.', '-' or spaces",
     "{where}: недопустимое имя пакета '{name}': имя пакета — это то, что пишется в `import`, поэтому оно "
     "должно быть идентификатором (буквы, цифры, '_'; не больше 64 символов) — без '/', '\\', '.', '-' и "
     "пробелов",
     "{where}: жарамсыз пакет атауы: '{name}': пакет атауы — `import` ішінде жазылатын атау, сондықтан "
     "ол идентификатор болуы керек (әріптер, цифрлар, '_'; ең көбі 64 таңба) — '/', '\\', '.', '-' және "
     "бос орындарсыз"),
    ("invalid package name '{name}': it is a TIL keyword",
     "недопустимое имя пакета '{name}': это ключевое слово TIL",
     "жарамсыз пакет атауы: '{name}': бұл TIL кілт сөзі"),
    ("{where:msg}: invalid package name '{name}': it is a TIL keyword",
     "{where}: недопустимое имя пакета '{name}': это ключевое слово TIL",
     "{where}: жарамсыз пакет атауы: '{name}': бұл TIL кілт сөзі"),
    ("invalid package name '{name}': '{dir}' is reserved (it is the directory packages are installed in)",
     "недопустимое имя пакета '{name}': имя '{dir}' зарезервировано (это каталог, куда устанавливаются "
     "пакеты)",
     "жарамсыз пакет атауы: '{name}': '{dir}' атауы резервтелген (пакеттер орнатылатын каталог)"),
    ("{where:msg}: invalid package name '{name}': '{dir}' is reserved (it is the directory packages are "
     "installed in)",
     "{where}: недопустимое имя пакета '{name}': имя '{dir}' зарезервировано (это каталог, куда "
     "устанавливаются пакеты)",
     "{where}: жарамсыз пакет атауы: '{name}': '{dir}' атауы резервтелген (пакеттер орнатылатын каталог)"),
    ("invalid package name '{name}': reserved file name on Windows",
     "недопустимое имя пакета '{name}': зарезервированное имя файла в Windows",
     "жарамсыз пакет атауы: '{name}': Windows жүйесінде резервтелген файл атауы"),
    ("{where:msg}: invalid package name '{name}': reserved file name on Windows",
     "{where}: недопустимое имя пакета '{name}': зарезервированное имя файла в Windows",
     "{where}: жарамсыз пакет атауы: '{name}': Windows жүйесінде резервтелген файл атауы"),
    # where a problem is (the {where:msg} prefix): a path, "til.json" (English), or
    ("{where:msg}: dependency",
     "{where}: зависимость",
     "{where}: тәуелділік"),
    ("registry {location:text} ({name} {version})",
     "реестр {location} ({name} {version})",
     "{location} тізілімі ({name} {version})"),
    ("registry {location:text}: package '{name}'",
     "реестр {location}: пакет '{name}'",
     "{location} тізілімі: '{name}' пакеті"),
    ("til.json of package '{name}' ({source:text})",
     "til.json пакета '{name}' ({source})",
     "'{name}' пакетінің til.json файлы ({source})"),
    ("cannot derive a package name from the directory name '{dir}': run `til pkg init NAME`",
     "не удаётся получить имя пакета из имени каталога '{dir}': выполните `til pkg init NAME`",
     "'{dir}' каталог атауынан пакет атауын шығару мүмкін емес: `til pkg init NAME` орындаңыз"),
    # versions
    ("invalid version {version:text} (expected MAJOR.MINOR.PATCH, e.g. \"1.2.0\")",
     "неверная версия {version} (ожидается MAJOR.MINOR.PATCH, например \"1.2.0\")",
     "жарамсыз нұсқа: {version} (MAJOR.MINOR.PATCH күтіледі, мысалы \"1.2.0\")"),
    ("{where:msg}: invalid version {version:text} (expected MAJOR.MINOR.PATCH, e.g. \"1.2.0\")",
     "{where}: неверная версия {version} (ожидается MAJOR.MINOR.PATCH, например \"1.2.0\")",
     "{where}: жарамсыз нұсқа: {version} (MAJOR.MINOR.PATCH күтіледі, мысалы \"1.2.0\")"),
    ("invalid version requirement '{req}'",
     "неверное требование к версии '{req}'",
     "нұсқаға қойылған талап жарамсыз: '{req}'"),
    # til.json, til.lock, the registry index
    ("{path:text}: invalid {what:msg} (not valid JSON: {reason:text})",
     "{path}: неверный {what} (это не корректный JSON: {reason})",
     "{path}: {what} жарамсыз (JSON дұрыс емес: {reason})"),
    ("{path:text}: invalid {what:msg} (JSON nested too deeply)",
     "{path}: неверный {what} (JSON вложен слишком глубоко)",
     "{path}: {what} жарамсыз (JSON тым терең кірістірілген)"),
    ("registry index",
     "индекс реестра",
     "тізілім индексі"),
    ("cannot read {path:text}: {reason:text}",
     "не удаётся прочитать {path}: {reason}",
     "{path} оқу мүмкін емес: {reason}"),
    ("{where:msg}: must contain a JSON object",
     "{where}: должен содержать объект JSON",
     "{where}: JSON нысаны болуы керек"),
    ("{where:msg}: \"dependencies\" must be an object {{\"name\": \"spec\"}}",
     "{where}: \"dependencies\" должно быть объектом {{\"name\": \"spec\"}}",
     "{where}: \"dependencies\" {{\"name\": \"spec\"}} нысаны болуы керек"),
    ("{where:msg}: \"main\" must be a relative path to a .til file inside the package",
     "{where}: \"main\" должно быть относительным путём к файлу .til внутри пакета",
     "{where}: \"main\" пакет ішіндегі .til файлына салыстырмалы жол болуы керек"),
    ("{where:msg}: \"registry\" must be a string (a path or URL of a registry index)",
     "{where}: \"registry\" должно быть строкой (путём или URL индекса реестра)",
     "{where}: \"registry\" жол болуы керек (тізілім индексінің жолы немесе URL)"),
    ("{path} not found",
     "{path} не найден",
     "{path} табылмады"),
    ("no til.json in {dir:text} or any parent directory (create one with `til pkg init`)",
     "нет til.json ни в {dir}, ни в родительских каталогах (создайте его командой `til pkg init`)",
     "{dir} ішінде де, жоғарғы каталогтарда да til.json жоқ (оны `til pkg init` арқылы жасаңыз)"),
    ("{path:text} is not a til.lock file (delete it, or run `til pkg install --update`)",
     "{path} — не файл til.lock (удалите его или выполните `til pkg install --update`)",
     "{path} til.lock файлы емес (оны жойыңыз немесе `til pkg install --update` орындаңыз)"),
    ("{path:text}: unsupported lockVersion {version:text} (this til writes version {ours:int})",
     "{path}: неподдерживаемая версия lockVersion {version} (этот til записывает версию {ours})",
     "{path}: lockVersion {version} нұсқасына қолдау жоқ (бұл til {ours} нұсқасын жазады)"),
    ("{path:text}: malformed entry for package '{name}' (delete til.lock or run `til pkg install --update`)",
     "{path}: повреждённая запись пакета '{name}' (удалите til.lock или выполните "
     "`til pkg install --update`)",
     "{path}: '{name}' пакетінің жазбасы бүлінген (til.lock жойыңыз немесе `til pkg install --update` "
     "орындаңыз)"),
    ("{path:text}: package '{name}': \"resolved\" must be a path, git+URL or archive URL",
     "{path}: пакет '{name}': \"resolved\" должно быть путём, git+URL или URL архива",
     "{path}: '{name}' пакеті: \"resolved\" жол, git+URL немесе архив URL болуы керек"),
    ("no til.lock in {dir:text} (run `til pkg install`)",
     "в {dir} нет til.lock (выполните `til pkg install`)",
     "{dir} ішінде til.lock жоқ (`til pkg install` орындаңыз)"),
    ("{path:text} already exists",
     "{path} уже существует",
     "{path} бұрыннан бар"),
    # dependency specs ("{where}: dependency 'x' = '...': ...")
    ("{where:msg}: dependency '{name}' needs a spec string (a path, git+URL, archive URL or version "
     "requirement)",
     "{where}: для зависимости '{name}' нужна строка-спецификация (путь, git+URL, URL архива или "
     "требование к версии)",
     "{where}: '{name}' тәуелділігіне спецификация жолы керек (жол, git+URL, архив URL немесе нұсқаға "
     "талап)"),
    ("{where:msg}: dependency '{name}': control characters in the spec",
     "{where}: зависимость '{name}': управляющие символы в спецификации",
     "{where}: '{name}' тәуелділігі: спецификацияда басқару таңбалары бар"),
    ("{where:msg}: dependency '{name}': the spec must be a string",
     "{where}: зависимость '{name}': спецификация должна быть строкой",
     "{where}: '{name}' тәуелділігі: спецификация жол болуы керек"),
    ("{where:msg}: dependency '{name}' = '{spec}' is a local path; only your project and packages that are "
     "local directories may have path dependencies",
     "{where}: зависимость '{name}' = '{spec}' — локальный путь; зависимости-пути допустимы только у "
     "вашего проекта и у пакетов, которые сами являются локальными каталогами",
     "{where}: '{name}' = '{spec}' тәуелділігі — жергілікті жол; жол-тәуелділіктер тек сіздің жобаңызда "
     "және өзі жергілікті каталог болып табылатын пакеттерде болады"),
    ("{where:msg}: dependency '{name}' = '{spec}' is not a path, git+URL, archive URL or version requirement",
     "{where}: зависимость '{name}' = '{spec}' — не путь, не git+URL, не URL архива и не требование к версии",
     "{where}: '{name}' = '{spec}' тәуелділігі жол да, git+URL да, архив URL да, нұсқаға талап та емес"),
    ("{where:msg}: dependency '{name}' = '{spec}' is not a path, git+URL, archive URL or version requirement "
     "(for the directory write ./{dir})",
     "{where}: зависимость '{name}' = '{spec}' — не путь, не git+URL, не URL архива и не требование к "
     "версии (для каталога напишите ./{dir})",
     "{where}: '{name}' = '{spec}' тәуелділігі жол да, git+URL да, архив URL да, нұсқаға талап та емес "
     "(каталог үшін ./{dir} деп жазыңыз)"),
    ("{where:msg}: dependency '{name}' = '{spec}': the registry name must be the dependency name "
     "(installing a package under another name is not supported)",
     "{where}: зависимость '{name}' = '{spec}': имя в реестре должно совпадать с именем зависимости "
     "(установка пакета под другим именем не поддерживается)",
     "{where}: '{name}' = '{spec}' тәуелділігі: тізілімдегі атау тәуелділік атауымен бірдей болуы керек "
     "(пакетті басқа атпен орнатуға қолдау жоқ)"),
    ("{where:msg}: dependency '{name}' = '{spec}': invalid git URL",
     "{where}: зависимость '{name}' = '{spec}': неверный git URL",
     "{where}: '{name}' = '{spec}' тәуелділігі: git URL жарамсыз"),
    ("{where:msg}: dependency '{name}' = '{spec}': invalid git revision '{rev}'",
     "{where}: зависимость '{name}' = '{spec}': неверная ревизия git '{rev}'",
     "{where}: '{name}' = '{spec}' тәуелділігі: git ревизиясы жарамсыз: '{rev}'"),
    ("{where:msg}: dependency '{name}' = '{spec}': unsupported git URL scheme '{scheme}' (use https, ssh, "
     "git or file)",
     "{where}: зависимость '{name}' = '{spec}': неподдерживаемая схема git URL '{scheme}' (используйте "
     "https, ssh, git или file)",
     "{where}: '{name}' = '{spec}' тәуелділігі: git URL схемасына қолдау жоқ: '{scheme}' (https, ssh, git "
     "немесе file қолданыңыз)"),
    ("{where:msg}: dependency '{name}' = '{spec}': a git repository given as a local path is only allowed in "
     "your project and in packages that are local directories (use git+file:///...)",
     "{where}: зависимость '{name}' = '{spec}': git-репозиторий, заданный локальным путём, допустим только "
     "в вашем проекте и в пакетах, которые являются локальными каталогами (используйте git+file:///...)",
     "{where}: '{name}' = '{spec}' тәуелділігі: жергілікті жолмен берілген git репозиторийіне тек сіздің "
     "жобаңызда және жергілікті каталог болып табылатын пакеттерде рұқсат етіледі (git+file:///... "
     "қолданыңыз)"),
    ("{where:msg}: dependency '{name}' = '{spec}': archives must be fetched over https:// (or file:// for a "
     "local archive)",
     "{where}: зависимость '{name}' = '{spec}': архивы загружаются только по https:// (или file:// для "
     "локального архива)",
     "{where}: '{name}' = '{spec}' тәуелділігі: архивтер тек https:// арқылы жүктеледі (жергілікті архив "
     "үшін file://)"),
    ("{where:msg}: dependency '{name}' = '{spec}': an archive URL must end in .tar.gz, .tgz or .zip",
     "{where}: зависимость '{name}' = '{spec}': URL архива должен оканчиваться на .tar.gz, .tgz или .zip",
     "{where}: '{name}' = '{spec}' тәуелділігі: архив URL .tar.gz, .tgz немесе .zip деп аяқталуы керек"),
    ("{where:msg}: dependency '{name}' = '{spec}': an archive URL must end in .tar.gz, .tgz or .zip (a local "
     "directory is a plain path; a git repository is git+file:///...)",
     "{where}: зависимость '{name}' = '{spec}': URL архива должен оканчиваться на .tar.gz, .tgz или .zip "
     "(локальный каталог указывается обычным путём, git-репозиторий — как git+file:///...)",
     "{where}: '{name}' = '{spec}' тәуелділігі: архив URL .tar.gz, .tgz немесе .zip деп аяқталуы керек "
     "(жергілікті каталог қарапайым жолмен, git репозиторийі git+file:///... түрінде беріледі)"),
    ("{where:msg}: dependency '{name}' = '{spec}': file:// URLs must name a local file "
     "(file:///path/to/archive)",
     "{where}: зависимость '{name}' = '{spec}': URL file:// должен указывать на локальный файл "
     "(file:///path/to/archive)",
     "{where}: '{name}' = '{spec}' тәуелділігі: file:// URL жергілікті файлды көрсетуі керек "
     "(file:///path/to/archive)"),
    ("{where:msg}: dependency '{name}' = '{spec}': unsupported URL fragment '#{fragment}' (expected "
     "#sha256=<64 hex digits>)",
     "{where}: зависимость '{name}' = '{spec}': неподдерживаемый фрагмент URL '#{fragment}' (ожидается "
     "#sha256=<64 шестнадцатеричные цифры>)",
     "{where}: '{name}' = '{spec}' тәуелділігі: URL фрагментіне қолдау жоқ: '#{fragment}' "
     "(#sha256=<64 он алтылық цифр> күтіледі)"),
    ("{url:text}: file:// URLs must name a local file",
     "{url}: URL file:// должен указывать на локальный файл",
     "{url}: file:// URL жергілікті файлды көрсетуі керек"),
    ("dependency '{name}' = '{spec}' is a version requirement, but no registry is configured: set "
     "TIL_REGISTRY or \"registry\" in til.json (a path or URL of a registry index), or give a path, git+URL "
     "or archive URL",
     "зависимость '{name}' = '{spec}' — требование к версии, но реестр не настроен: задайте TIL_REGISTRY "
     "или \"registry\" в til.json (путь или URL индекса реестра) либо укажите путь, git+URL или URL архива",
     "'{name}' = '{spec}' тәуелділігі — нұсқаға талап, бірақ тізілім бапталмаған: TIL_REGISTRY немесе "
     "til.json ішінде \"registry\" орнатыңыз (тізілім индексінің жолы немесе URL), немесе жол, git+URL "
     "не архив URL көрсетіңіз"),
    ("`til pkg add {name}` without a SPEC needs a registry: set TIL_REGISTRY or \"registry\" in til.json, "
     "or give a path, git+URL or archive URL",
     "для `til pkg add {name}` без SPEC нужен реестр: задайте TIL_REGISTRY или \"registry\" в til.json "
     "либо укажите путь, git+URL или URL архива",
     "SPEC көрсетілмеген `til pkg add {name}` үшін тізілім керек: TIL_REGISTRY немесе til.json ішінде "
     "\"registry\" орнатыңыз, немесе жол, git+URL не архив URL көрсетіңіз"),
    # the project
    ("'{name}' is this project: a package cannot depend on itself",
     "'{name}' — это сам проект: пакет не может зависеть от самого себя",
     "'{name}' — осы жобаның өзі: пакет өзіне тәуелді бола алмайды"),
    ("'{name}' is not a dependency in {path:text}",
     "'{name}' не является зависимостью в {path}",
     "'{name}' — {path} ішіндегі тәуелділік емес"),
    ("not a dependency of this project: {names:text}",
     "это не зависимости этого проекта: {names}",
     "бұл жобаның тәуелділіктері емес: {names}"),
    ("package names are only allowed with --update (`til pkg install --update NAME ...`)",
     "имена пакетов допустимы только с --update (`til pkg install --update NAME ...`)",
     "пакет атауларына тек --update бірге рұқсат етіледі (`til pkg install --update NAME ...`)"),
    ("--update and --frozen cannot be combined",
     "--update и --frozen нельзя использовать вместе",
     "--update және --frozen бірге қолданылмайды"),
    ("{dir:text} is inside an installed package ({modules:text}/); til pkg does not change installed copies "
     "(their own dependencies are installed flat in {flat:text}/ of the project) - run it in {project:text}",
     "{dir} находится внутри установленного пакета ({modules}/); til pkg не изменяет установленные копии "
     "(их собственные зависимости устанавливаются в общий {flat}/ проекта) — выполните команду в {project}",
     "{dir} орнатылған пакеттің ішінде ({modules}/); til pkg орнатылған көшірмелерді өзгертпейді (олардың "
     "өз тәуелділіктері жобаның ортақ {flat}/ каталогына орнатылады) — команданы {project} ішінде "
     "орындаңыз"),
    ("{path:text} is a symbolic link (-> {target:text}); til pkg manages {dir:text}/ itself and would delete "
     "files there, so it refuses to work through a link: remove the link (not the directory it points to) "
     "and run `til pkg install`",
     "{path} — символическая ссылка (-> {target}); til pkg сам управляет {dir}/ и удалял бы там файлы, "
     "поэтому через ссылку он работать отказывается: удалите ссылку (не каталог, на который она указывает) "
     "и выполните `til pkg install`",
     "{path} — символдық сілтеме (-> {target}); til pkg {dir}/ каталогын өзі басқарады және ондағы "
     "файлдарды жоюы мүмкін, сондықтан сілтеме арқылы жұмыс істеуден бас тартады: сілтемені жойыңыз (ол "
     "көрсететін каталогты емес) және `til pkg install` орындаңыз"),
    ("{path:text} is not a directory: remove it and run `til pkg install`",
     "{path} — не каталог: удалите его и выполните `til pkg install`",
     "{path} каталог емес: оны жойыңыз және `til pkg install` орындаңыз"),
    ("{path:text} resolves to {real:text}, outside the project; til pkg refuses to manage it",
     "{path} указывает на {real}, вне проекта; til pkg отказывается им управлять",
     "{path} жобадан тыс {real} орнын көрсетеді; til pkg оны басқарудан бас тартады"),
    # --frozen
    ("--frozen: {path:text} does not exist (run `til pkg install` first)",
     "--frozen: {path} не существует (сначала выполните `til pkg install`)",
     "--frozen: {path} жоқ (алдымен `til pkg install` орындаңыз)"),
    ("--frozen: til.lock is out of date with til.json (run `til pkg install`)",
     "--frozen: til.lock устарел относительно til.json (выполните `til pkg install`)",
     "--frozen: til.lock til.json бойынша ескірген (`til pkg install` орындаңыз)"),
    ("--frozen: til.lock is out of date with til.json: {problem:msg}",
     "--frozen: til.lock устарел относительно til.json: {problem}",
     "--frozen: til.lock til.json бойынша ескірген: {problem}"),
    ("(`til pkg install` resolves it again and writes til.lock; --frozen never does)",
     "(`til pkg install` разрешит зависимость заново и запишет til.lock; --frozen этого никогда не делает)",
     "(`til pkg install` тәуелділікті қайта шешіп, til.lock жазады; --frozen мұны ешқашан істемейді)"),
    ("--frozen: til.lock contradicts til.json: {problem:msg}",
     "--frozen: til.lock противоречит til.json: {problem}",
     "--frozen: til.lock til.json-ға қайшы: {problem}"),
    ("(til pkg never writes such an entry - was til.lock edited? `til pkg install` resolves the package from "
     "til.json and rewrites til.lock; --frozen never does)",
     "(til pkg никогда не записывает такую запись — til.lock редактировали вручную? `til pkg install` "
     "разрешит пакет по til.json и перезапишет til.lock; --frozen этого никогда не делает)",
     "(til pkg мұндай жазбаны ешқашан жазбайды — til.lock қолмен өзгертілді ме? `til pkg install` пакетті "
     "til.json бойынша қайта шешіп, til.lock қайта жазады; --frozen мұны ешқашан істемейді)"),
    ("--frozen: til.lock contradicts til.json: {name} is in til.lock, but nothing in til.json requires it "
     "any more",
     "--frozen: til.lock противоречит til.json: {name} есть в til.lock, но til.json его больше не требует",
     "--frozen: til.lock til.json-ға қайшы: {name} til.lock ішінде бар, бірақ til.json оны енді талап "
     "етпейді"),
    ("--frozen: til.lock contradicts til.json: {names:text} are in til.lock, but nothing in til.json "
     "requires them any more",
     "--frozen: til.lock противоречит til.json: {names} есть в til.lock, но til.json их больше не требует",
     "--frozen: til.lock til.json-ға қайшы: {names} til.lock ішінде бар, бірақ til.json оларды енді талап "
     "етпейді"),
    ("--frozen: til.lock is out of date with til.json: {name} is in til.lock, but nothing in til.json "
     "requires it any more",
     "--frozen: til.lock устарел относительно til.json: {name} есть в til.lock, но til.json его больше "
     "не требует",
     "--frozen: til.lock til.json бойынша ескірген: {name} til.lock ішінде бар, бірақ til.json оны енді "
     "талап етпейді"),
    ("--frozen: til.lock is out of date with til.json: {names:text} are in til.lock, but nothing in "
     "til.json requires them any more",
     "--frozen: til.lock устарел относительно til.json: {names} есть в til.lock, но til.json их больше "
     "не требует",
     "--frozen: til.lock til.json бойынша ескірген: {names} til.lock ішінде бар, бірақ til.json оларды "
     "енді талап етпейді"),
    # why an entry of til.lock does not fit til.json (--frozen, or "is not used")
    ("{requirer:text} requires {name} = {spec:text}, which is not in til.lock",
     "{requirer} требует {name} = {spec}, а этого нет в til.lock",
     "{requirer} {name} = {spec} талап етеді, бірақ ол til.lock ішінде жоқ"),
    ("{requirer:text} requires {name} = {spec:text}, til.lock has {name} = {locked:text}",
     "{requirer} требует {name} = {spec}, а в til.lock записано {name} = {locked}",
     "{requirer} {name} = {spec} талап етеді, ал til.lock ішінде {name} = {locked}"),
    ("til.lock resolves {name} to {resolved:text}, which is not the source {spec:text}",
     "til.lock разрешает {name} в {resolved}, а это не источник {spec}",
     "til.lock {name} пакетін {resolved} деп шешеді, бірақ бұл {spec} көзі емес"),
    ("til.lock resolves {name} to {resolved:text}, which does not name a commit by its full SHA",
     "til.lock разрешает {name} в {resolved}, а это не коммит, указанный полным SHA",
     "til.lock {name} пакетін {resolved} деп шешеді, бірақ бұл толық SHA арқылы көрсетілген коммит емес"),
    ("til.lock resolves {name} to {resolved:text}, but {spec:text} now names {path:text}",
     "til.lock разрешает {name} в {resolved}, но {spec} теперь указывает на {path}",
     "til.lock {name} пакетін {resolved} деп шешеді, бірақ {spec} енді {path} орнын көрсетеді"),
    ("til.lock resolves {name} to {resolved:text}, which is not the repository of {spec:text}",
     "til.lock разрешает {name} в {resolved}, а это не репозиторий {spec}",
     "til.lock {name} пакетін {resolved} деп шешеді, бірақ бұл {spec} репозиторийі емес"),
    ("til.lock resolves {name} to {resolved:text}, but {requirer:text} pins the commit {rev}",
     "til.lock разрешает {name} в {resolved}, но {requirer} закрепляет коммит {rev}",
     "til.lock {name} пакетін {resolved} деп шешеді, бірақ {requirer} {rev} коммитін бекітеді"),
    ("til.lock resolves {name} to {resolved:text}, but {requirer:text} pins {spec:text}",
     "til.lock разрешает {name} в {resolved}, но {requirer} закрепляет {spec}",
     "til.lock {name} пакетін {resolved} деп шешеді, бірақ {requirer} {spec} бекітеді"),
    ("{spec:text} does not pin the archive's sha256",
     "{spec} не закрепляет sha256 архива",
     "{spec} архивтің sha256 мәнін бекітпейді"),
    ("til.lock has no valid version for {name}",
     "в til.lock нет допустимой версии для {name}",
     "til.lock ішінде {name} үшін жарамды нұсқа жоқ"),
    ("{who:text} requires {name} {req:text}, til.lock has {name} {version}",
     "{who} требует {name} {req}, а в til.lock записано {name} {version}",
     "{who} {name} {req} талап етеді, ал til.lock ішінде {name} {version}"),
    ("the til.lock entry for '{name}' is not used: {problem:msg} (til pkg never writes such an entry - was "
     "til.lock edited?); '{name}' is resolved again",
     "запись til.lock для '{name}' не используется: {problem} (til pkg никогда не записывает такую запись — "
     "til.lock редактировали вручную?); '{name}' разрешается заново",
     "til.lock ішіндегі '{name}' жазбасы қолданылмайды: {problem} (til pkg мұндай жазбаны ешқашан жазбайды "
     "— til.lock қолмен өзгертілді ме?); '{name}' қайта шешіледі"),
    # resolving and fetching
    ("dependency cycle: {cycle:text}",
     "циклическая зависимость: {cycle}",
     "тәуелділіктер циклі: {cycle}"),
    ("could not find a consistent set of registry versions (gave up after {count:int} attempts)",
     "не удалось подобрать согласованный набор версий из реестра (попыток: {count}, поиск прекращён)",
     "тізілімнен үйлесімді нұсқалар жиынтығын табу мүмкін болмады ({count} әрекеттен кейін тоқтатылды)"),
    ("no version of '{name}' in the registry satisfies all requirements:",
     "ни одна версия '{name}' в реестре не удовлетворяет всем требованиям:",
     "тізілімдегі '{name}' нұсқаларының ешқайсысы барлық талаптарды қанағаттандырмайды:"),
    ("no version of '{name}' in the registry satisfies '{req}' (required by {requirer:text})",
     "ни одна версия '{name}' в реестре не удовлетворяет '{req}' (требует {requirer})",
     "тізілімдегі '{name}' нұсқаларының ешқайсысы '{req}' талабын қанағаттандырмайды (талап етуші: "
     "{requirer})"),
    ("  {who:text} requires {name} {req:text}",
     "  {who} требует {name} {req}",
     "  {who} {name} {req} талап етеді"),
    ("  available: {versions:text}",
     "  доступны: {versions}",
     "  қолжетімдісі: {versions}"),
    ("conflicting requirements for package '{name}':",
     "противоречивые требования к пакету '{name}':",
     "'{name}' пакетіне қойылған талаптар бір-біріне қайшы:"),
    ("  {who:text} requires {name} = {spec:text}",
     "  {who} требует {name} = {spec}",
     "  {who} {name} = {spec} талап етеді"),
    ("the first resolved to {name:text} from {source:text}; {dir:text}/ holds one copy of each package, so these "
     "requirements must agree",
     "первое разрешено в {name} из {source}; в {dir}/ хранится одна копия каждого пакета, поэтому эти "
     "требования должны совпадать",
     "біріншісі {source} көзінен {name} деп шешілді; {dir}/ ішінде әр пакеттің бір ғана көшірмесі "
     "сақталады, сондықтан бұл талаптар сәйкес келуі керек"),
    ("{requirer:text} requires '{name}', but the package at {path:text} is named '{other}' (installing a "
     "package under another name is not supported)",
     "{requirer} требует '{name}', но пакет в {path} называется '{other}' (установка пакета под другим "
     "именем не поддерживается)",
     "{requirer} '{name}' талап етеді, бірақ {path} ішіндегі пакеттің атауы '{other}' (пакетті басқа атпен "
     "орнатуға қолдау жоқ)"),
    ("the registry lists {name} {version} at {path:text}, but its til.json says version {other:text}",
     "в реестре указан {name} {version} в {path}, но в его til.json записана версия {other}",
     "тізілімде {name} {version} {path} орнында көрсетілген, бірақ оның til.json ішінде {other} нұсқасы "
     "жазылған"),
    ("packages '{name}' and '{other}' differ only in case; they cannot both be installed in {dir:text}/",
     "пакеты '{name}' и '{other}' отличаются только регистром букв; их нельзя установить вместе в {dir}/",
     "'{name}' және '{other}' пакеттері тек әріп регистрімен ерекшеленеді; екеуін бірге {dir}/ ішіне "
     "орнатуға болмайды"),
    ("package '{name}' = {spec:text}: a dependency cannot be the project itself or live inside {dir:text}/",
     "пакет '{name}' = {spec}: зависимость не может быть самим проектом или находиться внутри {dir}/",
     "'{name}' = {spec} пакеті: тәуелділік жобаның өзі бола алмайды немесе {dir}/ ішінде тұра алмайды"),
    ("package '{name}': cannot fetch a registry requirement directly",
     "пакет '{name}': требование к версии из реестра нельзя загрузить напрямую",
     "'{name}' пакеті: тізілімге қойылған талапты тікелей жүктеу мүмкін емес"),
    ("package '{name}': directory not found: {path:text} (paths are relative to the directory of the "
     "til.json that declares them)",
     "пакет '{name}': каталог не найден: {path} (пути отсчитываются от каталога til.json, в котором они "
     "записаны)",
     "'{name}' пакеті: каталог табылмады: {path} (жолдар оларды жариялайтын til.json каталогына "
     "қатысты)"),
    ("package '{name}': archive dependencies must pin the archive's sha256; its current hash is:",
     "пакет '{name}': зависимости-архивы должны закреплять sha256 архива; его текущий хеш:",
     "'{name}' пакеті: архив-тәуелділіктер архивтің sha256 мәнін бекітуі керек; оның қазіргі хеші:"),
    ("package '{name}': sha256 mismatch for {url:text}",
     "пакет '{name}': не совпадает sha256 для {url}",
     "'{name}' пакеті: {url} үшін sha256 сәйкес емес"),
    ("  expected {hash}",
     "  ожидался {hash}",
     "  күтілген {hash}"),
    ("  got      {hash}",
     "  получен  {hash}",
     "  алынған  {hash}"),
    ("the archive was not extracted",
     "архив не распакован",
     "архив ашылмады"),
    ("package '{name}': the files from {source:text} do not match til.lock",
     "пакет '{name}': файлы из {source} не совпадают с til.lock",
     "'{name}' пакеті: {source} файлдары til.lock-пен сәйкес емес"),
    ("  fetched  sha256 {hash}",
     "  загружено sha256 {hash}",
     "  жүктелген sha256 {hash}"),
    ("the source changed after til.lock was written; run `til pkg install --update {name}` to accept the new "
     "contents",
     "источник изменился после записи til.lock; чтобы принять новое содержимое, выполните "
     "`til pkg install --update {name}`",
     "til.lock жазылғаннан кейін көз өзгерген; жаңа мазмұнды қабылдау үшін `til pkg install --update {name}` "
     "орындаңыз"),
    ("package '{name}': its files changed while it was being installed",
     "пакет '{name}': его файлы изменились во время установки",
     "'{name}' пакеті: орнату кезінде оның файлдары өзгерді"),
    ("package '{name}' ({source:text}): its \"main\" file {path:text} does not exist",
     "пакет '{name}' ({source}): его файл \"main\" {path} не существует",
     "'{name}' пакеті ({source}): оның \"main\" файлы {path} жоқ"),
    ("package '{name}' has no \"main\", {name}.til, lib.til or mod.til: `import {name}` will not find it "
     "(only `import {name}.<module>`)",
     "у пакета '{name}' нет \"main\", {name}.til, lib.til или mod.til: `import {name}` его не найдёт "
     "(только `import {name}.<module>`)",
     "'{name}' пакетінде \"main\", {name}.til, lib.til немесе mod.til жоқ: `import {name}` оны таппайды "
     "(тек `import {name}.<module>`)"),
    ("package '{name}': git submodules are not fetched ({url:text}: {submodules:text})",
     "пакет '{name}': подмодули git не загружаются ({url}: {submodules})",
     "'{name}' пакеті: git ішкі модульдері жүктелмейді ({url}: {submodules})"),
    ("package '{name}': {rev} ({spec:text}) is the SHA of an annotated tag, not of a commit; it names the "
     "commit {sha}, and since til.lock records commits, every install checks it in the repository - pin "
     "the commit SHA to install without the network",
     "пакет '{name}': {rev} ({spec}) — это SHA аннотированного тега, а не коммита; он указывает на коммит "
     "{sha}, и поскольку til.lock записывает коммиты, каждая установка проверяет его в репозитории — "
     "закрепите SHA коммита, чтобы устанавливать без сети",
     "'{name}' пакеті: {rev} ({spec}) — коммиттің емес, аннотацияланған тегтің SHA мәні; ол {sha} коммитін "
     "көрсетеді, ал til.lock коммиттерді жазатындықтан, әр орнату оны репозиторийден тексереді — желісіз "
     "орнату үшін коммиттің SHA мәнін бекітіңіз"),
    # the registry
    ("registry index not found: {path:text}",
     "индекс реестра не найден: {path}",
     "тізілім индексі табылмады: {path}"),
    ("registry {location:text}: use https:// (or a local file)",
     "реестр {location}: используйте https:// (или локальный файл)",
     "{location} тізілімі: https:// қолданыңыз (немесе жергілікті файл)"),
    ("registry {location:text}: the index must be a JSON object",
     "реестр {location}: индекс должен быть объектом JSON",
     "{location} тізілімі: индекс JSON нысаны болуы керек"),
    ("package '{name}' is not in the registry {location:text}",
     "пакета '{name}' нет в реестре {location}",
     "'{name}' пакеті {location} тізілімінде жоқ"),
    ("registry {location:text}: '{name}' must map versions to sources",
     "реестр {location}: '{name}' должен сопоставлять версиям источники",
     "{location} тізілімі: '{name}' нұсқаларды көздерге сәйкестендіруі керек"),
    ("registry {location:text}: {name} {version:text}: the source must be a string",
     "реестр {location}: {name} {version}: источник должен быть строкой",
     "{location} тізілімі: {name} {version}: көз жол болуы керек"),
    ("registry {location:text}: {name} {version:text}: the source must be a path, git+URL or archive URL, "
     "not '{source}'",
     "реестр {location}: {name} {version}: источник должен быть путём, git+URL или URL архива, а не "
     "'{source}'",
     "{location} тізілімі: {name} {version}: көз жол, git+URL немесе архив URL болуы керек, '{source}' емес"),
    ("registry {location:text}: package '{name}': invalid version {version:text} (expected MAJOR.MINOR.PATCH, "
     "e.g. \"1.2.0\")",
     "реестр {location}: пакет '{name}': неверная версия {version} (ожидается MAJOR.MINOR.PATCH, например "
     "\"1.2.0\")",
     "{location} тізілімі: '{name}' пакеті: жарамсыз нұсқа: {version} (MAJOR.MINOR.PATCH күтіледі, мысалы "
     "\"1.2.0\")"),
    # downloads and archives
    ("cannot download {url:text}: HTTP {code:int} {reason:text}",
     "не удаётся загрузить {url}: HTTP {code} {reason}",
     "{url} жүктеу мүмкін емес: HTTP {code} {reason}"),
    ("cannot download {url:text}: it redirects to {target:text} (only https:// is allowed)",
     "не удаётся загрузить {url}: он перенаправляет на {target} (допустим только https://)",
     "{url} жүктеу мүмкін емес: ол {target} мекенжайына бағыттайды (тек https:// рұқсат етілген)"),
    ("cannot download {url:text}: it was redirected to {target:text} (only https:// is allowed)",
     "не удаётся загрузить {url}: произошло перенаправление на {target} (допустим только https://)",
     "{url} жүктеу мүмкін емес: ол {target} мекенжайына бағытталды (тек https:// рұқсат етілген)"),
    ("cannot download {url:text}: only https:// and file:// URLs are supported",
     "не удаётся загрузить {url}: поддерживаются только URL https:// и file://",
     "{url} жүктеу мүмкін емес: тек https:// және file:// URL мекенжайларына қолдау бар"),
    ("cannot download {url:text}: {reason:text}",
     "не удаётся загрузить {url}: {reason}",
     "{url} жүктеу мүмкін емес: {reason}"),
    ("{url:text}: download larger than {size:int} MB",
     "{url}: загрузка больше {size} МБ",
     "{url}: жүктелетін дерек {size} МБ-тан үлкен"),
    ("{url:text}: download larger than {size:int} bytes",
     "{url}: загрузка больше {size} байт",
     "{url}: жүктелетін дерек {size} байттан үлкен"),
    ("cannot extract {archive:text}: the archive is damaged or not a valid .tar.gz file ({reason:text})",
     "не удаётся распаковать {archive}: архив повреждён или это не корректный файл .tar.gz ({reason})",
     "{archive} ашу мүмкін емес: архив бүлінген немесе дұрыс .tar.gz файлы емес ({reason})"),
    ("cannot extract {archive:text}: the archive is damaged or not a valid .zip file ({reason:text})",
     "не удаётся распаковать {archive}: архив повреждён или это не корректный файл .zip ({reason})",
     "{archive} ашу мүмкін емес: архив бүлінген немесе дұрыс .zip файлы емес ({reason})"),
    ("cannot extract {archive:text}: {reason:text}",
     "не удаётся распаковать {archive}: {reason}",
     "{archive} ашу мүмкін емес: {reason}"),
    ("{archive:text}: unsafe entry name {name:text} (control characters)",
     "{archive}: небезопасное имя элемента {name} (управляющие символы)",
     "{archive}: элемент атауы қауіпті: {name} (басқару таңбалары)"),
    ("{archive:text}: unsafe entry '{name}' (absolute path)",
     "{archive}: небезопасный элемент '{name}' (абсолютный путь)",
     "{archive}: қауіпті элемент: '{name}' (абсолюттік жол)"),
    ("{archive:text}: unsafe entry '{name}' ('..' would escape the package directory)",
     "{archive}: небезопасный элемент '{name}' ('..' вывел бы за пределы каталога пакета)",
     "{archive}: қауіпті элемент: '{name}' ('..' пакет каталогынан шығарып жібереді)"),
    ("{archive:text}: unsafe entry '{name}' (':' in a file name)",
     "{archive}: небезопасный элемент '{name}' (':' в имени файла)",
     "{archive}: қауіпті элемент: '{name}' (файл атауында ':' бар)"),
    ("{archive:text}: entry '{name}' escapes the package directory",
     "{archive}: элемент '{name}' выходит за пределы каталога пакета",
     "{archive}: '{name}' элементі пакет каталогынан шығып кетеді"),
    ("{archive:text}: '{name}' is both a file and a directory",
     "{archive}: '{name}' — одновременно файл и каталог",
     "{archive}: '{name}' әрі файл, әрі каталог"),
    ("{archive:text}: file entry without a name",
     "{archive}: элемент-файл без имени",
     "{archive}: атауы жоқ файл элементі"),
    ("{archive:text}: duplicate entry '{name}'",
     "{archive}: повторяющийся элемент '{name}'",
     "{archive}: қайталанған элемент: '{name}'"),
    ("{archive:text}: '{name}' is inside a file",
     "{archive}: '{name}' находится внутри файла",
     "{archive}: '{name}' файлдың ішінде тұр"),
    ("{archive:text}: entry '{name}' collides with another entry",
     "{archive}: элемент '{name}' совпадает с другим элементом",
     "{archive}: '{name}' элементі басқа элементпен соқтығысады"),
    ("{archive:text}: entry '{name}' is a symbolic link (links are not allowed in packages)",
     "{archive}: элемент '{name}' — символическая ссылка (ссылки в пакетах не допускаются)",
     "{archive}: '{name}' элементі — символдық сілтеме (пакеттерде сілтемелерге рұқсат жоқ)"),
    ("{archive:text}: entry '{name}' is a symbolic link -> '{target}' (links are not allowed in packages)",
     "{archive}: элемент '{name}' — символическая ссылка -> '{target}' (ссылки в пакетах не допускаются)",
     "{archive}: '{name}' элементі — символдық сілтеме -> '{target}' (пакеттерде сілтемелерге рұқсат жоқ)"),
    ("{archive:text}: entry '{name}' is a hard link -> '{target}' (links are not allowed in packages)",
     "{archive}: элемент '{name}' — жёсткая ссылка -> '{target}' (ссылки в пакетах не допускаются)",
     "{archive}: '{name}' элементі — қатаң сілтеме -> '{target}' (пакеттерде сілтемелерге рұқсат жоқ)"),
    ("{archive:text}: entry '{name}' is not a regular file or directory",
     "{archive}: элемент '{name}' — не обычный файл и не каталог",
     "{archive}: '{name}' элементі кәдімгі файл да, каталог та емес"),
    ("{archive:text}: entry '{name}' is not a regular file",
     "{archive}: элемент '{name}' — не обычный файл",
     "{archive}: '{name}' элементі кәдімгі файл емес"),
    ("{archive:text}: more than {count:int} files",
     "{archive}: больше {count} файлов",
     "{archive}: {count} файлдан көп"),
    ("{archive:text}: more than {size:int} MB of files",
     "{archive}: больше {size} МБ файлов",
     "{archive}: файлдар {size} МБ-тан көп"),
    ("{archive:text}: more than {size:int} bytes of files",
     "{archive}: больше {size} байт файлов",
     "{archive}: файлдар {size} байттан көп"),
    ("{archive:text}: more than {size:int} MB of data after the end of the archive",
     "{archive}: больше {size} МБ данных после конца архива",
     "{archive}: архив соңынан кейін {size} МБ-тан көп дерек бар"),
    ("{archive:text}: more than {size:int} bytes of data after the end of the archive",
     "{archive}: больше {size} байт данных после конца архива",
     "{archive}: архив соңынан кейін {size} байттан көп дерек бар"),
    # the files of a package
    ("{dir:text}: more than {count:int} files in one package",
     "{dir}: больше {count} файлов в одном пакете",
     "{dir}: бір пакетте {count} файлдан көп"),
    ("{dir:text}: package larger than {size:int} MB",
     "{dir}: пакет больше {size} МБ",
     "{dir}: пакет {size} МБ-тан үлкен"),
    ("{dir:text}: package larger than {size:int} bytes",
     "{dir}: пакет больше {size} байт",
     "{dir}: пакет {size} байттан үлкен"),
    ("{dir:text}: '{name}' was added after installation (til pkg never installs {skipped} directories)",
     "{dir}: '{name}' добавлен после установки (til pkg никогда не устанавливает каталоги {skipped})",
     "{dir}: '{name}' орнатудан кейін қосылған (til pkg {skipped} каталогтарын ешқашан орнатпайды)"),
    ("{dir:text}: file name with a line break: {name:text}",
     "{dir}: имя файла с переводом строки: {name}",
     "{dir}: файл атауында жол ауыстыру бар: {name}"),
    ("{dir:text}: '{name}' is not a regular file or directory",
     "{dir}: '{name}' — не обычный файл и не каталог",
     "{dir}: '{name}' кәдімгі файл да, каталог та емес"),
    ("{dir:text}: two files differ only in case ('{name}'): they cannot be installed on this file system",
     "{dir}: два файла отличаются только регистром букв ('{name}'): их нельзя установить в этой файловой "
     "системе",
     "{dir}: екі файл тек әріп регистрімен ерекшеленеді ('{name}'): оларды бұл файлдық жүйеге орнату "
     "мүмкін емес"),
    # git
    ("git is not installed (it is needed for git+ dependencies)",
     "git не установлен (он нужен для зависимостей git+)",
     "git орнатылмаған (ол git+ тәуелділіктері үшін керек)"),
    ("{what:msg}: git {command} failed",
     "{what}: команда git {command} не выполнена",
     "{what}: git {command} командасы орындалмады"),
    ("{what:msg}: git {command} failed:",
     "{what}: команда git {command} не выполнена:",
     "{what}: git {command} командасы орындалмады:"),
    ("git {url:text}: no commit, tag or branch '{rev}'",
     "git {url}: нет коммита, тега или ветки '{rev}'",
     "git {url}: '{rev}' деген коммит, тег немесе тармақ жоқ"),
    ("{label:msg}: the file name {name:text} is not UTF-8",
     "{label}: имя файла {name} не в UTF-8",
     "{label}: {name} файл атауы UTF-8 емес"),
    ("{label:msg}: unsafe path '{path}' in the repository",
     "{label}: небезопасный путь '{path}' в репозитории",
     "{label}: репозиторийде қауіпті жол бар: '{path}'"),
    ("{label:msg}: the file name {name:text} cannot be installed on every OS (control characters, '\\' "
     "or ':')",
     "{label}: файл с именем {name} нельзя установить на любой ОС (управляющие символы, '\\' или ':')",
     "{label}: {name} атаулы файлды кез келген ОЖ-ге орнату мүмкін емес (басқару таңбалары, '\\' немесе "
     "':')"),
    ("{label:msg}: git cat-file output ended early",
     "{label}: вывод git cat-file оборвался",
     "{label}: git cat-file шығысы ерте үзілді"),
    ("{label:msg}: unexpected git ls-tree output {output:text}",
     "{label}: неожиданный вывод git ls-tree {output}",
     "{label}: git ls-tree шығысы күтпеген: {output}"),
    ("{label:msg}: '{name}' is a symbolic link (packages may only contain plain files and directories)",
     "{label}: '{name}' — символическая ссылка (в пакетах допустимы только обычные файлы и каталоги)",
     "{label}: '{name}' — символдық сілтеме (пакеттерде тек кәдімгі файлдар мен каталогтар болады)"),
    ("{label:msg}: '{name}' is not a regular file (git mode {mode})",
     "{label}: '{name}' — не обычный файл (режим git {mode})",
     "{label}: '{name}' кәдімгі файл емес (git режимі {mode})"),
    ("{label:msg}: cannot read '{name}' from the repository (git cat-file: {reason:text})",
     "{label}: не удаётся прочитать '{name}' из репозитория (git cat-file: {reason})",
     "{label}: репозиторийден '{name}' оқу мүмкін емес (git cat-file: {reason})"),
    ("{label:msg}: unexpected git cat-file output after '{name}'",
     "{label}: неожиданный вывод git cat-file после '{name}'",
     "{label}: '{name}' кейін git cat-file шығысы күтпеген"),
]

# Every entry, in lookup order (read-only: add entries to the lists above).
MESSAGES: List[Tuple[str, str, str]] = COMPILER_MESSAGES + RUNTIME_MESSAGES + COMMAND_LINE_MESSAGES
SCOPES: Tuple[str, ...] = ('compiler', 'runtime', 'command-line')

# English-only on purpose (checked by tools/i18n_coverage.py).
UNTRANSLATED: Tuple[str, ...] = (
    "panic",            # the default message of panic(): the name of the builtin
    # code, file names and keywords inside translated messages ({x:msg} parts):
    # "native backend: str.upper() (line 3)", "invalid til.json (...)", "let 'x' at a.til:1"
    "{name}()", "Option.{method}()", "Result.{method}()", "str.{method}()", "array.{method}()",
    "map.{method}()", "'{name}.{method}(...)'", "{left:type} {op} {right:type}",
    "#[ensures: {cond:text}]",          # a proof obligation that is the contract itself
    "til.json", "til.lock",
    "let", "var", "const",
    "  {url:text}#sha256={hash}",       # the dependency spec to write (til pkg)
    "til pkg add {name}", "git {url}", "git {url:text} ({sha})",     # where a til pkg problem is
    "pip install z3-solver",            # how to install Z3 (til prove)
    "{where:text}: \"name\"", "{where:text}: \"version\"",
    "  til.lock sha256 {hash}",
    # internal: til debug's copy of the VM loop does not fit the installed til.py
    "the source of TILVM.execute() is not available ({reason:text})",
    "TILVM.execute() has no `# @line-event` marker (src/til.py and src/tildebug.py do not belong together)",
    # printed at start-up, before the language is chosen (TIL_LANG names no
    # known language, and --lang / the program's pragma are not applied yet)
    "TIL_LANG={value} is not a message language (en, ru, kk); it is ignored",
    # printed exactly when this catalog cannot be used
    "message translations are not installed (tilmsg.py is missing); messages stay in English",
    "message translations could not be loaded ({problem:text}); messages stay in English",
)


# ═══════════════════════════════════════════════════════════════════════════
# templates
# ═══════════════════════════════════════════════════════════════════════════

Part = Union[str, Tuple[str, str]]          # literal text, or (placeholder name, kind)

# What a placeholder accepts when a message is matched (`{name}` / `{name:kind}`):
#   ''         a word: at least one character, no whitespace; between matching
#              quotes ('{x}', "{x}", `{x}`) any text on one line
#   'text'     any text: a condition, a path, a key, a message written by a user
#   'int'      an integer
#   'msg'      a nested message, translated too (any text)
#   'builtin'  the name of a builtin function: lowercase letters, digits and _
#   'type'     a type as the VMs name it: nil, bool, int, float, str, array,
#              map, fn, struct, or a capitalized name (Point, Shape.Circle, Нүкте);
#              also as the typed backend writes it (E-TYPE): [int], {str: int},
#              Option<T>, Result<T, E>, fn(int) -> str, str or int, ?3, None
# So free text in a program's own message ("Expected 3 items, got 2") does not
# fit "Expected {expected}, got {got}", whose placeholders are words, and
# "cannot compare apples and oranges" does not fit "cannot compare {left:type}
# and {right:type}".
KINDS = ('', 'text', 'int', 'msg', 'builtin', 'type')

MAX_LOOKUP = 10000      # a longer message is not looked up: it stays English
MAX_NESTING = 32        # {x:msg} placeholders inside each other: deeper ones stay English

_TOKEN = re.compile(r'\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)(?::([a-z]+))?\}|[{}]')
_QUOTES = "'\"`"
_SPACE = re.compile(r'\s')
_INT = re.compile(r'-?[0-9]+')
_BUILTIN = re.compile(r'[a-z_][a-z0-9_]*')
_TYPE_NAME = re.compile(r'[^\W\d_]\w*(?:\.[^\W\d_]\w*)*')
BUILTIN_TYPES = frozenset(('nil', 'bool', 'int', 'float', 'str', 'array', 'map', 'fn', 'struct', '?'))
_TYPE_WORDS = BUILTIN_TYPES | {'dynamic', 'unknown'}     # (tiltypes.show(): E-TYPE messages)
_TYPE_VAR = re.compile(r'\?[0-9]+')
MAX_TYPE_TEXT = 300     # a longer value is no type (and the parser below stays cheap)


def _type_name(value: str, pos: int) -> int:
    """End of the type name at value[pos:] (int, Point, Shape.Circle, ?3), or -1."""
    m = _TYPE_VAR.match(value, pos) or _TYPE_NAME.match(value, pos)
    if m is None:
        return -1
    word = m.group(0)
    if word in _TYPE_WORDS or word[0] == '?' or word[0].isupper():
        return m.end()
    return -1


def _type_expr(value: str, pos: int, depth: int) -> int:
    """End of the type at value[pos:] as tiltypes.show() writes it, or -1:
    NAME, [T], {K: V}, Option<T>, Result<T, E>, fn(A, B) -> R, and `str or T`.
    Recursive descent without backtracking: linear in the length."""
    if depth > 20:
        return -1
    end = -1
    if value.startswith('[', pos):
        end = _type_expr(value, pos + 1, depth + 1)
        end = end + 1 if end >= 0 and value.startswith(']', end) else -1
    elif value.startswith('{', pos):
        end = _type_expr(value, pos + 1, depth + 1)
        end = _type_expr(value, end + 2, depth + 1) if end >= 0 and value.startswith(': ', end) else -1
        end = end + 1 if end >= 0 and value.startswith('}', end) else -1
    elif value.startswith('Option<', pos):
        end = _type_expr(value, pos + 7, depth + 1)
        end = end + 1 if end >= 0 and value.startswith('>', end) else -1
    elif value.startswith('Result<', pos):
        end = _type_expr(value, pos + 7, depth + 1)
        end = _type_expr(value, end + 2, depth + 1) if end >= 0 and value.startswith(', ', end) else -1
        end = end + 1 if end >= 0 and value.startswith('>', end) else -1
    elif value.startswith('fn(', pos):
        end = pos + 3
        if not value.startswith(')', end):
            end = _type_expr(value, end, depth + 1)
            while end >= 0 and value.startswith(', ', end):
                end = _type_expr(value, end + 2, depth + 1)
        end = _type_expr(value, end + 5, depth + 1) if end >= 0 and value.startswith(') -> ', end) else -1
    else:
        end = _type_name(value, pos)
    if end >= 0 and value.startswith(' or ', end):        # "str or int": a string or a T
        end = _type_expr(value, end + 4, depth + 1)
    return end


def is_type(value: str) -> bool:
    """A type as the VMs or the typed backend name it (the 'type' placeholder kind)."""
    if value in BUILTIN_TYPES:
        return True
    if len(value) > MAX_TYPE_TEXT:
        return False
    return _type_expr(value, 0, 0) == len(value)
_NON_EMPTY = ('word', 'int', 'builtin', 'type')


def parse_template(text: str) -> List[Part]:
    """'no field {f} on {t}' -> ['no field ', ('f', ''), ' on ', ('t', '')]."""
    parts: List[Part] = []
    pos = 0
    for m in _TOKEN.finditer(text):
        if m.start() > pos:
            parts.append(text[pos:m.start()])
        tok = m.group(0)
        if tok in ('{{', '}}'):
            parts.append(tok[0])
        elif m.group(1):
            kind = m.group(2) or ''
            if kind not in KINDS:
                raise ValueError(f"unknown placeholder kind '{kind}' in {text!r} (use: {', '.join(KINDS[1:])})")
            parts.append((m.group(1), kind))
        else:
            raise ValueError(f"unescaped '{tok}' in {text!r} (write '{tok}{tok}')")
        pos = m.end()
    if pos < len(text):
        parts.append(text[pos:])
    merged: List[Part] = []
    for p in parts:
        if isinstance(p, str) and merged and isinstance(merged[-1], str):
            merged[-1] += p
        else:
            merged.append(p)
    return merged


def _accepts(shape: str, value: str) -> bool:
    """Is `value` text that a placeholder of this shape stands for?"""
    if shape == 'word':
        return value != '' and _SPACE.search(value) is None
    if shape == 'quoted':
        return '\n' not in value
    if shape == 'int':
        return _INT.fullmatch(value) is not None
    if shape == 'builtin':
        return _BUILTIN.fullmatch(value) is not None
    if shape == 'type':
        return is_type(value)
    return True                                     # 'text'


class Entry:
    """A catalog entry: the English template, how it matches, its translations."""

    def __init__(self, en: str, *translations: str, scope: str = 'compiler'):
        self.en = en
        self.scope = scope
        self.parts = parse_template(en)
        self.kinds: Dict[str, str] = {}
        self.shapes: List[Optional[str]] = []      # per part: None for literal text
        for i, p in enumerate(self.parts):
            if isinstance(p, str):
                self.shapes.append(None)
                continue
            if i and not isinstance(self.parts[i - 1], str):
                raise ValueError(f"adjacent placeholders in {en!r} cannot be told apart")
            self.kinds.setdefault(p[0], p[1])
            self.shapes.append(self._shape(i, p[1]))
        self.literal_size = sum(len(p) for p in self.parts if isinstance(p, str))
        first = self.parts[0] if self.parts else ''
        self.prefix = first if isinstance(first, str) else ''     # the literal text it starts with
        self.leading = None if isinstance(first, str) else first[0]
        self.texts: Dict[str, str] = {'en': en}
        self.targets: Dict[str, List[Part]] = {'en': self.parts}
        for lang, text in zip(COLUMNS[1:], translations):
            self.texts[lang] = text
            self.targets[lang] = parse_template(text)

    def _shape(self, i: int, kind: str) -> str:
        if kind == '':
            before = self.parts[i - 1] if i > 0 else ''
            after = self.parts[i + 1] if i + 1 < len(self.parts) else ''
            if isinstance(before, str) and isinstance(after, str) and before and after \
                    and before[-1] in _QUOTES and after[0] == before[-1]:
                return 'quoted'
            return 'word'
        return 'text' if kind == 'msg' else kind

    def match(self, message: str, any_value: Optional[str] = None) -> Optional[Dict[str, str]]:
        """The placeholder values when `message` is this template, else None.

        Linear time: a placeholder ends where the literal text after it first
        occurs (the last literal text ends the message). There is no
        backtracking, so a value that contains the text following it does not
        match: the message stays English. A value equal to `any_value` fits
        every kind (tools/i18n_coverage.py marks placeholders that way)."""
        parts, shapes = self.parts, self.shapes
        n, pos, last = len(message), 0, len(parts) - 1
        groups: Dict[str, str] = {}
        for i, p in enumerate(parts):
            if isinstance(p, str):
                if not message.startswith(p, pos):
                    return None
                pos += len(p)
                continue
            name = p[0]
            if name in groups:                       # the same name again: the same text
                if not message.startswith(groups[name], pos):
                    return None
                pos += len(groups[name])
                continue
            shape = shapes[i]
            if i == last:
                end = n
            elif i + 1 == last:
                end = n - len(parts[i + 1])
            else:
                end = message.find(parts[i + 1], pos + 1 if shape in _NON_EMPTY else pos)
            if end < pos:
                return None
            value = message[pos:end]
            if value != any_value and not _accepts(shape, value):
                return None
            groups[name] = value
            pos = end
        return groups if pos == n else None

    def render(self, mark: str) -> str:
        """The English text with every placeholder replaced by `mark`."""
        return ''.join(p if isinstance(p, str) else mark for p in self.parts)

    def apply(self, groups: Dict[str, str], lang: str, scope: Optional[str] = None, depth: int = 0) -> str:
        out = []
        for p in self.targets.get(lang, self.parts):
            if isinstance(p, str):
                out.append(p)
            else:
                value = groups.get(p[0]) or ''
                if self.kinds.get(p[0]) == 'msg' and depth < MAX_NESTING:
                    value = translate(value, lang, scope, depth + 1) or value
                out.append(value)
        return ''.join(out)

    def __repr__(self):
        return f"Entry({self.en!r})"


class _Index:
    """Lookup tables for some entries: the exact texts, and per first character
    of a message the templates that can match it, most literal text first (so
    "'f' expects 1 to 2 argument(s)" finds the "to" entry before the generic
    one; equal sizes keep the catalog order)."""

    def __init__(self, entries: List[Entry]):
        self.exact: Dict[str, Entry] = {}
        ordered: List[Entry] = []
        for e in entries:
            if e.kinds:
                ordered.append(e)
            else:
                self.exact.setdefault(e.render(''), e)         # the literal text, braces unescaped
        ordered.sort(key=lambda e: -e.literal_size)
        self.anywhere = [e for e in ordered if not e.prefix]   # templates that start with a placeholder
        self.by_first: Dict[str, List[Entry]] = {}
        for ch in sorted({e.prefix[0] for e in ordered if e.prefix}):
            self.by_first[ch] = [e for e in ordered if not e.prefix or e.prefix[0] == ch]

    def find(self, message: str, any_value: Optional[str]) -> Optional[Tuple[Entry, Dict[str, str]]]:
        e = self.exact.get(message)
        if e is not None:
            return e, {}
        for e in self.by_first.get(message[:1], self.anywhere):
            groups = e.match(message, any_value)
            if groups is not None:
                return e, groups
        return None


_ENTRIES: List[Entry] = []
_INDEXES: Dict[Optional[str], _Index] = {}


def _scope_lists() -> Tuple[Tuple[str, List[Tuple[str, str, str]]], ...]:
    return (('compiler', COMPILER_MESSAGES), ('runtime', RUNTIME_MESSAGES),
            ('command-line', COMMAND_LINE_MESSAGES))


def entries() -> List[Entry]:
    """All catalog entries (built on first use)."""
    if not _ENTRIES:
        built = [Entry(*row, scope=scope) for scope, rows in _scope_lists() for row in rows]
        indexes: Dict[Optional[str], _Index] = {None: _Index(built)}
        for scope in SCOPES:
            indexes[scope] = _Index([e for e in built if e.scope == scope])
        _INDEXES.update(indexes)
        _ENTRIES.extend(built)
    return _ENTRIES


def find(message: str, scope: Optional[str] = None,
         any_value: Optional[str] = None) -> Optional[Tuple[Entry, Dict[str, str]]]:
    """The catalog entry for an English message and its placeholder values.
    `scope` limits the lookup to one list: 'compiler', 'runtime' or 'command-line'."""
    entries()
    index = _INDEXES.get(scope)
    if index is None:
        raise ValueError(f"unknown scope {scope!r} (use: {', '.join(SCOPES)})")
    if len(message) > MAX_LOOKUP:
        return None
    return index.find(message, any_value)


def translate(message: str, lang: str, scope: Optional[str] = None, depth: int = 0) -> Optional[str]:
    """One message in `lang`, or None when the catalog has no entry for it."""
    if lang == 'en':
        return message
    if lang not in FRAME:
        return None
    hit = find(message, scope)
    if hit is None:
        return None
    entry, groups = hit
    return entry.apply(groups, lang, scope, depth)


# ═══════════════════════════════════════════════════════════════════════════
# whole diagnostics
# ═══════════════════════════════════════════════════════════════════════════
# The expressions below take linear time on any line (a diagnostic can quote
# program data): no two repetitions that can match the same characters follow
# each other, and the colour codes that end a line are split off by hand.

_ANSI_CODE = r'\x1b\[[0-9;]*m'
_ANSI = r'(?:' + _ANSI_CODE + r')*'
_GAP = r'(?:' + _ANSI_CODE + r'|[ \t])*'                # blanks and colour codes
_GUTTER_SPAN = 80       # the "  12 |" / " -->" of a source excerpt is within a line's first characters
_SOURCE_LINE = re.compile(r'^' + _GAP + r'(?:[0-9]+' + _GAP + r')?\|')
_LOCATION = re.compile(r'^' + _GAP + r'-->')
_SYNTAX = re.compile(r'^(?P<a0>' + _ANSI + r')Syntax Error: (?P<rest>.*)\Z', re.S)
_HEADER = re.compile(r'^(?P<a0>' + _ANSI + r')(?P<kind>error|warning|note|Error|Warning|Note)'
                     r'(?P<code>\[[^\]\s]*\])?: (?P<msg>.*)\Z', re.S)
_HINT = re.compile(r'^(?P<a0>' + _ANSI + r')(?P<lead>[ \t]*(?:= )?)(?P<kind>hint|note): (?P<msg>.*)\Z', re.S)
_REPORT = re.compile(r'^(?P<a0>' + _ANSI + r')(?P<kind>runtime error|fatal error): (?P<msg>.*)\Z', re.S)
_PROGRAM = re.compile(r'^(?P<a0>' + _ANSI + r')(?P<prog>tilvm|til): (?P<msg>.*)\Z', re.S)
# a line of `til` or one of its tools: "til: msg", "til pkg: error: msg", "til fit --run: msg"
# (at most three words after "til", none with a blank or a colon: linear time)
_TOOL = re.compile(r'^(?P<a0>' + _ANSI + r')(?P<prog>tilvm|til(?: [^\s:]{1,40}){0,3}): (?P<msg>.*)\Z', re.S)
_AT = re.compile(r'^(?P<indent>[ \t]+)at (?P<func>\S[^\n]*?)(?: \(line (?P<line>[0-9]+)\))?(?P<cr>\r?)$',
                 re.M)
_ANSI_CODE_RE = re.compile(_ANSI_CODE)


def _split_colour(text: str) -> Tuple[str, str]:
    """(text, the colour codes at its end)."""
    end = len(text)
    while True:
        k = text.rfind('\x1b[', 0, end)
        if k < 0 or _ANSI_CODE_RE.fullmatch(text, k, end) is None:
            return text[:end], text[end:]
        end = k


def _is_source(line: str) -> bool:
    return bool(_SOURCE_LINE.match(line, 0, _GUTTER_SPAN) or _LOCATION.match(line, 0, _GUTTER_SPAN))


def _message(text: str, lang: str, scope: Optional[str] = None) -> str:
    out = translate(text, lang, scope)
    return text if out is None else out


def _framed(m, word: str, lang: str, scope: Optional[str] = None) -> str:
    """A matched frame line: its colour, `word`, ': ' and the translated message."""
    msg, tail = _split_colour(m.group('msg'))
    return m.group('a0') + word + ': ' + _message(msg, lang, scope) + tail


def _report_line(m, lang: str) -> str:
    """'runtime error: msg' -> the frame word and msg, translated with RUNTIME_MESSAGES."""
    return _framed(m, FRAME[lang][m.group('kind')], lang, 'runtime')


def _at_line(m, lang: str) -> str:
    words = FRAME[lang]
    template = words['at_line'] if m.group('line') else words['at']
    return template.format(indent=m.group('indent'), func=m.group('func'), line=m.group('line')) + m.group('cr')


def split_frame(line: str) -> Tuple[Optional[str], str]:
    """(frame kind, message) of one diagnostic line: kind is 'header', 'hint',
    'report', 'program', 'at', 'source' or None for a bare line (then the
    message is the line)."""
    if _is_source(line):
        return 'source', ''
    syntax = _SYNTAX.match(line)
    if syntax:
        line = syntax.group('rest')
    for kind, rx in (('header', _HEADER), ('report', _REPORT), ('hint', _HINT), ('program', _TOOL)):
        m = rx.match(line)
        if m:
            msg = _split_colour(m.group('msg'))[0]
            if kind == 'program':                 # "til pkg: error: msg": the rest has a frame too
                inner, rest = split_frame(msg)
                if inner in ('header', 'hint', 'report'):
                    return inner, rest
            return kind, msg
    if syntax:
        return 'header', line
    if _AT.fullmatch(line):
        return 'at', ''
    return None, line


def localize_line(line: str, lang: str) -> str:
    """Translate one line of a compiler or command-line diagnostic (source
    excerpts are kept as they are)."""
    words = FRAME.get(lang)
    if words is None or not line.strip():
        return line
    if line.endswith('\r'):                       # Windows line ending
        return _localize_framed(line[:-1], lang, words, True) + '\r'
    return _localize_framed(line, lang, words, True)


def _localize_framed(line: str, lang: str, words: Dict[str, str], syntax_ok: bool) -> str:
    if _is_source(line):
        return line
    m = _SYNTAX.match(line) if syntax_ok else None
    if m:
        rest = m.group('rest')
        h = _HEADER.match(rest)
        if h and h.group('kind') == 'error' and not h.group('code'):
            # "Syntax Error: error: X" -> "Синтаксическая ошибка: X"
            return m.group('a0') + _framed(h, words['Syntax Error'], lang)
        return m.group('a0') + words['Syntax Error'] + ': ' + _localize_framed(rest, lang, words, False)
    m = _HEADER.match(line)
    if m:
        return _framed(m, words[m.group('kind')] + (m.group('code') or ''), lang)
    m = _REPORT.match(line)
    if m:
        return _report_line(m, lang)
    m = _HINT.match(line)
    if m:
        return _framed(m, m.group('lead') + words[m.group('kind')], lang)
    m = _TOOL.match(line)
    if m:
        # "til pkg: error: msg" -> "til pkg: ошибка: ..." (the program name stays)
        msg, tail = _split_colour(m.group('msg'))
        return m.group('a0') + m.group('prog') + ': ' + _localize_framed(msg, lang, words, False) + tail
    m = _AT.fullmatch(line)
    if m:
        return _at_line(m, lang)
    return _message(line, lang)


def localize(text: str, lang: str) -> str:
    """Translate a compiler or command-line diagnostic (any number of lines)
    into `lang` ('en' returns it unchanged). A VM's error report:
    localize_report()."""
    if lang not in FRAME or not text:
        return text
    return '\n'.join(localize_line(line, lang) for line in text.split('\n'))


class ReportLocalizer:
    """Translates what a VM writes to standard error, piece by piece as it
    arrives (feed whole lines; the last piece may lack its newline).

    Before the report, the runner's own lines ('tilvm: ...') are translated and
    any other line is copied. The report's first line ('runtime error: msg' /
    'fatal error: msg') is translated with RUNTIME_MESSAGES only. After it,
    traceback lines ('  at f (line 3)') are translated and every other line is
    copied unchanged: it is the rest of the program's own message, which is
    never looked up. The work is linear in the length of the text."""

    def __init__(self, lang: str):
        self.lang = lang if lang in FRAME else None
        self.in_report = False

    def feed(self, text: str) -> str:
        if self.lang is None or not text:
            return text
        out = []
        pos = 0
        while pos < len(text) and not self.in_report:
            nl = text.find('\n', pos)
            end = len(text) if nl < 0 else nl + 1
            out.append(self._before_report(text[pos:end]))
            pos = end
        if pos < len(text):
            out.append(_AT.sub(self._at, text[pos:]))
        return ''.join(out)

    def _before_report(self, line: str) -> str:
        body = line[:-1] if line.endswith('\n') else line
        end = line[len(body):]
        if body.endswith('\r'):
            body, end = body[:-1], '\r' + end
        m = _REPORT.match(body)
        if m:
            self.in_report = True
            return _report_line(m, self.lang) + end
        m = _PROGRAM.match(body)
        if m:
            return _framed(m, m.group('prog'), self.lang) + end
        return line

    def _at(self, m) -> str:
        return _at_line(m, self.lang)


def localize_report(text: str, lang: str) -> str:
    """Translate a VM's standard error at once (an uncaught-error report and the
    runner lines before it); see ReportLocalizer."""
    return ReportLocalizer(lang).feed(text)


# ═══════════════════════════════════════════════════════════════════════════
# self-check (used by tests/test_i18n.py)
# ═══════════════════════════════════════════════════════════════════════════

def check_catalog() -> List[str]:
    """Problems in the catalog: malformed templates, placeholders that differ
    between languages, duplicates, untranslated entries, frame gaps."""
    problems: List[str] = []
    seen: Dict[str, str] = {}
    lists = _scope_lists()
    if tuple(scope for scope, _ in lists) != SCOPES:
        problems.append(f"SCOPES {SCOPES} do not name the message lists")
    if MESSAGES != [row for _, rows in lists for row in rows]:
        problems.append("MESSAGES is not COMPILER_MESSAGES + RUNTIME_MESSAGES + COMMAND_LINE_MESSAGES "
                        "(add entries to one of those lists)")
    for scope, rows in lists:
        for i, row in enumerate(rows):
            where = f"{scope} entry {i} {row[0]!r}" if row else f"{scope} entry {i}"
            if len(row) != len(COLUMNS):
                problems.append(f"{where}: expected {len(COLUMNS)} texts ({', '.join(COLUMNS)})")
                continue
            if row[0] in seen:
                problems.append(f"{where}: duplicate of {seen[row[0]]}")
            seen[row[0]] = f"{scope} entry {i}"
            try:
                e = Entry(*row, scope=scope)
            except (ValueError, re.error) as err:
                problems.append(f"{where}: {err}")
                continue
            problems.extend(f"{where}: {p}" for p in _entry_problems(e, row))
    keys = None
    for lang, words in FRAME.items():
        if lang not in LANGUAGES:
            problems.append(f"FRAME language {lang!r} is not in LANGUAGES")
        if keys is None:
            keys = set(words)
        elif set(words) != keys:
            problems.append(f"FRAME[{lang!r}] keys differ: {sorted(set(words) ^ keys)}")
    for lang in LANGUAGES[1:]:
        if lang not in FRAME:
            problems.append(f"language {lang!r} has no FRAME words")
        if lang not in COLUMNS:
            problems.append(f"language {lang!r} has no MESSAGES column")
    return problems


def _entry_problems(e: Entry, row: Tuple[str, ...]) -> List[str]:
    problems = []
    names = {p[0] for p in e.parts if not isinstance(p, str)}
    for p in e.parts:
        if not isinstance(p, str) and p[1] and p[1] != e.kinds[p[0]]:
            problems.append(f"placeholder {p[0]} is used with kinds {e.kinds[p[0]]!r} and {p[1]!r}")
    if e.leading is not None and e.kinds[e.leading] in ('text', 'msg'):
        after = e.parts[1] if len(e.parts) > 1 and isinstance(e.parts[1], str) else ''
        # command-line messages often start with a file, URL or registry ("{path}: invalid
        # til.json (...)"); they never meet a program's text, so less text tells them apart
        least = 12 if e.scope == 'command-line' else 30
        if e.scope == 'runtime':
            problems.append(f"it starts with the free-text placeholder {e.leading}: a program's own "
                            f"sentence with that ending would match")
        elif len(after) < 3 or e.literal_size < least:
            problems.append(f"it starts with the free-text placeholder {e.leading}, followed by too "
                            f"little text to tell it apart")
    for lang in COLUMNS[1:]:
        text = e.texts[lang]
        if not text.strip():
            problems.append(f"empty {lang} text")
        got = {p[0] for p in e.targets[lang] if not isinstance(p, str)}
        if got != names:
            problems.append(f"{lang} placeholders {sorted(got)} != {sorted(names)}")
        for p in e.targets[lang]:
            if not isinstance(p, str) and p[1] and p[1] != e.kinds.get(p[0]):
                problems.append(f"{lang} placeholder {p[0]} has kind {p[1]}")
        if text == row[0] and re.search(r'[A-Za-z]{4}', row[0]):
            problems.append(f"{lang} text is the English text")
    return problems
