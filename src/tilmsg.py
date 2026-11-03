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

    # ── contract proofs (E-PROVE; written against the til31/prove branch) ──
    ("cannot prove '{func}': unsupported: {reason:text}",
     "нельзя доказать '{func}': не поддерживается: {reason}",
     "'{func}' функциясын дәлелдеу мүмкін емес: қолдау жоқ: {reason}"),
    ("the prover models int and bool values, if/elif/else, loops with `invariant`, and calls; "
     "see docs/TIL_PROOFS.md (til prove --allow-unsupported skips such functions)",
     "модуль доказательства моделирует значения int и bool, if/elif/else, циклы с `invariant` и вызовы; "
     "см. docs/TIL_PROOFS.md (til prove --allow-unsupported пропускает такие функции)",
     "дәлелдеу модулі int және bool мәндерін, if/elif/else, `invariant` бар циклдерді және шақыруларды "
     "модельдейді; docs/TIL_PROOFS.md қараңыз (til prove --allow-unsupported мұндай функцияларды "
     "өткізіп жібереді)"),
    ("'{func}': {what:text} fails for {inputs:text}",
     "'{func}': {what} нарушается при {inputs}",
     "'{func}': {inputs} кезінде {what} орындалмайды"),
    ("'{func}': cannot prove {what:text}",
     "'{func}': не удаётся доказать {what}",
     "'{func}': {what} дәлелденбеді"),
    ("'{func}': could not prove {what:text} (unknown: timeout, {ms:int} ms per query)",
     "'{func}': не удалось доказать {what} (неизвестно: истекло время, {ms} мс на запрос)",
     "'{func}': {what} дәлелденбеді (белгісіз: уақыт бітті, бір сұрауға {ms} мс)"),
    ("'{func}': could not prove {what:text} (unknown: {why:text}, {ms:int} ms per query)",
     "'{func}': не удалось доказать {what} (неизвестно: {why}, {ms} мс на запрос)",
     "'{func}': {what} дәлелденбеді (белгісіз: {why}, бір сұрауға {ms} мс)"),

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
    ("cannot find the TIL C VM sources (vm/tilvm.c). Set TIL_HOME to the TIL directory.",
     "не найдены исходники C VM для TIL (vm/tilvm.c). Укажите каталог TIL в TIL_HOME.",
     "TIL C VM бастапқы кодтары табылмады (vm/tilvm.c). TIL_HOME айнымалысында TIL каталогын көрсетіңіз."),
    ("unknown language '{lang}' (use: {choices:text})",
     "неизвестный язык '{lang}' (доступны: {choices})",
     "белгісіз тіл: '{lang}' (мүмкін мәндер: {choices})"),

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
    ("[TIL] No C compiler found - running on the TIL VM",
     "[TIL] компилятор C не найден — программа запускается на TIL VM",
     "[TIL] C компиляторы табылмады — бағдарлама TIL VM-де іске қосылады"),
    ("[TIL] native backend cannot compile this program; running it on the TIL VM",
     "[TIL] нативный бэкенд не может скомпилировать эту программу; она запускается на TIL VM",
     "[TIL] нативті бэкенд бұл бағдарламаны компиляциялай алмайды; ол TIL VM-де іске қосылады"),
    ("til {command}: not available in this installation ({module}.py is missing)",
     "til {command}: недоступно в этой установке (нет файла {module}.py)",
     "til {command}: бұл орнатылымда қолжетімсіз ({module}.py файлы жоқ)"),
    ("til check --prove: the prover (tilprove.py) is not installed",
     "til check --prove: модуль доказательства (tilprove.py) не установлен",
     "til check --prove: дәлелдеу модулі (tilprove.py) орнатылмаған"),
]

# Every entry, in lookup order (read-only: add entries to the lists above).
MESSAGES: List[Tuple[str, str, str]] = COMPILER_MESSAGES + RUNTIME_MESSAGES + COMMAND_LINE_MESSAGES
SCOPES: Tuple[str, ...] = ('compiler', 'runtime', 'command-line')

# English-only on purpose (checked by tools/i18n_coverage.py).
UNTRANSLATED: Tuple[str, ...] = (
    "panic",            # the default message of panic(): the name of the builtin
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
#              map, fn, struct, or a capitalized name (Point, Shape.Circle, Нүкте)
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
        return value in BUILTIN_TYPES or (_TYPE_NAME.fullmatch(value) is not None and value[0].isupper())
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
    for kind, rx in (('header', _HEADER), ('report', _REPORT), ('hint', _HINT), ('program', _PROGRAM)):
        m = rx.match(line)
        if m:
            return kind, _split_colour(m.group('msg'))[0]
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
    m = _PROGRAM.match(line)
    if m:
        return _framed(m, m.group('prog'), lang)
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
        if e.scope == 'runtime':
            problems.append(f"it starts with the free-text placeholder {e.leading}: a program's own "
                            f"sentence with that ending would match")
        elif len(after) < 3 or e.literal_size < 30:
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
