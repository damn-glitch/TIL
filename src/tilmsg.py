#!/usr/bin/env python3
"""
TIL message catalog (TIL 3.1): compiler and VM diagnostics in English,
Russian and Kazakh.

The compiler (src/til.py) and both VMs write their diagnostics in English.
The `til` command translates them at output time:

    import tilmsg
    tilmsg.localize(text, 'ru')        # a whole diagnostic, any number of lines
    tilmsg.translate(message, 'kk')    # one message, or None if it is not in the catalog

Only reports are translated. Values that programs see (the `error` of a
retry/timeout `else` branch, a panic message caught by `retry`) stay English.

A catalog entry is an English template with its Russian and Kazakh versions.
Placeholders: `{name}` any text, `{name:int}` an integer, `{name:msg}` a nested
message that is translated too; `{{` and `}}` are literal braces. Text inside
placeholders (identifiers, numbers, code, quotes) is copied unchanged. A
message without an entry stays English, but the frame around it (`error[E-VM]:`,
`= hint:`, `runtime error:`, `  at f (line 3)`) is still translated.

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
# MESSAGES: (English, Russian, Kazakh)
# Conventions: translations start with a lowercase letter (they follow
# "ошибка: ..." / "қате: ..."); Russian avoids number agreement with
# placeholders ("аргументов: 2"); Kazakh never attaches a suffix to a
# placeholder (the suffix goes on the following noun: "'f' функциясы").
# ═══════════════════════════════════════════════════════════════════════════

MESSAGES: List[Tuple[str, str, str]] = [
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
    ("function '{func}' performs effect '{effect}', but declares only #[effects: {allowed}]",
     "функция '{func}' выполняет эффект '{effect}', но объявляет только #[effects: {allowed}]",
     "'{func}' функциясы '{effect}' әсерін тудырады, бірақ тек #[effects: {allowed}] жарияланған"),
    ("function '{func}' performs effect '{effect}' (through the call to '{via}'), "
     "but declares only #[effects: {allowed}]",
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
    ("contract violation: call to '{func}' ({args}) breaks #[requires: {cond}]",
     "нарушение контракта: вызов '{func}' ({args}) не удовлетворяет #[requires: {cond}]",
     "келісімшарт бұзылды: '{func}' шақыруы ({args}) #[requires: {cond}] шартын бұзады"),
    ("proved false at compile time; fix the arguments",
     "на этапе компиляции доказано, что условие ложно; исправьте аргументы",
     "шарттың жалған екені компиляция кезінде дәлелденді; аргументтерді түзетіңіз"),

    # ── effects with resumption: handle / with / resume (TIL 3.1) ──────────
    # (written against the til31/effects branch, so its diagnostics are
    # translated as soon as it is merged)
    ("unhandled effect '{effect}'",
     "необработанный эффект '{effect}'",
     "өңделмеген әсер: '{effect}'"),
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
     "move this code before resume",
     "resume сразу завершает ветвь и продолжает обрабатываемое тело (хвостовое возобновление); "
     "перенесите этот код до resume",
     "resume тармақты бірден аяқтап, өңделетін денені жалғастырады (соңғы жалғастыру); "
     "бұл кодты resume алдына жылжытыңыз"),
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
    ("effect '{effect}' expects {count} argument(s), got {got}",
     "эффект '{effect}' ожидает аргументов: {count}, передано: {got}",
     "'{effect}' әсері {count} аргумент күтеді, бірақ {got} берілді"),
    ("the clause of the enclosing handle takes a different number of parameters",
     "ветвь объемлющего handle принимает другое число параметров",
     "сыртқы handle тармағы параметрлердің басқа санын қабылдайды"),
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
    ("unknown language '{lang}' (use: {choices})",
     "неизвестный язык '{lang}' (доступны: {choices})",
     "белгісіз тіл: '{lang}' (мүмкін мәндер: {choices})"),

    # ── type checker warnings of the native backend ────────────────────────
    ("Line {line:int}: {message:msg}",
     "строка {line}: {message}",
     "{line}-жол: {message}"),
    ("Type mismatch: expected {expected}, got {got}",
     "несоответствие типов: ожидался {expected}, получен {got}",
     "типтер сәйкес емес: {expected} күтілді, бірақ {got} алынды"),
    ("Cannot assign to immutable variable '{name}'",
     "нельзя присвоить значение неизменяемой переменной '{name}'",
     "өзгермейтін '{name}' айнымалысына мән меншіктеуге болмайды"),
    ("Condition must be bool, got {got}",
     "условие должно иметь тип bool, получен {got}",
     "шарт bool типті болуы керек, бірақ {got} алынды"),

    # ── runtime errors (reference VM and C VM) ─────────────────────────────
    ("division by zero",
     "деление на ноль",
     "нөлге бөлу"),
    ("{what} expects a number, got {type}",
     "{what} ожидает число, получено {type}",
     "{what} сан күтеді, бірақ {type} берілді"),
    ("unsupported operand types for {op}: {left} and {right}",
     "неподдерживаемые типы операндов для {op}: {left} и {right}",
     "{op} операциясы үшін операнд типтеріне қолдау жоқ: {left} және {right}"),
    ("cannot compare {left} and {right}",
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
    ("cannot convert {type} to int",
     "нельзя преобразовать {type} в int",
     "{type} типін int типіне түрлендіруге болмайды"),
    ("cannot convert {type} to float",
     "нельзя преобразовать {type} в float",
     "{type} типін float типіне түрлендіруге болмайды"),
    ("this TIL VM was built without floating point",
     "эта TIL VM собрана без поддержки чисел с плавающей точкой",
     "бұл TIL VM жылжымалы нүктелі сандарсыз құрастырылған"),
    ("index must be an integer",
     "индекс должен быть целым числом",
     "индекс бүтін сан болуы керек"),
    ("index out of bounds: index {index}, length {length}",
     "индекс вне границ: индекс {index}, длина {length}",
     "индекс шектен тыс: индекс {index}, ұзындық {length}"),
    ("key not found: {key}",
     "ключ не найден: {key}",
     "кілт табылмады: {key}"),
    ("{type} cannot be indexed",
     "{type} не поддерживает индексацию",
     "{type} типін индекстеуге болмайды"),
    ("{type} cannot be a map key",
     "{type} не может быть ключом словаря",
     "{type} сөздік кілті бола алмайды"),
    ("strings are immutable",
     "строки неизменяемы",
     "жолдарды өзгертуге болмайды"),
    ("{type} does not support item assignment",
     "{type} не поддерживает присваивание элементов",
     "{type} элементтеріне мән меншіктеуге болмайды"),
    ("no field '{field}' on {type}",
     "у {type} нет поля '{field}'",
     "{type} ішінде '{field}' өрісі жоқ"),
    ("{name}() is missing arguments",
     "{name}(): не хватает аргументов",
     "{name}() үшін аргументтер жетіспейді"),
    ("value is not iterable",
     "значение не является итерируемым",
     "бұл мән бойынша итерация жасауға болмайды"),
    ("{type} is not iterable",
     "{type} не является итерируемым",
     "{type} мәні бойынша итерация жасауға болмайды"),
    ("len() of {type}",
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
    ("cannot read file: {path}",
     "не удаётся прочитать файл: {path}",
     "файлды оқу мүмкін емес: {path}"),
    ("cannot write file: {path}",
     "не удаётся записать файл: {path}",
     "файлға жазу мүмкін емес: {path}"),
    ("file too large: {path}",
     "файл слишком большой: {path}",
     "файл тым үлкен: {path}"),
    ("{name}() expects a path string, got {type}",
     "{name}() ожидает путь к файлу (строку), получено {type}",
     "{name}() файл жолын (str) күтеді, бірақ {type} берілді"),
    ("{name}(): invalid path",
     "{name}(): недопустимый путь",
     "{name}(): файл жолы жарамсыз"),
    ("bytes_to_str() expects an array of ints 0..255",
     "bytes_to_str() ожидает массив целых чисел 0..255",
     "bytes_to_str() 0..255 аралығындағы бүтін сандар массивін күтеді"),
    ("str_bytes() expects a string, got {type}",
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
    ("insert index out of bounds: {index}",
     "индекс вставки вне границ: {index}",
     "кірістіру индексі шектен тыс: {index}"),
    ("remove index out of bounds: {index}",
     "индекс удаления вне границ: {index}",
     "жою индексі шектен тыс: {index}"),
    ("sort() needs an array of numbers or of strings",
     "sort() требует массив чисел или строк",
     "sort() үшін сандар немесе жолдар массиві керек"),
    ("unwrap() called on None",
     "unwrap() вызван для None",
     "unwrap() None мәніне шақырылды"),
    ("unwrap() called on Err: {error}",
     "unwrap() вызван для Err: {error}",
     "unwrap() Err мәніне шақырылды: {error}"),
    ("expect() failed",
     "expect() завершился неудачей",
     "expect() сәтсіз аяқталды"),
    ("no method '{method}' on {type}",
     "у {type} нет метода '{method}'",
     "{type} типінде '{method}' әдісі жоқ"),
    ("field '{field}' is not callable",
     "поле '{field}' нельзя вызвать",
     "'{field}' өрісін шақыруға болмайды"),
    ("{type} is not callable",
     "значение типа {type} нельзя вызвать",
     "{type} типті мәнді шақыруға болмайды"),
    ("{func}() takes {count} argument(s), got {got}",
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
    ("cannot negate {type}",
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
    ("? applied to Err: {error}",
     "оператор ? применён к Err: {error}",
     "? операторы Err мәніне қолданылды: {error}"),
    ("step limit exceeded ({limit})",
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
    ("bad opcode 0x{op} at {pc}",
     "неверный код операции 0x{op} по адресу {pc}",
     "{pc} адресінде қате операция коды: 0x{op}"),
    ("bad opcode",
     "неверный код операции",
     "қате операция коды"),

    # ── runtime errors compiled into programs (asserts, contracts, intents) ─
    ("Assertion failed",
     "проверка не пройдена",
     "тексеру сәтсіз аяқталды"),
    ("Assertion failed: {message}",
     "проверка не пройдена: {message}",
     "тексеру сәтсіз аяқталды: {message}"),
    ("Invariant violated",
     "инвариант нарушен",
     "инвариант бұзылды"),
    ("Invariant violated: {message}",
     "инвариант нарушен: {message}",
     "инвариант бұзылды: {message}"),
    ("contract violation: {func} requires {cond}",
     "нарушение контракта: {func} требует {cond}",
     "келісімшарт бұзылды: {func} талабы орындалмады: {cond}"),
    ("contract violation: {func} ensures {cond}",
     "нарушение контракта: {func} гарантирует {cond}",
     "келісімшарт бұзылды: {func} кепілдігі орындалмады: {cond}"),
    ("intent '{intent}' failed: {error:msg}",
     "намерение '{intent}' не выполнено: {error}",
     "'{intent}' ниеті орындалмады: {error}"),

    # ── command line ───────────────────────────────────────────────────────
    ("No input file specified",
     "не указан входной файл",
     "кіріс файлы көрсетілмеген"),
    ("File not found: {path}",
     "файл не найден: {path}",
     "файл табылмады: {path}"),
    ("unknown option {option}",
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
    ("message translations are not installed (tilmsg.py is missing); messages stay in English",
     "переводы сообщений не установлены (нет tilmsg.py); сообщения выводятся на английском",
     "хабарлама аудармалары орнатылмаған (tilmsg.py жоқ); хабарламалар ағылшын тілінде шығады"),
]

# English-only on purpose (checked by tools/i18n_coverage.py).
UNTRANSLATED: Tuple[str, ...] = (
    "panic",            # the default message of panic(): the name of the builtin
    # printed only when TIL_LANG names no known language, i.e. in English anyway
    "TIL_LANG={value} is not a message language (en, ru, kk); using English",
)

# ═══════════════════════════════════════════════════════════════════════════
# templates
# ═══════════════════════════════════════════════════════════════════════════

Part = Union[str, Tuple[str, str]]          # literal text, or (placeholder name, kind)

_TOKEN = re.compile(r'\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)(?::([a-z]+))?\}|[{}]')
_KIND_PATTERN = {'': r'.*?', 'int': r'-?\d+', 'msg': r'.*?'}


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
            if kind not in _KIND_PATTERN:
                raise ValueError(f"unknown placeholder kind '{kind}' in {text!r}")
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


class Entry:
    """A catalog entry: the English template, its regex and its translations."""

    def __init__(self, en: str, *translations: str):
        self.en = en
        self.parts = parse_template(en)
        self.kinds: Dict[str, str] = {}
        regex = []
        for p in self.parts:
            if isinstance(p, str):
                regex.append(re.escape(p))
            elif p[0] in self.kinds:
                regex.append(f"(?P={p[0]})")
            else:
                self.kinds[p[0]] = p[1]
                regex.append(f"(?P<{p[0]}>{_KIND_PATTERN[p[1]]})")
        self.pattern = re.compile(''.join(regex), re.S)
        self.literal_size = sum(len(p) for p in self.parts if isinstance(p, str))
        first = self.parts[0] if self.parts else ''
        self.leading = None if isinstance(first, str) else first[0]
        self.texts: Dict[str, str] = {'en': en}
        self.targets: Dict[str, List[Part]] = {'en': self.parts}
        for lang, text in zip(COLUMNS[1:], translations):
            self.texts[lang] = text
            self.targets[lang] = parse_template(text)

    def render(self, mark: str) -> str:
        """The English text with every placeholder replaced by `mark`."""
        return ''.join(p if isinstance(p, str) else mark for p in self.parts)

    def apply(self, groups: Dict[str, str], lang: str) -> str:
        out = []
        for p in self.targets.get(lang, self.parts):
            if isinstance(p, str):
                out.append(p)
            else:
                value = groups.get(p[0]) or ''
                if self.kinds.get(p[0]) == 'msg':
                    value = translate(value, lang) or value
                out.append(value)
        return ''.join(out)

    def __repr__(self):
        return f"Entry({self.en!r})"


_ENTRIES: List[Entry] = []
_EXACT: Dict[str, Entry] = {}
_ORDERED: List[Entry] = []


def entries() -> List[Entry]:
    """All catalog entries (built on first use)."""
    if not _ENTRIES:
        for row in MESSAGES:
            _ENTRIES.append(Entry(*row))
        for e in _ENTRIES:
            if not e.kinds:
                _EXACT.setdefault(e.render(''), e)          # the literal text, braces unescaped
            else:
                _ORDERED.append(e)
        # the most specific template wins: more literal text first (stable order)
        _ORDERED.sort(key=lambda e: -e.literal_size)
    return _ENTRIES


def find(message: str) -> Optional[Tuple[Entry, Dict[str, str]]]:
    """The catalog entry for an English message and its placeholder values."""
    entries()
    e = _EXACT.get(message)
    if e is not None:
        return e, {}
    for e in _ORDERED:
        m = e.pattern.fullmatch(message)
        if m is None:
            continue
        groups = m.groupdict()
        # a template that starts with a placeholder ("{type} is not callable")
        # only matches when that placeholder is a single word: it must not
        # swallow arbitrary text in front of a familiar ending
        if e.leading is not None and re.search(r'\s', groups.get(e.leading) or ''):
            continue
        return e, groups
    return None


def translate(message: str, lang: str) -> Optional[str]:
    """One message in `lang`, or None when the catalog has no entry for it."""
    if lang == 'en':
        return message
    if lang not in FRAME:
        return None
    hit = find(message)
    if hit is None:
        return None
    entry, groups = hit
    return entry.apply(groups, lang)


# ═══════════════════════════════════════════════════════════════════════════
# whole diagnostics
# ═══════════════════════════════════════════════════════════════════════════

_ANSI = r'(?:\x1b\[[0-9;]*m)*'
_SOURCE_LINE = re.compile(r'^' + _ANSI + r'[ \t]*' + _ANSI + r'[ \t]*\d*' + _ANSI + r'[ \t]*\|')
_LOCATION = re.compile(r'^' + _ANSI + r'[ \t]*-->')
_SYNTAX = re.compile(r'^(?P<a0>' + _ANSI + r')Syntax Error: (?P<rest>.*)$', re.S)
_HEADER = re.compile(r'^(?P<a0>' + _ANSI + r')(?P<kind>error|warning|note|Error|Warning|Note)'
                     r'(?P<code>\[[^\]\s]*\])?: (?P<msg>.*?)(?P<a1>' + _ANSI + r')$', re.S)
_HINT = re.compile(r'^(?P<a0>' + _ANSI + r')(?P<lead>[ \t]*(?:= )?)(?P<kind>hint|note): '
                   r'(?P<msg>.*?)(?P<a1>' + _ANSI + r')$', re.S)
_REPORT = re.compile(r'^(?P<a0>' + _ANSI + r')(?P<kind>runtime error|fatal error): '
                     r'(?P<msg>.*?)(?P<a1>' + _ANSI + r')$', re.S)
_AT = re.compile(r'^(?P<indent>[ \t]+)at (?P<func>\S.*?)(?: \(line (?P<line>\d+)\))?$', re.S)


def _message(text: str, lang: str) -> str:
    out = translate(text, lang)
    return text if out is None else out


def split_frame(line: str) -> Tuple[Optional[str], str]:
    """(frame kind, message) of one diagnostic line: kind is 'header', 'hint',
    'report', 'at', 'source' or None for a bare line (then the message is the line)."""
    if _SOURCE_LINE.match(line) or _LOCATION.match(line):
        return 'source', ''
    syntax = _SYNTAX.match(line)
    if syntax:
        line = syntax.group('rest')
    for kind, rx in (('header', _HEADER), ('report', _REPORT), ('hint', _HINT)):
        m = rx.match(line)
        if m:
            return kind, m.group('msg')
    if syntax:
        return 'header', line
    if _AT.match(line):
        return 'at', ''
    return None, line


def localize_line(line: str, lang: str) -> str:
    """Translate one line of a diagnostic (source excerpts are kept as they are)."""
    words = FRAME.get(lang)
    if words is None or not line.strip():
        return line
    if line.endswith('\r'):
        return localize_line(line[:-1], lang) + '\r'
    if _SOURCE_LINE.match(line) or _LOCATION.match(line):
        return line
    m = _SYNTAX.match(line)
    if m:
        rest = m.group('rest')
        h = _HEADER.match(rest)
        if h and h.group('kind') == 'error' and not h.group('code'):
            # "Syntax Error: error: X" -> "Синтаксическая ошибка: X"
            return (m.group('a0') + h.group('a0') + words['Syntax Error'] + ': ' +
                    _message(h.group('msg'), lang) + h.group('a1'))
        return m.group('a0') + words['Syntax Error'] + ': ' + localize_line(rest, lang)
    m = _HEADER.match(line)
    if m:
        return (m.group('a0') + words[m.group('kind')] + (m.group('code') or '') + ': ' +
                _message(m.group('msg'), lang) + m.group('a1'))
    m = _REPORT.match(line)
    if m:
        return m.group('a0') + words[m.group('kind')] + ': ' + _message(m.group('msg'), lang) + m.group('a1')
    m = _HINT.match(line)
    if m:
        return (m.group('a0') + m.group('lead') + words[m.group('kind')] + ': ' +
                _message(m.group('msg'), lang) + m.group('a1'))
    m = _AT.match(line)
    if m:
        template = words['at_line'] if m.group('line') else words['at']
        return template.format(indent=m.group('indent'), func=m.group('func'), line=m.group('line'))
    return _message(line, lang)


def localize(text: str, lang: str) -> str:
    """Translate a diagnostic (one or more lines) into `lang` ('en' returns it unchanged)."""
    if lang not in FRAME or not text:
        return text
    return '\n'.join(localize_line(line, lang) for line in text.split('\n'))


# ═══════════════════════════════════════════════════════════════════════════
# self-check (used by tests/test_i18n.py)
# ═══════════════════════════════════════════════════════════════════════════

def check_catalog() -> List[str]:
    """Problems in the catalog: malformed templates, placeholders that differ
    between languages, duplicates, untranslated entries, frame gaps."""
    problems: List[str] = []
    seen: Dict[str, int] = {}
    for i, row in enumerate(MESSAGES):
        if len(row) != len(COLUMNS):
            problems.append(f"entry {i} {row[0]!r}: expected {len(COLUMNS)} texts ({', '.join(COLUMNS)})")
            continue
        if row[0] in seen:
            problems.append(f"entry {i} {row[0]!r}: duplicate of entry {seen[row[0]]}")
        seen[row[0]] = i
        try:
            e = Entry(*row)
        except (ValueError, re.error) as err:
            problems.append(f"entry {i} {row[0]!r}: {err}")
            continue
        names = {p[0] for p in e.parts if not isinstance(p, str)}
        for lang in COLUMNS[1:]:
            text = e.texts[lang]
            if not text.strip():
                problems.append(f"entry {i} {row[0]!r}: empty {lang} text")
            got = {p[0] for p in e.targets[lang] if not isinstance(p, str)}
            if got != names:
                problems.append(f"entry {i} {row[0]!r}: {lang} placeholders {sorted(got)} != {sorted(names)}")
            for p in e.targets[lang]:
                if not isinstance(p, str) and p[1] and p[1] != e.kinds.get(p[0]):
                    problems.append(f"entry {i} {row[0]!r}: {lang} placeholder {p[0]} has kind {p[1]}")
            if text == row[0] and re.search(r'[A-Za-z]{4}', row[0]):
                problems.append(f"entry {i} {row[0]!r}: {lang} text is the English text")
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
