#!/usr/bin/env python3
"""
TIL Compiler Unit Tests
Tests for Lexer, Parser, TypeChecker, and CCodeGenerator.
Run: python3 -m pytest tests/test_compiler.py -v
"""

import sys
import os
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from til import (
    Lexer, TokenType, Parser, TypeChecker, CCodeGenerator, TILCompiler,
    IntLit, FloatLit, StringLit, BoolLit, Identifier, BinaryOp, UnaryOp,
    Call, Attribute, Cast, ArrayLit, Range, StructInit, MatchExpr,
    Block, VarDecl, Assignment, If, For, While, Loop, Return, Break, Continue,
    ExprStmt, FuncDef, StructDef, EnumDef, ImplBlock, Import, Program,
    Lambda, IfExpr, ListComprehension, NullCheck, DictLit,
    ErrorReporter, get_hint_for_error, ModuleResolver, resolve_imports,
    PrimitiveType, ArrayType, StructType, OptionType, ResultType,
    T_INT, T_FLOAT, T_STR, T_BOOL, T_VOID, T_UNKNOWN,
    KAZAKH_KEYWORDS,
)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def lex(src: str):
    return Lexer(src, "<test>").tokenize()

def parse(src: str):
    tokens = lex(src)
    return Parser(tokens, "<test>").parse()

def compile_to_c(src: str) -> str:
    compiler = TILCompiler()
    compiler.check_types = False
    return compiler.compile(src, "<test>")

def compile_and_run(src: str) -> str:
    """Compile TIL source to executable, run it, return stdout."""
    compiler = TILCompiler()
    compiler.check_types = False
    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as f:
        exe = f.name
    try:
        ok = compiler.compile_to_executable(src, exe, "<test>")
        assert ok, "Compilation failed"
        result = subprocess.run([exe], capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    finally:
        try:
            os.unlink(exe)
        except:
            pass


# ═══════════════════════════════════════════════════════════════
# LEXER TESTS
# ═══════════════════════════════════════════════════════════════

class TestLexer:
    def test_integer_literals(self):
        tokens = lex("42 0 100")
        ints = [t for t in tokens if t.type == TokenType.INT]
        assert len(ints) == 3
        assert ints[0].value == 42
        assert ints[1].value == 0
        assert ints[2].value == 100

    def test_float_literals(self):
        tokens = lex("3.14 0.5")
        floats = [t for t in tokens if t.type == TokenType.FLOAT]
        assert len(floats) == 2
        assert floats[0].value == 3.14

    def test_string_literals(self):
        tokens = lex('"hello" "world"')
        strs = [t for t in tokens if t.type == TokenType.STRING]
        assert len(strs) == 2
        assert strs[0].value == "hello"

    def test_keywords(self):
        tokens = lex("if else elif for while loop in return break continue")
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert TokenType.IF in types
        assert TokenType.ELSE in types
        assert TokenType.FOR in types
        assert TokenType.WHILE in types
        assert TokenType.RETURN in types

    def test_identifiers(self):
        tokens = lex("foo bar_baz x1")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        assert len(idents) == 3
        assert idents[0].value == "foo"

    def test_operators(self):
        tokens = lex("+ - * / % ** == != <= >=")
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert TokenType.PLUS in types
        assert TokenType.POWER in types
        assert TokenType.EQEQ in types

    def test_indentation(self):
        tokens = lex("if true\n    x = 1\n")
        types = [t.type for t in tokens]
        assert TokenType.INDENT in types
        assert TokenType.DEDENT in types

    def test_attribute(self):
        tokens = lex("#[level: 0]\nfoo()")
        attrs = [t for t in tokens if t.type == TokenType.ATTRIBUTE]
        assert len(attrs) == 1
        assert "level" in attrs[0].value

    def test_paren_multiline(self):
        """Newlines inside parens should be suppressed."""
        tokens = lex("foo(\n    1,\n    2\n)")
        # Should NOT have INDENT/DEDENT/NEWLINE between ( and )
        types = [t.type for t in tokens]
        # After LPAREN, there should be no NEWLINE before RPAREN
        in_paren = False
        for t in tokens:
            if t.type == TokenType.LPAREN:
                in_paren = True
            elif t.type == TokenType.RPAREN:
                in_paren = False
            elif in_paren:
                assert t.type not in (TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT), \
                    f"Got {t.type} inside parens"

    def test_range_operators(self):
        tokens = lex("1..10 1..=10")
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert TokenType.RANGE in types
        assert TokenType.RANGE_INCL in types


# ═══════════════════════════════════════════════════════════════
# PARSER TESTS
# ═══════════════════════════════════════════════════════════════

class TestParser:
    def test_hello_world(self):
        prog = parse('main()\n    print("Hello")\n')
        assert isinstance(prog, Program)
        assert len(prog.statements) == 1
        assert isinstance(prog.statements[0], FuncDef)
        assert prog.statements[0].name == "main"

    def test_variable_let(self):
        prog = parse('main()\n    let x = 42\n')
        func = prog.statements[0]
        body = func.body.statements[0]
        assert isinstance(body, VarDecl)
        assert body.name == "x"
        assert body.mutable == False

    def test_variable_var(self):
        prog = parse('main()\n    var x = 0\n')
        body = prog.statements[0].body.statements[0]
        assert isinstance(body, VarDecl)
        assert body.mutable == True

    def test_if_elif_else(self):
        prog = parse('main()\n    if x > 0\n        print("pos")\n    elif x == 0\n        print("zero")\n    else\n        print("neg")\n')
        stmt = prog.statements[0].body.statements[0]
        assert isinstance(stmt, If)
        assert len(stmt.elifs) == 1
        assert stmt.else_body is not None

    def test_for_range(self):
        prog = parse('main()\n    for i in 0..10\n        print(i)\n')
        stmt = prog.statements[0].body.statements[0]
        assert isinstance(stmt, For)
        assert stmt.var == "i"
        assert isinstance(stmt.iter, Range)

    def test_struct_definition(self):
        prog = parse('struct Point\n    x: float\n    y: float\n')
        assert isinstance(prog.statements[0], StructDef)
        assert prog.statements[0].name == "Point"
        assert len(prog.statements[0].fields) == 2

    def test_enum_basic(self):
        prog = parse('enum Color\n    Red\n    Green\n    Blue\n')
        assert isinstance(prog.statements[0], EnumDef)
        assert len(prog.statements[0].variants) == 3

    def test_enum_with_values(self):
        """Bug 1.2: Enum with = value should parse."""
        prog = parse('enum Dir\n    North = 0\n    East = 1\n')
        enum = prog.statements[0]
        assert isinstance(enum, EnumDef)
        assert enum.variants[0].value is not None
        assert isinstance(enum.variants[0].value, IntLit)
        assert enum.variants[0].value.value == 0

    def test_as_cast(self):
        """Bug 1.3: 'as' type cast should parse."""
        prog = parse('main()\n    let x = 10 as float\n')
        decl = prog.statements[0].body.statements[0]
        assert isinstance(decl.value, Cast)
        assert isinstance(decl.value.target_type, PrimitiveType)
        assert decl.value.target_type.name == "float"

    def test_self_in_expression(self):
        """Bug 1.1: 'self' should be usable in expressions."""
        prog = parse('struct Foo\n    x: int\nimpl Foo\n    get_x(self) -> int\n        return self.x\n')
        impl = prog.statements[1]
        assert isinstance(impl, ImplBlock)
        method = impl.methods[0]
        ret_stmt = method.body.statements[0]
        assert isinstance(ret_stmt, Return)
        assert isinstance(ret_stmt.value, Attribute)
        assert isinstance(ret_stmt.value.obj, Identifier)
        assert ret_stmt.value.obj.name == "self"

    def test_impl_block(self):
        prog = parse('struct Foo\n    x: int\nimpl Foo\n    new(x: int) -> Foo\n        return Foo { x: x }\n')
        impl = prog.statements[1]
        assert isinstance(impl, ImplBlock)
        assert impl.type_name == "Foo"
        assert len(impl.methods) == 1

    def test_match_expression(self):
        prog = parse('main()\n    match x\n        1 => print("one")\n        2 => print("two")\n')
        stmt = prog.statements[0].body.statements[0]
        assert isinstance(stmt, MatchExpr)
        assert len(stmt.arms) == 2

    def test_while_loop(self):
        prog = parse('main()\n    while x > 0\n        x -= 1\n')
        stmt = prog.statements[0].body.statements[0]
        assert isinstance(stmt, While)

    def test_loop_break(self):
        prog = parse('main()\n    loop\n        break\n')
        stmt = prog.statements[0].body.statements[0]
        assert isinstance(stmt, Loop)

    def test_array_type_syntax(self):
        """type[] syntax for array parameters."""
        prog = parse('foo(arr: int[]) -> int\n    return 0\n')
        param = prog.statements[0].params[0]
        assert isinstance(param.type, ArrayType)

    def test_multiline_call(self):
        """Multi-line function calls inside parens."""
        prog = parse('main()\n    let x = foo(\n        1,\n        2\n    )\n')
        decl = prog.statements[0].body.statements[0]
        assert isinstance(decl.value, Call)
        assert len(decl.value.args) == 2


# ═══════════════════════════════════════════════════════════════
# CODE GENERATOR TESTS
# ═══════════════════════════════════════════════════════════════

class TestCodeGen:
    def test_hello_world(self):
        c = compile_to_c('main()\n    print("Hello")\n')
        assert 'til_print_str("Hello")' in c
        assert 'int main(' in c

    def test_integer_math(self):
        c = compile_to_c('main()\n    let x = 2 + 3\n')
        assert '(2 + 3)' in c

    def test_cast_generates_c_cast(self):
        c = compile_to_c('main()\n    let x = 10 as float\n')
        assert '((double)10)' in c

    def test_enum_generates_c_enum(self):
        c = compile_to_c('enum Color\n    Red\n    Green\n    Blue\nmain()\n    let c = Color.Red\n')
        assert 'typedef enum' in c
        assert 'Color_Red' in c
        assert 'Color_Green' in c

    def test_enum_with_values_generates_c(self):
        c = compile_to_c('enum Http\n    OK = 200\n    NotFound = 404\nmain()\n    print(Http.OK)\n')
        assert 'Http_OK = 200' in c
        assert 'Http_NotFound = 404' in c

    def test_struct_and_constructor(self):
        c = compile_to_c('struct Point\n    x: float\n    y: float\nmain()\n    let p = Point { x: 1.0, y: 2.0 }\n')
        assert 'struct Point' in c
        assert 'Point_create' in c

    def test_method_forward_declarations(self):
        c = compile_to_c('struct Foo\n    x: int\nimpl Foo\n    get(self) -> int\n        return self.x\nmain()\n    let f = Foo { x: 42 }\n')
        assert 'static int64_t Foo_get(Foo* self);' in c

    def test_self_uses_arrow(self):
        c = compile_to_c('struct Foo\n    x: int\nimpl Foo\n    get(self) -> int\n        return self.x\nmain()\n    let f = Foo { x: 42 }\n')
        assert 'self->x' in c

    def test_string_concat_variables(self):
        c = compile_to_c('main()\n    let name = "World"\n    print("Hello, " + name)\n')
        assert 'til_str_concat' in c

    def test_print_float_variable(self):
        c = compile_to_c('main()\n    let x = 3.14\n    print(x)\n')
        assert 'til_print_float' in c

    def test_print_bool_variable(self):
        c = compile_to_c('main()\n    let x = true\n    print(x)\n')
        assert 'til_print_bool' in c

    def test_level_0_always_inline(self):
        c = compile_to_c('#[level: 0]\nfast(a: int) -> int\n    return a\nmain()\n    print(fast(1))\n')
        assert 'always_inline' in c

    def test_level_1_inline(self):
        c = compile_to_c('#[level: 1]\nfoo(a: int) -> int\n    return a\nmain()\n    print(foo(1))\n')
        assert 'inline' in c

    def test_default_parameters(self):
        c = compile_to_c('pow2(x: int, n: int = 2) -> int\n    return x\nmain()\n    print(pow2(3))\n')
        assert 'til_pow2(3, 2)' in c


# ═══════════════════════════════════════════════════════════════
# INTEGRATION TESTS (compile + run)
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    def test_hello_world(self):
        out = compile_and_run('main()\n    print("Hello, World!")\n')
        assert out == "Hello, World!"

    def test_arithmetic(self):
        out = compile_and_run('main()\n    print(2 + 3)\n')
        assert out == "5"

    def test_variables(self):
        out = compile_and_run('main()\n    var x = 10\n    x += 5\n    print(x)\n')
        assert out == "15"

    def test_if_else(self):
        out = compile_and_run('main()\n    let x = 10\n    if x > 5\n        print("big")\n    else\n        print("small")\n')
        assert out == "big"

    def test_for_loop(self):
        out = compile_and_run('main()\n    var sum = 0\n    for i in 1..=5\n        sum += i\n    print(sum)\n')
        assert out == "15"

    def test_function_call(self):
        out = compile_and_run('add(a: int, b: int) -> int\n    return a + b\nmain()\n    print(add(3, 4))\n')
        assert out == "7"

    def test_recursive_factorial(self):
        out = compile_and_run('fact(n: int) -> int\n    if n <= 1\n        return 1\n    return n * fact(n - 1)\nmain()\n    print(fact(5))\n')
        assert out == "120"

    def test_struct_and_method(self):
        src = """struct Point
    x: float
    y: float
impl Point
    new(x: float, y: float) -> Point
        return Point { x: x, y: y }
    get_x(self) -> float
        return self.x
main()
    let p = Point.new(3.0, 4.0)
    print(p.get_x())
"""
        out = compile_and_run(src)
        assert out == "3"

    def test_enum_values(self):
        src = """enum Status
    OK = 200
    NotFound = 404
main()
    print(Status.OK)
    print(Status.NotFound)
"""
        out = compile_and_run(src)
        assert out == "200\n404"

    def test_type_cast(self):
        out = compile_and_run('main()\n    let x = 10\n    let y = x as float\n    print(y)\n')
        assert out == "10"

    def test_string_concat(self):
        out = compile_and_run('main()\n    let name = "World"\n    print("Hello, " + name + "!")\n')
        assert out == "Hello, World!"

    def test_while_loop(self):
        out = compile_and_run('main()\n    var n = 5\n    while n > 0\n        n -= 1\n    print(n)\n')
        assert out == "0"

    def test_fizzbuzz_15(self):
        src = """main()
    for i in 1..=15
        if i % 15 == 0
            print("FizzBuzz")
        elif i % 3 == 0
            print("Fizz")
        elif i % 5 == 0
            print("Buzz")
        else
            print(i)
"""
        out = compile_and_run(src)
        lines = out.split('\n')
        assert lines[0] == "1"
        assert lines[2] == "Fizz"
        assert lines[4] == "Buzz"
        assert lines[14] == "FizzBuzz"


# ═══════════════════════════════════════════════════════════════
# EXAMPLE FILE TESTS
# ═══════════════════════════════════════════════════════════════

class TestExamples:
    """Ensure all bundled examples compile and run without errors."""

    EXAMPLES = [
        "examples/01_hello.til",
        "examples/02_variables.til",
        "examples/03_functions.til",
        "examples/04_structs.til",
        "examples/05_control_flow.til",
        "examples/06_enums.til",
        "examples/07_multilevel.til",
        "examples/basic/hello.til",
        "examples/basic/variables.til",
        "examples/basic/functions.til",
        "examples/basic/control_flow.til",
        "examples/advanced/structs.til",
        "examples/multilevel/multilevel.til",
    ]

    def _run_example(self, path):
        root = os.path.join(os.path.dirname(__file__), '..')
        full = os.path.join(root, path)
        result = subprocess.run(
            [sys.executable, os.path.join(root, 'src', 'til.py'), 'run', full],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"{path} failed:\n{result.stderr}"

    def test_01_hello(self):
        self._run_example("examples/01_hello.til")

    def test_02_variables(self):
        self._run_example("examples/02_variables.til")

    def test_03_functions(self):
        self._run_example("examples/03_functions.til")

    def test_04_structs(self):
        self._run_example("examples/04_structs.til")

    def test_05_control_flow(self):
        self._run_example("examples/05_control_flow.til")

    def test_06_enums(self):
        self._run_example("examples/06_enums.til")

    def test_07_multilevel(self):
        self._run_example("examples/07_multilevel.til")

    def test_basic_hello(self):
        self._run_example("examples/basic/hello.til")

    def test_basic_variables(self):
        self._run_example("examples/basic/variables.til")

    def test_basic_functions(self):
        self._run_example("examples/basic/functions.til")

    def test_basic_control_flow(self):
        self._run_example("examples/basic/control_flow.til")

    def test_advanced_structs(self):
        self._run_example("examples/advanced/structs.til")

    def test_multilevel(self):
        self._run_example("examples/multilevel/multilevel.til")


# ═══════════════════════════════════════════════════════════════
# PHASE 2: ERROR REPORTING TESTS
# ═══════════════════════════════════════════════════════════════

class TestErrorReporter:
    def test_format_error_basic(self):
        src = "let x = 10\nlet y = \n"
        reporter = ErrorReporter(src, "test.til", use_color=False)
        msg = reporter.format_error(2, 9, "Expected expression")
        assert "error" in msg
        assert "test.til:2:9" in msg
        assert "let y =" in msg

    def test_format_error_with_hint(self):
        src = "let x = 10\n"
        reporter = ErrorReporter(src, "test.til", use_color=False)
        msg = reporter.format_error(1, 5, "Bad syntax", hint="Try using 'var' instead")
        assert "hint" in msg
        assert "Try using" in msg

    def test_format_warning(self):
        src = "let x = 10\n"
        reporter = ErrorReporter(src, "test.til", use_color=False)
        msg = reporter.format_warning(1, 5, "Unused variable 'x'")
        assert "warning" in msg
        assert "test.til:1:5" in msg

    def test_get_hint_for_common_errors(self):
        assert get_hint_for_error("Expected IDENT, got EQ") != ""
        assert get_hint_for_error("Expected RPAREN") != ""
        assert get_hint_for_error("Unexpected token: NEWLINE") != ""

    def test_error_with_context_lines(self):
        src = "line1\nline2\nline3\nline4\nline5\n"
        reporter = ErrorReporter(src, "test.til", use_color=False)
        msg = reporter.format_error(3, 1, "Error here")
        assert "line2" in msg  # context line before
        assert "line3" in msg  # error line
        assert "line4" in msg  # context line after

    def test_lexer_error_is_formatted(self):
        """Lexer errors should use ErrorReporter."""
        try:
            lex("let x = `bad`")
            assert False, "Should have raised"
        except SyntaxError as e:
            msg = str(e)
            assert "Unexpected" in msg


# ═══════════════════════════════════════════════════════════════
# PHASE 2: LEXER ADDITIONAL TESTS
# ═══════════════════════════════════════════════════════════════

class TestLexerPhase2:
    def test_hex_literal(self):
        tokens = lex("0xFF")
        ints = [t for t in tokens if t.type == TokenType.INT]
        assert ints[0].value == 255

    def test_binary_literal(self):
        tokens = lex("0b1010")
        ints = [t for t in tokens if t.type == TokenType.INT]
        assert ints[0].value == 10

    def test_underscore_in_number(self):
        tokens = lex("1_000_000")
        ints = [t for t in tokens if t.type == TokenType.INT]
        assert ints[0].value == 1000000

    def test_scientific_notation(self):
        tokens = lex("1e3")
        floats = [t for t in tokens if t.type == TokenType.FLOAT]
        assert floats[0].value == 1000.0

    def test_escape_sequences(self):
        tokens = lex(r'"hello\nworld"')
        strs = [t for t in tokens if t.type == TokenType.STRING]
        assert '\n' in strs[0].value

    def test_bitwise_operators(self):
        tokens = lex("& | ^ ~ << >>")
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert TokenType.AMP in types
        assert TokenType.PIPE in types
        assert TokenType.CARET in types
        assert TokenType.TILDE in types
        assert TokenType.SHL in types
        assert TokenType.SHR in types

    def test_compound_assignments(self):
        tokens = lex("+= -= *= /=")
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert TokenType.PLUSEQ in types
        assert TokenType.MINUSEQ in types
        assert TokenType.STAREQ in types
        assert TokenType.SLASHEQ in types

    def test_empty_source(self):
        tokens = lex("")
        assert tokens[-1].type == TokenType.EOF

    def test_comment_skipped(self):
        tokens = lex("# this is a comment\nlet x = 1\n")
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF, TokenType.DEDENT)]
        assert TokenType.LET in types
        assert len([t for t in tokens if t.type == TokenType.IDENT]) == 1


# ═══════════════════════════════════════════════════════════════
# PHASE 2: PARSER ADDITIONAL TESTS
# ═══════════════════════════════════════════════════════════════

class TestParserPhase2:
    def test_lambda_expression(self):
        prog = parse('main()\n    let f = |x| x + 1\n')
        decl = prog.statements[0].body.statements[0]
        assert isinstance(decl.value, Lambda)
        assert len(decl.value.params) == 1
        assert decl.value.params[0][0] == "x"

    def test_lambda_multi_param(self):
        prog = parse('main()\n    let f = |x, y| x + y\n')
        decl = prog.statements[0].body.statements[0]
        assert isinstance(decl.value, Lambda)
        assert len(decl.value.params) == 2

    def test_ternary_if_expr(self):
        prog = parse('main()\n    let x = 1 if true else 0\n')
        decl = prog.statements[0].body.statements[0]
        assert isinstance(decl.value, IfExpr)

    def test_list_comprehension(self):
        prog = parse('main()\n    let arr = [x * 2 for x in 0..10]\n')
        decl = prog.statements[0].body.statements[0]
        assert isinstance(decl.value, ListComprehension)
        assert decl.value.var == "x"

    def test_list_comprehension_with_filter(self):
        prog = parse('main()\n    let arr = [x for x in 0..10 if x > 3]\n')
        decl = prog.statements[0].body.statements[0]
        assert isinstance(decl.value, ListComprehension)
        assert decl.value.condition is not None

    def test_null_check_operator(self):
        prog = parse('main()\n    let x = foo()?\n')
        decl = prog.statements[0].body.statements[0]
        assert isinstance(decl.value, NullCheck)

    def test_trait_definition(self):
        from til import TraitDef
        prog = parse('trait Printable\n    fn to_string(self) -> str\n        return ""\n')
        assert isinstance(prog.statements[0], TraitDef)

    def test_const_declaration(self):
        prog = parse('const PI = 3\nmain()\n    print(PI)\n')
        assert isinstance(prog.statements[0], VarDecl)
        assert prog.statements[0].is_const == True

    def test_unary_not(self):
        prog = parse('main()\n    let x = not true\n')
        decl = prog.statements[0].body.statements[0]
        assert isinstance(decl.value, UnaryOp)
        assert decl.value.op == "not"

    def test_power_operator(self):
        prog = parse('main()\n    let x = 2 ** 3\n')
        decl = prog.statements[0].body.statements[0]
        assert isinstance(decl.value, BinaryOp)
        assert decl.value.op == "**"

    def test_nested_if(self):
        prog = parse('main()\n    if true\n        if false\n            print("inner")\n')
        stmt = prog.statements[0].body.statements[0]
        assert isinstance(stmt, If)
        inner = stmt.then_body.statements[0]
        assert isinstance(inner, If)

    def test_continue_in_loop(self):
        prog = parse('main()\n    for i in 0..10\n        if i == 5\n            continue\n')
        for_stmt = prog.statements[0].body.statements[0]
        if_stmt = for_stmt.body.statements[0]
        assert isinstance(if_stmt.then_body.statements[0], Continue)


# ═══════════════════════════════════════════════════════════════
# PHASE 2: CODEGEN ADDITIONAL TESTS
# ═══════════════════════════════════════════════════════════════

class TestCodeGenPhase2:
    def test_bounds_checking_level2(self):
        """Level 2 (default) should generate bounds checks."""
        c = compile_to_c('main()\n    let arr = [1, 2, 3]\n    let x = arr[0]\n')
        assert 'til_bounds_check' in c

    def test_ternary_expression(self):
        c = compile_to_c('main()\n    let x = 1 if true else 0\n')
        assert '?' in c  # C ternary operator

    def test_lambda_generates_function(self):
        c = compile_to_c('main()\n    let f = |x| x + 1\n')
        assert '_til_lambda_' in c

    def test_for_array_literal(self):
        """For loop over array literal should work."""
        c = compile_to_c('main()\n    for x in [1, 2, 3]\n        print(x)\n')
        assert '_arr_x' in c  # array created for iteration

    def test_string_method_contains(self):
        c = compile_to_c('main()\n    let s = "hello world"\n    let b = s.contains("world")\n')
        assert 'til_str_contains' in c

    def test_string_method_trim(self):
        c = compile_to_c('main()\n    let s = "  hello  "\n    let t = s.trim()\n')
        assert 'til_str_trim' in c

    def test_string_method_to_upper(self):
        c = compile_to_c('main()\n    let s = "hello"\n    let u = s.to_upper()\n')
        assert 'til_str_to_upper' in c

    def test_list_comprehension_codegen(self):
        c = compile_to_c('main()\n    let arr = [x * 2 for x in 0..5]\n')
        assert '_comp_' in c
        assert 'malloc' in c

    def test_power_operator_codegen(self):
        c = compile_to_c('main()\n    let x = 2 ** 3\n')
        assert 'pow(2, 3)' in c

    def test_bitwise_operators_codegen(self):
        c = compile_to_c('main()\n    let x = 5 & 3\n    let y = 5 | 3\n    let z = 5 ^ 3\n')
        assert '(5 & 3)' in c
        assert '(5 | 3)' in c
        assert '(5 ^ 3)' in c

    def test_not_operator_codegen(self):
        c = compile_to_c('main()\n    let x = not true\n')
        assert '(!true)' in c

    def test_while_true_loop(self):
        c = compile_to_c('main()\n    loop\n        break\n')
        assert 'while (1)' in c

    def test_match_generates_switch(self):
        c = compile_to_c('main()\n    let x = 1\n    match x\n        1 => print("one")\n        2 => print("two")\n')
        assert 'switch' in c
        assert 'case 1' in c


# ═══════════════════════════════════════════════════════════════
# PHASE 2: INTEGRATION TESTS (compile + run)
# ═══════════════════════════════════════════════════════════════

class TestIntegrationPhase2:
    def test_ternary_expression_run(self):
        out = compile_and_run('main()\n    let x = 10 if true else 20\n    print(x)\n')
        assert out == "10"

    def test_ternary_false_branch(self):
        out = compile_and_run('main()\n    let x = 10 if false else 20\n    print(x)\n')
        assert out == "20"

    def test_for_array_literal_run(self):
        out = compile_and_run('main()\n    var sum = 0\n    for x in [10, 20, 30]\n        sum += x\n    print(sum)\n')
        assert out == "60"

    def test_power_operator_run(self):
        out = compile_and_run('main()\n    print(2 ** 10)\n')
        assert out == "1024"

    def test_bitwise_and_run(self):
        out = compile_and_run('main()\n    print(12 & 10)\n')
        assert out == "8"

    def test_bitwise_or_run(self):
        out = compile_and_run('main()\n    print(12 | 3)\n')
        assert out == "15"

    def test_not_operator_run(self):
        out = compile_and_run('main()\n    let x = not false\n    print(x)\n')
        assert out == "true"

    def test_nested_function_calls(self):
        src = "add(a: int, b: int) -> int\n    return a + b\nmain()\n    print(add(add(1, 2), add(3, 4)))\n"
        out = compile_and_run(src)
        assert out == "10"

    def test_string_concat_multi(self):
        out = compile_and_run('main()\n    let a = "Hello"\n    let b = " "\n    let c = "World"\n    print(a + b + c)\n')
        assert out == "Hello World"

    def test_string_contains(self):
        out = compile_and_run('main()\n    let s = "hello world"\n    print(s.contains("world"))\n')
        assert out == "true"

    def test_string_to_upper(self):
        out = compile_and_run('main()\n    let s = "hello"\n    print(s.to_upper())\n')
        assert out == "HELLO"

    def test_string_to_lower(self):
        out = compile_and_run('main()\n    let s = "HELLO"\n    print(s.to_lower())\n')
        assert out == "hello"

    def test_string_trim(self):
        out = compile_and_run('main()\n    let s = "  hello  "\n    print(s.trim())\n')
        assert out == "hello"

    def test_string_starts_with(self):
        out = compile_and_run('main()\n    let s = "hello world"\n    print(s.starts_with("hello"))\n')
        assert out == "true"

    def test_string_ends_with(self):
        out = compile_and_run('main()\n    let s = "hello world"\n    print(s.ends_with("world"))\n')
        assert out == "true"

    def test_string_replace(self):
        out = compile_and_run('main()\n    let s = "hello world"\n    print(s.replace("world", "TIL"))\n')
        assert out == "hello TIL"

    def test_list_comprehension_run(self):
        src = """main()
    let arr = [x * 2 for x in 0..5]
    var sum = 0
    for i in 0..5
        sum += arr[i]
    print(sum)
"""
        out = compile_and_run(src)
        assert out == "20"  # 0+2+4+6+8 = 20

    def test_list_comprehension_with_filter(self):
        src = """main()
    let arr = [x for x in 0..10 if x > 5]
    print(arr[0])
"""
        out = compile_and_run(src)
        assert out == "6"

    def test_bounds_check_catches_overflow(self):
        """Bounds checking should abort on out-of-range access."""
        src = """main()
    let arr = [1, 2, 3]
    let x = arr[10]
"""
        compiler = TILCompiler()
        compiler.check_types = False
        with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as f:
            exe = f.name
        try:
            ok = compiler.compile_to_executable(src, exe, "<test>")
            assert ok
            result = subprocess.run([exe], capture_output=True, text=True, timeout=10)
            assert result.returncode != 0  # should fail
            assert "Bounds check" in result.stderr
        finally:
            try:
                os.unlink(exe)
            except:
                pass

    def test_match_statement_run(self):
        src = """main()
    let x = 2
    match x
        1 => print("one")
        2 => print("two")
        3 => print("three")
"""
        out = compile_and_run(src)
        assert out == "two"

    def test_loop_with_counter(self):
        src = """main()
    var count = 0
    loop
        count += 1
        if count == 5
            break
    print(count)
"""
        out = compile_and_run(src)
        assert out == "5"

    def test_multiple_functions(self):
        src = """double(x: int) -> int
    return x * 2
triple(x: int) -> int
    return x * 3
main()
    print(double(5))
    print(triple(5))
"""
        out = compile_and_run(src)
        assert out == "10\n15"


# ═══════════════════════════════════════════════════════════════
# PHASE 2: TYPE CHECKER TESTS
# ═══════════════════════════════════════════════════════════════

class TestTypeChecker:
    def _check(self, src: str) -> list:
        tokens = lex(src)
        ast = Parser(tokens, "<test>").parse()
        checker = TypeChecker()
        return checker.check(ast)

    def test_no_errors_valid_code(self):
        errors = self._check('main()\n    let x = 42\n')
        assert len(errors) == 0

    def test_immutable_assignment_error(self):
        errors = self._check('main()\n    let x = 42\n    x = 10\n')
        assert any("immutable" in e for e in errors)

    def test_type_mismatch_warning(self):
        errors = self._check('main()\n    let x: int = "hello"\n')
        assert len(errors) > 0

    def test_function_return_type(self):
        errors = self._check('add(a: int, b: int) -> int\n    return a + b\nmain()\n    let x = add(1, 2)\n')
        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════
# PHASE 3: KAZAKH LANGUAGE TESTS
# ═══════════════════════════════════════════════════════════════

class TestKazakh:
    def test_kazakh_keywords_exist(self):
        assert 'егер' in KAZAKH_KEYWORDS  # if
        assert 'қайтару' in KAZAKH_KEYWORDS  # return
        assert 'құрылым' in KAZAKH_KEYWORDS  # struct
        assert 'функция' in KAZAKH_KEYWORDS  # fn

    def test_kazakh_if_keyword(self):
        """Kazakh 'егер' should work as 'if'."""
        tokens = lex('егер true\n    print("ok")\n')
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF, TokenType.INDENT, TokenType.DEDENT)]
        assert TokenType.IF in types

    def test_kazakh_for_loop(self):
        """Kazakh 'үшін ... ішінде' should work as 'for ... in'."""
        tokens = lex('үшін и ішінде 1..5\n    print(и)\n')
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF, TokenType.INDENT, TokenType.DEDENT)]
        assert TokenType.FOR in types
        assert TokenType.IN in types

    def test_kazakh_let_var(self):
        """Kazakh 'тұрақты'/'айнымалы' should work as 'let'/'var'."""
        tokens = lex('тұрақты x = 10\nайнымалы y = 20\n')
        types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert TokenType.LET in types
        assert TokenType.VAR in types

    def test_kazakh_example_compiles(self):
        """The Kazakh salam.til example should compile and run."""
        root = os.path.join(os.path.dirname(__file__), '..')
        path = os.path.join(root, 'examples', 'kazakh', 'salam.til')
        result = subprocess.run(
            [sys.executable, os.path.join(root, 'src', 'til.py'), 'run', path],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"Kazakh example failed:\n{result.stderr}"
        assert "7" in result.stdout
        assert "15" in result.stdout
        assert "120" in result.stdout

    def test_kazakh_hello_compile_and_run(self):
        """Simple Kazakh program should compile and run."""
        src = 'main()\n    тұрақты x = 42\n    print(x)\n'
        out = compile_and_run(src)
        assert out == "42"

    def test_unicode_var_name_mangling(self):
        """Unicode variable names should be mangled to valid C."""
        c = compile_to_c('main()\n    тұрақты нәтиже = 10\n    print(нәтиже)\n')
        assert '_u' in c  # mangled names contain _uXXXX
        assert 'int64_t' in c

    def test_unicode_function_name_mangling(self):
        """Unicode function names should be mangled to valid C."""
        c = compile_to_c('қосу(а: int, б: int) -> int\n    return а + б\nmain()\n    print(қосу(1, 2))\n')
        assert 'til__u' in c  # mangled function name


# ═══════════════════════════════════════════════════════════════
# PHASE 3: MODULE/IMPORT TESTS
# ═══════════════════════════════════════════════════════════════

class TestModuleSystem:
    def test_import_parses(self):
        """import statement should parse correctly."""
        prog = parse('import math_utils\nmain()\n    print(1)\n')
        assert any(isinstance(s, Import) for s in prog.statements)

    def test_from_import_parses(self):
        """from module import name should parse."""
        prog = parse('from utils import helper\nmain()\n    print(1)\n')
        imp = [s for s in prog.statements if isinstance(s, Import)][0]
        assert imp.module == "utils"
        assert imp.items == ["helper"]

    def test_module_resolver(self):
        """ModuleResolver should find .til files."""
        root = os.path.join(os.path.dirname(__file__), '..')
        resolver = ModuleResolver(os.path.join(root, 'examples'))
        path = resolver.resolve('math_utils', os.path.join(root, 'examples', '08_imports.til'))
        assert path is not None
        assert path.endswith('math_utils.til')

    def test_import_example_runs(self):
        """08_imports.til example should compile and run."""
        root = os.path.join(os.path.dirname(__file__), '..')
        path = os.path.join(root, 'examples', '08_imports.til')
        result = subprocess.run(
            [sys.executable, os.path.join(root, 'src', 'til.py'), 'run', path],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"Import example failed:\n{result.stderr}"
        assert "25" in result.stdout  # square(5)
        assert "27" in result.stdout  # cube(3)
        assert "10" in result.stdout  # clamp(15, 0, 10)


# ═══════════════════════════════════════════════════════════════
# PHASE 3: OPTION/RESULT TYPE TESTS
# ═══════════════════════════════════════════════════════════════

class TestOptionResult:
    def test_option_type_to_c(self):
        """Option<int> should generate TIL_Option_int."""
        gen = CCodeGenerator()
        c = gen.type_to_c(OptionType(T_INT))
        assert c == "TIL_Option_int"

    def test_option_float_type(self):
        gen = CCodeGenerator()
        c = gen.type_to_c(OptionType(T_FLOAT))
        assert c == "TIL_Option_float"

    def test_result_type_to_c(self):
        """Result<int, str> should generate TIL_Result_int."""
        gen = CCodeGenerator()
        c = gen.type_to_c(ResultType(T_INT, T_STR))
        assert c == "TIL_Result_int"

    def test_some_codegen(self):
        c = compile_to_c('main()\n    let x = Some(42)\n')
        assert 'til_Some_int(42)' in c

    def test_option_in_helpers(self):
        c = compile_to_c('main()\n    print(1)\n')
        assert 'TIL_Option_int' in c
        assert 'TIL_Result_int' in c


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
