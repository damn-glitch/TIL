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
    ExprStmt, FuncDef, StructDef, EnumDef, ImplBlock, Program,
    PrimitiveType, ArrayType, StructType, T_INT, T_FLOAT, T_STR, T_BOOL, T_VOID, T_UNKNOWN,
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


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
