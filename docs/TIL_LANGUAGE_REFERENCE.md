# TIL Language Reference

## Complete Language Specification v2.0

---

<div align="center">

**Author: Alisher Beisembekov**

*"Проще Python. Быстрее C. Умнее всех."*

*"Simpler than Python. Faster than C. Smarter than all."*

</div>

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Philosophy & Design Goals](#2-philosophy--design-goals)
3. [Multi-Level Programming System](#3-multi-level-programming-system)
4. [Lexical Structure](#4-lexical-structure)
5. [Types](#5-types)
6. [Variables & Constants](#6-variables--constants)
7. [Operators](#7-operators)
8. [Control Flow](#8-control-flow)
9. [Functions](#9-functions)
10. [Structures](#10-structures)
11. [Enumerations](#11-enumerations)
12. [Implementation Blocks](#12-implementation-blocks)
13. [Pattern Matching](#13-pattern-matching)
14. [Built-in Functions](#14-built-in-functions)
15. [Attributes](#15-attributes)
16. [Memory Model](#16-memory-model)
17. [Error Handling](#17-error-handling)
18. [Standard Library](#18-standard-library)
19. [Interoperability](#19-interoperability)
20. [Best Practices](#20-best-practices)
21. [Complete Examples](#21-complete-examples)
22. [Grammar Reference](#22-grammar-reference)

---

## 1. Introduction

### 1.1 What is TIL?

**TIL** (The Intelligent Language) is a multi-level, statically-typed programming language designed by **Alisher Beisembekov**. TIL combines the readability of Python, the performance of C, and introduces a revolutionary concept of **abstraction levels** that allows developers to choose the right trade-off between productivity and performance on a per-function basis.

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Level System** | 4 abstraction levels (0-3) in one language |
| **Native Performance** | Compiles to optimized C, then native code |
| **Python-like Syntax** | Clean, readable, indentation-based |
| **Static Typing** | Strong type system with inference |
| **Zero Runtime** | No VM, no garbage collector overhead |
| **C Interoperability** | Direct access to generated C code |
| **Single-File Compiler** | No complex build systems needed |

### 1.3 Hello World

```til
# TIL v2.0 - Hello World
# Author: Alisher Beisembekov

main()
    print("Hello, World!")
```

### 1.4 A More Complete Example

```til
# TIL v2.0 - Feature Showcase
# Author: Alisher Beisembekov

# Define a structure
struct Point
    x: float
    y: float

# Implement methods
impl Point
    new(x: float, y: float) -> Point
        return Point { x: x, y: y }
    
    distance(self) -> float
        return sqrt(self.x ** 2 + self.y ** 2)
    
    add(self, other: Point) -> Point
        return Point.new(self.x + other.x, self.y + other.y)

# Main entry point
main()
    let p1 = Point.new(3.0, 4.0)
    let p2 = Point.new(1.0, 2.0)
    
    print("Distance from origin:")
    print(p1.distance())  # Output: 5.0
    
    let p3 = p1.add(p2)
    print("Sum point distance:")
    print(p3.distance())  # Output: 7.211...
```

---

## 2. Philosophy & Design Goals

### 2.1 Core Philosophy

TIL was created by **Alisher Beisembekov** with the following principles:

1. **Readability First**: Code should be self-documenting
2. **Performance Without Sacrifice**: High-level code should compile to efficient machine code
3. **Gradual Optimization**: Start simple, optimize only where needed
4. **No Hidden Costs**: What you write is what you get
5. **Interoperability**: Work seamlessly with existing C ecosystems

### 2.2 The Multi-Level Approach

Traditional languages force you to choose:
- **High-level** (Python, Ruby): Easy to write, slow to run
- **Low-level** (C, Assembly): Fast to run, hard to write

TIL's innovation is allowing **both in the same program**:

```til
# Ultra-fast inner loop (Level 0)
#[level: 0]
dot_product(a: float, b: float, c: float, d: float) -> float
    return a * c + b * d

# Easy-to-use API (Level 3)
#[level: 3]
calculate_similarity(vec1, vec2)
    # High-level logic here
    return dot_product(vec1.x, vec1.y, vec2.x, vec2.y)
```

### 2.3 Design Decisions

| Decision | Rationale |
|----------|-----------|
| Indentation-based blocks | Cleaner code, fewer bugs from brace mismatches |
| No semicolons | Reduces visual noise |
| Explicit `self` in methods | Clear distinction between methods and functions |
| Type inference | Less typing while maintaining type safety |
| Compiles to C | Leverage decades of C compiler optimizations |

---

## 3. Multi-Level Programming System

### 3.1 Overview

TIL's **Multi-Level System** is its defining feature. Each function can operate at a different abstraction level:

| Level | Name | Description | Use Case |
|-------|------|-------------|----------|
| **0** | Hardware | SIMD, always_inline, raw performance | Inner loops, math kernels |
| **1** | Systems | C-like control, inline hints | Performance-critical code |
| **2** | Safe | Bounds checking, safety (default) | Most application code |
| **3** | Script | Python-like productivity | Prototyping, glue code |

### 3.2 Level 0: Hardware

**Purpose**: Maximum performance, direct hardware utilization

**Features**:
- Functions are always inlined
- SIMD intrinsics available
- No bounds checking
- No safety overhead

**Attributes**: `#[level: 0]`, `#[inline]`, `#[simd]`

```til
#[level: 0]
#[inline]
fast_multiply(a: float, b: float) -> float
    return a * b

#[level: 0]
vector_add(a: i32, b: i32, c: i32, d: i32) -> i32
    # Direct SIMD-style operation
    return a + b + c + d
```

**Generated C Code**:
```c
__attribute__((always_inline)) inline double fast_multiply(double a, double b) {
    return a * b;
}
```

### 3.3 Level 1: Systems

**Purpose**: C-like control with modern syntax

**Features**:
- Inline hints (compiler may choose)
- Manual memory control available
- Pointer operations
- Minimal overhead

**Attributes**: `#[level: 1]`

```til
#[level: 1]
factorial(n: int) -> int
    if n <= 1
        return 1
    return n * factorial(n - 1)

#[level: 1]
binary_search(arr: int[], target: int, low: int, high: int) -> int
    if low > high
        return -1
    
    let mid = (low + high) / 2
    
    if arr[mid] == target
        return mid
    elif arr[mid] > target
        return binary_search(arr, target, low, mid - 1)
    else
        return binary_search(arr, target, mid + 1, high)
```

### 3.4 Level 2: Safe (Default)

**Purpose**: Safe, productive code for most applications

**Features**:
- Bounds checking on arrays
- Type safety enforced
- Balanced performance/safety
- Default level if not specified

**Attributes**: `#[level: 2]` (optional, it's the default)

```til
# Level 2 is default, no attribute needed
struct User
    name: str
    age: int
    active: bool

impl User
    new(name: str, age: int) -> User
        return User { name: name, age: age, active: true }
    
    greet(self)
        print("Hello, I'm " + self.name)
    
    is_adult(self) -> bool
        return self.age >= 18

process_users(users: User[])
    for user in users
        if user.is_adult()
            user.greet()
```

### 3.5 Level 3: Script

**Purpose**: Maximum productivity, Python-like experience

**Features**:
- Minimal boilerplate
- Dynamic-style features
- Great for prototyping
- Easy string operations

**Attributes**: `#[level: 3]`

```til
#[level: 3]
quick_sort(arr)
    if len(arr) <= 1
        return arr
    
    let pivot = arr[0]
    let less = []
    let greater = []
    
    for i in 1..len(arr)
        if arr[i] <= pivot
            less.append(arr[i])
        else
            greater.append(arr[i])
    
    return quick_sort(less) + [pivot] + quick_sort(greater)

#[level: 3]
process_text(text)
    print("Processing: " + text)
    print("Length: " + str(len(text)))
    print("Uppercase: " + text.upper())
```

### 3.6 Mixing Levels

The true power of TIL is mixing levels in one program:

```til
# Level 0: Ultra-fast math kernel
#[level: 0]
compute_distance(x1: float, y1: float, x2: float, y2: float) -> float
    let dx = x2 - x1
    let dy = y2 - y1
    return sqrt(dx * dx + dy * dy)

# Level 2: Safe data structure
struct City
    name: str
    x: float
    y: float

impl City
    distance_to(self, other: City) -> float
        # Call Level 0 function from Level 2
        return compute_distance(self.x, self.y, other.x, other.y)

# Level 3: Easy-to-use API
#[level: 3]
find_nearest_city(cities, target)
    let nearest = cities[0]
    let min_dist = target.distance_to(nearest)
    
    for city in cities
        let dist = target.distance_to(city)
        if dist < min_dist
            min_dist = dist
            nearest = city
    
    return nearest

main()
    let cities = [
        City { name: "Astana", x: 71.4, y: 51.1 },
        City { name: "Almaty", x: 76.9, y: 43.2 },
        City { name: "Shymkent", x: 69.6, y: 42.3 }
    ]
    
    let target = City { name: "Target", x: 70.0, y: 45.0 }
    let nearest = find_nearest_city(cities, target)
    
    print("Nearest city: " + nearest.name)
```

---

## 4. Lexical Structure

### 4.1 Character Set

TIL source files are UTF-8 encoded. Identifiers can contain:
- Letters: `a-z`, `A-Z`
- Digits: `0-9` (not as first character)
- Underscore: `_`

### 4.2 Comments

```til
# Single-line comment

# Multi-line comments use
# multiple single-line comments

# TODO: TIL may support /* */ in future versions
```

### 4.3 Identifiers

```til
# Valid identifiers
my_variable
myVariable
_private
counter123
MAX_SIZE

# Invalid identifiers
123abc      # Cannot start with digit
my-var      # Hyphens not allowed
class       # Reserved keyword
```

### 4.4 Keywords

The following are reserved keywords in TIL:

| Category | Keywords |
|----------|----------|
| **Declarations** | `fn`, `let`, `var`, `const`, `mut`, `struct`, `enum`, `impl`, `trait`, `type`, `pub` |
| **Control Flow** | `if`, `else`, `elif`, `for`, `while`, `loop`, `match`, `in`, `return`, `break`, `continue` |
| **Logical** | `and`, `or`, `not` |
| **Literals** | `true`, `false`, `True`, `False`, `None` |
| **Other** | `self`, `as`, `import`, `from` |

### 4.5 Literals

#### Integer Literals
```til
let decimal = 42
let negative = -17
let with_underscores = 1_000_000
let hex = 0xFF
let binary = 0b1010
let octal = 0o755
```

#### Float Literals
```til
let pi = 3.14159
let scientific = 1.5e10
let negative_exp = 2.5e-3
let with_underscores = 1_234.567_890
```

#### String Literals
```til
let simple = "Hello, World!"
let with_escapes = "Line 1\nLine 2\tTabbed"
let with_quotes = "She said \"Hello\""
let single_quotes = 'Also valid'
```

#### Escape Sequences
| Sequence | Meaning |
|----------|---------|
| `\n` | Newline |
| `\t` | Tab |
| `\r` | Carriage return |
| `\\` | Backslash |
| `\"` | Double quote |
| `\'` | Single quote |

#### Boolean Literals
```til
let yes = true
let no = false
let also_yes = True    # Python-style also accepted
let also_no = False
```

### 4.6 Indentation

TIL uses **4 spaces** for indentation (not tabs):

```til
if condition
    # 4 spaces
    if nested_condition
        # 8 spaces
        do_something()
    else
        # 8 spaces
        do_other()
```

---

## 5. Types

### 5.1 Primitive Types

| Type | Size | Description | Range |
|------|------|-------------|-------|
| `int` | 64-bit | Signed integer | -2^63 to 2^63-1 |
| `float` | 64-bit | IEEE 754 double | ±1.8×10^308 |
| `bool` | 8-bit | Boolean | `true` / `false` |
| `char` | 8-bit | ASCII character | 0-255 |
| `str` | pointer | String | UTF-8 text |
| `void` | 0-bit | No value | (no value) |

### 5.2 Sized Integer Types

| Signed | Unsigned | Size |
|--------|----------|------|
| `i8` | `u8` | 8-bit |
| `i16` | `u16` | 16-bit |
| `i32` | `u32` | 32-bit |
| `i64` | `u64` | 64-bit |

```til
let byte: u8 = 255
let small: i16 = -32768
let normal: i32 = 2147483647
let big: i64 = 9223372036854775807
```

### 5.3 Floating Point Types

| Type | Size | Precision |
|------|------|-----------|
| `f32` | 32-bit | ~7 digits |
| `f64` | 64-bit | ~15 digits |
| `float` | 64-bit | Alias for `f64` |

```til
let single: f32 = 3.14
let double: f64 = 3.141592653589793
let default_float: float = 2.718281828
```

### 5.4 Strings

Strings in TIL are immutable sequences of characters:

```til
let greeting = "Hello"
let name = "World"
let message = greeting + ", " + name + "!"  # Concatenation

let length = len(message)  # Get length
```

### 5.5 Arrays

Arrays are fixed-size collections of elements:

```til
# Array literals
let numbers = [1, 2, 3, 4, 5]
let empty: int[] = []

# Accessing elements
let first = numbers[0]
let last = numbers[4]

# Array with type annotation
let floats: float[] = [1.0, 2.0, 3.0]

# Iterating
for num in numbers
    print(num)
```

### 5.6 Type Inference

TIL can infer types in most cases:

```til
let x = 42          # Inferred as int
let pi = 3.14       # Inferred as float
let name = "TIL"    # Inferred as str
let flag = true     # Inferred as bool

# Explicit type annotation when needed
let count: i32 = 0
let ratio: f32 = 0.5
```

### 5.7 Type Casting

Use the `as` keyword for explicit type conversion:

```til
let i = 42
let f = i as float          # int to float
let rounded = 3.7 as int    # float to int (truncates to 3)

let small: i8 = 100
let big = small as i64      # Widen

let char_code = 65
let letter = char_code as char  # int to char ('A')
```

---

## 6. Variables & Constants

### 6.1 Immutable Variables (let)

Variables declared with `let` cannot be reassigned:

```til
let x = 10
# x = 20  # ERROR: Cannot reassign immutable variable

let name = "TIL"
# name = "Other"  # ERROR
```

### 6.2 Mutable Variables (var)

Variables declared with `var` can be reassigned:

```til
var counter = 0
counter = counter + 1  # OK
counter += 1           # OK

var message = "Hello"
message = "World"      # OK
```

### 6.3 Constants (const)

Constants are compile-time values that never change:

```til
const PI = 3.14159265358979
const MAX_SIZE = 1000
const APP_NAME = "TIL Application"

# Constants are typically UPPER_SNAKE_CASE
const DAYS_IN_WEEK = 7
const HOURS_IN_DAY = 24
```

### 6.4 Mutability Keyword (mut)

The `mut` keyword can explicitly mark mutability:

```til
let mut x = 10  # Equivalent to var
x = 20          # OK

fn modify(mut value: int) -> int
    value = value * 2
    return value
```

### 6.5 Scope Rules

Variables are scoped to their containing block:

```til
main()
    let outer = 10
    
    if true
        let inner = 20
        print(outer)  # OK: outer is visible
        print(inner)  # OK: inner is visible
    
    print(outer)      # OK
    # print(inner)    # ERROR: inner is out of scope
```

### 6.6 Shadowing

Inner scopes can shadow outer variables:

```til
main()
    let x = 10
    print(x)  # 10
    
    if true
        let x = 20  # Shadows outer x
        print(x)    # 20
    
    print(x)  # 10 (outer x unchanged)
```

---

## 7. Operators

### 7.1 Arithmetic Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `a + b` |
| `-` | Subtraction | `a - b` |
| `*` | Multiplication | `a * b` |
| `/` | Division | `a / b` |
| `%` | Modulo | `a % b` |
| `**` | Power | `a ** b` |

```til
let sum = 10 + 5       # 15
let diff = 10 - 5      # 5
let product = 10 * 5   # 50
let quotient = 10 / 3  # 3 (integer division)
let remainder = 10 % 3 # 1
let power = 2 ** 10    # 1024
```

### 7.2 Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal | `a == b` |
| `!=` | Not equal | `a != b` |
| `<` | Less than | `a < b` |
| `>` | Greater than | `a > b` |
| `<=` | Less or equal | `a <= b` |
| `>=` | Greater or equal | `a >= b` |

```til
let a = 10
let b = 20

print(a == b)   # false
print(a != b)   # true
print(a < b)    # true
print(a > b)    # false
print(a <= 10)  # true
print(b >= 20)  # true
```

### 7.3 Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `and` | Logical AND | `a and b` |
| `or` | Logical OR | `a or b` |
| `not` | Logical NOT | `not a` |

```til
let a = true
let b = false

print(a and b)  # false
print(a or b)   # true
print(not a)    # false

# Short-circuit evaluation
if x != 0 and 10 / x > 1
    print("Safe division")
```

### 7.4 Bitwise Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&` | Bitwise AND | `a & b` |
| `\|` | Bitwise OR | `a \| b` |
| `^` | Bitwise XOR | `a ^ b` |
| `~` | Bitwise NOT | `~a` |
| `<<` | Left shift | `a << n` |
| `>>` | Right shift | `a >> n` |

```til
let a = 0b1010  # 10
let b = 0b1100  # 12

print(a & b)    # 0b1000 = 8
print(a | b)    # 0b1110 = 14
print(a ^ b)    # 0b0110 = 6
print(~a)       # -11 (inverts all bits)
print(a << 2)   # 0b101000 = 40
print(b >> 2)   # 0b0011 = 3
```

### 7.5 Assignment Operators

| Operator | Description | Equivalent |
|----------|-------------|------------|
| `=` | Assign | `a = b` |
| `+=` | Add and assign | `a = a + b` |
| `-=` | Subtract and assign | `a = a - b` |
| `*=` | Multiply and assign | `a = a * b` |
| `/=` | Divide and assign | `a = a / b` |

```til
var x = 10
x += 5   # x = 15
x -= 3   # x = 12
x *= 2   # x = 24
x /= 4   # x = 6
```

### 7.6 Range Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `..` | Exclusive range | `0..10` (0 to 9) |
| `..=` | Inclusive range | `0..=10` (0 to 10) |

```til
# Exclusive range (0, 1, 2, 3, 4)
for i in 0..5
    print(i)

# Inclusive range (0, 1, 2, 3, 4, 5)
for i in 0..=5
    print(i)
```

### 7.7 Operator Precedence

From highest to lowest:

1. `**` (power, right-associative)
2. `~`, unary `-` (bitwise NOT, negation)
3. `*`, `/`, `%` (multiplicative)
4. `+`, `-` (additive)
5. `<<`, `>>` (shifts)
6. `&` (bitwise AND)
7. `^` (bitwise XOR)
8. `|` (bitwise OR)
9. `==`, `!=`, `<`, `>`, `<=`, `>=` (comparison)
10. `not` (logical NOT)
11. `and` (logical AND)
12. `or` (logical OR)
13. `=`, `+=`, `-=`, `*=`, `/=` (assignment)

---

## 8. Control Flow

### 8.1 If Statements

```til
if condition
    # code

if condition
    # code
else
    # code

if condition1
    # code
elif condition2
    # code
elif condition3
    # code
else
    # code
```

**Example**:
```til
grade_letter(score: int) -> str
    if score >= 90
        return "A"
    elif score >= 80
        return "B"
    elif score >= 70
        return "C"
    elif score >= 60
        return "D"
    else
        return "F"
```

### 8.2 For Loops

#### Range-based For Loop
```til
# Exclusive range
for i in 0..10
    print(i)  # 0 through 9

# Inclusive range
for i in 1..=100
    print(i)  # 1 through 100

# With step (future feature)
# for i in 0..100 step 2
#     print(i)
```

#### Collection For Loop
```til
let names = ["Alice", "Bob", "Charlie"]

for name in names
    print("Hello, " + name)

let numbers = [1, 2, 3, 4, 5]
var sum = 0
for num in numbers
    sum += num
print(sum)  # 15
```

### 8.3 While Loops

```til
var count = 0
while count < 10
    print(count)
    count += 1

# With break
var i = 0
while true
    if i >= 5
        break
    print(i)
    i += 1
```

### 8.4 Loop (Infinite Loop)

```til
loop
    let input = get_input()
    
    if input == "quit"
        break
    
    process(input)
```

### 8.5 Break and Continue

```til
# break: Exit the loop entirely
for i in 0..100
    if i == 10
        break  # Stop at 10
    print(i)

# continue: Skip to next iteration
for i in 0..10
    if i % 2 == 0
        continue  # Skip even numbers
    print(i)  # Prints 1, 3, 5, 7, 9
```

### 8.6 Nested Loops

```til
# Multiplication table
for i in 1..=10
    for j in 1..=10
        print(i * j)
    print("---")

# With labeled breaks (future feature)
# outer: for i in 0..10
#     for j in 0..10
#         if condition
#             break outer
```

---

## 9. Functions

### 9.1 Function Declaration

TIL supports two syntaxes for function declaration:

```til
# Python-style (preferred)
add(a: int, b: int) -> int
    return a + b

# With fn keyword
fn multiply(a: int, b: int) -> int
    return a * b
```

### 9.2 Parameters

#### Required Parameters
```til
greet(name: str, age: int)
    print("Hello, " + name)
    print("You are " + age + " years old")
```

#### Default Parameters
```til
greet(name: str, greeting: str = "Hello")
    print(greeting + ", " + name + "!")

greet("World")              # "Hello, World!"
greet("World", "Hi")        # "Hi, World!"
```

#### Type Annotations
```til
# Explicit types
process(data: int[], size: int) -> bool
    # ...

# Generic-style (future)
# process<T>(data: T[], size: int) -> T
```

### 9.3 Return Values

```til
# Single return value
square(x: int) -> int
    return x * x

# No return value (void)
log_message(msg: str)
    print("[LOG] " + msg)
    # No return needed

# Early return
find_first_positive(arr: int[]) -> int
    for num in arr
        if num > 0
            return num
    return -1  # Not found
```

### 9.4 Recursive Functions

```til
factorial(n: int) -> int
    if n <= 1
        return 1
    return n * factorial(n - 1)

fibonacci(n: int) -> int
    if n <= 1
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Tail-recursive (optimized at Level 0/1)
#[level: 1]
factorial_tail(n: int, acc: int = 1) -> int
    if n <= 1
        return acc
    return factorial_tail(n - 1, n * acc)
```

### 9.5 Function Overloading

TIL doesn't support traditional overloading, but you can use default parameters:

```til
# Instead of overloading
format_number(n: int, decimals: int = 2, prefix: str = "") -> str
    # Format implementation
    return prefix + str(n)

# Call variations
format_number(42)
format_number(42, 3)
format_number(42, 2, "$")
```

### 9.6 Higher-Order Functions (Future)

```til
# Planned feature
# apply(f: fn(int) -> int, x: int) -> int
#     return f(x)
# 
# let doubled = apply(fn(x) => x * 2, 5)
```

---

## 10. Structures

### 10.1 Structure Definition

```til
struct Person
    name: str
    age: int
    email: str
```

### 10.2 Structure with Default Values

```til
struct Config
    host: str = "localhost"
    port: int = 8080
    debug: bool = false
```

### 10.3 Creating Instances

```til
# Named fields
let person = Person {
    name: "Alice",
    age: 30,
    email: "alice@example.com"
}

# With defaults
let config = Config {
    port: 3000
    # host and debug use defaults
}

# Positional (if constructor defined)
let point = Point.new(3.0, 4.0)
```

### 10.4 Accessing Fields

```til
let person = Person { name: "Bob", age: 25, email: "bob@example.com" }

print(person.name)   # "Bob"
print(person.age)    # 25

# Modifying (if mutable)
var user = Person { name: "Charlie", age: 20, email: "c@example.com" }
user.age = 21
```

### 10.5 Nested Structures

```til
struct Address
    street: str
    city: str
    country: str

struct Company
    name: str
    address: Address
    employees: int

let company = Company {
    name: "TIL Corp",
    address: Address {
        street: "123 Main St",
        city: "Astana",
        country: "Kazakhstan"
    },
    employees: 100
}

print(company.address.city)  # "Astana"
```

---

## 11. Enumerations

### 11.1 Basic Enum

```til
enum Color
    Red
    Green
    Blue
```

### 11.2 Enum with Values

```til
enum HttpStatus
    OK = 200
    NotFound = 404
    InternalError = 500

enum Weekday
    Monday = 1
    Tuesday = 2
    Wednesday = 3
    Thursday = 4
    Friday = 5
    Saturday = 6
    Sunday = 7
```

### 11.3 Using Enums

```til
enum Direction
    North
    South
    East
    West

move(dir: Direction, steps: int)
    match dir
        Direction.North =>
            print("Moving north " + str(steps) + " steps")
        Direction.South =>
            print("Moving south " + str(steps) + " steps")
        Direction.East =>
            print("Moving east " + str(steps) + " steps")
        Direction.West =>
            print("Moving west " + str(steps) + " steps")

main()
    move(Direction.North, 10)
```

---

## 12. Implementation Blocks

### 12.1 Basic Impl Block

```til
struct Rectangle
    width: float
    height: float

impl Rectangle
    # Constructor
    new(w: float, h: float) -> Rectangle
        return Rectangle { width: w, height: h }
    
    # Methods with self
    area(self) -> float
        return self.width * self.height
    
    perimeter(self) -> float
        return 2.0 * (self.width + self.height)
    
    # Method that modifies self (requires mutable receiver)
    scale(self, factor: float)
        self.width = self.width * factor
        self.height = self.height * factor
```

### 12.2 Static Methods vs Instance Methods

```til
impl Circle
    # Static method (no self) - called as Circle.from_diameter(...)
    from_diameter(d: float) -> Circle
        return Circle { radius: d / 2.0 }
    
    # Instance method (with self) - called as circle.area()
    area(self) -> float
        return 3.14159 * self.radius ** 2
```

### 12.3 Multiple Impl Blocks

```til
struct Vector2
    x: float
    y: float

# Math operations
impl Vector2
    add(self, other: Vector2) -> Vector2
        return Vector2 { x: self.x + other.x, y: self.y + other.y }
    
    subtract(self, other: Vector2) -> Vector2
        return Vector2 { x: self.x - other.x, y: self.y - other.y }

# Utility methods
impl Vector2
    length(self) -> float
        return sqrt(self.x ** 2 + self.y ** 2)
    
    normalize(self) -> Vector2
        let len = self.length()
        return Vector2 { x: self.x / len, y: self.y / len }
```

### 12.4 Constructor Patterns

```til
struct Person
    name: str
    age: int

impl Person
    # Primary constructor
    new(name: str, age: int) -> Person
        return Person { name: name, age: age }
    
    # Alternative constructor
    from_name(name: str) -> Person
        return Person { name: name, age: 0 }
    
    # Default constructor
    default() -> Person
        return Person { name: "Unknown", age: 0 }

main()
    let p1 = Person.new("Alice", 30)
    let p2 = Person.from_name("Bob")
    let p3 = Person.default()
```

---

## 13. Pattern Matching

### 13.1 Match Expression

```til
match value
    pattern1 =>
        # code for pattern1
    pattern2 =>
        # code for pattern2
    _ =>
        # default case (wildcard)
```

### 13.2 Matching Values

```til
describe_number(n: int) -> str
    match n
        0 =>
            return "zero"
        1 =>
            return "one"
        2 =>
            return "two"
        _ =>
            return "many"
```

### 13.3 Matching with Enums

```til
enum Result
    Success
    Error
    Pending

handle_result(r: Result)
    match r
        Result.Success =>
            print("Operation succeeded!")
        Result.Error =>
            print("Operation failed!")
        Result.Pending =>
            print("Operation in progress...")
```

### 13.4 Matching Ranges (Future)

```til
# Planned feature
# grade(score: int) -> str
#     match score
#         90..=100 =>
#             return "A"
#         80..89 =>
#             return "B"
#         70..79 =>
#             return "C"
#         _ =>
#             return "F"
```

---

## 14. Built-in Functions

### 14.1 Input/Output

| Function | Description | Example |
|----------|-------------|---------|
| `print(value)` | Print to stdout with newline | `print("Hello")` |
| `input(prompt)` | Read line from stdin | `let name = input("Name: ")` |

```til
main()
    print("Hello, World!")
    print(42)
    print(3.14)
    print(true)
    
    let name = input("Enter your name: ")
    print("Hello, " + name + "!")
```

### 14.2 Math Functions

| Function | Description | Example |
|----------|-------------|---------|
| `sqrt(x)` | Square root | `sqrt(16.0)` → `4.0` |
| `abs(x)` | Absolute value | `abs(-5)` → `5` |
| `pow(base, exp)` | Power | `pow(2, 10)` → `1024` |
| `sin(x)` | Sine (radians) | `sin(0.0)` → `0.0` |
| `cos(x)` | Cosine (radians) | `cos(0.0)` → `1.0` |
| `tan(x)` | Tangent (radians) | `tan(0.0)` → `0.0` |
| `log(x)` | Natural logarithm | `log(2.718...)` → `1.0` |
| `exp(x)` | e^x | `exp(1.0)` → `2.718...` |
| `floor(x)` | Round down | `floor(3.7)` → `3` |
| `ceil(x)` | Round up | `ceil(3.2)` → `4` |
| `round(x)` | Round to nearest | `round(3.5)` → `4` |
| `min(a, b)` | Minimum | `min(3, 7)` → `3` |
| `max(a, b)` | Maximum | `max(3, 7)` → `7` |

```til
main()
    print(sqrt(2.0))      # 1.414...
    print(pow(2, 8))      # 256
    print(abs(-42))       # 42
    
    let angle = 3.14159 / 4  # 45 degrees
    print(sin(angle))     # 0.707...
    print(cos(angle))     # 0.707...
    
    print(min(10, 20))    # 10
    print(max(10, 20))    # 20
```

### 14.3 String Functions

| Function | Description | Example |
|----------|-------------|---------|
| `len(s)` | String length | `len("hello")` → `5` |

```til
main()
    let text = "Hello, TIL!"
    print(len(text))  # 11
```

### 14.4 Type Functions (Future)

| Function | Description |
|----------|-------------|
| `typeof(x)` | Get type name as string |
| `sizeof(T)` | Get size of type in bytes |

---

## 15. Attributes

### 15.1 Attribute Syntax

Attributes are written as `#[name: value]` or `#[name]`:

```til
#[level: 0]
#[inline]
fast_function()
    # ...
```

### 15.2 Available Attributes

| Attribute | Description | Usage |
|-----------|-------------|-------|
| `#[level: N]` | Set abstraction level (0-3) | `#[level: 0]` |
| `#[inline]` | Suggest inlining | `#[inline]` |
| `#[noinline]` | Prevent inlining | `#[noinline]` |
| `#[pure]` | Mark as pure function | `#[pure]` |
| `#[unsafe]` | Allow unsafe operations | `#[unsafe]` |
| `#[extern]` | External linkage | `#[extern]` |
| `#[packed]` | Pack struct without padding | `#[packed]` |

### 15.3 Level Attribute

```til
#[level: 0]  # Hardware level
ultra_fast()
    # SIMD, always inlined

#[level: 1]  # Systems level
system_call()
    # C-like, optional inline

#[level: 2]  # Safe level (default)
safe_function()
    # Bounds checking, type safety

#[level: 3]  # Script level
quick_prototype()
    # Maximum productivity
```

### 15.4 Combining Attributes

```til
#[level: 0]
#[inline]
#[pure]
compute_hash(data: u8[], len: int) -> u64
    # Ultra-fast, inlined, no side effects
    var hash: u64 = 0
    for i in 0..len
        hash = hash * 31 + data[i] as u64
    return hash
```

---

## 16. Memory Model

### 16.1 Stack vs Heap

TIL uses a simple memory model:

- **Primitives**: Stored on stack
- **Structs**: Stored on stack (value types)
- **Strings**: Pointer on stack, data on heap
- **Arrays**: Fixed on stack, dynamic on heap

```til
main()
    let x = 42              # Stack
    let point = Point { x: 1.0, y: 2.0 }  # Stack
    let name = "TIL"        # Pointer on stack, "TIL" on heap
    let arr = [1, 2, 3]     # Fixed array on stack
```

### 16.2 Value Semantics

Structs in TIL have value semantics (copied on assignment):

```til
main()
    let p1 = Point { x: 1.0, y: 2.0 }
    let p2 = p1  # Copy of p1
    
    # p1 and p2 are independent
```

### 16.3 Pointers (Level 0/1)

At lower levels, pointers are available:

```til
#[level: 1]
swap(a: int*, b: int*)
    let temp = *a
    *a = *b
    *b = temp

#[level: 1]
main()
    var x = 10
    var y = 20
    swap(&x, &y)
    print(x)  # 20
    print(y)  # 10
```

---

## 17. Error Handling

### 17.1 Current Approach

TIL currently uses return values for error handling:

```til
# Return -1 for error
find_index(arr: int[], target: int) -> int
    for i in 0..len(arr)
        if arr[i] == target
            return i
    return -1  # Not found

# Return None/null
find_user(id: int) -> User
    # Returns null if not found
    # ...
```

### 17.2 Future: Result Type

Planned Result type for explicit error handling:

```til
# Future feature
# enum Result<T, E>
#     Ok(T)
#     Err(E)
# 
# read_file(path: str) -> Result<str, Error>
#     # ...
# 
# main()
#     match read_file("data.txt")
#         Ok(content) =>
#             print(content)
#         Err(e) =>
#             print("Error: " + e.message)
```

---

## 18. Standard Library

### 18.1 Core Module (Built-in)

Always available:
- `print`, `input`
- `sqrt`, `abs`, `pow`, `sin`, `cos`, `tan`, `log`, `exp`
- `floor`, `ceil`, `round`
- `min`, `max`
- `len`

### 18.2 Future Modules

Planned standard library modules:

| Module | Contents |
|--------|----------|
| `std.io` | File I/O, streams |
| `std.collections` | Vec, Map, Set |
| `std.string` | String utilities |
| `std.math` | Extended math |
| `std.time` | Date/time handling |
| `std.net` | Networking |
| `std.json` | JSON parsing |

---

## 19. Interoperability

### 19.1 C Code Generation

TIL compiles to readable C code:

```til
# TIL code
add(a: int, b: int) -> int
    return a + b
```

Generates:

```c
// Generated C
int64_t til_add(int64_t a, int64_t b) {
    return (a + b);
}
```

### 19.2 Viewing Generated C

```bash
til build program.til -c  # Output C code only
til build program.til --keep-c  # Keep .c file after compilation
```

### 19.3 Calling C Libraries (Future)

```til
# Planned feature
# #[extern "C"]
# fn printf(format: str, ...) -> int
# 
# main()
#     printf("Hello from C! %d\n", 42)
```

---

## 20. Best Practices

### 20.1 Code Style

**Naming Conventions**:
```til
# Variables and functions: snake_case
let user_name = "Alice"
calculate_total(items)

# Types and structs: PascalCase
struct UserProfile
enum HttpStatus

# Constants: UPPER_SNAKE_CASE
const MAX_CONNECTIONS = 100
```

**Indentation**:
- Use 4 spaces (not tabs)
- Be consistent

**Comments**:
```til
# Good: Explain WHY, not WHAT
# Calculate tax using the 2024 Kazakhstan tax brackets
let tax = calculate_tax(income)

# Bad: Obvious comments
# Add a and b together
let sum = a + b
```

### 20.2 Level Selection

| Use Level | When |
|-----------|------|
| 0 | Inner loops, SIMD, math kernels |
| 1 | Performance-critical algorithms |
| 2 | Most application code (default) |
| 3 | Prototyping, scripts, glue code |

### 20.3 Performance Tips

1. **Start at Level 2**, optimize only if needed
2. **Profile before optimizing** - measure, don't guess
3. **Use Level 0/1 only for hot paths**
4. **Avoid string concatenation in loops**

```til
# Bad
var result = ""
for i in 0..1000
    result = result + str(i)

# Better (when builder is available)
# var builder = StringBuilder.new()
# for i in 0..1000
#     builder.append(str(i))
# let result = builder.to_string()
```

### 20.4 Project Structure

```
my_project/
├── src/
│   ├── main.til         # Entry point
│   ├── models.til       # Data structures
│   ├── utils.til        # Utilities
│   └── math/
│       ├── vector.til
│       └── matrix.til
├── tests/
│   └── test_math.til
├── README.md
└── til.toml             # Project config (future)
```

---

## 21. Complete Examples

### 21.1 FizzBuzz

```til
# FizzBuzz in TIL
# Author: Alisher Beisembekov

main()
    for i in 1..=100
        if i % 15 == 0
            print("FizzBuzz")
        elif i % 3 == 0
            print("Fizz")
        elif i % 5 == 0
            print("Buzz")
        else
            print(i)
```

### 21.2 Binary Search

```til
# Binary Search in TIL
# Author: Alisher Beisembekov

#[level: 1]
binary_search(arr: int[], target: int, low: int, high: int) -> int
    if low > high
        return -1
    
    let mid = (low + high) / 2
    
    if arr[mid] == target
        return mid
    elif arr[mid] > target
        return binary_search(arr, target, low, mid - 1)
    else
        return binary_search(arr, target, mid + 1, high)

main()
    let numbers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    let target = 7
    
    let index = binary_search(numbers, target, 0, 9)
    
    if index >= 0
        print("Found at index:")
        print(index)
    else
        print("Not found")
```

### 21.3 Linked List

```til
# Linked List in TIL
# Author: Alisher Beisembekov

struct Node
    value: int
    next: Node  # Pointer to next node (or null)

struct LinkedList
    head: Node
    size: int

impl LinkedList
    new() -> LinkedList
        return LinkedList { head: None, size: 0 }
    
    push(self, value: int)
        let node = Node { value: value, next: self.head }
        self.head = node
        self.size = self.size + 1
    
    print_all(self)
        var current = self.head
        while current != None
            print(current.value)
            current = current.next

main()
    var list = LinkedList.new()
    
    list.push(10)
    list.push(20)
    list.push(30)
    
    list.print_all()  # 30, 20, 10
```

### 21.4 Simple Game Loop

```til
# Simple Game Loop
# Author: Alisher Beisembekov

struct Player
    x: float
    y: float
    health: int

struct Game
    player: Player
    running: bool
    score: int

impl Game
    new() -> Game
        return Game {
            player: Player { x: 0.0, y: 0.0, health: 100 },
            running: true,
            score: 0
        }
    
    update(self)
        # Update game state
        self.score = self.score + 1
        
        # Check win/lose conditions
        if self.player.health <= 0
            self.running = false
            print("Game Over!")
        
        if self.score >= 100
            self.running = false
            print("You Win!")
    
    render(self)
        print("Score: " + str(self.score))
        print("Health: " + str(self.player.health))
        print("Position: (" + str(self.player.x) + ", " + str(self.player.y) + ")")
        print("---")

main()
    var game = Game.new()
    
    # Game loop
    while game.running
        game.update()
        game.render()
        
        # Simulate some damage
        game.player.health = game.player.health - 1
```

### 21.5 Multi-Level Example

```til
# Multi-Level Programming Demo
# Author: Alisher Beisembekov
# 
# This example shows all 4 levels working together

# ═══════════════════════════════════════════════════════════
# LEVEL 0: Hardware - Maximum Performance
# ═══════════════════════════════════════════════════════════

#[level: 0]
#[inline]
fast_distance(x1: float, y1: float, x2: float, y2: float) -> float
    let dx = x2 - x1
    let dy = y2 - y1
    return sqrt(dx * dx + dy * dy)

#[level: 0]
dot_product(ax: float, ay: float, bx: float, by: float) -> float
    return ax * bx + ay * by

# ═══════════════════════════════════════════════════════════
# LEVEL 1: Systems - C-like Control
# ═══════════════════════════════════════════════════════════

#[level: 1]
quicksort(arr: int[], low: int, high: int)
    if low < high
        let pivot = partition(arr, low, high)
        quicksort(arr, low, pivot - 1)
        quicksort(arr, pivot + 1, high)

#[level: 1]
partition(arr: int[], low: int, high: int) -> int
    let pivot = arr[high]
    var i = low - 1
    
    for j in low..high
        if arr[j] <= pivot
            i = i + 1
            # Swap arr[i] and arr[j]
            let temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
    
    # Swap arr[i+1] and arr[high]
    let temp = arr[i + 1]
    arr[i + 1] = arr[high]
    arr[high] = temp
    
    return i + 1

# ═══════════════════════════════════════════════════════════
# LEVEL 2: Safe - Most Application Code
# ═══════════════════════════════════════════════════════════

struct Particle
    x: float
    y: float
    vx: float
    vy: float
    mass: float

impl Particle
    new(x: float, y: float) -> Particle
        return Particle {
            x: x,
            y: y,
            vx: 0.0,
            vy: 0.0,
            mass: 1.0
        }
    
    update(self, dt: float)
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
    
    distance_to(self, other: Particle) -> float
        # Call Level 0 function for performance
        return fast_distance(self.x, self.y, other.x, other.y)
    
    apply_force(self, fx: float, fy: float)
        self.vx = self.vx + fx / self.mass
        self.vy = self.vy + fy / self.mass

struct Simulation
    particles: Particle[]
    time: float

impl Simulation
    new() -> Simulation
        return Simulation { particles: [], time: 0.0 }
    
    add_particle(self, p: Particle)
        self.particles.append(p)
    
    step(self, dt: float)
        for p in self.particles
            p.update(dt)
        self.time = self.time + dt
    
    print_state(self)
        print("Time: " + str(self.time))
        for i in 0..len(self.particles)
            let p = self.particles[i]
            print("Particle " + str(i) + ": (" + str(p.x) + ", " + str(p.y) + ")")

# ═══════════════════════════════════════════════════════════
# LEVEL 3: Script - Easy Prototyping
# ═══════════════════════════════════════════════════════════

#[level: 3]
create_random_particles(count)
    let particles = []
    for i in 0..count
        let x = i * 10.0
        let y = i * 5.0
        particles.append(Particle.new(x, y))
    return particles

#[level: 3]
run_demo()
    print("=== TIL Multi-Level Demo ===")
    print("Author: Alisher Beisembekov")
    print("")
    
    # Use Level 0 math
    let dist = fast_distance(0.0, 0.0, 3.0, 4.0)
    print("Level 0 - Distance: " + str(dist))
    
    # Use Level 1 sorting
    var numbers = [64, 34, 25, 12, 22, 11, 90]
    quicksort(numbers, 0, 6)
    print("Level 1 - Sorted: ")
    for n in numbers
        print(n)
    
    # Use Level 2 simulation
    var sim = Simulation.new()
    let particles = create_random_particles(3)
    for p in particles
        sim.add_particle(p)
        p.apply_force(1.0, 0.5)
    
    print("")
    print("Level 2 - Simulation:")
    for step in 0..5
        sim.step(0.1)
    sim.print_state()
    
    print("")
    print("=== Demo Complete ===")

# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

main()
    run_demo()
```

---

## 22. Grammar Reference

### 22.1 EBNF Grammar

```ebnf
program         = { statement } ;

statement       = function_def
                | struct_def
                | enum_def
                | impl_block
                | variable_decl
                | assignment
                | if_stmt
                | for_stmt
                | while_stmt
                | loop_stmt
                | match_stmt
                | return_stmt
                | break_stmt
                | continue_stmt
                | expr_stmt
                ;

function_def    = { attribute } [ "fn" ] IDENT "(" [ params ] ")" [ "->" type ] NEWLINE block ;

params          = param { "," param } ;
param           = IDENT ":" type [ "=" expr ] ;

struct_def      = "struct" IDENT NEWLINE INDENT { field } DEDENT ;
field           = IDENT ":" type [ "=" expr ] NEWLINE ;

enum_def        = "enum" IDENT NEWLINE INDENT { variant } DEDENT ;
variant         = IDENT [ "=" expr ] NEWLINE ;

impl_block      = "impl" IDENT [ "for" IDENT ] NEWLINE INDENT { method_def } DEDENT ;
method_def      = { attribute } [ "fn" ] IDENT "(" [ params ] ")" [ "->" type ] NEWLINE block ;

variable_decl   = ( "let" | "var" | "const" ) IDENT [ ":" type ] [ "=" expr ] ;
assignment      = expr ( "=" | "+=" | "-=" | "*=" | "/=" ) expr ;

if_stmt         = "if" expr NEWLINE block { "elif" expr NEWLINE block } [ "else" NEWLINE block ] ;
for_stmt        = "for" IDENT "in" expr NEWLINE block ;
while_stmt      = "while" expr NEWLINE block ;
loop_stmt       = "loop" NEWLINE block ;
match_stmt      = "match" expr NEWLINE INDENT { match_arm } DEDENT ;
match_arm       = expr "=>" ( block | expr ) ;

return_stmt     = "return" [ expr ] ;
break_stmt      = "break" ;
continue_stmt   = "continue" ;
expr_stmt       = expr ;

block           = INDENT { statement NEWLINE } DEDENT ;

type            = IDENT [ "[" type "]" ] ;

expr            = or_expr ;
or_expr         = and_expr { "or" and_expr } ;
and_expr        = not_expr { "and" not_expr } ;
not_expr        = "not" not_expr | comparison ;
comparison      = bitor_expr { ( "==" | "!=" | "<" | ">" | "<=" | ">=" ) bitor_expr } ;
bitor_expr      = xor_expr { "|" xor_expr } ;
xor_expr        = bitand_expr { "^" bitand_expr } ;
bitand_expr     = shift_expr { "&" shift_expr } ;
shift_expr      = add_expr { ( "<<" | ">>" ) add_expr } ;
add_expr        = mul_expr { ( "+" | "-" ) mul_expr } ;
mul_expr        = pow_expr { ( "*" | "/" | "%" ) pow_expr } ;
pow_expr        = unary_expr [ "**" pow_expr ] ;
unary_expr      = ( "-" | "~" | "&" | "*" | "not" ) unary_expr | postfix_expr ;
postfix_expr    = primary { call | index | attr | range | cast } ;
call            = "(" [ args ] ")" ;
index           = "[" expr "]" ;
attr            = "." IDENT ;
range           = ( ".." | "..=" ) expr ;
cast            = "as" type ;
args            = expr { "," expr } ;
primary         = INT | FLOAT | STRING | "true" | "false" | "None" | IDENT | "(" expr ")" | array_lit | struct_lit ;
array_lit       = "[" [ args ] "]" ;
struct_lit      = IDENT "{" [ field_init { "," field_init } ] "}" ;
field_init      = IDENT ":" expr ;

attribute       = "#[" IDENT [ ":" value ] "]" ;
```

---

## Appendix A: Reserved Words

```
and       as        break     const     continue  elif
else      enum      false     False     fn        for
if        impl      in        let       loop      match
mut       None      not       or        pub       return
self      struct    trait     true      True      type
var       while
```

## Appendix B: Operator Summary

| Category | Operators |
|----------|-----------|
| Arithmetic | `+` `-` `*` `/` `%` `**` |
| Comparison | `==` `!=` `<` `>` `<=` `>=` |
| Logical | `and` `or` `not` |
| Bitwise | `&` `\|` `^` `~` `<<` `>>` |
| Assignment | `=` `+=` `-=` `*=` `/=` |
| Range | `..` `..=` |
| Other | `->` `=>` `.` `:` `as` |

---

<div align="center">

**TIL Language Reference v2.0**

**Author: Alisher Beisembekov**

*"Проще Python. Быстрее C. Умнее всех."*

© 2025 TIL Language. MIT License.

</div>
