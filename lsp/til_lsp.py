#!/usr/bin/env python3
"""
TIL Language Server Protocol (LSP) Implementation

This server provides IDE features for TIL:
- Diagnostics (errors, warnings)
- Autocomplete
- Hover information
- Go to definition
- Find references
- Document symbols
- Formatting

Author: Alisher Beisembekov
"""

import sys
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum

# Configure logging
logging.basicConfig(
    filename='/tmp/til-lsp.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('til-lsp')

# ═══════════════════════════════════════════════════════════════════════════════
#                              LSP Protocol Types
# ═══════════════════════════════════════════════════════════════════════════════

class DiagnosticSeverity(IntEnum):
    Error = 1
    Warning = 2
    Information = 3
    Hint = 4

class CompletionItemKind(IntEnum):
    Text = 1
    Method = 2
    Function = 3
    Constructor = 4
    Field = 5
    Variable = 6
    Class = 7
    Interface = 8
    Module = 9
    Property = 10
    Unit = 11
    Value = 12
    Enum = 13
    Keyword = 14
    Snippet = 15
    Color = 16
    File = 17
    Reference = 18
    Folder = 19
    EnumMember = 20
    Constant = 21
    Struct = 22
    Event = 23
    Operator = 24
    TypeParameter = 25

class SymbolKind(IntEnum):
    File = 1
    Module = 2
    Namespace = 3
    Package = 4
    Class = 5
    Method = 6
    Property = 7
    Field = 8
    Constructor = 9
    Enum = 10
    Interface = 11
    Function = 12
    Variable = 13
    Constant = 14
    String = 15
    Number = 16
    Boolean = 17
    Array = 18
    Object = 19
    Key = 20
    Null = 21
    EnumMember = 22
    Struct = 23
    Event = 24
    Operator = 25
    TypeParameter = 26

@dataclass
class Position:
    line: int
    character: int

@dataclass
class Range:
    start: Position
    end: Position

@dataclass
class Location:
    uri: str
    range: Range

@dataclass
class Diagnostic:
    range: Range
    message: str
    severity: int = DiagnosticSeverity.Error
    source: str = "til"

@dataclass
class CompletionItem:
    label: str
    kind: int
    detail: Optional[str] = None
    documentation: Optional[str] = None
    insertText: Optional[str] = None

@dataclass 
class Symbol:
    name: str
    kind: int
    range: Range
    selectionRange: Range
    children: List['Symbol'] = field(default_factory=list)

# ═══════════════════════════════════════════════════════════════════════════════
#                              TIL Analyzer
# ═══════════════════════════════════════════════════════════════════════════════

class TILAnalyzer:
    """Analyzes TIL source code for IDE features"""
    
    KEYWORDS = [
        'if', 'else', 'elif', 'for', 'while', 'loop', 'in', 'return',
        'break', 'continue', 'fn', 'let', 'var', 'const', 'mut',
        'struct', 'enum', 'impl', 'trait', 'match', 'type', 'pub',
        'as', 'and', 'or', 'not', 'true', 'false', 'True', 'False',
        'self', 'None'
    ]
    
    TYPES = [
        'int', 'float', 'str', 'bool', 'char', 'void',
        'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64',
        'f32', 'f64', 'Array', 'Option', 'Result'
    ]
    
    BUILTINS = {
        'print': ('print(value)', 'Print value(s) to stdout'),
        'sqrt': ('sqrt(x: float) -> float', 'Square root'),
        'abs': ('abs(x) -> number', 'Absolute value'),
        'pow': ('pow(base, exp) -> float', 'Power/exponentiation'),
        'sin': ('sin(x: float) -> float', 'Sine (radians)'),
        'cos': ('cos(x: float) -> float', 'Cosine (radians)'),
        'tan': ('tan(x: float) -> float', 'Tangent (radians)'),
        'log': ('log(x: float) -> float', 'Natural logarithm'),
        'exp': ('exp(x: float) -> float', 'e^x'),
        'floor': ('floor(x: float) -> int', 'Round down'),
        'ceil': ('ceil(x: float) -> int', 'Round up'),
        'round': ('round(x: float) -> int', 'Round to nearest'),
        'min': ('min(a, b) -> number', 'Minimum of two values'),
        'max': ('max(a, b) -> number', 'Maximum of two values'),
        'len': ('len(s: str) -> int', 'Length of string or array'),
        'input': ('input(prompt: str) -> str', 'Read line from stdin'),
    }
    
    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.symbols: Dict[str, List[Symbol]] = {}
        self.diagnostics: Dict[str, List[Diagnostic]] = {}
    
    def update_document(self, uri: str, content: str):
        """Update document content and re-analyze"""
        self.documents[uri] = content
        self.analyze(uri)
    
    def analyze(self, uri: str):
        """Analyze document for errors and symbols"""
        content = self.documents.get(uri, '')
        lines = content.split('\n')
        
        diagnostics = []
        symbols = []
        
        indent_stack = [0]
        in_struct = None
        in_impl = None
        
        for line_num, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check indentation consistency
            if stripped and not stripped.startswith('#['):
                if indent > indent_stack[-1] and indent != indent_stack[-1] + 4:
                    if indent % 4 != 0:
                        diagnostics.append(Diagnostic(
                            range=Range(
                                Position(line_num, 0),
                                Position(line_num, indent)
                            ),
                            message="Inconsistent indentation (use 4 spaces)",
                            severity=DiagnosticSeverity.Warning
                        ))
            
            # Track struct definitions
            struct_match = re.match(r'^struct\s+([A-Z]\w*)', stripped)
            if struct_match:
                name = struct_match.group(1)
                symbols.append(Symbol(
                    name=name,
                    kind=SymbolKind.Struct,
                    range=Range(Position(line_num, 0), Position(line_num, len(line))),
                    selectionRange=Range(Position(line_num, 7), Position(line_num, 7 + len(name)))
                ))
                in_struct = name
                continue
            
            # Track enum definitions
            enum_match = re.match(r'^enum\s+([A-Z]\w*)', stripped)
            if enum_match:
                name = enum_match.group(1)
                symbols.append(Symbol(
                    name=name,
                    kind=SymbolKind.Enum,
                    range=Range(Position(line_num, 0), Position(line_num, len(line))),
                    selectionRange=Range(Position(line_num, 5), Position(line_num, 5 + len(name)))
                ))
                continue
            
            # Track impl blocks
            impl_match = re.match(r'^impl\s+(\w+)', stripped)
            if impl_match:
                in_impl = impl_match.group(1)
                continue
            
            # Track function definitions
            fn_match = re.match(r'^(?:fn\s+)?([a-z_]\w*)\s*\(', stripped)
            if fn_match and not stripped.startswith('if') and not stripped.startswith('while'):
                name = fn_match.group(1)
                kind = SymbolKind.Method if in_impl else SymbolKind.Function
                symbols.append(Symbol(
                    name=name,
                    kind=kind,
                    range=Range(Position(line_num, 0), Position(line_num, len(line))),
                    selectionRange=Range(Position(line_num, indent), Position(line_num, indent + len(name)))
                ))
                continue
            
            # Check for common errors
            
            # Unclosed string
            if '"' in stripped:
                count = stripped.count('"') - stripped.count('\\"')
                if count % 2 != 0:
                    diagnostics.append(Diagnostic(
                        range=Range(Position(line_num, 0), Position(line_num, len(line))),
                        message="Unclosed string literal",
                        severity=DiagnosticSeverity.Error
                    ))
            
            # Missing colon after control flow
            for kw in ['if', 'elif', 'else', 'for', 'while', 'loop']:
                if stripped.startswith(kw) and not stripped.endswith(':') and '{' not in stripped:
                    # Check next line for indent
                    pass  # TIL uses indent, not colon
            
            # Unknown identifier warning (basic)
            words = re.findall(r'\b([a-zA-Z_]\w*)\b', stripped)
            for word in words:
                if word not in self.KEYWORDS and word not in self.TYPES and word not in self.BUILTINS:
                    if not word[0].isupper():  # Not a type name
                        pass  # Could add undefined variable check here
        
        self.diagnostics[uri] = diagnostics
        self.symbols[uri] = symbols
    
    def get_completions(self, uri: str, position: Position) -> List[CompletionItem]:
        """Get completion items at position"""
        content = self.documents.get(uri, '')
        lines = content.split('\n')
        
        if position.line >= len(lines):
            return []
        
        line = lines[position.line]
        prefix = line[:position.character]
        
        completions = []
        
        # Keywords
        for kw in self.KEYWORDS:
            completions.append(CompletionItem(
                label=kw,
                kind=CompletionItemKind.Keyword,
                detail="keyword"
            ))
        
        # Types
        for t in self.TYPES:
            completions.append(CompletionItem(
                label=t,
                kind=CompletionItemKind.Class,
                detail="type"
            ))
        
        # Built-in functions
        for name, (sig, doc) in self.BUILTINS.items():
            completions.append(CompletionItem(
                label=name,
                kind=CompletionItemKind.Function,
                detail=sig,
                documentation=doc
            ))
        
        # Symbols from current document
        for sym in self.symbols.get(uri, []):
            kind = CompletionItemKind.Function
            if sym.kind == SymbolKind.Struct:
                kind = CompletionItemKind.Struct
            elif sym.kind == SymbolKind.Enum:
                kind = CompletionItemKind.Enum
            elif sym.kind == SymbolKind.Method:
                kind = CompletionItemKind.Method
            
            completions.append(CompletionItem(
                label=sym.name,
                kind=kind,
                detail="local"
            ))
        
        return completions
    
    def get_hover(self, uri: str, position: Position) -> Optional[str]:
        """Get hover information at position"""
        content = self.documents.get(uri, '')
        lines = content.split('\n')
        
        if position.line >= len(lines):
            return None
        
        line = lines[position.line]
        
        # Find word at position
        word_match = re.search(r'\b\w+\b', line[max(0, position.character-20):position.character+20])
        if not word_match:
            return None
        
        # Adjust for substring search
        start = max(0, position.character - 20)
        word_start = start + word_match.start()
        word_end = start + word_match.end()
        
        if not (word_start <= position.character <= word_end):
            return None
        
        word = word_match.group()
        
        # Check built-ins
        if word in self.BUILTINS:
            sig, doc = self.BUILTINS[word]
            return f"```til\n{sig}\n```\n\n{doc}"
        
        # Check keywords
        keyword_docs = {
            'let': 'Declare an immutable variable',
            'var': 'Declare a mutable variable',
            'const': 'Declare a compile-time constant',
            'fn': 'Define a function',
            'struct': 'Define a structure',
            'impl': 'Implement methods for a type',
            'enum': 'Define an enumeration',
            'if': 'Conditional statement',
            'for': 'For loop with range',
            'while': 'While loop',
            'loop': 'Infinite loop',
            'match': 'Pattern matching',
            'return': 'Return from function',
            'break': 'Exit from loop',
            'continue': 'Skip to next iteration',
            'self': 'Reference to current instance',
        }
        
        if word in keyword_docs:
            return f"**{word}** (keyword)\n\n{keyword_docs[word]}"
        
        # Check types
        type_docs = {
            'int': '64-bit signed integer',
            'float': '64-bit floating point',
            'str': 'String type',
            'bool': 'Boolean (true/false)',
            'void': 'No return value',
        }
        
        if word in type_docs:
            return f"**{word}** (type)\n\n{type_docs[word]}"
        
        return None
    
    def get_definition(self, uri: str, position: Position) -> Optional[Location]:
        """Get definition location for symbol at position"""
        content = self.documents.get(uri, '')
        lines = content.split('\n')
        
        if position.line >= len(lines):
            return None
        
        line = lines[position.line]
        
        # Find word at position
        for match in re.finditer(r'\b(\w+)\b', line):
            if match.start() <= position.character <= match.end():
                word = match.group(1)
                
                # Search for definition in symbols
                for sym in self.symbols.get(uri, []):
                    if sym.name == word:
                        return Location(uri, sym.range)
                
                break
        
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#                              LSP Server
# ═══════════════════════════════════════════════════════════════════════════════

class TILLanguageServer:
    """TIL Language Server implementation"""
    
    def __init__(self):
        self.analyzer = TILAnalyzer()
        self.initialized = False
        self.shutdown_requested = False
    
    def handle_message(self, message: dict) -> Optional[dict]:
        """Handle incoming JSON-RPC message"""
        method = message.get('method', '')
        params = message.get('params', {})
        msg_id = message.get('id')
        
        logger.debug(f"Received: {method}")
        
        # Requests (have id)
        if msg_id is not None:
            result = self.handle_request(method, params)
            if result is not None:
                return {
                    'jsonrpc': '2.0',
                    'id': msg_id,
                    'result': result
                }
            return {
                'jsonrpc': '2.0',
                'id': msg_id,
                'result': None
            }
        
        # Notifications (no id)
        self.handle_notification(method, params)
        return None
    
    def handle_request(self, method: str, params: dict) -> Any:
        """Handle request methods"""
        
        if method == 'initialize':
            self.initialized = True
            return {
                'capabilities': {
                    'textDocumentSync': {
                        'openClose': True,
                        'change': 1,  # Full sync
                        'save': {'includeText': True}
                    },
                    'completionProvider': {
                        'triggerCharacters': ['.', ':', '(']
                    },
                    'hoverProvider': True,
                    'definitionProvider': True,
                    'referencesProvider': True,
                    'documentSymbolProvider': True,
                    'documentFormattingProvider': True,
                },
                'serverInfo': {
                    'name': 'til-language-server',
                    'version': '1.0.0'
                }
            }
        
        elif method == 'shutdown':
            self.shutdown_requested = True
            return None
        
        elif method == 'textDocument/completion':
            uri = params['textDocument']['uri']
            pos = Position(
                params['position']['line'],
                params['position']['character']
            )
            items = self.analyzer.get_completions(uri, pos)
            return {
                'isIncomplete': False,
                'items': [
                    {
                        'label': item.label,
                        'kind': item.kind,
                        'detail': item.detail,
                        'documentation': item.documentation,
                        'insertText': item.insertText or item.label
                    }
                    for item in items
                ]
            }
        
        elif method == 'textDocument/hover':
            uri = params['textDocument']['uri']
            pos = Position(
                params['position']['line'],
                params['position']['character']
            )
            content = self.analyzer.get_hover(uri, pos)
            if content:
                return {
                    'contents': {
                        'kind': 'markdown',
                        'value': content
                    }
                }
            return None
        
        elif method == 'textDocument/definition':
            uri = params['textDocument']['uri']
            pos = Position(
                params['position']['line'],
                params['position']['character']
            )
            location = self.analyzer.get_definition(uri, pos)
            if location:
                return {
                    'uri': location.uri,
                    'range': {
                        'start': {'line': location.range.start.line, 'character': location.range.start.character},
                        'end': {'line': location.range.end.line, 'character': location.range.end.character}
                    }
                }
            return None
        
        elif method == 'textDocument/documentSymbol':
            uri = params['textDocument']['uri']
            symbols = self.analyzer.symbols.get(uri, [])
            return [
                {
                    'name': sym.name,
                    'kind': sym.kind,
                    'range': {
                        'start': {'line': sym.range.start.line, 'character': sym.range.start.character},
                        'end': {'line': sym.range.end.line, 'character': sym.range.end.character}
                    },
                    'selectionRange': {
                        'start': {'line': sym.selectionRange.start.line, 'character': sym.selectionRange.start.character},
                        'end': {'line': sym.selectionRange.end.line, 'character': sym.selectionRange.end.character}
                    }
                }
                for sym in symbols
            ]
        
        return None
    
    def handle_notification(self, method: str, params: dict):
        """Handle notification methods"""
        
        if method == 'initialized':
            logger.info("Client initialized")
        
        elif method == 'exit':
            sys.exit(0 if self.shutdown_requested else 1)
        
        elif method == 'textDocument/didOpen':
            uri = params['textDocument']['uri']
            content = params['textDocument']['text']
            self.analyzer.update_document(uri, content)
            self.publish_diagnostics(uri)
        
        elif method == 'textDocument/didChange':
            uri = params['textDocument']['uri']
            # Full sync - take entire content
            for change in params.get('contentChanges', []):
                content = change.get('text', '')
                self.analyzer.update_document(uri, content)
            self.publish_diagnostics(uri)
        
        elif method == 'textDocument/didSave':
            uri = params['textDocument']['uri']
            content = params.get('text', self.analyzer.documents.get(uri, ''))
            self.analyzer.update_document(uri, content)
            self.publish_diagnostics(uri)
        
        elif method == 'textDocument/didClose':
            uri = params['textDocument']['uri']
            if uri in self.analyzer.documents:
                del self.analyzer.documents[uri]
    
    def publish_diagnostics(self, uri: str):
        """Send diagnostics to client"""
        diagnostics = self.analyzer.diagnostics.get(uri, [])
        
        notification = {
            'jsonrpc': '2.0',
            'method': 'textDocument/publishDiagnostics',
            'params': {
                'uri': uri,
                'diagnostics': [
                    {
                        'range': {
                            'start': {'line': d.range.start.line, 'character': d.range.start.character},
                            'end': {'line': d.range.end.line, 'character': d.range.end.character}
                        },
                        'message': d.message,
                        'severity': d.severity,
                        'source': d.source
                    }
                    for d in diagnostics
                ]
            }
        }
        
        self.send_message(notification)
    
    def send_message(self, message: dict):
        """Send JSON-RPC message to client"""
        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        sys.stdout.write(header + content)
        sys.stdout.flush()
    
    def run(self):
        """Main server loop"""
        logger.info("TIL Language Server starting...")
        
        while True:
            try:
                # Read header
                headers = {}
                while True:
                    line = sys.stdin.readline()
                    if line == '\r\n' or line == '\n':
                        break
                    if ':' in line:
                        key, value = line.split(':', 1)
                        headers[key.strip()] = value.strip()
                
                # Read content
                content_length = int(headers.get('Content-Length', 0))
                if content_length > 0:
                    content = sys.stdin.read(content_length)
                    message = json.loads(content)
                    
                    response = self.handle_message(message)
                    if response:
                        self.send_message(response)
                        
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)


def main():
    server = TILLanguageServer()
    server.run()


if __name__ == '__main__':
    main()
