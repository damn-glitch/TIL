" TIL Syntax Highlighting for Vim
" Author: Alisher Beisembekov
" Language: TIL
" Filetype: til

if exists("b:current_syntax")
    finish
endif

" Keywords
syn keyword tilKeyword if else elif for while loop in return
syn keyword tilKeyword break continue fn let var const mut
syn keyword tilKeyword struct enum impl trait match type pub as
syn keyword tilKeyword and or not self import from

" Types
syn keyword tilType int float str bool char void
syn keyword tilType i8 i16 i32 i64 u8 u16 u32 u64 f32 f64

" Booleans
syn keyword tilBoolean true false True False None

" Built-in functions
syn keyword tilBuiltin print input sqrt abs pow sin cos tan
syn keyword tilBuiltin log exp floor ceil round min max len

" Numbers
syn match tilNumber "\<\d\+\>"
syn match tilNumber "\<0x[0-9a-fA-F]\+\>"
syn match tilNumber "\<0b[01]\+\>"
syn match tilNumber "\<0o[0-7]\+\>"
syn match tilFloat "\<\d\+\.\d*\>"
syn match tilFloat "\<\d\+[eE][+-]\?\d\+\>"

" Strings
syn region tilString start='"' end='"' skip='\\"' contains=tilEscape
syn region tilString start="'" end="'" skip="\\'" contains=tilEscape
syn match tilEscape "\\." contained

" Comments
syn match tilComment "#.*$" contains=tilTodo
syn keyword tilTodo TODO FIXME XXX NOTE contained

" Attributes
syn region tilAttribute start="#\[" end="\]"

" Operators
syn match tilOperator "[-+*/%&|^~<>=!]"
syn match tilOperator "\.\."
syn match tilOperator "->"
syn match tilOperator "=>"

" Functions
syn match tilFunction "\<\w\+\>\s*("me=e-1

" Types (PascalCase)
syn match tilTypeName "\<[A-Z][a-zA-Z0-9_]*\>"

" Highlighting
hi def link tilKeyword Keyword
hi def link tilType Type
hi def link tilBoolean Boolean
hi def link tilBuiltin Function
hi def link tilNumber Number
hi def link tilFloat Float
hi def link tilString String
hi def link tilEscape Special
hi def link tilComment Comment
hi def link tilTodo Todo
hi def link tilAttribute PreProc
hi def link tilOperator Operator
hi def link tilFunction Function
hi def link tilTypeName Type

let b:current_syntax = "til"
