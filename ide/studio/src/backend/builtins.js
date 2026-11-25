// Parameter names of the builtin functions (NATIVES of src/til.py; the
// arity comes from tables.json). Their documentation is in the UI
// catalogs (builtin.<name>), so hover texts follow the UI language.
export const BUILTIN_PARAMS = {
    print: ['...values'], println: ['...values'], len: ['x'], str: ['x'], int: ['x'], float: ['x'], bool: ['x'],
    abs: ['x'], sqrt: ['x'], pow: ['base', 'exp'], sin: ['x'], cos: ['x'], tan: ['x'], log: ['x'], exp: ['x'],
    floor: ['x'], ceil: ['x'], round: ['x'], min: ['a', '...rest'], max: ['a', '...rest'], sum: ['values'],
    input: ['prompt?'], chr: ['code'], ord: ['ch'], range: ['start', 'stop?', 'step?'], type_of: ['x'],
    format: ['fmt', 'value'], clock_ms: [], random: ['a?', 'b?'], seed: ['n'], sleep_ms: ['ms'], exit: ['code?'],
    panic: ['message?'], iter: ['x'], read_file: ['path'], write_file: ['path', 'text'], file_exists: ['path'],
    args: [], bytes_to_str: ['bytes'], str_bytes: ['s'], screen_init: ['width', 'height'], screen_clear: ['color?'],
    draw_pixel: ['x', 'y', 'color'], draw_line: ['x0', 'y0', 'x1', 'y1', 'color'], draw_rect: ['x', 'y', 'w', 'h', 'color'],
    fill_rect: ['x', 'y', 'w', 'h', 'color'], draw_circle: ['cx', 'cy', 'r', 'color'], draw_text: ['x', 'y', 'text', 'color'],
    get_pixel: ['x', 'y'], screen_show: [], key_pressed: [], wait_key: [], screen_width: [], screen_height: [], energy_mode: ['mode?'],
};

export const BUILTIN_RETURNS = {
    len: 'int', str: 'str', int: 'int', float: 'float', bool: 'bool', abs: 'num', sqrt: 'float', pow: 'num', sin: 'float', cos: 'float',
    tan: 'float', log: 'float', exp: 'float', floor: 'int', ceil: 'int', round: 'int', input: 'str', chr: 'str', ord: 'int',
    type_of: 'str', format: 'str', clock_ms: 'int', random: 'num', read_file: 'str', file_exists: 'bool', args: '[str]',
    bytes_to_str: 'str', str_bytes: '[int]', get_pixel: 'int', key_pressed: 'int', wait_key: 'int', screen_width: 'int', screen_height: 'int',
};

export function builtinSignature(name) {
    const ps = BUILTIN_PARAMS[name];
    if (!ps) return `${name}(...)`;
    const ret = BUILTIN_RETURNS[name];
    return `${name}(${ps.join(', ')})${ret ? ` -> ${ret}` : ''}`;
}

/* methods by receiver (for completion after "."), from METHODS of src/til.py */
export const METHOD_RECEIVERS = {
    str: ['len', 'contains', 'starts_with', 'ends_with', 'trim', 'to_upper', 'to_lower', 'replace', 'slice', 'find', 'split', 'repeat', 'to_int', 'to_float', 'is_empty', 'index_of', 'reverse', 'eq', 'to_string', 'count'],
    array: ['len', 'push', 'pop', 'get', 'set', 'contains', 'slice', 'is_empty', 'insert', 'remove', 'index_of', 'reverse', 'sort', 'join', 'first', 'last', 'clear', 'sum', 'min', 'max', 'copy', 'count', 'map', 'filter', 'reduce', 'each', 'any', 'all'],
    map: ['len', 'get', 'set', 'has', 'delete', 'keys', 'values', 'get_or', 'is_empty', 'clear', 'copy', 'remove', 'contains'],
    option: ['unwrap', 'expect', 'unwrap_or', 'is_some', 'is_none'],
    result: ['unwrap', 'expect', 'unwrap_or', 'is_ok', 'is_err', 'error'],
};

export const METHOD_PARAMS = {
    push: ['value'], get: ['index'], set: ['index', 'value'], contains: ['x'], starts_with: ['prefix'], ends_with: ['suffix'],
    replace: ['old', 'new'], slice: ['start', 'end?'], find: ['sub'], split: ['sep'], repeat: ['n'], insert: ['index', 'value'],
    remove: ['index'], index_of: ['x'], join: ['sep'], has: ['key'], delete: ['key'], get_or: ['key', 'default'], expect: ['message'],
    unwrap_or: ['default'], map: ['f'], filter: ['f'], reduce: ['f', 'init'], each: ['f'], any: ['f'], all: ['f'], count: ['x'],
};
