#!/usr/bin/env python3
"""Package TIL bytecode as a TI-84 Plus CE AppVar (.8xv) for the TILVM program.

    python3 tbc2appvar.py program.tbc NAME [-o NAME.8xv] [--ram]
    python3 tbc2appvar.py program.til NAME [--strip]     (compiles the TIL source first)

NAME: 1-8 letters/digits, starting with a letter (case is kept). The AppVar is
marked "archived" unless --ram is given. TILVM lists every AppVar whose data
starts with the bytecode magic "TILB".

File format (TI-83 Plus family variable file, as used by TI Connect CE):
55-byte header ("**TI83F*", 1A 0A 00, 42-byte comment, data length), one
variable entry (13-byte entry header: 0D 00, data length, type 0x15 AppVar,
8-byte name, version, archive flag, data length again; then the data: u16
size + bytes) and a 16-bit checksum of the entry.
"""
import os
import re
import struct
import sys

APPVAR = 0x15
SIGNATURE = b'**TI83F*\x1a\x0a\x00'
MAX_DATA = 65500                     # TI-OS variables are limited to 64 KB
NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9]{0,7}$')


def make_appvar(name, data, archived=True, comment='TIL bytecode'):
    """Bytes of a .8xv file holding `data` in the AppVar `name`."""
    if not NAME_RE.match(name):
        raise ValueError(f"invalid AppVar name {name!r}: 1-8 letters/digits, starting with a letter")
    if len(data) > MAX_DATA:
        raise ValueError(f"{len(data)} bytes: too large for an AppVar (max {MAX_DATA})")
    body = struct.pack('<H', len(data)) + data                     # AppVar contents: size + bytes
    entry = struct.pack('<HHB8sBBH', 13, len(body), APPVAR, name.encode('ascii'),
                        0, 0x80 if archived else 0, len(body)) + body
    header = SIGNATURE + comment.encode('ascii')[:42].ljust(42, b'\0') + struct.pack('<H', len(entry))
    return header + entry + struct.pack('<H', sum(entry) & 0xFFFF)


def read_appvar(blob):
    """(name, data, archived) of the first AppVar in a .8xv file; checks the format."""
    if blob[:11] != SIGNATURE or len(blob) < 57:
        raise ValueError("not a TI-83 Plus family variable file")
    (size,) = struct.unpack_from('<H', blob, 53)
    entry = blob[55:55 + size]
    if len(entry) != size or len(blob) < 57 + size:
        raise ValueError("truncated file")
    (checksum,) = struct.unpack_from('<H', blob, 55 + size)
    if checksum != sum(entry) & 0xFFFF:
        raise ValueError("bad checksum")
    hlen, dlen, vtype, raw_name, _version, flag, dlen2 = struct.unpack_from('<HHB8sBBH', entry, 0)
    if hlen != 13 or vtype != APPVAR or dlen != dlen2 or len(entry) != 17 + dlen:
        raise ValueError("not an AppVar entry")
    (n,) = struct.unpack_from('<H', entry, 17)
    if n + 2 != dlen:
        raise ValueError("bad AppVar size")
    return raw_name.rstrip(b'\0').decode('ascii'), entry[19:19 + n], bool(flag & 0x80)


def main(argv):
    if len(argv) < 2 or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0 if argv and argv[0] in ('-h', '--help') else 2
    src, name = argv[0], argv[1]
    out = None
    if '-o' in argv:
        out = argv[argv.index('-o') + 1]
    if src.endswith('.til'):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
        import til  # noqa: E402
        with open(src, encoding='utf-8') as f:
            data = til.compile_to_bytecode(f.read(), src).serialize(strip='--strip' in argv)
    else:
        with open(src, 'rb') as f:
            data = f.read()
    if data[:4] != b'TILB':
        print(f"{src}: not TIL bytecode", file=sys.stderr)
        return 1
    try:
        blob = make_appvar(name, data, archived='--ram' not in argv)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = out or name + '.8xv'
    with open(out, 'wb') as f:
        f.write(blob)
    print(f"{out}: AppVar {name}, {len(data)} bytes of bytecode")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
