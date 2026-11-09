#!/usr/bin/env python3
"""Build a TIL program into the AVR VM firmware and run it on simavr; prints the
program's output (decoded from simavr's UART console)."""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
prog = os.path.abspath(sys.argv[1])
subprocess.run(['rm', '-f', os.path.join(HERE, 'build', 'program.h')])
b = subprocess.run(['make', '-s', '-C', HERE, f'PROGRAM={prog}'], capture_output=True, text=True)
if b.returncode != 0:
    sys.stderr.write(b.stdout + b.stderr)
    sys.exit(2)
r = subprocess.run(['simavr', '-m', 'atmega1284p', '-f', '16000000', os.path.join(HERE, 'build', 'tilvm.elf')],
                   capture_output=True, timeout=600)
text = r.stdout.decode('utf-8', 'replace') + r.stderr.decode('utf-8', 'replace')
lines = []
clean = re.sub(r'\x1b\[[0-9;]*m', '', text)
for line in clean.splitlines():
    if line.startswith('<<TIL-EOT>>'):
        break
    if line.startswith('Loaded ') or line.startswith('..'):
        continue
    lines.append(line[:-1] if line.endswith('.') else line)
out = "\n".join(lines)
if out.endswith("\n"):
    out = out[:-1]   # the newline written before the end marker
sys.stdout.write(out + "\n" if out else "")
