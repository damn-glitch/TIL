"""
TIL Compiler CLI entry point for pip installation.
Usage: til run program.til
"""

import sys
import os


def main():
    # The actual compiler is in src/til.py — locate it relative to this package
    # When installed via pip, til.py is bundled alongside this module
    compiler_path = os.path.join(os.path.dirname(__file__), 'til.py')

    if not os.path.exists(compiler_path):
        # Fallback: try src/til.py relative to project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        compiler_path = os.path.join(project_root, 'src', 'til.py')

    if not os.path.exists(compiler_path):
        print("Error: TIL compiler (til.py) not found.", file=sys.stderr)
        print("Try reinstalling: pip install --force-reinstall til-compiler", file=sys.stderr)
        sys.exit(1)

    # Import and run the compiler's main()
    import importlib.util
    spec = importlib.util.spec_from_file_location("til", compiler_path)
    til_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(til_module)
    sys.exit(til_module.main())


if __name__ == '__main__':
    main()
