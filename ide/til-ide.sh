#!/bin/bash
# TIL IDE Launcher for Linux/macOS
# Author: Alisher Beisembekov

echo "Starting TIL IDE..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8+"
    exit 1
fi

# Check if PyQt6 is installed
if ! python3 -c "import PyQt6" &> /dev/null; then
    echo "Installing PyQt6..."
    pip3 install PyQt6
fi

# Run the IDE
python3 "$SCRIPT_DIR/til_ide.py"
