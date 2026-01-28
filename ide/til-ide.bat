@echo off
REM TIL IDE Launcher for Windows
REM Author: Alisher Beisembekov

echo Starting TIL IDE...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if PyQt6 is installed
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo Installing PyQt6...
    pip install PyQt6
)

REM Run the IDE
python "%~dp0til_ide.py"
