@echo off
REM TIL Studio launcher for Windows: opens the IDE in your browser.
REM   til-ide.bat [FOLDER]        (default: the current folder)
REM Same as `til ide [FOLDER]`. Needs only Python 3.9+.
where til >nul 2>&1
if not errorlevel 1 (
    til ide %*
    exit /b %errorlevel%
)
where python >nul 2>&1
if errorlevel 1 (
    echo TIL Studio needs Python 3.9+: https://www.python.org/downloads/
    pause
    exit /b 1
)
python "%~dp0..\src\til.py" ide %*
