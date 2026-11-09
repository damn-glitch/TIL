# TIL Language Installer for Windows
# Author: Alisher Beisembekov
# Usage: irm https://til-dev.vercel.app/install.ps1 | iex

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "+--------------------------------------------+" -ForegroundColor Cyan
Write-Host "|              TIL INSTALLER                 |" -ForegroundColor Cyan
Write-Host "|     Author: Alisher Beisembekov            |" -ForegroundColor Cyan
Write-Host "|  `"Проще Python. Быстрее C. Умнее всех.`"    |" -ForegroundColor Cyan
Write-Host "+--------------------------------------------+" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Blue
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion found" -ForegroundColor Green
} catch {
    try {
        $pythonVersion = python3 --version 2>&1
        Write-Host "✓ $pythonVersion found" -ForegroundColor Green
    } catch {
        Write-Host "✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
        exit 1
    }
}

# Check C compiler
Write-Host "Checking C compiler..." -ForegroundColor Blue
$hasCompiler = $false
try {
    gcc --version 2>&1 | Out-Null
    Write-Host "✓ GCC found" -ForegroundColor Green
    $hasCompiler = $true
} catch {
    try {
        cl 2>&1 | Out-Null
        Write-Host "✓ MSVC found" -ForegroundColor Green
        $hasCompiler = $true
    } catch {
        Write-Host "⚠ No C compiler found - TIL will run programs on the TIL VM." -ForegroundColor Yellow
        Write-Host "  (install MinGW-w64 for native executables)" -ForegroundColor Yellow
    }
}

# Create directories
$TIL_HOME = "$env:USERPROFILE\.til"
$TIL_BIN = "$TIL_HOME\bin"
Write-Host "Installing to $TIL_HOME..." -ForegroundColor Blue
New-Item -ItemType Directory -Force -Path $TIL_BIN | Out-Null

# Download compiler
Write-Host "Downloading TIL compiler..." -ForegroundColor Blue
$TIL_RAW = "https://raw.githubusercontent.com/damn-glitch/TIL/main"
Invoke-WebRequest -Uri "$TIL_RAW/src/til.py" -OutFile "$TIL_BIN\til.py"
# TIL 3.1 tools next to the compiler (packages, proofs, device fit, debugger, messages, types)
foreach ($m in @("tilpkg", "tilprove", "tilfit", "tildebug", "tilmsg", "tiltypes")) {
    Invoke-WebRequest -Uri "$TIL_RAW/src/$m.py" -OutFile "$TIL_BIN\$m.py"
}

# Standard library and the portable TIL VM
Write-Host "Downloading standard library and TIL VM..." -ForegroundColor Blue
New-Item -ItemType Directory -Force -Path "$TIL_HOME\stdlib" | Out-Null
New-Item -ItemType Directory -Force -Path "$TIL_HOME\vm" | Out-Null
foreach ($f in @("math.til", "strings.til")) {
    Invoke-WebRequest -Uri "$TIL_RAW/stdlib/$f" -OutFile "$TIL_HOME\stdlib\$f"
}
foreach ($f in @("tilvm.c", "tilvm.h", "main.c", "Makefile", "README.md")) {
    Invoke-WebRequest -Uri "$TIL_RAW/vm/$f" -OutFile "$TIL_HOME\vm\$f"
}
# The self-hosted compiler (TIL written in TIL)
New-Item -ItemType Directory -Force -Path "$TIL_HOME\selfhost" | Out-Null
Invoke-WebRequest -Uri "$TIL_RAW/selfhost/tilc.til" -OutFile "$TIL_HOME\selfhost\tilc.til"

# Create batch wrapper
@"
@echo off
python "%~dp0til.py" %*
"@ | Out-File -FilePath "$TIL_BIN\til.bat" -Encoding ASCII

# Create PowerShell wrapper
@"
python "`$PSScriptRoot\til.py" `$args
"@ | Out-File -FilePath "$TIL_BIN\til.ps1" -Encoding UTF8

# Add to PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$TIL_BIN*") {
    [Environment]::SetEnvironmentVariable("Path", "$TIL_BIN;$currentPath", "User")
    Write-Host "✓ Added to PATH" -ForegroundColor Green
}

# Update current session
$env:Path = "$TIL_BIN;$env:Path"

# Verify
Write-Host ""
Write-Host "+--------------------------------------------+" -ForegroundColor Green
Write-Host "|         TIL INSTALLED SUCCESSFULLY!        |" -ForegroundColor Green
Write-Host "+--------------------------------------------+" -ForegroundColor Green
Write-Host ""

try {
    $version = python "$TIL_BIN\til.py" --version 2>&1
    Write-Host "Version: $version" -ForegroundColor Cyan
} catch {
    Write-Host "Version: 3.1.0" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "To start using TIL:" -ForegroundColor Yellow
Write-Host "  1. Restart PowerShell/Terminal"
Write-Host "  2. Create hello.til:"
Write-Host "     'main()"
Write-Host "         print(`"Hello, TIL!`")' | Out-File hello.til"
Write-Host "  3. Run it: til run hello.til          (or: til run --vm hello.til)"
Write-Host ""
Write-Host "Happy coding with TIL!" -ForegroundColor Cyan
Write-Host "Author: Alisher Beisembekov" -ForegroundColor Cyan
