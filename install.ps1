# TIL Language Installer for Windows
# Author: Alisher Beisembekov
# Usage: irm https://til-dev.vercel.app/install.ps1 | iex
# Install specific version: $env:TIL_VERSION="1.5"; irm https://til-dev.vercel.app/install.ps1 | iex

$ErrorActionPreference = "Stop"

# Version selection — default to 2.0
$TIL_VERSION = if ($env:TIL_VERSION) { $env:TIL_VERSION } else { "2.0" }

# Map version to git ref
switch -Regex ($TIL_VERSION) {
    "^1\.5" {
        $GitRef = "v1.5.0"
        $VersionDisplay = "1.5.0"
    }
    "^2\.0|^2$|^latest$" {
        $GitRef = "v2.0.0"
        $VersionDisplay = "2.0.0"
    }
    default {
        Write-Host "Unknown version: $TIL_VERSION" -ForegroundColor Red
        Write-Host "Available versions: 1.5, 2.0 (default)"
        exit 1
    }
}

Write-Host ""
Write-Host "+--------------------------------------------+" -ForegroundColor Cyan
Write-Host "|              TIL INSTALLER                 |" -ForegroundColor Cyan
Write-Host "|     Author: Alisher Beisembekov            |" -ForegroundColor Cyan
Write-Host "|  `"Проще Python. Быстрее C. Умнее всех.`"    |" -ForegroundColor Cyan
Write-Host "+--------------------------------------------+" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installing TIL v$VersionDisplay" -ForegroundColor Blue
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Blue
$pythonCmd = $null
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  $pythonVersion found" -ForegroundColor Green
    $pythonCmd = "python"
} catch {
    try {
        $pythonVersion = python3 --version 2>&1
        Write-Host "  $pythonVersion found" -ForegroundColor Green
        $pythonCmd = "python3"
    } catch {
        Write-Host "  Python not found. Please install Python 3.8+" -ForegroundColor Red
        exit 1
    }
}

# Check C compiler
Write-Host "Checking C compiler..." -ForegroundColor Blue
$hasCompiler = $false
try {
    gcc --version 2>&1 | Out-Null
    Write-Host "  GCC found" -ForegroundColor Green
    $hasCompiler = $true
} catch {
    try {
        cl 2>&1 | Out-Null
        Write-Host "  MSVC found" -ForegroundColor Green
        $hasCompiler = $true
    } catch {
        Write-Host "  No C compiler found. Install MinGW-w64 or Visual Studio for compilation." -ForegroundColor Yellow
    }
}

# Create directories
$TIL_HOME = "$env:USERPROFILE\.til"
$TIL_BIN = "$TIL_HOME\bin"
Write-Host "Installing to $TIL_HOME..." -ForegroundColor Blue
New-Item -ItemType Directory -Force -Path $TIL_BIN | Out-Null

# Download compiler — try tag first, fall back to main
Write-Host "Downloading TIL v$VersionDisplay compiler..." -ForegroundColor Blue
$compilerUrl = "https://raw.githubusercontent.com/damn-glitch/TIL/$GitRef/src/til.py"
$fallbackUrl = "https://raw.githubusercontent.com/damn-glitch/TIL/main/src/til.py"

try {
    Invoke-WebRequest -Uri $compilerUrl -OutFile "$TIL_BIN\til.py" -ErrorAction Stop
} catch {
    Write-Host "Tag $GitRef not found, downloading latest from main..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $fallbackUrl -OutFile "$TIL_BIN\til.py"
}

# Write version file
$VersionDisplay | Out-File -FilePath "$TIL_HOME\version" -Encoding UTF8 -NoNewline

# Create batch wrapper
@"
@echo off
$pythonCmd "%~dp0til.py" %*
"@ | Out-File -FilePath "$TIL_BIN\til.bat" -Encoding ASCII

# Create PowerShell wrapper
@"
$pythonCmd "`$PSScriptRoot\til.py" `$args
"@ | Out-File -FilePath "$TIL_BIN\til.ps1" -Encoding UTF8

# Add to PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$TIL_BIN*") {
    [Environment]::SetEnvironmentVariable("Path", "$TIL_BIN;$currentPath", "User")
    Write-Host "  Added to PATH" -ForegroundColor Green
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
    $version = & $pythonCmd "$TIL_BIN\til.py" --version 2>&1
    Write-Host "Version: $($version[0])" -ForegroundColor Cyan
} catch {
    Write-Host "Version: TIL Compiler v$VersionDisplay" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "To start using TIL:" -ForegroundColor Yellow
Write-Host "  1. Restart PowerShell/Terminal"
Write-Host "  2. Create hello.til:"
Write-Host "     'main()"
Write-Host "         print(`"Hello, TIL!`")' | Out-File hello.til"
Write-Host "  3. Run it: til run hello.til"
Write-Host ""
Write-Host "To install a different version:" -ForegroundColor Yellow
Write-Host '  $env:TIL_VERSION="1.5"; irm https://til-dev.vercel.app/install.ps1 | iex'
Write-Host '  $env:TIL_VERSION="2.0"; irm https://til-dev.vercel.app/install.ps1 | iex'
Write-Host ""
Write-Host "Happy coding with TIL!" -ForegroundColor Cyan
Write-Host "Author: Alisher Beisembekov" -ForegroundColor Cyan
