@echo off
setlocal

set "INTERP=C:\Users\nykan\Downloads\nekolang-vscode\NekoLang\nekolang_interpreter.py"
set "INITPY=C:\Users\nykan\AppData\Local\Programs\meow\meow_init.py"

if /I "%~1"=="init" (
    python "%INITPY%"
    goto :EOF
)

python "%INTERP%" %*
