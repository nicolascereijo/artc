@echo off
setlocal

REM Launch the internal PowerShell installer script with temporary bypass
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0artc_install_windows_core.ps1"

endlocal
