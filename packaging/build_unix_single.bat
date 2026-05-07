@echo off
rem Build the Unix (Linux x86_64) bundle with a single annotator (NHC only).
rem Cross-installs manylinux wheels from this Windows host.
rem Output: packaging\dist\digital-registrar-annotator-unix-single.tar.gz
setlocal
set "PLATFORM=unix"
set "ANNOTATOR_SET=single"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0_build_common.ps1"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo Build failed with exit code %RC%.
    pause
    exit /b %RC%
)
echo.
pause
endlocal
