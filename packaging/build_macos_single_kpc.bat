@echo off
rem Build the macOS (Apple Silicon, Mac mini M1/M2/M3+) bundle with a single
rem annotator (KPC only).
rem Output: packaging\dist\digital-registrar-annotator-macos-single_kpc.tar.gz
setlocal
set "PLATFORM=macos"
set "ANNOTATOR_SET=single_kpc"
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
