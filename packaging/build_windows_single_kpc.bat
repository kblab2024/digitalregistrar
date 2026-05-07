@echo off
rem Build the Windows bundle with a single annotator (KPC only).
rem Output: packaging\dist\digital-registrar-annotator-windows-single_kpc.zip
setlocal
set "PLATFORM=windows"
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
