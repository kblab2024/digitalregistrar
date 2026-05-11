@echo off
rem Thin wrapper — see packaging/build.py for the canonical entry point.
rem Cross-builds a Unix bundle from a Windows host.
setlocal
python "%~dp0build.py" --platform unix --annotators multi %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" ( echo. & echo Build failed with exit code %RC%. & pause & exit /b %RC% )
echo.
pause
endlocal
