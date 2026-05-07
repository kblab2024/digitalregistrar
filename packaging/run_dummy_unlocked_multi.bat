@echo off
setlocal
cd /d "%~dp0"

rem --- Skip Streamlit's one-time "please enter your email" prompt on first launch.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    (
        echo [general]
        echo email = ""
    ) > "%USERPROFILE%\.streamlit\credentials.toml"
)

set "REGISTRAR_ANNOTATE_BASE_DIR=%~dp0dummy"
set "PYTHONPATH=%~dp0app"
set "PYTHONDONTWRITEBYTECODE=1"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

echo ================================================================
echo   Digital Registrar - Annotator (dummy demo, unlocked)
echo ================================================================
echo   Data root : dummy\
echo   Annotators: NHC / KPC (sidebar also allows adding new annotators)
echo   URL       : http://localhost:8501
echo.
echo   Keep this window open while annotating.
echo   Close this window or press Ctrl+C to stop the server.
echo ================================================================
echo.

"%~dp0python\python.exe" -m streamlit run ^
    "%~dp0app\digital_registrar_research\annotation\app_canonical.py" ^
    --server.address=localhost ^
    --server.port=8501

echo.
echo Streamlit has stopped.
pause
endlocal
