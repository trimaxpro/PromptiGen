@echo off
setlocal

:: ═══════════════════════════════════════════════════════════════════
::  PromptiGen Launcher
::  Launches the Gradio web application
:: ═══════════════════════════════════════════════════════════════════

title PromptiGen

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║              PromptiGen                                  ║
echo  ║       Local AI Prompt Generator                          ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: ── Check if virtual environment exists ──────────────────────────
set "VENV_DIR=%~dp0venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo  [ERROR] Virtual environment not found.
    echo  Please run "install.bat" first to set up PromptiGen.
    echo.
    pause
    exit /b 1
)

:: ── Check if models exist ────────────────────────────────────────
set "WD14_MODEL=%~dp0model\wd-v1-4-convnext-tagger-v2\model.onnx"
if not exist "%WD14_MODEL%" (
    echo  [ERROR] WD14 tagger model not found.
    echo  Please run "install.bat" first to download required models.
    echo.
    pause
    exit /b 1
)

:: ── Activate virtual environment ─────────────────────────────────
echo  [*] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
echo  [OK] Environment ready.
echo.

:: ── Launch Gradio App ────────────────────────────────────────────
echo  [*] Starting PromptiGen...
echo  [*] The app will open in your default browser automatically.
echo  [*] Press Ctrl+C to stop the server.
echo.
echo  ────────────────────────────────────────────────────────────
echo.

python "%~dp0gradio_app.py"

:: ── Cleanup ──────────────────────────────────────────────────────
echo.
echo  [*] PromptiGen stopped.
pause
