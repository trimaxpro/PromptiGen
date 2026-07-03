@echo off
setlocal enabledelayedexpansion

:: ═══════════════════════════════════════════════════════════════════
::  PromptiGen Installer
::  Installs Python dependencies & downloads required AI models
:: ═══════════════════════════════════════════════════════════════════

title PromptiGen Installer

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║              PromptiGen - Installer                      ║
echo  ║         Local AI Prompt Generator Setup                  ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: ── Check for Python ──────────────────────────────────────────────
echo [1/6] Checking for Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Please install Python 3.10+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do (
    echo  [OK] Found Python %%v
)
echo.

:: ── Create virtual environment ───────────────────────────────────
echo [2/6] Setting up virtual environment...
set "VENV_DIR=%~dp0venv"
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo  [OK] Virtual environment already exists at: %VENV_DIR%
) else (
    echo  Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)
echo.

:: ── Activate venv ────────────────────────────────────────────────
echo [3/6] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo  [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo  [OK] Virtual environment activated.
echo.

:: ── Upgrade pip ──────────────────────────────────────────────────
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo  [OK] pip upgraded.
echo.

:: ── Install Python dependencies ─────────────────────────────────
echo [5/6] Installing Python dependencies...
echo  This may take several minutes depending on your internet speed.
echo.
echo  Installing core dependencies...
pip install numpy pillow onnxruntime requests gradio
if errorlevel 1 (
    echo  [ERROR] Failed to install core dependencies.
    pause
    exit /b 1
)
echo  [OK] Core dependencies installed.
echo.

echo  Installing CLIP/Vision dependencies (for enhanced tagging)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo  [WARNING] CUDA torch install failed, trying CPU version...
    pip install torch torchvision
    if errorlevel 1 (
        echo  [ERROR] Failed to install PyTorch.
        pause
        exit /b 1
    )
)
echo  [OK] PyTorch installed.
echo.

pip install transformers open_clip_torch safetensors
if errorlevel 1 (
    echo  [ERROR] Failed to install transformers/open_clip.
    pause
    exit /b 1
)
echo  [OK] Transformers and open_clip installed.
echo.

:: ── Download AI Models ──────────────────────────────────────────
echo [6/6] Downloading AI models from HuggingFace...
echo  This will download ~2.9 GB of model files. Please be patient.
echo.

:: Create model directories
set "CLIP_DIR=%~dp0model\clip_vision"
set "WD14_DIR=%~dp0model\wd-v1-4-convnext-tagger-v2"
if not exist "%CLIP_DIR%" mkdir "%CLIP_DIR%"
if not exist "%WD14_DIR%" mkdir "%WD14_DIR%"

:: Download CLIP-ViT-H-14 model (safetensors ~2.5 GB)
set "CLIP_MODEL=%CLIP_DIR%\model.safetensors"
if exist "%CLIP_MODEL%" (
    echo  [SKIP] CLIP model already exists: model.safetensors
) else (
    echo  [DOWNLOADING] CLIP-ViT-H-14 model ^(~2.5 GB^)...
    echo  Source: huggingface.co/Kuvshin/models-moved
    curl -L --progress-bar -o "%CLIP_MODEL%" "https://huggingface.co/Kuvshin/models-moved/resolve/bb47fb73d89d3a37eb907ec1792d8ed4d2f001fa/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    if errorlevel 1 (
        echo  [ERROR] Failed to download CLIP model.
        if exist "%CLIP_MODEL%" del "%CLIP_MODEL%"
        echo  You can manually download from:
        echo  https://huggingface.co/Kuvshin/models-moved/resolve/bb47fb73d89d3a37eb907ec1792d8ed4d2f001fa/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
        echo  Place it in: %CLIP_DIR%\model.safetensors
        pause
        exit /b 1
    )
    echo  [OK] CLIP model downloaded.
)
echo.

:: Download WD v1.4 ConvNeXt Tagger v2 ONNX model (~370 MB)
set "WD14_MODEL=%WD14_DIR%\model.onnx"
if exist "%WD14_MODEL%" (
    echo  [SKIP] WD14 tagger model already exists: model.onnx
) else (
    echo  [DOWNLOADING] WD v1.4 ConvNeXt Tagger v2 ^(~370 MB^)...
    echo  Source: huggingface.co/Bercraft/wd-v1-4-convnext-tagger-v2
    curl -L --progress-bar -o "%WD14_MODEL%" "https://huggingface.co/Bercraft/wd-v1-4-convnext-tagger-v2/resolve/main/model.onnx?download=true"
    if errorlevel 1 (
        echo  [ERROR] Failed to download WD14 tagger model.
        if exist "%WD14_MODEL%" del "%WD14_MODEL%"
        echo  You can manually download from:
        echo  https://huggingface.co/Bercraft/wd-v1-4-convnext-tagger-v2/resolve/main/model.onnx
        echo  Place it in: %WD14_DIR%\model.onnx
        pause
        exit /b 1
    )
    echo  [OK] WD14 tagger model downloaded.
)
echo.

:: Download selected_tags.csv (required for WD14 tagger)
set "WD14_TAGS=%WD14_DIR%\selected_tags.csv"
if exist "%WD14_TAGS%" (
    echo  [SKIP] WD14 tags file already exists: selected_tags.csv
) else (
    echo  [DOWNLOADING] WD14 tags file...
    curl -L --progress-bar -o "%WD14_TAGS%" "https://huggingface.co/Bercraft/wd-v1-4-convnext-tagger-v2/resolve/main/selected_tags.csv?download=true"
    if errorlevel 1 (
        echo  [ERROR] Failed to download tags file.
        if exist "%WD14_TAGS%" del "%WD14_TAGS%"
        pause
        exit /b 1
    )
    echo  [OK] WD14 tags file downloaded.
)
echo.

:: ── Done ─────────────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║           Installation Complete!                         ║
echo  ║                                                          ║
echo  ║   Run "run.bat" to launch PromptiGen                    ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  Installed components:
echo    [+] Python virtual environment (venv)
echo    [+] Core: numpy, pillow, onnxruntime, requests, gradio
echo    [+] Vision: torch, torchvision, transformers, open_clip_torch
echo    [+] Model: CLIP-ViT-H-14-laion2B-s32B-b79K (safetensors)
echo    [+] Model: WD v1.4 ConvNeXt Tagger v2 (ONNX)
echo    [+] Tags: selected_tags.csv
echo.
pause
