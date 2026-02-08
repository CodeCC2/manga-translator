@echo off
echo =========================================
echo   Manga Translator - Installation
echo =========================================
echo.

echo Installing base dependencies...
python -m pip install -r requirements.txt

echo.
echo =========================================
echo   Select PyTorch Version:
echo   1. CPU (Default - works on all computers)
echo   2. GPU (NVIDIA RTX/GTX with CUDA 12.8)
echo =========================================
echo.

set /p choice="Enter 1 or 2: "

if "%choice%"=="2" (
    echo.
    echo Installing PyTorch GPU version...
    python -m pip uninstall torch torchvision -y
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    echo GPU version installed!
) else (
    echo.
    echo CPU version already installed!
)

echo.
echo =========================================
echo   Installation Complete!
echo   Starting server...
echo =========================================
echo.

python -m uvicorn app.main:app --reload --port 8000
