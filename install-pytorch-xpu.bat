@echo off
REM ========================================
REM Arc-Forge - PyTorch XPU Installer
REM ========================================
echo.
echo  ___              ______                    
echo /   ^|  __________/ ____/___  _________ ____ 
echo / /^| ^| / ___/ ___/ /_  / __ \/ ___/ __ `/ _ \
echo / ___ ^|/ /  / /__/ __/ / /_/ / /  / /_/ /  __/
echo /_/  ^|_/_/   \___/_/    \____/_/   \__, /\___/ 
echo                                  /____/       
echo.
echo PyTorch XPU Installer for Intel Arc GPUs
echo.
echo ========================================
echo Choose PyTorch version:
echo ========================================
echo.
echo [1] STABLE (Recommended)
echo     PyTorch 2.7.x - Tested and reliable
echo.
echo [2] NIGHTLY (Bleeding Edge)
echo     Latest features, may have bugs
echo.
echo [3] Cancel
echo.

set /p choice="Enter your choice (1/2/3): "

if "%choice%"=="1" goto stable
if "%choice%"=="2" goto nightly
if "%choice%"=="3" goto cancel
goto invalid

:stable
echo.
echo Installing PyTorch Stable (XPU)...
echo.
.\venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --upgrade
goto verify

:nightly
echo.
echo Installing PyTorch Nightly (XPU)...
echo.
.\venv\Scripts\pip.exe install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu --upgrade
goto verify

:verify
echo.
echo ========================================
echo Verifying installation...
echo ========================================
.\venv\Scripts\python.exe -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'XPU available: {torch.xpu.is_available()}')"
echo.
echo Installation complete!
echo.
pause
goto end

:invalid
echo Invalid choice. Please run again and select 1, 2, or 3.
pause
goto end

:cancel
echo Cancelled.
goto end

:end
