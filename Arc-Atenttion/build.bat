@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo Intel Battlemage FlashAttention Build
echo Windows Production Build Script
echo ========================================

:: 1. Setup Environment
call setup_env.bat
if errorlevel 1 (
    echo [ERROR] Environment setup failed
    exit /b 1
)

:: 2. Run Python Build Script
echo.
echo [4/5] Running Python Build System...
python build.py
if errorlevel 1 (
    echo [ERROR] Build failed
    goto :Error
)

:: 3. Success
echo.
echo ========================================
echo [SUCCESS] Build Complete!
echo ========================================
echo.
echo To run benchmarks:
echo   python benchmark.py
echo.
goto :End

:Error
echo.
echo [ERROR] Build process failed. Check logs above.
echo.
exit /b 1

:End
endlocal
