@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo Intel Battlemage Benchmark Runner
echo ========================================

:: 1. Setup Environment relative to this script
call setup_env.bat
if errorlevel 1 (
    echo [ERROR] Environment setup failed
    exit /b 1
)

:: 2. Run Benchmark Suite
echo.
echo [INFO] Running Comprehensive Benchmark Suite...
python benchmark.py
if errorlevel 1 (
    echo [ERROR] Benchmark failed
    goto :Error
)

:Success
:: 3. Success
echo.
echo ========================================
echo [SUCCESS] Complete!
echo ========================================
goto :End

:Error
echo.
echo [ERROR] Process failed.
exit /b 1

:End
endlocal
