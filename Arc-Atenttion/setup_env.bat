@echo off

echo ========================================
echo Intel Battlemage Development Environment
echo ========================================

:: 1. Activate venv if exists
if exist venv\Scripts\activate.bat (
    echo [1/3] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARN] Virtual environment not found
    echo [INFO] Create with: python -m venv venv
)

:: 1.5 Setup Visual Studio (Required for C++ headers)
echo [1.5/3] Setting up Visual Studio Environment...
set "VS_PATH="

:: Try VS 2022 Community
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    goto :FoundVS
)
:: Try VS 2022 Professional
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    goto :FoundVS
)
:: Try VS 2022 BuildTools
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    goto :FoundVS
)
:: Try VS 2019 Community
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    goto :FoundVS
)

:FoundVS
if defined VS_PATH (
    echo [INFO] Found VS: "%VS_PATH%"
    call "%VS_PATH%" >nul
) else (
    echo [WARN] Visual Studio not found! Build might fail.
    echo [INFO] Ensure C++ Desktop Development workload is installed.
)

:: 2. Setup Intel oneAPI
echo [2/3] Setting up Intel oneAPI...
set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"

if not exist "%ONEAPI_ROOT%" (
    echo [ERROR] Intel oneAPI not found at "%ONEAPI_ROOT%"
    exit /b 1
)

call "%ONEAPI_ROOT%\setvars.bat" intel64 >nul
if errorlevel 1 (
    echo [ERROR] Failed to initialize oneAPI
    exit /b 1
)

:: 3. Set environment variables for Battlemage
echo [3/3] Configuring environment for Battlemage...
set ONEAPI_DEVICE_SELECTOR=level_zero:gpu
set SYCL_DEVICE_FILTER=level_zero:gpu
set SYCL_BE=PI_LEVEL_ZERO

echo.
echo ========================================
echo Environment ready for development!
echo.
echo Commands:
echo   python build.py     - Build the extension
echo   python benchmark.py - Run benchmarks
echo ========================================