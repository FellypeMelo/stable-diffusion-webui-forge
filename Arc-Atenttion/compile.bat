@echo off
setlocal enabledelayedexpansion

echo [Arc-Attention] Checking Build Environment...

:: 1. Search for Visual Studio 2022/2019 Build Tools (vcvars64.bat)
set "VS_FOUND=0"
set "VCVARS_PATH="

:: List of common paths for vcvars64.bat
set "PATHS[0]=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PATHS[1]=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PATHS[2]=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
set "PATHS[3]=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set "PATHS[4]=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PATHS[5]=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

for /L %%i in (0,1,5) do (
    if exist "!PATHS[%%i]!" (
        set "VCVARS_PATH=!PATHS[%%i]!"
        set "VS_FOUND=1"
        goto :FoundVS
    )
)

:FoundVS
if "!VS_FOUND!"=="1" (
    echo [INFO] Found Visual Studio: "!VCVARS_PATH!"
    echo [INFO] Initializing MSVC Environment...
    call "!VCVARS_PATH!" >nul 2>&1
) else (
    echo [WARN] Visual Studio vcvars64.bat NOT found in standard locations.
    echo [WARN] Attempting to proceed, but build may fail if not run from 'x64 Native Tools Command Prompt'.
)

:: 2. Initialize Intel oneAPI
if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
    echo [INFO] Initializing oneAPI Environment...
    call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
) else (
    echo [ERROR] Intel oneAPI 'setvars.bat' not found!
    echo Please fix your oneAPI installation.
    exit /b 1
)

:: 3. Run Build
echo.
echo [Arc-Attention] Compiling Optimization Kernel (Principal Engineer Edition)...
cd /d "%~dp0"
..\venv\Scripts\python.exe setup.py build_ext --inplace

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Compilation Failed!
    echo See messages above.
    exit /b 1
)

echo.
echo [SUCCESS] Kernel Updated Successfully!
echo You can now restart Forge.
exit /b 0
