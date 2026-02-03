@echo off
REM ========================================
REM Arc-Forge - Intel Arc Optimized
REM ========================================
echo.
echo  ___              ______                    
echo /   ^|  __________/ ____/___  _________ ____ 
echo / /^| ^| / ___/ ___/ /_  / __ \/ ___/ __ `/ _ \
echo / ___ ^|/ /  / /__/ __/ / /_/ / /  / /_/ /  __/
echo /_/  ^|_/_/   \___/_/    \____/_/   \__, /\___/ 
echo                                  /____/       
echo.
echo Intel Arc GPU Optimized Stable Diffusion WebUI
echo.

set PYTHON=
set GIT=
set VENV_DIR=

REM Arc-Forge auto-detects optimal settings for your GPU
REM Add --always-low-vram if you experience OOM errors
set COMMANDLINE_ARGS=--skip-torch-cuda-test

call webui.bat
