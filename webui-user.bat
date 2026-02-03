@echo off
REM ========================================
REM Arc-Forge - Intel Arc Optimized
REM https://github.com/FellypeMelo/Arc-Forge
REM ========================================

set PYTHON=
set GIT=
set VENV_DIR=

REM Arc-Forge auto-detects optimal settings for your GPU
REM Troubleshooting:
REM   - OOM errors? Add: --always-low-vram
REM   - Still crashing? Add: --always-no-vram --vae-in-cpu
set COMMANDLINE_ARGS=--skip-torch-cuda-test

call webui.bat

