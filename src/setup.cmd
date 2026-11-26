@echo off
rem One-time setup for Windows. Needs nothing installed: it fetches uv, uv fetches Python, then comfy-image sets up ComfyUI
rem and the models (it asks one question first). Everything goes into this folder. Safe to run again: it only adds what is missing.
cd /d "%~dp0.."
set "PY=.venvs\app\Scripts\python.exe"
if exist ".venvs\app\.image-ready" goto :run

set "UV="
for /f "delims=" %%i in ('where uv 2^>nul') do if not defined UV set "UV=%%i"
if not defined UV (
  set "UV=.venvs\uv\bin\uv.exe"
  if not exist ".venvs\uv\bin\uv.exe" (
    echo #comfy-image: Installing uv ^(the Python setup tool^)...
    set "UV_INSTALL_DIR=%CD%\.venvs\uv\bin"
    set "UV_NO_MODIFY_PATH=1"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >nul
  )
)
if not exist "%UV%" (
  echo #comfy-image: Could not install uv.
  goto :end
)

echo #comfy-image: Setting up Python ^(a minute^)...
if exist ".venvs\app" rmdir /s /q ".venvs\app"
"%UV%" venv --quiet --python 3.13 .venvs\app || goto :end
echo ok> .venvs\app\.image-ready

:run
"%PY%" src\comfy-image.py --setup

:end
pause
