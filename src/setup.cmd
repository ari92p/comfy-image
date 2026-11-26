@echo off
rem One-time setup for Windows. Needs nothing installed: it fetches uv, uv fetches Python.
rem Everything goes into .venvs\ in this folder. ComfyUI and the models are set up on the first start (it asks first).
cd /d "%~dp0.."
if exist ".venvs\app\.image-ready" (
  echo #easy-image: Already set up. Start it with: easy-image.cmd
  goto :end
)

set "UV="
for /f "delims=" %%i in ('where uv 2^>nul') do if not defined UV set "UV=%%i"
if not defined UV (
  set "UV=.venvs\uv\bin\uv.exe"
  if not exist ".venvs\uv\bin\uv.exe" (
    echo #easy-image: Installing uv ^(the Python setup tool^)...
    set "UV_INSTALL_DIR=%CD%\.venvs\uv\bin"
    set "UV_NO_MODIFY_PATH=1"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >nul
  )
)
if not exist "%UV%" (
  echo #easy-image: Could not install uv.
  goto :end
)

echo #easy-image: Setting up ^(a minute^)...
if exist ".venvs\app" rmdir /s /q ".venvs\app"
"%UV%" venv --quiet --python 3.13 .venvs\app || goto :end
echo ok> .venvs\app\.image-ready
echo #easy-image: Done. Start it with: easy-image.cmd

:end
pause
