@echo off
rem Short start for Windows: comfy-image  (runs comfy-image.py with the Python that setup.cmd made)
%~dp0.venvs\app\Scripts\python.exe "%~dp0src\comfy-image.py" %*
