@echo off
rem Short start for Windows: easy-image  (runs easy-image.py with the Python that setup.cmd made)
%~dp0.venvs\app\Scripts\python.exe "%~dp0src\easy-image.py" %*
