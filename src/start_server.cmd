@echo off
rem Starts the image service on port 8192 (reachable from other computers). Run setup.cmd once first.
cd /d "%~dp0.."
call easy-image.cmd --serve --host 0.0.0.0 --port 8192 -y
