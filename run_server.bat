@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto failed
)
set "UKMP_SERVER_PYTHON=%~dp0.venv\Scripts\python.exe"
"%UKMP_SERVER_PYTHON%" -m pip install -e ".[server,simulation]"
if errorlevel 1 goto failed
echo Starting the experiment manager at http://127.0.0.1:8000
"%UKMP_SERVER_PYTHON%" -u -m imitation_learning server --open-browser %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo Installation or server startup failed. Copy the error above if you need help.
if not defined SERVER_NO_PAUSE pause
exit /b 1
