@echo off
setlocal
cd /d "%~dp0"
set "DEMO_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "DEMO_PYTHON=%~dp0.venv\Scripts\python.exe"
echo Installing the local project and simulation support...
"%DEMO_PYTHON%" -m pip install -e ".[simulation]"
if errorlevel 1 (
  echo Installation failed. Keep this window open and copy the error.
  if not defined DEMO_NO_PAUSE pause
  exit /b 1
)
echo Generating or resuming all letters and methods. Please wait...
"%DEMO_PYTHON%" -u -m imitation_learning demo --output outputs/demo %*
set "DEMO_EXIT=%ERRORLEVEL%"
if not defined DEMO_NO_OPEN if exist "outputs\demo\robot\dashboard.html" start "" "%CD%\outputs\demo\robot\dashboard.html"
if not "%DEMO_EXIT%"=="0" echo A stage failed or a tracking criterion was not met. Review the output above.
echo Results are in outputs\demo. Run this file again to continue completed stages.
if not defined DEMO_NO_PAUSE pause
exit /b %DEMO_EXIT%
