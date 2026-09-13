@echo off
setlocal
cd /d "%~dp0"
title WAVELY Source Dashboard Test

set "PYTHON=%~dp0build\venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo WAVELY development Python was not found:
    echo %PYTHON%
    echo.
    echo Restore Python 3.11 and recreate the build virtual environment if needed.
    pause
    exit /b 1
)

set "SOURCE=%~dp0src\wavely_dashboard.py"
if not exist "%SOURCE%" (
    echo The current WAVELY source dashboard was not found:
    echo %SOURCE%
    echo.
    pause
    exit /b 1
)

set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
echo Starting the current source dashboard for testing...
echo Close the WAVELY window to return here.
echo.
"%PYTHON%" "%SOURCE%"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo WAVELY source dashboard exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%