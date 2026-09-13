@echo off
cd /d "%~dp0.."
title WAVELY Face Enrolment

if not exist "runtime\WavelyEnrol.exe" (
    echo WAVELY runtime was not found.
    pause
    exit /b 1
)



echo.
echo WAVELY FACE ENROLMENT
echo =====================
echo Stand at the new, farther camera position before continuing.
echo Keep the same lighting you expect to use normally.
echo Follow the prompts in the camera window.
echo Press Q or Escape in that window if you need to stop early.
echo.
pause

"runtime\WavelyEnrol.exe"

echo.
echo Enrolment finished. Next, run launchers\train_face_model.bat.
pause
