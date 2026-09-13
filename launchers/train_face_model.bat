@echo off
cd /d "%~dp0.."
title WAVELY Face Training

if not exist "runtime\WavelyTrain.exe" (
    echo WAVELY runtime was not found.
    pause
    exit /b 1
)



echo.
echo WAVELY FACE TRAINING
echo ====================
echo This combines your existing close-range samples with the new
echo farther-distance samples, then rebuilds the face model.
echo Keep this window open until it says Training complete.
echo.
pause

"runtime\WavelyTrain.exe"

echo.
echo Training finished. You can now start WAVELY normally.
pause
