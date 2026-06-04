@echo off
REM Download pre-exported RT-DETR ONNX model from GitHub Releases

REM Configuration - Update these when creating GitHub release
set GITHUB_REPO=fenggroup/vehicle-overtaking-tracker
set RELEASE_TAG=v1.0.0
set MODEL_FILE=rtdetr.onnx
set DOWNLOAD_URL=https://github.com/%GITHUB_REPO%/releases/download/%RELEASE_TAG%/%MODEL_FILE%

if exist "%MODEL_FILE%" (
    echo [WARNING] %MODEL_FILE% already exists!
    set /p OVERWRITE="Overwrite? (y/n): "
    if /i not "%OVERWRITE%"=="y" (
        exit /b 0
    )
)

curl -L -o "%MODEL_FILE%" "%DOWNLOAD_URL%"

if errorlevel 1 (
    echo [ERROR] Download failed!
    echo Please download manually from: %DOWNLOAD_URL%
    exit /b 1
)

if not exist "%MODEL_FILE%" (
    echo [ERROR] Model file not found after download!
    exit /b 1
)

for %%A in (%MODEL_FILE%) do set size=%%~zA

if %size% LSS 100000000 (
    echo [ERROR] File size too small ^(%size% bytes, expected ~169MB^)
    exit /b 1
)

echo Download complete: %MODEL_FILE% ^(%size% bytes^)
