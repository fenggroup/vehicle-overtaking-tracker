@echo off
REM Download RT-DETR weights and export to ONNX format

if not defined VIRTUAL_ENV (
    echo [ERROR] Virtual environment not detected!
    echo Please activate your venv first: .\venv\Scripts\activate
    exit /b 1
)

if not exist "models\rtdetr_l" mkdir models\rtdetr_l

python -c "import paddledet" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install paddlepaddle==2.6.0 paddledet==2.6.0 paddle2onnx==1.3.1 onnx==1.15.0
)

echo Downloading RT-DETR weights...
curl -L "https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_6x_coco.pdparams" -o models\rtdetr_l\model.pdparams

if not exist "models\rtdetr_l\model.pdparams" (
    echo [ERROR] Download failed!
    echo Please download manually from:
    echo https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rtdetr
    exit /b 1
)

echo Exporting to ONNX format...
python scripts\export_rtdetr_onnx.py --weights models\rtdetr_l\model.pdparams --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml --image-size 640 640 --device cpu --out rtdetr.onnx

if not exist "rtdetr.onnx" (
    echo [ERROR] ONNX export failed!
    exit /b 1
)

for %%A in (rtdetr.onnx) do set size=%%~zA

if %size% LSS 10000000 (
    echo [ERROR] Model file too small ^(%size% bytes, expected ~169MB^)
    exit /b 1
)

echo Export complete: rtdetr.onnx ^(%size% bytes^)
