#!/bin/bash
# Download RT-DETR weights and export to ONNX format

set -e

if [ -z "$VIRTUAL_ENV" ]; then
    echo "[ERROR] Virtual environment not detected!"
    echo "Please activate your venv first: source venv/bin/activate"
    exit 1
fi

mkdir -p models/rtdetr_l

if ! python -c "import paddledet" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install paddlepaddle==2.6.0 paddledet==2.6.0 paddle2onnx==1.3.1 onnx==1.15.0
fi

echo "Downloading RT-DETR weights..."
curl -L "https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_6x_coco.pdparams" \
     -o models/rtdetr_l/model.pdparams

if [ ! -f "models/rtdetr_l/model.pdparams" ]; then
    echo "[ERROR] Download failed!"
    echo "Please download manually from:"
    echo "https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rtdetr"
    exit 1
fi

echo "Exporting to ONNX format..."
python scripts/export_rtdetr_onnx.py \
    --weights models/rtdetr_l/model.pdparams \
    --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --image-size 640 640 \
    --device cpu \
    --out rtdetr.onnx

if [ ! -f "rtdetr.onnx" ]; then
    echo "[ERROR] ONNX export failed!"
    exit 1
fi

MODEL_SIZE=$(stat -f%z rtdetr.onnx 2>/dev/null || stat -c%s rtdetr.onnx)

if [ "$MODEL_SIZE" -lt 10000000 ]; then
    echo "[ERROR] Model file too small ($MODEL_SIZE bytes, expected ~169MB)"
    exit 1
fi

echo "Export complete: rtdetr.onnx ($MODEL_SIZE bytes)"
