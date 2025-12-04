# Quick Start Guide

**Note:** Model export requires **Python 3.8-3.11**. Python 3.12+ is not yet supported by PaddlePaddle.

## Get Running

### Option A: Use Pre-exported Model

If you already have `rtdetr.onnx` (169MB file), skip to running the tracker:

```powershell
# Activate your Python 3.12 venv
.\venv\Scripts\activate

# Run tracker
python main.py --input ./data/my_sequence --output ./results
```

### Option B: Export Model Yourself

**Requirements:** Python 3.8-3.11 (create separate environment)

#### Windows Users:

```powershell
# 1. Create Python 3.11 environment
conda create -n rtdetr-export python=3.11 -y
conda activate rtdetr-export

# 2. Install dependencies
pip install -r requirements-export.txt
pip install imgaug  # Additional requirement

# 3. Download weights & export
# Create models directory
mkdir models\rtdetr_l

# Download RT-DETR weights (100MB)
Invoke-WebRequest -Uri "https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_6x_coco.pdparams" -OutFile "models\rtdetr_l\model.pdparams"

# Run manual export (see docs/MODEL_SETUP.md for details)
```

#### Linux/macOS Users:

```bash
# 1. Create Python 3.11 environment
conda create -n rtdetr-export python=3.11 -y
conda activate rtdetr-export

# 2. Install dependencies
pip install -r requirements-export.txt
pip install imgaug

# 3. Download weights & export (see docs/MODEL_SETUP.md)
```

## What Happens

1. Downloads RT-DETR weights (~100MB) from PaddleDetection (Apache 2.0)
2. Converts to ONNX format for inference
3. Creates `rtdetr.onnx` in project root
4. Ready to track vehicles

## Troubleshooting

### Script fails with "curl not found"
**Windows:** Install curl or download manually from: https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rtdetr

**Linux/macOS:** Install curl: `sudo apt install curl` or `brew install curl`

### "No module named paddledet"
```bash
pip install -r requirements-export.txt
```

### Model file is too small or missing
The automated download might have failed. Manual steps:

1. Visit: https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rtdetr
2. Find **RT-DETR Model Zoo** section
3. Download `rtdetr_r50vd_6x_coco.pdparams` 
4. Save to: `models/rtdetr_l/model.pdparams`
5. Run export script:
   ```bash
   python scripts/export_rtdetr_onnx.py \
     --weights models/rtdetr_l/model.pdparams \
     --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
     --image-size 640 640 \
     --device cpu \
     --out rtdetr.onnx
   ```

## After Model Export

Once `rtdetr.onnx` exists:

1. The model file is permanent
2. You can delete the export environment to save space:
   ```bash
   conda remove -n rtdetr-export --all
   ```
3. Use your Python 3.11+ venv for all tracking operations:
   ```bash
   .\venv\Scripts\activate
   python main.py --input ./data/my_sequence --output ./results
   ```

**Note:** The 169MB `rtdetr.onnx` file is all you need to run the tracker.

## Need More Details?

See the comprehensive guide: [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md)

## License Note

**License Compatibility:**
- RT-DETR Model: Apache 2.0 License
- This Project: MIT License  
- Compatible and properly attributed

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for complete attribution.
