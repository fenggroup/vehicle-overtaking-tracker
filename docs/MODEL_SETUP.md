# RT-DETR Model Setup Guide

This guide covers obtaining and setting up the RT-DETR ONNX model using the official PaddleDetection repository (Apache 2.0 licensed).

## Prerequisites

- **Python 3.8-3.11** (Python 3.12+ not yet supported by PaddlePaddle for export)
- Git
- 10GB+ free disk space (for PaddleDetection repo and model weights)

**Note:** If you only want to run the tracker (not export models), Python 3.11+ works with `requirements.txt`. Only model export requires Python 3.8-3.11.

## Quick Setup

### Step 1: Create Python 3.11 Environment

**Important:** Don't use your main venv if it's Python 3.12+. Create a separate environment:

```bash
# Using conda/mamba
conda create -n rtdetr-export python=3.11 -y
conda activate rtdetr-export

# Install required dependencies
pip install paddlepaddle==2.6.0  # or paddlepaddle-gpu for CUDA
pip install paddledet==2.6.0
pip install paddle2onnx==1.3.1
pip install onnx==1.15.0
pip install imgaug  # Required by PaddleDetection

# Fix numpy version conflict
pip install "numpy<1.24"
```

**Alternative: Using pyenv**
```bash
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv-export
# Windows:
.\venv-export\Scripts\activate
# Linux/macOS:
source venv-export/bin/activate

# Then install dependencies as above
pip install -r requirements-export.txt
pip install imgaug
pip install "numpy<1.24"
```

### Step 2: Download Pre-trained RT-DETR Weights

**Option A: RT-DETR-L**
```bash
# Create models directory
mkdir -p models/rtdetr_l

# Download RT-DETR-L weights from PaddleDetection Model Zoo
# Visit: https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/configs/rtdetr

# Download link:
curl -L https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_6x_coco.pdparams -o models/rtdetr_l/model.pdparams
```

**Option B: RT-DETR-R50**
```bash
mkdir -p models/rtdetr_r50

# Download RT-DETR-R50 weights
curl -L https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_6x_coco.pdparams -o models/rtdetr_r50/model.pdparams
```

**Manual Download Alternative:**
1. Visit: https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rtdetr
2. Find the "Model Zoo" section
3. Download the `.pdparams` file for `rtdetr_r50vd_6x_coco` or `rtdetr_r101vd_6x_coco`
4. Save to `models/rtdetr_l/model.pdparams`

### Step 3: Export to ONNX

```bash
# Export RT-DETR-L to ONNX
python scripts/export_rtdetr_onnx.py \
  --weights models/rtdetr_l/model.pdparams \
  --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  --image-size 640 640 \
  --device cpu \
  --out rtdetr.onnx \
  --cache-dir .cache_pd
```

**Expected output:**
```
Cloning PaddleDetection...
Exporting Paddle inference model...
Converting to ONNX...
Export completed: D:\Research\Dissertation\code\rtdetr-vot\rtdetr.onnx
```

### Step 4: Verify the Model

```bash
# Check if model exists and has reasonable size (should be 80-300MB)
# Windows PowerShell:
Get-Item rtdetr.onnx | Select-Object Name, Length

# Linux/macOS:
ls -lh rtdetr.onnx
```

## Using the Model

Once you have `rtdetr.onnx` in your project root, run the tracker:

```bash
# Basic usage (model auto-detected in project root)
python main.py --input ./data/my_sequence --output ./results

# Or specify model path explicitly
python main.py \
  --input ./data/my_sequence \
  --output ./results \
  --onnx-model rtdetr.onnx
```

## Troubleshooting

### Issue: "AttributeError: module 'pkgutil' has no attribute 'ImpImporter'"
**Cause:** Using Python 3.12+, but PaddlePaddle requires Python 3.8-3.11

**Solution:** Create a Python 3.11 environment:
```bash
# Using conda/mamba
conda create -n rtdetr-export python=3.11 -y
conda activate rtdetr-export
pip install -r requirements-export.txt

# Or using pyenv
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv-export
source venv-export/bin/activate  # or .\venv-export\Scripts\activate on Windows
pip install -r requirements-export.txt
```

### Issue: "PaddleDetection not found"
**Solution:** Ensure paddledet is installed: `pip install paddledet==2.6.0`

### Issue: "Config file not found"
**Solution:** The export script automatically clones PaddleDetection. Ensure you have internet access and ~5GB disk space.

### Issue: Export fails with CUDA errors
**Solution:** Use `--device cpu` in the export command (already default)

### Issue: Model file too small (<10MB)
**Solution:** The export may have failed. Delete the file and re-run the export script with verbose output.

### Issue: "No module named 'ppdet'"
**Solution:** Install PaddleDetection: `pip install paddledet`

## Model Specifications

### RT-DETR-L (R50)
- **Backbone:** ResNet-50-vd
- **Input size:** 640×640
- **COCO mAP:** ~53.1
- **Inference speed:** ~30ms on CPU
- **Model size:** ~100MB ONNX

### RT-DETR-R101
- **Backbone:** ResNet-101-vd
- **Input size:** 640×640
- **COCO mAP:** ~54.3
- **Inference speed:** ~40ms on CPU
- **Model size:** ~180MB ONNX

## License Attribution

The RT-DETR model and weights are from:

**PaddleDetection**
- Repository: https://github.com/PaddlePaddle/PaddleDetection
- License: Apache License 2.0
- Copyright: (c) 2019 PaddlePaddle Authors

**RT-DETR Paper:**
- Title: "DETRs Beat YOLOs on Real-time Object Detection"
- Authors: Wenyu Lv, Shangliang Xu, Yian Zhao, et al.
- arXiv: https://arxiv.org/abs/2304.08069

This is compatible with the MIT License of this project.

## Advanced: Custom Model Training

To train RT-DETR on custom data:

1. Follow PaddleDetection's training guide: https://github.com/PaddlePaddle/PaddleDetection
2. Export your trained model using the same export script
3. Use with this tracker

## Alternative Models

While this project is designed for RT-DETR, you can adapt it for other ONNX models:

1. Ensure the model outputs COCO classes (or modify `valid_vehicle_classes`)
2. Adapt the `OnnxRTDetrDetector` class in `src/detectors.py`
3. Verify the output format matches supervision's `Detections` format

## Support

For model-specific issues:
- PaddleDetection Issues: https://github.com/PaddlePaddle/PaddleDetection/issues
- RT-DETR Paper: https://arxiv.org/abs/2304.08069

For tracker-specific issues:
- Open an issue in this repository
