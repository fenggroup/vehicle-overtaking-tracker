# Known Issues

## Model Export Issues

### Issue: Automated Scripts Don't Work

**Problem:** The `scripts/download_rtdetr.bat` and `scripts/download_rtdetr.sh` may fail due to:
- Python environment conflicts
- Missing dependencies (imgaug)
- Incorrect file paths in paddle2onnx

**Solution:** Use manual export process documented in [MODEL_SETUP.md](MODEL_SETUP.md)

**Status:** Automated scripts are provided for convenience but manual process is more reliable.

---

### Issue: Python 3.12 Not Supported

**Problem:** 
```
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
```

**Cause:** PaddlePaddle 2.6.0 doesn't support Python 3.12+

**Solution:**
1. Create separate Python 3.11 environment for export only
2. Use Python 3.12 for running the tracker

```bash
# For export only (one time)
conda create -n rtdetr-export python=3.11 -y
conda activate rtdetr-export

# For running tracker
.\venv\Scripts\activate  # Your Python 3.12 venv
```

---

### Issue: Missing imgaug Module

**Problem:**
```
ModuleNotFoundError: No module named 'imgaug'
```

**Solution:**
```bash
pip install imgaug
pip install "numpy<1.24"  # Fix version conflict
```

**Why:** PaddleDetection requires imgaug but doesn't list it in dependencies.

---

### Issue: NumPy Version Conflict

**Problem:**
```
paddledet 2.6.0 requires numpy<1.24, but you have numpy 2.3.3
```

**Solution:**
```bash
pip install "numpy<1.24"
```

---

### Issue: paddle2onnx Command Fails

**Problem:**
```
python.exe: No module named paddle2onnx.__main__
```

**Solution:** Use the executable directly:
```bash
# Windows
C:\Users\USERNAME\miniforge3\envs\rtdetr-export\Scripts\paddle2onnx.exe [args]

# Linux/macOS
paddle2onnx [args]
```

---

## Runtime Issues

### Issue: Model File Not Found

**Problem:**
```
FileNotFoundError: rtdetr.onnx
```

**Solution:** Ensure `rtdetr.onnx` (169MB) exists in project root.

If missing, export it first (see [MODEL_SETUP.md](MODEL_SETUP.md)).

---

### Issue: ONNX Runtime Error

**Problem:**
```
ONNXRuntimeError: Model loading failed
```

**Possible Causes:**
1. Incomplete download (file size should be ~169MB)
2. Corrupted file
3. Wrong ONNX opset version

**Solution:**
```bash
# Check file size
dir rtdetr.onnx  # Windows
ls -lh rtdetr.onnx  # Linux/macOS

# Should show ~169,000,000 bytes
# If much smaller, re-export the model
```

## Quick Reference

| Task | Python Version | Environment |
|------|---------------|-------------|
| Export model | 3.8-3.11 | `rtdetr-export` conda env |
| Run tracker | 3.11+ | Main venv (3.12 works) |
| Install runtime deps | Any 3.11+ | Main venv |
