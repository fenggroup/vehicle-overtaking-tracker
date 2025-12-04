# Model Management

## Current Model

- **Model**: RT-DETR-L (PaddleDetection)
- **Format**: ONNX (rtdetr.onnx)
- **Size**: ~169MB
- **Input**: 640×640 RGB images
- **License**: Apache 2.0
- **Source**: https://github.com/PaddlePaddle/PaddleDetection

## For Users

### Quick Start

**Option 1: Download Pre-exported ONNX**
```bash
# Windows
.\scripts\download_model.bat

# Linux/macOS
bash scripts/download_model.sh
```

**Option 2: Export Yourself**
```bash
# Windows
.\scripts\download_rtdetr.bat

# Linux/macOS
bash scripts/download_rtdetr.sh
```

See [QUICKSTART.md](QUICKSTART.md) for detailed export instructions.

## Sharing the Model

### For Internal Collaboration

**Hosting options**:
1. **Google Drive / Dropbox** - For small teams
2. **GitHub Releases** - For versioned releases
3. **Hugging Face Hub** - For ML models (version control)

**Example: Hugging Face**
```bash
# Upload (one-time)
pip install huggingface_hub
huggingface-cli upload your-org/rtdetr-vot rtdetr.onnx

# Download (collaborators)
huggingface-cli download your-org/rtdetr-vot rtdetr.onnx
```

### For Public Release

**Do NOT commit large model files to Git**. Instead:

1. **Use Git LFS** (if < 2GB and essential):
   ```bash
   git lfs install
   git lfs track "*.onnx"
   git add .gitattributes
   ```

2. **Use GitHub Releases**:
   - Create a release tag (e.g., `v1.0.0`)
   - Attach `rtdetr.onnx` as release asset
   - Update README with download link
   - Update download scripts with repository URL and tag

3. **Provide automated download script**:
   - `scripts/download_model.bat` (Windows)
   - `scripts/download_model.sh` (Linux/macOS)

## Upgrading to Newer RT-DETR Versions

### When to Upgrade

Consider upgrading if:
- New RT-DETR version released with improved accuracy
- Faster inference speed
- Better handling of your specific use case
- Bug fixes in detection

### Upgrade Workflow

**Note:** This is a general guide. Actual upgrade steps may vary depending on the RT-DETR version.

#### Step 1: Update Export Scripts

**Edit `scripts/export_rtdetr_onnx.py`**:
```python
# Change model URL to new version
MODEL_URL = "https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_v2_6x_coco.pdparams"
```

**Edit `scripts/download_rtdetr.bat` and `scripts/download_rtdetr.sh`**:
```bash
# Update download URL to new model version
curl -L "https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_v2_6x_coco.pdparams" -o models/rtdetr_l/model.pdparams
```

#### Step 2: Export New Model

```bash
# Activate environment with PaddlePaddle
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Run export script
bash scripts/download_rtdetr.sh  # or .\scripts\download_rtdetr.bat on Windows

# This creates rtdetr.onnx with new model
```

#### Step 3: Test Thoroughly

```bash
# Test on validation data
python main.py --input ./data/validation_trip --output ./results/test_v2

# Compare with previous version results
# Check:
# - Detection accuracy (false positives/negatives)
# - Processing speed (FPS)
# - Memory usage
# - Edge cases handling
```

**Create comparison report** (example format):
```markdown
## RT-DETR v2.0 vs v1.0 Comparison

### Accuracy
- True Positives: X% (was Y%)
- False Positives: X% (was Y%)
- False Negatives: X% (was Y%)

### Performance
- Speed: X FPS (was Y FPS)
- Memory: X GB (was Y GB)

### Recommendation: [Upgrade / Stay on current version]
```

#### Step 4: Create Release (if using GitHub)

```bash
# 1. Commit updated export scripts
git add scripts/
git commit -m "Update export scripts for RT-DETR v2.0"
git push

# 2. Create new tag
git tag -a v2.0.0 -m "Upgrade to RT-DETR v2.0"
git push origin v2.0.0

# 3. Go to GitHub → Releases → Create Release
#    - Tag: v2.0.0
#    - Title: "v2.0.0 - RT-DETR v2.0 Upgrade"
#    - Upload new rtdetr.onnx
#    - Add release notes (comparison report)
```

#### Step 5: Update Download Scripts

**Edit `scripts/download_model.bat` and `scripts/download_model.sh`**:
```bash
# Change release tag to new version
RELEASE_TAG="v2.0.0"  # was v1.0.0
```

```bash
# Commit changes
git add scripts/
git commit -m "Update download scripts to v2.0.0"
git push
```

#### Step 6: Update Documentation

**Update `docs/MODEL_MANAGEMENT.md`**:
```markdown
## Current Model (v2.0.0)  # Update version

- **Model**: RT-DETR-L v2.0 (PaddleDetection)  # Update
- **Release**: https://github.com/YOUR-ORG/rtdetr-vot/releases/tag/v2.0.0  # Update
```

**Update `README.md`**:
- Update any version references
- Add changelog entry

**Update `CHANGELOG.md`** (create if doesn't exist):
```markdown
## [2.0.0] - YYYY-MM-DD

### Changed
- Upgraded to RT-DETR v2.0
- [Add actual changes observed]

### Migration
- No code changes needed
- Download new model or re-run export scripts
```

### Backward Compatibility

The code is designed to work with RT-DETR ONNX models. Model path is configurable:
```python
tracker = VehiclePassTracker(
    onnx_model_path='rtdetr.onnx'  # Works with different RT-DETR versions
)
```

**Note:** Future RT-DETR versions may have different output formats. Test thoroughly before upgrading.

**Users can switch versions** (if using GitHub Releases):
```bash
# Download specific version
wget https://github.com/YOUR-ORG/rtdetr-vot/releases/download/v1.0.0/rtdetr.onnx -O rtdetr_v1.0.onnx
wget https://github.com/YOUR-ORG/rtdetr-vot/releases/download/v2.0.0/rtdetr.onnx -O rtdetr_v2.0.onnx

# Use specific version
python main.py --onnx-model rtdetr_v1.0.onnx  # Use old version
python main.py --onnx-model rtdetr_v2.0.onnx  # Use new version
```

## Model Verification

### Check Model Integrity

```python
import onnxruntime as ort

# Load model
session = ort.InferenceSession('rtdetr.onnx')

# Check inputs
print("Inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")

# Check outputs
print("Outputs:")
for out in session.get_outputs():
    print(f"  {out.name}: {out.shape} ({out.type})")
```

**Expected output** (for current RT-DETR model):
```
Inputs:
  image: [1, 3, 640, 640] (tensor(float))
  im_shape: [1, 2] (tensor(float))
  scale_factor: [1, 2] (tensor(float))
Outputs:
  [N, 6] (tensor(float))  # [class_id, score, x1, y1, x2, y2]
  [1] (tensor(int64))     # num_detections
```

## Best Practices

1. **Version Control**: Version your models (e.g., `v1.0`, `v2.0`)
2. **Documentation**: Document model version in code/README
3. **Testing**: Test new models thoroughly before deployment
4. **Backup**: Keep old model versions for comparison
5. **Licensing**: Respect model licenses (RT-DETR is Apache 2.0)

## Troubleshooting

### Model Not Found
```
Error: FileNotFoundError: rtdetr.onnx
Solution: Run setup script or download model manually
```

### Wrong Model Format
```
Error: ONNX model expects different input format
Solution: Re-export using scripts/export_rtdetr_onnx.py
```

### Performance Degradation
```
Issue: New model performs worse
Solution: Revert to previous model version, investigate differences
```

## Resources

- **RT-DETR Paper**: https://arxiv.org/abs/2304.08069
- **PaddleDetection**: https://github.com/PaddlePaddle/PaddleDetection
- **ONNX Runtime**: https://onnxruntime.ai/
- **Model Zoo**: https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rtdetr
