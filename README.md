# Vehicle Overtaking Tracker

A computer vision system for detecting and analyzing vehicle overtaking events from image sequences. The system combines deep learning-based object detection (RT-DETR) with computer vision tracking (ByteTrack) and geometric analysis for robust passing event validation.

> **New User?** Jump straight to **[docs/QUICKSTART.md](docs/QUICKSTART.md)** for automated 5-minute setup!

## License & Attribution

- **This Project:** MIT License (Copyright 2025 Feng Group)
- **RT-DETR Model:** Apache 2.0 License (PaddleDetection)
- **ByteTrack:** MIT License
- **All dependencies:** Permissive licenses (see [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md))

## System Requirements
- Python 3.8-3.11 (Python 3.12+ not supported by PaddlePaddle)
- Windows, Linux, or macOS
- 16GB RAM, 64-bit OS recommended

## Features
- Vehicle detection using RT-DETR (Real-Time DEtection TRansformer) via ONNX
- Multi-object tracking with ByteTrack
- Real-time visualization and event logging
- Support for multiple camera positions
- CSV export of detected events
- Permissive Apache/MIT license stack
- Jupyter notebook for interactive analysis (`vehicle_pass_tracker_analysis.ipynb`)

## Quick Start

### 1. Install Dependencies
```bash
# Create and activate virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Setup RT-DETR Model (First Time Only)

**You must obtain the RT-DETR ONNX model before running the tracker.**

#### Option A: Download Pre-exported Model (Recommended)

```bash
# Windows
.\scripts\download_model.bat

# Linux/macOS
bash scripts/download_model.sh
```

Downloads the pre-exported ONNX model (~169MB) from GitHub Releases. No PaddlePaddle installation needed.

#### Option B: Export Model Yourself

For manual export or if download fails:
1. See [docs/QUICKSTART.md](docs/QUICKSTART.md) for automated export
2. See [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md) for manual steps
3. See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) for troubleshooting

Note: Export requires Python 3.8-3.11 with PaddlePaddle

#### For Collaborators

See [docs/MODEL_MANAGEMENT.md](docs/MODEL_MANAGEMENT.md) for model sharing and versioning

### 3. Run the Tracker
```bash
# Basic usage (all platforms)
python main.py

# Advanced usage with options (use forward slashes for Windows/macOS/Linux)
python main.py \
    --input ./data/my_sequence \
    --output ./results \
    --mode debug \
    --camera-position bottom_center \
    --excluded-frames ./data/excluded_frames.csv
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| --input | Required | Path to input image sequence folder |
| --output | ./results | Path to output folder |
| --mode | demo | Operation mode (demo/debug) |
| --camera-position | bottom_center | Camera position (bottom_center/bottom_left) |
| --excluded-frames | None | Optional CSV file with frames to exclude |
| --onnx-model | rtdetr.onnx | Path to RT-DETR ONNX model file |
| --no-display | False | Run in headless mode (no visualization window) |
| --output-format | png | Output image format: png (lossless) or jpg (compressed) |

## Performance Optimization

### Headless Mode
For faster processing without GUI overhead:
```bash
python main.py --no-display
```
Useful for server deployments, batch processing, and remote execution.

### Output Format Selection
- **PNG**: Lossless, larger files (~2-5MB/frame), best quality
- **JPG**: Compressed, smaller files (~200-500KB/frame), good quality

### Background Frame Saving
Frames are saved asynchronously in a background thread to prevent I/O blocking.

### Input Data Requirements

#### Image Sequence
- Format: JPG or PNG
- Naming: frame000000.jpg, frame000001.jpg, etc.
- Resolution: Flexible (tested with IR and GoPro footage)
- Location: `data/input_trip/`

#### Ground Truth Data (Optional)
For validation:
```csv
# ground_truth.csv
frame_number,notes
1500,car_passing
1650,truck_passing
```

## Configuration Options

### Camera Position
```python
tracker = VehiclePassTracker(
    image_sequence_path='./data/input_trip/',
    output_folder='./results/',
    image_source_position='bottom_center',  # or 'bottom-left'
)
```

### Customizable Parameters (in tracker_types.py)

All algorithm parameters are configurable through the `TrackerConfig` dataclass:

```python
from src.tracker_types import TrackerConfig

config = TrackerConfig(
    # Detection parameters
    confidence_threshold=0.4,          # Minimum confidence for RT-DETR detections (0.0-1.0)
    valid_vehicle_classes=[2, 3, 5, 7], # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    
    # Angle-based filtering (overtaking direction)
    min_angle=0.0,                     # Minimum angle in degrees (default: right half of frame)
    max_angle=90.0,                    # Maximum angle in degrees
    min_angle_slope=0.7,               # Minimum angle increase rate (deg/frame) for overtaking
    min_angle_increase_count=5,        # Minimum frames with angle increase (out of history)
    angle_trend_threshold=0.5,         # Threshold for detecting angle trends (deg/frame)
    
    # Bounding box growth filtering (approaching vehicles)
    min_bbox_growth_rate=100.0,        # Minimum bbox area growth (px²/frame) - PRIMARY filter
    
    # History and evaluation windows
    history_window_size=10,            # Frames retained in angle/position history
    min_frames_for_evaluation=7,       # Minimum frames before evaluating overtaking
    probation_frames=2,                # Consecutive frames passing checks before yellow box
    
    # Confirmation line positions (fraction of frame width)
    confirmation_line_bottom_center=0.85,  # For rear-center camera
    confirmation_line_bottom_left=0.65,    # For rear-left camera
    
    # Event management
    cleanup_frames=30,                 # Frames of inactivity before removing track
    buffer_frames=5,                   # Buffer added to event boundaries
    tolerance_threshold=10,            # Tolerance for merging overlapping events (frames)
)
```


## Output Format

### CSV Output (vehicle_passing.csv)
```csv
pass_id,track_id,first_frame,last_frame,vehicle_class,passing_frame,ref_point_distance,passing_angle,bbox_x1,bbox_y1,bbox_x2,bbox_y2
1,145,1500,1530,2,1520,150.5,45.2,100,200,300,400
```

- **pass_id**: Unique event identifier
- **track_id**: Internal tracking ID
- **first_frame**: Event start frame
- **last_frame**: Event end frame
- **vehicle_class**: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
- **passing_frame**: Frame when vehicle crossed confirmation line
- **ref_point_distance**: Distance from reference point (pixels)
- **passing_angle**: Vehicle bearing angle at passing confirmation (degrees)
- **bbox_x1, bbox_y1, bbox_x2, bbox_y2**: Bounding box coordinates at passing

Auto-saved every 1000 frames with full post-processing applied.

### Visualizations
- Red bounding boxes: confirmed passing
- Vehicle class and ID labels
- All frames saved to `output_folder/inference_images/`


## Quick Start Examples

### 1. Basic Usage with Default Settings
```python
from src.tracker import VehiclePassTracker

tracker = VehiclePassTracker(
    image_sequence_path='./data/my_sequence/',
    output_folder='./my_results/'
)
tracker.process_sequence()
```

### 2. Advanced Configuration
```python
from src.tracker import VehiclePassTracker
from src.tracker_types import TrackerConfig

config = TrackerConfig(
    confidence_threshold=0.5,
    min_angle_slope=1.0,
    min_bbox_growth_rate=150.0,
    probation_frames=3,
    valid_vehicle_classes=[2, 3]  # Cars and motorcycles only
)

tracker = VehiclePassTracker(
    image_sequence_path='./data/my_sequence/',
    output_folder='./my_results/',
    mode='debug',
    image_source_position='bottom_left',
    config=config
)
tracker.process_sequence()
```

## Result Validation
```python
tracker = VehiclePassTracker(
    image_sequence_path='./data/sequence/',
    output_folder='./results/',
    excluded_frames_path='./data/excluded_frames.csv'
)
```

## Parameter Tuning Guide

### Core Algorithm Parameters

The algorithm uses **3 core checks** for overtaking detection:
1. **Angle slope** - Vehicle moving left->right (direction)
2. **Angle increases** - Consistent trend (jitter-robust)
3. **Bbox growth** - Vehicle approaching (distance)

### Detection Quality
- Increase `confidence_threshold` (0.4 → 0.5-0.6) for fewer false detections
- Decrease `confidence_threshold` (0.4 → 0.3) for distant vehicles
- Default 0.4 is optimized for validation datasets

### Overtaking Sensitivity

**Stricter detection (fewer false positives):**
```python
config = TrackerConfig(
    min_angle_slope=1.0,
    min_bbox_growth_rate=200.0,
    probation_frames=3,
    min_frames_for_evaluation=10
)
```

**More sensitive detection (catch all overtakes):**
```python
config = TrackerConfig(
    min_angle_slope=0.5,
    min_bbox_growth_rate=50.0,
    probation_frames=1,
    min_frames_for_evaluation=5
)
```

### Camera Setup Adjustments

```python
config = TrackerConfig(
    min_angle=0.0,
    max_angle=90.0,
    confirmation_line_bottom_center=0.85,
    confirmation_line_bottom_left=0.65,
)
```

### Common Scenarios

**Urban/low-speed environments:**
- `min_bbox_growth_rate`: 50-75
- `probation_frames`: 3-4
- `history_window_size`: 15

**Highway/high-speed environments:**
- `min_bbox_growth_rate`: 150-200
- `probation_frames`: 2
- `history_window_size`: 10

**Large vehicles only (trucks/buses):**
- `valid_vehicle_classes=[5, 7]`

## Common Issues and Solutions

### Poor Detection
- Check image quality and lighting
- Adjust confidence threshold
- Verify camera position matches configuration

### False Positives
- Increase `min_angle_slope` (0.7 -> 1.0)
- Increase `min_bbox_growth_rate` (100 -> 150-200)
- Increase `probation_frames` (2 -> 3-4)
- Increase `confidence_threshold` (0.4 -> 0.5)

### Missing Events
- Lower `min_angle_slope` (0.7 -> 0.5)
- Lower `min_bbox_growth_rate` (100 -> 50-75)
- Reduce `probation_frames` (2 -> 1)
- Reduce `confidence_threshold` (0.4 -> 0.3)
- Check excluded frames

### Processing Speed
- Enable GPU acceleration when available
- Adjust frame processing rate if needed
- Optimize image resolution for your use case

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## Citing This Work

If you use this software in your research, please cite:

**This Tracker:**
```bibtex
@software{vehicle_overtaking_tracker,
  author = {Padmanaban, Gandhimathi and Moustafa, Rayane and Feng, Fred},
  title = {Vehicle Overtaking Tracker: An Automated Overtaking Event Detection System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/fenggroup/vehicle-overtaking-tracker.git}
}
```

## Directory Structure
```
rtdetr-vot/
├── data/
│   ├── ground_truth_annotations/
│   └── input_trip/
├── docs/
│   ├── ALGORITHM_REFERENCE.md
│   ├── CONTRIBUTING.md
│   ├── KNOWN_ISSUES.md
│   ├── MODEL_MANAGEMENT.md
│   ├── MODEL_SETUP.md
│   ├── QUICKSTART.md
│   └── THIRD_PARTY_LICENSES.md
├── models/
│   └── rtdetr_l/
│       └── model.pdparams
├── results/
│   ├── inference_images/
│   ├── vehicle_passing.csv
│   └── tracking_debug.log
├── scripts/
│   ├── download_rtdetr.bat
│   ├── download_rtdetr.sh
│   └── export_rtdetr_onnx.py
├── src/
│   ├── detectors.py
│   ├── tracker.py
│   ├── tracker_types.py
│   ├── utils.py
│   └── visualization.py
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
└── vehicle_pass_tracker_analysis.ipynb
```

## Troubleshooting

See [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md) for model setup issues.

For runtime issues:
- Verify image sequence naming convention
- Confirm camera position matches configuration
- Check input paths are accessible
- Monitor RAM usage for large sequences
- On macOS, grant permissions for protected directories
- Use forward slashes (/) in paths


## License
MIT License - See LICENSE file for details

## Support

For issues:
1. Check [Issues](https://github.com/fenggroup/vehicle-overtaking-tracker/issues)
2. Review Troubleshooting section
3. Open a new issue with system details and error messages
