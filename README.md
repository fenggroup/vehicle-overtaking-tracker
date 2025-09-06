# Vehicle Pass Tracker

A computer vision system for detecting and analyzing vehicle passing events from image sequences. The system combines deep learning-based object detection (YOLOv5) with computer vision tracking (ByteTrack) and geometric analysis for robust passing event validation.

## System Requirements
- Python 3.8+
- Windows/Linux/macOS operating system
- Minimum recommended specs:
  - 16GB RAM
  - 64-bit OS
  - Intel Core i7/Apple M1 or equivalent
  - Storage space for output images

## Features
- Vehicle detection using YOLOv5
- Multi-object tracking with ByteTrack
- Real-time visualization and event logging
- Support for multiple camera positions
- CSV export of detected events
- Jupyter notebook for interactive analysis (`vehicle_pass_tracker_analysis.ipynb`)

## Quick Start

1. **Install Dependencies**
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

2. **Run the Tracker in CLI**
```bash
# Basic usage (all platforms)
python main.py --input ./data/my_sequence --output ./results

# Advanced usage with options (use forward slashes for Windows/macOS/Linux)
python main.py \
    --input ./data/my_sequence \
    --output ./results \
    --mode debug \
    --camera-position bottom_left \
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

## Input Data Requirements

### Image Sequence
- Format: JPG or PNG images
- Naming convention: frame000000.jpg, frame000001.jpg, etc.
- Resolution: Flexible (tested with both IR and GoPro cameras)
- Directory structure: Place images in `data/input_trip/`

### Ground Truth Data (Optional)
For validation purposes, you can provide:
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
```python
class TrackerConfig:
    min_frames_threshold: int = 5
    tolerance_threshold: int = 10
    confidence_threshold: float = 0.4
    valid_vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
```

## Output Format

### CSV Output (vehicle_passing.csv)
```csv
pass_id,track_id,first_frame,last_frame,vehicle_class
1,145,1500,1530,2  # Car passing event
2,146,1600,1625,3  # Motorcycle passing event
```
Columns:
- pass_id: Unique identifier for each passing event
- track_id: Internal tracking ID
- first_frame: Start frame of passing event
- last_frame: End frame of passing event
- vehicle_class: Vehicle type (2:Car, 3:Motorcycle, 5:Bus, 7:Truck)

### Visualizations
- Real-time display during processing
- Red bounding boxes around detected passing vehicles
- Vehicle class and ID labels
- All frames saved to `output_folder/inference_images/`


## Quick Start Examples

### 1. Basic Usage with Default Settings
```python
from src.tracker import VehiclePassTracker

# Basic usage with minimum configuration
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

# Custom configuration
config = TrackerConfig(
    min_frames_threshold=7,        # Minimum frames to confirm passing
    confidence_threshold=0.5,      # Higher confidence for detections
    valid_vehicle_classes=[2, 3]   # Only track cars and motorcycles
)

tracker = VehiclePassTracker(
    image_sequence_path='./data/my_sequence/',
    output_folder='./my_results/',
    mode='debug',                  # Enable detailed visualization
    image_source_position='bottom_left',
    config=config
)
tracker.process_sequence()
```

## Result Validation
```python
# Setup with ground truth comparison
tracker = VehiclePassTracker(
    image_sequence_path='./data/sequence/',
    output_folder='./results/',
    excluded_frames_path='./data/excluded_frames.csv'  # Skip unwanted frames
)
```

## Parameter Tuning Guide

### Detection Quality
- Increase `confidence_threshold` for fewer false positives
- Decrease for better detection of distant vehicles
- Default: 0.4 (optimized based on validation)

### Event Validation
- `min_frames_threshold`: Higher values (>5) for more reliable detection
- `tolerance_threshold`: Adjust based on frame rate
- `min_angle_change`: Modify based on camera angle

## Common Issues and Solutions

### Poor Detection
- Check image quality and lighting
- Adjust confidence threshold
- Verify camera position matches configuration

### False Positives
- Increase min_frames_threshold
- Increase confidence threshold
- Verify camera position setting

### Missing Events
- Lower confidence threshold
- Reduce min_frames_threshold
- Check excluded frames

### Processing Speed
- Enable GPU acceleration when available
- Adjust frame processing rate if needed
- Optimize image resolution for your use case

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citing This Work

If you use this software in your research, please cite:
```bibtex
@software{vehicle_pass_tracker,
  author = {Padmanaban, Gandhimathi and Feng, Fred},
  title = {Vehicle Pass Tracker: An Automated Overtaking Event Event Detection System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/fenggroup/vehicle-pass-tracker.git}
}
```

## Directory Structure
```
vehicle-pass-tracker/
├── data/
│   └── input_trip/          # Place input image sequence here
│       └── frame000000.jpg
├── results/
│   ├── inference_images/    # Annotated output frames
│   └── vehicle_passing.csv  # Detection results
└── src/                     # Source code
└── *.ipynb                   # Analysis scripts
```

## Troubleshooting
- Ensure image sequence follows correct naming convention
- Check camera position matches your setup
- Verify input paths exist and are accessible
- Monitor RAM usage for large sequences
- On macOS, grant necessary permissions if using protected directories
- Use forward slashes (/) in paths for cross-platform compatibility


## License
MIT License - See LICENSE file for details

## Support

For issues and questions:
1. Check the [Issues](https://github.com/fenggroup/vehicle-pass-tracker/issues) page
2. Review common problems in Troubleshooting section
3. Open a new issue with:
   - System details
   - Error messages
   - Sample data (if possible)
