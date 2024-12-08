# VehiclePassTracker

An automated algorithm for detecting and analyzing vehicle overtaking events using Image sequences from real world data. The system employs YOLOv8 object detection and ByteTrack for tracking, with geometric angle analysis to validate vehicle passing maneuvers.

## Features
- Vehicle detection using YOLOv8
- Object tracking with ByteTrack
- Angle-based vehicle passing validation
- Real-time visualization
- Comprehensive event logging
- Frame-by-frame analysis
- CSV export of detected events

## Repository Structure

vehicle-pass-tracker/
├── data/
│ ├── images/
│ │ ├── trip1/ # Place Trip 1 IR images here
│ │ └── trip2/ # Place Trip 2 IR images here
│ └── ground_truth_annotations/
│ ├── trip1/ # Place Trip 1 ground truth CSV files here
│ └── trip2/ # Place Trip 2 ground truth CSV files here
├── results/
│ └── overtaking_tracker/
│ ├── trip1/ # Results for Trip 1
│ └── trip2/ # Results for Trip 2
├── src/
│ ├── tracker.py # Core implementation
│ └── utils.py # Utility functions
├── main_trip1.py # Trip 1 execution script
├── main_trip2.py # Trip 2 execution script
└── requirements.txt

## Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 4GB+ GPU memory
- Storage space for output images

## Output
Annotated frames with vehicle detection and tracking
CSV file with vehicle passing event details
Real-time visualization of tracking process

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/vehicle-pass-tracker.git
cd vehicle-pass-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download image sequences:
Download Trip 1 images from [link in email] → data/images/trip1/
Download Trip 2 images from [link in email] → data/images/trip2/
<!-- Download Trip 1 ground truth from [link] → data/ground_truth_annotations/trip1/
Download Trip 2 ground truth from [link] → data/ground_truth_annotations/trip2/ -->

## Data Requirements
1. Image Sequence:
    Format: JPG images named as extract{frame_number}.jpg
    Example: extract1500.jpg, extract1501.jpg, etc.
2. Ground Truth Annotations:
    Format: CSV files:
        'ground_truth_xxxx_xx_xx.csv'
        'excluded_frames_xxxx_xx_xx.csv'

## Usage
1. For Trip 1 analysis:
```bash
python main_trip1.py
```

2. For Trip 2 analysis:
```bash
python main_trip2.py
```

## Output
Results are saved in results/:
    trip1/ or trip2/
        inference_images/: Annotated frames
        vehicle_passing.csv: Results of frame ranges with detected vehicle passing

## CSV Output Format
vehicle_passing_id: Unique identifier
track_id: Vehicle tracking ID
first_frame: Start frame of overtaking event
last_frame: End frame of overtaking event
vehicle_class: Type of vehicle (Car, Motorcycle, Bus, Truck)

## Analysis Tools
The repository includes overtaking_detection_analysis.ipynb for analyzing detection results:
    Features:
        Compares detection results with ground truth annotations
        Calculates performance metrics:
            Precision
            Recall
            F1 Score
        Handles frame range exclusions
        Generates video clips for false positive cases
Required Input Files:
    Detection results CSV from VehiclePassTracker
    Ground truth CSV files for each trip:
        'ground_truth_xxxx_xx_xx.csv'
        'excluded_frames_xxxx_xx_xx.csv'
    Inference images for visualization