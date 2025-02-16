"""
Main execution script for vehicle passing detection.
Provides a simple command-line interface for running the tracker.
"""
import argparse
from pathlib import Path
import logging
import time

from src.tracker import VehiclePassTracker
from src.types import TrackerConfig
from src.utils import log_timing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Vehicle Pass Tracker')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input image sequence folder')
    parser.add_argument('--output', type=str, default='./results',
                      help='Path to output folder')
    parser.add_argument('--mode', type=str, default='demo',
                      choices=['demo', 'debug'], help='Operation mode')
    parser.add_argument('--camera-position', type=str, 
                      default='bottom_center',
                      choices=['bottom_center', 'bottom_left'],
                      help='Camera position reference')
    parser.add_argument('--excluded-frames', type=str,
                      help='Optional CSV file with frames to exclude')
    return parser.parse_args()

def main():    
    """Run vehicle passing detection on input sequence."""
    args = parse_args()
    start_time = time.time()
    logger.info("Starting vehicle passing detection")

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    tracker = VehiclePassTracker(
        image_sequence_path=args.input,
        output_folder=args.output,
        mode=args.mode,
        image_source_position=args.camera_position,
        excluded_frames_path=args.excluded_frames
    )
    
    tracker.process_sequence()
    end_time = time.time()
    log_timing(args.output, 'Vehicle Passing Analysis', start_time, end_time)
    logger.info("Processing complete")

if __name__ == "__main__":
    main()