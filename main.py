from src.tracker import VehiclePassTracker
from .src.utils import log_timing
import time


def main():    
    # Trip 1 IR
    start_time = time.time()
    print(start_time)
    tracker = VehiclePassTracker(
        image_sequence_path='./data/input_trip/',
        output_folder='./results/',
        mode='demo',
        image_source_position='bottom_center',
        excluded_frames_path='./data/ground_truth_annotations/trip1/excluded_frames_2023_07_26.csv'
    )
    tracker.process_sequence()
    end_time = time.time()
    log_timing('./results/', 'Trip Label', start_time, end_time)

if __name__ == "__main__":
    main()