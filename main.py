from src.tracker import VehiclePassTracker
import time
from datetime import datetime
import os

def log_timing(output_folder, sequence_name, start_time, end_time):
    log_path = os.path.join(output_folder, 'timing_log.txt')
    with open(log_path, 'a') as f:
        f.write(f'{sequence_name}\n')
        print(start_time)
        f.write(f'Start time: {datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'End time: {datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Duration: {end_time - start_time:.2f} seconds\n\n')
        print(end_time)

def main():    
    # Trip 1 IR
    start_time = time.time()
    print(start_time)
    tracker = VehiclePassTracker(
        image_sequence_path='./data/trip_1_ir/',
        start_frame=1700,
        end_frame=18425,
        output_folder='./results/trip_1_ir/'
    )
    tracker.process_sequence()
    end_time = time.time()
    log_timing('./results/trip_1_ir/', 'Trip 1 IR', start_time, end_time)

if __name__ == "__main__":
    main()