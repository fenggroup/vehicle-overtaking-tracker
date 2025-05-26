"""
Utility functions for vehicle tracking and analysis.
"""
import cv2
import numpy as np
import os
from natsort import natsorted
from datetime import datetime

def get_sorted_images(folder_path):
    """Get all image files in a folder sorted by filename"""

    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all image files (assuming jpg/jpeg/png extensions)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Raise an error if no image files are found
    if not image_files:
        raise ValueError(f"No image files found in the folder '{folder_path}'.")

    # Sort files naturally using natsort
    return natsorted(image_files)    

def get_class_name(class_id):
    """Get class name from class ID"""
    class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
    return class_names.get(class_id, 'Unknown')

def get_reference_point(frame_height, frame_width, refn='bottom_center'):
    """Get reference point coordinates based on image source (eg. camera) position or setup"""
    if refn == 'bottom_center':
        return frame_width / 2, frame_height
    elif refn == 'bottom_left':
        return 0, frame_height
    else:
        raise ValueError(f"Unknown reference point type: {refn}")

def calculate_angle(x, y, frame_height, frame_width, image_source_position):
    """
    Calculate angle between vehicle position and reference point
    Returns angle in degrees where:
    - 0 degrees is straight up from bottom center
    - Positive angles towards the right side (0 to 90)
    """
    # Get reference point based on camera position
    ref_x, ref_y = get_reference_point(frame_height, frame_width, image_source_position)
    
    # Calculate vector components from reference to vehicle
    dx = x - ref_x
    dy = ref_y - y  # Inverted because y increases downward in image
    
    # Calculate angle using arctan2 for correct quadrant handling
    angle = np.degrees(np.arctan2(dx, dy))
    
    # Normalize angle to 0-90 range for right side passing
    return angle if 0 <= angle <= 90 else -1

def images_to_video(image_folder: str, 
                   start_frame: int, 
                   end_frame: int, 
                   output_file: str, 
                   fps: int = 10) -> None:
    """
    Convert a sequence of images to video with frame range expansion.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Adjust frame range to include buffer
    start_frame = max(1, start_frame - 5)
    end_frame = end_frame + 5

    # Get image files for the specified range
    images = []
    current_frame = start_frame
    while current_frame <= end_frame:
        image_path = os.path.join(image_folder, f'output_{current_frame}.jpg')
        if os.path.exists(image_path):
            images.append(image_path)
        current_frame += 1

    if not images:
        print("No images found in the specified range.")
        return

    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write frames to video
    for image_path in images:
        video.write(cv2.imread(image_path))

    # Release resources
    cv2.destroyAllWindows()
    video.release()

    print(f"Video saved as {output_file}")

def log_timing(output_folder, sequence_name, start_time, end_time):
    log_path = os.path.join(output_folder, 'timing_log.txt')
    with open(log_path, 'a') as f:
        f.write(f'{sequence_name}\n')
        print(start_time)
        f.write(f'Start time: {datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'End time: {datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Duration: {end_time - start_time:.2f} seconds\n\n')
        print(end_time)

def setup_detailed_logger(output_folder):
    """Setup detailed algorithm logger"""
    log_path = os.path.join(output_folder, 'algorithm_log.txt')
    
    # Create a string buffer to store logs
    class LogBuffer:
        def __init__(self, filepath):
            self.buffer = []
            self.filepath = filepath
        
        def write(self, message):
            self.buffer.append(message)
            
        def flush(self, force=False):
            if force or len(self.buffer) >= 1000:  # Flush every 1000 entries or when forced
                with open(self.filepath, 'a', encoding='utf-8') as f:
                    f.write(''.join(self.buffer))
                self.buffer = []
    
    return LogBuffer(log_path)