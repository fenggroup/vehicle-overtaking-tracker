"""
Utility functions for vehicle tracking and analysis.
"""
import cv2
import numpy as np
import os
from natsort import natsorted
from datetime import datetime

def get_sorted_images(folder_path):
    """Get all image files in a folder sorted by filename."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")

    # Sort using natural sort for proper numeric ordering
    return natsorted(image_files)    

def get_class_name(class_id):
    """Get class name from class ID"""
    class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
    return class_names.get(class_id, 'Unknown')

def get_reference_point(frame_height, frame_width, refn='bottom_center'):
    """Get reference point coordinates based on camera position."""
    if refn == 'bottom_center':
        return frame_width / 2, frame_height
    elif refn == 'bottom_left':
        return 0, frame_height
    else:
        raise ValueError(f"Unknown reference point type: {refn}")

def calculate_angle(x, y, frame_height, frame_width, image_source_position):
    """Calculate angle between vehicle position and reference point.
    
    Returns angle in degrees where 0° is straight ahead from the reference point
    and positive angles indicate rightward motion.
    """
    ref_x, ref_y = get_reference_point(frame_height, frame_width, image_source_position)
    
    # Calculate vector components from reference to vehicle
    dx = x - ref_x
    dy = ref_y - y  # Inverted because y increases downward in image
    
    angle = np.degrees(np.arctan2(dx, dy))
    
    # Normalize angle to 0-90 range for right side passing
    return angle if 0 <= angle <= 90 else -1

def log_timing(output_folder, sequence_name, start_time, end_time, frame_count=None):
    """Log timing and performance information."""
    import logging
    
    duration = end_time - start_time
    timing_msg = [
        f'\n{"="*60}',
        f'{sequence_name}',
        f'Start time: {datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}',
        f'End time: {datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")}',
        f'Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)'
    ]
    
    if frame_count:
        fps = frame_count / duration if duration > 0 else 0
        timing_msg.append(f'Frames processed: {frame_count}')
        timing_msg.append(f'Average speed: {fps:.2f} fps')
    
    timing_msg.append(f'{"="*60}\n')
    
    logger = logging.getLogger(__name__)
    for line in timing_msg:
        logger.info(line)