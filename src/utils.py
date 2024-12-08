"""
Utility functions for vehicle tracking and analysis.
"""
import cv2
import numpy as np
import os

def get_class_name(class_id):
    """Get class name from class ID"""
    class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
    return class_names.get(class_id, 'Unknown')

def calculate_angle(x, y, frame_height, frame_width):
    """
    Calculate angle between vehicle position and bottom center reference
    Returns angle in degrees where:
    - 0 degrees is straight up from bottom center
    - Positive angles towards the right side (0 to 90)
    """
    # Reference point is bottom center of frame
    ref_x = frame_width / 2
    ref_y = frame_height

    # Calculate relative position
    dx = x - ref_x
    dy = ref_y - y      # Inverted because y increases downward in image

    # Return angle in degrees
    return np.degrees(np.arctan2(dx, dy))

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