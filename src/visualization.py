"""
Visualization utilities for vehicle tracking and overtaking detection.
"""

import cv2
from .utils import get_class_name
from .utils import calculate_angle

def draw_visualizations(frame, detections, current_frame, confirmed_overtaking, previous_angles):
    """Draw detection visualizations on frame"""
    frame_height, frame_width = frame.shape[:2]
    
    # Draw frame number
    cv2.putText(frame, f'Frame: {current_frame}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw reference point at bottom center
    ref_point = (frame_width // 2, frame_height - 5)
    cv2.circle(frame, ref_point, 5, (0, 0, 255), -1)
    
    if len(detections) > 0:
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            
            if track_id in confirmed_overtaking:
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw red bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw line from reference point to vehicle center
                cv2.line(frame, ref_point, (center_x, center_y), (0, 255, 0), 1)

                # Draw angle
                if track_id in previous_angles:
                    current_angle = calculate_angle(center_x, center_y, frame_height, frame_width)
                    angle_text = f"{current_angle:.1f} deg"
                    cv2.putText(frame, angle_text, (center_x, y2 + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                # Draw track ID and class
                class_name = get_class_name(detections.class_id[i])
                label = f'ID: {track_id} ({class_name})'
                    
                # Add background to text for better visibility
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, 
                            (x1, y1-label_height-10),
                            (x1+label_width, y1),
                            (0, 0, 255),
                            -1)
                    
                # Draw white text on red background
                cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (255, 255, 255), 2)
                
    return frame