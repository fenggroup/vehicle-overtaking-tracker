"""
Visualization utilities for vehicle tracking and passing event detection.
"""

import cv2
from .utils import get_class_name

def draw_visualizations(
    frame, 
    detections, 
    current_frame, 
    confirmed_passing,
    potential_passing,
    current_angles,
    mode, 
    image_source_position, 
    active_tracks
):
    """Draw detection visualizations on frame."""
    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]
    
    # Draw reference point
    if image_source_position == 'bottom_center':
        ref_point = (frame_width // 2, frame_height - 5)
    elif image_source_position == 'bottom_left':
        ref_point = (0, frame_height - 5)
    
    if mode == 'debug':
        cv2.circle(annotated, ref_point, 5, (0, 0, 255), -1)

    if len(detections) > 0:
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            
            if track_id in potential_passing or track_id in confirmed_passing:
                if track_id not in active_tracks:
                    continue
                    
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                if track_id in confirmed_passing:
                    box_color = (0, 0, 255)  # Red: confirmed passing
                else:
                    box_color = (0, 255, 255)  # Yellow: potential passing

                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 5)

                if mode == 'debug':
                    cv2.line(annotated, ref_point, (center_x, center_y), (0, 255, 0), 1)
                    current_angle = current_angles.get(track_id, 0)
                    angle_text = f"{current_angle:.1f}°"
                    cv2.putText(annotated, angle_text, (center_x, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                label = f"ID: {active_tracks[track_id]['passing_id']} ({get_class_name(detections.class_id[i])})"
                
                # Position label above or below box
                y_pos = y1 - 10 if y1 > 30 else y2 + 20
                cv2.putText(annotated, label, (x1, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    return annotated