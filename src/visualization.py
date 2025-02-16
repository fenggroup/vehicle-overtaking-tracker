"""
Visualization utilities for vehicle tracking and overtaking detection.
"""

import cv2
from .utils import get_class_name
from .utils import calculate_angle

def draw_visualizations(frame, detections, current_frame, confirmed_overtaking, previous_angles, mode, image_source_position, overtaking_data, active_tracks):
    """Draw detection visualizations on frame"""
    frame_height, frame_width = frame.shape[:2]
    
    # Draw frame number
    cv2.putText(frame, f'Frame: {current_frame}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw reference point
    if image_source_position == 'bottom_center':
        ref_point = (frame_width // 2, frame_height - 5)
        cv2.circle(frame, ref_point, 5, (0, 0, 255), -1)
    elif image_source_position == 'bottom_left':
        ref_point = (0, frame_height - 5)
        cv2.circle(frame, ref_point, 5, (0, 0, 255), -1)        
    
    if len(detections) > 0:
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            
            # Only show vehicles that are in confirmed_overtaking
            if track_id in confirmed_overtaking:
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw red bounding box for confirmed overtaking
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Visualization code for confirmed overtaking
                if mode == 'debug':
                    # Draw line from reference point to vehicle center
                    cv2.line(frame, ref_point, (center_x, center_y), (0, 255, 0), 1)

                    # Draw angle
                    if track_id in previous_angles:
                        current_angle = calculate_angle(center_x, center_y, frame_height, frame_width, image_source_position)
                        angle_text = f"{current_angle:.1f} deg"
                        cv2.putText(frame, angle_text, (center_x, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Get class name and overtaking ID
                class_name = get_class_name(detections.class_id[i])
                overtaking_id = None
                
                # Check if track has an assigned overtaking ID in active tracks
                if track_id in active_tracks:
                    overtaking_id = active_tracks[track_id]['overtaking_id']
                
                # If not found in active tracks, check completed events
                if overtaking_id is None and 'track_id' in overtaking_data:
                    try:
                        idx = overtaking_data['track_id'].index(track_id)
                        overtaking_id = overtaking_data['overtaking_id'][idx]
                    except ValueError:
                        pass
                
                label = f'ID: {overtaking_id} ({class_name})'
                
                # Get label dimensions
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Decide whether to put label on top or bottom
                if y1 < label_height + 20:  # If too close to top
                    label_y = y2 + label_height + 10  # Put below box
                    bg_pt1 = (x1, y2 + label_height + 20)
                    bg_pt2 = (x1 + label_width, y2)
                else:  # Put above box
                    label_y = y1 - 10
                    bg_pt1 = (x1, y1 - label_height - 10)
                    bg_pt2 = (x1 + label_width, y1)
                
                # Draw label background
                cv2.rectangle(frame, bg_pt1, bg_pt2, (0, 0, 255), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (255, 255, 255), 2)

    return frame