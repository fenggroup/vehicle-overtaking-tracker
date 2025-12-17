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
    
    # Draw reference point (debug mode only)
    if mode == 'debug':
        if image_source_position == 'bottom_center':
            ref_point = (frame_width // 2, frame_height - 5)
        elif image_source_position == 'bottom_left':
            ref_point = (0, frame_height - 5)
        cv2.circle(annotated, ref_point, 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    if len(detections) > 0:
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            
            if track_id in potential_passing or track_id in confirmed_passing:
                if track_id not in active_tracks:
                    continue
                
                # Convert to integer coordinates
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Determine box color
                if track_id in confirmed_passing:
                    box_color = (0, 0, 255)  # Red: confirmed passing
                else:
                    box_color = (0, 255, 255)  # Yellow: potential passing

                # Draw anti-aliased rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3, lineType=cv2.LINE_AA)

                # Get track data for label
                track_data = active_tracks.get(track_id, {})
                passing_id = track_data.get('passing_id', '?')
                class_name = get_class_name(detections.class_id[i])

                label = f"{class_name} ID:{passing_id}"

                # Calculate text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Position label at top of box with padding
                label_x = x1
                label_y = max(th + baseline + 4, y1 - 4)  # Ensure label doesn't go above frame
                
                # Draw solid background rectangle for label
                bg_x1 = label_x
                bg_y1 = label_y - th - baseline - 4
                bg_x2 = label_x + tw + 4
                bg_y2 = label_y
                
                # Ensure background stays within frame bounds
                bg_y1 = max(0, bg_y1)
                bg_x2 = min(frame_width, bg_x2)
                
                # Draw filled background rectangle
                cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1, lineType=cv2.LINE_AA)
                
                # Draw text with anti-aliasing (black text on colored background)
                text_x = bg_x1 + 2
                text_y = bg_y2 - baseline - 2
                cv2.putText(
                    annotated,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text
                    thickness,
                    cv2.LINE_AA
                )

                # Debug mode: draw angle line and text
                if mode == 'debug':
                    if image_source_position == 'bottom_center':
                        ref_point = (frame_width // 2, frame_height - 5)
                    elif image_source_position == 'bottom_left':
                        ref_point = (0, frame_height - 5)
                    
                    cv2.line(annotated, ref_point, (center_x, center_y), (0, 255, 0), 1, lineType=cv2.LINE_AA)
                    current_angle = current_angles.get(track_id, 0)
                    angle_text = f"{current_angle:.1f} deg"
                    
                    # Draw angle text with background
                    (aw, ah), abaseline = cv2.getTextSize(angle_text, font, font_scale, thickness)
                    angle_bg_y1 = max(0, y2 + 4)
                    angle_bg_y2 = angle_bg_y1 + ah + abaseline + 4
                    angle_bg_x2 = min(frame_width, x1 + aw + 4)
                    cv2.rectangle(annotated, (x1, angle_bg_y1), (angle_bg_x2, angle_bg_y2), (0, 0, 255), -1, lineType=cv2.LINE_AA)
                    cv2.putText(annotated, angle_text, (x1 + 2, angle_bg_y2 - abaseline - 2),
                                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated
