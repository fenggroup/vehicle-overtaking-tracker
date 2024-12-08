"""
Core implementation of vehicle overtaking detection and tracking using YOLOv8 and ByteTrack.
"""
from ultralytics import YOLO
import cv2
import supervision as sv
import pandas as pd
import os
import numpy as np
from .visualization import draw_visualizations
from .utils import calculate_angle

class VehiclePassTracker:
    """
    Tracks and analyzes vehicles overtaking a bicycle using YOLOv8 and ByteTrack.
    Uses angle-based validation for overtaking detection.
    """
    
    def __init__(self, image_sequence_path, start_frame, end_frame, output_folder):
        # Initialize models and trackers
        self.model = YOLO('yolov8m.pt') # Initialize YOLO model
        self.tracker = sv.ByteTrack()   # Initialize ByteTrack tracker
        
        # Valid vehicle classes (car, motorcycle, bus, truck)
        self.valid_classes = [2, 3, 5, 7]
        
        # Path configurations
        self.image_sequence_path = image_sequence_path
        self.output_folder = output_folder
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.output_image_folder = output_folder + '/inference_images/'
        
        # Create output folders
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.output_image_folder, exist_ok=True)
        
        # Initialize tracking storage
        self._initialize_tracking_storage()

    def _initialize_tracking_storage(self):
        """Initialize data structures for tracking"""
        self.overtaking_data = {
            'overtaking_id': [],
            'track_id': [],
            'first_frame': [],
            'last_frame': [],
            'vehicle_class': []
        }

        # Track active vehicles
        self.active_tracks = {}
        self.overtaking_count = 0

        # Add angle tracking
        self.previous_angles = {}           # track_id: previous_angle
        self.angle_history = {}             # track_id: list of recent angles
        self.min_angle_change = 2           # minimum angle change to consider as approaching
        self.confirmed_overtaking = set()   # Set of confirmed overtaking vehicles

    def is_valid_overtaking(self, track_id, detection_idx, detections, frame_width, frame_height):
        """Validate if detection represents an overtaking event"""
        # Get current position
        x1, y1, x2, y2 = detections.xyxy[detection_idx]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check if it's a valid vehicle class
        if detections.class_id[detection_idx] not in self.valid_classes:
            return False
        
        # Calculate current angle
        current_angle = calculate_angle(center_x, center_y, frame_height, frame_width)
        
        # Check if vehicle is in right half (angle between 0 and 90)
        if not (0 <= current_angle <= 90):
            return False
        
        # If no previous angle, store and wait for next frame
        if track_id not in self.previous_angles:
            self.previous_angles[track_id] = current_angle
            self.angle_history[track_id] = [current_angle]
            return False
        
        # Update angle history
        self.angle_history[track_id].append(current_angle)
        if len(self.angle_history[track_id]) > 5:
            self.angle_history[track_id].pop(0)
        
        # Check if angle is consistently increasing (moving left to right)
        if len(self.angle_history[track_id]) >= 3:
            angles = self.angle_history[track_id]
            is_increasing = all(angles[i] < angles[i+1] for i in range(len(angles)-1))
            significant_change = (angles[-1] - angles[0]) >= self.min_angle_change
            
            # Update previous angle
            self.previous_angles[track_id] = current_angle
            return is_increasing and significant_change
        
        # Update previous angle
        self.previous_angles[track_id] = current_angle
        return False

    def process_sequence(self):
        """Process the entire image sequence"""
        current_frame = self.start_frame
        
        while current_frame <= self.end_frame:
            # Read frame
            frame_path = os.path.join(self.image_sequence_path, f'extract{current_frame}.jpg')
            frame = cv2.imread(frame_path)
            if frame is None:
                break
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Run YOLOv8 inference
            results = self.model(frame, conf=0.4)[0]    # Confidence interval for YOLOv8 model is set to 40%

            # Convert detections to supervision format
            detections = sv.Detections.from_ultralytics(results)

            # Update tracks using the correct method
            detections = self.tracker.update_with_detections(detections)
            
            # Process each track
            self.process_tracks(detections, frame_width, current_frame)

            # Draw visualizations
            annotated_frame = draw_visualizations(frame, detections, current_frame, 
                                               self.confirmed_overtaking, self.previous_angles)
            
            # Save output frame
            output_path = os.path.join(self.output_image_folder, f'output_{current_frame}.jpg')
            cv2.imwrite(output_path, annotated_frame)
            
            cv2.imshow("Vehicle Pass Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Display progress in log
            if current_frame % 100 == 0:
                print(f"Processing frame: {current_frame}")
            
            current_frame += 1
        
        cv2.destroyAllWindows()

        # Save tracking data to CSV
        self.save_to_csv()

    def process_tracks(self, detections, frame_width, current_frame):
        """Process detected tracks and update overtaking events"""
        if len(detections) == 0:
            return
            
        frame_height = frame_width * 9 // 16  # Assuming 16:9 aspect ratio
        active_ids_this_frame = set()
            
        for i, track_id in enumerate(detections.tracker_id):
            if self.is_valid_overtaking(track_id, i, detections, frame_width, frame_height):
                active_ids_this_frame.add(track_id)
                
                if track_id not in self.active_tracks:
                    self.active_tracks[track_id] = {
                        'first_frame': current_frame,
                        'last_seen': current_frame,
                        'vehicle_class': detections.class_id[i]
                    }
                    self.confirmed_overtaking.add(track_id)
                else:
                    # Update existing track
                    self.active_tracks[track_id]['last_seen'] = current_frame
        
        self._cleanup_tracks(active_ids_this_frame, current_frame)

    def _cleanup_tracks(self, active_ids_this_frame, current_frame):
        """Clean up inactive tracks and record completed overtaking events"""
        tracks_to_remove = []
        for track_id in self.active_tracks:
            if track_id not in active_ids_this_frame:
                if current_frame - self.active_tracks[track_id]['last_seen'] > 30:  # 30 frames threshold
                    if track_id in self.confirmed_overtaking:
                        # Record only if it was a confirmed overtaking event
                        self._record_overtaking_event(track_id)
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self._remove_track(track_id)

    def _record_overtaking_event(self, track_id):
        """Record a completed overtaking event"""
        self.overtaking_count += 1
        self.overtaking_data['overtaking_id'].append(self.overtaking_count)
        self.overtaking_data['track_id'].append(track_id)
        self.overtaking_data['first_frame'].append(self.active_tracks[track_id]['first_frame'])
        self.overtaking_data['last_frame'].append(self.active_tracks[track_id]['last_seen'])
        self.overtaking_data['vehicle_class'].append(self.active_tracks[track_id]['vehicle_class'])

    def _remove_track(self, track_id):
        """Remove a track and its associated data"""
        del self.active_tracks[track_id]
        if track_id in self.previous_angles:
            del self.previous_angles[track_id]
        if track_id in self.angle_history:
            del self.angle_history[track_id]
        if track_id in self.confirmed_overtaking:
            self.confirmed_overtaking.remove(track_id)

    def save_to_csv(self):
        """Save tracking results to CSV"""
        df = pd.DataFrame(self.overtaking_data)
        csv_path = os.path.join(self.output_folder, 'vehicle_passing.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        print(f"Total vehicle passing events detected: {self.overtaking_count}")