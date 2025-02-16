"""
Core implementation of vehicle overtaking detection and tracking using YOLOv5 and ByteTrack.
"""
from ultralytics import YOLO
import cv2
import supervision as sv
import pandas as pd
import os
from .visualization import draw_visualizations
from .utils import calculate_angle, get_sorted_images

class VehiclePassTracker:
    """
    Tracks and analyzes vehicles overtaking a bicycle using YOLOv5 and ByteTrack.
    Uses angle-based validation for overtaking detection.
    """
    
    def __init__(self, image_sequence_path, output_folder, mode='demo', 
                 image_source_position='bottom_center', 
                 excluded_frames_path=None,
                 min_frames_threshold=5,
                 tolerance_threshold=10):
        # Initialize models and trackers
        self.model = YOLO('yolov5lu.pt') # Initialize YOLO model
        self.tracker = sv.ByteTrack()   # Initialize ByteTrack tracker
        
        # Valid vehicle classes (car, motorcycle, bus, truck)
        self.valid_classes = [2, 3, 5, 7]
        
        # Path configurations
        self.image_sequence_path = image_sequence_path
        self.output_folder = output_folder
        self.output_image_folder = output_folder + '/inference_images/'
        
        # Create output folders
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.output_image_folder, exist_ok=True)
        
        # Initialize tracking storage
        self._initialize_tracking_storage()

        # Algorithm configurations
        self.mode = mode
        self.image_source_position = image_source_position

        # Post-processing configurations
        self.excluded_frames_path = excluded_frames_path
        self.min_frames_threshold = min_frames_threshold
        self.tolerance_threshold = tolerance_threshold
        self.excluded_ranges = None
        
        # Load exclusion ranges if provided
        if excluded_frames_path and os.path.exists(excluded_frames_path):
            self.excluded_ranges = pd.read_csv(excluded_frames_path)

        # Add tracking for temporary potential overtaking events
        self.potential_overtaking = {}  # track_id: consecutive valid frames count

    def _initialize_tracking_storage(self):
        """Initialize data structures for tracking"""
        self.overtaking_data = {
            'overtaking_id': [],
            'track_id': [],
            'first_frame': [],
            'last_frame': [],
            'vehicle_class': []
        }

        # Track active vehicles and maintain global counter
        self.active_tracks = {}
        self.completed_overtaking_ids = set()
        self.overtaking_count = 0

        # Add angle tracking
        self.previous_angles = {}           # track_id: previous_angle
        self.angle_history = {}             # track_id: list of recent angles
        self.min_angle_change = 2           # minimum angle change to consider as approaching
        self.confirmed_overtaking = set()   # Set of confirmed overtaking vehicles

    def is_valid_overtaking(self, track_id, detection_idx, detections, frame_width, frame_height, current_frame):
        """Validate if detection represents an overtaking event"""
        # First check if it's in excluded range
        if self._is_frame_range_excluded(current_frame, current_frame):
            return False
            
        # Check if this track already has enough valid frames
        if track_id in self.active_tracks:
            track_duration = current_frame - self.active_tracks[track_id]['first_frame']
            if track_duration < self.min_frames_threshold:
                return False

        # Get current position
        x1, y1, x2, y2 = detections.xyxy[detection_idx]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check if it's a valid vehicle class
        if detections.class_id[detection_idx] not in self.valid_classes:
            return False
        
        # Calculate current angle
        current_angle = calculate_angle(center_x, center_y, frame_height, frame_width, self.image_source_position)
        
        # Check if vehicle is in right half (angle between 0 and 90)
        if not (0 <= current_angle <= 90):
            return False
        
        # If no previous angle, store and wait for next frame
        if track_id not in self.previous_angles:
            self.previous_angles[track_id] = current_angle
            self.angle_history[track_id] = [current_angle]
            self.potential_overtaking[track_id] = 1
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
            
            if is_increasing and significant_change:
                self.potential_overtaking[track_id] = self.potential_overtaking.get(track_id, 0) + 1
                # Only return true if we've seen enough consecutive valid frames
                return self.potential_overtaking[track_id] >= self.min_frames_threshold
            
        # Reset counter if conditions not met
        self.potential_overtaking[track_id] = 0
        return False

    def process_sequence(self):
        """Process the entire image sequence"""

        # Get sorted image sequence
        sorted_image_sequence = get_sorted_images(self.image_sequence_path)

        # Initialize frame counter
        current_frame = 1
        
        for image in sorted_image_sequence:
            
            # Read frame
            frame_path = os.path.join(self.image_sequence_path, image)
            frame = cv2.imread(frame_path)
            if frame is None:
                break
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Run YOLOv5 inference
            results = self.model(frame, conf=0.4)[0]    # Confidence interval for YOLOv5 model is set to 40%

            # Convert detections to supervision format
            detections = sv.Detections.from_ultralytics(results)

            # Update tracks using the correct method
            detections = self.tracker.update_with_detections(detections)
            
            # Process each track
            self.process_tracks(detections, frame_width, current_frame)

            # Draw visualizations
            annotated_frame = draw_visualizations(frame, detections, current_frame, 
                                               self.confirmed_overtaking, self.previous_angles, self.mode, self.image_source_position, 
                                               self.overtaking_data, self.active_tracks)
            
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
            # Pass current_frame to is_valid_overtaking
            if self.is_valid_overtaking(track_id, i, detections, frame_width, frame_height, current_frame):
                active_ids_this_frame.add(track_id)
                
                if track_id not in self.active_tracks:
                    # Only start tracking if we have enough frames to be valid
                    self.overtaking_count += 1
                    self.active_tracks[track_id] = {
                        'first_frame': current_frame - self.min_frames_threshold + 1,
                        'last_seen': current_frame,
                        'vehicle_class': detections.class_id[i],
                        'overtaking_id': self.overtaking_count
                    }
                    
                    # Only add to confirmed if it meets duration criteria
                    track_duration = current_frame - (current_frame - self.min_frames_threshold + 1)
                    if track_duration >= self.min_frames_threshold:
                        self.confirmed_overtaking.add(track_id)
                        print(f"Added track {track_id} to confirmed_overtaking at frame {current_frame}")
                else:
                    # Update existing track
                    self.active_tracks[track_id]['last_seen'] = current_frame
                    
                    # Check if it should be confirmed
                    track_duration = current_frame - self.active_tracks[track_id]['first_frame']
                    if track_duration >= self.min_frames_threshold and track_id not in self.confirmed_overtaking:
                        self.confirmed_overtaking.add(track_id)
                        print(f"Added track {track_id} to confirmed_overtaking at frame {current_frame}")
        
        self._cleanup_tracks(active_ids_this_frame, current_frame)

    def _cleanup_tracks(self, active_ids_this_frame, current_frame):
        """Clean up inactive tracks and record completed overtaking events"""
        tracks_to_remove = []
        for track_id in self.active_tracks:
            if track_id not in active_ids_this_frame:
                if current_frame - self.active_tracks[track_id]['last_seen'] > 30:  # 30 frames threshold
                    track_duration = self.active_tracks[track_id]['last_seen'] - self.active_tracks[track_id]['first_frame']
                    
                    if (track_id in self.confirmed_overtaking and 
                        track_duration >= self.min_frames_threshold and
                        not self._is_frame_range_excluded(
                            self.active_tracks[track_id]['first_frame'],
                            self.active_tracks[track_id]['last_seen']
                        )):
                        self._record_overtaking_event(track_id)
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self._remove_track(track_id)

    def _record_overtaking_event(self, track_id):
        """Record a completed overtaking event"""
        track_data = self.active_tracks[track_id]
        overtaking_id = track_data['overtaking_id']
        
        self.overtaking_data['overtaking_id'].append(overtaking_id)
        self.overtaking_data['track_id'].append(track_id)
        self.overtaking_data['first_frame'].append(track_data['first_frame'])
        self.overtaking_data['last_frame'].append(track_data['last_seen'])
        self.overtaking_data['vehicle_class'].append(track_data['vehicle_class'])
        
        self.completed_overtaking_ids.add(overtaking_id)

    def _remove_track(self, track_id):
        """Remove a track and its associated data"""
        del self.active_tracks[track_id]
        if track_id in self.previous_angles:
            del self.previous_angles[track_id]
        if track_id in self.angle_history:
            del self.angle_history[track_id]
        if track_id in self.confirmed_overtaking:
            self.confirmed_overtaking.remove(track_id)
        if track_id in self.potential_overtaking:
            del self.potential_overtaking[track_id]

    def _is_frame_range_excluded(self, start_frame, end_frame):
        """Check if a frame range falls within excluded ranges"""
        if self.excluded_ranges is None:
            return False
            
        for _, exclusion in self.excluded_ranges.iterrows():
            if (start_frame >= exclusion['start_frame'] and 
                end_frame <= exclusion['end_frame']):
                return True
        return False

    def post_process_detections(self):
        """Post-process detected events to remove invalid detections"""
        if not self.overtaking_data['track_id']:  # If no detections, return early
            return

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(self.overtaking_data)
        
        # Apply filters:
        # 1. Remove detections that are too short
        df = df[df['last_frame'] - df['first_frame'] >= self.min_frames_threshold]

        # 2. Remove detections in excluded ranges
        if self.excluded_ranges is not None:
            valid_detections = []
            for idx, row in df.iterrows():
                if not self._is_frame_range_excluded(row['first_frame'], row['last_frame']):
                    valid_detections.append(idx)
            df = df.loc[valid_detections]

        # 3. Merge overlapping events only if they belong to the same track_id
        sorted_df = df.sort_values(['track_id', 'first_frame'])
        merged_detections = []
        
        # Group by track_id to handle each vehicle separately
        for _, group in sorted_df.groupby('track_id'):
            i = 0
            group_rows = group.to_dict('records')
            
            while i < len(group_rows):
                current = group_rows[i].copy()
                j = i + 1
                
                while j < len(group_rows):
                    next_det = group_rows[j]
                    # Only merge if it's the same track_id
                    if (next_det['track_id'] == current['track_id'] and 
                        next_det['first_frame'] <= current['last_frame'] + self.tolerance_threshold):
                        current['last_frame'] = max(current['last_frame'], next_det['last_frame'])
                        j += 1
                    else:
                        break
                        
                merged_detections.append(current)
                i = j
        
        # Convert merged detections back to DataFrame
        df = pd.DataFrame(merged_detections)

        # Reset the data structures with filtered results
        self.overtaking_data = {
            'overtaking_id': df['overtaking_id'].tolist(),
            'track_id': df['track_id'].tolist(),
            'first_frame': df['first_frame'].tolist(),
            'last_frame': df['last_frame'].tolist(),
            'vehicle_class': df['vehicle_class'].tolist()
        }
        
        self.overtaking_count = len(df)

    def save_to_csv(self):
        """Post-process and save tracking results to CSV"""
        # Apply post-processing
        self.post_process_detections()

        # Convert to DataFrame and save
        df = pd.DataFrame(self.overtaking_data)
        csv_path = os.path.join(self.output_folder, 'vehicle_passing.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        print(f"Total vehicle passing events detected: {self.overtaking_count}")