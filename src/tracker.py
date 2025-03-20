"""
Core implementation of vehicle passing detection and tracking.

This module provides the main VehiclePassTracker class that implements:
- Vehicle detection using YOLOv5
- Object tracking with ByteTrack
- Angle-based passing event validation
- Event logging and visualization

Example:
    tracker = VehiclePassTracker(
        image_sequence_path='./data/images/',
        output_folder='./results/',
        mode='demo'
    )
    tracker.process_sequence()
"""

import logging
from pathlib import Path
from typing import Dict, Set, Optional

import cv2
import supervision as sv
import pandas as pd
from ultralytics import YOLO
import os

from .visualization import draw_visualizations
from .utils import calculate_angle, get_sorted_images
from .types import VehiclePassEvent, TrackerConfig

# Configure logging
logger = logging.getLogger(__name__)

class VehiclePassTracker:
    """Tracks and analyzes vehicles passing a bicycle using YOLOv5 and ByteTrack."""
    
    def __init__(
        self, 
        image_sequence_path: str, 
        output_folder: str, 
        mode: str = 'demo',
        image_source_position: str = 'bottom_center',
        excluded_frames_path: Optional[str] = None,
        config: Optional[TrackerConfig] = None
    ):
        """
        Initialize the vehicle pass tracker with paths and configuration
        
        Args:
            image_sequence_path: Path to input image sequence
            output_folder: Where to save results
            mode: Operating mode ('demo' or 'debug')
            image_source_position: Camera position reference
            excluded_frames_path: Optional CSV with frames to exclude
            config: Optional custom configuration
        """
        self.config = config or TrackerConfig()
        
        # Initialize configuration attributes
        self.valid_classes = self.config.valid_vehicle_classes
        self.min_frames_threshold = self.config.min_frames_threshold
        self.tolerance_threshold = self.config.tolerance_threshold
        self.confidence_threshold = self.config.confidence_threshold
        self.min_angle_change = self.config.min_angle_change

        self._setup_paths(image_sequence_path, output_folder)
        self._initialize_models()
        
        # Algorithm settings
        self.mode = mode
        self.image_source_position = image_source_position
        self._load_exclusions(excluded_frames_path)
        
        # Initialize tracking state
        self._initialize_tracking_storage()

    def _initialize_models(self) -> None:
        """Initialize detection and tracking models."""
        try:
            # Initialize YOLOv5 model
            self.model = YOLO('yolov5lu.pt')  # Using YOLOv5 base model
            self.tracker = sv.ByteTrack()
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def _setup_paths(self, image_sequence_path: str, output_folder: str) -> None:
        """Setup input/output paths and create necessary directories."""
        self.image_sequence_path = Path(image_sequence_path)
        self.output_folder = Path(output_folder)
        self.output_image_folder = self.output_folder / 'inference_images'
        
        # Create output directories
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_image_folder.mkdir(parents=True, exist_ok=True)

    def _load_exclusions(self, excluded_frames_path: Optional[str]) -> None:
        """Load excluded frame ranges from CSV file."""
        self.excluded_ranges = None
        if excluded_frames_path:
            try:
                excluded_df = pd.read_csv(excluded_frames_path)
                if not all(col in excluded_df.columns for col in ['start_frame', 'end_frame']):
                    logger.warning("Excluded frames CSV must have 'start_frame' and 'end_frame' columns")
                else:
                    self.excluded_ranges = excluded_df
                    logger.info(f"Loaded {len(excluded_df)} excluded frame ranges")
            except Exception as e:
                logger.error(f"Failed to load excluded frames: {e}")
                self.excluded_ranges = None

    def _initialize_tracking_storage(self):
        """Initialize data structures for tracking"""
        self.passing_data = {
            'pass_id': [],
            'track_id': [],
            'first_frame': [],
            'last_frame': [],
            'vehicle_class': []
        }

# Track active vehicles and maintain global counter
# Track active vehicles and maintain global counter
        self.active_tracks = {}
        self.completed_pass_ids = set()
        self.pass_count = 0
        self.previous_angles = {}
        self.angle_history = {}
        self.confirmed_passing = set()
        self.potential_passing = {}

    def is_valid_passing(self, track_id, detection_idx, detections, frame_width, frame_height, current_frame):
        """Validate if detection represents a passing event"""
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
            self.potential_passing[track_id] = 1
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
                self.potential_passing[track_id] = self.potential_passing.get(track_id, 0) + 1
                # Only return true if we've seen enough consecutive valid frames
                return self.potential_passing[track_id] >= self.min_frames_threshold
            
        # Reset counter if conditions not met
        self.potential_passing[track_id] = 0
        return False

    def process_sequence(self):
        """Process the entire image sequence"""

        # Get sorted image sequence
        sorted_image_sequence = get_sorted_images(self.image_sequence_path)

        # Initialize frame counter
        current_frame = 0
        
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
                                               self.confirmed_passing, self.previous_angles, self.mode, self.image_source_position, 
                                               self.passing_data, self.active_tracks)
            
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
        """Process detected tracks and update passing events"""
        if len(detections) == 0:
            return
            
        frame_height = frame_width * 9 // 16  # Assuming 16:9 aspect ratio
        active_ids_this_frame = set()
            
        for i, track_id in enumerate(detections.tracker_id):
            # Pass current_frame to is_valid_passing
            if self.is_valid_passing(track_id, i, detections, frame_width, frame_height, current_frame):
                active_ids_this_frame.add(track_id)
                
                if track_id not in self.active_tracks:
                    # Only start tracking if we have enough frames to be valid
                    self.pass_count += 1
                    self.active_tracks[track_id] = {
                        'first_frame': current_frame - self.min_frames_threshold + 1,
                        'last_seen': current_frame,
                        'vehicle_class': detections.class_id[i],
                        'passing_id': self.pass_count  # Changed from pass_id to passing_id
                    }
                    
                    # Only add to confirmed if it meets duration criteria
                    track_duration = current_frame - (current_frame - self.min_frames_threshold + 1)
                    if track_duration >= self.min_frames_threshold:
                        self.confirmed_passing.add(track_id)
                        print(f"Added track {track_id} to confirmed_passing at frame {current_frame}")
                else:
                    # Update existing track
                    self.active_tracks[track_id]['last_seen'] = current_frame
                    
                    # Check if it should be confirmed
                    track_duration = current_frame - self.active_tracks[track_id]['first_frame']
                    if track_duration >= self.min_frames_threshold and track_id not in self.confirmed_passing:
                        self.confirmed_passing.add(track_id)
                        print(f"Added track {track_id} to confirmed_passing at frame {current_frame}")
        
        self._cleanup_tracks(active_ids_this_frame, current_frame)

    def _cleanup_tracks(self, active_ids_this_frame, current_frame):
        """Clean up inactive tracks and record completed passing events"""
        tracks_to_remove = []
        for track_id in self.active_tracks:
            if track_id not in active_ids_this_frame:
                if current_frame - self.active_tracks[track_id]['last_seen'] > 30:  # 30 frames threshold
                    track_duration = self.active_tracks[track_id]['last_seen'] - self.active_tracks[track_id]['first_frame']
                    
                    if (track_id in self.confirmed_passing and 
                        track_duration >= self.min_frames_threshold and
                        not self._is_frame_range_excluded(
                            self.active_tracks[track_id]['first_frame'],
                            self.active_tracks[track_id]['last_seen']
                        )):
                        self._record_passing_event(track_id)
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self._remove_track(track_id)

    def _record_passing_event(self, track_id):
        """Record a completed passing event"""
        track_data = self.active_tracks[track_id]
        pass_id = track_data['passing_id']  # Changed from pass_id to passing_id
        
        self.passing_data['pass_id'].append(pass_id)
        self.passing_data['track_id'].append(track_id)
        self.passing_data['first_frame'].append(track_data['first_frame'])
        self.passing_data['last_frame'].append(track_data['last_seen'])
        self.passing_data['vehicle_class'].append(track_data['vehicle_class'])
        
        self.completed_pass_ids.add(pass_id)

    def _remove_track(self, track_id):
        """Remove a track and its associated data"""
        del self.active_tracks[track_id]
        if track_id in self.previous_angles:
            del self.previous_angles[track_id]
        if track_id in self.angle_history:
            del self.angle_history[track_id]
        if track_id in self.confirmed_passing:
            self.confirmed_passing.remove(track_id)
        if track_id in self.potential_passing:
            del self.potential_passing[track_id]

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
        if not self.passing_data['track_id']:  # If no detections, return early
            return

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(self.passing_data)
        
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
        self.passing_data = {
            'pass_id': df['pass_id'].tolist(),
            'track_id': df['track_id'].tolist(),
            'first_frame': df['first_frame'].tolist(),
            'last_frame': df['last_frame'].tolist(),
            'vehicle_class': df['vehicle_class'].tolist()
        }
        
        self.pass_count = len(df)

    def save_to_csv(self):
        """Post-process and save tracking results to CSV"""
        # Apply post-processing
        self.post_process_detections()

        # Convert to DataFrame and save
        df = pd.DataFrame(self.passing_data)
        csv_path = os.path.join(self.output_folder, 'vehicle_passing.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        print(f"Total vehicle passing events detected: {self.pass_count}")