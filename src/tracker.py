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
from .utils import calculate_angle, get_sorted_images, get_class_name, setup_detailed_logger
from .tracker_types import VehiclePassEvent, TrackerConfig

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
        self.min_passing_frames_threshold = self.config.min_passing_frames_threshold
        self.buffer_frames = self.config.buffer_frames
        self.tolerance_threshold = self.config.tolerance_threshold
        self.confidence_threshold = self.config.confidence_threshold
        self.min_angle_change = self.config.min_angle_change

        self._setup_paths(image_sequence_path, output_folder)
        self._initialize_models()
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Setup file logging
        log_file = os.path.join(output_folder, 'tracking_debug.log')
        file_handler = logging.FileHandler(log_file, mode='w')  # 'w' mode to start fresh
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Remove dual logging handlers
        logging.getLogger('__main__').handlers = []
        
        # Setup detailed logging
        self.log_buffer = setup_detailed_logger(output_folder)
        
        # Algorithm settings
        self.mode = mode
        self.image_source_position = image_source_position
        self._load_exclusions(excluded_frames_path)
        self.image_sequence_length = 0
        
        # Initialize tracking state
        self._initialize_tracking_storage()
        self.current_angles = {}

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
            'vehicle_class': [],
            'passing_frame': []
        }

        self.active_tracks = {}
        self.completed_pass_ids = set()
        self.pass_count = 0
        self.previous_angles = {}
        self.angle_history = {}
        self.potential_passing = set()  # Tracks marked as potential passing
        self.confirmed_passing = set()  # Tracks that have passed 20 degrees
        self.current_angles = {}  # Current angles of active tracks
        self.potential_passing_tracks = {}  # Track data for potential passings
        self.confirmed_passing_frames = {}  # Store first frame of confirmed passing

    def is_valid_passing(self, track_id, detection_idx, detections, frame_width, frame_height, current_frame):
        """Validate if detection represents a passing event"""
        # Check if it's in excluded range
        if self._is_frame_range_excluded(current_frame, current_frame):
            return False
        
        # Calculate reference points based on camera position
        if self.image_source_position == 'bottom_center':
            # Set confirmation point at 85% of the frame width
            confirmation_x = frame_width * 0.85
        else: # bottom_left
            # Set confirmation point at 65% of the frame width
            confirmation_x = frame_width * 0.65

        # Get current position
        x1, y1, x2, y2 = detections.xyxy[detection_idx]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2        
        
        # Calculate and store current angle
        current_angle = calculate_angle(center_x, center_y, frame_height, frame_width, self.image_source_position)
        self.current_angles[track_id] = current_angle
        
        logger.info(f"Frame {current_frame} - Track {track_id}: current_angle={current_angle:.2f}, "
                    f"center_x={center_x:.2f}, center_y={center_y:.2f}, right_x={x2:.2f}, confirmation_x={confirmation_x:.2f}")
        
        # Check if it's a valid vehicle class
        if detections.class_id[detection_idx] not in self.valid_classes:
            return False
        
        # Check if vehicle is in right half of the frame (left to the bicycle)
        if not (0 <= current_angle <= 90):
            return False
            
        # Initialize angle tracking for new vehicles
        if track_id not in self.previous_angles:
            self.previous_angles[track_id] = current_angle
            self.angle_history[track_id] = [current_angle]
            return False
        
        # Update and maintain angle history
        self.angle_history[track_id].append(current_angle)
        if len(self.angle_history[track_id]) > 5:
            self.angle_history[track_id].pop(0)
        
        if track_id not in self.potential_passing:
            logger.debug(f"Track {track_id}: angle history len={len(self.angle_history[track_id])}, "
                         f"angles={self.angle_history[track_id]}")
        
        # Check if angle is consistently increasing (moving left to right)
        if len(self.angle_history[track_id]) >= 3:
            angles = self.angle_history[track_id]
            is_increasing = all(angles[i] < angles[i+1] for i in range(len(angles)-1))
            significant_change = (angles[-1] - angles[0]) >= self.min_angle_change
            
            if is_increasing and significant_change:
                # Track new potential passing event
                if track_id not in self.potential_passing:
                    self.potential_passing.add(track_id)
                    self.pass_count += 1
                    self.potential_passing_tracks[track_id] = {
                        'first_frame': current_frame,
                        'last_seen': current_frame,
                        'vehicle_class': detections.class_id[detection_idx],
                        'passing_id': self.pass_count
                    }
                else:
                    self.potential_passing_tracks[track_id]['last_seen'] = current_frame
                
                # Confirm passing based on right edge of bounding box position
                if x2 >= confirmation_x and track_id not in self.confirmed_passing:
                    self.confirmed_passing.add(track_id)
                    self.confirmed_passing_frames[track_id] = current_frame
                return True
                
        return False

    def process_sequence(self):
        """Process the entire image sequence"""

        # Get sorted image sequence
        sorted_image_sequence = get_sorted_images(self.image_sequence_path)

        self.image_sequence_length = len(sorted_image_sequence)

        # Initialize frame counter
        current_frame = 0
        
        try:
            for image in sorted_image_sequence:
                
                # Read frame
                frame_path = os.path.join(self.image_sequence_path, image)
                frame = cv2.imread(frame_path)
                if frame is None:
                    break
                
                # Get frame dimensions
                frame_height, frame_width = frame.shape[:2]

                # Run YOLOv5 inference
                results = self.model(frame, conf=self.confidence_threshold)[0]

                # Convert detections to supervision format
                detections = sv.Detections.from_ultralytics(results)
                
                # Update tracks with ByteTrack results
                detections = self.tracker.update_with_detections(detections)
                
                # Log detection details
                self._log_frame_details(current_frame, results, detections)

                # Process each track
                self.process_tracks(detections, frame_width, current_frame)

                # Draw visualizations
                annotated_frame = draw_visualizations(
                    frame, detections, current_frame, 
                    self.confirmed_passing,
                    self.potential_passing,    
                    self.current_angles,
                    self.mode, 
                    self.image_source_position, 
                    self.potential_passing_tracks
                )
                
                # Save output frame
                output_path = os.path.join(self.output_image_folder, f'output_{current_frame}.jpg')
                cv2.imwrite(output_path, annotated_frame)
                
                cv2.imshow("Vehicle Pass Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Display progress in log
                if current_frame % 100 == 0:
                    print(f"Processing frame: {current_frame}")
                
                # Force log flush every 10 frames
                if current_frame % 10 == 0:
                    for handler in logging.getLogger().handlers:
                        handler.flush()
                
                current_frame += 1
        finally:
            # Ensure all logs are written
            for handler in logging.getLogger().handlers:
                handler.flush()
                
            # Flush the log buffer
            if hasattr(self, 'log_buffer'):
                self.log_buffer.flush(force=True)

        # Final cleanup
        cv2.destroyAllWindows()

        # Save tracking data to CSV
        self.save_to_csv()

    def _log_frame_details(self, frame_number, yolo_results, detections):
        """Log detailed information about the current frame"""
        if len(detections) > 0 or self.potential_passing or self.confirmed_passing:
            log_entry = []
            log_entry.append(f"\nFrame {frame_number}:")
            
            # Log YOLO detections
            if len(detections) > 0:
                log_entry.append("  Detections:")
                valid_indices = range(min(len(detections.class_id), 
                                       len(yolo_results.boxes.conf)))
                
                for i in valid_indices:
                    class_id = detections.class_id[i]
                    if class_id in self.valid_classes:
                        track_id = detections.tracker_id[i]
                        class_name = get_class_name(class_id)
                        conf = yolo_results.boxes.conf[i]
                        log_entry.append(f"    - {class_name} "
                                    f"(Detection #{i}, ByteTrack ID: {track_id}, "
                                    f"Confidence: {conf:.2f})")
            
            # Log tracking status
            if self.potential_passing:
                log_entry.append("  Potential Passing ByteTrack IDs: " + 
                               ", ".join(map(str, self.potential_passing)))
            if self.confirmed_passing:
                log_entry.append("  Confirmed Passing ByteTrack IDs: " + 
                               ", ".join(map(str, self.confirmed_passing)))
            
            # Log directly using logger
            message = "\n".join(log_entry)
            logging.info(message)

    def process_tracks(self, detections, frame_width, current_frame):
        """Process detected tracks and update passing events"""
        if len(detections) == 0:
            return
            
        frame_height = frame_width * 9 // 16  # Assuming 16:9 aspect ratio
        active_ids_this_frame = set()
            
        for i, track_id in enumerate(detections.tracker_id):
            if self.is_valid_passing(track_id, i, detections, frame_width, frame_height, current_frame):
                active_ids_this_frame.add(track_id)
            
                # Store measurements only when track is first confirmed (at passing frame)
                if track_id in self.confirmed_passing and track_id in self.confirmed_passing_frames and self.confirmed_passing_frames[track_id] == current_frame:
                    x1, y1, x2, y2 = detections.xyxy[i]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                
                # Get reference point based on camera position
                if self.image_source_position == 'bottom_center':
                    ref_x = frame_width // 2
                    ref_y = frame_height - 5
                else: # bottom_left
                    ref_x = 0
                    ref_y = frame_height - 5
                
                # Calculate distance and angle
                ref_point_distance = ((center_x - ref_x)**2 + (center_y - ref_y)**2)**0.5
                passing_angle = calculate_angle(center_x, center_y, frame_height, frame_width, self.image_source_position)
                
                # Store measurements in track data
                self.potential_passing_tracks[track_id].update({
                'ref_point_distance': ref_point_distance,
                'passing_angle': passing_angle,
                'bbox_x1': x1,
                'bbox_y1': y1,
                'bbox_x2': x2,
                'bbox_y2': y2
                })
                
                logger.info(f"Stored measurements for track {track_id} at passing frame {current_frame}:")
                logger.info(f"  bbox_coordinates: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
                logger.info(f"  ref_point_distance: {ref_point_distance:.2f}")
                logger.info(f"  passing_angle: {passing_angle:.2f}")
            
                # Update last_seen frame whenever the track is active
                if track_id in self.potential_passing_tracks:
                    self.potential_passing_tracks[track_id]['last_seen'] = current_frame
        
        # Cleanup inactive tracks in one pass
        self._cleanup_tracks(active_ids_this_frame, current_frame)

    def _cleanup_tracks(self, active_ids_this_frame, current_frame):
        """Clean up inactive tracks and record completed passing events"""
        tracks_to_remove = []
        for track_id in list(self.potential_passing_tracks.keys()):
            track_data = self.potential_passing_tracks[track_id]
            # Remove tracks that have been inactive for 30 frames
            if track_id not in active_ids_this_frame:
                if current_frame - track_data['last_seen'] > 30:
                    if track_id in self.confirmed_passing:
                        # Pass the last_seen frame directly from track_data
                        self._record_passing_event(track_id, True)
                    tracks_to_remove.append(track_id)
        
        # Clean up removed tracks
        for track_id in tracks_to_remove:
            self._remove_track(track_id)

    def _record_passing_event(self, track_id, was_confirmed=False):
        """Record a completed passing event"""
        track_data = self.potential_passing_tracks[track_id]  
              
        # Only record confirmed passings in final CSV
        if not was_confirmed:
            return
            
        first_frame = track_data['first_frame']
        last_frame = track_data['last_seen']
        
        if last_frame - first_frame <= self.min_passing_frames_threshold:
            return
            
        # Get the passing frame from confirmed_passing_frames
        passing_frame = self.confirmed_passing_frames.get(track_id, -1)
            
        # Create event with stored measurements
        event = VehiclePassEvent(
            pass_id=track_data['passing_id'],
            track_id=track_id,
            first_frame=first_frame,
            last_frame=last_frame,
            vehicle_class=track_data['vehicle_class'],
            passing_frame=passing_frame,
            ref_point_distance=track_data.get('ref_point_distance', 0),
            passing_angle=track_data.get('passing_angle', 0),
            bbox_x1=track_data.get('bbox_x1', 0),
            bbox_y1=track_data.get('bbox_y1', 0),
            bbox_x2=track_data.get('bbox_x2', 0),
            bbox_y2=track_data.get('bbox_y2', 0)
        )
            
            # Update passing_data dictionary
        self.passing_data['pass_id'].append(event.pass_id)
        self.passing_data['track_id'].append(event.track_id)
        self.passing_data['first_frame'].append(event.first_frame)
        self.passing_data['last_frame'].append(event.last_frame)
        self.passing_data['passing_frame'].append(event.passing_frame)
        self.passing_data['vehicle_class'].append(event.vehicle_class)
        self.passing_data['ref_point_distance'].append(event.ref_point_distance)
        self.passing_data['passing_angle'].append(event.passing_angle)
        self.passing_data['bbox_x1'].append(event.bbox_x1)
        self.passing_data['bbox_y1'].append(event.bbox_y1)
        self.passing_data['bbox_x2'].append(event.bbox_x2)
        self.passing_data['bbox_y2'].append(event.bbox_y2)
            
        logger.info(f"Recorded confirmed passing event for track {track_id}")

    def _remove_track(self, track_id):
        """Remove a track and its associated data"""
        if track_id in self.potential_passing_tracks:
            del self.potential_passing_tracks[track_id]
        if track_id in self.previous_angles:
            del self.previous_angles[track_id]
        if track_id in self.angle_history:
            del self.angle_history[track_id]
        if track_id in self.potential_passing:
            self.potential_passing.remove(track_id)
        if track_id in self.confirmed_passing:
            self.confirmed_passing.remove(track_id)

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
        
        # Remove detections in excluded ranges
        if self.excluded_ranges is not None:
            valid_detections = []
            for idx, row in df.iterrows():
                if not self._is_frame_range_excluded(row['first_frame'], row['last_frame']):
                    valid_detections.append(idx)
            df = df.loc[valid_detections]

        # Merge overlapping events only if they belong to the same track_id
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
                    # Only merge if it's the same track_id and frames are within tolerance
                    if (next_det['track_id'] == current['track_id'] and 
                        next_det['first_frame'] <= current['last_frame'] + self.tolerance_threshold):
                        current['last_frame'] = max(current['last_frame'], next_det['last_frame'])
                        j += 1
                    else:
                        break
                        
                # Add buffer frames
                current['first_frame'] = max(0, current['first_frame'] - self.buffer_frames)
                current['last_frame'] = min(current['last_frame'] + self.buffer_frames, self.image_sequence_length - 1)
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
            'passing_frame': df['passing_frame'].tolist(),
            'vehicle_class': df['vehicle_class'].tolist(),
            'ref_point_distance': df['ref_point_distance'].tolist(),
            'passing_angle': df['passing_angle'].tolist(),
            'bbox_x1': df['bbox_x1'].tolist(),
            'bbox_y1': df['bbox_y1'].tolist(),
            'bbox_x2': df['bbox_x2'].tolist(),
            'bbox_y2': df['bbox_y2'].tolist()
        }
        
        self.pass_count = len(df)

    def save_to_csv(self):
        """Post-process and save tracking results to CSV"""
        # Apply post-processing
        self.post_process_detections()

        # Convert to DataFrame and sort by last_frame
        df = pd.DataFrame(self.passing_data)
        df = df.sort_values('last_frame')
        
        csv_path = os.path.join(self.output_folder, 'vehicle_passing.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        print(f"Total confirmed vehicle passing events: {len(df)}")