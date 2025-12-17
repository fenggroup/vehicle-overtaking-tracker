"""
Core implementation of vehicle passing detection and tracking.

This module provides the main VehiclePassTracker class that implements:
- Vehicle detection using RT-DETR via ONNX Runtime
- Object tracking with ByteTrack
- Angle-based passing event validation
- Event logging and visualization

Example:
    tracker = VehiclePassTracker(
        image_sequence_path='./data/images/',
        output_folder='./results/',
        mode='demo',
        onnx_model_path='rtdetr.onnx'
    )
    tracker.process_sequence()
"""

import logging
import os
from pathlib import Path
from typing import Optional
from queue import Queue, Empty
from threading import Thread

import cv2
import pandas as pd
import supervision as sv
import numpy as np

from .detectors import OnnxRTDetrDetector

from .visualization import draw_visualizations
from .utils import calculate_angle, get_sorted_images, get_class_name
from .tracker_types import VehiclePassEvent, TrackerConfig

# Configure logging
logger = logging.getLogger(__name__)

class VehiclePassTracker:
    """Tracks and analyzes vehicles passing a bicycle using RT-DETR and ByteTrack."""
    
    def __init__(
        self, 
        image_sequence_path: str, 
        output_folder: str, 
        mode: str = 'demo',
        image_source_position: str = 'bottom_center',
        excluded_frames_path: Optional[str] = None,
        config: Optional[TrackerConfig] = None,
        onnx_model_path: str = 'rtdetr.onnx',
        display: bool = True,
        output_format: str = 'png'
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
            onnx_model_path: Path to RT-DETR ONNX model
            display: Whether to show visualization window (False for headless)
            output_format: Image format for saved frames ('png' or 'jpg')
        """
        self.config = config or TrackerConfig()
        
        # Initialize configuration attributes
        self.valid_classes = self.config.valid_vehicle_classes
        self.min_event_duration_frames = self.config.min_event_duration_frames
        self.buffer_frames = self.config.buffer_frames
        self.tolerance_threshold = self.config.tolerance_threshold
        self.confidence_threshold = self.config.confidence_threshold
        self.cleanup_frames = self.config.cleanup_frames

        self._setup_paths(image_sequence_path, output_folder)
        self.onnx_model_path = onnx_model_path
        self.display = display
        self.output_format = output_format.lower()
        
        # Initialize background frame saving
        self.save_queue = Queue(maxsize=100)  # Buffer up to 100 frames
        self.save_thread = Thread(target=self._frame_saver_worker, daemon=True)
        self.save_thread_running = True
        self.save_thread.start()
        
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
            # Initialize ONNX RT-DETR detector
            self.detector = OnnxRTDetrDetector(
                self.onnx_model_path, 
                confidence_threshold=self.confidence_threshold
            )

            # Initialize ByteTrack with tuned parameters for RT-DETR
            # Note: supervision uses different parameter names than raw ByteTrack
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.55,      # Stricter: only create tracks for high-conf detections (was 0.25)
                lost_track_buffer=10,                 # Reduced further to limit ID reuse window (was 15)
                minimum_matching_threshold=0.70,      # Stricter: reduce cross-vehicle ID switches (was 0.8)
                frame_rate=10                         # Match actual frame rate
            )
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

    def _frame_saver_worker(self):
        """Background thread worker for saving frames to disk"""
        while self.save_thread_running:
            try:
                # Get frame from queue (timeout to allow checking running flag)
                item = self.save_queue.get(timeout=0.1)
                if item is None:  # Poison pill to stop thread
                    break
                
                output_path, frame = item
                
                # Save with appropriate format and quality
                if self.output_format == 'png':
                    # PNG: lossless, compression level 1 (fastest, still lossless)
                    cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                else:
                    # JPEG: quality 100 for best quality
                    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                
                self.save_queue.task_done()
            except Empty:
                # Queue empty, continue waiting
                continue
            except Exception as e:
                # Log actual errors but don't crash the thread
                logger.error(f"Error saving frame: {e}")
                self.save_queue.task_done()
                continue

    def _initialize_tracking_storage(self):
        """Initialize data structures for tracking"""
        self.passing_data = {
            'pass_id': [],
            'track_id': [],
            'first_frame': [],
            'last_frame': [],
            'vehicle_class': [],
            'passing_frame': [],
            'ref_point_distance': [],
            'passing_angle': [],
            'bbox_x1': [],
            'bbox_y1': [],
            'bbox_x2': [],
            'bbox_y2': []
        }

        # Initialize tracking state variables
        self.active_tracks = {}
        self.completed_pass_ids = set()
        self.pass_count = 0
        self.angle_history = {}
        self.bbox_area_history = {}  # Track bounding box area for distance/depth filtering
        self.probation_tracks = {}  # Tracks being evaluated before adding to potential_passing
        self.potential_passing = set()  # Tracks marked as potential passing
        self.confirmed_passing = set()  # Tracks that have passed 20 degrees
        self.current_angles = {}  # Current angles of active tracks
        self.potential_passing_tracks = {}  # Track data for potential passings
        self.confirmed_passing_frames = {}  # Store first frame of confirmed passing

    def suppress_duplicate_detections(self, detections: sv.Detections, iou_threshold: float = 0.90) -> sv.Detections:
        """Suppress near-identical duplicate boxes before ByteTrack (class-agnostic).

        This is intentionally conservative (high IoU) to avoid changing behavior
        beyond removing clear same-object duplicates (often caused by class flips).
        """
        if len(detections) <= 1:
            return detections

        boxes = detections.xyxy
        if boxes is None or len(boxes) <= 1:
            return detections

        scores = detections.confidence
        if scores is None or len(scores) != len(boxes):
            scores = np.ones((len(boxes),), dtype=np.float32)

        order = np.argsort(scores)[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            union = np.maximum(area_i + area_rest - inter, 1e-6)
            iou = inter / union

            order = rest[iou < iou_threshold]

        if len(keep) == len(detections):
            return detections

        keep_mask = np.zeros((len(detections),), dtype=bool)
        keep_mask[np.array(keep, dtype=np.int64)] = True
        return detections[keep_mask]
    
    def detect_same_frame_duplicates(self, confirmed_tracks_this_frame: dict) -> set:
        """
        Detect duplicate confirmations in the same frame.
        Returns set of track IDs that should be removed as duplicates.
        
        Args:
            confirmed_tracks_this_frame: Dict mapping track_id -> track info for current frame
            
        Returns:
            Set of track IDs to remove as duplicates
        """
        if len(confirmed_tracks_this_frame) <= 1:
            return set()
        
        import numpy as np
        duplicates = set()
        track_list = list(confirmed_tracks_this_frame.items())
        
        for i, (track_id1, info1) in enumerate(track_list):
            if track_id1 in duplicates:
                continue
                
            for track_id2, info2 in track_list[i+1:]:
                if track_id2 in duplicates:
                    continue
                
                angle_diff = abs(info1['angle'] - info2['angle'])
                distance_diff = abs(info1['distance'] - info2['distance'])
                
                # Calculate IoU
                bbox1 = info1['bbox']
                bbox2 = info2['bbox']
                x1_max = max(bbox1[0], bbox2[0])
                y1_max = max(bbox1[1], bbox2[1])
                x2_min = min(bbox1[2], bbox2[2])
                y2_min = min(bbox1[3], bbox2[3])
                
                intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                
                # Very strict criteria for same frame duplicates
                if (angle_diff < 0.5 and       # < 0.5° difference
                    distance_diff < 50 and     # < 50cm difference
                    iou > 0.5):                # Significant overlap
                    
                    # Keep higher confidence track, mark lower as duplicate
                    conf1 = info1.get('confidence', 0)
                    conf2 = info2.get('confidence', 0)
                    
                    if conf1 >= conf2:
                        duplicates.add(track_id2)
                        logger.info(f"Same-frame duplicate detected: Track {track_id2} is duplicate of {track_id1} "
                                  f"(angle_diff={angle_diff:.2f}°, dist_diff={distance_diff:.1f}cm, IoU={iou:.3f})")
                    else:
                        duplicates.add(track_id1)
                        logger.info(f"Same-frame duplicate detected: Track {track_id1} is duplicate of {track_id2} "
                                  f"(angle_diff={angle_diff:.2f}°, dist_diff={distance_diff:.1f}cm, IoU={iou:.3f})")
                        break  # track_id1 is marked, no need to continue inner loop
        
        return duplicates

    def is_valid_passing(self, track_id, detection_idx, detections, frame_width, frame_height, current_frame):
        """Validate if detection represents a passing event"""
        # Check if it's in excluded range
        if self._is_frame_range_excluded(current_frame, current_frame):
            return False
        
        # Calculate reference points based on camera position
        if self.image_source_position == 'bottom_center':
            # Set confirmation point using config parameter
            confirmation_x = frame_width * self.config.confirmation_line_bottom_center
        else: # bottom_left
            # Set confirmation point using config parameter
            confirmation_x = frame_width * self.config.confirmation_line_bottom_left

        # Get current position
        x1, y1, x2, y2 = detections.xyxy[detection_idx]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2        
        
        # Calculate and store current angle
        current_angle = calculate_angle(center_x, center_y, frame_height, frame_width, self.image_source_position)
        self.current_angles[track_id] = current_angle
        
        # Check if it's a valid vehicle class
        if detections.class_id[detection_idx] not in self.valid_classes:
            return False
        
        # Check if vehicle is in valid angle range (configured, default: right half of frame 0-90°)
        if not (self.config.min_angle <= current_angle <= self.config.max_angle):
            return False
            
        # Initialize angle and bbox size tracking for new vehicles
        if track_id not in self.angle_history:
            bbox_area = (x2 - x1) * (y2 - y1)
            self.angle_history[track_id] = [current_angle]
            self.bbox_area_history[track_id] = [bbox_area]
            return False
        
        # Update and maintain history (configurable window size for robust trend detection)
        bbox_area = (x2 - x1) * (y2 - y1)
        
        self.angle_history[track_id].append(current_angle)
        if len(self.angle_history[track_id]) > self.config.history_window_size:
            self.angle_history[track_id].pop(0)
        
        self.bbox_area_history[track_id].append(bbox_area)
        if len(self.bbox_area_history[track_id]) > self.config.history_window_size:
            self.bbox_area_history[track_id].pop(0)
        
        # Compute angle trend
        angle_trend = self._compute_angle_trend(track_id)
        
        bbox_area_current = (x2 - x1) * (y2 - y1)
        prev_area = self.bbox_area_history[track_id][-2] if len(self.bbox_area_history[track_id]) >= 2 else bbox_area_current
        area_growth = bbox_area_current - prev_area
        logger.info(f"Frame {current_frame} - Track {track_id}: angle={current_angle:.2f}, "
                    f"area={bbox_area_current:.0f}, darea={area_growth:.0f}, trend={angle_trend}")
        
        # Filter out oncoming traffic based on angle trend
        # Overtaking: angle increases (left→right), Oncoming: angle decreases (right→left)
        if angle_trend == 'decreasing':
            # Angle decreasing indicates oncoming traffic, or ByteTrack ID reuse.
            # If this track was already confirmed, we treat the reversal as an identity-switch
            # boundary: finalize the confirmed event ending at the previous frame, record it,
            # then reset/remove per-track state so the new vehicle can be evaluated cleanly.
            track_data = self.potential_passing_tracks.get(track_id)
            if track_data is not None and track_data.get('was_confirmed', False):
                end_frame = max(int(track_data.get('first_frame', 0)), int(current_frame) - 1)
                track_data['last_seen'] = end_frame
                logger.info(
                    f"Track {track_id}: Confirmed event ends at frame {end_frame} "
                    f"(angle trend reversed at frame {current_frame}); recording and resetting track state"
                )
                self._record_passing_event(track_id, True)
                self._remove_track(track_id)
                return False

            # Not confirmed yet: drop the candidate as oncoming/invalid.
            logger.info(f"Track {track_id}: Removing from potential/confirmed - angle trend reversed to decreasing")
            self._remove_track(track_id)
            return False
        
        if track_id not in self.potential_passing:
            logger.debug(f"Track {track_id}: angle history len={len(self.angle_history[track_id])}, "
                         f"angles={self.angle_history[track_id]}")
        
        # Check if angle is consistently increasing (moving left to right)
        if len(self.angle_history[track_id]) >= self.config.min_frames_for_evaluation:
            angles = self.angle_history[track_id]
            n = len(angles)

            # Calculate angle slope (degrees/frame) via linear regression
            x_vals = list(range(n))
            mean_x = sum(x_vals) / n
            mean_y = sum(angles) / n
            num = sum((x_vals[i] - mean_x) * (angles[i] - mean_y) for i in range(n))
            den = sum((x_vals[i] - mean_x) ** 2 for i in range(n))
            slope = (num / den) if den != 0 else 0.0

            # Count angle increases
            increase_count = sum(1 for i in range(n-1) if angles[i+1] > angles[i])
            
            # Calculate required increase count relative to history window size
            # With n frames, max possible increases is n-1
            # Require at least min_angle_increase_count, but cap at n-1 (max possible)
            required_increase_count = min(self.config.min_angle_increase_count, n - 1)

            # Check bounding box growth (approaching vehicles grow in size)
            bbox_areas = self.bbox_area_history[track_id]
            
            # Calculate bbox growth rate over history window
            bbox_growth_rate = (bbox_areas[-1] - bbox_areas[0]) / max(1, len(bbox_areas) - 1) if len(bbox_areas) >= 2 else 0
            
            # Debug logging for validation checks
            if track_id not in self.probation_tracks and track_id not in self.potential_passing:
                logger.debug(f"Frame {current_frame} - Track {track_id}: Validation checks - "
                           f"slope={slope:.3f} (need {self.config.min_angle_slope:.1f}), "
                           f"increase_count={increase_count}/{n-1} (need {required_increase_count}), "
                           f"bbox_growth_rate={bbox_growth_rate:.1f} (need {self.config.min_bbox_growth_rate:.1f}), "
                           f"history_len={n}")
            
            # Both conditions must be satisfied for valid overtaking detection
            if (
                slope >= self.config.min_angle_slope and
                increase_count >= required_increase_count and
                bbox_growth_rate >= self.config.min_bbox_growth_rate
            ):
                # Enter probation period
                if track_id not in self.probation_tracks and track_id not in self.potential_passing:
                    self.probation_tracks[track_id] = {
                        'start_frame': current_frame,
                        'frames_passed': 0
                    }
                    logger.info(f"Track {track_id}: Entered probation at frame {current_frame}")
                
                # If in probation, increment counter
                if track_id in self.probation_tracks:
                    self.probation_tracks[track_id]['frames_passed'] += 1
                    
                    # Graduate from probation after consecutive frames pass all checks
                    if self.probation_tracks[track_id]['frames_passed'] >= self.config.probation_frames:
                        if track_id not in self.potential_passing:
                            logger.info(f"Track {track_id}: Graduated from probation to potential_passing at frame {current_frame}")
                            self.potential_passing.add(track_id)
                            self.pass_count += 1
                            self.potential_passing_tracks[track_id] = {
                                'first_frame': current_frame,
                                'last_seen': current_frame,
                                'vehicle_class': detections.class_id[detection_idx],
                                'passing_id': self.pass_count,
                                'was_confirmed': False
                            }
                            del self.probation_tracks[track_id]
                
                # Update last_seen for already-passing tracks
                if track_id in self.potential_passing:
                    self.potential_passing_tracks[track_id]['last_seen'] = current_frame
                
                    # Confirm passing when vehicle reaches confirmation line
                    if x2 >= confirmation_x and track_id not in self.confirmed_passing:
                        # Collect confirmation info for duplicate suppression in this frame
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        if self.image_source_position == 'bottom_center':
                            ref_x = frame_width / 2
                            ref_y = frame_height - 5
                        else:
                            ref_x = 0
                            ref_y = frame_height - 5
                        ref_point_distance = ((center_x - ref_x) ** 2 + (center_y - ref_y) ** 2) ** 0.5
                        confidence = detections.confidence[detection_idx] if detection_idx < len(detections.confidence) else 0
                        self.confirmed_this_frame[track_id] = {
                            'angle': self.current_angles.get(track_id, 0),
                            'distance': ref_point_distance,
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence
                        }

                        self.confirmed_passing.add(track_id)
                        self.confirmed_passing_frames[track_id] = current_frame
                        self.potential_passing_tracks[track_id]['was_confirmed'] = True
                        self.potential_passing_tracks[track_id]['passing_frame'] = current_frame
                
                return True
            else:
                # Remove from probation if checks fail
                if track_id in self.probation_tracks:
                    logger.info(f"Track {track_id}: Removed from probation - failed checks")
                    del self.probation_tracks[track_id]
                
        return False

    def process_sequence(self):
        """Process the entire image sequence"""

        # Get sorted image sequence
        sorted_image_sequence = get_sorted_images(self.image_sequence_path)

        self.image_sequence_length = len(sorted_image_sequence)
        
        # Log processing start
        logger.info(f"="*60)
        logger.info(f"Starting processing of {self.image_sequence_length} frames")
        logger.info(f"Output folder: {self.output_folder}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Camera position: {self.image_source_position}")
        logger.info(f"Display: {'Enabled' if self.display else 'Disabled (headless)'}")
        logger.info(f"Output format: {self.output_format.upper()}")
        logger.info(f"="*60)

        # Initialize frame counter
        current_frame = 0
        last_logged_frame = -1  # Track last frame that was logged
        
        try:
            for image in sorted_image_sequence:
                
                # Read frame with highest quality settings
                frame_path = os.path.join(self.image_sequence_path, image)
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning(f"Failed to read frame: {frame_path}")
                    break
                
                # Get frame dimensions
                frame_height, frame_width = frame.shape[:2]

                # Run RT-DETR ONNX inference -> supervision Detections
                detections = self.detector.infer(frame)

                # suppress near-identical duplicate boxes (class-agnostic)
                # Helps prevent same-frame double IDs caused by duplicate detections/class flips.
                detections = self.suppress_duplicate_detections(detections, iou_threshold=0.90)

                # Update tracks with ByteTrack results
                detections = self.tracker.update_with_detections(detections)

                # Log detection details (with gap summary if frames were skipped due to no detections)
                self._log_frame_details(current_frame, detections, last_logged_frame)

                # Update last logged frame if this frame had content
                if len(detections) > 0 or self.potential_passing or self.confirmed_passing:
                    last_logged_frame = current_frame

                # Reset per-frame confirmation cache (used for same-frame duplicate suppression)
                self.confirmed_this_frame = {}

                # Process each track
                self.process_tracks(detections, frame_width, current_frame)
                
                # Detect and remove same-frame duplicates (ID switches / class flips on same object)
                if self.confirmed_this_frame:
                    duplicates = self.detect_same_frame_duplicates(self.confirmed_this_frame)
                    for dup_track_id in duplicates:
                        # Remove from confirmed sets
                        if dup_track_id in self.confirmed_passing:
                            self.confirmed_passing.discard(dup_track_id)
                        if dup_track_id in self.confirmed_passing_frames:
                            del self.confirmed_passing_frames[dup_track_id]
                        # Also drop from potential state to avoid later confirmations
                        if dup_track_id in self.potential_passing:
                            self.potential_passing.discard(dup_track_id)
                        if dup_track_id in self.potential_passing_tracks:
                            del self.potential_passing_tracks[dup_track_id]
                        logger.info(f"Removed duplicate track {dup_track_id} (same-frame overlap)")
                    # Clear for next frame
                    self.confirmed_this_frame = {}

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
                
                # Save output frame in background thread (non-blocking)
                file_ext = 'png' if self.output_format == 'png' else 'jpg'
                output_path = os.path.join(self.output_image_folder, f'output_{current_frame}.{file_ext}')
                # Add to queue for background saving
                self.save_queue.put((output_path, annotated_frame))
                
                # Display window (optional)
                if self.display:
                    cv2.imshow("Vehicle Pass Tracker", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return
                
                # Display progress in log
                if current_frame % 100 == 0:
                    print(f"Processing frame: {current_frame}", flush=True)
                
                # Periodic log flush to ensure logs are written
                if current_frame % 10 == 0:
                    for handler in logging.getLogger().handlers:
                        handler.flush()
                
                # Periodic CSV save to prevent data loss (every 1000 frames)
                # Use post_process=False to avoid destructive merging during active tracking
                if current_frame > 0 and current_frame % 1000 == 0:
                    if self.passing_data['track_id']:  # Only if we have data
                        logger.info(f"Saving intermediate results at frame {current_frame}...")
                        self.save_to_csv(post_process=False)
                
                current_frame += 1
        finally:
            # Wait for all frames to be saved
            print("Waiting for all frames to be saved...")
            self.save_queue.join()  # Wait for queue to be empty
            
            # Stop the background thread
            self.save_thread_running = False
            self.save_queue.put(None)  # Poison pill
            self.save_thread.join(timeout=5)  # Wait up to 5 seconds
            
            # Ensure all logs are written
            for handler in logging.getLogger().handlers:
                handler.flush()

        # Final cleanup
        if self.display:
            cv2.destroyAllWindows()

        # Save tracking data to CSV
        self.save_to_csv()

    def _log_frame_details(self, frame_number, detections, last_logged_frame):
        """Log detailed information about the current frame"""
        if len(detections) > 0 or self.potential_passing or self.confirmed_passing:
            log_entry = []
            
            # If there's a gap since last logged frame, add summary
            if last_logged_frame >= 0 and frame_number > last_logged_frame + 1:
                gap_start = last_logged_frame + 1
                gap_end = frame_number - 1
                if gap_start == gap_end:
                    log_entry.append(f"\n[Frame {gap_start}: No detections]")
                else:
                    log_entry.append(f"\n[Frames {gap_start}-{gap_end}: No detections]")
            
            log_entry.append(f"\nFrame {frame_number}:")
            
            # Log YOLO detections
            if len(detections) > 0:
                log_entry.append("  Detections:")
                for i in range(len(detections)):
                    class_id = detections.class_id[i]
                    if class_id in self.valid_classes:
                        track_id = detections.tracker_id[i]
                        class_name = get_class_name(class_id)
                        conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
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
        
        # Cleanup inactive tracks in one pass
        self._cleanup_tracks(active_ids_this_frame, current_frame)

    def _cleanup_tracks(self, active_ids_this_frame, current_frame):
        """Clean up inactive tracks and record completed passing events"""
        tracks_to_remove = []
        for track_id in list(self.potential_passing_tracks.keys()):
            track_data = self.potential_passing_tracks[track_id]
            # Remove tracks that have been inactive for cleanup_frames
            if track_id not in active_ids_this_frame:
                if current_frame - track_data['last_seen'] > self.cleanup_frames:
                    # Use was_confirmed flag instead of checking confirmed_passing set
                    # This ensures we record events even if removed from confirmed_passing
                    if track_data.get('was_confirmed', False):
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
        
        # Filter out events that are too short (likely noise or false positives)
        if last_frame - first_frame <= self.min_event_duration_frames:
            return
            
        # Get the passing frame from track data (stored when first confirmed)
        passing_frame = track_data.get('passing_frame', -1)
        pass_id = track_data['passing_id']
        
        # Prevent duplicate recording of the same pass_id
        if pass_id in self.completed_pass_ids:
            logger.warning(f"Pass ID {pass_id} (track {track_id}) already recorded, skipping duplicate")
            return
            
        # Create event with stored measurements
        event = VehiclePassEvent(
            pass_id=pass_id,
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
            
        # Mark this pass_id as completed
        self.completed_pass_ids.add(pass_id)
            
        logger.info(f"Recorded confirmed passing event for track {track_id} with pass_id {pass_id}")

    def _remove_track(self, track_id):
        """Remove a track and its associated data"""
        if track_id in self.potential_passing_tracks:
            del self.potential_passing_tracks[track_id]
        if track_id in self.confirmed_passing_frames:
            del self.confirmed_passing_frames[track_id]
        if track_id in self.angle_history:
            del self.angle_history[track_id]
        if track_id in self.bbox_area_history:
            del self.bbox_area_history[track_id]
        if track_id in self.probation_tracks:
            del self.probation_tracks[track_id]
        if track_id in self.current_angles:
            del self.current_angles[track_id]
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
    
    def _compute_angle_trend(self, track_id: int) -> str:
        """
        Compute angle trend based on angle history (camera-position invariant).
        
        Uses linear regression to handle detection noise better.
        
        Returns:
            'increasing': Angle increasing over time (overtaking: left->right)
            'decreasing': Angle decreasing over time (oncoming: right->left)
            'stable': No significant trend
        """
        if track_id not in self.angle_history:
            return 'stable'
        
        angles = self.angle_history[track_id]
        
        # Need at least 3 angles to determine trend
        if len(angles) < 3:
            return 'stable'
        
        # Simple linear regression: compute average slope
        # Using indices as x-values (time progression)
        n = len(angles)
        x_values = list(range(n))
        
        # Calculate means
        mean_x = sum(x_values) / n
        mean_y = sum(angles) / n
        
        # Calculate slope: m = Σ((x-mean_x)(y-mean_y)) / Σ((x-mean_x)²)
        numerator = sum((x_values[i] - mean_x) * (angles[i] - mean_y) for i in range(n))
        denominator = sum((x_values[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        # Threshold for significant trend (degrees per frame)
        # Uses config parameter for detecting increasing/decreasing trends
        trend_threshold = self.config.angle_trend_threshold
        
        if slope > trend_threshold:
            return 'increasing'  # Overtaking direction
        elif slope < -trend_threshold:
            return 'decreasing'  # Oncoming direction
        else:
            return 'stable'  # No clear trend

    
    def post_process_detections(self, merge_events=True):
        """Post-process detected events to remove invalid detections
        
        Args:
            merge_events: If True, merge overlapping events. Set to False for periodic saves
                         to avoid data loss during active tracking.
        """
        if not self.passing_data['track_id']:  # If no detections, return early
            return

        # Preserve all original pass_id values before any processing
        # This prevents duplicate recording if merged events lose some pass_id values
        original_pass_ids = set(self.passing_data['pass_id'])
        self.completed_pass_ids.update(original_pass_ids)

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(self.passing_data)
        
        # Remove detections in excluded ranges
        if self.excluded_ranges is not None:
            valid_detections = []
            for idx, row in df.iterrows():
                if not self._is_frame_range_excluded(row['first_frame'], row['last_frame']):
                    valid_detections.append(idx)
            df = df.loc[valid_detections]

        # Merge overlapping events only if merge_events=True (skip during periodic saves)
        if merge_events:
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
        else:
            # For periodic saves: just add buffer frames without merging
            detections = df.to_dict('records')
            for det in detections:
                det['first_frame'] = max(0, det['first_frame'] - self.buffer_frames)
                det['last_frame'] = min(det['last_frame'] + self.buffer_frames, self.image_sequence_length - 1)
            df = pd.DataFrame(detections)

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
        
        # Update pass_count to maximum pass_id to prevent reuse
        if len(df) > 0:
            self.pass_count = max(df['pass_id'].tolist())
        else:
            self.pass_count = 0

    def save_to_csv(self, post_process=True):
        """Save tracking results to CSV
        
        Args:
            post_process: If True, apply full post-processing (merging, filtering) before saving.
                         If False, save raw data without modification for periodic intermediate saves.
        """
        # For periodic saves, save raw data without modifying passing_data
        # This prevents data loss - events are preserved in memory for final save
        if post_process:
            # Apply full post-processing (merging, filtering, etc.) for final save
            # This modifies passing_data in place, which is OK for final save
            self.post_process_detections(merge_events=True)
            data_to_save = self.passing_data
        else:
            # For periodic saves, work on a COPY to avoid modifying passing_data
            # This ensures all events remain in memory for final save
            import copy
            data_copy = {
                'pass_id': self.passing_data['pass_id'].copy(),
                'track_id': self.passing_data['track_id'].copy(),
                'first_frame': self.passing_data['first_frame'].copy(),
                'last_frame': self.passing_data['last_frame'].copy(),
                'passing_frame': self.passing_data['passing_frame'].copy(),
                'vehicle_class': self.passing_data['vehicle_class'].copy(),
                'ref_point_distance': self.passing_data['ref_point_distance'].copy(),
                'passing_angle': self.passing_data['passing_angle'].copy(),
                'bbox_x1': self.passing_data['bbox_x1'].copy(),
                'bbox_y1': self.passing_data['bbox_y1'].copy(),
                'bbox_x2': self.passing_data['bbox_x2'].copy(),
                'bbox_y2': self.passing_data['bbox_y2'].copy()
            }
            # Update completed_pass_ids from current data (for duplicate prevention)
            if len(data_copy['pass_id']) > 0:
                current_pass_ids = set(data_copy['pass_id'])
                self.completed_pass_ids.update(current_pass_ids)
            data_to_save = data_copy

        # Convert to DataFrame and sort by last_frame
        df = pd.DataFrame(data_to_save)
        df = df.sort_values('last_frame')
        
        csv_path = os.path.join(self.output_folder, 'vehicle_passing.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path} ({len(df)} events)")
        print(f"Total confirmed vehicle passing events: {len(df)}")