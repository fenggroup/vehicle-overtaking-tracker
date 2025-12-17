"""
Type definitions and dataclasses for the vehicle pass tracker.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class VehiclePassEvent:
 """Data class for storing vehicle passing event information"""
 pass_id: int
 track_id: int
 first_frame: int
 last_frame: int
 vehicle_class: int
 passing_frame: int
 ref_point_distance: float
 passing_angle: float
 bbox_x1: float
 bbox_y1: float
 bbox_x2: float
 bbox_y2: float
 
@dataclass
class TrackerConfig:
    """Configuration parameters for vehicle tracking and overtaking detection."""
    
    # Detection parameters
    confidence_threshold: float = 0.4  # Minimum confidence for RT-DETR detections
    valid_vehicle_classes: List[int] = None  # COCO classes: car=2, motorcycle=3, bus=5, truck=7
    
    # Angle-based filtering
    min_angle: float = 0.0  # Minimum angle (degrees) for valid detections
    max_angle: float = 90.0  # Maximum angle (degrees) for valid detections
    min_angle_slope: float = 0.5  # Minimum angle increase rate (deg/frame) for overtaking
    min_angle_increase_count: int = 5  # Minimum number of frames with angle increase (out of history window)
    angle_trend_threshold: float = 0.2  # Threshold for detecting increasing/decreasing angle trends (deg/frame)
    
    # Bounding box growth filtering
    min_bbox_growth_rate: float = 60.0  # Minimum bbox area growth rate (px²/frame) for approaching vehicles
    
    # History and evaluation windows
    history_window_size: int = 6  # Number of frames to retain in angle/position history
    min_frames_for_evaluation: int = 3  # Minimum frames required before evaluating overtaking criteria
    probation_frames: int = 2  # Number of consecutive frames required to pass all checks before marking as potential passing
    
    # Confirmation line positions (fraction of frame width)
    confirmation_line_bottom_center: float = 0.85  # Confirmation line for bottom_center camera position
    confirmation_line_bottom_left: float = 0.65  # Confirmation line for bottom_left camera position
    
    # Event management
    cleanup_frames: int = 30  # Number of frames of inactivity before removing a track
    buffer_frames: int = 5  # Buffer frames added to event boundaries during post-processing
    tolerance_threshold: int = 10  # Tolerance for merging overlapping events (frames)
    min_event_duration_frames: int = 1  # Minimum event duration (frames) to record in CSV (filters out very short events/noise)
    
    # Legacy parameters (kept for backwards compatibility, not used in simplified algorithm)
    min_angle_change: float = 2.0  # Deprecated: use min_angle_slope instead

    def __post_init__(self):
        if self.valid_vehicle_classes is None:
            self.valid_vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
