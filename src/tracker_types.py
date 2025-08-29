"""
Type definitions and dataclasses for the vehicle pass tracker.
"""
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import numpy as np

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
 """Configuration parameters for vehicle tracking."""
 min_passing_frames_threshold: int = 1
 buffer_frames: int = 5
 tolerance_threshold: int = 10
 confidence_threshold: float = 0.4
 cleanup_frames: int = 30
 min_angle_change: float = 2.0
 valid_vehicle_classes: List[int] = None # car, motorcycle, bus, truck

 def __post_init__(self):
 if self.valid_vehicle_classes is None:
 self.valid_vehicle_classes = [2, 3, 5, 7]
