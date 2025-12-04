# Algorithm Reference

## Overview

Detects vehicles overtaking a bicycle from behind using RT-DETR object detection, ByteTrack multi-object tracking, and geometric validation based on vehicle bearing angle and bounding box growth.

## Core Principle

**Overtaking vehicles** (approaching from behind):
- Angle increases consistently (left to right in frame)
- Bounding box grows as vehicle approaches

**Oncoming/foreground vehicles**:
- Angle decreases (right to left) or increases briefly
- Bounding box remains stable or shrinks

## Algorithm Flow

```
Detection → Tracking → Angle Gate → History (7 frames) → Early Reject → Admission (3 checks) → Probation (2 frames) → Yellow Box → Confirmation → Red Box
```

## Step-by-Step

### 1. Detection & Tracking
- RT-DETR detects vehicles (confidence ≥ 0.4)
- ByteTrack assigns persistent IDs
- Filter: Only car, motorcycle, bus, truck (class IDs: 2, 3, 5, 7)

### 2. Angle Gate
- Calculate angle from reference point to vehicle centroid
- Reject if angle not in range: 0° ≤ angle ≤ 90°
- Only right half of frame considered

### 3. Build History
- Track last 10 frames of:
  - Angle (degrees)
  - Bounding box area (pixels²)
- Need ≥ 7 frames before evaluation

### 4. Early Rejection
- Calculate angle trend via linear regression
- If slope < -0.5 deg/frame → reject (oncoming traffic)
- If already in potential_passing → remove immediately

### 5. Admission Checks
All three conditions must pass:

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Angle slope | ≥ 0.7 deg/frame | Consistent angle increase (direction: left→right) |
| Angle increases | ≥ 5 in recent history (N up to 10; eval when N ≥ 7) | Majority show increase (handles jitter) |
| Bounding box growth | ≥ 100 px²/frame | Vehicle approaching (getting closer) |

### 6. Probation
- Must pass all 3 checks for 2 consecutive frames
- Fail once → removed from probation
- Pass 2 times → advance to potential_passing

### 7. Potential Passing (Yellow Box)
- Track shows yellow bounding box
- Continues until:
  - Crosses confirmation line → turns red
  - Trend reverses → removed
  - Inactive > 30 frames → removed

### 8. Confirmation (Red Box)
- Right edge of bounding box crosses confirmation line:
  - `bottom_left` camera: x2 ≥ 0.65 × width
  - `bottom_center` camera: x2 ≥ 0.85 × width
- Track shows red bounding box
- Passing frame recorded

### 9. Event Recording
- Track inactive > 30 frames → finalized
- Only confirmed (red) events recorded to CSV
- Post-processing: merge overlaps, add buffers

## Configuration Parameters

### Angle-Based
```python
angle_range = [0, 90]              # degrees (right half only)
trend_threshold = 0.5             # deg/frame (early reject if < -0.5)
min_angle_slope = 0.7             # deg/frame (admission)
min_increase_count = 5            # out of 7 frames
```

### Bounding Box Growth
```python
min_bbox_growth_rate = 100        # px²/frame (must be expanding)
```

### Frame Counts
```python
history_window = 10               # frames retained
min_frames_for_eval = 7           # frames before evaluation
probation_frames = 2              # consecutive passing frames
cleanup_frames = 30               # inactivity timeout
```

### Confirmation
```python
confirmation_line_bottom_left = 0.65 * width
confirmation_line_bottom_center = 0.85 * width
```

## Parameter Tuning

### Too Many False Positives
If foreground or stationary vehicles are incorrectly detected:
```python
min_bbox_growth_rate = 200        # from 100
min_angle_slope = 1.0             # from 0.7
probation_frames = 3               # from 2
```

### Missing Real Overtakes
If valid overtaking events are missed:
```python
min_bbox_growth_rate = 50          # from 100
min_angle_slope = 0.5              # from 0.7
min_increase_count = 4             # from 5
probation_frames = 1               # from 2
```

### Large Vehicles (Trucks/Buses)
The bounding box growth filter is size-independent. If large vehicles are missed, lower `min_bbox_growth_rate` to 50-80.

## Camera Setup

### Supported Positions
- `bottom_left`: Rear-left facing camera
- `bottom_center`: Rear-center facing camera

### Reference Point
- `bottom_left`: (0, frame_height - 5)
- `bottom_center`: (frame_width / 2, frame_height - 5)

## Output

### CSV Columns
- `pass_id`: Unique passing event ID
- `track_id`: ByteTrack ID
- `first_frame`: First frame of event
- `last_frame`: Last frame of event
- `passing_frame`: Frame when confirmed (crossed line)
- `vehicle_class`: Class ID (2=car, 3=motorcycle, 5=bus, 7=truck)
- `ref_point_distance`: Distance from reference point (pixels)
- `passing_angle`: Angle at passing frame (degrees)
- `bbox_x1, bbox_y1, bbox_x2, bbox_y2`: Bounding box at passing frame

### Log Files
- `tracking_debug.log`: Frame-by-frame tracking details
  - Format: `Frame X - Track Y: angle=..., area=..., darea=..., trend=...`

## Debugging

### Log Patterns

**Overtaking pattern**:
- `trend=increasing` or `stable`
- `area` increases steadily (e.g., 5000 → 8000 → 12000)
- `darea` positive and growing (e.g., +200 → +400 → +600)

**Foreground/stationary pattern**:
- `trend` may be `increasing` briefly
- `area` large and stable (e.g., 25000 → 24500 → 25200)
- `darea` near zero or negative (e.g., -50 → +100 → -80)

**Oncoming pattern**:
- `trend=decreasing`
- `area` shrinks as vehicle moves away

### Probation Tracking
```bash
grep "Entered probation" tracking_debug.log | wc -l
grep "Graduated from probation" tracking_debug.log | wc -l
grep "Removed from probation" tracking_debug.log | wc -l
```

Typical ratio: Removed/Graduated > 3:1 (most probation tracks are filtered out)
