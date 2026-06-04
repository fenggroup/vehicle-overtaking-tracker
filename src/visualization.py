"""src.visualization

Visualization utilities for vehicle tracking and passing event detection.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import get_class_name


_DEMO_DISTANCES_LOADED: bool = False
_DEMO_DISTANCE_BY_PASS_ID: Dict[int, float] = {}


def _try_load_demo_distances() -> None:
    """Best-effort load of data/demo_distances.csv."""
    global _DEMO_DISTANCES_LOADED, _DEMO_DISTANCE_BY_PASS_ID
    if _DEMO_DISTANCES_LOADED:
        return

    _DEMO_DISTANCES_LOADED = True

    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "demo_distances.csv"
    if not csv_path.exists():
        return

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            distance_map: Dict[int, float] = {}
            for raw in reader:
                if not raw:
                    continue
                try:
                    pass_id = int(str(raw.get("pass_id", "")).strip())
                    pred_distance_cm = float(str(raw.get("pred_distance_cm", "")).strip())
                    distance_map[pass_id] = pred_distance_cm
                except Exception:
                    continue
            _DEMO_DISTANCE_BY_PASS_ID = distance_map
    except Exception:
        _DEMO_DISTANCE_BY_PASS_ID = {}


def _format_distance_meters(distance_cm: float) -> str:
    distance_m = distance_cm / 100.0
    return f"{distance_m:.2f}".rstrip("0").rstrip(".")


def _get_distance_for_pass_id(pass_id: int) -> Optional[float]:
    _try_load_demo_distances()
    return _DEMO_DISTANCE_BY_PASS_ID.get(pass_id)


def _build_label_parts(
    passing_id,
    track_id: int,
    confirmed_passing: set,
    show_distance: bool,
    custom_space: int = 3,
) -> List[Tuple[str, int]]:
    """Return label parts as (text, spacing_after) tuples for rendering."""
    if not show_distance or track_id not in confirmed_passing or passing_id == '?':
        return [("ID:", custom_space), (str(passing_id), 0)]

    try:
        distance_cm = _get_distance_for_pass_id(int(passing_id))
    except (ValueError, TypeError):
        return [("ID:", custom_space), (str(passing_id), 0)]

    if distance_cm is None:
        return [("ID:", custom_space), (str(passing_id), 0)]

    if abs(distance_cm) < 1e-6:
        return [
            ("ID:", custom_space),
            (str(passing_id) + ";", custom_space),
            ("passing", custom_space),
            ("distance", custom_space),
            ("larger", custom_space),
            ("than", custom_space),
            ("3", custom_space),
            ("m", 0),
        ]

    dist_str = _format_distance_meters(distance_cm)
    return [
        ("ID:", custom_space),
        (str(passing_id) + ";", custom_space),
        ("passing", custom_space),
        ("distance:", custom_space),
        (dist_str, custom_space),
        ("m", 0),
    ]


def draw_visualizations(
    frame,
    detections,
    current_frame,
    confirmed_passing,
    potential_passing,
    current_angles,
    mode,
    image_source_position,
    active_tracks,
    show_distance: bool = False,
):
    """Draw detection visualizations on frame.

    Args:
        show_distance: When True, confirmed tracks display passing distance from
                       data/demo_distances.csv. Intended for demo/presentation runs only.
    """
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

                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                box_color = (0, 0, 255) if track_id in confirmed_passing else (0, 255, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3, lineType=cv2.LINE_AA)

                track_data = active_tracks.get(track_id, {})
                passing_id = track_data.get('passing_id', '?')

                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.8
                thickness = 1
                h_padding = 6
                v_padding = 6

                label_parts = _build_label_parts(
                    passing_id, track_id, confirmed_passing, show_distance
                )

                part_widths = []
                spacings = []
                max_height = 0
                max_baseline = 0
                for part_text, spacing in label_parts:
                    (pw, ph), pbaseline = cv2.getTextSize(part_text, font, font_scale, thickness)
                    part_widths.append(pw)
                    spacings.append(spacing)
                    max_height = max(max_height, ph)
                    max_baseline = max(max_baseline, pbaseline)

                total_width = sum(part_widths) + sum(spacings)

                label_y = max(max_height + max_baseline + v_padding + 4, y1 - 4)
                bg_x1 = x1
                bg_y1 = max(0, label_y - max_height - max_baseline - v_padding)
                bg_x2 = min(frame_width, x1 + total_width + 2 * h_padding)
                bg_y2 = label_y + v_padding

                cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1, lineType=cv2.LINE_AA)

                current_x = bg_x1 + h_padding
                text_y = bg_y1 + v_padding + max_height
                for idx, (part_text, spacing) in enumerate(label_parts):
                    cv2.putText(
                        annotated, part_text, (current_x, text_y),
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
                    )
                    current_x += part_widths[idx] + spacing

                if mode == 'debug':
                    if image_source_position == 'bottom_center':
                        ref_point = (frame_width // 2, frame_height - 5)
                    elif image_source_position == 'bottom_left':
                        ref_point = (0, frame_height - 5)

                    cv2.line(annotated, ref_point, (center_x, center_y), (0, 255, 0), 1, lineType=cv2.LINE_AA)
                    current_angle = current_angles.get(track_id, 0)
                    angle_text = f"{current_angle:.1f}°"

                    (aw, ah), abaseline = cv2.getTextSize(angle_text, font, font_scale, thickness)
                    angle_bg_y1 = max(0, y2 + 4)
                    angle_bg_y2 = angle_bg_y1 + ah + abaseline + 4
                    angle_bg_x2 = min(frame_width, x1 + aw + 4)
                    cv2.rectangle(annotated, (x1, angle_bg_y1), (angle_bg_x2, angle_bg_y2), (0, 0, 255), -1, lineType=cv2.LINE_AA)
                    cv2.putText(annotated, angle_text, (x1 + 2, angle_bg_y2 - abaseline - 2),
                                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated
