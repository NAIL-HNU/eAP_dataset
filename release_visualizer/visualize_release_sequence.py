#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import argparse
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None  # noqa: F841


RED = (0, 0, 255)
ORANGE = (0, 165, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (240, 240, 240)
GRAY = (46, 46, 46)
LIGHT_GRAY = (110, 110, 110)
DARK = (18, 18, 18)
BEV_BG = (248, 248, 248)
BEV_INFO_BG = (242, 242, 242)
BEV_GRID = (224, 224, 224)
BEV_BORDER = (196, 196, 196)
BEV_TEXT = (34, 34, 34)
DEFAULT_CATEGORY_COLOR = (90, 90, 90)
EVENT_WINDOW_US = 50_000
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected #RRGGBB color, got: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


CATEGORY_COLORS = {
    "car": hex_to_bgr("#E69F00"),
    "truck": hex_to_bgr("#0072B2"),
    "bus": hex_to_bgr("#D55E00"),
    "other_vehicle": hex_to_bgr("#8C564B"),
    "pedestrian": hex_to_bgr("#CC79A7"),
    "motorcycle": hex_to_bgr("#009E73"),
    "tricycle": hex_to_bgr("#7B61FF"),
    "bicycle": hex_to_bgr("#66A61E"),
    "pickupTruck": hex_to_bgr("#4E79A7"),
    "suv": hex_to_bgr("#F28E2B"),
    "van": hex_to_bgr("#56B4E9"),
}
CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
FRAME_FIELDS = {
    "file_name",
    "events_file",
    "sequence_id",
    "rgb_exposure_start_timestamp_us",
    "rgb_exposure_end_timestamp_us",
}
ANNOTATION_FIELDS = {
    "file_name",
    "events_file",
    "sequence_id",
    "instance_id",
    "bbox",
    "bbox_3d",
    "T_event_ego",
    "K_event",
    "velocity",
    "ttc",
    "rgb_exposure_start_timestamp_us",
    "rgb_exposure_end_timestamp_us",
}


@dataclass
class ObjectAnnotation:
    instance_id: str
    category: str
    bbox: Tuple[int, int, int, int]
    bbox_3d: Tuple[float, float, float, float, float, float, float]
    velocity: Tuple[float, float, float]
    ttc: Optional[float]
    T_event_ego: np.ndarray
    K_event: np.ndarray


@dataclass
class FrameRecord:
    sequence_id: str
    file_name: str
    image_path: Path
    events_path: Path
    exposure_start_us: int
    exposure_end_us: int
    objects: List[ObjectAnnotation] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize eAP sequence-local release bundles.")
    parser.add_argument("release_path", type=Path, help="Release root or a single sequence directory.")
    parser.add_argument("--sequence-id", type=str, default=None, help="Sequence ID to visualize when passing the release root.")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Directory for rendered videos.")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS.")
    parser.add_argument("--max-frames", type=int, default=None, help="Only render the first N frames.")
    parser.add_argument("--frame-step", type=int, default=1, help="Render every Nth frame.")
    parser.add_argument("--image-width", type=int, default=960, help="Width of each left-column panel.")
    parser.add_argument("--bev-width", type=int, default=760, help="Width of the BEV column.")
    parser.add_argument("--fwd-max", type=float, default=60.0, help="BEV forward range upper bound in meters.")
    parser.add_argument("--lat-max", type=float, default=30.0, help="BEV half lateral range in meters.")
    parser.add_argument("--pixel-diff", type=int, default=0, help="Extra x-axis pixel shift for event overlay.")
    parser.add_argument(
        "--event-window-us",
        type=int,
        default=None,
        help="Deprecated. Ignored; event rendering always uses a centered 50ms window.",
    )
    parser.add_argument("--list-sequences", action="store_true", help="List available sequence IDs and exit.")
    return parser.parse_args()


def read_pkl(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def is_sequence_dir(path: Path) -> bool:
    return (path / "frames.pkl").exists() and (path / "annotations.pkl").exists()


def iter_sequence_dirs(root: Path) -> List[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir() and is_sequence_dir(path)])


def choose_sequence_dir(release_path: Path, requested: Optional[str]) -> Path:
    if is_sequence_dir(release_path):
        if requested is not None and release_path.name != requested:
            raise RuntimeError(
                f"Input path points to sequence '{release_path.name}', but requested '{requested}'"
            )
        return release_path
    if not release_path.exists():
        raise FileNotFoundError(f"Missing release path: {release_path}")
    sequence_dirs = iter_sequence_dirs(release_path)
    if not sequence_dirs:
        raise RuntimeError(f"No sequence bundles found in {release_path}")
    if requested is None:
        if len(sequence_dirs) == 1:
            return sequence_dirs[0]
        available = ", ".join(path.name for path in sequence_dirs[:20])
        raise RuntimeError(f"Release root contains multiple sequences; please pass --sequence-id. Available: {available}")
    for path in sequence_dirs:
        if path.name == requested:
            return path
    available = ", ".join(path.name for path in sequence_dirs[:20])
    raise RuntimeError(f"Sequence '{requested}' not found. Available: {available}")


def list_sequences(release_path: Path) -> List[str]:
    if is_sequence_dir(release_path):
        return [release_path.name]
    return [path.name for path in iter_sequence_dirs(release_path)]


def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def load_matrix(value, shape: Tuple[int, int], field_name: str, row_id: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != shape:
        raise ValueError(f"Invalid {field_name} shape for row '{row_id}': expected {shape}, got {matrix.shape}")
    return matrix


def build_frame_record(sequence_dir: Path, row: Dict, row_id: str) -> FrameRecord:
    missing = FRAME_FIELDS - set(row)
    if missing:
        raise ValueError(f"Missing required frame fields {sorted(missing)} for row '{row_id}'")
    image_path = sequence_dir / row["file_name"]
    events_path = sequence_dir / row["events_file"]
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    return FrameRecord(
        sequence_id=str(row["sequence_id"]),
        file_name=str(row["file_name"]),
        image_path=image_path,
        events_path=events_path,
        exposure_start_us=int(row["rgb_exposure_start_timestamp_us"]),
        exposure_end_us=int(row["rgb_exposure_end_timestamp_us"]),
    )


def load_sequence_frames(sequence_dir: Path) -> List[FrameRecord]:
    frames_path = sequence_dir / "frames.pkl"
    annotations_path = sequence_dir / "annotations.pkl"
    frames_raw = read_pkl(frames_path)
    annotations_raw = read_pkl(annotations_path)
    if not isinstance(frames_raw, list):
        raise ValueError(f"frames.pkl must be a list: {frames_path}")
    if not isinstance(annotations_raw, list):
        raise ValueError(f"annotations.pkl must be a list: {annotations_path}")

    frames: Dict[str, FrameRecord] = {}
    for idx, row in enumerate(frames_raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid frame row type at {frames_path}:{idx}")
        row_id = str(row.get("file_name", f"frame_{idx}"))
        frame = build_frame_record(sequence_dir, row, row_id)
        if frame.file_name in frames:
            raise ValueError(f"Duplicate frame row for '{frame.file_name}' in {frames_path}")
        frames[frame.file_name] = frame

    for idx, row in enumerate(annotations_raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid annotation row type at {annotations_path}:{idx}")
        missing = ANNOTATION_FIELDS - set(row)
        if missing:
            raise ValueError(f"Missing required annotation fields {sorted(missing)} at {annotations_path}:{idx}")
        frame = frames.get(str(row["file_name"]))
        if frame is None:
            raise ValueError(f"Object row references frame missing from frames.pkl: {row['file_name']}")
        row_id = str(row.get("instance_id", f"object_{idx}"))
        frame.objects.append(
            ObjectAnnotation(
                instance_id=str(row["instance_id"]),
                category=str(row["category"]),
                bbox=tuple(int(v) for v in row["bbox"]),
                bbox_3d=tuple(float(v) for v in row["bbox_3d"]),
                velocity=tuple(float(v) for v in row["velocity"]),
                ttc=safe_float(row["ttc"]),
                T_event_ego=load_matrix(row["T_event_ego"], (4, 4), "T_event_ego", row_id),
                K_event=load_matrix(row["K_event"], (3, 3), "K_event", row_id),
            )
        )

    ordered_frames = sorted(
        frames.values(),
        key=lambda item: (item.exposure_start_us, item.file_name),
    )
    if not ordered_frames:
        raise RuntimeError(f"No frames found in {sequence_dir}")
    return ordered_frames


class EventStreamReader:
    def __init__(self, events_path: Path):
        self.events_path = events_path
        self.handle = h5py.File(str(events_path), "r")
        self.ms_to_idx = np.asarray(self.handle["ms_to_idx"], dtype=np.int64)
        self.events = self.handle["events"]
        self.last_timestamp_us = int(self.events["t"][-1])

    def close(self) -> None:
        self.handle.close()

    def extract(self, start_us: int, end_us: int, roi: Tuple[int, int], pixel_diff: int = 0) -> Dict[str, np.ndarray]:
        start_us = max(0, int(start_us))
        end_us = max(start_us, int(end_us))
        end_us = min(end_us, self.last_timestamp_us)

        if end_us <= start_us:
            return {
                "x": np.empty((0,), dtype=np.int16),
                "y": np.empty((0,), dtype=np.int16),
                "p": np.empty((0,), dtype=np.int8),
                "t": np.empty((0,), dtype=np.int64),
            }

        start_ms = min(start_us // 1000, len(self.ms_to_idx) - 1)
        end_ms = min(end_us // 1000, len(self.ms_to_idx) - 1)
        start_idx = int(self.ms_to_idx[start_ms])
        if end_ms + 1 < len(self.ms_to_idx):
            end_idx = int(self.ms_to_idx[end_ms + 1])
        else:
            end_idx = int(self.events["t"].shape[0])

        x = self.events["x"][start_idx:end_idx].astype(np.int32) + int(pixel_diff)
        y = self.events["y"][start_idx:end_idx].astype(np.int32)
        p = self.events["p"][start_idx:end_idx].astype(np.int8)
        t = self.events["t"][start_idx:end_idx].astype(np.int64)
        valid_t = (t >= start_us) & (t <= end_us)
        x = x[valid_t]
        y = y[valid_t]
        p = p[valid_t]
        t = t[valid_t]

        height, width = roi
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        return {
            "x": x[valid].astype(np.int16),
            "y": y[valid].astype(np.int16),
            "p": p[valid],
            "t": t[valid],
        }


def resize_keep_width(image: np.ndarray, target_width: int) -> np.ndarray:
    if image.shape[1] == target_width:
        return image
    scale = target_width / float(image.shape[1])
    target_height = max(1, int(round(image.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interp)


def pad_to_size(image: np.ndarray, target_h: int, target_w: int, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w = image.shape[:2]
    pad_top = max(0, (target_h - h) // 2)
    pad_bottom = max(0, target_h - h - pad_top)
    pad_left = max(0, (target_w - w) // 2)
    pad_right = max(0, target_w - w - pad_left)
    return cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)


def add_title_bar(image: np.ndarray, title: str, bar_h: int = 34) -> np.ndarray:
    bar = np.full((bar_h, image.shape[1], 3), 246, dtype=np.uint8)
    cv2.putText(bar, title, (12, 23), TEXT_FONT, 0.65, BEV_TEXT, 2, cv2.LINE_AA)
    cv2.line(bar, (0, bar_h - 1), (image.shape[1], bar_h - 1), BEV_BORDER, 1, cv2.LINE_AA)
    return np.vstack((bar, image))


def stack_vertical(images: List[np.ndarray], gap: int = 10, bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    max_w = max(img.shape[1] for img in images)
    padded = [pad_to_size(img, img.shape[0], max_w, bg_color) for img in images]
    if gap <= 0 or len(padded) == 1:
        return np.vstack(padded)
    spacer = np.full((gap, max_w, 3), bg_color, dtype=np.uint8)
    parts: List[np.ndarray] = []
    for idx, img in enumerate(padded):
        if idx > 0:
            parts.append(spacer)
        parts.append(img)
    return np.vstack(parts)


def stack_horizontal(images: List[np.ndarray], gap: int = 12, bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    max_h = max(img.shape[0] for img in images)
    padded = [pad_to_size(img, max_h, img.shape[1], bg_color) for img in images]
    if gap <= 0 or len(padded) == 1:
        return np.hstack(padded)
    spacer = np.full((max_h, gap, 3), bg_color, dtype=np.uint8)
    parts: List[np.ndarray] = []
    for idx, img in enumerate(padded):
        if idx > 0:
            parts.append(spacer)
        parts.append(img)
    return np.hstack(parts)


def short_instance_id(instance_id: str) -> str:
    if "_" in instance_id:
        return instance_id.split("_")[-1]
    return instance_id


def classify_ttc(ttc: Optional[float]) -> Tuple[str, Tuple[int, int, int]]:
    if ttc is None:
        return "large", GREEN
    ttc = float(ttc)
    if -10.0 <= ttc < 0.0:
        return "negative", BLUE
    if ttc < -10.0:
        return "negative", BLUE
    if 0.0 <= ttc < 3.0:
        return "crucial", RED
    if 3.0 <= ttc < 6.0:
        return "small", ORANGE
    return "large", GREEN


def color_for_ttc(ttc: Optional[float]) -> Tuple[int, int, int]:
    return classify_ttc(ttc)[1]


def color_for_category(category: str) -> Tuple[int, int, int]:
    return CATEGORY_COLORS.get(category, DEFAULT_CATEGORY_COLOR)


def format_ttc(ttc: Optional[float]) -> str:
    if ttc is None:
        return "inf"
    return f"{ttc:.1f}s"


def build_object_label(obj: ObjectAnnotation) -> str:
    return f"{obj.category}#{short_instance_id(obj.instance_id)} TTC:{format_ttc(obj.ttc)}"


def draw_text_label(
    canvas: np.ndarray,
    text: str,
    anchor: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    (text_w, text_h), baseline = cv2.getTextSize(text, TEXT_FONT, font_scale, thickness)
    max_x = max(4, canvas.shape[1] - text_w - 8)
    max_y = max(text_h + 6, canvas.shape[0] - baseline - 6)
    x = max(4, min(int(anchor[0]), max_x))
    y = max(text_h + 6, min(int(anchor[1]), max_y))
    top_left = (x - 4, y - text_h - 6)
    bottom_right = (x + text_w + 4, y + baseline + 4)
    cv2.rectangle(canvas, top_left, bottom_right, DARK, -1)
    cv2.rectangle(canvas, top_left, bottom_right, color, 1)
    cv2.putText(canvas, text, (x, y), TEXT_FONT, font_scale, color, thickness, cv2.LINE_AA)


def cuboid_label_anchor(pts_2d: Optional[np.ndarray], fallback_bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    if pts_2d is None or pts_2d.size == 0:
        x1, y1, _, _ = fallback_bbox
        return int(x1), int(y1) - 8
    min_x = int(np.min(pts_2d[:, 0]))
    min_y = int(np.min(pts_2d[:, 1]))
    return min_x, min_y - 8


def render_events_on_canvas(shape: Tuple[int, int, int], events: Dict[str, np.ndarray]) -> np.ndarray:
    canvas = np.full(shape, 255, dtype=np.uint8)
    if events["x"].size == 0:
        return canvas
    neg_mask = events["p"] == 0
    pos_mask = ~neg_mask
    canvas[events["y"][neg_mask], events["x"][neg_mask]] = np.array([0, 0, 255], dtype=np.uint8)
    canvas[events["y"][pos_mask], events["x"][pos_mask]] = np.array([255, 0, 0], dtype=np.uint8)
    return canvas


def bbox_to_corner_ego(x: float, y: float, z: float, l: float, h: float, w: float, yaw: float) -> np.ndarray:
    cx = w * np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.float32) / 2.0
    cy = l * np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.float32) / 2.0
    cz = h * np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float32) / 2.0
    cart = np.stack([cx, cy, cz, np.ones_like(cx)], axis=1)
    rot = np.array(
        [
            [math.cos(yaw), math.sin(yaw), 0.0, 0.0],
            [-math.sin(yaw), math.cos(yaw), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    corners = (rot @ cart.T).T
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z
    return corners[:, :3]


def project_cuboid_to_image(obj: ObjectAnnotation) -> Optional[np.ndarray]:
    corners_ego = bbox_to_corner_ego(*obj.bbox_3d)
    corners_h = np.concatenate(
        [corners_ego.astype(np.float32), np.ones((corners_ego.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    corners_cam = (obj.T_event_ego @ corners_h.T).T[:, :3]
    if np.count_nonzero(corners_cam[:, 2] > 1e-4) < 8:
        return None
    projected = (obj.K_event @ corners_cam.T).T
    pts_2d = projected[:, :2] / projected[:, 2:3]
    return np.round(pts_2d).astype(np.int32)


def draw_projected_3d_box(canvas: np.ndarray, obj: ObjectAnnotation, color: Tuple[int, int, int]) -> bool:
    pts_2d = project_cuboid_to_image(obj)
    if pts_2d is None:
        return False
    for start_idx, end_idx in CUBOID_EDGES:
        pt1 = tuple(int(v) for v in pts_2d[start_idx])
        pt2 = tuple(int(v) for v in pts_2d[end_idx])
        cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
    return True


def draw_front_annotations(image: np.ndarray, objects: List[ObjectAnnotation]) -> np.ndarray:
    canvas = image.copy()
    for obj in objects:
        geometry_color = color_for_category(obj.category)
        label_color = color_for_ttc(obj.ttc)
        pts_2d = project_cuboid_to_image(obj)
        if pts_2d is not None:
            for start_idx, end_idx in CUBOID_EDGES:
                pt1 = tuple(int(v) for v in pts_2d[start_idx])
                pt2 = tuple(int(v) for v in pts_2d[end_idx])
                cv2.line(canvas, pt1, pt2, geometry_color, 2, cv2.LINE_AA)
        draw_text_label(
            canvas,
            build_object_label(obj),
            cuboid_label_anchor(pts_2d, obj.bbox),
            label_color,
            font_scale=0.42,
            thickness=1,
        )
    return canvas


def draw_velocity_arrow(
    canvas: np.ndarray,
    to_px,
    start_fwd: float,
    start_lat: float,
    velocity_xy: Tuple[float, float],
    color: Tuple[int, int, int],
    lookahead_s: float = 0.6,
    min_speed_mps: float = 0.5,
    max_len_m: float = 8.0,
    thickness: int = 2,
) -> None:
    delta = np.array(velocity_xy, dtype=np.float32).reshape(2) * float(lookahead_s)
    speed = float(np.linalg.norm(delta))
    if not np.isfinite(speed) or speed < min_speed_mps * lookahead_s:
        return
    if speed > max_len_m and speed > 1e-6:
        delta *= max_len_m / speed
    start = to_px(start_fwd, start_lat)
    end = to_px(start_fwd + float(delta[0]), start_lat + float(delta[1]))
    if abs(end[0] - start[0]) + abs(end[1] - start[1]) <= 4:
        return
    start_vec = np.array(start, dtype=np.float32)
    end_vec = np.array(end, dtype=np.float32)
    direction = end_vec - start_vec
    length = float(np.linalg.norm(direction))
    if length <= 1e-3:
        return
    direction /= length
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)
    head_len = min(max(10.0, length * 0.26), 18.0)
    head_half_w = min(max(4.0, length * 0.10), 8.0)
    shaft_end = end_vec - direction * head_len
    cv2.line(
        canvas,
        tuple(np.round(start_vec).astype(int)),
        tuple(np.round(shaft_end).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.circle(canvas, tuple(np.round(start_vec).astype(int)), max(2, thickness + 1), color, -1, cv2.LINE_AA)
    tip = np.round(end_vec).astype(np.int32)
    left = np.round(shaft_end + perp * head_half_w).astype(np.int32)
    right = np.round(shaft_end - perp * head_half_w).astype(np.int32)
    cv2.fillConvexPoly(canvas, np.array([tip, left, right], dtype=np.int32), color, cv2.LINE_AA)


def draw_bev(
    objects: List[ObjectAnnotation],
    canvas_w: int,
    canvas_h: int,
    fwd_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    frame: FrameRecord,
    frame_index: int,
    total_frames: int,
    event_count: int,
) -> np.ndarray:
    canvas = np.full((canvas_h, canvas_w, 3), BEV_BG, dtype=np.uint8)
    info_band_h = 118
    margin = 18
    fwd_span = float(fwd_range[1] - fwd_range[0])
    lat_span = float(lat_range[1] - lat_range[0])
    usable_h = max(1, canvas_h - info_band_h - margin * 2)
    usable_w = max(1, canvas_w - margin * 2)
    px_per_m = min(usable_h / fwd_span, usable_w / lat_span)
    draw_h = max(1, int(round(fwd_span * px_per_m)))
    draw_w = max(1, int(round(lat_span * px_per_m)))
    origin_x = (canvas_w - draw_w) // 2
    origin_y = max(info_band_h + 10, canvas_h - margin - draw_h)

    def to_px(forward_m: float, lateral_m: float) -> Tuple[int, int]:
        lateral_screen = -float(lateral_m)
        col = origin_x + int(round((lateral_screen - lat_range[0]) * px_per_m))
        row = origin_y + int(round((fwd_range[1] - forward_m) * px_per_m))
        return col, row

    cv2.rectangle(canvas, (origin_x, origin_y), (origin_x + draw_w, origin_y + draw_h), BEV_BORDER, 1)

    for d in range(int(fwd_range[0]), int(fwd_range[1]) + 1, 10):
        _, row = to_px(float(d), 0.0)
        cv2.line(canvas, (origin_x, row), (origin_x + draw_w, row), BEV_GRID, 1)
        cv2.putText(canvas, f"{d}m", (8, max(16, row - 4)), TEXT_FONT, 0.42, LIGHT_GRAY, 1, cv2.LINE_AA)

    for lat in range(int(lat_range[0]), int(lat_range[1]) + 1, 10):
        col, _ = to_px(0.0, float(lat))
        cv2.line(canvas, (col, origin_y), (col, origin_y + draw_h), BEV_GRID, 1)
        if lat != 0:
            cv2.putText(canvas, f"{lat}m", (col + 4, 20), TEXT_FONT, 0.42, LIGHT_GRAY, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (canvas_w, info_band_h), BEV_INFO_BG, -1)
    cv2.line(canvas, (0, info_band_h - 1), (canvas_w, info_band_h - 1), BEV_BORDER, 1, cv2.LINE_AA)

    ego_box = np.array(
        [
            to_px(0.0, -0.9),
            to_px(4.5, -0.9),
            to_px(4.5, 0.9),
            to_px(0.0, 0.9),
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [ego_box], (56, 56, 56))
    cv2.polylines(canvas, [ego_box], True, WHITE, 2)
    draw_velocity_arrow(canvas, to_px, 2.25, 0.0, (8.0, 0.0), (56, 56, 56), lookahead_s=0.75, max_len_m=6.0, thickness=2)

    for obj in objects:
        x, y, z, l, h, w, yaw = obj.bbox_3d
        geometry_color = color_for_category(obj.category)
        label_color = color_for_ttc(obj.ttc)
        corners = bbox_to_corner_ego(x, y, z, l, h, w, yaw)
        footprint = corners[:4, :2]
        pts = np.array([to_px(float(p[0]), float(p[1])) for p in footprint], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, geometry_color, 2)

        center = to_px(x, y)
        cv2.circle(canvas, center, 4, geometry_color, -1)

        vx, vy = float(obj.velocity[0]), float(obj.velocity[1])
        draw_velocity_arrow(canvas, to_px, x, y, (vx, vy), geometry_color, lookahead_s=0.6, max_len_m=8.0, thickness=2)
        draw_text_label(canvas, build_object_label(obj), (center[0] + 8, center[1] - 8), label_color, font_scale=0.36, thickness=1)

    info_lines = [
        f"Seq: {frame.sequence_id}",
        f"Frame: {frame_index + 1}/{total_frames}",
        f"Exposure Start: {frame.exposure_start_us}",
        f"Objects: {len(objects)}",
        f"Events: {event_count}",
    ]
    for idx, text in enumerate(info_lines):
        cv2.putText(canvas, text, (12, 28 + idx * 22), TEXT_FONT, 0.55, BEV_TEXT, 2, cv2.LINE_AA)

    return canvas


def compose_frame(
    frame: FrameRecord,
    reader: EventStreamReader,
    frame_index: int,
    total_frames: int,
    *,
    image_width: int,
    bev_width: int,
    pixel_diff: int,
    fwd_range: Tuple[float, float],
    lat_range: Tuple[float, float],
) -> np.ndarray:
    image_bgr = cv2.imread(str(frame.image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {frame.image_path}")

    height, width = image_bgr.shape[:2]
    center_us = int(round((frame.exposure_start_us + frame.exposure_end_us) / 2.0))
    event_start_us = center_us - EVENT_WINDOW_US // 2
    event_end_us = center_us + EVENT_WINDOW_US // 2
    events = reader.extract(
        start_us=event_start_us,
        end_us=event_end_us,
        roi=(height, width),
        pixel_diff=pixel_diff,
    )

    rgb_panel = draw_front_annotations(image_bgr, frame.objects)
    event_panel = draw_front_annotations(render_events_on_canvas(image_bgr.shape, events), frame.objects)

    rgb_panel = resize_keep_width(rgb_panel, image_width)
    event_panel = resize_keep_width(event_panel, image_width)
    event_window_ms = EVENT_WINDOW_US / 1000.0

    left_column = stack_vertical(
        [
            add_title_bar(rgb_panel, "RGB"),
            add_title_bar(event_panel, f"Event Rendering ({event_window_ms:.2f} ms centered)"),
        ],
        gap=10,
        bg_color=BEV_BG,
    )

    bev_image = draw_bev(
        objects=frame.objects,
        canvas_w=bev_width,
        canvas_h=left_column.shape[0] - 34,
        fwd_range=fwd_range,
        lat_range=lat_range,
        frame=frame,
        frame_index=frame_index,
        total_frames=total_frames,
        event_count=int(events["x"].size),
    )
    right_column = add_title_bar(bev_image, "BEV + Frame Info")
    return stack_horizontal([left_column, right_column], gap=14, bg_color=BEV_BG)


class ReleaseSequenceVisualizer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.release_path = args.release_path.expanduser().resolve()
        self.sequence_dir = choose_sequence_dir(self.release_path, args.sequence_id)
        self.sequence_id = self.sequence_dir.name
        self.frames = load_sequence_frames(self.sequence_dir)
        self.output_dir = args.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fwd_range = (0.0, float(args.fwd_max))
        self.lat_range = (-float(args.lat_max), float(args.lat_max))

    def build_frame(self, frame: FrameRecord, reader: EventStreamReader, frame_index: int, total_frames: int) -> np.ndarray:
        return compose_frame(
            frame,
            reader,
            frame_index,
            total_frames,
            image_width=self.args.image_width,
            bev_width=self.args.bev_width,
            pixel_diff=self.args.pixel_diff,
            fwd_range=self.fwd_range,
            lat_range=self.lat_range,
        )

    def run(self) -> Path:
        frames = self.frames
        if self.args.frame_step > 1:
            frames = frames[:: self.args.frame_step]
        if self.args.max_frames is not None:
            frames = frames[: self.args.max_frames]
        if not frames:
            raise RuntimeError("No frames selected for rendering.")

        reader = EventStreamReader(frames[0].events_path)
        output_path = self.output_dir / f"{self.sequence_id}_release_visualization.mp4"
        writer: Optional[cv2.VideoWriter] = None

        try:
            for frame_index, frame in enumerate(tqdm(frames, desc=f"Rendering {self.sequence_id}")):
                composed = self.build_frame(frame, reader, frame_index, len(frames))
                if writer is None:
                    h, w = composed.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, float(self.args.fps), (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {output_path}")
                writer.write(composed)
        finally:
            reader.close()
            if writer is not None:
                writer.release()
        return output_path


def main() -> None:
    args = parse_args()
    if args.list_sequences:
        for seq in list_sequences(args.release_path.expanduser().resolve()):
            print(seq)
        return

    visualizer = ReleaseSequenceVisualizer(args)
    output_path = visualizer.run()
    print(f"saved_video={output_path}")


if __name__ == "__main__":
    main()
