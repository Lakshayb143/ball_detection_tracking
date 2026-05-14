#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ED_DIR = Path(__file__).resolve().parent
ROOT = ED_DIR.parent
if str(ED_DIR) not in sys.path:
    sys.path.insert(0, str(ED_DIR))

from airborne_rule import AirborneRuleConfig  # noqa: E402
from airborne_state_machine import (  # noqa: E402
    AirborneStateMachine,
    CompletedAirborneEvent,
    StateMachineConfig,
)


DEFAULT_DETECTIONS_DIR = ROOT / "detections_v5"
DEFAULT_FEATURES_ROOT = ROOT / "outputs" / "airborne_eval_v5"
DEFAULT_GT_DIR = ROOT / "ground_truths"
DEFAULT_VIDEOS_DIR = ROOT / "clips"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "airborne_prediction_visualizations_v5"


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {value!r}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}")
    return parsed


def natural_clip_key(name: str) -> Tuple[int, int, str]:
    match = re.fullmatch(r"clip(\d+)", name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def normalize_clip_name(value: str) -> str:
    text = str(value).strip()
    if text.isdigit():
        return f"clip{text}"
    if text.endswith(".json"):
        text = Path(text).stem
    if text.endswith("_actions"):
        text = text[: -len("_actions")]
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render batch videos showing airborne predictions from the event_detection "
            "state-machine pipeline."
        )
    )
    parser.add_argument("--clips", nargs="*", default=None, help="Optional subset: e.g. --clips 4 16")
    parser.add_argument("--detections_dir", type=Path, default=DEFAULT_DETECTIONS_DIR)
    parser.add_argument("--features_root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--gt_dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--videos_dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include_testing_clip",
        type=str2bool,
        default=True,
        help="Whether auto-discovery includes testing_clip_1080.",
    )
    parser.add_argument(
        "--prefer_existing_features",
        type=str2bool,
        default=True,
        help="Use outputs/airborne_eval_v5/<clip>/features.csv when present.",
    )
    parser.add_argument(
        "--buffer_l",
        type=positive_int,
        default=30,
        help="Feature buffer length when features must be computed from detection JSON.",
    )
    parser.add_argument("--show_gt", type=str2bool, default=True, help="Overlay GT airborne windows.")
    parser.add_argument("--show_ball", type=str2bool, default=True, help="Draw ball detections for context.")
    parser.add_argument("--show_trail", type=str2bool, default=True, help="Draw recent detection trail.")
    parser.add_argument("--trail_length", type=positive_int, default=18)
    parser.add_argument("--max_trail_gap", type=positive_int, default=5)
    parser.add_argument("--marker_radius", type=positive_int, default=8)
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS. 0 means infer from video.")
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--limit_frames", type=int, default=0, help="Debug: render only first N frames.")
    parser.add_argument("--overwrite", type=str2bool, default=True)
    return parser.parse_args()


def discover_clips(detections_dir: Path, include_testing_clip: bool) -> List[str]:
    clips = []
    for path in detections_dir.glob("*.json"):
        if path.stem.endswith("_phase1_rejections"):
            continue
        if path.stem == "testing_clip_1080" and not include_testing_clip:
            continue
        clips.append(path.stem)
    return sorted(clips, key=natural_clip_key)


def selected_clips(args: argparse.Namespace) -> List[str]:
    if args.clips:
        return sorted({normalize_clip_name(value) for value in args.clips}, key=natural_clip_key)
    return discover_clips(args.detections_dir, args.include_testing_clip)


def resolve_video_path(videos_dir: Path, clip_name: str) -> Path:
    candidates = [
        videos_dir / f"{clip_name}.mp4",
        videos_dir / f"{clip_name}_v5.mp4",
        videos_dir / f"{clip_name}_v4.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No source video found for {clip_name}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def maybe_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_feature_rows(features_csv: Path) -> List[dict]:
    rows: List[dict] = []
    with features_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {key: maybe_float(value) for key, value in raw.items()}
            row["frame"] = int(row["frame"])
            row["n_detections_in_buffer"] = int(row.get("n_detections_in_buffer", 0))
            row["frames_since_last_det"] = int(row.get("frames_since_last_det", 0))
            rows.append(row)
    return rows


def load_detection_payload(detections_path: Path) -> List[dict]:
    payload = json.loads(detections_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list detection JSON at {detections_path}")
    return payload


def detection_map(payload: Sequence[dict]) -> Dict[int, Tuple[float, float, Optional[float]]]:
    detections: Dict[int, Tuple[float, float, Optional[float]]] = {}
    for entry in payload:
        x = entry.get("x")
        y = entry.get("y")
        if x is None or y is None:
            continue
        conf_raw = entry.get("confidence")
        detections[int(entry["frame_idx"])] = (
            float(x),
            float(y),
            float(conf_raw) if conf_raw is not None else None,
        )
    return detections


def extract_feature_rows_from_detections(payload: Sequence[dict], buffer_l: int) -> List[dict]:
    det_by_frame = {}
    for entry in payload:
        frame_idx = int(entry["frame_idx"])
        x = entry.get("x")
        y = entry.get("y")
        conf = entry.get("confidence")
        det_by_frame[frame_idx] = (
            (float(x), float(y), float(conf) if conf is not None else 0.0)
            if x is not None and y is not None
            else None
        )

    if not det_by_frame:
        return []

    rows = []
    for frame_idx in range(max(det_by_frame) + 1):
        buffer_start = max(0, frame_idx - buffer_l + 1)
        buffer_dets = [
            (f, *det_by_frame[f])
            for f in range(buffer_start, frame_idx + 1)
            if f in det_by_frame and det_by_frame[f] is not None
        ]

        n_dets = len(buffer_dets)
        if n_dets > 0:
            ys = [det[2] for det in buffer_dets]
            median_y = float(np.median(ys))
            y_range = float(max(ys) - min(ys))
            last_frame, _last_x, last_y, _last_conf = buffer_dets[-1]
            frames_since_last = frame_idx - last_frame
        else:
            median_y = float("nan")
            y_range = float("nan")
            last_y = float("nan")
            frames_since_last = buffer_l

        if n_dets >= 2:
            f1, _x1, y1, _c1 = buffer_dets[-2]
            f2, _x2, y2, _c2 = buffer_dets[-1]
            dy_last_pair = float(y2 - y1)
            dy_per_frame_last_pair = float(dy_last_pair / max(1, f2 - f1))
            dy_buffer_first_to_last = float(buffer_dets[-1][2] - buffer_dets[0][2])
        else:
            dy_last_pair = float("nan")
            dy_per_frame_last_pair = float("nan")
            dy_buffer_first_to_last = float("nan")

        rows.append(
            {
                "frame": frame_idx,
                "n_detections_in_buffer": n_dets,
                "detection_density": n_dets / buffer_l,
                "median_y_in_buffer": median_y,
                "y_range_in_buffer": y_range,
                "frames_since_last_det": frames_since_last,
                "last_y": last_y,
                "dy_last_pair": dy_last_pair,
                "dy_per_frame_last_pair": dy_per_frame_last_pair,
                "dy_buffer_first_to_last": dy_buffer_first_to_last,
            }
        )
    return rows


def load_gt_windows(gt_dir: Path, features_root: Path, clip_name: str) -> List[dict]:
    gt_path = gt_dir / f"{clip_name}_actions.json"
    windows = []
    if gt_path.exists():
        payload = json.loads(gt_path.read_text(encoding="utf-8"))
        for event in payload.get("events", []):
            if "start_frame" not in event or "end_frame" not in event:
                continue
            windows.append(
                {
                    "start": int(event["start_frame"]),
                    "end": int(event["end_frame"]),
                    "action": str(event.get("action", "")),
                }
            )
    if windows:
        return windows

    eval_path = features_root / clip_name / "eval_result.json"
    if not eval_path.exists():
        return []
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    for item in payload.get("per_gt", []):
        if "gt_start" not in item or "gt_end" not in item:
            continue
        windows.append(
            {
                "start": int(item["gt_start"]),
                "end": int(item["gt_end"]),
                "action": "airborne",
            }
        )
    return windows


def run_airborne_detector(feature_rows: Sequence[dict]) -> List[CompletedAirborneEvent]:
    sm = AirborneStateMachine(AirborneRuleConfig(), StateMachineConfig())
    completed_events: List[CompletedAirborneEvent] = []
    for row in feature_rows:
        completed = sm.step(row)
        if completed is not None:
            completed_events.append(completed)

    if feature_rows and sm.state == sm.AIRBORNE and sm._active_event_start is not None:
        last_frame = int(feature_rows[-1]["frame"])
        completed_events.append(
            CompletedAirborneEvent(
                start_frame=sm._active_event_start,
                end_frame=last_frame,
                duration_frames=last_frame - sm._active_event_start,
                fire_score=sm._active_event_score,
                fire_breakdown=sm._active_event_breakdown,
                end_reason="still_active_at_end_of_video",
            )
        )
    return completed_events


def event_at_frame(frame_idx: int, events: Sequence[CompletedAirborneEvent]) -> Optional[CompletedAirborneEvent]:
    for event in events:
        if event.start_frame <= frame_idx <= predicted_end_frame(event):
            return event
    return None


def predicted_end_frame(event: CompletedAirborneEvent) -> int:
    return max(int(event.start_frame), int(event.start_frame + event.duration_frames - 1))


def gt_at_frame(frame_idx: int, windows: Sequence[dict]) -> Optional[dict]:
    for window in windows:
        if window["action"].lower() == "airborne" and window["start"] <= frame_idx <= window["end"]:
            return window
    return None


def draw_outlined_text(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(image: np.ndarray, lines: Sequence[str]) -> None:
    if not lines:
        return
    line_height = 24
    width = max(330, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0] for line in lines) + 28)
    height = 18 + line_height * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)
    for idx, line in enumerate(lines):
        draw_outlined_text(image, line, (22, 36 + idx * line_height), 0.62, (245, 245, 245), 1)


def draw_prediction_overlay(image: np.ndarray, event: CompletedAirborneEvent) -> None:
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h // 3), (w, 2 * h // 3), (0, 170, 255), -1)
    cv2.addWeighted(overlay, 0.20, image, 0.80, 0, image)
    label = "PRED AIRBORNE"
    sub = f"{event.start_frame}-{predicted_end_frame(event)}  score={event.fire_score:.2f}  {event.end_reason}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_size = cv2.getTextSize(label, font, 2.2, 5)[0]
    label_x = max(20, (w - label_size[0]) // 2)
    label_y = h // 2
    draw_outlined_text(image, label, (label_x, label_y), 2.2, (0, 220, 255), 5)
    sub_size = cv2.getTextSize(sub, font, 0.85, 2)[0]
    draw_outlined_text(image, sub, (max(20, (w - sub_size[0]) // 2), label_y + 44), 0.85, (255, 255, 255), 2)


def draw_gt_strip(image: np.ndarray, window: dict) -> None:
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (40, 40, 220), -1)
    cv2.addWeighted(overlay, 0.38, image, 0.62, 0, image)
    draw_outlined_text(
        image,
        f"GT AIRBORNE {window['start']}-{window['end']}",
        (w - 360, 30),
        0.72,
        (255, 255, 255),
        2,
    )


def detection_color(confidence: Optional[float]) -> Tuple[int, int, int]:
    if confidence is None:
        return (230, 230, 230)
    if confidence >= 0.65:
        return (40, 220, 40)
    if confidence >= 0.45:
        return (0, 220, 255)
    return (45, 45, 255)


def draw_ball_detection(
    image: np.ndarray,
    detection: Tuple[float, float, Optional[float]],
    marker_radius: int,
) -> None:
    x, y, conf = detection
    center = (int(round(x)), int(round(y)))
    color = detection_color(conf)
    cv2.circle(image, center, marker_radius + 3, (0, 0, 0), 4)
    cv2.circle(image, center, marker_radius + 3, color, 2)
    cv2.circle(image, center, max(2, marker_radius // 3), color, -1)
    label = "Ball" if conf is None else f"Ball {conf:.2f}"
    draw_outlined_text(image, label, (center[0] + marker_radius + 8, max(22, center[1] - 10)), 0.55, color, 1)


def draw_trail(
    image: np.ndarray,
    trail: Sequence[Tuple[int, float, float, Optional[float]]],
    max_gap: int,
) -> None:
    if len(trail) < 2:
        return
    for a, b in zip(trail[:-1], trail[1:]):
        fa, xa, ya, _ca = a
        fb, xb, yb, cb = b
        if fb - fa > max_gap:
            continue
        cv2.line(
            image,
            (int(round(xa)), int(round(ya))),
            (int(round(xb)), int(round(yb))),
            detection_color(cb),
            2,
            cv2.LINE_AA,
        )


def draw_timeline(
    image: np.ndarray,
    frame_idx: int,
    total_frames: int,
    pred_events: Sequence[CompletedAirborneEvent],
    gt_windows: Sequence[dict],
) -> None:
    if total_frames <= 0:
        return
    h, w = image.shape[:2]
    x0, x1 = 40, w - 40
    y_gt = h - 38
    y_pred = h - 20
    cv2.line(image, (x0, y_gt), (x1, y_gt), (120, 120, 120), 3)
    cv2.line(image, (x0, y_pred), (x1, y_pred), (120, 120, 120), 3)

    def fx(frame: int) -> int:
        return int(round(x0 + (x1 - x0) * max(0, min(total_frames - 1, frame)) / max(1, total_frames - 1)))

    for window in gt_windows:
        if window["action"].lower() != "airborne":
            continue
        cv2.line(image, (fx(window["start"]), y_gt), (fx(window["end"]), y_gt), (30, 30, 230), 7)
    for event in pred_events:
        cv2.line(image, (fx(event.start_frame), y_pred), (fx(predicted_end_frame(event)), y_pred), (0, 180, 255), 7)
    cv2.circle(image, (fx(frame_idx), y_gt), 6, (255, 255, 255), -1)
    draw_outlined_text(image, "GT", (10, y_gt + 5), 0.45, (255, 255, 255), 1)
    draw_outlined_text(image, "PRED", (10, y_pred + 5), 0.45, (255, 255, 255), 1)


def open_writer(output_path: Path, width: int, height: int, fps: float, codec: str) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    return writer


def predicted_intervals(pred_events: Sequence[CompletedAirborneEvent]) -> List[Tuple[int, int]]:
    return [(int(event.start_frame), int(predicted_end_frame(event))) for event in pred_events]


def gt_airborne_intervals(gt_windows: Sequence[dict]) -> List[Tuple[int, int]]:
    return [
        (int(window["start"]), int(window["end"]))
        for window in gt_windows
        if window["action"].lower() == "airborne"
    ]


def frames_in_intervals(intervals: Sequence[Tuple[int, int]]) -> set[int]:
    frames: set[int] = set()
    for start, end in intervals:
        if end < start:
            continue
        frames.update(range(start, end + 1))
    return frames


def render_clip(
    clip_name: str,
    video_path: Path,
    detections_path: Path,
    feature_rows: Sequence[dict],
    pred_events: Sequence[CompletedAirborneEvent],
    gt_windows: Sequence[dict],
    output_path: Path,
    args: argparse.Namespace,
) -> dict:
    pred_frames = frames_in_intervals(predicted_intervals(pred_events))
    gt_frames = frames_in_intervals(gt_airborne_intervals(gt_windows))
    if output_path.exists() and not args.overwrite:
        return {
            "clip": clip_name,
            "status": "skipped_exists",
            "video": str(video_path),
            "detections": str(detections_path),
            "output": str(output_path),
            "frames_rendered": 0,
            "n_pred_events": len(pred_events),
            "n_gt_airborne": len([w for w in gt_windows if w["action"].lower() == "airborne"]),
            "pred_active_frames": len(pred_frames),
            "gt_active_frames": len(gt_frames),
            "overlap_frames": len(pred_frames & gt_frames),
            "feature_rows": len(feature_rows),
        }

    det_by_frame = detection_map(load_detection_payload(detections_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(args.fps) if args.fps and args.fps > 0 else float(fps_in)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = open_writer(output_path, width, height, fps, args.codec)

    trail: Deque[Tuple[int, float, float, Optional[float]]] = deque(maxlen=args.trail_length)
    rendered = 0
    pred_active_frames = 0
    gt_active_frames = 0
    overlap_frames = 0
    last_detection_frame: Optional[int] = None

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.limit_frames > 0 and rendered >= args.limit_frames:
                break

            pred_event = event_at_frame(frame_idx, pred_events)
            gt_window = gt_at_frame(frame_idx, gt_windows) if args.show_gt else None
            if gt_window is not None:
                gt_active_frames += 1
                draw_gt_strip(frame, gt_window)
            if pred_event is not None:
                pred_active_frames += 1
                draw_prediction_overlay(frame, pred_event)
            if pred_event is not None and gt_window is not None:
                overlap_frames += 1

            detection = det_by_frame.get(frame_idx)
            if args.show_ball and detection is not None:
                last_detection_frame = frame_idx
                x, y, conf = detection
                trail.append((frame_idx, x, y, conf))
                if args.show_trail:
                    draw_trail(frame, list(trail), args.max_trail_gap)
                draw_ball_detection(frame, detection, args.marker_radius)
            elif trail and frame_idx - trail[-1][0] > args.max_trail_gap:
                trail.clear()

            gap = "NA" if last_detection_frame is None else str(frame_idx - last_detection_frame)
            lines = [
                f"{clip_name}  frame {frame_idx:04d}",
                f"pred airborne: {'yes' if pred_event else 'no'}",
                f"gt airborne: {'yes' if gt_window else 'no'}",
                f"gap since ball det: {gap}",
                f"pred events: {len(pred_events)}",
            ]
            draw_panel(frame, lines)
            draw_timeline(frame, frame_idx, total_frames, pred_events, gt_windows if args.show_gt else [])

            writer.write(frame)
            rendered += 1
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return {
        "clip": clip_name,
        "status": "rendered",
        "video": str(video_path),
        "detections": str(detections_path),
        "output": str(output_path),
        "frames_rendered": rendered,
        "n_pred_events": len(pred_events),
        "n_gt_airborne": len([w for w in gt_windows if w["action"].lower() == "airborne"]),
        "pred_active_frames": pred_active_frames,
        "gt_active_frames": gt_active_frames,
        "overlap_frames": overlap_frames,
        "feature_rows": len(feature_rows),
    }


def event_to_dict(event: CompletedAirborneEvent) -> dict:
    return {
        "start_frame": int(event.start_frame),
        "end_frame": int(predicted_end_frame(event)),
        "completed_end_frame": int(event.end_frame),
        "duration_frames": int(event.duration_frames),
        "fire_score": float(event.fire_score),
        "fire_breakdown": event.fire_breakdown,
        "end_reason": event.end_reason,
    }


def write_summary(output_dir: Path, rows: Sequence[dict], all_events: Dict[str, List[dict]]) -> Tuple[Path, Path]:
    summary_csv = output_dir / "summary.csv"
    events_json = output_dir / "predicted_events.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    events_json.write_text(json.dumps(all_events, indent=2), encoding="utf-8")
    return summary_csv, events_json


def main() -> None:
    args = parse_args()
    args.detections_dir = args.detections_dir.expanduser().resolve()
    args.features_root = args.features_root.expanduser().resolve()
    args.gt_dir = args.gt_dir.expanduser().resolve()
    args.videos_dir = args.videos_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clips = selected_clips(args)
    if not clips:
        raise FileNotFoundError(f"No clips found in {args.detections_dir}")

    summary_rows: List[dict] = []
    all_events: Dict[str, List[dict]] = {}
    print(f"[INFO] Rendering airborne prediction overlays for {len(clips)} clip(s)")
    for clip_name in clips:
        detections_path = args.detections_dir / f"{clip_name}.json"
        if not detections_path.exists():
            raise FileNotFoundError(f"Missing detections for {clip_name}: {detections_path}")
        video_path = resolve_video_path(args.videos_dir, clip_name)
        features_csv = args.features_root / clip_name / "features.csv"

        if args.prefer_existing_features and features_csv.exists():
            feature_rows = load_feature_rows(features_csv)
            feature_source = str(features_csv)
        else:
            feature_rows = extract_feature_rows_from_detections(load_detection_payload(detections_path), args.buffer_l)
            feature_source = str(detections_path)

        pred_events = run_airborne_detector(feature_rows)
        gt_windows = load_gt_windows(args.gt_dir, args.features_root, clip_name) if args.show_gt else []
        output_path = args.output_dir / f"{clip_name}_airborne_predictions.mp4"

        print(f"[INFO] {clip_name}: pred_events={len(pred_events)} -> {output_path.name}")
        row = render_clip(
            clip_name=clip_name,
            video_path=video_path,
            detections_path=detections_path,
            feature_rows=feature_rows,
            pred_events=pred_events,
            gt_windows=gt_windows,
            output_path=output_path,
            args=args,
        )
        row["feature_source"] = feature_source
        summary_rows.append(row)
        all_events[clip_name] = [event_to_dict(event) for event in pred_events]

    summary_csv, events_json = write_summary(args.output_dir, summary_rows, all_events)
    print(f"[INFO] Summary: {summary_csv}")
    print(f"[INFO] Predicted events: {events_json}")
    for row in summary_rows:
        print(
            "[DONE] {clip}: pred_events={n_pred_events}, frames={frames_rendered}, output={output}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
