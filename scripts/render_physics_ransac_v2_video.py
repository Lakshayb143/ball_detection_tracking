#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np


DEFAULT_VIDEO = Path("clips/clip1.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an annotated video for a physics-RANSAC v2 run.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output_video", type=Path, default=None)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--start_frame", type=int, default=1)
    parser.add_argument("--end_frame", type=int, default=0, help="0 means render through the end of the video/run.")
    parser.add_argument("--tail_length", type=int, default=16)
    parser.add_argument("--marker_radius", type=int, default=10)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--simple",
        action="store_true",
        help="Render only the tracked ball and trail. This is the default mode.",
    )
    mode_group.add_argument(
        "--detailed",
        action="store_true",
        help="Render the full diagnostic overlay with raw/final markers and HUD text.",
    )
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def maybe_float(value: object) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_frame_trace(path: Path) -> Dict[int, dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["frame_idx"]): row for row in csv.DictReader(handle)}


def load_point_records(path: Path) -> Dict[int, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = enumerate(raw)
    else:
        raise ValueError(f"Unsupported point output format: {type(raw).__name__}")

    records: Dict[int, dict] = {}
    for fallback_idx, value in items:
        if value is None:
            continue
        if isinstance(value, dict):
            frame_idx = int(value.get("frame_idx", value.get("frame", fallback_idx)))
            x = value.get("x")
            y = value.get("y")
            records[frame_idx] = {
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
                "source": str(value.get("source", "unknown")),
                "interpolated": bool(value.get("interpolated", value.get("interpolated_output", False))),
                "confidence": value.get("confidence", value.get("score")),
            }
        else:
            if len(value) < 2 or value[0] is None or value[1] is None:
                continue
            records[int(fallback_idx)] = {
                "x": float(value[0]),
                "y": float(value[1]),
                "source": "tracker",
                "interpolated": False,
                "confidence": value[4] if len(value) > 4 else 1.0,
            }
    return records


def draw_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    *,
    scale: float = 0.65,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_point(
    image: np.ndarray,
    x: float,
    y: float,
    *,
    color: Tuple[int, int, int],
    radius: int,
    label: str = "",
    thickness: int = 2,
) -> None:
    center = (int(round(x)), int(round(y)))
    cv2.circle(image, center, radius, color, thickness, cv2.LINE_AA)
    cv2.circle(image, center, 2, color, -1, cv2.LINE_AA)
    if label:
        draw_text(image, label, (center[0] + radius + 6, center[1] - radius - 4), scale=0.52, color=color, thickness=1)


def draw_trail(image: np.ndarray, trail: Deque[Tuple[float, float]]) -> None:
    if len(trail) < 2:
        return
    points = np.array([[int(round(x)), int(round(y))] for x, y in trail], dtype=np.int32)
    for idx in range(1, len(points)):
        alpha = idx / max(1, len(points) - 1)
        color = (int(80 + 120 * alpha), int(180 + 60 * alpha), int(80 + 80 * alpha))
        cv2.line(image, tuple(points[idx - 1]), tuple(points[idx]), color, 2, cv2.LINE_AA)


def infer_output_video(run_dir: Path, output_video: Optional[Path], *, detailed: bool) -> Path:
    if output_video is not None:
        return output_video.expanduser().resolve()
    if detailed:
        return run_dir / "physics_ransac_v2_overlay.mp4"
    return run_dir / "physics_ransac_v2_ball_tracking.mp4"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    detailed_mode = bool(args.detailed)
    output_video = infer_output_video(run_dir, args.output_video, detailed=detailed_mode)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    frame_trace = load_frame_trace(run_dir / "frame_trace.csv")
    summary = json.loads((run_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    tracker_json = Path(summary["tracker_json"]).expanduser().resolve()
    raw_points = load_point_records(tracker_json)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = int(args.end_frame) if int(args.end_frame) > 0 else min(frame_count, max(frame_trace))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*str(args.codec)),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video}")

    trail: Deque[Tuple[float, float]] = deque(maxlen=max(1, int(args.tail_length)))
    repaired_count = 0
    rendered = 0
    try:
        frame_idx = 1
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if frame_idx > end_frame:
                break
            if frame_idx < int(args.start_frame):
                frame_idx += 1
                continue

            trace = frame_trace.get(frame_idx, {})
            raw = raw_points.get(frame_idx, {})
            final_x = maybe_float(trace.get("final_x", ""))
            final_y = maybe_float(trace.get("final_y", ""))
            final_source = str(trace.get("final_source", ""))
            raw_x = raw.get("x")
            raw_y = raw.get("y")
            raw_interpolated = bool(raw.get("interpolated", False))

            if final_x is not None and final_y is not None:
                trail.append((final_x, final_y))
                if final_source == "ransac_repaired_interpolation":
                    repaired_count += 1
            draw_trail(image, trail)

            if detailed_mode:
                if raw_x is not None and raw_y is not None:
                    raw_color = (0, 180, 255) if raw_interpolated else (0, 255, 255)
                    raw_label = "v4 interp" if raw_interpolated else "v4 raw"
                    draw_point(
                        image,
                        float(raw_x),
                        float(raw_y),
                        color=raw_color,
                        radius=max(5, int(args.marker_radius) - 3),
                        label=raw_label,
                        thickness=2,
                    )

                if final_x is not None and final_y is not None:
                    if final_source == "ransac_repaired_interpolation":
                        color = (70, 255, 70)
                        label = "RANSAC repair"
                        if raw_x is not None and raw_y is not None:
                            cv2.line(
                                image,
                                (int(round(float(raw_x))), int(round(float(raw_y)))),
                                (int(round(final_x)), int(round(final_y))),
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                    elif final_source == "ransac_filled_gap":
                        color = (255, 0, 255)
                        label = "RANSAC fill"
                    else:
                        color = (0, 220, 0)
                        label = "final"
                    draw_point(
                        image,
                        final_x,
                        final_y,
                        color=color,
                        radius=int(args.marker_radius) + (4 if final_source.startswith("ransac_") else 0),
                        label=label,
                        thickness=3,
                    )

                if not parse_bool(trace.get("final_kept", False)):
                    draw_text(image, "NO FINAL OUTPUT", (18, 170), color=(0, 0, 255), scale=0.75, thickness=2)

                hud_lines = [
                    f"Frame {frame_idx:04d} | {Path(summary['run_name']).name}",
                    f"Final: {final_source or 'none'}",
                    f"Segment: {trace.get('segment', '') or 'outside'}  ok={trace.get('segment_physics_ok', '')}",
                    f"Fit: {trace.get('fit_source', '') or '-'}  residual={trace.get('ransac_residual_px', '') or '-'}",
                    f"Raw: {raw.get('source', 'none')}  interpolated={raw_interpolated}",
                    f"Repaired so far: {repaired_count}",
                ]
                for line_idx, text in enumerate(hud_lines):
                    draw_text(image, text, (18, 30 + line_idx * 26), scale=0.66, color=(255, 255, 255), thickness=1)
            elif final_x is not None and final_y is not None:
                draw_point(
                    image,
                    final_x,
                    final_y,
                    color=(0, 220, 0),
                    radius=int(args.marker_radius),
                    thickness=3,
                )

            writer.write(image)
            rendered += 1
            frame_idx += 1
    finally:
        writer.release()
        cap.release()

    print(f"[INFO] Rendered frames: {rendered}")
    print(f"[INFO] Repaired frames encountered: {repaired_count}")
    print(f"[INFO] Render mode: {'detailed' if detailed_mode else 'simple'}")
    print(f"[INFO] Output video: {output_video}")


if __name__ == "__main__":
    main()
