#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_FPS = 30.0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive float, got {value!r}")
    return parsed


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a prediction CSV or a rendered frame folder into a video."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--pred_csv",
        type=Path,
        help="Prediction CSV with frame_path/frame_name/x/y/visible columns, such as WASB-SBDT infer_frames output.",
    )
    mode.add_argument(
        "--rendered_frames_dir",
        type=Path,
        help="Directory of already-rendered frames to stitch into a video.",
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        help="Optional raw frame directory. Needed only when the CSV lacks valid frame_path values.",
    )
    parser.add_argument(
        "--output_video",
        type=Path,
        help="Output .mp4 path. Defaults next to the input.",
    )
    parser.add_argument(
        "--fps",
        type=positive_float,
        default=0.0,
        help="Output video fps. If omitted, infer from seqinfo.ini or extraction_summary.json, else use 30.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC video codec, default mp4v.",
    )
    parser.add_argument(
        "--tail_length",
        type=positive_int,
        default=12,
        help="How many recent visible points to draw as a trail in CSV-render mode.",
    )
    parser.add_argument(
        "--marker_radius",
        type=positive_int,
        default=9,
        help="Visible-point marker radius in CSV-render mode.",
    )
    parser.add_argument(
        "--line_thickness",
        type=positive_int,
        default=2,
        help="Line thickness for markers and trail.",
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=0.7,
        help="Font scale for overlay text in CSV-render mode.",
    )
    parser.add_argument(
        "--show_frame_label",
        type=str2bool,
        default=True,
        help="Whether to draw frame number text in CSV-render mode.",
    )
    parser.add_argument(
        "--show_score",
        type=str2bool,
        default=True,
        help="Whether to draw score text near visible points in CSV-render mode.",
    )
    parser.add_argument(
        "--show_num_candidates",
        type=str2bool,
        default=True,
        help="Whether to draw candidate count text in CSV-render mode.",
    )
    parser.add_argument(
        "--show_miss_label",
        type=str2bool,
        default=True,
        help="Whether to draw a small 'NO TRACK' label when visible is false in CSV-render mode.",
    )
    return parser.parse_args()


def parse_seqinfo_fps(seqinfo_path: Path) -> Optional[float]:
    if not seqinfo_path.exists():
        return None
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(seqinfo_path)
    if not parser.has_section("Sequence"):
        return None
    try:
        return float(parser.get("Sequence", "frameRate"))
    except (configparser.Error, ValueError):
        return None


def parse_extraction_summary_fps(summary_path: Path) -> Optional[float]:
    if not summary_path.exists():
        return None
    try:
        import json

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        fps = payload.get("source_fps")
        if fps in ("", None):
            return None
        return float(fps)
    except (OSError, ValueError, TypeError):
        return None


def infer_fps_from_context(paths: Sequence[Path]) -> float:
    for path in paths:
        if path is None:
            continue
        if path.is_file():
            search_dir = path.parent
        else:
            search_dir = path
        candidates = [
            search_dir / "seqinfo.ini",
            search_dir.parent / "seqinfo.ini",
            search_dir / "extraction_summary.json",
            search_dir.parent / "extraction_summary.json",
        ]
        for candidate in candidates:
            if candidate.name == "seqinfo.ini":
                fps = parse_seqinfo_fps(candidate)
            else:
                fps = parse_extraction_summary_fps(candidate)
            if fps is not None and fps > 0:
                return fps
    return DEFAULT_FPS


def infer_output_video(args: argparse.Namespace) -> Path:
    if args.output_video is not None:
        return args.output_video.expanduser().resolve()
    if args.pred_csv is not None:
        return args.pred_csv.expanduser().resolve().with_suffix(".mp4")
    rendered_dir = args.rendered_frames_dir.expanduser().resolve()
    return rendered_dir.parent / f"{rendered_dir.name}.mp4"


def list_rendered_frames(rendered_frames_dir: Path) -> List[Path]:
    frames = sorted(
        path
        for path in rendered_frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frames:
        raise FileNotFoundError(f"No rendered frames found in {rendered_frames_dir}")
    return frames


def bool_from_csv(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "on"}


def maybe_float(value: object) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_prediction_rows(pred_csv: Path, frames_dir: Optional[Path]) -> List[dict]:
    rows: List[dict] = []
    with pred_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_path = Path(str(row.get("frame_path", "")).strip()) if row.get("frame_path") else None
            if frame_path is not None and str(frame_path) and frame_path.exists():
                resolved_frame_path = frame_path
            else:
                frame_name = str(row.get("frame_name", "")).strip()
                if not frame_name or frames_dir is None:
                    raise FileNotFoundError(
                        "CSV row does not contain a valid frame_path and --frames_dir was not provided."
                    )
                resolved_frame_path = frames_dir / frame_name

            rows.append(
                {
                    "frame_index": int(float(row.get("frame_index", len(rows)) or len(rows))),
                    "frame_path": resolved_frame_path,
                    "frame_name": resolved_frame_path.name,
                    "num_candidates": int(float(row.get("num_candidates", 0) or 0)),
                    "x": maybe_float(row.get("x", "")),
                    "y": maybe_float(row.get("y", "")),
                    "visible": bool_from_csv(row.get("visible", False)),
                    "score": maybe_float(row.get("score", "")),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No prediction rows found in {pred_csv}")
    return rows


def open_video_writer(output_video: Path, sample_frame: Path, fps: float, codec: str) -> cv2.VideoWriter:
    image = cv2.imread(str(sample_frame))
    if image is None:
        raise RuntimeError(f"Could not read sample frame: {sample_frame}")
    height, width = image.shape[:2]
    ensure_parent(output_video)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_video}")
    return writer


def blend_color(index: int, total: int) -> Tuple[int, int, int]:
    if total <= 1:
        return (0, 200, 255)
    alpha = float(index + 1) / float(total)
    blue = int(round(255 * (1.0 - alpha)))
    green = int(round(160 + 95 * alpha))
    red = int(round(255 * alpha))
    return (blue, green, red)


def draw_prediction_overlay(
    frame,
    row: dict,
    trail_points: Iterable[Tuple[int, int]],
    args: argparse.Namespace,
):
    output = frame
    if args.show_frame_label:
        cv2.putText(
            output,
            f"Frame {int(row['frame_index']) + 1:06d}",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(args.font_scale),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if args.show_num_candidates:
        cv2.putText(
            output,
            f"Candidates: {int(row['num_candidates'])}",
            (24, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(max(0.45, args.font_scale * 0.8)),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    trail_list = list(trail_points)
    for index, (x, y) in enumerate(trail_list):
        color = blend_color(index, len(trail_list))
        radius = max(2, int(round(float(args.marker_radius) * (0.25 + 0.4 * (index + 1) / max(1, len(trail_list))))))
        cv2.circle(output, (x, y), radius, color, -1)

    if row["visible"] and row["x"] is not None and row["y"] is not None:
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        cv2.circle(output, (x, y), int(args.marker_radius), (0, 255, 0), int(args.line_thickness))
        cv2.line(output, (x - 12, y), (x + 12, y), (0, 255, 0), int(args.line_thickness))
        cv2.line(output, (x, y - 12), (x, y + 12), (0, 255, 0), int(args.line_thickness))
        if args.show_score and row["score"] is not None:
            cv2.putText(
                output,
                f"{float(row['score']):.3f}",
                (max(12, x - 26), max(24, y - 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(max(0.45, args.font_scale * 0.75)),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    elif args.show_miss_label:
        cv2.putText(
            output,
            "NO TRACK",
            (24, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(max(0.5, args.font_scale * 0.8)),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def render_csv_to_video(rows: Sequence[dict], output_video: Path, fps: float, args: argparse.Namespace) -> None:
    writer = open_video_writer(output_video, Path(rows[0]["frame_path"]), fps=fps, codec=args.codec)
    trail: Deque[Tuple[int, int]] = deque(maxlen=int(args.tail_length))
    try:
        for row in rows:
            frame_path = Path(row["frame_path"])
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_path}")
            if row["visible"] and row["x"] is not None and row["y"] is not None:
                trail.append((int(round(float(row["x"]))), int(round(float(row["y"])))))
            overlaid = draw_prediction_overlay(frame, row, trail, args)
            writer.write(overlaid)
    finally:
        writer.release()


def stitch_frames_to_video(frame_paths: Sequence[Path], output_video: Path, fps: float, codec: str) -> None:
    writer = open_video_writer(output_video, frame_paths[0], fps=fps, codec=codec)
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read rendered frame: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()

    frames_dir = args.frames_dir.expanduser().resolve() if args.frames_dir is not None else None
    output_video = infer_output_video(args)

    if args.pred_csv is not None:
        pred_csv = args.pred_csv.expanduser().resolve()
        rows = read_prediction_rows(pred_csv, frames_dir)
        fps = float(args.fps) if args.fps > 0 else infer_fps_from_context(
            [
                pred_csv,
                Path(rows[0]["frame_path"]).parent,
                frames_dir if frames_dir is not None else Path(rows[0]["frame_path"]).parent,
            ]
        )
        render_csv_to_video(rows, output_video=output_video, fps=fps, args=args)
        print(f"[INFO] Rendered prediction CSV: {pred_csv}")
        print(f"[INFO] Frames used: {Path(rows[0]['frame_path']).parent}")
        print(f"[INFO] Rows rendered: {len(rows)}")
    else:
        rendered_frames_dir = args.rendered_frames_dir.expanduser().resolve()
        frame_paths = list_rendered_frames(rendered_frames_dir)
        fps = float(args.fps) if args.fps > 0 else infer_fps_from_context([rendered_frames_dir])
        stitch_frames_to_video(frame_paths, output_video=output_video, fps=fps, codec=args.codec)
        print(f"[INFO] Stitched rendered frames: {rendered_frames_dir}")
        print(f"[INFO] Frames stitched: {len(frame_paths)}")

    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Output video: {output_video}")


if __name__ == "__main__":
    main()
