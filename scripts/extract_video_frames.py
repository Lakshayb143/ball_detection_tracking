#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_OUTPUT_ROOT = Path("outputs") / "video_frames"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {value!r}")
    return parsed


def infer_default_output_dir(video_path: Path, output_root: Path) -> Path:
    return output_root / video_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ordered video frames for temporal tracking or visual inspection."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input video path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory. Defaults to outputs/video_frames/<video_stem>/.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base output root used when --output_dir is omitted.",
    )
    parser.add_argument(
        "--layout",
        choices=("mot", "flat"),
        default="mot",
        help="mot creates output_dir/img1 plus seqinfo.ini; flat writes frames directly to output_dir.",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help="Frame image extension, for example .jpg or .png.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality when writing .jpg/.jpeg frames.",
    )
    parser.add_argument(
        "--png_compression",
        type=int,
        default=3,
        help="PNG compression level when writing .png frames.",
    )
    parser.add_argument(
        "--start_frame",
        type=non_negative_int,
        default=0,
        help="Zero-based frame index to start from.",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help="Inclusive zero-based end frame. Use -1 for the full video.",
    )
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=1,
        help="Keep every Nth frame.",
    )
    parser.add_argument(
        "--max_frames",
        type=positive_int,
        default=0,
        help="Optional cap on how many output frames to write. 0 means no cap.",
    )
    parser.add_argument(
        "--zero_pad",
        type=positive_int,
        default=6,
        help="Zero-padding width for frame filenames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output frame directory.",
    )
    return parser.parse_args()


def frame_output_dir(output_dir: Path, layout: str) -> Path:
    if layout == "mot":
        return output_dir / "img1"
    return output_dir


def validate_args(args: argparse.Namespace) -> None:
    image_ext = args.image_ext.lower()
    if not image_ext.startswith("."):
        image_ext = f".{image_ext}"
    if image_ext not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {args.image_ext!r}")
    args.image_ext = image_ext

    if args.end_frame >= 0 and args.end_frame < args.start_frame:
        raise ValueError(
            f"--end_frame ({args.end_frame}) must be -1 or >= --start_frame ({args.start_frame})"
        )
    if not (0 <= args.jpeg_quality <= 100):
        raise ValueError(f"--jpeg_quality must be in [0, 100], got {args.jpeg_quality}")
    if not (0 <= args.png_compression <= 9):
        raise ValueError(f"--png_compression must be in [0, 9], got {args.png_compression}")


def write_seqinfo(
    path: Path,
    *,
    sequence_name: str,
    frame_rate: float,
    seq_length: int,
    width: int,
    height: int,
    image_ext: str,
) -> None:
    content = "\n".join(
        [
            "[Sequence]",
            f"name={sequence_name}",
            "imDir=img1",
            f"frameRate={int(round(frame_rate)) if frame_rate > 0 else 0}",
            f"seqLength={seq_length}",
            f"imWidth={width}",
            f"imHeight={height}",
            f"imExt={image_ext}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def encode_params(image_ext: str, jpeg_quality: int, png_compression: int) -> list[int]:
    if image_ext in {".jpg", ".jpeg"}:
        return [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    if image_ext == ".png":
        return [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
    return []


def main() -> None:
    args = parse_args()
    validate_args(args)

    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_root = args.output_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else infer_default_output_dir(video_path, output_root.expanduser().resolve())
    )
    frames_dir = frame_output_dir(output_dir, args.layout)

    if frames_dir.exists() and any(frames_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output frame directory is not empty: {frames_dir}. Use --overwrite to continue."
        )

    ensure_dir(frames_dir)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    target_end_frame = args.end_frame
    if target_end_frame < 0 and source_frame_count > 0:
        target_end_frame = source_frame_count - 1

    write_params = encode_params(args.image_ext, args.jpeg_quality, args.png_compression)

    source_index = -1
    written_count = 0
    first_written_source_index: Optional[int] = None
    last_written_source_index: Optional[int] = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            source_index += 1
            if source_index < args.start_frame:
                continue
            if target_end_frame >= 0 and source_index > target_end_frame:
                break
            if (source_index - args.start_frame) % args.stride != 0:
                continue
            if args.max_frames > 0 and written_count >= args.max_frames:
                break

            written_count += 1
            if first_written_source_index is None:
                first_written_source_index = source_index
            last_written_source_index = source_index

            file_name = f"{written_count:0{args.zero_pad}d}{args.image_ext}"
            frame_path = frames_dir / file_name
            if not cv2.imwrite(str(frame_path), frame, write_params):
                raise RuntimeError(f"Failed to write frame: {frame_path}")
    finally:
        capture.release()

    if written_count == 0:
        raise RuntimeError("No frames were written. Check the start/end/stride settings.")

    if args.layout == "mot":
        write_seqinfo(
            output_dir / "seqinfo.ini",
            sequence_name=video_path.stem,
            frame_rate=source_fps,
            seq_length=written_count,
            width=frame_width,
            height=frame_height,
            image_ext=args.image_ext,
        )

    summary = {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "frames_dir": str(frames_dir),
        "layout": args.layout,
        "image_ext": args.image_ext,
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "start_frame": args.start_frame,
        "end_frame": target_end_frame,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "zero_pad": args.zero_pad,
        "frames_written": written_count,
        "first_written_source_frame": first_written_source_index,
        "last_written_source_frame": last_written_source_index,
    }
    (output_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Frames dir: {frames_dir}")
    print(f"[INFO] Wrote {written_count} frame(s) as {args.image_ext} with zero-pad {args.zero_pad}")
    if args.layout == "mot":
        print(f"[INFO] seqinfo.ini written to {output_dir / 'seqinfo.ini'}")
    print(f"[INFO] Summary written to {output_dir / 'extraction_summary.json'}")


if __name__ == "__main__":
    main()
