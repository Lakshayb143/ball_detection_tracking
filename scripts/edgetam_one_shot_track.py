#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from _edgetam_ball_track_common import (
    DEFAULT_DEVICE,
    DEFAULT_EDGETAM_CHECKPOINT,
    DEFAULT_EDGETAM_CONFIG,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_OFFLOAD_STATE_TO_CPU,
    DEFAULT_OFFLOAD_VIDEO_TO_CPU,
    DEFAULT_RFDETR_CONFIDENCE,
    DEFAULT_RFDETR_MODEL_PATH,
    DEFAULT_RFDETR_RESOLUTION,
    DEFAULT_USE_AMP,
    PROMPT_MODES,
    EdgeTAMVideoTracker,
    build_prompt_payload,
    build_rfdetr_detector,
    ensure_dir,
    infer_fps,
    list_frame_paths,
    render_track_video,
    scan_best_seed,
    str2bool,
    summarize_predictions,
    track_full_clip_windowed,
    write_predictions_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "edgetam_one_shot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a full clip with EdgeTAM from a single RF-DETR 1120 seed."
    )
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)

    parser.add_argument("--rfdetr_model_path", type=Path, default=DEFAULT_RFDETR_MODEL_PATH)
    parser.add_argument("--rfdetr_resolution", type=int, default=DEFAULT_RFDETR_RESOLUTION)
    parser.add_argument("--rfdetr_confidence", type=float, default=DEFAULT_RFDETR_CONFIDENCE)
    parser.add_argument("--optimize_for_inference", type=str2bool, default=True)
    parser.add_argument("--seed_frame_index", type=int, default=-1)
    parser.add_argument("--seed_scan_stride", type=int, default=1)
    parser.add_argument("--prompt_mode", choices=PROMPT_MODES, default="box")

    parser.add_argument("--edgetam_checkpoint", type=Path, default=DEFAULT_EDGETAM_CHECKPOINT)
    parser.add_argument("--edgetam_config", type=str, default=DEFAULT_EDGETAM_CONFIG)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--use_amp", type=str2bool, default=DEFAULT_USE_AMP)
    parser.add_argument("--mask_threshold", type=float, default=DEFAULT_MASK_THRESHOLD)
    parser.add_argument("--offload_video_to_cpu", type=str2bool, default=DEFAULT_OFFLOAD_VIDEO_TO_CPU)
    parser.add_argument("--offload_state_to_cpu", type=str2bool, default=DEFAULT_OFFLOAD_STATE_TO_CPU)
    parser.add_argument(
        "--async_loading_frames",
        type=str2bool,
        default=False,
        help="Defaults to false here to avoid aggressive full-clip background loading.",
    )
    parser.add_argument(
        "--max_window_frames",
        type=int,
        default=90,
        help="Track in bounded windows to keep EdgeTAM state small and avoid server stalls.",
    )
    parser.add_argument("--fps", type=float, default=0.0)
    return parser.parse_args()


def default_run_name(frames_dir: Path, prompt_mode: str) -> str:
    return f"{frames_dir.name}__one_shot__{prompt_mode}"


def main() -> None:
    args = parse_args()
    args.frames_dir = args.frames_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.rfdetr_model_path = args.rfdetr_model_path.expanduser().resolve()
    args.edgetam_checkpoint = args.edgetam_checkpoint.expanduser().resolve()

    frame_paths = list_frame_paths(
        args.frames_dir,
        start_frame=max(0, int(args.start_frame)),
        max_frames=max(0, int(args.max_frames)),
    )
    run_name = args.run_name.strip() or default_run_name(args.frames_dir, args.prompt_mode)
    run_dir = ensure_dir(args.output_root / run_name)
    fps = float(args.fps) if float(args.fps) > 0 else infer_fps(args.frames_dir)

    print(f"[INFO] Frames: {len(frame_paths)}")
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Prompt mode: {args.prompt_mode}")

    detector = build_rfdetr_detector(
        model_path=args.rfdetr_model_path,
        confidence_threshold=args.rfdetr_confidence,
        resolution=int(args.rfdetr_resolution) if int(args.rfdetr_resolution) > 0 else None,
        optimize_for_inference=bool(args.optimize_for_inference),
    )
    seed = scan_best_seed(
        detector,
        frame_paths,
        seed_frame_index=int(args.seed_frame_index),
        seed_scan_stride=max(1, int(args.seed_scan_stride)),
    )
    seed_frame = frame_paths[seed.frame_index]
    seed_shape = cv2.imread(str(seed_frame)).shape[:2]
    prompt_payload = build_prompt_payload(
        args.prompt_mode,
        seed.bbox_xyxy,
        seed_shape,
    )
    prompt_payload["mode"] = args.prompt_mode

    print(
        f"[INFO] Seed frame: {seed.frame_index + 1} ({seed.frame_name}) "
        f"score={seed.score:.4f}"
    )

    tracker = EdgeTAMVideoTracker(
        checkpoint_path=args.edgetam_checkpoint,
        config_name=args.edgetam_config,
        device=args.device,
        use_amp=bool(args.use_amp),
        mask_threshold=float(args.mask_threshold),
        offload_video_to_cpu=bool(args.offload_video_to_cpu),
        offload_state_to_cpu=bool(args.offload_state_to_cpu),
        async_loading_frames=bool(args.async_loading_frames),
    )
    predictions_by_index = track_full_clip_windowed(
        tracker=tracker,
        frame_paths=frame_paths,
        seed=seed,
        prompt_payload=prompt_payload,
        run_dir=run_dir,
        source="edgetam_one_shot",
        max_window_frames=max(2, int(args.max_window_frames)),
        async_loading_frames=bool(args.async_loading_frames),
    )

    predictions_csv = run_dir / "predictions.csv"
    summary_path = run_dir / "summary.json"
    output_video = run_dir / "overlay.mp4"

    write_predictions_csv(
        path=predictions_csv,
        frame_paths=frame_paths,
        predictions_by_index=predictions_by_index,
    )
    render_track_video(
        frame_paths=frame_paths,
        predictions_by_index=predictions_by_index,
        output_video=output_video,
        fps=fps,
        run_label=run_name,
    )

    summary = {
        "run_name": run_name,
        "frames_dir": str(args.frames_dir),
        "fps": fps,
        "prompt_mode": args.prompt_mode,
        "seed": {
            "frame_index": seed.frame_index + 1,
            "frame_name": seed.frame_name,
            "frame_path": seed.frame_path,
            "bbox_xyxy": seed.bbox_xyxy,
            "score": seed.score,
        },
        "prompt_preview": prompt_payload["prompt_preview"],
        "summary": summarize_predictions(
            predictions_by_index=predictions_by_index,
            total_frames=len(frame_paths),
        ),
        "outputs": {
            "predictions_csv": str(predictions_csv),
            "overlay_video": str(output_video),
        },
        "config": {
            "rfdetr_model_path": str(args.rfdetr_model_path),
            "rfdetr_resolution": int(args.rfdetr_resolution),
            "rfdetr_confidence": float(args.rfdetr_confidence),
            "seed_frame_index": int(args.seed_frame_index),
            "seed_scan_stride": int(args.seed_scan_stride),
            "edgetam_checkpoint": str(args.edgetam_checkpoint),
            "edgetam_config": args.edgetam_config,
            "device": args.device,
            "use_amp": bool(args.use_amp),
            "mask_threshold": float(args.mask_threshold),
            "offload_video_to_cpu": bool(args.offload_video_to_cpu),
            "offload_state_to_cpu": bool(args.offload_state_to_cpu),
            "async_loading_frames": bool(args.async_loading_frames),
            "max_window_frames": int(args.max_window_frames),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Predictions written to {predictions_csv}")
    print(f"[INFO] Overlay video written to {output_video}")
    print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
