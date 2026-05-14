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
    EdgeTAMVideoTracker,
    build_prompt_payload,
    build_rfdetr_detector,
    csv_write_dicts,
    ensure_dir,
    infer_fps,
    list_frame_paths,
    parse_mode_csv,
    render_track_video,
    scan_best_seed,
    str2bool,
    summarize_predictions,
    track_full_clip_windowed,
    write_predictions_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "edgetam_prompt_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare EdgeTAM prompt modes on the same RF-DETR seed."
    )
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--prompt_modes", type=str, default="box,point,box_point,box_point_neg")

    parser.add_argument("--rfdetr_model_path", type=Path, default=DEFAULT_RFDETR_MODEL_PATH)
    parser.add_argument("--rfdetr_resolution", type=int, default=DEFAULT_RFDETR_RESOLUTION)
    parser.add_argument("--rfdetr_confidence", type=float, default=DEFAULT_RFDETR_CONFIDENCE)
    parser.add_argument("--optimize_for_inference", type=str2bool, default=True)
    parser.add_argument("--seed_frame_index", type=int, default=-1)
    parser.add_argument("--seed_scan_stride", type=int, default=1)

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


def default_run_name(frames_dir: Path) -> str:
    return f"{frames_dir.name}__prompt_sweep"


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
    prompt_modes = parse_mode_csv(args.prompt_modes)
    run_name = args.run_name.strip() or default_run_name(args.frames_dir)
    run_dir = ensure_dir(args.output_root / run_name)
    fps = float(args.fps) if float(args.fps) > 0 else infer_fps(args.frames_dir)

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
    seed_shape = cv2.imread(str(frame_paths[seed.frame_index])).shape[:2]

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

    comparison_rows = []
    for prompt_mode in prompt_modes:
        mode_dir = ensure_dir(run_dir / prompt_mode)
        prompt_payload = build_prompt_payload(
            prompt_mode,
            seed.bbox_xyxy,
            seed_shape,
        )
        prompt_payload["mode"] = prompt_mode
        predictions_by_index = track_full_clip_windowed(
            tracker=tracker,
            frame_paths=frame_paths,
            seed=seed,
            prompt_payload=prompt_payload,
            run_dir=mode_dir,
            source=f"edgetam_prompt_sweep_{prompt_mode}",
            max_window_frames=max(2, int(args.max_window_frames)),
            async_loading_frames=bool(args.async_loading_frames),
        )

        predictions_csv = mode_dir / "predictions.csv"
        summary_path = mode_dir / "summary.json"
        overlay_video = mode_dir / "overlay.mp4"

        write_predictions_csv(
            path=predictions_csv,
            frame_paths=frame_paths,
            predictions_by_index=predictions_by_index,
        )
        render_track_video(
            frame_paths=frame_paths,
            predictions_by_index=predictions_by_index,
            output_video=overlay_video,
            fps=fps,
            run_label=f"{run_name} | {prompt_mode}",
        )

        run_summary = summarize_predictions(
            predictions_by_index=predictions_by_index,
            total_frames=len(frame_paths),
        )
        payload = {
            "run_name": run_name,
            "prompt_mode": prompt_mode,
            "frames_dir": str(args.frames_dir),
            "fps": fps,
            "seed": {
                "frame_index": seed.frame_index + 1,
                "frame_name": seed.frame_name,
                "frame_path": seed.frame_path,
                "bbox_xyxy": seed.bbox_xyxy,
                "score": seed.score,
            },
            "prompt_preview": prompt_payload["prompt_preview"],
            "summary": run_summary,
            "outputs": {
                "predictions_csv": str(predictions_csv),
                "overlay_video": str(overlay_video),
            },
            "config": {
                "max_window_frames": int(args.max_window_frames),
                "async_loading_frames": bool(args.async_loading_frames),
            },
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        comparison_rows.append(
            {
                "prompt_mode": prompt_mode,
                "predicted_frames": run_summary["predicted_frames"],
                "coverage": run_summary["coverage"],
                "mean_score": run_summary["mean_score"],
                "mean_area": run_summary["mean_area"],
                "predictions_csv": str(predictions_csv),
                "overlay_video": str(overlay_video),
            }
        )
        print(
            f"[INFO] prompt_mode={prompt_mode}: predicted_frames={run_summary['predicted_frames']} "
            f"coverage={run_summary['coverage']:.3f}"
        )

    comparison_csv = run_dir / "prompt_mode_comparison.csv"
    csv_write_dicts(comparison_csv, comparison_rows)
    print(f"[INFO] Comparison written to {comparison_csv}")


if __name__ == "__main__":
    main()
