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
    csv_write_dicts,
    ensure_dir,
    find_anchor_seed,
    infer_fps,
    list_frame_paths,
    parse_int_csv,
    render_track_video,
    shift_predictions_to_global,
    stage_frames_for_edgetam,
    str2bool,
    summarize_predictions,
    write_predictions_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "edgetam_sparse_reseed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a clip with EdgeTAM using sparse RF-DETR reseeds."
    )
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--intervals", type=str, default="15,30,60")
    parser.add_argument("--prompt_mode", choices=PROMPT_MODES, default="box")
    parser.add_argument("--anchor_search_radius", type=int, default=0)

    parser.add_argument("--rfdetr_model_path", type=Path, default=DEFAULT_RFDETR_MODEL_PATH)
    parser.add_argument("--rfdetr_resolution", type=int, default=DEFAULT_RFDETR_RESOLUTION)
    parser.add_argument("--rfdetr_confidence", type=float, default=DEFAULT_RFDETR_CONFIDENCE)
    parser.add_argument("--optimize_for_inference", type=str2bool, default=True)

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
        help="Defaults to false here to avoid repeated background full-clip loading.",
    )
    parser.add_argument("--fps", type=float, default=0.0)
    return parser.parse_args()


def default_run_name(frames_dir: Path, prompt_mode: str) -> str:
    return f"{frames_dir.name}__sparse_reseed__{prompt_mode}"


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
    intervals = parse_int_csv(args.intervals)
    if not intervals:
        raise ValueError("No intervals provided")

    detector = build_rfdetr_detector(
        model_path=args.rfdetr_model_path,
        confidence_threshold=args.rfdetr_confidence,
        resolution=int(args.rfdetr_resolution) if int(args.rfdetr_resolution) > 0 else None,
        optimize_for_inference=bool(args.optimize_for_inference),
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

    comparison_rows = []
    for interval in intervals:
        interval_dir = ensure_dir(run_dir / f"interval_{int(interval):03d}")
        predictions_by_index = {}
        anchor_rows = []
        anchor_indices = list(range(0, len(frame_paths), max(1, int(interval))))

        for anchor_index in anchor_indices:
            slot_start = anchor_index
            slot_end = min(len(frame_paths) - 1, anchor_index + int(interval) - 1)
            seed = find_anchor_seed(
                detector,
                frame_paths,
                anchor_index=anchor_index,
                search_radius=max(0, int(args.anchor_search_radius)),
            )
            if seed is None:
                anchor_rows.append(
                    {
                        "anchor_frame_index": anchor_index + 1,
                        "slot_start": slot_start + 1,
                        "slot_end": slot_end + 1,
                        "status": "miss",
                        "seed_frame_index": "",
                        "seed_frame_name": "",
                        "seed_score": "",
                    }
                )
                continue

            seed_shape = cv2.imread(str(frame_paths[seed.frame_index])).shape[:2]
            prompt_payload = build_prompt_payload(
                args.prompt_mode,
                seed.bbox_xyxy,
                seed_shape,
            )
            prompt_payload["mode"] = args.prompt_mode
            slot_frame_paths = list(frame_paths[slot_start : slot_end + 1])
            local_seed = type(seed)(
                frame_index=seed.frame_index - slot_start,
                frame_name=seed.frame_name,
                frame_path=seed.frame_path,
                bbox_xyxy=list(seed.bbox_xyxy),
                score=float(seed.score),
            )
            slot_staged_dir = stage_frames_for_edgetam(
                slot_frame_paths,
                interval_dir / "_staged_slots" / f"{slot_start:06d}_{slot_end:06d}",
            )
            local_predictions = tracker.track_bidirectional_local(
                staged_frames_dir=slot_staged_dir,
                frame_paths=slot_frame_paths,
                seed=local_seed,
                prompt_payload=prompt_payload,
                source=f"edgetam_sparse_reseed_{interval}",
            )
            shifted_predictions = shift_predictions_to_global(local_predictions, slot_start)
            predictions_by_index.update(shifted_predictions)

            anchor_rows.append(
                {
                    "anchor_frame_index": anchor_index + 1,
                    "slot_start": slot_start + 1,
                    "slot_end": slot_end + 1,
                    "status": "seeded",
                    "seed_frame_index": seed.frame_index + 1,
                    "seed_frame_name": seed.frame_name,
                    "seed_score": seed.score,
                }
            )

        predictions_csv = interval_dir / "predictions.csv"
        anchors_csv = interval_dir / "anchor_summary.csv"
        overlay_video = interval_dir / "overlay.mp4"
        summary_path = interval_dir / "summary.json"

        write_predictions_csv(
            path=predictions_csv,
            frame_paths=frame_paths,
            predictions_by_index=predictions_by_index,
        )
        csv_write_dicts(anchors_csv, anchor_rows)
        render_track_video(
            frame_paths=frame_paths,
            predictions_by_index=predictions_by_index,
            output_video=overlay_video,
            fps=fps,
            run_label=f"{run_name} | every {interval}",
        )

        run_summary = summarize_predictions(
            predictions_by_index=predictions_by_index,
            total_frames=len(frame_paths),
        )
        payload = {
            "run_name": run_name,
            "interval": int(interval),
            "frames_dir": str(args.frames_dir),
            "fps": fps,
            "prompt_mode": args.prompt_mode,
            "anchor_search_radius": int(args.anchor_search_radius),
            "summary": run_summary,
            "outputs": {
                "predictions_csv": str(predictions_csv),
                "anchor_summary_csv": str(anchors_csv),
                "overlay_video": str(overlay_video),
            },
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        comparison_rows.append(
            {
                "interval": int(interval),
                "predicted_frames": run_summary["predicted_frames"],
                "coverage": run_summary["coverage"],
                "mean_score": run_summary["mean_score"],
                "mean_area": run_summary["mean_area"],
                "predictions_csv": str(predictions_csv),
                "overlay_video": str(overlay_video),
            }
        )
        print(
            f"[INFO] interval={interval}: predicted_frames={run_summary['predicted_frames']} "
            f"coverage={run_summary['coverage']:.3f}"
        )

    comparison_csv = run_dir / "interval_comparison.csv"
    csv_write_dicts(comparison_csv, comparison_rows)
    print(f"[INFO] Comparison written to {comparison_csv}")


if __name__ == "__main__":
    main()
