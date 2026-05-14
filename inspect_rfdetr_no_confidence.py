#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from benchmark_dataset import default_annotation_path, resolve_sequences  # noqa: E402


DEFAULT_SCORE_FLOOR = 0.5
DEFAULT_MAX_FRAMES_PER_SEQ = 0
DEFAULT_SEQ_START = 0
DEFAULT_SEQ_END = 999
DEFAULT_SEQ_LIST = ""
DEFAULT_PRINT_LIMIT = 0
DEFAULT_OPTIMIZE_FOR_INFERENCE = True


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {value!r}")


def resolve_dataset_layout(data_root: Path) -> Tuple[Path, Optional[Path]]:
    flat_annotations = default_annotation_path(data_root)
    if flat_annotations is not None:
        return data_root, flat_annotations

    nested_images_dir = data_root / "images"
    nested_annotations = nested_images_dir / "_annotations.coco.json"
    if nested_annotations.exists():
        return nested_images_dir, nested_annotations

    return data_root, None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_path: Path
    resolution: Optional[int]


class RFDetrInspector:
    def __init__(self, spec: ModelSpec, optimize_for_inference: bool) -> None:
        if not spec.model_path.exists():
            raise FileNotFoundError(f"RF-DETR checkpoint not found: {spec.model_path}")

        from rfdetr import RFDETRMedium

        kwargs = {"pretrain_weights": str(spec.model_path)}
        if spec.resolution is not None and int(spec.resolution) > 0:
            kwargs["resolution"] = int(spec.resolution)

        self.spec = spec
        self.model = RFDETRMedium(**kwargs)
        if optimize_for_inference:
            try:
                self.model.optimize_for_inference()
                print(f"[INFO] Optimized {spec.name} for inference")
            except Exception as exc:
                print(f"[WARN] Could not optimize {spec.name}: {exc}")

    def predict_ball_detections(self, image_bgr: np.ndarray) -> List[dict]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Intentionally omit the confidence kwarg to inspect the model/API default behavior.
        detections = self.model.predict(image_rgb)
        if detections is None or len(detections) == 0:
            return []

        xyxy = np.asarray(detections.xyxy)
        confidence = np.asarray(detections.confidence).reshape(-1)
        class_ids = np.asarray(detections.class_id).reshape(-1)

        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            return []

        num_boxes = xyxy.shape[0]
        if confidence.size == 1 and num_boxes > 1:
            confidence = np.repeat(confidence, num_boxes)
        if class_ids.size == 1 and num_boxes > 1:
            class_ids = np.repeat(class_ids, num_boxes)

        num_items = min(num_boxes, confidence.size, class_ids.size)
        if num_items <= 0:
            return []

        output: List[dict] = []
        for index in range(num_items):
            if int(class_ids[index]) != 0:
                continue
            output.append(
                {
                    "xyxy": [float(value) for value in np.asarray(xyxy[index], dtype=np.float32).tolist()],
                    "score": float(confidence[index]),
                }
            )
        return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect whether RF-DETR returns any ball detections below a score floor when no confidence kwarg is passed."
    )
    parser.add_argument("--score_floor", type=float, default=DEFAULT_SCORE_FLOOR)
    parser.add_argument("--max_frames_per_seq", type=int, default=DEFAULT_MAX_FRAMES_PER_SEQ)
    parser.add_argument("--seq_start", type=int, default=DEFAULT_SEQ_START)
    parser.add_argument("--seq_end", type=int, default=DEFAULT_SEQ_END)
    parser.add_argument("--seq_list", type=str, default=DEFAULT_SEQ_LIST)
    parser.add_argument("--print_limit", type=int, default=DEFAULT_PRINT_LIMIT)
    parser.add_argument(
        "--optimize_for_inference",
        type=str2bool,
        default=DEFAULT_OPTIMIZE_FOR_INFERENCE,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    datasets = [
        ("test", REPO_ROOT / "test"),
        ("samy_test", REPO_ROOT / "samy_combined_ball_dataset" / "test"),
    ]
    models = [
        ModelSpec("rfdetr_ball", REPO_ROOT / "checkpoints" / "ball.pth", None),
        ModelSpec("rfdetr_ball_1120", REPO_ROOT / "checkpoints" / "ball_1120.pth", 1120),
    ]

    print("[INFO] This script calls RF-DETR as model.predict(image_rgb) with no confidence kwarg.")
    print(f"[INFO] Reporting ball detections with score < {args.score_floor}")

    inspectors: Dict[str, RFDetrInspector] = {}
    for spec in models:
        inspectors[spec.name] = RFDetrInspector(spec, optimize_for_inference=args.optimize_for_inference)

    for dataset_name, requested_root in datasets:
        resolved_root, annotations_path = resolve_dataset_layout(requested_root.resolve())
        sequences = resolve_sequences(
            data_root=resolved_root,
            seq_start=args.seq_start,
            seq_end=args.seq_end,
            seq_list=args.seq_list,
            max_frames_per_seq=args.max_frames_per_seq,
            annotations_path=annotations_path,
        )

        print()
        print(f"[DATASET] {dataset_name}")
        print(f"[INFO] Requested root: {requested_root.resolve()}")
        if resolved_root != requested_root.resolve():
            print(f"[INFO] Resolved image root: {resolved_root}")
        if annotations_path is not None:
            print(f"[INFO] Annotation path: {annotations_path}")
        print(f"[INFO] Sequences: {', '.join(sequence.name for sequence in sequences)}")

        for model_name, inspector in inspectors.items():
            print()
            print(f"[MODEL] {model_name}")
            total_ball_detections = 0
            below_floor_detections = 0
            min_score_seen: Optional[float] = None
            printed = 0

            for sequence in sequences:
                for frame_index, image_path in enumerate(sequence.image_paths, start=1):
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise RuntimeError(f"Could not read frame {image_path}")

                    detections = inspector.predict_ball_detections(image)
                    total_ball_detections += len(detections)

                    for detection in detections:
                        score = float(detection["score"])
                        if min_score_seen is None or score < min_score_seen:
                            min_score_seen = score

                        if score >= args.score_floor:
                            continue

                        below_floor_detections += 1
                        if args.print_limit > 0 and printed >= args.print_limit:
                            continue

                        printed += 1
                        print(
                            "[FOUND] "
                            f"dataset={dataset_name} model={model_name} sequence={sequence.name} "
                            f"frame_index={frame_index} file={sequence.file_names_by_frame[frame_index]} "
                            f"score={score:.6f} bbox_xyxy={detection['xyxy']}"
                        )

            min_score_text = f"{min_score_seen:.6f}" if min_score_seen is not None else "n/a"
            print(
                "[SUMMARY] "
                f"dataset={dataset_name} model={model_name} total_ball_detections={total_ball_detections} "
                f"below_floor={below_floor_detections} min_score_seen={min_score_text}"
            )
            if below_floor_detections == 0:
                print(
                    f"[OK] No ball detections below {args.score_floor} were returned by {model_name} on {dataset_name}"
                )


if __name__ == "__main__":
    main()
