#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EDGETAM_ROOT = REPO_ROOT / "EdgeTAM"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EDGETAM_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGETAM_ROOT))

from _eval_rfdetr_ball_only_common import RFDetrBallOnlyDetector  # noqa: E402
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROMPT_MODES = ("box", "point", "box_point", "box_point_neg")

DEFAULT_RFDETR_MODEL_PATH = REPO_ROOT / "checkpoints" / "ball_1120.pth"
DEFAULT_RFDETR_RESOLUTION = 1120
DEFAULT_RFDETR_CONFIDENCE = 0.01
DEFAULT_EDGETAM_CHECKPOINT = EDGETAM_ROOT / "checkpoints" / "edgetam.pt"
DEFAULT_EDGETAM_CONFIG = "configs/edgetam.yaml"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_USE_AMP = True
DEFAULT_MASK_THRESHOLD = 0.0
DEFAULT_FPS = 30.0
DEFAULT_OFFLOAD_VIDEO_TO_CPU = True
DEFAULT_OFFLOAD_STATE_TO_CPU = True
DEFAULT_ASYNC_LOADING_FRAMES = True
DEFAULT_ZERO_PAD = 6


@dataclass
class SeedDetection:
    frame_index: int
    frame_name: str
    frame_path: str
    bbox_xyxy: List[float]
    score: float


@dataclass
class TrackPrediction:
    frame_index: int
    frame_name: str
    frame_path: str
    visible: bool
    x1: Optional[float]
    y1: Optional[float]
    x2: Optional[float]
    y2: Optional[float]
    center_x: Optional[float]
    center_y: Optional[float]
    area: float
    score: float
    source: str
    prompt_mode: str
    direction: str
    anchor_frame_index: Optional[int]
    anchor_frame_name: str
    anchor_score: Optional[float]
    is_anchor: bool


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {value!r}")


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


def csv_write_dicts(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_int_csv(text: str) -> List[int]:
    values: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def parse_mode_csv(text: str) -> List[str]:
    modes: List[str] = []
    for token in str(text).split(","):
        mode = token.strip()
        if not mode:
            continue
        if mode not in PROMPT_MODES:
            raise ValueError(f"Unknown prompt mode: {mode}. Expected one of {PROMPT_MODES}")
        modes.append(mode)
    return modes


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
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    fps = payload.get("source_fps")
    if fps in ("", None):
        return None
    try:
        return float(fps)
    except (TypeError, ValueError):
        return None


def infer_fps(frames_dir: Path) -> float:
    candidates = [
        frames_dir / "seqinfo.ini",
        frames_dir.parent / "seqinfo.ini",
        frames_dir / "extraction_summary.json",
        frames_dir.parent / "extraction_summary.json",
    ]
    for candidate in candidates:
        if candidate.name == "seqinfo.ini":
            fps = parse_seqinfo_fps(candidate)
        else:
            fps = parse_extraction_summary_fps(candidate)
        if fps is not None and fps > 0:
            return fps
    return DEFAULT_FPS


def resolve_frames_root(frames_dir: Path) -> Path:
    frames_dir = frames_dir.expanduser().resolve()
    mot_frames = frames_dir / "img1"
    if mot_frames.is_dir():
        return mot_frames
    return frames_dir


def _frame_sort_key(path: Path) -> tuple[int, str]:
    matches = re.findall(r"\d+", path.stem)
    if matches:
        return int(matches[-1]), path.name
    return 10**18, path.name


def list_frame_paths(
    frames_dir: Path,
    start_frame: int = 0,
    max_frames: int = 0,
) -> List[Path]:
    resolved_frames_dir = resolve_frames_root(frames_dir)
    if not resolved_frames_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {resolved_frames_dir}")
    frame_paths = sorted(
        [
            path
            for path in resolved_frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=_frame_sort_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frame images found in {resolved_frames_dir}")
    frame_paths = frame_paths[start_frame:]
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        raise RuntimeError("No frames remain after applying start_frame/max_frames")
    return frame_paths


def stage_frames_for_edgetam(
    frame_paths: Sequence[Path],
    staging_dir: Path,
    zero_pad: int = DEFAULT_ZERO_PAD,
) -> Path:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    ensure_dir(staging_dir)

    for index, source_path in enumerate(frame_paths):
        staged_path = staging_dir / f"{index:0{zero_pad}d}.jpg"
        suffix = source_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            try:
                staged_path.symlink_to(source_path.resolve())
            except OSError:
                shutil.copy2(source_path, staged_path)
            continue

        image = cv2.imread(str(source_path))
        if image is None:
            raise RuntimeError(f"Could not read frame for staging: {source_path}")
        ok = cv2.imwrite(str(staged_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Could not convert frame to staged JPEG: {source_path}")

    return staging_dir


def load_bgr_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return image


def build_rfdetr_detector(
    model_path: Path,
    confidence_threshold: float,
    resolution: Optional[int],
    optimize_for_inference: bool,
) -> RFDetrBallOnlyDetector:
    return RFDetrBallOnlyDetector(
        model_path=model_path,
        ball_class_id=0,
        confidence_threshold=confidence_threshold,
        resolution=resolution,
        optimize_for_inference=optimize_for_inference,
    )


def best_detection_on_frame(
    detector: RFDetrBallOnlyDetector,
    frame_index: int,
    frame_path: Path,
) -> Optional[SeedDetection]:
    image = load_bgr_image(frame_path)
    detections = detector.predict_ball_detections(image)
    if not detections:
        return None
    best = max(detections, key=lambda item: float(item["score"]))
    return SeedDetection(
        frame_index=int(frame_index),
        frame_name=frame_path.name,
        frame_path=str(frame_path),
        bbox_xyxy=[float(value) for value in np.asarray(best["xyxy"], dtype=np.float32).tolist()],
        score=float(best["score"]),
    )


def scan_best_seed(
    detector: RFDetrBallOnlyDetector,
    frame_paths: Sequence[Path],
    *,
    seed_frame_index: int,
    seed_scan_stride: int,
) -> SeedDetection:
    if seed_frame_index >= 0:
        if seed_frame_index >= len(frame_paths):
            raise IndexError(
                f"seed_frame_index={seed_frame_index} is out of range for {len(frame_paths)} frames"
            )
        seed = best_detection_on_frame(detector, seed_frame_index, frame_paths[seed_frame_index])
        if seed is None:
            raise RuntimeError(f"No RF-DETR detection found on explicit seed frame {seed_frame_index}")
        return seed

    stride = max(1, int(seed_scan_stride))
    best_seed: Optional[SeedDetection] = None
    for frame_index in range(0, len(frame_paths), stride):
        candidate = best_detection_on_frame(detector, frame_index, frame_paths[frame_index])
        if candidate is None:
            continue
        if best_seed is None or candidate.score > best_seed.score:
            best_seed = candidate
    if best_seed is None:
        raise RuntimeError("RF-DETR did not find any seed candidate while scanning the clip")
    return best_seed


def find_anchor_seed(
    detector: RFDetrBallOnlyDetector,
    frame_paths: Sequence[Path],
    *,
    anchor_index: int,
    search_radius: int,
) -> Optional[SeedDetection]:
    best_seed: Optional[SeedDetection] = None
    best_distance: Optional[int] = None
    start_index = max(0, anchor_index - max(0, int(search_radius)))
    end_index = min(len(frame_paths) - 1, anchor_index + max(0, int(search_radius)))
    for frame_index in range(start_index, end_index + 1):
        candidate = best_detection_on_frame(detector, frame_index, frame_paths[frame_index])
        if candidate is None:
            continue
        distance = abs(frame_index - anchor_index)
        if best_seed is None:
            best_seed = candidate
            best_distance = distance
            continue
        if candidate.score > best_seed.score:
            best_seed = candidate
            best_distance = distance
            continue
        if candidate.score == best_seed.score and best_distance is not None and distance < best_distance:
            best_seed = candidate
            best_distance = distance
    return best_seed


def clip_xyxy(box_xyxy: Sequence[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(0.0, min(x2, width - 1))
    y2 = max(0.0, min(y2, height - 1))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def box_center_xy(box_xyxy: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def build_prompt_payload(
    prompt_mode: str,
    box_xyxy: Sequence[float],
    image_shape: Sequence[int],
    *,
    negative_margin_ratio: float = 0.75,
    negative_margin_min: float = 6.0,
) -> dict:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt_mode={prompt_mode!r}")

    height, width = int(image_shape[0]), int(image_shape[1])
    clipped_box = clip_xyxy(box_xyxy, width=width, height=height)
    center = box_center_xy(clipped_box)

    payload = {
        "box": None,
        "points": None,
        "labels": None,
        "prompt_preview": [],
    }

    if prompt_mode in {"box", "box_point", "box_point_neg"}:
        payload["box"] = clipped_box.copy()

    if prompt_mode in {"point", "box_point", "box_point_neg"}:
        points = [center.copy()]
        labels = [1]

        if prompt_mode == "box_point_neg":
            box_w = max(1.0, float(clipped_box[2] - clipped_box[0]))
            box_h = max(1.0, float(clipped_box[3] - clipped_box[1]))
            margin = max(negative_margin_min, negative_margin_ratio * max(box_w, box_h))
            cx, cy = float(center[0]), float(center[1])
            x1, y1, x2, y2 = [float(value) for value in clipped_box]
            negative_points = [
                [x1 - margin, cy],
                [x2 + margin, cy],
                [cx, y1 - margin],
                [cx, y2 + margin],
            ]
            for point in negative_points:
                clipped_point = np.asarray(
                    [
                        max(0.0, min(float(point[0]), width - 1)),
                        max(0.0, min(float(point[1]), height - 1)),
                    ],
                    dtype=np.float32,
                )
                points.append(clipped_point)
                labels.append(0)

        payload["points"] = np.asarray(points, dtype=np.float32)
        payload["labels"] = np.asarray(labels, dtype=np.int32)

    prompt_preview: List[dict] = []
    if payload["box"] is not None:
        prompt_preview.append(
            {
                "type": "box",
                "xyxy": [float(value) for value in payload["box"].tolist()],
            }
        )
    if payload["points"] is not None and payload["labels"] is not None:
        for point, label in zip(payload["points"], payload["labels"]):
            prompt_preview.append(
                {
                    "type": "point",
                    "xy": [float(point[0]), float(point[1])],
                    "label": int(label),
                }
            )
    payload["prompt_preview"] = prompt_preview
    return payload


def mask_logits_to_score(mask_logits: torch.Tensor, threshold: float) -> float:
    logits = mask_logits.squeeze().detach()
    if logits.numel() == 0:
        return 0.0
    positive_logits = logits[logits > threshold]
    if positive_logits.numel() == 0:
        return 0.0
    mean_logit = float(positive_logits.float().mean().item())
    return float(1.0 / (1.0 + np.exp(-mean_logit)))


def mask_logits_to_prediction(
    *,
    mask_logits: torch.Tensor,
    frame_index: int,
    frame_path: Path,
    prompt_mode: str,
    source: str,
    direction: str,
    anchor_frame_index: int,
    anchor_frame_name: str,
    anchor_score: float,
    threshold: float,
    is_anchor: bool,
) -> Optional[TrackPrediction]:
    mask = (mask_logits.squeeze() > threshold).detach()
    if mask.ndim != 2:
        return None
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return None

    ys = coords[:, 0].float()
    xs = coords[:, 1].float()
    x1 = float(xs.min().item())
    y1 = float(ys.min().item())
    x2 = float(xs.max().item() + 1.0)
    y2 = float(ys.max().item() + 1.0)
    center_x = float(xs.mean().item())
    center_y = float(ys.mean().item())
    area = float(coords.shape[0])

    return TrackPrediction(
        frame_index=int(frame_index) + 1,
        frame_name=frame_path.name,
        frame_path=str(frame_path),
        visible=True,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x=center_x,
        center_y=center_y,
        area=area,
        score=mask_logits_to_score(mask_logits, threshold=threshold),
        source=source,
        prompt_mode=prompt_mode,
        direction=direction,
        anchor_frame_index=int(anchor_frame_index) + 1,
        anchor_frame_name=anchor_frame_name,
        anchor_score=float(anchor_score),
        is_anchor=bool(is_anchor),
    )


def prediction_to_seed(prediction: TrackPrediction) -> SeedDetection:
    if not prediction.visible:
        raise ValueError("Cannot convert an invisible prediction to a seed")
    if None in (prediction.x1, prediction.y1, prediction.x2, prediction.y2):
        raise ValueError("Prediction does not contain a full bounding box")
    return SeedDetection(
        frame_index=int(prediction.frame_index) - 1,
        frame_name=prediction.frame_name,
        frame_path=prediction.frame_path,
        bbox_xyxy=[
            float(prediction.x1),
            float(prediction.y1),
            float(prediction.x2),
            float(prediction.y2),
        ],
        score=float(prediction.score),
    )


class EdgeTAMVideoTracker:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        config_name: str,
        device: str,
        use_amp: bool,
        mask_threshold: float,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
        async_loading_frames: bool,
    ) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"EdgeTAM checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.mask_threshold = float(mask_threshold)
        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self.offload_state_to_cpu = bool(offload_state_to_cpu)
        self.async_loading_frames = bool(async_loading_frames)
        self.predictor = build_sam2_video_predictor(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=str(self.device),
        )

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    def track_one_direction(
        self,
        *,
        staged_frames_dir: Path,
        frame_paths: Sequence[Path],
        seed: SeedDetection,
        prompt_payload: dict,
        reverse: bool,
        max_track_offset: int,
        source: str,
    ) -> Dict[int, TrackPrediction]:
        state = self.predictor.init_state(
            video_path=str(staged_frames_dir),
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=self.async_loading_frames,
        )

        predictions: Dict[int, TrackPrediction] = {}
        prompt_kwargs = {}
        if prompt_payload["points"] is not None:
            prompt_kwargs["points"] = prompt_payload["points"]
            prompt_kwargs["labels"] = prompt_payload["labels"]
        if prompt_payload["box"] is not None:
            prompt_kwargs["box"] = prompt_payload["box"]

        with torch.inference_mode():
            with self._autocast_context():
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=seed.frame_index,
                    obj_id=1,
                    **prompt_kwargs,
                )
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                    state,
                    start_frame_idx=seed.frame_index,
                    max_frame_num_to_track=max(0, int(max_track_offset)),
                    reverse=bool(reverse),
                ):
                    for obj_offset, obj_id in enumerate(out_obj_ids):
                        if int(obj_id) != 1:
                            continue
                        prediction = mask_logits_to_prediction(
                            mask_logits=out_mask_logits[obj_offset],
                            frame_index=int(out_frame_idx),
                            frame_path=frame_paths[int(out_frame_idx)],
                            prompt_mode=str(prompt_payload.get("mode", "unknown")),
                            source=source,
                            direction="reverse" if reverse else "forward",
                            anchor_frame_index=seed.frame_index,
                            anchor_frame_name=seed.frame_name,
                            anchor_score=seed.score,
                            threshold=self.mask_threshold,
                            is_anchor=int(out_frame_idx) == seed.frame_index,
                        )
                        if prediction is not None:
                            predictions[int(out_frame_idx)] = prediction

        self.predictor.reset_state(state)
        return predictions

    def track_bidirectional_local(
        self,
        *,
        staged_frames_dir: Path,
        frame_paths: Sequence[Path],
        seed: SeedDetection,
        prompt_payload: dict,
        source: str,
    ) -> Dict[int, TrackPrediction]:
        forward_predictions = self.track_one_direction(
            staged_frames_dir=staged_frames_dir,
            frame_paths=frame_paths,
            seed=seed,
            prompt_payload=prompt_payload,
            reverse=False,
            max_track_offset=max(0, len(frame_paths) - 1 - seed.frame_index),
            source=source,
        )
        reverse_predictions: Dict[int, TrackPrediction] = {}
        if seed.frame_index > 0:
            reverse_predictions = self.track_one_direction(
                staged_frames_dir=staged_frames_dir,
                frame_paths=frame_paths,
                seed=seed,
                prompt_payload=prompt_payload,
                reverse=True,
                max_track_offset=seed.frame_index,
                source=source,
            )
        merged = dict(reverse_predictions)
        merged.update(forward_predictions)
        return merged

    def track_bidirectional(
        self,
        *,
        staged_frames_dir: Path,
        frame_paths: Sequence[Path],
        seed: SeedDetection,
        prompt_payload: dict,
        source: str,
    ) -> Dict[int, TrackPrediction]:
        return self.track_bidirectional_local(
            staged_frames_dir=staged_frames_dir,
            frame_paths=frame_paths,
            seed=seed,
            prompt_payload=prompt_payload,
            source=source,
        )


def shift_predictions_to_global(
    local_predictions: Dict[int, TrackPrediction],
    global_start_index: int,
) -> Dict[int, TrackPrediction]:
    shifted: Dict[int, TrackPrediction] = {}
    for local_index, prediction in local_predictions.items():
        global_index = int(global_start_index) + int(local_index)
        shifted[global_index] = TrackPrediction(
            frame_index=global_index + 1,
            frame_name=prediction.frame_name,
            frame_path=prediction.frame_path,
            visible=prediction.visible,
            x1=prediction.x1,
            y1=prediction.y1,
            x2=prediction.x2,
            y2=prediction.y2,
            center_x=prediction.center_x,
            center_y=prediction.center_y,
            area=prediction.area,
            score=prediction.score,
            source=prediction.source,
            prompt_mode=prediction.prompt_mode,
            direction=prediction.direction,
            anchor_frame_index=(
                None
                if prediction.anchor_frame_index in (None, "")
                else int(global_start_index) + int(prediction.anchor_frame_index)
            ),
            anchor_frame_name=prediction.anchor_frame_name,
            anchor_score=prediction.anchor_score,
            is_anchor=prediction.is_anchor,
        )
    return shifted


def _merge_predictions_keep_best(
    target: Dict[int, TrackPrediction],
    incoming: Dict[int, TrackPrediction],
) -> None:
    for frame_index, prediction in incoming.items():
        current = target.get(frame_index)
        if current is None or float(prediction.score) >= float(current.score):
            target[frame_index] = prediction


def track_full_clip_windowed(
    *,
    tracker: EdgeTAMVideoTracker,
    frame_paths: Sequence[Path],
    seed: SeedDetection,
    prompt_payload: dict,
    run_dir: Path,
    source: str,
    max_window_frames: int,
    async_loading_frames: bool,
) -> Dict[int, TrackPrediction]:
    if max_window_frames <= 1:
        raise ValueError("max_window_frames must be > 1")

    global_predictions: Dict[int, TrackPrediction] = {}

    current_seed = seed
    while True:
        window_start = current_seed.frame_index
        window_end = min(len(frame_paths) - 1, window_start + max_window_frames - 1)
        window_frame_paths = list(frame_paths[window_start : window_end + 1])
        window_seed = SeedDetection(
            frame_index=0,
            frame_name=current_seed.frame_name,
            frame_path=current_seed.frame_path,
            bbox_xyxy=list(current_seed.bbox_xyxy),
            score=float(current_seed.score),
        )
        window_dir = stage_frames_for_edgetam(
            window_frame_paths,
            run_dir / "_edgetam_windows" / f"forward_{window_start:06d}_{window_end:06d}",
        )
        previous_async = tracker.async_loading_frames
        tracker.async_loading_frames = bool(async_loading_frames)
        local_predictions = tracker.track_one_direction(
            staged_frames_dir=window_dir,
            frame_paths=window_frame_paths,
            seed=window_seed,
            prompt_payload=prompt_payload,
            reverse=False,
            max_track_offset=len(window_frame_paths) - 1,
            source=source,
        )
        tracker.async_loading_frames = previous_async
        shifted_predictions = shift_predictions_to_global(local_predictions, window_start)
        _merge_predictions_keep_best(global_predictions, shifted_predictions)

        if window_end >= len(frame_paths) - 1:
            break

        predicted_indices = sorted(index for index in shifted_predictions if index > current_seed.frame_index)
        if not predicted_indices:
            break
        continuation_index = predicted_indices[-1]
        if continuation_index <= current_seed.frame_index:
            break
        current_seed = prediction_to_seed(global_predictions[continuation_index])
        if continuation_index >= len(frame_paths) - 1:
            break

    current_seed = seed
    while current_seed.frame_index > 0:
        window_end = current_seed.frame_index
        window_start = max(0, window_end - max_window_frames + 1)
        window_frame_paths = list(frame_paths[window_start : window_end + 1])
        window_seed = SeedDetection(
            frame_index=current_seed.frame_index - window_start,
            frame_name=current_seed.frame_name,
            frame_path=current_seed.frame_path,
            bbox_xyxy=list(current_seed.bbox_xyxy),
            score=float(current_seed.score),
        )
        window_dir = stage_frames_for_edgetam(
            window_frame_paths,
            run_dir / "_edgetam_windows" / f"reverse_{window_start:06d}_{window_end:06d}",
        )
        previous_async = tracker.async_loading_frames
        tracker.async_loading_frames = bool(async_loading_frames)
        local_predictions = tracker.track_one_direction(
            staged_frames_dir=window_dir,
            frame_paths=window_frame_paths,
            seed=window_seed,
            prompt_payload=prompt_payload,
            reverse=True,
            max_track_offset=window_seed.frame_index,
            source=source,
        )
        tracker.async_loading_frames = previous_async
        shifted_predictions = shift_predictions_to_global(local_predictions, window_start)
        _merge_predictions_keep_best(global_predictions, shifted_predictions)

        if window_start <= 0:
            break

        predicted_indices = sorted(index for index in shifted_predictions if index < current_seed.frame_index)
        if not predicted_indices:
            break
        continuation_index = predicted_indices[0]
        if continuation_index >= current_seed.frame_index:
            break
        current_seed = prediction_to_seed(global_predictions[continuation_index])
        if continuation_index <= 0:
            break

    return global_predictions


def render_track_video(
    *,
    frame_paths: Sequence[Path],
    predictions_by_index: Dict[int, TrackPrediction],
    output_video: Path,
    fps: float,
    run_label: str,
) -> None:
    first_frame = load_bgr_image(frame_paths[0])
    height, width = first_frame.shape[:2]
    ensure_dir(output_video.parent)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video}")

    try:
        for frame_index, frame_path in enumerate(frame_paths):
            image = load_bgr_image(frame_path)
            prediction = predictions_by_index.get(frame_index)
            if prediction is not None and prediction.visible:
                x1 = int(round(float(prediction.x1 or 0.0)))
                y1 = int(round(float(prediction.y1 or 0.0)))
                x2 = int(round(float(prediction.x2 or 0.0)))
                y2 = int(round(float(prediction.y2 or 0.0)))
                color = (0, 255, 255) if prediction.is_anchor else (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                if prediction.center_x is not None and prediction.center_y is not None:
                    cv2.circle(
                        image,
                        (int(round(prediction.center_x)), int(round(prediction.center_y))),
                        5,
                        color,
                        -1,
                    )
                label = (
                    f"{prediction.prompt_mode} {prediction.direction} "
                    f"{prediction.score:.3f}"
                )
                if prediction.is_anchor:
                    label = f"ANCHOR {label}"
                cv2.putText(
                    image,
                    label,
                    (max(0, x1), max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    image,
                    "NO TRACK",
                    (32, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                image,
                f"{run_label} | frame {frame_index + 1}/{len(frame_paths)}",
                (32, height - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()


def write_predictions_csv(
    *,
    path: Path,
    frame_paths: Sequence[Path],
    predictions_by_index: Dict[int, TrackPrediction],
) -> None:
    rows: List[dict] = []
    for frame_index, frame_path in enumerate(frame_paths):
        prediction = predictions_by_index.get(frame_index)
        if prediction is None:
            rows.append(
                {
                    "frame_index": frame_index + 1,
                    "frame_name": frame_path.name,
                    "frame_path": str(frame_path),
                    "visible": False,
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "center_x": "",
                    "center_y": "",
                    "area": 0.0,
                    "score": 0.0,
                    "source": "",
                    "prompt_mode": "",
                    "direction": "",
                    "anchor_frame_index": "",
                    "anchor_frame_name": "",
                    "anchor_score": "",
                    "is_anchor": False,
                }
            )
            continue
        rows.append(asdict(prediction))
    csv_write_dicts(path, rows)


def summarize_predictions(
    *,
    predictions_by_index: Dict[int, TrackPrediction],
    total_frames: int,
) -> dict:
    visible_predictions = [item for item in predictions_by_index.values() if item.visible]
    scores = [float(item.score) for item in visible_predictions]
    areas = [float(item.area) for item in visible_predictions]
    return {
        "predicted_frames": len(visible_predictions),
        "coverage": float(len(visible_predictions) / total_frames) if total_frames > 0 else 0.0,
        "mean_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "mean_area": float(sum(areas) / len(areas)) if areas else 0.0,
    }
