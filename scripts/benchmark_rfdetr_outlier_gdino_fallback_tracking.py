#!/usr/bin/env python3
"""
RF-DETR ball-only + outlier rejection + GroundingDINO fallback benchmark runner.

This is the third experiment:
1. Only the RF-DETR ball model is used as the primary detector.
2. No Kalman filter is used.
3. No player model is used.
4. Outlier rejection is the only tracking logic.
5. GroundingDINO runs as the fallback whenever RF-DETR gives no accepted ball.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRMedium


REPO_ROOT = Path(__file__).resolve().parents[1]

# Dataset / run defaults
DEFAULT_DATA_ROOT = Path("~/SoccerNet_tracking/tracking-2023/train").expanduser()
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "rfdetr_outlier_gdino_fallback_tracking"
DEFAULT_RUN_NAME = "rfdetr_outlier_gdino_fallback_v1"
DEFAULT_SEQ_START = 60
DEFAULT_SEQ_END = 70
DEFAULT_SEQ_LIST = ""
DEFAULT_MAX_FRAMES_PER_SEQ = 0

# RF-DETR defaults
DEFAULT_BALL_MODEL_PATH = REPO_ROOT / "checkpoints" / "ball.pth"
DEFAULT_BALL_CONFIDENCE = 0.7
DEFAULT_BALL_CLASS_ID = 0
DEFAULT_ENABLE_RFDETR_OPTIMIZE = True
DEFAULT_MAX_ACTIVE_GAP = 10

# Previous outlier logic defaults
DEFAULT_POSITION_THRESHOLD = 50.0
DEFAULT_VELOCITY_THRESHOLD = 100.0
DEFAULT_OUTLIER_HISTORY_FRAMES = 3
DEFAULT_OUTLIER_WAIT_FRAMES = 4
DEFAULT_OUTLIER_RESET_FRAMES = 3

# GroundingDINO fallback defaults
DEFAULT_DINO_CONFIG = REPO_ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
DEFAULT_DINO_CHECKPOINT = REPO_ROOT / "checkpoints" / "groundingdino_swint_ogc.pth"
DEFAULT_DINO_CHECKPOINT_PRIMARY_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
DEFAULT_DINO_CHECKPOINT_FALLBACK_URL = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
DEFAULT_TEXT_PROMPT = "soccer ball . football . ball ."
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_USE_AMP = True
DEFAULT_AUTO_DOWNLOAD_CHECKPOINT = True
DEFAULT_DINO_BOX_THRESHOLD = 0.18
DEFAULT_DINO_TEXT_THRESHOLD = 0.12
DEFAULT_DINO_MIN_BOX_AREA = 4.0
DEFAULT_DINO_MAX_BOX_AREA = 6000.0
DEFAULT_DINO_MIN_BOX_SIDE = 2.0
DEFAULT_ENABLE_GDINO_FALLBACK = True

# Evaluation defaults
DEFAULT_MATCH_IOU = 0.5
DEFAULT_TRACKEVAL_ROOT = REPO_ROOT / "external" / "TrackEval"


GROUNDING_DINO_ROOT = REPO_ROOT / "GroundingDINO"
if str(GROUNDING_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDING_DINO_ROOT))

import groundingdino.datasets.transforms as T  # type: ignore  # noqa: E402
from groundingdino.models import build_model  # type: ignore  # noqa: E402
from groundingdino.util.misc import clean_state_dict  # type: ignore  # noqa: E402
from groundingdino.util.slconfig import SLConfig  # type: ignore  # noqa: E402
from groundingdino.util.utils import get_phrases_from_posmap  # type: ignore  # noqa: E402


@dataclass
class DetectionCandidate:
    xyxy: np.ndarray
    score: float
    phrase: str
    source: str


@dataclass
class FrameRecord:
    frame: int
    gt_exists: bool
    gt_iou_raw: float
    gt_iou_final: float
    raw_candidate_count: int
    fallback_candidate_count: int
    raw_best_score: float
    fallback_best_score: float
    output_score: float
    output_source: str
    tracker_active: bool
    tracker_gap: int
    primary_ms: float
    fallback_ms: float
    tracking_ms: float
    total_ms: float


@dataclass
class SequenceSummary:
    sequence: str
    frames_total: int
    gt_frames: int
    raw_pred_frames: int
    final_pred_frames: int
    raw_tp: int
    raw_fp: int
    raw_fn: int
    raw_precision: float
    raw_recall: float
    raw_mota: float
    raw_mean_iou: float
    raw_zero_candidate_gt_frames: int
    raw_miss_streak_count: int
    raw_mean_miss_streak: float
    raw_max_miss_streak: int
    raw_recovered_miss_rate: float
    raw_fragments: int
    final_tp: int
    final_fp: int
    final_fn: int
    final_precision: float
    final_recall: float
    final_mota: float
    final_mean_iou: float
    final_miss_streak_count: int
    final_mean_miss_streak: float
    final_max_miss_streak: int
    final_recovered_miss_rate: float
    final_fragments: int
    gap_bridged_frames: int
    zero_candidate_gap_bridged_frames: int
    fallback_recoveries: int
    primary_ms_avg: float
    fallback_ms_avg: float
    tracking_ms_avg: float
    total_ms_avg: float
    runtime_fps: float


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


def prompt_with_period(text: str) -> str:
    normalized = text.strip().lower()
    return normalized if normalized.endswith(".") else f"{normalized}."


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    converted = boxes.clone()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def clip_xyxy(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = xyxy.astype(np.float32).copy()
    clipped[0] = np.clip(clipped[0], 0, width - 1)
    clipped[1] = np.clip(clipped[1], 0, height - 1)
    clipped[2] = np.clip(clipped[2], 0, width - 1)
    clipped[3] = np.clip(clipped[3], 0, height - 1)
    if clipped[2] < clipped[0]:
        clipped[2] = clipped[0]
    if clipped[3] < clipped[1]:
        clipped[3] = clipped[1]
    return clipped


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(xyxy[0]),
            float(xyxy[1]),
            float(max(0.0, xyxy[2] - xyxy[0])),
            float(max(0.0, xyxy[3] - xyxy[1])),
        ],
        dtype=np.float32,
    )


def xyxy_center(xyxy: np.ndarray) -> np.ndarray:
    return np.array([(xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0], dtype=np.float32)


def xyxy_wh(xyxy: np.ndarray) -> np.ndarray:
    return np.array([xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]], dtype=np.float32)


def center_wh_to_xyxy(center: np.ndarray, wh: np.ndarray) -> np.ndarray:
    half = wh / 2.0
    return np.array(
        [center[0] - half[0], center[1] - half[1], center[0] + half[0], center[1] + half[1]],
        dtype=np.float32,
    )


def area_of(xyxy: np.ndarray) -> float:
    wh = xyxy_wh(xyxy)
    return float(max(0.0, wh[0]) * max(0.0, wh[1]))


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x_a = max(float(box_a[0]), float(box_b[0]))
    y_a = max(float(box_a[1]), float(box_b[1]))
    x_b = min(float(box_a[2]), float(box_b[2]))
    y_b = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = area_of(box_a) + area_of(box_b) - inter
    return float(inter / union) if union > 0.0 else 0.0


def best_iou_against_gt(pred_box: Optional[np.ndarray], gt_boxes: Sequence[np.ndarray]) -> float:
    if pred_box is None or not gt_boxes:
        return 0.0
    return max(iou_xyxy(pred_box, gt_box) for gt_box in gt_boxes)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mean_or_zero(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def normalized_sequence_name(token: str) -> str:
    token = token.strip()
    if not token:
        raise ValueError("Empty sequence token")
    if token.upper().startswith("SNMOT-"):
        return token.upper()
    return f"SNMOT-{int(token):03d}"


def parse_sequence_list(seq_list: str) -> List[str]:
    return [normalized_sequence_name(item) for item in seq_list.split(",") if item.strip()]


def resolve_sequences(data_root: Path, seq_start: int, seq_end: int, seq_list: str) -> List[Path]:
    if seq_list.strip():
        names = parse_sequence_list(seq_list)
    else:
        if seq_end < seq_start:
            raise ValueError(f"seq_end ({seq_end}) must be >= seq_start ({seq_start})")
        names = [f"SNMOT-{seq_id:03d}" for seq_id in range(seq_start, seq_end + 1)]

    paths: List[Path] = []
    missing: List[str] = []
    for name in names:
        seq_dir = data_root / name
        if seq_dir.exists():
            paths.append(seq_dir)
        else:
            missing.append(name)

    if missing:
        raise FileNotFoundError(f"Missing sequence folders under {data_root}: {', '.join(missing)}")
    return paths


def csv_write_dicts(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def download_file(url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def ensure_groundingdino_checkpoint(
    checkpoint_path: Path,
    auto_download: bool,
    download_urls: Sequence[str],
) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path

    if not auto_download:
        raise FileNotFoundError(
            f"GroundingDINO checkpoint not found: {checkpoint_path}\n"
            "Enable auto-download or place the checkpoint at this path."
        )

    last_error: Optional[Exception] = None
    for url in download_urls:
        if not url:
            continue
        try:
            print(f"[INFO] Downloading GroundingDINO checkpoint from {url}")
            download_file(url, checkpoint_path)
            print(f"[INFO] Saved checkpoint to {checkpoint_path}")
            return checkpoint_path
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            print(f"[WARN] Failed to download from {url}: {exc}")

    raise FileNotFoundError(
        f"GroundingDINO checkpoint not found and automatic download failed: {checkpoint_path}\n"
        f"Last error: {last_error}"
    )


def find_ball_track_ids(gameinfo_path: Path) -> List[int]:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(gameinfo_path)
    if not parser.has_section("Sequence"):
        raise ValueError(f"Missing [Sequence] section in {gameinfo_path}")

    ball_ids: List[int] = []
    for key, value in parser.items("Sequence"):
        if key.lower().startswith("trackletid_") and value.lower().startswith("ball"):
            ball_ids.append(int(key.split("_")[-1]))

    if not ball_ids:
        raise ValueError(f"No ball tracklets found in {gameinfo_path}")
    return sorted(ball_ids)


def load_ball_gt(seq_dir: Path) -> Dict[int, List[np.ndarray]]:
    gt_path = seq_dir / "gt" / "gt.txt"
    ball_ids = set(find_ball_track_ids(seq_dir / "gameinfo.ini"))
    gt_by_frame: Dict[int, List[np.ndarray]] = {}

    with gt_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue

            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            if track_id not in ball_ids:
                continue

            x, y, w, h = map(float, parts[2:6])
            mark = float(parts[6]) if len(parts) > 6 else 1.0
            if mark <= 0 or w <= 0 or h <= 0:
                continue

            gt_box = np.array([x, y, x + w, y + h], dtype=np.float32)
            gt_by_frame.setdefault(frame_id, []).append(gt_box)

    return gt_by_frame


def miss_streak_stats(match_series: Sequence[bool]) -> Tuple[int, float, int, float]:
    streaks: List[int] = []
    recovered = 0
    index = 0
    while index < len(match_series):
        if match_series[index]:
            index += 1
            continue
        end = index
        while end < len(match_series) and not match_series[end]:
            end += 1
        streaks.append(end - index)
        if end < len(match_series):
            recovered += 1
        index = end

    if not streaks:
        return 0, 0.0, 0, 0.0
    return len(streaks), mean_or_zero(streaks), max(streaks), safe_div(recovered, len(streaks))


def fragmentation_count(match_series: Sequence[bool]) -> int:
    segments = 0
    in_match = False
    for matched in match_series:
        if matched and not in_match:
            segments += 1
            in_match = True
        elif not matched:
            in_match = False
    return max(0, segments - 1)


class GroundingDinoDetector:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        min_box_area: float,
        max_box_area: float,
        min_box_side: float,
        use_amp: bool,
    ) -> None:
        self.device = device
        self.prompt = prompt_with_period(prompt)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.min_box_side = min_box_side
        self.use_amp = use_amp and device.startswith("cuda")
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if not config_path.exists():
            raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")

        model_args = SLConfig.fromfile(str(config_path))
        model_args.device = device
        model = build_model(model_args)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(clean_state_dict(state_dict), strict=False)
        self.model = model.to(device).eval()

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        tensor, _ = self.transform(image_pil, None)
        return tensor

    def _run_model(self, image_bgr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        image_tensor = self._preprocess(image_bgr).to(self.device)
        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(image_tensor[None], captions=[self.prompt])
            else:
                outputs = self.model(image_tensor[None], captions=[self.prompt])

        prediction_logits = outputs["pred_logits"].sigmoid()[0].float().cpu()
        prediction_boxes = outputs["pred_boxes"][0].float().cpu()
        keep = prediction_logits.max(dim=1)[0] > self.box_threshold
        logits = prediction_logits[keep]
        boxes = prediction_boxes[keep]

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(self.prompt)
        phrases = [
            get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer).replace(".", "").strip()
            for logit in logits
        ]
        return boxes, logits.max(dim=1)[0], phrases

    def predict_candidates(self, image_bgr: np.ndarray, source: str = "detector") -> List[DetectionCandidate]:
        height, width = image_bgr.shape[:2]
        boxes, logits, phrases = self._run_model(image_bgr)
        if boxes.numel() == 0:
            return []

        scaled = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
        xyxy_boxes = cxcywh_to_xyxy(scaled)
        candidates: List[DetectionCandidate] = []
        for box, score, phrase in zip(xyxy_boxes, logits, phrases):
            xyxy = clip_xyxy(box.numpy(), width=width, height=height)
            wh = xyxy_wh(xyxy)
            area = area_of(xyxy)
            if wh[0] < self.min_box_side or wh[1] < self.min_box_side:
                continue
            if area < self.min_box_area or area > self.max_box_area:
                continue
            candidates.append(
                DetectionCandidate(
                    xyxy=xyxy,
                    score=float(score.item()),
                    phrase=phrase or "ball",
                    source=source,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def predict_candidates_in_crop(self, image_bgr: np.ndarray, crop_xyxy: np.ndarray) -> List[DetectionCandidate]:
        x1, y1, x2, y2 = [int(round(value)) for value in crop_xyxy]
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        candidates = self.predict_candidates(crop, source="crop")
        translated: List[DetectionCandidate] = []
        for item in candidates:
            xyxy = item.xyxy.copy()
            xyxy[[0, 2]] += x1
            xyxy[[1, 3]] += y1
            translated.append(
                DetectionCandidate(
                    xyxy=xyxy,
                    score=item.score,
                    phrase=item.phrase,
                    source="crop",
                )
            )

        translated.sort(key=lambda item: item.score, reverse=True)
        return translated


class RFDetrBallDetector:
    def __init__(
        self,
        ball_model_path: Path,
        ball_confidence: float,
        ball_class_id: int,
        optimize_for_inference: bool,
    ) -> None:
        if not ball_model_path.exists():
            raise FileNotFoundError(f"RF-DETR ball checkpoint not found: {ball_model_path}")

        self.ball_confidence = ball_confidence
        self.ball_class_id = ball_class_id
        self.ball_model = RFDETRMedium(pretrain_weights=str(ball_model_path))

        if optimize_for_inference:
            try:
                self.ball_model.optimize_for_inference()
                print("[INFO] Optimized RF-DETR ball model for inference")
            except Exception as exc:
                print(f"[WARN] Could not optimize RF-DETR ball model: {exc}")

    def predict_ball_candidates(self, image_bgr: np.ndarray) -> List[DetectionCandidate]:
        detections = self.ball_model.predict(image_bgr, confidence=self.ball_confidence)
        if len(detections) == 0:
            return []

        class_id = np.asarray(detections.class_id)
        detections = detections[class_id == self.ball_class_id]
        if len(detections) == 0:
            return []

        candidates: List[DetectionCandidate] = []
        for xyxy, score in zip(detections.xyxy, detections.confidence):
            candidates.append(
                DetectionCandidate(
                    xyxy=xyxy.astype(np.float32),
                    score=float(score),
                    phrase="ball",
                    source="rfdetr",
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates


class OutlierDetector:
    def __init__(
        self,
        position_threshold: float,
        velocity_threshold: float,
        max_frames: int,
        outlier_wait_frames: int,
        max_suspension_frames: int,
    ) -> None:
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.position_buffer: List[np.ndarray] = []
        self.velocity_buffer: List[np.ndarray] = []
        self.max_frames = max_frames
        self.outlier_wait_frames = outlier_wait_frames
        self.max_suspension_frames = max_suspension_frames
        self.outlier_frames = 0
        self.tracking_suspended = False
        self.suspension_frames = 0

    def add_frame(self, position: np.ndarray, velocity: Optional[np.ndarray]) -> None:
        self.position_buffer.append(position.copy())
        if len(self.position_buffer) > self.max_frames:
            self.position_buffer = self.position_buffer[-self.max_frames :]
        if velocity is not None:
            self.velocity_buffer.append(velocity.copy())
            if len(self.velocity_buffer) > self.max_frames:
                self.velocity_buffer = self.velocity_buffer[-self.max_frames :]

    def is_outlier(self, new_position: np.ndarray, new_velocity: Optional[np.ndarray]) -> Tuple[bool, bool]:
        if len(self.position_buffer) < 3:
            return False, True

        avg_position = np.mean(self.position_buffer, axis=0)
        is_position_outlier = np.linalg.norm(new_position - avg_position) > self.position_threshold

        is_velocity_outlier = False
        if new_velocity is not None and len(self.velocity_buffer) >= 2:
            avg_velocity = np.mean(self.velocity_buffer, axis=0)
            is_velocity_outlier = np.linalg.norm(new_velocity - avg_velocity) > self.velocity_threshold

        is_outlier = bool(is_position_outlier or is_velocity_outlier)
        if is_outlier:
            self.outlier_frames += 1
            if self.outlier_frames >= self.outlier_wait_frames:
                self.reset()
                return True, False
            self.tracking_suspended = True
            self.suspension_frames = self.outlier_frames
            return True, False

        self.outlier_frames = 0
        self.tracking_suspended = False
        self.suspension_frames = 0
        return False, True

    def should_reset_tracking(self) -> bool:
        if self.tracking_suspended and self.suspension_frames >= self.max_suspension_frames:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self.outlier_frames = 0
        self.tracking_suspended = False
        self.suspension_frames = 0
        self.position_buffer = []
        self.velocity_buffer = []


class OutlierOnlyTracker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.outlier_detector = OutlierDetector(
            position_threshold=args.position_threshold,
            velocity_threshold=args.velocity_threshold,
            max_frames=args.outlier_history_frames,
            outlier_wait_frames=args.outlier_wait_frames,
            max_suspension_frames=args.outlier_reset_frames,
        )
        self.track_active = False
        self.track_lost_count = 0
        self.prev_position: Optional[np.ndarray] = None
        self.last_box_xyxy: Optional[np.ndarray] = None

    def select_best_candidate(self, candidates: Sequence[DetectionCandidate]) -> Optional[DetectionCandidate]:
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.score)

    def try_accept_candidate(self, candidate: DetectionCandidate) -> bool:
        current_position = xyxy_center(candidate.xyxy)
        current_velocity = current_position - self.prev_position if self.prev_position is not None else None
        _, should_use = self.outlier_detector.is_outlier(current_position, current_velocity)

        if should_use:
            self.track_active = True
            self.track_lost_count = 0
            self.outlier_detector.add_frame(current_position, current_velocity)
            self.prev_position = current_position.copy()
            self.last_box_xyxy = candidate.xyxy.copy()
            return True

        if self.outlier_detector.should_reset_tracking():
            self.reset()
        return False

    def handle_miss(self) -> None:
        if not self.track_active:
            return
        self.track_lost_count += 1
        if self.track_lost_count > self.args.max_active_gap:
            self.reset()

    def reset(self) -> None:
        self.outlier_detector.reset()
        self.track_active = False
        self.track_lost_count = 0
        self.prev_position = None
        self.last_box_xyxy = None


def evaluate_sequence(
    sequence_name: str,
    frame_records: Sequence[FrameRecord],
    gt_by_frame: Dict[int, List[np.ndarray]],
    raw_pred_frames: int,
    final_pred_frames: int,
    fallback_recoveries: int,
    match_iou: float,
) -> SequenceSummary:
    gt_frames = sum(1 for frame_id in range(1, len(frame_records) + 1) if gt_by_frame.get(frame_id))

    raw_tp = raw_fp = raw_fn = 0
    final_tp = final_fp = final_fn = 0
    raw_ious: List[float] = []
    final_ious: List[float] = []
    raw_zero_candidate_gt_frames = 0
    gap_bridged_frames = 0
    zero_candidate_gap_bridged_frames = 0

    for record in frame_records:
        if record.gt_exists:
            raw_match = record.gt_iou_raw >= match_iou
            final_match = record.gt_iou_final >= match_iou

            if record.raw_candidate_count == 0:
                raw_zero_candidate_gt_frames += 1

            if raw_match:
                raw_tp += 1
                raw_ious.append(record.gt_iou_raw)
            else:
                raw_fn += 1
                if record.raw_candidate_count > 0:
                    raw_fp += 1

            if final_match:
                final_tp += 1
                final_ious.append(record.gt_iou_final)
            else:
                final_fn += 1
                if record.output_source != "none":
                    final_fp += 1

            if final_match and not raw_match:
                gap_bridged_frames += 1
                if record.raw_candidate_count == 0:
                    zero_candidate_gap_bridged_frames += 1
        else:
            if record.raw_candidate_count > 0:
                raw_fp += 1
            if record.output_source != "none":
                final_fp += 1

    raw_match_series = [record.gt_iou_raw >= match_iou for record in frame_records if record.gt_exists]
    final_match_series = [record.gt_iou_final >= match_iou for record in frame_records if record.gt_exists]
    raw_streak_count, raw_mean_streak, raw_max_streak, raw_recovered = miss_streak_stats(raw_match_series)
    final_streak_count, final_mean_streak, final_max_streak, final_recovered = miss_streak_stats(final_match_series)

    primary_ms = [item.primary_ms for item in frame_records]
    fallback_ms = [item.fallback_ms for item in frame_records]
    tracking_ms = [item.tracking_ms for item in frame_records]
    total_ms = [item.total_ms for item in frame_records]

    return SequenceSummary(
        sequence=sequence_name,
        frames_total=len(frame_records),
        gt_frames=gt_frames,
        raw_pred_frames=raw_pred_frames,
        final_pred_frames=final_pred_frames,
        raw_tp=raw_tp,
        raw_fp=raw_fp,
        raw_fn=raw_fn,
        raw_precision=safe_div(raw_tp, raw_tp + raw_fp),
        raw_recall=safe_div(raw_tp, gt_frames),
        raw_mota=1.0 - safe_div(raw_fn + raw_fp, gt_frames),
        raw_mean_iou=mean_or_zero(raw_ious),
        raw_zero_candidate_gt_frames=raw_zero_candidate_gt_frames,
        raw_miss_streak_count=raw_streak_count,
        raw_mean_miss_streak=raw_mean_streak,
        raw_max_miss_streak=raw_max_streak,
        raw_recovered_miss_rate=raw_recovered,
        raw_fragments=fragmentation_count(raw_match_series),
        final_tp=final_tp,
        final_fp=final_fp,
        final_fn=final_fn,
        final_precision=safe_div(final_tp, final_tp + final_fp),
        final_recall=safe_div(final_tp, gt_frames),
        final_mota=1.0 - safe_div(final_fn + final_fp, gt_frames),
        final_mean_iou=mean_or_zero(final_ious),
        final_miss_streak_count=final_streak_count,
        final_mean_miss_streak=final_mean_streak,
        final_max_miss_streak=final_max_streak,
        final_recovered_miss_rate=final_recovered,
        final_fragments=fragmentation_count(final_match_series),
        gap_bridged_frames=gap_bridged_frames,
        zero_candidate_gap_bridged_frames=zero_candidate_gap_bridged_frames,
        fallback_recoveries=fallback_recoveries,
        primary_ms_avg=mean_or_zero(primary_ms),
        fallback_ms_avg=mean_or_zero(fallback_ms),
        tracking_ms_avg=mean_or_zero(tracking_ms),
        total_ms_avg=mean_or_zero(total_ms),
        runtime_fps=safe_div(1000.0, mean_or_zero(total_ms)),
    )


def aggregate_sequence_summaries(summaries: Sequence[SequenceSummary]) -> Dict[str, float]:
    if not summaries:
        return {}

    totals: Dict[str, float] = {
        "frames_total": 0.0,
        "gt_frames": 0.0,
        "raw_pred_frames": 0.0,
        "final_pred_frames": 0.0,
        "raw_tp": 0.0,
        "raw_fp": 0.0,
        "raw_fn": 0.0,
        "final_tp": 0.0,
        "final_fp": 0.0,
        "final_fn": 0.0,
        "gap_bridged_frames": 0.0,
        "zero_candidate_gap_bridged_frames": 0.0,
        "fallback_recoveries": 0.0,
    }

    for summary in summaries:
        for key in totals:
            totals[key] += float(getattr(summary, key))

    totals["raw_precision"] = safe_div(totals["raw_tp"], totals["raw_tp"] + totals["raw_fp"])
    totals["raw_recall"] = safe_div(totals["raw_tp"], totals["gt_frames"])
    totals["raw_mota"] = 1.0 - safe_div(totals["raw_fn"] + totals["raw_fp"], totals["gt_frames"])
    totals["final_precision"] = safe_div(totals["final_tp"], totals["final_tp"] + totals["final_fp"])
    totals["final_recall"] = safe_div(totals["final_tp"], totals["gt_frames"])
    totals["final_mota"] = 1.0 - safe_div(totals["final_fn"] + totals["final_fp"], totals["gt_frames"])
    totals["primary_ms_avg"] = mean_or_zero([item.primary_ms_avg for item in summaries])
    totals["fallback_ms_avg"] = mean_or_zero([item.fallback_ms_avg for item in summaries])
    totals["tracking_ms_avg"] = mean_or_zero([item.tracking_ms_avg for item in summaries])
    totals["total_ms_avg"] = mean_or_zero([item.total_ms_avg for item in summaries])
    totals["runtime_fps"] = mean_or_zero([item.runtime_fps for item in summaries])
    totals["sequence_count"] = float(len(summaries))
    return totals


def write_track_predictions(path: Path, predictions: Sequence[Tuple[int, np.ndarray, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for frame_id, xyxy, score in predictions:
            x, y, w, h = xyxy_to_xywh(xyxy)
            handle.write(f"{frame_id},1,{x:.3f},{y:.3f},{w:.3f},{h:.3f},{score:.6f},-1,-1,-1\n")


def build_trackeval_files(
    work_dir: Path,
    sequence_dirs: Sequence[Path],
    prediction_paths: Dict[str, Path],
) -> Tuple[Path, Path, Path, str]:
    gt_root = ensure_dir(work_dir / "gt")
    trackers_root = ensure_dir(work_dir / "trackers")
    output_root = ensure_dir(work_dir / "output")
    tracker_name = "rfdetr_outlier_gdino_fallback_tracker"
    tracker_data_dir = ensure_dir(trackers_root / tracker_name / "data")
    seqmap_path = work_dir / "seqmap.txt"

    with seqmap_path.open("w", encoding="utf-8") as handle:
        handle.write("name\n")
        for seq_dir in sequence_dirs:
            handle.write(f"{seq_dir.name}\n")

    for seq_dir in sequence_dirs:
        target_seq_gt_dir = ensure_dir(gt_root / seq_dir.name / "gt")
        gt_output_path = target_seq_gt_dir / "gt.txt"
        gt_by_frame = load_ball_gt(seq_dir)
        with gt_output_path.open("w", encoding="utf-8") as handle:
            for frame_id in sorted(gt_by_frame.keys()):
                for gt_box in gt_by_frame[frame_id]:
                    x, y, w, h = xyxy_to_xywh(gt_box)
                    handle.write(f"{frame_id},1,{x:.3f},{y:.3f},{w:.3f},{h:.3f},1,1,1,-1\n")

        seqinfo_src = seq_dir / "seqinfo.ini"
        if seqinfo_src.exists():
            shutil.copy2(seqinfo_src, gt_root / seq_dir.name / "seqinfo.ini")

        shutil.copy2(prediction_paths[seq_dir.name], tracker_data_dir / f"{seq_dir.name}.txt")

    return gt_root, trackers_root, output_root, tracker_name


def parse_trackeval_summary(output_root: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for candidate in output_root.rglob("*summary*"):
        if not candidate.is_file():
            continue
        try:
            if candidate.suffix.lower() == ".txt":
                lines = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(lines) >= 2:
                    headers = [item.strip() for item in lines[0].replace(",", " ").split()]
                    values = [item.strip() for item in lines[1].replace(",", " ").split()]
                    if len(headers) == len(values):
                        for header, value in zip(headers, values):
                            try:
                                metrics[header] = float(value)
                            except ValueError:
                                continue
        except OSError:
            continue
    return metrics


def run_trackeval(
    trackeval_root: Path,
    work_dir: Path,
    sequence_dirs: Sequence[Path],
    prediction_paths: Dict[str, Path],
) -> Dict[str, object]:
    run_script = trackeval_root / "scripts" / "run_mot_challenge.py"
    if not run_script.exists():
        return {"status": "skipped", "reason": f"TrackEval not found at {trackeval_root}"}

    gt_root, trackers_root, output_root, tracker_name = build_trackeval_files(work_dir, sequence_dirs, prediction_paths)
    command = [
        sys.executable,
        str(run_script),
        "--BENCHMARK",
        "SNMOTBALL",
        "--SPLIT_TO_EVAL",
        "train",
        "--GT_FOLDER",
        str(gt_root),
        "--TRACKERS_FOLDER",
        str(trackers_root),
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--SEQMAP_FILE",
        str(work_dir / "seqmap.txt"),
        "--SKIP_SPLIT_FOL",
        "True",
        "--DO_PREPROC",
        "False",
        "--METRICS",
        "HOTA",
        "CLEAR",
        "Identity",
        "--OUTPUT_FOLDER",
        str(output_root),
    ]

    process = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "stdout": process.stdout[-4000:],
        "stderr": process.stderr[-4000:],
        "metrics": parse_trackeval_summary(output_root),
        "output_dir": str(output_root),
    }


def process_sequence(
    seq_dir: Path,
    rfdetr_detector: RFDetrBallDetector,
    gdino_detector: Optional[GroundingDinoDetector],
    args: argparse.Namespace,
    run_dir: Path,
) -> Tuple[SequenceSummary, Path]:
    seq_name = seq_dir.name
    image_dir = seq_dir / "img1"
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise FileNotFoundError(f"No frames found in {image_dir}")

    if args.max_frames_per_seq > 0:
        image_paths = image_paths[: args.max_frames_per_seq]

    gt_by_frame = load_ball_gt(seq_dir)
    tracker = OutlierOnlyTracker(args=args)

    seq_output_dir = ensure_dir(run_dir / seq_name)
    prediction_path = seq_output_dir / "tracker_predictions.txt"
    frame_trace_path = seq_output_dir / "frame_trace.csv"

    frame_records: List[FrameRecord] = []
    predictions: List[Tuple[int, np.ndarray, float]] = []
    raw_pred_frames = 0
    final_pred_frames = 0
    fallback_recoveries = 0

    for frame_id, image_path in enumerate(image_paths, start=1):
        total_start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read frame {image_path}")

        primary_start = time.perf_counter()
        raw_candidates = rfdetr_detector.predict_ball_candidates(image)
        primary_ms = (time.perf_counter() - primary_start) * 1000.0
        raw_best = raw_candidates[0] if raw_candidates else None
        raw_best_box = raw_best.xyxy if raw_best is not None else None
        if raw_best is not None:
            raw_pred_frames += 1

        tracking_start = time.perf_counter()
        accepted: Optional[DetectionCandidate] = None
        primary_selected = tracker.select_best_candidate(raw_candidates)
        if primary_selected is not None and tracker.try_accept_candidate(primary_selected):
            accepted = primary_selected

        fallback_candidates: List[DetectionCandidate] = []
        fallback_ms = 0.0
        if accepted is None and args.enable_gdino_fallback and gdino_detector is not None:
            fallback_start = time.perf_counter()
            fallback_candidates = gdino_detector.predict_candidates(image, source="gdino_fallback")
            fallback_ms = (time.perf_counter() - fallback_start) * 1000.0

            fallback_selected = tracker.select_best_candidate(fallback_candidates)
            if fallback_selected is not None:
                fallback_selected = DetectionCandidate(
                    xyxy=fallback_selected.xyxy,
                    score=fallback_selected.score,
                    phrase=fallback_selected.phrase,
                    source="gdino_fallback",
                )
                if tracker.try_accept_candidate(fallback_selected):
                    accepted = fallback_selected
                    fallback_recoveries += 1

        if accepted is None:
            tracker.handle_miss()
            output_box = None
            output_score = 0.0
            output_source = "none"
        else:
            output_box = accepted.xyxy.copy()
            output_score = accepted.score
            output_source = accepted.source
        tracking_ms = (time.perf_counter() - tracking_start) * 1000.0

        if output_box is not None:
            output_box = clip_xyxy(output_box, width=image.shape[1], height=image.shape[0])
            predictions.append((frame_id, output_box, output_score))
            final_pred_frames += 1

        gt_boxes = gt_by_frame.get(frame_id, [])
        gt_iou_raw = best_iou_against_gt(raw_best_box, gt_boxes)
        gt_iou_final = best_iou_against_gt(output_box, gt_boxes)
        total_ms = (time.perf_counter() - total_start) * 1000.0

        frame_records.append(
            FrameRecord(
                frame=frame_id,
                gt_exists=bool(gt_boxes),
                gt_iou_raw=gt_iou_raw,
                gt_iou_final=gt_iou_final,
                raw_candidate_count=len(raw_candidates),
                fallback_candidate_count=len(fallback_candidates),
                raw_best_score=raw_best.score if raw_best is not None else 0.0,
                fallback_best_score=fallback_candidates[0].score if fallback_candidates else 0.0,
                output_score=output_score,
                output_source=output_source,
                tracker_active=tracker.track_active,
                tracker_gap=tracker.track_lost_count,
                primary_ms=primary_ms,
                fallback_ms=fallback_ms,
                tracking_ms=tracking_ms,
                total_ms=total_ms,
            )
        )

    write_track_predictions(prediction_path, predictions)
    csv_write_dicts(frame_trace_path, [asdict(record) for record in frame_records])
    summary = evaluate_sequence(
        sequence_name=seq_name,
        frame_records=frame_records,
        gt_by_frame=gt_by_frame,
        raw_pred_frames=raw_pred_frames,
        final_pred_frames=final_pred_frames,
        fallback_recoveries=fallback_recoveries,
        match_iou=args.match_iou,
    )
    return summary, prediction_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RF-DETR ball-only + outlier rejection + GroundingDINO fallback benchmark runner."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--seq_start", type=int, default=DEFAULT_SEQ_START)
    parser.add_argument("--seq_end", type=int, default=DEFAULT_SEQ_END)
    parser.add_argument("--seq_list", type=str, default=DEFAULT_SEQ_LIST)
    parser.add_argument("--max_frames_per_seq", type=int, default=DEFAULT_MAX_FRAMES_PER_SEQ)

    parser.add_argument("--ball_model_path", type=Path, default=DEFAULT_BALL_MODEL_PATH)
    parser.add_argument("--ball_confidence", type=float, default=DEFAULT_BALL_CONFIDENCE)
    parser.add_argument("--ball_class_id", type=int, default=DEFAULT_BALL_CLASS_ID)
    parser.add_argument("--enable_rfdetr_optimize", type=str2bool, default=DEFAULT_ENABLE_RFDETR_OPTIMIZE)
    parser.add_argument("--max_active_gap", type=int, default=DEFAULT_MAX_ACTIVE_GAP)

    parser.add_argument("--position_threshold", type=float, default=DEFAULT_POSITION_THRESHOLD)
    parser.add_argument("--velocity_threshold", type=float, default=DEFAULT_VELOCITY_THRESHOLD)
    parser.add_argument("--outlier_history_frames", type=int, default=DEFAULT_OUTLIER_HISTORY_FRAMES)
    parser.add_argument("--outlier_wait_frames", type=int, default=DEFAULT_OUTLIER_WAIT_FRAMES)
    parser.add_argument("--outlier_reset_frames", type=int, default=DEFAULT_OUTLIER_RESET_FRAMES)

    parser.add_argument("--dino_config", type=Path, default=DEFAULT_DINO_CONFIG)
    parser.add_argument("--dino_checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--auto_download_checkpoint", type=str2bool, default=DEFAULT_AUTO_DOWNLOAD_CHECKPOINT)
    parser.add_argument("--dino_checkpoint_primary_url", type=str, default=DEFAULT_DINO_CHECKPOINT_PRIMARY_URL)
    parser.add_argument("--dino_checkpoint_fallback_url", type=str, default=DEFAULT_DINO_CHECKPOINT_FALLBACK_URL)
    parser.add_argument("--text_prompt", type=str, default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--use_amp", type=str2bool, default=DEFAULT_USE_AMP)
    parser.add_argument("--dino_box_threshold", type=float, default=DEFAULT_DINO_BOX_THRESHOLD)
    parser.add_argument("--dino_text_threshold", type=float, default=DEFAULT_DINO_TEXT_THRESHOLD)
    parser.add_argument("--dino_min_box_area", type=float, default=DEFAULT_DINO_MIN_BOX_AREA)
    parser.add_argument("--dino_max_box_area", type=float, default=DEFAULT_DINO_MAX_BOX_AREA)
    parser.add_argument("--dino_min_box_side", type=float, default=DEFAULT_DINO_MIN_BOX_SIDE)
    parser.add_argument("--enable_gdino_fallback", type=str2bool, default=DEFAULT_ENABLE_GDINO_FALLBACK)

    parser.add_argument("--match_iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--trackeval_root", type=Path, default=DEFAULT_TRACKEVAL_ROOT)
    parser.add_argument("--run_trackeval", type=str2bool, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.ball_model_path = args.ball_model_path.expanduser().resolve()
    args.dino_config = args.dino_config.expanduser().resolve()
    args.dino_checkpoint = args.dino_checkpoint.expanduser().resolve()
    args.trackeval_root = args.trackeval_root.expanduser().resolve()
    if args.enable_gdino_fallback:
        args.dino_checkpoint = ensure_groundingdino_checkpoint(
            checkpoint_path=args.dino_checkpoint,
            auto_download=args.auto_download_checkpoint,
            download_urls=[
                args.dino_checkpoint_primary_url,
                args.dino_checkpoint_fallback_url,
            ],
        )

    run_dir = ensure_dir(args.output_root / args.run_name)
    sequence_dirs = resolve_sequences(
        data_root=args.data_root,
        seq_start=args.seq_start,
        seq_end=args.seq_end,
        seq_list=args.seq_list,
    )

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Sequences: {', '.join(seq_dir.name for seq_dir in sequence_dirs)}")
    print(f"[INFO] Device: {args.device}")
    print("[INFO] RF-DETR primary: ball model only")
    print(f"[INFO] GroundingDINO fallback enabled: {args.enable_gdino_fallback}")

    rfdetr_detector = RFDetrBallDetector(
        ball_model_path=args.ball_model_path,
        ball_confidence=args.ball_confidence,
        ball_class_id=args.ball_class_id,
        optimize_for_inference=args.enable_rfdetr_optimize,
    )
    gdino_detector: Optional[GroundingDinoDetector] = None
    if args.enable_gdino_fallback:
        gdino_detector = GroundingDinoDetector(
            config_path=args.dino_config,
            checkpoint_path=args.dino_checkpoint,
            device=args.device,
            prompt=args.text_prompt,
            box_threshold=args.dino_box_threshold,
            text_threshold=args.dino_text_threshold,
            min_box_area=args.dino_min_box_area,
            max_box_area=args.dino_max_box_area,
            min_box_side=args.dino_min_box_side,
            use_amp=args.use_amp,
        )

    summaries: List[SequenceSummary] = []
    prediction_paths: Dict[str, Path] = {}
    for seq_dir in sequence_dirs:
        print(f"[INFO] Processing {seq_dir.name}")
        summary, prediction_path = process_sequence(seq_dir, rfdetr_detector, gdino_detector, args, run_dir)
        summaries.append(summary)
        prediction_paths[seq_dir.name] = prediction_path
        print(
            f"[INFO] {seq_dir.name}: raw_recall={summary.raw_recall:.4f}, "
            f"final_recall={summary.final_recall:.4f}, gap_bridged={summary.gap_bridged_frames}, "
            f"fps={summary.runtime_fps:.2f}"
        )

    csv_write_dicts(run_dir / "sequence_summary.csv", [asdict(summary) for summary in summaries])
    aggregate = aggregate_sequence_summaries(summaries)

    trackeval_result: Dict[str, object] = {"status": "skipped", "reason": "disabled"}
    if args.run_trackeval:
        trackeval_result = run_trackeval(
            trackeval_root=args.trackeval_root,
            work_dir=ensure_dir(run_dir / "trackeval_work"),
            sequence_dirs=sequence_dirs,
            prediction_paths=prediction_paths,
        )

    experiment_summary = {
        "run_name": args.run_name,
        "data_root": str(args.data_root),
        "sequences": [seq_dir.name for seq_dir in sequence_dirs],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "aggregate": aggregate,
        "trackeval": trackeval_result,
    }
    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, indent=2)

    print(f"[INFO] Sequence summary written to {run_dir / 'sequence_summary.csv'}")
    print(f"[INFO] Experiment summary written to {run_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
