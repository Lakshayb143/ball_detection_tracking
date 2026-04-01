#!/usr/bin/env python3
"""
GroundingDINO-only benchmark runner for SoccerNet SNMOT ball tracking.

This script is intentionally isolated from SST and Grounded-Segment-Anything.
Edit the DEFAULT_* values if you prefer not to pass arguments in the terminal.
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
from transformers import BertModel
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

# Dataset / run defaults
DEFAULT_DATA_ROOT = Path("~/SoccerNet_tracking/tracking-2023/train").expanduser()
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "groundingdino_ball_tracking"
DEFAULT_RUN_NAME = "groundingdino_gap_bridge_v1"
DEFAULT_SEQ_START = 60
DEFAULT_SEQ_END = 70
DEFAULT_SEQ_LIST = ""
DEFAULT_MAX_FRAMES_PER_SEQ = 0

# GroundingDINO defaults
DEFAULT_DINO_CONFIG = REPO_ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
DEFAULT_DINO_CHECKPOINT = REPO_ROOT / "checkpoints" / "groundingdino_swint_ogc.pth"
DEFAULT_DINO_CHECKPOINT_PRIMARY_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
DEFAULT_DINO_CHECKPOINT_FALLBACK_URL = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
DEFAULT_TEXT_PROMPT = "soccer ball . football . ball ."
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_USE_AMP = True
DEFAULT_AUTO_DOWNLOAD_CHECKPOINT = True
DEFAULT_BOX_THRESHOLD = 0.18
DEFAULT_TEXT_THRESHOLD = 0.12
DEFAULT_MIN_BOX_AREA = 4.0
DEFAULT_MAX_BOX_AREA = 6000.0
DEFAULT_MIN_BOX_SIDE = 2.0

# Tracking / recovery defaults
DEFAULT_INIT_CONFIDENCE = 0.22
DEFAULT_BASE_GATE_DISTANCE = 80.0
DEFAULT_GATE_GROWTH_PER_MISS = 25.0
DEFAULT_ENABLE_CROP_REDETECT = True
DEFAULT_CROP_EXPANSION = 10.0
DEFAULT_MIN_CROP_SIZE = 160
DEFAULT_MAX_CROP_SIZE = 640
DEFAULT_MAX_PREDICT_GAP = 4
DEFAULT_MAX_ACTIVE_GAP = 10
DEFAULT_BOX_EMA = 0.65
DEFAULT_BRIDGE_SCORE_DECAY = 0.85
DEFAULT_MIN_BRIDGE_SCORE = 0.05

# Evaluation defaults
DEFAULT_MATCH_IOU = 0.5
DEFAULT_TRACKEVAL_ROOT = REPO_ROOT / "external" / "TrackEval"


GROUNDING_DINO_ROOT = REPO_ROOT / "GroundingDINO"
if str(GROUNDING_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDING_DINO_ROOT))

if not hasattr(BertModel, "get_head_mask"):
    raise RuntimeError(
        "This GroundingDINO code is not compatible with the installed transformers package.\n"
        "GroundingDINO in this repo expects BertModel.get_head_mask, which is missing in newer transformers releases.\n"
        "Recommended fix on your Linux box:\n"
        "  pip uninstall -y transformers\n"
        "  pip install 'transformers>=4.37,<5'\n"
        "GroundingDINO issue reference: https://github.com/IDEA-Research/GroundingDINO/issues/446"
    )

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
    crop_candidate_count: int
    raw_best_score: float
    crop_best_score: float
    output_score: float
    output_source: str
    tracker_active: bool
    tracker_gap: int
    detector_ms: float
    crop_ms: float
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
    crop_recoveries: int
    detector_ms_avg: float
    crop_ms_avg: float
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


class ConstantVelocityKalman:
    def __init__(self, dt: float = 1.0) -> None:
        self.A = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        self.Q = np.diag([2.0, 2.0, 6.0, 6.0]).astype(np.float32)
        self.R = np.diag([12.0, 12.0]).astype(np.float32)
        self.P = np.eye(4, dtype=np.float32) * 50.0
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False

    def initialize(self, center: np.ndarray) -> None:
        self.x[:] = 0.0
        self.x[0, 0] = float(center[0])
        self.x[1, 0] = float(center[1])
        self.P = np.eye(4, dtype=np.float32) * 50.0
        self.initialized = True

    def predict(self) -> np.ndarray:
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2, 0].copy()

    def update(self, center: np.ndarray) -> None:
        measurement = center.reshape(2, 1).astype(np.float32)
        innovation = measurement - (self.H @ self.x)
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.P = (np.eye(4, dtype=np.float32) - k @ self.H) @ self.P


class SingleBallTracker:
    def __init__(
        self,
        init_confidence: float,
        base_gate_distance: float,
        gate_growth_per_miss: float,
        enable_crop_redetect: bool,
        crop_expansion: float,
        min_crop_size: int,
        max_crop_size: int,
        max_predict_gap: int,
        max_active_gap: int,
        box_ema: float,
        bridge_score_decay: float,
        min_bridge_score: float,
    ) -> None:
        self.init_confidence = init_confidence
        self.base_gate_distance = base_gate_distance
        self.gate_growth_per_miss = gate_growth_per_miss
        self.enable_crop_redetect = enable_crop_redetect
        self.crop_expansion = crop_expansion
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.max_predict_gap = max_predict_gap
        self.max_active_gap = max_active_gap
        self.box_ema = box_ema
        self.bridge_score_decay = bridge_score_decay
        self.min_bridge_score = min_bridge_score

        self.kalman = ConstantVelocityKalman()
        self.active = False
        self.frames_since_update = 0
        self.last_score = 0.0
        self.smoothed_wh: Optional[np.ndarray] = None
        self.prediction_center: Optional[np.ndarray] = None

    def predict_state(self) -> Optional[Dict[str, float]]:
        self.prediction_center = None
        if not self.active or not self.kalman.initialized:
            return None

        self.prediction_center = self.kalman.predict()
        gate_distance = self.base_gate_distance + self.frames_since_update * self.gate_growth_per_miss
        return {"gate_distance": float(gate_distance)}

    def select_candidate(
        self,
        candidates: Sequence[DetectionCandidate],
        prediction: Optional[Dict[str, float]],
    ) -> Optional[DetectionCandidate]:
        if not candidates:
            return None

        if prediction is None or self.prediction_center is None:
            best = max(candidates, key=lambda item: item.score)
            return best if best.score >= self.init_confidence else None

        gate_distance = prediction["gate_distance"]
        gated: List[Tuple[float, float, DetectionCandidate]] = []
        for item in candidates:
            distance = float(np.linalg.norm(xyxy_center(item.xyxy) - self.prediction_center))
            if distance <= gate_distance:
                score = item.score - 0.15 * (distance / max(gate_distance, 1e-6))
                gated.append((score, -distance, item))

        if not gated:
            return None
        gated.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return gated[0][2]

    def build_search_crop(self, frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        if not self.enable_crop_redetect or self.prediction_center is None or self.smoothed_wh is None:
            return None

        height, width = frame_shape[:2]
        crop_size = float(max(self.min_crop_size, max(self.smoothed_wh) * self.crop_expansion))
        crop_size = float(min(crop_size, self.max_crop_size))
        half = crop_size / 2.0
        crop_xyxy = np.array(
            [
                self.prediction_center[0] - half,
                self.prediction_center[1] - half,
                self.prediction_center[0] + half,
                self.prediction_center[1] + half,
            ],
            dtype=np.float32,
        )
        crop_xyxy = clip_xyxy(crop_xyxy, width=width, height=height)
        if crop_xyxy[2] <= crop_xyxy[0] or crop_xyxy[3] <= crop_xyxy[1]:
            return None
        return crop_xyxy

    def commit(self, accepted: Optional[DetectionCandidate]) -> Tuple[Optional[np.ndarray], float, str]:
        if accepted is not None:
            center = xyxy_center(accepted.xyxy)
            wh = xyxy_wh(accepted.xyxy)
            if not self.active or not self.kalman.initialized:
                self.kalman.initialize(center)
            else:
                self.kalman.update(center)
            self.active = True
            self.frames_since_update = 0
            self.last_score = accepted.score
            self.smoothed_wh = wh if self.smoothed_wh is None else (self.box_ema * self.smoothed_wh + (1.0 - self.box_ema) * wh)
            return accepted.xyxy.copy(), accepted.score, accepted.source

        if not self.active or self.prediction_center is None or self.smoothed_wh is None:
            return None, 0.0, "none"

        self.frames_since_update += 1
        if self.frames_since_update > self.max_active_gap:
            self.reset()
            return None, 0.0, "none"

        if self.frames_since_update <= self.max_predict_gap:
            bridge_xyxy = center_wh_to_xyxy(self.prediction_center, self.smoothed_wh)
            score = max(self.min_bridge_score, self.last_score * (self.bridge_score_decay ** self.frames_since_update))
            return bridge_xyxy, score, "bridge"

        return None, 0.0, "none"

    def reset(self) -> None:
        self.kalman = ConstantVelocityKalman()
        self.active = False
        self.frames_since_update = 0
        self.last_score = 0.0
        self.smoothed_wh = None
        self.prediction_center = None


def evaluate_sequence(
    sequence_name: str,
    frame_records: Sequence[FrameRecord],
    gt_by_frame: Dict[int, List[np.ndarray]],
    raw_pred_frames: int,
    final_pred_frames: int,
    crop_recoveries: int,
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

    detector_ms = [item.detector_ms for item in frame_records]
    crop_ms = [item.crop_ms for item in frame_records]
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
        crop_recoveries=crop_recoveries,
        detector_ms_avg=mean_or_zero(detector_ms),
        crop_ms_avg=mean_or_zero(crop_ms),
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
        "crop_recoveries": 0.0,
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
    totals["detector_ms_avg"] = mean_or_zero([item.detector_ms_avg for item in summaries])
    totals["crop_ms_avg"] = mean_or_zero([item.crop_ms_avg for item in summaries])
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
    tracker_name = "groundingdino_tracker"
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
    detector: GroundingDinoDetector,
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
    tracker = SingleBallTracker(
        init_confidence=args.init_confidence,
        base_gate_distance=args.base_gate_distance,
        gate_growth_per_miss=args.gate_growth_per_miss,
        enable_crop_redetect=args.enable_crop_redetect,
        crop_expansion=args.crop_expansion,
        min_crop_size=args.min_crop_size,
        max_crop_size=args.max_crop_size,
        max_predict_gap=args.max_predict_gap,
        max_active_gap=args.max_active_gap,
        box_ema=args.box_ema,
        bridge_score_decay=args.bridge_score_decay,
        min_bridge_score=args.min_bridge_score,
    )

    seq_output_dir = ensure_dir(run_dir / seq_name)
    prediction_path = seq_output_dir / "tracker_predictions.txt"
    frame_trace_path = seq_output_dir / "frame_trace.csv"

    frame_records: List[FrameRecord] = []
    predictions: List[Tuple[int, np.ndarray, float]] = []
    raw_pred_frames = 0
    final_pred_frames = 0
    crop_recoveries = 0

    for frame_id, image_path in enumerate(image_paths, start=1):
        total_start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read frame {image_path}")

        detector_start = time.perf_counter()
        raw_candidates = detector.predict_candidates(image, source="detector")
        detector_ms = (time.perf_counter() - detector_start) * 1000.0
        raw_best = raw_candidates[0] if raw_candidates else None
        raw_best_box = raw_best.xyxy if raw_best is not None else None
        if raw_best is not None:
            raw_pred_frames += 1

        tracking_start = time.perf_counter()
        prediction = tracker.predict_state()
        accepted = tracker.select_candidate(raw_candidates, prediction)

        crop_candidates: List[DetectionCandidate] = []
        crop_ms = 0.0
        if accepted is None and prediction is not None and args.enable_crop_redetect:
            crop_xyxy = tracker.build_search_crop(image.shape)
            if crop_xyxy is not None:
                crop_start = time.perf_counter()
                crop_candidates = detector.predict_candidates_in_crop(image, crop_xyxy)
                crop_ms = (time.perf_counter() - crop_start) * 1000.0
                accepted = tracker.select_candidate(crop_candidates, prediction)
                if accepted is not None:
                    crop_recoveries += 1

        output_box, output_score, output_source = tracker.commit(accepted)
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
                crop_candidate_count=len(crop_candidates),
                raw_best_score=raw_best.score if raw_best is not None else 0.0,
                crop_best_score=crop_candidates[0].score if crop_candidates else 0.0,
                output_score=output_score,
                output_source=output_source,
                tracker_active=tracker.active,
                tracker_gap=tracker.frames_since_update,
                detector_ms=detector_ms,
                crop_ms=crop_ms,
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
        crop_recoveries=crop_recoveries,
        match_iou=args.match_iou,
    )
    return summary, prediction_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GroundingDINO-only benchmark runner for SoccerNet SNMOT ball tracking."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--seq_start", type=int, default=DEFAULT_SEQ_START)
    parser.add_argument("--seq_end", type=int, default=DEFAULT_SEQ_END)
    parser.add_argument("--seq_list", type=str, default=DEFAULT_SEQ_LIST)
    parser.add_argument("--max_frames_per_seq", type=int, default=DEFAULT_MAX_FRAMES_PER_SEQ)

    parser.add_argument("--dino_config", type=Path, default=DEFAULT_DINO_CONFIG)
    parser.add_argument("--dino_checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--auto_download_checkpoint", type=str2bool, default=DEFAULT_AUTO_DOWNLOAD_CHECKPOINT)
    parser.add_argument("--dino_checkpoint_primary_url", type=str, default=DEFAULT_DINO_CHECKPOINT_PRIMARY_URL)
    parser.add_argument("--dino_checkpoint_fallback_url", type=str, default=DEFAULT_DINO_CHECKPOINT_FALLBACK_URL)
    parser.add_argument("--text_prompt", type=str, default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--use_amp", type=str2bool, default=DEFAULT_USE_AMP)
    parser.add_argument("--box_threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text_threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    parser.add_argument("--min_box_area", type=float, default=DEFAULT_MIN_BOX_AREA)
    parser.add_argument("--max_box_area", type=float, default=DEFAULT_MAX_BOX_AREA)
    parser.add_argument("--min_box_side", type=float, default=DEFAULT_MIN_BOX_SIDE)

    parser.add_argument("--init_confidence", type=float, default=DEFAULT_INIT_CONFIDENCE)
    parser.add_argument("--base_gate_distance", type=float, default=DEFAULT_BASE_GATE_DISTANCE)
    parser.add_argument("--gate_growth_per_miss", type=float, default=DEFAULT_GATE_GROWTH_PER_MISS)
    parser.add_argument("--enable_crop_redetect", type=str2bool, default=DEFAULT_ENABLE_CROP_REDETECT)
    parser.add_argument("--crop_expansion", type=float, default=DEFAULT_CROP_EXPANSION)
    parser.add_argument("--min_crop_size", type=int, default=DEFAULT_MIN_CROP_SIZE)
    parser.add_argument("--max_crop_size", type=int, default=DEFAULT_MAX_CROP_SIZE)
    parser.add_argument("--max_predict_gap", type=int, default=DEFAULT_MAX_PREDICT_GAP)
    parser.add_argument("--max_active_gap", type=int, default=DEFAULT_MAX_ACTIVE_GAP)
    parser.add_argument("--box_ema", type=float, default=DEFAULT_BOX_EMA)
    parser.add_argument("--bridge_score_decay", type=float, default=DEFAULT_BRIDGE_SCORE_DECAY)
    parser.add_argument("--min_bridge_score", type=float, default=DEFAULT_MIN_BRIDGE_SCORE)

    parser.add_argument("--match_iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--trackeval_root", type=Path, default=DEFAULT_TRACKEVAL_ROOT)
    parser.add_argument("--run_trackeval", type=str2bool, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.dino_config = args.dino_config.expanduser().resolve()
    args.dino_checkpoint = args.dino_checkpoint.expanduser().resolve()
    args.trackeval_root = args.trackeval_root.expanduser().resolve()
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
    print(f"[INFO] Crop re-detect enabled: {args.enable_crop_redetect}")

    detector = GroundingDinoDetector(
        config_path=args.dino_config,
        checkpoint_path=args.dino_checkpoint,
        device=args.device,
        prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        min_box_area=args.min_box_area,
        max_box_area=args.max_box_area,
        min_box_side=args.min_box_side,
        use_amp=args.use_amp,
    )

    summaries: List[SequenceSummary] = []
    prediction_paths: Dict[str, Path] = {}
    for seq_dir in sequence_dirs:
        print(f"[INFO] Processing {seq_dir.name}")
        summary, prediction_path = process_sequence(seq_dir, detector, args, run_dir)
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
