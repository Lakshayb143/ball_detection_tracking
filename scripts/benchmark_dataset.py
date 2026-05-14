from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ANNOTATION_FILE_NAME = "_annotations.coco.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FLAT_SEQUENCE_NAME = "FLAT-COCO"


@dataclass
class BenchmarkSequence:
    name: str
    image_paths: List[Path]
    file_names_by_frame: Dict[int, str]
    image_ids_by_frame: Dict[int, Optional[int]]
    original_frames_by_frame: Dict[int, int]
    source_dir: Optional[Path] = None


def normalized_sequence_name(token: str) -> str:
    token = token.strip()
    if not token:
        raise ValueError("Empty sequence token")
    if token.upper().startswith("SNMOT-"):
        return token.upper()
    return f"SNMOT-{int(token):03d}"


def parse_sequence_list(seq_list: str) -> List[str]:
    return [normalized_sequence_name(item) for item in seq_list.split(",") if item.strip()]


def default_annotation_path(data_root: Path) -> Optional[Path]:
    candidate = data_root / ANNOTATION_FILE_NAME
    return candidate if candidate.exists() else None


def resolve_annotated_image_path(data_root: Path, file_name: str) -> Optional[Path]:
    relative_path = Path(file_name)
    candidates: List[Path] = [data_root / relative_path]

    if "images" in relative_path.parts:
        stripped_parts = [part for part in relative_path.parts if part != "images"]
        if stripped_parts:
            candidates.append(data_root / Path(*stripped_parts))

    candidates.append(data_root / relative_path.name)

    if len(relative_path.parts) == 1:
        candidates.append(data_root / "images" / relative_path.name)

    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def parse_sequence_and_frame(file_name: str) -> Tuple[str, int]:
    relative_path = Path(file_name)
    stem = relative_path.stem
    if "_" not in stem:
        parent_parts = [part for part in relative_path.parent.parts if part != "images"]
        if parent_parts:
            return "__".join(parent_parts), 0
        return FLAT_SEQUENCE_NAME, 0
    sequence_token, frame_token = stem.rsplit("_", 1)
    try:
        sequence_name = normalized_sequence_name(sequence_token)
    except (TypeError, ValueError):
        parent_parts = [part for part in relative_path.parent.parts if part != "images"]
        sequence_name = "__".join(parent_parts) if parent_parts else FLAT_SEQUENCE_NAME

    try:
        original_frame = int(frame_token)
    except (TypeError, ValueError):
        original_frame = 0

    return sequence_name, original_frame


def _requested_sequence_names(
    discovered_names: Sequence[str],
    seq_start: int,
    seq_end: int,
    seq_list: str,
) -> List[str]:
    if seq_list.strip():
        requested = parse_sequence_list(seq_list)
        missing = [name for name in requested if name not in set(discovered_names)]
        if missing:
            raise FileNotFoundError(f"Missing requested sequences: {', '.join(missing)}")
        return requested

    if seq_end < seq_start:
        raise ValueError(f"seq_end ({seq_end}) must be >= seq_start ({seq_start})")

    if not all(
        name.upper().startswith("SNMOT-") and name.split("-")[-1].isdigit()
        for name in discovered_names
    ):
        return list(discovered_names)

    ranged = [
        name
        for name in discovered_names
        if seq_start <= int(name.split("-")[-1]) <= seq_end
    ]
    return ranged if ranged else list(discovered_names)


def _build_flat_coco_sequences(
    data_root: Path,
    seq_start: int,
    seq_end: int,
    seq_list: str,
    max_frames_per_seq: int,
    annotations_path: Optional[Path] = None,
) -> List[BenchmarkSequence]:
    annotation_path = annotations_path or default_annotation_path(data_root)
    if annotation_path is not None:
        images = json.loads(annotation_path.read_text(encoding="utf-8")).get("images", [])
        image_rows = [
            {
                "image_id": image.get("id"),
                "file_name": str(image["file_name"]),
            }
            for image in images
        ]
    else:
        image_rows = [
            {
                "image_id": None,
                "file_name": path.name,
            }
            for path in sorted(data_root.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]

    grouped: Dict[str, List[dict]] = {}
    for row in image_rows:
        file_name = row["file_name"]
        sequence_name, original_frame = parse_sequence_and_frame(file_name)
        image_path = resolve_annotated_image_path(data_root, file_name)
        if image_path is None:
            continue
        grouped.setdefault(sequence_name, []).append(
            {
                "image_id": row["image_id"],
                "file_name": file_name,
                "image_path": image_path,
                "original_frame": original_frame,
            }
        )

    discovered_names = sorted(grouped.keys())
    if not discovered_names:
        raise FileNotFoundError(f"No annotated images found under {data_root}")

    selected_names = _requested_sequence_names(discovered_names, seq_start, seq_end, seq_list)
    sequences: List[BenchmarkSequence] = []
    for name in selected_names:
        rows = sorted(grouped[name], key=lambda row: (row["original_frame"], row["file_name"]))
        if max_frames_per_seq > 0:
            rows = rows[:max_frames_per_seq]

        image_paths = [Path(row["image_path"]) for row in rows]
        file_names_by_frame = {index: row["file_name"] for index, row in enumerate(rows, start=1)}
        image_ids_by_frame = {index: row["image_id"] for index, row in enumerate(rows, start=1)}
        original_frames_by_frame = {
            index: int(row["original_frame"]) for index, row in enumerate(rows, start=1)
        }
        sequences.append(
            BenchmarkSequence(
                name=name,
                image_paths=image_paths,
                file_names_by_frame=file_names_by_frame,
                image_ids_by_frame=image_ids_by_frame,
                original_frames_by_frame=original_frames_by_frame,
                source_dir=None,
            )
        )
    return sequences


def _build_legacy_sequences(
    data_root: Path,
    seq_start: int,
    seq_end: int,
    seq_list: str,
    max_frames_per_seq: int,
) -> List[BenchmarkSequence]:
    if seq_list.strip():
        names = parse_sequence_list(seq_list)
    else:
        if seq_end < seq_start:
            raise ValueError(f"seq_end ({seq_end}) must be >= seq_start ({seq_start})")
        names = [f"SNMOT-{seq_id:03d}" for seq_id in range(seq_start, seq_end + 1)]

    sequences: List[BenchmarkSequence] = []
    missing: List[str] = []
    for name in names:
        seq_dir = data_root / name
        image_dir = seq_dir / "img1"
        if not image_dir.exists():
            missing.append(name)
            continue

        image_paths = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if max_frames_per_seq > 0:
            image_paths = image_paths[:max_frames_per_seq]

        sequences.append(
            BenchmarkSequence(
                name=name,
                image_paths=image_paths,
                file_names_by_frame={index: path.name for index, path in enumerate(image_paths, start=1)},
                image_ids_by_frame={index: None for index in range(1, len(image_paths) + 1)},
                original_frames_by_frame={
                    index: index for index in range(1, len(image_paths) + 1)
                },
                source_dir=seq_dir,
            )
        )

    if missing:
        raise FileNotFoundError(f"Missing sequence folders under {data_root}: {', '.join(missing)}")
    return sequences


def resolve_sequences(
    data_root: Path,
    seq_start: int,
    seq_end: int,
    seq_list: str,
    max_frames_per_seq: int = 0,
    annotations_path: Optional[Path] = None,
) -> List[BenchmarkSequence]:
    data_root = data_root.expanduser().resolve()
    if annotations_path is not None and annotations_path.exists():
        return _build_flat_coco_sequences(
            data_root=data_root,
            seq_start=seq_start,
            seq_end=seq_end,
            seq_list=seq_list,
            max_frames_per_seq=max_frames_per_seq,
            annotations_path=annotations_path,
        )
    if default_annotation_path(data_root) is not None:
        return _build_flat_coco_sequences(
            data_root=data_root,
            seq_start=seq_start,
            seq_end=seq_end,
            seq_list=seq_list,
            max_frames_per_seq=max_frames_per_seq,
        )
    return _build_legacy_sequences(
        data_root=data_root,
        seq_start=seq_start,
        seq_end=seq_end,
        seq_list=seq_list,
        max_frames_per_seq=max_frames_per_seq,
    )
