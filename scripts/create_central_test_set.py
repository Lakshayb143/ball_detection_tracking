#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark_sets" / "central_test_v1"
DEFAULT_TARGET_IMAGES = 2000
DEFAULT_SEED = 20260423
CENTRAL_BALL_CATEGORY_ID = 3
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SourcePart:
    source: str
    split: str
    root: Path
    annotations: Path


@dataclass
class Candidate:
    source: str
    split: str
    root: Path
    annotations_path: Path
    image_path: Path
    image_id: int
    file_name: str
    width: int
    height: int
    ball_annotations: List[dict]
    group_key: str
    sha1: str = ""


DEFAULT_SOURCE_WEIGHTS = {
    "samy_test": 0.35,
    "tracking_test": 0.30,
    "soccer_tracking_6": 0.15,
    "soccer2": 0.10,
    "sod_full": 0.10,
}


def default_source_parts() -> List[SourcePart]:
    return [
        SourcePart(
            source="samy_test",
            split="test",
            root=REPO_ROOT / "samy_combined_ball_dataset" / "test" / "images",
            annotations=REPO_ROOT
            / "samy_combined_ball_dataset"
            / "test"
            / "images"
            / "_annotations.coco.json",
        ),
        SourcePart(
            source="tracking_test",
            split="test",
            root=REPO_ROOT / "test",
            annotations=REPO_ROOT / "test" / "_annotations.coco.json",
        ),
        SourcePart(
            source="soccer_tracking_6",
            split="merged",
            root=REPO_ROOT / "Soccer-Tracking-6",
            annotations=REPO_ROOT / "Soccer-Tracking-6" / "_annotations_merged.coco.json",
        ),
        SourcePart(
            source="soccer2",
            split="train",
            root=REPO_ROOT / "Soccer-2" / "train",
            annotations=REPO_ROOT / "Soccer-2" / "train" / "_annotations.coco.json",
        ),
        SourcePart(
            source="soccer2",
            split="valid",
            root=REPO_ROOT / "Soccer-2" / "valid",
            annotations=REPO_ROOT / "Soccer-2" / "valid" / "_annotations.coco.json",
        ),
        SourcePart(
            source="soccer2",
            split="test",
            root=REPO_ROOT / "Soccer-2" / "test",
            annotations=REPO_ROOT / "Soccer-2" / "test" / "_annotations.coco.json",
        ),
        SourcePart(
            source="sod_full",
            split="full",
            root=REPO_ROOT / "SOD_Dataset",
            annotations=REPO_ROOT / "SOD_Dataset" / "_annotations.coco.json",
        ),
    ]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_category_name(value: object) -> str:
    return str(value or "").strip().lower()


def infer_ball_category_ids(payload: dict) -> set[int]:
    ids = {
        int(category["id"])
        for category in payload.get("categories", [])
        if normalize_category_name(category.get("name")) == "ball"
    }
    if ids:
        return ids
    raise ValueError("Could not infer a category named 'ball'")


def resolve_image_path(root: Path, file_name: str) -> Optional[Path]:
    relative = Path(file_name)
    basename = relative.name
    parts = relative.parts
    candidates = [
        root / relative,
        root / "images" / relative,
        root / basename,
        root / "images" / basename,
    ]

    if len(parts) >= 3 and parts[1] == "images":
        candidates.extend(
            [
                root / parts[0] / basename,
                root / parts[0] / "images" / basename,
            ]
        )
    if len(parts) >= 2 and parts[0] == "images":
        candidates.append(root / basename)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def infer_group_key(source: str, split: str, file_name: str) -> str:
    stem = Path(file_name).stem
    if source == "tracking_test":
        return stem.rsplit("_", 1)[0]

    stem = stem.split(".rf.", 1)[0]
    for suffix in ("_jpg", "_jpeg", "_png"):
        if suffix in stem:
            stem = stem.split(suffix, 1)[0]

    for separator in ("-", "_"):
        head, sep, tail = stem.rpartition(separator)
        if sep and tail.isdigit():
            return f"{source}:{split}:{head}"
    return f"{source}:{split}:{stem}"


def iter_candidates(part: SourcePart) -> Iterable[Candidate]:
    if not part.annotations.exists():
        return []

    payload = load_json(part.annotations)
    ball_category_ids = infer_ball_category_ids(payload)
    annotations_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        if int(annotation.get("category_id", -1)) not in ball_category_ids:
            continue
        bbox = annotation.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, width, height = [float(value) for value in bbox]
        if width <= 0 or height <= 0:
            continue
        annotations_by_image_id[int(annotation["image_id"])].append(annotation)

    candidates: List[Candidate] = []
    for image in payload.get("images", []):
        image_path = resolve_image_path(part.root, str(image["file_name"]))
        if image_path is None:
            continue
        candidates.append(
            Candidate(
                source=part.source,
                split=part.split,
                root=part.root,
                annotations_path=part.annotations,
                image_path=image_path,
                image_id=int(image["id"]),
                file_name=str(image["file_name"]),
                width=int(image.get("width") or 0),
                height=int(image.get("height") or 0),
                ball_annotations=list(annotations_by_image_id.get(int(image["id"]), [])),
                group_key=infer_group_key(part.source, part.split, str(image["file_name"])),
            )
        )
    return candidates


def allocate_quotas(
    source_counts: dict[str, int],
    weights: dict[str, float],
    target_images: int,
) -> dict[str, int]:
    active_sources = [
        source
        for source, count in sorted(source_counts.items())
        if count > 0 and weights.get(source, 0.0) > 0.0
    ]
    if not active_sources:
        raise RuntimeError("No active sources are available")

    quotas = {source: 0 for source in active_sources}
    remaining = int(target_images)
    uncapped = set(active_sources)

    while remaining > 0 and uncapped:
        weight_sum = sum(weights[source] for source in uncapped)
        raw = {
            source: remaining * (weights[source] / weight_sum)
            for source in sorted(uncapped)
        }
        increments = {source: int(raw[source]) for source in raw}
        missing = remaining - sum(increments.values())
        for source in sorted(raw, key=lambda item: raw[item] - int(raw[item]), reverse=True):
            if missing <= 0:
                break
            increments[source] += 1
            missing -= 1

        progressed = False
        for source, increment in increments.items():
            capacity = source_counts[source] - quotas[source]
            take = min(increment, capacity)
            if take > 0:
                quotas[source] += take
                remaining -= take
                progressed = True
            if quotas[source] >= source_counts[source]:
                uncapped.discard(source)

        if not progressed:
            break

    if sum(quotas.values()) != target_images:
        raise RuntimeError(
            f"Could only allocate {sum(quotas.values())} of {target_images} requested images"
        )
    return quotas


def sample_diverse(candidates: List[Candidate], quota: int, rng: random.Random) -> List[Candidate]:
    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate.group_key].append(candidate)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    groups = list(buckets)
    rng.shuffle(groups)
    selected: List[Candidate] = []
    while len(selected) < quota and groups:
        next_groups = []
        for group in groups:
            if len(selected) >= quota:
                break
            bucket = buckets[group]
            if bucket:
                selected.append(bucket.pop())
            if bucket:
                next_groups.append(group)
        groups = next_groups
        rng.shuffle(groups)

    if len(selected) != quota:
        raise RuntimeError(f"Expected {quota} samples but selected {len(selected)}")
    return selected


def ensure_empty_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_root}. "
            "Use a new output path to avoid accidentally changing a frozen benchmark."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(parents=True, exist_ok=True)


def build_central_set(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    output_root = args.output_root.expanduser().resolve()
    ensure_empty_output_root(output_root)

    source_parts = default_source_parts()
    source_candidates: dict[str, list[Candidate]] = defaultdict(list)
    skipped_parts: List[dict] = []

    for part in source_parts:
        if not part.annotations.exists():
            skipped_parts.append(
                {
                    "source": part.source,
                    "split": part.split,
                    "reason": "annotations_not_found",
                    "annotations": str(part.annotations),
                }
            )
            continue
        part_candidates = list(iter_candidates(part))
        if not part_candidates:
            skipped_parts.append(
                {
                    "source": part.source,
                    "split": part.split,
                    "reason": "no_resolved_images",
                    "annotations": str(part.annotations),
                    "root": str(part.root),
                }
            )
            continue
        source_candidates[part.source].extend(part_candidates)

    seen_hashes: set[str] = set()
    unique_candidates: dict[str, list[Candidate]] = defaultdict(list)
    duplicate_count = 0
    for source, candidates in source_candidates.items():
        for candidate in candidates:
            candidate.sha1 = sha1_file(candidate.image_path)
            if candidate.sha1 in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(candidate.sha1)
            unique_candidates[source].append(candidate)

    source_counts = {source: len(candidates) for source, candidates in unique_candidates.items()}
    quotas = allocate_quotas(
        source_counts=source_counts,
        weights=DEFAULT_SOURCE_WEIGHTS,
        target_images=args.target_images,
    )

    selected: List[Candidate] = []
    for source, quota in quotas.items():
        selected.extend(sample_diverse(unique_candidates[source], quota, rng))
    rng.shuffle(selected)

    images = []
    annotations = []
    manifest_rows = []
    annotation_id = 1

    for new_image_id, candidate in enumerate(selected, start=1):
        extension = candidate.image_path.suffix.lower()
        new_file_name = f"{candidate.source}__{new_image_id:06d}{extension}"
        destination = output_root / "images" / new_file_name
        shutil.copy2(candidate.image_path, destination)

        images.append(
            {
                "id": new_image_id,
                "file_name": new_file_name,
                "width": candidate.width,
                "height": candidate.height,
                "source": candidate.source,
                "source_split": candidate.split,
                "source_image_id": candidate.image_id,
                "source_file_name": candidate.file_name,
                "source_annotations": str(candidate.annotations_path),
                "source_sha1": candidate.sha1,
            }
        )

        for source_annotation in candidate.ball_annotations:
            bbox = [float(value) for value in source_annotation["bbox"]]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": new_image_id,
                    "category_id": CENTRAL_BALL_CATEGORY_ID,
                    "bbox": bbox,
                    "area": float(source_annotation.get("area", bbox[2] * bbox[3])),
                    "iscrowd": int(source_annotation.get("iscrowd", 0)),
                    "source_annotation_id": source_annotation.get("id"),
                }
            )
            annotation_id += 1

        manifest_rows.append(
            {
                "new_image_id": new_image_id,
                "new_file_name": new_file_name,
                "source": candidate.source,
                "source_split": candidate.split,
                "source_image_id": candidate.image_id,
                "source_file_name": candidate.file_name,
                "source_image_path": str(candidate.image_path),
                "source_annotations": str(candidate.annotations_path),
                "group_key": candidate.group_key,
                "sha1": candidate.sha1,
                "ball_gt_count": len(candidate.ball_annotations),
                "width": candidate.width,
                "height": candidate.height,
            }
        )

    coco = {
        "info": {
            "description": "Frozen central ball detection benchmark set",
            "version": "central_test_v1",
            "seed": args.seed,
            "target_images": args.target_images,
        },
        "licenses": [],
        "categories": [
            {
                "id": CENTRAL_BALL_CATEGORY_ID,
                "name": "ball",
                "supercategory": "sports",
            }
        ],
        "images": images,
        "annotations": annotations,
    }
    (output_root / "_annotations.coco.json").write_text(
        json.dumps(coco, indent=2),
        encoding="utf-8",
    )

    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    source_summary = []
    for source in sorted(quotas):
        rows = [row for row in manifest_rows if row["source"] == source]
        source_summary.append(
            {
                "source": source,
                "selected_images": len(rows),
                "available_unique_images": source_counts[source],
                "ball_gt_images": sum(1 for row in rows if int(row["ball_gt_count"]) > 0),
                "ball_gt_count": sum(int(row["ball_gt_count"]) for row in rows),
            }
        )

    summary = {
        "output_root": str(output_root),
        "images": len(images),
        "annotations": len(annotations),
        "ball_category_id": CENTRAL_BALL_CATEGORY_ID,
        "seed": args.seed,
        "target_images": args.target_images,
        "source_weights": DEFAULT_SOURCE_WEIGHTS,
        "source_summary": source_summary,
        "skipped_parts": skipped_parts,
        "duplicate_images_skipped": duplicate_count,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Wrote {len(images)} images to {output_root / 'images'}")
    print(f"[INFO] Wrote {len(annotations)} ball annotations")
    print(f"[INFO] COCO annotations: {output_root / '_annotations.coco.json'}")
    print(f"[INFO] Manifest: {output_root / 'manifest.csv'}")
    print(f"[INFO] Summary: {output_root / 'summary.json'}")
    if skipped_parts:
        print("[WARN] Some configured source parts were skipped:")
        for skipped in skipped_parts:
            print(f"  - {skipped['source']}:{skipped['split']} ({skipped['reason']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a frozen 2000-image central ball benchmark set from local COCO datasets."
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target_images", type=int, default=DEFAULT_TARGET_IMAGES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    build_central_set(parse_args())


if __name__ == "__main__":
    main()
