#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


DEFAULT_OUTPUTS_ROOT = Path("outputs")
EXPECTED_SUMMARY_FILES = {
    "benchmark_metrics.csv",
    "experiment_summary.json",
    "sequence_summary.csv",
    "detections.json",
}
EXPECTED_SEQUENCES = [
    "SNMOT-109",
    "SNMOT-112",
    "SNMOT-128",
    "SNMOT-143",
    "SNMOT-158",
    "SNMOT-165",
]
OUTLIER_MODES: Dict[str, str] = {
    "off": "--position_threshold 1000000 --velocity_threshold 1000000",
    "position_only": "--position_threshold 60 --velocity_threshold 1000000",
    "velocity_only": "--position_threshold 1000000 --velocity_threshold 40",
    "both": "--position_threshold 60 --velocity_threshold 40",
}
GDINO_CROP_ARGS = (
    "--gdino_crop_expansion 12 --gdino_min_crop_size 192 "
    "--gdino_max_crop_size 640 --gdino_full_frame_on_reset true"
)
SST_CROP_ARGS = (
    "--sst_crop_expansion 12 --sst_min_crop_size 192 "
    "--sst_max_crop_size 640 --sst_full_frame_on_reset true"
)


@dataclass(frozen=True)
class ExpectedRun:
    group: str
    run_name: str
    command: str

    @property
    def run_dir(self) -> Path:
        return Path(self.group) / self.run_name


def build_expected_runs() -> List[ExpectedRun]:
    runs: List[ExpectedRun] = []
    for mode, outlier_args in OUTLIER_MODES.items():
        runs.append(
            ExpectedRun(
                group="rfdetr_outlier_gdino_fullframe_fallback_tracking",
                run_name=f"rfdetr_outlier_only_{mode}",
                command=(
                    "python scripts/benchmark_rfdetr_outlier_gdino_fullframe_fallback_tracking.py "
                    f'--run_name "rfdetr_outlier_only_{mode}" '
                    "--enable_gdino_fallback false "
                    f"{outlier_args}"
                ),
            )
        )
        runs.append(
            ExpectedRun(
                group="rfdetr_outlier_gdino_fullframe_fallback_tracking",
                run_name=f"rfdetr_gdino_fullframe_{mode}",
                command=(
                    "python scripts/benchmark_rfdetr_outlier_gdino_fullframe_fallback_tracking.py "
                    f'--run_name "rfdetr_gdino_fullframe_{mode}" '
                    "--enable_gdino_fallback true "
                    f"{outlier_args}"
                ),
            )
        )
        runs.append(
            ExpectedRun(
                group="rfdetr_outlier_sst_fullframe_fallback_tracking",
                run_name=f"rfdetr_sst_fullframe_{mode}",
                command=(
                    "python scripts/benchmark_rfdetr_outlier_sst_fullframe_fallback_tracking.py "
                    f'--run_name "rfdetr_sst_fullframe_{mode}" '
                    "--enable_sst_fallback true "
                    f"{outlier_args}"
                ),
            )
        )
        runs.append(
            ExpectedRun(
                group="rfdetr_groundingdino_fallback_tracking",
                run_name=f"rfdetr_gdino_crop_{mode}",
                command=(
                    "python scripts/benchmark_rfdetr_groundingdino_fallback_tracking.py "
                    f'--run_name "rfdetr_gdino_crop_{mode}" '
                    f"{GDINO_CROP_ARGS} "
                    f"{outlier_args}"
                ),
            )
        )
        runs.append(
            ExpectedRun(
                group="rfdetr_sst_fallback_tracking",
                run_name=f"rfdetr_sst_crop_{mode}",
                command=(
                    "python scripts/benchmark_rfdetr_sst_fallback_tracking.py "
                    f'--run_name "rfdetr_sst_crop_{mode}" '
                    f"{SST_CROP_ARGS} "
                    f"{outlier_args}"
                ),
            )
        )
    return runs


def status_for_run(outputs_root: Path, run: ExpectedRun) -> Dict[str, object]:
    run_dir = outputs_root / run.group / run.run_name
    present_files = {item.name for item in run_dir.iterdir() if item.is_file()} if run_dir.exists() else set()
    present_sequences = [
        name for name in EXPECTED_SEQUENCES if (run_dir / name).is_dir()
    ]
    missing_files = sorted(EXPECTED_SUMMARY_FILES - present_files)

    if EXPECTED_SUMMARY_FILES.issubset(present_files):
        status = "complete"
    elif run_dir.exists():
        status = "partial"
    else:
        status = "missing"

    return {
        "group": run.group,
        "run_name": run.run_name,
        "status": status,
        "run_dir": str(run_dir),
        "sequence_dirs_present": len(present_sequences),
        "sequence_dirs_expected": len(EXPECTED_SEQUENCES),
        "present_sequences": ",".join(present_sequences),
        "missing_files": ",".join(missing_files),
        "resume_command": run.command,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "run_name",
        "status",
        "run_dir",
        "sequence_dirs_present",
        "sequence_dirs_expected",
        "present_sequences",
        "missing_files",
        "resume_command",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report complete/partial/missing status for the 20-run ablation matrix."
    )
    parser.add_argument("--outputs_root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--write_csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.expanduser().resolve()

    rows = [status_for_run(outputs_root, run) for run in build_expected_runs()]

    complete = [row for row in rows if row["status"] == "complete"]
    partial = [row for row in rows if row["status"] == "partial"]
    missing = [row for row in rows if row["status"] == "missing"]

    print(
        f"[INFO] Ablation matrix status: "
        f"{len(complete)} complete, {len(partial)} partial, {len(missing)} missing"
    )

    if partial:
        print("\n[INFO] Partial runs:")
        for row in partial:
            print(
                f"- {row['group']}/{row['run_name']} "
                f"({row['sequence_dirs_present']}/{row['sequence_dirs_expected']} sequences)"
            )

    if missing:
        print("\n[INFO] Missing runs:")
        for row in missing:
            print(f"- {row['group']}/{row['run_name']}")

    if partial or missing:
        print("\n[INFO] Resume commands:")
        for row in rows:
            if row["status"] != "complete":
                print(row["resume_command"])

    if args.write_csv is not None:
        output_path = args.write_csv.expanduser().resolve()
        write_csv(output_path, rows)
        print(f"\n[INFO] Wrote status CSV to {output_path}")


if __name__ == "__main__":
    main()
