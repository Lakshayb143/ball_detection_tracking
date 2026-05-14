"""
Batch-run ball_outlier_interpolator_v5.py over all clips that don't yet have
outputs in detections_v5/.

Usage:
  python scripts/run_detections.py              # all missing clips
  python scripts/run_detections.py --clips 3 5  # specific clips by number
  python scripts/run_detections.py --force       # re-run even if JSON exists

Outputs per clip:
  detections_v5/<clip_name>.json          — per-frame ball positions
  outputs/videos_v5/<clip_name>_v5.mp4    — annotated video
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
CLIPS_DIR   = ROOT / "clips"
GT_DIR      = ROOT / "ground_truths"
DET_DIR     = ROOT / "detections_v5"
VID_OUT_DIR = ROOT / "outputs" / "videos_v5"
TRACKER     = ROOT / "ball_outlier_interpolator_v5.py"

DET_DIR.mkdir(parents=True, exist_ok=True)
VID_OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_clips(clip_filter: list[int] | None) -> list[Path]:
    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    if clip_filter:
        names = {f"clip{n}.mp4" for n in clip_filter}
        clips = [c for c in clips if c.name in names]
    return clips


def run_clip(video: Path, force: bool) -> bool:
    name       = video.stem                                  # e.g. "clip3"
    det_json   = DET_DIR   / f"{name}.json"
    vid_out    = VID_OUT_DIR / f"{name}_v5.mp4"
    gt_json    = GT_DIR    / f"{name}_actions.json"

    if det_json.exists() and vid_out.exists() and not force:
        print(f"[skip]  {name} — detection JSON and output video already exist")
        return True

    if not gt_json.exists():
        print(f"[warn]  {name} — no ground-truth actions file at {gt_json}, running without it")
        gt_json = gt_json.parent / "clip1_actions.json"     # fallback placeholder path (unused at inference)

    print(f"[run]   {name}  video={video.name}")
    cmd = [
        sys.executable, str(TRACKER),
        "--video",        str(video),
        "--output-video", str(vid_out),
        "--output-json",  str(det_json),
        "--actions-json", str(gt_json),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[FAIL]  {name} exited with code {result.returncode}")
        return False
    print(f"[done]  {name} -> {det_json}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", type=int, metavar="N",
                        help="Clip numbers to process (e.g. 3 5 7). Default: all missing.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if detection JSON already exists.")
    args = parser.parse_args()

    clips = find_clips(args.clips)
    if not clips:
        print("No matching clips found.")
        return

    results = {clip: run_clip(clip, args.force) for clip in clips}

    print("\n=== Summary ===")
    for clip, ok in results.items():
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {clip.name}")


if __name__ == "__main__":
    main()
