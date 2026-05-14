"""
Physics-RANSAC visualizer for airborne ball segments.

For each airborne segment (from your existing airborne detector), fits a
parabola y(t) = a*t^2 + b*t + c to the ball's vertical position, marks
inlier vs outlier detections per frame, and overlays the fit on the video.

You decide visually whether the rejected detections look like real FPs.

VISUAL ENCODING
  YELLOW circle (small)   current tracker output, always shown
  GREEN circle (big)      this frame's detection is an INLIER under physics
  RED circle (big)        this frame's detection is an OUTLIER under physics
  MAGENTA curve           the fitted parabola, drawn for the active segment
  GRAY curve              same, but for segments judged PHYSICS_BAD overall
  WHITE HUD (top-left)    frame index, segment id, fit coefficients, verdict

INPUTS YOU MUST PREPARE
  TRACKER_OUTPUT_PATH   JSON dumped from ball_outlier_interpolator_v4.py
                        Default format: {frame_idx_str: [x, y, w, h]}
  AIRBORNE_SEGS_PATH    JSON dumped from your airborne event detector
                        Default format: [[start_frame, end_frame], ...]

If your formats differ, change ONLY the two adapter functions at top.

OUTPUT
  An annotated MP4 + per-segment stats printed to stdout.

Usage:
  python physics_ransac_viz.py
"""

import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION — adjust paths and tuning knobs here
# ============================================================================

VIDEO_PATH          = "/home/lakshay/lx/ball_detection_tracking/clips/clip1.mp4"
TRACKER_OUTPUT_PATH = "/home/lakshay/lx/ball_detection_tracking/detections_v5/clip1.json"
AIRBORNE_SEGS_PATH  = "/home/lakshay/lx/ball_detection_tracking/ground_truths/clip1_actions.json"
OUTPUT_VIDEO_PATH   = "physics_ransac_realtime_viz.mp4"

# Physics-fit parameters — start with these defaults, then tune by eye
MIN_SEGMENT_FRAMES  = 6      # don't fit a segment shorter than this
INLIER_PIXEL_TOL    = 30.0   # |y_actual - y_predicted| <= this -> inlier
GRAVITY_A_MIN       = 0.05   # min plausible value of a (px/frame^2)
GRAVITY_A_MAX       = 5.0    # max plausible value of a
INLIER_RATIO_OK     = 0.7    # min inlier ratio for "PHYSICS_OK" verdict

# Online/delayed mode. A decision for frame f only uses detections up through
# frame f + ONLINE_LOOKAHEAD_FRAMES, so this can be run with fixed latency.
USE_ONLINE_DELAYED_RANSAC = True
ONLINE_LOOKAHEAD_FRAMES = 15
ONLINE_HISTORY_FRAMES = 45  # set to 0 to use all segment history so far

# NOTE: GRAVITY_A_MIN and GRAVITY_A_MAX depend on your camera (zoom, fps).
# After a first run, look at printed `a` values for segments that LOOK like
# clean parabolas in the video. Set GRAVITY_A_MIN to ~half the smallest
# clean-parabola `a`, GRAVITY_A_MAX to ~2x the largest. Re-run.

# ============================================================================
# ADAPTER FUNCTIONS — change ONLY these if your data formats differ
# ============================================================================

def load_tracker_output(path):
    """Returns dict: {int frame_idx: (x, y, w, h)}.
    Supports:
      - {frame_idx_str: [x, y, w, h]}
      - [{"frame_idx": i, "x": center_x, "y": center_y, ...}, ...]
    Point outputs are returned with w=h=0 because the rest of this script
    only needs the ball center."""
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        tracker_data = {}
        for k, v in raw.items():
            if v is None:
                continue
            if isinstance(v, dict):
                frame_idx = int(v.get("frame_idx", v.get("frame", k)))
                x, y = v.get("x"), v.get("y")
                if x is None or y is None:
                    continue
                w = v.get("w", v.get("width", 0.0))
                h = v.get("h", v.get("height", 0.0))
                tracker_data[frame_idx] = (float(x), float(y), float(w), float(h))
            else:
                if len(v) < 2 or v[0] is None or v[1] is None:
                    continue
                x, y = float(v[0]), float(v[1])
                w = float(v[2]) if len(v) > 2 else 0.0
                h = float(v[3]) if len(v) > 3 else 0.0
                tracker_data[int(k)] = (x, y, w, h)
        return tracker_data

    if isinstance(raw, list):
        tracker_data = {}
        for i, rec in enumerate(raw):
            if rec is None:
                continue
            if isinstance(rec, dict):
                frame_idx = int(rec.get("frame_idx", rec.get("frame", i)))
                x, y = rec.get("x"), rec.get("y")
                if x is None or y is None:
                    continue
                if "bbox" in rec:
                    x, y, w, h = rec["bbox"][:4]
                else:
                    # detections_v5 stores center points as x/y.
                    w = rec.get("w", rec.get("width", 0.0))
                    h = rec.get("h", rec.get("height", 0.0))
                tracker_data[frame_idx] = (float(x), float(y), float(w), float(h))
            else:
                if len(rec) < 2 or rec[0] is None or rec[1] is None:
                    continue
                x, y = float(rec[0]), float(rec[1])
                w = float(rec[2]) if len(rec) > 2 else 0.0
                h = float(rec[3]) if len(rec) > 3 else 0.0
                tracker_data[i] = (x, y, w, h)
        return tracker_data

    raise ValueError(f"Unsupported tracker JSON format in {path}: {type(raw).__name__}")


def load_airborne_segments(path):
    """Returns list of (int start_frame, int end_frame).
    Supports JSON list of [start, end] pairs or {"events": [...]}."""
    with open(path) as f:
        raw = json.load(f)

    events = raw.get("events", raw) if isinstance(raw, dict) else raw
    segments = []
    for event in events:
        if isinstance(event, dict):
            if "start_frame" in event and "end_frame" in event:
                segments.append((int(event["start_frame"]), int(event["end_frame"])))
            elif "frame" in event and "duration_frames" in event:
                start = int(event["frame"])
                segments.append((start, start + int(event["duration_frames"])))
            else:
                raise ValueError(f"Unsupported airborne event format: {event}")
        else:
            s, e = event
            segments.append((int(s), int(e)))
    return segments


# ============================================================================
# PHYSICS FIT
# ============================================================================

def fit_parabola(ts, ys):
    """Closed-form least squares fit y = a*t^2 + b*t + c.
    Returns (a, b, c, residuals)."""
    coeffs = np.polyfit(ts, ys, 2)
    y_pred = np.polyval(coeffs, ts)
    residuals = np.abs(ys - y_pred)
    return coeffs[0], coeffs[1], coeffs[2], residuals


def fit_tracker_frames(frames, xs_centers, ys_centers, seg_start, seg_end):
    """Fit physics for an explicit set of tracker frames."""
    if len(frames) < MIN_SEGMENT_FRAMES:
        return None

    ts = np.array([f - seg_start for f in frames], dtype=np.float64)
    ys = np.array(ys_centers, dtype=np.float64)
    xs = np.array(xs_centers, dtype=np.float64)

    a, b, c, residuals = fit_parabola(ts, ys)
    inlier_mask = residuals <= INLIER_PIXEL_TOL
    inlier_ratio = float(inlier_mask.mean())

    # Linear x(t) just for drawing the curve over the image
    x_slope, x_intercept = np.polyfit(ts, xs, 1)

    physics_ok, a_ok, ratio_ok = physics_verdict(a, inlier_ratio)

    return {
        "seg_start": seg_start,
        "seg_end": seg_end,
        "frames": frames,
        "ts": ts,
        "xs": xs,
        "ys": ys,
        "a": float(a), "b": float(b), "c": float(c),
        "x_slope": float(x_slope),
        "x_intercept": float(x_intercept),
        "residuals": residuals,
        "inlier_mask": inlier_mask,
        "inlier_ratio": inlier_ratio,
        "physics_ok": physics_ok,
        "a_ok": a_ok,
        "ratio_ok": ratio_ok,
    }


def physics_verdict(a, inlier_ratio):
    """Per-segment OK/BAD decision."""
    a_ok = GRAVITY_A_MIN <= a <= GRAVITY_A_MAX
    ratio_ok = inlier_ratio >= INLIER_RATIO_OK
    return (a_ok and ratio_ok), a_ok, ratio_ok


def process_segment(seg_start, seg_end, tracker_data):
    """Fit physics to one airborne segment. Returns dict or None."""
    frames, xs_centers, ys_centers = [], [], []
    for f in range(seg_start, seg_end + 1):
        if f in tracker_data:
            x, y, w, h = tracker_data[f]
            frames.append(f)
            xs_centers.append(x + w / 2.0)
            ys_centers.append(y + h / 2.0)

    return fit_tracker_frames(frames, xs_centers, ys_centers, seg_start, seg_end)


def process_segment_window(seg_start, seg_end, tracker_data, fit_start, fit_end):
    """Fit using only data visible to the delayed online filter."""
    fit_start = max(seg_start, int(fit_start))
    fit_end = min(seg_end, int(fit_end))
    frames, xs_centers, ys_centers = [], [], []
    for f in range(fit_start, fit_end + 1):
        if f in tracker_data:
            x, y, w, h = tracker_data[f]
            frames.append(f)
            xs_centers.append(x + w / 2.0)
            ys_centers.append(y + h / 2.0)

    fit = fit_tracker_frames(frames, xs_centers, ys_centers, seg_start, seg_end)
    if fit is None:
        return None
    fit["fit_start"] = fit_start
    fit["fit_end"] = fit_end
    fit["mode"] = "online_delayed"
    return fit


def build_online_delayed_decisions(segments, tracker_data, lookahead_frames, history_frames):
    """Return frame_idx -> decision using only a bounded future lookahead."""
    decisions_by_frame = {}
    pending_frames = 0

    for seg_start, seg_end in segments:
        for frame_idx in range(seg_start, seg_end + 1):
            if frame_idx not in tracker_data:
                continue

            fit_end = min(seg_end, frame_idx + int(lookahead_frames))
            if history_frames > 0:
                fit_start = max(seg_start, frame_idx - int(history_frames))
            else:
                fit_start = seg_start

            fit = process_segment_window(
                seg_start, seg_end, tracker_data, fit_start, fit_end
            )
            if fit is None or frame_idx not in fit["frames"]:
                pending_frames += 1
                continue

            idx = fit["frames"].index(frame_idx)
            decisions_by_frame[frame_idx] = {
                "fit": fit,
                "is_inlier": bool(fit["inlier_mask"][idx]),
                "residual": float(fit["residuals"][idx]),
                "decision_ready_frame": frame_idx + int(lookahead_frames),
                "lookahead_frames": int(lookahead_frames),
                "history_frames": int(history_frames),
            }

    return decisions_by_frame, pending_frames


# ============================================================================
# DRAWING
# ============================================================================

def draw_ball_marker(frame, cx, cy, color, radius=12, thickness=3):
    cv2.circle(frame, (int(cx), int(cy)), radius, color, thickness)


def draw_parabola_curve(frame, fit, frame_w, frame_h):
    """Draw the fitted parabola, dense sampling for smoothness."""
    ts_dense = np.linspace(fit["ts"][0], fit["ts"][-1], 80)
    y_dense = np.polyval([fit["a"], fit["b"], fit["c"]], ts_dense)
    x_dense = fit["x_slope"] * ts_dense + fit["x_intercept"]

    pts = []
    for xx, yy in zip(x_dense, y_dense):
        if 0 <= xx < frame_w and 0 <= yy < frame_h:
            pts.append([int(xx), int(yy)])
    if len(pts) < 2:
        return

    color = (255, 0, 255) if fit["physics_ok"] else (128, 128, 128)
    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, 2)


def draw_hud(frame, frame_idx, fit=None, decision=None):
    """Top-left HUD."""
    lines = [f"Frame {frame_idx}"]
    if USE_ONLINE_DELAYED_RANSAC:
        ready_frame = frame_idx + ONLINE_LOOKAHEAD_FRAMES
        lines.append(f"ONLINE delayed fit: +{ONLINE_LOOKAHEAD_FRAMES} frames (ready at {ready_frame})")
    if fit is not None:
        n_inl = int(fit["inlier_mask"].sum())
        n_tot = len(fit["frames"])
        lines.append(f"Segment {fit['seg_start']}-{fit['seg_end']}")
        if fit.get("mode") == "online_delayed":
            lines.append(f"fit window {fit['fit_start']}-{fit['fit_end']}")
        lines.append(f"a = {fit['a']:.3f}  (allowed {GRAVITY_A_MIN}..{GRAVITY_A_MAX})")
        lines.append(f"inliers {n_inl}/{n_tot} = {fit['inlier_ratio']*100:.0f}%")
        if decision is not None:
            label = "INLIER" if decision["is_inlier"] else "OUTLIER"
            lines.append(f"this frame: {label}, residual={decision['residual']:.1f}px")
        verdict = "PHYSICS_OK" if fit["physics_ok"] else "PHYSICS_BAD"
        if not fit["a_ok"]:
            verdict += "  (a out of range)"
        if not fit["ratio_ok"]:
            verdict += "  (low inliers)"
        lines.append(verdict)
    elif USE_ONLINE_DELAYED_RANSAC:
        lines.append("physics decision pending/not in airborne fit")

    y0 = 30
    for i, line in enumerate(lines):
        # Black outline for readability over arbitrary backgrounds
        cv2.putText(frame, line, (10, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load inputs
    tracker_data = load_tracker_output(TRACKER_OUTPUT_PATH)
    segments = load_airborne_segments(AIRBORNE_SEGS_PATH)
    print(f"Loaded {len(tracker_data)} tracked frames, {len(segments)} airborne segments")

    fits_by_frame = {}      # offline mode: frame_idx -> fit dict
    decisions_by_frame = {} # online mode: frame_idx -> delayed decision dict
    pending_frames = 0
    all_fits = []

    if USE_ONLINE_DELAYED_RANSAC:
        print(
            f"Running ONLINE delayed RANSAC: lookahead={ONLINE_LOOKAHEAD_FRAMES} frames, "
            f"history={ONLINE_HISTORY_FRAMES if ONLINE_HISTORY_FRAMES > 0 else 'all'} frames"
        )
        decisions_by_frame, pending_frames = build_online_delayed_decisions(
            segments,
            tracker_data,
            lookahead_frames=ONLINE_LOOKAHEAD_FRAMES,
            history_frames=ONLINE_HISTORY_FRAMES,
        )
        for seg_start, seg_end in segments:
            seg_decisions = [
                d for f, d in decisions_by_frame.items()
                if seg_start <= f <= seg_end
            ]
            n_inliers = sum(1 for d in seg_decisions if d["is_inlier"])
            n_outliers = sum(1 for d in seg_decisions if not d["is_inlier"])
            print(
                f"  Segment {seg_start}-{seg_end}: online decisions={len(seg_decisions)}, "
                f"inliers={n_inliers}, outliers={n_outliers}"
            )
    else:
        # Process all segments up front, build per-frame lookup
        for seg_start, seg_end in segments:
            fit = process_segment(seg_start, seg_end, tracker_data)
            if fit is None:
                print(f"  Segment {seg_start}-{seg_end}: SKIPPED (too few detections)")
                continue
            all_fits.append(fit)
            for f in range(seg_start, seg_end + 1):
                fits_by_frame[f] = fit
            verdict_str = "OK" if fit["physics_ok"] else "BAD"
            print(f"  Segment {seg_start}-{seg_end}: "
                  f"a={fit['a']:.3f}, "
                  f"inliers={fit['inlier_ratio']*100:.0f}%, "
                  f"verdict={verdict_str}")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {W}x{H} @ {fps:.1f} fps, {n_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (W, H))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        decision = decisions_by_frame.get(frame_idx)
        fit = decision["fit"] if decision is not None else fits_by_frame.get(frame_idx)

        # Always: small yellow marker for the existing tracker output
        if frame_idx in tracker_data:
            x, y, w, h = tracker_data[frame_idx]
            cx, cy = x + w / 2.0, y + h / 2.0
            draw_ball_marker(frame, cx, cy, (0, 255, 255), radius=8, thickness=2)

        # If in an airborne segment: parabola curve + inlier/outlier marker
        if fit is not None:
            draw_parabola_curve(frame, fit, W, H)

            if USE_ONLINE_DELAYED_RANSAC and decision is not None:
                color = (0, 255, 0) if decision["is_inlier"] else (0, 0, 255)
                if frame_idx in tracker_data:
                    x, y, w, h = tracker_data[frame_idx]
                    cx, cy = x + w / 2.0, y + h / 2.0
                    draw_ball_marker(frame, cx, cy, color, radius=14, thickness=3)
            elif frame_idx in fit["frames"]:
                idx = fit["frames"].index(frame_idx)
                is_inlier = bool(fit["inlier_mask"][idx])
                color = (0, 255, 0) if is_inlier else (0, 0, 255)
                if frame_idx in tracker_data:
                    x, y, w, h = tracker_data[frame_idx]
                    cx, cy = x + w / 2.0, y + h / 2.0
                    draw_ball_marker(frame, cx, cy, color, radius=14, thickness=3)

        draw_hud(frame, frame_idx, fit, decision)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Summary
    if USE_ONLINE_DELAYED_RANSAC:
        n_ok = 0
        n_bad = 0
        total_inliers = sum(1 for d in decisions_by_frame.values() if d["is_inlier"])
        total_outliers = sum(1 for d in decisions_by_frame.values() if not d["is_inlier"])
    else:
        n_ok = sum(1 for f in all_fits if f["physics_ok"])
        n_bad = sum(1 for f in all_fits if not f["physics_ok"])
        total_inliers = sum(int(f["inlier_mask"].sum()) for f in all_fits)
        total_outliers = sum(int((~f["inlier_mask"]).sum()) for f in all_fits)
    print(f"\n=== Summary ===")
    if USE_ONLINE_DELAYED_RANSAC:
        print(f"Mode:                ONLINE_DELAYED (+{ONLINE_LOOKAHEAD_FRAMES} frames)")
        print(f"Decision frames:     {len(decisions_by_frame)}")
        print(f"Pending/too-short:   {pending_frames}")
    else:
        print(f"Segments fit:        {len(all_fits)}")
        print(f"  PHYSICS_OK:        {n_ok}")
        print(f"  PHYSICS_BAD:       {n_bad}")
    print(f"Frame-level totals (within fit segments):")
    print(f"  inliers (kept):    {total_inliers}")
    print(f"  outliers (reject): {total_outliers}")
    print(f"\nWrote {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
