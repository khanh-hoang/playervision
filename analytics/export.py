"""Turn tracks into per-player pitch heatmaps, an overview image and summary stats."""
import os

import cv2
import numpy as np
import pandas as pd

from analytics.heatmap import density, overlay
from analytics.merge import merge_tracks
from analytics.pitch import PITCH_L, PITCH_W, draw_pitch, load_homography, to_pitch
from vision.track import foot_points


def read_frame(video, n):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame {n}")
    return f


def reliability_label(coverage):
    return "high" if coverage >= 85 else "medium" if coverage >= 60 else "low"


def export_players(video, tracks, calib, out, merge=True, min_frames=150, min_coverage=0.0,
                   sigma=1.5, compress=3.0):
    """Write player_<id>.png, overview.png and summary.csv into out. Returns a list of summary dicts."""
    H = load_homography(calib)
    df = tracks.copy()
    if merge:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.release()
        df = merge_tracks(df, H, fps)
    pts = foot_points(df)
    os.makedirs(out, exist_ok=True)
    scale, margin = 10, 20
    total_frames = df.frame.nunique()
    rows = []
    for tid, g in pts.groupby("track_id"):
        m = to_pitch(H, g[["x", "y"]].to_numpy())
        inside = (m[:, 0] > 0) & (m[:, 0] < PITCH_L) & (m[:, 1] > 2) & (m[:, 1] < PITCH_W - 2)
        if len(g) < min_frames or inside.mean() < 0.9:  # skip fragments and people on the touchline
            continue
        coverage = 100 * len(g) / total_frames
        if coverage < min_coverage:
            continue
        label = reliability_label(coverage)
        canvas = draw_pitch(scale, margin)
        ch, cw = canvas.shape[:2]
        img = overlay(canvas, density(m * scale + margin, (cw, ch), sigma * scale, compress))
        cv2.putText(img, f"Player {tid} | tracked {coverage:.0f}% of clip ({label} reliability)",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(f"{out}/player_{tid}.png", img)
        sm = np.stack([np.convolve(m[:, i], np.ones(5) / 5, "valid") for i in (0, 1)], 1)  # damp jitter
        dist = float(np.hypot(*np.diff(sm, axis=0).T).sum())
        rows.append({"track_id": int(tid), "frames": len(g), "coverage_pct": round(coverage, 1),
                     "reliability": label, "approx_distance_m": round(dist, 1)})
    cols = ["track_id", "frames", "coverage_pct", "reliability", "approx_distance_m"]
    pd.DataFrame(rows, columns=cols).to_csv(f"{out}/summary.csv", index=False)
    if rows:
        _write_overview(video, df, [r["track_id"] for r in rows], f"{out}/overview.png")
    return rows


FONT = cv2.FONT_HERSHEY_SIMPLEX


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _place_labels(boxes, sizes, shape):
    """Greedy label placement around each box. Returns (rects, number of labels that still overlap)."""
    h_img, w_img = shape[:2]
    placed, rects, bad = [], [], 0
    for i, ((x1, y1, x2, y2), (w, h)) in enumerate(zip(boxes, sizes)):
        cands = []
        for k in range(4):  # try further out each round
            d = k * (h + 4)
            cands += [(x1, y1 - h - 4 - d), (x1, y2 + 4 + d), (x2 + 4 + d, y1), (x1 - w - 4 - d, y1),
                      (x2 - w, y1 - h - 4 - d), (x2 - w, y2 + 4 + d)]
        pick = None
        for cx, cy in cands:
            r = (cx, cy, cx + w, cy + h)
            if (r[0] >= 0 and r[1] >= 0 and r[2] <= w_img and r[3] <= h_img
                    and not any(_overlap(r, p) for p in placed)
                    and not any(_overlap(r, tuple(b)) for j, b in enumerate(boxes) if j != i)):
                pick = r
                break
        if pick is None:
            bad += 1
            pick = (x1, y1 - h - 4, x1 + w, y1 - 4)
        placed.append(pick)
        rects.append(pick)
    return rects, bad


def _color(tid):
    hsv = np.uint8([[[(int(tid) * 47) % 180, 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _write_overview(video, df, kept, path):
    """Frame with many players visible and readable labels, each player in its own colour."""
    sub = df[df.track_id.isin(kept)]
    counts = sub.groupby("frame").track_id.nunique()
    shape = read_frame(video, int(counts.index[0])).shape
    candidates = counts[counts >= counts.max() - 2].index
    best = None
    for n in candidates:
        rows = sub[sub.frame == n]
        boxes = rows[["x1", "y1", "x2", "y2"]].astype(int).to_numpy()
        sizes = [cv2.getTextSize(str(t), FONT, 0.9, 2)[0] for t in rows.track_id]
        sizes = [(w + 8, h + 8) for w, h in sizes]
        _, bad = _place_labels(boxes, sizes, shape)
        score = len(rows) - 3 * bad
        if best is None or score > best[0]:
            best = (score, int(n), rows, boxes, sizes)
    _, n, rows, boxes, sizes = best
    img = read_frame(video, n)
    rects, _ = _place_labels(boxes, sizes, img.shape)
    for tid, (x1, y1, x2, y2), (lx1, ly1, lx2, ly2) in zip(rows.track_id, boxes, rects):
        col = _color(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.rectangle(img, (int(lx1), int(ly1)), (int(lx2), int(ly2)), col, -1)
        cv2.putText(img, str(tid), (int(lx1) + 4, int(ly2) - 5), FONT, 0.9, (0, 0, 0), 2)
        cv2.line(img, (int((lx1 + lx2) / 2), int((ly1 + ly2) / 2)), (int((x1 + x2) / 2), int((y1 + y2) / 2)), col, 1)
    cv2.imwrite(path, img)
