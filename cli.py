"""PlayerVision CLI.

  python cli.py track clip.mp4 --out tracks.csv
  python cli.py preview clip.mp4 tracks.csv          # image with every track ID drawn
  python cli.py heatmap clip.mp4 tracks.csv --id 7   # heatmap for one player
"""
import argparse

import cv2
import pandas as pd

from analytics.heatmap import density, overlay
from analytics.export import export_players, read_frame
from analytics.pitch import draw_calibration_check, draw_pitch, load_homography, to_pitch, PITCH_L, PITCH_W
from vision.track import foot_points, track_video


def cmd_track(a):
    df = track_video(a.video, a.model, a.stride, imgsz=a.imgsz,
                     on_progress=lambda i: print(f"\rframe {i}", end="", flush=True))
    df.to_csv(a.out, index=False)
    print(f"\n{df.track_id.nunique()} tracks, {len(df)} detections -> {a.out}")


def cmd_preview(a):
    df = pd.read_csv(a.tracks)
    n = a.frame if a.frame is not None else int(df.frame.median())
    img = read_frame(a.video, n)
    nearest = df.frame.iloc[(df.frame - n).abs().argmin()]
    for r in df[df.frame == nearest].itertuples():
        cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 255, 0), 2)
        cv2.putText(img, str(r.track_id), (int(r.x1), int(r.y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imwrite(a.out, img)
    print(f"IDs on frame {nearest} -> {a.out}")


def cmd_heatmap(a):
    pts = foot_points(pd.read_csv(a.tracks))
    pts = pts[pts.track_id == a.id]
    if pts.empty:
        raise SystemExit(f"No track with id {a.id}")
    bg = read_frame(a.video, a.bg_frame)
    h, w = bg.shape[:2]
    d = density(pts[["x", "y"]].to_numpy(), (w, h), a.sigma, a.compress)
    cv2.imwrite(a.out, overlay(bg, d))
    print(f"{len(pts)} positions -> {a.out}")


def cmd_check(a):
    H = load_homography(a.calib)
    cv2.imwrite(a.out, draw_calibration_check(read_frame(a.video, a.frame), H))
    print(f"calibration check -> {a.out}")


def cmd_pitchmap(a):
    H = load_homography(a.calib)
    pts = foot_points(pd.read_csv(a.tracks))
    pts = pts[pts.track_id.isin(a.id)]
    if pts.empty:
        raise SystemExit(f"No track with ids {a.id}")
    m = to_pitch(H, pts[["x", "y"]].to_numpy())
    scale, margin = 10, 20
    canvas = draw_pitch(scale, margin)
    ch, cw = canvas.shape[:2]
    d = density(m * scale + margin, (cw, ch), a.sigma * scale, a.compress)
    cv2.imwrite(a.out, overlay(canvas, d))
    print(f"{len(pts)} positions, x {m[:, 0].min():.0f}-{m[:, 0].max():.0f} m, "
          f"y {m[:, 1].min():.0f}-{m[:, 1].max():.0f} m -> {a.out}")


def cmd_export(a):
    rows = export_players(a.video, pd.read_csv(a.tracks), a.calib, a.out, merge=not a.no_merge,
                          min_frames=a.min_frames, min_coverage=a.min_coverage,
                          sigma=a.sigma, compress=a.compress)
    print(f"{len(rows)} players -> {a.out}/ (player_<id>.png, overview.png, summary.csv)")


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)

    t = s.add_parser("track")
    t.add_argument("video")
    t.add_argument("--out", default="tracks.csv")
    t.add_argument("--model", default="yolo11n.pt")
    t.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    t.add_argument("--imgsz", type=int, default=1280)
    t.set_defaults(fn=cmd_track)

    v = s.add_parser("preview")
    v.add_argument("video")
    v.add_argument("tracks")
    v.add_argument("--frame", type=int)
    v.add_argument("--out", default="preview.png")
    v.set_defaults(fn=cmd_preview)

    h = s.add_parser("heatmap")
    h.add_argument("video")
    h.add_argument("tracks")
    h.add_argument("--id", type=int, required=True)
    h.add_argument("--bg-frame", type=int, default=0)
    h.add_argument("--sigma", type=float, default=25)
    h.add_argument("--compress", type=float, default=3.0)
    h.add_argument("--out", default="heatmap.png")
    h.set_defaults(fn=cmd_heatmap)

    c = s.add_parser("check", help="draw the calibrated pitch grid on a frame")
    c.add_argument("video")
    c.add_argument("--calib", default="calibration.json")
    c.add_argument("--frame", type=int, default=0)
    c.add_argument("--out", default="calib_check.png")
    c.set_defaults(fn=cmd_check)

    m = s.add_parser("pitchmap", help="top-down 2D heatmap on a pitch")
    m.add_argument("tracks")
    m.add_argument("--id", type=int, nargs="+", required=True, help="one or more IDs (merged)")
    m.add_argument("--calib", default="calibration.json")
    m.add_argument("--sigma", type=float, default=2.0, help="blur in metres")
    m.add_argument("--compress", type=float, default=3.0)
    m.add_argument("--out", default="pitchmap.png")
    m.set_defaults(fn=cmd_pitchmap)

    e = s.add_parser("export", help="heatmap for every player + overview image")
    e.add_argument("video")
    e.add_argument("tracks")
    e.add_argument("--calib", default="calibration.json")
    e.add_argument("--min-frames", type=int, default=150, help="ignore shorter tracks")
    e.add_argument("--sigma", type=float, default=1.5)
    e.add_argument("--compress", type=float, default=3.0)
    e.add_argument("--min-coverage", type=float, default=0, help="only export players tracked at least this %% of the clip")
    e.add_argument("--no-merge", action="store_true", help="skip joining broken tracks")
    e.add_argument("--out", default="output")
    e.set_defaults(fn=cmd_export)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
