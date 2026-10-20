"""Detect players with YOLO and keep persistent IDs with ByteTrack."""
import pandas as pd


def track_video(video, model_name="yolo11n.pt", stride=1, conf=0.25, imgsz=1280, on_progress=None):
    """Return a DataFrame: frame, track_id, x1, y1, x2, y2, conf (frame = original index)."""
    from ultralytics import YOLO

    model = YOLO(model_name)
    rows = []
    results = model.track(
        source=video, tracker="bytetrack.yaml", persist=True, classes=[0],
        conf=conf, imgsz=imgsz, vid_stride=stride, stream=True, verbose=False,
    )
    for i, r in enumerate(results):
        if r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            for tid, box, c in zip(ids, r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                rows.append((i * stride, tid, *box, c))
        if on_progress:
            on_progress(i)
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])


def foot_points(df):
    """Bottom-centre of each box: where the player touches the ground."""
    out = df[["frame", "track_id"]].copy()
    out["x"] = (df.x1 + df.x2) / 2
    out["y"] = df.y2
    return out
