"""Split tracker ID swaps and stitch broken tracks of the same player back together."""
import numpy as np

from analytics.pitch import to_pitch


def merge_tracks(df, H, fps, max_gap_s=2.0, max_speed=8.0, slack_m=1.5):
    """Return a copy of df with clean, sequential track_ids (frame, track_id, x1, y1, x2, y2 required).

    1. A track is cut wherever it jumps faster than max_speed (the tracker swapped players).
    2. A piece that starts soon after, and near where, another ended joins that player.
    H maps image px to pitch metres.
    """
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    feet = np.stack([(df.x1 + df.x2) / 2, df.y2], 1)
    df["mx"], df["my"] = to_pitch(H, feet).T

    same = df.track_id.eq(df.track_id.shift())
    d = np.hypot(df.mx.diff(), df.my.diff())
    dt = df.frame.diff() / fps
    cut = same & (d > max_speed * dt + slack_m)
    df["piece"] = df.track_id.astype(str) + "_" + cut.groupby(df.track_id).cumsum().astype(str)

    info = []
    for piece, g in df.groupby("piece"):
        a, b = g.iloc[0], g.iloc[-1]
        info.append((int(a.frame), int(b.frame), piece, (a.mx, a.my), (b.mx, b.my)))
    info.sort()

    chains = []  # [chain_id, end_frame, end_pos]
    remap = {}
    for start, end, piece, p0, p1 in info:
        best, best_d = None, None
        for c in chains:
            gap = (start - c[1]) / fps
            if gap <= 0 or gap > max_gap_s:
                continue
            dist = float(np.hypot(p0[0] - c[2][0], p0[1] - c[2][1]))
            if dist <= max_speed * gap + slack_m and (best_d is None or dist < best_d):
                best, best_d = c, dist
        if best is None:
            chains.append([len(chains) + 1, end, p1])
            remap[piece] = len(chains)
        else:
            remap[piece] = best[0]
            best[1], best[2] = end, p1
    df["track_id"] = df.piece.map(remap)
    return df.drop(columns=["mx", "my", "piece"])
