import numpy as np
import pandas as pd

from analytics.merge import merge_tracks


def _box(frame, tid, x, y):
    return dict(frame=frame, track_id=tid, x1=x - 5, y1=y - 20, x2=x + 5, y2=y)


def _counts(rows, fps=10):
    out = merge_tracks(pd.DataFrame(rows), np.eye(3), fps)  # 1 px = 1 m
    return sorted(out.groupby("track_id").frame.count().tolist())


def test_close_in_time_and_space_is_merged_far_is_not():
    rows = [_box(f, 1, 10 + f, 50) for f in range(0, 10)]
    rows += [_box(f, 2, 10 + f, 50) for f in range(12, 20)]   # continues track 1
    rows += [_box(f, 3, 90, 90) for f in range(12, 20)]        # far away
    assert _counts(rows) == [8, 18]


def test_impossible_jump_inside_a_track_is_split():
    rows = [_box(f, 1, 10, 50) for f in range(0, 10)]
    rows += [_box(f, 1, 80, 20) for f in range(10, 20)]        # tracker swapped to another player
    assert _counts(rows) == [10, 10]
