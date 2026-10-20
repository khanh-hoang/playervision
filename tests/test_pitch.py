import numpy as np

from analytics.pitch import load_homography, to_pitch


def test_calibration_points_map_to_pitch(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"image": [[0,0],[100,0],[100,50],[0,50]], "pitch": [[0,0],[10,0],[10,5],[0,5]]}')
    H = load_homography(p)
    out = to_pitch(H, [[50, 25]])
    assert np.allclose(out, [[5, 2.5]], atol=1e-3)
