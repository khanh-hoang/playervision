import numpy as np

from analytics.heatmap import density


def test_peak_is_where_player_stood():
    xy = np.array([[50, 40]] * 20 + [[10, 10]])
    d = density(xy, (100, 80), sigma=3)
    assert d.shape == (80, 100)
    assert np.unravel_index(d.argmax(), d.shape) == (40, 50)
    assert d.max() == 1.0


def test_out_of_bounds_points_ignored():
    d = density(np.array([[-5, 500]]), (100, 80))
    assert d.max() == 0
