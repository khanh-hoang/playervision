import cv2
import numpy as np


def density(xy, size, sigma=25.0, compress=3.0):
    """Normalised 0..1 density map of shape (h, w) from an (N, 2) array of x, y."""
    w, h = size
    acc = np.zeros((h, w), np.float32)
    for x, y in np.asarray(xy):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            acc[yi, xi] += 1
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    if acc.max() > 0:
        acc = np.log1p(acc / acc.max() * compress)
        acc /= acc.max()
    return acc


def overlay(background, dens, alpha=0.6, threshold=0.03):
    color = cv2.applyColorMap((dens * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(background, 1 - alpha, color, alpha, 0)
    return np.where((dens > threshold)[..., None], blended, background)
