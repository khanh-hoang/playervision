"""Map image coordinates to a 105x68 m pitch and draw top-down views."""
import json

import cv2
import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0


def load_homography(path):
    """Image -> pitch (metres) homography from a JSON file with matching 'image' and 'pitch' point lists."""
    with open(path) as f:
        c = json.load(f)
    H, _ = cv2.findHomography(np.float32(c["image"]), np.float32(c["pitch"]))
    return H


def to_pitch(H, xy):
    pts = np.float32(xy).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _px(p, scale, margin):
    return (int(round(p[0] * scale + margin)), int(round(p[1] * scale + margin)))


def draw_pitch(scale=10, margin=20):
    """Green pitch canvas with markings. Returns the image."""
    w, h = int(PITCH_L * scale + 2 * margin), int(PITCH_W * scale + 2 * margin)
    img = np.full((h, w, 3), (60, 120, 50), np.uint8)
    white = (255, 255, 255)

    def rect(x0, y0, x1, y1):
        cv2.rectangle(img, _px((x0, y0), scale, margin), _px((x1, y1), scale, margin), white, 2)

    cy = PITCH_W / 2
    rect(0, 0, PITCH_L, PITCH_W)
    cv2.line(img, _px((PITCH_L / 2, 0), scale, margin), _px((PITCH_L / 2, PITCH_W), scale, margin), white, 2)
    cv2.circle(img, _px((PITCH_L / 2, cy), scale, margin), int(9.15 * scale), white, 2)
    for side in (0, PITCH_L):
        sgn = 1 if side == 0 else -1
        rect(side, cy - 20.16, side + sgn * 16.5, cy + 20.16)
        rect(side, cy - 9.16, side + sgn * 5.5, cy + 9.16)
    return img


def draw_calibration_check(frame, H):
    """Project a 5 m pitch grid and the centre circle onto the frame to check the calibration."""
    Hinv = np.linalg.inv(H)
    out = frame.copy()

    def poly(pts, color):
        p = to_pitch(Hinv, pts).astype(np.int32)
        cv2.polylines(out, [p], False, color, 2)

    for x in np.arange(0, PITCH_L + 1, 5):
        poly([(x, 0), (x, PITCH_W)], (0, 255, 255) if x != PITCH_L / 2 else (0, 0, 255))
    for y in np.arange(0, PITCH_W + 1, 5):
        poly([(0, y), (PITCH_L, y)], (0, 255, 255))
    a = np.linspace(0, 2 * np.pi, 60)
    poly(np.stack([PITCH_L / 2 + 9.15 * np.cos(a), PITCH_W / 2 + 9.15 * np.sin(a)], 1), (255, 0, 255))
    return out
