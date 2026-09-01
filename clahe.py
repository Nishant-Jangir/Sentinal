"""
CLAHE (Contrast Limited Adaptive Histogram Equalization).

Applied on the L channel of LAB color space only -- applying it directly on
BGR/RGB channels independently causes color shifts/artifacts.
"""

import cv2
import numpy as np


class CLAHEEnhancer:
    def __init__(self, clip_limit: float = 2.0, tile_grid=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_eq = self.clahe.apply(l_channel)
        merged = cv2.merge((l_eq, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
