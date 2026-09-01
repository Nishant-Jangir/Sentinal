"""
Lightweight motion detector using background subtraction.
Cheap enough to run on all 30 feeds continuously on CPU.
"""

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, min_area: int = 500):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False
        )
        self.min_area = min_area

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns (motion_detected: bool, roi: (x,y,w,h) or None)
        roi is the bounding box of the largest motion contour, useful for
        feeding directly into Real-ESRGAN's ROI mode.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = self.bg_subtractor.apply(gray)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return False, None

        x, y, w, h = cv2.boundingRect(largest)
        return True, (x, y, w, h)
