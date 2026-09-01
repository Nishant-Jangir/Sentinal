"""
CameraWorker: owns one camera feed's state and runs the tiered pipeline:

  LIVE tier (every frame):   Zero-DCE -> CLAHE           (cheap, always on)
  HEAVY tier (gated):        Real-ESRGAN on ROI or frame (expensive, triggered
                              by motion or manual "inspect" request, and rate
                              limited by a global semaphore so only
                              MAX_CONCURRENT_HEAVY_JOBS run on the GPU at once)
"""

import time
import cv2
import numpy as np

import config
from core.motion_detector import MotionDetector


class CameraWorker:
    def __init__(self, camera_id: int, zero_dce, clahe, real_esrgan, heavy_semaphore):
        self.camera_id = camera_id
        self.zero_dce = zero_dce
        self.clahe = clahe
        self.real_esrgan = real_esrgan
        self.heavy_semaphore = heavy_semaphore  # threading.Semaphore shared across workers

        self.motion_detector = MotionDetector(min_area=config.MOTION_MIN_AREA)
        self.frame_count = 0
        self.manual_inspect_requested = False
        self.last_heavy_output = None  # cached last enhanced ROI/frame for UI display

    def request_inspect(self):
        """Called by the UI/operator to force a HEAVY-tier pass on next frame."""
        self.manual_inspect_requested = True

    def process_frame(self, raw_frame: np.ndarray) -> dict:
        """
        Runs the tiered pipeline on one incoming frame.
        Returns a dict with the live-quality frame always, and a heavy-tier
        result only when one was actually computed this call.
        """
        self.frame_count += 1
        frame = cv2.resize(raw_frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        # ---- LIVE tier: cheap, always runs ----
        live_frame = frame
        if config.LIVE_APPLY_ZERO_DCE:
            live_frame = self.zero_dce.enhance(live_frame)
        if config.LIVE_APPLY_CLAHE:
            live_frame = self.clahe.enhance(live_frame)

        result = {
            "camera_id": self.camera_id,
            "live_frame": live_frame,
            "heavy_frame": None,
            "heavy_triggered_by": None,
        }

        # ---- Decide whether HEAVY tier should run this frame ----
        motion_found, roi = (False, None)
        if config.MOTION_TRIGGERS_HEAVY:
            motion_found, roi = self.motion_detector.detect(live_frame)

        should_run_heavy = False
        trigger_reason = None

        if self.manual_inspect_requested:
            should_run_heavy = True
            trigger_reason = "manual"
            self.manual_inspect_requested = False
        elif motion_found and (self.frame_count % config.HEAVY_FRAME_SKIP == 0):
            should_run_heavy = True
            trigger_reason = "motion"

        if not should_run_heavy:
            return result

        # ---- HEAVY tier: gated by global semaphore so we never exceed
        #      MAX_CONCURRENT_HEAVY_JOBS on the GPU at once, across all 30 cams ----
        acquired = self.heavy_semaphore.acquire(blocking=False)
        if not acquired:
            # GPU is busy with other cameras' heavy jobs right now -- skip this
            # round rather than blocking and backing up the live feed.
            return result

        try:
            roi_arg = roi if (config.REAL_ESRGAN_ROI_ONLY and roi is not None) else None
            heavy_out = self.real_esrgan.enhance(live_frame, roi=roi_arg)
            result["heavy_frame"] = heavy_out
            result["heavy_triggered_by"] = trigger_reason
            self.last_heavy_output = heavy_out
        finally:
            self.heavy_semaphore.release()

        return result
