"""
Real-ESRGAN wrapper for super-resolution + artifact removal.

Requires: pip install realesrgan basicsr
Weights: download RealESRGAN_x4plus.pth into weights/ (see README.md)

This is the expensive stage -- keep calls to .enhance() gated by the
tiering/motion logic in core/camera_worker.py, don't call it on every frame.
"""

import numpy as np
import torch


class RealESRGANEnhancer:
    def __init__(self, model_path: str, device: torch.device,
                 outscale: int = 2, tile: int = 200, use_fp16: bool = True):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        self.outscale = outscale
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                         num_block=23, num_grow_ch=32, scale=4)
        try:
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=use_fp16 and device.type == "cuda",
                device=device,
            )
            print(f"[Real-ESRGAN] Loaded weights from {model_path}")
        except FileNotFoundError:
            self.upsampler = None
            print(f"[Real-ESRGAN] WARNING: weights not found at {model_path}. "
                  f"Heavy-tier enhancement will be skipped until weights are added. "
                  f"See README.md for download link.")

    def enhance(self, frame_bgr: np.ndarray, roi: tuple = None) -> np.ndarray:
        """
        frame_bgr: full frame, uint8 BGR
        roi: optional (x, y, w, h) -- if given, only this region is super-resolved
             and pasted back (much cheaper than full-frame SR).
        Returns the frame with the (possibly ROI-limited) enhancement applied.
        Note: if roi is used, output frame keeps original size except the ROI
        patch, which is upscaled then resized back down to fit -- for a
        "zoom and enhance" UI you'd instead want to return the raw upscaled
        crop directly. See core/camera_worker.py for both usage patterns.
        """
        if self.upsampler is None:
            return frame_bgr  # graceful no-op if weights aren't present yet

        if roi is not None:
            x, y, w, h = roi
            crop = frame_bgr[y:y + h, x:x + w]
            enhanced_crop, _ = self.upsampler.enhance(crop, outscale=self.outscale)
            return enhanced_crop  # caller decides how to display/store this

        enhanced_full, _ = self.upsampler.enhance(frame_bgr, outscale=self.outscale)
        return enhanced_full
