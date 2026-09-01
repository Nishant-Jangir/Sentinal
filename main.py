"""
Entry point. For the hackathon demo:
  - Replace CAMERA_SOURCES with your actual RTSP URLs / video file paths.
  - If you don't have 30 real feeds available, point several entries at the
    same test video -- it's fine for demoing the architecture and load
    behavior, and is a completely normal thing to do for a hackathon demo.
"""

import cv2
import time

import config
from models.zero_dce import ZeroDCEEnhancer
from models.real_esrgan_wrapper import RealESRGANEnhancer
from processing.clahe import CLAHEEnhancer
from core.manager import PipelineManager

# TODO: replace with real RTSP URLs, e.g. "rtsp://user:pass@ip:554/stream1"
CAMERA_SOURCES = [f"sample_videos/demo_feed.mp4" for _ in range(config.NUM_CAMERAS)]


def on_result(result: dict):
    """
    Simple demo callback: just print when heavy tier fires.
    Replace this with actual display (e.g. write to a Flask/OpenCV window
    grid, or push frames to a web dashboard via websockets).
    """
    if result["heavy_frame"] is not None:
        print(f"[Cam {result['camera_id']}] HEAVY tier ran "
              f"(trigger={result['heavy_triggered_by']})")


def main():
    print(f"Device: {config.DEVICE}, FP16: {config.USE_FP16}")

    zero_dce = ZeroDCEEnhancer(config.ZERO_DCE_MODEL_PATH, config.DEVICE, config.USE_FP16)
    clahe = CLAHEEnhancer(config.CLAHE_CLIP_LIMIT, config.CLAHE_TILE_GRID)
    real_esrgan = RealESRGANEnhancer(
        config.REAL_ESRGAN_MODEL_PATH, config.DEVICE,
        outscale=config.REAL_ESRGAN_OUTSCALE, tile=config.REAL_ESRGAN_TILE,
        use_fp16=config.USE_FP16,
    )

    manager = PipelineManager(CAMERA_SOURCES, zero_dce, clahe, real_esrgan, on_result=on_result)
    manager.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        manager.stop()


if __name__ == "__main__":
    main()
