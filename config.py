"""
Central configuration for the CCTV enhancement pipeline.
Tune these values based on your GPU headroom during testing.
"""

import torch

# ---------- Hardware ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = True  # half precision -- important for speed on 4060 (8GB VRAM)

# ---------- Camera feeds ----------
NUM_CAMERAS = 30
FRAME_WIDTH = 640      # downscale incoming feeds to this before processing
FRAME_HEIGHT = 360

# ---------- Tiering strategy ----------
# "LIVE" tier: cheap ops only, runs on every frame of every camera.
# "HEAVY" tier: Real-ESRGAN, runs only when triggered (motion / on-demand).
LIVE_APPLY_ZERO_DCE = True
LIVE_APPLY_CLAHE = True

# How many cameras' HEAVY tier can run concurrently on the GPU at once.
# This is your real concurrency limit -- tune by watching VRAM/util during testing.
MAX_CONCURRENT_HEAVY_JOBS = 2

# Only run HEAVY tier every N frames per camera even when triggered
# (avoids re-enhancing near-identical consecutive frames).
HEAVY_FRAME_SKIP = 5

# Motion detection trigers HEAVY tier automatically. Also allow manual
# on-demand trigger (operator clicks a camera) -- see core/manager.py
MOTION_TRIGGERS_HEAVY = True
MOTION_MIN_AREA = 500  # pixels; ignore tiny noise-level motion

# ---------- Real-ESRGAN ----------
REAL_ESRGAN_MODEL_PATH = "weights/RealESRGAN_x4plus.pth"
REAL_ESRGAN_OUTSCALE = 2   # use 2x not 4x for speed; upscale further only on-demand
REAL_ESRGAN_TILE = 200     # tile inference to bound VRAM use
REAL_ESRGAN_ROI_ONLY = True  # if True, only super-res the motion/detection ROI, not full frame

# ---------- Zero-DCE ----------
ZERO_DCE_MODEL_PATH = "weights/zero_dce.pth"

# ---------- CLAHE ----------
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
