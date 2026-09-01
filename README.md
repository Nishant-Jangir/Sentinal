# CCTV Enhancement Pipeline

Zero-DCE → CLAHE → Real-ESRGAN, tiered for real-time use across many feeds.

## Architecture

- **LIVE tier** (every frame, every camera): Zero-DCE (exposure) → CLAHE (contrast).
  Cheap enough to run on all 30 feeds continuously.
- **HEAVY tier** (gated): Real-ESRGAN super-resolution, triggered by motion
  detection or a manual "inspect this camera" request, rate-limited by
  `config.MAX_CONCURRENT_HEAVY_JOBS` so it never overwhelms the GPU.

```
Raw frame → Zero-DCE → CLAHE → [motion/manual trigger?] → Real-ESRGAN (ROI or full) → output
```

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # if using system python
```

### Model weights

1. **Real-ESRGAN**: download `RealESRGAN_x4plus.pth` from the official repo release
   page (search "Real-ESRGAN x4plus weights github release") and place at
   `weights/RealESRGAN_x4plus.pth`.
2. **Zero-DCE**: download pretrained weights from the official Zero-DCE repo
   (search "Zero-DCE pytorch pretrained weights github"), place at
   `weights/zero_dce.pth`. Note the architecture in `models/zero_dce.py` is a
   from-scratch reimplementation matching the paper — if the official weights'
   layer names don't match exactly, you may need to write a small key-remapping
   script, or just retrain Zero-DCE yourself (it trains fast, no paired data needed).

The code runs without weights present (prints a warning and no-ops / returns
random output), so you can test the pipeline architecture end-to-end before
weights are sorted.

### Test feeds

`main.py` currently points all 30 "cameras" at one sample video file for
demo purposes. For the hackathon, either:
- Use a handful of real Gujarat Police camera feed URLs if you have access, or
- Duplicate 2-3 different sample CCTV-style videos across the 30 slots
  (varying content is more convincing in a demo than 30 identical feeds).

Run with: `python main.py`

## Hardware reality check (Legion 5i, RTX 4060, 30 feeds)

**LIVE tier (Zero-DCE + CLAHE) on 30 feeds simultaneously: yes, feasible.**
Zero-DCE is a ~79K parameter network; both stages combined should comfortably
hit real-time (25-30 FPS) across 30 streams at 640x360 on a 4060.

**HEAVY tier (Real-ESRGAN) on all 30 feeds simultaneously in real time: no.**
Real-ESRGAN x4plus on a single 4060 laptop GPU typically runs ~10-20 FPS on
one ~480p stream with fp16 + tiling. Running that live across 30 streams
would need roughly 15-30x more throughput than one 4060 provides. This isn't
a code problem — it's a hardware ceiling, and no amount of optimization in
this codebase changes the order of magnitude.

**What this pipeline actually does instead** (see `config.py`):
- Runs Real-ESRGAN only on cameras/frames where motion was detected
  (`MOTION_TRIGGERS_HEAVY`), and only on the motion ROI, not the full frame
  (`REAL_ESRGAN_ROI_ONLY`) — this cuts pixel count and job frequency drastically.
- Caps concurrent GPU jobs at `MAX_CONCURRENT_HEAVY_JOBS` (start at 2, raise
  it during testing while watching `nvidia-smi` for VRAM/util headroom).
- Supports on-demand "inspect this camera" (`manager.request_inspect(cam_id)`)
  for an operator to trigger full enhancement on one feed at a time — this is
  the strongest hackathon demo moment: click a camera, watch it go from
  grainy to sharp in ~1-2 seconds.

**Pitch framing that matches the engineering**: "always-on lightweight
enhancement for monitoring all 30 feeds, full forensic-grade super-resolution
on-demand for investigation" is both more honest and a better story than
claiming full real-time SR across all feeds.

## Tuning checklist before the demo

- [ ] Confirm actual FPS on LIVE tier with all 30 threads running (`nvidia-smi -l 1`
      in another terminal while `main.py` runs)
- [ ] Tune `MAX_CONCURRENT_HEAVY_JOBS` up until VRAM/util maxes out, then back off by 1
- [ ] Tune `HEAVY_FRAME_SKIP` and `MOTION_MIN_AREA` to avoid heavy-tier thrashing
      on tiny/noisy motion
- [ ] Wire `on_result()` in `main.py` to an actual display (OpenCV window grid
      or a simple web dashboard) instead of just printing
