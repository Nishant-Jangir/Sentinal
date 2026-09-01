"""
Zero-DCE (Zero-Reference Deep Curve Estimation) for low-light enhancement.

Architecture: 7 conv layers predict 8 sets of curve parameter maps (24 channels
total, 3 per curve x 8 curves), which are iteratively applied to the input
image to brighten it without needing paired ground-truth data.

Reference: Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light
Image Enhancement", CVPR 2020. This is a from-scratch reimplementation of the
published architecture, not copied code.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2


class DCENet(nn.Module):
    def __init__(self, num_filters: int = 32):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, num_filters, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        # Skip connections concat features -> double channels in
        self.e_conv5 = nn.Conv2d(num_filters * 2, num_filters, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(num_filters * 2, num_filters, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(num_filters * 2, 24, 3, 1, 1)  # 8 curves x 3 channels

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        curves = torch.split(x_r, 3, dim=1)  # 8 tensors of shape [B,3,H,W]
        enhanced = x
        for curve in curves:
            enhanced = enhanced + curve * (torch.pow(enhanced, 2) - enhanced)
        return enhanced, x_r


class ZeroDCEEnhancer:
    """Convenience wrapper: load once, call .enhance(frame) per-frame (BGR uint8 in/out)."""

    def __init__(self, weights_path: str, device: torch.device, use_fp16: bool = True):
        self.device = device
        self.use_fp16 = use_fp16 and device.type == "cuda"
        self.model = DCENet().to(device)
        try:
            state_dict = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"[Zero-DCE] Loaded weights from {weights_path}")
        except FileNotFoundError:
            print(f"[Zero-DCE] WARNING: weights not found at {weights_path}. "
                  f"Using randomly initialized weights -- output will look wrong "
                  f"until you download/train real weights. See README.md.")
        self.model.eval()
        if self.use_fp16:
            self.model.half()

    @torch.no_grad()
    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.use_fp16:
            tensor = tensor.half()
        enhanced, _ = self.model(tensor)
        out = enhanced.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
