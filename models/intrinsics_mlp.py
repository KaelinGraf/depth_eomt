# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import math

import torch
import torch.nn as nn


class IntrinsicsMLP(nn.Module):
    """Maps camera intrinsics + image size to a single cam_token.

    Inspired by UniDepth-V1's camera embedding. Designed to be prepended
    to the ViT patch sequence so depth predictions condition on FOV /
    focal length. See docs/depth_knowledge.md §4.3.
    """

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, K: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        # K: [B, 3, 3] camera intrinsics matrix
        # image_size: (H, W)
        # returns: cam_token [B, 1, embed_dim]
        H, W = image_size
        fx = K[:, 0, 0] / float(W)
        fy = K[:, 1, 1] / float(H)
        cx = K[:, 0, 2] / float(W)
        cy = K[:, 1, 2] / float(H)
        logH = torch.full_like(fx, math.log(float(H)))
        logW = torch.full_like(fx, math.log(float(W)))
        v = torch.stack([fx, fy, cx, cy, logW, logH], dim=-1)
        return self.net(v).unsqueeze(1)
