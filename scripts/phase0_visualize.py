"""Generate a depth viz on a held-out val frame using the Phase-0 checkpoint.

Run AFTER scripts/phase0_freeze_train.py finishes.

Pulls the latest checkpoint, builds EoMTInference around it, runs on the
first frame in the val split, and saves the 4-panel viz (with depth) to
scripts/phase0_outputs/heldout_depth_viz.png. Also saves a side-by-side of
predicted vs GT depth as heldout_pred_vs_gt.png so qualitative correlation
can be eyeballed against the criterion in #12.
"""
from __future__ import annotations

import os
import sys
import json
import glob
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/kaelin/BinPicking/eomt"
sys.path.insert(0, REPO)
os.chdir(REPO)

from inference import EoMTInference

VAL_DIR = Path("/home/kaelin/BinPicking/SDG/IS/Outputs/monocular_dataset/val")
OUTPUTS = Path(REPO) / "scripts" / "phase0_outputs"


def find_latest_ckpt() -> str:
    candidates = sorted(OUTPUTS.glob("checkpoints/phase0-*.ckpt"))
    if not candidates:
        # Fall back to last.ckpt
        last = OUTPUTS / "checkpoints" / "last.ckpt"
        if last.exists():
            return str(last)
        raise FileNotFoundError(f"No phase0 checkpoint under {OUTPUTS}/checkpoints/")
    return str(candidates[-1])


def main():
    ckpt = find_latest_ckpt()
    print(f"Using checkpoint: {ckpt}")

    inf = EoMTInference(
        ckpt_path=ckpt,
        device="cuda",
        img_size=(1280, 1280),
        num_classes=2,
        stuff_classes=[0],
        mask_thresh=0.01,
        overlap_thresh=0.1,
    )

    # Pick the first val frame.
    frames = sorted([d for d in VAL_DIR.iterdir() if d.is_dir() and d.name.startswith("frame_")])
    frame = frames[0]
    print(f"Held-out frame: {frame}")

    img_bgr = cv2.imread(str(frame / "rgb.png"))
    depth_gt = np.load(frame / "depth.npy").astype(np.float32)

    # Load intrinsics.
    K = None
    info_path = next(frame.glob("*scene_info.json"))
    with open(info_path) as f:
        info = json.load(f)
    if isinstance(info, dict) and "camera" in info and "cam_K" in info["camera"]:
        K = np.asarray(info["camera"]["cam_K"], dtype=np.float32).reshape(3, 3)
    print(f"Intrinsics K provided: {K is not None}")

    result = inf(img_bgr, intrinsics=K)

    # Save the standard 4-panel viz (includes depth panel from chunk-#14).
    panel_path = OUTPUTS / "heldout_depth_viz.png"
    inf.visualize(result, img_bgr, save_path=str(panel_path), show=False)
    print(f"Saved panel viz: {panel_path}")

    # Predicted vs GT depth side-by-side, plus an error map.
    pred = result.depth
    gt = depth_gt

    # Mask GT validity for stats.
    valid = np.isfinite(gt) & (gt > 0) & (gt < 10.0)
    if valid.any():
        rmse = float(np.sqrt(((pred[valid] - gt[valid]) ** 2).mean()))
        abs_rel = float((np.abs(pred[valid] - gt[valid]) / gt[valid]).mean())
        print(f"RMSE (valid pixels): {rmse:.3f} m")
        print(f"AbsRel: {abs_rel:.3f}")
    else:
        rmse = float("nan")
        abs_rel = float("nan")

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    im0 = axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    im1 = axes[1].imshow(gt, cmap="turbo", vmin=np.nanmin(gt[valid]) if valid.any() else 0.0,
                         vmax=np.nanmax(gt[valid]) if valid.any() else 1.0)
    axes[1].set_title("GT depth (m)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred, cmap="turbo",
                         vmin=np.nanmin(gt[valid]) if valid.any() else 0.0,
                         vmax=np.nanmax(gt[valid]) if valid.any() else 1.0)
    axes[2].set_title("Predicted depth (m)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    err = np.abs(pred - gt)
    err[~valid] = 0.0
    im3 = axes[3].imshow(err, cmap="magma")
    axes[3].set_title(f"|Pred - GT| (RMSE={rmse:.3f}m, AbsRel={abs_rel:.3f})")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    side_path = OUTPUTS / "heldout_pred_vs_gt.png"
    plt.savefig(str(side_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pred-vs-gt viz: {side_path}")


if __name__ == "__main__":
    main()
