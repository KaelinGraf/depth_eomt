"""Render grasp candidates overlaid on the RGB image of a scene.

For each (frame, instance) JSON, sample N grasps (default 10) balanced between
valid and invalid, project to image space via the camera intrinsics, and draw:
  - An arrow from a point along the *approach axis behind the grasp* (the
    gripper-base direction) to the grasp midpoint (where the part sits between
    the fingers). Tail = gripper base ish; head = the part's grasp point.
  - Two small squares at the finger contact points.

Green = valid (object_in_gripper True). Red = invalid (False).

Usage:
    python -m scripts.grasp_dataset.visualize \\
        /path/to/grasp_dataset/monocular/val/frame_01018__inst_*.grasps.json \\
        --out inference_outputs/grasps_frame_1018.png
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# Make the package importable when run as `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.grasp_dataset.gripper import GRIPPER_DEPTH  # noqa: E402


def _project(K: np.ndarray, p_cam: np.ndarray) -> np.ndarray:
    """Project a (N, 3) point set from camera frame to image pixel coords (N, 2)."""
    p = p_cam.reshape(-1, 3)
    uvw = p @ K.T  # (N, 3): [u*z, v*z, z]
    z = uvw[:, 2:3]
    z_safe = np.where(np.abs(z) > 1e-9, z, 1e-9)
    return uvw[:, :2] / z_safe


def _balanced_sample(in_gripper: list[bool], n_total: int = 10, rng=None):
    """Return up to n_total indices balanced between True/False (5/5 or as close)."""
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.asarray(in_gripper, dtype=bool)
    valid_idx = np.where(arr)[0]
    invalid_idx = np.where(~arr)[0]
    n_each = n_total // 2
    sel_valid = rng.choice(valid_idx, min(n_each, len(valid_idx)), replace=False) if len(valid_idx) else np.array([], dtype=int)
    sel_invalid = rng.choice(invalid_idx, min(n_total - len(sel_valid), len(invalid_idx)), replace=False) if len(invalid_idx) else np.array([], dtype=int)
    # If one side is short, top up the other side
    n_short = n_total - len(sel_valid) - len(sel_invalid)
    if n_short > 0:
        pool = valid_idx if len(invalid_idx) < len(valid_idx) else invalid_idx
        sel_extra = rng.choice(pool, min(n_short, len(pool)), replace=False)
        if pool is valid_idx:
            sel_valid = np.concatenate([sel_valid, sel_extra])
        else:
            sel_invalid = np.concatenate([sel_invalid, sel_extra])
    return sel_valid, sel_invalid


def _load_rgb(frame_dir: Path) -> np.ndarray:
    img = Image.open(frame_dir / "rgb.png").convert("RGB")
    return np.asarray(img)


def _draw_one_grasp(ax, grasp_obj_4x4, T_obj2cam, K, color, width_m,
                    approach_stub_m=0.03, lw=1.8, alpha=0.95, image_size=None):
    """Draw one grasp on `ax`. Visualisation anchored at the GRASP MIDPOINT
    (where the part sits between the fingers), NOT the gripper base (which is
    GRIPPER_DEPTH=19.5 cm behind the grasp along approach — drawing the base
    as the arrow tail makes grasps look like they're in empty space far from
    the part, even though the grasp itself is correctly on the part).

    Drawn elements (all near the part):
      - Short approach STUB (~3 cm), tail in air, arrow head AT midpoint
      - Line segment between the two finger contact points (closing line)
      - Filled circles at the contact points themselves
    """
    g_cam = T_obj2cam @ np.asarray(grasp_obj_4x4)
    origin_cam = g_cam[:3, 3]   # gripper base
    x_axis_cam = g_cam[:3, 0]   # closing
    z_axis_cam = g_cam[:3, 2]   # approach
    midpoint_cam = origin_cam + GRIPPER_DEPTH * z_axis_cam

    # Skip grasps behind / very close to camera
    if midpoint_cam[2] <= 0.05:
        return False

    # Short approach stub: starts approach_stub_m BEFORE midpoint along approach
    stub_tail_cam = midpoint_cam - approach_stub_m * z_axis_cam
    pts3d = np.stack([
        stub_tail_cam,
        midpoint_cam,
        midpoint_cam + 0.5 * width_m * x_axis_cam,   # +X finger contact
        midpoint_cam - 0.5 * width_m * x_axis_cam,   # -X finger contact
    ])
    pts2d = _project(K, pts3d)
    p_stub, p_mid, p_left, p_right = pts2d

    # Bounds-check on midpoint only (the visual anchor)
    if image_size is not None:
        W, H = image_size
        if not (-20 <= p_mid[0] <= W + 20 and -20 <= p_mid[1] <= H + 20):
            return False

    # Short approach stub: tail in air → head AT midpoint
    ax.add_patch(FancyArrowPatch(
        (p_stub[0], p_stub[1]),
        (p_mid[0], p_mid[1]),
        color=color, arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, alpha=alpha,
    ))
    # Closing line between the two contact points
    ax.plot([p_left[0], p_right[0]], [p_left[1], p_right[1]],
            color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")
    # Filled dots at the two contact points (where the gripper actually touches)
    for p in (p_left, p_right):
        ax.plot(p[0], p[1], marker='o', markersize=4,
                markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=0.7, alpha=alpha)
    return True


def visualize_grasps(
    json_path: Path,
    out_path: Path,
    *,
    n_grasps_to_draw: int = 10,
    seed: int = 0,
):
    """Render the grasps from one (frame, instance) JSON onto its RGB image."""
    with open(json_path) as f:
        d = json.load(f)
    meta = d["scene_metadata"]
    K = np.asarray(meta["camera_intrinsics"], dtype=np.float64).reshape(3, 3)
    R = np.asarray(meta["object_pose_cam"]["R_m2c"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(meta["object_pose_cam"]["t_m2c"], dtype=np.float64).reshape(3)
    T_obj2cam = np.eye(4); T_obj2cam[:3, :3] = R; T_obj2cam[:3, 3] = t

    # Load RGB image from the scene_info path's parent dir
    frame_dir = Path(meta["scene_info_path"]).parent
    rgb = _load_rgb(frame_dir)
    H, W = rgb.shape[:2]

    transforms = np.asarray(d["grasps"]["transforms"], dtype=np.float64)
    widths = np.asarray(d["grasps"]["widths"], dtype=np.float64)
    in_gripper = d["grasps"]["object_in_gripper"]
    reasons = d["grasps"].get("filter_reasons", [""] * len(in_gripper))

    rng = np.random.default_rng(seed)
    sel_valid, sel_invalid = _balanced_sample(in_gripper, n_grasps_to_draw, rng=rng)
    n_valid_avail = sum(in_gripper)
    n_invalid_avail = len(in_gripper) - n_valid_avail

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(rgb)
    drawn_v, drawn_i = 0, 0
    for idx in sel_valid:
        if _draw_one_grasp(ax, transforms[idx], T_obj2cam, K, "#2ECC71",
                           width_m=float(widths[idx]), lw=1.6, image_size=(W, H)):
            drawn_v += 1
    for idx in sel_invalid:
        if _draw_one_grasp(ax, transforms[idx], T_obj2cam, K, "#E74C3C",
                           width_m=float(widths[idx]), lw=1.2, alpha=0.8, image_size=(W, H)):
            drawn_i += 1

    # Title
    fid = meta["frame_id"]; inst = meta["instance_seg_id"]; obj = meta["object_name"][:40]
    vis = meta.get("object_visibility_ratio", 0.0)
    title = (f"frame {fid} inst {inst} ({obj})\n"
             f"vis={vis:.2f} · drawn green={drawn_v}/{n_valid_avail} valid · "
             f"red={drawn_i}/{n_invalid_avail} invalid")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_paths", nargs="+", type=Path,
                    help="One or more *.grasps.json paths")
    ap.add_argument("--out-dir", type=Path, default=Path("inference_outputs/grasp_vis"))
    ap.add_argument("--n", type=int, default=10, help="grasps to draw per frame")
    args = ap.parse_args()
    for p in args.json_paths:
        out = args.out_dir / (p.stem + ".png")
        visualize_grasps(p, out, n_grasps_to_draw=args.n)


if __name__ == "__main__":
    main()
