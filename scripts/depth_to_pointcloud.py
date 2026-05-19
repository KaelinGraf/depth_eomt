#!/usr/bin/env python3
"""
Back-project EoMT's predicted depth into a 3D point cloud and compare it
against the ground-truth depth, both unprojected with the frame's real
camera intrinsics.

Points are coloured either by **surface-normal direction** (default — the
informative view for "is the depth sharp or hilly") or by the original RGB
image. With `--color normal`, the pred cloud is coloured by normals derived
*from the predicted depth* (via finite-difference cross product), and the
GT cloud by the loaded `normals.npy` (after the OpenGL→OpenCV flip — see
`training/depth_loss.py` module header). Direction → RGB: (n + 1) / 2.

Outputs (into --out dir):
  - pred.ply / gt.ply   : coloured point clouds for an interactive viewer
  - pointcloud_cmp.png  : offscreen-rendered pred-vs-GT snapshots (inline-viewable)

Usage:
  python3 scripts/depth_to_pointcloud.py <frame_dir> --ckpt <ckpt> \\
      [--out /tmp/pc] [--color normal|rgb] [--show]

  <frame_dir> is a Replicator frame_* dir (needs rgb.png + depth.npy +
  *scene_info*.json; normals.npy is optional but enables the GT normal
  colouring). --show additionally opens an interactive Open3D window
  (blocking — run it backgrounded).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------
# Geometry helpers (numpy mirrors of training/depth_loss.py — kept
# inline so this script has no torch dependency at runtime).
# ---------------------------------------------------------------


def load_intrinsics(frame_dir: Path) -> np.ndarray:
    """[3,3] K from the Replicator scene-info JSON (cam_K, row-major)."""
    j = sorted(frame_dir.glob("*scene_info*.json"))[0]
    cam = json.load(open(j))["camera"]
    return np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3)


def load_gt_normals(frame_dir: Path) -> np.ndarray | None:
    """[H,W,3] GT normals in OpenCV camera coords, or None if absent.

    Replicator emits OpenGL-style normals (X right, Y up, Z toward camera);
    we negate Y and Z to match our OpenCV unprojection convention. Verified
    convention check lives in `training/depth_loss.py`.
    """
    p = frame_dir / "normals.npy"
    if not p.exists():
        return None
    n = np.load(p)[..., :3].astype(np.float32)
    n = n * np.array([1.0, -1.0, -1.0], dtype=np.float32)   # OpenGL → OpenCV
    return n


def unproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole back-projection. Returns points [H, W, 3] in camera coords."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    Z = depth.astype(np.float64)
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return np.stack([X, Y, Z], axis=-1)


def normals_from_depth(points: np.ndarray) -> np.ndarray:
    """[H,W,3] unit normals from a camera-space point map.

    Matches `training/depth_loss.py:depth_to_normals` exactly: forward
    differences for the tangent vectors, then `cross(t_y, t_x)` for the
    outward-pointing normal on the OpenCV convention. Trailing row / column
    are edge-replicated.
    """
    tx = points[:, 1:] - points[:, :-1]    # [H, W-1, 3]
    ty = points[1:, :] - points[:-1, :]    # [H-1, W, 3]
    tx = tx[:-1, :]                        # [H-1, W-1, 3]
    ty = ty[:, :-1]                        # [H-1, W-1, 3]
    n = np.cross(ty, tx, axis=-1)
    norms = np.linalg.norm(n, axis=-1, keepdims=True).clip(min=1e-8)
    n = n / norms
    return np.pad(n, ((0, 1), (0, 1), (0, 0)), mode="edge")


def normal_to_rgb(n: np.ndarray) -> np.ndarray:
    """[-1, 1] unit normals → [0, 1] RGB for visualisation."""
    return (n.astype(np.float64) + 1.0) * 0.5


# ---------------------------------------------------------------
# Point-cloud assembly
# ---------------------------------------------------------------


def filter_flat(points: np.ndarray, colors: np.ndarray,
                dmin: float = 0.2, dmax: float = 2.5):
    """Flatten + drop pixels outside [dmin, dmax] m (outliers blow up the
    render's bounding box). Returns (pts[N,3], cols[N,3])."""
    pts = points.reshape(-1, 3)
    cols = colors.reshape(-1, 3)
    valid = np.isfinite(pts).all(1) & (pts[:, 2] > dmin) & (pts[:, 2] < dmax)
    return pts[valid], cols[valid]


def make_pcd(pts, cols):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0))
    return pcd


# ---------------------------------------------------------------
# Offscreen render (used when --show is not passed)
# ---------------------------------------------------------------


def render_snapshots(pcd_pred, pcd_gt, save_path: Path, color_mode: str):
    """Offscreen-render pred and GT from a 3/4 view into one PNG.

    Uses the legacy Visualizer with a hidden window (needs an X display).
    Lets open3d auto-fit the camera, then tilts a bit so depth relief is
    visible. Falls back to a matplotlib 3D scatter if the GL path fails.
    """
    import open3d as o3d

    try:
        shots = []
        for pcd in (pcd_pred, pcd_gt):
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=900, height=900)
            vis.add_geometry(pcd)
            opt = vis.get_render_option()
            opt.point_size = 1.5
            opt.background_color = np.array([1.0, 1.0, 1.0])
            # Open3D's auto-fit (post add_geometry) is consistent across
            # geometries; manual rotation reacts differently to slightly
            # different point distributions and ends up edge-on on one
            # cloud but not the other. Keep the default — for an oblique
            # 3/4 view, use --show (interactive) instead.
            for _ in range(3):
                vis.poll_events(); vis.update_renderer()
            shots.append(np.asarray(vis.capture_screen_float_buffer(do_render=True)))
            vis.destroy_window()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        suffix = " (normal-coloured)" if color_mode == "normal" else " (RGB)"
        ax[0].imshow(shots[0]); ax[0].set_title("Predicted depth" + suffix, fontsize=14)
        ax[1].imshow(shots[1]); ax[1].set_title("Ground truth" + suffix, fontsize=14)
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return "open3d"
    except Exception as e:
        print(f"  open3d offscreen render failed ({e!r}); using matplotlib scatter")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(18, 9))
        for i, (pcd, title) in enumerate(
            [(pcd_pred, "Predicted"), (pcd_gt, "Ground truth")]
        ):
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors)
            if len(pts) > 40000:
                idx = np.random.default_rng(0).choice(len(pts), 40000, replace=False)
                pts, cols = pts[idx], cols[idx]
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols, s=1, marker=".")
            ax.set_title(f"{title} depth -> point cloud", fontsize=14)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=-70, azim=-90)
        fig.tight_layout()
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return "matplotlib"


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame_dir", type=Path, help="Replicator frame_* directory")
    ap.add_argument("--ckpt", type=str, required=True, help="EoMT .ckpt path")
    ap.add_argument("--out", type=Path, default=Path("/tmp/eomt_pointcloud"))
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--device", type=str, default="cpu", help="cpu keeps clear of a training run on the GPU")
    ap.add_argument("--color", choices=("normal", "rgb"), default="normal",
                    help="point colouring: 'normal' (direction → RGB, sharp edges show as colour discontinuities)"
                         " or 'rgb' (original image colours)")
    ap.add_argument("--show", action="store_true", help="also open an interactive Open3D window (blocking)")
    args = ap.parse_args()

    import cv2
    from inference import EoMTInference

    fdir: Path = args.frame_dir
    args.out.mkdir(parents=True, exist_ok=True)

    K = load_intrinsics(fdir)
    rgb = cv2.cvtColor(cv2.imread(str(fdir / "rgb.png")), cv2.COLOR_BGR2RGB)
    gt_depth = np.load(fdir / "depth.npy").astype(np.float32)
    if gt_depth.ndim == 3:
        gt_depth = gt_depth.squeeze()
    gt_normals = load_gt_normals(fdir)

    print(f"frame: {fdir}  | rgb {rgb.shape}  gt_depth {gt_depth.shape}"
          f"  | gt_normals: {'yes' if gt_normals is not None else 'absent'}"
          f"  | colour mode: {args.color}")
    print(f"intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    model = EoMTInference(ckpt_path=args.ckpt, device=args.device,
                          img_size=(args.img_size, args.img_size))
    result = model(str(fdir / "rgb.png"))
    pred_depth = result.depth.astype(np.float32)

    # Scale-offset diagnostic: SI-Log is scale-invariant-ish, so the pred
    # cloud may sit at a slightly different scale than GT.
    m = np.isfinite(gt_depth) & (gt_depth > 0)
    ratio = float(np.median(pred_depth[m] / gt_depth[m]))
    print(f"pred/GT median depth ratio: {ratio:.3f}   "
          f"(pred {pred_depth[m].mean():.3f}m  gt {gt_depth[m].mean():.3f}m)")

    pred_points = unproject(pred_depth, K)
    gt_points = unproject(gt_depth, K)

    if args.color == "normal":
        pred_cols = normal_to_rgb(normals_from_depth(pred_points))
        # Prefer loaded GT normals (analytic, sharp). If absent, derive
        # from GT depth to stay self-consistent with the pred side.
        gt_n = gt_normals if gt_normals is not None else normals_from_depth(gt_points)
        gt_cols = normal_to_rgb(gt_n)
    else:  # rgb
        pred_cols = rgb.astype(np.float64) / 255.0
        gt_cols = rgb.astype(np.float64) / 255.0

    pred_pcd = make_pcd(*filter_flat(pred_points, pred_cols))
    gt_pcd = make_pcd(*filter_flat(gt_points, gt_cols))

    import open3d as o3d

    o3d.io.write_point_cloud(str(args.out / "pred.ply"), pred_pcd)
    o3d.io.write_point_cloud(str(args.out / "gt.ply"), gt_pcd)
    print(f"wrote {args.out}/pred.ply ({len(pred_pcd.points)} pts) "
          f"and gt.ply ({len(gt_pcd.points)} pts)  [colour: {args.color}]")

    if args.show:
        import copy

        gt_vis = copy.deepcopy(gt_pcd)
        gt_vis.translate((0.7, 0.0, 0.0))
        print("opening interactive Open3D window — pred (left) vs GT (right); "
              "close the window to exit")
        o3d.visualization.draw_geometries(
            [pred_pcd, gt_vis],
            window_name=f"EoMT depth: pred (left) vs GT (right)  [{args.color}]",
        )
    else:
        snap = args.out / "pointcloud_cmp.png"
        backend = render_snapshots(pred_pcd, gt_pcd, snap, color_mode=args.color)
        print(f"wrote {snap}  (via {backend})")


if __name__ == "__main__":
    main()
