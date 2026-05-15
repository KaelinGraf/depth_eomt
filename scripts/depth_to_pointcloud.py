#!/usr/bin/env python3
"""
Back-project EoMT's predicted depth into a 3D point cloud and compare it
against the ground-truth depth, both unprojected with the frame's real
camera intrinsics.

Outputs (into --out dir):
  - pred.ply / gt.ply   : coloured point clouds for an interactive viewer
  - pointcloud_cmp.png  : offscreen-rendered pred-vs-GT snapshots (inline-viewable)

Usage:
  python3 scripts/depth_to_pointcloud.py <frame_dir> --ckpt <ckpt> [--out /tmp/pc] [--show]

  <frame_dir> is a Replicator frame_* dir (needs rgb.png + depth.npy +
  *scene_info*.json). --show additionally opens an interactive Open3D
  window (blocking — run it backgrounded).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_intrinsics(frame_dir: Path) -> np.ndarray:
    """[3,3] K from the Replicator scene-info JSON (cam_K, row-major)."""
    j = sorted(frame_dir.glob("*scene_info*.json"))[0]
    cam = json.load(open(j))["camera"]
    return np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3)


def unproject(depth: np.ndarray, K: np.ndarray, rgb: np.ndarray,
              dmin: float = 0.2, dmax: float = 2.5):
    """Pinhole back-projection. depth is z-buffer (distance_to_image_plane),
    so Z = depth and X/Y follow directly. Points outside [dmin, dmax] m are
    dropped — bin scenes are < 2 m, so anything beyond is an outlier that
    would otherwise blow up the render's bounding box. Returns
    (points[N,3], colors[N,3])."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    Z = depth.astype(np.float64)
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    cols = (rgb.reshape(-1, 3).astype(np.float64) / 255.0)
    valid = np.isfinite(pts).all(1) & (pts[:, 2] > dmin) & (pts[:, 2] < dmax)
    return pts[valid], cols[valid]


def make_pcd(pts, cols):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def render_snapshots(pcd_pred, pcd_gt, save_path: Path):
    """Offscreen-render pred and GT from two angles into one PNG.

    Uses the legacy Visualizer with a hidden window (needs an X display,
    which this box has at DISPLAY=:1). Falls back to a matplotlib 3D
    scatter if the GL path is unavailable.
    """
    import open3d as o3d

    try:
        shots = []
        for pcd in (pcd_pred, pcd_gt):
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=900, height=900)
            vis.add_geometry(pcd)
            vc = vis.get_view_control()
            vc.set_front([0.0, -0.4, -1.0])
            vc.set_up([0.0, -1.0, 0.0])
            vc.set_zoom(0.7)
            for _ in range(2):  # let the renderer settle
                vis.poll_events()
                vis.update_renderer()
            img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            shots.append(img)
            vis.destroy_window()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        ax[0].imshow(shots[0]); ax[0].set_title("Predicted depth -> point cloud", fontsize=14)
        ax[1].imshow(shots[1]); ax[1].set_title("Ground-truth depth -> point cloud", fontsize=14)
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return "open3d"
    except Exception as e:  # GL context unavailable -> matplotlib fallback
        print(f"  open3d offscreen render failed ({e!r}); using matplotlib scatter")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(18, 9))
        for i, (pcd, title) in enumerate(
            [(pcd_pred, "Predicted"), (pcd_gt, "Ground truth")]
        ):
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors)
            if len(pts) > 40000:  # downsample for a responsive scatter
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame_dir", type=Path, help="Replicator frame_* directory")
    ap.add_argument("--ckpt", type=str, required=True, help="EoMT .ckpt path")
    ap.add_argument("--out", type=Path, default=Path("/tmp/eomt_pointcloud"))
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--device", type=str, default="cpu", help="cpu keeps clear of a training run on the GPU")
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

    print(f"frame: {fdir}  | rgb {rgb.shape}  gt_depth {gt_depth.shape}")
    print(f"intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    model = EoMTInference(ckpt_path=args.ckpt, device=args.device,
                          img_size=(args.img_size, args.img_size))
    result = model(str(fdir / "rgb.png"))          # auto-loads K from scene_info
    pred_depth = result.depth.astype(np.float32)   # [H, W] metres, original-res

    # Scale-offset diagnostic: SI-Log is scale-invariant-ish, so the pred
    # cloud may sit at a slightly different scale than GT.
    m = np.isfinite(gt_depth) & (gt_depth > 0)
    ratio = float(np.median(pred_depth[m] / gt_depth[m]))
    print(f"pred/GT median depth ratio: {ratio:.3f}   "
          f"(pred {pred_depth[m].mean():.3f}m  gt {gt_depth[m].mean():.3f}m)")

    pred_pcd = make_pcd(*unproject(pred_depth, K, rgb))
    gt_pcd = make_pcd(*unproject(gt_depth, K, rgb))

    import open3d as o3d

    o3d.io.write_point_cloud(str(args.out / "pred.ply"), pred_pcd)
    o3d.io.write_point_cloud(str(args.out / "gt.ply"), gt_pcd)
    print(f"wrote {args.out}/pred.ply ({len(pred_pcd.points)} pts) and gt.ply ({len(gt_pcd.points)} pts)")

    if args.show:
        import copy

        gt_vis = copy.deepcopy(gt_pcd)
        gt_vis.translate((0.7, 0.0, 0.0))  # GT shifted right of pred for side-by-side
        print("opening interactive Open3D window — pred (left) vs GT (right); "
              "close the window to exit")
        o3d.visualization.draw_geometries(
            [pred_pcd, gt_vis],
            window_name="EoMT depth: predicted (left) vs ground-truth (right)",
        )
    else:
        snap = args.out / "pointcloud_cmp.png"
        backend = render_snapshots(pred_pcd, gt_pcd, snap)
        print(f"wrote {snap}  (via {backend})")


if __name__ == "__main__":
    main()
