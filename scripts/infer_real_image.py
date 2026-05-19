"""Run EoMT depth inference on a single real image with explicit camera
intrinsics, then open an interactive Open3D point-cloud viewer.

Usage:
    python scripts/infer_real_image.py <image_path> --ckpt <ckpt.ckpt> \\
        --fx <fx> --fy <fy> --cx <cx> --cy <cy> [--out <dir>] [--no-show]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import EoMTInference  # noqa: E402
from scripts.depth_to_pointcloud import (  # noqa: E402
    unproject, normals_from_depth, normal_to_rgb, make_pcd, filter_flat,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--out", type=Path, default=Path("/tmp/pc_real"))
    ap.add_argument("--color", choices=("normal", "rgb"), default="normal")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], dtype=np.float64)
    print(f"image: {args.image}")
    print(f"intrinsics: fx={args.fx:.2f} fy={args.fy:.2f} cx={args.cx:.2f} cy={args.cy:.2f}")

    model = EoMTInference(ckpt_path=str(args.ckpt), device=args.device,
                          img_size=(args.img_size, args.img_size))
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not load: {args.image}")
    print(f"image shape: {image.shape}")
    result = model(image, intrinsics=K)
    if result.depth is None:
        raise SystemExit("Model has no depth output enabled.")
    print(f"pred depth: min={result.depth.min():.3f}  max={result.depth.max():.3f}  "
          f"median={np.median(result.depth):.3f} (m)")

    # Back-project pred depth into a coloured point cloud.
    pts = unproject(result.depth, K)
    if args.color == "normal":
        n = normals_from_depth(pts)
        cols = normal_to_rgb(n)
    else:  # rgb
        cols = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    pts_f = pts.reshape(-1, 3)
    cols_f = cols.reshape(-1, 3)
    pts_f, cols_f = filter_flat(pts_f, cols_f)
    pcd = make_pcd(pts_f, cols_f)

    out_ply = args.out / "pred_real.ply"
    import open3d as o3d
    o3d.io.write_point_cloud(str(out_ply), pcd)
    print(f"wrote {out_ply} ({len(pts_f)} pts)  [colour: {args.color}]")

    if not args.no_show:
        print("opening Open3D window — close it to exit")
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
