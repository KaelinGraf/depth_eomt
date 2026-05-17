"""Per-frame scene I/O for the grasp-dataset pipeline.

Loads:
    - scene_info.json (camera intrinsics + per-instance pose & metadata)
    - depth.npy (monocular dataset) OR points.npy (sl dataset)
    - instance mask (raw png with integer per-pixel instance IDs)

Resolves the prim_path → catalog object name → USD asset path mapping.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import glob
import json
import numpy as np
from PIL import Image


@dataclass
class SceneObject:
    """One annotated instance from scene_info.json (only those with visible pixels)."""
    seg_id: int
    obj_class: str            # "part" | "background" | other
    prim_path: str
    catalog_name: str          # e.g. "distractor_6041..." or "cylinder"
    usd_filepath: Optional[str]  # may be None if catalog lookup failed
    R_m2c: np.ndarray         # (3, 3) rotation, object→camera
    t_m2c: np.ndarray         # (3,) translation in metres, camera frame
    scale_m2c: float          # uniform scalar
    canonical_extent: np.ndarray  # (3,) axis-aligned bbox extent in object local frame
    visibility_ratio: float
    occlusion_ratio: float


@dataclass
class Scene:
    """Aggregated per-frame data for grasp generation.

    Coordinate frame note: we work in CAMERA frame throughout (matches
    scene_info's cam_R_m2c / cam_t_m2c convention). World-up / world-down
    directions are derived per-frame via cam_R_w2c and passed where needed.
    """
    dataset: str               # "monocular" | "sl"
    split: str                 # "train" | "val"
    frame_id: int
    frame_dir: Path
    K: np.ndarray              # (3, 3) camera intrinsics in pixels
    image_size: tuple[int, int]  # (W, H)
    cam_origin_world: np.ndarray  # (3,) — origin of camera in working frame = (0,0,0)
    world_down_in_cam: np.ndarray  # (3,) — world -Z direction expressed in cam frame
    instance_mask: np.ndarray  # (H, W) int32 — per-pixel segmentation ID
    depth_or_points: np.ndarray  # monocular: (H, W) float; sl: (H, W, 3) float
    objects: list[SceneObject]
    scene_info_path: Path
    instance_mask_path: Path
    depth_or_points_path: Path


def _load_catalog(objects_json_path: Path = Path("/home/kaelin/BinPicking/SDG/IS/Config/objects.json")) -> dict[str, str]:
    """{catalog_name: usd_filepath} flat dict across bins + parts + distractors."""
    with open(objects_json_path) as f:
        cat = json.load(f)
    out = {}
    for category in cat.values():
        if not isinstance(category, dict):
            continue
        for entry in category.values():
            if not isinstance(entry, dict): continue
            name = entry.get("name")
            usd = entry.get("usd_filepath")
            if name and usd:
                out[name] = usd
    return out


_CATALOG_CACHE: dict[str, str] | None = None


def get_catalog() -> dict[str, str]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        _CATALOG_CACHE = _load_catalog()
    return _CATALOG_CACHE


def _parse_objects(scene_info: dict, catalog: dict[str, str]) -> list[SceneObject]:
    out: list[SceneObject] = []
    for obj in scene_info.get("objects", []):
        try:
            prim = obj["prim_path"]
            catalog_name = prim.split("/")[-2]   # /World/Pools/<name>/part_XXX
            usd = catalog.get(catalog_name)
            R = np.asarray(obj["pose"]["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(obj["pose"]["cam_t_m2c"], dtype=np.float64).reshape(3)
            s = float(obj["pose"]["scale_m2c"][0])
            canon = np.asarray(obj["canonical_extent"], dtype=np.float64).reshape(3)
            out.append(SceneObject(
                seg_id=int(obj["segmentation_id"]),
                obj_class=str(obj.get("class", "unknown")),
                prim_path=prim,
                catalog_name=catalog_name,
                usd_filepath=usd,
                R_m2c=R, t_m2c=t, scale_m2c=s,
                canonical_extent=canon,
                visibility_ratio=float(obj.get("visibility_ratio", 0.0)),
                occlusion_ratio=float(obj.get("occlusion_ratio", 1.0)),
            ))
        except (KeyError, ValueError, IndexError) as e:
            # skip malformed entries (bins with no canonical_extent etc.)
            continue
    return out


def _find_scene_info_path(frame_dir: Path) -> Path:
    cands = sorted(frame_dir.glob("Replicator*scene_info.json"))
    if not cands:
        raise FileNotFoundError(f"No Replicator*scene_info.json in {frame_dir}")
    return cands[0]


def _find_instance_raw_path(frame_dir: Path) -> Path:
    cands = sorted(frame_dir.glob("Replicator*instance_raw.png"))
    if not cands:
        raise FileNotFoundError(f"No Replicator*instance_raw.png in {frame_dir}")
    return cands[0]


def _load_instance_mask(p: Path) -> np.ndarray:
    """Load Replicator's per-pixel int32 instance IDs."""
    arr = np.asarray(Image.open(p))
    if arr.ndim == 3:
        # RGBA viz png — convert via the same packing Replicator uses
        # raw png is uint16 or int32; the *_raw.png is the canonical source
        raise ValueError(
            f"{p} appears to be the RGBA viz, not the raw. "
            "Make sure you pointed at *_instance_raw.png."
        )
    return arr.astype(np.int32)


def load_scene(
    frame_dir: Path | str,
    dataset: str,         # "monocular" or "sl"
    split: str,           # "train" or "val"
) -> Scene:
    """Load one frame's worth of data for grasp generation."""
    frame_dir = Path(frame_dir)
    catalog = get_catalog()

    sinfo_path = _find_scene_info_path(frame_dir)
    with open(sinfo_path) as f:
        scene_info = json.load(f)
    cam = scene_info["camera"]
    K = np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    W, H = cam.get("resolution", [None, None])
    if W is None or H is None:
        # infer from depth
        W = H = None
    # cam_R_w2c rotates a world-frame vector into camera frame; world -Z (gravity
    # down) expressed in cam frame is therefore cam_R_w2c @ [0,0,-1].
    cam_R_w2c = np.asarray(cam.get("cam_R_w2c", np.eye(3).tolist()),
                            dtype=np.float64).reshape(3, 3)
    world_down_in_cam = cam_R_w2c @ np.array([0., 0., -1.])
    world_down_in_cam = world_down_in_cam / max(np.linalg.norm(world_down_in_cam), 1e-9)
    objects = _parse_objects(scene_info, catalog)

    inst_path = _find_instance_raw_path(frame_dir)
    inst_mask = _load_instance_mask(inst_path)

    if dataset == "monocular":
        dp_path = frame_dir / "depth.npy"
        dop = np.load(dp_path).astype(np.float32)
    elif dataset == "sl":
        dp_path = frame_dir / "points.npy"
        dop = np.load(dp_path).astype(np.float32)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    if W is None or H is None:
        H, W = (dop.shape[:2] if dop.ndim >= 2 else (None, None))

    # The "world" frame we work in IS the camera frame (everything in scene_info
    # is already cam_*_m2c). Camera origin is therefore (0,0,0) in this frame.
    cam_origin = np.zeros(3, dtype=np.float64)

    # Extract frame id from dir name "frame_NNNNN"
    try:
        frame_id = int(frame_dir.name.split("_")[-1])
    except ValueError:
        frame_id = -1

    return Scene(
        dataset=dataset, split=split, frame_id=frame_id,
        frame_dir=frame_dir, K=K, image_size=(W, H),
        cam_origin_world=cam_origin,
        world_down_in_cam=world_down_in_cam,
        instance_mask=inst_mask,
        depth_or_points=dop,
        objects=objects,
        scene_info_path=sinfo_path,
        instance_mask_path=inst_path,
        depth_or_points_path=dp_path,
    )


def scene_pointcloud_world(
    scene: Scene,
    max_points: int = 200000,
    exclude_seg_ids: Optional[set[int]] = None,
) -> np.ndarray:
    """Build a (N, 3) point cloud of the visible scene surface in camera frame.

    Args:
        max_points: random subsample cap.
        exclude_seg_ids: if provided, drop pixels whose instance_mask value is in
            this set BEFORE unprojection. Used by the orchestrator to exclude
            the target object's visible surface from the collision proxy (else
            the gripper would always "collide" with the target itself).
    """
    rng = np.random.default_rng(0)
    excl_mask = None
    if exclude_seg_ids:
        excl_mask = np.isin(scene.instance_mask, list(exclude_seg_ids))

    if scene.dataset == "monocular":
        H, W = scene.depth_or_points.shape
        depth = scene.depth_or_points
        valid = np.isfinite(depth) & (depth > 1e-3) & (depth < 10.0)
        if excl_mask is not None:
            # match shapes — instance_mask may differ if depth was at a different
            # resolution. Both should be (H, W); guard with a shape check.
            if excl_mask.shape == valid.shape:
                valid = valid & ~excl_mask
        ys, xs = np.where(valid)
        if len(xs) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        z = depth[ys, xs].astype(np.float64)
        K_inv = np.linalg.inv(scene.K)
        pixels_h = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones_like(xs, dtype=np.float64)], axis=-1)
        rays = pixels_h @ K_inv.T
        pts = rays * z[:, None]
    elif scene.dataset == "sl":
        # points.npy: (H, W, 3), invalid = (-1, -1, -1)
        H, W = scene.depth_or_points.shape[:2]
        pts_grid = scene.depth_or_points.reshape(-1, 3)
        valid = ~np.all(pts_grid == -1.0, axis=1)
        if excl_mask is not None and excl_mask.shape == (H, W):
            valid = valid & ~excl_mask.reshape(-1)
        pts = pts_grid[valid].astype(np.float64)
    else:
        raise ValueError(scene.dataset)

    if len(pts) > max_points:
        sel = rng.choice(len(pts), max_points, replace=False)
        pts = pts[sel]
    return pts
