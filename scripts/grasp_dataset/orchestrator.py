"""Per-frame grasp dataset orchestrator.

For one frame:
    1. Load scene (depth/points, instance mask, scene_info, objects)
    2. Build a scene-wide collision context:
       a. Load each known mesh (parts + bins), place at world (camera) frame
       b. Build voxel proxy from depth/SL point cloud
       c. Combine into a CollisionManager (the target object is excluded
          per-grasp at check time)
    3. For each visible part (class=="part" and visibility ≥ threshold):
       a. Sample 1000 antipodal candidates on the target mesh (object frame)
       b. Transform each to world frame
       c. Run filter chain (top-down, side, collision); save ALL with reasons
    4. Emit one GraspGen-compatible JSON per (frame, instance, object) under
       grasp_dataset/{dataset}/{split}/frame_XXXXX__inst_NN__objname.grasps.json

The JSON also carries scene_metadata so a custom GraspGen-compatible loader can
recover full scene context for the encoder.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import logging
import time
import numpy as np
import trimesh

from .scene_io import Scene, SceneObject, load_scene, scene_pointcloud_world
from .mesh_io import load_usd_mesh
from .antipodal import sample_antipodal_grasps
from .collision import build_scene_collision_manager, check_grasp_collision
from .filters import topdown_soft_filter, wrong_side_filter
from .gripper import gripper_mesh_at_grasp, GRIPPER_MAX_APERTURE, GRIPPER_DEPTH


logger = logging.getLogger(__name__)


# Mesh cache: USD path → (raw trimesh, canonical_extent applied None)
# Because canonical_extent varies per-instance we cache the RAW mesh and clone+rescale.
_MESH_CACHE: dict[str, trimesh.Trimesh] = {}


def _get_mesh_for_instance(obj: SceneObject) -> Optional[trimesh.Trimesh]:
    """Load USD mesh for an object, rescaled per-axis to its canonical_extent.

    Returns None if USD path is missing or load failed (object will be skipped).
    """
    if obj.usd_filepath is None:
        return None
    try:
        if obj.usd_filepath not in _MESH_CACHE:
            _MESH_CACHE[obj.usd_filepath] = load_usd_mesh(obj.usd_filepath, target_extent=None)
        base = _MESH_CACHE[obj.usd_filepath]
        m = base.copy()
        raw_ext = np.ptp(m.bounds, axis=0)
        safe = np.where(raw_ext > 1e-9, raw_ext, 1.0)
        scale_per_axis = obj.canonical_extent / safe
        m.apply_scale(scale_per_axis)
        return m
    except Exception as e:
        logger.warning(f"mesh load failed {obj.usd_filepath}: {e}")
        return None


def _mesh_world_pose(obj: SceneObject, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Place a mesh (in object local frame, centroid-aligned) at the object's
    pose in world (camera) frame."""
    out = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = obj.R_m2c
    T[:3, 3] = obj.t_m2c
    out.apply_transform(T)
    return out


def _build_scene_meshes(scene: Scene) -> dict[str, trimesh.Trimesh]:
    """Load + place ALL annotated meshes (parts + bins) in world (camera) frame.

    Returns {obj_key: trimesh} where obj_key = f"seg{seg_id}__{catalog_name}".
    These will be added as named obstacles in the CollisionManager. The target
    object's key is excluded at check time.
    """
    out: dict[str, trimesh.Trimesh] = {}
    for obj in scene.objects:
        m = _get_mesh_for_instance(obj)
        if m is None or len(m.faces) == 0:
            continue
        m_world = _mesh_world_pose(obj, m)
        key = f"seg{obj.seg_id}__{obj.catalog_name}"
        out[key] = m_world
    return out


def _object_world_pose_4x4(obj: SceneObject) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = obj.R_m2c
    T[:3, 3] = obj.t_m2c
    return T


def _grasp_pose_object_to_world(grasp_obj: np.ndarray, T_obj2world: np.ndarray) -> np.ndarray:
    return T_obj2world @ grasp_obj


def _grasp_pose_world_to_object(grasp_world: np.ndarray, T_obj2world: np.ndarray) -> np.ndarray:
    T_world2obj = np.linalg.inv(T_obj2world)
    return T_world2obj @ grasp_world


def process_frame(
    frame_dir: Path,
    dataset: str,
    split: str,
    output_root: Path,
    *,
    n_grasps_per_part: int = 1000,
    min_visibility: float = 0.3,
    topdown_cutoff_deg: float = 80.0,
    topdown_knee_deg: float = 60.0,
    voxel_size: float = 0.015,
    voxel_max_count: int = 0,    # 0 = disable voxel proxy (default; SDG annotates
                                  # every visible object so the proxy is redundant
                                  # and dominates noise). Re-enable if upgrading to
                                  # full-scene state from a regenerated SDG.
    skip_existing: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Process one frame; write per-instance JSONs; return a summary dict."""
    t0 = time.time()
    scene = load_scene(frame_dir, dataset=dataset, split=split)
    rng = np.random.default_rng(seed + scene.frame_id)

    # Determine target parts up-front so we can skip the heavy collision build
    # when there's nothing to do.
    target_objs = [o for o in scene.objects
                   if o.obj_class == "part"
                   and o.visibility_ratio >= min_visibility
                   and o.usd_filepath is not None]
    if not target_objs:
        return {"frame": scene.frame_id, "n_parts": 0, "elapsed_s": time.time() - t0,
                "msg": "no eligible parts"}

    # Build known-mesh scene context once (reused across all target parts;
    # target's mesh is excluded at collision-check time via exclude_names).
    scene_meshes = _build_scene_meshes(scene)

    # Pre-compute summary of all scene objects for the metadata block (custom
    # loader uses this to construct scene-aware encoders).
    scene_objects_meta = [{
        "seg_id": o.seg_id,
        "class": o.obj_class,
        "catalog_name": o.catalog_name,
        "R_m2c": o.R_m2c.tolist(),
        "t_m2c": o.t_m2c.tolist(),
        "scale_m2c": o.scale_m2c,
        "canonical_extent": o.canonical_extent.tolist(),
        "visibility_ratio": o.visibility_ratio,
        "usd_filepath": o.usd_filepath,
    } for o in scene.objects]

    # Output directory
    out_dir = output_root / dataset / split
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_origin = scene.cam_origin_world

    summary = {"frame": scene.frame_id, "n_parts": len(target_objs),
               "per_part": [], "elapsed_s": 0.0}

    for tgt in target_objs:
        # Sanitise object name for filename (some catalog names contain "/" or other)
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in tgt.catalog_name)
        out_path = out_dir / f"frame_{scene.frame_id:05d}__inst_{tgt.seg_id:05d}__{safe_name}.grasps.json"
        if skip_existing and out_path.exists():
            summary["per_part"].append({"seg_id": tgt.seg_id, "skipped": True, "path": str(out_path)})
            continue

        tgt_mesh_obj = _get_mesh_for_instance(tgt)  # in object local frame
        if tgt_mesh_obj is None:
            summary["per_part"].append({"seg_id": tgt.seg_id, "error": "mesh_load_failed"})
            continue
        T_obj2world = _object_world_pose_4x4(tgt)
        obj_centroid_world = T_obj2world[:3, 3]

        # Per-target voxel proxy: exclude THIS object's visible pixels (else
        # gripper would always "collide" with the target's own surface points)
        scene_pcd_target_excl = scene_pointcloud_world(
            scene, max_points=120000,
            exclude_seg_ids={tgt.seg_id},
        )
        mgr = build_scene_collision_manager(
            known_meshes_world=scene_meshes,
            voxel_pcd_world=scene_pcd_target_excl,
            voxel_size=voxel_size,
            voxel_max_count=voxel_max_count,
        )

        # World-down direction expressed in OBJECT frame (for sampler bias)
        # T_obj2world rotates obj→cam; cam is our working frame. world_down_in_cam
        # → world_down_in_obj = R_obj2cam.T @ world_down_in_cam
        R_obj2cam = T_obj2world[:3, :3]
        world_down_in_obj = R_obj2cam.T @ scene.world_down_in_cam

        # 1. Sample antipodal candidates in OBJECT frame, biased toward world-down
        sample_res = sample_antipodal_grasps(
            tgt_mesh_obj,
            n_candidates=n_grasps_per_part,
            approach_bias_world=world_down_in_obj,
            approach_bias_strength=2.0,
            seed=int(seed + scene.frame_id * 1000 + tgt.seg_id),
        )
        grasps_obj = sample_res["transforms"]    # (M, 4, 4) in object frame
        widths = sample_res["widths"]            # (M,)

        # 2. Transform to cam (world) frame & filter
        target_excl = {f"seg{tgt.seg_id}__{tgt.catalog_name}"}
        transforms_obj_out = []
        widths_out = []
        in_gripper = []
        reasons = []
        topdown_angles_deg = []

        for g_obj, w in zip(grasps_obj, widths):
            g_world = T_obj2world @ g_obj

            failure_reasons = []

            # (a) top-down soft filter (rejection-sampled), in cam frame using
            # the per-frame world-down direction.
            ok_td, theta_deg = topdown_soft_filter(
                g_world,
                world_down_in_frame=scene.world_down_in_cam,
                cutoff_deg=topdown_cutoff_deg,
                knee_deg=topdown_knee_deg,
                rng=rng,
            )
            topdown_angles_deg.append(theta_deg)
            if not ok_td:
                failure_reasons.append("topdown_rejected")

            # (b) wrong-side filter — gripper must approach into the scene
            # (positive Z in cam frame), not back toward camera.
            if wrong_side_filter(g_world):
                failure_reasons.append("wrong_side")

            # (c) scene collision check
            gripper_mesh_w = gripper_mesh_at_grasp(g_world, float(w))
            coll = check_grasp_collision(mgr, gripper_mesh_w, exclude_names=target_excl)
            if coll is not None:
                failure_reasons.append(coll)

            valid = len(failure_reasons) == 0
            transforms_obj_out.append(g_obj.tolist())
            widths_out.append(float(w))
            in_gripper.append(bool(valid))
            reasons.append(",".join(failure_reasons))

        # 3. Emit JSON
        from .gripper import GRIPPER_DEPTH  # local import to keep top tidy
        payload = {
            "object": {
                "file": tgt.usd_filepath,
                "scale": 1.0,  # mesh is pre-scaled per-axis to canonical_extent
                "canonical_extent": tgt.canonical_extent.tolist(),
            },
            "gripper": {
                "name": "robotiq_2f_140",
                "max_aperture": GRIPPER_MAX_APERTURE,
                "depth": GRIPPER_DEPTH,
                "transform_offset_from_asset_to_graspgen_convention": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            },
            "grasps": {
                "transforms": transforms_obj_out,
                "widths": widths_out,
                "object_in_gripper": in_gripper,
                "filter_reasons": reasons,
                "topdown_angle_deg": topdown_angles_deg,
            },
            "scene_metadata": {
                "dataset": dataset,
                "split": split,
                "frame_id": scene.frame_id,
                "instance_seg_id": tgt.seg_id,
                "object_name": tgt.catalog_name,
                "object_pose_cam": {
                    "R_m2c": tgt.R_m2c.tolist(),
                    "t_m2c": tgt.t_m2c.tolist(),
                    "scale_m2c": tgt.scale_m2c,
                },
                "object_visibility_ratio": tgt.visibility_ratio,
                "camera_intrinsics": scene.K.tolist(),
                "scene_info_path": str(scene.scene_info_path),
                "instance_mask_path": str(scene.instance_mask_path),
                "scene_pcd_source": str(scene.depth_or_points_path),
                "scene_objects": scene_objects_meta,
            },
        }
        with open(out_path, "w") as f:
            json.dump(payload, f)

        n_valid = sum(in_gripper)
        summary["per_part"].append({
            "seg_id": tgt.seg_id,
            "name": tgt.catalog_name,
            "n_grasps": len(in_gripper),
            "n_valid": n_valid,
            "path": str(out_path),
        })
        if verbose:
            print(f"  frame {scene.frame_id} inst {tgt.seg_id} ({tgt.catalog_name}): "
                  f"{n_valid}/{len(in_gripper)} valid → {out_path.name}")

    summary["elapsed_s"] = time.time() - t0
    return summary
