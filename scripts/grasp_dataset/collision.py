"""Scene-wide collision filter for grasp candidates.

Two collision sources for each grasp candidate:
    1. Known meshes: per-instance meshes from scene_info.json placed at their
       world poses. We can name what was collided with — useful diagnostic
       signal for the discriminator.
    2. Depth-derived voxel proxy: the visible scene surface that may include
       unannotated geometry (bin walls not in JSON, distractor surfaces
       classified as background). Voxelised at ~7mm and added as anonymous
       collision boxes. Catches what JSON misses.

Uses python-fcl via trimesh.collision.CollisionManager.
"""
from __future__ import annotations
from typing import Optional
import logging
import numpy as np
import trimesh
from trimesh.collision import CollisionManager

logger = logging.getLogger(__name__)


def build_scene_collision_manager(
    *,
    known_meshes_world: dict[str, trimesh.Trimesh],
    voxel_pcd_world: Optional[np.ndarray] = None,
    voxel_size: float = 0.007,
    voxel_max_count: int = 80000,
) -> CollisionManager:
    """Build a CollisionManager populated with scene obstacles.

    Args:
        known_meshes_world: {name: trimesh} of all obstacles whose identity we
            know. Meshes already transformed to world frame.
        voxel_pcd_world: optional (N, 3) point cloud (world frame) of the
            visible scene surface. Voxelised into a coarse occupancy grid and
            added as anonymous "scene_proxy" obstacle.
        voxel_size: voxel edge length, metres.
        voxel_max_count: cap on voxel count to avoid blowing up the collision
            tree (subsample if exceeded).
    """
    mgr = CollisionManager()
    for name, mesh in known_meshes_world.items():
        if mesh is None or len(mesh.faces) == 0:
            continue
        mgr.add_object(name, mesh)

    if voxel_pcd_world is not None and len(voxel_pcd_world) > 0:
        # Snap points to voxel grid, dedupe
        pcd = np.asarray(voxel_pcd_world, dtype=np.float64)
        vox_idx = np.floor(pcd / voxel_size).astype(np.int64)
        # dedupe
        _, unique_pos = np.unique(vox_idx, axis=0, return_index=True)
        vox_idx = vox_idx[unique_pos]
        if len(vox_idx) > voxel_max_count:
            sel = np.random.default_rng(0).choice(len(vox_idx), voxel_max_count, replace=False)
            vox_idx = vox_idx[sel]
        # Build one box per voxel; trimesh's collision manager can handle many
        # boxes via a single mesh union
        if len(vox_idx) > 0:
            half = voxel_size / 2
            centres = (vox_idx + 0.5) * voxel_size
            # Create one cube template, instantiate via translation + concatenate
            cube = trimesh.creation.box(extents=(voxel_size, voxel_size, voxel_size))
            scene_proxy = trimesh.util.concatenate(
                [cube.copy().apply_translation(c) for c in centres]
            )
            mgr.add_object("scene_proxy", scene_proxy)

    return mgr


def check_grasp_collision(
    mgr: CollisionManager,
    gripper_mesh_world: trimesh.Trimesh,
    *,
    exclude_names: Optional[set[str]] = None,
) -> Optional[str]:
    """Test the gripper mesh (already placed in world) against the scene.

    Args:
        mgr: pre-built collision manager (scene minus target object).
        gripper_mesh_world: gripper mesh in world frame.
        exclude_names: any scene names to ignore (already removed from mgr is
            preferred, but this is a safety net for the target name).

    Returns:
        Name of one colliding scene object (with prefix `collision:`), or None
        if no collision.
    """
    in_collision, contact_names = mgr.in_collision_single(
        gripper_mesh_world, return_names=True
    )
    if not in_collision:
        return None
    # contact_names is a set[str] of scene-object names the queried mesh hit.
    scene_names = set(contact_names)
    if exclude_names:
        scene_names -= set(exclude_names)
    if not scene_names:
        return None
    return "collision:" + sorted(scene_names)[0]
