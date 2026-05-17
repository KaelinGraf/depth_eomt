"""USD-to-trimesh loader for SDG part assets.

The SDG asset USDs ship with two mesh prims per file: one visual (`/.../Geometry/...`)
and one collision (`/.../CollisionGeom`). We prefer the collision geometry — it's
designed for physics and is lower poly, so FCL collision checks are faster.

The mesh scale chain in SDG is non-trivial: applying `objects.json:scale_factor` to
the raw USD bbox gives nonsense (micrometres for distractors), so the catalog
`scale_factor` is NOT a USD-units→metres conversion. The ground truth scale for any
given scene instance is `scene_info.json:objects[i].canonical_extent`, which is the
axis-aligned bbox of the spawned prim in its local frame *after* all SDG spawn-time
scaling has been applied. To make our trimesh match the scene geometry exactly, we
load the raw USD mesh and rescale per-axis so its bbox matches canonical_extent.

This is a small per-axis deformation (~5-10%) and intentionally accepted because:
- Each grasp is tied to a specific (frame, instance) — no cross-instance transfer
- The grasps are stored in the *instance* frame, not a canonical asset frame
- The deformation is whatever SDG actually applied, so our collision checks are
  testing the same geometry the gripper would interact with in reality
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging
import numpy as np
import trimesh
from pxr import Usd, UsdGeom

logger = logging.getLogger(__name__)


def _gather_mesh_points_and_faces(stage: Usd.Stage, prefer_collision: bool = True):
    """Walk USD stage, gather mesh prims, return (verts, faces) in world frame.

    Returns concatenated arrays across all matched mesh prims.
    """
    collision_meshes = []
    visual_meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path_str = prim.GetPath().pathString
        # Heuristic: 'CollisionGeom' or 'collision' anywhere in path → collision
        is_collision = "CollisionGeom" in path_str or "/collision" in path_str.lower()
        target = collision_meshes if is_collision else visual_meshes
        target.append(prim)

    candidates = collision_meshes if (prefer_collision and collision_meshes) else visual_meshes
    if not candidates:
        # Fall back to whatever exists
        candidates = collision_meshes + visual_meshes
    if not candidates:
        raise ValueError(f"No mesh prims found in stage {stage.GetRootLayer().identifier}")

    all_v, all_f = [], []
    base = 0
    tcode = Usd.TimeCode.Default()
    for mp in candidates:
        mesh = UsdGeom.Mesh(mp)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not indices:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.int32)
        indices = np.asarray(indices, dtype=np.int32)

        # Apply local-to-world (within the USD itself) so multiple meshes line up
        xf = np.asarray(UsdGeom.Xformable(mp).ComputeLocalToWorldTransform(tcode), dtype=np.float64)
        # USD matrices are row-vector convention: v' = v @ M
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_world = (pts_h @ xf)[:, :3]

        # Triangulate polygons (most SDG meshes are already tris; handle quads too)
        cursor = 0
        for c in counts:
            if c == 3:
                all_f.append([indices[cursor] + base, indices[cursor+1] + base, indices[cursor+2] + base])
            elif c == 4:
                a, b, cc, d = indices[cursor:cursor+4] + base
                all_f.append([a, b, cc]); all_f.append([a, cc, d])
            else:
                # n-gon fan triangulation
                first = indices[cursor] + base
                for k in range(1, c-1):
                    all_f.append([first, indices[cursor+k] + base, indices[cursor+k+1] + base])
            cursor += c

        all_v.append(pts_world)
        base += len(pts)

    verts = np.vstack(all_v)
    faces = np.asarray(all_f, dtype=np.int64)
    return verts, faces


def load_usd_mesh(
    usd_path: str | Path,
    target_extent: Optional[np.ndarray] = None,
    prefer_collision: bool = True,
) -> trimesh.Trimesh:
    """Load a USD asset as a trimesh, optionally rescaling to a target AABB extent.

    Args:
        usd_path: path to .usd file
        target_extent: target (Lx, Ly, Lz) AABB extent in metres. When provided, the
            mesh is rescaled per-axis so its AABB matches this. Use the scene_info
            object's `canonical_extent` value for instance-accurate geometry.
        prefer_collision: prefer CollisionGeom over visual geometry when both exist.
    """
    usd_path = str(usd_path)
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise IOError(f"USD failed to open: {usd_path}")
    verts, faces = _gather_mesh_points_and_faces(stage, prefer_collision=prefer_collision)
    if len(faces) == 0:
        raise ValueError(f"No triangles after triangulation: {usd_path}")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Centre on geometric centroid (so the local frame's origin matches the
    # SDG canonical_extent which is centroid-aligned, per camera_monocular.py:804).
    centroid = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-centroid)

    if target_extent is not None:
        raw_extent = np.ptp(mesh.bounds, axis=0)  # (Lx, Ly, Lz)
        # Avoid zero-div for degenerate axes
        safe = np.where(raw_extent > 1e-9, raw_extent, 1.0)
        scale_per_axis = np.asarray(target_extent, dtype=np.float64) / safe
        mesh.apply_scale(scale_per_axis)
    return mesh
