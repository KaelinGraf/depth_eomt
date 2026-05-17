"""Antipodal grasp sampler for parallel-jaw grippers.

Algorithm (CPU, no warp):
    1. Sample N surface points + normals on the object mesh.
    2. For each point, ray-cast inward (along -normal) through the mesh and
       collect the exit hits (back faces of the part).
    3. For each exit hit, check antipodal validity:
       - Exit normal ≈ -entry normal (within angular tolerance)
       - Distance from entry to exit < gripper max aperture
    4. For each valid (entry, exit) pair, sample one approach direction in the
       plane perpendicular to the closing axis. The approach is biased toward
       world -Z (i.e. coming from above) but not crushed — top-down weighting
       happens as a separate filter so failures are preserved.
    5. Build the grasp pose via `grasp_pose_from_antipodal_pair`.

The sampler is intentionally lenient. It overproduces candidate grasps; downstream
filters (collision, side, top-down) prune them. Failed candidates are saved.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import trimesh

from .gripper import (
    GRIPPER_MAX_APERTURE,
    grasp_pose_from_antipodal_pair,
)


def _random_perpendicular_unit(axis: np.ndarray, bias: np.ndarray, bias_strength: float = 1.0) -> np.ndarray:
    """Sample a unit vector perpendicular to `axis`, biased toward the projection of `bias`.

    bias_strength in [0, ∞): 0 = uniform on the perpendicular circle, large = always
    the projection of `bias` (clipped to unit length).
    """
    proj = bias - np.dot(bias, axis) * axis
    pn = np.linalg.norm(proj)
    if pn > 1e-6 and bias_strength > 0:
        proj = proj / pn
    else:
        # bias parallel to axis; pick any perpendicular direction
        fallback = np.array([0., 0., 1.]) if abs(axis[2]) < 0.95 else np.array([1., 0., 0.])
        proj = fallback - np.dot(fallback, axis) * fallback
        proj = proj / np.linalg.norm(proj)
        bias_strength = 0.0
    # Build perpendicular basis (proj, ortho)
    ortho = np.cross(axis, proj)
    ortho = ortho / np.linalg.norm(ortho)
    # Sample angle theta around the perpendicular circle, biased toward 0
    # Use a wrapped-Gaussian centred at 0 with sigma controlled by bias_strength
    if bias_strength <= 0:
        theta = np.random.uniform(-np.pi, np.pi)
    else:
        sigma = np.pi / max(bias_strength, 0.1)  # at bias_strength=π, sigma≈π/π=1 rad
        theta = np.random.normal(0.0, sigma)
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    return np.cos(theta) * proj + np.sin(theta) * ortho


def sample_antipodal_grasps(
    mesh: trimesh.Trimesh,
    n_candidates: int,
    *,
    max_aperture: float = GRIPPER_MAX_APERTURE,
    antiparallel_tol_deg: float = 25.0,
    approach_bias_world: Optional[np.ndarray] = None,
    approach_bias_strength: float = 1.0,
    seed: Optional[int] = None,
):
    """Sample antipodal grasp candidates on `mesh`.

    Args:
        mesh: trimesh in object frame (any unit, but world units recommended)
        n_candidates: target number of candidate grasps to produce. The sampler
            oversamples surface points to compensate for rejection.
        max_aperture: gripper max jaw aperture, metres. Grasps with closing
            distance > this are rejected at sampling time.
        antiparallel_tol_deg: maximum angular deviation between entry and exit
            surface normals to count as antipodal.
        approach_bias_world: optional world-frame approach direction to bias the
            sampler toward (typically [0, 0, -1] for top-down). If None, uniform.
        approach_bias_strength: 0 = no bias (uniform on perpendicular circle),
            larger = tighter Gaussian around the bias direction.
        seed: RNG seed for reproducibility.

    Returns:
        dict with keys:
            'transforms': (M, 4, 4) float64 — grasp poses in object frame
            'widths': (M,) float64 — antipodal distances in metres
            'approach_world_angle_rad': (M,) — for each grasp, the angle between
                the approach axis (grasp_R[:, 2] in world frame, ASSUMING
                approach_bias_world's frame is world) and (0, 0, -1). Used by
                downstream top-down filter; if approach_bias_world is in OBJECT
                frame, treat this metric accordingly.
        Where M ≤ n_candidates depending on how many surface samples produce
        valid antipodal pairs.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed if seed is not None else None)

    # Oversample surface points — most won't have valid antipodal exit
    oversample = max(2, int(np.ceil(n_candidates * 4)))
    pts, face_idx = trimesh.sample.sample_surface(mesh, oversample)
    pts = np.asarray(pts, dtype=np.float64)
    normals = mesh.face_normals[face_idx]
    # Flip normals to ensure they point outward (sample_surface returns face normals
    # which are already outward for closed meshes but degenerate cases possible)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(norms > 1e-9, normals / norms, np.array([0., 0., 1.]))

    # Cast rays from each point along -normal (inward); collect ALL hits
    ray_origins = pts + normals * 1e-5  # nudge outside to avoid self-hit
    ray_dirs = -normals
    locations, idx_ray, idx_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_dirs,
        multiple_hits=True,
    )
    if len(locations) == 0:
        return {
            'transforms': np.zeros((0, 4, 4), dtype=np.float64),
            'widths': np.zeros((0,), dtype=np.float64),
        }

    # For each ray, find the FURTHEST hit along the ray direction within aperture.
    # (Sampling inside the part — we want the exit through the opposite surface.)
    # Group hits by ray index.
    cos_tol = float(np.cos(np.deg2rad(antiparallel_tol_deg)))
    transforms = []
    widths = []
    approach_bias_world = (
        np.asarray(approach_bias_world, dtype=np.float64)
        if approach_bias_world is not None else None
    )

    # Build per-ray hit lists
    hits_per_ray: dict[int, list[tuple[np.ndarray, int]]] = {}
    for loc, ir, it in zip(locations, idx_ray, idx_tri):
        hits_per_ray.setdefault(int(ir), []).append((loc, int(it)))

    for ir, hits in hits_per_ray.items():
        if len(hits) < 1:
            continue
        p_entry = pts[ir]
        n_entry = normals[ir]
        # Hits sorted by distance from entry
        dists = [np.dot(loc - p_entry, -n_entry) for loc, _ in hits]
        order = np.argsort(dists)
        # Take the furthest hit that is within aperture and antipodal
        best = None
        for k in order[::-1]:
            d = dists[k]
            if d <= 1e-4:  # exit before entry — skip
                continue
            if d > max_aperture:
                continue
            loc, tri = hits[k]
            n_exit = mesh.face_normals[tri]
            nx = n_exit / max(np.linalg.norm(n_exit), 1e-9)
            # Antipodal check: exit normal ≈ -entry normal
            if np.dot(nx, -n_entry) >= cos_tol:
                best = (loc, nx, d)
                break
        if best is None:
            continue
        p_exit, n_exit, d = best
        # Convention: gripper +X (closing) points from right→left. We arbitrarily
        # assign p_entry as the +X side; the network is symmetric to this anyway
        # (most pipelines also store reflection augmentation).
        p_left, p_right = p_entry, p_exit
        # Sample approach direction in the plane perpendicular to closing axis
        x_axis = (p_left - p_right) / d
        if approach_bias_world is not None:
            approach = _random_perpendicular_unit(
                x_axis, approach_bias_world, bias_strength=approach_bias_strength
            )
        else:
            # uniform
            approach = _random_perpendicular_unit(x_axis, np.array([0., 0., 1.]), bias_strength=0.0)
        try:
            T, w = grasp_pose_from_antipodal_pair(p_left, p_right, approach)
        except ValueError:
            continue
        transforms.append(T)
        widths.append(w)
        if len(transforms) >= n_candidates:
            break

    if not transforms:
        return {
            'transforms': np.zeros((0, 4, 4), dtype=np.float64),
            'widths': np.zeros((0,), dtype=np.float64),
        }
    return {
        'transforms': np.stack(transforms, axis=0),
        'widths': np.asarray(widths, dtype=np.float64),
    }
