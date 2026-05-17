"""Robotiq 2F-140 collision geometry, parameterised by jaw aperture.

Convention (matches `ml_deps/GraspGen/config/grippers/robotiq_2f_140.yaml`):
    - grasp_pose origin: at the gripper BASE link
    - +Z: approach direction (gripper moves in +Z to reach grasp; fingertips at z=DEPTH)
    - +X: closing direction (left finger at +X, right at -X)
    - +Y: right-handed Z × X

For a grasp with antipodal width w (the inter-fingertip distance at closure):
- Two finger pads centred at (±w/2, 0, DEPTH - 0.035), each a box 0.0075 × 0.03 × 0.07 m
- Palm/body: cuboid centred at (0, 0, 0.05), approximating the base_link extent
- Connecting fingers (outer/inner finger) approximated as a box per side

Gripper geometry returned in *grasp frame*. Transform by grasp_pose to put in world.
"""
from __future__ import annotations
import numpy as np
import trimesh

GRIPPER_DEPTH = 0.195    # z-offset from base to fingertip (matches GraspGen yaml)
GRIPPER_MAX_APERTURE = 0.12   # m, matches GraspGen yaml
FINGER_PAD_LX = 0.0075   # finger thickness (along closing axis)
FINGER_PAD_LY = 0.030    # finger width
FINGER_PAD_LZ = 0.070    # finger length (along approach)
PALM_EXTENT = (0.075, 0.085, 0.100)  # X, Y, Z extent of palm cuboid behind grasp
# Connecting arm (between palm and finger pad) — chunky box on each side
ARM_EXTENT = (0.025, 0.040, 0.090)  # X, Y, Z


def _box(extent, centre):
    m = trimesh.creation.box(extents=extent)
    m.apply_translation(np.asarray(centre, dtype=np.float64))
    return m


def build_gripper_at_aperture(width: float, include_arms: bool = True) -> trimesh.Trimesh:
    """Return a single trimesh of the gripper in grasp frame at the given jaw width.

    Args:
        width: distance between finger pad inner faces at closure, metres. Clamped
            to [0, GRIPPER_MAX_APERTURE].
        include_arms: include the chunky connecting arms between palm and fingers
            for tighter collision (more realistic for cluttered scenes).
    """
    w = float(np.clip(width, 0.0, GRIPPER_MAX_APERTURE))
    # Finger pad centres: at z = DEPTH - LZ/2 (fingertip face at +Z = DEPTH)
    z_pad = GRIPPER_DEPTH - FINGER_PAD_LZ / 2
    left_pad = _box((FINGER_PAD_LX, FINGER_PAD_LY, FINGER_PAD_LZ),
                    (+w/2 + FINGER_PAD_LX/2, 0.0, z_pad))
    right_pad = _box((FINGER_PAD_LX, FINGER_PAD_LY, FINGER_PAD_LZ),
                     (-w/2 - FINGER_PAD_LX/2, 0.0, z_pad))
    palm = _box(PALM_EXTENT, (0.0, 0.0, PALM_EXTENT[2] / 2 - 0.005))

    parts = [palm, left_pad, right_pad]
    if include_arms:
        z_arm = (PALM_EXTENT[2] - 0.01 + z_pad - FINGER_PAD_LZ/2) / 2
        arm_h = (z_pad - FINGER_PAD_LZ/2) - (PALM_EXTENT[2] - 0.01)
        arm_h = max(0.0, arm_h)
        if arm_h > 0.01:
            ax, ay, _ = ARM_EXTENT
            left_arm = _box((ax, ay, arm_h),
                            (+w/2 + ax/2 + 0.005, 0.0, z_arm))
            right_arm = _box((ax, ay, arm_h),
                             (-w/2 - ax/2 - 0.005, 0.0, z_arm))
            parts += [left_arm, right_arm]
    return trimesh.util.concatenate(parts)


def grasp_pose_from_antipodal_pair(p_left: np.ndarray, p_right: np.ndarray,
                                    approach_dir: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) grasp pose from an antipodal point pair + approach.

    Args:
        p_left, p_right: two surface points (3,) the gripper will pinch (in any
            consistent frame — object or world). p_left ends up at +X side.
        approach_dir: desired approach direction (3,), unit vector. Must be
            close to perpendicular to (p_right - p_left); we project it onto
            the plane perpendicular to the closing axis.

    Returns:
        4x4 SE(3) where origin = gripper base, +Z = approach, +X = closing
        (from p_right to p_left).
    """
    p_left = np.asarray(p_left, dtype=np.float64)
    p_right = np.asarray(p_right, dtype=np.float64)
    closing = p_left - p_right
    width = np.linalg.norm(closing)
    if width < 1e-6:
        raise ValueError("antipodal pair degenerate (zero width)")
    x_axis = closing / width

    # Project approach onto plane perpendicular to closing axis, then normalise
    a = np.asarray(approach_dir, dtype=np.float64)
    z_axis = a - np.dot(a, x_axis) * x_axis
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        # approach was parallel to closing — pick any perpendicular
        # use world -Z if not aligned, else world +Y
        fallback = np.array([0., 0., -1.]) if abs(x_axis[2]) < 0.95 else np.array([0., 1., 0.])
        z_axis = fallback - np.dot(fallback, x_axis) * x_axis
        z_norm = np.linalg.norm(z_axis)
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Grasp midpoint (where the object lies between fingers)
    midpoint = (p_left + p_right) / 2.0
    # Gripper base sits at midpoint - DEPTH * approach
    origin = midpoint - GRIPPER_DEPTH * z_axis

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = origin
    return T, float(width)


def gripper_mesh_at_grasp(grasp_pose: np.ndarray, width: float) -> trimesh.Trimesh:
    """Return the gripper collision mesh placed at the world-frame grasp pose."""
    g = build_gripper_at_aperture(width)
    g.apply_transform(grasp_pose)
    return g
