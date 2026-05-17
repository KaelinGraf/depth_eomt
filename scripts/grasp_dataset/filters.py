"""Per-grasp soft / hard filters (top-down weighting, side-of-mesh).

All filters take a grasp pose in WORLD frame (4x4 SE(3)) and return a string
reason (e.g. "topdown_rejected", "wrong_side") if the grasp fails, else "".
"""
from __future__ import annotations
import numpy as np


def approach_angle_to_vertical_rad(
    grasp_pose: np.ndarray,
    world_down_in_frame: np.ndarray = np.array([0., 0., -1.]),
) -> float:
    """Angle between grasp +Z axis (approach) and the world-down direction.

    `world_down_in_frame` is the world -Z direction expressed in whatever frame
    the grasp_pose is in. Default assumes the grasp is already in world frame.
    """
    z_axis = grasp_pose[:3, 2]
    cos_th = float(np.clip(np.dot(z_axis, world_down_in_frame), -1.0, 1.0))
    return float(np.arccos(cos_th))


def topdown_soft_filter(
    grasp_pose: np.ndarray,
    *,
    world_down_in_frame: np.ndarray = np.array([0., 0., -1.]),
    cutoff_deg: float = 80.0,
    knee_deg: float = 60.0,
    rng: np.random.Generator | None = None,
) -> tuple[bool, float]:
    """Soft rejection of grasps far from top-down.

    Acceptance prob = 1 - sigmoid((theta_deg - knee_deg) / temperature),
    where temperature is set so that at cutoff_deg the probability is ~0.005.
    At 0° → ~1.0, at knee_deg (60°) → 0.5, at cutoff_deg (80°) → ~0.005.

    Returns:
        (accepted: bool, theta_deg: float)
    """
    if rng is None:
        rng = np.random.default_rng()
    theta_deg = float(np.degrees(approach_angle_to_vertical_rad(grasp_pose, world_down_in_frame)))
    # temperature: solve 1 - sigmoid((cutoff-knee)/T) = 0.005
    #   => sigmoid((cutoff-knee)/T) = 0.995
    #   => (cutoff-knee)/T = logit(0.995) ≈ 5.293
    #   => T = (cutoff-knee) / 5.293
    T = max(0.5, (cutoff_deg - knee_deg) / 5.293)
    p_accept = 1.0 - 1.0 / (1.0 + np.exp(-(theta_deg - knee_deg) / T))
    return bool(rng.uniform() < p_accept), theta_deg


def wrong_side_filter(
    grasp_cam_pose: np.ndarray,
) -> bool:
    """Reject grasps that approach FROM behind the camera (gripper would have to
    reach through the camera body / from outside the workspace).

    Convention: cam frame +Z points into the scene. A usable bin-picking grasp
    must have its approach axis with positive Z in cam frame (gripper moves
    deeper into the scene to reach the grasp, not back toward the camera).

    Returns:
        True if WRONG side (should be rejected).
    """
    approach_z = float(grasp_cam_pose[2, 2])  # the Z component of the grasp's +Z axis in cam frame
    return approach_z <= 0.0  # gripper approaches from behind / parallel to image plane
