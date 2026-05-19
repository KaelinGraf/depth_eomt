# ---------------------------------------------------------------
# Monocular depth supervision for the DA3-style DPT head added to
# EoMT. Implements scale-invariant log-depth (SI-Log) regression as
# described by Eigen+2014 and used by Depth-Anything V1/V2/V3.
# ---------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F


def loss_depth_silog(
    d_pred: torch.Tensor,
    d_gt: torch.Tensor,
    *,
    lambda_var: float = 0.85,
    d_max: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-invariant log-depth loss.

    SI-Log(d, d̂) = sqrt( mean(R²) − λ · mean(R)² )    where R = log d − log d̂

    Args:
        d_pred: Predicted depth, shape [B, 1, H, W], strictly positive
            (assumes the head's `exp` activation has already been applied).
        d_gt: Ground-truth depth, shape [B, 1, H, W], in metres. May contain
            non-positive / non-finite values for invalid pixels — they are
            masked out before the reduction.
        lambda_var: Weight on the bias term (DA2/DA3 default 0.85;
            paper uses 0.5–1.0).
        d_max: Pixels with `d_gt >= d_max` are treated as invalid. Defaults
            to 10 m which is generous for bin-picking (real bins are < 2 m).
        eps: Numerical stability for log + final sqrt.

    Returns:
        Scalar SI-Log loss. Returns a graph-preserving zero if no pixel is
        valid (lets training continue past degenerate frames without NaN).
    """
    valid = torch.isfinite(d_gt) & (d_gt > 0) & (d_gt < d_max)
    if not valid.any():
        # graph-preserving zero so we don't break the autograd path
        return d_pred.sum() * 0.0

    log_pred = torch.log(d_pred.clamp_min(eps))
    log_gt = torch.log(d_gt.clamp_min(eps))
    log_diff = (log_pred - log_gt)[valid]

    var_term = (log_diff ** 2).mean()
    bias_term = log_diff.mean() ** 2
    out = torch.sqrt((var_term - lambda_var * bias_term).clamp_min(eps))
    # Belt-and-suspenders against any residual non-finite value (eomt.forward
    # source-clamps d_pred to [1e-4, 20], so this should never fire).
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------
# Geometry helpers (used by the gradient / normal / consistency losses).
#
# Convention: depth is z-buffer (distance_to_image_plane). Camera coords
# are OpenCV-style (X right, Y down, Z forward into the scene). Replicator's
# `normals` annotator emits OpenGL-style camera normals (X right, Y up, Z
# toward camera), so the dataset loader applies a (+1, -1, -1) flip on
# normals.npy to bring GT into OpenCV. After that flip:
#   median cosine(depth_to_normals(GT_depth), GT_normals) is +0.95-0.99
#   across val frames (mean ~0.6-0.96 — dragged down only by edge pixels
#   where the forward-diff straddles two surfaces, a finite-difference
#   artifact, not a convention bug). See the __main__ self-test below.
#
# The cross-product order in `depth_to_normals` is `cross(t_y, t_x)`, which
# gives an outward normal (-Z, toward camera) for a frontal surface — the
# sign that matches Replicator GT after the loader flip.
# ---------------------------------------------------------------


def valid_depth_mask(d_gt: torch.Tensor, d_max: float = 10.0) -> torch.Tensor:
    """Bool mask of pixels usable for depth-based losses: finite, > 0, < d_max."""
    return torch.isfinite(d_gt) & (d_gt > 0) & (d_gt < d_max)


def _forward_diff(x: torch.Tensor):
    """Forward differences along W (dx) and H (dy) on a [B, C, H, W] tensor.
    Returns (dx, dy), each [B, C, H, W] with the last column/row zero-padded
    so the output shape matches the input. Forward (not central) diffs are
    cheaper and equivalent in expectation for L1 gradient losses.
    """
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def _grad_valid(valid: torch.Tensor):
    """Per-direction validity masks for forward-diff gradients: the gradient
    at pixel (u, v) is valid only when both that pixel and its forward
    neighbour are valid. Returns (valid_x, valid_y), each same shape as
    `valid` (the last column / row are forced False — no neighbour).
    """
    vx = valid & F.pad(valid[..., :, 1:], (0, 1, 0, 0), value=False)
    vy = valid & F.pad(valid[..., 1:, :], (0, 0, 0, 1), value=False)
    vx[..., :, -1] = False
    vy[..., -1, :] = False
    return vx, vy


def unproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pinhole back-projection. depth: [B, 1, H, W] metres. K: [B, 3, 3] in
    pixel units, in the same coordinate frame as the input image. Returns
    points: [B, 3, H, W] in camera-space (OpenCV: X right, Y down, Z fwd).
    """
    B, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    u = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    v = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    fx = K[:, 0, 0].to(dtype).view(B, 1, 1, 1)
    fy = K[:, 1, 1].to(dtype).view(B, 1, 1, 1)
    cx = K[:, 0, 2].to(dtype).view(B, 1, 1, 1)
    cy = K[:, 1, 2].to(dtype).view(B, 1, 1, 1)
    Z = depth
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return torch.cat([X, Y, Z], dim=1)


def depth_to_normals(points: torch.Tensor) -> torch.Tensor:
    """Per-pixel surface normals from a camera-space point map.

    points: [B, 3, H, W]. Returns unit normals [B, 3, H, W] with the trailing
    row / column edge-replicated to keep shape. The cross-product order is
    `cross(t_y, t_x)` (verified against Replicator GT to give matching sign
    on this dataset — see __main__).
    """
    tx = points[..., :, 1:] - points[..., :, :-1]   # [B, 3, H,   W-1]
    ty = points[..., 1:, :] - points[..., :-1, :]   # [B, 3, H-1, W]
    tx = tx[..., :-1, :]                            # [B, 3, H-1, W-1]
    ty = ty[..., :, :-1]                            # [B, 3, H-1, W-1]
    n = torch.cross(ty, tx, dim=1)                  # outward normal on this convention
    n = F.normalize(n, dim=1, eps=1e-8)
    return F.pad(n, (0, 1, 0, 1), mode="replicate")


# ---------------------------------------------------------------
# fp32 wrapper for the aux losses
#
# The gradient / normal / consistency losses do finite-difference cross
# products and log() of (post-exp) depth — both are sensitive to fp16
# numerics: tangent vectors of equal-depth neighbours underflow to 0,
# `cross(0, 0)` is 0, the backward through `F.normalize(0)` is undefined,
# and a single NaN element contaminates the whole reduction. Running
# these losses in fp32 regardless of the outer Lightning autocast
# context eliminates the NaN spikes observed at ~3 % of batches in run
# `gvp8` (the fp16-throughout curriculum run, stopped at step 1627).
#
# SI-Log is intentionally NOT wrapped — it was stable in run `hw3te1ap`
# (the baseline run) and the wrapper isn't free.
# ---------------------------------------------------------------


def _aux_loss_fp32(fn):
    """Decorator: run an aux-loss function under torch.autocast(enabled=False),
    with floating-point tensor args promoted to fp32. Bool / int tensors
    (the `valid` masks) pass through unchanged."""

    def wrapped(*args, **kwargs):
        first = next(
            (a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)),
            None,
        )
        if first is None:
            return fn(*args, **kwargs)

        def _cast(x):
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                return x.float()
            return x

        with torch.autocast(first.device.type, enabled=False):
            return fn(*[_cast(a) for a in args],
                      **{k: _cast(v) for k, v in kwargs.items()})

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


# ---------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------


@_aux_loss_fp32
def loss_depth_grad(
    d_pred: torch.Tensor, d_gt: torch.Tensor,
    *, d_max: float = 10.0, eps: float = 1e-6,
) -> torch.Tensor:
    """Gradient-matching loss on log-depth residuals (MiDaS / DA-style).

    L = mean(|∂x R| + |∂y R|),  R = log d̂ − log d_gt.
    Operates in log space so it pairs naturally with SI-Log and stays
    scale-aware for narrow depth ranges. Per-direction validity is the
    AND of each pixel with its forward neighbour.
    """
    valid = valid_depth_mask(d_gt, d_max)
    if not valid.any():
        return d_pred.sum() * 0.0
    R = torch.log(d_pred.clamp_min(eps)) - torch.log(d_gt.clamp_min(eps))
    dx, dy = _forward_diff(R)
    vx, vy = _grad_valid(valid)
    nx = vx.sum().clamp_min(1)
    ny = vy.sum().clamp_min(1)
    out = (dx.abs() * vx).sum() / nx + (dy.abs() * vy).sum() / ny
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@_aux_loss_fp32
def loss_normal_angular(
    n_pred: torch.Tensor, n_gt: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    """1 − cosine angular loss between predicted and GT normals over `valid`.

    n_pred, n_gt: [B, 3, H, W] unit. valid: [B, 1, H, W] bool. Uses 1−cos
    rather than arccos for smoother gradients near the antipode.
    """
    if not valid.any():
        return n_pred.sum() * 0.0
    cos = (n_pred * n_gt).sum(dim=1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    out = ((1.0 - cos) * valid).sum() / valid.sum().clamp_min(1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@_aux_loss_fp32
def loss_normal_grad(
    n_pred: torch.Tensor, n_gt: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    """L1 gradient-matching loss on normal maps, summed over the 3 channels.

    `valid` (per-pixel) is propagated to the gradient positions via _grad_valid.
    """
    if not valid.any():
        return n_pred.sum() * 0.0
    dxp, dyp = _forward_diff(n_pred)
    dxg, dyg = _forward_diff(n_gt)
    vx, vy = _grad_valid(valid)
    # broadcast [B,1,H,W] valid masks over the 3 normal channels; the
    # denominator accounts for the 3 channels per valid pixel position.
    cx = vx.sum().clamp_min(1) * 3
    cy = vy.sum().clamp_min(1) * 3
    out = (((dxp - dxg).abs() * vx).sum() / cx
           + ((dyp - dyg).abs() * vy).sum() / cy)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@_aux_loss_fp32
def loss_depth_normal_consistency(
    d_pred: torch.Tensor, n_pred: torch.Tensor, K: torch.Tensor, valid: torch.Tensor,
    *, eps: float = 1e-7,
) -> torch.Tensor:
    """Mutual-consistency: the head's normals should agree with the normals
    *derived from the predicted depth* (unproject + cross-product). Gradients
    flow through both n_pred (head) and d_pred (depth head via the unproject),
    which is the mechanism that makes the depth output piecewise-planar with
    sharp edges (the "hilly landscape" cure).

    d_pred: [B,1,H,W] (post-exp metres), n_pred: [B,3,H,W] unit,
    K: [B,3,3], valid: [B,1,H,W] bool.
    """
    points = unproject(d_pred, K)
    n_from_depth = depth_to_normals(points)
    # cross-product is undefined at the trailing edge — drop those pixels.
    valid_inner = valid.clone()
    valid_inner[..., -1, :] = False
    valid_inner[..., :, -1] = False
    if not valid_inner.any():
        return d_pred.sum() * 0.0
    cos = (n_from_depth * n_pred).sum(dim=1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    out = ((1.0 - cos) * valid_inner).sum() / valid_inner.sum().clamp_min(1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------
# Self-test — verifies the normal coordinate convention against the
# Replicator GT. Run `python -m training.depth_loss` (or `python
# training/depth_loss.py`) on a populated dataset to confirm that the
# cross-product order and OpenCV unprojection agree with normals.npy.
# A mean cosine > 0.9 means we're good; near -1 means flip the cross
# order; intermediate means the GT uses a different axis convention.
# ---------------------------------------------------------------


def _selftest_normal_convention(frame_dir):
    """Sanity-check that depth_to_normals(GT_depth) agrees with GT_normals
    after the (+1,-1,-1) loader flip. Reports mean + median cosine + the
    fraction of pixels >0.9; median should be > 0.94 across frames."""
    import json
    from pathlib import Path
    import numpy as np

    frame_dir = Path(frame_dir)
    K = np.asarray(json.load(open(next(frame_dir.glob("*scene_info*.json"))))[
        "camera"]["cam_K"], dtype=np.float32).reshape(3, 3)
    d = np.load(frame_dir / "depth.npy").astype(np.float32)
    n_gt = np.load(frame_dir / "normals.npy")[..., :3].astype(np.float32)
    # OpenGL -> OpenCV (negate Y and Z). See module header.
    n_gt = n_gt * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    d_t = torch.from_numpy(d)[None, None]
    K_t = torch.from_numpy(K)[None]
    n_gt_t = torch.from_numpy(np.transpose(n_gt, (2, 0, 1)))[None]

    pts = unproject(d_t, K_t)
    n_pred = depth_to_normals(pts)
    valid = valid_depth_mask(d_t)
    valid[..., -1, :] = False
    valid[..., :, -1] = False
    cos = (n_pred * n_gt_t).sum(dim=1)[valid[:, 0]].numpy()

    import numpy as _np
    print(f"frame: {frame_dir.name}  "
          f"cos: mean={cos.mean():+.3f}  median={_np.median(cos):+.3f}  "
          f">0.9 frac={(cos > 0.9).mean():.1%}  <0 frac={(cos < 0).mean():.1%}")
    return float(_np.median(cos))


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/kaelin/BinPicking/SDG/IS/Outputs/monocular_dataset/val/frame_1018"
    )
    _selftest_normal_convention(target)
