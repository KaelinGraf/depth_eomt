# ---------------------------------------------------------------
# Monocular depth supervision for the DA3-style DPT head added to
# EoMT. Implements scale-invariant log-depth (SI-Log) regression as
# described by Eigen+2014 and used by Depth-Anything V1/V2/V3.
# ---------------------------------------------------------------

from __future__ import annotations

import torch


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
    return torch.sqrt((var_term - lambda_var * bias_term).clamp_min(eps))
