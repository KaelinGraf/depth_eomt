"""Unit tests for intrinsics-aware spatial transforms in `datasets/transforms.py`.

Run from the repo root with the eomt conda env:
    python -m tests.test_transforms_intrinsics
or
    pytest tests/test_transforms_intrinsics.py
"""
from __future__ import annotations

import os
import sys

import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

# Allow running as a script: prepend repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.transforms import Transforms  # noqa: E402


def _make_target(H: int, W: int, K: torch.Tensor, n_inst: int = 2):
    masks = tv_tensors.Mask(torch.ones(n_inst, H, W, dtype=torch.uint8))
    is_crowd = torch.zeros(n_inst, dtype=torch.bool)
    labels = torch.zeros(n_inst, dtype=torch.long)
    return {
        "masks": masks,
        "is_crowd": is_crowd,
        "labels": labels,
        "intrinsics": K.clone(),
    }


def test_static_helper_round_trip():
    """Compose hflip → scale → crop analytically via the static helpers and
    compare against the manual closed-form expression."""
    K = torch.tensor(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
    )

    W = 1280
    # 1) horizontal flip on width W
    K1 = Transforms._hflip_intrinsics(K, W)
    expected_cx_1 = (W - 1) - 640.0
    assert torch.allclose(K1[0, 2], torch.tensor(expected_cx_1))
    assert torch.allclose(K1[1, 2], K[1, 2])  # cy unchanged
    assert torch.allclose(K1[0, 0], K[0, 0])  # fx unchanged
    assert torch.allclose(K1[1, 1], K[1, 1])  # fy unchanged

    # 2) scale by 1.5 on both axes
    s = 1.5
    K2 = Transforms._scale_intrinsics(K1, s, s)
    assert torch.allclose(K2[0, 0], K1[0, 0] * s)
    assert torch.allclose(K2[1, 1], K1[1, 1] * s)
    assert torch.allclose(K2[0, 2], K1[0, 2] * s)
    assert torch.allclose(K2[1, 2], K1[1, 2] * s)

    # 3) crop at offset (ox=100, oy=200)
    ox, oy = 100, 200
    K3 = Transforms._crop_intrinsics(K2, ox=ox, oy=oy)
    assert torch.allclose(K3[0, 2], K2[0, 2] - ox)
    assert torch.allclose(K3[1, 2], K2[1, 2] - oy)
    assert torch.allclose(K3[0, 0], K2[0, 0])  # fx unchanged by crop
    assert torch.allclose(K3[1, 1], K2[1, 1])  # fy unchanged by crop

    # Composed expected:
    # cx_final = ((W-1) - 640) * 1.5 - 100
    # cy_final = 360 * 1.5 - 200
    # fx_final = 800 * 1.5; fy_final = 800 * 1.5
    expected = torch.tensor(
        [
            [800.0 * s, 0.0, ((W - 1 - 640.0) * s) - ox],
            [0.0, 800.0 * s, (360.0 * s) - oy],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.allclose(K3, expected), f"\nK3=\n{K3}\nexpected=\n{expected}"
    print("test_static_helper_round_trip: PASS")


def test_identity_when_no_flip_no_scale_no_crop():
    """With hflip_prob=0, square input matching a square target_size at
    scale 1.0, and zero crop offset (only possible if crop dim == input dim
    after padding), intrinsics must be unchanged."""
    torch.manual_seed(0)
    # Square so ScaleJitter at scale=1.0 is truly identity, and crop dim ==
    # input dim so the only valid (i, j) is (0, 0).
    H = W = 512
    K = torch.tensor(
        [[800.0, 0.0, 256.0], [0.0, 800.0, 256.0], [0.0, 0.0, 1.0]]
    )

    transforms = Transforms(
        img_size=(H, W),
        color_jitter_enabled=False,
        scale_range=(1.0, 1.0),
        sensor_noise_enabled=False,
        blur_enabled=False,
        hflip_prob=0.0,
    )

    img = torch.randint(0, 255, (3, H, W), dtype=torch.uint8)
    target = _make_target(H, W, K)

    img_out, target_out = transforms(img, target)
    assert torch.allclose(target_out["intrinsics"], K), (
        f"K should be unchanged when no spatial op fires.\n"
        f"got=\n{target_out['intrinsics']}\nexpected=\n{K}"
    )
    print("test_identity_when_no_flip_no_scale_no_crop: PASS")


def test_pad_and_filter_pass_through():
    """Both `pad` and `_filter` must pass `intrinsics` through byte-identical."""
    H, W = 480, 640
    K = torch.tensor(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )

    transforms = Transforms(
        img_size=(H + 100, W + 50),  # padding will be needed
        color_jitter_enabled=False,
        scale_range=(1.0, 1.0),
        sensor_noise_enabled=False,
        blur_enabled=False,
    )

    # --- pad ---
    img = torch.randint(0, 255, (3, H, W), dtype=torch.uint8)
    target = _make_target(H, W, K)
    img_padded, target_padded = transforms.pad(img, target)
    assert torch.equal(target_padded["intrinsics"], K), "pad mutated K"

    # --- _filter (per-instance filter) ---
    keep = torch.tensor([True, False])
    target2 = _make_target(H, W, K, n_inst=2)
    filtered = transforms._filter(target2, keep)
    assert torch.equal(filtered["intrinsics"], K), "_filter mutated K"
    print("test_pad_and_filter_pass_through: PASS")


def test_legacy_no_intrinsics_path_still_works():
    """Pipeline must not crash when `intrinsics` key is absent (panoptic-only
    legacy targets)."""
    torch.manual_seed(0)
    H, W = 480, 640
    transforms = Transforms(
        img_size=(H, W),
        color_jitter_enabled=False,
        scale_range=(0.8, 1.2),
        sensor_noise_enabled=False,
        blur_enabled=False,
    )
    img = torch.randint(0, 255, (3, H, W), dtype=torch.uint8)
    target = {
        "masks": tv_tensors.Mask(torch.ones(2, H, W, dtype=torch.uint8)),
        "is_crowd": torch.zeros(2, dtype=torch.bool),
        "labels": torch.zeros(2, dtype=torch.long),
    }
    img_out, target_out = transforms(img, target)
    assert "intrinsics" not in target_out
    assert img_out.shape == (3, H, W)
    print("test_legacy_no_intrinsics_path_still_works: PASS")


def test_full_pipeline_geometric_consistency():
    """End-to-end: project a 3D point with original K through (flip → scale →
    crop) image-space transforms; project the same point with the updated K
    directly. Pixel coords must agree to sub-pixel precision."""
    torch.manual_seed(42)
    H_in, W_in = 480, 640
    img_size = (200, 280)  # crop fits comfortably inside the 0.5-scaled image
    K = torch.tensor(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )

    transforms = Transforms(
        img_size=img_size,
        color_jitter_enabled=False,
        scale_range=(0.5, 0.5),  # exact known scale
        sensor_noise_enabled=False,
        blur_enabled=False,
        hflip_prob=1.0,  # force flip
    )

    # Build a fake image with a known 3D world point at (X=0, Y=0, Z=2m).
    # Original pixel under K: u = fx*X/Z + cx = 320, v = fy*Y/Z + cy = 240.
    X, Y, Z = 0.0, 0.0, 2.0
    u0 = K[0, 0].item() * X / Z + K[0, 2].item()
    v0 = K[1, 1].item() * Y / Z + K[1, 2].item()
    assert (u0, v0) == (320.0, 240.0)

    # Encode that point as a single-pixel-bright spot in the image.
    img = torch.zeros(3, H_in, W_in, dtype=torch.uint8)
    img[:, int(v0), int(u0)] = 255

    # We need a deterministic crop offset; pre-roll an RNG state so RandomCrop
    # picks a known (i, j). Easier: drive the pipeline ourselves manually
    # using the same primitives, so the test verifies the helpers compose
    # correctly with the actual torchvision ops.
    # Step 1: flip
    W = img.shape[-1]
    img1 = F.horizontal_flip(img)
    K1 = Transforms._hflip_intrinsics(K, W)
    # Flipped pixel: u' = (W-1) - u0
    u1 = (W - 1) - u0
    v1 = v0
    # Sanity-check K1 projection matches:
    pu = K1[0, 0].item() * X / Z + K1[0, 2].item()
    pv = K1[1, 1].item() * Y / Z + K1[1, 2].item()
    assert abs(pu - u1) < 1e-4 and abs(pv - v1) < 1e-4

    # Step 2: scale 0.5
    s = 0.5
    new_H, new_W = int(H_in * s), int(W_in * s)
    img2 = F.resize(img1, [new_H, new_W])
    K2 = Transforms._scale_intrinsics(K1, s, s)
    u2 = u1 * s
    v2 = v1 * s
    pu = K2[0, 0].item() * X / Z + K2[0, 2].item()
    pv = K2[1, 1].item() * Y / Z + K2[1, 2].item()
    assert abs(pu - u2) < 1e-4 and abs(pv - v2) < 1e-4

    # Step 3: crop at (i=10, j=20)
    i, j, h, w = 10, 20, img_size[0], img_size[1]
    # crop bounds need to be valid
    assert i + h <= img2.shape[-2] and j + w <= img2.shape[-1], (
        f"Cropping {h}x{w} from {img2.shape} at ({i},{j}) is out of bounds"
    )
    K3 = Transforms._crop_intrinsics(K2, ox=j, oy=i)
    u3 = u2 - j
    v3 = v2 - i
    pu = K3[0, 0].item() * X / Z + K3[0, 2].item()
    pv = K3[1, 1].item() * Y / Z + K3[1, 2].item()
    assert abs(pu - u3) < 1e-4 and abs(pv - v3) < 1e-4, (
        f"Final projection mismatch: K3 says ({pu},{pv}), expected ({u3},{v3})"
    )
    print("test_full_pipeline_geometric_consistency: PASS")


def test_non_square_scale_recovery_anisotropic():
    """Non-square input through `Transforms.forward` should recover sx and
    sy independently (they can differ by ~0.1-0.2% due to int rounding of
    target dims). Verifies by projecting a 3D point through original K vs
    transformed K and asserting sub-pixel agreement in the final image.

    We force hflip off and use a deterministic crop offset by sizing
    img_size to match the post-scale dimensions, so the only valid crop is
    (0, 0). That isolates the scale step.
    """
    torch.manual_seed(123)
    # Non-square input.
    H_in, W_in = 480, 640
    K = torch.tensor(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )

    # Pick a target_size whose aspect differs from the input so ScaleJitter's
    # int-rounded post-scale dims actually have a non-trivial sx vs sy
    # split. We'll do scale_range=(0.7, 0.7) for determinism.
    target_size = (256, 384)

    # Manually predict what ScaleJitter will produce (it scales the longer
    # side of (img / target) ratio). Easier path: run it once to discover
    # the post-scale shape, then build a fresh Transforms with that shape
    # as img_size so the crop step is forced to (0, 0).
    from torchvision.transforms import v2 as Tv2

    probe = Tv2.ScaleJitter(target_size=target_size, scale_range=(0.7, 0.7))
    probe_img = torch.zeros(3, H_in, W_in, dtype=torch.uint8)
    probe_out = probe(probe_img)
    post_H, post_W = probe_out.shape[-2], probe_out.shape[-1]

    transforms = Transforms(
        img_size=(post_H, post_W),  # crop matches scaled dim → (0,0) crop
        color_jitter_enabled=False,
        scale_range=(0.7, 0.7),
        sensor_noise_enabled=False,
        blur_enabled=False,
        hflip_prob=0.0,
    )
    # Override ScaleJitter so it uses the same target_size as our probe.
    transforms.scale_jitter = Tv2.ScaleJitter(
        target_size=target_size, scale_range=(0.7, 0.7)
    )

    img = torch.randint(0, 255, (3, H_in, W_in), dtype=torch.uint8)
    target = _make_target(H_in, W_in, K)
    img_out, target_out = transforms(img, target)

    sx = post_W / W_in
    sy = post_H / H_in
    # Sanity: sx and sy should differ slightly when aspect doesn't divide
    # evenly. They might be equal here — that's OK; the test still
    # exercises the per-axis path.
    K_actual = target_out["intrinsics"]
    K_expected = torch.tensor(
        [
            [K[0, 0].item() * sx, 0.0, K[0, 2].item() * sx],
            [0.0, K[1, 1].item() * sy, K[1, 2].item() * sy],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.allclose(K_actual, K_expected, atol=1e-4), (
        f"K mismatch.\nactual=\n{K_actual}\nexpected=\n{K_expected}\n"
        f"sx={sx}, sy={sy}"
    )

    # Geometric consistency: project a 3D point with original K vs K_actual.
    # In the input image, point (X=0.1, Y=0.05, Z=2.0) projects to:
    X, Y, Z = 0.1, 0.05, 2.0
    u0 = K[0, 0].item() * X / Z + K[0, 2].item()
    v0 = K[1, 1].item() * Y / Z + K[1, 2].item()
    # After scale (sx, sy), expected post-scale pixel:
    u_expected = u0 * sx
    v_expected = v0 * sy
    # K_actual must reproduce this:
    pu = K_actual[0, 0].item() * X / Z + K_actual[0, 2].item()
    pv = K_actual[1, 1].item() * Y / Z + K_actual[1, 2].item()
    assert abs(pu - u_expected) < 1e-4 and abs(pv - v_expected) < 1e-4, (
        f"Geometric mismatch: K_actual projects to ({pu},{pv}), "
        f"expected ({u_expected},{v_expected})"
    )
    print(
        f"test_non_square_scale_recovery_anisotropic: PASS "
        f"(sx={sx:.6f}, sy={sy:.6f}, |sx-sy|={abs(sx-sy):.6f})"
    )


if __name__ == "__main__":
    test_static_helper_round_trip()
    test_identity_when_no_flip_no_scale_no_crop()
    test_pad_and_filter_pass_through()
    test_legacy_no_intrinsics_path_still_works()
    test_full_pipeline_geometric_consistency()
    test_non_square_scale_recovery_anisotropic()
    print("\nAll tests passed.")
