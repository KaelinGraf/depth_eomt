"""Phase 0 sanity: freeze-all-but-depth-head training run (Task #12).

Loads the production yaml's MaskClassificationPanoptic + ReplicatorDataModule
at 1280x1280, freezes everything except `depth_head` and `intrinsics_mlp`,
overrides lr=1e-3 and max_epochs=2, and runs Trainer.fit().

Three execution modes:

  - `--smoke-only`: one forward+backward, validates wiring + frozen set + memory.
  - `--dry-run-steps N`: N optimizer steps (default 20) with autocast + scaler.
    Validates non-divergent loss trajectory and exercises the optimizer
    update path without committing to the full convergence run.
  - default: full Trainer.fit() at `max_epochs=2`. **Only run when a human
    has explicitly authorised the compute commitment** — the full path is
    ~6 hours of GPU at 1280×1280.

NOT committed; one-off diagnostic.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

sys.path.insert(0, "/home/kaelin/BinPicking/eomt")
os.chdir("/home/kaelin/BinPicking/eomt")  # ckpt_path is relative in yaml

from training.mask_classification_panoptic import MaskClassificationPanoptic
from datasets.iscar_bp import ReplicatorDataModule
from models.eomt import EoMT
from models.vit import ViT


# Defaults mirror configs/dinov3/occlusion_bp/panoptic/eomt_large_1280.yaml
IMG_SIZE = (1280, 1280)
NUM_CLASSES = 2
STUFF_CLASSES = [0]
CKPT_PATH = "checkpoints/eomt_large_640.bin"
DATA_PATH = "/home/kaelin/BinPicking/SDG/IS/Outputs/monocular_dataset"
OUTPUT_DIR = Path("/home/kaelin/BinPicking/eomt/scripts/phase0_outputs")


def build_model() -> MaskClassificationPanoptic:
    encoder = ViT(
        img_size=IMG_SIZE,
        patch_size=16,
        backbone_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=200,
        num_blocks=4,
        masked_attn_enabled=True,
        enable_occlusion=True,
        depth_taps=(4, 11, 17, 23),
        use_intrinsics=True,
    )
    model = MaskClassificationPanoptic(
        network=network,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        stuff_classes=STUFF_CLASSES,
        attn_mask_annealing_enabled=True,
        attn_mask_annealing_start_steps=[0, 118256, 177384, 236512],
        attn_mask_annealing_end_steps=[12000, 177384, 236512, 295640],
        lr=1e-3,  # Phase 0 override (yaml is 1e-4)
        llrd=0.8,
        llrd_l2_enabled=False,
        lr_mult=1.0,
        weight_decay=0.05,
        poly_power=0.9,
        warmup_steps=[2000, 3000],
        depth_coefficient=1.0,
        ckpt_path=CKPT_PATH,
        delta_weights=True,
        load_ckpt_class_head=False,
        use_area_weighting=True,
    )
    return model


def freeze_all_but_depth_and_intrinsics(model: MaskClassificationPanoptic) -> None:
    """Freeze everything except depth_head and intrinsics_mlp."""
    n_trainable = 0
    n_frozen = 0
    for name, p in model.named_parameters():
        is_trainable = ("depth_head" in name) or ("intrinsics_mlp" in name)
        p.requires_grad = is_trainable
        if is_trainable:
            n_trainable += p.numel()
        else:
            n_frozen += p.numel()
    logging.info(
        "Phase 0 freeze: %d trainable, %d frozen", n_trainable, n_frozen
    )


def smoke_test_single_step(model: MaskClassificationPanoptic, dm: ReplicatorDataModule) -> None:
    """Run ONE training_step on ONE batch. Assert:
        a) forward + backward complete.
        b) Only depth_head and intrinsics_mlp params have non-None grad.
        c) loss_depth is finite AND nonzero.
        d) Report peak GPU memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    imgs, targets = batch
    imgs = imgs.to(device)
    targets = [{k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # (a) forward + backward — mirror Trainer "16-mixed" autocast.
    # Forward in fp16 autocast, loss in fp32 (Lightning's behaviour).
    model.zero_grad(set_to_none=True)
    loss_depth = _compute_depth_loss(model, imgs, targets, use_autocast=True)
    loss_depth.backward()

    # (b) Only depth_head + intrinsics_mlp have grad
    bad_with_grad = []
    for name, p in model.named_parameters():
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        is_target = ("depth_head" in name) or ("intrinsics_mlp" in name)
        if has_grad and not is_target:
            bad_with_grad.append(name)
        if is_target and not has_grad:
            # Expected to receive grad through the depth path.
            pass
    n_dh_with_grad = sum(
        1
        for n, p in model.named_parameters()
        if ("depth_head" in n) and p.grad is not None and p.grad.abs().sum() > 0
    )
    n_im_with_grad = sum(
        1
        for n, p in model.named_parameters()
        if ("intrinsics_mlp" in n) and p.grad is not None and p.grad.abs().sum() > 0
    )

    # (c) finite, non-zero
    assert torch.isfinite(loss_depth), f"loss_depth not finite: {loss_depth.item()}"
    assert loss_depth.item() != 0.0, f"loss_depth is exactly zero (likely all-invalid mask)"

    # (d) peak memory
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"[SMOKE] loss_depth = {loss_depth.item():.4f}")
    print(f"[SMOKE] depth_head params with non-zero grad: {n_dh_with_grad}")
    print(f"[SMOKE] intrinsics_mlp params with non-zero grad: {n_im_with_grad}")
    print(f"[SMOKE] params OUTSIDE depth+intr that received grad: {len(bad_with_grad)}")
    if bad_with_grad:
        print(f"[SMOKE]   first few unexpected: {bad_with_grad[:5]}")
    print(f"[SMOKE] peak GPU memory: {peak_mb:.0f} MB")

    assert not bad_with_grad, (
        f"Frozen params received grad: {bad_with_grad[:5]} (+{len(bad_with_grad)-5} more)"
    )
    assert n_dh_with_grad > 0, "No depth_head params received grad"
    # intrinsics_mlp may not receive grad if the loss-graph attribution skips
    # it on this particular batch; we allow zero but log it.
    print(f"[SMOKE] ALL ASSERTIONS PASSED")
    return peak_mb


def _compute_depth_loss(model, imgs, targets, use_autocast: bool = True):
    """Forward + criterion, returning the SI-Log depth loss tensor.

    Forward (and matcher/criterion internals) run inside autocast to match
    Lightning's `precision="16-mixed"` behaviour. The depth tensor is
    upcast to fp32 BEFORE entering the SI-Log path: `log(d_pred.clamp_min(1e-6))`
    underflows in fp16 because 1e-6 < fp16 min normal (~6e-5), producing
    log(0) = -inf → NaN. Upcasting only depth_pred keeps the masking and
    classification losses in fp16 for memory savings.
    """
    intrinsics = None
    if targets and "intrinsics" in targets[0]:
        intrinsics = torch.stack(
            [t["intrinsics"] for t in targets], dim=0
        ).to(imgs.device, imgs.dtype)

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_autocast
        else torch.amp.autocast(device_type="cuda", enabled=False)
    )
    with ctx:
        (
            mask_logits_per_block,
            class_logits_per_block,
            occlusion_logits_per_block,
            depth_pred,
            _,
        ) = model(imgs, intrinsics=intrinsics)

        # Upcast depth_pred outside the autocast scope but inside the same
        # criterion call. The criterion handles mask/class/occlusion in fp16
        # (autocast-friendly) and SI-Log in fp32 (numerically safe).
        depth_pred_fp32 = depth_pred.float() if depth_pred is not None else None
        depth_gt = None
        if depth_pred_fp32 is not None and "depth" in targets[0]:
            depth_gt = torch.stack(
                [t["depth"] for t in targets], dim=0
            ).to(depth_pred_fp32.device, depth_pred_fp32.dtype)

        last_block_idx = len(mask_logits_per_block) - 1
        losses = model.criterion(
            masks_queries_logits=mask_logits_per_block[last_block_idx],
            class_queries_logits=class_logits_per_block[last_block_idx],
            occlusion_queries_logits=occlusion_logits_per_block[last_block_idx],
            targets=targets,
            depth_pred=depth_pred_fp32,
            depth_gt=depth_gt,
        )
    return losses["loss_depth"]


def _move_targets_to_device(targets, device):
    return [
        {k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()}
        for t in targets
    ]


def _submodule_key(name: str) -> str:
    """Group a fully-qualified param name into a coarse submodule bucket."""
    parts = name.split(".")
    if parts[0] == "network" and len(parts) >= 2:
        if parts[1] == "encoder":
            return "encoder"
        return parts[1]
    return parts[0]


def dry_run(
    model: MaskClassificationPanoptic,
    dm: ReplicatorDataModule,
    n_steps: int,
    base_lr: float = 1e-3,
    lr_warmup_steps: int = 0,
    use_autocast: bool = True,
) -> dict:
    """Run N optimizer steps. Returns loss trajectory + per-submodule grad audit.

    Mirrors Lightning's `precision="16-mixed"` path: AdamW over `requires_grad`
    params only, `torch.amp.autocast(fp16)` forward, `GradScaler` backward,
    `gradient_clip_val=0.01` clip on grad norm.

    `lr_warmup_steps > 0` enables a linear warmup from `base_lr * 0.01` to
    `base_lr` over the first N steps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in trainable_named]
    frozen_named = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_autocast)

    dm.setup("fit")
    loader = dm.train_dataloader()
    data_iter = iter(loader)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    losses: list[float] = []
    nan_steps: list[int] = []
    leak_audit_done = False
    leaks: set[str] = set()
    seen_nonzero_grad: set[str] = set()  # trainable params that ever had nonzero grad
    submod_max_norm: dict[str, float] = {}  # submodule → max ||grad|| across run

    for step in range(n_steps):
        # Linear LR warmup
        if lr_warmup_steps > 0 and step < lr_warmup_steps:
            warmup_factor = 0.01 + 0.99 * (step / max(1, lr_warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * warmup_factor

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        imgs, targets = batch
        imgs = imgs.to(device)
        targets = _move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        loss_depth = _compute_depth_loss(
            model, imgs, targets, use_autocast=use_autocast
        )
        loss_val = float(loss_depth.detach())
        if not (loss_val == loss_val) or loss_val in (float("inf"), float("-inf")):
            nan_steps.append(step + 1)

        if use_autocast:
            scaler.scale(loss_depth).backward()
            scaler.unscale_(optimizer)
        else:
            loss_depth.backward()

        # Per-step grad audit: track grad norm per submodule, dead-part detection,
        # leaky-freeze detection. Frozen audit only needs to happen once because
        # requires_grad=False guarantees grad is None — but verify defensively.
        for name, p in trainable_named:
            if p.grad is not None:
                gn = float(p.grad.detach().abs().norm().item())
                if gn > 0:
                    seen_nonzero_grad.add(name)
                key = _submodule_key(name)
                if gn > submod_max_norm.get(key, 0.0):
                    submod_max_norm[key] = gn

        if not leak_audit_done:
            for name, p in frozen_named:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    leaks.add(name)
            leak_audit_done = True

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.01)
        if use_autocast:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        losses.append(loss_val)
        # Print every step for the first 10, then every 10th step thereafter,
        # plus on NaN.
        is_nan = step + 1 in nan_steps
        if step < 10 or (step + 1) % 10 == 0 or is_nan:
            lr_now = optimizer.param_groups[0]["lr"]
            tag = "  NaN!" if is_nan else ""
            print(f"[DRY] step {step + 1:4d}/{n_steps}  lr={lr_now:.2e}  loss_depth={loss_val:.4f}{tag}")

    wall = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Dead-part detection: any trainable param that never saw a nonzero grad.
    dead_parts = sorted(
        n for n, _ in trainable_named if n not in seen_nonzero_grad
    )

    return {
        "losses": losses,
        "wall_s": wall,
        "peak_mb": peak_mb,
        "nan_steps": nan_steps,
        "leaks": sorted(leaks),
        "dead_parts": dead_parts,
        "submod_max_norm": submod_max_norm,
        "n_trainable": len(trainable_named),
        "n_frozen": len(frozen_named),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true", help="Run only the smoke step and exit")
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=0,
        help="Run N optimizer steps with autocast + scaler, then exit. "
             "0 disables (default). Use this for scope-C validation.",
    )
    parser.add_argument(
        "--dry-run-lr",
        type=float,
        default=1e-3,
        help="Base LR for the dry-run optimizer. Spec value is 1e-3. Lower "
             "for diagnostic NaN exploration without changing the Phase-0 spec.",
    )
    parser.add_argument(
        "--dry-run-lr-warmup-steps",
        type=int,
        default=0,
        help="Linear LR warmup from `dry_run_lr * 0.01` to `dry_run_lr` "
             "over the first N steps. 0 disables.",
    )
    parser.add_argument(
        "--dry-run-precision",
        choices=["16-mixed", "32-true"],
        default="16-mixed",
        help="Precision mode for the dry-run. Matches Lightning's option name.",
    )
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------- Build ---------
    print("[BUILD] Constructing model + datamodule …")
    t0 = time.time()
    model = build_model()
    print(f"[BUILD] model constructed in {time.time() - t0:.1f}s")
    freeze_all_but_depth_and_intrinsics(model)

    dm = ReplicatorDataModule(
        path=DATA_PATH,
        stuff_classes=STUFF_CLASSES,
        num_workers=4,
        batch_size=1,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        color_jitter_enabled=True,
        sensor_noise_enabled=True,
        blur_enabled=True,
        scale_range=(1.0, 1.0),
    )

    # --------- Smoke ---------
    print("\n[SMOKE] Running single-step smoke test …")
    t0 = time.time()
    peak_mb = smoke_test_single_step(model, dm)
    print(f"[SMOKE] step time {time.time() - t0:.1f}s, peak {peak_mb:.0f} MB\n")

    if args.smoke_only:
        return

    # --------- Dry-run (scope C) ---------
    if args.dry_run_steps > 0:
        use_autocast = args.dry_run_precision == "16-mixed"
        print(
            f"\n[DRY] Running {args.dry_run_steps}-step dry-run "
            f"(lr={args.dry_run_lr:g}, warmup={args.dry_run_lr_warmup_steps}, "
            f"precision={args.dry_run_precision}) …"
        )
        result = dry_run(
            model,
            dm,
            n_steps=args.dry_run_steps,
            base_lr=args.dry_run_lr,
            lr_warmup_steps=args.dry_run_lr_warmup_steps,
            use_autocast=use_autocast,
        )
        ls = result["losses"]
        finite_ls = [x for x in ls if x == x and x not in (float("inf"), float("-inf"))]
        n_first = min(5, len(ls))
        n_last = min(5, len(ls))
        print(f"\n[DRY-SUMMARY] steps:               {len(ls)}")
        print(f"[DRY-SUMMARY] first {n_first} losses:    {[f'{x:.4f}' for x in ls[:n_first]]}")
        print(f"[DRY-SUMMARY] last  {n_last} losses:    {[f'{x:.4f}' for x in ls[-n_last:]]}")
        if finite_ls:
            print(f"[DRY-SUMMARY] min / max finite loss: {min(finite_ls):.4f} / {max(finite_ls):.4f}")
        print(f"[DRY-SUMMARY] NaN/Inf step count:    {len(result['nan_steps'])}")
        if result["nan_steps"]:
            print(f"[DRY-SUMMARY]   first NaN at step: {result['nan_steps'][0]}")
        print(f"[DRY-SUMMARY] wall-clock:            {result['wall_s']:.1f}s "
              f"({result['wall_s'] / max(1, len(ls)):.2f}s/step)")
        print(f"[DRY-SUMMARY] peak GPU memory:       {result['peak_mb']:.0f} MB")

        # Per-submodule max grad-norm (across the whole run).
        print(f"[DRY-SUMMARY] trainable params:      {result['n_trainable']}")
        print(f"[DRY-SUMMARY] frozen params:         {result['n_frozen']}")
        print(f"[DRY-SUMMARY] per-submodule max ||grad|| (trainable only):")
        for submod, mx in sorted(result["submod_max_norm"].items()):
            print(f"[DRY-SUMMARY]   {submod:20s}  {mx:.3e}")

        # Verdicts.
        dead = result["dead_parts"]
        leaks = result["leaks"]
        print()
        if dead:
            print(f"[DRY-VERDICT] dead parts: {dead[:10]}"
                  + (f"  (+{len(dead) - 10} more)" if len(dead) > 10 else ""))
        else:
            print("[DRY-VERDICT] no dead parts detected")
        if leaks:
            print(f"[DRY-VERDICT] leaky freeze: {leaks[:10]}"
                  + (f"  (+{len(leaks) - 10} more)" if len(leaks) > 10 else ""))
        else:
            print("[DRY-VERDICT] no leaky freeze")
        print(f"[DRY-VERDICT] NaN events: {len(result['nan_steps'])}")
        # Assert hard fail conditions; soft fails (NaN) are reported, not raised,
        # because the human asked for diagnostic-not-gate.
        assert not leaks, "Frozen-set leak detected — see verdict above"
        return

    # --------- Full Fit ---------
    print(f"[FIT] Starting {args.max_epochs}-epoch Phase 0 fit …")
    csv_logger = CSVLogger(save_dir=str(OUTPUT_DIR), name="phase0_logs")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(OUTPUT_DIR / "checkpoints"),
        filename="phase0-{epoch}-{step}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=1,
        precision="16-mixed",
        gradient_clip_val=0.01,
        gradient_clip_algorithm="norm",
        logger=csv_logger,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=20,
        enable_model_summary=False,
        check_val_every_n_epoch=1,
    )
    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    wall = time.time() - t0
    print(f"\n[FIT] DONE in {wall:.0f}s ({wall / 60:.1f} min)")
    print(f"[FIT] Final ckpt: {ckpt_cb.last_model_path}")
    print(f"[FIT] CSV log dir: {csv_logger.log_dir}")
    print(f"[FIT] Peak GPU memory (post-fit): {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()
