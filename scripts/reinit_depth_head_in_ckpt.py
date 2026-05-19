#!/usr/bin/env python
"""Re-initialise the depth_head sub-module weights in a training checkpoint.

Motivation: during the broken Stage A phase (wd=0.05 + fp16 underflow), the
depth_head weights got shrunk to ~half their initial magnitude. The head is now
stuck producing a near-constant ~1.5m output dominated by its (also shrunk)
final bias. Loading fresh DPT init into just the depth_head, while preserving
everything else (encoder, panoptic heads at PQ 73.5, etc.), gives depth a clean
slate without throwing away ~17 hours of panoptic training.

Approach: instantiate a fresh `DPT` with identical architecture, copy its
fresh weights into the training ckpt's state_dict (overwriting the shrunk
ones), and zero out the depth_head's optimizer state so Adam starts fresh
on those params.
"""
import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.dpt import DPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source ckpt to read")
    p.add_argument("--dst", required=True, help="Destination ckpt to write")
    p.add_argument("--embed_dim", type=int, default=1024, help="ViT-L embed_dim")
    p.add_argument("--patch_size", type=int, default=16)
    args = p.parse_args()

    print(f"Loading {args.src}...")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Construct a fresh depth_head with the same hparams as the training one
    # (must match the constructor call in models/eomt.py exactly).
    torch.manual_seed(0)
    fresh = DPT(
        dim_in=args.embed_dim,
        patch_size=args.patch_size,
        output_dim=1,
        activation="sigmoid",
        features=256,
        out_channels=(256, 512, 1024, 1024),
        pos_embed=False,
        down_ratio=1,
        head_name="depth",
        use_sky_head=False,
        norm_type="idt",
        fusion_block_inplace=False,
    )
    fresh_sd = fresh.state_dict()

    # Overwrite ALL depth_head.* keys in the training ckpt with fresh ones.
    prefix = "network.depth_head."
    depth_keys = [k for k in sd if k.startswith(prefix)]
    print(f"Found {len(depth_keys)} depth_head keys in source ckpt")
    n_replaced = 0
    n_missing = 0
    for k in depth_keys:
        local = k[len(prefix):]
        if local not in fresh_sd:
            print(f"  ! key {local} not in fresh DPT — leaving as-is")
            n_missing += 1
            continue
        old_rms = sd[k].float().norm().item() / max(1, sd[k].numel()) ** 0.5
        new_w = fresh_sd[local].to(sd[k].dtype)
        new_rms = new_w.float().norm().item() / max(1, new_w.numel()) ** 0.5
        print(f"  {local}: RMS {old_rms:.6f} → {new_rms:.6f}")
        sd[k] = new_w
        n_replaced += 1
    # Also check for keys in fresh that aren't in src (shouldn't happen).
    for local in fresh_sd:
        if prefix + local not in sd:
            print(f"  ! fresh key {local} not in source — added")
            sd[prefix + local] = fresh_sd[local]
    print(f"Replaced {n_replaced}, missing {n_missing}")

    # Reset Adam optimizer state for any depth_head params. Otherwise the
    # accumulated first/second moments from the shrunk regime would drag the
    # fresh weights right back down.
    opt_states = ckpt.get("optimizer_states", [])
    if opt_states:
        opt = opt_states[0]
        # Map param id (int) to its param-group name (we look up which name
        # corresponds to the depth_head subtree via the param-group `name`).
        param_groups = opt.get("param_groups", [])
        # AdamW-style: each param_group has {'params': [id], 'name': 'network.depth_head.xxx', ...}
        depth_param_ids = set()
        for pg in param_groups:
            name = pg.get("name", "")
            if name.startswith(prefix):
                for pid in pg.get("params", []):
                    depth_param_ids.add(pid)
        print(f"Found {len(depth_param_ids)} depth_head param IDs in optimizer state")
        state = opt.get("state", {})
        reset_count = 0
        for pid in list(depth_param_ids):
            if pid in state:
                # Zero out exp_avg and exp_avg_sq (Adam's m and v).
                st = state[pid]
                for k_st in ("exp_avg", "exp_avg_sq"):
                    if k_st in st and isinstance(st[k_st], torch.Tensor):
                        st[k_st].zero_()
                # Reset step counter to 0 so Adam's bias correction restarts.
                if "step" in st:
                    if isinstance(st["step"], torch.Tensor):
                        st["step"].zero_()
                    else:
                        st["step"] = 0
                reset_count += 1
        print(f"Reset Adam state for {reset_count} depth_head params")

    print(f"Saving to {args.dst}...")
    torch.save(ckpt, args.dst)
    print("Done.")


if __name__ == "__main__":
    main()
