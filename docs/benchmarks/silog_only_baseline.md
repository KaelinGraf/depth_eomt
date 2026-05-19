# Baseline — SI-Log depth only (run `hw3te1ap`)

The reference run used to compare the upcoming **curriculum** run
(`silog + depth_grad + normal + normal_grad + consistency`) against. This
run trained with **only** the SI-Log depth term active (plus the existing
panoptic mask / dice / class / occlusion losses). No depth-gradient loss,
no normal prediction, no consistency.

| context | value |
|---|---|
| wandb | `kgra470-iscar-plus/eomt/hw3te1ap` |
| commit at start | `8d1f2b2` (pre-curriculum integration) |
| config | `configs/dinov3/occlusion_bp/panoptic/eomt_large_640.yaml` |
| resolution / batch | 640×640 / bs=4 |
| stopped after | epoch 4 (`global_step 17704`, `_runtime ≈ 1h55m`) |
| reason for stop | swap to curriculum run |

## Val metrics (epoch 4)

| metric | value | epoch-0 reading | trend |
|---|---:|---:|---|
| `val_depth_absrel` ↓ | **0.0749** | 0.1063 | −29 % |
| `val_depth_rmse` ↓  | **0.0835** m | 0.1107 m | −25 % |
| `val_depth_delta1` ↑ | **0.9705** | 0.9185 | +5.2 pp |
| `val_depth_silog` ↓ | **0.0593** | 0.0783 | −24 % |
| `val_pq_all` ↑      | **0.7411** | 0.6607 | +8.0 pp |
| `val_pq_things` ↑   | **0.5733** | 0.4974 | +7.6 pp |
| `val_pq_stuff` ↑    | **0.9089** | 0.8242 | +8.5 pp |
| `train_loss_total`  | 20.46 | ~32 | — |
| `train_loss_depth` (SI-Log)  | 0.113 | 0.157 | −28 % |

## Known limitations of this baseline (what the curriculum should improve)

1. **"Hilly landscape" depth surface** — per 3-D point-cloud diagnostics on
   epoch-1 val frames, parts have rounded relief instead of sharp tops + cliff
   edges. The pixel-wise SI-Log loss is minimised by smoothness, so more
   epochs of this baseline are unlikely to fix it. Curriculum addresses this
   with `loss_normal` + `loss_normal_grad` + `loss_consistency`.
2. **~9 % global depth scale offset** (pred reads 1.07 m where GT is 0.99 m).
   SI-Log is scale-invariant-ish; the gradient term in the curriculum is *not*
   and should help pin scale.
3. **Bin-wall / height under-determined**, especially under the per-frame
   bin-scale + position randomisation. Normals attack this directly by
   supervising wall orientation.

## How to read a comparison

The curriculum run should be expected to:
- match or beat all of `absrel / rmse / silog` once `aux_ramp` has reached 1
  (after `aux_loss_warmup_steps = 10_000`, ≈ epoch 3 at bs=4),
- improve `delta1` (sharper surfaces → fewer pixels in the soft-edge tail),
- hold `val_pq_*` within ±2 pp of this baseline — the panoptic guardrail
  (training_plan.md §3 Stage B).

If curriculum AbsRel is within ±5 % of baseline but the qualitative 3-D
reconstruction shows crisp part edges and a believable bin floor, that's
a *win* — the metrics will under-credit the improvement (AbsRel is dominated
by the easy 90 % of pixels; structure quality lives in the long tail).
