# Training Plan: EoMT × DA3 Monocular Depth (640×640)

This is the **training-execution** plan for the depth-augmented EoMT model.
It is the sequel to `docs/depth_integration_plan.md` (which covered the
*code* integration — all 17 of its tasks are closed) and assumes
`docs/depth_knowledge.md` as the DA3 reference. Read both first.

> **Active config: `configs/dinov3/occlusion_bp/panoptic/eomt_large_640.yaml`.**
> The dataset on disk is 640×640, so this is the config to train with —
> it matches the data with no upscaling waste. `eomt_large_1280.yaml`
> exists for a future regenerate-at-1280 path; numbers for it are noted
> in-line where they differ.

The user-level instruction this plan expands:

> For epochs 0–10, freeze DINO to let the randomised heads stabilise.
> Then, until convergence, unfreeze DINO so it can adapt to the
> requirements of depth estimation.

That instinct is sound and is honoured below — but it is enriched with
what Depth Anything 3 (arXiv:2511.10647, the paper the DPT head is ported
from) actually did, and with how the freeze/unfreeze maps onto machinery
the EoMT codebase **already has**.

---

## 1. What DA3 actually did (paper-grounded)

Relevant sections of the paper: §3.3 (training objectives), §3.4
(implementation details), §4.4 (metric model), §5.2 (the 3DGS
application).

| Aspect | DA3 main model (§3.4) | DA3 metric model (§4.4) | DA3 3DGS app (§5.2) |
|---|---|---|---|
| Backbone | **Trained throughout**, never frozen | **Trained throughout** at low LR | **Frozen**, only the GS-DPT head trained |
| Warmup | 8k steps / 200k total (4%) | — | — |
| Peak LR | 2×10⁻⁴ | encoder **5×10⁻⁶**, decoder **5×10⁻⁵** (decoder = 10× encoder) | — |
| Optimiser | — | AdamW | — |
| Batch | dynamic, token-count-constant; ablations bs=128 on 32×H100 | bs=64, 160k iters | — |
| Loss | L1 depth + confidence + log term, **+ gradient loss** (α=1) | depth + gradient + sky-mask | photometric + scale-shift-invariant depth |
| Resolution | 504 base, multi-res sampled | 504, varying aspect | varying |
| Depth repr. | **exponential depth** (better near-camera discrimination) | same | — |
| Supervision | GT depth → teacher pseudo-labels at step 120k (60%) | 20% GT / 80% teacher | teacher-student |

**The two facts that matter most for our plan:**

1. **DA3 does *not* freeze the backbone to train depth.** The only place
   a frozen backbone appears in the whole paper is the *downstream 3DGS
   application* (§5.2: *"we initialize the DA3 backbone from pretrained
   weights and freeze it when training, tuning only the GS-DPT head … to
   avoid unstable training"*). For the depth models themselves, DA3's way
   of "letting the backbone adapt slowly" is **(a) an 8k-step warmup**
   and, for the metric model, **(b) a 10× lower encoder LR than the
   decoder** — not a hard freeze.

2. **DA3 always pairs the depth loss with a gradient-matching loss**
   (`L_grad`, α=1, eq. 3) for edge sharpness. Our criterion
   (`training/depth_loss.py`) is SI-Log only.

So the user's freeze-then-unfreeze is a *more conservative* variant of
DA3's recipe. It is reasonable here — our backbone is not a generic
DINO, it is **EoMT-panoptic-fine-tuned** and shared with the downstream
grasp model, so protecting it from a cold, fresh-head gradient blast for
the first few epochs is well-motivated. This plan keeps the hard freeze
as **Stage A**, then in **Stage B** unfreezes the backbone *at a low,
warmed-up LR* — i.e. it merges the user's plan with DA3's "warmup +
discriminative LR" mechanism rather than snapping the backbone straight
to the head LR.

A documented alternative (DA3-faithful, less conservative) is given in
§9.

---

## 2. Our setup (recap)

- **Model**: `MaskClassificationPanoptic` → `EoMT` → DINOv3 ViT-L/16
  shared backbone, 24 blocks, 200 queries prepended at block 20.
  Multi-task heads: `mask_head` + `upscale` + `q` (panoptic),
  `class_head`, `occlusion_head`, `depth_head` (DPT), `intrinsics_mlp`.
- **Checkpoint**: `checkpoints/eomt_large_640.bin`, `delta_weights: true`.
  From the Phase-0 dry-run log, loading this ckpt **re-initialises**
  `class_head`, `occlusion_head`, `depth_head.*`, `intrinsics_mlp.*`
  (~32M params) — the backbone, `mask_head`, `upscale`, `q` load from
  the checkpoint and are already panoptic-competent.
- **Dataset**: `/home/kaelin/BinPicking/SDG/IS/Outputs/monocular_dataset`
  — **14,178 train / 1,577 val** frames (synthetic Replicator, clean
  metric depth GT, per-frame randomised intrinsics, **640×640** —
  RGB, `depth.npy` and `normals.npy` all verified 640² on disk).
  At `batch_size: 4` (the 640 yaml) that is **3,544 steps/epoch**
  (drop_last) → **354,400 steps** over 100 epochs. The
  `attn_mask_annealing_*_steps` have been **recomputed** for this
  (both yamls — §8.6 is done).
- **Hard constraint**: the encoder is **shared with the downstream grasp
  model**. Stage B must not let depth gradients degrade panoptic /
  occlusion quality — the panoptic losses stay on throughout, and PQ is a
  monitored guardrail (§7).
- **Convergence budget**: ~40–60 epochs total expected (Stage A 0–10,
  Stage B ~10–50), `max_epochs: 100` leaves headroom. See the frame-count
  estimate already on file: 14k unique frames is sufficient for
  in-distribution convergence.

The "frozen heads stabilise" target in Stage A is **all four fresh heads**
(`class_head`, `occlusion_head`, `depth_head`, `intrinsics_mlp`), not just
depth — they were all re-initialised by the `delta_weights` load.

---

## 3. The two-stage plan

### Stage A — epochs 0–10: frozen backbone, heads stabilise

**Trainable**: everything *except* `network.encoder.backbone.*`
(i.e. all heads — pretrained `mask_head`/`upscale`/`q` *and* the four
fresh heads). Backbone in eval-grad-off.

**Loss**: full multi-task objective — `mask`, `dice`, `class`,
`occlusion`, **`depth` (SI-Log)**. This is *not* a depth-only phase; the
fresh `class_head` and `occlusion_head` also need to stabilise against
the frozen encoder.

**LR**: heads at `lr = 2e-4` (the 640 yaml value; matches DA3's main-model
peak LR), warmed up over ~2000 steps then poly-decay. Backbone LR = 0
(frozen).

**Why 10 epochs**: a frozen-backbone DPT head on this narrow synthetic
domain converges fast — the Phase-0 smoke already showed a sane
`loss_depth = 0.173` at init. 10 epochs (**35,440 steps** at bs=4) is
generous insurance, not a tight requirement. Treat it as a *ceiling*:
if the head losses (`loss_depth`, `loss_class`, `loss_occlusion`) have
clearly plateaued by epoch 6–8, the unfreeze can be brought forward —
the epoch-10 boundary should be a yaml knob, not a constant.

**Exit criteria (all must hold before unfreezing)**:
- `loss_depth` curve has flattened (relative slope per epoch < ~5%).
- `loss_class`, `loss_occlusion` plateaued — no longer in their initial
  steep drop.
- Validation depth metrics (AbsRel, δ1 — see §8 prerequisite) are stable
  epoch-to-epoch.
- No NaN/inf in any loss; gradient norms on the heads are O(1), not
  exploding or vanished.

### Stage B — epoch 10 → convergence: unfreeze backbone, adapt to depth

**Trainable**: everything.

**LR**: the unfreeze must not blast a cold backbone with the head LR.
At the unfreeze boundary the backbone LR **warms up from 0** over a
`vit_warmup` window (DA3 used 4% of training; 3k–8k steps here), then
follows poly-decay. Its **peak is held well below the head LR** — DA3's
metric model used a 10× encoder/decoder gap; mirror that:

- heads: peak `2e-4` (already decaying on poly from Stage A)
- backbone: peak `~2e-5` (≈10× lower), via `llrd` / `lr_mult` (§6, §10)

**Loss**: same multi-task objective, **plus gradient-matching loss**
added to the depth term (DA3 eq. 3; `docs/depth_knowledge.md` §6.5
recommends adding it once SI-Log is stable — Stage B is exactly that
point). Optionally enable intrinsics-dropout here (§7).

**Panoptic guardrail**: PQ / mask-AP / occlusion-MAE on the val set must
stay within ≈±2 pp of their Stage-A-end values. A drop means depth
gradients are corrupting the shared features — respond by lowering
`depth_coefficient` (1.0 → 0.5) or the backbone LR, *not* by re-freezing.

**Exit criteria (convergence)**:
- Val AbsRel / δ1 plateau for ≥5 consecutive epochs (early-stop trigger).
- Val `loss_depth` plateau, train/val gap not widening (widening gap ⇒
  the limit is data diversity, not training — see the frame-count note).
- Panoptic metrics within the ±2 pp guardrail.

---

## 4. How freeze/unfreeze maps onto existing code

**Key finding: the EoMT codebase already implements freeze-then-unfreeze
via the LR schedule.** `training/two_stage_warmup_poly_schedule.py`
(`TwoStageWarmupPolySchedule`) holds **backbone param-group LR at exactly
0** for the first `non_vit_warmup` steps, then ramps it over `vit_warmup`
steps, then poly-decays. Non-backbone (head) groups warm up over
`non_vit_warmup` and poly-decay from there. The split is wired in
`lightning_module.py:configure_optimizers` (`backbone_param_groups` vs
`other_param_groups`, lines 106–164). `warmup_steps` is the
`(non_vit_warmup, vit_warmup)` tuple — currently `[2000, 3000]` in the
yaml.

So a *soft* version of the user's plan is **already running** — it just
freezes the backbone for only 2000 steps (~0.14 epoch), not 10 epochs.

**The problem**: `non_vit_warmup` is a *single knob doing two jobs* — it
sets both the head warmup length **and** the backbone freeze length. We
want head warmup ≈ 2000 steps but backbone freeze ≈ 35k steps (10
epochs at bs=4). Cranking `non_vit_warmup` to 35k would also stretch the
head warmup ramp across all 10 epochs — wrong.

### Implementation — recommended: decouple the scheduler (Option A)

Modify `TwoStageWarmupPolySchedule` to take a separate
`backbone_freeze_steps` (default 0 = current behaviour):

- **head groups** (`i >= num_backbone_params`): warm over
  `non_vit_warmup`, poly-decay after — *unchanged*.
- **backbone groups**: `lr = 0` while `step < backbone_freeze_steps`;
  then linear ramp over `vit_warmup`; then poly-decay with the decay
  clock starting at `backbone_freeze_steps + vit_warmup`.

~10–15 LOC, fully contained, default-off so the panoptic-only configs
are unaffected. Plumb `backbone_freeze_steps` through
`MaskClassificationPanoptic.__init__` → `lightning_module` →
`configure_optimizers` (mirror how `warmup_steps` already threads
through), and expose it in `eomt_large_640.yaml`.

This makes the user's plan a **pure yaml change**:
`backbone_freeze_steps: 35440` (10 epochs × 3,544 steps/epoch at bs=4),
`warmup_steps: [2000, 8000]` (head warmup 2k; backbone ramp 8k after the
freeze). For the 1280 yaml the equivalent is `141780` (10 × 14,178).

### Implementation — complement: a `requires_grad` callback (Option B)

The scheduler's LR=0 makes the backbone *not update*, but
`loss.backward()` still computes backbone gradients every step — no
compute or memory saving during Stage A. The Phase-0 smoke hit
**20.6 GB peak**, but that was at 1280×1280 / bs=1; at 640×640 / bs=4
the per-image token count is ¼ and attention memory drops sharply, so
memory is far less tight here — the callback's value at 640 is mostly
the **compute** saving (no backbone backward) and a cleaner Adam state,
not OOM avoidance.

Add a small Lightning callback (`training/freeze_callback.py`, new) that
sets `requires_grad=False` on `network.encoder.backbone.*` at fit start
and flips it back `True` at `global_step == backbone_freeze_steps`. The
Phase-0 script already has the exact freeze predicate to reuse
(`scripts/phase0_freeze_train.py:91 freeze_all_but_depth_and_intrinsics`
— generalise it to "freeze backbone only").

A and B are **complementary, not alternatives**: the scheduler guarantees
LR-correctness (and a *warmed* unfreeze), the callback buys the
compute/memory saving and a cleaner Adam state (no second-moment stats
accumulated from gradients that were never applied). Params with
`requires_grad=False` may stay in the optimizer param groups safely —
AdamW skips `None` grads — so no optimizer rebuild is needed at the
unfreeze boundary.

> If only one is implemented, do **Option A** — a warmed unfreeze is the
> point of the whole plan; the callback is an optimisation.

---

## 5. Loss schedule

| Term | Stage A | Stage B | Source |
|---|---|---|---|
| `mask`, `dice`, `class`, `occlusion` | on | on | existing EoMT |
| `depth` SI-Log (`loss_depth_silog`) | on | on | `training/depth_loss.py`, `docs/depth_knowledge.md` §6.1 |
| `depth` gradient-matching (`L_grad`) | **off** | **on** | DA3 eq. 3; `depth_knowledge.md` §6.5 — "add once SI-Log converges" |
| `depth` surface-normal loss | off | **optional** | not in DA3's depth model — DA3 used it only in its *Teacher* (§4.1 eqs. 4–6). Grasp-relevant; derivable from GT depth + intrinsics, no new head. Try in Stage B if grasp-pose quality needs sharper local surfaces |
| intrinsics dropout | off | optional on | DA3 pose-cond. prob 0.2 analog (§7) |

> **Status note.** The depth *loss* (`loss_depth_silog`) is **already
> wired into training** — `mask_classification_loss.py:99-100` +
> `loss_total` weighting, fed from `training_step`. What is missing is
> depth *evaluation metrics on the val set* (§8.1) — distinct from the
> loss. Normals are **not** wired anywhere (no head, no loss); the only
> place DA3 used a normals loss is its Teacher model, not the depth
> model itself.

`depth_coefficient` starts at `1.0` (yaml). If the panoptic guardrail
trips in Stage B, drop to `0.5` before touching LRs. SI-Log is already
per-image scale-invariant, so DA3's GT scale-normalisation (§3.3) is not
strictly needed; the gradient-matching term, however, is *not*
scale-invariant — normalise GT depth per-image (or weight `L_grad` low,
~0.5×) when adding it.

---

## 6. Concrete hyperparameters

Anchored to `eomt_large_640.yaml` + DA3 §3.4 / §4.4. Numbers in **bold**
are changes from the current yaml.

| Hyper | Stage A | Stage B | Notes |
|---|---|---|---|
| `max_epochs` | — | **~60** (early-stop on val depth) | yaml says 100; cap is fine, early-stop is the real terminator |
| `backbone_freeze_steps` | **35440** (=10 epochs × 3,544 steps/epoch) | — | new knob (§4 Option A); make it a yaml value |
| `warmup_steps` | `[2000, 8000]` | — | head warmup 2k (unchanged); **backbone ramp 8k** (was 3k) — DA3 used 4% of training |
| head `lr` | `2e-4` | `2e-4` (poly-decaying) | the 640 yaml value; matches DA3's main-model peak LR |
| backbone peak LR | 0 (frozen) | **~2e-5** | ≈10× below heads, DA3-metric-style; via `llrd`/`lr_mult` — see §10 |
| `llrd` | — | `< 1` (e.g. **0.9**) + `llrd_l2_enabled: True` | so the decay reaches the last 4 blocks too (§10) |
| `accumulate_grad_batches` | optional **2** | optional **2** | bs=4 is already a usable batch; accumulation (→ effective 8) is optional, not required as it was at bs=1. Add to `trainer:` in yaml only if Stage-B gradients look noisy |
| `batch_size` | **2** | **2** | bs=4 **OOMs** (confirmed): full-model 640²/bs4 train needs ~25+ GiB (fwd+bwd over the *unfrozen* backbone ×4). The Phase-0 20.6 GiB figure is *not* comparable — it was frozen-backbone. bs=2 + `expandable_segments` fits alongside other GPU users; bs=4 only on a near-empty 32 GiB card. **bs change → recompute anneal steps (§8.6).** |
| `depth_coefficient` | `1.0` | `1.0` → `0.5` if guardrail trips | yaml |
| `attn_mask_annealing_*_steps` | `[0,141760,212640,283520]` / `[14385,212640,283520,354400]` | — | **done** — recomputed for 14,178 frames @ bs4 (§8.6) |
| precision | `16-mixed` | `16-mixed` | as Phase-0 |

---

## 7. Intrinsics conditioning during training

The `intrinsics_mlp` cam_token is a deliberate departure from DA3 mono
(`CLAUDE.md`, `depth_knowledge.md` §11.2). DA3's multi-view path
randomly *drops* pose conditioning with probability 0.2 (§3.4) so the
model degrades gracefully without it. Mirror this: in Stage B, with
probability ~0.2–0.5 per batch, pass `intrinsics=None` (the forward path
already handles `None` → no cam_token, bit-identical to pre-intrinsics
behaviour per memory chunk 9). This prevents the model becoming brittle
to intrinsics it has never seen, and is cheap insurance for the
multi-camera deployment rationale. Stage A can keep intrinsics always-on
(heads are stabilising; one fewer moving part).

---

## 8. Prerequisites — code changes before this plan can run

Items 1 and 6 are **done**; 2–5 and 7 remain before this plan can run.

1. **Depth validation metrics — ✅ DONE.** `mask_classification_panoptic.py`
   now scores depth on the val set: `init_metrics_depth()` (a
   `nn.ModuleDict` of four `MeanMetric` accumulators, created after the
   ckpt load so it doesn't trip `_raise_on_incompatible`),
   `update_metrics_depth()` (per-image **AbsRel**, **RMSE**, **δ1** =
   `max(d/d̂,d̂/d)<1.25`, and **SI-Log** over the validity mask;
   resamples pred→GT grid if the eval transform letterboxed; skips
   targets with no `depth`), and `_on_eval_epoch_end_depth()` (logs
   `metrics/val_depth_{absrel,rmse,delta1,silog}`, wired into
   `on_validation_epoch_end`). `eval_step` no longer discards the depth
   prediction. The `silog` metric reuses `loss_depth_silog`, so it is
   directly comparable to the training `loss_depth` curve. Verified:
   imports clean in the `eomt` conda env; metric logic smoke-tested
   (perfect prediction → absrel/rmse/silog≈0, δ1≈1; letterbox-mismatch
   and no-depth-key paths both crash-free). Not exercised: a real
   end-to-end `validation_step` (needs the full model + data).

2. **Scheduler decoupling** (§4 Option A) — `backbone_freeze_steps` in
   `TwoStageWarmupPolySchedule` + plumbing + yaml. Required for the
   warmed unfreeze.

3. **Freeze callback** (§4 Option B) — recommended for
   compute/memory savings during Stage A. Reuse the Phase-0 freeze
   predicate.

4. **Gradient-matching loss** — add `loss_depth_grad` to
   `training/depth_loss.py` and wire a Stage-B-gated coefficient through
   `MaskClassificationLoss` (mirror how `depth_coefficient` threads).
   DA3 eq. 3: `‖∇ₓd̂ − ∇ₓd‖₁ + ‖∇_yd̂ − ∇_yd‖₁`.

5. **`accumulate_grad_batches`** in the yaml `trainer:` block.

6. **Recompute `attn_mask_annealing_*_steps` — ✅ DONE.** Both yamls
   updated for the 14,178-frame dataset, preserving the COCO fractional
   positions (block-0 → first ~4%, blocks 1–3 → [40-60%], [60-80%],
   [80-100%]):
   - `eomt_large_640.yaml` (bs=4, 354,400 total steps):
     `start=[0,141760,212640,283520]`, `end=[14385,212640,283520,354400]`.
   - `eomt_large_1280.yaml` (bs=1, 1,417,800 total steps):
     `start=[0,567120,850680,1134240]`, `end=[57548,850680,1134240,1417800]`.
   Both comments refreshed with the derivation.

7. **Intrinsics dropout** (§7) — small change in `training_step`'s
   intrinsics stacking (`lightning_module.py:~183`).

---

## 9. Documented alternative — DA3-faithful (no hard freeze)

If Stage A turns out to be unnecessary (heads stabilise in 1–2 epochs,
which is plausible given the Phase-0 smoke and the narrow domain), the
DA3-faithful recipe is simpler and is the fallback:

- **No freeze.** Train the backbone from step 0 at a low LR.
- Keep `warmup_steps` ≈ `[2000, 8000]` — the existing 2k-step backbone
  freeze + 8k ramp *is* DA3's "8k warmup".
- Backbone peak LR `~2e-5`, heads `2e-4` (the 10× gap).
- Gradient-matching loss on from the start (DA3 uses it throughout).

This collapses to "Stage B from epoch 0". It is *less* protective of the
shared panoptic features, which is the only reason it is the fallback
and not the default — the encoder feeding the grasp model is the asset
worth being conservative about. Decide between the two after watching
Stage A for ~2 epochs: if panoptic metrics are rock-stable with the
backbone frozen, the hard freeze is doing no harm and the conservative
plan stands; if class/occlusion can't stabilise against a frozen encoder,
switch to this.

---

## 10. Risks, footguns, rollback

- **Backbone unfreeze degrades panoptic.** Highest-priority risk
  (shared encoder). Mitigation: low backbone LR (~2e-5), warmed ramp,
  panoptic guardrail (§3 Stage B). Rollback: lower `depth_coefficient`,
  then lower backbone LR — do *not* re-freeze (the heads have already
  adapted to the frozen features; re-freezing mid-Stage-B strands them).
- **OOM — confirmed at 640²/bs4.** A first launch crashed with CUDA OOM
  on the first training step. Full-model training (fwd+bwd over the
  *unfrozen* backbone) needs ~25+ GiB at bs=4; on a shared 32 GiB card a
  live robotics stack (`zivid_camera` + `perception_node` + `grasp_node`)
  routinely holds ~10 GiB. The Phase-0 20.6 GiB figure is *not* a useful
  bound — it was a frozen-backbone run. Mitigations, in order:
  `batch_size: 2` (or 1); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
  gradient checkpointing on the ViT blocks (`depth_integration_plan.md`
  §11 risk 2); or free the GPU. A bs change invalidates the anneal-step
  counts (§8.6) — recompute.
- **`lr_mult` is dead code.** In `lightning_module.py:134`, the `elif
  (is_block or block_i == 0) and self.lr_mult != 1.0` branch is
  unreachable — its condition is identical to the `if` on line 131 that
  always catches first. So `lr_mult` currently has **no effect on block
  params**. To get the DA3-style 10× backbone/head LR gap, either fix
  this `elif` (make it apply `lr_mult` multiplicatively to *all*
  backbone groups) or use `llrd < 1` with `llrd_l2_enabled: True` (so
  the decay also reaches the last 4 masked-attn blocks, which the
  `lr = self.lr` override at line 137–145 otherwise pins to full LR).
  Recommend fixing `lr_mult` — it is the cleaner single knob for "whole
  backbone slower than heads". Flag, not yet tasked.
- **Depth scale convention.** `depth_knowledge.md` §11.6: if the SDG
  writer emits `distance_to_camera` rather than `distance_to_image_plane`,
  SI-Log converges to a non-zero floor. The data-verify chunk (memory
  chunk 2) concluded it *is* `distance_to_image_plane` — but if Stage A
  `loss_depth` plateaus suspiciously high, re-check this first.
- **Train/val gap widening in Stage B.** Means the ceiling is data
  diversity, not optimisation — adding epochs won't help; broaden the
  SDG scene/object/intrinsics randomisation instead. The 1,577-frame val
  set is the instrument for this.
- **Stale anneal schedule** — ✅ resolved (§8.6); both yamls recomputed
  for the 14,178-frame dataset.

---

## 11. Execution checklist

```
[x] §8.1  Depth val metrics (AbsRel, RMSE, δ1, SI-Log) in eval_step
[ ] §8.2  Decouple TwoStageWarmupPolySchedule (backbone_freeze_steps)
[ ] §8.3  Add backbone freeze/unfreeze callback
[ ] §8.4  Add gradient-matching loss (Stage-B gated)
[ ] §8.5  Add accumulate_grad_batches to yaml (optional at bs=4)
[x] §8.6  Recompute attn_mask_annealing_*_steps for 14,178 frames
[ ] §8.7  Add intrinsics dropout to training_step
[ ] §10   Decide: fix dead lr_mult elif vs use llrd+llrd_l2_enabled
[ ] --    Stage A: launch eomt_large_640.yaml, watch head losses
          + val depth metrics, epochs 0–10
[ ] --    Decision gate: heads plateaued? panoptic stable? → unfreeze
[ ]       (or fall back to §9 DA3-faithful if freeze proves pointless)
[ ] --    Stage B: unfreeze, watch panoptic guardrail + val depth
[ ] --    Early-stop on val AbsRel/δ1 plateau (~epoch 40–60)
```

If adding surface normals (§12), insert its checklist items into the
above before the Stage-B launch.

---

## 12. Adding surface normals as a learning objective

**Motivation.** Surface normals are a *local, scale-invariant* geometric
cue. Supervising them — or, more precisely, supervising a
**depth↔normal consistency** — regularises depth to be locally planar on
flat surfaces and sharp at edges, in a way that plain depth-gradient
matching (§5) cannot, because the normal is computed in 3D and accounts
for perspective. This is the Virtual-Normal / GeoNet family, and it is
exactly what DA3 used to refine its *Teacher* model's geometry (paper
§4.1, eqs. 4–6). Note DA3 used normals **only in the Teacher**, never in
the depth model itself — so this is a deliberate, justified departure,
motivated additionally by the downstream grasp model wanting a normal
map directly.

### 12.1 What's in the data

Verified on disk (`monocular_dataset/train/frame_0/normals.npy`):

- Shape **`(640, 640, 4)` float32**, range `[-1, 1]`.
- Channels 0–2: XYZ normal vector, **already unit-normalised**
  (‖xyz‖ = 1.000 at every pixel).
- Channel 3: constant `1.0` — Replicator's alpha/validity pad. Drop it
  (here it carries no information; if a future regen has invalid pixels
  it becomes the validity mask).
- Z-channel mean ≈ +0.80 ⇒ normals face the camera ⇒ **almost certainly
  camera/view-space**. ⚠ **Verify the coordinate convention before
  wiring the consistency loss** (§12.6) — same class of check as the
  depth-convention item in `depth_knowledge.md` §11.6. The
  depth→normal computation in Tier 1 must produce normals in the *same*
  space as the GT.
- The data is **640×640** (RGB, depth and normals all 640²). This is
  why the active config is `eomt_large_640.yaml` (see the header note) —
  resolved, no longer an open decision.

### 12.2 Tier 1 — virtual-normal consistency loss (no new parameters)

The minimal, highest-leverage version. It is a **loss term only** — no
architecture change — and is the mechanism that actually makes normals
improve *depth*.

**Computation** (new helpers in `training/depth_loss.py`):

1. `unproject(depth, K) -> points`: per-pixel camera-space point
   `P(u,v) = D(u,v) · K⁻¹·[u,v,1]ᵀ`. `K` is `target["intrinsics"]` —
   already loaded per frame and already transform-corrected (chunk 6),
   so no extra plumbing for the matrix itself. Depth is
   `distance_to_image_plane` (Z-depth) so the unprojection is the
   standard `[(u−cx)/fx·Z, (v−cy)/fy·Z, Z]`.
2. `depth_to_normals(points) -> n_d`: tangents by central finite
   difference `∂P/∂u ≈ P(u+1,v) − P(u−1,v)`, `∂P/∂v` likewise;
   `n_d = normalize(cross(∂P/∂u, ∂P/∂v))`.
3. `loss_normal_virtual(depth_pred, normals_gt, K)`:
   `L_vn = mean(1 − ⟨n_d, n_gt⟩)` over the SI-Log validity mask
   (`isfinite(d_gt) & d_gt>0 & d_gt<d_max`), `n_d` derived from
   **predicted** depth. Gradients flow `depth_pred → n_d → L_vn` — that
   is the regularisation path.

**Robustness footgun.** At depth discontinuities the finite differences
explode and the cross-product yields garbage normals. Two mitigations,
in order of preference:

- DA3's distance-weighted neighbour scheme (eqs. 4–5): sample 4
  neighbours per centre pixel, weight each neighbour-derived normal by
  consistency so far/discontinuous neighbours are down-weighted.
- Simpler fallback: additionally mask pixels whose local depth gradient
  exceeds a threshold (they are edges; the depth loss already supervises
  them).

**Wiring** — mirror `depth_coefficient` exactly:

- `training/depth_loss.py`: add `unproject`, `depth_to_normals`,
  `loss_normal_virtual`.
- `training/mask_classification_loss.py`: add `normal_coefficient` to
  `__init__` (next to `depth_coefficient`, line 39/50); in `forward`,
  accept `normals_gt` + `intrinsics`, and when present compute
  `out["loss_normal"] = loss_normal_virtual(depth_pred, normals_gt, K)`
  (right after the `loss_depth` block at line 99–100); add an
  `elif "normal" in loss_key` branch in `loss_total` (line ~212,
  *before* the mask branch — `"normal"` doesn't contain `"mask"` but
  order-match the existing `"depth"` comment rationale).
- `training/mask_classification_panoptic.py`: pass
  `normal_coefficient` through `__init__` → `MaskClassificationLoss(...)`
  (mirror line 41/82); ensure `training_step` already stacks
  `target["normals"]` and `target["intrinsics"]` and passes them to the
  criterion (intrinsics stacking already exists for the depth path —
  reuse it).
- `datasets/iscar_bp.py`: load `normals.npy`, slice `[:3]` channels,
  attach `target["normals"]` (see §12.4).
- `datasets/transforms.py`: normals transform as **vectors**, not just
  pixels (see §12.4 — this is a footgun).

`loss_normal_virtual` needs **no normal head and no change to the model
forward** — it reads `depth_pred` (already returned) and the GT. This is
why Tier 1 is the recommended first step: it is contained entirely in
the loss + data layers.

**Loss schedule.** Like `L_grad`, enable in **Stage B** (once SI-Log is
stable). `normal_coefficient` start ≈ `0.5` (the term is not
scale-invariant; keep it from dominating SI-Log). If the panoptic
guardrail trips, lower it before lowering LRs.

### 12.3 Tier 2 — explicit normal prediction head

Adds a normal *output* the grasp model can consume directly, and gives
the shared encoder a strong geometric gradient signal. Built on top of
Tier 1 (Tier 1's consistency loss can then optionally run against the
*predicted* normals too — GeoNet-style mutual consistency, §12.5).

**Head.** A parallel DPT reading the **same 4 taps** as `depth_head`:

```python
# models/eomt.py __init__, alongside self.depth_head
self.normal_head = DPT(
    dim_in=self.encoder.backbone.embed_dim,
    patch_size=patch_size[0],
    output_dim=3,
    activation="tanh",          # bounded pre-normalisation, dpt.py:314
    predict_conf=False,         # NEW dpt.py param — see below
    features=256,
    out_channels=(256, 512, 1024, 1024),
    pos_embed=False, down_ratio=1,
    head_name="normal", use_sky_head=False,
    norm_type="idt",            # taps already normed (chunk 4)
)
```

**`models/dpt.py` change — `predict_conf`.** `output_dim > 1` currently
auto-sets `has_conf = True` (`dpt.py:85`), which makes the head split
the last channel off as a confidence map (`dpt.py:256-259`) — wrong for
3 clean normal channels. Add a `predict_conf: bool = True` ctor param
and set `self.has_conf = (output_dim > 1) and predict_conf`. ~3 lines,
backwards-compatible (depth head and any multi-view DualDPT path keep
their current behaviour by default).

**Forward.** The tap capture in `EoMT.forward` is unchanged — the same
four normed patch-token taps already feed `depth_head`; just make a
second call:

```python
normal_logits = self.normal_head(feats, H=H, W=W,
                                 patch_start_idx=0, chunk_size=None)["normal"]
normal_pred = F.normalize(normal_logits, dim=1)   # [B, 3, H, W], unit
```

Return tuple grows **5 → 6**:
`(mask_per_layer, class_per_layer, occ_per_layer, depth, normal_pred, query_tokens)`.
Update every unpack site — same set the depth integration already
touched (`depth_integration_plan.md` §5 "Tuple change"): `inference.py`
(`__call__` + `EoMTResult` — add `normal: np.ndarray`),
`lightning_module.py` (`forward`, `training_step`),
`mask_classification_panoptic.py:eval_step` (currently unpacks the
5-tuple — extend to 6, score the normal).

**Direct supervision loss.** `loss_normal_angular(normal_pred,
normals_gt)` = `mean(arccos(clamp(⟨n̂, n_gt⟩, −1, 1)))` (or the cheaper
`1 − ⟨n̂, n_gt⟩`) over the valid mask. Add it under its own
`normal_coefficient` (or split into `normal_direct` /
`normal_consistency` keys if you want independent weighting). DA3's
teacher uses the angular form `E(n̂, n)` (eq. 6).

**Cost.** A full second DPT is ~the size of `depth_head` (~25–30M
params). Acceptable for a research iteration. If param budget bites
later, the optimisation is DA3's **Dual-DPT** (paper Fig. 3): share the
`projects` + `resize_layers` "reassembly", fork only `scratch` (the
RefineNet fusion + output convs) into depth and normal branches. Note as
a follow-up; do not start there.

### 12.4 Data plumbing (both tiers)

**`datasets/iscar_bp.py`** — mirror the `depth.npy` load (lines 87–93):

```python
normals_path = frame_dir / "normals.npy"
if normals_path.exists():
    n_arr = np.load(normals_path).astype(np.float32)[..., :3]   # drop alpha
    n_arr = np.transpose(n_arr, (2, 0, 1))                       # [3, H, W]
    target["normals"] = tv_tensors.Mask(torch.from_numpy(n_arr))
```

**`datasets/transforms.py`** — ⚠ **footgun: normals are direction
vectors, not a passive raster.** `tv_tensors.Mask` carries the *pixels*
through resize/crop/pad/flip, but does **not** transform the *vector
values*. Under a horizontal flip the X component must be **negated**;
under a rotation the vector must be rotated. The pipeline currently does
only hflip (`scale_range:[1.0,1.0]`, no rotation aug), so the minimum is:
on hflip, negate channel 0 of `target["normals"]`. This is the same
special-case pattern established for `intrinsics` in chunk 6 — add a
`"normals"` branch to the transform forward exactly where the flip is
applied, and `_filter` must skip it (single-tensor, not per-instance,
like `depth`). If rotation aug is ever enabled, the rotation matrix must
also be applied to the (x,y) components — leave a comment to that
effect.

### 12.5 Optional — mutual depth↔normal consistency (GeoNet-style)

With Tier 2 in place, the strongest "normals improve depth" coupling is
to also penalise the **predicted-depth-derived** normal `n_d` (§12.2)
against the **predicted** normal `n̂`:
`L_mc = mean(1 − ⟨n_d, n̂⟩)`. This makes the two heads mutually
regularising — the depth head is pulled toward producing geometry whose
implied normals match the (GT-supervised) normal head, and vice-versa.
Add as a third small term; keep its coefficient low (~0.25) and enable
last, after both Tier-1 and Tier-2 losses are stable.

### 12.6 Prerequisites / checklist additions

```
[ ] §12.1  Verify normals coordinate space (camera vs world) — load a
           frame, cross-check against intrinsics-unprojected depth
[ ] §12.4  iscar_bp.py: load normals.npy[:3] → target["normals"]
[ ] §12.4  transforms.py: hflip negates normals channel 0; _filter skips
[ ] §12.2  depth_loss.py: unproject + depth_to_normals + loss_normal_virtual
[ ] §12.2  mask_classification_loss.py: normal_coefficient + loss_total branch
[ ] §12.2  mask_classification_panoptic.py: thread normal_coefficient,
           pass normals+intrinsics to criterion
[ ] --     Tier 1 done — enable L_vn in Stage B, coeff ~0.5
[ ] §12.3  dpt.py: predict_conf param
[ ] §12.3  eomt.py: normal_head + 2nd DPT call + 5→6 tuple
[ ] §12.3  update all 6-tuple unpack sites (inference, lightning, panoptic)
[ ] §12.3  loss_normal_angular direct supervision
[ ] §12.3  normal val metric (mean angular error) in eval_step
[ ] §12.5  (optional) mutual consistency L_mc, coeff ~0.25, enable last
```

Tier 1 is independently shippable and is the recommended first
increment. Tier 2 and §12.5 are layered on once Tier 1 shows the depth
metrics (AbsRel/δ1) actually improve.

---

## 13. Numerics & fp16 stability (lessons from the first curriculum run)

The first attempts at running the full curriculum surfaced an
**fp16-numerics class of NaN** in the training loss that wasn't visible
in the SI-Log-only baseline. Documenting it here so future runs don't
re-learn the lesson.

### 13.1 Symptom

Under Lightning's `precision: 16-mixed`, the *total* training loss
(`losses/train_loss_total`) intermittently displayed `nan` on the tqdm
progress bar.

- **Run `gvp8`** (curriculum, no fp32 wraps): ~3 % of batches NaN, 48
  events between step 110 and step 1627 — onset early, before
  `aux_ramp` was contributing meaningfully in magnitude (a single NaN
  *element* contaminates the reduction regardless of coefficient).
- **Run `pn3d`** (curriculum, aux losses wrapped in fp32 but depth not
  source-clamped): rate dropped to ~1.5 % then *climbed* to ~2.5 % by
  step 322 — a partial fix.

The run kept making progress in both cases (Lightning's `GradScaler`
skips the optimiser step on each non-finite gradient), but burned ~2-3 %
of batches and the trajectory was getting worse, not better.

### 13.2 Root causes

Two distinct fp16 hazards, each capable of producing a NaN element that
then propagates:

1. **`exp` activation in the DPT depth head overflows to `inf` under
   fp16.** `models/dpt.py` applies `exp(main_logits)` for `head_name="depth"`;
   under `autocast(float16)` any logit above ~11 produces `+inf`. A
   single `inf` pixel in `depth_pred` poisons every downstream term:
   - SI-Log: `log(inf) = inf` → `mean(R²) = inf`, `mean(R)² = inf`,
     `inf − λ·inf = NaN` even before the sqrt.
   - `loss_depth_grad`: same `log` chain.
   - `loss_depth_normal_consistency`: `unproject(inf, K) = inf` points →
     `cross(inf, inf) = NaN` in the implied-normal computation.
2. **Cross-products of fp16 tangent vectors at near-equal-depth
   neighbours.** In `depth_to_normals`, when two adjacent depth pixels
   round to the *same* fp16 value (the entire bin floor on a smooth
   prediction is a large patch of this), the tangent vectors are
   exactly zero, `cross(0, 0) = 0`, and the autograd path through
   `F.normalize(0, eps)` has an undefined derivative — backward yields
   NaN gradients for that pixel.

### 13.3 The three layers of clamping (all needed)

Source-side clamps prevent the NaNs from ever existing; sink-side
`nan_to_num` floors catch anything that slips through.

**(a) `models/eomt.py:forward` — clamp the depth output and sanitize
the normal output.**

```python
# Right after the depth head call + .unsqueeze(1):
depth = depth.clamp(min=1e-4, max=20.0)

# Right after the normal head call + F.normalize:
normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
```

Real bin depths are 0.6–0.9 m and the SI-Log validity mask already
discards `d_gt ≥ 10 m`, so `max=20` only chops infinities; `min=1e-4`
symmetrically guards underflow.

**(b) `training/depth_loss.py` — `_aux_loss_fp32` decorator on the four
aux losses.** Forces fp32 throughout the loss body regardless of the
caller's autocast context. Floating-point tensor args are promoted to
`.float()`; bool / int tensors (the `valid` masks) pass through.
Applied to: `loss_depth_grad`, `loss_normal_angular`, `loss_normal_grad`,
`loss_depth_normal_consistency`. SI-Log is intentionally **not**
wrapped — it was stable in the baseline run and the wrapper isn't free.

**(c) `training/depth_loss.py` — `torch.nan_to_num` on every loss's
return value.** Belt-and-suspenders: if anything slips through (a, b),
the loss term floors to 0 (effectively skipping that batch's gradient
for the affected term — the same behavior `GradScaler` would impose
later but applied per-term, not whole-batch).

### 13.4 Verification

- **Unit-test the loss path with adversarial inputs** —
  `training/depth_loss.py` should produce finite output when fed
  tensors containing `inf` and `nan` elements directly. The smoke
  pattern (also useful for future regressions):
  - constant-depth fp16 batch (exercises the cross-product underflow)
  - tensors with sprinkled `inf` / `nan` elements (exercises the floors)
  - both should return finite, sensible values; the aux losses should
    return in fp32 dtype regardless of caller context.
- **Monitor `train_loss_total=nan` count** during the first ~500 steps
  of any curriculum run. A non-zero count there means one of (a)–(c)
  isn't catching the path that's NaN'ing.

### 13.5 What this changes upstream

§11 checklist — add to the prerequisites:

```
[x] §13.3a  eomt.py: clamp depth ∈ [1e-4, 20], nan_to_num the normal
[x] §13.3b  depth_loss.py: _aux_loss_fp32 on the 4 aux losses
[x] §13.3c  depth_loss.py: nan_to_num on every loss return
```

**Verification run `nyyo`** (with all three layers of clamping landed):
trained to **step 535** of epoch 0 → **`nan_count = 0`** across the
full window. No checkpoint was saved (stopped before the first epoch
boundary; Lightning's default `ModelCheckpoint` only fires at
`on_train_epoch_end`, and there's no `every_n_train_steps` schedule
in the yaml). The next real curriculum run should either reach
epoch 0's end (~22 min at bs=4 / 2.65 it/s) for the default ckpt to
land, or add a step-based ModelCheckpoint to `eomt_large_640.yaml`
under `trainer.callbacks` if mid-epoch resumability is wanted.

Run `pn3d` was the partial-fix attempt (aux losses fp32 but no source
clamp on `depth`); stopped at step 417 (~12 % epoch 0), NaN rate
climbing (2.5 % and rising). Run `gvp8` was the unpatched original
(~3 % NaN). Neither is a useful checkpoint for downstream comparison —
those checkpoint dirs can be safely deleted.

---

End of plan.
