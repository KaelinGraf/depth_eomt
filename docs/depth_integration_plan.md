# Depth Integration Plan: DA3-mono Depth Head into EoMT (1280×1280)

This document is the implementation plan for two coupled changes to
`/home/kaelin/BinPicking/eomt`:

1. raise the project resolution from **640×640 → 1280×1280** to match the new
   dataset, and
2. graft a **monocular depth head** that follows Depth Anything 3's recipe
   (`docs/da3_monocular.md`) onto the existing DINOv3 ViT-L/16 backbone, in
   parallel with the existing mask / class / occlusion heads
   (`docs/eomt_architecture.md`).

All paths in this doc are relative to `/home/kaelin/BinPicking/eomt/`
unless noted. Line numbers reference the current `agent_test` branch.

---

## 1. Objective and constraints

We add a **DPT depth head** that taps four intermediate layers of the
*single shared* DINOv3 ViT-L backbone (taps `[4, 11, 17, 23]`). The shared
encoder is a hard requirement: the patch tokens going to the downstream
flow-matching grasp model must encode geometry *and* segmentation, so two
encoders are not allowed.

High-level data flow:

```
RGB image [B, 3, 1280, 1280] (in [0,1])
          │
          ▼  EoMT.forward applies (x − pixel_mean)/pixel_std        eomt.py:163
          │
          ▼  patch_embed → 80×80 patch tokens, [B, 5+6400, 1024]    eomt.py:169-172
          │
          ▼  ViT blocks 0..19 (no queries yet)
          │      ├─ tap @ block 4 out  ─┐
          │      ├─ tap @ block 11 out ─┤  patch tokens, [B, 6400, 1024]
          │      └─ tap @ block 17 out ─┤  (no queries present)
          │                             │
          ▼  block 20: prepend 200 q    │
          │  blocks 20..23 run with     │
          │  masked-attn loop           │
          │      ├─ tap @ block 23 out ─┘  needs slice off queries
          │      │
          │      └─► _predict() → (mask_logits, class_logits,
          │                        occlusion_logits, q)
          ▼
         DPT(taps[4,11,17,23], H=1280, W=1280) → depth [B, 1, 1280, 1280]
```

Constraints carried in:

- Single backbone (DINOv3 ViT-L/16, 24 blocks, embed_dim 1024).
- Patch size **16** (not DA3's 14): override `patch_size=16` in DPT.
- 4 taps `[4, 11, 17, 23]`, matching DA3 mono recipe.
- `output_dim=1`, `use_sky_head=False`, `activation="exp"`.
- No intrinsics input, no normals head.
- ImageNet mean/std normalisation already happens *inside* `EoMT.forward`
  (`eomt.py:163`) so the depth head — which runs after the backbone —
  automatically sees the same normalised features the backbone uses.
- Loss: SI-Log on log-depth, primary; gradient-matching optional later.
- Depth head supervised **only at the final layer** (justification in §6).

---

## 2. Architecture diagram (1280×1280)

```
input [B, 3, 1280, 1280]   (float, [0, 1])
   │
   │  ImageNet normalise (eomt.py:163)
   ▼
patch_embed (16×16, stride 16)
   │   → [B, 5 + 6400, 1024]    (5 = 1 CLS + 4 register)
   │   → +pos_embed             (interpolated to 80×80 grid; vit.py:60-63)
   ▼
┌─ ViT block 0
│
├─ block 4   ───────────────►  feat4_full   [B, 5+6400, 1024]   (TAP)
├─ block 11  ───────────────►  feat11_full  [B, 5+6400, 1024]   (TAP)
├─ block 17  ───────────────►  feat17_full  [B, 5+6400, 1024]   (TAP)
│
├─ block 20: prepend q [200, 1024]; sequence is now [B, 200+5+6400, 1024]
├─ block 20  ─►  _predict (deep-sup mask/class/occ, layer 0)
├─ block 21  ─►  _predict (deep-sup, layer 1)
├─ block 22  ─►  _predict (deep-sup, layer 2)
├─ block 23  ─►  _predict (deep-sup, layer 3)            and TAP feat23_full
│              feat23_full = [B, 200+5+6400, 1024]   (queries present!)
│
└─ post-loop _predict (layer 4 = final)
        ├─ mask_logits   [B, 200, 320, 320]   (160 → 320 because img_size doubled)
        ├─ class_logits  [B, 200, 3]
        ├─ occlusion     [B, 200]
        └─ query_tokens  [B, 200, 1024]

Depth path (NEW):
   feat4_full   ──► slice patch [B, 6400, 1024]                   ┐
   feat11_full  ──► slice patch [B, 6400, 1024]                   │
   feat17_full  ──► slice patch [B, 6400, 1024]                   │  4 stages
   feat23_full  ──► slice off 200 q + 5 prefix → [B, 6400, 1024]  │
                                                                  ▼
                  DPT(patch_size=16, dim_in=1024,
                      out_channels=[256,512,1024,1024], features=256)
                       │
                       │ per-stage projects + resize_layers
                       │   stage 0: 80×80 → 320×320×256    (×4)
                       │   stage 1: 80×80 → 160×160×512    (×2)
                       │   stage 2: 80×80 → 80×80×1024     (×1)
                       │   stage 3: 80×80 → 40×40×1024     (/2)
                       │
                       │ scratch + RefineNet pyramid (top-down)
                       │   refine4: 40×40 → 80×80
                       │   refine3:        → 160×160
                       │   refine2:        → 320×320
                       │   refine1:        → 640×640 (4× patch_grid, 256 ch)
                       │
                       │ output_conv1  256→128, 3×3
                       │ bilinear upsample 640×640 → 1280×1280
                       │ output_conv2  128→32 (3×3) → ReLU → 32→1 (1×1)
                       │ exp(·)
                       ▼
                  depth [B, 1, 1280, 1280]
```

Token-layout reminder for tap layer 23 (post-query-prepend, `eomt.py:179-181`):

```
[ 200 queries | 5 prefix (CLS+4 reg) | 6400 patches ]
```

So slicing for the depth tap at block 23 is
`x[:, num_q + num_prefix_tokens :, :]` exactly as `_predict` already does
at `eomt.py:70`.

---

## 3. Resolution upgrade 640 → 1280

Every place 640 (or a 40-derived constant) appears, plus what changes.

| File:line | Current | New | Why |
|-----------|---------|-----|-----|
| `inference.py:76` | `img_size: tuple = (640, 640)` (default ctor arg) | `(1280, 1280)` | inference default |
| `inference.py:319` | comment "640x640" | "1280x1280" | doc only |
| `datasets/iscar_bp.py:143` | `img_size: tuple[int, int] = (640, 640)` | `(1280, 1280)` | `ReplicatorDataModule` ctor default |
| `configs/dinov3/occlusion_bp/panoptic/eomt_large_640.yaml` | filename + content target 640 | rename to `eomt_large_1280.yaml`, add `img_size: [1280, 1280]` under both `model.init_args` and `data.init_args` (mirroring `configs/dinov2/ade20k/panoptic/eomt_large_1280.yaml:24`) | training config |
| (NEW) `configs/.../eomt_large_1280.yaml` | n/a | `data.init_args.img_size: [1280, 1280]`; `model.init_args.img_size: [1280, 1280]` (linked); also bump `attn_mask_annealing_*_steps` proportionally to whatever the new dataset epoch count gives (`unknown — verify at implementation time`) | the `main.py:127-134` link wires `data.img_size` into both `model.img_size`, `model.network.img_size`, and `model.network.encoder.img_size`. So a single yaml entry suffices once present. |
| `models/eomt.py:57-63` | derives `num_upscale = log2(16) − 2 = 2`. Patch size unchanged. | unchanged | log2(16)−2 = 2 → 2 ScaleBlocks. Mask logits go from 40→80→160 to 80→160→**320**, doubling alongside image. |
| `models/vit.py:60-63` | `grid_size = (img_size[0]//patch_size, img_size[1]//patch_size)` | unchanged code; runtime shape becomes 80×80 | grid is computed from `img_size`, so this just propagates. |
| ViT pos-embed interp | `_pos_embed` is the HF `Dinov3Embeddings` method called at `eomt.py:171-172`; it interpolates the position embedding to the new `grid_size` at runtime | unchanged code | `unknown — verify at implementation time` whether HF's DINOv3 pos-embed interp degrades quality at 80×80 vs 40×40; if there's a `position_embeddings` cache it may need invalidation. Test: load the existing `panoptic_1280.bin` checkpoint (`eomt/checkpoints/panoptic_1280.bin` exists) and verify it forwards cleanly at 1280. |
| `inference.py:144-177` `_preprocess` | letterbox to 640×640 | letterbox to 1280×1280 (no code change; uses `self.img_size`) | already parameterised. |
| `inference.py:319-330` postprocess | mask logits 160×160 → 640×640 → crop → orig | now 320×320 → 1280×1280 → crop → orig | already parameterised by `self.img_size`. |
| `datasets/transforms.py:42-43` | `T.ScaleJitter(target_size=img_size,…)` and `T.RandomCrop(img_size)` | unchanged code; runtime size doubles | aug already parameterised; with `scale_range=(0.1, 2.0)` (yaml) jitter from 128 to 2560, then crop to 1280 — verify `scale_range` is still appropriate for 1280 inputs (`unknown — verify at implementation time`; consider tightening to `(0.5, 1.5)` to avoid extreme aspect distortion on larger crops). |
| `training/lightning_module.py:706-754` | `resize_and_pad_imgs_instance_panoptic` and `revert_resize_and_pad_logits_instance_panoptic` use `self.img_size` | unchanged | already parameterised. |
| `data.init_args.batch_size` in BP yaml | `4` at 640 (`yaml:39`) | likely needs to drop to 1 or 2 at 1280 due to ~4× memory. Concrete value `unknown — verify at implementation time`. | sequence length goes from 1805 → 6605 (≈3.66× tokens), attention is O(N²) so memory ≈13×; expect to need either grad-checkpointing or batch_size=1. |

There is no other hard-coded `640` in the runtime code path; the
hard-coded literals only appear in yaml file *names* and a few comments.

---

## 4. Depth head module

### 4.1 Where DPT lives

**Recommendation: copy `Depth-Anything-3/src/depth_anything_3/model/dpt.py`
verbatim into `eomt/models/dpt.py`** (option A).

Justification:
- The DA3 repo is not pip-installable as currently structured under
  `/home/kaelin/BinPicking/Depth-Anything-3/` (no `pyproject.toml` /
  `setup.py` discovered; `unknown — verify at implementation time`).
- Copying isolates the EoMT package from upstream churn and lets us
  change `patch_size` from 14 to 16 cleanly.
- DPT is ~460 LOC and self-contained; the only external imports are
  `head_utils.{Permute, custom_interpolate, create_uv_grid,
  position_grid_to_embed}`. We need only `custom_interpolate`; copy the
  three helpers used (or vendor `head_utils` too).

Files to add:
- `eomt/models/dpt.py` — copied from `Depth-Anything-3/src/depth_anything_3/model/dpt.py`.
- `eomt/models/dpt_utils.py` — copied subset of
  `Depth-Anything-3/src/depth_anything_3/model/utils/head_utils.py`
  containing only `Permute`, `custom_interpolate`, `create_uv_grid`,
  `position_grid_to_embed` (we only need `custom_interpolate` if
  `pos_embed=False`, but copy all four for robustness).
- Adjust the import line at the top of the new `eomt/models/dpt.py` to
  `from models.dpt_utils import Permute, custom_interpolate, create_uv_grid, position_grid_to_embed`.

### 4.2 EoMT wrapper instantiation

In `models/eomt.py`, add to `__init__` (after the existing heads, after
line 63):

```python
from models.dpt import DPT
self.depth_taps: tuple[int, int, int, int] = (4, 11, 17, 23)
self.depth_head = DPT(
    dim_in=self.encoder.backbone.embed_dim,    # 1024
    patch_size=patch_size[0],                  # 16, from line 57-58
    output_dim=1,
    activation="exp",
    features=256,
    out_channels=(256, 512, 1024, 1024),
    pos_embed=False,
    down_ratio=1,
    head_name="depth",
    use_sky_head=False,
    norm_type="layer",                         # apply LN to taps for stability
    fusion_block_inplace=False,
)
```

`norm_type="layer"` deviates from the DA3 default ("idt"); reasoning: DA3
relies on its pretrained checkpoint to make raw taps usable, but we are
training DPT from scratch on EoMT's already-`norm`-ed patch tokens (see
§9 for caveat about *which* norm) and adding a per-tap `LayerNorm(1024)`
costs 4×2048 params and stabilises early training.

### 4.3 Tap-layer hook in `EoMT.forward`

Insert tap captures inside the existing block loop in `eomt.py:177-209`.
Concretely:

- Initialise `depth_taps_features = []` next to the per-layer lists at
  `eomt.py:175`.
- After the `block(...)` finishes (i.e. after `eomt.py:209`'s mlp
  add/skip), check `if i in self.depth_taps:` and append a *patch-only*
  tensor:
  - if queries have **not** been prepended yet (i.e. `i <
    len(blocks) - num_blocks`, which is true for taps 4, 11, 17 with
    `num_blocks=4`):
    `depth_taps_features.append(x[:, self.encoder.backbone.num_prefix_tokens:, :])`
    → `[B, 6400, 1024]`
  - if queries **have** been prepended (true for tap 23, since 23 ≥ 20):
    `depth_taps_features.append(x[:, self.num_q + self.encoder.backbone.num_prefix_tokens:, :])`
    → `[B, 6400, 1024]`

  (this is exactly the same slice as `_predict` does at `eomt.py:70`.)

This keeps the captured tensor strictly the patch-token portion at
1024 channels in the same `[B, N_patch, C]` ordering DPT expects (DPT's
`_forward_impl` does its own permute+reshape to `[B, C, ph, pw]` at
`dpt.py:221`).

`patch_start_idx=0` is passed to `DPT.forward` because we already
sliced the prefix off when we captured.

### 4.4 Final DPT call

After the post-loop `_predict` (`eomt.py:211`), add:

```python
# Wrap each [B, N, C] tap as [B, S=1, N, C] to match DPT.forward's expectations
H, W = self.encoder.backbone.patch_embed.grid_size[0] * patch_size[0], \
       self.encoder.backbone.patch_embed.grid_size[1] * patch_size[1]
feats = [t.unsqueeze(1) for t in depth_taps_features]   # each [B, 1, 6400, 1024]
depth_out = self.depth_head(feats, H=H, W=W, patch_start_idx=0, chunk_size=None)
depth = depth_out["depth"].squeeze(1)   # [B, 1280, 1280] (S=1 squeezed; depth is 1-ch)
```

Note `DPT._forward_impl` returns `[B, H, W]` (not `[B,1,H,W]`) for `output_dim=1`
because of the `.squeeze(1)` at `dpt.py:256`. We expose it as `[B, 1, H, W]`
by adding a `.unsqueeze(1)` in our wrapper before returning, so callers
have a uniform CHW layout.

---

## 5. Forward pass changes

Pseudocode for the new `EoMT.forward` (replacing `eomt.py:162-222`):

```python
def forward(self, x):
    x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std    # eomt.py:163

    rope = None
    if hasattr(self.encoder.backbone, "rope_embeddings"):
        rope = self.encoder.backbone.rope_embeddings(x)

    x = self.encoder.backbone.patch_embed(x)
    if hasattr(self.encoder.backbone, "_pos_embed"):
        x = self.encoder.backbone._pos_embed(x)

    attn_mask = None
    mask_logits_per_layer, class_logits_per_layer, occlusion_logits_per_layer = [], [], []
    depth_taps = []                       # NEW

    n_blocks = len(self.encoder.backbone.blocks)
    n_prefix = self.encoder.backbone.num_prefix_tokens

    for i, block in enumerate(self.encoder.backbone.blocks):
        # 1. Prepend queries before block (n_blocks - num_blocks)
        if i == n_blocks - self.num_blocks:
            x = torch.cat(
                (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
            )

        # 2. Deep-supervision pre-block predict (existing)
        if (
            self.masked_attn_enabled
            and i >= n_blocks - self.num_blocks
        ):
            mask_logits, class_logits, occlusion_logits, _ = self._predict(
                self.encoder.backbone.norm(x)
            )
            mask_logits_per_layer.append(mask_logits)
            class_logits_per_layer.append(class_logits)
            if self.enable_occlusion:
                occlusion_logits_per_layer.append(occlusion_logits)
            attn_mask = self._attn_mask(x, mask_logits, i)

        # 3. Run the block (unchanged from eomt.py:195-209)
        ...

        # 4. NEW depth tap (after block executes)
        if i in self.depth_taps:
            queries_present = i >= n_blocks - self.num_blocks
            start = (self.num_q + n_prefix) if queries_present else n_prefix
            depth_taps.append(x[:, start:, :].contiguous())

    # 5. Final-block predict (existing)
    mask_logits, class_logits, occlusion_logits, query_tokens = self._predict(
        self.encoder.backbone.norm(x)
    )
    mask_logits_per_layer.append(mask_logits)
    class_logits_per_layer.append(class_logits)
    if self.enable_occlusion:
        occlusion_logits_per_layer.append(occlusion_logits)

    # 6. NEW: single DPT call
    H = self.encoder.backbone.patch_embed.grid_size[0] * \
        self.encoder.backbone.patch_embed.patch_size[0]
    W = self.encoder.backbone.patch_embed.grid_size[1] * \
        self.encoder.backbone.patch_embed.patch_size[1]
    feats = [t.unsqueeze(1) for t in depth_taps]    # [B, 1, N, C] each
    depth = self.depth_head(feats, H=H, W=W,
                            patch_start_idx=0,
                            chunk_size=None)["depth"]
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)                  # [B, 1, H, W]

    return (
        mask_logits_per_layer,
        class_logits_per_layer,
        occlusion_logits_per_layer if self.enable_occlusion else None,
        depth,
        query_tokens,
    )
```

**Tuple change**: from 4-tuple to **5-tuple**
`(mask_logits_per_layer, class_logits_per_layer, occ_per_layer, depth, query_tokens)`.
Every caller of `EoMT.forward` and `LightningModule.forward` must be
updated:
- `inference.py:310-312`
- `training/lightning_module.py:178` (forward returns network output verbatim)
- `training/lightning_module.py:183` (training_step unpack)
- `training/mask_classification_panoptic.py:105` (eval_step unpack)

**Tap features at layer 17 vs 23 — important caveat**: the doc
`docs/eomt_architecture.md:215-227` says deep supervision uses
`self.encoder.backbone.norm(x)` to feed `_predict`, but the depth taps
above feed **un-normed** intermediate features into DPT. With
`norm_type="layer"` on the DPT input (§4.2) this is fine and is closer to
what DA3 actually does — DA3 only norms the *final-layer* output via
`encoder.norm`, so for taps 4/11/17 DA3 is also using un-normed features
(see `vision_transformer.py:372-394` in DA3, which only norms the last
layer). For tap 23 specifically, DPT in DA3 does receive the normed
output (because it's the post-final-norm output). To stay faithful to
DA3, the cleanest solution is:
- `depth_taps[3]` (the layer-23 tap) should use
  `self.encoder.backbone.norm(x)` **before** the patch-only slice, so the
  layer-23 tokens are normed.
- `depth_taps[0..2]` (layers 4, 11, 17) stay un-normed.

This matches DA3's `_get_intermediate_layers_not_chunked` behaviour and
is the correct interpretation.

---

## 6. Loss integration

### 6.1 New loss function

Add a new method to `MaskClassificationLoss`
(`training/mask_classification_loss.py`) and a free helper:

```python
# eomt/training/depth_loss.py  (new file)
import torch
import torch.nn.functional as F

def loss_depth_silog(
    d_pred: torch.Tensor,        # [B, 1, H, W], strictly positive (post-exp)
    d_gt:   torch.Tensor,        # [B, 1, H, W], metres, may contain NaN/inf
    *,
    lambda_var: float = 0.85,
    d_max: float = 10.0,         # bin-picking scenes < 2m, but allow margin
    eps: float = 1e-6,
) -> torch.Tensor:
    """SI-Log on log-depth. NaN-safe; ignores invalid pixels."""
    valid = torch.isfinite(d_gt) & (d_gt > 0) & (d_gt < d_max)
    if not valid.any():
        return d_pred.sum() * 0.0  # graph-preserving zero
    log_diff = torch.log(d_pred.clamp_min(eps)) - torch.log(d_gt.clamp_min(eps))
    log_diff = log_diff[valid]
    var_term = (log_diff ** 2).mean()
    bias_term = log_diff.mean() ** 2
    return torch.sqrt((var_term - lambda_var * bias_term).clamp_min(eps))
```

Unit test (placed in `tests/test_depth_loss.py` or smoke-tested in a
notebook):

```python
d_pred = torch.exp(torch.randn(2, 1, 32, 32))
d_gt   = torch.exp(torch.randn(2, 1, 32, 32))
assert loss_depth_silog(d_pred, d_gt).item() > 0
assert loss_depth_silog(d_pred, d_pred).item() < 1e-3   # zero on identity
```

### 6.2 Wiring into the criterion

In `MaskClassificationLoss.__init__`
(`training/mask_classification_loss.py:25-59`):
- Add `depth_coefficient: float = 1.0` and store `self.depth_coefficient`.

In `MaskClassificationLoss.forward`
(`training/mask_classification_loss.py:62-90`):
- Take a new optional kwarg `depth_pred: Optional[torch.Tensor] = None`
  and `depth_gt: Optional[torch.Tensor] = None`.
- If both present, return `{**existing, "loss_depth": loss_depth_silog(depth_pred, depth_gt)}`.

In `MaskClassificationLoss.loss_total`
(`training/mask_classification_loss.py:196-219`), add a new substring
branch *before* the `raise ValueError`:

```python
elif "depth" in loss_key:
    weighted_loss = loss * self.depth_coefficient
```

In `training/mask_classification_panoptic.py:73-84`, add
`depth_coefficient=depth_coefficient` to the `MaskClassificationLoss(...)`
call, and add `depth_coefficient: float = 1.0` to
`MaskClassificationPanoptic.__init__` signature.

In `training/lightning_module.py:180-199` (`training_step`), update the
unpack:

```python
mask_logits_per_block, class_logits_per_block, occlusion_logits_per_block, depth_pred, _ = self(imgs)
depth_gt = self._collate_depth_gt(targets).to(depth_pred.device, depth_pred.dtype)
```

and pass `depth_pred=depth_pred, depth_gt=depth_gt` only on the **last
layer** loss call (see §6.3).

### 6.3 Deep supervision: NO for depth

**Recommendation**: do *not* deep-supervise depth.

Justification:
- Mask deep supervision exists because the queries-vs-targets matching is
  a global combinatorial problem; iterating gives the matcher a
  sharper signal at later layers and lets the masked-attention loop
  steer query attention. Depth is a per-pixel regression with no
  matcher and its loss surface is convex in the prediction; iterating
  it would just multiply compute and slow the per-step time without
  improving accuracy.
- DPT itself reads from 4 layers of the encoder, so it already uses
  intermediate-layer information via the pyramid; running it 5× would
  not add information beyond what the pyramid encodes.
- DA3 runs DPT once after the final block; matching their recipe
  preserves the option of loading DA3 pretrained DPT weights.

So the loss key `"loss_depth"` is added once (final layer), without
`_block_<i>` suffix, and `loss_total` weights it with
`self.depth_coefficient = 1.0`.

### 6.4 Validity mask

`mask = isfinite(d_gt) & (d_gt > 0) & (d_gt < d_max)`. Replicator's
`distance_to_image_plane` returns `inf` for sky / non-hits; SDG's
`depth.npy` is float32 and stores `0.0` (or `NaN`) for invalid — verify
which at implementation time and adjust the predicate. Set
`d_max = 10.0` (metres) for bin-picking; bin scenes are normally
< 2 m so this is a generous outlier rejection.

---

## 7. Dataset adapter

The current dataloader is `ReplicatorDataset` in
`datasets/iscar_bp.py:25-133`. It returns `(img, target)` where target
holds `masks`, `labels`, `occlusion`, `is_crowd`. No depth field.

### 7.1 Source data

SDG output (per `claudeMd` user role + their existing pipeline) is laid
out as:
```
/home/kaelin/BinPicking/SDG/IS/Outputs/monocular_dataset/frame_*/
    rgb.png
    depth.npy        (NEW; float32, metres, NaN for invalid)
    instance_id_segmentation_*.png
    Replicator_scene_info.json
```

`unknown — verify at implementation time` whether the depth file is
named `depth.npy` exactly or e.g. `distance_to_image_plane_*.npy` —
adjust the glob in §7.2 step 1. In any case the dataset class needs to
load it.

### 7.2 Changes to `ReplicatorDataset.__getitem__` (`iscar_bp.py:42-133`)

1. After step 1 (RGB load, ~line 48), find and load
   `depth.npy`:
   ```python
   depth_path = frame_dir / "depth.npy"
   depth = np.load(depth_path).astype(np.float32)   # [H, W]
   depth_t = tv_tensors.Mask(torch.from_numpy(depth)[None, ...])
   # using tv_tensors.Mask so the spatial transforms in transforms.py
   # apply the same crop/flip as masks/img
   ```
   (`tv_tensors.Mask` is the right wrapper here even for floats; it tags
   the tensor as "follow geometric ops" without claiming int dtype. If
   that breaks because torchvision insists on integer dtype for `Mask`,
   fall back to a custom tv_tensor or carry depth as a separate channel
   appended to `target`.)

2. Add to the `target` dict (around `iscar_bp.py:119`):
   ```python
   target["depth"] = depth_t
   ```

3. Update `transforms.py:Transforms.forward`
   (`datasets/transforms.py:123-147`) so depth survives all the geometric
   transforms. Currently `_filter`, `pad`, `random_horizontal_flip`,
   `scale_jitter`, `random_crop` all use `target` keys and most of them
   pass through `tv_tensors.Mask` correctly. The two that don't are:
   - `_filter` at `transforms.py:120-121` filters by **per-instance**
     `keep` mask. Depth is single-channel, not per-instance. Special-case
     it so it isn't filtered; explicit code:
     ```python
     def _filter(self, target, keep):
         filtered = {}
         for k, v in target.items():
             if k == "depth":
                 filtered[k] = v
             else:
                 filtered[k] = wrap(v[keep], like=v)
         return filtered
     ```
   - `pad` (line 108-118) needs to pad depth too — append `target["depth"]
     = F.pad(target["depth"], padding, fill=float('nan'))` so padded
     regions are masked out by the SI-Log validity check.

4. Aug interaction: scale_jitter rescales depth values when it shrinks the
   image — but **monocular metric depth must rescale inversely with the
   image rescale factor**? Actually no, for monocular depth supervision
   under crops the *physical* depth at each retained pixel is unchanged
   (we're not changing focal length). The spatial location moves, the
   metric value at a pixel stays the same. So no value-side rescale is
   needed. *Caveat*: this is correct for crop+flip+pad; **incorrect** for
   `ScaleJitter` if you interpret it as "zoom into the scene" (which
   would change apparent depth). DA2 / DA3 sidestep this by not applying
   scale jitter during training — but DA3 mono uses synthetic data with
   fixed intrinsics. **Recommendation**: disable `scale_jitter` for
   depth-aware training, or set `scale_range=(1.0, 1.0)` in the BP yaml
   to avoid this ambiguity. Document this in the yaml.

5. `train_collate` (`datasets/lightning_data_module.py:41-48`) currently
   stacks `imgs` and keeps `targets` as a list of dicts. Depth is
   per-image so this works as-is — `target["depth"]` stays in each dict.

   In the loss code, build a stacked depth tensor from the list:
   ```python
   depth_gt = torch.stack([t["depth"] for t in targets], dim=0)  # [B, 1, H, W]
   ```

6. The validation transform path
   (`iscar_bp.py:170-176`, `val_transforms`) sets `scale_range=(1.0,
   1.0)` already, so depth aug-side just needs the pad/flip to carry the
   depth field through; same `_filter`/`pad` updates apply.

---

## 8. Inference path changes

### 8.1 Dataclass

`inference.py:42-50` — extend `EoMTResult`:

```python
@dataclass
class EoMTResult:
    panoptic_mask: np.ndarray
    class_mask: np.ndarray
    segments: List[Dict[str, Any]]
    query_tokens: np.ndarray
    raw_masks: np.ndarray
    scores: np.ndarray
    depth: np.ndarray              # NEW, [H, W] float32, metres
```

### 8.2 `__call__` (`inference.py:291-332`)

Change the unpack at `inference.py:310` to:
```python
mask_logits_per_layer, class_logits_per_layer, occ_per_layer, depth_pred, query_tokens = self.model(tensor / 255.0)
```

After mask postprocessing (after line 330), add depth postprocessing
(mirror the same letterbox-crop-resize):
```python
# depth_pred: [1, 1, 1280, 1280]
sh, sw = self._scaled_size
depth_pred = depth_pred[:, :, :sh, :sw]
depth_pred = F.interpolate(depth_pred, self._original_size,
                           mode="bilinear", align_corners=False)
depth_np = depth_pred[0, 0].cpu().numpy().astype(np.float32)
```

Pass `depth_np` into `_postprocess` (extend its signature) or attach it
to the result directly:

```python
result = self._postprocess(mask_logits, class_logits, occ_logits, query_tokens)
result.depth = depth_np
return result
```

### 8.3 `_postprocess` (`inference.py:179-289`)

Easiest: leave it unchanged and set `depth` on the returned dataclass at
the call site (above). Otherwise add a `depth: np.ndarray` parameter and
plumb through; both branches that construct `EoMTResult` (the
empty-keep branch at `:213` and the success branch at `:282`) need
`depth=depth` added.

### 8.4 Visualisation (optional)

In `inference.py:visualize` (`inference.py:338-431`), turn the 1×3 figure
into 1×4 and add a depth panel:
```python
axes[3].imshow(result.depth, cmap="turbo")
axes[3].set_title("Depth (m)")
axes[3].axis("off")
```

---

## 9. Checkpoint / training plan

### 9.1 Loading existing EoMT weights

The existing `ckpt_path: "checkpoints/eomt_large_640.bin"` (yaml line 12)
and `panoptic_1280.bin` are loaded via
`LightningModule._load_ckpt` + `load_state_dict(strict=False)`
(`lightning_module.py:99-102`). The `strict_loading = False` flag at
`lightning_module.py:79` already makes missing keys (the new `depth_head`
will be missing) non-fatal.

There is one trap: `_raise_on_incompatible`
(`lightning_module.py:946-959`) **does** raise on missing keys. Update
that filter to also exempt `"depth_head"`:
```python
missing_keys = [
    key for key in incompatible_keys.missing_keys
    if "class_head" not in key and "class_predictor" not in key
       and "depth_head" not in key       # NEW
]
```

Equivalently, if `delta_weights=True` (the BP yaml has it),
`_zero_init_outside_encoder` zeros the depth head and then
`_reinit_missing_modules` (`lightning_module.py:882-912`) detects it and
re-initialises with PyTorch defaults. This is the correct path; just
verify `depth_head` is included in `reinit_modules` after the load.

### 9.2 Loading DA3 pretrained DPT (optional)

The user has DA3 files at `/home/kaelin/BinPicking/Depth-Anything-3/`.
`unknown — verify at implementation time` whether they have downloaded
`da3mono-large` weights (HuggingFace `depth-anything/DA3-mono-large` or
local under `Depth-Anything-3/checkpoints/`). If yes:

1. Load DA3 mono ckpt (Hugging Face safetensors or .pth — verify
   format). The weights live under a `head.*` submodule prefix in DA3's
   `DepthAnything3Net` (see `da3.py`).
2. Strip the `head.` prefix, remap to our `depth_head.` prefix.
3. **Patch-size mismatch caveat**: DA3 uses patch_size=14, we use 16. The
   only places `patch_size` shows up structurally inside DPT are
   `_forward_impl` lines 215 and 233-234 (computing `ph,pw` and
   `h_out,w_out`). All learned tensors (projects, resize_layers, scratch,
   refinenets, output_convs) are shape-independent of patch_size. So the
   weights are directly compatible.
4. The `dim_in`, `out_channels` and `features` all match
   (1024 / [256,512,1024,1024] / 256), so projects[*] and scratch.layer*
   are also direct copies.
5. Use `model.depth_head.load_state_dict(remapped_da3_state, strict=False)`.

Pseudocode for the remap:
```python
da3_ckpt = torch.load(da3_path, map_location="cpu")  # might be safetensors
da3_head = {
    k.replace("head.", "", 1): v
    for k, v in da3_ckpt.items()
    if k.startswith("head.") and "sky" not in k  # drop sky head
}
model.depth_head.load_state_dict(da3_head, strict=False)
```

### 9.3 Training schedule

Reuse the existing two-stage warmup-poly schedule
(`training/two_stage_warmup_poly_schedule.py`) which is already wired in
`lightning_module.py:158-164`. Concrete numbers (based on
`configs/dinov3/occlusion_bp/panoptic/eomt_large_640.yaml` defaults):

| Hyper | 640 value | 1280 + depth value |
|---|---|---|
| lr | 2e-4 | 1e-4 (halve when image scale doubles, conservative) |
| warmup_steps | [2000, 3000] | [2000, 3000] (unchanged; step-based not epoch) |
| max_epochs | 100 | start at 100; depth converges quickly so this is fine |
| batch_size | 4 | 1 or 2 (memory-bound; verify) |
| `depth_coefficient` | n/a | 1.0 (start) |
| `attn_mask_annealing_*_steps` | tuned for 640 | rescale to new step count or keep same fractions of training |

**Phased training**:
- Phase 0 (smoke test, optional): freeze everything except `depth_head`,
  train SI-Log only for ~2 epochs at lr=1e-3 → confirm head learns at
  all (loss should drop below 0.2).
- Phase 1 (joint): unfreeze, full loss with all coefficients
  (mask=3.0, dice=7.0, class=2.0, occlusion=1.0, depth=1.0). Use existing
  warmup. This is the main run.

There's no separate optimiser group needed for `depth_head`; it falls
into the `other_param_groups` branch (`lightning_module.py:150-153`)
naturally and gets the full `self.lr`. That is the desired behaviour
since the head is from-scratch (or DA3-pretrained but at much lower
data domain).

---

## 10. Validation plan

### 10.1 Depth head in isolation

1. Set `ckpt_path` to the existing `panoptic_1280.bin`.
2. Set `delta_weights=True` (existing behaviour) so encoder loads, and
   non-encoder modules are zeroed-then-reinit'd. Manually freeze
   everything except `depth_head` in a one-off training script:
   ```python
   for n, p in model.named_parameters():
       p.requires_grad = "depth_head" in n
   ```
3. Train SI-Log only for ~2 epochs at lr=1e-3 on the SDG dataset.
4. **Pass criterion**: `loss_depth` drops below 0.1 within 2 epochs and
   visualised depth on a held-out frame correlates qualitatively with the
   ground-truth depth (use `inference.visualize` with the new depth
   panel).

### 10.2 No regression on mask/class/occlusion

1. Train two short runs (~5 epochs) on a small held-out fold of the SDG
   dataset:
   - Run A: existing EoMT (no depth head, no resolution change) at 640.
   - Run B: new combined model at 1280 with depth_coefficient=1.0.
2. Compare PQ, mask AP, occlusion MAE between A and B *on the same
   evaluation frames* (downsample B's predictions to 640 for parity).
3. **Pass criterion**: B's PQ within ±2 pp of A. Anything worse means
   either the resolution upgrade hurt (debug pos-embed interpolation)
   or the depth gradient is dominating (try `depth_coefficient=0.5`).

---

## 11. Risks and open questions

Top 3 things most likely to go wrong:

1. **Pos-embed interpolation at 80×80 may degrade DINOv3 quality.** HF's
   `Dinov3Embeddings._interpolate_pos_encoding` interpolates from the
   pretrain resolution. The existing `panoptic_1280.bin` checkpoint
   suggests this has been tested, but the BP fine-tuned checkpoint
   `eomt_large_640.bin` was trained at 40×40 — sudden jump to 80×80 may
   require a few epochs of warmup before mask quality returns.
2. **Memory blowup at 1280.** The token sequence is 6605 (vs 1805); ViT
   attention is O(N²) in the patch dimension only (DA3 uses chunking),
   so peak attention memory ≈ 13× per layer. Without grad checkpointing
   the L4 ViT may OOM on a 24 GB GPU at any batch size > 1. Plan: enable
   `torch.utils.checkpoint` on the ViT blocks if OOM, or reduce
   `batch_size` to 1.
3. **Depth ground-truth scale convention.** Replicator has historically
   shipped depth as either `distance_to_camera` (Z-distance to camera
   centre, perpendicular to image plane) or `distance_to_image_plane`
   (perpendicular distance). DPT predicts perpendicular ray distance.
   `unknown — verify at implementation time` which the SDG writer
   produces; if it's `distance_to_camera`, convert to image-plane Z by
   multiplying by `cos(theta)` where theta is per-pixel ray angle from
   optical axis (computed from intrinsics). If wrong, SI-Log will
   converge to a non-zero floor.

Other open questions (resolve at implementation time):

- Exact filename convention for `depth.npy` in SDG output (§7.1).
- DA3 `da3mono-large` checkpoint location and format (§9.2).
- BP yaml `attn_mask_annealing_*_steps` — these are step indices, so
  they need rescaling to the new dataset size × new batch size.
- `tv_tensors.Mask` accepting float32 — torchvision may want a custom
  TVTensor subclass; if so, write `class DepthMap(tv_tensors.TVTensor)`.
- Whether `T.ScaleJitter` is geometrically valid for monocular depth
  ground-truth (§7.2 step 4). Recommend disabling.
- HF DINOv3 `_pos_embed` runtime cost at 80×80 — if it re-interpolates
  every forward, that's a perf hit; cache the interpolated embed once.

---

## 12. Ordered task list

Each step lists the file paths to touch, summary of changes, and
estimated complexity (S/M/L = small/medium/large).

**Phase A: Resolution upgrade (touches the most files; must land first
since depth head shapes are derived from image size).**

1. **[S] Create new training yaml.**
   File: `configs/dinov3/occlusion_bp/panoptic/eomt_large_1280.yaml`
   (copy of `_640.yaml`). Add `data.init_args.img_size: [1280, 1280]`.
   Update `name:` field to `iscar_bp_panoptic_eomt_large_1280_dinov3`.
   Drop `batch_size` to 1 (verify GPU memory). Update
   `attn_mask_annealing_*_steps` to whatever the new
   `len(dataset)/batch_size * max_epochs` gives.

2. **[S] Update inference default.**
   File: `inference.py:76`. Change `img_size: tuple = (640, 640)` to
   `(1280, 1280)`. Update comment at `:319`.

3. **[S] Update dataset module default.**
   File: `datasets/iscar_bp.py:143`. Change `img_size: tuple[int, int] =
   (640, 640)` to `(1280, 1280)`. (Effective value comes from yaml; this
   is just the ctor default.)

4. **[M] Sanity-test resolution change without depth head.**
   Run inference with `panoptic_1280.bin` at 1280 to confirm pos-embed
   interp + 80×80 grid + 320×320 mask logits all flow correctly. Fix
   anything broken before adding depth.

**Phase B: Depth head integration.**

5. **[M] Vendor DPT into the EoMT package.**
   Files: NEW `models/dpt.py` (copy of
   `Depth-Anything-3/src/depth_anything_3/model/dpt.py` with adjusted
   imports). NEW `models/dpt_utils.py` (subset of
   `Depth-Anything-3/src/depth_anything_3/model/utils/head_utils.py`
   containing `Permute`, `custom_interpolate`, `create_uv_grid`,
   `position_grid_to_embed`).

6. **[M] Wire DPT into the EoMT model.**
   File: `models/eomt.py`.
   - In `__init__` after line 63: instantiate `self.depth_head` (§4.2)
     and `self.depth_taps = (4, 11, 17, 23)`.
   - In `forward` (lines 162-222): add tap captures inside the block
     loop (§5), call DPT once after the post-loop `_predict`, return
     5-tuple.

7. **[M] Update all forward-pass consumers for the new tuple.**
   Files:
   - `inference.py:310-312` — unpack 5-tuple, postprocess depth.
   - `inference.py:42-50` — extend `EoMTResult` with `depth: np.ndarray`.
   - `inference.py:179-289` — set `depth` on result (or add param).
   - `training/lightning_module.py:175-178` (forward), `:180-199`
     (training_step) — unpack 5-tuple.
   - `training/mask_classification_panoptic.py:105` — unpack 5-tuple in
     `eval_step`.

**Phase C: Loss and dataset.**

8. **[S] Add SI-Log loss.**
   File: NEW `training/depth_loss.py` with `loss_depth_silog` (§6.1).

9. **[M] Wire depth loss into criterion.**
   File: `training/mask_classification_loss.py`.
   - Add `depth_coefficient` to `__init__` (`:25-59`).
   - Extend `forward` (`:62-90`) to optionally compute `loss_depth`.
   - Add `"depth"` branch in `loss_total` (`:196-219`).

10. **[S] Wire `depth_coefficient` through panoptic training module.**
    File: `training/mask_classification_panoptic.py:73-84` and signature
    at `:17-46`.

11. **[M] Make `_raise_on_incompatible` tolerate missing
    `depth_head` keys.**
    File: `training/lightning_module.py:946-959`. Add `"depth_head" not
    in key` to the missing-keys filter.

12. **[L] Extend `ReplicatorDataset` to load depth.**
    File: `datasets/iscar_bp.py:42-133`.
    - Load `depth.npy` per frame, wrap as a `tv_tensors.Mask` (or custom
      `DepthMap` TVTensor if Mask's int-only constraint bites).
    - Add `target["depth"]`.

13. **[M] Make transforms depth-aware.**
    File: `datasets/transforms.py:108-147`.
    - Update `pad` to also pad `target["depth"]` with NaN fill.
    - Update `_filter` to skip the per-instance filter for depth.
    - Optionally disable / restrict `ScaleJitter` (set yaml
      `scale_range: [1.0, 1.0]`).

14. **[S] Pass depth_gt into the criterion call.**
    File: `training/lightning_module.py:180-199`. After unpacking,
    `depth_gt = torch.stack([t["depth"] for t in targets])`. Pass
    `depth_pred=depth_pred, depth_gt=depth_gt` only on the **last
    iteration** of the deep-supervision loop (not all 5).

**Phase D: Validation and training.**

15. **[M] Run §10.1 isolated depth-head training.** Confirm SI-Log
    converges, checkpoint the result.

16. **[L] Run §10.2 joint training.** Compare PQ vs the existing 640
    baseline. Iterate on `depth_coefficient` if needed.

17. **[S] Optional: load DA3 pretrained DPT.** Per §9.2 — only if
    Phase D shows depth head trains slowly from scratch.

18. **[S] Optional: visualise depth in `inference.visualize`** (§8.4).

---

End of plan.
