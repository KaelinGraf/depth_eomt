# Depth Anything 3 — Monocular Depth Path

This document describes the encoder, head, upsampling chain, and intrinsics
handling of **DA3 monocular** (`da3mono-large`) so the existing EOMT project
can absorb the same recipe for monocular depth estimation. Multi-view, GS,
and pose-decoder branches are explicitly out of scope here.

Source repository: `/home/kaelin/BinPicking/Depth-Anything-3/src/depth_anything_3/`

## TL;DR — what the planner needs to copy

1. **Backbone:** plain DINOv2 ViT-Large (`vitl`, 24 blocks, embed_dim 1024,
   patch_size 14). Tap intermediate features at layers `[4, 11, 17, 23]`.
2. **Head:** single `DPT` with `output_dim=1`. No confidence channel, no
   normals head, no separate camera head. (DA3 mono produces depth ONLY.)
3. **Upsampling:** per-stage Conv1×1 projection → ConvTranspose ×4 / ×2 /
   identity / Conv stride-2 to align scales → 4-stage RefineNet feature
   pyramid (top-down with `bilinear interpolate`) → bilinear upsample to
   the input resolution → 3×3 conv → 3×3 conv → ReLU → 1×1 conv → 1
   channel → `exp` activation.
4. **Intrinsics input:** monocular DA3 does **not** consume intrinsics.
   The 9-d pose encoding (`extri_intri_to_pose_encoding` →
   `[T(3), quat(4), fov_h, fov_w]`) and the `CameraEnc` block exist only
   in the multi-view variants (`da3-large`, `da3-base`, etc.). For
   monocular mode `cam_enc=None`, `cam_dec=None`, `alt_start=-1`.
5. **Normals:** DA3 has no normals head anywhere in the released code.
   The "ray" output that appears alongside depth in `DualDPT` is the
   per-pixel camera-ray direction used for multi-view geometry, not a
   surface normal.

If your existing dataset does need normals, that supervision has to come
from a separate source (e.g. the Replicator `normals` annotator you
already have in `MonocularWriter`); DA3's monocular recipe doesn't
include it.

---

## 1. Backbone

### 1.1 Architecture

```yaml
# configs/da3mono-large.yaml
net:
  name: vitl                   # DINOv2 ViT-L
  out_layers: [4, 11, 17, 23]  # 4 intermediate taps for the DPT pyramid
  alt_start: -1                # multi-view alt-frame attention DISABLED
  qknorm_start: -1             # legacy qk-norm DISABLED
  rope_start: -1               # rope DISABLED
  cat_token: false             # do NOT concat local+global features at taps
```

`DinoV2` (`model/dinov2/dinov2.py:22`) wraps a stock DINOv2 ViT-L built
with `vit_large(img_size=518, patch_size=14, ffn_layer="mlp", alt_start=-1,
qknorm_start=-1, rope_start=-1, cat_token=false)` (`dinov2.py:49-57`). The
`img_size=518` here is the position-embedding pretrain resolution; the
ViT supports arbitrary input via `interpolate_pos_encoding`
(`vision_transformer.py:220-254`), so the actual inference image size
just needs to be a multiple of 14.

### 1.2 Tensor shapes through the backbone

For an input batch `x: [B, 3, H, W]` reshaped to `[B, S=1, 3, H, W]`
inside DA3 (mono == single view, S=1):

- patch grid: `H/14 × W/14` (e.g. for 1288×1288 input → 92×92 = 8464
  patches; for 1078×1078 → 77×77 = 5929)
- patch tokens: `[B*S, 1+R+P, C]` after `prepare_tokens_with_masks`
  (`vision_transformer.py:261-280`):
  - `1` cls token
  - `R` register tokens (default 4 for ViT-L)
  - `P = (H/14) * (W/14)` patch tokens
  - `C = 1024` (ViT-L embed dim)

For mono, all 24 blocks run as standard ViT self-attention (the
`alt_start: -1` gate at `vision_transformer.py:314-338` keeps every
attention "local"; no cross-frame global attention or camera-token
injection fires). Output of `get_intermediate_layers` (`dinov2.py:59-64`,
`vision_transformer.py:372-394`) is a list of 4 tensors, one per
out_layer, each shaped `[B, S, 1+R+P, C]` after the final `LayerNorm`.

The DPT head consumes these via `feats[i][0]` — `da3.py:178-179`
unwraps the `(local, global)` tuple to just the global features when
`cat_token=False`.

### 1.3 Patch-start index

DPT cuts off the cls/register tokens at patch_start_idx=0 by default in
the mono call path (`da3.py:209` calls `head(feats, H, W, patch_start_idx=0)`).

WAIT — this is mono-specific. The `DinoV2.forward` returns features that
are STILL pre-fixed by `1+R` non-patch tokens. In mono, DA3 passes
`patch_start_idx=0` because `dpt._forward_impl` line 218 does
`x = feats[take_idx][:, patch_start_idx:]` — but it's iterating over a
`[B*S, N_patch, C]` tensor reshaped at line 179 from
`feats[i][0].reshape(B*S, N, C)`. Inspecting more carefully:
`feats[i][0].shape = [B, S, N, C]` where `N = 1+R+P`, then reshape →
`[B*S, N, C]`, then `x[:, 0:]` keeps everything (cls + register +
patches). The reshape from N tokens to `(C, H/14, W/14)` at `dpt.py:221`
**will silently include the cls and register tokens as extra "patches"**.

In practice this works because (a) `H/14 * W/14` patches dominate (e.g.
8464 vs 5 non-patch), and (b) the per-stage 1×1 conv learns to suppress
the contribution of the constant non-patch slots. But it is a quirk to
be aware of when porting — if you'd rather be explicit, pass
`patch_start_idx=1+num_register_tokens` and adjust the reshape's `N`
accordingly.

---

## 2. The DPT depth head

Source: `model/dpt.py:31-336`.

### 2.1 Per-stage projections

```python
# dpt.py:94-111
self.projects = ModuleList([
    Conv2d(1024, 256,  1),   # taps 4, 11, 17, 23 each Conv1×1 to per-stage channels
    Conv2d(1024, 512,  1),
    Conv2d(1024, 1024, 1),
    Conv2d(1024, 1024, 1),
])

self.resize_layers = ModuleList([
    ConvTranspose2d(256,  256,  kernel=4, stride=4),  # ×4 upsample (stage 0)
    ConvTranspose2d(512,  512,  kernel=2, stride=2),  # ×2 upsample (stage 1)
    Identity(),                                        # stage 2 (full patch grid)
    Conv2d(1024, 1024, kernel=3, stride=2, padding=1) # /2 downsample (stage 3)
])
```

Stage shapes for input `H×W` (e.g. 1288×1288 → patch grid 92×92):

| stage | tap layer | post-`projects` | post-`resize` |
|------:|-----------|-----------------|---------------|
| 0     | 4 (early) | `[B, 256, 92, 92]`   | `[B, 256, 368, 368]` |
| 1     | 11        | `[B, 512, 92, 92]`   | `[B, 512, 184, 184]` |
| 2     | 17        | `[B, 1024, 92, 92]`  | `[B, 1024, 92, 92]`  |
| 3     | 23 (last) | `[B, 1024, 92, 92]`  | `[B, 1024, 46, 46]`  |

Pyramid scales: ×4 / ×2 / ×1 / /2 of the patch grid. Earliest features
get the most upsampling (high-res, low channels) and latest features get
downsampled and kept high-channel — classic DPT design.

### 2.2 Scratch + RefineNet fusion

```python
# dpt.py:113-122
self.scratch = _make_scratch([256, 512, 1024, 1024], features=256, expand=False)
# layerN_rn: Conv 3×3 channels→features, no bias  (dpt.py:362-376)

self.scratch.refinenet1 = _make_fusion_block(features=256, has_residual=True)
self.scratch.refinenet2 = _make_fusion_block(features=256)
self.scratch.refinenet3 = _make_fusion_block(features=256)
self.scratch.refinenet4 = _make_fusion_block(features=256, has_residual=False)  # top
```

Fusion order is **top-down (l4 → l3 → l2 → l1)**, with each
`FeatureFusionBlock` doing:

1. (Optional residual add of lateral input: `y = y + ResidualConvUnit(x_lateral)`)
2. `ResidualConvUnit(y)` (two 3×3 convs, bias, ReLU before each conv)
3. `bilinear interpolate` to the target size (either ×2 or `size=`
   provided by the caller — `dpt.py:268-284`)
4. `Conv2d(features, features, 1)` contraction

The exact fusion call sequence (`dpt.py:268-284`):
```python
out = refinenet4(l4_rn, size=l3_rn.shape[2:])      # 46×46 → 92×92
out = refinenet3(out, l3_rn, size=l2_rn.shape[2:]) # → 184×184
out = refinenet2(out, l2_rn, size=l1_rn.shape[2:]) # → 368×368
out = refinenet1(out, l1_rn)                       # ×2 default → 736×736
```

After `refinenet1` the feature map is at `4 × patch_grid` resolution
(input H × 4/14 ≈ H × 0.286).

### 2.3 Final upsample + output convs

```python
# dpt.py:233-263
h_out = ph * patch_size / down_ratio   # = H if down_ratio=1
w_out = pw * patch_size / down_ratio   # = W if down_ratio=1

fused = self.scratch.output_conv1(fused)            # 256 → 128, 3×3
fused = custom_interpolate(fused, (h_out, w_out),    # bilinear to FULL input size
                           mode='bilinear', align_corners=True)

main_logits = self.scratch.output_conv2(fused)       # depth head:
# Sequential(
#   Conv2d(128, 32, 3, padding=1),
#   ReLU,
#   Conv2d(32, output_dim=1, 1)
# )
out['depth'] = exp(main_logits).squeeze(1)           # activation='exp'
```

The `exp` activation (`dpt.py:294-295`) means the head predicts
log-depth and the output is positive metric depth. Multiplied by ground
truth this corresponds to an L1-on-log-depth loss target — see § 5.

### 2.4 Sky head (drop for our use case)

DA3's DPT also defines a `sky_output_conv2` that emits a 1-channel sky
mask used by the multi-view post-processing (`_process_mono_sky_estimation`,
`da3.py:155-179`). For bin-picking with no sky, set
`use_sky_head=False` when constructing the DPT.

---

## 3. Upsampling — exact recipe

Putting it all together for a `1288×1288` input (mono, ViT-L, patch 14 →
patch grid 92×92):

```
input image          1288 × 1288 × 3
└─ ViT patch embed   →  92 × 92 × 1024 (4 tap layers at 4, 11, 17, 23)
   │
   ├─ tap 4  → Conv1×1 → 256-ch → ConvTranspose×4 → 368×368×256 ─┐
   ├─ tap 11 → Conv1×1 → 512-ch → ConvTranspose×2 → 184×184×512 ─┤  RefineNet
   ├─ tap 17 → Conv1×1 → 1024 → identity         →  92×92×1024 ─┤  pyramid
   └─ tap 23 → Conv1×1 → 1024 → Conv stride-2    →  46×46×1024 ─┘   (top-down,
                                                                   bilinear)
   │
   refinenet4(46×46) → 92×92 → refinenet3 → 184×184 → refinenet2 →
       368×368 → refinenet1 → 736×736 (= 4 × patch_grid, 256 ch)
   │
   output_conv1: Conv3×3 256→128
   bilinear interpolate: 736×736 → 1288×1288
   output_conv2: Conv3×3 128→32, ReLU, Conv1×1 32→1
   exp(·)
   │
   output:  1288 × 1288  depth
```

The key reuse-points for porting into EOMT:

- The 4-tap fusion pyramid is generic. Wire it from EOMT's DINOv3 ViT-L
  by exposing the same kind of intermediate-layer hook (EOMT already
  taps the last layer for queries; you'll need a parallel route for
  layers 4/11/17/23).
- The exact upsampling ratios are determined by `out_channels` and the
  `resize_layers` ConvTranspose strides — keep them at their defaults,
  these are the values DA3 ships pretrained.

---

## 4. Camera intrinsics — only relevant for multi-view, NOT mono

For completeness: the multi-view DA3 variants encode camera parameters
into a 9-d pose vector and project it into a per-view "cam_token" that
is **prepended** to the patch sequence and processed jointly with image
patches by the trunk transformer.

- `extri_intri_to_pose_encoding` (`model/utils/transform.py:19-38`)
  produces a `[B, S, 9]` vector per view: `[t_x, t_y, t_z, q_x, q_y,
  q_z, q_w, fov_h, fov_w]` — translation (camera-to-world), quaternion
  (xyzw), and the two FOVs derived from `intrinsics[..., 0,0]`,
  `intrinsics[..., 1,1]` and image size.
- `CameraEnc` (`model/cam_enc.py:23-80`) wraps that 9-d pose with
  `Mlp(9 → 512 → 1024)`, LayerNorm, then a 4-block transformer "trunk"
  with `dim=1024`, `num_heads=16`, `mlp_ratio=4`. Output is a `[B, S, 1024]`
  cam_token, one per input view.
- That cam_token is injected via `x[:, :, 0] = cam_token` at
  `vision_transformer.py:331`, replacing the cls token at the layer
  controlled by `alt_start`. Only the multi-view variants set
  `alt_start >= 0`.

**For the EOMT integration this is irrelevant** — your pipeline trains
on a single image and Replicator already supplies depth-perfect ground
truth. There is no calibration ambiguity to inject. Skip the entire
`CameraEnc`/`CameraDec` plumbing.

---

## 5. Loss function (inferred — training code is not in the repo)

The released DA3 repo is inference-only: there is no training loop or
loss class shipped. Based on the activation choice, the paper
(`arxiv.org/abs/2511.10647`), and DA2's training recipe, the loss for
monocular depth is a combination of:

1. **Scale-invariant log-depth loss (SI-Log, from Eigen+2014)**
   ```
   L_silog(d, d̂) = √( mean( (log d - log d̂)² )
                     - λ · mean( log d - log d̂ )² )       (λ ≈ 0.85)
   ```
   With the head's `exp(·)` activation, this reduces to an L2 on the raw
   logit minus a scale-shift term, which is well-conditioned and
   scale-aware.

2. **Multi-scale gradient matching** (from MiDaS; standard for sharp
   edges in monocular depth):
   ```
   L_grad = Σ_s mean( |∇x R_s| + |∇y R_s| ),  R_s = log d - log d̂
                                                  at scale 2^-s, s∈{0,1,2,3}
   ```

3. **L1 on inverse depth** (used by DA1/DA2 to be robust to outliers
   especially in foreground regions):
   ```
   L_inv = mean( |1/d - 1/d̂| )
   ```

Total loss is typically `α · L_silog + β · L_grad + γ · L_inv`, weights
tuned to ~`(1.0, 0.5, 0.1)` in DA2's released training code. For the
EOMT integration, **start with SI-Log only** — it's the dominant term
and getting it wired correctly with the new head is more important than
exactly reproducing weights you can't validate against the release.

If sky / out-of-range pixels are present in your dataset they should be
masked (`mask = isfinite(d_gt) & (d_gt > 0) & (d_gt < D_max)`) before
the loss is reduced. Replicator's `distance_to_image_plane` returns
`inf` for sky/no-hit; mask those out of the training tensor.

---

## 6. Input preprocessing

DA3 mono input pipeline (`utils/io/input_processor.py`):

1. PIL load → BGR→RGB
2. `_resize_longest_side(img, target_size)` aspect-preserving resize so the
   long side matches `process_res` (default ≈ 700 for `da3mono-large`).
3. `_make_divisible_by_resize(img, PATCH_SIZE=14)` rounds each dimension
   to the nearest multiple of 14 (no padding).
4. To tensor [0,1] float32 → `Normalize(mean=[0.485,0.456,0.406],
   std=[0.229,0.224,0.225])` — **standard ImageNet normalisation**.
5. Final shape: `[1, 3, H_pad, W_pad]` where both are 14-divisible.

Note: this differs from EOMT's `inference.py:_preprocess` which only
divides by 255 and does NOT apply ImageNet mean/std — see EOMT doc for
the discrepancy. When wiring DA3 into EOMT you need to match the
backbone's expected normalisation; if you keep EOMT's existing DINOv3
weights then EOMT's "no ImageNet normalisation" pipeline must also be
the input to the new depth head.

---

## 7. Mapping to EOMT's intended setup

EOMT operates on 640×640 today and is being upgraded to 1280×1280.
DA3's patch_size is 14 and EOMT's DINOv3 patch_size is **16**, so:

- 1280 / 16 = 80 → patch grid 80×80 (EOMT)
- 1288 / 14 = 92 → patch grid 92×92 (DA3 reference)

DPT in DA3 hard-codes `patch_size=14` only as a multiplier in
`dpt._forward_impl` for computing `h_out, w_out` from the patch grid
shape. Override this to `patch_size=16` when constructing the DPT for
the EOMT integration. The 4-stage pyramid ratios (×4, ×2, ×1, /2) and
the RefineNet fusion are otherwise patch-size-agnostic — they operate
on whatever resolution the per-stage `resize_layers` produce.

So the EOMT planner can copy DPT verbatim with only:
- `patch_size = 16`
- `out_channels = [256, 512, 1024, 1024]` (matches DINOv3-L embed_dim 1024)
- `dim_in = 1024`
- `output_dim = 1`
- `head_name = "depth"`
- `use_sky_head = False`
- 4 tap layers chosen at roughly the same depth fractions DA3 uses on
  ViT-L. DINOv3-L is also 24 blocks, so layers `[4, 11, 17, 23]` map
  one-to-one. Confirm at the EOMT side how query tokens are inserted at
  layer N — the depth taps must come from BEFORE the EOMT query tokens
  arrive (i.e. tap the patch sequence pre-query-injection at each tap
  layer; the early taps will be unaffected; the last tap may need to
  re-extract patch tokens by slicing off the trailing query block).
