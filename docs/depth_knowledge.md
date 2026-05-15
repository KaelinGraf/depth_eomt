# Depth Anything 3 — Monocular Depth Knowledge

This is the **canonical reference** for what DA3 actually does in monocular
mode and what the EoMT integration must therefore implement. Future agents
working on the depth path in this repo are required (by `CLAUDE.md`) to
read this file before making changes.

It supersedes `docs/da3_monocular.md`, which has multiple factual errors
catalogued in §13.

All DA3 file paths are relative to
`/home/kaelin/BinPicking/eomt/Depth-Anything-3/src/depth_anything_3/`
unless prefixed with `eomt:`. EoMT paths are relative to
`/home/kaelin/BinPicking/eomt/`.

---

## 1. TL;DR — what DA3 monocular actually is

| Aspect                  | DA3 mono (`da3mono-large`)                                  |
|-------------------------|-------------------------------------------------------------|
| Backbone                | DINOv2 ViT-L (24 blocks, embed_dim 1024, patch 14)          |
| Register tokens         | **0** (DINOv2 vit_large default; not the 4 in DINOv3)       |
| Tap layers              | `[4, 11, 17, 23]` (out of 24)                               |
| Tap features given to DPT | **All four normed** by the encoder's final `LayerNorm`. Patch-only (CLS stripped). |
| Camera intrinsics input | **NONE.** No `cam_enc`, no `cam_dec`, no FOV/pose injection. Multi-view-only. |
| Head                    | Single `DPT` with `output_dim=1`, `activation="exp"`        |
| Sky head                | **Present** by default in mono, used to clip sky pixels     |
| Normals head            | **Does not exist** in DA3 anywhere                          |
| Confidence channel      | Not produced for mono (`has_conf` = `output_dim > 1` only)  |
| Output                  | `exp(logits)` → strictly positive metric depth, `[B, S, H, W]` |
| Preprocessing           | PIL RGB → longest-side resize to 504 → 14-divisible → ImageNet mean/std |
| Loss                    | **Not in the released repo** (inference-only). Inferred SI-Log + grad-matching + L1-inverse with weights ≈ `(1.0, 0.5, 0.1)`. |

The two facts that matter most for the EoMT integration and that are most
commonly misremembered:

1. **Mono DA3 takes no camera intrinsics.** The 9-d pose encoding
   (`extri_intri_to_pose_encoding`) and the `CameraEnc` MLP+transformer
   live only in the multi-view path. If the user wants an "intrinsics MLP"
   on EoMT depth, that is *not* part of DA3 mono — it's a custom borrow
   from multi-view DA3 or from VGGT/MiDaS metric-depth recipes.
2. **All four taps are normed by the encoder's final `LayerNorm`** before
   DPT sees them — not just the last tap. This is critical if DA3
   pretrained DPT weights are ever loaded into the EoMT depth head.

---

## 2. Backbone

### 2.1 Config

`configs/da3mono-large.yaml:6-17` (paths under `Depth-Anything-3/src/depth_anything_3/`):

```yaml
net:
  name: vitl                   # DINOv2 ViT-L
  out_layers: [4, 11, 17, 23]  # 4 intermediate taps for DPT
  alt_start: -1                # multi-view alt-frame attention DISABLED
  qknorm_start: -1             # legacy qk-norm DISABLED
  rope_start: -1               # rope DISABLED
  cat_token: false             # do NOT concat (local, global) at taps
```

`DinoV2.__init__` (`model/dinov2/dinov2.py:33-57`) maps these to
`vit_large(img_size=518, patch_size=14, ffn_layer="mlp", ...)`.

### 2.2 Register tokens — DA3 uses 0, EoMT uses 4

`model/dinov2/vision_transformer.py:429-440` shows `vit_large` is
hardcoded to `embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4`, with
**default `num_register_tokens=0`**. `DinoV2` does not override this. So
the DA3 mono backbone produces a token sequence of `[CLS | patches]`,
length `1 + P`, with `P = (H/14) * (W/14)`.

EoMT uses HuggingFace's DINOv3-L (`facebook/dinov3-vitl16-pretrain-lvd1689m`),
which has **4 register tokens** plus 1 CLS = 5 prefix tokens
(`eomt:models/vit.py:66`). Implications:

- The patch-only slice in the EoMT depth tap is `x[:, num_prefix:, :]`
  with `num_prefix = 5`, not `x[:, 1:, :]`. This is correctly handled in
  `eomt:models/eomt.py:246-258`.
- Registers absorb high-norm "outlier patch" artifacts present in DINOv2
  (Darcet+2023). A DPT head trained against DINOv2 features (DA3 mono)
  has implicitly learned to live with those outliers; trained against
  DINOv3 (EoMT), the same DPT topology will train more cleanly from
  scratch.

### 2.3 Tap mechanism — patches-only, fully normed

The full call chain in mono mode:

1. `DepthAnything3Net.forward` (`model/da3.py:100-153`): with
   `extrinsics=None`, calls `self.backbone(x, cam_token=None, ...)`.
2. `DinoV2.forward` (`dinov2.py:59-64`) forwards into
   `self.pretrained.get_intermediate_layers(x, self.out_layers, ...)`.
3. `prepare_tokens_with_masks` (`vision_transformer.py:261-280`) builds
   the token sequence `[CLS | patches]` (no registers).
4. `_get_intermediate_layers_not_chunked` (`vision_transformer.py:300-349`)
   iterates blocks. With `alt_start=-1`, every gate at lines 314, 323,
   333 is skipped — each block runs `process_attention(x, blk, "local")`
   (line 338). At each `i ∈ blocks_to_take`, line 346 appends
   `(out_x[:, :, 0], out_x)` — `(cls_only, full_seq)`.
5. `get_intermediate_layers` (`vision_transformer.py:372-398`) then:
   - **Line 384**: `outputs = [self.norm(out[1]) for out in outputs]` —
     applies the **encoder's final `LayerNorm` to every tap**.
   - **Line 396**: slices off CLS+register: `out[..., 1 + num_register :, :]`.
     With 0 registers this is `out[..., 1:, :]` — patch tokens only.
   - Returns a tuple of `(patch_tokens_normed, cls_token)` per tap, shape
     `([B, S, P, C], [B, S, C])`.
6. `_process_depth_head` (`da3.py:205-209`) calls
   `self.head(feats, H, W, patch_start_idx=0)`. DPT unpacks via
   `B, S, N, C = feats[0][0].shape` — `feats[i][0]` is the patch-only
   normed tensor.

Inside DPT, `dpt.py:219` does `x = self.norm(x)`, which is `nn.Identity()`
for `da3mono-large` because `norm_type="idt"` is the default
(`dpt.py:88-92`). So no extra norm.

⇒ **DPT receives patches-only tokens already LN-normed by `encoder.norm`,
with `patch_start_idx=0` correct because no prefix tokens are present.**

### 2.4 Implications for the EoMT integration

Currently `eomt:models/eomt.py:246-258` only norms the layer-23 tap:

```python
if i == n_blocks - 1:
    tap_x = self.encoder.backbone.norm(tap_x)
depth_tap_feats.append(tap_x[:, start:, :].contiguous())
```

This is the **wrong** behaviour for DA3 fidelity. DA3 mono norms **all
four taps**. To match — and to keep the option of loading DA3 pretrained
DPT weights live — change the tap capture so all four taps go through
`self.encoder.backbone.norm` before the patch slice. See §11.

EoMT's final-norm is the same `LayerNorm` already used by `_predict`
(`eomt:models/eomt.py:211`); applying it to the tap captures does not
add learned parameters or change the panoptic path.

---

## 3. The DPT head

Source: `model/dpt.py:31-336`. Vendored verbatim into
`eomt:models/dpt.py` with only the helper-imports redirected to
`eomt:models/dpt_utils.py`.

### 3.1 Per-stage projections

`dpt.py:94-96`:
```python
self.projects = ModuleList([
    Conv2d(dim_in=1024, oc, 1, stride=1, padding=0)
    for oc in out_channels  # (256, 512, 1024, 1024) for ViT-L
])
```

So Conv1×1 1024→{256, 512, 1024, 1024} for stages 0..3.

### 3.2 Per-stage resize_layers

`dpt.py:100-111`:
```python
ConvTranspose2d(256,  256,  kernel=4, stride=4, padding=0)   # stage 0: ×4
ConvTranspose2d(512,  512,  kernel=2, stride=2, padding=0)   # stage 1: ×2
Identity()                                                    # stage 2: ×1
Conv2d(1024, 1024, kernel=3, stride=2, padding=1)            # stage 3: /2
```

For input `H × W` (e.g. EoMT 1280×1280 with patch 16, so patch grid
80×80):

| stage | tap | post-`projects` channels × HW | post-`resize_layers` |
|------:|-----|------------------------------|-----------------------|
| 0     | 4   | 256, 80×80                   | 256, **320×320**      |
| 1     | 11  | 512, 80×80                   | 512, **160×160**      |
| 2     | 17  | 1024, 80×80                  | 1024, 80×80           |
| 3     | 23  | 1024, 80×80                  | 1024, **40×40**       |

### 3.3 Scratch (lateral 3×3) and RefineNet pyramid

`_make_scratch` (`dpt.py:362-376`) creates `layer{1..4}_rn`: 3×3
channels-only convs `in_channels → features=256`, no bias. So all four
laterals come out at 256 channels.

`_make_fusion_block` (`dpt.py:342-359`) builds
`FeatureFusionBlock(features=256, activation=ReLU, deconv=False, bn=False,
expand=False, align_corners=True, has_residual=...)`. Topology
(`FeatureFusionBlock.forward`, `dpt.py:436-458`):

1. If `has_residual=True` and a lateral input is provided: `y = xs[0] + ResidualConvUnit(xs[1])`.
2. Always: `y = ResidualConvUnit(y)`.
3. Bilinear upsample: target size driven by `size=` kwarg (else
   `scale_factor=2`).
4. `Conv2d(features, features, 1)` contraction (`out_conv`).

`ResidualConvUnit` (`dpt.py:379-404`):
`ReLU → Conv 3×3 → ReLU → Conv 3×3 → skip-add(x)`.

`refinenet1` has `has_residual=True`; `refinenet4` has
`has_residual=False`. The fusion call sequence (`dpt.py:268-284`):

```python
out = refinenet4(l4_rn,        size=l3_rn.shape[2:])  # /2 grid → 1× grid
out = refinenet3(out, l3_rn,   size=l2_rn.shape[2:])  # 1× grid → 2× grid
out = refinenet2(out, l2_rn,   size=l1_rn.shape[2:])  # 2× grid → 4× grid
out = refinenet1(out, l1_rn)                          # 4× grid → ×2 default → 8× grid
```

`refinenet1` does **not** receive a `size=` arg, so it uses
`scale_factor=2`. **The fused map after refinenet1 is at 8 × patch_grid**
(= 2 × the stage-0 resized resolution = 2 × 4 × patch_grid).

For EoMT 1280×1280 (patch_grid 80×80): fused = **640×640 × 256**.

### 3.4 Output convs and final upsample

`output_conv1` (`dpt.py:127-129`): `Conv2d(256, 128, 3, padding=1)`. No
activation, no LN.

The fused map is then bilinearly interpolated to the full output
resolution (`dpt.py:234-237` and `:241`):
```python
h_out = ph * patch_size // down_ratio   # = H if down_ratio=1
w_out = pw * patch_size // down_ratio   # = W if down_ratio=1
fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear",
                            align_corners=True)
```

`down_ratio=1` is the mono default (`dpt.py:60`); only DualDPT/GS heads
override it. So for mono, fused is upsampled `8 × patch_grid → H × W`,
e.g. 640×640 → 1280×1280 for EoMT.

`output_conv2` for the depth head (`dpt.py:138-143`):
```python
Sequential(
    Conv2d(128, 32, kernel=3, stride=1, padding=1),
    *ln_seq,                       # empty unless use_ln_for_heads=True
    ReLU(inplace=True),
    Conv2d(32, output_dim=1, kernel=1, stride=1, padding=0),
)
```

ReLU sits between the 32-conv and the 1-conv; the final 1×1 has no
activation before `exp`.

### 3.5 Activation — `exp`

`dpt.py:253-256`:
```python
outs[self.head_main] = self._apply_activation_single(
    main_logits, self.activation     # "exp"
).squeeze(1)
```

`_apply_activation_single` for `"exp"` is literally `torch.exp(x)`
(`dpt.py:294-295`). After the outer `forward` `view(B, S, ...)` reshape
(`dpt.py:188`), the return is `outs["depth"]` of shape
`[B, S, H, W]` — strictly positive metric depth. **No clamping, no
inverse-depth or disparity transform inside the head.**

### 3.6 Sky head — present by default in mono

`use_sky_head=True` is the **DPT default** (`dpt.py:56`) and
`da3mono-large.yaml` does not override it. So the released
`da3mono-large` does carry a sky head:
- Topology mirrors the depth head (`Conv 3×3 128→32 → ReLU → Conv 1×1
  32→1`) — `dpt.py:146-154`.
- Activation is `relu` (`sky_activation="relu"` default, `dpt.py:58,
  311-328`).
- Output key `out["sky"]`, shape `[B, S, H, W]`.
- Used downstream by `_process_mono_sky_estimation` (`da3.py:155-179`)
  to clamp sky pixels to a quantile-derived max depth.

For bin-picking with no sky, **disable**: pass `use_sky_head=False` at
DPT instantiation. The current EoMT integration does this correctly
(`eomt:models/eomt.py:75-90`).

### 3.7 Confidence channel

`has_conf = output_dim > 1` (`dpt.py:78`). For mono `output_dim=1`, **no
confidence channel is predicted**. The `_conf` outputs only appear in
`DualDPT` (multi-view).

### 3.8 Patch-size mismatch (DA3 14 → DINOv3 16)

`patch_size` enters DPT at three points only:

- `dpt.py:66`: stored as `self.patch_size` for shape math.
- `dpt.py:215`: `ph, pw = H // self.patch_size, W // self.patch_size` —
  the patch-grid shape used for the per-stage reshape.
- `dpt.py:233-234`: `h_out = int(ph * self.patch_size / self.down_ratio)`
  — the upsample target.

All learned parameters (`projects`, `resize_layers`, `_make_scratch`
laterals, `RefineNet`s, `output_conv*`) are **shape-independent of patch
size**. ⇒ Switching `patch_size=14 → 16` requires only the constructor
arg; **DA3 mono pretrained DPT weights are byte-compatible** with a
`patch_size=16` instantiation as far as tensor shapes are concerned. The
EoMT integration sets `patch_size=patch_size[0]=16` at
`eomt:models/eomt.py:77`.

### 3.9 DPT input chunking

`forward` (`dpt.py:166-198`) accepts `chunk_size=8` by default — chunks
the batch×view dimension to limit peak memory for large multi-view
inputs. For mono `S=1`, EoMT passes `chunk_size=None`
(`eomt:models/eomt.py:283`), running unchunked. Fine for `B*S = B`.

---

## 4. Camera intrinsics — mono uses NONE

This is the section most worth getting right.

### 4.1 What the multi-view path does (for context)

For multi-view variants (`da3-large.yaml`, `da3-base.yaml`,
`da3metric-large.yaml`):

1. `extri_intri_to_pose_encoding` (`model/utils/transform.py:19-38`)
   produces a `[B, S, 9]` per-view vector
   `[t_x, t_y, t_z, q_x, q_y, q_z, q_w, fov_h, fov_w]` from extrinsics
   and intrinsics (FOVs derived from `intrinsics[..., 0,0]` and
   `intrinsics[..., 1,1]`).
2. `CameraEnc` (`model/cam_enc.py:23-80`) wraps this 9-d vector with
   `Mlp(9 → 512 → 1024)` + `LayerNorm` + 4-block transformer trunk
   (`dim=1024, num_heads=16, mlp_ratio=4`). Output: `[B, S, 1024]` —
   one cam_token per view.
3. The cam_token is injected into the ViT trunk at `vision_transformer.py:331`:
   `x[:, :, 0] = cam_token` — replacing the CLS slot at the layer
   indicated by `alt_start`. Only multi-view configs set `alt_start ≥ 0`.

### 4.2 What mono does — nothing

`da3.py:124-130`:
```python
if extrinsics is not None:
    cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
else:
    cam_token = None
```

`da3mono-large.yaml` sets neither `cam_enc` nor `cam_dec`, so both modules
are `None` (`da3.py:73-80`) — they don't exist as `nn.Module`s. Mono
inference call sites (`api.py`, `cli.py`, `services/inference_service.py`)
all pass `extrinsics=None, intrinsics=None`. The camera-token injection
path in `vision_transformer.py:314-331` is gated by `alt_start != -1`;
with `alt_start=-1` it never executes.

⇒ **DA3 mono is purely visual: RGB in → metric depth out, with no
geometric conditioning whatsoever.** No FOV. No focal length. No
extrinsics. No cam_token.

### 4.3 Implications for the EoMT integration

The user has expressed an interest in adding "input mlp for camera
intrinsics". This is **not** part of DA3 mono. Three options if
intrinsics conditioning is genuinely wanted:

- **Option A (do nothing).** Match DA3 mono. Bin-picking has fixed
  intrinsics per camera setup; no scale ambiguity exists at
  inference if training and test cameras share intrinsics. This is the
  default the integration currently follows.
- **Option B (borrow DA3 multi-view's CameraEnc).** Add a
  `CameraEnc(9 → 512 → 1024)` MLP that produces a cam_token from
  `(extri_intri_to_pose_encoding(extr, intr, image_size))`. Inject it
  as a replacement CLS at the first block (or earlier than EoMT's query
  insertion at block 20). Cost: changes to `eomt.py:forward` to weave
  in a cam_token; new dataset field for intrinsics. Benefit:
  generalisation across cameras with differing focal lengths.
- **Option C (FOV-only conditioning, MiDaS/UniDepth-style).** Skip the
  multi-view machinery and add a small per-image FOV embedding
  (`Mlp(2 → C)` on `[fov_h, fov_w]`) added to the patch tokens before
  block 0. Lighter than B; more honest about "the only thing that
  matters monocularly is FOV".

If the user picks A (recommended for tightest DA3 fidelity), the
"input mlp for camera intrinsics" mention should be deleted from any
todo lists. If B or C, it is a separate scope of work not currently
implemented.

---

## 5. Output parameterization

Single channel, `exp(logits)` → strictly positive metric depth. No
clamping, no inverse-depth, no disparity. Final shape `[B, S, H, W]`
after the `view(B, S, ...)` reshape at `dpt.py:188`. EoMT's wrapper
flattens `S=1` and re-adds a channel dim to expose `[B, 1, H, W]` for
downstream consumption (`eomt:models/eomt.py:283-289`).

The supervision ground truth determines the units. For Replicator's
`distance_to_image_plane` annotator the ground truth is metric metres,
so the head learns metres directly.

---

## 6. Loss functions

**The released DA3 repo is inference-only.** No `loss.py`, no
`Loss` class, no training loop. The only "loss" matches in the source
tree (`bench/utils.py:366,428`) are camera-pose evaluation errors at
benchmark time, not training.

So the loss formulation has to be inferred from:

- The head's `exp` activation → predicts log-depth → consistent with
  losses on log-space.
- The DA3 paper (`arxiv.org/abs/2511.10647`) and DA2's open training
  recipe.

Standard monocular-depth loss for this family:

### 6.1 SI-Log (Eigen+2014, λ ≈ 0.85)

Let `R = log d̂ - log d_gt` over valid pixels (`isfinite(d_gt) &
(d_gt > 0) & (d_gt < D_max)`).

```
L_silog = sqrt( mean(R^2) - λ * mean(R)^2 )
```

`λ = 1` is purely scale-invariant (variance of R). `λ = 0.85` mixes in
some absolute-scale signal. With the head's `exp` activation this is
equivalent to L2 on the raw logit minus a scale-shift term.

### 6.2 Multi-scale gradient matching (MiDaS)

Encourages sharp depth edges:

```
L_grad = Σ_s mean( |∂_x R_s| + |∂_y R_s| ),  R_s = R at scale 2^-s, s ∈ {0,1,2,3}
```

### 6.3 L1 on inverse depth (DA1/DA2)

Robust to foreground outliers in outdoor / wide-range scenes:

```
L_inv = mean( |1/d̂ - 1/d_gt| )
```

### 6.4 Total

Typical DA2 weights: `1.0 * L_silog + 0.5 * L_grad + 0.1 * L_inv`. DA3
paper claims a similar combination but without exposed coefficients.

### 6.5 Recommendation for EoMT bin-picking

For synthetic Replicator GT (perfect supervision, scenes < 2 m, no
sky), **start with SI-Log only** — it's the dominant term and the one
with the cleanest interaction with the `exp` activation. The current
implementation `eomt:training/depth_loss.py` does this. Optionally add
gradient matching once SI-Log converges, for edge sharpness. **Skip
L1-inverse**: useful for outdoor foreground bias, irrelevant for
bin-picking.

---

## 7. Input preprocessing

`utils/io/input_processor.py:35-113`:

- Constants: `NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225])` — **standard ImageNet stats**.
  `PATCH_SIZE = 14` (line 57).
- Default `process_res = 504`, `process_res_method = "upper_bound_resize"`
  (every caller path uses these — `api.py:145-146`, `cli.py:124`,
  `services/inference_service.py:48-49,105-106,186-187`).
- Pipeline (`_process_one`, `:219-257`):
  1. PIL load → `.convert("RGB")` (`_load_image`, `:298-307`). PIL
     produces RGB directly; **no BGR→RGB** step.
  2. `_resize_longest_side(img, 504)` aspect-preserving (`:324-334`),
     `cv2.INTER_CUBIC` for upsample, `cv2.INTER_AREA` for downsample.
  3. `_make_divisible_by_resize(img, 14)` rounds dims to nearest
     multiple of 14 (`:365-383`).
  4. `T.ToTensor()` → float32 [0,1] CHW.
  5. `T.Normalize(mean, std)` — ImageNet normalisation.
- Final shape after stacking: `(1, N, 3, H_pad, W_pad)`, both spatial
  dims 14-divisible.

**Important for the EoMT integration**: `eomt:models/eomt.py:163`
already applies the same ImageNet normalisation internally. So the
features the depth head sees are ImageNet-normed, matching DA3's
expectation. *Do not* re-normalise externally before feeding the model
— it would be double-normalised.

EoMT's preprocessing differs from DA3's in three ways that don't break
the integration but are worth noting:

- EoMT letterboxes to a square (zero-padding bottom/right),
  not 14-divisible aspect-preserving.
- EoMT uses bilinear (`cv2.INTER_LINEAR`) at all sizes, not
  cubic/area split.
- EoMT divides by 255 at the call site, not via `T.ToTensor()`.

---

## 8. Sky / normals / confidence

| Head      | Mono?   | Notes                                                            |
|-----------|---------|------------------------------------------------------------------|
| Sky       | **Yes (default)** | `use_sky_head=True` default; `da3mono-large.yaml` doesn't override. Used to clip sky pixels in postprocess. Disable for bin-picking. |
| Normals   | **No**  | DA3 has no normals head anywhere in `model/`. The "ray" output in `DualDPT` is camera-ray direction (multi-view geometry), not a surface normal. |
| Confidence| **No**  | `has_conf = output_dim > 1`; mono has `output_dim=1`. Conf only in `DualDPT`. |

If the EoMT pipeline requires normals supervision, that has to come from
a separate head; DA3's recipe does not provide it.

---

## 9. DPT input/output cheat-sheet for EoMT

For EoMT 1280×1280 input with DINOv3-L (patch 16, 24 blocks, 5 prefix
tokens, 200 queries prepended at block 20):

```
Per tap:
  x at block i ──► slice prefix (and queries if i ≥ 20) ──►
                   apply encoder.norm ──►                   (see §11; not yet
                                                             implemented for taps 4/11/17)
                   patch tokens [B, 80*80=6400, 1024]

DPT call:
  feats_wrapped = [(t.unsqueeze(1),) for t in [tap4, tap11, tap17, tap23]]
                  # each [(B, S=1, 6400, 1024),]
  depth = depth_head(feats_wrapped, H=1280, W=1280, patch_start_idx=0,
                     chunk_size=None)["depth"]
                  # [B, S=1, 1280, 1280]

DPT internal pyramid (channels × spatial):
  stage 0  256 × 320×320
  stage 1  512 × 160×160
  stage 2  1024 × 80×80
  stage 3  1024 × 40×40

  scratch.layer{1..4}_rn  → 256 × {320×320, 160×160, 80×80, 40×40}

  RefineNet pyramid (top-down, bilinear-interp driven by size= arg):
    refinenet4(l4_rn, size=l3_rn) → 256 × 80×80
    refinenet3(_,    l3_rn, size=l2_rn) → 256 × 160×160
    refinenet2(_,    l2_rn, size=l1_rn) → 256 × 320×320
    refinenet1(_,    l1_rn)             → 256 × 640×640   (×2 default)

  output_conv1 (256→128 3×3) → 128 × 640×640
  custom_interpolate(bilinear) → 128 × 1280×1280
  output_conv2 (128→32 3×3, ReLU, 32→1 1×1) → 1 × 1280×1280
  exp(·) → metric depth, [B, 1, 1280, 1280]
```

---

## 10. Layer-tap considerations for DINOv3

DINOv3 ViT-L is also 24 blocks, so `[4, 11, 17, 23]` is range-valid. The
fractional split (`5/24, 12/24, 18/24, 24/24` ≈ `0.21, 0.5, 0.75, 1.0`)
is the standard DPT/MiDaS "evenly spaced + final" recipe and is
backbone-agnostic.

That said, four DINOv3 vs DINOv2 differences could shift what's at each
layer:

1. **RoPE.** DINOv3 applies Rotary Position Embedding *at each block's
   attention*. DINOv2 (and DA3's wrapper, `rope_start=-1`) uses
   absolute learned positional embedding only at embed time. So in
   DINOv3 every block re-injects position; layer-4 features may carry
   *more* spatial structure than DINOv2 layer-4.
2. **Gated MLP (SwiGLU).** DINOv3 uses gated MLPs; DA3's DINOv2 uses
   plain MLP (`ffn_layer="mlp"`, `dinov2.py:48`). Affects activation
   distributions but not the per-layer "what's encoded" curve
   dramatically.
3. **Register tokens (4 vs 0).** DINOv3 has 4 registers absorbing
   high-norm artifacts; DA3's DINOv2 has 0. Cleaner late-layer features
   in DINOv3 — DPT trained from scratch on EoMT should converge faster.
4. **Query prepend at block 20.** EoMT prepends 200 query tokens before
   block 20 (`eomt:models/eomt.py:177-181`). Tap 23 lives inside the
   "query era": those late-stage patch tokens attend to (and from) the
   query stream, so they're query-conditioned. The patch slice strips
   the queries, but the patch-token contents themselves are not "pure
   visual" any more. For DA3 fidelity, `[4, 11, 17, 19]` would tap
   before queries arrive. The current `[4, 11, 17, 23]` is the
   plan-as-written compromise.

Recommendations (to evaluate empirically; not in code yet):

- **Default `[4, 11, 17, 23]`** is the safe baseline. Ship it.
- If edge sharpness is poor, try `[6, 12, 18, 23]` — RoPE may make
  layer-4 redundant.
- If panoptic queries dominate the late blocks (visible as depth
  artifacts that correlate with mask boundaries), try
  `[4, 11, 17, 19]` — last tap pre-query.
- If DA3 pretrained DPT weights are loaded, the four-tap recipe is
  fixed at `[4, 11, 17, 23]` (else the pretrained scratch convs see
  feature distributions they weren't trained on).

The constructor at `eomt:models/eomt.py:30` accepts an arbitrary tuple,
and `configs/dinov3/occlusion_bp/panoptic/eomt_large_1280.yaml:40`
exposes `depth_taps` as a yaml override. Tuning is a yaml change, no
code edit needed.

---

## 11. Open issues against the current EoMT integration

The audit at the time this doc was written (see git log) found the
integration is **substantially code-complete** against
`docs/depth_integration_plan.md` §12 (steps 1-14 done, optional
steps 17-18 not done). The deviations from DA3 fidelity that future
agents should be aware of:

### 11.1 ⚠️ All four taps should be normed, not just tap 23

**Current** (`eomt:models/eomt.py:248-256`):
```python
tap_x = x
if i == n_blocks - 1:
    tap_x = self.encoder.backbone.norm(tap_x)
depth_tap_feats.append(tap_x[:, start:, :].contiguous())
```

**DA3-faithful** (per §2.3):
```python
tap_x = self.encoder.backbone.norm(x)
depth_tap_feats.append(tap_x[:, start:, :].contiguous())
```

This is required if DA3 pretrained DPT weights are to be loaded
(plan §9.2). Without it, the per-stage 1×1 projections see un-normed
features at taps 4/11/17 and DA3-pretrained projections will produce
out-of-distribution activations.

The current code chose `norm_type="layer"` for the in-DPT norm
(`eomt:models/eomt.py:84`), which compensates partially by adding
per-tap LN inside DPT. That's defensible for from-scratch training but
breaks pretrained-weight loading and diverges from the DA3 recipe.

Recommendation: change tap capture to norm all four taps **and** keep
`norm_type="layer"` until from-scratch training is confirmed
working, then optionally flip to `"idt"` if loading DA3 weights.

### 11.2 ⚠️ "Intrinsics MLP" is multi-view only

The user's mention of an "input MLP for camera intrinsics" describes
the `CameraEnc` module that lives only in DA3's multi-view path. Mono
DA3 has no intrinsics input. Current EoMT integration correctly does
not include one. If intrinsics conditioning is desired, it is a
separate scope of work — see §4.3.

### 11.3 attn_mask annealing steps not rescaled

`configs/dinov3/occlusion_bp/panoptic/eomt_large_1280.yaml:21-22`
leaves `attn_mask_annealing_*_steps` at the 640 values with an inline
comment to rescale. Empirical, training-time fix.

### 11.4 ckpt_path

`eomt_large_1280.yaml:12` points at `checkpoints/eomt_large_640.bin`
even though `checkpoints/panoptic_1280.bin` may be a better resolution
match. Verify by trying to load both.

### 11.5 SDG depth filename

`datasets/iscar_bp.py:55-61` hardcodes `depth.npy`. Verify at training
time that the `MonocularWriter` class at
`/home/kaelin/BinPicking/SDG/IS/camera_monocular.py:564` emits exactly
that filename and not e.g. `distance_to_image_plane_*.npy`.

### 11.6 Depth scale convention

Replicator can output either `distance_to_camera` (Z to camera centre,
i.e. ray length) or `distance_to_image_plane` (perpendicular Z to image
plane). DPT predicts the latter (image-plane Z). If the SDG writer
produces `distance_to_camera`, depths must be converted by
`Z_plane = Z_ray * cos(theta)` per pixel where `theta` is the per-pixel
ray angle from the optical axis (computable from intrinsics). Wrong
choice silently makes SI-Log converge to a non-zero floor.

### 11.7 Sky head correctly disabled

`eomt:models/eomt.py:85` sets `use_sky_head=False`. Correct for
bin-picking. No issue — flagged here only for completeness because
DA3's default is the opposite.

### 11.8 Optional plan items

- §17 (DA3 pretrained DPT load): not done. Would require fixing 11.1
  first.
- §18 (depth viz panel in `inference.visualize`): not done. Currently
  `inference.py:374` is still a 1×3 figure.

---

## 12. Quick-reference table for porting

| Concept                    | DA3 mono                          | EoMT integration target          |
|----------------------------|-----------------------------------|----------------------------------|
| Backbone                   | DINOv2 ViT-L (24 blk, patch 14)   | DINOv3 ViT-L (24 blk, patch 16)  |
| Register tokens            | 0                                 | 4                                |
| Prefix tokens (CLS+reg)    | 1                                 | 5                                |
| Tap layers                 | `[4, 11, 17, 23]`                 | `[4, 11, 17, 23]` (yaml-tunable) |
| Tap norm                   | `encoder.norm` on all 4 taps      | **Currently norm only tap 23 — FIXME §11.1** |
| Tap slice                  | Patch only (no CLS/reg)           | Patch only (no CLS/reg/queries)  |
| Image size                 | longest-side 504, 14-divisible    | 1280×1280 letterboxed            |
| Patch grid                 | (H/14)×(W/14)                     | 80×80                            |
| ImageNet norm              | external in input_processor       | internal in `EoMT.forward`       |
| DPT `dim_in`               | 1024                              | 1024                             |
| DPT `out_channels`         | (256, 512, 1024, 1024)            | (256, 512, 1024, 1024)           |
| DPT `features`             | 256                               | 256                              |
| DPT `output_dim`           | 1                                 | 1                                |
| DPT `activation`           | `"exp"`                           | `"exp"`                          |
| DPT `use_sky_head`         | True (default — disable for BP)   | False                            |
| DPT `norm_type`            | `"idt"` (because taps pre-normed) | `"layer"` (compensating, see §11.1) |
| Camera intrinsics          | Not used                          | Not used                         |
| Confidence channel         | Not produced                      | Not produced                     |
| Final output shape         | `[B, S=1, H, W]`                  | `[B, 1, H, W]`                   |
| Loss                       | Inferred SI-Log + grad-match + L1-inv | SI-Log only (good baseline)  |

---

## 13. Corrections to `docs/da3_monocular.md`

Catalogued so future readers don't re-introduce the same errors:

1. **§1.2 / §1.3:** Claims DA3 mono uses 4 register tokens. It uses 0.
   `vit_large` defaults `num_register_tokens=0`
   (`vision_transformer.py:429`); `DinoV2.__init__` does not override.
2. **§1.2 / §1.3:** Claims DPT silently treats CLS+register as extra
   patches with `patch_start_idx=0`. Wrong. `get_intermediate_layers`
   strips the prefix at `vision_transformer.py:396` before returning;
   DPT receives patch tokens only and `patch_start_idx=0` is correct.
3. **§1.2:** Claims only the final-layer tap is normed. All four taps
   are normed by the same `self.norm` at `vision_transformer.py:384`.
4. **§2.4:** Implies the sky head is multi-view only.
   `use_sky_head=True` is the DPT default and `da3mono-large.yaml`
   doesn't override it. Released mono has a sky head.
5. **§3:** "fused = 736×736 = 4 × patch_grid". Numerically right for
   1288×1288, but the multiplier is 8 × patch_grid, not 4.
   `refinenet1`'s default `scale_factor=2` doubles `4 × patch_grid`.
6. **§6:** Claims "BGR→RGB". DA3 uses PIL `.convert("RGB")` — RGB
   directly, no BGR→RGB step.
7. **§6:** Claims default `process_res ≈ 700`. Default is 504; every
   caller uses 504.
8. **§7:** Closing paragraph claims EoMT does NOT apply ImageNet
   mean/std. EoMT *does* apply it internally
   (`eomt:models/eomt.py:163`).

End of file.
