# EoMT Architecture (Binpicking config, DINOv3 ViT-L/16, 640x640)

This document describes the as-built architecture of the EoMT (Encoder-only
Mask Transformer) model in `/home/kaelin/BinPicking/eomt`, specifically as
configured for the bin-picking panoptic+occlusion task
(`configs/dinov3/occlusion_bp/panoptic/eomt_large_640.yaml`). It is intended
as input to a planner that wants to wire a second head (e.g. Depth Anything
V3 monocular depth) onto the same backbone.

All file:line references are relative to `/home/kaelin/BinPicking/eomt/`.

---

## 1. Backbone

- **Variant**: DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`),
  loaded via HuggingFace `AutoModel.from_pretrained` and adapted to a
  timm-style API by `ViT.transformers_to_timm` (`models/vit.py:54`).
- **Loading site**: `inference.py:125`
  ```python
  encoder = ViT(img_size=self.img_size,
                backbone_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
  ```
- **Patch size**: 16 (read from HF config in `vit.py:57-59`).
- **Image size**: 640 x 640 (`inference.py:76`, `img_size=(640, 640)`).
- **Patch grid**: 40 x 40 = **1600 patch tokens** (`vit.py:60-63`).
- **Embed dim**: 1024 (DINOv3 ViT-L hidden size, set by HF config and copied
  to `backbone.embed_dim` at `vit.py:65`). The class head dimensions and
  query embeddings all key off `encoder.backbone.embed_dim` (e.g.
  `eomt.py:37-47`), so the planner should treat 1024 as the canonical token
  width.
- **Num layers**: 24 (DINOv3 ViT-L). Exposed as `backbone.blocks` after the
  `transformers_to_timm` rename at `vit.py:67`.
- **Prefix tokens**: `backbone.num_prefix_tokens = num_register_tokens + 1`
  (`vit.py:66`). DINOv3 uses 4 register tokens + 1 CLS = **5 prefix tokens**.
- **ImageNet normalization** is registered as buffers on the `ViT` wrapper
  (`vit.py:48-52`) and is applied by `EoMT.forward` itself
  (`eomt.py:163`), not by the user.

### Input tensor flow

| Stage                                    | Shape                          |
|------------------------------------------|--------------------------------|
| `EoMT.forward` input (after `/255.0`)    | `[B, 3, 640, 640]`             |
| After `(x - pixel_mean) / pixel_std`     | `[B, 3, 640, 640]`             |
| After `backbone.patch_embed(x)`          | `[B, 5 + 1600, 1024]` = `[B, 1605, 1024]` (CLS+4 register+1600 patch tokens; positional embeds added by `_pos_embed` at `eomt.py:171-172`) |
| Each ViT block input/output              | `[B, N, 1024]` (N grows once queries are concatenated) |

---

## 2. Token flow through EoMT

The encoder runs as a single stack, but the last `num_blocks` ViT layers are
re-purposed: query tokens are prepended to the sequence and updated jointly
with the patch tokens.

### Query tokens (`self.q`)

- Defined in `eomt.py:37`:
  `self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)`
- For the BP config: `num_q = 200`, embed_dim = 1024 → `q.weight` is
  `[200, 1024]` (`configs/.../eomt_large_640.yaml:27`).
- They are pure learnable parameters (one row per query), conceptually like
  Mask2Former object queries but injected mid-encoder rather than into a
  separate decoder.

### Insertion point

In `EoMT.forward` (`eomt.py:177-181`):

```python
for i, block in enumerate(self.encoder.backbone.blocks):
    if i == len(self.encoder.backbone.blocks) - self.num_blocks:
        x = torch.cat(
            (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
        )
```

- `num_blocks` defaults to **4** (`eomt.py:24`); the BP yaml does not
  override it, so the BP model uses **4**.
- With 24 ViT blocks, queries are **prepended** (not appended) to the
  sequence before block index 20, i.e. they are processed by the last
  4 blocks (indices 20, 21, 22, 23).
- After concat, the sequence becomes
  `[B, num_q + num_prefix + num_patch, 1024] = [B, 200 + 5 + 1600, 1024] = [B, 1805, 1024]`.
- Layout used everywhere downstream:
  `[ Q queries | CLS + 4 reg | 1600 patch tokens ]`.
  See the slicing in `_predict` (`eomt.py:66-70`):
  - queries: `x[:, :num_q, :]` → `[B, 200, 1024]`
  - patch tokens: `x[:, num_q + num_prefix_tokens :, :]` → `[B, 1600, 1024]`

### Masked attention (during the last `num_blocks` blocks)

Inside each of the last 4 blocks, a per-block mask (`_attn_mask` at
`eomt.py:133-160`) is computed from the current intermediate mask
predictions. Queries can only attend to patch tokens that the previous
layer's mask said belonged to them. This mask is annealed away during
training (`attn_mask_probs` buffer + `_disable_attn_mask` at
`eomt.py:84-94`); at inference (`attn_mask_probs == 1`) the mask is fully
applied.

### Final query output

After the last block, `_predict(self.encoder.backbone.norm(x))`
(`eomt.py:211`) runs once more and returns the final query tokens
`q = x[:, :num_q, :]` → `[B, 200, 1024]`.

These are the embeddings exposed to the user as `query_tokens` (used in
`inference.py:286` and saved into `EoMTResult.query_tokens`).

---

## 3. Output heads

All three heads are applied to the same query slice `q = x[:, :num_q, :]`
inside `EoMT._predict` (`eomt.py:65-81`).

### 3.1 Class head (`self.class_head`)

- Definition (`eomt.py:39`):
  `nn.Linear(embed_dim, num_classes + 1)`.
- For BP: `num_classes = 2`, so output is `[B, 200, 3]` (the extra slot is
  the "no-object" / void class used by the Hungarian matcher and is also
  treated as background in inference; see `inference.py:210`).
- Input: query tokens `[B, 200, 1024]`. Output: `[B, 200, 3]`.

### 3.2 Mask head (`self.mask_head`)

- Definition (`eomt.py:41-47`): 3-layer MLP
  `Linear(1024,1024) → GELU → Linear(1024,1024) → GELU → Linear(1024,1024)`.
  No final activation; output is a per-query embedding of size 1024.

### 3.3 Mask logits — exact upsampling path

This is the part most relevant for grafting on a depth head, because the
mask path is where EoMT goes from token space back to image space.

The patch tokens are reshaped back to a 2D feature map and upsampled by
`self.upscale`, then dot-producted with the mask-head embedding of each
query (`eomt.py:70-77`):

```python
x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]   # [B, 1600, 1024]
x = x.transpose(1, 2).reshape(B, 1024, 40, 40)                        # [B, C, H/p, W/p]
mask_logits = einsum("bqc, bchw -> bqhw",
                     self.mask_head(q),    # [B, 200, 1024]
                     self.upscale(x))      # [B, 1024, H', W']
```

`self.upscale` is **not** bilinear interpolation; it is a learned
convolutional decoder (`eomt.py:61-63`):

```python
num_upscale = max(1, int(math.log2(max_patch_size)) - 2)   # = 2 for patch_size 16
self.upscale = nn.Sequential(*[ScaleBlock(embed_dim) for _ in range(num_upscale)])
```

Each `ScaleBlock` (see `models/scale_block.py:11-38`) is:

```
ConvTranspose2d(C, C, kernel=2, stride=2)      # 2x upsample
GELU
Conv2d(C, C, 3, padding=1, groups=C, bias=False)  # depthwise refinement
LayerNorm2d(C)
```

So with patch_size 16 and `num_upscale = log2(16) - 2 = 2`, we get **two
ScaleBlocks**, taking the patch grid from 40x40 → 80x80 → **160x160**.

Final mask-logit tensor produced inside `_predict`:

```
mask_logits: [B, num_q=200, H/4=160, W/4=160]
```

i.e. it is at 1/4 of the input resolution, **not** the original 640x640.
Restoring it to 640x640 (and then to the user's image) is done by
`inference.py:319-330` using `F.interpolate(..., mode="bilinear")` —
described in section 5 below.

### 3.4 Occlusion head (`self.occlusion_head`)

Defined (`eomt.py:49-55`) only when `enable_occlusion=True` (true for BP):

```
Linear(1024, 1024) → GELU → Linear(1024, 1024) → GELU → Linear(1024, 1)
```

It is a **per-query MLP** (no attention pool, no cross-token interaction):
applied directly to `q` in `_predict` (`eomt.py:79`):

```python
occlusion_logits = self.occlusion_head(q).squeeze(-1)   # [B, num_q] = [B, 200]
```

The output is a single scalar logit per query. At inference it is
sigmoided and interpreted as a visibility / non-occlusion ratio
(`inference.py:207`, key `"visibility_ratio"`).

### 3.5 Query-token output

`_predict` also returns the raw normalized queries `q` (post-`norm`,
post-mask-head-input, but pre-`mask_head` MLP):
`[B, num_q, embed_dim] = [B, 200, 1024]` (`eomt.py:81`, surfaced through
`inference.py:286`).

---

## 4. Per-layer outputs

`EoMT.forward` collects predictions at multiple depths
(`eomt.py:175, 187-191, 211-215`):

- For each of the last `num_blocks` blocks (indices
  `len(blocks) - num_blocks` ... `len(blocks) - 1`), **before** the block
  runs, it:
  1. Applies `self.encoder.backbone.norm(x)`,
  2. Calls `self._predict(...)` to get
     `(mask_logits, class_logits, occlusion_logits, _)`,
  3. Appends each of those to the per-layer lists,
  4. Uses the produced `mask_logits` to build the attention mask for that
     block.
- After the loop, `_predict` is run **one more time** on the final output
  (post-block-23), and that final tuple is appended too.

So with `num_blocks=4`, the lists each contain **5 entries** (one before
each of the last 4 blocks plus one after the last block).

There is **one shared head** of each kind (`class_head`, `mask_head`,
`occlusion_head`, `upscale`) — it is the same nn.Module called at every
depth. There are **no per-layer copies**.

Inference uses only the final entry (`inference.py:315-317`):

```python
mask_logits = mask_logits_per_layer[-1]
class_logits = class_logits_per_layer[-1]
occ_logits = occ_per_layer[-1] if occ_per_layer is not None else None
```

Training applies the loss at every entry (deep supervision); see section 6.

The function returns
`(mask_logits_per_layer, class_logits_per_layer, occlusion_logits_per_layer | None, query_tokens)`
(`eomt.py:217-222`).

---

## 5. Inference postprocessing

In `EoMTInference._preprocess` (`inference.py:144-177`) the original image
of shape `(H, W)` is fit inside 640x640 with aspect-preserving resize:

- `scale = min(640 / H, 640 / W)`
- `new_h, new_w = round(H * scale), round(W * scale)` → stored as
  `self._scaled_size`.
- The image is bilinearly resized to `(new_h, new_w)` and placed in the
  top-left corner of a 640x640 zero canvas (bottom-right zero pad).
- Original `(H, W)` stored as `self._original_size`.

After `__call__` runs the model, the mask logits come out at 160x160. The
post-resize logic (`inference.py:319-330`) is exactly:

```python
# 1) upsample 160x160 → padded 640x640 (covers the entire padded canvas)
mask_logits = F.interpolate(mask_logits, self.img_size,
                            mode="bilinear", align_corners=False)

# 2) crop the zero-padding region away (keep top-left scaled-image area)
sh, sw = self._scaled_size
mask_logits = mask_logits[:, :, :sh, :sw]

# 3) bilinearly resize from (sh, sw) back to original (H, W)
mask_logits = F.interpolate(mask_logits, self._original_size,
                            mode="bilinear", align_corners=False)
```

This mirrors `LightningModule.revert_resize_and_pad_logits_instance_panoptic`
(`training/lightning_module.py:739-754`).

`_postprocess` (`inference.py:179-289`) then:
1. Softmaxes class logits, keeps queries with `class != background-slot` and
   `score > mask_thresh` (default 0.01) (`inference.py:201-210`).
2. Sigmoids mask logits, picks per-pixel argmax over kept queries (weighted
   by their class score) → `mask_ids` (`inference.py:222-223`).
3. For each kept query: requires `orig_mask = sigmoid > 0.5` and
   `final_area / orig_area >= overlap_thresh` (default 0.1)
   (`inference.py:235-246`).
4. Stuff classes (configurable, default `[0]`) get merged into one segment
   per class (`inference.py:249-261`).
5. Returns `EoMTResult` containing `panoptic_mask`, `class_mask`, segment
   list (with `occlusion_score`, `visibility_ratio`, `area`,
   `confidence`...), `query_tokens` (`[num_q, embed_dim]`, here
   `[200, 1024]`), `raw_masks` (`[N, H, W]` bool), and `scores` (`[N]`).

---

## 6. Losses (training)

The training criterion is `MaskClassificationLoss`
(`training/mask_classification_loss.py:25`), a subclass of HuggingFace's
`Mask2FormerLoss`. It is constructed in
`training/mask_classification_panoptic.py:73-84`.

### 6.1 Hungarian matching

`Mask2FormerHungarianMatcher` (`mask_classification_loss.py:54-59`) matches
the 200 predicted queries to ground-truth masks per image, using
- `cost_class = class_coefficient` (BP yaml leaves default = 2.0),
- `cost_mask  = mask_coefficient`  (default 3.0),
- `cost_dice  = dice_coefficient`  (default 7.0).
The matcher uses point sampling with `num_points = 25600` for mask costs.

### 6.2 Loss terms (per layer)

Computed in `MaskClassificationLoss.forward`
(`mask_classification_loss.py:62-90`):

- `loss_mask` — sigmoid cross-entropy on point-sampled mask logits.
  `mask_classification_loss.py:165` (default branch) or weighted variant
  (`mask_classification_loss.py:144-150`) when `use_area_weighting=True`
  (which the BP yaml sets, line 15).
- `loss_dice` — point-sampled Dice loss
  (`mask_classification_loss.py:166` or `:152-157` with area weighting).
- `loss_cross_entropy` — class CE with empty-class re-weighted by
  `no_object_coefficient` (default 0.1) (`mask_classification_loss.py:50-52`,
  inherited Mask2Former `loss_labels` populates this key).
- `loss_occlusion` — **smooth L1** between
  `sigmoid(occlusion_queries_logits[matched])` and
  ground-truth `target["occlusion"]` (visibility ratios in [0, 1]),
  computed only over matched queries whose target class is a "thing"
  (`class > 0`); see `loss_occlusion`
  (`mask_classification_loss.py:174-194`). Despite the name "BCE" being
  natural for visibility, it is in fact **smooth-L1 on sigmoided logits**.

Targets shape (from `iscar_bp.py`):

- `target["masks"]`: `[N_inst, H, W]` boolean mask per instance
  (`iscar_bp.py:105`).
- `target["labels"]`: `[N_inst]` int (class IDs) (`iscar_bp.py:106`).
- `target["occlusion"]`: `[N_inst]` float in `[0, 1]`, sourced from
  `obj["visibility_ratio"]` in the dataset metadata
  (`iscar_bp.py:95, 107`).
- `target["is_crowd"]`: per-instance crowd flag.

### 6.3 Loss weighting and total

`loss_total` (`mask_classification_loss.py:196-219`) applies coefficients by
key substring match:

| Key substring | Coefficient (BP defaults from `mask_classification_panoptic.py`) |
|---------------|-------------------------------------------------------------------|
| `mask`        | `mask_coefficient` = 3.0                                          |
| `dice`        | `dice_coefficient` = 7.0                                          |
| `cross_entropy` | `class_coefficient` = 2.0                                       |
| `occlusion`   | `occlusion_coefficient` = 1.0                                     |

### 6.4 Deep supervision

`LightningModule.training_step` (`training/lightning_module.py:180-199`)
iterates over `mask_logits_per_block`, `class_logits_per_block`,
`occlusion_logits_per_block`, applies the criterion at **every** block's
output, suffixes loss keys with `_block_<i>` (last block has no suffix), and
sums them all in `loss_total`. So with `num_blocks=4`, deep supervision is
applied at 5 depths.

---

## 7. Image preprocessing

Performed by `EoMTInference._preprocess` (`inference.py:144-177`):

1. Input: `[H, W, 3]` BGR uint8 (OpenCV default).
2. **BGR → RGB** via `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`
   (`inference.py:161`).
3. **Aspect-preserving resize** with `scale = min(640/H, 640/W)`
   (bilinear, `cv2.INTER_LINEAR`) (`inference.py:164-169`).
4. **Letterbox / zero-pad** to 640x640 in the **top-left corner**:
   the resized image is placed at `padded[:new_h, :new_w, :]` and the
   bottom/right is zero (`inference.py:171-173`).
5. To tensor: `[1, 3, 640, 640]` float32, **range [0, 255]**
   (`inference.py:176`).
6. Division by 255.0 happens at the call site
   (`inference.py:311`, `self.model(tensor / 255.0)`), so the model sees
   `[0, 1]` floats.
7. **ImageNet normalisation** is applied **inside** `EoMT.forward`
   (`eomt.py:163`):
   `x = (x - encoder.pixel_mean) / encoder.pixel_std`
   where `pixel_mean = [0.485, 0.456, 0.406]`,
   `pixel_std = [0.229, 0.224, 0.225]` (`vit.py:48-52`).

   So strictly speaking, EoMT **does** apply ImageNet mean/std — but it does
   so internally, after the user passes `[0, 1]` RGB. From the user's
   perspective `_preprocess` does only BGR→RGB + letterbox + /255; do **not**
   pre-normalise externally or it will be normalised twice.

---

## 8. Dimensions and constants (BP config)

| Constant                | Value                          | Source                                |
|-------------------------|--------------------------------|---------------------------------------|
| Input image size        | 640 x 640                      | `inference.py:76`, yaml link via `main.py:127-129` |
| Patch size              | 16                             | DINOv3 ViT-L/16; `vit.py:57-59`       |
| Patch grid (H/p, W/p)   | 40 x 40 (1600 patches)         | `vit.py:60-63`                        |
| Embed dim               | 1024                           | DINOv3 ViT-L hidden size; `vit.py:65` |
| ViT layers              | 24                             | DINOv3 ViT-L; `backbone.blocks`       |
| Num prefix tokens       | 5 (1 CLS + 4 register)         | `vit.py:66`                           |
| `num_q` (queries)       | 200                            | `configs/.../eomt_large_640.yaml:27`  |
| `num_blocks`            | 4 (default; BP yaml does not override) | `eomt.py:24`                  |
| Num classes             | 2 (`background`, `part`)       | yaml line 13; `inference.py:75`; CLASS_NAMES `inference.py:66` |
| Class-head output       | num_classes + 1 = 3            | `eomt.py:39`                          |
| Mask logits resolution  | 160 x 160 (4× upsample of 40×40 via 2 ScaleBlocks) | `eomt.py:57-63`, `scale_block.py`  |
| Stuff classes           | `[0]` (background)             | yaml line 37                          |
| Occlusion enabled       | True                           | yaml line 26                          |

### Token-sequence layout summary (post-query-insertion, last 4 blocks)

```
index 0           ... num_q-1  | num_q ... num_q+4 | num_q+5 ... num_q+1604
                                                     (= 1605 ... 1804)
[ 200 query tokens             | 5 prefix tokens   | 1600 patch tokens     ]
total tokens N = 1805,    embed C = 1024
```

This is the layout the planner needs to slice into when adding a new head
(e.g. a depth decoder that consumes the patch-token portion `[B, 1600, C]`
or its 40x40 reshape after the final block).

---

## 9. Notes for grafting a second head (e.g. DA3 depth)

These are observations from the code, not recommendations:

- The patch-token feature map at the final block is recoverable as
  `x[:, num_q + num_prefix_tokens :, :].transpose(1,2).reshape(B, 1024, 40, 40)`
  (`eomt.py:70-73`). This is what `self.upscale` consumes for masks.
- A depth head sharing the same backbone could either:
  (a) tap the same final patch-token feature map after `_predict` (giving
  `[B, 1024, 40, 40]`), or
  (b) tap intermediate ViT blocks (DA3 typically uses 4 evenly-spaced ViT
  layers as a feature pyramid). Currently nothing in `EoMT.forward`
  exposes intermediate hidden states to callers; this would need code
  changes to keep references during the loop in `eomt.py:177-209`.
- `forward` currently returns 4 items (`eomt.py:217-222`); adding a depth
  output means either expanding the tuple or wrapping in a dataclass.
- The ImageNet mean/std normalisation inside `EoMT.forward`
  (`eomt.py:163`) is shared, so a depth head reading from the encoder gets
  the same normalisation pipeline — no extra wiring needed for input
  normalisation.
- DA3 uses a different patch size and image resolution by default; this
  EoMT instance is locked to 640x640 / patch 16 (40x40 token grid). Any
  depth head must operate at this token grid (or upsample from it).
- Backbone-only LR (`llrd`, layer-wise learning-rate decay) is set up in
  `LightningModule.configure_optimizers`
  (`training/lightning_module.py:106-156`); a new head would by default
  fall into the `other_param_groups` branch (full `self.lr`).

Where details are not determinable from code:

- The **exact number of register tokens** is read at runtime from the HF
  config (`vit.py:66`). For `facebook/dinov3-vitl16-pretrain-lvd1689m` it
  is 4 (giving num_prefix_tokens = 5), but verify via
  `model.encoder.backbone.num_prefix_tokens` at training time.
- The **DA3 integration** itself is not in this repo; the planner will need
  to choose how to fuse depth supervision (separate optimizer group? joint
  loss? frozen backbone?). Not determined from code; check at training time.
