# Project Memory — EoMT × DA3 monocular depth integration

Append-only log. Per `~/.claude/CLAUDE.md`, every chunk closed by an
orchestrator gets one entry. Do not edit prior entries; do not
consolidate.

---

## [2026-05-08 14:50] Chunk 0: DA3 audit, integration audit, knowledge file

**Goal**: Establish a code-grounded reference for DA3 monocular depth,
audit the current EoMT integration state against
`docs/depth_integration_plan.md`, and produce a TODO list for the
remaining work. No code edits in this chunk.

**Tasks**: #1-#6 (all completed). Pre-team. Single-orchestrator session
with two ad-hoc forks for read-only investigation.

**Forks** (one-shot, no team):
- `da3-deepdive` — full audit of `Depth-Anything-3/src/depth_anything_3/`
  for the monocular path. Surfaced 8 corrections to
  `docs/da3_monocular.md` (catalogued in §13 of the new
  `docs/depth_knowledge.md`).
- `eomt-state-audit` — audit of all `M`/`??` files vs the
  `docs/depth_integration_plan.md` §12 ordered task list. Result: 13/14
  non-optional plan items are code-complete; only verification and
  empirical/training-time items remain.

**Files written**:
- `docs/depth_knowledge.md` (NEW) — canonical DA3-mono reference;
  supersedes `docs/da3_monocular.md`.
- `CLAUDE.md` (NEW) — agent-facing index, requires
  `docs/depth_knowledge.md` reading before depth work; encodes the
  depth-taps + intrinsics-MLP design decisions.
- `.gitignore` — added `Depth-Anything-3/`.
- `memory.md` (this file) — created per global protocol.

**Surprises / context for future chunks**:
- **DA3 mono uses 0 register tokens, not 4**. `vit_large(num_register_tokens=0)`
  default; `DinoV2.__init__` doesn't override
  (`vision_transformer.py:429`). EoMT's DINOv3 has 4. So the prefix-token
  count differs between DA3 and EoMT (1 vs 5).
- **All 4 DA3 taps are normed** by the encoder's final `LayerNorm`
  (`vision_transformer.py:384`), not just tap 23 as the plan and current
  EoMT code assume. Current `models/eomt.py:248-256` only norms tap 23.
  This is functionally OK from-scratch (compensated by DPT's
  `norm_type="layer"`) but blocks loading DA3 pretrained DPT weights.
- **DA3 mono ingests no camera intrinsics**. The 9-d pose encoding and
  `CameraEnc` are multi-view-only. The user's mention of an "intrinsics
  MLP" was based on a misread of the multi-view path. Decision: still
  add intrinsics conditioning (UniDepth-V1-style), but as a deliberate
  departure from DA3 mono, not a port.
- **DPT released code is inference-only**. No training loss class
  ships. Loss formulation has to be inferred from the `exp` activation
  + DA2 recipe (SI-Log + grad-match + L1-inv at ~1.0/0.5/0.1).
- **`use_sky_head=True` is DPT default** and `da3mono-large.yaml`
  doesn't override it; released mono has a sky head used to clip sky
  pixels. EoMT correctly disables it.
- **Patch-size 14 → 16 swap is byte-compatible** for DA3 pretrained
  DPT weights — `patch_size` only affects shape math in
  `_forward_impl`, not learned tensors.

**Decisions made (encoded in `CLAUDE.md`)**:
- Depth taps: `[4, 11, 17, 23]` (one-to-one DA3 port; yaml-overridable).
- Intrinsics: yes, UniDepth-V1-style 6-d cam_vec → MLP → 1024-d cam_token,
  prepended to ViT sequence after `_pos_embed`. Gated by yaml flag.

**Follow-ups created**: Tasks #7-#19 (next chunk). See `TaskList`.

**Status**: chunk closed.

---

## [2026-05-11 22:05] Chunk 1: IntrinsicsMLP module

**Goal**: Land the new `models/intrinsics_mlp.py` so downstream EoMT.forward wiring (#4) has the import target.

**Tasks**: #3.

**Actor** (`actor-intr-module`):
- `models/intrinsics_mlp.py` (NEW, 43 lines) — `IntrinsicsMLP` class per the spec.
- Iteration count: 1.
- Final status: approved on first round.

**Reviewer** (`reviewer-intr-module`):
- Ran a 13-bucket edge-case suite locally (topology, B=1/4, non-square images, non-identity K, gradient flow, K.grad selectivity, dtype/device handling). All passed.
- Sign-off: APPROVED.

**Surprises / context for future chunks**:
- `image_size` is `(H, W)` tuple ordering. Downstream `#4` must pass tuples in that order.
- Float32 batched-matmul nondeterminism produces ~2e-6 max-abs diff between unbatched vs batched forward of the same K. Use `atol >= 1e-5` in any comparison test downstream.
- Param count: 264,960. Negligible overhead.
- Module accepts float32 K only; downstream depth path runs float32/autocast so this is fine.

**Re-fan-out**: `#4` (Wire IntrinsicsMLP into EoMT.forward) is now blocked only by `#1` (depth-norm). Will dispatch the intr-module pair to `#4` as soon as `#1` closes — same context, no re-spawn cost. Pair kept idle in the meantime.

**Follow-ups created**: none.

---

## [2026-05-11 22:06] Chunk 2: viz panel + data-verify (parallel)

**Goal**: Land the depth panel in `inference.visualize` and verify the SDG dataset's depth/intrinsics format.

**Tasks**: #14, #2.

### #14 viz panel

**Actor** (`actor-viz-panel`):
- `inference.py:374` figsize 1×3 → 1×4 (26, 7).
- `inference.py:436-444` new Panel 4: turbo heatmap with tight-fit colorbar, None-safe fallback to "Depth: N/A" placeholder.
- 1 round.

**Reviewer** (`reviewer-viz-panel`):
- Smoke-tested 4 cases (with-depth + segments, depth=None, constant-value depth, empty-segments). All clean.
- Sign-off: APPROVED.

**Surprise**: `result.depth` defaults to `None`, so pre-depth checkpoints render the N/A panel without crashing. Figsize bump is sized for the colorbar; future extra panels would need another bump.

### #2 data-verify

**Actor** (`actor-data-verify`):
- Read-only investigation across `/home/kaelin/BinPicking/SDG/IS/camera_monocular.py` + sample frame on disk.
- Findings (full transcript in memory): depth is `distance_to_image_plane` (decisive plane-fit ratio 245-265× in favour); JSON schema at `info["camera"]` has `cam_K` (9-elt row-major) and `resolution` ([W, H]); intrinsics randomised per-frame, must load each `__getitem__`; no NaN/inf in depth; all images 1280×1280.
- 1 round.

**Reviewer** (`reviewer-data-verify`):
- Re-ran plane-fit on three patches independently; ratios reproduced.
- Sign-off: APPROVED.

**Decisive findings for downstream tasks**:
1. Task #8 (depth convention conversion) is a NO-OP — closed.
2. Task #5 schema: `info["camera"]["cam_K"]` row-major 9-elt; `info["camera"]["resolution"]` is `[W, H]` (not H,W).
3. `IntrinsicsMLP.image_size` expects `(H, W)`; the JSON gives `[W, H]` — beware the swap in dataset loader.

**Surprises / context**:
- Per-frame intrinsics randomisation rules out caching K across frames.
- Replicator's path-traced rendering hits the full scene; no sky/no-hit invalid pixels.
- Plane-fit verdict was decisive on patches with depth-std < 2 mm; noisier patches give ambiguous ratios.

**Doc bug discovered**: `CLAUDE.md` and `docs/depth_knowledge.md` §11.5 reference a nonexistent `data_writer.py`. Actual is `class MonocularWriter` at `camera_monocular.py:564`. Created Task #16 for the fix.

**Re-fan-out same turn**:
- Task #8 closed as no-op.
- Task #5 (dataset K load) — dispatched to data-verify pair (best SDG context).
- Task #16 (doc fix) — dispatched to viz-panel pair (their files, freshly idle).

**Follow-ups created**: #16 (doc fix).

---

## [2026-05-11 22:07] Chunk 3: anneal-step rescale

**Goal**: Update `attn_mask_annealing_*_steps` in the 1280 yaml so the annealing window stays at the same fractional position under the new dataset/batch.

**Tasks**: #10.

**Actor** (`actor-anneal-tune`):
- `configs/dinov3/occlusion_bp/panoptic/eomt_large_1280.yaml:17-22` — replaced placeholder comment and both step lists.
- New: `start_steps=[0, 118256, 177384, 236512]`, `end_steps=[12000, 177384, 236512, 295640]`.
- Calculation: 640 had 826 batches × 100 epochs = 82,600 total; 1280 has 3304 × 100 = 330,400 total. Exact 4× ratio, zero rounding error preserving fractional positions.
- 1 round.

**Reviewer** (`reviewer-anneal-tune`):
- Independently rebuilt the data module at both batch sizes; reproduced step counts; verified fractional-position preservation.
- Sign-off: APPROVED.

**Surprises / context for future chunks**:
- The 640 yaml's `data.path: /home/kaelin/BinPicking/SDG/IS/Outputs/batch_5` is stale — directory no longer exists. The 640 baseline run from its own yaml is currently un-reproducible without updating the path to `monocular_dataset`.
- The 640 yaml's anneal step values were inherited verbatim from upstream `dinov2/coco/panoptic/eomt_large_1280.yaml` (COCO total ~73,910 steps); they were never derived from the IS dataset size at 640. The new 1280 values therefore preserve the COCO fractional positions, applied to IS — not an IS-native schedule.

**Follow-ups discovered (not actioned)**:
- Optional: update 640 yaml `data.path` → `monocular_dataset` to make the 640 baseline re-runnable.
- Optional: re-derive an IS-native anneal schedule (vs inherited COCO).

Pair parked on standby for `#11` (ckpt-pick) once `#1` closes — they have the yaml + data-pipeline-rebuild context.

---

## [2026-05-11 22:08] Chunk 4: depth-norm fix (Task #1)

**Goal**: Norm all four depth taps via `encoder.backbone.norm` and flip DPT `norm_type` to `"idt"`, matching DA3 mono fidelity.

**Tasks**: #1.

**Actor** (`actor-depth-norm`):
- `models/eomt.py:86-88` — DPT `norm_type` "layer" → "idt" + comment update.
- `models/eomt.py:246-255` — unconditional `tap_x = self.encoder.backbone.norm(x)` for all four taps + comment referencing DA3's `get_intermediate_layers` (vision_transformer.py:384).
- 1 round.

**Reviewer** (`reviewer-depth-norm`):
- Smoke-tested 3 ways: random-tensor forward at 1280×1280, real-image forward, panoptic bit-equality between `depth_taps=(4,11,17,23)` and `depth_taps=None` (max-abs-diff = 0.0 across all panoptic outputs).
- End-to-end smoke through `inference.py` against trained ckpt `eomt/b1k8byj1/checkpoints/epoch=23-step=32400.ckpt` — loads cleanly with strict=False; 53 panoptic segments + finite/positive depth at 800×1200.
- Sign-off: APPROVED.

**Surprises / context for future chunks**:
- `norm_type="idt"` makes `depth_head.norm` a parameter-free `nn.Identity()` — no `depth_head.norm.weight/bias` in state_dict. Silent for fresh runs (strict=False) but relevant for Task #15 (pretrained DA3 DPT load) since DA3 mono also uses idt — compatible.
- Untrained DPT depth output has small std (~5e-3 random, ~1.2e-2 real image). Expected; needs training.
- **Bit-equality invariant**: panoptic outputs (mask_logits, class_logits, occlusion_logits, query_tokens) are byte-identical between depth-on and depth-off variants with shared weights. Future chunks must preserve this — the tap path must remain a pure read of `x`, no in-place ops on `tap_x` that share storage with `x`. `self.encoder.backbone.norm(x)` already returns a fresh tensor; keep it that way.

**Re-fan-out same turn**:
- Task #4 (Wire IntrinsicsMLP into EoMT.forward) → intr-module pair (had #3 context).
- Task #11 (Choose checkpoint for 1280) → depth-norm pair (just touched model architecture).
- Task #9 (yaml + lightning init knobs) is still blocked on #4; anneal-tune pair parked.
- Task #15 (Optional DA3 pretrained DPT load) held — premature pre-training; revisit if Phase 0/1 shows slow convergence.

**Follow-ups created**: none.

---

## [2026-05-11 22:09] Chunk 5: doc-fix (Task #16)

**Goal**: Replace nonexistent-file references (`data_writer.py`) in two project docs with the actual writer location.

**Tasks**: #16.

**Actor** (`actor-viz-panel`):
- `CLAUDE.md:71-76` — Replicator dataset bullet now names `MonocularWriter` + `camera_monocular.py:564`.
- `docs/depth_knowledge.md:666-669` — §11.5 swapped bogus path for the correct class+line.
- 1 round.

**Reviewer** (`reviewer-viz-panel`):
- `rg` sweep: zero hits for `data_writer.py`, one hit each for `MonocularWriter` and `camera_monocular.py:564` across both files. Read prose in context.
- Sign-off: APPROVED.

**Surprises / context**:
- `CLAUDE.md` and `docs/depth_knowledge.md` are still untracked (`??`) in this branch. Working-tree edits only; `git diff` won't show them until staged. Heads-up for whoever lands the eventual commit.

**Re-fan-out**: viz-panel pair → Task #17 (640 yaml data.path fix). Same turn.

**Follow-ups created**: #17.

---

## [2026-05-11 22:10] Chunk 6: intrinsics-aware transforms (Task #6)

**Goal**: Make `datasets/transforms.py` correctly update camera intrinsics under all spatial transforms (resize, crop, flip), preserving the camera matrix consistency end-to-end.

**Tasks**: #6.

**Actor** (`actor-transforms-intr`):
- `datasets/transforms.py:31, 42-46, 117-126, 128-159, 161-218` — replaced torchvision's `T.RandomHorizontalFlip` with manual Bernoulli + `F.horizontal_flip` (so flip can be observed and K updated); added `_filter` special-case for `depth` and `intrinsics`; pad already NaN-fills depth; three new static helpers `_scale_intrinsics(K, sx, sy)`, `_crop_intrinsics(K, ox, oy)`, `_hflip_intrinsics(K, W)`; `forward` now drives flip/scale/pad/crop explicitly so K can update at each step. All gated on `"intrinsics" in target` for backwards compat.
- New `tests/test_transforms_intrinsics.py` (~330 lines, 6 tests, all pass).
- 2 rounds.

**Reviewer** (`reviewer-transforms-intr`):
- Pushed back on iteration 1: torchvision's `ScaleJitter` int-rounds H,W independently, so for non-square inputs `sx ≠ sy` (up to ~0.2%); single-axis scale was wrong. Actor fixed via per-axis sx/sy derivation; new test `test_non_square_scale_recovery_anisotropic` exercises the path (|sx−sy| = 5.2e-4 observed).
- Verified the spec's round-trip independently: starting K → 1.5× resize → crop (100,200) → flip → matches analytical `[[1200, 0, -61], [0, 1200, 340], [0, 0, 1]]`.
- Sign-off: APPROVED.

**Surprises / context for future chunks**:
- **`K` MUST remain a plain `torch.Tensor` end-to-end.** Torchvision v2 leaves plain Tensors untouched through its transforms; if anyone wraps K as a `TVTensor`, the scale-step K update becomes a double-transform. Dataset loader (#5) must pass K as plain Tensor, not a TVTensor subclass.
- The dataset (#5) reads `info["camera"]["cam_K"]` as a 9-element row-major list and `resolution` as `[W, H]`. Reshape to `(3,3)`; no conversion needed at the transforms layer.
- New ctor kwarg `hflip_prob` (default 0.5) is back-compatible; pass 0.0 to disable flips on a particular run.
- Pre-existing uint8/bool indexing deprecation warning from torchvision in `_filter` on `is_crowd` — paper-cut, not in scope, not tracked.

**Re-fan-out**: transforms-intr pair on standby — downstream task #7 needs #4 + #5 + #6 (last one just landed); pair will be re-engaged if integration testing surfaces issues.

**Follow-ups created**: none.

---

## [2026-05-11 22:12] Chunk 7: 640 yaml path fix (Task #17)

**Goal**: Update stale `data.path: batch_5` → `monocular_dataset` so the 640 yaml builds.

**Tasks**: #17.

**Actor** (`actor-viz-panel`):
- `configs/dinov3/occlusion_bp/panoptic/eomt_large_640.yaml:36` — single-line swap.
- 1 round.

**Reviewer** (`reviewer-viz-panel`):
- `rg` confirmed zero `batch_5` hits.
- Re-ran `main.py fit --config eomt_large_640.yaml --print_config` to confirm config resolves cleanly.
- Sign-off: APPROVED.

**⚠ Surprise / important for Task #13 (Phase 1)**:
- The file named `eomt_large_640.yaml` actually has `img_size: [1280, 1280]` and `batch_size: 4`. Pre-existing condition (not from this chunk). So the repo does NOT currently contain a true 640-resolution baseline yaml. Task #13's "PQ vs 640 baseline" framing assumes one exists.
- Follow-up logged: either configure a genuine 640-res yaml or rename the existing one to reflect what it actually is, BEFORE running #13.

**Re-fan-out**: viz-panel pair on standby (no immediate-fit task).

**Follow-ups created**: none formally tasked yet; "640 baseline yaml" issue noted for #13 prep.

---

## [2026-05-11 22:12] Chunk 8: intrinsics in dataset (Task #5)

**Goal**: Load camera intrinsics from `*_scene_info.json` per frame and attach to `target["intrinsics"]` as a plain `[3,3]` float32 tensor.

**Tasks**: #5.

**Actor** (`actor-data-verify`):
- `datasets/iscar_bp.py:14, 23, 26-54, 172` — new `_parse_intrinsics(scene_info)` helper, module-level once-only warning guard, attachment to target before transforms call.
- Primary path: `cam_K` 9-elt row-major → (3,3) float32.
- Fallback: split fx/fy/cx/cy schema → manual construction (defensive; never exercised on current data).
- Fail-safe: missing/malformed → identity K + one warning.
- 1 round.

**Reviewer** (`reviewer-data-verify`):
- Live test on 3304-frame `train/` set; 5 random frames, all (3,3) float32 with varied fx, principal point at image centre.
- Verified intrinsics-aware transform path fires end-to-end (cx 640→639 even at scale_range=[1.0,1.0]).
- Re-ran `tests/test_transforms_intrinsics.py` — 7/7 PASS including legacy-no-intrinsics regression.
- Sign-off: APPROVED.

**Surprises / context for future chunks**:
- **For Task #7**: read `image_size = img.shape[-2:]` (H, W) when calling `IntrinsicsMLP`. **Do NOT** read from `scene_info["camera"]["resolution"]`, which is `[W, H]`. (Reiterated from chunk #2.)
- The intrinsics-aware transforms are already firing — cx shifts observed at scale 1.0 due to RandomCrop. Not a bug, but #7 should add a debug log to verify cam_token sees post-transform principal point.
- Pre-existing `__del__` AttributeError in `datasets/dataset.py:288` (subclass doesn't call `super().__init__()`, so `self.zip` is never set; base `close()` assumes it is). Benign — only fires on shutdown. Deferred follow-up.

**Re-fan-out**: data-verify pair on standby (dataset-domain work for this sprint done).

**Follow-ups discovered (not tasked)**:
- 640-resolution baseline yaml or rename (see chunk #7 notes).
- Cosmetic `__del__` fix in `datasets/dataset.py`.

---

## [2026-05-11 22:17] Chunk 9: wire IntrinsicsMLP into EoMT.forward (Task #4)

**Goal**: Add `use_intrinsics` kwarg, instantiate `intrinsics_mlp`, prepend cam_token after `_pos_embed`, audit all prefix-index slicing sites to account for `extra_prefix`.

**Tasks**: #4.

**Actor** (`actor-intr-module`):
- `models/eomt.py` only — L17 (import), L32-49 (ctor + gated module), L104-141 + L180-212 (`_predict`, `_disable_attn_mask`, `_attn_mask` accept `extra_prefix`), L214-245 (forward signature + pixel H/W stash + gated cam_token prepend), L268-313 (audit-site offsets + `_predict` callers).
- 1 round.

**Reviewer** (`reviewer-intr-module`):
- 9-bucket real-instantiation suite against DINOv3 ViT-L at 224×224, fp32 CPU. Includes:
  - **Bit-equality invariant** verified byte-exact: `forward(x, intrinsics=None)` outputs are byte-identical to pre-#4 behaviour. Means existing checkpoints loaded under `use_intrinsics=True` but called with `intrinsics=None` produce identical panoptic/depth outputs to pre-#4 — useful A/B test capability.
  - Realistic K outputs DO diverge (max-abs mask diff ~1.1, depth diff ~1.6e-3) — proves cam_token influences attention, not dropped by slicing.
  - Param delta = 264,960 exact.
  - State-dict has 4 `intrinsics_mlp.*` keys iff `use_intrinsics=True`.
  - B=2 with different per-batch focals runs clean.
  - **RoPE source-verified** at `transformers/models/dinov3_vit/modeling_dinov3_vit.py:220-241`: `apply_rotary_pos_emb` does `num_prefix_tokens = num_tokens - num_patches`, only rotates patch tokens. cam_token sits in prefix region, absorbed correctly — no rotation needed.
- Sign-off: APPROVED.

**Surprises / context for future chunks**:
- **`LightningModule.forward` at `training/lightning_module.py:175-178` and inference call sites still pass only `x`** — they will continue to work (gracefully degrades to None-intrinsics, no cam_token). Task #7 wires this up end-to-end.
- **For Task #9 (now in flight)**: `_raise_on_incompatible` at `training/lightning_module.py:~984` exempts `depth_head.*` but NOT `intrinsics_mlp.*`. Old EoMT checkpoints will trip the missing-keys check under `use_intrinsics=True`. Surface to anneal-tune pair: add `intrinsics_mlp` to the exemption (done — message sent).
- Forward signature now `forward(x, intrinsics: Optional[Tensor] = None)`. Bit-equality with `intrinsics=None` is byte-exact, not just shape-equal.

**Re-fan-out**:
- intr-module pair queued for #7 (best context fit). #7 now blocked-by #9 due to file conflict on `training/lightning_module.py` and `training/mask_classification_panoptic.py` (both files touched by #9 too). Serialize.
- Anneal-tune pair on #9 with the new `_raise_on_incompatible` exemption added to scope.

**Follow-ups created**: none formal; the `_raise_on_incompatible` exemption added to #9's scope via message.

---

## [2026-05-11 22:18] Chunk 10: 1280 ckpt-pick (Task #11)

**Goal**: Decide which of `eomt_large_640.bin` vs `panoptic_1280.bin` loads cleanest at 1280×1280 with the post-#1 architecture; update yaml `ckpt_path`.

**Tasks**: #11.

**Actor** (`actor-depth-norm`):
- No code changes. Yaml already pointed at `eomt_large_640.bin` (winning choice). Smoke-test-driven verdict only.
- 2 rounds (1 iteration to retract a non-deterministic class-argmax comparison).

**Reviewer** (`reviewer-depth-norm`):
- Tensor-by-tensor ckpt comparison: 437-key schema, differ on 433/437 (different training runs).
- Reviewer-independent reproduction with monkey-patched `nn.Module.load_state_dict` to capture missing/unexpected keys.
- Deterministic-metric verdict:
  - `n_active_queries (≥1% of 320×320 grid)`: 640.bin = 7, panoptic_1280.bin = 4.
  - `top_query_pct`: 73.1% vs 83.7% — 640 has less single-query collapse.
  - mid-band mask sigmoid mass (0.4–0.6): 0.012 vs 0.009.
- Winning ckpt: **eomt_large_640.bin** (already configured at yaml line 12; no edit needed).
- Sign-off: APPROVED.

**Surprises / context for future chunks**:
- **DINOv3 uses RoPE, NO learned pos_embed.** Both 640 and 1280 ckpts have only `patch_embed.patch_embeddings.weight`; no `pos_embed` / `position_embeddings` tensor. So image-size changes do NOT trigger any pos_embed interpolation — encoder is fully shape-flexible via RoPE. This invalidates the original Task #11 premise that "panoptic_1280 wins because its pos-embed was already trained at 80×80".
- **Deterministic-surface map** (load-bearing for future ckpt comparisons at init):
  - Deterministic: `mask_logits_per_layer`, `query_tokens`. Touches: `q.weight`, `mask_head`, `upscale`, `encoder.*` — all loaded.
  - Non-deterministic: `class_logits_per_layer`, `occlusion_logits_per_layer`, `depth`. Touches `class_head`, `occlusion_head`, `depth_head`, `intrinsics_mlp` — re-init'd by `_reinit_missing_modules` without seed control. Either seed before init or inspect deterministic outputs only.
- **panoptic_1280.bin depth mean ≈ 0.85 vs 640.bin ≈ 1.01.** Both feed a fresh DPT, so the gap reflects encoder feature-distribution differences at the four tap layers. May matter for Task #15 if DA3 pretrained DPT expects ImageNet-stat-normalised feature scales — 640's distribution is closer to that expectation.
- Both ckpts trigger the SAME `_reinit_missing_modules` set: `class_head` + `depth_head.*` + `intrinsics_mlp.*` + `occlusion_head.*` (~32M params re-init'd).

**Re-fan-out**: depth-norm pair on standby. #15 remains held; pair is the natural fit if/when training-time signal warrants pretrained-DPT load.

**Follow-ups created**: none.

---

## [2026-05-11 22:22] Chunk 11: yaml + lightning init knobs (Task #9)

**Goal**: Expose `use_intrinsics: true` in the 1280 yaml so Lightning CLI routes it into the EoMT ctor; verify `_raise_on_incompatible` exemption.

**Tasks**: #9.

**Actor** (`actor-anneal-tune`):
- `configs/dinov3/occlusion_bp/panoptic/eomt_large_1280.yaml:36-45` — added `use_intrinsics: true` adjacent to existing `depth_taps`, with 4-line comment.
- Deliberately did NOT touch `mask_classification_panoptic.py` or `main.py` (see below).
- 1 round.

**Reviewer** (`reviewer-anneal-tune`):
- Confirmed `_raise_on_incompatible` exemption for `intrinsics_mlp.*` was already in place at `training/lightning_module.py:985-988` (picked up during #4 close or pre-existed). Unit-tested it directly: `intrinsics_mlp` and `depth_head` exempted, real backbone missing keys still raise.
- Smoke-test (independently re-run): `use_intrinsics=True/False` instantiation toggles the `intrinsics_mlp` attribute correctly; ctor against `eomt_large_640.bin` with `delta_weights: True` does not raise.
- Sign-off: APPROVED.

**Surprises / context for future chunks (load-bearing)**:
- **Task description scope error caught at assignment time**: original #9 spec called for a `use_intrinsics` kwarg on `MaskClassificationPanoptic.__init__`. That's a dead kwarg — Lightning CLI's subclass-mode parser routes `model.init_args.network.init_args.use_intrinsics` directly to `EoMT.__init__`; the LightningModule wrapper sees an already-constructed network. **Pattern to remember**: nested-instantiation flags belong on the leaf class only; do NOT plumb them through the LightningModule wrapper. This caveat applies to any future yaml flag.
- Lightning CLI's subclass-mode parser handles arbitrary nested `init_args` automatically. New EoMT kwargs need only: (a) the kwarg on the class, (b) the value in the yaml. No `link_arguments` or panoptic-class plumbing unless the value also lives data-side.
- The actor's smoke-test pattern (write toggled yaml to `/tmp/`, re-instantiate via `LightningCLI(..., run=False)`) is a good template; exercises the real CLI parser, not the constructor directly.

**Re-fan-out**:
- intr-module pair → Task #7 (Pass intrinsics through Lightning + inference). Was queued; now active.
- anneal-tune pair on standby.

**Follow-up addendum (2026-05-11 22:25)**: reviewer-anneal-tune surfaced a finding from the actor's addendum work — the real `checkpoints/eomt_large_640.bin` predates the **occlusion head** (added in commit 00d760c) as well as depth_head/intrinsics_mlp. `occlusion_head` is NOT exempt from the missing-keys raise. Currently OK because the 1280 yaml has `delta_weights: true` which absorbs missing keys silently. **But if Task #12 (Phase 0 freeze training) chooses to use `delta_weights: false` to lock loaded weights as frozen, it will hit a fresh `ValueError: Missing keys: ['network.occlusion_head.*']`.** Mitigations when it arises: (a) add `occlusion_head` to the exemption list (matches established pattern), or (b) require `delta_weights: false` configs to use `panoptic_1280.bin` instead of `eomt_large_640.bin`. Must include this in the #12 dispatch brief.

The actor's Branch C addendum also strengthened the #9 verification: previous smoke covered only `delta_weights: true` (where the exemption is inert through `_add_state_dicts`); new test goes through the `delta_weights: false` path where the exemption is actually load-bearing. 10/10 PASS + negative control.

**Follow-ups created**: none formal; #12 brief will incorporate the occlusion_head note.

---

## [2026-05-11 22:29] Chunk 12: intrinsics through Lightning + inference (Task #7)

**Goal**: Plumb intrinsics from dataloader → training_step → `model.forward(x, intrinsics=K)`; mirror for eval_step + inference path; apply letterbox K-scaling.

**Tasks**: #7.

**Actor** (`actor-intr-module`):
- `training/lightning_module.py:L175-178, L180-198` — forward signature + training_step intrinsics stack + model call.
- `training/mask_classification_panoptic.py:L106-135` — eval_step intrinsics stack with per-image letterbox K-scaling via `scale_img_size_instance_panoptic`.
- `inference.py:L145-199, L313-345` — `_preprocess` accepts and letterbox-scales K; `__call__` accepts intrinsics kwarg, passes through.
- 1 round.

**Reviewer** (`reviewer-intr-module`):
- 15-check parity suite, real DINOv3 ViT-L at 128/224 px, B=1 and B=2, fp32 CPU.
- Key checks:
  - `LightningModule.forward(imgs)` byte-identical to `forward(imgs, intrinsics=None)` across all 5 return values.
  - With explicit K, mask logits diverge (max-abs ~1.69) — intrinsics genuinely influence model.
  - End-to-end `training_step` runs with synthetic batch including `target["intrinsics"]`; loss=38.35 finite.
  - Letterbox K-scaling math byte-exact: `(100,200)` → `(224,224)` padded, K scaled by 1.12 byte-exact.
  - Stale-state hygiene in `_preprocess`: `_scaled_intrinsics = None` reset each call.
  - End-to-end `EoMTInference(img, intrinsics=K)` produces well-formed result; depth differs from None case by 5.3e-3.
- Sign-off: APPROVED.

**Surprises / context for future chunks**:
- Eval-step letterbox K-scaling is currently a no-op (val_transforms has `scale_range=(1.0, 1.0)`) but actor proactively wired it general — harden-for-future.
- `scale_img_size_instance_panoptic` rounds via `round()`, so sub-pixel scale factors introduce sub-percent error. Moot for current Replicator path (1280→1280 exact).
- **Pre-existing bug surfaced during testing**: `lightning_module.py:211-212` does `zip(..., None)` if `enable_occlusion=False` → `TypeError: 'NoneType' object is not iterable`. BP config has `enable_occlusion=True` so this won't bite current path; flagged as follow-up only if no-occlusion config is ever needed.

**Re-fan-out**:
- intr-module pair → Task #12 (Phase 0 freeze training) immediately. They have the freshest LightningModule + intrinsics-pipeline context.
- All other code-level tasks closed. Only training tasks (#12, #13) and optional #15 remain.

**Follow-ups created**: (low-priority) `lightning_module.py:211-212` zip-None guard if no-occlusion config is ever needed. Not tasked.

---

## [2026-05-11 22:46] Session paused — resume handoff

**State**: All 17 task-list code-level chunks closed. Only #12 (Phase 0 dry-run) is in_progress with owner="human"; #13 (Phase 1 training) pending; #15 (Optional DA3 pretrained DPT) held.

**Why paused**: human course-corrected away from agent-led training-run execution to save iteration tokens. All 12 teammates received `shutdown_request`; team-task list and `eomt-depth` team config persist.

**On resume — first orchestrator actions**:

1. **Re-read this `memory.md` end-to-end** to recover full sprint context. Chunks 1-12 cover the depth + intrinsics integration; chunks 6, 9, 10 contain critical surprises (anisotropic scale, MaskClassificationPanoptic kwarg dead-end, occlusion_head not-exempt risk).
2. **Check task list state** (`TaskList`) — should still show 14 completed, #12 in_progress (human-owned), #13/#15 pending.
3. **Write the Phase 0 dry-run script** the human asked for: standalone `scripts/phase0_dryrun.py` that builds the LightningModule directly (NOT via `main.py` LightningCLI complexity), freezes all params except `depth_head.*` + `intrinsics_mlp.*`, runs ~200 train steps with `limit_train_batches=200, limit_val_batches=0`, logs to wandb under `project: "eomt", name: "phase0_dryrun"`, and exposes per-step grad-norm metrics per submodule + dead/leaky param counters. Reference signatures: `MaskClassificationPanoptic.__init__` (see `training/mask_classification_panoptic.py`), `ReplicatorDataModule.__init__` (see `datasets/iscar_bp.py`), and the yaml at `configs/dinov3/occlusion_bp/panoptic/eomt_large_1280.yaml` as the canonical kwarg set.
4. **Carry-forward heads-ups for the script** (load-bearing):
   - Use `delta_weights=True` + `checkpoints/eomt_large_640.bin` (matches existing yaml; avoids the occlusion_head-not-exempt issue documented in chunks #9, #11 addendum).
   - At 1280×1280 batch_size=1 expect ~2-4 GiB activation memory per forward; OOM is a real risk.
   - Acceptance is diagnostic, NOT convergence: no NaN, no dead grads on `depth_head.*` / `intrinsics_mlp.*`, no leaky grads elsewhere, no OOM, populated wandb curves.
   - The user is explicit: data has flawed scenes; do NOT gate on `loss_depth < 0.1`.

**Files modified during this sprint** (per `git status` heuristic):
- `models/eomt.py` (chunks #1, #4)
- `models/intrinsics_mlp.py` (NEW, chunk #3)
- `training/lightning_module.py` (chunks #4 prep, #7, #9 verification)
- `training/mask_classification_panoptic.py` (chunk #7)
- `training/depth_loss.py` (pre-sprint)
- `datasets/iscar_bp.py` (chunks #5, #13 setup)
- `datasets/transforms.py` (chunk #6)
- `tests/test_transforms_intrinsics.py` (NEW, chunk #6)
- `inference.py` (chunks #7, #14)
- `configs/dinov3/occlusion_bp/panoptic/eomt_large_1280.yaml` (chunks #9, #10)
- `configs/dinov3/occlusion_bp/panoptic/eomt_large_640.yaml` (chunk #17)
- `CLAUDE.md` (chunk #16)
- `docs/depth_knowledge.md` (chunk #16)

**Team disposition**: 12 teammates received shutdown_request at 22:46. Team `eomt-depth` config + shared task list still on disk at `~/.claude/teams/eomt-depth/` — re-spawning teammates is cheap (no fresh team setup needed) if any chunk needs re-dispatch. Recommended: skip pair-dispatch for the Phase 0 / Phase 1 training runs (single GPU, single-shot work, human-supervised).

---
