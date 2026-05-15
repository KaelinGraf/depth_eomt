# CLAUDE.md

This repository extends EoMT (Encoder-only Mask Transformer) for the
ISCAR bin-picking task: panoptic segmentation + per-instance occlusion
prediction + (in progress) DA3-style monocular depth, all on a shared
DINOv3 ViT-L/16 backbone. Training enters via `main.py` →
PyTorch-Lightning; inference via `inference.py`.

## Required reading before depth-related work

Any agent making changes that touch the depth path **must** read
`docs/depth_knowledge.md` first. It is the canonical, code-grounded
reference for what DA3 monocular actually does and what the EoMT
integration must therefore implement, including:

- The exact tap mechanism (all four taps normed by `encoder.norm`,
  patch-only, `patch_start_idx=0`).
- The correct claim that **DA3 monocular consumes no camera
  intrinsics** (mention of an "intrinsics MLP" describes the
  multi-view CameraEnc, not mono).
- Known deviations of the current EoMT code from DA3 fidelity (notably
  that only tap 23 is currently normed — see §11.1 of
  `docs/depth_knowledge.md`).

`docs/depth_knowledge.md` supersedes `docs/da3_monocular.md`. The
older file is kept for diff history but contains errors catalogued in
§13 of the new file. Do not cite the old file.

## Other docs, by purpose

- `docs/eomt_architecture.md` — as-built audit of the EoMT model at
  640×640 / panoptic+occlusion / DINOv3. Ground truth for the panoptic
  side of the model.
- `docs/depth_integration_plan.md` — the implementation plan for
  640→1280 + DA3 depth integration. Most of §12's items 1-14 are
  already done; verify state via `git status` and the modified files
  before re-doing work.
- `docs/depth_knowledge.md` — DA3 reference + open issues against
  current code. **Read first** for any depth-related task.

## Decisions encoded in this branch

- **Depth taps: `[4, 11, 17, 23]`** (DA3 default, ported one-to-one to
  DINOv3 ViT-L). Yaml-overridable. Standard DPT/MiDaS recipe; preserves
  the option of loading DA3 pretrained DPT weights. Discussion of
  alternatives is in `docs/depth_knowledge.md` §10.
- **All four depth taps must be normed** by `encoder.backbone.norm`
  before DPT consumes them, matching DA3's `get_intermediate_layers`.
  This is currently only done for tap 23 — see `docs/depth_knowledge.md`
  §11.1 and the open task list.
- **Intrinsics conditioning: yes, UniDepth-V1-style.** A small
  intrinsics MLP (`Linear(6,256) → GELU → Linear(256, embed_dim)`) maps
  `[fx/W, fy/H, cx/W, cy/H, log(W), log(H)]` to a single cam_token
  prepended to the ViT sequence. This is a deliberate departure from
  `da3mono-large` (which has no intrinsics input); rationale is that
  monocular metric depth benefits broadly from FOV conditioning and
  bin-picking deployment may span multiple cameras. Gated by
  `use_intrinsics` yaml flag, defaulting on.
- **DA3 multi-view `CameraEnc` is NOT borrowed.** It includes
  extrinsics (meaningless for mono) and a 4-block transformer trunk
  (overkill). Our intrinsics MLP is a stripped-down equivalent.

## Conventions

- Configs live under `configs/dinov3/occlusion_bp/panoptic/`. The
  active 1280-resolution + depth yaml is `eomt_large_1280.yaml`.
- `Depth-Anything-3/` is a vendored upstream reference repo. It is
  gitignored and should not be modified — copy code into
  `models/dpt.py` / `models/dpt_utils.py` instead (already done for the
  current DPT).
- Replicator dataset path:
  `/home/kaelin/BinPicking/SDG/IS/Outputs/monocular_dataset/`. Ground
  truth depth field is `depth.npy` per frame, written by
  `MonocularWriter` in
  `/home/kaelin/BinPicking/SDG/IS/camera_monocular.py:564` (verify the
  writer emits this filename if changing data sources).
