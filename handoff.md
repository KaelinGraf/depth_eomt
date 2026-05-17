# EoMT + Grasp Dataset/Fine-tune Pipeline — Handoff

**Snapshot date:** 2026-05-18
**Branch:** `grasp-dataset-pipeline` (EOMT repo); changes also touch `BinPicking/SDG/IS/` (no commits there).

## What this repo does (two parallel workstreams)

This repo is a fork of EoMT (Encoder-only Mask Transformer) extended for ISCAR bin-picking, on a shared DINOv3 ViT-L/16 backbone. Two parallel workstreams currently live here:

1. **EoMT panoptic segmentation + depth + (paused) normals.** Training enters via `main.py` → PyTorch Lightning. Best ckpt: PQ All ~76, val_depth_silog ~0.04, val_depth_delta1 ~99.4 % on 14 k synthetic Replicator frames. See `docs/depth_knowledge.md` and `docs/eomt_architecture.md` for the architectural ground truth — **read `docs/depth_knowledge.md` first** for any depth-related task.

2. **Grasp dataset generation + GraspGen fine-tune pipeline.** Lives under `scripts/grasp_dataset/` (generation) and `scripts/grasp_train/` (fine-tune wrappers). Operates on the Replicator outputs in `/home/kaelin/BinPicking/SDG/IS/Outputs/{monocular,sl}_dataset/` and writes to `/home/kaelin/BinPicking/SDG/IS/Outputs/grasp_dataset/`.

Most recent active work is workstream 2. Read `/home/kaelin/BinPicking/SDG/IS/Outputs/grasp_dataset/handoff.md` for the dataset-side handoff, and `/home/kaelin/bp_runtime/ml_deps/GraspGen/handoff.md` for the fine-tune-side handoff.

## Active state — grasp dataset generation (as of 2026-05-18)

A background sweep is running across both datasets and both splits. Status:

| Split | Status | Output |
|---|---|---|
| `sl/val` | done | 2,799 grasp JSONs |
| `sl/train` | done | 25,601 grasp JSONs |
| `monocular/val` | done | 28,935 grasp JSONs |
| `monocular/train` | running (~35 % at snapshot) | 94k → ~225 k expected total |

Background task id `bz87a0z3a` (see `/tmp/gen_all.log`, `/tmp/gen_mono_train.log`). Rate ~0.2 frames/s wall-clock across 20 workers. ETA ~15 h from snapshot.

After `monocular/train` finishes, the orchestrator runs `--emit-graspgen-index` automatically (writes `splits/{train,val}.txt` + `map_uuid_to_path.json` under `grasp_dataset/`). After THAT, re-run `prepare_data.py` (cheap, idempotent) so the GraspGen-formatted dir picks up the new data.

## File map — workstream 2 (grasp)

### Generation (`scripts/grasp_dataset/`)

| File | What |
|---|---|
| `orchestrator.py` | `process_frame(frame_dir, ...)` → emits one JSON per visible part with grasps + labels + scene metadata |
| `scene_io.py` | Loads scene_info.json + instance mask + depth/points. `SceneObject` + `Scene` dataclasses. Catalog lookup via `/home/kaelin/BinPicking/SDG/IS/Config/objects.json`. |
| `mesh_io.py` | USD → trimesh. **Asset-aware xform handling** — meshes with their own scale op use only the mesh's local transform (Isaac Sim drops intermediate Xform-only parent scales); legacy meshes (no own scale) use full ancestor chain. Critical for YCB-style assets like `_19_pitcher_base` whose visible mesh would otherwise be 5× too small. |
| `antipodal.py` | CPU antipodal sampler via `mesh.ray.intersects_location`. Approach bias sampled on perpendicular circle. |
| `filters.py` | `topdown_soft_filter` (knee 60°, cutoff 80°), `wrong_side_filter` (gripper must approach into scene). |
| `collision.py` | Builds trimesh `CollisionManager` from scene meshes; per-grasp `in_collision_single`. |
| `gripper.py` | Robotiq 2F-140 collision geometry (palm + 2 fingers + arms) parameterised by aperture. Convention: base origin, Z=approach, X=closing. |
| `generate_grasps.py` | Multiprocess CLI. `--dataset {monocular,sl} --split {train,val} --jobs N`. Emits `splits/{train,val}.txt` + `map_uuid_to_path.json` via `--emit-graspgen-index`. |
| `visualize.py` | Renders grasp overlays. `--all-valid` mode for density visualisation; default mode draws 10 valid + 10 invalid balanced. |

### Fine-tune wrappers (`scripts/grasp_train/`)

| File | What |
|---|---|
| `prepare_data.py` | Walks `grasp_dataset/{mono,sl}/{train,val}/*.grasps.json`, exports each unique USD to OBJ (cached), **merges all per-frame grasps per object into one consolidated record** (the loader is per-object, not per-frame), emits 85/15 object-disjoint splits. Output at `/home/kaelin/bp_runtime/ml_deps/eomt/grasp_finetune_data/`. |
| `finetune_dis.sh` | Wraps `train_graspgen.py` for the discriminator. Uses pretrained Robotiq 2F-140 ckpt from `models/checkpoints/`. Env var `TARGET_EPOCH_OVERRIDE=<N>` for short validation runs. |
| `finetune_gen.sh` | Same template for the diffusion generator. |

## Critical lessons baked in (do NOT re-discover)

1. **SDG `canonical_extent` is the post-physics AABB, not the asset's intrinsic shape.** It varies wildly per instance of the same asset (a flat plate lying flat vs standing on its edge gives very different AABBs). Per-axis fitting to canonical_extent corrupts mesh shape. Current code uses scale_m2c (uniform per-instance) + OOBB when present, falls back gracefully.

2. **Isaac Sim drops intermediate Xform-level scales for YCB-style assets at runtime.** A pitcher with `/Root/Geometry: scale 0.15` AND `/Root/Geometry/.../Mesh: scale 0.01` renders at extent_raw × 0.01 (only mesh-local scale), NOT × 0.15 × 0.01. `mesh_io.py` handles both cases via `has_own_scale` check.

3. **Bin USDs aren't in `objects.json`** (those paths are stale). They live at `assets/bins/<bin>/<bin>_inst.usd`. `orchestrator._match_bin()` matches the scene's bin to a disk USD by sorted canonical_extent (within 5 cm sum-|Δ|). For SL data with canonical_extent=None, the bin is missing from collision — fixed when SDG is regenerated (patches already applied to `/home/kaelin/BinPicking/SDG/IS/camera_monocular.py` and `scene_builder.py`, see those files).

4. **Outlier parts re-tagged `background` by `update_semantic_labels_for_outliers` still have their `usd_filepath`.** Treat them as real meshes, not boxes (~50 % more valid grasps on cluttered SL scenes).

5. **SDG scene_info dispatch is on `dataset_version`, not `dataset_name`** — `v2` → `load_object_grasp_datapoint_objaverse` (right one for our JSON layout). `v1` → assumes `category_model_scale` naming and crashes on ours.

6. **GraspGen discriminator_ratio must be 7 elements** (the last two are on-policy slots, set to 0 in our case). 5-element form raises `ConfigIndexError`.

7. **`data.preload_dataset=True` (GraspGen default) OOMs on our cache size** (~50 GB for 411 objects × 7 redundancy). Use `preload_dataset=False` — disk reads are fine.

8. **Objects with <300 positive grasps break the loader** (can't fill the requested 150 positives per object). 19 such objects existed at snapshot; ad-hoc filtered out of splits via inline Python — **fold into `prepare_data.py` as `--min-positives 300` flag when next iterating on the script**.

9. **All work runs under `bp_runtime/ros_venv`** for grasp training (has hydra, h5py, grasp_gen editable, torch 2.8 cu129 — Blackwell-ready). The `eomt` conda env lacks those deps and is for EoMT training/inference only.

## Pending follow-ups

- **Finish `monocular/train` data gen** (running, ~15 h ETA at snapshot).
- **Re-run `prepare_data.py`** when above finishes (idempotent; picks up new data). Add the `--min-positives 300` filter into the script itself rather than the ad-hoc fix used at snapshot.
- **Validate the discriminator smoke test** (TARGET_EPOCH_OVERRIDE=9755). Last attempt was blocked by CPU contention with the 20 data-gen workers — load avg 39 on 24 cores. After data gen completes, smoke test will run unimpeded.
- **Launch real discriminator fine-tune** (target +200 epochs from pretrained 9750), then generator (+500 from pretrained 20000).
- **Sim-to-real fine-tune phase** when real Zivid 2+ MR60 data lands (week of 2026-05-19). Lower LR (1e-6), possibly freeze PointNet encoder briefly.
- **SDG regeneration** (deferred): the patches in `BinPicking/SDG/IS/camera_monocular.py` and `scene_builder.py` add OOBB recording + buried-part enumeration + full-scene-USD export per iteration. None take effect until SDG re-runs. Substantial cost (~days), so batch with other improvements.

## Where everything lives (paths)

```
EOMT repo (this repo):               /home/kaelin/bp_runtime/ml_deps/eomt/
  scripts/grasp_dataset/             grasp generation pipeline
  scripts/grasp_train/               GraspGen fine-tune wrappers
  inference_outputs/grasp_vis*/      visualisations (pushed to GitHub)

SDG output:                          /home/kaelin/BinPicking/SDG/IS/Outputs/
  monocular_dataset/{train,val}/     14k+1.5k Replicator frames (mono camera)
  sl_dataset/{train,val}/            5.4k+0.6k frames (structured-light camera)
  grasp_dataset/{mono,sl}/{train,val}/  generated grasp JSONs
  grasp_dataset/handoff.md           dataset-side handoff

GraspGen-formatted data:             /home/kaelin/bp_runtime/ml_deps/eomt/grasp_finetune_data/
  object_dataset/<name>.obj          one OBJ per unique asset (USD-exported)
  grasp_data/robotiq_2f_140/         consolidated per-object grasp JSONs
  splits/robotiq_2f_140/             object-disjoint train/val splits

Fine-tune outputs:                   /home/kaelin/bp_runtime/ml_deps/eomt/grasp_finetune_results/
  logs/robotiq_2f_140_{dis,gen}_finetune/   TensorBoard + console_log.txt
  cache/                             h5 caches built by GraspGen's loader

GraspGen repo:                       /home/kaelin/bp_runtime/ml_deps/GraspGen/
  models/checkpoints/                pretrained Robotiq weights (already local)
  scripts/train_graspgen.py          shared training entry for both heads
  handoff.md                         fine-tune-side handoff
```

## Quickstart on a fresh session

```bash
# 1. Check data gen status
tail -3 /tmp/gen_all.log

# 2. If sweep is done, re-run prepare_data (idempotent; cached USD exports)
source ~/bp_runtime/setup_env.sh
python /home/kaelin/bp_runtime/ml_deps/eomt/scripts/grasp_train/prepare_data.py

# 3. Apply <300-positive filter (TODO: fold into prepare_data.py)
#    See "Pending follow-ups" above.

# 4. Smoke-test discriminator (5 epochs over pretrained)
TARGET_EPOCH_OVERRIDE=9755 bash /home/kaelin/bp_runtime/ml_deps/eomt/scripts/grasp_train/finetune_dis.sh

# 5. Real fine-tune
bash /home/kaelin/bp_runtime/ml_deps/eomt/scripts/grasp_train/finetune_dis.sh   # ~1-2 days
bash /home/kaelin/bp_runtime/ml_deps/eomt/scripts/grasp_train/finetune_gen.sh   # ~2-3 days
```

## Git state

Branch `grasp-dataset-pipeline` is ahead of `master` by all the work in this handoff. Last commit at snapshot: `a4d7be7`. PR has not been opened yet — pipeline still iterating.

Visualisations of grasps land in `inference_outputs/grasp_vis*` directories and are pushed to GitHub for review.
