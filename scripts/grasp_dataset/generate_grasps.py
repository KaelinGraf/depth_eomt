"""CLI driver: run the grasp-dataset pipeline over many frames in parallel.

Examples:
    # Smoke (process 5 monocular val frames)
    python -m scripts.grasp_dataset.generate_grasps \\
        --dataset monocular --split val --max-frames 5 --jobs 4

    # Full monocular train
    python -m scripts.grasp_dataset.generate_grasps \\
        --dataset monocular --split train --jobs 16

    # SL val
    python -m scripts.grasp_dataset.generate_grasps \\
        --dataset sl --split val --jobs 8

After both datasets+splits have been processed, run with --emit-graspgen-index to
write the splits/{train,val}.txt + map_uuid_to_path.json that GraspGen's loader
expects.
"""
from __future__ import annotations
import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

# Make package importable when run as `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.grasp_dataset.orchestrator import process_frame  # noqa: E402


# Dataset roots (hard-coded to current SDG output locations; adjust if moved)
MONO_ROOT = Path("/home/kaelin/BinPicking/SDG/IS/Outputs/monocular_dataset")
SL_ROOT = Path("/home/kaelin/BinPicking/SDG/IS/Outputs/sl_dataset")
OUTPUT_ROOT = Path("/home/kaelin/BinPicking/SDG/IS/Outputs/grasp_dataset")


def _dataset_root(dataset: str) -> Path:
    return {"monocular": MONO_ROOT, "sl": SL_ROOT}[dataset]


def _list_frames(dataset: str, split: str) -> list[Path]:
    return sorted([
        p for p in (_dataset_root(dataset) / split).iterdir()
        if p.is_dir() and p.name.startswith("frame_")
    ])


def _worker(args):
    frame_dir, dataset, split, output_root, n_grasps, min_vis, td_cut, td_knee, seed = args
    try:
        from scripts.grasp_dataset.orchestrator import process_frame
        return process_frame(
            frame_dir=frame_dir,
            dataset=dataset, split=split, output_root=output_root,
            n_grasps_per_part=n_grasps,
            min_visibility=min_vis,
            topdown_cutoff_deg=td_cut,
            topdown_knee_deg=td_knee,
            voxel_max_count=0,
            skip_existing=True,
            seed=seed,
            verbose=False,
        )
    except Exception as e:
        return {"frame_dir": str(frame_dir), "error": str(e)}


def emit_graspgen_index(output_root: Path):
    """Walk grasp_dataset/{dataset}/{split}/ for all *.grasps.json, emit:
        - splits/train.txt (UUIDs)
        - splits/val.txt
        - map_uuid_to_path.json (uuid → relative path)
    Following GraspGen's expected layout. UUID = filename stem (without
    .grasps.json suffix).
    """
    uuid_map = {}
    splits = {"train": [], "val": []}
    for dataset in ("monocular", "sl"):
        for split in ("train", "val"):
            d = output_root / dataset / split
            if not d.exists():
                continue
            for p in sorted(d.glob("*.grasps.json")):
                uuid = f"{dataset}__{split}__{p.stem.replace('.grasps', '')}"
                rel = p.relative_to(output_root)
                uuid_map[uuid] = str(rel)
                splits[split].append(uuid)
    (output_root / "splits").mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        (output_root / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n")
    with open(output_root / "map_uuid_to_path.json", "w") as f:
        json.dump(uuid_map, f, indent=2)
    print(f"Wrote splits/{{train,val}}.txt: {len(splits['train'])} train / {len(splits['val'])} val UUIDs")
    print(f"Wrote map_uuid_to_path.json with {len(uuid_map)} entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["monocular", "sl"], required=False)
    ap.add_argument("--split", choices=["train", "val"], required=False)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--n-grasps", type=int, default=1000)
    ap.add_argument("--min-visibility", type=float, default=0.3)
    ap.add_argument("--topdown-cutoff-deg", type=float, default=80.0)
    ap.add_argument("--topdown-knee-deg", type=float, default=60.0)
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2),
                    help="parallel processes (default: cpu_count - 2)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="limit to first N frames (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit-graspgen-index", action="store_true",
                    help="After processing (or on its own), build splits + uuid map")
    args = ap.parse_args()

    if args.emit_graspgen_index and not args.dataset:
        emit_graspgen_index(args.output_root)
        return

    if not args.dataset or not args.split:
        ap.error("--dataset and --split required unless --emit-graspgen-index alone")

    frames = _list_frames(args.dataset, args.split)
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    print(f"[{args.dataset}/{args.split}] {len(frames)} frames; jobs={args.jobs}")

    job_args = [
        (f, args.dataset, args.split, args.output_root, args.n_grasps,
         args.min_visibility, args.topdown_cutoff_deg, args.topdown_knee_deg,
         args.seed)
        for f in frames
    ]

    t0 = time.time()
    completed = 0
    total_valid = 0
    total_grasps = 0
    total_parts = 0
    errors = 0

    if args.jobs == 1:
        for ja in job_args:
            r = _worker(ja)
            completed += 1
            if "error" in r:
                errors += 1
            else:
                total_parts += r.get("n_parts", 0)
                for pp in r.get("per_part", []):
                    total_valid += pp.get("n_valid", 0)
                    total_grasps += pp.get("n_grasps", 0)
            if completed % 10 == 0 or completed == len(job_args):
                _print_progress(completed, len(job_args), t0, total_parts, total_valid, total_grasps, errors)
    else:
        with mp.Pool(processes=args.jobs) as pool:
            for r in pool.imap_unordered(_worker, job_args):
                completed += 1
                if "error" in r:
                    errors += 1
                else:
                    total_parts += r.get("n_parts", 0)
                    for pp in r.get("per_part", []):
                        total_valid += pp.get("n_valid", 0)
                        total_grasps += pp.get("n_grasps", 0)
                if completed % 10 == 0 or completed == len(job_args):
                    _print_progress(completed, len(job_args), t0, total_parts, total_valid, total_grasps, errors)

    print(f"\nDone {args.dataset}/{args.split}: {completed} frames, {total_parts} parts, "
          f"{total_valid}/{total_grasps} valid grasps ({100*total_valid/max(total_grasps,1):.1f}%), "
          f"{errors} errors, {time.time()-t0:.0f}s")

    if args.emit_graspgen_index:
        emit_graspgen_index(args.output_root)


def _print_progress(completed, total, t0, parts, valid, grasps, errors):
    elapsed = time.time() - t0
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    print(f"  [{completed}/{total}] {rate:.1f} fr/s · parts={parts} · valid={valid}/{grasps} "
          f"({100*valid/max(grasps,1):.1f}%) · errors={errors} · "
          f"elapsed={elapsed:.0f}s · eta={eta:.0f}s")


if __name__ == "__main__":
    main()
