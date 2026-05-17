"""Convert our per-(frame, instance) grasp JSONs into the layout GraspGen's
ObjectPickDataset expects, with an 85/15 object-disjoint split.

Outputs (under --out-root):
    object_dataset/<catalog_name>.obj           # one OBJ per unique asset (USD-exported)
    grasp_data/robotiq_2f_140/<catalog_name>.grasps.json
                                                # all grasps for the asset across
                                                # every frame it appeared in, merged
    splits/robotiq_2f_140/{train,val,valid}.txt # object-disjoint, line = "<name>.obj"

Merging rationale: GraspGen's loader treats each row of train.txt as one object;
it renders the OBJ from a random viewpoint per __getitem__ and reads the grasps
from the matching JSON. Scene context (R_m2c, occlusion, neighbour parts) is
dropped by that loader, so per-frame separation buys nothing — we concatenate
all (transform, label) pairs across frames into one consolidated record.
"""
from __future__ import annotations
import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

# Make our project package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.grasp_dataset.mesh_io import load_usd_mesh  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GRIPPER = "robotiq_2f_140"
VAL_FRACTION = 0.15
SEED = 42


def gather_grasps(grasp_root: Path) -> dict[str, list[Path]]:
    """{catalog_name -> [json paths]} across both datasets and splits."""
    per_object: dict[str, list[Path]] = defaultdict(list)
    n_files = 0
    for jp in grasp_root.glob("**/*.grasps.json"):
        n_files += 1
        # Use scene_metadata.object_name (== catalog_name) as the merge key.
        with open(jp) as f:
            d = json.load(f)
        cname = d.get("scene_metadata", {}).get("object_name")
        if not cname:
            continue
        per_object[cname].append(jp)
    log.info(f"Scanned {n_files} grasp JSONs → {len(per_object)} unique objects")
    return per_object


def export_obj(usd_path: str, out_path: Path) -> bool:
    """Export the asset USD to OBJ. Returns True on success."""
    if out_path.exists():
        return True
    try:
        mesh = load_usd_mesh(usd_path, target_extent=None)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_path)
        return True
    except Exception as e:
        log.warning(f"USD→OBJ failed for {usd_path}: {e}")
        return False


def merge_grasps(json_paths: list[Path], obj_rel_path: str, source_count: int) -> dict:
    """Concatenate transforms + object_in_gripper across every per-frame JSON of
    one asset into a single GraspGen-compatible record."""
    transforms, widths, labels, reasons = [], [], [], []
    gripper_block = None
    for jp in json_paths:
        with open(jp) as f:
            d = json.load(f)
        g = d["grasps"]
        transforms.extend(g["transforms"])
        widths.extend(g.get("widths", [0.0] * len(g["object_in_gripper"])))
        labels.extend(g["object_in_gripper"])
        reasons.extend(g.get("filter_reasons", [""] * len(g["object_in_gripper"])))
        if gripper_block is None:
            gripper_block = d.get("gripper")
    return {
        "object": {"file": obj_rel_path, "scale": 1.0},
        "gripper": gripper_block,
        "grasps": {
            "transforms": transforms,
            "widths": widths,
            "object_in_gripper": labels,
            "filter_reasons": reasons,
        },
        "source_frame_count": source_count,
        "n_grasps_total": len(labels),
        "n_grasps_valid": int(sum(labels)),
    }


def write_splits(split_dir: Path, train_names: list[str], val_names: list[str]):
    split_dir.mkdir(parents=True, exist_ok=True)
    train_lines = "\n".join(f"{n}.obj" for n in sorted(train_names)) + "\n"
    val_lines = "\n".join(f"{n}.obj" for n in sorted(val_names)) + "\n"
    (split_dir / "train.txt").write_text(train_lines)
    # GraspGen variously expects val.txt OR valid.txt; emit both so either works.
    (split_dir / "val.txt").write_text(val_lines)
    (split_dir / "valid.txt").write_text(val_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp-root", type=Path,
                    default=Path("/home/kaelin/BinPicking/SDG/IS/Outputs/grasp_dataset"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("/home/kaelin/bp_runtime/ml_deps/eomt/grasp_finetune_data"))
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be done without writing files")
    args = ap.parse_args()

    obj_dir = args.out_root / "object_dataset"
    grasp_out_dir = args.out_root / "grasp_data" / GRIPPER
    split_dir = args.out_root / "splits" / GRIPPER

    per_object = gather_grasps(args.grasp_root)
    if not per_object:
        log.error(f"No grasp JSONs found under {args.grasp_root}")
        sys.exit(1)

    # Convert each unique USD → OBJ once + merge per-frame grasps.
    converted, skipped = [], []
    for cname, json_paths in sorted(per_object.items()):
        with open(json_paths[0]) as f:
            usd_path = json.load(f)["object"]["file"]
        obj_path = obj_dir / f"{cname}.obj"
        if args.dry_run:
            log.info(f"[dry] would convert {cname} ({len(json_paths)} frames)")
            converted.append(cname)
            continue
        if not export_obj(usd_path, obj_path):
            skipped.append(cname)
            continue
        merged = merge_grasps(json_paths, f"{cname}.obj", len(json_paths))
        grasp_out_dir.mkdir(parents=True, exist_ok=True)
        with open(grasp_out_dir / f"{cname}.grasps.json", "w") as f:
            json.dump(merged, f)
        converted.append(cname)
        if len(converted) % 50 == 0:
            log.info(f"  {len(converted)}/{len(per_object)} converted ({len(skipped)} skipped)")

    if skipped:
        log.warning(f"Skipped {len(skipped)} objects (USD load failed): {skipped[:5]}...")

    # Object-disjoint split.
    rng = random.Random(args.seed)
    objects = converted[:]
    rng.shuffle(objects)
    n_val = max(1, int(round(len(objects) * args.val_fraction)))
    val_names = objects[:n_val]
    train_names = objects[n_val:]

    if not args.dry_run:
        write_splits(split_dir, train_names, val_names)
    log.info(f"Done: {len(train_names)} train objects, {len(val_names)} val objects (object-disjoint)")
    log.info(f"  object_dataset: {obj_dir}")
    log.info(f"  grasp_data:     {grasp_out_dir}")
    log.info(f"  splits:         {split_dir}")


if __name__ == "__main__":
    main()
