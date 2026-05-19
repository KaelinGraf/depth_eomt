import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.dataset import Dataset
from PIL import Image
import numpy as np
import json
import logging
from torchvision import tv_tensors

import random

CLASS_MAPPING = {
    "background": 0,
    "part": 1}

_intrinsics_warned = False


def _parse_intrinsics(scene_info) -> torch.Tensor:
    """Extract a (3, 3) float32 K matrix from the Replicator scene-info JSON.

    Primary schema: ``scene_info["camera"]["cam_K"]`` row-major 9-elt list.
    Fallback schema: ``scene_info["camera"]["fx"|"fy"|"cx"|"cy"]`` scalars.
    If neither is present (e.g. list-shaped JSON), return identity and warn
    once per process.
    """
    global _intrinsics_warned
    cam = scene_info.get("camera") if isinstance(scene_info, dict) else None
    if isinstance(cam, dict):
        cam_K = cam.get("cam_K")
        if cam_K is not None:
            return torch.tensor(cam_K, dtype=torch.float32).reshape(3, 3)
        fx, fy = cam.get("fx"), cam.get("fy")
        cx, cy = cam.get("cx"), cam.get("cy")
        if None not in (fx, fy, cx, cy):
            return torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
    if not _intrinsics_warned:
        logging.warning(
            "ReplicatorDataset: no camera intrinsics found in scene_info; "
            "using identity K. Downstream intrinsics-conditioned heads will "
            "see a meaningless cam_token."
        )
        _intrinsics_warned = True
    return torch.eye(3, dtype=torch.float32)


class ReplicatorDataset(Dataset):
    """
    Custom Map-style dataset to handle unzipped frame_N/ directory structures.
    """
    def __init__(self, data_dir: Path, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

        # Find all frame folders recursively (e.g., batch_6/frame_0, ...).
        all_dirs = sorted(d for d in self.data_dir.rglob("frame_*") if d.is_dir())
        # Drop incomplete frames: interrupted SDG runs leave partial or empty
        # frame_* dirs. A usable frame needs rgb + instance_raw + scene_info;
        # filtering here keeps __len__ honest and avoids per-item retries.
        self.frame_dirs = [d for d in all_dirs if self._is_complete(d)]

        n_dropped = len(all_dirs) - len(self.frame_dirs)
        if n_dropped:
            logging.warning(
                "ReplicatorDataset: skipped %d/%d incomplete frame dirs in %s",
                n_dropped, len(all_dirs), self.data_dir,
            )
        if len(self.frame_dirs) == 0:
            raise RuntimeError(f"No complete frame directories found in {self.data_dir}")

    @staticmethod
    def _is_complete(frame_dir: Path) -> bool:
        return (
            (frame_dir / "rgb.png").is_file()
            and any(frame_dir.glob("*instance_raw*"))
            and any(frame_dir.glob("*scene_info*.json"))
        )

    def __len__(self):
        return len(self.frame_dirs)

    def __getitem__(self, idx):
        frame_dir = self.frame_dirs[idx]
        contents = list(frame_dir.iterdir())
        
        # 1. Load RGB Image
        img_path = frame_dir / "rgb.png"
        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))

        # 1b. Load depth (metres, float32, NaN for invalid pixels). Wrap as
        #    tv_tensors.Mask so the spatial transforms (resize, crop, flip,
        #    pad) treat it as a follow-the-image tensor. Mask accepts float
        #    dtypes; if a future torchvision restricts this, swap for a
        #    custom TVTensor subclass.
        depth_path = frame_dir / "depth.npy"
        depth = None
        if depth_path.exists():
            depth_arr = np.load(depth_path).astype(np.float32)
            if depth_arr.ndim == 2:
                depth_arr = depth_arr[None, ...]   # add channel dim → [1, H, W]
            depth = tv_tensors.Mask(torch.from_numpy(depth_arr))

        # 1c. Load per-pixel surface normals (Replicator's `normals` annotator,
        #     [H,W,4] with channel 3 = constant alpha=1.0). Replicator emits
        #     OpenGL-style camera-space normals (X right, Y up, Z toward
        #     camera); flip (1, -1, -1) here to match our OpenCV unprojection
        #     convention (X right, Y down, Z fwd into scene). After the flip,
        #     a frontal-facing surface has Z ≈ -1 (normal points toward
        #     camera, outward from surface). Verified against GT depth via
        #     `training/depth_loss.py:_selftest_normal_convention` — median
        #     cosine > 0.94 across val frames.
        normals_path = frame_dir / "normals.npy"
        normals = None
        if normals_path.exists():
            n_arr = np.load(normals_path)[..., :3].astype(np.float32)      # [H, W, 3]
            n_arr = n_arr * np.array([1.0, -1.0, -1.0], dtype=np.float32)  # OpenGL→OpenCV
            n_arr = np.transpose(n_arr, (2, 0, 1))                          # → [3, H, W]
            normals = tv_tensors.Mask(torch.from_numpy(n_arr))

        # 2. Load 16-bit Grayscale Mask. Raw-mask filenames carry a Replicator
        # prefix, so match by pattern. `_is_complete` (ctor) guarantees a hit;
        # the explicit None-check turns any future miss into a clear error
        # instead of an UnboundLocalError.
        raw_mask_path = next(
            (f for f in contents
             if "instance_raw" in f.name
             and f.suffix.lower() in (".png", ".jpg", ".jpeg")),
            None,
        )
        if raw_mask_path is None:
            raise FileNotFoundError(f"No instance_raw image in {frame_dir}")
        raw_mask = np.array(Image.open(raw_mask_path), dtype=np.int32)

        # 3. Load Scene Info
        json_path = next(
            (f for f in contents if "scene_info" in f.name and f.suffix == ".json"),
            None,
        )
        if json_path is None:
            raise FileNotFoundError(f"No scene_info JSON in {frame_dir}")
        with open(json_path, "r") as f:
            scene_info = json.load(f)
            
        # Replicator usually stores instances in a list
        objects = scene_info if isinstance(scene_info, list) else scene_info.get("objects", [])

        masks, labels, occlusions, is_crowd = [], [], [], []
        
        for obj in objects:
            seg_id = obj.get("segmentation_id")
            
            raw_cls_id = obj.get("class", 0) #can be '['class_id']', in this case strip 
            if isinstance(raw_cls_id, str):
                raw_cls_id = raw_cls_id.strip('[]')
            if "background" in raw_cls_id.lower():
                raw_cls_id = "background"
            else:
                raw_cls_id = "part" # For simplicity, we can treat all non-background objects as "part".
            cls_id = CLASS_MAPPING.get(raw_cls_id, -1) # Default to background if not found
            if cls_id == -1:
                raise ValueError(f"Class ID {raw_cls_id} not found in CLASS_MAPPING. Please update the mapping.")

            instance_mask = (raw_mask == seg_id)
            
            # Skip objects completely off-screen
            if not instance_mask.any():
                continue
                
            masks.append(instance_mask)
            labels.append(cls_id)
            occlusions.append(obj.get("visibility_ratio", 0.0))
            is_crowd.append(False) # Synthetic datasets rarely have "crowd" regions
        has_foreground = any(cls > 0 for cls in labels)
        
        if not has_foreground:
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]

        # 4. Format into Tensors
        if len(masks) > 0:
            masks = tv_tensors.Mask(torch.from_numpy(np.stack(masks)).bool())
            labels = torch.tensor(labels, dtype=torch.long)
            occlusions = torch.tensor(occlusions, dtype=torch.float32)
            is_crowd = torch.tensor(is_crowd, dtype=torch.bool)
        else:
            # Handle empty frames safely
            h, w = raw_mask.shape
            masks = torch.empty((0, h, w), dtype=torch.bool)
            labels, occlusions, is_crowd = (
                torch.empty((0,), dtype=torch.long), 
                torch.empty((0,), dtype=torch.float32), 
                torch.empty((0,), dtype=torch.bool)
            )

        target = {
            "masks": masks,
            "labels": labels,
            "occlusion": occlusions,
            "is_crowd": is_crowd  # Required by the base collate/metric functions
        }
        if depth is not None:
            target["depth"] = depth
        if normals is not None:
            target["normals"] = normals

        target["intrinsics"] = _parse_intrinsics(scene_info)

        # 5. Apply the library's native Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if isinstance(img, torch.Tensor):
            img = img.contiguous()

        return img, target


class ReplicatorDataModule(LightningDataModule):
    def __init__(
        self,
        path: str, # Path to the folder containing 'train' and 'val' subfolders
        stuff_classes: list[int],
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1280, 1280),
        num_classes: int = 2,
        color_jitter_enabled=False,
        sensor_noise_enabled=True,
        blur_enabled=True,
        scale_range=(0.1, 2.0),
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        # We reuse the exact same transforms as COCO
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            sensor_noise_enabled=sensor_noise_enabled,
            blur_enabled=blur_enabled,
            scale_range=scale_range,
        )

        self.val_transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=False,
            sensor_noise_enabled=False,
            blur_enabled=False,
            scale_range=(1.0, 1.0),
        )

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        # Instead of the base Dataset, we instantiate our custom ReplicatorDataset
        # Assuming you split your batches into "train" and "val" folders inside the main path
        
        if stage == "fit" or stage is None:
            self.train_dataset = ReplicatorDataset(
                data_dir=Path(self.path) / "train",
                transforms=self.transforms
            )
            self.val_dataset = ReplicatorDataset(
                data_dir=Path(self.path) / "val",
                transforms=self.val_transforms
            )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate, # Inherited from base LightningDataModule
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate, # Inherited from base LightningDataModule
            **self.dataloader_kwargs,
        )

class _NoDepthNormalsWrapper:
    """Wraps a Dataset to strip 'depth' and 'normals' keys from its target
    dicts. Needed when mixing synth (has depth/normals) with real (doesn't);
    training_step's `if "depth" in targets[0]` shortcut crashes on the first
    mixed batch where the leading sample is synth — by stripping synth's
    extras, the check uniformly fails and the optional aux-loss path is
    skipped. depth/normals heads still forward (enable_depth/enable_normals
    are model-side flags) but with coefficients = 0 they contribute nothing."""

    def __init__(self, ds):
        self._ds = ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        img, target = self._ds[idx]
        target.pop("depth", None)
        target.pop("normals", None)
        return img, target


class MixedReplicatorDataModule(ReplicatorDataModule):
    """Run-2 data module: ConcatDataset(real_train, synth_train) with a
    WeightedRandomSampler so each epoch's expected sample count is
    `real_per_epoch` real + `synth_per_epoch` synth (i.e. 2:1 synth:real
    when called as 45 / 90). Val stays real-only to track in-domain quality."""

    def __init__(self, real_path, synth_path,
                 real_per_epoch: int = 45, synth_per_epoch: int = 90, **kwargs):
        # Parent uses `real_path` as the "primary" — its val/ subdir becomes val.
        super().__init__(path=real_path, **kwargs)
        self.synth_path = synth_path
        self.real_per_epoch = int(real_per_epoch)
        self.synth_per_epoch = int(synth_per_epoch)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            real = ReplicatorDataset(
                data_dir=Path(self.path) / "train", transforms=self.transforms)
            synth = ReplicatorDataset(
                data_dir=Path(self.synth_path) / "train", transforms=self.transforms)
            # Strip depth/normals from synth so target dict shape matches real
            # (real frames don't have those files — see _NoDepthNormalsWrapper).
            synth = _NoDepthNormalsWrapper(synth)
            self.train_dataset = ConcatDataset([real, synth])
            n_r, n_s = len(real), len(synth)
            # Item weights: real items at weight 1, synth items at weight
            # (synth_per_epoch/real_per_epoch) * (n_r/n_s) so that after the
            # sampler normalises, expected per-epoch draw counts hit the spec.
            ratio = (self.synth_per_epoch / max(self.real_per_epoch, 1)) * (n_r / max(n_s, 1))
            weights = [1.0] * n_r + [ratio] * n_s
            self._sampler = WeightedRandomSampler(
                weights,
                num_samples=self.real_per_epoch + self.synth_per_epoch,
                replacement=True,
            )
            bs = self.dataloader_kwargs["batch_size"]
            logging.info(
                "MixedReplicatorDataModule: real=%d, synth=%d, "
                "per-epoch target = %d real + %d synth (%d total = %d steps@bs=%d)",
                n_r, n_s, self.real_per_epoch, self.synth_per_epoch,
                self.real_per_epoch + self.synth_per_epoch,
                (self.real_per_epoch + self.synth_per_epoch) // max(bs, 1),
                bs,
            )
            self.val_dataset = ReplicatorDataset(
                data_dir=Path(self.path) / "val", transforms=self.val_transforms)
        return self

    def train_dataloader(self):
        # Mirror ReplicatorDataModule.train_dataloader but pass sampler instead
        # of shuffle (mutually exclusive in DataLoader). dataloader_kwargs gives
        # us batch_size + num_workers + pin_memory + persistent_workers.
        return DataLoader(
            self.train_dataset,
            sampler=self._sampler,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )
