import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.dataset import Dataset
from PIL import Image
import numpy as np
import json
from torchvision import tv_tensors 

import random

CLASS_MAPPING = {
    "background": 0, 
    "part": 1}




class ReplicatorDataset(Dataset):
    """
    Custom Map-style dataset to handle unzipped frame_N/ directory structures.
    """
    def __init__(self, data_dir: Path, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        
        # Find all frame folders recursively (e.g., batch_6/frame_0, batch_6/frame_1...)
        self.frame_dirs = sorted([d for d in self.data_dir.rglob("frame_*") if d.is_dir()])
        
        if len(self.frame_dirs) == 0:
            raise RuntimeError(f"No frame directories found in {self.data_dir}")

    def __len__(self):
        return len(self.frame_dirs)

    def __getitem__(self, idx):
        frame_dir = self.frame_dirs[idx]
        contents = list(frame_dir.iterdir())
        
        # 1. Load RGB Image
        img_path = frame_dir / "rgb.png"
        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
    
        
        # 2. Load 16-bit Grayscale Mask
        #raw mask paths contain the replicator number, so we need to find the file that matches the pattern
        for file in contents:
            if "instance_raw" in file.name and file.suffix in [".png", ".jpg", ".jpeg"]:
                raw_mask_path = file
                break
        raw_mask = np.array(Image.open(raw_mask_path), dtype=np.int32)
        
        # 3. Load Scene Info
        for file in contents:
            if "scene_info" in file.name and file.suffix in [".json"]:
                json_path = file
                break
        # json_path = frame_dir / "Replicator_scene_info.json"
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
        img_size: tuple[int, int] = (640, 640),
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