# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationPanoptic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 25600,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.55,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 3.0,
        dice_coefficient: float = 7.0,
        class_coefficient: float = 2.0,
        occlusion_coefficient: float = 1.0,
        mask_thresh: float = 0.5,
        overlap_thresh: float = 0.7,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        use_area_weighting: bool = False,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = stuff_classes

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            occlusion_coefficient=occlusion_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
            use_area_weighting=use_area_weighting,
        )

        thing_classes = [i for i in range(num_classes) if i not in stuff_classes]
        self.init_metrics_panoptic(
            thing_classes,
            stuff_classes,
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1,
        )

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch
        targets_original = targets
        target_occlusion_scores = [target.get("occlusion", None) for target in targets]

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer, occlusion_logits_per_layer, _ = self(transformed_imgs)

        is_crowds = [target["is_crowd"] for target in targets]
        targets = self.to_per_pixel_targets_panoptic(targets)

        for i, (mask_logits, class_logits, occlusion_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer, occlusion_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )
            preds_and_info = self.to_per_pixel_preds_panoptic( #image segment info is tuple of (segment_id, class_id, occlusion_score)
                mask_logits,
                class_logits,
                self.stuff_classes,
                self.mask_thresh,
                self.overlap_thresh,
                occlusion_logits=occlusion_logits
            )
            preds, image_segment_info = zip(*preds_and_info)
            preds = list(preds)
            image_segment_info = list(image_segment_info)
            self.update_metrics_panoptic(preds, targets, is_crowds, i, image_segment_info=image_segment_info,target_occlusion_scores=target_occlusion_scores)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_panoptic("val")

    def on_validation_end(self):
        self._on_eval_end_panoptic("val")
