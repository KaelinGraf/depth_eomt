# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
    sample_point,
    sigmoid_cross_entropy_loss,
    dice_loss,
)


class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        occlusion_coefficient: float = None,
        use_area_weighting: bool = False,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.occlusion_coefficient = occlusion_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        self.use_area_weighting = use_area_weighting
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
        occlusion_queries_logits: Optional[torch.Tensor] = None,
    ):
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]
        
        if occlusion_queries_logits is not None:
            occlusion_labels = [target["occlusion"].float() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)
        if occlusion_queries_logits is not None:
            loss_occlusion = self.loss_occlusion(occlusion_queries_logits, occlusion_labels, indices,class_labels)
            return {**loss_masks, **loss_classes, **loss_occlusion}

        return {**loss_masks, **loss_classes}

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        # Base code copied from HF Mask2FormerLoss
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)

        pred_masks = masks_queries_logits[src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        num_masks_clamped = torch.clamp(num_masks_tensor / world_size, min=1).item()

        if self.use_area_weighting:
            # Calculate target area per mask (summing over height and width)
            mask_areas = target_masks.reshape(target_masks.shape[0], -1).sum(dim=-1)
            mean_area = mask_areas.mean()
            # Area ratio weight clamped between 1.0 (large masks) and 25.0 (tiny masks)
            weights = torch.clamp((mean_area + 1e-5) / (mask_areas + 1e-5), min=1.0, max=25.0)

            # To prevent geometric loss explosion, we must treat this as a true Weighted Average.
            weight_sum = weights.sum() if len(weights) > 0 else 1.0
            weight_sum_tensor = torch.as_tensor(
                weight_sum, dtype=torch.float, device=masks_queries_logits.device
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(weight_sum_tensor)
                # the world_size is already computed above
            weight_sum_clamped = torch.clamp(weight_sum_tensor / world_size, min=1).item()

            # --- Weighted BCE Loss ---
            bce_loss_instance = torch.nn.functional.binary_cross_entropy_with_logits(
                point_logits, point_labels.float(), reduction="none"
            )
            # Average over the points for each mask, then multiply by inverse area weight
            bce_loss_instance = bce_loss_instance.mean(dim=1)
            weighted_bce_loss = (bce_loss_instance * weights).sum() / weight_sum_clamped

            # --- Weighted Dice Loss ---
            probs = point_logits.sigmoid()
            numerator = 2 * (probs * point_labels).sum(dim=-1)
            denominator = probs.sum(dim=-1) + point_labels.sum(dim=-1)
            dice_loss_instance = 1 - (numerator + 1) / (denominator + 1)
            weighted_dice_loss = (dice_loss_instance * weights).sum() / weight_sum_clamped

            loss_masks_dict = {
                "loss_mask": weighted_bce_loss,
                "loss_dice": weighted_dice_loss,
            }
        else:
            loss_masks_dict = {
                "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks_clamped),
                "loss_dice": dice_loss(point_logits, point_labels, num_masks_clamped),
            }

        del pred_masks
        del target_masks
        return loss_masks_dict
    
    
    def loss_occlusion(self,occlusion_queries_logits, occlusion_labels, indices,class_labels):
        """
        Compute the occlusion loss using smoothed l1 loss.
        args:
            occlusion_queries_logits: Tensor of shape (batch_size, num_queries) containing the predicted occlusion logits for each query.
            occlusion_labels: List of tensors, where each tensor contains the ground truth occlusion labels (0-1) for the corresponding target.
            indices: List of tuples containing the matched indices between predictions and targets.
        """
        n_batches,n_queries = occlusion_queries_logits.shape
        src_occlusions, target_occlusions = [],[]
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            thing_mask = class_labels[batch_idx][tgt_idx] > 0
            thing_mask = thing_mask.to(src_idx.device)
            if thing_mask.any():
                src_occlusions.append(occlusion_queries_logits[batch_idx, src_idx[thing_mask]])
                target_occlusions.append(occlusion_labels[batch_idx][tgt_idx[thing_mask]])
        
        src_occlusions = torch.cat(src_occlusions)
        target_occlusions = torch.cat(target_occlusions)
        loss_occlusion = torch.nn.functional.smooth_l1_loss(torch.sigmoid(src_occlusions), target_occlusions)
        return {"loss_occlusion": loss_occlusion}

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            elif "occlusion" in loss_key:
                weighted_loss = loss * self.occlusion_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore
