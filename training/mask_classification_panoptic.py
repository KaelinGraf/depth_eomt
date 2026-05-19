# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics import MeanMetric

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule
from training.depth_loss import loss_depth_silog


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
        backbone_freeze_steps: int = 0,
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 3.0,
        dice_coefficient: float = 7.0,
        class_coefficient: float = 2.0,
        occlusion_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        depth_grad_coefficient: float = 0.0,
        normal_coefficient: float = 0.0,
        normal_grad_coefficient: float = 0.0,
        consistency_coefficient: float = 0.0,
        aux_loss_warmup_steps: int = 0,
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
            backbone_freeze_steps=backbone_freeze_steps,
            aux_loss_warmup_steps=aux_loss_warmup_steps,
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
            depth_coefficient=depth_coefficient,
            depth_grad_coefficient=depth_grad_coefficient,
            normal_coefficient=normal_coefficient,
            normal_grad_coefficient=normal_grad_coefficient,
            consistency_coefficient=consistency_coefficient,
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

        # Depth regression metrics — only when the network carries a depth head.
        # Created after super().__init__() (which loads the ckpt), so these keys
        # are never seen by load_state_dict — same pattern as init_metrics_panoptic.
        if getattr(self.network, "depth_taps", None) is not None:
            self.init_metrics_depth()

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

        # Mirror the per-image resize that `resize_and_pad_imgs_instance_panoptic`
        # applies to `imgs` onto each K, so the intrinsics reach the model in
        # the same coordinate frame as `transformed_imgs`. Top-left placement
        # in the padded canvas, so no cx/cy shift is needed.
        intrinsics = None
        if targets and "intrinsics" in targets[0]:
            scaled = []
            for tgt, size in zip(targets, img_sizes):
                K = tgt["intrinsics"].clone()
                new_h, new_w = self.scale_img_size_instance_panoptic(size)
                sx = new_w / size[-1]
                sy = new_h / size[-2]
                K[0, 0] = K[0, 0] * sx
                K[0, 2] = K[0, 2] * sx
                K[1, 1] = K[1, 1] * sy
                K[1, 2] = K[1, 2] * sy
                scaled.append(K)
            intrinsics = torch.stack(scaled, dim=0).to(
                transformed_imgs.device, transformed_imgs.dtype
            )

        (
            mask_logits_per_layer,
            class_logits_per_layer,
            occlusion_logits_per_layer,
            depth_pred,
            _normal_pred,
            _,
        ) = self(transformed_imgs, intrinsics=intrinsics)

        # Depth validation metrics (AbsRel / RMSE / delta1 / SI-Log). Must run
        # before `targets` is reassigned to the per-pixel panoptic form below,
        # while it is still the list of per-image dicts carrying "depth".
        if depth_pred is not None and hasattr(self, "depth_metrics"):
            self.update_metrics_depth(depth_pred, targets)

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
        if hasattr(self, "depth_metrics"):
            self._on_eval_epoch_end_depth("val")

    def on_validation_end(self):
        self._on_eval_end_panoptic("val")

    # ------------------------------------------------------------------
    # Depth regression metrics (monocular DA3-style depth head)
    # ------------------------------------------------------------------
    def init_metrics_depth(self):
        # MeanMetric accumulators — each fed a per-image scalar, averaged over
        # the val set. ModuleDict so they move with the module and sync across
        # ranks via torchmetrics' own compute().
        self.depth_metrics = nn.ModuleDict(
            {
                "absrel": MeanMetric(),
                "rmse": MeanMetric(),
                "delta1": MeanMetric(),
                "silog": MeanMetric(),
            }
        )

    @torch.compiler.disable
    def update_metrics_depth(
        self,
        depth_pred: torch.Tensor,
        targets: List[dict],
        d_max: float = 10.0,
        eps: float = 1e-6,
    ):
        """Accumulate per-image depth metrics over valid GT pixels.

        `depth_pred` is [B, 1, H, W] (post-exp, metres); `targets` is the list
        of per-image dicts, each optionally carrying `depth` as [1, h, w].
        """
        for bi, tgt in enumerate(targets):
            if "depth" not in tgt:
                continue
            d_gt = tgt["depth"].to(depth_pred.device).float()  # [1, h, w]
            d_pred = depth_pred[bi : bi + 1].float()           # [1, 1, H, W]
            # Model output may differ from GT resolution if the eval transform
            # letterboxed; align by resampling pred onto the GT grid.
            if d_pred.shape[-2:] != d_gt.shape[-2:]:
                d_pred = F.interpolate(
                    d_pred, size=d_gt.shape[-2:], mode="bilinear", align_corners=False
                )
            d_pred = d_pred[0]                                 # [1, h, w]

            valid = torch.isfinite(d_gt) & (d_gt > 0) & (d_gt < d_max)
            if not valid.any():
                continue
            dp = d_pred[valid].clamp_min(eps)
            dg = d_gt[valid].clamp_min(eps)
            ratio = torch.maximum(dp / dg, dg / dp)

            self.depth_metrics["absrel"].update((torch.abs(dp - dg) / dg).mean())
            self.depth_metrics["rmse"].update(torch.sqrt(((dp - dg) ** 2).mean()))
            self.depth_metrics["delta1"].update((ratio < 1.25).float().mean())
            self.depth_metrics["silog"].update(loss_depth_silog(d_pred, d_gt))

    def _on_eval_epoch_end_depth(self, log_prefix):
        for name, metric in self.depth_metrics.items():
            self.log(f"metrics/{log_prefix}_depth_{name}", metric.compute())
            metric.reset()
