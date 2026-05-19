"""Freeze everything except the EOMT prediction heads (and optionally the last
N transformer blocks) for parameter-efficient fine-tuning. Backbone LLRD in
the LightningModule's optimizer config keeps the unfrozen blocks at a low LR
(lr × lr_mult × llrd^(depth - i)), so the unfrozen blocks adapt slowly."""

import logging

from lightning.pytorch import Callback, LightningModule, Trainer


class HeadOnlyFreezeCallback(Callback):
    """Keep `network.class_head.*` and `network.mask_head.*` trainable.
    If `last_n_blocks > 0`, also keep the last N transformer blocks of
    `network.encoder.backbone.blocks` trainable."""

    HEAD_PREFIXES = ("network.class_head.", "network.mask_head.")

    def __init__(self, last_n_blocks: int = 0):
        super().__init__()
        self.last_n_blocks = int(last_n_blocks)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        keep_prefixes = list(self.HEAD_PREFIXES)
        if self.last_n_blocks > 0:
            n_blocks = len(pl_module.network.encoder.backbone.blocks)
            first_kept = n_blocks - self.last_n_blocks
            for i in range(first_kept, n_blocks):
                keep_prefixes.append(f"network.encoder.backbone.blocks.{i}.")
        kept, frozen = 0, 0
        for name, p in pl_module.named_parameters():
            if any(name.startswith(pref) for pref in keep_prefixes):
                p.requires_grad = True
                kept += p.numel()
            else:
                p.requires_grad = False
                frozen += p.numel()
        logging.info(
            "HeadOnlyFreezeCallback(last_n_blocks=%d): %d trainable, %d frozen",
            self.last_n_blocks, kept, frozen,
        )
