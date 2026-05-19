# ---------------------------------------------------------------
# © 2026 ISCAR Bin-Picking. Licensed under the MIT License.
# ---------------------------------------------------------------
"""Stage-A backbone freeze callback.

Sets `requires_grad=False` on every parameter under
`network.encoder.backbone.*` at fit start, then flips it back to
True at `global_step == backbone_freeze_steps`. Complements the
scheduler's lr=0 plumbing by also skipping backbone backward and
keeping Adam state clean.
"""

import logging
from lightning.pytorch import Callback, LightningModule, Trainer


class BackboneFreezeCallback(Callback):
    def __init__(self, freeze_steps: int):
        super().__init__()
        self.freeze_steps = int(freeze_steps)
        self._frozen = False

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        if self.freeze_steps <= 0:
            return
        if trainer.global_step >= self.freeze_steps:
            return  # resumed past the freeze window
        n_frozen = 0
        for name, p in pl_module.named_parameters():
            if name.startswith("network.encoder.backbone."):
                p.requires_grad = False
                n_frozen += p.numel()
        self._frozen = True
        logging.info(
            "BackboneFreezeCallback: froze %d backbone params until step %d",
            n_frozen,
            self.freeze_steps,
        )

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int):
        if not self._frozen:
            return
        if trainer.global_step >= self.freeze_steps:
            n_unfrozen = 0
            for name, p in pl_module.named_parameters():
                if name.startswith("network.encoder.backbone."):
                    p.requires_grad = True
                    n_unfrozen += p.numel()
            self._frozen = False
            logging.info(
                "BackboneFreezeCallback: unfroze %d backbone params at step %d",
                n_unfrozen,
                trainer.global_step,
            )
