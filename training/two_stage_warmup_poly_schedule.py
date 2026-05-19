# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from torch.optim.lr_scheduler import LRScheduler


class TwoStageWarmupPolySchedule(LRScheduler):
    def __init__(
        self,
        optimizer,
        num_backbone_params: int,
        warmup_steps: tuple[int, int],
        total_steps: int,
        poly_power: float,
        backbone_freeze_steps: int = 0,
        last_epoch=-1,
    ):
        self.num_backbone_params = num_backbone_params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.poly_power = poly_power
        # Stage-A hard-freeze window: backbone lr=0 until this step, THEN it
        # warms up over vit_warmup. Head schedule is unaffected. Default 0 ⇒
        # legacy behaviour (backbone freeze == non_vit_warmup).
        self.backbone_freeze_steps = backbone_freeze_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        non_vit_warmup, vit_warmup = self.warmup_steps
        # Backbone starts ramping only after the explicit freeze window if set;
        # otherwise it falls back to the head's non_vit_warmup boundary.
        bb_freeze = max(self.backbone_freeze_steps, non_vit_warmup)
        for i, base_lr in enumerate(self.base_lrs):
            if i >= self.num_backbone_params:
                if non_vit_warmup > 0 and step < non_vit_warmup:
                    lr = base_lr * (step / non_vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup)
                    lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power
            else:
                if step < bb_freeze:
                    lr = 0
                elif step < bb_freeze + vit_warmup:
                    lr = base_lr * ((step - bb_freeze) / vit_warmup)
                else:
                    adjusted = max(0, step - bb_freeze - vit_warmup)
                    max_steps = max(1, self.total_steps - bb_freeze - vit_warmup)
                    lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power

            lrs.append(lr)
        return lrs
    
