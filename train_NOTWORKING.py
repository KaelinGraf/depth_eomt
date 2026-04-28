import torch
import lightning as L 
from datasets.iscar_bp import ReplicatorDataset
from training.mask_classification_panoptic import MaskClassificationPanoptic
from models.eomt import EoMT
from models.vit import ViT
import torch.nn.functional as F

def main():
    cpkt_path = "checkpoints/eomt_large_640.bin"
    data_module = ReplicatorDataset(data_dir="/home/kaelin/BinPicking/SDG/IS/Outputs/batch_6", transforms=None)
    backbone = ViT(img_size=(640, 640), patch_size=16, backbone_name="dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd", ckpt_path="dino_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    network = EoMT(backbone, num_classes=2, num_q=200, num_blocks=4, masked_attn_enabled=True, enable_occlusion=True)
    model = MaskClassificationPanoptic(
        network, 
        img_size=(640, 640), 
        num_classes=2, 
        stuff_classes=[0], 
        attn_mask_annealing_enabled=True,
        attn_mask_annealing_start_steps =[0, 29564, 44346, 59128],
        attn_mask_annealing_end_steps = [3000, 44346, 59128, 73910],
        lr = 2e-4,
        llrd_l2_enabled=False,
        warmup_steps = [2000, 3000],
        )
        
        

        
        
    

        
