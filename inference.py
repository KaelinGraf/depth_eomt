"""
EoMT Inference Module
=====================
Standalone inference class for the EoMT panoptic segmentation model with
occlusion prediction. Loads a trained checkpoint and provides a simple
__call__ interface for single-image inference.

Usage:
    from inference import EoMTInference

    model = EoMTInference("path/to/checkpoint.ckpt")
    result = model("path/to/image.png")

    # result contains:
    #   - panoptic_mask: [H, W] int, segment IDs per pixel
    #   - class_mask:    [H, W] int, class ID per pixel
    #   - segments:      list of dicts with segment_id, class_id, occlusion_score, area
    #   - query_tokens:  [num_q, embed_dim] raw query embeddings for diffusion conditioning
    #   - raw_masks:     [N, H, W] per-instance binary masks

    model.visualize(result, "path/to/image.png")
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EoMTResult:
    """Container for inference results."""
    panoptic_mask: np.ndarray      # [H, W] segment IDs
    class_mask: np.ndarray         # [H, W] class IDs
    segments: List[Dict[str, Any]] # per-segment info
    query_tokens: np.ndarray       # [num_q, embed_dim] for diffusion conditioning
    raw_masks: np.ndarray          # [N, H, W] per-instance binary masks
    scores: np.ndarray             # [N] confidence scores
    depth: Optional[np.ndarray] = None   # [H, W] float32 metres, or None if depth head disabled
    normal: Optional[np.ndarray] = None  # [3, H, W] unit normals (OpenCV cam coords), or None


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class EoMTInference:
    """
    Standalone inference wrapper for EoMT panoptic segmentation + occlusion.

    Loads the full Lightning checkpoint, extracts model weights, and provides
    a clean __call__ interface with no Lightning/Trainer dependencies at
    inference time.
    """

    CLASS_NAMES = {0: "background", 1: "part"}  # Adjust for your dataset

    def __init__(
        self,
        ckpt_path: str = "/home/kaelin/BinPicking/eomt/eomt/b1k8byj1/checkpoints/epoch=23-step=32400.ckpt",
        device: str = "cuda",
        mask_thresh: float = 0.01,
        overlap_thresh: float = 0.1,
        stuff_classes: List[int] = None,
        num_classes: int = 2,
        img_size: tuple = (640, 640),
    ):
        """
        Args:
            ckpt_path: Path to Lightning .ckpt file.
            device: "cuda" or "cpu".
            mask_thresh: Confidence threshold for keeping predictions.
            overlap_thresh: Minimum mask overlap ratio to keep.
            stuff_classes: List of "stuff" class IDs (background).
            num_classes: Total number of classes.
            img_size: Model input resolution (H, W).
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = stuff_classes or [0]
        self.num_classes = num_classes
        self.img_size = img_size

        # Load model from checkpoint
        self.model = self._load_from_checkpoint(ckpt_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"EoMT loaded from {ckpt_path} on {self.device}")
        print(f"  Classes: {self.CLASS_NAMES}")
        print(f"  Input size: {self.img_size}")
        print(f"  Occlusion: {self.model.enable_occlusion}")

    def _load_from_checkpoint(self, ckpt_path: str):
        """Load EoMT network from a Lightning checkpoint."""
        from models.eomt import EoMT
        from models.vit import ViT

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        # Extract network weights (strip 'network.' prefix from Lightning module)
        network_state = {
            k.replace("network.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("network.")
        }

        # Determine config from checkpoint
        has_occlusion = any("occlusion_head" in k for k in network_state)
        num_q = network_state["q.weight"].shape[0]

        # Build network
        encoder = ViT(img_size=self.img_size, backbone_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
        model = EoMT(
            encoder=encoder,
            num_q=num_q,
            num_classes=self.num_classes,
            enable_occlusion=has_occlusion,
        )

        # Load weights
        incompatible = model.load_state_dict(network_state, strict=False)
        if incompatible.unexpected_keys:
            print(f"  Warning: unexpected keys: {incompatible.unexpected_keys}")
        if incompatible.missing_keys:
            missing_non_encoder = [k for k in incompatible.missing_keys if "encoder" not in k]
            if missing_non_encoder:
                print(f"  Warning: missing keys: {missing_non_encoder}")

        return model

    @staticmethod
    def _intrinsics_from_scene_info(json_path) -> Optional[np.ndarray]:
        """Pull a [3, 3] K from a Replicator scene-info JSON (cam_K, row-major).

        Returns None if the file is missing/malformed so callers can fall
        back to no-intrinsics inference on non-Replicator images.
        """
        import json

        try:
            with open(json_path) as f:
                cam = json.load(f).get("camera", {})
            cam_K = cam.get("cam_K")
            if cam_K is not None:
                return np.asarray(cam_K, dtype=np.float32).reshape(3, 3)
        except (OSError, ValueError, KeyError):
            pass
        return None

    def _preprocess(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Preprocess an image for inference.

        Matches the training pipeline: aspect-ratio-preserving resize to fit
        within img_size, then zero-pad to the full img_size.

        Args:
            image: [H, W, 3] BGR uint8 (OpenCV format).
            intrinsics: optional [3, 3] camera matrix in *original* image
                coordinates; will be letterbox-scaled to match the
                preprocessed tensor and stashed on `self._scaled_intrinsics`.

        Returns:
            Tensor [1, 3, img_H, img_W] float32, range [0, 255].
        """
        h, w = image.shape[:2]
        self._original_size = (h, w)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Compute scale factor (fit within img_size, preserving aspect ratio)
        scale = min(self.img_size[0] / h, self.img_size[1] / w)
        new_h, new_w = round(h * scale), round(w * scale)
        self._scaled_size = (new_h, new_w)

        # Resize preserving aspect ratio
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to img_size (bottom-right padding with zeros)
        padded = np.zeros((*self.img_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        # Letterbox-scale the intrinsics into the padded canvas coordinate
        # frame. Top-left placement → no cx/cy translation, only the resize
        # ratio applies to fx, fy, cx, cy.
        self._scaled_intrinsics = None
        if intrinsics is not None:
            K = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3).copy()
            K[0, 0] *= scale
            K[1, 1] *= scale
            K[0, 2] *= scale
            K[1, 2] *= scale
            self._scaled_intrinsics = (
                torch.from_numpy(K).unsqueeze(0).to(self.device)
            )

        # To tensor [1, 3, H, W]
        tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def _postprocess(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        occlusion_logits: Optional[torch.Tensor],
        query_tokens: torch.Tensor,
    ) -> EoMTResult:
        """
        Post-process model outputs into panoptic predictions.

        Args:
            mask_logits: [1, num_q, H, W] mask predictions.
            class_logits: [1, num_q, num_classes+1] class predictions.
            occlusion_logits: [1, num_q] occlusion predictions (optional).
            query_tokens: [1, num_q, embed_dim] query embeddings.

        Returns:
            EoMTResult with all predictions.
        """
        h, w = self._original_size

        # Class scores
        class_probs = class_logits.softmax(dim=-1)
        scores, classes = class_probs.max(-1)  # [1, num_q]
        scores = scores[0]
        classes = classes[0]

        # Occlusion scores
        occ_scores = occlusion_logits[0].sigmoid() if occlusion_logits is not None else None

        # Filter by confidence and non-background
        keep = classes.ne(class_logits.shape[-1] - 1) & (scores > self.mask_thresh)

        if not keep.any():
            return EoMTResult(
                panoptic_mask=np.zeros((h, w), dtype=np.int32),
                class_mask=np.full((h, w), self.num_classes, dtype=np.int32),
                segments=[],
                query_tokens=query_tokens[0].cpu().numpy(),
                raw_masks=np.zeros((0, h, w), dtype=bool),
                scores=np.array([]),
            )

        masks = mask_logits[0].sigmoid()  # [num_q, H, W]
        mask_ids = (scores[..., None, None] * masks)[keep].argmax(0)  # [H, W]

        # Build panoptic map
        panoptic_mask = np.full((h, w), -1, dtype=np.int32)
        class_mask = np.full((h, w), self.num_classes, dtype=np.int32)
        segments = []
        raw_mask_list = []
        score_list = []
        stuff_segment_ids = {}
        segment_id = 0

        for k, class_id in enumerate(classes[keep].tolist()):
            orig_mask = masks[keep][k] >= 0.5
            new_mask = mask_ids == k
            final_mask = (orig_mask & new_mask).cpu().numpy()

            orig_area = orig_mask.sum().item()
            new_area = new_mask.sum().item()
            final_area = final_mask.sum()

            if orig_area == 0 or new_area == 0 or final_area == 0:
                continue
            if new_area / orig_area < self.overlap_thresh:
                continue

            # Handle stuff merging
            if class_id in self.stuff_classes:
                if class_id in stuff_segment_ids:
                    existing_id = stuff_segment_ids[class_id]
                    panoptic_mask[final_mask] = existing_id
                    class_mask[final_mask] = class_id
                    # Update area of existing segment
                    for seg in segments:
                        if seg["segment_id"] == existing_id:
                            seg["area"] = int((panoptic_mask == existing_id).sum())
                    continue
                else:
                    stuff_segment_ids[class_id] = segment_id

            panoptic_mask[final_mask] = segment_id
            class_mask[final_mask] = class_id

            occ_val = float(occ_scores[keep][k].item()) if occ_scores is not None else None

            segments.append({
                "segment_id": segment_id,
                "class_id": class_id,
                "class_name": self.CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "confidence": float(scores[keep][k].item()),
                "occlusion_score": occ_val,
                "visibility_ratio": occ_val,
                "area": int(final_area),
                "is_stuff": class_id in self.stuff_classes,
            })

            raw_mask_list.append(final_mask)
            score_list.append(float(scores[keep][k].item()))
            segment_id += 1

        return EoMTResult(
            panoptic_mask=panoptic_mask,
            class_mask=class_mask,
            segments=segments,
            query_tokens=query_tokens[0].cpu().numpy(),
            raw_masks=np.stack(raw_mask_list) if raw_mask_list else np.zeros((0, h, w), dtype=bool),
            scores=np.array(score_list),
        )

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, np.ndarray],
        intrinsics: Optional[np.ndarray] = None,
    ) -> EoMTResult:
        """
        Run inference on a single image.

        Args:
            image: Path to image file, or [H, W, 3] BGR numpy array.
            intrinsics: optional [3, 3] camera intrinsics in *original*
                image coordinates. When provided, the model's depth path
                conditions on FOV via the intrinsics MLP. When None, the
                model falls back to its no-intrinsics behaviour (gated by
                `EoMT.use_intrinsics`).

        Returns:
            EoMTResult with panoptic segmentation + occlusion predictions.
        """
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            # Auto-load camera intrinsics from a sibling Replicator scene-info
            # JSON so the depth path conditions on FOV the way it did in
            # training. Silently skipped for non-Replicator images.
            if intrinsics is None:
                scene_jsons = sorted(img_path.parent.glob("*scene_info*.json"))
                if scene_jsons:
                    intrinsics = self._intrinsics_from_scene_info(scene_jsons[0])
            image = cv2.imread(str(img_path))
            if image is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")

        tensor = self._preprocess(image, intrinsics=intrinsics)
        K = getattr(self, "_scaled_intrinsics", None)

        # Forward pass
        (
            mask_logits_per_layer, class_logits_per_layer, occ_per_layer,
            depth_pred, normal_pred, query_tokens,
        ) = self.model(tensor / 255.0, intrinsics=K)

        # Use final layer predictions
        mask_logits = mask_logits_per_layer[-1]
        class_logits = class_logits_per_layer[-1]
        occ_logits = occ_per_layer[-1] if occ_per_layer is not None else None

        # The mask outputs are downsampled, first restore them back to the padded img_size
        mask_logits = F.interpolate(
            mask_logits, self.img_size, mode="bilinear", align_corners=False
        )

        # Crop out padding region, then resize to original image size
        # (mirrors revert_resize_and_pad_logits_instance_panoptic from training)
        sh, sw = self._scaled_size
        mask_logits = mask_logits[:, :, :sh, :sw]
        mask_logits = F.interpolate(
            mask_logits, self._original_size, mode="bilinear", align_corners=False
        )

        # Mirror the same letterbox-crop-resize for the depth output so it
        # lands in the original image coordinate frame.
        depth_np = None
        if depth_pred is not None:
            # depth_pred: [1, 1, H_pad, W_pad]
            depth_pred = depth_pred[:, :, :sh, :sw]
            depth_pred = F.interpolate(
                depth_pred, self._original_size, mode="bilinear", align_corners=False
            )
            depth_np = depth_pred[0, 0].detach().cpu().numpy().astype(np.float32)

        # Same letterbox-crop-resize for the normal map. Bilinear interp
        # un-normalises the vectors at fractional sample points; renormalise
        # afterwards so the output is a clean unit-vector field.
        normal_np = None
        if normal_pred is not None:
            normal_pred = normal_pred[:, :, :sh, :sw]
            normal_pred = F.interpolate(
                normal_pred, self._original_size, mode="bilinear", align_corners=False
            )
            normal_pred = F.normalize(normal_pred, dim=1, eps=1e-8)
            normal_np = normal_pred[0].detach().cpu().numpy().astype(np.float32)

        result = self._postprocess(mask_logits, class_logits, occ_logits, query_tokens)
        result.depth = depth_np
        result.normal = normal_np
        return result

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

    def visualize(
        self,
        result: EoMTResult,
        image: Union[str, np.ndarray],
        save_path: Optional[str] = None,
        show: bool = True,
        alpha: float = 0.5,
    ):
        """
        Visualize panoptic segmentation results overlaid on the input image.

        Args:
            result: EoMTResult from __call__.
            image: Original image (path or array).
            save_path: If set, saves visualization to this path.
            show: If True, displays with matplotlib.
            alpha: Mask overlay transparency.
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 4, figsize=(26, 7))

        # --- Panel 1: Original image ---
        axes[0].imshow(image_rgb)
        axes[0].set_title("Input Image", fontsize=14)
        axes[0].axis("off")

        # --- Panel 2: Panoptic segmentation overlay ---
        overlay = image_rgb.copy().astype(np.float32)
        unique_segments = [s for s in result.segments if not s["is_stuff"]]

        # Generate distinct colors
        colors = {}
        for idx, seg in enumerate(unique_segments):
            hue = idx / max(len(unique_segments), 1)
            rgb = np.array(hsv_to_rgb((hue, 0.8, 0.9))) * 255
            colors[seg["segment_id"]] = rgb

        for seg in unique_segments:
            mask = result.panoptic_mask == seg["segment_id"]
            color = colors[seg["segment_id"]]
            overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

            # Draw segment label
            ys, xs = np.where(mask)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                label = f"{seg['class_name']}"
                if seg["visibility_ratio"] is not None:
                    label += f" ({seg['visibility_ratio']:.0%})"
                axes[1].text(
                    cx, cy, label,
                    color="white", fontsize=8, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
                )

        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title(f"Panoptic Segmentation ({len(unique_segments)} objects)", fontsize=14)
        axes[1].axis("off")

        # --- Panel 3: Occlusion heatmap ---
        occ_map = np.zeros((*result.panoptic_mask.shape, 3), dtype=np.float32)
        for seg in unique_segments:
            mask = result.panoptic_mask == seg["segment_id"]
            vis = seg["visibility_ratio"] if seg["visibility_ratio"] is not None else 1.0

            # Green = fully visible, Red = heavily occluded
            r = 1.0 - vis
            g = vis
            occ_map[mask] = [r, g, 0.2]

        # Blend with grayscale image
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray_rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        blended = gray_rgb * 0.4 + occ_map * 0.6
        blended = np.clip(blended, 0, 1)

        axes[2].imshow(blended)
        axes[2].set_title("Visibility Heatmap (green=visible, red=occluded)", fontsize=14)
        axes[2].axis("off")

        # --- Panel 4: Depth heatmap ---
        if result.depth is not None:
            im = axes[3].imshow(result.depth, cmap="turbo")
            axes[3].set_title("Depth (m)", fontsize=14)
            fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        else:
            axes[3].imshow(np.zeros(result.panoptic_mask.shape, dtype=np.uint8), cmap="gray", vmin=0, vmax=1)
            axes[3].set_title("Depth: N/A", fontsize=14)
        axes[3].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def print_results(self, result: EoMTResult):
        """Print a formatted summary of inference results."""
        print(f"\n{'='*60}")
        print(f"  EoMT Inference Results")
        print(f"  Detected {len(result.segments)} segments")
        print(f"{'='*60}")

        for seg in result.segments:
            vis = seg["visibility_ratio"]
            vis_str = f"{vis:.1%}" if vis is not None else "N/A"
            print(
                f"  [{seg['segment_id']:3d}] {seg['class_name']:15s}  "
                f"conf={seg['confidence']:.3f}  "
                f"vis={vis_str:>6s}  "
                f"area={seg['area']:>6d}px"
            )

        print(f"{'='*60}")
        print(f"  Query tokens shape: {result.query_tokens.shape}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EoMT Panoptic Inference")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to .ckpt file")
    parser.add_argument("--save", type=str, default=None, help="Save visualization path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=640, help="Model input resolution (square); match training")
    parser.add_argument("--no-show", action="store_true", help="Don't display matplotlib window")
    args = parser.parse_args()

    kwargs = {"device": args.device, "img_size": (args.img_size, args.img_size)}
    if args.ckpt is not None:
        kwargs["ckpt_path"] = args.ckpt
    model = EoMTInference(**kwargs)
    result = model(args.image)
    model.print_results(result)
    model.visualize(result, args.image, save_path=args.save, show=not args.no_show)
