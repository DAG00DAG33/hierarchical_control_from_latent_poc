from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoExtractor:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        feature_type: str = "cls",
        spatial_pool: int = 4,
    ) -> None:
        if feature_type not in {"cls", "spatial"}:
            raise ValueError(f"Unsupported DINO feature_type '{feature_type}'")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.feature_type = feature_type
        self.spatial_pool = spatial_pool

    @torch.inference_mode()
    def encode_batch(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 4:
            raise ValueError(f"Expected RGB batch with shape B,H,W,C, got {images.shape}")
        if images.shape[-1] == 4:
            images = images[..., :3]
        if images.dtype != np.uint8:
            images = np.clip(images, 0, 255).astype(np.uint8)
        pil_images = [Image.fromarray(img) for img in images]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        tokens = outputs.last_hidden_state
        if self.feature_type == "cls":
            features = tokens[:, 0]
        else:
            cls = tokens[:, 0]
            patches = tokens[:, 1:]
            grid_size = int(patches.shape[1] ** 0.5)
            if grid_size * grid_size != patches.shape[1]:
                raise ValueError(f"Expected square DINO patch grid, got {patches.shape[1]} tokens")
            patch_grid = patches.reshape(patches.shape[0], grid_size, grid_size, patches.shape[-1])
            patch_grid = patch_grid.permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(patch_grid, (self.spatial_pool, self.spatial_pool))
            pooled = pooled.permute(0, 2, 3, 1).reshape(pooled.shape[0], -1)
            features = torch.cat([cls, pooled], dim=-1)
        return features.detach().cpu().numpy().astype(np.float32)


def dino_from_config(config: Any, device: torch.device) -> DinoExtractor:
    return DinoExtractor(
        config.get("dino.model_name"),
        device,
        feature_type=str(config.get("dino.feature_type", "cls")),
        spatial_pool=int(config.get("dino.spatial_pool", 4)),
    )


def batched(items: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
