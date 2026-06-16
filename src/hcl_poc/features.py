from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoExtractor:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

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
        features = outputs.last_hidden_state[:, 0]
        return features.detach().cpu().numpy().astype(np.float32)


def batched(items: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]

