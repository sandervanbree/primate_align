#!/usr/bin/env python3
"""
Helper for robust CLIP extraction with orientation-safe CLS pooling.
Used to avoid token-first (T, B, D) vs batch-first (B, T, D) discrepancies from OpenAI CLIP hooks.

Public API:
- extract_clip_features(image_dir, variant, layers, batch_size, device, center_crop=True)
    Returns: dict[layer_name] = numpy.ndarray of shape (N, D)

Notes:
- This module does not write to disk; the caller (e.g., extract_features._extract_clip) is responsible for saving and summarizing.
"""

from __future__ import annotations

import os
import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


def _load_images_sorted(img_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff")
    paths: List[str] = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(img_dir, "**", e), recursive=True))
    return sorted(paths)


def _cls_pool_guard(x: torch.Tensor, batch_size_hint: int) -> torch.Tensor:
    """
    Orientation-safe pooling to produce (B, D).
    - 4D (B, C, H, W): GAP over H,W
    - 3D tokens: support (B, T, D) or (T, B, D) by using batch_size_hint
    - 2D (B, D): passthrough
    - Fallback: average over non-batch dims
    """
    if isinstance(x, (list, tuple)):
        x = x[0]
    if x.ndim == 4:
        return x.mean(dim=(2, 3))
    if x.ndim == 3:
        a, b, c = x.shape
        # If first dim equals batch size: (B, T, D)
        if a == batch_size_hint:
            return x[:, 0, :]
        # If second dim equals batch size: (T, B, D)
        if b == batch_size_hint:
            return x[0, :, :]
        # Heuristic fallback: prefer taking first along leading dim as tokens
        return x[0, :, :] if a >= b else x[:, 0, :]
    if x.ndim == 2:
        return x
    if x.ndim > 2:
        dims = tuple(range(1, x.ndim))
        return x.mean(dim=dims)
    return x


def extract_clip_features(
    image_dir: str,
    variant: str,
    layers: List[str],
    batch_size: int,
    device: str,
    center_crop: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract features for given CLIP layers with robust pooling to (B, D).

    Args:
        image_dir: root directory of images (can contain class subfolders)
        variant: CLIP variant string (e.g., 'ViT-B/16')
        layers: list of layer/module names to extract (e.g., 'visual.transformer.resblocks.0')
        batch_size: number of images per batch
        device: 'cuda' or 'cpu'
        center_crop: whether to use CLIP's default center crop (applies via preprocess)

    Returns:
        Dict[layer_name, np.ndarray (N, D)]
    """
    import clip as _clip  # OpenAI CLIP

    model, preprocess = _clip.load(variant, device=device, jit=False)
    model.eval()

    want = set(layers)
    mod_map: Dict[str, torch.nn.Module] = {"visual": model.visual}
    for key, mod in model.visual.named_modules():
        tag = "visual" if key == "" else f"visual.{key}"
        mod_map[tag] = mod
    buf: Dict[str, List[np.ndarray]] = {L: [] for L in want}
    hooks = []
    missing = sorted(L for L in want if L not in mod_map)
    if missing:
        raise ValueError(f"Requested CLIP layers not found: {missing}")

    def _mk(name: str):
        def _h(m, _, out):
            x = _cls_pool_guard(out, batch_size_hint=current_bs).detach().cpu().numpy()
            buf[name].append(x)
        return _h

    for L in want:
        mod = mod_map.get(L)
        if mod is None:
            continue
        hooks.append(mod.register_forward_hook(_mk(L)))

    files = _load_images_sorted(image_dir)
    if not files:
        for h in hooks: h.remove()
        return {k: np.zeros((0, 0), dtype=np.float32) for k in want}

    # batching
    global current_bs  # updated per batch for hook pooling guard
    current_bs = 0

    with torch.inference_mode():
        for j in range(0, len(files), batch_size):
            batch_files = files[j:j + batch_size]
            ims = [Image.open(p).convert("RGB") for p in batch_files]
            px = torch.stack([preprocess(im) for im in ims]).to(device)
            current_bs = len(ims)
            # triggers hooks
            _ = model.encode_image(px)

    for h in hooks:
        h.remove()

    # stack across batches and return
    out: Dict[str, np.ndarray] = {}
    for L, parts in buf.items():
        if not parts:
            continue
        X = np.concatenate(parts, axis=0)
        # Assert final dimensionality is 2D (N, D)
        if X.ndim != 2:
            # As last safety, reduce to 2D
            if X.ndim == 3:
                # treat as tokens, take CLS along axis 1 by default
                X = X[:, 0, :]
            else:
                axes = tuple(range(1, X.ndim))
                X = X.mean(axis=axes)
        out[L] = X
    return out
