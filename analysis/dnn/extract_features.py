"""Minimal layer-wise DNN feature extraction used for manuscript analyses."""

DEVICE = "cuda"

import os, csv, glob
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from PIL import Image

# THINGSvision imports (primary path)
from thingsvision import get_extractor
from thingsvision.utils.data import ImageDataset, DataLoader
from thingsvision.utils.storing import save_features
from extractors.clip_extractor import extract_clip_features

# Resolve paths relative to this file.
BASE_DIR = os.path.dirname(__file__)
CFG = os.path.join(BASE_DIR, "layer_config.toml")
ARCH_DIR = os.path.join(BASE_DIR, "arch")
LAYER_GALLERY = os.path.join(BASE_DIR, "layer_gallery.csv")

# TOML loader
try:
    import tomllib as _toml
    def _load_toml(p: str) -> Dict[str, Any]:
        with open(p, "rb") as f: return _toml.load(f)
except Exception:
    import toml as _toml
    def _load_toml(p: str) -> Dict[str, Any]:
        with open(p, "r") as f: return _toml.load(f)

def _ensure_dirs(*paths: str):
    for p in paths: os.makedirs(p, exist_ok=True)

def _resolve_path(p: str) -> str:
    # resolve any relative path against this file's directory
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(BASE_DIR, p))

def _load_layer_idx_map(csv_path: str) -> Dict[Tuple[str, str], int]:
    idx = {}
    try:
        with open(csv_path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    idx[(row["model"], row["layer_name"])] = int(row["layer_index"])
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return idx

def _idx_for(model: str, layer: str, idx_map: Dict[Tuple[str, str], int]) -> int:
    # exact match
    if (model, layer) in idx_map:
        return idx_map[(model, layer)]
    # parent prefix: take earliest child index among leaves starting with "layer."
    pref = layer + "."
    cand = [v for (m, l), v in idx_map.items() if m == model and l.startswith(pref)]
    if cand:
        return min(cand)

    # deterministic fallback for ViT-style block naming (e.g., dinov3, dinov2)
    def _parse_block_index(name: str) -> int:
        try:
            if name.startswith("blocks."):
                return int(name.split(".")[1])
        except Exception:
            pass
        return -1

    if model.startswith("dinov3") or model.startswith("dinov2"):
        bi = _parse_block_index(layer)
        if bi >= 0:
            return bi
        if layer == "norm":
            # place norm after the last block (12 for ViT-B)
            return 12

    return -1

def _append_summary(summary_tsv: str, row: Dict[str, Any]):
    head = not os.path.exists(summary_tsv)
    with open(summary_tsv, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "layer_name", "layer_index", "n_images", "feat_shape", "save_path", "layer_seq"],
            delimiter="\t"
        )
        if head: w.writeheader()
        w.writerow(row)

def _purge_model_rows(summary_tsv: str, model: str):
    if not os.path.exists(summary_tsv): return
    with open(summary_tsv, "r") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows = [r for r in rows if r.get("model") != model]
    with open(summary_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","layer_name","layer_index","n_images","feat_shape","save_path","layer_seq"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)

def _finalize_layer_seq(summary_tsv: str, model: str):
    if not os.path.exists(summary_tsv): return
    with open(summary_tsv, "r") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    # compute seq 0..K-1 for this model, ordered by layer_index asc
    tgt = [(i, r) for i, r in enumerate(rows) if r.get("model") == model]
    # coerce layer_index to int, default large if missing
    for _, r in tgt:
        try: r["_li"] = int(r.get("layer_index", -1))
        except Exception: r["_li"] = -1
    tgt_sorted = sorted(tgt, key=lambda t: (t[1]["_li"], t[1].get("layer_name","")))
    for seq, (i, r) in enumerate(tgt_sorted):
        rows[i]["layer_seq"] = str(seq)
        if "_li" in rows[i]: del rows[i]["_li"]
    # rewrite file with unified header
    with open(summary_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","layer_name","layer_index","n_images","feat_shape","save_path","layer_seq"], delimiter="\t")
        w.writeheader()
        for r in rows:
            r.pop("_li", None)
            if "layer_seq" not in r: r["layer_seq"] = "-1"
            w.writerow(r)

def _extract_tv(model_cfg: Dict[str, Any], g: Dict[str, Any]):
    name = model_cfg["name"]
    model_name = model_cfg.get("model_name", name)
    src = model_cfg["source"]
    pretrained = bool(model_cfg.get("pretrained", True))
    variant = model_cfg.get("variant", None)

    dev = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    kwargs = dict(model_name=model_name, pretrained=pretrained, model_path=None, device=dev, source=src)
    if variant: kwargs["model_parameters"] = {"variant": variant}
    ext = get_extractor(**kwargs)
    try:
        ext.model.eval()
    except Exception:
        pass

    ds = ImageDataset(
        root=g["images_dir"],
        out_path=g["output_dir"],
        backend=ext.get_backend(),
        transforms=ext.get_transformations(apply_center_crop=g["center_crop"], resize_dim=256, crop_dim=224),
        class_names=None,
        file_names=g.get("files_rel"),
    )
    loader = DataLoader(dataset=ds, batch_size=g["batch_size"], backend=ext.get_backend())

    for layer in model_cfg["layers"]:
        feats = ext.extract_features(batches=loader, module_name=layer, flatten_acts=g["flatten"], output_type="ndarray")
        # Reduce to B x D deterministically:
        # - For conv feature maps (B x C x H x W): global average pool over H,W
        # - For transformer token sequences (B x T x D): CLS token (index 0) at final norm
        if isinstance(feats, np.ndarray):
            if feats.ndim == 4:
                # B x C x H x W -> B x C
                feats = feats.mean(axis=(2, 3))
            elif feats.ndim == 3:
                # ViT-style pooling: patches-mean for intermediates; CLS for final "norm"
                is_tx = (
                    ("transformer" in layer)
                    or ("resblocks" in layer)
                    or ("blocks." in layer)
                    or (layer == "norm")
                    or (name.startswith("dino"))
                    or (name.startswith("dinov2"))
                )
                if is_tx and feats.shape[1] >= 1:
                    if layer == "norm":
                        feats = feats[:, 0, :]          # CLS at final norm
                    else:
                        feats = feats[:, 1:, :].mean(axis=1)  # mean over patches only
                else:
                    axes = tuple(range(1, feats.ndim))
                    feats = feats.mean(axis=axes)
        model_dir = os.path.join(g["output_dir"], name)
        layer_dir = os.path.join(model_dir, layer)
        _ensure_dirs(layer_dir)
        save_pfx = os.path.join(layer_dir, "features")
        save_features(feats, out_path=save_pfx, file_format=g["file_format"])
        rel_path = os.path.join("features", name, layer, "features.npy")
        _append_summary(
            os.path.join(g["output_dir"], "features_summary.tsv"),
            dict(
                model=name,
                layer_name=layer,
                layer_index=_idx_for(name, layer, g["idx_map"]),
                n_images=int(feats.shape[0]),
                feat_shape="x".join(map(str, feats.shape[1:])),
                save_path=rel_path,
                layer_seq=-1
            )
        )
    # finalize contiguous sequence for this model
    _finalize_layer_seq(os.path.join(g["output_dir"], "features_summary.tsv"), name)

# Minimal DINOv3 adapter (HF)
def _load_images_sorted(img_dir: str) -> List[str]:
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff")
    paths = []
    for e in exts: paths.extend(glob.glob(os.path.join(img_dir, "**", e), recursive=True))
    return sorted(paths)

def _dinov3_forward(cfg_layers: List[str], img_dir: str, out_dir: str, batch_size: int, dev: str, flatten: bool, file_format: str, model_name: str, idx_map: Dict[Tuple[str, str], int], files: List[str], center_crop: bool):
    from transformers import AutoImageProcessor, AutoModel
    model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    proc = AutoImageProcessor.from_pretrained(model_id)
    # align crop choice
    try:
        proc.do_center_crop = bool(center_crop)
        proc.size = {"shortest_edge": 256}
        proc.crop_size = {"height": 224, "width": 224}
    except Exception:
        pass
    model = AutoModel.from_pretrained(model_id)
    model.to(dev); model.eval()

    n = len(files)
    if n == 0: return
    feats_per_layer = {L: [] for L in cfg_layers}

    with torch.inference_mode():
        for i in range(0, n, batch_size):
            batch_files = files[i:i+batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_files]
            px = proc(images=imgs, return_tensors="pt").pixel_values.to(dev)
            out = model(pixel_values=px, output_hidden_states=True)
            hs = out.hidden_states   # [emb, blk0..blk11]
            last = out.last_hidden_state

            for L in cfg_layers:
                if L.startswith("blocks."):
                    idx = int(L.split(".")[1])
                    x = hs[idx+1]              # (B, seq, H)
                    x = x[:, 1:, :].mean(dim=1)  # mean over patches
                elif L == "norm":
                    x = last[:, 0, :]          # CLS at final norm
                else:
                    continue
                if flatten and x.ndim > 2:
                    x = x.flatten(1)
                feats_per_layer[L].append(x.detach().cpu().numpy())

    # stack and save
    for L, chunks in feats_per_layer.items():
        if not chunks: continue
        X = np.concatenate(chunks, axis=0)
        model_dir = os.path.join(out_dir, model_name)
        layer_dir = os.path.join(model_dir, L)
        _ensure_dirs(layer_dir)
        pfx = os.path.join(layer_dir, "features")
        save_features(X, out_path=pfx, file_format=file_format)
        _append_summary(
            os.path.join(out_dir, "features_summary.tsv"),
            dict(
                model=model_name,
                layer_name=L,
                layer_index=_idx_for(model_name, L, idx_map),
                n_images=int(X.shape[0]),
                feat_shape="x".join(map(str, X.shape[1:])),
                save_path=os.path.join("features", model_name, L, "features.npy"),
                layer_seq=-1
            )
        )
    # finalize sequence for this model
    _finalize_layer_seq(os.path.join(out_dir, "features_summary.tsv"), model_name)

def _extract_dinov3(model_cfg: Dict[str, Any], g: Dict[str, Any]):
    dev = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    _dinov3_forward(
        cfg_layers=model_cfg["layers"],
        img_dir=g["images_dir"],
        out_dir=g["output_dir"],
        batch_size=g["batch_size"],
        dev=dev,
        flatten=g["flatten"],
        file_format=g["file_format"],
        model_name=model_cfg["name"],
        idx_map=g["idx_map"],
        files=g.get("files_abs", _load_images_sorted(g["images_dir"])),
        center_crop=g["center_crop"],
    )
# CLIP adapter using robust extractor helper (always B x D)
def _extract_clip(model_cfg: Dict[str, Any], g: Dict[str, Any]):
    dev = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    name = model_cfg["name"]
    variant = model_cfg.get("variant", "ViT-B/16")
    layers = list(model_cfg["layers"])
    feats_by_layer = extract_clip_features(
        image_dir=g["images_dir"],
        variant=variant,
        layers=layers,
        batch_size=g["batch_size"],
        device=dev,
        center_crop=g["center_crop"],
    )
    out_dir = g["output_dir"]
    for L, X in feats_by_layer.items():
        mdir = os.path.join(out_dir, name)
        ldir = os.path.join(mdir, L)
        _ensure_dirs(ldir)
        pfx = os.path.join(ldir, "features")
        save_features(X, out_path=pfx, file_format=g["file_format"])
        _append_summary(
            os.path.join(out_dir, "features_summary.tsv"),
            dict(
                model=name,
                layer_name=L,
                layer_index=_idx_for(name, L, g["idx_map"]),
                n_images=int(X.shape[0]),
                feat_shape="x".join(map(str, X.shape[1:])),
                save_path=os.path.join("features", name, L, "features.npy"),
                layer_seq=-1
            )
        )
    _finalize_layer_seq(os.path.join(out_dir, "features_summary.tsv"), name)

def main():
    cfg = _load_toml(CFG)
    g = dict(
        images_dir=_resolve_path(cfg["dataset"]["images_dir"]),
        output_dir=_resolve_path(cfg["general"]["output_dir"]),
        flatten=bool(cfg["general"].get("flatten", True)),
        batch_size=int(cfg["general"].get("batch_size", 32)),
        center_crop=bool(cfg["general"].get("center_crop", True)),
        file_format=str(cfg["general"].get("file_format", "npy")),
    )
    # load index map for gallery lookups
    g["idx_map"] = _load_layer_idx_map(LAYER_GALLERY)
    _ensure_dirs(ARCH_DIR, g["output_dir"])

    for m in cfg.get("model", []):
        # drop existing rows for this model to avoid duplicates on re-runs
        _purge_model_rows(os.path.join(g["output_dir"], "features_summary.tsv"), m.get("name",""))
        if m.get("impl", "") == "dinov3":
            _extract_dinov3(m, g)
        elif m.get("name", "") == "clip":
            _extract_clip(m, g)
        else:
            _extract_tv(m, g)

if __name__ == "__main__":
    main()
