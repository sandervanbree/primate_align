import os, csv, torch, torch.nn as nn
from thingsvision import get_extractor

# defaults (edit as needed)
BASE_DIR = os.path.dirname(__file__)
ARCH_DIR = os.path.join(BASE_DIR, "arch")
OUT_CSV = os.path.join(BASE_DIR, "layer_gallery.csv")
DEVICE = "cuda"
MODELS = [
    dict(name="alexnet", model_name="alexnet", source="torchvision"),
    dict(name="resnet50", model_name="resnet50", source="torchvision"),
    dict(name="vgg16", model_name="vgg16", source="torchvision"),
    dict(name="clip", model_name="clip", source="custom", variant="ViT-B/16"),
    dict(name="clip_rn50", model_name="clip", source="custom", variant="RN50"),
    dict(name="dinov2-vit-base-p14", model_name="dinov2-vit-base-p14", source="ssl"),
]

def _ensure_dirs():
    os.makedirs(ARCH_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

def _kind(mname, m):
    n = m.__class__.__name__
    if isinstance(m, nn.Conv2d): return "convolutional"
    if isinstance(m, nn.Linear): return "fully_connected"
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)): return "normalization"
    if isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU)): return "activation"
    if any(isinstance(m, t) for t in (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d)): return "pooling"
    if "attn" in mname or "transformer" in mname or "Attention" in n or "attention" in n: return "attention"
    return "other"

def _leaf_modules(model):
    return [(n, m) for n, m in model.named_modules() if n and len(list(m.children())) == 0]

def _get_extractor(cfg):
    dev = DEVICE if torch.cuda.is_available() else "cpu"
    kwargs = dict(
        model_name=cfg["model_name"],
        pretrained=cfg.get("pretrained", True),
        model_path=None,
        device=dev,
        source=cfg["source"],
    )
    if cfg.get("variant"):
        kwargs["model_parameters"] = {"variant": cfg["variant"]}
    return get_extractor(**kwargs)

def main():
    _ensure_dirs()
    rows = []
    for cfg in MODELS:
        ext = _get_extractor(cfg)
        # NOTE: uncomment the line below, if you are uncertain about layer naming of CLIP / others
        # extractor.show_model()
        # save show_model to file for provenance
        arch_path = os.path.join(ARCH_DIR, f"{cfg['name']}.txt")
        import io, contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                ext.show_model()
        except Exception:
            pass
        txt = buf.getvalue()
        if not txt.strip():
            txt = repr(ext.model)
        with open(arch_path, "w") as f:
            f.write(txt)
        idx = 0
        for lname, mod in _leaf_modules(ext.model):
            rows.append({"model": cfg["name"], "layer_name": lname, "layer_index": idx, "layer_kind": _kind(lname, mod)})
            idx += 1
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer_name", "layer_index", "layer_kind"])
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
