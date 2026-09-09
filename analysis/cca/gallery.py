#!/usr/bin/env python3
"""
Signed CCA axis gallery.

This mirrors the sNMF gallery layout, but renders two rows per CCA axis:
positive-loading images followed by negative-loading images. CCA axes are shown
in raw component order.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps, font_manager
from matplotlib import colors as mcolors
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from analysis.supp.gallery_style import render_cca_supp_gallery


FAMILY_STATE_KEY = {
    "cross-species": "cca_all",
    "human": "cca_hum",
    "monkey": "cca_mon",
}

FAMILY_CROSSVIEW_KEY = {
    "cross-species": "cross-species",
    "human": "human_only",
    "monkey": "monkey_only",
}

TOP_AXES = 15
TOP_IMAGES = 20
SUBSAMPLE_TOP_FRAC = 0.01
SEED = 42
IMG_PX = 150
LABEL_W = 270
BAR_W = 138
WC_W = 420
GAP_COL = 18
PAIR_GAP = IMG_PX // 4
OUTLINE_WIDTH = 12
DPI = 400
PANEL_DPI = 300
INFO_BG = "#fbfbfb"
INFO_EDGE = "#d4d4d4"

POS_COLOR = config.plotting.get("positive_color", "#A8393F")
NEG_COLOR = config.plotting.get("negative_color", "#366EA9")
HUM_COLOR = config.plotting.get("human_color", "#7c5799")
MON_COLOR = config.plotting.get("monkey_color", "#bda855")
MISSING_COLOR = "#e2e2e2"

IMG_CACHE: dict[str, np.ndarray] = {}
ARIAL_PATH = None


def configure_fonts() -> None:
    global ARIAL_PATH
    ARIAL_PATH = _get_font_path(("Arial", "Liberation Sans", "DejaVu Sans"))
    if ARIAL_PATH:
        try:
            font_manager.fontManager.addfont(ARIAL_PATH)
            name = font_manager.FontProperties(fname=ARIAL_PATH).get_name()
        except Exception:
            name = "Arial"
    else:
        name = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [name, "Arial", "Liberation Sans", "DejaVu Sans"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def _get_font_path(preferred=("Arial", "Liberation Sans", "DejaVu Sans")):
    for name in preferred:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if path:
                return path
        except Exception:
            continue
    try:
        return font_manager.findfont("sans-serif")
    except Exception:
        return None


def _blank_tile(size: int = IMG_PX) -> np.ndarray:
    return np.full((size, size, 3), 200, np.uint8)


def load_tile(path: Path) -> np.ndarray:
    key = str(path)
    if key in IMG_CACHE:
        return IMG_CACHE[key]
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((IMG_PX, IMG_PX), Image.BILINEAR)
        arr = np.asarray(im).copy()
        bw = max(1, IMG_PX // 60)
        arr[:bw] = 64
        arr[-bw:] = 64
        arr[:, :bw] = 64
        arr[:, -bw:] = 64
    except Exception:
        arr = _blank_tile()
    IMG_CACHE[key] = arr
    return arr


def _fig_to_array(fig, target_width: int) -> np.ndarray:
    fig.tight_layout(pad=0.05)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    arr = buf.reshape(h, w, 4)[..., :3]
    plt.close(fig)
    if h != IMG_PX or w != target_width:
        arr = np.asarray(Image.fromarray(arr).resize((target_width, IMG_PX), Image.BILINEAR))
    return arr


def render_label_panel(entry: dict) -> np.ndarray:
    fig = plt.figure(figsize=(LABEL_W / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95)
    ax.set_facecolor(INFO_BG)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INFO_EDGE)
        spine.set_linewidth(0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pole = entry["pole"]
    color = POS_COLOR if pole == "positive" else NEG_COLOR
    ax.text(0.06, 0.66, f"Comp {entry['component']}", ha="left", va="center",
            fontsize=11, fontweight="bold")
    ax.text(0.06, 0.25, f"{pole.title()} pole", ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=color)
    return _fig_to_array(fig, LABEL_W)


def render_species_corr_panel(human_r: float, monkey_r: float) -> np.ndarray:
    fig = plt.figure(figsize=(BAR_W / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.12, top=0.90)
    ax.set_facecolor(INFO_BG)
    vals = [human_r, monkey_r]
    heights = [float(v) if np.isfinite(v) else 0.0 for v in vals]
    colors = [HUM_COLOR if np.isfinite(human_r) else MISSING_COLOR,
              MON_COLOR if np.isfinite(monkey_r) else MISSING_COLOR]
    alphas = [1.0 if np.isfinite(v) else 0.35 for v in vals]
    xpos = np.arange(2)
    bars = ax.bar(xpos, heights, color=colors, edgecolor="black", linewidth=0.8, width=0.68)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([])
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["0", "1"])
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    return _fig_to_array(fig, BAR_W)


def _trunc_cmap(name="Reds", minval=0.75, maxval=1.0, n=256):
    base = colormaps.get_cmap(name)
    colors = base(np.linspace(minval, maxval, n))
    return mpl.cm.colors.ListedColormap(colors)


def render_wordcloud(enrichment: list[tuple[str, float]], pole: str) -> np.ndarray:
    freqs = {cat.replace("_", " "): max(auc, 0.51) for cat, auc in enrichment if auc > 0.5}
    if not freqs:
        return _text_panel(WC_W, "Enrichment < 0.5 AUROC")
    return render_text_cloud(freqs, cmap_name="Blues" if pole == "negative" else "Reds")


def _text_panel(width: int, text: str) -> np.ndarray:
    fig = plt.figure(figsize=(width / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, wrap=True)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return _fig_to_array(fig, width)


def render_text_cloud(freqs: dict[str, float], cmap_name: str = "Reds") -> np.ndarray:
    items = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)
    vals = [v for _, v in items]
    mn, mx = min(vals), max(vals)
    cmap = _trunc_cmap(cmap_name, minval=0.75, maxval=1.0)
    fig = plt.figure(figsize=(WC_W / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.94)
    left_ax = ax.inset_axes([0.04, 0.08, 0.42, 0.84])
    right_ax = ax.inset_axes([0.54, 0.08, 0.42, 0.84])
    for subax in (left_ax, right_ax):
        subax.set_xlim(0, 1)
        subax.set_ylim(0, 1)
        subax.axis("off")

    y_levels = [0.72, 0.28]
    for idx, (word, val) in enumerate(items[:4]):
        t = 1.0 if mx == mn else (val - mn) / (mx - mn)
        col = 0 if idx < 2 else 1
        row = idx if idx < 2 else idx - 2
        subax = left_ax if col == 0 else right_ax
        r, g, b, _ = cmap(t)
        label = _shorten(word)
        subax.text(0.04, y_levels[row], label, ha="left", va="center",
                   fontsize=10, color=(r, g, b), fontweight="bold", clip_on=True)
    return _fig_to_array(fig, WC_W)


def _shorten(label: str, max_chars: int = 6) -> str:
    label = label.replace("_", " ")
    if len(label) <= max_chars:
        return label
    return label[:max_chars - 1].rstrip() + "."


def derive_categories(stims: list[str] | np.ndarray) -> list[str]:
    return ["_".join(str(s).split("_")[:-1]) if "_" in str(s) else str(s) for s in stims]


def build_img_paths(stims: list[str] | np.ndarray, cats: list[str]) -> list[Path]:
    img_root = Path(config.data_dir) / "things" / "images"
    return [img_root / cat / f"{stim}.jpg" for stim, cat in zip(stims, cats)]


def compute_enrichment(weights: np.ndarray, cats: list[str], cat_counts: Counter) -> list[tuple[str, float]]:
    n = len(weights)
    if n == 0:
        return []
    order = np.argsort(weights)
    sorted_w = weights[order]
    ranks = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_w[j] == sorted_w[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    aucs = []
    for cat, base in cat_counts.items():
        n_pos = base
        n_neg = n - n_pos
        if n_pos < 5 or n_neg <= 0:
            continue
        mask = np.array([c == cat for c in cats])
        sum_ranks = ranks[mask].sum()
        auc = (sum_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        if np.isfinite(auc):
            aucs.append((cat, float(auc)))
    aucs.sort(key=lambda x: x[1], reverse=True)
    return aucs[:5]


def load_state() -> dict:
    state_fp = Path(config.results_dir) / "cca_state.pkl"
    if not state_fp.exists():
        raise FileNotFoundError(f"CCA state not found: {state_fp}")
    with open(state_fp, "rb") as f:
        return pickle.load(f)


def load_crossview() -> dict:
    crossview_fp = Path(config.results_dir) / "crossview_results.pkl"
    if not crossview_fp.exists():
        raise FileNotFoundError(f"Cross-view results not found: {crossview_fp}")
    with open(crossview_fp, "rb") as f:
        return pickle.load(f)


def mean_components_for_family(state: dict, family: str) -> np.ndarray:
    key = FAMILY_STATE_KEY[family]
    cca = state[key]
    views = sorted(cca["components"].keys())
    return np.mean([np.asarray(cca["components"][v], float) for v in views], axis=0)


def ranked_axes(crossview: dict, family: str, top: int) -> list[tuple[int, float]]:
    key = FAMILY_CROSSVIEW_KEY[family]
    corrs = np.asarray(crossview[key]["comp_corrs"], float)
    n = min(int(top), corrs.size)
    return [(int(i), float(corrs[i])) for i in range(n)]


def species_corrs_for_component(crossview: dict, family: str, comp_idx: int) -> tuple[float, float]:
    key = FAMILY_CROSSVIEW_KEY[family]
    view_corrs = crossview[key].get("view_comp_corrs", {})

    def _mean(prefix: str) -> float:
        vals = []
        for view, arr in view_corrs.items():
            if str(view).startswith(prefix):
                arr = np.asarray(arr, float)
                if comp_idx < arr.size and np.isfinite(arr[comp_idx]):
                    vals.append(float(arr[comp_idx]))
        return float(np.mean(vals)) if vals else np.nan

    return _mean("human"), _mean("monkey")


def _outline_image_row(img_block: np.ndarray, color: str, width: int = OUTLINE_WIDTH) -> np.ndarray:
    arr = img_block.copy()
    rgb = np.array(mcolors.to_rgb(color)) * 255.0
    rgb = rgb.astype(np.uint8)
    arr[:width, :, :] = rgb
    arr[-width:, :, :] = rgb
    arr[:, :width, :] = rgb
    arr[:, -width:, :] = rgb
    return arr


def build_entries(
    comps: np.ndarray,
    axes: list[tuple[int, float]],
    crossview: dict,
    family: str,
    cats: list[str],
    cat_counts: Counter,
) -> list[dict]:
    entries = []
    for rank_idx, (comp_idx, crossview_r) in enumerate(axes, start=1):
        scores = comps[:, comp_idx]
        human_r, monkey_r = species_corrs_for_component(crossview, family, comp_idx)
        for pole in ("positive", "negative"):
            pole_scores = scores if pole == "positive" else -scores
            entries.append({
                "rank_idx": rank_idx,
                "component": comp_idx + 1,
                "comp_idx": comp_idx,
                "crossview_r": crossview_r,
                "human_crossview_r": human_r,
                "monkey_crossview_r": monkey_r,
                "pole": pole,
                "scores": scores,
                "pole_scores": pole_scores,
                "enrichment": compute_enrichment(pole_scores, cats, cat_counts),
            })
    return entries


def select_image_indices(entry: dict, mode: str) -> tuple[np.ndarray, int]:
    pole_scores = entry["pole_scores"]
    order = np.argsort(pole_scores)[::-1]
    if mode == "normal":
        return order[:TOP_IMAGES], TOP_IMAGES
    if mode != "subsampled":
        raise ValueError(f"Unknown image selection mode: {mode}")

    pool_n = max(TOP_IMAGES, int(np.ceil(len(order) * SUBSAMPLE_TOP_FRAC)))
    pool = order[:pool_n]
    seed = SEED + entry["component"] * 1009 + (0 if entry["pole"] == "positive" else 100_000)
    rng = np.random.default_rng(seed)
    selected = rng.choice(pool, size=min(TOP_IMAGES, len(pool)), replace=False)
    selected = np.asarray(sorted(selected, key=lambda i: pole_scores[i], reverse=True), dtype=int)
    return selected, pool_n


def assemble_canvas(entries: list[dict], img_paths: list[Path], mode: str) -> np.ndarray | None:
    if not entries:
        return None

    width_images = TOP_IMAGES * IMG_PX
    gap_col = np.full((IMG_PX, GAP_COL, 3), 255, np.uint8)
    total_width = LABEL_W + BAR_W + 3 * GAP_COL + WC_W + width_images
    pair_gap = np.full((PAIR_GAP, total_width, 3), 255, np.uint8)
    parts = []

    for i, ent in enumerate(entries):
        order, _ = select_image_indices(ent, mode)
        tiles = [load_tile(img_paths[j]) for j in order]
        while len(tiles) < TOP_IMAGES:
            tiles.append(_blank_tile())

        img_block = np.hstack(tiles)
        outline_color = POS_COLOR if ent["pole"] == "positive" else NEG_COLOR
        img_block = _outline_image_row(img_block, outline_color)

        label_panel = render_label_panel(ent)
        corr_panel = render_species_corr_panel(ent["human_crossview_r"], ent["monkey_crossview_r"])
        wc_panel = render_wordcloud(ent["enrichment"], ent["pole"])
        row = np.hstack([
            label_panel,
            gap_col.copy(),
            corr_panel,
            gap_col.copy(),
            wc_panel,
            gap_col.copy(),
            img_block,
        ])
        parts.append(row)

        is_negative_row = ent["pole"] == "negative"
        has_next_axis = i < len(entries) - 1
        if is_negative_row and has_next_axis:
            parts.append(pair_gap.copy())

    return np.vstack(parts)


def save_summary(entries: list[dict], img_paths: list[Path], stims: list[str], out_path: Path, mode: str) -> None:
    rows = []
    for ent in entries:
        order, pool_n = select_image_indices(ent, mode)
        rows.append({
            "rank": ent["rank_idx"],
            "component": ent["component"],
            "pole": ent["pole"],
            "image_selection": mode,
            "subsample_pool_n": pool_n if mode == "subsampled" else "",
            "crossview_r": ent["crossview_r"],
            "human_crossview_r": ent["human_crossview_r"],
            "monkey_crossview_r": ent["monkey_crossview_r"],
            "top_categories": ";".join(cat for cat, _ in ent["enrichment"]),
            "top_category_auc": ";".join(f"{auc:.4f}" for _, auc in ent["enrichment"]),
            "top_stimuli": ";".join(str(stims[i]) for i in order),
            "top_image_paths": ";".join(str(img_paths[i]) for i in order),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def default_output_paths(family: str, top: int, mode: str) -> tuple[Path, Path]:
    mode_suffix = "" if mode == "normal" else "_subsampled1pct"
    fig_fp = Path(config.fig_dir) / "cca_analysis" / family / "gallery" / f"top{top}_signed_gallery{mode_suffix}.pdf"
    res_fp = Path(config.results_dir) / "cca_analysis" / family / f"top{top}_signed_gallery{mode_suffix}.csv"
    return fig_fp, res_fp


def render_gallery(
    family: str,
    top: int,
    mode: str,
    out_path: Path | None = None,
    summary_path: Path | None = None,
) -> None:
    configure_fonts()
    state = load_state()
    crossview = load_crossview()
    stims = list(state["all_stims"])
    cats = derive_categories(stims)
    img_paths = build_img_paths(stims, cats)
    cat_counts = Counter(cats)
    comps = mean_components_for_family(state, family)
    axes = ranked_axes(crossview, family, top)
    axes = [(idx, r) for idx, r in axes if idx < comps.shape[1]]
    entries = build_entries(comps, axes, crossview, family, cats, cat_counts)
    if not entries:
        print(f"[cca.gallery] {family}: no axes to render.")
        return

    default_fig, default_res = default_output_paths(family, top, mode)
    fig_fp = out_path or default_fig
    render_cca_supp_gallery(
        entries=entries,
        image_paths=img_paths,
        family=family,
        mode=mode,
        select_images=select_image_indices,
        output_path=fig_fp,
        positive_color=POS_COLOR,
        negative_color=NEG_COLOR,
        dpi=DPI,
    )

    res_fp = summary_path or default_res
    save_summary(entries, img_paths, stims, res_fp, mode)
    print(f"[cca.gallery] saved {fig_fp}")
    print(f"[cca.gallery] saved {res_fp}")
    print(f"[cca.gallery] axes: {[(idx + 1, round(r, 4)) for idx, r in axes]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render signed CCA axis galleries.")
    parser.add_argument("--family", choices=["all", *sorted(FAMILY_STATE_KEY)], default="cross-species")
    parser.add_argument("--top", type=int, default=TOP_AXES)
    parser.add_argument("--mode", choices=["normal", "subsampled", "both"], default="normal")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    families = sorted(FAMILY_STATE_KEY) if args.family == "all" else [args.family]
    modes = ["normal", "subsampled"] if args.mode == "both" else [args.mode]
    if (args.out or args.summary) and (len(families) > 1 or len(modes) > 1):
        raise ValueError("--out/--summary can only be used with one family and one mode.")

    for family in families:
        for mode in modes:
            render_gallery(family, args.top, mode, args.out, args.summary)


if __name__ == "__main__":
    main()
