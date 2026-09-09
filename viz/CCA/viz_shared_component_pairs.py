#!/usr/bin/env python3
"""Shared-space CCA scatterplots for selected component pairs."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config


DEFAULT_PAIRS = ((3, 4), (5, 6))
DEFAULT_FORMATS = ("pdf", "png")
POS_COLOR = config.plotting.get("positive_color", "#A8393F")
NEG_COLOR = config.plotting.get("negative_color", "#366EA9")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": True,
        }
    )


def load_shared_components() -> tuple[np.ndarray, list[str]]:
    state_path = Path(config.results_dir) / "cca_state.pkl"
    with state_path.open("rb") as f:
        state = pickle.load(f)

    cca = state["cca_all"]
    views = sorted(cca["components"])
    comps = np.mean([np.asarray(cca["components"][view]) for view in views], axis=0)
    stims = [str(stim) for stim in state["all_stims"]]
    return comps, stims


def derive_categories(stims: list[str]) -> list[str]:
    return ["_".join(stim.split("_")[:-1]) if "_" in stim else stim for stim in stims]


def add_border(arr: np.ndarray, thick: int | None = None, color=(128, 128, 128)) -> np.ndarray:
    arr = arr.copy()
    if thick is None:
        thick = max(2, min(arr.shape[:2]) // 30)
    arr[:thick, :] = color
    arr[-thick:, :] = color
    arr[:, :thick] = color
    arr[:, -thick:] = color
    return arr


def load_square_image(path: Path, size: int) -> np.ndarray:
    try:
        im = Image.open(path).convert("RGB")
        width, height = im.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((size, size), Image.BILINEAR)
        return add_border(np.asarray(im), thick=max(7, size // 30))
    except Exception:
        arr = np.full((size, size, 3), 190, dtype=np.uint8)
        return add_border(arr, thick=max(7, size // 30))


def select_radial_examples(coords: np.ndarray, n_images: int, min_dist_frac: float) -> list[int]:
    center = coords.mean(axis=0)
    offsets = coords - center
    dist = np.linalg.norm(offsets, axis=1)
    theta = np.arctan2(offsets[:, 1], offsets[:, 0])
    spread = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1e-8)
    min_dist = min_dist_frac * spread

    bins = np.linspace(-np.pi, np.pi, n_images + 1)
    bin_idx = np.digitize(theta, bins) - 1
    selected: list[int] = []
    used_bins: set[int] = set()

    for idx in np.argsort(dist)[::-1]:
        this_bin = int(bin_idx[idx])
        if this_bin in used_bins:
            continue
        if selected and np.min(np.linalg.norm(coords[idx] - coords[selected], axis=1)) < min_dist:
            continue
        selected.append(int(idx))
        used_bins.add(this_bin)
        if len(selected) >= n_images:
            return selected

    for idx in np.argsort(dist)[::-1]:
        if int(idx) in selected:
            continue
        if selected and np.min(np.linalg.norm(coords[idx] - coords[selected], axis=1)) < min_dist:
            continue
        selected.append(int(idx))
        if len(selected) >= n_images:
            break

    return selected


def plot_component_pair(
    comps: np.ndarray,
    stims: list[str],
    cats: list[str],
    pair: tuple[int, int],
    *,
    n_images: int,
    image_size: int,
    image_zoom: float,
    min_dist_frac: float,
    label_axes: bool,
) -> plt.Figure:
    c_x, c_y = pair
    coords = comps[:, [c_x - 1, c_y - 1]]
    center = coords.mean(axis=0)
    dist = np.linalg.norm(coords - center, axis=1)
    dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    colors = plt.cm.Greys(0.35 + 0.2 * dist_norm)

    fig, ax = plt.subplots(figsize=(5.8, 5.8), dpi=300)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=11,
        c=colors,
        alpha=1.0,
        edgecolors="none",
        marker="o",
        linewidth=0,
        zorder=1,
    )

    selected = select_radial_examples(coords, n_images=n_images, min_dist_frac=min_dist_frac)

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    spread = max(x_max - x_min, y_max - y_min, 1e-8)
    offset = 0.22 * spread
    img_root = Path(config.data_dir) / "things" / "images"

    for idx in selected:
        direction = coords[idx] - center
        norm = np.linalg.norm(direction)
        unit = direction / norm if norm > 0 else np.array([1.0, 0.0])
        thumb_xy = coords[idx] + unit * offset

        ax.plot(
            [coords[idx, 0], thumb_xy[0]],
            [coords[idx, 1], thumb_xy[1]],
            color="0.2",
            alpha=0.75,
            lw=2.0,
            zorder=2,
        )

        img_path = img_root / cats[idx] / f"{stims[idx]}.jpg"
        arr = load_square_image(img_path, image_size)
        box = OffsetImage(arr, zoom=image_zoom)
        ann = AnnotationBbox(box, thumb_xy, frameon=False, pad=0.08, zorder=4)
        ax.add_artist(ann)

    selected_coords = coords[selected]
    ax.scatter(
        selected_coords[:, 0],
        selected_coords[:, 1],
        s=34,
        color="0.2",
        alpha=0.9,
        edgecolors="none",
        marker="o",
        zorder=5,
    )

    ax.set_xlim(x_min - offset * 1.15, x_max + offset * 1.15)
    ax.set_ylim(y_min - offset * 1.15, y_max + offset * 1.15)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if label_axes:
        ax.text(
            0.5,
            -0.018,
            f"Shared family C{c_x}",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=11,
            color="0.15",
        )
        ax.text(
            -0.018,
            0.5,
            f"Shared family C{c_y}",
            ha="right",
            va="center",
            rotation=90,
            transform=ax.transAxes,
            fontsize=11,
            color="0.15",
        )

    ax.text(
        0.02,
        0.98,
        f"C{c_x} vs C{c_y}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="0.12",
    )
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.04, top=0.99)
    return fig


def parse_pair(pair_text: str) -> tuple[int, int]:
    parts = pair_text.replace("C", "").replace("c", "").replace(":", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected pair like 3,4; got {pair_text!r}")
    try:
        pair = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer pair; got {pair_text!r}") from exc
    if pair[0] < 1 or pair[1] < 1:
        raise argparse.ArgumentTypeError("Component indices are 1-based and must be positive")
    return pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        nargs="+",
        type=parse_pair,
        default=DEFAULT_PAIRS,
        help="1-based component pairs, e.g. --pairs 3,4 5,6",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(config.fig_dir) / "cca_analysis" / "cross-species" / "component_pairs",
    )
    parser.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS)
    parser.add_argument("--n-images", type=int, default=35)
    parser.add_argument("--image-size", type=int, default=220)
    parser.add_argument("--image-zoom", type=float, default=0.23)
    parser.add_argument("--min-dist-frac", type=float, default=0.035)
    parser.add_argument(
        "--axis-labels",
        action="store_true",
        help="Add outside-axis component labels. Off by default to avoid thumbnail collisions.",
    )
    args = parser.parse_args()

    configure_style()
    comps, stims = load_shared_components()
    cats = derive_categories(stims)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for pair in args.pairs:
        if max(pair) > comps.shape[1]:
            raise ValueError(f"Requested {pair}, but only {comps.shape[1]} components are available")
        fig = plot_component_pair(
            comps,
            stims,
            cats,
            pair,
            n_images=args.n_images,
            image_size=args.image_size,
            image_zoom=args.image_zoom,
            min_dist_frac=args.min_dist_frac,
            label_axes=args.axis_labels,
        )
        stem = f"shared_component_pair_C{pair[0]}_vs_C{pair[1]}"
        for fmt in args.formats:
            out_path = args.out_dir / f"{stem}.{fmt.lstrip('.')}"
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=600)
            print(f"Saved {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
