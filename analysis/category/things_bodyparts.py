#!/usr/bin/env python3
"""
Standalone body-part covariate visualization for Supplementary Figure.

Draws a word cloud (category-level means) and a deterministic top-image grid
for the behavioral "body part" dimension from the THINGS dataset, using the
paper's mosaic/density layout without subsampling.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter1d

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from functions.plotting import setup_style
from model.feature.core.feature_io import load_stims, load_beh

# Layout (mirrors analysis/nmf/viz/viz_grid deterministic grid)
ROWS = 10
COLS = 10
IMG_PX = 200
MOSAIC_BORDER_WIDTH = 10.0
TOP_N = ROWS * COLS


def _find_bodypart(names, key='body part'):
    key = key.lower()
    for i, name in enumerate(names):
        if key in name.lower():
            return i, name
    return None, None


def load_bodypart_scores():
    stims, cats = load_stims()
    X, names, _ = load_beh(stims)
    if X.size == 0 or not names:
        return None, None, stims, cats
    idx, label = _find_bodypart(names)
    if idx is None:
        return None, None, stims, cats
    return np.asarray(X[:, idx], float), label, stims, cats


def load_img_simple(path, size):
    try:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((size, size), Image.BILINEAR)

        arr = np.array(im)
        b_color = (64, 64, 64)
        bw = max(1, size // 60)
        arr[:bw, :] = b_color; arr[-bw:, :] = b_color
        arr[:, :bw] = b_color; arr[:, -bw:] = b_color
        return Image.fromarray(arr)
    except Exception:
        blank = np.full((size, size, 3), 128, np.uint8)
        return Image.fromarray(blank)


def _build_mosaic(paths, rows, cols, px):
    blank = np.full((px, px, 3), 128, np.uint8)
    tiles = [load_img_simple(p, px) if p is not None else Image.fromarray(blank) for p in paths]
    tiles += [Image.fromarray(blank)] * (rows * cols - len(tiles))
    tiles = [np.asarray(t) for t in tiles]
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    return grid


def _border(ax, color='black', lw=3):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor='none', edgecolor=color,
                           linewidth=lw, clip_on=False))


def _plot_distribution_positive_only(ax, vec, order, ipos, inv):
    vals = vec[order]; total = len(vals)

    def _smooth(y): return gaussian_filter1d(y, sigma=len(y) / 50) if len(y) > 0 else y

    x = np.linspace(0, total - 1, 500)
    y = _smooth(np.interp(x, np.arange(total), vals))
    if y.max() > 0:
        y *= (vals.max() or 1) / y.max()

    for i in range(len(x) - 1):
        frac = x[i] / total if total else 0
        ax.fill_between(x[i:i + 2], 0, y[i:i + 2], color=cm.Reds(0.3 + 0.7 * frac), lw=0)

    ax.axhline(0, color='k', lw=0)
    ax.set_xlim(0, total)
    y0, y1 = ax.get_ylim(); dy = y1 - y0
    ax.set_ylim(y0 - 0.05 * dy, y1 + 0.05 * dy)
    for sp in ax.spines.values():
        sp.set_visible(False)

    xp_all = inv[ipos]; xp0, xp1 = xp_all.min(), xp_all.max()
    y0, y1 = ax.get_ylim()
    pos_color = config.plotting.get('positive_color', '#8B0000')
    ax.plot([xp0, xp1], [0, 0], color=pos_color, lw=7)
    ax.plot([xp0, xp0], [0, y1], color=pos_color, lw=7)
    ax.plot([xp1, xp1], [0, y1], color=pos_color, lw=7)
    ax.set_xticks([]); ax.set_yticks([])


def _build_positive_mosaic(scores, stims, cats, img_dir, rows=ROWS, cols=COLS, img_px=IMG_PX):
    order = np.argsort(scores)[::-1]
    ipos = order[:rows * cols]
    paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ipos]
    mosaic = _build_mosaic(paths, rows, cols, img_px)
    return mosaic, ipos


def _wordcloud_array(cat_scores):
    if WordCloud is None:
        return np.full((IMG_PX, IMG_PX, 3), 235, np.uint8)
    freqs = {c.replace('_', ' '): max(0.001, float(v)) for c, v in cat_scores.items()
             if np.isfinite(v) and v > 0}
    if not freqs:
        return np.full((IMG_PX, IMG_PX, 3), 235, np.uint8)
    cmap = _trunc_cmap('Reds', minval=0.75, maxval=1.0)
    wc = WordCloud(
        width=900, height=900, background_color='white', colormap=cmap,
        prefer_horizontal=1.0, min_font_size=28,
    )
    wc.generate_from_frequencies(freqs)

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        val = freqs.get(word, 0.5)
        keys = list(freqs.values())
        mn, mx = min(keys), max(keys)
        t = 1.0 if mx == mn else (val - mn) / (mx - mn)
        r, g, b, _ = cmap(t)
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    wc.recolor(color_func=color_func)
    return wc.to_array()


def _trunc_cmap(name, minval=0.0, maxval=1.0):
    base = cm.get_cmap(name)
    colors = base(np.linspace(minval, maxval, 256))
    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)


def plot_bodypart_covariate(scores, stims, cats, out_path, label='Body part'):
    if scores is None or len(scores) == 0:
        print('[bodypart] no scores found; skipping figure.')
        return

    img_dir = os.path.join(config.data_dir, 'things', 'images')
    mosaic, ipos = _build_positive_mosaic(scores, stims, cats, img_dir)
    order = np.argsort(scores)
    inv = np.empty(len(order), int); inv[order] = np.arange(len(order))

    # Category-level averages for word cloud
    cat_scores = {}
    for c in np.unique(cats):
        idx = [i for i, cat in enumerate(cats) if cat == c]
        vals = np.asarray(scores)[idx]
        cat_scores[c] = float(np.nanmean(vals))
    wc_arr = _wordcloud_array(cat_scores)

    setup_style()
    density_h = 180
    fig_w = (COLS * IMG_PX + wc_arr.shape[1]) / 110
    fig_h = (ROWS * IMG_PX + density_h) / 100
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.08)

    ax_wc = fig.add_subplot(gs[0])
    ax_wc.imshow(wc_arr); ax_wc.axis('off')
    ax_wc.set_title(f'{label} (by category)', fontsize=12, pad=6)

    right = gs[1].subgridspec(2, 1, height_ratios=[ROWS * IMG_PX, density_h], hspace=0.04)
    ax_img = fig.add_subplot(right[0])
    ax_img.imshow(mosaic, interpolation='nearest')
    ax_img.axis('off')
    ax_img.set_title(f'Top {TOP_N} stimuli ({label})', fontsize=12, pad=4)
    ax_img.set_xlim(-0.5, mosaic.shape[1] - 0.5)
    ax_img.set_ylim(mosaic.shape[0] - 0.5, -0.5)
    ax_img.set_aspect('equal')
    _border(ax_img, color=config.plotting.get('positive_color', '#8B0000'), lw=MOSAIC_BORDER_WIDTH)

    ax_den = fig.add_subplot(right[1])
    _plot_distribution_positive_only(ax_den, scores, order, ipos, inv)
    ax_den.axis('off')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[bodypart] saved figure to {out_path}")


def main():
    scores, label, stims, cats = load_bodypart_scores()
    if scores is None:
        print('[bodypart] body-part dimension not found in behavioral features.')
        return
    fig_dir = Path(__file__).resolve().parent / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / 'bodypart_covariate.pdf'
    plot_bodypart_covariate(scores, stims, cats, out_path, label or 'Body part')


if __name__ == '__main__':
    main()
