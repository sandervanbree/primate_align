import os, sys
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.paths import config
from functions.plotting import (
    narrow_figure, medium_figure, wide_figure, square_figure, large_figure,
    legend_figure, setup_style, style_axes, format_axes
)


# Shared image helpers
def _square_and_resize(path, target):
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, upper = (w - side) // 2, (h - side) // 2
        im = im.crop((left, upper, left + side, upper + side))
        im = im.resize((target, target), Image.BILINEAR)

        # Inset dark border for consistent look
        im_arr = np.array(im)
        thick = max(2, target // 60)
        border_color = (64, 64, 64)
        im_arr[:thick, :] = border_color
        im_arr[-thick:, :] = border_color
        im_arr[:, :thick] = border_color
        im_arr[:, -thick:] = border_color
        return Image.fromarray(im_arr)
    except Exception:
        blank = np.full((target, target, 3), 128, np.uint8)
        thick = max(2, target // 60)
        border_color = (64, 64, 64)
        blank[:thick, :] = border_color
        blank[-thick:, :] = border_color
        blank[:, :thick] = border_color
        blank[:, -thick:] = border_color
        return Image.fromarray(blank)


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
        arr[:bw, :] = b_color
        arr[-bw:, :] = b_color
        arr[:, :bw] = b_color
        arr[:, -bw:] = b_color
        return Image.fromarray(arr)
    except Exception:
        blank = np.full((size, size, 3), 128, np.uint8)
        return Image.fromarray(blank)


def _build_mosaic(paths, rows, cols, px):
    blank = np.full((px, px, 3), 128, np.uint8)
    tiles = [_square_and_resize(p, px) if p is not None else Image.fromarray(blank) for p in paths]
    tiles += [Image.fromarray(blank)] * (rows * cols - len(tiles))
    tiles = [np.asarray(t) for t in tiles]
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    return grid


def _border(ax, color='black', lw=3):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor='none', edgecolor=color,
                           linewidth=lw, clip_on=False))


# Shared color helper
def _bone_shades(n: int, start: float, end: float):
    tvals = [0.5 * (start + end)] if n <= 1 else np.linspace(start, end, n)
    cols = []
    for t in tvals:
        r, g, b, _ = cm.bone(float(t))
        cols.append((r, g, b))
    return cols


# Shared duplet bars
def plot_duplets_from_enc_scores(domain: str, family: str, comp_indices: list,
                                 feature_label_fontsize: float = 11,
                                 extra_label_gap_frac: float = 0.01):
    if not comp_indices:
        return None

    # Load color ranges from config
    vis_range = config.plotting.get('visual_bone_range', [0.7, 0.8])
    beh_range = config.plotting.get('behavioral_bone_range', [0.4, 0.5])
    sem_range = config.plotting.get('semantic_bone_range', [0.1, 0.2])

    vis_shades = _bone_shades(2, *vis_range)
    beh_shades = _bone_shades(2, *beh_range)
    sem_shades = _bone_shades(2, *sem_range)

    bar_width = 0.5
    gap_between_components = 0.6
    gap_within_groups = 0.0  # No gap between v/b pairs (tighter duplets)
    positions, heights, colors, labels, group_centers, yerrs = [], [], [], [], [], []
    tick_labels = []

    # Load behavioral dimension names for proper labeling
    try:
        beh_dim_names = load_behav_dim_names()
    except:
        beh_dim_names = []

    current_offset = 0.0
    for ci in comp_indices:
        scores = load_enc_scores(domain, family, ci)
        if scores is None:
            continue

        # Get top 2 per feature type
        top_vis = sorted(scores['visual'].items(), key=lambda x: x[1]['mean'], reverse=True)[:2]
        top_beh = sorted(scores['behavioral'].items(), key=lambda x: x[1]['mean'], reverse=True)[:2]
        top_sem = sorted(scores['semantic'].items(), key=lambda x: x[1]['mean'], reverse=True)[:2]

        groups = []
        if top_vis:
            groups.append(('visual', top_vis, vis_shades))
        if top_beh:
            groups.append(('behavioral', top_beh, beh_shades))
        if top_sem:
            groups.append(('semantic', top_sem, sem_shades))

        if not groups:
            continue

        first_idx = len(positions)
        x = current_offset

        for gi, (gname, entries, palette) in enumerate(groups):
            for k, (raw_name, stats) in enumerate(entries):
                positions.append(x)
                heights.append(float(stats['mean']))
                colors.append(palette[min(k, len(palette)-1)])

                if gname == 'behavioral' and beh_dim_names and raw_name.startswith('SPOSE'):
                    try:
                        idx = int(raw_name.split()[-1]) - 1
                        if 0 <= idx < len(beh_dim_names):
                            label = beh_dim_names[idx]
                        else:
                            label = raw_name
                    except Exception:
                        label = raw_name
                else:
                    label = raw_name
                labels.append(label)
                yerrs.append(float(stats.get('std', 0.0)))
                x += bar_width

            if gi < len(groups) - 1:
                x += gap_within_groups

        last_idx = len(positions) - 1
        if last_idx >= first_idx:
            center = (positions[first_idx] + positions[last_idx]) / 2.0
            group_centers.append(center)
            tick_labels.append(str(int(ci)))
        current_offset = x + gap_between_components

    from functions.plotting import wide_figure
    fig = wide_figure()
    ax = fig.add_subplot(111)
    ax.bar(positions, heights, width=bar_width, color=colors,
           edgecolor='black', linewidth=1.2, align='center', zorder=2)

    for x, h, ye in zip(positions, heights, yerrs):
        if ye and ye > 0:
            ax.errorbar(x, h, yerr=ye, fmt='none', ecolor='black',
                        elinewidth=1.8, capsize=0, zorder=3)

    ymax_data = max([h + ye for h, ye in zip(heights, yerrs)] or [1.0])
    ax.set_ylim(0, ymax_data * 1.60)
    ax.margins(y=0)

    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    base_pad = (max(heights) if heights else 1.0) * 0.012
    extra_gap = yrange * float(extra_label_gap_frac)

    for x, h, name, ye in zip(positions, heights, labels, yerrs):
        if name:
            lab = name
            for ch in ['-', '/', ' ']:
                if ch in lab:
                    lab = lab.split(ch)[0]
                    break
            ax.text(x, h + ye + base_pad + extra_gap, lab.replace('_', ' '), rotation=90,
                    ha='center', va='bottom', fontsize=feature_label_fontsize, fontweight='bold', clip_on=False)

    ax.set_ylabel('ΔR²')
    style_axes(ax); format_axes(ax, precision=3)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Component')
    for label in ax.get_xticklabels():
        label.set_fontstyle('normal')
        label.set_fontweight('normal')

    plt.tight_layout(pad=0.6)
    return fig


def make_legend():
    vis_range = config.plotting.get('visual_bone_range', [0.7, 0.8])
    beh_range = config.plotting.get('behavioral_bone_range', [0.4, 0.5])
    sem_range = config.plotting.get('semantic_bone_range', [0.1, 0.2])

    vis_shades = _bone_shades(2, *vis_range)
    beh_shades = _bone_shades(2, *beh_range)
    sem_shades = _bone_shades(2, *sem_range)

    from functions.plotting import narrow_figure
    fig = narrow_figure()
    ax = fig.add_subplot(111)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=vis_shades[0], ec='black', lw=1.0, label='Visual'),
        plt.Rectangle((0, 0), 1, 1, color=beh_shades[0], ec='black', lw=1.0, label='Behavioral'),
        plt.Rectangle((0, 0), 1, 1, color=sem_shades[0], ec='black', lw=1.0, label='Semantic'),
    ]
    ax.axis('off')
    _ = ax.legend(handles=handles, frameon=False, loc='center')
    plt.tight_layout(pad=0.2)
    return fig


def make_ridge_legend():
    """Legend mapping bars to full subject labels.

    Items: Human 1, Human 2, Human 3, Monkey 1, Monkey 2 with alpha gradations to match bars.
    """
    fig = legend_figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    col_h = config.plotting.get('human_color', '#7c5799')
    col_m = config.plotting.get('monkey_color', '#bda855')
    items = [
        ('Human 1', col_h, 1.0), ('Human 2', col_h, 0.8), ('Human 3', col_h, 0.6),
        ('Monkey 1', col_m, 1.0), ('Monkey 2', col_m, 0.8)
    ]
    handles = []
    for lab, col, a in items:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=col, alpha=a, ec='black', lw=1.0, label=lab))
    ax.legend(handles=handles, loc='center', frameon=False, ncol=5)
    plt.tight_layout(pad=0.05)
    return fig


def load_enc_matrix(domain: str, family: str):
    """Load unified per-family R^2 matrices and feature metadata."""
    base = config.results_dir / 'feature' / 'component' / domain / family
    m = pd.read_csv(base / 'r2_drop_mean.csv')
    s = pd.read_csv(base / 'r2_drop_std.csv')
    cols = list(m.columns)
    names = [c.split('::', 1)[1] if '::' in c else c for c in cols]
    groups = [c.split('::', 1)[0] if '::' in c else 'unknown' for c in cols]
    return m.values, s.values, names, groups

def load_drop_matrix(domain: str, family: str):
    """Load per-family drop-out ΔR² matrices and feature metadata."""
    base = config.results_dir / 'feature' / 'component' / domain / family
    m = pd.read_csv(base / 'r2_drop_mean.csv')
    s = pd.read_csv(base / 'r2_drop_std.csv')
    cols = list(m.columns)
    names = [c.split('::', 1)[1] if '::' in c else c for c in cols]
    groups = [c.split('::', 1)[0] if '::' in c else 'unknown' for c in cols]
    return m.values, s.values, names, groups

def load_enc_scores(domain: str, family: str, comp_idx: int):
    """Load per-feature drop-out ΔR² for one component (1-indexed)."""
    M, S, names, groups = load_drop_matrix(domain, family)
    i = int(comp_idx) - 1
    row_m = M[i]
    row_s = S[i]
    out = {'visual': {}, 'behavioral': {}, 'semantic': {}}
    for val, sd, nm, gp in zip(row_m, row_s, names, groups):
        out[gp][nm] = {'mean': float(val), 'std': float(sd)}
    return out


def save_fig_std(domain: str, family: str, fig_type: str, fig, *, comp_idx: int = None, extra: str = None,
                 image_high_dpi: bool = False, transparent=None, facecolor=None):
    """Unified save helper under figures/<domain>_analysis/<family>/<fig_type>/...

    - Duplets: always 'bar_duplet.pdf'
    - If comp_idx is provided: 'comp_<N>.<fmt>'
    - Else if extra provided: '<extra>.<fmt>'
    - Else: '<fig_type>.<fmt>'
    """
    if fig is None:
        return
    base = f"{domain.lower()}_analysis"
    out_dir = os.path.join(config.fig_dir, base, family, fig_type)
    os.makedirs(out_dir, exist_ok=True)

    fmt = config.plotting.get('savefig_format', 'pdf')
    if fig_type == 'duplets' and comp_idx is None and not extra:
        fname = 'bar_duplet.pdf'
        fmt = 'pdf'
    elif comp_idx is not None:
        fname = f"comp_{comp_idx + 1}.{fmt}"
    elif extra:
        fname = f"{extra}.{fmt}"
    else:
        fname = f"{fig_type}.{fmt}"

    path = os.path.join(out_dir, fname)
    save_dpi = 600 if image_high_dpi or fig_type in ('images', 'selected') else 300
    save_kwargs = {'bbox_inches': 'tight', 'dpi': save_dpi}
    if transparent is not None:
        save_kwargs['transparent'] = transparent
    if facecolor is not None:
        save_kwargs['facecolor'] = facecolor
    fig.savefig(path, **save_kwargs)
    print(f"    Saved to {path}")
    plt.close(fig)


def get_viz_rank(best_rank: int, available_ranks, override: int = None):
    if override is None:
        return best_rank
    if override in available_ranks:
        return override
    closest = min(available_ranks, key=lambda x: abs(x - override))
    return closest


def load_behav_dim_names():
    """Load behavioral dimension names and create shortened versions."""
    labels_path = os.path.join(config.data_dir, 'things', 'behav_embed', 'variables', 'labels_short.txt')
    with open(labels_path, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    short_names = []
    for name in names:
        # Take everything before first "/" or "-"
        for sep in ['/', '-']:
            if sep in name:
                short_name = name.split(sep)[0]
                break
        else:
            short_name = name
        short_names.append(short_name.replace('_', ' '))

    return short_names
