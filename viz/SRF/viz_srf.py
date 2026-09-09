#!/usr/bin/env python3
"""
Visualization for SRF Analysis Results
Creates figures showing CV scores and component visualizations using the same approach as CCA families.
"""

import os
import sys
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, FuncFormatter
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.feature.core.ridge_utils import ridge_cv_fold
from config.paths import config
from functions.plotting import narrow_figure, setup_style, style_axes, square_panel
from functions.viz_common import save_fig_std

# Border thickness for image mosaics
MOSAIC_BORDER_WIDTH = 4.0

# Components overview grid parameters
OVERVIEW_COLS = 30
OVERVIEW_IMG_SIZE = 100
OVERVIEW_ROW_SPACING = 15  # Pixels of white space between component rows
GUARD_COL_WIDTH = OVERVIEW_IMG_SIZE
GUARD_GAP = 12
GUARD_FOLDS = 5

# Toggle for per-component plots (not selected)
PLOT_ALL = False
FIRST_N_COMPONENTS = 20

# Component selections (1-indexed) per family for SRF
SELECTED_COMPONENTS = {
    'all':    [1, 10, 20, 47],
    'human':  [13, 27, 28, 33],
    'monkey': [12, 42, 46, 53],
}

# Grid size for selected component image mosaics
SELECTED_ROWS = 2
SELECTED_COLS = 6
RSM_TOP_N = 200

# Subsampling settings
TOP_PCT = 1.0  # Percentile for subsampling top images

def subsample_top_pct(vec, n_cells, pct=TOP_PCT, pos=True):
    n_top = max(1, int(len(vec) * pct / 100))
    sorted_idx = np.argsort(vec)[::-1] if pos else np.argsort(vec)
    top_cand = sorted_idx[:n_top]
    return np.random.choice(top_cand, size=min(n_cells, len(top_cand)), replace=False)

def load_srf_results():
    """Load comprehensive SRF analysis results."""
    output_dir = os.path.join(config.results_dir, 'srf_results')
    results_file = os.path.join(output_dir, 'srf_all_results.pkl')

    if not os.path.exists(results_file):
        print(f"ERROR: SRF results not found at {results_file}")
        print("Please run model/SRF/SRF.py first.")
        return None

    with open(results_file, 'rb') as f:
        data = pickle.load(f)

    # Check which families are available
    available_families = list(data['results'].keys())
    all_families = ['all', 'human', 'monkey']
    missing_families = [f for f in all_families if f not in available_families]

    print(f"Available families: {available_families}")
    if missing_families:
        print(f"Missing families: {missing_families}")
        print("(These can be added by re-running model/SRF/SRF.py)")

    return data

def plot_cv_scores(results, fig_dir):
    """Create narrow figures showing CV scores for each family following gridsearch style."""
    setup_style()

    # Color palette from config
    color_palette = {
        'all': config.plotting.get('shared_color', '#81B7B3'),
        'human': config.plotting.get('human_color', '#7c5799'),
        'monkey': config.plotting.get('monkey_color', '#bda855')
    }

    for family_name, result in results.items():
        cv_df = result['cv_results']
        grouped = cv_df.groupby('rank')['score'].agg(['mean', 'std']).reset_index()

        fig = narrow_figure()
        ax = fig.add_axes([.1, .1, .85, .78])

        # Get data
        rank_vals = grouped['rank'].values
        score_vals = grouped['mean'].values
        std_vals = grouped['std'].values

        # Set y-axis range with padding (±20% around min/max)
        y_min, y_max = (score_vals - std_vals).min(), (score_vals + std_vals).max()
        y_range = y_max - y_min
        y_padding = y_range * 0.2
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Plot the main line with error bars in black (like gridsearch)
        ax.errorbar(rank_vals, score_vals, yerr=std_vals,
                   marker='o', linestyle='-', color='black', linewidth=3,
                   markersize=9, capsize=4, capthick=1.5, elinewidth=2, zorder=5)

        ax.set_xlabel('Rank')
        ax.set_ylabel('CV Score (MSE)')
        ax.set_title(f'{family_name.replace("_", " ").title()} SRF')

        # Format x-axis with integer ticks and set range starting from 10
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.set_xlim(left=10)

        # Format y-axis to show appropriate precision
        def y_formatter(x, pos):
            if abs(x) < 0.01:
                return f'{x:.2e}'
            else:
                return f'{x:.3f}'

        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Mark best rank with family-specific color
        best_rank = result['best_rank']
        family_color = color_palette.get(family_name, 'steelblue')
        ax.axvline(best_rank, color=family_color, linestyle='--', alpha=0.8,
                   linewidth=3.5, zorder=6)

        style_axes(ax)

        save_fig_std('srf', family_name, 'cv_scores', fig)


def _rank_curve(result):
    if 'rank_scores' in result and result['rank_scores'] is not None:
        df = result['rank_scores'].copy()
        if {'candidate_rank', 'val_mse_mean'}.issubset(df.columns):
            yerr_col = 'val_mse_sem' if 'val_mse_sem' in df.columns else 'val_mse_std'
            yerr = df[yerr_col].fillna(0.0).to_numpy(float) if yerr_col in df.columns else None
            return (
                df['candidate_rank'].to_numpy(int),
                df['val_mse_mean'].to_numpy(float),
                yerr,
            )

    cv_df = result['cv_results']
    grouped = cv_df.groupby('rank')['score'].agg(['mean', 'sem']).reset_index()
    return (
        grouped['rank'].to_numpy(int),
        grouped['mean'].to_numpy(float),
        grouped['sem'].fillna(0.0).to_numpy(float),
    )


def plot_rank_search_grid(results, fig_dir):
    """Save one combined rank-CV grid across SRF families."""
    setup_style()
    family_order = [fam for fam in ('monkey', 'human', 'all') if fam in results]
    if not family_order:
        return

    color_palette = {
        'all': config.plotting.get('shared_color', '#81B7B3'),
        'human': config.plotting.get('human_color', '#7c5799'),
        'monkey': config.plotting.get('monkey_color', '#bda855'),
    }

    fig, axes = plt.subplots(
        1,
        len(family_order),
        figsize=(3.0 * len(family_order), 2.4),
        sharey=False,
        dpi=300,
    )
    axes = np.atleast_1d(axes)

    for ax, family_name in zip(axes, family_order):
        result = results[family_name]
        ranks, mse, yerr = _rank_curve(result)
        color = color_palette.get(family_name, 'steelblue')
        ax.errorbar(
            ranks,
            mse,
            yerr=yerr,
            marker='o',
            linestyle='-',
            color='black',
            ecolor='black',
            linewidth=2.0,
            markersize=4.5,
            capsize=2.5,
            elinewidth=1.2,
            zorder=4,
        )
        best_rank = int(result['best_rank'])
        ax.axvline(best_rank, color=color, linestyle='--', linewidth=2.0, zorder=5)
        ax.scatter([best_rank], [float(result['best_score'])], color=color, s=36, zorder=6)
        ax.set_title(family_name.replace('_', ' ').title())
        ax.set_xlabel('Rank')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2e}'))
        style_axes(ax)

    axes[0].set_ylabel('Held-out MSE')
    fig.tight_layout(w_pad=1.2)

    out_dir = os.path.join(fig_dir, 'rank_search')
    os.makedirs(out_dir, exist_ok=True)
    fmt = config.plotting.get('savefig_format', 'pdf')
    out_path = os.path.join(out_dir, f'cv_mse_grid.{fmt}')
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"    Saved to {out_path}")


def get_viz_rank(family_name, best_rank, available_ranks):
    """Get visualization rank from config or use CV best, ensuring it's available."""
    custom_ranks = config.hyperparameters.get('srf', {}).get('custom_ranks', {})
    target_rank = custom_ranks.get(family_name, best_rank)

    # Find closest available rank
    if target_rank in available_ranks:
        return target_rank
    else:
        closest_rank = min(available_ranks, key=lambda x: abs(x - target_rank))
        if target_rank != best_rank:
            print(f"  Custom rank {target_rank} not available, using closest: {closest_rank}")
        return closest_rank

def _square_and_resize(path, target):
    """Load → center-crop to a square → resize to `target`×`target` px."""
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, upper = (w - side) // 2, (h - side) // 2
        im = im.crop((left, upper, left + side, upper + side))
        im = im.resize((target, target), Image.BILINEAR)

        # Add dark grey inward border
        im_arr = np.array(im)
        thick = max(2, target // 60)  # Scale border with image size
        border_color = (64, 64, 64)  # Dark grey

        # Top and bottom borders
        im_arr[:thick, :] = border_color
        im_arr[-thick:, :] = border_color

        # Left and right borders
        im_arr[:, :thick] = border_color
        im_arr[:, -thick:] = border_color

        return Image.fromarray(im_arr)
    except:
        # Return gray square with border if image can't be loaded
        blank = np.full((target, target, 3), 128, np.uint8)
        thick = max(2, target // 60)
        border_color = (64, 64, 64)
        blank[:thick, :] = border_color
        blank[-thick:, :] = border_color
        blank[:, :thick] = border_color
        blank[:, -thick:] = border_color
        return Image.fromarray(blank)

def _build_mosaic(paths, rows, cols, px):
    """
    Return a single numpy array of shape (rows*px, cols*px, 3)
    that contains all images tiled left-to-right, top-to-bottom.
    Missing tiles get mid-grey.
    """
    blank = np.full((px, px, 3), 128, np.uint8)
    tiles = [_square_and_resize(p, px) if p is not None else Image.fromarray(blank)
             for p in paths]
    # Pad with blank tiles if we don't have enough images
    tiles += [Image.fromarray(blank)] * (rows * cols - len(tiles))
    tiles = [np.asarray(t) for t in tiles]

    grid = np.vstack([
        np.hstack(tiles[r * cols:(r + 1) * cols])
        for r in range(rows)
    ])
    return grid

def _border(ax, color='black', lw=3):
    """
    Add a rectangle that runs exactly around the axes (0‒1 in Axes coords).
    Because it lives in ax.transAxes it is immune to GridSpec rounding.
    """
    ax.add_patch(
        Rectangle((0, 0), 1, 1,
                  transform=ax.transAxes,  # <- axes-relative
                  facecolor='none',
                  edgecolor=color,
                  linewidth=lw,
                  clip_on=False)           # let the stroke stick out
    )

def _guard_tile(val, color, width, height):
    v = float(np.clip(val, -0.2, 1.0)) if np.isfinite(val) else np.nan
    base = np.asarray(mpl.colors.to_rgb(color))
    white = np.ones(3)
    if not np.isfinite(v):
        rgb = np.ones(3) * 0.9
    elif v <= 0.0:
        t = min(1.0, -v)
        rgb = white * (1.0 - 0.35 * t)
    else:
        t = min(1.0, v)
        mix = 0.85 * t
        rgb = white * (1.0 - mix) + base * mix
    tile = (rgb * 255.0).astype(np.uint8)
    arr = np.tile(tile, (height, width, 1))
    b = max(1, width // 60)
    arr[:b, :, :] = 64
    arr[-b:, :, :] = 64
    arr[:, :b, :] = 64
    arr[:, -b:, :] = 64
    return arr


def _guard_text_color(val):
    if not np.isfinite(val):
        return 'black'
    return 'white' if val >= 0.55 else 'black'


def ridge_guard(W, cca_res, n_folds=GUARD_FOLDS):
    if cca_res is None:
        return None
    comps = cca_res.get('components') or {}
    views = sorted(comps)
    if not views:
        return None
    humans = [v for v in views if v.lower().startswith('human')]
    monkeys = [v for v in views if v.lower().startswith('monkey')]
    want = humans + monkeys
    if not want:
        return None
    mats = {v: np.asarray(comps[v], float) for v in want}
    n_comp = W.shape[1]
    n_jobs = int(config.analysis.get('n_jobs', 4))

    def _fit(idx):
        y = W[:, idx]
        return {v: ridge_cv_fold(mats[v], y, n_folds=n_folds)[0] for v in want}

    rows = Parallel(n_jobs=n_jobs)(delayed(_fit)(i) for i in range(n_comp))
    per_view = {v: np.array([row[v] for row in rows], float) for v in want}

    labels, colors, vals = [], [], []
    if humans:
        labels.append('Human')
        colors.append(config.plotting.get('human_color', '#7c5799'))
        vals.append(np.nanmean([per_view[v] for v in humans], axis=0))
    if monkeys:
        labels.append('Monkey')
        colors.append(config.plotting.get('monkey_color', '#bda855'))
        vals.append(np.nanmean([per_view[v] for v in monkeys], axis=0))
    if not vals:
        return None
    guard_vals = np.stack(vals, axis=1)
    return {
        'values': guard_vals,
        'labels': labels,
        'colors': colors,
        'per_view': per_view,
        'views': want,
    }


def _plot_distribution_positive_only(ax, vec, order, ipos, inv, pos_cand=None):
    """Draw positive-only density curve and brackets for SRF components."""
    vals = vec[order]
    total = len(vals)

    def _smooth(y):
        return gaussian_filter1d(y, sigma=len(y) / 50) if len(y) > 0 else y

    # Only positive side since SRF components are non-negative
    x = np.linspace(0, total - 1, 500)
    y = _smooth(np.interp(x, np.arange(total), vals))
    if y.max() > 0:
        y *= (vals.max() or 1) / y.max()

    for i in range(len(x) - 1):
        frac = x[i] / total
        ax.fill_between(x[i:i + 2], 0, y[i:i + 2], color=cm.Reds(0.3 + 0.7 * frac), lw=0)

    ax.axhline(0, color='k', lw=0)
    ax.set_xlim(0, total)
    y0, y1 = ax.get_ylim()
    dy = y1 - y0
    ax.set_ylim(y0 - 0.05 * dy, y1 + 0.05 * dy)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Brackets & arrows for positive selection
    if pos_cand is not None:
        xp_all = inv[pos_cand]
    else:
        xp_all = inv[ipos]

    xp0, xp1 = xp_all.min(), xp_all.max()

    # Update y-limits after adjusting
    y0, y1 = ax.get_ylim()

    # Get positive color from config
    pos_color = config.plotting.get('positive_color', '#8B0000')

    ax.plot([xp0, xp1], [0, 0], color=pos_color, lw=4)
    ax.plot([xp0, xp0], [0, y1], color=pos_color, lw=4)
    ax.plot([xp1, xp1], [0, y1], color=pos_color, lw=4)

def load_img_simple(path, size):
    """Load and prep single image for overview."""
    try:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((size, size), Image.BILINEAR)

        # Add border
        arr = np.array(im)
        b_color = (64, 64, 64)
        bw = max(1, size // 60)
        arr[:bw, :] = b_color
        arr[-bw:, :] = b_color
        arr[:, :bw] = b_color
        arr[:, -bw:] = b_color

        return Image.fromarray(arr)
    except:
        blank = np.full((size, size, 3), 128, np.uint8)
        return Image.fromarray(blank)

def plot_components_overview(W, all_stims, cats, img_dir, save_path, family_name, viz_rank,
                             cols=OVERVIEW_COLS, img_size=OVERVIEW_IMG_SIZE,
                             imp_mean=None, feature_names=None, feature_groups=None,
                             guard=None):
    """SRF overview: left feature bars + labels (if available) and right image rows (positive-only)."""
    n_comps = W.shape[1]

    guard_vals = guard.get('values') if guard else None
    guard_labels = guard.get('labels') if guard else []
    guard_colors = guard.get('colors') if guard else []
    guard_cols = guard_vals.shape[1] if isinstance(guard_vals, np.ndarray) else 0
    guard_w = guard_cols * GUARD_COL_WIDTH + (GUARD_GAP if guard_cols else 0)
    guard_centers = [GUARD_COL_WIDTH * (i + 0.5) for i in range(guard_cols)]
    guard_gap_tile = np.full((img_size, GUARD_GAP, 3), 255, np.uint8) if guard_cols else None

    # Build image rows (positive-only)
    all_rows = []
    for comp_idx in range(n_comps):
        w = W[:, comp_idx]
        top_idx = subsample_top_pct(w, cols)
        row_tiles = []
        if guard_cols:
            gvals = guard_vals[comp_idx]
            for gval, col in zip(gvals, guard_colors):
                row_tiles.append(_guard_tile(gval, col, GUARD_COL_WIDTH, img_size))
            if guard_gap_tile is not None:
                row_tiles.append(guard_gap_tile.copy())
        for idx in top_idx:
            stim = all_stims[idx]
            cat = cats[idx]
            path = os.path.join(img_dir, cat, f"{stim}.jpg")
            row_tiles.append(np.array(load_img_simple(path, img_size)))
        row = np.hstack(row_tiles)
        all_rows.append(row)
        if comp_idx < n_comps - 1:
            all_rows.append(np.full((OVERVIEW_ROW_SPACING, row.shape[1], 3), 255, np.uint8))

    grid = np.vstack(all_rows)
    H_img, W_img = grid.shape[:2]

    # No importance: compose grid and overlay component numbers
    if imp_mean is None or feature_names is None or feature_groups is None:
        canvas = grid
        guard_x0 = 0
        mosaic_x0 = guard_w
        fig_w = canvas.shape[1] / 100.0
        fig_h = canvas.shape[0] / 100.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        ax.imshow(canvas); ax.axis('off')
        if guard_cols:
            hdr_y = -max(8, GUARD_GAP)
            for gi, lab in enumerate(guard_labels):
                ax.text(guard_x0 + guard_centers[gi], hdr_y, lab, ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
        for i in range(n_comps):
            y0 = i * (img_size + OVERVIEW_ROW_SPACING)
            if guard_cols:
                for gi, val in enumerate(guard_vals[i]):
                    txt = '--' if not np.isfinite(val) else f"{val:.2f}"
                    ax.text(guard_x0 + guard_centers[gi], y0 + img_size * 0.5, txt,
                            ha='center', va='center', fontsize=8, fontweight='bold',
                            color=_guard_text_color(val))
            ax.text(mosaic_x0 + 8, y0 + 14, f'C{i+1}', ha='left', va='top', fontsize=10, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', edgecolor='none', alpha=0.6))
        plt.tight_layout(); plt.savefig(save_path, dpi=180, bbox_inches='tight'); plt.close(); return

    # Build small panels per component (bars + two-column labels)
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FigureCanvasAgg
    import matplotlib.pyplot as _plt

    def _short(name):
        lab = name
        for ch in ['-', '/', ' ', '_']:
            if ch in lab:
                lab = lab.split(ch)[0]
                break
        return lab

    bar_panel_height = img_size
    BAR_PANEL_WIDTH_PX = 240
    BAR_GAP_PX = 30
    vis_range = config.plotting.get('visual_range', [0.9, 0.92])
    beh_range = config.plotting.get('behavioral_range', [0.6, 0.62])

    vis_shades = [cm.bone(t) for t in np.linspace(vis_range[0], vis_range[1], 3)]
    beh_shades = [cm.bone(t) for t in np.linspace(beh_range[0], beh_range[1], 3)]

    panels = []
    n_pan = min(n_comps, imp_mean.shape[0])
    for comp_idx in range(n_pan):
        vals = np.array(imp_mean[comp_idx], dtype=float)
        idx_vis = [i for i, g in enumerate(feature_groups) if g == 'visual']
        idx_beh = [i for i, g in enumerate(feature_groups) if g == 'behavioral']
        top_vis = sorted(idx_vis, key=lambda i: vals[i], reverse=True)[:3]
        top_beh = sorted(idx_beh, key=lambda i: vals[i], reverse=True)[:3]
        bar_vals = [vals[i] for i in top_vis] + [vals[i] for i in top_beh]
        bar_cols = [vis_shades[k][:3] for k in range(len(top_vis))] + [beh_shades[k][:3] for k in range(len(top_beh))]
        vis_labels = [feature_names[i] for i in top_vis]
        beh_labels = [feature_names[i] for i in top_beh]

        fig_w_in = BAR_PANEL_WIDTH_PX / 100.0
        fig_h_in = bar_panel_height / 100.0
        _fig = _plt.figure(figsize=(fig_w_in, fig_h_in), dpi=100)
        bar_ax = _fig.add_axes([0.08, 0.15, 0.50, 0.70])
        txt_ax = _fig.add_axes([0.63, 0.15, 0.34, 0.70])
        x = np.arange(len(bar_vals))
        bar_ax.bar(x, bar_vals, color=bar_cols, edgecolor='black', linewidth=0.8)
        bar_ax.set_xticks([])
        bar_ax.tick_params(axis='y', labelsize=6, length=3)
        bar_ax.set_ylabel(r'$\Delta R^2$', fontsize=7)
        for sp in ['top', 'right']:
            bar_ax.spines[sp].set_visible(False)
        ymax = max(bar_vals) * 1.15 if len(bar_vals) and max(bar_vals) > 0 else 1.0
        bar_ax.set_ylim(0, ymax)
        bar_ax.margins(x=0)

        txt_ax.axis('off')
        txt_ax.text(0.00, 0.97, 'Visual', fontsize=8, fontweight='bold', va='top')
        txt_ax.text(0.55, 0.97, 'Behav', fontsize=8, fontweight='bold', va='top')
        for r in range(max(len(vis_labels), len(beh_labels))):
            y = 0.90 - r * 0.35
            if r < len(vis_labels):
                txt_ax.text(0.00, y, _short(vis_labels[r]), fontsize=8, fontweight='bold', va='top')
            if r < len(beh_labels):
                txt_ax.text(0.55, y, _short(beh_labels[r]), fontsize=8, fontweight='bold', va='top')

        canvas = _FigureCanvasAgg(_fig)
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        w, h = canvas.get_width_height()
        panel = buf.reshape(h, w, 3)
        _plt.close(_fig)
        if panel.shape[0] != bar_panel_height or panel.shape[1] != BAR_PANEL_WIDTH_PX:
            panel = np.array(Image.fromarray(panel).resize((BAR_PANEL_WIDTH_PX, bar_panel_height), Image.BILINEAR))
        panels.append(panel)

    total_w = BAR_PANEL_WIDTH_PX + BAR_GAP_PX + W_img
    canvas = np.full((H_img, total_w, 3), 255, np.uint8)
    for comp_idx in range(n_comps):
        y0 = comp_idx * (img_size + OVERVIEW_ROW_SPACING)
        if comp_idx < len(panels):
            canvas[y0:y0 + img_size, :BAR_PANEL_WIDTH_PX] = panels[comp_idx]
    x_cursor = BAR_PANEL_WIDTH_PX + BAR_GAP_PX
    canvas[:, x_cursor:x_cursor + W_img] = grid

    fig_w = total_w / 100.0
    fig_h = H_img / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(canvas); ax.axis('off')
    guard_x0 = x_cursor
    mosaic_x0 = x_cursor + guard_w
    if guard_cols:
        hdr_y = -max(8, GUARD_GAP)
        for gi, lab in enumerate(guard_labels):
            ax.text(guard_x0 + guard_centers[gi], hdr_y, lab, ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
    for i in range(n_comps):
        y0 = i * (img_size + OVERVIEW_ROW_SPACING)
        if guard_cols:
            for gi, val in enumerate(guard_vals[i]):
                txt = '--' if not np.isfinite(val) else f"{val:.2f}"
                ax.text(guard_x0 + guard_centers[gi], y0 + img_size * 0.5, txt,
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color=_guard_text_color(val))
        ax.text(mosaic_x0 + 8, y0 + 14, f'C{i+1}', ha='left', va='top', fontsize=10, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', edgecolor='none', alpha=0.6))
    plt.tight_layout(); plt.savefig(save_path, dpi=180, bbox_inches='tight'); plt.close()

def plot_comp_imgs_positive_only(comp_idx, W, stims, cats, img_dir=None, title=None, rows=4, cols=6,
                                img_px=180, dpi=300, subsample=True, sub_n=100, random_seed=42):
    """Plot SRF component images with positive-only grid (no negative box)"""
    if img_dir is None:
        img_dir = os.path.join(config.data_dir, 'things', 'images')

    if random_seed is not None:
        np.random.seed(random_seed)

    vec = W[:, comp_idx]
    n_cells = rows * cols
    order = np.argsort(vec)

    # Choose only positive examples (SRF components are non-negative)
    if subsample:
        n_total = len(vec)
        n_cand = min(sub_n, n_total // 2)
        pos_cand = order[-n_cand:]
        ipos = np.random.choice(pos_cand, size=min(n_cells, len(pos_cand)), replace=False)
    else:
        ipos = subsample_top_pct(vec, n_cells)

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    if subsample:
        xp_all = inv[pos_cand]
    else:
        xp_all = inv[ipos]

    # Figure with only positive grid and distribution (no negative)
    fw = cols * img_px / dpi  # Keep thumbnails square
    fh = (rows + 0.5) * img_px / dpi  # Only positive grid + distribution
    fig = plt.figure(figsize=(fw, fh), dpi=dpi, constrained_layout=False)
    gs_master = GridSpec(
        2, 1,
        height_ratios=[rows, 0.5],  # positive grid – distribution
        hspace=0,
        figure=fig
    )

    # 1) Positive grid (top) - build mosaic
    pos_paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ipos]
    pos_mosaic = _build_mosaic(pos_paths, rows, cols, img_px)
    ax_top = fig.add_subplot(gs_master[0])
    ax_top.imshow(pos_mosaic, interpolation='nearest', aspect='auto')
    ax_top.axis('off')
    ax_top.set_xlim(-0.5, pos_mosaic.shape[1] - 0.5)
    ax_top.set_ylim(pos_mosaic.shape[0] - 0.5, -0.5)
    _border(ax_top, color=config.plotting.get('positive_color', '#8B0000'),
            lw=MOSAIC_BORDER_WIDTH)

    # 2) Distribution axis (bottom)
    ax_center = fig.add_subplot(gs_master[1])
    ax_center.set_xlim(-0.5, pos_mosaic.shape[1] - 0.5)
    ax_center.margins(x=0, y=0)  # no padding left/right or top/bottom
    _plot_distribution_positive_only(ax_center, vec, order, ipos, inv,
                                   pos_cand if subsample else None)
    ax_center.set_yticks([])
    ax_center.set_xticks([])

    if title:
        fig.suptitle(f"{title} – Component {comp_idx+1}", fontsize=14, y=0.99)

    return fig

def save_srf_fig(fig, fig_type, family_name, comp_idx=None, extra_info=""):
    """Save SRF analysis figure with structured path and name in srf_analysis folder."""
    if not fig:
        return

    fig_dir = os.path.join(config.fig_dir, 'srf_analysis', family_name, fig_type)
    os.makedirs(fig_dir, exist_ok=True)

    fmt = config.plotting.get('savefig_format', 'pdf')

    # Special-case duplets to mirror CCA naming
    if fig_type == 'duplets' and not extra_info and comp_idx is None:
        fname = 'bar_duplet.pdf'
        fmt = 'pdf'
    elif comp_idx is not None:
        fname = f'comp_{comp_idx+1}.{fmt}'
    elif extra_info:
        fname = f'{extra_info}.{fmt}'
    else:
        fname = f'{fig_type}.{fmt}'

    fig_path = os.path.join(fig_dir, fname)

    # Use higher DPI for image plots
    save_dpi = 600 if fig_type == 'images' else 300
    fig.savefig(fig_path, bbox_inches='tight', dpi=save_dpi)
    print(f"    Saved to {fig_path}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────
# RSM / RDM visualizations

def _load_cca_state():
    """Load the CCA state used to build RSMs (robust to two common locations)."""
    # Prefer explicit cache_file if present
    state_fp = getattr(config, 'cache_file', None)
    if state_fp is None or not os.path.exists(state_fp):
        # Fallback to legacy path used in viz_cca_families
        maybe_fp = os.path.join(config.results_dir, 'cca_state.pkl')
        state_fp = maybe_fp if os.path.exists(maybe_fp) else None
    if not state_fp:
        print("ERROR: Could not locate CCA state file (config.cache_file or results/cca_state.pkl)")
        return None
    try:
        with open(state_fp, 'rb') as f:
            s = pickle.load(f)
        return s
    except Exception as e:
        print(f"ERROR: Failed to read CCA state from {state_fp}: {e}")
        return None

def _compute_mean_rsm(cca_results, fisher=True, view_weights=None):
    """Match SRF.compute_mean_rsm: Fisher-z average of per-view cosine RSMs, rescaled to [0,1]."""
    from sklearn.metrics.pairwise import cosine_similarity
    views = sorted(cca_results['components'])
    sims = []
    for v in views:
        comps = cca_results['components'][v]
        sim = cosine_similarity(comps)
        sims.append(sim)
    sims = np.stack(sims)

    if view_weights is None:
        view_weights = np.ones(len(views)) / len(views)
    else:
        view_weights = np.asarray(view_weights) / np.sum(view_weights)

    if fisher:
        sims_clipped = np.clip(sims, -0.9999, 0.9999)
        sims_z = np.arctanh(sims_clipped)
        mean_z = (view_weights[:, None, None] * sims_z).sum(axis=0)
        mean_sim = np.tanh(mean_z)
    else:
        mean_sim = (view_weights[:, None, None] * sims).sum(axis=0)

    # Rescale to [0,1]
    mn, mx = mean_sim.min(), mean_sim.max()
    denom = (mx - mn) if (mx - mn) != 0 else 1.0
    mean_sim = (mean_sim - mn) / denom
    return mean_sim

def _plot_matrix(mat, title=None, cmap='magma', vmin=0.0, vmax=1.0, *, bare=False, outline=True, top_n=None):
    """Plot a square matrix with magma colormap and compact styling.

    bare=True suppresses title, axis labels, and colorbar (useful for insets).
    outline=True draws a black rectangle around the matrix area.
    top_n: if provided, crops to the top-left top_n×top_n submatrix.
    """
    if top_n is not None and top_n > 0:
        mat = mat[:top_n, :top_n]
    fig, ax = square_panel()
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    if not bare:
        if title:
            ax.set_title(title)
        ax.set_xlabel('Stimuli')
        ax.set_ylabel('Stimuli')
    # Hide tick labels for compactness
    ax.set_xticks([]); ax.set_yticks([])
    style_axes(ax)
    # Colorbar (thin) unless bare
    if not bare:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(length=3)
    # Black outline around matrix area
    if outline:
        h, w = mat.shape[:2]
        ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, edgecolor='black', linewidth=4.0, zorder=5))
    return fig


# ──────────────────────────────────────────────────────────────────────
# RSM / RDM visualizations

def visualize_rsms(results):
    """Compute and save the RSM (only) for each family, matching SRF settings.

    We output only the RSM because SRF operates on a similarity matrix.
    """
    setup_style()
    s = _load_cca_state()
    if s is None:
        print("Skipping RSM/RDM plots: no CCA state available")
        return

    fisher_z = config.hyperparameters.get('srf', {}).get('use_fisher_z', True)

    fam_map = {
        'all': s.get('cca_all'),
        'human': s.get('cca_hum'),
        'monkey': s.get('cca_mon'),
    }

    for fam_key in results.keys():
        cca_res = fam_map.get(fam_key)
        if cca_res is None:
            print(f"RSM: missing CCA results for family '{fam_key}', skipping")
            continue

        mean_rsm = _compute_mean_rsm(cca_res, fisher=fisher_z)

        # RSM figure (bare inset style, cropped to first RSM_TOP_N entries)
        fig_rsm = _plot_matrix(mean_rsm, title=None, cmap='magma', vmin=0.0, vmax=1.0,
                               bare=True, outline=True, top_n=RSM_TOP_N)
        save_fig_std('srf', fam_key, 'rsm', fig_rsm, extra=f"{fam_key}_RSM")

def visualize_selected_components(results, all_stims, fig_dir,
                                  selection_map, rows=4, cols=6):
    """Visualize user-selected components (positive-only) for each family.

    selection_map: dict with keys in {'all','human','monkey'} and values as 1-indexed lists.
    rows, cols: grid size for the positive-only mosaics.
    """
    cats = ['_'.join(s.split('_')[:-1]) if '_' in s else s for s in all_stims]
    img_dir = os.path.join(config.data_dir, 'things', 'images')

    for fam_key, result in results.items():
        # Normalize family key to our selection keys
        sel_key = fam_key  # expected to already be 'all' | 'human' | 'monkey'
        chosen = selection_map.get(sel_key, []) or []
        if not chosen:
            continue

        available_ranks = list(result['components'].keys())
        viz_rank = get_viz_rank(fam_key, result['best_rank'], available_ranks)
        W = result['components'][viz_rank]

        print(f"\nSelected components for {fam_key} (rank {viz_rank}): {chosen}")

        # Save selected component grids
        for comp_id in chosen:
            # Convert to 0-indexed; skip invalid
            ci = int(comp_id) - 1
            if ci < 0 or ci >= W.shape[1]:
                print(f"  Warning: component {comp_id} out of range (1..{W.shape[1]}), skipping")
                continue

            title = f"{fam_key.upper()} Component {comp_id} (rank {viz_rank})"
            fig = plot_comp_imgs_positive_only(
                ci, W, all_stims, cats,
                img_dir=img_dir,
                title=title,
                rows=rows, cols=cols,
                img_px=360, dpi=300
            )
            # Save into a dedicated 'selected' folder
            save_fig_std('srf', fam_key, 'selected', fig, comp_idx=ci, image_high_dpi=True)

def visualize_components(results, all_stims, fig_dir, cca_state=None):
    """Visualize SRF components using selected ranks."""
    cats = ['_'.join(s.split('_')[:-1]) if '_' in s else s for s in all_stims]
    cca_all = cca_state.get('cca_all') if isinstance(cca_state, dict) else None

    for family_name, result in results.items():
        available_ranks = list(result['components'].keys())
        viz_rank = get_viz_rank(family_name, result['best_rank'], available_ranks)
        W = result['components'][viz_rank]

        custom_ranks = config.hyperparameters.get('srf', {}).get('custom_ranks', {})
        is_custom = family_name in custom_ranks
        rank_info = f"custom rank {viz_rank}" if is_custom else f"best rank {viz_rank}"

        print(f"Visualizing {family_name} components ({rank_info})")

        # Create components overview
        print(f"  Creating components overview (guard-enhanced where applicable)...")
        img_dir = os.path.join(config.data_dir, 'things', 'images')
        overview_path = os.path.join(fig_dir, family_name, 'overview', 'components_overview.png')
        os.makedirs(os.path.dirname(overview_path), exist_ok=True)
        guard_data = None
        if family_name == 'all' and cca_all is not None:
            print("    Computing ridge guard (cross-species views)...")
            guard_data = ridge_guard(W, cca_all)
            if guard_data:
                guard_dir = os.path.dirname(overview_path)
                df_guard = pd.DataFrame({'Component': np.arange(1, guard_data['values'].shape[0] + 1)})
                for gi, lab in enumerate(guard_data['labels']):
                    df_guard[f"{lab}_R2"] = guard_data['values'][:, gi]
                guard_csv = os.path.join(guard_dir, 'ridge_guard.csv')
                df_guard.to_csv(guard_csv, index=False)
                mean_vals = [f"{lab}:{np.nanmean(guard_data['values'][:, gi]):.3f}" for gi, lab in enumerate(guard_data['labels'])]
                print(f"      Guard means {' | '.join(mean_vals)}")
                print(f"      Guard summary saved to {guard_csv}")
        need_overview = guard_data is not None or not os.path.exists(overview_path)
        if need_overview:
            plot_components_overview(W, all_stims, cats, img_dir, overview_path, family_name, viz_rank, guard=guard_data)
            print(f"    Saved overview to {overview_path}")
        else:
            print(f"    Overview exists, skipping: {overview_path}")

        # Individual component plots
        if PLOT_ALL:
            n_viz = min(FIRST_N_COMPONENTS, W.shape[1])
            for comp_idx in range(n_viz):
                print(f"  Component {comp_idx+1}: Generating image plot...")
                fig_images = plot_comp_imgs_positive_only(
                    comp_idx, W, all_stims, cats,
                    img_dir=img_dir,
                    title=f"{family_name.upper()} Component {comp_idx+1} (rank {viz_rank})",
                    img_px=360, dpi=300
                )
                save_fig_std('srf', family_name, 'images', fig_images, comp_idx=comp_idx, image_high_dpi=True)
            print(f"  Saved {n_viz} component plots for {family_name}")
        else:
            print(f"  Skipping 'plot all' individual component plots (PLOT_ALL=False)")

def create_rank_summary(results, fig_dir):
    """Create rank selection summary table."""
    table_data = []
    custom_ranks = config.hyperparameters.get('srf', {}).get('custom_ranks', {})

    for family_name, result in results.items():
        available_ranks = list(result['components'].keys())
        viz_rank = get_viz_rank(family_name, result['best_rank'], available_ranks)
        is_custom = family_name in custom_ranks

        table_data.append({
            'Family': family_name.upper(),
            'CV Best Rank': result['best_rank'],
            'Viz Rank': viz_rank,
            'Custom': 'Yes' if is_custom else 'No',
            'CV Score': f"{result['best_score']:.4f}"
        })

    df = pd.DataFrame(table_data)

    summary_dir = os.path.join(fig_dir, 'summaries')
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, 'rank_summary.csv')
    df.to_csv(csv_path, index=False)

    print("\n" + "="*50)
    print("RANK SELECTION SUMMARY")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    print(f"Table saved: {csv_path}")

def main():
    """Main visualization function."""
    print("Loading SRF results...")
    data = load_srf_results()
    if data is None: return

    results = data['results']
    all_stims = data['all_stims']

    # Check available families
    available_families = list(results.keys())
    if not available_families:
        print("ERROR: No families found in results.")
        return

    fig_dir = os.path.join(config.fig_dir, 'srf_analysis')
    if os.path.exists(fig_dir):
        import shutil
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Creating visualizations in {fig_dir}...")
    print(f"Processing {len(available_families)} families: {available_families}")

    print("\nStep 1: CV scores (full rank range)")
    plot_cv_scores(results, fig_dir)
    plot_rank_search_grid(results, fig_dir)

    print("\nStep 2: Component visualization (selected ranks)")
    cca_state = _load_cca_state()
    visualize_components(results, all_stims, fig_dir, cca_state=cca_state)

    # Selected components (positive-only) if not plotting all
    if not PLOT_ALL:
        print("\nStep 3: Selected components (positive-only)")
        visualize_selected_components(
            results, all_stims, fig_dir,
            selection_map=SELECTED_COMPONENTS,
            rows=SELECTED_ROWS, cols=SELECTED_COLS
        )
    else:
        print("\nStep 3: Skipping selected components (PLOT_ALL=True)")

    print("\nStep 4: RSM matrices")
    visualize_rsms(results)

    print("\nStep 5: Summary")
    create_rank_summary(results, fig_dir)

    print(f"\nVisualization complete. Plots saved in: {fig_dir}")
    print("To change viz rank: edit config.toml [hyperparameters.srf.custom_ranks]")

    # Summary of what was processed
    all_families = ['all', 'human', 'monkey']
    missing_families = [f for f in all_families if f not in available_families]
    if missing_families:
        print(f"\nNote: Missing families {missing_families} - run model/SRF/SRF.py to complete")

if __name__ == "__main__":
    main()
