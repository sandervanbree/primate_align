# Visualization for CCA component analysis and interpretation
# Updates in this version:
# - Feature labels above duplet bars are slightly larger (only those labels), and spaced a bit further from error bars
# - Error bars and labels are perfectly centered on bars
# - No whitespace between bars and x-axis (bars sit flush on baseline)
# - Tight layout to prevent y-axis/labels from bleeding out of the canvas
# - Support selecting multiple families at the top; loops across each
# - Duplet bar figure is saved to .../<family>/duplets/bar_duplet.pdf

import os, sys, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import matplotlib as mpl, matplotlib.gridspec as gridspec, matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from functions.plotting import (
    setup_style, setup_style_ridge, style_axes, format_axes,
    narrow_figure, medium_figure, wide_figure, square_figure, large_figure, square_panel,
    add_panel_axes
)
from config.paths import config
from functions.viz_common import save_fig_std, plot_duplets_from_enc_scores, make_legend, load_enc_scores, load_enc_matrix

# Style: default sizes; we will only bump the feature labels above the duplet bars.
setup_style(rc={'axes.ymargin': 0.0, 'axes.xmargin': 0.0})
np.random.seed(43)

# ──────────────────────────────────────────────────────────────────────
# Configuration

# Select one or more families to process. Options: 'cross-species', 'human', 'monkey'
TARGET_FAMILIES = ['cross-species', 'human', 'monkey']

# How many components to visualize for each figure family:
# - GRID_COMP_RANGE: per-component grids (Ridge R² + top/bottom image mosaics)
# - DUPLET_COMP_RANGE: feature-duplet bars from joint importance
GRID_COMP_RANGE   = 20
DUPLET_COMP_RANGE = 6

SHOW_INLINE_IMAGE_GRIDS = True
BONE_SEM_RANGE = (0.25, 0.45)
BONE_VIS_RANGE = (0.65, 0.75)
MOSAIC_BORDER_WIDTH = 4.0
OVERVIEW_COLS = 30
OVERVIEW_IMG_SIZE = 60
OVERVIEW_ROW_SPACING = 10
MAX_OVERVIEW_COMPONENTS = 20

ORDER = ['human_01', 'human_02', 'human_03', 'monkey_F', 'monkey_N']
LABELS = {'human_01': 'Human 1', 'human_02': 'Human 2', 'human_03': 'Human 3',
          'monkey_F': 'Monkey F', 'monkey_N': 'Monkey N'}

# Overview layout settings
INCLUDE_DUPLET_BARS = True  # toggle to add per-component top ΔR² bars into overview
BAR_PANEL_WIDTH_PX = 240    # widened for clearer bars + label table
BAR_GAP_PX = 30             # a bit more gap

# Selected components (1-indexed), defaults to manuscript-facing panels.
CCA_SELECTED = { 'cross-species': [1,2,3,4,5,6], 'human': [1,2,3,4], 'monkey': [1,2,3,4] }

# Subsampling settings
TOP_PCT = 1.0  # Percentile for subsampling top images

# ──────────────────────────────────────────────────────────────────────
# Load cached CCA results
state_fp = os.path.join(config.results_dir, 'cca_state.pkl')
if not os.path.exists(state_fp):
    raise FileNotFoundError(f"CCA state file not found: {state_fp}")

print(f"Loading cached CCA results from {state_fp}...")
with open(state_fp, 'rb') as f:
    s = pickle.load(f)

cca_all   = s['cca_all']
cca_hum   = s['cca_hum']
cca_mon   = s['cca_mon']
stims_all = s['all_stims']
X_all     = s['X_all']
X_monkey  = s['X_monkey']
X_human   = s['X_human']

print("Loaded cached CCA results successfully.")

# Extract canonical components and prepare data
views_all = sorted(cca_all['components'].keys())
views_hum = sorted(cca_hum['components'].keys())
views_mon = sorted(cca_mon['components'].keys())

comps_all = np.mean([cca_all['components'][v] for v in views_all], axis=0)
comps_hum = np.mean([cca_hum['components'][v] for v in views_hum], axis=0)
comps_mon = np.mean([cca_mon['components'][v] for v in views_mon], axis=0)

stims = stims_all
cats = ['_'.join(sv.split('_')[:-1]) if '_' in sv else sv for sv in stims]
img_base_dir = os.path.join(config.data_dir, 'things', 'images')

print(f"Loaded {len(stims)} stimuli, {len(set(cats))} categories")
print(f"Components shape - All: {comps_all.shape}, Human: {comps_hum.shape}, Monkey: {comps_mon.shape}")

# ──────────────────────────────────────────────────────────────────────
# Helpers

def subsample_top_pct(vec, n_cells, pct=TOP_PCT, pos=True):
    n_top = max(1, int(len(vec) * pct / 100))
    sorted_idx = np.argsort(vec)[::-1] if pos else np.argsort(vec)
    top_cand = sorted_idx[:n_top]
    return np.random.choice(top_cand, size=min(n_cells, len(top_cand)), replace=False)

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
        arr = np.array(im)
        b_color = (64, 64, 64)
        bw = max(1, size // 60)
        arr[:bw, :] = b_color; arr[-bw:, :] = b_color
        arr[:, :bw] = b_color; arr[:, -bw:] = b_color
        return Image.fromarray(arr)
    except Exception:
        blank = np.full((size, size, 3), 128, np.uint8)
        return Image.fromarray(blank)

def plot_overview_cca(comps, all_stims, cats, img_dir, save_path, family_name):
    """Create compact overview showing all CCA components with pos/neg pairs."""
    n_comps = min(comps.shape[1], MAX_OVERVIEW_COMPONENTS)
    rows_per_comp = 2
    spacing_per_comp = OVERVIEW_ROW_SPACING
    total_height = n_comps * (rows_per_comp * OVERVIEW_IMG_SIZE + spacing_per_comp)
    fig_h = total_height / 100
    fig_w = OVERVIEW_COLS * OVERVIEW_IMG_SIZE / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    all_rows, y_positions = [], []
    for comp_idx in range(n_comps):
        vec = comps[:, comp_idx]
        order = np.argsort(vec)

        pos_y_start = len(all_rows) * OVERVIEW_IMG_SIZE
        pos_idx = subsample_top_pct(vec, OVERVIEW_COLS)
        pos_row = np.hstack([
            np.array(load_img_simple(os.path.join(img_dir, cats[i], f"{all_stims[i]}.jpg"), OVERVIEW_IMG_SIZE))
            for i in pos_idx
        ])
        all_rows.append(pos_row)

        neg_y_start = len(all_rows) * OVERVIEW_IMG_SIZE
        neg_idx = subsample_top_pct(vec, OVERVIEW_COLS, pos=False)
        neg_row = np.hstack([
            np.array(load_img_simple(os.path.join(img_dir, cats[i], f"{all_stims[i]}.jpg"), OVERVIEW_IMG_SIZE))
            for i in neg_idx
        ])
        all_rows.append(neg_row)

        y_positions.append((pos_y_start, pos_y_start + OVERVIEW_IMG_SIZE, neg_y_start, neg_y_start + OVERVIEW_IMG_SIZE))
        if comp_idx < n_comps - 1:
            spacing = np.full((spacing_per_comp, OVERVIEW_COLS * OVERVIEW_IMG_SIZE, 3), 255, np.uint8)
            all_rows.append(spacing)

    full_grid = np.vstack(all_rows)

    # Compose canvas without ridge panel
    canvas = full_grid
    mosaic_x0 = 0

    fig_w = canvas.shape[1] / 100.0
    fig_h = canvas.shape[0] / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(canvas); ax.axis('off')
    for (py0, _, ny0, _) in y_positions:
        ax.add_patch(Rectangle((mosaic_x0, py0), OVERVIEW_COLS * OVERVIEW_IMG_SIZE, OVERVIEW_IMG_SIZE,
                               fill=False, edgecolor=config.plotting.get('positive_color', '#8B0000'),
                               linewidth=MOSAIC_BORDER_WIDTH))
        ax.add_patch(Rectangle((mosaic_x0, ny0), OVERVIEW_COLS * OVERVIEW_IMG_SIZE, OVERVIEW_IMG_SIZE,
                               fill=False, edgecolor=config.plotting.get('negative_color', '#000080'),
                               linewidth=MOSAIC_BORDER_WIDTH))
    for i in range(n_comps):
        y0 = i * (rows_per_comp * OVERVIEW_IMG_SIZE + spacing_per_comp)
        ax.text(mosaic_x0 + 8, y0 + 14, f'C{i+1}', ha='left', va='top', fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', edgecolor='none', alpha=0.6))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_overview_cca_with_duplet_bars(comps, all_stims, cats, img_dir, save_path, family_name,
                                       imp_mean=None, feature_names=None, feature_groups=None):
    """Extended overview: existing pos/neg mosaics + per-component top feature bars + side table.
    Feature names appear in a two-column text table (Visual | Behav) below the bars.
    """
    if (not INCLUDE_DUPLET_BARS) or imp_mean is None or feature_names is None or feature_groups is None:
        return plot_overview_cca(comps, all_stims, cats, img_dir, save_path, family_name)

    n_comps = min(comps.shape[1], MAX_OVERVIEW_COMPONENTS, imp_mean.shape[0])
    rows_per_comp = 2
    spacing_per_comp = OVERVIEW_ROW_SPACING

    all_rows = []
    for comp_idx in range(n_comps):
        vec = comps[:, comp_idx]
        order = np.argsort(vec)
        pos_idx = subsample_top_pct(vec, OVERVIEW_COLS)
        pos_row = np.hstack([
            np.array(load_img_simple(os.path.join(img_dir, cats[i], f"{all_stims[i]}.jpg"), OVERVIEW_IMG_SIZE))
            for i in pos_idx
        ])
        all_rows.append(pos_row)
        neg_idx = subsample_top_pct(vec, OVERVIEW_COLS, pos=False)
        neg_row = np.hstack([
            np.array(load_img_simple(os.path.join(img_dir, cats[i], f"{all_stims[i]}.jpg"), OVERVIEW_IMG_SIZE))
            for i in neg_idx
        ])
        all_rows.append(neg_row)
        if comp_idx < n_comps - 1:
            spacing = np.full((spacing_per_comp, OVERVIEW_COLS * OVERVIEW_IMG_SIZE, 3), 255, np.uint8)
            all_rows.append(spacing)

    full_grid = np.vstack(all_rows)
    H_img, W_img = full_grid.shape[:2]

    import matplotlib.pyplot as _plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FigureCanvasAgg

    def _short(name):
        lab = name
        for ch in ['-', '/', ' ', '_']:
            if ch in lab:
                lab = lab.split(ch)[0]
                break
        return lab

    vis_range = config.plotting.get('visual_range', [0.9, 0.92])
    beh_range = config.plotting.get('behavioral_range', [0.6, 0.62])

    vis_shades = _bone_shades(3, *vis_range)
    beh_shades = _bone_shades(3, *beh_range)

    bar_panel_height = 2 * OVERVIEW_IMG_SIZE
    panels = []
    for comp_idx in range(n_comps):
        vals = np.array(imp_mean[comp_idx], dtype=float)
        idx_vis = [i for i,g in enumerate(feature_groups) if g == 'visual']
        idx_beh = [i for i,g in enumerate(feature_groups) if g == 'behavioral']
        top_vis = sorted(idx_vis, key=lambda i: vals[i], reverse=True)[:3]
        top_beh = sorted(idx_beh, key=lambda i: vals[i], reverse=True)[:3]
        bar_vals = [vals[i] for i in top_vis] + [vals[i] for i in top_beh]
        bar_cols = vis_shades[:len(top_vis)] + beh_shades[:len(top_beh)]
        vis_labels = [feature_names[i] for i in top_vis]
        beh_labels = [feature_names[i] for i in top_beh]

        panel = np.full((bar_panel_height, BAR_PANEL_WIDTH_PX, 3), 255, np.uint8)
        if len(bar_vals) == 0:
            panels.append(panel)
            continue

        # Figure divided: left 55% bars, right 40% text table (remaining for padding)
        fig_w_in = BAR_PANEL_WIDTH_PX / 100.0
        fig_h_in = bar_panel_height / 100.0
        _fig = _plt.figure(figsize=(fig_w_in, fig_h_in), dpi=100)
        bar_ax = _fig.add_axes([0.08, 0.08, 0.50, 0.84])
        txt_ax = _fig.add_axes([0.63, 0.08, 0.34, 0.84])
        x = np.arange(len(bar_vals))
        bar_ax.bar(x, bar_vals, color=bar_cols, edgecolor='black', linewidth=0.8)
        bar_ax.set_xticks([])
        bar_ax.tick_params(axis='y', labelsize=6, length=3)
        bar_ax.set_ylabel(r'$\Delta R^2$', fontsize=7)
        for spine in ['top','right']:
            bar_ax.spines[spine].set_visible(False)
        ymax = max(bar_vals) * 1.15 if max(bar_vals) > 0 else 1.0
        bar_ax.set_ylim(0, ymax)
        bar_ax.margins(x=0)

        txt_ax.axis('off')
        # Column headers
        txt_ax.text(0.00, 0.97, 'Visual', fontsize=8, fontweight='bold', va='top')
        txt_ax.text(0.55, 0.97, 'Behav', fontsize=8, fontweight='bold', va='top')
        # List up to 3 rows
        for r in range(max(len(vis_labels), len(beh_labels))):
            y = 0.90 - r * 0.18
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
            from PIL import Image as _Image
            panel = np.array(_Image.fromarray(panel).resize((BAR_PANEL_WIDTH_PX, bar_panel_height), _Image.BILINEAR))
        panels.append(panel)

    # Compose final canvas (panel on left now)
    total_width = BAR_PANEL_WIDTH_PX + BAR_GAP_PX + W_img
    canvas = np.full((H_img, total_width, 3), 255, np.uint8)
    # Place bar panels first
    for comp_idx, panel in enumerate(panels):
        block_start = comp_idx * (rows_per_comp * OVERVIEW_IMG_SIZE + spacing_per_comp)
        y0 = block_start
        canvas[y0:y0+panel.shape[0], :BAR_PANEL_WIDTH_PX] = panel
    # Place image mosaic to the right
    canvas[:H_img, BAR_PANEL_WIDTH_PX + BAR_GAP_PX:BAR_PANEL_WIDTH_PX + BAR_GAP_PX + W_img] = full_grid

    fig_w = total_width / 100.0
    fig_h = H_img / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(canvas)
    ax.axis('off')

    mosaic_x0 = BAR_PANEL_WIDTH_PX + BAR_GAP_PX  # left edge of image mosaics
    for i in range(n_comps):
        y_pos = i * (rows_per_comp * OVERVIEW_IMG_SIZE + spacing_per_comp) + OVERVIEW_IMG_SIZE
        ax.text(mosaic_x0 - 6, y_pos, f'C{i+1}', ha='right', va='center', fontsize=8)
        block_start = i * (rows_per_comp * OVERVIEW_IMG_SIZE + spacing_per_comp)
        pos_y0 = block_start
        neg_y0 = block_start + OVERVIEW_IMG_SIZE
        ax.add_patch(Rectangle((mosaic_x0, pos_y0), OVERVIEW_COLS * OVERVIEW_IMG_SIZE, OVERVIEW_IMG_SIZE,
                               fill=False, edgecolor=config.plotting.get('positive_color', '#8B0000'),
                               linewidth=MOSAIC_BORDER_WIDTH))
        ax.add_patch(Rectangle((mosaic_x0, neg_y0), OVERVIEW_COLS * OVERVIEW_IMG_SIZE, OVERVIEW_IMG_SIZE,
                               fill=False, edgecolor=config.plotting.get('negative_color', '#000080'),
                               linewidth=MOSAIC_BORDER_WIDTH))

    ax.set_title(f"{family_name.upper()} CCA Components Overview (first {n_comps} components)", fontsize=14, pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

"""
Removed subject-wise ridge helpers and plots.
"""


def _bone_shades(n: int, start: float, end: float):
    tvals = [0.5 * (start + end)] if n <= 1 else np.linspace(start, end, n)
    cols = []
    for t in tvals:
        r, g, b, _ = cm.bone(float(t))
        cols.append((r, g, b))
    return cols

def plot_duplets_from_joint(imp_mean: np.ndarray, names: list, groups: list,
                             comp_indices: list, std: np.ndarray = None,
                             feature_label_fontsize: float = 11,
                             extra_label_gap_frac: float = 0.01):
    """
    Grouped bars from joint matrix with visual/behavioral/semantic tags.
    - Bars are CENTER-aligned; error bars and labels use the same center x.
    - No whitespace at the x-axis (ax.margins(y=0)).
    - Tight layout to avoid bleed.
    - Feature labels above bars use a slightly larger fontsize (only these labels).
    - A tiny extra gap between error bar and label (extra_label_gap_frac of y-range).
    """
    if imp_mean is None or names is None or groups is None or not comp_indices:
        return None

    # Load color ranges from config
    vis_range = config.plotting.get('visual_range', [0.9, 0.92])
    beh_range = config.plotting.get('behavioral_range', [0.6, 0.62])
    sem_range = config.plotting.get('semantic_range', [0.2, 0.22])

    vis_shades = _bone_shades(2, *vis_range)
    beh_shades = _bone_shades(2, *beh_range)
    sem_shades = _bone_shades(2, *sem_range)

    bar_width = 0.5
    gap_between_groups = 0.6
    gap_within_groups = 0.0  # No gap between v/b/s pairs
    bars_per_group = 6  # 2 per group × 3 groups
    positions, heights, colors, labels, group_centers, yerrs = [], [], [], [], [], []

    idx_vis_all = [i for i,g in enumerate(groups) if g == 'visual']
    idx_beh_all = [i for i,g in enumerate(groups) if g == 'behavioral']
    idx_sem_all = [i for i,g in enumerate(groups) if g == 'semantic']

    # Load behavioral dimension names for proper labeling
    try:
        from functions.viz_common import load_behav_dim_names
        beh_dim_names = load_behav_dim_names()
    except:
        beh_dim_names = []

    for gi, ci in enumerate(comp_indices):
        vals = imp_mean[ci]
        idx_v = sorted(idx_vis_all, key=lambda i: vals[i], reverse=True)[:2]
        idx_b = sorted(idx_beh_all, key=lambda i: vals[i], reverse=True)[:2]
        idx_s = sorted(idx_sem_all, key=lambda i: vals[i], reverse=True)[:2]
        names_v = [names[j] for j in idx_v]
        names_b = [names[j] for j in idx_b]
        names_s = [names[j] for j in idx_s]
        vals_v = vals[idx_v] if len(idx_v)>0 else []
        vals_b = vals[idx_b] if len(idx_b)>0 else []
        vals_s = vals[idx_s] if len(idx_s)>0 else []
        stds_v = std[ci][idx_v] if (std is not None and len(idx_v)>0) else []
        stds_b = std[ci][idx_b] if (std is not None and len(idx_b)>0) else []
        stds_s = std[ci][idx_s] if (std is not None and len(idx_s)>0) else []

        group_start = gi * (bars_per_group * bar_width + gap_between_groups)

        # Visual bars
        for k in range(2):
            positions.append(group_start + k * bar_width)
            heights.append(float(vals_v[k]) if k < len(vals_v) else 0.0)
            colors.append(vis_shades[k])
            labels.append(names_v[k] if k < len(names_v) else '')
            yerrs.append(float(stds_v[k]) if (k < len(stds_v)) else 0.0)

        # Behavioral bars (with gap after visual)
        beh_start = group_start + 2 * bar_width + gap_within_groups
        for k in range(2):
            positions.append(beh_start + k * bar_width)
            heights.append(float(vals_b[k]) if k < len(vals_b) else 0.0)
            colors.append(beh_shades[k])
            # Use actual behavioral dimension names if available
            if k < len(names_b) and beh_dim_names and names_b[k].startswith('SPOSE'):
                try:
                    idx = int(names_b[k].split()[-1]) - 1
                    if 0 <= idx < len(beh_dim_names):
                        beh_name = beh_dim_names[idx]
                    else:
                        beh_name = names_b[k]
                except:
                    beh_name = names_b[k]
            else:
                beh_name = names_b[k]
            labels.append(beh_name)
            yerrs.append(float(stds_b[k]) if (k < len(stds_b)) else 0.0)

        # Semantic bars (with gap after behavioral)
        sem_start = beh_start + 2 * bar_width + gap_within_groups
        for k in range(2):
            positions.append(sem_start + k * bar_width)
            heights.append(float(vals_s[k]) if k < len(vals_s) else 0.0)
            colors.append(sem_shades[k])
            labels.append(names_s[k] if k < len(names_s) else '')
            yerrs.append(float(stds_s[k]) if (k < len(stds_s)) else 0.0)

        group_centers.append(group_start + (bars_per_group - 1) * bar_width / 2)

    fig = wide_figure()
    ax = fig.add_subplot(111)

    # Bars centered at 'positions'
    bars = ax.bar(positions, heights, width=bar_width, color=colors,
                  edgecolor='black', linewidth=1.2, align='center', zorder=2)

    # Error bars centered on the same x
    for x, h, ye in zip(positions, heights, yerrs):
        if ye and ye > 0:
            ax.errorbar(x, h, yerr=ye, fmt='none', ecolor='black',
                        elinewidth=1.0, capsize=0, zorder=3)

    # Determine y headroom and padding
    ymax_data = max([h + ye for h, ye in zip(heights, yerrs)] or [1.0])
    ax.set_ylim(0, ymax_data * 1.60)  # a touch more headroom to avoid clipping labels
    ax.margins(y=0)                    # bars sit flush on baseline

    # A small base pad plus an extra fixed fraction of the y-range for label spacing
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    base_pad = (max(heights) if heights else 1.0) * 0.012
    extra_gap = yrange * float(extra_label_gap_frac)

    # Labels centered above the bars (with slightly larger fontsize)
    for x, h, name, ye in zip(positions, heights, labels, yerrs):
        if name:
            lab = name
            for ch in ['-', '/', ' ']:
                if ch in lab:
                    lab = lab.split(ch)[0]
                    break
            ax.text(x, h + ye + base_pad + extra_gap, lab.replace('_',' '), rotation=90,
                    ha='center', va='bottom', fontsize=feature_label_fontsize, clip_on=False)

    ax.set_ylabel(r'$\Delta R^2$')
    style_axes(ax); format_axes(ax, precision=3)

    # Component numbering under groups
    ax.set_xticks(group_centers)
    comp_labels = [str(ci+1) for ci in comp_indices]
    ax.set_xticklabels(comp_labels)
    ax.set_xlabel('Component')

    # Ensure x-axis labels are normal style
    for label in ax.get_xticklabels():
        label.set_fontstyle('normal')
        label.set_fontweight('normal')

    plt.tight_layout(pad=0.6)
    return fig

def make_legend():
    vis_range = config.plotting.get('visual_range', [0.9, 0.92])
    beh_range = config.plotting.get('behavioral_range', [0.6, 0.62])
    sem_range = config.plotting.get('semantic_range', [0.2, 0.22])

    vis_shades = _bone_shades(2, *vis_range)
    beh_shades = _bone_shades(2, *beh_range)
    sem_shades = _bone_shades(2, *sem_range)

    fig = narrow_figure()
    ax = fig.add_subplot(111)
    handles = [
        plt.Rectangle((0,0),1,1, color=vis_shades[0], ec='black', lw=1.0, label='Visual'),
        plt.Rectangle((0,0),1,1, color=beh_shades[0], ec='black', lw=1.0, label='Behavioral'),
        plt.Rectangle((0,0),1,1, color=sem_shades[0], ec='black', lw=1.0, label='Semantic'),
    ]
    ax.axis('off')
    _ = ax.legend(handles=handles, frameon=False, loc='center')
    plt.tight_layout(pad=0.2)
    return fig

def _draw_mini_profile(ax, imp_row, std_row, names, groups,
                        feature_label_fontsize=8, ylim_max=None):
    vis_range = config.plotting.get('visual_range', [0.9, 0.92])
    beh_range = config.plotting.get('behavioral_range', [0.6, 0.62])

    vis_shades = _bone_shades(3, *vis_range)
    beh_shades = _bone_shades(3, *beh_range)

    idx_vis_all = [i for i, g in enumerate(groups) if g == 'visual']
    idx_beh_all = [i for i, g in enumerate(groups) if g == 'behavioral']

    vals = imp_row
    idx_v = sorted(idx_vis_all, key=lambda i: vals[i], reverse=True)[:3]
    idx_b = sorted(idx_beh_all, key=lambda i: vals[i], reverse=True)[:3]

    vals_v = [float(vals[i]) for i in idx_v]
    vals_b = [float(vals[i]) for i in idx_b]
    stds_v = [float(std_row[i]) if std_row is not None else 0.0 for i in idx_v]
    stds_b = [float(std_row[i]) if std_row is not None else 0.0 for i in idx_b]
    labs_v = [names[i] for i in idx_v]
    labs_b = [names[i] for i in idx_b]

    heights = vals_v + vals_b
    yerrs = stds_v + stds_b
    colors = vis_shades[:len(vals_v)] + beh_shades[:len(vals_b)]

    xpos = np.arange(len(heights))
    ax.bar(xpos, heights, color=colors, edgecolor='black', linewidth=1.0, zorder=2)
    for x, h, ye in zip(xpos, heights, yerrs):
        if ye and ye > 0:
            ax.errorbar(x, h, yerr=ye, fmt='none', ecolor='black', elinewidth=1.0, capsize=0, zorder=3)

    if ylim_max is None:
        ymax = max([h + ye for h, ye in zip(heights, yerrs)] or [1.0]) * 1.35
    else:
        ymax = float(ylim_max)
    ax.set_ylim(0, ymax)
    ax.margins(y=0)
    ax.set_xticks([])
    ax.set_ylabel(r'$\Delta R^2$', fontsize=9)

    # Labels above bars
    yrange = ymax
    base_pad = (max(heights) if heights else 1.0) * 0.01
    extra_gap = yrange * 0.01
    short = []
    for name in (labs_v + labs_b):
        lab = name
        for ch in ['-', '/', ' ']:
            if ch in lab:
                lab = lab.split(ch)[0]
                break
        short.append(lab.replace('_', ' '))
    for x, h, ye, lab in zip(xpos, heights, yerrs, short):
        ax.text(x, h + ye + base_pad + extra_gap, lab, rotation=90, ha='center', va='bottom', fontsize=feature_label_fontsize)
    style_axes(ax); format_axes(ax)

def _build_pos_neg_mosaics_for_comp(comps, ci, stims, cats, img_dir, rows=2, cols=6, img_px=180):
    vec = comps[:, ci]
    n_cells = rows * cols
    order = np.argsort(vec)
    ipos = subsample_top_pct(vec, n_cells)
    ineg = subsample_top_pct(vec, n_cells, pos=False)
    pos_paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ipos]
    neg_paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ineg]
    return _build_mosaic(pos_paths, rows, cols, img_px), _build_mosaic(neg_paths, rows, cols, img_px)

def plot_top_components_profiles_and_grids(analysis_family_name, comps, stims, cats, img_dir,
                                           imp_mean, imp_std, names, groups,
                                           top_k=5, rows=2, cols=6, img_px=180):
    # Rank components by maximum ΔR² over all features
    scores = np.max(imp_mean, axis=1)
    order = np.argsort(scores)[::-1]
    chosen = order[:min(top_k, imp_mean.shape[0])]

    # Determine a shared y-limit for profiles
    ymax = 0.0
    for ci in chosen:
        vals = imp_mean[ci]
        stds = imp_std[ci] if imp_std is not None else np.zeros_like(vals)
        idx_vis_all = [i for i, g in enumerate(groups) if g == 'visual']
        idx_beh_all = [i for i, g in enumerate(groups) if g == 'behavioral']
        idx_v = sorted(idx_vis_all, key=lambda i: vals[i], reverse=True)[:3]
        idx_b = sorted(idx_beh_all, key=lambda i: vals[i], reverse=True)[:3]
        local_max = 0.0
        for j in (idx_v + idx_b):
            local_max = max(local_max, float(vals[j] + (stds[j] if stds is not None else 0.0)))
        ymax = max(ymax, local_max)
    ymax = ymax * 1.35 if ymax > 0 else 1.0

    # Figure layout: rows = number of chosen components, 2 columns
    # Increase height because we show two mosaics (pos & neg) per row
    fig_h = max(5, 2.6 * len(chosen))
    fig_w = 10
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs = GridSpec(len(chosen), 2, width_ratios=[1.0, 3.0], hspace=0.8, wspace=0.25, figure=fig)

    for r, ci in enumerate(chosen):
        ax_prof = fig.add_subplot(gs[r, 0])
        _draw_mini_profile(ax_prof, imp_mean[ci], imp_std[ci] if imp_std is not None else None,
                           names, groups, feature_label_fontsize=8, ylim_max=ymax)
        ax_prof.set_title(f'C{ci+1} profile', fontsize=10)

        # Nested grid for positive (top) and negative (bottom) mosaics
        cell_gs = gs[r, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.05)

        pos_mosaic, neg_mosaic = _build_pos_neg_mosaics_for_comp(
            comps, ci, stims, cats, img_dir, rows=rows, cols=cols, img_px=img_px
        )

        ax_pos = fig.add_subplot(cell_gs[0])
        ax_pos.imshow(pos_mosaic, interpolation='nearest', aspect='auto')
        ax_pos.axis('off')
        _border(ax_pos, color=config.plotting.get('positive_color', '#8B0000'), lw=MOSAIC_BORDER_WIDTH)

        ax_neg = fig.add_subplot(cell_gs[1])
        ax_neg.imshow(neg_mosaic, interpolation='nearest', aspect='auto')
        ax_neg.axis('off')
        _border(ax_neg, color=config.plotting.get('negative_color', '#000080'), lw=MOSAIC_BORDER_WIDTH)

    fig.suptitle(f"{analysis_family_name.upper()} – Top {len(chosen)} Components by Max ΔR²", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, [(int(ci)+1, float(scores[ci])) for ci in chosen]

def plot_comp_imgs(ci, comps, stims, cats, img_dir=None, title=None, rows=2, cols=5,
                   img_px=180, dpi=300):
    """Plot component images with positive and negative grids."""
    if img_dir is None:
        img_dir = img_base_dir

    vec = comps[:, ci]
    n_cells = rows * cols
    order = np.argsort(vec)
    ineg = subsample_top_pct(vec, n_cells, pos=False)
    ipos = subsample_top_pct(vec, n_cells)

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    fw = cols * img_px / dpi
    fh = (2 * rows + 1) * img_px / dpi
    fig = plt.figure(figsize=(fw, fh), dpi=dpi, constrained_layout=False)
    gs_master = GridSpec(3, 1, height_ratios=[rows, 1, rows], hspace=0, figure=fig)

    pos_paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ipos]
    pos_mosaic = _build_mosaic(pos_paths, rows, cols, img_px)
    ax_top = fig.add_subplot(gs_master[0])
    ax_top.imshow(pos_mosaic, interpolation='nearest', aspect='auto')
    ax_top.axis('off')
    ax_top.set_xlim(-0.5, pos_mosaic.shape[1] - 0.5)
    ax_top.set_ylim(pos_mosaic.shape[0] - 0.5, -0.5)
    _border(ax_top, color=config.plotting.get('positive_color', '#8B0000'), lw=MOSAIC_BORDER_WIDTH)

    ax_center = fig.add_subplot(gs_master[1])
    ax_center.set_xlim(-0.5, pos_mosaic.shape[1] - 0.5)
    ax_center.margins(x=0, y=0)
    _plot_distribution(ax_center, vec, order, ineg, ipos, inv)
    ax_center.set_yticks([]); ax_center.set_xticks([])

    neg_paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ineg]
    neg_mosaic = _build_mosaic(neg_paths, rows, cols, img_px)
    ax_bot = fig.add_subplot(gs_master[2])
    ax_bot.imshow(neg_mosaic, interpolation='nearest', aspect='auto')
    ax_bot.axis('off')
    ax_bot.set_xlim(-0.5, neg_mosaic.shape[1] - 0.5)
    ax_bot.set_ylim(neg_mosaic.shape[0] - 0.5, -0.5)
    _border(ax_bot, color=config.plotting.get('negative_color', '#000080'), lw=MOSAIC_BORDER_WIDTH)

    if title:
        fig.suptitle(f"{title} – Component {ci+1}", fontsize=14, y=0.99)
    return fig

def _square_and_resize(path, target):
    """Load → center-crop to square → resize."""
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, upper = (w - side) // 2, (h - side) // 2
        im = im.crop((left, upper, left + side, upper + side))
        im = im.resize((target, target), Image.BILINEAR)
        im_arr = np.array(im)
        thick = max(2, target // 60)
        border_color = (64, 64, 64)
        im_arr[:thick, :] = border_color; im_arr[-thick:, :] = border_color
        im_arr[:, :thick] = border_color; im_arr[:, -thick:] = border_color
        return Image.fromarray(im_arr)
    except Exception:
        blank = np.full((target, target, 3), 128, np.uint8)
        thick = max(2, target // 60); border_color = (64, 64, 64)
        blank[:thick, :] = border_color; blank[-thick:, :] = border_color
        blank[:, :thick] = border_color; blank[:, -thick:] = border_color
        return Image.fromarray(blank)

def _build_mosaic(paths, rows, cols, px):
    """Return single numpy array with all images tiled."""
    blank = np.full((px, px, 3), 128, np.uint8)
    tiles = [_square_and_resize(p, px) if p is not None else Image.fromarray(blank) for p in paths]
    tiles += [Image.fromarray(blank)] * (rows * cols - len(tiles))
    tiles = [np.asarray(t) for t in tiles]
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    return grid

def _border(ax, color='black', lw=3):
    """Add rectangle around axes."""
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor='none', edgecolor=color,
                           linewidth=lw, clip_on=False))

def _plot_distribution(ax, vec, order, ineg, ipos, inv):
    """Draw density curve with brackets."""
    vals = vec[order]; total = len(vals); z0 = np.searchsorted(vals, 0)

    def _smooth(y): return gaussian_filter1d(y, sigma=len(y) / 50) if len(y) > 0 else y

    if z0 > 0:
        x = np.linspace(0, z0 - 1, 500)
        y = _smooth(np.interp(x, np.arange(z0), np.abs(vals[:z0])))
        if y.max() > 0: y *= (np.abs(vals[:z0]).max() or 1) / y.max()
        for i in range(len(x) - 1):
            frac = (z0 - x[i]) / z0
            ax.fill_between(x[i:i + 2], 0, -y[i:i + 2], color=cm.Blues(0.3 + 0.7 * frac), lw=0)
    if total - z0 > 0:
        x = np.linspace(z0, total - 1, 500)
        y = _smooth(np.interp(x, np.arange(z0, total), vals[z0:]))
        if y.max() > 0: y *= (vals[z0:].max() or 1) / y.max()
        for i in range(len(x) - 1):
            frac = (x[i] - z0) / (total - z0)
            ax.fill_between(x[i:i + 2], 0, y[i:i + 2], color=cm.Reds(0.3 + 0.7 * frac), lw=0)

    ax.axhline(0, color='k', lw=0)
    ax.set_xlim(0, total)
    y0, y1 = ax.get_ylim(); dy = y1 - y0
    ax.set_ylim(y0 - 0.05 * dy, y1 + 0.05 * dy)
    for sp in ax.spines.values(): sp.set_visible(False)

    xn_all = inv[ineg]; xp_all = inv[ipos]
    xn0, xn1 = xn_all.min(), xn_all.max()
    xp0, xp1 = xp_all.min(), xp_all.max()
    y0, y1 = ax.get_ylim()

    neg_color = config.plotting.get('negative_color', '#000080')
    pos_color = config.plotting.get('positive_color', '#8B0000')
    ax.plot([xn0, xn1], [0, 0], color=neg_color, lw=4)
    ax.plot([xn0, xn0], [y0, 0], color=neg_color, lw=4)
    ax.plot([xn1, xn1], [y0, 0], color=neg_color, lw=4)
    ax.plot([xp0, xp1], [0, 0], color=pos_color, lw=4)
    ax.plot([xp0, xp0], [0, y1], color=pos_color, lw=4)
    ax.plot([xp1, xp1], [0, y1], color=pos_color, lw=4)

def plot_radial(c1, c2, comps, stims, cats, img_dir, n=None, zm=0.65, mindist=0.002,
                offset=0.2):
    """Plot radial distribution."""
    co = comps[:, [c1, c2]]
    ctr = co.mean(0)

    fig = large_figure()
    ax = fig.add_subplot(111)

    dists = np.linalg.norm(co - ctr, axis=1)
    dist_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
    colors = plt.cm.Greys(0.35 + 0.2 * dist_norm)

    ax.scatter(co[:,0], co[:,1], s=12, c=colors, alpha=1.0, edgecolors='none', marker='o', linewidth=0)

    if n and n < len(co):
        d = np.linalg.norm(co - ctr, axis=1)
        th = np.arctan2(co[:,1]-ctr[1], co[:,0]-ctr[0])
        bins = np.linspace(-np.pi, np.pi, n+1)
        bidx = np.digitize(th, bins) - 1
        sel, used = [], set()
        for idx in np.argsort(d)[::-1]:
            b = bidx[idx]
            if b in used: continue
            if not sel or np.min(np.linalg.norm(co[idx]-co[sel], axis=1)) >= mindist:
                sel.append(idx); used.add(b)
            if len(sel) >= n: break
        if len(sel) < n:
            for idx in np.argsort(d)[::-1]:
                if idx in sel: continue
                if np.min(np.linalg.norm(co[idx]-co[sel], axis=1)) >= mindist:
                    sel.append(idx)
                if len(sel) >= n: break
        idxs = sel
    else:
        idxs = list(range(len(co)))

    x0, x1 = co[:,0].min(), co[:,0].max()
    y0, y1 = co[:,1].min(), co[:,1].max()
    dr = max(x1-x0, y1-y0)
    od = offset * dr

    for i in idxs:
        v = co[i] - ctr
        dist = np.linalg.norm(v)
        u = v/dist if dist>0 else np.array([1.,0.])
        pt = co[i] + u * od

        ax.plot([co[i,0], pt[0]], [co[i,1], pt[1]], color='0.2', alpha=0.8, lw=2.5)

        path = os.path.join(img_dir, cats[i], f"{stims[i]}.jpg")
        try:
            im = Image.open(path).convert("RGB").resize((200,200), Image.BILINEAR)
            im_arr = np.array(im)
            thick = max(8, 200 // 25)
            im_arr = _add_img_border(im_arr, thick=thick)
            oi = OffsetImage(im_arr, zoom=zm/3)
            ab = AnnotationBbox(oi, pt, frameon=False, pad=0.08)
            ax.add_artist(ab)
        except Exception:
            pass

    sel_pts = co[idxs]
    ax.scatter(sel_pts[:,0], sel_pts[:,1], s=40, color='0.2', alpha=0.9,
               edgecolors='none', marker='o')

    ax.set_xlim(x0 - od, x1 + od); ax.set_ylim(y0 - od, y1 + od)
    ax.axis('off')
    plt.tight_layout()
    return fig

def _add_img_border(im_arr, thick=None):
    """Add light grey inward border to image array"""
    if thick is None:
        thick = max(2, min(im_arr.shape[:2]) // 60)
    border_color = (128, 128, 128)
    im_arr[:thick, :] = border_color; im_arr[-thick:, :] = border_color
    im_arr[:, :thick] = border_color; im_arr[:, -thick:] = border_color
    return im_arr

def save_fig(fig, fig_type, analysis_family, comp_idx=None, extra_info="", show_inline=False):
    if not fig:
        return
    image_high_dpi = fig_type in ('images', 'selected')
    save_fig_std('cca', analysis_family, fig_type, fig,
                 comp_idx=comp_idx, extra=extra_info, image_high_dpi=image_high_dpi)

# ──────────────────────────────────────────────────────────────────────
# Main execution
if __name__ == "__main__":

    os.makedirs(config.fig_dir, exist_ok=True)

    # Clear CCA analysis folder before writing new figures
    cca_fig_dir = os.path.join(config.fig_dir, 'cca_analysis')
    if os.path.exists(cca_fig_dir):
        import shutil
        shutil.rmtree(cca_fig_dir)
    os.makedirs(cca_fig_dir, exist_ok=True)

    family_map = {
        'cross-species': (comps_all, X_all, 'cross-species'),
        'human':         (comps_hum, X_human, 'human'),
        'monkey':        (comps_mon, X_monkey, 'monkey'),
    }

    for fam in TARGET_FAMILIES:
        if fam not in family_map:
            raise ValueError(f"Invalid TARGET_FAMILIES entry: {fam}")

    for fam in TARGET_FAMILIES:
        target_comps, target_data, analysis_family_name = family_map[fam]
        print(f"\n--- Generating plots for '{analysis_family_name}' components ---")

        print(f"--- Generating components overview grid (first {MAX_OVERVIEW_COMPONENTS} components) ---")
        overview_path = os.path.join(config.fig_dir, 'cca_analysis', analysis_family_name, 'overview', 'components_overview.png')

        # Load unified R^2 matrices for bars and top-features
        imp_mean, imp_std, feature_names, feature_groups = load_enc_matrix('cca', analysis_family_name)

        try:
            if INCLUDE_DUPLET_BARS and imp_mean is not None and feature_names is not None and feature_groups is not None:
                plot_overview_cca_with_duplet_bars(target_comps, stims_all, cats, img_base_dir, overview_path, analysis_family_name,
                                                   imp_mean=imp_mean, feature_names=feature_names, feature_groups=feature_groups)
            else:
                plot_overview_cca(target_comps, stims_all, cats, img_base_dir, overview_path, analysis_family_name)
        except Exception as e:
            print(f"  Failed enhanced overview, falling back to original: {e}")
            plot_overview_cca(target_comps, stims_all, cats, img_base_dir, overview_path, analysis_family_name)
        print(f"Saved overview to {overview_path}")

        # Generate duplets plot (use DUPLET_COMP_RANGE or config-selected indices)
        if imp_mean is not None and feature_names is not None and feature_groups is not None:
            sel = CCA_SELECTED.get(analysis_family_name, [1,2,3,4])
            fig_duplets = plot_duplets_from_enc_scores('cca', analysis_family_name, sel,
                feature_label_fontsize=9,
                extra_label_gap_frac=0.012
            )
            save_fig(fig_duplets, 'duplets', analysis_family_name)

            # Optional: standalone legend
            fig_legend = make_legend()
            save_fig(fig_legend, 'legend', analysis_family_name, extra_info='duplet_legend')

            # Top features: rank components and render profile+grid rows
            print(f"--- Generating top features grid (profiles + images) for {analysis_family_name} ---")
            fig_top, summary = plot_top_components_profiles_and_grids(
                analysis_family_name, target_comps, stims, cats, img_base_dir,
                imp_mean, imp_std, feature_names, feature_groups, top_k=5, rows=2, cols=6, img_px=180
            )
            save_fig(fig_top, 'top_features', analysis_family_name, extra_info='top5_profiles_grids')

            # Save CSV summary
            top_dir = os.path.join(config.fig_dir, 'cca_analysis', analysis_family_name, 'top_features')
            os.makedirs(top_dir, exist_ok=True)
            df = pd.DataFrame(summary, columns=['Component', 'MaxDeltaR2'])
            csv_path = os.path.join(top_dir, 'top5_summary.csv')
            df.to_csv(csv_path, index=False)
            print(f"    Saved summary to {csv_path}")

        # Selected components (pos/neg image grids)
        sel = CCA_SELECTED.get(analysis_family_name, [1,2,3,4])
        for comp_id in sel:
            ci = int(comp_id) - 1
            if ci < 0 or ci >= target_comps.shape[1]:
                continue
            print(f"  Selected {analysis_family_name}: Component {comp_id} images...")
            fig_images = plot_comp_imgs(ci, target_comps, stims, cats, img_dir=img_base_dir,
                                        title=f"{analysis_family_name.title()} Component {comp_id}",
                                        img_px=360, dpi=300)
            save_fig(fig_images, 'selected', analysis_family_name, comp_idx=ci,
                     show_inline=SHOW_INLINE_IMAGE_GRIDS)

        # Removed subject-wise ridge plotting

        print(f"--- Generating radial plot for components 1 vs 2 ({analysis_family_name}) ---")
        fig_radial = plot_radial(0, 1, target_comps, stims, cats, img_base_dir, n=35)
        save_fig(fig_radial, 'radial', analysis_family_name, extra_info='comp_radial')

    print("\nAnalysis complete. All figures saved for families:", ", ".join(TARGET_FAMILIES))
