#!/usr/bin/env python3
"""
sNMF component gallery builder.

Expects ridge guard CSVs produced by analysis/nmf/ridge_guard.py and renders
large overview grids for shared, human-specific, and monkey-specific factors.
"""

from pathlib import Path
import sys
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colormaps
from matplotlib import font_manager
from PIL import Image

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

from config.paths import config
from analysis.nmf.utils import (
    load_srf_results,
    pick_rank,
    build_img_paths,
    ensure_dir,
    load_guard_table,
    derive_categories,
)
from analysis.supp.gallery_style import render_nmf_supp_gallery

HERE = Path(__file__).resolve().parent
RESULT_DIR = ensure_dir(HERE / 'results')
ensure_dir(RESULT_DIR / 'guard')
FIG_DIR = ensure_dir(config.fig_dir / 'nmf_analysis')

# Gallery layout
TOP_IMAGES = 20
IMG_PX = 150
LABEL_W = 270
BAR_W = 138
WC_W = 420
GAP_COL = 18
ROW_GAP = 16
DPI = 300
PANEL_DPI = 300
INFO_BG = '#fbfbfb'
INFO_EDGE = '#d4d4d4'
ARIAL_PATH = None

# Thresholds
R2_SHARED = 0.20
R2_SPEC = 0.20
R2_RATIO = 0.50

HUM_COLOR = config.plotting.get('human_color', '#7c5799')
MON_COLOR = config.plotting.get('monkey_color', '#bda855')
POS_COLOR = config.plotting.get('positive_color', '#A8393F')

GALLERY_SPECS = {
    'shared': {
        'family': 'all',
        'title': 'Shared components (cross-species sNMF)',
        'filename': 'shared_gallery.pdf',
        'summary': 'shared_gallery.csv',
        'top_n': 25,
        'dpi': 400,
        'filter': lambda df: (df['human_r2'] >= R2_SHARED) & (df['monkey_r2'] >= R2_SHARED),
        'score': lambda row: np.minimum(row['human_r2'], row['monkey_r2']),
        'label_prefix': 'ALL',
    },
    'human': {
        'family': 'human',
        'title': 'Human-specific components (human sNMF)',
        'filename': 'human_gallery.pdf',
        'summary': 'human_gallery.csv',
        'filter': lambda df: (df['human_r2'] >= R2_SPEC) & (df['monkey_r2'] < (R2_RATIO * df['human_r2'])),
        'score': lambda row: row['human_r2'],
        'label_prefix': 'HUM',
    },
    'monkey': {
        'family': 'monkey',
        'title': 'Monkey-specific components (monkey sNMF)',
        'filename': 'monkey_gallery.pdf',
        'summary': 'monkey_gallery.csv',
        'filter': lambda df: (df['monkey_r2'] >= R2_SPEC) & (df['human_r2'] < (R2_RATIO * df['monkey_r2'])),
        'score': lambda row: row['monkey_r2'],
        'label_prefix': 'MON',
    },
}

IMG_CACHE = {}


def configure_fonts():
    global ARIAL_PATH
    ARIAL_PATH = _get_font_path(('Arial', 'Liberation Sans', 'DejaVu Sans'))
    if ARIAL_PATH:
        try:
            font_manager.fontManager.addfont(ARIAL_PATH)
            name = font_manager.FontProperties(fname=ARIAL_PATH).get_name()
        except Exception:
            name = 'Arial'
    else:
        name = 'Arial'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = [name, 'Arial', 'Liberation Sans', 'DejaVu Sans']
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    return name


def load_component_matrix(results, family, rank):
    return np.asarray(results[family]['components'][rank], float)


def _blank_tile(size):
    return np.full((size, size, 3), 200, np.uint8)


def load_tile(path):
    key = str(path)
    if key in IMG_CACHE:
        return IMG_CACHE[key]
    try:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((IMG_PX, IMG_PX), Image.BILINEAR)
        arr = np.array(im)
        bw = max(1, IMG_PX // 60)
        arr[:bw] = 64
        arr[-bw:] = 64
        arr[:, :bw] = 64
        arr[:, -bw:] = 64
    except Exception:
        arr = _blank_tile(IMG_PX)
    IMG_CACHE[key] = arr
    return arr


def _fig_to_array(fig, target_width):
    fig.tight_layout(pad=0.05)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    arr = buf.reshape(h, w, 4)[..., :3]
    plt.close(fig)
    if h != IMG_PX or w != target_width:
        arr = np.asarray(Image.fromarray(arr).resize((target_width, IMG_PX), Image.BILINEAR))
    return arr


def _text_panel(width, text):
    fig = plt.figure(figsize=(width / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, wrap=True)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return _fig_to_array(fig, width)


def render_label_panel(entry):
    fig = plt.figure(figsize=(LABEL_W / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95)
    ax.set_facecolor(INFO_BG)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INFO_EDGE)
        spine.set_linewidth(0.8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis('off')
    rank_idx = entry['rank_idx']
    comp_id = entry['component']
    ax.text(0.06, 0.73, f"Factor {rank_idx}", ha='left', va='center', fontsize=11, fontweight='bold')
    ax.text(0.06, 0.38, f"({comp_id} raw)", ha='left', va='center', fontsize=9.5)
    return _fig_to_array(fig, LABEL_W)


def render_bar_panel(human, monkey):
    fig = plt.figure(figsize=(BAR_W / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.12, top=0.90)
    ax.set_facecolor(INFO_BG)
    vals = [float(human), float(monkey)]
    vals = [v if np.isfinite(v) else 0.0 for v in vals]
    colors = [HUM_COLOR, MON_COLOR]
    xpos = np.arange(2)
    ax.bar(xpos, vals, color=colors, edgecolor='black', linewidth=0.8, width=0.68)
    ax.set_xticks([])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(['0', '1'])
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35)
    return _fig_to_array(fig, BAR_W)


def _get_font_path(preferred=('Arial', 'Liberation Sans', 'DejaVu Sans')):
    for name in preferred:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if path:
                return path
        except Exception:
            continue
    try:
        return font_manager.findfont('sans-serif')
    except Exception:
        return None


def _trunc_cmap(name='Reds', minval=0.75, maxval=1.0, n=256):
    base = colormaps.get_cmap(name)
    colors = base(np.linspace(minval, maxval, n))
    return mpl.cm.colors.ListedColormap(colors)


def compute_enrichment(weights, cats, cat_counts):
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


def render_wordcloud(enrichment):
    freqs = {cat.replace('_', ' '): max(auc, 0.51) for cat, auc in enrichment if auc > 0.5}
    if not freqs:
        return _text_panel(WC_W, 'Enrichment < 0.5 AUROC')
    return render_text_cloud(freqs)


def render_text_cloud(freqs):
    if not freqs:
        return _text_panel(WC_W, 'Enrichment < 0.5 AUROC')
    items = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)
    vals = [v for _, v in items]
    mn, mx = min(vals), max(vals)
    cmap = _trunc_cmap('Reds', minval=0.75, maxval=1.0)
    fig = plt.figure(figsize=(WC_W / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.94)
    left_ax = ax.inset_axes([0.04, 0.08, 0.42, 0.84])
    right_ax = ax.inset_axes([0.54, 0.08, 0.42, 0.84])
    for subax in (left_ax, right_ax):
        subax.set_xlim(0, 1)
        subax.set_ylim(0, 1)
        subax.axis('off')
    y_levels = [0.72, 0.28]
    font_size = 10

    def _shorten(label, max_chars=6):
        label = label.replace('_', ' ')
        if len(label) <= max_chars:
            return label
        return label[:max_chars - 1].rstrip() + '.'

    for idx, (word, val) in enumerate(items[:4]):
        if mx == mn:
            t = 1.0
        else:
            t = (val - mn) / (mx - mn)
        col = 0 if idx < 2 else 1
        row = idx if idx < 2 else idx - 2
        subax = left_ax if col == 0 else right_ax
        y = y_levels[row]
        r, g, b, _ = cmap(t)
        label = _shorten(word)
        subax.text(0.04, y, label, ha='left', va='center', fontsize=font_size,
                   color=(r, g, b), fontweight='bold', clip_on=True)
    return _fig_to_array(fig, WC_W)


def build_entries(spec_key, spec, guard_df, W, cats, cat_counts):
    df = guard_df.copy()
    mask = spec['filter'](df)
    df = df[mask].copy()
    if df.empty:
        return []
    df['score'] = df.apply(spec['score'], axis=1)
    df.sort_values('score', ascending=False, inplace=True)
    top_n = spec.get('top_n')
    if top_n is not None:
        df = df.head(int(top_n)).copy()
    entries = []
    for rank_idx, (_, row) in enumerate(df.iterrows(), start=1):
        idx = int(row['component']) - 1
        weights = W[:, idx]
        comp_id = int(row['component'])
        entries.append({
            'comp_idx': idx,
            'label': f"{spec['label_prefix']} N{rank_idx} C{comp_id}",
            'rank_idx': rank_idx,
            'component': comp_id,
            'score': float(row['score']),
            'human_r2': float(row['human_r2']),
            'human_r2_std': float(row.get('human_r2_std', np.nan)),
            'monkey_r2': float(row['monkey_r2']),
            'monkey_r2_std': float(row.get('monkey_r2_std', np.nan)),
            'weights': weights,
            'enrichment': compute_enrichment(weights, cats, cat_counts),
        })
    return entries


def assemble_canvas(entries, img_paths):
    if not entries:
        return None, None
    width_images = TOP_IMAGES * IMG_PX
    gap_col = np.full((IMG_PX, GAP_COL, 3), 255, np.uint8)
    total_width = LABEL_W + BAR_W + 3 * GAP_COL + WC_W + width_images
    gap_row = np.full((ROW_GAP, total_width, 3), 255, np.uint8)

    parts = []

    for ent in entries:
        weights = ent['weights']
        order = np.argsort(weights)[::-1][:TOP_IMAGES]
        tiles = [load_tile(img_paths[i]) for i in order]
        while len(tiles) < TOP_IMAGES:
            tiles.append(_blank_tile(IMG_PX))

        img_block = np.hstack(tiles)
        label_panel = render_label_panel(ent)
        bar_panel = render_bar_panel(ent['human_r2'], ent['monkey_r2'])
        wc_panel = render_wordcloud(ent['enrichment'])

        row = np.hstack([
            label_panel,
            gap_col.copy(),
            bar_panel,
            gap_col.copy(),
            wc_panel,
            gap_col.copy(),
            img_block,
        ])
        parts.append(row)
        parts.append(gap_row.copy())

    canvas = np.vstack(parts[:-1])
    return canvas, None


def render_gallery(spec_key, entries, img_paths, title, fname, out_path=None):
    if not entries:
        print(f"[gallery] {spec_key}: no components matched thresholds.")
        return
    render_dpi = int(GALLERY_SPECS[spec_key].get('dpi', DPI))
    out_fp = Path(out_path) if out_path is not None else FIG_DIR / fname
    render_nmf_supp_gallery(
        entries, img_paths, spec_key, title, out_fp, dpi=render_dpi,
    )
    print(f"[gallery] saved {out_fp}")


def save_summary(entries, path):
    if not entries:
        return
    rows = []
    for ent in entries:
        rows.append({
            'label': ent['label'],
            'component': ent['comp_idx'] + 1,
            'human_r2': ent['human_r2'],
            'human_r2_std': ent.get('human_r2_std'),
            'monkey_r2': ent['monkey_r2'],
            'monkey_r2_std': ent.get('monkey_r2_std'),
            'score': ent['score'],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Render SRF galleries from guard CSVs.")
    parser.add_argument('--which', nargs='+', choices=list(GALLERY_SPECS.keys()), default=None,
                        help='Which galleries to build (default: all).')
    args = parser.parse_args()

    configure_fonts()

    results, stims = load_srf_results()
    img_paths = build_img_paths(stims)
    cats = derive_categories(stims)
    cat_counts = Counter(cats)

    targets = args.which or list(GALLERY_SPECS.keys())

    for key in targets:
        spec = GALLERY_SPECS[key]
        family = spec['family']
        guard_df = load_guard_table(family)
        rank = pick_rank(family, results[family])
        W = load_component_matrix(results, family, rank)
        entries = build_entries(key, spec, guard_df, W, cats, cat_counts)

        render_gallery(key, entries, img_paths, spec['title'], spec['filename'])
        summary_fp = RESULT_DIR / spec['summary']
        save_summary(entries, summary_fp)
        if entries:
            print(f"[gallery] {key}: {len(entries)} components listed; summary -> {summary_fp}")


if __name__ == '__main__':
    main()
