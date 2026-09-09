#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from PIL import Image

SPECIES = ('human', 'monkey')
UNIVERSAL = 'all'
DDOF = 1
EPS = 1e-8

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from functions.plotting import (
    setup_style,
    style_axes,
    format_axes,
    square_panel,
    medium_figure,
)
from model.feature.core.feature_io import load_cca_components, load_stims, load_beh

ASYMM_CFG = config.hyperparameters.get('category_asymm', {})
N_PERM = int(ASYMM_CFG.get('n_perm', 200))
SIG_ALPHA = float(ASYMM_CFG.get('alpha', 0.05))
SEED = int(ASYMM_CFG.get('seed', 42))
ANIMACY_PERM = int(ASYMM_CFG.get('animacy_perm', 5000))
N_BOOT = int(ASYMM_CFG.get('n_boot', 1000))


def load_things_properties():
    props_path = Path(config.data_dir) / 'things' / 'things_properties.csv'
    if not props_path.exists():
        raise FileNotFoundError(f'THINGS properties file missing: {props_path}')
    df = pd.read_csv(props_path)
    df['category'] = df['uniqueID'].str.lower().str.replace(' ', '_')
    lives_threshold = 4.0
    category_lives = dict(zip(df['category'], df['lives']))
    return category_lives, lives_threshold


def group_by_cat(stims, cats):
    lookup = {}
    for idx, cat in enumerate(cats):
        lookup.setdefault(cat, []).append(idx)
    names = sorted(lookup)
    groups = [np.asarray(lookup[c], int) for c in names]
    return names, groups


def energy_ratio(X, groups, norm='l2'):
    means = []
    for ids in groups:
        vals = X[ids]
        if vals.shape[0] == 0:
            means.append(np.nan)
            continue
        if norm == 'l1':
            energy = np.sum(np.abs(vals), axis=1)
        else:
            energy = np.sum(vals ** 2, axis=1)
        means.append(float(np.mean(energy)))
    return np.asarray(means, float)


def load_components():
    comps = {}
    order = None
    for fam in SPECIES + (UNIVERSAL,):
        X, st = load_cca_components(fam)
        comps[fam] = np.asarray(X, float)
        order = st if order is None else order
        if st != order:
            raise ValueError('Stimulus order mismatch among CCA components.')
    return comps, order


def find_behavior_dimension(names, key='body part'):
    key = key.lower()
    for i, name in enumerate(names):
        if key in name.lower():
            return i, name
    return None, None


def load_bodypart_scores(order):
    X, names, _ = load_beh(order)
    if X.size == 0 or not names:
        return None, None
    idx, label = find_behavior_dimension(names, key='body part')
    if idx is None:
        return None, None
    return X[:, idx], label


def build_table(norm='l1'):
    comps, order = load_components()
    all_stims, all_cats = load_stims()
    cat_map = {s: c for s, c in zip(all_stims, all_cats)}
    cats_order = [cat_map[s] for s in order]
    cat_names, groups = group_by_cat(order, cats_order)

    stats = {}
    for fam, X in comps.items():
        stats[fam] = energy_ratio(X, groups, norm)

    suffix = '_l1' if norm == 'l1' else '_l2'
    df = pd.DataFrame({
        'category': cat_names,
        f'{SPECIES[0]}_energy{suffix}': stats[SPECIES[0]],
        f'{SPECIES[1]}_energy{suffix}': stats[SPECIES[1]],
        f'{UNIVERSAL}_energy{suffix}': stats[UNIVERSAL],
    })

    human_col = f'{SPECIES[0]}_energy{suffix}'
    monkey_col = f'{SPECIES[1]}_energy{suffix}'
    universal_col = f'{UNIVERSAL}_energy{suffix}'
    df['human_rel'] = (df[human_col] + EPS) / (df[universal_col] + EPS)
    df['monkey_rel'] = (df[monkey_col] + EPS) / (df[universal_col] + EPS)
    df['human_rel_z'] = (df['human_rel'] - df['human_rel'].mean()) / df['human_rel'].std(ddof=DDOF)
    df['monkey_rel_z'] = (df['monkey_rel'] - df['monkey_rel'].mean()) / df['monkey_rel'].std(ddof=DDOF)
    df['species_asym_z'] = df['human_rel_z'] - df['monkey_rel_z']
    return df, comps, groups, order, cats_order


def perm_thresholds(comps, groups, n_perm, n_jobs, norm='l1'):
    n = next(iter(comps.values())).shape[0]
    seeds = np.random.SeedSequence(SEED).spawn(n_perm)

    def worker(seed):
        rng = np.random.default_rng(seed.generate_state(1)[0])
        perm = rng.permutation(n)
        energies = {}
        for fam, X in comps.items():
            energies[fam] = energy_ratio(X[perm], groups, norm)
        human_rel = (energies[SPECIES[0]] + EPS) / (energies[UNIVERSAL] + EPS)
        monkey_rel = (energies[SPECIES[1]] + EPS) / (energies[UNIVERSAL] + EPS)
        h_z = (human_rel - human_rel.mean()) / human_rel.std(ddof=DDOF)
        m_z = (monkey_rel - monkey_rel.mean()) / monkey_rel.std(ddof=DDOF)
        return h_z - m_z

    outs = Parallel(n_jobs=n_jobs)(delayed(worker)(sd) for sd in seeds)
    samples = np.concatenate(outs)
    lo = np.quantile(samples, SIG_ALPHA / 2)
    hi = np.quantile(samples, 1 - SIG_ALPHA / 2)
    return (lo, hi)


def load_animacy(df):
    lives_map, threshold = load_things_properties()
    cats = df['category'].astype(str).str.lower().str.replace(' ', '_')
    scores = np.array([lives_map.get(c, threshold) for c in cats], float)
    df['animacy_score'] = scores
    df['is_living'] = scores >= threshold
    return df, threshold


def attach_bodypart_scores(df, groups, order):
    stim_scores, label = load_bodypart_scores(order)
    if stim_scores is None:
        df['bodypart_score'] = 0.0
        return df, np.zeros(len(order)), label
    cat_scores = []
    for ids in groups:
        vals = stim_scores[ids]
        cat_scores.append(float(np.nanmean(vals)))
    df['bodypart_score'] = cat_scores
    return df, stim_scores, label


def safe_corr(a, b):
    if a.size == 0 or b.size == 0:
        return np.nan
    da = a - a.mean()
    db = b - b.mean()
    denom = np.sqrt(np.sum(da ** 2) * np.sum(db ** 2))
    if denom == 0:
        return np.nan
    return float(np.sum(da * db) / denom)


def corr_with_perm(x, y, n_perm=ANIMACY_PERM, seed=SEED):
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, float)[mask]
    y = np.asarray(y, float)[mask]
    if len(x) < 3:
        return np.nan, np.nan
    r = safe_corr(x, y)
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_perm, float)
    for i in range(n_perm):
        perm_stats[i] = safe_corr(x, rng.permutation(y))
    p = (np.sum(np.abs(perm_stats) >= abs(r)) + 1) / (n_perm + 1)
    return r, float(p)


def bootstrap_ci(arrays, func, n_boot=N_BOOT, seed=SEED + 555):
    arrays = [np.asarray(a, float) for a in arrays]
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    arrays = [arr[mask] for arr in arrays]
    if len(arrays[0]) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = np.arange(len(arrays[0]))
    vals = np.empty(n_boot, float)
    for i in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        sampled = [arr[sample] for arr in arrays]
        vals[i] = func(*sampled)
    low, high = np.nanpercentile(vals, [2.5, 97.5])
    return float(low), float(high)


def partial_corr(x, y, z):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = np.asarray(x, float)[mask]
    y = np.asarray(y, float)[mask]
    z = np.asarray(z, float)[mask]
    if len(x) < 4:
        return np.nan

    def regress(resp):
        z_mean = z.mean()
        r_mean = resp.mean()
        var_z = np.sum((z - z_mean) ** 2)
        if var_z == 0:
            return 0.0, r_mean
        slope = float(np.sum((z - z_mean) * (resp - r_mean)) / var_z)
        intercept = float(r_mean - slope * z_mean)
        return slope, intercept

    sx, ix = regress(x)
    sy, iy = regress(y)
    resid_x = x - (sx * z + ix)
    resid_y = y - (sy * z + iy)
    return safe_corr(resid_x, resid_y)


def partial_corr_with_perm(x, y, z, n_perm=ANIMACY_PERM, seed=SEED + 99):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = np.asarray(x, float)[mask]
    y = np.asarray(y, float)[mask]
    z = np.asarray(z, float)[mask]
    if len(x) < 4:
        return np.nan, np.nan

    r = partial_corr(x, y, z)
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_perm, float)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        perm_stats[i] = partial_corr(x, y_perm, z)
    p = (np.sum(np.abs(perm_stats) >= abs(r)) + 1) / (n_perm + 1)
    return r, float(p)


def add_jitter(x, y, jitter_strength=0.008):
    np.random.seed(42)
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    x_jit = np.random.normal(0, jitter_strength * (x_range if x_range > 0 else 1.0), len(x))
    y_jit = np.random.normal(0, jitter_strength * (y_range if y_range > 0 else 1.0), len(y))
    return x + x_jit, y + y_jit


def plot_scatter(df, out_path, thresholds, sig_col):
    fig, ax = square_panel()
    x_all = df['monkey_rel_z'].to_numpy()
    y_all = df['human_rel_z'].to_numpy()
    sig_mask = df[sig_col].to_numpy()
    if 'is_living' not in df.columns:
        raise ValueError('is_living missing; animacy annotation required.')
    is_living = df['is_living'].to_numpy(dtype=bool)

    overall_min = min(np.nanmin(x_all), np.nanmin(y_all))
    overall_max = max(np.nanmax(x_all), np.nanmax(y_all))
    overall_range = overall_max - overall_min if overall_max > overall_min else 1.0
    lim_min = overall_min - 0.10 * overall_range
    lim_max = overall_max + 0.10 * overall_range
    x_jit, y_jit = add_jitter(x_all, y_all)

    x_span = np.linspace(lim_min, lim_max, 300)
    ax.fill_between(x_span, x_span + thresholds[0], x_span + thresholds[1], color='0.85', alpha=0.4, zorder=0)

    GREY_DOT = '#888888'
    HUMAN_COLOR = '#7c5799'
    MONKEY_COLOR = '#bda855'
    EDGE_DARK = '#333333'

    ns = ~sig_mask
    ax.scatter(x_jit[ns & is_living], y_jit[ns & is_living], c=GREY_DOT, s=25, marker='o', edgecolors='none', alpha=0.35, zorder=2)
    ax.scatter(x_jit[ns & ~is_living], y_jit[ns & ~is_living], c=GREY_DOT, s=25, marker='^', edgecolors='none', alpha=0.35, zorder=2)

    above = y_all > x_all
    sig_human = sig_mask & above
    sig_monkey = sig_mask & ~above
    ax.scatter(x_jit[sig_human & is_living], y_jit[sig_human & is_living], c=HUMAN_COLOR, s=55, marker='o', edgecolors=EDGE_DARK, linewidths=1.2, alpha=0.9, zorder=3)
    ax.scatter(x_jit[sig_human & ~is_living], y_jit[sig_human & ~is_living], c=HUMAN_COLOR, s=55, marker='^', edgecolors=EDGE_DARK, linewidths=1.2, alpha=0.9, zorder=3)
    ax.scatter(x_jit[sig_monkey & is_living], y_jit[sig_monkey & is_living], c=MONKEY_COLOR, s=55, marker='o', edgecolors=EDGE_DARK, linewidths=1.2, alpha=0.9, zorder=3)
    ax.scatter(x_jit[sig_monkey & ~is_living], y_jit[sig_monkey & ~is_living], c=MONKEY_COLOR, s=55, marker='^', edgecolors=EDGE_DARK, linewidths=1.2, alpha=0.9, zorder=3)

    ax.plot([lim_min, lim_max], [lim_min, lim_max], ls='--', color='0.4', lw=2)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel('Monkey bias (Z)')
    ax.set_ylabel('Human bias (Z)')

    most_human_idx = df['species_asym_z'].idxmax()
    most_monkey_idx = df['species_asym_z'].idxmin()
    for eidx, ecol in [(most_human_idx, HUMAN_COLOR), (most_monkey_idx, MONKEY_COLOR)]:
        ex, ey = df.loc[eidx, 'monkey_rel_z'], df.loc[eidx, 'human_rel_z']
        emk = 'o' if df.loc[eidx, 'is_living'] else '^'
        ax.scatter(ex, ey, c=ecol, s=55, marker=emk, edgecolors=EDGE_DARK, linewidths=2.4, alpha=0.9, zorder=4)
        ax.text(ex + 0.1, ey + 0.1, df.loc[eidx, 'category'], fontsize=9)

    style_axes(ax)
    format_axes(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    legend_path = out_path.with_name(out_path.stem + '_legend.pdf')
    legend_fig = plt.figure(figsize=(3, 2))
    legend_ax = legend_fig.add_subplot(111)
    style_axes(legend_ax)
    format_axes(legend_ax)
    legend_ax.scatter(0.05, 0.8, c=HUMAN_COLOR, s=55, marker='o', edgecolors=EDGE_DARK, linewidths=1.2)
    legend_ax.text(0.15, 0.8, 'Human-biased', va='center')
    legend_ax.scatter(0.05, 0.6, c=MONKEY_COLOR, s=55, marker='o', edgecolors=EDGE_DARK, linewidths=1.2)
    legend_ax.text(0.15, 0.6, 'Monkey-biased', va='center')
    legend_ax.scatter(0.05, 0.4, c='black', s=55, marker='o')
    legend_ax.text(0.15, 0.4, 'Living', va='center')
    legend_ax.scatter(0.05, 0.2, c='black', s=55, marker='^')
    legend_ax.text(0.15, 0.2, 'Non-living', va='center')
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')
    legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close(legend_fig)


def plot_asymmetry_bars(df, out_path, top_k=12):
    """
    Horizontal bar plot (old aesthetics):
      - Top: top_k most human-asymmetric (positive species_asym_z) bars going RIGHT.
      - Bottom: top_k most monkey-asymmetric (negative species_asym_z) bars going LEFT.
      - Non-significant bars use a duller version of the species color.
      - Category names placed at the middle of each bar.
    """
    # Select extremes
    human_top = df.nlargest(top_k, 'species_asym_z').copy()
    monkey_top = df.nsmallest(top_k, 'species_asym_z').copy()

    # Sort each side by magnitude - longest bars at top
    human_top = human_top.sort_values('species_asym_z', ascending=True)   # Lowest first (for top placement)
    monkey_top = monkey_top.sort_values('species_asym_z', ascending=False) # Least negative first (for top placement)

    # Colors: gradient based on rank (position), with significance distinction
    def colorize(series_is_sig, values, species='human'):
        base = config.plotting.get('human_color', '#7c5799') if species == 'human' else config.plotting.get('monkey_color', '#bda855')
        base_rgba = plt.matplotlib.colors.to_rgba(base)

        # Create rank-based gradient (0 to 1 from first to last position)
        n_items = len(values)

        cols = []
        for i, sig in enumerate(series_is_sig):
            if sig:
                # Significant: gradient from 0.3 (first/lowest) to 0.7 (last/highest)
                rank_intensity = i / (n_items - 1) if n_items > 1 else 0.0
                alpha = 0.3 + (0.4 * rank_intensity)  # Range: 0.3 to 0.7
            else:
                # Non-significant: constant low alpha
                alpha = 0.25
            cols.append((*base_rgba[:3], alpha))
        return cols

    human_colors = colorize(human_top['significant'].to_numpy(), human_top['species_asym_z'].to_numpy(), 'human')
    monkey_colors = colorize(monkey_top['significant'].to_numpy(), monkey_top['species_asym_z'].to_numpy(), 'monkey')

    # Figure - use square format but adjust height manually for better proportions
    fig = medium_figure(height='square')
    ax = fig.add_subplot(111)

    # Y positions - same levels for both sides
    y_pos = np.arange(top_k)

    EDGE_DARK = '#333333'

    # Human bars (positive, extending right) - full height to connect without gaps
    human_vals = human_top['species_asym_z'].to_numpy()
    ax.barh(y_pos, human_vals, height=1.0, color=human_colors, edgecolor=EDGE_DARK, linewidth=2.5, zorder=3)

    # Monkey bars (negative, extending left) - full height to connect without gaps
    monkey_vals = monkey_top['species_asym_z'].to_numpy()
    ax.barh(y_pos, monkey_vals, height=1.0, color=monkey_colors, edgecolor=EDGE_DARK, linewidth=2.5, zorder=3)

    # Baseline - thicker midline
    ax.axvline(0.0, color='0.3', lw=3.0, zorder=4)

    # Labels at the center of each bar
    for i, (val, name) in enumerate(zip(human_vals, human_top['category'].to_list())):
        ax.text(val / 2, i, name, ha='center', va='center', fontsize=9, color='black')

    for i, (val, name) in enumerate(zip(monkey_vals, monkey_top['category'].to_list())):
        ax.text(val / 2, i, name, ha='center', va='center', fontsize=9, color='black')

    # Add x-axis while keeping clean appearance
    ax.set_yticks([])  # No y-axis labels
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Keep bottom spine for x-axis
    ax.spines['bottom'].set_visible(True)

    # Set fixed x-axis limits
    ax.set_xlim(-7, 7)
    ax.set_ylim(-0.5, top_k - 0.5)  # Tight to bars

    # Set x-axis label
    ax.set_xlabel('Asymmetry (Human - Monkey, Z)')

    # Apply styling to ensure consistent appearance
    style_axes(ax)
    format_axes(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_contrib_bars(stats, out_path):
    # Minimal bars: Animacy, Body part | Animacy, Animacy | Body part
    labels = ['Animacy', 'Body part | Animacy', 'Animacy | Body part']
    values = [
        stats['animacy_corr'],
        stats['body_partial_corr'],
        stats['anim_partial_corr'],
    ]

    fig = medium_figure(height='square')
    ax = fig.add_subplot(111)
    xs = np.arange(len(labels))
    human_color = config.plotting.get('human_color', '#7c5799')
    monkey_color = config.plotting.get('monkey_color', '#bda855')

    def _lighten(hex_color, mix=0.2):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        # blend toward white for subtler fill
        r = r * (1 - mix) + 1.0 * mix
        g = g * (1 - mix) + 1.0 * mix
        b = b * (1 - mix) + 1.0 * mix
        return (r, g, b)

    colors = [_lighten(human_color, mix=0.15) if v >= 0 else _lighten(monkey_color, mix=0.15) for v in values]
    ax.bar(xs, values, color=colors, edgecolor='#2e2e2e', linewidth=1.8, zorder=3, width=0.75)
    ax.axhline(0, color='#333333', linewidth=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(['Animacy', 'Body\n(~ Animacy)', 'Animacy\n(~ Body)'], rotation=45, ha='right')
    ylim = max(0.05, np.nanmax(np.abs(values)) + 0.05)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylabel('Correlation (r)')
    style_axes(ax)
    format_axes(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.22, right=0.95)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def summarize(df):
    stats = {
        'mean_species_asym_z': float(df['species_asym_z'].mean()),
        'std_species_asym_z': float(df['species_asym_z'].std()),
        'median_energy_diff_rel': float((df['human_rel'] - df['monkey_rel']).median()),
        'n_sig': int(df['significant'].sum()),
        'n_total': int(len(df)),
        'mean_animacy_score': float(df['animacy_score'].mean()),
        'mean_bodypart_score': float(df['bodypart_score'].mean()),
    }
    living = df['is_living'].to_numpy(dtype=bool)
    asym = df['species_asym_z'].to_numpy()
    stats.update({
        'mean_living_asym': float(asym[living].mean()),
        'mean_nonliving_asym': float(asym[~living].mean()),
    })
    return stats


def save_contrib_stats(stats, out_path):
    lines = [
        f"Animacy               r = {stats['animacy_corr']:.3f} (p = {stats['animacy_p']:.4g})",
        f"Body part | Animacy   r = {stats['body_partial_corr']:.3f} (p = {stats['body_partial_p']:.4g})",
        f"Animacy | Body part   r = {stats['anim_partial_corr']:.3f} (p = {stats['anim_partial_p']:.4g})",
        f"Animacy vs Body part  r = {stats['animacy_body_corr']:.3f} (p = {stats['animacy_body_p']:.4g})",
        "",
        f"Bootstrap 95% CI:",
        f"  Animacy             [{stats['animacy_ci_low']:.3f}, {stats['animacy_ci_high']:.3f}]",
        f"  Body part | Animacy [{stats['body_partial_ci_low']:.3f}, {stats['body_partial_ci_high']:.3f}]",
        f"  Animacy | Body part [{stats['anim_partial_ci_low']:.3f}, {stats['anim_partial_ci_high']:.3f}]",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def print_significant(df):
    sig = df['significant'].to_numpy()
    human_sig = df[sig & (df['species_asym_z'] > 0)].sort_values('species_asym_z', ascending=False)
    monkey_sig = df[sig & (df['species_asym_z'] < 0)].sort_values('species_asym_z', ascending=True)
    print(f'[significance] total significant: {sig.sum()} / {len(df)}')
    print(f'[significance] human-biased (count={len(human_sig)}):')
    if len(human_sig):
        print(human_sig[['category', 'species_asym_z', 'human_rel_z', 'monkey_rel_z']].to_string(index=False))
    print(f'[significance] monkey-biased (count={len(monkey_sig)}):')
    if len(monkey_sig):
        print(monkey_sig[['category', 'species_asym_z', 'human_rel_z', 'monkey_rel_z']].to_string(index=False))


def save_bodypart_grid(order, cats_order, stim_scores, out_path, top_n=100, tile=128, cols=10):
    if stim_scores is None or len(stim_scores) == 0:
        return
    base = Path(config.data_dir) / 'things' / 'images'
    idxs = np.argsort(stim_scores)[::-1][:top_n]
    rows = int(np.ceil(len(idxs) / cols))
    canvas = Image.new('RGB', (cols * tile, rows * tile), (245, 245, 245))
    exts = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')

    def resolve(stim, cat):
        stem = Path(stim).stem
        candidates = [base / cat, base / cat.lower(), base]
        for folder in candidates:
            for ext in exts:
                fp = folder / f'{stem}{ext}'
                if fp.exists():
                    return fp
        for folder in base.iterdir() if base.exists() else []:
            if folder.is_dir() and folder.name.lower() == cat.lower():
                for ext in exts:
                    fp = folder / f'{stem}{ext}'
                    if fp.exists():
                        return fp
        return None

    for i, idx in enumerate(idxs):
        stim = order[idx]
        cat = cats_order[idx]
        path = resolve(stim, cat)
        r, c = divmod(i, cols)
        try:
            if path is None:
                raise FileNotFoundError
            im = Image.open(path).convert('RGB').resize((tile, tile), Image.BILINEAR)
        except Exception:
            im = Image.new('RGB', (tile, tile), (200, 200, 200))
        canvas.paste(im, (c * tile, r * tile))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


def main():
    setup_style()
    n_jobs = int(config.analysis.get('n_jobs', 4))
    norm = 'l1'
    print(f'\n=== {norm.upper()} NORM ANALYSIS ===')

    df, comps, groups, order, cats_order = build_table(norm)
    df, _ = load_animacy(df)
    df, stim_scores, beh_label = attach_bodypart_scores(df, groups, order)
    thresholds = perm_thresholds(comps, groups, N_PERM, n_jobs, norm)
    lo, hi = thresholds

    df['sig_lower'] = lo
    df['sig_upper'] = hi
    df['significant'] = (df['species_asym_z'] <= lo) | (df['species_asym_z'] >= hi)

    animacy_corr, animacy_p = corr_with_perm(df['animacy_score'], df['species_asym_z'])
    body_corr, body_p = corr_with_perm(df['bodypart_score'], df['species_asym_z'])
    body_partial_corr, body_partial_p = partial_corr_with_perm(df['bodypart_score'], df['species_asym_z'], df['animacy_score'])
    anim_partial_corr, anim_partial_p = partial_corr_with_perm(df['animacy_score'], df['species_asym_z'], df['bodypart_score'])
    anim_body_corr, anim_body_p = corr_with_perm(df['animacy_score'], df['bodypart_score'])

    stats = summarize(df)
    stats.update({
        'animacy_corr': animacy_corr,
        'animacy_p': animacy_p,
        'body_corr': body_corr,
        'body_p': body_p,
        'body_partial_corr': body_partial_corr,
        'body_partial_p': body_partial_p,
        'anim_partial_corr': anim_partial_corr,
        'anim_partial_p': anim_partial_p,
        'animacy_body_corr': anim_body_corr,
        'animacy_body_p': anim_body_p,
    })

    anim_ci = bootstrap_ci([df['animacy_score'], df['species_asym_z']], lambda a, b: safe_corr(a, b))
    body_ci = bootstrap_ci([df['bodypart_score'], df['species_asym_z']], lambda a, b: safe_corr(a, b))
    body_partial_ci = bootstrap_ci([df['bodypart_score'], df['species_asym_z'], df['animacy_score']], partial_corr)
    anim_partial_ci = bootstrap_ci([df['animacy_score'], df['species_asym_z'], df['bodypart_score']], partial_corr)
    stats.update({
        'animacy_ci_low': anim_ci[0],
        'animacy_ci_high': anim_ci[1],
        'body_ci_low': body_ci[0],
        'body_ci_high': body_ci[1],
        'body_partial_ci_low': body_partial_ci[0],
        'body_partial_ci_high': body_partial_ci[1],
        'anim_partial_ci_low': anim_partial_ci[0],
        'anim_partial_ci_high': anim_partial_ci[1],
    })

    _here = Path(__file__).resolve().parent
    res_dir = _here / 'results'
    fig_dir = _here / 'figures'
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_path = res_dir / f'energy_{norm}_metrics.csv'
    df.sort_values('species_asym_z').to_csv(df_path, index=False)
    summary = pd.DataFrame([stats])
    summary.to_csv(res_dir / f'energy_{norm}_summary.csv', index=False)
    save_contrib_stats(stats, res_dir / 'category_contrib_stats.txt')
    print(f'Saved metrics to {df_path}')
    print(f'Permutation thresholds: [{lo:.4f}, {hi:.4f}] using {N_PERM} shuffles (n_jobs={n_jobs}).')
    print(f'[animacy] r={animacy_corr:.4f}, p={animacy_p:.4g}')
    print(f'[body] r={body_corr:.4f}, p={body_p:.4g}')
    print(f'[body | animacy] r={body_partial_corr:.4f}, p={body_partial_p:.4g}')
    print(f'[animacy | body] r={anim_partial_corr:.4f}, p={anim_partial_p:.4g}')
    print(f'[animacy vs body] r={anim_body_corr:.4f}, p={anim_body_p:.4g}')

    print_significant(df)

    scatter_path = fig_dir / 'cat_asymm_scatter.pdf'
    plot_scatter(df, scatter_path, thresholds, sig_col='significant')
    print(f'Saved scatter to {scatter_path} and legend to {scatter_path.with_name(scatter_path.stem + "_legend.pdf")}')

    dist_path = fig_dir / 'cat_asymm_dist.pdf'
    plot_asymmetry_bars(df, dist_path, top_k=12)
    print(f'Saved asymmetry distribution to {dist_path}')

    contrib_path = fig_dir / 'cat_contrib_bars.pdf'
    plot_contrib_bars({
        'animacy_corr': stats['animacy_corr'],
        'body_corr': stats['body_corr'],
        'body_partial_corr': stats['body_partial_corr'],
        'anim_partial_corr': stats['anim_partial_corr'],
        'animacy_p': animacy_p,
        'body_p': body_p,
        'body_partial_p': body_partial_p,
        'anim_partial_p': anim_partial_p,
        'animacy_ci': (stats['animacy_ci_low'], stats['animacy_ci_high']),
        'body_ci': (stats['body_ci_low'], stats['body_ci_high']),
        'body_partial_ci': (stats['body_partial_ci_low'], stats['body_partial_ci_high']),
        'anim_partial_ci': (stats['anim_partial_ci_low'], stats['anim_partial_ci_high']),
    }, contrib_path)
    print(f'Saved contribution bars to {contrib_path}')

    if beh_label:
        grid_path = fig_dir / 'bodypart_beh_top100_grid.jpg'
        save_bodypart_grid(order, cats_order, stim_scores, grid_path)
        print(f'Saved body-part behavior grid ({beh_label}) to {grid_path}')


if __name__ == '__main__':
    main()
