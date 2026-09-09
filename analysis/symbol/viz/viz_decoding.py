#!/usr/bin/env python3
"""Unified visualization for decoding results."""
import sys
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from functions.plotting import setup_style, format_axes, style_axes, wide_figure
from scipy import stats

# Config ----------------------------------------------------------------------
FEATURES = ['symbol', 'icon', 'ensemble', 'face', 'body_part', 'random']
ORTHO_FEATURES = ['symbol', 'icon', 'ensemble']
RESULTS_DIR = Path(config.root_dir) / 'analysis' / 'symbol' / 'results'
FIG_DIR = config.fig_dir / 'symbol' / 'decoding'
FIG_SUPP_DIR = config.fig_dir / 'symbol' / 'supplementary'
LEGEND_DIR = FIG_DIR / 'legend'
STATS_DIR = Path(config.root_dir) / 'analysis' / 'symbol' / 'results' / 'statistics'
STATS_DIR_MAIN = STATS_DIR / 'main'
STATS_DIR_SUPP = STATS_DIR / 'supplementary'

RESULT_FILE = 'decode_main.pkl'
RES_RESULT_FILE = 'decode_residualized_main.pkl'

CONTROL_COLORS = {
    'face': '#7B9E8A',
    'body_part': '#A8B5A0'
}
RANDOM_BASE_COLOR = '#2B2B2B'
RANDOM_LIGHTNESS = {
    'v1': 0.55,
    'v4': 0.4,
    'it': 0.25,
    'residualized': 0.35,
    'default': 0.35
}
SYMBOL_LIGHTEN_MIX = 0.35
ENSEMBLE_DARKEN_MIX = 0.25
FACE_LIGHTEN_MIX = 0.35
BODY_DARKEN_MIX = 0.2
RANDOM_MIX_DEFAULT = RANDOM_LIGHTNESS.get('it', RANDOM_LIGHTNESS['default'])
HUMAN_LIGHTEN_MIX = 0.35


# Utilities -------------------------------------------------------------------
def cleanup_outputs():
    """Remove stale figures/statistics prior to regeneration."""
    for path in (FIG_DIR, FIG_SUPP_DIR):
        if path.exists():
            shutil.rmtree(path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_SUPP_DIR.mkdir(parents=True, exist_ok=True)
    LEGEND_DIR.mkdir(parents=True, exist_ok=True)

    if STATS_DIR.exists():
        shutil.rmtree(STATS_DIR)
    STATS_DIR_MAIN.mkdir(parents=True, exist_ok=True)
    STATS_DIR_SUPP.mkdir(parents=True, exist_ok=True)


def darken_color(color, mix=0.3):
    mix = max(0.0, min(1.0, mix))
    rgb = np.array(to_rgb(color))
    black = np.zeros(3)
    return tuple(rgb * (1 - mix) + black * mix)


def get_feature_colors(region=None):
    base_icon = config.plotting.get('symbol_color', '#2874A6')
    icon_color = to_rgb(base_icon)
    symbol_color = lighten_color(base_icon, mix=SYMBOL_LIGHTEN_MIX)
    ensemble_color = darken_color(base_icon, mix=ENSEMBLE_DARKEN_MIX)

    face_base = CONTROL_COLORS['face']
    body_base = CONTROL_COLORS['body_part']
    face_color = lighten_color(face_base, mix=FACE_LIGHTEN_MIX)
    body_color = darken_color(body_base, mix=BODY_DARKEN_MIX)

    colors = {
        'symbol': symbol_color,
        'icon': icon_color,
        'ensemble': ensemble_color,
        'face': face_color,
        'body_part': body_color,
    }
    colors['random'] = lighten_color(RANDOM_BASE_COLOR, mix=RANDOM_MIX_DEFAULT)
    return colors


def lighten_color(color, mix=0.55):
    mix = max(0.0, min(1.0, mix))
    rgb = np.array(to_rgb(color))
    white = np.ones(3)
    return tuple(rgb * (1 - mix) + white * mix)


def get_feature_labels():
    return {
        'symbol': 'Symbol',
        'icon': 'Icon',
        'ensemble': 'Ensemble',
        'face': 'Face',
        'body_part': 'Body part',
        'random': 'Random'
    }


def fdr_correction(p_vals):
    p_vals = np.asarray(p_vals, dtype=float)
    finite = np.isfinite(p_vals)
    q_vals = np.full_like(p_vals, np.nan, dtype=float)
    if not finite.any():
        return q_vals
    idx = np.where(finite)[0]
    order = np.argsort(p_vals[finite])
    ordered = p_vals[finite][order]
    n = len(ordered)
    prev = 1.0
    q_temp = np.empty(n, dtype=float)
    for i in range(n - 1, -1, -1):
        q = min(ordered[i] * n / (i + 1), prev)
        q_temp[i] = q
        prev = q
    q_vals[idx[order]] = q_temp
    return q_vals


def save_stats_table(stats_data, out_path):
    df = pd.DataFrame(stats_data)
    df.to_csv(out_path, index=False, float_format='%.6f')
    print(f"Saved statistics: {out_path}")


def paired_stats(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return None
    a = a[:n]
    b = b[:n]
    t_val, p_val = stats.ttest_rel(a, b, nan_policy='omit')
    return {
        'delta': float(np.nanmean(a - b)),
        't_val': float(t_val),
        'p_val': float(p_val),
        'n_pairs': int(np.isfinite(a - b).sum())
    }


def ensure_species_legend(base_color, human_color):
    path = LEGEND_DIR / 'decode_species.pdf'
    if path.exists():
        return
    setup_style()
    fig = plt.figure(figsize=(3.2, 2.2))
    ax = fig.add_subplot(111)
    handles = [
        Patch(facecolor=base_color, edgecolor='black', linewidth=1.2, label='Monkey projection'),
        Patch(facecolor=human_color, edgecolor='black', linewidth=1.2, label='Human projection')
    ]
    ax.legend(handles=handles, loc='center', frameon=False)
    ax.axis('off')
    fig.savefig(path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"Saved legend: {path}")


def ensure_residual_legend(base_color, human_color):
    path = LEGEND_DIR / 'decode_residualized_species.pdf'
    if path.exists():
        return
    setup_style()
    fig = plt.figure(figsize=(3.2, 2.2))
    ax = fig.add_subplot(111)
    handles = [
        Patch(facecolor=base_color, edgecolor='black', linewidth=1.2, hatch='///', label='Monkey residualized'),
        Patch(facecolor=human_color, edgecolor='black', linewidth=1.2, hatch='///', label='Human residualized')
    ]
    ax.legend(handles=handles, loc='center', frameon=False)
    ax.axis('off')
    fig.savefig(path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"Saved legend: {path}")


def paired_q_map(paired, features):
    feats, p_vals = [], []
    for feat in features:
        entry = paired.get(feat)
        if entry and np.isfinite(entry['p_val']):
            feats.append(feat)
            p_vals.append(entry['p_val'])
    if not feats:
        return {}
    q_vals = fdr_correction(np.array(p_vals, dtype=float))
    return {feat: q for feat, q in zip(feats, q_vals)}


def build_contrast_map(records):
    """Return {(feature, contrast): record} for quick lookup."""
    lookup = {}
    for r in records or []:
        key = (r.get('feature'), r.get('contrast'))
        lookup[key] = r
    return lookup


# Data loading ----------------------------------------------------------------
def load_region_main(region):
    path = RESULTS_DIR / region / RESULT_FILE
    if not path.exists():
        print(f"No decoding results for {region.upper()}: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_residualized_results():
    path = RESULTS_DIR / RES_RESULT_FILE
    if not path.exists():
        print(f"No residualized results: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_progression_data():
    data = {}
    for region in ('v1', 'v4', 'it'):
        main = load_region_main(region)
        if main:
            data[region] = main['results']['per_species']
    return data


def compute_region_contrasts(region_data):
    """Compute IT-V4 and IT-V1 contrasts per species with FDR."""
    records = {'monkey': [], 'human': []}
    for species in ('monkey', 'human'):
        for feat in FEATURES:
            it = region_data.get('it', {}).get(species, {}).get(feat)
            v4 = region_data.get('v4', {}).get(species, {}).get(feat)
            v1 = region_data.get('v1', {}).get(species, {}).get(feat)
            if it and v4:
                stats_pair = paired_stats(it['fold_aucs'], v4['fold_aucs'])
                if stats_pair:
                    records[species].append({
                        'feature': feat,
                        'contrast': 'IT-V4',
                        'delta': stats_pair['delta'],
                        't_val': stats_pair['t_val'],
                        'p_val': stats_pair['p_val'],
                        'n_pairs': stats_pair['n_pairs'],
                        'it_mean_auc': it['mean_auc'],
                        'other_mean_auc': v4['mean_auc']
                    })
            if it and v1:
                stats_pair = paired_stats(it['fold_aucs'], v1['fold_aucs'])
                if stats_pair:
                    records[species].append({
                        'feature': feat,
                        'contrast': 'IT-V1',
                        'delta': stats_pair['delta'],
                        't_val': stats_pair['t_val'],
                        'p_val': stats_pair['p_val'],
                        'n_pairs': stats_pair['n_pairs'],
                        'it_mean_auc': it['mean_auc'],
                        'other_mean_auc': v1['mean_auc']
                    })

    for species, recs in records.items():
        if not recs:
            continue
        p_vals = [r['p_val'] for r in recs]
        q_vals = fdr_correction(np.array(p_vals, dtype=float))
        for r, q in zip(recs, q_vals):
            r['q_val_fdr'] = q
    return records


def compute_species_region_contrasts(region_data, other='v4', features=ORTHO_FEATURES):
    """Compare the IT-minus-other gain between species (foldwise human minus macaque)."""
    records = []
    for feat in features:
        gains = {}
        for species in ('human', 'monkey'):
            it = region_data.get('it', {}).get(species, {}).get(feat)
            ref = region_data.get(other, {}).get(species, {}).get(feat)
            if not it or not ref:
                break
            gains[species] = np.asarray(it['fold_aucs'], float) - np.asarray(ref['fold_aucs'], float)
        if len(gains) < 2:
            continue
        stats_pair = paired_stats(gains['human'], gains['monkey'])
        if not stats_pair:
            continue
        records.append({
            'feature': feat,
            'contrast': f"IT-{other.upper()}",
            'delta_delta_auc': stats_pair['delta'],
            't_val': stats_pair['t_val'],
            'p_val': stats_pair['p_val'],
            'n_pairs': stats_pair['n_pairs'],
            'human_gain': float(gains['human'].mean()),
            'macaque_gain': float(gains['monkey'].mean())
        })
    if records:
        q_vals = fdr_correction(np.array([r['p_val'] for r in records], dtype=float))
        for r, q in zip(records, q_vals):
            r['q_val_fdr'] = q
    return records


def compute_residualization_contrasts(main_data, residualized):
    """Compare residualized vs main IT decoding within each species."""
    records = []
    for species in ('monkey', 'human'):
        base = main_data.get(species, {})
        res = residualized.get(species, {}).get('features', {})
        for feat in FEATURES:
            a = res.get(feat)
            b = base.get(feat)
            if not a or not b:
                continue
            stats_pair = paired_stats(a['fold_aucs'], b['fold_aucs'])
            if not stats_pair:
                continue
            records.append({
                'species': species,
                'feature': feat,
                'delta_residual_minus_main': stats_pair['delta'],
                't_val': stats_pair['t_val'],
                'p_val': stats_pair['p_val'],
                'n_pairs': stats_pair['n_pairs'],
                'residual_mean_auc': a['mean_auc'],
                'main_mean_auc': b['mean_auc']
            })

    if records:
        for species in ('monkey', 'human'):
            idx = [i for i, r in enumerate(records) if r['species'] == species]
            if not idx:
                continue
            p_vals = np.array([records[i]['p_val'] for i in idx], dtype=float)
            q_vals = fdr_correction(p_vals)
            for i, q in zip(idx, q_vals):
                records[i]['q_val_fdr'] = q
    return records


# Plotting helpers ------------------------------------------------------------
def plot_combined_decoding(region, data):
    per_species = data['results']['per_species']
    paired = data['results']['paired']
    if not per_species.get('human') or not per_species.get('monkey'):
        print(f"Skipping {region.upper()} combined plot: need both species.")
        return

    features = [f for f in FEATURES if f in per_species['monkey'] and f in per_species['human']]
    if not features:
        print(f"No overlapping features for {region.upper()}.")
        return

    colors = get_feature_colors(region)
    labels = get_feature_labels()

    x_pos = np.arange(len(features))
    width = 0.35
    monkey_means = [per_species['monkey'][f]['mean_auc'] for f in features]
    monkey_sems = [per_species['monkey'][f]['sem_auc'] for f in features]
    human_means = [per_species['human'][f]['mean_auc'] for f in features]
    human_sems = [per_species['human'][f]['sem_auc'] for f in features]

    base_color = colors[features[0]]
    human_color = lighten_color(base_color, mix=HUMAN_LIGHTEN_MIX)
    ensure_species_legend(base_color, human_color)

    q_map = paired_q_map(paired, features)

    setup_style()
    fig = wide_figure()
    ax = fig.add_subplot(111)

    chance_color = config.plotting.get('chance_color', '#000000')
    ax.axhline(0.5, color=chance_color, linestyle='--', linewidth=2.5, alpha=0.8, zorder=10)

    for i, feat in enumerate(features):
        color = colors[feat]
        h_color = lighten_color(color, mix=HUMAN_LIGHTEN_MIX)
        ax.bar(x_pos[i] - width/2, monkey_means[i], width*0.9,
               color=color, edgecolor='black', linewidth=1.2, zorder=3)
        ax.bar(x_pos[i] + width/2, human_means[i], width*0.9,
               color=h_color, edgecolor='black', linewidth=1.2, zorder=2)
        ax.errorbar(x_pos[i] - width/2, monkey_means[i], yerr=monkey_sems[i], fmt='none',
                    ecolor='black', elinewidth=2.0, capsize=0, zorder=5)
        ax.errorbar(x_pos[i] + width/2, human_means[i], yerr=human_sems[i], fmt='none',
                    ecolor='black', elinewidth=2.0, capsize=0, zorder=5)

        q_val = q_map.get(feat, np.nan)
        entry = paired.get(feat)
        if entry and np.isfinite(q_val):
            y_max = max(monkey_means[i] + monkey_sems[i], human_means[i] + human_sems[i]) + 0.02
            if q_val < 0.001:
                marker = '***'
            elif q_val < 0.01:
                marker = '**'
            elif q_val < 0.05:
                marker = '*'
            else:
                marker = None
            if marker:
                ax.text(x_pos[i], y_max, marker, ha='center', fontsize=14)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[f] for f in features])
    ax.set_ylabel('AUC')
    ax.set_ylim(0.4, 1.0)
    style_axes(ax)
    ax.set_axisbelow(True)
    format_axes(ax, axis='y', precision=3)

    region_dir = FIG_DIR / region
    region_dir.mkdir(parents=True, exist_ok=True)
    out_path = region_dir / 'decode_species.pdf'
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"Saved: {out_path}")

    labels_map = get_feature_labels()
    stats_records = []
    for i, feat in enumerate(features):
        entry = paired.get(feat)
        stats_records.append({
            'feature': labels_map[feat],
            'monkey_mean_auc': monkey_means[i],
            'monkey_sem_auc': monkey_sems[i],
            'human_mean_auc': human_means[i],
            'human_sem_auc': human_sems[i],
            'delta_monkey_minus_human': entry['mean_diff'] if entry else np.nan,
            't_val': entry['t_val'] if entry else np.nan,
            'p_val': entry['p_val'] if entry else np.nan,
            'q_val_fdr': q_map.get(feat, np.nan)
        })
    stats_path = STATS_DIR_MAIN / f'decode_{region}.csv'
    save_stats_table(stats_records, stats_path)


def plot_null_decoding(region, species, per_species):
    data = per_species.get(species, {})
    features = [f for f in FEATURES if f in data]
    if not features:
        print(f"No data for supplementary null plot: {region.upper()} · {species}.")
        return

    colors = get_feature_colors(region)
    labels = get_feature_labels()

    x_pos = np.arange(len(features))
    width = 0.4

    obs_means = [data[f]['mean_auc'] for f in features]
    obs_sems = [data[f]['sem_auc'] for f in features]

    null_means, null_sems = [], []
    for f in features:
        null = data[f]['null_means']
        if isinstance(null, np.ndarray) and null.size:
            null_means.append(np.mean(null))
            null_sems.append(np.std(null, ddof=1) / np.sqrt(len(null)))
        else:
            null_means.append(0.5)
            null_sems.append(0.0)

    setup_style()
    fig = wide_figure()
    ax = fig.add_subplot(111)

    chance_color = config.plotting.get('chance_color', '#000000')
    ax.axhline(0.5, color=chance_color, linestyle='--', linewidth=2.5, alpha=0.8, zorder=10)

    obs_colors = [colors[f] for f in features]
    ax.bar(x_pos - width/2, obs_means, width*0.9, color=obs_colors,
           edgecolor='black', linewidth=1.2, zorder=3)
    ax.errorbar(x_pos - width/2, obs_means, yerr=obs_sems, fmt='none',
                ecolor='black', elinewidth=2.0, capsize=0, zorder=5)

    ax.bar(x_pos + width/2, null_means, width*0.9, color='#999999',
           edgecolor='black', linewidth=1.2, zorder=2)
    ax.errorbar(x_pos + width/2, null_means, yerr=null_sems, fmt='none',
                ecolor='black', elinewidth=2.0, capsize=0, zorder=5)

    p_vals = [data[f]['p_val'] for f in features]
    q_vals = fdr_correction(np.array(p_vals, dtype=float))
    for i, feat in enumerate(features):
        marker = None
        q = q_vals[i]
        if np.isfinite(q):
            if q < 0.001:
                marker = '***'
            elif q < 0.01:
                marker = '**'
            elif q < 0.05:
                marker = '*'
        if marker:
            y_max = obs_means[i] + obs_sems[i] + 0.02
            ax.text(x_pos[i] - width/2, y_max, marker, ha='center', fontsize=14)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[f] for f in features])
    ax.set_ylabel('AUC')
    ax.set_ylim(0.4, 1.0)
    style_axes(ax)
    ax.set_axisbelow(True)
    format_axes(ax, axis='y', precision=3)

    supp_dir = FIG_SUPP_DIR / region
    supp_dir.mkdir(parents=True, exist_ok=True)
    out_path = supp_dir / f'decode_null_{species}.pdf'
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"Saved: {out_path}")

    stats_data = {
        'feature': [labels[f] for f in features],
        'mean_auc': obs_means,
        'sem_auc': obs_sems,
        'null_mean': null_means,
        'p_val': p_vals,
        'q_val_fdr': q_vals.tolist()
    }
    stats_path = STATS_DIR_SUPP / f'decode_null_{region}_{species}.csv'
    save_stats_table(stats_data, stats_path)


def plot_residualized_combined(results):
    per_species = results['per_species']
    paired = results['paired']
    if not per_species.get('human') or not per_species.get('monkey'):
        print("Skipping residualized combined plot: need both species.")
        return

    features_h = per_species['human'].get('features', {})
    features_m = per_species['monkey'].get('features', {})
    features = [f for f in FEATURES if f in features_h and f in features_m]
    if not features:
        print("No overlapping features for residualized combined plot.")
        return

    colors = get_feature_colors('residualized')
    labels = get_feature_labels()

    x_pos = np.arange(len(features))
    width = 0.35

    monkey_means = [features_m[f]['mean_auc'] for f in features]
    monkey_sems = [features_m[f]['sem_auc'] for f in features]
    human_means = [features_h[f]['mean_auc'] for f in features]
    human_sems = [features_h[f]['sem_auc'] for f in features]

    base_color = colors[features[0]]
    human_color = lighten_color(base_color, mix=HUMAN_LIGHTEN_MIX)
    ensure_residual_legend(base_color, human_color)

    q_map = paired_q_map(paired, features)

    setup_style()
    fig = wide_figure()
    ax = fig.add_subplot(111)

    chance_color = config.plotting.get('chance_color', '#000000')
    ax.axhline(0.5, color=chance_color, linestyle='--', linewidth=2.5, alpha=0.8, zorder=10)

    for i, feat in enumerate(features):
        color = colors[feat]
        h_color = lighten_color(color, mix=HUMAN_LIGHTEN_MIX)
        ax.bar(x_pos[i] - width/2, monkey_means[i], width*0.9,
               color=color, edgecolor='black', linewidth=1.2, hatch='///', zorder=3)
        ax.bar(x_pos[i] + width/2, human_means[i], width*0.9,
               color=h_color, edgecolor='black', linewidth=1.2, hatch='///', zorder=2)
        ax.errorbar(x_pos[i] - width/2, monkey_means[i], yerr=monkey_sems[i], fmt='none',
                    ecolor='black', elinewidth=2.0, capsize=0, zorder=5)
        ax.errorbar(x_pos[i] + width/2, human_means[i], yerr=human_sems[i], fmt='none',
                    ecolor='black', elinewidth=2.0, capsize=0, zorder=5)

        q_val = q_map.get(feat, np.nan)
        entry = paired.get(feat)
        if entry and np.isfinite(q_val):
            y_max = max(monkey_means[i] + monkey_sems[i], human_means[i] + human_sems[i]) + 0.02
            if q_val < 0.001:
                marker = '***'
            elif q_val < 0.01:
                marker = '**'
            elif q_val < 0.05:
                marker = '*'
            else:
                marker = None
            if marker:
                ax.text(x_pos[i], y_max, marker, ha='center', fontsize=14)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[f] for f in features])
    ax.set_ylabel('AUC')
    ax.set_ylim(0.4, 1.0)
    style_axes(ax)
    ax.set_axisbelow(True)
    format_axes(ax, axis='y', precision=3)

    out_dir = FIG_DIR / 'residualized'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'decode_residualized_species.pdf'
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"Saved: {out_path}")

    labels_map = get_feature_labels()
    stats_records = []
    for i, feat in enumerate(features):
        entry = paired.get(feat)
        stats_records.append({
            'feature': labels_map[feat],
            'monkey_mean_auc': monkey_means[i],
            'monkey_sem_auc': monkey_sems[i],
            'human_mean_auc': human_means[i],
            'human_sem_auc': human_sems[i],
            'delta_monkey_minus_human': entry['mean_diff'] if entry else np.nan,
            't_val': entry['t_val'] if entry else np.nan,
            'p_val': entry['p_val'] if entry else np.nan,
            'q_val_fdr': q_map.get(feat, np.nan)
        })
    stats_path = STATS_DIR_MAIN / 'decode_residualized.csv'
    save_stats_table(stats_records, stats_path)


def plot_residualized_nulls(results):
    per_species = results['per_species']
    for species in ('monkey', 'human'):
        features_data = per_species.get(species, {}).get('features', {})
        if not features_data:
            continue
        payload = {species: features_data}
        plot_null_decoding('residualized', species, payload)


def plot_region_progression(species, region_data, contrast_map=None):
    regions = ['v1', 'v4', 'it']

    data = {r: region_data[r][species] for r in regions if r in region_data and species in region_data[r]}
    if not data:
        print(f"No progression data for {species}.")
        return

    features = [f for f in FEATURES if all(f in data[r] for r in data)]
    if not features:
        print(f"No overlapping features across regions for {species}.")
        return

    base_colors = get_feature_colors('it')
    if species == 'human':
        species_colors = {k: lighten_color(v, mix=HUMAN_LIGHTEN_MIX) for k, v in base_colors.items()}
    else:
        species_colors = base_colors
    labels = get_feature_labels()

    setup_style()
    fig = wide_figure()
    ax = fig.add_subplot(111)

    n_features = len(features)
    n_regions = len(data)
    x_pos = np.arange(n_features)
    width = 0.8 / n_regions
    stats_records = []
    feat_max = np.zeros(n_features, dtype=float)
    global_max = 0.0

    for i, region in enumerate(regions):
        if region not in data:
            continue
        means = [data[region][f]['mean_auc'] for f in features]
        sems = [data[region][f]['sem_auc'] for f in features]
        bar_colors = [species_colors[f] for f in features]

        offset = x_pos + i * width
        bars = ax.bar(offset, means, width * 0.9,
                      color=bar_colors, edgecolor='black', linewidth=1.2, zorder=3)
        ax.errorbar(offset, means, yerr=sems, fmt='none',
                    ecolor='black', elinewidth=2.0, capsize=0, zorder=5)
        region_label = region.upper()
        if species == 'human' and region_label == 'V4':
            region_label = 'hV4'
        for j, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
            y_pos = mean + sem + 0.012
            ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, region_label,
                    ha='center', va='bottom', rotation=60, fontsize=10, fontweight='bold')
            feat_max[j] = max(feat_max[j], y_pos)
            global_max = max(global_max, y_pos)

        for j, feat in enumerate(features):
            stats_records.append({
                'feature': labels[feat],
                'region': region.upper(),
                'mean_auc': means[j],
                'sem_auc': sems[j],
                'species': species
            })

    def sig_marker(q):
        if not np.isfinite(q):
            return None
        if q < 0.001:
            return '***'
        if q < 0.01:
            return '**'
        if q < 0.05:
            return '*'
        return None

    if contrast_map is not None:
        bump_small = 0.02
        for j, feat in enumerate(features):
            pos_v1 = x_pos[j]
            pos_v4 = x_pos[j] + width
            pos_it = x_pos[j] + 2 * width
            y_base = feat_max[j] + 0.02
            for label, pos_other, offset in (('IT-V4', pos_v4, 0.0),
                                             ('IT-V1', pos_v1, 0.025)):
                rec = contrast_map.get((feat, label))
                if not rec:
                    continue
                mark = sig_marker(rec.get('q_val_fdr', np.nan))
                if not mark:
                    continue
                y = y_base + offset
                ax.plot([pos_other, pos_other, pos_it, pos_it],
                        [y, y + bump_small, y + bump_small, y],
                        color='black', linewidth=1.2)
                ax.text((pos_other + pos_it) / 2.0, y + bump_small + 0.005, mark,
                        ha='center', va='bottom', fontsize=11)
                global_max = max(global_max, y + bump_small + 0.015)

    chance_color = config.plotting.get('chance_color', '#000000')
    ax.axhline(0.5, color=chance_color, linestyle='--', linewidth=2.5, alpha=0.8, zorder=10)

    ax.set_xticks(x_pos + width * (n_regions - 1) / 2)
    ax.set_xticklabels([labels[f] for f in features])
    ax.set_ylabel('AUC')
    y_top = max(1.0, global_max + 0.05) if global_max else 1.0
    ax.set_ylim(0.4, y_top)
    style_axes(ax)
    ax.set_axisbelow(True)
    format_axes(ax, axis='y', precision=3)

    out_dir = FIG_DIR / 'progression'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'decode_progression_{species}.pdf'
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"Saved: {out_path}")

    stats_path = STATS_DIR_MAIN / f'decode_progression_{species}.csv'
    save_stats_table(stats_records, stats_path)


# Orchestration ---------------------------------------------------------------
def viz_region_decoding(region='it'):
    data = load_region_main(region)
    if not data:
        return
    plot_combined_decoding(region, data)
    per_species = data['results']['per_species']
    for species in ('monkey', 'human'):
        if species in per_species:
            plot_null_decoding(region, species, per_species)


def viz_residualized():
    results = load_residualized_results()
    if not results:
        return
    plot_residualized_combined(results)
    plot_residualized_nulls(results)
    main_it = load_region_main('it')
    if main_it:
        recs = compute_residualization_contrasts(main_it['results']['per_species'], results['per_species'])
        if recs:
            labels = get_feature_labels()
            rows = []
            for r in recs:
                rows.append({
                    'species': r['species'],
                    'feature': labels.get(r['feature'], r['feature']),
                    'delta_residual_minus_main': r['delta_residual_minus_main'],
                    't_val': r['t_val'],
                    'p_val': r['p_val'],
                    'q_val_fdr': r.get('q_val_fdr', np.nan),
                    'n_pairs': r['n_pairs'],
                    'residual_mean_auc': r['residual_mean_auc'],
                    'main_mean_auc': r['main_mean_auc']
                })
            out_path = STATS_DIR_MAIN / 'decode_residualization_contrasts.csv'
            save_stats_table(rows, out_path)


def viz_progression():
    region_data = load_progression_data()
    if not region_data:
        return
    contrasts = compute_region_contrasts(region_data)
    labels = get_feature_labels()
    for species in ('monkey', 'human'):
        cmap = build_contrast_map(contrasts.get(species))
        plot_region_progression(species, region_data, contrast_map=cmap)
        recs = contrasts.get(species) or []
        if recs:
            rows = []
            for r in recs:
                rows.append({
                    'feature': labels.get(r['feature'], r['feature']),
                    'contrast': r['contrast'],
                    'delta_it_minus_other': r['delta'],
                    't_val': r['t_val'],
                    'p_val': r['p_val'],
                    'q_val_fdr': r.get('q_val_fdr', np.nan),
                    'n_pairs': r['n_pairs'],
                    'it_mean_auc': r['it_mean_auc'],
                    'other_mean_auc': r['other_mean_auc']
                })
            out_path = STATS_DIR_MAIN / f'decode_region_contrasts_{species}.csv'
            save_stats_table(rows, out_path)

    for other in ('v4', 'v1'):
        recs = compute_species_region_contrasts(region_data, other=other)
        if not recs:
            continue
        rows = [{
            'feature': labels.get(r['feature'], r['feature']),
            'contrast': r['contrast'],
            'delta_delta_auc': r['delta_delta_auc'],
            't_val': r['t_val'],
            'p_val': r['p_val'],
            'q_val_fdr': r.get('q_val_fdr', np.nan),
            'n_pairs': r['n_pairs'],
            'human_gain': r['human_gain'],
            'macaque_gain': r['macaque_gain']
        } for r in recs]
        save_stats_table(rows, STATS_DIR_MAIN / f'decode_species_region_contrasts_{other}.csv')


def main():
    print("Visualizing decoding results...\n")
    cleanup_outputs()

    for region in ('it', 'v1', 'v4'):
        viz_region_decoding(region)

    viz_residualized()
    viz_progression()

    print("\nDone.")


if __name__ == '__main__':
    main()
