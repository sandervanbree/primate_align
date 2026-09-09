#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analysis.nmf.gallery import GALLERY_SPECS
from analysis.nmf.utils import derive_categories, ensure_dir, load_srf_results, load_guard_table, pick_rank
from config.paths import config
from functions.plotting import format_axes, setup_style, style_axes, wide_roomy_figure
from model.feature.core.feature_io import load_beh
from model.feature.core.ridge_utils import get_ridge_params

FMT = config.plotting.get('savefig_format', 'pdf')
SUPP_FIG_DIR = ensure_dir(config.fig_dir / 'supp' / 'species_specific')
SUPP_RES_DIR = ensure_dir(config.results_dir / 'supp' / 'species_specific')
N_FOLDS = 10
N_PERM = 10_000
PERM_BATCH_SIZE = 250
SEED = 42
BASE_GREY = '#b3b3b3'
HUM_COLOR = config.plotting.get('human_color', '#7c5799')


def _find_bodypart_idx(names, key='body part'):
    key = key.lower()
    for i, name in enumerate(names):
        if key in name.lower():
            return i
    return None


def load_bodypart_scores(stims):
    X, names, _ = load_beh(stims)
    if X.size == 0 or not names:
        raise RuntimeError('behavioral embedding missing or empty')
    idx = _find_bodypart_idx(names)
    if idx is None:
        raise RuntimeError('body-part dimension not found in behavioral labels')
    return np.asarray(X[:, idx], float), names[idx]


def order_components(family, guard_df, n_components):
    key = 'human' if family == 'human' else 'monkey'
    spec = GALLERY_SPECS[key]
    if guard_df is None or guard_df.empty:
        return list(range(1, n_components + 1))
    df = guard_df.copy()
    df = df[spec['filter'](df)]
    if df.empty:
        return list(range(1, n_components + 1))
    df['score'] = df.apply(spec['score'], axis=1)
    df = df.sort_values('score', ascending=False)
    return df['component'].astype(int).tolist()


def ridge_body_to_components(body, W, comp_ids, kf):
    X = np.asarray(body, float).reshape(-1, 1)
    valid = [c for c in comp_ids if 1 <= c <= W.shape[1]]
    alphas, _ = get_ridge_params()

    r2_folds = np.zeros((len(valid), N_FOLDS))
    for fold_idx, (tr, te) in enumerate(kf.split(X)):
        xs = StandardScaler().fit(X[tr])
        Xtr = xs.transform(X[tr]).ravel()
        Xte = xs.transform(X[te]).ravel()
        for j, cid in enumerate(valid):
            ci = cid - 1
            ytr = W[tr, ci]
            yte = W[te, ci]
            alpha = RidgeCV(alphas=alphas, cv=3).fit(Xtr.reshape(-1, 1), ytr).alpha_
            r2 = Ridge(alpha=alpha).fit(Xtr.reshape(-1, 1), ytr).score(Xte.reshape(-1, 1), yte)
            r2_folds[j, fold_idx] = r2

    means = r2_folds.mean(axis=1)
    stds = r2_folds.std(axis=1)
    return valid, means, stds, r2_folds


def plot_family(family, comp_ids, means, stds, out_path, highlights=None, reference=None):
    setup_style()
    fig = wide_roomy_figure()
    ax = fig.add_subplot(111)
    x = np.arange(1, len(comp_ids) + 1)
    colors = [BASE_GREY] * len(comp_ids)
    for component in highlights or []:
        if component in comp_ids:
            colors[comp_ids.index(component)] = HUM_COLOR

    edge_lw = 0.8 if family == 'monkey' else 1.2
    ax.bar(x, means, color=colors, edgecolor='black', linewidth=edge_lw, width=0.82)
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black',
                elinewidth=edge_lw + 0.4, capsize=0, zorder=3)
    if reference is not None:
        ax.axhline(reference, color='black', linestyle='--', linewidth=1.0)

    style_axes(ax)
    format_axes(ax, precision=3)
    ax.set_xlabel(f"{family.title()} sNMF components (ranked)")
    ax.set_ylabel('Encoding (R²)')
    ticks = [1] + [t for t in range(10, len(comp_ids) + 1, 10)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_ylim(0, 0.1)
    ax.set_title('Body part dimension Encoding')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'[body-encoding] saved {out_path}')


def _subset_stats(indices, inverse, Y, n_categories):
    counts = np.bincount(inverse[indices], minlength=n_categories).astype(float)
    y_sum = np.zeros((n_categories, Y.shape[1]), float)
    np.add.at(y_sum, inverse[indices], Y[indices])
    return counts, y_sum, y_sum.sum(axis=0), np.sum(Y[indices] ** 2, axis=0)


def _centered_stats(x, counts, y_sum, y_total, y_sq_total, x_mean, y_mean):
    n = counts.sum()
    x_total = x @ counts
    x_sq = (x ** 2) @ counts
    sxx = x_sq - 2 * x_mean * x_total + n * x_mean ** 2
    syy = y_sq_total - 2 * y_mean * y_total + n * y_mean ** 2
    sxy = (
        x @ y_sum - x_mean[:, None] * y_total
        - x_total[:, None] * y_mean + n * x_mean[:, None] * y_mean
    )
    return sxx, syy, sxy


def _ridge_cv_permutations(x, Y, inverse, outer_splits, alphas):
    n_categories = x.shape[1]
    scores = []
    for outer_train, outer_test in outer_splits:
        train_stats = _subset_stats(outer_train, inverse, Y, n_categories)
        train_counts, _, train_y_total, _ = train_stats
        train_n = len(outer_train)
        outer_x_mean = (x @ train_counts) / train_n
        outer_x_var = (x ** 2) @ train_counts / train_n - outer_x_mean ** 2
        z = (x - outer_x_mean[:, None]) / np.sqrt(np.maximum(outer_x_var[:, None], 1e-24))

        inner_score = np.zeros((x.shape[0], Y.shape[1], len(alphas)), float)
        inner = KFold(n_splits=3, shuffle=False)
        for inner_train_pos, inner_val_pos in inner.split(outer_train):
            inner_train = outer_train[inner_train_pos]
            inner_val = outer_train[inner_val_pos]
            inner_train_stats = _subset_stats(inner_train, inverse, Y, n_categories)
            inner_val_stats = _subset_stats(inner_val, inverse, Y, n_categories)
            inner_counts, _, inner_y_total, _ = inner_train_stats
            inner_n = len(inner_train)
            x_mean = (z @ inner_counts) / inner_n
            y_mean = inner_y_total / inner_n
            sxx, _, sxy = _centered_stats(z, *inner_train_stats, x_mean, y_mean)
            sxx_val, syy_val, sxy_val = _centered_stats(
                z, *inner_val_stats, x_mean, y_mean
            )
            slope = sxy[:, :, None] / (sxx[:, None, None] + alphas[None, None, :])
            rss = (
                syy_val[None, :, None] - 2 * slope * sxy_val[:, :, None]
                + slope ** 2 * sxx_val[:, None, None]
            )
            y_val = Y[inner_val]
            denominator = np.sum((y_val - y_val.mean(axis=0)) ** 2, axis=0)
            inner_score += 1 - rss / denominator[None, :, None]

        alpha_idx = np.argmax(inner_score, axis=2)
        outer_y_mean = train_y_total / train_n
        sxx, _, sxy = _centered_stats(
            z, *train_stats, np.zeros(x.shape[0]), outer_y_mean
        )
        slopes = sxy[:, :, None] / (sxx[:, None, None] + alphas[None, None, :])
        slope = np.take_along_axis(slopes, alpha_idx[:, :, None], axis=2)[:, :, 0]
        test_stats = _subset_stats(outer_test, inverse, Y, n_categories)
        sxx_test, syy_test, sxy_test = _centered_stats(
            z, *test_stats, np.zeros(x.shape[0]), outer_y_mean
        )
        rss = syy_test[None, :] - 2 * slope * sxy_test + slope ** 2 * sxx_test[:, None]
        y_test = Y[outer_test]
        denominator = np.sum((y_test - y_test.mean(axis=0)) ** 2, axis=0)
        scores.append(1 - rss / denominator[None, :])
    return np.mean(scores, axis=0)


def permutation_stats(body, stims, results, fam_data):
    categories = np.asarray(derive_categories(stims))
    _, inverse = np.unique(categories, return_inverse=True)
    n_categories = inverse.max() + 1
    body_by_category = np.asarray([body[inverse == i][0] for i in range(n_categories)])
    if not np.allclose(body, body_by_category[inverse]):
        raise RuntimeError('body-part scores vary within concept')

    matrices = []
    labels = []
    observed = []
    for family, (comp_ids, means, _, _) in fam_data.items():
        rank = pick_rank(family, results[family])
        W = np.asarray(results[family]['components'][rank], float)
        matrices.append(W[:, np.asarray(comp_ids) - 1])
        labels.extend((family, component) for component in comp_ids)
        observed.extend(means)
    Y = np.column_stack(matrices)
    observed = np.asarray(observed)
    outer_splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(Y))
    alphas, _ = get_ridge_params()
    rng = np.random.default_rng(SEED)
    exceedances = np.zeros(Y.shape[1], dtype=int)
    for start in range(0, N_PERM, PERM_BATCH_SIZE):
        n_batch = min(PERM_BATCH_SIZE, N_PERM - start)
        permuted = np.stack([rng.permutation(body_by_category) for _ in range(n_batch)])
        null = _ridge_cv_permutations(permuted, Y, inverse, outer_splits, alphas)
        exceedances += np.sum(null >= observed[None, :], axis=0)

    p_values = (exceedances + 1) / (N_PERM + 1)
    q_values = multipletests(p_values, method='fdr_bh')[1]
    return {
        label: {'p_perm': p_values[i], 'q_fdr': q_values[i]}
        for i, label in enumerate(labels)
    }


def _assemble(left_path, right_path, out_path):
    left = Image.open(left_path).convert('RGB')
    right = Image.open(right_path).convert('RGB')
    fig = plt.figure(
        figsize=((left.width + right.width) / 300.0, max(left.height, right.height) / 300.0),
        dpi=300,
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[left.width, right.width], wspace=0.12)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_left.imshow(left)
    ax_right.imshow(right)
    ax_left.axis('off')
    ax_right.axis('off')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def build_family_data(report=True):
    results, stims = load_srf_results()
    body, _ = load_bodypart_scores(stims)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fam_data = {}

    for family in ('human', 'monkey'):
        if family not in results:
            print(f"[body-encoding] {family}: missing in SRF results, skipping.")
            continue
        rank = pick_rank(family, results[family])
        W = np.asarray(results[family]['components'][rank], float)
        try:
            guard_df = load_guard_table(family)
        except FileNotFoundError:
            guard_df = None
        comp_order = order_components(family, guard_df, W.shape[1])
        comp_ids, means, stds, folds = ridge_body_to_components(body, W, comp_order, kf)
        fam_data[family] = (comp_ids, means, stds, folds)

    stats = permutation_stats(body, stims, results, fam_data)
    if report and 'human' in fam_data and 'monkey' in fam_data:
        hum_ids, hum_means, _, _ = fam_data['human']
        for idx in np.argsort(hum_means)[-2:][::-1]:
            stat = stats[('human', hum_ids[idx])]
            print(
                f"[body-encoding] human rank {idx + 1} (comp {hum_ids[idx]}): "
                f"R²={hum_means[idx]:.3f}, p={stat['p_perm']:.4g}, q={stat['q_fdr']:.4g}"
            )
    return fam_data, stats


def save_supp_figure_and_stats(fig_path, stats_path, report=True):
    fam_data, permutation = build_family_data(report=report)
    rows = []
    hum_ids, hum_means, _, _ = fam_data.get('human', ([], [], [], []))
    highlighted = {
        hum_ids[idx] for idx in np.argsort(hum_means)[-2:]
    } if len(hum_ids) >= 2 else set(hum_ids)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        panel_paths = {}
        for family in ('human', 'monkey'):
            if family not in fam_data:
                continue
            comp_ids, means, stds, _ = fam_data[family]
            panel_path = tmp_dir / f'bodypart_{family}.png'
            highlights = [
                component for component in comp_ids
                if family == 'human' and component in highlighted
            ]
            other = 'monkey' if family == 'human' else 'human'
            reference = np.max(fam_data[other][1]) if other in fam_data else None
            plot_family(
                family, comp_ids, means, stds, panel_path,
                highlights=highlights, reference=reference,
            )
            panel_paths[family] = panel_path
            for rank_idx, (comp_id, mean, std) in enumerate(zip(comp_ids, means, stds), start=1):
                stat = permutation[(family, comp_id)]
                rows.append({
                    'family': family,
                    'rank': rank_idx,
                    'component': comp_id,
                    'r2_mean': mean,
                    'r2_std': std,
                    'p_perm': stat['p_perm'],
                    'q_fdr': stat['q_fdr'],
                })

        if 'human' in panel_paths and 'monkey' in panel_paths:
            _assemble(panel_paths['human'], panel_paths['monkey'], fig_path)
        elif panel_paths:
            only_path = next(iter(panel_paths.values()))
            fig_path = Path(fig_path)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            Image.open(only_path).save(fig_path)

    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(stats_path, sep='\t', index=False)
    print(f"[supp.bodypart_encoding] saved {fig_path}")
    print(f"[supp.bodypart_encoding] saved {stats_path}")


def main():
    fig_path = SUPP_FIG_DIR / f'bodypart_encoding.{FMT}'
    stats_path = SUPP_RES_DIR / 'bodypart_encoding_stats.tsv'
    save_supp_figure_and_stats(fig_path, stats_path, report=True)


if __name__ == '__main__':
    main()
