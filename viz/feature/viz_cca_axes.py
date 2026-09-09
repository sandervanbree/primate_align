from __future__ import annotations
import sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Wedge
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr, rankdata
import colorsys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.paths import config
from functions.plotting import (
    setup_style,
    medium_figure,
    square_figure,
    style_axes,
    format_axes,
    wide_roomy_figure,
)
from model.feature.core.ridge_utils import get_ridge_params

OUT_DIR = config.fig_dir / 'feature' / 'cca_axes'
FMT = config.plotting.get('savefig_format', 'pdf')
SET_METRIC = 'r2'  # 'r' or 'r2'
TOP_N = None  # Show all components (set to None for all)
SCATTER_DPI = 600
SCATTER_SIZE = 15  # Point size in scatter plots

def metric_label():
    return 'Encoding (Pearson r)' if SET_METRIC == 'r' else 'Encoding (R²)'

def desaturate_color(hex_color: str, factor: float = 0.3) -> tuple:
    """Desaturate hex color by reducing saturation."""
    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    s_new = s * factor
    rgb_new = colorsys.hsv_to_rgb(h, s_new, v)
    return rgb_new

def _add_img_border(img_arr, thick=8, color=(0,0,0)):
    """Add border around image array."""
    h, w, c = img_arr.shape
    new_h, new_w = h + 2*thick, w + 2*thick
    bordered = np.zeros((new_h, new_w, c), dtype=img_arr.dtype)
    bordered[:] = color
    bordered[thick:thick+h, thick:thick+w] = img_arr
    return bordered

def load_alexnet_fc6():
    """Load AlexNet FC6 features, rank transform, and compute PC1."""
    feat_path = config.data_dir / 'dnn' / 'features' / 'alexnet' / 'classifier.1' / 'features' / 'features.npy'
    if not feat_path.exists():
        raise FileNotFoundError(f'AlexNet FC6 not found: {feat_path}')
    X = np.load(feat_path)
    print(f'Loaded AlexNet FC6: {X.shape}')

    # Rank transform each feature (column-wise)
    X_ranked = np.zeros_like(X)
    for col in range(X.shape[1]):
        X_ranked[:, col] = rankdata(X[:, col])

    pca = PCA(n_components=1)
    pca.fit(X_ranked)
    pc1_scores = X_ranked @ pca.components_[0]
    print(f'PC1 scores (from rank-transformed): {pc1_scores.shape}')
    print(f'PC1 explained variance ratio: {pca.explained_variance_ratio_[0]:.4f}')
    return pc1_scores.reshape(-1, 1)

def load_cca_components():
    """Load CCA component scores for cross-species only."""
    state_fp = config.results_dir / 'cca_state.pkl'
    if not state_fp.exists():
        raise FileNotFoundError(f'cca_state.pkl not found: {state_fp}')

    with open(state_fp, 'rb') as f:
        s = pickle.load(f)

    cca_all = s['cca_all']
    views_all = sorted(cca_all['components'].keys())
    comps_all = np.mean([cca_all['components'][v] for v in views_all], axis=0)
    all_stims = s['all_stims']

    # Use all or top N components for cross-species
    comp_scores = {}
    if TOP_N is None:
        n_use = comps_all.shape[1]
    else:
        n_use = min(TOP_N, comps_all.shape[1])
    comp_scores['cross-species'] = comps_all[:, :n_use]
    print(f'cross-species: {comps_all.shape} -> using {n_use} components')

    return comp_scores, comps_all, all_stims

def load_features(all_stims, pc1_scores):
    """Load animacy, curvature, AlexNet PC1, and spiky-stubby aligned to CCA stimuli."""
    # Load THINGS properties for animacy (lives)
    props_fp = config.data_dir / 'things' / 'things_properties.csv'
    props_df = pd.read_csv(props_fp)

    # Load visual properties
    vis_fp = config.data_dir / 'features' / 'vis_props' / 'vis_props.tsv'
    vis_df = pd.read_csv(vis_fp, sep='\t', index_col='image')

    # Extract category from stimulus (e.g., 'aardvark_01b' -> 'aardvark')
    cats = [s.rsplit('_', 1)[0] for s in all_stims]

    # Map animacy (lives) - normalize to [0, 1]
    cat_to_lives = dict(zip(props_df['uniqueID'], props_df['lives']))
    lives = np.array([cat_to_lives.get(c, np.nan) for c in cats])
    lives_norm = (lives - np.nanmin(lives)) / (np.nanmax(lives) - np.nanmin(lives))

    # Map visual properties - already processed (in [0, 1])
    curv = np.array([vis_df.loc[s, 'curvature_proc'] if s in vis_df.index else np.nan for s in all_stims])
    spiky_stubby = np.array([vis_df.loc[s, 'spiky_stubby_proc'] if s in vis_df.index else np.nan for s in all_stims])

    # Use PC1 scores as proxy for spiky_stubby, normalize to [0, 1]
    pc1 = pc1_scores.ravel()
    pc1_norm = (pc1 - np.min(pc1)) / (np.max(pc1) - np.min(pc1))

    print(f'Loaded features for {len(all_stims)} stimuli:')
    print(f'  animacy (lives): {np.sum(~np.isnan(lives_norm))} valid, range [{np.nanmin(lives_norm):.3f}, {np.nanmax(lives_norm):.3f}]')
    print(f'  curvature: {np.sum(~np.isnan(curv))} valid, range [{np.nanmin(curv):.3f}, {np.nanmax(curv):.3f}]')
    print(f'  AlexNet PC1: {len(pc1_norm)} valid, range [{np.min(pc1_norm):.3f}, {np.max(pc1_norm):.3f}]')
    print(f'  spiky-stubby: {np.sum(~np.isnan(spiky_stubby))} valid, range [{np.nanmin(spiky_stubby):.3f}, {np.nanmax(spiky_stubby):.3f}]')

    return {
        'animacy': lives_norm,
        'curvature': curv,
        'alexnet_spiky': pc1_norm,
        'spiky_stubby': spiky_stubby,
    }

def plot_scatter(comps, feat_vals, feat_name, cmap_name, c1=0, c2=1, highlight_top_n=None, highlight_idxs=None, out_path=None):
    """Plot scatter of two CCA components colored by feature."""
    coords = comps[:, [c1, c2]]

    # Filter out NaN values
    valid = ~np.isnan(feat_vals)
    coords_v = coords[valid]
    vals_v = feat_vals[valid]

    setup_style()
    fig = square_figure()
    ax = fig.add_subplot(111)

    # Muted colormap
    base_cmap = plt.colormaps[cmap_name]
    muted_cmap = base_cmap(np.linspace(0, 1, 256))
    muted_cmap[:, :3] = muted_cmap[:, :3] * 0.85 + 0.15  # Desaturate and lighten
    muted_cmap = matplotlib.colors.ListedColormap(muted_cmap)

    sc = ax.scatter(coords_v[:, 0], coords_v[:, 1], c=vals_v, cmap=muted_cmap,
                   vmin=0, vmax=1, s=25, alpha=0.9, edgecolors='none')

    # Highlight top N by Component 1 value if requested
    if highlight_top_n is not None:
        comp1_vals = comps[:, c1]
        top_idxs = np.argsort(comp1_vals)[::-1][:highlight_top_n]
        top_coords = coords[top_idxs]
        ax.scatter(top_coords[:, 0], top_coords[:, 1], s=50, color='red',
                  edgecolors='darkred', linewidth=1.5, zorder=5)

    # Highlight specific indices if provided
    if highlight_idxs is not None:
        highlight_coords = coords[highlight_idxs]
        ax.scatter(highlight_coords[:, 0], highlight_coords[:, 1], s=50, color='red',
                  edgecolors='darkred', linewidth=1.5, zorder=5)

    # Set axis limits with padding to prevent bleeding
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    pad = 0.05 * max(x_range, y_range)
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.axis('off')

    # Colorbar - 50% width, positioned lower
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.05])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0', '1'])

    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

    out = Path(out_path) if out_path is not None else OUT_DIR / f'scatter_{feat_name}.{FMT}'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'Saved scatter: {out}')

def compute_ridge_cv(X, comp_scores, k=10):
    """Compute cross-validated R² using ridge regression (like ridge_component.py)."""
    alphas, _ = get_ridge_params()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    results = {}
    for fam, Y in comp_scores.items():
        n_comp = Y.shape[1]
        r2_folds = np.zeros((k, n_comp))

        for fold_idx, (tr, te) in enumerate(kf.split(X)):
            # Standardize
            xs = StandardScaler().fit(X[tr])
            Xtr = xs.transform(X[tr])
            Xte = xs.transform(X[te])

            # Predict each component
            for comp_idx in range(n_comp):
                ytr = Y[tr, comp_idx]
                yte = Y[te, comp_idx]

                # Inner CV to select alpha
                alpha = RidgeCV(alphas=alphas, cv=3).fit(Xtr, ytr).alpha_
                # Fit and score (returns R²)
                r2 = Ridge(alpha=alpha).fit(Xtr, ytr).score(Xte, yte)
                r2_folds[fold_idx, comp_idx] = r2

        # Compute mean and std across folds
        r2_mean = r2_folds.mean(axis=0)
        r2_std = r2_folds.std(axis=0)
        results[fam] = {'mean': r2_mean, 'std': r2_std, 'folds': r2_folds}
        print(f'{fam}: computed CV R² for {n_comp} components')

    return results

def compute_feature_correlations(features, comps):
    """Compute Pearson correlations between features and CCA components 1 and 2."""
    results = []
    for feat_name in ['animacy', 'curvature']:
        feat_vals = features[feat_name]
        valid = ~np.isnan(feat_vals)

        for comp_idx, comp_name in [(0, 'Component 1'), (1, 'Component 2')]:
            comp_vals = comps[:, comp_idx]
            r, p = pearsonr(feat_vals[valid], comp_vals[valid])
            results.append({
                'feature': feat_name,
                'component': comp_name,
                'r': r,
                'p_value': p,
                'n': valid.sum()
            })

    df = pd.DataFrame(results)
    out = OUT_DIR / 'feature_component_correlations.csv'
    df.to_csv(out, index=False)
    print(f'Saved feature correlations: {out}')

    # Print formatted results for manuscript
    print('\n=== Feature-Component Correlations (for manuscript) ===')
    for _, row in df.iterrows():
        print(f"{row['feature']:>10} vs {row['component']}: r = {row['r']:7.4f}, p = {row['p_value']:.4e}, n = {row['n']}")

    return df

def compute_feature_ridge_cv(features_dict, comps, k=10):
    """Evaluate features (animacy, curvature, spiky_stubby) via ridge regression on CCA components."""
    from scipy.stats import ttest_rel
    from statsmodels.stats.multitest import multipletests

    alphas, _ = get_ridge_params()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    results = {}
    for feat_name, feat_vals in features_dict.items():
        # Skip NaN values
        valid = ~np.isnan(feat_vals)
        feat_v = feat_vals[valid]
        comps_v = comps[valid]

        n_comp = comps_v.shape[1]
        r2_folds = np.zeros((k, n_comp))

        for fold_idx, (tr, te) in enumerate(kf.split(feat_v)):
            # Standardize features
            fs = StandardScaler().fit(feat_v[tr].reshape(-1, 1))
            ftr = fs.transform(feat_v[tr].reshape(-1, 1)).ravel()
            fte = fs.transform(feat_v[te].reshape(-1, 1)).ravel()

            # Predict each component from this feature
            for comp_idx in range(n_comp):
                ytr = comps_v[tr, comp_idx]
                yte = comps_v[te, comp_idx]

                # Inner CV to select alpha
                alpha = RidgeCV(alphas=alphas, cv=3).fit(ftr.reshape(-1, 1), ytr).alpha_
                # Fit and score
                r2 = Ridge(alpha=alpha).fit(ftr.reshape(-1, 1), ytr).score(fte.reshape(-1, 1), yte)
                r2_folds[fold_idx, comp_idx] = r2

        r2_mean = r2_folds.mean(axis=0)
        r2_std = r2_folds.std(axis=0)
        results[feat_name] = {'mean': r2_mean, 'std': r2_std, 'folds': r2_folds}
        print(f'{feat_name}: computed CV R² predicting {n_comp} components')

    return results

def save_feature_stats(results):
    """Save ridge regression statistics for features predicting CCA components."""
    rows = []
    for feat_name, data in results.items():
        r2_mean = data['mean']
        r2_std = data['std']
        r2_folds = data['folds']

        for comp_idx in range(len(r2_mean)):
            row = {
                'feature': feat_name,
                'component': comp_idx + 1,
                'r2_mean': r2_mean[comp_idx],
                'r2_std': r2_std[comp_idx],
                'r2_rank': int(np.argsort(-r2_mean)[comp_idx]) + 1,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    out = OUT_DIR / f'feature_ridge_stats.csv'
    df.to_csv(out, index=False)
    print(f'Saved feature ridge statistics: {out}')

    # Print summary
    print(f'\n=== Feature Ridge Regression Summary ===')
    for feat_name in results.keys():
        feat_data = df[df['feature'] == feat_name]
        mean_r2 = feat_data['r2_mean'].mean()
        print(f'{feat_name}: mean R² across components = {mean_r2:.4f}')

def plot_top_comp1_grid(comps, stims, cats, n=12, n_cols=3, grid_idxs=None, suffix=''):
    """Plot grid showing top N images by Component 1 value or specific indices."""
    img_dir = config.data_dir / 'things' / 'images'

    # Get indices - either provided or top N by Component 1
    if grid_idxs is not None:
        top_idxs = grid_idxs[:n]
    else:
        comp1_vals = comps[:, 0]
        top_idxs = np.argsort(comp1_vals)[::-1][:n]

    n_rows = int(np.ceil(n / n_cols))
    img_px = 200
    border_thick = 8

    # Build tiles
    tiles = []
    for i in top_idxs:
        path = img_dir / cats[i] / f"{stims[i]}.jpg"
        try:
            im = Image.open(path).convert("RGB").resize((img_px, img_px), Image.BILINEAR)
            im_arr = np.array(im)
            im_arr = _add_img_border(im_arr, thick=border_thick, color=(64, 64, 64))
            tiles.append(im_arr)
        except Exception:
            blank = np.full((img_px + 2*border_thick, img_px + 2*border_thick, 3), 128, np.uint8)
            tiles.append(blank)

    # Pad with blanks
    blank = np.full((img_px + 2*border_thick, img_px + 2*border_thick, 3), 128, np.uint8)
    tiles += [blank] * (n_rows * n_cols - len(tiles))

    # Build grid
    grid = np.vstack([np.hstack(tiles[r * n_cols:(r + 1) * n_cols]) for r in range(n_rows)])

    # Plot
    setup_style()
    fig, ax = plt.subplots(figsize=(n_cols * 2, n_rows * 2), dpi=100)
    ax.imshow(grid)
    ax.axis('off')

    fname = f'top_comp1_grid{suffix}.{FMT}' if suffix else f'top_comp1_grid.{FMT}'
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'Saved top Component 1 grid: {out}')

def plot_nonliving_negative_comp1_grid(comps, stims, cats, lives_vals, n=40, n_cols=8, out_path=None):
    """Plot grid of most negative Component 1 images with below-median livingness."""
    img_dir = config.data_dir / 'things' / 'images'

    comp1_vals = comps[:, 0]
    lives_median = np.nanmedian(lives_vals)

    # Filter to below-median livingness (non-living)
    nonliving_mask = lives_vals < lives_median
    nonliving_idxs = np.where(nonliving_mask)[0]

    # Sort by Component 1 value (ascending = most negative first)
    comp1_nonliving = comp1_vals[nonliving_idxs]
    sorted_order = np.argsort(comp1_nonliving)
    top_idxs = nonliving_idxs[sorted_order[:n]]

    print(f'Non-living items (below median lives={lives_median:.3f}): {len(nonliving_idxs)}')
    print(f'Selected {n} most negative Component 1 values: [{comp1_vals[top_idxs].min():.3f}, {comp1_vals[top_idxs].max():.3f}]')

    n_rows = int(np.ceil(n / n_cols))
    img_px = 150
    border_thick = 6

    # Build tiles
    tiles = []
    for i in top_idxs:
        path = img_dir / cats[i] / f"{stims[i]}.jpg"
        try:
            im = Image.open(path).convert("RGB").resize((img_px, img_px), Image.BILINEAR)
            im_arr = np.array(im)
            im_arr = _add_img_border(im_arr, thick=border_thick, color=(64, 64, 64))
            tiles.append(im_arr)
        except Exception:
            blank = np.full((img_px + 2*border_thick, img_px + 2*border_thick, 3), 128, np.uint8)
            tiles.append(blank)

    # Pad with blanks
    blank = np.full((img_px + 2*border_thick, img_px + 2*border_thick, 3), 128, np.uint8)
    tiles += [blank] * (n_rows * n_cols - len(tiles))

    # Build grid
    grid = np.vstack([np.hstack(tiles[r * n_cols:(r + 1) * n_cols]) for r in range(n_rows)])

    # Plot
    setup_style()
    fig, ax = plt.subplots(figsize=(n_cols * 1.5, n_rows * 1.5), dpi=100)
    ax.imshow(grid)
    ax.axis('off')

    out = Path(out_path) if out_path is not None else OUT_DIR / f'nonliving_negative_comp1_grid.{FMT}'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'Saved non-living negative Component 1 grid: {out}')

def plot_variances_with_stats(results, out_path=None, stats_path=None):
    """Bar plot for cross-species components."""
    base_col = config.plotting.get('shared_color', '#81B7B3')

    # Only cross-species
    r2_mean = results['cross-species']['mean']
    r2_std = results['cross-species']['std']
    r2_folds = results['cross-species']['folds']
    n_comp = len(r2_mean)

    positions, heights, errors, colors = [], [], [], []
    width = 1.0  # Full width bars so they connect
    x_positions = np.arange(1, n_comp + 1)  # 1-indexed component numbers

    # More muted colors for non-highlighted components
    dark_col = desaturate_color(base_col, factor=0.25)  # More desaturation
    h, s, v = colorsys.rgb_to_hsv(*dark_col)
    dark_col = colorsys.hsv_to_rgb(h, s, v * 0.75)  # More darkening

    for comp_idx in range(n_comp):
        positions.append(x_positions[comp_idx])
        heights.append(r2_mean[comp_idx])
        errors.append(r2_std[comp_idx])

        # Highlight component 2
        if comp_idx == 1:
            rgb = tuple(int(base_col.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            colors.append(rgb)
        else:
            colors.append(dark_col)

    setup_style()
    fig = wide_roomy_figure()
    ax = fig.add_subplot(111)

    # Plot bars with error bars (standard codebase style)
    ax.bar(positions, heights, width=width*0.9, color=colors, edgecolor='black', linewidth=1.2, align='center')
    ax.errorbar(positions, heights, yerr=errors, fmt='none', ecolor='black',
               elinewidth=2.0, capsize=0, zorder=3)

    # Add significance star above component 2
    comp2_idx = 1
    comp2_height = heights[comp2_idx] + errors[comp2_idx]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    star_offset = y_range * 0.04
    ax.text(positions[comp2_idx], comp2_height + star_offset, '*',
           ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Styling
    style_axes(ax)
    format_axes(ax, precision=3)
    ax.set_xlabel('Component')
    ax.set_ylabel(metric_label())

    # Set x-axis ticks based on number of components
    if n_comp <= 20:
        tick_marks = [1] + list(range(5, n_comp + 1, 5))
    else:
        tick_marks = [1] + list(range(10, n_comp + 1, 10))
    ax.set_xticks(sorted(set(tick_marks)))

    y_vals = [h + e for h, e in zip(heights, errors) if np.isfinite(h)]
    if y_vals:
        y_max = max(y_vals)
        ax.set_ylim(0, y_max * 1.15)  # Extra room for star
    ax.margins(y=0)

    fig.tight_layout(pad=0.4)
    suffix = f'_all{n_comp}' if TOP_N is None else f'_top{TOP_N}'
    out = Path(out_path) if out_path is not None else OUT_DIR / f'alexnet_fc6_pc1{suffix}.{FMT}'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'\nSaved figure: {out}')

    # Statistical test: is component 2 significantly higher than all others?
    comp2_folds = r2_folds[:, 1]  # Component 2 is index 1 (0-indexed)

    # Paired t-tests against all other components
    from scipy.stats import ttest_rel
    from statsmodels.stats.multitest import multipletests

    sig_tests = []
    p_values = []
    for comp_idx in range(n_comp):
        if comp_idx == 1:  # Skip self-comparison
            continue
        other_folds = r2_folds[:, comp_idx]
        t_stat, p_val = ttest_rel(comp2_folds, other_folds)
        sig_tests.append({
            'component': comp_idx + 1,
            't_statistic': t_stat,
            'p_value': p_val
        })
        p_values.append(p_val)

    # FDR correction using Benjamini-Hochberg
    p_values = np.array(p_values)
    rejected, q_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    # Add FDR results to sig_tests
    for i, test in enumerate(sig_tests):
        test['q_value'] = q_values[i]
        test['fdr_significant'] = rejected[i]
        test['significant_05'] = test['p_value'] < 0.05
        test['significant_01'] = test['p_value'] < 0.01

    # Save statistics to CSV
    import pandas as pd

    # Build compact table for supplementary materials
    stats_rows = []
    for comp_idx in range(n_comp):
        row = {
            'comp': comp_idx + 1,
            'r2': round(r2_mean[comp_idx], 3),
            'se': round(r2_std[comp_idx], 3),
            'rank': int(np.argsort(-r2_mean)[comp_idx]) + 1
        }

        if comp_idx == 1:  # Component 2 - self-comparison
            row['t_vs_c2'] = '-'
            row['q_fdr'] = '-'
            row['sig'] = '-'
            row['delta_r2'] = 0.0
        else:
            test = next((t for t in sig_tests if t['component'] == comp_idx + 1), None)
            if test:
                row['t_vs_c2'] = round(test['t_statistic'], 2)
                row['q_fdr'] = f"{test['q_value']:.2e}" if test['q_value'] < 0.001 else round(test['q_value'], 3)
                row['sig'] = '*' if (test['fdr_significant'] and test['t_statistic'] > 0) else ''
            else:
                row['t_vs_c2'] = '-'
                row['q_fdr'] = '-'
                row['sig'] = ''
            row['delta_r2'] = round(r2_mean[1] - r2_mean[comp_idx], 3)

        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)

    suffix = f'_all{n_comp}' if TOP_N is None else f'_top{TOP_N}'
    stats_out = Path(stats_path) if stats_path is not None else OUT_DIR / f'alexnet_fc6_pc1{suffix}_stats.csv'
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(stats_out, index=False)
    print(f'Saved statistics: {stats_out}')

    # Print top 10
    print(f'\n=== Top 10 components by {metric_label()} ===')
    top_comps = stats_df.sort_values('r2', ascending=False).head(10)
    for i, row in enumerate(top_comps.itertuples(), 1):
        marker = ' *** TARGET ***' if row.comp == 2 else ''
        print(f'  {i}. Component {row.comp}: R² = {row.r2:.3f} ± {row.se:.3f}{marker}')

    # Print statistical summary
    print('\n=== Component 2 Statistical Significance ===')
    n_sig_uncorr = sum(1 for test in sig_tests if test['significant_05'] and test['t_statistic'] > 0)
    n_sig_fdr = sum(1 for test in sig_tests if test['fdr_significant'] and test['t_statistic'] > 0)
    print(f'Component 2 is significantly higher than {n_sig_uncorr}/{len(sig_tests)} other components (uncorrected p < 0.05)')
    print(f'Component 2 is significantly higher than {n_sig_fdr}/{len(sig_tests)} other components (FDR-corrected q < 0.05)')

    # Show worst case (smallest t-statistic where comp 2 is still higher)
    higher_tests = [t for t in sig_tests if t['t_statistic'] > 0]
    if higher_tests:
        weakest = min(higher_tests, key=lambda x: x['t_statistic'])
        print(f'Weakest comparison: Component 2 vs Component {weakest["component"]}')
        print(f'  t = {weakest["t_statistic"]:.3f}, p = {weakest["p_value"]:.4e}, q = {weakest["q_value"]:.4e}, FDR sig = {weakest["fdr_significant"]}')
    else:
        weakest = None

    # Manuscript summary line
    if weakest is not None:
        weakest_comp = weakest['component']
        weakest_t = weakest['t_statistic']
        weakest_q = weakest['q_value']
    else:
        weakest_comp = 'N/A'
        weakest_t = float('nan')
        weakest_q = float('nan')

    print(f'\n=== For Manuscript (bare stats) ===')
    print(f'Component 2 R² = {r2_mean[1]:.3f} ± {r2_std[1]:.3f}')
    print(f'Min paired t vs Component {weakest_comp}: {weakest_t:.2f}')
    print(f'Corresponding FDR q = {weakest_q:.2e}; all other comparisons also q < 0.05')

def main():
    n_str = 'all' if TOP_N is None else f'top {TOP_N}'
    print(f'\n=== CCA axes: AlexNet FC6 PC1 with {n_str} cross-species CCA components (CV ridge) ===')
    X = load_alexnet_fc6()
    comp_scores, comps_all, all_stims = load_cca_components()
    results = compute_ridge_cv(X, comp_scores)
    plot_variances_with_stats(results)

    print('\n=== Generating scatter plots ===')
    features = load_features(all_stims, X)

    # Compute correlations between features and components 1 and 2
    compute_feature_correlations(features, comps_all)

    # Get aberrant indices (top 25 most negative non-living Component 1)
    comp1_vals = comps_all[:, 0]
    lives_median = np.nanmedian(features['animacy'])
    nonliving_mask = features['animacy'] <= lives_median
    nonliving_idxs = np.where(nonliving_mask)[0]
    comp1_nonliving = comp1_vals[nonliving_idxs]
    sorted_order = np.argsort(comp1_nonliving)
    aberrant_idxs = nonliving_idxs[sorted_order[:25]]
    print(f'Aberrant: {len(aberrant_idxs)} most negative non-living items (below median={lives_median:.3f}), comp1 range [{comp1_vals[aberrant_idxs].min():.4f}, {comp1_vals[aberrant_idxs].max():.4f}]')

    # Scatter plots
    plot_scatter(comps_all, features['animacy'], 'animacy', 'PRGn', c1=0, c2=1)
    plot_scatter(comps_all, features['animacy'], 'animacy_aberrant', 'PRGn', c1=0, c2=1, highlight_idxs=aberrant_idxs)
    plot_scatter(comps_all, features['curvature'], 'curvature', 'copper', c1=0, c2=1, highlight_top_n=12)
    plot_scatter(comps_all, features['alexnet_spiky'], 'alexnet_spiky', 'cividis', c1=0, c2=1)
    plot_scatter(comps_all, 1.0 - features['spiky_stubby'], 'spiky_stubby', 'cividis', c1=0, c2=1)

    print('\n=== Ridge regression: features predicting CCA components ===')
    feat_results = compute_feature_ridge_cv(features, comps_all, k=10)
    save_feature_stats(feat_results)

    print('\n=== Generating Component 1 image grids ===')
    all_cats = [s.rsplit('_', 1)[0] for s in all_stims]
    plot_top_comp1_grid(comps_all, all_stims, all_cats, n=12, n_cols=3)
    plot_top_comp1_grid(comps_all, all_stims, all_cats, n=25, n_cols=5, grid_idxs=aberrant_idxs, suffix='_aberrant')

    print('\n=== Generating non-living negative Component 1 grid ===')
    plot_nonliving_negative_comp1_grid(comps_all, all_stims, all_cats, features['animacy'], n=40, n_cols=8)

    print('Done.\n')

if __name__ == '__main__':
    main()
