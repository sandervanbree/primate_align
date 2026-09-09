from __future__ import annotations
import os, sys, pickle, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm
from matplotlib import patches
from matplotlib.lines import Line2D
try:
    from scipy import stats
except Exception:
    stats = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.paths import config
from functions.plotting import (
    setup_style,
    medium_figure,
    wide_figure,
    large_figure,
    square_figure,
    narrow_figure,
    style_axes,
    format_axes,
    panel_figure,
)

BASE_DIR = Path(config.root_dir)
OUT_DIR = BASE_DIR / 'figures' / 'feature'
FMT = config.plotting.get('savefig_format', 'pdf')
GROUPS = ['macaque', 'human']
LEGEND_DIR = OUT_DIR / 'legend'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LEGEND_DIR.mkdir(parents=True, exist_ok=True)

_ARCH_CONF = config.plotting.get('model_arch', {})
_ARCH_BLOCKS = []
_CANON_TO_ARCH = {}
_CANON_KEYS = []
for arch_key, ent in _ARCH_CONF.items():
    canon_models = [str(m).lower() for m in ent.get('models', [])]
    if not canon_models:
        continue
    block = {
        'arch': str(arch_key),
        'canon_models': canon_models,
        'cmap': str(ent.get('cmap', config.plotting.get('model_cmap', 'magma'))),
        'tmin': float(ent.get('tmin', config.plotting.get('model_cmap_tmin', 0.25))),
        'tmax': float(ent.get('tmax', config.plotting.get('model_cmap_tmax', 0.85))),
    }
    _ARCH_BLOCKS.append(block)
    for canon in canon_models:
        _CANON_TO_ARCH[canon] = block['arch']
        _CANON_KEYS.append(canon)

ARCH_SPACING = float(config.plotting.get('model_arch_spacing', 0.0))
_MODEL_HATCH = {str(k).lower(): str(v) for k, v in config.plotting.get('model_hatch', {}).items()}

SET_METRIC = 'r2'

def _canon_model_name(name: str) -> str:
    n = str(name).lower()
    for canon in _CANON_KEYS:
        if n == canon or canon in n:
            return canon
    raise KeyError(f"Model '{name}' not registered in config.plotting.model_arch")

def _ordered_models(models):
    unique = list(dict.fromkeys(models))
    out = []
    for block in _ARCH_BLOCKS:
        for canon in block['canon_models']:
            for m in unique:
                if m in out:
                    continue
                try:
                    if _canon_model_name(m) == canon:
                        out.append(m)
                except KeyError:
                    continue
    remaining = [m for m in unique if m not in out]
    if remaining:
        raise KeyError(f"Missing model_arch config for: {', '.join(remaining)}")
    return out

def _model_arch(name: str) -> str:
    c = _canon_model_name(name)
    return _CANON_TO_ARCH.get(c, 'other')

def _soft_color(col, mix=0.2):
    arr = np.array(col[:3], float)
    muted = arr * (1.0 - mix) + mix
    alpha = col[3] if len(col) == 4 else 1.0
    return (muted[0], muted[1], muted[2], alpha)

def model_colors(models):
    models = _ordered_models(models)
    if not models:
        return {}
    out = {}
    for block in _ARCH_BLOCKS:
        block_models = [m for m in models if _canon_model_name(m) in block['canon_models']]
        if not block_models:
            continue
        cmap_name = block['cmap']
        tmin = block['tmin']
        tmax = block['tmax']
        try:
            cmap = plt.colormaps.get_cmap(cmap_name)
        except Exception:
            cmap = cm.get_cmap(cmap_name)
        ts = np.linspace(tmin, tmax, max(2, len(block_models)))
        for i, m in enumerate(block_models):
            out[m] = _soft_color(cmap(float(ts[i])))
    return out

def _model_hatch(name: str) -> str:
    try:
        canon = _canon_model_name(name)
    except KeyError:
        return ''
    return _MODEL_HATCH.get(canon, '')

def other_colors():
    bone_vals = [0.25, 0.5, 0.75]
    return [cm.bone(v) for v in bone_vals]

def metric_stats(entry: dict) -> tuple[float, float]:
    if SET_METRIC == 'r':
        folds = entry.get('r_w_folds') or entry.get('r_folds')
        if folds:
            v = np.array(folds, float)
            return float(v.mean()), float(v.std())
        return (
            float(entry.get('r_w_mean', entry.get('r_mean', np.nan))),
            float(entry.get('r_w_std', entry.get('r_std', np.nan))),
        )
    else:  # 'r2'
        folds = entry.get('r2_w_folds') or entry.get('r2_folds')
        if folds:
            v = np.array(folds, float)
            return float(v.mean()), float(v.std())
        return float(entry.get('r2_w_mean', entry.get('r2_mean', np.nan))), float(entry.get('r2_w_std', entry.get('r2_std', np.nan)))

def metric_label():
    return 'Encoding (Pearson r)' if SET_METRIC == 'r' else 'Encoding (R²)'

def _fold_series(row, prefixes):
    for prefix in prefixes:
        cols = []
        for key in row.index:
            key_s = str(key)
            if not key_s.startswith(prefix):
                continue
            suffix = key_s[len(prefix):]
            if suffix.isdigit():
                cols.append((int(suffix), key))
        if cols:
            vals = [float(row[key]) for _, key in sorted(cols) if pd.notna(row[key])]
            if vals:
                return vals
    return []

def _r_stats(entry: dict) -> tuple[float, float]:
    # Legacy wrapper for compatibility
    return metric_stats(entry)

def _score_stats(entry: dict) -> tuple[float, float]:
    folds = entry.get('r2_w_folds') or entry.get('r2_folds')
    if folds:
        r2f = np.array(folds, float)
        return float(r2f.mean()), float(r2f.std())
    mu = float(entry.get('r2_w_mean', entry.get('r2_mean', np.nan)))
    sd = float(entry.get('r2_w_std', entry.get('r2_std', np.nan)))
    return mu, sd

def load_csv_results(csv_path: Path):
    df = pd.read_csv(csv_path)
    per_layer = {}
    best = {'macaque': {}, 'human': {}}
    for _, row in df.iterrows():
        family = row['family']
        model = row['model']
        layer = row['layer']
        dval = row.get('depth', 0.0)
        try:
            depth = float(dval)
            if not np.isfinite(depth):
                depth = 0.0
        except Exception:
            depth = 0.0
        if model not in per_layer:
            per_layer[model] = {}
        if layer not in per_layer[model]:
            per_layer[model][layer] = {'depth': depth}

        # Weighted Pearson-r folds preferred (computed as mean component-wise r)
        r_folds = _fold_series(row, ['r_w_f', 'r_f'])

        entry = {
            'r_mean': float(row.get('r_w_mean', row.get('r_mean', np.nan))),
            'r_std': float(row.get('r_w_std', row.get('r_std', np.nan))),
            'r_folds': r_folds if r_folds else None,
            'r2_mean': float(row.get('r2_w_mean', row.get('r2_mean', np.nan))),
            'r2_std': float(row.get('r2_w_std', row.get('r2_std', np.nan))),
            'n_folds': int(row['n_folds']),
            'n_features': int(row['n_features'])
        }
        if family == 'cross-species':
            per_layer[model][layer].setdefault('human', entry)
            per_layer[model][layer].setdefault('macaque', entry)
        elif family == 'human':
            per_layer[model][layer]['human'] = entry
        elif family == 'monkey':
            per_layer[model][layer]['macaque'] = entry
        for species in ['human', 'macaque']:
            if species in per_layer[model][layer]:
                metric_val = metric_stats(per_layer[model][layer][species])[0]
                if model not in best[species]:
                    best[species][model] = {'layer': layer, 'metric': metric_val}
                elif metric_val > best[species][model]['metric']:
                    best[species][model] = {'layer': layer, 'metric': metric_val}
    return {'per_layer': per_layer, 'best': best}

def load_other_results():
    results = {}
    csv_path = BASE_DIR / 'results' / 'feature' / 'species' / 'cca' / 'other_features_weighted.csv'
    if not csv_path.exists():
        return results
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        family = row['family']
        feature_type = row['feature_type']
        if family not in results:
            results[family] = {}

        r_folds = _fold_series(row, ['r_w_f', 'r_f'])

        entry = {
            'r_mean': float(row.get('r_w_mean', row.get('r_mean', np.nan))),
            'r_std': float(row.get('r_w_std', row.get('r_std', np.nan))),
            'r_folds': r_folds if r_folds else None,
            'r2_mean': float(row.get('r2_w_mean', row.get('r2_mean', np.nan))),
            'r2_std': float(row.get('r2_w_std', row.get('r2_std', np.nan))),
            'n_folds': int(row['n_folds']),
            'n_features': int(row['n_features'])
        }

        results[family][feature_type] = entry
    return results

def load_combined_results():
    dnn_res = load_csv_results(BASE_DIR / 'results' / 'feature' / 'species' / 'cca' / 'dnn_features_weighted.csv')
    other_res = load_other_results()
    return {'dnn': dnn_res, 'other': other_res}

def plot_bars_combined(res):
    os.makedirs(OUT_DIR, exist_ok=True)
    dnn = res['dnn']
    other = res['other']
    best = dnn['best']
    model_names = list(set(best['macaque'].keys()) | set(best['human'].keys()))
    models = _ordered_models(model_names) if model_names else []
    model_cols = model_colors(models)
    other_cols = other_colors()
    feature_types = ['visual', 'behavioral', 'semantic']

    setup_style()
    fig = wide_figure()
    ax = fig.add_subplot(111)

    # Parameters
    species_gap = 1.2  # gap between species
    intra_gap = 0.3  # gap within a species between the 3 features and DNN
    width = 0.3  # width of bar
    pos, heights, errs, colors, labels, hatches = [], [], [], [], [], []

    x = 0.0
    centers = []
    arch_lookup = {m: _model_arch(m) for m in models}

    for grp in ['macaque', 'human']:
        species_key = 'monkey' if grp == 'macaque' else grp
        start_idx = len(pos)
        # First, other features
        for i, ftype in enumerate(feature_types):
            fdata = other.get(species_key, {}).get(ftype, {})
            if fdata:
                r_mean, r_std = metric_stats(fdata)
            else:
                r_mean, r_std = (0.0, 0.0)
            pos.append(x)
            heights.append(r_mean)
            errs.append(r_std)
            colors.append(other_cols[i])
            labels.append(ftype)
            hatches.append('')
            x += width
        x += intra_gap  # Add intra-species gap before DNNs
        # Then, DNN models
        prev_arch = None
        for m in models:
            arch = arch_lookup.get(m, 'other')
            if prev_arch is not None and arch != prev_arch:
                x += ARCH_SPACING
            b = best[grp].get(m)
            if b and m in dnn['per_layer'] and b['layer'] in dnn['per_layer'][m]:
                mean_r, std_r = metric_stats(dnn['per_layer'][m][b['layer']][grp])
            else:
                mean_r, std_r = (np.nan, 0.0)
            pos.append(x)
            heights.append(mean_r)
            errs.append(std_r)
            colors.append(model_cols[m])
            labels.append(m)
            hatches.append(_model_hatch(m))
            x += width
            prev_arch = arch
        species_pos = pos[start_idx:]
        if species_pos:
            centers.append((species_pos[0] + species_pos[-1]) / 2)
        x += species_gap

    bars = ax.bar(pos, heights, width=width*0.9, color=colors, edgecolor='black', linewidth=1.2)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax.errorbar(pos, heights, yerr=errs, fmt='none', ecolor='black', elinewidth=2.0, capsize=0, zorder=3)

    style_axes(ax); format_axes(ax, precision=3)
    ax.set_xticks(centers)
    ax.set_xticklabels(['Macaque', 'Human'])
    ax.set_xlabel('Species')
    ax.set_ylabel(metric_label())
    vals = [h + (e if np.isfinite(e) else 0.0) for h, e in zip(heights, errs) if np.isfinite(h)]
    if SET_METRIC == 'r':
        ymax = np.nanmax(vals) if vals else 1.0
        ax.set_ylim(0.0, float(ymax * 1.2))
    else:  # 'r2'
        ymin = min(0.0, np.nanmin([h - (e if np.isfinite(e) else 0.0) for h, e in zip(heights, errs) if np.isfinite(h)]) * 1.1) if vals else 0.0
        ymax = max(0.0, np.nanmax(vals) * 1.1) if vals else 1.0
        ax.set_ylim(ymin, ymax)

    # Legend: other + models
    other_handles = [plt.Rectangle((0,0),1,1,color=other_cols[i], ec='black', lw=1.0, label=ftype.title()) for i, ftype in enumerate(feature_types)]
    model_handles = []
    for m in models:
        rect = plt.Rectangle((0,0),1,1,color=model_cols[m], ec='black', lw=1.0, label=m)
        hatch = _model_hatch(m)
        if hatch:
            rect.set_hatch(hatch)
        model_handles.append(rect)
    all_handles = other_handles + model_handles

    # Save legend separately
    legend_fig = plt.figure(figsize=(6, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.legend(handles=all_handles, frameon=False, loc='center', ncol=min(len(all_handles), 5))
    legend_ax.axis('off')
    legend_path = LEGEND_DIR / f"bars_combined_legend.{FMT}"
    legend_fig.savefig(legend_path, bbox_inches='tight', dpi=600)
    plt.close(legend_fig)
    print(f"Saved legend: {legend_path}")

    fig.tight_layout(pad=0.4)
    out = OUT_DIR / f"bars_combined.{FMT}"
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"Saved: {out}")

def plot_lines_layerwise(res):
    os.makedirs(OUT_DIR, exist_ok=True)
    per = res['dnn']['per_layer']
    model_list = list(per.keys())
    models = _ordered_models(model_list) if model_list else []
    col = model_colors(models)

    ymax = 0.0
    for grp in ['macaque', 'human']:
        for m in models:
            for lay, v in per[m].items():
                mu, sd = metric_stats(v[grp])
                if np.isfinite(mu + sd):
                    ymax = max(ymax, mu + sd)
    ymax = float(ymax * 1.2)
    ymin = 0.0

    def plot_group(grp: str):
        setup_style()
        fig = wide_figure()
        ax = fig.add_subplot(111)
        for m in models:
            layers = per[m]
            rows = []
            for lay, v in layers.items():
                mu, sd = metric_stats(v[grp])
                rows.append((float(v['depth']), mu, sd))
            if not rows:
                continue
            rows.sort(key=lambda t: t[0])
            depth = np.array([t[0] for t in rows])
            mu = np.array([t[1] for t in rows])
            sd = np.array([t[2] for t in rows])
            if depth.size > 0:
                dmin, dmax = depth.min(), depth.max()
                span = dmax - dmin
                d = (depth - dmin) / span * 100.0 if span > 0 else np.zeros_like(depth)
            else:
                d = depth
            ax.plot(d, mu, color=col[m], label=m, lw=2.5)
            ax.fill_between(d, mu - sd, mu + sd, color=col[m], alpha=0.45, linewidth=0)

        style_axes(ax); format_axes(ax, precision=3)
        ax.set_title('Macaque' if grp == 'macaque' else 'Human')
        ax.set_xlabel('Layer depth (%)')
        ax.set_ylabel(metric_label())
        ax.set_ylim(0.0, ymax)  # Simplified - layerwise plots typically positive
        # Legend saved separately below
        fig.tight_layout(pad=0.4)
        out = OUT_DIR / f"layerwise_{grp}.{FMT}"
        fig.savefig(out, bbox_inches='tight', dpi=600)
        plt.close(fig)
        print(f"Saved: {out}")

    plot_group('macaque')
    plot_group('human')

    # Save legend separately
    handles = [plt.Line2D([0],[0], color=col[m], lw=2.5, label=m) for m in models]
    legend_fig = plt.figure(figsize=(4, 3))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.legend(handles=handles, frameon=False, loc='center', ncol=1)
    legend_ax.axis('off')
    legend_path = LEGEND_DIR / f"layerwise_legend.{FMT}"
    legend_fig.savefig(legend_path, bbox_inches='tight', dpi=600)
    plt.close(legend_fig)
    print(f"Saved legend: {legend_path}")

def plot_layer_cloud(res):
    os.makedirs(OUT_DIR, exist_ok=True)
    per = res['dnn']['per_layer']

    def r_of(g: dict) -> float:
        return metric_stats(g)[0]

    xs, ys, cols = [], [], []
    for m, layers in per.items():
        if not layers:
            continue
        depths = [float(v['depth']) for v in layers.values()]
        dmin, dmax = min(depths), max(depths)
        for lay, v in layers.items():
            d = float(v['depth'])
            span = dmax - dmin
            dp = (d - dmin) / span * 100.0 if span > 0 else 0.0
            x = r_of(v['human'])
            y = r_of(v['macaque'])
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x); ys.append(y); cols.append(dp)

    if not xs:
        return

    xs = np.array(xs, float); ys = np.array(ys, float); cols = np.array(cols, float)
    xmax = float(np.nanmax(xs)); ymax = float(np.nanmax(ys))
    pad = 1.1
    x0, x1 = 0.0, min(1.0, xmax * pad if np.isfinite(xmax) else 1.0)
    y0, y1 = 0.0, min(1.0, ymax * pad if np.isfinite(ymax) else 1.0)

    setup_style()
    fig, ax = panel_figure(width='square')

    norm = plt.Normalize(0, 100)
    sc = ax.scatter(xs, ys, c=cols, cmap='magma', norm=norm, s=18, edgecolor='black', linewidth=0.4, zorder=3)
    d0, d1 = max(0.0, min(x0,y0)), min(x1,y1)
    ax.plot([d0, d1], [d0, d1], color='gray', lw=1.2, ls='--', zorder=1)

    style_axes(ax); format_axes(ax, precision=3)
    ax.set_xlabel(f'{metric_label()}, Human')
    ax.set_ylabel(f'{metric_label()}, Macaque')
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Layer depth (%)')

    fig.tight_layout(pad=0.4)
    out = OUT_DIR / f"layer_cloud.{FMT}"
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"Saved: {out}")

def plot_layerwise_both(res):
    os.makedirs(OUT_DIR, exist_ok=True)
    per = res['dnn']['per_layer']
    model_list = list(per.keys())
    models = _ordered_models(model_list) if model_list else []

    def r_of(g: dict) -> float:
        return metric_stats(g)[0]

    hc = config.plotting.get('human_color', '#7c5799')
    mc = config.plotting.get('monkey_color', '#bda855')
    sigc = config.plotting.get('significant', '#961a1a')

    grid = np.linspace(0, 100, 201)
    human_curves, monkey_curves = [], []
    bins = [0,20,40,60,80,100]
    bin_h = [[] for _ in range(len(bins)-1)]
    bin_m = [[] for _ in range(len(bins)-1)]
    ymax = 0.0
    back_alpha = 0.25

    setup_style()
    fig = narrow_figure()
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.78])

    for m in models:
        layers = per[m]
        if not layers:
            continue
        depths = [float(v['depth']) for v in layers.values()]
        dmin, dmax = min(depths), max(depths)
        rows_h, rows_m = [], []
        for lay, v in layers.items():
            d = float(v['depth'])
            span = dmax - dmin
            dp = (d - dmin) / span * 100.0 if span > 0 else 0.0
            rh = r_of(v['human']); rm = r_of(v['macaque'])
            rows_h.append((dp, rh)); rows_m.append((dp, rm))
            for val in (rh, rm):
                if np.isfinite(val): ymax = max(ymax, val)
            for bi in range(len(bins)-1):
                a, b = bins[bi], bins[bi+1]
                if (dp >= a) and (dp < b if b < 100 else dp <= b):
                    if np.isfinite(rh): bin_h[bi].append(float(rh))
                    if np.isfinite(rm): bin_m[bi].append(float(rm))
        rows_h.sort(key=lambda t: t[0]); rows_m.sort(key=lambda t: t[0])
        dh = np.array([t[0] for t in rows_h]); rh = np.array([t[1] for t in rows_h])
        dm = np.array([t[0] for t in rows_m]); rm = np.array([t[1] for t in rows_m])
        ax.plot(dh, rh, color=hc, alpha=back_alpha, lw=1.4)
        ax.plot(dm, rm, color=mc, alpha=back_alpha, lw=1.4)
        if dh.size >= 2 and dm.size >= 2:
            human_curves.append(np.interp(grid, dh, rh, left=np.nan, right=np.nan))
            monkey_curves.append(np.interp(grid, dm, rm, left=np.nan, right=np.nan))

    if human_curves and monkey_curves:
        H = np.vstack(human_curves)
        M = np.vstack(monkey_curves)
        mu_h = np.nanmean(H, axis=0); sd_h = np.nanstd(H, axis=0)
        mu_m = np.nanmean(M, axis=0); sd_m = np.nanstd(M, axis=0)
        ax.plot(grid, mu_h, color=hc, lw=3.0)
        ax.fill_between(grid, mu_h - sd_h, mu_h + sd_h, color=hc, alpha=0.18, linewidth=0)
        ax.plot(grid, mu_m, color=mc, lw=3.0)
        ax.fill_between(grid, mu_m - sd_m, mu_m + sd_m, color=mc, alpha=0.18, linewidth=0)
        ymax = max(ymax, float(np.nanmax(mu_h + sd_h)), float(np.nanmax(mu_m + sd_m)))

        print('\nLayerwise both: per-bin Welch t-tests (layers Human vs Macaque)')
        # --- Collect per-bin p-values, then apply FDR (Benjamini–Hochberg) across bins ---
        pvals = []
        valid_bins = []
        bin_stats = []  # store (i, a, b, h_vals, m_vals, t, p)
        for i in range(len(bins)-1):
            a, b = bins[i], bins[i+1]
            h_vals = np.array(bin_h[i], float)
            m_vals = np.array(bin_m[i], float)
            if h_vals.size < 2 or m_vals.size < 2:
                print(f'  {a:2d}-{b:3d}%: insufficient layers (H={h_vals.size}, M={m_vals.size})')
                continue
            if stats is not None:
                t, p = stats.ttest_ind(h_vals, m_vals, equal_var=False)
                bin_stats.append((i, a, b, h_vals, m_vals, t, p))
                pvals.append(float(p))
                valid_bins.append(i)
                print(f'  {a:2d}-{b:3d}%: H n={h_vals.size:3d}, mean={np.nanmean(h_vals):.3f}; '
                      f'M n={m_vals.size:3d}, mean={np.nanmean(m_vals):.3f}; t={t:.2f}, p={p:.3g}')
            else:
                print(f'  {a:2d}-{b:3d}%: scipy unavailable')

        # Apply FDR if we have p-values
        if pvals:
            pvals = np.array(pvals, float)
            m_tests = len(pvals)
            order = np.argsort(pvals)
            p_sorted = pvals[order]
            ranks = np.arange(1, m_tests + 1, dtype=float)
            q_sorted = p_sorted * m_tests / ranks
            # enforce monotonicity
            q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
            # map back to original order of pvals list
            qvals = np.empty_like(q_sorted)
            qvals[order] = q_sorted

            # Report FDR-adjusted q and shade significant bins
            for k, (i, a, b, h_vals, m_vals, t, p) in enumerate(bin_stats):
                q = qvals[k]
                print(f'    -> FDR q={q:.3g} for bin {a:2d}-{b:3d}%')
                if q < 0.05:
                    h = max(1e-6, 0.06 * (ymax if ymax > 0 else 1.0))
                    ax.fill_between([a, b], [0, 0], [h, h], color='#7a3843', alpha=0.30, linewidth=0)

    style_axes(ax); format_axes(ax, precision=3)
    ax.set_xlabel('Layer depth (%)')
    ax.set_ylabel(metric_label())
    ax.set_title('Layer-wise fit: human vs macaque', pad=4)
    # Fix x-axis to [0,100] with no extra padding (match species_modality style)
    ax.set_xlim(0.0, 100.0)
    ax.margins(x=0)
    if SET_METRIC == 'r':
        ax.set_ylim(0.0, float(ymax * 1.2 if ymax > 0 else 1.0))
    else:  # 'r2'
        ymin_vals = [h - s for h, s in zip(mu_h, sd_h)] + [h - s for h, s in zip(mu_m, sd_m)]
        ymin = min(0.0, np.nanmin(ymin_vals) * 1.1) if ymin_vals else 0.0
        ax.set_ylim(ymin, float(ymax * 1.2 if ymax > 0 else 1.0))

    handles = [
        Line2D([0],[0], color=hc, lw=3.0, label='Human'),
        Line2D([0],[0], color=mc, lw=3.0, label='Macaque'),
        patches.Patch(facecolor='#7a3843', alpha=0.4, edgecolor='none', label='FDR q < 0.05')
    ]

    # Save legend separately
    legend_fig = plt.figure(figsize=(4, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.legend(handles=handles, frameon=False, loc='center', ncol=1)
    legend_ax.axis('off')
    legend_path = LEGEND_DIR / f"layerwise_both_legend.{FMT}"
    legend_fig.savefig(legend_path, bbox_inches='tight', dpi=600)
    plt.close(legend_fig)
    print(f"Saved legend: {legend_path}")

    out = OUT_DIR / f"layerwise_both.{FMT}"
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"Saved: {out}")

def main():
    res = load_combined_results()
    plot_bars_combined(res)
    plot_lines_layerwise(res)
    plot_layer_cloud(res)
    plot_layerwise_both(res)

if __name__ == '__main__':
    main()
