#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from functions.plotting import setup_style, medium_panel, style_axes, panel_figure, narrow_figure
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import r2_score
try:
    from scipy import stats
except Exception:
    stats = None
from model.feature.core.feature_io import load_cca_components, load_vis, load_sem
from model.feature.core.ridge_utils import get_ridge_params, get_ridge_cv_settings
from model.CCA.significance import get_significant_component_indices

# ---- Key choices ----
DOMAIN = 'cca'
FAMILIES = ['human', 'monkey']  # species-specific spaces
CV_SETTINGS = get_ridge_cv_settings()
N_FOLDS = int(CV_SETTINGS['outer_folds'])
N_REPEATS = int(CV_SETTINGS['outer_repeats'])
INNER_FOLDS = int(CV_SETTINGS['inner_folds'])
SEED = int(CV_SETTINGS['seed'])
MAX_COMPONENTS = 30  # plotting crop only
SUPP_TOP_KS = [5, 10, 15, 20]
PLOT_SUPP = False
USE_COMPONENT_STATS = False
BOOTSTRAP = True
BOOT_N = int(config.hyperparameters.get('modality', {}).get('n_boot', 200))
BOOT_SEED = int(config.hyperparameters.get('modality', {}).get('seed', 42))
BOOT_COMP_CROP = int(config.hyperparameters.get('modality', {}).get('comp_crop', 32))
BOOT_VERBOSE = int(config.hyperparameters.get('modality', {}).get('verbose', 10))
BOOT_JOBS = int(config.hyperparameters.get('modality', {}).get('n_jobs', min(int(config.analysis.get('n_jobs', 4)), 32)))

# Local color scheme (distinct from global plotting defaults)
VISUAL_COLOR = '#55889E'
SEMANTIC_COLOR = '#C7522E'
SHARED_COLOR = '#B2996E'  # darker than #E1D3B8


def _family_color(fam: str):
    plot_cfg = getattr(config, 'plotting', {}) or {}
    return {
        'human': plot_cfg.get('human_color', '#7c5799'),
        'monkey': plot_cfg.get('monkey_color', '#bda855'),
        'cross-species': plot_cfg.get('shared_color', '#81B7B3'),
        'all': plot_cfg.get('shared_color', '#81B7B3'),
    }.get(fam, '#4c5b6b')


def _family_label(fam: str):
    return {
        'human': 'Human',
        'monkey': 'Monkey',
        'cross-species': 'Cross-species',
        'all': 'All',
    }.get(fam, fam)


def _cca_family_to_key(family: str) -> str:
    return {'cross-species': 'all', 'human': 'human', 'monkey': 'monkey'}.get(family, 'all')


def _load_comp_weights() -> dict:
    """Load held-out canonical correlations from crossview_results.pkl.

    Returns dict: {'cross-species': arr, 'human': arr, 'monkey': arr}.
    If file is missing, returns empty dict to trigger equal-weight fallback.
    """
    p = config.results_dir / 'crossview_results.pkl'
    if not p.exists():
        print(f"[species_modality] crossview_results not found at {p}; using equal weights")
        return {}
    import pickle
    with open(p, 'rb') as f:
        d = pickle.load(f)
    out = {}
    if 'cross-species' in d:
        out['cross-species'] = np.asarray(d['cross-species'].get('comp_corrs', []), float)
    if 'human_only' in d:
        out['human'] = np.asarray(d['human_only'].get('comp_corrs', []), float)
    if 'monkey_only' in d:
        out['monkey'] = np.asarray(d['monkey_only'].get('comp_corrs', []), float)
    return out


def _comp_weights(family: str, n_comp: int) -> np.ndarray:
    w_all = _load_comp_weights()
    w = np.asarray(w_all.get(family, []), float)
    keep = get_significant_component_indices(family)
    if keep.size == n_comp and keep.size and keep.max() < w.size:
        w = w[keep]
    if w.size == 0:
        w = np.ones(n_comp, float)
    if w.size < n_comp:
        pad = np.ones(n_comp - w.size, float)
        w = np.concatenate([w, pad])
    w = np.maximum(w[:n_comp], 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return w


def _ridge_cv_metrics(X, y, n_folds=N_FOLDS, seed=SEED, alphas=None):
    alphas = np.asarray(alphas if alphas is not None else [0.001,0.01,0.1,1.0,10.0], float)
    outer_cv = KFold(n_splits=int(n_folds), shuffle=True, random_state=seed) if N_REPEATS == 1 else None
    if outer_cv is None:
        splitter = RepeatedKFold(
            n_splits=int(n_folds),
            n_repeats=max(1, N_REPEATS),
            random_state=seed,
        )
    else:
        splitter = outer_cv
    vals_r2, vals_r = [], []
    for tr, te in splitter.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        xs = StandardScaler().fit(Xtr)
        Xtr = xs.transform(Xtr); Xte = xs.transform(Xte)
        inner_splits = max(2, min(INNER_FOLDS, len(tr)))
        rc = RidgeCV(alphas=alphas, cv=inner_splits)
        rc.fit(Xtr, ytr)
        yp = rc.predict(Xte)
        r2 = r2_score(yte, yp)
        vals_r2.append(float(r2))
        vals_r.append(float(np.sqrt(max(r2, 0.0))))
    vr2 = np.array(vals_r2, float)
    vr = np.array(vals_r, float)
    return (
        float(vr.mean()), float(vr.std()), vr.tolist(),
        float(vr2.mean()), float(vr2.std()), vr2.tolist()
    )


def _ridge_cv_metrics_fixed(X, y, alpha, n_folds=N_FOLDS, seed=SEED):
    outer_cv = KFold(n_splits=int(n_folds), shuffle=True, random_state=seed) if N_REPEATS == 1 else None
    if outer_cv is None:
        splitter = RepeatedKFold(
            n_splits=int(n_folds),
            n_repeats=max(1, N_REPEATS),
            random_state=seed,
        )
    else:
        splitter = outer_cv
    vals_r2, vals_r = [], []
    for tr, te in splitter.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        xs = StandardScaler().fit(Xtr)
        Xtr = xs.transform(Xtr); Xte = xs.transform(Xte)
        rc = Ridge(alpha=float(alpha))
        rc.fit(Xtr, ytr)
        yp = rc.predict(Xte)
        r2 = r2_score(yte, yp)
        vals_r2.append(float(r2))
        vals_r.append(float(np.sqrt(max(r2, 0.0))))
    vr2 = np.array(vals_r2, float)
    vr = np.array(vals_r, float)
    return (
        float(vr.mean()), float(vr.std()), vr.tolist(),
        float(vr2.mean()), float(vr2.std()), vr2.tolist()
    )


def _prefit_alphas(data: dict, alphas):
    out = {}
    for fam, ent in data.items():
        Xv = ent['Xv']; Xs = ent['Xs']; Y = ent['Y']
        Xvs = np.hstack([Xv, Xs])

        def _fit_alpha(X, y):
            xs = StandardScaler().fit(X)
            Xz = xs.transform(X)
            rc = RidgeCV(alphas=alphas, cv=max(2, INNER_FOLDS))
            rc.fit(Xz, y)
            return float(rc.alpha_)

        y0 = Y[:, 0]
        out[fam] = {
            'visual': _fit_alpha(Xv, y0),
            'semantic': _fit_alpha(Xs, y0),
            'joint': _fit_alpha(Xvs, y0),
        }
        print(f"[species_modality] {fam}: prefit alphas vis={out[fam]['visual']:.4g} sem={out[fam]['semantic']:.4g} joint={out[fam]['joint']:.4g}")
    return out


def _venn_for_family(family: str, n_folds=N_FOLDS, seed=SEED, top_k=None):
    key = _cca_family_to_key(family)
    Y, stims = load_cca_components(key)  # (n_stim, n_comp)
    Xv, _, _ = load_vis(stims)
    Xs, _, _ = load_sem(stims)
    if Xv.shape[1] == 0 or Xs.shape[1] == 0:
        print(f"[species_modality] {family}: missing visual or semantic features")
        return None

    # component alignment weights (for both selection + weighting)
    w_raw = _comp_weights(family, Y.shape[1])
    comp_ids = np.arange(Y.shape[1], dtype=int)
    if top_k is not None and top_k > 0 and top_k < Y.shape[1]:
        idx = np.argsort(w_raw)[::-1][:int(top_k)]
        Y = Y[:, idx]
        w_raw = w_raw[idx]
        comp_ids = comp_ids[idx]

    alphas, _ = get_ridge_params()
    r_v = []; r_s = []; r_vs = []
    r2_v = []; r2_s = []; r2_vs = []
    for k in range(Y.shape[1]):
        y = Y[:, k]
        mr, _, _, mr2, _, _ = _ridge_cv_metrics(Xv, y, n_folds=n_folds, seed=seed, alphas=alphas)
        r_v.append(mr); r2_v.append(mr2)
        mr, _, _, mr2, _, _ = _ridge_cv_metrics(Xs, y, n_folds=n_folds, seed=seed, alphas=alphas)
        r_s.append(mr); r2_s.append(mr2)
        Xvs = np.hstack([Xv, Xs])
        mr, _, _, mr2, _, _ = _ridge_cv_metrics(Xvs, y, n_folds=n_folds, seed=seed, alphas=alphas)
        r_vs.append(mr); r2_vs.append(mr2)
        if (k+1) % 10 == 0 or (k+1) == Y.shape[1]:
            print(f"[species_modality] {family}: comp {k+1}/{Y.shape[1]}")
    r_v = np.asarray(r_v, float)
    r_s = np.asarray(r_s, float)
    r_vs = np.asarray(r_vs, float)
    r2_v = np.asarray(r2_v, float)
    r2_s = np.asarray(r2_s, float)
    r2_vs = np.asarray(r2_vs, float)

    uniq_v = r2_vs - r2_s
    uniq_s = r2_vs - r2_v
    shared = (r2_v + r2_s) - r2_vs
    shared_r = np.sqrt(np.clip(shared, 0.0, None))

    # Weights from crossview held-out component correlations
    w = np.asarray(w_raw, float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()

    uv_r2 = float((w * uniq_v).sum())
    us_r2 = float((w * uniq_s).sum())
    sh_r2 = float((w * shared).sum())
    out = {
        'family': family,
        'n_components': int(Y.shape[1]),
        'top_k': int(top_k) if top_k is not None else None,
        'weight_sum': float(w.sum()),
        'r_visual_weighted': float((w * r_v).sum()),
        'r_semantic_weighted': float((w * r_s).sum()),
        'r_joint_weighted': float((w * r_vs).sum()),
        'r2_visual_weighted': float((w * r2_v).sum()),
        'r2_semantic_weighted': float((w * r2_s).sum()),
        'r2_joint_weighted': float((w * r2_vs).sum()),
        'unique_visual_r2': uv_r2,
        'unique_semantic_r2': us_r2,
        'shared_r2': sh_r2,
        'unique_visual_r': float(np.sqrt(max(uv_r2, 0.0))),
        'unique_semantic_r': float(np.sqrt(max(us_r2, 0.0))),
        'shared_r': float(np.sqrt(max(sh_r2, 0.0))),
    }
    per_comp = pd.DataFrame({
        'component': comp_ids + 1,
        'r_visual': r_v,
        'r_semantic': r_s,
        'r_shared': shared_r,
        'r_joint': r_vs,
        'r2_visual': r2_v,
        'r2_semantic': r2_s,
        'r2_shared': shared,
        'r2_joint': r2_vs,
    })
    return out, per_comp


def _bootstrap_semvis(families, n_boot=200, seed=42, comp_crop=None, top_k=None, n_folds=N_FOLDS):
    data = {}
    n_stim = None
    for fam in families:
        key = _cca_family_to_key(fam)
        Y, stims = load_cca_components(key)
        Xv, _, _ = load_vis(stims)
        Xs, _, _ = load_sem(stims)
        if n_stim is None:
            n_stim = Y.shape[0]
        if Y.shape[0] != n_stim:
            raise ValueError(f"[bootstrap] mismatched stim count for {fam}: {Y.shape[0]} vs {n_stim}")

        w = _comp_weights(fam, Y.shape[1])
        if top_k is not None and top_k > 0 and top_k < Y.shape[1]:
            idx = np.argsort(w)[::-1][:int(top_k)]
            Y = Y[:, idx]
            w = w[idx]
        if comp_crop is not None and comp_crop > 0 and comp_crop < Y.shape[1]:
            Y = Y[:, :int(comp_crop)]
            w = w[:int(comp_crop)]

        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        data[fam] = {'Y': Y, 'Xv': Xv, 'Xs': Xs, 'w': w, 'n_comp': Y.shape[1]}

    alphas, _ = get_ridge_params()
    rng = np.random.default_rng(int(seed))
    seeds = rng.integers(0, 1_000_000_000, size=int(n_boot))

    n_jobs = max(1, min(int(BOOT_JOBS), int(n_boot)))

    # prefit alphas once per family/feature set
    alpha_map = _prefit_alphas(data, alphas)

    def _one(bseed):
        bseed = int(bseed)
        brng = np.random.default_rng(bseed)
        idx = brng.integers(0, n_stim, size=n_stim)
        out = {}
        for fam, ent in data.items():
            Yb = ent['Y'][idx]
            Xvb = ent['Xv'][idx]
            Xsb = ent['Xs'][idx]
            Xvsb = np.hstack([Xvb, Xsb])
            w = ent['w']
            r2_v = np.zeros(ent['n_comp'], float)
            r2_s = np.zeros(ent['n_comp'], float)
            r2_vs = np.zeros(ent['n_comp'], float)
            for k in range(ent['n_comp']):
                y = Yb[:, k]
                _, _, _, mr2, _, _ = _ridge_cv_metrics_fixed(Xvb, y, alpha_map[fam]['visual'], n_folds=n_folds, seed=bseed)
                r2_v[k] = mr2
                _, _, _, mr2, _, _ = _ridge_cv_metrics_fixed(Xsb, y, alpha_map[fam]['semantic'], n_folds=n_folds, seed=bseed)
                r2_s[k] = mr2
                _, _, _, mr2, _, _ = _ridge_cv_metrics_fixed(Xvsb, y, alpha_map[fam]['joint'], n_folds=n_folds, seed=bseed)
                r2_vs[k] = mr2
            uniq_v = r2_vs - r2_s
            uniq_s = r2_vs - r2_v
            uv = float((w * uniq_v).sum())
            us = float((w * uniq_s).sum())
            dom = float(us - uv)
            out[fam] = (uv, us, dom)
        return out

    outs = Parallel(
        n_jobs=n_jobs,
        prefer='processes',
        backend='loky',
        max_nbytes='64M',
        mmap_mode='r',
        verbose=BOOT_VERBOSE
    )(delayed(_one)(s) for s in seeds)

    boot = {fam: {'unique_visual_r2': [], 'unique_semantic_r2': [], 'dominance': []} for fam in families}
    for r in outs:
        for fam in families:
            uv, us, dom = r[fam]
            boot[fam]['unique_visual_r2'].append(uv)
            boot[fam]['unique_semantic_r2'].append(us)
            boot[fam]['dominance'].append(dom)

    def _summ(vals):
        v = np.asarray(vals, float)
        mu = float(np.mean(v))
        lo, hi = np.quantile(v, [0.025, 0.975])
        p = float(2.0 * min((v <= 0).mean(), (v >= 0).mean()))
        return mu, float(lo), float(hi), p

    rows = []
    for fam in families:
        for metric in ['unique_visual_r2', 'unique_semantic_r2', 'dominance']:
            mu, lo, hi, p = _summ(boot[fam][metric])
            rows.append({
                'family': fam,
                'metric': metric,
                'mean': mu,
                'ci_low': lo,
                'ci_high': hi,
                'p_two_sided': p,
                'n_boot': int(n_boot),
                'comp_crop': int(comp_crop) if comp_crop is not None else None,
                'top_k': int(top_k) if top_k is not None else None,
                'n_folds': int(n_folds),
                'n_repeats': int(N_REPEATS),
                'seed': int(seed),
            })

    if 'human' in families and 'monkey' in families:
        dh = np.asarray(boot['human']['dominance'], float)
        dm = np.asarray(boot['monkey']['dominance'], float)
        delta = dh - dm
        mu, lo, hi, p = _summ(delta)
        rows.append({
            'family': 'human_minus_monkey',
            'metric': 'dominance_diff',
            'mean': mu,
            'ci_low': lo,
            'ci_high': hi,
            'p_two_sided': p,
            'n_boot': int(n_boot),
            'comp_crop': int(comp_crop) if comp_crop is not None else None,
            'top_k': int(top_k) if top_k is not None else None,
            'n_folds': int(n_folds),
            'n_repeats': int(N_REPEATS),
            'seed': int(seed),
        })

    out_dir = config.results_dir / 'feature' / 'modality_components' / DOMAIN
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_crop{int(comp_crop)}" if comp_crop is not None else ""
    p = out_dir / f'bootstrap_semvis{tag}.csv'
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"[species_modality] wrote {p}")


def _plot_venn(vals: dict):
    from matplotlib_venn import venn2
    setup_style()
    fig, ax = panel_figure(width='square')
    ax.set_axis_off()

    # Signed values for labels; non-negative for area (use r^2 for areas/labels)
    uv_signed = float(vals.get('unique_visual_r2', 0.0))
    us_signed = float(vals.get('unique_semantic_r2', 0.0))
    sh_signed = float(vals.get('shared_r2', 0.0))
    a = max(uv_signed, 0.0)
    b = max(us_signed, 0.0)
    ab = max(sh_signed, 0.0)

    v = venn2(subsets=(a, b, ab), set_labels=('Visual', 'Semantic'), ax=ax)

    if v.get_patch_by_id('10'):
        p = v.get_patch_by_id('10')
        p.set_facecolor(VISUAL_COLOR)
        p.set_edgecolor('black')
        p.set_alpha(0.7)
    if v.get_patch_by_id('01'):
        p = v.get_patch_by_id('01')
        p.set_facecolor(SEMANTIC_COLOR)
        p.set_edgecolor('black')
        p.set_alpha(0.7)
    if v.get_patch_by_id('11'):
        p = v.get_patch_by_id('11')
        p.set_facecolor(SHARED_COLOR)
        p.set_edgecolor('black')
        p.set_alpha(0.7)

    # Replace subset labels with signed values (3 decimals, r^2 units)
    if v.get_label_by_id('10'):
        v.get_label_by_id('10').set_text(f"{uv_signed:.3f}")
    if v.get_label_by_id('01'):
        v.get_label_by_id('01').set_text(f"{us_signed:.3f}")
    if v.get_label_by_id('11'):
        v.get_label_by_id('11').set_text(f"{sh_signed:.3f}")
    # Tweak set labels
    if v.set_labels[0]: v.set_labels[0].set_color('black')
    if v.set_labels[1]: v.set_labels[1].set_color('black')

    ax.set_title(f"{_family_label(vals['family'])} weighted variance partition (r)", pad=2)

    out_dir = config.fig_dir / 'feature' / 'venn'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"venn_{vals['family']}.pdf"
    fig.savefig(out_p, bbox_inches='tight')
    plt.close(fig)
    print(f"[species_modality] saved {out_p}")


def _plot_component_lines(family: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    setup_style()
    fig = narrow_figure()
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.78])

    if MAX_COMPONENTS and MAX_COMPONENTS > 0:
        df_plot = df.head(MAX_COMPONENTS)
    else:
        df_plot = df
    x = df_plot['component'].to_numpy()
    ax.plot(x, df_plot['r_visual'], color=VISUAL_COLOR, label='Visual', linewidth=3.4)
    ax.plot(x, df_plot['r_semantic'], color=SEMANTIC_COLOR, label='Semantic', linewidth=3.4)
    ax.plot(x, df_plot['r_shared'], color=SHARED_COLOR, label='Shared', linewidth=3.4)

    ax.set_xlabel('Component')
    ax.set_ylabel('Cross-validated partial r')
    ax.set_title(f"{_family_label(family)} component-wise fit", pad=4)
    ax.set_ylim(0.0, 0.8)
    ax.legend(frameon=False, loc='upper right')
    style_axes(ax)

    out_dir = config.fig_dir / 'feature' / 'venn'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"component_traces_{family}.pdf"
    fig.savefig(out_p, bbox_inches='tight')
    plt.close(fig)
    print(f"[species_modality] saved {out_p}")


def _component_stats(per_comp: dict, summaries: dict, out_fig_dir, out_name: str = 'species_modality_stats.csv'):
    if 'human' not in per_comp or 'monkey' not in per_comp:
        return
    dh = per_comp['human']; dm = per_comp['monkey']
    if dh.empty or dm.empty:
        return
    # Align by component index
    n = int(min(dh.shape[0], dm.shape[0]))
    dh = dh.iloc[:n].copy(); dm = dm.iloc[:n].copy()
    d_h = dh['r_semantic'] - dh['r_visual']
    d_m = dm['r_semantic'] - dm['r_visual']
    inter = d_h - d_m

    def _summ(label, fam, d):
        v = d.to_numpy().astype(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {'label': label, 'family': fam, 'n_components': 0}
        tval = np.nan; pval = np.nan
        if stats is not None and v.size > 1:
            tval, pval = stats.ttest_1samp(v, 0.0)
        return {
            'label': label,
            'family': fam,
            'n_components': int(v.size),
            'mean_delta_r': float(v.mean()),
            'std_delta_r': float(v.std(ddof=1)) if v.size>1 else 0.0,
            't_value_vs_zero': float(tval) if np.isfinite(tval) else np.nan,
            'p_value_vs_zero': float(pval) if np.isfinite(pval) else np.nan,
        }

    rows = []
    # Weighted roll-ups (per full CCA space)
    for fam in ('human','monkey'):
        s = summaries.get(fam, {})
        if not s: continue
        rows.append({
            'label': f'{fam}_weighted',
            'family': fam,
            'n_components': int(s.get('n_components', 0)),
            'r_visual_weighted': float(s.get('r_visual_weighted', np.nan)),
            'r_semantic_weighted': float(s.get('r_semantic_weighted', np.nan)),
            'r_joint_weighted': float(s.get('r_joint_weighted', np.nan)),
            'unique_visual_r': float(s.get('unique_visual_r', np.nan)),
            'unique_semantic_r': float(s.get('unique_semantic_r', np.nan)),
            'shared_r': float(s.get('shared_r', np.nan)),
            'unique_visual_r2': float(s.get('unique_visual_r2', np.nan)),
            'unique_semantic_r2': float(s.get('unique_semantic_r2', np.nan)),
            'shared_r2': float(s.get('shared_r2', np.nan)),
            'unique_delta_r': float(s.get('unique_semantic_r', np.nan) - s.get('unique_visual_r', np.nan)),
            'unique_delta_r2': float(s.get('unique_semantic_r2', np.nan) - s.get('unique_visual_r2', np.nan)),
        })

    rows.append(_summ('human_sem_minus_vis', 'human', d_h))
    rows.append(_summ('monkey_sem_minus_vis', 'monkey', d_m))
    rows.append(_summ('human_minus_monkey_delta', 'human_minus_monkey', inter))

    out_dir = out_fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / out_name
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"[species_modality] wrote {p}")


if __name__ == '__main__':
    def _run(top_k=None, tag=''):
        rows = []
        per_comp = {}
        summaries = {}
        for fam in FAMILIES:
            res = _venn_for_family(fam, n_folds=N_FOLDS, seed=SEED, top_k=top_k)
            if res is None:
                continue
            vals, df = res
            rows.append(vals)
            per_comp[fam] = df
            summaries[fam] = vals
            if top_k is None or PLOT_SUPP:
                _plot_venn(vals)

        if rows:
            res_dir = config.results_dir / 'feature' / 'modality_components' / DOMAIN
            res_dir.mkdir(parents=True, exist_ok=True)
            name = f'venn_weighted_variance{tag}.csv'
            p = res_dir / name
            pd.DataFrame(rows).to_csv(p, index=False)
            print(f"[species_modality] wrote {p}")
            fig_dir = config.fig_dir / 'feature' / 'venn'
            fig_dir.mkdir(parents=True, exist_ok=True)
            for fam, df in per_comp.items():
                if df is None or df.empty:
                    continue
                if top_k is None or PLOT_SUPP:
                    _plot_component_lines(fam, df)
                fp = res_dir / f'{fam}_component_traces{tag}.csv'
                df.to_csv(fp, index=False)
                print(f"[species_modality] wrote {fp}")
            if USE_COMPONENT_STATS:
                _component_stats(per_comp, summaries, fig_dir, out_name=f'species_modality_stats{tag}.csv')

    # main (full)
    _run(top_k=None, tag='')
    # supplementary: top-k most aligned axes
    for k in SUPP_TOP_KS:
        _run(top_k=int(k), tag=f'_top{k}')

    if BOOTSTRAP:
        comp_crop = BOOT_COMP_CROP if BOOT_COMP_CROP > 0 else None
        _bootstrap_semvis(FAMILIES, n_boot=BOOT_N, seed=BOOT_SEED, comp_crop=comp_crop, top_k=None, n_folds=N_FOLDS)
