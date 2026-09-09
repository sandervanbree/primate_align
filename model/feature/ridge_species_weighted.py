#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import sys, pickle
import toml
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from model.feature.core.feature_io import (
    load_vis, load_beh, load_sem,
    load_cca_components, load_srf_components
)
from model.feature.core.ridge_utils import get_ridge_params
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from model.feature.core.ridge_utils import get_ridge_cv_settings
from model.CCA.significance import get_significant_component_indices

# ---------------------------------------------------------------------
BASE_DIR = Path(config.root_dir)
RESULTS_ROOT = BASE_DIR / 'results' / 'feature' / 'species'
CV_SETTINGS = get_ridge_cv_settings()
OUTER_FOLDS = int(CV_SETTINGS['outer_folds'])
OUTER_REPEATS = int(CV_SETTINGS['outer_repeats'])
INNER_FOLDS = int(CV_SETTINGS['inner_folds'])
RNG_SEED = int(CV_SETTINGS['seed'])
PCA_MAX_DIM = int(CV_SETTINGS['pca_max_dim'])


def _load_comp_weights() -> dict:
    """Load held-out cross-view component correlations."""
    p = config.results_dir / 'crossview_results.pkl'
    if not p.exists():
        print(f"[ridge_species_weighted] crossview_results not found at {p}; using equal weights")
        return {}
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
    return w / w.sum()

def load_dnn_features(stims):
    """Load per-layer DNN features via features_summary.tsv and collect depths from layer_seq.

    Only keep layers listed in config/dnn_layers.toml per model.
    """
    base = Path(config.dnn_dir) / 'features'
    tsv = base / 'features_summary.tsv'
    if not tsv.exists():
        return {}, {}
    n = len(stims)
    df = pd.read_csv(tsv, sep='\t')
    # filter by config/dnn_layers.toml
    sel_p = Path(config.root_dir) / 'config' / 'dnn_layers.toml'
    sel = toml.load(sel_p)
    allow = {}
    for ent in sel.get('model', []):
        name = str(ent.get('name','')).strip().lower()
        lays = {str(x) for x in ent.get('layers', [])}
        if name and lays:
            allow[name] = lays
    df = df[df.apply(lambda r: str(r['layer_name']) in allow.get(str(r['model']).strip().lower(), set()), axis=1)]
    out, depths = {}, {}
    for _, r in df.iterrows():
        model = str(r['model']).strip()
        layer = str(r['layer_name']).strip()
        p2 = Path(config.dnn_dir) / str(r['save_path']).strip()
        p2 = p2.parent / 'features' / 'features.npy'
        X = np.load(p2)
        k = f"{model}_{layer}"
        out[k] = X.astype(np.float32, copy=False)
        depths[k] = float(r['layer_seq'])
    return out, depths


def _get_comps_and_stims(domain: str, family: str):
    if domain == 'cca':
        fam_map = {'cross-species': 'all', 'human': 'human', 'monkey': 'monkey'}
        key = fam_map.get(family, 'all')
        comps, stims = load_cca_components(key)
    else:
        comps, stims = load_srf_components(family)
    return comps, stims


def compute_species_scores(domain='cca', families=None):
    if families is None:
        families = (['cross-species', 'human', 'monkey'] if domain == 'cca' else ['all', 'human', 'monkey'])

    for family in families:
        try:
            Y, stims = _get_comps_and_stims(domain, family)
        except Exception:
            continue
        w_comp = _comp_weights(family, Y.shape[1])

        X_vis, names_vis, _ = load_vis(stims)
        X_beh, names_beh, _ = load_beh(stims)
        X_sem, names_sem, _ = load_sem(stims)
        dnn_features, dnn_depths = load_dnn_features(stims)

        results = {}
        n_jobs = int(config.analysis.get('n_jobs', 4))
        print(f"[ridge_species_weighted] {domain}/{family}: Y={Y.shape} vis={X_vis.shape} beh={X_beh.shape} sem={X_sem.shape} dnn_layers={len(dnn_features)}")

        # ridge params
        alphas, _ = get_ridge_params()

        def ridge_cv(X, Y, pca_n=None):
            outer_cv = RepeatedKFold(
                n_splits=max(2, OUTER_FOLDS),
                n_repeats=max(1, OUTER_REPEATS),
                random_state=RNG_SEED,
            )
            vals_r2, vals_r = [], []
            vals_r2_w, vals_r_w = [], []
            for tr, te in outer_cv.split(X):
                Xtr, Xte = X[tr], X[te]
                Ytr, Yte = Y[tr], Y[te]
                xs = StandardScaler().fit(Xtr)
                Xtr = xs.transform(Xtr); Xte = xs.transform(Xte)
                if pca_n and Xtr.shape[1] > pca_n:
                    k = int(min(pca_n, Xtr.shape[1], Xtr.shape[0] - 1))
                    pca = PCA(n_components=k, svd_solver='randomized', random_state=RNG_SEED)
                    Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
                inner_splits = max(2, min(INNER_FOLDS, len(tr)))
                rc = RidgeCV(alphas=alphas, cv=inner_splits).fit(Xtr, Ytr)
                Yp = rc.predict(Xte)
                # Per-output held-out R²; primary encoding metric.
                num = ((Yte - Yp)**2).sum(axis=0)
                den = ((Yte - Yte.mean(axis=0))**2).sum(axis=0) + 1e-12
                r2_vec = 1.0 - (num / den)  # (n_comp,)

                # Per-output held-out Pearson r; retained as a diagnostic.
                yt = Yte - Yte.mean(axis=0, keepdims=True)
                yp = Yp - Yp.mean(axis=0, keepdims=True)
                r_num = (yt * yp).sum(axis=0)
                r_den = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0)) + 1e-12
                r_vec = r_num / r_den  # (n_comp,)

                fold_r2 = float(np.mean(r2_vec))
                fold_r = float(np.mean(r_vec))
                vals_r2.append(fold_r2)
                vals_r.append(fold_r)

                fold_r2_w = float(np.sum(w_comp * r2_vec))
                fold_r_w = float(np.sum(w_comp * r_vec))
                vals_r2_w.append(fold_r2_w)
                vals_r_w.append(fold_r_w)
            vals_r2 = np.array(vals_r2, float)
            vals_r = np.array(vals_r, float)
            vals_r2_w = np.array(vals_r2_w, float)
            vals_r_w = np.array(vals_r_w, float)
            return (
                float(vals_r.mean()), float(vals_r.std()), [float(x) for x in vals_r],
                float(vals_r2.mean()), float(vals_r2.std()), [float(x) for x in vals_r2],
                float(vals_r_w.mean()), float(vals_r_w.std()), [float(x) for x in vals_r_w],
                float(vals_r2_w.mean()), float(vals_r2_w.std()), [float(x) for x in vals_r2_w],
            )

        # Visual joint model
        if X_vis.shape[1] > 0:
            mr, sr, rfolds, mr2, sr2, r2folds, mrw, srw, rwfolds, mr2w, sr2w, r2wfolds = ridge_cv(X_vis, Y)
            results['visual'] = {
                'r_mean': mr, 'r_std': sr, 'r_folds': rfolds,
                'r2_mean': mr2, 'r2_std': sr2, 'r2_folds': r2folds,
                'r_w_mean': mrw, 'r_w_std': srw, 'r_w_folds': rwfolds,
                'r2_w_mean': mr2w, 'r2_w_std': sr2w, 'r2_w_folds': r2wfolds,
                'n_features': X_vis.shape[1]
            }

        # Behavioral joint model
        if X_beh.shape[1] > 0:
            mr, sr, rfolds, mr2, sr2, r2folds, mrw, srw, rwfolds, mr2w, sr2w, r2wfolds = ridge_cv(X_beh, Y)
            results['behavioral'] = {
                'r_mean': mr, 'r_std': sr, 'r_folds': rfolds,
                'r2_mean': mr2, 'r2_std': sr2, 'r2_folds': r2folds,
                'r_w_mean': mrw, 'r_w_std': srw, 'r_w_folds': rwfolds,
                'r2_w_mean': mr2w, 'r2_w_std': sr2w, 'r2_w_folds': r2wfolds,
                'n_features': X_beh.shape[1]
            }

        # Semantic joint model
        if X_sem.shape[1] > 0:
            mr, sr, rfolds, mr2, sr2, r2folds, mrw, srw, rwfolds, mr2w, sr2w, r2wfolds = ridge_cv(X_sem, Y)
            results['semantic'] = {
                'r_mean': mr, 'r_std': sr, 'r_folds': rfolds,
                'r2_mean': mr2, 'r2_std': sr2, 'r2_folds': r2folds,
                'r_w_mean': mrw, 'r_w_std': srw, 'r_w_folds': rwfolds,
                'r2_w_mean': mr2w, 'r2_w_std': sr2w, 'r2_w_folds': r2wfolds,
                'n_features': X_sem.shape[1]
            }

        # DNN layer models (with optional PCA)
        results['dnn'] = {}
        pca_n = PCA_MAX_DIM
        if dnn_features:
            items = list(dnn_features.items())
            completed = 0
            n_jobs_eff = max(1, min(n_jobs, len(items)))

            def _fit(pair):
                nonlocal completed
                k, Xd = pair
                if Xd.shape[0] != len(stims):
                    return k, None
                mr, sr, rfolds, mr2, sr2, r2folds, mrw, srw, rwfolds, mr2w, sr2w, r2wfolds = ridge_cv(Xd, Y, pca_n=pca_n)
                completed += 1
                if completed % 15 == 0 or completed == len(items):
                    print(f"[ridge_species_weighted] {domain}/{family}: dnn {completed}/{len(items)} layers")
                return k, (mr, sr, rfolds, mr2, sr2, r2folds, mrw, srw, rwfolds, mr2w, sr2w, r2wfolds, Xd.shape[1])
            outs = Parallel(n_jobs=n_jobs_eff)(delayed(_fit)(it) for it in items)
            for k, val in outs:
                if val is None: continue
                mr, sr, rfolds, mr2, sr2, r2folds, mrw, srw, rwfolds, mr2w, sr2w, r2wfolds, nf = val
                results['dnn'][k] = {
                    'r_mean': mr, 'r_std': sr, 'r_folds': rfolds,
                    'r2_mean': mr2, 'r2_std': sr2, 'r2_folds': r2folds,
                    'r_w_mean': mrw, 'r_w_std': srw, 'r_w_folds': rwfolds,
                    'r2_w_mean': mr2w, 'r2_w_std': sr2w, 'r2_w_folds': r2wfolds,
                    'n_features': nf
                }
            print(f"[ridge_species_weighted] {domain}/{family}: dnn {len(results['dnn'])}/{len(dnn_features)} layers done")

        # Save species results
        out_dir = RESULTS_ROOT / domain
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / f'{family}.pkl', 'wb') as f:
            pickle.dump(results, f)

        # Save per-family CSV (unweighted + weighted columns)
        rows = []
        for ftype in ['visual', 'behavioral', 'semantic']:
            if ftype in results:
                r = results[ftype]
                rows.append({
                    'feature_type': ftype,
                    'r2_mean': r['r2_mean'], 'r2_std': r['r2_std'],
                    'r2_w_mean': r['r2_w_mean'], 'r2_w_std': r['r2_w_std'],
                    'n_features': r['n_features']
                })
        for layer_key, r in results.get('dnn', {}).items():
            rows.append({
                'feature_type': f'dnn_{layer_key}',
                'r2_mean': r['r2_mean'], 'r2_std': r['r2_std'],
                'r2_w_mean': r['r2_w_mean'], 'r2_w_std': r['r2_w_std'],
                'n_features': r['n_features']
            })
        pd.DataFrame(rows).to_csv(out_dir / f'{family}.csv', index=False)
        print(f"[ridge_species_weighted] {domain}/{family}: done")

        # Aggregate rows for new CSVs
        def _fold_cols(folds, prefix='r2_f'):
            out = {}
            for i, v in enumerate(folds, 1):
                out[f'{prefix}{i}'] = v
            return out

        if not hasattr(compute_species_scores, '_buf_other'):
            compute_species_scores._buf_other = []
            compute_species_scores._buf_dnn = []

        for ftype in ['visual','behavioral','semantic']:
            if ftype in results:
                ent = results[ftype]
                row = {
                    'family': family,
                    'feature_type': ftype,
                    'n_features': ent['n_features'],
                    'r_mean': ent['r_mean'], 'r_std': ent['r_std'],
                    'r2_mean': ent['r2_mean'], 'r2_std': ent['r2_std'],
                    'r_w_mean': ent['r_w_mean'], 'r_w_std': ent['r_w_std'],
                    'r2_w_mean': ent['r2_w_mean'], 'r2_w_std': ent['r2_w_std'],
                    'n_folds': OUTER_FOLDS,
                    'n_repeats': OUTER_REPEATS,
                }
                row.update(_fold_cols(ent.get('r_folds', []), 'r_f'))
                row.update(_fold_cols(ent.get('r2_folds', []), 'r2_f'))
                row.update(_fold_cols(ent.get('r_w_folds', []), 'r_w_f'))
                row.update(_fold_cols(ent.get('r2_w_folds', []), 'r2_w_f'))
                compute_species_scores._buf_other.append(row)

        for layer_key, ent in results.get('dnn', {}).items():
            model, layer = (str(layer_key).rsplit('_', 1) + [''])[:2]
            row = {
                'family': family,
                'model': model,
                'layer': layer,
                'depth': float(dnn_depths.get(layer_key, np.nan)),
                'n_features': ent['n_features'],
                'r_mean': ent['r_mean'], 'r_std': ent['r_std'],
                'r2_mean': ent['r2_mean'], 'r2_std': ent['r2_std'],
                'r_w_mean': ent['r_w_mean'], 'r_w_std': ent['r_w_std'],
                'r2_w_mean': ent['r2_w_mean'], 'r2_w_std': ent['r2_w_std'],
                'n_folds': OUTER_FOLDS,
                'n_repeats': OUTER_REPEATS,
            }
            row.update(_fold_cols(ent.get('r_folds', []), 'r_f'))
            row.update(_fold_cols(ent.get('r2_folds', []), 'r2_f'))
            row.update(_fold_cols(ent.get('r_w_folds', []), 'r_w_f'))
            row.update(_fold_cols(ent.get('r2_w_folds', []), 'r2_w_f'))
            compute_species_scores._buf_dnn.append(row)

if __name__ == '__main__':
    compute_species_scores('cca', families=['cross-species','human','monkey'])
    out_root = RESULTS_ROOT / 'cca'
    out_root.mkdir(parents=True, exist_ok=True)
    if hasattr(compute_species_scores, '_buf_other') and compute_species_scores._buf_other:
        pd.DataFrame(compute_species_scores._buf_other).to_csv(out_root / 'other_features_weighted.csv', index=False)
        print(f"[ridge_species_weighted] wrote {out_root/'other_features_weighted.csv'}")
    if hasattr(compute_species_scores, '_buf_dnn') and compute_species_scores._buf_dnn:
        pd.DataFrame(compute_species_scores._buf_dnn).to_csv(out_root / 'dnn_features_weighted.csv', index=False)
        print(f"[ridge_species_weighted] wrote {out_root/'dnn_features_weighted.csv'}")
