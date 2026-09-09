#!/usr/bin/env python3
"""Decode symbol/icon/ensemble/face/body_part from foldwise CCA projections."""
import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import numpy as np
import pandas as pd
import rcca
import toml
from joblib import Parallel, delayed
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from functions.load import load_macq, load_mri, match_datasets
from model.CCA.significance import get_significant_component_count
from model.feature.core.ridge_utils import get_ridge_params, get_ridge_cv_settings


DECODE_CFG = config.hyperparameters.get('decoding', {})
RESID_CFG = config.hyperparameters.get('residualized_decoding', {})
RIDGE_CV = get_ridge_cv_settings()
RIDGE_ALPHAS, _ = get_ridge_params()

N_FOLDS = int(DECODE_CFG.get('outer_folds', 5))
N_PERMS = int(DECODE_CFG.get('n_perm', 200))
RANDOM_STATE = int(DECODE_CFG.get('seed', 42))
INNER_FOLDS = int(DECODE_CFG.get('inner_folds', 3))
RANDOM_DRAWS = int(DECODE_CFG.get('random_draws', 100))
RANDOM_POS_FRAC = float(DECODE_CFG.get('random_pos_frac', 0.5))

try:
    N_CC = int(get_significant_component_count('cross-species'))
except Exception:
    N_CC = int(config.hyperparameters.get('crossview', {}).get('n_cc', 90))
CCA_REG = 10 ** float(config.hyperparameters.get('cca_families', {}).get('cross-species', {}).get('reg', 4.0))
PCA_DIM = int(RIDGE_CV.get('pca_max_dim', 250))
DNN_PCA_DIM = int(RESID_CFG.get('pca_pre_dim', PCA_DIM))
DNN_CCA_REG = float(RESID_CFG.get('cca_reg', 1.0))
EARLY_DEPTH_THRESHOLD = float(RESID_CFG.get('early_depth_threshold', 0.25))
RIDGE_INNER_FOLDS = int(RIDGE_CV.get('inner_folds', 3))

FEATURES_CORE = ['symbol', 'icon', 'ensemble', 'face', 'body_part']
RANDOM_FEATURE = 'random'
FEATURES_ALL = FEATURES_CORE + [RANDOM_FEATURE]

OUT_DIR_BASE = Path(config.root_dir) / 'analysis' / 'symbol' / 'results'
RESULT_FILE = 'decode_main.pkl'
SUPP_FILE = 'decode_nulls.pkl'
RES_RESULT_FILE = 'decode_residualized_main.pkl'
RES_SUPP_FILE = 'decode_residualized_nulls.pkl'

ROI_MAP = {
    'it': {'monkey': config.analysis['roi_monkey'], 'human': config.analysis['roi_human']},
    'v1': {'monkey': 'v1', 'human': 'V1'},
    'v4': {'monkey': 'v4', 'human': 'hV4'},
}
HUMAN_VIEWS = ['human_01', 'human_02', 'human_03']
MONKEY_VIEWS = ['monkey_N', 'monkey_F']
SPECIES_VIEWS = {'human': HUMAN_VIEWS, 'monkey': MONKEY_VIEWS}


@dataclass
class ProjectionFold:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_human: np.ndarray
    test_human: np.ndarray
    train_monkey: np.ndarray
    test_monkey: np.ndarray


def category_ids(stims):
    return np.array([str(s).rsplit('_', 1)[0] for s in stims])


def load_annotations(stims):
    ann = pd.read_csv(config.get_annotations_path())
    ann = ann.rename(columns={'numerosity': 'ensemble'})
    ann['stim'] = ann['filename'].astype(str)
    ann = ann.set_index('stim').reindex(stims)
    return {feat: ann[feat].fillna(0).astype(int).to_numpy() for feat in FEATURES_CORE}


def load_region_views(region):
    region = region.lower()
    if region not in ROI_MAP:
        raise ValueError(f'Unknown region: {region}')
    roi = ROI_MAP[region]
    raw = {}
    for monkey in config.analysis['monkey_ids']:
        X, stims, _ = load_macq(
            monkey=monkey,
            roi=roi['monkey'],
            min_reliab=config.analysis['default_reliability'],
        )
        raw[f'monkey_{monkey}'] = (X, stims)
    for sub in config.analysis['human_subjects']:
        X, stims, _ = load_mri(
            sub=sub,
            roi=roi['human'],
            min_splithalf=config.analysis['default_split'],
        )
        raw[f'human_{sub}'] = (X, stims)
    matched, common = match_datasets(raw)
    return np.asarray(common), {k: np.asarray(v) for k, v in matched.items()}


def build_human_groups(view_name, stims):
    sid = view_name.split('_')[-1]
    meta_p = config.mri_dir / 'betas_csv' / f'sub-{sid}_StimulusMetadata.csv'
    if not meta_p.exists():
        return None
    df = pd.read_csv(meta_p)
    cols = {c.lower(): c for c in df.columns}
    if {'stimulus', 'session', 'run'} - set(cols):
        return None
    df['stim'] = df[cols['stimulus']].astype(str).str.replace('.jpg', '', regex=False)
    pos = {s: i for i, s in enumerate(df['stim'])}
    order = np.array([pos.get(s, -1) for s in stims], dtype=int)
    labels = np.zeros(len(stims), dtype=int)
    ok = order >= 0
    if ok.any():
        session = df[cols['session']].to_numpy()
        run = df[cols['run']].to_numpy()
        labels[ok] = (session * 100 + run).astype(int)[order[ok]]
    return labels


def group_demean_train_test(X, train_idx, test_idx, groups):
    Xtr = np.asarray(X[train_idx], dtype=np.float32)
    Xte = np.asarray(X[test_idx], dtype=np.float32)
    if groups is None:
        return Xtr, Xte
    gtr = np.asarray(groups)[train_idx]
    gte = np.asarray(groups)[test_idx]
    global_mean = Xtr.mean(axis=0, keepdims=True)
    means = {}
    for val in np.unique(gtr):
        mask = gtr == val
        means[val] = Xtr[mask].mean(axis=0, keepdims=True)
        Xtr[mask] = Xtr[mask] - means[val]
    for val in np.unique(gte):
        mask = gte == val
        Xte[mask] = Xte[mask] - means.get(val, global_mean)
    return Xtr, Xte


def preprocess_train_test(X, train_idx, test_idx, pca_dim=PCA_DIM, seed=RANDOM_STATE, groups=None):
    Xtr, Xte = group_demean_train_test(X, train_idx, test_idx, groups)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    if pca_dim and Xtr.shape[1] > pca_dim:
        k = int(min(pca_dim, Xtr.shape[1], Xtr.shape[0] - 1))
        pca = PCA(n_components=k, svd_solver='randomized', random_state=seed)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)
        pc_scaler = StandardScaler().fit(Xtr)
        Xtr = pc_scaler.transform(Xtr)
        Xte = pc_scaler.transform(Xte)
    return Xtr.astype(np.float32, copy=False), Xte.astype(np.float32, copy=False)


def make_splits(y, groups, base_splits=N_FOLDS):
    y = np.asarray(y, dtype=int)
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return []
    n_splits = min(base_splits, pos, neg)
    if n_splits < 2:
        return []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    idx = np.arange(len(y))
    return list(cv.split(idx, y, groups))


def shared_splits(labels, groups):
    # The orthography labels have enough positives for the configured fold count.
    # Use one deterministic category-grouped split for paired region/species contrasts.
    return make_splits(labels['symbol'], groups)


def fit_neural_fold(region, fold_id, train_idx, test_idx, views, human_groups, pca_dim=PCA_DIM):
    view_names = [v for v in sorted(views) if v in set(HUMAN_VIEWS + MONKEY_VIEWS)]
    train_views, test_views = [], []
    for i, view in enumerate(view_names):
        groups = human_groups.get(view)
        Xtr, Xte = preprocess_train_test(
            views[view], train_idx, test_idx,
            pca_dim=pca_dim,
            seed=RANDOM_STATE + 1000 * fold_id + i,
            groups=groups,
        )
        train_views.append(Xtr)
        test_views.append(Xte)
    cca = rcca.CCA(kernelcca=False, reg=CCA_REG, numCC=N_CC, verbose=False)
    cca.train(train_views)
    train_proj, test_proj = {}, {}
    for i, view in enumerate(view_names):
        ptr = train_views[i] @ cca.ws[i]
        pte = test_views[i] @ cca.ws[i]
        scaler = StandardScaler().fit(ptr)
        train_proj[view] = scaler.transform(ptr).astype(np.float32, copy=False)
        test_proj[view] = scaler.transform(pte).astype(np.float32, copy=False)
    train_h = np.mean([train_proj[v] for v in HUMAN_VIEWS if v in train_proj], axis=0)
    test_h = np.mean([test_proj[v] for v in HUMAN_VIEWS if v in test_proj], axis=0)
    train_m = np.mean([train_proj[v] for v in MONKEY_VIEWS if v in train_proj], axis=0)
    test_m = np.mean([test_proj[v] for v in MONKEY_VIEWS if v in test_proj], axis=0)
    return ProjectionFold(fold_id, np.asarray(train_idx), np.asarray(test_idx), train_h, test_h, train_m, test_m)


def fit_neural_projections(region, stims, views, folds, jobs=1, pca_dim=PCA_DIM):
    human_groups = {
        view: build_human_groups(view, stims)
        for view in views
        if view.startswith('human_') and config.analysis.get('demean_groups', True)
    }
    tasks = [
        (region, fold_id, tr, te, views, human_groups, pca_dim)
        for fold_id, (tr, te) in enumerate(folds)
    ]
    print(f'  {region.upper()}: fitting foldwise neural MCCA ({len(tasks)} folds, reg={CCA_REG:g}, n_cc={N_CC}, pca={pca_dim})')
    return Parallel(n_jobs=min(max(1, jobs), len(tasks)), verbose=10)(
        delayed(fit_neural_fold)(*task) for task in tasks
    )


def decode_auc(Xtr, Xte, ytr, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return np.nan
    clf = LogisticRegressionCV(
        cv=INNER_FOLDS,
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight='balanced',
        scoring='roc_auc',
    )
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.decision_function(Xte)))


def decode_feature_from_folds(folds, y):
    out = {'human': [], 'monkey': []}
    for fp in folds:
        for species in ('human', 'monkey'):
            Xtr = fp.train_human if species == 'human' else fp.train_monkey
            Xte = fp.test_human if species == 'human' else fp.test_monkey
            out[species].append(decode_auc(Xtr, Xte, y[fp.train_idx], y[fp.test_idx]))
    return out


def permutation_test(folds, labels, n_perms=N_PERMS, random_state=RANDOM_STATE, jobs=1):
    if n_perms <= 0:
        return {species: {feat: np.array([]) for feat in labels} for species in ('human', 'monkey')}

    seeds = [random_state + 100_000 + i for i in range(n_perms)]

    def run_perm(seed):
        rng = np.random.default_rng(seed)
        rows = []
        for feat, y0 in labels.items():
            y = rng.permutation(y0)
            aucs = decode_feature_from_folds(folds, y)
            for species in ('human', 'monkey'):
                rows.append((species, feat, float(np.nanmean(aucs[species]))))
        return rows

    chunks = Parallel(n_jobs=min(max(1, jobs), n_perms), verbose=10)(
        delayed(run_perm)(seed) for seed in seeds
    )
    nulls = {species: {feat: [] for feat in labels} for species in ('human', 'monkey')}
    for chunk in chunks:
        for species, feat, auc in chunk:
            if np.isfinite(auc):
                nulls[species][feat].append(auc)
    return {
        species: {feat: np.asarray(vals, dtype=float) for feat, vals in feats.items()}
        for species, feats in nulls.items()
    }


def summarize_feature_aucs(aucs, nulls=None):
    stats_by_species = {'human': {}, 'monkey': {}}
    for species in ('human', 'monkey'):
        for feat, vals in aucs[species].items():
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mean_auc = float(np.mean(vals))
            sem_auc = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            null_means = np.array([]) if nulls is None else nulls.get(species, {}).get(feat, np.array([]))
            p_val = np.nan
            if null_means.size:
                p_val = float((np.sum(null_means >= mean_auc) + 1) / (len(null_means) + 1))
            stats_by_species[species][feat] = {
                'fold_aucs': vals.tolist(),
                'mean_auc': mean_auc,
                'sem_auc': sem_auc,
                'null_means': null_means,
                'p_val': p_val,
                'n_folds': int(vals.size),
            }
    return stats_by_species


def generate_random_draws(n_samples, groups, n_draws=RANDOM_DRAWS, pos_frac=RANDOM_POS_FRAC, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    draws = []
    idx = np.arange(n_samples)
    pos_count = max(1, min(n_samples - 1, int(round(n_samples * pos_frac))))
    for _ in range(n_draws):
        y = np.zeros(n_samples, dtype=int)
        pos_idx = rng.choice(idx, size=pos_count, replace=False)
        y[pos_idx] = 1
        draws.append({'y': y, 'pos_idx': pos_idx})
    return draws


def random_baseline_species(folds, random_draws):
    out = {}
    for species in ('human', 'monkey'):
        fold_values, draw_means, draw_stats, draw_fold_aucs = [], [], [], []
        for draw in random_draws:
            y = draw['y']
            vals = []
            for fp in folds:
                Xtr = fp.train_human if species == 'human' else fp.train_monkey
                Xte = fp.test_human if species == 'human' else fp.test_monkey
                vals.append(decode_auc(Xtr, Xte, y[fp.train_idx], y[fp.test_idx]))
            vals = [float(v) for v in vals if np.isfinite(v)]
            if not vals:
                continue
            fold_values.extend(vals)
            draw_mean = float(np.mean(vals))
            draw_means.append(draw_mean)
            draw_stats.append({'mean_auc': draw_mean, 'n_folds': len(vals)})
            draw_fold_aucs.append(vals)
        arr = np.asarray(draw_means, dtype=float)
        if arr.size:
            out[species] = {
                'fold_aucs': [float(v) for v in fold_values],
                'mean_auc': float(arr.mean()),
                'sem_auc': float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
                'null_means': np.array([]),
                'p_val': np.nan,
                'n_draws': int(arr.size),
                'draw_stats': draw_stats,
                'draw_means': arr.tolist(),
                'draw_fold_aucs': draw_fold_aucs,
                'base_mean_for_perm': float(arr.mean()),
            }
    return out


def paired_species_stats(per_species):
    paired = {}
    feats = sorted(set(per_species.get('human', {})).intersection(per_species.get('monkey', {})))
    for feat in feats:
        if feat == RANDOM_FEATURE:
            auc_h = per_species['human'][feat].get('draw_means', [])
            auc_m = per_species['monkey'][feat].get('draw_means', [])
        else:
            auc_h = per_species['human'][feat]['fold_aucs']
            auc_m = per_species['monkey'][feat]['fold_aucs']
        n = min(len(auc_h), len(auc_m))
        if n < 2:
            continue
        auc_h = np.asarray(auc_h[:n], dtype=float)
        auc_m = np.asarray(auc_m[:n], dtype=float)
        t_val, p_val = stats.ttest_rel(auc_m, auc_h, nan_policy='omit')
        paired[feat] = {
            't_val': float(t_val),
            'p_val': float(p_val),
            'mean_diff': float(np.nanmean(auc_m - auc_h)),
            'n_pairs': int(np.isfinite(auc_m - auc_h).sum()),
        }
    return paired


def decode_from_projections(folds, labels, groups, jobs=1, run_permutation=True):
    aucs = {'human': {}, 'monkey': {}}
    for feat, y in labels.items():
        vals = decode_feature_from_folds(folds, y)
        for species in ('human', 'monkey'):
            aucs[species][feat] = vals[species]
    nulls = permutation_test(folds, labels, jobs=jobs) if run_permutation else None
    per_species = summarize_feature_aucs(aucs, nulls)

    random_draws = generate_random_draws(len(groups), groups)
    rand = random_baseline_species(folds, random_draws)
    for species in ('human', 'monkey'):
        if species in rand:
            per_species[species][RANDOM_FEATURE] = rand[species]
    return {
        'per_species': per_species,
        'paired': paired_species_stats(per_species),
        'random_draws': [{'pos_idx': d['pos_idx'].tolist()} for d in random_draws],
    }


def load_early_dnn_per_model(stims):
    base = Path(config.dnn_dir) / 'features'
    tsv = base / 'features_summary.tsv'
    if not tsv.exists():
        raise FileNotFoundError(f'DNN feature summary not found: {tsv}')
    df = pd.read_csv(tsv, sep='\t')
    layer_cfg = toml.load(Path(config.root_dir) / 'config' / 'dnn_layers.toml')
    allowed = {}
    for ent in layer_cfg.get('model', []):
        name = str(ent.get('name', '')).strip().lower()
        layers = {str(x) for x in ent.get('layers', [])}
        if name:
            allowed[name] = layers
    df = df[df.apply(lambda r: str(r['layer_name']) in allowed.get(str(r['model']).strip().lower(), set()), axis=1)].copy()
    df['depth_norm'] = df.groupby('model')['layer_seq'].transform(lambda s: s / s.max())
    df = df[df['depth_norm'] <= EARLY_DEPTH_THRESHOLD]
    feats = {}
    for model in sorted(df['model'].unique()):
        blocks = []
        for _, row in df[df['model'] == model].sort_values('layer_seq').iterrows():
            p = Path(config.dnn_dir) / str(row['save_path']).strip()
            p = p.parent / 'features' / 'features.npy'
            x = np.load(p, mmap_mode='r')
            x = np.asarray(x, dtype=np.float32)
            if x.ndim > 2:
                x = x.reshape((x.shape[0], int(np.prod(x.shape[1:]))))
            blocks.append(x)
        feats[model] = np.hstack(blocks).astype(np.float32, copy=False)
        print(f'  Early DNN {model}: {feats[model].shape}')
    n = len(stims)
    for model, X in feats.items():
        if X.shape[0] < n:
            raise ValueError(f'DNN features for {model} have {X.shape[0]} rows, expected at least {n}')
        feats[model] = X[:n]
    return feats


def fit_dnn_control_fold(fold_id, train_idx, test_idx, dnn_feats):
    models = sorted(dnn_feats)
    train_views, test_views = [], []
    for i, model in enumerate(models):
        Xtr, Xte = preprocess_train_test(
            dnn_feats[model], train_idx, test_idx,
            pca_dim=DNN_PCA_DIM,
            seed=RANDOM_STATE + 10_000 + 100 * fold_id + i,
            groups=None,
        )
        train_views.append(Xtr)
        test_views.append(Xte)
    cca = rcca.CCA(kernelcca=False, reg=DNN_CCA_REG, numCC=N_CC, verbose=False)
    cca.train(train_views)
    train_proj, test_proj = [], []
    for i in range(len(models)):
        ptr = train_views[i] @ cca.ws[i]
        pte = test_views[i] @ cca.ws[i]
        scaler = StandardScaler().fit(ptr)
        train_proj.append(scaler.transform(ptr).astype(np.float32, copy=False))
        test_proj.append(scaler.transform(pte).astype(np.float32, copy=False))
    return np.mean(train_proj, axis=0), np.mean(test_proj, axis=0)


def fit_dnn_controls(stims, folds, jobs=1):
    dnn_feats = load_early_dnn_per_model(stims)
    tasks = [(fold_id, tr, te, dnn_feats) for fold_id, (tr, te) in enumerate(folds)]
    print(f'  IT: fitting foldwise early-DNN controls ({len(tasks)} folds, reg={DNN_CCA_REG:g}, pca={DNN_PCA_DIM})')
    return Parallel(n_jobs=min(max(1, jobs), len(tasks)), verbose=10)(
        delayed(fit_dnn_control_fold)(*task) for task in tasks
    )


def residualize_projection_fold(fp, ctrl_train, ctrl_test):
    def resid(Xtr, Xte):
        scaler = StandardScaler().fit(ctrl_train)
        Ctr = scaler.transform(ctrl_train)
        Cte = scaler.transform(ctrl_test)
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=max(2, RIDGE_INNER_FOLDS))
        ridge.fit(Ctr, Xtr)
        return (
            (Xtr - ridge.predict(Ctr)).astype(np.float32, copy=False),
            (Xte - ridge.predict(Cte)).astype(np.float32, copy=False),
        )

    train_h, test_h = resid(fp.train_human, fp.test_human)
    train_m, test_m = resid(fp.train_monkey, fp.test_monkey)
    return ProjectionFold(fp.fold, fp.train_idx, fp.test_idx, train_h, test_h, train_m, test_m)


def make_residualized_folds(neural_folds, controls):
    return [
        residualize_projection_fold(fp, ctrl_train, ctrl_test)
        for fp, (ctrl_train, ctrl_test) in zip(neural_folds, controls)
    ]


def save_region_results(region, results):
    out_dir = OUT_DIR_BASE / region
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'region': region,
        'results': results,
        'state_path': f'foldwise:{region}',
        'random_draws': results.get('random_draws', []),
    }
    with open(out_dir / RESULT_FILE, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    supp_payload = {
        'region': region,
        'per_species': results['per_species'],
        'state_path': f'foldwise:{region}',
        'random_draws': results.get('random_draws', []),
    }
    with open(out_dir / SUPP_FILE, 'wb') as f:
        pickle.dump(supp_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Saved {region.upper()} decoding results to {out_dir / RESULT_FILE}')


def save_residualized_results(results):
    OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR_BASE / RES_RESULT_FILE, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    supp_payload = {
        'per_species': results['per_species'],
        'random_draws': results.get('random_draws', []),
    }
    with open(OUT_DIR_BASE / RES_SUPP_FILE, 'wb') as f:
        pickle.dump(supp_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Saved residualized decoding results to {OUT_DIR_BASE / RES_RESULT_FILE}')


def run_region(region, jobs=1, run_main=True, run_residualized=False, run_permutation=True):
    stims, views = load_region_views(region)
    labels = load_annotations(stims)
    groups = category_ids(stims)
    folds = shared_splits(labels, groups)
    if not folds:
        raise RuntimeError(f'No valid category-held-out folds for {region}')

    neural_folds = fit_neural_projections(region, stims, views, folds, jobs=jobs)
    main_results = None
    if run_main:
        print(f'  {region.upper()}: decoding annotations')
        main_results = decode_from_projections(neural_folds, labels, groups, jobs=jobs, run_permutation=run_permutation)
        save_region_results(region, main_results)

    residualized_results = None
    if run_residualized:
        if region != 'it':
            raise ValueError('Residualized decoding is currently defined for IT only.')
        controls = fit_dnn_controls(stims, folds, jobs=jobs)
        residual_folds = make_residualized_folds(neural_folds, controls)
        print('  IT: decoding annotations after early-DNN residualization')
        res = decode_from_projections(residual_folds, labels, groups, jobs=jobs, run_permutation=run_permutation)
        residualized_results = {
            'per_species': {
                species: {'features': feats, 'n': len(stims)}
                for species, feats in res['per_species'].items()
            },
            'paired': res['paired'],
            'random_draws': res.get('random_draws', []),
        }
        save_residualized_results(residualized_results)

    return main_results, residualized_results


def run_pipeline(regions=('it',), jobs=None, run_main=True, run_residualized=False, run_permutation=True):
    OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)
    if jobs is None:
        jobs = int(config.analysis.get('n_jobs', 1))
    results = {}
    for region in regions:
        print('\n' + '=' * 72)
        print(f'Orthography decoding: {region.upper()}')
        print('=' * 72)
        results[region] = run_region(
            region,
            jobs=jobs,
            run_main=run_main,
            run_residualized=(run_residualized and region == 'it'),
            run_permutation=run_permutation,
        )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regions', nargs='+', default=['it'], choices=sorted(ROI_MAP))
    parser.add_argument('--jobs', type=int, default=int(config.analysis.get('n_jobs', 1)))
    parser.add_argument('--residualized', action='store_true', help='Also run IT early-DNN residualized decoding.')
    parser.add_argument('--skip-main', action='store_true', help='Only write residualized output.')
    parser.add_argument('--no-permutation', action='store_true', help='Skip label-permutation nulls.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_pipeline(
        regions=args.regions,
        jobs=args.jobs,
        run_main=not args.skip_main,
        run_residualized=args.residualized,
        run_permutation=not args.no_permutation,
    )


if __name__ == '__main__':
    main()
