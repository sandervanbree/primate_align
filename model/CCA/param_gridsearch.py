# Computation only - hyperparameter grid search for CCA
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import numpy as np, pandas as pd
import sys, warnings, pickle
import rcca
import time
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from functions.load import load_mri, load_macq, match_datasets
from config.paths import config
from functions.cca import cca_cv, cca_nested_cv
import pandas as pd

warnings.filterwarnings('ignore')

# === Parameters Loaded from Config ===
HPARAMS = config.hyperparameters['gridsearch']
APARAMS = config.analysis
EPARAMS = config.hyperparameters.get('eval_ridge_mcca', {})

NUM_COMPONENTS = HPARAMS['n_components']
NUM_JOBS = APARAMS['n_jobs']
K_FOLDS_INNER = int(EPARAMS.get('inner_folds', HPARAMS['k_folds_inner']))
K_FOLDS_OUTER = int(EPARAMS.get('outer_folds', HPARAMS['k_folds_outer']))
NUM_REPS = int(EPARAMS.get('outer_repeats', HPARAMS['n_reps']))
RNG_SEED = int(EPARAMS.get('seed', 42))
np.random.seed(RNG_SEED)

# Generate regularization parameters from config values
REG_PARAMS = np.logspace(
    HPARAMS['reg_start'], HPARAMS['reg_stop'], HPARAMS['reg_num']
)

DEFAULT_RELIABILITY = APARAMS['default_reliability']
DEFAULT_SPLIT = APARAMS['default_split']

# Load and match all datasets
print("Loading datasets...")
raw_data = {}
for monkey_id in APARAMS['monkey_ids']:
    print(f"  Loading monkey {monkey_id}...")
    arr, stim, _ = load_macq(
        monkey=monkey_id,
        min_reliab=DEFAULT_RELIABILITY,
        roi=APARAMS['roi_monkey']
    )
    raw_data[f'monkey_{monkey_id}'] = (arr, stim)

for human_id in APARAMS['human_subjects']:
    print(f"  Loading human {human_id}...")
    arr, stim, _ = load_mri(
        sub=human_id,
        roi=APARAMS['roi_human'],
        min_splithalf=DEFAULT_SPLIT
    )
    raw_data[f'human_{human_id}'] = (arr, stim)

data_all, stims = match_datasets(raw_data)
print(f"Loaded {len(data_all)} views with {data_all[list(data_all.keys())[0]].shape[0]} stimuli")

# Define view sets for different analyses
view_sets = {
    'cross-species': data_all,
    'human_only': {k: v for k,v in data_all.items() if 'human' in k},
    'monkey_only': {k: v for k,v in data_all.items() if 'monkey' in k},
}

# Start timing
start_time = time.time()
print(f"\nStarting gridsearch at {datetime.now().strftime('%H:%M:%S')}")
print(f"Total tasks: 3 families × (1 nested CV + {len(REG_PARAMS)} reg sweeps) = {3 * (1 + len(REG_PARAMS))}")
print(f"Parallelization strategy:")
print(f"  - Nested CV: 3 families in parallel, each using {NUM_JOBS // 3} cores")
print(f"  - Reg sweep: {len(REG_PARAMS)} params × 3 families = {len(REG_PARAMS) * 3} tasks using {NUM_JOBS} cores")
print(f"  - Expected core utilization: ~{min(80, 3 * (1 + len(REG_PARAMS)))} cores")
print(f"Eval profile: outer_folds={K_FOLDS_OUTER}, outer_repeats={NUM_REPS}, inner_folds={K_FOLDS_INNER}, seed={RNG_SEED}")

# --- 1. Get Unbiased Performance Estimates via Nested CV ---
print("\n--- Computing Nested CV for all families in parallel ---")

def _group_demean_np(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float).copy()
    if g is None or g.size == 0:
        return X
    for val in np.unique(g):
        mask = (g == val)
        if mask.any():
            X[mask] -= X[mask].mean(axis=0, keepdims=True)
    return X


def _build_groups_for_views(views, stims):
    groups = {}
    for v in views:
        if v.startswith('human_'):
            sid = v.split('_')[-1]
            meta_p = os.path.join(config.mri_dir, 'betas_csv', f'sub-{sid}_StimulusMetadata.csv')
            if not os.path.exists(meta_p):
                continue
            df = pd.read_csv(meta_p)
            cols = {c.lower(): c for c in df.columns}
            if not all(k in cols for k in ('stimulus','session','run')):
                continue
            df['stim'] = df[cols['stimulus']].astype(str).str.replace('.jpg','', regex=False)
            pos = {s:i for i,s in enumerate(df['stim'])}
            order = [pos.get(s, -1) for s in stims]
            labels = np.zeros(len(stims), dtype=int)
            ok = np.array(order) >= 0
            if ok.any():
                session = df[cols['session']].to_numpy()
                run     = df[cols['run']].to_numpy()
                codes   = (session * 100 + run).astype(int)
                labels[ok] = codes[np.array(order)[ok]]
            groups[v] = labels
    return groups


def run_nested_cv_for_family(family_name, views):
    """Run nested CV for a single family"""
    print(f"Starting nested CV for {family_name}...")
    family_start = time.time()

    # Pre-demean views if enabled in config
    if APARAMS.get('demean_groups', True):
        groups = _build_groups_for_views(views, stims)
        views = {k: _group_demean_np(v, groups.get(k, None)) for k, v in views.items()}
    results = cca_nested_cv(
        views, reg_params=REG_PARAMS, n_cc=NUM_COMPONENTS,
        k_outer=K_FOLDS_OUTER, k_inner=K_FOLDS_INNER,
        n_reps=NUM_REPS, n_jobs=max(1, NUM_JOBS // 3), seed=RNG_SEED  # Divide cores among families
    )
    results['family'] = family_name

    family_duration = time.time() - family_start
    print(f"Completed {family_name}: R = {results['mean_corr']:.3f} +/- {results['std_corr']:.3f} ({family_duration:.1f}s)")
    return results

# Run all families in parallel
nested_cv_results = Parallel(n_jobs=3)(
    delayed(run_nested_cv_for_family)(family, views)
    for family, views in view_sets.items()
)

df_nested = pd.DataFrame(nested_cv_results)

# --- 2. Find Optimal Hyperparameters on Full Dataset for Final Model ---
print("\n--- Finding optimal hyperparameters on full dataset ---")

# Parallelize the regularization sweep across all family×reg combinations
def evaluate_reg_param(family_name, views, reg):
    """Evaluate one regularization parameter for one family"""
    if APARAMS.get('demean_groups', True):
        groups = _build_groups_for_views(views, stims)
        views = {k: _group_demean_np(v, groups.get(k, None)) for k, v in views.items()}
    cv_result, _ = cca_cv(
        views, reg=reg, n_cc=NUM_COMPONENTS,
        n_folds=K_FOLDS_INNER, n_reps=NUM_REPS, n_jobs=1, random_state=RNG_SEED
    )
    return family_name, reg, cv_result['mean_corr']

# Create all family×reg combinations
all_combinations = []
for family, views in view_sets.items():
    for reg in REG_PARAMS:
        all_combinations.append((family, views, reg))

print(f"Running {len(all_combinations)} family×reg combinations in parallel...")
print(f"Using {NUM_JOBS} cores for {len(all_combinations)} tasks")
reg_start_time = time.time()

# Run all combinations in parallel - use all cores since this is the main bottleneck
all_results = Parallel(n_jobs=NUM_JOBS)(
    delayed(evaluate_reg_param)(family, views, reg)
    for family, views, reg in all_combinations
)

reg_duration = time.time() - reg_start_time
print(f"All reg sweeps completed in {reg_duration:.1f} seconds")

# Organize results by family
print("\nOrganizing results...")
best_params = {}
reg_sweep_results = {}

for family in view_sets.keys():
    family_results = [(reg, corr) for fname, reg, corr in all_results if fname == family]
    family_results.sort()  # Sort by reg

    df_reg = pd.DataFrame(family_results, columns=['regularization', 'crossview_r'])
    best_reg = df_reg.loc[df_reg['crossview_r'].idxmax(), 'regularization']
    best_corr = df_reg['crossview_r'].max()

    best_params[family] = {'reg': best_reg}
    reg_sweep_results[family] = df_reg

    print(f"Best regularization for {family}: {best_reg:.2e} (CV R = {best_corr:.3f})")

# Save all results for visualization
print("\nSaving results...")

# Ensure results directory exists
params_path = config.results_dir / 'cca_params.pkl'
params_path.parent.mkdir(parents=True, exist_ok=True)

results_to_save = {
    'nested_cv_results': df_nested,
    'reg_sweep_results': reg_sweep_results,
    'best_params': best_params,
    'reg_params': REG_PARAMS,
    'view_sets': list(view_sets.keys()),
    'config': {
        'num_components': NUM_COMPONENTS,
        'k_folds_inner': K_FOLDS_INNER,
        'k_folds_outer': K_FOLDS_OUTER,
        'num_reps': NUM_REPS
    }
}

with open(params_path, 'wb') as f:
    pickle.dump(results_to_save, f)

total_duration = time.time() - start_time
print(f"\n=== FINAL SUMMARY ===")
print(f"Total runtime: {timedelta(seconds=int(total_duration))}")
print(f"Completed at: {datetime.now().strftime('%H:%M:%S')}")
print(f"Saved results to {params_path}")
