# Computation only - CCA analysis for different species families
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from functions.load import load_mri, load_macq, match_datasets
from config.paths import config
from functions.cca import fit_cca

np.random.seed(42)

# Parameters from config.toml
PARAMS = config.analysis
CCA_PARAMS = config.hyperparameters['cca_families']
CROSSVIEW_CFG = config.hyperparameters['crossview']
FIT_N_CC = int(CROSSVIEW_CFG['n_cc'])

# Convert regularization exponents back to 10**x values
for k, v in CCA_PARAMS.items():
    CCA_PARAMS[k]['reg'] = 10**v['reg']

print("Computing CCA results from scratch...")

# Load datasets using reusable loaders from functions.load, then match
data_all = {}

# Load monkey data
for m in PARAMS['monkey_ids']:
    X, st, _ = load_macq(
        monkey=m,
        roi=PARAMS['roi_monkey'],
        min_reliab=PARAMS['default_reliability']
    )
    data_all[f'monkey_{m}'] = (X, st)
    print(f"Loaded monkey_{m}: {X.shape}")

# Load human data
for h in PARAMS['human_subjects']:
    X, st, _ = load_mri(
        sub=h,
        roi=PARAMS['roi_human'],
        min_splithalf=PARAMS['default_split']
    )
    data_all[f'human_{h}'] = (X, st)
    print(f"Loaded human_{h}: {X.shape}")

# Match datasets
all_X, all_stims = match_datasets(data_all)
print(f"Common stimuli (all): {len(all_stims)}")

X_all = all_X
X_monkey = {k: v for k, v in all_X.items() if 'monkey' in k}
X_human = {k: v for k, v in all_X.items() if 'human' in k}

# Define CCA computation function for parallelization
def _build_groups_for_human_view(view_name: str, stims: list[str]):
    """Return integer group labels per row for human view: session*100 + run.

    Falls back to ones if metadata is unavailable.
    """
    sid = view_name.split('_')[-1]
    meta_p = os.path.join(config.mri_dir, 'betas_csv', f'sub-{sid}_StimulusMetadata.csv')
    if not os.path.exists(meta_p):
        return None
    df = pd.read_csv(meta_p)
    cols = {c.lower(): c for c in df.columns}
    if 'stimulus' not in cols or 'session' not in cols or 'run' not in cols:
        return None
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
    return labels


def compute_family_cca(family_name, data, params):
    print(f"Running {family_name} CCA...")
    # Optionally demean by (session,run) for human views
    demean = PARAMS.get('demean_groups', True)
    group_labels = {}
    if demean:
        stims = all_stims
        for v in sorted(data.keys()):
            if v.startswith('human_'):
                g = _build_groups_for_human_view(v, stims)
                if g is not None:
                    group_labels[v] = g
    if group_labels:
        return family_name, fit_cca(data, n_cc=FIT_N_CC, group_labels=group_labels, demean_groups=True, **params)
    return family_name, fit_cca(data, n_cc=FIT_N_CC, **params)

# Compute CCA for each dataset combination in parallel
family_tasks = [
    ('cross-species', X_all, CCA_PARAMS['cross-species']),
    ('human-only', X_human, CCA_PARAMS['human']),
    ('monkey-only', X_monkey, CCA_PARAMS['monkey'])
]

results = Parallel(n_jobs=min(3, PARAMS['n_jobs']))(
    delayed(compute_family_cca)(name, data, params)
    for name, data, params in family_tasks
)

# Unpack results
result_dict = dict(results)
cca_all = result_dict['cross-species']
cca_hum = result_dict['human-only']
cca_mon = result_dict['monkey-only']

# Cache results
os.makedirs(config.results_dir, exist_ok=True)

state = {
    'cca_all': cca_all,
    'cca_hum': cca_hum,
    'cca_mon': cca_mon,
    'all_stims': all_stims,
    'X_all': X_all,
    'X_monkey': X_monkey,
    'X_human': X_human
}

state_fp = config.cache_file
with open(state_fp, 'wb') as f:
    pickle.dump(state, f)
print(f"Cached results to {state_fp}")

print("CCA families computation complete!")
