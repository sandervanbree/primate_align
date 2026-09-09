import os, sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.paths import config
from model.CCA.significance import load_crossview_results, get_significant_component_indices


# --- basics ---
def _cache() -> dict:
    with open(config.cache_file, 'rb') as f:
        return pickle.load(f)

def load_stims() -> Tuple[List[str], List[str]]:
    st = _cache()
    stims = st.get('all_stims')
    if stims is None: raise FileNotFoundError('all_stims missing in cache_file')
    cats = [_stim_to_cat(s).lower() for s in stims]
    return stims, cats

def _stim_to_cat(stim: str) -> str:
    base = os.path.basename(stim)
    stem = os.path.splitext(base)[0]
    # Category is everything before the last underscore
    return stem.rsplit('_', 1)[0] if '_' in stem else stem


# --- visual features (from vis_props.tsv) ---
VIS_COLS = ['contrast','colorfulness','spatial_freq','texture','spiky_stubby','curvature','size']

def load_vis(stims: List[str], use_proc: bool = True) -> Tuple[np.ndarray, List[str], List[str]]:
    fp = Path(config.data_dir)/'features'/'vis_props'/'vis_props.tsv'
    if not fp.exists():
        raise FileNotFoundError(f'visual features TSV not found: {fp}')
    df = pd.read_csv(fp, sep='\t')
    df = df.set_index('image')
    if use_proc:
        want = [f'{c}_proc' for c in VIS_COLS]
        cols = [c for c in want if c in df.columns]
        disp = [c.replace('_proc','') for c in cols]
    else:
        cols = [c for c in VIS_COLS if c in df.columns]
        disp = cols
    Xdf = df.reindex(stims).loc[:, cols]
    v = Xdf.values.astype(np.float32)
    if np.isnan(v).any():
        # Handle column-wise imputation to avoid all-NaN warnings
        for col_idx in range(v.shape[1]):
            col_data = v[:, col_idx]
            if np.all(np.isnan(col_data)):
                # All-NaN column: fill with 0
                v[:, col_idx] = 0.0
            elif np.any(np.isnan(col_data)):
                # Partial NaN: impute with median
                col_median = np.nanmedian(col_data)
                if not np.isnan(col_median):
                    v[np.isnan(col_data), col_idx] = col_median
                else:
                    # Median is NaN, fill with 0
                    v[np.isnan(col_data), col_idx] = 0.0
    X = v
    names = disp
    groups = ['visual']*len(cols)
    return X, names, groups


# --- behavioral: SPoSE 66d embedding ---
def load_beh(stims: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    root = Path(config.data_dir)/'things'/'behav_embed'
    uid_p = root/'variables'/'unique_id.txt'
    emb_p = root/'data'/'spose_embedding_66d_sorted.txt'
    labels_p = root/'variables'/'labels_short.txt'
    if not uid_p.exists() or not emb_p.exists():
        return np.zeros((len(stims),0)), [], []
    # Normalize categories to match unique_id.txt (lowercase, underscores)
    cats = [_stim_to_cat(s).strip().lower() for s in stims]
    with open(uid_p,'r') as f:
        uids=[ln.strip().lower() for ln in f if ln.strip()]
    M = np.loadtxt(emb_p)
    idx = {u:i for i,u in enumerate(uids)}
    X = np.zeros((len(stims), M.shape[1]), float)
    fill = M.mean(0)
    for i,c in enumerate(cats):
        j = idx.get(c)
        X[i] = M[j] if j is not None else fill
    # Load actual behavioral dimension names and crop
    if labels_p.exists():
        with open(labels_p,'r') as f:
            raw_names = [ln.strip() for ln in f if ln.strip()]
        names = []
        for name in raw_names[:X.shape[1]]:
            # Find position of first "/" or "-"
            sep_pos = len(name)  # Default to end of string
            for sep in ['/', '-']:
                pos = name.find(sep)
                if pos != -1 and pos < sep_pos:
                    sep_pos = pos

            if sep_pos < len(name):
                cropped = name[:sep_pos].strip()
            else:
                cropped = name
            names.append(cropped.replace('_', ' '))
    else:
        names=[f'spose_{k+1}' for k in range(X.shape[1])]
    groups=['behavioral']*len(names)
    return X, names, groups

# --- semantic: category attributes from categories TSV ---
def load_sem(stims: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    # try config path then fixedUniqueID variant
    ctsv = getattr(config, 'categories_tsv', None)
    cand = [Path(ctsv) if ctsv else None, Path(config.data_dir)/'things'/'Categories_final_20200131_fixedUniqueID.tsv']
    fp = next((p for p in cand if p and p.exists()), None)
    if fp is None:
        return np.zeros((len(stims),0)), [], []
    df = pd.read_table(fp)
    # prefer uniqueID as key; fallback to Word
    key = 'uniqueID' if 'uniqueID' in df.columns else 'Word'
    df[key]=df[key].astype(str).str.strip().str.lower().str.replace(' ','_')
    # semantic columns: exclude identifiers and definition columns
    cols=[c for c in df.columns if c not in ('uniqueID','Word') and not str(c).lower().startswith('definition')]
    B = (df[cols].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0).values>0).astype(int)
    idx = {df[key].iloc[i]:i for i in range(len(df))}
    cats = [_stim_to_cat(s).lower() for s in stims]
    X = np.zeros((len(stims), len(cols)), int)
    miss=0
    for i,c in enumerate(cats):
        j=idx.get(c)
        if j is None: miss+=1
        else: X[i]=B[j]
    names=[str(c) for c in cols]
    groups=['semantic']*len(names)
    return X, names, groups


# --- CCA/SRF components ---
def load_cca_components(family: str) -> Tuple[np.ndarray, List[str]]:
    st = _cache()
    crossview = load_crossview_results()
    fam_key = {'all': 'cross-species', 'human': 'human', 'monkey': 'monkey'}.get(family, 'cross-species')
    keep = get_significant_component_indices(fam_key, crossview)
    section = {'all': 'cross-species', 'human': 'human_only', 'monkey': 'monkey_only'}[family if family in ('all', 'human', 'monkey') else 'all']
    comps_by_view = crossview['components'][section]
    comps = np.mean([np.asarray(comps_by_view[v])[:, keep] for v in sorted(comps_by_view)], axis=0)
    return comps, st['all_stims']

def load_srf_components(family: str) -> Tuple[np.ndarray, List[str]]:
    p = Path(config.results_dir)/'srf_results'/'srf_all_results.pkl'
    with open(p,'rb') as f: d=pickle.load(f)
    ent = d['results'][family]
    br = ent['best_rank']
    comps = np.asarray(ent['components'][br])
    return comps, d['all_stims']

# Convenience aliases
load_cca_comps = load_cca_components
load_srf_comps = load_srf_components


# --- DNN feature IO ---
def dnn_root() -> Path:
    a = Path(config.data_dir)/'features'/'dnn'
    if a.exists(): return a
    b = Path(config.dnn_dir) if hasattr(config, 'dnn_dir') else None
    return a if b is None else Path(b)

def load_dnn_layers(models: Dict[str, List[str]], n: int) -> Dict[str, Dict[str, np.ndarray]]:
    root = dnn_root(); out={}
    for m, layers in models.items():
        md = {}
        for ly in layers:
            for cand in [root/m/f'{ly}.pkl', root/m/f'{ly.replace(".","_")}.npy', root/m/f'{ly}.npy']:
                if Path(cand).exists():
                    try:
                        if str(cand).endswith('.pkl'):
                            with open(cand,'rb') as f: d=pickle.load(f)
                            X = np.asarray(d.get('features') or d.get('activations') or d.get('X') or d)
                        else:
                            X = np.load(cand, mmap_mode='r')
                        if X.shape[0] >= n:
                            md[ly] = X[:n]
                        break
                    except Exception:
                        pass
        if md: out[m]=md
    return out
