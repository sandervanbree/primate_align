#!/usr/bin/env python3
"""
Shared helpers for sNMF utilities (guard, gallery, viz).
"""

import os
import sys
import pickle
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from model.CCA.significance import get_significant_components, load_crossview_results

FAMILIES = ['all', 'human', 'monkey']


def load_srf_results():
    res_fp = Path(config.results_dir) / 'srf_results' / 'srf_all_results.pkl'
    if not res_fp.exists():
        raise FileNotFoundError(f"SRF results not found at {res_fp}")
    with open(res_fp, 'rb') as f:
        data = pickle.load(f)
    return data['results'], data['all_stims']

def load_cca_state():
    fp = Path(config.cache_file)
    if not fp.exists():
        raise FileNotFoundError(f"CCA state not found at {fp}")
    with open(fp, 'rb') as f:
        return pickle.load(f)


def pick_rank(family, result):
    custom = config.hyperparameters.get('srf', {}).get('custom_ranks', {})
    target = custom.get(family, result['best_rank'])
    ranks = list(result['components'].keys())
    if target in ranks:
        return target
    return result['best_rank']


def derive_categories(stims):
    cats = []
    for s in stims:
        cats.append('_'.join(s.split('_')[:-1]) if '_' in s else s)
    return cats


def build_img_paths(stims):
    cats = derive_categories(stims)
    img_root = Path(config.data_dir) / 'things' / 'images'
    return [img_root / cat / f"{stim}.jpg" for stim, cat in zip(stims, cats)]


def guard_views_for_family(family, cca_state):
    crossview = load_crossview_results()
    if family == 'all':
        comps = get_significant_components('all', crossview)
        human = [k for k in comps if k.lower().startswith('human')]
        monkey = [k for k in comps if k.lower().startswith('monkey')]
    else:
        comps_hum = get_significant_components('human', crossview)
        comps_mon = get_significant_components('monkey', crossview)
        comps = {**comps_hum, **comps_mon}
        human = list(comps_hum.keys())
        monkey = list(comps_mon.keys())
    return comps, human, monkey


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def guard_csv_path(family: str) -> Path:
    return Path(__file__).resolve().parent / 'results' / 'guard' / f"guard_{family}.csv"


def load_guard_table(family: str) -> pd.DataFrame:
    path = guard_csv_path(family)
    if not path.exists():
        raise FileNotFoundError(f"Guard CSV missing for {family}: {path}")
    return pd.read_csv(path)
