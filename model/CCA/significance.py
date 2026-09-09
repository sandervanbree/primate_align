import pickle
from pathlib import Path
import sys
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config


_FAMILY_TO_SECTION = {
    "all": "cross-species",
    "cross-species": "cross-species",
    "cross_species": "cross-species",
    "human": "human_only",
    "human-only": "human_only",
    "human_only": "human_only",
    "monkey": "monkey_only",
    "monkey-only": "monkey_only",
    "monkey_only": "monkey_only",
}


def _normalize_family_name(family: str) -> str:
    key = str(family).strip().lower()
    if key not in _FAMILY_TO_SECTION:
        raise KeyError(f"Unknown CCA family: {family}")
    return _FAMILY_TO_SECTION[key]


def load_crossview_results() -> dict:
    fp = Path(config.results_dir) / "crossview_results.pkl"
    if not fp.exists():
        raise FileNotFoundError(f"Crossview results not found: {fp}")
    with open(fp, "rb") as f:
        return pickle.load(f)


def get_significant_component_indices(family: str, crossview: Optional[dict] = None) -> np.ndarray:
    crossview = load_crossview_results() if crossview is None else crossview
    fam_key = _normalize_family_name(family)
    sig = crossview["significance"][fam_key]
    return np.asarray(sig["sig_idx"], dtype=int)


def get_significant_component_count(family: str, crossview: Optional[dict] = None) -> int:
    return int(get_significant_component_indices(family, crossview).size)


def get_significant_components(family: str, crossview: Optional[dict] = None) -> dict:
    crossview = load_crossview_results() if crossview is None else crossview
    fam_key = _normalize_family_name(family)
    keep = get_significant_component_indices(family, crossview)
    comps = crossview["components"][fam_key]
    return {view: np.asarray(mat)[:, keep] for view, mat in comps.items()}
