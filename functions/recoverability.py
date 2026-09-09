from __future__ import annotations

import numpy as np


def normalize_species_recoverability(family_stats):
    all_m = np.concatenate([np.asarray(st["mean_m"], dtype=float) for st in family_stats.values()])
    all_h = np.concatenate([np.asarray(st["mean_h"], dtype=float) for st in family_stats.values()])
    ref_m = float(np.nanmax(all_m)) or 1.0
    ref_h = float(np.nanmax(all_h)) or 1.0

    scaled = {}
    for family, st in family_stats.items():
        scaled[family] = dict(st)
        scaled[family]["mean_m"] = np.asarray(st["mean_m"], dtype=float) / ref_m
        scaled[family]["mean_h"] = np.asarray(st["mean_h"], dtype=float) / ref_h
    return scaled, ref_m, ref_h


def retained_balance_values(family_stats):
    scaled, ref_m, ref_h = normalize_species_recoverability(family_stats)
    values = {}
    for family, st in scaled.items():
        retained = np.asarray(st.get("is_sig", np.ones_like(st["mean_m"], dtype=bool)), dtype=bool)
        h = np.asarray(st["mean_h"], dtype=float)
        m = np.asarray(st["mean_m"], dtype=float)
        vals = (h[retained] - m[retained]) ** 2
        values[family] = vals[np.isfinite(vals)]
    return values, ref_m, ref_h
