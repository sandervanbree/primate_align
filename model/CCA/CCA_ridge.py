# CCA_ridge.py
"""
Ridge CV encoding of CCA component families (species generalization).

What it does
------------
• Loads cached CCA state (cca_state.pkl) for subject feature matrices.
• Loads crossview_results.pkl to obtain per-family component scores and nulls.
• Per family, predicts component scores from each subject’s features via
  a pipeline: StandardScaler → (optional) PCA(whiten) → Ridge.
  Uses nested CV (outer RepeatedKFold; inner KFold grid-search over alpha).
• Aggregates per-component CV R² across subjects (mean for humans and monkeys),
  annotates significance via FDR (Benjamini–Hochberg) using stored nulls,
  and writes ridge_results.pkl for downstream visualization.

Notes
-----
• Parallelism and ridge regularization grids are driven by config.toml.
• PCA dimensionality is set through the ridge CV configuration.
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits  # BLAS thread control

from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests

# ──────────────────────────────────────────────────────────────────────
# Project imports / I/O configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.paths import config
from model.feature.core.ridge_utils import get_ridge_params, get_ridge_cv_settings

# ──────────────────────────────────────────────────────────────────────
# RUNTIME / CV SETTINGS
TOTAL_CORES = os.cpu_count() or 8
ALPHA_GRID, _ridge_cfg_jobs = get_ridge_params()
CFG_ANALYSIS_JOBS = int(config.analysis.get("n_jobs", _ridge_cfg_jobs))
MAX_JOBLIB_JOBS = max(1, min(CFG_ANALYSIS_JOBS, TOTAL_CORES))
EPARAMS = config.hyperparameters.get("eval_ridge_mcca", {})
CV_SETTINGS = get_ridge_cv_settings()

USE_PCA = True
PCA_K = int(CV_SETTINGS["pca_max_dim"])
PCA_WHITEN = True

OUTER_SPLITS = int(CV_SETTINGS["outer_folds"])
OUTER_REPEATS = int(CV_SETTINGS["outer_repeats"])
INNER_SPLITS = int(CV_SETTINGS["inner_folds"])
RNG_SEED = int(CV_SETTINGS["seed"])
FDR_ALPHA = float(EPARAMS.get("fdr_alpha", 0.05))

# ──────────────────────────────────────────────────────────────────────
# Helpers

def _load_cached_cca_state():
    state_fp = Path(config.results_dir) / "cca_state.pkl"
    if not state_fp.exists():
        raise FileNotFoundError(f"CCA state file not found: {state_fp}")
    with open(state_fp, "rb") as f:
        return pickle.load(f)

def _outer_nested_cv_multi_r2(
    X, Y, alpha_grid, use_pca, pca_k, pca_whiten,
    outer_splits, outer_repeats, inner_splits, rng_seed
):
    """
    Fit pipeline with nested CV and return per-component CV R².
    X: (n_stim x p), Y: (n_stim x K) multi-target.
    """
    n_samples, p = X.shape
    eff_k = min(pca_k, p, n_samples - 1) if use_pca else 0

    outer_cv = RepeatedKFold(
        n_splits=max(2, min(outer_splits, n_samples)),
        n_repeats=max(1, outer_repeats),
        random_state=rng_seed
    )

    y_true_all, y_pred_all = [], []
    for tr, te in outer_cv.split(X):
        inner_cv = KFold(
            n_splits=max(2, min(inner_splits, len(tr))),
            shuffle=True, random_state=rng_seed
        )

        Xtr_raw, Xte_raw = X[tr], X[te]
        Ytr, Yte = Y[tr], Y[te]

        alpha_scores = np.zeros(len(alpha_grid), dtype=float)
        for inner_tr, inner_va in inner_cv.split(Xtr_raw):
            X_inner_tr = Xtr_raw[inner_tr]
            X_inner_va = Xtr_raw[inner_va]
            Y_inner_tr = Ytr[inner_tr]
            Y_inner_va = Ytr[inner_va]

            scaler = StandardScaler()
            X_inner_tr = scaler.fit_transform(X_inner_tr)
            X_inner_va = scaler.transform(X_inner_va)

            if use_pca:
                pca = PCA(n_components=eff_k, whiten=pca_whiten, random_state=rng_seed)
                X_inner_tr = pca.fit_transform(X_inner_tr)
                X_inner_va = pca.transform(X_inner_va)

            for ai, alpha in enumerate(alpha_grid):
                ridge = Ridge(alpha=float(alpha))
                ridge.fit(X_inner_tr, Y_inner_tr)
                pred_inner = ridge.predict(X_inner_va)
                alpha_scores[ai] += r2_score(Y_inner_va, pred_inner)

        best_alpha = float(alpha_grid[int(np.argmax(alpha_scores))])

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)

        if use_pca:
            pca = PCA(n_components=eff_k, whiten=pca_whiten, random_state=rng_seed)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)

        ridge = Ridge(alpha=best_alpha)
        ridge.fit(Xtr, Ytr)
        y_pred_all.append(ridge.predict(Xte))
        y_true_all.append(Y[te])

    Y_true = np.concatenate(y_true_all, axis=0)
    Y_pred = np.concatenate(y_pred_all, axis=0)
    return r2_score(Y_true, Y_pred, multioutput="raw_values").astype(float)

def _species_of_view(view_name: str) -> str:
    if view_name.startswith("human_"):
        return "human"
    if view_name.startswith("monkey_"):
        return "monkey"
    return "unknown"


def _target_views_within_family(
    comp_views: dict[str, np.ndarray],
    subject_view: str,
    family_name: str,
) -> list[str]:
    """Select target views for one subject.

    For cross-species family, use true leave-one-view-out over all remaining
    views in that family. For within-species families, keep same-species target
    preference with fallbacks.
    """
    fam_views = list(comp_views.keys())
    if family_name == "Cross-species":
        if subject_view in comp_views:
            return [v for v in fam_views if v != subject_view]
        return fam_views

    subj_species = _species_of_view(subject_view)
    if subject_view in comp_views:
        same_species = [v for v in fam_views if v != subject_view and _species_of_view(v) == subj_species]
        if same_species:
            return same_species
        # fall back to all other views if no same-species partner exists
        return [v for v in fam_views if v != subject_view]
    # subject view not part of family → use all views belonging to the opposite / available species
    if subj_species == "human":
        return [v for v in fam_views if _species_of_view(v) == "monkey"] or fam_views
    if subj_species == "monkey":
        return [v for v in fam_views if _species_of_view(v) == "human"] or fam_views
    return fam_views


def _eval_family_lovo(comp_views: dict[str, np.ndarray], comp_indices, species_dict, family_name: str):
    """Return per-subject CV R² with leave-one-view-out targets."""
    if not species_dict:
        return {}
    keys = sorted(species_dict.keys())
    n_jobs = min(MAX_JOBLIB_JOBS, len(keys)) or 1

    cols = np.asarray(comp_indices, int)

    def _one_subject(view_name: str):
        target_views = _target_views_within_family(comp_views, view_name, family_name)
        if not target_views:
            return None, None
        targets = [comp_views[v][:, cols] for v in target_views]
        Y = np.mean(targets, axis=0)
        Xs = species_dict[view_name]
        r2_vec = _outer_nested_cv_multi_r2(
            Xs, Y,
            ALPHA_GRID, USE_PCA, PCA_K, PCA_WHITEN,
            OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS, RNG_SEED
        )
        return view_name, r2_vec

    # Keep inner linear algebra single-threaded while subjects run in parallel.
    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_subject)(k) for k in keys if k in species_dict
        )
    return {k: v for k, v in results if k is not None and v is not None}

def _agg_over_subjects(r2_dict, n_comp, how="mean"):
    """Aggregate (subject × comps) → (comps,) via mean or median."""
    if not r2_dict:
        return np.full(n_comp, np.nan, dtype=float)
    arr = np.stack(list(r2_dict.values()), axis=0)
    return np.nanmedian(arr, axis=0) if how == "median" else np.nanmean(arr, axis=0)

def _summary_bottleneck(mean_h, mean_m):
    """Median of per-component bottleneck (min across species)."""
    if len(mean_h) == 0:
        return np.nan
    return float(np.nanmedian(np.minimum(mean_h, mean_m)))

# ──────────────────────────────────────────────────────────────────────
# Main

def main():
    print("\nSpecies generalization (Ridge CV) computations…")
    crossview_fp = Path(config.results_dir) / "crossview_results.pkl"
    with open(crossview_fp, "rb") as f:
        crossview = pickle.load(f)

    state = _load_cached_cca_state()
    X_human = state["X_human"]
    X_monkey = state["X_monkey"]

    comp_mats = {}
    comp_views_by_fam = {}
    for fam_key, comp_key in [
        ("Cross-species", "cross-species"),
        ("Human-only", "human_only"),
        ("Monkey-only", "monkey_only"),
    ]:
        fam_comps = crossview["components"][comp_key]
        comp_views_by_fam[fam_key] = fam_comps
        comp_mats[fam_key] = np.mean([fam_comps[v] for v in fam_comps], axis=0)

    all_idxs = {}
    for fam, views in comp_views_by_fam.items():
        example = next(iter(views.values()))
        all_idxs[fam] = list(range(example.shape[1]))

    n_stim = next(iter(comp_mats.values())).shape[0]
    for sid, X in X_human.items():
        if X.shape[0] != n_stim:
            raise ValueError(
                f"Human subject '{sid}' has n_stim={X.shape[0]} but components expect {n_stim}."
            )
    for sid, X in X_monkey.items():
        if X.shape[0] != n_stim:
            raise ValueError(
                f"Monkey subject '{sid}' has n_stim={X.shape[0]} but components expect {n_stim}."
            )

    print(
        f"  Parallelization: joblib≤{MAX_JOBLIB_JOBS} (threads), "
        f"BLAS threads≤{max(1, TOTAL_CORES // MAX_JOBLIB_JOBS)}"
    )
    print(f"  PCA: enabled={USE_PCA}, k={PCA_K}, whiten={PCA_WHITEN}")
    print(
        f"  Outer CV: {OUTER_SPLITS}-fold × {OUTER_REPEATS}; "
        f"Inner CV: {INNER_SPLITS}-fold"
    )
    print(f"  Alpha grid: {ALPHA_GRID}\n")

    alpha = FDR_ALPHA

    def compute_sig_and_fdr(obs_corrs, null_corrs, alpha=0.05):
        """Compute p-values, FDR correction, and significance thresholds."""
        p_vals = []
        for comp in range(len(obs_corrs)):
            p_vals.append(np.mean(null_corrs[:, comp] >= obs_corrs[comp]))
        p_vals = np.array(p_vals)
        rejected, q_vals, _, _ = multipletests(p_vals, alpha=alpha, method="fdr_bh")
        fdr_sig_comps = np.where(rejected)[0].tolist()
        fdr_thresh_corr = np.min(obs_corrs[rejected]) if fdr_sig_comps else None
        return {
            "p_values": p_vals.tolist(),
            "q_values": q_vals.tolist(),
            "significant_fdr": fdr_sig_comps,
            "fdr_threshold_corr": fdr_thresh_corr,
            "optimal_n_cc": len(fdr_sig_comps),
        }

    fams = {
        "Cross-species": crossview["cross-species"],
        "Human-only": crossview["human_only"],
        "Monkey-only": crossview["monkey_only"],
    }
    null_dists = crossview["null_distributions"]

    sig_results = {}
    for fam_key, null_key in [
        ("Cross-species", "cross-species"),
        ("Human-only", "human_only"),
        ("Monkey-only", "monkey_only"),
    ]:
        sig_results[null_key] = compute_sig_and_fdr(
            fams[fam_key]["comp_corrs"],
            null_dists[null_key],
            alpha,
        )

    print(f"\n=== FDR CORRECTION RESULTS (alpha={alpha}) ===")
    for fam_key, sig in sig_results.items():
        fdr_sig = len(sig["significant_fdr"])
        print(
            f"{fam_key.capitalize()}: {fdr_sig} significant components after FDR: "
            f"{sig['significant_fdr']}"
        )

    sig_idxs = {
        "Cross-species": sorted(sig_results["cross-species"]["significant_fdr"]),
        "Human-only": sorted(sig_results["human_only"]["significant_fdr"]),
        "Monkey-only": sorted(sig_results["monkey_only"]["significant_fdr"]),
    }
    for fam in sig_idxs:
        k_max = comp_mats[fam].shape[1]
        sig_idxs[fam] = [ci for ci in sig_idxs[fam] if ci < k_max]

    family_stats = {}
    for fam, mat in comp_mats.items():
        comp_ids = all_idxs[fam]
        K = len(comp_ids)
        if K == 0:
            print(f"  {fam}: no components. Skipping.")
            family_stats[fam] = dict(
                mean_h=np.array([]),
                mean_m=np.array([]),
                comp_ids=[],
                is_sig=np.array([]),
                sig_ids=[],
            )
            continue

        print(f"  {fam}: evaluating {K} components → {comp_ids}")
        comp_views = comp_views_by_fam[fam]
        r2_hum = _eval_family_lovo(comp_views, comp_ids, X_human, fam)
        r2_mon = _eval_family_lovo(comp_views, comp_ids, X_monkey, fam)

        mean_h = _agg_over_subjects(r2_hum, K, how="mean")
        mean_m = _agg_over_subjects(r2_mon, K, how="mean")

        sig_set = set(sig_idxs[fam])
        is_sig = np.array([i in sig_set for i in comp_ids], dtype=bool)

        family_stats[fam] = dict(
            mean_h=mean_h,
            mean_m=mean_m,
            comp_ids=comp_ids,
            is_sig=is_sig,
            sig_ids=sorted(sig_idxs[fam]),
        )

    results_fp = Path(config.results_dir) / "ridge_results.pkl"
    with open(results_fp, "wb") as f:
        pickle.dump({"family_stats": family_stats, "COMP_MATS": comp_mats}, f)
    print(f"\nResults saved to: {results_fp}")

    print("\n=== Summary across components ===")
    for fam, st in family_stats.items():
        bn = _summary_bottleneck(st["mean_h"], st["mean_m"])
        hum_mean = float(np.nanmean(st["mean_h"])) if len(st["mean_h"]) else np.nan
        mon_mean = float(np.nanmean(st["mean_m"])) if len(st["mean_m"]) else np.nan
        print(
            f"  {fam:13s} | n={len(st['comp_ids']):3d} | "
            f"H mean R²: {hum_mean:.3f} | M mean R²: {mon_mean:.3f} | "
            f"Median bottleneck: {bn:.3f}"
        )

    print("\nDone.")

if __name__ == "__main__":
    main()
