from operator import truediv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from joblib import Parallel, delayed
import rcca
from sklearn.base import BaseEstimator
import time

def _group_demean(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract per-group row means from X (group-wise demeaning across rows).

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    groups : array-like, shape (n_samples,)
        Group label for each row (e.g., (session, run) codes).
    """
    X = np.asarray(X)
    g = np.asarray(groups)
    if X.shape[0] != g.shape[0]:
        raise ValueError("groups length must match number of rows in X")
    X_dm = X.copy()
    for val in np.unique(g):
        mask = (g == val)
        if np.any(mask):
            X_dm[mask] = X_dm[mask] - X_dm[mask].mean(axis=0, keepdims=True)
    return X_dm

def cca_cv(data_views,
           reg,
           n_cc=10,
           n_folds=5,
           n_reps=1,
           n_jobs=1,
           random_state=42,
           n_perm=0,
           n_reps_perm=1,
           resample_with_replacement=False,
           perm_jobs=None,
           verbose=False):
    """
    Cross-validated CCA, *optionally* with a permutation/bootstrap null.

    Parameters
    ----------
    data_views : dict
        Dictionary with view names as keys and data arrays as values
    reg : float
        Regularization parameter for CCA
    n_cc : int
        Number of canonical components to compute
    n_folds : int
        Number of cross-validation folds
    n_reps : int
        Number of cross-validation repetitions for empirical analysis
    n_jobs : int
        Number of parallel jobs for CV folds
    random_state : int
        Random seed for reproducibility
    n_perm : int
        Number of permutations for null distribution (0 = no permutations)
    n_reps_perm : int
        Number of cross-validation repetitions for each permutation (default: 1)
    resample_with_replacement : bool
        If True, use bootstrap resampling; if False, use shuffling
    perm_jobs : int, optional
        Number of parallel jobs for permutations (defaults to n_jobs)
    verbose : bool
        Whether to print progress information

    Returns
    -------
    obs : dict
        The usual output with keys 'comp_corrs', 'mean_corr', etc.
    null_corrs : ndarray, shape (n_perm, n_cc) or None
        If n_perm>0, the null distribution of component correlations.
        Otherwise returns None.
    """
    # 1) compute the observed correlations
    obs = _cca_cv_core(data_views, reg, n_cc, n_folds, n_reps, n_jobs, random_state)

    # 2) if no permutations requested, bail out
    if n_perm <= 0:
        return obs, None

    # 3) build nulls with robust seed management
    if verbose:
        print(f"Computing {n_perm} permutations with {n_reps_perm} repetitions each...")

    # Generate all permutation seeds from master RNG
    rng = np.random.RandomState(random_state)
    perm_seeds = rng.randint(0, 2**30, size=n_perm)

    # Set up parallelism
    perm_jobs = perm_jobs or n_jobs

    def process_permutation(perm_data):
        seed, perm_idx = perm_data
        perm_rng = np.random.RandomState(seed)

        # for each view, shuffle or bootstrap the rows
        permuted = {}
        for v, X in data_views.items():
            n_samples = X.shape[0]
            if resample_with_replacement:
                idx = perm_rng.randint(0, n_samples, size=n_samples)
            else:
                idx = perm_rng.permutation(n_samples)
            permuted[v] = X[idx]

        # call the *core* function again, but with specified repetitions for permutations
        null_res = _cca_cv_core(permuted,
                               reg,
                               n_cc=n_cc,
                               n_folds=n_folds,
                               n_reps=n_reps_perm,  # Use n_reps_perm for permutations
                               n_jobs=1,  # Use single job for each permutation to avoid oversubscription
                               random_state=perm_rng.randint(2**30))

        # Progress update
        if verbose and (perm_idx + 1) % max(1, n_perm // 10) == 0:
            print(f"  Completed {perm_idx + 1}/{n_perm} permutations")

        return null_res['comp_corrs']

    # Parallelize across permutations with progress tracking
    perm_data = [(seed, idx) for idx, seed in enumerate(perm_seeds)]

    if perm_jobs == 1:
        null_results = [process_permutation(pd) for pd in perm_data]
    else:
        null_results = Parallel(n_jobs=min(perm_jobs, n_perm))(
            delayed(process_permutation)(pd) for pd in perm_data
        )

    null_corrs = np.array(null_results)  # Shape: (n_perm, n_cc)

    if verbose:
        print(f"Completed all {n_perm} permutations")

    return obs, null_corrs


def _cca_cv_core(data_views, reg, n_cc, n_folds, n_reps, n_jobs, random_state):
    """
    *without* any permutation logic.
    Returns just the dict of observed stats.
    """
    view_names = sorted(data_views.keys())
    all_fold_tasks = []
    for rep in range(n_reps):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state + rep)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data_views[view_names[0]])):
            all_fold_tasks.append((rep, fold_idx, train_idx, test_idx))

    def process_single_fold(task_data):
        rep, fold_idx, train_idx, test_idx = task_data
        train_proc, test_proc = [], []
        for view_name in view_names:
            scaler = StandardScaler().fit(data_views[view_name][train_idx])
            train_proc.append(scaler.transform(data_views[view_name][train_idx]))
            test_proc.append(scaler.transform(data_views[view_name][test_idx]))
        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_cc, verbose=False)
        cca.train(train_proc)
        test_projs = [test_proc[i] @ cca.ws[i] for i in range(len(view_names))]

        comp_corrs = np.zeros(n_cc)
        for comp in range(n_cc):
            comp_pair_corrs = []
            for i in range(len(view_names)):
                for j in range(i+1, len(view_names)):
                    pi = test_projs[i][:, comp]
                    pj = test_projs[j][:, comp]
                    comp_pair_corrs.append(abs(np.corrcoef(pi, pj)[0,1]))
            comp_corrs[comp] = np.mean(comp_pair_corrs)

        view_comp_corrs = {}
        for i, view_name in enumerate(view_names):
            view_comp_corrs[view_name] = np.zeros(n_cc)
            for comp in range(n_cc):
                pi = test_projs[i][:, comp]
                other_corrs = []
                for j in range(len(view_names)):
                    if j != i:
                        pj = test_projs[j][:, comp]
                        other_corrs.append(abs(np.corrcoef(pi, pj)[0,1]))
                view_comp_corrs[view_name][comp] = np.mean(other_corrs)

        view_corrs = {v: np.mean(view_comp_corrs[v]) for v in view_names}
        return {
            'rep': rep,
            'fold': fold_idx,
            'view_corrs': view_corrs,
            'view_comp_corrs': view_comp_corrs,
            'comp_corrs': comp_corrs,
            'mean_corr': np.mean(comp_corrs)
        }

    if n_jobs == 1:
        fold_results = [process_single_fold(task) for task in all_fold_tasks]
    else:
        fold_results = Parallel(n_jobs=min(n_jobs, len(all_fold_tasks)))(
            delayed(process_single_fold)(task) for task in all_fold_tasks
        )

    all_view_corrs = {v: [] for v in view_names}
    all_comp_corrs = []
    all_view_comp_corrs = {v: [] for v in view_names}
    for fold_result in fold_results:
        for view_name in fold_result['view_corrs']:
            all_view_corrs[view_name].append(fold_result['view_corrs'][view_name])
            all_view_comp_corrs[view_name].append(fold_result['view_comp_corrs'][view_name])
        all_comp_corrs.append(fold_result['comp_corrs'])

    all_comp_corrs = np.array(all_comp_corrs)
    mean_comp_corrs = all_comp_corrs.mean(axis=0)
    view_comp_means = {}
    view_comp_stds = {}
    for view_name in all_view_comp_corrs:
        view_comp_array = np.array(all_view_comp_corrs[view_name])
        view_comp_means[view_name] = view_comp_array.mean(axis=0)
        view_comp_stds[view_name] = view_comp_array.std(axis=0)

    overall_mean = np.mean([r['mean_corr'] for r in fold_results])
    overall_std = np.std([r['mean_corr'] for r in fold_results])
    view_means = {v: np.mean(all_view_corrs[v]) for v in all_view_corrs}
    view_stds = {v: np.std(all_view_corrs[v]) for v in all_view_corrs}

    return {
        'mean_corr': overall_mean,
        'std_corr': overall_std,
        'view_corrs': view_means,
        'view_stds': view_stds,
        'comp_corrs': mean_comp_corrs,
        'comp_stds': all_comp_corrs.std(axis=0),
        'view_comp_corrs': view_comp_means,
        'view_comp_stds': view_comp_stds,
        'fold_results': fold_results
    }
def cca_nested_cv(views, reg_params, n_cc=10, k_outer=5, k_inner=5, n_reps=1, n_jobs=1, seed=42):
    """
    Nested cross-validated CCA with optimized parallelization.
    """
    view_names = sorted(views.keys())
    outer_kf = KFold(n_splits=k_outer, shuffle=True, random_state=seed)

    def process_outer_fold(fold_data):
        i, (train_idx, test_idx) = fold_data
        outer_train = {v: views[v][train_idx] for v in view_names}
        outer_test = {v: views[v][test_idx] for v in view_names}

        def evaluate_reg_param(reg):
            inner_results, _ = cca_cv(
                outer_train, reg=reg, n_cc=n_cc, n_folds=k_inner,
                n_reps=n_reps, n_jobs=1, random_state=seed
            )
            return reg, inner_results['mean_corr']

        available_jobs = max(1, n_jobs // k_outer)
        inner_jobs = min(available_jobs, len(reg_params))

        if inner_jobs == 1:
            inner_results = [evaluate_reg_param(reg) for reg in reg_params]
        else:
            inner_results = Parallel(n_jobs=inner_jobs)(
                delayed(evaluate_reg_param)(reg) for reg in reg_params
            )

        best_reg = max(inner_results, key=lambda x: x[1])[0]
        fit_res = fit_cca(outer_train, reg=best_reg, n_cc=n_cc)
        cca, scalers = fit_res['cca'], fit_res['scalers']
        test_proc = [scalers[v].transform(outer_test[v]) for v in view_names]
        test_projs = [test_proc[i] @ cca.ws[i] for i in range(len(view_names))]

        comp_corrs = np.zeros(n_cc)
        for comp in range(n_cc):
            pair_corrs = []
            for v1 in range(len(view_names)):
                for v2 in range(v1 + 1, len(view_names)):
                    pi = test_projs[v1][:, comp]
                    pj = test_projs[v2][:, comp]
                    pair_corrs.append(abs(np.corrcoef(pi, pj)[0,1]))
            comp_corrs[comp] = np.mean(pair_corrs)

        return np.mean(comp_corrs)

    outer_fold_data = list(enumerate(outer_kf.split(views[view_names[0]])))

    print(f"  Processing {k_outer} outer folds with {len(reg_params)} reg params each...")

    if n_jobs == 1:
        outer_scores = [process_outer_fold(fold_data) for fold_data in outer_fold_data]
    else:
        outer_scores = Parallel(n_jobs=min(n_jobs, k_outer))(
            delayed(process_outer_fold)(fold_data) for fold_data in outer_fold_data
        )

    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    print(f"  Outer CV complete: R = {mean_score:.3f} ± {std_score:.3f}")

    return {'mean_corr': mean_score, 'std_corr': std_score, 'scores': outer_scores}

def fit_cca(views, reg, n_cc=10, group_labels=None, demean_groups=True):
    """
    Fit CCA on a full dataset (for final model after hyperparameter selection).
    """
    view_names = sorted(views.keys())

    # Standardize each view
    proc_views = []
    scalers = {}
    for v_name in view_names:
        Xv = views[v_name]
        if demean_groups and group_labels is not None and v_name in group_labels and group_labels[v_name] is not None:
            Xv = _group_demean(Xv, group_labels[v_name])
        scaler = StandardScaler().fit(Xv)
        scalers[v_name] = scaler
        proc_views.append(scaler.transform(Xv))

    # Fit CCA
    cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_cc, verbose=False)
    cca.train(proc_views)

    # Get canonical components (projections of training data)
    comps = {view_names[i]: cca.comps[i] for i in range(len(view_names))}

    return {
        'cca': cca,
        'scalers': scalers,
        'view_names': view_names,
        'components': comps,
        'weights': {view_names[i]: cca.ws[i] for i in range(len(view_names))}
    }
