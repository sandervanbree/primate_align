import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.paths import config

def get_ridge_params():
    r = config.hyperparameters.get('ridge', {})
    # Prefer reg_* if provided (log10 range, lin spacing in log space)
    if all(k in r for k in ('reg_start', 'reg_stop', 'reg_num')):
        a0 = float(r.get('reg_start'))
        a1 = float(r.get('reg_stop'))
        n = int(r.get('reg_num'))
        alphas = np.logspace(a0, a1, num=max(2, n))
    else:
        alphas = r.get('alphas', [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    n_jobs = config.analysis.get('n_jobs', 4)
    return np.array(alphas, float), n_jobs

def get_ridge_cv_settings():
    legacy = config.hyperparameters.get('eval_ridge_mcca', {})
    shared = config.hyperparameters.get('ridge_cv', {})
    viz = config.hyperparameters.get('viz', {})
    return {
        'outer_folds': int(shared.get('outer_folds', legacy.get('outer_folds', 5))),
        'outer_repeats': int(shared.get('outer_repeats', legacy.get('outer_repeats', 3))),
        'inner_folds': int(shared.get('inner_folds', legacy.get('inner_folds', 3))),
        'seed': int(shared.get('seed', legacy.get('seed', 42))),
        'pca_max_dim': int(shared.get('pca_max_dim', viz.get('ridge_pca_n', 250))),
    }

def ridge_cv_fold(X, y, n_folds=10, seed=42):
    """RidgeCV with k-fold stats. Returns (mean_r2, std_r2)."""
    alphas, n_jobs = get_ridge_params()
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        ridge = RidgeCV(alphas=alphas, cv=3)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        scores.append(r2_score(y_test, y_pred))

    return np.mean(scores), np.std(scores)

def ridge_joint(X, y, n_folds=10, seed=42):
    """Joint ridge for all features together. Returns (r2_mean, r2_std)."""
    return ridge_cv_fold(X, y, n_folds, seed)


def ridge_univariate_vec_icv(X, y, n_folds=10, n_inner=3, seed=42):
    """Vectorized univariate ridge with inner 3-fold CV per outer fold.

    - Keeps the exact model selection scheme (inner CV over alphas) but
      computes all features at once to avoid thousands of tiny fits.
    - Outer KFold still evaluates generalization; scores are R^2 per feature.

    Returns
    -------
    r2_mean : (p,) float array
        Mean R^2 across outer folds for each feature column.
    r2_std : (p,) float array
        Std of R^2 across outer folds for each feature column.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, p = X.shape
    alphas, _ = get_ridge_params()

    outer = KFold(n_splits=int(n_folds), shuffle=True, random_state=seed)
    r2_folds = []

    for tr_idx, te_idx in outer.split(X):
        Xtr_full, Xte_full = X[tr_idx], X[te_idx]
        ytr_full, yte_full = y[tr_idx], y[te_idx]

        # Standardize on outer train; center y on outer train mean
        xs = StandardScaler().fit(Xtr_full)
        Xtr = xs.transform(Xtr_full)
        Xte = xs.transform(Xte_full)
        ymu_outer = float(ytr_full.mean())
        ytr0 = ytr_full - ymu_outer

        # Inner CV on outer train indices (same as RidgeCV(cv=3) but batched)
        inner = KFold(n_splits=int(n_inner), shuffle=True, random_state=seed)
        r2_inner_sum = np.zeros((p, alphas.size), float)

        for itr, ival in inner.split(Xtr):
            X_in_tr = Xtr[itr];  X_in_val = Xtr[ival]
            y_in_tr = ytr0[itr]; y_in_val = ytr_full[ival]

            # Re-standardize inside inner train to match per-fold scaler
            mu = X_in_tr.mean(axis=0)
            sd = X_in_tr.std(axis=0, ddof=0)
            sd_safe = np.where(sd > 1e-12, sd, 1.0)
            Xtr_s = (X_in_tr - mu) / sd_safe
            Xva_s = (X_in_val - mu) / sd_safe

            # Center w.r.t. inner train mean for predictions
            ymu_inner = float(y_in_tr.mean())
            ytr_s = y_in_tr - ymu_inner
            yva_s = y_in_val - ymu_inner

            # Sufficient stats (vectorized across features)
            sxx = (Xtr_s**2).sum(axis=0)             # (p,)
            sxy = Xtr_s.T @ ytr_s                    # (p,)
            sxxv = (Xva_s**2).sum(axis=0)            # (p,)
            syv = Xva_s.T @ yva_s                     # (p,)

            a = alphas[None, :]                       # (1, A)
            sxx_m = sxx[:, None]
            sxy_m = sxy[:, None]
            sxxv_m = sxxv[:, None]
            syv_m = syv[:, None]
            denom = sxx_m + a
            w = sxy_m / (denom + 1e-12)              # (p, A)

            # Validation R^2 (numerically matches RidgeCV scoring)
            den_val = ((y_in_val - y_in_val.mean())**2).sum() + 1e-12
            yy = float(((yva_s)**2).sum())
            rss_val = yy - 2.0*(w * syv_m) + (w**2) * sxxv_m
            r2_val = 1.0 - (rss_val / den_val)
            r2_inner_sum += r2_val

        # Select alpha per feature (argmax over inner folds' mean R^2)
        a_idx = np.argmax(r2_inner_sum, axis=1)

        # Fit outer train stats under outer scaler/centering
        sxx_tr = (Xtr**2).sum(axis=0)
        sxy_tr = Xtr.T @ ytr0
        w_star = sxy_tr / (sxx_tr + alphas[a_idx] + 1e-12)

        # Evaluate on outer test (R^2 for each feature independently)
        yhat_te = (Xte * w_star[None, :]) + ymu_outer  # (nte, p)
        den = ((yte_full - yte_full.mean())**2).sum() + 1e-12
        num = ((yte_full[:, None] - yhat_te)**2).sum(axis=0)
        r2_fold = 1.0 - num / den                      # (p,)
        r2_folds.append(r2_fold)

    R = np.vstack(r2_folds)  # (K, p)
    return R.mean(axis=0), R.std(axis=0)
