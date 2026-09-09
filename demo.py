"""Run cross-validated CCA on a small simulated five-view dataset."""

from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from functions.cca import cca_cv


def main():
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(120, 3))
    views = {
        f"view_{i + 1}": latent @ rng.normal(size=(3, 12))
        + 0.1 * rng.normal(size=(120, 12))
        for i in range(5)
    }
    with threadpool_limits(limits=1):
        observed, _ = cca_cv(
            views, reg=1.0, n_cc=3, n_folds=3, n_reps=1,
            n_jobs=1, random_state=42,
        )
    correlations = np.asarray(observed["comp_corrs"])
    if not np.isfinite(correlations).all():
        raise RuntimeError("CCA returned non-finite correlations.")

    out = Path(__file__).resolve().parent / "results" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "simulated_views.npz", **views)
    np.savetxt(
        out / "component_correlations.csv",
        np.column_stack((np.arange(1, 4), correlations)),
        delimiter=",", header="component,mean_held_out_correlation",
        comments="", fmt=["%d", "%.6f"],
    )
    print("Mean held-out correlations:", np.round(correlations, 3))
    print("Saved simulated data and component correlations to", out)


if __name__ == "__main__":
    main()
