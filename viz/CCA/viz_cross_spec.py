# -*- coding: utf-8 -*-
# species_generalization_map.py
"""
Visualization for species generalization (Ridge CV).

What it does
------------
• Loads precomputed results from ridge_results.pkl.
• Plots one scatter figure: each component is a point
  (x = mean CV R² in monkeys, y = mean CV R² in humans), colored by family.
• Only labels the first four components per family (by ascending component ID)
  as “C1”, “C2”, “C3”, “C4”. The rest are unlabeled.
• Saves the legend as a separate file.

Notes
-----
• Visualization style matches paper helpers in `functions.plotting`.
• Background shading matches the gradient used in the scree plot (identical
  alpha schedule).

Outputs
-------
Raw mode:
<fig_dir>/crossview/species_ridge.pdf
<fig_dir>/crossview/species_ridge_legend.pdf

Normalized mode:
<fig_dir>/crossview/species_ridge_normalized.pdf
<fig_dir>/crossview/species_ridge_normalized_legend.pdf
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from functions.plotting import setup_style, style_axes, format_axes, square_panel, medium_figure
from functions.recoverability import normalize_species_recoverability
from config.paths import config

# ──────────────────────────────────────────────────────────────────────
# STYLE & OUTPUT
setup_style()
OUT = Path(config.fig_dir) / "crossview"
OUT.mkdir(parents=True, exist_ok=True)

# Plot mode: divide each species' R² by its max across all families/components
NORMALIZE_AXES = True

# Colors (match the paper)
col_hum = config.plotting.get("human_color")
col_mon = config.plotting.get("monkey_color")
col_uni = config.plotting.get("shared_color")
col_grey = config.plotting.get("nonsignificant_color", "#D0D0D0")  # Light grey for non-significant
edge_col = "#333333"  # dark grey outlines

FAM_COLORS = {
    "Cross-species": col_uni,
    "Human-only": col_hum,
    "Monkey-only": col_mon,
}

# ──────────────────────────────────────────────────────────────────────
# Helpers

def _load_ridge_results():
    results_fp = Path(config.results_dir) / "ridge_results.pkl"
    if not results_fp.exists():
        raise FileNotFoundError(f"Ridge results file not found: {results_fp}")
    with open(results_fp, "rb") as f:
        return pickle.load(f)

def _add_gradient_like_scree(ax, lo, hi, n_bands=50):
    """
    Add background gradient identical in spirit to the scree plot’s code:

        y_fill = np.linspace(0, max(y)*1.1, 50)
        alpha_val = 0.2 * (1 - y_val / max(y_fill))

    Here we adapt to arbitrary y-limits [lo, hi] by normalizing to that range.
    """
    x_line = np.linspace(lo, hi, 200)
    y_fill = np.linspace(lo, hi, n_bands)
    denom = (hi - lo) if (hi - lo) != 0 else 1.0
    for i, y0 in enumerate(y_fill[:-1]):
        y1 = y_fill[i + 1]
        # Match the scree plot's alpha schedule (darker at lower y)
        alpha_val = 0.2 * (1 - (y0 - lo) / denom)
        ax.fill_between(x_line, y0, y1, alpha=alpha_val, color="gray",
                        edgecolor="none", zorder=0)

def _normalized_family_stats(family_stats):
    return normalize_species_recoverability(family_stats)

def _species_generalization_map(family_stats, family_colors, title, out_path, x_label, y_label):
    """
    Returns the legend handles so we can save a separate legend.
    """
    fig, ax = square_panel()  # fixed panel geometry (same in all scripts)

    # x = monkey mean CV R^2, y = human mean CV R^2
    all_x, all_y = [], []

    # Build custom legend handles (families + non-significant)
    legend_handles = {}

    for fam, st in family_stats.items():
        x_all = np.asarray(st["mean_m"])  # monkeys on X
        y_all = np.asarray(st["mean_h"])  # humans on Y
        comp_ids = np.asarray(st["comp_ids"])
        is_sig = st.get("is_sig", None)
        if is_sig is None:
            # Backward-compat: if old results file, treat everything as significant
            is_sig = np.ones_like(x_all, dtype=bool)
        else:
            is_sig = np.asarray(is_sig, dtype=bool)

        # Determine which components to label: only the first 4 per family by ascending comp_id
        label_mask = np.zeros_like(comp_ids, dtype=bool)
        if comp_ids.size:
            top4_idx = np.argsort(comp_ids)[:min(4, comp_ids.size)]
            label_mask[top4_idx] = True

        # Split by significance
        x_sig, y_sig = x_all[is_sig], y_all[is_sig]
        x_nsg, y_nsg = x_all[~is_sig], y_all[~is_sig]

        ids_sig = comp_ids[is_sig]
        ids_nsg = comp_ids[~is_sig]
        lab_sig = label_mask[is_sig]
        lab_nsg = label_mask[~is_sig]

        # Non-significant points (grey), WITH dark grey outline - foreground for visibility
        if x_nsg.size:
            ax.scatter(
                x_nsg, y_nsg, s=38, alpha=0.85,
                edgecolors=edge_col, linewidths=0.6,
                c=col_grey, label=None, zorder=4
            )
            # Label ONLY if part of the per-family top-4 (C*), slightly bigger, reduced offset
            for xi, yi, cid, do_lab in zip(x_nsg, y_nsg, ids_nsg, lab_nsg):
                if do_lab:
                    ax.annotate(
                        f"C{int(cid)+1}", xy=(xi, yi),
                        xytext=(1, 1), textcoords="offset points",
                        fontsize=10, ha="left", va="bottom", alpha=0.95, zorder=7
                    )

        # Significant points (colored by family), WITH dark grey outline
        if x_sig.size:
            ax.scatter(
                x_sig, y_sig, s=38, alpha=0.95,
                edgecolors=edge_col, linewidths=0.6,
                c=family_colors.get(fam, "#777777"), label=None, zorder=3
            )
            # Label ONLY if part of the per-family top-4 (C*), slightly bigger, reduced offset
            for xi, yi, cid, do_lab in zip(x_sig, y_sig, ids_sig, lab_sig):
                if do_lab:
                    ax.annotate(
                        f"C{int(cid)+1}", xy=(xi, yi),
                        xytext=(1, 1), textcoords="offset points",
                        fontsize=10, ha="left", va="bottom", alpha=0.98, zorder=8
                    )

        # Save for limits
        all_x.append(x_all); all_y.append(y_all)

        # Prepare family legend handle (one per family) — show outline as well
        if fam not in legend_handles:
            legend_handles[fam] = Line2D(
                [0], [0], marker='o', linestyle='None',
                markerfacecolor=family_colors.get(fam, "#777777"),
                markeredgecolor=edge_col, markeredgewidth=0.6,
                markersize=6, label=fam
            )

    X = np.concatenate(all_x) if all_x else np.array([0.0])
    Y = np.concatenate(all_y) if all_y else np.array([0.0])
    xmin, xmax = np.nanmin(X), np.nanmax(X)
    ymin, ymax = np.nanmin(Y), np.nanmax(Y)
    padx = 0.05 * (xmax - xmin + 1e-9)
    pady = 0.05 * (ymax - ymin + 1e-9)
    lo = min(xmin - padx, ymin - pady, -0.1)   # allow negatives
    hi = max(xmax + padx, ymax + pady,  0.1)

    # Background gradient shading (identical alpha schedule to scree plot)
    _add_gradient_like_scree(ax, lo, hi, n_bands=50)

    # y=x diagonal (thick, a bit more grey)
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.7", lw=3, zorder=1)  # y=x

    # Limits, ticks & labels
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # Explicit 0.2-step ticks so both axes match
    tick_lo = np.floor(lo / 0.2) * 0.2
    tick_hi = np.ceil(hi / 0.2) * 0.2
    ticks = np.arange(tick_lo, tick_hi + 0.01, 0.2)
    ticks = ticks[(ticks >= lo - 0.01) & (ticks <= hi + 0.01)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # No legend on the main plot; we'll save it separately.
    style_axes(ax); format_axes(ax)
    # no tight_layout; no bbox_inches='tight'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # Build and return separate legend handles
    nsg_handle = Line2D(
        [0], [0], marker='o', linestyle='None',
        markerfacecolor=col_grey, markeredgecolor=edge_col,
        markeredgewidth=0.6, markersize=6,
        label="Not significant (p > 0.05)"
    )
    handles_in_order = [
        legend_handles.get("Cross-species"),
        legend_handles.get("Human-only"),
        legend_handles.get("Monkey-only"),
        nsg_handle,
    ]
    return [h for h in handles_in_order if h is not None]

def _save_legend(handles, out_path):
    """Save a standalone legend figure."""
    fig = plt.figure(figsize=(3.2, 2.0))
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=1,
        borderaxespad=0.0
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _summary_bottleneck(mean_h, mean_m):
    """Median of per-component bottleneck (min across species)."""
    if len(mean_h) == 0:
        return np.nan
    return float(np.nanmedian(np.minimum(mean_h, mean_m)))

# ──────────────────────────────────────────────────────────────────────
# Main

print("\nSpecies generalization (Ridge CV) visualization…")
results = _load_ridge_results()
family_stats = results["family_stats"]

if NORMALIZE_AXES:
    family_stats_plot, max_m, max_h = _normalized_family_stats(family_stats)
    x_label = r"Monkey $\mathrm{R}^2$ (normalized)"
    y_label = r"Human $\mathrm{R}^2$ (normalized)"
    title = ""
    out_name = "species_ridge_normalized.pdf"
    legend_name = "species_ridge_normalized_legend.pdf"
    print(f"  Normalization: enabled (R² / max)")
    print(f"  Monkey max R²: {max_m:.6f}; Human max R²: {max_h:.6f}")
else:
    family_stats_plot = family_stats
    x_label = r"Monkey $\mathrm{R}^2$"
    y_label = r"Human $\mathrm{R}^2$"
    title = "Cross-species generalizability"
    out_name = "species_ridge.pdf"
    legend_name = "species_ridge_legend.pdf"
    print("  Normalization: disabled")

# Plot (main) + separate legend
SGM_OUT = OUT / out_name
handles = _species_generalization_map(
    family_stats_plot,
    FAM_COLORS,
    title=title,
    out_path=SGM_OUT,
    x_label=x_label,
    y_label=y_label,
)
LEG_OUT = OUT / legend_name
_save_legend(handles, LEG_OUT)

print(f"\nMap saved to: {SGM_OUT}")
print(f"Legend saved to: {LEG_OUT}")

# Console summary
print("\n=== Summary across components ===")
for fam, st in family_stats.items():
    bn = _summary_bottleneck(st["mean_h"], st["mean_m"])
    hum_mean = float(np.nanmean(st["mean_h"])) if len(st["mean_h"]) else np.nan
    mon_mean = float(np.nanmean(st["mean_m"])) if len(st["mean_m"]) else np.nan
    print(f"  {fam:13s} | n={len(st['comp_ids']):3d} | "
          f"H mean R²: {hum_mean:.3f} | M mean R²: {mon_mean:.3f} | "
          f"Median bottleneck: {bn:.3f}")

print("\nDone.")

# ──────────────────────────────────────────────────────────────────────
def scree_plot(fam_res, null_dist, sig_res, title, show_legend=False):
    """Create scree plot with null distribution, FDR threshold, and significance highlighting."""
    fig = medium_figure()
    ax = fig.add_subplot(111)

    # Expect caller to define n_cc; fall back to length of comp_corrs
    n_cc = fam_res.get('n_cc', len(fam_res.get('comp_corrs', []))) if isinstance(fam_res, dict) else len(fam_res.comp_corrs)
    col_sig = config.plotting.get("sig_color", "#7a3843")

    x = np.arange(1, n_cc + 1)
    y = fam_res['comp_corrs']
    sig_comps = [comp + 1 for comp in sig_res['significant_fdr']]  # Convert to 1-indexed

    # Null distribution 95th percentile
    null_95 = np.percentile(null_dist, 95, axis=0)

    # Gradient background (identical alpha schedule)
    y_fill = np.linspace(0, max(y)*1.1 if len(y) else 1.0, 50)
    for i, y_val in enumerate(y_fill[:-1]):
        # Flipped gradient in given code: alpha decreases with y
        alpha_val = 0.2 * (1 - y_val / y_fill.max())
        ax.fill_between(x, y_val, y_fill[i+1], alpha=alpha_val, color='gray', edgecolor='none', zorder=0)

    # 95% significance threshold line
    ax.plot(x, null_95, color=col_sig, linestyle='-', linewidth=2.25,
            zorder=4, label='Null 95th percentile')

    # Significant component highlighting - continuous ranges (back layer)
    if sig_comps:
        # Find continuous ranges of significant components
        ranges = []
        start = sig_comps[0]
        end = sig_comps[0]

        for i in range(1, len(sig_comps)):
            if sig_comps[i] == end + 1:
                end = sig_comps[i]
            else:
                ranges.append((start, end))
                start = sig_comps[i]
                end = sig_comps[i]
        ranges.append((start, end))

        # Fill continuous ranges with transparent purple
        for start_comp, end_comp in ranges:
            if start_comp <= len(y) and end_comp <= len(y):
                x_range = np.arange(start_comp, end_comp + 1)
                y_range = y[start_comp-1:end_comp]  # Convert back to 0-indexed for array indexing
                ax.fill_between(x_range, 0, y_range, alpha=0.4, color='#7a3843',
                                edgecolor='none', zorder=1)

    # Main observed correlations line (single color)
    ax.plot(x, y, 'k-', lw=3, zorder=5)

    ax.set_xlim(0, n_cc + 1)
    ax.set_ylim(0, 0.95)
    # Natural integer spacing starting from 0, components are 1-indexed
    ax.locator_params(axis='x', integer=True)

    # Clean formatting
    format_axes(ax)
    ax.set_xlabel('Component')
    ax.set_ylabel('Cross-view similarity\n(Pearson\'s $r$)')
    ax.set_title(f'{title}')

    # Add legend only for monkey plot
    if show_legend:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#7a3843', alpha=0.4, edgecolor='none', label='p < 0.05'),
            Line2D([0], [0], color=col_sig, lw=2.25, label='95th percentile H$_0$')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=False)

    style_axes(ax)
    plt.tight_layout()

    return fig
