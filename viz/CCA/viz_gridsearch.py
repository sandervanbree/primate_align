# Visualization for hyperparameter grid search results
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from matplotlib.ticker import LogLocator, MaxNLocator, FuncFormatter
import pickle
from pathlib import Path
import sys, os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from functions.plotting import setup_style, style_axes, format_axes, square_figure, narrow_figure
from config.paths import config

setup_style()

# Load results - only the main gridsearch file
results_path = config.results_dir / 'cca_params.pkl'

if not results_path.exists():
    raise FileNotFoundError(f"Gridsearch results not found at {results_path}")

with open(results_path, 'rb') as f:
    results = pickle.load(f)

# Handle gridsearch result structure
if 'nested_cv_results' in results:
    # Full gridsearch with nested CV
    df_nested = results['nested_cv_results']
    cv_type = 'nested'
else:
    # Fallback for simple CV
    df_nested = results['cv_results']
    cv_type = 'simple'

reg_sweep_results = results['reg_sweep_results']
best_params = results['best_params']

# Print best parameters table
print("\n=== BEST HYPERPARAMETERS ===")
print(f"{'Family':<15} {'Reg (λ)':<12} {'CV R':<8}")
print("-" * 35)
for family, params in best_params.items():
    # Get the corresponding performance from df_nested
    family_perf = df_nested[df_nested['family'] == family]['mean_corr'].iloc[0]
    print(f"{family:<15} {params['reg']:<12.2e} {family_perf:<8.3f}")
print()

# Create output directory
output_dir = config.fig_dir / 'param_gridsearch'
output_dir.mkdir(parents=True, exist_ok=True)

# Color palette from config.toml
color_palette = {
    'cross-species': config.plotting['shared_color'],
    'human_only': config.plotting['human_color'],
    'monkey_only': config.plotting['monkey_color']
}

# --- 1. Plot regularization sweeps ---
for family, df_reg in reg_sweep_results.items():
    fig = narrow_figure()
    ax = fig.add_axes([.1, .1, .85, .78])

    # Get data
    reg_vals = df_reg['regularization'].values
    corr_vals = df_reg['crossview_r'].values

    # Set family-specific y-axis range (±20% around min/max)
    y_min, y_max = corr_vals.min(), corr_vals.max()
    y_range = y_max - y_min
    y_padding = y_range * 0.2
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Plot the data
    ax.plot(reg_vals, corr_vals,
            marker='o', linestyle='-', color='black', linewidth=3,
            markersize=9, zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Regularization (λ)')
    ax.set_ylabel('Cross-view similarity (Pearson\'s R)')
    ax.set_title(family.replace('_', ' ').title())

    # Set log spacing for x-axis with fewer ticks to prevent overlap
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))

    def log_formatter(x, pos):
        if x >= 1000:
            return f'{x:.0e}'
        else:
            return f'{x:.0f}'

    ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))

    # Rotate x-axis labels to prevent overlap
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Set y-axis to show 2 decimal places and prevent duplicate labels
    def y_formatter(x, pos):
        return f'{x:.2f}'

    ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
    # Force y-axis to show proper tick spacing
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Mark best regularization with family-specific color
    best_reg = best_params[family]['reg']
    family_color = color_palette[family]
    ax.axvline(best_reg, color=family_color, linestyle='--', alpha=0.8,
               linewidth=3.5, zorder=6)

    style_axes(ax)

    fig_path = output_dir / f'reg_sweep_{family}.pdf'
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved sweep figure to {fig_path}")

print("All gridsearch figures saved!")
