import os, sys
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
from contextlib import contextmanager
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.paths import config

# ──────────────────────────────────────────────────────────────────────
# Figure dimensions

class FigureDimensions:
    """Legacy/roomy figure dimensions."""
    standard_height = 3.0
    narrow_width = 3.25
    medium_width = 4.5
    wide_width = 6.5
    square_size = 4.5
    large_size = 7.0
    legend_height = 0.8

class CompactFigureDimensions:
    """Default compact dimensions for stacking many plots."""
    # ~20–25% smaller canvases so more subplots fit without shrinking style
    standard_height = 2.2
    narrow_width = 2.8
    medium_width = 3.6
    wide_width = 5.2
    square_size = 3.6
    large_size = 5.6
    legend_height = 0.6

class PanelMargins:
    """Fixed subplot margins (figure fractions) for identical panel size across scripts."""
    left   = 0.22
    right  = 0.04
    bottom = 0.20
    top    = 0.08

# ──────────────────────────────────────────────────────────────────────
# Figure creation

def create_figure(width='medium', height='standard', dpi=300, *, compact=True):
    """Create a standardized figure.

    width: 'narrow' | 'medium' | 'wide' | 'square' | 'large'
    height: 'standard' | 'legend' | 'square'
    compact: True uses CompactFigureDimensions (DEFAULT). False uses legacy sizes.
    """
    Dims = CompactFigureDimensions if compact else FigureDimensions

    if height == 'square' or width == 'square':
        return plt.figure(figsize=(Dims.square_size, Dims.square_size), dpi=dpi)

    if width == 'large':
        return plt.figure(figsize=(Dims.large_size, Dims.large_size), dpi=dpi)

    width_map = {
        'narrow': Dims.narrow_width,
        'medium': Dims.medium_width,
        'wide':   Dims.wide_width,
    }
    fig_width = width_map.get(width, Dims.medium_width)

    if height == 'legend':
        return plt.figure(figsize=(fig_width, Dims.legend_height), dpi=dpi)

    return plt.figure(figsize=(fig_width, Dims.standard_height), dpi=dpi)

# Convenience constructors (now compact by default)
narrow_figure = partial(create_figure, width='narrow', compact=True)
medium_figure = partial(create_figure, width='medium', compact=True)
wide_figure   = partial(create_figure, width='wide',   compact=True)
square_figure = partial(create_figure, width='square', compact=True)
large_figure  = partial(create_figure, width='large',  compact=True)
legend_figure = partial(create_figure, height='legend', compact=True)

# Roomy variants if needed explicitly
narrow_roomy_figure = partial(create_figure, width='narrow', compact=False)
medium_roomy_figure = partial(create_figure, width='medium', compact=False)
wide_roomy_figure   = partial(create_figure, width='wide',   compact=False)
square_roomy_figure = partial(create_figure, width='square', compact=False)
large_roomy_figure  = partial(create_figure, width='large',  compact=False)
legend_roomy_figure = partial(create_figure, height='legend', compact=False)

# ──────────────────────────────────────────────────────────────────────
# Fixed panel helpers for identical axes geometry across scripts

def add_panel_axes(fig=None, *, left=None, right=None, bottom=None, top=None):
    """Create a single Axes with fixed margins so panel geometry is identical."""
    fig = fig or plt.gcf()
    L = PanelMargins.left   if left   is None else float(left)
    R = PanelMargins.right  if right  is None else float(right)
    B = PanelMargins.bottom if bottom is None else float(bottom)
    T = PanelMargins.top    if top    is None else float(top)
    ax = fig.add_axes([L, B, 1.0 - L - R, 1.0 - B - T])
    return ax

def panel_figure(width='medium', height='standard', dpi=300, *, compact=True,
                 left=None, right=None, bottom=None, top=None):
    """Return (fig, ax) with fixed panel geometry."""
    fig = create_figure(width=width, height=height, dpi=dpi, compact=compact)
    ax = add_panel_axes(fig, left=left, right=right, bottom=bottom, top=top)
    return fig, ax

# Shortcuts you can import
medium_panel = partial(panel_figure, width='medium')
square_panel = partial(panel_figure, width='square')

# Optional: quick debug printer
def report_panel(ax):
    fig = ax.figure
    bb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    print(f"Panel size: {bb.width:.3f} in x {bb.height:.3f} in")

# ──────────────────────────────────────────────────────────────────────
# Styling

def _scaled(v, f):  # small helper
    return round(v * f, 2)

def setup_style(style='ticks', context='paper', font_scale=1.0, rc=None, *, compact=True, element_scale=1.15):
    """Setup universal plotting style.

    compact (DEFAULT): trims whitespace and bumps visual weight.
    element_scale: multiplies line/marker/spine/tick thicknesses.
    """
    # Slight font bump in compact mode
    fs = font_scale * (1.12 if compact else 1.0)

    sns.set_theme(style=style, context=context, palette='deep',
                  font='Arial', font_scale=fs, rc=rc or {})

    base = {
        # Fonts
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.titleweight': 'bold',

        # Lines & markers (scaled)
        'lines.linewidth': _scaled(3.0, element_scale),
        'lines.markersize': _scaled(9.0, element_scale),
        'patch.linewidth': _scaled(2.0, element_scale),
        'lines.solid_capstyle': 'butt',
        'lines.dash_capstyle': 'butt',

        # Axes & ticks (scaled)
        'axes.linewidth': _scaled(2.0, element_scale),
        'axes.edgecolor': 'black',
        'axes.labelpad': 4,
        'xtick.major.size': _scaled(6.0, element_scale),
        'ytick.major.size': _scaled(6.0, element_scale),
        'xtick.major.width': _scaled(1.5, element_scale),
        'ytick.major.width': _scaled(1.5, element_scale),

        # Trim empty margins around data
        'axes.xmargin': 0.04,
        'axes.ymargin': 0.0,

        # Output
        'figure.dpi': 300,
        'figure.constrained_layout.use': False,  # you call tight_layout() downstream
        'figure.autolayout': False,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'savefig.bbox': None,
        'savefig.pad_inches': 0.01,
        'savefig.transparent': True,
        'text.usetex': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    }

    if compact:
        base.update({
            'axes.labelpad': 2,
            'axes.titlepad': 3,
            'xtick.major.size': _scaled(5.0, element_scale),
            'ytick.major.size': _scaled(5.0, element_scale),
            'xtick.major.width': _scaled(1.6, element_scale),
            'ytick.major.width': _scaled(1.6, element_scale),
            'xtick.major.pad': 1.5,
            'ytick.major.pad': 1.5,
            # Compact legends
            'legend.frameon': False,
            'legend.borderpad': 0.2,
            'legend.labelspacing': 0.2,
            'legend.handlelength': 1.0,
            'legend.handletextpad': 0.3,
            'legend.borderaxespad': 0.2,
            'legend.columnspacing': 0.6,
        })

    mpl.rcParams.update(base)

def setup_style_ridge():
    """Specialized compact style for ridge bar plots (tight spacing).

    - Zero x/y margins
    - Smaller tick label padding
    - Keep strong lines/edges from default compact style
    """
    rc = {
        'axes.xmargin': 0.0,
        'axes.ymargin': 0.0,
        'xtick.major.pad': 1.0,
        'ytick.major.pad': 1.0,
    }
    setup_style(rc=rc, compact=True, element_scale=1.15)

def style_axes(ax, offset=6, *, compact=True):
    """Apply minimal, compact axis styling by default."""
    # Hide top/right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Smaller outward offset in compact mode
    eff_offset = 0 if compact else offset

    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(mpl.rcParams['axes.linewidth'])
        ax.spines[spine].set_position(('outward', eff_offset))

    ax.tick_params(which='major',
                   length=mpl.rcParams['xtick.major.size'],
                   width=mpl.rcParams['xtick.major.width'],
                   direction='out', top=False, right=False,
                   pad=mpl.rcParams.get('xtick.major.pad', 2))

def format_axes(ax=None, axis='both', precision=1):
    """Format tick labels with clean decimals.

    Only applies numeric formatting to axes that don't already have string labels.
    This prevents overwriting custom categorical labels.
    """
    ax = ax or plt.gca()

    def clean_formatter(x, pos):
        if x == 0:
            return '0'
        if abs(x) == int(x) and x == int(x):
            return str(int(x))
        s = f"{x:.{precision}f}"
        # trim trailing zeros and trailing dot
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

    fmt = FuncFormatter(clean_formatter)

    # Only format x-axis if it doesn't have custom string labels
    if axis in ('x', 'both'):
        current_labels = [t.get_text() for t in ax.get_xticklabels()]
        # Check if labels are non-empty strings (indicates custom labels)
        has_custom_labels = any(label and not label.replace('.', '').replace('-', '').replace('+', '').isdigit()
                                for label in current_labels)
        if not has_custom_labels:
            ax.xaxis.set_major_formatter(fmt)

    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(fmt)

def adjust_tick_spacing(ax, x_spacing=None, y_spacing=None):
    """Use MaxNLocator to limit tick density."""
    if x_spacing is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_spacing))
    if y_spacing is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_spacing))

@contextmanager
def axes_style(style='white', rc=None):
    with sns.axes_style(style, rc=rc or {}):
        yield

@contextmanager
def plot_context(context='notebook', font_scale=1.0, rc=None):
    with sns.plotting_context(context, font_scale=font_scale, rc=rc or {}):
        yield

def despine_axes(ax=None, **kwargs):
    """Remove spines with compact defaults."""
    ax = ax or plt.gca()
    sns.despine(ax=ax,
                offset=kwargs.get('offset', 2),
                trim=kwargs.get('trim', True),
                left=kwargs.get('left', False),
                right=kwargs.get('right', False),
                top=kwargs.get('top', False),
                bottom=kwargs.get('bottom', False))

def offset_spines(ax, offset=6):
    """Offset left and bottom spines outward."""
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_position(('outward', offset))

# Optional helpers (no need to call if you already use tight_layout + savefig.bbox='tight')
def compactify(ax=None, *, margins=(0.01, 0.02), pad_ticks=True):
    """Trim margins and nudge labels closer."""
    ax = ax or plt.gca()
    ax.margins(x=margins[0], y=margins[1])
    if pad_ticks:
        ax.tick_params(pad=mpl.rcParams.get('xtick.major.pad', 2))
    if ax.title:
        try:
            ax.title.set_pad(mpl.rcParams.get('axes.titlepad', 3))
        except Exception:
            pass
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_pad(mpl.rcParams.get('axes.labelpad', 2))

def export_compact(fig=None, path='figure.pdf', **savefig_kwargs):
    """Tight bbox, tiny padding, transparent background."""
    fig = fig or plt.gcf()
    defaults = dict(bbox_inches='tight', pad_inches=0.01, transparent=True)
    defaults.update(savefig_kwargs)
    fig.savefig(path, **defaults)
    return path

# ──────────────────────────────────────────────────────────────────────
# Backward-compatibility aliases
nar = narrow_figure
med = medium_figure
wide = wide_figure
sq = square_figure
lg = large_figure
leg = legend_figure
style = style_axes
clean_fmt = format_axes
