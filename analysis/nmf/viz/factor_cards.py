#!/usr/bin/env python3
"""Shared main-text renderer for sNMF factor-card composites.

The renderer is intentionally agnostic to how factors were selected: callers
may supply a ranked prefix or an explicit curated list.  Every factor is shown
as the same self-contained unit: a 3 x 3 exemplar mosaic, a compact category
word cloud, and paired human / macaque CCA-space re-expression bars.

This module owns presentation only.  Supplementary ranked-strip galleries use
the same palette, icons, word-colour scaling, and image-loading helpers through
``analysis.supp.gallery_style`` while retaining their denser strip layout.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None


ASSETS = PROJECT_ROOT / 'assets'

HUMAN = config.plotting.get('human_color', '#7c5799')
MONKEY = config.plotting.get('monkey_color', '#bda855')
SHARED = config.plotting.get('shared_color', '#81B7B3')

INK = '#1c1c1e'
INK_SOFT = '#686d72'
GRID_OUTLINE = '#26262a'
OUTLINE_RGB = (0x26, 0x26, 0x2a)

# Neutral gray panel surfaces.
PANEL_FILL = '#e4e7e9'
PANEL_RULE = '#bcc2c7'

# The empty portion of each bar is deliberately near-white so it separates
# from the gray information panel without demanding attention.
TRACK_FILL = '#fafbfc'
TRACK_EDGE = '#d3d8dc'

GAUGE_MAX = 0.85
ICON_ZOOM = 0.030
FONT_PATH = None
FAMILY_COLOR = {'shared': SHARED, 'human': HUMAN, 'monkey': MONKEY}


def configure_fonts():
    global FONT_PATH
    for name in ('Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'):
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if path:
                font_manager.fontManager.addfont(path)
                family = font_manager.FontProperties(fname=path).get_name()
                FONT_PATH = path
                break
        except Exception:
            continue
    else:
        family = 'DejaVu Sans'
        FONT_PATH = font_manager.findfont('DejaVu Sans')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [family, 'Arial', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def _entry(entry, name, default=None):
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)


@lru_cache(maxsize=8)
def _icon_rgba(name, color):
    image = Image.open(ASSETS / f'{name}.png').convert('RGBA')
    arr = np.asarray(image).astype(float) / 255.0
    luminance = arr[..., :3].mean(axis=2)
    red, green, blue = to_rgb(color)
    factor = np.clip(0.45 + 0.62 * luminance, 0.0, 1.0)
    out = np.empty_like(arr)
    out[..., 0] = red * factor
    out[..., 1] = green * factor
    out[..., 2] = blue * factor
    out[..., 3] = arr[..., 3]
    return out


@lru_cache(maxsize=4096)
def _load_square(path, size):
    try:
        image = Image.open(path).convert('RGB')
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((size, size), Image.LANCZOS)
    except Exception:
        return Image.new('RGB', (size, size), (205, 205, 205))


def exemplar_mosaic(entry, image_paths, rows=3, cols=3, tile_px=210, gap=4):
    weights = np.asarray(_entry(entry, 'weights'), float)
    indices = np.argsort(weights)[::-1][:rows * cols]
    width = cols * tile_px + (cols + 1) * gap
    height = rows * tile_px + (rows + 1) * gap
    canvas = Image.new('RGB', (width, height), OUTLINE_RGB)
    for position, stimulus_index in enumerate(indices):
        row, col = divmod(position, cols)
        tile = _load_square(str(image_paths[int(stimulus_index)]), tile_px)
        x = gap + col * (tile_px + gap)
        y = gap + row * (tile_px + gap)
        canvas.paste(tile, (x, y))
    return np.asarray(canvas)


def _size_color(family_color, strength):
    hue, saturation, value = rgb_to_hsv(np.array(to_rgb(family_color)))
    saturation = float(np.clip(saturation * (0.9 + 1.7 * strength), 0, 1))
    value = float(value * (0.88 - 0.16 * strength))
    return hsv_to_rgb(np.array([hue, saturation, value]))


def add_wordcloud(fig, rect, entry, width_in, height_in, family_color,
                  max_words=5):
    ax = fig.add_axes(rect)
    ax.axis('off')
    ax.patch.set_visible(False)
    enrichment = _entry(entry, 'enrichment', [])
    items = [(re.sub(r'\d+$', '', category).replace('_', ' ').strip(), float(auc))
             for category, auc in enrichment if auc > 0.5][:max_words]
    if not items:
        return
    maximum = max(auc for _, auc in items)

    if WordCloud is None:
        for index, (word, auc) in enumerate(items):
            strength = (auc - 0.5) / (maximum - 0.5 + 1e-9)
            ax.text(0.5, 0.86 - index * 0.18, word, ha='center', va='center',
                    fontsize=9 + 5 * strength, fontweight='bold',
                    color=_size_color(family_color, strength))
        return

    frequencies = {word: max(auc - 0.49, 0.02) for word, auc in items}
    component = int(_entry(entry, 'component', _entry(entry, 'comp_id', 0)))
    cloud = WordCloud(
        width=max(160, int(width_in * 240)),
        height=max(160, int(height_in * 240)),
        font_path=FONT_PATH,
        background_color=None,
        mode='RGBA',
        prefer_horizontal=1.0,
        relative_scaling=0.5,
        margin=1,
        min_font_size=15,
        random_state=1700 + component,
    ).generate_from_frequencies(frequencies)

    font_sizes = [font_size for (_word, font_size, _position, _orientation, _color)
                  in cloud.layout_]
    if font_sizes:
        minimum_size, maximum_size = min(font_sizes), max(font_sizes)

        def color_by_size(word, font_size, **kwargs):
            strength = ((font_size - minimum_size)
                        / (maximum_size - minimum_size + 1e-9))
            red, green, blue = _size_color(family_color, strength)
            return int(255 * red), int(255 * green), int(255 * blue)

        cloud.recolor(color_func=color_by_size)
    ax.imshow(cloud, interpolation='bilinear', aspect='auto',
              extent=[0.04, 0.96, 0.05, 0.95])


def add_predictability_bars(fig, rect, entry):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1.14)
    ax.set_ylim(-0.85, 1.85)
    ax.axis('off')
    ax.patch.set_visible(False)

    bar_start = 0.20
    bar_width = 0.42
    half_height = 0.34
    value_x = bar_start + bar_width + 0.05
    values = (
        (1, 'person', HUMAN, float(_entry(entry, 'human_r2', np.nan))),
        (0, 'monkey', MONKEY, float(_entry(entry, 'monkey_r2', np.nan))),
    )
    for y, icon, color, value in values:
        display_value = value if np.isfinite(value) else 0.0
        if abs(display_value) < 0.005:
            display_value = 0.0
        fraction = np.clip(display_value / GAUGE_MAX, 0, 1)
        icon_box = AnnotationBbox(
            OffsetImage(_icon_rgba(icon, color), zoom=ICON_ZOOM),
            (0.075, y), frameon=False, box_alignment=(0.5, 0.5),
            xycoords=ax.transData, clip_on=False)
        ax.add_artist(icon_box)
        ax.add_patch(Rectangle(
            (bar_start, y - half_height), bar_width, 2 * half_height,
            facecolor=TRACK_FILL, edgecolor=TRACK_EDGE, linewidth=0.45))
        if fraction > 0.01:
            ax.add_patch(Rectangle(
                (bar_start, y - half_height), bar_width * fraction,
                2 * half_height, facecolor=color, edgecolor='none'))
        label = f'{display_value:.2f}  R²' if np.isfinite(value) else 'n/a'
        ax.text(value_x, y, label, ha='left', va='center', fontsize=8.6,
                color=INK, fontweight='normal')


def render_factor_cards(entries, image_paths, output_path, *, family='shared', cols=4,
                        mosaic=(3, 3), tile_px=210):
    """Render ranked or curated ``entries`` as approved factor cards."""
    configure_fonts()
    entries = list(entries)
    if not entries:
        raise ValueError('Cannot render an empty NMF factor-card composite.')
    if family not in FAMILY_COLOR:
        raise ValueError(f'Unknown factor-card family: {family}')

    rows = int(np.ceil(len(entries) / cols))
    mosaic_rows, mosaic_cols = mosaic

    # Keep card dimensions stable when callers choose three rather than four
    # columns for smaller curated sets.
    fig_width = 4.05 * cols
    margin_left = 0.36
    margin_right = 0.36
    col_gap = 0.27
    row_gap = 0.16
    top = 0.14
    bottom = 0.14
    card_width = ((fig_width - margin_left - margin_right
                   - (cols - 1) * col_gap) / cols)

    pad = 0.14
    pad_top = 0.09
    pad_bottom = 0.12
    factor_title_height = 0.34
    panel_gap = 0.15
    interior_width = card_width - 2 * pad
    grid_width = interior_width * 0.57
    panel_width = interior_width - grid_width - panel_gap
    tile_width = grid_width / mosaic_cols
    image_height = tile_width * mosaic_rows
    card_height = (pad_top + factor_title_height + image_height + pad_bottom)
    fig_height = (top + rows * card_height + (rows - 1) * row_gap + bottom)

    panel_pad = 0.075
    panel_bottom = 0.045
    bars_height = 0.50
    bars_gap = 0.05
    cloud_gap = 0.055
    family_color = FAMILY_COLOR[family]

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=170)
    fig.patch.set_facecolor('white')
    background_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    background_ax.axis('off')

    def rect(x, y_top, width, height):
        return [x / fig_width, 1 - (y_top + height) / fig_height,
                width / fig_width, height / fig_height]

    def text_y(y_top):
        return 1 - y_top / fig_height

    for position, entry in enumerate(entries):
        row, col = divmod(position, cols)
        x = margin_left + col * (card_width + col_gap)
        y_top = top + row * (card_height + row_gap)
        content_x = x + pad
        content_y = y_top + pad_top + factor_title_height
        factor_number = int(_entry(entry, 'rank_idx', position + 1))
        fig.text((x + card_width / 2) / fig_width,
                 text_y(y_top + pad_top + factor_title_height * 0.52),
                 f'Factor {factor_number}', ha='center', va='center',
                 fontsize=14.5, fontweight='normal', color=INK)

        image_ax = fig.add_axes(rect(content_x, content_y,
                                     grid_width, image_height))
        image_ax.imshow(exemplar_mosaic(
            entry, image_paths, rows=mosaic_rows, cols=mosaic_cols,
            tile_px=tile_px, gap=4), interpolation='lanczos', aspect='auto')
        image_ax.axis('off')
        image_ax.add_patch(Rectangle(
            (0, 0), 1, 1, transform=image_ax.transAxes, fill=False,
            edgecolor=GRID_OUTLINE, linewidth=2.0, clip_on=False))

        panel_x = content_x + grid_width + panel_gap
        background_ax.add_patch(FancyBboxPatch(
            (panel_x / fig_width,
             1 - (content_y + image_height) / fig_height),
            panel_width / fig_width,
            image_height / fig_height,
            boxstyle='round,pad=0,rounding_size=0.010',
            mutation_aspect=fig_height / fig_width,
            facecolor=PANEL_FILL, edgecolor='none',
            transform=fig.transFigure, clip_on=False, zorder=0.6))

        panel_content_width = panel_width - 2 * panel_pad
        bars_y = content_y + image_height - panel_bottom - bars_height
        rule_y = bars_y - bars_gap
        cloud_top = content_y + panel_pad
        cloud_height = rule_y - cloud_gap - cloud_top
        add_wordcloud(
            fig, rect(panel_x + panel_pad, cloud_top,
                      panel_content_width, cloud_height),
            entry, panel_content_width, cloud_height, family_color,
            max_words=5)

        rule_width = panel_content_width * 0.76
        background_ax.add_patch(Rectangle(
            ((panel_x + panel_width / 2 - rule_width / 2) / fig_width,
             text_y(rule_y)),
            rule_width / fig_width, 0.010 / fig_height,
            transform=fig.transFigure, facecolor=PANEL_RULE,
            edgecolor='none', clip_on=False, zorder=0.65))
        add_predictability_bars(
            fig, rect(panel_x + panel_pad, bars_y,
                      panel_content_width, bars_height), entry)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor='white', transparent=False)
    plt.close(fig)
    print(f'[nmf.factor_cards] saved {output_path}')
    return output_path
