#!/usr/bin/env python3
"""Shared rendering style for supplementary CCA and sNMF galleries.

The supplementary galleries remain dense, one-row-per-factor/pole overviews.
This module only owns their presentation: a neutral information panel, category
word cloud, icon-based species bars, and outlined exemplar strip.  Ranking,
selection, and summary generation stay with the CCA and NMF analysis modules.
"""

from __future__ import annotations

import re
from pathlib import Path
import sys
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analysis.nmf.viz import factor_cards as FC

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None


HUMAN = FC.HUMAN
MONKEY = FC.MONKEY
INK = FC.INK
INK_SOFT = FC.INK_SOFT
PANEL_FILL = FC.PANEL_FILL
TRACK_FILL = FC.TRACK_FILL
TRACK_EDGE = FC.TRACK_EDGE

N_IMAGES = 20
TILE = 150
TILE_GAP = 4
ROW_H = TILE + 2 * TILE_GAP
LABEL_W_NMF = 235
LABEL_W_CCA = 225
BAR_W = 285
WORDCLOUD_W = 440
INFO_GAP = 8
IMAGE_GAP = 22
ROW_GAP = 16
POLE_GAP = 4
AXIS_GAP = 26
PANEL_DPI = 300
HEADER_H = 150
HEADER_GAP = 22


def _rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(round(channel * 255)) for channel in to_rgb(color))


def _fig_to_array(fig: plt.Figure, width: int) -> np.ndarray:
    fig.canvas.draw()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    array = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
    array = array.reshape(canvas_height, canvas_width, 4)[..., :3]
    plt.close(fig)
    if (canvas_width, canvas_height) != (width, ROW_H):
        array = np.asarray(
            Image.fromarray(array).resize((width, ROW_H), Image.Resampling.BILINEAR)
        )
    return array


def _blank_panel(width: int) -> np.ndarray:
    return np.tile(np.array(_rgb(PANEL_FILL), np.uint8), (ROW_H, width, 1))


def _render_nmf_label(rank: int, component: int) -> np.ndarray:
    fig = plt.figure(
        figsize=(LABEL_W_NMF / PANEL_DPI, ROW_H / PANEL_DPI), dpi=PANEL_DPI
    )
    fig.patch.set_facecolor(PANEL_FILL)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5, 0.62, f"Factor {rank}", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=INK,
    )
    ax.text(
        0.5, 0.30, f"(raw {component})", ha="center", va="center",
        fontsize=8.5, color=INK_SOFT,
    )
    return _fig_to_array(fig, LABEL_W_NMF)


def _render_cca_label(component: int, pole: str, pole_color: str) -> np.ndarray:
    fig = plt.figure(
        figsize=(LABEL_W_CCA / PANEL_DPI, ROW_H / PANEL_DPI), dpi=PANEL_DPI
    )
    fig.patch.set_facecolor(PANEL_FILL)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5, 0.66, f"Comp {component}", ha="center", va="center",
        fontsize=13, fontweight="bold", color=INK,
    )
    ax.text(
        0.5, 0.29, pole.title(), ha="center", va="center",
        fontsize=11, fontweight="bold", color=pole_color,
    )
    return _fig_to_array(fig, LABEL_W_CCA)


def _render_species_bars(
    human_value: float,
    monkey_value: float,
    scale: float,
) -> np.ndarray:
    fig = plt.figure(figsize=(BAR_W / PANEL_DPI, ROW_H / PANEL_DPI), dpi=PANEL_DPI)
    fig.patch.set_facecolor(PANEL_FILL)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])
    ax.axis("off")
    ax.patch.set_visible(False)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.85, 1.85)

    bar_x0, bar_width, half_height = 0.33, 0.34, 0.36
    value_x = bar_x0 + bar_width + 0.06
    values = (
        (1, "person", HUMAN, human_value),
        (0, "monkey", MONKEY, monkey_value),
    )
    for y, icon, color, value in values:
        finite = np.isfinite(value)
        display_value = float(value) if finite else 0.0
        if abs(display_value) < 0.005:
            display_value = 0.0
        fraction = np.clip(display_value / scale, 0, 1)
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(FC._icon_rgba(icon, color), zoom=0.030),
                (0.18, y), frameon=False, box_alignment=(0.5, 0.5),
                xycoords=ax.transData, clip_on=False,
            )
        )
        ax.add_patch(
            Rectangle(
                (bar_x0, y - half_height), bar_width, 2 * half_height,
                facecolor=TRACK_FILL, edgecolor=TRACK_EDGE, linewidth=0.45,
            )
        )
        if fraction > 0.01:
            ax.add_patch(
                Rectangle(
                    (bar_x0, y - half_height), bar_width * fraction,
                    2 * half_height, facecolor=color, edgecolor="none",
                )
            )
        ax.text(
            value_x, y, f"{display_value:.2f}" if finite else "n/a",
            ha="left", va="center", fontsize=8.8, color=INK,
        )
    return _fig_to_array(fig, BAR_W)


def _render_wordcloud(
    enrichment: list[tuple[str, float]],
    color: str,
) -> np.ndarray:
    items = [
        (re.sub(r"\d+$", "", category).replace("_", " ").strip(), float(auc))
        for category, auc in enrichment
        if auc > 0.5
    ][:5]
    if not items or WordCloud is None:
        return _blank_panel(WORDCLOUD_W)

    frequencies = {word: max(auc - 0.49, 0.02) for word, auc in items}
    cloud_width = int(WORDCLOUD_W * 0.90)
    cloud_height = int(ROW_H * 0.82)
    cloud = WordCloud(
        width=cloud_width,
        height=cloud_height,
        font_path=FC.FONT_PATH,
        background_color=None,
        mode="RGBA",
        prefer_horizontal=1.0,
        relative_scaling=0.5,
        margin=2,
        min_font_size=20,
        random_state=42,
    ).generate_from_frequencies(frequencies)

    font_sizes = [size for (_word, size, _position, _orientation, _color) in cloud.layout_]
    if font_sizes:
        minimum, maximum = min(font_sizes), max(font_sizes)

        def color_by_size(word, font_size, **kwargs):
            strength = (font_size - minimum) / (maximum - minimum + 1e-9)
            red, green, blue = FC._size_color(color, strength)
            return int(255 * red), int(255 * green), int(255 * blue)

        cloud.recolor(color_func=color_by_size)

    panel = Image.new("RGBA", (WORDCLOUD_W, ROW_H), _rgb(PANEL_FILL) + (255,))
    panel.alpha_composite(
        Image.fromarray(cloud.to_array(), "RGBA"),
        ((WORDCLOUD_W - cloud_width) // 2, (ROW_H - cloud_height) // 2),
    )
    return np.asarray(panel.convert("RGB"))


def _nmf_image_strip(order: np.ndarray, image_paths: list[Path]) -> np.ndarray:
    width = N_IMAGES * TILE + (N_IMAGES + 1) * TILE_GAP
    canvas = Image.new("RGB", (width, ROW_H), FC.OUTLINE_RGB)
    for position, image_index in enumerate(order[:N_IMAGES]):
        tile = FC._load_square(str(image_paths[int(image_index)]), TILE)
        canvas.paste(tile, (TILE_GAP + position * (TILE + TILE_GAP), TILE_GAP))
    return np.asarray(canvas)


def _cca_image_strip(
    order: np.ndarray,
    image_paths: list[Path],
    pole_color: str,
    outline_width: int = 7,
) -> np.ndarray:
    tile_size = ROW_H - 2 * outline_width
    width = N_IMAGES * tile_size + (N_IMAGES + 1) * outline_width
    canvas = Image.new("RGB", (width, ROW_H), _rgb(pole_color))
    for position, image_index in enumerate(order[:N_IMAGES]):
        tile = FC._load_square(str(image_paths[int(image_index)]), tile_size)
        canvas.paste(
            tile,
            (outline_width + position * (tile_size + outline_width), outline_width),
        )
    return np.asarray(canvas)


def _render_header(width: int, title: str, subtitle: str) -> np.ndarray:
    fig = plt.figure(figsize=(width / PANEL_DPI, HEADER_H / PANEL_DPI), dpi=PANEL_DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.006, 0.66, title, ha="left", va="center",
        fontsize=15, fontweight="bold", color=INK,
    )
    ax.text(
        0.006, 0.24, subtitle, ha="left", va="center",
        fontsize=9.5, color=INK_SOFT,
    )
    fig.canvas.draw()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    array = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
    array = array.reshape(canvas_height, canvas_width, 4)[..., :3]
    plt.close(fig)
    if (canvas_width, canvas_height) != (width, HEADER_H):
        array = np.asarray(
            Image.fromarray(array).resize((width, HEADER_H), Image.Resampling.BILINEAR)
        )
    return array


def _save_pdf(canvas: np.ndarray, output_path: Path, dpi: int) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = canvas.shape[:2]
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(canvas, interpolation="none")
    ax.axis("off")
    fig.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def render_nmf_supp_gallery(
    entries: list[dict],
    image_paths: list[Path],
    family: str,
    title: str,
    output_path: Path,
    dpi: int = 400,
) -> None:
    """Render a dense one-row-per-factor sNMF supplementary gallery."""
    FC.configure_fonts()
    if not entries:
        raise ValueError("Cannot render an empty sNMF supplementary gallery.")

    family_color = FC.FAMILY_COLOR[family]
    info_separator = np.full((ROW_H, INFO_GAP, 3), _rgb(PANEL_FILL), np.uint8)
    image_separator = np.full((ROW_H, IMAGE_GAP, 3), 255, np.uint8)
    rows = []
    for entry in entries:
        order = np.argsort(entry["weights"])[::-1][:N_IMAGES]
        info = np.hstack([
            _render_nmf_label(entry["rank_idx"], entry["component"]),
            info_separator,
            _render_species_bars(entry["human_r2"], entry["monkey_r2"], FC.GAUGE_MAX),
            info_separator,
            _render_wordcloud(entry["enrichment"], family_color),
        ])
        rows.append(np.hstack([info, image_separator, _nmf_image_strip(order, image_paths)]))

    width = rows[0].shape[1]
    row_separator = np.full((ROW_GAP, width, 3), 255, np.uint8)
    body_parts = []
    for index, row in enumerate(rows):
        body_parts.append(row)
        if index < len(rows) - 1:
            body_parts.append(row_separator)
    body = np.vstack(body_parts)

    rank_text = {
        "shared": "Ranked by shared predictability",
        "human": "Ranked by human predictability",
        "monkey": "Ranked by macaque predictability",
    }[family]
    subtitle = (
        f"{rank_text}  ·  20 highest-loading exemplars, category word cloud, "
        "and human / macaque R² per factor"
    )
    header = _render_header(width, title, subtitle)
    header_gap = np.full((HEADER_GAP, width, 3), 255, np.uint8)
    _save_pdf(np.vstack([header, header_gap, body]), output_path, dpi)


def render_cca_supp_gallery(
    entries: list[dict],
    image_paths: list[Path],
    family: str,
    mode: str,
    select_images: Callable[[dict, str], tuple[np.ndarray, int]],
    output_path: Path,
    positive_color: str,
    negative_color: str,
    dpi: int = 400,
) -> None:
    """Render a dense two-poles-per-axis CCA supplementary gallery."""
    FC.configure_fonts()
    if not entries:
        raise ValueError("Cannot render an empty CCA supplementary gallery.")

    finite_values = [
        float(entry[key])
        for entry in entries
        for key in ("human_crossview_r", "monkey_crossview_r")
        if np.isfinite(entry[key])
    ]
    maximum = max(finite_values) if finite_values else 0.8
    bar_scale = max(0.5, min(1.0, np.ceil(maximum * 20) / 20))

    info_separator = np.full((ROW_H, INFO_GAP, 3), _rgb(PANEL_FILL), np.uint8)
    image_separator = np.full((ROW_H, IMAGE_GAP, 3), 255, np.uint8)
    rows: list[tuple[np.ndarray, str]] = []
    for entry in entries:
        pole_color = positive_color if entry["pole"] == "positive" else negative_color
        order, _ = select_images(entry, mode)
        info = np.hstack([
            _render_cca_label(entry["component"], entry["pole"], pole_color),
            info_separator,
            _render_species_bars(
                entry["human_crossview_r"], entry["monkey_crossview_r"], bar_scale
            ),
            info_separator,
            _render_wordcloud(entry["enrichment"], pole_color),
        ])
        row = np.hstack([
            info,
            image_separator,
            _cca_image_strip(order, image_paths, pole_color),
        ])
        rows.append((row, entry["pole"]))

    width = rows[0][0].shape[1]
    pole_separator = np.full((POLE_GAP, width, 3), 255, np.uint8)
    axis_separator = np.full((AXIS_GAP, width, 3), 255, np.uint8)
    body_parts = []
    for index, (row, pole) in enumerate(rows):
        body_parts.append(row)
        if index < len(rows) - 1:
            body_parts.append(pole_separator if pole == "positive" else axis_separator)
    body = np.vstack(body_parts)

    family_label = {
        "cross-species": "Cross-species",
        "human": "Human",
        "monkey": "Monkey",
    }[family]
    selection_text = (
        "20 top exemplars" if mode == "normal"
        else "20 exemplars sampled from the top 1%"
    )
    subtitle = (
        f"Positive (red) and negative (blue) pole per axis  ·  {selection_text}, "
        "category word cloud, and human / macaque cross-view r"
    )
    header = _render_header(width, f"{family_label} signed CCA axes", subtitle)
    header_gap = np.full((HEADER_GAP, width, 3), 255, np.uint8)
    _save_pdf(np.vstack([header, header_gap, body]), output_path, dpi)
