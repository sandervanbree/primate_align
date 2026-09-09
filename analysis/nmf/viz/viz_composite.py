#!/usr/bin/env python3
"""Main-text sNMF factor-card composites.

The shared display is the first twelve factors in ridge-recoverability order.
Human- and macaque-derived displays use manuscript-curated factor ranks, which
are resolved to their raw component identifiers at render time.  All families
use the same renderer and visual assets.

Examples
--------
All default composites (shared 12, human 8, macaque 8):
    python analysis/nmf/viz/viz_composite.py

Ranked human selection:
    python analysis/nmf/viz/viz_composite.py --families human --top 8

Explicit raw-component selection:
    python analysis/nmf/viz/viz_composite.py \
        --families human monkey \
        --custom human:98,11,16,86,74,5,35,76 \
        --custom monkey:18,14,30,15,22,74,69,40
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from analysis.nmf.gallery import GALLERY_SPECS, compute_enrichment
from analysis.nmf.utils import (
    build_img_paths,
    derive_categories,
    load_guard_table,
    load_srf_results,
    pick_rank,
)
from analysis.nmf.viz.factor_cards import render_factor_cards


# Ranked factor numbers in the supplementary gallery order.  These are kept as
# factor ranks rather than raw component identifiers so the card labels remain
# directly traceable to the full supplementary displays.
CURATED_FACTOR_RANKS = {
    'all': [],
    'human': [1, 28, 13, 15, 3, 9, 12, 36],
    'monkey': [1, 6, 8, 31, 10, 22, 24, 25],
}

DEFAULT_TOP = {'all': 12, 'human': 8, 'monkey': 8}
DEFAULT_COLS = {'all': 4, 'human': 4, 'monkey': 4}
DISPLAY_FAMILY = {'all': 'shared', 'human': 'human', 'monkey': 'monkey'}
SPEC_BY_FAMILY = {spec['family']: spec for spec in GALLERY_SPECS.values()}


def parse_custom(values):
    custom = {family: [] for family in CURATED_FACTOR_RANKS}
    for value in values or []:
        if ':' not in value:
            raise ValueError(f"Custom selection must be FAMILY:ID,ID,...; got {value!r}")
        family, raw_ids = value.split(':', 1)
        family = family.strip().lower()
        if family not in custom:
            raise ValueError(f'Unknown family in custom selection: {family}')
        ids = [int(token.strip()) for token in raw_ids.replace(';', ',').split(',')
               if token.strip()]
        if not ids:
            raise ValueError(f'No component identifiers supplied for {family}.')
        custom[family] = ids
    return custom


def ranked_component_ids(family, guard, top_n):
    spec = SPEC_BY_FAMILY[family]
    selected = guard[spec['filter'](guard)].copy()
    if selected.empty:
        return []
    selected['score'] = selected.apply(spec['score'], axis=1)
    selected.sort_values('score', ascending=False, inplace=True)
    return selected['component'].astype(int).tolist()[:int(top_n)]


def build_entries(family, component_ids, W, guard, cats, cat_counts,
                  factor_numbers=None):
    if factor_numbers is None:
        factor_numbers = range(1, len(component_ids) + 1)
    entries = []
    for factor_number, component_id in zip(factor_numbers, component_ids):
        component_id = int(component_id)
        component_index = component_id - 1
        if component_index < 0 or component_index >= W.shape[1]:
            raise ValueError(
                f'{family}: component {component_id} is outside 1..{W.shape[1]}.')
        matches = guard[guard['component'] == component_id]
        if matches.empty:
            raise ValueError(f'{family}: component {component_id} missing from guard table.')
        row = matches.iloc[0]
        weights = np.asarray(W[:, component_index], float)
        entries.append({
            'rank_idx': factor_number,
            'component': component_id,
            'weights': weights,
            'enrichment': compute_enrichment(weights, cats, cat_counts),
            'human_r2': float(row.get('human_r2', np.nan)),
            'monkey_r2': float(row.get('monkey_r2', np.nan)),
        })
    return entries


def build(families=('all', 'human', 'monkey'), top=None, cols=None, custom_values=None,
          mosaic=(3, 3), tile_px=210):
    custom = parse_custom(custom_values)
    results, stimuli = load_srf_results()
    image_paths = build_img_paths(stimuli)
    categories = derive_categories(stimuli)
    category_counts = Counter(categories)

    outputs = []
    for family in families:
        guard = load_guard_table(family)
        component_ids = custom.get(family) or []
        factor_numbers = None

        if not component_ids:
            curated_ranks = CURATED_FACTOR_RANKS.get(family) or []
            if curated_ranks:
                ranked_ids = ranked_component_ids(family, guard, max(curated_ranks))
                component_ids = [ranked_ids[rank - 1] for rank in curated_ranks]
                factor_numbers = curated_ranks
            else:
                component_ids = ranked_component_ids(
                    family, guard, top if top is not None else DEFAULT_TOP[family])

        rank = pick_rank(family, results[family])
        W = np.asarray(results[family]['components'][rank], float)
        entries = build_entries(
            family, component_ids, W, guard, categories, category_counts,
            factor_numbers=factor_numbers)

        family_cols = int(cols) if cols is not None else DEFAULT_COLS[family]
        out_dir = Path(config.fig_dir) / 'nmf_analysis' / family / 'composite'
        pdf = out_dir / 'composite.pdf'
        render_factor_cards(
            entries,
            image_paths,
            pdf,
            family=DISPLAY_FAMILY[family],
            cols=family_cols,
            mosaic=mosaic,
            tile_px=tile_px,
        )
        print(
            f'[viz_composite] {family}: rank={rank}, '
            f'factors={list(factor_numbers) if factor_numbers is not None else "ranked"}, '
            f'components={component_ids}, output={pdf}')
        outputs.append(pdf)
    return outputs


def main():
    parser = argparse.ArgumentParser(description='Render main-text sNMF factor-card composites.')
    parser.add_argument('--families', nargs='+', default=['all', 'human', 'monkey'],
                        choices=['all', 'human', 'monkey'])
    parser.add_argument('--top', type=int, default=None,
                        help='Override the ranked prefix size for every requested family.')
    parser.add_argument('--cols', type=int, default=None,
                        help='Factor-card columns (default: 4).')
    parser.add_argument('--custom', action='append', default=[],
                        help='Explicit raw component IDs, e.g. human:98,11,16. Repeatable.')
    parser.add_argument('--mosaic', default='3x3',
                        help='Exemplar grid per factor, e.g. 3x3 or 2x6.')
    parser.add_argument('--tile-px', type=int, default=210)
    args = parser.parse_args()

    try:
        mosaic = tuple(int(value) for value in args.mosaic.lower().split('x'))
    except Exception as exc:
        raise ValueError(f'Invalid mosaic specification: {args.mosaic!r}') from exc
    if len(mosaic) != 2 or min(mosaic) < 1:
        raise ValueError(f'Invalid mosaic specification: {args.mosaic!r}')

    build(
        families=args.families,
        top=args.top,
        cols=args.cols,
        custom_values=args.custom,
        mosaic=mosaic,
        tile_px=args.tile_px,
    )


if __name__ == '__main__':
    main()
