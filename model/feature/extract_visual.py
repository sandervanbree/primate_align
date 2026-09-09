#!/usr/bin/env python3
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from model.feature.core.vis_props_build import main as compute_vis_props

def extract_visual_features():
    """Extract and save visual properties to vis_props.tsv."""
    compute_vis_props()

if __name__ == '__main__':
    extract_visual_features()
