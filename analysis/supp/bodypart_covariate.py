#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from analysis.category import things_bodyparts as bp


def main() -> None:
    scores, label, stims, cats = bp.load_bodypart_scores()
    if scores is None:
        raise RuntimeError("body-part dimension not found")
    out_path = config.fig_dir / "supp" / "category" / f"bodypart_covariate.{config.plotting.get('savefig_format', 'pdf')}"
    bp.plot_bodypart_covariate(scores, stims, cats, out_path, label or "Body part")
    print(f"[supp.bodypart_covariate] saved {out_path}")


if __name__ == "__main__":
    main()
