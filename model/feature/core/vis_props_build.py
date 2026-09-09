#!/usr/bin/env python3
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from joblib import Parallel, delayed
import cv2
from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.paths import config

VIS_COLS = ['contrast','colorfulness','spatial_freq','texture','spiky_stubby','curvature','size']

def _rms_contrast(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.sqrt(np.mean((g - g.mean())**2)))

def _colorfulness(img):
    b,g,r = cv2.split(img.astype(np.float32))
    rg = r - g; yb = 0.5*(r + g) - b
    s = np.sqrt(np.var(rg) + np.var(yb))
    m = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    return float(s + 0.3*m)

def _spatial_freq(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.var(cv2.Laplacian(g, cv2.CV_64F)))

def _texture(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(sx**2 + sy**2)))


def _spiky_stubby(seg_path: str) -> float:
    if not os.path.exists(seg_path):
        return np.nan
    mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.nan
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 4:
        return np.nan
    try:
        hull = ConvexHull(coords)
        hull_area = max(hull.volume, 1.0)
        actual = float(np.sum(bw > 0))
        return float(1.0 / (actual / hull_area))
    except Exception:
        return np.nan

def _process(img_path: str):
    nm = os.path.splitext(os.path.basename(img_path))[0]
    im = cv2.imread(img_path)
    if im is None:
        return {'image': nm, 'contrast': np.nan, 'colorfulness': np.nan, 'spatial_freq': np.nan, 'texture': np.nan, 'spiky_stubby': np.nan, 'curvature': np.nan, 'size': np.nan}

    # Segmented object masks
    seg_path = os.path.join(config.data_dir, 'things', 'segmented_img', f"{nm}.jpg")

    return {
        'image': nm,
        'contrast': _rms_contrast(im),
        'colorfulness': _colorfulness(im),
        'spatial_freq': _spatial_freq(im),
        'texture': _texture(im),
        'spiky_stubby': _spiky_stubby(seg_path),
        'curvature': np.nan,  # Will be loaded separately
        'size': np.nan,  # Will be loaded separately
    }

def _rank_proc(df: pd.DataFrame) -> pd.DataFrame:
    x = df.apply(pd.to_numeric, errors='coerce')
    # optional log1p for skewed positives (skip spiky_stubby as it's already normalized)
    to_log=[]
    for c in x.columns:
        if c == 'spiky_stubby': continue  # Already in [0,1] range
        v = x[c].dropna()
        if len(v)>1 and (v>0).all() and (abs(skew(v))>1 or kurtosis(v)>3):
            to_log.append(c)
    if to_log:
        x[to_log] = np.log1p(x[to_log])
    x = x.fillna(x.median(numeric_only=True))
    for c in x.columns:
        x[c] = rankdata(x[c], method='average') / len(x)
    return x

def main(n_jobs: int = None):
    n_jobs = int(n_jobs or config.analysis.get('n_jobs', 8))
    img_root = Path(config.data_dir)/'things'/'images'
    out_fp = Path(config.data_dir)/'features'/'vis_props'/'vis_props.tsv'
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    img_paths: List[str] = []
    for cat in sorted(os.listdir(img_root)):
        p = img_root/cat
        if p.is_dir():
            for f in os.listdir(p):
                if f.endswith('.jpg'):
                    img_paths.append(str(p/f))

    rows = Parallel(n_jobs=n_jobs)(delayed(_process)(p) for p in img_paths)
    df = pd.DataFrame(rows).set_index('image')

    # Load curvature ratings
    curv_paths = [
        Path(config.data_dir)/'things'/'curvature-ratings.csv',
    ]
    curv = {}
    for path in curv_paths:
        if path.exists():
            try:
                curv_df = pd.read_csv(path)
                for _, row in curv_df.iterrows():
                    fn = row['image_filename']
                    base = os.path.basename(fn).replace('.jpg', '') if isinstance(fn, str) else str(fn).replace('.jpg', '')
                    curv[base] = float(row['Answer.curvature'])
                break
            except Exception:
                continue
    df['curvature'] = [curv.get(s, np.nan) for s in df.index]
    if np.isfinite(df['curvature']).sum() > 0:
        mx = np.nanmax(df['curvature'])
        df['curvature'] = mx - df['curvature']

    # Load size estimates
    size_paths = [
        Path(config.data_dir)/'things'/'size_fixed.csv',
    ]
    size_map = {}
    for path in size_paths:
        if path.exists():
            try:
                sd = pd.read_csv(path, sep=';')
                size_map = {str(r['word']).strip(): float(r['meanSize']) for _, r in sd.iterrows()}
                break
            except Exception:
                continue

    # Map categories to sizes
    cats = [s.split('_')[0] if '_' in s else s for s in df.index]
    df['size'] = [size_map.get(c, np.nan) for c in cats]

    # Process all 7 features
    raw_cols = ['contrast','colorfulness','spatial_freq','texture','spiky_stubby','curvature','size']

    # Handle missing values with median imputation for spiky_stubby
    if df['spiky_stubby'].isna().any():
        median_spiky = df['spiky_stubby'].median()
        if np.isfinite(median_spiky):
            df['spiky_stubby'] = df['spiky_stubby'].fillna(median_spiky)
            print(f"Imputed {df['spiky_stubby'].isna().sum()} missing spiky_stubby values with median: {median_spiky:.4f}")

    proc = _rank_proc(df[raw_cols]).add_suffix('_proc')
    out = pd.concat([df, proc], axis=1).reset_index()
    out.to_csv(out_fp, sep='\t', index=False)
    print(f"Saved {len(out)} rows → {out_fp}")

    # Print summary of spiky_stubby values
    spiky_vals = df['spiky_stubby'].dropna()
    if len(spiky_vals) > 0:
        print(f"Spiky_stubby stats: mean={spiky_vals.mean():.4f}, std={spiky_vals.std():.4f}, min={spiky_vals.min():.4f}, max={spiky_vals.max():.4f}")
        print(f"Non-zero spiky_stubby values: {(spiky_vals > 0).sum()}/{len(spiky_vals)}")

if __name__=='__main__':
    main()
