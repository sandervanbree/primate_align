# Visualization for CCA category indications and enrichment analysis
import os, sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Wedge
from PIL import Image
from pathlib import Path

from scipy.stats import skew, kurtosis
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from joblib import Parallel, delayed

# Analysis configuration
MAX_COMPONENTS = 10

# Radial plot configuration
RADIAL_SPECIES = 'cross-species'  # 'cross-species', 'human', or 'monkey'
RADIAL_COMP1 = 1  # component index (1-based)
RADIAL_COMP2 = 2  # component index (1-based)
RADIAL_ANGLE_MIN = 325  # degrees
RADIAL_ANGLE_MAX = 35  # degrees

# GOOD CORNERS (SAVED):
# - cross_comp_3_vs_4_radial_corner_190-260
#  - cross_comp_4_vs_6_radial_corner_190-260.pdf
# - monkey_comp_4_vs_5_radial_corner_290-360.pdf
# - human_comp_1_vs_2_radial_corner_290-360.pdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.paths import config
from functions.plotting import (
    setup_style, style_axes, format_axes,
    narrow_figure, medium_figure, wide_figure, square_figure, legend_figure
)
from model.feature.core.ridge_utils import get_ridge_params
from model.feature.core.feature_io import load_vis

# Define colors
COLOR_EXPECTED = '#686868'
COLOR_OBSERVED_FACES = '#C73E73'
COLOR_OBSERVED_BODY = '#902A82'
IMG_BASE_DIR = config.data_dir / 'things' / 'images'

def _analysis_jobs(default=4):
    try:
        return int(config.analysis.get('n_jobs', default))
    except Exception:
        return default

def _default_alpha():
    alphas, _ = get_ridge_params()
    if alphas.size == 0:
        return 1.0
    return float(np.median(alphas))

def _add_img_border(img_arr, thick=8, color=(0,0,0)):
    """Add border around image array."""
    h, w, c = img_arr.shape
    new_h, new_w = h + 2*thick, w + 2*thick
    bordered = np.zeros((new_h, new_w, c), dtype=img_arr.dtype)
    bordered[:] = color
    bordered[thick:thick+h, thick:thick+w] = img_arr
    return bordered

def get_comps(cca_obj):
    """Average component weights across runs."""
    mats = [cca_obj['components'][k] for k in sorted(cca_obj['components'])]
    return np.mean(np.stack(mats, axis=0), axis=0)

def plot_cum(families, stims, label_map, label_name, n_comps=10):
    """Cumulative capture plots with universal dimensions."""
    colors = plt.cm.magma(np.linspace(0.15, 0.85, n_comps))

    figs = {}
    for name, cca in families.items():
        fig = narrow_figure()  # Universal narrow figure
        ax = fig.add_subplot(111)
        comps = get_comps(cca)

        for comp in range(n_comps):
            df = pd.DataFrame({'stim': stims, 'load': np.abs(comps[:, comp])})
            df['is_item'] = df['stim'].map(label_map).fillna(False)
            df = df.sort_values('load', ascending=False)
            total = df['is_item'].sum()
            if total == 0: continue

            x = np.linspace(0, 100, len(df))
            y = np.cumsum(df['is_item']) / total * 100
            ax.plot(x, y, linewidth=3, color=colors[comp], label=str(comp+1))

        ax.plot([0, 100], [0, 100], '--', color='black', alpha=0.9, linewidth=6)
        ax.set_xlabel('Top Loading %')
        ax.set_ylabel(f'% of {label_name}')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)

        style_axes(ax)
        format_axes(ax)
        plt.tight_layout()
        figs[name] = fig

    # Legend figure
    leg_fig = legend_figure(width='medium')
    leg_ax = leg_fig.add_subplot(111)
    leg_ax.axis('off')

    for comp in range(n_comps):
        leg_ax.plot([], [], linewidth=3, color=colors[comp], label=str(comp+1))

    leg_ax.legend(title='Component', loc='center', ncol=4, frameon=False,
                  handlelength=1, handletextpad=0.3, columnspacing=0.6,
                  labelspacing=0.2, borderpad=0.2, borderaxespad=0.1,
                  title_fontsize='large')
    plt.tight_layout()

    return figs, leg_fig

def get_std_loads(comps, n_comps=None):
    """Z-score loadings and average absolute."""
    if n_comps is None:
        n_comps = min(MAX_COMPONENTS, comps.shape[1])
    sub = comps[:,:n_comps]
    std = np.empty_like(sub)
    for i in range(n_comps):
        vals = sub[:,i]
        std[:,i] = (vals-vals.mean())/vals.std()
    avg = np.mean(np.abs(std), axis=1)
    return avg, std

def plot_top_imgs(avg_load, stims, cats, title, n_top=8, img_sz=180, zm=0.14):
    """Bar chart with top images using universal sizing."""
    top_idx = np.argsort(avg_load)[-n_top:][::-1]
    top_stims = [stims[i] for i in top_idx]
    top_vals = avg_load[top_idx]

    fig = medium_figure(dpi=200)
    ax = fig.add_subplot(111)

    pos = np.arange(n_top)
    colors = plt.cm.Greys_r(np.linspace(0.2, 0.8, n_top))
    bars = ax.bar(pos, top_vals, width=0.5, color=colors,
                  edgecolor='black', linewidth=2)

    for bar, i, stim in zip(bars, top_idx, top_stims):
        x = bar.get_x() + bar.get_width()/2
        y = bar.get_height()
        path = IMG_BASE_DIR / cats[i] / f'{stim}.jpg'
        try:
            im = Image.open(path).convert('RGB').resize((img_sz, img_sz), Image.LANCZOS)
            arr = np.array(im)
            box = OffsetImage(arr, zoom=zm)
            ab = AnnotationBbox(box, (x,y), xybox=(0,25), xycoords='data',
                               boxcoords='offset points', pad=0, frameon=True,
                               bboxprops=dict(edgecolor='black', linewidth=4))
            ax.add_artist(ab)
        except:
            ax.text(x, y+0.02, 'NA', ha='center', va='bottom')

    ax.set_xticks(pos)
    ax.set_xlim(-0.5, n_top-0.5)
    ax.set_ylabel('Mean of loadings (Z)')
    ax.set_xlabel('Image Rank')
    ax.set_ylim(0, 2.2)

    style_axes(ax)
    format_axes(ax)
    ax.set_xticklabels([str(i+1) for i in range(n_top)])  # Set labels after styling
    plt.tight_layout()
    return fig, top_idx

def preprocess_features(df, skew_thr=1.0, kurt_thr=3.0,
                        invert_feats=('curvature',),
                        always_log=('spatial_freq',)):
    """Numeric coercion, invert, log, impute, min-max scale."""
    proc=df.apply(pd.to_numeric,errors='coerce').dropna(axis=1,how='all')
    for f in invert_feats:
        if f in proc:
            vals=proc[f].dropna()
            proc[f]=vals.max()-proc[f]
    to_log=[]
    for c in proc:
        vals=proc[c].dropna()
        if len(vals)>1 and (vals>0).all():
            if abs(skew(vals))>skew_thr or kurtosis(vals)>kurt_thr or c in always_log:
                to_log.append(c)
    if to_log:
        proc[to_log]=np.log1p(proc[to_log])
    proc=proc.fillna(proc.median())
    proc=(proc-proc.min())/(proc.max()-proc.min())
    return proc, to_log

def hierarchical_cv_analysis(feats, cat_mat, comps, n_components=None,
                             alpha=None, n_splits=5, n_perms=500):
    """Compute ΔR² & p-values per component."""
    if alpha is None:
        alpha = _default_alpha()
    if n_components is None:
        n_components=min(MAX_COMPONENTS, comps.shape[1])
    cv=KFold(n_splits=n_splits,shuffle=True,random_state=42)
    n_comps=min(n_components,comps.shape[1])
    observed=np.zeros(n_comps)
    pvals=np.zeros(n_comps)
    base_r2=np.zeros(n_comps)
    full_r2=np.zeros(n_comps)
    X_full=np.hstack([feats,cat_mat])
    def proc(c):
        y=comps[:,c]
        b_scores=cross_val_score(Ridge(alpha),feats,y,cv=cv,scoring='r2',n_jobs=1)
        f_scores=cross_val_score(Ridge(alpha),X_full,y,cv=cv,scoring='r2',n_jobs=1)
        b_r2=b_scores.mean(); f_r2=f_scores.mean()
        delta=f_r2-b_r2
        nulls=[]
        for _ in range(n_perms):
            perm=np.random.permutation(len(cat_mat))
            Xp=np.hstack([feats,cat_mat[perm]])
            ps=cross_val_score(Ridge(alpha),Xp,y,cv=cv,scoring='r2',n_jobs=1)
            nulls.append(ps.mean()-b_r2)
        p=np.mean(np.array(nulls)>=delta)
        return delta,p,b_r2,f_r2
    results=Parallel(n_jobs=min(_analysis_jobs(),20))(
        delayed(proc)(c) for c in range(n_comps))
    for i,(d,p,b,f) in enumerate(results):
        observed[i]=d; pvals[i]=p; base_r2[i]=b; full_r2[i]=f
    return observed,pvals,base_r2,full_r2

def compute_enrichment_analysis(comps, feats, labels, label_name,
                                n_components=None, top_pct=0.1):
    """Compute expected vs observed enrichment and ratio per component."""
    if n_components is None:
        n_components=min(MAX_COMPONENTS, comps.shape[1])
    n_stim=len(labels)
    n_top=int(n_stim*top_pct)
    exp_rates=np.zeros(n_components)
    obs_rates=np.zeros(n_components)
    ratios=np.zeros(n_components)
    for c in range(n_components):
        scores=comps[:,c]
        pred=cross_val_predict(Ridge(_default_alpha()),feats,scores,cv=5)
        idx_e=np.argsort(np.abs(pred))[-n_top:]
        idx_o=np.argsort(np.abs(scores))[-n_top:]
        er=labels[idx_e].mean()
        orate=labels[idx_o].mean()
        exp_rates[c]=er; obs_rates[c]=orate
        ratios[c]=orate/er if er>0 else np.nan
    return exp_rates,obs_rates,ratios

def plot_enrich_bars(exp, obs, ratios, title, obs_color, n_disp=None, leg_loc='upper left'):
    """Bar plot of expected vs observed % using universal sizing."""
    if n_disp is None: n_disp = 8
    n = min(len(exp), n_disp, 8)
    exp_pct, obs_pct = exp[:n]*100, obs[:n]*100
    x, w = np.arange(n), 0.35

    fig = medium_figure(dpi=300)
    ax = fig.add_subplot(111)

    ax.bar(x-w/2, exp_pct, w, color=COLOR_EXPECTED, edgecolor='black',
           linewidth=2, label='Expected')
    ax.bar(x+w/2, obs_pct, w, color=obs_color, edgecolor='black',
           linewidth=2, label='Observed')
    ax.set_xlabel('Component')
    ax.set_ylabel('% of Images')
    ax.set_xticks(x)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])

    ax.legend(loc=leg_loc, frameon=False)
    style_axes(ax)
    format_axes(ax)
    ax.set_xticklabels([str(i+1) for i in x])  # Set labels after styling
    plt.tight_layout()
    return fig

def plot_radial(c1, c2, comps, stims, cats, img_dir, n=None, zm=0.5,
                mindist=0.0012, offset=0.1):
    """Plot radial distribution using universal sizing."""
    co = comps[:, [c1, c2]]
    ctr = co.mean(0)
    img_dir = Path(img_dir)

    fig = square_figure()
    ax = fig.add_subplot(111)

    # Distance-based styling
    dists = np.linalg.norm(co - ctr, axis=1)
    dist_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
    colors = plt.cm.Greys(0.35 + 0.2 * dist_norm)

    ax.scatter(co[:,0], co[:,1], s=40, c=colors, alpha=1.0,
               edgecolors='none', marker='o', linewidth=0)

    if n and n < len(co):
        d = np.linalg.norm(co - ctr, axis=1)
        th = np.arctan2(co[:,1]-ctr[1], co[:,0]-ctr[0])
        bins = np.linspace(-np.pi, np.pi, n+1)
        bidx = np.digitize(th, bins) - 1
        sel, used = [], set()
        for idx in np.argsort(d)[::-1]:
            b = bidx[idx]
            if b in used: continue
            if not sel or np.min(np.linalg.norm(co[idx]-co[sel], axis=1)) >= mindist:
                sel.append(idx)
                used.add(b)
            if len(sel) >= n: break
        if len(sel) < n:
            for idx in np.argsort(d)[::-1]:
                if idx in sel: continue
                if np.min(np.linalg.norm(co[idx]-co[sel], axis=1)) >= mindist:
                    sel.append(idx)
                if len(sel) >= n: break
        idxs = sel
    else:
        idxs = list(range(len(co)))

    x0, x1 = co[:,0].min(), co[:,0].max()
    y0, y1 = co[:,1].min(), co[:,1].max()
    dr = max(x1-x0, y1-y0)
    od = offset * dr

    for i in idxs:
        v = co[i] - ctr
        dist = np.linalg.norm(v)
        u = v/dist if dist>0 else np.array([1.,0.])
        pt = co[i] + u * od

        ax.plot([co[i,0], pt[0]], [co[i,1], pt[1]], color='0.2', alpha=0.8, lw=4.5)

        path = img_dir / cats[i] / f"{stims[i]}.jpg"
        try:
            im = Image.open(path).convert("RGB").resize((280,280), Image.BILINEAR)
            im_arr = np.array(im)
            thick = max(8, 280 // 30)
            im_arr = _add_img_border(im_arr, thick=thick)
            oi = OffsetImage(im_arr, zoom=zm/2)
            ab = AnnotationBbox(oi, pt, frameon=False, pad=0.1)
            ax.add_artist(ab)
        except:
            pass

    sel_pts = co[idxs]
    ax.scatter(sel_pts[:,0], sel_pts[:,1], s=80, color='0.2', alpha=0.9,
               edgecolors='none', marker='o')

    ax.set_xlim(x0 - od, x1 + od)
    ax.set_ylim(y0 - od, y1 + od)
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_radial_corner(c1, c2, comps, stims, cats, img_dir, n=None, zm=0.5,
                       mindist=0.0008, offset=0.05, ang_lims=None,
                       scat_scale=1.0):
    """Advanced radial plot with angular regions using universal sizing."""
    co = comps[:, [c1, c2]]
    ctr = co.mean(0)
    img_dir = Path(img_dir)
    co_s = (co - ctr) * scat_scale + ctr

    th = np.degrees(np.arctan2(co[:,1] - ctr[1], co[:,0] - ctr[0]))

    if ang_lims is not None:
        deg_min, deg_max = ang_lims
        deg_min = ((deg_min + 180) % 360) - 180
        deg_max = ((deg_max + 180) % 360) - 180
        if deg_min <= deg_max:
            region_mask = (th >= deg_min) & (th <= deg_max)
        else:
            region_mask = (th >= deg_min) | (th <= deg_max)
    else:
        region_mask = np.ones(len(co), dtype=bool)

    if n and n < len(co):
        d = np.linalg.norm(co - ctr, axis=1)
        bins = np.linspace(-np.pi, np.pi, n+1)
        bidx = np.digitize(np.radians(th), bins) - 1
        sel, used = [], set()
        for idx in np.argsort(d)[::-1]:
            b = bidx[idx]
            if b in used: continue
            if not sel or np.min(np.linalg.norm(co[idx]-co[sel], axis=1)) >= mindist:
                sel.append(idx)
                used.add(b)
            if len(sel) >= n: break
        if len(sel) < n:
            for idx in np.argsort(d)[::-1]:
                if idx in sel: continue
                if np.min(np.linalg.norm(co[idx]-co[sel], axis=1)) >= mindist:
                    sel.append(idx)
                if len(sel) >= n: break
        idxs = sel
    else:
        idxs = list(range(len(co)))

    sel = [i for i in idxs if region_mask[i]]
    if not sel:
        raise ValueError("No points in angular window.")

    ordered = [i for i,_ in sorted(zip(sel, th[sel]), key=lambda x: x[1])]
    m = len(ordered)

    r_all = np.linalg.norm(co - ctr, axis=1)
    r_max = r_all.max()
    full_dr = max(co[:,0].max()-co[:,0].min(), co[:,1].max()-co[:,1].min())
    circ_r = r_max * scat_scale + offset * full_dr * 0.6

    if ang_lims:
        deg_min, deg_max = ang_lims
    else:
        deg_min, deg_max = -180, 180
    span = (deg_max - deg_min) if deg_min <= deg_max else (deg_max +360 - deg_min)

    # Use a wider display range for better figure utilization
    display_span = 120  # degrees - wider arc for displaying images
    display_center = (deg_min + deg_max) / 2 if deg_min <= deg_max else ((deg_min + deg_max + 360) / 2) % 360
    display_min = display_center - display_span / 2
    display_max = display_center + display_span / 2

    if m > 1:
        angs_out = [display_min + display_span * (k/(m-1)) for k in range(m)]
    else:
        angs_out = [display_center]

    fig = square_figure()
    ax = fig.add_subplot(111)

    # Build per-point colors: grey scale based on distance for all points
    dists_s = np.linalg.norm(co_s - ctr, axis=1)
    dist_norm = (dists_s - dists_s.min()) / (dists_s.max() - dists_s.min() + 1e-8)

    # 1) grey-scale non-highlighted points, behind everything
    ax.scatter(
        co_s[:,0], co_s[:,1],
        s=18,
        c=plt.cm.Greys(0.35 + 0.2 * dist_norm),
        edgecolors='none',
        marker='o',
        linewidth=0,
        zorder=1
    )

    # 2) wedge in the middle
    if ang_lims:
        # Mix base #F66C5C [0.965,0.424,0.361] 50/50 with grey [0.5,0.5,0.5] => less saturation
        base_rgb = mpl.colors.to_rgb('#F66C5C')
        mixed_rgb = (
            (base_rgb[0] + 0.5) / 2,
            (base_rgb[1] + 0.5) / 2,
            (base_rgb[2] + 0.5) / 2
        )
        wedge_less_sat = (*mixed_rgb, 0.55)  # alpha ≈0.55 for more intensity

        wedge_radius = circ_r + offset * full_dr * 0.5
        purple_wedge = Wedge(
            center=tuple(ctr),
            r=wedge_radius,
            theta1=deg_min, theta2=deg_max,
            facecolor=wedge_less_sat,
            edgecolor=wedge_less_sat[:3] + (0.8,),
            linestyle='--', linewidth=2,
            zorder=2
        )
        ax.add_patch(purple_wedge)

    # 3) thumbnail connectors & images
    for i, ang in zip(ordered, angs_out):
        rad = np.radians(ang)
        thumb_xy = ctr + np.array([np.cos(rad), np.sin(rad)]) * circ_r

        # thinner lines, drawn on top of wedge
        ax.plot(
            [thumb_xy[0], co_s[i,0]],
            [thumb_xy[1], co_s[i,1]],
            color='0.2', alpha=0.8, lw=2.5,
            zorder=3
        )

        path = img_dir / cats[i] / f"{stims[i]}.jpg"
        try:
            im = Image.open(path).convert("RGB").resize((220,220), Image.BILINEAR)
            im_arr = np.array(im)
            thick = max(7, 220 // 30)
            im_arr = _add_img_border(im_arr, thick=thick)
            oi = OffsetImage(im_arr, zoom=zm/2.5)
            ab = AnnotationBbox(oi, thumb_xy, frameon=False, pad=0.09, zorder=4)
            ax.add_artist(ab)
        except Exception:
            pass

    # 4) highlighted dots, smallest z-order above images
    sel_pts = co_s[ordered]
    ax.scatter(
        sel_pts[:,0], sel_pts[:,1],
        s=60,                   # a bit smaller
        color='0.2', alpha=0.9,
        edgecolors='none', marker='o',
        zorder=5
    )

    pad = offset * full_dr
    ax.set_xlim(ctr[0] - circ_r - pad, ctr[0] + circ_r + pad)
    ax.set_ylim(ctr[1] - circ_r - pad, ctr[1] + circ_r + pad)

    ax.axis('off')
    plt.tight_layout()
    return fig

def save_cat_fig(fig, fig_type, analysis_family, extra_info=""):
    """Save category indication figure with structured path and naming."""
    if not fig: return
    fig_dir = Path(config.fig_dir) / 'cat_indications' / analysis_family / fig_type
    fig_dir.mkdir(parents=True, exist_ok=True)
    fmt = config.plotting.get('savefig_format', 'png')
    fname = f"{extra_info}.{fmt}" if extra_info else f"{fig_type}.{fmt}"
    path = fig_dir / fname
    dpi = 600 if fig_type == 'images' else 300
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    print(f"    Saved to {path}")
    plt.close(fig)

# --- Main execution ---
if __name__=="__main__":
    # Setup plotting style
    setup_style()

    # Override with custom style for all figures
    plt.rcParams.update({
        'figure.constrained_layout.use': False,
        'figure.autolayout': False
    })

    # Ensure figure directory exists
    (Path(config.fig_dir) / 'cat_indications').mkdir(parents=True, exist_ok=True)

    # Load CCA state
    state_fp = Path(config.results_dir) / 'cca_state.pkl'
    if not os.path.exists(state_fp):
        raise FileNotFoundError(f"CCA state file not found: {state_fp}")

    print(f"Loading cached CCA results from {state_fp}...")
    with open(state_fp,'rb') as f:
        state=pickle.load(f)

    cca_all, cca_hum, cca_mon = state['cca_all'], state['cca_hum'], state['cca_mon']
    all_stims = state['all_stims']
    all_cats  = state.get('all_cats',
                          ['_'.join(s.split('_')[:-1]) if '_' in s else s
                           for s in all_stims])

    print("Loaded cached CCA results successfully.")
    print(f"Available keys in CCA state: {list(state.keys())}")

    # Load face/body maps from annotations (repo copy preferred)
    annot_fp = Path(config.get_annotations_path())

    face_map = {}
    clothing_map = {}

    if os.path.exists(annot_fp):
        annot_df = pd.read_csv(annot_fp)
        annot_df['stimulus'] = annot_df['filename'].str.replace('.jpg', '')

        # Face trials: face=1 AND (human=1 OR monkey=1)
        face_mask = (annot_df['face'] == 1) & ((annot_df['human'] == 1) | (annot_df['monkey'] == 1))
        face_stims = set(annot_df.loc[face_mask, 'stimulus'])
        face_map = {stim: (stim in face_stims) for stim in all_stims}

        # Body trials: body_part=1 AND (human=1 OR monkey=1)
        body_mask = (annot_df['body_part'] == 1) & ((annot_df['human'] == 1) | (annot_df['monkey'] == 1))
        body_stims = set(annot_df.loc[body_mask, 'stimulus'])
        clothing_map = {stim: (stim in body_stims) for stim in all_stims}

        print(f"Face labels loaded: {sum(face_map.values())} images")
        print(f"Body labels loaded: {sum(clothing_map.values())} images")
    else:
        print(f"Warning: Annotations file not found at {annot_fp}")

    families={'cross-species':cca_all,'human':cca_hum,'monkey':cca_mon}

    # --- Cumulative plots ---
    print("--- Generating cumulative capture plots ---")

    if face_map:
        print("  Generating face cumulative capture plots...")
        figs, legend_fig = plot_cum(families,all_stims,face_map,'Faces')
        for fam_name, fig in figs.items():
            save_cat_fig(fig,'cumulative',fam_name,f'faces_cumulative_{fam_name}')
        save_cat_fig(legend_fig,'cumulative','all','faces_legend')

    if clothing_map:
        print("  Generating body/clothing cumulative capture plots...")
        figs, legend_fig = plot_cum(families,all_stims,clothing_map,'Bodies')
        for fam_name, fig in figs.items():
            save_cat_fig(fig,'cumulative',fam_name,f'clothing_cumulative_{fam_name}')
        save_cat_fig(legend_fig,'cumulative','all','clothing_legend')

    # --- Standardized loadings & top images ---
    print("--- Generating standardized loadings analysis ---")

    for fam_name,cca in families.items():
        print(f"  Analyzing {fam_name} components...")
        comps=get_comps(cca)
        avg,_=get_std_loads(comps,n_comps=MAX_COMPONENTS)
        fig,_idx=plot_top_imgs(avg,all_stims,all_cats,
                               f'{fam_name.title()} CCA - Top Images')
        save_cat_fig(fig,'top_images',fam_name,f'standardized_loadings_{fam_name}')

    # --- Load visual features for enrichment analysis ---
    print("--- Loading visual features for enrichment analysis ---")
    try:
        visual_features, vis_names, _ = load_vis(all_stims, use_proc=True)
    except FileNotFoundError as err:
        print(f"Visual features file not found. Skipping enrichment analysis. ({err})")
    else:
        visual_features = np.asarray(visual_features, dtype=np.float32)
        print(f"Loaded visual feature matrix: {visual_features.shape[0]} stims × {visual_features.shape[1]} features")

        # Create category matrix
        unique_cats = sorted(set(all_cats))
        cat_matrix = np.zeros((len(all_stims), len(unique_cats)))
        cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
        for i, cat in enumerate(all_cats):
            cat_matrix[i, cat_to_idx[cat]] = 1
        print(f"Category matrix shape: {cat_matrix.shape}")

        # --- Hierarchical CV analysis with caching ---
        print("--- Loading/saving hierarchical CV analysis ---")
        cv_results_dir = Path(config.results_dir) / 'cat_indications'
        cv_results_dir.mkdir(parents=True, exist_ok=True)
        cv_results_fp = cv_results_dir / 'cv_results.pkl'

        if cv_results_fp.exists():
            print(f"  Loading cached CV results from {cv_results_fp}...")
            with open(cv_results_fp, 'rb') as f:
                cv_results = pickle.load(f)
            print(f"  Loaded CV results for families: {list(cv_results.keys())}")
        else:
            print("  No cached results found. Running hierarchical CV analysis...")
            cv_results = {}
            for family_name, family_comps in [('cross-species', get_comps(cca_all)), ('human', get_comps(cca_hum)), ('monkey', get_comps(cca_mon))]:
                print(f"    Analyzing {family_name} family...")
                deltas, p_vals, base_r2, full_r2 = hierarchical_cv_analysis(
                    visual_features, cat_matrix, family_comps,
                    n_components=MAX_COMPONENTS, n_perms=50  # Reduced for speed
                )
                cv_results[family_name] = {
                    'deltas': deltas,
                    'p_values': p_vals,
                    'base_r2': base_r2,
                    'full_r2': full_r2
                }
            # Save results for future use
            print(f"  Saving CV results to {cv_results_fp}...")
            with open(cv_results_fp, 'wb') as f:
                pickle.dump(cv_results, f)

        # --- Enrichment analysis ---
        print("--- Running enrichment analysis ---")
        if face_map and clothing_map:
            n_stims = len(all_stims)
            face_labels = np.array([face_map.get(stim, False) for stim in all_stims])
            body_labels = np.array([clothing_map.get(stim, False) for stim in all_stims])
            print(f"  Face images: {face_labels.sum()}/{n_stims}")
            print(f"  Body/clothing images: {body_labels.sum()}/{n_stims}")

            for family_name, family_comps in [('cross-species', get_comps(cca_all))]:  # Focus on cross-species for enrichment
                print(f"    Analyzing {family_name} family enrichment...")

                face_expected, face_observed, face_ratios = compute_enrichment_analysis(
                    family_comps, visual_features, face_labels, "Faces", n_components=MAX_COMPONENTS)
                body_expected, body_observed, body_ratios = compute_enrichment_analysis(
                    family_comps, visual_features, body_labels, "Bodies", n_components=MAX_COMPONENTS)

                print(f"    Generating enrichment plots...")
                fig_faces = plot_enrich_bars(face_expected, face_observed, face_ratios,
                                             "Faces", COLOR_OBSERVED_FACES, leg_loc='upper right')
                save_cat_fig(fig_faces, 'enrichment', family_name, 'faces_enrichment')

                fig_body = plot_enrich_bars(body_expected, body_observed, body_ratios,
                                            "Bodies", COLOR_OBSERVED_BODY, leg_loc='upper left')
                save_cat_fig(fig_body, 'enrichment', family_name, 'body_enrichment')

        # --- Radial corner plot ---
        print("--- Generating radial corner visualization ---")

        # Component indices for radial plots (convert from 1-based to 0-based)
        comp1_idx, comp2_idx = RADIAL_COMP1 - 1, RADIAL_COMP2 - 1

        # Create corner-highlighted radial plot only for specified species
        if RADIAL_SPECIES in families:
            print(f"  Creating corner-highlighted plot for {RADIAL_SPECIES}...")
            cca = families[RADIAL_SPECIES]
            fig_corner = plot_radial_corner(
                comp1_idx, comp2_idx, get_comps(cca), all_stims, all_cats,
                IMG_BASE_DIR,
                n=60, zm=0.35, ang_lims=(RADIAL_ANGLE_MIN, RADIAL_ANGLE_MAX), scat_scale=0.6
            )
            save_cat_fig(fig_corner, 'radial', RADIAL_SPECIES, f'comp_{RADIAL_COMP1}_vs_{RADIAL_COMP2}_radial_corner_{RADIAL_ANGLE_MIN}-{RADIAL_ANGLE_MAX}')
        else:
            print(f"  Warning: {RADIAL_SPECIES} not found in families. Skipping corner plot.")

        print(f"\nCategory indications analysis complete! All figures saved to {Path(config.fig_dir) / 'cat_indications'}")
