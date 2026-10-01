import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, numpy as np, pandas as pd, matplotlib, warnings, sys
warnings.filterwarnings('ignore'); matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, BoundaryNorm
import cartopy.crs as ccrs, cartopy.feature as cfeat, cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.strtree import STRtree
from scipy.stats import wasserstein_distance
sys.path.insert(0, ROOT + '/src'); import data as D
R = ROOT; OUT = f'{R}/figs'
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})
# continent -> colour, fixed entity mapping (categorical slots 1..6)
CONT = {'North America': '#2a78d6', 'Europe': '#eb6834', 'Oceania': '#1baf7a', 'Africa': '#eda100', 'South America': '#e87ba4', 'Asia': '#008300'}
T = np.load(f'{R}/data/arrays/targets.npz', allow_pickle=True); names = np.array(T['names']); Q = T['Q']
fs = glob.glob(f'{R}/data/caravan/**/attributes_other_*.csv', recursive=True)
co = pd.concat([pd.read_csv(f)[['gauge_id', 'gauge_lat', 'gauge_lon']] for f in fs]).drop_duplicates('gauge_id').set_index('gauge_id').reindex(names)
lat, lon = co.gauge_lat.values, co.gauge_lon.values
# --- continent of every station: point-in-polygon on Natural Earth countries (nearest polygon for coastal misses)
recs = list(shpreader.Reader(shpreader.natural_earth('50m', 'cultural', 'admin_0_countries')).records())
geoms = [r.geometry for r in recs]; cont_of = [r.attributes['CONTINENT'] for r in recs]; tree = STRtree(geoms)
def cont(la, lo):
    if not (np.isfinite(la) and np.isfinite(lo)): return None
    p = Point(lo, la)
    for i in tree.query(p):
        if geoms[i].contains(p): return cont_of[i]
    return cont_of[int(tree.nearest(p))]
cs = np.array([cont(a, b) for a, b in zip(lat, lon)], dtype=object); cs = np.where(np.isin(cs, list(CONT)), cs, None)
print('continent counts (all stations):', {k: int((cs == k).sum()) for k in CONT}, 'unassigned', int(pd.isna(cs).sum()))
evb = np.load(f'{R}/results/s424fsm_s0_perbasin.npz')['basins']; ev = np.array([np.where(names == b)[0][0] for b in evb])

# ======================= Fig A: observed stations per year by continent =======================
years = np.arange(1980, 2025); d0 = lambda y: D.d2i(f'{y}-01-01'); 
def counts(sel):
    M = np.zeros((len(CONT), len(years)), int)
    for j, y in enumerate(years):
        e = min(D.d2i(f'{y + 1}-01-01'), Q.shape[1]); ok = np.isfinite(Q[sel][:, d0(y):e]).sum(1) >= 180
        for i, c in enumerate(CONT): M[i, j] = int((ok & (cs[sel] == c)).sum())
    return M
fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[1.35, 1]), sharey=False); fig.subplots_adjust(left=0.06, right=0.985, bottom=0.22, top=0.8, wspace=0.16)
for ax, sel, ttl in ((axs[0], np.arange(len(names)), f'All stations used for training ({len(names):,})'), (axs[1], ev, f'Evaluation stations ({len(ev):,})')):
    M = counts(sel); bot = np.zeros(len(years))
    ax.axvspan(1980, 2020, color='#e6f4e6', lw=0, zorder=0); ax.axvspan(2020, 2021, color='#dcebf7', lw=0, zorder=0); ax.axvspan(2021, 2024, color='#fde6d3', lw=0, zorder=0)
    for i, (c, col) in enumerate(CONT.items()):
        ax.bar(years + 0.5, M[i], bottom=bot, width=0.94, color=col, edgecolor=SURF, linewidth=0.5, label=c, zorder=2); bot += M[i]
    ax.set_xlim(1980, 2025); ax.set_title(ttl, loc='left', fontsize=11); ax.set_xlabel('year'); ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.text(0.0, -0.001, '', transform=ax.transAxes)
axs[0].set_ylabel('stations with ≥180 observed days in the year')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower left', ncol=6, frameon=False, fontsize=9.5, bbox_to_anchor=(0.05, 0.0), labelcolor=INK2, columnspacing=1.6)
import matplotlib.patches as mp
fig.legend(handles=[mp.Patch(color='#e6f4e6', label='train 1980–2019'), mp.Patch(color='#dcebf7', label='validation 2020'), mp.Patch(color='#fde6d3', label='test 2021–2023 (2024 partial)')], loc='lower left', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.05, 0.055), labelcolor=INK2, columnspacing=1.6)
fig.text(0.06, 0.965, 'Streamflow observations available over time', fontsize=14, fontweight='bold', va='top'); fig.text(0.06, 0.905, 'Caravan v1.6 + GRDC-Caravan stations; the drop after 2018 is why only ≈2,100 stations can be scored in the test years', fontsize=10, color=INK2, va='top')
fig.savefig(f'{OUT}/figA_gauges_per_year.png', dpi=200); plt.close(fig)

# ======================= helper: map + CDF-by-continent =======================
def map_cdf(values, sel, fname, title, sub, cmap, norm, cbar_label, xlab, ext, xlim, legend_loc='lower right'):
    fig = plt.figure(figsize=(14, 5.3)); ax = fig.add_axes([0.02, 0.17, 0.62, 0.66], projection=ccrs.Robinson()); ax.set_extent([-170, 180, -58, 80], ccrs.PlateCarree())
    ax.add_feature(cfeat.OCEAN.with_scale('110m'), facecolor='#e8f0f6', edgecolor='none'); ax.add_feature(cfeat.LAND.with_scale('110m'), facecolor='#f3f2ee', edgecolor='none'); ax.add_feature(cfeat.COASTLINE.with_scale('110m'), edgecolor=MUTED, linewidth=0.4)
    v = values; m = np.isfinite(v) & np.isfinite(lat[sel]) & np.isfinite(lon[sel]); o = np.argsort(v[m])
    sc = ax.scatter(lon[sel][m][o], lat[sel][m][o], c=v[m][o], cmap=cmap, norm=norm, s=16, transform=ccrs.PlateCarree(), edgecolors=SURF, linewidths=0.3)
    cax = fig.add_axes([0.14, 0.09, 0.4, 0.028]); cb = fig.colorbar(sc, cax=cax, orientation='horizontal', extend=ext); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9, length=0); cb.set_label(cbar_label, color=INK2, fontsize=9.5)
    ax2 = fig.add_axes([0.70, 0.17, 0.28, 0.66]); allv = v[m]
    for c, col in CONT.items():
        x = np.sort(v[m & (cs[sel] == c)]) if False else np.sort(v[(cs[sel] == c) & np.isfinite(v)])
        if len(x) >= 10: ax2.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, lw=1.6, label=f'{c} (n={len(x):,})')
    xa = np.sort(allv); ax2.plot(xa, np.arange(1, len(xa) + 1) / len(xa), color=INK, lw=2.6, label=f'Global (n={len(xa):,})', zorder=5)
    md, p90 = np.median(allv), np.percentile(allv, 90); ax2.axvline(md, color=MUTED, lw=1, ls=(0, (4, 3))); ax2.text(md, 0.02, f' median\n {md:.3f}', color=INK2, fontsize=8.5, va='bottom')
    ax2.set_xlim(*xlim); ax2.set_ylim(0, 1.02); ax2.set_xlabel(xlab); ax2.set_ylabel('CDF'); ax2.grid(color=GRID, lw=0.8); ax2.set_axisbelow(True); ax2.legend(frameon=False, fontsize=8.5, loc=legend_loc, labelcolor=INK2)
    fig.text(0.02, 0.965, title, fontsize=14, fontweight='bold', va='top'); fig.text(0.02, 0.905, sub, fontsize=10, color=INK2, va='top')
    fig.savefig(f'{OUT}/{fname}', dpi=200); plt.close(fig)
    return md, p90

# ======================= Fig B: ERA5-Land vs HRES precipitation distance (same diagnostic as the AIFL paper) =======================
rE = D.rows_for(list(names[ev]), 'ERA5_LAND'); rH = D.rows_for(list(names[ev]), 'HRES')
era = D.load_rows('ERA5_LAND', 'total_precipitation', rE); hres = D.load_rows('HRES', 'total_precipitation', rH)[:, :, 0]      # lead-1 forecast valid on each day
ds = pd.date_range('2016-01-01', '2023-12-31'); iE = np.array([D.d2i(d.strftime('%Y-%m-%d')) for d in ds[[0, -1]]]); nE = iE[1] - iE[0] + 1; nH = len(ds)
e = era[:, iE[0]:iE[0] + nE]; h = hres[:, :nH]
w1 = np.array([wasserstein_distance(e[i], h[i]) / max(e[i].mean(), 1e-6) for i in range(len(ev))])       # W1 / mean daily precipitation (ERA5-Land)
print('W1 median', np.median(w1), 'P90', np.percentile(w1, 90))
cmapW = LinearSegmentedColormap.from_list('w', ['#fdeee2', '#f6b48f', '#ee7f56', '#dc4a4a', '#a3306c', '#55285f', '#231a3a'])
mdW, p90W = map_cdf(w1, ev, 'figB_precip_distance_map_cdf.png', 'How different are the reanalysis and the forecast precipitation? (ERA5-Land vs IFS HRES, lead 1)',
                    f'Normalized Wasserstein distance W₁ / mean daily precipitation, daily totals 2016–2023, n = {len(ev):,} stations', cmapW, BoundaryNorm([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0], cmapW.N, extend='max'),
                    'normalized Wasserstein distance (W₁)', 'W₁', 'max', (0, 0.8))

# ======================= Fig C: result map + CDF by continent =======================
def sav(names_, k): return np.nanmean([np.load(f'{R}/results/{n}_perbasin.npz')[k] for n in names_], 0)
DIV = LinearSegmentedColormap.from_list('div', ['#a32828', '#e34948', '#f2a9a8', '#f0efec', '#9ec5f4', '#3987e5', '#184f95'])
for lead in (1, 10):
    d = sav(['s424fsm_s0', 's424fsm_s1', 's424fsm_s2'], f'traj{lead}_kge') - sav(['b2d_gct', 'b2d_s1', 'b2d_s2'], f'traj{lead}_kge')
    map_cdf(d, np.arange(len(ev)) if False else ev, f'figC_dkge_lead{lead}_map_cdf.png', f'Skill gain over the AIFL-style baseline, lead {lead} day' + ('s' if lead > 1 else ''),
            'Δ KGE′ = KNOU-Flood − baseline, median of 3 seeds, test 2021–2023; CDF curves to the right of 0 mean KNOU-Flood is better', DIV, TwoSlopeNorm(0, -0.3, 0.3),
            'Δ KGE′   (blue: KNOU-Flood better · red: baseline better)', 'Δ KGE′', 'both', (-0.5, 0.9), legend_loc='upper left')
print('done')
