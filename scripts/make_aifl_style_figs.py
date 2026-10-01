"""AIFL-style figures for the paper: Fig. 2 (stations by experimental stage + catchment-area histograms), Fig. 5 (distributions of the performance metrics, LT1),
Fig. 6 (KGE' map + CDF by continent).  KNOU-Flood = mean of seeds s424fsm_s{0,1,2}; baseline = b2d_{gct,s1,s2}; AIFL protocol, LT1."""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, numpy as np, pandas as pd, matplotlib, warnings, sys
warnings.filterwarnings('ignore'); matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeat, cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.strtree import STRtree
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
R = ROOT; OUT = f'{R}/figs'
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'; BLUE, ORG = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})
CONT = {'North America': '#2a78d6', 'Europe': '#eb6834', 'Oceania': '#1baf7a', 'Africa': '#eda100', 'South America': '#e87ba4'}
T = np.load(f'{R}/data/arrays/targets.npz', allow_pickle=True); names = np.array(T['names']); Q = T['Q']; area = T['area']
co = pd.concat([pd.read_csv(f)[['gauge_id', 'gauge_lat', 'gauge_lon']] for f in glob.glob(f'{R}/data/caravan/**/attributes_other_*.csv', recursive=True)]).drop_duplicates('gauge_id').set_index('gauge_id').reindex(names)
lat, lon = co.gauge_lat.values, co.gauge_lon.values
d2i = lambda s: int((np.datetime64(s) - np.datetime64('1980-01-01')).astype(int))
pre = np.isfinite(Q[:, :d2i('2020-01-01')]).sum(1) >= 3 * 365                                   # pre-training pool (observations 1980-2019)
evb = np.load(f'{R}/results/s424fsm_s0_perbasin.npz')['basins']; ev = np.isin(names, evb)       # fine-tuning / test stations
recs = list(shpreader.Reader(shpreader.natural_earth('50m', 'cultural', 'admin_0_countries')).records()); geoms = [r.geometry for r in recs]; cont_of = [r.attributes['CONTINENT'] for r in recs]; tree = STRtree(geoms)
def cont(la, lo):
    if not (np.isfinite(la) and np.isfinite(lo)): return None
    p = Point(lo, la)
    for i in tree.query(p):
        if geoms[i].contains(p): return cont_of[i]
    return cont_of[int(tree.nearest(p))]
def base_map(ax):
    ax.add_feature(cfeat.OCEAN.with_scale('110m'), facecolor='#e8f0f6', edgecolor='none'); ax.add_feature(cfeat.LAND.with_scale('110m'), facecolor='#f3f2ee', edgecolor='none'); ax.add_feature(cfeat.COASTLINE.with_scale('110m'), edgecolor=MUTED, linewidth=0.4)
# ---------------- Fig. 2
fig = plt.figure(figsize=(14, 6.4)); ax = fig.add_axes([0.01, 0.1, 0.99, 0.78], projection=ccrs.Robinson()); ax.set_extent([-170, 180, -58, 80], ccrs.PlateCarree()); base_map(ax)
only_pre = pre & ~ev & np.isfinite(lat); m_ev = ev & np.isfinite(lat)
ax.scatter(lon[only_pre], lat[only_pre], s=3.2, color='#3a8f5c', alpha=0.55, transform=ccrs.PlateCarree(), linewidths=0, label=f'pre-training only ({int(only_pre.sum()):,})')
ax.scatter(lon[m_ev], lat[m_ev], s=9, color='#d1531a', alpha=0.85, transform=ccrs.PlateCarree(), linewidths=0.3, edgecolors='white', label=f'fine-tuning and test ({int(m_ev.sum()):,})')
ax.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=9.5, bbox_to_anchor=(0.02, 0.04), markerscale=3)
for rect, sel, col, ttl in (([0.60, 0.66, 0.16, 0.15], pre & np.isfinite(area), '#3a8f5c', 'pre-training'), ([0.81, 0.66, 0.16, 0.15], ev, '#d1531a', 'fine-tuning / test')):
    ia = fig.add_axes(rect); ia.hist(np.log10(area[sel]), bins=np.linspace(-0.5, 6.7, 40), color=col, alpha=0.85, edgecolor='none'); ia.set_title(ttl, fontsize=8.5, loc='left', pad=2); ia.tick_params(labelsize=7.5); ia.set_xlabel('log$_{10}$ area (km$^2$)', fontsize=8, labelpad=1)
    ia.set_yticks([]); ia.spines['left'].set_visible(False)
fig.savefig(f'{OUT}/fig_stage_map.png', dpi=170, bbox_inches='tight', pad_inches=0.12); plt.close(fig)
# ---------------- Fig. 5 / 6 data
def avg(names_, key): return np.nanmean([np.load(f'{R}/results/{n}_perbasin.npz')[key] for n in names_], 0)
NEW, BASE = ['s424fsm_s0', 's424fsm_s1', 's424fsm_s2'], ['b2d_gct', 'b2d_s1', 'b2d_s2']; bas = np.load(f'{R}/results/s424fsm_s0_perbasin.npz')['basins']
M = {m: (avg(NEW, f'lt1_{m}'), avg(BASE, f'lt1_{m}')) for m in ('kge', 'nse', 'r', 'gamma', 'beta')}
fig, axs = plt.subplots(1, 5, figsize=(17, 3.6)); fig.subplots_adjust(left=0.035, right=0.985, bottom=0.2, top=0.74, wspace=0.28)
for ax, (m, lab, rng_, ideal) in zip(axs, (('kge', "KGE$'$", (-1, 1), 1.0), ('nse', 'NSE', (-1, 1), 1.0), ('r', 'Pearson $r$', (0, 1), 1.0), ('gamma', 'variability ratio $\\gamma$', (0, 2.5), 1.0), ('beta', 'bias ratio $\\beta$', (0, 2.5), 1.0))):
    a, b = M[m]; ok = np.isfinite(a); a2 = a[ok]; b2 = b[np.isfinite(b)]; bins = np.linspace(rng_[0], rng_[1], 41)
    ax.hist(np.clip(a2, *rng_), bins=bins, color='#c9d9ee', edgecolor='none'); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlabel(lab); ax.axvline(ideal, color='#3a8f5c', ls=(0, (4, 3)), lw=1.2)
    ax2 = ax.twinx(); x = np.sort(np.clip(a2, *rng_)); ax2.plot(x, np.arange(1, len(x) + 1) / len(x), color=BLUE, lw=2); xb = np.sort(np.clip(b2, *rng_)); ax2.plot(xb, np.arange(1, len(xb) + 1) / len(xb), color=ORG, lw=1.6, ls='--')
    ax2.set_ylim(0, 1.02); ax2.spines['right'].set_visible(True); ax2.tick_params(labelsize=8); ax.set_xlim(*rng_)
    ax.set_title(f'median {np.nanmedian(a2):.2f}  (baseline {np.nanmedian(b2):.2f})', fontsize=9.5, loc='left', color=INK2)
fig.savefig(f'{OUT}/fig_metric_distributions.png', dpi=170, bbox_inches='tight', pad_inches=0.12); plt.close(fig)
# ---------------- Fig. 6: KGE' map + CDF by continent
idx = np.array([np.where(names == b)[0][0] for b in bas]); kge = M['kge'][0]; cs = np.array([cont(lat[i], lon[i]) for i in idx], dtype=object); cs = np.where(np.isin(cs, list(CONT)), cs, None)
fig = plt.figure(figsize=(15, 5.4)); ax = fig.add_axes([0.01, 0.16, 0.62, 0.68], projection=ccrs.Robinson()); ax.set_extent([-170, 180, -58, 80], ccrs.PlateCarree()); base_map(ax)
ok = np.isfinite(kge) & np.isfinite(lat[idx]); o = np.argsort(kge[ok])
cm = LinearSegmentedColormap.from_list('k', ['#8f2121', '#e34948', '#f2a9a8', '#f0efec', '#9ec5f4', '#3987e5', '#184f95'])
sc = ax.scatter(lon[idx][ok][o], lat[idx][ok][o], c=kge[ok][o], cmap=cm, norm=TwoSlopeNorm(0.5, -0.2, 1.0), s=11, transform=ccrs.PlateCarree(), edgecolors=SURF, linewidths=0.25)
cax = fig.add_axes([0.15, 0.1, 0.34, 0.025]); cb = fig.colorbar(sc, cax=cax, orientation='horizontal', extend='both'); cb.outline.set_visible(False); cb.set_label("KGE$'$ (LT1)", color=INK2, fontsize=9.5)
ca = fig.add_axes([0.69, 0.16, 0.29, 0.68])
for c, col in CONT.items():
    v = np.sort(kge[(cs == c) & np.isfinite(kge)]); 
    if len(v) > 5: ca.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=1.8, label=f'{c} (n={len(v)}, median {np.median(v):.2f})')
v = np.sort(kge[np.isfinite(kge)]); ca.plot(v, np.arange(1, len(v) + 1) / len(v), color='black', lw=2.4, label=f'Global (n={len(v)}, median {np.median(v):.2f})')
ca.set_xlim(-0.5, 1); ca.set_xlabel("KGE$'$"); ca.set_ylabel('CDF'); ca.grid(color=GRID, lw=0.7); ca.legend(fontsize=8, frameon=False, loc='upper left')
fig.savefig(f'{OUT}/fig_kge_map_continent.png', dpi=170, bbox_inches='tight', pad_inches=0.12); plt.close(fig); print('saved')
