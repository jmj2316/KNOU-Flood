import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.environ.get('KNOU_FLOOD_FIGS', ROOT + '/figs'); os.makedirs(FIGS, exist_ok=True)
import glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs, cartopy.feature as cfeat
R = ROOT; OUT = f'{R}/figs'
# --- palette (dataviz reference instance, light mode)
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
C_OURS, C_BASE, C_FUSED = '#2a78d6', '#eb6834', '#1baf7a'          # categorical slots 1,2,3 (entity -> fixed colour)
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})
DIV = LinearSegmentedColormap.from_list('div', ['#8f2121', '#c23b3a', '#e34948', '#ee9a99', '#f6cccb', '#f0efec', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95'][::-1][::-1])
# diverging: blue arm = KNOU-Flood better, red arm = AIFL recipe better ; neutral gray midpoint
DIV = LinearSegmentedColormap.from_list('div', ['#a32828', '#e34948', '#f2a9a8', '#f0efec', '#9ec5f4', '#3987e5', '#184f95'])

def load(names): return [np.load(f'{R}/results/{n}_perbasin.npz') for n in names]
A = load(['s424fsm_s0', 's424fsm_s1', 's424fsm_s2']); B = load(['b2d_gct', 'b2d_s1', 'b2d_s2'])
M1F = load(['b2d_h1024'])[0]; basins = A[0]['basins']
for z in A + B: assert (z['basins'] == basins).all()
def savg(L, k): return np.nanmean([z[k] for z in L], 0)
# --- station coordinates
fs = glob.glob(f'{R}/data/caravan/**/attributes_other_*.csv', recursive=True)
co = pd.concat([pd.read_csv(f)[['gauge_id', 'gauge_lat', 'gauge_lon']] for f in fs]).drop_duplicates('gauge_id').set_index('gauge_id').reindex(basins)
lat, lon = co.gauge_lat.values, co.gauge_lon.values; okc = np.isfinite(lat) & np.isfinite(lon)
print('stations with coordinates:', int(okc.sum()), 'of', len(basins))

# ======================= Fig 1: map of per-station improvement =======================
fig = plt.figure(figsize=(11, 10.4))
for i, (lead, ttl) in enumerate(((1, 'Lead 1 day'), (10, 'Lead 10 days'))):
    ax = fig.add_axes([0.03, 0.535 - i * 0.44, 0.94, 0.355], projection=ccrs.Robinson()); ax.set_global(); ax.set_extent([-170, 180, -58, 80], ccrs.PlateCarree())
    ax.add_feature(cfeat.LAND.with_scale('110m'), facecolor='#f3f2ee', edgecolor='none'); ax.add_feature(cfeat.COASTLINE.with_scale('110m'), edgecolor=MUTED, linewidth=0.4)
    d = savg(A, f'traj{lead}_kge') - savg(B, f'traj{lead}_kge'); m = okc & np.isfinite(d)
    o = np.argsort(np.abs(d[m]))                     # strongest differences drawn last so they stay visible
    sc = ax.scatter(lon[m][o], lat[m][o], c=d[m][o], cmap=DIV, norm=TwoSlopeNorm(0, -0.3, 0.3), s=16, transform=ccrs.PlateCarree(), edgecolors=SURF, linewidths=0.3)
    ax.set_title(f'{ttl}:  Δ KGE per station (KNOU-Flood − baseline)', loc='left', fontsize=12, color=INK, pad=6)
    ax.text(0.01, 0.06, f'KNOU-Flood better at {np.mean(d[m] > 0) * 100:.0f}% of {int(m.sum())} stations\nmedian Δ = {np.median(d[m]):+.3f}', transform=ax.transAxes, fontsize=10, color=INK2,
            bbox=dict(facecolor=SURF, edgecolor='none', alpha=0.85, pad=3))
cax = fig.add_axes([0.32, 0.075, 0.36, 0.013]); cb = fig.colorbar(sc, cax=cax, orientation='horizontal', extend='both'); cb.outline.set_visible(False)
cb.set_label('Δ KGE   (blue: KNOU-Flood better · red: baseline better · clipped at ±0.3)', color=INK2, fontsize=9); cb.ax.tick_params(labelsize=9, length=0)
fig.savefig(f'{FIGS}/fig1_map_delta_kge.png', dpi=300, bbox_inches='tight', pad_inches=0.12); fig.savefig(f'{FIGS}/fig1_map_delta_kge.pdf', bbox_inches='tight', pad_inches=0.12); plt.close(fig)

# ======================= Fig 2: skill vs lead time =======================
leads = np.arange(1, 11); fig, axs = plt.subplots(1, 3, figsize=(13, 4.3)); fig.subplots_adjust(left=0.05, right=0.985, bottom=0.17, top=0.8, wspace=0.22)
def curve(L, m): return np.array([[np.nanmedian(z[f'traj{j}_{m}']) for j in leads] for z in L])
for ax, m, lab in zip(axs, ('kge', 'nse', 'rmse'), ('KGE  (higher is better)', 'NSE  (higher is better)', 'RMSE, mm/day  (lower is better)')):
    ca, cb_ = curve(A, m), curve(B, m); cf = curve([M1F], m)[0]
    ax.fill_between(leads, ca.min(0), ca.max(0), color=C_OURS, alpha=0.18, lw=0); ax.fill_between(leads, cb_.min(0), cb_.max(0), color=C_BASE, alpha=0.18, lw=0)
    ax.plot(leads, cf, color=C_FUSED, lw=2, marker='o', ms=4, mec=SURF, mew=1, zorder=3)
    ax.plot(leads, cb_.mean(0), color=C_BASE, lw=2, marker='o', ms=4, mec=SURF, mew=1, zorder=4); ax.plot(leads, ca.mean(0), color=C_OURS, lw=2.4, marker='o', ms=4.5, mec=SURF, mew=1, zorder=5)
    if m == 'rmse': ax.plot(leads, [np.nanmedian(A[0]['traj1_clim_rmse'])] * 10, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1); ax.text(1.0, np.nanmedian(A[0]['traj1_clim_rmse']) + 0.006, 'climatology', color=MUTED, fontsize=8.5, va='bottom')
    ax.set_xlabel('forecast lead time (days)'); ax.set_title(lab, loc='left', fontsize=11, color=INK); ax.set_xticks(leads); ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlim(0.7, 14.2)
    ax.annotate('KNOU-Flood', (10, ca.mean(0)[-1]), xytext=(6, -5 if m == 'rmse' else 0), textcoords='offset points', color=C_OURS, fontsize=9, va='center', fontweight='bold')
    ax.annotate('Baseline', (10, cb_.mean(0)[-1]), xytext=(6, 0), textcoords='offset points', color=C_BASE, fontsize=9, va='center', fontweight='bold')
    ax.annotate('1024 units', (10, cf[-1]), xytext=(6, 7 if m == 'rmse' else 0), textcoords='offset points', color=C_FUSED, fontsize=9, va='center', fontweight='bold')
axs[0].plot([], [], color=C_FUSED, lw=2, marker='o', ms=4, label='Baseline, 1024 units (1 seed)')
h = [plt.Line2D([], [], color=C_OURS, lw=2.4, marker='o', ms=4.5, label='KNOU-Flood: single LSTM, 424 days, probabilistic, GraphCast-fused (3 seeds, band = min–max)'),
     plt.Line2D([], [], color=C_BASE, lw=2, marker='o', ms=4, label='Baseline (3 seeds, band = min–max)'),
     plt.Line2D([], [], color=C_FUSED, lw=2, marker='o', ms=4, label='Baseline, 1024 units (1 seed)')]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005), labelcolor=INK2)
fig.savefig(f'{FIGS}/fig2_skill_vs_lead.png', dpi=300, bbox_inches='tight', pad_inches=0.12); fig.savefig(f'{FIGS}/fig2_skill_vs_lead.pdf', bbox_inches='tight', pad_inches=0.12); plt.close(fig)

# ======================= Fig 3: per-station paired scatter =======================
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6)); fig.subplots_adjust(left=0.05, right=0.985, bottom=0.14, top=0.8, wspace=0.22)
for ax, lead in zip(axs, (1, 5, 10)):
    x, y = savg(B, f'traj{lead}_kge'), savg(A, f'traj{lead}_kge'); m = np.isfinite(x) & np.isfinite(y); shown = m & (x > -0.6) & (y > -0.6)
    ax.scatter(x[shown], y[shown], s=7, color=C_OURS, alpha=0.32, lw=0)
    ax.plot([-0.6, 1], [-0.6, 1], color=INK2, lw=1); ax.set_xlim(-0.6, 1); ax.set_ylim(-0.6, 1); ax.set_aspect('equal'); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel('Baseline, KGE'); ax.set_ylabel('KNOU-Flood, KGE') if lead == 1 else None
    ax.set_title(f'Lead {lead} day' + ('s' if lead > 1 else ''), loc='left', fontsize=11); 
    ax.text(0.03, 0.97, f'{np.mean(y[m] > x[m]) * 100:.0f}% of stations above the line\nmedian: {np.median(x[m]):.2f} → {np.median(y[m]):.2f}\n({int(m.sum() - shown.sum())} stations below −0.6 not drawn)', transform=ax.transAxes, va='top', fontsize=9.5, color=INK2)
fig.savefig(f'{FIGS}/fig3_paired_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.12); fig.savefig(f'{FIGS}/fig3_paired_scatter.pdf', bbox_inches='tight', pad_inches=0.12); plt.close(fig)
print('done')
