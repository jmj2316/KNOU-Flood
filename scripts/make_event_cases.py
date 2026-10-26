"""Flood-event case studies: for a chosen event, every 10-day forecast issued from 9 days before to 1 day after the observed peak,
colored by issue day, with observed and model-simulated return levels.  Events are drawn AT RANDOM (seeded) from all test-period observed peaks
>= the observed 10-yr level, one per region -- never selected on model skill.  Point forecast = ensemble mean of KNOU-Flood (seed 1) and, beside it, the baseline (seed 1).
usage: make_event_cases.py [seed_tag s1] """
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.environ.get('KNOU_FLOOD_FIGS', ROOT + '/figs'); os.makedirs(FIGS, exist_ok=True)
import sys, glob, numpy as np, pandas as pd, matplotlib, matplotlib.dates as mdates, warnings
warnings.filterwarnings('ignore'); matplotlib.use('Agg')
import matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeat
sys.path.insert(0, ROOT + '/scripts'); import event_metrics as EM
R = ROOT; TAG = sys.argv[1] if len(sys.argv) > 1 else 's1'
A = np.load(f'{R}/results/dump_evt_s424_{TAG}.npz'); B = np.load(f'{R}/results/dump_evt_b2d_{TAG}.npz')
assert (A['basins'] == B['basins']).all() and (A['dates'] == B['dates']).all()
basins = list(A['basins']); dates = A['dates']; n = len(dates); obs = A['obs']
T = np.load(f'{R}/data/arrays/targets.npz', allow_pickle=True); tn = list(T['names']); Qall = T['Q']; alld = np.datetime64('1980-01-01') + np.arange(Qall.shape[1])
rd = lambda kind: pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/data/caravan/**/attributes_{kind}_*.csv', recursive=True)]).drop_duplicates('gauge_id').set_index('gauge_id')
oth = rd('other').reindex(basins); area = oth.area.values; lat = oth.gauge_lat.values; lon = oth.gauge_lon.values
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'; C_OURS, C_BASE, C_SIM = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})
# ---------- candidate events (observations only)
W = 12; cands = []
for g in range(len(basins)):
    if not (np.isfinite(area[g]) and 200 <= area[g] <= 60000 and np.isfinite(lat[g])): continue
    thr = EM.thresholds(Qall[tn.index(basins[g])], alld)
    if thr is None: continue
    o = obs[g]; hit = np.where(np.isfinite(o) & (o >= thr[3]))[0]; hit = hit[(hit >= W) & (hit < n - W)]
    best = None
    for i in hit:
        w = o[i - W:i + W + 1]
        if np.isfinite(w).all() and o[i] == np.nanmax(o[max(0, i - 10):i + 11]) and (best is None or o[i] > o[best]): best = i
    if best is not None and all(np.isfinite(A[f'mean{j}'][g, best - W + 1:best + W]).sum() > 15 for j in (1, 5, 10)): cands.append((g, best, thr))
print('candidate stations with an observed >=10-yr peak in the test window:', len(cands))
def region(g):
    if -12 <= lon[g] <= 40 and 34 <= lat[g] <= 72: return 'Europe'
    if -170 <= lon[g] <= -50 and 15 <= lat[g] <= 75: return 'North America'
    return 'Rest of world'
rng = np.random.default_rng(7); picks = []
for reg in ('Europe', 'North America', 'Rest of world'):
    c = [x for x in cands if region(x[0]) == reg]; print(f'  {reg}: {len(c)} candidates')
    if c: picks.append(c[rng.integers(len(c))])
# ---------- figure
cmB, cmO = matplotlib.colormaps['Blues'], matplotlib.colormaps['Oranges']; col = lambda cm, d: cm(0.25 + 0.75 * (d + 9) / 10)
fig = plt.figure(figsize=(17, 5.6 * len(picks) + 1.6)); gs = fig.add_gridspec(len(picks), 3, width_ratios=[0.8, 2.3, 2.3], left=0.03, right=0.935, top=0.86, bottom=0.085, wspace=0.2, hspace=0.46)
to_cms = lambda q, g: q * area[g] / 86.4                                       # mm/d -> m3/s
def place_labels(ax, items, yl):                                             # de-overlapped right-edge labels for return-level lines
    items = sorted(items); ys = []; gap = 0.042 * (yl[1] - yl[0])
    for y, t, c in items: ys.append(y if not ys else max(y, ys[-1] + gap))
    for (y, t, c), yy in zip(items, ys): ax.text(1.008, yy, t, transform=ax.get_yaxis_transform(), fontsize=8, color=c, va='center', ha='left', clip_on=False)
for r, (g, pk, thr) in enumerate(picks):
    sc = area[g] / 86.4; ts = lambda x: x * sc; pdate = dates[pk]; d_idx = np.arange(pk - W, pk + W + 1)
    sim = {nm: EM.thresholds(Z['era_sim'][g], alld[int(Z['era_lo']):int(Z['era_lo']) + Z['era_sim'].shape[1]]) for nm, Z in (('A', A), ('B', B))}   # each model has its own ERA5-driven record
    # map
    ax = fig.add_subplot(gs[r, 0], projection=ccrs.PlateCarree()); ax.set_extent([lon[g] - 9, lon[g] + 9, lat[g] - 6, lat[g] + 6], ccrs.PlateCarree())
    ax.add_feature(cfeat.OCEAN.with_scale('50m'), facecolor='#d8e7f0', edgecolor='none'); ax.add_feature(cfeat.LAND.with_scale('50m'), facecolor='#f3f2ee', edgecolor='none')
    ax.add_feature(cfeat.BORDERS.with_scale('110m'), edgecolor=MUTED, linewidth=0.4); ax.add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor=MUTED, linewidth=0.4)
    ax.scatter([lon[g]], [lat[g]], s=70, color='#d1531a', edgecolors='white', linewidths=1.0, transform=ccrs.PlateCarree(), zorder=5)
    ax.set_title(f"{str(oth.gauge_name.values[g]).title()}\n{oth.country.values[g]} · {area[g]:,.0f} km²\npeak {str(pdate)} · {region(g)}", loc='left', fontsize=9.5, color=INK, pad=5)
    ymax = 0; axs = []
    for k, (nm, Z, cm, who) in enumerate((('A', A, cmB, 'KNOU-Flood (ensemble mean)'), ('B', B, cmO, 'Baseline'))):
        ax = fig.add_subplot(gs[r, 1 + k], sharey=axs[0] if k else None); axs.append(ax); dd = dates[d_idx].astype('datetime64[ns]')
        ax.plot(dd, ts(obs[g][d_idx]), color=INK, lw=2.0, ls=(0, (5, 2)), label='Observed', zorder=6)
        fc_peak = {}
        for d in range(-9, 2):
            t = pk + d; js = [j for j in range(1, 11) if 0 <= t + j < n and np.isfinite(Z[f'mean{j}'][g, t + j])]
            if len(js) < 2: continue
            x = dates[[t + j for j in js]].astype('datetime64[ns]'); y = ts(np.array([Z[f'mean{j}'][g, t + j] for j in js]))
            ax.plot(x, y, color=col(cm, d), lw=1.9, zorder=3, label=None); ax.plot(x[:1], y[:1], 'o', ms=3.2, color=col(cm, d), zorder=4)
            ymax = max(ymax, np.nanmax(y))
        for j in (1, 3, 5, 7): fc_peak[j] = Z[f'mean{j}'][g, pk]
        ymax = max(ymax, np.nanmax(ts(obs[g][d_idx])))
        ax.grid(axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d')); plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8.5)
        ax.set_title(who, loc='left', fontsize=10.5, color=INK, pad=17)
        ax.text(0.0, 1.012, 'forecast / observed at the peak day, issued 1 / 3 / 5 / 7 days ahead:  ' + ' / '.join(f'{fc_peak[j] / obs[g][pk]:.2f}' for j in (1, 3, 5, 7)),
                transform=ax.transAxes, fontsize=8.6, color=INK2, va='bottom', ha='left')
        if k == 0: ax.set_ylabel('Streamflow (m³ s⁻¹)')
        else: ax.tick_params(labelleft=False)
    top = 1.12 * ymax
    for k, (nm, ax) in enumerate(zip('AB', axs)):
        ax.set_ylim(0, top); items = []
        for i, rp in enumerate(EM.RPS):
            if rp not in (2, 5, 20): continue
            for v, c, lab in ((thr[i], MUTED, f'Obs {rp:g}-yr'), (sim[nm][i] if sim[nm] is not None else np.nan, C_SIM, f'Sim {rp:g}-yr')):
                y = ts(v) if np.isfinite(v) else np.nan
                if np.isfinite(y) and y <= top: ax.axhline(y, color=c, lw=1.0, ls=(0, (4, 3)), zorder=1); items.append((y, lab, c))
        place_labels(ax, items, (0, top))

from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=INK, lw=2, ls=(0, (5, 2)), label='Observed'), Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3)), label='Observed return levels (Gumbel)'),
                    Line2D([], [], color=C_SIM, lw=1, ls=(0, (4, 3)), label='Return levels of the model own ERA5-driven simulation'), Line2D([], [], color=C_OURS, lw=2, label='Final-model forecasts'),
                    Line2D([], [], color=C_BASE, lw=2, label='Baseline forecasts')], loc='upper center', bbox_to_anchor=(0.485, 0.925), ncol=5, fontsize=9, frameon=False)
# colour bars (issue day relative to the observed peak)
for k, (cm, x0) in enumerate(((cmB, 0.27), (cmO, 0.62))):
    cax = fig.add_axes([x0, 0.028, 0.23, 0.011]); sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-9.5, 1.5), cmap=matplotlib.colors.ListedColormap([col(cm, d) for d in np.linspace(-9, 1, 11)]))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=range(-9, 2, 2)); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=8.5)
    cb.set_label('Issue day relative to the observed peak (day 0)', fontsize=8.8, color=INK2)
fig.savefig(f'{FIGS}/figF_event_cases.png', dpi=300, bbox_inches='tight', pad_inches=0.12); fig.savefig(f'{FIGS}/figF_event_cases.pdf', bbox_inches='tight', pad_inches=0.12)
print('saved', [(basins[g], str(dates[pk])) for g, pk, _ in picks])
