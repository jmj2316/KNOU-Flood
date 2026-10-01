import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np, matplotlib, warnings, sys
warnings.filterwarnings('ignore'); matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
sys.path.insert(0, ROOT + '/src'); from metrics import kge_prime, rmse, nse
R = ROOT
import re
def nice(n):
    t = n.title(); return re.sub(r', ([A-Z][a-z])\b', lambda m: ', ' + m.group(1).upper(), t)
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'; C_OURS, C_AIFL = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})
import json as _json; sys.path.insert(0, f'{R}/scripts'); import probmetrics as PM
_lam = _json.load(open(f'{R}/results/prob_eval.json'))['lam']
F = np.load(f'{R}/results/dump_pm_s424mc_s1_test.npz'); B = np.load(f'{R}/results/dump_pm_b2d_s1_test.npz'); cases = {c['id']: c for c in json.load(open(f'{R}/results/cases.json'))}
dates = F['dates']; sel = (dates >= np.datetime64('2022-01-01')) & (dates <= np.datetime64('2022-12-31')); dd = dates[sel].astype('datetime64[ns]')
order = ['GRDC_4105850', 'GRDC_5869401', 'GRDC_4151513', 'GRDC_4125940']          # snow-dominated, humid rain-fed, arid, large basin
LEADS = (1, 7)
fig, axs = plt.subplots(4, 2, figsize=(13, 12.2), sharex=True); fig.subplots_adjust(left=0.06, right=0.985, bottom=0.06, top=0.83, hspace=0.42, wspace=0.11)
for r, gid in enumerate(order):
    i = list(F['basins']).index(gid); c = cases[gid]; obs = F['obs'][i][sel]
    ymax = 0
    for k, lead in enumerate(LEADS):
        ax = axs[r, k]; ens = PM.inflate(F[f'ens{lead}'][i][sel][None].astype(np.float32), _lam[f'1|ours_mc|{lead}'] if f'1|ours_mc|{lead}' in _lam else 1.0)[0]; mu = np.nanmean(ens, 1); lo, hi = np.nanpercentile(ens, [10, 90], 1); base = B[f'mean{lead}'][i][sel]
        ax.fill_between(dd, lo, hi, color=C_OURS, alpha=0.22, lw=0, label='KNOU-Flood (MC dropout, spread x lambda), 10–90% of 16 members' if (r == 0 and k == 0) else None)
        ax.plot(dd, base, color=C_AIFL, lw=1.3, label='AIFL-style baseline' if (r == 0 and k == 0) else None)
        ax.plot(dd, mu, color=C_OURS, lw=1.5, label='KNOU-Flood, ensemble mean' if (r == 0 and k == 0) else None)
        ax.plot(dd, obs, color=INK, lw=1.5, ls='-', label='observed' if (r == 0 and k == 0) else None, alpha=0.9)
        ok = np.isfinite(obs)
        for name, sim, col, yy in (('new', mu, C_OURS, 0.93), ('base', base, C_AIFL, 0.80)):
            ax.text(0.012, yy, f"{'KNOU-Flood' if name == 'new' else 'baseline '}  KGE′ {kge_prime(obs, sim):.2f} · NSE {nse(obs, sim):.2f} · RMSE {rmse(obs, sim):.2f}", transform=ax.transAxes, fontsize=8.6, color=col, va='top', fontweight='bold')
        ax.grid(axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True); ymax = max(ymax, np.nanmax([np.nanmax(hi), np.nanmax(base), np.nanmax(obs)]))
        ax.set_title(f'lead {lead} day' + ('s' if lead > 1 else ''), loc='left', fontsize=10, color=INK2, pad=3)
        if k == 0: ax.set_ylabel('discharge, mm/day')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for k in range(2): axs[r, k].set_ylim(0, ymax * 1.28)
    axs[r, 0].text(-0.055, 1.17 if r > 0 else 1.17, f"{c['stratum']}: {nice(c['name'])} — {c['country']} (area {c['area']:,.0f} km²)", transform=axs[r, 0].transAxes, fontsize=10.5, color=INK, fontweight='bold', va='bottom')
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='upper left', ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.055, 0.915), labelcolor=INK2)
fig.text(0.06, 0.985, 'Forecast case studies, 2022', fontsize=14, fontweight='bold', va='top'); fig.text(0.06, 0.957, 'One randomly drawn station per regime (not selected by skill) · forecast issued 1 / 7 days before the valid day', fontsize=10, color=INK2, va='top'); fig.text(0.06, 0.937, 'Scores are for the plotted year. Arid station: flows stay below 0.3 mm/day, so its scores are not meaningful.', fontsize=10, color=INK2, va='top')
fig.savefig(f'{R}/figs/figD_case_studies_2022.png', dpi=200); plt.close(fig)
# also print the yearly numbers so the text can quote them
for gid in order:
    i = list(F['basins']).index(gid); obs = F['obs'][i][sel]
    for lead in LEADS:
        mu = np.nanmean(F[f'ens{lead}'][i][sel], 1); base = B[f'mean{lead}'][i][sel]
        print(f"{cases[gid]['stratum']:18s} lead{lead:2d}: new KGE′ {kge_prime(obs, mu):5.2f} RMSE {rmse(obs, mu):5.2f} | base KGE′ {kge_prime(obs, base):5.2f} RMSE {rmse(obs, base):5.2f}")
