"""Calibration / probabilistic-skill figure from results/prob_eval.json + prob_eval_perbasin.npz -> figs/figG_probabilistic.png"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = ROOT; J = json.load(open(f'{R}/results/prob_eval.json')); S = J['summary']; seeds = J['seeds']; Z = np.load(f'{R}/results/prob_eval_perbasin.npz'); LEADS = (1, 3, 5, 7, 10)
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'; BLUE, BLUE2, ORG, ORG2 = '#2a78d6', '#86b6ef', '#eb6834', '#a8471c'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})
STY = {'ours_raw': (BLUE, '-', 'o', 'KNOU-Flood, as trained'), 'ours_mc_infl': ('#184f95', '-', 'D', 'KNOU-Flood + MC dropout, spread x lambda'), 'ours_infl': (BLUE2, '--', 's', 'KNOU-Flood, spread x lambda (fit on 2020)'),
       'b2d_mc_infl': (ORG, '-', 'o', 'AIFL recipe, MC dropout, spread x lambda'), 'b2d_dress_sqrt': (ORG2, '--', 's', 'AIFL recipe, residual dressing'), 'clim': (MUTED, ':', None, 'Climatology (DOY)')}
fig, axs = plt.subplots(2, 3, figsize=(16, 9.4)); fig.subplots_adjust(left=0.05, right=0.985, top=0.84, bottom=0.07, hspace=0.3, wspace=0.22)
for k, lead in enumerate((1, 10)):
    ax = axs[0, k]
    for n in ('ours_raw', 'ours_infl', 'ours_mc_infl', 'b2d_mc_infl', 'b2d_dress_sqrt'):
        h = sum(Z[f'hist|{sd}|{n}|{lead}'] for sd in seeds); c, ls, m, lab = STY[n]; ax.plot(range(17), h / h.sum(), color=c, ls=ls, marker=m, ms=4, lw=1.6, label=lab if k == 0 else None)
    ax.axhline(1 / 17, color=INK, lw=1, ls=':'); ax.set_title(f'Rank histogram, lead {lead} d  (flat = reliable)', loc='left', fontsize=10.5); ax.set_xlabel('rank of the observation among 16 members'); ax.set_ylabel('frequency'); ax.grid(axis='y', color=GRID, lw=0.7)
def line(ax, key, ttl, yl, ref=None):
    for n in ('ours_raw', 'ours_infl', 'ours_mc_infl', 'b2d_mc_infl', 'b2d_dress_sqrt', 'clim'):
        if f'{n}_{LEADS[0]}' not in S and n != 'clim': continue
        c, ls, m, lab = STY[n]
        if n == 'clim' and key != 'crps': continue
        y = [S[f'{n}_{l}'][key] if n != 'clim' else J['clim_crps'][str(l)] for l in LEADS]; ax.plot(LEADS, y, color=c, ls=ls, marker=m, ms=4.5, lw=1.8)
    if ref is not None: ax.axhline(ref, color=INK, lw=1, ls=':')
    ax.set_title(ttl, loc='left', fontsize=10.5); ax.set_xlabel('lead time (days)'); ax.set_ylabel(yl); ax.grid(axis='y', color=GRID, lw=0.7); ax.set_xticks(LEADS)
line(axs[0, 2], 'crps', 'CRPS (median over stations, lower is better)', 'mm / day')
line(axs[1, 0], 'spread_rmse', 'Spread / RMSE of the ensemble mean (1 = reliable)', 'ratio', ref=1.0)
line(axs[1, 1], 'cover80', 'Coverage of the q10-q90 interval (dotted: 12/17 for 16 members)', 'fraction', ref=12 / 17)
line(axs[1, 2], 'outside', 'Observation outside the ensemble range (dotted: ideal 2/17)', 'fraction', ref=2 / 17)
fig.legend(handles=[plt.Line2D([], [], color=STY[n][0], ls=STY[n][1], marker=STY[n][2], label=STY[n][3]) for n in STY], loc='upper center', bbox_to_anchor=(0.5, 0.945), ncol=3, fontsize=9, frameon=False)
fig.savefig(f'{R}/figs/figG_probabilistic.png', dpi=140, bbox_inches='tight', pad_inches=0.12); print('saved')
