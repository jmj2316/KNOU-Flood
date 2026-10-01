"""Event-forecast figure from results/event_members_report.json -> figs/figH_event_members.png  (CSI vs return period, test, seeds pooled; bars = min-max over seeds)"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = ROOT; J = json.load(open(f'{R}/results/event_members_report.json')); RPS = (1.5, 2, 5, 10, 20, 50); LEADS = (1, 5, 10)
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, BLUE2, BLUE3, ORG, ORG2, ORG3 = '#184f95', '#2a78d6', '#86b6ef', '#eb6834', '#a8471c', '#f2a97f'
SYS = [('ours_mc_infl|tuned', BLUE, '-', 'o', 'KNOU-Flood, member-based (MC dropout, spread x lambda)'), ('ours_raw|tuned', BLUE2, '-', 's', 'KNOU-Flood, member-based (as trained)'), ('ours_mean|det', BLUE3, '--', '^', 'KNOU-Flood, ensemble mean'),
       ('b2d_mc|tuned', ORG, '-', 's', 'AIFL recipe, MC-dropout members'), ('b2d_dress_sqrt|tuned', ORG2, ':', 'd', 'AIFL recipe, residual-dressed members'), ('b2d_det|det', ORG3, '--', '^', 'AIFL recipe, deterministic')]
have = lambda k, key: f'{k}|1|1.5|{key.split("|")[0]}|{key.split("|")[1]}' in J
SYS = [s for s in SYS if have('single', s[0])]
fig, axs = plt.subplots(2, 3, figsize=(16, 9), sharey='row'); fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.075, hspace=0.3, wspace=0.08)
for r, kind in enumerate(('single', 'dual')):
    for c, lead in enumerate(LEADS):
        ax = axs[r, c]
        for key, col, ls, mk, lab in SYS:
            ys = [J[f'{kind}|{lead}|{rp:g}|{key.split("|")[0]}|{key.split("|")[1]}'] for rp in RPS]; v = np.array([y['csi'] for y in ys]); lo = np.array([min(y['per_seed']) for y in ys]); hi = np.array([max(y['per_seed']) for y in ys])
            ax.errorbar(range(6), v, yerr=[v - lo, hi - v], color=col, ls=ls, marker=mk, ms=4.5, lw=1.7, capsize=2, elinewidth=0.8, label=lab if (r == 0 and c == 0) else None)
        ax.set_xticks(range(6)); ax.set_xticklabels([f'{rp:g}' for rp in RPS]); ax.set_yscale('log'); ax.grid(axis='y', color=GRID, lw=0.7, which='both'); ax.set_xlabel('return period (years)')
        ax.set_title(f'{"Observed return levels" if kind == "single" else "Model-simulated return levels"} · lead {lead} d', loc='left', fontsize=10.5)
        if c == 0: ax.set_ylabel('CSI = hits / (hits + misses + false alarms)')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=3, fontsize=9, frameon=False)
fig.text(0.055, 0.985, 'Flood-event detection: member-based event forecasts vs. ensemble-mean and deterministic forecasts', fontsize=14, fontweight='bold', va='top')
fig.text(0.055, 0.955, 'Test 2021 - 2023-12, pooled over stations and days, exact-day matching.  Member-based: an event is forecast when at least a fraction p* of the 16 members exceeds the return level;\np* (and the spread factor lambda) are tuned on 2020 only.  Bars: min-max over seeds.',
         fontsize=9.3, color=INK2, va='top')
for c in range(3): axs[0, c].set_ylim(2e-3, 0.5); axs[1, c].set_ylim(0.04, 0.45)
fig.savefig(f'{R}/figs/figH_event_members.png', dpi=140); print('saved')
