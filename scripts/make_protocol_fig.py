"""Protocol effect: the same model weights scored (a) with the full-window protocol (LT-k forecast in every input step) and (b) operational ('trajectory': past = lead-1 forecasts, day t+j = lead-j forecast).
test 2021 - 2023-12, median over stations, 3 seeds (band = min-max over seeds) -> figs/figI_protocol_effect.png"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.environ.get('KNOU_FLOOD_FIGS', ROOT + '/figs'); os.makedirs(FIGS, exist_ok=True)
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = ROOT; SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'; BLUE, BLUE_L, ORG, ORG_L = '#2a78d6', '#86b6ef', '#eb6834', '#f2a97f'
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})
NEW = ['s424fsm_s0', 's424fsm_s1', 's424fsm_s2']; BASE = ['b2d_gct', 'b2d_s1', 'b2d_s2']; L = np.arange(1, 11)
def curve(names, proto, m): return np.array([[np.nanmedian(np.load(f'{R}/results/{n}_perbasin.npz')[f'{proto}{j}_{m}']) for j in L] for n in names])
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2)); fig.subplots_adjust(left=0.05, right=0.985, top=0.70, bottom=0.12, wspace=0.2)
for ax, m, ttl, yl in zip(axs, ('kge', 'nse', 'rmse'), ("KGE (higher is better)", 'NSE (higher is better)', 'RMSE, mm/day (lower is better)'), ("KGE", 'NSE', 'mm / day')):
    for names, proto, col, ls, lab in ((BASE, 'lt', ORG, '-', 'Baseline, full-window protocol'), (BASE, 'traj', ORG_L, '--', 'Baseline, same weights, operational protocol'),
                                       (NEW, 'lt', BLUE, '-', 'KNOU-Flood, full-window protocol'), (NEW, 'traj', BLUE_L, '--', 'KNOU-Flood, operational protocol')):
        c = curve(names, proto, m); ax.fill_between(L, c.min(0), c.max(0), color=col, alpha=0.18, lw=0); ax.plot(L, c.mean(0), color=col, ls=ls, lw=2, marker='o', ms=3.5, label=lab)
    if m == 'kge': ax.plot([1, 10], [0.663, 0.215], 'D', color=INK, ms=6, label='Published values (Taccari et al., 2026)')
    if m == 'nse': ax.axhline(0, color=INK, lw=0.8, ls=':'); ax.plot([1], [0.53], 'D', color=INK, ms=6)
    if m == 'rmse': ax.axhline(0.715, color=INK, lw=0.8, ls=':'); ax.text(10, 0.723, 'climatology', ha='right', fontsize=8.5, color=INK2)
    ax.set_title(ttl, loc='left', fontsize=10.5); ax.set_xlabel('forecast lead time (days)'); ax.set_ylabel(yl); ax.set_xticks(L); ax.grid(axis='y', color=GRID, lw=0.7)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.83), ncol=3, fontsize=9, frameon=False)
fig.savefig(f'{FIGS}/figI_protocol_effect.png', dpi=300, bbox_inches='tight', pad_inches=0.12); fig.savefig(f'{FIGS}/figI_protocol_effect.pdf', bbox_inches='tight', pad_inches=0.12); print('saved')
