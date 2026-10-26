"""Graphical abstract (Environmental Modelling & Software): model schematic + lead-time curves of the test period.
Sized to be read at 13 cm width -> figs/graphical_abstract.{pdf,png}.  Curves: median KGE over stations, mean of three seeds (results/*_perbasin.npz)."""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.environ.get('KNOU_FLOOD_FIGS', ROOT + '/figs'); os.makedirs(FIGS, exist_ok=True)
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R = ROOT; SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BLUE, BLUE_L, ORG, ORG_L, GREEN, PURP = '#2a78d6', '#cde2fb', '#eb6834', '#f2a97f', '#1baf7a', '#7b5cc4'
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 7, 'text.color': INK, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.spines.top': False, 'axes.spines.right': False})
NEW = ['s424fsm_s0', 's424fsm_s1', 's424fsm_s2']; BASE = ['b2d_gct', 'b2d_s1', 'b2d_s2']; L = np.arange(1, 11)
def curve(names, proto): return np.mean([[np.nanmedian(np.load(f'{R}/results/{n}_perbasin.npz')[f'{proto}{j}_kge']) for j in L] for n in names], 0)

fig = plt.figure(figsize=(13 / 2.54, 5.6 / 2.54))
S = fig.add_axes([0, 0, 0.60, 1]); S.set_xlim(0, 60); S.set_ylim(0, 56); S.axis('off')
def box(x, y, w, h, fc, ec, title, lines, tc=INK):
    S.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.6', fc=fc, ec=ec, lw=0.8))
    S.text(x + w / 2, y + h - 2.6, title, ha='center', va='top', fontsize=7.2, fontweight='bold', color=tc)
    S.text(x + w / 2, y + h - 7.4, lines, ha='center', va='top', fontsize=6.1, color=INK2, linespacing=1.25)
def arrow(x0, y0, x1, y1):
    S.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.9, color=INK2))
# inputs
box(1.5, 30, 17, 22, '#fdf1ea', ORG, 'Forecasts', 'ECMWF HRES\n(physics-based)\n+ GraphCast\n(machine learning)\nlead-aware, 1–10 d')
box(1.5, 4, 17, 21, '#eef6f1', GREEN, 'Pre-training', 'ERA5-Land\nreanalysis\n1980–2019\n16,565 stations')
# core
box(22.5, 16, 15.5, 25, BLUE_L, BLUE, 'KNOU-Flood', 'single-layer LSTM\n0.41 M parameters\n424-day window\nlatent noise in the\nlast 60 steps', tc=BLUE)
# output: ensemble fan thumbnail (illustrative)
box(42, 16, 16.5, 25, '#f3effb', PURP, 'Ensemble', '16 members\nfair CRPS', tc=PURP)
t = np.linspace(0, 1, 40); rng = np.random.default_rng(3)
for k in range(9):
    yv = 18.8 + 6.0 * np.exp(-((t - 0.55) / 0.2) ** 2) * (0.75 + 0.07 * k) + 0.25 * rng.standard_normal(40).cumsum() / 6
    S.plot(44 + 12.5 * t, yv, color=PURP, lw=0.5, alpha=0.55)
S.plot(44 + 12.5 * t, 18.8 + 6.0 * np.exp(-((t - 0.52) / 0.18) ** 2), color=INK, lw=0.9)
arrow(18.9, 41, 22.0, 33); arrow(18.9, 14.5, 22.0, 23); arrow(38.4, 28.5, 41.6, 28.5)
S.text(30.25, 9.5, 'Scored at 2,099 stations\nworldwide, 2021–2023', ha='center', va='center', fontsize=6.2, color=INK2, linespacing=1.25)
# right: lead-time curves
A = fig.add_axes([0.685, 0.20, 0.295, 0.70])
A.plot(L, curve(NEW, 'traj'), color=BLUE, lw=1.8, marker='o', ms=2.6, label='KNOU-Flood')
A.plot(L, curve(BASE, 'traj'), color=ORG, lw=1.4, marker='o', ms=2.4, label='Baseline')
A.plot(L, curve(BASE, 'lt'), color=ORG_L, lw=1.2, ls='--', label='Baseline, full-window\nprotocol')
A.set_xticks([1, 4, 7, 10]); A.set_ylim(0.15, 0.82); A.set_yticks([0.2, 0.4, 0.6]); A.grid(axis='y', color=GRID, lw=0.6)
A.set_xlabel('lead time (days)', fontsize=6.6); A.set_ylabel('median KGE', fontsize=6.6); A.tick_params(labelsize=6.4, length=2)
A.legend(fontsize=5.6, frameon=False, loc='upper right', handlelength=1.6, borderaxespad=0.1)
fig.savefig(f'{FIGS}/graphical_abstract.pdf'); fig.savefig(f'{FIGS}/graphical_abstract.png', dpi=600); print('saved')
