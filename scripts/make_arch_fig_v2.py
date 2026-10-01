"""Publication-style schematic of KNOU-Flood -> figs/figE_architecture.{pdf,png}   (a: architecture and ensemble generation with a shared prefix; b: training and evaluation timeline)"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
R = ROOT
INK, INK2, LINE = '#1f2933', '#52606d', '#7b8794'
C = dict(hres='#3b78c4', season='#8d99a6', lead='#e39a47', gc='#2fa07a', stat='#7c6bb0', noise='#cf5560', lstm='#f2dfa2', lstm_e='#c9a43b', shared='#e8eef7', shared_e='#8aa4c8', fc='#bfe8d3', fc_e='#59b58f',
         pre='#4da36f', ft='#e5873a', val='#4a86d9', test='#c9503f')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.2, 'text.color': INK, 'axes.edgecolor': LINE, 'pdf.fonttype': 42})
fig = plt.figure(figsize=(6.6, 5.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 132); ax.set_ylim(0, 110); ax.axis('off')
def box(x, y, w, h, fc='white', ec=LINE, lw=0.7, r=1.2, **kw): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, **kw))
def arrow(x0, y0, x1, y1, c=LINE, lw=0.8, ms=5, rad=0.0): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c, shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}'))
def txt(x, y, s, size=6.2, c=INK, ha='left', va='center', w='normal', **kw): ax.text(x, y, s, fontsize=size, color=c, ha=ha, va=va, fontweight=w, **kw)
rng = np.random.default_rng(3)
txt(0.4, 108.4, 'a', 9, w='bold'); txt(4.0, 108.4, 'Architecture and ensemble generation', 7.2, w='bold'); txt(0.4, 37.4, 'b', 9, w='bold'); txt(4.0, 37.4, 'Training and evaluation', 7.2, w='bold')
# ---------------------------------------------------------------- inputs
txt(1, 104.0, 'Dynamic forcing', 6.8, w='bold'); txt(1, 101.1, '14 channels $\\times$ 424 days', 5.6, c=INK2)
y, rh, x0, xw = 98.4, 1.9, 1, 18
for name, n, col, desc in (('HRES', 5, C['hres'], 'TP, T2M, SSR, STR, SP'), ('Season', 2, C['season'], 'sin, cos of day of year'), ('Lead-aware', 2, C['lead'], 'forecast flag, lead / 10'), ('GraphCast', 5, C['gc'], 'TP, T2M, GC$-$HRES, flag')):
    top = y
    for k in range(n):
        v = np.cumsum(rng.normal(0, 1, 70)); v = (v - v.min()) / (np.ptp(v) + 1e-9)
        for j in range(70): ax.add_patch(Rectangle((x0 + j * xw / 70, y - rh), xw / 70 + 0.02, rh - 0.12, fc=col, ec='none', alpha=0.22 + 0.72 * v[j]))
        y -= rh
    ym = (top + y) / 2; txt(x0 + xw + 1.6, ym + 0.95, name, 5.8, w='bold', c=col); txt(x0 + xw + 1.6, ym - 1.05, desc, 4.9, c=INK2); y -= 0.7
y_end = y + 0.7
ax.plot([x0 + xw * 0.92] * 2, [98.6, y_end - 0.2], color=INK, lw=0.6, ls=(0, (2, 1.6))); txt(x0, y_end - 2.0, 'past 414 d', 5.0, c=INK2); txt(x0 + xw, y_end - 2.0, '+10 d', 5.0, c=INK2, ha='right')
txt(1, 63.4, 'Static attributes', 6.8, w='bold'); txt(1, 60.5, '212 per station', 5.6, c=INK2)
for j in range(48): ax.add_patch(Rectangle((1 + j * 18 / 48, 54.4), 18 / 48 + 0.02, 4.2, fc=C['stat'], ec='none', alpha=0.2 + 0.75 * rng.random()))
txt(20.6, 57.6, 'HydroATLAS, climate', 4.9, c=INK2); txt(20.6, 55.6, 'indices, area, UTC offset', 4.9, c=INK2)
txt(1, 51.0, 'Latent noise', 6.8, w='bold'); txt(1, 48.3, '$\\varepsilon\\sim\\mathcal{N}(0,I_8)$, one draw per member', 5.4, c=INK2)
for k in range(3):
    for j in range(8): ax.add_patch(Rectangle((1 + j * 2.25, 45.0 - k * 2.05), 2.1, 1.85, fc=C['noise'], ec='none', alpha=0.25 + 0.7 * rng.random()))
txt(19.6, 44.0, 'members 1, 2, ... $K$', 4.9, c=INK2)
# ---------------------------------------------------------------- embeddings
for (yb, hb, lab) in ((77.0, 12.4, 'Dynamic\nembedding'), (51.6, 9.2, 'Static\nembedding')):
    box(40, yb, 15, hb, fc='#f4f6f9'); txt(47.5, yb + hb * 0.63, lab, 5.8, ha='center', w='bold'); txt(47.5, yb + hb * 0.17, 'MLP 30-20-64, tanh', 4.7, ha='center', c=INK2)
arrow(37.2, 83.2, 39.8, 83.2); arrow(37.2, 56.2, 39.8, 56.2)
# ---------------------------------------------------------------- LSTM
txt(58, 105.6, 'LSTM, 1 layer, 256 units', 6.8, w='bold'); txt(58, 102.7, 'input per step: 64 + 64 (+ 8 noise)', 5.2, c=INK2)
cw = 6.0
def cell(cx, cy, fc, ec): box(cx - cw / 2, cy - cw / 2, cw, cw, fc=fc, ec=ec, lw=0.7, r=1.0); txt(cx, cy, 'LSTM', 4.3, ha='center', c=INK2)
cy0 = 74.5
box(57, cy0 - 5.2, 25.5, 10.4, fc='none', ec=C['shared_e'], lw=0.7, r=1.4, ls=(0, (3, 2)))
pre_x = [61.5, 69.5]
for i, cx in enumerate(pre_x):
    cell(cx, cy0, C['shared'], C['shared_e'])
    if i: arrow(pre_x[i - 1] + cw / 2, cy0, cx - cw / 2, cy0, ms=4)
arrow(pre_x[-1] + cw / 2, cy0, 76.6, cy0, ms=4); txt(78.6, cy0, '$\\cdots$', 8, ha='center')
txt(57.2, 86.4, 'steps 1 to 364', 5.4, w='bold', c='#4a6fa5'); txt(57.2, 84.0, 'noise = 0; computed once,', 4.9, c=INK2); txt(57.2, 81.9, 'shared by all members', 4.9, c=INK2)
mem_y = [91.0, cy0, 58.0]
for k, my in enumerate(mem_y):
    box(88, my - 5.0, 24, 10.0, fc='#fff7f6', ec='#e3b3b8', lw=0.6, r=1.3)
    box(89.2, my - 2.0, 4.2, 4.0, fc=C['noise'], ec='none', r=0.8); txt(91.3, my, f'$\\varepsilon_{{{"K" if k == 2 else k + 1}}}$', 4.7, ha='center', c='white')
    arrow(93.6, my, 96.0, my, c=C['noise'], ms=4)
    cell(99.0, my, C['lstm'], C['lstm_e']); cell(107.0, my, C['fc'], C['fc_e']); arrow(102.0, my, 104.0, my, ms=4)
    arrow(82.8, cy0, 87.8, my, ms=4, rad=0.0)
txt(100, 83.6, '$\\vdots$', 8, ha='center', c=INK2) if False else txt(100, 66.3, '$\\vdots$', 8, ha='center', c=INK2)
txt(100, 97.9, 'steps 365 to 424 (50 past days + 10 forecast days)', 5.2, w='bold', c=C['noise'], ha='center')
for (xa, xb, lab, col) in ((57, 82.5, 'computed once', '#4a6fa5'), (88, 112, 'member-specific', C['noise'])):
    ax.plot([xa, xb], [51.0, 51.0], color=col, lw=0.8); ax.plot([xa, xa], [51.0, 52.0], color=col, lw=0.8); ax.plot([xb, xb], [51.0, 52.0], color=col, lw=0.8)
    txt((xa + xb) / 2, 49.2, lab, 5.0, c=col, ha='center')
txt(57, 46.4, 'day $-413$', 4.8, c=INK2); txt(112, 46.4, 'day $+10$', 4.8, c=INK2, ha='right'); txt(90, 46.4, 'issue day', 4.8, c=INK2, ha='center')
# ---------------------------------------------------------------- head and output
box(115.5, 66.2, 15.5, 16.5, fc='#f4f6f9'); txt(123.25, 78.4, 'Last 10', 5.4, ha='center', w='bold'); txt(123.25, 76.0, 'hidden states', 5.4, ha='center', w='bold'); txt(123.25, 72.0, 'Dropout 0.4', 5.2, ha='center'); txt(123.25, 69.6, 'Linear 256 $\\rightarrow$ 1', 5.2, ha='center')
for my, ty in ((91.0, 80.6), (cy0, 74.5), (58.0, 68.4)): arrow(112.2, my, 115.3, ty, ms=4)
arrow(123.25, 66.0, 123.25, 62.2, ms=4)
ia = ax.inset_axes([116.8, 47.2, 13.4, 14.0], transform=ax.transData); ia.patch.set_alpha(0); days = np.arange(1, 11)
base = 3 + 2.4 * np.exp(-0.5 * ((days - 4.3) / 1.9) ** 2)
for k in range(16): ia.plot(days, base * (0.78 + 0.44 * rng.random()) + 0.12 * rng.normal(0, 1, 10).cumsum(), color='#2a78d6', lw=0.45, alpha=0.5)
ia.plot(days, base, color='#184f95', lw=1.4); ia.plot(days, base * 1.05 + np.array([0, 0.1, 0.3, 0.4, 0.2, 0, -0.2, -0.1, -0.2, -0.3]), color=INK, lw=0.8, ls=(0, (3, 1.5)))
ia.set_xticks([]); ia.set_yticks([]); ia.spines[['top', 'right']].set_visible(False); ia.set_xlabel('forecast day', fontsize=4.8, labelpad=1, color=INK2); ia.set_ylabel('discharge', fontsize=4.8, labelpad=1, color=INK2)
txt(123.25, 43.6, '16 members;', 4.9, ha='center', c=INK2); txt(123.25, 41.9, 'mean = point forecast', 4.9, ha='center', c=INK2)
# ---------------------------------------------------------------- panel b
x_of = lambda yr: 17 + (yr - 1980) * 1.6
for yr in (1980, 1990, 2000, 2010, 2020):
    ax.plot([x_of(yr)] * 2, [15.4, 16.4], color=LINE, lw=0.6); txt(x_of(yr), 13.4, str(yr), 5.0, ha='center', c=INK2)
ax.plot([x_of(1980), x_of(2024)], [16.4, 16.4], color=LINE, lw=0.6)
for yb, lab, segs, t1, t2 in ((31.4, 'Pre-training', [(1980, 2020, C['pre'])], 'ERA5-Land, 16,565 stations', 'normalized MSE; noise and GraphCast inputs zero'),
                              (25.0, 'Fine-tuning', [(2016, 2020, C['ft'])], 'HRES + GraphCast, 2,113 stations', '50% lead-1 and 50% trajectory windows; fair CRPS'),
                              (18.6, 'Validation, test', [(2020, 2021, C['val']), (2021, 2024, C['test'])], 'validation 2020, test 2021 to 2023', 'design choices, $\\lambda_j$ and $p^*$ use 2020 only')):
    txt(1, yb + 2.1, lab, 6.0, w='bold')
    for a, b, col in segs: box(x_of(a), yb, (b - a) * 1.6, 4.2, fc=col, ec='none', r=0.8)
    txt(x_of(2024) + 2.6, yb + 3.3, t1, 5.4, w='bold'); txt(x_of(2024) + 2.6, yb + 0.9, t2, 4.8, c=INK2)
txt(x_of(2000), 33.5, 'reanalysis', 5.2, c='white', ha='center') if False else txt(x_of(2000), 33.5, '', 5)
txt(x_of(2000), 33.45, 'reanalysis forcing', 5.0, c=INK2, ha='center') if False else None
box(1, 2.2, 130, 6.4, fc='#f4f6f9', ec='none', r=0.9)
txt(66, 5.4, 'Inference: 16 members, output dropout kept active, deviations from the ensemble mean scaled by $\\lambda_j$ (one value per lead, fitted on 2020)', 5.2, ha='center')
fig.savefig(f'{R}/figs/figE_architecture.pdf'); fig.savefig(f'{R}/figs/figE_architecture.png', dpi=220); print('saved')
