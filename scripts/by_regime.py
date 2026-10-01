import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, numpy as np
T=np.load(ROOT + '/data/arrays/targets.npz',allow_pickle=True)
names=list(T['names']); sn=list(T['static_names']); St=T['static']; src=T['source']
A=np.load(sys.argv[1]); B=np.load(sys.argv[2]); proto=sys.argv[3]
idx=np.array([names.index(b) for b in A['basins']])
def col(c): return St[idx, sn.index(c)]
feats={'frac_snow':col('frac_snow') if 'frac_snow' in sn else None, 'aridity_ERA5_LAND':col('aridity_ERA5_LAND') if 'aridity_ERA5_LAND' in sn else None, 'log_area':col('log_area')}
print('basins',len(idx),'sources',{s:int((src[idx]==s).sum()) for s in np.unique(src[idx])})
def report(mask,label):
    row=[]
    for j in (1,3,5,10):
        a=A[f'{proto}{j}_kge'][mask]; b=B[f'{proto}{j}_kge'][mask]; ok=np.isfinite(a)&np.isfinite(b)
        row.append(np.median(a[ok]-b[ok]))
    print(f'  {label:28s} n={int(mask.sum()):4d}  median ΔKGE′ (A−B) lead1/3/5/10: '+' '.join(f'{r:+.3f}' for r in row))
for f,v in feats.items():
    if v is None: continue
    q=np.quantile(v,[1/3,2/3]); print(f'\n{f} terciles (low / mid / high):')
    for lab,m in (('low',v<=q[0]),('mid',(v>q[0])&(v<=q[1])),('high',v>q[1])): report(m,lab)
print('\nby source:')
for s in np.unique(src[idx]): report(src[idx]==s, s)
