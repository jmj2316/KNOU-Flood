"""Caravan v1.6 core + GRDC-Caravan -> data/arrays/targets.npz (names, Q[mm/d] on the ERA axis, z-scored statics)."""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, xarray as xr, glob, os, time, warnings, json
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings('ignore')
R = ROOT; C = f'{R}/data/caravan'
era_names = set(np.load(f'{R}/data/arrays/ERA5_LAND_basins.npy'))
dates = pd.date_range('1980-01-01', '2024-10-31'); assert len(dates) == 16376
files = sorted(glob.glob(f'{C}/**/timeseries/**/*.nc', recursive=True))
print('nc files', len(files), flush=True)
def read(fn):
    gid = os.path.basename(fn)[:-3]
    if gid not in era_names: return gid, None
    try:
        with xr.open_dataset(fn) as d:
            q = d['streamflow'].to_series().reindex(dates)
            return gid, q.values.astype(np.float32)
    except Exception as e:
        return gid, None
t0 = time.time(); Qd = {}
with ProcessPoolExecutor(16) as ex:   # HDF5 is not thread-safe -> processes
    for i, (gid, q) in enumerate(ex.map(read, files, chunksize=32)):
        if q is not None and np.isfinite(q).any(): Qd[gid] = q      # duplicates across core/GRDC: first file wins
        if i % 4000 == 0: print(i, len(Qd), f'{time.time()-t0:.0f}s', flush=True)
names = sorted(Qd); Q = np.stack([Qd[n] for n in names]); print('basins with Q:', len(names), flush=True)
# ---- statics
def read_attr(kind):
    fs = sorted(glob.glob(f'{C}/**/attributes_{kind}_*.csv', recursive=True))
    df = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates('gauge_id').set_index('gauge_id'); return df
A = {k: read_attr(k) for k in ['hydroatlas', 'caravan', 'other']}
oth = A['other']; print('other cols', oth.columns.tolist())
X = pd.concat([A['hydroatlas'], A['caravan']], axis=1).reindex(names)
X = X.select_dtypes(include=[np.number]).loc[:, lambda d: d.notna().mean() > 0.5]
for c in [c for c in X.columns if c.startswith('area') or c.endswith('_area')]: pass
area = oth['area'].reindex(names).values if 'area' in oth else np.full(len(names), np.nan)
lon = oth['gauge_lon'].reindex(names).values if 'gauge_lon' in oth else np.zeros(len(names))
X['log_area'] = np.log1p(np.nan_to_num(area, nan=np.nanmedian(area)))
X['utc_offset'] = np.round(np.nan_to_num(lon) / 15.0)
X = X.fillna(X.median()); mu, sd = X.mean(), X.std().replace(0, 1)
Z = ((X - mu) / sd).clip(-5, 5).values.astype(np.float32)
src = np.array([n.split('_')[0] for n in names])
np.savez(f'{R}/data/arrays/targets.npz', names=np.array(names), Q=Q, static=Z, static_names=np.array(X.columns), source=src, area=area.astype(np.float32))
i16, i20, i21 = (dates.get_loc(pd.Timestamp(x)) for x in ['2016-01-01', '2020-01-01', '2021-01-01'])
cnt = lambda lo, hi: np.isfinite(Q[:, lo:hi]).sum(1)
print('static dim', Z.shape[1], '| pre(>=3y obs 1980-2019):', int((cnt(0, i20) >= 1095).sum()),
      '| ft(>=1y obs 2016-19):', int((cnt(i16, i20) >= 365).sum()),
      '| test(>=1y obs 2021-24 & ft):', int(((cnt(i21, len(dates)) >= 365) & (cnt(i16, i20) >= 365)).sum()))
import collections
print('test basins by source:', dict(collections.Counter(src[(cnt(i21, len(dates)) >= 365) & (cnt(i16, i20) >= 365)])))
print('DONE_TARGETS', flush=True)
