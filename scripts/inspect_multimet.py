import xarray as xr, numpy as np, pandas as pd, collections, json
P='gs://caravan-multimet/v1/{}/timeseries.zarr/'
def op(p): return xr.open_zarr(P.format(p),chunks=None,storage_options=dict(token='anon'))
ds={}
for p in ['ERA5_LAND','HRES','GRAPHCAST','CPC','IMERG','CHIRPS']:
    d=op(p); ds[p]=d
    dt=pd.to_datetime(d['date'].values)
    print(f'{p:10s} dates {dt[0].date()} .. {dt[-1].date()}  n={len(dt)}  basins={d.sizes["basin"]}  vars={[v for v in d.data_vars][:6]}')
    if 'lead_time' in d: print('   lead_time values:',d['lead_time'].values, d['lead_time'].attrs)
print()
print('HRES units attr:\n',ds['HRES'].attrs.get('Units'))
print('ERA5 units attr (head):\n',str(ds['ERA5_LAND'].attrs.get('Units'))[:900])
b=ds['HRES']['basin'].values.astype(str)
pref=[x.split('_')[0] for x in b]
c=collections.Counter(pref); print('\nHRES basin prefix counts:',dict(c))
# contiguity of prefixes
runs=[];s=0
for i in range(1,len(pref)+1):
    if i==len(pref) or pref[i]!=pref[s]:
        runs.append((pref[s],s,i-1)); s=i
print('n runs',len(runs)); print(runs[:40])
print('sorted?',bool(np.all(b[:-1]<=b[1:])))
b2=ds['ERA5_LAND']['basin'].values.astype(str)
print('ERA5 basins',len(b2),'== HRES set?',set(b2)==set(b), ' same order?',bool(np.array_equal(b2[:len(b)],b)) if len(b2)>=len(b) else 'n/a', 'ERA5 minus HRES:',sorted(set(b2)-set(b))[:5],'HRES minus ERA5:',sorted(set(b)-set(b2))[:5])
# sample values for one basin
i=list(b).index([x for x in b if x.startswith('camels_')][0])
h=ds['HRES'].isel(basin=i,date=slice(0,3))
print('\nsample basin',b[i]); print(h['hres_total_precipitation'].values, '\n', h['hres_temperature_2m'].values)
e=ds['ERA5_LAND'].sel(date=h['date'].values[:3]).isel(basin=list(b2).index(b[i]))
print('ERA5 same dates TP',e['era5land_total_precipitation'].values,' T2M',e['era5land_temperature_2m'].values)
print('HRES dates',pd.to_datetime(h['date'].values))
