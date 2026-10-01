# Which valid day does HRES/GraphCast [date d, lead k] correspond to? Compare with ERA5-Land precip at d+j.
import xarray as xr, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
P='gs://caravan-multimet/v1/{}/timeseries.zarr/'
op=lambda p: xr.open_zarr(P.format(p),chunks=None,storage_options=dict(token='anon'),decode_timedelta=False)
h=op('HRES'); g=op('GRAPHCAST'); e=op('ERA5_LAND')
rng=np.random.default_rng(0)
bh=h['basin'].values.astype(str); be=e['basin'].values.astype(str)
common=np.array(sorted(set(bh)&set(be)))
pick=rng.choice(common,128,replace=False)   # 128 random basins (touches many chunks, but only reads slices)
ih=np.array([np.where(bh==b)[0][0] for b in pick]); ie=np.array([np.where(be==b)[0][0] for b in pick])
d0,d1='2018-01-01','2018-12-31'
for name,ds,var,evar in [('HRES',h,'hres_total_precipitation','era5land_total_precipitation'),('HRES',h,'hres_temperature_2m','era5land_temperature_2m'),('GC',g,'graphcast_total_precipitation','era5land_total_precipitation')]:
    x=ds[var].sel(date=slice(d0,d1)).isel(basin=ih).values          # (b, date, lead)
    dates=pd.to_datetime(ds['date'].sel(date=slice(d0,d1)).values)
    y=e[evar].sel(date=slice('2017-12-25','2019-01-15')).isel(basin=ie)
    ydates=pd.to_datetime(y['date'].values); yv=y.values
    print(f'--- {name} {var}: corr(HRES[d,lead k], ERA5[d+s]) rows=k(1..10) cols=s(-1..+2)')
    for k in range(10):
        row=[]
        for s in (-1,0,1,2):
            idx=np.searchsorted(ydates,dates+pd.Timedelta(days=k+s))
            a=x[:,:,k].ravel(); b=yv[:,idx].ravel()
            m=np.isfinite(a)&np.isfinite(b)
            row.append(np.corrcoef(a[m],b[m])[0,1])
        print(f'  lead{k+1:2d}: '+'  '.join(f'{r:6.3f}' for r in row), '  best s =',(-1,0,1,2)[int(np.argmax(row))])
