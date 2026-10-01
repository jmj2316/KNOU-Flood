"""zarr (local copy) -> float32 .npy memmaps; ERA5-Land trimmed to 1980-01-01.. ; forecasts kept whole."""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, xarray as xr, pandas as pd, os, time, warnings, json
warnings.filterwarnings('ignore')
R = ROOT + '/data/multimet/{}/timeseries.zarr'
O = ROOT + '/data/arrays'
PLAN = {
 'ERA5_LAND': ('era5land_', ['total_precipitation','temperature_2m','surface_net_solar_radiation','surface_net_thermal_radiation','surface_pressure','u_component_of_wind_10m','v_component_of_wind_10m'], '1980-01-01', None),
 'HRES': ('hres_', ['total_precipitation','temperature_2m','surface_net_solar_radiation','surface_net_thermal_radiation','surface_pressure'], None, None),
 'GRAPHCAST': ('graphcast_', ['total_precipitation','temperature_2m','u_component_of_wind_10m','u_component_of_wind_10m'][:3]+['v_component_of_wind_10m'], None, None),
 'CPC': ('cpc_', ['precipitation'], '1980-01-01', None),
 'IMERG': ('imerg_', ['precipitation'], None, None),
}
t0 = time.time(); meta = {}
for prod, (pre, vars_, d0, d1) in PLAN.items():
    ds = xr.open_zarr(R.format(prod), chunks=None, decode_timedelta=False)
    dates = pd.to_datetime(ds['date'].values)
    sl = slice(np.searchsorted(dates, pd.Timestamp(d0)) if d0 else 0, len(dates))
    dsel = dates[sl]
    np.save(f'{O}/{prod}_basins.npy', ds['basin'].values.astype(str))
    np.save(f'{O}/{prod}_dates.npy', dsel.values.astype('datetime64[D]'))
    meta[prod] = dict(n_basin=int(ds.sizes['basin']), d0=str(dsel[0].date()), d1=str(dsel[-1].date()), n_date=len(dsel), vars=vars_)
    for v in vars_:
        fn = f'{O}/{prod}_{v}.npy'
        if os.path.exists(fn): continue
        a = ds[pre + v].isel(date=sl).values.astype(np.float32)
        np.save(fn + '.tmp.npy', a); os.replace(fn + '.tmp.npy', fn)
        print(f'{prod:10s} {v:32s} {a.shape} nan={np.isnan(a).mean():.4f} {a.nbytes/1e9:5.2f} GB  t={time.time()-t0:5.0f}s', flush=True)
json.dump(meta, open(f'{O}/meta.json', 'w'), indent=1)
print('DONE_ARRAYS', flush=True)
