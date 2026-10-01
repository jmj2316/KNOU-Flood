"""Copy the needed Caravan-MultiMet zarr arrays from the public GCS bucket to local disk (same zarr layout).
Resumable: skips arrays whose local chunk count already matches remote."""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gcsfs, os, sys, time
fs = gcsfs.GCSFileSystem(token='anon')
B = 'caravan-multimet/v1'
OUT = ROOT + '/data/multimet'
PLAN = {
 'ERA5_LAND': ['era5land_total_precipitation','era5land_temperature_2m','era5land_surface_net_solar_radiation',
               'era5land_surface_net_thermal_radiation','era5land_surface_pressure',
               'era5land_u_component_of_wind_10m','era5land_v_component_of_wind_10m'],
 'HRES': ['hres_total_precipitation','hres_temperature_2m','hres_surface_net_solar_radiation',
          'hres_surface_net_thermal_radiation','hres_surface_pressure'],
 'GRAPHCAST': ['graphcast_total_precipitation','graphcast_temperature_2m',
               'graphcast_u_component_of_wind_10m','graphcast_v_component_of_wind_10m'],
 'CPC': ['cpc_precipitation'], 'IMERG': ['imerg_precipitation'],
}
COORDS = ['basin','date','lead_time']
def sync(remote_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    rem = [p for p in fs.find(remote_dir)]
    todo = []
    for p in rem:
        rel = p[len(remote_dir):].lstrip('/')
        lp = os.path.join(local_dir, rel)
        if not (os.path.exists(lp) and os.path.getsize(lp) == fs.info(p)['size'] if False else os.path.exists(lp)):
            todo.append((p, lp))
    if todo:
        fs.get([t[0] for t in todo], [t[1] for t in todo], batch_size=64)
    return len(rem), len(todo)
t0 = time.time()
for prod, vars_ in PLAN.items():
    base = f'{B}/{prod}/timeseries.zarr'
    lbase = f'{OUT}/{prod}/timeseries.zarr'
    os.makedirs(lbase, exist_ok=True)
    for f in ['.zgroup','.zattrs','.zmetadata']:
        fs.get(f'{base}/{f}', f'{lbase}/{f}')
    for v in COORDS + vars_:
        try:
            if not fs.exists(f'{base}/{v}'): continue
        except Exception: continue
        n, k = sync(f'{base}/{v}', f'{lbase}/{v}')
        print(f'{prod:10s} {v:44s} files={n:4d} fetched={k:4d}  t={time.time()-t0:7.0f}s', flush=True)
print('DONE_FORCING', time.strftime('%c'), flush=True)
