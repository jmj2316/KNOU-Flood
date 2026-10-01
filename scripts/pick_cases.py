"""Stratified RANDOM choice of 4 case-study gauges (no cherry-picking by skill): one per regime, among test gauges with >=95% observed days in 2022."""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, json, numpy as np, pandas as pd
R = ROOT
T = np.load(f'{R}/data/arrays/targets.npz', allow_pickle=True); names = list(T['names']); Q = T['Q']
b = np.load(f'{R}/results/tc424fsm_gct_perbasin.npz')['basins']
d0 = (np.datetime64('2022-01-01') - np.datetime64('1980-01-01')).astype(int); d1 = (np.datetime64('2022-12-31') - np.datetime64('1980-01-01')).astype(int)
idx = np.array([names.index(x) for x in b]); cov = np.isfinite(Q[idx, d0:d1 + 1]).mean(1)
rd = lambda kind: pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/data/caravan/**/attributes_{kind}_*.csv', recursive=True)]).drop_duplicates('gauge_id').set_index('gauge_id')
car, oth = rd('caravan').reindex(b), rd('other').reindex(b)
df = pd.DataFrame({'snow': car.frac_snow.values, 'arid': car.aridity_ERA5_LAND.values, 'area': oth.area.values, 'cov': cov, 'country': oth.country.values, 'name': oth.gauge_name.values, 'lat': oth.gauge_lat.values, 'lon': oth.gauge_lon.values}, index=b)
ok = df['cov'] >= 0.95
strata = {'snow-dominated': ok & (df.snow >= 0.4) & (df.arid < 1.0), 'humid, rain-fed': ok & (df.snow < 0.05) & (df.arid < 0.8), 'arid': ok & (df.arid >= 2.0), 'large river basin': ok & (df.area >= 20000)}
rng = np.random.default_rng(42); cases = []
for k, m in strata.items():
    c = df[m]; pick = c.index[rng.integers(len(c))]; r = df.loc[pick]
    cases.append(dict(stratum=k, id=pick, country=str(r.country), name=str(r['name']), area=float(r.area), snow=float(r.snow), arid=float(r.arid), n_candidates=int(len(c))))
    print(f"{k:18s} candidates={len(c):4d}  -> {pick}  {r.country} | {r['name']} | area {r.area:.0f} km2 | snow {r.snow:.2f} | aridity {r.arid:.2f}")
json.dump(cases, open(f'{R}/results/cases.json', 'w'), indent=1)
