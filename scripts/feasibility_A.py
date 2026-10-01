"""Feasibility check for 'forecast-disagreement-conditioned noise': does the HRES-GraphCast disagreement predict the size of our discharge-forecast error beyond what the flow level and the
ensemble spread already tell?  Test 2021 - 2023-12, final model seed S (ensemble as trained), leads 1/3/5/7/10.
D_target = sum over the last 3 forecast days (leads j-2..j, each at its own lead) of |GC - HRES| precipitation, scaled by the gauge's mean daily precipitation (D_all = all forecast days 1..j; D_T = same for T2M, scaled by 1 K).
error e = |obs - ensemble mean| / (sigma_obs + 0.1).   usage: feasibility_A.py [seed]"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, numpy as np
sys.path.insert(0, ROOT + '/scripts'); import probmetrics as PM
R = PM.R; sd = int(sys.argv[1]) if len(sys.argv) > 1 else 1; rng = np.random.default_rng(0)
Z = PM.load_dump(f'pm_s424_s{sd}_test'); bas = Z['basins']; dates = Z['dates']; obs = Z['obs']; T = len(dates)
A = f'{R}/data/arrays'; hb = list(np.load(f'{A}/HRES_basins.npy', allow_pickle=True)); gb = list(np.load(f'{A}/GRAPHCAST_basins.npy', allow_pickle=True))
hd = np.load(f'{A}/HRES_dates.npy').astype('datetime64[D]'); gd = np.load(f'{A}/GRAPHCAST_dates.npy').astype('datetime64[D]')
hi = np.array([hb.index(b) for b in bas]); gi = np.array([gb.index(b) for b in bas]); o_h, o_g = np.argsort(hi), np.argsort(gi)
def load(name, idx, order):
    a = np.load(f'{A}/{name}.npy', mmap_mode='r'); out = np.empty((len(idx),) + a.shape[1:], np.float32); out[order] = a[idx[order]]; return out          # (B, dates, 10)
PH, PG = load('HRES_total_precipitation', hi, o_h), load('GRAPHCAST_total_precipitation', gi, o_g); TH, TG = load('HRES_temperature_2m', hi, o_h), load('GRAPHCAST_temperature_2m', gi, o_g)
d0 = dates[0]; ofs_h = int((d0 - hd[0]).astype(int)); ofs_g = int((d0 - gd[0]).astype(int)); print('basins', len(bas), 'dates', T, 'offsets', ofs_h, ofs_g, flush=True)
pbar = np.nanmean(PH[:, :, 0], 1)[:, None] + 1e-3                                             # mean daily precipitation per gauge (lead-1 HRES), scaling
sig = np.nanstd(obs, 1)[:, None] + 0.1
def dis(j, which, kind):
    """sum over forecast days m in `which` (valid day = target - j + m, lead m) of |GC-HRES|; arrays aligned to the dump dates (B,T)."""
    tot = np.zeros((len(bas), T), np.float32)
    for m in which:
        sh = np.arange(T) - j + m                                                             # valid day index (dump axis) of forecast day m
        ih, ig = sh + ofs_h, sh + ofs_g; ok = (ih >= 0) & (ih < PH.shape[1]) & (ig >= 0) & (ig < PG.shape[1])
        a = np.full((len(bas), T), np.nan, np.float32); P, Q = (PH, PG) if kind == 'P' else (TH, TG)
        a[:, ok] = np.abs(Q[:, ig[ok], m - 1] - P[:, ih[ok], m - 1]); tot += a
    return tot
def spear(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
def auc_top(x, e, q=0.9):
    thr = np.quantile(e, q); pos = e >= thr; r = np.argsort(np.argsort(x)) + 1.0; n1, n0 = pos.sum(), (~pos).sum(); return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def r2(X, y):
    X = np.column_stack([np.ones(len(y))] + X); b = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ b; return float(1 - res.var() / y.var())
print(f'seed {sd}. columns: Spearman with |error| | AUC for the largest 10% errors | R2 of log-error regression   (features: level = log(forecast flow/sigma), spread = log(ensemble std/sigma))', flush=True)
LEADS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else PM.LEADS
for j in LEADS:
    ens = Z[f'ens{j}'][:].astype(np.float32); mean = ens.mean(2); spr = ens.std(2, ddof=1); err = np.abs(obs - mean) / sig
    feats = {'level': np.log(np.maximum(mean, 0) / sig + 0.01), 'spread': np.log(spr / sig + 0.003), 'D_target': np.log(dis(j, range(max(1, j - 2), j + 1), 'P') / pbar + 0.01),
             'D_all': np.log(dis(j, range(1, j + 1), 'P') / pbar + 0.01), 'D_T': np.log(dis(j, range(1, j + 1), 'T') + 0.05)}
    ok = np.isfinite(err) & np.all([np.isfinite(v) for v in feats.values()], 0); idx = np.flatnonzero(ok.ravel()); idx = rng.choice(idx, min(1_500_000, len(idx)), replace=False)
    e = np.log(err.ravel()[idx] + 0.02); F = {k: v.ravel()[idx] for k, v in feats.items()}; er = err.ravel()[idx]
    print(f'\nlead {j}: {len(idx)} gauge-days')
    for k in F: print(f'   {k:9s}  Spearman {spear(F[k], er):+.3f}   AUC {auc_top(F[k], er):.3f}')
    sp_lv = r2([F['level']], F['spread']); sp_D = r2([F['level'], F['D_all']], F['spread']); print(f'   does the ensemble spread already use the disagreement?  R2(spread ~ level) {sp_lv:.4f} -> + D_all {sp_D:.4f} (+{sp_D - sp_lv:.4f});  Spearman(spread, D_all) {spear(F["spread"], F["D_all"]):+.3f}')
    base = r2([F['level'], F['spread']], e); print(f'   R2 log-error:  level+spread {base:.4f}', ' | + D_target {:.4f} (+{:.4f})'.format(r2([F['level'], F['spread'], F['D_target']], e), r2([F['level'], F['spread'], F['D_target']], e) - base),
          '| + D_all {:.4f} (+{:.4f})'.format(r2([F['level'], F['spread'], F['D_all']], e), r2([F['level'], F['spread'], F['D_all']], e) - base), '| + D_T {:.4f} (+{:.4f})'.format(r2([F['level'], F['spread'], F['D_T']], e), r2([F['level'], F['spread'], F['D_T']], e) - base),
          '| + all three {:.4f} (+{:.4f})'.format(r2([F['level'], F['spread'], F['D_target'], F['D_all'], F['D_T']], e), r2([F['level'], F['spread'], F['D_target'], F['D_all'], F['D_T']], e) - base))
