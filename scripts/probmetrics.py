"""Probabilistic verification helpers for KNOU-Flood ensembles. All arrays are (B gauges, T valid days[, K members]); NaN = not scored."""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
R = ROOT; LEADS = (1, 3, 5, 7, 10)

def load_dump(name):
    return np.load(f'{R}/results/dump_{name}.npz')

def crps_fair(ens, y):
    """fair ensemble CRPS, ens (B,T,K), y (B,T) -> (B,T); NaN where y or any member is NaN. physical units."""
    e = ens.astype(np.float32); K = e.shape[2]
    t1 = np.abs(e - y[..., None]).mean(2)
    xs = np.sort(e, 2); w = (2 * np.arange(K) - K + 1).astype(np.float32)
    t2 = (xs * w).sum(2) / (K * (K - 1))                 # 1/(2K(K-1)) * sum_{k,l}|xk-xl|
    return t1 - t2

def chunks(B, n=256):
    for i in range(0, B, n): yield slice(i, min(B, i + n))

def clim_crps(basins, dates, obs, Qall, names, n_window=15, min_n=30):
    """CRPS of the day-of-year climatological ensemble (observations of 1980-2015 within +-n_window days of the valid day's DOY), per gauge/day."""
    i16 = int((np.datetime64('2016-01-01') - np.datetime64('1980-01-01')).astype(int)); d0 = np.datetime64('1980-01-01') + np.arange(i16)
    doy_tr = (d0 - d0.astype('datetime64[Y]')).astype(int); doy_ev = (dates - dates.astype('datetime64[Y]')).astype(int)
    out = np.full(obs.shape, np.nan, np.float32); idx = {n: i for i, n in enumerate(names)}
    win = {d: (np.abs(((doy_tr - d + 183) % 366) - 183) <= n_window) for d in np.unique(doy_ev)}
    for b, g in enumerate(basins):
        q = Qall[idx[g], :i16]; fin = np.isfinite(q)
        for d, m in win.items():
            s = np.sort(q[m & fin]); n = len(s)
            if n < min_n: continue
            ts = np.where(doy_ev == d)[0]; y = obs[b, ts]; ok = np.isfinite(y)
            if not ok.any(): continue
            cs = np.concatenate([[0.0], np.cumsum(s)]); pos = np.searchsorted(s, y[ok])
            t1 = (y[ok] * pos - cs[pos] + (cs[n] - cs[pos]) - y[ok] * (n - pos)) / n
            t2 = (s * (2 * np.arange(n) - n + 1)).sum() / (n * (n - 1))
            out[b, ts[ok]] = t1 - t2
    return out

def dress_members(val_mean, val_obs, test_mean, K=16, space='sqrt', min_pairs=60):
    """Residual ('error') dressing of a deterministic forecast: K empirical quantiles (p=(k+.5)/K) of the gauge- and lead-specific residuals of the validation year,
    in sqrt-flow space (space='sqrt') or additively ('add'); gauges with < min_pairs validation pairs use the residuals pooled over all gauges. returns (B,T,K)."""
    f = (lambda x: np.sqrt(np.maximum(x, 0))) if space == 'sqrt' else (lambda x: x); fi = (lambda z: np.maximum(z, 0) ** 2) if space == 'sqrt' else (lambda z: np.maximum(z, 0))
    r = f(val_obs) - f(val_mean); ok = np.isfinite(r); ps = (np.arange(K) + 0.5) / K
    pooled = np.quantile(r[ok], ps) if ok.sum() else np.zeros(K)
    Q = np.tile(pooled, (val_mean.shape[0], 1))
    for b in range(val_mean.shape[0]):
        rb = r[b][ok[b]]
        if len(rb) >= min_pairs: Q[b] = np.quantile(rb, ps)
    return Q, f, fi

def apply_dress(Q, f, fi, mean):
    return fi(f(mean)[..., None] + Q[:, None, :]).astype(np.float32)        # (B,T,K); NaN means stay NaN

def inflate(ens, lam):
    m = ens.mean(2, keepdims=True); return np.maximum(m + lam * (ens - m), 0)

def rank_hist(ens, y, rng):
    """rank of y among the K members with random tie breaking -> counts (K+1,)"""
    K = ens.shape[2]; e = ens.astype(np.float32); ok = np.isfinite(y) & np.isfinite(e).all(2)
    lt = (e < y[..., None]).sum(2); eq = (e == y[..., None]).sum(2); r = lt + np.floor(rng.random(y.shape) * (eq + 1)).astype(int)
    return np.bincount(r[ok], minlength=K + 1)

def summarize(ens, mean, y, sigma_b, clim_c, basins_mask=None):
    """per-gauge scores for one lead. returns dict of (B,) arrays: crps, crps_clim, rmse_mean, spread (sqrt mean variance, K-corrected), cover80"""
    B, T, K = ens.shape; out = {k: np.full(B, np.nan, np.float32) for k in ('crps', 'crps_clim', 'rmse', 'spread', 'cover80', 'n')}
    for sl in chunks(B):
        e = ens[sl].astype(np.float32); yy = y[sl]; c = crps_fair(e, yy); ok = np.isfinite(c) & np.isfinite(clim_c[sl])
        c = np.where(ok, c, np.nan); cc = np.where(ok, clim_c[sl], np.nan); m = e.mean(2)
        err2 = np.where(ok, (m - yy) ** 2, np.nan); var = np.where(ok, e.var(2, ddof=1), np.nan)
        q10, q90 = np.quantile(e, [0.1, 0.9], axis=2); cov = np.where(ok, (yy >= q10) & (yy <= q90), np.nan)
        out['crps'][sl] = np.nanmean(c, 1); out['crps_clim'][sl] = np.nanmean(cc, 1); out['rmse'][sl] = np.sqrt(np.nanmean(err2, 1))
        out['spread'][sl] = np.sqrt(np.nanmean(var, 1)) * np.sqrt((K + 1) / K); out['cover80'][sl] = np.nanmean(cov, 1); out['n'][sl] = ok.sum(1)
    return out
