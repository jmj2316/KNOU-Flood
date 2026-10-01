"""Flood-event verification in the AIFL style (arXiv 2602.16579 sec. 4.2), for our forecasts.

Return-period thresholds : Gumbel fitted with L-moments to annual maxima, RP in {1.5, 2, 5, 10, 20, 50} years.
  observed events   : discharge >= threshold from the OBSERVED annual maxima (whole record, >= MIN_YEARS years)
  forecast events   :  dual   = threshold from the model's own ERA5-driven simulation 1980-2024 (annual maxima of the simulation)
                       single = the observed threshold
Exact-day matching (zero-day tolerance). POD = hits / observed events, FAR = false alarms / forecast events, precision = 1 - FAR.
Counts are pooled over all gauges and test days, separately for every lead time.

usage:  event_metrics.py DUMP.npz [DUMP2.npz ...]      (dumps written by train.py --dump_all --dump_era)
        event_metrics.py --selftest
"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, numpy as np
R = ROOT
RPS = (1.5, 2, 5, 10, 20, 50); MIN_YEARS = 10; EULER = 0.5772156649

def gumbel_levels(am, rps=RPS):
    """L-moment Gumbel fit -> return levels. am: 1-D annual maxima (NaN removed)."""
    x = np.sort(am); n = len(x)
    b0 = x.mean(); b1 = np.sum(np.arange(n) / (n - 1) * x) / n
    l1, l2 = b0, 2 * b1 - b0
    a = l2 / np.log(2); xi = l1 - EULER * a
    return np.array([xi - a * np.log(-np.log(1 - 1 / T)) for T in rps])

def annual_maxima(series, dates, min_days=300):
    yrs = dates.astype('datetime64[Y]').astype(int) + 1970; out = []
    for y in np.unique(yrs):
        v = series[yrs == y]; v = v[np.isfinite(v)]
        if len(v) >= min_days: out.append(v.max())
    return np.array(out)

def thresholds(series, dates):
    am = annual_maxima(series, dates)
    return gumbel_levels(am) if len(am) >= MIN_YEARS else None

def verify(obs, fc, thr_obs, thr_fc, day_ok):
    """obs, fc: (B,T); thr_*: (B,len(RPS)) with NaN rows for gauges without a threshold. returns dict rp -> (hit, miss, fa)."""
    res = {}
    for i, rp in enumerate(RPS):
        to, tf = thr_obs[:, i][:, None], thr_fc[:, i][:, None]
        ok = day_ok & np.isfinite(to) & np.isfinite(tf)
        eo, ef = (obs >= to) & ok, (fc >= tf) & ok
        res[rp] = (int((eo & ef).sum()), int((eo & ~ef).sum()), int((~eo & ef).sum()))
    return res

def summarise(c):
    hit, miss, fa = c; pod = hit / (hit + miss) if hit + miss else np.nan; far = fa / (hit + fa) if hit + fa else np.nan
    return pod, far, 1 - far if far == far else np.nan, hit + miss

def run(path):
    Z = np.load(path); B = Z['basins']; dates = Z['dates']; obs_w = Z['obs']; lo = int(Z['lo'])
    T = np.load(f'{R}/data/arrays/targets.npz', allow_pickle=True); names = list(T['names']); Q = T['Q']
    idx = np.array([names.index(b) for b in B]); alld = np.datetime64('1980-01-01') + np.arange(Q.shape[1])
    thr_obs = np.array([thresholds(Q[i], alld) if thresholds(Q[i], alld) is not None else np.full(len(RPS), np.nan) for i in idx])
    have_era = 'era_sim' in Z.files
    if have_era:
        era = Z['era_sim']; e_lo = int(Z['era_lo']); ed = alld[e_lo:e_lo + era.shape[1]]
        thr_mod = np.array([thresholds(era[k], ed) if thresholds(era[k], ed) is not None else np.full(len(RPS), np.nan) for k in range(len(B))])
    print(f'{path}: {len(B)} gauges, observed thresholds for {int(np.isfinite(thr_obs[:, 0]).sum())}' + (f', model thresholds for {int(np.isfinite(thr_mod[:, 0]).sum())}' if have_era else ', NO era_sim (dual threshold skipped)'))
    day_ok = np.isfinite(obs_w)
    print(f"{'lead':>4} {'RP':>4} | {'obs events':>10} | {'single: POD  FAR  prec':>26} | {'dual: POD  FAR  prec':>24}")
    out = {}
    for j in range(1, 11):
        f = Z[f'mean{j}']; sing = verify(obs_w, f, thr_obs, thr_obs, day_ok); dual = verify(obs_w, f, thr_obs, thr_mod, day_ok) if have_era else None
        out[j] = (sing, dual)
        if j in (1, 3, 5, 7, 10):
            for rp in RPS:
                ps = summarise(sing[rp]); pd_ = summarise(dual[rp]) if dual else (np.nan,) * 4
                print(f'{j:>4} {rp:>4} | {ps[3]:>10d} | {ps[0]:>8.3f} {ps[1]:>6.3f} {ps[2]:>6.3f}   | {pd_[0]:>8.3f} {pd_[1]:>6.3f} {pd_[2]:>6.3f}')
    return out

def selftest():
    rng = np.random.default_rng(0); a, xi = 3.0, 10.0
    am = xi - a * np.log(-np.log(rng.random(200000)))            # Gumbel(xi, a)
    est = gumbel_levels(am); true = np.array([xi - a * np.log(-np.log(1 - 1 / T)) for T in RPS])
    print('L-moment Gumbel levels vs truth:', np.round(est, 2), np.round(true, 2)); assert np.allclose(est, true, rtol=0.02)
    # counting logic: 1 gauge, obs exceeds on days 2,5 ; forecast exceeds on days 2,3 -> hit 1, miss 1, false alarm 1
    obs = np.array([[0, 0, 9, 0, 0, 9, 0]], float); fc = np.array([[0, 0, 8, 8, 0, 0, 0]], float)
    thr = np.full((1, len(RPS)), 5.0); r = verify(obs, fc, thr, thr, np.ones_like(obs, bool))[2]
    print('counts (hit, miss, fa):', r); assert r == (1, 1, 1)
    pod, far, prec, n = summarise(r); assert abs(pod - 0.5) < 1e-9 and abs(far - 0.5) < 1e-9 and n == 2
    print('selftest OK')

if __name__ == '__main__':
    if '--selftest' in sys.argv: selftest()
    else:
        for p in sys.argv[1:]: run(p)
