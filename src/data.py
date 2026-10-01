"""GPU-resident window sampler for KNOU-Flood.

Axes
  ERA axis : daily, 1980-01-01 .. 2024-10-31 (MultiMet ERA5-Land, UTC)   -> Q is aligned to this axis
  FC  axis : HRES 2016-01-01 .. 2024-09-30 ; GraphCast 2016-01-02 .. 2023-12-21.
             fc[b, d, k-1] = forecast VALID on day d, lead k=1..10 (== AIFL's "LT-k series" X[:,:,k-1]).
Windows are L days (module global, set by train.py: 180 for the single clock, 7*weeks+fast_len for the two-clock model);
the model emits the last N=10 daily discharges. Channels = vars + sin/cos(day-of-year) + 2 trajectory channels (zeros unless mode=traj).
Days before the forecast archive (older history in long windows) fall back to ERA5-Land.
  'era'  : inputs = ERA5-Land, y = Q[t-9..t]                          (pre-training / oracle evaluation)
  'lt'   : inputs = fc[:, :, k-1] for every day, y = Q[t-9..t]        (AIFL protocol; fine-tune k=1)
  'traj' : issue day t, past (<=t) uses lead-1 series, day t+j uses lead j ; y = Q[t+1..t+10]   (operational)
           + 2 extra channels (is_forecast, lead/10)
"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, pandas as pd

A = ROOT + '/data/arrays'
ERA0 = np.datetime64('1980-01-01')
L, N = 180, 10
ERA_VARS = ['total_precipitation', 'temperature_2m', 'surface_net_solar_radiation',
            'surface_net_thermal_radiation', 'surface_pressure']
HRES_VARS = ERA_VARS                      # same five variables (AIFL set)
GC_VARS = ['total_precipitation', 'temperature_2m', 'u_component_of_wind_10m', 'v_component_of_wind_10m']

def full_stats(names_full, era_vars):
    """the reanalysis normalisation statistics exactly as Store computes them on the FULL basin list (400 random basins, pre-2020) - lets light runs use a basin subset without changing the input scaling."""
    r = rows_for(names_full, 'ERA5_LAND'); idx = np.random.default_rng(0).choice(len(names_full), min(400, len(names_full)), replace=False)
    sub = np.stack([load_rows('ERA5_LAND', v, r[idx]) for v in era_vars], -1)[:, :d2i('2020-01-01')]
    return (sub.reshape(-1, sub.shape[-1]).mean(0), sub.reshape(-1, sub.shape[-1]).std(0) + 1e-6)

def d2i(date, origin=ERA0): return int((np.datetime64(date) - origin) / np.timedelta64(1, 'D'))

def rows_for(names, product):
    """indices of `names` in the product's basin axis (-1 if absent)."""
    b = np.load(f'{A}/{product}_basins.npy'); pos = {n: i for i, n in enumerate(b)}
    return np.array([pos.get(n, -1) for n in names])

def load_rows(product, var, rows):
    a = np.load(f'{A}/{product}_{var}.npy', mmap_mode='r')
    order = np.argsort(rows); out = np.empty((len(rows),) + a.shape[1:], np.float32)
    out[order] = a[rows[order]]
    return out

class Store:
    def __init__(s, names, Q, static, dev, era_vars=ERA_VARS, stats=None):
        """names: master basin list; Q (B,Te) aligned to ERA axis (NaN = missing); static (B,S) already z-scored."""
        s.names, s.dev = list(names), dev
        s.Q = torch.tensor(Q, dtype=torch.float32, device=dev)
        s.static = torch.tensor(static, dtype=torch.float32, device=dev)
        r = rows_for(names, 'ERA5_LAND'); assert (r >= 0).all(), 'basin missing in ERA5'
        era = np.stack([load_rows('ERA5_LAND', v, r) for v in era_vars], -1)       # (B,Te,F)
        if stats is None:
            sub = era[np.random.default_rng(0).choice(len(era), min(400, len(era)), replace=False), :d2i('2020-01-01')]
            stats = (sub.reshape(-1, sub.shape[-1]).mean(0), sub.reshape(-1, sub.shape[-1]).std(0) + 1e-6)
        s.stats = stats
        s.era = torch.tensor((era - stats[0]) / stats[1], dtype=torch.float32, device=dev); del era
        s.Te = s.era.shape[1]
        dd = np.datetime64('1980-01-01') + np.arange(s.Te)
        ang = 2 * np.pi * ((dd - dd.astype('datetime64[Y]')).astype(int)) / 365.25
        s.doy = torch.tensor(np.stack([np.sin(ang), np.cos(ang)], -1), dtype=torch.float32, device=dev)   # (Te,2)
        s.fc = {}; s.zero_extra = False; s.anchor = None; s.fusion = False; s.fusion_mode = 'full'; s.nowcast = None
        valid = ~torch.isnan(s.Q)
        first = valid.float().argmax(1); last = s.Te - 1 - valid.flip(1).float().argmax(1)
        s.first, s.last, s.has = first.cpu().numpy(), last.cpu().numpy(), valid.any(1).cpu().numpy()
        s.sigma = torch.tensor(np.nanstd(Q[:, :d2i('2020-01-01')], axis=1), dtype=torch.float32, device=dev).nan_to_num(1.0)

    def add_anchor(s, split_idx, half=7, chunk=1500):
        """Basin seasonal-discharge anchor: smoothed (+-half days) day-of-year mean of observations BEFORE `split_idx`, evaluated on the ERA axis.
        For dates inside the observation period the target date's own +-half-day window is removed (leave-one-year-out),
        so training windows never see an in-sample climatology. Normalised by the basin's observed mean/std (pre-split).
        Computed in basin chunks (float64) to keep GPU memory small."""
        dev = s.dev; B = len(s.names); out = torch.zeros(B, s.Te, dtype=torch.float32, device=dev)
        dates = np.datetime64('1980-01-01') + np.arange(s.Te); dd = torch.tensor((dates - dates.astype('datetime64[Y]')).astype(int), device=dev)
        for i0 in range(0, B, chunk):
            Qd = s.Q[i0:i0 + chunk].double().clone(); Qd[:, split_idx:] = float('nan')
            m = torch.isfinite(Qd); q0 = torch.where(m, Qd, torch.zeros_like(Qd)); mf = m.double()
            Sd = torch.zeros(len(Qd), 366, dtype=torch.float64, device=dev); Cd = torch.zeros_like(Sd)
            Sd.index_add_(1, dd, q0); Cd.index_add_(1, dd, mf)
            circ = lambda x: sum(torch.roll(x, k, 1) for k in range(-half, half + 1))
            Ss, Cs = circ(Sd), circ(Cd)
            def roll(x):
                cs = torch.cumsum(torch.nn.functional.pad(x, (half + 1, half)), 1)
                return cs[:, 2 * half + 1:] - cs[:, :-(2 * half + 1)]
            num = Ss[:, dd] - roll(q0); den = Cs[:, dd] - roll(mf)
            anc = torch.where(den > 0.5, num / den.clamp(min=1), torch.full_like(num, float('nan')))
            mu = (q0.sum(1) / mf.sum(1).clamp(min=1))[:, None]; var = (((q0 - mu) ** 2) * mf).sum(1) / mf.sum(1).clamp(min=1)
            sd = var.sqrt()[:, None].clamp(min=1e-3)
            out[i0:i0 + chunk] = ((anc - mu) / sd).nan_to_num(0.0).clamp(-5, 20).float()
            del Qd, q0, mf, Sd, Cd, Ss, Cs, num, den, anc
        s.anchor = out

    def add_nowcast(s):
        """Observation-type precipitation (CPC gauge analysis, IMERG-Early) on the ERA axis, normalised like ERA5-Land precipitation.
        Available only for days up to the issue day in trajectory windows (real-time analysis), never for forecast days."""
        chans = []
        for prod, first in (('CPC', '1980-01-01'), ('IMERG', '2000-06-01')):
            rr = rows_for(s.names, prod); ok = rr >= 0
            a = load_rows(prod, 'precipitation', np.where(ok, rr, 0)); off = d2i(first)
            full = np.full((len(s.names), s.Te), np.nan, np.float32)
            n = min(a.shape[1], s.Te - off); full[:, off:off + n] = a[:, :n]; full[~ok] = np.nan
            full = (full - s.stats[0][0]) / s.stats[1][0]
            chans.append(torch.tensor(full, dtype=torch.float32, device=s.dev)); del a, full
        s.nowcast = chans

    def _now(s, b, d, issue, lt=False):
        """[cpc, imerg, avail_cpc, avail_imerg] ; issue=None -> every position visible (simulation windows), else only d <= issue."""
        if s.nowcast is None: return []
        if lt: return [d.new_zeros(d.shape[0], d.shape[1], 4, dtype=torch.float32)]
        vis = torch.ones_like(d, dtype=torch.bool) if issue is None else (d <= issue[:, None])
        outs, av = [], []
        for c in s.nowcast:
            v = c[b[:, None], d]; ok = torch.isfinite(v) & vis
            outs.append(torch.where(ok, v, torch.zeros_like(v))); av.append(ok.float())
        return [torch.stack(outs + av, -1)]

    def add_forecast(s, key, product, var_names, fc_rows_master, first_date, stats_idx=None):
        """fc_rows_master: indices (into master list) of basins that get forecast tensors; -> s.fc[key]=(tensor, master->local map)."""
        rr = rows_for([s.names[i] for i in fc_rows_master], product)
        ok = rr >= 0; fc_rows_master = np.asarray(fc_rows_master)[ok]; rr = rr[ok]
        arrs = [load_rows(product, v, rr) for v in var_names]                     # (Bf,Td,10)
        x = np.stack(arrs, -1)                                                    # (Bf,Td,10,F)
        si = list(range(len(s.stats[0]))) if stats_idx is None else list(stats_idx)
        assert x.shape[-1] == len(si), 'forecast vars must match the reanalysis vars they are normalised with'
        x = (x - s.stats[0][si]) / s.stats[1][si]
        loc = -np.ones(len(s.names), int); loc[fc_rows_master] = np.arange(len(fc_rows_master))
        s.fc[key] = dict(x=torch.tensor(x, dtype=torch.float32, device=s.dev), loc=torch.tensor(loc, device=s.dev),
                         off=d2i(first_date), n=x.shape[1])

    # ---- window construction -------------------------------------------------
    def _win(s, b, t, mode, key=None, k=1):
        """b: (B,) master idx ; t: (B,) ERA-axis idx of last day (era/lt) or issue day (traj)."""
        dev = s.dev; ar = torch.arange(L, device=dev)
        if mode == 'era':
            d = t[:, None] - (L - 1) + ar
            x = s.era[b[:, None], d]
            return torch.cat([x, s.doy[d]] + s._anc(b, d) + [x.new_zeros(x.shape[0], x.shape[1], 2)] + s._fus(b, d, None, x, 0, 0) + s._now(b, d, None), -1), s.Q[b[:, None], d[:, -N:]]
        f = s.fc[key]; loc = f['loc'][b]; off = f['off']
        if mode == 'lt':
            d = t[:, None] - (L - 1) + ar; lead = torch.full_like(ar, k - 1)
        else:   # traj: window covers t-(L-N)+1 .. t+N ; positions after t are forecasts of lead 1..N, earlier ones lead 1
            d = t[:, None] - (L - N - 1) + ar
            lead = torch.clamp(ar - (L - N - 1), min=0)                            # 0 for past, j for day t+j
        di = d - off; inside = (di >= 0)[..., None]
        li = lead if mode == 'lt' else torch.clamp(lead - 1, min=0)                # lead index (LT1 for the past)
        xf = f['x'][loc[:, None], torch.clamp(di, min=0), li[None] if li.dim() == 1 else li]
        x = torch.where(inside, xf, s.era[b[:, None], d])                          # before the archive -> ERA5-Land
        if mode == 'lt':
            extra = x.new_zeros(x.shape[0], x.shape[1], 2)
        else:
            extra = torch.stack([(lead > 0).float(), lead.float() / N], -1)[None].expand(len(b), -1, -1)
            if s.zero_extra: extra = extra * 0
        return torch.cat([x, s.doy[d]] + s._anc(b, d) + [extra] + s._fus(b, d, li, xf, 0, 0) + s._now(b, d, t if mode == 'traj' else None, lt=(mode == 'lt')), -1), s.Q[b[:, None], d[:, -N:]]

    def _fus(s, b, d, li, xh, n_b, L_):
        """GraphCast channels [gc_TP, gc_T2M, gc-hres (TP,T2M), available] aligned with the HRES window xh (B,L,>=2); zeros where GC is absent."""
        if not s.fusion: return []
        if li is None: return [xh.new_zeros(xh.shape[0], xh.shape[1], 5)]
        g = s.fc['gc']; loc = g['loc'][b]; di = d - g['off']
        ok = ((di >= 0) & (di < g['n']) & (loc[:, None] >= 0))[..., None]
        v = g['x'][torch.clamp(loc, min=0)[:, None], torch.clamp(di, 0, g['n'] - 1), li[None] if li.dim() == 1 else li]
        v = torch.where(ok, v, torch.zeros_like(v)); dv = torch.where(ok, v - xh[..., :2], torch.zeros_like(v))
        if s.fusion_mode == 'nodiff': dv = torch.zeros_like(dv)       # ablation: keep GraphCast TP/T2M and the availability flag, drop the GraphCast-minus-HRES difference channels
        return [torch.cat([v, dv, ok.float()], -1)]

    def _anc(s, b, d): return [] if s.anchor is None else [s.anchor[b[:, None], d][..., None]]

    def batch(s, b, t, mode, key=None, k=1, ok_only=True):
        x, y = s._win(b, t, mode, key, k)
        good = torch.isfinite(x).all((1, 2))
        return x, s.static[b], y, good, s.sigma[b]
