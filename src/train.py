"""KNOU-Flood: pre-train on ERA5-Land, fine-tune on forecasts, evaluate lead 1..10.

  python src/train.py --name b2_lt --track hres --ft lt      # AIFL recipe (fine-tune on LT1 series)
  python src/train.py --name m1    --track hres --ft traj    # trajectory inputs (ours)
Results -> results/<name>.json ; checkpoints -> ckpt/<name>_{pre,ft}.pt
"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import data as D
from data import Store, d2i, N, ERA_VARS, HRES_VARS, GC_VARS
from model import RiverLSTM, TwoClockLSTM, norm_mse, norm_mse_lead, crps_fair_norm
from metrics import kge_prime, nse, rmse

R = ROOT
p = argparse.ArgumentParser()
p.add_argument('--name', required=True); p.add_argument('--track', default='hres', choices=['hres', 'gc'])
p.add_argument('--ft', default='lt', choices=['none', 'lt', 'traj', 'mix']); p.add_argument('--hidden', type=int, default=256)
p.add_argument('--bs', type=int, default=512); p.add_argument('--seed', type=int, default=0)
p.add_argument('--pre_epochs', type=int, default=20); p.add_argument('--pre_upd', type=int, default=500)
p.add_argument('--ft_epochs', type=int, default=10); p.add_argument('--ft_upd', type=int, default=300)
p.add_argument('--lr', type=float, default=5e-4); p.add_argument('--ft_lr', type=float, default=1.25e-4)
p.add_argument('--max_basins', type=int, default=0, help='debug: cap #basins'); p.add_argument('--dev', default='cuda')
p.add_argument('--pre_ckpt', default='', help='reuse a pre-trained checkpoint (skip pre-training)')
p.add_argument('--ft_pool', default='test', choices=['test', 'all'], help='test = AIFL parity (fine-tune on the basins that are tested); all = every basin with >=1y obs in 2016-19')
p.add_argument('--arch', default='single', choices=['single', 'tc'], help='single = AIFL-style one LSTM ; tc = slow/fast two-clock model')
p.add_argument('--L', type=int, default=180, help='single-clock window (days)')
p.add_argument('--weeks', type=int, default=17, help='tc: slow-clock length (weeks)'); p.add_argument('--fast_len', type=int, default=60, help='tc: fast-clock length (days)')
p.add_argument('--hidden_s', type=int, default=128)
p.add_argument('--split', default='test', choices=['val', 'test'], help='val = 2020 (for design decisions) ; test = 2021+ (final only)')
p.add_argument('--lead_w', action='store_true', help='traj fine-tune: equalise the gradient share of the 10 lead times (loss_j / EMA(loss_j))')
p.add_argument('--anchor', action='store_true', help='add the leave-one-year-out seasonal-discharge anchor channel')
p.add_argument('--fusion', action='store_true', help='add GraphCast channels (gc TP/T2M, gc-hres differences, availability) next to HRES')
p.add_argument('--tw_lambda', type=float, default=0.0, help='>0: add lambda x threshold-weighted CRPS (chaining function max(x, t_b), t_b = gauge q90 of the 1980-2015 observations) to the fine-tuning loss (upper-tail emphasis)')
p.add_argument('--tw_q', type=float, default=0.9)
p.add_argument('--fusion_mode', default='full', choices=['full', 'nodiff'], help='nodiff (ablation): zero the two GraphCast-minus-HRES difference channels')
p.add_argument('--gc_window', action='store_true', help='score only days covered by GraphCast (<= 2023-12-21) so fusion and HRES-only models share the same target days')
p.add_argument('--noise_dim', type=int, default=0, help='tc: >0 makes the fast clock stochastic (latent noise members, fair-CRPS loss)')
p.add_argument('--K', type=int, default=8, help='ensemble members in training'); p.add_argument('--K_eval', type=int, default=16)
p.add_argument('--nowcast', action='store_true', help='add CPC / IMERG-Early precipitation for the days up to the issue day (observation-anchored look-back)')
p.add_argument('--accum', type=int, default=1, help='stochastic runs: split each batch into this many micro-batches (same gradient, less memory)')
p.add_argument('--pre_only', action='store_true', help='stop after saving the ERA5 pre-trained checkpoint')
p.add_argument('--dump_names', default='', help='comma-separated gauge ids: evaluate only these and save their forecast time series (+ ensemble members) to results/dump_<name>.npz')
p.add_argument('--dump_all', action='store_true', help='evaluate all gauges, save their trajectory forecasts (ensemble mean) to results/dump_<name>.npz and exit')
p.add_argument('--dump_members', default='', help='comma-separated leads: with --dump_all also save the ensemble members (float16) of these leads, (B,T,K_eval)')
p.add_argument('--mc_dropout', action='store_true', help='keep the output dropout active when sampling: deterministic models give K_eval MC-dropout members, stochastic ones get extra dropout spread (mean output stays the dropout-free pass for deterministic models)')
p.add_argument('--dump_era', action='store_true', help='with --dump_all: also save the model driven by ERA5-Land over the whole record (for model-specific flood thresholds)')
p.add_argument('--K_sim', type=int, default=4, help='ensemble members used for the long ERA5-driven simulation')
p.add_argument('--noise_last', type=int, default=0, help='single clock: inject the latent noise only in the last N steps (earlier steps shared by all members)')
p.add_argument('--traj_only', action='store_true', help='skip the LT-k protocol and the ERA5 oracle')
p.add_argument('--holdout_frac', type=float, default=0.0, help='ungauged-basin test: hold out this fraction of 2°x2° spatial blocks (all gauges inside) from pre-training AND fine-tuning; score only on the held-out gauges')
p.add_argument('--holdout_seed', type=int, default=0)
p.add_argument('--holdout_file', default='', help='text file of gauge ids removed from ALL training (ungauged test on a fixed gauge list, e.g. gauges a reference model never saw)')
p.add_argument('--score_file', default='', help='with --holdout_file: score only these held-out gauges')
p.add_argument('--eval_traj', action='store_true', help='also score the model under the trajectory protocol')
p.add_argument('--ft_ckpt', default='', help='skip training; evaluate this fine-tuned checkpoint')
a = p.parse_args()
L = a.L if a.arch == 'single' else 7 * a.weeks + a.fast_len          # total daily window fed to the model
D.L = L
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = a.dev if torch.cuda.is_available() else 'cpu'
os.makedirs(f'{R}/results', exist_ok=True); os.makedirs(f'{R}/ckpt', exist_ok=True)

# ---------------------------------------------------------------- data
T = np.load(f'{R}/data/arrays/targets.npz', allow_pickle=True)
names, Q, static = list(T['names']), T['Q'], T['static']
if a.max_basins: names, Q, static = names[:a.max_basins], Q[:a.max_basins], static[:a.max_basins]
era_vars = ERA_VARS if a.track == 'hres' else GC_VARS
fc_vars, fc_prod, fc_key, fc_first = (HRES_VARS, 'HRES', 'fc', '2016-01-01') if a.track == 'hres' else (GC_VARS, 'GRAPHCAST', 'fc', '2016-01-02')
fc_last = '2024-09-30' if a.track == 'hres' else '2023-12-21'
t0 = time.time()
i16, i20, i21, i_end = d2i('2016-01-01'), d2i('2020-01-01'), d2i('2021-01-01'), d2i(fc_last)
cnt = lambda lo, hi: np.isfinite(Q[:, lo:hi]).sum(1)
pre_pool = np.where(cnt(0, i20) >= 3 * 365)[0]                                  # obs 1980-2019
ft_pool = np.where(cnt(i16, i20) >= 365)[0]                                     # obs during forecast archive (train yrs)
test_pool = np.where((cnt(i21, i_end + 1) >= 365) & (cnt(i16, i20) >= 365))[0]  # AIFL-style: same basins fine-tuned & tested
if a.ft_pool == 'test': ft_pool = test_pool
HELD = None
if a.holdout_frac > 0:   # spatial-block hold-out (whole 2°x2° blocks incl. duplicates and near neighbours are removed from training)
    import glob, pandas as pd
    co = pd.concat([pd.read_csv(f)[['gauge_id', 'gauge_lat', 'gauge_lon']] for f in glob.glob(f'{R}/data/caravan/**/attributes_other_*.csv', recursive=True)]).drop_duplicates('gauge_id').set_index('gauge_id').reindex(names)
    blk = (np.floor(co.gauge_lat.values / 2) * 1000 + np.floor(co.gauge_lon.values / 2)).astype(np.int64)
    tb = np.unique(blk[test_pool]); rngh = np.random.default_rng(a.holdout_seed); ho_blocks = rngh.choice(tb, int(round(a.holdout_frac * len(tb))), replace=False)
    HELD = np.isin(blk, ho_blocks)
    n_before = len(test_pool); pre_pool, ft_pool = pre_pool[~HELD[pre_pool]], ft_pool[~HELD[ft_pool]]
    test_pool = test_pool[HELD[test_pool]]                      # only held-out gauges are scored
    print(f'HOLD-OUT: {len(ho_blocks)}/{len(tb)} blocks -> {len(test_pool)}/{n_before} test gauges held out ({int(HELD.sum())} gauges of all pools removed from training)', flush=True)
    assert not (HELD[pre_pool].any() or HELD[ft_pool].any())
if a.holdout_file:
    hold = set(open(a.holdout_file).read().split()); HELD = np.array([n in hold for n in names])
    n_before = len(test_pool); pre_pool, ft_pool = pre_pool[~HELD[pre_pool]], ft_pool[~HELD[ft_pool]]
    keepm = HELD[test_pool]
    if a.score_file: sc = set(open(a.score_file).read().split()); keepm &= np.array([names[i] in sc for i in test_pool])
    test_pool = test_pool[keepm]
    print(f'HOLD-OUT FILE: {int(HELD.sum())} gauges removed from training; scoring {len(test_pool)}/{n_before} test gauges; pre pool {len(pre_pool)}, fine-tune pool {len(ft_pool)}', flush=True)
    assert not (HELD[pre_pool].any() or HELD[ft_pool].any())
stats = None
if a.pre_ckpt or a.ft_ckpt:   # light mode: no pre-training -> keep only the basins that are fine-tuned / scored (same input scaling as the full run)
    stats = D.full_stats(names, era_vars); keep = np.union1d(ft_pool, test_pool)
    remap = -np.ones(len(names), int); remap[keep] = np.arange(len(keep))
    names, Q, static = [names[i] for i in keep], Q[keep], static[keep]
    ft_pool, test_pool, pre_pool = remap[ft_pool], remap[test_pool], np.array([], int)
    cnt = lambda lo, hi: np.isfinite(Q[:, lo:hi]).sum(1)
S = Store(names, Q, static, dev, era_vars, stats=stats)
if a.tw_lambda > 0:                                                                  # per-gauge tail threshold t_b (mm/d); gauges without history get a huge value -> the extra term vanishes
    _i16 = int((np.datetime64('2016-01-01') - np.datetime64('1980-01-01')).astype(int)); S.tq = torch.full((S.Q.shape[0],), 1e6, device=dev)
    for _i in range(0, S.Q.shape[0], 400):
        _q = torch.nanquantile(S.Q[_i:_i + 400, :_i16].float(), a.tw_q, dim=1); S.tq[_i:_i + 400] = torch.where(torch.isfinite(_q), _q, torch.full_like(_q, 1e6))
    print(f'tail threshold (q{a.tw_q}) median {float(S.tq[S.tq < 1e5].median()):.3f} mm/d, lambda {a.tw_lambda}', flush=True)
if a.anchor: S.add_anchor(i16)
if a.nowcast: assert a.track == 'hres'; S.add_nowcast()
S.add_forecast(fc_key, fc_prod, fc_vars, np.union1d(ft_pool, test_pool), fc_first)
has_fc = S.fc[fc_key]['loc'].cpu().numpy() >= 0
ft_pool, test_pool = ft_pool[has_fc[ft_pool]], test_pool[has_fc[test_pool]]
if a.fusion:
    assert a.track == 'hres'; S.fusion = True; S.fusion_mode = a.fusion_mode
    S.add_forecast('gc', 'GRAPHCAST', GC_VARS[:2], np.union1d(ft_pool, test_pool), '2016-01-02', stats_idx=[0, 1])
FOFF, FN = S.fc[fc_key]['off'], S.fc[fc_key]['n']                # forecast archive: first ERA-axis index, #days
F = len(era_vars) + 4 + int(a.anchor) + (5 if a.fusion else 0) + (4 if a.nowcast else 0)                                       # vars + sin/cos(doy) + (is_forecast, lead) trajectory channels
print(f'[{time.time()-t0:.0f}s] basins={len(names)} pre={len(pre_pool)} ft={len(ft_pool)} test={len(test_pool)} dev={dev}', flush=True)

def sample(pool, lo, hi, n, mode):
    """random (basin, t) pairs with t inside the basin's observed span and [lo,hi]."""
    b = pool[np.random.randint(0, len(pool), n)]
    l = np.maximum(S.first[b] + 0, lo); h = np.minimum(S.last[b], hi)
    ok = h > l; b, l, h = b[ok], l[ok], h[ok]
    t = l + (np.random.rand(len(b)) * (h - l + 1)).astype(int)
    return torch.tensor(b, device=dev), torch.tensor(t, device=dev)

def tw_loss(p, y, sg, tq):
    base = crps_fair_norm(p, y, sg)
    if a.tw_lambda > 0: base = base + a.tw_lambda * crps_fair_norm(p.clamp(min=tq[:, None, None]), y.clamp(min=tq[:, None]), sg)
    return base

def train(model, pool, lo, hi, mode, epochs, upd, lr, tag, k=1):
    opt = torch.optim.Adam(model.parameters(), lr); tot = epochs * upd; step = 0; ema = None
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / max(1, tot // 10)) * 0.5 * (1 + np.cos(np.pi * i / tot)))
    for ep in range(epochs):
        model.train(); ls = []
        for _ in range(upd):
            md, l_, h_ = mode, lo, hi
            if mode == 'mix': md, l_, h_ = ('lt', FOFF + 179, i20 - 1) if np.random.rand() < 0.5 else ('traj', FOFF + 169, i20 - 1 - N)
            b, t = sample(pool, l_, h_, int(a.bs * 1.3), md)
            x, st, y, good, sg = S.batch(b, t, md, fc_key, k)
            good &= torch.isfinite(y).any(1)
            if good.sum() < 32: continue
            idx = good.nonzero().squeeze(1)[:a.bs]
            if STOCH and a.accum > 1:
                opt.zero_grad(); loss = 0.0
                for ch in torch.chunk(idx, a.accum):
                    l_ = tw_loss(model(x[ch], st[ch], K=a.K), y[ch], sg[ch], S.tq[b][ch] if a.tw_lambda > 0 else None) * (len(ch) / len(idx)); l_.backward(); loss += l_.detach()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); ls.append(float(loss)); continue
            if STOCH:
                loss = tw_loss(model(x[idx], st[idx], K=a.K), y[idx], sg[idx], S.tq[b][idx] if a.tw_lambda > 0 else None)
            elif a.lead_w and md == 'traj':
                lj = norm_mse_lead(model(x[idx], st[idx]), y[idx], sg[idx])
                ema = lj.detach() if ema is None else 0.98 * ema + 0.02 * lj.detach()
                loss = (lj / ema.clamp(min=1e-3)).mean() * ema.mean()          # same overall scale, equal lead shares
            else:
                loss = norm_mse(model(x[idx], st[idx]), y[idx], sg[idx])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            ls.append(loss.item())
        print(f'  {tag} ep{ep+1}/{epochs} loss={np.mean(ls):.4f} lr={sched.get_last_lr()[0]:.2e} t={time.time()-t0:.0f}s', flush=True)

@torch.no_grad()
def predict(model, basins, lo, hi, mode, k=1, bs=None):
    bs = bs or (192 if STOCH else 2048)     # ensembles multiply the batch by K_eval
    """returns sim (B, hi-lo+1) for lead k ('lt'/'era' blocks of N days) or dict lead->sim ('traj')."""
    model.eval(); B = len(basins); bt = torch.tensor(basins, device=dev)
    if mode in ('era', 'lt'):
        sim = np.full((B, hi - lo + 1), np.nan, np.float32)
        for t in range(lo + N - 1, hi + 1, N):
            for i in range(0, B, bs):
                bb = bt[i:i + bs]; tt = torch.full_like(bb, t)
                x, st, _, good, _ = S.batch(bb, tt, mode, fc_key, k)
                o_ = model(x, st, K=a.K_eval) if STOCH else model(x, st)
                out = (o_.mean(1) if STOCH else o_).float().cpu().numpy(); out[~good.cpu().numpy()] = np.nan
                sim[i:i + bs, t - N + 1 - lo:t + 1 - lo] = out
        return sim
    sims = {j: np.full((B, hi - lo + 1), np.nan, np.float32) for j in range(1, N + 1)}
    want = [int(x) for x in a.dump_members.split(',')] if a.dump_members else (list(range(1, N + 1)) if a.dump_names else [])
    PROB = STOCH or a.mc_dropout
    ENS = {j: np.full((B, hi - lo + 1, a.K_eval), np.nan, np.float16 if a.dump_members else np.float32) for j in want} if (PROB and want) else None
    cr = {j: np.zeros(B) for j in range(1, N + 1)}; cn = {j: np.zeros(B) for j in range(1, N + 1)}; sp = {j: np.zeros(B) for j in range(1, N + 1)}
    for t in range(max(lo - N, FOFF + 169), hi):   # issue days; day t+j scored if inside [lo,hi]
        for i in range(0, B, bs):
            bb = bt[i:i + bs]; tt = torch.full_like(bb, t)
            x, st, _, good, _ = S.batch(bb, tt, 'traj', fc_key)
            if a.mc_dropout: model.drop.train()
            if STOCH: o_ = model(x, st, K=a.K_eval); mem = o_
            elif a.mc_dropout: model.drop.eval(); o_ = model(x, st); model.drop.train(); mem = torch.stack([model(x, st) for _ in range(a.K_eval)], 1)   # o_ = dropout-free point forecast, mem = MC-dropout members
            else: o_ = model(x, st); mem = None
            if a.mc_dropout: model.drop.eval()
            out = (o_.mean(1) if STOCH else o_).float().cpu().numpy(); out[~good.cpu().numpy()] = np.nan
            for j in range(1, N + 1):
                d = t + j
                if lo <= d <= hi:
                    sims[j][i:i + bs, d - lo] = out[:, j - 1]
                    if ENS is not None and j in ENS: ENS[j][i:i + bs, d - lo] = mem[:, :, j - 1].float().cpu().numpy()
                    if mem is not None:
                        m_ = mem[:, :, j - 1]; obs = S.Q[bb, d]; ok = (torch.isfinite(obs) & good)
                        K_ = m_.shape[1]; c = (m_ - torch.nan_to_num(obs)[:, None]).abs().mean(1) - (m_[:, :, None] - m_[:, None, :]).abs().sum((1, 2)) / (2 * K_ * (K_ - 1))
                        cr[j][i:i + bs] += (c * ok).double().cpu().numpy(); cn[j][i:i + bs] += ok.double().cpu().numpy(); sp[j][i:i + bs] += (m_.std(1) * ok).double().cpu().numpy()
    predict.ens = ENS
    if PROB: predict.extra = {j: dict(crps=cr[j] / np.maximum(cn[j], 1), spread=sp[j] / np.maximum(cn[j], 1)) for j in range(1, N + 1)}
    return sims

_clim = {}
def clim_table(basins):
    """(B,366) day-of-year mean discharge from pre-2016 observations (the 'climatology' reference)."""
    key = basins.tobytes()
    if key not in _clim:
        tr = np.datetime64('1980-01-01') + np.arange(i16); dd = (tr - tr.astype('datetime64[Y]')).astype(int)
        o = Q[basins, :i16]; m = np.isfinite(o); tab = np.zeros((len(basins), 366)); cn = np.zeros((len(basins), 366))
        for d in range(366):
            sel = dd == d; tab[:, d] = np.where(m[:, sel], o[:, sel], 0).sum(1); cn[:, d] = m[:, sel].sum(1)
        _clim[key] = np.where(cn > 0, tab / np.maximum(cn, 1), np.nan)
    return _clim[key]

def score(sim, basins, lo, hi):
    obs = Q[basins, lo:hi + 1]
    dates = np.datetime64('1980-01-01') + np.arange(lo, hi + 1); dd = (dates - dates.astype('datetime64[Y]')).astype(int)
    clim = clim_table(basins)[:, dd]
    res = dict(kge=[], nse=[], rmse=[], clim_rmse=[], beta=[], mae=[], r=[], gamma=[])
    for r in range(len(basins)):
        o, sm = obs[r], sim[r]; m = np.isfinite(o) & np.isfinite(sm)
        res['r'].append(float(np.corrcoef(o[m], sm[m])[0, 1]) if m.sum() > 30 and o[m].std() > 0 and sm[m].std() > 0 else np.nan)
        res['gamma'].append(float((sm[m].std() / sm[m].mean()) / (o[m].std() / o[m].mean())) if m.sum() > 30 and o[m].mean() != 0 and sm[m].mean() != 0 and o[m].std() > 0 else np.nan)
        res['kge'].append(kge_prime(o, sm)); res['nse'].append(nse(o, sm)); res['rmse'].append(rmse(o, sm))
        res['mae'].append(float(np.mean(np.abs(o[m] - sm[m]))) if m.sum() > 30 else np.nan); res['clim_rmse'].append(rmse(o, clim[r])); res['beta'].append(np.nanmean(sm[m]) / np.nanmean(o[m]) if m.sum() > 30 and np.nanmean(o[m]) else np.nan)
    med = {k: float(np.nanmedian(v)) for k, v in res.items()}
    med['n'] = int(np.isfinite(res['kge']).sum()); med['frac_rmse_lt_clim'] = float(np.nanmean(np.array(res['rmse']) < np.array(res['clim_rmse'])))
    return med, res

# ---------------------------------------------------------------- run
if a.arch == 'single': model = RiverLSTM(F, static.shape[1], a.hidden, noise_dim=a.noise_dim, noise_last=a.noise_last).to(dev)
else: model = TwoClockLSTM(F, static.shape[1], a.hidden, a.hidden_s, a.weeks, a.fast_len, noise_dim=a.noise_dim).to(dev)
STOCH = a.noise_dim > 0
print(f'arch={a.arch} window={L}d params={sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)
def load_padded(model, path):
    sd = torch.load(path, map_location=dev); cur = model.state_dict()
    for k, v in sd.items():
        if v.shape != cur[k].shape:
            assert v.dim() == 2 and v.shape[0] == cur[k].shape[0] and v.shape[1] < cur[k].shape[1], f'cannot pad {k}: {v.shape} -> {cur[k].shape}'
            v = torch.cat([v, v.new_zeros(v.shape[0], cur[k].shape[1] - v.shape[1])], 1); print(f'  padded {k} -> {tuple(v.shape)}', flush=True)
        cur[k] = v
    model.load_state_dict(cur)
if a.ft_ckpt: load_padded(model, a.ft_ckpt)
else:
    if a.pre_ckpt: load_padded(model, a.pre_ckpt)
    else:
        train(model, pre_pool, L - 1, i20 - 1, 'era', a.pre_epochs, a.pre_upd, a.lr, 'pre')
        torch.save(model.state_dict(), f'{R}/ckpt/{a.name}_pre.pt')
        if a.pre_only: print('DONE pre-only', a.name, flush=True); sys.exit(0)
    if a.ft == 'lt':
        train(model, ft_pool, FOFF + 179, i20 - 1, 'lt', a.ft_epochs, a.ft_upd, a.ft_lr, 'ft-lt1', k=1)
    elif a.ft == 'traj':
        train(model, ft_pool, FOFF + 169, i20 - 1 - N, 'traj', a.ft_epochs, a.ft_upd, a.ft_lr, 'ft-traj')
    elif a.ft == 'mix':
        train(model, ft_pool, 0, 0, 'mix', a.ft_epochs, a.ft_upd, a.ft_lr, 'ft-mix')
    torch.save(model.state_dict(), f'{R}/ckpt/{a.name}_ft.pt')

if a.split == 'test':
    ev_pool = test_pool; lo, hi = max(i21, FOFF + 180), min(i_end, FOFF + FN - 1) - N      # keep every lead on the same target days
    if a.gc_window: hi = min(hi, d2i('2023-12-21') - N)
else:                    # validation year 2020: never used for training; basins with >=180 observed days
    ev_pool = test_pool[cnt(i20, i21)[test_pool] >= 180]; lo, hi = i20, i21 - 1 - N
if a.dump_names:
    want = a.dump_names.split(','); ev_pool = np.array([i for i in ev_pool if names[i] in want]); assert len(ev_pool) == len(want), (len(ev_pool), len(want))
out = dict(args=vars(a), n_test=int(len(ev_pool)), split=a.split, leads={}, leads_traj={}); PB = {}
def _score(sim, key):
    med, res = score(sim, ev_pool, lo, hi)
    for m in ('kge', 'nse', 'rmse', 'clim_rmse', 'beta', 'mae', 'r', 'gamma'): PB[f'{key}_{m}'] = np.array(res[m], np.float32)
    ex = getattr(predict, 'extra', None)
    if ex and key.startswith('traj'):
        j = int(key[4:]); PB[f'{key}_crps'] = ex[j]['crps'].astype(np.float32); PB[f'{key}_spread'] = ex[j]['spread'].astype(np.float32)
        med['crps'] = float(np.nanmedian(ex[j]['crps'])); med['spread'] = float(np.nanmedian(ex[j]['spread']))
    return med
if a.ft != 'traj' and not a.traj_only:      # AIFL protocol: LT-k series in every step of the window (also for the LT1/trajectory mix)
    for k in range(1, N + 1):
        out['leads'][k] = _score(predict(model, ev_pool, lo, hi, 'lt', k=k), f'lt{k}')
        print('  lt-protocol lead', k, {m: round(v, 3) for m, v in out['leads'][k].items()}, flush=True)
if a.ft in ('traj', 'mix') or a.eval_traj:   # operational protocol: past = LT1, future day t+j = lead j
    S.zero_extra = a.ft not in ('traj', 'mix')   # a model never trained with the flag/lead channels must not see them
    SIMS = predict(model, ev_pool, lo, hi, 'traj')
    for j, sm in SIMS.items():
        out['leads_traj'][j] = _score(sm, f'traj{j}')
        print('  traj-protocol lead', j, {m: round(v, 3) for m, v in out['leads_traj'][j].items()}, flush=True)
    S.zero_extra = False
if a.dump_all:
    dd = {f'mean{j}': SIMS[j] for j in SIMS}
    if a.dump_members and getattr(predict, 'ens', None) is not None: dd.update({f'ens{j}': predict.ens[j] for j in predict.ens})
    if a.dump_era:
        a.K_eval = a.K_sim; lo_e = L + 10; hi_e = S.Te - 1
        dd['era_sim'] = predict(model, ev_pool, lo_e, hi_e, 'era'); dd['era_lo'] = np.array(lo_e)
    np.savez(f'{R}/results/dump_{a.name}.npz', basins=np.array([names[i] for i in ev_pool]), dates=(np.datetime64('1980-01-01') + np.arange(lo, hi + 1)).astype('datetime64[D]'), lo=np.array(lo), obs=Q[ev_pool, lo:hi + 1], **dd)
    print('DONE dump_all', a.name, flush=True); sys.exit(0)
if a.dump_names:
    dd = {f'mean{j}': SIMS[j] for j in SIMS}; ens = getattr(predict, 'ens', None)
    if ens is not None: dd.update({f'ens{j}': ens[j] for j in ens})
    np.savez(f'{R}/results/dump_{a.name}.npz', basins=np.array([names[i] for i in ev_pool]), dates=(np.datetime64('1980-01-01') + np.arange(lo, hi + 1)).astype('datetime64[D]'), obs=Q[ev_pool, lo:hi + 1], **dd)
    print('DONE dump', a.name, flush=True); sys.exit(0)
out['era5_oracle'] = _score(predict(model, ev_pool, lo, hi, 'era'), 'era')   # ERA5-Land-driven upper reference (not a forecast)
json.dump(out, open(f'{R}/results/{a.name}.json', 'w'), indent=1)
np.savez(f'{R}/results/{a.name}_perbasin.npz', basins=np.array([names[i] for i in ev_pool]), **PB)
print('DONE', a.name, 'lt', {j: round(v['kge'], 3) for j, v in out['leads'].items()}, 'traj', {j: round(v['kge'], 3) for j, v in out['leads_traj'].items()}, 'oracle', round(out['era5_oracle']['kge'], 3), flush=True)
