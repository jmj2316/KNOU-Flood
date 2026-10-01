"""Does the upper-tail-weighted CRPS fine-tuning (tw_s*) help flood events?  Compares the MC-dropout ensembles of tw (pm_twmc_*) with the standard final model (pm_s424mc_*), both with
validation-fitted spread scaling lambda and p* tuned on 2020, on observed (single-threshold) return levels; per-seed CSI, plus CRPS / calibration.   usage: tw_eval.py [seeds=0,1,2]"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, json, numpy as np
sys.path.insert(0, ROOT + '/scripts'); import probmetrics as PM, event_metrics as EM
R = PM.R; LEADS = PM.LEADS; RPS = EM.RPS; seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '0,1,2').split(',')]; GRID = np.arange(0.5, 6.01, 0.25)
T = np.load(f'{R}/data/arrays/targets.npz', allow_pickle=True); tn = list(T['names']); Qall = T['Q']
def counts(ens_fn, y, thr, B, K=16):
    out = {ri: [np.zeros(K + 1, np.int64), np.zeros(K + 1, np.int64)] for ri in range(len(RPS))}
    for sl in PM.chunks(B):
        e = ens_fn(sl); yy = y[sl]; fin = np.isfinite(yy) & np.isfinite(e).all(2)
        for ri in range(len(RPS)):
            th = thr[sl, ri]; ok = fin & np.isfinite(th)[:, None]
            if not ok.any(): continue
            lv = np.rint((e >= th[:, None, None]).mean(2) * K).astype(int); ev = yy >= th[:, None]
            out[ri][0] += np.bincount(lv[ok & ev], minlength=K + 1); out[ri][1] += np.bincount(lv[ok & ~ev], minlength=K + 1)
    return out
def csi(n1, n0, lv): h, m, f = n1[lv:].sum(), n1[:lv].sum(), n0[lv:].sum(); return h / (h + m + f) if h + m + f else np.nan
res = {}; crps = {}
for name in ('s424mc', 'twmc'):
    for sd in seeds:
        Zv, Zt = PM.load_dump(f'pm_{name}_s{sd}_val'), PM.load_dump(f'pm_{name}_s{sd}_test'); bt, bv = Zt['basins'], Zv['basins']; yv, yt = Zv['obs'], Zt['obs']
        thr_t = np.load(f'{R}/results/thr_cache_s{sd}.npz')['obs']; tpos = {g: i for i, g in enumerate(bt)}; keep = np.array([g in tpos for g in bv]); thr_v = thr_t[[tpos[g] for g in bv[keep]]]
        sig = np.nanstd(yv, 1)
        for lead in LEADS:
            ev, et = Zv[f'ens{lead}'], Zt[f'ens{lead}']; best = (1e9, 1.0)
            for l in GRID:
                c = np.concatenate([np.nanmean(PM.crps_fair(PM.inflate(ev[s].astype(np.float32), l), yv[s]), 1) for s in PM.chunks(len(bv))]); o = np.nanmean(c / (sig + 0.1))
                if o < best[0]: best = (o, l)
            lam = best[1]; fn = lambda sl, E=et, lam=lam: PM.inflate(E[sl].astype(np.float32), lam)
            evk = ev[keep]; cv = counts(lambda sl: PM.inflate(evk[sl].astype(np.float32), lam), yv[keep], thr_v, int(keep.sum())); ct = counts(fn, yt, thr_t, len(bt))
            for ri in range(len(RPS)):
                grp = (0, 1) if ri < 2 else (2, 3, 4, 5); K = len(ct[ri][0]) - 1
                bl = max(range(1, K + 1), key=lambda lv: np.nanmean([csi(*cv[r], lv) for r in grp])); res[(name, sd, lead, ri)] = csi(*ct[ri], bl)
            cr = np.concatenate([np.nanmean(PM.crps_fair(PM.inflate(et[s].astype(np.float32), lam), yt[s]), 1) for s in PM.chunks(len(bt))]); crps[(name, sd, lead)] = (float(np.nanmedian(cr)), lam)
            print(f'{name} seed {sd} lead {lead} done (lambda {lam})', flush=True)
J = json.load(open(f'{R}/results/event_members_report.json'))
print('\nCSI, observed return levels, MC-dropout ensembles with spread scaling, p* tuned on 2020. columns per seed: standard final | tail-weighted (tw) | best AIFL-recipe baseline system')
for lead in LEADS:
    print(f'\n--- lead {lead} ---')
    for ri, rp in enumerate(RPS):
        base = np.max([J[f'single|{lead}|{rp:g}|{n}|{m}']['per_seed'] for n, m in (('b2d_det', 'det'), ('b2d_mc', 'tuned'), ('b2d_mc_infl', 'tuned'), ('b2d_dress_sqrt', 'tuned'))], 0)
        print(f'  RP{rp:>4}: ' + ' | '.join(f'seed{sd}: {res[("s424mc", sd, lead, ri)]:.3f} {res[("twmc", sd, lead, ri)]:.3f} {base[sd]:.3f}' for sd in seeds))
print('\nCRPS (median over gauges, mm/d; with lambda): standard final vs tw')
for lead in LEADS: print(f'  lead {lead}: ' + ' | '.join(f'seed{sd}: {crps[("s424mc", sd, lead)][0]:.3f} vs {crps[("twmc", sd, lead)][0]:.3f}' for sd in seeds))
