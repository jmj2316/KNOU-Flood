"""Probabilistic verification of the final model vs. fair probabilistic versions of the AIFL-recipe baseline (test 2021 - 2023-12, leads 1/3/5/7/10).
Systems (all K=16 members, physical units mm/d):
  ours_raw      final model (latent-noise ensemble), as trained
  ours_infl     ours with member deviations from the ensemble mean inflated by lambda_j (one scalar per lead, fitted on validation 2020 by CRPS); mean unchanged
  b2d_mc        AIFL-recipe baseline, MC-dropout members (dropout 0.4 kept on)
  b2d_mc_infl   same, with the same validation-fitted inflation
  b2d_dress_*   deterministic baseline + residual dressing (quantiles of the gauge/lead-specific residuals of validation 2020; sqrt space / additive)
  clim          day-of-year climatological ensemble (1980-2015, +-15 days)
usage: prob_eval.py [seeds e.g. 0,1,2]     (needs results/dump_pm_{s424,b2d}_s{S}_{val,test}.npz)"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, numpy as np, json
sys.path.insert(0, ROOT + '/scripts'); import probmetrics as PM
R = PM.R; LEADS = PM.LEADS; seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '1').split(',')]
T = np.load(f'{R}/data/arrays/targets.npz', allow_pickle=True); tn = list(T['names']); Qall = T['Q']
GRID = np.arange(0.5, 6.01, 0.25); rng = np.random.default_rng(0)
import os
MC = all(os.path.exists(f'{R}/results/dump_pm_s424mc_s{sd}_{sp}.npz') for sd in seeds for sp in ('val', 'test'))      # MC dropout on top of the latent noise (final model)
SYS = ['ours_raw', 'ours_infl'] + (['ours_mc', 'ours_mc_infl'] if MC else []) + ['ours_dress_sqrt', 'b2d_mc', 'b2d_mc_infl', 'b2d_dress_sqrt', 'b2d_dress_add', 'clim']
def members_for(system, seed, lead, split, Zs, val_fit=None):
    Z = Zs[(system.split('_')[0], seed, split)]; return Z[f'ens{lead}']
res = {}; hist = {}; lam = {}
for sd in seeds:
    Zs = {(m, sd, sp): PM.load_dump(f'pm_{m}_s{sd}_{sp}') for m in (('s424', 'b2d', 's424mc') if MC else ('s424', 'b2d')) for sp in ('val', 'test')}
    bt = Zs[('s424', sd, 'test')]['basins']; bv = Zs[('s424', sd, 'val')]['basins']; dates = Zs[('s424', sd, 'test')]['dates']; dv = Zs[('s424', sd, 'val')]['dates']
    assert (Zs[('b2d', sd, 'test')]['basins'] == bt).all() and (Zs[('b2d', sd, 'val')]['basins'] == bv).all()
    y = Zs[('s424', sd, 'test')]['obs']; yv = Zs[('s424', sd, 'val')]['obs']; vpos = {g: i for i, g in enumerate(bv)}; inval = np.array([g in vpos for g in bt])
    cc_t = PM.clim_crps(bt, dates, y, Qall, tn); cc_v = PM.clim_crps(bv, dv, yv, Qall, tn); print(f'seed {sd}: test basins {len(bt)}, validation basins {len(bv)} (overlap {inval.sum()})', flush=True)
    sig_v = np.nanstd(yv, 1)
    for lead in LEADS:
        E = {'s424': (Zs[('s424', sd, 'val')][f'ens{lead}'], Zs[('s424', sd, 'test')][f'ens{lead}']), 'b2d': (Zs[('b2d', sd, 'val')][f'ens{lead}'], Zs[('b2d', sd, 'test')][f'ens{lead}'])}
        if MC: E['s424mc'] = (Zs[('s424mc', sd, 'val')][f'ens{lead}'], Zs[('s424mc', sd, 'test')][f'ens{lead}'])
        def fit_lambda(ev):
            best, bl = 1e9, 1.0
            for l in GRID:
                c = np.concatenate([np.nanmean(PM.crps_fair(PM.inflate(ev[s].astype(np.float32), l), yv[s]), 1) for s in PM.chunks(len(bv))])
                o = np.nanmean(c / (sig_v + 0.1))
                if o < best: best, bl = o, l
            return bl
        lam[(sd, 'ours', lead)] = fit_lambda(E['s424'][0]); lam[(sd, 'b2d_mc', lead)] = fit_lambda(E['b2d'][0])
        if MC: lam[(sd, 'ours_mc', lead)] = fit_lambda(E['s424mc'][0])
        # dressing from validation residuals of the deterministic (dropout-free) baseline
        vm, vo = Zs[('b2d', sd, 'val')][f'mean{lead}'], yv; tm = Zs[('b2d', sd, 'test')][f'mean{lead}']
        dr = {sp: PM.dress_members(vm, vo, tm, space=sp) for sp in ('sqrt', 'add')}   # fitted on val gauges; map val basins -> test basins below
        for sp in ('sqrt', 'add'):
            Qv, f, fi = dr[sp]; Qt = np.tile(np.median(Qv, 0), (len(bt), 1)); Qt[inval] = Qv[[vpos[g] for g in bt[inval]]]
            dr[sp] = (Qt, f, fi)
        vmo, tmo = Zs[('s424', sd, 'val')][f'mean{lead}'], Zs[('s424', sd, 'test')][f'mean{lead}']        # our ensemble mean, dressed exactly like the baseline
        Qo, fo, fio = PM.dress_members(vmo, yv, None, space='sqrt'); Qot = np.tile(np.median(Qo, 0), (len(bt), 1)); Qot[inval] = Qo[[vpos[g] for g in bt[inval]]]
        systems = {'ours_dress_sqrt': lambda sl: PM.apply_dress(Qot[sl], fo, fio, tmo[sl]), 'ours_raw': lambda sl: E['s424'][1][sl].astype(np.float32), 'ours_infl': lambda sl: PM.inflate(E['s424'][1][sl].astype(np.float32), lam[(sd, 'ours', lead)]),
                   'b2d_mc': lambda sl: E['b2d'][1][sl].astype(np.float32), 'b2d_mc_infl': lambda sl: PM.inflate(E['b2d'][1][sl].astype(np.float32), lam[(sd, 'b2d_mc', lead)]),
                   'b2d_dress_sqrt': lambda sl: PM.apply_dress(dr['sqrt'][0][sl], dr['sqrt'][1], dr['sqrt'][2], tm[sl]), 'b2d_dress_add': lambda sl: PM.apply_dress(dr['add'][0][sl], dr['add'][1], dr['add'][2], tm[sl])}
        if MC: systems.update({'ours_mc': lambda sl: E['s424mc'][1][sl].astype(np.float32), 'ours_mc_infl': lambda sl: PM.inflate(E['s424mc'][1][sl].astype(np.float32), lam[(sd, 'ours_mc', lead)])})
        for name, fn in systems.items():
            acc = {k: [] for k in ('crps', 'crps_clim', 'rmse', 'spread', 'cover80', 'n')}; h = 0
            for sl in PM.chunks(len(bt)):
                e = fn(sl); m = e.mean(2); s_ = PM.summarize(e, m, y[sl], None, cc_t[sl]); [acc[k].append(s_[k]) for k in acc]; h = h + PM.rank_hist(e, y[sl], rng)
            res[(sd, name, lead)] = {k: np.concatenate(v) for k, v in acc.items()}; hist[(sd, name, lead)] = h
        res[(sd, 'clim', lead)] = {'crps': np.nanmean(cc_t, 1), 'crps_clim': np.nanmean(cc_t, 1)}
        print(f'  seed {sd} lead {lead}: lambda ours {lam[(sd, "ours", lead)]:.2f}, b2d-mc {lam[(sd, "b2d_mc", lead)]:.2f}', flush=True)
# ---------------- report
def med(v): return float(np.nanmedian(v))
print('\n==== medians over gauges, mean over seeds %s   (CRPS mm/d; CRPSS vs day-of-year climatology; spread/RMSE ideal 1; q10-q90 coverage ideal 12/17=0.706 for K=16; obs outside ensemble range ideal %.3f) ====' % (seeds, 2 / 17))
out = {}; climd = {}
for lead in LEADS:
    print(f'\n--- lead {lead} ---\n{"system":16s} {"CRPS":>7s} {"CRPSS":>7s} {"spread/RMSE":>12s} {"cover80":>8s} {"outside":>8s} {"RMSE(mean)":>11s}')
    for name in SYS:
        r = [res[(sd, name, lead)] for sd in seeds]; crps = np.nanmean([x['crps'] for x in r], 0)
        if name == 'clim': print(f'{name:16s} {med(crps):7.3f} {0:7.3f}'); climd[str(lead)] = med(crps); continue
        cs = np.nanmean([1 - x['crps'] / x['crps_clim'] for x in r], 0); sr = np.nanmean([x['spread'] / x['rmse'] for x in r], 0); cv = np.nanmean([x['cover80'] for x in r], 0); rm = np.nanmean([x['rmse'] for x in r], 0)
        hh = sum(hist[(sd, name, lead)] for sd in seeds); out_frac = (hh[0] + hh[-1]) / hh.sum()
        print(f'{name:16s} {med(crps):7.3f} {med(cs):7.3f} {med(sr):12.3f} {med(cv):8.3f} {out_frac:8.3f} {med(rm):11.3f}')
        out[f'{name}_{lead}'] = dict(crps=med(crps), crpss=med(cs), spread_rmse=med(sr), cover80=med(cv), outside=float(out_frac), rmse=med(rm))
print('\n==== paired bootstrap over gauges (seed-mean CRPS, median difference A - B, 95% CI; negative = A better) ====')
for lead in LEADS:
    g = lambda n: np.nanmean([res[(sd, n, lead)]['crps'] for sd in seeds], 0)
    for a_, b_ in (('ours_raw', 'b2d_mc_infl'), ('ours_raw', 'b2d_dress_sqrt'), ('ours_infl', 'b2d_mc_infl'), ('ours_infl', 'b2d_dress_sqrt'), ('ours_dress_sqrt', 'b2d_dress_sqrt'), ('ours_mc', 'b2d_mc_infl'), ('ours_mc_infl', 'b2d_mc_infl'), ('ours_mc_infl', 'b2d_dress_sqrt'), ('ours_mc_infl', 'ours_infl')):
        if a_ not in SYS: continue
        a, b = g(a_), g(b_); ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]; ii = np.random.default_rng(0).integers(0, len(a), (2000, len(a))); d = np.median(a[ii] - b[ii], 1); lo, hi = np.percentile(d, [2.5, 97.5])
        print(f'  lead {lead:2d}  {a_:16s} - {b_:16s}: median CRPS {np.median(a):.3f} vs {np.median(b):.3f}, paired median diff {np.median(a - b):+.4f} [{lo:+.4f},{hi:+.4f}]  better at {np.mean(a < b) * 100:.0f}% of gauges')
np.savez(f'{R}/results/prob_eval_perbasin.npz', **{f'{sd}|{n}|{l}|{k}': v for (sd, n, l), d in res.items() for k, v in d.items()}, **{f'hist|{sd}|{n}|{l}': h for (sd, n, l), h in hist.items()})
json.dump(dict(summary=out, clim_crps=climd, lam={f'{k[0]}|{k[1]}|{k[2]}': float(v) for k, v in lam.items()}, seeds=seeds), open(f'{R}/results/prob_eval.json', 'w'), indent=1)
