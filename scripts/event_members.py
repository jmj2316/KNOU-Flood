"""Member-based flood-event forecasts vs. ensemble-mean / deterministic event forecasts (test 2021 - 2023-12), using the dumps of the final model and the AIFL-recipe baseline.
An event = discharge >= return level RP (Gumbel of the observed annual maxima, RP in 1.5 ... 50 yr), exact-day matching, pooled over gauges and days (same definitions as event_metrics.py).
Forecast event:  'mean' systems: point forecast >= threshold;  member systems: fraction of members >= threshold is >= p*  (p* tuned per system/lead/RP on validation 2020, never on test).
single threshold = observed return level;  dual = return level of the model's own ERA5-driven simulation.
Contingency counts are kept per probability level (K+1 levels), so any p*, POD/precision/CSI, ROC-AUC and Brier score follow from them.
usage: event_members.py [seeds]   (needs prob_eval.py output results/prob_eval.json for the inflation factors)"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, json, numpy as np, os
sys.path.insert(0, ROOT + '/scripts'); import probmetrics as PM, event_metrics as EM
MCMODE = os.environ.get('EVT_MC') == '1'          # EVT_MC=1: only the MC-dropout-on-top variant of the final model -> results/event_members_counts_mc.pkl
PKL = 'event_members_counts_mc.pkl' if MCMODE else 'event_members_counts.pkl'; MODELS = ('s424mc',) if MCMODE else ('s424', 'b2d')
R = PM.R; LEADS = PM.LEADS; RPS = EM.RPS; seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '1').split(',')]
T = np.load(f'{R}/data/arrays/targets.npz', allow_pickle=True); tn = list(T['names']); Qall = T['Q']; alld = np.datetime64('1980-01-01') + np.arange(Qall.shape[1])
LAM = json.load(open(f'{R}/results/prob_eval.json'))['lam']
def evt(model, sd): return f'evt_{model}' + ('' if sd == 0 else f'_s{sd}')
def thresholds(basins, sd):
    cp = f'{R}/results/thr_cache_s{sd}.npz'
    if os.path.exists(cp): z = np.load(cp); assert (z['basins'] == basins).all(); return z['obs'], z['ours'], z['b2d']
    nil = np.full(len(RPS), np.nan); ob = np.array([EM.thresholds(Qall[tn.index(g)], alld) if EM.thresholds(Qall[tn.index(g)], alld) is not None else nil for g in basins]); out = {}
    for key, m in (('ours', 's424'), ('b2d', 'b2d')):
        Z = np.load(f'{R}/results/dump_{evt(m, sd)}.npz'); assert (Z['basins'] == basins).all(); era = Z['era_sim']; ed = alld[int(Z['era_lo']):int(Z['era_lo']) + era.shape[1]]
        out[key] = np.array([EM.thresholds(era[k], ed) if EM.thresholds(era[k], ed) is not None else nil for k in range(len(basins))])
    np.savez(cp, basins=basins, obs=ob, ours=out['ours'], b2d=out['b2d']); return ob, out['ours'], out['b2d']
def level_counts(frac, ev, ok, K):
    lv = np.rint(frac * K).astype(int); return np.bincount(lv[ok & ev], minlength=K + 1), np.bincount(lv[ok & ~ev], minlength=K + 1)
def build_systems(sd, split, lead, Zs, lamfit, dress):
    """returns {name: fn(slice)->(members (b,T,K))}"""
    if MCMODE:
        E_c = Zs[('s424mc', split)][f'ens{lead}']; lc = lamfit['ours_mc']
        return {'ours_mc': (lambda sl: E_c[sl].astype(np.float32), 'ours', 16), 'ours_mc_infl': (lambda sl: PM.inflate(E_c[sl].astype(np.float32), lc), 'ours', 16)}
    E_o = Zs[('s424', split)][f'ens{lead}']; E_b = Zs[('b2d', split)][f'ens{lead}']; mo = Zs[('s424', split)][f'mean{lead}']; mb = Zs[('b2d', split)][f'mean{lead}']
    lo_, lb_ = lamfit['ours'], lamfit['b2d_mc']
    f = {'ours_mean': (lambda sl: mo[sl][..., None], 'ours', 1), 'b2d_det': (lambda sl: mb[sl][..., None], 'b2d', 1),
         'ours_raw': (lambda sl: E_o[sl].astype(np.float32), 'ours', 16), 'ours_infl': (lambda sl: PM.inflate(E_o[sl].astype(np.float32), lo_), 'ours', 16),
         'b2d_mc': (lambda sl: E_b[sl].astype(np.float32), 'b2d', 16), 'b2d_mc_infl': (lambda sl: PM.inflate(E_b[sl].astype(np.float32), lb_), 'b2d', 16)}
    for sp in ('sqrt', 'add'):
        Qd, fq, fi = dress[sp]; f[f'b2d_dress_{sp}'] = ((lambda sl, Qd=Qd, fq=fq, fi=fi: PM.apply_dress(Qd[sl], fq, fi, mb[sl])), 'b2d', 16)
    return f
import pickle
res = pickle.load(open(f'{R}/results/{PKL}', 'rb')) if os.path.exists(f'{R}/results/{PKL}') else {}    # merge: seeds already in the file are skipped
for sd in seeds:
    if any(k[0] == sd for k in res): print(f'seed {sd}: already computed, skipping', flush=True); continue
    # thresholds are defined on the test gauge list; validation gauges are matched by id
    Zt = {m: PM.load_dump(f'pm_{m}_s{sd}_test') for m in MODELS + (('s424',) if MCMODE else ())}; bt = Zt['s424']['basins']; thr_obs, thr_o, thr_b = thresholds(bt, sd); tpos = {g: i for i, g in enumerate(bt)}
    Zv = {m: PM.load_dump(f'pm_{m}_s{sd}_val') for m in MODELS + (('s424',) if MCMODE else ())}; bv = Zv['s424']['basins']; yv = Zv['s424']['obs']; yt = Zt['s424']['obs']
    keep = np.array([g in tpos for g in bv]); vi = np.array([tpos[g] for g in bv[keep]])            # validation gauges that are test gauges (thresholds needed)
    for split, Z, bas, y, tix in (('val', Zv, bv[keep], yv[keep], vi), ('test', Zt, bt, yt, np.arange(len(bt)))):
        to, tob, tb_ = thr_o[tix], thr_obs[tix], thr_b[tix]
        for lead in LEADS:
            lamfit = {'ours_mc': LAM[f'{sd}|ours_mc|{lead}']} if MCMODE else {'ours': LAM[f'{sd}|ours|{lead}'], 'b2d_mc': LAM[f'{sd}|b2d_mc|{lead}']}
            # dressing fitted on validation residuals (val gauges); test gauges without validation data -> median residual quantiles
            dress = {}
            for sp in (() if MCMODE else ('sqrt', 'add')):
                vm, vo = Zv['b2d'][f'mean{lead}'], yv; Qv, fq, fi = PM.dress_members(vm, vo, None, space=sp); vpos = {g: i for i, g in enumerate(bv)}
                Qs = np.tile(np.median(Qv, 0), (len(bas), 1)); ii = [i for i, g in enumerate(bas) if g in vpos]; Qs[ii] = Qv[[vpos[bas[i]] for i in ii]]; dress[sp] = (Qs, fq, fi)
            sel = keep if split == 'val' else slice(None)
            Zs = {(m, split): {f'ens{lead}': Z[m][f'ens{lead}'][sel], f'mean{lead}': Z[m][f'mean{lead}'][sel]} for m in MODELS}
            systems = build_systems(sd, split, lead, Zs, lamfit, dress)
            for name, (fn, which, K) in systems.items():
                thr_fc_dual = to if which == 'ours' else tb_
                cnt = {(ri, kind): [np.zeros(K + 1, np.int64), np.zeros(K + 1, np.int64)] for ri in range(len(RPS)) for kind in ('single', 'dual')}
                for sl in PM.chunks(len(bas)):
                    e = fn(sl); yy = y[sl]; fin = np.isfinite(yy) & np.isfinite(e).all(2)
                    for ri in range(len(RPS)):
                        for kind, thf in (('single', tob[sl, ri]), ('dual', thr_fc_dual[sl, ri])):
                            okg = np.isfinite(tob[sl, ri]) & np.isfinite(thf); ok = fin & okg[:, None]
                            if not ok.any(): continue
                            frac = (e >= thf[:, None, None]).mean(2); ev = yy >= tob[sl, ri][:, None]
                            n1, n0 = level_counts(frac, ev, ok, K); cnt[(ri, kind)][0] += n1; cnt[(ri, kind)][1] += n0
                for (ri, kind), (n1, n0) in cnt.items(): res[(sd, split, name, lead, ri, kind)] = (n1, n0)
            print(f'seed {sd} {split} lead {lead} done', flush=True)
pickle.dump(res, open(f'{R}/results/{PKL}', 'wb')); print('saved counts', len(res))
