"""Does adding MC dropout on top of the latent noise (final model, seed 1) improve calibration / CRPS?  Validation-fitted spread scaling as in prob_eval.py."""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, numpy as np
sys.path.insert(0, ROOT + '/scripts'); import probmetrics as PM
R = PM.R; sd = 1; rng = np.random.default_rng(0)
T = np.load(f'{R}/data/arrays/targets.npz', allow_pickle=True); tn = list(T['names']); Qall = T['Q']
Z = {(v, sp): PM.load_dump(f'pm_{v}_s{sd}_{sp}') for v in ('s424', 's424mc') for sp in ('val', 'test')}
bt = Z[('s424', 'test')]['basins']; bv = Z[('s424', 'val')]['basins']; y = Z[('s424', 'test')]['obs']; yv = Z[('s424', 'val')]['obs']; dates = Z[('s424', 'test')]['dates']; sig = np.nanstd(yv, 1)
cc = PM.clim_crps(bt, dates, y, Qall, tn)
print(f'{"lead":>4s} {"system":14s} {"lambda":>6s} {"CRPS":>7s} {"CRPSS":>7s} {"spread/RMSE":>12s} {"q10-q90":>8s} {"outside":>8s}')
for lead in (1, 5, 10):
    for v in ('s424', 's424mc'):
        ev, et = Z[(v, 'val')][f'ens{lead}'], Z[(v, 'test')][f'ens{lead}']
        best = (1e9, 1.0)
        for l in np.arange(0.5, 6.01, 0.25):
            c = np.concatenate([np.nanmean(PM.crps_fair(PM.inflate(ev[s].astype(np.float32), l), yv[s]), 1) for s in PM.chunks(len(bv))]); o = np.nanmean(c / (sig + 0.1))
            if o < best[0]: best = (o, l)
        for lam, tag in ((1.0, 'raw'), (best[1], 'infl')):
            acc = {k: [] for k in ('crps', 'crps_clim', 'rmse', 'spread', 'cover80', 'n')}; h = 0
            for sl in PM.chunks(len(bt)):
                e = PM.inflate(et[sl].astype(np.float32), lam) if tag == 'infl' else et[sl].astype(np.float32); s_ = PM.summarize(e, e.mean(2), y[sl], None, cc[sl]); [acc[k].append(s_[k]) for k in acc]; h = h + PM.rank_hist(e, y[sl], rng)
            a = {k: np.concatenate(v) for k, v in acc.items()}
            print(f'{lead:>4d} {v + "_" + tag:14s} {lam:6.2f} {np.nanmedian(a["crps"]):7.3f} {np.nanmedian(1 - a["crps"] / a["crps_clim"]):7.3f} {np.nanmedian(a["spread"] / a["rmse"]):12.3f} {np.nanmedian(a["cover80"]):8.3f} {(h[0] + h[-1]) / h.sum():8.3f}')
