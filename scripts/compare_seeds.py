"""Seed-averaged paired comparison of two model families.  usage: compare_seeds.py A0,A1,A2 B0,B1,B2   (A, B = result names, e.g. s424fsm_s0,... ; results/<name>_perbasin.npz)"""
import sys, numpy as np
A = sys.argv[1].split(','); B = sys.argv[2].split(',')
LA = [np.load(f'results/{n}_perbasin.npz') for n in A]; LB = [np.load(f'results/{n}_perbasin.npz') for n in B]
for z in LA + LB: assert (z['basins'] == LA[0]['basins']).all()
rng = np.random.default_rng(0)
for proto in ('traj', 'lt'):
    print(f'\n=== {proto} protocol (A={A[0][:-1]}*, B={B[0][:-1]}*): seed-averaged per-basin metrics, paired bootstrap ===')
    for m, sg in (('kge', 1), ('nse', 1), ('rmse', -1)):
        wins = worse = 0; lines = []
        for j in range(1, 11):
            a = np.nanmean([z[f'{proto}{j}_{m}'] for z in LA], 0); b = np.nanmean([z[f'{proto}{j}_{m}'] for z in LB], 0); ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
            ii = rng.integers(0, len(a), (2000, len(a))); d = np.median(a[ii], 1) - np.median(b[ii], 1); lo, hi = np.percentile(d, [2.5, 97.5])
            good = (lo * sg > 0) if sg > 0 else (hi < 0); bad = (hi * sg < 0) if sg > 0 else (lo > 0); wins += good; worse += bad
            if j in (1, 5, 10): lines.append(f'LT{j}: {np.median(a):.3f} vs {np.median(b):.3f} (d {np.median(a) - np.median(b):+.3f} [{lo:+.3f},{hi:+.3f}])')
        print(f'  {m.upper():4s} A better {wins}/10, A worse {worse}/10 | ' + ' ; '.join(lines))
