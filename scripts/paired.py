"""Paired bootstrap over basins between two per-basin result files (same basin list).
usage: paired.py A_perbasin.npz B_perbasin.npz [proto=traj] [nboot=2000]   -> Δ = A − B ; positive KGE/NSE and negative RMSE favour A."""
import sys, numpy as np
A = np.load(sys.argv[1]); B = np.load(sys.argv[2]); proto = sys.argv[3] if len(sys.argv) > 3 else 'traj'; nb = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
assert (A['basins'] == B['basins']).all(), 'different basin lists'
rng = np.random.default_rng(0); n = len(A['basins']); idx = rng.integers(0, n, (nb, n))
print(f'{proto}: median Δ (A−B) with 95% paired-bootstrap CI over {n} basins')
allw = {m: [] for m in ('kge', 'nse', 'rmse')}
for m, sgn in (('kge', 1), ('nse', 1), ('rmse', -1)):
    print(f'\n  {m.upper()}   lead   A       B       Δ(A−B)   CI95              A better?')
    for j in range(1, 11):
        a, b = A[f'{proto}{j}_{m}'], B[f'{proto}{j}_{m}']; ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]; ii = rng.integers(0, len(a), (nb, len(a)))
        d = np.median(a[ii], 1) - np.median(b[ii], 1); lo, hi = np.percentile(d, [2.5, 97.5]); d0 = np.median(a) - np.median(b)
        better = (lo * sgn > 0) if sgn > 0 else (hi < 0)
        allw[m].append((d0, lo, hi, better))
        print(f'         {j:2d}  {np.median(a):6.3f}  {np.median(b):6.3f}  {d0:+7.4f}  [{lo:+.4f},{hi:+.4f}]  {"YES" if better else ""}')
print('\nsummary (#leads where A is significantly better / worse):')
for m, sgn in (('kge', 1), ('nse', 1), ('rmse', -1)):
    w = allw[m]; better = sum(x[3] for x in w); worse = sum((x[2] * sgn < 0) if sgn > 0 else (x[1] > 0) for x in w)
    print(f'  {m}: better {better}/10, worse {worse}/10')
