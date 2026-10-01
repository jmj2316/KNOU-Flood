"""3-seed comparison: per-seed medians + paired bootstrap on seed-averaged per-basin metrics (test 2021–2023-12)."""
import numpy as np, sys
# usage: seeds3.py [A_prefix A_seed0_name]  e.g.  seeds3.py s424fsm s424fsm_s0   (default: tc424fsm vs b2d)
if len(sys.argv) >= 3: A = [sys.argv[2], f'{sys.argv[1]}_s1', f'{sys.argv[1]}_s2']; TAG = sys.argv[1]
else: A = ['tc424fsm_gct', 'tc424fsm_s1', 'tc424fsm_s2']; TAG = 'tc424fsm'
B = ['b2d_gct', 'b2d_s1', 'b2d_s2']
LA = [np.load(f'results/{n}_perbasin.npz') for n in A]; LB = [np.load(f'results/{n}_perbasin.npz') for n in B]
for z in LA + LB: assert (z['basins'] == LA[0]['basins']).all()
print('basins', len(LA[0]['basins']))
rng = np.random.default_rng(0)
for proto in ('traj', 'lt'):
    print(f'\n=== {proto} protocol: per-seed medians (KGE′ | NSE | RMSE mm/d) for leads 1 / 5 / 10 ===')
    for name, L in ((f'ours {TAG}', LA), ('AIFL-recipe b2d', LB)):
        for s, z in enumerate(L):
            row = []
            for m in ('kge', 'nse', 'rmse'):
                row.append(' '.join(f"{np.nanmedian(z[f'{proto}{j}_{m}']):6.3f}" for j in (1, 5, 10)))
            print(f'  {name:16s} seed{s}: ' + ' | '.join(row))
    print(f'  --- paired bootstrap on seed-averaged per-basin metrics (A=ours, B=AIFL-recipe) ---')
    tot = {}
    for m, sg in (('kge', 1), ('nse', 1), ('rmse', -1)):
        wins = 0; worse = 0; lines = []
        for j in range(1, 11):
            a = np.nanmean([z[f'{proto}{j}_{m}'] for z in LA], 0); b = np.nanmean([z[f'{proto}{j}_{m}'] for z in LB], 0)
            ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
            ii = rng.integers(0, len(a), (2000, len(a))); d = np.median(a[ii], 1) - np.median(b[ii], 1); lo, hi = np.percentile(d, [2.5, 97.5])
            good = (lo * sg > 0) if sg > 0 else (hi < 0); bad = (hi * sg < 0) if sg > 0 else (lo > 0)
            wins += good; worse += bad
            if j in (1, 5, 10): lines.append(f'lead{j}: {np.median(a):.3f} vs {np.median(b):.3f} (Δ {np.median(a)-np.median(b):+.3f} [{lo:+.3f},{hi:+.3f}])')
        print(f'  {m.upper():4s} significantly better {wins}/10, worse {worse}/10 | ' + ' ; '.join(lines))
    # per-seed sign consistency at the basin level
    print('  per-seed lead-1/10 KGE′ gap (ours − b2d, median over basins): ' + ', '.join(f"seed{s}: {np.nanmedian(LA[s][f'{proto}1_kge']-LB[s][f'{proto}1_kge']):+.3f}/{np.nanmedian(LA[s][f'{proto}10_kge']-LB[s][f'{proto}10_kge']):+.3f}" for s in range(3)))
