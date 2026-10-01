"""Seed-consistency summary of results/event_members_report.json: in how many (lead, RP) cells, and in how many seeds, is the member-based forecast of the final model better (higher CSI) than
(a) its own ensemble-mean forecast, (b) the deterministic baseline, (c) the BEST of all baseline systems (deterministic, MC dropout, dressed)."""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np
J = json.load(open(ROOT + '/results/event_members_report.json')); LEADS = (1, 3, 5, 7, 10); RPS = (1.5, 2, 5, 10, 20, 50); BASE = ('b2d_det|det', 'b2d_mc|tuned', 'b2d_mc_infl|tuned', 'b2d_dress_sqrt|tuned', 'b2d_dress_add|tuned')
def per(kind, lead, rp, sysmode): return np.array(J[f'{kind}|{lead}|{rp}|{sysmode.split("|")[0]}|{sysmode.split("|")[1]}']['per_seed'])
ns = len(per('single', 1, 1.5, 'ours_mean|det'))
for kind in ('single', 'dual'):
    print(f'\n=== {kind} threshold; per (lead, RP) cell: #seeds (of {ns}) in which ours member-based (p* tuned on 2020) has higher CSI than ... ===')
    OURS = [o for o in ('ours_raw|tuned', 'ours_mc|tuned', 'ours_mc_infl|tuned') if f'{kind}|1|1.5|{o.split("|")[0]}|tuned' in J]
    for ours in OURS:
        print(f'  system {ours}:')
        print(f'    {"RP":>5s}  ' + '  '.join(f'lead{l:>2d}: own-mean / det-base / best-base' for l in LEADS))
        for rp in RPS:
            cells = []
            for l in LEADS:
                o = per(kind, l, rp, ours); m = per(kind, l, rp, 'ours_mean|det'); d = per(kind, l, rp, 'b2d_det|det'); b = np.max([per(kind, l, rp, s) for s in BASE], 0)
                cells.append(f'{(o > m).sum()} / {(o > d).sum()} / {(o > b).sum()}'.center(33))
            print(f'    {rp:>5g}  ' + '  '.join(cells))
    tot = {}
    for ours in OURS:
      tot = {}
      if True:
        for grp, rps in (('common 1.5-2 yr', (1.5, 2)), ('5-10 yr', (5, 10)), ('20-50 yr', (20, 50))):
            for ref, refname in (('own', 'ours_mean|det'), ('det', 'b2d_det|det'), ('best', None)):
                w = n = 0
                for l in LEADS:
                    for rp in rps:
                        o = per(kind, l, rp, ours); r = per(kind, l, rp, refname) if refname else np.max([per(kind, l, rp, s) for s in BASE], 0); w += (o > r).sum(); n += len(o)
                tot[(grp, ref)] = f'{w}/{n}'
        print(f'  fraction of (lead, RP, seed) cells won by {ours}:')
        for grp in ('common 1.5-2 yr', '5-10 yr', '20-50 yr'): print(f'    {grp:16s} vs own ensemble-mean {tot[(grp, "own")]:>6s} | vs deterministic baseline {tot[(grp, "det")]:>6s} | vs best baseline system {tot[(grp, "best")]:>6s}')
