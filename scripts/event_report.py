"""Report from results/event_members_counts.pkl: CSI / POD / precision / ROC-AUC / Brier skill of member-based event forecasts vs deterministic ones (test), p* tuned on validation 2020.
usage: event_report.py [seeds]"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, pickle, json, numpy as np
sys.path.insert(0, ROOT + '/scripts'); import event_metrics as EM, probmetrics as PM
R = PM.R; RPS = EM.RPS; LEADS = PM.LEADS; import os
res = pickle.load(open(f'{R}/results/event_members_counts.pkl', 'rb'))
if os.path.exists(f'{R}/results/event_members_counts_mc.pkl'): res.update(pickle.load(open(f'{R}/results/event_members_counts_mc.pkl', 'rb')))
HAVE_MC = any(k[2] == 'ours_mc' for k in res)
seeds = sorted({k[0] for k in res if k[2] == 'ours_raw'}) if len(sys.argv) < 2 else [int(x) for x in sys.argv[1].split(',')]
DET = ('ours_mean', 'b2d_det'); PROB = ('ours_raw', 'ours_infl', 'b2d_mc', 'b2d_mc_infl', 'b2d_dress_sqrt', 'b2d_dress_add') + (('ours_mc', 'ours_mc_infl') if HAVE_MC else ())
def cnt(sd, split, name, lead, ri, kind): return res[(sd, split, name, lead, ri, kind)]
def hmf(n1, n0, lv): return n1[lv:].sum(), n1[:lv].sum(), n0[lv:].sum()
def csi(h, m, f): return h / (h + m + f) if h + m + f else np.nan
GROUPS = {'common (1.5,2 yr)': (0, 1), 'rare (5-50 yr)': (2, 3, 4, 5)}
def tuned_level(sd, name, lead, kind, ri):
    g = [grp for grp in GROUPS.values() if ri in grp][0]; K = len(cnt(sd, 'val', name, lead, 0, kind)[0]) - 1; best, bl = -1, 1
    for lv in range(1, K + 1):
        v = np.nanmean([csi(*hmf(*cnt(sd, 'val', name, lead, r, kind), lv)) for r in g])
        if v > best: best, bl = v, lv
    return bl
def point(name, lead, ri, kind, mode):
    """pooled over seeds; returns (csi, pod, prec, per-seed csi list)"""
    H = M = F = 0; per = []
    for sd in seeds:
        n1, n0 = cnt(sd, 'test', name, lead, ri, kind); K = len(n1) - 1
        lv = 1 if K == 1 else (tuned_level(sd, name, lead, kind, ri) if mode == 'tuned' else (K + 1) // 2 if mode == 'half' else 1)
        h, m, f = hmf(n1, n0, lv); H += h; M += m; F += f; per.append(csi(h, m, f))
    return csi(H, M, F), (H / (H + M) if H + M else np.nan), (H / (H + F) if H + F else np.nan), per
def auc_bss(name, lead, ri, kind):
    n1 = sum(cnt(sd, 'test', name, lead, ri, kind)[0] for sd in seeds); n0 = sum(cnt(sd, 'test', name, lead, ri, kind)[1] for sd in seeds); K = len(n1) - 1
    c0 = np.concatenate([[0], np.cumsum(n0)])[:-1]; auc = float((n1 * (c0 + 0.5 * n0)).sum() / (n1.sum() * n0.sum())); p = np.arange(K + 1) / K
    bs = float((n1 * (1 - p) ** 2 + n0 * p ** 2).sum() / (n1.sum() + n0.sum())); base = n1.sum() / (n1.sum() + n0.sum()); return auc, 1 - bs / (base * (1 - base))
out = {}
for kind in ('single', 'dual'):
    print(f'\n################ {kind.upper()} threshold, test, seeds {seeds} pooled ################')
    for lead in LEADS:
        print(f'\n--- lead {lead}: CSI (POD/precision) ---')
        COLS = [('ours_mean', 'det'), ('ours_raw', 'tuned')] + ([('ours_mc', 'tuned'), ('ours_mc_infl', 'tuned')] if HAVE_MC else []) + [('b2d_det', 'det'), ('b2d_mc', 'tuned'), ('b2d_mc_infl', 'tuned'), ('b2d_dress_sqrt', 'tuned')]
        print(f'{"RP":>5s}  ' + '  '.join(f'{n + ("*" if m == "tuned" else ""):>16s}' for n, m in COLS))
        for ri, rp in enumerate(RPS):
            cells = []
            for name, mode in COLS + [('ours_infl', 'tuned'), ('b2d_dress_add', 'tuned')]:
                c, po, pr, per = point(name, lead, ri, kind, mode); cells.append(f'{c:.3f} ({po:.2f}/{pr:.2f})') if (name, mode) in COLS else None; out[f'{kind}|{lead}|{rp}|{name}|{mode}'] = dict(csi=c, pod=po, prec=pr, per_seed=per)
            print(f'{rp:>5g}  ' + '  '.join(f'{x:>16s}' for x in cells))
    print(f'\n--- threshold-free skill of the probabilistic systems (ROC-AUC | Brier skill score vs base rate), {kind} ---')
    for lead in (1, 5, 10):
        for ri, rp in enumerate(RPS):
            if rp not in (2, 5, 10): continue
            print(f'  lead {lead:2d} RP{rp:>3g}: ' + '  '.join(f'{n}: {auc_bss(n, lead, ri, kind)[0]:.3f}|{auc_bss(n, lead, ri, kind)[1]:+.3f}' for n in ('ours_raw',) + (('ours_mc', 'ours_mc_infl') if HAVE_MC else ()) + ('b2d_mc', 'b2d_mc_infl', 'b2d_dress_sqrt')))
json.dump(out, open(f'{R}/results/event_members_report.json', 'w'), indent=1)
