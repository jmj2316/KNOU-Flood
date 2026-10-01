"""Per-seed event-verification comparison (hit/miss/false-alarm counts from event_metrics.run), summarised as CSI = hit/(hit+miss+fa).
usage: event_seeds.py A_dump B_dump [A_dump B_dump ...]   (pairs of dumps: new model, baseline; one pair per seed)"""
import os
ROOT = os.environ.get('KNOU_FLOOD_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, io, contextlib, numpy as np
sys.path.insert(0, ROOT + '/scripts')
import event_metrics as EM
def counts(path):
    with contextlib.redirect_stdout(io.StringIO()): out = EM.run(path)
    return out                                   # out[lead] = (single{rp:(h,m,fa)}, dual{rp:(h,m,fa)})
def csi(c): h, m, f = c; return h / (h + m + f) if h + m + f else np.nan
def pod(c): h, m, f = c; return h / (h + m) if h + m else np.nan
def prec(c): h, m, f = c; return h / (h + f) if h + f else np.nan
args = sys.argv[1:]; pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
for s, (pa, pb) in enumerate(pairs):
    A, B = counts(pa), counts(pb)
    print(f'\n##### pair {s}: A={pa.split("/")[-1]}  B={pb.split("/")[-1]}')
    for kind, k in (('single', 0), ('dual', 1)):
        print(f'  [{kind} threshold]   mean over leads 1-10 :  POD A/B | precision A/B | CSI A/B | #leads CSI(A)>CSI(B)')
        for rp in EM.RPS:
            g = lambda f, O: np.nanmean([f(O[j][k][rp]) for j in range(1, 11)])
            wins = sum(csi(A[j][k][rp]) > csi(B[j][k][rp]) for j in range(1, 11))
            print(f'    RP{rp:>4}:  {g(pod, A):.3f}/{g(pod, B):.3f} | {g(prec, A):.3f}/{g(prec, B):.3f} | {g(csi, A):.3f}/{g(csi, B):.3f} | {wins}/10')
