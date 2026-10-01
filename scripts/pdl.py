"""Parallel, resumable range downloader (Zenodo is ~0.15 MB/s per connection, scales with connections).
usage: pdl.py URL OUT SIZE [streams] [part_MB]"""
import sys, os, time, threading, urllib.request, json
from concurrent.futures import ThreadPoolExecutor
url, out, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 32
PART = (int(sys.argv[5]) if len(sys.argv) > 5 else 32) * 1024 * 1024
donef = out + '.done'
done = set(json.load(open(donef))) if os.path.exists(donef) else set()
fd = os.open(out, os.O_RDWR | os.O_CREAT)
os.ftruncate(fd, size)
parts = [(i, i * PART, min(size, (i + 1) * PART) - 1) for i in range((size + PART - 1) // PART)]
lock = threading.Lock(); got = [sum(min(size,(i+1)*PART)-i*PART for i in done)]
def fetch(p):
    i, s, e = p
    if i in done: return
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={s}-{e}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                pos = s
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    os.pwrite(fd, b, pos); pos += len(b)
                    with lock: got[0] += len(b)
            if pos != e + 1: raise IOError(f'short read {pos} != {e+1}')
            with lock:
                done.add(i); json.dump(sorted(done), open(donef, 'w'))
            return
        except Exception as ex:
            time.sleep(5 * (attempt + 1))
    print('PART FAILED', i, flush=True)
def report():
    t0 = time.time(); g0 = got[0]
    while got[0] < size:
        time.sleep(30)
        print(f'{time.strftime("%H:%M:%S")} {got[0]/1e9:6.2f}/{size/1e9:.2f} GB  {(got[0]-g0)/(time.time()-t0)/1e6:5.1f} MB/s avg  parts {len(done)}/{len(parts)}', flush=True)
threading.Thread(target=report, daemon=True).start()
with ThreadPoolExecutor(N) as ex: list(ex.map(fetch, parts))
ok = len(done) == len(parts)
print('DONE' if ok else 'INCOMPLETE', out, time.strftime('%c'), flush=True)
