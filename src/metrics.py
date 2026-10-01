import numpy as np
def kge_prime(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim); o, s = obs[m], sim[m]
    if len(o) < 30 or o.std() == 0 or o.mean() == 0: return np.nan
    r = np.corrcoef(o, s)[0, 1]; beta = s.mean() / o.mean(); gamma = (s.std() / s.mean()) / (o.std() / o.mean()) if s.mean() != 0 else np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
def nse(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim); o, s = obs[m], sim[m]
    if len(o) < 30 or o.var() == 0: return np.nan
    return 1 - ((o - s) ** 2).sum() / ((o - o.mean()) ** 2).sum()
def rmse(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim); return float(np.sqrt(np.mean((obs[m] - sim[m]) ** 2))) if m.any() else np.nan
