"""AIFL-style embedding LSTM (Taccari et al. 2026): static/dynamic tanh-MLP embeddings -> 1-layer LSTM -> last-n daily outputs."""
import torch, torch.nn as nn

class EmbMLP(nn.Module):
    def __init__(s, n_in, sizes=(30, 20, 64)):
        super().__init__()
        layers, d = [], n_in
        for h in sizes: layers += [nn.Linear(d, h), nn.Tanh()]; d = h
        s.net = nn.Sequential(*layers); s.n_out = d
    def forward(s, x): return s.net(x)

class RiverLSTM(nn.Module):
    def __init__(s, n_dyn, n_stat, hidden=256, n_out=10, dropout=0.4, forget_bias=3.0, noise_dim=0, noise_last=0):
        super().__init__()
        s.dyn, s.stat = EmbMLP(n_dyn), EmbMLP(n_stat)
        s.noise_dim, s.noise_last = noise_dim, noise_last     # noise_last>0: latent noise only in the last `noise_last` steps, earlier steps are shared by all members
        s.lstm = nn.LSTM(s.dyn.n_out + s.stat.n_out + noise_dim, hidden, batch_first=True)   # noise columns LAST (deterministic checkpoints load zero-padded)
        s.drop, s.head, s.n_out = nn.Dropout(dropout), nn.Linear(hidden, 1), n_out
        with torch.no_grad():   # NeuralHydrology-style forget-gate bias init
            s.lstm.bias_hh_l0[hidden:2 * hidden].fill_(forget_bias)
    def forward(s, x_dyn, x_stat, K=1):          # (B,T,F), (B,S) -> (B,n_out)  or, if noise_dim>0, (B,K,n_out)
        B, T = x_dyn.shape[:2]
        e = torch.cat([s.dyn(x_dyn), s.stat(x_stat)[:, None].expand(-1, T, -1)], -1)
        if s.noise_dim == 0:
            h, _ = s.lstm(e)
            return s.head(s.drop(h[:, -s.n_out:])).squeeze(-1)
        if s.noise_last and s.noise_last < T:
            sp = T - s.noise_last                                                     # shared prefix (zero noise), then K noisy continuations
            _, (hn, cn) = s.lstm(torch.cat([e[:, :sp], e.new_zeros(B, sp, s.noise_dim)], -1))
            e2 = e[:, sp:].repeat_interleave(K, 0); eps = torch.randn(B * K, 1, s.noise_dim, device=e.device, dtype=e.dtype).expand(-1, s.noise_last, -1)
            h, _ = s.lstm(torch.cat([e2, eps], -1), (hn.repeat_interleave(K, 1).contiguous(), cn.repeat_interleave(K, 1).contiguous()))
            return s.head(s.drop(h[:, -s.n_out:])).squeeze(-1).view(B, K, s.n_out)
        e = e.repeat_interleave(K, 0); eps = torch.randn(B * K, 1, s.noise_dim, device=e.device, dtype=e.dtype).expand(-1, T, -1)
        h, _ = s.lstm(torch.cat([e, eps], -1))
        return s.head(s.drop(h[:, -s.n_out:])).squeeze(-1).view(B, K, s.n_out)

def norm_mse(pred, y, sigma_b, eps=0.1):
    """NH 'NSE loss': squared error weighted by 1/(basin target std + eps)^2 ; NaN targets are masked."""
    m = torch.isfinite(y)
    w = 1.0 / (sigma_b[:, None] + eps) ** 2
    return (((pred - torch.nan_to_num(y)) ** 2) * w * m).sum() / m.sum().clamp(min=1)


class TwoClockLSTM(nn.Module):
    """Slow clock + fast clock (the 'two clocks' structure of our S2S work, transplanted to rivers).

    slow clock : weekly-mean forcing over `weeks` weeks BEFORE the fast window -> LSTM_s (7x slower step).
                 Its final state z is the catchment's slow memory (snow / soil / groundwater).
    fast clock : daily inputs over `fast_len` days (past + the N forecast days, with is_forecast / lead channels)
                 -> LSTM_f initialised from z and conditioned on z at every step; emits the N daily discharges.
    output     : q = slow baseline(z) + fast anomaly(h)     (additive split, so the slow head can only carry low-frequency mass)
    Input x: (B, 7*weeks + fast_len, C) daily channels; the first 7*weeks days are aggregated to weeks inside the model.
    """
    def __init__(s, n_dyn, n_stat, hidden_f=256, hidden_s=128, weeks=17, fast_len=60, n_out=10, dropout=0.4, forget_bias=3.0, noise_dim=0):
        super().__init__()
        s.weeks, s.fast_len, s.n_out, s.noise_dim = weeks, fast_len, n_out, noise_dim
        s.stat = EmbMLP(n_stat); s.dyn_s = EmbMLP(n_dyn); s.dyn_f = EmbMLP(n_dyn)
        s.lstm_s = nn.LSTM(s.dyn_s.n_out + s.stat.n_out, hidden_s, batch_first=True)
        s.h0, s.c0 = nn.Linear(hidden_s, hidden_f), nn.Linear(hidden_s, hidden_f)
        s.lstm_f = nn.LSTM(s.dyn_f.n_out + s.stat.n_out + hidden_s + noise_dim, hidden_f, batch_first=True)   # noise columns LAST (so a deterministic checkpoint loads zero-padded)
        s.drop, s.head_f, s.head_s = nn.Dropout(dropout), nn.Linear(hidden_f, 1), nn.Linear(hidden_s, 1)
        with torch.no_grad():
            for l, h in ((s.lstm_s, hidden_s), (s.lstm_f, hidden_f)): l.bias_hh_l0[h:2 * h].fill_(forget_bias)
    def forward(s, x, st, return_parts=False, K=1):
        """deterministic (noise_dim=0): (B,N).  stochastic (noise_dim>0): K latent-noise members -> (B,K,N)."""
        B, T, C = x.shape; W = s.weeks
        assert T == 7 * W + s.fast_len, f'window {T} != 7*{W}+{s.fast_len}'
        xs = x[:, :7 * W].reshape(B, W, 7, C).mean(2); xf = x[:, 7 * W:]
        e = s.stat(st)
        hs, _ = s.lstm_s(torch.cat([s.dyn_s(xs), e[:, None].expand(-1, W, -1)], -1))
        z = hs[:, -1]
        h0, c0 = torch.tanh(s.h0(z)), s.c0(z)
        Tf = xf.shape[1]
        inp = torch.cat([s.dyn_f(xf), e[:, None].expand(-1, Tf, -1), z[:, None].expand(-1, Tf, -1)], -1)
        q_slow = s.head_s(s.drop(z))                       # (B,1) broadcast over the N days
        if s.noise_dim == 0:
            hf, _ = s.lstm_f(inp, (h0[None].contiguous(), c0[None].contiguous()))
            q_fast = s.head_f(s.drop(hf[:, -s.n_out:])).squeeze(-1)
            q = q_slow + q_fast
            return (q, q_slow.expand_as(q), q_fast) if return_parts else q
        # stochastic fast clock: one latent 'scenario' vector per member, held over the whole fast window
        inp = inp.repeat_interleave(K, 0); eps = torch.randn(B * K, 1, s.noise_dim, device=x.device, dtype=x.dtype).expand(-1, Tf, -1)
        hf, _ = s.lstm_f(torch.cat([inp, eps], -1), (h0.repeat_interleave(K, 0)[None].contiguous(), c0.repeat_interleave(K, 0)[None].contiguous()))
        q_fast = s.head_f(s.drop(hf[:, -s.n_out:])).squeeze(-1).view(B, K, s.n_out)
        return q_slow[:, None, :] + q_fast

def norm_mse_lead(pred, y, sigma_b, eps=0.1):
    """per-output (lead) normalised squared error, shape (N,) ; NaN targets masked."""
    m = torch.isfinite(y); w = 1.0 / (sigma_b[:, None] + eps) ** 2
    e = ((pred - torch.nan_to_num(y)) ** 2) * w * m
    return e.sum(0) / m.sum(0).clamp(min=1)


def crps_fair_norm(pred, y, sigma_b, eps=0.1):
    """fair (unbiased) ensemble CRPS, weighted 1/(basin std+eps) like the NSE-type loss. pred (B,K,N), y (B,N). NaN targets masked."""
    B, K, N = pred.shape; m = torch.isfinite(y); yy = torch.nan_to_num(y)
    t1 = (pred - yy[:, None]).abs().mean(1)                                         # (B,N)
    t2 = (pred[:, :, None] - pred[:, None, :]).abs().sum((1, 2)) / (2 * K * (K - 1))  # (B,N)
    w = 1.0 / (sigma_b[:, None] + eps)
    return ((t1 - t2) * w * m).sum() / m.sum().clamp(min=1)
