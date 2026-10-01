# Model architecture and hyperparameters (AIFL Table 2 layout)

AIFL column = arXiv 2602.16579v2 (Table 2 + text). "Baseline" = our re-implementation of the AIFL recipe (`b2d`). "Final model" = `s424fsm` (updated 2026-10-01; earlier versions of this table described the two-clock variant `tc424fsm`, see the note at the end).

| Component | AIFL (paper) | AIFL-style baseline (ours) | Final model (ours) |
|---|---|---|---|
| Recurrent core | single-layer LSTM | single-layer LSTM | single-layer LSTM with a stochastic latent input |
| Hidden state size | 1024 | 256 | 256 |
| Dropout (output) | 0.4 | 0.4 | 0.4 |
| Static embedding | 3-layer MLP (30,20,64), tanh | same | same |
| Dynamic embedding | 3-layer MLP (30,20,64), tanh | same | same |
| Temporal window | 180 days | 180 days | 424 days = 414 past + 10 forecast days |
| Output sequence length | 10 days | 10 days | 10 days (hidden states of the last 10 steps) |
| Stochastic component | none (deterministic) | none | latent noise eps ~ N(0, I_8), one draw per member, appended to the LSTM input in the last 60 steps only; the first 364 steps are run once and shared by all members; 8 members in training, 16 at inference; ensemble mean = point forecast |
| LSTM input per step | n/r | 128 (64 dynamic + 64 static) | 136 (64 dynamic + 64 static + 8 noise) |
| Dynamic inputs | 5 variables (SSR, STR, SP, T2M, TP) | same 5 + sin/cos day-of-year | same 5 + sin/cos day-of-year + 2 trajectory channels (is-forecast, lead/10) + 5 GraphCast channels (TP, T2M, GC-HRES differences of both, availability) = 14 |
| Static inputs | 203 (HydroATLAS) + seasonal + UTC offset | 212 (HydroATLAS + Caravan climate indices + log area + UTC offset) | 212 |
| Forecast source | IFS control, lead 1–10 | HRES, lead 1–10 | HRES + GraphCast, lead 1–10 |
| Parameters | ~4.7 M (our count of the same architecture) | 0.41 M (406 k) | 0.41 M (414 k) |
| Pre-training data | ERA5-Land 1980–2019, 18,588 gauges | ERA5-Land 1980–2019, 16,565 gauges | same as baseline |
| Fine-tuning data | IFS LT1, 2016–2019, 2,010 gauges | HRES LT1, 2016–2019, 2,113 gauges | HRES + GraphCast, 2016–2019, 2,113 gauges |
| Validation / test | 2020 / 2021–2024, 2,003 gauges | 2020 / 2021–2024-09 (2,113 gauges) | 2020 / 2021–2023-12-21 (GraphCast coverage; 2,099 scored gauges) |
| Optimizer | Adam | Adam | Adam |
| Learning rate (pre / fine-tune) | 4e-4 / 1e-4 | 5e-4 / 1.25e-4 | 5e-4 / 1.25e-4 |
| Schedule | cosine, 10 / 5 warm-up epochs | cosine, warm-up = 10% of updates | same |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Batch size | 512 | 512 | 512 windows x 8 members (2 gradient-accumulation micro-batches) |
| Epochs x updates (pre) | 100 x <=10,000 | 60 x 3,000 (180 k updates) | 60 x 3,000 (deterministic: noise columns and GraphCast channels zero) |
| Epochs x updates (fine-tune) | 30 x <=10,000 | 30 x 1,500 (45 k) | 30 x 1,500 (45 k) |
| Fine-tuning windows | LT1 series in every step | LT1 series in every step | 50% LT1 windows + 50% trajectory windows (past = LT1, day t+j = lead j) |
| Loss | normalised MSE, eps = 0.1 | squared error / (basin std + 0.1)^2 | fair ensemble CRPS / (basin std + 0.1) in fine-tuning; normalised MSE in pre-training |
| Target noise | Gaussian, sigma = 0.02 | none | none |
| Forget-gate bias init | n/r | 3.0 | 3.0 |
| Seeds | 3 (end-to-end) | 3 (end-to-end) | 3 (end-to-end), all three evaluated on test |
| Hardware | n/r | 1 x A40 (pre-training ~1–2 h alone) | 1 x A40 (pre-training ~4–5 h with 2–3 jobs sharing the GPU) |

n/r = not reported in the text we could read.

## What changed relative to the baseline (and what each part is worth)
Validation-year decomposition (seed 1) and test results are in `results/` and the project memory. Order of importance: stochastic CRPS head (largest) > GraphCast fusion > 424-day window > mixed LT1/trajectory fine-tuning (mainly repairs the lead-1 gap). Latent noise in the last 60 steps only is a compute saving (shared prefix), not a skill gain.

## Note on the discarded two-clock variant (`tc424fsm`)
An earlier version of the model split the window into a weekly slow-clock LSTM (52 weeks, 128 units) and a daily fast-clock LSTM (60 days, 256 units) coupled through the slow final state (0.75 M parameters). A window-matched control showed that its advantage over the baseline came from the longer window. Validation year 2020: in the deterministic control the single 424-day LSTM differs from the two-clock model in KGE' at 0 of 10 leads and is better in NSE at 5 of 10 leads; with the full recipe (stochastic + fusion + mix) the final single-clock model vs `tc424fsm` has 0 of 10 significant leads in KGE', NSE and RMSE in both protocols (CRPS slightly lower for the single clock, lead 1: 0.154 vs 0.164). The final model therefore keeps the simpler single clock and the "two clocks" claim is withdrawn.
