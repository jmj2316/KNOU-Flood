# KNOU-Flood — pre-registered win criteria and rules (written 2026-09-30, after seeing single-seed test numbers for v0/d/M1d/TC180)

Honesty rules
* Every design decision (architecture, loss, hyper-parameters, checkpoint) is made on the VALIDATION year 2020 (`--split val`).
  The test years 2021-2024 are scored once per final configuration (and all test numbers are reported, wins or not).
* No post-hoc bias / damping correction. Every table carries physical-unit RMSE and climatology RMSE, not only KGE'/NSE.
* Comparisons are paired over the same basins; claims need a 95% paired bootstrap CI (scripts/paired.py) and >=3 seeds.
* The AIFL-protocol numbers (LT-k forcing in every step) and the operational trajectory numbers are always reported separately;
  trajectory numbers are never compared with AIFL's published numbers.

Win criteria
* W1 (structure): the two-clock model beats the single-clock model with the same window, hidden size, features, budget and
  fine-tuning, on the trajectory protocol: median paired improvement with CI excluding 0 for KGE' AND RMSE at >= 7 of 10 leads
  (seed-averaged, 3 seeds), and not worse at any lead.
* W2 (recipe): our best model is not worse than the AIFL-recipe baseline (B2, LT1 fine-tune) at lead 1-3 and better at lead 4-10
  (i.e. Pareto-dominates B2 and M1) on the trajectory protocol.
* W3 (published SOTA, AIFL protocol): with hidden 1024 / AIFL budget, LT1 median KGE' >= 0.663 and NSE >= 0.53, LT10 KGE' >= 0.215
  (AIFL's published numbers; different basin list -> reported as indicative only).
