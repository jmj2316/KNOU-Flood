# KNOU-Flood

Lead-aware, probabilistic LSTM streamflow forecasts for the reanalysis-to-forecast domain shift
(code, trained weights and per-station results for the paper *"KNOU-Flood: Advancing global streamflow forecasting with lead-aware probabilistic machine learning"*, Minjong Cheon, Korea National Open University; manuscript in preparation).

KNOU-Flood extends the AIFL recipe (ERA5-Land pre-training, forecast fine-tuning, single-layer LSTM) with a 424-day input window, lead-aware forecast inputs from ECMWF HRES and GraphCast, and a probabilistic output (latent-noise ensemble trained with the fair CRPS, dropout kept active at inference). The AIFL code is not public, so the repository also contains our re-implementation of its recipe (the *AIFL-recipe baseline*) trained under identical data, stations, splits and metrics.

## Layout
```
src/        model (model.py), GPU-resident data store (data.py), training and evaluation (train.py), metrics (metrics.py)
scripts/    data download/preparation, paired bootstrap, probabilistic and flood-event evaluation, figure scripts
weights/    pre-trained (*_pre.pt) and fine-tuned (*_ft.pt) checkpoints of every model in the paper (about 1.6 MB each; the 1024-unit baseline 19 MB)
results/    per-station metrics (*_perbasin.npz) and summaries (*.json) of the models in the paper; prob_eval.json, event_members_report.json
notes/      pre-registered evaluation protocol and model summary table
```
Large intermediate files (forecast dumps, arrays) are not included.

## Environment
Python 3.10, PyTorch 2.5.1 (CUDA 12.1), numpy 1.26, pandas 2.3, xarray 2025.6, scipy 1.14, matplotlib 3.10, cartopy 0.25, shapely 2.1 (see `requirements.txt`). One 46 GB GPU is sufficient; the whole data set is kept on the GPU. Set `KNOU_FLOOD_ROOT` to the repository root if the code is run from elsewhere.

## Data
Public data: Caravan v1.6 and the GRDC-Caravan extension (Zenodo) and Caravan-MultiMet (ERA5-Land, ECMWF HRES and GraphCast forecasts; Google Cloud Storage `gs://caravan-multimet`).
```
bash scripts/dl_streamflow.sh && bash scripts/extract_core.sh      # Caravan + GRDC-Caravan (about 32 GB)
python scripts/dl_forcing.py                                       # MultiMet zarr -> data/multimet
python scripts/build_targets.py && python scripts/build_arrays.py  # data/arrays/*.npy, targets.npz
```

## Training (commands used in the paper)
Pre-training (ERA5-Land, deterministic; seed 1 shown) and fine-tuning of KNOU-Flood:
```
python src/train.py --name s424_s1 --arch single --L 424 --hidden 256 --pre_only --pre_epochs 60 --pre_upd 3000 --seed 1 --track hres --bs 512 --dev cuda
python src/train.py --name s424fsm_s1 --arch single --L 424 --hidden 256 --noise_dim 8 --noise_last 60 --K 8 --K_eval 16 --accum 2 --seed 1 \
       --fusion --ft mix --pre_ckpt ckpt/s424_s1_pre.pt --ft_epochs 30 --ft_upd 1500 --split test --gc_window --track hres --bs 512 --dev cuda
```
Variants: `--tw_lambda 1.0 --tw_q 0.9` (tail-weighted CRPS), no `--fusion` (HRES only), `--fusion --fusion_mode nodiff` (no GraphCast-minus-HRES channels), `--holdout_frac 0.2` (spatial block hold-out).
AIFL-recipe baseline (180-day window, deterministic, lead-1 forecasts in all steps; both protocols are evaluated):
```
python src/train.py --name b2d --track hres --ft lt --hidden 256 --seed 0 --pre_epochs 60 --pre_upd 3000 --ft_epochs 30 --ft_upd 1500 \
       --eval_traj --split test --gc_window --bs 512 --dev cuda        # --hidden 1024 for the larger control
```
Ensemble members and dropout-based ensembles for the probabilistic and event evaluation: add `--dump_all --dump_members 1,3,5,7,10 --mc_dropout --traj_only` to an evaluation run with `--ft_ckpt weights/<name>_ft.pt`.

## Evaluation
```
python scripts/paired.py results/A_perbasin.npz results/B_perbasin.npz traj 2000     # paired bootstrap over stations
python scripts/compare_seeds.py s424fsm_s0,s424fsm_s1,s424fsm_s2 b2d_gct,b2d_s1,b2d_s2
python scripts/prob_eval.py 0,1,2 ; python scripts/event_members.py 0,1,2 ; python scripts/event_report.py ; python scripts/tw_eval.py 0,1,2
```
Protocols: *AIFL protocol* (lead-k forecast in every input step) = `lt<k>_*` keys; *operational protocol* (lead-1 forecasts in the past, lead-j forecast for forecast day t+j) = `traj<j>_*` keys in `results/*_perbasin.npz`.

## Main results (test 2021-2023, about 2,100 stations, three seeds; median KGE', operational protocol)
| | LT1 | LT5 | LT10 |
|---|---|---|---|
| KNOU-Flood | 0.655 | 0.550 | 0.481 |
| AIFL-recipe baseline | 0.618 | 0.459 | 0.340 |

Model names in `weights/` and `results/`: `s424fsm_s{0,1,2}` = KNOU-Flood (seed 0 is pre-trained as `m1d_L424v`), `tw_s*` = tail-weighted variant, `b2d*` = AIFL-recipe baseline (`b2d_gct`, `b2d_s1`, `b2d_s2` are the three seeds scored on the test window; `b2d_ft/b2d_s1_ft/b2d_s2_ft` their weights), `abl_nodiff_s*`, `abl_nofus_s*` = ablations, `ho_*` = spatial hold-out.

## Notes
* The scripts were written for the paper's cluster; paths are relative to `KNOU_FLOOD_ROOT`, but the job queue and monitoring helpers are not included.
* Absolute scores are not comparable with the published AIFL numbers (different station list, period and training budget); see the paper.
* License: to be specified before the public release.
