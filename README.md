# Bharatiya-Antariksh-Hackathon-2026-ISRO
Forecasting and/or Nowcasting of Solar Flares using combined Soft and Hard X-ray data from Aditya-L1-Problem Statement 15**
 
Solar flare nowcasting (real-time detection + physical characterization) and
forecasting (30-minute-ahead probability) built on **real ISRO Aditya-L1
data** — SoLEXS (2.8–20 keV) and HEL1OS (20–150 keV) FITS files from PRADAN,
calibrated with real ISRO CALDB ARF/RMF response files.
 
No synthetic or GOES-substitute data anywhere in this pipeline. Every result
below is measured on real instrument data from 12 real dates.
 
## Why this matters
 
Most comparable submissions validate on GOES XRS or synthetic light curves.
This system ingests actual SoLEXS `.pi` spectra and HEL1OS band FITS,
forward-folds real ARF effective-area curves, and reports honest skill
metrics — including where the evidence runs out — rather than inflating
confidence past what the data supports.
 
## Two pipelines
 
| | `nowcast_physics.py` | `forecast_pipline2.py` |
|---|---|---|
| Question | "What is happening right now?" | "Will a significant flare start in the next 30 minutes?" |
| Method | Poisson-FOCuS event detection, forward-folded Cash-statistic spectral fit, Neupert lag scan, tri-band independent voting | RandomForest / XGBoost / MLP ensemble, walk-forward calibration, physics-informed features |
| Output | Per-flare temperature, gamma, QPP period, confidence score, PNG report | Calibrated probability, threshold sweep, walk-forward TSS |
 
## Results (honest, on real held-out test dates)
 
- Walk-forward TSS: **+0.358** vs persistence baseline **+0.023**, beating
  persistence in **4/4** unseen-day folds
- Single-split TSS: **+0.210** vs persistence **+0.016**
- Reliability-checked: this system reports a confidence ceiling (~8%) and
  states explicitly when that ceiling is an evidence limit, not a modeling
  bug — see `reliability_report()` in `forecast_pipline2.py`
Full baseline comparison (climatology, persistence, 5 model candidates) is
printed by every training run and written to `test_predictions.csv`.
 
## Quick start
 
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
 
python nowcast_physics.py       # real-time flare detection + report PNGs
python forecast_pipline2.py     # [1] train, or [2] forecast a real day
```
 
Forecast mode lets you replay any real date as a live hindcast, probe any
UTC moment (data hard-truncated at that instant — no future leakage), and
adjust alarm sensitivity interactively without retraining.
 
## Data layout expected
 
```
Dataset SOLEXS/     real PRADAN SoLEXS .pi / .lc files, one folder per date
dataset HEL1OS/     real PRADAN HEL1OS FITS files
Tools/              ISRO CALDB ARF/RMF calibration files
physics_output/     generated reports, catalogs, trained model (gitignored)
```
 
Edit the folder constants at the top of `nowcast_physics.py` to match your
local paths.
 
## Repo structure
 
```
nowcast_physics.py         real-time flare detection & characterization
forecast_pipline2.py       30-min-ahead ML forecaster
competitor_benchmark.py    optional: benchmark against other public pipelines
                            on this project's own real data (private use —
                            see file header for constraints)
docs/
  Nowcast_Case_Study.docx  plain-language walkthrough, function-referenced
  Forecast_Case_Study.docx plain-language walkthrough, function-referenced
requirements.txt
```
 
## Known limitations (stated honestly, not hidden)
 
- 12 real dates, ~47 significant-flare positive examples in the test set —
  small-sample noise affects both this system and any comparison against it
- Forecaster catches roughly a third of significant flares at the balanced
  (F1) operating point; the sensitivity dial trades this against false-alarm
  rate — see the printed threshold sweep
- "Significant" is defined as ≥8× local background, not GOES class; a flare
  on an already-elevated background can be large in absolute class but a
  small relative jump, and will not trigger the alarm — by design, stated
  in the forecast output itself
- Per-GOES-class probability is not offered — insufficient class-labeled
  positives to calibrate honestly with the current dataset size

 
