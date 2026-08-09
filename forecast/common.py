"""
forecast_pipeline.py
=====================
Aditya-L1 solar flare FORECASTING pipeline: predicts whether a flare will
START in the next 30 minutes, using only the last 30 minutes of X-ray data.

HOW THIS RELATES TO nowcast_physics.py
---------------------------------------
This is a SEPARATE topic from nowcasting (real-time detection). But it reuses
nowcast_physics.py's file readers, real CALDB calibration, and flare detector
rather than re-implementing them -- three good reasons:
  1. Consistency: "what counts as a flare" should mean the same thing in both
     halves of the project.
  2. The nowcast detector IS this module's ground-truth labeler ("nowcast
     referee" design from the proposal): its detections train and grade the
     forecaster. The forecaster NEVER sees nowcast's output at prediction
     time -- only raw X-ray history -- so it has to earn its own answer and
     still works standalone once the two systems are separated.
  3. Real W/m^2 from CALDB (built for nowcasting) makes better features than
     the old approximate "counts x guessed constant" this pipeline used
     before CALDB existed.

THE THREE LABEL FIXES (this is the part that was broken before)
------------------------------------------------------------------
Last attempt scored AUC 0.33 -- worse than a coin flip -- because of the
"decay-phase trap": after a flare ends, X-ray brightness stays elevated for
60-90 minutes while it decays. Those windows got labelled "no flare coming"
(correct, technically) but LOOKED like active/bright conditions, while real
quiet pre-flare periods looked dim. The model learned exactly backwards:
bright = safe, dim = danger.

  FIX 1 (decay exclusion):    any feature window whose end-time falls within
                              DECAY_EXCLUDE_MIN minutes after a previous
                              flare's END is dropped entirely from training.
  FIX 2 (background norm):   every flux-based feature is expressed as a
                              multiple of that day's own local background,
                              not a raw count/flux value -- makes a quiet day
                              and a noisy day comparable.
  FIX 3 (significant only):  a window is only labelled 1 ("flare coming") if
                              the upcoming flare peaks at >= SIGNIFICANT_RATIO
                              times local background. Tiny microflares are
                              excluded from the positive class as label noise.

WHAT THIS SCRIPT DOES
-----------------------
  1. Finds every SoLEXS date (reuses nowcast_physics.find_solexs_dates).
  2. For each date: reads SoLEXS (+ HEL1OS if present), runs the SAME flare
     detector as nowcasting to get ground-truth flare start/end/peak times.
  3. Slides a window across the day (every STEP_MIN minutes). Each window's
     FEATURES come from the LOOKBACK_MIN minutes ending at that moment; its
     LABEL is whether a significant flare starts in the next LEAD_MIN minutes.
  4. Extracts the 42 physics-informed features (Section: FEATURE EXTRACTION).
  5. Trains Random Forest + XGBoost (GradientBoosting fallback) on a
     TIME-based split (train on earlier dates, test on later ones the model
     has never seen -- never a random split, that would leak the future into
     training).
  6. Reports AUC, precision, recall, and average lead time (how many minutes
     before the real flare start the model first raised an alarm).
  7. Cross-checks the forecaster against the nowcast referee on the test set
     (correct-alarm / correct-quiet / false-alarm / missed-flare tally).
  8. Saves the best model + scaler + a full predictions CSV.

HOW TO USE
-----------
Just run it: `python forecast_pipeline.py`. Uses the same SOLEXS_FOLDER /
HELOS_FOLDER / TOOLS_FOLDER paths as nowcast_physics.py (imported from it,
so there's only one place to edit paths).
"""

import os, sys, warnings, math, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# --- reuse everything already built & tested in nowcast_physics.py ---------
import nowcast_physics as ncp

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# =============================================================================
#  CONFIG
# =============================================================================
LOOKBACK_MIN        = 30     # feature window: last N minutes of history
LEAD_MIN             = 30     # forecast horizon: predict flare in next N minutes
STEP_MIN             = 5      # slide the window forward this many minutes each time
DECAY_EXCLUDE_MIN   = 90     # FIX 1: CEILING -- never exclude beyond this long
DECAY_EXCLUDE_FLOOR_MIN = 10  # always exclude at least this long (flux can dip
                              # transiently right after a peak without meaning
                              # the decay is really over)
DECAY_FLUX_THRESHOLD = 2.0    # once past the floor, exclusion ends as soon as
                              # flux (x-background) genuinely drops below this
                              # -- not just because the clock hit 90 minutes.
                              # A fixed 90-min blanket exclusion was found to
                              # remove exactly the "recent flare -> another one
                              # soon" training examples that let the model
                              # learn flare clustering (persistence beat the
                              # real model on TSS with this fixed; the theory
                              # is this dynamic version gives those examples
                              # back once flux has genuinely settled).
_TIME_CAP_MIN = 24.0 * 60.0   # cap "time since last flare" at 24h so a long
                              # quiet stretch doesn't dominate tree splits
HISTORY_DECAY_TAU_MIN = 360.0  # Hawkes-style decay tau for decayed_flare_history (6h)
SIGNIFICANT_RATIO    = 8.0    # FIX 3: only flares peaking >= this x background count as label=1
MIN_WINDOW_COVERAGE  = 0.6    # a window needs at least this fraction of expected bins present

TEST_FRACTION = 0.3           # last N% of dates (by calendar order) held out for testing
ALARM_THRESHOLD = 0.5         # probability threshold for "alarm raised"

# -- LIMITATION FIXES (config) ------------------------------------------------
DENSE_POSITIVE_STRIDE_MIN = 1  # fine window stride (minutes) inside pre-flare
                               # periods only -- multiplies real positive
                               # training windows ~STEP_MIN-fold without any
                               # synthetic data (see build_windows_for_date)
OPERATING_OBJECTIVE = "f2"     # which threshold becomes the live alarm point:
                               #   "f0.5" = fewer false alarms (precision-leaning)
                               #   "f1"   = balanced (old behaviour)
                               #   "f2"   = catch more flares (recall-leaning)
                               #   "tss"  = maximize the TSS skill score
                               # The sweep prints ALL four so the trade-off is
                               # an explicit, visible choice -- not a default.
CALIBRATION_FOLDS = 5          # walk-forward folds for OUT-OF-FOLD calibration
ENERGY_UPGRADE_MIN_RATIO = 5.0 # the "big real energy" promotion to significant
                               # can only UPGRADE borderline events (ratio >=
                               # this floor) -- never tiny blips. Without this
                               # floor the per-day 70th-percentile energy rule
                               # promoted the "biggest 3 of 9 microflares" on
                               # quiet days to positives, inflating the
                               # positive class to ~35% (seen on real data).

MODEL_OUTPUT_FOLDER = os.path.join(ncp.OUTPUT_FOLDER, "forecast_model")


