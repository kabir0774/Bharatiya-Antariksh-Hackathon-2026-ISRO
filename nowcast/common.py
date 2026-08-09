#!/usr/bin/env python3
# =============================================================================
#  nowcast_physics.py
#  ISRO Aditya-L1 Hackathon 2026  --  Nowcast Physics + Cross-Check Module
# =============================================================================
#
#  WHAT THIS FILE DOES (in plain words)
#  ------------------------------------
#  1. Finds a flare in the SoLEXS soft X-ray light-curve (same method as the
#     main pipeline: rolling quiet-Sun background + 5-sigma threshold).
#  2. For every flare it then measures FOUR physics signatures:
#        (a) Coronal heating  -> plasma temperature from the SoLEXS spectrum
#        (b) QPP pulsations    -> a repeating rhythm inside the flare
#        (c) Particle acceleration -> the hard X-ray slope (gamma, delta) from HEL1OS
#        (d) Neupert effect    -> does the hard X-ray match the rate-of-rise of the soft X-ray
#  3. It runs SIX consistency cross-checks (each returns PASS / FLAG / N/A).
#  4. It combines them into ONE confidence score per flare.
#  5. It computes total radiated energy TWO ways:
#        - radiated_energy_proxy  : counts x seconds (always available)
#        - radiated_energy_jm2    : REAL joules/m^2, using the SoLEXS CALDB
#          ARF/RMF (caldb_calibration.py) to convert counts -> W/m^2 first.
#          Needs TOOLS_FOLDER (below) to point at Tools\CALDB; if that
#          folder isn't found, the script prints a warning and the real
#          columns come out as None -- everything else still runs.
#  6. It writes everything into one enriched catalog: physics_catalog.csv
#
#  IMPORTANT: This file does NOT change your existing pipeline. It is standalone.
#  Run it, look at the CSV, and if you like it we merge it later.
#
#  HOW TO RUN
#  ----------
#     (activate your venv, then)
#     python nowcast_physics.py
#
#  It reads your real files in the exact format we confirmed:
#     SoLEXS .lc  : table RATE,     columns TIME, COUNTS          (1 per second)
#     SoLEXS .pi  : table SPECTRUM, columns TSTART, CHANNEL, COUNTS(340 array)
#     HEL1OS LC   : bands named e.g. CDTE1_LC_BAND_20.00KEV_TO_30.00KEV,
#                   columns MJD, ISOT, CTR, STAT_ERR
# =============================================================================

# ------------------------- CONFIG (edit these paths) -------------------------
SOLEXS_FOLDER = r"C:\kabir\ISRO hackathon\Dataset SOLEXS"
HELOS_FOLDER  = r"C:\kabir\ISRO hackathon\dataset HEL1OS"
OUTPUT_FOLDER = r"C:\kabir\ISRO hackathon\physics_output"
REPORT_FOLDER = r"C:\kabir\ISRO hackathon\Nowcast_Physics_output"   # report PNGs go here
TOOLS_FOLDER  = r"C:\kabir\ISRO hackathon\Tools"   # where caldb_calibration.py + CALDB\ live

# ZIP auto-extract: drop fresh PRADAN zips STRAIGHT INTO the data folders
# above. On every run the pipeline extracts any zip it hasn't seen before
# (right there, next to the zip), remembers it in .extracted_zips.txt, and
# skips it on later runs. No separate ZIP folder needed.
SOLEXS_ZIP_FOLDER = SOLEXS_FOLDER
HELOS_ZIP_FOLDER  = HELOS_FOLDER

ASK_DATE = True           # True  = show available dates, ask which one to run.
                          # False = process every SoLEXS date automatically.

PLOT_REPORTS = True       # True = write one multi-panel report PNG per date
                          # into OUTPUT_FOLDER (report_<date>_<mode>.png).
AUTO_OPEN_REPORTS = True  # True = open each report PNG in your default image
                          # viewer right after it's saved. Re-running the same
                          # date never overwrites -- it saves report_..._2.png,
                          # _3.png, etc, so old runs stay on disk.

# --- Detection settings (kept identical to the main nowcast pipeline) --------
BIN_SEC        = 10       # rebin the 1-second data into 10-second bins
KEV_TO_JOULE_CONST = 1.602176634e-16   # 1 keV in joules (matches caldb_calibration.py)
CM2_TO_M2_CONST    = 1e-4               # 1 cm^2 in m^2 (matches caldb_calibration.py)
BG_WINDOW_BINS = 120      # rolling background window (120 x 10s = 20 minutes)
SIGMA_THRESH   = 5        # flag when flux rises > 5 sigma above background
MIN_DUR_BINS   = 6        # a flare must last >= 6 bins (60 s); rejects cosmic rays
MERGE_GAP_S    = 120      # events separated by < 2 min are one flare (merge them)

# --- v4: Poisson-FOCuS detection engine + extra false-positive guards -------
# Fixes the "34-53 flares/day" over-detection found on real data. A fixed
# rolling-window sigma test can't tell a real few-second burst from a slow
# drift using the same math -- it either misses short bursts or false-fires
# on slow rises. FOCuS tests every possible window length ending "now" at
# once (a real changepoint algorithm -- Ward et al. 2023, used for onboard
# GRB triggers) and only fires on the one that's genuinely significant.
FOCUS_THRESHOLD   = 300.0  # empirically calibrated, not guessed: ran 60 independent
                           # quiet-noise simulations (rates 5-60 counts/sec, matching
                           # our real SoLEXS/HEL1OS range) -- max FOCuS stat reached
                           # in pure noise was 133. Real injected flares hit
                           # 30,000-240,000. 300 gives >2x margin above the worst
                           # noise case observed while sitting ~100x below the
                           # weakest real flare tested -- huge, safe separation.
FOCUS_MAX_CURVES  = 64     # safety cap on retained candidate changepoints
FOCUS_MAX_WINDOW_BINS = 360   # cap the longest window FOCuS will consider to
                              # 1 hour (360 x 10s). Without this, one big burst
                              # can leave a hull candidate whose LLR takes
                              # HOURS to naturally decay below threshold even
                              # after the rate is back to normal (confirmed:
                              # without the cap, a single early flare kept the
                              # rest of the day flagged as "still in event").
                              # No real solar flare in our data runs anywhere
                              # near an hour, so this costs nothing real.
MIN_RISE_RATIO    = 1.4    # noise-blip gate: peak must beat its own start by
                           # this much (matches the GOES 4-min rise convention)
                           # or the candidate is silently dropped, never logged
FRED_ALPHA_FAST   = 0.5    # shape-filter fast pole (short memory, tracks rise)
FRED_ALPHA_SLOW   = 0.9    # shape-filter slow pole (long memory, tracks decay)

# --- SoLEXS energy calibration (SoLEXS User Manual v1.0) ---------------------
CH_GAIN_LOW  = 0.0476     # keV per channel, channels 1-168
CH_GAIN_HIGH = 0.0952     # keV per channel, channels 169-340
CH_BREAK     = 168

# --- Energy bands (keV) for the temperature proxy ----------------------------
# Floor is 2.8 keV, not SoLEXS's raw 2 keV floor -- ISRO's SoLEXS manual
# Section 4.5.2 explicitly says the ARF/RMF aren't calibrated below 2.8 keV
# and to exclude those channels. (Old BAND_A started at 1.0 keV, which was
# both below the raw 2 keV detection floor AND inside the uncalibrated zone.)
BAND_A = (2.8, 6.0)       # cool  ("A")
BAND_B = (6.0, 12.0)      # hot   ("B")  -> heats up before/around a flare
BAND_C = (12.0, 22.0)     # hard  ("C")  -> early particle-acceleration hint, SoLEXS ceiling

# --- Cross-check thresholds (starting values; we tune after first run) -------
NEUPERT_MIN_CORR   = 0.30     # check 1: hard vs d(SXR)/dt correlation must exceed this
HEAT_TIMING_MAX_S  = 300      # check 3: temperature peak within 5 min of flux peak
QPP_AGREE_FRAC     = 0.30     # check 5: soft & hard QPP periods within 30%
SIG_EVENT_RATIO    = 8        # a "significant" flare peaks >= 8x background

# =============================================================================
import os, sys, glob, re, warnings, subprocess, math
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.stats import linregress, pearsonr

import matplotlib
matplotlib.use("Agg")                     # headless: save PNGs, never open windows
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Real CALDB calibration (SoLEXS counts -> W/m^2). Optional: if the
# module or files aren't found, we print a warning and fall back to the
# old counts-based proxy so the pipeline still runs. -------------------------
if TOOLS_FOLDER not in sys.path:
    sys.path.insert(0, TOOLS_FOLDER)
try:
    from caldb_calibration import SolexsCalibrator, Hel1osCalibrator
    SOLEXS_CAL = SolexsCalibrator()
    HEL1OS_CAL = {"cdte": Hel1osCalibrator("cdte"), "czt": Hel1osCalibrator("czt")}
    print("[CALDB] SoLEXS + HEL1OS calibration loaded -- radiated energy will use real W/m^2.")
except Exception as e:
    SOLEXS_CAL = None
    HEL1OS_CAL = None
    print(f"[CALDB] Calibration NOT loaded ({e})")
    print("[CALDB] Falling back to the counts-only radiated-energy proxy.")

# MJD of the Unix epoch (1970-01-01).  Your files use MJDREFI = 40587, i.e. Unix.
MJD_UNIX_EPOCH = 40587.0

def mjd_to_unix(mjd):
    """Convert Modified Julian Date -> Unix seconds."""
    return (np.asarray(mjd, dtype=float) - MJD_UNIX_EPOCH) * 86400.0

def unix_to_utc(t):
    """Unix seconds -> readable UTC string."""
    try:
        return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "?"

def ch_to_keV(ch):
    """SoLEXS channel number -> energy in keV (two-slope calibration)."""
    ch = np.asarray(ch, dtype=float)
    return np.where(ch <= CH_BREAK,
                    ch * CH_GAIN_LOW,
                    CH_BREAK * CH_GAIN_LOW + (ch - CH_BREAK) * CH_GAIN_HIGH)


