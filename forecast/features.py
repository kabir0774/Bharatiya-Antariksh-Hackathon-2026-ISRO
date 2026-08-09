from .common import *
from .data import *

# =============================================================================
#  SECTION 3 -- FEATURE EXTRACTION (36 physics-informed features)
# =============================================================================

FEATURE_NAMES = [
    # -- SoLEXS total flux, background-normalized (FIX 2) (10) --------------
    "flux_mean", "flux_max", "flux_end", "flux_trend_slope", "flux_std",
    "dflux_mean", "dflux_max", "dflux_pos_frac", "excess_above_bg", "max_sigma",
    # -- SoLEXS energy-calibrated bands (8) ----------------------------------
    "flux_bandA", "flux_bandB", "flux_bandC",
    "ratio_B_A", "ratio_C_total", "ratio_B_A_trend", "ratio_B_A_end", "spectral_slope",
    # -- SoLEXS activity shape (5) -------------------------------------------
    "n_micro_peaks", "recent_vs_early", "flux_skewness", "flux_start", "micro_energy",
    # -- HEL1OS hard X-ray, zero-filled if unavailable (9) -------------------
    "helos_flux_mean", "helos_flux_max", "helos_trend_slope", "helos_std",
    "helos_dflux_max", "helos_max_sigma", "helos_recent_vs_early",
    "helos_n_micro_peaks", "helos_precursor",
    # -- Cross-instrument (4) -------------------------------------------------
    "helos_to_solexs_ratio", "helos_lead_signal", "cross_corr", "hard_soft_trend_diff",
    # -- NEW: event-clustering / Hawkes-style history (2) ---------------------
    # flares genuinely cluster in time (one flare raises the odds of another
    # soon after); these were missing entirely before -- every other feature
    # only looks INSIDE the current window, never backward at flare history.
    "time_since_last_flare_min", "decayed_flare_history",
    # -- NEW: data-quality (1) -------------------------------------------------
    # how much of this window is real data vs interpolated/missing -- lets
    # the model learn "less sure right now" instead of training on gaps as
    # if they were clean signal.
    "data_gap_frac",
    # -- NEW: real calibrated temperature trend (1) ----------------------------
    # ratio_B_A/_trend above are a raw-counts hardness PROXY for heating;
    # this is the real ARF-corrected coronal temperature (same physics as
    # nowcast's measure_temperature) computed on just this window, so it is
    # a genuinely calibrated MK trend, not a proxy.
    "temp_trend_MK_per_min",
    # -- NEW: per-window particle-acceleration steepness (2) -------------------
    # LIMITATION FIX: gamma was measured by the nowcaster per flare but never
    # made it into the forecaster's clues. These are computed per WINDOW from
    # HEL1OS band rates >= 20 keV (same log(rate/width) vs log(energy) slope
    # nowcast's measure_acceleration uses). A hardening spectrum (gamma
    # FALLING -- more high-energy photons relative to low) is a classic
    # pre-flare particle-acceleration signature; gamma_trend < 0 = hardening.
    # Zero-filled when HEL1OS has no coverage, like the other HEL1OS clues.
    "gamma_window", "gamma_trend",
    # -- NEW: precursor CONCENTRATION score (1) -------------------------------
    # LIMITATION: with ~47 real positive examples spread across 42 diffuse
    # features, a black-box model rarely has enough tightly-clustered evidence
    # to honestly output a HIGH probability anywhere -- the calibrator ends up
    # reporting mostly small numbers because most feature combinations really
    # ARE that safe, and forcing a bigger number without more evidence would
    # just be a lie. This feature is one deliberate attempt at a real fix
    # WITHOUT needing new data: instead of scattering the strongest known
    # precursor signals (background excess, statistical significance,
    # micro-flare burstiness, spectral hardening, heating rate) across five
    # separate dimensions the model has to learn to combine on its own, this
    # multiplies squashed (0,1) versions of each together. The product is
    # only large when ALL of them are simultaneously elevated -- an AND, not
    # an OR -- which is a rarer, sharper, more specific condition than any
    # one signal alone. If a real high-confidence precursor STATE exists in
    # this data, concentrating the evidence like this gives the model its
    # best honest shot at finding it; reliability_report() shows whether it
    # actually did.
    "precursor_concentration",
]
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 43, f"expected 43 features, got {N_FEATURES}"


def _slope(y):
    if len(y) < 2 or np.all(~np.isfinite(y)):
        return 0.0
    x = np.arange(len(y))
    good = np.isfinite(y)
    if good.sum() < 2:
        return 0.0
    try:
        return float(np.polyfit(x[good], y[good], 1)[0])
    except Exception:
        return 0.0


def _count_micro_peaks(y):
    from scipy.signal import find_peaks
    if len(y) < 5:
        return 0, 0.0
    y = np.nan_to_num(y)
    pk, props = find_peaks(y, prominence=max(np.std(y), 1e-6) * 0.5)
    heights = y[pk] if len(pk) else np.array([0.0])
    return len(pk), float(np.mean(heights))


def extract_features(flux_n, bandA_n, bandB_n, bandC_n, helos_n,
                     time_since_last_flare_min=None, decayed_flare_history=0.0,
                     data_gap_frac=0.0, temp_trend_mk_per_min=0.0,
                     gamma_window=0.0, gamma_trend=0.0):
    """
    Extract the 40-feature vector from one window's arrays. ALL flux-style
    inputs (flux_n, bandA/B/C_n, helos_n) are already background-normalized
    (FIX 2) -- i.e. "how many times above local quiet-Sun", not raw counts.
    helos_n may be None if HEL1OS has no coverage this date/window -- those
    9 features + the 4 cross-instrument features are zero-filled, matching
    the "works standalone on SoLEXS-only dates" design goal.

    The last 4 (indices 36-39) are computed OUTSIDE this window's own array
    -- clustering history, data quality, and real temperature trend -- and
    passed in by build_windows_for_date(), which has the context (the day's
    full event list, the day's coverage, the day's pi_data) this function
    doesn't see.
    """
    f = np.zeros(N_FEATURES)
    flux_n = np.asarray(flux_n, dtype=float)
    n = len(flux_n)
    if n < 3:
        return f

    end_seg = flux_n[-max(n // 10, 1):]
    start_seg = flux_n[:max(n // 10, 1)]
    d = np.diff(flux_n)

    f[0] = np.nanmean(flux_n)                                  # flux_mean
    f[1] = np.nanmax(flux_n)                                    # flux_max
    f[2] = np.nanmean(end_seg)                                  # flux_end
    f[3] = _slope(flux_n)                                        # flux_trend_slope
    f[4] = np.nanstd(flux_n)                                    # flux_std
    f[5] = np.nanmean(d) if len(d) else 0.0                     # dflux_mean
    f[6] = np.nanmax(d) if len(d) else 0.0                      # dflux_max
    f[7] = float(np.mean(d > 0)) if len(d) else 0.0             # dflux_pos_frac
    f[8] = float(np.nanmax(flux_n) - 1.0)                       # excess_above_bg (flux_n is already x-background, so 1.0 = at background)
    f[9] = float(np.nanmax(flux_n))                              # max_sigma (proxy: normalized peak height)

    if bandA_n is not None and bandB_n is not None and bandC_n is not None:
        a, b, c = np.nanmean(bandA_n), np.nanmean(bandB_n), np.nanmean(bandC_n)
        tot = a + b + c
        f[10], f[11], f[12] = a, b, c
        f[13] = b / max(a, 1e-6)                                 # ratio_B_A
        f[14] = c / max(tot, 1e-6)                               # ratio_C_total
        ratio_series = bandB_n / np.maximum(bandA_n, 1e-6)
        f[15] = _slope(ratio_series)                              # ratio_B_A_trend
        f[16] = np.nanmean(ratio_series[-max(len(ratio_series)//10, 1):])  # ratio_B_A_end
        with np.errstate(divide="ignore", invalid="ignore"):
            f[17] = np.log10(max(c, 1e-6)) / max(np.log10(max(a, 1e-6)), 1e-6)  # spectral_slope

    n_pk, micro_e = _count_micro_peaks(flux_n)
    f[18] = n_pk                                                  # n_micro_peaks
    f[19] = np.nanmean(end_seg) / max(np.nanmean(start_seg), 1e-6)  # recent_vs_early
    f[20] = float(pd.Series(flux_n).skew()) if n >= 3 else 0.0    # flux_skewness
    f[21] = np.nanmean(start_seg)                                 # flux_start
    f[22] = micro_e                                                # micro_energy

    if helos_n is not None and len(helos_n) >= 3 and np.any(np.isfinite(helos_n)):
        h = np.asarray(helos_n, dtype=float)
        hd = np.diff(h)
        h_end = h[-max(len(h) // 10, 1):]
        h_start = h[:max(len(h) // 10, 1)]
        f[23] = np.nanmean(h)                                     # helos_flux_mean
        f[24] = np.nanmax(h)                                      # helos_flux_max
        f[25] = _slope(h)                                          # helos_trend_slope
        f[26] = np.nanstd(h)                                      # helos_std
        f[27] = np.nanmax(hd) if len(hd) else 0.0                 # helos_dflux_max
        f[28] = float(np.nanmax(h))                                # helos_max_sigma (proxy)
        f[29] = np.nanmean(h_end) / max(np.nanmean(h_start), 1e-6)  # helos_recent_vs_early
        hn_pk, _ = _count_micro_peaks(h)
        f[30] = hn_pk                                              # helos_n_micro_peaks
        # precursor: is HEL1OS rising while SoLEXS (this window) is still flat/falling?
        f[31] = 1.0 if (_slope(h) > 0 and f[3] <= 0) else 0.0     # helos_precursor

        f[32] = np.nanmean(h) / max(np.nanmean(flux_n), 1e-6)     # helos_to_solexs_ratio
        f[33] = np.nanmax(h_start)                                 # helos_lead_signal
        m = min(len(h), len(flux_n))
        if m >= 3 and np.std(h[:m]) > 0 and np.std(flux_n[:m]) > 0:
            f[34] = float(np.corrcoef(h[:m], flux_n[:m])[0, 1])   # cross_corr
        f[35] = _slope(h) - _slope(flux_n)                         # hard_soft_trend_diff

    # -- NEW (36-39): clustering / data-quality / real temperature -----------
    f[36] = (min(time_since_last_flare_min, _TIME_CAP_MIN) if time_since_last_flare_min is not None
            else _TIME_CAP_MIN)                                    # time_since_last_flare_min
    f[37] = float(decayed_flare_history)                           # decayed_flare_history
    f[38] = float(np.clip(data_gap_frac, 0.0, 1.0))                # data_gap_frac
    f[39] = float(temp_trend_mk_per_min)                           # temp_trend_MK_per_min
    f[40] = float(gamma_window)                                    # gamma_window (0 = no HEL1OS/no fit)
    f[41] = float(gamma_trend)                                     # gamma_trend  (<0 = spectrum hardening)

    # f[42] precursor_concentration -- see FEATURE_NAMES comment for the
    # reasoning. Each _squash(...) turns one raw signal into a (0,1) "how
    # elevated is this, really" score via a logistic centered on a
    # physically-motivated threshold; the product is near 1 only when every
    # signal is elevated TOGETHER, near 0 if even one is weak.
    def _squash(x, center, scale):
        try:
            return 1.0 / (1.0 + math.exp(-(float(x) - center) / max(scale, 1e-9)))
        except OverflowError:
            return 0.0 if x < center else 1.0

    s_excess = _squash(f[8], center=8.0, scale=3.0)     # background jump nearing the "significant" floor
    s_sigma = _squash(f[9], center=3.0, scale=1.5)      # statistical significance of the peak
    s_burst = _squash(f[18], center=3.0, scale=2.0)     # micro-flare burstiness
    s_hard = _squash(-gamma_trend, center=0.3, scale=0.3)   # spectrum actively hardening
    s_heat = _squash(temp_trend_mk_per_min, center=0.05, scale=0.05)  # coronal heating rate
    f[42] = s_excess * s_sigma * s_burst * s_hard * s_heat

    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
