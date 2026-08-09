from .common import *
from .io_data import *

def measure_temperature(pi_data, t_start, t_end, t_peak, calibrator=None):
    """
    (a) CORONAL HEATING.
    Slice the SoLEXS spectrum over the flare, build the hot/cool band ratio,
    convert to an approximate temperature, and find when the temperature
    peaked relative to the flux peak.

    ACCURACY: if `calibrator` (SolexsCalibrator) is given, band counts are
    divided by the real per-channel ARF effective area before ratioing --
    this removes the instrument's own energy-dependent sensitivity curve
    from the hot/cool ratio, which raw counts do not. Without a calibrator,
    falls back to the raw-counts ratio (still physically motivated, just
    carries the instrument's response shape as a small bias). Either way
    this is a two-band ratio proxy, not a full CHIANTI/chisoth spectral
    fit (Section 4.4 of the SoLEXS manual) -- state that plainly if asked
    how it compares to a real XSPEC fit.

    Returns dict or None.
    """
    if pi_data is None:
        return None
    pt, spec, chan = pi_data
    m = (pt >= t_start) & (pt <= t_end)
    if m.sum() < 3:
        return None
    times = pt[m]
    seg = spec[m]
    energy = ch_to_keV(chan)

    if calibrator is not None:
        # effective-area-corrected band flux: sum(counts/area) is proportional
        # to true incident photon flux, not just "however many landed here".
        area = calibrator.eff_area_cm2
        n = min(len(area), seg.shape[1], len(energy))
        area_safe = np.where(area[:n] > 0, area[:n], np.nan)
        maskA = (energy[:n] >= BAND_A[0]) & (energy[:n] < BAND_A[1])
        maskB = (energy[:n] >= BAND_B[0]) & (energy[:n] < BAND_B[1])
        A = np.nansum(seg[:, :n][:, maskA] / area_safe[maskA], axis=1) if maskA.any() else np.zeros(len(seg))
        B = np.nansum(seg[:, :n][:, maskB] / area_safe[maskB], axis=1) if maskB.any() else np.zeros(len(seg))
        # Poisson variance of a weighted sum: Var(sum N_i/a_i) = sum N_i/a_i^2
        # (each channel's raw count N_i is Poisson, Var(N_i)=N_i).
        varA = np.nansum(seg[:, :n][:, maskA] / area_safe[maskA] ** 2, axis=1) if maskA.any() else np.zeros(len(seg))
        varB = np.nansum(seg[:, :n][:, maskB] / area_safe[maskB] ** 2, axis=1) if maskB.any() else np.zeros(len(seg))
    else:
        A = band_counts(seg, chan, *BAND_A)      # cool
        B = band_counts(seg, chan, *BAND_B)      # hot
        varA, varB = np.maximum(A, 0.0), np.maximum(B, 0.0)   # raw Poisson: Var(N)=N

    # differential flux per keV, then hot/cool ratio (temperature proxy)
    wa = BAND_A[1] - BAND_A[0]
    wb = BAND_B[1] - BAND_B[0]
    dA = np.maximum(A / wa, 1e-6)
    dB = np.maximum(B / wb, 1e-6)
    ratio = dB / dA                              # rises with temperature

    # Approximate temperature from a simple bremsstrahlung ratio (LABELLED PROXY).
    # diff_B/diff_A = exp(-(E_B-E_A)/kT)  ->  kT = (E_B-E_A)/ln(diff_A/diff_B)
    # Band midpoints used directly now (old code used fixed 2.0/5.0 keV that
    # didn't match BAND_A/BAND_B's actual edges).
    EA = 0.5 * (BAND_A[0] + BAND_A[1])
    EB = 0.5 * (BAND_B[0] + BAND_B[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        kT = (EB - EA) / np.log(np.maximum(dA / dB, 1.0000001))
    kT = np.clip(np.nan_to_num(kT, nan=0.0, posinf=5.0), 0.05, 5.0)   # keV, capped
    T_MK = kT / 0.08617                          # kT[keV] -> temperature in MK

    peak_i = int(np.argmax(ratio))
    temp_peak_time = float(times[peak_i])

    # ---- REAL 1-sigma ERROR BAR on the temperature (limitation fix #2) ------
    # kT = dE / ln(dA/dB). Propagating Poisson errors on A and B through
    # that formula: sigma_kT = (kT^2/dE) * sqrt((sA/A)^2 + (sB/B)^2).
    # Evaluated at the hottest sample (the same one temp_MK is read from).
    hot_i = int(np.nanargmax(T_MK)) if np.isfinite(T_MK).any() else peak_i
    temp_MK_err = None
    Ah, Bh = float(A[hot_i]), float(B[hot_i])
    if Ah > 0 and Bh > 0:
        rel = math.sqrt(max(varA[hot_i], 0.0) / Ah ** 2 +
                        max(varB[hot_i], 0.0) / Bh ** 2)
        kT_hot = float(kT[hot_i])
        sigma_kT = (kT_hot ** 2 / max(EB - EA, 1e-6)) * rel
        temp_MK_err = float(sigma_kT / 0.08617)            # keV -> MK, same as T_MK

    return dict(
        temp_MK        = float(np.nanmax(T_MK)),
        temp_MK_err    = temp_MK_err,                      # real +/- from counting stats
        hardness_ratio = float(np.nanmax(ratio)),
        temp_peak_time = temp_peak_time,
        temp_offset_s  = float(temp_peak_time - t_peak),   # +ve = temp peaks after flux
        curve_t        = times,                            # for the report plot
        curve_T_MK     = T_MK,                             # for the report plot
        calibrated     = calibrator is not None,
    )


def fit_isothermal_spectrum(pi_data, calibrator, t_start, t_end,
                            elo_kev=2.8, ehi_kev=12.0):
    """
    FULL SPECTRAL FIT (limitation fix #3): instead of comparing just TWO
    energy bands, fit an isothermal thermal-bremsstrahlung model to EVERY
    calibrated SoLEXS channel between elo_kev and ehi_kev at once, the same
    forward-folding way a professional tool like XSPEC does it:

        1. Guess a temperature kT.
        2. Compute the model photon spectrum a plasma at that kT would emit:
               f(E)  ~  K * exp(-E/kT) / (E * sqrt(kT))
           (standard thermal bremsstrahlung continuum shape; Gaunt factor
           approximated as constant over 2.8-12 keV -- stated plainly.)
        3. Fold it through the instrument's OWN response: multiply by the
           real per-channel ARF effective area and channel width, so the
           model predicts DETECTOR COUNTS, never comparing apples to oranges.
        4. Score model-vs-data with the Cash statistic (the correct
           maximum-likelihood statistic for Poisson counts -- Cash 1979,
           the same statistic XSPEC uses for low-count data), NOT chi-square.
        5. Repeat over a fine grid of kT; the best kT is the fit, and the
           range where Cash rises by <= 1 from its minimum is the REAL
           1-sigma confidence interval on the temperature.

    The normalisation K never needs its own search: for a pure scale factor
    the Cash-minimising K has the exact closed form K = sum(n)/sum(shape).

    Returns dict(T_MK_fit, T_MK_err_lo, T_MK_err_hi, cash_min, n_chan,
                 n_counts) or None if there's no spectrum/calibrator or too
    few counts to fit.
    """
    if pi_data is None or calibrator is None:
        return None
    pt, spec, chan = pi_data
    m = (pt >= t_start) & (pt <= t_end)
    if m.sum() < 3:
        return None

    energy = ch_to_keV(chan)
    area = calibrator.eff_area_cm2
    n = min(len(area), spec.shape[1], len(energy))
    e = np.asarray(energy[:n], dtype=float)
    a = np.asarray(area[:n], dtype=float)
    de = np.abs(np.gradient(e))                       # per-channel width in keV

    ok = (e >= elo_kev) & (e <= ehi_kev) & (a > 0) & (de > 0)
    if ok.sum() < 10:
        return None

    # total counts per channel over the flare window (integer-ish Poisson data)
    counts = np.asarray(spec[m][:, :n].sum(axis=0), dtype=float)[ok]
    counts = np.maximum(counts, 0.0)
    if counts.sum() < 50:                             # too few photons to fit
        return None
    e_ok, a_ok, de_ok = e[ok], a[ok], de[ok]

    def cash(n_obs, m_pred):
        # C = 2 * sum( m - n + n*ln(n/m) ); n=0 terms contribute 2m.
        m_pred = np.maximum(m_pred, 1e-30)
        term = m_pred - n_obs
        pos = n_obs > 0
        term[pos] += n_obs[pos] * np.log(n_obs[pos] / m_pred[pos])
        return 2.0 * float(np.sum(term))

    kT_grid = np.geomspace(0.08, 5.0, 160)            # keV (~0.9 to 58 MK)
    C = np.empty(len(kT_grid))
    for i, kT in enumerate(kT_grid):
        shape = np.exp(-e_ok / kT) / (e_ok * math.sqrt(kT)) * a_ok * de_ok
        ssum = shape.sum()
        if not np.isfinite(ssum) or ssum <= 0:
            C[i] = np.inf
            continue
        K = counts.sum() / ssum                       # closed-form best norm
        C[i] = cash(counts, K * shape)

    best = int(np.argmin(C))
    if not np.isfinite(C[best]) or best in (0, len(kT_grid) - 1):
        return None                                   # fit ran into a grid edge
    kT_fit = float(kT_grid[best])

    # 1-sigma interval: where Cash <= Cash_min + 1 (standard 1-parameter rule).
    # The crossings are found by LINEAR INTERPOLATION between grid points, so
    # a very well-constrained fit (interval narrower than one grid step)
    # still reports its real, small-but-nonzero error instead of 0.
    target = C[best] + 1.0
    kT_lo, kT_hi = kT_grid[0], kT_grid[-1]
    for i in range(best, 0, -1):                      # walk left to the crossing
        if C[i - 1] >= target:
            f = (target - C[i]) / max(C[i - 1] - C[i], 1e-12)
            kT_lo = float(kT_grid[i] + f * (kT_grid[i - 1] - kT_grid[i]))
            break
    for i in range(best, len(kT_grid) - 1):           # walk right to the crossing
        if C[i + 1] >= target:
            f = (target - C[i]) / max(C[i + 1] - C[i], 1e-12)
            kT_hi = float(kT_grid[i] + f * (kT_grid[i + 1] - kT_grid[i]))
            break

    KEV_TO_MK = 1.0 / 0.08617
    return dict(
        T_MK_fit    = kT_fit * KEV_TO_MK,
        T_MK_err_lo = (kT_fit - kT_lo) * KEV_TO_MK,
        T_MK_err_hi = (kT_hi - kT_fit) * KEV_TO_MK,
        cash_min    = float(C[best]),
        n_chan      = int(ok.sum()),
        n_counts    = float(counts.sum()),
    )


def measure_qpp(bt, flux, s_idx, e_idx, label="soft"):
    """
    (b) QUASI-PERIODIC PULSATIONS.
    Look for a repeating rhythm inside the flare: detrend, count sub-peaks,
    and confirm the period with an FFT. Returns period in seconds or None.
    """
    seg = flux[s_idx:e_idx].astype(float)
    if len(seg) < 12:                            # need at least ~2 minutes
        return None
    # remove the slow flare shape so only the wiggles remain
    trend = pd.Series(seg).rolling(5, center=True, min_periods=1).mean().values
    wig = seg - trend
    if np.std(wig) < 1e-6:
        return None

    # (1) count sub-peaks
    pk, _ = find_peaks(wig, distance=3)
    n_sub = len(pk)

    # (2) FFT periodogram
    w = wig - wig.mean()
    freqs = np.fft.rfftfreq(len(w), d=BIN_SEC)   # Hz
    power = np.abs(np.fft.rfft(w)) ** 2
    if len(power) < 3:
        return None
    power[0] = 0.0                               # ignore the zero-frequency term
    fi = int(np.argmax(power))
    if freqs[fi] <= 0:
        return None
    period = 1.0 / freqs[fi]

    # significance: the peak must stand clearly above the noise floor and we
    # need at least ~2-3 sub-peaks and a couple of cycles inside the flare
    strong = power[fi] > (power.mean() + 3 * power.std())
    enough_cycles = (e_idx - s_idx) * BIN_SEC > 2 * period
    if n_sub >= 3 and strong and enough_cycles:
        return float(period)
    return None


def measure_acceleration(helos_bands, t_start, t_end):
    """
    (c) PARTICLE ACCELERATION.
    Use the HEL1OS energy-band count rates (>= 20 keV) during the flare to fit a
    power law:  log(differential rate) vs log(energy).  Slope = -gamma.
    Returns (gamma, delta, hard_peak_time) or None if not enough hard bands.
    """
    if not helos_bands:
        return None
    energies, diffs = [], []
    hard_curve_t, hard_curve = None, None
    for b in helos_bands:
        m = (b["t"] >= t_start) & (b["t"] <= t_end)
        if m.sum() < 2:
            continue
        e_ctr = 0.5 * (b["e_lo"] + b["e_hi"])
        width = max(b["e_hi"] - b["e_lo"], 1e-3)
        mean_rate = float(np.nanmean(b["ctr"][m]))
        # collect the wide "total" band separately for timing/Neupert
        if b["e_lo"] <= 2.0 and b["e_hi"] >= 80.0:
            hard_curve_t, hard_curve = b["t"][m], b["ctr"][m]
        # only bands strictly above 20 keV are clean for the acceleration slope
        if b["e_lo"] >= 20.0 and mean_rate > 0:
            energies.append(e_ctr)
            diffs.append(mean_rate / width)      # differential rate per keV

    hard_peak_time = None
    if hard_curve is not None and len(hard_curve) > 0:
        hard_peak_time = float(hard_curve_t[int(np.argmax(hard_curve))])

    if len(energies) < 2:
        return dict(gamma=None, delta=None, gamma_err=None,
                    hard_peak_time=hard_peak_time,
                    energies=energies, diffs=diffs)

    logE = np.log10(np.array(energies))
    logF = np.log10(np.array(diffs))
    fitres = linregress(logE, logF)
    slope = fitres.slope
    gamma = -float(slope)                        # photon index
    delta = gamma + 1.0                          # electron index (thick-target)
    # REAL error bar on gamma (limitation fix #2): the least-squares fit's
    # own standard error on the slope IS the 1-sigma uncertainty on gamma
    # (gamma = -slope, and flipping the sign doesn't change the error).
    # With only 2 points there are no residual degrees of freedom, so the
    # stderr comes back 0/NaN -- report None honestly in that case.
    gamma_err = float(fitres.stderr) if (len(energies) > 2 and
                                         np.isfinite(fitres.stderr) and
                                         fitres.stderr > 0) else None
    # sanity: gamma should be positive & physical (~1.5 to 8)
    if not (1.0 < gamma < 10.0):
        gamma, delta, gamma_err = None, None, None
    return dict(gamma=gamma, delta=delta, gamma_err=gamma_err,
                hard_peak_time=hard_peak_time,
                energies=energies, diffs=diffs)


def measure_acceleration_time_resolved(helos_spec, calibrator, ev, n_slices=5):
    """
    RICHER version of particle acceleration: fits a SEPARATE power law at
    several moments through the flare (rise, near-peak, decay...), not just
    one number for the whole event. Uses the real per-channel HEL1OS
    spectrum at each moment.

    ACCURACY: each channel's counts are divided by the real ARF effective
    area before fitting (area-corrected -- proportional to true photon
    flux), not raw counts. A raw counts-vs-energy fit is biased by the
    instrument's own energy-dependent sensitivity curve; correcting for
    that first gives a more physically honest spectral index than a
    quick-look raw-counts scatter would.

    Returns a list of dicts (one per time slice, chronological):
      {t, energies, counts (area-corrected), gamma, delta}
    Empty list if no spectra/calibrator or not enough valid slices.
    """
    if helos_spec is None or calibrator is None:
        return []
    t, spec, exp = helos_spec
    m = (t >= ev["start"]) & (t <= ev["end"])
    idx = np.where(m)[0]
    if len(idx) < 2:
        return []
    pick = sorted(set(idx[np.linspace(0, len(idx) - 1, min(n_slices, len(idx))).astype(int)].tolist()))

    energy = calibrator.chan_emid
    area = calibrator.eff_area_cm2
    elo = calibrator.recommended_elo_kev
    valid = (energy >= elo) & (area > 0)

    slices = []
    for row_idx in pick:
        counts = spec[row_idx]
        n = min(len(counts), len(energy))
        vmask = valid[:n]
        e = energy[:n][vmask]
        c = counts[:n][vmask]
        ac = c / area[:n][vmask]                  # area-corrected -> proportional to true photon flux
        good = (c > 0) & np.isfinite(ac) & (ac > 0)
        if good.sum() < 5:
            continue
        e_g, ac_g = e[good], ac[good]
        try:
            fitres = linregress(np.log10(e_g), np.log10(ac_g))
            slope, intercept = fitres.slope, fitres.intercept
        except Exception:
            continue
        gamma = -float(slope)
        if not (0.5 < gamma < 12.0):
            continue
        g_err = float(fitres.stderr) if (np.isfinite(fitres.stderr) and fitres.stderr > 0) else None
        slices.append(dict(t=float(t[row_idx]), energies=e_g, counts=ac_g,
                           gamma=gamma, gamma_err=g_err,
                           delta=gamma + 1.0, intercept=float(intercept)))
    return slices


def measure_neupert(bt, flux, s_idx, e_idx, helos_bands):
    """
    (d) NEUPERT EFFECT.
    The rate of rise of the soft X-ray flux, d(SXR)/dt, should look like the
    hard X-ray light-curve.  We line them up on a common clock and correlate.
    Returns correlation (-1..1) or None.
    """
    if not helos_bands:
        return None
    # widest band = total hard X-ray
    total = None
    for b in helos_bands:
        if b["e_lo"] <= 2.0 and b["e_hi"] >= 80.0:
            total = b
    if total is None:
        total = max(helos_bands, key=lambda b: b["e_hi"] - b["e_lo"])

    t0, t1 = bt[s_idx], bt[e_idx]
    hm = (total["t"] >= t0) & (total["t"] <= t1)
    if hm.sum() < 4:
        return None

    # d(SXR)/dt over the flare, sampled on the SoLEXS bins
    sxr_t = bt[s_idx:e_idx]
    sxr = flux[s_idx:e_idx].astype(float)
    dsxr = np.gradient(sxr, sxr_t)

    # interpolate the hard X-ray onto the SoLEXS time grid
    hard_on_sxr = np.interp(sxr_t, total["t"][hm], total["ctr"][hm])
    if np.std(dsxr) < 1e-9 or np.std(hard_on_sxr) < 1e-9:
        return None
    try:
        corr, _ = pearsonr(dsxr, hard_on_sxr)
    except Exception:
        return None
    return float(corr)


def measure_neupert_lag(bt, flux, s_idx, e_idx, helos_bands, max_lag_s=300):
    """
    QUANTITATIVE Neupert check (limitation fix #3): instead of one
    correlation at zero lag, scan time offsets of the hard X-ray curve from
    -max_lag_s to +max_lag_s and find where the correlation with d(SXR)/dt
    truly peaks. Physics says hard X-rays and the soft-X-ray rise rate
    should line up NEAR zero lag -- a best-fit lag of many minutes means
    the "match" was coincidence, however high the raw correlation was.
    Reports the number a scientist would ask for: best lag (seconds,
    positive = hard X-rays LEAD) and the correlation at that lag.

    Returns dict(best_lag_s, corr_at_best, corr_zero) or None.
    """
    if not helos_bands:
        return None
    total = None
    for b in helos_bands:
        if b["e_lo"] <= 2.0 and b["e_hi"] >= 80.0:
            total = b
    if total is None:
        total = max(helos_bands, key=lambda b: b["e_hi"] - b["e_lo"])

    t0, t1 = bt[s_idx], bt[e_idx]
    hm = (total["t"] >= t0 - max_lag_s) & (total["t"] <= t1 + max_lag_s)
    if hm.sum() < 4:
        return None
    ht, hc = total["t"][hm], total["ctr"][hm]

    sxr_t = bt[s_idx:e_idx]
    sxr = flux[s_idx:e_idx].astype(float)
    if len(sxr_t) < 4:
        return None
    dsxr = np.gradient(sxr, sxr_t)
    if np.std(dsxr) < 1e-9:
        return None

    lags = np.arange(-max_lag_s, max_lag_s + 1, BIN_SEC, dtype=float)
    best_lag, best_corr, corr_zero = None, -2.0, None
    for lag in lags:
        # positive lag = hard curve shifted LATER = hard X-rays led by `lag`
        hard_on_sxr = np.interp(sxr_t, ht + lag, hc)
        if np.std(hard_on_sxr) < 1e-9:
            continue
        try:
            r, _ = pearsonr(dsxr, hard_on_sxr)
        except Exception:
            continue
        if lag == 0:
            corr_zero = float(r)
        if r > best_corr:
            best_corr, best_lag = float(r), float(lag)
    if best_lag is None:
        return None
    return dict(best_lag_s=best_lag, corr_at_best=best_corr, corr_zero=corr_zero)


# =============================================================================
#  SECTION 5.5  --  REPORT PLOTTING (one multi-panel PNG per date)
# =============================================================================

def qpp_diagnostics(bt, flux, s_idx, e_idx):
    """
    Same maths as measure_qpp but returns everything the report plot needs:
    excess curve, smoothed trend, sub-peak positions, FFT freqs+power, period.
    Returns dict or None.
    """
    seg = flux[s_idx:e_idx].astype(float)
    if len(seg) < 12:
        return None
    trend = pd.Series(seg).rolling(5, center=True, min_periods=1).mean().values
    wig = seg - trend
    if np.std(wig) < 1e-6:
        return None
    pk, _ = find_peaks(wig, distance=3)
    w = wig - wig.mean()
    freqs = np.fft.rfftfreq(len(w), d=BIN_SEC)
    power = np.abs(np.fft.rfft(w)) ** 2
    if len(power) < 3:
        return None
    power[0] = 0.0
    fi = int(np.argmax(power))
    period = 1.0 / freqs[fi] if freqs[fi] > 0 else None
    t_min = (bt[s_idx:e_idx] - bt[0]) / 60.0          # minutes since obs start
    bg_local = np.minimum(trend, seg)
    return dict(t_min=t_min, excess=seg - seg.min(), smooth=trend - seg.min(),
                peaks=pk, freqs=freqs, power=power, period=period)


def build_hxr_series(helos_bands):
    """
    HEL1OS-only detection input: take the widest band (the 'total' one),
    rebin its 1-second count rate onto BIN_SEC bins -> (bt, flux).
    Returns (bt, flux, band) or None.
    """
    if not helos_bands:
        return None
    total = max(helos_bands, key=lambda b: b["e_hi"] - b["e_lo"])
    bt, fx = rebin(total["t"], total["ctr"])
    return bt, fx, total


def solexs_flux_series(bt, pi_data, calibrator, elo_kev=2.8, ehi_kev=None):
    """
    Real W/m^2 for EVERY bin of the report plot (not just per-flare peaks).
    Bins the 1-second SoLEXS spectra onto the exact same bt grid as the
    counts light curve (so the two overlay perfectly), summing counts per
    bin and tracking how many 1-second spectra actually landed in each bin
    (the real exposure -- usually bin_sec, less at data gaps).

    elo_kev/ehi_kev restrict the integration to a specific energy band (used
    for the 3-way energy-resolved plot); defaults to SoLEXS's full reliable
    range if ehi_kev is left as None.

    Returns an array the same length as bt (NaN where there's no spectrum
    coverage for that bin), or None if pi_data/calibrator are unavailable.
    """
    if pi_data is None or calibrator is None or len(bt) < 2:
        return None
    pt, spec, chan = pi_data
    n_ch = spec.shape[1]
    n_bins = len(bt)
    bin_sec = float(bt[1] - bt[0]) if n_bins > 1 else BIN_SEC

    acc = np.zeros((n_bins, n_ch))
    cnt = np.zeros(n_bins)
    idx = ((pt - bt[0]) / bin_sec).astype(int)
    ok = (idx >= 0) & (idx < n_bins)
    np.add.at(acc, idx[ok], spec[ok])
    np.add.at(cnt, idx[ok], 1)

    flux = np.full(n_bins, np.nan)
    good = cnt > 0
    flux[good] = calibrator.counts_to_flux_series(acc[good], exposure_sec=cnt[good],
                                                  elo_kev=elo_kev, ehi_kev=ehi_kev)
    return flux


def hel1os_flux_series(helos_spec, calibrator, elo_kev=None, ehi_kev=None):
    """
    Real W/m^2 at HEL1OS's own native cadence (20 s), using the vectorized
    batch calibration. elo_kev/ehi_kev restrict to a specific band (used for
    the 3-way energy-resolved plot); defaults to the calibrator's own
    recommended range if left as None. Returns (times_unix, flux_wm2) or None.
    """
    if helos_spec is None or calibrator is None:
        return None
    t, spec, exp = helos_spec
    flux = calibrator.counts_to_flux_series(spec, exposure_sec=exp, elo_kev=elo_kev, ehi_kev=ehi_kev)
    return t, flux


def build_three_band_series(mode, bt, pi_data, helos_spec_cdte, helos_spec_czt):
    """
    Real continuous W/m^2 curves for the three canonical bands, for the
    energy-resolved overview plot (row 0). Each entry is (times, flux_wm2)
    or None if that band has no coverage at all this date.

      band1 (2.8-20 keV) : SoLEXS, when this mode has SoLEXS. In HEL1OS-only
                           mode there's no SoLEXS at all, so band1 falls back
                           to CDTE's own 5-20 keV sub-range instead (still a
                           real, calibrated curve, just from the hard-X-ray
                           detector's own low end rather than the dedicated
                           soft-X-ray instrument).
      band2 (20-60 keV)  : CZT preferred, CDTE fallback.
      band3 (60-150 keV) : CZT only (CDTE's calibration doesn't reach this high).
    """
    out = {"band1": None, "band2": None, "band3": None}

    if mode != "HEL1OS" and pi_data is not None and SOLEXS_CAL is not None:
        f = solexs_flux_series(bt, pi_data, SOLEXS_CAL, elo_kev=2.8, ehi_kev=20.0)
        if f is not None:
            out["band1"] = (bt, f)
    elif mode == "HEL1OS" and helos_spec_cdte is not None and HEL1OS_CAL is not None:
        # CDTE's effective area near 5 keV is ~0.000017 cm^2 vs ~0.05 cm^2 at
        # 9.5 keV -- dividing low counts by that near-zero area amplifies
        # ordinary Poisson noise into huge spikes. Use the calibrator's own
        # recommended floor (9.5 keV), not an arbitrary lower number.
        r = hel1os_flux_series(helos_spec_cdte, HEL1OS_CAL["cdte"],
                               elo_kev=HEL1OS_CAL["cdte"].recommended_elo_kev, ehi_kev=20.0)
        if r is not None:
            out["band1"] = r

    if HEL1OS_CAL is not None:
        if helos_spec_czt is not None:
            r = hel1os_flux_series(helos_spec_czt, HEL1OS_CAL["czt"], elo_kev=20.0, ehi_kev=60.0)
            if r is not None:
                out["band2"] = r
        elif helos_spec_cdte is not None:
            r = hel1os_flux_series(helos_spec_cdte, HEL1OS_CAL["cdte"], elo_kev=20.0, ehi_kev=60.0)
            if r is not None:
                out["band2"] = r
        if helos_spec_czt is not None:
            r = hel1os_flux_series(helos_spec_czt, HEL1OS_CAL["czt"], elo_kev=60.0, ehi_kev=150.0)
            if r is not None:
                out["band3"] = r
    return out
