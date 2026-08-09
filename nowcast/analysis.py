from .common import *
from .io_data import *
from .detection import *
from .physics import *

# =============================================================================
#  SECTION 6  --  THE SIX CROSS-CHECKS + CONFIDENCE
# =============================================================================

def run_cross_checks(ev, temp, qpp_soft, qpp_hard, accel, neupert, eband=None,
                     band_confirm=None, cross_agree=None,
                     specfit=None, band_vote=None):
    """
    Each check returns 'PASS', 'FLAG', or 'N/A'.

    v4 confidence: noisy-OR instead of a bare (#PASS-#FLAG)/#applicable ratio.
    A plain ratio treats "3 PASS out of 3 applicable" the same as "3 PASS out
    of 6 applicable, other 3 N/A" once you divide by "applicable" -- it can't
    tell strong evidence from thin evidence. noisy-OR combines PASSes as
    independent probabilistic evidence (each one raises confidence, and they
    compound properly instead of just averaging), and now ALSO folds in two
    detector-agreement signals as extra evidence: how far FOCuS's statistic
    sat above its threshold, and whether the burst's own shape genuinely
    looked like a flare (fred_shape_stat). FLAGs apply a penalty afterward --
    evidence against isn't the mirror image of evidence for.
    """
    checks = {}

    # 1. Neupert correlation must exceed the minimum
    if neupert is None:
        checks["neupert"] = "N/A"
    else:
        checks["neupert"] = "PASS" if neupert >= NEUPERT_MIN_CORR else "FLAG"

    # 2. Acceleration timing: hard X-ray peak at or before the soft peak
    if accel is None or accel.get("hard_peak_time") is None:
        checks["accel_timing"] = "N/A"
    else:
        checks["accel_timing"] = "PASS" if accel["hard_peak_time"] <= ev["peak"] + BIN_SEC else "FLAG"

    # 3. Heating timing: temperature peaks within 5 minutes of the flux peak
    if temp is None:
        checks["heat_timing"] = "N/A"
    else:
        checks["heat_timing"] = "PASS" if abs(temp["temp_offset_s"]) <= HEAT_TIMING_MAX_S else "FLAG"

    # 4. Energy budget: strong acceleration should come with real heating
    if temp is None or accel is None or accel.get("gamma") is None:
        checks["energy_budget"] = "N/A"
    else:
        strong_accel = accel["gamma"] < 5.0            # flatter spectrum = stronger accel
        good_heating = temp["temp_MK"] > 8.0
        # inconsistent only if strong acceleration but almost no heating
        checks["energy_budget"] = "FLAG" if (strong_accel and not good_heating) else "PASS"

    # 5. QPP witness: soft & hard pulsation periods agree within 30%
    if qpp_soft is None or qpp_hard is None:
        checks["qpp_witness"] = "N/A"
    else:
        rel = abs(qpp_soft - qpp_hard) / max(qpp_soft, qpp_hard)
        checks["qpp_witness"] = "PASS" if rel <= QPP_AGREE_FRAC else "FLAG"

    # 6. Class vs acceleration: a bright flare should not have implausibly weak accel
    if accel is None or accel.get("gamma") is None:
        checks["class_accel"] = "N/A"
    else:
        bright = ev["ratio"] >= SIG_EVENT_RATIO
        very_weak_accel = accel["gamma"] > 7.0         # extremely steep = almost no accel
        checks["class_accel"] = "FLAG" if (bright and very_weak_accel) else "PASS"

    # 7. cross-band agreement: did independent energy bands corroborate
    # each other, or did only one band see anything despite a bright flare?
    if eband is None or eband.get("n_bands_covered", 0) == 0:
        checks["cross_band"] = "N/A"
    elif eband["n_bands_covered"] >= 2:
        checks["cross_band"] = "PASS"
    else:
        checks["cross_band"] = "FLAG" if ev["ratio"] >= SIG_EVENT_RATIO else "N/A"

    # 8. NEW -- independent per-band FOCuS confirmation: did another band's
    # OWN raw counts also show a genuine changepoint here (not just "had
    # some flux", which check 7 already covers) -- real corroborating
    # detection, not corroborating brightness.
    if band_confirm is None:
        checks["band_confirm"] = "N/A"
    else:
        checks["band_confirm"] = "PASS" if band_confirm["confirmed"] else "FLAG"

    # 9. NEW -- cross-band classification agreement: do the bands that DID
    # have coverage agree on how severe this was (same GOES tier), not just
    # which one happened to dominate.
    if cross_agree is None or cross_agree.get("agree") is None:
        checks["class_agree"] = "N/A"
    else:
        checks["class_agree"] = "PASS" if cross_agree["agree"] else "FLAG"

    # 10. NEW -- full spectral fit vs 2-band proxy: the forward-folded
    # isothermal Cash fit (fit_isothermal_spectrum) and the quick 2-band
    # ratio proxy should tell the same temperature story. Agreement within
    # a factor of 2 = the fast proxy is trustworthy for this event;
    # disagreement is a real physics red flag worth showing.
    if specfit is None or temp is None:
        checks["temp_fit_agree"] = "N/A"
    else:
        hi = max(specfit["T_MK_fit"], temp["temp_MK"])
        lo = max(min(specfit["T_MK_fit"], temp["temp_MK"]), 1e-6)
        checks["temp_fit_agree"] = "PASS" if hi / lo <= 2.0 else "FLAG"

    # 11. NEW -- independent tri-band vote: of the energy bands that had
    # data today, how many INDEPENDENTLY discovered this event with their
    # own full-day detector (not confirmation around a known event --
    # genuine side-by-side discovery)? Needs >= 2 covered bands to be a
    # meaningful vote at all.
    if band_vote is None or band_vote["votes_covered"] < 2:
        checks["band_vote"] = "N/A"
    elif band_vote["votes_yes"] >= 2:
        checks["band_vote"] = "PASS"
    else:
        # one lone yes on a bright flare that other covered bands ignored
        checks["band_vote"] = "FLAG" if ev["ratio"] >= SIG_EVENT_RATIO else "N/A"

    n_pass = sum(v == "PASS" for v in checks.values())
    n_flag = sum(v == "FLAG" for v in checks.values())
    n_appl = n_pass + n_flag                      # how many checks could actually run

    # ---- v4: noisy-OR confidence + detector-agreement evidence --------------
    evidence = [0.80 for _ in range(n_pass)]       # each PASS: independent evidence FOR

    focus_stat = ev.get("focus_stat", 0.0) or 0.0
    focus_margin = focus_stat / max(FOCUS_THRESHOLD, 1e-9)
    if focus_margin > 0:
        evidence.append(focus_margin / (focus_margin + 1.0))    # squashed into (0,1)

    fred = ev.get("fred_shape_stat", 0.0) or 0.0
    excess = max(ev["peak_flux"] - ev["bg"], 1e-6)
    if fred > 0:
        evidence.append(min(fred / excess, 1.0) * 0.6)

    if band_confirm is not None and band_confirm["confirmed"]:
        evidence.append(band_confirm["match_score"])       # independent per-band detection agreement

    # every INDEPENDENT band discovery beyond the first is genuinely separate
    # evidence -- separate detector, separate photons, separate full-day scan
    if band_vote is not None and band_vote["votes_yes"] >= 2:
        evidence.extend([0.5] * (band_vote["votes_yes"] - 1))

    base_conf = noisy_or(evidence)
    penalty = 1.0 - 0.5 * (n_flag / max(n_appl, 1))    # each flag pulls confidence down
    conf_pct = round(100.0 * base_conf * penalty, 1)

    raw = (n_pass - n_flag) / n_appl if n_appl > 0 else 0.0     # kept for display/comparison
    coverage = f"{n_pass}/{n_appl}" if n_appl > 0 else "0/0"
    return checks, n_pass, n_flag, round(raw, 3), conf_pct, coverage


# =============================================================================
#  SECTION 7  --  TOTAL RADIATED ENERGY (proxy)
# =============================================================================

def radiated_energy(bt, flux, ev):
    """
    Add up the flux ABOVE background across the whole event.
    Units: counts (a proxy for total radiated energy). This is the fallback
    used when real CALDB calibration isn't available -- see
    radiated_energy_calibrated() below for the real W/m^2 / J/m^2 version.
    """
    seg = flux[ev["s_idx"]:ev["e_idx"]]
    excess = np.maximum(seg - ev["bg"], 0.0)
    return float(np.sum(excess) * BIN_SEC)             # counts x seconds


def solexs_background_flux(pi_data, solexs_cal, t_start, lookback_s=120):
    """
    Real background irradiance (W/m^2) just before the flare: average the
    per-channel spectrum over the `lookback_s` seconds before t_start, then
    calibrate that one averaged spectrum. Returns 0.0 if there isn't enough
    quiet-time spectrum to average (e.g. flare sits right at a data gap).
    """
    pt, spec, chan = pi_data
    m = (pt >= t_start - lookback_s) & (pt < t_start)
    if m.sum() < 5:
        return 0.0
    mean_spec = spec[m].mean(axis=0)
    return solexs_cal.counts_to_flux(mean_spec, exposure_sec=1.0)


def radiated_energy_calibrated(pi_data, solexs_cal, ev):
    """
    THE REAL VERSION (uses the CALDB effective area from caldb_calibration.py).
    For every 1-second spectrum inside the flare, convert counts -> W/m^2
    using the real SoLEXS ARF/RMF, subtract a real quiet-Sun background
    irradiance, and integrate the excess over time.

    Returns dict(peak_flux_wm2, bg_flux_wm2, radiated_energy_jm2) or None if
    there's no spectrum coverage for this event (e.g. .pi file missing, or
    the event falls in a spectrum gap).

    Plain-language: peak_flux_wm2 is "how bright at the brightest second",
    like a single GOES reading. radiated_energy_jm2 is "how much total
    energy landed per square metre over the whole flare" -- brightness x
    duration, the real energy-budget number the confidence-score check 4
    (energy_budget) is really asking about.
    """
    if pi_data is None or solexs_cal is None:
        return None
    pt, spec, chan = pi_data
    m = (pt >= ev["start"]) & (pt <= ev["end"])
    if m.sum() < 1:
        return None

    bg_flux = solexs_background_flux(pi_data, solexs_cal, ev["start"])
    times = pt[m]
    fluxes = np.array([solexs_cal.counts_to_flux(row, exposure_sec=1.0) for row in spec[m]])
    fluxes = np.nan_to_num(fluxes, nan=0.0)

    excess = np.maximum(fluxes - bg_flux, 0.0)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    dt = dt if dt > 0 else 1.0

    return dict(
        peak_flux_wm2=float(np.nanmax(fluxes)),
        bg_flux_wm2=float(bg_flux),
        radiated_energy_jm2=float(np.sum(excess) * dt),
    )


CANONICAL_BANDS = {
    "band1": (2.8, 20.0),   # SoLEXS's reliable range (2.8 keV floor -- see BAND_A note above)
    "band2": (20.0, 60.0),
    "band3": (60.0, 150.0),
}


def _peak_flux_in_window(t, spec, calibrator, t_start, t_end, elo_kev, ehi_kev, default_exposure=1.0):
    """
    Real peak W/m^2 in [elo_kev, ehi_kev] during [t_start, t_end], from a
    (times, spectra, [exposures]) tuple. Returns None if no rows fall inside
    the window. Used by classify_energy_band() -- one call per canonical band
    per available detector.
    """
    if t is None or calibrator is None:
        return None
    m = (t >= t_start) & (t <= t_end)
    if m.sum() < 1:
        return None
    fluxes = calibrator.counts_to_flux_series(spec[m], exposure_sec=default_exposure,
                                              elo_kev=elo_kev, ehi_kev=ehi_kev)
    fluxes = np.nan_to_num(fluxes, nan=0.0)
    return float(np.nanmax(fluxes)) if len(fluxes) else None


def raw_band_counts_series(t, spec, calibrator, elo_kev, ehi_kev):
    """
    Sum RAW counts (not flux -- no area division) within [elo_kev, ehi_kev]
    for every row of a HEL1OS spectra array, using the calibrator's own
    channel-energy map (chan_emid) so the band edges match exactly what the
    flux calibration uses for the same range. This is the input FOCuS needs
    -- FOCuS is built for actual Poisson counts, not already-calibrated flux
    (dividing by area breaks the Poisson assumption FOCuS's math relies on).
    Returns (times, counts_per_row).
    """
    emid = calibrator.chan_emid
    n = min(spec.shape[1], len(emid))
    mask = (emid[:n] >= elo_kev) & (emid[:n] < ehi_kev)
    if not mask.any():
        return t, np.zeros(spec.shape[0])
    return t, spec[:, :n][:, mask].sum(axis=1)


def band_focus_confirmation(t, counts, ev, exposure_sec=20.0, threshold=FOCUS_THRESHOLD):
    """
    Independent per-band detection confirmation (the real substance of
    "per-band detection + merge", scoped to fit our existing one-master-
    catalog architecture rather than rebuilding it around three parallel
    catalogs): run the SAME Poisson-FOCuS math on this band's own raw
    counts, restricted to a window around the already-detected event, and
    check whether an INDEPENDENT excursion shows up here too.

    This mirrors the reference pipeline's MATCH_SCORE idea (temporal
    proximity of an independent per-band detection) without requiring a
    full separate catalog-merge pipeline: the "match" is against our
    already-confirmed master event, and the score blends timing overlap
    with how strong the confirming band's own FOCuS statistic was.

    Returns dict(confirmed: bool, match_score: float in [0,1], focus_stat: float)
    or None if there's no data in this band for this event's window.
    """
    lookback_s = 600.0
    m = (t >= ev["start"] - lookback_s) & (t <= ev["end"] + lookback_s)
    if m.sum() < 5:
        return None
    tt = t[m]
    cc = np.maximum(counts[m], 0.0)

    # background = median rate over this same slice (same "true typical rate"
    # reasoning as the primary detector's bg_mu0 -- see detect_events docstring)
    bg_rate = np.median(cc) if len(cc) else 0.0
    bg_counts = np.full(len(cc), max(bg_rate, 1e-6))

    stat, _ = poisson_focus_scan(cc, bg_counts, threshold=threshold)
    peak_i = int(np.argmax(stat))
    peak_stat = float(stat[peak_i])
    peak_t = float(tt[peak_i])

    confirmed = peak_stat > threshold
    # timing proximity to the master event's own peak, asymmetric like the
    # reference's association window isn't needed here (same instrument
    # family, no cross-spacecraft lead/lag) -- just "close to this event".
    dt = abs(peak_t - ev["peak"])
    window = max(ev["end"] - ev["start"], BIN_SEC * 6)
    proximity = max(0.0, 1.0 - dt / window)
    strength = min(peak_stat / threshold, 3.0) / 3.0 if threshold > 0 else 0.0
    match_score = 0.6 * proximity + 0.4 * strength if confirmed else 0.0

    return dict(confirmed=confirmed, match_score=float(match_score), focus_stat=peak_stat)


def detect_band_events(t, counts_per_sample, min_dur_s=60.0, merge_gap_s=MERGE_GAP_S,
                       bg_window_s=1200.0, threshold=FOCUS_THRESHOLD,
                       rise_ratio=MIN_RISE_RATIO):
    """
    A FULL standalone flare detector for one energy band's own raw counts,
    over the WHOLE day (limitation fix #1). This is the same architecture as
    the master detect_events() -- rolling-median background, Poisson-FOCuS
    trigger, minimum duration, gap merging, noise-blip rise gate -- but
    cadence-agnostic, so it runs identically on SoLEXS 1-second spectra and
    HEL1OS 20-second spectra. Unlike band_focus_confirmation() (which only
    peeks around an ALREADY-found event), this discovers events on its own
    with no knowledge of what any other band saw.

    t                 : sample times (unix seconds), any cadence
    counts_per_sample : RAW counts in this band per sample (Poisson data)

    Returns a list of dicts: {start, peak, end, focus_stat, peak_rate, bg_rate}.
    """
    t = np.asarray(t, dtype=float)
    c = np.maximum(np.asarray(counts_per_sample, dtype=float), 0.0)
    if len(t) < 10:
        return []
    order = np.argsort(t)
    t, c = t[order], c[order]

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return []

    # rebin to ~10 s so all bands are judged on the same footing as the master
    bin_s = max(BIN_SEC, dt)
    edges = np.floor((t - t[0]) / bin_s).astype(int)
    n_bins = edges.max() + 1
    binned = np.bincount(edges, weights=c, minlength=n_bins)
    bt = t[0] + (np.arange(n_bins) + 0.5) * bin_s
    keep = np.bincount(edges, minlength=n_bins) > 0        # bins with real samples
    bt, binned = bt[keep], binned[keep]
    if len(bt) < 10:
        return []

    win = max(int(bg_window_s / bin_s), 6)
    s = pd.Series(binned)
    bg = s.rolling(win, center=True, min_periods=max(win // 4, 2)).median()
    bg = bg.ffill().bfill().values
    bg = np.maximum(bg, 1e-6)

    stat, _ = poisson_focus_scan(binned, bg, threshold=threshold)
    above = stat > threshold

    min_dur_bins = max(int(min_dur_s / bin_s), 2)
    raw, in_evt, s_idx = [], False, 0
    for i in range(len(above)):
        if above[i] and not in_evt:
            in_evt, s_idx = True, i
        elif not above[i] and in_evt:
            if i - s_idx >= min_dur_bins:
                raw.append((s_idx, i))
            in_evt = False
    if in_evt and len(above) - s_idx >= min_dur_bins:
        raw.append((s_idx, len(above) - 1))

    gap_bins = max(int(merge_gap_s / bin_s), 1)
    merged = []
    for seg in raw:
        if merged and seg[0] - merged[-1][1] <= gap_bins:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    out = []
    for a, b in merged:
        pk = a + int(np.argmax(binned[a:b])) if b > a else a
        quiet = max(float(bg[a]), 1e-6)
        if float(binned[pk]) < rise_ratio * quiet:          # same noise-blip gate
            continue
        out.append(dict(start=float(bt[a]), peak=float(bt[pk]), end=float(bt[b]),
                        focus_stat=float(np.max(stat[a:b])) if b > a else float(stat[a]),
                        peak_rate=float(binned[pk] / bin_s),
                        bg_rate=float(bg[pk] / bin_s)))
    return out


def triband_independent_vote(mode, pi_data, helos_spec_cdte, helos_spec_czt):
    """
    THREE INDEPENDENT DETECTORS, SIDE BY SIDE (limitation fix #1).

    Runs detect_band_events() separately on each canonical band's own raw
    counts for the FULL day -- none of the three sees the others' data or
    the master detector's output:

        band1 (2.8-20 keV)  : SoLEXS spectrum counts (CDTE fallback in
                              HEL1OS-only mode, from 9.5-20 keV)
        band2 (20-60 keV)   : CZT preferred, CDTE fallback
        band3 (60-150 keV)  : CZT only

    Returns dict:
        catalogs : {band: [events]} for every band that had data ("covered")
        covered  : list of band names with data
    Use vote_for_event() afterwards to count, for one master event, how many
    of these independent detectors discovered it too.
    """
    catalogs, covered = {}, []

    # band1: SoLEXS raw counts summed 2.8-20 keV per 1-s spectrum row
    if mode != "HEL1OS" and pi_data is not None:
        pt, spec, chan = pi_data
        e = ch_to_keV(chan)
        n = min(spec.shape[1], len(e))
        msk = (e[:n] >= 2.8) & (e[:n] < 20.0)
        if msk.any():
            catalogs["band1"] = detect_band_events(pt, spec[:, :n][:, msk].sum(axis=1))
            covered.append("band1")
    elif mode == "HEL1OS" and helos_spec_cdte is not None and HEL1OS_CAL is not None:
        t_cd, spec_cd, _ = helos_spec_cdte
        lo = HEL1OS_CAL["cdte"].recommended_elo_kev
        _, cc = raw_band_counts_series(t_cd, spec_cd, HEL1OS_CAL["cdte"], lo, 20.0)
        catalogs["band1"] = detect_band_events(t_cd, cc)
        covered.append("band1")

    if HEL1OS_CAL is not None:
        src = None
        if helos_spec_czt is not None:
            src = (helos_spec_czt, HEL1OS_CAL["czt"])
        elif helos_spec_cdte is not None:
            src = (helos_spec_cdte, HEL1OS_CAL["cdte"])
        if src is not None:
            (t_h, spec_h, _), cal_h = src
            _, cc2 = raw_band_counts_series(t_h, spec_h, cal_h, *CANONICAL_BANDS["band2"])
            catalogs["band2"] = detect_band_events(t_h, cc2)
            covered.append("band2")
        if helos_spec_czt is not None:
            t_z, spec_z, _ = helos_spec_czt
            _, cc3 = raw_band_counts_series(t_z, spec_z, HEL1OS_CAL["czt"], *CANONICAL_BANDS["band3"])
            catalogs["band3"] = detect_band_events(t_z, cc3)
            covered.append("band3")

    return dict(catalogs=catalogs, covered=covered)


def vote_for_event(ev, triband, pad_s=120.0):
    """
    The VOTE: for one master event, ask each independent band detector
    "did YOU also discover an event overlapping this time window?" (with a
    small pad, since a hard X-ray burst legitimately leads the soft peak).

    Returns dict:
        votes_yes / votes_covered : e.g. 2 of 3
        voting_bands              : ["band1", "band2"]
        vote_str                  : "2/3"  (for the CSV / report table)
    """
    yes = []
    for band in triband["covered"]:
        for be in triband["catalogs"].get(band, []):
            if be["start"] - pad_s <= ev["end"] and be["end"] + pad_s >= ev["start"]:
                yes.append(band)
                break
    n_cov = len(triband["covered"])
    return dict(votes_yes=len(yes), votes_covered=n_cov,
                voting_bands=yes,
                vote_str=f"{len(yes)}/{n_cov}" if n_cov else "0/0")


def flux_uncertainty(counts, calibrator, exposure_sec, elo_kev, ehi_kev=None):
    """
    Real 1-sigma uncertainty on a calibrated flux value from Poisson counting
    statistics (item: real error bars, not just point numbers).

    Same weight derivation as counts_to_flux_series (width cancels; weight[ch]
    = E[ch]*KEV_TO_JOULE/area[ch]) -- flux = sum(counts * weight) / exposure.
    For independent Poisson counts, Var(N_i) = N_i, so:

        Var(flux) = sum(N_i * weight_i^2) / exposure^2
        sigma_flux = sqrt(Var(flux)) / CM2_TO_M2

    Returns sigma in W/m^2, or None if the calibrator lacks the needed arrays.
    """
    if calibrator is None:
        return None
    try:
        emid = calibrator.chan_emid
        area = calibrator.eff_area_cm2
        n = min(len(counts), len(emid))
        mask = (emid[:n] >= elo_kev) & (emid[:n] <= (ehi_kev if ehi_kev else emid.max())) & (area[:n] > 0)
        weight = np.where(mask, (emid[:n] * KEV_TO_JOULE_CONST) / area[:n], 0.0)
        weight = np.nan_to_num(weight, nan=0.0, posinf=0.0)
        counts_arr = np.maximum(np.asarray(counts[:n], dtype=float), 0.0)
        var_flux = np.sum(counts_arr * weight ** 2) / (exposure_sec ** 2)
        sigma = math.sqrt(max(var_flux, 0.0)) / CM2_TO_M2_CONST
        return float(sigma)
    except Exception:
        return None


def classify_cross_band_agreement(eband):
    """
    Item: cross-band classification agreement. classify_energy_band() already
    tells us which band DOMINATES; this checks whether the bands that DID have
    coverage actually AGREE on how severe the flare was (same GOES tier), not
    just which one happened to be biggest. A flare where band1 says "X-class"
    and band2 says "quiet" is a genuinely different, more suspect situation
    than one where both bands agree it was big.

    Returns dict(agree: bool or None, tiers: {band: letter}) -- None if fewer
    than 2 bands had real coverage (nothing to agree or disagree about).
    """
    if eband is None or eband.get("n_bands_covered", 0) < 2:
        return dict(agree=None, tiers={})
    tiers = {}
    for key, wm2 in (("band1", eband.get("band1_wm2")), ("band2", eband.get("band2_wm2")),
                     ("band3", eband.get("band3_wm2"))):
        if wm2 is not None and wm2 > 0:
            cls = goes_class(wm2)
            if cls and cls[0] in "ABCMX":     # skip the "<A1" sub-threshold edge case
                tiers[key] = cls[0]     # letter tier only (A/B/C/M/X), ignore mantissa
    if len(tiers) < 2:
        return dict(agree=None, tiers=tiers)
    letters = set(tiers.values())
    ladder = "ABCMX"
    max_gap = max(ladder.index(a) - ladder.index(b) for a in letters for b in letters)
    return dict(agree=(max_gap <= 1), tiers=tiers, max_tier_gap=max_gap)



def classify_energy_band(pi_data, helos_spec_cdte, helos_spec_czt, ev):
    """
    Real peak W/m^2 in each of the three canonical bands during this event,
    using whichever real spectra actually cover that band:
      band1  2.8-20 keV   : SoLEXS (SOLEXS_CAL)
      band2  20-60 keV    : CZT preferred (its native range), CDTE fallback
      band3  60-150 keV   : CZT only -- CDTE's response doesn't extend this
                             high (HEL1OS manual: CdTe ~8-70 keV, CZT ~20-150 keV)
    dominant_band = whichever of the three has the highest peak flux (all
    three are now real physical W/m^2, so they're directly comparable).
    Returns dict with all three values (None where no coverage) + a label.
    """
    b1 = b2 = b3 = None
    b2_source = b3_source = None

    if pi_data is not None and SOLEXS_CAL is not None:
        pt, spec, chan = pi_data
        lo, hi = CANONICAL_BANDS["band1"]
        b1 = _peak_flux_in_window(pt, spec, SOLEXS_CAL, ev["start"], ev["end"], lo, hi, default_exposure=1.0)

    if HEL1OS_CAL is not None:
        lo, hi = CANONICAL_BANDS["band2"]
        if helos_spec_czt is not None:
            t, spec, exp = helos_spec_czt
            b2 = _peak_flux_in_window(t, spec, HEL1OS_CAL["czt"], ev["start"], ev["end"], lo, hi,
                                      default_exposure=np.median(exp) if len(exp) else 20.0)
            b2_source = "CZT" if b2 is not None else None
        if b2 is None and helos_spec_cdte is not None:
            t, spec, exp = helos_spec_cdte
            b2 = _peak_flux_in_window(t, spec, HEL1OS_CAL["cdte"], ev["start"], ev["end"], lo, hi,
                                      default_exposure=np.median(exp) if len(exp) else 20.0)
            b2_source = "CDTE" if b2 is not None else None

        lo, hi = CANONICAL_BANDS["band3"]
        if helos_spec_czt is not None:
            t, spec, exp = helos_spec_czt
            b3 = _peak_flux_in_window(t, spec, HEL1OS_CAL["czt"], ev["start"], ev["end"], lo, hi,
                                      default_exposure=np.median(exp) if len(exp) else 20.0)
            b3_source = "CZT" if b3 is not None else None

    candidates = [("band1 (2.8-20 keV, soft)", b1), ("band2 (20-60 keV, medium)", b2),
                 ("band3 (60-150 keV, hard)", b3)]
    valid = [(name, v) for name, v in candidates if v is not None and v > 0]
    dominant = max(valid, key=lambda x: x[1])[0] if valid else None

    return dict(band1_wm2=b1, band2_wm2=b2, band3_wm2=b3,
               band2_source=b2_source, band3_source=b3_source, dominant_band=dominant,
               n_bands_covered=len(valid))


def hard_flux_calibrated(helos_spec, calibrator, ev, lookback_s=600):
    """
    HEL1OS version of the calibrated flux: convert each 20-second hard X-ray
    spectrum inside the flare to W/m^2 (CdTe >=9.5 keV or CZT >=35 keV floor
    is applied inside the calibrator), subtract a real pre-flare background,
    and integrate the excess.

    helos_spec : (times_unix, spectra[N,ch], exposures[N]) from
                 read_helos_spectra(), or None.
    calibrator : Hel1osCalibrator("cdte") or ("czt").

    Returns dict(hard_peak_flux_wm2, hard_bg_flux_wm2, hard_energy_jm2) or
    None if this event has no spectra coverage. lookback is 10 min (not 2)
    because at 20 s cadence, 2 min = only 6 spectra -- too noisy a background.
    """
    if helos_spec is None or calibrator is None:
        return None
    ht, hspec, hexp = helos_spec
    m = (ht >= ev["start"]) & (ht <= ev["end"])
    if m.sum() < 1:
        return None

    # real quiet background: mean spectrum over the 10 min before the flare
    bm = (ht >= ev["start"] - lookback_s) & (ht < ev["start"])
    if bm.sum() >= 3:
        bg_spec = hspec[bm].mean(axis=0)
        bg_exp = float(np.median(hexp[bm]))
        bg_flux = calibrator.counts_to_flux(bg_spec, exposure_sec=bg_exp)
    else:
        bg_flux = 0.0

    times = ht[m]
    fluxes = np.array([calibrator.counts_to_flux(row, exposure_sec=e)
                       for row, e in zip(hspec[m], hexp[m])])
    fluxes = np.nan_to_num(fluxes, nan=0.0)

    excess = np.maximum(fluxes - bg_flux, 0.0)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 20.0
    dt = dt if dt > 0 else 20.0

    # REAL error bar on the peak hard flux (limitation fix #2): same Poisson
    # propagation flux_uncertainty() already does for the soft flux, applied
    # to the spectrum row the peak was read from, over the calibrator's own
    # valid range (its recommended low-energy floor upward).
    pk_i = int(np.nanargmax(fluxes))
    pk_exp = float(hexp[m][pk_i]) if hexp[m][pk_i] > 0 else 20.0
    hard_peak_flux_err = flux_uncertainty(
        hspec[m][pk_i], calibrator, pk_exp,
        elo_kev=getattr(calibrator, "recommended_elo_kev", 9.5))

    return dict(
        hard_peak_flux_wm2=float(np.nanmax(fluxes)),
        hard_peak_flux_wm2_err=hard_peak_flux_err,
        hard_bg_flux_wm2=float(bg_flux),
        hard_energy_jm2=float(np.sum(excess) * dt),
    )


_GOES_THRESHOLDS = [('X', 1e-4), ('M', 1e-5), ('C', 1e-6), ('B', 1e-7), ('A', 1e-8)]

def goes_class(flux_wm2, prefix=""):
    """
    REAL flare class from calibrated irradiance (W/m^2) -- the standard
    NOAA/GOES letter+number scale: X >= 1e-4, M >= 1e-5, C >= 1e-6,
    B >= 1e-7, A >= 1e-8 W/m^2; the number after the letter is
    flux / class-floor (so 3.5e-5 W/m^2 = "M3.5").

    prefix="HXR-" is used for HEL1OS-only (hard X-ray, 8-150 keV) events so
    the label can never be mistaken for a real GOES entry -- GOES classes
    are defined on 1-8 Angstrom (~1.5-12 keV) SOFT X-ray only. A SoLEXS
    (2.8-22 keV) class will be CLOSE to a same-flare GOES entry but is not
    a guaranteed match; state the instrument, don't claim GOES-equivalence.

    Returns None if flux_wm2 is None/non-positive (caller should fall back
    to approx_class()).
    """
    if flux_wm2 is None or flux_wm2 <= 0:
        return None
    for ltr, lo in _GOES_THRESHOLDS:
        if flux_wm2 >= lo:
            return f"{prefix}{ltr}{flux_wm2/lo:.1f}"
    return f"{prefix}<A1"


def utc_formatter(t0_unix):
    """
    matplotlib FuncFormatter: x = hours since t0_unix -> 'HH:MM' UTC label.
    Use with ax.xaxis.set_major_formatter(utc_formatter(t0)).
    """
    def fmt(x, pos):
        return unix_to_utc(t0_unix + x * 3600.0)[11:16]
    return mticker.FuncFormatter(fmt)


def utc_formatter_minutes(t0_unix):
    """
    Same as utc_formatter() but for axes in MINUTES since t0_unix (QPP
    detection, coronal heating -- both plot against minutes, not hours).
    """
    def fmt(x, pos):
        return unix_to_utc(t0_unix + x * 60.0)[11:16]
    return mticker.FuncFormatter(fmt)


def _log_safe(arr, floor=1e-12):
    """Floor an array to a tiny positive value so log-scale plotting doesn't
    choke on exact zeros (log(0) = -inf). Display-only; never mutates data
    used for detection or catalog numbers."""
    return np.maximum(np.nan_to_num(np.asarray(arr, dtype=float), nan=floor), floor)


def approx_class(ratio, peak_rate):
    """
    A RELATIVE strength label (not an absolute GOES class).
    v2: uses BOTH the ratio (peak/background) and the absolute count rate.
    Why: a flare detected on the decaying tail of a bigger flare sits on an
    inflated background, so its ratio looks tiny (1.1x) even when its absolute
    rate is huge (7900 counts/s). Whichever measure is stronger wins.
    Thresholds come from the real catalog distribution (95 flares, 12 days).
    """
    by_ratio = ("X-like" if ratio >= 50 else
                "M-like" if ratio >= 20 else
                "C-like" if ratio >= 8  else "sub-C")
    by_rate  = ("X-like" if peak_rate >= 5000 else
                "M-like" if peak_rate >= 1500 else
                "C-like" if peak_rate >= 400  else "sub-C")
    order = ["sub-C", "C-like", "M-like", "X-like"]
    return by_ratio if order.index(by_ratio) >= order.index(by_rate) else by_rate
