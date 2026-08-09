from .common import *
from .data import *
from .features import *

# =============================================================================
#  SECTION 4 -- WINDOWING + LABELING (the three fixes live here)
# =============================================================================

def _compute_event_energies(events, pi_data):
    """
    Real radiated energy (J/m^2) per event, using the SAME real ARF-based
    calibration nowcast uses -- None where calibration/spectrum coverage
    isn't available for that event. Used to upgrade FIX 3 (below) beyond a
    pure peak-ratio criterion: a flare that stays moderately bright for 20
    minutes can release more real energy than one that spikes briefly, and
    peak ratio alone can't tell the difference.
    """
    energies = []
    for e in events:
        cal = None
        if pi_data is not None and ncp.SOLEXS_CAL is not None:
            try:
                cal = ncp.radiated_energy_calibrated(pi_data, ncp.SOLEXS_CAL, e)
            except Exception:
                cal = None
        energies.append(cal["radiated_energy_jm2"] if cal else None)
    return energies


def build_windows_for_date(date, day):
    """
    Slide a window across one date's data. Returns a list of dicts:
      {date, window_end_unix, window_end_h, label, features, flare_start (if label=1)}
    """
    bt, flux, bg = day["bt"], day["flux"], day["bg"]
    events = day["events"]
    if len(bt) < 10:
        return [], 0

    bin_sec = float(bt[1] - bt[0])
    lookback_bins = int(LOOKBACK_MIN * 60 / bin_sec)
    lead_bins = int(LEAD_MIN * 60 / bin_sec)
    step_bins = max(int(STEP_MIN * 60 / bin_sec), 1)

    flux_n_full = flux / bg   # FIX 2: background-normalized, once per day (fast)

    bandA_n = day["bandA"] / bg if day["bandA"] is not None else None
    bandB_n = day["bandB"] / bg if day["bandB"] is not None else None
    bandC_n = day["bandC"] / bg if day["bandC"] is not None else None

    helos_n_full = None
    if day["helos_bt"] is not None:
        # align HEL1OS onto the SoLEXS bt grid (nearest-bin), normalize by its own background
        h_bg = rolling_background(day["helos_flux"])
        h_norm = day["helos_flux"] / h_bg
        helos_n_full = np.interp(bt, day["helos_bt"], h_norm, left=np.nan, right=np.nan)

    # significant flares (FIX 3): peak ratio OR real radiated energy in the
    # top 30% of this date's own events -- percentile-based, not a guessed
    # absolute joule cutoff (the exact trap we called out in the reference
    # pipeline's 5e-9 constant). Falls back to ratio-only when calibration
    # isn't available for this date (e.g. no .pi spectrum file).
    event_energies = _compute_event_energies(events, day.get("pi_data"))
    valid_energies = [en for en in event_energies if en is not None]
    energy_threshold = float(np.percentile(valid_energies, 70)) if len(valid_energies) >= 3 else None

    sig_events = []
    for e, en in zip(events, event_energies):
        is_sig = e["ratio"] >= SIGNIFICANT_RATIO
        # FIX 3 upgrade WITH FLOOR: real energy can promote a BORDERLINE event
        # (ratio >= ENERGY_UPGRADE_MIN_RATIO) that the peak-ratio test just
        # missed -- but a per-day energy percentile alone must never promote
        # a microflare on a quiet day (every day has a "top 30%", even a day
        # of pure blips; that's relative, not significant).
        if (not is_sig and en is not None and energy_threshold is not None
                and en >= energy_threshold and e["ratio"] >= ENERGY_UPGRADE_MIN_RATIO):
            is_sig = True
        if is_sig:
            sig_events.append(e)
    sig_events.sort(key=lambda e: e["end"])   # chronological, needed for the clustering loop below

    windows = []
    # -- LIMITATION FIX (thin positives): the coarse STEP_MIN grid gives each
    # flare only ~LEAD_MIN/STEP_MIN positive windows. Positives are the
    # scarce, precious class (~200 total across all dates), so ADD a fine
    # 1-minute grid of window ends restricted to genuine pre-flare periods
    # (window ends whose next LEAD_MIN minutes contain a significant flare
    # start). Every added window is a REAL, distinct moment of real data --
    # nothing synthetic, nothing duplicated -- it's simply sampling the rare
    # class more densely than the common one. Negatives keep the coarse grid.
    end_indices = set(range(lookback_bins, len(bt) - 1, step_bins))
    coarse_grid = set(end_indices)      # remember which ends are the natural grid
    fine_step = max(int(DENSE_POSITIVE_STRIDE_MIN * 60 / bin_sec), 1)
    for e in sig_events:
        i0 = int(np.searchsorted(bt, e["start"] - LEAD_MIN * 60))
        i1 = int(np.searchsorted(bt, e["start"]))
        for idx in range(max(i0, lookback_bins), min(i1, len(bt) - 1), fine_step):
            end_indices.add(idx)

    for end_idx in sorted(end_indices):
        w_start_idx = end_idx - lookback_bins
        t_end = bt[end_idx]

        # FIX 1: decay-phase exclusion -- skip only while flux is GENUINELY
        # still elevated after a SIGNIFICANT prior flare's end, not just
        # because the clock hasn't hit DECAY_EXCLUDE_MIN yet. A fixed
        # 90-minute blanket version was found (on real data) to strip out
        # exactly the "recent flare -> another one soon" windows that would
        # teach the model flare clustering -- once flux has genuinely
        # settled back near background, that window is a legitimate
        # training example again, even if it's only 30 minutes after the
        # last flare's end. DECAY_EXCLUDE_FLOOR_MIN still guarantees a
        # minimum exclusion (flux can dip transiently right after a peak
        # without the decay really being over).
        in_decay = False
        current_flux_n = flux_n_full[end_idx] if end_idx < len(flux_n_full) else 0.0
        for e in sig_events:
            dt = t_end - e["end"]
            if not (0 <= dt <= DECAY_EXCLUDE_MIN * 60):
                continue
            if dt <= DECAY_EXCLUDE_FLOOR_MIN * 60 or current_flux_n >= DECAY_FLUX_THRESHOLD:
                in_decay = True
                break
        if in_decay:
            continue

        seg = flux_n_full[w_start_idx:end_idx]
        coverage = np.isfinite(seg).mean() if len(seg) else 0.0
        if coverage < MIN_WINDOW_COVERAGE:
            continue

        # label: does a SIGNIFICANT flare start within the next LEAD_MIN minutes?
        horizon_end = t_end + LEAD_MIN * 60
        upcoming = [e for e in sig_events if t_end < e["start"] <= horizon_end]
        label = 1 if upcoming else 0
        flare_start = min(e["start"] for e in upcoming) if upcoming else None

        bA = bandA_n[w_start_idx:end_idx] if bandA_n is not None else None
        bB = bandB_n[w_start_idx:end_idx] if bandB_n is not None else None
        bC = bandC_n[w_start_idx:end_idx] if bandC_n is not None else None
        hN = helos_n_full[w_start_idx:end_idx] if helos_n_full is not None else None

        # -- event-clustering features: look BACKWARD at flare history, not
        # just inside this window (every other feature only sees the window) --
        prior = [e for e in sig_events if e["end"] <= t_end]
        if prior:
            last_end = prior[-1]["end"]
            time_since_min = (t_end - last_end) / 60.0
            decayed_hist = sum(
                math.exp(-(t_end - e["end"]) / (HISTORY_DECAY_TAU_MIN * 60.0)) for e in prior
            )
        else:
            time_since_min = None
            decayed_hist = 0.0

        data_gap_frac = 1.0 - coverage

        # -- real calibrated temperature trend on just this window (not a
        # whole flare -- measure_temperature works on any time slice) --
        temp_trend = 0.0
        pi_data = day.get("pi_data")
        if pi_data is not None and ncp.SOLEXS_CAL is not None:
            try:
                tp = ncp.measure_temperature(pi_data, bt[w_start_idx], t_end, t_end,
                                             calibrator=ncp.SOLEXS_CAL)
                if tp is not None and len(tp.get("curve_t", [])) >= 3:
                    ct_min = (tp["curve_t"] - tp["curve_t"][0]) / 60.0
                    temp_trend = _slope(tp["curve_T_MK"]) if len(tp["curve_T_MK"]) < 2 else \
                        float(np.polyfit(ct_min, tp["curve_T_MK"], 1)[0])
            except Exception:
                temp_trend = 0.0

        # -- per-window gamma clues (LIMITATION FIX: acceleration steepness
        # as a forecasting input). gamma_window = slope over the whole
        # lookback; gamma_trend = late-half minus early-half gamma, so a
        # NEGATIVE trend means the spectrum is hardening (pre-flare sign).
        gamma_window, gamma_trend = 0.0, 0.0
        if day.get("helos_bands"):
            t_w0 = bt[w_start_idx]
            g_full = _window_gamma(day["helos_bands"], t_w0, t_end)
            if g_full is not None:
                gamma_window = g_full
                t_mid = 0.5 * (t_w0 + t_end)
                g_early = _window_gamma(day["helos_bands"], t_w0, t_mid)
                g_late = _window_gamma(day["helos_bands"], t_mid, t_end)
                if g_early is not None and g_late is not None:
                    gamma_trend = g_late - g_early

        feats = extract_features(seg, bA, bB, bC, hN,
                                 time_since_last_flare_min=time_since_min,
                                 decayed_flare_history=decayed_hist,
                                 data_gap_frac=data_gap_frac,
                                 temp_trend_mk_per_min=temp_trend,
                                 gamma_window=gamma_window,
                                 gamma_trend=gamma_trend)

        windows.append(dict(
            date=date, window_end_unix=t_end,
            window_end_h=(t_end - bt[0]) / 3600.0,
            label=label, flare_start=flare_start,
            dense=0 if end_idx in coarse_grid else 1,   # 1 = extra fine-grid window
            features=feats,
        ))
    return windows, len(sig_events)


# =============================================================================
#  SECTION 5 -- DATASET ASSEMBLY ACROSS ALL DATES
# =============================================================================

def build_dataset():
    dates = ncp.find_solexs_dates(ncp.SOLEXS_FOLDER)
    if not dates:
        print("[ERROR] No SoLEXS dates found. Check SOLEXS_FOLDER in nowcast_physics.py.")
        sys.exit(1)

    all_windows = []
    print(f"Building forecast dataset from {len(dates)} date(s)...")
    for date, sol_paths in sorted(dates.items()):
        helos_paths = ncp.find_helos_for_date(ncp.HELOS_FOLDER, date)
        try:
            day = assemble_day(date, sol_paths, helos_paths)
        except Exception as e:
            print(f"  [warn] {date}: failed to assemble ({e}), skipping")
            continue
        wins, n_sig_labeled = build_windows_for_date(date, day)
        n_pos = sum(w["label"] for w in wins)
        print(f"  {date}: {len(wins)} windows, {n_pos} positive "
              f"({len(day['events'])} raw flares, "
              f"{n_sig_labeled} labeled significant "
              f"[{sum(1 for e in day['events'] if e['ratio'] >= SIGNIFICANT_RATIO)} by ratio], "
              f"HEL1OS {'yes' if day['helos_bt'] is not None else 'no'})")
        all_windows.extend(wins)

    if not all_windows:
        print("[ERROR] No usable windows built. Check data coverage.")
        sys.exit(1)

    df = pd.DataFrame({
        "date": [w["date"] for w in all_windows],
        "window_end_unix": [w["window_end_unix"] for w in all_windows],
        "window_end_h": [w["window_end_h"] for w in all_windows],
        "label": [w["label"] for w in all_windows],
        "flare_start": [w["flare_start"] for w in all_windows],
        "dense": [w["dense"] for w in all_windows],
    })
    X = np.vstack([w["features"] for w in all_windows])
    for i, name in enumerate(FEATURE_NAMES):
        df[name] = X[:, i]

    print(f"\nTotal windows: {len(df)}  |  positive: {df['label'].sum()} "
          f"({100*df['label'].mean():.1f}%)")
    return df
