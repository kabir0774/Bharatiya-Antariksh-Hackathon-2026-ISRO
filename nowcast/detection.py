from .common import *

# =============================================================================
#  SECTION 4  --  FLARE DETECTION  (same algorithm as the main pipeline)
# =============================================================================

def poisson_focus_scan(counts, bg_counts, threshold=FOCUS_THRESHOLD, max_curves=FOCUS_MAX_CURVES,
                        max_window_bins=FOCUS_MAX_WINDOW_BINS):
    """
    Poisson-FOCuS changepoint statistic (Ward et al. 2023 -- built for onboard
    GRB/CubeSat triggers, the correct math for low-count "clicks" data).
    Offline/batch version of the online algorithm: one pass over the whole
    array, same math, since our pipeline processes a full day at a time
    rather than truly streaming.

    THE ACTUAL FIX for over-detection: our old test compared each bin to ONE
    fixed 20-minute background window. A burst lasting 20 seconds and a rise
    lasting 20 minutes get the same yardstick, so the detector either misses
    short real bursts or false-fires on slow drift. FOCuS instead asks, at
    every bin: "is there ANY recent stretch -- 2 bins back, 50 bins back,
    500 bins back -- whose average rate is a statistically significant excess
    over its own local background?" It tests every window length at once via
    a cheap trick (keep only the lower convex hull of cumulative counts vs
    time; a window length that can never win is provably prunable), so this
    stays fast despite "checking everything."

    TWO fixes found only by testing this against synthetic data, both real:

    1. RESET ON EXIT: once the statistic crosses the threshold and then falls
       back below it, the hull is reset to "now" -- otherwise one big burst
       early in the day keeps dominating the convex hull for the rest of the
       day and masks smaller later flares. Their reference does the same
       thing -- explicitly documented as "collapse the candidate list back to
       the current state ... so accumulated evidence does not bleed into the
       next event."
    2. MAX WINDOW CAP: even with the reset above, a single very strong burst
       can leave a hull candidate whose LLR takes literally HOURS to decay
       back below threshold on its own, even once the true rate is back to
       normal -- because the statistic is (correctly) detecting that the
       long-run average since that old changepoint is STILL slightly above
       background, which becomes statistically "significant" again once the
       segment is long enough, even though physically nothing is happening.
       Their reference avoids this because an independent signal (their
       soft-band FSM's own peak/end rule) tells FOCuS exactly when to reset,
       not FOCuS's own statistic. We don't have that independent signal here,
       so instead we just cap how far back any candidate window is allowed to
       reach (FOCUS_MAX_WINDOW_BINS, 1 hour) -- no real flare in our data
       runs anywhere near that long, so this costs nothing real and closes
       the runaway.

    counts    : per-bin COUNTS (not rate) -- flux[i] * bin_sec, approximately.
    bg_counts : expected background COUNTS per bin, same length (bg rate x bin_sec).
    threshold : used to decide when to reset the hull after an alarm ends.
    max_window_bins : oldest a retained candidate changepoint is allowed to be.

    Returns (stat, onset_of): stat[i] = the best log-likelihood-ratio ending
    at bin i (0 if nothing stands out); onset_of[i] = the bin index the best
    segment started at (-1 if stat[i] == 0).
    """
    n = len(counts)
    stat = np.zeros(n)
    onset_of = np.full(n, -1, dtype=int)
    if n == 0:
        return stat, onset_of

    csum = 0.0
    bgsum = 0.0
    hull_c = [0.0]      # cumulative counts at each retained candidate changepoint
    hull_t = [0]        # bin index of each retained candidate
    hull_bg = [0.0]      # cumulative background at each retained candidate
    was_above = False

    for i in range(n):
        c = max(float(counts[i]), 0.0)
        b0 = max(float(bg_counts[i]), 1e-9)
        csum += c
        bgsum += b0
        t = i + 1

        # Maintain the lower convex hull of (t, csum) -- the FOCuS pruning
        # trick: a candidate changepoint that a later one always beats (steeper
        # average rate to "now", for every possible future "now") can never be
        # the best segment again, so it's dropped for good. The hull normally
        # holds only a handful of points.
        new_c, new_t = csum, t
        while len(hull_c) >= 2:
            c1, t1 = hull_c[-2], hull_t[-2]
            c2, t2 = hull_c[-1], hull_t[-1]
            left = (c2 - c1) * (new_t - t2)
            right = (new_c - c2) * (t2 - t1)
            if left >= right:
                hull_c.pop(); hull_t.pop(); hull_bg.pop()
            else:
                break
        hull_c.append(new_c); hull_t.append(new_t); hull_bg.append(bgsum)
        if len(hull_c) > max_curves:
            del hull_c[0:len(hull_c) - max_curves]
            del hull_t[0:len(hull_t) - max_curves]
            del hull_bg[0:len(hull_bg) - max_curves]

        # Evict candidates older than the max window -- bounds how far back
        # any single segment can reach (see docstring fix #2 above).
        cutoff = t - max_window_bins
        while len(hull_t) >= 2 and hull_t[0] < cutoff:
            del hull_c[0]; del hull_t[0]; del hull_bg[0]

        # Evaluate the best-fit segment (any retained candidate -> now).
        best_stat, best_tau = 0.0, -1
        for k in range(len(hull_t) - 1):
            seg_counts = csum - hull_c[k]
            seg_bins = t - hull_t[k]
            seg_bg = bgsum - hull_bg[k]
            if seg_bins <= 0 or seg_bg <= 0:
                continue
            lam_hat = seg_counts / seg_bins
            mu0 = seg_bg / seg_bins
            if lam_hat <= mu0:
                continue          # only test rate INCREASES
            llr = seg_counts * math.log(lam_hat / mu0) - (seg_counts - seg_bins * mu0)
            if llr > best_stat:
                best_stat, best_tau = llr, hull_t[k]
        stat[i] = best_stat
        onset_of[i] = best_tau

        now_above = best_stat > threshold
        if was_above and not now_above:
            # Just exited an alarm: re-anchor the hull at "now" so this
            # burst's evidence can't keep masking whatever comes next.
            hull_c = [csum]; hull_t = [t]; hull_bg = [bgsum]
        was_above = now_above

    return stat, onset_of


def fred_shape_statistic(flux, alpha_fast=FRED_ALPHA_FAST, alpha_slow=FRED_ALPHA_SLOW):
    """
    Cheap shape confirmer: does this rise actually LOOK like a flare (fast
    rise, slower decay), or just noise? Two EMAs (a fast one that tracks the
    signal closely, a slow one that lags) approximate the classic FRED pulse
    shape (fast-rise/exponential-decay = difference of two exponentials).
    The statistic runs near zero for flat noise and peaks for a genuine
    FRED-shaped burst. O(n), two floats of running state -- used as extra
    EVIDENCE for confidence scoring, not a hard pass/fail gate (an unusual
    but real flare shouldn't get thrown out just for not fitting a template).
    """
    n = len(flux)
    if n == 0:
        return np.zeros(0)
    fast = np.empty(n); slow = np.empty(n)
    f = s = float(flux[0])
    for i in range(n):
        x = float(flux[i])
        f = x if i == 0 else alpha_fast * f + (1 - alpha_fast) * x
        s = x if i == 0 else alpha_slow * s + (1 - alpha_slow) * x
        fast[i] = f; slow[i] = s
    return slow - fast


def noisy_or(probs):
    """
    Combine independent pieces of evidence into one confidence number:
    confidence = 1 - product(1 - p_i). One strong signal alone already gives
    high confidence; agreeing signals push it higher -- the standard way to
    fuse independent probabilistic evidence (used here instead of a bare
    "X out of N checks passed" ratio, which treats a 3-of-3 flare the same
    as a 3-of-6 one even though the second has weaker evidence per check).
    Ignores None entries (no evidence either way, not "evidence against").
    """
    prod = 1.0
    any_seen = False
    for p in probs:
        if p is None:
            continue
        any_seen = True
        pc = 0.0 if p < 0.0 else (1.0 if p > 1.0 else float(p))
        prod *= (1.0 - pc)
    return (1.0 - prod) if any_seen else 0.0


def detect_events(bt, flux):
    """
    v4: Poisson-FOCuS changepoint detection (primary trigger) + two extra
    false-positive guards, replacing the old fixed-window sigma test.

    WHY: the v3 (10th-percentile + MAD, 5-sigma) detector found 34-53
    "flares"/day on real data -- not physically real, an artifact of testing
    every bin against one fixed 20-minute window regardless of how long the
    real burst is. v4 fixes this three ways:

      1. TRIGGER: Poisson-FOCuS (poisson_focus_scan) tests every window
         length at once using the correct statistic for count data, instead
         of one fixed window with a Gaussian-style sigma test.
      2. NOISE-BLIP GATE: after building a candidate, require its peak to
         beat its own start by >= MIN_RISE_RATIO (matches the GOES 4-minute
         rise convention). Anything that doesn't really grow is dropped
         silently -- never logged as a flare at all, not even a weak one.
      3. SHAPE EVIDENCE: fred_shape_statistic scores whether the candidate
         actually looks like fast-rise/slow-decay. Stored on the event for
         run_cross_checks() to fold into confidence -- not a hard gate (an
         odd-shaped real flare shouldn't be thrown out), just evidence.

    Background estimation, background floor, event merging, and the
    data-gap flag are unchanged from v3 (10th-percentile background was
    already a real improvement, kept). The old MAD-based sigma is still
    computed and kept on the event dict (ev["sigma"]) purely for display/
    continuity -- FOCuS's own statistic (ev["focus_stat"]) is what actually
    decides what's a flare now.

    Returns a list of events, each a dict with start/peak/end index & time
    (same keys as v3, plus focus_stat and fred_shape_stat).
    """
    if len(flux) < BG_WINDOW_BINS:
        return []
    s = pd.Series(flux)
    bg = s.rolling(BG_WINDOW_BINS, center=True, min_periods=BG_WINDOW_BINS // 4).quantile(0.10)
    bg = bg.ffill().bfill().values

    excess = flux - bg
    diffs = np.diff(excess, prepend=excess[0])
    mad = (pd.Series(np.abs(diffs))
           .rolling(BG_WINDOW_BINS, center=True, min_periods=1)
           .median().bfill().ffill().to_numpy())
    noise = 1.4826 * mad / np.sqrt(2)                     # MAD -> std-equivalent
    med = np.nanmedian(noise[noise > 0]) if (noise > 0).any() else 1.0
    noise[noise <= 0] = med if np.isfinite(med) and med > 0 else 1.0
    noise = np.maximum(noise, 1.0)                         # never below 1 count/s

    # ---- background floor ---------------------------------------------------
    day_bg = np.median(bg[bg > 0]) if (bg > 0).any() else 1.0
    bg_floor = max(0.10 * day_bg, 1.0)         # never below 10% of typical, or 1
    bg_safe = np.maximum(bg, bg_floor)

    sigma = (flux - bg_safe) / noise           # kept for display/continuity only

    # ---- v4 trigger: Poisson-FOCuS, not a fixed-window sigma test -----------
    # IMPORTANT: FOCuS needs mu0 = the TRUE expected background rate, not the
    # 10th-percentile FLOOR used for bg/ratio/classification above. A 10th
    # percentile is deliberately biased low (that's its job for reporting a
    # quiet-Sun floor) -- but ~90% of ordinary samples exceed their own local
    # 10th percentile BY DEFINITION, so feeding that into a changepoint test
    # makes even pure noise look like sustained "excess", and the statistic
    # eventually crosses any threshold given enough bins (confirmed: this was
    # the second bug found testing v4 -- pure noise was flagged as one
    # all-day "flare"). The rolling MEDIAN is the correct, unbiased estimate
    # of "typical rate" for this specific use.
    bg_mu0 = s.rolling(BG_WINDOW_BINS, center=True, min_periods=BG_WINDOW_BINS // 4).median()
    bg_mu0 = bg_mu0.ffill().bfill().values
    bg_mu0 = np.maximum(bg_mu0, bg_floor)

    counts_bin = np.maximum(flux, 0.0) * BIN_SEC       # rate -> approx counts/bin
    bg_counts_bin = np.maximum(bg_mu0, 1e-6) * BIN_SEC
    focus_stat, focus_onset = poisson_focus_scan(counts_bin, bg_counts_bin)
    above = focus_stat > FOCUS_THRESHOLD

    raw, in_evt, s_idx = [], False, 0
    for i in range(len(above)):
        if above[i] and not in_evt:
            in_evt, s_idx = True, i
        elif not above[i] and in_evt:
            if i - s_idx >= MIN_DUR_BINS:
                raw.append((s_idx, i))
            in_evt = False
    if in_evt and len(above) - s_idx >= MIN_DUR_BINS:      # event still open at day end
        raw.append((s_idx, len(above) - 1))

    # ---- merge events separated by a short dip --------------------------------
    MERGE_GAP_BINS = int(MERGE_GAP_S / BIN_SEC)
    merged = []
    for seg in raw:
        if merged and seg[0] - merged[-1][1] <= MERGE_GAP_BINS:
            merged[-1] = (merged[-1][0], seg[1])       # extend the previous event
        else:
            merged.append(seg)

    fred_full = fred_shape_statistic(flux)

    events = []
    for s_idx, e_idx in merged:
        pk = s_idx + int(np.argmax(flux[s_idx:e_idx]))

        # ---- noise-blip rejection gate: must really grow, or it's dropped ----
        # Use the real background at this time as the "quiet" reference, not
        # flux[s_idx] -- with FOCuS, s_idx can land ANYWHERE inside the
        # already-rising segment (not necessarily the true pre-flare quiet
        # bin the old sigma-threshold detector always started at), so
        # flux[s_idx] can already be partway up the flare itself. Comparing
        # against that would make the gate reject real flares by accident
        # (confirmed: this exact bug produced 0/3 real flares in testing).
        start_level = max(float(bg_safe[s_idx]), 1e-6)
        if float(flux[pk]) < MIN_RISE_RATIO * start_level:
            continue

        raw_bg = float(bg[pk])                          # the UNfloored background
        events.append(dict(
            s_idx=s_idx, p_idx=pk, e_idx=e_idx,
            start=float(bt[s_idx]), peak=float(bt[pk]), end=float(bt[e_idx]),
            peak_flux=float(flux[pk]),
            bg=float(bg_safe[pk]),
            ratio=float(flux[pk] / bg_safe[pk]),
            sigma=float(sigma[pk]),
            data_gap=bool(raw_bg < bg_floor),
            focus_stat=float(np.max(focus_stat[s_idx:e_idx])) if e_idx > s_idx else 0.0,
            fred_shape_stat=float(np.max(fred_full[s_idx:e_idx])) if e_idx > s_idx else 0.0,
        ))
    return events


# =============================================================================
#  SECTION 5  --  THE FOUR PHYSICS SIGNATURES
# =============================================================================

