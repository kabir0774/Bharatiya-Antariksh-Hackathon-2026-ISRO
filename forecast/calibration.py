from .common import *

#  SECTION 8 -- LEAD TIME
# =============================================================================

def reliability_report(y_test, prob, n_bins=10):
    """
    THE HONEST ANSWER TO "why can't it say 60%?"

    A calibrated model outputting mostly 2-8% on a ~4% base rate is not
    necessarily broken -- it may be correctly reporting that most moments
    truly are that safe. But there are two very different reasons the
    ceiling could be low, and they need opposite fixes:

      (A) EVIDENCE problem: no window in this dataset has ever landed in a
          feature region with enough real positive support to justify a high
          number. Fix: more real flare days, or features that concentrate
          the signal into a sharper cluster.
      (B) MODELING problem: the calibrator itself is broken (self-fit
          leakage, isotonic collapse, etc). Fix: a code bug -- already
          guarded against elsewhere in this file, but this report is the
          tripwire that would catch a NEW one.

    This prints, for each predicted-probability band, how many test windows
    landed there and what fraction ACTUALLY flared (the reliability curve).
    A trustworthy model's bars roughly match the diagonal (predicted ~=
    observed). A ceiling with LOW bin counts near the top is an evidence
    problem (A) -- there simply isn't a densely-supported high-confidence
    region yet. A ceiling with bars that DISAGREE with the diagonal even at
    well-populated bins is a modeling problem (B).
    """
    edges = np.linspace(0.0, max(float(prob.max()), 1e-6) * 1.001, n_bins + 1)
    print(f"\nReliability check -- \"when we say X%, does X% of the time a flare "
          f"really happen?\" (test set, {len(y_test)} windows, {int(y_test.sum())} positive):")
    print(f"  {'predicted range':>18}  {'n windows':>10}  {'n positive':>11}  "
          f"{'observed rate':>14}  {'support':>8}")
    max_reached = 0.0
    max_reached_support = 0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (prob >= lo) & (prob < hi if i < n_bins - 1 else prob <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        npos = int(y_test[m].sum())
        obs = npos / n
        support = "thin" if npos < 5 else ("ok" if npos < 15 else "solid")
        print(f"  {100*lo:6.1f}-{100*hi:5.1f}%      {n:10d}  {npos:11d}  "
              f"{100*obs:12.1f}%      {support:>8}")
        if n > 0:
            max_reached = hi
            max_reached_support = npos
    print(f"\n  Highest confidence this model EVER reaches on this data: "
          f"{100*max_reached:.1f}%, resting on {max_reached_support} real positive "
          f"example(s) in that band.")
    if max_reached < 0.30:
        print(f"  This is an EVIDENCE ceiling, not a broken calibrator: no window in the")
        print(f"  current data sits in a feature region with enough clustered positive")
        print(f"  support to honestly justify a higher number. Two real fixes -- (1) more")
        print(f"  real flare days so a genuinely high-confidence cluster can emerge, or")
        print(f"  (2) features that concentrate the strongest precursor signals into fewer,")
        print(f"  sharper dimensions instead of spreading evidence across {N_FEATURES} features.")
        print(f"  Forcing the number higher WITHOUT either of those would mean the number")
        print(f"  stops being true -- which is the one thing this system will not do.")


def sweep_thresholds(y_test, prob, thresholds=None):
    """
    At 2-3% positive rate, the default 0.5 threshold is almost never the
    right operating point. This sweeps thresholds and reports precision/
    recall at each.

    LIMITATION FIX: the sweep used to hand back only the best-F1 point,
    silently deciding the false-alarms-vs-missed-flares trade-off. Now every
    threshold also gets F0.5 (precision-leaning: false alarms cost more),
    F2 (recall-leaning: a missed flare costs more), and TSS (the field's own
    skill score), and the function returns the best threshold for EACH
    objective. Which one becomes the live alarm point is a visible config
    choice (OPERATING_OBJECTIVE), made by the person running the system --
    an astronaut-safety user wants F2; a "don't cry wolf" alert service
    wants F0.5.

    Returns (results_list, best_thresholds_dict) where best_thresholds_dict
    has keys "f0.5", "f1", "f2", "tss".
    """
    if thresholds is None:
        # ADAPTIVE GRID: a fixed 0.05..0.90 ladder assumes probabilities are
        # spread across [0,1], but honestly-calibrated rare-event scores can
        # live almost entirely below 0.2 -- a fixed grid then sees ONE usable
        # threshold and every "operating point" collapses onto it (seen on
        # real data). Build the ladder from where the scores actually are:
        # quantiles of the predicted probabilities themselves, plus the
        # classic ladder for continuity.
        qs = np.quantile(prob, np.linspace(0.02, 0.995, 40))
        # round to 3 decimals BEFORE dedup -- the same precision the results
        # store and the alarm applies, so no two rows can share a printed
        # threshold and the tally always matches its sweep row exactly
        thresholds = np.unique(np.round(np.concatenate([
            qs, np.arange(0.05, 0.95, 0.05)]), 3))
        thresholds = thresholds[(thresholds > 0) & (thresholds < 1)]

    def fbeta(prec, rec, beta):
        b2 = beta * beta
        return (1 + b2) * prec * rec / max(b2 * prec + rec, 1e-9)

    results = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y_test == 1)))
        fp = int(np.sum((pred == 1) & (y_test == 0)))
        fn = int(np.sum((pred == 0) & (y_test == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        results.append(dict(threshold=round(float(t), 3), precision=prec, recall=rec,
                            f0_5=fbeta(prec, rec, 0.5),
                            f1=fbeta(prec, rec, 1.0),
                            f2=fbeta(prec, rec, 2.0),
                            tss=tss(y_test, pred),
                            tp=tp, fp=fp, fn=fn))
    if not results:
        d = ALARM_THRESHOLD
        return results, {"f0.5": d, "f1": d, "f2": d, "tss": d}
    best_thresholds = {
        "f0.5": max(results, key=lambda r: r["f0_5"])["threshold"],
        "f1":   max(results, key=lambda r: r["f1"])["threshold"],
        "f2":   max(results, key=lambda r: r["f2"])["threshold"],
        "tss":  max(results, key=lambda r: r["tss"])["threshold"],
    }
    return results, best_thresholds
