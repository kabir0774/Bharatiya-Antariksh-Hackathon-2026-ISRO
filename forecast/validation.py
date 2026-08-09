from .common import *
from .evaluation import *

def walk_forward_summary(df, best_proto, oversample=False, n_folds=5):
    """
    ROBUSTNESS CHECK: the single train/test split leans on whichever active
    day happens to fall in the test dates (e.g. one date carrying 5 of the
    significant flares). This walks forward across ALL dates -- each fold
    retrains a fresh clone of the winning model on strictly-earlier dates
    and tests on the next date -- and reports TSS per fold, model vs
    persistence, each at its OWN best threshold (same treatment for both, so
    the comparison is fair; both numbers are "best case at the ideal
    threshold" -- what matters is the GAP and its consistency, not the
    absolute values). The mean +/- spread across folds is a far sturdier
    claim than one split.
    """
    from sklearn.base import clone
    folds = rolling_origin_date_splits(df, n_folds=n_folds)
    if not folds:
        print("\nWalk-forward summary: not enough dates for folds.")
        return

    def _best_tss(y, prob):
        res, _ = sweep_thresholds(y, prob)
        return max((r["tss"] for r in res), default=0.0)

    print(f"\nWalk-forward robustness summary ({len(folds)} folds, each tests one unseen date;")
    print("model vs persistence, each at its own best threshold -- the GAP is the claim):")
    rows_m, rows_p = [], []
    for fold_train, fold_test, fold_date in folds:
        yte = fold_test["label"].values
        if yte.sum() == 0 or yte.sum() == len(yte):
            print(f"  {fold_date}: skipped (no positive/negative mix in this date's windows)")
            continue
        Xtr, ytr = fold_train[FEATURE_NAMES].values, fold_train["label"].values
        Xte = fold_test[FEATURE_NAMES].values
        sc = StandardScaler().fit(Xtr)
        try:
            c = clone(best_proto)
            if oversample:
                r = np.random.default_rng(42)
                p_i, n_i = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
                if len(p_i) > 0 and len(n_i) > len(p_i):
                    extra = r.choice(p_i, size=len(n_i) - len(p_i), replace=True)
                    idx = np.concatenate([np.arange(len(ytr)), extra])
                    Xtr2, ytr2 = Xtr[idx], ytr[idx]
                else:
                    Xtr2, ytr2 = Xtr, ytr
                c.fit(sc.transform(Xtr2), ytr2)
            else:
                c.fit(sc.transform(Xtr), ytr)
            prob = c.predict_proba(sc.transform(Xte))[:, 1]
        except Exception as exc:
            print(f"  {fold_date}: skipped ({exc})")
            continue
        m_tss = _best_tss(yte, prob)
        p_tss = _best_tss(yte, PersistenceBaseline().fit().predict_proba(Xte))
        rows_m.append(m_tss)
        rows_p.append(p_tss)
        beat = "BEATS persistence" if m_tss > p_tss else "does NOT beat persistence"
        print(f"  {fold_date}: model TSS {m_tss:+.3f}  vs persistence {p_tss:+.3f}  -> {beat} "
              f"({int(yte.sum())} pos / {len(yte)} windows)")
    if rows_m:
        gap = np.array(rows_m) - np.array(rows_p)
        print(f"  ACROSS FOLDS: model TSS {np.mean(rows_m):+.3f} +/- {np.std(rows_m):.3f}   "
              f"persistence {np.mean(rows_p):+.3f} +/- {np.std(rows_p):.3f}")
        print(f"  mean GAP over persistence: {np.mean(gap):+.3f}  "
              f"(beats it in {int((gap > 0).sum())}/{len(gap)} folds) -- this fold-averaged "
              f"gap is the sturdier headline number, not the single-split TSS.")


def measure_lead_time(test_df, best_prob, threshold=ALARM_THRESHOLD):
    """
    For each real upcoming flare in the test set, find the EARLIEST window
    (by window_end time) where the model already triggered an alarm before
    that flare started. Lead time = flare_start - that window's end time.
    """
    test_df = test_df.copy()
    test_df["prob"] = best_prob
    test_df["alarm"] = best_prob >= threshold

    lead_times = []
    for flare_start, grp in test_df[test_df["label"] == 1].groupby("flare_start"):
        triggered = grp[grp["alarm"]]
        if len(triggered) == 0:
            continue
        earliest = triggered["window_end_unix"].min()
        lead_min = (flare_start - earliest) / 60.0
        if lead_min >= 0:
            lead_times.append(lead_min)
    return lead_times


# =============================================================================
#  SECTION 9 -- NOWCAST-REFEREE CROSS-CHECK
# =============================================================================

def nowcast_referee_tally(test_df, best_prob, threshold=ALARM_THRESHOLD):
    """
    Four-way tally against ground truth (the nowcast detector's own labels,
    which is what built test_df['label'] in the first place): correct alarm,
    correct quiet, false alarm, missed flare. This is a sanity re-statement
    of precision/recall in the plain-language form used in the proposal.
    """
    alarm = best_prob >= threshold
    label = test_df["label"].values.astype(bool)
    correct_alarm = int(np.sum(alarm & label))
    correct_quiet = int(np.sum(~alarm & ~label))
    false_alarm = int(np.sum(alarm & ~label))
    missed = int(np.sum(~alarm & label))
    return dict(correct_alarm=correct_alarm, correct_quiet=correct_quiet,
               false_alarm=false_alarm, missed=missed)
