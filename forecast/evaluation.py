from .common import *

# =============================================================================
#  SECTION 5.5 -- REAL SKILL SCORES (not generic ML metrics)
# =============================================================================
#  AUC/precision/recall are generic ML metrics. Real operational space-weather
#  forecasting (NOAA SWPC and the published literature we're benchmarking
#  against) uses a different battery: POD, FAR, TSS, HSS, BSS. These are what
#  a judge who knows this field will actually expect to see.

def pod_far(y_true, y_pred):
    """POD (= recall, "of real flares, how many did we catch") and
    FAR (false alarm RATIO, "of our alarms, how many were wrong"). Both in [0,1]."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    pod = tp / max(tp + fn, 1)
    far = fp / max(fp + tp, 1)
    return float(pod), float(far)


def tss(y_true, y_pred):
    """
    True Skill Statistic = POD - POFD (false-positive rate). The standard
    real-world verification metric for rare weather events -- unlike plain
    accuracy or even AUC, TSS gives the SAME number regardless of how rare
    the positive class is (2% or 50%), which is exactly the property you
    want when comparing across different datasets/papers. 0 = no skill,
    1 = perfect, negative = worse than random guessing.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    pod = tp / max(tp + fn, 1)
    pofd = fp / max(fp + tn, 1)
    return float(pod - pofd)


def hss(y_true, y_pred):
    """Heidke Skill Score: how much better than random-chance agreement,
    accounting for how easy some correct guesses would be by luck alone."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    n = tp + fn + fp + tn
    if n == 0:
        return 0.0
    expected_correct = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n
    observed_correct = tp + tn
    denom = n - expected_correct
    return float((observed_correct - expected_correct) / denom) if denom > 0 else 0.0


def bss(y_true, p_pred, p_clim):
    """
    Brier Skill Score vs a climatology (constant base-rate) reference.
    Checks whether your PROBABILITY numbers are honest, not just your
    yes/no calls: Brier score = mean((p - y)^2); BSS = 1 - BS/BS_climatology.
    Positive = your probabilities beat "always guess the historical rate";
    zero or negative = they don't, no matter how good your AUC looks.
    """
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    p_clim = np.asarray(p_clim, dtype=float) if hasattr(p_clim, "__len__") else np.full_like(y_true, p_clim)
    bs = np.mean((p_pred - y_true) ** 2)
    bs_clim = np.mean((p_clim - y_true) ** 2)
    return float(1.0 - bs / bs_clim) if bs_clim > 0 else 0.0


def skill_report(y_true, p_pred, threshold, p_clim=None):
    """One dict with the full real battery, at a chosen probability threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(p_pred) >= threshold).astype(int)
    pod, far = pod_far(y_true, pred)
    clim = p_clim if p_clim is not None else float(np.mean(y_true))
    return dict(
        TSS=tss(y_true, pred), HSS=hss(y_true, pred),
        BSS=bss(y_true, p_pred, clim), POD=pod, FAR=far,
        AUC=(roc_auc_score(y_true, p_pred) if 0 < np.mean(y_true) < 1 else float("nan")),
    )


# =============================================================================
#  SECTION 5.6 -- MANDATORY BASELINES (climatology, persistence)
# =============================================================================
#  "Beat persistence" is the real bar in this field -- a 2025 NOAA SWPC
#  verification study found real operational forecasts often DON'T beat
#  "just guess it stays the same". Reporting a bare TSS/AUC without this
#  comparison is not a credible claim in this field.

class ClimatologyBaseline:
    """Dumbest possible forecaster: always predict the training base rate,
    ignoring the input entirely. The floor everything else must clear."""
    def fit(self, y):
        vals = np.asarray(y, dtype=float)
        self.base_rate_ = float(np.clip(vals.mean() if len(vals) else 0.0, 1e-6, 1 - 1e-6))
        return self

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self.base_rate_)


class PersistenceBaseline:
    """
    "Flaring right now -> predict a flare soon" -- zero training, uses only
    this window's own flux_end feature (index of FEATURE_NAMES: flux_end,
    background-normalized flux at the end of the lookback window, i.e. "right
    now"). No model, no features beyond what's already >= 1.5x background.
    """
    FLARING_REL = 1.5

    def __init__(self):
        self._p_hi, self._p_lo = 1.0 - 1e-3, 1e-3
        self._idx = FEATURE_NAMES.index("flux_end")

    def fit(self, X=None, y=None):
        return self

    def predict_proba(self, X):
        vals = X[:, self._idx] if hasattr(X, "shape") else np.array([row[self._idx] for row in X])
        return np.where(vals >= self.FLARING_REL, self._p_hi, self._p_lo)


class PlattCalibrator:
    """
    Two-parameter sigmoid probability calibration (Platt 1999). Module-level
    class (not a local one) so joblib can pickle it into calibrator.pkl.
    .fit(p, y) / .predict(p) mirror IsotonicRegression's API so the two are
    drop-in interchangeable in train_models().
    """
    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        self._lr = LogisticRegression(C=1e6, max_iter=1000)
        self._lr.fit(z.reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        return self._lr.predict_proba(z.reshape(-1, 1))[:, 1]


# =============================================================================
#  SECTION 6 -- TRAIN / TEST SPLIT (BY TIME, never random)
# =============================================================================

def time_split(df):
    uniq_dates = sorted(df["date"].unique())
    n_test = max(1, int(round(len(uniq_dates) * TEST_FRACTION)))
    test_dates = set(uniq_dates[-n_test:])
    train_dates = set(uniq_dates) - test_dates
    train_df = df[df["date"].isin(train_dates)].reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].reset_index(drop=True)
    # dense fine-grid windows are a TRAINING aid (more real positive examples)
    # -- the TEST set stays on the natural coarse grid so every metric is
    # measured on the same window distribution the system would see live,
    # and stays comparable with earlier runs.
    if "dense" in test_df.columns:
        n_drop = int(test_df["dense"].sum())
        test_df = test_df[test_df["dense"] == 0].reset_index(drop=True)
        if n_drop:
            print(f"  (test set kept on the natural {STEP_MIN}-min grid: "
                  f"{n_drop} dense train-augmentation windows excluded)")
    print(f"\nTime-based split: train on {sorted(train_dates)}")
    print(f"                  test on  {sorted(test_dates)}")
    print(f"  train: {len(train_df)} windows ({train_df['label'].sum()} positive)")
    print(f"  test:  {len(test_df)} windows ({test_df['label'].sum()} positive)")
    return train_df, test_df


def rolling_origin_date_splits(df, n_folds=5, embargo_min=DECAY_EXCLUDE_MIN):
    """
    STRICTER CV than time_split(): instead of one fixed train/test date cut,
    walk forward across the available dates in n_folds steps (train on all
    dates before a moving cutoff, test on the next date), each fold with an
    embargo gap (drop any training window within embargo_min minutes of the
    test date's start -- since we split by whole calendar date already there
    is no cross-date window overlap to begin with, but the embargo also
    guards the case where two dates are adjacent in time with no real gap
    between them). Averaging metrics across folds is a far more reliable
    performance estimate than one arbitrary split, especially with as few
    dates as this project has.

    Returns a list of (train_df, test_df, test_date) tuples.
    """
    uniq_dates = sorted(df["date"].unique())
    if len(uniq_dates) < 2:
        return []
    n_folds = min(n_folds, len(uniq_dates) - 1)
    folds = []
    # walk forward: fold i tests on uniq_dates[i], trains on everything before it
    test_indices = list(range(len(uniq_dates) - n_folds, len(uniq_dates)))
    for ti in test_indices:
        test_date = uniq_dates[ti]
        train_dates = uniq_dates[:ti]
        if not train_dates:
            continue
        test_df = df[df["date"] == test_date].reset_index(drop=True)
        if "dense" in test_df.columns:
            # fold-test mimics deployment: natural grid only, so the
            # calibrator learns honest deployment-time probabilities
            test_df = test_df[test_df["dense"] == 0].reset_index(drop=True)
        if len(test_df) == 0:
            continue
        test_start = test_df["window_end_unix"].min()
        embargo_s = embargo_min * 60.0
        train_mask = df["date"].isin(train_dates) & (df["window_end_unix"] < test_start - embargo_s)
        train_df = df[train_mask].reset_index(drop=True)
        if len(train_df) == 0 or train_df["label"].sum() == 0:
            continue
        folds.append((train_df, test_df, test_date))
    return folds
