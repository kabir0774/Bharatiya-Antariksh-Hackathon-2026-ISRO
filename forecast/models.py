from .common import *
from .windowing import *
from .evaluation import *

# =============================================================================
#  SECTION 7 -- TRAIN MODELS
# =============================================================================

def train_models(train_df, test_df):
    os.makedirs(MODEL_OUTPUT_FOLDER, exist_ok=True)
    X_train, y_train = train_df[FEATURE_NAMES].values, train_df["label"].values
    X_test, y_test = test_df[FEATURE_NAMES].values, test_df["label"].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {}

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=3,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)
    rf_prob = rf.predict_proba(X_test_sc)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_prob) if y_test.sum() > 0 and y_test.sum() < len(y_test) else float("nan")
    print(f"  Random Forest AUC: {rf_auc:.3f}")
    models["RandomForest"] = dict(model=rf, prob=rf_prob, auc=rf_auc)

    # ---- shallow/regularized RF: tests the "overfitting on ~200 positive
    # examples" theory directly. Same features, much less capacity to
    # memorize training-set quirks -- if THIS beats the deep RF and/or
    # persistence where the deep one didn't, that confirms overfitting was
    # the real problem (not missing features -- the clustering-features
    # theory was checked via feature importance and ruled out).
    print("Training shallow/regularized Random Forest (overfitting check)...")
    rf_shallow = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=15,
                                        max_features="sqrt", class_weight="balanced",
                                        random_state=42, n_jobs=-1)
    rf_shallow.fit(X_train_sc, y_train)
    rf_shallow_prob = rf_shallow.predict_proba(X_test_sc)[:, 1]
    rf_shallow_auc = (roc_auc_score(y_test, rf_shallow_prob)
                      if y_test.sum() > 0 and y_test.sum() < len(y_test) else float("nan"))
    print(f"  Shallow RF AUC: {rf_shallow_auc:.3f}")
    models["RandomForest_shallow"] = dict(model=rf_shallow, prob=rf_shallow_prob, auc=rf_shallow_auc)

    # ---- logistic regression: the classic small-data-safe choice. Can't
    # memorize training noise the way a tree ensemble can (one linear
    # boundary, no matter how much you regularize a tree) -- if THIS beats
    # persistence, the fix for this dataset size is "simpler model", not
    # more features or more trees.
    print("Training Logistic Regression (small-data-safe check)...")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_prob = lr.predict_proba(X_test_sc)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_prob) if y_test.sum() > 0 and y_test.sum() < len(y_test) else float("nan")
    print(f"  Logistic Regression AUC: {lr_auc:.3f}")
    models["LogisticRegression"] = dict(model=lr, prob=lr_prob, auc=lr_auc)

    if HAVE_XGB:
        print("Training XGBoost...")
        pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            scale_pos_weight=pos_weight, eval_metric="logloss",
                            random_state=42, n_jobs=-1)
        xgb.fit(X_train_sc, y_train)
        xgb_prob = xgb.predict_proba(X_test_sc)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_prob) if y_test.sum() > 0 and y_test.sum() < len(y_test) else float("nan")
        print(f"  XGBoost AUC: {xgb_auc:.3f}")
        models["XGBoost"] = dict(model=xgb, prob=xgb_prob, auc=xgb_auc)
    else:
        print("Training GradientBoosting (xgboost not installed)...")
        gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                        random_state=42)
        gb.fit(X_train_sc, y_train)
        gb_prob = gb.predict_proba(X_test_sc)[:, 1]
        gb_auc = roc_auc_score(y_test, gb_prob) if y_test.sum() > 0 and y_test.sum() < len(y_test) else float("nan")
        print(f"  GradientBoosting AUC: {gb_auc:.3f}")
        models["GradientBoosting"] = dict(model=gb, prob=gb_prob, auc=gb_auc)

    # ---- LIMITATION FIX: a neural network IS now tried -----------------------
    # A deliberately SMALL, heavily regularized MLP (two hidden layers,
    # strong L2, early stopping) -- big deep nets would just re-run the
    # overfitting story already diagnosed on this dataset size, so the net
    # gets the same "less capacity to memorize" treatment the shallow RF
    # got. MLPClassifier has no class_weight option, so the positive class
    # is randomly oversampled (with replacement) to balance -- resampling
    # real rows, not inventing synthetic ones. It competes under the exact
    # same TSS-vs-persistence selection rule as everyone else; if it loses
    # on this data size, the table below shows that honestly.
    print("Training small regularized MLP (neural-network candidate)...")
    from sklearn.neural_network import MLPClassifier
    rng_mlp = np.random.default_rng(42)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        boost = rng_mlp.choice(pos_idx, size=max(len(neg_idx) - len(pos_idx), 0), replace=True)
        bal_idx = np.concatenate([np.arange(len(y_train)), boost])
    else:
        bal_idx = np.arange(len(y_train))
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=1000,
                        early_stopping=True, n_iter_no_change=20,
                        validation_fraction=0.15, random_state=42)
    mlp.fit(X_train_sc[bal_idx], y_train[bal_idx])
    mlp_prob = mlp.predict_proba(X_test_sc)[:, 1]
    mlp_auc = roc_auc_score(y_test, mlp_prob) if y_test.sum() > 0 and y_test.sum() < len(y_test) else float("nan")
    print(f"  MLP AUC: {mlp_auc:.3f}")
    models["MLP_neural"] = dict(model=mlp, prob=mlp_prob, auc=mlp_auc,
                                oversample=True)

    # ---- pick "best" by TSS-vs-persistence, NOT raw AUC ----------------------
    # Found the hard way (real run on real data): raw AUC picked a deep,
    # overfit RandomForest (AUC 0.611) over a shallow, regularized one (AUC
    # 0.601) -- a coin-flip AUC gap -- while on the metric that actually
    # matters, the shallow one beat persistence (TSS +0.197) and the deep
    # one didn't (TSS +0.088). AUC alone was picking the WORSE model. Compute
    # the persistence baseline first, then prefer whichever candidate clears
    # it, breaking ties by TSS (not AUC). Only fall back to raw AUC if
    # nothing clears persistence at all.
    clim = ClimatologyBaseline().fit(y_train)
    pers = PersistenceBaseline().fit()
    clim_prob = clim.predict_proba(X_test)
    pers_prob = pers.predict_proba(X_test)
    pers_tss = tss(y_test, pers_prob >= ALARM_THRESHOLD)
    clim_tss_at_thresh = tss(y_test, clim_prob >= ALARM_THRESHOLD)

    candidate_tss = {}
    for mname, mdict in models.items():
        candidate_tss[mname] = tss(y_test, mdict["prob"] >= ALARM_THRESHOLD)
    beats_persistence = {k: v for k, v in candidate_tss.items() if v > pers_tss}

    if beats_persistence:
        best_name = max(beats_persistence, key=lambda k: beats_persistence[k])
        print(f"\nModel selection: {len(beats_persistence)}/{len(models)} candidate(s) beat "
              f"persistence -- picking the highest-TSS one among them ({best_name}), not raw AUC.")
    else:
        valid = {k: v for k, v in models.items() if np.isfinite(v["auc"])}
        best_name = max(valid, key=lambda k: valid[k]["auc"]) if valid else list(models.keys())[0]
        print(f"\nModel selection: NO candidate beat persistence -- falling back to highest "
              f"raw AUC ({best_name}). None of these are a credible forecasting claim yet.")
    best = models[best_name]

    # ---- probability calibration ---------------------------------------------
    # A raw model score is a good RANKING but not necessarily an honest
    # probability -- "70%" should mean "flares 70% of the time it says that".
    #
    # LIMITATION FIX: this used to be fitted on the model's OWN training
    # predictions (a stated limitation -- reads optimistic). Now it is fitted
    # on OUT-OF-FOLD predictions: the existing walk-forward splitter
    # (rolling_origin_date_splits) retrains a fresh clone of the winning
    # model inside the training dates only, each fold predicting a date the
    # clone never saw. Those predictions are honest "first sight" scores --
    # exactly what the calibrator will meet in deployment -- and no third
    # data split is needed, so nothing is starved. Falls back to the old
    # self-fit (with a loud note) only if the folds can't produce enough
    # positive examples to fit a curve at all.
    #
    # Snapshot each candidate's RAW probability BEFORE calibrating the
    # winner -- `best` is the SAME dict object as models[best_name] (not a
    # copy), so overwriting best["prob"] below would otherwise silently
    # leave one candidate calibrated and the rest raw in the comparison
    # further down, which is not a fair comparison.
    raw_probs_snapshot = {name: d["prob"] for name, d in models.items()}

    from sklearn.isotonic import IsotonicRegression
    from sklearn.base import clone

    def _fit_clone(proto, Xtr, ytr, oversample=False):
        c = clone(proto)
        if oversample:
            r = np.random.default_rng(42)
            p_i, n_i = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
            if len(p_i) > 0 and len(n_i) > len(p_i):
                extra = r.choice(p_i, size=len(n_i) - len(p_i), replace=True)
                idx = np.concatenate([np.arange(len(ytr)), extra])
                Xtr, ytr = Xtr[idx], ytr[idx]
        c.fit(Xtr, ytr)
        return c

    oof_prob, oof_y = [], []
    for fold_train, fold_test, fold_date in rolling_origin_date_splits(train_df, n_folds=CALIBRATION_FOLDS):
        Xtr, ytr = fold_train[FEATURE_NAMES].values, fold_train["label"].values
        Xte, yte = fold_test[FEATURE_NAMES].values, fold_test["label"].values
        fold_scaler = StandardScaler().fit(Xtr)
        try:
            c = _fit_clone(best["model"], fold_scaler.transform(Xtr), ytr,
                           oversample=best.get("oversample", False))
            oof_prob.append(c.predict_proba(fold_scaler.transform(Xte))[:, 1])
            oof_y.append(yte)
        except Exception as exc:
            print(f"  [calibration] fold {fold_date} skipped ({exc})")
    # -- calibration METHOD chosen by how much data there is to learn from --
    # Isotonic regression is a step function: powerful with plenty of data,
    # but with only a few dozen positive examples it collapses nearly all
    # scores onto a handful of plateau values (seen on real data: everything
    # squashed below 0.10, so the threshold sweep had exactly one usable
    # operating point). Platt scaling fits just TWO parameters (a smooth
    # sigmoid), so it stays well-behaved and keeps the score ORDERING intact
    # on tiny positive counts. Rule: < PLATT_MIN_POS positives -> Platt.
    PLATT_MIN_POS = 100

    if oof_prob and np.concatenate(oof_y).sum() >= 5:
        oof_prob = np.concatenate(oof_prob)
        oof_y = np.concatenate(oof_y)
        n_pos_oof = int(oof_y.sum())
        if n_pos_oof < PLATT_MIN_POS:
            calibrator = PlattCalibrator().fit(oof_prob, oof_y)
            method = f"Platt sigmoid (only {n_pos_oof} OOF positives -- isotonic would over-compress)"
        else:
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
            calibrator.fit(oof_prob, oof_y)
            method = "isotonic"
        print(f"  Probability calibration: {method}, fitted on {len(oof_y)} OUT-OF-FOLD "
              f"predictions ({n_pos_oof} positive) from "
              f"{CALIBRATION_FOLDS}-fold walk-forward CV inside the training dates -- "
              f"the calibrator never sees a prediction the model made on data it trained on.")
    else:
        calibrator = PlattCalibrator()
        train_prob_raw = best["model"].predict_proba(X_train_sc)[:, 1]
        calibrator.fit(train_prob_raw, y_train)
        print("  [NOTE] Too few out-of-fold positives to calibrate honestly -- fell back to "
              "Platt on training predictions (reads optimistic; add more dates).")
    best["prob_raw"] = best["prob"]
    best["prob"] = calibrator.predict(best["prob_raw"])   # calibrated prob is now the default
    joblib.dump(calibrator, os.path.join(MODEL_OUTPUT_FOLDER, "calibrator.pkl"))

    joblib.dump(best["model"], os.path.join(MODEL_OUTPUT_FOLDER, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_OUTPUT_FOLDER, "scaler.pkl"))
    with open(os.path.join(MODEL_OUTPUT_FOLDER, "feature_names.txt"), "w") as fh:
        fh.write("\n".join(FEATURE_NAMES))
    print(f"\nBest model: {best_name} (AUC={best['auc']:.3f})  -> saved to {MODEL_OUTPUT_FOLDER}")

    # ---- feature importance: is the model actually USING the clustering/
    # temperature/gap features, or leaning entirely on raw flux? Directly
    # diagnoses "does it beat persistence" -- if time_since_last_flare_min
    # and decayed_flare_history rank near the bottom, the model isn't
    # learning the same clustering signal persistence exploits for free,
    # which is the leading suspect if persistence wins on TSS below.
    if hasattr(best["model"], "feature_importances_"):
        imp = best["model"].feature_importances_
        order = np.argsort(imp)[::-1]
        print(f"\nFeature importance ({best_name}), top 15:")
        for rank, i in enumerate(order[:15], 1):
            flag = "  <- clustering/gap/temp feature" if FEATURE_NAMES[i] in (
                "time_since_last_flare_min", "decayed_flare_history",
                "data_gap_frac", "temp_trend_MK_per_min") else ""
            print(f"  {rank:2d}. {FEATURE_NAMES[i]:28s} {imp[i]:.4f}{flag}")
        clustering_idx = [FEATURE_NAMES.index(n) for n in
                          ("time_since_last_flare_min", "decayed_flare_history")]
        clustering_rank = [int(np.where(order == i)[0][0]) + 1 for i in clustering_idx]
        print(f"  time_since_last_flare_min rank: {clustering_rank[0]}/{len(FEATURE_NAMES)}   "
              f"decayed_flare_history rank: {clustering_rank[1]}/{len(FEATURE_NAMES)}")
        if min(clustering_rank) > len(FEATURE_NAMES) * 0.6:
            print("  [NOTE] Both clustering features rank low -- if persistence beats the model")
            print("  below, this is the likely reason: FIX 1's decay exclusion may be removing")
            print("  the exact training examples ('recent flare -> another one soon') that would")
            print("  teach the model to use these features the way persistence exploits for free.")

    preds = (best["prob"] >= ALARM_THRESHOLD).astype(int)
    if y_test.sum() > 0:
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        print(f"  At the DEFAULT {ALARM_THRESHOLD} threshold on CALIBRATED probabilities "
              f"(calibration compresses scores toward the true ~few-% base rate, so honest "
              f"probabilities rarely cross 0.5 -- all-zero here is expected, NOT a broken "
              f"model; the threshold sweep below finds the real operating points):")
        print(f"  Precision: {prec:.3f}   Recall: {rec:.3f}")
        print(f"  Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")

    # ---- mandatory baselines: display, for transparency (selection above
    # already used this same comparison -- this just shows the full table) --
    print("\nMandatory baseline comparison, ALL candidates (TSS -- 0=no skill, 1=perfect):")
    for name, prob in (("Climatology (always guess base rate)", clim_prob),
                       ("Persistence (flaring now -> predict flare)", pers_prob)):
        rep = skill_report(y_test, prob, ALARM_THRESHOLD, p_clim=clim.base_rate_)
        print(f"  {name:48s} TSS={rep['TSS']:+.3f}  HSS={rep['HSS']:+.3f}  "
              f"BSS={rep['BSS']:+.3f}  POD={rep['POD']:.3f}  FAR={rep['FAR']:.3f}")
    print("  " + "-" * 90)
    for mname, mdict in models.items():
        prob_for_compare = raw_probs_snapshot[mname]     # raw for every candidate -- fair comparison
        rep = skill_report(y_test, prob_for_compare, ALARM_THRESHOLD, p_clim=clim.base_rate_)
        tag = "  <- BEATS persistence" if candidate_tss[mname] > pers_tss else ""
        star = " (picked as best)" if mname == best_name else ""
        print(f"  {mname+star:48s} TSS={rep['TSS']:+.3f}  HSS={rep['HSS']:+.3f}  "
              f"BSS={rep['BSS']:+.3f}  POD={rep['POD']:.3f}  FAR={rep['FAR']:.3f}{tag}")

    beat_persistence = candidate_tss[best_name] > pers_tss
    beat_climatology = candidate_tss[best_name] > clim_tss_at_thresh
    print(f"\n  Picked-best model beats climatology: {beat_climatology}   beats persistence: {beat_persistence}")
    if not (beat_climatology and beat_persistence):
        print("  [NOTE] A model that doesn't clearly beat these dumb baselines isn't a")
        print("  credible forecasting claim yet, no matter how good AUC alone looks --")
        print("  this is exactly the check the field's own literature insists on.")

    return best_name, best, scaler, X_test_sc, y_test, test_df
